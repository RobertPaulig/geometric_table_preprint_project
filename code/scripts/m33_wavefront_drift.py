#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, required=True, choices=["occupancy", "values"])
    p.add_argument("--n-start-list", type=str, required=True, help="comma-separated n_start values")
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--n-step", type=int, required=True)
    p.add_argument("--window-rows", type=int, required=True)
    p.add_argument("--k-max", type=int, required=True)
    p.add_argument("--dt", type=int, default=5)
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument("--peaks", type=int, default=3)
    p.add_argument("--conf-min", type=float, default=0.15)
    p.add_argument("--sanity", type=str, default="none", choices=["none", "permute_cols"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, params: Dict[str, object]) -> Dict[str, object]:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != "m33_manifest.json"])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / "m33_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _roll_sum(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    k = int(window)
    kernel = np.ones(k, dtype=float) / float(k)
    return np.convolve(x, kernel, mode="same")


def _normalize_1d(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    xf = xf - float(xf.mean())
    sigma = float(xf.std())
    if not np.isfinite(sigma) or sigma <= 1e-9:
        return xf
    return xf / sigma


def _xcorr_shift_1d(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    aa = _normalize_1d(a)
    bb = _normalize_1d(b)
    fa = np.fft.rfft(aa)
    fb = np.fft.rfft(bb)
    corr = np.fft.irfft(fa * np.conj(fb), n=aa.size)
    corr = np.real(corr)
    peak_idx = int(np.argmax(corr))
    peak = float(corr[peak_idx]) / float(aa.size)

    n = aa.size
    shift = float(peak_idx)
    if shift > n / 2:
        shift -= float(n)

    return shift, peak


def _values_profile(n_top: int, H: int, K: int) -> np.ndarray:
    prof = np.zeros(K, dtype=np.float64)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        for n in range(first, n_end, k):
            q = n // k
            prof[k - 1] += 1.0 / float(q)
    return prof


def _occupancy_profile(n_top: int, H: int, K: int) -> np.ndarray:
    prof = np.zeros(K, dtype=np.float64)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        count = 0
        for _n in range(first, n_end, k):
            count += 1
        prof[k - 1] = float(count)
    return prof


def build_profile(mode: str, n_top: int, H: int, K: int) -> np.ndarray:
    if mode == "values":
        return _values_profile(n_top, H, K)
    if mode == "occupancy":
        return _occupancy_profile(n_top, H, K)
    raise ValueError(mode)


def maybe_permute_cols(profile: np.ndarray, sanity: str, rng: np.random.Generator) -> np.ndarray:
    if sanity == "none":
        return profile
    if sanity == "permute_cols":
        perm = rng.permutation(profile.size)
        return profile[perm]
    raise ValueError(sanity)


def topk_peaks(profile: np.ndarray, *, k: int) -> List[Tuple[int, float]]:
    if k <= 0:
        return []
    x = profile
    idx = np.argsort(x)[-k:][::-1]
    return [(int(i), float(x[int(i)])) for i in idx]

def _subpixel_peak_pos(x: np.ndarray, idx: int) -> float:
    n = x.size
    if idx <= 0 or idx >= n - 1:
        return float(idx)
    c_m1 = float(x[idx - 1])
    c_0 = float(x[idx])
    c_p1 = float(x[idx + 1])
    denom = (c_m1 - 2.0 * c_0 + c_p1)
    if denom == 0.0:
        return float(idx)
    frac = 0.5 * (c_m1 - c_p1) / denom
    if not np.isfinite(frac) or abs(frac) > 1.0:
        return float(idx)
    return float(idx) + float(frac)


@dataclass(frozen=True)
class TrackRow:
    n_start: int
    t: int
    n_top: int
    peak_id: int
    k_peak: int
    k_peak_sub: float
    peak_value: float


@dataclass(frozen=True)
class DxRow:
    n_start: int
    t: int
    dx_profile: float
    corr_peak: float
    dx_peak: Optional[float]
    dx_corr: Optional[float]


def _assign_peaks(prev: Optional[List[int]], candidates: List[Tuple[int, float]], peaks: int) -> List[Tuple[int, float]]:
    if not candidates:
        return [(-1, float("nan")) for _ in range(peaks)]

    if prev is None:
        chosen = candidates[:peaks]
        while len(chosen) < peaks:
            chosen.append((-1, float("nan")))
        return chosen

    used = set()
    ordered: List[Tuple[int, float]] = []
    for pid in range(peaks):
        prev_k = prev[pid]
        best = None
        best_dist = None
        for k_idx, val in candidates:
            if k_idx in used:
                continue
            dist = abs(int(k_idx) - int(prev_k))
            if best is None or dist < best_dist:
                best = (k_idx, val)
                best_dist = dist
        if best is None:
            ordered.append((-1, float("nan")))
        else:
            used.add(best[0])
            ordered.append(best)
    return ordered


def fit_slope_per_n(xs: List[float], ys: List[float], n_step: int) -> float:
    if len(xs) < 2:
        return float("nan")
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    vx = float(np.var(x))
    if vx == 0.0:
        return float("nan")
    slope_per_t = float(np.cov(x, y, bias=True)[0, 1] / vx)
    return slope_per_t / float(n_step)


def summarize_nstart(dx_rows: List[DxRow], tracks: List[TrackRow], *, conf_min: float, n_step: int) -> Dict[str, object]:
    valid = [r for r in dx_rows if (r.corr_peak >= conf_min and np.isfinite(r.dx_profile))]
    valid_frac = float(len(valid) / len(dx_rows)) if dx_rows else float("nan")
    dx_valid = [float(r.dx_profile) for r in valid]

    k0 = [r for r in tracks if r.peak_id == 0 and r.k_peak >= 1]
    xs = [float(r.t) for r in k0]
    ys = [float(r.k_peak_sub) for r in k0]
    slope = fit_slope_per_n(xs, ys, n_step)

    return {
        "n_pairs": len(dx_rows),
        "valid_frac": valid_frac,
        "mean_dx": float(np.mean(dx_valid)) if dx_valid else float("nan"),
        "median_dx": float(np.median(dx_valid)) if dx_valid else float("nan"),
        "std_dx": float(np.std(dx_valid)) if dx_valid else float("nan"),
        "slope_kpeak": slope,
    }


def run_single_nstart(
    *,
    mode: str,
    n_start: int,
    frames: int,
    n_step: int,
    H: int,
    K: int,
    dt: int,
    smooth: int,
    peaks: int,
    sanity: str,
    seed: int,
) -> Tuple[List[DxRow], List[TrackRow]]:
    rng = np.random.default_rng(seed + (n_start % 1000003))
    profiles: List[np.ndarray] = []

    for t in range(frames):
        n_top = n_start + t * n_step
        prof = build_profile(mode, n_top, H, K)
        prof = maybe_permute_cols(prof, sanity, rng)
        prof = _roll_sum(prof, smooth)
        profiles.append(prof)

    dx_rows: List[DxRow] = []
    tracks: List[TrackRow] = []
    prev_peaks: Optional[List[int]] = None
    main_peak_idx: List[int] = []
    main_peak_sub: List[float] = []

    # Pass 1: peak assignment / tracking.
    for t in range(frames):
        n_top = n_start + t * n_step
        cand = topk_peaks(profiles[t], k=max(peaks * 4, peaks))
        assigned = _assign_peaks(prev_peaks, cand, peaks)
        prev_peaks = [k for (k, _v) in assigned]
        main_peak_idx.append(int(prev_peaks[0]) if prev_peaks and prev_peaks[0] >= 0 else -1)
        main_peak_sub.append(_subpixel_peak_pos(profiles[t], main_peak_idx[-1]) + 1.0 if main_peak_idx[-1] >= 0 else float("nan"))
        for pid, (k_idx, val) in enumerate(assigned):
            k_peak = int(k_idx) + 1 if k_idx >= 0 else -1
            k_sub = _subpixel_peak_pos(profiles[t], int(k_idx)) + 1.0 if k_idx >= 0 else float("nan")
            tracks.append(TrackRow(n_start=n_start, t=t, n_top=n_top, peak_id=pid, k_peak=k_peak, k_peak_sub=float(k_sub), peak_value=float(val)))

    # Pass 2: dx estimates.
    for t in range(frames - dt):
        dx_peak: Optional[float] = None
        if main_peak_idx[t] >= 0 and main_peak_idx[t + dt] >= 0:
            dx_peak = float((main_peak_idx[t + dt] + 1) - (main_peak_idx[t] + 1))

        dx_profile = float("nan")
        if np.isfinite(main_peak_sub[t]) and np.isfinite(main_peak_sub[t + dt]):
            dx_profile = float(main_peak_sub[t + dt] - main_peak_sub[t])

        shift, peak = local_profile_shift(profiles[t], profiles[t + dt], center_idx=main_peak_idx[t], window=129, max_shift=64)
        dx_rows.append(DxRow(n_start=n_start, t=t, dx_profile=float(dx_profile), corr_peak=float(peak), dx_peak=dx_peak, dx_corr=float(shift)))

    return dx_rows, tracks


def local_profile_shift(a: np.ndarray, b: np.ndarray, *, center_idx: int, window: int, max_shift: int) -> Tuple[float, float]:
    if center_idx < 0:
        return _xcorr_shift_1d(a, b)

    win = int(window)
    half = win // 2
    ms = int(max_shift)

    # High-pass: remove long-scale trend to emphasize local "front" structure.
    hp = 51
    a_hp = a - _roll_sum(a, hp)
    b_hp = b - _roll_sum(b, hp)

    def pad_extract(x: np.ndarray, left: int, right: int) -> np.ndarray:
        l = max(0, left)
        r = min(x.size, right)
        seg = x[l:r]
        if seg.size == 0:
            return np.zeros((right - left,), dtype=float)
        if l > left:
            seg = np.pad(seg, (l - left, 0), mode="edge")
        if r < right:
            seg = np.pad(seg, (0, right - r), mode="edge")
        return seg.astype(np.float64, copy=False)

    a_seg = pad_extract(a_hp, center_idx - half, center_idx - half + win)
    b_big = pad_extract(b_hp, center_idx - half - ms, center_idx - half + win + ms)

    a0 = a_seg - float(a_seg.mean())
    norm_a = float(np.linalg.norm(a0))
    if not np.isfinite(norm_a) or norm_a <= 1e-12:
        return 0.0, 0.0

    # Dot products for each shift window (b_win · a0).
    dots = np.correlate(b_big.astype(np.float64, copy=False), a0.astype(np.float64, copy=False), mode="valid")
    if dots.size != 2 * ms + 1:
        return _xcorr_shift_1d(a, b)

    # Normalize per-window using demeaned window norm computed from sums/sumsq.
    csum = np.cumsum(np.pad(b_big, (1, 0), mode="constant"))
    csum2 = np.cumsum(np.pad(b_big * b_big, (1, 0), mode="constant"))
    norms = np.empty_like(dots, dtype=np.float64)
    for i in range(dots.size):
        start = i
        end = i + win
        s = float(csum[end] - csum[start])
        s2 = float(csum2[end] - csum2[start])
        mean = s / float(win)
        var = max(0.0, (s2 / float(win)) - mean * mean)
        norms[i] = math.sqrt(var * float(win))

    denom = (norm_a * norms) + 1e-12
    scores = dots / denom
    best_i = int(np.argmax(scores))
    best_score = float(scores[best_i])
    shift = float(best_i - ms)
    if 0 < best_i < (scores.size - 1):
        c_m1 = float(scores[best_i - 1])
        c_0 = float(scores[best_i])
        c_p1 = float(scores[best_i + 1])
        denom_p = (c_m1 - 2.0 * c_0 + c_p1)
        if denom_p != 0.0:
            frac = 0.5 * (c_m1 - c_p1) / denom_p
            if abs(frac) <= 1.0:
                shift += float(frac)
    return shift, best_score


def write_tracks_csv(path: Path, rows: List[TrackRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_start", "t", "n_top", "peak_id", "k_peak", "k_peak_sub", "peak_value"])
        for r in rows:
            w.writerow([r.n_start, r.t, r.n_top, r.peak_id, r.k_peak, f"{r.k_peak_sub:.8g}", f"{r.peak_value:.8g}"])


def write_dx_csv(path: Path, rows: List[DxRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n_start", "t", "dx_profile", "corr_peak", "dx_peak", "dx_corr"])
        for r in rows:
            w.writerow([
                r.n_start,
                r.t,
                f"{r.dx_profile:.8g}",
                f"{r.corr_peak:.8g}",
                "" if r.dx_peak is None else f"{r.dx_peak:.8g}",
                "" if r.dx_corr is None else f"{r.dx_corr:.8g}",
            ])


def write_summary_by_nstart_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["n_start", "mean_dx", "median_dx", "std_dx", "slope_kpeak", "valid_frac", "notes"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def make_plots(out_dir: Path, *, tracks: List[TrackRow], dx_rows: List[DxRow], summary_rows: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    n_starts = sorted({r.n_start for r in tracks})

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for n0 in n_starts:
        pts = [r for r in tracks if r.n_start == n0 and r.peak_id == 0 and r.k_peak >= 1]
        ts = [r.t for r in pts]
        ks = [r.k_peak_sub for r in pts]
        ax.plot(ts, ks, linewidth=0.9, label=f"n_start={n0}")
    ax.set_title("k_peak(t) for main peak (subpixel)")
    ax.set_xlabel("t")
    ax.set_ylabel("k_peak (subpixel)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "m33_kpeak_vs_t.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    for n0 in sorted({r.n_start for r in dx_rows}):
        pts = [r for r in dx_rows if r.n_start == n0]
        ts = [r.t for r in pts]
        dx = [r.dx_profile for r in pts]
        ax.plot(ts, dx, linewidth=0.8, label=f"n_start={n0}")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title("dx_profile(t) from peak drift (subpixel)")
    ax.set_xlabel("t")
    ax.set_ylabel("dx_profile (columns)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "m33_dx_profile_vs_t.png", dpi=160)
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8.5, 4.0))
    xs = [r["n_start"] for r in summary_rows]
    mdx = [r["median_dx"] for r in summary_rows]
    sl = [r["slope_kpeak"] for r in summary_rows]
    ax1.plot(xs, mdx, marker="o", linewidth=1.0, label="median dx_profile (valid)")
    ax1.set_xscale("log")
    ax1.set_xlabel("n_start (log scale)")
    ax1.set_ylabel("median dx_profile (columns)")
    ax1.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax2 = ax1.twinx()
    ax2.plot(xs, sl, marker="s", linewidth=1.0, color="#d62728", label="slope k_peak per n")
    ax2.set_ylabel("slope_kpeak (Δk/Δn)")
    ax1.set_title("drift strength vs n_start")
    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "m33_drift_vs_nstart.png", dpi=160)
    plt.close(fig)


def make_sanity_compare(parent_out: Path, real_summary: List[Dict[str, object]], sanity_summary: List[Dict[str, object]]) -> None:
    import matplotlib.pyplot as plt

    by_n_real = {int(r["n_start"]): r for r in real_summary}
    by_n_san = {int(r["n_start"]): r for r in sanity_summary}
    n_starts = sorted(set(by_n_real.keys()) & set(by_n_san.keys()))

    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    xs = n_starts
    y_real = [by_n_real[n]["mean_dx"] for n in xs]
    y_san = [by_n_san[n]["mean_dx"] for n in xs]
    ax.plot(xs, y_real, marker="o", linewidth=1.0, label="real mean dx")
    ax.plot(xs, y_san, marker="o", linewidth=1.0, label="permute_cols mean dx")
    ax.set_xscale("log")
    ax.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_xlabel("n_start (log)")
    ax.set_ylabel("mean dx_profile")
    ax.set_title("sanity compare: mean dx_profile vs n_start")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(parent_out / "m33_sanity_compare.png", dpi=160)
    plt.close(fig)


def write_table_tex(path: Path, real_rows: List[Dict[str, object]], sanity_rows: Optional[List[Dict[str, object]]]) -> None:
    by_n_real = {int(r["n_start"]): r for r in real_rows}
    by_n_san = {int(r["n_start"]): r for r in sanity_rows} if sanity_rows else {}
    n_starts = sorted(by_n_real.keys())

    lines: List[str] = []
    lines.append("\\begin{tabular}{rrrrrr}\\hline")
    lines.append("n\\_start & valid & mean dx & slope $\\Delta k/\\Delta n$ & sanity mean dx & sanity valid \\\\ \\hline")
    for n0 in n_starts:
        r = by_n_real[n0]
        s = by_n_san.get(n0)
        mean_dx = r.get("mean_dx", float("nan"))
        sl = r.get("slope_kpeak", float("nan"))
        vf = r.get("valid_frac", float("nan"))
        if s is None:
            lines.append(f"{n0} & {vf:.3g} & {mean_dx:.3g} & {sl:.3g} &  &  \\\\")
        else:
            mean_s = s.get("mean_dx", float("nan"))
            vf_s = s.get("valid_frac", float("nan"))
            lines.append(f"{n0} & {vf:.3g} & {mean_dx:.3g} & {sl:.3g} & {mean_s:.3g} & {vf_s:.3g} \\\\")
    lines.append("\\hline\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    n_starts = parse_int_list(args.n_start_list)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    all_dx: List[DxRow] = []
    all_tracks: List[TrackRow] = []
    summary_rows: List[Dict[str, object]] = []

    for n0 in n_starts:
        dx_rows, tracks = run_single_nstart(
            mode=args.mode,
            n_start=n0,
            frames=args.frames,
            n_step=args.n_step,
            H=args.window_rows,
            K=args.k_max,
            dt=args.dt,
            smooth=args.smooth,
            peaks=args.peaks,
            sanity=args.sanity,
            seed=args.seed,
        )
        all_dx.extend(dx_rows)
        all_tracks.extend(tracks)
        summ = summarize_nstart(dx_rows, tracks, conf_min=args.conf_min, n_step=args.n_step)
        summ["n_start"] = n0
        summ["notes"] = ""
        summary_rows.append(summ)

    runtime_s = float(time.time() - started)

    write_tracks_csv(out_dir / "m33_wavefront_tracks.csv", all_tracks)
    write_dx_csv(out_dir / "m33_dx_profile.csv", all_dx)
    write_summary_by_nstart_csv(out_dir / "m33_summary_by_nstart.csv", summary_rows)

    summary_json = {
        "params": {
            "mode": args.mode,
            "n_start_list": n_starts,
            "frames": args.frames,
            "n_step": args.n_step,
            "window_rows": args.window_rows,
            "k_max": args.k_max,
            "dt": args.dt,
            "smooth": args.smooth,
            "peaks": args.peaks,
            "conf_min": args.conf_min,
            "sanity": args.sanity,
            "seed": args.seed,
        },
        "runtime_seconds": runtime_s,
        "summary_by_nstart": summary_rows,
    }
    (out_dir / "m33_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    make_plots(out_dir, tracks=all_tracks, dx_rows=all_dx, summary_rows=summary_rows)

    # If this is the sanity run, update parent's compare/table/manifest.
    if args.sanity == "permute_cols":
        parent = out_dir.parent
        real_summary_csv = parent / "m33_summary_by_nstart.csv"
        if real_summary_csv.exists():
            real_rows: List[Dict[str, object]] = []
            with real_summary_csv.open("r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    real_rows.append({
                        "n_start": int(r["n_start"]),
                        "mean_dx": float(r["mean_dx"]) if r["mean_dx"] else float("nan"),
                        "median_dx": float(r["median_dx"]) if r["median_dx"] else float("nan"),
                        "valid_frac": float(r["valid_frac"]) if r["valid_frac"] else float("nan"),
                        "slope_kpeak": float(r["slope_kpeak"]) if r["slope_kpeak"] else float("nan"),
                    })
            make_sanity_compare(parent, real_rows, summary_rows)
            write_table_tex(parent / "m33_table.tex", real_rows, summary_rows)
            write_manifest(parent, params={
                **(json.loads((parent / "m33_summary.json").read_text(encoding="utf-8"))["params"]),
                "runtime_seconds": float(json.loads((parent / "m33_summary.json").read_text(encoding="utf-8")).get("runtime_seconds", float("nan"))),
            })
    else:
        # For real run, create placeholder table (will be overwritten once sanity run completes).
        write_table_tex(out_dir / "m33_table.tex", summary_rows, None)
        write_manifest(out_dir, params={**summary_json["params"], "runtime_seconds": runtime_s})

    write_manifest(out_dir, params={**summary_json["params"], "runtime_seconds": runtime_s})
    print(f"OK: wrote M33 wavefront drift to {out_dir}")


if __name__ == "__main__":
    main()
