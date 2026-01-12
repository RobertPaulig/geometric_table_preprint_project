from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated int list")
    return out


def parse_q0_grid(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Expected --q0-grid as start,end,step (e.g. 6,40,1)")
    start, end, step = (int(parts[0]), int(parts[1]), int(parts[2]))
    if step == 0:
        raise ValueError("q0 step must be non-zero")
    q0s: List[int] = []
    if step > 0:
        v = start
        while v <= end:
            q0s.append(v)
            v += step
    else:
        v = start
        while v >= end:
            q0s.append(v)
            v += step
    if not q0s:
        raise ValueError("q0 grid produced an empty list")
    return q0s


def parse_float_pair(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected start,end (e.g. 78.0,90.0)")
    a = float(parts[0])
    b = float(parts[1])
    if b <= a:
        raise ValueError("Expected end > start")
    return a, b


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


def write_manifest(out_dir: Path, params: Dict[str, Any], *, name: str) -> None:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != name])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(int(w), dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def r2_linear(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 3:
        return float("nan")
    x0 = x.astype(np.float64)
    y0 = y.astype(np.float64)
    x_mean = float(x0.mean())
    y_mean = float(y0.mean())
    ss_x = float(np.sum((x0 - x_mean) ** 2))
    if ss_x <= 0:
        return float("nan")
    slope = float(np.sum((x0 - x_mean) * (y0 - y_mean)) / ss_x)
    intercept = y_mean - slope * x_mean
    y_hat = intercept + slope * x0
    ss_res = float(np.sum((y0 - y_hat) ** 2))
    ss_tot = float(np.sum((y0 - y_mean) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return 1.0 - ss_res / ss_tot


def k_window_for_frame(n_top: int, H: int, W: int, q0: int) -> Tuple[int, int]:
    n_mid = n_top + (H // 2)
    k_c = int(n_mid // q0)
    k_start = int(k_c - (W // 2))
    if k_start < 1:
        k_start = 1
    k_end = int(k_start + W - 1)
    return k_start, k_end


def build_values_invq_heatmap_and_profile(*, n_top: int, H: int, k_start: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    img = np.zeros((H, W), dtype=np.float32)
    prof = np.zeros(W, dtype=np.float32)
    n_end = n_top + H

    for j in range(W):
        k = k_start + j
        first = ((n_top + k - 1) // k) * k
        if first >= n_end:
            continue
        for n in range(first, n_end, k):
            q = n // k
            w = 1.0 / float(q)
            img[n - n_top, j] = w
            prof[j] += w

    return img, prof


def top_peaks(profile: np.ndarray, peaks: int) -> Tuple[List[int], List[float]]:
    peaks = int(peaks)
    if peaks <= 0 or profile.size == 0:
        return [], []
    k = min(peaks, int(profile.size))
    idx = np.argpartition(-profile, kth=k - 1)[:k]
    idx = idx[np.argsort(-profile[idx])]
    return (idx.tolist(), [float(profile[i]) for i in idx.tolist()])


def theta_grid(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("theta step must be positive")
    n = int(math.floor((end - start) / step + 0.5)) + 1
    out = start + step * np.arange(n, dtype=np.float64)
    out = out[out <= end + 1e-12]
    if out.size == 0:
        raise ValueError("theta grid is empty")
    return out


def precompute_line_indices(
    *,
    H: int,
    W: int,
    thetas_deg: np.ndarray,
    x_offsets: List[int],
    x_center: float,
    y_center: float,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.arange(H, dtype=np.float64)
    n_theta = int(thetas_deg.size)
    n_off = len(x_offsets)

    x_idx = np.full((n_theta, n_off, H), 0, dtype=np.int32)
    mask = np.zeros((n_theta, n_off, H), dtype=np.bool_)

    for i, th in enumerate(thetas_deg.tolist()):
        tanv = math.tan(math.radians(float(th)))
        if abs(tanv) < 1e-9:
            continue
        dx_dy = 1.0 / tanv
        for j, off in enumerate(x_offsets):
            x = x_center + float(off) + (y - y_center) * dx_dy
            xi = np.rint(x).astype(np.int32)
            ok = (xi >= 0) & (xi < W)
            mask[i, j, :] = ok
            xi = np.clip(xi, 0, W - 1)
            x_idx[i, j, :] = xi

    return x_idx, mask


def quadratic_refine_peak(thetas: np.ndarray, score: np.ndarray, argmax_idx: int, *, half_window: int = 3) -> float:
    i0 = int(argmax_idx)
    lo = max(0, i0 - int(half_window))
    hi = min(int(thetas.size) - 1, i0 + int(half_window))
    x = thetas[lo : hi + 1]
    y = score[lo : hi + 1]
    if x.size < 3:
        return float(thetas[i0])
    a, b, _c = np.polyfit(x, y, deg=2)
    if not np.isfinite(a) or a >= 0:
        return float(thetas[i0])
    x_peak = float(-b / (2.0 * a))
    return float(np.clip(x_peak, float(x.min()), float(x.max())))


def peakiness(score_total: np.ndarray) -> Tuple[float, float, float]:
    mx = float(np.max(score_total))
    med = float(np.median(score_total))
    scale = mad(score_total)
    if scale <= 0:
        scale = float(np.std(score_total))
    p = float((mx - med) / (scale + 1e-9))
    return p, mx, med


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def compute_run_metrics(
    *,
    n_start: int,
    q0: int,
    H: int,
    W: int,
    frames: int,
    n_step: int,
    dt: int,
    smooth: int,
    peaks: int,
    conf_min: float,
    thetas_deg: np.ndarray,
    x_idx: np.ndarray,
    mask: np.ndarray,
    topk_offsets: int,
    sanity: str,
    seed: int,
) -> Dict[str, Any]:
    started = time.time()

    perms: Optional[List[np.ndarray]] = None
    if sanity == "permute_cols":
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_start, q0, 41]))
        perms = [rng.permutation(W).astype(np.int32) for _ in range(frames)]

    y_idx = np.arange(H, dtype=np.int32)
    n_theta = int(thetas_deg.size)
    score_total = np.zeros(n_theta, dtype=np.float64)

    x_peaks0: List[int] = []
    z_series: List[float] = []
    k_starts: List[int] = []

    for t in range(frames):
        n_top = n_start + t * n_step
        k_start, _k_end = k_window_for_frame(n_top, H, W, q0)
        k_starts.append(int(k_start))
        img, prof = build_values_invq_heatmap_and_profile(n_top=n_top, H=H, k_start=k_start, W=W)
        if perms is not None:
            perm = perms[t]
            img = img[:, perm]
            prof = prof[perm]

        prof_sm = rolling_mean(prof, smooth)
        idx, vals = top_peaks(prof_sm, peaks)
        x0 = int(idx[0]) if idx else -1
        x_peaks0.append(x0)

        peak_val = float(vals[0]) if vals else float("nan")
        baseline = float(np.median(prof_sm))
        scale = mad(prof_sm)
        if scale <= 0:
            scale = float(np.std(prof_sm))
        z = float((peak_val - baseline) / (scale + 1e-9)) if math.isfinite(peak_val) else float("nan")
        z_series.append(z)

        vals2 = img[y_idx[None, None, :], x_idx]
        vals2 = vals2 * mask
        sums = np.sum(vals2, axis=2)
        if topk_offsets <= 1:
            score_frame = np.max(sums, axis=1)
        else:
            k = min(int(topk_offsets), sums.shape[1])
            topk = np.partition(sums, kth=sums.shape[1] - k, axis=1)[:, -k:]
            score_frame = np.mean(topk, axis=1)
        score_total += score_frame

    dxs: List[float] = []
    confs: List[float] = []
    for t in range(dt, frames):
        x_now = x_peaks0[t]
        x_prev = x_peaks0[t - dt]
        z_now = z_series[t]
        z_prev = z_series[t - dt]
        if x_now >= 0 and x_prev >= 0 and math.isfinite(z_now) and math.isfinite(z_prev):
            k_now = k_starts[t] + int(x_now)
            k_prev = k_starts[t - dt] + int(x_prev)
            dxs.append(float(k_now - k_prev))
            confs.append(float(min(z_now, z_prev)))
        else:
            dxs.append(float("nan"))
            confs.append(float("nan"))

    dx_arr = np.array(dxs, dtype=np.float64)
    conf_arr = np.array(confs, dtype=np.float64)
    valid_mask = np.isfinite(dx_arr) & np.isfinite(conf_arr) & (conf_arr >= float(conf_min))
    valid_frac = float(valid_mask.mean()) if valid_mask.size else 0.0
    mean_dx = float(np.mean(dx_arr[valid_mask])) if valid_mask.any() else float("nan")
    median_dx = float(np.median(dx_arr[valid_mask])) if valid_mask.any() else float("nan")
    q_eff = float(dt) / mean_dx if (math.isfinite(mean_dx) and mean_dx > 0) else float("nan")

    t_idx = np.arange(dt, frames, dtype=np.float64)
    k_arr = np.array(
        [(k_starts[t] + int(x_peaks0[t])) if x_peaks0[t] >= 0 else float("nan") for t in range(dt, frames)],
        dtype=np.float64,
    )
    if valid_mask.any():
        t_fit = t_idx[valid_mask]
        k_fit = k_arr[valid_mask]
    else:
        t_fit = np.array([], dtype=np.float64)
        k_fit = np.array([], dtype=np.float64)
    slope_kpeak = float(np.polyfit(t_fit, k_fit, deg=1)[0]) if t_fit.size >= 3 else float("nan")
    track_r2 = r2_linear(t_fit, k_fit) if t_fit.size >= 3 else float("nan")

    argmax = int(np.argmax(score_total))
    theta_star = float(thetas_deg[argmax])
    theta_sub = quadratic_refine_peak(thetas_deg, score_total, argmax, half_window=3)
    pk, score_peak, score_med = peakiness(score_total)

    runtime_s = float(time.time() - started)
    return {
        "n_start": int(n_start),
        "q0": int(q0),
        "dt": int(dt),
        "sanity": sanity,
        "runtime_s": runtime_s,
        "mean_dx": float(mean_dx),
        "median_dx": float(median_dx),
        "q_eff": float(q_eff),
        "slope_kpeak": float(slope_kpeak),
        "track_r2": float(track_r2),
        "valid_frac": float(valid_frac),
        "theta_star": float(theta_star),
        "theta_subdeg": float(theta_sub),
        "angle_score_peak": float(score_peak),
        "angle_score_median": float(score_med),
        "peakiness": float(pk),
    }


def angle_scores_per_frame(
    *,
    n_start: int,
    q0: int,
    H: int,
    W: int,
    frames: int,
    n_step: int,
    smooth: int,
    thetas_deg: np.ndarray,
    x_idx: np.ndarray,
    mask: np.ndarray,
    topk_offsets: int,
    sanity: str,
    seed: int,
) -> np.ndarray:
    perms: Optional[List[np.ndarray]] = None
    if sanity == "permute_cols":
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_start, q0, 41]))
        perms = [rng.permutation(W).astype(np.int32) for _ in range(frames)]

    y_idx = np.arange(H, dtype=np.int32)
    score_pf = np.zeros((frames, int(thetas_deg.size)), dtype=np.float64)
    for t in range(frames):
        n_top = n_start + t * n_step
        k_start, _k_end = k_window_for_frame(n_top, H, W, q0)
        img, prof = build_values_invq_heatmap_and_profile(n_top=n_top, H=H, k_start=k_start, W=W)
        if perms is not None:
            perm = perms[t]
            img = img[:, perm]
            prof = prof[perm]
        _ = rolling_mean(prof, smooth)

        vals2 = img[y_idx[None, None, :], x_idx]
        vals2 = vals2 * mask
        sums = np.sum(vals2, axis=2)
        if topk_offsets <= 1:
            score_frame = np.max(sums, axis=1)
        else:
            k = min(int(topk_offsets), sums.shape[1])
            topk = np.partition(sums, kth=sums.shape[1] - k, axis=1)[:, -k:]
            score_frame = np.mean(topk, axis=1)
        score_pf[t, :] = score_frame
    return score_pf


def bootstrap_theta_subdeg_ci(
    *,
    thetas_deg: np.ndarray,
    score_pf: np.ndarray,
    bootstrap: int,
    seed: int,
) -> Tuple[float, float, float]:
    total = np.sum(score_pf, axis=0)
    argmax = int(np.argmax(total))
    theta_sub = quadratic_refine_peak(thetas_deg, total, argmax, half_window=3)

    rng = np.random.default_rng(np.random.SeedSequence([seed, 42, 12345]))
    n_frames = int(score_pf.shape[0])
    boot = np.zeros(int(bootstrap), dtype=np.float64)
    for b in range(int(bootstrap)):
        idx = rng.integers(0, n_frames, size=n_frames, dtype=np.int32)
        tot = np.sum(score_pf[idx, :], axis=0)
        i = int(np.argmax(tot))
        boot[b] = quadratic_refine_peak(thetas_deg, tot, i, half_window=3)

    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return float(theta_sub), lo, hi


def build_table_tex(out_path: Path, best_rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$n_{\mathrm{start}}$ & $q_0$ & $dt$ & $\theta_{\mathrm{sub}}$ & CI & peakiness & $R^2$ \\")
    lines.append(r"\hline")
    for r in best_rows:
        n0 = int(r["n_start"])
        q0 = int(r["q0_best"])
        dt = int(r["dt"])
        th = float(r["theta_subdeg"])
        lo = float(r["theta_ci_lo"])
        hi = float(r["theta_ci_hi"])
        pk = float(r["peakiness"])
        r2 = float(r["track_r2"])
        lines.append(f"{n0} & {q0} & {dt} & {th:.3f} & [{lo:.3f},{hi:.3f}] & {pk:.3g} & {r2:.3f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_heatmap(
    *,
    out_path: Path,
    n_starts: List[int],
    q0s: List[int],
    values: np.ndarray,
    title: str,
    cmap: str = "viridis",
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=150)
    im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)
    ax.set_xlabel("q0")
    ax.set_ylabel("n_start")
    ax.set_xticks(np.arange(len(q0s)))
    ax.set_xticklabels([str(q) for q in q0s], fontsize=7, rotation=90)
    ax.set_yticks(np.arange(len(n_starts)))
    ax.set_yticklabels([f"{n}" for n in n_starts], fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_best_q0(
    *,
    out_path: Path,
    best_rows: List[Dict[str, Any]],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    n = np.array([int(r["n_start"]) for r in best_rows], dtype=np.int64)
    q = np.array([int(r["q0_best"]) for r in best_rows], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=150)
    ax.plot(n, q, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("n_start")
    ax.set_ylabel("best q0")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_curve(
    *,
    out_path: Path,
    thetas: np.ndarray,
    score_total: np.ndarray,
    theta_star: float,
    theta_subdeg: float,
    theta_ci: Tuple[float, float],
    theta_dx: Optional[float],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=150)
    ax.plot(thetas, score_total, linewidth=1.6)
    ax.axvline(theta_star, color="#00d4ff", linewidth=1.2, linestyle="--", label=f"theta*={theta_star:.3f}°")
    ax.axvline(theta_subdeg, color="#00ff6a", linewidth=1.2, linestyle="-", label=f"theta_sub={theta_subdeg:.3f}°")
    lo, hi = theta_ci
    if math.isfinite(lo) and math.isfinite(hi):
        ax.axvspan(lo, hi, color="#00ff6a", alpha=0.12, label="subdeg CI")
    if theta_dx is not None and math.isfinite(theta_dx):
        ax.axvline(theta_dx, color="#ffd200", linewidth=1.1, linestyle=":", label=f"theta_dx={theta_dx:.3f}°")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("score(theta)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_compare_real_sanity(
    *,
    out_path: Path,
    real_csv: Path,
    sanity_csv: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    def load(p: Path) -> Tuple[np.ndarray, np.ndarray]:
        rows = list(csv.DictReader(p.open("r", encoding="utf-8")))
        th = np.array([float(r["theta_deg"]) for r in rows], dtype=np.float64)
        sc = np.array([float(r["score_total"]) for r in rows], dtype=np.float64)
        return th, sc

    th_r, sc_r = load(real_csv)
    th_s, sc_s = load(sanity_csv)
    if th_r.size != th_s.size or float(np.max(np.abs(th_r - th_s))) > 1e-9:
        raise ValueError("real and sanity theta grids do not match")

    def norm(x: np.ndarray) -> np.ndarray:
        mn = float(np.min(x))
        mx = float(np.max(x))
        if not math.isfinite(mn) or not math.isfinite(mx) or mx <= mn:
            return np.zeros_like(x)
        return (x - mn) / (mx - mn)

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=150)
    ax.plot(th_r, norm(sc_r), linewidth=1.6, label="real (normalized)")
    ax.plot(th_s, norm(sc_s), linewidth=1.3, alpha=0.85, label="sanity permute_cols (normalized)")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("score(theta) normalized")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-start-list", type=str, required=True)
    p.add_argument("--q0-grid", type=str, required=True)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--mode", type=str, default="values", choices=["values"])
    p.add_argument("--weights", type=str, default="invq", choices=["invq"])
    p.add_argument("--theta-range-deg", type=str, default="78.0,90.0")
    p.add_argument("--theta-step-deg", type=float, default=0.1)
    p.add_argument("--theta-refine-halfwidth-deg", type=float, default=0.6)
    p.add_argument("--theta-refine-step-deg", type=float, default=0.02)
    p.add_argument("--target-dx", type=float, default=0.8)
    p.add_argument("--dt-min", type=int, default=5)
    p.add_argument("--dt-max", type=int, default=80)
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument("--peaks", type=int, default=3)
    p.add_argument("--conf-min", type=float, default=0.15)
    p.add_argument("--r2-guard", type=float, default=0.95)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--sanity", type=str, default="none", choices=["none", "permute_cols"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    n_starts = parse_int_list(args.n_start_list)
    q0s = parse_q0_grid(args.q0_grid)
    max_n_start = int(max(n_starts))

    H = int(args.H)
    W = int(args.W)
    frames = int(args.frames)
    n_step = int(args.n_step)
    smooth = int(args.smooth)
    peaks = int(args.peaks)
    conf_min = float(args.conf_min)
    r2_guard = float(args.r2_guard)
    bootstrap = int(args.bootstrap)
    sanity = str(args.sanity)
    seed = int(args.seed)

    theta_a, theta_b = parse_float_pair(args.theta_range_deg)
    thetas = theta_grid(theta_a, theta_b, float(args.theta_step_deg))
    x_center = (W - 1) / 2.0
    y_center = (H - 1) / 2.0
    x_offsets = [-64, -32, -16, 0, 16, 32, 64]
    x_idx, mask = precompute_line_indices(H=H, W=W, thetas_deg=thetas, x_offsets=x_offsets, x_center=x_center, y_center=y_center)

    started = time.time()

    sweep_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    peakiness_grid = np.full((len(n_starts), len(q0s)), np.nan, dtype=np.float64)

    dt_min = int(args.dt_min)
    dt_max = int(args.dt_max)
    target_dx = float(args.target_dx)

    # If this is a sanity run in a nested directory, try to align the "best score(theta)" curve
    # with the real run (same q0 and same theta grid) so the compare plot is unambiguous.
    parent_real_best_q0: Optional[int] = None
    parent_real_curve_thetas: Optional[np.ndarray] = None
    if sanity == "permute_cols" and out_root.name == "sanity_permute_cols":
        parent = out_root.parent
        best_csv = parent / "m41_best_by_nstart.csv"
        if best_csv.exists():
            with best_csv.open("r", encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    try:
                        if int(float(r.get("n_start", "nan"))) == max_n_start:
                            parent_real_best_q0 = int(float(r.get("q0_best", "nan")))
                            break
                    except Exception:
                        continue
        curve_csv = parent / "m41_best_score_theta.csv"
        if curve_csv.exists():
            rows = list(csv.DictReader(curve_csv.open("r", encoding="utf-8", newline="")))
            if rows:
                parent_real_curve_thetas = np.array([float(r["theta_deg"]) for r in rows], dtype=np.float64)

    for i_n, n_start in enumerate(n_starts):
        per_q_rows: List[Dict[str, Any]] = []
        for i_q, q0 in enumerate(q0s):
            dt_auto = int(round(target_dx * float(q0)))
            dt_auto = max(dt_min, min(dt_max, dt_auto))
            dt_auto = min(dt_auto, frames - 1)
            m = compute_run_metrics(
                n_start=int(n_start),
                q0=int(q0),
                H=H,
                W=W,
                frames=frames,
                n_step=n_step,
                dt=dt_auto,
                smooth=smooth,
                peaks=peaks,
                conf_min=conf_min,
                thetas_deg=thetas,
                x_idx=x_idx,
                mask=mask,
                topk_offsets=3,
                sanity=sanity,
                seed=seed,
            )
            m["dt_auto"] = dt_auto
            sweep_rows.append(m)
            per_q_rows.append(m)
            peakiness_grid[i_n, i_q] = float(m["peakiness"])

        best_idx: Optional[int] = None
        best_score = -float("inf")
        fallback_best_score = -float("inf")
        fallback_best_idx: Optional[int] = None
        for i_q, row in enumerate(per_q_rows):
            r2v = float(row["track_r2"])
            valid_frac = float(row["valid_frac"])
            pk = float(row["peakiness"])
            score = pk * math.sqrt(max(r2v if math.isfinite(r2v) else 0.0, 0.0)) * math.sqrt(max(valid_frac, 0.0))
            if score > fallback_best_score:
                fallback_best_score = score
                fallback_best_idx = i_q
            if math.isfinite(r2v) and r2v >= r2_guard and score > best_score:
                best_score = score
                best_idx = i_q

        used_guard = True
        if best_idx is None:
            best_idx = fallback_best_idx
            best_score = fallback_best_score
            used_guard = False
        assert best_idx is not None
        best = per_q_rows[int(best_idx)]
        q0_best = int(best["q0"])
        dt_best = int(best["dt"])

        th0 = float(best["theta_subdeg"])
        halfw = float(args.theta_refine_halfwidth_deg)
        th_start = max(theta_a, th0 - halfw)
        th_end = min(theta_b, th0 + halfw)
        thetas_fine = theta_grid(th_start, th_end, float(args.theta_refine_step_deg))
        x_idx_f, mask_f = precompute_line_indices(
            H=H,
            W=W,
            thetas_deg=thetas_fine,
            x_offsets=x_offsets,
            x_center=x_center,
            y_center=y_center,
        )
        score_pf = angle_scores_per_frame(
            n_start=int(n_start),
            q0=q0_best,
            H=H,
            W=W,
            frames=frames,
            n_step=n_step,
            smooth=smooth,
            thetas_deg=thetas_fine,
            x_idx=x_idx_f,
            mask=mask_f,
            topk_offsets=3,
            sanity=sanity,
            seed=seed,
        )
        theta_sub, theta_lo, theta_hi = bootstrap_theta_subdeg_ci(
            thetas_deg=thetas_fine,
            score_pf=score_pf,
            bootstrap=bootstrap,
            seed=seed + int(n_start) + int(q0_best),
        )
        score_total_fine = np.sum(score_pf, axis=0)
        argmax_fine = int(np.argmax(score_total_fine))
        theta_star_fine = float(thetas_fine[argmax_fine])

        q_eff_dx = float(best["q_eff"])
        theta_dx = float(math.degrees(math.atan(q_eff_dx))) if (math.isfinite(q_eff_dx) and q_eff_dx > 0) else float("nan")

        if int(n_start) == max_n_start:
            # real: plot best-of as usual; sanity: align to real best if available
            q0_curve = q0_best
            dt_curve = dt_best
            thetas_curve = thetas_fine
            score_total_curve = score_total_fine
            theta_star_curve = theta_star_fine
            theta_sub_curve = theta_sub
            theta_lo_curve = theta_lo
            theta_hi_curve = theta_hi
            theta_dx_curve = theta_dx

            if sanity == "permute_cols" and parent_real_best_q0 is not None and parent_real_curve_thetas is not None:
                q0_curve = int(parent_real_best_q0)
                # reuse the exact theta grid from the real run
                thetas_curve = parent_real_curve_thetas
                x_idx_c, mask_c = precompute_line_indices(
                    H=H,
                    W=W,
                    thetas_deg=thetas_curve,
                    x_offsets=x_offsets,
                    x_center=x_center,
                    y_center=y_center,
                )
                score_pf_c = angle_scores_per_frame(
                    n_start=int(n_start),
                    q0=q0_curve,
                    H=H,
                    W=W,
                    frames=frames,
                    n_step=n_step,
                    smooth=smooth,
                    thetas_deg=thetas_curve,
                    x_idx=x_idx_c,
                    mask=mask_c,
                    topk_offsets=3,
                    sanity=sanity,
                    seed=seed,
                )
                theta_sub_curve, theta_lo_curve, theta_hi_curve = bootstrap_theta_subdeg_ci(
                    thetas_deg=thetas_curve,
                    score_pf=score_pf_c,
                    bootstrap=bootstrap,
                    seed=seed + int(n_start) + int(q0_curve) + 999,
                )
                score_total_curve = np.sum(score_pf_c, axis=0)
                argmax_c = int(np.argmax(score_total_curve))
                theta_star_curve = float(thetas_curve[argmax_c])
                # pull dt/q_eff from the per-q sweep (same run)
                row_c = next((r for r in per_q_rows if int(r["q0"]) == q0_curve), None)
                if row_c is not None:
                    dt_curve = int(row_c["dt"])
                    q_eff_dx_c = float(row_c["q_eff"])
                    theta_dx_curve = (
                        float(math.degrees(math.atan(q_eff_dx_c))) if (math.isfinite(q_eff_dx_c) and q_eff_dx_c > 0) else float("nan")
                    )

            score_csv = out_root / "m41_best_score_theta.csv"
            with score_csv.open("w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["theta_deg", "score_total"])
                w.writeheader()
                for th, sc in zip(thetas_curve.tolist(), score_total_curve.tolist()):
                    w.writerow({"theta_deg": float(th), "score_total": float(sc)})
            plot_score_curve(
                out_path=out_root / "m41_best_score_theta.png",
                thetas=thetas_curve,
                score_total=score_total_curve,
                theta_star=theta_star_curve,
                theta_subdeg=theta_sub_curve,
                theta_ci=(theta_lo_curve, theta_hi_curve),
                theta_dx=theta_dx_curve,
                title=f"M41 score(theta) near peak ({sanity})  n_start={int(n_start)}  q0={q0_curve}  dt={dt_curve}",
            )

        hit_boundary = q0_best == min(q0s) or q0_best == max(q0s)
        best_rows.append(
            {
                "n_start": int(n_start),
                "q0_best": q0_best,
                "dt": dt_best,
                "peakiness": float(best["peakiness"]),
                "peakiness_score": float(best_score),
                "theta_star": float(best["theta_star"]),
                "theta_subdeg": float(theta_sub),
                "theta_ci_lo": float(theta_lo),
                "theta_ci_hi": float(theta_hi),
                "theta_dx": float(theta_dx),
                "q_eff": float(best["q_eff"]),
                "mean_dx": float(best["mean_dx"]),
                "track_r2": float(best["track_r2"]),
                "valid_frac": float(best["valid_frac"]),
                "used_r2_guard": bool(used_guard),
                "hit_boundary": bool(hit_boundary),
            }
        )

    write_csv(
        out_root / "m41_q0_sweep.csv",
        sweep_rows,
        [
            "n_start",
            "q0",
            "dt",
            "mean_dx",
            "median_dx",
            "q_eff",
            "slope_kpeak",
            "track_r2",
            "valid_frac",
            "theta_star",
            "theta_subdeg",
            "angle_score_peak",
            "angle_score_median",
            "peakiness",
            "sanity",
        ],
    )
    write_csv(
        out_root / "m41_best_by_nstart.csv",
        best_rows,
        [
            "n_start",
            "q0_best",
            "dt",
            "peakiness",
            "peakiness_score",
            "theta_star",
            "theta_subdeg",
            "theta_ci_lo",
            "theta_ci_hi",
            "theta_dx",
            "q_eff",
            "mean_dx",
            "track_r2",
            "valid_frac",
            "used_r2_guard",
            "hit_boundary",
        ],
    )

    plot_best_q0(out_path=out_root / "m41_best_q0_vs_nstart.png", best_rows=best_rows, title=f"M41 best q0 by peakiness score ({sanity})")
    plot_heatmap(
        out_path=out_root / "m41_peakiness_heatmap.png",
        n_starts=n_starts,
        q0s=q0s,
        values=peakiness_grid,
        title=f"M41 peakiness across (n_start,q0) ({sanity})",
        cmap="viridis",
    )
    build_table_tex(out_root / "m41_table.tex", best_rows)

    summary = {
        "params": {
            "n_start_list": args.n_start_list,
            "q0_grid": args.q0_grid,
            "H": H,
            "W": W,
            "frames": frames,
            "n_step": n_step,
            "mode": args.mode,
            "weights": args.weights,
            "theta_range_deg": args.theta_range_deg,
            "theta_step_deg": float(args.theta_step_deg),
            "theta_refine_halfwidth_deg": float(args.theta_refine_halfwidth_deg),
            "theta_refine_step_deg": float(args.theta_refine_step_deg),
            "target_dx": float(target_dx),
            "dt_min": dt_min,
            "dt_max": dt_max,
            "smooth": smooth,
            "peaks": peaks,
            "conf_min": conf_min,
            "r2_guard": r2_guard,
            "bootstrap": bootstrap,
            "sanity": sanity,
            "seed": seed,
            "out_dir": str(out_root).replace("\\", "/"),
        },
        "git_sha": git_sha(),
        "runtime_s": float(time.time() - started),
        "best_by_nstart": best_rows,
    }
    (out_root / "m41_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_manifest(out_root, {"sanity": sanity, "note": "run manifest"}, name="m41_manifest.json")

    if sanity == "permute_cols" and out_root.name == "sanity_permute_cols":
        parent = out_root.parent
        real_csv = parent / "m41_best_score_theta.csv"
        sanity_csv = out_root / "m41_best_score_theta.csv"
        if real_csv.exists() and sanity_csv.exists():
            plot_compare_real_sanity(
                out_path=parent / "m41_sanity_compare.png",
                real_csv=real_csv,
                sanity_csv=sanity_csv,
                title="M41 sanity compare: score(theta) real vs permute_cols (normalized)",
            )
        write_manifest(
            parent,
            {"note": "parent includes real + sanity outputs", "sanity_dir": str(out_root.relative_to(parent)).replace("\\", "/")},
            name="m41_manifest.json",
        )

    print(f"OK: wrote M41 outputs to {out_root}")


if __name__ == "__main__":
    main()
