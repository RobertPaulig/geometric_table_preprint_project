from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_str_list(s: str) -> List[str]:
    out: List[str] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(part)
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


def write_manifest(out_dir: Path, params: Dict[str, Any]) -> None:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != "m35_manifest.json"])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / "m35_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = int(w)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same")


def mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def phase_corr_shift_1d(a: np.ndarray, b: np.ndarray) -> Tuple[int, float]:
    a0 = a.astype(np.float32)
    b0 = b.astype(np.float32)
    a0 = a0 - float(a0.mean())
    b0 = b0 - float(b0.mean())
    a_std = float(a0.std())
    b_std = float(b0.std())
    if a_std <= 0 or b_std <= 0:
        return 0, 0.0
    a0 = a0 / a_std
    b0 = b0 / b_std

    fa = np.fft.rfft(a0)
    fb = np.fft.rfft(b0)
    r = fa * np.conj(fb)
    denom = np.abs(r)
    denom[denom == 0] = 1.0
    r /= denom
    corr = np.fft.irfft(r, n=a0.shape[0])
    idx = int(np.argmax(corr))
    peak = float(corr[idx])

    n = a0.shape[0]
    shift = idx
    if shift > n // 2:
        shift -= n
    return shift, peak


def values_profile_invq(n_top: int, H: int, K: int) -> np.ndarray:
    prof = np.zeros(K, dtype=np.float32)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        acc = 0.0
        for n in range(first, n_end, k):
            acc += float(k) / float(n)  # 1/q = k/n
        prof[k - 1] = acc
    return prof


def values_profile_log1pq(n_top: int, H: int, K: int) -> np.ndarray:
    prof = np.zeros(K, dtype=np.float32)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        acc = 0.0
        for n in range(first, n_end, k):
            q = n // k
            acc += math.log1p(float(q))
        prof[k - 1] = acc
    return prof


def occupancy_profile_ones(n_top: int, H: int, K: int) -> np.ndarray:
    prof = np.zeros(K, dtype=np.float32)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        count = 0
        for _n in range(first, n_end, k):
            count += 1
        prof[k - 1] = float(count)
    return prof


def build_profile(n_top: int, H: int, K: int, weights: str) -> np.ndarray:
    if weights == "invq":
        return values_profile_invq(n_top, H, K)
    if weights == "log1pq":
        return values_profile_log1pq(n_top, H, K)
    if weights == "ones":
        return occupancy_profile_ones(n_top, H, K)
    raise ValueError(f"Unknown weights: {weights}")


def best_peaks(profile_sm: np.ndarray, peaks: int) -> Tuple[List[int], List[float]]:
    peaks = int(peaks)
    if peaks <= 0:
        return [], []
    idx = np.argpartition(-profile_sm, kth=min(peaks, profile_sm.size - 1) - 1)[:peaks]
    idx = idx[np.argsort(-profile_sm[idx])]
    ks = [int(i + 1) for i in idx.tolist()]
    vals = [float(profile_sm[i]) for i in idx.tolist()]
    return ks, vals


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


@dataclass(frozen=True)
class RunConfig:
    n_start: int
    K: int
    H: int
    weights: str
    frames: int
    n_step: int
    dt: int
    smooth: int
    peaks: int
    conf_min: float
    sanity: str  # none|permute_cols


@dataclass
class RunResult:
    run_id: str
    n_start: int
    K: int
    H: int
    weights: str
    smooth: int
    dt: int
    frames: int
    conf_min: float
    sanity: str
    runtime_s: float
    mean_dx: float
    median_dx: float
    q_eff: float
    slope_kpeak: float
    track_r2: float
    valid_frac: float
    wave_strength: float
    notes: str


def run_id(cfg: RunConfig) -> str:
    n = cfg.n_start
    n_tag = f"n{n//1_000_000}e6" if n % 1_000_000 == 0 else f"n{n}"
    return f"{n_tag}_K{cfg.K}_H{cfg.H}_w{cfg.weights}_dt{cfg.dt}_sm{cfg.smooth}_{cfg.sanity}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_run(cfg: RunConfig, out_dir: Path, seed: int) -> Tuple[RunResult, Dict[int, Dict[int, int]]]:
    started = time.time()
    rng = np.random.default_rng(seed + 17 * cfg.n_start + 131 * cfg.K)
    perm: Optional[np.ndarray] = None
    if cfg.sanity == "permute_cols":
        perm = rng.permutation(cfg.K)

    dxs: List[int] = []
    confs: List[float] = []
    k_peak_series: List[int] = []
    peak_val_series: List[float] = []
    baseline_series: List[float] = []
    mad_series: List[float] = []

    profiles: List[np.ndarray] = []
    profiles_sm: List[np.ndarray] = []

    for t in range(cfg.frames):
        n_top = cfg.n_start + t * cfg.n_step
        prof = build_profile(n_top, cfg.H, cfg.K, cfg.weights)
        if perm is not None:
            prof = prof[perm]

        prof_sm = rolling_mean(prof, cfg.smooth)
        profiles.append(prof)
        profiles_sm.append(prof_sm)

        ks, vals = best_peaks(prof_sm, cfg.peaks)
        k0 = ks[0] if ks else 0
        v0 = vals[0] if vals else float("nan")
        k_peak_series.append(k0)
        peak_val_series.append(v0)
        baseline = float(np.median(prof_sm))
        baseline_series.append(baseline)
        mad_series.append(mad(prof_sm))

        if t >= cfg.dt:
            shift, peak = phase_corr_shift_1d(profiles_sm[t - cfg.dt], prof_sm)
            dxs.append(shift)
            confs.append(peak)

    dx_arr = np.array(dxs, dtype=np.float64)
    conf_arr = np.array(confs, dtype=np.float64)
    valid_mask = conf_arr >= float(cfg.conf_min)
    valid_frac = float(valid_mask.mean()) if valid_mask.size else 0.0

    mean_dx = float(np.mean(dx_arr[valid_mask])) if valid_mask.any() else float("nan")
    median_dx = float(np.median(dx_arr[valid_mask])) if valid_mask.any() else float("nan")
    q_eff = float(cfg.dt) / mean_dx if (not math.isnan(mean_dx) and mean_dx > 0) else float("nan")

    # k_peak slope + r2 (use only frames where we have dx confidence info, i.e. t>=dt)
    t_idx = np.arange(cfg.dt, cfg.frames, dtype=np.float64)
    k_arr = np.array(k_peak_series[cfg.dt :], dtype=np.float64)
    if valid_mask.any():
        t_fit = t_idx[valid_mask]
        k_fit = k_arr[valid_mask]
    else:
        t_fit = np.array([], dtype=np.float64)
        k_fit = np.array([], dtype=np.float64)

    slope_kpeak = float(np.polyfit(t_fit, k_fit, deg=1)[0]) if t_fit.size >= 3 else float("nan")
    track_r2 = r2_linear(t_fit, k_fit) if t_fit.size >= 3 else float("nan")

    prominence = np.array(peak_val_series, dtype=np.float64) - np.array(baseline_series, dtype=np.float64)
    denom = np.array(mad_series, dtype=np.float64) + 1e-9
    wave_strength = float(np.median(prominence / denom)) if prominence.size else float("nan")

    runtime_s = float(time.time() - started)

    tracks: Dict[int, Dict[int, int]] = {}
    for t in range(cfg.frames):
        ks, _ = best_peaks(profiles_sm[t], cfg.peaks)
        tracks[t] = {pid: ks[pid] for pid in range(min(cfg.peaks, len(ks)))}

    res = RunResult(
        run_id=run_id(cfg),
        n_start=cfg.n_start,
        K=cfg.K,
        H=cfg.H,
        weights=cfg.weights,
        smooth=cfg.smooth,
        dt=cfg.dt,
        frames=cfg.frames,
        conf_min=float(cfg.conf_min),
        sanity=cfg.sanity,
        runtime_s=runtime_s,
        mean_dx=mean_dx,
        median_dx=median_dx,
        q_eff=q_eff,
        slope_kpeak=slope_kpeak,
        track_r2=track_r2,
        valid_frac=valid_frac,
        wave_strength=wave_strength,
        notes="",
    )

    summary_path = out_dir / "run_summary.json"
    summary_path.write_text(json.dumps({"config": asdict(cfg), "result": asdict(res)}, indent=2), encoding="utf-8")
    return res, tracks


def render_tail_overlay_mp4(
    *,
    out_dir: Path,
    cfg: RunConfig,
    tracks: Dict[int, Dict[int, int]],
    seed: int,
    sanity: str,
    fps: int,
    format_: str,
) -> Path:
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    ensure_dir(frames_dir)

    rng = np.random.default_rng(seed + 19 * cfg.n_start + 7 * cfg.K)
    perm: Optional[np.ndarray] = None
    if sanity == "permute_cols":
        perm = rng.permutation(cfg.K)

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=110)
    ax.set_xlabel("k (columns)")
    ax.set_ylabel("rows (n increasing down)")
    colors = ["#00d4ff", "#00ff6a", "#ffd200"]
    lines = [ax.plot([], [], color=colors[i], linewidth=1.6, alpha=0.95, label=f"peak{i+1}")[0] for i in range(min(cfg.peaks, 3))]

    vmax = 0.02 if cfg.weights != "ones" else None
    img0 = build_profile(cfg.n_start, cfg.H, cfg.K, cfg.weights)  # 1D; make cheap 2D for background
    bg0 = np.tile(img0.reshape(1, -1), (cfg.H, 1))
    if perm is not None:
        bg0 = bg0[:, perm]
    im = ax.imshow(bg0, aspect="auto", origin="upper", cmap="magma", vmin=0.0, vmax=vmax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    for t in range(min(cfg.frames, 300)):
        n_top = cfg.n_start + t * cfg.n_step
        prof = build_profile(n_top, cfg.H, cfg.K, cfg.weights)
        if perm is not None:
            prof = prof[perm]
        bg = np.tile(prof.reshape(1, -1), (cfg.H, 1))
        im.set_data(bg)

        ax.set_title(f"M35 best overlay ({sanity})  n_start={cfg.n_start}  t={t}  K={cfg.K}  w={cfg.weights}")

        for pid, ln in enumerate(lines):
            xs: List[float] = []
            ys: List[float] = []
            for y in range(cfg.H):
                tt = t - y
                if tt < 0:
                    break
                k_hist = tracks.get(tt, {}).get(pid)
                if k_hist is None or k_hist <= 0:
                    continue
                xs.append(float(k_hist - 1))
                ys.append(float(y))
            if len(xs) < 2:
                ln.set_visible(False)
            else:
                ln.set_data(xs, ys)
                ln.set_visible(True)

        frame_path = frames_dir / f"frame_{t:04d}.png"
        fig.savefig(frame_path)
        if t in (0, 150):
            shutil.copyfile(frame_path, out_dir / f"keyframe_t{t:03d}.png")

    plt.close(fig)

    if format_ == "mp4":
        out_video = out_dir / "m35_best_overlay_tail.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(out_video),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        out_video = out_dir / "m35_best_overlay_tail.gif"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "scale=512:-1:flags=lanczos",
            str(out_video),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for p in frames_dir.glob("frame_*.png"):
        p.unlink()
    frames_dir.rmdir()

    return out_video


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    import csv

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def plot_summary(out_dir: Path, rows: List[RunResult], n_starts: List[int], K_list: List[int], weights_list: List[str]) -> None:
    import matplotlib.pyplot as plt

    def filt(sanity: str = "none") -> List[RunResult]:
        return [r for r in rows if r.sanity == sanity]

    rows_none = filt("none")

    # wave_strength vs n_start (lines by weights; best K per weights)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    for w in weights_list:
        ys = []
        xs = []
        for n0 in n_starts:
            cand = [r for r in rows_none if r.weights == w and r.n_start == n0]
            if not cand:
                continue
            best = max(cand, key=lambda r: (r.wave_strength if not math.isnan(r.wave_strength) else -1e18))
            xs.append(n0)
            ys.append(best.wave_strength)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{w} (best K)")
    ax.set_xscale("log")
    ax.set_xlabel("n_start")
    ax.set_ylabel("wave_strength (median prominence / MAD)")
    ax.set_title("M35: wave strength vs n_start (best K per weights)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m35_wave_strength_vs_nstart.png")
    plt.close(fig)

    # mean_dx vs n_start (best per weights)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    for w in weights_list:
        ys = []
        xs = []
        for n0 in n_starts:
            cand = [r for r in rows_none if r.weights == w and r.n_start == n0]
            if not cand:
                continue
            best = max(cand, key=lambda r: (r.wave_strength if not math.isnan(r.wave_strength) else -1e18))
            xs.append(n0)
            ys.append(best.mean_dx)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{w} (best K)")
    ax.set_xscale("log")
    ax.set_xlabel("n_start")
    ax.set_ylabel("mean_dx (phase corr shift)")
    ax.set_title("M35: mean dx vs n_start (best config per weights)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m35_mean_dx_vs_nstart.png")
    plt.close(fig)

    # q_eff vs n_start
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    for w in weights_list:
        ys = []
        xs = []
        for n0 in n_starts:
            cand = [r for r in rows_none if r.weights == w and r.n_start == n0]
            if not cand:
                continue
            best = max(cand, key=lambda r: (r.wave_strength if not math.isnan(r.wave_strength) else -1e18))
            xs.append(n0)
            ys.append(best.q_eff)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{w} (best K)")
    ax.set_xscale("log")
    ax.set_xlabel("n_start")
    ax.set_ylabel("q_eff ≈ dt/mean_dx")
    ax.set_title("M35: q_eff proxy vs n_start (best config per weights)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m35_qeff_vs_nstart.png")
    plt.close(fig)

    # heatmap wave_strength (K x weights) for each n_start (small multiples)
    ncols = len(n_starts)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.4), dpi=140, squeeze=False)
    for j, n0 in enumerate(n_starts):
        mat = np.full((len(weights_list), len(K_list)), np.nan, dtype=np.float64)
        for wi, w in enumerate(weights_list):
            for ki, K in enumerate(K_list):
                match = [r for r in rows_none if r.n_start == n0 and r.weights == w and r.K == K]
                if match:
                    mat[wi, ki] = match[0].wave_strength
        ax = axes[0, j]
        im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"n_start={n0}")
        ax.set_xticks(range(len(K_list)))
        ax.set_xticklabels([str(k) for k in K_list])
        ax.set_yticks(range(len(weights_list)))
        ax.set_yticklabels(weights_list)
        ax.set_xlabel("K")
        if j == 0:
            ax.set_ylabel("weights")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "m35_heatmap_wave_strength.png")
    plt.close(fig)

    # track_r2 summary (best per weights)
    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=140)
    for w in weights_list:
        ys = []
        xs = []
        for n0 in n_starts:
            cand = [r for r in rows_none if r.weights == w and r.n_start == n0]
            if not cand:
                continue
            best = max(cand, key=lambda r: (r.wave_strength if not math.isnan(r.wave_strength) else -1e18))
            xs.append(n0)
            ys.append(best.track_r2)
        if xs:
            ax.plot(xs, ys, marker="o", label=f"{w} (best K)")
    ax.set_xscale("log")
    ax.set_xlabel("n_start")
    ax.set_ylabel("track_r2")
    ax.set_title("M35: k_peak(t) linearity (R^2) vs n_start (best config per weights)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m35_track_r2.png")
    plt.close(fig)


def make_table_tex(out_path: Path, top_rows: List[RunResult]) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$n_{\mathrm{start}}$ & $K$ & weights & mean\_dx & $q_{\mathrm{eff}}$ & strength & $R^2$ \\")
    lines.append(r"\hline")
    for r in top_rows:
        qeff = r.q_eff
        qeff_s = f"{qeff:.3g}" if not math.isnan(qeff) else "nan"
        lines.append(
            f"{r.n_start} & {r.K} & {r.weights} & {r.mean_dx:.3g} & {qeff_s} & {r.wave_strength:.3g} & {r.track_r2:.3g} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-start-list", type=str, required=True)
    p.add_argument("--K-list", type=str, required=True)
    p.add_argument("--H-policy", type=str, default="matchK", choices=["matchK"])
    p.add_argument("--weights", type=str, default="invq,log1pq,ones")
    p.add_argument("--frames", type=int, default=600)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--dt", type=int, default=5)
    p.add_argument("--smooth-policy", type=str, default="auto", choices=["auto"])
    p.add_argument("--peaks", type=int, default=3)
    p.add_argument("--conf-min", type=float, default=0.15)
    p.add_argument("--sanity", type=str, default="one_per_nstart", choices=["none", "one_per_nstart"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def smooth_auto(K: int) -> int:
    base = max(1, int(round(K / 64)))
    w = base * 9
    if w < 7:
        w = 7
    if w % 2 == 0:
        w += 1
    return int(w)


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    runs_dir = out_root / "runs"
    ensure_dir(runs_dir)

    n_starts = parse_int_list(args.n_start_list)
    K_list = parse_int_list(args.K_list)
    weights_list = parse_str_list(args.weights)

    seed = int(args.seed)

    rows: List[RunResult] = []
    runs_manifest: Dict[str, Any] = {"params": vars(args), "git_sha": git_sha(), "runs": []}
    skipped_n: List[int] = []

    # If a very large n_start appears to be too slow, skip it and record.
    max_seconds_first_run = 75.0

    for n0 in n_starts:
        n_started = time.time()
        first_run_runtime: Optional[float] = None

        for K in K_list:
            H = K
            smooth = smooth_auto(K)
            for w in weights_list:
                cfg = RunConfig(
                    n_start=n0,
                    K=K,
                    H=H,
                    weights=w,
                    frames=int(args.frames),
                    n_step=int(args.n_step),
                    dt=int(args.dt),
                    smooth=int(smooth),
                    peaks=int(args.peaks),
                    conf_min=float(args.conf_min),
                    sanity="none",
                )
                rid = run_id(cfg)
                out_dir = runs_dir / rid
                ensure_dir(out_dir)

                res, tracks = compute_run(cfg, out_dir, seed=seed)
                rows.append(res)
                runs_manifest["runs"].append({"run_id": rid, "dir": str(out_dir.relative_to(out_root)).replace("\\", "/")})

                if first_run_runtime is None:
                    first_run_runtime = res.runtime_s
                    if first_run_runtime > max_seconds_first_run and n0 >= 200_000_000:
                        skipped_n.append(n0)
                        break

            if n0 in skipped_n:
                break
        if n0 in skipped_n:
            # remove any partial run dirs for that n_start beyond the ones already computed
            continue

        # sanity: one permute_cols per n_start (fixed baseline config)
        if args.sanity == "one_per_nstart":
            K = 512 if 512 in K_list else K_list[0]
            H = K
            smooth = smooth_auto(K)
            cfg_s = RunConfig(
                n_start=n0,
                K=K,
                H=H,
                weights="invq" if "invq" in weights_list else weights_list[0],
                frames=int(args.frames),
                n_step=int(args.n_step),
                dt=int(args.dt),
                smooth=int(smooth),
                peaks=int(args.peaks),
                conf_min=float(args.conf_min),
                sanity="permute_cols",
            )
            rid = run_id(cfg_s)
            out_dir = runs_dir / rid
            ensure_dir(out_dir)
            res, _tracks = compute_run(cfg_s, out_dir, seed=seed)
            rows.append(res)
            runs_manifest["runs"].append({"run_id": rid, "dir": str(out_dir.relative_to(out_root)).replace("\\", "/")})

        _ = n_started  # reserved for future runtime summaries

    # write summary tables
    header = [
        "run_id",
        "n_start",
        "K",
        "H",
        "weights",
        "smooth",
        "dt",
        "frames",
        "conf_min",
        "sanity",
        "runtime_s",
        "mean_dx",
        "median_dx",
        "q_eff",
        "slope_kpeak",
        "track_r2",
        "valid_frac",
        "wave_strength",
        "notes",
    ]
    rows_dicts = [asdict(r) for r in rows]
    write_csv(out_root / "m35_summary.csv", rows_dicts, header)

    summary_json = {
        "params": vars(args),
        "git_sha": git_sha(),
        "skipped_n_start": skipped_n,
        "n_runs": len(rows),
    }
    (out_root / "m35_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")
    (out_root / "m35_runs_manifest.json").write_text(json.dumps(runs_manifest, indent=2), encoding="utf-8")

    # plots
    n_starts_used = [n for n in n_starts if n not in skipped_n]
    plot_summary(out_root, rows, n_starts_used, K_list, weights_list)

    # best-of overlay (largest available n_start)
    if n_starts_used:
        best_n = max(n_starts_used)
        cand = [r for r in rows if r.sanity == "none" and r.n_start == best_n]
        if cand:
            def visibility_score(rr: RunResult) -> float:
                ws = rr.wave_strength if not math.isnan(rr.wave_strength) else 0.0
                r2 = rr.track_r2 if not math.isnan(rr.track_r2) else 0.0
                vf = rr.valid_frac if not math.isnan(rr.valid_frac) else 0.0
                return float(ws) * math.sqrt(max(r2, 0.0)) * math.sqrt(max(vf, 0.0))

            best = max(cand, key=visibility_score)

            cfg_best = RunConfig(
                n_start=int(best.n_start),
                K=int(best.K),
                H=int(best.H),
                weights=str(best.weights),
                frames=int(args.frames),
                n_step=int(args.n_step),
                dt=int(args.dt),
                smooth=int(best.smooth),
                peaks=int(args.peaks),
                conf_min=float(args.conf_min),
                sanity="none",
            )
            best_dir = runs_dir / best.run_id
            tracks_path = best_dir / "tracks_top3.json"
            # recompute tracks deterministically and save minimal json (for reproducibility)
            _, tracks = compute_run(cfg_best, best_dir, seed=seed)
            tracks_path.write_text(json.dumps(tracks, indent=2), encoding="utf-8")

            overlay_root = out_root / "best_overlay"
            real_dir = overlay_root / "real"
            sanity_dir = overlay_root / "sanity"
            ensure_dir(real_dir)
            ensure_dir(sanity_dir)
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("ffmpeg not found; cannot render best overlay mp4")

            real_mp4 = render_tail_overlay_mp4(
                out_dir=real_dir,
                cfg=cfg_best,
                tracks=tracks,
                seed=seed,
                sanity="none",
                fps=30,
                format_="mp4",
            )
            sanity_mp4 = render_tail_overlay_mp4(
                out_dir=sanity_dir,
                cfg=cfg_best,
                tracks=tracks,
                seed=seed,
                sanity="permute_cols",
                fps=30,
                format_="mp4",
            )

            # preview: stitch keyframes (real vs sanity)
            import matplotlib.pyplot as plt
            import matplotlib.image as mpimg

            fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.5), dpi=140)
            real_kf = real_dir / "keyframe_t150.png"
            sanity_kf = sanity_dir / "keyframe_t150.png"
            axes[0].imshow(mpimg.imread(real_kf))
            axes[0].set_axis_off()
            axes[0].set_title("real (tail)")
            axes[1].imshow(mpimg.imread(sanity_kf))
            axes[1].set_axis_off()
            axes[1].set_title("sanity permute_cols (tail)")
            fig.tight_layout()
            fig.savefig(overlay_root / "m35_best_preview.png")
            plt.close(fig)

            (overlay_root / "m35_best_config.json").write_text(
                json.dumps(
                    {
                        "best_run_id": best.run_id,
                        "visibility_score": visibility_score(best),
                        "config": asdict(cfg_best),
                        "real_video": str(real_mp4.relative_to(out_root)).replace("\\", "/"),
                        "sanity_video": str(sanity_mp4.relative_to(out_root)).replace("\\", "/"),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            # tex table: top-1 per n_start (best by visibility score)
            top_rows: List[RunResult] = []
            for n0 in n_starts_used:
                cand_n = [r for r in rows if r.sanity == "none" and r.n_start == n0]
                if not cand_n:
                    continue
                top_rows.append(max(cand_n, key=visibility_score))
            make_table_tex(out_root / "m35_table.tex", top_rows)

    write_manifest(out_root, params={**vars(args), "skipped_n_start": skipped_n})
    print(f"OK: wrote M35 sweep to {out_root}")


if __name__ == "__main__":
    main()

