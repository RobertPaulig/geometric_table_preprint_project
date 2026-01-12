from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated int list")
    return out


def parse_theta_range(s: str) -> Tuple[float, float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("Expected --theta-range-deg as start,end (e.g. 88.0,90.0)")
    a = float(parts[0])
    b = float(parts[1])
    if b <= a:
        raise ValueError("theta range must satisfy end > start")
    return a, b


def theta_grid(start: float, end: float, step: float) -> np.ndarray:
    if step <= 0:
        raise ValueError("theta step must be positive")
    # include end if within half-step
    n = int(math.floor((end - start) / step + 0.5)) + 1
    out = start + step * np.arange(n, dtype=np.float64)
    out = out[out <= end + 1e-12]
    if out.size == 0:
        raise ValueError("theta grid is empty")
    return out


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


def build_values_invq_profile(*, n_top: int, H: int, k_start: int, W: int) -> np.ndarray:
    prof = np.zeros(W, dtype=np.float32)
    n_end = n_top + H

    for j in range(W):
        k = k_start + j
        first = ((n_top + k - 1) // k) * k
        if first >= n_end:
            continue
        acc = 0.0
        for n in range(first, n_end, k):
            q = n // k
            acc += 1.0 / float(q)
        prof[j] = float(acc)

    return prof


def build_values_invq_heatmap(*, n_top: int, H: int, k_start: int, W: int) -> np.ndarray:
    img = np.zeros((H, W), dtype=np.float32)
    n_end = n_top + H

    for j in range(W):
        k = k_start + j
        first = ((n_top + k - 1) // k) * k
        if first >= n_end:
            continue
        for n in range(first, n_end, k):
            q = n // k
            img[n - n_top, j] = 1.0 / float(q)

    return img


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    w = int(w)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def top_peaks(profile: np.ndarray, peaks: int) -> Tuple[List[int], List[float]]:
    peaks = int(peaks)
    if peaks <= 0:
        return [], []
    if profile.size == 0:
        return [], []
    k = min(peaks, int(profile.size))
    idx = np.argpartition(-profile, kth=k - 1)[:k]
    idx = idx[np.argsort(-profile[idx])]
    return (idx.tolist(), [float(profile[i]) for i in idx.tolist()])


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


def angle_scores_per_frame(
    *,
    n_start: int,
    q0: int,
    H: int,
    W: int,
    frames: int,
    n_step: int,
    thetas_deg: np.ndarray,
    x_idx: np.ndarray,
    mask: np.ndarray,
    topk_offsets: int,
    sanity: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y_idx = np.arange(H, dtype=np.int32)
    n_theta = int(thetas_deg.size)

    score = np.zeros((frames, n_theta), dtype=np.float64)
    score_raw_sum = np.zeros((frames, n_theta), dtype=np.float64)

    perms: Optional[List[np.ndarray]] = None
    if sanity == "permute_cols":
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_start, q0, 40]))
        perms = [rng.permutation(W).astype(np.int32) for _ in range(frames)]

    for t in range(frames):
        n_top = n_start + t * n_step
        k_start, _k_end = k_window_for_frame(n_top, H, W, q0)
        img = build_values_invq_heatmap(n_top=n_top, H=H, k_start=k_start, W=W)
        if perms is not None:
            img = img[:, perms[t]]

        vals = img[y_idx[None, None, :], x_idx]  # (theta, off, y)
        vals = vals * mask
        sums = np.sum(vals, axis=2)  # (theta, off)
        score_raw_sum[t, :] = np.mean(sums, axis=1)
        if topk_offsets <= 1:
            score[t, :] = np.max(sums, axis=1)
        else:
            k = min(int(topk_offsets), sums.shape[1])
            topk = np.partition(sums, kth=sums.shape[1] - k, axis=1)[:, -k:]
            score[t, :] = np.mean(topk, axis=1)

    return score, score_raw_sum


def bootstrap_theta_ci(
    *,
    thetas_deg: np.ndarray,
    per_frame_scores: np.ndarray,
    bootstrap: int,
    seed: int,
) -> Tuple[float, float, float, np.ndarray]:
    # Returns (theta_star, ci_low, ci_high, boot_thetas).
    n_frames = int(per_frame_scores.shape[0])
    total = np.sum(per_frame_scores, axis=0)
    theta_star = float(thetas_deg[int(np.argmax(total))])

    rng = np.random.default_rng(np.random.SeedSequence([seed, 40, 12345]))
    boot = np.zeros(int(bootstrap), dtype=np.float64)
    for b in range(int(bootstrap)):
        idx = rng.integers(0, n_frames, size=n_frames, dtype=np.int32)
        tot = np.sum(per_frame_scores[idx, :], axis=0)
        boot[b] = float(thetas_deg[int(np.argmax(tot))])

    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return theta_star, lo, hi, boot


def bootstrap_mean_ci(x: np.ndarray, bootstrap: int, seed: int) -> Tuple[float, float, float, np.ndarray]:
    # Returns (mean, lo, hi, boot_means).
    x0 = x[np.isfinite(x)].astype(np.float64)
    if x0.size == 0:
        return float("nan"), float("nan"), float("nan"), np.array([], dtype=np.float64)
    mean = float(np.mean(x0))
    rng = np.random.default_rng(np.random.SeedSequence([seed, 41, 54321]))
    boot = np.zeros(int(bootstrap), dtype=np.float64)
    n = int(x0.size)
    for b in range(int(bootstrap)):
        idx = rng.integers(0, n, size=n, dtype=np.int32)
        boot[b] = float(np.mean(x0[idx]))
    lo = float(np.percentile(boot, 2.5))
    hi = float(np.percentile(boot, 97.5))
    return mean, lo, hi, boot


def theta_from_q(q: float) -> float:
    if not math.isfinite(q) or q <= 0:
        return float("nan")
    return float(math.degrees(math.atan(q)))


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def plot_score_curve(
    *,
    out_path: Path,
    thetas_deg: np.ndarray,
    score_total: np.ndarray,
    theta_star: float,
    theta_dx: Optional[float],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=150)
    ax.plot(thetas_deg, score_total, linewidth=1.6)
    ax.axvline(theta_star, color="#00d4ff", linewidth=1.3, linestyle="--", label=f"theta*={theta_star:.3f}°")
    if theta_dx is not None and math.isfinite(theta_dx):
        ax.axvline(theta_dx, color="#ffd200", linewidth=1.2, linestyle=":", label=f"theta_dx={theta_dx:.3f}°")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("score(theta)")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_angle_ci(
    *,
    out_path: Path,
    dt_list: List[int],
    theta_star: float,
    theta_ci: Tuple[float, float],
    theta_dx: List[float],
    theta_dx_ci: List[Tuple[float, float]],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.array(dt_list, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=150)
    ax.fill_between(xs, theta_ci[0], theta_ci[1], color="#00d4ff", alpha=0.20, label="theta* CI")
    ax.plot(xs, [theta_star for _ in dt_list], color="#00d4ff", linewidth=1.6, label="theta* (angle)")

    y = np.array(theta_dx, dtype=np.float64)
    ylo = np.array([c[0] for c in theta_dx_ci], dtype=np.float64)
    yhi = np.array([c[1] for c in theta_dx_ci], dtype=np.float64)
    ax.errorbar(xs, y, yerr=[y - ylo, yhi - y], fmt="o-", color="#ffd200", linewidth=1.4, markersize=4, label="theta_dx (from dx)")

    ax.set_xlabel("dt (frames)")
    ax.set_ylabel("theta (deg)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_compare_dx_vs_angle(
    *,
    out_path: Path,
    dt_list: List[int],
    q_eff_dx: List[float],
    q_eff_dx_ci: List[Tuple[float, float]],
    q_eff_angle: float,
    q_eff_angle_ci: Tuple[float, float],
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    xs = np.array(dt_list, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7.2, 3.9), dpi=150)

    y = np.array(q_eff_dx, dtype=np.float64)
    ylo = np.array([c[0] for c in q_eff_dx_ci], dtype=np.float64)
    yhi = np.array([c[1] for c in q_eff_dx_ci], dtype=np.float64)
    ax.errorbar(xs, y, yerr=[y - ylo, yhi - y], fmt="o-", color="#ffd200", linewidth=1.4, markersize=4, label="q_eff (from dx)")

    ax.fill_between(xs, q_eff_angle_ci[0], q_eff_angle_ci[1], color="#00d4ff", alpha=0.20, label="q_eff(angle) CI")
    ax.plot(xs, [q_eff_angle for _ in dt_list], color="#00d4ff", linewidth=1.6, label="q_eff (from angle)")

    ax.set_xlabel("dt (frames)")
    ax.set_ylabel("q_eff")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_sanity_compare(
    *,
    out_path: Path,
    thetas_deg: np.ndarray,
    score_real: np.ndarray,
    theta_real: float,
    score_sanity: np.ndarray,
    theta_sanity: float,
) -> None:
    import matplotlib.pyplot as plt

    def norm(x: np.ndarray) -> np.ndarray:
        x0 = x.astype(np.float64)
        m = float(np.max(x0)) if np.isfinite(x0).any() else 1.0
        if m <= 0:
            m = 1.0
        return x0 / m

    fig, ax = plt.subplots(figsize=(7.4, 4.0), dpi=150)
    ax.plot(thetas_deg, norm(score_real), linewidth=1.6, label=f"real (theta*={theta_real:.3f}°)")
    ax.plot(thetas_deg, norm(score_sanity), linewidth=1.3, alpha=0.85, label=f"sanity (theta*={theta_sanity:.3f}°)")
    ax.set_xlabel("theta (deg)")
    ax.set_ylabel("score(theta) (normalized)")
    ax.set_title("M40 sanity compare: score(theta) real vs permute_cols")
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_table_tex(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$dt$ & mean\_dx & $q_{\mathrm{eff}}$ & $\theta^*$ & $\theta^*$ CI & $\theta_{\mathrm{dx}}$ \\")
    lines.append(r"\hline")
    for r in rows:
        dt = int(r["dt"])
        mean_dx = float(r["dx_mean"])
        q_eff = float(r["q_eff"]) if math.isfinite(float(r["q_eff"])) else float("nan")
        theta_star = float(r["theta_star"])
        ci_lo = float(r["theta_ci_lo"])
        ci_hi = float(r["theta_ci_hi"])
        theta_dx = float(r["theta_dx"]) if math.isfinite(float(r["theta_dx"])) else float("nan")
        lines.append(
            f"{dt} & {mean_dx:.3g} & {q_eff:.3g} & {theta_star:.3f} & [{ci_lo:.3f},{ci_hi:.3f}] & {theta_dx:.3f} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-start", type=int, required=True)
    p.add_argument("--q0", type=int, required=True)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--mode", type=str, default="values", choices=["values"])
    p.add_argument("--weights", type=str, default="invq", choices=["invq"])
    p.add_argument("--theta-range-deg", type=str, required=True)
    p.add_argument("--theta-step-deg", type=float, default=0.02)
    p.add_argument("--dt-list", type=str, required=True)
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument("--peaks", type=int, default=3)
    p.add_argument("--conf-min", type=float, default=0.15)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--sanity", type=str, default="none", choices=["none", "permute_cols"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)

    n_start = int(args.n_start)
    q0 = int(args.q0)
    H = int(args.H)
    W = int(args.W)
    frames = int(args.frames)
    n_step = int(args.n_step)
    dt_list = parse_int_list(args.dt_list)
    smooth = int(args.smooth)
    peaks = int(args.peaks)
    conf_min = float(args.conf_min)
    bootstrap = int(args.bootstrap)
    sanity = str(args.sanity)
    seed = int(args.seed)

    theta_start, theta_end = parse_theta_range(args.theta_range_deg)
    thetas = theta_grid(theta_start, theta_end, float(args.theta_step_deg))

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found (required for overlay mp4 outputs in other milestones)")

    # Peak tracking (k_peak) for dx estimation.
    started = time.time()
    perms: Optional[List[np.ndarray]] = None
    if sanity == "permute_cols":
        rng = np.random.default_rng(np.random.SeedSequence([seed, n_start, q0, 40]))
        perms = [rng.permutation(W).astype(np.int32) for _ in range(frames)]

    k_starts: List[int] = []
    x_peak0: List[int] = []
    z_peak: List[float] = []
    k_peak_abs: List[float] = []
    for t in range(frames):
        n_top = n_start + t * n_step
        k_start, _k_end = k_window_for_frame(n_top, H, W, q0)
        k_starts.append(int(k_start))
        prof = build_values_invq_profile(n_top=n_top, H=H, k_start=k_start, W=W)
        if perms is not None:
            prof = prof[perms[t]]
        prof_sm = rolling_mean(prof, smooth)
        idx, vals = top_peaks(prof_sm, peaks)
        x0 = int(idx[0]) if idx else -1
        x_peak0.append(x0)
        peak_val = float(vals[0]) if vals else float("nan")
        baseline = float(np.median(prof_sm))
        scale = mad(prof_sm)
        if scale <= 0:
            scale = float(np.std(prof_sm))
        z = float((peak_val - baseline) / (scale + 1e-9)) if math.isfinite(peak_val) else float("nan")
        z_peak.append(z)
        k_peak_abs.append(float(k_start + x0) if x0 >= 0 else float("nan"))

    # Angle scan (score(theta) near vertical).
    x_center = (W - 1) / 2.0
    y_center = (H - 1) / 2.0
    x_offsets = [-64, -32, -16, 0, 16, 32, 64]
    x_idx, mask = precompute_line_indices(H=H, W=W, thetas_deg=thetas, x_offsets=x_offsets, x_center=x_center, y_center=y_center)
    # Reuse the same sanity permutations for the angle score computation.
    score_pf, _score_raw_pf = angle_scores_per_frame(
        n_start=n_start,
        q0=q0,
        H=H,
        W=W,
        frames=frames,
        n_step=n_step,
        thetas_deg=thetas,
        x_idx=x_idx,
        mask=mask,
        topk_offsets=3,
        sanity=sanity,
        seed=seed,
    )
    theta_star, theta_ci_lo, theta_ci_hi, boot_thetas = bootstrap_theta_ci(
        thetas_deg=thetas, per_frame_scores=score_pf, bootstrap=bootstrap, seed=seed
    )
    score_total = np.sum(score_pf, axis=0)
    score_peak = float(np.max(score_total))

    # dx estimates per dt + bootstrap CI.
    rows: List[Dict[str, Any]] = []
    theta_dx_list: List[float] = []
    theta_dx_ci_list: List[Tuple[float, float]] = []
    q_eff_dx_list: List[float] = []
    q_eff_dx_ci_list: List[Tuple[float, float]] = []
    dx_ci_list: List[Tuple[float, float]] = []

    for dt in dt_list:
        dxs: List[float] = []
        for t in range(dt, frames):
            x_now = x_peak0[t]
            x_prev = x_peak0[t - dt]
            if x_now < 0 or x_prev < 0:
                continue
            conf = min(z_peak[t], z_peak[t - dt])
            if not math.isfinite(conf) or conf < conf_min:
                continue
            k_now = k_starts[t] + x_now
            k_prev = k_starts[t - dt] + x_prev
            dxs.append(float(k_now - k_prev))

        dx_arr = np.array(dxs, dtype=np.float64)
        dx_mean, dx_lo, dx_hi, boot_dx = bootstrap_mean_ci(dx_arr, bootstrap=bootstrap, seed=seed + dt)
        dx_ci_list.append((dx_lo, dx_hi))

        q_eff = float(dt) / dx_mean if (math.isfinite(dx_mean) and dx_mean > 0) else float("nan")
        q_eff_dx_list.append(q_eff)
        q_eff_boot = float(dt) / boot_dx if boot_dx.size > 0 else np.array([], dtype=np.float64)
        q_eff_lo = float(np.percentile(q_eff_boot, 2.5)) if q_eff_boot.size > 0 else float("nan")
        q_eff_hi = float(np.percentile(q_eff_boot, 97.5)) if q_eff_boot.size > 0 else float("nan")
        q_eff_dx_ci_list.append((q_eff_lo, q_eff_hi))

        theta_dx = theta_from_q(q_eff)
        theta_dx_list.append(theta_dx)
        theta_dx_boot = np.degrees(np.arctan(q_eff_boot)) if q_eff_boot.size > 0 else np.array([], dtype=np.float64)
        th_lo = float(np.percentile(theta_dx_boot, 2.5)) if theta_dx_boot.size > 0 else float("nan")
        th_hi = float(np.percentile(theta_dx_boot, 97.5)) if theta_dx_boot.size > 0 else float("nan")
        theta_dx_ci_list.append((th_lo, th_hi))

        rows.append(
            {
                "dt": int(dt),
                "dx_mean": float(dx_mean),
                "dx_ci_lo": float(dx_lo),
                "dx_ci_hi": float(dx_hi),
                "q_eff": float(q_eff),
                "q_eff_ci_lo": float(q_eff_lo),
                "q_eff_ci_hi": float(q_eff_hi),
                "theta_star": float(theta_star),
                "theta_ci_lo": float(theta_ci_lo),
                "theta_ci_hi": float(theta_ci_hi),
                "theta_dx": float(theta_dx),
                "theta_dx_ci_lo": float(th_lo),
                "theta_dx_ci_hi": float(th_hi),
                "angle_score_peak": float(score_peak),
                "sanity": sanity,
            }
        )

    # Outputs (real or sanity dir).
    write_csv(
        out_root / "m40_angle_vs_dt.csv",
        rows,
        [
            "dt",
            "dx_mean",
            "dx_ci_lo",
            "dx_ci_hi",
            "q_eff",
            "q_eff_ci_lo",
            "q_eff_ci_hi",
            "theta_star",
            "theta_ci_lo",
            "theta_ci_hi",
            "theta_dx",
            "theta_dx_ci_lo",
            "theta_dx_ci_hi",
            "angle_score_peak",
            "sanity",
        ],
    )

    # Score curves: show same score(theta), but dt-specific dx-based theta marker.
    dt_set = set(dt_list)
    if 5 in dt_set:
        theta_dx_5 = theta_dx_list[dt_list.index(5)]
        plot_score_curve(
            out_path=out_root / "m40_theta_score_curve_dt5.png",
            thetas_deg=thetas,
            score_total=score_total,
            theta_star=theta_star,
            theta_dx=theta_dx_5,
            title="M40 score(theta) near vertical (dt=5 marker)",
        )
    if 40 in dt_set:
        theta_dx_40 = theta_dx_list[dt_list.index(40)]
        plot_score_curve(
            out_path=out_root / "m40_theta_score_curve_dt40.png",
            thetas_deg=thetas,
            score_total=score_total,
            theta_star=theta_star,
            theta_dx=theta_dx_40,
            title="M40 score(theta) near vertical (dt=40 marker)",
        )

    # CI plot: theta* vs dt + theta_dx(dt) with CI.
    plot_angle_ci(
        out_path=out_root / "m40_angle_ci.png",
        dt_list=dt_list,
        theta_star=theta_star,
        theta_ci=(theta_ci_lo, theta_ci_hi),
        theta_dx=theta_dx_list,
        theta_dx_ci=theta_dx_ci_list,
        title="M40: angle estimate vs dt (dx vs angle)",
    )

    q_eff_angle = float(math.tan(math.radians(theta_star))) if math.isfinite(theta_star) else float("nan")
    q_eff_angle_boot = np.tan(np.radians(boot_thetas)) if boot_thetas.size > 0 else np.array([], dtype=np.float64)
    q_eff_angle_ci = (
        float(np.percentile(q_eff_angle_boot, 2.5)) if q_eff_angle_boot.size > 0 else float("nan"),
        float(np.percentile(q_eff_angle_boot, 97.5)) if q_eff_angle_boot.size > 0 else float("nan"),
    )
    plot_compare_dx_vs_angle(
        out_path=out_root / "m40_compare_dx_vs_angle.png",
        dt_list=dt_list,
        q_eff_dx=q_eff_dx_list,
        q_eff_dx_ci=q_eff_dx_ci_list,
        q_eff_angle=q_eff_angle,
        q_eff_angle_ci=q_eff_angle_ci,
        title="M40: q_eff from dx vs q_eff from angle",
    )

    # TeX table
    build_table_tex(out_root / "m40_table.tex", rows)

    runtime_s = float(time.time() - started)
    summary = {
        "params": vars(args),
        "git_sha": git_sha(),
        "runtime_s": runtime_s,
        "theta_star_deg": theta_star,
        "theta_ci_deg": [theta_ci_lo, theta_ci_hi],
        "angle_score_peak": score_peak,
        "q_eff_angle": q_eff_angle,
        "q_eff_angle_ci": list(q_eff_angle_ci),
    }
    (out_root / "m40_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # If sanity run, also write a compare figure into parent dir.
    parent_root = out_root if out_root.name != "sanity_permute_cols" else out_root.parent
    if sanity == "permute_cols" and out_root.name == "sanity_permute_cols":
        real_csv = parent_root / "m40_angle_vs_dt.csv"
        if real_csv.exists():
            # Load real score curve from parent by recomputing from its own per-frame score cache is not stored;
            # instead, we recompute a real curve quickly (only one config).
            score_real_pf, _ = angle_scores_per_frame(
                n_start=n_start,
                q0=q0,
                H=H,
                W=W,
                frames=frames,
                n_step=n_step,
                thetas_deg=thetas,
                x_idx=x_idx,
                mask=mask,
                topk_offsets=3,
                sanity="none",
                seed=seed,
            )
            score_real_total = np.sum(score_real_pf, axis=0)
            theta_real = float(thetas[int(np.argmax(score_real_total))])
            theta_sanity = float(thetas[int(np.argmax(score_total))])
            plot_sanity_compare(
                out_path=parent_root / "m40_sanity_compare.png",
                thetas_deg=thetas,
                score_real=score_real_total,
                theta_real=theta_real,
                score_sanity=score_total,
                theta_sanity=theta_sanity,
            )

    # Manifests (parent updated after sanity).
    write_manifest(out_root, params={**vars(args), "runtime_s": runtime_s}, name="m40_manifest.json")
    if out_root.name == "sanity_permute_cols":
        write_manifest(parent_root, params={"note": "parent includes real + sanity outputs", "git_sha": git_sha()}, name="m40_manifest.json")

    print(f"OK: wrote M40 outputs to {out_root}")


if __name__ == "__main__":
    main()

