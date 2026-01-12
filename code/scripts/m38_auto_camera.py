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


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def parse_q0_grid(s: str) -> List[int]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Expected --q0-grid as start,end,step (e.g. 6,20,1)")
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
    w = int(w)
    kernel = np.ones(w, dtype=np.float32) / float(w)
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


def n_tag(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"n{n//1_000_000}e6"
    return f"n{n}"


def n_sci_tag(n: int) -> str:
    n0 = int(n)
    if n0 == 0:
        return "n0"
    exp = 0
    m = n0
    while m % 10 == 0:
        m //= 10
        exp += 1
    return f"n{m}e{exp}"


def detect_artifact_prefix(out_dir: Path) -> str:
    base = out_dir.name
    if base == "sanity_permute_cols":
        base = out_dir.parent.name
    if base.startswith("m") and len(base) >= 2 and base[1].isdigit():
        return base
    return "m38"


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
    if peaks <= 0:
        return [], []
    if profile.size == 0:
        return [], []
    k = min(peaks, int(profile.size))
    idx = np.argpartition(-profile, kth=k - 1)[:k]
    idx = idx[np.argsort(-profile[idx])]
    return (idx.tolist(), [float(profile[i]) for i in idx.tolist()])


@dataclass(frozen=True)
class RunConfig:
    n_start: int
    q0: int
    H: int
    W: int
    frames: int
    n_step: int
    dt: int
    smooth: int
    peaks: int
    conf_min: float
    mode: str
    weights: str
    sanity: str  # none|permute_cols

    def run_id(self) -> str:
        return f"{n_tag(self.n_start)}_q{self.q0}_H{self.H}_W{self.W}_{self.weights}_dt{self.dt}_sm{self.smooth}_{self.sanity}"


@dataclass
class RunResult:
    run_id: str
    n_start: int
    q0: int
    H: int
    W: int
    weights: str
    dt: int
    smooth: int
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
    visibility_score: float
    notes: str


def render_keyframe(
    *,
    out_path: Path,
    cfg: RunConfig,
    img: np.ndarray,
    k_start: int,
    tracks: Dict[int, Dict[int, int]],
    t_key: int,
    mean_dx: float,
    q_eff: float,
) -> None:
    import matplotlib.pyplot as plt

    H, W = img.shape
    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=130)
    ax.imshow(img, aspect="auto", origin="upper", cmap="magma", vmin=0.0, vmax=0.02)
    ax.set_xlabel(f"k index within window (k_start={k_start})")
    ax.set_ylabel("rows (n increasing down)")
    ax.set_title(f"M38 auto-camera ({cfg.sanity})  n_start={cfg.n_start}  q0={cfg.q0}  t={t_key}")

    colors = ["#00d4ff", "#00ff6a", "#ffd200"]
    any_line = False
    for pid in range(min(cfg.peaks, 3)):
        xs: List[float] = []
        ys: List[float] = []
        for y in range(cfg.H):
            tt = t_key - y
            if tt < 0:
                break
            k_hist = tracks.get(tt, {}).get(pid)
            if k_hist is None or k_hist < 0:
                continue
            x = float(k_hist)
            if 0.0 <= x < float(W):
                xs.append(x)
                ys.append(float(y))
        if len(xs) >= 2:
            ax.plot(xs, ys, color=colors[pid], linewidth=1.6, alpha=0.95, label=f"peak{pid+1}")
            any_line = True

    qeff_s = f"{q_eff:.3g}" if (not math.isnan(q_eff)) else "nan"
    mdx_s = f"{mean_dx:.3g}" if (not math.isnan(mean_dx)) else "nan"
    txt = f"W={cfg.W}  H={cfg.H}  dt={cfg.dt}  mean_dx={mdx_s}  q_eff~{qeff_s}"
    ax.text(
        0.01,
        0.01,
        txt,
        transform=ax.transAxes,
        fontsize=9.0,
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, pad=3),
    )
    if any_line:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_best_overlay_video(
    *,
    out_dir: Path,
    artifact_prefix: str,
    cfg: RunConfig,
    tracks: Dict[int, Dict[int, int]],
    k_starts: List[int],
    perms_by_t: Optional[List[np.ndarray]],
    fps: int,
    format_: str,
) -> Path:
    import matplotlib.pyplot as plt

    ensure_dir(out_dir)
    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    ensure_dir(frames_dir)

    fig, ax = plt.subplots(figsize=(7.2, 5.8), dpi=110)
    colors = ["#00d4ff", "#00ff6a", "#ffd200"]
    lines = [
        ax.plot([], [], color=colors[i], linewidth=1.6, alpha=0.95, label=f"peak{i+1}")[0]
        for i in range(min(cfg.peaks, 3))
    ]

    n_top0 = cfg.n_start
    k_start0, _ = k_window_for_frame(n_top0, cfg.H, cfg.W, cfg.q0)
    img0, _prof0 = build_values_invq_heatmap_and_profile(n_top=n_top0, H=cfg.H, k_start=k_start0, W=cfg.W)
    if perms_by_t is not None:
        img0 = img0[:, perms_by_t[0]]
    im = ax.imshow(img0, aspect="auto", origin="upper", cmap="magma", vmin=0.0, vmax=0.02)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("k index (within window)")
    ax.set_ylabel("rows (n increasing down)")
    fig.tight_layout()

    key_ts = {0, 150}
    for t in range(cfg.frames):
        n_top = cfg.n_start + t * cfg.n_step
        k_start = k_starts[t]
        img, _prof = build_values_invq_heatmap_and_profile(n_top=n_top, H=cfg.H, k_start=k_start, W=cfg.W)
        if perms_by_t is not None:
            img = img[:, perms_by_t[t]]
        im.set_data(img)
        ax.set_title(f"M38 best overlay ({cfg.sanity})  n_start={cfg.n_start}  q0={cfg.q0}  t={t}  k_start={k_start}")

        for pid, ln in enumerate(lines):
            xs: List[float] = []
            ys: List[float] = []
            for y in range(cfg.H):
                tt = t - y
                if tt < 0:
                    break
                k_hist = tracks.get(tt, {}).get(pid)
                if k_hist is None or k_hist < 0:
                    continue
                x = float(k_hist)
                if 0.0 <= x < float(cfg.W):
                    xs.append(x)
                    ys.append(float(y))
            if len(xs) < 2:
                ln.set_visible(False)
            else:
                ln.set_data(xs, ys)
                ln.set_visible(True)

        frame_path = frames_dir / f"frame_{t:04d}.png"
        fig.savefig(frame_path)
        if t in key_ts:
            shutil.copyfile(frame_path, out_dir / f"keyframe_t{t:03d}.png")

    plt.close(fig)

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; cannot render video")

    if format_ == "mp4":
        out_video = out_dir / f"{artifact_prefix}_best_overlay_tail.mp4"
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
        out_video = out_dir / f"{artifact_prefix}_best_overlay_tail.gif"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "scale=768:-1:flags=lanczos",
            str(out_video),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for p in frames_dir.glob("frame_*.png"):
        p.unlink()
    frames_dir.rmdir()
    return out_video


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def plot_visibility_heatmap(*, out_path: Path, results: List[RunResult], n_starts: List[int], q0s: List[int]) -> None:
    import matplotlib.pyplot as plt

    val_by_key: Dict[Tuple[int, int], float] = {(r.n_start, r.q0): float(r.visibility_score) for r in results}
    mat = np.full((len(q0s), len(n_starts)), np.nan, dtype=np.float64)
    for i, q0 in enumerate(q0s):
        for j, n0 in enumerate(n_starts):
            mat[i, j] = val_by_key.get((n0, q0), float("nan"))

    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    im = ax.imshow(mat, aspect="auto", origin="lower")
    ax.set_title("M38 auto-camera: visibility_score heatmap")
    ax.set_xlabel("n_start")
    ax.set_ylabel("q0")
    ax.set_xticks(list(range(len(n_starts))), labels=[n_tag(n0) for n0 in n_starts], rotation=0)
    ax.set_yticks(list(range(len(q0s))), labels=[str(q) for q in q0s])
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("visibility_score")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_q0_vs_nstart(*, out_path: Path, best_rows: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    xs = [int(r["n_start"]) for r in best_rows]
    ys = [int(r["best_q0"]) for r in best_rows]
    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150)
    ax.plot(xs, ys, marker="o", linewidth=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("n_start (log)")
    ax.set_ylabel("best q0")
    ax.set_title("M38 auto-camera: best q0 vs n_start")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_score_vs_q0(
    *,
    out_path: Path,
    results: List[RunResult],
    n_start: int,
    q0s: List[int],
    title_prefix: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = [r for r in results if int(r.n_start) == int(n_start)]
    score_by_q0 = {int(r.q0): float(r.visibility_score) for r in rows}
    r2_by_q0 = {int(r.q0): float(r.track_r2) for r in rows}
    xs = [int(q) for q in q0s]
    ys = [float(score_by_q0.get(int(q), float("nan"))) for q in xs]
    zs = [float(r2_by_q0.get(int(q), float("nan"))) for q in xs]

    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=150)
    ax.plot(xs, ys, marker="o", linewidth=1.5, markersize=3.0, label="visibility_score")
    ax.set_xlabel("q0")
    ax.set_ylabel("visibility_score")
    ax.set_title(f"{title_prefix} score vs q0 ({n_sci_tag(int(n_start))})")
    ax.grid(True, alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(xs, zs, color="#ff8800", alpha=0.65, linewidth=1.2, label="track_r2")
    ax2.set_ylabel("track_r2")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def render_preview_grid(parent_root: Path, n_starts: List[int]) -> Optional[Path]:
    sanity_root = parent_root / "sanity_permute_cols"
    if not sanity_root.exists():
        return None

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(len(n_starts), 2, figsize=(10.0, 3.2 * len(n_starts)), dpi=140)
    if len(n_starts) == 1:
        axes = np.array([axes])

    for i, n0 in enumerate(n_starts):
        tag = n_tag(n0)
        for col, label in enumerate(["real", "sanity"]):
            kf = parent_root / "best_overlay" / tag / label / "keyframe_t150.png"
            if not kf.exists():
                kf = parent_root / "best_overlay" / tag / label / "keyframe_t000.png"
            if not kf.exists():
                return None
            ax = axes[i, col]
            ax.imshow(mpimg.imread(kf))
            ax.set_axis_off()
            ax.set_title(f"{label} {tag}")

    fig.tight_layout()
    prefix = detect_artifact_prefix(parent_root)
    out_path = parent_root / f"{prefix}_preview_grid.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_table_tex(
    *,
    out_path: Path,
    best_rows: List[Dict[str, Any]],
    sanity_by_key: Dict[Tuple[int, int], RunResult],
) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$n_{\mathrm{start}}$ & $q_0$ & mean\_dx & $q_{\mathrm{eff}}$ & $R^2$ & score \\")
    lines.append(r"\hline")
    for r in best_rows:
        n0 = int(r["n_start"])
        q0 = int(r["best_q0"])
        mean_dx = float(r["mean_dx"])
        q_eff = float(r["q_eff"])
        r2 = float(r["track_r2"])
        score = float(r["visibility_score"])
        lines.append(
            f"{n0} & {q0} & {mean_dx:.3g} & {q_eff:.3g} & {r2:.3g} & {score:.3g} \\\\"
        )
        sanity = sanity_by_key.get((n0, q0))
        if sanity is not None:
            lines.append(
                f"\\multicolumn{{2}}{{r}}{{sanity}} & {sanity.mean_dx:.3g} & {sanity.q_eff:.3g} & {sanity.track_r2:.3g} & {sanity.visibility_score:.3g} \\\\"
            )
        lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-start-list", type=str, required=True)
    p.add_argument("--q0-grid", type=str, required=True, help="start,end,step (inclusive), e.g. 6,20,1")
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--dt", type=int, default=5)
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument("--peaks", type=int, default=3)
    p.add_argument("--conf-min", type=float, default=0.15)
    p.add_argument("--mode", type=str, default="values", choices=["values"])
    p.add_argument("--weights", type=str, default="invq", choices=["invq"])
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    p.add_argument("--sanity", type=str, default="none", choices=["none", "permute_cols"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    runs_root = out_root / "runs"
    ensure_dir(runs_root)

    artifact_prefix = detect_artifact_prefix(out_root)
    is_boundary_test = artifact_prefix == "m39"
    r2_guard = 0.95 if is_boundary_test else 0.9

    n_starts = parse_int_list(args.n_start_list)
    q0s = parse_q0_grid(args.q0_grid)

    seed = int(args.seed)

    if args.format == "mp4" and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; rerun with --format gif")

    results: List[RunResult] = []
    per_run_tracks: Dict[str, Dict[int, Dict[int, int]]] = {}
    per_run_kstarts: Dict[str, List[int]] = {}
    per_run_perms: Dict[str, Optional[List[np.ndarray]]] = {}

    # Evaluate all (n_start, q0).
    for n0 in n_starts:
        for q0 in q0s:
            cfg = RunConfig(
                n_start=n0,
                q0=q0,
                H=int(args.H),
                W=int(args.W),
                frames=int(args.frames),
                n_step=int(args.n_step),
                dt=int(args.dt),
                smooth=int(args.smooth),
                peaks=int(args.peaks),
                conf_min=float(args.conf_min),
                mode=args.mode,
                weights=args.weights,
                sanity=args.sanity,
            )
            rid = cfg.run_id()
            run_dir = runs_root / rid
            ensure_dir(run_dir)

            started = time.time()

            perms_by_t: Optional[List[np.ndarray]] = None
            if cfg.sanity == "permute_cols":
                perm_rng = np.random.default_rng(np.random.SeedSequence([seed, cfg.n_start, cfg.q0]))
                perms_by_t = [perm_rng.permutation(cfg.W) for _ in range(cfg.frames)]

            x_peaks0: List[int] = []
            z_series: List[float] = []
            k_starts: List[int] = []
            tracks: Dict[int, Dict[int, int]] = {}  # t -> peak_id -> x_index within window (0..W-1)

            for t in range(cfg.frames):
                n_top = cfg.n_start + t * cfg.n_step
                k_start, _k_end = k_window_for_frame(n_top, cfg.H, cfg.W, cfg.q0)
                k_starts.append(k_start)

                prof = build_values_invq_profile(n_top=n_top, H=cfg.H, k_start=k_start, W=cfg.W)
                if perms_by_t is not None:
                    prof = prof[perms_by_t[t]]
                prof_sm = rolling_mean(prof, cfg.smooth)

                idx, vals = top_peaks(prof_sm, cfg.peaks)
                peaks_here: Dict[int, int] = {}
                for pid, i in enumerate(idx[: cfg.peaks]):
                    peaks_here[pid] = int(i)
                tracks[t] = peaks_here

                x0 = peaks_here.get(0, -1)
                x_peaks0.append(x0)

                peak_val = float(vals[0]) if vals else float("nan")
                baseline = float(np.median(prof_sm))
                scale = mad(prof_sm)
                if scale <= 0:
                    scale = float(np.std(prof_sm))
                z = float((peak_val - baseline) / (scale + 1e-9)) if not math.isnan(peak_val) else float("nan")
                z_series.append(z)

            dxs: List[float] = []
            confs: List[float] = []
            for t in range(cfg.dt, cfg.frames):
                x_now = x_peaks0[t]
                x_prev = x_peaks0[t - cfg.dt]
                if x_now >= 0 and x_prev >= 0 and math.isfinite(z_series[t]) and math.isfinite(z_series[t - cfg.dt]):
                    k_now = k_starts[t] + int(x_now)
                    k_prev = k_starts[t - cfg.dt] + int(x_prev)
                    dxs.append(float(k_now - k_prev))
                    confs.append(float(min(z_series[t], z_series[t - cfg.dt])))
                else:
                    dxs.append(float("nan"))
                    confs.append(float("nan"))

            dx_arr = np.array(dxs, dtype=np.float64)
            conf_arr = np.array(confs, dtype=np.float64)
            valid_mask = np.isfinite(dx_arr) & np.isfinite(conf_arr) & (conf_arr >= float(cfg.conf_min))
            valid_frac = float(valid_mask.mean()) if valid_mask.size else 0.0
            mean_dx = float(np.mean(dx_arr[valid_mask])) if valid_mask.any() else float("nan")
            median_dx = float(np.median(dx_arr[valid_mask])) if valid_mask.any() else float("nan")
            q_eff = float(cfg.dt) / mean_dx if (not math.isnan(mean_dx) and mean_dx > 0) else float("nan")

            t_idx = np.arange(cfg.dt, cfg.frames, dtype=np.float64)
            k_arr = np.array(
                [(k_starts[t] + int(x_peaks0[t])) if x_peaks0[t] >= 0 else float("nan") for t in range(cfg.dt, cfg.frames)],
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

            wave_strength = float(np.nanmedian(np.array(z_series, dtype=np.float64)))
            ws = wave_strength if not math.isnan(wave_strength) else 0.0
            r2v = track_r2 if not math.isnan(track_r2) else 0.0
            vis_score = float(ws) * math.sqrt(max(r2v, 0.0)) * math.sqrt(max(valid_frac, 0.0))

            runtime_s = float(time.time() - started)

            if not is_boundary_test:
                # keyframes (useful for grid previews; skipped for boundary sweeps to keep output compact)
                t_key = min(150, cfg.frames - 1)
                for t_k in [0, t_key]:
                    n_top = cfg.n_start + t_k * cfg.n_step
                    k_start = k_starts[t_k]
                    img, _prof = build_values_invq_heatmap_and_profile(n_top=n_top, H=cfg.H, k_start=k_start, W=cfg.W)
                    if perms_by_t is not None:
                        img = img[:, perms_by_t[t_k]]
                    out_path = run_dir / f"keyframe_t{t_k:03d}.png"
                    render_keyframe(
                        out_path=out_path,
                        cfg=cfg,
                        img=img,
                        k_start=k_start,
                        tracks=tracks,
                        t_key=t_k,
                        mean_dx=mean_dx,
                        q_eff=q_eff,
                    )

            res = RunResult(
                run_id=rid,
                n_start=cfg.n_start,
                q0=cfg.q0,
                H=cfg.H,
                W=cfg.W,
                weights=cfg.weights,
                dt=cfg.dt,
                smooth=cfg.smooth,
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
                visibility_score=vis_score,
                notes="",
            )
            (run_dir / "run_summary.json").write_text(
                json.dumps({"config": asdict(cfg), "result": asdict(res)}, indent=2), encoding="utf-8"
            )

            results.append(res)
            per_run_tracks[rid] = tracks
            per_run_kstarts[rid] = k_starts
            per_run_perms[rid] = perms_by_t

    # Write per-run sweep CSV in the invoked out_dir.
    sweep_header = list(asdict(results[0]).keys()) if results else []
    write_csv(out_root / f"{artifact_prefix}_q0_sweep.csv", [asdict(r) for r in results], sweep_header)
    (out_root / f"{artifact_prefix}_run.json").write_text(
        json.dumps({"params": vars(args), "git_sha": git_sha(), "n_runs": len(results)}, indent=2), encoding="utf-8"
    )

    parent_root = out_root if out_root.name != "sanity_permute_cols" else out_root.parent

    if args.sanity == "none":
        # Select best q0 per n_start (guard: track_r2 >= r2_guard).
        best_rows: List[Dict[str, Any]] = []
        for n0 in n_starts:
            cand = [r for r in results if r.n_start == n0 and math.isfinite(r.track_r2) and r.track_r2 >= r2_guard]
            notes = ""
            if cand:
                best = max(cand, key=lambda r: r.visibility_score)
            else:
                cand2 = [r for r in results if r.n_start == n0]
                best = max(cand2, key=lambda r: r.visibility_score)
                notes = f"no_candidate_passed_r2_guard_{r2_guard:.2f}"
            best_rows.append(
                {
                    "n_start": int(best.n_start),
                    "best_q0": int(best.q0),
                    "best_run_id": best.run_id,
                    "visibility_score": float(best.visibility_score),
                    "track_r2": float(best.track_r2),
                    "mean_dx": float(best.mean_dx),
                    "q_eff": float(best.q_eff),
                    "valid_frac": float(best.valid_frac),
                    "wave_strength": float(best.wave_strength),
                    "notes": notes,
                }
            )

        best_rows = sorted(best_rows, key=lambda r: int(r["n_start"]))
        best_csv = parent_root / f"{artifact_prefix}_best_by_nstart.csv"
        write_csv(
            best_csv,
            best_rows,
            [
                "n_start",
                "best_q0",
                "best_run_id",
                "visibility_score",
                "track_r2",
                "mean_dx",
                "q_eff",
                "valid_frac",
                "wave_strength",
                "notes",
            ],
        )

        # Plots (from real sweep).
        plot_visibility_heatmap(
            out_path=parent_root / f"{artifact_prefix}_visibility_heatmap.png",
            results=results,
            n_starts=n_starts,
            q0s=q0s,
        )
        plot_q0_vs_nstart(out_path=parent_root / f"{artifact_prefix}_q0_vs_nstart.png", best_rows=best_rows)
        if is_boundary_test:
            for n0 in n_starts:
                plot_score_vs_q0(
                    out_path=parent_root / f"{artifact_prefix}_score_vs_q0_{n_sci_tag(n0)}.png",
                    results=results,
                    n_start=n0,
                    q0s=q0s,
                    title_prefix=artifact_prefix,
                )

        # Render best overlay per n_start (real).
        overlay_targets = [max(n_starts)] if is_boundary_test else [int(r["n_start"]) for r in best_rows]
        for r in best_rows:
            n0 = int(r["n_start"])
            if n0 not in overlay_targets:
                continue
            q0 = int(r["best_q0"])
            rid_prefix = f"{n_tag(n0)}_q{q0}_"
            rid = next((res.run_id for res in results if res.run_id.startswith(rid_prefix)), None)
            if rid is None:
                continue
            out_dir = (
                (parent_root / "best_overlay" / "real")
                if is_boundary_test
                else (parent_root / "best_overlay" / n_tag(n0) / "real")
            )
            cfg = RunConfig(
                n_start=n0,
                q0=q0,
                H=int(args.H),
                W=int(args.W),
                frames=int(args.frames),
                n_step=int(args.n_step),
                dt=int(args.dt),
                smooth=int(args.smooth),
                peaks=int(args.peaks),
                conf_min=float(args.conf_min),
                mode=args.mode,
                weights=args.weights,
                sanity="none",
            )
            render_best_overlay_video(
                out_dir=out_dir,
                artifact_prefix=artifact_prefix,
                cfg=cfg,
                tracks=per_run_tracks[rid],
                k_starts=per_run_kstarts[rid],
                perms_by_t=per_run_perms[rid],
                fps=int(args.fps),
                format_=args.format,
            )

    else:
        # sanity: reuse best q0 per n_start from parent real selection, to get apples-to-apples overlays.
        best_csv = parent_root / f"{artifact_prefix}_best_by_nstart.csv"
        if best_csv.exists():
            best_rows = load_csv_rows(best_csv)
            # map (n_start, q0) -> RunResult for sanity table, and render overlays.
            res_by_key: Dict[Tuple[int, int], RunResult] = {(r.n_start, r.q0): r for r in results}
            sanity_for_table: Dict[Tuple[int, int], RunResult] = {}

            overlay_targets = [max(n_starts)] if is_boundary_test else [int(br["n_start"]) for br in best_rows]
            for br in best_rows:
                n0 = int(br["n_start"])
                if n0 not in overlay_targets:
                    continue
                q0 = int(br["best_q0"])
                rid_prefix = f"{n_tag(n0)}_q{q0}_"
                rid = next((res.run_id for res in results if res.run_id.startswith(rid_prefix)), None)
                if rid is None:
                    continue
                out_dir = (
                    (parent_root / "best_overlay" / "sanity")
                    if is_boundary_test
                    else (parent_root / "best_overlay" / n_tag(n0) / "sanity")
                )
                cfg = RunConfig(
                    n_start=n0,
                    q0=q0,
                    H=int(args.H),
                    W=int(args.W),
                    frames=int(args.frames),
                    n_step=int(args.n_step),
                    dt=int(args.dt),
                    smooth=int(args.smooth),
                    peaks=int(args.peaks),
                    conf_min=float(args.conf_min),
                    mode=args.mode,
                    weights=args.weights,
                    sanity="permute_cols",
                )
                render_best_overlay_video(
                    out_dir=out_dir,
                    artifact_prefix=artifact_prefix,
                    cfg=cfg,
                    tracks=per_run_tracks[rid],
                    k_starts=per_run_kstarts[rid],
                    perms_by_t=per_run_perms[rid],
                    fps=int(args.fps),
                    format_=args.format,
                )
                rres = res_by_key.get((n0, q0))
                if rres is not None:
                    sanity_for_table[(n0, q0)] = rres

            # Build combined preview grid + TeX table (real + sanity rows).
            if not is_boundary_test:
                render_preview_grid(parent_root, n_starts=n_starts)
            best_rows_any: List[Dict[str, Any]] = []
            for br in best_rows:
                # Keep only the numeric columns required by build_table_tex.
                best_rows_any.append(
                    {
                        "n_start": int(br["n_start"]),
                        "best_q0": int(br["best_q0"]),
                        "mean_dx": float(br["mean_dx"]),
                        "q_eff": float(br["q_eff"]),
                        "track_r2": float(br["track_r2"]),
                        "visibility_score": float(br["visibility_score"]),
                    }
                )
            build_table_tex(
                out_path=parent_root / f"{artifact_prefix}_table.tex",
                best_rows=best_rows_any,
                sanity_by_key=sanity_for_table,
            )

            # Final summary with boundary notes (written to parent root).
            q0_max = max(int(q) for q in q0s)
            best_rows_out: List[Dict[str, Any]] = []
            for br in best_rows:
                n0 = int(br["n_start"])
                q0 = int(br["best_q0"])
                hit_upper = bool(is_boundary_test and q0 == q0_max and float(br.get("track_r2", "nan")) >= r2_guard)
                best_rows_out.append(
                    {
                        "n_start": n0,
                        "best_q0": q0,
                        "visibility_score": float(br.get("visibility_score", "nan")),
                        "track_r2": float(br.get("track_r2", "nan")),
                        "mean_dx": float(br.get("mean_dx", "nan")),
                        "q_eff": float(br.get("q_eff", "nan")),
                        "hit_upper_boundary": hit_upper,
                        "notes": br.get("notes", ""),
                    }
                )
            (parent_root / f"{artifact_prefix}_summary.json").write_text(
                json.dumps(
                    {
                        "artifact_prefix": artifact_prefix,
                        "params": vars(args),
                        "q0_grid": q0s,
                        "r2_guard": r2_guard,
                        "boundary_test": is_boundary_test,
                        "best_by_nstart": best_rows_out,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

    # Manifests
    write_manifest(
        out_root,
        params={**vars(args), "artifact_prefix": artifact_prefix, "runtime_note": "per-run runtimes in runs/*/run_summary.json"},
        name=f"{artifact_prefix}_manifest.json",
    )
    if out_root.name == "sanity_permute_cols":
        write_manifest(
            parent_root,
            params={"artifact_prefix": artifact_prefix, "note": "parent manifest includes real + sanity outputs"},
            name=f"{artifact_prefix}_manifest.json",
        )

    print(f"OK: wrote {artifact_prefix} outputs to {out_root}")


if __name__ == "__main__":
    main()

