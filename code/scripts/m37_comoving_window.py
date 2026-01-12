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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def k_window_for_frame(n_top: int, H: int, W: int, q0: int) -> Tuple[int, int]:
    n_mid = n_top + (H // 2)
    k_c = int(n_mid // q0)
    k_start = int(k_c - (W // 2))
    if k_start < 1:
        k_start = 1
    k_end = int(k_start + W - 1)
    return k_start, k_end


def build_values_invq_heatmap_and_profile(
    *,
    n_top: int,
    H: int,
    k_start: int,
    W: int,
) -> Tuple[np.ndarray, np.ndarray]:
    # returns (heatmap HxW, profile length W)
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
        n = self.n_start
        n_tag = f"n{n//1_000_000}e6" if n % 1_000_000 == 0 else f"n{n}"
        return f"{n_tag}_q{self.q0}_H{self.H}_W{self.W}_{self.weights}_dt{self.dt}_sm{self.smooth}_{self.sanity}"


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
    ax.set_xlabel(f"k index (offset from k_start={k_start})")
    ax.set_ylabel("rows (n increasing down)")
    ax.set_title(f"M37 comoving window ({cfg.sanity})  n_start={cfg.n_start}  q0={cfg.q0}  t={t_key}")

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
            if k_hist is None or k_hist <= 0:
                continue
            x = float(k_hist - k_start)
            if 0.0 <= x < float(W):
                xs.append(x)
                ys.append(float(y))
        if len(xs) >= 2:
            ax.plot(xs, ys, color=colors[pid], linewidth=1.6, alpha=0.95, label=f"peak{pid+1}")
            any_line = True

    qeff_s = f"{q_eff:.3g}" if (not math.isnan(q_eff)) else "nan"
    txt = f"W={cfg.W}  H={cfg.H}  dt={cfg.dt}  mean_dx={mean_dx:.3g}  q_eff≈{qeff_s}"
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
    cfg: RunConfig,
    tracks: Dict[int, Dict[int, int]],
    k_starts: List[int],
    perm: Optional[np.ndarray],
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
    lines = [ax.plot([], [], color=colors[i], linewidth=1.6, alpha=0.95, label=f"peak{i+1}")[0] for i in range(min(cfg.peaks, 3))]

    n_top0 = cfg.n_start
    k_start0, _ = k_window_for_frame(n_top0, cfg.H, cfg.W, cfg.q0)
    img0, _prof0 = build_values_invq_heatmap_and_profile(n_top=n_top0, H=cfg.H, k_start=k_start0, W=cfg.W)
    if perm is not None:
        img0 = img0[:, perm]
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
        if perm is not None:
            img = img[:, perm]
        im.set_data(img)
        ax.set_title(f"M37 best overlay ({cfg.sanity})  n_start={cfg.n_start}  q0={cfg.q0}  t={t}  k_start={k_start}")

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
                x = float(k_hist - k_start)
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
        out_video = out_dir / "m37_best_overlay_tail.mp4"
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
        out_video = out_dir / "m37_best_overlay_tail.gif"
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


def maybe_make_preview_grid(parent_root: Path, n_starts: List[int], q0_list: List[int], *, t_key: int) -> Optional[Path]:
    real_root = parent_root
    sanity_root = parent_root / "sanity_permute_cols"
    if not sanity_root.exists():
        return None

    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    nrows = len(n_starts)
    ncols = len(q0_list) * 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows), dpi=140)
    if nrows == 1:
        axes = np.array([axes])

    for i, n0 in enumerate(n_starts):
        for j, q0 in enumerate(q0_list):
            rid_real = RunConfig(
                n_start=n0,
                q0=q0,
                H=0,
                W=0,
                frames=0,
                n_step=0,
                dt=0,
                smooth=0,
                peaks=0,
                conf_min=0.0,
                mode="values",
                weights="invq",
                sanity="none",
            ).run_id()
            # Replace placeholders with actual prefixes used in run_id; we only need n/q tags.
            rid_real = f"{('n'+str(n0//1_000_000)+'e6') if n0 % 1_000_000 == 0 else ('n'+str(n0))}_q{q0}"

            # run_id prefix matching in our actual dirs:
            def find_run_dir(root: Path, sanity: str) -> Optional[Path]:
                runs = root / ("runs" if sanity == "none" else "runs")
                if not runs.exists():
                    return None
                for d in runs.iterdir():
                    if not d.is_dir():
                        continue
                    if d.name.startswith(rid_real) and d.name.endswith(f"_{sanity}" if sanity != "none" else "_none"):
                        return d
                # fallback: prefix only
                for d in runs.iterdir():
                    if d.is_dir() and d.name.startswith(rid_real):
                        return d
                return None

            d_real = find_run_dir(real_root, "none")
            d_sanity = find_run_dir(sanity_root, "permute_cols")
            if d_real is None or d_sanity is None:
                return None

            for col_off, (label, d) in enumerate([("real", d_real), ("sanity", d_sanity)]):
                kf = d / f"keyframe_t{t_key:03d}.png"
                if not kf.exists():
                    kf = d / "keyframe_t000.png"
                if not kf.exists():
                    return None
                ax = axes[i, 2 * j + col_off]
                ax.imshow(mpimg.imread(kf))
                ax.set_axis_off()
                ax.set_title(f"{label} q0={q0}  n={n0}")

    fig.tight_layout()
    out_path = parent_root / "m37_preview_grid.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-start-list", type=str, required=True)
    p.add_argument("--q0-list", type=str, required=True)
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


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def main() -> None:
    args = parse_args()
    out_root = Path(args.out_dir)
    ensure_dir(out_root)
    runs_root = out_root / "runs"
    ensure_dir(runs_root)

    n_starts = parse_int_list(args.n_start_list)
    q0_list = parse_int_list(args.q0_list)

    seed = int(args.seed)
    rng = np.random.default_rng(seed)

    if args.format == "mp4" and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; rerun with --format gif")

    results: List[RunResult] = []
    per_run_tracks: Dict[str, Dict[int, Dict[int, int]]] = {}
    per_run_kstarts: Dict[str, List[int]] = {}
    per_run_perm: Dict[str, Optional[np.ndarray]] = {}

    for n0 in n_starts:
        for q0 in q0_list:
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
            out_dir = runs_root / rid
            ensure_dir(out_dir)

            started = time.time()

            perm: Optional[np.ndarray] = None
            if cfg.sanity == "permute_cols":
                perm = rng.permutation(cfg.W)

            k_peaks0: List[int] = []
            z_series: List[float] = []
            k_starts: List[int] = []
            tracks: Dict[int, Dict[int, int]] = {}

            for t in range(cfg.frames):
                n_top = cfg.n_start + t * cfg.n_step
                k_start, _k_end = k_window_for_frame(n_top, cfg.H, cfg.W, cfg.q0)
                k_starts.append(k_start)

                prof = build_values_invq_profile(n_top=n_top, H=cfg.H, k_start=k_start, W=cfg.W)
                if perm is not None:
                    prof = prof[perm]
                prof_sm = rolling_mean(prof, cfg.smooth)

                idx, vals = top_peaks(prof_sm, cfg.peaks)
                # map peak indices back to absolute k, accounting for permutation
                peaks_here: Dict[int, int] = {}
                for pid, i in enumerate(idx[: cfg.peaks]):
                    if perm is None:
                        k_abs = k_start + int(i)
                    else:
                        k_abs = k_start + int(perm[int(i)])
                    peaks_here[pid] = int(k_abs)
                tracks[t] = peaks_here

                k0 = peaks_here.get(0, 0)
                k_peaks0.append(k0)

                peak_val = float(vals[0]) if vals else float("nan")
                baseline = float(np.median(prof_sm))
                scale = mad(prof_sm)
                if scale <= 0:
                    scale = float(np.std(prof_sm))
                z = float((peak_val - baseline) / (scale + 1e-9)) if not math.isnan(peak_val) else float("nan")
                z_series.append(z)

            # drift from peak0 dk over dt
            dxs: List[float] = []
            confs: List[float] = []
            for t in range(cfg.dt, cfg.frames):
                k_now = k_peaks0[t]
                k_prev = k_peaks0[t - cfg.dt]
                if k_now > 0 and k_prev > 0 and math.isfinite(z_series[t]) and math.isfinite(z_series[t - cfg.dt]):
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
            k_arr = np.array(k_peaks0[cfg.dt :], dtype=np.float64)
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

            # keyframes (tail overlay) for preview grid
            t_key = min(150, cfg.frames - 1)
            for t_k in [0, t_key]:
                n_top = cfg.n_start + t_k * cfg.n_step
                k_start = k_starts[t_k]
                img, _prof = build_values_invq_heatmap_and_profile(n_top=n_top, H=cfg.H, k_start=k_start, W=cfg.W)
                if perm is not None:
                    img = img[:, perm]
                out_path = out_dir / f"keyframe_t{t_k:03d}.png"
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
            (out_dir / "run_summary.json").write_text(
                json.dumps({"config": asdict(cfg), "result": asdict(res)}, indent=2), encoding="utf-8"
            )

            results.append(res)
            per_run_tracks[rid] = tracks
            per_run_kstarts[rid] = k_starts
            per_run_perm[rid] = perm

    # write summary
    header = list(asdict(results[0]).keys())
    write_csv(out_root / "m37_summary.csv", [asdict(r) for r in results], header)
    (out_root / "m37_summary.json").write_text(
        json.dumps({"params": vars(args), "git_sha": git_sha(), "n_runs": len(results)}, indent=2), encoding="utf-8"
    )

    parent_root = out_root if out_root.name != "sanity_permute_cols" else out_root.parent

    # best-of overlay: real picks best; sanity reuses best config from parent.
    best_cfg_path = parent_root / "m37_best_config.json"
    if args.sanity == "none":
        # pick best for largest n_start among this run
        max_n = max(n_starts)
        cand = [r for r in results if r.n_start == max_n]
        best = max(cand, key=lambda r: r.visibility_score)
        best_cfg = {"best_run_id": best.run_id, "visibility_score": best.visibility_score, "n_start": best.n_start, "q0": best.q0}
        best_cfg_path.write_text(json.dumps(best_cfg, indent=2), encoding="utf-8")

        best_dir = parent_root / "best_overlay" / "real"
        best_dir.mkdir(parents=True, exist_ok=True)
        tracks = per_run_tracks[best.run_id]
        k_starts = per_run_kstarts[best.run_id]
        perm = per_run_perm[best.run_id]
        cfg = RunConfig(
            n_start=best.n_start,
            q0=best.q0,
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
            out_dir=best_dir,
            cfg=cfg,
            tracks=tracks,
            k_starts=k_starts,
            perm=perm,
            fps=int(args.fps),
            format_=args.format,
        )

    else:
        if best_cfg_path.exists():
            best_cfg = json.loads(best_cfg_path.read_text(encoding="utf-8"))
            best_n = int(best_cfg["n_start"])
            best_q0 = int(best_cfg["q0"])
            best_run_id_prefix = f"{('n'+str(best_n//1_000_000)+'e6') if best_n % 1_000_000 == 0 else ('n'+str(best_n))}_q{best_q0}"
            best_run_id = None
            for r in results:
                if r.run_id.startswith(best_run_id_prefix):
                    best_run_id = r.run_id
                    break
            if best_run_id is not None:
                best_dir = parent_root / "best_overlay" / "sanity"
                best_dir.mkdir(parents=True, exist_ok=True)
                tracks = per_run_tracks[best_run_id]
                k_starts = per_run_kstarts[best_run_id]
                perm = per_run_perm[best_run_id]
                cfg = RunConfig(
                    n_start=best_n,
                    q0=best_q0,
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
                    out_dir=best_dir,
                    cfg=cfg,
                    tracks=tracks,
                    k_starts=k_starts,
                    perm=perm,
                    fps=int(args.fps),
                    format_=args.format,
                )

        # build preview grid if real exists
        maybe_make_preview_grid(parent_root, n_starts=n_starts, q0_list=q0_list, t_key=min(150, int(args.frames) - 1))

    # manifests
    write_manifest(out_root, params={**vars(args), "runtime_note": "per-run runtimes in runs/*/run_summary.json"}, name="m37_manifest.json")
    if out_root.name == "sanity_permute_cols":
        # update parent manifest to include both real+sanity+preview
        write_manifest(parent_root, params={"note": "parent manifest includes real + sanity outputs"}, name="m37_manifest.json")

    print(f"OK: wrote M37 outputs to {out_root}")


if __name__ == "__main__":
    main()
