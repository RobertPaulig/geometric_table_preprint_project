#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m33-dir", type=str, required=True)
    p.add_argument("--n-start-list", type=str, required=True)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--K", type=int, default=512)
    p.add_argument("--dt", type=int, default=5)
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument("--mode", type=str, default="values", choices=["values", "occupancy"])
    p.add_argument("--weights", type=str, default="invq", choices=["invq"])
    p.add_argument("--peaks", type=int, default=3)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--format", type=str, default="mp4", choices=["mp4", "gif"])
    p.add_argument("--sanity", type=str, default="none", choices=["none", "permute_cols"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


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


def write_manifest(out_dir: Path, params: Dict[str, object]) -> None:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != "m34_manifest.json"])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / "m34_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_m33_tracks(m33_dir: Path, sanity: str) -> Dict[int, Dict[int, Dict[int, int]]]:
    # returns: n_start -> t -> peak_id -> k_peak
    tracks_path = m33_dir / "m33_wavefront_tracks.csv"
    if sanity == "permute_cols":
        tracks_path = m33_dir / "sanity_permute_cols" / "m33_wavefront_tracks.csv"
    out: Dict[int, Dict[int, Dict[int, int]]] = {}
    with tracks_path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_start = int(row["n_start"])
            t = int(row["t"])
            peak_id = int(row["peak_id"])
            k_peak = int(row["k_peak"])
            out.setdefault(n_start, {}).setdefault(t, {})[peak_id] = k_peak
    return out


def load_m33_summary_by_nstart(m33_dir: Path, sanity: str) -> Dict[int, Dict[str, float]]:
    p = m33_dir / "m33_summary_by_nstart.csv"
    if sanity == "permute_cols":
        p = m33_dir / "sanity_permute_cols" / "m33_summary_by_nstart.csv"
    out: Dict[int, Dict[str, float]] = {}
    with p.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            n_start = int(row["n_start"])
            out[n_start] = {
                "mean_dx": float(row["mean_dx"]) if row["mean_dx"] else float("nan"),
                "median_dx": float(row["median_dx"]) if row["median_dx"] else float("nan"),
                "slope_kpeak": float(row["slope_kpeak"]) if row["slope_kpeak"] else float("nan"),
                "valid_frac": float(row["valid_frac"]) if row["valid_frac"] else float("nan"),
            }
    return out


def values_heatmap(n_top: int, H: int, K: int) -> np.ndarray:
    img = np.zeros((H, K), dtype=np.float32)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        for n in range(first, n_end, k):
            q = n // k
            img[n - n_top, k - 1] = 1.0 / float(q)
    return img


def occupancy_heatmap(n_top: int, H: int, K: int) -> np.ndarray:
    img = np.zeros((H, K), dtype=np.uint8)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        for n in range(first, n_end, k):
            img[n - n_top, k - 1] = 1
    return img.astype(np.float32)


def build_heatmap(mode: str, n_top: int, H: int, K: int) -> np.ndarray:
    if mode == "values":
        return values_heatmap(n_top, H, K)
    if mode == "occupancy":
        return occupancy_heatmap(n_top, H, K)
    raise ValueError(mode)


def make_perm(K: int, seed: int, n_start: int) -> np.ndarray:
    rng = np.random.default_rng(seed + (n_start % 1000003))
    return rng.permutation(K)


def render_video_for_nstart(
    *,
    n_start: int,
    tracks: Dict[int, Dict[int, int]],
    summary: Dict[str, float],
    args: argparse.Namespace,
    out_dir: Path,
) -> Tuple[Path, List[Path]]:
    import matplotlib.pyplot as plt

    frames_dir = out_dir / "frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    perm = None
    if args.sanity == "permute_cols":
        perm = make_perm(args.K, args.seed, n_start)

    keyframes: List[Path] = []
    key_t = {0, args.frames // 2}

    vmax = None
    if args.mode == "values":
        vmax = 0.02

    fig, ax = plt.subplots(figsize=(6.2, 6.2), dpi=110)
    ax.set_xlabel("k (columns)")
    ax.set_ylabel("rows (n increasing down)")
    colors = ["#00d4ff", "#00ff6a", "#ffd200"]
    vlines = [
        ax.axvline(
            x=0,
            color=colors[i],
            linewidth=1.2,
            alpha=0.9,
            visible=False,
            label=f"peak{i+1}",
        )
        for i in range(min(args.peaks, 3))
    ]

    mean_dx = float(summary.get("mean_dx", float("nan")))
    slope = float(summary.get("slope_kpeak", float("nan")))
    q_eff = float("nan")
    if mean_dx > 0:
        q_eff = float(args.dt) / float(mean_dx)
    txt = f"dt={args.dt}  mean_dx={mean_dx:.3g}  slope={slope:.3g}  q_eff≈{q_eff:.3g}"
    text_box = ax.text(
        0.01,
        0.01,
        txt,
        transform=ax.transAxes,
        fontsize=8.5,
        color="white",
        bbox=dict(facecolor="black", alpha=0.45, pad=3),
    )

    # init with first frame
    img0 = build_heatmap(args.mode, n_start, args.H, args.K)
    if perm is not None:
        img0 = img0[:, perm]
    im = ax.imshow(img0, aspect="auto", origin="upper", cmap="magma", vmin=0.0, vmax=vmax)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    for t in range(args.frames):
        n_top = n_start + t * args.n_step
        img = build_heatmap(args.mode, n_top, args.H, args.K)
        if perm is not None:
            img = img[:, perm]
        im.set_data(img)
        ax.set_title(f"M34 overlay ({args.sanity})  n_start={n_start}  t={t}")

        peaks_here = tracks.get(t, {})
        for pid, vl in enumerate(vlines):
            k_peak = peaks_here.get(pid)
            if k_peak is None or k_peak <= 0:
                vl.set_visible(False)
                continue
            vl.set_xdata([k_peak - 1, k_peak - 1])
            vl.set_visible(True)

        frame_path = frames_dir / f"frame_{t:04d}.png"
        fig.savefig(frame_path)

        if t in key_t:
            kf = out_dir / f"keyframe_t{t:03d}.png"
            shutil.copyfile(frame_path, kf)
            keyframes.append(kf)

    plt.close(fig)

    # Encode video
    if args.format == "mp4":
        out_video = out_dir / "m34_overlay.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
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
        # gif via ffmpeg as well (smaller codepath)
        out_video = out_dir / "m34_overlay.gif"
        cmd = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-vf",
            "scale=512:-1:flags=lanczos",
            str(out_video),
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # remove frames to keep repo smaller; keyframes stay
    for p in frames_dir.glob("frame_*.png"):
        p.unlink()
    frames_dir.rmdir()

    return out_video, keyframes


def make_preview_grid(
    *,
    out_root: Path,
    n_starts: List[int],
    real_dirs: Dict[int, Path],
    sanity_dirs: Dict[int, Path],
) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(len(n_starts), 2, figsize=(9.0, 3.0 * len(n_starts)), dpi=140)
    if len(n_starts) == 1:
        axes = np.array([axes])
    for i, n0 in enumerate(n_starts):
        for j, (label, d) in enumerate([("real", real_dirs[n0]), ("sanity", sanity_dirs[n0])]):
            kf = d / "keyframe_t150.png"
            if not kf.exists():
                kf = d / "keyframe_t000.png"
            img = mpimg.imread(kf)
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(f"{label} n_start={n0}")
    fig.tight_layout()
    out_path = out_root / "m34_preview_grid.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def maybe_make_preview_grid(parent_dir: Path, n_starts: List[int]) -> Path | None:
    real_root = parent_dir / "real"
    sanity_root = parent_dir / "sanity_permute_cols"
    if not real_root.exists() or not sanity_root.exists():
        return None

    real_dirs = {n0: (real_root / f"n{n0//1000000}e6") for n0 in n_starts}
    sanity_dirs = {n0: (sanity_root / f"n{n0//1000000}e6") for n0 in n_starts}

    def has_any_keyframe(d: Path) -> bool:
        return (d / "keyframe_t150.png").exists() or (d / "keyframe_t000.png").exists()

    if not all(d.exists() and has_any_keyframe(d) for d in real_dirs.values()):
        return None
    if not all(d.exists() and has_any_keyframe(d) for d in sanity_dirs.values()):
        return None

    return make_preview_grid(out_root=parent_dir, n_starts=n_starts, real_dirs=real_dirs, sanity_dirs=sanity_dirs)


def main() -> None:
    args = parse_args()
    m33_dir = Path(args.m33_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    n_starts = parse_int_list(args.n_start_list)
    tracks_by_n = load_m33_tracks(m33_dir, args.sanity)
    summary_by_n = load_m33_summary_by_nstart(m33_dir, args.sanity)

    if args.format == "mp4" and shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found; rerun with --format gif")

    started = time.time()

    videos: Dict[int, Path] = {}
    keyframes_all: List[Path] = []
    per_n_dirs: Dict[int, Path] = {}

    for n0 in n_starts:
        sub = f"n{n0//1000000}e6" if n0 % 1000000 == 0 else f"n{n0}"
        out_dir = out_root / sub
        out_dir.mkdir(parents=True, exist_ok=True)
        per_n_dirs[n0] = out_dir

        v, kfs = render_video_for_nstart(
            n_start=n0,
            tracks=tracks_by_n.get(n0, {}),
            summary=summary_by_n.get(n0, {}),
            args=args,
            out_dir=out_dir,
        )
        videos[n0] = v
        keyframes_all.extend(kfs)

    runtime_s = float(time.time() - started)

    summary = {
        "params": {
            "mode": args.mode,
            "weights": args.weights,
            "n_start_list": n_starts,
            "frames": args.frames,
            "n_step": args.n_step,
            "H": args.H,
            "K": args.K,
            "dt": args.dt,
            "smooth": args.smooth,
            "peaks": args.peaks,
            "fps": args.fps,
            "format": args.format,
            "sanity": args.sanity,
            "seed": args.seed,
        },
        "m33_inputs": {
            "dir": str(m33_dir).replace("\\", "/"),
            "tracks_csv": str((m33_dir / ("sanity_permute_cols/m33_wavefront_tracks.csv" if args.sanity == "permute_cols" else "m33_wavefront_tracks.csv"))).replace("\\", "/"),
            "summary_csv": str((m33_dir / ("sanity_permute_cols/m33_summary_by_nstart.csv" if args.sanity == "permute_cols" else "m33_summary_by_nstart.csv"))).replace("\\", "/"),
        },
        "runtime_seconds": runtime_s,
        "videos": {str(n0): str(videos[n0].relative_to(out_root)).replace("\\", "/") for n0 in n_starts},
    }
    (out_root / "m34_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    preview_path = maybe_make_preview_grid(out_root.parent, n_starts)

    write_manifest(out_root, params={**summary["params"], "runtime_seconds": runtime_s})
    if preview_path is not None:
        # update parent manifest to include preview
        write_manifest(out_root.parent, params={"note": "parent manifest includes preview grid across real vs sanity"})

    print(f"OK: wrote M34 overlay to {out_root}")


if __name__ == "__main__":
    main()
