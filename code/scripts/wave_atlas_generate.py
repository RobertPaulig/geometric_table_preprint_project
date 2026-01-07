#!/usr/bin/env python
# code/scripts/wave_atlas_generate.py
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import List

import numpy as np

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise SystemExit("imageio is required for PNG/GIF output") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=3000)
    p.add_argument("--K", type=int, default=120)
    p.add_argument("--H", type=int, default=220)
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--out-dir", type=str, default="out/wave_atlas")
    p.add_argument("--diag-N", type=int, action="append", default=[60, 420, 2520, 27720])
    p.add_argument("--no-gif", action="store_true", default=False)
    return p.parse_args()


def build_occupancy(N: int, K: int) -> np.ndarray:
    occ = np.zeros((N, K), dtype=np.uint8)
    for k in range(1, K + 1):
        occ[k - 1 :: k, k - 1] = 1
    return occ


def build_values_log(N: int, K: int) -> np.ndarray:
    vals = np.zeros((N, K), dtype=np.float32)
    for k in range(1, K + 1):
        idx = np.arange(k - 1, N, k)
        n_vals = idx + 1
        q = n_vals // k
        vals[idx, k - 1] = np.log1p(q)
    return vals


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr - vmin) / (vmax - vmin)
    return (scaled * 255).astype(np.uint8)


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, img)


def save_scroll_gif(path: Path, occ: np.ndarray, H: int, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, mode="I", duration=0.2) as w:
        for start in range(0, max(1, occ.shape[0] - H + 1), step):
            frame = occ[start : start + H, :]
            w.append_data(frame * 255)


def diagonal_hits(N: int, K: int) -> List[int]:
    ks = []
    for k in range(1, K + 1):
        if N % k == 0:
            ks.append(k)
    return ks


def save_diagonal_plot(out_dir: Path, N: int, K: int) -> None:
    ks = diagonal_hits(N, K)
    vals = [N // k - 1 for k in ks]
    csv_path = out_dir / f"diagonal_hits_N{N}_K{K}.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "N_over_k_minus_1"])
        for k, v in zip(ks, vals):
            w.writerow([k, v])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(ks, vals, marker="o", linewidth=1.0, markersize=3)
    ax.set_title(f"Diagonal hits: N={N}, K={K}")
    ax.set_xlabel("k")
    ax.set_ylabel("N/k - 1")
    fig.tight_layout()
    fig.savefig(out_dir / f"diagonal_hits_N{N}_K{K}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    occ = build_occupancy(args.N, args.K)
    occ_img = (occ * 255).astype(np.uint8)
    save_png(out_dir / f"divisibility_occ_N{args.N}_K{args.K}.png", occ_img)

    if not args.no_gif:
        save_scroll_gif(
            out_dir / f"divisibility_occ_scroll_N{args.N}_K{args.K}_H{args.H}_step{args.step}.gif",
            occ,
            args.H,
            args.step,
        )

    vals = build_values_log(args.N, args.K)
    vals_img = normalize_to_uint8(vals)
    save_png(out_dir / f"divisibility_val_log_N{args.N}_K{args.K}.png", vals_img)

    for diagN in args.diag_N:
        save_diagonal_plot(out_dir, diagN, args.K)

    print(f"OK: wrote atlas artifacts to {out_dir}")


if __name__ == "__main__":
    main()
