#!/usr/bin/env python
# code/scripts/wave_metrics.py
from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

try:
    import imageio.v2 as imageio
except Exception as exc:  # pragma: no cover
    raise SystemExit("imageio is required for PNG output") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=3000)
    p.add_argument("--K", type=int, default=120)
    p.add_argument("--H", type=int, default=220)
    p.add_argument("--step", type=int, default=60)
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/metrics")
    p.add_argument("--diag-N", type=int, action="append", default=[60, 420, 2520, 27720])
    p.add_argument("--window-mode", choices=["sliding"], default="sliding")
    return p.parse_args()


def occupancy_window(N: int, K: int, start: int, H: int) -> np.ndarray:
    occ = np.zeros((H, K), dtype=np.uint8)
    for k in range(1, K + 1):
        first = ((start + k - 1) // k) * k
        for n in range(first, start + H):
            if n % k == 0:
                occ[n - start, k - 1] = 1
    return occ


def col_density(occ: np.ndarray) -> np.ndarray:
    return occ.mean(axis=0)


def q_counts(occ: np.ndarray, start: int) -> Counter:
    H, K = occ.shape
    counts = Counter()
    for i in range(H):
        n = start + i
        for k in range(1, K + 1):
            if occ[i, k - 1]:
                q = n // k
                counts[q] += 1
    return counts


def q_count_formula(start: int, H: int, K: int, q: int) -> int:
    if q <= 0:
        return 0
    lo = (start + q - 1) // q
    hi = (start + H - 1) // q
    hi = min(K, hi)
    if hi < lo:
        return 0
    return hi - lo + 1


def diag_hits(N0: int, K: int) -> Tuple[List[int], int]:
    hits = [k for k in range(1, K + 1) if N0 % k == 0]
    run_len = 0
    cur = 0
    prev = None
    for k in hits:
        if prev is None or k == prev + 1:
            cur += 1
        else:
            run_len = max(run_len, cur)
            cur = 1
        prev = k
    run_len = max(run_len, cur)
    return hits, run_len


def save_heatmap(path: Path, mat: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vmin = float(mat.min())
    vmax = float(mat.max())
    if vmax <= vmin:
        img = np.zeros(mat.shape, dtype=np.uint8)
    else:
        img = ((mat - vmin) / (vmax - vmin) * 255).astype(np.uint8)
    imageio.imwrite(path, img)


def save_bar_png(path: Path, xs: List[int], ys: List[int], title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(xs, ys, width=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_diag_raster(path: Path, hits: List[int], K: int, title: str) -> None:
    import matplotlib.pyplot as plt

    arr = np.zeros(K, dtype=np.uint8)
    for k in hits:
        if 1 <= k <= K:
            arr[k - 1] = 1
    fig, ax = plt.subplots(figsize=(6, 1.6))
    ax.imshow(arr[np.newaxis, :], aspect="auto", cmap="gray_r")
    ax.set_yticks([])
    ax.set_xlabel("k")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    harmonic = sum(1.0 / k for k in range(1, args.K + 1))
    occ_expected = harmonic / args.K

    windows = list(range(1, args.N - args.H + 2, args.step))
    col_rows = []
    q_rows = []
    q_check_rows = []
    window_rows = []

    col_heat = np.zeros((len(windows), args.K), dtype=float)

    for wi, start in enumerate(windows):
        occ = occupancy_window(args.N, args.K, start, args.H)
        dens = col_density(occ)
        col_heat[wi, :] = dens
        for k in range(1, args.K + 1):
            col_rows.append([start, k, float(dens[k - 1])])

        # q counts
        qc = q_counts(occ, start)
        for q, cnt in qc.most_common(50):
            q_rows.append([start, q, cnt])
            cnt_formula = q_count_formula(start, args.H, args.K, q)
            q_check_rows.append([start, q, cnt, cnt_formula, int(cnt == cnt_formula)])

        # window summaries
        occ_density = float(occ.mean())
        row_nonempty = occ.sum(axis=1)
        window_rows.append([
            start,
            occ_density,
            occ_expected,
            occ_density - occ_expected,
            float(np.mean(row_nonempty)),
            float(np.var(row_nonempty)),
        ])

    # Column density CSV + plots
    with (out_dir / "col_density_windows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["win_start", "k", "density"])
        w.writerows(col_rows)

    save_heatmap(out_dir / "col_density_heatmap.png", col_heat)
    first_start = windows[0] if windows else 1
    first_dens = col_heat[0, :] if windows else np.zeros(args.K, dtype=float)
    save_bar_png(
        out_dir / "col_density_k.png",
        list(range(1, args.K + 1)),
        [float(x) for x in first_dens],
        title=f"Column density (window start {first_start})",
        xlabel="k",
        ylabel="density",
    )

    # q top CSV + plots for first/mid/last windows
    with (out_dir / "q_top_windows.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["win_start", "q", "count"])
        w.writerows(q_rows)

    with (out_dir / "q_counts_check.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["win_start", "q", "count_observed", "count_formula", "ok_flag"])
        w.writerows(q_check_rows)

    if windows:
        sample_starts = [windows[0], windows[len(windows) // 2], windows[-1]]
        for start in sample_starts:
            top = [r for r in q_rows if r[0] == start][:20]
            if not top:
                continue
            qs = [r[1] for r in top]
            cs = [r[2] for r in top]
            save_bar_png(
                out_dir / f"q_top_bars_win{start}.png",
                qs,
                cs,
                title=f"Top q counts (window start {start})",
                xlabel="q",
                ylabel="count",
            )

    # Diagonal hits summary + rasters
    with (out_dir / "diag_hits_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N0", "K", "num_hits", "run_len", "first_k", "last_k"])
        for N0 in args.diag_N:
            hits, run_len = diag_hits(N0, args.K)
            first_k = hits[0] if hits else 0
            last_k = hits[-1] if hits else 0
            w.writerow([N0, args.K, len(hits), run_len, first_k, last_k])
            save_diag_raster(
                out_dir / f"diag_hits_raster_N{N0}_K{args.K}.png",
                hits,
                args.K,
                title=f"Diagonal hits raster: N={N0}, K={args.K}",
            )

    # Window summary CSV
    with (out_dir / "window_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "win_start",
            "occ_density",
            "occ_density_expected",
            "occ_error",
            "row_nonempty_mean",
            "row_nonempty_var",
        ])
        w.writerows(window_rows)

    # Column density baseline vs 1/k
    mean_dens = col_heat.mean(axis=0) if len(windows) > 0 else np.zeros(args.K, dtype=float)
    with (out_dir / "col_density_error.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["k", "observed_density", "baseline_1_over_k", "error"])
        for k in range(1, args.K + 1):
            baseline = 1.0 / k
            observed = float(mean_dens[k - 1])
            w.writerow([k, observed, baseline, observed - baseline])
    save_bar_png(
        out_dir / "col_density_vs_1_over_k.png",
        list(range(1, args.K + 1)),
        [float(x) for x in mean_dens],
        title="Column density vs 1/k (mean across windows)",
        xlabel="k",
        ylabel="density",
    )

    print(f"OK: wrote wave metrics to {out_dir}")


if __name__ == "__main__":
    main()
