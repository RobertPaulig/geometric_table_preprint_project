#!/usr/bin/env python
# code/scripts/m7_overlay_plot.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

from wave_atlas_io import open_text, resolve_existing_or_gz


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m10-dir", type=str, required=True)
    p.add_argument("--m12-dir", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def read_spectrum(path: Path, max_points: int = 20000) -> Tuple[List[int], List[float]]:
    xs: List[int] = []
    ys: List[float] = []
    resolved = resolve_existing_or_gz(path)
    with open_text(resolved, "rt", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(int(row["f_idx"]))
            ys.append(float(row["power"]))
            if len(xs) >= max_points:
                break
    return xs, ys


def main() -> None:
    args = parse_args()
    m10_dir = Path(args.m10_dir)
    m12_dir = Path(args.m12_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, label, base in [
        (axes[0], "m=10", m10_dir),
        (axes[1], "m=12", m12_dir),
    ]:
        for name, color in [
            ("detrend_fft_power_detrended.csv", "#1f77b4"),
            ("cond_p0_fft_power.csv", "#d62728"),
        ]:
            xs, ys = read_spectrum(base / name)
            ax.plot(xs, ys, linewidth=0.8, color=color, label=name.replace(".csv", ""))
        ax.set_title(label)
        ax.set_xlabel("f_idx")
        ax.set_yscale("log")
    axes[0].set_ylabel("power")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
