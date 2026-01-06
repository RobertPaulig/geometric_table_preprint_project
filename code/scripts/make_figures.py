# code/scripts/make_figures.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_rows(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append({
                "center": int(row["center"]),
                "is_twin_center": int(row["is_twin_center"]),
                "core_gc_spectral_gap": float(row["core_gc_spectral_gap"]),
                "core_gc_entropy": float(row["core_gc_entropy"]),
                "core_edges": float(row["core_edges"]),
            })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--label", type=str, default="ones")
    p.add_argument("--input", type=str, default="")
    p.add_argument("--out-dir", type=str, default="fig")
    args = p.parse_args()

    if args.input:
        input_path = Path(args.input)
        label = args.label
    else:
        label = args.label
        input_path = Path("out") / f"batch_summary_{label}.csv"
    rows = read_rows(input_path)
    twins = [r for r in rows if r["is_twin_center"] == 1]
    non_twins = [r for r in rows if r["is_twin_center"] == 0]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Boxplot: core gap
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([
        [r["core_gc_spectral_gap"] for r in non_twins],
        [r["core_gc_spectral_gap"] for r in twins],
    ], tick_labels=["non-twin", "twin"], showfliers=True)
    ax.set_ylabel("core_gc_spectral_gap")
    ax.set_title("Core Gap: Twins vs Non-twins")
    fig.tight_layout()
    fig.savefig(out_dir / f"core_gap_box_twins_{label}.png", dpi=150)
    plt.close(fig)

    # Boxplot: core entropy
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.boxplot([
        [r["core_gc_entropy"] for r in non_twins],
        [r["core_gc_entropy"] for r in twins],
    ], tick_labels=["non-twin", "twin"], showfliers=True)
    ax.set_ylabel("core_gc_entropy")
    ax.set_title("Core Entropy: Twins vs Non-twins")
    fig.tight_layout()
    fig.savefig(out_dir / f"core_entropy_box_twins_{label}.png", dpi=150)
    plt.close(fig)

    # Scatter: core edges vs gap
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(
        [r["core_edges"] for r in non_twins],
        [r["core_gc_spectral_gap"] for r in non_twins],
        s=18,
        alpha=0.7,
        label="non-twin",
    )
    ax.scatter(
        [r["core_edges"] for r in twins],
        [r["core_gc_spectral_gap"] for r in twins],
        s=28,
        alpha=0.9,
        label="twin",
        marker="s",
    )
    for c in (600, 840, 1000):
        row = next((r for r in rows if r["center"] == c), None)
        if row:
            ax.scatter([row["core_edges"]], [row["core_gc_spectral_gap"]], s=60, edgecolors="black", facecolors="none")
            ax.text(row["core_edges"], row["core_gc_spectral_gap"], f" {c}", fontsize=8)
    ax.set_xlabel("core_edges")
    ax.set_ylabel("core_gc_spectral_gap")
    ax.set_title("Core Gap vs Core Edges")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"core_gap_scatter_edges_{label}.png", dpi=150)
    plt.close(fig)

    # Histogram: core gap
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    vals = [r["core_gc_spectral_gap"] for r in rows]
    ax.hist(vals, bins=20, alpha=0.8, color="steelblue")
    for c, color in [(600, "red"), (840, "orange"), (1000, "black")]:
        row = next((r for r in rows if r["center"] == c), None)
        if row:
            ax.axvline(row["core_gc_spectral_gap"], color=color, linestyle="--", linewidth=1.5, label=str(c))
    ax.set_xlabel("core_gc_spectral_gap")
    ax.set_ylabel("count")
    ax.set_title("Core Gap Distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / f"core_gap_hist_{label}.png", dpi=150)
    plt.close(fig)

    print(f"OK: wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
