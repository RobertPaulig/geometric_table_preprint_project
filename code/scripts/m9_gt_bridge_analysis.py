#!/usr/bin/env python
# code/scripts/m9_gt_bridge_analysis.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--p0", type=int, required=True)
    p.add_argument("--p1", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--perm", type=int, default=1000)
    return p.parse_args()


def read_csv(path: Path) -> Dict[str, np.ndarray]:
    import pandas as pd

    df = pd.read_csv(path)
    return {c: df[c].to_numpy() for c in df.columns}


def save_bar(path: Path, xs: List[int], ys: List[float], title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(xs, ys, width=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_boxplot(path: Path, groups: List[np.ndarray], labels: List[str], title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_violin(path: Path, groups: List[np.ndarray], labels: List[str], title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    parts = ax.violinplot(groups, showextrema=True)
    for pc in parts["bodies"]:
        pc.set_alpha(0.6)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def perm_pvalue(a: np.ndarray, b: np.ndarray, iters: int, seed: int = 123) -> float:
    rng = np.random.default_rng(seed)
    obs = abs(a.mean() - b.mean())
    x = np.concatenate([a, b])
    n = len(a)
    count = 0
    for _ in range(iters):
        rng.shuffle(x)
        diff = abs(x[:n].mean() - x[n:].mean())
        if diff >= obs:
            count += 1
    return (count + 1) / (iters + 1)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = read_csv(Path(args.csv))

    # residue bars
    t_mod_p0 = data[f"t_mod_{args.p0}"].astype(int)
    core_edges = data["core_edges"]
    gc_frac = data["core_gc_fraction"]
    means_edges = []
    means_gc = []
    for r in range(args.p0):
        mask = t_mod_p0 == r
        means_edges.append(float(core_edges[mask].mean()) if mask.any() else 0.0)
        means_gc.append(float(gc_frac[mask].mean()) if mask.any() else 0.0)
    save_bar(out_dir / "m9_core_edges_by_mod_p0.png", list(range(args.p0)), means_edges,
             f"core_edges by t mod {args.p0}", "residue", "mean core_edges")
    save_bar(out_dir / "m9_gcfrac_by_mod_p0.png", list(range(args.p0)), means_gc,
             f"core_gc_fraction by t mod {args.p0}", "residue", "mean gc_fraction")

    # layer effects
    groups_edges = []
    groups_gap = []
    groups_entropy = []
    labels = ["all", "L1", "L2", "L3"]
    masks = [
        np.ones_like(core_edges, dtype=bool),
        data["layer_allow_L1"] == 1,
        data["layer_allow_L2"] == 1,
        data["layer_allow_L3"] == 1,
    ]
    for m in masks:
        groups_edges.append(core_edges[m])
        groups_gap.append(data["core_gc_spectral_gap"][m])
        groups_entropy.append(data["core_gc_entropy"][m])
    save_boxplot(out_dir / "m9_layers_box_core_edges.png", groups_edges, labels, "core_edges by conditioning layer", "core_edges")
    save_boxplot(out_dir / "m9_layers_box_gap.png", groups_gap, labels, "core_gc_spectral_gap by layer", "gap")
    save_boxplot(out_dir / "m9_layers_box_entropy.png", groups_entropy, labels, "core_gc_entropy by layer", "entropy")

    # twin vs non
    is_twin = data["is_twin"] == 1
    twin_gap = data["core_gc_spectral_gap"][is_twin]
    non_gap = data["core_gc_spectral_gap"][~is_twin]
    twin_iso = data["twin_isolates"][is_twin]
    non_iso = data["twin_isolates"][~is_twin]
    save_violin(out_dir / "m9_twin_non_gap.png", [twin_gap, non_gap], ["twin", "non"], "core_gc_spectral_gap", "gap")
    save_violin(out_dir / "m9_twin_non_twin_isolates.png", [twin_iso, non_iso], ["twin", "non"], "twin_isolates", "isolates")

    # summary
    dist_p0 = data[f"dist_to_forbid_{args.p0}"]
    corr = 0.0
    if len(dist_p0) > 1 and np.std(dist_p0) > 0 and np.std(core_edges) > 0:
        corr = float(np.corrcoef(dist_p0, core_edges)[0, 1])
    summary = {
        "n_total": int(len(core_edges)),
        "n_twin": int(is_twin.sum()),
        "n_non": int((~is_twin).sum()),
        "perm_p_gap": float(perm_pvalue(twin_gap, non_gap, args.perm)),
        "perm_p_twin_isolates": float(perm_pvalue(twin_iso, non_iso, args.perm)),
        "corr_dist_p0_core_edges": corr,
        "layer_counts": {
            "all": int(len(core_edges)),
            "L1": int(masks[1].sum()),
            "L2": int(masks[2].sum()),
            "L3": int(masks[3].sum()),
        },
    }
    Path(out_dir / "m9_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"OK: wrote M9 analysis to {out_dir}")


if __name__ == "__main__":
    main()
