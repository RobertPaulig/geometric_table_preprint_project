# code/scripts/batch_centers.py
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

from geometric_table import BuildParams, compute_rowproj_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--center-min", type=int, default=500)
    p.add_argument("--center-max", type=int, default=1500)
    p.add_argument("--M", type=int, default=30)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--h", type=int, default=200)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--primitive", action="store_true", default=True)
    p.add_argument("--include-centers", type=str, default="600,840,1000")
    p.add_argument("--out", type=str, default="out/batch_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    centers = list(range(args.center_min, args.center_max + 1))
    include = [int(x) for x in args.include_centers.split(",") if x.strip()]
    include = [c for c in include if c in centers]
    pool = [c for c in centers if c not in include]
    needed = max(0, args.M - len(include))
    if needed > len(pool):
        raise ValueError("M cannot exceed range size")
    sample = include + rng.sample(pool, needed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "center",
        "n_nodes",
        "n_edges",
        "n_components",
        "zero_count_all",
        "isolated_nodes",
        "isolated_fraction",
        "gc_size",
        "gc_fraction",
        "gc_edges",
        "gc_spectral_gap",
        "gc_entropy",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for center in sample:
            params = BuildParams(
                center=center,
                h=args.h,
                K=args.K,
                primitive=bool(args.primitive),
                weight="ones",
                graph_mode="rowproj",
            )
            _, _, _, metrics, _, _, _, _ = compute_rowproj_metrics(params, neigs=50)
            row = {k: metrics.get(k) for k in fieldnames}
            row["center"] = center
            w.writerow(row)

    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
