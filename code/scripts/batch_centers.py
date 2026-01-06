# code/scripts/batch_centers.py
from __future__ import annotations

import argparse
import csv
import random
import math
from pathlib import Path


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True

from geometric_table import (
    BuildParams,
    compute_rowproj_metrics,
    compute_core_metrics_fast,
    compute_core_gc_size_fast,
    choose_hybrid_K,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--center-min", type=int, default=500)
    p.add_argument("--center-max", type=int, default=1500)
    p.add_argument("--M", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--h", type=int, default=200)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--primitive", action="store_true", default=True)
    p.add_argument("--include-centers", type=str, default="600,840,1000")
    p.add_argument("--weight", choices=["ones", "idf"], default="ones")
    p.add_argument("--core-r", type=int, default=30)
    p.add_argument("--center-set", choices=["all", "even", "six"], default="all")
    p.add_argument("--fast-core", action="store_true", default=False)
    p.add_argument("--auto-k", action="store_true", default=False, help="deprecated; kept for compatibility")
    p.add_argument("--alpha", type=float, default=1.0, help="scale-law coefficient for K0")
    p.add_argument("--growth", type=float, default=1.5, help="multiplicative bump for K")
    p.add_argument("--max-bumps", type=int, default=6, help="max auto-tuning iterations")
    p.add_argument("--min-gc-size", type=int, default=10, help="target core GC size")
    p.add_argument("--k-max", type=int, default=20000, help="upper cap for K")
    p.add_argument("--out", type=str, default="out/batch_summary.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    if args.center_set == "six":
        start = int(math.ceil(args.center_min / 6))
        end = int(math.floor(args.center_max / 6))
        centers = [6 * m for m in range(start, end + 1)]
    elif args.center_set == "even":
        start = args.center_min + (args.center_min % 2)
        centers = list(range(start, args.center_max + 1, 2))
    else:
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
        "is_twin_center",
        "center_set",
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
        "core_nodes",
        "core_edges",
        "core_components",
        "core_isolated_nodes",
        "core_gc_size",
        "core_gc_fraction",
        "core_gc_spectral_gap",
        "core_gc_entropy",
        "core_gc_zero_count_all",
        "K0",
        "K_used",
        "k_bumps",
        "hit_kmax",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for center in sample:
            if args.fast_core:
                K0, k_used, k_bumps, hit_kmax, _ = choose_hybrid_K(
                    center=center,
                    core_r=args.core_r,
                    primitive=bool(args.primitive),
                    weight=args.weight,
                    alpha=args.alpha,
                    growth=args.growth,
                    min_core_gc=args.min_gc_size,
                    max_bumps=args.max_bumps,
                    K_max=args.k_max,
                    eps=1e-12,
                )
                metrics = compute_core_metrics_fast(
                    center=center,
                    core_r=args.core_r,
                    K=k_used,
                    primitive=bool(args.primitive),
                    weight=args.weight,
                    eps=1e-12,
                )
            else:
                k_used = args.K
                K0 = k_used
                k_bumps = 0
                hit_kmax = False
                params = BuildParams(
                    center=center,
                    h=args.h,
                    K=args.K,
                    primitive=bool(args.primitive),
                    weight=args.weight,
                    graph_mode="rowproj",
                )
                _, _, _, metrics, _, _, _, _ = compute_rowproj_metrics(
                    params, neigs=50, core_r=args.core_r
                )
            row = {k: metrics.get(k) for k in fieldnames}
            row["center"] = center
            row["is_twin_center"] = int(is_prime(center - 1) and is_prime(center + 1))
            row["center_set"] = args.center_set
            row["K0"] = K0
            row["K_used"] = k_used
            row["k_bumps"] = k_bumps
            row["hit_kmax"] = int(hit_kmax)
            w.writerow(row)

    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
