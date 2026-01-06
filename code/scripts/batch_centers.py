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

from geometric_table import BuildParams, compute_rowproj_metrics, compute_core_metrics_fast, compute_core_gc_size_fast


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
    p.add_argument("--auto-k", action="store_true", default=False)
    p.add_argument("--k-max", type=int, default=5000)
    p.add_argument("--k-step", type=int, default=200)
    p.add_argument("--min-gc-size", type=int, default=10)
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
        "K_used",
        "auto_k_iters",
    ]

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for center in sample:
            if args.fast_core:
                if args.auto_k:
                    k_used = args.K
                    iters = 0
                    gc_size = compute_core_gc_size_fast(
                        center=center,
                        core_r=args.core_r,
                        K=k_used,
                        primitive=bool(args.primitive),
                        weight=args.weight,
                        eps=1e-12,
                    )
                    while gc_size < args.min_gc_size and k_used < args.k_max:
                        k_used = min(args.k_max, k_used + args.k_step)
                        iters += 1
                        gc_size = compute_core_gc_size_fast(
                            center=center,
                            core_r=args.core_r,
                            K=k_used,
                            primitive=bool(args.primitive),
                            weight=args.weight,
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
                    iters = 0
                    metrics = compute_core_metrics_fast(
                        center=center,
                        core_r=args.core_r,
                        K=args.K,
                        primitive=bool(args.primitive),
                        weight=args.weight,
                        eps=1e-12,
                    )
            else:
                k_used = args.K
                iters = 0
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
            row["K_used"] = k_used
            row["auto_k_iters"] = iters
            w.writerow(row)

    print(f"OK: wrote {out_path}")


if __name__ == "__main__":
    main()
