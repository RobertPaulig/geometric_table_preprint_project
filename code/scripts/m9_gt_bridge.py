#!/usr/bin/env python
# code/scripts/m9_gt_bridge.py
from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from geometric_table import compute_core_metrics_from_rows, compute_core_metrics_fast


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wheel-csv", type=str, required=True)
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--sample", type=int, default=4000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--core-r", type=int, default=30)
    p.add_argument("--core-rt", type=int, default=30)
    p.add_argument("--K", type=int, default=120)
    p.add_argument("--primitive", action="store_true")
    p.add_argument("--weight", type=str, default="ones")
    p.add_argument("--eps", type=float, default=1e-12)
    p.add_argument("--layer-primes", type=str, required=True)
    p.add_argument("--row-mode", type=str, default="consecutive", choices=["consecutive", "wheel"])
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_t_lists(path: Path) -> Tuple[List[int], List[int]]:
    twins: List[int] = []
    non: List[int] = []
    with open_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["t"])
            is_twin = int(row["is_twin"])
            if is_twin:
                twins.append(t)
            else:
                non.append(t)
    return twins, non


def sample_t(twins: List[int], non: List[int], n: int, seed: int) -> Tuple[List[int], int, int]:
    rng = random.Random(seed)
    target_twins = min(len(twins), n // 2)
    target_non = min(len(non), n - target_twins)
    if target_twins < n // 2:
        target_non = min(len(non), n - target_twins)
    twin_sample = rng.sample(twins, target_twins) if target_twins else []
    non_sample = rng.sample(non, target_non) if target_non else []
    return twin_sample + non_sample, target_twins, target_non


def mod_inv(B: int, p: int) -> int:
    return pow(B % p, -1, p)


def dist_to_forbid(t: int, p: int, inv: int) -> int:
    r1 = inv % p
    r2 = (-inv) % p
    d1 = min((t - r1) % p, (r1 - t) % p)
    d2 = min((t - r2) % p, (r2 - t) % p)
    return min(d1, d2)


def main() -> None:
    args = parse_args()
    primes = [int(p.strip()) for p in args.layer_primes.split(",") if p.strip()]
    if not primes:
        raise ValueError("--layer-primes is empty")

    twins, non = read_t_lists(Path(args.wheel_csv))
    t_list, n_twins, n_non = sample_t(twins, non, args.sample, args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    invs = {p: mod_inv(args.B, p) for p in primes}
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = [
            "t", "center", "is_twin",
            "core_edges", "core_components", "core_isolated_nodes",
            "core_gc_size", "core_gc_fraction", "core_gc_spectral_gap", "core_gc_entropy",
        ]
        if args.row_mode == "consecutive":
            header.extend([
                "twin_isolates", "twin_deg_sum",
                "twin_deg_minus1", "twin_deg_plus1",
                "twin_is_isolated_minus1", "twin_is_isolated_plus1",
                "twin_in_gc_minus1", "twin_in_gc_plus1",
            ])
        for p in primes:
            header.extend([f"t_mod_{p}", f"is_forbidden_{p}", f"dist_to_forbid_{p}"])
        header.extend(["layer_hits_L1", "layer_hits_L2", "layer_hits_L3", "layer_allow_L1", "layer_allow_L2", "layer_allow_L3"])
        w.writerow(header)
        for t in t_list:
            center = args.B * t
            if args.row_mode == "wheel":
                rows = [args.B * (t + i) for i in range(-args.core_rt, args.core_rt + 1)]
                metrics = compute_core_metrics_from_rows(
                    rows=rows,
                    center_value=None,
                    K=args.K,
                    primitive=args.primitive,
                    weight=args.weight,
                    eps=args.eps,
                    include_twin=False,
                )
            else:
                metrics = compute_core_metrics_fast(
                    center=center,
                    core_r=args.core_r,
                    K=args.K,
                    primitive=args.primitive,
                    weight=args.weight,
                    eps=args.eps,
                )
            row = [
                t, center, int(t in twins),
                metrics["core_edges"], metrics["core_components"], metrics["core_isolated_nodes"],
                metrics["core_gc_size"], metrics["core_gc_fraction"], metrics["core_gc_spectral_gap"], metrics["core_gc_entropy"],
            ]
            if args.row_mode == "consecutive":
                row.extend([
                    metrics["twin_isolates"], metrics["twin_deg_sum"],
                    metrics["twin_deg_minus1"], metrics["twin_deg_plus1"],
                    metrics["twin_is_isolated_minus1"], metrics["twin_is_isolated_plus1"],
                    metrics["twin_in_gc_minus1"], metrics["twin_in_gc_plus1"],
                ])
            forb_flags = []
            for p in primes:
                inv = invs[p]
                r = t % p
                is_forb = int(r == inv % p or r == (-inv) % p)
                forb_flags.append(is_forb)
                row.extend([r, is_forb, dist_to_forbid(t, p, inv)])
            # layer hits/allow for up to three primes
            layer_hits = [0, 0, 0]
            if len(forb_flags) >= 1:
                layer_hits[0] = int(forb_flags[0] == 1)
            if len(forb_flags) >= 2:
                layer_hits[1] = int(forb_flags[0] or forb_flags[1])
            if len(forb_flags) >= 3:
                layer_hits[2] = int(forb_flags[0] or forb_flags[1] or forb_flags[2])
            layer_allow = [int(h == 0) for h in layer_hits]
            row.extend(layer_hits + layer_allow)
            w.writerow(row)

    print(f"OK: wrote {out_path} (twins={n_twins}, non={n_non})")


if __name__ == "__main__":
    main()
