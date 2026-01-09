#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-max", type=int, default=500)
    p.add_argument("--q-max", type=int, default=2000)
    p.add_argument("--ord-min", type=int, default=10)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p : n + 1 : p] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def factorize(n: int) -> List[Tuple[int, int]]:
    factors: List[Tuple[int, int]] = []
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            exp = 0
            while x % d == 0:
                x //= d
                exp += 1
            factors.append((d, exp))
        d = 3 if d == 2 else d + 2
    if x > 1:
        factors.append((x, 1))
    return factors


def divisors_from_factors(factors: List[Tuple[int, int]]) -> List[int]:
    divisors = [1]
    for p, e in factors:
        new = []
        pow_p = 1
        for _ in range(e):
            pow_p *= p
            new.extend([d * pow_p for d in divisors])
        divisors.extend(new)
    return sorted(divisors)


def ord_mod(a: int, q: int) -> int:
    if q <= 1:
        return 1
    factors = factorize(q - 1)
    divisors = divisors_from_factors(factors)
    for d in divisors:
        if pow(a, d, q) == 1:
            return d
    return q - 1


def save_heatmap(path: Path, mat: np.ndarray, title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_hist(path: Path, data: List[int], title: str, xlabel: str, ylabel: str, bins: int = 60) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(data, bins=bins, color="#4C72B0", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qs = [q for q in sieve_primes(args.q_max) if q != 2]
    p_max = int(args.p_max)
    ords: Dict[int, int] = {}
    for q in qs:
        ords[q] = ord_mod(2, q)

    mat = np.zeros((len(qs), p_max), dtype=np.uint8)
    for i, q in enumerate(qs):
        d = ords[q]
        if d <= 0:
            continue
        mat[i, d - 1 : p_max : d] = 1

    save_heatmap(
        out_dir / "m15_divisibility_heatmap.png",
        mat,
        "Divisibility of 2^p-1 by prime q (M15)",
        "p (1..p_max)",
        "q index (primes up to q_max)",
    )

    ord_values = list(ords.values())
    save_hist(
        out_dir / "m15_ord_hist.png",
        ord_values,
        "Order histogram ord_q(2) for primes q",
        "ord_q(2)",
        "count",
    )

    filtered_qs = [q for q in qs if ords[q] >= args.ord_min]
    filtered_ords = [ords[q] for q in filtered_qs]

    counts: Dict[int, int] = {}
    for d in ord_values:
        counts[d] = counts.get(d, 0) + 1
    top_periods = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[:12]

    top_rows = []
    for d, c in top_periods:
        examples = [q for q in qs if ords[q] == d][:5]
        top_rows.append({"ord": d, "count": c, "example_q": examples})

    with (out_dir / "m15_top_periods.json").open("w", encoding="utf-8") as f:
        json.dump({
            "p_max": p_max,
            "q_max": args.q_max,
            "ord_min": args.ord_min,
            "n_primes": len(qs),
            "n_filtered": len(filtered_qs),
            "top_periods": top_rows,
        }, f, indent=2)

    with (out_dir / "m15_top_periods.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ord", "count", "example_q"])
        for row in top_rows:
            w.writerow([row["ord"], row["count"], ",".join(str(x) for x in row["example_q"])])

    # Save filtered heatmap for reference (optional artifact)
    if filtered_qs:
        mat_f = np.zeros((len(filtered_qs), p_max), dtype=np.uint8)
        for i, q in enumerate(filtered_qs):
            d = ords[q]
            mat_f[i, d - 1 : p_max : d] = 1
        save_heatmap(
            out_dir / "m15_divisibility_heatmap_filtered.png",
            mat_f,
            f"Divisibility heatmap (ord >= {args.ord_min})",
            "p (1..p_max)",
            "q index (filtered primes)",
        )

    print(f"OK: wrote M15 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
