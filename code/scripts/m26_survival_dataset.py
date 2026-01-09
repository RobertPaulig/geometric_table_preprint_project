#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-max", type=int, required=True)
    p.add_argument("--Q0", type=int, required=True)
    p.add_argument("--Q-list", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--mersenne-strict", type=int, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--label", type=str, required=True)
    return p.parse_args()


def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p: n + 1: p] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def factorize(n: int, primes: List[int]) -> List[int]:
    factors: List[int] = []
    x = n
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            factors.append(p)
            while x % p == 0:
                x //= p
    if x > 1:
        factors.append(x)
    return factors


def ord_mod_2(q: int, primes: List[int]) -> int:
    if q <= 2:
        return 1
    factors = factorize(q - 1, primes)
    d = q - 1
    for p in factors:
        while d % p == 0 and pow(2, d // p, q) == 1:
            d //= p
    return d


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    if args.Q0 <= 0:
        raise ValueError("Q0 must be positive")
    q_list = parse_int_list(args.Q_list)
    if not q_list:
        raise ValueError("Q-list is empty")
    if args.Q0 > max(q_list):
        raise ValueError("Q0 must be <= max(Q-list)")

    mersenne_strict = int(args.mersenne_strict) == 1
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)

    Q_max = max(q_list)
    primes_q = sieve_primes(Q_max)
    primes_q = [q for q in primes_q if q != 2]
    primes_factor = sieve_primes(int(math.isqrt(Q_max)) + 1)

    ords: Dict[int, int] = {}
    for q in primes_q:
        ords[q] = ord_mod_2(q, primes_factor)

    primes_p = sieve_primes(args.p_max)
    primes_p = [p for p in primes_p if p >= 2]
    primes_p_set = set(primes_p)

    # killed flags for each Q in list and Q0
    killed_q0 = {p: 0 for p in primes_p}
    killed_by_Q: Dict[int, Dict[int, int]] = {Q: {p: 0 for p in primes_p} for Q in q_list}

    for q in primes_q:
        d = ords[q]
        if d in primes_p_set:
            if q <= args.Q0:
                killed_q0[d] = 1
            for Q in q_list:
                if q <= Q:
                    killed_by_Q[Q][d] = 1

    # hazard features per Q (delta Q0->Q)
    ap_count_delta: Dict[int, Dict[int, int]] = {Q: {p: 0 for p in primes_p} for Q in q_list}
    ap_harm_delta: Dict[int, Dict[int, float]] = {Q: {p: 0.0 for p in primes_p} for Q in q_list}

    for q in primes_q:
        if q <= args.Q0:
            continue
        if mersenne_strict and (q % 8 not in (1, 7)):
            continue
        base = q - 1
        if mersenne_strict:
            base //= 2
        q_factors = factorize(base, primes_factor)
        for p in q_factors:
            if p not in primes_p_set:
                continue
            for Q in q_list:
                if q <= Q:
                    ap_count_delta[Q][p] += 1
                    ap_harm_delta[Q][p] += p / (q - 1)

    # write dataset
    header = ["p", "killed_Q0"]
    for Q in q_list:
        header.append(f"ap_count_delta_{Q}")
        header.append(f"ap_harm_delta_{Q}")
    for Q in q_list:
        header.append(f"survive_{Q}")

    rows = []
    for p in primes_p:
        row = {"p": p, "killed_Q0": killed_q0[p]}
        for Q in q_list:
            row[f"ap_count_delta_{Q}"] = ap_count_delta[Q][p]
            row[f"ap_harm_delta_{Q}"] = ap_harm_delta[Q][p]
        for Q in q_list:
            row[f"survive_{Q}"] = 0 if killed_by_Q[Q][p] == 1 else 1
        rows.append(row)

    dataset_csv = out_dir / "m26_dataset.csv"
    with dataset_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "p_max": args.p_max,
        "Q0": args.Q0,
        "Q_list": q_list,
        "mersenne_strict": mersenne_strict,
        "n_primes": len(primes_p),
        "dataset": str(dataset_csv),
    }
    (out_dir / "m26_dataset_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"OK: wrote dataset to {dataset_csv}")


if __name__ == "__main__":
    main()
