#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-max", type=int, default=100000)
    p.add_argument("--Q0", type=int, default=50000)
    p.add_argument("--Q1", type=int, default=200000)
    p.add_argument("--test-costs", type=str, default="1,3600,86400")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


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


def save_line_plot(path: Path, xs: List[float], ys: List[float],
                   title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.Q0 > args.Q1:
        raise ValueError("Q0 must be <= Q1")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primes_q = sieve_primes(args.Q1)
    primes_q = [q for q in primes_q if q != 2]
    primes_factor = sieve_primes(int(math.isqrt(args.Q1)) + 1)

    ords: Dict[int, int] = {}
    for q in primes_q:
        ords[q] = ord_mod_2(q, primes_factor)

    primes_p = sieve_primes(args.p_max)
    primes_p_set = set(primes_p)

    ap_harm_delta: Dict[int, float] = {p: 0.0 for p in primes_p}
    for q in primes_q:
        if q <= args.Q0:
            continue
        q_factors = factorize(q - 1, primes_factor)
        for p in q_factors:
            if p in primes_p_set:
                ap_harm_delta[p] += p / (q - 1)

    killed_q0 = {p: 0 for p in primes_p}
    killed_q1 = {p: 0 for p in primes_p}
    for q in primes_q:
        d = ords[q]
        if d in killed_q1:
            killed_q1[d] = 1
            if q <= args.Q0:
                killed_q0[d] = 1

    rows = []
    for p in primes_p:
        row = {
            "p": p,
            "killed_Q0": killed_q0[p],
            "survive_Q1": 1 if killed_q1[p] == 0 else 0,
            "hazard": ap_harm_delta[p],
        }
        rows.append(row)

    # Build queue: killed_Q0 at tail, survivors sorted by hazard ascending
    queue = sorted(
        rows,
        key=lambda r: (r["killed_Q0"], r["hazard"])
    )
    for rank, r in enumerate(queue, start=1):
        r["rank"] = rank

    queue_csv = out_dir / "m22_queue.csv"
    with queue_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(queue[0].keys()))
        w.writeheader()
        w.writerows(queue)

    n = len(queue)
    overall_survival = sum(r["survive_Q1"] for r in queue) / n if n else 0.0

    fractions = np.linspace(0.02, 0.5, 25)
    enrich_vals = []
    tests_avoided_vals = []
    for frac in fractions:
        k = max(1, int(n * frac))
        tested = queue[:k]
        tail = queue[k:]
        survival_top = sum(r["survive_Q1"] for r in tested) / k
        enrich_vals.append(survival_top / overall_survival if overall_survival > 0 else 0.0)
        bad_in_tail = sum(1 - r["survive_Q1"] for r in tail)
        tests_avoided_vals.append(bad_in_tail)

    save_line_plot(
        out_dir / "m22_enrichment_curve.png",
        list(fractions),
        enrich_vals,
        "Survival enrichment vs tested fraction (M22)",
        "tested fraction",
        "enrichment",
    )

    save_line_plot(
        out_dir / "m22_tests_avoided.png",
        list(fractions),
        tests_avoided_vals,
        "Bad tests avoided vs tested fraction (M22)",
        "tested fraction",
        "bad tests avoided",
    )

    test_costs = parse_float_list(args.test_costs)
    compute_saved = {}
    for cost in test_costs:
        compute_saved[str(cost)] = [float(v * cost) for v in tests_avoided_vals]

    # Store compute saved for the canonical fractions
    top_fracs = [0.01, 0.02, 0.10]
    top_summary = {}
    for frac in top_fracs:
        k = max(1, int(n * frac))
        tested = queue[:k]
        survival_top = sum(r["survive_Q1"] for r in tested) / k
        enrich = survival_top / overall_survival if overall_survival > 0 else 0.0
        tail = queue[k:]
        bad_in_tail = sum(1 - r["survive_Q1"] for r in tail)
        top_summary[str(frac)] = {
            "enrichment": enrich,
            "bad_tests_avoided": bad_in_tail,
            "compute_saved": {str(cost): bad_in_tail * cost for cost in test_costs},
        }

    summary = {
        "p_max": args.p_max,
        "Q0": args.Q0,
        "Q1": args.Q1,
        "n_primes": n,
        "overall_survival": overall_survival,
        "fractions": list(map(float, fractions)),
        "enrichment": list(map(float, enrich_vals)),
        "tests_avoided": list(map(float, tests_avoided_vals)),
        "compute_saved": compute_saved,
        "top_summary": top_summary,
    }

    (out_dir / "m22_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"OK: wrote M22 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
