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
    p.add_argument("--p-max-list", type=str, required=True)
    p.add_argument("--p-mode", type=str, default="prime", choices=["all", "prime"])
    p.add_argument("--pairs", type=str, required=True)
    p.add_argument("--weight", type=str, default="inv_q",
                   choices=["ones", "inv_q", "inv_logq", "logq"])
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_pairs(raw: str) -> List[Tuple[int, int]]:
    pairs = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split(":")
        if len(parts) != 2:
            raise ValueError(f"Invalid pair: {chunk}")
        pairs.append((int(parts[0]), int(parts[1])))
    return pairs


def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p: n + 1: p] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def factorize(n: int, primes: List[int]) -> List[Tuple[int, int]]:
    factors: List[Tuple[int, int]] = []
    x = n
    for p in primes:
        if p * p > x:
            break
        if x % p == 0:
            exp = 0
            while x % p == 0:
                x //= p
                exp += 1
            factors.append((p, exp))
    if x > 1:
        factors.append((x, 1))
    return factors


def ord_mod_2(q: int, primes: List[int]) -> int:
    if q <= 2:
        return 1
    factors = factorize(q - 1, primes)
    d = q - 1
    for p, _exp in factors:
        while d % p == 0 and pow(2, d // p, q) == 1:
            d //= p
    return d


def weight_fn(q: int, mode: str) -> float:
    if mode == "ones":
        return 1.0
    if mode == "inv_q":
        return 1.0 / q
    if mode == "inv_logq":
        return 1.0 / math.log(q)
    if mode == "logq":
        return math.log(q)
    raise ValueError(f"Unknown weight mode: {mode}")


def rankdata_avg_ties(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=float)
    n = len(values)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    ranks = rankdata_avg_ties(scores)
    sum_ranks_pos = ranks[pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def iter_multiples(p_min: int, p_max: int, d: int) -> range:
    start = ((p_min + d - 1) // d) * d
    return range(start, p_max + 1, d)


def save_line_plot(path: Path, xs: List[int], ys_list: List[List[float]], labels: List[str],
                   title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for ys, label in zip(ys_list, labels):
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    p_max_list = parse_int_list(args.p_max_list)
    pairs = parse_pairs(args.pairs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q1_max = max(q1 for _q0, q1 in pairs)
    primes_q = sieve_primes(q1_max)
    primes_q = [q for q in primes_q if q != 2]
    primes_for_factor = sieve_primes(int(math.isqrt(q1_max)) + 1)

    ords: Dict[int, int] = {}
    for q in primes_q:
        ords[q] = ord_mod_2(q, primes_for_factor)

    primes_p = set(sieve_primes(max(p_max_list)))

    rows = []
    for p_max in p_max_list:
        p_min = 2
        p_vals = np.arange(p_min, p_max + 1, dtype=int)
        p_is_prime = np.array([1 if p in primes_p else 0 for p in p_vals], dtype=np.uint8)
        keep = p_is_prime == 1 if args.p_mode == "prime" else np.ones_like(p_is_prime, dtype=bool)

        for q0, q1 in pairs:
            weight_by_d: Dict[int, float] = {}
            for q in primes_q:
                if q > q0:
                    break
                d = ords[q]
                weight_by_d[d] = weight_by_d.get(d, 0.0) + weight_fn(q, args.weight)

            scores = np.zeros(len(p_vals), dtype=float)
            for d, w in weight_by_d.items():
                for p in iter_multiples(p_min, p_max, d):
                    scores[p - p_min] += w

            survive = np.ones(len(p_vals), dtype=np.uint8)
            d_set = set(ords[q] for q in primes_q if q <= q1)
            for d in d_set:
                for p in iter_multiples(p_min, p_max, d):
                    survive[p - p_min] = 0

            quietness = -scores[keep]
            survive_keep = survive[keep]
            auc = auc_score(survive_keep, quietness)
            overall_survival = float(survive_keep.mean()) if len(survive_keep) else 0.0

            order = np.argsort(quietness)[::-1]
            k = max(1, int(len(order) * 0.1))
            top_survival = float(survive_keep[order[:k]].mean()) if len(order) else 0.0
            enrichment_10 = top_survival / overall_survival if overall_survival > 0 else 0.0

            rows.append({
                "p_max": p_max,
                "Q0": q0,
                "Q1": q1,
                "weight": args.weight,
                "overall_survival": overall_survival,
                "auc": auc,
                "enrichment_10": enrichment_10,
            })

    summary_csv = out_dir / "m18b_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_json = out_dir / "m18b_summary.json"
    summary_json.write_text(json.dumps({
        "p_mode": args.p_mode,
        "p_max_list": p_max_list,
        "pairs": pairs,
        "weight": args.weight,
        "rows": rows,
    }, indent=2), encoding="utf-8")

    labels = [f"Q0={q0},Q1={q1}" for q0, q1 in pairs]
    auc_lines = [[] for _ in pairs]
    enr_lines = [[] for _ in pairs]
    for i, (q0, q1) in enumerate(pairs):
        for p_max in p_max_list:
            row = next(r for r in rows if r["p_max"] == p_max and r["Q0"] == q0 and r["Q1"] == q1)
            auc_lines[i].append(row["auc"])
            enr_lines[i].append(row["enrichment_10"])

    save_line_plot(
        out_dir / "m18b_auc_vs_pmax.png",
        p_max_list,
        auc_lines,
        labels,
        "M18b AUC vs p_max",
        "p_max",
        "AUC",
    )

    save_line_plot(
        out_dir / "m18b_enrichment_vs_pmax.png",
        p_max_list,
        enr_lines,
        labels,
        "M18b enrichment@10% vs p_max",
        "p_max",
        "enrichment@10%",
    )

    print(f"OK: wrote M18b artifacts to {out_dir}")


if __name__ == "__main__":
    main()
