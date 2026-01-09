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
    p.add_argument("--p-max", type=int, default=50000)
    p.add_argument("--Q0", type=int, default=50000)
    p.add_argument("--Q1", type=int, default=200000)
    p.add_argument("--weights", type=str, required=True)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_str_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


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


def weight_fn(q: int, d: int, mode: str, rand_pos: Dict[int, float],
              rand_sign: Dict[int, float]) -> float:
    if mode == "ones":
        return 1.0
    if mode == "inv_q":
        return 1.0 / q
    if mode == "inv_logq":
        return 1.0 / math.log(q)
    if mode == "logq":
        return math.log(q)
    if mode == "q":
        return float(q)
    if mode == "inv_d":
        return 1.0 / d
    if mode == "inv_logd":
        return 1.0 / math.log(d)
    if mode == "logd":
        return math.log(d)
    if mode == "rand_pos":
        return rand_pos[q]
    if mode == "rand_sign":
        return rand_sign[q]
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


def save_bar_plot(path: Path, labels: List[str], values: List[float],
                  title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(range(len(values)), values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    weights = parse_str_list(args.weights)
    primes_q = sieve_primes(args.Q1)
    primes_q = [q for q in primes_q if q != 2]
    primes_for_factor = sieve_primes(int(math.isqrt(args.Q1)) + 1)

    ords: Dict[int, int] = {}
    for q in primes_q:
        ords[q] = ord_mod_2(q, primes_for_factor)

    p_min = 2
    p_max = args.p_max
    p_vals = np.arange(p_min, p_max + 1, dtype=int)
    primes_p = set(sieve_primes(p_max))
    p_is_prime = np.array([1 if p in primes_p else 0 for p in p_vals], dtype=np.uint8)
    p_is_comp = np.array([1 if p >= 4 and p not in primes_p else 0 for p in p_vals], dtype=np.uint8)

    d_set = set(ords[q] for q in primes_q if q <= args.Q1)
    survive = np.ones(len(p_vals), dtype=np.uint8)
    for d in d_set:
        for p in iter_multiples(p_min, p_max, d):
            survive[p - p_min] = 0

    subsets = {
        "all": np.ones_like(p_is_prime, dtype=bool),
        "prime": p_is_prime == 1,
        "composite": p_is_comp == 1,
    }

    rng = np.random.default_rng(args.seed)
    rand_pos = {q: float(rng.random()) for q in primes_q}
    rand_sign = {q: (1.0 if rng.random() < 0.5 else -1.0) for q in primes_q}

    rows = []
    for mode in weights:
        weight_by_d: Dict[int, float] = {}
        for q in primes_q:
            if q > args.Q0:
                break
            d = ords[q]
            weight_by_d[d] = weight_by_d.get(d, 0.0) + weight_fn(q, d, mode, rand_pos, rand_sign)

        scores = np.zeros(len(p_vals), dtype=float)
        for d, w in weight_by_d.items():
            for p in iter_multiples(p_min, p_max, d):
                scores[p - p_min] += w

        for subset_name, mask in subsets.items():
            quietness = -scores[mask]
            survive_keep = survive[mask]
            auc = auc_score(survive_keep, quietness)
            overall_survival = float(survive_keep.mean()) if len(survive_keep) else 0.0

            order = np.argsort(quietness)[::-1]
            k2 = max(1, int(len(order) * 0.02))
            k10 = max(1, int(len(order) * 0.10))
            top_survival_2 = float(survive_keep[order[:k2]].mean()) if len(order) else 0.0
            top_survival_10 = float(survive_keep[order[:k10]].mean()) if len(order) else 0.0
            enrich_2 = top_survival_2 / overall_survival if overall_survival > 0 else 0.0
            enrich_10 = top_survival_10 / overall_survival if overall_survival > 0 else 0.0

            rows.append({
                "weight": mode,
                "subset": subset_name,
                "overall_survival": overall_survival,
                "auc": auc,
                "enrichment_2": enrich_2,
                "enrichment_10": enrich_10,
            })

    summary_csv = out_dir / "m20b_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary_json = out_dir / "m20b_summary.json"
    summary_json.write_text(json.dumps({
        "p_max": p_max,
        "Q0": args.Q0,
        "Q1": args.Q1,
        "weights": weights,
        "rows": rows,
    }, indent=2), encoding="utf-8")

    for subset in ["all", "prime", "composite"]:
        subset_rows = [r for r in rows if r["subset"] == subset]
        labels = [r["weight"] for r in subset_rows]
        auc_vals = [r["auc"] for r in subset_rows]
        enr_vals = [r["enrichment_10"] for r in subset_rows]

        save_bar_plot(
            out_dir / f"m20b_auc_by_weight_{subset}.png",
            labels,
            auc_vals,
            f"M20b AUC by weight ({subset})",
            "AUC",
        )

        save_bar_plot(
            out_dir / f"m20b_enrichment10_by_weight_{subset}.png",
            labels,
            enr_vals,
            f"M20b enrichment@10% by weight ({subset})",
            "enrichment@10%",
        )

        # Score -> survival bins
        for row in subset_rows:
            if row["weight"] == "inv_q":
                break
        else:
            row = subset_rows[0]
        mode = row["weight"]

        weight_by_d: Dict[int, float] = {}
        for q in primes_q:
            if q > args.Q0:
                break
            d = ords[q]
            weight_by_d[d] = weight_by_d.get(d, 0.0) + weight_fn(q, d, mode, rand_pos, rand_sign)

        scores = np.zeros(len(p_vals), dtype=float)
        for d, w in weight_by_d.items():
            for p in iter_multiples(p_min, p_max, d):
                scores[p - p_min] += w

        mask = subsets[subset]
        quietness = -scores[mask]
        survive_keep = survive[mask]
        if len(quietness) > 0:
            bins = np.linspace(quietness.min(), quietness.max(), 21)
            bin_centers = []
            bin_means = []
            for i in range(len(bins) - 1):
                bmask = (quietness >= bins[i]) & (quietness < bins[i + 1])
                if bmask.any():
                    bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
                    bin_means.append(float(survive_keep[bmask].mean()))
            if bin_centers:
                save_line_plot(
                    out_dir / f"m20b_score_vs_survival_bins_{subset}.png",
                    bin_centers,
                    bin_means,
                    f"M20b score vs survival bins ({subset})",
                    "quietness bin center (-score)",
                    "survival rate",
                )

    notes = (
        "Prime subset is degenerate for ord_q(2)|p because p is prime: hits require ord_q(2)=p, "
        "so weights do not reorder survivors. Composite p allows multiple periods per p, so weights "
        "change scores and ranking."
    )
    (out_dir / "m20b_notes.txt").write_text(notes, encoding="utf-8")

    print(f"OK: wrote M20b artifacts to {out_dir}")


if __name__ == "__main__":
    main()
