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
    p.add_argument("--out-dir", type=str, required=True)
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


def save_bar_plot(path: Path, labels: List[str], values: List[float],
                  title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.bar(range(len(values)), values, color="#4C72B0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
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

    ap_count_q0: Dict[int, int] = {p: 0 for p in primes_p}
    ap_count_delta: Dict[int, int] = {p: 0 for p in primes_p}
    ap_harm_delta: Dict[int, float] = {p: 0.0 for p in primes_p}
    min_q_delta: Dict[int, int] = {p: 0 for p in primes_p}

    for q in primes_q:
        q_factors = factorize(q - 1, primes_factor)
        for p in q_factors:
            if p not in primes_p_set:
                continue
            if q <= args.Q0:
                ap_count_q0[p] += 1
            else:
                ap_count_delta[p] += 1
                ap_harm_delta[p] += p / (q - 1)
                if min_q_delta[p] == 0 or q < min_q_delta[p]:
                    min_q_delta[p] = q

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
        kq0 = killed_q0[p]
        kq1 = killed_q1[p]
        death_later = 1 if (kq1 == 1 and kq0 == 0) else 0
        row = {
            "p": p,
            "killed_Q0": kq0,
            "killed_Q1": kq1,
            "death_later": death_later,
            "ap_count_Q0": ap_count_q0[p],
            "ap_count_delta": ap_count_delta[p],
            "ap_harm_delta": ap_harm_delta[p],
            "min_q_delta": min_q_delta[p],
            "logp": math.log(p),
        }
        rows.append(row)

    dataset_csv = out_dir / "m21_dataset.csv"
    with dataset_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    z_rows = [r for r in rows if r["killed_Q0"] == 0]
    y = np.array([r["death_later"] for r in z_rows], dtype=np.uint8)
    hazard = np.array([r["ap_harm_delta"] for r in z_rows], dtype=float)

    auc = auc_score(y, hazard)
    base_rate = float(y.mean()) if len(y) else 0.0

    order = np.argsort(hazard)[::-1]
    def enrich(frac: float) -> float:
        k = max(1, int(len(order) * frac))
        top = float(y[order[:k]].mean()) if len(order) else 0.0
        return top / base_rate if base_rate > 0 else 0.0

    enrich_1 = enrich(0.01)
    enrich_2 = enrich(0.02)
    enrich_10 = enrich(0.10)

    # score -> survival bins
    if len(hazard) > 0:
        bins = np.linspace(hazard.min(), hazard.max(), 21)
        bin_centers = []
        bin_means = []
        for i in range(len(bins) - 1):
            mask = (hazard >= bins[i]) & (hazard < bins[i + 1])
            if mask.any():
                bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
                bin_means.append(float(y[mask].mean()))
        if bin_centers:
            save_line_plot(
                out_dir / "m21_hazard_vs_death_bins.png",
                bin_centers,
                bin_means,
                "Hazard score vs death_later bins (M21)",
                "hazard bin center",
                "P(death_later)",
            )

    # enrichment curve
    fractions = np.linspace(0.02, 0.5, 25)
    enrich_vals = [enrich(float(frac)) for frac in fractions]
    save_line_plot(
        out_dir / "m21_enrichment_curve.png",
        list(fractions),
        enrich_vals,
        "Enrichment@k for death_later (M21)",
        "top fraction by hazard",
        "enrichment",
    )

    # feature importance (absolute corr)
    features = ["ap_count_delta", "ap_harm_delta", "min_q_delta", "logp"]
    corrs = []
    for feat in features:
        x = np.array([r[feat] for r in z_rows], dtype=float)
        if x.std() == 0 or y.std() == 0:
            corr = 0.0
        else:
            corr = float(np.corrcoef(x, y)[0, 1])
        corrs.append(abs(corr))
    save_bar_plot(
        out_dir / "m21_feature_importance.png",
        features,
        corrs,
        "Feature importance (|corr| with death_later)",
        "|corr|",
    )

    # scatter: logp vs hazard
    if len(hazard) > 0:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.scatter([r["logp"] for r in z_rows], hazard, s=8, alpha=0.6)
        ax.set_title("log(p) vs hazard (M21)")
        ax.set_xlabel("log(p)")
        ax.set_ylabel("hazard score")
        fig.tight_layout()
        fig.savefig(out_dir / "m21_scatter_logp_vs_hazard.png", dpi=150)
        plt.close(fig)

    summary = {
        "p_max": args.p_max,
        "Q0": args.Q0,
        "Q1": args.Q1,
        "n_primes": len(primes_p),
        "n_Z": len(z_rows),
        "death_later_rate": base_rate,
        "auc": auc,
        "enrichment_1": enrich_1,
        "enrichment_2": enrich_2,
        "enrichment_10": enrich_10,
    }

    summary_json = out_dir / "m21_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    summary_csv = out_dir / "m21_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary.keys()))
        w.writeheader()
        w.writerow(summary)

    print(f"OK: wrote M21 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
