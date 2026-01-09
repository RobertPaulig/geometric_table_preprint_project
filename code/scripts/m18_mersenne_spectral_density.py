#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-min", type=int, default=2)
    p.add_argument("--p-max", type=int, default=5000)
    p.add_argument("--p-mode", type=str, default="prime", choices=["all", "prime"])
    p.add_argument("--Q0", type=int, default=5000)
    p.add_argument("--Q1", type=int, default=20000)
    p.add_argument("--weight", type=str, default="inv_q",
                   choices=["ones", "inv_q", "inv_logq", "logq"])
    p.add_argument("--smooth-window", type=int, default=0)
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


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.copy()
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(values, kernel, mode="same")


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


def iter_multiples(p_min: int, p_max: int, d: int) -> Iterable[int]:
    start = ((p_min + d - 1) // d) * d
    return range(start, p_max + 1, d)


def save_heatmap(path: Path, mat: np.ndarray, title: str, xlabel: str, ylabel: str,
                 xticklabels: List[str], yticklabels: List[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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

    if args.p_min > args.p_max:
        raise ValueError("p-min must be <= p-max")
    if args.Q0 > args.Q1:
        raise ValueError("Q0 must be <= Q1")

    primes_q = sieve_primes(args.Q1)
    primes_q = [q for q in primes_q if q != 2]
    primes_for_factor = sieve_primes(int(math.isqrt(args.Q1)) + 1)

    ords: Dict[int, int] = {}
    for q in primes_q:
        ords[q] = ord_mod_2(q, primes_for_factor)

    # Build weights by period for Q0
    weight_by_d: Dict[int, float] = {}
    for q in primes_q:
        if q > args.Q0:
            break
        d = ords[q]
        weight_by_d[d] = weight_by_d.get(d, 0.0) + weight_fn(q, args.weight)

    p_min = int(args.p_min)
    p_max = int(args.p_max)
    n = p_max - p_min + 1
    scores = np.zeros(n, dtype=float)

    for d, w in weight_by_d.items():
        for p in iter_multiples(p_min, p_max, d):
            scores[p - p_min] += w

    smooth = rolling_mean(scores, args.smooth_window)

    # Survival against Q1
    survive = np.ones(n, dtype=np.uint8)
    d_set_q1 = set(ords[q] for q in primes_q)
    for d in d_set_q1:
        for p in iter_multiples(p_min, p_max, d):
            survive[p - p_min] = 0

    # Prime mask for p_mode filtering
    primes_p = set(sieve_primes(p_max))
    p_vals = np.arange(p_min, p_max + 1, dtype=int)
    p_is_prime = np.array([1 if p in primes_p else 0 for p in p_vals], dtype=np.uint8)

    if args.p_mode == "prime":
        keep = p_is_prime == 1
    else:
        keep = np.ones_like(p_is_prime, dtype=bool)

    quietness = -scores[keep]
    survive_keep = survive[keep]
    auc = auc_score(survive_keep, quietness)

    overall_survival = float(survive_keep.mean()) if len(survive_keep) else 0.0

    # Enrichment curve
    order = np.argsort(quietness)[::-1]
    total = len(order)
    enrich_x = []
    enrich_y = []
    if total > 0 and overall_survival > 0:
        for frac in np.linspace(0.02, 0.5, 25):
            k = max(1, int(total * frac))
            top_survive = survive_keep[order[:k]].mean()
            enrich_x.append(frac)
            enrich_y.append(top_survive / overall_survival)

    # Score vs survival bins
    n_bins = 20
    if total > 0:
        bins = np.linspace(quietness.min(), quietness.max(), n_bins + 1)
        bin_means = []
        bin_centers = []
        for i in range(n_bins):
            mask = (quietness >= bins[i]) & (quietness < bins[i + 1])
            if mask.any():
                bin_means.append(survive_keep[mask].mean())
                bin_centers.append((bins[i] + bins[i + 1]) / 2.0)

    # Heatmap for top periods
    top_periods = sorted(weight_by_d.items(), key=lambda x: (-x[1], x[0]))[:20]
    p_bins = 120
    bin_edges = np.linspace(p_min, p_max + 1, p_bins + 1, dtype=int)
    heat = np.zeros((len(top_periods), p_bins), dtype=float)
    for i, (d, _w) in enumerate(top_periods):
        hits = np.zeros(n, dtype=np.uint8)
        for p in iter_multiples(p_min, p_max, d):
            hits[p - p_min] = 1
        for b in range(p_bins):
            start = bin_edges[b] - p_min
            end = bin_edges[b + 1] - p_min
            if end > start:
                heat[i, b] = hits[start:end].mean()

    save_heatmap(
        out_dir / "m18_density_heatmap.png",
        heat,
        "M18 spectral kill density (top periods)",
        "p bins",
        "period d",
        [str(int(x)) for x in bin_edges[:: max(1, p_bins // 6)]],
        [str(d) for d, _w in top_periods],
    )

    if total > 0 and bin_centers:
        save_line_plot(
            out_dir / "m18_score_vs_survival.png",
            bin_centers,
            bin_means,
            "Survival vs quietness bins (Q1)",
            "quietness bin center (-score)",
            "survival rate",
        )

    if enrich_x:
        save_line_plot(
            out_dir / "m18_enrichment_curve.png",
            enrich_x,
            enrich_y,
            "Enrichment@k vs top fraction (Q1)",
            "top fraction of p (quietest)",
            "enrichment",
        )

    score_csv = out_dir / "m18_score.csv"
    with score_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "score", "smooth_score", "survive_Q1", "is_prime"])
        for p, s, sm, surv, isp in zip(p_vals, scores, smooth, survive, p_is_prime):
            if args.p_mode == "prime" and isp == 0:
                continue
            w.writerow([p, f"{s:.6g}", f"{sm:.6g}", int(surv), int(isp)])

    top_csv = out_dir / "m18_top_periods.csv"
    with top_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["period_d", "weight_mass"])
        for d, wgt in top_periods:
            w.writerow([d, f"{wgt:.6g}"])

    top_json = out_dir / "m18_top_periods.json"
    top_json.write_text(json.dumps({
        "Q0": args.Q0,
        "Q1": args.Q1,
        "weight": args.weight,
        "top_periods": [{"d": d, "weight": wgt} for d, wgt in top_periods],
    }, indent=2), encoding="utf-8")

    summary = {
        "p_min": p_min,
        "p_max": p_max,
        "p_mode": args.p_mode,
        "Q0": args.Q0,
        "Q1": args.Q1,
        "weight": args.weight,
        "smooth_window": args.smooth_window,
        "overall_survival_rate": overall_survival,
        "auc": auc,
        "enrichment_curve": {"fractions": enrich_x, "values": enrich_y},
    }
    (out_dir / "m18_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"OK: wrote M18 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
