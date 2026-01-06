# code/scripts/matched_twins.py
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from geometric_table import compute_core_metrics_fast, compute_core_edges_only


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


def paired_permutation_pvalue(deltas: List[float], iters: int, seed: int) -> float:
    if not deltas:
        return 1.0
    rng = random.Random(seed)
    obs = abs(sum(deltas) / len(deltas))
    count = 0
    for _ in range(iters):
        s = 0.0
        for d in deltas:
            s += d if rng.random() < 0.5 else -d
        diff = abs(s / len(deltas))
        if diff >= obs:
            count += 1
    return (count + 1) / (iters + 1)


def sign_test(deltas: List[float]) -> float:
    pos = sum(1 for d in deltas if d > 0)
    n = len(deltas)
    if n == 0:
        return 1.0
    # two-sided binomial test with p=0.5
    from math import comb
    k = min(pos, n - pos)
    p = 0.0
    for i in range(0, k + 1):
        p += comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * p)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--center-min", type=int, default=500)
    p.add_argument("--center-max", type=int, default=200000)
    p.add_argument("--core-r", type=int, default=30)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--primitive", action="store_true", default=True)
    p.add_argument("--weight", choices=["ones", "idf"], default="ones")
    p.add_argument("--max-d", type=int, default=50)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--out-csv", type=str, default="out/matched_pairs_six_core30.csv")
    p.add_argument("--out-json", type=str, default="out/matched_analysis_six_core30.json")
    p.add_argument("--out-fig-gap", type=str, default="fig/matched_delta_gap_six.png")
    p.add_argument("--out-fig-entropy", type=str, default="fig/matched_delta_entropy_six.png")
    args = p.parse_args()

    rng = random.Random(args.seed)
    twins = []
    start = int(math.ceil(args.center_min / 6))
    end = int(math.floor(args.center_max / 6))
    for m in range(start, end + 1):
        c = 6 * m
        if is_prime(c - 1) and is_prime(c + 1):
            twins.append(c)

    pairs = []
    edges_cache: Dict[int, int] = {}
    metrics_cache: Dict[int, Dict[str, float]] = {}
    for c in twins:
        if c in metrics_cache:
            twin_metrics = metrics_cache[c]
        else:
            twin_metrics = compute_core_metrics_fast(
                center=c,
                core_r=args.core_r,
                K=args.K,
                primitive=bool(args.primitive),
                weight=args.weight,
                eps=1e-12,
            )
            metrics_cache[c] = twin_metrics
        twin_edges = twin_metrics["core_edges"]
        best = None
        for d in range(-args.max_d, args.max_d + 1):
            if d == 0:
                continue
            ctrl = c + 6 * d
            if ctrl < args.center_min or ctrl > args.center_max:
                continue
            if is_prime(ctrl - 1) and is_prime(ctrl + 1):
                continue
            if ctrl in edges_cache:
                ctrl_edges = edges_cache[ctrl]
            else:
                ctrl_edges = compute_core_edges_only(
                    center=ctrl,
                    core_r=args.core_r,
                    K=args.K,
                    primitive=bool(args.primitive),
                    weight=args.weight,
                    eps=1e-12,
                )
                edges_cache[ctrl] = ctrl_edges
            diff = abs(twin_edges - ctrl_edges)
            cand = (diff, ctrl, ctrl_edges)
            if best is None or cand[0] < best[0]:
                best = cand
        if best is None:
            continue
        _, ctrl, _ = best
        if ctrl in metrics_cache:
            ctrl_metrics = metrics_cache[ctrl]
        else:
            ctrl_metrics = compute_core_metrics_fast(
                center=ctrl,
                core_r=args.core_r,
                K=args.K,
                primitive=bool(args.primitive),
                weight=args.weight,
                eps=1e-12,
            )
            metrics_cache[ctrl] = ctrl_metrics
        pairs.append({
            "twin_center": c,
            "control_center": ctrl,
            "twin_gap": twin_metrics["core_gc_spectral_gap"],
            "control_gap": ctrl_metrics["core_gc_spectral_gap"],
            "twin_entropy": twin_metrics["core_gc_entropy"],
            "control_entropy": ctrl_metrics["core_gc_entropy"],
            "twin_edges": twin_edges,
            "control_edges": ctrl_metrics["core_edges"],
        })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()) if pairs else [])
        if pairs:
            w.writeheader()
            w.writerows(pairs)

    deltas_gap = [p["twin_gap"] - p["control_gap"] for p in pairs]
    deltas_entropy = [p["twin_entropy"] - p["control_entropy"] for p in pairs]

    report = {
        "n_pairs": len(pairs),
        "mean_delta_gap": float(sum(deltas_gap) / len(deltas_gap)) if deltas_gap else 0.0,
        "mean_delta_entropy": float(sum(deltas_entropy) / len(deltas_entropy)) if deltas_entropy else 0.0,
        "perm_p_delta_gap": paired_permutation_pvalue(deltas_gap, args.iters, args.seed),
        "perm_p_delta_entropy": paired_permutation_pvalue(deltas_entropy, args.iters, args.seed + 1),
        "sign_test_p_gap": sign_test(deltas_gap),
        "fraction_gap_positive": float(sum(1 for d in deltas_gap if d > 0) / len(deltas_gap)) if deltas_gap else 0.0,
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Simple histograms
    import matplotlib.pyplot as plt

    Path(args.out_fig_gap).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.hist(deltas_gap, bins=20, color="steelblue", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Matched Δ core gap (twin - control)")
    ax.set_xlabel("delta_gap")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.out_fig_gap, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.hist(deltas_entropy, bins=20, color="darkorange", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Matched Δ core entropy (twin - control)")
    ax.set_xlabel("delta_entropy")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.out_fig_entropy, dpi=150)
    plt.close(fig)

    print(f"OK: wrote {out_csv}, {args.out_json}, figures")


if __name__ == "__main__":
    main()
