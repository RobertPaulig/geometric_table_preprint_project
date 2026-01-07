# code/scripts/matched_twins.py
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

from geometric_table import (
    compute_core_metrics_fast,
    compute_core_edges_only,
    compute_core_gc_size_fast,
    choose_hybrid_K,
)


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


def sign_test_with_ties(deltas: List[float], half_ties: bool) -> Tuple[float, float]:
    """
    Return (p_value, frac_positive) with optional half-weight for ties.
    """
    n_total = len(deltas)
    if n_total == 0:
        return 1.0, 0.0
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    ties = n_total - pos - neg
    if half_ties:
        pos_eff = pos + 0.5 * ties
        n_eff = pos + neg + ties
        frac_pos = pos_eff / n_eff if n_eff else 0.0
        # approximate using binomial with adjusted successes
        # use normal approx for speed
        p_hat = 0.5
        mean = n_eff * p_hat
        var = n_eff * p_hat * (1 - p_hat)
        z = 0.0 if var == 0 else abs(pos_eff - mean) / (var ** 0.5)
        from math import erf, sqrt
        p_two = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
        p_two = min(1.0, max(0.0, p_two))
        return float(p_two), float(frac_pos)
    else:
        # exclude ties
        n_eff = pos + neg
        if n_eff == 0:
            return 1.0, 0.5
        frac_pos = pos / n_eff
        p_val = sign_test([d for d in deltas if d != 0])
        return float(p_val), float(frac_pos)


def median(vals: List[float]) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return 0.5 * (s[mid - 1] + s[mid])
    return s[mid]


def bootstrap_median(vals: List[float], iters: int, seed: int) -> Tuple[float, float]:
    rng = random.Random(seed)
    if not vals:
        return float("nan"), float("nan")
    medians = []
    for _ in range(iters):
        sample = [vals[rng.randrange(0, len(vals))] for _ in range(len(vals))]
        medians.append(median(sample))
    medians.sort()
    lo = medians[int(0.025 * len(medians))]
    hi = medians[int(0.975 * len(medians)) - 1]
    return lo, hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--center-min", type=int, default=500)
    p.add_argument("--center-max", type=int, default=200000)
    p.add_argument("--core-r", type=int, default=30)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--primitive", action="store_true", default=True)
    p.add_argument("--weight", choices=["ones", "idf"], default="ones")
    p.add_argument("--max-d", type=int, default=50)
    p.add_argument("--max-d-strict", type=int, default=200, help="expanded search window for strict matching")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--alpha", type=float, default=1.0, help="scale-law coefficient for K0")
    p.add_argument("--growth", type=float, default=1.5, help="multiplicative bump for K")
    p.add_argument("--max-bumps", type=int, default=6, help="max auto-tuning iterations")
    p.add_argument("--min-gc-size", type=int, default=10, help="target core GC size")
    p.add_argument("--k-max", type=int, default=20000, help="upper cap for K")
    p.add_argument("--delta-logK-tol", type=float, default=0.05, help="strict: |delta log K| tolerance")
    p.add_argument("--delta-edges-tol", type=float, default=10.0, help="strict: absolute edge diff tolerance")
    p.add_argument("--delta-edges-rel-tol", type=float, default=0.02, help="strict: relative edge diff tolerance")
    p.add_argument("--delta-frac-tol", type=float, default=0.05, help="strict: gc fraction diff tolerance")
    p.add_argument("--centers-csv", type=str, default="")
    p.add_argument("--restrict-to-csv", action="store_true", default=False)
    p.add_argument("--out-csv", type=str, default="out/matched_pairs_six_core30.csv")
    p.add_argument("--out-json", type=str, default="out/matched_analysis_six_core30.json")
    p.add_argument("--out-fig-gap", type=str, default="fig/matched_delta_gap_six.png")
    p.add_argument("--out-fig-entropy", type=str, default="fig/matched_delta_entropy_six.png")
    args = p.parse_args()

    rng = random.Random(args.seed)
    twins = []
    csv_centers = set()
    if args.centers_csv:
        path = Path(args.centers_csv)
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                for row in r:
                    try:
                        c = int(float(row["center"]))
                    except Exception:
                        continue
                    if c < args.center_min or c > args.center_max:
                        continue
                    csv_centers.add(c)
                    if is_prime(c - 1) and is_prime(c + 1):
                        twins.append(c)
        else:
            raise FileNotFoundError(path)
    else:
        start = int(math.ceil(args.center_min / 6))
        end = int(math.floor(args.center_max / 6))
        for m in range(start, end + 1):
            c = 6 * m
            if is_prime(c - 1) and is_prime(c + 1):
                twins.append(c)

    pairs = []
    metrics_cache: Dict[int, Dict[str, float]] = {}
    k_cache: Dict[int, Tuple[int, int, int, int]] = {}  # center -> (K0, K_used, k_bumps, hit_kmax)

    def get_metrics(center: int) -> Tuple[Dict[str, float], int, int, int, int]:
        if center in metrics_cache:
            m = metrics_cache[center]
            K0, K_used, k_bumps, hit_kmax = k_cache[center]
            return m, K0, K_used, k_bumps, hit_kmax
        K0, K_used, k_bumps, hit_kmax, _ = choose_hybrid_K(
            center=center,
            core_r=args.core_r,
            primitive=bool(args.primitive),
            weight=args.weight,
            alpha=args.alpha,
            growth=args.growth,
            min_core_gc=args.min_gc_size,
            max_bumps=args.max_bumps,
            K_max=args.k_max,
            eps=1e-12,
        )
        m = compute_core_metrics_fast(
            center=center,
            core_r=args.core_r,
            K=K_used,
            primitive=bool(args.primitive),
            weight=args.weight,
            eps=1e-12,
        )
        metrics_cache[center] = m
        k_cache[center] = (K0, K_used, k_bumps, int(hit_kmax))
        return m, K0, K_used, k_bumps, int(hit_kmax)

    for c in twins:
        twin_metrics, twin_k0, twin_k, twin_k_bumps, twin_hit_kmax = get_metrics(c)
        twin_edges = twin_metrics["core_edges"]
        candidates = []
        for d in range(-args.max_d_strict, args.max_d_strict + 1):
            if d == 0:
                continue
            ctrl = c + 6 * d
            if ctrl < args.center_min or ctrl > args.center_max:
                continue
            if args.restrict_to_csv and csv_centers and ctrl not in csv_centers:
                continue
            if is_prime(ctrl - 1) and is_prime(ctrl + 1):
                continue
            ctrl_metrics, ctrl_k0, ctrl_k, ctrl_k_bumps, ctrl_hit_kmax = get_metrics(ctrl)
            cand = {
                "ctrl": ctrl,
                "ctrl_metrics": ctrl_metrics,
                "ctrl_k0": ctrl_k0,
                "ctrl_k": ctrl_k,
                "ctrl_k_bumps": ctrl_k_bumps,
                "ctrl_hit_kmax": ctrl_hit_kmax,
                "d": d,
            }
            candidates.append(cand)

        if not candidates:
            continue

        # raw: best by edge diff within primary window
        raw_candidates = [cnd for cnd in candidates if abs(cnd["d"]) <= args.max_d]
        if raw_candidates:
            raw_best = min(raw_candidates, key=lambda cnd: abs(twin_edges - cnd["ctrl_metrics"]["core_edges"]))
        else:
            raw_best = min(candidates, key=lambda cnd: abs(twin_edges - cnd["ctrl_metrics"]["core_edges"]))

        # strict: enforce delta tolerances, allow up to max_d_strict
        strict_candidates = []
        for cnd in candidates:
            ctrl_edges = cnd["ctrl_metrics"]["core_edges"]
            ctrl_k = cnd["ctrl_k"]
            ctrl_frac = cnd["ctrl_metrics"]["core_gc_fraction"]
            delta_edges = twin_edges - ctrl_edges
            rel_ok = abs(delta_edges) <= max(args.delta_edges_tol, args.delta_edges_rel_tol * max(1.0, twin_edges))
            delta_logK = math.log(max(1.0, twin_k)) - math.log(max(1.0, ctrl_k))
            delta_frac = twin_metrics["core_gc_fraction"] - ctrl_frac
            edges_ok = rel_ok  # относительный допуск с минимальным порогом
            if abs(delta_logK) <= args.delta_logK_tol and edges_ok and abs(delta_frac) <= args.delta_frac_tol:
                strict_candidates.append(cnd)
        strict_best = None
        if strict_candidates:
            strict_best = min(strict_candidates, key=lambda cnd: abs(twin_edges - cnd["ctrl_metrics"]["core_edges"]))

        def build_pair(selected, strict_flag: bool) -> Dict[str, float]:
            ctrl_metrics = selected["ctrl_metrics"]
            ctrl_k = selected["ctrl_k"]
            delta_logK = math.log(max(1.0, twin_k)) - math.log(max(1.0, ctrl_k))
            delta_edges = twin_edges - ctrl_metrics["core_edges"]
            delta_frac = twin_metrics["core_gc_fraction"] - ctrl_metrics["core_gc_fraction"]
            return {
                "twin_center": c,
                "control_center": selected["ctrl"],
                "twin_gap": twin_metrics["core_gc_spectral_gap"],
                "control_gap": ctrl_metrics["core_gc_spectral_gap"],
                "twin_entropy": twin_metrics["core_gc_entropy"],
                "control_entropy": ctrl_metrics["core_gc_entropy"],
                "twin_edges": twin_edges,
                "control_edges": ctrl_metrics["core_edges"],
                "twin_gc_fraction": twin_metrics["core_gc_fraction"],
                "control_gc_fraction": ctrl_metrics["core_gc_fraction"],
                "twin_K0": twin_k0,
                "twin_K_used": twin_k,
                "twin_K_bumps": twin_k_bumps,
                "twin_hit_kmax": int(twin_hit_kmax),
                "control_K0": selected["ctrl_k0"],
                "control_K_used": ctrl_k,
                "control_K_bumps": selected["ctrl_k_bumps"],
                "control_hit_kmax": int(selected["ctrl_hit_kmax"]),
                "delta_K_used": twin_k - ctrl_k,
                "delta_logK": delta_logK,
                "delta_edges": delta_edges,
                "delta_gc_fraction": delta_frac,
                "d": selected["d"],
                "strict_pass": int(strict_flag),
            }

        pairs.append(build_pair(raw_best, strict_flag=False))
        if strict_best is not None:
            pairs.append(build_pair(strict_best, strict_flag=True))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pairs[0].keys()) if pairs else [])
        if pairs:
            w.writeheader()
            w.writerows(pairs)

    def stats(subset: List[Dict[str, float]], seed_offset: int) -> Dict[str, float]:
        deltas_gap = [p["twin_gap"] - p["control_gap"] for p in subset]
        deltas_entropy = [p["twin_entropy"] - p["control_entropy"] for p in subset]
        deltas_edges = [p["delta_edges"] for p in subset]
        mean_delta_gap = float(sum(deltas_gap) / len(deltas_gap)) if deltas_gap else 0.0
        mean_delta_entropy = float(sum(deltas_entropy) / len(deltas_entropy)) if deltas_entropy else 0.0
        frac_gap_positive = float(sum(1 for d in deltas_gap if d > 0) / len(deltas_gap)) if deltas_gap else 0.0
        med_gap = median(deltas_gap)
        med_ent = median(deltas_entropy)
        med_edges = median(deltas_edges)
        ci_gap_lo, ci_gap_hi = bootstrap_median(deltas_gap, iters=2000, seed=args.seed + seed_offset)
        ci_ent_lo, ci_ent_hi = bootstrap_median(deltas_entropy, iters=2000, seed=args.seed + seed_offset + 1)
        sanity_warning = ""
        if deltas_gap and abs(mean_delta_gap) < 1e-3 and frac_gap_positive < 0.1:
            sanity_warning = "warning: near-zero mean with very low positive fraction; inspect delta_gap calculation"
        return {
            "n_pairs": len(subset),
            "mean_delta_gap": mean_delta_gap,
            "mean_delta_entropy": mean_delta_entropy,
            "median_delta_gap": med_gap,
            "median_delta_entropy": med_ent,
            "delta_edges_min": float(min(deltas_edges)) if deltas_edges else float("nan"),
            "delta_edges_median": med_edges,
            "delta_edges_max": float(max(deltas_edges)) if deltas_edges else float("nan"),
            "median_delta_gap_ci": [ci_gap_lo, ci_gap_hi],
            "median_delta_entropy_ci": [ci_ent_lo, ci_ent_hi],
            "perm_p_delta_gap": paired_permutation_pvalue(deltas_gap, args.iters, args.seed + seed_offset),
            "perm_p_delta_entropy": paired_permutation_pvalue(deltas_entropy, args.iters, args.seed + seed_offset + 1),
            "sign_test_p_gap": sign_test(deltas_gap),
            "fraction_gap_positive": frac_gap_positive,
            "sanity_warning": sanity_warning,
        }

    raw_pairs = [p for p in pairs if not p["strict_pass"]]
    strict_pairs = [p for p in pairs if p["strict_pass"]]

    def sign_stats(subset: List[Dict[str, float]]) -> Dict[str, float]:
        deltas_gap = [p["twin_gap"] - p["control_gap"] for p in subset]
        p_excl, frac_excl = sign_test_with_ties(deltas_gap, half_ties=False)
        p_half, frac_half = sign_test_with_ties(deltas_gap, half_ties=True)
        ties = len(deltas_gap) - sum(1 for d in deltas_gap if d != 0)
        return {
            "n_pairs_total": len(deltas_gap),
            "n_pairs_nonzero": len(deltas_gap) - ties,
            "n_ties": ties,
            "sign_p_excluding_ties": p_excl,
            "frac_pos_excluding_ties": frac_excl,
            "sign_p_with_half_ties": p_half,
            "frac_pos_with_half_ties": frac_half,
        }

    report = {
        "n_pairs_raw": len(raw_pairs),
        "n_pairs_strict": len(strict_pairs),
        "raw": {**stats(raw_pairs, seed_offset=0), **sign_stats(raw_pairs)},
        "strict": {**stats(strict_pairs, seed_offset=10000), **sign_stats(strict_pairs)},
    }
    Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Simple histograms on raw pairs
    import matplotlib.pyplot as plt

    Path(args.out_fig_gap).parent.mkdir(parents=True, exist_ok=True)
    deltas_gap = [p["twin_gap"] - p["control_gap"] for p in raw_pairs]
    deltas_entropy = [p["twin_entropy"] - p["control_entropy"] for p in raw_pairs]
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.hist(deltas_gap, bins=20, color="steelblue", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Matched core gap (twin - control)")
    ax.set_xlabel("delta_gap")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.out_fig_gap, dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.hist(deltas_entropy, bins=20, color="darkorange", alpha=0.8)
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title("Matched core entropy (twin - control)")
    ax.set_xlabel("delta_entropy")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(args.out_fig_entropy, dpi=150)
    plt.close(fig)

    print(f"OK: wrote {out_csv}, {args.out_json}, figures")


if __name__ == "__main__":
    main()
