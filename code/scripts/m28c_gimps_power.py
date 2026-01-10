#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-ranges", type=str, nargs="+", required=True, help="pairs: a,b a,b ...")
    p.add_argument("--tf-q-limit-list", type=str, required=True, help="e.g. 1000000,2000000,5000000,10000000")
    p.add_argument("--Q0", type=int, required=True)
    p.add_argument("--q-horizon", type=int, required=True)
    p.add_argument("--mersenne-strict", type=int, required=True)
    p.add_argument("--seeds", type=str, required=True, help="e.g. 123,456,789")
    p.add_argument("--prp-cost-hours", type=str, required=True, help="e.g. 1,24,168")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_range_pair(s: str) -> Tuple[int, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Bad range pair: {s!r}, expected a,b")
    a, b = int(parts[0]), int(parts[1])
    if a > b:
        a, b = b, a
    return a, b


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    sieve = np.ones(n + 1, dtype=bool)
    sieve[:2] = False
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p: n + 1: p] = False
    return [i for i in range(2, n + 1) if sieve[i]]


def primes_in_ranges(ranges: List[Tuple[int, int]]) -> Dict[str, List[int]]:
    max_b = max(b for _, b in ranges)
    primes = sieve_primes(max_b)
    out = {}
    for a, b in ranges:
        key = f"{a}-{b}"
        out[key] = [p for p in primes if a <= p <= b]
    return out


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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_logit(X: np.ndarray, y: np.ndarray, lr: float = 0.2, steps: int = 1200) -> Tuple[np.ndarray, float]:
    w_vec = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(steps):
        z = X @ w_vec + b
        p = sigmoid(z)
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float((p - y).mean())
        w_vec -= lr * grad_w
        b -= lr * grad_b
    return w_vec, b


def isotonic_fit(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    x = scores[order]
    y = y[order]
    n = len(y)
    weights = np.ones(n)
    yhat = y.astype(float).copy()
    i = 0
    while i < n - 1:
        if yhat[i] > yhat[i + 1]:
            total_w = weights[i] + weights[i + 1]
            avg = (weights[i] * yhat[i] + weights[i + 1] * yhat[i + 1]) / total_w
            yhat[i] = avg
            yhat[i + 1] = avg
            weights[i] = total_w
            weights[i + 1] = total_w
            j = i
            while j > 0 and yhat[j - 1] > yhat[j]:
                total_w = weights[j - 1] + weights[j]
                avg = (weights[j - 1] * yhat[j - 1] + weights[j] * yhat[j]) / total_w
                yhat[j - 1] = avg
                yhat[j] = avg
                weights[j - 1] = total_w
                weights[j] = total_w
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    return x, yhat


def isotonic_predict(x_grid: np.ndarray, y_grid: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.interp(scores, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    n = len(scores)
    while i < n:
        j = i + 1
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = ranks[pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def enrichment_at_budgets(y_true: np.ndarray, scores: np.ndarray, budgets: List[float]) -> Tuple[float, Dict[float, float]]:
    base = float(y_true.mean()) if len(y_true) else 0.0
    order = np.argsort(scores)[::-1]
    enrich = {}
    for frac in budgets:
        k = max(1, int(round(frac * len(y_true))))
        hit_rate = float(y_true[order[:k]].mean())
        enrich[frac] = (hit_rate / base) if base > 0 else float("nan")
    return base, enrich


def bootstrap_ci(vals: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    lo = float(np.quantile(vals, alpha / 2.0))
    hi = float(np.quantile(vals, 1 - alpha / 2.0))
    return lo, hi


def save_line_plot(path: Path, xs: List[float], series: Dict[str, List[float]],
                   title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for label, ys in series.items():
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sanity_plot(path: Path, aucs: Dict[str, List[float]], enrich1: Dict[str, List[float]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axes[0]
    for label, ys in aucs.items():
        ax.plot(range(len(ys)), ys, marker="o", label=label)
    ax.axhline(0.5, linestyle="--", color="#888888", linewidth=1)
    ax.set_title("Permutation sanity: AUC")
    ax.set_xlabel("tf_q_limit index")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    for label, ys in enrich1.items():
        ax.plot(range(len(ys)), ys, marker="o", label=label)
    ax.axhline(1.0, linestyle="--", color="#888888", linewidth=1)
    ax.set_title("Permutation sanity: enrichment@1%")
    ax.set_xlabel("tf_q_limit index")
    ax.set_ylabel("enrichment")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


@dataclass
class Precomputed:
    death_q: Dict[int, Optional[int]]
    ap_count_by_limit: Dict[int, np.ndarray]
    ap_harm_by_limit: Dict[int, np.ndarray]
    p_list: List[int]
    p_to_idx: Dict[int, int]


def precompute_q_axis(
    p_list_union: List[int],
    Q0: int,
    q_horizon: int,
    tf_q_limits: List[int],
    mersenne_strict: bool,
) -> Precomputed:
    p_to_idx = {p: i for i, p in enumerate(p_list_union)}
    n_p = len(p_list_union)
    max_tf = max(tf_q_limits)

    death_q: Dict[int, Optional[int]] = {p: None for p in p_list_union}
    running_count = np.zeros(n_p, dtype=np.int32)
    running_harm = np.zeros(n_p, dtype=np.float64)
    ap_count_by_limit: Dict[int, np.ndarray] = {}
    ap_harm_by_limit: Dict[int, np.ndarray] = {}

    primes_factor = sieve_primes(int(math.isqrt(q_horizon)) + 1)
    primes_q = sieve_primes(q_horizon)
    primes_q = [q for q in primes_q if q != 2]

    limits_sorted = sorted(tf_q_limits)
    lim_idx = 0

    for q in primes_q:
        if mersenne_strict and (q % 8 not in (1, 7)):
            continue

        # snapshot when we pass limits
        while lim_idx < len(limits_sorted) and q > limits_sorted[lim_idx]:
            L = limits_sorted[lim_idx]
            ap_count_by_limit[L] = running_count.copy()
            ap_harm_by_limit[L] = running_harm.copy()
            lim_idx += 1

        d = ord_mod_2(q, primes_factor)
        if d in p_to_idx and death_q[d] is None:
            death_q[d] = q

        if q <= Q0:
            continue
        if q > max_tf:
            continue

        base = q - 1
        if mersenne_strict:
            base //= 2
        q_factors = factorize(base, primes_factor)
        for pf in q_factors:
            idx = p_to_idx.get(pf)
            if idx is None:
                continue
            running_count[idx] += 1
            running_harm[idx] += pf / (q - 1)

    # fill remaining limits
    while lim_idx < len(limits_sorted):
        L = limits_sorted[lim_idx]
        ap_count_by_limit[L] = running_count.copy()
        ap_harm_by_limit[L] = running_harm.copy()
        lim_idx += 1

    return Precomputed(
        death_q=death_q,
        ap_count_by_limit=ap_count_by_limit,
        ap_harm_by_limit=ap_harm_by_limit,
        p_list=p_list_union,
        p_to_idx=p_to_idx,
    )


def main() -> None:
    args = parse_args()
    p_ranges = [parse_range_pair(s) for s in args.p_ranges]
    tf_q_limits = parse_int_list(args.tf_q_limit_list)
    tf_q_limits = sorted(set(tf_q_limits))
    if not tf_q_limits:
        raise ValueError("tf-q-limit-list is empty")
    seeds = parse_int_list(args.seeds)
    if not seeds:
        raise ValueError("seeds is empty")
    prp_cost_hours = [float(x) for x in args.prp_cost_hours.split(",") if x.strip()]
    if not prp_cost_hours:
        raise ValueError("prp-cost-hours is empty")
    if args.Q0 <= 0:
        raise ValueError("Q0 must be positive")
    if args.q_horizon <= 0:
        raise ValueError("q-horizon must be positive")
    if max(tf_q_limits) >= args.q_horizon:
        raise ValueError("tf_q_limit must be < q_horizon for a post-TF task")

    mersenne_strict = int(args.mersenne_strict) == 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p_by_range = primes_in_ranges(p_ranges)
    p_union = sorted({p for ps in p_by_range.values() for p in ps})
    if not p_union:
        raise ValueError("No primes in p-ranges")

    pre = precompute_q_axis(
        p_list_union=p_union,
        Q0=args.Q0,
        q_horizon=args.q_horizon,
        tf_q_limits=tf_q_limits,
        mersenne_strict=mersenne_strict,
    )

    # Dataset rows: (range, seed, p, tf_q_limit)
    dataset_rows: List[Dict[str, object]] = []
    for range_key, ps in p_by_range.items():
        for seed in seeds:
            for tf_q_limit in tf_q_limits:
                count_vec = pre.ap_count_by_limit[tf_q_limit]
                harm_vec = pre.ap_harm_by_limit[tf_q_limit]
                for p in ps:
                    death = pre.death_q[p]
                    passes_tf = 1 if (death is None or death > tf_q_limit) else 0
                    dies_by_horizon = 0
                    if passes_tf == 1 and (death is not None and death <= args.q_horizon):
                        dies_by_horizon = 1

                    idx = pre.p_to_idx[p]
                    dataset_rows.append(
                        {
                            "p_range": range_key,
                            "seed": seed,
                            "p": p,
                            "tf_q_limit": tf_q_limit,
                            "passes_tf": passes_tf,
                            "dies_by_horizon": dies_by_horizon,
                            "ap_count_delta_tf": int(count_vec[idx]),
                            "ap_harm_delta_tf": float(harm_vec[idx]),
                        }
                    )

    dataset_path = out_dir / "m28c_dataset.csv.gz"
    dataset_header = [
        "p_range",
        "seed",
        "p",
        "tf_q_limit",
        "passes_tf",
        "dies_by_horizon",
        "ap_count_delta_tf",
        "ap_harm_delta_tf",
    ]
    with gzip.open(dataset_path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=dataset_header)
        writer.writeheader()
        writer.writerows(dataset_rows)

    budgets = [0.01, 0.05]
    metrics_raw: List[Dict[str, object]] = []
    sanity_rows: List[Dict[str, object]] = []
    savings_raw: List[Dict[str, object]] = []

    # Evaluate per (range, seed, tf_q_limit) on held-out subset and retain out-of-sample predictions for CI.
    B = 1000
    P = 50  # permutations for sanity (pooled over seeds per group)
    rng_ci = np.random.default_rng(0)
    metrics_ci: List[Dict[str, object]] = []
    savings_ci: List[Dict[str, object]] = []

    comb_by_range_limit: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    oos_scores_by_range_limit: Dict[Tuple[str, int], Dict[int, List[float]]] = {}
    oos_label_by_key: Dict[Tuple[str, int, int], int] = {}

    for range_key in sorted(p_by_range.keys()):
        for tf_q_limit in tf_q_limits:
            y_all_tests = []
            s_all_tests = []

            for seed in seeds:
                sub = [
                    r for r in dataset_rows
                    if r["p_range"] == range_key
                    and int(r["tf_q_limit"]) == tf_q_limit
                    and int(r["seed"]) == seed
                    and int(r["passes_tf"]) == 1
                ]
                n_pass = len(sub)
                if n_pass < 50:
                    metrics_raw.append(
                        {
                            "p_range": range_key,
                            "seed": seed,
                            "tf_q_limit": tf_q_limit,
                            "N_passes_tf": n_pass,
                            "base_rate": "",
                            "auc": "",
                            "enrichment@1pct": "",
                            "enrichment@5pct": "",
                            "note": "too_few_passes_tf",
                        }
                    )
                    continue

                p_all = np.array([int(r["p"]) for r in sub], dtype=int)
                y_all = np.array([int(r["dies_by_horizon"]) for r in sub], dtype=float)
                X_all = np.array(
                    [
                        [
                            math.log(int(r["p"])),
                            float(r["ap_count_delta_tf"]),
                            math.log1p(float(r["ap_harm_delta_tf"])),
                            float(r["ap_harm_delta_tf"]) / (1.0 + float(r["ap_count_delta_tf"])),
                        ]
                        for r in sub
                    ],
                    dtype=float,
                )
                X_all = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-9)

                rng = np.random.default_rng(seed)
                order = rng.permutation(n_pass)
                n_train = max(10, int(round(0.8 * n_pass)))
                train_idx = order[:n_train]
                test_idx = order[n_train:]
                if len(test_idx) < 10:
                    test_idx = order[max(0, n_pass - 10):]

                X_train = X_all[train_idx]
                y_train = y_all[train_idx]
                X_test = X_all[test_idx]
                y_test = y_all[test_idx]

                w_vec, b = fit_logit(X_train, y_train)
                logits_test = X_test @ w_vec + b
                probs_test = sigmoid(logits_test)
                x_grid, y_grid = isotonic_fit(probs_test, y_test)
                probs_iso_test = isotonic_predict(x_grid, y_grid, probs_test)

                base_rate, enrich = enrichment_at_budgets(y_test, probs_iso_test, budgets=budgets)
                order_by_score = np.argsort(probs_iso_test)[::-1]
                k1 = max(1, int(round(0.01 * len(y_test))))
                k5 = max(1, int(round(0.05 * len(y_test))))
                hit1 = float(y_test[order_by_score[:k1]].mean()) if len(y_test) else float("nan")
                hit5 = float(y_test[order_by_score[:k5]].mean()) if len(y_test) else float("nan")
                auc = auc_score(y_test, probs_iso_test)

                metrics_raw.append(
                    {
                        "p_range": range_key,
                        "seed": seed,
                        "tf_q_limit": tf_q_limit,
                        "N_passes_tf": n_pass,
                        "N_test": int(len(test_idx)),
                        "base_rate": base_rate,
                        "auc": auc,
                        "enrichment@1pct": enrich[0.01],
                        "enrichment@5pct": enrich[0.05],
                        "hit_rate@1pct": hit1,
                        "hit_rate@5pct": hit5,
                        "note": "",
                    }
                )

                y_all_tests.append(y_test)
                s_all_tests.append(probs_iso_test)

                # retain out-of-sample predictions (for queues)
                p_test = p_all[test_idx]
                bucket = oos_scores_by_range_limit.setdefault((range_key, tf_q_limit), {})
                for pp, yy, ss in zip(p_test.tolist(), y_test.tolist(), probs_iso_test.tolist()):
                    bucket.setdefault(int(pp), []).append(float(ss))
                    oos_label_by_key[(range_key, tf_q_limit, int(pp))] = int(yy)

                # per-seed savings (relative to random baseline) on test
                for frac in budgets:
                    k = max(1, int(round(frac * len(y_test))))
                    top = float(y_test[np.argsort(probs_iso_test)[::-1][:k]].mean())
                    delta = top - base_rate
                    for cost_h in prp_cost_hours:
                        saved = delta * k * cost_h * 3600.0
                        savings_raw.append(
                            {
                                "p_range": range_key,
                                "seed": seed,
                                "tf_q_limit": tf_q_limit,
                                "budget_fraction": frac,
                                "prp_cost_hours": cost_h,
                                "saved_compute_seconds": saved,
                            }
                        )

            if not y_all_tests:
                continue

            y_comb = np.concatenate(y_all_tests)
            s_comb = np.concatenate(s_all_tests)
            comb_by_range_limit[(range_key, tf_q_limit)] = (y_comb, s_comb)

            # permutation sanity on combined test set
            auc_perm = []
            e1_perm = []
            e5_perm = []
            for _ in range(P):
                perm = rng_ci.permutation(len(s_comb))
                s_perm = s_comb[perm]
                auc_perm.append(auc_score(y_comb, s_perm))
                _, enrich_p = enrichment_at_budgets(y_comb, s_perm, budgets=budgets)
                e1_perm.append(enrich_p[0.01])
                e5_perm.append(enrich_p[0.05])
            sanity_rows.append(
                {
                    "p_range": range_key,
                    "tf_q_limit": tf_q_limit,
                    "auc_perm_mean": float(np.mean(auc_perm)),
                    "auc_perm_ci_low": float(np.quantile(auc_perm, 0.025)),
                    "auc_perm_ci_high": float(np.quantile(auc_perm, 0.975)),
                    "enrichment@1pct_perm_mean": float(np.nanmean(e1_perm)),
                    "enrichment@1pct_perm_ci_low": float(np.nanquantile(e1_perm, 0.025)),
                    "enrichment@1pct_perm_ci_high": float(np.nanquantile(e1_perm, 0.975)),
                    "enrichment@5pct_perm_mean": float(np.nanmean(e5_perm)),
                    "enrichment@5pct_perm_ci_low": float(np.nanquantile(e5_perm, 0.025)),
                    "enrichment@5pct_perm_ci_high": float(np.nanquantile(e5_perm, 0.975)),
                }
            )

            # observed metrics on combined test set + bootstrap CI over observations
            base_rate_obs, enrich_obs = enrichment_at_budgets(y_comb, s_comb, budgets=budgets)
            auc_obs = auc_score(y_comb, s_comb)

            idxs = rng_ci.integers(0, len(y_comb), size=(B, len(y_comb)))
            auc_boot = []
            e1_boot = []
            e5_boot = []
            br_boot = []
            for b_idx in idxs:
                yb = y_comb[b_idx]
                sb = s_comb[b_idx]
                br, enr = enrichment_at_budgets(yb, sb, budgets=budgets)
                br_boot.append(br)
                auc_boot.append(auc_score(yb, sb))
                e1_boot.append(enr[0.01])
                e5_boot.append(enr[0.05])
            auc_boot = np.array(auc_boot, dtype=float)
            e1_boot = np.array(e1_boot, dtype=float)
            e5_boot = np.array(e5_boot, dtype=float)
            br_boot = np.array(br_boot, dtype=float)
            auc_lo, auc_hi = bootstrap_ci(auc_boot)
            e1_lo, e1_hi = bootstrap_ci(e1_boot)
            e5_lo, e5_hi = bootstrap_ci(e5_boot)
            br_lo, br_hi = bootstrap_ci(br_boot)

            metrics_ci.append(
                {
                    "p_range": range_key,
                    "tf_q_limit": tf_q_limit,
                    "N_seeds": len(seeds),
                    "N_test_total": int(len(y_comb)),
                    "base_rate_mean": base_rate_obs,
                    "base_rate_ci_low": br_lo,
                    "base_rate_ci_high": br_hi,
                    "auc_mean": auc_obs,
                    "auc_ci_low": auc_lo,
                    "auc_ci_high": auc_hi,
                    "enrichment@1pct_mean": enrich_obs[0.01],
                    "enrichment@1pct_ci_low": e1_lo,
                    "enrichment@1pct_ci_high": e1_hi,
                    "enrichment@5pct_mean": enrich_obs[0.05],
                    "enrichment@5pct_ci_low": e5_lo,
                    "enrichment@5pct_ci_high": e5_hi,
                }
            )

            # savings CI on combined observations (top-k by score)
            for frac in budgets:
                k = max(1, int(round(frac * len(y_comb))))
                order = np.argsort(s_comb)[::-1]
                top_rate = float(y_comb[order[:k]].mean())
                delta = top_rate - base_rate_obs
                for cost_h in prp_cost_hours:
                    saved_obs = delta * k * cost_h * 3600.0
                    saved_boot = []
                    for b_idx in idxs:
                        yb = y_comb[b_idx]
                        sb = s_comb[b_idx]
                        br, _ = enrichment_at_budgets(yb, sb, budgets=[frac])
                        order_b = np.argsort(sb)[::-1]
                        k_b = max(1, int(round(frac * len(yb))))
                        top_b = float(yb[order_b[:k_b]].mean())
                        saved_boot.append((top_b - br) * k_b * cost_h * 3600.0)
                    saved_boot = np.array(saved_boot, dtype=float)
                    lo, hi = bootstrap_ci(saved_boot)
                    savings_ci.append(
                        {
                            "p_range": range_key,
                            "tf_q_limit": tf_q_limit,
                            "budget_fraction": frac,
                            "prp_cost_hours": cost_h,
                            "saved_compute_seconds_mean": saved_obs,
                            "saved_compute_seconds_ci_low": lo,
                            "saved_compute_seconds_ci_high": hi,
                        }
                    )

    # Pooled (both p-ranges) metrics/CI as a reference (main conclusions should hold per-range).
    if len(p_by_range) >= 2:
        pooled_key = "pooled"
        for tf_q_limit in tf_q_limits:
            ys = []
            ss = []
            for range_key in sorted(p_by_range.keys()):
                comb = comb_by_range_limit.get((range_key, tf_q_limit))
                if comb is not None:
                    ys.append(comb[0])
                    ss.append(comb[1])
            if not ys:
                continue
            y_pool = np.concatenate(ys)
            s_pool = np.concatenate(ss)

            auc_perm = []
            e1_perm = []
            e5_perm = []
            for _ in range(P):
                perm = rng_ci.permutation(len(s_pool))
                s_perm = s_pool[perm]
                auc_perm.append(auc_score(y_pool, s_perm))
                _, enrich_p = enrichment_at_budgets(y_pool, s_perm, budgets=budgets)
                e1_perm.append(enrich_p[0.01])
                e5_perm.append(enrich_p[0.05])
            sanity_rows.append(
                {
                    "p_range": pooled_key,
                    "tf_q_limit": tf_q_limit,
                    "auc_perm_mean": float(np.mean(auc_perm)),
                    "auc_perm_ci_low": float(np.quantile(auc_perm, 0.025)),
                    "auc_perm_ci_high": float(np.quantile(auc_perm, 0.975)),
                    "enrichment@1pct_perm_mean": float(np.nanmean(e1_perm)),
                    "enrichment@1pct_perm_ci_low": float(np.nanquantile(e1_perm, 0.025)),
                    "enrichment@1pct_perm_ci_high": float(np.nanquantile(e1_perm, 0.975)),
                    "enrichment@5pct_perm_mean": float(np.nanmean(e5_perm)),
                    "enrichment@5pct_perm_ci_low": float(np.nanquantile(e5_perm, 0.025)),
                    "enrichment@5pct_perm_ci_high": float(np.nanquantile(e5_perm, 0.975)),
                }
            )

            base_rate_obs, enrich_obs = enrichment_at_budgets(y_pool, s_pool, budgets=budgets)
            auc_obs = auc_score(y_pool, s_pool)

            idxs = rng_ci.integers(0, len(y_pool), size=(B, len(y_pool)))
            auc_boot = []
            e1_boot = []
            e5_boot = []
            br_boot = []
            for b_idx in idxs:
                yb = y_pool[b_idx]
                sb = s_pool[b_idx]
                br, enr = enrichment_at_budgets(yb, sb, budgets=budgets)
                br_boot.append(br)
                auc_boot.append(auc_score(yb, sb))
                e1_boot.append(enr[0.01])
                e5_boot.append(enr[0.05])
            auc_lo, auc_hi = bootstrap_ci(np.array(auc_boot, dtype=float))
            e1_lo, e1_hi = bootstrap_ci(np.array(e1_boot, dtype=float))
            e5_lo, e5_hi = bootstrap_ci(np.array(e5_boot, dtype=float))
            br_lo, br_hi = bootstrap_ci(np.array(br_boot, dtype=float))

            metrics_ci.append(
                {
                    "p_range": pooled_key,
                    "tf_q_limit": tf_q_limit,
                    "N_seeds": len(seeds),
                    "N_test_total": int(len(y_pool)),
                    "base_rate_mean": base_rate_obs,
                    "base_rate_ci_low": br_lo,
                    "base_rate_ci_high": br_hi,
                    "auc_mean": auc_obs,
                    "auc_ci_low": auc_lo,
                    "auc_ci_high": auc_hi,
                    "enrichment@1pct_mean": enrich_obs[0.01],
                    "enrichment@1pct_ci_low": e1_lo,
                    "enrichment@1pct_ci_high": e1_hi,
                    "enrichment@5pct_mean": enrich_obs[0.05],
                    "enrichment@5pct_ci_low": e5_lo,
                    "enrichment@5pct_ci_high": e5_hi,
                }
            )

            for frac in budgets:
                k = max(1, int(round(frac * len(y_pool))))
                order = np.argsort(s_pool)[::-1]
                top_rate = float(y_pool[order[:k]].mean())
                delta = top_rate - base_rate_obs
                for cost_h in prp_cost_hours:
                    saved_obs = delta * k * cost_h * 3600.0
                    saved_boot = []
                    for b_idx in idxs:
                        yb = y_pool[b_idx]
                        sb = s_pool[b_idx]
                        br, _ = enrichment_at_budgets(yb, sb, budgets=[frac])
                        order_b = np.argsort(sb)[::-1]
                        k_b = max(1, int(round(frac * len(yb))))
                        top_b = float(yb[order_b[:k_b]].mean())
                        saved_boot.append((top_b - br) * k_b * cost_h * 3600.0)
                    lo, hi = bootstrap_ci(np.array(saved_boot, dtype=float))
                    savings_ci.append(
                        {
                            "p_range": pooled_key,
                            "tf_q_limit": tf_q_limit,
                            "budget_fraction": frac,
                            "prp_cost_hours": cost_h,
                            "saved_compute_seconds_mean": saved_obs,
                            "saved_compute_seconds_ci_low": lo,
                            "saved_compute_seconds_ci_high": hi,
                        }
                    )

    # Queues: top-5% among TF survivors, using pooled out-of-sample scores across seeds.
    range_keys_sorted = sorted(p_by_range.keys())
    range_tags = {rk: f"range{chr(ord('A') + i)}" for i, rk in enumerate(range_keys_sorted)}

    pass_ps_by_range_limit: Dict[Tuple[str, int], List[int]] = {}
    for row in dataset_rows:
        if int(row["passes_tf"]) != 1:
            continue
        key = (str(row["p_range"]), int(row["tf_q_limit"]))
        pass_ps_by_range_limit.setdefault(key, []).append(int(row["p"]))
    for key in pass_ps_by_range_limit:
        pass_ps_by_range_limit[key] = sorted(set(pass_ps_by_range_limit[key]))

    def tf_label(L: int) -> str:
        if L % 1_000_000 == 0:
            return f"{L // 1_000_000}M"
        return str(L)

    for range_key in range_keys_sorted:
        for L in tf_q_limits:
            ps_pass = pass_ps_by_range_limit.get((range_key, L), [])
            if not ps_pass:
                continue
            k = max(1, int(math.ceil(0.05 * len(ps_pass))))
            scores_map = oos_scores_by_range_limit.get((range_key, L), {})
            scored = []
            for p in ps_pass:
                scores = scores_map.get(p)
                if not scores:
                    continue
                scored.append((p, float(np.mean(scores)), int(len(scores))))
            if not scored:
                continue
            scored.sort(key=lambda t: t[1], reverse=True)
            scored = scored[: min(k, len(scored))]

            tag = range_tags.get(range_key, range_key.replace("-", "_"))
            qpath = out_dir / f"m28c_queue_{tag}_tf{tf_label(L)}_top5pct.csv"
            with qpath.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["p", "score_mean_oos", "n_oos_scores", "dies_by_horizon"],
                )
                writer.writeheader()
                for p, s_mean, n_scores in scored:
                    writer.writerow(
                        {
                            "p": p,
                            "score_mean_oos": s_mean,
                            "n_oos_scores": n_scores,
                            "dies_by_horizon": oos_label_by_key.get((range_key, L, p), ""),
                        }
                    )

    # Save raw metrics
    raw_path = out_dir / "m28c_metrics_raw.csv"
    with raw_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for r in metrics_raw for k in r.keys()})
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_raw)

    sanity_path = out_dir / "m28c_sanity_permutation.csv"
    with sanity_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "p_range",
            "tf_q_limit",
            "auc_perm_mean",
            "auc_perm_ci_low",
            "auc_perm_ci_high",
            "enrichment@1pct_perm_mean",
            "enrichment@1pct_perm_ci_low",
            "enrichment@1pct_perm_ci_high",
            "enrichment@5pct_perm_mean",
            "enrichment@5pct_perm_ci_low",
            "enrichment@5pct_perm_ci_high",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sanity_rows)

    savings_raw_path = out_dir / "m28c_savings_raw.csv"
    with savings_raw_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["p_range", "seed", "tf_q_limit", "budget_fraction", "prp_cost_hours", "saved_compute_seconds"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(savings_raw)

    metrics_ci_path = out_dir / "m28c_metrics_ci.csv"
    with metrics_ci_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "p_range",
            "tf_q_limit",
            "N_seeds",
            "N_test_total",
            "base_rate_mean",
            "base_rate_ci_low",
            "base_rate_ci_high",
            "auc_mean",
            "auc_ci_low",
            "auc_ci_high",
            "enrichment@1pct_mean",
            "enrichment@1pct_ci_low",
            "enrichment@1pct_ci_high",
            "enrichment@5pct_mean",
            "enrichment@5pct_ci_low",
            "enrichment@5pct_ci_high",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics_ci)

    savings_ci_path = out_dir / "m28c_savings_ci.csv"
    with savings_ci_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "p_range",
            "tf_q_limit",
            "budget_fraction",
            "prp_cost_hours",
            "saved_compute_seconds_mean",
            "saved_compute_seconds_ci_low",
            "saved_compute_seconds_ci_high",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(savings_ci)

    # Figures (use mean over seeds; CI in CSV)
    xs = [float(x) for x in tf_q_limits]
    for_plot = {(r["p_range"], int(r["tf_q_limit"])): r for r in metrics_ci}
    auc_series = {}
    e1_series = {}
    e5_series = {}
    for range_key in sorted(p_by_range.keys()):
        auc_series[range_key] = [float(for_plot.get((range_key, L), {}).get("auc_mean", float("nan"))) for L in tf_q_limits]
        e1_series[f"{range_key} @1%"] = [
            float(for_plot.get((range_key, L), {}).get("enrichment@1pct_mean", float("nan"))) for L in tf_q_limits
        ]
        e5_series[f"{range_key} @5%"] = [
            float(for_plot.get((range_key, L), {}).get("enrichment@5pct_mean", float("nan"))) for L in tf_q_limits
        ]

    save_line_plot(
        out_dir / "m28c_auc_vs_tf_q_limit.png",
        xs,
        auc_series,
        "AUC vs TF proxy depth on q-axis (passes_tf=1)",
        "tf_q_limit (q-axis)",
        "AUC",
    )
    save_line_plot(
        out_dir / "m28c_enrichment_vs_tf_q_limit.png",
        xs,
        {**e1_series, **e5_series},
        "Enrichment vs TF proxy depth on q-axis (passes_tf=1)",
        "tf_q_limit (q-axis)",
        "enrichment vs random",
    )

    # Savings plot: top5% @ 24h by default
    default_cost = 24.0 if 24.0 in prp_cost_hours else float(prp_cost_hours[0])
    savings_plot = {}
    for range_key in sorted(p_by_range.keys()):
        ys = []
        for L in tf_q_limits:
            row = next(
                (
                    r for r in savings_ci
                    if r["p_range"] == range_key
                    and int(r["tf_q_limit"]) == L
                    and float(r["budget_fraction"]) == 0.05
                    and float(r["prp_cost_hours"]) == default_cost
                ),
                None,
            )
            ys.append(float(row["saved_compute_seconds_mean"]) if row else float("nan"))
        savings_plot[range_key] = ys
    save_line_plot(
        out_dir / "m28c_savings_vs_tf_q_limit.png",
        xs,
        savings_plot,
        f"Savings vs TF proxy depth (top5%, PRP={int(default_cost)}h)",
        "tf_q_limit (q-axis)",
        "compute-seconds saved",
    )

    # Sanity plot: per range, use permutation means
    sanity_auc = {k: [] for k in sorted(p_by_range.keys())}
    sanity_e1 = {k: [] for k in sorted(p_by_range.keys())}
    for range_key in sorted(p_by_range.keys()):
        for L in tf_q_limits:
            row = next((r for r in sanity_rows if r["p_range"] == range_key and int(r["tf_q_limit"]) == L), None)
            sanity_auc[range_key].append(float(row["auc_perm_mean"]) if row else float("nan"))
            sanity_e1[range_key].append(float(row["enrichment@1pct_perm_mean"]) if row else float("nan"))
    save_sanity_plot(out_dir / "m28c_sanity_plot.png", sanity_auc, sanity_e1)

    # TeX table: show mean +/- CI for enrichment@5% and AUC, per range
    table_lines = [
        r"\begin{tabular}{lrrrr}\hline",
        r"TF proxy $Q_{\mathrm{TF}}$ & AUC (A) & enrich@5\% (A) & AUC (B) & enrich@5\% (B) \\ \hline",
    ]
    range_keys_sorted = sorted(p_by_range.keys())
    range_a = range_keys_sorted[0]
    range_b = range_keys_sorted[1] if len(range_keys_sorted) > 1 else range_keys_sorted[0]
    for L in tf_q_limits:
        ra = for_plot.get((range_a, L), {})
        rb = for_plot.get((range_b, L), {})
        table_lines.append(
            f"{L} & {float(ra.get('auc_mean', float('nan'))):.3f} "
            f"[{float(ra.get('auc_ci_low', float('nan'))):.3f},{float(ra.get('auc_ci_high', float('nan'))):.3f}] & "
            f"{float(ra.get('enrichment@5pct_mean', float('nan'))):.2f} "
            f"[{float(ra.get('enrichment@5pct_ci_low', float('nan'))):.2f},{float(ra.get('enrichment@5pct_ci_high', float('nan'))):.2f}] & "
            f"{float(rb.get('auc_mean', float('nan'))):.3f} "
            f"[{float(rb.get('auc_ci_low', float('nan'))):.3f},{float(rb.get('auc_ci_high', float('nan'))):.3f}] & "
            f"{float(rb.get('enrichment@5pct_mean', float('nan'))):.2f} "
            f"[{float(rb.get('enrichment@5pct_ci_low', float('nan'))):.2f},{float(rb.get('enrichment@5pct_ci_high', float('nan'))):.2f}] \\\\"
        )
    table_lines.append(r"\hline\end{tabular}")
    (out_dir / "m28c_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    # Dataset summary
    summary = {
        "proxy_note": "TF proxy depth on q-axis (not literal GIMPS TF bits).",
        "p_ranges": [{"key": k, "N_p": len(ps)} for k, ps in sorted(p_by_range.items())],
        "tf_q_limit_list": tf_q_limits,
        "Q0": args.Q0,
        "q_horizon": args.q_horizon,
        "mersenne_strict": mersenne_strict,
        "seeds": seeds,
        "prp_cost_hours": prp_cost_hours,
        "counts": {},
    }
    for range_key, ps in p_by_range.items():
        for L in tf_q_limits:
            for seed in seeds:
                n_total = len(ps)
                sub_pass = [
                    r for r in dataset_rows
                    if r["p_range"] == range_key and int(r["tf_q_limit"]) == L and int(r["seed"]) == seed and int(r["passes_tf"]) == 1
                ]
                dies = sum(int(r["dies_by_horizon"]) for r in sub_pass)
                summary["counts"][f"{range_key}|{seed}|{L}"] = {
                    "N_total": n_total,
                    "N_passes_tf": len(sub_pass),
                    "N_dies_by_horizon_among_passes": int(dies),
                }
    (out_dir / "m28c_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Manifest
    manifest = {
        "command": "python code/scripts/m28c_gimps_power.py",
        "params": {
            "p_ranges": p_ranges,
            "tf_q_limit_list": tf_q_limits,
            "Q0": args.Q0,
            "q_horizon": args.q_horizon,
            "mersenne_strict": mersenne_strict,
            "seeds": seeds,
            "prp_cost_hours": prp_cost_hours,
        },
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m28c_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK: wrote M28c artifacts to {out_dir}")


if __name__ == "__main__":
    main()
