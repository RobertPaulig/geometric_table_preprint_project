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
    p.add_argument("--budgets", type=str, required=True)
    p.add_argument("--random-iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--test-costs", type=str, default="1,3600,86400")
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


def save_line_plot(path: Path, xs: List[float], series: Dict[str, List[float]],
                   title: str, xlabel: str, ylabel: str,
                   band: Tuple[List[float], List[float]] | None = None) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    for label, ys in series.items():
        ax.plot(xs, ys, marker="o", label=label)
    if band is not None:
        lo, hi = band
        ax.fill_between(xs, lo, hi, color="#4C72B0", alpha=0.15, label="Random 95% CI")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    args = parse_args()
    if args.Q0 > args.Q1:
        raise ValueError("Q0 must be <= Q1")

    budgets = parse_float_list(args.budgets)
    test_costs = parse_float_list(args.test_costs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    primes_q = sieve_primes(args.Q1)
    primes_q = [q for q in primes_q if q != 2]
    primes_factor = sieve_primes(int(math.isqrt(args.Q1)) + 1)

    ords: Dict[int, int] = {}
    for q in primes_q:
        ords[q] = ord_mod_2(q, primes_factor)

    primes_p = sieve_primes(args.p_max)
    primes_p = [p for p in primes_p if p >= 2]
    primes_p_set = set(primes_p)

    # killed flags
    killed_q0 = {p: 0 for p in primes_p}
    killed_q1 = {p: 0 for p in primes_p}
    for q in primes_q:
        d = ords[q]
        if d in killed_q1:
            killed_q1[d] = 1
            if q <= args.Q0:
                killed_q0[d] = 1

    survive_q1 = {p: 0 if killed_q1[p] == 1 else 1 for p in primes_p}

    # hazard features (M21)
    ap_harm_delta: Dict[int, float] = {p: 0.0 for p in primes_p}
    for q in primes_q:
        if q <= args.Q0:
            continue
        q_factors = factorize(q - 1, primes_factor)
        for p in q_factors:
            if p in primes_p_set:
                ap_harm_delta[p] += p / (q - 1)

    # M18 score (inv_q) for Q0
    score_m18 = {p: 0.0 for p in primes_p}
    for q in primes_q:
        if q > args.Q0:
            break
        d = ords[q]
        if d in score_m18:
            score_m18[d] += 1.0 / q

    # build method orderings (lower score = better)
    hazard_vals = {p: ap_harm_delta[p] for p in primes_p}

    def order_by(values: Dict[int, float]) -> List[int]:
        return sorted(values.keys(), key=lambda p: (values[p], p))

    order_m18 = order_by(score_m18)

    # M21-only: killed_Q0 last, then hazard
    order_m21 = sorted(
        primes_p,
        key=lambda p: (killed_q0[p], hazard_vals[p], p),
    )

    # M22: same queue definition (killed_Q0 last, hazard within)
    order_m22 = list(order_m21)

    rng = np.random.default_rng(args.seed)
    n_total = len(primes_p)

    def eval_order(order: List[int], k: int) -> Tuple[float, int]:
        picked = order[:k]
        survived = sum(survive_q1[p] for p in picked)
        yield_rate = survived / k if k > 0 else 0.0
        bad_tests = k - survived
        return yield_rate, bad_tests

    summary_rows = []
    random_ci = {}
    method_names = ["random", "m18", "m21", "m22"]

    for frac in budgets:
        k = max(1, int(round(frac * n_total)))

        # random baseline
        rand_yields = []
        rand_bad = []
        for _ in range(args.random_iters):
            idx = rng.choice(n_total, size=k, replace=False)
            picked = [primes_p[i] for i in idx]
            survived = sum(survive_q1[p] for p in picked)
            rand_yields.append(survived / k)
            rand_bad.append(k - survived)
        rand_yields_arr = np.array(rand_yields)
        rand_bad_arr = np.array(rand_bad)
        rand_mean_y = float(rand_yields_arr.mean())
        rand_mean_bad = float(rand_bad_arr.mean())
        rand_y_ci = 1.96 * float(rand_yields_arr.std(ddof=1)) / math.sqrt(len(rand_yields_arr))
        rand_bad_ci = 1.96 * float(rand_bad_arr.std(ddof=1)) / math.sqrt(len(rand_bad_arr))
        random_ci[frac] = (rand_mean_y, rand_y_ci, rand_mean_bad, rand_bad_ci)

        summary_rows.append({
            "budget": frac,
            "method": "random",
            "yield": rand_mean_y,
            "yield_ci": rand_y_ci,
            "bad_tests": rand_mean_bad,
            "bad_tests_ci": rand_bad_ci,
            "bad_tests_avoided": 0.0,
        })

        # deterministic methods
        for name, order in [("m18", order_m18), ("m21", order_m21), ("m22", order_m22)]:
            y, bad = eval_order(order, k)
            bad_avoided = rand_mean_bad - bad
            summary_rows.append({
                "budget": frac,
                "method": name,
                "yield": y,
                "yield_ci": 0.0,
                "bad_tests": bad,
                "bad_tests_ci": 0.0,
                "bad_tests_avoided": bad_avoided,
            })

    # compute saved for each test cost
    for row in summary_rows:
        for c in test_costs:
            row[f"saved_{c}"] = row["bad_tests_avoided"] * c

    # write summary
    summary_csv = out_dir / "m23_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    summary_json = out_dir / "m23_summary.json"
    summary_json.write_text(
        json.dumps({
            "p_max": args.p_max,
            "Q0": args.Q0,
            "Q1": args.Q1,
            "budgets": budgets,
            "random_iters": args.random_iters,
            "test_costs": test_costs,
            "n_primes": n_total,
            "rows": summary_rows,
        }, indent=2),
        encoding="utf-8",
    )

    # plots
    def series_for(method: str, key: str) -> List[float]:
        vals = []
        for frac in budgets:
            for row in summary_rows:
                if row["budget"] == frac and row["method"] == method:
                    vals.append(float(row[key]))
                    break
        return vals

    yield_series = {
        "M18": series_for("m18", "yield"),
        "M21": series_for("m21", "yield"),
        "M22": series_for("m22", "yield"),
        "Random": series_for("random", "yield"),
    }
    rand_lo = []
    rand_hi = []
    for frac in budgets:
        mean_y, ci_y, _, _ = random_ci[frac]
        rand_lo.append(mean_y - ci_y)
        rand_hi.append(mean_y + ci_y)
    save_line_plot(
        out_dir / "m23_yield_vs_budget.png",
        budgets,
        yield_series,
        "Yield vs budget (M23)",
        "budget fraction",
        "survival rate in selected",
        band=(rand_lo, rand_hi),
    )

    avoided_series = {
        "M18": series_for("m18", "bad_tests_avoided"),
        "M21": series_for("m21", "bad_tests_avoided"),
        "M22": series_for("m22", "bad_tests_avoided"),
        "Random": series_for("random", "bad_tests_avoided"),
    }
    save_line_plot(
        out_dir / "m23_bad_tests_avoided_vs_budget.png",
        budgets,
        avoided_series,
        "Bad tests avoided vs budget (M23)",
        "budget fraction",
        "bad tests avoided vs random",
    )

    for c in test_costs:
        key = f"saved_{c}"
        series = {
            "M18": series_for("m18", key),
            "M21": series_for("m21", key),
            "M22": series_for("m22", key),
        }
        save_line_plot(
            out_dir / f"m23_compute_saved_vs_budget_{int(c)}.png",
            budgets,
            series,
            f"Compute saved vs budget (cost={c:.0f}s)",
            "budget fraction",
            "compute-seconds saved",
        )

    # table for selected budgets
    table_budgets = [0.01, 0.02, 0.10]
    table_methods = ["random", "m18", "m21", "m22"]
    table_rows = []
    for frac in table_budgets:
        for method in table_methods:
            row = next(
                r for r in summary_rows
                if r["budget"] == frac and r["method"] == method
            )
            saved = "/".join(
                f"{int(round(row[f'saved_{c}']))}"
                for c in test_costs
            )
            table_rows.append((frac, method, row["yield"], saved))

    tex_lines = [
        r"\begin{tabular}{lrrl}\hline",
        r"Budget & Method & Yield & Saved (1s/1h/1d) \\ \hline",
    ]
    label_map = {
        "random": "Random",
        "m18": "M18",
        "m21": "M21",
        "m22": "M22",
    }
    for frac, method, y, saved in table_rows:
        tex_lines.append(f"{frac:.3f} & {label_map[method]} & {y:.3f} & {saved} \\\\")
    tex_lines.append(r"\hline\end{tabular}")

    (out_dir / "m23_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # queue snapshot for reproducibility (top-k for smallest budget)
    min_frac = min(budgets)
    k_min = max(1, int(round(min_frac * n_total)))
    queue_csv = out_dir / "m23_queue.csv"
    with queue_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "killed_Q0", "survive_Q1", "hazard", "score_m18"])
        for p in order_m22[:k_min]:
            w.writerow([p, killed_q0[p], survive_q1[p], hazard_vals[p], score_m18[p]])

    print(f"OK: wrote M23 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
