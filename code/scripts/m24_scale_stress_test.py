#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-max", type=int, required=True)
    p.add_argument("--Q0", type=int, required=True)
    p.add_argument("--Q1", type=int, required=True)
    p.add_argument("--budgets", type=str, required=True)
    p.add_argument("--random-iters", type=int, required=True)
    p.add_argument("--test-costs", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--label", type=str, required=True)
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


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    if args.Q0 > args.Q1:
        raise ValueError("Q0 must be <= Q1")

    budgets = parse_float_list(args.budgets)
    test_costs = parse_float_list(args.test_costs)

    out_root = Path(args.out_dir)
    out_dir = out_root / args.label
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

    killed_q0 = {p: 0 for p in primes_p}
    killed_q1 = {p: 0 for p in primes_p}
    for q in primes_q:
        d = ords[q]
        if d in killed_q1:
            killed_q1[d] = 1
            if q <= args.Q0:
                killed_q0[d] = 1

    survive_q1 = {p: 0 if killed_q1[p] == 1 else 1 for p in primes_p}

    # hazard (M21)
    hazard_vals: Dict[int, float] = {p: 0.0 for p in primes_p}
    for q in primes_q:
        if q <= args.Q0:
            continue
        q_factors = factorize(q - 1, primes_factor)
        for p in q_factors:
            if p in primes_p_set:
                hazard_vals[p] += p / (q - 1)

    # M18 score (inv_q) from Q0
    score_m18 = {p: 0.0 for p in primes_p}
    for q in primes_q:
        if q > args.Q0:
            break
        d = ords[q]
        if d in score_m18:
            score_m18[d] += 1.0 / q

    def order_by(values: Dict[int, float]) -> List[int]:
        return sorted(values.keys(), key=lambda p: (values[p], p))

    order_m18 = order_by(score_m18)
    order_m21 = sorted(primes_p, key=lambda p: (killed_q0[p], hazard_vals[p], p))
    order_m22 = list(order_m21)

    rank_m18 = {p: i + 1 for i, p in enumerate(order_m18)}
    rank_m22 = {p: i + 1 for i, p in enumerate(order_m22)}

    n_total = len(primes_p)
    rng = np.random.default_rng(args.seed)

    def eval_order(order: List[int], k: int) -> Tuple[float, int]:
        picked = order[:k]
        survived = sum(survive_q1[p] for p in picked)
        yield_rate = survived / k if k > 0 else 0.0
        bad_tests = k - survived
        return yield_rate, bad_tests

    summary_rows = []
    random_ci: Dict[float, Tuple[float, float, float, float]] = {}

    for frac in budgets:
        k = max(1, int(round(frac * n_total)))

        rand_yields = []
        rand_bad = []
        for _ in range(args.random_iters):
            idx = rng.choice(n_total, size=k, replace=False)
            picked = [primes_p[i] for i in idx]
            survived = sum(survive_q1[p] for p in picked)
            rand_yields.append(survived / k)
            rand_bad.append(k - survived)
        rand_y_arr = np.array(rand_yields)
        rand_bad_arr = np.array(rand_bad)
        rand_mean_y = float(rand_y_arr.mean())
        rand_mean_bad = float(rand_bad_arr.mean())
        rand_y_ci = 1.96 * float(rand_y_arr.std(ddof=1)) / math.sqrt(len(rand_y_arr))
        rand_bad_ci = 1.96 * float(rand_bad_arr.std(ddof=1)) / math.sqrt(len(rand_bad_arr))
        random_ci[frac] = (rand_mean_y, rand_y_ci, rand_mean_bad, rand_bad_ci)

        summary_rows.append({
            "budget": frac,
            "method": "random",
            "n_selected": k,
            "yield": rand_mean_y,
            "yield_ci_low": rand_mean_y - rand_y_ci,
            "yield_ci_high": rand_mean_y + rand_y_ci,
            "bad_tests": rand_mean_bad,
            "bad_tests_ci_low": rand_mean_bad - rand_bad_ci,
            "bad_tests_ci_high": rand_mean_bad + rand_bad_ci,
            "bad_tests_avoided_vs_random": 0.0,
        })

        for name, order in [("m18", order_m18), ("m21", order_m21), ("m22", order_m22)]:
            y, bad = eval_order(order, k)
            bad_avoided = rand_mean_bad - bad
            summary_rows.append({
                "budget": frac,
                "method": name,
                "n_selected": k,
                "yield": y,
                "yield_ci_low": "",
                "yield_ci_high": "",
                "bad_tests": bad,
                "bad_tests_ci_low": "",
                "bad_tests_ci_high": "",
                "bad_tests_avoided_vs_random": bad_avoided,
            })

    for row in summary_rows:
        for c in test_costs:
            row[f"saved_sec_cost{int(c)}"] = row["bad_tests_avoided_vs_random"] * c

    summary_csv = out_dir / "m24_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    summary_json = out_dir / "m24_summary.json"
    summary_json.write_text(
        json.dumps({
            "label": args.label,
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
        out_dir / "m24_yield_vs_budget.png",
        budgets,
        yield_series,
        "Yield vs budget (M24)",
        "budget fraction",
        "survival rate in selected",
        band=(rand_lo, rand_hi),
    )

    avoided_series = {
        "M18": series_for("m18", "bad_tests_avoided_vs_random"),
        "M21": series_for("m21", "bad_tests_avoided_vs_random"),
        "M22": series_for("m22", "bad_tests_avoided_vs_random"),
        "Random": series_for("random", "bad_tests_avoided_vs_random"),
    }
    save_line_plot(
        out_dir / "m24_bad_tests_avoided_vs_budget.png",
        budgets,
        avoided_series,
        "Bad tests avoided vs budget (M24)",
        "budget fraction",
        "bad tests avoided vs random",
    )

    for c in test_costs:
        key = f"saved_sec_cost{int(c)}"
        series = {
            "M18": series_for("m18", key),
            "M21": series_for("m21", key),
            "M22": series_for("m22", key),
        }
        label = f"{int(c)}s"
        save_line_plot(
            out_dir / f"m24_compute_saved_vs_budget_{label}.png",
            budgets,
            series,
            f"Compute saved vs budget (cost={int(c)}s)",
            "budget fraction",
            "compute-seconds saved",
        )

    # table for selected budgets
    table_budgets = [0.01, 0.02, 0.10, 0.20, 0.50]
    table_methods = ["random", "m18", "m21", "m22"]
    label_map = {"random": "Random", "m18": "M18", "m21": "M21", "m22": "M22"}
    tex_lines = [
        r"\begin{tabular}{lrrrl}\hline",
        r"Budget & Method & Yield & Bad tests & Saved (1h/1d) \\ \hline",
    ]
    for frac in table_budgets:
        for method in table_methods:
            row = next(
                r for r in summary_rows
                if r["budget"] == frac and r["method"] == method
            )
            saved = f"{int(round(row['saved_sec_cost3600']))}/{int(round(row['saved_sec_cost86400']))}"
            tex_lines.append(
                f"{frac:.3f} & {label_map[method]} & {float(row['yield']):.3f} & "
                f"{float(row['bad_tests']):.2f} & {saved} \\\\"
            )
    tex_lines.append(r"\hline\end{tabular}")
    (out_dir / "m24_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # queue snapshot
    queue_csv = out_dir / "m24_queue.csv"
    with queue_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p", "method_rank_M18", "killed_Q0", "hazard", "rank_M22", "survive_Q1"])
        for p in order_m22:
            w.writerow([
                p,
                rank_m18[p],
                killed_q0[p],
                hazard_vals[p],
                rank_m22[p],
                survive_q1[p],
            ])

    # manifest with sha256
    manifest = {
        "label": args.label,
        "p_max": args.p_max,
        "Q0": args.Q0,
        "Q1": args.Q1,
        "budgets": budgets,
        "random_iters": args.random_iters,
        "test_costs": test_costs,
        "seed": args.seed,
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m24_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"OK: wrote M24 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
