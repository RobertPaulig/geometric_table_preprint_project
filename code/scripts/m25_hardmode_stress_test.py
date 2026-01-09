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
    p.add_argument("--hazard-modes", type=str, required=True)
    p.add_argument("--mersenne-strict", type=int, required=True)
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


def parse_str_list(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def save_hist(path: Path, data: List[int], title: str, xlabel: str, ylabel: str,
              bins: int = 40) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.hist(data, bins=bins, color="#4C72B0", alpha=0.85)
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

    budgets = parse_float_list(args.budgets)
    test_costs = parse_float_list(args.test_costs)
    hazard_modes = parse_str_list(args.hazard_modes)
    mersenne_strict = int(args.mersenne_strict) == 1

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

    # hazard features in (Q0, Q1], optionally strict
    ap_count_delta: Dict[int, int] = {p: 0 for p in primes_p}
    ap_harm_delta: Dict[int, float] = {p: 0.0 for p in primes_p}

    for q in primes_q:
        if q <= args.Q0:
            continue
        if mersenne_strict and (q % 8 not in (1, 7)):
            continue
        base = q - 1
        if mersenne_strict:
            base //= 2
        q_factors = factorize(base, primes_factor)
        for p in q_factors:
            if p in primes_p_set:
                ap_count_delta[p] += 1
                ap_harm_delta[p] += p / (q - 1)

    # hazard modes on Z = killed_Q0 == 0
    hazard_binary = {p: (1.0 if ap_count_delta[p] > 0 else 0.0) for p in primes_p}
    hazard_count = {p: math.log(1.0 + ap_count_delta[p]) for p in primes_p}
    hazard_harmonic = {p: ap_harm_delta[p] for p in primes_p}
    hazard_map = {
        "binary": hazard_binary,
        "count": hazard_count,
        "harmonic": hazard_harmonic,
    }

    # orderings per hazard mode (M22 variants)
    def order_for(hazard: Dict[int, float]) -> List[int]:
        return sorted(primes_p, key=lambda p: (killed_q0[p], hazard[p], p))

    orders = {mode: order_for(hazard_map[mode]) for mode in hazard_modes}
    ranks = {
        mode: {p: i + 1 for i, p in enumerate(order)}
        for mode, order in orders.items()
    }

    rng = np.random.default_rng(args.seed)
    n_total = len(primes_p)

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

        for mode in hazard_modes:
            y, bad = eval_order(orders[mode], k)
            bad_avoided = rand_mean_bad - bad
            summary_rows.append({
                "budget": frac,
                "method": f"M22_{mode}",
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

    summary_csv = out_dir / "m25_summary.csv"
    fieldnames = list(summary_rows[0].keys())
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(summary_rows)

    summary_json = out_dir / "m25_summary.json"
    summary_json.write_text(
        json.dumps({
            "label": args.label,
            "p_max": args.p_max,
            "Q0": args.Q0,
            "Q1": args.Q1,
            "budgets": budgets,
            "random_iters": args.random_iters,
            "test_costs": test_costs,
            "hazard_modes": hazard_modes,
            "mersenne_strict": mersenne_strict,
            "n_primes": n_total,
            "rows": summary_rows,
        }, indent=2),
        encoding="utf-8",
    )

    # hardness diagnostics on Z
    z_ps = [p for p in primes_p if killed_q0[p] == 0]
    z_counts = [ap_count_delta[p] for p in z_ps]
    z_counts_sorted = sorted(z_counts)
    def quantile(q: float) -> float:
        if not z_counts_sorted:
            return 0.0
        idx = int(round(q * (len(z_counts_sorted) - 1)))
        return float(z_counts_sorted[idx])

    hardness = {
        "n_Z": len(z_ps),
        "fraction_Z_with_ap_count_delta_gt0": float(sum(1 for c in z_counts if c > 0) / len(z_counts)) if z_counts else 0.0,
        "ap_count_delta_mean": float(np.mean(z_counts)) if z_counts else 0.0,
        "ap_count_delta_median": quantile(0.5),
        "ap_count_delta_p90": quantile(0.9),
        "ap_count_delta_p99": quantile(0.99),
    }
    (out_dir / "m25_hardness.json").write_text(
        json.dumps(hardness, indent=2),
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

    yield_series = {f"M22_{m}": series_for(f"M22_{m}", "yield") for m in hazard_modes}
    yield_series["Random"] = series_for("random", "yield")
    rand_lo = []
    rand_hi = []
    for frac in budgets:
        mean_y, ci_y, _, _ = random_ci[frac]
        rand_lo.append(mean_y - ci_y)
        rand_hi.append(mean_y + ci_y)
    save_line_plot(
        out_dir / "m25_yield_vs_budget.png",
        budgets,
        yield_series,
        "Yield vs budget (M25)",
        "budget fraction",
        "survival rate in selected",
        band=(rand_lo, rand_hi),
    )

    avoided_series = {f"M22_{m}": series_for(f"M22_{m}", "bad_tests_avoided_vs_random") for m in hazard_modes}
    avoided_series["Random"] = series_for("random", "bad_tests_avoided_vs_random")
    save_line_plot(
        out_dir / "m25_bad_tests_avoided_vs_budget.png",
        budgets,
        avoided_series,
        "Bad tests avoided vs budget (M25)",
        "budget fraction",
        "bad tests avoided vs random",
    )

    # compute saved at 1 day
    key_1d = f"saved_sec_cost{int(86400)}"
    saved_series = {f"M22_{m}": series_for(f"M22_{m}", key_1d) for m in hazard_modes}
    save_line_plot(
        out_dir / "m25_compute_saved_vs_budget_1d.png",
        budgets,
        saved_series,
        "Compute saved vs budget (cost=1d)",
        "budget fraction",
        "compute-seconds saved",
    )

    # hazard count histogram and bins (use harmonic score)
    if z_counts:
        save_hist(
            out_dir / "m25_hazard_count_hist.png",
            z_counts,
            "ap_count_delta histogram on Z (M25)",
            "ap_count_delta",
            "count",
        )
        hazard_vals = [hazard_harmonic[p] for p in z_ps]
        surv_vals = [survive_q1[p] for p in z_ps]
        if hazard_vals:
            bins = np.linspace(min(hazard_vals), max(hazard_vals), 21)
            bin_centers = []
            bin_means = []
            for i in range(len(bins) - 1):
                mask = [(bins[i] <= h < bins[i + 1]) for h in hazard_vals]
                if any(mask):
                    vals = [sv for sv, m in zip(surv_vals, mask) if m]
                    bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
                    bin_means.append(float(np.mean(vals)))
            if bin_centers:
                save_line_plot(
                    out_dir / "m25_hazard_vs_survival_bins.png",
                    bin_centers,
                    {"survival": bin_means},
                    "Survival vs hazard bins on Z (M25)",
                    "hazard bin center",
                    "P(survive_Q1)",
                )

    # table for selected budgets
    table_budgets = [0.01, 0.02, 0.10, 0.20, 0.50]
    label_map = {"random": "Random"}
    for mode in hazard_modes:
        label_map[f"M22_{mode}"] = f"M22_{mode}"
    tex_lines = [
        r"\begin{tabular}{lrrrl}\hline",
        r"Budget & Method & Yield & Bad tests & Saved (1h/1d) \\ \hline",
    ]
    for frac in table_budgets:
        for method in ["random"] + [f"M22_{m}" for m in hazard_modes]:
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
    (out_dir / "m25_table.tex").write_text("\n".join(tex_lines), encoding="utf-8")

    # queue snapshot
    queue_csv = out_dir / "m25_queue.csv"
    with queue_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "p",
            "killed_Q0",
            "killed_Q1",
            "survive_Q1",
            "hazard_binary",
            "hazard_count",
            "hazard_harmonic",
            "rank_binary",
            "rank_count",
            "rank_harmonic",
        ])
        for p in orders[hazard_modes[0]]:
            w.writerow([
                p,
                killed_q0[p],
                killed_q1[p],
                survive_q1[p],
                hazard_binary[p],
                hazard_count[p],
                hazard_harmonic[p],
                ranks.get("binary", {}).get(p, ""),
                ranks.get("count", {}).get(p, ""),
                ranks.get("harmonic", {}).get(p, ""),
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
        "hazard_modes": hazard_modes,
        "mersenne_strict": mersenne_strict,
        "seed": args.seed,
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m25_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    print(f"OK: wrote M25 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
