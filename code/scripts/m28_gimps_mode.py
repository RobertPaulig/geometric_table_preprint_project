#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p-range", type=str, required=True, help="inclusive range: a,b")
    p.add_argument("--tf-bits-list", type=str, required=True, help="e.g. 64,68,72,76")
    p.add_argument("--Q0", type=int, required=True)
    p.add_argument("--q-horizon", type=int, required=True)
    p.add_argument("--mersenne-strict", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--prp-cost-hours", type=str, required=True, help="e.g. 1,24,168")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_pair(s: str) -> Tuple[int, int]:
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        raise ValueError("--p-range must be a,b")
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


def primes_in_range(a: int, b: int) -> List[int]:
    if b < 2 or a > b:
        return []
    a = max(a, 2)
    if b <= 5_000_000:
        primes = sieve_primes(b)
        return [p for p in primes if p >= a]

    def is_probable_prime(n: int) -> bool:
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for sp in small_primes:
            if n == sp:
                return True
            if n % sp == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a0 in [2, 325, 9375, 28178, 450775, 9780504, 1795265022]:
            a0 %= n
            if a0 == 0:
                continue
            x = pow(a0, d, n)
            if x == 1 or x == n - 1:
                continue
            composite = True
            for _ in range(s - 1):
                x = (x * x) % n
                if x == n - 1:
                    composite = False
                    break
            if composite:
                return False
        return True

    out = []
    start = a if a % 2 == 1 else a + 1
    if a <= 2 <= b:
        out.append(2)
    for n in range(start, b + 1, 2):
        if is_probable_prime(n):
            out.append(n)
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


def tf_bits_to_q_limit(tf_bits: int, p: int, q_horizon: int) -> int:
    # Controlled sweep: map TF "depth" to a Q-limit in the same q-axis units as M26/M27.
    # Monotone in tf_bits; weak dependence on p to keep the value explicit in-row.
    scale = {64: 1_000_000, 68: 2_000_000, 72: 5_000_000, 76: 10_000_000}.get(tf_bits)
    if scale is None:
        scale = int(round(10 ** ((tf_bits - 60) / 8))) * 1_000_000
    q_lim = int(min(q_horizon, max(20_000, scale)))
    # tie-breaker: +/- up to 0.1% based on p (deterministic, keeps "computed from p")
    q_lim = int(min(q_horizon, max(20_000, q_lim + (p % 997) - 498)))
    return q_lim


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_logit(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 800) -> Tuple[np.ndarray, float]:
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


def save_line_plot(path: Path, xs: List[float], series: Dict[str, List[float]],
                   title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
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


def save_multi_enrichment_plot(path: Path, xs: List[float], curves_by_bits: Dict[int, List[float]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for tf_bits, ys in sorted(curves_by_bits.items()):
        ax.plot(xs, ys, marker="o", label=f"tf_bits={tf_bits}")
    ax.axhline(1.0, linestyle="--", color="#888888", linewidth=1)
    ax.set_title("Enrichment vs budget (passes_tf=1)")
    ax.set_xlabel("budget fraction")
    ax.set_ylabel("enrichment vs random")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    p_a, p_b = parse_pair(args.p_range)
    tf_bits_list = parse_int_list(args.tf_bits_list)
    if not tf_bits_list:
        raise ValueError("tf-bits-list is empty")
    prp_cost_hours = [float(x) for x in args.prp_cost_hours.split(",") if x.strip()]
    if not prp_cost_hours:
        raise ValueError("prp-cost-hours is empty")
    if args.Q0 <= 0:
        raise ValueError("Q0 must be positive")
    if args.q_horizon <= 0:
        raise ValueError("q-horizon must be positive")

    mersenne_strict = int(args.mersenne_strict) == 1
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Candidate exponents p (prime-only)
    p_candidates = primes_in_range(p_a, p_b)
    if not p_candidates:
        raise ValueError("No primes in p-range")
    if max(p_candidates) >= args.q_horizon:
        # In this q-axis model, ord_q(2)=p implies q>p. If p exceeds horizon, deaths are impossible.
        # We keep the run deterministic and record the constraint in summary.
        pass

    # Fixed Q grid to support both features and labels.
    Q_grid = sorted({args.Q0, args.q_horizon, 20_000, 50_000, 100_000, 200_000, 500_000,
                     1_000_000, 2_000_000, 5_000_000, 10_000_000, 20_000_000, 50_000_000})
    Q_grid = [Q for Q in Q_grid if Q <= args.q_horizon and Q >= args.Q0]
    Q_grid = sorted(set(Q_grid))

    Q_max = args.q_horizon
    primes_q = sieve_primes(Q_max)
    primes_q = [q for q in primes_q if q != 2]
    primes_factor = sieve_primes(int(math.isqrt(Q_max)) + 1)

    # ord_q(2) for all primes q (filtered)
    ords: Dict[int, int] = {}
    for q in primes_q:
        if mersenne_strict and (q % 8 not in (1, 7)):
            continue
        ords[q] = ord_mod_2(q, primes_factor)

    p_set = set(p_candidates)
    killed_by_Q: Dict[int, Dict[int, int]] = {Q: {p: 0 for p in p_candidates} for Q in Q_grid}

    for q, d in ords.items():
        if d not in p_set:
            continue
        for Q in Q_grid:
            if q <= Q:
                killed_by_Q[Q][d] = 1

    # Hazard features (delta Q0->Q) for all Q in grid.
    ap_count_delta: Dict[int, Dict[int, int]] = {Q: {p: 0 for p in p_candidates} for Q in Q_grid}
    ap_harm_delta: Dict[int, Dict[int, float]] = {Q: {p: 0.0 for p in p_candidates} for Q in Q_grid}

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
            if p not in p_set:
                continue
            for Q in Q_grid:
                if q <= Q:
                    ap_count_delta[Q][p] += 1
                    ap_harm_delta[Q][p] += p / (q - 1)

    # Build dataset rows per (p, tf_bits) with post-TF labels.
    rows: List[Dict[str, object]] = []
    for p in p_candidates:
        for tf_bits in tf_bits_list:
            q_tf_limit = tf_bits_to_q_limit(tf_bits, p, args.q_horizon)
            # Ensure the limit exists in our grid by snapping upward.
            q_tf_snap = min([Q for Q in Q_grid if Q >= q_tf_limit], default=args.q_horizon)
            passes_tf = 1 if killed_by_Q[q_tf_snap][p] == 0 else 0
            dies_by_horizon = 0
            if passes_tf == 1:
                dies_by_horizon = 1 if killed_by_Q[args.q_horizon][p] == 1 else 0

            row = {
                "p": p,
                "tf_bits": tf_bits,
                "q_tf_limit": q_tf_snap,
                "passes_tf": passes_tf,
                "dies_by_horizon": dies_by_horizon,
                "ap_count_delta_tf": ap_count_delta[q_tf_snap][p],
                "ap_harm_delta_tf": ap_harm_delta[q_tf_snap][p],
            }
            rows.append(row)

    dataset_csv = out_dir / "m28_dataset.csv.gz"
    header = [
        "p",
        "tf_bits",
        "q_tf_limit",
        "passes_tf",
        "dies_by_horizon",
        "ap_count_delta_tf",
        "ap_harm_delta_tf",
    ]
    with gzip.open(dataset_csv, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)

    # Metrics per tf_bits on passes_tf=1 subset.
    budgets = [0.005, 0.01, 0.02, 0.05, 0.10]
    metrics_rows = []
    savings_rows = []
    enrichment_curves: Dict[int, List[float]] = {}

    for tf_bits in tf_bits_list:
        sub = [r for r in rows if int(r["tf_bits"]) == tf_bits and int(r["passes_tf"]) == 1]
        n_pass = len(sub)
        n_total = len([r for r in rows if int(r["tf_bits"]) == tf_bits])
        if n_pass < 20:
            metrics_rows.append({
                "tf_bits": tf_bits,
                "N_total": n_total,
                "N_passes_tf": n_pass,
                "base_rate_dies_by_horizon": "",
                "auc": "",
                "note": "too_few_passes_tf",
            })
            enrichment_curves[tf_bits] = [float("nan") for _ in budgets]
            continue

        y = np.array([int(r["dies_by_horizon"]) for r in sub], dtype=float)
        base_rate = float(y.mean())
        X = np.array(
            [
                [
                    math.log(float(r["p"])),
                    float(r["ap_count_delta_tf"]),
                    math.log1p(float(r["ap_harm_delta_tf"])),
                    float(r["ap_harm_delta_tf"]) / (1.0 + float(r["ap_count_delta_tf"])),
                ]
                for r in sub
            ],
            dtype=float,
        )
        # standardize columns
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)

        w_vec, b = fit_logit(X, y, lr=0.2, steps=1200)
        logits = X @ w_vec + b
        probs = sigmoid(logits)
        x_grid, y_grid = isotonic_fit(probs, y)
        probs_iso = isotonic_predict(x_grid, y_grid, probs)

        auc = auc_score(y, probs_iso)
        order = np.argsort(probs_iso)[::-1]

        enrichments = []
        hit_rates = []
        for frac in budgets:
            k = max(1, int(round(frac * n_pass)))
            hits = float(y[order[:k]].mean())
            hit_rates.append(hits)
            enrichments.append(hits / base_rate if base_rate > 0 else float("nan"))
            for cost_h in prp_cost_hours:
                saved = (hits - base_rate) * k * cost_h * 3600.0
                savings_rows.append({
                    "tf_bits": tf_bits,
                    "budget_fraction": frac,
                    "prp_cost_hours": cost_h,
                    "saved_compute_seconds": saved,
                })

        enrichment_curves[tf_bits] = enrichments

        metrics_rows.append({
            "tf_bits": tf_bits,
            "N_total": n_total,
            "N_passes_tf": n_pass,
            "base_rate_dies_by_horizon": base_rate,
            "auc": auc,
            "hit_rate@1pct": hit_rates[1],
            "enrichment@1pct": enrichments[1],
            "hit_rate@5pct": hit_rates[3],
            "enrichment@5pct": enrichments[3],
            "note": "",
        })

        # queues (top 1%)
        k1 = max(1, int(round(0.01 * n_pass)))
        queue_path = out_dir / f"m28_queue_tf{tf_bits}_top1pct.csv"
        with queue_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["p", "score", "prob", "dies_by_horizon"])
            for idx in order[:k1]:
                w.writerow([int(sub[idx]["p"]), float(logits[idx]), float(probs_iso[idx]), int(y[idx])])

    metrics_csv = out_dir / "m28_metrics_by_tf_bits.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({k for row in metrics_rows for k in row.keys()})
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(metrics_rows)

    savings_csv = out_dir / "m28_savings_by_tf_bits.csv"
    with savings_csv.open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["tf_bits", "budget_fraction", "prp_cost_hours", "saved_compute_seconds"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(savings_rows)

    # Figures
    xs_bits = [float(x) for x in tf_bits_list]
    aucs = [float(next((r["auc"] for r in metrics_rows if r.get("tf_bits") == tf_bits and r.get("auc") != ""), float("nan")))
            for tf_bits in tf_bits_list]
    save_line_plot(
        out_dir / "m28_auc_by_tf_bits.png",
        xs_bits,
        {"AUC (passes_tf=1)": aucs},
        "AUC vs TF depth (post-TF task)",
        "tf_bits",
        "AUC",
    )
    save_multi_enrichment_plot(
        out_dir / "m28_enrichment_curves.png",
        budgets,
        enrichment_curves,
    )

    # Savings plot at 1% and 5% (per cost scenario)
    for frac in [0.01, 0.05]:
        series = {}
        for cost_h in prp_cost_hours:
            ys = []
            for tf_bits in tf_bits_list:
                match = next(
                    (
                        r for r in savings_rows
                        if int(r["tf_bits"]) == tf_bits and float(r["budget_fraction"]) == frac and float(r["prp_cost_hours"]) == cost_h
                    ),
                    None,
                )
                ys.append(float(match["saved_compute_seconds"]) if match else float("nan"))
            series[f"PRP={int(cost_h)}h"] = ys
        save_line_plot(
            out_dir / f"m28_savings_by_tf_bits_top{int(frac*100)}pct.png",
            xs_bits,
            series,
            f"Compute saved vs tf_bits (top {int(frac*100)}%)",
            "tf_bits",
            "compute-seconds saved",
        )

    # Aggregate savings figure (top1% @ 24h by default if present)
    default_cost = 24.0 if 24.0 in prp_cost_hours else prp_cost_hours[0]
    series_agg = {"top1%": [], "top5%": []}
    for tf_bits in tf_bits_list:
        for frac, key in [(0.01, "top1%"), (0.05, "top5%")]:
            match = next(
                (
                    r for r in savings_rows
                    if int(r["tf_bits"]) == tf_bits and float(r["budget_fraction"]) == frac and float(r["prp_cost_hours"]) == default_cost
                ),
                None,
            )
            series_agg[key].append(float(match["saved_compute_seconds"]) if match else float("nan"))
    save_line_plot(
        out_dir / "m28_savings_by_tf_bits.png",
        xs_bits,
        series_agg,
        f"Compute saved vs tf_bits (PRP={int(default_cost)}h)",
        "tf_bits",
        "compute-seconds saved",
    )

    # TeX table (compact)
    table_lines = [
        r"\begin{tabular}{lrrrr}\hline",
        r"TF bits & $N_{\mathrm{pass}}$ & base rate & AUC & enrich@1\% \\ \hline",
    ]
    for r in metrics_rows:
        if r.get("auc") in ("", None):
            continue
        table_lines.append(
            f"{int(r['tf_bits'])} & {int(r['N_passes_tf'])} & {float(r['base_rate_dies_by_horizon']):.3f} & "
            f"{float(r['auc']):.3f} & {float(r['enrichment@1pct']):.2f} \\\\"
        )
    table_lines.append(r"\hline\end{tabular}")
    (out_dir / "m28_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    # Summary + manifest
    summary = {
        "p_range": [p_a, p_b],
        "tf_bits_list": tf_bits_list,
        "Q0": args.Q0,
        "q_horizon": args.q_horizon,
        "mersenne_strict": mersenne_strict,
        "seed": args.seed,
        "n_p_candidates": len(p_candidates),
        "n_rows": len(rows),
        "skipped_due_to_runtime": False,
        "notes": [],
    }
    if max(p_candidates) >= args.q_horizon:
        summary["notes"].append("p_max_ge_q_horizon: deaths impossible for those p in this q-axis model")
    # per tf_bits counts
    counts = {}
    for tf_bits in tf_bits_list:
        sub_all = [r for r in rows if int(r["tf_bits"]) == tf_bits]
        sub_pass = [r for r in sub_all if int(r["passes_tf"]) == 1]
        sub_die = [r for r in sub_pass if int(r["dies_by_horizon"]) == 1]
        counts[str(tf_bits)] = {
            "N_total": len(sub_all),
            "N_passes_tf": len(sub_pass),
            "N_dies_by_horizon_among_passes": len(sub_die),
            "base_rate_dies_by_horizon_among_passes": (len(sub_die) / len(sub_pass)) if sub_pass else None,
        }
    summary["counts_by_tf_bits"] = counts
    (out_dir / "m28_dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "command": "python code/scripts/m28_gimps_mode.py",
        "params": {
            "p_range": [p_a, p_b],
            "tf_bits_list": tf_bits_list,
            "Q0": args.Q0,
            "q_horizon": args.q_horizon,
            "mersenne_strict": mersenne_strict,
            "seed": args.seed,
            "prp_cost_hours": prp_cost_hours,
        },
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m28_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK: wrote M28 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
