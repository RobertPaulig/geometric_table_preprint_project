#!/usr/bin/env python
# code/scripts/wheel_wave_overlay.py
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def lcm_upto(m: int) -> int:
    v = 1
    for k in range(1, m + 1):
        v = math.lcm(v, k)
    return v


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wheel-csv", type=str, required=True)
    p.add_argument("--m", type=int, default=12)
    p.add_argument("--K", type=int, default=120)
    p.add_argument("--H", type=int, default=220)
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m4")
    p.add_argument("--out-csv", type=str, default="out/wave_atlas/wheel_wave_m12_features.csv")
    p.add_argument("--out-json", type=str, default="out/wave_atlas/wheel_wave_m12_summary.json")
    return p.parse_args()


def diag_hits_stats(N: int, K: int) -> Tuple[int, int]:
    hits = [k for k in range(1, K + 1) if N % k == 0]
    run_len = 0
    cur = 0
    prev = None
    for k in hits:
        if prev is None or k == prev + 1:
            cur += 1
        else:
            run_len = max(run_len, cur)
            cur = 1
        prev = k
    run_len = max(run_len, cur)
    return len(hits), run_len


def t_small_div_count(t: int, K: int) -> int:
    return sum(1 for d in range(1, K + 1) if t % d == 0)


def t_max_consecutive_div_run(t: int, K: int) -> int:
    s = 1
    for d in range(2, K + 1):
        if t % d == 0:
            s = d
        else:
            break
    return s


def local_window_stats(N: int, H: int, K: int) -> Tuple[float, float, float]:
    start = max(1, N - H // 2)
    end = start + H - 1
    counts = [0] * H
    for k in range(1, K + 1):
        first = ((start + k - 1) // k) * k
        for n in range(first, end + 1, k):
            counts[n - start] += 1
    row_arr = np.array(counts, dtype=float)
    occ_density = float(row_arr.sum() / (H * K))
    return occ_density, float(row_arr.mean()), float(row_arr.var())


def main() -> None:
    args = parse_args()
    B = lcm_upto(args.m)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with Path(args.wheel_csv).open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["t"])
            N = int(row["N"])
            is_twin = int(row["is_twin"])
            minus_is_prime = int(row["minus_is_prime"])
            plus_is_prime = int(row["plus_is_prime"])
            spf_minus = int(row["spf_minus"])
            spf_plus = int(row["spf_plus"])
            minus_is_semiprime = int(row.get("minus_is_semiprime", 0))
            plus_is_semiprime = int(row.get("plus_is_semiprime", 0))

            diag_hits_count, diag_run_len = diag_hits_stats(N, args.K)
            diag_density = diag_hits_count / args.K
            t_div_count = t_small_div_count(t, args.K)
            t_run = t_max_consecutive_div_run(t, args.K)
            occ_density_local, mean_divcount, var_divcount = local_window_stats(N, args.H, args.K)

            rows.append({
                "m": args.m,
                "B": B,
                "t": t,
                "N": N,
                "is_twin": is_twin,
                "minus_is_prime": minus_is_prime,
                "plus_is_prime": plus_is_prime,
                "spf_minus": spf_minus,
                "spf_plus": spf_plus,
                "minus_is_semiprime": minus_is_semiprime,
                "plus_is_semiprime": plus_is_semiprime,
                "diag_hits_count_K": diag_hits_count,
                "diag_run_len_K": diag_run_len,
                "diag_hits_density": diag_density,
                "t_small_div_count": t_div_count,
                "t_max_consecutive_div_run": t_run,
                "occ_density_local": occ_density_local,
                "mean_row_divcount_local": mean_divcount,
                "var_row_divcount_local": var_divcount,
            })

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            w.writeheader()
            w.writerows(rows)

    def summarize(group: List[Dict[str, float]]) -> Dict[str, float]:
        if not group:
            return {}
        def mean(key: str) -> float:
            return float(sum(r[key] for r in group) / len(group))
        def median(key: str) -> float:
            vals = sorted(r[key] for r in group)
            mid = len(vals) // 2
            if len(vals) % 2 == 0:
                return float(0.5 * (vals[mid - 1] + vals[mid]))
            return float(vals[mid])
        return {
            "diag_run_len_mean": mean("diag_run_len_K"),
            "diag_run_len_median": median("diag_run_len_K"),
            "t_small_div_mean": mean("t_small_div_count"),
            "t_small_div_median": median("t_small_div_count"),
        }

    twins = [r for r in rows if r["is_twin"] == 1]
    non = [r for r in rows if r["is_twin"] == 0]

    top_run = sorted(rows, key=lambda r: r["diag_run_len_K"], reverse=True)[:10]
    summary = {
        "counts": {
            "n_rows": len(rows),
            "n_twins": len(twins),
            "n_non_twins": len(non),
        },
        "summary_twin": summarize(twins),
        "summary_non_twin": summarize(non),
        "top10_diag_run_len": [
            {"N": r["N"], "t": r["t"], "run_len": r["diag_run_len_K"], "is_twin": r["is_twin"]}
            for r in top_run
        ],
        "sanity": {
            "K": args.K,
            "m": args.m,
        },
    }
    Path(args.out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # plots
    import matplotlib.pyplot as plt

    def save_hist(path: Path, vals_t: List[int], vals_n: List[int], title: str, xlabel: str) -> None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(vals_n, bins=20, alpha=0.6, label="non-twin", color="gray")
        ax.hist(vals_t, bins=20, alpha=0.7, label="twin", color="steelblue")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.legend()
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    def save_scatter(path: Path, vals_t: List[Tuple[float, float]], vals_n: List[Tuple[float, float]]) -> None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if vals_n:
            ax.scatter([v[0] for v in vals_n], [v[1] for v in vals_n], s=8, alpha=0.5, color="gray", label="non-twin")
        if vals_t:
            ax.scatter([v[0] for v in vals_t], [v[1] for v in vals_t], s=10, alpha=0.8, color="steelblue", label="twin")
        ax.set_xlabel("diag_run_len_K")
        ax.set_ylabel("t_small_div_count")
        ax.set_title("Run length vs t divisor count")
        ax.legend()
        fig.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)

    run_t = [r["diag_run_len_K"] for r in twins]
    run_n = [r["diag_run_len_K"] for r in non]
    tdiv_t = [r["t_small_div_count"] for r in twins]
    tdiv_n = [r["t_small_div_count"] for r in non]

    save_hist(out_dir / "m4_runlen_hist.png", run_t, run_n, "diag_run_len_K (twin vs non)", "diag_run_len_K")
    save_hist(out_dir / "m4_tdiv_hist.png", tdiv_t, tdiv_n, "t_small_div_count (twin vs non)", "t_small_div_count")
    save_scatter(
        out_dir / "m4_scatter_runlen_vs_tdiv.png",
        [(r["diag_run_len_K"], r["t_small_div_count"]) for r in twins],
        [(r["diag_run_len_K"], r["t_small_div_count"]) for r in non],
    )

    print(f"OK: wrote overlay to {out_csv}, {args.out_json}, plots in {out_dir}")


if __name__ == "__main__":
    main()
