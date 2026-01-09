#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--layers", type=str, required=True)
    p.add_argument("--count", type=int, default=1000000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def allowed_residues(B: int, p: int) -> List[int]:
    inv = pow(B % p, -1, p)
    forbid = {inv % p, (-inv) % p}
    return [r for r in range(p) if r not in forbid]


def combine_residues(residues: List[int], mod: int, p: int, allowed: List[int]) -> Tuple[List[int], int]:
    inv = pow(mod, -1, p)
    new: List[int] = []
    for r in residues:
        r_mod = r % p
        for a in allowed:
            t = ((a - r_mod) * inv) % p
            new.append(r + mod * t)
    new.sort()
    return new, mod * p


def generate_gaps(residues: List[int], L: int, count: int) -> Tuple[List[int], float]:
    n_res = len(residues)
    if n_res == 0 or count <= 1:
        return [], 0.0
    needed_blocks = (count + n_res - 1) // n_res
    gaps: List[int] = []
    prev_t = None
    start = time.perf_counter()
    produced = 0
    for q in range(needed_blocks):
        base = q * L
        for r in residues:
            t = base + r
            if prev_t is not None:
                gaps.append(int(t - prev_t))
            prev_t = t
            produced += 1
            if produced >= count:
                elapsed = time.perf_counter() - start
                return gaps, elapsed
    elapsed = time.perf_counter() - start
    return gaps, elapsed


def save_line_plot(path: Path, xs: List[int], ys: List[float], title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, ys, marker="o")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_hist(path: Path, data: List[int], title: str, xlabel: str, ylabel: str, bins: int = 60) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
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
    primes = parse_int_list(args.layers)
    if not primes:
        raise ValueError("--layers is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    residues = [0]
    L = 1

    summary_rows = []
    layer_counts = []
    survival_rates = []
    throughputs = []

    for i, p in enumerate(primes, start=1):
        allowed = allowed_residues(args.B, p)
        residues, L = combine_residues(residues, L, p, allowed)
        survival_rate = len(residues) / float(L)
        gaps, elapsed = generate_gaps(residues, L, args.count)
        throughput = float(args.count) / elapsed if elapsed > 0 else 0.0

        layer_counts.append(i)
        survival_rates.append(survival_rate)
        throughputs.append(throughput)

        if gaps:
            gaps_arr = np.array(gaps, dtype=np.int64)
            gap_stats = {
                "mean_gap": float(np.mean(gaps_arr)),
                "median_gap": float(np.median(gaps_arr)),
                "min_gap": int(np.min(gaps_arr)),
                "max_gap": int(np.max(gaps_arr)),
            }
        else:
            gap_stats = {"mean_gap": 0.0, "median_gap": 0.0, "min_gap": 0, "max_gap": 0}

        summary_rows.append({
            "layers": i,
            "primes": ",".join(str(x) for x in primes[:i]),
            "L": int(L),
            "allowed_residues": int(len(residues)),
            "survival_rate": float(survival_rate),
            "throughput": float(throughput),
            "count": int(args.count),
            **gap_stats,
        })

        if i == len(primes):
            if gaps:
                save_hist(
                    out_dir / "m14_candidate_gap_hist.png",
                    gaps,
                    f"Candidate gap histogram (layers={i})",
                    "gap in t",
                    "count",
                )

    save_line_plot(
        out_dir / "m14_survival_vs_layers.png",
        layer_counts,
        survival_rates,
        "Survival rate vs layers",
        "layers",
        "survival rate",
    )
    save_line_plot(
        out_dir / "m14_throughput_vs_layers.png",
        layer_counts,
        throughputs,
        "Throughput vs layers",
        "layers",
        "candidates/sec",
    )

    summary_csv = out_dir / "m14_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    summary_json = out_dir / "m14_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"OK: wrote {summary_csv}")


if __name__ == "__main__":
    main()
