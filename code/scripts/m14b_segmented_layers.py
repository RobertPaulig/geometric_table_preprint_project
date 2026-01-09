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
    p.add_argument("--layers", type=str, default="")
    p.add_argument("--layer-count", type=int, default=0)
    p.add_argument("--segment-len", type=int, default=200000)
    p.add_argument("--segments", type=int, default=4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def gen_primes(start: int, count: int) -> List[int]:
    primes: List[int] = []
    n = max(2, start)
    while len(primes) < count:
        is_prime = True
        d = 2
        while d * d <= n:
            if n % d == 0:
                is_prime = False
                break
            d += 1 if d == 2 else 2
        if is_prime:
            primes.append(n)
        n += 1 if n == 2 else 2
    return primes


def forbidden_residues(B: int, p: int) -> Tuple[int, int]:
    inv = pow(B % p, -1, p)
    return inv % p, (-inv) % p


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
    if not primes and args.layer_count > 0:
        primes = gen_primes(start=13, count=args.layer_count)
    if not primes:
        raise ValueError("Provide --layers or --layer-count")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    segment_len = int(args.segment_len)
    segments = int(args.segments)
    total_t = segment_len * segments

    forb = {p: forbidden_residues(args.B, p) for p in primes}

    layer_counts = []
    survival_rates = []
    throughputs = []
    summary_rows = []
    gap_hist: List[int] = []

    for i in range(1, len(primes) + 1):
        primes_i = primes[:i]
        start_time = time.perf_counter()
        survivors = 0
        for s in range(segments):
            base = int(rng.integers(0, 10**9)) + s * segment_len
            t_vals = base + np.arange(segment_len, dtype=np.int64)
            mask = np.ones(segment_len, dtype=bool)
            for p in primes_i:
                r = t_vals % p
                f1, f2 = forb[p]
                mask &= (r != f1) & (r != f2)
            survivors += int(mask.sum())
            if i == len(primes) and s == 0:
                surv = t_vals[mask]
                if surv.size > 1:
                    gap_hist = (surv[1:] - surv[:-1]).astype(int).tolist()
        elapsed = time.perf_counter() - start_time
        survival = survivors / float(total_t)
        throughput = float(total_t) / elapsed if elapsed > 0 else 0.0

        layer_counts.append(i)
        survival_rates.append(float(survival))
        throughputs.append(float(throughput))

        row = {
            "layers": i,
            "primes": ",".join(str(p) for p in primes_i),
            "segment_len": segment_len,
            "segments": segments,
            "survival_rate": float(survival),
            "throughput": float(throughput),
        }
        summary_rows.append(row)

    save_line_plot(
        out_dir / "m14b_survival_vs_layers.png",
        layer_counts,
        survival_rates,
        "Segmented survival vs layers",
        "layers",
        "survival rate",
    )
    save_line_plot(
        out_dir / "m14b_throughput_vs_layers.png",
        layer_counts,
        throughputs,
        "Segmented throughput vs layers",
        "layers",
        "candidates/sec",
    )
    if gap_hist:
        save_hist(
            out_dir / "m14b_candidate_gap_hist.png",
            gap_hist,
            f"Candidate gap histogram (layers={len(primes)})",
            "gap in t",
            "count",
        )

    summary_csv = out_dir / "m14b_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    summary_json = out_dir / "m14b_summary.json"
    summary_json.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    print(f"OK: wrote {summary_csv}")


if __name__ == "__main__":
    main()
