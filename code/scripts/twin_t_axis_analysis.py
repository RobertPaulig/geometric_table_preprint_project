#!/usr/bin/env python
# code/scripts/twin_t_axis_analysis.py
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wheel-csv", type=str, required=True)
    p.add_argument("--t-max", type=int, default=200000)
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m5")
    p.add_argument("--max-mod", type=int, default=2000)
    p.add_argument("--topK", type=int, default=30)
    p.add_argument("--check-prime", type=int, default=0, help="optional prime for divisibility checks")
    p.add_argument("--perm-iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=12345)
    return p.parse_args()


def read_twins(path: Path, t_max: int) -> np.ndarray:
    x = np.zeros(t_max + 1, dtype=np.int8)
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["t"])
            if t > t_max:
                continue
            x[t] = int(row["is_twin"])
    return x[1:]


def fft_power(y: np.ndarray) -> np.ndarray:
    Y = np.fft.rfft(y)
    return (np.abs(Y) ** 2).astype(float)


def autocorr_fft(y: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(y)
    f = np.fft.rfft(y, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    ac = ac / ac[0] if ac[0] != 0 else ac
    return ac


def save_line_png(path: Path, xs: List[int], ys: List[float], title: str, xlabel: str, ylabel: str, logy: bool = False) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.plot(xs, ys, linewidth=1.0)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def mod_lift(twin_idx: np.ndarray, T: int, m: int, pbar: float) -> Tuple[List[float], float]:
    # counts per residue for all t in [1..T]
    base = T // m
    rem = T % m
    counts = np.full(m, base, dtype=int)
    if rem:
        counts[:rem] += 1
    # twin counts per residue
    twins = np.bincount(twin_idx % m, minlength=m)
    lift = []
    chi2 = 0.0
    for r in range(m):
        pr = twins[r] / counts[r] if counts[r] else 0.0
        lift.append(pr / pbar if pbar > 0 else 0.0)
        exp = counts[r] * pbar
        if exp > 0:
            chi2 += (twins[r] - exp) ** 2 / exp
    return lift, chi2


def save_mod_lift_png(path: Path, lifts: List[float], m: int) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.bar(list(range(m)), lifts, width=0.9)
    ax.set_title(f"Residue lift (mod {m})")
    ax.set_xlabel("r")
    ax.set_ylabel("lift")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    x = read_twins(Path(args.wheel_csv), args.t_max)
    T = len(x)
    x_mean = float(x.mean())
    y = x.astype(float) - x_mean
    twin_idx = np.nonzero(x)[0] + 1

    power = fft_power(y)
    freqs = np.arange(len(power))
    power_no_dc = power.copy()
    power_no_dc[0] = 0.0
    top_idx = np.argsort(power_no_dc)[-args.topK:][::-1]
    top_rows = []
    for rank, idx in enumerate(top_idx, start=1):
        if idx == 0:
            continue
        period = T / idx
        top_rows.append([rank, int(idx), float(period), float(power[idx])])

    fft_csv = out_dir / "fft_top_peaks.csv"
    with fft_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "f_idx", "period_estimate", "power"])
        w.writerows(top_rows)

    save_line_png(
        out_dir / "fft_power.png",
        list(freqs[1:]),
        [float(v) for v in power[1:]],
        title="FFT power (twin indicator)",
        xlabel="frequency index",
        ylabel="power",
        logy=True,
    )

    # permutation control (shuffle)
    perm_max = []
    for _ in range(args.perm_iters):
        xs = x.copy()
        rng.shuffle(xs)
        ys = xs.astype(float) - xs.mean()
        p = fft_power(ys)
        p[0] = 0.0
        perm_max.append(float(np.max(p)))
    perm_max.sort()

    peak_sig = []
    for _, idx, _, powv in top_rows:
        p_like = sum(1 for v in perm_max if v >= powv) / len(perm_max)
        peak_sig.append({"f_idx": idx, "power": powv, "perm_p_like": p_like})
    Path(out_dir / "fft_peak_significance.json").write_text(
        json.dumps(peak_sig, indent=2), encoding="utf-8"
    )

    # autocorr
    max_lag = min(5000, T - 1)
    ac = autocorr_fft(y, max_lag)
    save_line_png(
        out_dir / "autocorr.png",
        list(range(1, max_lag + 1)),
        [float(v) for v in ac[1:]],
        title="Autocorrelation (twin indicator)",
        xlabel="lag",
        ylabel="corr",
        logy=False,
    )

    top_lags = sorted(range(1, max_lag + 1), key=lambda i: abs(ac[i]), reverse=True)[:20]
    with (out_dir / "autocorr_top_lags.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["lag", "corr"])
        for lag in top_lags:
            w.writerow([lag, float(ac[lag])])

    # mod lifts + chi2 summary
    mod_summary = []
    for m in range(2, args.max_mod + 1):
        lift, chi2 = mod_lift(twin_idx, T, m, x_mean)
        mod_summary.append([m, max(lift), min(lift), chi2, x_mean])
    with (out_dir / "mod_lift_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "max_lift", "min_lift", "chi2", "pbar"])
        w.writerows(mod_summary)

    mod_summary_sorted = sorted(mod_summary, key=lambda r: r[3], reverse=True)[:20]
    with (out_dir / "mod_chi2_top.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "max_lift", "min_lift", "chi2", "pbar"])
        w.writerows(mod_summary_sorted)

    # example lifts for selected moduli
    for m in (210, 420, 840):
        if m <= args.max_mod:
            lift, _ = mod_lift(twin_idx, T, m, x_mean)
            save_mod_lift_png(out_dir / f"mod_m{m}_lift.png", lift, m)

    # divisibility checks for top lags and top moduli
    div_summary = {}
    if args.check_prime:
        p = args.check_prime
        div_summary = {
            "prime": p,
            "top_lags_count": len(top_lags),
            "top_lags_divisible": sum(1 for lag in top_lags if lag % p == 0),
            "top_mod_count": len(mod_summary_sorted),
            "top_mod_divisible": sum(1 for r in mod_summary_sorted if int(r[0]) % p == 0),
        }

    summary = {
        "counts": {"T": int(T), "twin_count": int(sum(x)), "twin_rate": float(x_mean)},
        "check_prime_summary": div_summary,
    }
    Path(out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"OK: wrote M5 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
