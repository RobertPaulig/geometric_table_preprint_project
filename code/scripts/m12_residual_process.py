#!/usr/bin/env python
# code/scripts/m12_residual_process.py
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m-list", type=str, default="10,12,14")
    p.add_argument("--t-max", type=int, default=200000)
    p.add_argument("--p-max", type=int, default=200000)
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m12")
    return p.parse_args()


def open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_twins(path: Path, t_max: int) -> np.ndarray:
    x = np.zeros(t_max + 1, dtype=np.int8)
    with open_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["t"])
            if t > t_max:
                continue
            x[t] = int(row["is_twin"])
    return x[1:]


def lcm_upto(m: int) -> int:
    val = 1
    for i in range(1, m + 1):
        val = val * i // math.gcd(val, i)
    return val


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    r = int(math.isqrt(n))
    for d in range(3, r + 1, 2):
        if n % d == 0:
            return False
    return True


def next_primes(start: int, count: int) -> List[int]:
    out = []
    n = start + 1
    while len(out) < count:
        if is_prime(n):
            out.append(n)
        n += 1
    return out


def allowed_mask(B: int, t_max: int, primes: List[int]) -> np.ndarray:
    mask = np.ones(t_max, dtype=bool)
    for p in primes:
        inv = pow(B % p, -1, p)
        forbid = {inv % p, (-inv) % p}
        for r in forbid:
            mask[r::p] = False
    return mask


def residual_sequence(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    allowed_x = x[mask]
    return allowed_x.astype(np.int8)


def gap_distribution(x: np.ndarray) -> List[int]:
    idx = np.flatnonzero(x)
    if len(idx) < 2:
        return []
    return list(np.diff(idx))


def dispersion_index(x: np.ndarray, window: int = 5000) -> float:
    n = len(x)
    if n < window:
        return 0.0
    counts = []
    for start in range(0, n - window + 1, window):
        counts.append(int(x[start:start + window].sum()))
    mean = np.mean(counts)
    var = np.var(counts)
    return float(var / mean) if mean > 0 else 0.0


def autocorr(x: np.ndarray, max_lag: int = 200) -> np.ndarray:
    x = x.astype(float)
    x = x - x.mean()
    n = len(x)
    ac = []
    denom = np.dot(x, x)
    if denom == 0:
        return np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        ac.append(np.dot(x[:-lag], x[lag:]) / denom)
    return np.array(ac)


def save_line(path: Path, xs: List[int], ys: List[float], title: str, xlabel: str, ylabel: str, logy: bool = False) -> None:
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


def save_hist(path: Path, data: List[int], title: str, xlabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.hist(data, bins=50, density=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("density")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m_list = [int(x.strip()) for x in args.m_list.split(",") if x.strip()]
    disp_rows = []

    for m in m_list:
        B = lcm_upto(m)
        p0, p1, p2 = next_primes(m, 3)
        primes = [p0, p1, p2]
        wheel_csv = Path(f"out/wheel_scan_m{m}_t{args.t_max}.csv")
        if not wheel_csv.exists():
            wheel_csv = Path(f"out/wheel_scan_m{m}_t{args.t_max}.csv.gz")
        x = read_twins(wheel_csv, args.t_max)
        mask = allowed_mask(B, args.t_max, primes)
        x_res = residual_sequence(x, mask)
        gaps = gap_distribution(x_res)
        ac = autocorr(x_res, max_lag=200)
        disp = dispersion_index(x_res, window=5000)

        m_dir = out_dir / f"m{m}"
        m_dir.mkdir(parents=True, exist_ok=True)
        if gaps:
            save_hist(m_dir / "residual_gaps.png", gaps, f"Residual gaps (m={m})", "gap")
        save_line(m_dir / "residual_acf.png", list(range(1, len(ac) + 1)), ac.tolist(), f"Residual ACF (m={m})", "lag", "acf")
        save_line(m_dir / "residual_dispersion.png", [1], [disp], f"Dispersion (m={m})", "index", "Var/Mean")

        summary = {
            "m": m,
            "B": B,
            "p0": p0,
            "p1": p1,
            "p2": p2,
            "t_max": args.t_max,
            "allowed_count": int(mask.sum()),
            "residual_events": int(x_res.sum()),
            "mean_gap": float(np.mean(gaps) if gaps else 0.0),
            "dispersion": float(disp),
        }
        Path(m_dir / "residual_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        disp_rows.append(summary)

    # compare dispersion
    xs = [row["m"] for row in disp_rows]
    ys = [row["dispersion"] for row in disp_rows]
    save_line(out_dir / "m12_compare_dispersion.png", xs, ys, "Dispersion by m (residual)", "m", "Var/Mean")
    Path(out_dir / "m12_summary.json").write_text(json.dumps(disp_rows, indent=2), encoding="utf-8")

    print(f"OK: wrote residual artifacts to {out_dir}")


if __name__ == "__main__":
    main()
