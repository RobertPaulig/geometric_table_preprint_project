#!/usr/bin/env python
# code/scripts/twin_t_axis_detrend.py
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np

from wave_atlas_io import open_text


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wheel-csv", type=str, required=True)
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--p0", type=int, default=0)
    p.add_argument("--t-max", type=int, default=0)
    p.add_argument("--seg-len", type=int, default=20000)
    p.add_argument("--smooth-len", type=int, default=5000)
    p.add_argument("--perm", type=int, default=100)
    p.add_argument("--cond-max-mod", type=int, default=0)
    p.add_argument("--cond-primes", type=str, default="")
    p.add_argument("--max-layer", type=int, default=0)
    p.add_argument("--out-dir", type=str, required=True)
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


def fft_power(y: np.ndarray) -> np.ndarray:
    Y = np.fft.rfft(y)
    return (np.abs(Y) ** 2).astype(float)


def autocorr_fft(y: np.ndarray, max_lag: int) -> np.ndarray:
    n = len(y)
    f = np.fft.rfft(y, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[: max_lag + 1]
    return ac / ac[0] if ac[0] != 0 else ac


def rolling_mean(x: np.ndarray, L: int) -> np.ndarray:
    if L <= 1:
        return x.astype(float)
    kernel = np.ones(L) / L
    return np.convolve(x, kernel, mode="same")


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


def save_spectrum_csv(path: Path, power: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open_text(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["f_idx", "power"])
        for i in range(1, len(power)):
            w.writerow([i, float(power[i])])


def save_multi_line(path: Path, series: List[Tuple[str, List[float]]], title: str, xlabel: str, ylabel: str, logy: bool = False) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for label, ys in series:
        xs = list(range(1, len(ys) + 1))
        ax.plot(xs, ys, linewidth=0.9, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def segment_fft(y: np.ndarray, seg_len: int, topk: int) -> Tuple[List[float], List[List[Tuple[int, float]]]]:
    n = len(y)
    segs = []
    peaks = []
    for start in range(0, n, seg_len):
        seg = y[start : start + seg_len]
        if len(seg) < seg_len:
            break
        segs.append(seg.mean())
        p = fft_power(seg - seg.mean())
        p[0] = 0.0
        idx = np.argsort(p)[-topk:][::-1]
        peaks.append([(int(i), float(p[i])) for i in idx])
    return segs, peaks


def conditioning_indices(T: int, p0: int, inv: int) -> List[int]:
    forbid = {inv % p0, (-inv) % p0}
    return [t for t in range(1, T + 1) if (t % p0) not in forbid]


def lcm(a: int, b: int) -> int:
    return a // math.gcd(a, b) * b


def allowed_residues(B: int, primes: List[int]) -> List[int]:
    L = 1
    forbids = []
    for p in primes:
        L = lcm(L, p)
        inv = pow(B % p, -1, p)
        forbids.append((p, {inv % p, (-inv) % p}))
    allowed = []
    for r in range(L):
        ok = True
        for p, forbid in forbids:
            if (r % p) in forbid:
                ok = False
                break
        if ok:
            allowed.append(r)
    return allowed


def conditional_spectrum(z: np.ndarray, B: int, primes: List[int]) -> Tuple[np.ndarray, int]:
    T = len(z)
    L = 1
    for p in primes:
        L = lcm(L, p)
    allowed = allowed_residues(B, primes)
    lengths = []
    starts = []
    for r in allowed:
        start = L if r == 0 else r
        if start > T:
            continue
        length = (T - start) // L + 1
        lengths.append(length)
        starts.append(start)
    if not lengths:
        return np.array([]), 0
    min_len = min(lengths)
    power_sum = None
    count = 0
    for start in starts:
        idxs = (start - 1) + np.arange(min_len) * L
        seg = z[idxs]
        seg = seg - seg.mean()
        p = fft_power(seg)
        p[0] = 0.0
        if power_sum is None:
            power_sum = p
        else:
            power_sum += p
        count += 1
    return power_sum / max(1, count), min_len


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


def next_primes(p0: int, count: int = 5) -> List[int]:
    primes = []
    n = p0 + 1
    while len(primes) < count:
        if is_prime(n):
            primes.append(n)
        n += 1
    return primes


def fit_harmonic(period: float, primes: List[int], max_d: int = 12) -> Tuple[int, int, str, float, float]:
    best = (0, 0, "", 0.0, float("inf"))
    for p in primes:
        for d in range(1, max_d + 1):
            for mode, denom in (("p/d", d), ("p/(2d)", 2 * d)):
                cand = p / denom
                rel = abs(period - cand) / period if period > 0 else float("inf")
                if rel < best[4]:
                    best = (p, d, mode, cand, rel)
    return best


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_max = args.t_max
    if t_max <= 0:
        raise ValueError("--t-max required")
    x = read_twins(Path(args.wheel_csv), t_max)
    T = len(x)
    x_mean = float(x.mean())
    y = x.astype(float) - x_mean

    # segmentation
    seg_rates, seg_peaks = segment_fft(x.astype(float), args.seg_len, 5)
    with (out_dir / "segment_rates.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_idx", "rate"])
        for i, rate in enumerate(seg_rates):
            w.writerow([i, rate])
    save_line(out_dir / "segment_rates.png", list(range(len(seg_rates))), seg_rates, "Twin rate per segment", "segment", "rate")
    with (out_dir / "segment_fft_top.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_idx", "rank", "f_idx", "power"])
        for i, peaks in enumerate(seg_peaks):
            for r, (fidx, powv) in enumerate(peaks, start=1):
                w.writerow([i, r, fidx, powv])
    # peak stability scatter (top peak)
    top_periods = []
    for i, peaks in enumerate(seg_peaks):
        if peaks:
            fidx = peaks[0][0]
            top_periods.append((i, args.seg_len / fidx if fidx > 0 else 0))
    if top_periods:
        save_line(
            out_dir / "segment_peak_stability.png",
            [p[0] for p in top_periods],
            [p[1] for p in top_periods],
            "Top peak period per segment",
            "segment",
            "period",
        )

    # detrend (rolling mean)
    p_hat = rolling_mean(x.astype(float), args.smooth_len)
    var = np.maximum(p_hat * (1 - p_hat), 1e-6)
    z = (x - p_hat) / np.sqrt(var)

    # raw vs detrended FFT
    power_raw = fft_power(y)
    power_raw[0] = 0.0
    power_det = fft_power(z - z.mean())
    power_det[0] = 0.0
    save_line(out_dir / "detrend_fft_power.png", list(range(1, len(power_raw))), [float(v) for v in power_raw[1:]],
              "FFT power (raw)", "f_idx", "power", logy=True)
    save_line(out_dir / "detrend_fft_power_detrended.png", list(range(1, len(power_det))), [float(v) for v in power_det[1:]],
              "FFT power (detrended)", "f_idx", "power", logy=True)
    save_spectrum_csv(out_dir / "detrend_fft_power_raw.csv.gz", power_raw)
    save_spectrum_csv(out_dir / "detrend_fft_power_detrended.csv.gz", power_det)

    # raw vs detrended autocorr
    max_lag = min(5000, T - 1)
    ac_raw = autocorr_fft(y, max_lag)
    ac_det = autocorr_fft(z - z.mean(), max_lag)
    save_line(out_dir / "detrend_autocorr.png", list(range(1, max_lag + 1)),
              [float(v) for v in ac_raw[1:]], "Autocorr (raw)", "lag", "corr")
    save_line(out_dir / "detrend_autocorr_detrended.png", list(range(1, max_lag + 1)),
              [float(v) for v in ac_det[1:]], "Autocorr (detrended)", "lag", "corr")

    if args.p0 > 0:
        # conditioning on p0
        inv = pow(args.B % args.p0, -1, args.p0)
        cond_idx = conditioning_indices(T, args.p0, inv)
        cond_x = x[np.array(cond_idx) - 1]
        cond_y = cond_x.astype(float) - cond_x.mean()
        cond_power = fft_power(cond_y)
        cond_power[0] = 0.0
        save_line(out_dir / "cond_p0_fft_power.png", list(range(1, len(cond_power))),
                  [float(v) for v in cond_power[1:]], f"Conditional FFT (p0={args.p0})", "f_idx", "power", logy=True)
        save_spectrum_csv(out_dir / "cond_p0_fft_power.csv.gz", cond_power)
        cond_ac = autocorr_fft(cond_y, max_lag)
        save_line(out_dir / "cond_p0_autocorr.png", list(range(1, max_lag + 1)),
                  [float(v) for v in cond_ac[1:]], f"Conditional autocorr (p0={args.p0})", "lag", "corr")

        # conditional peaks
        idx = np.argsort(cond_power)[-20:][::-1]
        with (out_dir / "cond_p0_top_peaks.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "f_idx", "period_estimate", "power"])
            for r, fidx in enumerate(idx, start=1):
                w.writerow([r, int(fidx), float(len(cond_y) / fidx) if fidx > 0 else 0.0, float(cond_power[fidx])])
        primes = next_primes(args.p0, count=5)
        with (out_dir / "cond_top_peaks_fit.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["rank", "period_estimate", "p", "d", "mode", "fit_period", "rel_error"])
            for r, fidx in enumerate(idx, start=1):
                period = float(len(cond_y) / fidx) if fidx > 0 else 0.0
                p, d, mode, cand, rel = fit_harmonic(period, primes)
                w.writerow([r, period, p, d, mode, cand, rel])

        # conditional mod lift summary
        mod_rows = []
        cond_max = args.cond_max_mod if args.cond_max_mod > 0 else 840
        for m in range(2, cond_max + 1):
            counts = [0] * m
            twins = [0] * m
            for t in cond_idx:
                r = t % m
                counts[r] += 1
                twins[r] += int(x[t - 1])
            pbar = sum(twins) / sum(counts) if sum(counts) else 0.0
            lift = []
            chi2 = 0.0
            for r in range(m):
                pr = twins[r] / counts[r] if counts[r] else 0.0
                lift.append(pr / pbar if pbar > 0 else 0.0)
                exp = counts[r] * pbar
                if exp > 0:
                    chi2 += (twins[r] - exp) ** 2 / exp
            mod_rows.append([m, max(lift), min(lift), chi2, pbar])
            if m in (420, 840):
                save_line(out_dir / f"cond_p0_mod_lift_m{m}.png", list(range(m)), lift, f"Conditional lift mod {m}", "r", "lift")
        mod_rows_sorted = sorted(mod_rows, key=lambda r: r[3], reverse=True)[:20]
        with (out_dir / "cond_p0_mod_chi2_top.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["m", "max_lift", "min_lift", "chi2", "pbar"])
            w.writerows(mod_rows_sorted)

    # permutation control on detrended (segment-wise shuffle)
    rng = random.Random(12345)
    peak_obs = np.sort(power_det)[-5:][::-1]
    perm_peaks = [[] for _ in range(5)]
    perm_iters = args.perm
    if T >= 1_000_000:
        perm_iters = min(20, perm_iters)
    for _ in range(perm_iters):
        zs = z.copy()
        rng.shuffle(zs)
        p = fft_power(zs - zs.mean())
        p[0] = 0.0
        top = np.sort(p)[-5:][::-1]
        for i in range(5):
            perm_peaks[i].append(float(top[i]))
    perm_summary = []
    for i in range(5):
        vals = sorted(perm_peaks[i]) if perm_peaks[i] else [0.0]
        q95 = vals[int(0.95 * len(vals))]
        p_like = sum(1 for v in vals if v >= peak_obs[i]) / len(vals)
        perm_summary.append({
            "rank": i + 1,
            "observed": float(peak_obs[i]),
            "perm_q95": float(q95),
            "p_like": float(p_like),
        })
    Path(out_dir / "m7_perm_control.json").write_text(json.dumps(perm_summary, indent=2), encoding="utf-8")

    if args.p0 > 0:
        Path(out_dir / "p0_forbidden_classes.json").write_text(json.dumps({
            "B": args.B,
            "p0": args.p0,
            "inv": inv,
            "forbidden": [inv % args.p0, (-inv) % args.p0],
        }, indent=2), encoding="utf-8")

    Path(out_dir / "detrend_summary.json").write_text(json.dumps({
        "smooth_len": args.smooth_len,
        "mean_x": x_mean,
        "mean_p_hat": float(p_hat.mean()),
        "var_z": float(np.var(z)),
    }, indent=2), encoding="utf-8")

    # sequential conditioning (M8)
    if args.cond_primes:
        primes_all = [int(p.strip()) for p in args.cond_primes.split(",") if p.strip()]
        if not primes_all:
            raise ValueError("--cond-primes provided but empty")
        max_layer = args.max_layer if args.max_layer > 0 else len(primes_all)
        layer_powers = []
        energy = []
        top_peaks_rows = []
        for layer in range(1, max_layer + 1):
            primes_layer = primes_all[:layer]
            power, nlen = conditional_spectrum(z, args.B, primes_layer)
            if power.size == 0:
                continue
            label = f"layer{layer}"
            save_spectrum_csv(out_dir / f"m8_cond_power_{label}.csv.gz", power)
            layer_powers.append((label, [float(v) for v in power[1:]]))
            total_power = float(np.sum(power[1:]))
            top_peak = float(np.max(power[1:])) if len(power) > 1 else 0.0
            energy.append({"layer": layer, "total_power": total_power, "top_peak": top_peak, "nlen": nlen})
            idx = np.argsort(power)[-10:][::-1]
            candidates = next_primes(primes_layer[-1], count=5)
            for r, fidx in enumerate(idx, start=1):
                period = float(nlen / fidx) if fidx > 0 else 0.0
                p, d, mode, cand, rel = fit_harmonic(period, candidates)
                if mode == "p/d":
                    fit_str = f"{p}/{d}"
                else:
                    fit_str = f"{p}/(2*{d})"
                top_peaks_rows.append([layer, r, period, float(power[fidx]), fit_str, rel])

        if layer_powers:
            save_multi_line(out_dir / "m8_fft_layers.png", layer_powers, "FFT power by conditioning layer", "f_idx", "power", logy=True)
        with (out_dir / "m8_top_peaks_by_layer.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer", "rank", "period", "power", "harmonic_fit", "rel_error"])
            w.writerows(top_peaks_rows)
        Path(out_dir / "m8_energy_by_layer.json").write_text(json.dumps(energy, indent=2), encoding="utf-8")
        if energy:
            save_line(out_dir / "m8_energy_by_layer.png",
                      [e["layer"] for e in energy],
                      [e["top_peak"] for e in energy],
                      "Top-peak power by layer",
                      "layer",
                      "power",
                      logy=True)

    print(f"OK: wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
