#!/usr/bin/env python
# code/scripts/m11_multiwheel_survey.py
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from pathlib import Path
from typing import List, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m-min", type=int, default=8)
    p.add_argument("--m-max", type=int, default=14)
    p.add_argument("--t-max", type=int, default=200000)
    p.add_argument("--p-max", type=int, default=200000)
    p.add_argument("--perm", type=int, default=30)
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m11")
    return p.parse_args()


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


def run_cmd(cmd: List[str]) -> None:
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "m11_summary.csv"

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["m", "B", "p0", "p1", "p2", "E1", "E2", "E3", "top_period_L1", "top_period_L2", "top_period_L3"])

        for m in range(args.m_min, args.m_max + 1):
            B = lcm_upto(m)
            p0, p1, p2 = next_primes(m, 3)

            wheel_csv = Path(f"out/wheel_scan_m{m}_t{args.t_max}.csv")
            wheel_json = Path(f"out/wheel_scan_m{m}_t{args.t_max}.json")
            run_cmd([
                "python", "code/scripts/wheel_scan.py",
                "--m", str(m),
                "--t-max", str(args.t_max),
                "--p-max", str(args.p_max),
                "--out-csv", str(wheel_csv),
                "--out-json", str(wheel_json),
            ])

            m8_dir = out_dir / f"m{m}"
            m8_dir.mkdir(parents=True, exist_ok=True)
            run_cmd([
                "python", "code/scripts/twin_t_axis_detrend.py",
                "--wheel-csv", str(wheel_csv),
                "--B", str(B),
                "--t-max", str(args.t_max),
                "--smooth-len", "5000",
                "--seg-len", "20000",
                "--cond-primes", f"{p0},{p1},{p2}",
                "--max-layer", "3",
                "--perm", str(args.perm),
                "--out-dir", str(m8_dir),
            ])

            with (m8_dir / "m8_energy_by_layer.json").open("r", encoding="utf-8") as jf:
                energy = {int(d["layer"]): d for d in json.load(jf)}
            with (m8_dir / "m8_top_peaks_by_layer.csv").open("r", encoding="utf-8", newline="") as pf:
                r = csv.DictReader(pf)
                top_periods = {}
                for row in r:
                    layer = int(row["layer"])
                    if layer not in top_periods:
                        top_periods[layer] = float(row["period"])

            w.writerow([
                m, B, p0, p1, p2,
                energy.get(1, {}).get("top_peak", 0.0),
                energy.get(2, {}).get("top_peak", 0.0),
                energy.get(3, {}).get("top_peak", 0.0),
                top_periods.get(1, 0.0),
                top_periods.get(2, 0.0),
                top_periods.get(3, 0.0),
            ])

    print(f"OK: wrote {summary_path}")


if __name__ == "__main__":
    main()
