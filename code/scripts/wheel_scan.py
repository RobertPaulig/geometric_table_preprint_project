#!/usr/bin/env python
# code/scripts/wheel_scan.py
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import List, Tuple


def lcm_upto(m: int) -> int:
    v = 1
    for k in range(1, m + 1):
        v = math.lcm(v, k)
    return v


def sieve_primes(n: int) -> List[int]:
    if n < 2:
        return []
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for p in range(2, int(n ** 0.5) + 1):
        if is_prime[p]:
            step = p
            start = p * p
            is_prime[start:n + 1:step] = [False] * (((n - start) // step) + 1)
    return [i for i, v in enumerate(is_prime) if v]


def is_probable_prime(n: int) -> bool:
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n % p == 0:
            return n == p

    # deterministic Miller-Rabin for 64-bit
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1

    def check(a: int) -> bool:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            return True
        for _ in range(s - 1):
            x = (x * x) % n
            if x == n - 1:
                return True
        return False

    for a in (2, 3, 5, 7, 11, 13, 17):
        if a % n == 0:
            continue
        if not check(a):
            return False
    return True


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=12)
    p.add_argument("--t-max", type=int, default=200000)
    p.add_argument("--p-max", type=int, default=200000)
    p.add_argument("--out-csv", type=str, default="")
    p.add_argument("--out-json", type=str, default="")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    m = args.m
    t_max = args.t_max
    p_max = args.p_max
    B = lcm_upto(m)

    out_csv = args.out_csv or f"out/wheel_scan_m{m}_t{t_max}.csv"
    out_json = args.out_json or f"out/wheel_scan_m{m}_t{t_max}.json"

    primes = sieve_primes(p_max)
    spf_minus = [0] * (t_max + 1)
    spf_plus = [0] * (t_max + 1)

    for p in primes:
        if B % p == 0:
            continue
        inv = pow(B % p, -1, p)
        # t ≡ inv (mod p) => p | (tB - 1)
        t = inv
        while t <= t_max:
            if spf_minus[t] == 0:
                spf_minus[t] = p
            t += p
        # t ≡ -inv (mod p) => p | (tB + 1)
        t = (-inv) % p
        if t == 0:
            t = p
        while t <= t_max:
            if spf_plus[t] == 0:
                spf_plus[t] = p
            t += p

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    counts = {
        "t_max": int(t_max),
        "p_max": int(p_max),
        "m": int(m),
        "B": int(B),
        "minus_prime": 0,
        "plus_prime": 0,
        "twins": 0,
        "minus_semiprime": 0,
        "plus_semiprime": 0,
        "both_semiprime": 0,
        "one_prime_one_semiprime": 0,
        "no_small_factor_both": 0,
    }

    spf_minus_counter = Counter()
    spf_plus_counter = Counter()

    for t in range(1, t_max + 1):
        N = B * t
        n_minus = N - 1
        n_plus = N + 1

        sm = spf_minus[t]
        sp = spf_plus[t]
        minus_is_prime = 0
        plus_is_prime = 0
        minus_is_semiprime = 0
        plus_is_semiprime = 0
        minus_q = 0
        plus_q = 0

        if sm == 0:
            minus_is_prime = int(is_probable_prime(n_minus))
            if not minus_is_prime:
                pass
        else:
            q = n_minus // sm
            if is_probable_prime(q):
                minus_is_semiprime = 1
                minus_q = q
            spf_minus_counter[sm] += 1

        if sp == 0:
            plus_is_prime = int(is_probable_prime(n_plus))
            if not plus_is_prime:
                pass
        else:
            q = n_plus // sp
            if is_probable_prime(q):
                plus_is_semiprime = 1
                plus_q = q
            spf_plus_counter[sp] += 1

        if sm == 0 and sp == 0:
            counts["no_small_factor_both"] += 1

        counts["minus_prime"] += minus_is_prime
        counts["plus_prime"] += plus_is_prime
        counts["minus_semiprime"] += minus_is_semiprime
        counts["plus_semiprime"] += plus_is_semiprime
        if minus_is_prime and plus_is_prime:
            counts["twins"] += 1
        if minus_is_semiprime and plus_is_semiprime:
            counts["both_semiprime"] += 1
        if (minus_is_prime and plus_is_semiprime) or (plus_is_prime and minus_is_semiprime):
            counts["one_prime_one_semiprime"] += 1

        rows.append({
            "m": m,
            "B": B,
            "t": t,
            "N": N,
            "spf_minus": sm,
            "minus_is_prime": minus_is_prime,
            "minus_is_semiprime": minus_is_semiprime,
            "minus_q_if_semiprime": minus_q,
            "spf_plus": sp,
            "plus_is_prime": plus_is_prime,
            "plus_is_semiprime": plus_is_semiprime,
            "plus_q_if_semiprime": plus_q,
            "is_twin": int(minus_is_prime and plus_is_prime),
        })

    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {
        "counts": counts,
        "spf_minus_top20": spf_minus_counter.most_common(20),
        "spf_plus_top20": spf_plus_counter.most_common(20),
        "sanity": {
            "spf_minus_le_m": sum(v for p, v in spf_minus_counter.items() if p <= m),
            "spf_plus_le_m": sum(v for p, v in spf_plus_counter.items() if p <= m),
        },
    }
    Path(out_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"OK: wrote {out_csv}, {out_json}")


if __name__ == "__main__":
    main()
