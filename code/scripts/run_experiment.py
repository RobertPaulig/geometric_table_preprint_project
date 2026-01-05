# code/scripts/run_experiment.py
from __future__ import annotations

import argparse
import os

from geometric_table import BuildParams, run_rowproj_experiment


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--center", type=int, required=True)
    p.add_argument("--h", type=int, default=200)
    p.add_argument("--K", type=int, default=200)
    p.add_argument("--primitive", action="store_true", help="use primitive cells gcd(k,q)=1")
    p.add_argument("--weight", choices=["ones", "atan", "log"], default="ones")
    p.add_argument("--graph-mode", choices=["rowproj"], default="rowproj")
    p.add_argument("--neigs", type=int, default=50)
    p.add_argument("--out", type=str, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)

    params = BuildParams(
        center=args.center,
        h=args.h,
        K=args.K,
        primitive=bool(args.primitive),
        weight=args.weight,
        graph_mode=args.graph_mode,
    )

    # For now only rowproj is implemented (v2 focus)
    run_rowproj_experiment(params, out_dir=out_dir, neigs=args.neigs)

    print(f"OK: wrote artifacts to {out_dir}")


if __name__ == "__main__":
    main()
