from __future__ import annotations

import argparse
from geometric_table import (
    WindowParams,
    build_primitive_bipartite_edges,
    build_normalized_laplacian,
    top_eigenvalues,
    write_evidence_pack,
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--center", type=int, required=True)
    ap.add_argument("--h", type=int, default=200)
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--weight", type=str, choices=["ones", "atan", "logq"], default="ones")
    ap.add_argument("--no-primitive", action="store_true")
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    params = WindowParams(
        center=args.center,
        h=args.h,
        K=args.K,
        primitive=not args.no_primitive,
        weight=args.weight,
    )

    U, V, edges = build_primitive_bipartite_edges(params)
    L = build_normalized_laplacian(U, V, edges)
    eigs = top_eigenvalues(L, r=args.r)
    write_evidence_pack(args.out, params, U, V, edges, eigs)

if __name__ == "__main__":
    main()
