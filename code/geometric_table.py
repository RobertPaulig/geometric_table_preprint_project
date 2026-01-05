# code/geometric_table.py
from __future__ import annotations

import csv
import json
import math
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Any

import numpy as np


WeightMode = Literal["ones", "atan", "log"]
GraphMode = Literal["rowproj"]


@dataclass(frozen=True)
class BuildParams:
    center: int
    h: int
    K: int
    primitive: bool
    weight: WeightMode
    graph_mode: GraphMode = "rowproj"
    eps: float = 1e-12  # numerical threshold


def weight_of_q(q: int, mode: WeightMode) -> float:
    if mode == "ones":
        return 1.0
    if mode == "atan":
        return float(math.atan(q))
    if mode == "log":
        return float(math.log(q))
    raise ValueError(f"Unknown weight mode: {mode}")


def window_rows(center: int, h: int) -> List[int]:
    lo = max(1, center - h)
    hi = center + h
    return list(range(lo, hi + 1))


def build_row_to_qs(params: BuildParams) -> Tuple[List[int], Dict[int, List[int]], Dict[int, float]]:
    """
    Returns:
      rows: list of row numbers n in the window
      q_to_row_indices: q -> list of row indices i where row n has a primitive factor-pair n = k*q with k<=K
      q_weight: q -> weight(q)
    """
    rows = window_rows(params.center, params.h)
    q_to_row_indices: Dict[int, List[int]] = {}
    q_weight: Dict[int, float] = {}

    # For each row n, enumerate k<=K dividing n and produce q=n//k
    for i, n in enumerate(rows):
        for k in range(1, params.K + 1):
            if n % k != 0:
                continue
            q = n // k

            if params.primitive:
                # Primitive cell criterion for Geometric Table:
                # n = k*q is "primitive" iff gcd(k,q)=1.
                if math.gcd(k, q) != 1:
                    continue

            q_to_row_indices.setdefault(q, []).append(i)
            if q not in q_weight:
                q_weight[q] = weight_of_q(q, params.weight)

    return rows, q_to_row_indices, q_weight


def build_row_projection_adjacency(
    rows: List[int],
    q_to_row_indices: Dict[int, List[int]],
    q_weight: Dict[int, float],
) -> np.ndarray:
    """
    Build weighted row-row adjacency A (dense) where two rows i,j connect
    if they share at least one q. Weight sums over shared q weights.
    """
    m = len(rows)
    A = np.zeros((m, m), dtype=float)

    for q, idx in q_to_row_indices.items():
        if len(idx) <= 1:
            continue
        w = q_weight[q]

        # Add clique weight on rows that share q
        ix = np.ix_(idx, idx)
        A[ix] += w

    # remove self-loops
    np.fill_diagonal(A, 0.0)
    return A


def connected_components_count(A: np.ndarray, eps: float) -> int:
    """
    Count connected components in an undirected weighted adjacency matrix.
    Edge exists if A[i,j] > eps.
    """
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)

    neighbors: List[np.ndarray] = [np.flatnonzero(A[i] > eps) for i in range(n)]

    comps = 0
    for start in range(n):
        if visited[start]:
            continue
        comps += 1
        stack = [start]
        visited[start] = True
        while stack:
            v = stack.pop()
            for u in neighbors[v]:
                if not visited[u]:
                    visited[u] = True
                    stack.append(int(u))
    return comps


def normalized_laplacian(A: np.ndarray, eps: float) -> np.ndarray:
    """
    L_norm = I - D^{-1/2} A D^{-1/2}, with convention:
    if degree is 0, row/col stays 0 (diagonal set to 0).
    """
    m = A.shape[0]
    d = A.sum(axis=1)
    inv_sqrt = np.zeros_like(d)
    mask = d > eps
    inv_sqrt[mask] = 1.0 / np.sqrt(d[mask])

    # core: I - inv_sqrt * A * inv_sqrt
    L = np.eye(m, dtype=float) - (inv_sqrt[:, None] * A * inv_sqrt[None, :])

    # isolated nodes: set diagonal to 0 instead of 1
    iso = ~mask
    L[iso, iso] = 0.0

    # numerical clip
    L[np.abs(L) < eps] = 0.0
    return L


def spectral_metrics(evals: np.ndarray, eps: float) -> Dict[str, float]:
    """
    evals should be eigenvalues of L_norm (ideally in [0,2]).
    Returns spectral gap (first positive after zeros) and entropy over positive evals.
    """
    e = np.array(evals, dtype=float)
    # Clip for numerical stability
    e[e < 0] = 0.0
    e[e > 2] = 2.0
    e.sort()

    zero_count = int(np.sum(e <= 1e-10))
    positive = e[e > 1e-10]

    if positive.size == 0:
        return {
            "zero_count_all": float(zero_count),
            "spectral_gap": 0.0,
            "spectral_entropy": 0.0,
        }

    spectral_gap = float(positive[0])

    # entropy on normalized positive spectrum
    s = float(np.sum(positive))
    p = positive / s
    # avoid log(0)
    p = p[p > 0]
    H = float(-np.sum(p * np.log(p)))

    return {
        "zero_count_all": float(zero_count),
        "spectral_gap": spectral_gap,
        "spectral_entropy": H,
    }


def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_edges_csv(path: str, rows: List[int], A: np.ndarray, eps: float) -> int:
    """
    Write undirected edges i<j with weight A[i,j].
    Returns number of edges.
    """
    m = len(rows)
    cnt = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["u", "v", "weight"])
        for i in range(m):
            for j in range(i + 1, m):
                val = A[i, j]
                if val > eps:
                    w.writerow([rows[i], rows[j], float(val)])
                    cnt += 1
    return cnt


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksums(out_dir: str, filenames: List[str]) -> None:
    lines = []
    for name in filenames:
        fp = os.path.join(out_dir, name)
        digest = sha256_file(fp)
        lines.append(f"{digest}  {name}")
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def run_rowproj_experiment(params: BuildParams, out_dir: str, neigs: int = 50) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rows, q_to_row_indices, q_weight = build_row_to_qs(params)
    A = build_row_projection_adjacency(rows, q_to_row_indices, q_weight)

    m = len(rows)
    n_components = connected_components_count(A, eps=params.eps)

    Lnorm = normalized_laplacian(A, eps=params.eps)
    evals_all = np.linalg.eigvalsh(Lnorm)
    evals_all = np.clip(evals_all, 0.0, 2.0)
    evals_all.sort()

    metrics = {}
    metrics.update({
        "n_nodes": m,
        "n_edges": int(np.sum(np.triu(A, 1) > params.eps)),
        "n_components": int(n_components),
        "degree_min": float(np.min(A.sum(axis=1))),
        "degree_mean": float(np.mean(A.sum(axis=1))),
        "degree_max": float(np.max(A.sum(axis=1))),
    })
    metrics.update(spectral_metrics(evals_all, eps=params.eps))

    # Persist artifacts
    write_json(os.path.join(out_dir, "params.json"), {
        "center": params.center,
        "h": params.h,
        "K": params.K,
        "primitive": params.primitive,
        "weight": params.weight,
        "graph_mode": params.graph_mode,
        "eps": params.eps,
        "neigs_saved": min(neigs, m),
    })

    write_json(os.path.join(out_dir, "nodes.json"), {
        "nodes": [{"id": int(n), "type": "row"} for n in rows]
    })

    n_edges_written = write_edges_csv(os.path.join(out_dir, "edges.csv"), rows, A, eps=params.eps)

    write_json(os.path.join(out_dir, "eigenvalues.json"), {
        "eigenvalues_all": [float(x) for x in evals_all.tolist()],
        "eigenvalues_head": [float(x) for x in evals_all[:min(neigs, m)].tolist()],
    })

    write_json(os.path.join(out_dir, "metrics.json"), {
        **metrics,
        "edges_written": int(n_edges_written),
    })

    write_checksums(out_dir, ["params.json", "nodes.json", "edges.csv", "eigenvalues.json", "metrics.json"])
