from __future__ import annotations

from dataclasses import dataclass
from math import atan
from typing import Dict, List, Tuple, Literal
import math
import json
import csv
import hashlib
import os

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import eigsh

WeightMode = Literal["ones", "atan", "logq"]

def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)

@dataclass(frozen=True)
class WindowParams:
    center: int
    h: int
    K: int
    primitive: bool = True
    weight: WeightMode = "ones"

def weight_value(q: int, mode: WeightMode) -> float:
    if mode == "ones":
        return 1.0
    if mode == "atan":
        return float(atan(q))
    if mode == "logq":
        return float(math.log(q))
    raise ValueError(f"Unknown weight mode: {mode}")

def build_primitive_bipartite_edges(params: WindowParams) -> Tuple[List[int], List[int], List[Tuple[int,int,float]]]:
    """Build edges for bipartite graph U (rows n) and V (quotients q).

    Include edge (n,q) iff exists k<=K with n=kq and (if primitive) gcd(k,q)=1.
    Returns (U_nodes, V_nodes, edges) where edges = (u_index, v_index, weight).
    """
    c, h, K = params.center, params.h, params.K
    n_min = max(1, c - h)
    n_max = c + h
    U_nodes = list(range(n_min, n_max + 1))

    V_set = set()
    raw_edges: List[Tuple[int,int,float]] = []

    for u_idx, n in enumerate(U_nodes):
        for k in range(1, K + 1):
            if n % k != 0:
                continue
            q = n // k
            if params.primitive and gcd(k, q) != 1:
                continue
            V_set.add(q)
            raw_edges.append((u_idx, q, weight_value(q, params.weight)))

    V_nodes = sorted(V_set)
    v_index: Dict[int, int] = {q: i for i, q in enumerate(V_nodes)}
    edges = [(u_idx, v_index[q], w) for (u_idx, q, w) in raw_edges]
    return U_nodes, V_nodes, edges

def build_normalized_laplacian(U_nodes: List[int], V_nodes: List[int], edges: List[Tuple[int,int,float]]) -> csr_matrix:
    """Build normalized Laplacian L_norm = I - D^{-1/2} A D^{-1/2}."""
    nU, nV = len(U_nodes), len(V_nodes)
    total = nU + nV

    rows, cols, data = [], [], []
    for u, v, w in edges:
        uu = u
        vv = nU + v
        rows.extend([uu, vv])
        cols.extend([vv, uu])
        data.extend([w, w])

    A = coo_matrix((data, (rows, cols)), shape=(total, total)).tocsr()
    deg = np.asarray(A.sum(axis=1)).reshape(-1)

    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)

    Dm12 = csr_matrix((inv_sqrt, (np.arange(total), np.arange(total))), shape=(total, total))
    I = csr_matrix(np.eye(total))
    return I - (Dm12 @ A @ Dm12)

def top_eigenvalues(Lnorm: csr_matrix, r: int = 30) -> np.ndarray:
    """Compute smallest r eigenvalues of L_norm (symmetric PSD)."""
    if Lnorm.shape[0] <= 2:
        return np.array([0.0])
    r = min(r, Lnorm.shape[0] - 2)
    vals = eigsh(Lnorm, k=r, which="SM", return_eigenvectors=False)
    return np.sort(np.real(vals))

def spectral_metrics(eigs: np.ndarray) -> dict:
    eigs = np.asarray(eigs, dtype=float)
    gap = float(eigs[1] - eigs[0]) if eigs.size >= 2 else float("nan")
    s = float(np.sum(eigs))
    if s > 0:
        p = eigs / s
        p = p[p > 0]
        entropy = float(-np.sum(p * np.log(p)))
    else:
        entropy = float("nan")
    return {"spectral_gap": gap, "spectral_entropy": entropy}

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_evidence_pack(out_dir: str, params: WindowParams, U_nodes: List[int], V_nodes: List[int], edges: List[Tuple[int,int,float]], eigs: np.ndarray) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "params.json"), "w", encoding="utf-8") as f:
        json.dump(params.__dict__, f, indent=2)

    with open(os.path.join(out_dir, "nodes.json"), "w", encoding="utf-8") as f:
        json.dump({"U": U_nodes, "V": V_nodes}, f, indent=2)

    with open(os.path.join(out_dir, "edges.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["u_index", "v_index", "weight"])
        w.writerows(edges)

    with open(os.path.join(out_dir, "eigenvalues.json"), "w", encoding="utf-8") as f:
        json.dump({"eigenvalues": eigs.tolist()}, f, indent=2)

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(spectral_metrics(eigs), f, indent=2)

    files = ["params.json", "nodes.json", "edges.csv", "eigenvalues.json", "metrics.json"]
    with open(os.path.join(out_dir, "checksums.sha256"), "w", encoding="utf-8") as f:
        for fn in files:
            p = os.path.join(out_dir, fn)
            f.write(f"{sha256_file(p)}  {fn}\n")
