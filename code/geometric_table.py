# code/geometric_table.py
from __future__ import annotations

import csv
import json
import math
import os
import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Literal, Any, Iterable

import numpy as np


WeightMode = Literal["ones", "atan", "log", "idf"]
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


def divisors_up_to_K(n: int, K: int) -> Iterable[int]:
    """
    Enumerate divisors k of n with k <= K efficiently by factoring n.
    """
    factors: List[Tuple[int, int]] = []
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            exp = 0
            while x % d == 0:
                x //= d
                exp += 1
            factors.append((d, exp))
        d = 3 if d == 2 else d + 2
    if x > 1:
        factors.append((x, 1))

    divisors = [1]
    for p, e in factors:
        new_divs = []
        pow_p = 1
        for _ in range(e):
            pow_p *= p
            new_divs.extend([d * pow_p for d in divisors])
        divisors.extend(new_divs)

    for k in divisors:
        if k <= K:
            yield k


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

    # For each row n, enumerate divisors k<=K and produce q=n//k
    for i, n in enumerate(rows):
        for k in divisors_up_to_K(n, params.K):
            q = n // k

            if params.primitive and math.gcd(k, q) != 1:
                continue

            q_to_row_indices.setdefault(q, []).append(i)
            if q not in q_weight and params.weight != "idf":
                q_weight[q] = weight_of_q(q, params.weight)

    if params.weight == "idf":
        m = len(rows)
        for q, idx in q_to_row_indices.items():
            freq = max(1, len(idx))
            q_weight[q] = float(math.log(1.0 + (m / float(freq))))

    return rows, q_to_row_indices, q_weight


def build_row_to_qs_for_rows(
    rows: List[int],
    K: int,
    primitive: bool,
    weight: WeightMode,
) -> Tuple[Dict[int, List[int]], Dict[int, float]]:
    """
    Build q->row_indices and q->weight for a given list of rows.
    """
    q_to_row_indices: Dict[int, List[int]] = {}
    q_weight: Dict[int, float] = {}
    for i, n in enumerate(rows):
        for k in divisors_up_to_K(n, K):
            q = n // k
            if primitive and math.gcd(k, q) != 1:
                continue
            q_to_row_indices.setdefault(q, []).append(i)
            if q not in q_weight and weight != "idf":
                q_weight[q] = weight_of_q(q, weight)

    if weight == "idf":
        m = len(rows)
        for q, idx in q_to_row_indices.items():
            freq = max(1, len(idx))
            q_weight[q] = float(math.log(1.0 + (m / float(freq))))

    return q_to_row_indices, q_weight


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


def connected_components(A: np.ndarray, eps: float) -> List[List[int]]:
    """
    Return connected components in an undirected weighted adjacency matrix.
    Edge exists if A[i,j] > eps.
    """
    n = A.shape[0]
    visited = np.zeros(n, dtype=bool)

    neighbors: List[np.ndarray] = [np.flatnonzero(A[i] > eps) for i in range(n)]

    comps: List[List[int]] = []
    for start in range(n):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        comp = [start]
        while stack:
            v = stack.pop()
            for u in neighbors[v]:
                if not visited[u]:
                    visited[u] = True
                    stack.append(int(u))
                    comp.append(int(u))
        comps.append(comp)
    return comps


def largest_component_indices(A: np.ndarray, eps: float) -> List[int]:
    comps = connected_components(A, eps=eps)
    if not comps:
        return []
    return max(comps, key=len)


def count_edges(A: np.ndarray, eps: float) -> int:
    return int(np.sum(np.triu(A, 1) > eps))


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


def spectral_summary_with_head_tail(
    evals: np.ndarray, eps: float, head: int, tail: int
) -> Tuple[Dict[str, float], List[float], List[float]]:
    metrics = spectral_metrics(evals, eps=eps)
    e = np.array(evals, dtype=float)
    e[e < 0] = 0.0
    e[e > 2] = 2.0
    e.sort()
    head_vals = [float(x) for x in e[: min(head, e.size)].tolist()]
    tail_vals = [float(x) for x in e[-min(tail, e.size) :].tolist()]
    return metrics, head_vals, tail_vals


def compute_rowproj_metrics(
    params: BuildParams, neigs: int = 50, core_r: int = 30
) -> Tuple[
    List[int],
    np.ndarray,
    np.ndarray,
    Dict[str, Any],
    List[float],
    List[float],
    List[float],
    List[float],
]:
    rows, q_to_row_indices, q_weight = build_row_to_qs(params)
    A = build_row_projection_adjacency(rows, q_to_row_indices, q_weight)

    m = len(rows)
    degrees = A.sum(axis=1)
    isolated_nodes = int(np.sum(degrees <= params.eps))

    comps = connected_components(A, eps=params.eps)
    n_components = len(comps)
    gc_idx = max(comps, key=len) if comps else []

    Lnorm = normalized_laplacian(A, eps=params.eps)
    evals_all = np.linalg.eigvalsh(Lnorm)
    evals_all = np.clip(evals_all, 0.0, 2.0)
    evals_all.sort()

    metrics = {
        "n_nodes": m,
        "n_edges": count_edges(A, eps=params.eps),
        "n_components": int(n_components),
        "isolated_nodes": isolated_nodes,
        "isolated_fraction": float(isolated_nodes / m) if m else 0.0,
        "degree_min": float(np.min(degrees)) if m else 0.0,
        "degree_mean": float(np.mean(degrees)) if m else 0.0,
        "degree_max": float(np.max(degrees)) if m else 0.0,
    }

    base_metrics, all_head, all_tail = spectral_summary_with_head_tail(
        evals_all, eps=params.eps, head=neigs, tail=10
    )
    metrics.update(base_metrics)

    # Giant component metrics
    if gc_idx:
        A_gc = A[np.ix_(gc_idx, gc_idx)]
        Lnorm_gc = normalized_laplacian(A_gc, eps=params.eps)
        evals_gc = np.linalg.eigvalsh(Lnorm_gc)
        evals_gc = np.clip(evals_gc, 0.0, 2.0)
        evals_gc.sort()
        gc_metrics, gc_head, gc_tail = spectral_summary_with_head_tail(
            evals_gc, eps=params.eps, head=20, tail=10
        )
        gc_size = len(gc_idx)
        metrics.update({
            "gc_size": int(gc_size),
            "gc_fraction": float(gc_size / m) if m else 0.0,
            "gc_edges": count_edges(A_gc, eps=params.eps),
            "gc_spectral_gap": float(gc_metrics["spectral_gap"]),
            "gc_entropy": float(gc_metrics["spectral_entropy"]),
            "gc_zero_count_all": float(gc_metrics["zero_count_all"]),
            "gc_eigenvalues_head": gc_head,
            "gc_eigenvalues_tail": gc_tail,
        })
    else:
        metrics.update({
            "gc_size": 0,
            "gc_fraction": 0.0,
            "gc_edges": 0,
            "gc_spectral_gap": 0.0,
            "gc_entropy": 0.0,
            "gc_zero_count_all": 0.0,
            "gc_eigenvalues_head": [],
            "gc_eigenvalues_tail": [],
        })

    evals_head = [float(x) for x in evals_all[: min(neigs, m)].tolist()]
    evals_tail = [float(x) for x in evals_all[-min(10, m) :].tolist()]

    # Core window metrics
    core_lo = params.center - core_r
    core_hi = params.center + core_r
    core_idx = [i for i, n in enumerate(rows) if core_lo <= n <= core_hi]
    if core_idx:
        A_core = A[np.ix_(core_idx, core_idx)]
        core_nodes = len(core_idx)
        core_comps = connected_components(A_core, eps=params.eps)
        core_n_components = len(core_comps)
        core_gc_idx = max(core_comps, key=len) if core_comps else []
        core_degrees = A_core.sum(axis=1)
        core_isolated = int(np.sum(core_degrees <= params.eps))

        metrics.update({
            "core_nodes": int(core_nodes),
            "core_edges": count_edges(A_core, eps=params.eps),
            "core_components": int(core_n_components),
            "core_isolated_nodes": core_isolated,
        })

        if core_gc_idx:
            A_core_gc = A_core[np.ix_(core_gc_idx, core_gc_idx)]
            L_core_gc = normalized_laplacian(A_core_gc, eps=params.eps)
            evals_core_gc = np.linalg.eigvalsh(L_core_gc)
            evals_core_gc = np.clip(evals_core_gc, 0.0, 2.0)
            evals_core_gc.sort()
            core_metrics, core_head, core_tail = spectral_summary_with_head_tail(
                evals_core_gc, eps=params.eps, head=20, tail=10
            )
            core_gc_size = len(core_gc_idx)
            metrics.update({
                "core_gc_size": int(core_gc_size),
                "core_gc_fraction": float(core_gc_size / core_nodes) if core_nodes else 0.0,
                "core_gc_spectral_gap": float(core_metrics["spectral_gap"]),
                "core_gc_entropy": float(core_metrics["spectral_entropy"]),
                "core_gc_zero_count_all": float(core_metrics["zero_count_all"]),
                "core_gc_eigenvalues_head": core_head,
                "core_gc_eigenvalues_tail": core_tail,
            })
        else:
            metrics.update({
                "core_gc_size": 0,
                "core_gc_fraction": 0.0,
                "core_gc_spectral_gap": 0.0,
                "core_gc_entropy": 0.0,
                "core_gc_zero_count_all": 0.0,
                "core_gc_eigenvalues_head": [],
                "core_gc_eigenvalues_tail": [],
            })
    else:
        metrics.update({
            "core_nodes": 0,
            "core_edges": 0,
            "core_components": 0,
            "core_isolated_nodes": 0,
            "core_gc_size": 0,
            "core_gc_fraction": 0.0,
            "core_gc_spectral_gap": 0.0,
            "core_gc_entropy": 0.0,
            "core_gc_zero_count_all": 0.0,
            "core_gc_eigenvalues_head": [],
            "core_gc_eigenvalues_tail": [],
        })

    return rows, A, evals_all, metrics, evals_head, evals_tail, all_head, all_tail


def compute_core_metrics_fast(
    center: int,
    core_r: int,
    K: int,
    primitive: bool,
    weight: WeightMode,
    eps: float,
) -> Dict[str, Any]:
    rows = list(range(max(1, center - core_r), center + core_r + 1))
    q_to_row_indices, q_weight = build_row_to_qs_for_rows(rows, K, primitive, weight)
    A = build_row_projection_adjacency(rows, q_to_row_indices, q_weight)
    core_nodes = len(rows)
    core_degrees = A.sum(axis=1)
    core_isolated = int(np.sum(core_degrees <= eps))
    core_edges = count_edges(A, eps=eps)
    core_comps = connected_components(A, eps=eps)
    core_n_components = len(core_comps)
    core_gc_idx = max(core_comps, key=len) if core_comps else []
    # twin-row indices within core window
    idx_minus = center - 1 - rows[0] if rows and (center - 1) >= rows[0] and (center - 1) <= rows[-1] else None
    idx_plus = center + 1 - rows[0] if rows and (center + 1) >= rows[0] and (center + 1) <= rows[-1] else None
    def twin_local(idx: int | None) -> Dict[str, Any]:
        if idx is None:
            return {"deg": 0.0, "is_isolated": 1, "in_gc": 0, "edges_incident": 0.0}
        deg = float(core_degrees[idx])
        is_iso = int(deg <= eps)
        in_gc = int(core_gc_idx and idx in core_gc_idx)
        edges_inc = float(np.sum(A[idx] > eps))
        return {"deg": deg, "is_isolated": is_iso, "in_gc": in_gc, "edges_incident": edges_inc}
    twin_minus = twin_local(idx_minus)
    twin_plus = twin_local(idx_plus)

    metrics = {
        "core_nodes": int(core_nodes),
        "core_edges": int(core_edges),
        "core_components": int(core_n_components),
        "core_isolated_nodes": core_isolated,
        "twin_deg_minus1": twin_minus["deg"],
        "twin_deg_plus1": twin_plus["deg"],
        "twin_is_isolated_minus1": twin_minus["is_isolated"],
        "twin_is_isolated_plus1": twin_plus["is_isolated"],
        "twin_in_gc_minus1": twin_minus["in_gc"],
        "twin_in_gc_plus1": twin_plus["in_gc"],
        "twin_edges_inc_minus1": twin_minus["edges_incident"],
        "twin_edges_inc_plus1": twin_plus["edges_incident"],
        "twin_isolates": float(twin_minus["is_isolated"] + twin_plus["is_isolated"]),
        "twin_deg_sum": float(twin_minus["deg"] + twin_plus["deg"]),
    }

    if core_gc_idx:
        A_core_gc = A[np.ix_(core_gc_idx, core_gc_idx)]
        L_core_gc = normalized_laplacian(A_core_gc, eps=eps)
        evals_core_gc = np.linalg.eigvalsh(L_core_gc)
        evals_core_gc = np.clip(evals_core_gc, 0.0, 2.0)
        evals_core_gc.sort()
        core_metrics, core_head, core_tail = spectral_summary_with_head_tail(
            evals_core_gc, eps=eps, head=20, tail=10
        )
        core_gc_size = len(core_gc_idx)
        metrics.update({
            "core_gc_size": int(core_gc_size),
            "core_gc_fraction": float(core_gc_size / core_nodes) if core_nodes else 0.0,
            "core_gc_spectral_gap": float(core_metrics["spectral_gap"]),
            "core_gc_entropy": float(core_metrics["spectral_entropy"]),
            "core_gc_zero_count_all": float(core_metrics["zero_count_all"]),
            "core_gc_eigenvalues_head": core_head,
            "core_gc_eigenvalues_tail": core_tail,
        })
    else:
        metrics.update({
            "core_gc_size": 0,
            "core_gc_fraction": 0.0,
            "core_gc_spectral_gap": 0.0,
            "core_gc_entropy": 0.0,
            "core_gc_zero_count_all": 0.0,
            "core_gc_eigenvalues_head": [],
            "core_gc_eigenvalues_tail": [],
        })

    return metrics


def compute_core_gc_size_fast(
    center: int,
    core_r: int,
    K: int,
    primitive: bool,
    weight: WeightMode,
    eps: float,
) -> int:
    rows = list(range(max(1, center - core_r), center + core_r + 1))
    q_to_row_indices, q_weight = build_row_to_qs_for_rows(rows, K, primitive, weight)
    A = build_row_projection_adjacency(rows, q_to_row_indices, q_weight)
    comps = connected_components(A, eps=eps)
    if not comps:
        return 0
    return len(max(comps, key=len))


def compute_core_edges_only(
    center: int,
    core_r: int,
    K: int,
    primitive: bool,
    weight: WeightMode,
    eps: float,
) -> int:
    rows = list(range(max(1, center - core_r), center + core_r + 1))
    q_to_row_indices, q_weight = build_row_to_qs_for_rows(rows, K, primitive, weight)
    A = build_row_projection_adjacency(rows, q_to_row_indices, q_weight)
    return count_edges(A, eps=eps)


def choose_hybrid_K(
    center: int,
    core_r: int,
    primitive: bool,
    weight: WeightMode,
    alpha: float = 1.0,
    growth: float = 1.5,
    min_core_gc: int = 10,
    max_bumps: int = 6,
    K_max: int = 5000,
    eps: float = 1e-12,
) -> Tuple[int, int, int, bool, int]:
    """
    Hybrid K policy: start from scale-law K0 ~ alpha*sqrt(center), then
    multiplicatively increase until the core GC reaches min_core_gc or limits.
    Returns (K0, K_eff, bumps, hit_kmax, final_gc_size).
    """
    K0 = int(math.ceil(alpha * math.sqrt(max(1, center))))
    K_eff = max(1, min(K_max, K0))
    bumps = 0
    gc_size = compute_core_gc_size_fast(
        center=center,
        core_r=core_r,
        K=K_eff,
        primitive=primitive,
        weight=weight,
        eps=eps,
    )
    while gc_size < min_core_gc and bumps < max_bumps and K_eff < K_max:
        K_eff = min(K_max, int(math.ceil(growth * K_eff)))
        bumps += 1
        gc_size = compute_core_gc_size_fast(
            center=center,
            core_r=core_r,
            K=K_eff,
            primitive=primitive,
            weight=weight,
            eps=eps,
        )
    hit_kmax = K_eff >= K_max and gc_size < min_core_gc
    return K0, K_eff, bumps, hit_kmax, gc_size


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


def run_rowproj_experiment(
    params: BuildParams, out_dir: str, neigs: int = 50, core_r: int = 30
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    rows, A, evals_all, metrics, evals_head, evals_tail, _, _ = compute_rowproj_metrics(
        params, neigs=neigs, core_r=core_r
    )
    m = len(rows)

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
        "core_r": core_r,
    })

    write_json(os.path.join(out_dir, "nodes.json"), {
        "nodes": [{"id": int(n), "type": "row"} for n in rows]
    })

    n_edges_written = write_edges_csv(os.path.join(out_dir, "edges.csv"), rows, A, eps=params.eps)

    write_json(os.path.join(out_dir, "eigenvalues.json"), {
        "eigenvalues_all": [float(x) for x in evals_all.tolist()],
        "eigenvalues_head": evals_head,
        "eigenvalues_tail": evals_tail,
    })

    write_json(os.path.join(out_dir, "metrics.json"), {
        **metrics,
        "edges_written": int(n_edges_written),
    })

    write_checksums(out_dir, ["params.json", "nodes.json", "edges.csv", "eigenvalues.json", "metrics.json"])
