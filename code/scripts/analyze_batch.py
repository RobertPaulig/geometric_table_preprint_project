# code/scripts/analyze_batch.py
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def read_rows(path: Path) -> List[Dict[str, float]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            parsed = {}
            for k, v in row.items():
                if k == "center_set":
                    parsed[k] = v
                    continue
                if k in ("center", "is_twin_center"):
                    parsed[k] = int(v)
                else:
                    if v is None or v == "":
                        parsed[k] = float("nan")
                        continue
                    parsed[k] = float(v)
            rows.append(parsed)
    return rows


def mean_std(vals: List[float]) -> Tuple[float, float]:
    m = sum(vals) / len(vals)
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return m, math.sqrt(var)


def cohens_d(a: List[float], b: List[float]) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    ma = sum(a) / len(a)
    mb = sum(b) / len(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1) if len(a) > 1 else 0.0
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1) if len(b) > 1 else 0.0
    pooled = ((len(a) - 1) * va + (len(b) - 1) * vb) / max(1, (len(a) + len(b) - 2))
    if pooled <= 0:
        return float("nan")
    return (ma - mb) / math.sqrt(pooled)


def permutation_pvalue(a: List[float], b: List[float], iters: int, seed: int) -> float:
    combined = a + b
    n_a = len(a)
    n = len(combined)
    if n_a == 0 or len(b) == 0:
        return 1.0
    obs = abs(sum(a) / len(a) - sum(b) / len(b))
    count = 0
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        perm_idx = rng.permutation(n)
        a_perm = [combined[i] for i in perm_idx[:n_a]]
        b_perm = [combined[i] for i in perm_idx[n_a:]]
        diff = abs(sum(a_perm) / len(a_perm) - sum(b_perm) / len(b_perm))
        if diff >= obs:
            count += 1
    return (count + 1) / (iters + 1)


def permutation_pvalue_beta_twin(
    rows: List[Dict[str, float]],
    target: str,
    iters: int,
    seed: int,
) -> float:
    if not rows:
        return 1.0
    obs = abs(regression(rows, target=target)["beta_twin"])
    count = 0
    rng = np.random.default_rng(seed)
    for _ in range(iters):
        perm = []
        perm_labels = rng.permutation([r["is_twin_center"] for r in rows])
        for r, lab in zip(rows, perm_labels):
            r2 = dict(r)
            r2["is_twin_center"] = int(lab)
            perm.append(r2)
        diff = abs(regression(perm, target=target)["beta_twin"])
        if diff >= obs:
            count += 1
    return (count + 1) / (iters + 1)


def corr(x: List[float], y: List[float]) -> float:
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = sum((a - mx) ** 2 for a in x)
    deny = sum((b - my) ** 2 for b in y)
    den = math.sqrt(denx * deny)
    return num / den if den != 0 else 0.0


def regression(rows: List[Dict[str, float]], target: str) -> Dict[str, float]:
    y = np.array([r[target] for r in rows], dtype=float)
    log_K = np.log(np.maximum(1.0, np.array([r.get("K_used", 0.0) for r in rows], dtype=float)))
    X = np.column_stack([
        np.ones(len(rows), dtype=float),
        np.array([r["is_twin_center"] for r in rows], dtype=float),
        np.array([r["core_edges"] for r in rows], dtype=float),
        np.array([r["core_gc_fraction"] for r in rows], dtype=float),
        log_K,
    ])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return {
        "beta0": float(beta[0]),
        "beta_twin": float(beta[1]),
        "beta_core_edges": float(beta[2]),
        "beta_core_gc_fraction": float(beta[3]),
        "beta_log_K_used": float(beta[4]),
        "r2": r2,
    }


def bootstrap_beta_twin(rows: List[Dict[str, float]], target: str, iters: int, seed: int) -> Tuple[float, float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample = [rows[rng.randrange(0, len(rows))] for _ in range(len(rows))]
        vals.append(regression(sample, target=target)["beta_twin"])
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals)) - 1]
    return lo, hi


def write_latex_table(path: Path, stats: Dict[str, Dict[str, Tuple[float, float]]], counts: Dict[str, int]) -> None:
    lines = []
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Group & $n$ & Core gap (mean$\pm$sd) & Core entropy (mean$\pm$sd) \\")
    lines.append(r"\midrule")
    for label in ("twins", "non_twins"):
        n = counts[label]
        gap_m, gap_s = stats[label]["core_gc_spectral_gap"]
        ent_m, ent_s = stats[label]["core_gc_entropy"]
        lines.append(f"{label.replace('_', ' ')} & {n} & {gap_m:.4f} $\\pm$ {gap_s:.4f} & {ent_m:.4f} $\\pm$ {ent_s:.4f} \\\\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, required=True)
    p.add_argument("--out-dir", type=str, default="out")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--permutation-iters", type=int, default=10000)
    p.add_argument("--bootstrap-iters", type=int, default=2000)
    p.add_argument("--label", type=str, default="ones")
    args = p.parse_args()

    rows_all = read_rows(Path(args.input))
    rows = []
    filtered = []
    n_dropped_nan = 0
    for r in rows_all:
        vals = [
            r.get("core_edges"),
            r.get("core_gc_fraction"),
            r.get("core_gc_spectral_gap"),
            r.get("core_gc_entropy"),
            r.get("core_gc_size"),
        ]
        if any(v is None or math.isnan(v) for v in vals):
            n_dropped_nan += 1
            continue
        if r.get("core_gc_size", 0) < 3:
            n_dropped_nan += 1
            continue
        filtered.append(r)

    rows = filtered
    total_twins = sum(1 for r in rows_all if r.get("is_twin_center") == 1)
    total_non_twins = sum(1 for r in rows_all if r.get("is_twin_center") == 0)
    twins = [r for r in rows if r["is_twin_center"] == 1]
    non_twins = [r for r in rows if r["is_twin_center"] == 0]

    counts = {
        "total": len(rows),
        "twins": len(twins),
        "non_twins": len(non_twins),
        "n_dropped_nan": n_dropped_nan,
        "n_used_regression": len(rows),
        "survival_rate_total": float(len(rows) / max(1, len(rows_all))),
        "survival_rate_twins": float(len(twins) / max(1, total_twins)),
        "survival_rate_non_twins": float(len(non_twins) / max(1, total_non_twins)),
    }

    def k_stats(subset: List[Dict[str, float]]) -> Dict[str, float]:
        if not subset:
            return {"K_used_min": float("nan"), "K_used_median": float("nan"), "K_used_max": float("nan"), "hit_kmax_fraction": float("nan")}
        k_vals = sorted([r.get("K_used", float("nan")) for r in subset if not math.isnan(r.get("K_used", float("nan")))])
        if not k_vals:
            return {"K_used_min": float("nan"), "K_used_median": float("nan"), "K_used_max": float("nan"), "hit_kmax_fraction": float("nan")}
        mid = len(k_vals) // 2
        if len(k_vals) % 2 == 0:
            median_k = 0.5 * (k_vals[mid - 1] + k_vals[mid])
        else:
            median_k = k_vals[mid]
        hit = sum(1 for r in subset if r.get("hit_kmax", 0))
        return {
            "K_used_min": float(k_vals[0]),
            "K_used_median": float(median_k),
            "K_used_max": float(k_vals[-1]),
            "hit_kmax_fraction": float(hit / max(1, len(subset))),
        }

    class_stats = {
        "twins": k_stats(twins),
        "non_twins": k_stats(non_twins),
        "all": k_stats(rows),
    }

    metrics = [
        "core_gc_spectral_gap",
        "core_gc_entropy",
        "core_gc_fraction",
        "core_edges",
        "core_components",
        "twin_isolates",
        "twin_deg_sum",
    ]
    group_stats = {
        "twins": {m: mean_std([r[m] for r in twins]) for m in metrics},
        "non_twins": {m: mean_std([r[m] for r in non_twins]) for m in metrics},
    }

    d_gap = cohens_d([r["core_gc_spectral_gap"] for r in twins], [r["core_gc_spectral_gap"] for r in non_twins])
    d_ent = cohens_d([r["core_gc_entropy"] for r in twins], [r["core_gc_entropy"] for r in non_twins])
    effects = {
        "cohens_d_gap": d_gap,
        "cohens_d_entropy": d_ent,
        "effect_size_warning": (
            "undefined (zero variance or tiny group)"
            if (math.isnan(d_gap) or math.isnan(d_ent))
            else ""
        ),
    }

    pvals = {
        "perm_p_groupdiff_gap": permutation_pvalue(
            [r["core_gc_spectral_gap"] for r in twins],
            [r["core_gc_spectral_gap"] for r in non_twins],
            iters=args.permutation_iters,
            seed=args.seed,
        ),
        "perm_p_groupdiff_entropy": permutation_pvalue(
            [r["core_gc_entropy"] for r in twins],
            [r["core_gc_entropy"] for r in non_twins],
            iters=args.permutation_iters,
            seed=args.seed + 1,
        ),
        "perm_p_beta_twin_gap": permutation_pvalue_beta_twin(
            rows,
            target="core_gc_spectral_gap",
            iters=args.permutation_iters,
            seed=args.seed + 2,
        ),
        "perm_p_beta_twin_entropy": permutation_pvalue_beta_twin(
            rows,
            target="core_gc_entropy",
            iters=args.permutation_iters,
            seed=args.seed + 3,
        ),
        "perm_p_beta_twin_isolates": permutation_pvalue_beta_twin(
            rows,
            target="twin_isolates",
            iters=args.permutation_iters,
            seed=args.seed + 4,
        ),
        "perm_p_beta_twin_deg_sum": permutation_pvalue_beta_twin(
            rows,
            target="twin_deg_sum",
            iters=args.permutation_iters,
            seed=args.seed + 5,
        ),
    }

    corrs = {
        "corr_core_gap_edges": corr([r["core_gc_spectral_gap"] for r in rows], [r["core_edges"] for r in rows]),
        "corr_core_gap_gc_fraction": corr([r["core_gc_spectral_gap"] for r in rows], [r["core_gc_fraction"] for r in rows]),
    }

    reg_gap = regression(rows, target="core_gc_spectral_gap")
    reg_ent = regression(rows, target="core_gc_entropy")
    reg_iso = regression(rows, target="twin_isolates")
    reg_deg = regression(rows, target="twin_deg_sum")
    beta_lo_g, beta_hi_g = bootstrap_beta_twin(rows, target="core_gc_spectral_gap", iters=args.bootstrap_iters, seed=args.seed)
    beta_lo_e, beta_hi_e = bootstrap_beta_twin(rows, target="core_gc_entropy", iters=args.bootstrap_iters, seed=args.seed + 7)
    beta_lo_iso, beta_hi_iso = bootstrap_beta_twin(rows, target="twin_isolates", iters=args.bootstrap_iters, seed=args.seed + 13)
    beta_lo_deg, beta_hi_deg = bootstrap_beta_twin(rows, target="twin_deg_sum", iters=args.bootstrap_iters, seed=args.seed + 17)
    reg_gap["beta_twin_ci"] = [beta_lo_g, beta_hi_g]
    reg_ent["beta_twin_ci"] = [beta_lo_e, beta_hi_e]
    reg_iso["beta_twin_ci"] = [beta_lo_iso, beta_hi_iso]
    reg_deg["beta_twin_ci"] = [beta_lo_deg, beta_hi_deg]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "label": args.label,
        "counts": counts,
        "center_set": rows[0].get("center_set", "unknown") if rows else "unknown",
        "group_stats": group_stats,
        "effects": effects,
        "p_values": pvals,
        "correlations": corrs,
        "regression_gap": reg_gap,
        "regression_entropy": reg_ent,
        "regression_twin_isolates": reg_iso,
        "regression_twin_deg_sum": reg_deg,
        "class_stats": class_stats,
    }
    (out_dir / f"analysis_report_{args.label}.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    write_latex_table(out_dir / f"latex_table_groups_{args.label}.tex", group_stats, counts)

    top = sorted(rows, key=lambda r: r["core_gc_spectral_gap"], reverse=True)[:30]
    top_path = out_dir / f"top_centers_by_core_gap_{args.label}.csv"
    with top_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["center", "is_twin_center", "core_gc_spectral_gap", "core_gc_entropy", "core_edges", "core_gc_fraction"])
        for r in top:
            w.writerow([
                r["center"],
                r["is_twin_center"],
                r["core_gc_spectral_gap"],
                r["core_gc_entropy"],
                r["core_edges"],
                r["core_gc_fraction"],
            ])

    print(f"OK: wrote analysis_report_{args.label}.json, latex_table_groups_{args.label}.tex, top_centers_by_core_gap_{args.label}.csv")


if __name__ == "__main__":
    main()
