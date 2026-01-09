#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import gzip
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from geometric_table import compute_core_metrics_from_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--wheel-csv", type=str, required=True)
    p.add_argument("--B", type=int, required=True)
    p.add_argument("--core-rt", type=int, default=30)
    p.add_argument("--row-mode", type=str, default="wheel", choices=["wheel"])
    p.add_argument("--K-list", type=str, required=True)
    p.add_argument("--eps-list", type=str, required=True)
    p.add_argument("--layer-primes", type=str, required=True)
    p.add_argument("--sample", type=int, default=2000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def open_csv(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def read_t_lists(path: Path) -> Tuple[List[int], List[int]]:
    twins: List[int] = []
    non: List[int] = []
    with open_csv(path) as f:
        r = csv.DictReader(f)
        for row in r:
            t = int(row["t"])
            is_twin = int(row["is_twin"])
            if is_twin:
                twins.append(t)
            else:
                non.append(t)
    return twins, non


def sample_t(twins: List[int], non: List[int], n: int, seed: int) -> Tuple[List[int], int, int]:
    rng = random.Random(seed)
    target_twins = min(len(twins), n // 2)
    target_non = min(len(non), n - target_twins)
    if target_twins < n // 2:
        target_non = min(len(non), n - target_twins)
    twin_sample = rng.sample(twins, target_twins) if target_twins else []
    non_sample = rng.sample(non, target_non) if target_non else []
    return twin_sample + non_sample, target_twins, target_non


def mod_inv(B: int, p: int) -> int:
    return pow(B % p, -1, p)


def dist_to_forbid(t: int, p: int, inv: int) -> int:
    r1 = inv % p
    r2 = (-inv) % p
    d1 = min((t - r1) % p, (r1 - t) % p)
    d2 = min((t - r2) % p, (r2 - t) % p)
    return min(d1, d2)


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def variance_of_means(values: np.ndarray, groups: List[np.ndarray]) -> float:
    means = []
    for g in groups:
        if g.any():
            means.append(float(values[g].mean()))
    if not means:
        return 0.0
    return float(np.var(means))


def corr_or_zero(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) <= 1:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def save_bar(path: Path, xs: List[int], ys: List[float], title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.bar(xs, ys, width=0.8)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_boxplot(path: Path, groups: List[np.ndarray], labels: List[str], title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    primes = parse_int_list(args.layer_primes)
    if not primes:
        raise ValueError("--layer-primes is empty")
    p0 = primes[0]
    K_list = parse_int_list(args.K_list)
    eps_list = parse_float_list(args.eps_list)
    if not K_list or not eps_list:
        raise ValueError("--K-list or --eps-list is empty")

    twins, non = read_t_lists(Path(args.wheel_csv))
    t_list, n_twins, n_non = sample_t(twins, non, args.sample, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "m13_metrics.csv"

    invs = {p: mod_inv(args.B, p) for p in primes}

    metric_fields = [
        "n_components",
        "gc_size",
        "gc_fraction",
        "second_gc_size",
        "second_gc_fraction",
        "isolated_nodes",
        "triangle_count",
        "transitivity",
        "avg_clustering",
        "adj_spectral_radius",
        "adj_entropy",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = [
            "t", "is_twin", "K", "eps",
            f"t_mod_{p0}", f"dist_to_forbid_{p0}",
            "layer_allow_L1", "layer_allow_L2", "layer_allow_L3",
        ] + metric_fields
        w.writerow(header)

        for t in t_list:
            base_rows = [args.B * (t + i) for i in range(-args.core_rt, args.core_rt + 1)]
            r = t % p0
            dist = dist_to_forbid(t, p0, invs[p0])
            forb_flags = []
            for p in primes[:3]:
                inv = invs[p]
                r_p = t % p
                is_forb = int(r_p == inv % p or r_p == (-inv) % p)
                forb_flags.append(is_forb)
            layer_hits = [0, 0, 0]
            if len(forb_flags) >= 1:
                layer_hits[0] = int(forb_flags[0] == 1)
            if len(forb_flags) >= 2:
                layer_hits[1] = int(forb_flags[0] or forb_flags[1])
            if len(forb_flags) >= 3:
                layer_hits[2] = int(forb_flags[0] or forb_flags[1] or forb_flags[2])
            layer_allow = [int(h == 0) for h in layer_hits]

            for K in K_list:
                for eps in eps_list:
                    metrics = compute_core_metrics_from_rows(
                        rows=base_rows,
                        center_value=None,
                        K=K,
                        primitive=False,
                        weight="ones",
                        eps=eps,
                        include_twin=False,
                    )
                    row = [
                        t, int(t in twins), K, eps,
                        r, dist,
                        layer_allow[0], layer_allow[1], layer_allow[2],
                        metrics["core_components"],
                        metrics["core_gc_size"],
                        metrics["core_gc_fraction"],
                        metrics["core_second_gc_size"],
                        metrics["core_second_gc_fraction"],
                        metrics["core_isolated_nodes"],
                        metrics["core_triangle_count"],
                        metrics["core_transitivity"],
                        metrics["core_avg_clustering"],
                        metrics["core_adj_spectral_radius"],
                        metrics["core_adj_entropy"],
                    ]
                    w.writerow(row)

    summary_rows = []
    import pandas as pd

    df = pd.read_csv(out_csv)
    for K in K_list:
        for eps in eps_list:
            sub = df[(df["K"] == K) & (df["eps"] == eps)]
            if sub.empty:
                continue
            t_mod = sub[f"t_mod_{p0}"].astype(int).to_numpy()
            dist = sub[f"dist_to_forbid_{p0}"].to_numpy()
            masks_mod = [t_mod == r for r in range(p0)]
            masks_layer = [
                sub["layer_allow_L1"] == 1,
                sub["layer_allow_L2"] == 1,
                sub["layer_allow_L3"] == 1,
            ]
            for metric in metric_fields:
                vals = sub[metric].to_numpy()
                mod_score = variance_of_means(vals, masks_mod)
                layer_score = variance_of_means(vals, masks_layer)
                corr = corr_or_zero(dist, vals)
                summary_rows.append({
                    "metric": metric,
                    "K": int(K),
                    "eps": float(eps),
                    "mod_effect_score": float(mod_score),
                    "layer_effect_score": float(layer_score),
                    "corr_dist": float(corr),
                })

    summary_csv = out_dir / "m13_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

    if summary_rows:
        df_sum = pd.DataFrame(summary_rows)
        df_sum["combined_score"] = df_sum["mod_effect_score"] + df_sum["layer_effect_score"] + df_sum["corr_dist"].abs()
        idx_best = df_sum["combined_score"].idxmax()
        best = df_sum.loc[idx_best].to_dict()

        per_metric = []
        for metric, g in df_sum.groupby("metric"):
            i = g["combined_score"].idxmax()
            per_metric.append(df_sum.loc[i].to_dict())
        per_metric.sort(key=lambda x: x["combined_score"], reverse=True)
        top2 = per_metric[:2]

        best_metric = best["metric"]
        best_K = int(best["K"])
        best_eps = float(best["eps"])
        sub_best = df[(df["K"] == best_K) & (df["eps"] == best_eps)]

        means_by_mod = []
        for r in range(p0):
            mask = sub_best[f"t_mod_{p0}"] == r
            means_by_mod.append(float(sub_best.loc[mask, best_metric].mean()) if mask.any() else 0.0)
        save_bar(
            out_dir / "m13_best_metric_mean_by_mod_p0.png",
            list(range(p0)),
            means_by_mod,
            f"{best_metric} mean by t mod {p0} (K={best_K}, eps={best_eps:g})",
            "residue",
            best_metric,
        )

        groups = [
            sub_best.loc[sub_best["layer_allow_L1"] == 1, best_metric].to_numpy(),
            sub_best.loc[sub_best["layer_allow_L2"] == 1, best_metric].to_numpy(),
            sub_best.loc[sub_best["layer_allow_L3"] == 1, best_metric].to_numpy(),
        ]
        save_boxplot(
            out_dir / "m13_best_metric_by_layer_box.png",
            groups,
            ["L1", "L2", "L3"],
            f"{best_metric} by layer (K={best_K}, eps={best_eps:g})",
            best_metric,
        )

        summary = {
            "best_metric": best_metric,
            "best_K": best_K,
            "best_eps": best_eps,
            "best_combined_score": float(best["combined_score"]),
            "top2_metrics": [
                {
                    "metric": x["metric"],
                    "K": int(x["K"]),
                    "eps": float(x["eps"]),
                    "combined_score": float(x["combined_score"]),
                }
                for x in top2
            ],
            "sample": {
                "n_total": int(len(df)),
                "n_twins": int((df["is_twin"] == 1).sum()),
                "n_non": int((df["is_twin"] == 0).sum()),
            },
        }
        (out_dir / "m13_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        def tex_escape(name: str) -> str:
            return name.replace("_", "\\_")

        table_path = out_dir / "m13_top2_table.tex"
        with table_path.open("w", encoding="utf-8") as f:
            f.write("\\begin{tabular}{lcc}\\hline\n")
            f.write("metric & K & eps \\\\\\hline\n")
            for x in top2:
                metric = tex_escape(str(x["metric"]))
                f.write(f"{metric} & {int(x['K'])} & {float(x['eps']):g} \\\\\n")
            f.write("\\hline\\end{tabular}\n")

    print(f"OK: wrote {out_csv} (twins={n_twins}, non={n_non})")


if __name__ == "__main__":
    main()
