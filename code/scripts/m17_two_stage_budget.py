#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m14b-summary", type=str, required=True)
    p.add_argument("--layers-points", type=str, required=True)
    p.add_argument("--N-raw", type=float, default=1e6)
    p.add_argument("--N-raw-big", type=float, default=1e9)
    p.add_argument("--workers", type=str, required=True)
    p.add_argument("--c1-list", type=str, required=True)
    p.add_argument("--c2-list", type=str, required=True)
    p.add_argument("--r1-list", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def read_summary(path: Path) -> Dict[int, Dict[str, float]]:
    rows: Dict[int, Dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            layers = int(row["layers"])
            rows[layers] = {
                "survival": float(row["survival_rate"]),
                "throughput": float(row["throughput"]),
            }
    return rows


def save_line_plot(path: Path, xs: List[int], ys_list: List[List[float]], labels: List[str],
                   title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    for ys, label in zip(ys_list, labels):
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_heatmap(path: Path, mat: np.ndarray, title: str, xlabel: str, ylabel: str,
                 xticklabels: List[str], yticklabels: List[str]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(xticklabels)))
    ax.set_xticklabels(xticklabels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(yticklabels)))
    ax.set_yticklabels(yticklabels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers_points = parse_int_list(args.layers_points)
    c1_list = parse_float_list(args.c1_list)
    c2_list = parse_float_list(args.c2_list)
    r1_list = parse_float_list(args.r1_list)
    workers_list = parse_int_list(args.workers)

    summary = read_summary(Path(args.m14b_summary))

    xs = []
    surv = []
    gen_cost = []
    for L in layers_points:
        if L not in summary:
            raise ValueError(f"Layers {L} not found in summary")
        xs.append(L)
        s = summary[L]["survival"]
        thr = summary[L]["throughput"]
        cg = 1.0 / thr if thr > 0 else 0.0
        surv.append(s)
        gen_cost.append(cg)

    combos = []
    for c1 in c1_list:
        for c2 in c2_list:
            for r1 in r1_list:
                combos.append((c1, c2, r1))

    # Use up to 9 combos for readable plots
    plot_combos = combos[:9]
    labels = [f"c1={c1:g}, c2={c2:g}, r1={r1:g}" for c1, c2, r1 in plot_combos]

    total_cost_lines = []
    time_saved_lines = []
    for c1, c2, r1 in plot_combos:
        total_cost = [cg + s * (c1 + r1 * c2) for cg, s in zip(gen_cost, surv)]
        total_cost_lines.append(total_cost)
        base = total_cost[0] * args.N_raw
        time_saved = [base - (t * args.N_raw) for t in total_cost]
        time_saved_lines.append(time_saved)

    save_line_plot(
        out_dir / "m17_total_cost_vs_layers.png",
        xs,
        total_cost_lines,
        labels,
        "Two-stage total cost per raw candidate (M17)",
        "layers",
        "seconds per raw candidate",
    )

    save_line_plot(
        out_dir / "m17_time_saved_vs_layers.png",
        xs,
        time_saved_lines,
        labels,
        f"Time saved vs layers for N_raw={args.N_raw:.0e}",
        "layers",
        "seconds saved",
    )

    # Optimal layers heatmap for each (c2, r1) at c1 = median
    c1_ref = c1_list[len(c1_list) // 2]
    heat = np.zeros((len(r1_list), len(c2_list)), dtype=int)
    for i, r1 in enumerate(r1_list):
        for j, c2 in enumerate(c2_list):
            costs = [cg + s * (c1_ref + r1 * c2) for cg, s in zip(gen_cost, surv)]
            best_idx = int(np.argmin(costs))
            heat[i, j] = xs[best_idx]

    save_heatmap(
        out_dir / "m17_optimal_layers_heatmap.png",
        heat,
        f"Optimal layers (c1={c1_ref:g})",
        "c2 (sec)",
        "r1",
        [f"{c2:g}" for c2 in c2_list],
        [f"{r1:g}" for r1 in r1_list],
    )

    # Break-even thresholds for steps using c1_ref and r1_ref
    c1_ref = c1_list[len(c1_list) // 2]
    r1_ref = r1_list[len(r1_list) // 2]
    break_rows = []
    for i in range(1, len(xs)):
        s_old, s_new = surv[i - 1], surv[i]
        cg_old, cg_new = gen_cost[i - 1], gen_cost[i]
        ds = s_old - s_new
        dcg = cg_new - cg_old
        if ds > 0 and r1_ref > 0:
            c2_star = (dcg / ds - c1_ref) / r1_ref
        else:
            c2_star = 0.0
        break_rows.append({
            "step": f"{xs[i-1]}->{xs[i]}",
            "delta_gen_cost": dcg,
            "delta_survival": ds,
            "c2_threshold_sec": c2_star,
        })

    break_csv = out_dir / "m17_break_even.csv"
    with break_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(break_rows[0].keys()))
        w.writeheader()
        w.writerows(break_rows)

    break_tex = out_dir / "m17_break_even_table.tex"
    with break_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrr}\\hline\n")
        f.write("Layers & $\\Delta c_g$ (s) & $\\Delta s$ & $c_2^*$ (s) \\\\\\hline\n")
        for row in break_rows:
            f.write(f"{row['step']} & {row['delta_gen_cost']:.3e} & {row['delta_survival']:.3f} & {row['c2_threshold_sec']:.3g} \\\\\n")
        f.write("\\hline\\end{tabular}\n")

    # Scenario table (up to 9)
    scenario_rows = []
    for c1, c2, r1 in combos[:9]:
        costs = [cg + s * (c1 + r1 * c2) for cg, s in zip(gen_cost, surv)]
        best_idx = int(np.argmin(costs))
        scenario_rows.append({
            "c1": c1,
            "c2": c2,
            "r1": r1,
            "best_L": xs[best_idx],
            "best_cost": costs[best_idx],
        })

    scen_csv = out_dir / "m17_scenarios.csv"
    with scen_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scenario_rows[0].keys()))
        w.writeheader()
        w.writerows(scenario_rows)

    scen_tex = out_dir / "m17_scenarios_table.tex"
    with scen_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{rrrrr}\\hline\n")
        f.write("$c_1$ & $c_2$ & $r_1$ & $L^*$ & $C(L^*)$ \\\\\\hline\n")
        for row in scenario_rows:
            f.write(
                f"{row['c1']:.0e} & {row['c2']:.0e} & {row['r1']:.0e} & {row['best_L']} & {row['best_cost']:.3e} \\\\\n"
            )
        f.write("\\hline\\end{tabular}\n")

    summary_json = out_dir / "m17_summary.json"
    summary_json.write_text(json.dumps({
        "layers_points": xs,
        "N_raw": args.N_raw,
        "N_raw_big": args.N_raw_big,
        "workers": workers_list,
        "c1_list": c1_list,
        "c2_list": c2_list,
        "r1_list": r1_list,
        "break_even_ref": {"c1": c1_ref, "r1": r1_ref},
        "scenarios": scenario_rows,
    }, indent=2), encoding="utf-8")

    print(f"OK: wrote M17 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
