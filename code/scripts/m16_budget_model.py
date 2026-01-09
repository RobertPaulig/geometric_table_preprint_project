#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m14b-summary", type=str, required=True)
    p.add_argument("--layers-points", type=str, required=True)
    p.add_argument("--test-costs", type=str, required=True)
    p.add_argument("--N-raw", type=float, default=1e9)
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


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layers_points = parse_int_list(args.layers_points)
    test_costs = parse_float_list(args.test_costs)
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

    total_cost_lines = []
    time_saved_lines = []
    labels = []
    for ct in test_costs:
        labels.append(f"{ct:g}s")
        total_cost = [cg + s * ct for cg, s in zip(gen_cost, surv)]
        total_cost_lines.append(total_cost)
        time = [args.N_raw * c for c in total_cost]
        base = time[0]
        saved = [base - t for t in time]
        time_saved_lines.append(saved)

    save_line_plot(
        out_dir / "m16_total_cost_vs_layers.png",
        xs,
        total_cost_lines,
        labels,
        "Total cost per raw candidate vs layers (M16)",
        "layers",
        "seconds per raw candidate",
    )

    save_line_plot(
        out_dir / "m16_time_saved_vs_layers.png",
        xs,
        time_saved_lines,
        labels,
        f"Time saved vs layers for N_raw={args.N_raw:.0e}",
        "layers",
        "seconds saved",
    )

    # Break-even table between successive layer points
    break_rows = []
    for i in range(1, len(xs)):
        L0, L1 = xs[i - 1], xs[i]
        dg = gen_cost[i] - gen_cost[i - 1]
        ds = surv[i - 1] - surv[i]
        thr_ms = (dg / ds) * 1e3 if ds > 0 else 0.0
        break_rows.append({
            "step": f"{L0}->{L1}",
            "delta_gen_cost": dg,
            "delta_survival": ds,
            "test_cost_threshold_ms": thr_ms,
        })

    break_csv = out_dir / "m16_break_even.csv"
    with break_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(break_rows[0].keys()))
        w.writeheader()
        w.writerows(break_rows)

    break_tex = out_dir / "m16_break_even_table.tex"
    with break_tex.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrr}\\hline\n")
        f.write("Layers & $\\Delta c_g$ (s) & $\\Delta s$ & $c_t^*$ (ms) \\\\\\hline\n")
        for row in break_rows:
            f.write(f"{row['step']} & {row['delta_gen_cost']:.3e} & {row['delta_survival']:.3f} & {row['test_cost_threshold_ms']:.3f} \\\\\n")
        f.write("\\hline\\end{tabular}\n")

    summary_json = out_dir / "m16_summary.json"
    summary_json.write_text(json.dumps({
        "layers_points": xs,
        "test_costs_sec": test_costs,
        "N_raw": args.N_raw,
        "break_even": break_rows,
    }, indent=2), encoding="utf-8")

    print(f"OK: wrote M16 artifacts to {out_dir}")


if __name__ == "__main__":
    main()
