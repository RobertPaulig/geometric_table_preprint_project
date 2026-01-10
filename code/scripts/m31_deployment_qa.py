#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fit-dir", type=str, required=True)
    p.add_argument("--Q-targets", type=str, required=True)
    p.add_argument("--budgets", type=str, required=True)
    p.add_argument("--prp-cost-hours", type=str, required=True)
    p.add_argument("--runs", type=str, nargs="+", required=True, help="entries: name:a,b")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_run(s: str) -> Tuple[str, str]:
    if ":" not in s:
        raise ValueError(f"Bad run spec: {s!r}, expected name:a,b")
    name, pr = s.split(":", 1)
    name = name.strip()
    pr = pr.strip()
    if not name:
        raise ValueError(f"Bad run name in {s!r}")
    return name, pr


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_line_plot(path: Path, xs: List[float], series: Dict[str, List[float]],
                   title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for label, ys in series.items():
        ax.plot(xs, ys, marker="o", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    fit_dir = Path(args.fit_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Q_targets = parse_int_list(args.Q_targets)
    budgets = sorted(set(parse_float_list(args.budgets)))
    prp_cost_hours = sorted(set(parse_float_list(args.prp_cost_hours)))
    if not Q_targets or not budgets or not prp_cost_hours:
        raise ValueError("empty Q-targets/budgets/prp-cost-hours")

    runs = [parse_run(s) for s in args.runs]
    run_dirs: Dict[str, Path] = {}

    # Run M30 for each slice (writes run_<name>/...).
    for name, p_range in runs:
        run_path = out_dir / f"run_{name}"
        run_path.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "code/scripts/m30_deployment_queue.py",
            "--fit-dir",
            str(fit_dir),
            "--p-range",
            p_range,
            "--Q-targets",
            ",".join(str(q) for q in Q_targets),
            "--budgets",
            ",".join(str(b) for b in budgets),
            "--prp-cost-hours",
            ",".join(str(h) for h in prp_cost_hours),
            "--out-dir",
            str(run_path),
        ]
        subprocess.run(cmd, check=True)
        run_dirs[name] = run_path

    # Collect savings and queue overlap.
    savings_rows: List[Dict[str, object]] = []
    queue_sets: Dict[Tuple[str, int], set[int]] = {}

    for name, _ in runs:
        run_path = run_dirs[name]
        for Q in Q_targets:
            savings_csv = run_path / f"m30_savings_by_budget_Q{Q}.csv"
            for row in csv.DictReader(savings_csv.open(encoding="utf-8")):
                savings_rows.append(
                    {
                        "run": name,
                        "Q": int(row["Q"]),
                        "budget_fraction": float(row["budget_fraction"]),
                        "prp_cost_hours": float(row["prp_cost_hours"]),
                        "saved_compute_seconds": float(row["saved_compute_seconds"]),
                    }
                )

            queue_csv = run_path / f"m30_queue_Q{Q}_top1pct.csv"
            ps = set()
            for r in csv.DictReader(queue_csv.open(encoding="utf-8")):
                ps.add(int(r["p"]))
            queue_sets[(name, Q)] = ps

    savings_out = out_dir / "m31_savings_summary.csv"
    with savings_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["run", "Q", "budget_fraction", "prp_cost_hours", "saved_compute_seconds"],
        )
        writer.writeheader()
        writer.writerows(savings_rows)

    overlap_rows: List[Dict[str, object]] = []
    names = [n for n, _ in runs]
    for Q in Q_targets:
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a = queue_sets[(names[i], Q)]
                b = queue_sets[(names[j], Q)]
                inter = len(a & b)
                union = len(a | b)
                jac = (inter / union) if union else float("nan")
                overlap_rows.append(
                    {
                        "Q": Q,
                        "run_a": names[i],
                        "run_b": names[j],
                        "jaccard_top1pct": jac,
                        "intersection_size": inter,
                        "union_size": union,
                    }
                )

    overlap_out = out_dir / "m31_queue_overlap.csv"
    with overlap_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Q", "run_a", "run_b", "jaccard_top1pct", "intersection_size", "union_size"],
        )
        writer.writeheader()
        writer.writerows(overlap_rows)

    # Plot: saved_compute_seconds at budget=1% and cost=24h for each Q across runs.
    plot_budget = 0.01
    plot_cost = 24.0 if 24.0 in prp_cost_hours else prp_cost_hours[0]
    series = {}
    for Q in Q_targets:
        ys = []
        for name in names:
            row = next(
                r
                for r in savings_rows
                if r["run"] == name
                and int(r["Q"]) == Q
                and abs(float(r["budget_fraction"]) - plot_budget) < 1e-12
                and abs(float(r["prp_cost_hours"]) - plot_cost) < 1e-12
            )
            ys.append(float(row["saved_compute_seconds"]))
        series[f"Q={Q}"] = ys

    plot_path = out_dir / "m31_savings_summary.png"
    save_line_plot(
        plot_path,
        list(range(len(names))),
        series,
        f"M31: savings dispersion across runs (budget={plot_budget:.3g}, PRP={int(plot_cost)}h)",
        "run index",
        "saved_compute_seconds",
    )

    # TeX table: mean/std over runs for budget=1% and PRP=24h, plus mean jaccard.
    table_lines = [
        r"\begin{tabular}{lrrr}\hline",
        r"Q & mean saved (s) & std (s) & mean Jaccard@top1\% \\ \hline",
    ]
    for Q in Q_targets:
        vals = []
        for name in names:
            row = next(
                r
                for r in savings_rows
                if r["run"] == name
                and int(r["Q"]) == Q
                and abs(float(r["budget_fraction"]) - plot_budget) < 1e-12
                and abs(float(r["prp_cost_hours"]) - plot_cost) < 1e-12
            )
            vals.append(float(row["saved_compute_seconds"]))
        vals = np.array(vals, dtype=float)

        jac = [
            float(r["jaccard_top1pct"])
            for r in overlap_rows
            if int(r["Q"]) == Q and not math.isnan(float(r["jaccard_top1pct"]))
        ]
        jac_mean = float(np.mean(jac)) if jac else float("nan")
        table_lines.append(f"{Q} & {float(np.mean(vals)):.0f} & {float(np.std(vals)):.0f} & {jac_mean:.3f} \\\\")
    table_lines.append(r"\hline\end{tabular}")
    (out_dir / "m31_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    runs_manifest = {
        "fit_dir": str(fit_dir),
        "Q_targets": Q_targets,
        "budgets": budgets,
        "prp_cost_hours": prp_cost_hours,
        "runs": [{"name": name, "p_range": pr, "out_dir": str(run_dirs[name])} for name, pr in runs],
        "run_files_sha256": {},
    }
    for name in names:
        run_path = run_dirs[name]
        for p in sorted(run_path.glob("*")):
            if p.is_file():
                runs_manifest["run_files_sha256"][f"run_{name}/{p.name}"] = sha256_file(p)
    (out_dir / "m31_runs_manifest.json").write_text(json.dumps(runs_manifest, indent=2), encoding="utf-8")

    manifest = {"files": {}}
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(out_dir)).replace("\\", "/")
            manifest["files"][rel] = sha256_file(p)
    (out_dir / "m31_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK: wrote M31 QA artifacts to {out_dir}")


if __name__ == "__main__":
    import math

    main()

