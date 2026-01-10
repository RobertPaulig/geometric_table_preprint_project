#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fit-dir", type=str, required=True, help="M27/M26 fit dir with m26_model_summary.json")
    p.add_argument("--dataset-csv", type=str, default="", help="optional override; otherwise taken from fit dir manifest")
    p.add_argument("--p-range", type=str, default="", help="optional inclusive p-range filter: a,b")
    p.add_argument("--Q-targets", type=str, required=True, help="comma-separated, e.g. 20000000,50000000")
    p.add_argument("--budgets", type=str, required=True, help="comma-separated fractions, e.g. 0.001,0.01,0.05")
    p.add_argument("--prp-cost-hours", type=str, required=True, help="comma-separated, e.g. 1,24,168")
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def parse_range_pair(s: str) -> Tuple[int, int]:
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(f"Bad p-range: {s!r}, expected a,b")
    a, b = int(parts[0]), int(parts[1])
    if a > b:
        a, b = b, a
    return a, b


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def isotonic_fit(scores: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(scores)
    x = scores[order]
    y = y[order]
    n = len(y)
    weights = np.ones(n)
    yhat = y.astype(float).copy()
    i = 0
    while i < n - 1:
        if yhat[i] > yhat[i + 1]:
            total_w = weights[i] + weights[i + 1]
            avg = (weights[i] * yhat[i] + weights[i + 1] * yhat[i + 1]) / total_w
            yhat[i] = avg
            yhat[i + 1] = avg
            weights[i] = total_w
            weights[i + 1] = total_w
            j = i
            while j > 0 and yhat[j - 1] > yhat[j]:
                total_w = weights[j - 1] + weights[j]
                avg = (weights[j - 1] * yhat[j - 1] + weights[j] * yhat[j]) / total_w
                yhat[j - 1] = avg
                yhat[j] = avg
                weights[j - 1] = total_w
                weights[j] = total_w
                j -= 1
            i = max(j, 0)
        else:
            i += 1
    return x, yhat


def isotonic_predict(x_grid: np.ndarray, y_grid: np.ndarray, scores: np.ndarray) -> np.ndarray:
    return np.interp(scores, x_grid, y_grid, left=y_grid[0], right=y_grid[-1])


def save_line_plot(path: Path, xs: List[float], series: Dict[str, List[float]],
                   title: str, xlabel: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
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
    budgets = parse_float_list(args.budgets)
    prp_cost_hours = parse_float_list(args.prp_cost_hours)
    if not Q_targets:
        raise ValueError("Q-targets is empty")
    if not budgets:
        raise ValueError("budgets is empty")
    if any(b <= 0 or b >= 1 for b in budgets):
        raise ValueError("budgets must be in (0,1)")
    budgets = sorted(set(budgets))
    if not prp_cost_hours:
        raise ValueError("prp-cost-hours is empty")

    summary_path = fit_dir / "m26_model_summary.json"
    manifest_path = fit_dir / "m27_manifest.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing {summary_path}")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing {manifest_path}")

    model_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fit_Q = int(model_summary["fit_Q"])
    w_vec = np.array(model_summary["coefficients"]["w"], dtype=float)
    b = float(model_summary["coefficients"]["b"])

    fit_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.dataset_csv.strip():
        dataset_csv = Path(args.dataset_csv)
    else:
        dataset_csv = Path(fit_manifest["dataset_csv"])
    if not dataset_csv.exists():
        raise FileNotFoundError(f"dataset_csv from manifest does not exist: {dataset_csv}")

    rows = list(csv.DictReader(dataset_csv.open(encoding="utf-8")))
    if not rows:
        raise ValueError("Dataset is empty")

    if args.p_range.strip():
        p_lo, p_hi = parse_range_pair(args.p_range)
        rows = [r for r in rows if p_lo <= int(r["p"]) <= p_hi]
        if not rows:
            raise ValueError(f"p-range filter left empty dataset: {p_lo},{p_hi}")

    # Reconstruct the baseline feature pipeline (M26/M27): X = [-killed_Q0, log1p(ap_harm_delta_fitQ)]
    feat_name = f"ap_harm_delta_{fit_Q}"
    y_fit_name = f"survive_{fit_Q}"
    for Q in Q_targets:
        if f"survive_{Q}" not in rows[0]:
            raise ValueError(f"Dataset missing survive_{Q} for Q target {Q}")
    if feat_name not in rows[0]:
        raise ValueError(f"Dataset missing {feat_name} for fit_Q {fit_Q}")
    if y_fit_name not in rows[0]:
        raise ValueError(f"Dataset missing {y_fit_name} for fit_Q {fit_Q}")

    p_vals = np.array([int(r["p"]) for r in rows], dtype=int)
    killed = np.array([1.0 if r["killed_Q0"] == "1" else 0.0 for r in rows], dtype=float)
    harm = np.array([float(r[feat_name]) for r in rows], dtype=float)
    X = np.column_stack([-killed, np.log1p(harm)]).astype(float)

    logits = X @ w_vec + b
    probs = sigmoid(logits)
    y_fit = np.array([float(r[y_fit_name]) for r in rows], dtype=float)

    # Refit isotonic mapping on the same dataset (no new data generation; ensures calibrated probs like M27)
    x_grid, y_grid = isotonic_fit(probs, y_fit)
    probs_iso = isotonic_predict(x_grid, y_grid, probs)

    order = np.argsort(probs_iso)[::-1]  # higher predicted survival first (queue for expensive tests)

    summary: Dict[str, object] = {
        "fit_dir": str(fit_dir),
        "dataset_csv": str(dataset_csv),
        "p_range": args.p_range.strip() or None,
        "fit_Q": fit_Q,
        "Q_targets": Q_targets,
        "budgets": budgets,
        "prp_cost_hours": prp_cost_hours,
        "N": int(len(rows)),
        "notes": [
            "Queue sorts by predicted survival (descending) for the chosen fit model.",
            "Savings use a simple compute-saved proxy relative to random selection at each Q_target.",
        ],
        "targets": {},
    }

    for Q in Q_targets:
        yq = np.array([float(r[f"survive_{Q}"]) for r in rows], dtype=float)
        base_survival = float(yq.mean())
        base_bad = 1.0 - base_survival

        # queues
        for frac in [0.01, 0.05, 0.10, 0.001]:
            if frac not in budgets and frac not in (0.001, 0.01):
                continue
        top1 = max(1, int(round(0.01 * len(rows))))
        top_path = out_dir / f"m30_queue_Q{Q}_top1pct.csv"
        with top_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["p", "pred_survival", "killed_Q0"])
            for idx in order[:top1]:
                writer.writerow([int(p_vals[idx]), f"{probs_iso[idx]:.12f}", int(killed[idx])])

        savings_rows = []
        for frac in budgets:
            k = max(1, int(round(frac * len(rows))))
            py = float(probs_iso[order[:k]].mean())  # expected survival of the chosen queue slice
            bad_pred = k * (1.0 - py)
            bad_random = k * base_bad
            for cost_h in prp_cost_hours:
                saved_seconds = (bad_random - bad_pred) * cost_h * 3600.0
                savings_rows.append(
                    {
                        "Q": Q,
                        "budget_fraction": frac,
                        "prp_cost_hours": cost_h,
                        "base_survival_rate": base_survival,
                        "predicted_survival_topk": py,
                        "predicted_dead_topk": 1.0 - py,
                        "random_dead_topk": base_bad,
                        "saved_compute_seconds": saved_seconds,
                    }
                )

        savings_csv = out_dir / f"m30_savings_by_budget_Q{Q}.csv"
        with savings_csv.open("w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "Q",
                "budget_fraction",
                "prp_cost_hours",
                "base_survival_rate",
                "predicted_survival_topk",
                "predicted_dead_topk",
                "random_dead_topk",
                "saved_compute_seconds",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(savings_rows)

        # plot per cost scenario
        xs = budgets
        series = {}
        for cost_h in prp_cost_hours:
            ys = [float(r["saved_compute_seconds"]) for r in savings_rows if float(r["prp_cost_hours"]) == cost_h]
            series[f"PRP={int(cost_h)}h"] = ys
        savings_png = out_dir / f"m30_savings_by_budget_Q{Q}.png"
        save_line_plot(
            savings_png,
            xs,
            series,
            f"M30: compute saved vs budget (Q={Q})",
            "budget fraction (top-k by predicted survival)",
            "compute-seconds saved (vs random)",
        )

        summary["targets"][str(Q)] = {
            "base_survival_rate": base_survival,
            "base_dead_rate": base_bad,
            "queue_top1pct_path": str(top_path),
            "savings_csv": str(savings_csv),
            "savings_png": str(savings_png),
        }

    (out_dir / "m30_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "command": "python code/scripts/m30_deployment_queue.py",
        "params": {
            "fit_dir": str(fit_dir),
            "dataset_csv": str(dataset_csv),
            "p_range": args.p_range.strip() or None,
            "Q_targets": Q_targets,
            "budgets": budgets,
            "prp_cost_hours": prp_cost_hours,
        },
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m30_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK: wrote M30 deployment artifacts to {out_dir}")


if __name__ == "__main__":
    main()

