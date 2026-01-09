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
    p.add_argument("--dataset-csv", type=str, required=True)
    p.add_argument("--fit-Q", type=int, required=True)
    p.add_argument("--eval-Q-list", type=str, required=True)
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def fit_logit(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 800) -> Tuple[np.ndarray, float]:
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(steps):
        z = X @ w + b
        p = sigmoid(z)
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float((p - y).mean())
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


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


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return 0.0
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    n = len(scores)
    while i < n:
        j = i + 1
        while j < n and scores[order[j]] == scores[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    sum_ranks_pos = ranks[pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier_score(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((probs - y_true) ** 2))


def logloss(y_true: np.ndarray, probs: np.ndarray) -> float:
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

def rankdata(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a)
    ranks = np.empty_like(order, dtype=float)
    i = 0
    n = len(a)
    while i < n:
        j = i + 1
        while j < n and a[order[j]] == a[order[i]]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    if denom == 0.0:
        return 0.0
    return float(np.sum(rx * ry) / denom)


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


def save_reliability(path: Path, probs: np.ndarray, y_true: np.ndarray, title: str) -> None:
    import matplotlib.pyplot as plt

    bins = np.linspace(0, 1, 11)
    bin_centers = []
    bin_means = []
    bin_counts = []
    for i in range(len(bins) - 1):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.any():
            bin_centers.append((bins[i] + bins[i + 1]) / 2.0)
            bin_means.append(float(y_true[mask].mean()))
            bin_counts.append(int(mask.sum()))
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="#888888")
    ax.plot(bin_centers, bin_means, marker="o", color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("predicted survival")
    ax.set_ylabel("observed survival")
    for x, y, c in zip(bin_centers, bin_means, bin_counts):
        ax.text(x, y, str(c), fontsize=7, ha="center", va="bottom")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    args = parse_args()
    eval_Qs = parse_int_list(args.eval_Q_list)
    if not eval_Qs:
        raise ValueError("eval-Q-list is empty")
    if args.fit_Q not in eval_Qs:
        eval_Qs.append(args.fit_Q)
        eval_Qs = sorted(set(eval_Qs))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(Path(args.dataset_csv).open(encoding="utf-8")))
    if not rows:
        raise ValueError("Dataset is empty")

    fit_Q = args.fit_Q
    feature_name = f"ap_harm_delta_{fit_Q}"
    y_name = f"survive_{fit_Q}"

    X = np.array([[1.0 if r["killed_Q0"] == "1" else 0.0,
                   float(r[feature_name])] for r in rows], dtype=float)
    y = np.array([float(r[y_name]) for r in rows], dtype=float)

    # invert killed_Q0 to act as strong negative: use -killed in model
    X[:, 0] = -X[:, 0]
    X[:, 1] = np.log1p(X[:, 1])

    w, b = fit_logit(X, y, lr=0.2, steps=1200)
    logits = X @ w + b
    probs = sigmoid(logits)

    x_grid, y_grid = isotonic_fit(probs, y)
    probs_iso = isotonic_predict(x_grid, y_grid, probs)

    # eval across Qs
    aucs = {}
    briers = {}
    loglosses = {}
    for Q in eval_Qs:
        yq = np.array([float(r[f"survive_{Q}"]) for r in rows], dtype=float)
        aucs[Q] = auc_score(yq, probs_iso)
        briers[Q] = brier_score(yq, probs_iso)
        loglosses[Q] = logloss(yq, probs_iso)

    # calibration plots
    save_reliability(
        out_dir / "m26_calibration_fitQ.png",
        probs_iso,
        y,
        f"Calibration (fit Q={fit_Q})",
    )
    save_reliability(
        out_dir / "m26_calibration_by_Q.png",
        probs_iso,
        np.array([float(r[f"survive_{eval_Qs[-1]}"]) for r in rows], dtype=float),
        f"Calibration vs Q={eval_Qs[-1]}",
    )
    for Q in eval_Qs:
        yq = np.array([float(r[f"survive_{Q}"]) for r in rows], dtype=float)
        save_reliability(
            out_dir / f"m27_calibration_Q{Q}.png",
            probs_iso,
            yq,
            f"Calibration vs Q={Q}",
        )

    # AUC/Brier/Logloss plots
    save_line_plot(
        out_dir / "m26_auc_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"AUC": [aucs[Q] for Q in eval_Qs]},
        "AUC by Q (M26)",
        "Q",
        "AUC",
    )
    save_line_plot(
        out_dir / "m26_brier_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"Brier": [briers[Q] for Q in eval_Qs]},
        "Brier by Q (M26)",
        "Q",
        "Brier",
    )
    save_line_plot(
        out_dir / "m26_logloss_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"LogLoss": [loglosses[Q] for Q in eval_Qs]},
        "LogLoss by Q (M26)",
        "Q",
        "LogLoss",
    )
    save_line_plot(
        out_dir / "m27_auc_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"AUC": [aucs[Q] for Q in eval_Qs]},
        "AUC by Q (M27)",
        "Q",
        "AUC",
    )
    save_line_plot(
        out_dir / "m27_brier_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"Brier": [briers[Q] for Q in eval_Qs]},
        "Brier by Q (M27)",
        "Q",
        "Brier",
    )
    save_line_plot(
        out_dir / "m27_logloss_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"LogLoss": [loglosses[Q] for Q in eval_Qs]},
        "LogLoss by Q (M27)",
        "Q",
        "LogLoss",
    )

    # predicted yield/compute saved curves
    budgets = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    order = np.argsort(probs_iso)[::-1]
    pred_yield = []
    for frac in budgets:
        k = max(1, int(round(frac * len(probs_iso))))
        pred_yield.append(float(probs_iso[order[:k]].mean()))
    save_line_plot(
        out_dir / "m26_predicted_yield_vs_budget.png",
        budgets,
        {"Predicted": pred_yield},
        "Predicted yield vs budget (M26)",
        "budget fraction",
        "predicted survival",
    )

    # compute saved at 1 day relative to random
    random_rate = float(y.mean())
    saved = []
    for frac, py in zip(budgets, pred_yield):
        k = max(1, int(round(frac * len(probs_iso))))
        bad_random = k * (1.0 - random_rate)
        bad_pred = k * (1.0 - py)
        saved.append((bad_random - bad_pred) * 86400.0)
    save_line_plot(
        out_dir / "m26_predicted_compute_saved_1d.png",
        budgets,
        {"Predicted": saved},
        "Predicted compute saved vs budget (1d cost)",
        "budget fraction",
        "compute-seconds saved",
    )

    # predicted yield/compute saved for extrapolation Q targets (20M/50M)
    target_Qs = [Q for Q in eval_Qs if Q in (20000000, 50000000)]
    for Q in target_Qs:
        yq = np.array([float(r[f"survive_{Q}"]) for r in rows], dtype=float)
        pred_yield_q = []
        for frac in budgets:
            k = max(1, int(round(frac * len(probs_iso))))
            pred_yield_q.append(float(probs_iso[order[:k]].mean()))
        save_line_plot(
            out_dir / f"m27_predicted_yield_vs_budget_Q{Q}.png",
            budgets,
            {"Predicted": pred_yield_q},
            f"Predicted yield vs budget (Q={Q})",
            "budget fraction",
            "predicted survival",
        )
        random_rate_q = float(yq.mean())
        saved_q = []
        for frac, py in zip(budgets, pred_yield_q):
            k = max(1, int(round(frac * len(probs_iso))))
            bad_random = k * (1.0 - random_rate_q)
            bad_pred = k * (1.0 - py)
            saved_q.append((bad_random - bad_pred) * 86400.0)
        save_line_plot(
            out_dir / f"m27_predicted_compute_saved_1d_Q{Q}.png",
            budgets,
            {"Predicted": saved_q},
            f"Predicted compute saved vs budget (Q={Q}, 1d cost)",
            "budget fraction",
            "compute-seconds saved",
        )

    # rank stability (Spearman) vs survival labels
    rank_rows = []
    for Q in eval_Qs:
        yq = np.array([float(r[f"survive_{Q}"]) for r in rows], dtype=float)
        rho = spearman_corr(probs_iso, yq)
        rank_rows.append((Q, rho))
    rank_csv = out_dir / "m27_rank_stability.csv"
    with rank_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Q", "spearman_rho"])
        for Q, rho in rank_rows:
            w.writerow([Q, f"{rho:.6f}"])
    save_line_plot(
        out_dir / "m27_rank_stability.png",
        [float(Q) for Q, _ in rank_rows],
        {"Spearman": [rho for _, rho in rank_rows]},
        "Rank stability vs Q (M27)",
        "Q",
        "Spearman rho",
    )

    # summary table
    table_lines = [
        r"\begin{tabular}{lrrr}\hline",
        r"Q & AUC & Brier & LogLoss \\ \hline",
    ]
    for Q in eval_Qs:
        table_lines.append(
            f"{Q} & {aucs[Q]:.3f} & {briers[Q]:.4f} & {loglosses[Q]:.4f} \\\\"
        )
    table_lines.append(r"\hline\end{tabular}")
    (out_dir / "m26_table.tex").write_text("\n".join(table_lines), encoding="utf-8")
    (out_dir / "m27_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    summary = {
        "fit_Q": fit_Q,
        "eval_Qs": eval_Qs,
        "coefficients": {"w": w.tolist(), "b": b},
        "auc_by_Q": aucs,
        "brier_by_Q": briers,
        "logloss_by_Q": loglosses,
    }
    (out_dir / "m26_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # metrics by Q for M27
    metrics_csv = out_dir / "m27_metrics_by_Q.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Q", "AUC", "Brier", "LogLoss"])
        for Q in eval_Qs:
            w.writerow([Q, f"{aucs[Q]:.6f}", f"{briers[Q]:.6f}", f"{loglosses[Q]:.6f}"])

    # manifest
    manifest = {
        "dataset_csv": args.dataset_csv,
        "fit_Q": fit_Q,
        "eval_Qs": eval_Qs,
        "model": args.model,
        "seed": args.seed,
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m26_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out_dir / "m27_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK: wrote M26 model artifacts to {out_dir}")


if __name__ == "__main__":
    main()
