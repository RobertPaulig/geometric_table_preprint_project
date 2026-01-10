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

    p.add_argument("--bin-Q-list", type=str, default="", help="comma-separated, must be <= fit-Q and include fit-Q")
    p.add_argument("--use-count", type=int, default=1)
    p.add_argument("--use-diff", type=int, default=1)
    p.add_argument("--use-shape", type=int, default=1)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def signed_log1p(x: np.ndarray) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def fit_logit(X: np.ndarray, y: np.ndarray, lr: float = 0.2, steps: int = 1400) -> Tuple[np.ndarray, float]:
    w_vec = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(steps):
        z = X @ w_vec + b
        p = sigmoid(z)
        grad_w = X.T @ (p - y) / len(y)
        grad_b = float((p - y).mean())
        w_vec -= lr * grad_w
        b -= lr * grad_b
    return w_vec, b


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


def build_feature_matrix(
    killed_Q0: np.ndarray,
    harm_cum: np.ndarray,
    count_cum: np.ndarray,
    bin_Qs: List[int],
    use_count: bool,
    use_diff: bool,
    use_shape: bool,
) -> Tuple[np.ndarray, List[str]]:
    n = len(killed_Q0)
    n_bins = harm_cum.shape[1]
    if n_bins != len(bin_Qs):
        raise ValueError("harm_cum columns must match bin_Qs")

    harm_bins = np.empty_like(harm_cum)
    harm_bins[:, 0] = harm_cum[:, 0]
    harm_bins[:, 1:] = harm_cum[:, 1:] - harm_cum[:, :-1]

    count_bins = np.empty_like(count_cum)
    count_bins[:, 0] = count_cum[:, 0]
    count_bins[:, 1:] = count_cum[:, 1:] - count_cum[:, :-1]

    feats: List[np.ndarray] = []
    names: List[str] = []

    feats.append(-killed_Q0.astype(float))
    names.append("neg_killed_Q0")

    for i, Q in enumerate(bin_Qs):
        feats.append(np.log1p(harm_bins[:, i]))
        names.append(f"log1p_harm_bin_Q{Q}")

    if use_count:
        for i, Q in enumerate(bin_Qs):
            feats.append(np.log1p(count_bins[:, i]))
            names.append(f"log1p_count_bin_Q{Q}")

    if use_diff and n_bins >= 2:
        harm_diff_fwd = harm_bins[:, 1:] - harm_bins[:, :-1]
        for i in range(n_bins - 1):
            feats.append(signed_log1p(harm_diff_fwd[:, i]))
            names.append(f"harm_diff_fwd_endQ{bin_Qs[i + 1]}")
        if n_bins >= 3:
            harm_diff_c = (harm_bins[:, 2:] - harm_bins[:, :-2]) / 2.0
            for i in range(n_bins - 2):
                feats.append(signed_log1p(harm_diff_c[:, i]))
                names.append(f"harm_diff_central_atQ{bin_Qs[i + 1]}")

        if use_count:
            count_diff_fwd = count_bins[:, 1:] - count_bins[:, :-1]
            for i in range(n_bins - 1):
                feats.append(signed_log1p(count_diff_fwd[:, i].astype(float)))
                names.append(f"count_diff_fwd_endQ{bin_Qs[i + 1]}")
            if n_bins >= 3:
                count_diff_c = (count_bins[:, 2:] - count_bins[:, :-2]) / 2.0
                for i in range(n_bins - 2):
                    feats.append(signed_log1p(count_diff_c[:, i].astype(float)))
                    names.append(f"count_diff_central_atQ{bin_Qs[i + 1]}")

    if use_shape:
        hb = harm_bins
        total = hb.sum(axis=1)
        probs = np.zeros_like(hb, dtype=float)
        mask = total > 0
        probs[mask] = hb[mask] / total[mask, None]
        eps = 1e-12

        entropy = -np.sum(probs * np.log(probs + eps), axis=1)
        max_frac = np.max(probs, axis=1)
        if n_bins >= 2:
            tail_frac = np.sum(probs[:, -2:], axis=1)
        else:
            tail_frac = probs[:, -1]
        max_jump = np.zeros(n, dtype=float)
        if n_bins >= 2:
            max_jump = np.max(np.abs(np.diff(probs, axis=1)), axis=1)

        # slope of cumulative harm vs log(Q)
        x = np.log(np.array(bin_Qs, dtype=float))
        x_c = x - float(np.mean(x))
        denom = float(np.sum(x_c ** 2)) if float(np.sum(x_c ** 2)) > 0 else 1.0
        cum = np.cumsum(hb, axis=1)
        cum_c = cum - cum.mean(axis=1, keepdims=True)
        slope = np.sum(cum_c * x_c[None, :], axis=1) / denom

        feats.extend([entropy, max_frac, tail_frac, slope, max_jump])
        names.extend(["harm_entropy", "harm_max_frac", "harm_tail_frac", "harm_slope_cum_vs_logQ", "harm_max_jump"])

    X = np.column_stack(feats).astype(float)
    return X, names


def main() -> None:
    args = parse_args()
    eval_Qs = parse_int_list(args.eval_Q_list)
    if not eval_Qs:
        raise ValueError("eval-Q-list is empty")
    if args.fit_Q not in eval_Qs:
        eval_Qs.append(args.fit_Q)
        eval_Qs = sorted(set(eval_Qs))

    use_count = int(args.use_count) == 1
    use_diff = int(args.use_diff) == 1
    use_shape = int(args.use_shape) == 1

    bin_Qs = parse_int_list(args.bin_Q_list) if args.bin_Q_list.strip() else [Q for Q in eval_Qs if Q <= args.fit_Q]
    bin_Qs = sorted(set(bin_Qs))
    if not bin_Qs:
        raise ValueError("bin-Q-list resolved to empty")
    if any(Q > args.fit_Q for Q in bin_Qs):
        raise ValueError("Leakage guard: bin-Q-list contains Q > fit-Q")
    if args.fit_Q not in bin_Qs:
        raise ValueError("Leakage guard: bin-Q-list must include fit-Q")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(Path(args.dataset_csv).open(encoding="utf-8")))
    if not rows:
        raise ValueError("Dataset is empty")

    p_vals = np.array([int(r["p"]) for r in rows], dtype=int)
    killed = np.array([1.0 if r["killed_Q0"] == "1" else 0.0 for r in rows], dtype=float)

    harm_cum = np.column_stack([np.array([float(r[f"ap_harm_delta_{Q}"]) for r in rows], dtype=float) for Q in bin_Qs])
    count_cum = np.column_stack([np.array([float(r[f"ap_count_delta_{Q}"]) for r in rows], dtype=float) for Q in bin_Qs])

    y_fit = np.array([float(r[f"survive_{args.fit_Q}"]) for r in rows], dtype=float)

    X_raw, feature_names = build_feature_matrix(
        killed_Q0=killed,
        harm_cum=harm_cum,
        count_cum=count_cum,
        bin_Qs=bin_Qs,
        use_count=use_count,
        use_diff=use_diff,
        use_shape=use_shape,
    )
    X_mean = X_raw.mean(axis=0)
    X_std = X_raw.std(axis=0) + 1e-9
    X = (X_raw - X_mean) / X_std

    w_vec, b = fit_logit(X, y_fit, lr=0.2, steps=1400)
    probs = sigmoid(X @ w_vec + b)
    x_grid, y_grid = isotonic_fit(probs, y_fit)
    probs_iso = isotonic_predict(x_grid, y_grid, probs)

    aucs: Dict[int, float] = {}
    briers: Dict[int, float] = {}
    loglosses: Dict[int, float] = {}
    spearman: Dict[int, float] = {}
    for Q in eval_Qs:
        yq = np.array([float(r[f"survive_{Q}"]) for r in rows], dtype=float)
        aucs[Q] = auc_score(yq, probs_iso)
        briers[Q] = brier_score(yq, probs_iso)
        loglosses[Q] = logloss(yq, probs_iso)
        spearman[Q] = spearman_corr(probs_iso, yq)

    # calibration plots
    for Q in eval_Qs:
        yq = np.array([float(r[f"survive_{Q}"]) for r in rows], dtype=float)
        save_reliability(out_dir / f"m29_calibration_Q{Q}.png", probs_iso, yq, f"Calibration vs Q={Q} (M29)")

    save_line_plot(
        out_dir / "m29_auc_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"AUC": [aucs[Q] for Q in eval_Qs]},
        "AUC by Q (M29 multiscale)",
        "Q",
        "AUC",
    )
    save_line_plot(
        out_dir / "m29_brier_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"Brier": [briers[Q] for Q in eval_Qs]},
        "Brier by Q (M29 multiscale)",
        "Q",
        "Brier",
    )
    save_line_plot(
        out_dir / "m29_logloss_by_Q.png",
        [float(Q) for Q in eval_Qs],
        {"LogLoss": [loglosses[Q] for Q in eval_Qs]},
        "LogLoss by Q (M29 multiscale)",
        "Q",
        "LogLoss",
    )
    save_line_plot(
        out_dir / "m29_rank_stability.png",
        [float(Q) for Q in eval_Qs],
        {"Spearman": [spearman[Q] for Q in eval_Qs]},
        "Rank transfer vs Q (Spearman, M29 multiscale)",
        "Q",
        "Spearman rho",
    )
    rank_csv = out_dir / "m29_rank_stability.csv"
    with rank_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "spearman_rho"])
        for Q in eval_Qs:
            writer.writerow([Q, f"{spearman[Q]:.6f}"])

    metrics_csv = out_dir / "m29_metrics_by_Q.csv"
    with metrics_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "AUC", "Brier", "LogLoss", "Spearman"])
        for Q in eval_Qs:
            writer.writerow([Q, f"{aucs[Q]:.6f}", f"{briers[Q]:.6f}", f"{loglosses[Q]:.6f}", f"{spearman[Q]:.6f}"])

    table_lines = [
        r"\begin{tabular}{lrrrr}\hline",
        r"Q & AUC & Brier & LogLoss & Spearman \\ \hline",
    ]
    for Q in eval_Qs:
        table_lines.append(
            f"{Q} & {aucs[Q]:.3f} & {briers[Q]:.4f} & {loglosses[Q]:.4f} & {spearman[Q]:.3f} \\\\"
        )
    table_lines.append(r"\hline\end{tabular}")
    (out_dir / "m29_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    preds_csv = out_dir / "m29_predictions.csv"
    with preds_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["p", "pred_survival"])
        for p, pr in zip(p_vals.tolist(), probs_iso.tolist()):
            writer.writerow([p, f"{pr:.12f}"])

    summary = {
        "dataset_csv": args.dataset_csv,
        "fit_Q": args.fit_Q,
        "eval_Qs": eval_Qs,
        "bin_Qs": bin_Qs,
        "model": args.model,
        "seed": args.seed,
        "flags": {"use_count": use_count, "use_diff": use_diff, "use_shape": use_shape},
        "feature_names": feature_names,
        "feature_mean": X_mean.tolist(),
        "feature_std": X_std.tolist(),
        "coefficients": {"w": w_vec.tolist(), "b": b},
        "isotonic": {"x_grid": x_grid.tolist(), "y_grid": y_grid.tolist()},
        "auc_by_Q": aucs,
        "brier_by_Q": briers,
        "logloss_by_Q": loglosses,
        "spearman_by_Q": spearman,
    }
    (out_dir / "m29_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    manifest = {
        "dataset_csv": args.dataset_csv,
        "fit_Q": args.fit_Q,
        "eval_Qs": eval_Qs,
        "bin_Qs": bin_Qs,
        "model": args.model,
        "seed": args.seed,
        "flags": {"use_count": use_count, "use_diff": use_diff, "use_shape": use_shape},
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m29_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK: wrote M29 multiscale artifacts to {out_dir}")


if __name__ == "__main__":
    main()
