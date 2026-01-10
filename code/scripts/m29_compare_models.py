#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-fit10M", type=str, required=True)
    p.add_argument("--baseline-fit20M", type=str, required=True)
    p.add_argument("--multiscale-fit10M", type=str, required=True)
    p.add_argument("--multiscale-fit20M", type=str, required=True)
    p.add_argument("--dataset-csv", type=str, required=True)
    p.add_argument("--Q-target-list", type=str, default="20000000,50000000")
    p.add_argument("--permutations", type=int, default=200)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def read_metrics_csv(path: Path) -> Dict[int, Dict[str, float]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: Dict[int, Dict[str, float]] = {}
    for r in rows:
        Q = int(r["Q"])
        out[Q] = {k: float(r[k]) for k in r.keys() if k != "Q"}
    return out


def read_spearman_csv(path: Path) -> Dict[int, float]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    out: Dict[int, float] = {}
    for r in rows:
        Q = int(r["Q"])
        out[Q] = float(r["spearman_rho"])
    return out


def save_compare_plot(path: Path, Qs: List[int], y_a: List[float], y_b: List[float],
                      label_a: str, label_b: str, title: str, ylabel: str) -> None:
    import matplotlib.pyplot as plt

    xs = [float(Q) for Q in Qs]
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(xs, y_a, marker="o", label=label_a)
    ax.plot(xs, y_b, marker="o", label=label_b)
    ax.set_title(title)
    ax.set_xlabel("Q")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_sanity_plot(path: Path, perm_vals: Dict[int, Dict[str, List[float]]]) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    ax = axes[0]
    for Q, d in perm_vals.items():
        ax.plot(range(len(d["auc"])), d["auc"], marker="o", linewidth=1, label=f"Q={Q}")
    ax.axhline(0.5, linestyle="--", color="#888888", linewidth=1)
    ax.set_title("Permutation sanity: AUC")
    ax.set_xlabel("permutation index")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7, loc="best")

    ax = axes[1]
    for Q, d in perm_vals.items():
        ax.plot(range(len(d["spearman"])), d["spearman"], marker="o", linewidth=1, label=f"Q={Q}")
    ax.axhline(0.0, linestyle="--", color="#888888", linewidth=1)
    ax.set_title("Permutation sanity: Spearman")
    ax.set_xlabel("permutation index")
    ax.set_ylabel("Spearman rho")
    ax.legend(fontsize=7, loc="best")

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    Q_targets = parse_int_list(args.Q_target_list)
    if not Q_targets:
        raise ValueError("Q-target-list is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base10 = Path(args.baseline_fit10M)
    base20 = Path(args.baseline_fit20M)
    multi10 = Path(args.multiscale_fit10M)
    multi20 = Path(args.multiscale_fit20M)

    base10_metrics = read_metrics_csv(base10 / "m27_metrics_by_Q.csv")
    base20_metrics = read_metrics_csv(base20 / "m27_metrics_by_Q.csv")
    base10_s = read_spearman_csv(base10 / "m27_rank_stability.csv")
    base20_s = read_spearman_csv(base20 / "m27_rank_stability.csv")

    multi10_metrics = read_metrics_csv(multi10 / "m29_metrics_by_Q.csv")
    multi20_metrics = read_metrics_csv(multi20 / "m29_metrics_by_Q.csv")
    multi10_s = read_spearman_csv(multi10 / "m29_rank_stability.csv")
    multi20_s = read_spearman_csv(multi20 / "m29_rank_stability.csv")

    all_Qs = sorted(
        set(base10_metrics.keys())
        | set(base20_metrics.keys())
        | set(multi10_metrics.keys())
        | set(multi20_metrics.keys())
    )

    # compare metrics by Q (all 4 models)
    compare_csv = out_dir / "m29_compare_metrics_by_Q.csv"
    with compare_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "model", "AUC", "Brier", "LogLoss", "Spearman"])
        for Q in all_Qs:
            for name, m, s in [
                ("baseline_fit10M", base10_metrics, base10_s),
                ("baseline_fit20M", base20_metrics, base20_s),
                ("multiscale_fit10M", multi10_metrics, multi10_s),
                ("multiscale_fit20M", multi20_metrics, multi20_s),
            ]:
                row = m.get(Q)
                if row is None:
                    continue
                writer.writerow(
                    [
                        Q,
                        name,
                        f"{row.get('AUC', float('nan')):.6f}",
                        f"{row.get('Brier', float('nan')):.6f}",
                        f"{row.get('LogLoss', float('nan')):.6f}",
                        f"{s.get(Q, float('nan')):.6f}",
                    ]
                )

    # compare plots for fit10M (baseline vs multiscale)
    auc_base = [float(base10_metrics[Q]["AUC"]) for Q in all_Qs if Q in base10_metrics]
    auc_multi = [float(multi10_metrics[Q]["AUC"]) for Q in all_Qs if Q in multi10_metrics]
    Qs_fit10 = [Q for Q in all_Qs if Q in base10_metrics and Q in multi10_metrics]
    save_compare_plot(
        out_dir / "m29_compare_auc_by_Q.png",
        Qs_fit10,
        [float(base10_metrics[Q]["AUC"]) for Q in Qs_fit10],
        [float(multi10_metrics[Q]["AUC"]) for Q in Qs_fit10],
        "baseline (fit@10M)",
        "multiscale (fit@10M)",
        "AUC vs Q (fit@10M): baseline vs multiscale",
        "AUC",
    )
    save_compare_plot(
        out_dir / "m29_compare_rank_stability.png",
        Qs_fit10,
        [float(base10_s.get(Q, float('nan'))) for Q in Qs_fit10],
        [float(multi10_s.get(Q, float('nan'))) for Q in Qs_fit10],
        "baseline (fit@10M)",
        "multiscale (fit@10M)",
        "Rank transfer vs Q (fit@10M): baseline vs multiscale",
        "Spearman rho",
    )

    # compare table (Q=20M/50M): baseline vs multiscale + extrapolation gaps (fit20 - fit10)
    table_lines = [
        r"\begin{tabular}{llrrrrr}\hline",
        r"Metric & Q & Base@10M & Multi@10M & $\Delta$(Multi-Base) & Gap(Base) & Gap(Multi) \\ \hline",
    ]
    for metric in ["AUC", "Spearman"]:
        for Q in Q_targets:
            if metric == "AUC":
                b10 = float(base10_metrics.get(Q, {}).get("AUC", float("nan")))
                b20 = float(base20_metrics.get(Q, {}).get("AUC", float("nan")))
                m10 = float(multi10_metrics.get(Q, {}).get("AUC", float("nan")))
                m20 = float(multi20_metrics.get(Q, {}).get("AUC", float("nan")))
            else:
                b10 = float(base10_s.get(Q, float("nan")))
                b20 = float(base20_s.get(Q, float("nan")))
                m10 = float(multi10_s.get(Q, float("nan")))
                m20 = float(multi20_s.get(Q, float("nan")))
            delta = m10 - b10
            gap_b = b20 - b10
            gap_m = m20 - m10
            table_lines.append(
                f"{metric} & {Q} & {b10:.3f} & {m10:.3f} & {delta:+.3f} & {gap_b:+.3f} & {gap_m:+.3f} \\\\"
            )
    table_lines.append(r"\hline\end{tabular}")
    (out_dir / "m29_compare_table.tex").write_text("\n".join(table_lines), encoding="utf-8")

    # permutation sanity (shuffle multiscale_fit10M probs)
    pred_path = multi10 / "m29_predictions.csv"
    pred_rows = list(csv.DictReader(pred_path.open(encoding="utf-8")))
    pred_by_p = {int(r["p"]): float(r["pred_survival"]) for r in pred_rows}

    data_rows = list(csv.DictReader(Path(args.dataset_csv).open(encoding="utf-8")))
    p_data = [int(r["p"]) for r in data_rows]
    missing = [p for p in p_data if p not in pred_by_p]
    if missing:
        raise ValueError(f"Missing predictions for {len(missing)} p values")

    scores = np.array([pred_by_p[p] for p in p_data], dtype=float)
    y_by_Q = {Q: np.array([float(r[f"survive_{Q}"]) for r in data_rows], dtype=float) for Q in Q_targets}

    rng = np.random.default_rng(args.seed)
    perm_vals: Dict[int, Dict[str, List[float]]] = {Q: {"auc": [], "spearman": []} for Q in Q_targets}
    for _ in range(int(args.permutations)):
        s_perm = rng.permutation(scores)
        for Q in Q_targets:
            yq = y_by_Q[Q]
            perm_vals[Q]["auc"].append(auc_score(yq, s_perm))
            perm_vals[Q]["spearman"].append(spearman_corr(s_perm, yq))

    sanity_csv = out_dir / "m29_sanity_permutation.csv"
    with sanity_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "metric", "n_perm", "mean", "ci_low", "ci_high"])
        for Q in Q_targets:
            for metric in ["auc", "spearman"]:
                vals = np.array(perm_vals[Q][metric], dtype=float)
                writer.writerow(
                    [
                        Q,
                        metric,
                        len(vals),
                        f"{float(np.mean(vals)):.6f}",
                        f"{float(np.quantile(vals, 0.025)):.6f}",
                        f"{float(np.quantile(vals, 0.975)):.6f}",
                    ]
                )
    save_sanity_plot(out_dir / "m29_sanity_plot.png", perm_vals)

    # compare manifest
    manifest = {
        "inputs": {
            "baseline_fit10M": str(base10),
            "baseline_fit20M": str(base20),
            "multiscale_fit10M": str(multi10),
            "multiscale_fit20M": str(multi20),
            "dataset_csv": args.dataset_csv,
        },
        "params": {"Q_targets": Q_targets, "permutations": int(args.permutations), "seed": int(args.seed)},
        "files": {},
    }
    for path in sorted(out_dir.glob("*")):
        if path.is_file():
            manifest["files"][path.name] = sha256_file(path)
    (out_dir / "m29_compare_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"OK: wrote M29 compare artifacts to {out_dir}")


if __name__ == "__main__":
    main()

