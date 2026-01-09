#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--fit10-summary", type=str, required=True)
    p.add_argument("--fit20-summary", type=str, required=True)
    p.add_argument("--Q-list", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    return p.parse_args()


def parse_int_list(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def load_metrics(path: Path) -> Dict[int, Dict[str, float]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    aucs = {int(k): float(v) for k, v in data["auc_by_Q"].items()}
    briers = {int(k): float(v) for k, v in data["brier_by_Q"].items()}
    loglosses = {int(k): float(v) for k, v in data["logloss_by_Q"].items()}
    metrics = {}
    for Q in aucs:
        metrics[Q] = {"AUC": aucs[Q], "Brier": briers[Q], "LogLoss": loglosses[Q]}
    return metrics


def main() -> None:
    args = parse_args()
    qs = parse_int_list(args.Q_list)
    if not qs:
        raise ValueError("Q-list is empty")

    fit10 = load_metrics(Path(args.fit10_summary))
    fit20 = load_metrics(Path(args.fit20_summary))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for Q in qs:
        if Q not in fit10 or Q not in fit20:
            raise ValueError(f"Q={Q} not found in summaries")
        row = {
            "Q": Q,
            "AUC_fit10M": fit10[Q]["AUC"],
            "AUC_fit20M": fit20[Q]["AUC"],
            "AUC_gap": fit20[Q]["AUC"] - fit10[Q]["AUC"],
            "Brier_fit10M": fit10[Q]["Brier"],
            "Brier_fit20M": fit20[Q]["Brier"],
            "Brier_gap": fit20[Q]["Brier"] - fit10[Q]["Brier"],
            "LogLoss_fit10M": fit10[Q]["LogLoss"],
            "LogLoss_fit20M": fit20[Q]["LogLoss"],
            "LogLoss_gap": fit20[Q]["LogLoss"] - fit10[Q]["LogLoss"],
        }
        rows.append(row)

    csv_path = out_dir / "m27_extrapolation_gap.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "Q",
                "AUC_fit10M",
                "AUC_fit20M",
                "AUC_gap",
                "Brier_fit10M",
                "Brier_fit20M",
                "Brier_gap",
                "LogLoss_fit10M",
                "LogLoss_fit20M",
                "LogLoss_gap",
            ],
        )
        w.writeheader()
        for row in rows:
            w.writerow(row)

    json_path = out_dir / "m27_extrapolation_gap.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"OK: wrote extrapolation gap to {csv_path}")


if __name__ == "__main__":
    main()
