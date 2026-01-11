from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m35-summary-csv", type=str, default="out/wave_atlas/m35/m35_summary.csv")
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m36")
    return p.parse_args()


def git_sha() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, params: Dict[str, Any]) -> None:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != "m36_manifest.json"])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / "m36_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ffloat(x: str) -> float:
    x = (x or "").strip()
    if x.lower() in {"nan", ""}:
        return float("nan")
    return float(x)


@dataclass
class Row:
    run_id: str
    n_start: int
    K: int
    H: int
    weights: str
    smooth: int
    dt: int
    frames: int
    conf_min: float
    sanity: str
    mean_dx: float
    median_dx: float
    q_eff: float
    slope_kpeak: float
    track_r2: float
    valid_frac: float
    wave_strength: float

    def visibility_score(self) -> float:
        ws = self.wave_strength if not math.isnan(self.wave_strength) else 0.0
        r2 = self.track_r2 if not math.isnan(self.track_r2) else 0.0
        vf = self.valid_frac if not math.isnan(self.valid_frac) else 0.0
        return float(ws) * math.sqrt(max(r2, 0.0)) * math.sqrt(max(vf, 0.0))


def load_m35_rows(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(
                Row(
                    run_id=r["run_id"],
                    n_start=int(r["n_start"]),
                    K=int(r["K"]),
                    H=int(r["H"]),
                    weights=r["weights"],
                    smooth=int(r["smooth"]),
                    dt=int(r["dt"]),
                    frames=int(r["frames"]),
                    conf_min=ffloat(r["conf_min"]),
                    sanity=r["sanity"],
                    mean_dx=ffloat(r["mean_dx"]),
                    median_dx=ffloat(r["median_dx"]),
                    q_eff=ffloat(r["q_eff"]),
                    slope_kpeak=ffloat(r["slope_kpeak"]),
                    track_r2=ffloat(r["track_r2"]),
                    valid_frac=ffloat(r["valid_frac"]),
                    wave_strength=ffloat(r["wave_strength"]),
                )
            )
    return rows


def linfit(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    x0 = x.astype(np.float64)
    y0 = y.astype(np.float64)
    mask = np.isfinite(x0) & np.isfinite(y0)
    x0 = x0[mask]
    y0 = y0[mask]
    if x0.size < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "r2": float("nan")}
    A = np.vstack([x0, np.ones_like(x0)]).T
    slope, intercept = np.linalg.lstsq(A, y0, rcond=None)[0].tolist()
    y_hat = slope * x0 + intercept
    ss_res = float(np.sum((y0 - y_hat) ** 2))
    ss_tot = float(np.sum((y0 - float(np.mean(y0))) ** 2))
    r2 = float("nan") if ss_tot <= 0 else 1.0 - ss_res / ss_tot
    return {"slope": float(slope), "intercept": float(intercept), "r2": r2}


def write_csv(path: Path, rows: List[Dict[str, Any]], header: List[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in header})


def make_table_tex(out_path: Path, best: List[Row]) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r l r r r}")
    lines.append(r"\hline")
    lines.append(r"$n_{\mathrm{start}}$ & $K$ & weights & strength & score & mean\_dx \\")
    lines.append(r"\hline")
    for r in best:
        lines.append(
            f"{r.n_start} & {r.K} & {r.weights} & {r.wave_strength:.3g} & {r.visibility_score():.3g} & {r.mean_dx:.3g} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_best(out_dir: Path, best: List[Row], fit: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    xs = np.array([r.n_start for r in best], dtype=np.float64)
    ys = np.array([r.wave_strength for r in best], dtype=np.float64)
    xlog = np.log(xs)

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    ax.scatter(xs, ys, s=55, color="#00d4ff", edgecolor="black", linewidth=0.5, label="best per n_start")
    ax.set_xscale("log")
    ax.set_xlabel("n_start")
    ax.set_ylabel("wave_strength")
    ax.set_title("M36: best wave_strength vs n_start (log scale)")
    ax.grid(True, alpha=0.3)
    if math.isfinite(fit["slope"]) and math.isfinite(fit["intercept"]):
        xline = np.linspace(np.min(xlog), np.max(xlog), 200)
        yline = fit["slope"] * xline + fit["intercept"]
        ax.plot(np.exp(xline), yline, color="#ff6a00", linewidth=2.0, label=f"fit vs ln(n): R^2={fit['r2']:.2g}")
        ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "m36_wave_strength_best_vs_nstart.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=140)
    ks = np.array([r.K for r in best], dtype=np.float64)
    ax.plot(xs, ks, marker="o", color="#00ff6a")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n_start")
    ax.set_ylabel("best K")
    ax.set_title("M36: best window size K vs n_start")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "m36_best_K_vs_nstart.png")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    m35_summary = Path(args.m35_summary_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "m36_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    rows = load_m35_rows(m35_summary)
    rows_none = [r for r in rows if r.sanity == "none"]
    if not rows_none:
        raise RuntimeError(f"No sanity=none rows in {m35_summary}")

    best_by_n: Dict[int, Row] = {}
    for r in rows_none:
        cur = best_by_n.get(r.n_start)
        if cur is None or r.visibility_score() > cur.visibility_score():
            best_by_n[r.n_start] = r

    best = [best_by_n[k] for k in sorted(best_by_n.keys())]

    best_rows_csv: List[Dict[str, Any]] = []
    for r in best:
        d = asdict(r)
        d["visibility_score"] = r.visibility_score()
        best_rows_csv.append(d)

    header = list(best_rows_csv[0].keys())
    write_csv(out_dir / "m36_best_config_by_nstart.csv", best_rows_csv, header)

    x = np.log(np.array([r.n_start for r in best], dtype=np.float64))
    y = np.array([r.wave_strength for r in best], dtype=np.float64)
    fit = linfit(x, y)
    fit_json = {
        "source": str(m35_summary).replace("\\", "/"),
        "n_points": int(len(best)),
        "model": "wave_strength ~ a + b * ln(n_start)",
        "fit": fit,
        "points": [{"n_start": r.n_start, "wave_strength": r.wave_strength} for r in best],
    }
    (out_dir / "m36_scaling_fit.json").write_text(json.dumps(fit_json, indent=2), encoding="utf-8")

    make_table_tex(out_dir / "m36_table.tex", best)
    plot_best(plots_dir, best, fit)

    params = {"m35_summary_csv": str(m35_summary).replace("\\", "/")}
    write_manifest(out_dir, params=params)
    print(f"OK: wrote M36 analysis to {out_dir}")


if __name__ == "__main__":
    main()

