from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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


def write_manifest(out_dir: Path, params: Dict[str, Any], *, name: str) -> None:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != name])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def to_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def q_from_theta(theta_deg: float) -> float:
    # Our theta is measured from the x-axis; vertical is 90°.
    # For a ray n=q*k in (k,n) coordinates: slope dn/dk = q, hence theta = arctan(q) and q = tan(theta).
    return float(math.tan(math.radians(theta_deg)))


def theta_from_q(q: float) -> float:
    if not math.isfinite(q):
        return float("nan")
    return float(math.degrees(math.atan(q)))


def mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def robust_summary(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"n": 0, "median": float("nan"), "mad": float("nan"), "p10": float("nan"), "p90": float("nan")}
    return {
        "n": int(x.size),
        "median": float(np.median(x)),
        "mad": float(mad(x)),
        "p10": float(np.percentile(x, 10)),
        "p90": float(np.percentile(x, 90)),
    }


def build_table_tex(out_path: Path, best_rows: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r r r r r r}")
    lines.append(r"\hline")
    lines.append(
        r"$n_{\mathrm{start}}$ & $q_0$ & $dt$ & $q_{\mathrm{dx}}$ & $q_{\theta}$ (CI) & $\Delta q$ & $\Delta\theta$ \\"
    )
    lines.append(r"\hline")
    for r in best_rows:
        n0 = to_int(r.get("n_start", "0"))
        q0 = to_int(r.get("q0_best", "0"))
        dt = to_int(r.get("dt", "0"))
        q_dx = to_float(r.get("q_eff", "nan"))
        th = to_float(r.get("theta_subdeg", "nan"))
        th_lo = to_float(r.get("theta_ci_lo", "nan"))
        th_hi = to_float(r.get("theta_ci_hi", "nan"))
        q_th = q_from_theta(th) if math.isfinite(th) else float("nan")
        q_lo = q_from_theta(th_lo) if math.isfinite(th_lo) else float("nan")
        q_hi = q_from_theta(th_hi) if math.isfinite(th_hi) else float("nan")
        th_dx = theta_from_q(q_dx) if (math.isfinite(q_dx) and q_dx > 0) else float("nan")
        dtheta = th_dx - th if (math.isfinite(th_dx) and math.isfinite(th)) else float("nan")
        dq = q_dx - q_th if (math.isfinite(q_dx) and math.isfinite(q_th)) else float("nan")
        q_ci = f"{q_th:.2f} [{q_lo:.2f},{q_hi:.2f}]" if (math.isfinite(q_lo) and math.isfinite(q_hi)) else f"{q_th:.2f}"
        lines.append(f"{n0} & {q0} & {dt} & {q_dx:.2f} & {q_ci} & {dq:+.2f} & {dtheta:+.3f} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_q_vs_q(
    *,
    out_path: Path,
    rows: List[Dict[str, Any]],
    title: str,
    r2_min: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=150)

    colors = {
        1000000: "#4c78a8",
        50000000: "#f58518",
        200000000: "#54a24b",
    }

    for sanity, marker, alpha in [("none", "o", 0.8), ("permute_cols", "x", 0.45)]:
        for n0 in sorted({int(r["n_start"]) for r in rows if int(r["n_start"]) > 0}):
            xs: List[float] = []
            ys: List[float] = []
            for r in rows:
                if int(r["n_start"]) != n0:
                    continue
                if r.get("sanity") != sanity:
                    continue
                if sanity == "none" and float(r.get("track_r2", 0.0)) < float(r2_min):
                    continue
                q_th = float(r.get("q_from_theta", float("nan")))
                q_dx = float(r.get("q_eff", float("nan")))
                if not (math.isfinite(q_th) and math.isfinite(q_dx)):
                    continue
                xs.append(q_th)
                ys.append(q_dx)
            if not xs:
                continue
            color = colors.get(n0, "#888888")
            label = f"n_start={n0:g} ({sanity})" if sanity != "none" else f"n_start={n0:g}"
            ax.scatter(xs, ys, s=18, alpha=alpha, marker=marker, color=color, label=label)

    lo = 0.0
    hi = 45.0
    ax.plot([lo, hi], [lo, hi], color="#000000", linewidth=1.0, alpha=0.35, linestyle="--", label="y=x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$q_{\theta}=\tan(\theta_{\mathrm{sub}})$")
    ax.set_ylabel(r"$q_{\mathrm{dx}}$ (from $\Delta k$)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_delta_q_by_nstart(
    *,
    out_path: Path,
    rows: List[Dict[str, Any]],
    title: str,
    r2_min: float,
) -> None:
    import matplotlib.pyplot as plt

    n_starts = sorted({int(r["n_start"]) for r in rows if int(r["n_start"]) > 0})
    data: List[np.ndarray] = []
    labels: List[str] = []
    for n0 in n_starts:
        vals: List[float] = []
        for r in rows:
            if int(r["n_start"]) != n0:
                continue
            if r.get("sanity") != "none":
                continue
            if float(r.get("track_r2", 0.0)) < float(r2_min):
                continue
            dq = float(r.get("delta_q", float("nan")))
            if math.isfinite(dq):
                vals.append(dq)
        if not vals:
            continue
        data.append(np.array(vals, dtype=np.float64))
        labels.append(f"{n0:g}")

    fig, ax = plt.subplots(figsize=(6.6, 3.6), dpi=150)
    ax.boxplot(data, labels=labels, showfliers=False)
    ax.axhline(0.0, color="#000000", linewidth=1.0, alpha=0.35, linestyle="--")
    ax.set_xlabel("n_start")
    ax.set_ylabel(r"$\Delta q = q_{\mathrm{dx}} - q_{\theta}$")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="M42: consistency check between q-from-angle and q-from-dx (M41).")
    p.add_argument("--m41-dir", type=str, default="out/wave_atlas/m41")
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m42")
    p.add_argument("--r2-min", type=float, default=0.95)
    args = p.parse_args()

    t0 = time.time()

    m41_dir = Path(args.m41_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    sweep_real = m41_dir / "m41_q0_sweep.csv"
    best_real = m41_dir / "m41_best_by_nstart.csv"
    if not sweep_real.exists():
        raise FileNotFoundError(f"Missing required file: {sweep_real}")
    if not best_real.exists():
        raise FileNotFoundError(f"Missing required file: {best_real}")

    rows: List[Dict[str, Any]] = []

    def ingest(path: Path) -> None:
        for r in load_csv_dicts(path):
            n0 = to_int(r.get("n_start", "0"))
            q0 = to_int(r.get("q0", r.get("q0_best", "0")))
            dt = to_int(r.get("dt", "0"))
            q_eff = to_float(r.get("q_eff", "nan"))
            track_r2 = to_float(r.get("track_r2", "nan"))
            th = to_float(r.get("theta_subdeg", r.get("theta_star", "nan")))
            sanity = r.get("sanity", "")

            q_th = q_from_theta(th) if math.isfinite(th) else float("nan")
            th_dx = theta_from_q(q_eff) if (math.isfinite(q_eff) and q_eff > 0) else float("nan")
            dtheta = th_dx - th if (math.isfinite(th_dx) and math.isfinite(th)) else float("nan")
            dq = q_eff - q_th if (math.isfinite(q_eff) and math.isfinite(q_th)) else float("nan")

            rows.append(
                {
                    "n_start": n0,
                    "q0": q0,
                    "dt": dt,
                    "q_eff": q_eff,
                    "theta_subdeg": th,
                    "q_from_theta": q_th,
                    "theta_from_qeff": th_dx,
                    "delta_theta": dtheta,
                    "delta_q": dq,
                    "track_r2": track_r2,
                    "peakiness": to_float(r.get("peakiness", "nan")),
                    "sanity": sanity,
                }
            )

    ingest(sweep_real)
    sweep_sanity = m41_dir / "sanity_permute_cols" / "m41_q0_sweep.csv"
    if sweep_sanity.exists():
        ingest(sweep_sanity)

    # Write per-row consistency (includes sanity points if present)
    out_csv = out_dir / "m42_consistency.csv"
    fieldnames = [
        "n_start",
        "q0",
        "dt",
        "q_eff",
        "theta_subdeg",
        "q_from_theta",
        "theta_from_qeff",
        "delta_theta",
        "delta_q",
        "track_r2",
        "peakiness",
        "sanity",
    ]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    # Summary stats on real, stable rows
    r2_min = float(args.r2_min)
    summaries: Dict[str, Any] = {}
    for n0 in sorted({int(r["n_start"]) for r in rows if int(r["n_start"]) > 0}):
        xs = np.array(
            [
                float(r["delta_q"])
                for r in rows
                if int(r["n_start"]) == n0 and r.get("sanity") == "none" and float(r.get("track_r2", 0.0)) >= r2_min and math.isfinite(float(r["delta_q"]))
            ],
            dtype=np.float64,
        )
        ts = np.array(
            [
                float(r["delta_theta"])
                for r in rows
                if int(r["n_start"]) == n0
                and r.get("sanity") == "none"
                and float(r.get("track_r2", 0.0)) >= r2_min
                and math.isfinite(float(r["delta_theta"]))
            ],
            dtype=np.float64,
        )
        summaries[str(n0)] = {
            "delta_q": robust_summary(xs),
            "delta_theta_deg": robust_summary(ts),
        }

    # Build table (from M41 best rows)
    best_rows = load_csv_dicts(best_real)
    build_table_tex(out_dir / "m42_table.tex", best_rows)

    plot_q_vs_q(
        out_path=out_dir / "m42_q_from_theta_vs_qeff.png",
        rows=rows,
        title=f"M42: q(theta) vs q(dx) (filter: R^2 >= {r2_min:.2f} on real)",
        r2_min=r2_min,
    )
    plot_delta_q_by_nstart(
        out_path=out_dir / "m42_delta_q_by_nstart.png",
        rows=rows,
        title=f"M42: delta_q by n_start (R^2 >= {r2_min:.2f} on real)",
        r2_min=r2_min,
    )

    out_summary = out_dir / "m42_summary.json"
    out_summary.write_text(
        json.dumps(
            {
                "params": {
                    "m41_dir": str(m41_dir).replace("\\", "/"),
                    "r2_min": r2_min,
                    "note": "q_from_theta uses q = tan(theta_subdeg); theta is measured from x-axis (vertical=90deg).",
                },
                "git_sha": git_sha(),
                "runtime_s": float(time.time() - t0),
                "per_n_start": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    write_manifest(out_dir, params={"m41_dir": str(m41_dir).replace("\\", "/"), "r2_min": r2_min}, name="m42_manifest.json")


if __name__ == "__main__":
    main()

