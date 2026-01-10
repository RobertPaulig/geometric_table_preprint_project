#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", type=str, default="occupancy", choices=["occupancy"])
    p.add_argument("--n-start", type=int, required=True)
    p.add_argument("--frames", type=int, required=True)
    p.add_argument("--n-step", type=int, required=True)
    p.add_argument("--window-rows", type=int, required=True)
    p.add_argument("--k-max", type=int, required=True)
    p.add_argument("--dt", type=int, default=5)
    p.add_argument("--method", type=str, default="phase_corr", choices=["phase_corr"])
    p.add_argument("--sanity", type=str, default="none", choices=["none", "permute_cols", "permute_rows"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out-dir", type=str, required=True)
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


def write_manifest(out_dir: Path, params: Dict[str, object]) -> Dict[str, object]:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != "m32_manifest.json"])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / "m32_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def occupancy_frame(n_top: int, H: int, K: int) -> np.ndarray:
    frame = np.zeros((H, K), dtype=np.uint8)
    n_end = n_top + H
    for k in range(1, K + 1):
        first = ((n_top + k - 1) // k) * k
        for n in range(first, n_end, k):
            frame[n - n_top, k - 1] = 1
    return frame


def normalize_frame(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float32, copy=False)
    mu = float(xf.mean())
    sigma = float(xf.std())
    if not np.isfinite(sigma) or sigma <= 1e-8:
        return xf - mu
    return (xf - mu) / sigma


def phase_corr_shift(a: np.ndarray, b: np.ndarray) -> Tuple[int, int, float]:
    fa = np.fft.rfft2(a)
    fb = np.fft.rfft2(b)
    cps = fa * np.conj(fb)
    denom = np.abs(cps)
    denom[denom == 0] = 1.0
    cps /= denom
    corr = np.fft.irfft2(cps, s=a.shape)
    corr_abs = np.abs(corr)
    peak_idx = int(np.argmax(corr_abs))
    peak_y, peak_x = np.unravel_index(peak_idx, corr_abs.shape)
    peak_val = float(corr_abs[peak_y, peak_x])

    H, W = corr_abs.shape
    dy = int(peak_y)
    dx = int(peak_x)
    if dy > H // 2:
        dy -= H
    if dx > W // 2:
        dx -= W
    return dy, dx, peak_val


def maybe_permute(frame: np.ndarray, sanity: str, rng: np.random.Generator) -> np.ndarray:
    if sanity == "none":
        return frame
    if sanity == "permute_cols":
        perm = rng.permutation(frame.shape[1])
        return frame[:, perm]
    if sanity == "permute_rows":
        perm = rng.permutation(frame.shape[0])
        return frame[perm, :]
    raise ValueError(f"unknown sanity: {sanity}")


@dataclass(frozen=True)
class FlowRow:
    t: int
    n_top: int
    dt: int
    dy: int
    dx: int
    dy_residual: int
    dx_residual: int
    confidence: float
    q_eff: Optional[float]


def compute_flow(
    *,
    n_start: int,
    frames: int,
    n_step: int,
    H: int,
    K: int,
    dt: int,
    sanity: str,
    seed: int,
) -> List[FlowRow]:
    rng = np.random.default_rng(seed)

    prepared: List[np.ndarray] = []
    n_tops: List[int] = []
    for t in range(frames):
        n_top = n_start + t * n_step
        f = occupancy_frame(n_top, H, K)
        f = maybe_permute(f, sanity, rng)
        prepared.append(normalize_frame(f))
        n_tops.append(n_top)

    rows: List[FlowRow] = []
    expected_dy = -(dt * n_step)
    delta_n = dt * n_step

    for t in range(frames - dt):
        a = prepared[t]
        b = prepared[t + dt]
        dy, dx, conf = phase_corr_shift(a, b)
        dy_res = int(dy - expected_dy)
        dx_res = int(dx)
        q_eff: Optional[float] = None
        if dx_res > 0:
            q_eff = float(delta_n) / float(dx_res)
        rows.append(
            FlowRow(
                t=t,
                n_top=n_tops[t],
                dt=dt,
                dy=dy,
                dx=dx,
                dy_residual=dy_res,
                dx_residual=dx_res,
                confidence=conf,
                q_eff=q_eff,
            )
        )
    return rows


def write_flow_csv(path: Path, rows: List[FlowRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t", "n_top", "dt", "dy", "dx", "dy_residual", "dx_residual", "confidence", "q_eff"])
        for r in rows:
            w.writerow(
                [
                    r.t,
                    r.n_top,
                    r.dt,
                    r.dy,
                    r.dx,
                    r.dy_residual,
                    r.dx_residual,
                    f"{r.confidence:.8g}",
                    "" if r.q_eff is None else f"{r.q_eff:.8g}",
                ]
            )


def percentile(xs: List[float], p: float) -> float:
    if not xs:
        return float("nan")
    return float(np.percentile(np.array(xs, dtype=float), p))


def top_peaks_rounded(xs: List[float], *, max_q: int = 64, topk: int = 8) -> List[Dict[str, object]]:
    if not xs:
        return []
    counts: Dict[int, int] = {}
    for v in xs:
        if not np.isfinite(v):
            continue
        q = int(round(v))
        if q <= 0 or q > max_q:
            continue
        counts[q] = counts.get(q, 0) + 1
    items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [{"q_round": int(q), "count": int(c)} for q, c in items[:topk]]


def make_plots(out_dir: Path, rows: List[FlowRow], *, title_suffix: str = "") -> None:
    import matplotlib.pyplot as plt

    ts = [r.t for r in rows]
    dxr = [r.dx_residual for r in rows]
    conf = [r.confidence for r in rows]
    qeff = [r.q_eff for r in rows if r.q_eff is not None and np.isfinite(r.q_eff)]

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    ax.plot(ts, dxr, linewidth=0.8)
    ax.set_title(f"dx_residual vs t{title_suffix}")
    ax.set_xlabel("t")
    ax.set_ylabel("dx_residual (columns)")
    fig.tight_layout()
    fig.savefig(out_dir / "m32_dx_residual_by_t.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(qeff, bins=40, color="steelblue", alpha=0.85)
    ax.set_title(f"q_eff histogram{title_suffix}")
    ax.set_xlabel("q_eff ≈ (dt*n_step)/dx_residual")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "m32_qeff_hist.png", dpi=160)
    plt.close(fig)

    angles = []
    for r in rows:
        if r.dx == 0 and r.dy == 0:
            continue
        angles.append(math.degrees(math.atan2(float(r.dy), float(r.dx))))
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.hist(angles, bins=60, color="gray", alpha=0.8)
    ax.set_title(f"drift angle histogram (atan2(dy,dx), degrees){title_suffix}")
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_dir / "m32_angle_hist.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    ax.plot(ts, conf, linewidth=0.8, color="#d62728")
    ax.set_title(f"phase-corr confidence vs t{title_suffix}")
    ax.set_xlabel("t")
    ax.set_ylabel("confidence (peak)")
    fig.tight_layout()
    fig.savefig(out_dir / "m32_confidence_by_t.png", dpi=160)
    plt.close(fig)


def write_table_tex(path: Path, summary: Dict[str, object], sanity_summary: Optional[Dict[str, object]]) -> None:
    dx_mu = summary["dx_residual_mean"]
    dx_sd = summary["dx_residual_std"]
    q_med = summary["q_eff_p50"]
    q_p90 = summary["q_eff_p90"]
    dx_mu_s = None
    q_med_s = None
    if sanity_summary is not None:
        dx_mu_s = sanity_summary["dx_residual_mean"]
        q_med_s = sanity_summary["q_eff_p50"]

    lines = []
    lines.append("\\begin{tabular}{lrrrr}\\hline")
    lines.append("run & mean dx$_{res}$ & std dx$_{res}$ & median $q_{eff}$ & $q_{eff}$ p90 \\\\ \\hline")
    lines.append(f"real & {dx_mu:.3g} & {dx_sd:.3g} & {q_med:.3g} & {q_p90:.3g} \\\\")
    if dx_mu_s is not None and q_med_s is not None:
        lines.append(f"permute\\_cols & {dx_mu_s:.3g} & {sanity_summary['dx_residual_std']:.3g} & {q_med_s:.3g} & {sanity_summary['q_eff_p90']:.3g} \\\\")
    lines.append("\\hline\\end{tabular}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_flow_csv(path: Path) -> List[FlowRow]:
    rows: List[FlowRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            q = row.get("q_eff", "")
            q_eff = float(q) if q not in ("", None) else None
            rows.append(
                FlowRow(
                    t=int(row["t"]),
                    n_top=int(row["n_top"]),
                    dt=int(row["dt"]),
                    dy=int(row["dy"]),
                    dx=int(row["dx"]),
                    dy_residual=int(row["dy_residual"]),
                    dx_residual=int(row["dx_residual"]),
                    confidence=float(row["confidence"]),
                    q_eff=q_eff,
                )
            )
    return rows


def summarize(rows: List[FlowRow]) -> Dict[str, object]:
    dxr = [float(r.dx_residual) for r in rows]
    qeff = [float(r.q_eff) for r in rows if r.q_eff is not None and np.isfinite(r.q_eff)]
    pos_frac = float(sum(1 for r in rows if r.dx_residual > 0) / len(rows)) if rows else float("nan")
    return {
        "n_pairs": len(rows),
        "dx_residual_mean": float(np.mean(dxr)) if dxr else float("nan"),
        "dx_residual_std": float(np.std(dxr)) if dxr else float("nan"),
        "dx_residual_pos_frac": pos_frac,
        "q_eff_p10": percentile(qeff, 10),
        "q_eff_p50": percentile(qeff, 50),
        "q_eff_p90": percentile(qeff, 90),
        "q_eff_peaks_rounded": top_peaks_rounded(qeff, max_q=64, topk=10),
    }


def make_sanity_compare(parent_out_dir: Path, real_rows: List[FlowRow], sanity_dir: Path) -> None:
    import matplotlib.pyplot as plt

    sanity_csv = sanity_dir / "m32_flow_vectors.csv"
    if not sanity_csv.exists():
        return

    sanity_rows = read_flow_csv(sanity_csv)
    ts_r = [r.t for r in real_rows]
    dx_r = [r.dx_residual for r in real_rows]
    ts_s = [r.t for r in sanity_rows]
    dx_s = [r.dx_residual for r in sanity_rows]

    fig, ax = plt.subplots(figsize=(8.6, 3.2))
    ax.plot(ts_r, dx_r, linewidth=0.9, label="real", color="#1f77b4")
    ax.plot(ts_s, dx_s, linewidth=0.9, label="permute_cols", color="#d62728", alpha=0.85)
    ax.set_title("sanity compare: dx_residual vs t (real vs permute_cols)")
    ax.set_xlabel("t")
    ax.set_ylabel("dx_residual")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(parent_out_dir / "m32_sanity_compare.png", dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    rows = compute_flow(
        n_start=args.n_start,
        frames=args.frames,
        n_step=args.n_step,
        H=args.window_rows,
        K=args.k_max,
        dt=args.dt,
        sanity=args.sanity,
        seed=args.seed,
    )
    runtime_s = float(time.time() - started)

    flow_csv = out_dir / "m32_flow_vectors.csv"
    write_flow_csv(flow_csv, rows)
    make_plots(out_dir, rows, title_suffix=f" ({args.sanity})" if args.sanity != "none" else "")

    summary = summarize(rows)
    summary_json = {
        "params": {
            "mode": args.mode,
            "n_start": args.n_start,
            "frames": args.frames,
            "n_step": args.n_step,
            "window_rows": args.window_rows,
            "k_max": args.k_max,
            "dt": args.dt,
            "method": args.method,
            "sanity": args.sanity,
            "seed": args.seed,
        },
        "runtime_seconds": runtime_s,
        **summary,
    }
    (out_dir / "m32_flow_summary.json").write_text(json.dumps(summary_json, indent=2), encoding="utf-8")

    sanity_summary: Optional[Dict[str, object]] = None
    if args.sanity == "none":
        sanity_dir = out_dir / "sanity_permute_cols"
        sanity_json = sanity_dir / "m32_flow_summary.json"
        if sanity_json.exists():
            sanity_summary = json.loads(sanity_json.read_text(encoding="utf-8"))
        else:
            sanity_csv = sanity_dir / "m32_flow_vectors.csv"
            if sanity_csv.exists():
                sanity_summary = summarize(read_flow_csv(sanity_csv))
        write_table_tex(out_dir / "m32_table.tex", summary, sanity_summary)

        if sanity_dir.exists():
            make_sanity_compare(out_dir, rows, sanity_dir)
    elif args.sanity == "permute_cols":
        parent = out_dir.parent
        real_csv = parent / "m32_flow_vectors.csv"
        if real_csv.exists():
            make_sanity_compare(parent, read_flow_csv(real_csv), out_dir)

    params = summary_json["params"]
    params["runtime_seconds"] = runtime_s
    write_manifest(out_dir, params=params)
    print(f"OK: wrote M32 flow to {out_dir}")


if __name__ == "__main__":
    main()

