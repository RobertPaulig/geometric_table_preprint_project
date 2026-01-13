from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
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


def parse_int_list(s: str) -> List[int]:
    out: List[int] = []
    for part in s.split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    if not out:
        raise ValueError("Expected a non-empty comma-separated int list")
    return out


def n_start_list_from_args(args: argparse.Namespace) -> List[int]:
    if args.n_start is not None and args.n_start_list is not None:
        raise ValueError("Use either --n-start or --n-start-list, not both")
    if args.n_start is not None:
        return [int(args.n_start)]
    if args.n_start_list is not None:
        return parse_int_list(args.n_start_list)
    raise ValueError("Expected --n-start or --n-start-list")


def run_cmd(cmd: List[str], *, env: Dict[str, str]) -> None:
    subprocess.check_call(cmd, env=env)


def python_cmd(script: str, args: List[str]) -> List[str]:
    return [sys.executable, script, *args]


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_int(x: str) -> int:
    try:
        return int(float(x))
    except Exception:
        return 0


def to_float(x: str) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def q_from_theta(theta_deg: float) -> float:
    return float(math.tan(math.radians(theta_deg)))


def k_window_for_frame(n_top: int, H: int, W: int, q0: int) -> int:
    n_mid = n_top + (H // 2)
    k_c = int(n_mid // q0)
    k_start = int(k_c - (W // 2))
    if k_start < 1:
        k_start = 1
    return k_start


def build_values_invq_profile(*, n_top: int, H: int, k_start: int, W: int) -> np.ndarray:
    prof = np.zeros(W, dtype=np.float32)
    n_end = n_top + H
    for j in range(W):
        k = k_start + j
        first = ((n_top + k - 1) // k) * k
        if first >= n_end:
            continue
        acc = 0.0
        for n in range(first, n_end, k):
            q = n // k
            acc += 1.0 / float(q)
        prof[j] = float(acc)
    return prof


def rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    kernel = np.ones(int(w), dtype=np.float32) / float(w)
    return np.convolve(x.astype(np.float32), kernel, mode="same")


def mad(x: np.ndarray) -> float:
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def top_peak(profile: np.ndarray) -> Tuple[int, float]:
    if profile.size == 0:
        return -1, float("nan")
    i = int(np.argmax(profile))
    return i, float(profile[i])


def bootstrap_q_eff_ci(
    *,
    n_start: int,
    q0: int,
    dt: int,
    H: int,
    W: int,
    frames: int,
    n_step: int,
    smooth: int,
    conf_min: float,
    seed: int,
    bootstrap: int,
) -> Tuple[float, float, float]:
    # Recompute dx series exactly like M41 (peak0 tracking + z-based confidence),
    # then bootstrap the mean(dx) to obtain a CI for q_eff = dt/mean(dx).
    x_peaks0: List[int] = []
    z_series: List[float] = []
    k_starts: List[int] = []

    for t in range(frames):
        n_top = n_start + t * n_step
        k_start = k_window_for_frame(n_top, H, W, q0)
        k_starts.append(k_start)
        prof = build_values_invq_profile(n_top=n_top, H=H, k_start=k_start, W=W)
        prof_sm = rolling_mean(prof, smooth)
        x0, peak_val = top_peak(prof_sm)
        x_peaks0.append(int(x0))
        baseline = float(np.median(prof_sm))
        scale = mad(prof_sm)
        if scale <= 0:
            scale = float(np.std(prof_sm))
        z = float((peak_val - baseline) / (scale + 1e-9)) if math.isfinite(peak_val) else float("nan")
        z_series.append(z)

    dxs: List[float] = []
    confs: List[float] = []
    for t in range(dt, frames):
        x_now = x_peaks0[t]
        x_prev = x_peaks0[t - dt]
        z_now = z_series[t]
        z_prev = z_series[t - dt]
        if x_now >= 0 and x_prev >= 0 and math.isfinite(z_now) and math.isfinite(z_prev):
            k_now = k_starts[t] + int(x_now)
            k_prev = k_starts[t - dt] + int(x_prev)
            dxs.append(float(k_now - k_prev))
            confs.append(float(min(z_now, z_prev)))
        else:
            dxs.append(float("nan"))
            confs.append(float("nan"))

    dx_arr = np.array(dxs, dtype=np.float64)
    conf_arr = np.array(confs, dtype=np.float64)
    valid_mask = np.isfinite(dx_arr) & np.isfinite(conf_arr) & (conf_arr >= float(conf_min))
    dx_valid = dx_arr[valid_mask]
    if dx_valid.size < 3 or bootstrap <= 0:
        return float("nan"), float("nan"), float("nan")

    rng = np.random.default_rng(np.random.SeedSequence([seed, n_start, q0, dt, 43]))
    q_boot = np.zeros(int(bootstrap), dtype=np.float64)
    for b in range(int(bootstrap)):
        idx = rng.integers(0, dx_valid.size, size=dx_valid.size, dtype=np.int32)
        mean_dx = float(np.mean(dx_valid[idx]))
        q_boot[b] = float(dt) / mean_dx if mean_dx > 0 else float("nan")
    q_boot = q_boot[np.isfinite(q_boot)]
    if q_boot.size < max(10, int(bootstrap) // 10):
        return float("nan"), float("nan"), float("nan")
    mid = float(np.median(q_boot))
    lo = float(np.percentile(q_boot, 2.5))
    hi = float(np.percentile(q_boot, 97.5))
    return mid, lo, hi


def build_table_tex(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r r r r r r}")
    lines.append(r"\hline")
    lines.append(
        r"$n_{\mathrm{start}}$ & $q_0$ & $dt$ & $\theta_{\mathrm{sub}}$ (CI) & $q_{\theta}$ (CI) & $q_{\mathrm{dx}}$ (CI) & pass \\"
    )
    lines.append(r"\hline")
    for r in rows:
        n0 = int(r["n_start"])
        q0 = int(r["q0_best"])
        dt = int(r["dt_best"])
        th = float(r["theta_subdeg"])
        th_lo = float(r["theta_ci_lo"])
        th_hi = float(r["theta_ci_hi"])
        q_th = float(r["q_theta"])
        q_th_lo = float(r["q_theta_ci_lo"])
        q_th_hi = float(r["q_theta_ci_hi"])
        q_dx = float(r["q_eff"])
        q_dx_lo = float(r.get("q_eff_ci_lo", float("nan")))
        q_dx_hi = float(r.get("q_eff_ci_hi", float("nan")))
        ok = bool(r["pass_overall"])
        pass_s = r"\checkmark" if ok else r"\times"
        qdx_ci = f"{q_dx:.2f} [{q_dx_lo:.2f},{q_dx_hi:.2f}]" if math.isfinite(q_dx_lo) and math.isfinite(q_dx_hi) else f"{q_dx:.2f}"
        lines.append(
            f"{n0} & {q0} & {dt} & {th:.3f} [{th_lo:.3f},{th_hi:.3f}] & {q_th:.2f} [{q_th_lo:.2f},{q_th_hi:.2f}] & {qdx_ci} & {pass_s} \\\\"
        )
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_csv(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "n_start",
        "q0_best",
        "dt_best",
        "theta_subdeg",
        "theta_ci_lo",
        "theta_ci_hi",
        "q_theta",
        "q_theta_ci_lo",
        "q_theta_ci_hi",
        "q_eff",
        "q_eff_ci_lo",
        "q_eff_ci_hi",
        "track_r2_real",
        "peakiness_real",
        "track_r2_sanity",
        "peakiness_sanity",
        "pass_real",
        "pass_sanity",
        "pass_overall",
        "fail_reasons",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r.get(k, "") for k in fieldnames}
            if isinstance(row.get("fail_reasons"), list):
                row["fail_reasons"] = "; ".join(row["fail_reasons"])
            w.writerow(row)


def make_preview(
    *,
    out_path: Path,
    real_img_path: Path,
    sanity_img_path: Path,
    title: str,
    subtitle_real: str,
    subtitle_sanity: str,
) -> None:
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt

    real_img = mpimg.imread(real_img_path)
    sanity_img = mpimg.imread(sanity_img_path)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12.5, 5.4), dpi=140)
    axes[0].imshow(real_img)
    axes[0].set_title(f"real\\n{subtitle_real}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(sanity_img)
    axes[1].set_title(f"sanity (permute_cols)\\n{subtitle_sanity}", fontsize=10)
    axes[1].axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="M43: one-command wave instrument (M41 + M42 + overlay + PASS/FAIL).")
    p.add_argument("--n-start", type=int, default=None)
    p.add_argument("--n-start-list", type=str, default=None)
    p.add_argument("--q0-grid", type=str, default="6,40,1")
    p.add_argument("--H", type=int, default=512)
    p.add_argument("--W", type=int, default=1024)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--n-step", type=int, default=1)
    p.add_argument("--mode", type=str, default="values", choices=["values"])
    p.add_argument("--weights", type=str, default="invq", choices=["invq"])
    p.add_argument("--theta-range-deg", type=str, default="78.0,90.0")
    p.add_argument("--theta-step-deg", type=float, default=0.1)
    p.add_argument("--theta-refine-halfwidth-deg", type=float, default=0.6)
    p.add_argument("--theta-refine-step-deg", type=float, default=0.02)
    p.add_argument("--target-dx", type=float, default=0.8)
    p.add_argument("--dt-min", type=int, default=5)
    p.add_argument("--dt-max", type=int, default=80)
    p.add_argument("--smooth", type=int, default=9)
    p.add_argument("--peaks", type=int, default=3)
    p.add_argument("--conf-min", type=float, default=0.15)
    p.add_argument("--r2-guard", type=float, default=0.95)
    p.add_argument("--bootstrap", type=int, default=200)
    p.add_argument("--overlay-fps", type=int, default=30)
    p.add_argument("--overlay-format", type=str, default="mp4", choices=["mp4", "gif"])
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m43")
    args = p.parse_args()

    t0 = time.time()
    out_root = Path(args.out_dir)
    if out_root.exists() and any(out_root.rglob("*")) and not bool(args.overwrite):
        raise RuntimeError(f"Output dir exists and is not empty: {out_root} (use --overwrite 1)")
    ensure_dir(out_root)

    n_starts = n_start_list_from_args(args)
    n_starts_str = ",".join(str(x) for x in n_starts)

    # canonical layout
    real_m41_dir = out_root / "real" / "m41"
    real_m42_dir = out_root / "real" / "m42"
    sanity_m41_dir = out_root / "sanity" / "m41"
    sanity_m42_dir = out_root / "sanity" / "m42"
    overlay_real_dir = out_root / "overlay" / "real"
    overlay_sanity_dir = out_root / "overlay" / "sanity"
    for d in [real_m41_dir, real_m42_dir, sanity_m41_dir, sanity_m42_dir, overlay_real_dir, overlay_sanity_dir]:
        ensure_dir(d)

    env = dict(os.environ)
    env["PYTHONPATH"] = "code"

    # 1) M41 real+sanity
    m41_common = [
        "--n-start-list",
        n_starts_str,
        "--q0-grid",
        str(args.q0_grid),
        "--H",
        str(int(args.H)),
        "--W",
        str(int(args.W)),
        "--frames",
        str(int(args.frames)),
        "--n-step",
        str(int(args.n_step)),
        "--mode",
        str(args.mode),
        "--weights",
        str(args.weights),
        "--theta-range-deg",
        str(args.theta_range_deg),
        "--theta-step-deg",
        str(float(args.theta_step_deg)),
        "--theta-refine-halfwidth-deg",
        str(float(args.theta_refine_halfwidth_deg)),
        "--theta-refine-step-deg",
        str(float(args.theta_refine_step_deg)),
        "--target-dx",
        str(float(args.target_dx)),
        "--dt-min",
        str(int(args.dt_min)),
        "--dt-max",
        str(int(args.dt_max)),
        "--smooth",
        str(int(args.smooth)),
        "--peaks",
        str(int(args.peaks)),
        "--conf-min",
        str(float(args.conf_min)),
        "--r2-guard",
        str(float(args.r2_guard)),
        "--bootstrap",
        str(int(args.bootstrap)),
        "--seed",
        str(int(args.seed)),
    ]

    run_cmd(python_cmd("code/scripts/m41_angle_auto_camera.py", [*m41_common, "--sanity", "none", "--out-dir", str(real_m41_dir)]), env=env)
    run_cmd(
        python_cmd(
            "code/scripts/m41_angle_auto_camera.py", [*m41_common, "--sanity", "permute_cols", "--out-dir", str(sanity_m41_dir)]
        ),
        env=env,
    )

    # 2) M42 real+sanity (stored for reproducibility)
    run_cmd(
        python_cmd(
            "code/scripts/m42_angle_q_consistency.py",
            ["--m41-dir", str(real_m41_dir), "--r2-min", str(float(args.r2_guard)), "--out-dir", str(real_m42_dir)],
        ),
        env=env,
    )
    run_cmd(
        python_cmd(
            "code/scripts/m42_angle_q_consistency.py",
            ["--m41-dir", str(sanity_m41_dir), "--r2-min", str(float(args.r2_guard)), "--out-dir", str(sanity_m42_dir)],
        ),
        env=env,
    )

    # 3) PASS/FAIL per n_start (use sanity metrics at the same q0 as real-best)
    best_real_rows = read_csv_rows(real_m41_dir / "m41_best_by_nstart.csv")
    sweep_sanity_rows = read_csv_rows(sanity_m41_dir / "m41_q0_sweep.csv")
    sanity_by_key: Dict[Tuple[int, int], Dict[str, str]] = {}
    for r in sweep_sanity_rows:
        sanity_by_key[(to_int(r.get("n_start", "0")), to_int(r.get("q0", "0")))] = r

    thr_real_r2 = 0.95
    thr_real_peak = 10.0
    thr_sanity_r2 = 0.10
    thr_sanity_peak = 6.0

    results: List[Dict[str, Any]] = []
    fail_reasons_all: List[str] = []

    for r in best_real_rows:
        n0 = to_int(r.get("n_start", "0"))
        q0_best = to_int(r.get("q0_best", "0"))
        dt_best = to_int(r.get("dt", "0"))
        th = to_float(r.get("theta_subdeg", "nan"))
        th_lo = to_float(r.get("theta_ci_lo", "nan"))
        th_hi = to_float(r.get("theta_ci_hi", "nan"))
        q_eff = to_float(r.get("q_eff", "nan"))
        r2_real = to_float(r.get("track_r2", "nan"))
        pk_real = to_float(r.get("peakiness", "nan"))

        q_theta = q_from_theta(th) if math.isfinite(th) else float("nan")
        q_theta_lo = q_from_theta(th_lo) if math.isfinite(th_lo) else float("nan")
        q_theta_hi = q_from_theta(th_hi) if math.isfinite(th_hi) else float("nan")

        sanity_row = sanity_by_key.get((n0, q0_best))
        r2_sanity = to_float(sanity_row.get("track_r2", "nan")) if sanity_row else float("nan")
        pk_sanity = to_float(sanity_row.get("peakiness", "nan")) if sanity_row else float("nan")

        pass_real = (math.isfinite(r2_real) and r2_real >= thr_real_r2) and (math.isfinite(pk_real) and pk_real >= thr_real_peak)
        pass_sanity = (math.isfinite(r2_sanity) and r2_sanity <= thr_sanity_r2) and (math.isfinite(pk_sanity) and pk_sanity <= thr_sanity_peak)

        reasons: List[str] = []
        if not pass_real:
            if not (math.isfinite(r2_real) and r2_real >= thr_real_r2):
                reasons.append(f"real:R2<{thr_real_r2}")
            if not (math.isfinite(pk_real) and pk_real >= thr_real_peak):
                reasons.append(f"real:peakiness<{thr_real_peak}")
        if sanity_row is None:
            reasons.append("sanity:missing_q0_row")
        elif not pass_sanity:
            if not (math.isfinite(r2_sanity) and r2_sanity <= thr_sanity_r2):
                reasons.append(f"sanity:R2>{thr_sanity_r2}")
            if not (math.isfinite(pk_sanity) and pk_sanity <= thr_sanity_peak):
                reasons.append(f"sanity:peakiness>{thr_sanity_peak}")

        pass_overall = bool(pass_real and pass_sanity)
        if not pass_overall:
            fail_reasons_all.append(f"n_start={n0}: " + ", ".join(reasons))

        q_mid, q_ci_lo, q_ci_hi = bootstrap_q_eff_ci(
            n_start=n0,
            q0=q0_best,
            dt=dt_best,
            H=int(args.H),
            W=int(args.W),
            frames=int(args.frames),
            n_step=int(args.n_step),
            smooth=int(args.smooth),
            conf_min=float(args.conf_min),
            seed=int(args.seed),
            bootstrap=int(args.bootstrap),
        )
        if math.isfinite(q_mid):
            q_eff_ci_lo = q_ci_lo
            q_eff_ci_hi = q_ci_hi
        else:
            q_eff_ci_lo = float("nan")
            q_eff_ci_hi = float("nan")

        results.append(
            {
                "n_start": int(n0),
                "q0_best": int(q0_best),
                "dt_best": int(dt_best),
                "theta_subdeg": float(th),
                "theta_ci_lo": float(th_lo),
                "theta_ci_hi": float(th_hi),
                "q_theta": float(q_theta),
                "q_theta_ci_lo": float(q_theta_lo),
                "q_theta_ci_hi": float(q_theta_hi),
                "q_eff": float(q_eff),
                "q_eff_ci_lo": float(q_eff_ci_lo),
                "q_eff_ci_hi": float(q_eff_ci_hi),
                "track_r2_real": float(r2_real),
                "peakiness_real": float(pk_real),
                "track_r2_sanity": float(r2_sanity),
                "peakiness_sanity": float(pk_sanity),
                "pass_real": bool(pass_real),
                "pass_sanity": bool(pass_sanity),
                "pass_overall": bool(pass_overall),
                "fail_reasons": reasons,
            }
        )

    pass_real_all = all(bool(r["pass_real"]) for r in results) if results else False
    pass_sanity_all = all(bool(r["pass_sanity"]) for r in results) if results else False
    pass_overall_all = bool(pass_real_all and pass_sanity_all)

    # 4) Overlay mp4 for the largest n_start
    n_overlay = int(max(n_starts))
    row_overlay = next((r for r in results if int(r["n_start"]) == n_overlay), None)
    if row_overlay is None:
        raise RuntimeError("Failed to select overlay row")
    q0_overlay = int(row_overlay["q0_best"])
    dt_overlay = int(row_overlay["dt_best"])

    tmp_overlay_root = out_root / "_tmp_overlay"
    if tmp_overlay_root.exists():
        shutil.rmtree(tmp_overlay_root)
    ensure_dir(tmp_overlay_root)

    m37_common = [
        "--n-start-list",
        str(int(n_overlay)),
        "--q0-list",
        str(int(q0_overlay)),
        "--H",
        str(int(args.H)),
        "--W",
        str(int(args.W)),
        "--frames",
        str(int(args.frames)),
        "--n-step",
        str(int(args.n_step)),
        "--dt",
        str(int(dt_overlay)),
        "--smooth",
        str(int(args.smooth)),
        "--peaks",
        str(int(args.peaks)),
        "--conf-min",
        str(float(args.conf_min)),
        "--mode",
        str(args.mode),
        "--weights",
        str(args.weights),
        "--fps",
        str(int(args.overlay_fps)),
        "--format",
        str(args.overlay_format),
        "--seed",
        str(int(args.seed)),
    ]

    run_cmd(python_cmd("code/scripts/m37_comoving_window.py", [*m37_common, "--sanity", "none", "--out-dir", str(tmp_overlay_root)]), env=env)
    run_cmd(
        python_cmd(
            "code/scripts/m37_comoving_window.py",
            [*m37_common, "--sanity", "permute_cols", "--out-dir", str(tmp_overlay_root / "sanity_permute_cols")],
        ),
        env=env,
    )

    real_best_dir = tmp_overlay_root / "best_overlay" / "real"
    sanity_best_dir = tmp_overlay_root / "best_overlay" / "sanity"
    ext = "mp4" if args.overlay_format == "mp4" else "gif"
    real_vid = real_best_dir / f"m37_best_overlay_tail.{ext}"
    sanity_vid = sanity_best_dir / f"m37_best_overlay_tail.{ext}"
    if not real_vid.exists() or not sanity_vid.exists():
        raise FileNotFoundError("Missing overlay videos from M37 best_overlay")

    out_real_vid = overlay_real_dir / f"m43_overlay_tail.{ext}"
    out_sanity_vid = overlay_sanity_dir / f"m43_overlay_tail.{ext}"
    shutil.copyfile(real_vid, out_real_vid)
    shutil.copyfile(sanity_vid, out_sanity_vid)

    # preview from keyframe t=150 (fallback t=000)
    real_key = real_best_dir / "keyframe_t150.png"
    sanity_key = sanity_best_dir / "keyframe_t150.png"
    if not real_key.exists() or not sanity_key.exists():
        real_key = real_best_dir / "keyframe_t000.png"
        sanity_key = sanity_best_dir / "keyframe_t000.png"

    preview_path = out_root / "m43_preview.png"
    sub_real = (
        f"n_start={n_overlay}  q0={q0_overlay}  dt={dt_overlay}\n"
        f"theta={row_overlay['theta_subdeg']:.3f} [{row_overlay['theta_ci_lo']:.3f},{row_overlay['theta_ci_hi']:.3f}]\n"
        f"q_eff={row_overlay['q_eff']:.2f} [{row_overlay['q_eff_ci_lo']:.2f},{row_overlay['q_eff_ci_hi']:.2f}]\n"
        f"R2={row_overlay['track_r2_real']:.3f}  peakiness={row_overlay['peakiness_real']:.2f}  pass={row_overlay['pass_overall']}"
    )
    sub_sanity = (
        f"eval@q0={q0_overlay}\n"
        f"R2={row_overlay['track_r2_sanity']:.3f}  peakiness={row_overlay['peakiness_sanity']:.2f}  pass={row_overlay['pass_sanity']}"
    )
    make_preview(
        out_path=preview_path,
        real_img_path=real_key,
        sanity_img_path=sanity_key,
        title="M43 measure_wave preview",
        subtitle_real=sub_real,
        subtitle_sanity=sub_sanity,
    )
    shutil.rmtree(tmp_overlay_root)

    build_table_tex(out_root / "m43_table.tex", results)
    write_best_csv(out_root / "m43_best_by_nstart.csv", results)

    summary = {
        "params": {
            **{k: v for k, v in vars(args).items() if k not in ["n_start", "n_start_list"]},
            "n_start_list": n_starts_str,
            "layout": {
                "real_m41": "real/m41",
                "real_m42": "real/m42",
                "sanity_m41": "sanity/m41",
                "sanity_m42": "sanity/m42",
                "overlay_real": "overlay/real",
                "overlay_sanity": "overlay/sanity",
            },
        },
        "git_sha": git_sha(),
        "runtime_s": float(time.time() - t0),
        "thresholds": {
            "real": {"track_r2_min": thr_real_r2, "peakiness_min": thr_real_peak},
            "sanity": {"track_r2_max": thr_sanity_r2, "peakiness_max": thr_sanity_peak},
        },
        "pass_real": bool(pass_real_all),
        "pass_sanity": bool(pass_sanity_all),
        "pass_overall": bool(pass_overall_all),
        "fail_reasons": fail_reasons_all,
        "overlay": {
            "n_start": int(n_overlay),
            "q0": int(q0_overlay),
            "dt": int(dt_overlay),
            "real_video": str(out_real_vid.relative_to(out_root)).replace("\\", "/"),
            "sanity_video": str(out_sanity_vid.relative_to(out_root)).replace("\\", "/"),
        },
        "best_by_nstart": results,
    }
    (out_root / "m43_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(out_root, params=summary["params"], name="m43_manifest.json")
    print(f"OK: wrote M43 outputs to {out_root}")


if __name__ == "__main__":
    main()

