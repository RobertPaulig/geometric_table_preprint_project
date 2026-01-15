from __future__ import annotations

import argparse
import csv
import hashlib
import json
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_manifest(out_dir: Path, params: Dict[str, Any], *, name: str = "m44_manifest.json") -> None:
    files = sorted([p for p in out_dir.rglob("*") if p.is_file() and p.name != name])
    manifest = {
        "params": params,
        "git_sha": git_sha(),
        "files": {str(p.relative_to(out_dir)).replace("\\", "/"): sha256_file(p) for p in files},
    }
    (out_dir / name).write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def format_nstart(n: int) -> str:
    if n % 1_000_000 == 0:
        return f"{n // 1_000_000}e6"
    return str(n)


def python_cmd(script: str, args: List[str]) -> List[str]:
    return [sys.executable, script, *args]


def run_cmd(cmd: List[str], *, env: Dict[str, str]) -> None:
    subprocess.check_call(cmd, env=env)


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def plot_pass_heatmap(out_path: Path, n_starts: List[int], seeds: List[int], pass_mat: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.6, 2.8), dpi=150)
    im = ax.imshow(pass_mat.astype(float), vmin=0.0, vmax=1.0, cmap="RdYlGn")
    ax.set_xticks(np.arange(len(seeds)))
    ax.set_xticklabels([str(s) for s in seeds])
    ax.set_yticks(np.arange(len(n_starts)))
    ax.set_yticklabels([format_nstart(n) for n in n_starts])
    ax.set_xlabel("seed")
    ax.set_ylabel("n_start")
    ax.set_title("M44: PASS matrix (M43 instrument)")

    for i in range(pass_mat.shape[0]):
        for j in range(pass_mat.shape[1]):
            ax.text(j, i, "PASS" if pass_mat[i, j] else "FAIL", ha="center", va="center", fontsize=8, color="#000000")

    cbar = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03)
    cbar.set_label("pass_overall", rotation=90)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_theta_ci_by_nstart(out_path: Path, rows: List[Dict[str, Any]], n_starts: List[int], seeds: List[int]) -> None:
    import matplotlib.pyplot as plt

    x_pos = np.arange(len(n_starts), dtype=float)
    seed_to_offset = {s: (i - (len(seeds) - 1) / 2.0) * 0.08 for i, s in enumerate(seeds)}
    colors = {1_000_000: "#4c78a8", 50_000_000: "#f58518", 200_000_000: "#54a24b"}

    fig, ax = plt.subplots(figsize=(7.0, 3.2), dpi=150)
    for s in seeds:
        xs: List[float] = []
        ys: List[float] = []
        yerr_lo: List[float] = []
        yerr_hi: List[float] = []
        cs: List[str] = []
        for i, n0 in enumerate(n_starts):
            r = next((rr for rr in rows if int(rr["n_start"]) == n0 and int(rr["seed"]) == s), None)
            if r is None:
                continue
            th = float(r.get("theta_subdeg", float("nan")))
            lo = float(r.get("theta_ci_lo", float("nan")))
            hi = float(r.get("theta_ci_hi", float("nan")))
            if not (np.isfinite(th) and np.isfinite(lo) and np.isfinite(hi)):
                continue
            xs.append(float(i) + seed_to_offset[s])
            ys.append(th)
            yerr_lo.append(th - lo)
            yerr_hi.append(hi - th)
            cs.append(colors.get(n0, "#888888"))
        if not xs:
            continue
        ax.errorbar(xs, ys, yerr=[yerr_lo, yerr_hi], fmt="o", markersize=4, linewidth=1.0, capsize=2, label=f"seed={s}", color="#333333")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([format_nstart(n) for n in n_starts])
    ax.set_xlabel("n_start")
    ax.set_ylabel(r"$\theta_{\mathrm{sub}}$ (deg)")
    ax.set_title("M44: theta_subdeg with bootstrap CI (real)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, framealpha=0.9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_q0_dt_scatter(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.4), dpi=150)
    colors = {1_000_000: "#4c78a8", 50_000_000: "#f58518", 200_000_000: "#54a24b"}
    markers = {123: "o", 456: "s", 789: "D"}

    for r in rows:
        n0 = int(r["n_start"])
        seed = int(r["seed"])
        q0 = float(r.get("q0_best", float("nan")))
        dt = float(r.get("dt_best", float("nan")))
        if not (np.isfinite(q0) and np.isfinite(dt)):
            continue
        ax.scatter(q0, dt, s=45, alpha=0.9, color=colors.get(n0, "#888888"), marker=markers.get(seed, "o"))

    ax.set_xlabel("q0_best")
    ax.set_ylabel("dt_best")
    ax.set_title("M44: selected (q0, dt) across n_start × seed")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_peakiness_boxplot(out_path: Path, rows: List[Dict[str, Any]], n_starts: List[int]) -> None:
    import matplotlib.pyplot as plt

    data: List[List[float]] = []
    labels: List[str] = []
    for n0 in n_starts:
        vals = [float(r.get("peakiness_real", float("nan"))) for r in rows if int(r["n_start"]) == n0]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            continue
        data.append(vals)
        labels.append(format_nstart(n0))

    fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=150)
    if data:
        ax.boxplot(data, labels=labels, showfliers=False)
        ax.axhline(10.0, color="#000000", linewidth=1.0, alpha=0.35, linestyle="--", label="threshold=10")
        ax.set_xlabel("n_start")
        ax.set_ylabel("peakiness (real)")
        ax.set_title("M44: peakiness distribution by n_start (real)")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8, framealpha=0.9, loc="best")
    else:
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_table_tex(out_path: Path, rows: List[Dict[str, Any]]) -> None:
    lines: List[str] = []
    lines.append(r"\begin{tabular}{r r r r r r r r}")
    lines.append(r"\hline")
    lines.append(r"$n_{\mathrm{start}}$ & seed & pass & $q_0$ & $dt$ & $\theta_{\mathrm{sub}}$ & $R^2$ & peakiness \\")
    lines.append(r"\hline")
    for r in rows:
        n0 = int(r["n_start"])
        seed = int(r["seed"])
        passed = "PASS" if bool(r.get("pass_overall")) else "FAIL"
        q0 = int(r.get("q0_best", 0)) if str(r.get("q0_best", "")).strip() else 0
        dt = int(r.get("dt_best", 0)) if str(r.get("dt_best", "")).strip() else 0
        th = float(r.get("theta_subdeg", float("nan")))
        r2 = float(r.get("track_r2_real", float("nan")))
        pk = float(r.get("peakiness_real", float("nan")))
        th_s = f"{th:.3f}" if np.isfinite(th) else "--"
        r2_s = f"{r2:.3f}" if np.isfinite(r2) else "--"
        pk_s = f"{pk:.2f}" if np.isfinite(pk) else "--"
        lines.append(f"{n0} & {seed} & {passed} & {q0} & {dt} & {th_s} & {r2_s} & {pk_s} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description="M44: repeatability panel for the M43 measure_wave instrument.")
    p.add_argument("--n-start-list", type=str, default="1000000,50000000,200000000")
    p.add_argument("--seeds", type=str, default="123,456,789")
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
    p.add_argument("--out-dir", type=str, default="out/wave_atlas/m44")
    p.add_argument("--overwrite", type=int, default=0, choices=[0, 1])
    args = p.parse_args()

    out_root = Path(args.out_dir)
    if out_root.exists() and any(out_root.rglob("*")) and not bool(args.overwrite):
        raise RuntimeError(f"Output dir exists and is not empty: {out_root} (use --overwrite 1)")
    if out_root.exists() and bool(args.overwrite):
        shutil.rmtree(out_root)
    ensure_dir(out_root)

    runs_root = out_root / "runs"
    ensure_dir(runs_root)
    overlay_fail_root = out_root / "overlay_failures"
    ensure_dir(overlay_fail_root)

    n_starts = [int(x) for x in args.n_start_list.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    n_starts = sorted(n_starts)
    seeds = sorted(seeds)

    env = dict(os.environ)
    env["PYTHONPATH"] = "code"

    ref_n = 200_000_000
    ref_seed = 123

    panel_rows: List[Dict[str, Any]] = []
    run_index: List[Dict[str, Any]] = []
    t0 = time.time()

    for n0 in n_starts:
        for seed in seeds:
            run_name = f"n{format_nstart(n0)}_seed{seed}"
            run_dir = runs_root / run_name
            overlay_mode = "auto" if (n0 == ref_n and seed == ref_seed) else "none"
            ensure_dir(run_dir)

            cmd = python_cmd(
                "code/scripts/m43_measure_wave.py",
                [
                    "--n-start",
                    str(int(n0)),
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
                    "--overlay",
                    str(overlay_mode),
                    "--overlay-fps",
                    str(int(args.overlay_fps)),
                    "--overlay-format",
                    str(args.overlay_format),
                    "--seed",
                    str(int(seed)),
                    "--out-dir",
                    str(run_dir),
                ],
            )

            run_err: Optional[str] = None
            try:
                run_cmd(cmd, env=env)
            except subprocess.CalledProcessError as e:
                run_err = f"exit={e.returncode}"

            summary_path = run_dir / "m43_summary.json"
            if not summary_path.exists():
                panel_rows.append(
                    {
                        "n_start": n0,
                        "seed": seed,
                        "overlay_mode": overlay_mode,
                        "pass_overall": False,
                        "run_error": run_err or "missing_summary",
                        "run_dir": str(run_dir.relative_to(out_root)).replace("\\", "/"),
                    }
                )
                run_index.append({"n_start": n0, "seed": seed, "run_dir": str(run_dir.relative_to(out_root)).replace("\\", "/"), "ok": False})
                continue

            s = load_json(summary_path)
            best = (s.get("best_by_nstart") or [{}])[0]
            row = {
                "n_start": int(n0),
                "seed": int(seed),
                "overlay_mode": overlay_mode,
                "pass_real": bool(s.get("pass_real")),
                "pass_sanity": bool(s.get("pass_sanity")),
                "pass_overall": bool(s.get("pass_overall")),
                "fail_reasons": "; ".join(s.get("fail_reasons") or []),
                "q0_best": best.get("q0_best"),
                "dt_best": best.get("dt_best"),
                "theta_subdeg": best.get("theta_subdeg"),
                "theta_ci_lo": best.get("theta_ci_lo"),
                "theta_ci_hi": best.get("theta_ci_hi"),
                "q_eff": best.get("q_eff"),
                "q_eff_ci_lo": best.get("q_eff_ci_lo"),
                "q_eff_ci_hi": best.get("q_eff_ci_hi"),
                "track_r2_real": best.get("track_r2_real"),
                "peakiness_real": best.get("peakiness_real"),
                "track_r2_sanity": best.get("track_r2_sanity"),
                "peakiness_sanity": best.get("peakiness_sanity"),
                "runtime_s": s.get("runtime_s"),
                "run_dir": str(run_dir.relative_to(out_root)).replace("\\", "/"),
            }
            panel_rows.append(row)
            run_index.append({"n_start": n0, "seed": seed, "run_dir": str(run_dir.relative_to(out_root)).replace("\\", "/"), "ok": True})

            # Overlay for failures is allowed only under overlay_failures/.
            if overlay_mode == "none" and not bool(row.get("pass_overall")):
                fail_dir = overlay_fail_root / run_name
                if fail_dir.exists():
                    shutil.rmtree(fail_dir)
                ensure_dir(fail_dir)
                cmd_fail = cmd.copy()
                # Replace out-dir and overlay mode
                for i in range(len(cmd_fail) - 1):
                    if cmd_fail[i] == "--overlay":
                        cmd_fail[i + 1] = "auto"
                    if cmd_fail[i] == "--out-dir":
                        cmd_fail[i + 1] = str(fail_dir)
                try:
                    run_cmd(cmd_fail, env=env)
                except subprocess.CalledProcessError:
                    pass

    # Root-level tables
    panel_fields = [
        "n_start",
        "seed",
        "overlay_mode",
        "pass_real",
        "pass_sanity",
        "pass_overall",
        "fail_reasons",
        "q0_best",
        "dt_best",
        "theta_subdeg",
        "theta_ci_lo",
        "theta_ci_hi",
        "q_eff",
        "q_eff_ci_lo",
        "q_eff_ci_hi",
        "track_r2_real",
        "peakiness_real",
        "track_r2_sanity",
        "peakiness_sanity",
        "runtime_s",
        "run_dir",
        "run_error",
    ]
    for r in panel_rows:
        r.setdefault("run_error", "")
    write_csv(out_root / "m44_metrics_panel.csv", panel_rows, panel_fields)

    # Pass matrix
    pass_mat = np.zeros((len(n_starts), len(seeds)), dtype=bool)
    for i, n0 in enumerate(n_starts):
        for j, seed in enumerate(seeds):
            r = next((rr for rr in panel_rows if int(rr["n_start"]) == n0 and int(rr["seed"]) == seed), None)
            pass_mat[i, j] = bool(r.get("pass_overall")) if r else False

    pass_matrix_rows: List[Dict[str, Any]] = []
    for i, n0 in enumerate(n_starts):
        row: Dict[str, Any] = {"n_start": n0}
        for j, seed in enumerate(seeds):
            row[f"seed{seed}"] = "PASS" if pass_mat[i, j] else "FAIL"
        pass_matrix_rows.append(row)
    write_csv(out_root / "m44_pass_matrix.csv", pass_matrix_rows, ["n_start"] + [f"seed{seed}" for seed in seeds])

    # Plots
    plot_pass_heatmap(out_root / "m44_pass_heatmap.png", n_starts=n_starts, seeds=seeds, pass_mat=pass_mat)
    plot_theta_ci_by_nstart(out_root / "m44_theta_ci_by_nstart.png", rows=panel_rows, n_starts=n_starts, seeds=seeds)
    plot_q0_dt_scatter(out_root / "m44_q0_dt_scatter.png", rows=panel_rows)
    plot_peakiness_boxplot(out_root / "m44_peakiness_boxplot.png", rows=panel_rows, n_starts=n_starts)

    build_table_tex(out_root / "m44_table.tex", panel_rows)

    summary = {
        "params": {
            "n_start_list": n_starts,
            "seeds": seeds,
            "reference_overlay": {"n_start": ref_n, "seed": ref_seed, "policy": "only this run has overlay; failures allowed under overlay_failures/"},
            "m43_params": {
                "q0_grid": args.q0_grid,
                "H": int(args.H),
                "W": int(args.W),
                "frames": int(args.frames),
                "n_step": int(args.n_step),
                "mode": args.mode,
                "weights": args.weights,
                "theta_range_deg": args.theta_range_deg,
                "theta_step_deg": float(args.theta_step_deg),
                "theta_refine_halfwidth_deg": float(args.theta_refine_halfwidth_deg),
                "theta_refine_step_deg": float(args.theta_refine_step_deg),
                "target_dx": float(args.target_dx),
                "dt_min": int(args.dt_min),
                "dt_max": int(args.dt_max),
                "smooth": int(args.smooth),
                "peaks": int(args.peaks),
                "conf_min": float(args.conf_min),
                "r2_guard": float(args.r2_guard),
                "bootstrap": int(args.bootstrap),
            },
            "thresholds": {"real": {"track_r2_min": 0.95, "peakiness_min": 10}, "sanity": {"track_r2_max": 0.10, "peakiness_max": 6}},
        },
        "git_sha": git_sha(),
        "runtime_s": float(time.time() - t0),
        "pass_overall_count": int(np.sum(pass_mat)),
        "total_runs": int(pass_mat.size),
        "runs": run_index,
    }
    (out_root / "m44_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_manifest(out_root, params=summary["params"], name="m44_manifest.json")

    print(f"OK: wrote M44 outputs to {out_root}")


if __name__ == "__main__":
    main()

