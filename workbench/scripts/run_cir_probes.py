#!/usr/bin/env python3
"""Single resumable CLI entry point for the full CIR 18-suite / 29-failure probe campaign.

Strictly enforces AGENTS.md rules:
  - Sequential GPU execution (1 GPU job at a time; never overlaps GPU processes)
  - CPU cap <= 10 threads (OMP_NUM_THREADS<=8, DataLoader num_workers<=8)
  - Resumable per (stage, dataset, model) via DONE.json completion markers

Usage:
  workbench/envs/ctl312h/bin/python workbench/scripts/run_cir_probes.py --stage all
  workbench/envs/ctl312h/bin/python workbench/scripts/run_cir_probes.py --stage inference --datasets edir --models clip4cir,conesep
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path("/home/heheboiz/data/cir")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workbench.probes import DEFAULT_PROBE_OUT  # noqa: E402
from workbench.probes.cohort import MODEL_BY_KEY, MODEL_SPECS  # noqa: E402

CTL_PY = str(REPO_ROOT / "workbench/envs/ctl312h/bin/python")
LIMN_PY = str(REPO_ROOT / "workbench/envs/limn310-adapted/bin/python")


def _run_cmd(cmd: List[str], env_extra: Dict[str, str], log_file: Path, label: str) -> bool:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    env.update(env_extra)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[RUN] {label} -> {log_file}", flush=True)
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode == 0:
        print(f"[OK ] {label} ({dt:.1f}s)", flush=True)
        return True
    print(f"[ERR] {label} rc={proc.returncode} ({dt:.1f}s) — tail:", flush=True)
    tail = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()[-25:]
    for line in tail:
        print("   ", line, flush=True)
    return False


def _group_models_for_dispatch(selected_keys: List[str], dataset: str) -> List[Tuple[str, List[str]]]:
    """Always run each model in its own clean isolated subprocess (zero third_party `lavis` sys.modules collision!)."""
    return [(mk, [mk]) for mk in selected_keys]


def _pythonpath_for_group(models: List[str]) -> str:
    # For shared ilearn group, model_adapter inserts each model's third_party dir cleanly at load time
    paths = [str(REPO_ROOT)]
    for mk in models:
        for p in MODEL_BY_KEY[mk].pythonpath.split(":"):
            if p and p not in paths:
                paths.append(p)
    return ":".join(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified resumable CIR diagnostic probe suite runner (E01-E18)")
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "cohorts", "arbiter", "inference", "train_e02", "train_e16", "train_e17", "e18", "aggregate"],
    )
    parser.add_argument("--datasets", type=str, default="edir,pinpoint,fashioniq")
    parser.add_argument("--models", type=str, default="all", help="'all' or comma-separated model keys")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_PROBE_OUT)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = out_dir / "logs"
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    if args.models.strip().lower() == "all":
        selected_models = [m.key for m in MODEL_SPECS]
    else:
        selected_models = [m.strip().lower() for m in args.models.split(",") if m.strip()]

    nw = max(0, min(int(args.num_workers), 8))
    failures: List[str] = []

    # Stage 1: Cohorts & Manifest
    if args.stage in ("all", "cohorts"):
        if not (out_dir / "manifest/diagnostic_cohorts.json").is_file() or args.force:
            ok = _run_cmd(
                [CTL_PY, "-u", str(REPO_ROOT / "workbench/probes/build_cohorts.py"), "--out-dir", str(out_dir)],
                {"PYTHONPATH": str(REPO_ROOT)},
                log_dir / "stage_cohorts.log",
                "cohorts",
            )
            if not ok:
                sys.exit(1)

    # Stage 2: Frozen CLIP ViT-L/14 Arbiter
    if args.stage in ("all", "arbiter"):
        ok = _run_cmd(
            [LIMN_PY, "-u", str(REPO_ROOT / "workbench/probes/arbiter_clip.py"), "--datasets", ",".join(datasets), "--out-dir", str(out_dir), "--num-workers", str(nw)],
            {"PYTHONPATH": str(REPO_ROOT)},
            log_dir / "stage_arbiter.log",
            "arbiter_clip",
        )
        if not ok:
            sys.exit(1)

    # Stage 3: Full Inference & Causal Intervention Suites (E01, E03-E15) across all selected models & datasets
    if args.stage in ("all", "inference"):
        worker = str(REPO_ROOT / "workbench/probes/worker_eval.py")
        for ds in datasets:
            groups = _group_models_for_dispatch(selected_models, ds)
            print(f"\n=== INFERENCE STAGE: dataset={ds} ({len(groups)} dispatch groups) ===", flush=True)
            for gname, gmodels in groups:
                # Check if all models in group already DONE
                if not args.force and all((out_dir / "models" / ds / m / "DONE.json").is_file() for m in gmodels):
                    print(f"[SKIP] {ds}/{gname} ({gmodels}) already DONE", flush=True)
                    continue
                py_exe = MODEL_BY_KEY[gmodels[0]].python
                pp = _pythonpath_for_group(gmodels) if len(gmodels) == 1 else str(REPO_ROOT)
                if len(gmodels) > 1 and gname == "cir_evag":
                    pp = _pythonpath_for_group(gmodels)
                cmd = [
                    py_exe, "-u", worker,
                    "--models", ",".join(gmodels),
                    "--dataset", ds,
                    "--out-dir", str(out_dir),
                    "--num-workers", str(nw),
                ]
                if args.force:
                    cmd.append("--force")
                ok = _run_cmd(cmd, {"PYTHONPATH": pp}, log_dir / f"infer_{ds}_{gname}.log", f"infer:{ds}:{gname}")
                if not ok:
                    failures.append(f"infer:{ds}:{gname}")

    # Stage 4: Controlled Training Arms E02 (Same-Scene Negative & Ref-Negative Intervention)
    if args.stage in ("all", "train_e02"):
        runner_t = str(REPO_ROOT / "workbench/probes/train_arms.py")
        if Path(runner_t).is_file():
            for mk, py_exe, pp in [
                ("clip4cir", MODEL_BY_KEY["clip4cir"].python, MODEL_BY_KEY["clip4cir"].pythonpath),
                ("conesep", MODEL_BY_KEY["conesep"].python, MODEL_BY_KEY["conesep"].pythonpath),
                ("sprc", MODEL_BY_KEY["sprc"].python, MODEL_BY_KEY["sprc"].pythonpath),
            ]:
                cmd = [py_exe, "-u", runner_t, "--suite", "e02", "--model", mk, "--out-dir", str(out_dir), "--num-workers", str(nw)]
                if args.force:
                    cmd.append("--force")
                ok = _run_cmd(cmd, {"PYTHONPATH": pp}, log_dir / f"train_e02_{mk}.log", f"train_e02:{mk}")
                if not ok:
                    failures.append(f"train_e02:{mk}")

    # Stage 5: Controlled Training Arms E16 (Cross-Domain Gradient Conflict + PCGrad + Fairness)
    if args.stage in ("all", "train_e16"):
        runner_t = str(REPO_ROOT / "workbench/probes/train_arms.py")
        if Path(runner_t).is_file():
            for mk, py_exe, pp in [
                ("clip4cir", MODEL_BY_KEY["clip4cir"].python, MODEL_BY_KEY["clip4cir"].pythonpath),
                ("conesep", MODEL_BY_KEY["conesep"].python, MODEL_BY_KEY["conesep"].pythonpath),
                ("sprc", MODEL_BY_KEY["sprc"].python, MODEL_BY_KEY["sprc"].pythonpath),
            ]:
                cmd = [py_exe, "-u", runner_t, "--suite", "e16", "--model", mk, "--out-dir", str(out_dir), "--num-workers", str(nw)]
                if args.force:
                    cmd.append("--force")
                ok = _run_cmd(cmd, {"PYTHONPATH": pp}, log_dir / f"train_e16_{mk}.log", f"train_e16:{mk}")
                if not ok:
                    failures.append(f"train_e16:{mk}")

    # Stage 6: Training Dynamics E17 (history.jsonl all 16 models + multi-stage init/early/best/final probe eval)
    if args.stage in ("all", "train_e17"):
        runner_t = str(REPO_ROOT / "workbench/probes/train_arms.py")
        if Path(runner_t).is_file():
            ok = _run_cmd(
                [CTL_PY, "-u", runner_t, "--suite", "e17_history", "--model", "all", "--out-dir", str(out_dir)],
                {"PYTHONPATH": str(REPO_ROOT)},
                log_dir / "e17_history_all16.log",
                "e17_history_all16",
            )
            if not ok:
                failures.append("e17_history_all16")
            for mk in ("clip4cir", "conesep", "sprc"):
                cmd = [MODEL_BY_KEY[mk].python, "-u", runner_t, "--suite", "e17_trajectory", "--model", mk, "--out-dir", str(out_dir), "--num-workers", str(nw)]
                if args.force:
                    cmd.append("--force")
                ok = _run_cmd(cmd, {"PYTHONPATH": MODEL_BY_KEY[mk].pythonpath}, log_dir / f"e17_traj_{mk}.log", f"e17_traj:{mk}")
                if not ok:
                    failures.append(f"e17_traj:{mk}")

    # Stage 7: E18 Blind Human Audit Pack
    if args.stage in ("all", "e18"):
        if not (out_dir / "e18_blind_audit/e18_summary.json").is_file() or args.force:
            ok = _run_cmd(
                [CTL_PY, "-u", str(REPO_ROOT / "workbench/probes/e18_audit_pack.py"), "--out-dir", str(out_dir), "--per-stratum", "20"],
                {"PYTHONPATH": str(REPO_ROOT)},
                log_dir / "stage_e18.log",
                "e18_blind_audit",
            )
            if not ok:
                failures.append("e18")

    # Stage 8: Cross-Model Aggregation, E04 Probe Fitting, BH-FDR, Coverage Matrix & Final Report
    if args.stage in ("all", "aggregate"):
        agg_script = REPO_ROOT / "workbench/probes/aggregate_report.py"
        if agg_script.is_file():
            ok = _run_cmd(
                [CTL_PY, "-u", str(agg_script), "--out-dir", str(out_dir)],
                {"PYTHONPATH": str(REPO_ROOT)},
                log_dir / "stage_aggregate.log",
                "aggregate_report",
            )
            if not ok:
                failures.append("aggregate")

    print(f"\n=== CIR PROBES FINISHED | stage={args.stage} | failures={failures} ===", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
