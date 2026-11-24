#!/usr/bin/env python3
"""Sequential, resumable orchestrator for cross-benchmark checkpoint transfer evaluation:
  1) FashionIQ -> PinPoint (14/15 models)
  2) CIRR      -> EDIR     (10/15 models)
  3) FashionIQ -> EDIR     (14/15 models)

Strictly runs 1 GPU model at a time. Supports `--verify-only` to run live GPU strict
checkpoint load + forward smoke verification on ALL 24 checkpoints before full evaluation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

REPO = Path("/home/heheboiz/data/cir")
ENVS = REPO / "workbench" / "envs"
CIR_ENV_PY = REPO / "workbench" / "artifacts" / "environments" / "cir-005ef4e1b02999a5" / "bin" / "python"
WORKER = REPO / "workbench" / "benchmarks" / "trainers" / "eval_transfer_worker.py"
FIQ_ROOT = REPO / "workbench" / "artifacts" / "checkpoint_selection" / "20261106T002617+0700"
QURE_CKPT_ROOT = REPO / "workbench" / "artifacts" / "qure_checkpoints"
CIRR_ROOT = REPO / "workbench" / "artifacts" / "cirr_checkpoints"
OUT_ROOT = REPO / "workbench" / "artifacts" / "transfer_evals"

ENV_SPECS: Dict[str, Dict[str, str]] = {
    "clip4cir": {"python": str(ENVS / "clip4cir310-sm120-pep610/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/CLIP4Cir/src"},
    "encoder": {"python": str(ENVS / "encoder310-adapted/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/ENCODER"},
    "tgcir": {"python": str(ENVS / "tgcir39-la/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/TGCIR"},
    "clvcnet": {"python": str(ENVS / "clvc39-la/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/CLVCNet"},
    "dcnet": {"python": str(ENVS / "dcnet310-sm120/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/DCNet"},
    "sprc": {"python": str(ENVS / "sprc310-sm120/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/SPRC/src"},
    "qure": {"python": str(ENVS / "qure310-sm120/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/QuRe"},
    "mtst_ptha": {"python": str(CIR_ENV_PY), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/MTST_PTHA/MTST_PTHA/src"},
    "hint": {"python": str(ENVS / "ilearn39-adapted/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/HINT"},
    "habit": {"python": str(ENVS / "ilearn39-adapted/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/HABIT"},
    "intent": {"python": str(ENVS / "ilearn39-adapted/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/INTENT"},
    "conesep": {"python": str(ENVS / "ilearn39-adapted/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/ConeSep"},
    "airknow": {"python": str(ENVS / "airknow310-la/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/AirKnow"},
    "csmcir": {"python": str(CIR_ENV_PY), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/CSMCIR/src"},
    "tme": {"python": str(ENVS / "tme-adapted/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/TME/src"},
    "limn": {"python": str(ENVS / "limn310-adapted/bin/python"), "pythonpath": f"{REPO}:{REPO}/workbench/third_party/LIMN/LIMN"},
}

FIQ_CHECKPOINTS: List[Dict[str, Any]] = [
    {"model": "encoder", "variant": "fiq_clean", "ckpts": [str(FIQ_ROOT / "cohort/downloads/encoder/fashioniq.pt")]},
    {"model": "tgcir", "variant": "fiq_clean", "ckpts": [str(FIQ_ROOT / "downloads/tgcir/x/checkpoints/Fashioniq.pt")]},
    {
        "model": "clip4cir",
        "variant": "fiq_clean",
        "ckpts": [
            str(FIQ_ROOT / "checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_clip_RN50x4_fullft.pt"),
            str(FIQ_ROOT / "checkpoints/clip4cir/FashionIQ/RN50x4_fullft/fiq_comb_RN50x4_fullft.pt"),
        ],
    },
    {
        "model": "limn",
        "variant": "fiq_3head_ens",
        "dataset_type_ids": "0,1,2",
        "ckpts": [
            str(FIQ_ROOT / "cohort/downloads/limn/0_dress_best_model.pt"),
            str(FIQ_ROOT / "cohort/downloads/limn/0_shirt_best_model.pt"),
            str(FIQ_ROOT / "cohort/downloads/limn/0_toptee_best_model.pt"),
        ],
    },
    {
        "model": "clvcnet",
        "variant": "fiq_3head_ens",
        "ckpts": [
            str(FIQ_ROOT / "checkpoints/clvc/dress_local_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/dress_global_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/shirt_local_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/shirt_global_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/toptee_local_model.pt"),
            str(FIQ_ROOT / "checkpoints/clvc/toptee_global_model.pt"),
        ],
    },
    {"model": "sprc", "variant": "fiq_vitl_clean", "ckpts": [str(FIQ_ROOT / "checkpoints/sprc/sprc_fiq_vitl.pt")]},
    {"model": "hint", "variant": "fiq_clean", "ckpts": [str(FIQ_ROOT / "cohort/downloads/hint/fashioniq.pt")]},
    {"model": "habit", "variant": "fiq_n02", "ckpts": [str(FIQ_ROOT / "cohort/downloads/habit/HABIT-FIQ_N0.2.pt")]},
    {"model": "intent", "variant": "fiq_n02", "ckpts": [str(FIQ_ROOT / "cohort/downloads/intent/intent_fiq_0.2.pt")]},
    {"model": "conesep", "variant": "fiq_n02", "ckpts": [str(FIQ_ROOT / "cohort/downloads/conesep/ConeSep-FIQ-N0.2.pt")]},
    {"model": "airknow", "variant": "fiq_n05", "ckpts": [str(FIQ_ROOT / "checkpoints/airknow/airknow_fiq_0.5.pt")]},
    {"model": "csmcir", "variant": "fiq_clean", "ckpts": [str(FIQ_ROOT / "cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt")]},
    {"model": "tme", "variant": "fiq_clean", "ckpts": [str(FIQ_ROOT / "cohort/ckpt/tme/fashioniq_0_best_model.pth")]},
    {"model": "dcnet", "variant": "fiq_clean", "ckpts": [str(FIQ_ROOT / "checkpoints/dcnet/trained_model.pth")]},
    {"model": "qure", "variant": "fiq_vitl_repro", "ckpts": [str(QURE_CKPT_ROOT / "fashioniq/best_model.pth")]},
]

CIRR_CHECKPOINTS: List[Dict[str, Any]] = [
    {"model": "encoder", "variant": "cirr_clean", "ckpts": [str(CIRR_ROOT / "encoder/cirr.pt")]},
    {"model": "tgcir", "variant": "cirr_clean", "ckpts": [str(CIRR_ROOT / "tgcir/CIRR.pt")]},
    {
        "model": "clip4cir",
        "variant": "cirr_clean",
        "ckpts": [
            str(CIRR_ROOT / "clip4cir/cirr_clip_RN50x4_fullft.pt"),
            str(CIRR_ROOT / "clip4cir/cirr_comb_RN50x4_fullft.pt"),
        ],
    },
    {"model": "limn", "variant": "cirr_clean", "dataset_type_ids": "5", "ckpts": [str(CIRR_ROOT / "limn/0_cirr_best_model.pt")]},
    {"model": "sprc", "variant": "cirr_vitl_clean", "ckpts": [str(CIRR_ROOT / "sprc/sprc_cirr_vitl.pt")]},
    {"model": "mtst_ptha", "variant": "cirr_clean", "ckpts": [str(CIRR_ROOT / "mtst_ptha/tuned_mtst_cirr.pt")]},
    {"model": "hint", "variant": "cirr_clean", "ckpts": [str(CIRR_ROOT / "hint/cirr.pt")]},
    {"model": "habit", "variant": "cirr_n02", "ckpts": [str(CIRR_ROOT / "habit/HABIT-CIRR-N0.2.pt")]},
    {"model": "intent", "variant": "cirr_n02", "ckpts": [str(CIRR_ROOT / "intent/intent_cirr_0.2.pt")]},
    {"model": "csmcir", "variant": "cirr_clean", "ckpts": [str(CIRR_ROOT / "csmcir/cirr_tuned_clip_arithmetic.pt")]},
    {"model": "qure", "variant": "cirr_vitl_repro", "ckpts": [str(CIRR_ROOT / "qure/qure_cirr_vitl.pth")]},
]


def _run_job(
    spec: Dict[str, Any],
    source_benchmark: str,
    target_dataset: str,
    num_workers: int,
    smoke_only: bool,
    force: bool,
) -> bool:
    model = spec["model"]
    env_spec = ENV_SPECS[model]
    suite_tag = f"{source_benchmark}_to_{target_dataset}"
    out_dir = OUT_ROOT / suite_tag / model
    res_file = out_dir / "test_results.json"

    if not smoke_only and res_file.is_file() and not force:
        print(f"  [SKIP] {suite_tag} / {model} (already done)", flush=True)
        return True

    cmd = [
        env_spec["python"],
        "-u",
        str(WORKER),
        "--model", model,
        "--variant", spec["variant"],
        "--source-benchmark", source_benchmark,
        "--dataset", target_dataset,
        "--num-workers", str(num_workers),
        "--out-dir", str(out_dir),
        "--dataset-type-ids", spec.get("dataset_type_ids", "5"),
        "--ckpts", *spec["ckpts"],
    ]
    if smoke_only:
        cmd.append("--smoke-only")

    env = dict(os.environ, PYTHONPATH=env_spec["pythonpath"])
    t0 = time.time()
    if smoke_only:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        dt = time.time() - t0
        if proc.returncode == 0:
            last = [l for l in proc.stdout.splitlines() if l.strip()][-1]
            print(f"  [PASS] {source_benchmark:<9} {model:<10} ({dt:4.1f}s) | {last}", flush=True)
            return True
        print(f"  [FAIL] {source_benchmark:<9} {model:<10} ({dt:4.1f}s) rc={proc.returncode}", flush=True)
        print("    STDERR:", proc.stderr[-700:], flush=True)
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "eval.log"
    print(f"  [START] {suite_tag:<22} / {model:<10} -> {log_file}", flush=True)
    with log_file.open("w", encoding="utf-8") as lf:
        proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    if proc.returncode == 0 and res_file.is_file():
        print(f"  [DONE ] {suite_tag:<22} / {model:<10} in {dt/60:.1f} min", flush=True)
        return True
    print(f"  [FAIL ] {suite_tag:<22} / {model:<10} rc={proc.returncode} ({dt/60:.1f}m) see {log_file}", flush=True)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Run cross-benchmark CIR transfer evaluations")
    parser.add_argument("--suites", type=str, default="fiq_to_pinpoint,cirr_to_edir,fiq_to_edir")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--verify-only", action="store_true", help="GPU smoke-verify all 24 checkpoints first")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.verify_only:
        print("=== VERIFYING ALL 24 TRANSFER CHECKPOINTS ON GPU (14 FashionIQ + 10 CIRR) ===", flush=True)
        failures = []
        for spec in FIQ_CHECKPOINTS:
            if not _run_job(spec, "fashioniq", "pinpoint", args.num_workers, smoke_only=True, force=True):
                failures.append(f"fiq:{spec['model']}")
        for spec in CIRR_CHECKPOINTS:
            if not _run_job(spec, "cirr", "edir", args.num_workers, smoke_only=True, force=True):
                failures.append(f"cirr:{spec['model']}")
        total = len(FIQ_CHECKPOINTS) + len(CIRR_CHECKPOINTS)
        print(f"\n=== VERIFY SUMMARY: {total - len(failures)}/{total} PASSED | failures={failures} ===", flush=True)
        if failures:
            sys.exit(1)
        return

    suite_list = [s.strip().lower() for s in args.suites.split(",") if s.strip()]
    failures: List[str] = []
    for suite in suite_list:
        print(f"\n==================== SUITE: {suite} ====================", flush=True)
        if suite == "fiq_to_pinpoint":
            items, src, tgt = FIQ_CHECKPOINTS, "fashioniq", "pinpoint"
        elif suite == "cirr_to_edir":
            items, src, tgt = CIRR_CHECKPOINTS, "cirr", "edir"
        elif suite == "fiq_to_edir":
            items, src, tgt = FIQ_CHECKPOINTS, "fashioniq", "edir"
        else:
            raise ValueError(f"Unknown suite: {suite}")
        for spec in items:
            ok = _run_job(spec, src, tgt, args.num_workers, smoke_only=False, force=args.force)
            if not ok:
                failures.append(f"{suite}:{spec['model']}")

    print(f"\n[TRANSFER SUITE FINISHED] failures={failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
