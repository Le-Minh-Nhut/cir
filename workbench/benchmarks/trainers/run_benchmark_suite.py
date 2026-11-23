#!/usr/bin/env python3
"""Sequential, resumable benchmark runner + live Markdown leaderboard generator
for all 15 CIR models on EDIR and PinPoint canonical splits.

Strictly runs 1 GPU model at a time (per AGENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path("/home/heheboiz/data/cir")
ENVS = REPO_ROOT / "workbench" / "envs"
CIR_ENV_PY = REPO_ROOT / "workbench" / "artifacts" / "environments" / "cir-005ef4e1b02999a5" / "bin" / "python"
TRAINERS = REPO_ROOT / "workbench" / "benchmarks" / "trainers"
DOCS_DIR = REPO_ROOT / "workbench" / "docs" / "reviews" / "20261123-edir-pinpoint-splits"

# Ordered fast-to-heavy so rapid results land first!
MODEL_QUEUE: List[Dict[str, Any]] = [
    {
        "key": "clip4cir",
        "run_id": "clip4cir_rn50x4",
        "result_field": "final_clip4cir_test",
        "python": str(ENVS / "clip4cir310-sm120-pep610/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/CLIP4Cir/src",
        "script": str(TRAINERS / "train_clip4cir.py"),
        "extra_args": ["--clip-epochs", "{epochs}", "--clip-patience", "{patience}"],
    },
    {
        "key": "encoder",
        "run_id": "encoder_vitb32",
        "result_field": "final_encoder_test",
        "python": str(ENVS / "encoder310-adapted/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/ENCODER",
        "script": str(TRAINERS / "train_encoder.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "tgcir",
        "run_id": "tgcir_vitb16",
        "result_field": "final_tgcir_test",
        "python": str(ENVS / "tgcir39-la/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/TGCIR",
        "script": str(TRAINERS / "train_tgcir.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "clvcnet",
        "run_id": "clvcnet_resnet50",
        "result_field": "final_clvcnet_test",
        "python": str(ENVS / "clvc39-la/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/CLVCNet",
        "script": str(TRAINERS / "train_clvcnet.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "dcnet",
        "run_id": "dcnet_resnet50",
        "result_field": "final_dcnet_test",
        "python": str(ENVS / "dcnet310-sm120/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/DCNet",
        "script": str(TRAINERS / "train_dcnet.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "sprc",
        "run_id": "sprc_pretrain",
        "result_field": "final_sprc_test",
        "python": str(ENVS / "sprc310-sm120/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/SPRC/src",
        "script": str(TRAINERS / "train_sprc.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "mtst_ptha",
        "run_id": "mtst_ptha_blip2",
        "result_field": "final_mtst_ptha_test",
        "python": str(CIR_ENV_PY),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/MTST_PTHA/MTST_PTHA/src",
        "script": str(TRAINERS / "train_mtst_ptha.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "hint",
        "run_id": "hint_blip2",
        "result_field": "final_hint_test",
        "python": str(ENVS / "ilearn39-adapted/bin/python"),
        "pythonpath": f"{REPO_ROOT}",
        "script": str(TRAINERS / "train_ilearn.py"),
        "extra_args": ["--model", "hint", "--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "intent",
        "run_id": "intent_blip2",
        "result_field": "final_intent_test",
        "python": str(ENVS / "ilearn39-adapted/bin/python"),
        "pythonpath": f"{REPO_ROOT}",
        "script": str(TRAINERS / "train_ilearn.py"),
        "extra_args": ["--model", "intent", "--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "habit",
        "run_id": "habit_blip2",
        "result_field": "final_habit_test",
        "python": str(ENVS / "ilearn39-adapted/bin/python"),
        "pythonpath": f"{REPO_ROOT}",
        "script": str(TRAINERS / "train_ilearn.py"),
        "extra_args": ["--model", "habit", "--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "conesep",
        "run_id": "conesep_blip2",
        "result_field": "final_conesep_test",
        "python": str(ENVS / "ilearn39-adapted/bin/python"),
        "pythonpath": f"{REPO_ROOT}",
        "script": str(TRAINERS / "train_ilearn.py"),
        "extra_args": ["--model", "conesep", "--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "airknow",
        "run_id": "airknow_blip2",
        "result_field": "final_airknow_test",
        "python": str(ENVS / "airknow310-la/bin/python"),
        "pythonpath": f"{REPO_ROOT}",
        "script": str(TRAINERS / "train_ilearn.py"),
        "extra_args": ["--model", "airknow", "--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "csmcir",
        "run_id": "csmcir_blip2",
        "result_field": "final_csmcir_test",
        "python": str(CIR_ENV_PY),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/CSMCIR/src",
        "script": str(TRAINERS / "train_csmcir.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "tme",
        "run_id": "tme_blip2",
        "result_field": "final_tme_test",
        "python": str(ENVS / "tme-adapted/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/TME/src",
        "script": str(TRAINERS / "train_tme.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
    {
        "key": "limn",
        "run_id": "limn_vitl14",
        "result_field": "final_limn_test",
        "python": str(ENVS / "limn310-adapted/bin/python"),
        "pythonpath": f"{REPO_ROOT}:{REPO_ROOT}/workbench/third_party/LIMN/LIMN",
        "script": str(TRAINERS / "train_limn.py"),
        "extra_args": ["--num-epochs", "{epochs}", "--patience", "{patience}"],
    },
]


def regenerate_leaderboard(dataset: str, split_tag: str = "70_10_20_seed42") -> Path:
    base = REPO_ROOT / "workbench" / "artifacts" / "benchmark_runs" / dataset / split_tag
    rows = []
    for spec in MODEL_QUEUE:
        res_path = base / spec["run_id"] / "test_results.json"
        if not res_path.is_file():
            continue
        data = json.loads(res_path.read_text(encoding="utf-8"))
        blk = data.get(spec["result_field"])
        if not blk:
            continue
        rows.append((spec["key"], spec["run_id"], data, blk))

    out_path = DOCS_DIR / f"LEADERBOARD_{dataset.upper()}.md"
    lines: List[str] = [f"# Bảng Xếp Hạng & Phân Rã Lỗi Trên {dataset.upper()} (`{split_tag}`)", ""]
    lines.append(f"- **Số model đã hoàn tất:** `{len(rows)} / {len(MODEL_QUEUE)}`")
    lines.append("")

    if dataset == "edir":
        rows.sort(key=lambda x: x[3]["overall"]["Recall@1"], reverse=True)
        lines.append("## 1. Tổng Quan (Full Corpus `28,645`, `1,003` test queries)")
        lines.append("")
        lines.append("| Model | Best Ep | Overall `R@1` | `R@5` | `R@10` | `R@50` | `MRR` | Macro-15 `R@1` |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for key, run_id, data, blk in rows:
            ov = blk["overall"]
            mc = blk["macro_subcategory"]
            ep = data.get("best_epoch", data.get("best_stage2_epoch", "-"))
            lines.append(
                f"| **{key}** (`{run_id}`) | {ep} | **{ov['Recall@1']*100:.2f}%** | {ov['Recall@5']*100:.2f}% | "
                f"{ov['Recall@10']*100:.2f}% | {ov['Recall@50']*100:.2f}% | {ov['MRR']:.4f} | {mc['Recall@1']*100:.2f}% |"
            )
        if rows:
            subcats = sorted(rows[0][3]["per_subcategory"].keys())
            lines.extend(["", "## 2. Ma Trận `Recall@1` Theo 15 Subcategory", ""])
            header = "| Model | " + " | ".join(subcats) + " |"
            sep = "|---|" + "|".join("---:" for _ in subcats) + "|"
            lines.extend([header, sep])
            for key, _, _, blk in rows:
                ps = blk["per_subcategory"]
                vals = [f"{ps[c]['Recall@1']*100:.1f}%" for c in subcats]
                lines.append(f"| **{key}** | " + " | ".join(vals) + " |")
    else:
        rows.sort(key=lambda x: x[3]["overall"]["mAP@10"], reverse=True)
        lines.append("## 1. Tổng Quan (Full Corpus `109,598`, `1,337` test queries)")
        lines.append("")
        lines.append("| Model | Best Ep | `P@1` | `P@10` | `mAP@10` | `NegRecall@10` | `mAP@10_noNeg` | `ling_sens_range` |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for key, run_id, data, blk in rows:
            ov = blk["overall"]
            ep = data.get("best_epoch", data.get("best_stage2_epoch", "-"))
            lines.append(
                f"| **{key}** (`{run_id}`) | {ep} | {ov['precision@1']:.4f} | {ov['precision@10']:.4f} | "
                f"**{ov['mAP@10']:.4f}** | {ov['NegRecall@10']:.4f} | {ov['mAP@10_noNeg']:.4f} | {ov['ling_sens_range']:.4f} |"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all 15 CIR models sequentially on EDIR or PinPoint")
    parser.add_argument("--dataset", type=str, required=True, choices=["edir", "pinpoint"])
    parser.add_argument("--split-tag", type=str, default="70_10_20_seed42")
    parser.add_argument("--models", type=str, default="all", help="'all' or comma-separated keys")
    parser.add_argument("--num-epochs", type=int, default=25)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--val-corpus-scope", type=str, default="split", choices=["split", "full"])
    parser.add_argument("--force", action="store_true", help="Re-run even if test_results.json exists")
    args = parser.parse_args()

    wanted = None if args.models == "all" else {m.strip().lower() for m in args.models.split(",") if m.strip()}
    selected = [s for s in MODEL_QUEUE if wanted is None or s["key"] in wanted]
    base_dir = REPO_ROOT / "workbench" / "artifacts" / "benchmark_runs" / args.dataset / args.split_tag

    print(f"[SUITE] dataset={args.dataset} models={[s['key'] for s in selected]} epochs={args.num_epochs}", flush=True)
    failures: List[str] = []

    for idx, spec in enumerate(selected, 1):
        run_dir = base_dir / spec["run_id"]
        res_file = run_dir / "test_results.json"
        if res_file.is_file() and not args.force:
            print(f"[{idx}/{len(selected)}] SKIP {spec['key']} (already completed: {res_file})", flush=True)
            regenerate_leaderboard(args.dataset, args.split_tag)
            continue

        run_dir.mkdir(parents=True, exist_ok=True)
        log_file = run_dir / "train.log"
        extra = [
            a.format(epochs=args.num_epochs, patience=args.patience)
            for a in spec["extra_args"]
        ]
        cmd = [
            spec["python"],
            "-u",
            spec["script"],
            "--dataset",
            args.dataset,
            "--split-tag",
            args.split_tag,
            "--num-workers",
            str(args.num_workers),
            "--val-corpus-scope",
            args.val_corpus_scope,
            "--run-dir",
            str(run_dir),
            *extra,
        ]
        env = dict(os.environ, PYTHONPATH=spec["pythonpath"])
        t0 = time.time()
        print(f"\n[{idx}/{len(selected)}] START {spec['key']} -> log={log_file}", flush=True)
        with log_file.open("w", encoding="utf-8") as lf:
            proc = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        if proc.returncode == 0 and res_file.is_file():
            lb = regenerate_leaderboard(args.dataset, args.split_tag)
            print(f"[{idx}/{len(selected)}] DONE  {spec['key']} in {dt/60:.1f} min -> updated {lb.name}", flush=True)
        else:
            print(f"[{idx}/{len(selected)}] FAIL  {spec['key']} rc={proc.returncode} ({dt/60:.1f} min) — see {log_file}", flush=True)
            failures.append(spec["key"])

    lb = regenerate_leaderboard(args.dataset, args.split_tag)
    print(f"\n[SUITE FINISHED] leaderboard={lb} failures={failures}", flush=True)
    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
