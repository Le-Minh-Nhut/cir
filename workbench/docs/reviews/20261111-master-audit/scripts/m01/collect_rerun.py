"""Collect the pinned-relation ENCODER rerun's numbers into one evidence file (read-only)."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
RUN = "ce89a9a27c4cceadeaaf30b0"
SUMMARY = REPO / "workbench/artifacts/composition/summary" / f"encoder_fashioniq_{RUN}"
RUNS = REPO / "workbench/artifacts/composition/runs" / RUN

lines: list[str] = [f"== ENCODER pinned-relation run {RUN}", ""]
for path in sorted(SUMMARY.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    lines.append(f"{path.name} run={data['run_id']} status={data['execution_status']} "
                 f"reused={data.get('reused_query_result_count')} cache={data.get('cache_verify_mode')}")
    for key in ("multimodal", "text_only", "image_only"):
        bucket = metrics[key]
        lines.append(f"   {key}: scored={bucket['scored_count']} coverage={bucket['coverage']:.6f} "
                     f"R@10={bucket['recall_at_10']:.4f} R@50={bucket['recall_at_50']:.4f} "
                     f"nDCG={bucket['ndcg_full']:.6f} MRR={bucket['mrr']:.6f}")
    lines.append(f"   composition_gap_ndcg={metrics['composition_gap_ndcg']:.6f} "
                 f"gap_cohort_count={metrics['gap_cohort_count']}")
    lines.append("")

lines.append("== persisted construction state (not in the checkpoint)")
for path in sorted(RUNS.glob("construction_state/*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    lines.append(f"{path.name} shape={data['shape']} seed={data['seed']} sha256={data['sha256']} "
                 f"in_state_dict={data['in_state_dict']}")
lines.append("")

lines.append("== superseded runs, bytes untouched")
for run in ("5dee7425dd46c5f154808aac", "6a528e877c328772e2c7bb28"):
    path = REPO / f"workbench/artifacts/composition/summary/encoder_fashioniq_{run}/full.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    lines.append(f"{run} mm_R@10={data['metrics']['multimodal']['recall_at_10']:.4f} "
                 f"bytes={os.path.getsize(path)}")
lines.append("reason: relation drawn from an unpinned generator")

print("\n".join(lines))
