#!/usr/bin/env python3
"""Derive R@1 for COMPLETED Full runs from the exact ranks already on disk. No GPU, no re-ranking.

Why this exists: `ConditionMetrics` shipped without `recall_at_1`, so the published Full runs cannot
supply it and its absence is not a zero (98 §3, FINAL_REPORT §5). Every number this tool prints is a
function of `runs/<run_id>/ranks/*.json`, which is the same table the published R@10/R@50 came from
-- nothing is re-encoded and nothing is re-scored.

Three properties it asserts rather than assumes:

1. The stored summary is REPRODUCIBLE from the rank table. R@10/R@50/scored_count/coverage are
   recomputed with the same `metrics.compose` the runner used and compared bit-for-bit; a mismatch
   refuses the backfill instead of publishing an R@1 that does not belong beside those numbers.
2. R@1 <= R@10 <= R@50 per condition, and R@1 is `hit_at_k(rank, 1)` (rank exactly 1).
3. A condition with no scorable ranks yields `null` with a reason, never 0.

The output is a NEW file. An existing `summary/**/full.json` is refused by name, because published
artifacts are immutable.

Usage:
  python workbench/scripts/backfill_recall_at_1.py \
      --run tgcir:7b1fa85007bffdab931f7f19 --run clvc_net:eab10106b2df8e97fb7a5ce5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.composition.metrics import compute_condition_metrics

SUBSET = "full"
METRIC_BLOCKS = {"multimodal": "multimodal", "text_only_black_image": "text_only",
                 "image_only_empty_text": "image_only"}

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-root", type=Path, default=None,
                        help="artifacts root holding composition/{runs,summary} "
                             "(default: WORKBENCH_ARTIFACTS_ROOT)")
    parser.add_argument("--run", action="append", required=True, metavar="MODEL_ID:RUN_ID",
                        help="repeatable; the ranks file decides the attempt and the summary")
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON (default: <artifacts>/composition/derived/recall_at_1/"
                             "<model>_<attempt>_<subset>.json)")
    return parser

def _artifacts_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    from workbench.backend.operator_config import resolve_config

    return Path(resolve_config().WORKBENCH_ARTIFACTS_ROOT).expanduser().resolve()

def _ranks_file(composition: Path, run_id: str, model_id: str) -> Path:
    matches = sorted((composition / "runs" / run_id / "ranks").glob(f"{model_id}_*.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one ranks file for {model_id}/{run_id}, "
                         f"found {[path.name for path in matches]}")
    return matches[0]

def _summary_file(composition: Path, attempt: str, subset: str) -> Path:
    matches = sorted(composition.glob(f"summary/*_{attempt}/{subset}.json"))
    if len(matches) != 1:
        raise SystemExit(f"expected exactly one {subset} summary for attempt {attempt}, "
                         f"found {[str(path) for path in matches]}")
    return matches[0]

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def derive(composition: Path, model_id: str, run_id: str) -> dict:
    ranks_path = _ranks_file(composition, run_id, model_id)
    ranks_payload = json.loads(ranks_path.read_text(encoding="utf-8"))
    attempt = ranks_payload.get("attempt") or run_id
    summary_path = _summary_file(composition, attempt, SUBSET)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("subset") != SUBSET:
        raise SystemExit(f"{summary_path} is not the {SUBSET} summary")

    table = ranks_payload.get("ranks") or {}
    qids = sorted({qid for condition in table.values() for qid in condition})
    if summary.get("query_count") != len(qids):
        # The rank table holds a rank (or None) for EVERY query, so its qid set is the cohort the
        # published metrics used. A different count means the two do not describe one cohort.
        raise SystemExit(f"{model_id}: rank table covers {len(qids)} qids but the summary declares "
                         f"query_count {summary.get('query_count')}; refusing to compare cohorts")

    conditions = {}
    problems = []
    for condition in summary.get("conditions") or list(table):
        block = (summary.get("metrics") or {}).get(METRIC_BLOCKS.get(condition, condition)) or {}
        if condition not in table:
            conditions[condition] = {
                "recall_at_1": None, "query_count": summary.get("query_count"), "scored_count": 0,
                "coverage": 0.0,
                "unavailable_reason": "the rank artifact holds no table for this condition"}
            continue
        derived = compute_condition_metrics(condition, table[condition])
        checks = {
            "scored_count": (derived.scored_count, block.get("scored_count")),
            "coverage": (derived.coverage, block.get("coverage")),
            "recall_at_10": (derived.recall_at_10, block.get("recall_at_10")),
            "recall_at_50": (derived.recall_at_50, block.get("recall_at_50")),
        }
        for name, (mine, published) in checks.items():
            if mine is None or published is None:
                if mine != published:
                    problems.append(f"{condition}.{name}: recomputed {mine!r} vs published {published!r}")
            elif abs(float(mine) - float(published)) > 1e-12:
                problems.append(f"{condition}.{name}: recomputed {mine!r} vs published {published!r}")
        if derived.recall_at_1 is not None:
            for higher in (derived.recall_at_10, derived.recall_at_50):
                if higher is not None and derived.recall_at_1 > higher + 1e-12:
                    problems.append(f"{condition}: R@1 {derived.recall_at_1} > R@10/R@50 {higher}")
        hits = sum(1 for rank in table[condition].values() if rank == 1)
        conditions[condition] = {
            "recall_at_1": derived.recall_at_1,
            "recall_at_10": derived.recall_at_10,
            "recall_at_50": derived.recall_at_50,
            "query_count": derived.query_count,
            "scored_count": derived.scored_count,
            "coverage": derived.coverage,
            "rank_1_count": hits,
            "unavailable_reason": derived.unavailable_reason,
        }
    if problems:
        raise SystemExit(f"{model_id}/{attempt}: summary is not reproducible from the rank table, "
                         f"refusing the backfill:\n  " + "\n  ".join(problems))

    return {
        "run_id": run_id, "model_id": model_id, "attempt": attempt, "subset": SUBSET,
        "query_count": summary.get("query_count"),
        "condition_status": ranks_payload.get("condition_status"),
        "metric": "recall_at_1",
        "definition": "100 * #{queries whose exact target rank == 1} / scored_count; same scored "
                      "cohort as R@10/R@50 on the same rank table",
        "consistency": "recomputed R@10/R@50/scored_count/coverage equal the published summary; "
                       "R@1 <= R@10 <= R@50 holds per condition",
        "conditions": conditions,
        # Which published files this was derived from, and their bytes, so a reader can tell the
        # derivation covers the artifact they are looking at.
        "derived_from": {"ranks": {"path": str(ranks_path), "sha256": _digest(ranks_path)},
                         "summary": {"path": str(summary_path), "sha256": _digest(summary_path)}},
    }

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    composition = _artifacts_root(args.artifacts_root) / "composition"
    derived = []
    for spec in args.run:
        model_id, separator, run_id = spec.partition(":")
        if not separator or not run_id:
            raise SystemExit(f"--run expects MODEL_ID:RUN_ID, got {spec!r}")
        derived.append(derive(composition, model_id, run_id))

    if args.output is None:
        if len(derived) == 1:
            record = derived[0]
            output = (composition / "derived" / "recall_at_1" /
                      f"{record['model_id']}_{record['attempt']}_{SUBSET}.json")
        else:
            names = "+".join(sorted(f"{record['model_id']}_{record['attempt']}" for record in derived))
            output = composition / "derived" / "recall_at_1" / f"{names}_{SUBSET}.json"
    else:
        output = args.output.expanduser().resolve()
    if output.name in {f"{subset}.json" for subset in ("full", "sf", "validated")} and "summary" in output.parts:
        raise SystemExit(f"refusing to write into a published summary: {output}")
    payload = derived[0] if len(derived) == 1 else {"runs": derived}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "runs": [
        {"model_id": record["model_id"], "attempt": record["attempt"],
         "conditions": {condition: {"recall_at_1": block["recall_at_1"],
                                    "scored_count": block["scored_count"],
                                    "rank_1_count": block.get("rank_1_count")}
                        for condition, block in record["conditions"].items()}}
        for record in derived]}, indent=2, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
