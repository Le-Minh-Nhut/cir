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

Which metric block holds a condition's numbers is read from `metrics.gap_conditions` (the artifact
names its own triple from d16a1ba onwards) and the ROLE blocks are positional, so a run whose image
role is an adapter ablation (`clvc_image_encoder_only`) is handled; a summary that predates that field
uses the protocol triple and says so. A condition the mapping cannot place is REFUSED by name, and
where the summary already carries `recall_at_1` this tool CHECKS it rather than inventing a second
number: `already_present` / `agrees_with_summary` are recorded and a disagreement refuses the run.

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
#: The three ROLE blocks `CompositionResult` uses, in `compose`'s positional order. The blocks are
#: named by role, not by condition, so the condition behind a block has to be read from the artifact:
#: a run whose gap triple resolved to `clvc_image_encoder_only` keeps its metrics in
#: `metrics["image_only"]`, and a name-keyed map would look for `metrics["clvc_image_encoder_only"]`,
#: find nothing, and refuse a run that is perfectly well formed.
ROLE_BLOCKS = ("multimodal", "text_only", "image_only")
#: What the block of each protocol condition is. Used ONLY when the summary predates
#: `metrics.gap_conditions` (written from d16a1ba onwards); the artifact is the authority whenever it
#: can name its own triple.
LEGACY_METRIC_BLOCKS = {"multimodal": "multimodal", "text_only_black_image": "text_only",
                        "image_only_empty_text": "image_only"}

def _role_mapping(summary: dict) -> tuple[dict[str, str], str, list[str]]:
    """Map each scored condition to the metric block that holds its numbers.

    Returns `(mapping, source, problems)`. `source` is `"gap_conditions"` when the artifact names its
    own triple, `"legacy_protocol_triple"` when it predates that field.

    The mapping is read positionally from `metrics.gap_conditions`, because that is exactly how
    `compose` assigns the roles. It is then CROSS-CHECKED against each block's own `condition` field:
    a triple that says one name and a block that says another means the artifact and the reader
    disagree about which condition a number belongs to, and that is not something to average over.
    """
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), dict) else {}
    gap_conditions = metrics.get("gap_conditions")
    problems: list[str] = []
    if isinstance(gap_conditions, (list, tuple)) and len(gap_conditions) == len(ROLE_BLOCKS) \
            and len(set(gap_conditions)) == len(ROLE_BLOCKS):
        mapping = dict(zip(gap_conditions, ROLE_BLOCKS))
        for condition, block_name in mapping.items():
            block = metrics.get(block_name)
            if not isinstance(block, dict):
                problems.append(f"metrics.gap_conditions names {condition!r} in the {block_name!r} "
                                f"role but the summary has no such block")
                continue
            recorded = block.get("condition")
            if recorded is not None and recorded != condition:
                problems.append(f"metrics.{block_name}.condition is {recorded!r} but "
                                f"metrics.gap_conditions puts {condition!r} in that role")
        return mapping, "gap_conditions", problems
    if gap_conditions is not None:
        problems.append(f"metrics.gap_conditions is present but is not three distinct names: "
                        f"{gap_conditions!r}")
        return {}, "invalid_gap_conditions", problems
    return dict(LEGACY_METRIC_BLOCKS), "legacy_protocol_triple", problems

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

    mapping, mapping_source, mapping_problems = _role_mapping(summary)
    problems: list[str] = list(mapping_problems)
    listed = list(summary.get("conditions") or list(table))
    # Refuse an unplaceable condition BY NAME. Skipping it would publish a record that silently
    # covers fewer conditions than the run, which reads exactly like a run that measured fewer.
    unplaced = [condition for condition in listed if condition not in mapping]
    if unplaced:
        raise SystemExit(
            f"{model_id}/{attempt}: cannot place condition(s) {unplaced} in a metric block; "
            f"role mapping came from {mapping_source} = {mapping}. Refusing rather than skipping a "
            f"condition the run scored")

    conditions = {}
    for condition in listed:
        block = (summary.get("metrics") or {}).get(mapping.get(condition, "")) or {}
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
        # The artifact may ALREADY carry R@1 (every run from the metrics.py change onwards does).
        # Then this tool is not deriving a number, it is CHECKING one, and it must say so: a derived
        # file that contradicts the artifact is worse than no derived file.
        published_r1 = block.get("recall_at_1")
        already_present = "recall_at_1" in block
        if already_present:
            if published_r1 is None or derived.recall_at_1 is None:
                if published_r1 != derived.recall_at_1:
                    problems.append(f"{condition}.recall_at_1: recomputed {derived.recall_at_1!r} vs "
                                    f"published {published_r1!r}")
            elif abs(float(derived.recall_at_1) - float(published_r1)) > 1e-12:
                problems.append(f"{condition}.recall_at_1: recomputed {derived.recall_at_1!r} vs "
                                f"published {published_r1!r}")
        conditions[condition] = {
            "recall_at_1": derived.recall_at_1,
            "recall_at_10": derived.recall_at_10,
            "recall_at_50": derived.recall_at_50,
            "query_count": derived.query_count,
            "scored_count": derived.scored_count,
            "coverage": derived.coverage,
            "rank_1_count": hits,
            "unavailable_reason": derived.unavailable_reason,
            "metric_block": mapping[condition],
            "already_present": already_present,
            "summary_recall_at_1": published_r1,
            "agrees_with_summary": (None if not already_present else
                                    (derived.recall_at_1 is None and published_r1 is None)
                                    or (derived.recall_at_1 is not None and published_r1 is not None
                                        and abs(float(derived.recall_at_1)
                                                - float(published_r1)) <= 1e-12)),
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
                       "R@1 <= R@10 <= R@50 holds per condition; where the summary already carried "
                       "recall_at_1, the derived value equals it to 1e-12",
        "role_mapping": {"source": mapping_source, "mapping": mapping,
                         "explanation": (
                             "condition -> CompositionResult block, read positionally from "
                             "metrics.gap_conditions; the blocks are named by ROLE, so this is what "
                             "says which condition filled the image role"
                             if mapping_source == "gap_conditions" else
                             "the summary predates metrics.gap_conditions, so the protocol triple's "
                             "fixed condition -> block mapping was assumed")},
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
