#!/usr/bin/env python3
"""How much of a run's three-condition gap rests on caption-derived routing? CPU only, no re-run.

Why this exists: CLVC-Net's category routing is `loader assignment only` for the queries the release
loader scoped itself, and `reference stem + edit caption` resolved against the pinned caption
annotation for the queries it left unscoped (117 of 6003). Under the conditions that remove the edit
-- `text_only_black_image`, `clvc_image_encoder_only` -- those 117 still had their category chosen
from the caption text. This tool quantifies the consequence from the ranks already on disk:

* cohort A -- every scored query: must reproduce the artifact's `metrics.composition_gap_ndcg` to
  1e-12, or the tool stops instead of reporting a second, unexplained number.
* cohort B -- the loader-scoped queries only: same formula, same three conditions, same rank table.
  This is the caption-free cohort, because a loader assignment never reads the edit text.

Cohort membership is NOT assumed from qid ranges or from the published aggregate count. It is the
resolver replayed on CPU (`model_limn.resolve_query_scope`, the same function the run called) against
the pinned annotation, and the replayed totals are CROSS-CHECKED against the artifact's
`adapter_query_scope_resolution`. A replay that disagrees with the run refuses the derivation: a
cohort computed from a routing rule the run did not use is not the run's cohort.

Output is a NEW file under `<artifacts>/composition/derived/category_routing_inputs/`. A published
summary is never written: published artifacts are immutable, and this tool records the digests of
every file it read instead.

Usage:
  python workbench/scripts/derive_routing_cohort_gap.py --run clvc_net:da0a2162be0e159a1a456cec
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.composition.metrics import compose

SUBSET = "full"
#: `resolve_query_scope`'s reason for a query the release loader scoped itself. Everything else
#: (`annotation:<category>`, `unresolved:<why>`) was decided with the edit caption as a key.
LOADER_REASON = "loader"
#: The resolver's key, spelled once so the declaration and the number cannot describe two rules.
RESOLVER_KEY = "reference_stem+edit_caption"

def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifacts-root", type=Path, default=None,
                        help="artifacts root holding composition/{runs,summary} "
                             "(default: WORKBENCH_ARTIFACTS_ROOT)")
    parser.add_argument("--run", action="append", required=True, metavar="MODEL_ID:RUN_ID",
                        help="repeatable; the ranks file decides the attempt and the summary")
    parser.add_argument("--output", type=Path, default=None,
                        help="output JSON (default: <artifacts>/composition/derived/"
                             "category_routing_inputs/<model>_<attempt>_<subset>.json)")
    return parser

def _artifacts_root(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.expanduser().resolve()
    from workbench.backend.operator_config import resolve_config

    return Path(resolve_config().WORKBENCH_ARTIFACTS_ROOT).expanduser().resolve()

def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

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

def routing_reasons(model_id: str, release_root: Path | None = None,
                    fashioniq_root: Path | None = None) -> tuple[dict[str, str], dict]:
    """Replay the adapter's routing on CPU: qid -> reason, plus the annotation provenance.

    The SAME module-level resolver the adapter calls, against the same pinned annotation, so the
    reasons are the ones the run acted on. No worker, no checkpoint, no GPU: routing reads
    annotations only.

    ``release_root`` / ``fashioniq_root`` default to the operator config. They are parameters because
    the test suite deliberately runs with no operator file (a fresh clone, `conftest.py`), while
    these tests DO want the staged release — the same reason the CPython tests take a config path.
    """
    from importlib import import_module

    from workbench.backend.operator_config import resolve_config
    from workbench.composition.data import (
        ANNOTATION_PIN,
        assign_garment_categories,
        load_release,
        resolved_annotation_pin,
        select_queries,
    )

    config = resolve_config()
    module = import_module(f"workbench.composition.model_{model_id}")
    release = Path(release_root or config.WORKBENCH_CIRCUS_RELEASE_ROOT)
    fashioniq = Path(fashioniq_root or config.FASHIONIQ_ROOT)
    data = assign_garment_categories(load_release(release, verify_hashes=False), fashioniq)
    index = module.caption_scope_index(fashioniq, resolved_annotation_pin(data, ANNOTATION_PIN))
    reasons = {query.qid: module.resolve_query_scope(index, query, data.candidate_names)["reason"]
               for query in select_queries(data, SUBSET)}
    return reasons, index.report()

def derive(composition: Path, model_id: str, run_id: str, *,
           release_root: Path | None = None, fashioniq_root: Path | None = None) -> dict:
    ranks_path = _ranks_file(composition, run_id, model_id)
    ranks_payload = json.loads(ranks_path.read_text(encoding="utf-8"))
    attempt = ranks_payload.get("attempt") or run_id
    summary_path = _summary_file(composition, attempt, SUBSET)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("subset") != SUBSET:
        raise SystemExit(f"{summary_path} is not the {SUBSET} summary")

    table = ranks_payload.get("ranks") or {}
    conditions = tuple(summary.get("metrics", {}).get("gap_conditions")
                       or summary.get("conditions") or ())
    if len(conditions) != 3:
        raise SystemExit(f"{model_id}/{attempt}: the artifact names no usable three-condition "
                         f"triple ({conditions!r}); refusing to guess role order")

    reasons, annotation = routing_reasons(model_id, release_root, fashioniq_root)
    scope_block = summary.get("adapter_query_scope_resolution") or {}
    loader_qids = {qid for qid, reason in reasons.items() if reason == LOADER_REASON}
    resolved_by_caption = {qid for qid, reason in reasons.items()
                           if reason.startswith("annotation:")}
    unresolved_qids = {qid for qid, reason in reasons.items()
                       if reason.startswith("unresolved:")}
    unscoped = set(scope_block.get("unscoped_qids") or ())
    problems: list[str] = []
    for name, mine, artifact in (
            ("loader_scoped", len(loader_qids), scope_block.get("loader_scoped")),
            ("adapter_routed", len(loader_qids) + len(resolved_by_caption),
             scope_block.get("adapter_routed")),
            ("still_unscoped", len(unresolved_qids), scope_block.get("still_unscoped"))):
        if isinstance(artifact, int) and mine != artifact:
            problems.append(f"replayed {name} {mine} != artifact's {artifact}")
    if unscoped and unresolved_qids != unscoped:
        # The run's own list of unscored qids vs the replay's: this is the routing RULE agreeing or
        # not, not an aggregate that can coincide by accident.
        problems.append(f"replayed unresolved qids {sorted(unresolved_qids)} != artifact's "
                        f"unscoped_qids {sorted(unscoped)}")
    if problems:
        raise SystemExit(f"{model_id}/{attempt}: routing replay does not match the artifact "
                         f"({problems}); refusing to derive a cohort from another rule")

    full = compose({condition: table.get(condition, {}) for condition in conditions},
                   query_count=summary.get("query_count"), conditions=conditions)
    published = summary.get("metrics", {}).get("composition_gap_ndcg")
    if full.composition_gap_ndcg is None or published is None:
        raise SystemExit(f"{model_id}/{attempt}: the artifact has no gap to reproduce "
                         f"({published!r}); nothing to check cohort A against")
    if abs(full.composition_gap_ndcg - float(published)) > 1e-12:
        raise SystemExit(f"{model_id}/{attempt}: cohort A reproduces {full.composition_gap_ndcg!r} "
                         f"but the artifact published {published!r}; refusing to publish a second "
                         f"number from a rank table that no longer matches its summary")

    scoped = {condition: {qid: rank for qid, rank in (table.get(condition) or {}).items()
                          if qid in loader_qids} for condition in conditions}
    caption_free = compose(scoped, query_count=len(loader_qids), conditions=conditions)
    return {
        "model_id": model_id, "attempt": attempt, "subset": SUBSET,
        "conditions": list(conditions),
        "gap_conditions": list(full.gap_conditions),
        "inputs": {"ranks": {"path": str(ranks_path), "sha256": _digest(ranks_path)},
                   "summary": {"path": str(summary_path), "sha256": _digest(summary_path)}},
        "annotation": annotation,
        "routing": {
            "resolver_key": RESOLVER_KEY,
            "loader_scoped_count": len(loader_qids),
            "resolver_routed_count": len(resolved_by_caption),
            "still_unscoped_count": len(unresolved_qids),
            "scored_query_count": len(loader_qids) + len(resolved_by_caption),
            "artifact_loader_scoped": scope_block.get("loader_scoped"),
            "artifact_adapter_routed": scope_block.get("adapter_routed"),
            "artifact_still_unscoped": scope_block.get("still_unscoped"),
            "replay_agrees_with_artifact": not problems,
            "resolver_routed_qids": sorted(resolved_by_caption),
        },
        "cohort_a_all_scored": {
            "label": "every query with an exact rank in all three conditions (the PUBLISHED cohort)",
            "query_count": summary.get("query_count"),
            "gap_cohort_count": full.gap_cohort_count,
            "composition_gap_ndcg": full.composition_gap_ndcg,
            "published_gap_ndcg": published,
            "reproduces_published": True,
            "gap_multimodal_ndcg": full.gap_multimodal_ndcg,
            "gap_text_only_ndcg": full.gap_text_only_ndcg,
            "gap_image_only_ndcg": full.gap_image_only_ndcg,
            "mrr_gap": full.mrr_gap,
            "caption_free": False,
        },
        "cohort_b_loader_scoped": {
            "label": "the loader-scoped queries only: routing never read the edit caption",
            "query_count": len(loader_qids),
            "gap_cohort_count": caption_free.gap_cohort_count,
            "composition_gap_ndcg": caption_free.composition_gap_ndcg,
            "gap_multimodal_ndcg": caption_free.gap_multimodal_ndcg,
            "gap_text_only_ndcg": caption_free.gap_text_only_ndcg,
            "gap_image_only_ndcg": caption_free.gap_image_only_ndcg,
            "mrr_gap": caption_free.mrr_gap,
            "gap_unavailable_reason": caption_free.gap_unavailable_reason,
            "caption_free": True,
        },
        "claim_narrowed": (
            f"under {', '.join(conditions)} the category of {len(resolved_by_caption)} of "
            f"{len(loader_qids) + len(resolved_by_caption)} scored queries was chosen from "
            f"(reference stem, edit caption), "
            f"so a condition that removes the edit still used it for routing; cohort B "
            f"({len(loader_qids)} loader-scoped queries) does not"),
    }

def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    composition = _artifacts_root(args.artifacts_root) / "composition"
    for spec in args.run:
        model_id, _, run_id = spec.partition(":")
        record = derive(composition, model_id, run_id)
        destination = args.output or (composition / "derived" / "category_routing_inputs"
                                      / f"{model_id}_{record['attempt']}_{SUBSET}.json")
        destination = Path(destination).expanduser()
        if destination.exists():
            raise SystemExit(f"{destination} already exists; refusing to rewrite a derived artifact "
                             f"(pass --output to choose a new path)")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"{model_id}/{record['attempt']}: cohort A gap "
              f"{record['cohort_a_all_scored']['composition_gap_ndcg']!r} over "
              f"{record['cohort_a_all_scored']['gap_cohort_count']} queries -> "
              f"{destination}")
        print(f"{model_id}/{record['attempt']}: cohort B gap "
              f"{record['cohort_b_loader_scoped']['composition_gap_ndcg']!r} over "
              f"{record['cohort_b_loader_scoped']['gap_cohort_count']} queries")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
