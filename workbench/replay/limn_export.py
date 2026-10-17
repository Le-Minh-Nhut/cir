"""Per-query export for LIMN — the entry point that owns the optional capability.

Deliberately separate from :mod:`workbench.replay.limn`, which is the source-faithful native replay
AND the entry point the environment planner traces to decide LIMN's interpreter. Anything imported
from that module becomes a declared requirement of LIMN's environment, so the observer (numpy) and
the canonical result schema (pydantic) must not live there — not even as function-local imports,
because the tracer walks the whole file. This module may import them; nothing traces it.

The exporter never becomes the authority. It calls the same verified model and the same native
dataset that the untouched ``test.py::test`` used, re-derives the ranking, and refuses to publish
unless the aggregate reconstructed from those rankings reproduces the native hit counts exactly.
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.limn import (LIMN_CHECKPOINTS, MODEL, native_category, preflight)
from workbench.replay.limn_observer import ObservedCategory, cross_check_with_native, observe_category
# Deliberately NOT imported here: `publication` pulls the pydantic-backed canonical schema, and this
# module runs under the MODEL interpreter, which is provisioned from the model's own requirements and
# does not contain pydantic. Importing it made the exporter crash with `ModuleNotFoundError: No module
# named 'pydantic'` on a clean LIMN environment, so the aggregate succeeded while every per-query
# export failed (P1). This module now emits plain JSON and the PARENT publishes it -- see
# `limn_observer_io` for the interchange format.

DEFAULT_TOP_K = 200


def observe_and_verify(category: str, checkpoint_root: Path, dataset_root: Path, source_root: Path,
                       top_k: int = DEFAULT_TOP_K, batch_size: int = 32
                       ) -> tuple[float | None, ObservedCategory, tuple[str, ...]]:
    """Run the native test AND the observer on the SAME loaded objects, then cross-check.

    Returns ``(native_r1, observed, mismatches)``. ``mismatches`` empty is the only state in which
    the caller may publish. The native aggregate is never modified to make them agree.
    """
    with native_category(category, checkpoint_root, dataset_root, source_root) as loaded:
        native_test, model, dataset = loaded["native_test"], loaded["model"], loaded["dataset"]
        metrics = dict(native_test.test(SimpleNamespace(local_rank=0, batch_size=batch_size),
                                        model, dataset, category))
        observed = observe_category(category, model, dataset, dataset_root, batch_size=batch_size,
                                    top_k=top_k)
    mismatches = tuple(cross_check_with_native(observed, metrics))
    return metrics.get(f"{category}_r1"), observed, mismatches


def observation_payload(observed: list[ObservedCategory], mismatches: dict[str, tuple[str, ...]],
                        *, run_id: str, top_k: int) -> dict:
    """The interchange between the model interpreter and the publishing interpreter.

    Everything here is plain JSON-serialisable data. It exists so the observer never has to import
    the canonical schema: this side runs in the model's environment, which has torch/numpy but not
    pydantic, and importing pydantic there made every export fail on a clean LIMN environment.
    """
    categories = []
    for observation in observed:
        categories.append({
            "category": observation.category,
            "gallery_size": observation.gallery_size,
            "eligible_count": observation.eligible_count,
            "queries": [
                {
                    "annotation_index": query.annotation_index,
                    "reference_id": query.reference_id,
                    "target_id": query.target_id,
                    "raw_captions": list(query.raw_captions),
                    "model_input_text": query.model_input_text,
                    "target_rank": query.target_rank,
                    "top_results": [{"rank": rank, "image_id": image_id, "score": score}
                                    for rank, image_id, score in query.top_results],
                }
                for query in observation.queries
            ],
        })
    return {
        "artifact_type": "limn_observation_payload",
        "schema_version": 1,
        "model_id": "limn",
        "method_name": "LIMN",
        "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "run_id": run_id,
        "top_k": top_k,
        "upstream_commit": MODEL["upstream_commit_sha"],
        "checkpoint_members": {
            category: {"filename": LIMN_CHECKPOINTS[category]["filename"],
                       "checkpoint_id": LIMN_CHECKPOINTS[category]["checkpoint_id"],
                       "sha256": LIMN_CHECKPOINTS[category]["expected_sha256"]}
            for category in CATEGORIES
        },
        "instrumentation_parity": ("INSTRUMENTATION_PARITY_MISMATCH" if mismatches
                                   else "INSTRUMENTATION_PARITY_VERIFIED"),
        "instrumentation_mismatches": {name: list(items) for name, items in mismatches.items()},
        "categories": categories,
    }


def export(checkpoint_root: Path, dataset_root: Path, source_root: Path, results_root: Path,
           top_k: int = DEFAULT_TOP_K, run_id: str | None = None,
           command_digest: str | None = None, environment_digest: str | None = None,
           runtime_adaptation: str | None = None) -> int:
    """OBSERVE the per-query rankings and write a plain-JSON payload for the parent to publish.

    This function runs under the MODEL interpreter, so it does only what that environment supports:
    load the checkpoint through the native loader, re-derive the rankings, cross-check them against
    the native aggregate, and write the result. It does NOT validate or publish a canonical artifact
    -- that is the workbench interpreter's job (`workbench.replay.limn_publish`), which is where
    pydantic lives.

    ``run_id`` is the authoritative execution identity from the orchestrator. Without it the payload
    is marked as a development run and the publisher will refuse to call it evidence.
    """
    problems = preflight(checkpoint_root, dataset_root, source_root, CATEGORIES)
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        return 1

    observed: list[ObservedCategory] = []
    mismatches: dict[str, tuple[str, ...]] = {}
    for category in CATEGORIES:
        _, observation, category_mismatches = observe_and_verify(
            category, checkpoint_root, dataset_root, source_root, top_k=top_k)
        observed.append(observation)
        if category_mismatches:
            mismatches[category] = category_mismatches

    identity = run_id or f"dev-limn-base_iter0_all_categories-fashioniq_val_split"
    payload = observation_payload(observed, mismatches, run_id=identity, top_k=top_k)
    payload["command_digest"] = command_digest
    payload["environment_digest"] = environment_digest
    payload["runtime_adaptation"] = runtime_adaptation

    results_root.mkdir(parents=True, exist_ok=True)
    staged = results_root / f".observation-{identity}.payload.json"
    staged.write_text(json.dumps(payload), encoding="utf-8")

    # The JSON is the machine-readable contract the orchestrator parses, and it is emitted as ONE
    # compact line so a caller can identify it even if a compiled dependency writes to fd 1.
    print(json.dumps({
        "model_id": "limn", "run_id": identity, "protocol": "fashioniq_val_split", "top_k": top_k,
        "instrumentation_parity": payload["instrumentation_parity"],
        "instrumentation_mismatches": payload["instrumentation_mismatches"],
        "payload": str(staged),
        "queries": sum(len(category["queries"]) for category in payload["categories"]),
    }, separators=(",", ":")))
    print("[PENDING] PC GPU validation: instrumentation parity was checked against this run's own "
          "native aggregate, not against a real target-GPU inference", file=sys.stderr)
    return 0 if not mismatches else 1


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="root the canonical per-query ResultRun is published under")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--run-id", default=None,
                        help="the authoritative execution identity from the orchestrator; without "
                             "it the artifact is a development run and is marked as one")
    parser.add_argument("--command-digest", default=None,
                        help="the invocation's command digest, as recorded by the orchestrator")
    parser.add_argument("--environment-digest", default=None,
                        help="the invocation's environment digest, as recorded by the orchestrator")
    parser.add_argument("--runtime-adaptation", default=None,
                        help="the declared runtime adaptation this run executed under")
    args = parser.parse_args(argv)
    return export(args.checkpoint_root, args.dataset_root, args.source_root,
                  args.results_root, top_k=args.top_k, run_id=args.run_id,
                  command_digest=args.command_digest,
                  environment_digest=args.environment_digest,
                  runtime_adaptation=args.runtime_adaptation)


if __name__ == "__main__":
    raise SystemExit(main())
