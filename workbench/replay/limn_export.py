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
from workbench.replay.publication import publish_observed_categories

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


def export(checkpoint_root: Path, dataset_root: Path, source_root: Path, results_root: Path,
           top_k: int = DEFAULT_TOP_K) -> int:
    """Export the canonical per-query artifact for the FULL three-category bundle."""
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

    status = {
        "model_id": "limn", "protocol": "fashioniq_val_split", "top_k": top_k,
        "categories": list(CATEGORIES),
        "instrumentation_parity": ("INSTRUMENTATION_PARITY_MISMATCH" if mismatches
                                   else "INSTRUMENTATION_PARITY_VERIFIED"),
        "instrumentation_mismatches": {category: list(items) for category, items in mismatches.items()},
        "publication": ("WITHHELD_ON_PARITY_MISMATCH" if mismatches else "PENDING_PC_GPU_VALIDATION"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if mismatches:
        # The native aggregate is untouched and already recorded by the aggregate path; the
        # per-query artifact is withheld because it does not reproduce it.
        for category, items in sorted(mismatches.items()):
            for item in items:
                print(f"[BLOCKED] QUERY_EXPORT_PARITY_MISMATCH {category}: {item}", file=sys.stderr)
        print(json.dumps(status, indent=2))
        return 1

    checkpoint = LIMN_CHECKPOINTS["dress"]
    aggregate = {"macro_r10": None, "macro_r50": None, "macro_mean": None}
    r1_by_category = {observation.category: observation.recall(1) for observation in observed}
    # The macro aggregate is the arithmetic mean of the per-category values, which is what
    # `macro_aggregate` computes for the native results; derived here from the observed rankings so
    # the artifact never carries a number no ranking supports.
    r10 = sum(observation.recall(10) for observation in observed) / len(observed)
    r50 = sum(observation.recall(50) for observation in observed) / len(observed)
    aggregate.update({"macro_r10": r10, "macro_r50": r50, "macro_mean": (r10 + r50) / 2})
    del r1_by_category

    destination = publish_observed_categories(
        observed, results_root, parity_mismatches=mismatches,
        model_id="limn", method_name="LIMN", checkpoint_id="base_iter0_all_categories",
        protocol_id="fashioniq_val_split", top_k_saved=top_k,
        source_commit=MODEL["upstream_commit_sha"], checkpoint_sha256=checkpoint["expected_sha256"],
        checkpoint_source=checkpoint.get("source"), data_kind="experiment",
        reproduced_metrics=aggregate,
    )
    print(json.dumps({**status, "published": str(destination)}, indent=2))
    print("[PENDING] PC GPU validation: instrumentation parity was checked against this run's own "
          "native aggregate, not against a real target-GPU inference")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True,
                        help="root the canonical per-query ResultRun is published under")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args(argv)
    return export(args.checkpoint_root, args.dataset_root, args.source_root,
                  args.results_root, top_k=args.top_k)


if __name__ == "__main__":
    raise SystemExit(main())
