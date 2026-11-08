"""Measurement identity must include the scoring path, or a scoring change reads as a repeat.

Two runs of the same model that differ ONLY in scoring implementation are two different
measurements. Before this fix `measurement_identity` keyed on preprocessing/environment/checkpoint/
gallery only, so such a pair hashed to one `group_id` and was labelled `duplicate` -- "the same thing
measured twice" -- which is false and would merge a `optimize=True` contraction with a plain `einsum`.

The live cohort really contains both implementations:
`conesep`/`csmcir`/`ptha_mtst` use `einsum(..., optimize=True)`; `habit`/`hint`/`intent`/`tme` use
plain `einsum`; `limn`/`encoder` use matmul.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend.composition import (
    annotate_duplicate_attempts,
    measurement_identity,
    scoring_implementation,
)

BASELINE = {
    "model_id": "habit", "checkpoint_id": "fiq_n02", "checkpoint_identity": "ck",
    "protocol_id": "circus_fashioniq_release_v1", "subset": "full",
    "gallery_identity_sha256": "g" * 64, "query_count": 6003, "seed": 0,
    "preprocessing": {"id": "p"}, "conditions": ["multimodal"],
    "runtime": {"model_interpreter": {"environment_fingerprint": "env"}},
}


def test_scoring_only_difference_yields_distinct_group_ids():
    """The defect: identical except the scoring implementation -> same group -> false duplicate."""
    first = dict(BASELINE, scoring_module_source_hash="aaaa")
    second = dict(BASELINE, scoring_module_source_hash="bbbb")
    a = measurement_identity(first)
    b = measurement_identity(second)
    assert a["group_id"] != b["group_id"], (
        "two runs differing only in the scoring path must be distinct measurements")
    assert a["confidence"] == "full" and b["confidence"] == "full"


def test_scoring_hash_from_the_recorded_identity_block_is_used():
    """The hash lives in the rank artifact's identity, not the summary; it must still count."""
    summary = dict(BASELINE)
    recorded = {"scoring_module_source_hash": "cccc", "adapter_module_source_hash": "dddd"}
    facts = measurement_identity(summary, recorded)
    assert facts["identity"]["scoring_module_source_hash"] == "cccc"
    assert facts["identity"]["adapter_module_source_hash"] == "dddd"
    assert facts["confidence"] == "full"


def test_an_older_artifact_missing_the_scoring_hash_is_partial_not_duplicate():
    """HABIT `497c25c…` and the four early rows predate the hash: say so, do not guess."""
    older = dict(BASELINE)
    newer = dict(BASELINE, scoring_module_source_hash="eeee")
    annotate_duplicate_attempts([older, newer])
    assert older["duplicate_status"] == "undetermined", (
        "a row whose scoring path cannot be proven must never be called a duplicate")
    assert older["measurement_identity_confidence"] == "partial"
    assert "scoring_module_source_hash" in older["measurement_identity_missing_inputs"]
    assert newer["duplicate_status"] == "unique", (
        "the post-change row is complete and stands alone")


def test_two_old_artifacts_without_the_field_are_undetermined_not_duplicates():
    first = dict(BASELINE)
    second = dict(BASELINE)
    annotate_duplicate_attempts([first, second])
    for row in (first, second):
        assert row["duplicate_status"] == "undetermined"
        assert "scoring_module_source_hash" in row["measurement_identity_missing_inputs"]


def test_a_true_duplicate_is_still_detected():
    first = dict(BASELINE, scoring_module_source_hash="ffff")
    second = dict(BASELINE, scoring_module_source_hash="ffff")
    annotate_duplicate_attempts([first, second])
    assert first["duplicate_status"] == "duplicate"
    assert first["duplicate_attempt_count"] == 2
    assert first["measurement_group_id"] == second["measurement_group_id"]


@pytest.mark.parametrize("model_id,impl", [
    ("conesep", "einsum-optimized"),
    ("csmcir", "einsum-optimized"),
    ("ptha_mtst", "einsum-optimized"),
    ("habit", "einsum"),
    ("hint", "einsum"),
    ("intent", "einsum"),
    ("tme", "einsum"),
    ("limn", "matmul"),
    ("encoder", "matmul"),
])
def test_scoring_impl_is_derived_from_the_adapter_source(model_id, impl):
    result = scoring_implementation(model_id, "h", "h")
    assert result["impl"] == impl, result
    assert result["basis"]


def test_an_unresolvable_model_reports_unknown_rather_than_guessing():
    result = scoring_implementation("released_audit", None, None)
    assert result["impl"] == "unknown"
    assert "no composition adapter module" in result["basis"]
    assert result["scoring_hash_known"] is False


def test_the_recorded_cohort_really_has_two_implementations(tmp_path: Path):
    """Sanity: the two implementations this fix exists for are both present in the registry."""
    impls = {scoring_implementation(model, "h", "h")["impl"]
             for model in ("conesep", "csmcir", "ptha_mtst", "habit", "limn")}
    assert "einsum-optimized" in impls
    assert impls & {"einsum", "matmul"}
    assert len(impls) >= 2, f"the cohort must expose more than one implementation, saw {impls}"
