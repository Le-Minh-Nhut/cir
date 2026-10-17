"""Per-query artifact: category gallery sizing, atomic publication, and tamper detection.

Covers three things the canonical exporter must get right, each with a positive control so a system
that rejects everything cannot pass:

* §16 category-specific gallery sizes — a dress rank is bounded by the dress gallery, not by the
  largest category's;
* §17/§18 run-scoped atomic publication and independent validation;
* §30 tamper resistance — the verifier must reject a rewritten artifact, INCLUDING the case where
  the rewrite is otherwise schema-valid.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from workbench.backend.schemas.results import ResultRun
from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.replay.publication import ParityMismatch, build_result_run, publish, publish_observed_categories, run_id_for


def _query(category: str, index: int, gallery_size: int, target_rank: int, top_depth: int) -> ObservedQuery:
    reference, target = f"{category}-ref-{index}", f"{category}-target-{index}"
    top = []
    for rank in range(1, top_depth + 1):
        if rank == target_rank:
            image_id = target
        elif rank == 1:
            image_id = "distractor-1"
        else:
            image_id = f"distractor-{index}-{rank}"
        top.append((rank, image_id, round(1.0 - rank / (gallery_size + 10), 6)))
    return ObservedQuery(
        annotation_index=index, reference_id=reference, target_id=target,
        raw_captions=("change one", "detail two"), model_input_text="change one and detail two",
        target_rank=target_rank, top_results=tuple(top), category=category,
    )


def _category(category: str, gallery_size: int, count: int = 3, target_rank: int = 2,
              top_depth: int = 200) -> ObservedCategory:
    return ObservedCategory(
        category=category, gallery_size=gallery_size, eligible_count=gallery_size - 1,
        queries=tuple(_query(category, index, gallery_size, target_rank, top_depth) for index in range(count)),
    )


def _identity(**overrides):
    base = dict(model_id="limn", method_name="LIMN", checkpoint_id="base_iter0_all_categories",
                protocol_id="fashioniq_val_split", top_k_saved=200, source_commit="a" * 40,
                checkpoint_sha256="b" * 64, data_kind="mock")
    base.update(overrides)
    return base


def test_category_gallery_sizes_are_per_category_not_a_single_scalar() -> None:
    """The scalar could not express this: dress is smaller than toptee. §16."""
    run = build_result_run([_category("dress", 100, top_depth=100),
                            _category("shirt", 150, top_depth=150),
                            _category("toptee", 200, top_depth=200)], **_identity())
    assert run.run.category_gallery_sizes == {"dress": 100, "shirt": 150, "toptee": 200}
    # The scalar remains the declared upper bound so old readers keep a sane value.
    assert run.run.gallery_size == 200
    assert run.run.category_eligible_counts == {"dress": 99, "shirt": 149, "toptee": 199}


def test_a_dress_rank_beyond_the_dress_gallery_is_refused() -> None:
    """The regression this guards: a toptee-sized dress rank would pass a global-only check."""
    with pytest.raises(ValueError, match="exceeds its category gallery size"):
        build_result_run([_category("dress", 100, target_rank=150)], **_identity())


def test_category_sizes_must_agree_with_the_scalar() -> None:
    run = build_result_run([_category("dress", 100, top_depth=100), _category("toptee", 200)], **_identity())
    payload = run.model_dump(mode="json")
    payload["run"]["gallery_size"] = 999  # no longer the max of the per-category map
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_a_category_missing_from_the_size_map_is_refused() -> None:
    run = build_result_run([_category("dress", 100, top_depth=100), _category("toptee", 200)], **_identity())
    payload = run.model_dump(mode="json")
    payload["run"]["category_gallery_sizes"] = {"dress": 100}  # toptee dropped
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_schema_v2_without_the_new_field_still_validates() -> None:
    """Backward compatibility: a pre-existing artifact has no per-category map and must load."""
    run = build_result_run([_category("dress", 200)], **_identity(top_k_saved=200))
    payload = run.model_dump(mode="json")
    del payload["run"]["category_gallery_sizes"]
    del payload["run"]["category_eligible_counts"]
    reloaded = ResultRun.model_validate(payload)
    assert reloaded.run.category_gallery_sizes is None
    assert reloaded.run.gallery_size == 200


def test_publication_is_run_scoped_and_never_overwrites_another_run(tmp_path: Path) -> None:
    first = build_result_run([_category("dress", 100, top_depth=100)], **_identity(checkpoint_sha256="c" * 64))
    second = build_result_run([_category("dress", 100, top_depth=100)], **_identity(checkpoint_sha256="d" * 64))
    assert first.run.run_id != second.run.run_id
    path_a = publish(first, tmp_path)
    path_b = publish(second, tmp_path)
    assert path_a != path_b
    assert path_a.exists() and path_b.exists()
    # Both are independently loadable — publication did not clobber a sibling run.
    assert ResultRun.model_validate_json(path_a.read_text()).run.checkpoint_sha256 == "c" * 64
    assert ResultRun.model_validate_json(path_b.read_text()).run.checkpoint_sha256 == "d" * 64


def test_publication_path_is_protocol_scoped(tmp_path: Path) -> None:
    run = build_result_run([_category("dress", 100, top_depth=100)], **_identity())
    path = publish(run, tmp_path)
    assert path.parent.name == "limn"
    assert path.parent.parent.name == "fashioniq_val_split"


def test_rerun_with_identical_identity_is_the_same_run_id() -> None:
    """Determinism: the identity describes the inputs, so a rerun is not a new run."""
    first = run_id_for(model_id="limn", checkpoint_id="base_iter0_dress",
                       protocol_id="fashioniq_val_split", source_commit="a" * 40,
                       checkpoint_sha256="b" * 64)
    second = run_id_for(model_id="limn", checkpoint_id="base_iter0_dress",
                        protocol_id="fashioniq_val_split", source_commit="a" * 40,
                        checkpoint_sha256="b" * 64)
    assert first == second


def test_changed_source_commit_changes_the_run_id() -> None:
    first = run_id_for(model_id="limn", checkpoint_id="c", protocol_id="fashioniq_val_split",
                       source_commit="a" * 40, checkpoint_sha256="b" * 64)
    second = run_id_for(model_id="limn", checkpoint_id="c", protocol_id="fashioniq_val_split",
                        source_commit="9" * 40, checkpoint_sha256="b" * 64)
    assert first != second


def test_tampered_target_rank_is_detected(tmp_path: Path) -> None:
    """A rewritten rank that no longer matches the saved Top-K must not validate. §30."""
    run = build_result_run([_category("dress", 300, target_rank=2, top_depth=200)], **_identity())
    path = publish(run, tmp_path)
    payload = json.loads(path.read_text())
    # Claim the target is rank 1 while slot 1 still holds a distractor.
    payload["queries"][0]["target_rank"] = 1
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_tampered_target_image_id_is_detected() -> None:
    run = build_result_run([_category("dress", 300, target_rank=5, top_depth=200)], **_identity())
    payload = run.model_dump(mode="json")
    payload["queries"][0]["top_results"][4]["image_id"] = "not-the-target"
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_tampered_query_id_is_detected() -> None:
    """query_id is derived from canonical identity, so it cannot be relabelled. §13."""
    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity())
    payload = run.model_dump(mode="json")
    payload["queries"][0]["query_id"] = "dress:0:swapped:ids"
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_swapped_protocol_is_detected() -> None:
    """The protocol determines the literature label; a swap that contradicts it is refused."""
    run = build_result_run([_category("dress", 300)], **_identity())
    payload = run.model_dump(mode="json")
    payload["run"]["protocol_id"] = "fashioniq_original_split"  # label still "val"
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_truncated_top_results_is_detected() -> None:
    run = build_result_run([_category("dress", 300, target_rank=5, top_depth=200)], **_identity())
    payload = run.model_dump(mode="json")
    payload["queries"][0]["top_results"] = payload["queries"][0]["top_results"][:100]
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_non_contiguous_ranks_are_detected() -> None:
    run = build_result_run([_category("dress", 300, target_rank=5, top_depth=200)], **_identity())
    payload = run.model_dump(mode="json")
    payload["queries"][0]["top_results"][3]["rank"] = 99
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_duplicate_query_ids_are_detected() -> None:
    run = build_result_run([_category("dress", 300, count=2, target_rank=5)], **_identity())
    payload = run.model_dump(mode="json")
    payload["queries"][1]["query_id"] = payload["queries"][0]["query_id"]
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_unchanged_valid_artifact_remains_accepted(tmp_path: Path) -> None:
    """POSITIVE CONTROL: the tamper checks above must not be a system that rejects everything."""
    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity())
    path = publish(run, tmp_path)
    reloaded = ResultRun.model_validate_json(path.read_text())
    assert reloaded.run.run_id == run.run.run_id
    assert len(reloaded.queries) == len(run.queries)


def test_unknown_extra_field_is_refused() -> None:
    """extra='forbid' means a smuggled field cannot ride along unnoticed."""
    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity())
    payload = run.model_dump(mode="json")
    payload["run"]["injected"] = "evil"
    with pytest.raises(ValidationError):
        ResultRun.model_validate(payload)


def test_parity_mismatch_blocks_publication(tmp_path: Path) -> None:
    """PARITY 1 failure: the native aggregate stands and nothing is published. §12."""
    with pytest.raises(ParityMismatch, match="QUERY_EXPORT_PARITY_MISMATCH"):
        publish_observed_categories([_category("dress", 20)], tmp_path,
                                    parity_mismatches=("dress_r1: observer 33.0 != native 34.0",),
                                    **_identity())
    assert list(tmp_path.rglob("*.json")) == []


def test_clean_parity_publishes(tmp_path: Path) -> None:
    path = publish_observed_categories([_category("dress", 20, top_depth=20)], tmp_path,
                                       parity_mismatches=(), **_identity(top_k_saved=20))
    assert path.exists()


def test_synthetic_run_is_labelled_and_never_experiment(tmp_path: Path) -> None:
    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity(data_kind="mock"))
    assert run.run.data_kind == "mock"
    path = publish(run, tmp_path)
    assert json.loads(path.read_text())["run"]["data_kind"] == "mock"


def test_unsupported_protocol_is_refused() -> None:
    with pytest.raises(ValueError, match="unsupported protocol_id"):
        build_result_run([_category("dress", 300)], **_identity(protocol_id="fashioniq_invented"))


def test_a_relabelled_run_id_is_refused_for_an_experiment(tmp_path: Path) -> None:
    """§30 'swap run_id': run_id is derived from provenance, so a relabelled artifact is caught.

    Without the identity recomputation this artifact would pass every schema check, and a consumer
    keying on run_id would attribute one run's numbers to another.
    """
    from workbench.replay.publication import verify_run_identity

    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity(data_kind="experiment"))
    assert verify_run_identity(run) == [], "the derived identity is consistent by construction"

    payload = run.model_dump(mode="json")
    payload["run"]["run_id"] = "run-someone-elses"
    relabelled = ResultRun.model_validate(payload)  # schema-valid, identity-inconsistent
    mismatches = verify_run_identity(relabelled)
    assert mismatches and "relabelled" in mismatches[0]


def test_swapped_checkpoint_identity_changes_the_run_id() -> None:
    """Swapping the checkpoint changes the derived identity, so the two cannot be confused."""
    from workbench.replay.publication import verify_run_identity

    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity(data_kind="experiment"))
    payload = run.model_dump(mode="json")
    payload["run"]["checkpoint_id"] = "base_iter0_toptee"  # a different category's model
    swapped = ResultRun.model_validate(payload)
    assert verify_run_identity(swapped), "a swapped checkpoint must not keep the same identity"


def test_publication_refuses_an_inconsistent_identity(tmp_path: Path) -> None:
    """The gate is enforced at the writer, not only at a reader."""
    from workbench.replay.publication import publish

    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity(data_kind="experiment"))
    payload = run.model_dump(mode="json")
    payload["run"]["run_id"] = "run-injected"
    with pytest.raises(ValueError, match="relabelled"):
        publish(ResultRun.model_validate(payload), tmp_path)


def test_a_mock_artifact_is_not_held_to_derived_identity() -> None:
    """Mock fixtures are declared development data; the identity rule protects experiments only."""
    from workbench.replay.publication import verify_run_identity

    payload = build_result_run([_category("dress", 300, target_rank=5)], **_identity(data_kind="mock")).model_dump(mode="json")
    payload["run"]["run_id"] = "mock_csmcir-fiq_a"
    assert verify_run_identity(ResultRun.model_validate(payload)) == []
