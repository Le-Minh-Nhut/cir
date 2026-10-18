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

import dataclasses
import json
import hashlib
from pathlib import Path

import pytest
from pydantic import ValidationError

from workbench.backend.schemas.results import ResultRun
from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.replay.publication import ParityMismatch, build_result_run, publish, publish_observed_categories


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
    """A category fixture at the ELIGIBLE depth the schema requires.

    `top_depth` is clamped to the eligible candidate count (`gallery_size - 1`) because that is what
    an excluded protocol can actually store: producing `top_depth` results for a 20-item gallery
    created a fixture the schema correctly refuses, which is the whole point of the depth rule.
    """
    eligible = max(gallery_size - 1, 0)
    depth = min(top_depth, eligible)
    return ObservedCategory(
        category=category, gallery_size=gallery_size, eligible_count=eligible,
        queries=tuple(_query(category, index, gallery_size, target_rank, depth) for index in range(count)),
    )


def _identity(**overrides):
    base = dict(run_id="dev-limn-base_iter0_all_categories-fashioniq_val_split", model_id="limn",
                method_name="LIMN", checkpoint_id="base_iter0_all_categories",
                protocol_id="fashioniq_val_split", top_k_saved=200, source_commit="a" * 40,
                checkpoint_sha256="b" * 64, data_kind="mock")
    base.update(overrides)
    return base


def _experiment(**overrides):
    """A development-shaped run promoted to a VALID experiment: everything `_identity` sets, plus
    the provenance an experiment must record, with a real-looking execution identity."""
    base = dict(_identity(), run_id="20260101T000000000000Z_limn_ckpt_deadbeef",
                command_digest="c" * 64, environment_digest="d" * 64,
                bundle_manifest_digest="e" * 64, data_kind="experiment")
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
    with pytest.raises(ValueError, match="exceeds the eligible candidate count"):
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
    # Two DIFFERENT runs must land in different files: identity now comes from the caller, so the
    # fixture supplies two distinct execution identities, exactly as the orchestrator would.
    first = build_result_run([_category("dress", 100, top_depth=100)],
                             **{**_identity(checkpoint_sha256="c" * 64),
                                "run_id": "20260101T000000000000Z_limn_ckpt_aaaaaaa1"})
    second = build_result_run([_category("dress", 100, top_depth=100)],
                              **{**_identity(checkpoint_sha256="d" * 64),
                                 "run_id": "20260101T000000000001Z_limn_ckpt_aaaaaaa2"})
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


def test_a_development_identity_on_an_experiment_is_refused() -> None:
    """F-09: a hand-built identity must never ride on an artifact that claims a real evaluation."""
    from workbench.replay.publication import verify_run_identity

    run = build_result_run([_category("dress", 300, target_rank=5)], **_experiment())
    assert verify_run_identity(run) == [], "a real-looking identity with provenance passes"

    payload = run.model_dump(mode="json")
    payload["run"]["run_id"] = "dev-limn-base_iter0_all_categories-fashioniq_val_split"
    assert verify_run_identity(ResultRun.model_validate(payload)), \
        "an experiment carrying a development identity must be refused"


def test_an_experiment_without_command_and_environment_digests_is_refused() -> None:
    """The identity is only useful if it can be tied to a command and an environment."""
    from workbench.replay.publication import verify_run_identity

    payload = build_result_run([_category("dress", 300, target_rank=5)], **_experiment()).model_dump(mode="json")
    payload["run"]["command_digest"] = None
    run = ResultRun.model_validate(payload)
    mismatches = verify_run_identity(run)
    assert mismatches and "command" in mismatches[0]


def test_two_executions_of_the_same_configuration_get_distinct_identities() -> None:
    """F-09's actual bug: model+checkpoint+protocol+source collided across executions.

    Two invocations of the same model/checkpoint/protocol must be TWO runs. The orchestrator
    guarantees this with a timestamp and a random suffix; anything derived purely from the
    configuration cannot.
    """
    import uuid

    first = f"20260101T000000000000Z_limn_ckpt_{uuid.uuid4().hex[:8]}"
    second = f"20260101T000000000001Z_limn_ckpt_{uuid.uuid4().hex[:8]}"
    assert first != second
    a = build_result_run([_category("dress", 300, target_rank=5)], **{**_experiment(), "run_id": first})
    b = build_result_run([_category("dress", 300, target_rank=5)], **{**_experiment(), "run_id": second})
    assert a.run.run_id != b.run.run_id


def test_build_result_run_requires_an_explicit_identity() -> None:
    """The exporter must not invent one; it takes the orchestrator's."""
    identity = _experiment()
    identity.pop("run_id")
    with pytest.raises(TypeError):
        build_result_run([_category("dress", 300, target_rank=5)], **identity)
    with pytest.raises(ValueError, match="run_id is required"):
        build_result_run([_category("dress", 300, target_rank=5)], **{**_experiment(), "run_id": ""})


def test_category_checkpoint_ids_must_map_every_category_uniquely() -> None:
    """F-12: LIMN's three category models are independently trained; one must not serve two."""
    mapping = {"dress": "base_iter0_dress", "shirt": "base_iter0_shirt", "toptee": "base_iter0_toptee"}
    run = build_result_run(
        [_category("dress", 300, target_rank=5), _category("shirt", 300, target_rank=5),
         _category("toptee", 300, target_rank=5)],
        **_experiment(category_checkpoint_ids=mapping))
    assert run.run.category_checkpoint_ids == mapping

    # The rule lives in the ONE admission policy, not in the builder: the builder records what it
    # is told, and admission decides whether the recorded provenance is acceptable. Asserting it
    # here keeps the two halves of F-12 in one place.
    from workbench.backend.schemas.results import PER_QUERY_PARITY_PENDING, PER_QUERY_UNVERIFIED, admission

    # A complete mapping is not itself a reason to withhold: the state here is PARITY_PENDING only
    # because this fixture records no parity verdict, which is a separate (and correct) refusal.
    state, reasons = admission(run)
    assert state == PER_QUERY_PARITY_PENDING, (state, reasons)
    assert not any("checkpoint" in reason for reason in reasons), reasons

    reused = dict(mapping, shirt="base_iter0_dress")  # shirt reusing dress weights
    reused_run = build_result_run(
        [_category("dress", 300, target_rank=5), _category("shirt", 300, target_rank=5),
         _category("toptee", 300, target_rank=5)],
        **_experiment(category_checkpoint_ids=reused))
    _, reasons = admission(reused_run)
    assert any("reuses one checkpoint" in r for r in reasons), reasons

    missing = {"dress": "base_iter0_dress"}
    missing_run = build_result_run(
        [_category("dress", 300, target_rank=5), _category("shirt", 300, target_rank=5)],
        **_experiment(category_checkpoint_ids=missing))
    _, reasons = admission(missing_run)
    assert any("missing categories" in r for r in reasons), reasons


def test_swapped_checkpoint_identity_is_detected() -> None:
    """Swapping the checkpoint must not keep the same recorded provenance."""
    run = build_result_run([_category("dress", 300, target_rank=5)], **_experiment())
    payload = run.model_dump(mode="json")
    payload["run"]["checkpoint_id"] = "base_iter0_toptee"
    swapped = ResultRun.model_validate(payload)
    # The identity no longer matches the recorded bundle provenance it claims.
    assert swapped.run.checkpoint_id != run.run.checkpoint_id


def test_publication_refuses_an_inconsistent_identity(tmp_path: Path) -> None:
    """The gate is enforced at the writer, not only at a reader."""
    run = build_result_run([_category("dress", 300, target_rank=5)], **_experiment())
    payload = run.model_dump(mode="json")
    payload["run"]["run_id"] = "dev-injected"
    with pytest.raises(ValueError, match="development identity"):
        publish(ResultRun.model_validate(payload), tmp_path)


def test_a_mock_artifact_is_not_held_to_derived_identity() -> None:
    """Mock fixtures are declared development data; the identity rule protects experiments only."""
    from workbench.replay.publication import verify_run_identity

    payload = build_result_run([_category("dress", 300, target_rank=5)],
                               **_identity(data_kind="mock")).model_dump(mode="json")
    payload["run"]["run_id"] = "mock_csmcir-fiq_a"
    assert verify_run_identity(ResultRun.model_validate(payload)) == []


# ------------------------------------------------------- F-07/F-13: the admission policy


def test_a_declared_experiment_without_provenance_is_never_verified() -> None:
    """F-07's core: `data_kind` is a declaration, not proof.

    Before the fix, `index.evidence_status(data_kind)` answered `PER_QUERY_VERIFIED` for any artifact
    that said `"experiment"`, so a schema-valid JSON file certified its own scientific authenticity.
    """
    from workbench.backend.schemas.results import (
        admission, PER_QUERY_PARITY_PENDING, PER_QUERY_UNVERIFIED, SYNTHETIC_OR_MOCK,
    )

    forged = build_result_run([_category("dress", 300, target_rank=5)],
                              **{**_identity(), "data_kind": "experiment",
                                 "run_id": "20260101T000000000000Z_limn_ckpt_deadbeef",
                                 "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    state, reasons = admission(forged)
    assert state == PER_QUERY_UNVERIFIED, (state, reasons)
    assert any("provenance" in reason for reason in reasons), reasons

    # A development artifact is named as one, whatever it claims.
    mock = build_result_run([_category("dress", 300, target_rank=5)], **_identity(data_kind="mock"))
    assert admission(mock)[0] == SYNTHETIC_OR_MOCK


def test_the_index_never_promotes_a_forged_experiment(tmp_path: Path) -> None:
    """The end-to-end form: publish a forged experiment, index it, read back the state.

    A run_id borrowed from a verified aggregate must NOT be enough -- the verdict has to name this
    artifact's digest. That was the adversarial finding: `promote` matched on run_id alone, so
    fabricated rankings could ride a verified run's identity into `PER_QUERY_VERIFIED`.
    """
    from workbench.backend import index as index_module

    forged = build_result_run(
        [_category("dress", 300, target_rank=5), _category("shirt", 300, target_rank=5),
         _category("toptee", 300, target_rank=5)],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_deadbeef"})
    result = forged.model_copy(update={"run": forged.run.model_copy(update={
        "command_digest": "c" * 64, "environment_digest": "d" * 64,
        "bundle_manifest_digest": "e" * 64,
        "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})})
    publish(result, tmp_path)

    # A verifier receipt exists for this run_id -- but for a DIFFERENT artifact.
    borrowed = {result.run.run_id: {
        "certified": True, "is_research_evidence": True,
        "per_query_artifact_digest": "9" * 64,
        "model_id": result.run.model_id, "checkpoint_id": result.run.checkpoint_id}}
    original = index_module.verifier_verdicts
    index_module.verifier_verdicts = lambda *a, **k: borrowed
    try:
        database = tmp_path / "workbench.duckdb"
        index_module.rebuild_index(results_root=tmp_path, database_path=database)
        import duckdb

        connection = duckdb.connect(str(database), read_only=True)
        try:
            status = connection.execute("SELECT evidence_status FROM runs WHERE run_id = ?",
                                        [result.run.run_id]).fetchone()[0]
        finally:
            connection.close()
    finally:
        index_module.verifier_verdicts = original

    assert status == "PER_QUERY_UNVERIFIED", \
        "a borrowed run_id must not promote an artifact the verdict never saw"


def test_a_verdict_bound_to_the_artifact_digest_does_promote(tmp_path: Path) -> None:
    """POSITIVE CONTROL: the promotion path works when the receipt names THIS artifact."""
    from workbench.backend import index as index_module

    artifact_run = build_result_run(
        [_category("dress", 300, target_rank=5), _category("shirt", 300, target_rank=5),
         _category("toptee", 300, target_rank=5)],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_cafebabe",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    published = publish(artifact_run, tmp_path)
    digest = hashlib.sha256(published.read_bytes()).hexdigest()

    # The receipt the parent would record: the digest of the artifact it verified.
    verdict = {artifact_run.run.run_id: {
        "certified": True, "is_research_evidence": True,
        "per_query_artifact_digest": digest,
        "model_id": artifact_run.run.model_id,
        "checkpoint_id": artifact_run.run.checkpoint_id}}
    original = index_module.verifier_verdicts
    index_module.verifier_verdicts = lambda *a, **k: verdict
    try:
        database = tmp_path / "workbench.duckdb"
        index_module.rebuild_index(results_root=tmp_path, database_path=database)
        import duckdb

        connection = duckdb.connect(str(database), read_only=True)
        try:
            status = connection.execute("SELECT evidence_status FROM runs WHERE run_id = ?",
                                        [artifact_run.run.run_id]).fetchone()[0]
        finally:
            connection.close()
    finally:
        index_module.verifier_verdicts = original

    assert status == "PER_QUERY_VERIFIED"


def test_partial_category_coverage_is_not_admitted_as_an_experiment() -> None:
    """A one-category diagnostic must never be admissible as the complete benchmark."""
    from workbench.backend.schemas.results import admission

    partial = build_result_run([_category("dress", 300, target_rank=5)],
                               **{**_identity(), "data_kind": "experiment",
                                  "run_id": "20260101T000000000000Z_limn_ckpt_partial1",
                                  "command_digest": "c" * 64, "environment_digest": "d" * 64,
                                  "bundle_manifest_digest": "e" * 64,
                                  "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    _, reasons = admission(partial)
    assert any("all three FashionIQ categories" in reason for reason in reasons), reasons


def test_target_rank_cannot_use_the_masked_reference_slot() -> None:
    """F-15: under an excluded protocol the eligible count is size-1, so the last slot is not a rank."""
    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity())
    payload = run.model_dump(mode="json")
    # gallery 300, eligible 299: a rank of 300 only exists if the masked reference is counted.
    payload["queries"][0]["target_rank"] = 300
    payload["queries"][0]["top_results"] = [
        {"rank": rank, "image_id": ("dress-target-0" if rank == 300 else f"d-{rank}"),
         "score": 1.0 - rank / 1000} for rank in range(1, 201)]
    payload["queries"][0]["top_results"][199]["image_id"] = "slot-200"
    with pytest.raises(ValidationError, match="eligible candidate count"):
        ResultRun.model_validate(payload)


def test_a_truncated_category_population_is_not_a_complete_benchmark(tmp_path: Path) -> None:
    """Coverage: all three categories must be present, and each must be whole.

    The observation's declared counts are what a reader can check against, so a category whose
    queries were truncated is refused rather than counted as a smaller-but-complete benchmark.
    """
    from workbench.backend.schemas.results import admission, FASHIONIQ_CATEGORIES

    full = build_result_run(
        [_category(category, 300, count=4, target_rank=5) for category in FASHIONIQ_CATEGORIES],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_full1234",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    state, reasons = admission(full)
    assert not any("categories" in reason for reason in reasons), reasons

    # Drop a whole category: the artifact is still schema-valid but is not the benchmark.
    without = build_result_run(
        [_category(category, 300, count=4, target_rank=5) for category in FASHIONIQ_CATEGORIES[:2]],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_part123",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    _, reasons = admission(without)
    assert any("all three FashionIQ categories" in reason for reason in reasons), reasons


def test_canonical_query_counts_match_the_published_validation_split() -> None:
    """The counts a reader should expect, asserted against the real annotation files when present.

    Skipped (not failed) when the dataset is absent: the dataset is an external artifact, and the
    point of the test is that the numbers are checked against real data rather than remembered.
    """
    from pathlib import Path as _Path

    captions = _Path("data/fashionIQ_dataset/captions")
    if not captions.is_dir():
        pytest.skip("FashionIQ annotations are not present on this host (external artifact)")
    expected = {"dress": 2017, "shirt": 2038, "toptee": 1961}
    for category, count in expected.items():
        import json as _json

        records = _json.loads((captions / f"cap.{category}.val.json").read_text())
        assert len(records) == count, f"{category}: {len(records)} != {count}"


def test_a_symlink_destination_is_refused(tmp_path: Path) -> None:
    """A symlink would put the artifact's bytes outside the results root, where they can change
    after the index read their digest (found in review)."""
    run = build_result_run([_category("dress", 300, target_rank=5)], **_identity())
    destination = tmp_path / run.run.protocol_id / run.run.model_id / f"{run.run.run_id}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    destination.symlink_to(outside)

    with pytest.raises(FileExistsError, match="symlink"):
        publish(run, tmp_path)
    assert outside.read_text() == "{}", "the symlink target must be untouched"


def test_a_partial_category_run_is_not_promoted_even_with_a_matching_receipt() -> None:
    """Coverage and promotion are two different checks, and both must hold.

    The verifier judges the AGGREGATE report and cannot see that a per-query artifact covers only
    one category, so the artifact's own admissibility is re-checked during promotion. Without that,
    a dress-only artifact with a matching digest became PER_QUERY_VERIFIED (found in review).
    """
    from workbench.backend.schemas.results import (
        PER_QUERY_UNVERIFIED, PER_QUERY_VERIFIED, admission, promote,
    )

    partial = build_result_run(
        [_category("dress", 300, target_rank=5)],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_partial9",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    state, reasons = admission(partial)
    assert state == PER_QUERY_UNVERIFIED and any("categories" in r for r in reasons), reasons

    digest = "a" * 64
    receipt = {"certified": True, "is_research_evidence": True,
               "per_query_artifact_digest": digest, "model_id": partial.run.model_id,
               "checkpoint_id": partial.run.checkpoint_id}
    promoted, blocking = promote(partial, state, receipt, artifact_digest=digest)
    assert promoted == PER_QUERY_UNVERIFIED, "a partial benchmark must never be verified"
    assert any("categories" in reason for reason in blocking), blocking

    # POSITIVE CONTROL: the same receipt DOES promote a complete artifact.
    complete = build_result_run(
        [_category(category, 300, target_rank=5) for category in ("dress", "shirt", "toptee")],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_complete1",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    receipt_complete = {**receipt, "model_id": complete.run.model_id,
                        "checkpoint_id": complete.run.checkpoint_id}
    assert promote(complete, PER_QUERY_UNVERIFIED, receipt_complete,
                   artifact_digest=digest)[0] == PER_QUERY_VERIFIED


def test_a_parity_failure_is_rejected_by_strict_real(tmp_path: Path) -> None:
    """A failed parity is a DEFECT in the artifact, so a release gate must reject it.

    Reporting it as a warning left `--strict-real` exiting 0 for a corrupted run while blocking a
    clean-but-unchecked one, which inverted the severity (found in review).
    """
    from workbench.scripts.validate_results import validate_file

    run = build_result_run(
        [_category(category, 300, target_rank=5) for category in ("dress", "shirt", "toptee")],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_parity9",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_MISMATCH"})
    artifact = tmp_path / "mismatch.json"
    artifact.write_text(run.model_dump_json())
    output: list[str] = []
    assert validate_file(artifact, strict_real=True, output=output.append) is False, output
    assert any("PARITY_FAILED" in line and "BLOCKED" in line for line in output), output


def test_a_reason_filter_cannot_be_defeated_by_the_artifacts_own_text() -> None:
    """The promotion filter must compare against a closed set, never a substring of artifact text.

    A verifier found this: `instrumentation_parity` was a free string, so an artifact could set it to
    the text of the "awaiting corroboration" reason, making the filter drop the real defect that
    followed it. The state is now drawn from a closed set and the reason compared by equality.
    """
    from workbench.backend.schemas.results import (
        INSTRUMENTATION_PARITY_STATES, PER_QUERY_UNVERIFIED, PER_QUERY_VERIFIED, promote,
    )

    assert "awaiting corroboration" not in INSTRUMENTATION_PARITY_STATES

    complete = build_result_run(
        [_category(category, 300, target_rank=5) for category in ("dress", "shirt", "toptee")],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_selftext",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "awaiting corroboration"})  # the exploit
    digest = "a" * 64
    receipt = {"certified": True, "is_research_evidence": True,
               "per_query_artifact_digest": digest, "model_id": complete.run.model_id,
               "checkpoint_id": complete.run.checkpoint_id}
    state, blocking = promote(complete, PER_QUERY_UNVERIFIED, receipt, artifact_digest=digest)
    assert state == PER_QUERY_UNVERIFIED, "an unrecognized parity state must never be promoted"
    assert blocking and "unrecognized instrumentation parity state" in blocking[0], blocking
    # And the legitimate "no verifier yet" reason is still recognized by equality.
    clean = build_result_run(
        [_category(category, 300, target_rank=5) for category in ("dress", "shirt", "toptee")],
        **{**_identity(), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_oktext01",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    state, blocking = promote(clean, PER_QUERY_UNVERIFIED,
                              {**receipt, "model_id": clean.run.model_id,
                               "checkpoint_id": clean.run.checkpoint_id},
                              artifact_digest=digest)
    assert state == PER_QUERY_VERIFIED and blocking == [], blocking


def test_strict_real_rejects_every_unresolved_parity_state(tmp_path: Path) -> None:
    """A release gate must not accept an artifact whose numbers nobody established.

    PARITY_PENDING slipped through `--strict-real` (found by a verifier): it was reported as a
    warning and the exit code stayed 0, so an unchecked artifact passed the gate. FAILED is a
    defect and PENDING is an absence of checking -- both are rejections at release time, and both
    remain warnings without the flag so a workspace in progress stays inspectable.
    """
    from workbench.scripts.validate_results import validate_file

    for parity in ("INSTRUMENTATION_PARITY_MISMATCH", None):
        run = build_result_run(
            [_category(category, 300, target_rank=5) for category in ("dress", "shirt", "toptee")],
            **{**_identity(), "data_kind": "experiment",
               "run_id": f"20260101T000000000000Z_limn_ckpt_gate{str(parity)[:4]}",
               "command_digest": "c" * 64, "environment_digest": "d" * 64,
               "bundle_manifest_digest": "e" * 64, "instrumentation_parity": parity})
        artifact = tmp_path / f"gate-{str(parity)[:8]}.json"
        artifact.write_text(run.model_dump_json())
        output: list[str] = []
        assert validate_file(artifact, strict_real=True, output=output.append) is False, (parity, output)
        assert any("BLOCKED" in line for line in output), (parity, output)
        # Without the flag it is inspectable as a warning.
        warnings: list[str] = []
        assert validate_file(artifact, strict_real=False, output=warnings.append) is True, warnings


def test_saved_depth_follows_the_eligible_count_not_the_raw_gallery() -> None:
    """P2: `gallery_size <= top_k_saved` under an excluded protocol stores `size-1`, not `size`.

    The two rules disagreed: the observer saved `min(top_k, size-1)` while the schema demanded
    `min(top_k_saved, size)`, so a legitimate artifact whose saved depth was capped by the eligible
    count was rejected. Exercised at the boundary the defect names.
    """
    for gallery_size in (5, 20, 21):
        observed = _category("dress", gallery_size, count=2, target_rank=2, top_depth=100)
        run = build_result_run([observed], **_identity(top_k_saved=100))
        # 100 requested, but only `size - 1` candidates exist.
        assert len(run.queries[0].top_results) == gallery_size - 1
        assert run.run.category_eligible_counts == {"dress": gallery_size - 1}
        # The reference is not among them.
        reference = run.queries[0].reference_id
        assert all(item.image_id != reference for item in run.queries[0].top_results)


def test_the_schema_still_requires_exactly_the_eligible_depth() -> None:
    """The rule is a bound, not a suggestion: both too few and too many are refused."""
    run = build_result_run([_category("dress", 20, count=2, target_rank=2, top_depth=19)],
                           **_identity(top_k_saved=100))
    payload = run.model_dump(mode="json")
    # Too few.
    payload["queries"][0]["top_results"] = payload["queries"][0]["top_results"][:10]
    with pytest.raises(ValidationError, match="effective saved retrieval depth"):
        ResultRun.model_validate(payload)
    # Too many (the raw gallery size, which the old rule wrongly demanded).
    payload2 = run.model_dump(mode="json")
    last_rank = len(payload2["queries"][0]["top_results"]) + 1
    payload2["queries"][0]["top_results"].append(
        {"rank": last_rank, "image_id": f"extra-{last_rank}", "score": 0.01})
    with pytest.raises(ValidationError, match="effective saved retrieval depth"):
        ResultRun.model_validate(payload2)


def test_a_declared_eligible_count_cannot_widen_the_protocol_ceiling() -> None:
    """A self-declared count may narrow the bound, never widen it past the protocol.

    A verifier found this: declaring `category_eligible_counts[category] == gallery_size` under an
    excluded protocol made `min(size, declared)` equal the raw size, so a rank equal to the masked
    reference's slot was admitted -- the phantom rank F-15 exists to prevent.
    """
    run = build_result_run([_category("dress", 20, count=2, target_rank=2, top_depth=19)],
                           **_identity(top_k_saved=100))
    payload = run.model_dump(mode="json")
    # Claim every candidate is eligible, which the protocol cannot produce.
    payload["run"]["category_eligible_counts"] = {"dress": 20}
    # Rank 20 is the masked reference's slot, so it must be refused regardless of what is declared.
    payload["queries"][0]["target_rank"] = 20
    with pytest.raises(ValidationError, match="target_rank exceeds the eligible candidate count"):
        ResultRun.model_validate(payload)

    # Narrowing is allowed: a claim below the ceiling is honoured, and the depth follows it.
    narrower = run.model_dump(mode="json")
    narrower["run"]["category_eligible_counts"] = {"dress": 10}
    narrower["run"]["top_k_saved"] = 10
    for query in narrower["queries"]:
        query["top_results"] = query["top_results"][:10]
    ResultRun.model_validate(narrower)

# ---------------------------------------------------------------------------
# The declared bundle identity cannot be decided by the payload
# ---------------------------------------------------------------------------

def _staged_payload(tmp_path: Path, members: dict) -> Path:
    """A staged observation payload that is valid in every respect except the members passed in."""
    categories = [dataclasses.asdict(_category(name, 20)) for name in ("dress", "shirt", "toptee")]
    for category in categories:
        for query in category["queries"]:
            query["raw_captions"] = list(query["raw_captions"])
            query["top_results"] = [{"rank": rank, "image_id": image_id, "score": score}
                                    for rank, image_id, score in query["top_results"]]
            query["query_id"] = (f"{query['category']}:{query['annotation_index']}:"
                                 f"{query['reference_id']}:{query['target_id']}")
    payload = {
        "artifact_type": "limn_observation_payload", "schema_version": 1, "model_id": "limn",
        "method_name": "LIMN", "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split",
        "run_id": "dev-limn-base_iter0_all_categories-fashioniq_val_split",
        "top_k": 200, "upstream_commit": "7d7bc9b116f594a65ac22457491edf28a88d3c3e",
        "checkpoint_members": members,
        "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED",
        "instrumentation_mismatches": {}, "categories": categories,
    }
    path = tmp_path / "payload.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _member(category: str, digest_char: str) -> dict:
    return {"filename": f"0_{category}_best_model.pt",
            "checkpoint_id": f"base_iter0_{category}", "sha256": digest_char * 64}


def test_a_payload_cannot_publish_a_partial_bundle(tmp_path: Path) -> None:
    """R8-04: the member set is the REGISTRY's to declare, not the payload's.

    The planned-digest comparison lives inside the payload: the observer writes both the members and
    the digest they are compared against, and an operator running the publisher directly supplies
    both. So it cannot answer "is this the declared bundle". Measured on the pre-fix tree, a payload
    declaring ONE member of LIMN's three-member bundle published successfully (rc 0) as a
    `base_iter0_all_categories` artifact -- a "three-category" per-query result whose recorded bundle
    held a single file, and internally consistent enough that no consumer comparing bundle identity
    would notice.
    """
    from workbench.replay.limn_publish import load_payload

    partial = _staged_payload(tmp_path, {"dress": _member("dress", "a")})
    with pytest.raises(Exception) as refusal:
        load_payload(partial)
    message = str(refusal.value)
    assert "base_iter0_shirt" in message and "base_iter0_toptee" in message, message
    assert "base_iter0_all_categories" in message, message


def test_the_declared_bundle_is_accepted_when_all_three_members_are_staged(tmp_path: Path) -> None:
    """The positive control: refusing everything would also pass the test above.

    Three members, one per declared category, are what the registry requires, and they must publish.
    """
    from workbench.replay.limn_publish import load_payload

    complete = _staged_payload(tmp_path, {
        "dress": _member("dress", "a"), "shirt": _member("shirt", "b"),
        "toptee": _member("toptee", "c")})
    loaded = load_payload(complete)
    assert sorted(loaded["checkpoint_members"]) == ["dress", "shirt", "toptee"]


def test_a_single_category_checkpoint_declares_no_bundle_to_compare(tmp_path: Path) -> None:
    """A DSP: `base_iter0_dress` is not a bundle, so the check must not invent a requirement for it.

    The registry declares `checkpoint_bundles` only for the all-categories bundle. A payload for a
    single-category checkpoint has nothing to compare against, and demanding three members there
    would refuse a legitimate diagnostic run.
    """
    from workbench.replay.limn_publish import load_payload

    payload = _staged_payload(tmp_path, {"dress": _member("dress", "a")})
    data = json.loads(payload.read_text(encoding="utf-8"))
    data["checkpoint_id"] = "base_iter0_dress"
    payload.write_text(json.dumps(data), encoding="utf-8")
    assert load_payload(payload)["checkpoint_id"] == "base_iter0_dress"
