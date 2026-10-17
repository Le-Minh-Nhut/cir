"""Controlled synthetic end-to-end: observation -> publication -> validation -> index -> API.

§31 asks for one full chain, and §32 asks that the chain use the REAL producer rather than
hand-written JSON fed to a mocked consumer. So this test drives the actual
``build_result_run`` -> ``publish`` path, then the actual DuckDB rebuild and the actual storage
read layer, and asserts the API's analysis contract on the result:

* two runs of the same protocol and the same query universe compare correctly;
* a third run under a DIFFERENT protocol is refused, not silently compared;
* explicitly synthetic output stays visibly synthetic and never becomes research evidence.

The three protocols are deliberately constructed to produce DIFFERENT target ranks for the same
canonical query identity, so a test cannot pass if the implementation quietly collapses them.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.backend import index as index_module
from workbench.backend import storage
from workbench.backend.schemas.results import ResultRun
from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.replay.publication import build_result_run, publish


def _observed(category: str, *, gallery_size: int, ranks: list[int], top_depth: int,
              annotate: int = 0) -> ObservedCategory:
    """One category whose queries have the requested target ranks, via the real schema path."""
    queries = []
    for offset, rank in enumerate(ranks):
        index = annotate + offset
        reference, target = f"{category}-ref-{index}", f"{category}-target-{index}"
        top = []
        for position in range(1, top_depth + 1):
            image_id = target if position == rank else ("shared-distractor" if position == 1 else f"d-{index}-{position}")
            top.append((position, image_id, round(1.0 - position / (gallery_size + 10), 6)))
        queries.append(ObservedQuery(
            annotation_index=index, reference_id=reference, target_id=target,
            raw_captions=("change one", "detail two"),
            model_input_text=f"variant-{annotate} change one and detail two",
            target_rank=rank, top_results=tuple(top), category=category,
        ))
    return ObservedCategory(category=category, gallery_size=gallery_size,
                            eligible_count=gallery_size - 1, queries=tuple(queries))


def _identity(model_id: str, protocol_id: str, label: str, *, noise: int | None = 0):
    return dict(run_id=f"dev-{model_id}-{protocol_id}-{label}", model_id=model_id,
                method_name=model_id.upper(), checkpoint_id=f"{model_id}-ckpt",
                protocol_id=protocol_id, top_k_saved=20, source_commit="a" * 40,
                checkpoint_sha256=label * 64, data_kind="mock")


@pytest.fixture
def chain(tmp_path: Path, monkeypatch):
    """Two compatible synthetic runs plus one incompatible-protocol run, all really published."""
    results = tmp_path / "results"
    gallery_size, top_depth = 50, 20

    # Same protocol, same canonical query universe (indices 0..5 in one category), different ranks.
    baseline = build_result_run(
        [_observed("dress", gallery_size=gallery_size, ranks=[1, 1, 12, 30, 4, 9], top_depth=top_depth)],
        **_identity("mock-alpha", "fashioniq_val_split", "1"))
    # Same canonical identities, different ranks. Ranks past the saved depth are legal: the target
    # simply does not appear in the stored Top-K.
    variant = build_result_run(
        [_observed("dress", gallery_size=gallery_size, ranks=[1, 7, 12, 30, 45, 3], top_depth=top_depth)],
        **_identity("mock-beta", "fashioniq_val_split", "2"))
    # A different protocol: same query identity strings, incompatible candidate universe.
    other = build_result_run(
        [_observed("dress", gallery_size=gallery_size, ranks=[1, 2, 3, 4, 5, 6], top_depth=top_depth)],
        **_identity("mock-gamma", "fashioniq_original_split", "3"))

    paths = [publish(baseline, results), publish(variant, results), publish(other, results)]
    database = tmp_path / "workbench.duckdb"
    # Explicitly declare that nothing is corroborated here: the promotion path reads the real
    # evidence ledger, and a test must not depend on host ledger state.
    monkeypatch.setattr(index_module, "verifier_verdicts", lambda *a, **k: {})
    count = index_module.rebuild_index(results_root=results, database_path=database)
    return {"paths": paths, "database": database, "count": count,
            "baseline": baseline, "variant": variant, "other": other}


def test_all_three_runs_index(chain) -> None:
    assert chain["count"] == 3
    assert all(path.exists() for path in chain["paths"])


def test_compatible_runs_compare_correctly(chain) -> None:
    """Real storage read layer, real analysis contract, on really-published artifacts."""
    ids = [chain["baseline"].run.run_id, chain["variant"].run.run_id]
    runs, maps = storage.validate_analysis_selection(ids, k=10, database_path=chain["database"])
    assert {run["protocol_id"] for run in runs} == {"fashioniq_val_split"}
    assert storage.get_query_universe(ids, chain["database"])[ids[0]].keys() == storage.get_query_universe(ids, chain["database"])[ids[1]].keys()


def test_common_distractors_are_found_across_compatible_runs(chain) -> None:
    ids = [chain["baseline"].run.run_id, chain["variant"].run.run_id]
    from workbench.backend.analysis import analysis_rows_from_maps

    _, maps = storage.get_analysis_input(ids, k=10, database_path=chain["database"])
    rows = analysis_rows_from_maps(ids, maps, 10)
    assert rows, "the analysis produced no rows"
    for row in rows:
        assert set(row["target_rank"]) == set(ids)
    # Wherever rank 1 is not the target, both runs retrieved the same rank-1 distractor, so the
    # analysis must surface it as a shared distractor with both runs listed.
    observed = [item for row in rows for item in row["common_distractors"]
                if item["image_id"] == "shared-distractor"]
    assert observed, "the shared distractor never surfaced in any query's analysis row"
    # Where NEITHER run's target occupied rank 1, both retrieved this distractor, so at least one
    # query row must attribute it to both runs. (Where a run's target IS rank 1, that run's rank-1
    # slot holds the target and the distractor is legitimately that run's alone.)
    shared_by_both = [item for item in observed if item["run_count"] == 2]
    assert shared_by_both, f"never attributed to both runs: {observed}"
    assert all(set(item["run_ids"]) == set(ids) for item in shared_by_both)


def test_incompatible_protocol_is_refused(chain) -> None:
    """The regression this guards: composing across protocols without complaint."""
    ids = [chain["baseline"].run.run_id, chain["other"].run.run_id]
    with pytest.raises(storage.WorkbenchError) as caught:
        storage.validate_analysis_selection(ids, k=10, database_path=chain["database"])
    assert caught.value.code == "cross_protocol"


def test_protocols_do_not_collapse_to_the_same_ranks(chain) -> None:
    """The fixture's own guarantee: the three protocols' ranks genuinely differ."""
    baseline = {q.query_id: q.target_rank for q in chain["baseline"].queries}
    other = {q.query_id: q.target_rank for q in chain["other"].queries}
    assert set(baseline) == set(other), "canonical identities must align for the fixture to mean anything"
    assert any(baseline[qid] != other[qid] for qid in baseline), "protocols must not produce identical ranks"


def test_synthetic_output_stays_visibly_synthetic(chain) -> None:
    ids = [chain["baseline"].run.run_id]
    runs, _ = storage.validate_analysis_selection(ids, k=10, database_path=chain["database"])
    assert runs[0]["data_kind"] == "mock"
    connection = storage._connect(chain["database"])
    try:
        status = connection.execute("SELECT evidence_status FROM runs WHERE run_id = ?", [ids[0]]).fetchone()[0]
    finally:
        connection.close()
    # Never PER_QUERY_VERIFIED: that label is reserved for a real experiment.
    assert status == "SYNTHETIC_OR_MOCK"


def test_synthetic_never_becomes_research_evidence() -> None:
    """The authoritative verifier refuses a non-experiment report at the source. §19/§31.

    The publication path cannot launder a synthetic run into evidence because the verifier reads
    ``data_kind`` off the report itself and refuses anything that is not ``experiment``.
    """
    from workbench.scripts import verify_run

    reasons = verify_run._evidence_kind_reasons({"data_kind": "mock"})
    assert reasons and "not a real experiment" in reasons[0]
    assert verify_run._evidence_kind_reasons({"data_kind": "experiment"}) == []


def test_a_published_synthetic_artifact_declares_itself_mock(chain) -> None:
    """Belt and braces on the artifact itself: the canonical file carries the honest kind."""
    payload = json.loads(chain["paths"][0].read_text())
    assert payload["run"]["data_kind"] == "mock"
    assert payload["run"]["data_kind"] != "experiment"
    assert ResultRun.model_validate(payload).run.data_kind == "mock"


def test_query_page_returns_the_published_ranks(chain) -> None:
    """The API's own read path returns exactly what was published, not a re-derivation."""
    run_id = chain["baseline"].run.run_id
    page = storage.query_page(run_id, limit=100, database_path=chain["database"])
    assert page["total"] == 6
    published = {q.query_id: q.target_rank for q in chain["baseline"].queries}
    returned = {item["query_id"]: item["target_rank"] for item in page["items"]}
    assert returned == published


def test_top_results_scores_are_served_from_the_artifact(chain) -> None:
    run_id = chain["baseline"].run.run_id
    query = chain["baseline"].queries[0]
    served = storage.get_query(run_id, query.query_id, chain["database"])
    assert [item["rank"] for item in served["top_results"]] == list(range(1, len(query.top_results) + 1))
    assert served["top_results"][0]["score"] == pytest.approx(query.top_results[0].score)


def test_raw_captions_and_model_input_text_are_distinct(chain) -> None:
    """§13: the export must not overwrite one with the other."""
    query = chain["baseline"].queries[0]
    assert query.raw_captions == ["change one", "detail two"]
    assert "and" in query.model_input_text
    assert query.model_input_text != " ".join(query.raw_captions)


# --------------------------------------------- §27: the evidence boundary in the same chain

def test_a_forged_experiment_reaching_the_index_stays_unverified(chain, tmp_path, monkeypatch) -> None:
    """The chain end to end: publish a forged experiment, index it, read the state back.

    This is the cross-module form of F-07/F-17. The forged artifact is schema-valid and complete
    across categories, and it borrows the `run_id` of a run a verifier has spoken about -- everything
    a naive check would accept. It must still be UNVERIFIED, because the verdict does not name this
    artifact's bytes.
    """
    from workbench.backend import index as index_module
    from workbench.replay.publication import publish

    forged = build_result_run(
        [_observed(category, gallery_size=50, ranks=[1, 9, 30, 4], top_depth=20)
         for category in ("dress", "shirt", "toptee")],
        **{**_identity("mock-forged", "fashioniq_val_split", "9"), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_forged01",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    publish(forged, tmp_path)

    verdicts = {forged.run.run_id: {
        "certified": True, "is_research_evidence": True,
        "per_query_artifact_digest": "0" * 64,          # a DIFFERENT artifact's digest
        "model_id": forged.run.model_id, "checkpoint_id": forged.run.checkpoint_id}}
    monkeypatch.setattr(index_module, "verifier_verdicts", lambda *a, **k: verdicts)
    database = tmp_path / "forged.duckdb"
    index_module.rebuild_index(results_root=tmp_path, database_path=database)
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        status = connection.execute("SELECT evidence_status FROM runs WHERE run_id = ?",
                                    [forged.run.run_id]).fetchone()[0]
    finally:
        connection.close()
    assert status == "PER_QUERY_UNVERIFIED", status


def test_a_receipt_naming_the_artifact_verifies_it(chain, tmp_path, monkeypatch) -> None:
    """POSITIVE CONTROL for the same boundary: the promotion path is reachable."""
    import hashlib

    from workbench.backend import index as index_module
    from workbench.replay.publication import publish

    subject = build_result_run(
        [_observed(category, gallery_size=50, ranks=[1, 9, 30, 4], top_depth=20)
         for category in ("dress", "shirt", "toptee")],
        **{**_identity("mock-good", "fashioniq_val_split", "8"), "data_kind": "experiment",
           "run_id": "20260101T000000000000Z_limn_ckpt_goodrun1",
           "command_digest": "c" * 64, "environment_digest": "d" * 64,
           "bundle_manifest_digest": "e" * 64,
           "instrumentation_parity": "INSTRUMENTATION_PARITY_VERIFIED"})
    path = publish(subject, tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    verdicts = {subject.run.run_id: {
        "certified": True, "is_research_evidence": True,
        "per_query_artifact_digest": digest,
        "model_id": subject.run.model_id, "checkpoint_id": subject.run.checkpoint_id}}
    monkeypatch.setattr(index_module, "verifier_verdicts", lambda *a, **k: verdicts)
    database = tmp_path / "good.duckdb"
    index_module.rebuild_index(results_root=tmp_path, database_path=database)
    import duckdb

    connection = duckdb.connect(str(database), read_only=True)
    try:
        status = connection.execute("SELECT evidence_status FROM runs WHERE run_id = ?",
                                    [subject.run.run_id]).fetchone()[0]
    finally:
        connection.close()
    assert status == "PER_QUERY_VERIFIED", status
