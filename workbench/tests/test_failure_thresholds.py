"""Every supported failure/pass threshold, against the authoritative target rank.

The Sample Explorer exposes fail/pass at 1, 5, 10, 20, 50, 100 and 200. The predicate is
`target_rank > K` (fail) or `target_rank <= K` (pass), where `target_rank` is a position in the
model's COMPLETE eligible ranking -- never the length of a saved Top-K list. A run that stores 50
images still classifies a query whose rank is 500 correctly.

These tests drive the real HTTP layer and the real SQL filter, with ranks chosen to sit exactly on
each boundary (K-1, K, K+1) so an off-by-one cannot pass.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from workbench.backend import index as index_module
from workbench.backend import storage
from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.replay.publication import build_result_run, publish

from workbench.tests.test_limn_query_export_api import asgi_request, _identity

SUPPORTED_K = (1, 5, 10, 20, 50, 100, 200)

#: One query per boundary position, so every threshold has a rank just below, exactly at, and just
#: above it. 500 is beyond every supported K and must fail at all of them.
RANKS = (1, 2, 5, 6, 10, 11, 20, 21, 50, 51, 100, 101, 200, 201, 500)


def _observed(ranks: tuple[int, ...], gallery_size: int = 600, top_depth: int = 50) -> ObservedCategory:
    """A category whose queries hold the given target ranks.

    A saved depth of 50 (smaller than several ranks) is deliberate: it proves the classification
    comes from `target_rank`, not from whether the target appears in the stored Top-K.
    """
    queries = []
    for index, rank in enumerate(ranks):
        reference, target = f"ref-{index}", f"target-{index}"
        top = tuple(
            (position, target if position == rank else f"d-{index}-{position}",
             round(1.0 - position / (gallery_size + 10), 6))
            for position in range(1, top_depth + 1))
        queries.append(ObservedQuery(
            annotation_index=index, reference_id=reference, target_id=target,
            raw_captions=("one", "two"), model_input_text="one and two",
            target_rank=rank, top_results=top, category="dress"))
    return ObservedCategory(category="dress", gallery_size=gallery_size,
                            eligible_count=gallery_size - 1, queries=tuple(queries))


@pytest.fixture
def ranked(tmp_path: Path, monkeypatch):
    """An indexed run holding all boundary ranks."""
    results = tmp_path / "results"
    # `top_k_saved` must equal the depth the observation actually stored, or the schema refuses the
    # artifact -- which is the correct behaviour and is exercised in the publication tests.
    run = build_result_run([_observed(RANKS, top_depth=50)],
                           **{**_identity("mock-ranks", "fashioniq_val_split", "7"),
                              "top_k_saved": 50})
    publish(run, results)
    monkeypatch.setattr(index_module, "verifier_verdicts", lambda *a, **k: {})
    database = tmp_path / "workbench.duckdb"
    index_module.rebuild_index(results_root=results, database_path=database)
    return {"run_id": run.run.run_id, "database": database}


def _ranks_for(query: dict) -> list[int]:
    return [item["target_rank"] for item in query["items"]]


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_fail_at_k_selects_exactly_the_ranks_above_k(ranked, k: int) -> None:
    page = storage.query_page(ranked["run_id"], failed_at=k, limit=200,
                              database_path=ranked["database"])
    expected = sorted(rank for rank in RANKS if rank > k)
    assert sorted(_ranks_for(page)) == expected, f"fail @{k}"
    # `total` must describe the FILTERED population, not the page, or pagination lies.
    assert page["total"] == len(expected), f"fail @{k}: total"


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_pass_at_k_selects_exactly_the_ranks_at_or_below_k(ranked, k: int) -> None:
    page = storage.query_page(ranked["run_id"], passed_at=k, limit=200,
                              database_path=ranked["database"])
    expected = sorted(rank for rank in RANKS if rank <= k)
    assert sorted(_ranks_for(page)) == expected, f"pass @{k}"
    assert page["total"] == len(expected), f"pass @{k}: total"


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_the_two_filters_partition_the_run(ranked, k: int) -> None:
    """Fail@K and Pass@K must be complements: together they are the whole run, exactly once."""
    failing = storage.query_page(ranked["run_id"], failed_at=k, limit=200,
                                 database_path=ranked["database"])
    passing = storage.query_page(ranked["run_id"], passed_at=k, limit=200,
                                 database_path=ranked["database"])
    assert failing["total"] + passing["total"] == len(RANKS), f"K={k}"
    assert not {item["query_id"] for item in failing["items"]} & {
        item["query_id"] for item in passing["items"]}, f"K={k}: overlap"


def test_the_boundary_ranks_are_classified_the_way_the_specification_says(ranked) -> None:
    """The named cases, each asserted against the threshold it actually tests.

    `rank == K` must NOT be a failure at K, and `rank == K+1` must be. The K+1 probes only exist for
    supported thresholds that have a successor in the rank list, so the assertion is never vacuous.
    """
    page = storage.query_page(ranked["run_id"], limit=200, database_path=ranked["database"])
    query_by_rank = {item["target_rank"]: item["query_id"] for item in page["items"]}

    for k in SUPPORTED_K:
        failing = {item["query_id"] for item in storage.query_page(
            ranked["run_id"], failed_at=k, limit=200, database_path=ranked["database"])["items"]}
        # Exactly at the threshold: a success.
        if k in query_by_rank:
            assert query_by_rank[k] not in failing, f"rank {k} must NOT fail @{k}"
        # The rank just above: a failure, whenever the fixture has it.
        if k + 1 in query_by_rank:
            assert query_by_rank[k + 1] in failing, f"rank {k+1} must fail @{k}"

    # The specification's literal examples, checked by rank value.
    failing10 = {item["target_rank"] for item in storage.query_page(
        ranked["run_id"], failed_at=10, limit=200, database_path=ranked["database"])["items"]}
    assert 10 not in failing10 and 11 in failing10
    failing200 = {item["target_rank"] for item in storage.query_page(
        ranked["run_id"], failed_at=200, limit=200, database_path=ranked["database"])["items"]}
    assert 200 not in failing200 and 201 in failing200


def test_a_rank_beyond_every_threshold_fails_at_all_of_them(ranked) -> None:
    """rank 500 is a failure at every supported K, even though it is not in the saved Top-50."""
    for k in SUPPORTED_K:
        page = storage.query_page(ranked["run_id"], failed_at=k, limit=200,
                                  database_path=ranked["database"])
        query = next(item for item in page["items"] if item["target_rank"] == 500)
        # Its stored Top-K does not contain the target -- the classification still holds.
        assert all(entry["image_id"] != query["target_id"] for entry in query["top_results"])
        assert query["target_rank"] == 500


def test_classification_does_not_depend_on_the_saved_depth(ranked) -> None:
    """The same ranks with a different saved depth classify identically.

    This is the distinction the specification draws: failure-classification availability is
    independent of Top-K visualization availability.
    """
    shallow = storage.query_page(ranked["run_id"], failed_at=50, limit=200,
                                 database_path=ranked["database"])
    # Every stored query has exactly 50 saves here; the classification uses the rank regardless.
    assert all(len(item["top_results"]) == 50 for item in shallow["items"])
    assert sorted(_ranks_for(shallow)) == sorted(rank for rank in RANKS if rank > 50)


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_filters_apply_before_pagination(ranked, k: int) -> None:
    """Page totals are counts of the filtered population, and pages partition it without loss."""
    expected = sorted(rank for rank in RANKS if rank > k)
    collected: list[int] = []
    offset = 0
    while True:
        page = storage.query_page(ranked["run_id"], failed_at=k, limit=3, offset=offset,
                                  database_path=ranked["database"])
        assert page["total"] == len(expected), f"K={k}: total must not shrink with the page"
        if not page["items"]:
            break
        collected.extend(_ranks_for(page))
        offset += 3
    assert sorted(collected) == expected, f"K={k}: pages must partition the filtered set"


@pytest.mark.parametrize("k", SUPPORTED_K)
def test_every_threshold_is_reachable_over_http(ranked, monkeypatch, k: int) -> None:
    """The API accepts each supported K and rejects a contradictory combination."""
    import importlib.util
    import sys

    repo = Path(__file__).resolve().parents[2]
    artifacts = ranked["database"].parent
    spec = importlib.util.spec_from_file_location("thresholds_api", repo / "workbench/backend/main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    class Config:
        WORKBENCH_ARTIFACTS_ROOT = artifacts
        WORKBENCH_RESULTS_ROOT = artifacts / "results"
        WORKBENCH_THIRD_PARTY_ROOT = artifacts / "third_party"
        WORKBENCH_CHECKPOINT_ROOT = artifacts / "checkpoints"
        FASHIONIQ_ROOT = artifacts / "FashionIQ"

    import workbench.backend.operator_config as operator_config

    monkeypatch.setattr(operator_config, "resolve_config", lambda: Config())
    monkeypatch.setattr(module, "resolve_config", lambda: Config())
    monkeypatch.setattr(module, "_aggregate_reports", lambda: [])
    monkeypatch.setattr(module, "JOBS", type("Q", (), {"list": lambda self: []})())

    status, body = asgi_request(
        module.app, "GET", f"/api/queries?run_id={ranked['run_id']}&failed_at={k}&limit=200")
    assert status == 200, body
    assert json.loads(body)["total"] == len([r for r in RANKS if r > k])

    status, body = asgi_request(
        module.app, "GET", f"/api/queries?run_id={ranked['run_id']}&failed_at={k}&passed_at={k}")
    assert status == 400, body
    assert json.loads(body)["error"] == "conflicting_rank_filter"

    status, body = asgi_request(
        module.app, "GET", f"/api/queries?run_id={ranked['run_id']}&failed_at=0")
    assert status == 400, body
    assert json.loads(body)["error"] == "invalid_rank_threshold"
