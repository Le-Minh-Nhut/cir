"""The HTTP layer over the per-query results the exporter actually publishes.

`test_limn_query_export_e2e.py` drives the real producer chain into the real DuckDB index and the
real storage read layer. This file takes the last step: it drives the real FastAPI application
through the ASGI protocol, so routing, parameter validation, response serialization and error
status codes over per-query artifacts are all exercised — not just the handler functions.

The subject is the contract an operator depends on: a published artifact is reachable through
`/api/runs`, its queries through `/api/queries`, its ranks through `/api/analysis`, and a
cross-protocol selection is refused with a named error rather than a silently mixed answer.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.replay.publication import build_result_run, publish
from workbench.backend import index as index_module

REPOSITORY = Path(__file__).resolve().parents[2]


def asgi_request(app, method: str, path: str) -> tuple[int, bytes]:
    """Send one request through the application; return (status, body)."""
    query, _, raw = path.partition("?")
    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": method.upper(), "scheme": "http", "path": query, "raw_path": query.encode(),
        "query_string": raw.encode(), "root_path": "",
        "headers": [(b"host", b"testserver")], "client": ("127.0.0.1", 50123),
        "server": ("testserver", 80),
    }
    response: dict = {"status": None, "body": b""}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    return response["status"], response["body"]


def _observed(category: str, ranks: list[int], gallery_size: int = 40, top_depth: int = 20,
              annotate: int = 0, text: str = "one") -> ObservedCategory:
    queries = []
    for offset, rank in enumerate(ranks):
        index = annotate + offset
        reference, target = f"{category}-r{index}", f"{category}-t{index}"
        top = []
        for position in range(1, top_depth + 1):
            image_id = target if position == rank else ("shared" if position == 1 else f"x{index}-{position}")
            top.append((position, image_id, round(1.0 - position / (gallery_size + 10), 6)))
        queries.append(ObservedQuery(
            annotation_index=index, reference_id=reference, target_id=target,
            raw_captions=(text, "two"), model_input_text=f"{text} and two",
            target_rank=rank, top_results=tuple(top), category=category))
    return ObservedCategory(category=category, gallery_size=gallery_size,
                            eligible_count=gallery_size - 1, queries=tuple(queries))


def _identity(model_id: str, protocol_id: str, tag: str) -> dict:
    return dict(model_id=model_id, method_name=model_id.upper(), checkpoint_id=f"{model_id}-ckpt",
                protocol_id=protocol_id, top_k_saved=20, source_commit="a" * 40,
                checkpoint_sha256=tag * 64, data_kind="mock")


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real application, with its artifact root redirected onto freshly published artifacts."""
    artifacts = tmp_path / "artifacts"
    results = artifacts / "results"
    artifacts.mkdir(parents=True)

    # §20: two runs may be compared only when their canonical identity matches -- including the
    # ORIGINAL captions. `model_input_text` may differ (different preprocessing), so the two runs
    # below share raw captions and differ only in the text actually fed to each model.
    alpha = build_result_run([_observed("dress", [1, 9, 30, 4], text="original")],
                             **_identity("mock-alpha", "fashioniq_val_split", "1"))
    beta = build_result_run([_observed("dress", [1, 9, 30, 4], text="original")],
                            **_identity("mock-beta", "fashioniq_val_split", "2"))
    beta = beta.model_copy(update={"queries": [
        query.model_copy(update={"model_input_text": "a different composition"})
        for query in beta.queries]})
    gamma = build_result_run([_observed("dress", [1, 1, 1, 1], text="original")],
                             **_identity("mock-gamma", "fashioniq_original_split", "3"))
    for run in (alpha, beta, gamma):
        publish(run, results)
    index_module.rebuild_index(results_root=results, database_path=artifacts / "workbench.duckdb")

    # The REAL module, not a fresh spec-loaded copy: `storage` resolves the index through
    # `operator_config.resolve_config` while `main` has its own imported name, so patching only one
    # of them would have the routes read a different index than the fixture wrote. Patching the
    # shared owner (`operator_config`) makes every reader and writer agree, which is what production
    # does and what these tests must therefore exercise.
    import workbench.backend.operator_config as operator_config
    from workbench.backend import main as module

    class Config:
        WORKBENCH_ARTIFACTS_ROOT = artifacts
        WORKBENCH_RESULTS_ROOT = results
        WORKBENCH_THIRD_PARTY_ROOT = tmp_path / "third_party"
        WORKBENCH_CHECKPOINT_ROOT = tmp_path / "checkpoints"
        FASHIONIQ_ROOT = tmp_path / "FashionIQ"

    monkeypatch.setattr(operator_config, "resolve_config", lambda: Config())
    monkeypatch.setattr(module, "resolve_config", lambda: Config())
    monkeypatch.setattr(module, "load_registry", lambda *a, **k: {
        "protocols": {}, "models": [{"model_id": "mock-alpha", "method_name": "Alpha"}],
    })
    monkeypatch.setattr(module, "_aggregate_reports", lambda: [])
    monkeypatch.setattr(module, "JOBS", type("Q", (), {"list": lambda self: []})())
    return {"module": module, "app": module.app, "runs": (alpha, beta, gamma)}


def test_published_runs_are_listed_by_the_api(api) -> None:
    status, body = asgi_request(api["app"], "GET", "/api/runs")
    assert status == 200, body
    listing = json.loads(body)
    assert len(listing) == 3
    ids = {run["run_id"] for run in listing}
    assert ids == {run.run.run_id for run in api["runs"]}
    # The new metadata reaches the API projection, not just the file on disk.
    alpha = next(run for run in listing if run["run_id"] == api["runs"][0].run.run_id)
    assert alpha["category_gallery_sizes"] == {"dress": 40}
    assert alpha["category_eligible_counts"] == {"dress": 39}
    assert alpha["data_kind"] == "mock"


def test_protocol_filter_partitions_the_runs(api) -> None:
    status, body = asgi_request(api["app"], "GET", "/api/runs?protocol_id=fashioniq_val_split")
    assert status == 200
    assert len(json.loads(body)) == 2


def test_queries_endpoint_returns_the_published_ranks(api) -> None:
    run_id = api["runs"][0].run.run_id
    status, body = asgi_request(api["app"], "GET", f"/api/queries?run_id={run_id}&limit=50")
    assert status == 200, body
    page = json.loads(body)
    assert page["total"] == 4
    published = {q.query_id: q.target_rank for q in api["runs"][0].queries}
    assert {item["query_id"]: item["target_rank"] for item in page["items"]} == published


def test_query_page_orders_worst_first_by_default(api) -> None:
    run_id = api["runs"][0].run.run_id
    _, body = asgi_request(api["app"], "GET", f"/api/queries?run_id={run_id}&limit=50")
    ranks = [item["target_rank"] for item in json.loads(body)["items"]]
    assert ranks == sorted(ranks, reverse=True)


def test_failed_at_filter_selects_the_failures(api) -> None:
    run_id = api["runs"][0].run.run_id
    _, body = asgi_request(api["app"], "GET", f"/api/queries?run_id={run_id}&failed_at=10&limit=50")
    returned = {item["query_id"] for item in json.loads(body)["items"]}
    failing = {q.query_id for q in api["runs"][0].queries if q.target_rank > 10}
    assert returned == failing


def test_cross_protocol_analysis_is_refused_by_the_api(api) -> None:
    """The refusal must be a named error over HTTP, not a mixed-provenance answer."""
    ids = ",".join([api["runs"][0].run.run_id, api["runs"][2].run.run_id])
    status, body = asgi_request(api["app"], "GET", f"/api/analysis?run_ids={ids}&k=10")
    assert status == 400, body
    payload = json.loads(body)
    assert payload["error"] == "cross_protocol"
    assert set(payload["details"]["protocol_ids"]) == {"fashioniq_val_split", "fashioniq_original_split"}


def test_compatible_analysis_succeeds_over_http(api) -> None:
    ids = ",".join([api["runs"][0].run.run_id, api["runs"][1].run.run_id])
    status, body = asgi_request(api["app"], "GET", f"/api/analysis?run_ids={ids}&k=10")
    assert status == 200, body
    rows = json.loads(body)
    assert rows and all(set(row["target_rank"]) == set(ids.split(",")) for row in rows)


def test_compare_returns_the_saved_top_k_for_both_runs(api) -> None:
    alpha_id, beta_id = api["runs"][0].run.run_id, api["runs"][1].run.run_id
    query_id = api["runs"][0].queries[0].query_id
    status, body = asgi_request(
        api["app"], "GET", f"/api/compare/{query_id}?run_ids={alpha_id},{beta_id}&k=5")
    assert status == 200, body
    payload = json.loads(body)
    assert len(payload) == 2
    assert all(len(entry["query"]["top_results"]) == 5 for entry in payload)


def test_compare_with_an_unknown_query_is_a_named_error(api) -> None:
    """Operator-input errors are 400 by the app's own policy; what matters is the named code."""
    alpha_id = api["runs"][0].run.run_id
    status, body = asgi_request(api["app"], "GET", f"/api/compare/nope:nope:nope:nope?run_ids={alpha_id}&k=5")
    assert status == 400, body
    assert json.loads(body)["error"] == "query_not_found"


def test_failure_jaccard_over_http(api) -> None:
    ids = ",".join([api["runs"][0].run.run_id, api["runs"][1].run.run_id])
    status, body = asgi_request(api["app"], "GET", f"/api/analysis/failure-jaccard?run_ids={ids}&k=10")
    assert status == 200, body
    pairs = json.loads(body)
    # Both runs fail the same queries, so every pair's Jaccard is 1.0.
    assert pairs and all(pair["jaccard"] == 1.0 for pair in pairs)


def test_runs_with_different_raw_captions_cannot_be_compared(api, tmp_path: Path) -> None:
    """§20's other half: `model_input_text` may differ, the ORIGINAL captions may not.

    Two runs whose canonical identity disagrees on raw captions are not the same experiment, and
    the API must refuse them by name rather than aligning them anyway.
    """
    import workbench.backend.operator_config as operator_config
    from workbench.replay.publication import publish
    from workbench.backend import index as index_module

    results = api["module"].resolve_config().WORKBENCH_RESULTS_ROOT
    divergent = build_result_run([_observed("dress", [1, 9, 30, 4], text="not-the-original")],
                                 **_identity("mock-delta", "fashioniq_val_split", "4"))
    publish(divergent, results)
    index_module.rebuild_index(results_root=results, database_path=results.parent / "workbench.duckdb")

    ids = ",".join([api["runs"][0].run.run_id, divergent.run.run_id])
    status, body = asgi_request(api["app"], "GET", f"/api/analysis?run_ids={ids}&k=10")
    assert status == 400, body
    payload = json.loads(body)
    assert payload["error"] == "canonical_query_mismatch"
    assert "raw_captions" in payload["details"]["mismatches"]


def test_runs_that_differ_only_in_model_input_text_are_comparable(api) -> None:
    """The positive control for the rule above: this is the legitimate cross-model case."""
    ids = ",".join([api["runs"][0].run.run_id, api["runs"][1].run.run_id])
    status, body = asgi_request(api["app"], "GET", f"/api/analysis?run_ids={ids}&k=10")
    assert status == 200, body


def test_invalid_pagination_is_refused(api) -> None:
    run_id = api["runs"][0].run.run_id
    status, body = asgi_request(api["app"], "GET", f"/api/queries?run_id={run_id}&limit=0")
    assert status == 400, body
    assert json.loads(body)["error"] == "invalid_pagination"


def test_unknown_run_is_a_named_error(api) -> None:
    status, body = asgi_request(api["app"], "GET", "/api/queries?run_id=run-does-not-exist")
    assert status == 400, body
    assert json.loads(body)["error"] == "run_not_found"


def test_a_missing_index_is_a_503_not_a_400(api, monkeypatch, tmp_path: Path) -> None:
    """The intentional distinction: a missing serving index blames the server, not the caller."""
    import workbench.backend.operator_config as operator_config

    class Empty:
        WORKBENCH_ARTIFACTS_ROOT = tmp_path / "nowhere"
        WORKBENCH_RESULTS_ROOT = tmp_path / "nowhere" / "results"
        WORKBENCH_THIRD_PARTY_ROOT = tmp_path / "nowhere" / "third_party"
        WORKBENCH_CHECKPOINT_ROOT = tmp_path / "nowhere" / "checkpoints"
        FASHIONIQ_ROOT = tmp_path / "nowhere" / "FashionIQ"

    monkeypatch.setattr(operator_config, "resolve_config", lambda: Empty())
    monkeypatch.setattr(api["module"], "resolve_config", lambda: Empty())
    status, body = asgi_request(api["app"], "GET", "/api/queries?run_id=anything")
    assert status == 503, body
    assert json.loads(body)["error"] == "index_unavailable"
