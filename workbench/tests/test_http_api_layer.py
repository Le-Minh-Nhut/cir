"""The HTTP layer itself: routing, request validation, status codes and serialization.

`test_phase1_acceptance_scenarios.py` calls the handler function directly, which exercises the
queue and the operator but not the ASGI stack. These tests drive the real application object
through the ASGI protocol, so routing, body validation, response models and error status codes are
all exercised.

No HTTP client library is installed in this environment, so the driver below speaks ASGI directly:
it builds the scope, feeds the request body, and collects the response. That is the same interface
Starlette's own test client would use.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


def asgi_request(app, method: str, path: str, body: bytes = b"",
                 query: bytes = b"") -> tuple[int, dict, bytes]:
    """Send one request through the application and return (status, headers, body)."""
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": method.upper(),
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": query,
        "root_path": "",
        "headers": [
            (b"host", b"testserver"),
            (b"content-type", b"application/json"),
            (b"content-length", str(len(body)).encode()),
        ],
        "client": ("127.0.0.1", 50123),
        "server": ("testserver", 80),
    }
    received = {"sent": False}
    response: dict = {"status": None, "headers": [], "body": b""}

    async def receive() -> dict:
        if not received["sent"]:
            received["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    import asyncio

    async def run() -> None:
        await app(scope, receive, send)

    asyncio.run(run())
    headers = {key.decode().lower(): value.decode() for key, value in response["headers"]}
    return response["status"], headers, response["body"]


@pytest.fixture
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """The real FastAPI application, with its data paths redirected into a temporary workspace."""
    import importlib.util

    repo = tmp_path / "repo"
    (repo / "workbench" / "artifacts").mkdir(parents=True)

    spec = importlib.util.spec_from_file_location(
        "http_layer_api", REPOSITORY / "workbench" / "backend" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module.ROOT = repo / "workbench"
    module.load_registry = lambda *a, **k: {
        "protocols": {"fashioniq_original_split": {"literature_split_label": "original"}},
        "models": [{
            "model_id": "synthetic_model", "method_name": "Synthetic",
            "native_protocol": "fashioniq_original_split",
            "supported_protocols": ["fashioniq_original_split"],
            "literature_split_label": "original",
            "fashioniq_layout": "fashioniq_standard",
            "source_available": True, "source_dir": "Synthetic",
            "checkpoint_variants": [{"checkpoint_id": "ckpt", "filename": "m.pt",
                                     "checkpoint_mapping_status": "VERIFIED_METADATA"}],
        }],
    }
    # The plan guard is exercised elsewhere; here the route and the queue hand-off are the
    # subject, so a ready plan is supplied.
    plan = type("P", (), {
        "model_id": "synthetic_model", "checkpoint_id": "ckpt",
        "protocol": "fashioniq_original_split", "dataset_root": repo / "FashionIQ",
        "pin": "a" * 40, "actual_source_commit": "a" * 40,
        "checkpoint": repo / "m.pt", "command": ["true"], "cwd": repo,
    })()
    (repo / "m.pt").write_bytes(b"x")
    monkeypatch.setattr(module, "guarded_plan", lambda *a, **k: (plan, []))
    monkeypatch.setattr(module, "resolve_config", lambda: type("C", (), {
        "WORKBENCH_THIRD_PARTY_ROOT": repo / "third_party",
        "WORKBENCH_ARTIFACTS_ROOT": repo / "workbench" / "artifacts",
        "FASHIONIQ_ROOT": repo / "FashionIQ"})())
    system = __import__("workbench.backend.runtime", fromlist=["runtime"])
    monkeypatch.setattr(system, "require_launch_ready", lambda model: "/usr/bin/python3")
    monkeypatch.setattr(module, "sha256_file", lambda path, **k: "0" * 64)
    # Aggregate reports are served only when the evidence ledger corroborates them; the API
    # fixture stands in for a healthy ledger: one entry per report on disk, carrying the
    # report's real digest and identity. (A separate test below covers the refused case.)
    import hashlib
    from workbench.scripts import evaluate_models as _em

    def _fake_ledger(config):
        reports_dir = config.WORKBENCH_ARTIFACTS_ROOT / "reports"
        entries = []
        if reports_dir.is_dir():
            for path in sorted(reports_dir.glob("*_aggregate.json")):
                try:
                    report = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                entries.append({"run_id": report.get("run_id"), "model_id": report.get("model_id"),
                                "checkpoint_id": report.get("checkpoint_id"),
                                "protocol_id": report.get("protocol_id"),
                                "report_digest": digest, "return_code": 0,
                                "invocation_status": "INVOCATION_COMPLETED"})
        return entries

    monkeypatch.setattr(_em, "read_evidence_ledger", _fake_ledger)
    monkeypatch.setattr(module, "JOBS", type("Q", (), {
        "enqueue": lambda self, *a, **k: {"job_id": "job-1", "status": "queued",
                                          "log_directory": str(repo / "queue_jobs" / "job-1")},
        "list": lambda self: [{"job_id": "job-1", "status": "queued"}],
        "cancel": lambda self, job_id: {"job_id": job_id, "status": "cancelled"},
    })())
    return module


# ------------------------------------------------------------------------- job submission

def test_a_valid_submission_is_accepted_with_a_job_identity(api) -> None:
    status, headers, body = asgi_request(
        api.app, "POST", "/api/jobs",
        json.dumps({"model_id": "synthetic_model", "checkpoint_id": "ckpt",
                    "protocol_id": "fashioniq_original_split", "top_k": 200}).encode())
    assert status == 200, body
    assert headers.get("content-type", "").startswith("application/json")
    payload = json.loads(body)
    assert payload["job_id"]
    assert payload["status"] == "queued"


def test_an_unknown_body_field_is_rejected_by_validation(api) -> None:
    """Pydantic must own the request shape; the handler is never reached with a bad body."""
    status, _, body = asgi_request(
        api.app, "POST", "/api/jobs",
        json.dumps({"model_id": "synthetic_model", "checkpoint_id": "ckpt"}).encode())
    assert status == 422, body
    detail = json.loads(body)["detail"]
    assert any("protocol_id" in str(item.get("loc")) for item in detail), detail


def test_an_out_of_range_top_k_is_rejected(api) -> None:
    """`top_k` is bounded by the field declaration, not by the queue."""
    status, _, body = asgi_request(
        api.app, "POST", "/api/jobs",
        json.dumps({"model_id": "synthetic_model", "checkpoint_id": "ckpt",
                    "protocol_id": "fashioniq_original_split", "top_k": 0}).encode())
    assert status == 422, body


def test_an_unknown_model_returns_404(api) -> None:
    status, _, body = asgi_request(
        api.app, "POST", "/api/jobs",
        json.dumps({"model_id": "nope", "checkpoint_id": "ckpt",
                    "protocol_id": "fashioniq_original_split"}).encode())
    assert status == 404, body
    assert json.loads(body)["detail"]["error"] == "model_not_found"


def test_an_unknown_checkpoint_returns_404(api) -> None:
    status, _, body = asgi_request(
        api.app, "POST", "/api/jobs",
        json.dumps({"model_id": "synthetic_model", "checkpoint_id": "nope",
                    "protocol_id": "fashioniq_original_split"}).encode())
    assert status == 404, body
    assert json.loads(body)["detail"]["error"] == "checkpoint_not_found"


def test_a_malformed_body_is_rejected_before_the_handler(api) -> None:
    status, _, body = asgi_request(api.app, "POST", "/api/jobs", b"{not json")
    assert status == 422, body


# ------------------------------------------------------------------------- job listing

def test_job_status_is_serialized_by_the_route(api) -> None:
    status, _, body = asgi_request(api.app, "GET", "/api/jobs")
    assert status == 200, body
    jobs = json.loads(body)
    assert jobs and jobs[0]["job_id"] == "job-1"
    assert jobs[0]["status"] == "queued"


def test_cancellation_returns_the_job_state(api) -> None:
    status, _, body = asgi_request(api.app, "POST", "/api/jobs/job-1/cancel")
    assert status == 200, body
    assert json.loads(body)["status"] == "cancelled"


def test_cancelling_an_unknown_job_is_a_client_error(api) -> None:
    """A missing job must be a 4xx, not an unhandled exception."""
    def raise_key_error(job_id: str):
        raise KeyError(job_id)

    api.JOBS.cancel = raise_key_error
    status, _, body = asgi_request(api.app, "POST", "/api/jobs/absent/cancel")
    assert status == 400, body


def test_an_unknown_route_is_404(api) -> None:
    """The routing layer is really in play: an unregistered path does not reach a handler."""
    status, _, _ = asgi_request(api.app, "GET", "/api/not-a-route")
    assert status == 404

# ------------------------------------------------------------------------- aggregate reports

def test_benchmark_runs_endpoint_lists_aggregate_reports_only(api, tmp_path) -> None:
    """`/api/benchmark-runs` serves aggregate reports, including runs with no per-query export."""
    reports = tmp_path / "repo" / "workbench" / "artifacts" / "reports"
    reports.mkdir(parents=True)
    good = {
        "artifact_type": "official_aggregate_evaluation_report",
        "run_id": "run-1", "model_id": "synthetic_model", "checkpoint_id": "ckpt",
        "protocol_id": "fashioniq_original_split",
        "benchmark_gate": {"status": "PASS_WITHIN_5_PERCENT", "passed": True},
        "variant_provenance": {"caption_alias_staged": True},
    }
    (reports / "synthetic_model_ckpt_run-1_aggregate.json").write_text(json.dumps(good))
    (reports / "broken_aggregate.json").write_text("{not json")
    (reports / "wrong_type_aggregate.json").write_text(json.dumps({"artifact_type": "something_else"}))
    status, _, body = asgi_request(api.app, "GET", "/api/benchmark-runs")
    assert status == 200
    payload = json.loads(body)
    assert len(payload) == 1 and payload[0]["run_id"] == "run-1"
    assert payload[0]["benchmark_gate"]["passed"] is True
    assert payload[0]["variant_provenance"]["caption_alias_staged"] is True

def test_run_gates_endpoint_keys_by_run_id(api, tmp_path) -> None:
    reports = tmp_path / "repo" / "workbench" / "artifacts" / "reports"
    reports.mkdir(parents=True)
    (reports / "m_c_r_aggregate.json").write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report", "run_id": "r1",
        "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
        "parity_status": "PARITY_VERIFIED"}))
    status, _, body = asgi_request(api.app, "GET", "/api/runs/gate")
    assert status == 200
    payload = json.loads(body)
    assert payload["r1"]["parity_status"] == "PARITY_VERIFIED"

def test_a_report_without_ledger_corroboration_is_not_served(api, tmp_path, capsys) -> None:
    """A parseable aggregate report with no matching evidence-ledger entry is a forgery until
    proven otherwise: the API must not serve it as evidence, whatever its gate fields claim."""
    from workbench.scripts import evaluate_models as _em
    reports = tmp_path / "repo" / "workbench" / "artifacts" / "reports"
    reports.mkdir(parents=True)
    (reports / "m_c_forged_aggregate.json").write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report", "run_id": "forged",
        "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
        "benchmark_gate": {"status": "PASS_WITHIN_5_PERCENT", "passed": True}}))
    # No ledger entry for run_id "forged" at all.
    monkey = pytest.MonkeyPatch()
    monkey.setattr(_em, "read_evidence_ledger", lambda config: [])
    try:
        status, _, body = asgi_request(api.app, "GET", "/api/benchmark-runs")
        assert status == 200
        assert json.loads(body) == []
        status, _, body = asgi_request(api.app, "GET", "/api/runs/gate")
        assert json.loads(body) == {}
    finally:
        monkey.undo()

def test_a_parseable_but_incomplete_report_is_isolated_not_served(api, tmp_path, capsys) -> None:
    """A truncated or corrupted aggregate report must not take the endpoints down.

    The file parses as JSON and carries the right artifact_type, but the fields the endpoints
    index by are gone. It is skipped (with a reason on stderr) and every other report still loads.
    """
    reports = tmp_path / "repo" / "workbench" / "artifacts" / "reports"
    reports.mkdir(parents=True)
    good = {"artifact_type": "official_aggregate_evaluation_report", "run_id": "good",
            "model_id": "m", "checkpoint_id": "c", "protocol_id": "p"}
    bad_cases = [
        {"artifact_type": "official_aggregate_evaluation_report"},                       # everything gone
        {"artifact_type": "official_aggregate_evaluation_report", "run_id": 7},          # wrong type
        {"artifact_type": "official_aggregate_evaluation_report", "run_id": "x",
         "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
         "benchmark_gate": "not-an-object"},                                             # gate wrong type
    ]
    for index, case in enumerate(bad_cases):
        (reports / f"bad{index}_aggregate.json").write_text(json.dumps(case))
    (reports / "good_aggregate.json").write_text(json.dumps(good))
    status, _, body = asgi_request(api.app, "GET", "/api/benchmark-runs")
    assert status == 200
    payload = json.loads(body)
    assert [row["run_id"] for row in payload] == ["good"]
    status, _, body = asgi_request(api.app, "GET", "/api/runs/gate")
    assert status == 200 and list(json.loads(body)) == ["good"]
    stderr = capsys.readouterr().err
    assert "bad0_aggregate.json" in stderr and "missing" in stderr

def test_wrong_typed_nested_gate_fields_are_normalized_not_crashing(api, tmp_path, capsys) -> None:
    """The frontend dereferences benchmark_gate.comparisons as a list of objects and iterates the
    metric maps with Object.entries. A truncated report carrying the wrong shapes must degrade to
    missing evidence, never to a shape the frontend would crash on."""
    reports = tmp_path / "repo" / "workbench" / "artifacts" / "reports"
    reports.mkdir(parents=True)
    base = {"artifact_type": "official_aggregate_evaluation_report",
            "model_id": "m", "checkpoint_id": "c", "protocol_id": "p"}
    # Evidence ledger keys entries by run_id, so each fixture report carries its own.
    (reports / "bad_gate_aggregate.json").write_text(json.dumps(
        {**base, "run_id": "x1", "benchmark_gate": {"comparisons": {"metric": "R@10"}}}))
    (reports / "mixed_elements_aggregate.json").write_text(json.dumps(
        {**base, "run_id": "x2", "benchmark_gate": {"comparisons": [None, "oops", {"metric": "R@10", "relative_error_pct": 2.0}]}}))
    (reports / "bad_variant_aggregate.json").write_text(json.dumps(
        {**base, "run_id": "x3", "variant_provenance": "not-an-object", "category_metrics": ["dress"]}))
    (reports / "null_category_aggregate.json").write_text(json.dumps(
        {**base, "run_id": "x4", "category_metrics": {"dress": None}}))
    status, _, body = asgi_request(api.app, "GET", "/api/benchmark-runs")
    assert status == 200
    payload = json.loads(body)
    assert any(row["benchmark_gate"]["comparisons"] == [] for row in payload)
    assert any(row["benchmark_gate"]["comparisons"] == [{"metric": "R@10", "relative_error_pct": 2.0}]
               for row in payload)
    for row in payload:
        assert isinstance(row["benchmark_gate"]["comparisons"], list)
        assert all(isinstance(c, dict) for c in row["benchmark_gate"]["comparisons"])
        assert isinstance(row["variant_provenance"], dict)
        # A category map whose values are not objects degrades to null: the frontend iterates
        # the values with Object.entries and would crash on null.
        if row["category_metrics"] is not None:
            assert all(isinstance(v, dict) for v in row["category_metrics"].values())

# ------------------------------------------------------------------------- duplicate submissions

def test_a_duplicate_submission_is_refused_atomically_by_the_queue(api) -> None:
    """The handler's pre-check reads the queue outside its lock, so two concurrent POSTs could
    both pass it and both enqueue. The queue must be the one that refuses — it decides
    atomically with the append — and the refusal must reach the client as the documented 409
    body naming the job that already holds the GPU."""
    from workbench.backend.jobs import DuplicateJobError

    mock = api.JOBS
    # The registered snapshot looks empty: the pre-check cannot be what refuses the second POST.
    mock.list = lambda: []
    accepted = []

    def enqueue(*args, **kwargs):
        if accepted:
            raise DuplicateJobError("job-1", "running")
        accepted.append(kwargs)
        return {"job_id": "job-1", "status": "queued",
                "log_directory": "/tmp/queue_jobs/job-1"}

    mock.enqueue = enqueue
    body = json.dumps({"model_id": "synthetic_model", "checkpoint_id": "ckpt",
                       "protocol_id": "fashioniq_original_split"}).encode()

    status, _, first = asgi_request(api.app, "POST", "/api/jobs", body)
    assert status == 200, first
    assert json.loads(first)["job_id"] == "job-1"

    status, _, second = asgi_request(api.app, "POST", "/api/jobs", body)
    assert status == 409, second
    detail = json.loads(second)["detail"]
    assert detail["error"] == "duplicate_job", detail
    assert detail["details"] == {"job_id": "job-1", "status": "running"}, detail

def test_the_queue_is_asked_to_decide_the_duplicate_question(api) -> None:
    """The authoritative refusal only happens if the caller opts into it: passing
    `refuse_duplicate=True` is what binds the check to the queue lock."""
    mock = api.JOBS
    mock.list = lambda: []
    seen = {}

    def enqueue(*args, **kwargs):
        seen.update(kwargs)
        return {"job_id": "job-2", "status": "queued", "log_directory": "/tmp/queue_jobs/job-2"}

    mock.enqueue = enqueue
    status, _, body = asgi_request(
        api.app, "POST", "/api/jobs",
        json.dumps({"model_id": "synthetic_model", "checkpoint_id": "ckpt",
                    "protocol_id": "fashioniq_original_split"}).encode())
    assert status == 200, body
    assert seen.get("refuse_duplicate") is True, seen

# ------------------------------------------------------------------------- application lifecycle

def test_the_app_shutdown_hook_stops_the_queue() -> None:
    """`docker stop` SIGTERMs the backend, which FastAPI answers by running its shutdown
    handlers. An evaluator child leads its own session, so without a hook registered there the
    child is never reaped and keeps the GPU: the lifecycle event itself is the subject here."""
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "lifecycle_api", REPOSITORY / "workbench" / "backend" / "main.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    stopped: list[bool] = []
    module.JOBS = type("Q", (), {"shutdown": lambda self: stopped.append(True) or []})()

    handlers = module.app.router.on_shutdown
    assert handlers, "the application registers no shutdown handler"
    for handler in handlers:
        handler()

    assert stopped == [True], "the shutdown handler must stop the queue (and its job groups)"


def test_cohort_export_with_none_metrics_renders_na_cells(api, tmp_path, monkeypatch) -> None:
    """When a cohort has no matching query rows, its metrics are None.
    Markdown and CSV exports must return HTTP 200 and render explicit N/A cells,
    never raise 500 or output nan / fabricated numbers."""
    cohorts_path = tmp_path / "repo" / "workbench" / "artifacts" / "cohorts.json"
    cohort_data = [{
        "cohort_id": "cohort-no-matches",
        "protocol_id": "fashioniq_original_split",
        "run_ids": ["run-empty-1"],
        "query_ids": [],
        "definition": {"category": "dress"},
        "notes": "",
    }]
    cohorts_path.write_text(json.dumps(cohort_data), encoding="utf-8")
    monkeypatch.setattr(api, "COHORTS_PATH", cohorts_path)

    # 1. Markdown export
    status_md, _, body_md = asgi_request(
        api.app, "GET", "/api/cohorts/cohort-no-matches/export/markdown"
    )
    assert status_md == 200, body_md
    content_md = json.loads(body_md)["content"]
    assert "N/A" in content_md
    assert "nan" not in content_md.lower()
    expected_md_row = "| run-empty-1 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |"
    assert expected_md_row in content_md

    # 2. CSV export
    status_csv, _, body_csv = asgi_request(
        api.app, "GET", "/api/cohorts/cohort-no-matches/export/csv"
    )
    assert status_csv == 200, body_csv
    content_csv = json.loads(body_csv)["content"]
    assert "N/A" in content_csv
    assert "nan" not in content_csv.lower()
    expected_csv_row = "run-empty-1,0,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A"
    assert expected_csv_row in content_csv.replace("\r\n", "\n")

    # 3. JSON export
    status_json, _, body_json = asgi_request(
        api.app, "GET", "/api/cohorts/cohort-no-matches/export/json"
    )
    assert status_json == 200, body_json
    data = json.loads(body_json)
    assert data["metrics"][0]["r1"] is None

    # 4. Computed None metrics with query_ids specified
    cohort_data.append({
        "cohort_id": "cohort-unmatched-queries",
        "protocol_id": "fashioniq_original_split",
        "run_ids": ["run-empty-2"],
        "query_ids": ["nonexistent_q1"],
        "definition": {"category": "shirt"},
        "notes": "",
    })
    cohorts_path.write_text(json.dumps(cohort_data), encoding="utf-8")
    monkeypatch.setattr(
        api,
        "cohort_metrics",
        lambda run_ids, query_ids: [{
            "run_id": r,
            "query_count": 0,
            "r1": None,
            "r5": None,
            "r10": None,
            "r20": None,
            "r50": None,
            "r100": None,
            "mean_target_rank": None,
            "median_target_rank": None,
        } for r in run_ids],
    )
    status_md2, _, body_md2 = asgi_request(
        api.app, "GET", "/api/cohorts/cohort-unmatched-queries/export/markdown"
    )
    assert status_md2 == 200, body_md2
    assert "| run-empty-2 | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |" in json.loads(body_md2)["content"]

    status_csv2, _, body_csv2 = asgi_request(
        api.app, "GET", "/api/cohorts/cohort-unmatched-queries/export/csv"
    )
    assert status_csv2 == 200, body_csv2
    assert "run-empty-2,0,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A" in json.loads(body_csv2)["content"].replace("\r\n", "\n")


# ------------------------------------------------------------------------- startup index lifecycle

def test_first_boot_without_database_builds_index_and_serves_runs(api, tmp_path) -> None:
    """On a fresh boot without workbench.duckdb, startup builds the index and /api/runs answers 200."""
    db_path = tmp_path / "repo" / "workbench" / "artifacts" / "workbench.duckdb"
    db_path.unlink(missing_ok=True)
    assert not db_path.exists()

    for handler in api.app.router.on_startup:
        handler()
    api.wait_for_startup_index(timeout=5.0)
    assert db_path.is_file(), "startup must create workbench.duckdb when absent"

    status, _, body = asgi_request(api.app, "GET", "/api/runs")
    assert status == 200, body
    payload = json.loads(body)
    assert isinstance(payload, list)
    assert payload == []

    # Second boot: existing index must be left untouched.
    mtime_before = db_path.stat().st_mtime_ns
    for handler in api.app.router.on_startup:
        handler()
    api.wait_for_startup_index(timeout=5.0)
    assert db_path.stat().st_mtime_ns == mtime_before, "existing index must not be rebuilt on boot"


def test_startup_index_build_failure_yields_honest_typed_error(api, tmp_path, monkeypatch) -> None:
    """When startup index build fails, /api/runs answers 503 index_unavailable with the real cause."""
    db_path = tmp_path / "repo" / "workbench" / "artifacts" / "workbench.duckdb"
    db_path.unlink(missing_ok=True)
    assert not db_path.exists()

    def failing_rebuild(*args, **kwargs):
        raise RuntimeError("simulated duckdb disk failure")

    monkeypatch.setattr(api, "rebuild_index", failing_rebuild)

    for handler in api.app.router.on_startup:
        handler()
    api.wait_for_startup_index(timeout=5.0)
    assert not db_path.exists()

    status, _, body = asgi_request(api.app, "GET", "/api/runs")
    assert status == 503, body
    err = json.loads(body)
    assert err["error"] == "index_unavailable"
    assert "simulated duckdb disk failure" in err["message"] or "simulated duckdb disk failure" in str(err.get("details"))
    assert not isinstance(err, list), "must return an error payload, never a fabricated list"


def test_runs_reports_index_unavailable_while_startup_build_is_in_progress(api, tmp_path, monkeypatch) -> None:
    """While startup index build is in flight, /api/runs reports 503 rather than fabricating runs."""
    import threading

    db_path = tmp_path / "repo" / "workbench" / "artifacts" / "workbench.duckdb"
    db_path.unlink(missing_ok=True)
    assert not db_path.exists()

    started = threading.Event()
    proceed = threading.Event()

    def slow_rebuild(*args, **kwargs):
        started.set()
        proceed.wait(timeout=5.0)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.touch()
        return 0

    monkeypatch.setattr(api, "rebuild_index", slow_rebuild)

    for handler in api.app.router.on_startup:
        handler()
    assert started.wait(timeout=2.0)

    # While in flight, must report 503 building
    status, _, body = asgi_request(api.app, "GET", "/api/runs")
    assert status == 503, body
    err = json.loads(body)
    assert err["error"] == "index_unavailable"
    assert err["details"]["status"] == "building"

    # Release build and verify 200 once ready
    proceed.set()
    api.wait_for_startup_index(timeout=5.0)
    monkeypatch.setattr(api, "list_runs", lambda *a, **k: [])
    status, _, body = asgi_request(api.app, "GET", "/api/runs")
    assert status == 200, body
    assert json.loads(body) == []
