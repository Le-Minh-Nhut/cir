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
        "parity_status": "PARITY_VERIFIED"}))
    status, _, body = asgi_request(api.app, "GET", "/api/runs/gate")
    assert status == 200
    payload = json.loads(body)
    assert payload["r1"]["parity_status"] == "PARITY_VERIFIED"
