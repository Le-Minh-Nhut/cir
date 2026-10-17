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
        "WORKBENCH_CHECKPOINT_ROOT": repo / "checkpoints",
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
                                "invocation_status": "INVOCATION_COMPLETED",
                                "entry_digest": f"entry-{digest[:16]}"})
        return entries

    monkeypatch.setattr(_em, "read_evidence_ledger", _fake_ledger)
    # The fixture's reader above corroborates by content, not by HMAC, so the ledger key is never
    # consulted -- but the route still asks whether one is *usable*, and on a host carrying two
    # conflicting keys that answer is a refusal (a real behavior a separate test covers). The
    # fixture stands in for a healthy deployment: one usable key.
    monkeypatch.setattr(_em, "ledger_hmac_key", lambda config: "0" * 64)
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

# ------------------------------------------------------------------- verifier-backed evidence

#: A fully valid LIMN report: the macro is the mean of the three categories below and the mean of
#: the two macro recalls, because `verify_run.py` re-derives that arithmetic rather than trusting
#: the report's claim.
VALID_EVIDENCE = {
    "extraction_status": "EXTRACTED_VERIFIED",
    "metric_extraction_available": True,
    "category_metrics": {"dress": {"r10": 25.5, "r50": 55.2},
                         "shirt": {"r10": 28.1, "r50": 58.4},
                         "toptee": {"r10": 30.2, "r50": 60.1}},
    "observed_metrics": {"r10": 27.9333, "r50": 57.9, "mean": 42.9167},
}

def _evidence_report(tmp_path: Path, run_id: str, *, proof: str | None = None,
                     **overrides) -> None:
    """Write one corroborated aggregate report into the fixture workspace.

    The API fixture's ledger corroborates whatever is on disk, so every fixture here is
    *authenticated*: the differences between them are exactly the research-evidence questions the
    verifier owns, which is what makes them able to fail independently.

    `proof` writes the completion-proof file, which selects the certification MODE outright: a
    report with a proof is certified through the authoritative verifier in *any* process, and a
    report without one is certified at ledger level only in a development process whose operator
    opted in (`WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`) -- otherwise it is refused by name.
    """
    reports = tmp_path / "repo" / "workbench" / "artifacts" / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    report = {"artifact_type": "official_aggregate_evaluation_report", "run_id": run_id,
              "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
              "protocol_id": "fashioniq_val_split", "checkpoint_sha256": "c" * 64,
              "return_code": 0, "invocation_status": "INVOCATION_COMPLETED",
              "benchmark_gate": {"status": "PASS_WITHIN_5_PERCENT", "passed": True},
              "parity_status": "PARITY_VERIFIED", **VALID_EVIDENCE, **overrides}
    (reports / f"limn_base_iter0_all_categories_{run_id}_aggregate.json").write_text(
        json.dumps(report))
    if proof is not None:
        (reports / f"limn_base_iter0_all_categories_{run_id}_completion.json").write_text(proof)

def _report_path(tmp_path: Path, run_id: str) -> Path:
    return (tmp_path / "repo" / "workbench" / "artifacts" / "reports" /
            f"limn_base_iter0_all_categories_{run_id}_aggregate.json")

def _write_proof(tmp_path: Path, run_id: str, proof: str) -> None:
    """Write this run's completion proof, the file `main.py` keys the certification mode on."""
    _report_path(tmp_path, run_id).with_name(
        f"limn_base_iter0_all_categories_{run_id}_completion.json").write_text(proof)

def _write_checkpoint_bundle(tmp_path: Path) -> str:
    """The three LIMN category weights the registry's bundle declares, and their manifest digest.

    `base_iter0_all_categories` is a *bundle* checkpoint, so the verifier resolves its three
    declared members under `WORKBENCH_CHECKPOINT_ROOT` and re-hashes them against the manifest the
    proof records. A deployment has exactly these files; the fixture builds them so the fixture
    exercises that path instead of failing on a missing weight.
    """
    from workbench.scripts.evaluate_models import build_artifact_manifest, manifest_digest

    from workbench.backend.registry import checkpoint_by_id, model_by_id

    model = model_by_id("limn")
    bundle = checkpoint_by_id(model, "base_iter0_all_categories")
    root = tmp_path / "repo" / "checkpoints" / "limn"
    root.mkdir(parents=True, exist_ok=True)
    names = []
    for member_id in bundle["bundle_members"]:
        variant = checkpoint_by_id(model, member_id)
        (root / variant["filename"]).write_bytes(f"weights:{member_id}".encode())
        names.append(variant["filename"])
    return manifest_digest(build_artifact_manifest(root, names)) or ""

def _valid_completion_proof(tmp_path: Path, run_id: str) -> str:
    """A completion proof the authoritative verifier accepts end to end.

    Every ingredient `validate_run_manifest` checks is real here: the report digest it names, the
    run directory with its `command.json` and logs, the evaluator's environment identity, the
    source pin of the registered model, the checkpoint bundle's manifest, and the evidence-ledger
    entry the proof claims. This is what makes the deployment shape provable: the proof's
    environment identity is exactly what proof validation needs, so a process serving it must
    reach the verifier.
    """
    import hashlib
    from workbench.backend.registry import model_by_id

    artifacts = tmp_path / "repo" / "workbench" / "artifacts"
    report_path = _report_path(tmp_path, run_id)
    run_dir = artifacts / "logs" / f"limn__base_iter0_all_categories__{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(json.dumps({
        "run_id": run_id, "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split", "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED"}))
    (run_dir / "stdout.log").write_text("recall computed\n")
    (run_dir / "stderr.log").write_text("")
    digest = hashlib.sha256(report_path.read_bytes()).hexdigest()
    logs = {"stdout": (run_dir / "stdout.log").read_text(),
            "stderr": (run_dir / "stderr.log").read_text()}
    return json.dumps({
        "run_id": run_id, "model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
        "protocol_id": "fashioniq_val_split", "completion_status": "COMPLETED", "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED",
        "source_actual_commit": model_by_id("limn")["upstream_commit_sha"],
        "checkpoint_sha256": "c" * 64, "environment_digest": "e" * 64,
        "evidence_digest": hashlib.sha256(json.dumps(logs, sort_keys=True).encode()).hexdigest(),
        "bundle_manifest_digest": _write_checkpoint_bundle(tmp_path),
        "ledger_entry_digest": f"entry-{digest[:16]}",
        "report": str(report_path), "report_digest": digest, "run_directory": str(run_dir)})

def _benchmark_runs(api) -> list[dict]:
    status, _, body = asgi_request(api.app, "GET", "/api/benchmark-runs")
    assert status == 200, body
    return json.loads(body)

def _receipt(api, run_id: str) -> dict:
    """The one served report's verification receipt, asserting the artifact stays visible."""
    rows = _benchmark_runs(api)
    assert [row["run_id"] for row in rows] == [run_id], "the artifact stays visible either way"
    return rows[0]["verification"]

@pytest.fixture
def counted_verifier(monkeypatch):
    """Count every entry into the authoritative verifier without changing its verdict.

    The subject of the mode tests is *which* branch judged the report, so the count is the
    evidence; the verdict stays the real verifier's.
    """
    from workbench.scripts import verify_run
    calls: list[str] = []
    original = verify_run.verify
    monkeypatch.setattr(verify_run, "verify",
                        lambda *args, **kwargs: (calls.append(args[0]) or original(*args, **kwargs)))
    return calls

def test_a_production_process_with_no_proof_refuses_by_name_without_entering_the_verifier(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """DEFECT E (reproduced first): corroboration alone is not evidence.

    The report's bytes match the ledger and the invocation completed, so the artifact is real and
    its execution is authenticated — but `verify_run.py` validates the completion proof, and this
    report has none. It may stay visible; it must not be served as certified research evidence.

    This is a *production* process: no pytest, no development-mode declaration, no opt-in. The
    refusal must therefore name the missing proof, and the verifier must never be entered: there
    is nothing for it to validate.
    """
    _evidence_report(tmp_path, "run-no-proof")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    api._CERTIFICATION_CACHE.clear()

    verification = _receipt(api, "run-no-proof")
    assert verification["artifact_exists"] is True
    assert verification["execution_authenticated"] is True
    assert verification["is_research_evidence"] is False
    assert verification["certified"] is False
    assert verification["development_mode"] is False
    assert "no completion proof" in verification["evidence_reasons"][0], verification["evidence_reasons"]
    assert counted_verifier == [], "a report with no proof must never reach the verifier"

def test_a_development_process_without_the_opt_in_refuses_by_name(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """DEFECT E in its second shape: pytest alone is not a licence to certify.

    The opt-in is what relaxes proof validation, and it is honoured in a development process only.
    A development process that has not opted in refuses a report with no proof exactly as a
    deployment does.
    """
    _evidence_report(tmp_path, "run-dev-no-opt-in")
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "workbench/tests/test_http_api_layer.py::dev")
    api._CERTIFICATION_CACHE.clear()

    verification = _receipt(api, "run-dev-no-opt-in")
    assert verification["certified"] is False
    assert verification["is_research_evidence"] is False
    assert verification["development_mode"] is False
    assert verification["evidence_reasons"], "the refusal must name its reason"
    assert counted_verifier == [], "a refused report has nothing for the verifier to resolve"

def test_a_production_process_with_a_valid_proof_certifies_through_the_authoritative_verifier(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """DEFECT E's inverted half: the deployment *is* the proof-validating case.

    A production process serving a report whose completion proof exists has exactly what proof
    validation needs — the evaluator's environment identity — so the authoritative verifier is the
    branch that judges it. Its verdict is what the receipt carries, verbatim and alone.
    """
    _evidence_report(tmp_path, "run-deploy")
    _write_proof(tmp_path, "run-deploy", _valid_completion_proof(tmp_path, "run-deploy"))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    api._CERTIFICATION_CACHE.clear()

    verification = _receipt(api, "run-deploy")
    assert counted_verifier == ["run-deploy"], "the authoritative verifier must be consulted"
    assert verification["development_mode"] is False
    # The verifier's own verdict is what the receipt carries -- here, its acceptance.
    assert verification["is_research_evidence"] is True, verification["reasons"]
    assert verification["certified"] is True

def test_the_development_opt_in_does_not_apply_to_a_deployment(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """The opt-in is honoured in a *development* process only.

    A deployment that merely exports the variable is still a deployment: its reports are certified
    through the authoritative verifier when a proof exists, and refused by name when none does. The
    opt-in must not turn the missing proof into a ledger-level certificate here.
    """
    _evidence_report(tmp_path, "run-deploy-opt-in")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.setenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", "1")
    api._CERTIFICATION_CACHE.clear()

    verification = _receipt(api, "run-deploy-opt-in")
    assert verification["development_mode"] is False, "the opt-in must not reach a deployment"
    assert verification["certified"] is False
    assert verification["is_research_evidence"] is False
    assert "no completion proof" in verification["evidence_reasons"][0], verification["evidence_reasons"]
    assert counted_verifier == [], "the opt-in must not divert a deployment onto the ledger path"

def test_a_proof_is_never_skipped_for_the_opt_in_in_a_development_process(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """In a development process the opt-in relaxes ledger certification only: a proof still
    selects the authoritative verifier, whose verdict is then the receipt's."""
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WORKBENCH_DEVELOPMENT_MODE", "1")
    monkeypatch.setenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", "1")
    _evidence_report(tmp_path, "run-opt-in-with-proof")
    _write_proof(tmp_path, "run-opt-in-with-proof",
                 _valid_completion_proof(tmp_path, "run-opt-in-with-proof"))
    api._CERTIFICATION_CACHE.clear()

    verification = _receipt(api, "run-opt-in-with-proof")
    assert counted_verifier == ["run-opt-in-with-proof"], "a proof is never skipped for the opt-in"
    assert verification["development_mode"] is False, "a proof is judged in deployment mode"
    assert verification["is_research_evidence"] is True, verification["reasons"]

#: Does a development process certify at ledger level on this host at all?
#:
#: `main.py` certifies through the verifier's own metric rules when there is no proof to validate,
#: so the answer belongs to the host, not to this test: a host with no committed benchmark
#: reference (or an unregistered model) legitimately ends in a refusal. The tests below assert the
#: branch that was taken -- `development_mode` -- and adapt the certified/refused expectation to
#: the rules that branch really applies. Neither branch may ever be a fabrication, and neither may
#: be silently assumed.
def _development_certifies_here() -> bool:
    from workbench.scripts import verify_run
    metric_reasons, _ = verify_run.metric_evidence_reasons(
        "limn", "base_iter0_all_categories", "fashioniq_val_split", VALID_EVIDENCE)
    return not [*metric_reasons, *verify_run._evidence_kind_reasons(VALID_EVIDENCE)]

def test_a_corroborated_report_with_a_rejected_completion_proof_is_not_certified_evidence(
        api, tmp_path, counted_verifier) -> None:
    """A proof file that does not validate is exactly what the authoritative verifier refuses on,
    and a deployment has a proof file, so this is the receipt the API must carry: authenticated
    execution, refused evidence, with the verifier's own reason attached."""
    _evidence_report(tmp_path, "run-bad-proof", proof="{}")
    api._CERTIFICATION_CACHE.clear()
    verification = _receipt(api, "run-bad-proof")
    assert counted_verifier == ["run-bad-proof"], "a proof is judged by the verifier, not skipped"
    assert verification["artifact_exists"] is True
    assert verification["execution_authenticated"] is True
    assert verification["is_research_evidence"] is False
    assert verification["certified"] is False
    assert verification["development_mode"] is False
    # The reason text is the verifier's own sentence, which is the receipt's evidence that the
    # verdict came from `verify_run` and not from a rule re-implemented here.
    assert verification["reasons"][0].startswith("completion proof rejected:"), verification["reasons"]

def test_a_report_whose_extraction_did_not_run_is_not_certified_evidence(
        api, tmp_path, monkeypatch) -> None:
    """The metrics were never obtained: no branch certifies such a report.

    This report has no proof, so a development process judges it at ledger level -- and the
    verifier's extraction rule refuses it *there*, which is what the ledger path exists to keep
    true. A deployment would refuse it for the missing proof instead; both refusals stand.
    """
    monkeypatch.setenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", "1")
    _evidence_report(tmp_path, "run-no-extraction",
                     metric_extraction_available=False, extraction_status="NOT_EXTRACTED")
    api._CERTIFICATION_CACHE.clear()
    verification = _receipt(api, "run-no-extraction")
    assert verification["execution_authenticated"] is True
    assert verification["is_research_evidence"] is False
    assert verification["certified"] is False
    assert any("extraction" in reason for reason in verification["evidence_reasons"])

def test_a_report_whose_macro_contradicts_its_categories_is_not_certified_evidence(
        api, tmp_path, monkeypatch) -> None:
    """Internal arithmetic that does not add up is not evidence, however good it looks."""
    monkeypatch.setenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", "1")
    _evidence_report(tmp_path, "run-bad-macro",
                     observed_metrics={"r10": 90.0, "r50": 95.0, "mean": 92.5})
    api._CERTIFICATION_CACHE.clear()
    verification = _receipt(api, "run-bad-macro")
    assert verification["execution_authenticated"] is True
    assert verification["is_research_evidence"] is False
    assert verification["certified"] is False
    assert verification["evidence_reasons"], "the macro rule must name its reason"

def test_a_production_process_certifies_a_fully_valid_report_through_the_verifier(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """Positive control: a real deployment shape ends in `certified=true`, which is what F1 broke.

    No pytest, no development-mode declaration, no opt-in: the process is a deployment. The report
    is corroborated by the ledger and its completion proof validates, so the authoritative verifier
    is the branch that judges it -- and its `is_research_evidence` is what the receipt carries.
    """
    _evidence_report(tmp_path, "run-valid")
    _write_proof(tmp_path, "run-valid", _valid_completion_proof(tmp_path, "run-valid"))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    api._CERTIFICATION_CACHE.clear()

    rows = _benchmark_runs(api)
    assert [row["run_id"] for row in rows] == ["run-valid"]
    verification = rows[0]["verification"]
    assert counted_verifier == ["run-valid"], "the deployment must consult the authoritative verifier"
    assert verification["is_research_evidence"] is True, verification["reasons"]
    assert verification["certified"] is True
    assert verification["execution_authenticated"] is True
    assert verification["artifact_exists"] is True
    assert verification["development_mode"] is False
    # Both directions of "the receipt reflects the verifier's verdict": an accepted proof yields
    # no reasons at all, and the rejected case below carries the verifier's own refusal sentence.
    assert verification["reasons"] == []
    assert verification["evidence_reasons"] == []
    # The paper-parity verdict stays a separate, visible answer beside the evidence verdict.
    assert rows[0]["parity_status"] == "PARITY_VERIFIED"
    assert rows[0]["benchmark_gate"]["passed"] is True

def test_the_response_distinguishes_artifact_exists_from_valid_evidence(api, tmp_path) -> None:
    """A historical or failed artifact stays visible and is labelled, never hidden.

    Both a certified and a refused report appear, and the four questions are separately readable:
    the refused one is authenticated but not evidence, the certified one is both.
    """
    _evidence_report(tmp_path, "run-good")
    _write_proof(tmp_path, "run-good", _valid_completion_proof(tmp_path, "run-good"))
    _evidence_report(tmp_path, "run-bad",
                     observed_metrics={"r10": 1.0, "r50": 2.0, "mean": 1.5})
    _write_proof(tmp_path, "run-bad", _valid_completion_proof(tmp_path, "run-bad"))
    rows = {row["run_id"]: row for row in _benchmark_runs(api)}
    assert set(rows) == {"run-good", "run-bad"}, "both artifacts remain visible"
    assert rows["run-good"]["verification"]["certified"] is True
    assert rows["run-bad"]["verification"]["certified"] is False
    for row in rows.values():
        assert row["verification"]["artifact_exists"] is True
        assert row["verification"]["execution_authenticated"] is True
    # The gate endpoint carries the same receipt, so a joined `/api/runs` record cannot lose it.
    status, _, body = asgi_request(api.app, "GET", "/api/runs/gate")
    assert status == 200
    gates = json.loads(body)
    assert gates["run-good"]["verification"]["is_research_evidence"] is True
    assert gates["run-bad"]["verification"]["is_research_evidence"] is False

def test_a_proof_gaining_report_is_re_judged_never_re_served_a_development_receipt(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """DEFECT F5 (reproduced first): a receipt computed under one mode must not outlive the mode.

    A development process opts in and serves a report with no proof; the ledger-level receipt is
    cached. The evaluator then writes the completion proof for that same run -- the bytes of the
    report and of the ledger are untouched, so digest and ledger stamp are unchanged. The next
    request must judge the report under the *deployment* mode (the proof's own identity is what
    validation needs) and enter the authoritative verifier, instead of serving the stale
    development receipt whose `development_mode` claimed the proof had not been validated here.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WORKBENCH_DEVELOPMENT_MODE", "1")
    monkeypatch.setenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", "1")
    _evidence_report(tmp_path, "run-gains-proof")
    api._CERTIFICATION_CACHE.clear()
    before = _receipt(api, "run-gains-proof")
    assert before["development_mode"] is True, "the opt-in must reach the development branch"
    assert _development_certifies_here() is before["certified"], (
        "the development branch's verdict must be the verifier's own rules, not a constant: "
        f"{before}")

    assert counted_verifier == [], "the development branch must not enter the verifier"
    _write_proof(tmp_path, "run-gains-proof",
                 _valid_completion_proof(tmp_path, "run-gains-proof"))
    after = _receipt(api, "run-gains-proof")
    assert counted_verifier == ["run-gains-proof"], (
        f"the proof-gaining request must be re-judged by the verifier, saw {counted_verifier}")
    assert after["development_mode"] is False, "a proof must never be served under a development receipt"
    assert after["is_research_evidence"] is True, after["reasons"]

def test_the_certification_mode_is_part_of_the_cache_key(api, tmp_path, monkeypatch) -> None:
    """F5's missing key dimension, isolated: only the MODE changes between the two requests.

    The report bytes, the ledger and the proof's (non-)existence are byte-for-byte identical across
    the two requests; the first is refused because no proof exists and no opt-in was given, the
    second is certified at ledger level because the operator opted in. The receipt computed for the
    first mode must not answer for the second, which is why the mode sits in the key beside the
    digest, the ledger stamp and the proof stamp.
    """
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("WORKBENCH_DEVELOPMENT_MODE", "1")
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    _evidence_report(tmp_path, "run-by-mode")
    api._CERTIFICATION_CACHE.clear()

    refused = _receipt(api, "run-by-mode")
    assert refused["development_mode"] is False
    assert "no completion proof" in refused["evidence_reasons"][0], refused["evidence_reasons"]

    monkeypatch.setenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", "1")
    certified = _receipt(api, "run-by-mode")
    assert certified["development_mode"] is True, "the mode change must be recomputed, not cached"
    assert certified["is_research_evidence"] is _development_certifies_here(), certified
    assert len(api._CERTIFICATION_CACHE) == 2, (
        "a receipt computed under one certification mode must never answer for another")

def test_a_corroborated_report_is_verified_once_per_ledger_revision(
        api, tmp_path, counted_verifier) -> None:
    """The endpoint must not re-hash multi-GB checkpoints per dashboard request.

    The receipt is the expensive part (the authoritative verifier validates the completion proof,
    which re-hashes the checkpoint members), so it is cached on the report digest and the
    certification mode; a second request must not enter the verifier again.
    """
    # A proof file puts the request on the authoritative verifier path, which is the expensive one
    # the receipt exists to avoid repeating. Its content is a stub: the cost being measured is the
    # verifier being entered, not a pass.
    _evidence_report(tmp_path, "run-cached")
    _write_proof(tmp_path, "run-cached", _valid_completion_proof(tmp_path, "run-cached"))
    api._CERTIFICATION_CACHE.clear()
    first = _benchmark_runs(api)
    second = _benchmark_runs(api)
    assert len(counted_verifier) == 1, (
        f"the verifier must run once per report per ledger revision, saw {len(counted_verifier)}")
    assert first == second
    assert first[0]["verification"]["certified"] is True, first[0]["verification"]["reasons"]

# ------------------------------------------------------------------------- A7 certification policy

def test_a7_report_with_no_proof_is_never_certified_true(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """A7 (a): a report with no proof is never certified=true."""
    _evidence_report(tmp_path, "run-a7-no-proof")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    api._CERTIFICATION_CACHE.clear()

    verification = _receipt(api, "run-a7-no-proof")
    assert verification["certified"] is False
    assert verification["is_research_evidence"] is False
    assert verification["execution_authenticated"] is True
    assert verification["state"] == "proof_unavailable"
    assert verification["certification_mode"] == "proof_unavailable"
    assert counted_verifier == []


def test_a7_api_distinguishes_no_proof_available_from_no_ledger_corroboration(
        api, tmp_path, monkeypatch) -> None:
    """A7 (b): the API distinguishes 'no proof available' from 'no ledger corroboration' in its payload."""
    # 1. Ledger-corroborated report whose completion proof is unavailable
    _evidence_report(tmp_path, "run-a7-proof-unavailable")
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    api._CERTIFICATION_CACHE.clear()

    receipt_proof_unavailable = _receipt(api, "run-a7-proof-unavailable")
    assert receipt_proof_unavailable["execution_authenticated"] is True
    assert receipt_proof_unavailable["state"] == "proof_unavailable"
    assert receipt_proof_unavailable["certification_mode"] == "proof_unavailable"
    assert receipt_proof_unavailable["certified"] is False
    assert any("no completion proof" in r for r in receipt_proof_unavailable["evidence_reasons"])

    # 2. No ledger corroboration: omitted from benchmark runs, and gate projection marks uncorroborated
    receipt_uncorroborated = api._gate_fields({"run_id": "run-a7-uncorroborated"})["verification"]
    assert receipt_uncorroborated["execution_authenticated"] is False
    assert receipt_uncorroborated["state"] == "uncorroborated"
    assert receipt_uncorroborated["certification_mode"] == "uncorroborated"
    assert receipt_uncorroborated["certified"] is False

    # Distinct states in payload
    assert receipt_proof_unavailable["execution_authenticated"] != receipt_uncorroborated["execution_authenticated"]
    assert receipt_proof_unavailable["state"] != receipt_uncorroborated["state"]
    assert receipt_proof_unavailable["certification_mode"] != receipt_uncorroborated["certification_mode"]


def test_a7_launcher_reuses_ledger_corroborated_run_without_proof_and_queues_when_uncorroborated(
        tmp_path) -> None:
    """A7 (c): launcher reuses completed run whose proof is gone, and queues fresh run when uncorroborated."""
    from workbench.tests.test_pc_launcher import setup_sandbox, run_pc, _create_corroborated_run

    # (c-1) Reuses completed run whose proof is gone instead of queueing duplicate
    sandbox1, env1 = setup_sandbox(tmp_path / "reused", with_env=True)
    env1["MOCK_RUNS_STATUS"] = "corroborated"
    post_jobs1 = sandbox1 / "post_jobs.log"
    env1["MOCK_POST_JOBS_FILE"] = str(post_jobs1)
    _create_corroborated_run(sandbox1, "run_limn_corr_01", omit_proof=True)

    proc1 = run_pc(sandbox1, [], env1)
    assert proc1.returncode == 0
    assert "Tái sử dụng kết quả có sẵn" in proc1.stdout
    assert "Phát hiện kết quả đánh giá đã hoàn thành và được sổ cái chứng thực" in proc1.stdout
    assert not post_jobs1.exists() or len(post_jobs1.read_text().strip()) == 0

    # (c-2) Queues fresh run when genuinely no corroboration (forged digest)
    sandbox2, env2 = setup_sandbox(tmp_path / "forged", with_env=True)
    env2["MOCK_RUNS_STATUS"] = "corroborated"
    post_jobs2 = sandbox2 / "post_jobs.log"
    env2["MOCK_POST_JOBS_FILE"] = str(post_jobs2)
    _create_corroborated_run(sandbox2, "run_limn_corr_01", forged_digest=True)

    proc2 = run_pc(sandbox2, [], env2)
    assert "Tái sử dụng kết quả có sẵn" not in proc2.stdout
    assert post_jobs2.exists() and len(post_jobs2.read_text().strip().splitlines()) > 0


def test_a7_report_with_invalid_proof_is_refused_api(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """A7 (d-1): report with an invalid proof is refused by API verifier, not served as certified."""
    _evidence_report(tmp_path, "run-a7-invalid-proof", proof=json.dumps({"completion_status": "FAILED", "return_code": 1}))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    api._CERTIFICATION_CACHE.clear()

    verification = _receipt(api, "run-a7-invalid-proof")
    assert verification["certified"] is False
    assert verification["is_research_evidence"] is False
    assert verification["execution_authenticated"] is True
    assert verification["certification_mode"] == "deployment"
    assert verification["state"] == "refused"
    assert counted_verifier == ["run-a7-invalid-proof"]


def test_a7_report_with_invalid_proof_is_refused_launcher(tmp_path) -> None:
    """A7 (d-2): launcher dedup refuses invalid proof and queues fresh evaluation."""
    from workbench.tests.test_pc_launcher import setup_sandbox, run_pc, _create_corroborated_run

    sandbox, env = setup_sandbox(tmp_path / "launcher_invalid", with_env=True)
    env["MOCK_RUNS_STATUS"] = "corroborated"
    post_jobs = sandbox / "post_jobs.log"
    env["MOCK_POST_JOBS_FILE"] = str(post_jobs)
    _create_corroborated_run(sandbox, "run_limn_corr_01", omit_proof=False)
    proof_file = sandbox / "workbench" / "artifacts" / "reports" / "limn_base_iter0_all_categories_run_limn_corr_01_completion.json"
    proof_file.write_text(json.dumps({"completion_status": "FAILED", "return_code": 1}))

    proc = run_pc(sandbox, [], env)
    assert "Tái sử dụng kết quả có sẵn" not in proc.stdout
    assert post_jobs.exists() and len(post_jobs.read_text().strip().splitlines()) > 0

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

# ------------------------------------------------- checkpoint-bound receipt cache
def _checkpoint_root(tmp_path: Path) -> tuple[Path, list[str]]:
    """The LIMN bundle's weight directory and declared member filenames, as the run resolved them."""
    from workbench.backend.registry import checkpoint_by_id, model_by_id

    model = model_by_id("limn")
    bundle = checkpoint_by_id(model, "base_iter0_all_categories")
    names = [checkpoint_by_id(model, member_id)["filename"] for member_id in bundle["bundle_members"]]
    return tmp_path / "repo" / "checkpoints" / "limn", names

def _certified_deployment_run(api, tmp_path, monkeypatch, run_id: str) -> dict:
    """One report the authoritative verifier accepts in a production process, with its proof."""
    _evidence_report(tmp_path, run_id)
    _write_proof(tmp_path, run_id, _valid_completion_proof(tmp_path, run_id))
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.delenv("WORKBENCH_DEVELOPMENT_MODE", raising=False)
    monkeypatch.delenv("WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES", raising=False)
    api._CERTIFICATION_CACHE.clear()
    receipt = _receipt(api, run_id)
    assert receipt["certified"] is True, receipt["reasons"]
    return receipt

def test_a_checkpoint_edit_invalidates_an_already_issued_certificate(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """DEFECT A (reproduced first): the cache key ignored the artifact the certificate is about.

    `verify_run` re-hashes the run's declared bundle members and refuses a report whose members no
    longer match the manifest its own proof recorded. The receipt cache was keyed on the report
    digest, the policy mode, the ledger and the proof — every input *except* the checkpoint — so a
    rewritten weight kept serving `certified=true` from cache while a cleared cache refused it.
    Here the report, the ledger and the proof are deliberately left byte-identical: the checkpoint
    member is the only thing that changes, and it must be enough on its own.
    """
    _certified_deployment_run(api, tmp_path, monkeypatch, "run-tamper")
    were_consulted = list(counted_verifier)

    root, names = _checkpoint_root(tmp_path)
    victim = root / names[0]
    before = victim.read_bytes()
    victim.write_bytes(b"different weights, same run")

    # The report, the ledger and the proof are untouched: the claim under test is that the
    # checkpoint's own change is what has to invalidate the receipt.
    after = _receipt(api, "run-tamper")
    assert after["certified"] is False, (
        "a certificate must not survive a change to the artifact it certifies")
    assert after["state"] == "refused"
    assert any("checkpoint" in reason.lower() for reason in after["reasons"]), after["reasons"]
    assert counted_verifier != were_consulted, "the verifier must be re-asked, not the cache trusted"

    # False-FAIL direction: restoring the original bytes must restore the verdict, or the cache
    # key would be invalidating legitimate evidence rather than tracking the artifact.
    victim.write_bytes(before)
    restored = _receipt(api, "run-tamper")
    assert restored["certified"] is True, restored["reasons"]

def test_an_unchanged_checkpoint_does_not_re_enter_the_verifier_per_request(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """The other half of DEFECT A: the fix must not turn every dashboard poll into a re-verification.

    The receipt cache exists because proof validation re-hashes multi-GB members. A key that moved
    on its own would be a false FAIL of the most expensive kind, so the token it adds is asserted
    to be stable across identical requests.
    """
    _certified_deployment_run(api, tmp_path, monkeypatch, "run-stable")
    calls = len(counted_verifier)
    for _ in range(3):
        assert _receipt(api, "run-stable")["certified"] is True
    assert len(counted_verifier) == calls, "an unchanged checkpoint must stay cached"

def test_the_checkpoint_change_token_covers_every_declared_member(
        api, tmp_path, monkeypatch, counted_verifier) -> None:
    """A bundle is three weights: editing any one of them invalidates the certificate.

    Guards the specific shape of DEFECT A's fix — a token over only the first member would have
    passed the single-member test above while leaving two thirds of the bundle uncovered.
    """
    _certified_deployment_run(api, tmp_path, monkeypatch, "run-permember")
    root, names = _checkpoint_root(tmp_path)
    assert len(names) >= 2, "the LIMN bundle is expected to declare several members"
    for name in names:
        original = (root / name).read_bytes()
        (root / name).write_bytes(b"edited " + name.encode())
        assert _receipt(api, "run-permember")["certified"] is False, (
            f"editing member {name} must invalidate the certificate")
        (root / name).write_bytes(original)
    assert _receipt(api, "run-permember")["certified"] is True
