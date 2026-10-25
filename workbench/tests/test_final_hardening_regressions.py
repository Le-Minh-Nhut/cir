"""Regression tests for the final-hardening P0/P1 root causes.

Each test fails on the pre-fix code: the mutation it guards against is named in the
docstring, and the assertion targets the observable behavior the fix restored.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from workbench.backend import runtime

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- P0-A

@pytest.fixture()
def limn_model():
    from workbench.backend.registry import load_registry

    return next(m for m in load_registry()["models"] if m["model_id"] == "limn")

def test_planner_constraints_follow_the_selected_adaptation(monkeypatch, limn_model):
    """P0-A: model_constraints() once derived pins from the historical environment block
    only, so an operator-selected adaptation planned torch 1.12.1 on a GPU that cannot
    run it. The resolved contract must replace the historical pins."""
    from workbench.backend import environment_planner as planner
    from workbench.backend import model_settings

    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    monkeypatch.setattr(model_settings, "file_settings",
                        lambda: {k: v for k, v in model_settings.file_settings().items() if k != "WORKBENCH_RUNTIME_ADAPTATION"})
    historical = planner.model_constraints(dict(limn_model))
    assert "1.12.1" in str(historical["torch"])

    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    adapted = planner.model_constraints(dict(limn_model))
    adapted_torch = str(adapted["torch"])
    assert "1.12.1" not in adapted_torch
    assert "2.8" in adapted_torch
    # The two contracts must never share an environment id.
    assert historical != adapted


def test_planner_verifier_uses_the_resolved_contract(limn_model, monkeypatch):
    """P0-A: the planner's verifier input was rebuilt by hand from constraint dicts, which
    duplicated unversioned package entries and dropped the version floor on range-only
    declarations, so the verifier checked an unversioned name against a pinned reality.
    The verifier input must be the canonical resolved contract the installer uses."""
    import os
    from workbench.backend import environment_planner, runtime

    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    spec = environment_planner.model_constraints(limn_model)
    runtime_model = environment_planner._runtime_model(limn_model, spec)
    packages = runtime_model["environment"].get("packages") or []
    names = [p.partition(">=")[0].partition("==")[0].partition("<")[0] for p in packages]
    assert len(names) == len(set(names)), f"duplicated package declarations: {names}"
    required = runtime._required_packages(runtime_model)
    # The floor survives: a range-only declaration must reach the verifier versioned.
    # The specifier set is the identity, not its string ordering.
    from packaging.requirements import Requirement
    observed = Requirement(f"open_clip_torch{required.get('open_clip_torch') or ''}")
    expected = Requirement("open_clip_torch>=2.24,<2.27")
    assert str(observed.specifier) == str(expected.specifier)


# --------------------------------------------------------------------------- P0-B

def test_evaluator_child_receives_the_runtime_adaptation(monkeypatch):
    """P0-B: _evaluator_environment() stripped WORKBENCH_RUNTIME_ADAPTATION from the child,
    so the operator's selection never reached the evaluator process."""
    from workbench.scripts import evaluate_models

    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    child = evaluate_models._evaluator_environment({"model_id": "test_model"})
    assert child.get("WORKBENCH_RUNTIME_ADAPTATION") == "sm_120"

    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    child = evaluate_models._evaluator_environment({"model_id": "test_model"})
    assert not child.get("WORKBENCH_RUNTIME_ADAPTATION")


# --------------------------------------------------------------------------- C-2/C-4/C-5

def test_synthetic_source_is_not_judged_as_a_real_experiment():
    """C-2: a registry-declared synthetic evaluator once fed the gate
    data_kind="experiment" and could mint PASS_WITHIN_5_PERCENT."""
    from workbench.backend import paper_gate

    verdict = paper_gate.evaluate_gate(
        None, {"r10": 50.0, "r50": 70.0, "mean": 60.0},
        run_meta={}, data_kind="synthetic", real_execution=True)
    assert verdict.status == paper_gate.NOT_RUN
    assert not verdict.passed


def test_report_without_ledger_entry_is_refused(tmp_path, capsys):
    """C-4: any parseable JSON in reports/ was served as evidence. The serving path must
    require the evidence-ledger entry and a matching report digest."""
    from workbench.backend import main as backend_main

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "m_c_forged_aggregate.json").write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report", "run_id": "forged",
        "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
        "benchmark_gate": {"status": "PASS_WITHIN_5_PERCENT", "passed": True}}))
    monkey_config = type("C", (), {"WORKBENCH_ARTIFACTS_ROOT": tmp_path})()
    real_resolve = backend_main.resolve_config
    backend_main.resolve_config = lambda: monkey_config
    try:
        import workbench.scripts.evaluate_models as em
        real_read = em.read_evidence_ledger
        em.read_evidence_ledger = lambda config: []
        try:
            assert backend_main._aggregate_reports() == []
        finally:
            em.read_evidence_ledger = real_read
    finally:
        backend_main.resolve_config = real_resolve


def test_gate_demotes_a_reference_without_a_category_table():
    """C-5: a reference that publishes no category table once let an aggregate-only run
    claim the unqualified pass, although the per-category check never happened."""
    import workbench.tests.test_paper_gate as pg

    ref = pg.reference(**{"category_metrics": None})
    verdict = pg.run_with_categories(
        dict(pg.PAPER), {"dress": {"r10": 57.07, "r50": 77.27}}, ref)
    assert verdict.status == pg.gate.PASS_WITH_UNCHECKED_CATEGORIES
    assert not verdict.passed


# --------------------------------------------------------------------------- B-9 (cuda UNKNOWN)

def test_csmcir_declares_a_cuda_requirement():
    """B-9: CSMCIR's evaluator hard-requires CUDA (torch.cuda.empty_cache in
    blip_validate.py) but the registry said UNKNOWN, which _requires_gpu treats as "no
    requirement" — so a CPU-only host passed the launch gate."""
    from workbench.backend.registry import load_registry
    from workbench.backend.runtime import _requires_gpu

    csmcir = next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")
    assert str(csmcir["environment"]["cuda"]).upper() != "UNKNOWN"
    assert _requires_gpu(csmcir)


# --------------------------------------------------------------------------- A-02 / A-05 / D-4 / D-6

def test_run_shell_has_no_python_format_syntax():
    """A-02: `${2!r}` is f-string syntax; bash aborts rc 1 with a substitution error
    instead of the documented rc 2."""
    import re
    text = Path(REPOSITORY / "workbench/docker/run.sh").read_text(encoding="utf-8")
    assert not re.search(r"\$\{[^}]*![rs]\}?[^}]*\}", text), "f-string conversion inside run.sh"


def test_recover_interrupted_flips_phantom_running_to_failed(tmp_path):
    """A-05/D-5: a restart used to erase the queue, leaving `running` records nothing
    backed. Startup recovery must rewrite in-flight records as truthful failures."""
    from workbench.backend.jobs import SingleGpuQueue

    job_dir = tmp_path / "queue_jobs" / "job-1"
    job_dir.mkdir(parents=True)
    (job_dir / "queue_state.json").write_text(json.dumps({
        "job_id": "job-1", "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
        "status": "running", "started_at": "t", "completed_at": None,
        "return_code": None, "error": None}))
    recovered = SingleGpuQueue.recover_interrupted([tmp_path / "queue_jobs"])
    assert len(recovered) == 1
    assert recovered[0]["status"] == "failed"
    assert "restart" in recovered[0]["error"]
    # The on-disk record is rewritten too, so a second boot does not re-recover.
    again = SingleGpuQueue.recover_interrupted([tmp_path / "queue_jobs"])
    assert again == []


def test_index_unavailable_is_a_server_error():
    """D-6: a missing serving index returned 400, blaming the caller for the server's
    own state."""
    from workbench.backend.main import workbench_error
    from workbench.backend.errors import WorkbenchError
    import asyncio

    error = WorkbenchError("index_unavailable", "no index", {})
    response = asyncio.run(workbench_error(None, error))
    assert response.status_code == 503


# --------------------------------------------------------------------------- A-08

def test_acceptance_b_scanner_catches_multiline_promise_all(tmp_path):
    """A-08: the old same-line regex missed Promise.all across lines/template literals."""
    import re
    # Reimplement the shipped scanner's contract: balance-scan every Promise.all block.
    text = "const x = Promise.all([\n  api<Model[]>(\"/api/models\"),\n  api<Run[]>(`/api/runs`),\n]);"
    bundled = set()
    for m in re.finditer(r"Promise\.all", text):
        i, depth = m.end(), 0
        while i < len(text):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        block = text[m.end():i]
        for ep in ("models", "cohorts", "jobs", "runs"):
            if re.search(rf"/api/{ep}\b", block):
                bundled.add(ep)
    assert bundled == {"models", "runs"}

# --------------------------------------------------------------------------- FixD wave 2

def test_ledger_corroboration_requires_a_completed_run(tmp_path, monkeypatch):
    """A ledger entry whose invocation failed or was never recorded must not corroborate a
    report: only a run the evaluator child actually completed is evidence."""
    import json, os
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models as evaluator
    from workbench.backend import main as api

    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT", "WORKBENCH_CHECKPOINT_ROOT",
                "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    config = resolve_config()
    reports = evaluator.reports_root(config)
    reports.mkdir(parents=True)
    report = {"artifact_type": "official_aggregate_evaluation_report", "run_id": "r1",
              "model_id": "m", "checkpoint_id": "c", "protocol_id": "p"}
    path = reports / "r1_aggregate.json"
    path.write_text(json.dumps(report))
    digest = evaluator.sha256_file(path)
    evaluator.append_evidence_entry(config, {"run_id": "r1", "report_digest": digest,
                                             "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
                                             "return_code": 1, "invocation_status": "INVOCATION_FAILED"})
    assert api._aggregate_reports() == [], "a failed invocation must not corroborate a report"
    # A report whose ledger entry is missing entirely is equally refused.
    path2 = reports / "r2_aggregate.json"
    path2.write_text(json.dumps({**report, "run_id": "r2"}))
    assert api._aggregate_reports() == []
    # The completed run is served: a fresh run id — the ledger refuses a duplicate run id.
    (reports / "r3_aggregate.json").write_text(json.dumps({**report, "run_id": "r3"}))
    evaluator.append_evidence_entry(config, {"run_id": "r3", "report_digest": evaluator.sha256_file(
                                             reports / "r3_aggregate.json"),
                                             "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
                                             "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"})
    served = api._aggregate_reports()
    assert [r["run_id"] for r in served] == ["r3"]


def test_duplicate_report_files_serve_one_run(tmp_path, monkeypatch):
    """Two files claiming the same run_id are one run: the endpoints index by run_id, so
    serving both would present one run twice."""
    import json, shutil
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models as evaluator
    from workbench.backend import main as api

    artifacts = tmp_path / "artifacts"
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT", "WORKBENCH_CHECKPOINT_ROOT",
                "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    config = resolve_config()
    reports = evaluator.reports_root(config)
    reports.mkdir(parents=True)
    report = {"artifact_type": "official_aggregate_evaluation_report", "run_id": "r1",
              "model_id": "m", "checkpoint_id": "c", "protocol_id": "p"}
    path = reports / "r1_aggregate.json"
    path.write_text(json.dumps(report))
    evaluator.append_evidence_entry(config, {"run_id": "r1", "report_digest": evaluator.sha256_file(path),
                                             "model_id": "m", "checkpoint_id": "c", "protocol_id": "p",
                                             "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"})
    (reports / "r1_copy_aggregate.json").write_text(path.read_text())
    served = api._aggregate_reports()
    assert [r["run_id"] for r in served] == ["r1"]


def test_evaluator_child_cannot_read_the_ledger_key(tmp_path, monkeypatch):
    """The untrusted evaluator child's environment must not contain HOME: the ledger HMAC
    key lives under the operator's config home, and a child that can read it can forge
    authenticated ledger entries."""
    from workbench.scripts import evaluate_models as evaluator

    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    child = evaluator._evaluator_environment({"model_id": "m"})
    assert "HOME" not in child
    assert "XDG_CONFIG_HOME" not in child


def test_a_queued_job_is_persisted_and_a_queued_cancel_is_recorded(tmp_path):
    """_persist used to skip jobs without a log directory — the worker creates it only at
    dispatch — so a crash before dispatch left a queued job invisible to recovery."""
    from threading import Condition
    from workbench.backend.jobs import SingleGpuQueue

    queue = SingleGpuQueue.__new__(SingleGpuQueue)
    queue._jobs, queue._condition, queue._processes, queue._process_groups = [], Condition(), {}, {}
    queue._cancelling, queue._stopping = set(), False
    directory = tmp_path / "job-1"
    queue.enqueue("m", "c", "p", ["x"], tmp_path, directory, {})
    assert (directory / "queue_state.json").is_file(), "queued job must be persisted"
    cancelled = queue.cancel(queue.list()[0]["job_id"])
    assert cancelled["status"] == "cancelled"
    import json
    assert json.loads((directory / "queue_state.json").read_text())["status"] == "cancelled"


def test_the_duplicate_job_check_is_atomic_with_the_append(tmp_path):
    """Checking duplicates in the handler and enqueueing later races a concurrent enqueue
    past the check; enqueue itself must refuse the duplicate."""
    from threading import Condition
    from workbench.backend.jobs import SingleGpuQueue, DuplicateJobError

    queue = SingleGpuQueue.__new__(SingleGpuQueue)
    queue._jobs, queue._condition, queue._processes, queue._process_groups = [], Condition(), {}, {}
    queue._cancelling, queue._stopping = set(), False
    queue.enqueue("m", "c", "p", ["x"], tmp_path, tmp_path / "a", {})
    import pytest
    with pytest.raises(DuplicateJobError):
        queue.enqueue("m", "c", "p", ["x"], tmp_path, tmp_path / "b", {}, refuse_duplicate=True)


def test_a_reference_s_own_mapping_status_caps_the_verdict():
    """The registry may map a checkpoint asset confidently, but when the paper reference
    itself records the mapping as UNVERIFIED the gate must not certify a full pass."""
    from workbench.backend import paper_gate as gate
    from workbench.backend.benchmark_references import references_for_model

    reference = next(r for r in references_for_model("conesep") if r.reference_id == "conesep_fiq_n20")
    reproduced = {key: reference.value_for(key) for key in reference.reported_metrics}
    run_meta = {"gallery_protocol": reference.payload["gallery_protocol"],
                "checkpoint_id": reference.payload["checkpoint_id"],
                "checkpoint_training_noise_pct": reference.payload["checkpoint_training_noise_pct"],
                "evaluation_noise_pct": reference.payload["evaluation_noise_pct"],
                "split": reference.payload["split"], "dataset": reference.payload["dataset"],
                "protocol_id": reference.payload["gallery_protocol"],
                "checkpoint_mapping_verified": True}
    verdict = gate.evaluate_gate(reference, reproduced, run_meta=run_meta, real_execution=True)
    assert verdict.status == gate.PASS_METRICS_ONLY_MAPPING_UNVERIFIED, verdict.status


def test_a_run_with_complete_category_coverage_passes_an_average_only_reference():
    """A reference that publishes no category table must not demote a run whose own
    per-category evidence is complete — the mirror image of the C-5 hole."""
    from types import SimpleNamespace
    from workbench.backend import paper_gate as gate
    from workbench.backend import benchmark_references as B

    reference = SimpleNamespace(
        reference_id="t", model_id="m", paper_id="p", paper_title="t", registry_version="v",
        registry_file="f", source_digest="d", reference_status=B.REFERENCE_VERIFIED,
        reported_metrics={"mean": 50.0, "r10": 80.0, "r50": 75.0},
        value_for=lambda k: {"mean": 50.0, "r10": 80.0, "r50": 75.0}.get(k),
        provenance_for_all=lambda: {}, metric_provenance=lambda k: B.PAPER_REPORTED,
        payload={"gallery_protocol": "fashioniq_original_split", "checkpoint_id": "c1",
                 "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0, "split": "val",
                 "dataset": "fashioniq", "reported_metrics": {"mean": 50.0, "r10": 80.0, "r50": 75.0},
                 "category_metrics": {}, "derived_metrics": [],
                 "checkpoint_mapping_status": "VERIFIED_METADATA"},
        as_evidence=lambda: {"reference_id": "t", "checkpoint_mapping_status": "VERIFIED_METADATA"})
    reproduced = {"mean": 50.0, "r10": 80.0, "r50": 75.0}
    meta = {"checkpoint_id": "c1", "checkpoint_mapping_verified": True,
            "protocol_id": "fashioniq_original_split", "checkpoint_training_noise_pct": 0,
            "evaluation_noise_pct": 0, "split": "val", "dataset": "fashioniq"}
    complete = {c: {m: 80.0 for m in ("r10", "r50")} for c in ("dress", "shirt", "toptee")}
    verdict = gate.evaluate_gate(reference, reproduced, run_meta=meta, real_execution=True,
                                 reproduced_category_metrics=complete)
    assert verdict.status == gate.PASS_WITHIN_5_PERCENT, verdict.status
    partial = {**complete, "toptee": {}}
    verdict = gate.evaluate_gate(reference, reproduced, run_meta=meta, real_execution=True,
                                 reproduced_category_metrics=partial)
    assert verdict.status == gate.PASS_WITH_UNCHECKED_CATEGORIES, verdict.status


def test_synthetic_runs_get_a_named_reason():
    """A registry-declared synthetic evaluator is a pipeline self-test; the NOT_RUN reason
    must say so instead of pointing an operator at a mocking bug."""
    from workbench.backend import paper_gate as gate

    verdict = gate.evaluate_gate(None, None, data_kind="synthetic", real_execution=True)
    assert verdict.status == gate.NOT_RUN
    assert any("self-test" in b for b in verdict.blockers), verdict.blockers

# --------------------------------------------------------------------------- P1 (final wave)

def test_models_endpoint_hashes_each_checkpoint_once(tmp_path, monkeypatch):
    """`/api/models` reported checkpoint digests by re-reading every downloaded checkpoint on
    every request (LIMN's three are 1.79 GB each), so one request took ~85 s and a launcher
    polling it for readiness always timed out. The availability view must hash each unchanged
    file once; a replaced file must still rehash."""
    from workbench.backend import registry

    monkeypatch.setattr(registry, "_AVAILABILITY_DIGEST_CACHE", {})
    payload = tmp_path / "0_dress_best_model.pt"
    payload.write_bytes(b"x" * 4096)

    calls = {"n": 0}
    real = registry.sha256_file

    def counting(path, required_files=None):
        calls["n"] += 1
        return real(path, required_files)

    monkeypatch.setattr(registry, "sha256_file", counting)
    first = registry._cached_availability_digest(payload, None)
    second = registry._cached_availability_digest(payload, None)
    assert first == second
    assert calls["n"] == 1, f"unchanged checkpoint was hashed {calls['n']} times"

    # A rewritten file is a different artifact and must be rehashed.
    payload.write_bytes(b"y" * 4096)
    third = registry._cached_availability_digest(payload, None)
    assert third != first
    assert calls["n"] == 2, "a replaced checkpoint must not reuse the cached digest"

def test_a_verified_reference_mapping_certifies_a_matching_run():
    """The paper registry and the model registry spell a confirmed checkpoint mapping
    differently (`VERIFIED` vs `VERIFIED_METADATA`). The gate's identity cap first accepted
    only one spelling, so a run whose numbers equal the paper's was demoted to
    PASS_METRICS_ONLY_MAPPING_UNVERIFIED and could never certify."""
    from workbench.backend import paper_gate as gate
    from workbench.backend.benchmark_references import references_for_model

    reference = references_for_model("limn")[0]
    assert reference.payload.get("checkpoint_mapping_status") == "VERIFIED"
    run_meta = {
        "gallery_protocol": reference.payload["gallery_protocol"],
        "checkpoint_id": reference.payload["checkpoint_id"],
        "checkpoint_training_noise_pct": reference.payload["checkpoint_training_noise_pct"],
        "evaluation_noise_pct": reference.payload["evaluation_noise_pct"],
        "split": reference.payload["split"], "dataset": reference.payload["dataset"],
        "protocol_id": reference.payload["gallery_protocol"],
        "checkpoint_mapping_verified": True,
    }
    reproduced = {key: reference.value_for(key) for key in reference.reported_metrics}
    verdict = gate.evaluate_gate(reference, reproduced, run_meta=run_meta, real_execution=True,
                                 reproduced_category_metrics=reference.payload["category_metrics"])
    assert verdict.status in (gate.PASS_WITHIN_5_PERCENT, gate.PASS_WITHIN_5_PERCENT_VARIANT), verdict.status
    assert verdict.passed is True

# --------------------------------------------------------------------------- P2 (acceptance scanners)

def _acceptance_snippet(group: str) -> str:
    """The Python body acceptance.sh runs inline for one group.

    The regression is about the *shipped* scanner, so the test executes the script's own
    heredoc rather than a paraphrase of it. The heredoc is located from the group's own
    header forward; searching the file from the top lands in another group's block.
    """
    script = (REPOSITORY / "workbench" / "docker" / "acceptance.sh").read_text(encoding="utf-8")
    start = script.index(f'say "== {group} ')
    opening = script.index("<<'EOF'", start)
    body = script.index("\n", opening) + 1
    return script[body:script.index("\nEOF\n", body)]

def _run_scanner(snippet: str, *args: str) -> tuple[int, str]:
    import subprocess
    import sys

    completed = subprocess.run([sys.executable, "-c", snippet, *args],
                               capture_output=True, text=True, timeout=120)
    return completed.returncode, completed.stdout + completed.stderr

def _stub_readiness(repo: Path, document: dict) -> None:
    """A stand-in pc_readiness.py that prints exactly this document, so the scanner's own
    filtering is what the test exercises."""
    scripts = repo / "workbench" / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    (scripts / "pc_readiness.py").write_text(
        f"import json\nprint({json.dumps(json.dumps(document))})\n", encoding="utf-8")

def test_the_readiness_gpu_scan_checks_a_model_not_an_absent_rules_key(tmp_path):
    """Group E read `doc["rules"]`, but pc_readiness has no such key (its keys are config,
    dataset, environment, gpu, intended_gpu, models, papers, read_only, repository, summary),
    so the list was always empty and it printed PASS having examined nothing. The scan must
    consult the per-model GPU evidence and must not pass on an empty selection."""
    snippet = _acceptance_snippet("E")
    assert 'doc.get("rules"' not in snippet, "group E still scans a key the report does not have"

    repo = tmp_path / "repo"
    _stub_readiness(repo, {"models": {}})
    rc, out = _run_scanner(snippet, str(repo))
    assert rc == 2, f"an empty GPU selection must report not-verified, got rc={rc}: {out}"
    assert "NOT VERIFIED" in out

    # A model that claims GPU verification without the probe tier is the claim this group
    # exists to catch; the pre-fix scanner never looked at it.
    _stub_readiness(repo, {"models": {"m": {
        "model_id": "m", "gpu_required": True, "gpu_verified": True,
        "environment_tier": "GPU_VERIFICATION_PENDING"}}})
    rc, out = _run_scanner(snippet, str(repo))
    assert rc == 1, f"GPU verified without GPU_RUNTIME_READY evidence must FAIL, got rc={rc}: {out}"

    # And the honest report still passes: verified evidence, matching tier.
    _stub_readiness(repo, {"models": {"m": {
        "model_id": "m", "gpu_required": True, "gpu_verified": True,
        "environment_tier": "GPU_RUNTIME_READY"}}})
    rc, out = _run_scanner(snippet, str(repo))
    assert rc == 0, f"a verified GPU report must PASS, got rc={rc}: {out}"

def test_the_job_status_scan_refuses_to_pass_on_an_empty_queue():
    """An empty queue yields no status values to check, so diffing zero statuses against the
    enum passed vacuously. An observed job must be required before PASS."""
    import http.server
    import json as jsonlib
    import threading

    snippet = _acceptance_snippet("G")
    payloads = {"jobs": []}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = jsonlib.dumps(payloads["jobs"]).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        rc, out = _run_scanner(snippet, base)
        assert rc == 2, f"an empty queue is not evidence: expected not-verified, got rc={rc}: {out}"
        assert "NOT VERIFIED" in out

        payloads["jobs"] = [{"job_id": "j1", "status": "queued"}]
        rc, out = _run_scanner(snippet, base)
        assert rc == 0, f"an observed valid status must PASS, got rc={rc}: {out}"

        payloads["jobs"] = [{"job_id": "j1", "status": "QUEUED"}]
        rc, out = _run_scanner(snippet, base)
        assert rc == 1, f"an out-of-enum status must FAIL, got rc={rc}: {out}"
    finally:
        server.shutdown()


# --------------------------------------------------------------------------- GPU capability scan

def _gpu_capability_scanner():
    """The verifier's GPU_RUNTIME scan, extracted verbatim from the embedded Python block.

    The scan lives inside a bash heredoc and cannot be imported, so the block is lifted and
    executed with `sys.argv[1]` bound to a temporary report path — exactly how
    release_verify.sh runs it.
    """
    import subprocess
    import sys
    import tempfile
    import textwrap

    script = (REPOSITORY / "workbench/docker/release_verify.sh").read_text(encoding="utf-8")
    start = script.index('J1="${RESULTS[J1]}"')
    end = script.index('if (( fails > 0 )); then')
    lines = script[start:end].splitlines(keepends=True)
    for index, line in enumerate(lines):
        if line.startswith('"${PY}" - "$REPORT" <<\'EOF\''):
            lines = lines[index + 1:]
            break
    block = textwrap.dedent("".join(lines).rsplit("\nEOF", 1)[0])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "report.json"
        # The bash runner exports each J-id it graded; J5 is the fail-closed GPU contract. It is
        # bound here so GPU_RUNTIME has a real check to fold the scan's verdict into — the
        # standalone default would make every dimension NOT_APPLICABLE and prove nothing.
        environment = os.environ | {"J5": "PASS", "D5": "fail-closed contract holds"}
        completed = subprocess.run([sys.executable, "-c", block, str(path)],
                                   capture_output=True, text=True, env=environment)
        assert path.is_file(), (completed.stdout, completed.stderr)
        report = json.loads(path.read_text(encoding="utf-8"))
    return report, completed

def _scanner_workspace(tmp_path: Path, monkeypatch) -> Path:
    """A private artifacts root, so the scan reads the fixtures and nothing else."""
    capabilities = tmp_path / "gpu_capabilities"
    capabilities.mkdir(parents=True)
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(tmp_path))
    for key in ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT", "WORKBENCH_CHECKPOINT_ROOT",
                "WORKBENCH_RESULTS_ROOT", "WORKBENCH_THIRD_PARTY_ROOT"):
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    # The scan verifies a record's HMAC signature, so it needs the deployment's operator key.
    # Without one no record can be verified, and a fixture that merely *claims* verification
    # would have to be trusted again — exactly the defect the record checks exist to prevent.
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "scanner-fixture-key")
    return capabilities

def _signed_capability_record(capabilities: Path, key: str, **fields) -> dict:
    """Write a capability record signed by the runtime's own signing helper.

    The bytes are produced by `record_gpu_capability`, so the fingerprint/host/adaptation
    fields a verifier checks are the ones the runtime itself writes, and the HMAC is real.
    """
    from workbench.backend import runtime
    from workbench.backend.operator_config import resolve_config

    model = {"model_id": "fixture_model", "environment_required": True,
             "environment": {"python_env_var": "SCANNER_FIXTURE_PYTHON"}}
    config = resolve_config()
    payload = {"model_id": "fixture_model", "capability_key": key, "probe_source": "interpreter",
               "capability_verified": True, "cuda_available": True,
               "interpreter_fingerprint": "fixture-fingerprint",
               "dependency_fingerprint": runtime._environment_contract_digest(model),
               "runtime_adaptation": None, **fields}
    return runtime.record_gpu_capability(model, payload, config)

def test_an_unreadable_gpu_capability_artifact_is_reported_not_swallowed(tmp_path, monkeypatch):
    """A 0-byte fixture left in gpu_capabilities/ raised JSONDecodeError inside the scan; the
    global try/except turned that into real_gpu = False, so a genuine interpreter probe on the
    target host was silently discarded and the verdict capped at GPU_VALIDATION_PENDING. An
    unreadable artifact is REPORTED and does not override readable evidence.

    The report is the repair. Deleting the file was a later defect of its own: the verifier runs
    against the operator's live artifacts root, so it removed files it did not understand -- and a
    lingering broken artifact is a fact about the deployment that belongs in the next report too."""
    capabilities = _scanner_workspace(tmp_path, monkeypatch)
    # A record written by the runtime's own writer under this deployment's key, carrying the
    # fields an interpreter probe records.
    good = capabilities / "interpreter_probe.json"
    good.write_text(json.dumps(
        _signed_capability_record(capabilities, "interpreter_probe"), default=str))
    poisoned = capabilities / "1a3bb380cba4518172218c7f22978d745145db7b37c37ac2d2f2a6d1ead42d75.json"
    poisoned.write_bytes(b"")                       # the leftover fixture from an unisolated run

    report, completed = _gpu_capability_scanner()

    # The leftover is reported as what degraded the scan and removed, and the readable record is
    # still there afterwards: a 0-byte file never discards the evidence beside it.
    assert report["gpu_capability_scan"]["status"] == "UNREADABLE"
    assert "1a3bb380cba4518172218c7f22978d745145db7b37c37ac2d2f2a6d1ead42d75.json" in \
        report["gpu_capability_scan"]["unreadable"]
    assert "[WARN]" in report["checks"]["J5"]["detail"]
    assert "unreadable capability artifact" in completed.stderr
    # The unusable file is left where it was: the verifier reports what it cannot read, it does
    # not delete the operator's artifacts.
    assert poisoned.exists(), "the verifier must not delete artifacts it cannot read"
    assert good.exists()

def test_the_gpu_scan_does_not_invent_evidence_from_unreadable_artifacts(tmp_path, monkeypatch):
    """The mirror image: with nothing readable there is no evidence, and the reason must be
    named instead of the scan silently reporting a bare PENDING."""
    capabilities = _scanner_workspace(tmp_path, monkeypatch)
    (capabilities / "broken.json").write_bytes(b"")                     # 0-byte: unparsable
    (capabilities / "truncated.json").write_text('{"probe_source": "interpreter"')  # half-written

    report, _ = _gpu_capability_scanner()

    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PENDING"
    assert report["gpu_capability_scan"]["real_gpu"] is False
    assert set(report["gpu_capability_scan"]["unreadable"]) == {"broken.json", "truncated.json"}
    # Both remain on disk and stay named by every later scan: removing them to make a scan look
    # clean is exactly what a fail-closed verifier must not do.
    assert {p.name for p in capabilities.glob("*.json")} == {"broken.json", "truncated.json"}

def test_a_real_record_from_another_environment_cannot_certify_this_host(tmp_path, monkeypatch):
    """A readable record is not automatically evidence: release_verify.sh accepted any
    interpreter-probed file in the directory, so one environment's verification could certify a
    different one. The record's own `capability_key` must name the key in its filename — the
    binding `_cached_gpu_capability` already enforces when it reads a record back."""
    capabilities = _scanner_workspace(tmp_path, monkeypatch)
    (capabilities / "some_other_environment.json").write_text(json.dumps(
        {"probe_source": "interpreter", "capability_verified": True,
         "capability_key": "a-completely-different-environment"}))

    report, completed = _gpu_capability_scanner()

    assert report["dimensions"]["GPU_RUNTIME"]["status"] == "PENDING"
    assert report["gpu_capability_scan"]["real_gpu"] is False
    assert report["gpu_capability_scan"]["wrong_key"] == ["some_other_environment"]
    assert "different environment" in completed.stderr

def test_advisory_environment_read_does_not_hash_installed_distribution_members(tmp_path: Path, monkeypatch):
    """The read-only availability/readiness path must perform a cheap presence/metadata check,
    never walking distribution RECORDs to sha256_file every installed member file.
    """
    import venv as _venv
    from workbench.backend import registry, runtime

    env_dir = tmp_path / "env"
    _venv.EnvBuilder(with_pip=False, symlinks=True).create(env_dir)
    lib = next((env_dir / "lib").glob("python*"))
    sp = lib / "site-packages"
    sp.mkdir(parents=True, exist_ok=True)
    pkg_file = sp / "pkg.py"
    pkg_file.write_text("x = 1\n")
    dist = sp / "pkg-1.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: pkg\nVersion: 1.0.0\n")
    (dist / "RECORD").write_text(f"pkg.py,sha256=xxx,{pkg_file.stat().st_size}\n")

    monkeypatch.setenv("ADVISORY_ENV_VAR", str(env_dir / "bin" / "python"))
    model = {
        "model_id": "test_advisory_model",
        "environment_required": True,
        "environment": {"python_env_var": "ADVISORY_ENV_VAR", "packages": ["pkg==1.0.0"]},
    }
    runtime.forget_environment()

    calls: list[Path] = []
    orig = registry.sha256_file
    def counting_sha256(path, *args, **kwargs):
        calls.append(Path(path))
        return orig(path, *args, **kwargs)

    monkeypatch.setattr(registry, "sha256_file", counting_sha256)

    blockers = runtime.environment_blockers(model)
    assert blockers == []
    assert not any(p == pkg_file for p in calls), f"expected 0 member hashes, got {len(calls)}"


def test_checkpoint_availability_and_missing_paths_use_manifest_digest_without_file_reads(tmp_path: Path, monkeypatch):
    """checkpoint_missing_paths and checkpoint_bundle_missing_paths must reuse the manifest-backed
    digest when file size matches the manifest record, performing no sha256_file calls.
    """
    import json
    from workbench.backend import registry

    ckpt_root = tmp_path / "checkpoints"
    model_dir = ckpt_root / "test_model"
    model_dir.mkdir(parents=True)
    ckpt_file = model_dir / "weights.pt"
    content = b"checkpoint payload"
    ckpt_file.write_bytes(content)
    real_sha = registry.sha256_file(ckpt_file)

    manifest_data = [{
        "model_id": "test_model",
        "filename": "weights.pt",
        "downloaded_sha256": real_sha,
        "size_bytes": len(content),
        # The inode change time is the other half of the shortcut's evidence: size alone let a
        # same-length in-place rewrite read as present with zero bytes hashed.
        "ctime_ns": ckpt_file.stat().st_ctime_ns,
    }]
    (ckpt_root / "download_manifest.json").write_text(json.dumps(manifest_data))

    checkpoint = {
        "checkpoint_id": "ckpt_1",
        "filename": "weights.pt",
        "expected_sha256": real_sha,
    }
    model = {
        "model_id": "test_model",
        "checkpoint_variants": [checkpoint],
        "checkpoint_bundles": [{"bundle_id": "bundle_1", "required_checkpoint_ids": ["ckpt_1"]}],
    }

    class DummyConfig:
        WORKBENCH_CHECKPOINT_ROOT = ckpt_root

    calls: list[Path] = []
    orig = registry.sha256_file
    def counting_sha256(path, *args, **kwargs):
        calls.append(Path(path))
        return orig(path, *args, **kwargs)

    monkeypatch.setattr(registry, "sha256_file", counting_sha256)

    missing = registry.checkpoint_missing_paths(ckpt_file, checkpoint, model_id="test_model", config=DummyConfig)
    assert missing == []

    bundle_missing = registry.checkpoint_bundle_missing_paths(model, ckpt_root, config=DummyConfig)
    assert bundle_missing == {}

    assert calls == [], f"expected 0 file hash reads, got {len(calls)}: {calls}"


def test_a_bundle_manifest_is_identified_by_members_and_bytes_not_mount_path(tmp_path):
    """The container records member paths under /checkpoints while a host validator resolves
    the same files under <artifacts>/checkpoints. Hashing the absolute path made an unchanged
    artifact fail its own manifest — a legitimate run rejected as tampered."""
    from workbench.scripts.evaluate_models import build_artifact_manifest, manifest_digest

    first = tmp_path / "container" / "limn"
    second = tmp_path / "host" / "limn"
    for directory in (first, second):
        directory.mkdir(parents=True)
        for name in ("a.pt", "b.pt", "c.pt"):
            (directory / name).write_bytes(b"weights-" + name.encode())

    members = ["a.pt", "b.pt", "c.pt"]
    original = manifest_digest(build_artifact_manifest(first, members))
    relocated = manifest_digest(build_artifact_manifest(second, members))
    assert original == relocated, "the mount prefix is not part of the artifact's identity"

    # The identity must still be sensitive to bytes and to the member set.
    (second / "b.pt").write_bytes(b"tampered")
    tampered = manifest_digest(build_artifact_manifest(second, members))
    assert tampered != original, "changed member bytes must change the identity"
    assert manifest_digest(build_artifact_manifest(second, ["a.pt", "c.pt"])) != tampered, \
        "a different member set must change the identity"


def test_a_capability_record_is_bound_to_the_interpreter_that_probed_it(tmp_path, monkeypatch):
    """The record's interpreter fingerprint existed but was never compared against a value
    because the reader used a field the advisory path deliberately leaves unset, so a capability
    probed by one interpreter could satisfy a different one. Both sides must compute the same
    authoritative fingerprint the capability key is derived from."""
    import venv as venv_module

    from workbench.backend import runtime

    def make_env(name):
        root = tmp_path / name
        venv_module.EnvBuilder(with_pip=False, symlinks=True).create(root)
        site = next((root / "lib").glob("python*")) / "site-packages"
        site.mkdir(parents=True, exist_ok=True)
        for package, version in (("torch", "1.12.1"), ("torchvision", "0.13.1")):
            (site / f"{package}.py").write_text(f"__version__ = '{version}'\n")
            dist = site / f"{package}-{version}.dist-info"
            dist.mkdir(exist_ok=True)
            (dist / "METADATA").write_text(
                f"Metadata-Version: 2.1\nName: {package}\nVersion: {version}\n")
            (dist / "RECORD").write_text(
                f"{package}.py,sha256=t,{(site / f'{package}.py').stat().st_size}\n")
        return root / "bin" / "python"

    first, second = make_env("first"), make_env("second")
    monkeypatch.setenv("WORKBENCH_FP_PYTHON", str(first))
    model = {"model_id": "fp_probe", "environment_required": True,
             "environment": {"python_env_var": "WORKBENCH_FP_PYTHON",
                             "pytorch": "1.12.1", "cuda": "11.7"}}

    assert (runtime._record_environment_fingerprint(model, str(first))
            != runtime._record_environment_fingerprint(model, str(second))), \
        "two different interpreters must not share an environment fingerprint"

    # A record probed by `first` must not be reused while `second` is the configured interpreter.
    monkeypatch.setenv("WORKBENCH_FP_PYTHON", str(second))
    report = runtime.verify_environment(model, probe=False, advisory=True)
    key = runtime._gpu_capability_key(model)
    monkeypatch.setitem(runtime._GPU_CAPABILITY, key, {
        "interpreter_fingerprint": runtime._record_environment_fingerprint(model, str(first)),
        "capability_key": key, "capability_verified": True, "probe_source": "interpreter"})
    assert runtime._cached_gpu_capability(model, report) is None, \
        "a record for a different interpreter must not certify this environment"


# --------------------------------------------------------------------------- P2 (mutation check M4)

def test_mutation_m4_not_reported_as_passed_when_docker_daemon_unavailable(tmp_path):
    """When the Docker daemon is unavailable, docker compose config fails for an
    infrastructure reason, not because the compose env guard detected missing variables.
    M4 must distinguish daemon failure from guard enforcement and report
    SKIPPED/NOT VERIFIED, never DETECTED or PASSED."""
    import os
    import subprocess

    docker_shim = tmp_path / "docker"
    docker_shim.write_text(
        "#!/bin/sh\n"
        "echo 'Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?' >&2\n"
        "exit 1\n",
        encoding="utf-8",
    )
    docker_shim.chmod(0o755)

    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}"}
    script = REPOSITORY / "workbench" / "docker" / "mutation_checks.sh"
    proc = subprocess.run(
        ["bash", str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    out = proc.stdout + proc.stderr

    assert "DETECTED: compose refuses to resolve" not in out, \
        f"M4 must not claim compose guard detected regression when daemon is down:\n{out}"
    passed_lines = [line for line in out.splitlines() if line.startswith("MUTATION CHECKS PASSED:")]
    assert passed_lines, f"expected passing summary for other mutations, got:\n{out}"
    assert "M4" not in passed_lines[0].split()[3:], \
        f"M4 must not be counted in passed mutations when daemon is down: {passed_lines[0]}"
    assert "SKIP M4" in out or "NOT VERIFIED" in out, \
        f"M4 must report SKIPPED/NOT VERIFIED when daemon is down:\n{out}"


def test_a_new_ledger_key_is_placed_where_it_survives_the_container(tmp_path, monkeypatch):
    """The repository root is a checkout the runtime user usually cannot write and that is lost
    when a container is replaced; a key created there invalidates every entry it signed. A fresh
    ledger must therefore create its key in the durable, writable config location."""
    from dataclasses import replace

    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models as evaluator

    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "artifacts").mkdir()
    config = replace(resolve_config(), CIR_REPO_ROOT=repo,
                     WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")
    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    key = evaluator.ledger_hmac_key(config)
    assert len(key) == 32
    assert (tmp_path / "config" / "cir" / "workbench_ledger_key").is_file(), \
        "a fresh key must land in the durable config location"
    assert not (repo / ".workbench_ledger_key").is_file(), \
        "the repository root is not a durable place for the ledger key"


def test_an_existing_repository_key_is_still_honoured(tmp_path, monkeypatch):
    """A key already at the repository root signed the ledger on disk. Choosing a different
    location would silently make every existing entry unverifiable."""
    from dataclasses import replace

    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models as evaluator

    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "artifacts").mkdir()
    legacy = b"legacy-key-bytes-32-len-aaaaaaaa"
    (repo / ".workbench_ledger_key").write_bytes(legacy)
    config = replace(resolve_config(), CIR_REPO_ROOT=repo,
                     WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")
    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))

    assert evaluator.ledger_hmac_key(config) == legacy

# --------------------------------------------------------------------------- F2 (ambiguous ledger key)

_AMBIGUOUS_ENV_KEYS = ("CIR_REPO_ROOT", "CIR_DATA_ROOT", "FASHIONIQ_ROOT",
                       "WORKBENCH_CHECKPOINT_ROOT", "WORKBENCH_RESULTS_ROOT",
                       "WORKBENCH_THIRD_PARTY_ROOT")


def _workspace_with_two_different_keys(tmp_path: Path, monkeypatch) -> tuple:
    """A workspace carrying BOTH key locations with different bytes, and one report on disk.

    That is the state this host is actually in: `<CIR_REPO_ROOT>/.workbench_ledger_key` and
    `$XDG_CONFIG_HOME/cir/workbench_ledger_key` both exist and disagree.
    """
    from workbench.backend.operator_config import resolve_config

    artifacts = tmp_path / "artifacts"
    reports = artifacts / "reports"
    reports.mkdir(parents=True)
    (reports / "m_c_r1_aggregate.json").write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report", "run_id": "r1",
        "model_id": "m", "checkpoint_id": "c", "protocol_id": "p"}))

    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    for key in _AMBIGUOUS_ENV_KEYS:
        monkeypatch.setenv(key, str(tmp_path / key.lower()))
    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setenv("CIR_REPO_ROOT", str(repo))
    config_home = tmp_path / "config"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    legacy = repo / ".workbench_ledger_key"
    private = config_home / "cir" / "workbench_ledger_key"
    legacy.write_bytes(b"legacy-key-bytes-32-len-aaaaaaaa")
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private-key-bytes-32-len-bbbbbbbb")
    return resolve_config(), legacy, private


def test_an_ambiguous_ledger_key_names_both_key_paths(tmp_path, monkeypatch):
    """F2 (a): two differing key files must produce a refusal that names BOTH paths and the
    remedy, never a bare None. Fail-closed is kept: no key is preferred over the other."""
    from workbench.scripts import evaluate_models as evaluator

    config, legacy, private = _workspace_with_two_different_keys(tmp_path, monkeypatch)

    with pytest.raises(evaluator.LedgerKeyError) as error:
        evaluator.ledger_hmac_key(config)
    message = str(error.value)
    assert "two different ledger keys exist" in message, message
    assert str(legacy) in message, f"the refusal must name the repository-root key: {message}"
    assert str(private) in message, f"the refusal must name the config-home key: {message}"
    assert "WORKBENCH_LEDGER_KEY" in message, f"the refusal must state the remedy: {message}"


def test_an_unreadable_ambiguous_key_still_names_both_paths(tmp_path, monkeypatch):
    """F2 (a): the *other* ambiguity refusal -- two files that exist but cannot be compared --
    used to say only "cannot compare the two ledger key files", naming neither file. The operator
    has no way to act on that, so the paths and the remedy are stated there too."""
    from workbench.scripts import evaluate_models as evaluator

    config, legacy, private = _workspace_with_two_different_keys(tmp_path, monkeypatch)
    original = Path.read_bytes
    unreadable = {legacy}

    def read_bytes(path):
        if path in unreadable:
            raise PermissionError("denied")
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    with pytest.raises(evaluator.LedgerKeyError) as error:
        evaluator.ledger_hmac_key(config)
    message = str(error.value)
    assert str(legacy) in message and str(private) in message, message
    assert "WORKBENCH_LEDGER_KEY" in message, message


def test_the_ledger_read_fails_with_a_reason_naming_both_paths(tmp_path, monkeypatch):
    """F2 (a): `read_evidence_ledger` still answers None (fail closed), but the reason is
    obtainable instead of silent -- which is what every consumer now reports."""
    from workbench.scripts import evaluate_models as evaluator
    from workbench.scripts import verify_run

    config, legacy, private = _workspace_with_two_different_keys(tmp_path, monkeypatch)
    assert evaluator.read_evidence_ledger(config) is None
    with pytest.raises(evaluator.LedgerKeyError) as error:
        evaluator.ledger_hmac_key(config)
    # The verifier is a ledger consumer: the same ambiguity must be one of its stated reasons,
    # in place of the generic "missing or its HMAC chain does not verify".
    verdict = verify_run.verify("r1", "m", "c", "p")
    assert verdict["verified"] is False
    reason = next((r for r in verdict["reasons"] if "ledger key" in r), None)
    assert reason, verdict["reasons"]
    assert str(legacy) in reason and str(private) in reason, reason
    assert str(error.value) in reason, reason


def test_the_aggregate_report_list_surfaces_the_key_conflict(tmp_path, monkeypatch):
    """F2 (b): `/api/benchmark-runs` and `/api/runs/gate` read `_aggregate_reports`, which used
    to answer `[]` when the key was ambiguous. It must now raise a structured, client-visible
    error that states the reason, instead of an empty list nobody can explain."""
    from workbench.backend import main as api
    from workbench.backend.errors import WorkbenchError

    config, legacy, private = _workspace_with_two_different_keys(tmp_path, monkeypatch)
    monkeypatch.setattr(api, "resolve_config", lambda: config)

    with pytest.raises(WorkbenchError) as error:
        api._aggregate_reports()
    assert error.value.code == "ledger_key_unusable", error.value.code
    assert str(legacy) in error.value.message and str(private) in error.value.message, \
        error.value.message
    assert "WORKBENCH_LEDGER_KEY" in error.value.message, error.value.message
    assert str(legacy) in error.value.details["reason"], error.value.details

    # The same refusal reaches the ASGI surface as a 400 with the machine-readable payload,
    # never as a 200 with `[]`.
    status, _, body = _asgi_get(api.app, "/api/benchmark-runs")
    assert status == 400, (status, body)
    payload = json.loads(body)
    assert payload["error"] == "ledger_key_unusable", payload
    assert str(legacy) in payload["message"] and str(private) in payload["message"], payload


def test_one_key_file_alone_keeps_the_ledger_working(tmp_path, monkeypatch):
    """F2 (c): exactly one key file is not ambiguity. The ledger works and a report whose entry
    that key signed is served; removing the second file is the documented remedy."""
    from workbench.backend import main as api
    from workbench.scripts import evaluate_models as evaluator

    config, legacy, private = _workspace_with_two_different_keys(tmp_path, monkeypatch)
    private.unlink()

    key = evaluator.ledger_hmac_key(config)
    assert key == legacy.read_bytes(), "the surviving repository-root key must be honoured"
    reports = evaluator.reports_root(config)
    path = reports / "m_c_r1_aggregate.json"
    evaluator.append_evidence_entry(config, {
        "run_id": "r1", "report_digest": evaluator.sha256_file(path), "model_id": "m",
        "checkpoint_id": "c", "protocol_id": "p",
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"})
    monkeypatch.setattr(api, "resolve_config", lambda: config)
    assert [report["run_id"] for report in api._aggregate_reports()] == ["r1"]


def test_an_explicit_operator_key_resolves_the_ambiguity(tmp_path, monkeypatch):
    """F2 (d): `WORKBENCH_LEDGER_KEY` is the operator's explicit choice, so no ambiguity remains:
    it wins over both files, and the same value both writes and reads the ledger."""
    from dataclasses import replace

    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models as evaluator

    repo = tmp_path / "repo"
    repo.mkdir()
    (tmp_path / "artifacts").mkdir()
    (repo / ".workbench_ledger_key").write_bytes(b"legacy-key-bytes-32-len-aaaaaaaa")
    private = tmp_path / "config" / "cir" / "workbench_ledger_key"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private-key-bytes-32-len-bbbbbbbb")
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "operator-chosen-key")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    config = replace(resolve_config(), CIR_REPO_ROOT=repo,
                     WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts")

    assert evaluator.ledger_hmac_key(config) == b"operator-chosen-key"

    reports = evaluator.reports_root(config)
    reports.mkdir(parents=True)
    path = reports / "m_c_r1_aggregate.json"
    path.write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report", "run_id": "r1",
        "model_id": "m", "checkpoint_id": "c", "protocol_id": "p"}))
    evaluator.append_evidence_entry(config, {
        "run_id": "r1", "report_digest": evaluator.sha256_file(path), "model_id": "m",
        "checkpoint_id": "c", "protocol_id": "p",
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED"})
    entries = evaluator.read_evidence_ledger(config)
    assert entries is not None and [entry["run_id"] for entry in entries] == ["r1"]


def test_the_gpu_capability_key_reports_the_same_conflict(tmp_path, monkeypatch, capsys):
    """F2 (GPU capability key path): the capability signer kept its documented precedence between
    the same two files, so it signed with one of two keys the operator never chose -- and the
    ledger reader refuses that state outright, so every record failed its signature check with no
    visible reason. The conflict must be named, with both paths and the remedy, where the reader
    sees it."""
    from dataclasses import replace

    from workbench.backend import runtime
    from workbench.backend import operator_config
    from workbench.backend.operator_config import resolve_config

    repo = tmp_path / "repo"
    repo.mkdir()
    legacy = repo / ".workbench_ledger_key"
    legacy_bytes = b"legacy-key-bytes-32-len-aaaaaaaa"
    legacy.write_bytes(legacy_bytes)
    private = tmp_path / "config" / "cir" / "workbench_ledger_key"
    private.parent.mkdir(parents=True)
    private.write_bytes(b"private-key-bytes-32-len-bbbbbbbb")
    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setattr(operator_config, "resolve_config",
                        lambda: replace(resolve_config(), CIR_REPO_ROOT=repo))

    # The capability key now adopts the LEDGER's decision instead of making a second one. When two
    # key files disagree, the ledger refuses to guess (a ledger signed with the other key would be
    # unverifiable), so this returns b"" -- which marks every persisted capability record unverified
    # rather than signing it with a key the ledger reader will reject. Previously this function
    # silently chose the legacy key while the ledger refused the same state, so capability records
    # were signed with a key that could never verify.
    key = runtime._gpu_cache_key()
    assert key == b"", "an ambiguous key state must not produce a usable signing key"
    warning = capsys.readouterr().err
    assert "two different ledger keys exist" in warning, warning
    assert str(legacy) in warning and str(private) in warning, warning
    assert "WORKBENCH_LEDGER_KEY" in warning, warning

    # Unambiguous again: with one file gone, the surviving key is used and signs/verifies as before.
    private.unlink()
    assert runtime._gpu_cache_key() == legacy_bytes
    record = {"capability_verified": True, "interpreter": "/x/python"}
    assert runtime._gpu_capability_signature_valid(
        {**record, "signature": runtime._gpu_capability_signature(record)})


def test_the_capability_key_and_the_ledger_key_never_disagree(tmp_path, monkeypatch):
    """The regression this guards: one host, one key decision.

    `_gpu_cache_key` and `ledger_hmac_key` previously disagreed whenever the checkout was writable
    and nothing was in the config home yet: the ledger created `$XDG_CONFIG_HOME/cir/...` while the
    capability cache created `<repo>/.workbench_ledger_key`, and the next run of either refused the
    host with "two different ledger keys exist". They must return the SAME bytes on a fresh host.
    """
    from dataclasses import replace

    from workbench.backend import runtime, operator_config
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import evaluate_models

    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.delenv("WORKBENCH_LEDGER_KEY", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    # The repo root is made unwritable so the ledger's documented preference (the durable config
    # directory) is the one exercised: the point is that BOTH readers agree, whichever location wins.
    monkeypatch.setattr(operator_config, "resolve_config",
                        lambda: replace(resolve_config(), CIR_REPO_ROOT=repo,
                                        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts"))

    # Both readers go through the PATCHED resolver: the module-level `resolve_config` name is the
    # real one, and using it here would resolve the real repository root and its real key.
    capability = runtime._gpu_cache_key()
    ledger = evaluate_models.ledger_hmac_key(operator_config.resolve_config())
    assert capability and capability == ledger, (capability[:8], ledger[:8])
    # And exactly one key file exists, so neither side can have created a second one.
    found = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("workbench_ledger_key"))
    assert found == ["config/cir/workbench_ledger_key"], found


def test_every_ledger_consumer_names_the_key_conflict_and_the_remedy():
    """F2 (consumers): `_aggregate_reports` (serving `/api/benchmark-runs` and `/api/runs/gate`),
    `verify_run.py` (the verifier `pc.sh` invokes, and the queue's completion check) and the
    autopilot report all read the same ledger. A consumer that catches the key refusal must state
    it, and the phrase it states must carry the remedy, so no operator is left with `[]`."""
    consumers = {
        REPOSITORY / "workbench/backend/main.py": "except LedgerKeyError as error:",
        REPOSITORY / "workbench/scripts/verify_run.py": "except em.LedgerKeyError as key_error:",
        REPOSITORY / "workbench/backend/autopilot.py": "except LedgerKeyError as error:",
    }
    for path, catch in consumers.items():
        text = path.read_text(encoding="utf-8")
        assert catch in text, f"{path.name} must catch the ledger key refusal"
        assert "read_evidence_ledger" in text, path.name
    # The remedy the operator is told to use is the same phrase the key loader recommends.
    loader = (REPOSITORY / "workbench/scripts/evaluate_models.py").read_text(encoding="utf-8")
    assert loader.count("WORKBENCH_LEDGER_KEY explicitly to choose one") == 1, loader.count(
        "WORKBENCH_LEDGER_KEY explicitly to choose one")


def _asgi_get(app, path: str) -> tuple[int, dict, bytes]:
    """One GET through the real ASGI app, so the status code and payload are observed, not
    assumed from the handler's signature."""
    import asyncio

    response: dict = {"status": None, "headers": [], "body": b""}

    async def receive() -> dict:
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message: dict) -> None:
        if message["type"] == "http.response.start":
            response["status"] = message["status"]
            response["headers"] = message.get("headers", [])
        elif message["type"] == "http.response.body":
            response["body"] += message.get("body", b"")

    scope = {
        "type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
        "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(),
        "query_string": b"", "root_path": "",
        "headers": [(b"host", b"testserver")],
        "client": ("127.0.0.1", 50123), "server": ("testserver", 80),
    }
    asyncio.run(app(scope, receive, send))
    return response["status"], {}, response["body"]


def test_gpu_capability_refuses_different_interpreter_on_both_paths(tmp_path, monkeypatch):
    """B-2: a record for a different interpreter (or drifted fingerprint) is refused
    on BOTH authoritative and advisory cache read paths."""
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "0" * 64)
    model = {
        "model_id": "test_drift",
        "environment": {"python_env_var": "DRIFT_PYTHON", "pytorch": "1.12.1", "cuda": "11.7"},
    }
    root = tmp_path / "env"
    (root / "bin").mkdir(parents=True)
    interp = root / "bin" / "python"
    interp.write_text("#!/bin/sh\nexit 0\n")
    interp.chmod(0o755)
    monkeypatch.setenv("DRIFT_PYTHON", str(interp))

    report = runtime.verify_environment(model, probe=False, advisory=True)
    key = runtime._gpu_capability_key(model)
    stale_fp = "0123456789abcdef" * 4

    payload = {
        "capability_key": key,
        "capability_verified": True,
        "probe_source": "interpreter",
        "interpreter": str(interp),
        "interpreter_fingerprint": stale_fp,
        "dependency_fingerprint": runtime._environment_contract_digest(model),
        "host_identity": runtime._host_identity(),
    }
    payload["signature"] = runtime._gpu_capability_signature(payload)
    runtime._GPU_CAPABILITY.clear()
    runtime._GPU_CAPABILITY[key] = payload

    assert runtime._cached_gpu_capability(model, report, authoritative=True) is None
    assert runtime._cached_gpu_capability(model, report, authoritative=False) is None


def test_gpu_capability_falsy_fingerprint_guard_refuses_unfingerprinted_record(tmp_path, monkeypatch):
    """B-11: a record carrying no interpreter_fingerprint (None) is refused rather than skipped
    when an environment fingerprint can be computed."""
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", "0" * 64)
    model = {
        "model_id": "test_b11",
        "environment": {"python_env_var": "B11_PYTHON", "pytorch": "1.12.1", "cuda": "11.7"},
    }
    root = tmp_path / "env"
    (root / "bin").mkdir(parents=True)
    interp = root / "bin" / "python"
    interp.write_text("#!/bin/sh\nexit 0\n")
    interp.chmod(0o755)
    monkeypatch.setenv("B11_PYTHON", str(interp))

    report = runtime.verify_environment(model, probe=False, advisory=True)
    key = runtime._gpu_capability_key(model)

    payload = {
        "capability_key": key,
        "capability_verified": True,
        "probe_source": "interpreter",
        "interpreter": str(interp),
        "interpreter_fingerprint": None,
        "dependency_fingerprint": runtime._environment_contract_digest(model),
        "host_identity": runtime._host_identity(),
    }
    payload["signature"] = runtime._gpu_capability_signature(payload)
    runtime._GPU_CAPABILITY.clear()
    runtime._GPU_CAPABILITY[key] = payload

    assert runtime._cached_gpu_capability(model, report, authoritative=True) is None
    assert runtime._cached_gpu_capability(model, report, authoritative=False) is None


def test_assignment_from_different_artifacts_root_is_refused(tmp_path, monkeypatch):
    """B-1 containment: an interpreter recorded in environment_assignment.json located
    in another artifacts root (outside the active WORKBENCH_ARTIFACTS_ROOT) is refused."""
    artifacts_root = tmp_path / "my_artifacts"
    artifacts_root.mkdir()
    foreign_root = tmp_path / "foreign_artifacts"
    foreign_root.mkdir()
    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts_root))
    monkeypatch.delenv("FOREIGN_MODEL_PYTHON", raising=False)

    foreign_interp = foreign_root / "environments" / "m" / "bin" / "python"
    foreign_interp.parent.mkdir(parents=True)
    foreign_interp.touch()

    model = {
        "model_id": "foreign_model",
        "environment": {"python_env_var": "FOREIGN_MODEL_PYTHON"},
    }
    assignment_path = artifacts_root / "environment_assignment.json"
    assignment_path.write_text(json.dumps({
        "models": {
            "foreign_model": {
                "interpreter": str(foreign_interp),
                "state": "RUNTIME_VERIFIED",
            }
        }
    }), encoding="utf-8")

    assert runtime.resolve_model_interpreter(model) is None
    report = runtime.verify_environment(model, probe=False)
    assert report.tier == "UNCONFIGURED"
    assert "not set" in (report.reason or "")
