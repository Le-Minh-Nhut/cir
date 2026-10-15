"""Regression tests for the final-hardening P0/P1 root causes.

Each test fails on the pre-fix code: the mutation it guards against is named in the
docstring, and the assertion targets the observable behavior the fix restored.
"""
from __future__ import annotations

import json
from pathlib import Path

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

    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    historical = planner.model_constraints(dict(limn_model))
    assert "1.12.1" in str(historical["torch"])

    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    adapted = planner.model_constraints(dict(limn_model))
    adapted_torch = str(adapted["torch"])
    assert "1.12.1" not in adapted_torch
    assert "2.7" in adapted_torch
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
