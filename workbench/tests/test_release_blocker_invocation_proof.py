"""P0-2 regression: an invocation with no valid completion proof is never a `completed` job.

Two guarantees are pinned here, both against the real functions rather than a stub of them:

* ``evaluate_models.execute()`` records a machine-readable *invocation verdict* beside the run —
  `PROOF_VALIDATED` only when the evaluator's own ``validate_run_manifest`` accepted the proof
  minted for this invocation, `NO_OBSERVABLE_OUTPUT` / `PROOF_REJECTED` / `EXECUTION_FAILED`
  otherwise — while still returning the **native** child exit code as evidence.
* ``SingleGpuQueue`` consults that verdict: a job whose invocation produced no observable output,
  or whose completion proof was rejected, settles `failed` with a named reason and never
  `completed`.

``workbench/scripts/verify_run.py`` is the standalone consumer of the same evidence; it exits 0
only for a fully corroborated run and names the missing or forged ingredient otherwise.
"""
from __future__ import annotations

import json
import secrets
import sys
import time
import uuid
from pathlib import Path

import pytest
import yaml

from workbench.backend.jobs import SingleGpuQueue
from workbench.backend.operator_config import WorkbenchConfig
import workbench.scripts.evaluate_models as em
import workbench.scripts.verify_run as verify_run

PIN = "a" * 40
MODEL = "limn"
CHECKPOINT = "ckpt_v1"
PROTOCOL = "fashioniq_original_split"

#: A LIMN-shaped aggregate: the three categories the macro is defined over, with the declared
#: macro recomputable from them (the parser rejects an inconsistent one).
LIMN_CATEGORIES = [
    {"category": "dress", "r1": 10.0, "r10": 20.0, "r50": 45.0},
    {"category": "shirt", "r1": 11.0, "r10": 20.0, "r50": 45.0},
    {"category": "toptee", "r1": 12.0, "r10": 20.0, "r50": 45.0},
]
LIMN_AGGREGATE = {"macro_r10": 20.0, "macro_r50": 45.0, "macro_mean": 32.5, "complete": True}

def reporting_evaluator(tmp_path: Path, *, exit_code: int = 0) -> list[str]:
    """A synthetic official evaluator printing a LIMN-shaped aggregate, then exiting."""
    script = tmp_path / f"evaluator_{uuid.uuid4().hex[:8]}.py"
    script.write_text(
        "import json, sys\n"
        'print("evaluating")\n'
        f"print(json.dumps({{'model_id': 'limn', 'aggregate': {LIMN_AGGREGATE!r},"
        f" 'categories': {LIMN_CATEGORIES!r}}}))\n"
        f"sys.exit({exit_code})\n", encoding="utf-8")
    return [sys.executable, str(script)]

def silent_evaluator(tmp_path: Path) -> list[str]:
    """Exits 0 and prints nothing at all."""
    script = tmp_path / f"silent_{uuid.uuid4().hex[:8]}.py"
    script.write_text("import sys\nsys.exit(0)\n", encoding="utf-8")
    return [sys.executable, str(script)]

def registry_document() -> dict:
    """The one model ``verify_run.py`` must be able to resolve back to its checkpoint."""
    return {
        "schema_version": 1,
        "protocols": {PROTOCOL: {"label": "Original Split", "literature_split_label": "original",
                                 "evaluation_noise_pct": 0}},
        "models": [{
            "model_id": MODEL, "method_name": "LIMN", "research_generation": "recent",
            "publication_year": 2024, "preparation_contract": "manual_standard",
            "upstream_repo_url": "https://example.invalid/limn.git",
            "upstream_default_branch": "main", "upstream_commit_sha": PIN,
            "source_available": True, "source_dir": "LIMN",
            "fashioniq_layout": "fashioniq_standard", "literature_split_label": "original",
            "native_protocol": PROTOCOL, "supported_protocols": [PROTOCOL],
            "reported_r10": 20.0, "reported_r50": 45.0, "reported_mean": 32.5,
            "reproduction_status": "NOT_RUN", "command_status": "REPLAY_COMMAND_IMPLEMENTED",
            "checkpoint_variants": [{
                "checkpoint_id": CHECKPOINT, "filename": "model.pt",
                "download_url": "https://example.invalid/model.pt",
                "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0,
                "expected_sha256": None, "status": "URL_VERIFIED",
                "checkpoint_mapping_status": "VERIFIED_METADATA",
            }],
        }],
    }

@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    """A real artifact root, one real checkpoint, one registry and one ledger key."""
    repo = tmp_path / "repo"
    source = tmp_path / "source"
    for directory in (repo, source):
        directory.mkdir()
    checkpoints = tmp_path / "checkpoints" / "limn"
    checkpoints.mkdir(parents=True)
    checkpoint = checkpoints / "model.pt"
    checkpoint.write_bytes(b"limn-weights-v1")
    (tmp_path / "FashionIQ").mkdir()

    registry_path = tmp_path / "models.yaml"
    registry_path.write_text(yaml.safe_dump(registry_document()), encoding="utf-8")

    artifacts = repo / "workbench" / "artifacts"
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", secrets.token_hex(32))
    monkeypatch.setenv("WORKBENCH_REGISTRY_PATH", str(registry_path))
    # `verify_run.py` resolves its own config, so the same paths must be visible to it.
    for key, value in (("CIR_REPO_ROOT", repo), ("WORKBENCH_ARTIFACTS_ROOT", artifacts),
                       ("WORKBENCH_CHECKPOINT_ROOT", tmp_path / "checkpoints"),
                       ("WORKBENCH_RESULTS_ROOT", tmp_path / "results"),
                       ("WORKBENCH_THIRD_PARTY_ROOT", tmp_path / "third_party"),
                       ("FASHIONIQ_ROOT", tmp_path / "FashionIQ"),
                       ("CIR_DATA_ROOT", tmp_path / "data")):
        monkeypatch.setenv(key, str(value))

    config = WorkbenchConfig(
        CIR_REPO_ROOT=repo, CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "FashionIQ", WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
        WORKBENCH_ARTIFACTS_ROOT=artifacts)
    return {"repo": repo, "source": source, "checkpoint": checkpoint, "config": config,
            "artifacts": artifacts, "tmp": tmp_path}

def plan_for(env: dict, command: list[str], *, mutated_member: bool = False) -> em.EvaluationPlan:
    """The plan the operator is asked to execute.

    ``mutated_member`` builds a multi-member checkpoint whose *recorded* manifest no longer
    matches the live bytes: exactly the case of a declared checkpoint member that changed after
    the manifest was taken, which the evaluator's own validator must refuse.
    """
    members = None
    bundle = None
    manifest_digest = None
    artifact_type = "single_file"
    if mutated_member:
        artifact_type = "category_specific_whole_models"
        members = [{"filename": "model.pt", "path": str(env["checkpoint"]), "sha256": "0" * 64}]
        bundle = {member["filename"]: member for member in members}
        manifest_digest = em.manifest_digest(bundle)
    return em.EvaluationPlan(
        model_id=MODEL, checkpoint_id=CHECKPOINT, protocol=PROTOCOL,
        dataset_root=env["tmp"] / "FashionIQ", source=env["source"],
        checkpoint=env["checkpoint"], cwd=env["source"], command=command,
        pin=PIN, actual_source_commit=PIN, paper_metrics=None,
        artifact_type=artifact_type, directory_members=members, checkpoint_bundle=bundle,
        bundle_manifest_digest=manifest_digest)

def run_operator(env: dict, plan: em.EvaluationPlan) -> tuple[int, str]:
    """Run the real ``execute()`` and return ``(native exit code, run id)``."""
    run_id = f"p02{uuid.uuid4().hex[:10]}"
    return em.execute(plan, env["config"], run_id=run_id), run_id

def outcome_of(env: dict, plan: em.EvaluationPlan, run_id: str) -> dict:
    directory = em.log_dir(env["config"], plan, em.datetime.now(em.timezone.utc), run_id)
    return json.loads((directory / "invocation_outcome.json").read_text(encoding="utf-8"))

def settle(queue: SingleGpuQueue, job_id: str, *, timeout: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = next(item for item in queue.list() if item["job_id"] == job_id)
        if job["status"] in {"completed", "failed", "blocked", "cancelled"}:
            return job
        time.sleep(0.02)
    raise AssertionError(f"job never settled: {queue.list()}")

def queue_for(env: dict, run_id: str, plan: em.EvaluationPlan) -> dict:
    """Dispatch a real job whose provenance names the run the operator just recorded.

    The command is a no-op that exits 0: the queue must decide from the operator's recorded
    verdict, never from an exit code that cannot distinguish an output-less run from a real one.
    """
    queue = SingleGpuQueue()
    job = queue.enqueue(
        MODEL, CHECKPOINT, PROTOCOL, [sys.executable, "-c", "pass"], env["repo"],
        env["repo"] / "queue_jobs" / uuid.uuid4().hex,
        {"model_id": MODEL, "checkpoint_id": CHECKPOINT, "protocol_id": PROTOCOL,
         "run_id": run_id,
         "run_directory": str(em.log_dir(env["config"], plan, em.datetime.now(em.timezone.utc),
                                         run_id))})
    settled = settle(queue, job["job_id"])
    queue.shutdown()
    return settled

def test_a_dispatch_with_no_recorded_verdict_never_completes(env):
    """A run that left no verdict: `failed`, because silence is not a completion proof.

    This is the shape of a job whose invocation produced no observable output: the operator
    records `NO_OBSERVABLE_OUTPUT`, and a caller that cannot find *any* verdict has proved
    nothing either, so it must not publish success.
    """
    queue = SingleGpuQueue()
    job = queue.enqueue(
        MODEL, CHECKPOINT, PROTOCOL, [sys.executable, "-c", "pass"], env["repo"],
        env["repo"] / "queue_jobs" / uuid.uuid4().hex,
        {"model_id": MODEL, "checkpoint_id": CHECKPOINT, "protocol_id": PROTOCOL,
         "run_id": "no-verdict-run",
         "run_directory": str(env["tmp"] / "logs" / "limn__ckpt_v1__no-verdict-run")})
    settled = settle(queue, job["job_id"])
    queue.shutdown()
    assert settled["status"] == "failed", settled
    assert settled["return_code"] == 0, "the child's own exit code is still recorded"
    assert "no valid completion proof" in (settled["error"] or "")

# ------------------------------------------------------------------ execute() outcome verdicts

def test_exit_zero_without_observable_output_never_completes(env):
    """(a) Exit 0 with no observable output: the job is never `completed`."""
    plan = plan_for(env, silent_evaluator(env["tmp"]))
    code, run_id = run_operator(env, plan)
    assert code == 0, "the native exit code is evidence and must not be overwritten"

    outcome = outcome_of(env, plan, run_id)
    assert outcome["verdict"] == "NO_OBSERVABLE_OUTPUT"
    assert outcome["completed"] is False
    assert outcome["return_code"] == 0, "the native code is preserved in the verdict too"
    assert not em.completion_proof_path(env["config"], plan, run_id).exists()

    settled = queue_for(env, run_id, plan)
    assert settled["status"] == "failed", settled
    assert settled["return_code"] == 0, "the original child exit code is preserved as evidence"
    assert "no observable evaluation output" in (settled["error"] or "")
    assert settled["error"] in settled["stderr_tail"]

def test_an_invalid_completion_proof_never_completes(env):
    """(b) A proof that fails the evaluator's own validation: the job is never `completed`."""
    plan = plan_for(env, reporting_evaluator(env["tmp"]), mutated_member=True)
    code, run_id = run_operator(env, plan)
    assert code == 0

    outcome = outcome_of(env, plan, run_id)
    assert outcome["verdict"] == "PROOF_REJECTED", outcome
    assert outcome["completed"] is False
    assert "manifest" in (outcome["reason"] or ""), outcome["reason"]
    assert not em.latest_pointer(env["config"], plan, "successful").exists(), \
        "a rejected proof must never advance latest_successful"

    settled = queue_for(env, run_id, plan)
    assert settled["status"] == "failed", settled
    # The queue names why, from the operator's own recorded reason.
    assert "checkpoint members no longer match the recorded artifact manifest" in \
        (settled["error"] or ""), settled

def test_a_proof_accepted_by_the_operator_completes(env):
    """(c) A validated proof: the job is `completed`."""
    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    code, run_id = run_operator(env, plan)
    assert code == 0

    outcome = outcome_of(env, plan, run_id)
    assert outcome["verdict"] == "PROOF_VALIDATED", outcome
    assert outcome["completed"] is True

    settled = queue_for(env, run_id, plan)
    assert settled["status"] == "completed", settled
    assert settled["error"] is None

def test_a_nonzero_exit_records_the_native_code_as_evidence(env):
    """(d) A failed evaluator: the job fails and its native code survives."""
    plan = plan_for(env, reporting_evaluator(env["tmp"], exit_code=7))
    code, run_id = run_operator(env, plan)
    assert code == 7
    outcome = outcome_of(env, plan, run_id)
    assert outcome["verdict"] == "EXECUTION_FAILED"
    assert outcome["return_code"] == 7

    settled = queue_for(env, run_id, plan)
    assert settled["status"] == "failed"
    # The queue reports the exit code of *its own* child; the evaluator's native code survives
    # as evidence in the run's metadata, report, ledger entry and invocation verdict, which is
    # where `test_out02b` and `test_exit02` read it.
    assert settled["error"]
    assert outcome["return_code"] == 7

# ------------------------------------------------------------------------ verify_run.py

def verify(env, run_id: str) -> tuple[int, dict]:
    """Call the real CLI and return ``(exit code, printed JSON)``."""
    from io import StringIO
    import contextlib

    stream = StringIO()
    with contextlib.redirect_stdout(stream):
        code = verify_run.main(["--run-id", run_id, "--model", MODEL,
                                "--checkpoint", CHECKPOINT, "--protocol", PROTOCOL, "--json"])
    return code, json.loads(stream.getvalue())

def test_verify_run_accepts_a_fully_corroborated_run(env):
    """(e) Positive: every ingredient agrees, so the CLI exits 0 with the metrics and the gate."""
    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    code, run_id = run_operator(env, plan)
    assert code == 0

    exit_code, verdict = verify(env, run_id)
    assert exit_code == 0, verdict["reasons"]
    assert verdict["verified"] is True and verdict["reasons"] == []
    assert verdict["report_path"] == str(em.run_report_path(env["config"], plan, run_id))
    # The report's own identity, including the commit the run came from, after the requested
    # identity was checked against it field by field.
    assert verdict["identity"] == {"run_id": run_id, "model_id": MODEL,
                                  "checkpoint_id": CHECKPOINT, "protocol_id": PROTOCOL,
                                  "git_commit": PIN, "runtime_adaptation": None}
    for category in ("dress", "shirt", "toptee"):
        assert verdict["metrics"]["category"][category]["r10"] == 20.0
        assert verdict["metrics"]["category"][category]["r50"] == 45.0
    assert "gate" in verdict

def test_verify_run_reports_a_missing_report(env):
    exit_code, verdict = verify(env, "never-ran")
    assert exit_code != 0
    assert any("aggregate report not found" in reason for reason in verdict["reasons"]), verdict
    assert verdict["report_path"] is None

def test_verify_run_reports_a_missing_ledger_entry(env):
    """A run with no ledger entry is not corroborated, whatever its artifacts say."""
    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    run_operator(env, plan)
    em.evidence_ledger_path(env["config"]).unlink()

    exit_code, verdict = verify(env, "any-run")
    assert exit_code != 0
    assert any("ledger is missing or its HMAC chain does not verify" in reason
               for reason in verdict["reasons"]), verdict

def test_verify_run_reports_a_report_digest_mismatch(env):
    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    _, run_id = run_operator(env, plan)
    report = em.run_report_path(env["config"], plan, run_id)
    report.write_text(report.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    exit_code, verdict = verify(env, run_id)
    assert exit_code != 0
    assert any("report digest does not match the report file" in reason
               for reason in verdict["reasons"]), verdict

def test_verify_run_reports_a_nonzero_recorded_return_code(env):
    plan = plan_for(env, reporting_evaluator(env["tmp"], exit_code=9))
    code, run_id = run_operator(env, plan)
    assert code == 9

    exit_code, verdict = verify(env, run_id)
    assert exit_code != 0
    assert any("records return code 9" in reason for reason in verdict["reasons"]), verdict

def test_verify_run_reports_a_wrong_invocation_status(env):
    plan = plan_for(env, silent_evaluator(env["tmp"]))
    run_operator(env, plan)
    run_id = json.loads((em.evidence_ledger_path(env["config"]).read_text()
                         .strip().splitlines()[-1]))["run_id"]

    exit_code, verdict = verify(env, run_id)
    assert exit_code != 0
    assert any("expected 'INVOCATION_COMPLETED'" in reason for reason in verdict["reasons"]), verdict

def test_verify_run_reports_a_missing_completion_proof(env):
    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    _, run_id = run_operator(env, plan)
    em.completion_proof_path(env["config"], plan, run_id).unlink()

    exit_code, verdict = verify(env, run_id)
    assert exit_code != 0
    assert any("completion proof rejected" in reason for reason in verdict["reasons"]), verdict

def test_verify_run_reports_a_missing_category(env):
    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    _, run_id = run_operator(env, plan)
    report_path = em.run_report_path(env["config"], plan, run_id)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    del report["category_metrics"]["shirt"]
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    exit_code, verdict = verify(env, run_id)
    assert exit_code != 0
    assert any("category shirt is missing from the report" in reason
               for reason in verdict["reasons"]), verdict

def test_verify_run_reports_an_invalid_recall(env):
    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    _, run_id = run_operator(env, plan)
    report_path = em.run_report_path(env["config"], plan, run_id)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["category_metrics"]["toptee"]["r50"] = 250.0
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    exit_code, verdict = verify(env, run_id)
    assert exit_code != 0
    assert any("category toptee r50 is not a finite Recall" in reason
               for reason in verdict["reasons"]), verdict


def test_a_forged_verdict_file_cannot_publish_a_completion(tmp_path, monkeypatch):
    """The verdict file lives in the run directory the operator was dispatched with, so the
    untrusted evaluator child can write it. A claimed PROOF_VALIDATED must therefore be
    re-derived from the report, the ledger and the completion proof — not believed."""
    import json

    from workbench.backend.jobs import Job, settled_status

    run_dir = tmp_path / "logs" / "limn__base_iter0_all_categories__FORGED"
    run_dir.mkdir(parents=True)
    (run_dir / "invocation_outcome.json").write_text(json.dumps(
        {"run_id": "FORGED", "verdict": "PROOF_VALIDATED", "completed": True}))

    job = Job("forged", "limn", "base_iter0_all_categories", "fashioniq_val_split", [], "",
              str(run_dir), {"model_id": "limn", "checkpoint_id": "base_iter0_all_categories",
                             "protocol_id": "fashioniq_val_split", "run_id": "FORGED",
                             "run_directory": str(run_dir)})
    status, reason = settled_status(job, 0, "PROOF_VALIDATED", None)
    assert status == "failed", "a hand-written verdict must never publish a completion"
    assert reason and "does not verify" in reason, reason
