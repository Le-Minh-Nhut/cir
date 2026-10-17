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

A run also has to carry usable *metric* evidence, not merely a parseable report: extraction must
have happened, the LIMN macro must equal the mean of its own three categories (the arithmetic
``workbench/replay/common.py::macro_aggregate`` defines), a synthetic self-test may not present
itself as a real result, and a model whose committed reference publishes metrics must carry the
comparison -- or be refused. Those contracts are pinned below against a real ``execute()`` run.
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

#: A published CSMCIR row (``workbench/registry/paper_benchmarks/papers.yaml``,
#: ``csmcir_fiq_val_ours_vitg``): the model whose protocol publishes metrics and which a report
#: without them therefore cannot be certified for.
CSMCIR_PIN = "774f94e2076ff17ea91703a6239d2a08f0e1a44e"
CSMCIR_CHECKPOINT = "fashioniq"
CSMCIR_PROTOCOL = "fashioniq_original_split"

#: An evaluator output the real ``parse_csmcir_output`` reads: per-category recalls and their
#: macro, all within 5 percent relative error of the published row above.
CSMCIR_GOOD_OUTPUT = {
    "dress_recall_at10": 52.45, "dress_recall_at50": 74.81,
    "shirt_recall_at10": 57.70, "shirt_recall_at50": 75.76,
    "toptee_recall_at10": 61.14, "toptee_recall_at50": 80.98,
    "average_recall_at10": 57.10, "average_recall_at50": 77.18, "average_recall": 67.14,
}

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

def reporting_evaluator(tmp_path: Path, *, exit_code: int = 0,
                       aggregate: dict | None = None,
                       categories: list | None = None) -> list[str]:
    """A synthetic official evaluator printing a LIMN-shaped aggregate, then exiting.

    ``aggregate``/``categories`` let a case emit what the extraction parser will *refuse* -- a
    declared macro that disagrees with the categories printed beside it, or an incomplete run --
    which is the shape of a real evaluator whose own output did not add up.
    """
    script = tmp_path / f"evaluator_{uuid.uuid4().hex[:8]}.py"
    script.write_text(
        "import json, sys\n"
        'print("evaluating")\n'
        f"print(json.dumps({{'model_id': 'limn',"
        f" 'aggregate': {aggregate if aggregate is not None else LIMN_AGGREGATE!r},"
        f" 'categories': {categories if categories is not None else LIMN_CATEGORIES!r}}}))\n"
        f"sys.exit({exit_code})\n", encoding="utf-8")
    return [sys.executable, str(script)]

def evaluator(tmp_path: Path, *lines: str) -> list[str]:
    """An evaluator that prints exactly ``lines`` and exits 0: the run's own raw output."""
    script = tmp_path / f"evaluator_{uuid.uuid4().hex[:8]}.py"
    script.write_text("import json, sys\n" + "".join(f"{line}\n" for line in lines),
                      encoding="utf-8")
    return [sys.executable, str(script)]


def non_limn_registry(model_id: str = "csmcir", *, pin: str = CSMCIR_PIN) -> dict:
    """The registry document ``verify_run.py`` and ``execute()`` must resolve, one real model.

    ``model_id`` is read from the registry rather than special-cased in the verifier, so this
    fixture is the only thing that decides which model the run under test claims to be.
    """
    document = registry_document()
    model = document["models"][0]
    model.update({
        "model_id": model_id, "method_name": model_id.upper(), "source_dir": model_id.upper(),
        "upstream_commit_sha": pin, "native_protocol": CSMCIR_PROTOCOL,
        "supported_protocols": [CSMCIR_PROTOCOL], "literature_split_label": "original",
        "reported_r10": 57.07, "reported_r50": 77.27, "reported_mean": 67.17,
    })
    document["protocols"] = {CSMCIR_PROTOCOL: {"label": "Original Split",
                                               "literature_split_label": "original",
                                               "evaluation_noise_pct": 0}}
    variant = model["checkpoint_variants"][0]
    variant.update({"checkpoint_id": CSMCIR_CHECKPOINT,
                    "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0})
    return document


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

def plan_for(env: dict, command: list[str], *, mutated_member: bool = False,
             model_id: str = MODEL, checkpoint_id: str = CHECKPOINT,
             protocol: str = PROTOCOL, pin: str = PIN) -> em.EvaluationPlan:
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
    # A non-LIMN model's checkpoint lives under its own registry-declared path; `pin` is the
    # commit that model's registry entry declares, because the completion validator compares the
    # proof's source commit against the *registry* pin, not against the plan's.
    checkpoint = (env["checkpoint"] if model_id == MODEL
                  else env["tmp"] / "checkpoints" / model_id / "model.pt")
    return em.EvaluationPlan(
        model_id=model_id, checkpoint_id=checkpoint_id, protocol=protocol,
        dataset_root=env["tmp"] / "FashionIQ", source=env["source"],
        checkpoint=checkpoint, cwd=env["source"], command=command,
        pin=pin, actual_source_commit=pin, paper_metrics=None,
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

def verify_as(env, run_id: str, model: str, checkpoint: str,
              protocol: str) -> tuple[int, dict]:
    """Call the real CLI and return ``(exit code, printed JSON)``."""
    from io import StringIO
    import contextlib

    stream = StringIO()
    with contextlib.redirect_stdout(stream):
        code = verify_run.main(["--run-id", run_id, "--model", model,
                                "--checkpoint", checkpoint, "--protocol", protocol, "--json"])
    return code, json.loads(stream.getvalue())


def verify(env, run_id: str) -> tuple[int, dict]:
    return verify_as(env, run_id, MODEL, CHECKPOINT, PROTOCOL)


def run_and_read_report(env, command: list[str], **plan_kwargs) -> tuple[str, dict]:
    """Execute one real run and return ``(run_id, its aggregate report)``."""
    plan = plan_for(env, command, **plan_kwargs)
    code, run_id = run_operator(env, plan)
    assert code == 0, "the evaluator's own exit code is what the operator returns"
    return run_id, json.loads(em.run_report_path(env["config"], plan, run_id).read_text())


def report_reasons(env, run_id: str, mutations: dict) -> list[str]:
    """Reasons the verifier gives for a run whose report was altered in exactly one way.

    The metrics a run produces are whatever its evaluator printed, so the report's metric fields
    are the honest freedom of the run -- not an attack surface. To test the metric contract in
    isolation, this rewrites the field and then *re-corroborates* the altered report the way the
    operator would have: a fresh ledger witness over the new bytes, and the completion proof
    re-pointed at both. Every other ingredient (the real subprocess log, the real proof, the
    real checkpoint bundle) stays exactly as the run left it, so the reasons returned are the
    metric-evidence ones and not digest noise.
    """
    report_path = next(em.reports_root(env["config"]).glob(f"*_{run_id}_aggregate.json"))
    # The run is already in the ledger; the ledger refuses a duplicate run id (by design), so the
    # stale witness is dropped and a fresh one is recorded for the altered bytes.
    em.evidence_ledger_path(env["config"]).unlink()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(mutations)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    witness = em.append_evidence_entry(env["config"], {
        "run_id": run_id, "model_id": report["model_id"],
        "checkpoint_id": report["checkpoint_id"], "protocol_id": report["protocol_id"],
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED",
        "report_digest": em.sha256_file(report_path)})
    proof_path = report_path.with_name(f"{report['model_id']}_{report['checkpoint_id']}"
                                       f"_{run_id}_completion.json")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["report_digest"] = em.sha256_file(report_path)
    proof["ledger_entry_digest"] = witness["entry_digest"]
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    return verdict_reasons(env, run_id, report["model_id"], report["checkpoint_id"],
                           report["protocol_id"])


def recorroborated_verdict(env, run_id: str, mutations: dict) -> dict:
    """Re-corroborate an altered report and return the verifier's verdict, unasserted.

    Same mutation + fresh ledger witness + re-pointed proof as `report_reasons`, but it does not
    demand a refusal: a caller testing WHICH question refuses a run needs the verdict for cases
    where one of the two answers is legitimately positive.
    """
    report_path = next(em.reports_root(env["config"]).glob(f"*_{run_id}_aggregate.json"))
    em.evidence_ledger_path(env["config"]).unlink()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(mutations)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    witness = em.append_evidence_entry(env["config"], {
        "run_id": run_id, "model_id": report["model_id"],
        "checkpoint_id": report["checkpoint_id"], "protocol_id": report["protocol_id"],
        "return_code": 0, "invocation_status": "INVOCATION_COMPLETED",
        "report_digest": em.sha256_file(report_path)})
    proof_path = report_path.with_name(f"{report['model_id']}_{report['checkpoint_id']}"
                                       f"_{run_id}_completion.json")
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    proof["report_digest"] = em.sha256_file(report_path)
    proof["ledger_entry_digest"] = witness["entry_digest"]
    proof_path.write_text(json.dumps(proof, indent=2) + "\n", encoding="utf-8")
    _, verdict = verify_as(env, run_id, report["model_id"], report["checkpoint_id"],
                           report["protocol_id"])
    return verdict


def verdict_reasons(env, run_id: str, model: str = MODEL, checkpoint: str = CHECKPOINT,
                    protocol: str = PROTOCOL) -> list[str]:
    """Every reason a run is not admissible, from both questions the verifier answers.

    A run can be refused for being incompletely corroborated (`reasons`) or for being a
    corroborated artefact that is not evidence about the paper (`evidence_reasons`, e.g. a
    declared self-test). A caller testing admissibility needs both.
    """
    exit_code, verdict = verify_as(env, run_id, model, checkpoint, protocol)
    assert exit_code != 0, "this case must not certify"
    return list(verdict["reasons"]) + list(verdict.get("evidence_reasons") or [])

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
    # The macro the report states is the mean of the categories it printed, so this run's
    # evidence passes the consistency contract as well as the corroboration one.
    assert verdict["metrics"]["observed"] == {"r10": 20.0, "r50": 45.0, "mean": 32.5}
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


# ---------------------------------------------------- metric evidence (P0-2) and non-LIMN models

def test_verify_run_reports_an_incomplete_extraction(env):
    """A run that exited 0 but whose metrics were never extracted is not a certified result.

    The evaluator printed an incomplete declaration (``complete: false``), so the source-backed
    parser itself refused to extract anything. The categories still look perfect; only the
    report's own ``extraction_status``/``metric_extraction_available`` say no extraction
    happened, which is exactly the state the old verifier certified.
    """
    incomplete = {"macro_r10": 20.0, "macro_r50": 45.0, "macro_mean": 32.5, "complete": False}
    run_id, report = run_and_read_report(
        env, reporting_evaluator(env["tmp"], aggregate=incomplete))
    assert report["extraction_status"] == "EVALUATION_COMPLETED_METRICS_UNPARSED", report
    assert report["metric_extraction_available"] is False
    assert report["observed_metrics"] is None

    reasons = verdict_reasons(env, run_id)
    assert any("no metric extraction was performed" in reason for reason in reasons), reasons


def test_verify_run_reports_a_failed_extraction_status(env):
    """An ``extraction_status`` that names a failure is refused even when metrics are present.

    This is the report an older or differently-built evaluator could leave behind: usable
    numbers in ``observed_metrics`` while its own status field says the extraction did not
    succeed. The status is the report's own claim about itself and it wins.
    """
    run_id, _ = run_and_read_report(env, reporting_evaluator(env["tmp"]))
    reasons = report_reasons(env, run_id, {"extraction_status": "EVALUATION_FAILED"})
    assert any("the run's metrics were not extracted" in reason for reason in reasons), reasons


def test_verify_run_reports_a_macro_inconsistent_with_its_categories(env):
    """The declared macro must equal the mean of the categories the report itself carries.

    The categories are untouched and the corroboration chain is intact; only the aggregate the
    report states is wrong (its R@10 is half the mean of the three categories it printed).
    """
    run_id, _ = run_and_read_report(env, reporting_evaluator(env["tmp"]))
    reasons = report_reasons(env, run_id, {
        "observed_metrics": {"r10": 10.0, "r50": 45.0, "mean": 27.5}})
    assert any("is inconsistent with its categories" in reason for reason in reasons), reasons


def test_verify_run_reports_a_macro_mean_that_is_not_the_mean_of_r10_and_r50(env):
    """``mean = (r10 + r50) / 2`` is part of the macro contract, checked separately.

    Here each macro Recall matches its categories exactly and only the mean is wrong, so a check
    that compared just ``r10``/``r50`` would still certify it.
    """
    run_id, _ = run_and_read_report(env, reporting_evaluator(env["tmp"]))
    reasons = report_reasons(env, run_id, {
        "observed_metrics": {"r10": 20.0, "r50": 45.0, "mean": 40.0}})
    assert any("macro mean" in reason and "is inconsistent" in reason for reason in reasons), \
        reasons


def test_verify_run_refuses_a_synthetic_run_presenting_as_real(env):
    """A report that declares itself synthetic is a pipeline self-test, never evidence.

    The run is otherwise perfect: real subprocess, real ledger witness, real completion proof,
    categories and macro in agreement. Only the run's own declaration disqualifies the numbers.
    """
    run_id, _ = run_and_read_report(env, reporting_evaluator(env["tmp"]))
    verdict = recorroborated_verdict(env, run_id, {"is_synthetic": True})
    assert verdict["verified"] is True, "the execution itself is corroborated"
    assert verdict["is_research_evidence"] is False, "a self-test is not evidence"
    assert any("declares itself synthetic" in reason
               for reason in verdict["evidence_reasons"]), verdict["evidence_reasons"]


def test_verify_run_refuses_a_report_whose_data_kind_is_not_an_experiment(env):
    """``data_kind`` is the vocabulary the gate refuses on; the verifier must refuse it too."""
    run_id, _ = run_and_read_report(env, reporting_evaluator(env["tmp"]))
    verdict = recorroborated_verdict(env, run_id, {"data_kind": "mock"})
    assert verdict["verified"] is True, "the execution itself is corroborated"
    assert verdict["is_research_evidence"] is False, "a mock report is not evidence"
    assert any("not a real experiment" in reason
               for reason in verdict["evidence_reasons"]), verdict["evidence_reasons"]


#: The honest CSMCIR evaluator output and the registry document that names that model, so the
#: non-LIMN contract is exercised through the real registry rather than a special case.
def csmcir_env(env: dict, model_id: str = "csmcir", pin: str = CSMCIR_PIN) -> tuple[str, str]:
    """Point the run's registry at a non-LIMN model whose committed reference publishes metrics.

    Returns the pin that model's registry entry declares and the checkpoint id the run will use:
    ``ckpt_v1`` is this fixture's own checkpoint, registered as the CSMCIR variant so the plan,
    the report, the completion proof and this verifier all name one checkpoint. (The registry's
    committed CSMCIR row is anchored on its own checkpoint id; only the pin and the published
    row are taken from the real registry here.)
    """
    import os

    document = non_limn_registry(model_id, pin=pin)
    document["models"][0]["checkpoint_variants"][0]["checkpoint_id"] = CHECKPOINT
    Path(os.environ["WORKBENCH_REGISTRY_PATH"]).write_text(
        yaml.safe_dump(document), encoding="utf-8")
    checkpoints = env["tmp"] / "checkpoints" / model_id
    checkpoints.mkdir(parents=True, exist_ok=True)
    (checkpoints / "model.pt").write_bytes(f"{model_id}-weights-v1".encode())
    return pin, CHECKPOINT


def test_verify_run_accepts_a_non_limn_model_carrying_its_published_comparison(env):
    """The positive half of the non-LIMN rule: a real comparison against a real reference.

    CSMCIR's committed row publishes R@10/R@50/mean; an evaluator output within the gate's
    5 percent relative rule is the evidence its protocol can produce, and it certifies.
    """
    pin, checkpoint = csmcir_env(env)
    run_id, report = run_and_read_report(env, evaluator(
        env["tmp"], "print(json.dumps(%r))" % CSMCIR_GOOD_OUTPUT),
        model_id="csmcir", checkpoint_id=checkpoint, pin=pin)
    assert report["extraction_status"] == "EXTRACTED_VERIFIED", report
    assert report["observed_metrics"] == {"r10": 57.1, "r50": 77.18, "mean": 67.14}

    exit_code, verdict = verify_as(env, run_id, "csmcir", checkpoint, CSMCIR_PROTOCOL)
    assert exit_code == 0, verdict["reasons"]
    assert verdict["verified"] is True


def test_verify_run_refuses_a_model_whose_reference_publishes_metrics_the_report_lacks(env):
    """A model whose protocol publishes metrics but whose report carries none is refused.

    The evaluator ran, exited 0 and even printed a line -- so this is not the silent-evaluator
    case -- but it printed nothing the source-backed parser can read. Nothing was compared, so
    nothing can be certified.
    """
    pin, checkpoint = csmcir_env(env)
    run_id, report = run_and_read_report(
        env, evaluator(env["tmp"], "print('evaluator finished, no recall lines')"),
        model_id="csmcir", checkpoint_id=checkpoint, pin=pin)
    assert report["observed_metrics"] is None, report

    reasons = verdict_reasons(env, run_id, "csmcir", checkpoint, CSMCIR_PROTOCOL)
    assert any("publishes r10, r50, mean but the report carries none" in reason
               for reason in reasons), reasons


def test_verify_run_refuses_a_non_limn_model_with_no_verified_reference(env):
    """A model with no verified committed row cannot produce a comparison, so it is refused.

    ``pair`` is registered and its reference rows are all UNVERIFIED, which is the honest state
    of that model's evidence: the verifier must say so rather than certify the run anyway.
    """
    model_id, pin = "pair", "b" * 40
    pin, checkpoint = csmcir_env(env, model_id, pin)
    run_id, _ = run_and_read_report(
        env, evaluator(env["tmp"], "print(json.dumps(%r))" % CSMCIR_GOOD_OUTPUT),
        model_id=model_id, checkpoint_id=checkpoint, pin=pin)

    reasons = verdict_reasons(env, run_id, model_id, checkpoint, CSMCIR_PROTOCOL)
    assert any("no verified committed reference publishes a metric" in reason
               for reason in reasons), reasons


def test_verify_run_refuses_a_non_limn_run_when_extraction_not_performed(env):
    """A non-LIMN run whose metrics were never extracted is refused."""
    pin, checkpoint = csmcir_env(env)
    run_id, _ = run_and_read_report(
        env, evaluator(env["tmp"], "print(json.dumps(%r))" % CSMCIR_GOOD_OUTPUT),
        model_id="csmcir", checkpoint_id=checkpoint, pin=pin)
    reasons = report_reasons(env, run_id, {
        "metric_extraction_available": False,
        "extraction_status": "EVALUATION_COMPLETED_METRICS_UNPARSED",
    })
    assert any("no metric extraction was performed" in reason for reason in reasons), reasons
    assert any("the run's metrics were not extracted" in reason for reason in reasons), reasons


def test_verify_run_refuses_a_non_limn_run_with_a_nan_metric(env):
    """A non-LIMN run carrying a NaN metric alongside finite ones is refused."""
    pin, checkpoint = csmcir_env(env)
    run_id, _ = run_and_read_report(
        env, evaluator(env["tmp"], "print(json.dumps(%r))" % CSMCIR_GOOD_OUTPUT),
        model_id="csmcir", checkpoint_id=checkpoint, pin=pin)
    reasons = report_reasons(env, run_id, {
        "observed_metrics": {"r10": 57.1, "r50": float("nan"), "mean": 67.14},
    })
    assert any("not a finite Recall" in reason for reason in reasons), reasons


def test_verify_run_refuses_a_non_limn_run_with_macro_inconsistent_with_categories(env):
    """A non-LIMN run whose declared macro contradicts its categories is refused."""
    pin, checkpoint = csmcir_env(env)
    run_id, _ = run_and_read_report(
        env, evaluator(env["tmp"], "print(json.dumps(%r))" % CSMCIR_GOOD_OUTPUT),
        model_id="csmcir", checkpoint_id=checkpoint, pin=pin)
    reasons = report_reasons(env, run_id, {
        "category_metrics": {
            "dress": {"r10": 10.0, "r50": 10.0},
            "shirt": {"r10": 10.0, "r50": 10.0},
            "toptee": {"r10": 10.0, "r50": 10.0},
        },
        "observed_metrics": {"r10": 57.1, "r50": 77.18, "mean": 67.14},
    })
    assert any("is inconsistent with its categories" in reason for reason in reasons), reasons


def test_verify_run_refuses_a_non_limn_run_with_an_out_of_range_metric(env):
    """A non-LIMN run carrying an out-of-range metric is refused."""
    pin, checkpoint = csmcir_env(env)
    run_id, _ = run_and_read_report(
        env, evaluator(env["tmp"], "print(json.dumps(%r))" % CSMCIR_GOOD_OUTPUT),
        model_id="csmcir", checkpoint_id=checkpoint, pin=pin)
    reasons = report_reasons(env, run_id, {
        "observed_metrics": {"r10": 57.1, "r50": 150.0, "mean": 67.14},
    })
    assert any("not a finite Recall" in reason for reason in reasons), reasons

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


# ---------------------------------------------------------------------------------------------
# Registry/schema blockers D-2, D-3, D-4: a legitimate run must never be refused for a reason the
# operator cannot act on, a registry-declared data kind must be representable without becoming
# evidence, and a model whose authors published nothing must not read like a forged result.
#
# D-2 and D-4 are proved against the *real* committed registries (``registry/models.yaml`` and
# ``registry/paper_benchmarks/papers.yaml``), because the defects are properties of those files.
# ``verify_run.verify`` resolves its registry through ``WORKBENCH_REGISTRY_PATH`` (which the ``env``
# fixture overrides), so these cases call the same function ``verify`` calls for the same inputs
# (``verify_run.metric_evidence_reasons``) rather than re-deriving its wording.
# ---------------------------------------------------------------------------------------------

from workbench.backend import benchmark_references as benchmarks
from workbench.backend import paper_gate

PTHA_ROWS = ("ptha_mtst_fiq_table3_fashiqmtst_blip2", "ptha_mtst_fiq_no_pretrain_table3")
PTHA_PUBLISHED = {"r10": 54.99, "r50": 76.28, "mean": 65.64}


def test_a_legitimate_ptha_run_is_refused_with_the_registry_ambiguity_named():
    """D-2: two verified rows bind one condition, so the refusal must name that and pick neither.

    The two committed rows declare the same protocol, the same (unstated) training noise and the
    same checkpoint mapping, so no selector can choose between them on evidence the registry
    carries. Reporting "no verified committed reference publishes a metric" for such a run is
    false -- both rows publish all three -- and sends the operator looking for a missing table row
    that exists.
    """
    selection = benchmarks.select_reference_outcome(
        "ptha_mtst", checkpoint_training_noise_pct=None,
        protocol_id="fashioniq_original_split")

    assert selection.outcome == benchmarks.SELECTION_AMBIGUOUS, selection
    assert selection.reference is None, "an ambiguous registry must not yield a row to compare against"
    assert sorted(row.reference_id for row in selection.matched) == sorted(PTHA_ROWS), \
        selection.matched
    assert all(row.value_for("r10") is not None for row in selection.matched), \
        "both competing rows do publish metrics"
    for row_id in PTHA_ROWS:
        assert row_id in selection.reason, selection.reason
    assert "more than once" in selection.reason and "fashioniq_original_split" in selection.reason, \
        selection.reason
    # `(reference, reason)` stays consistent with the outcome: no row, and the same prose.
    reference, reason = benchmarks.select_reference(
        "ptha_mtst", checkpoint_training_noise_pct=None,
        protocol_id="fashioniq_original_split")
    assert reference is None and reason == selection.reason


def test_the_verifier_never_claims_no_row_publishes_a_metric_when_one_does():
    """D-2 at the verifier's own entry point: the reason must survive into the invocation proof."""
    reasons, outcome = verify_run.metric_evidence_reasons(
        "ptha_mtst", "fashioniq", "fashioniq_original_split",
        {"observed_metrics": dict(PTHA_PUBLISHED)})

    assert outcome == benchmarks.SELECTION_AMBIGUOUS, outcome
    assert reasons, "an ambiguous registry still refuses the run"
    joined = " ".join(reasons)
    assert "no verified committed reference publishes a metric" not in joined, joined
    for row_id in PTHA_ROWS:
        assert row_id in joined, joined
    assert "until the registry is resolved" in joined, joined


def test_a_registry_with_one_verified_row_is_still_selected_by_condition():
    """The D-2 fix must not turn condition selection into first-match-wins.

    HABIT has four verified rows under one protocol and they are distinguished by training noise;
    each condition still resolves to exactly its own row, and a condition no row states still
    refuses.
    """
    for noise, expected in ((0, "habit_aaai26_table1_fiq_n00"), (20, "habit_aaai26_table1_fiq_n20"),
                            (50, "habit_aaai26_table1_fiq_n50"), (80, "habit_aaai26_table1_fiq_n80")):
        selection = benchmarks.select_reference_outcome(
            "habit", checkpoint_training_noise_pct=noise,
            protocol_id="fashioniq_full_gallery_ref_excluded")
        assert selection.outcome == benchmarks.SELECTION_MATCHED, selection
        assert selection.reference.reference_id == expected, selection.reference
    missing = benchmarks.select_reference_outcome("habit", checkpoint_training_noise_pct=99)
    assert missing.outcome == benchmarks.SELECTION_NO_MATCHING_CONDITION, missing
    assert missing.reference is None


def test_a_model_whose_author_published_no_number_is_a_named_permanent_condition():
    """D-4: `combiner_rn50x4_noft` has no comparable row and must be reported as such.

    The model is registered, its parser is implemented and it can be executed, but its paper
    prints no FashionIQ row for the noft Combiner -- so the honest outcome is "no publishable
    comparison", distinct from a failed verification and from a comparison that measured a
    deviation. The registry says so itself: the model declares no `reported_*` value.
    """
    assert benchmarks.unbenchmarked("combiner_rn50x4_noft") is True
    assert benchmarks.unbenchmarked("clip4cir_rn50x4_fullft") is False, \
        "the fullft sibling does publish 43.78/66.93 and must stay comparable"

    selection = benchmarks.select_reference_outcome(
        "combiner_rn50x4_noft", checkpoint_training_noise_pct=0,
        protocol_id="fashioniq_original_split")
    assert selection.outcome == benchmarks.SELECTION_NO_PUBLISHED_COMPARISON, selection
    assert selection.author_published_no_reference is True, selection
    assert selection.reference is None, "there is nothing to compare against, so no row is returned"
    assert all(row.value_for("r10") is None for row in selection.candidates), selection.candidates

    reasons, outcome = verify_run.metric_evidence_reasons(
        "combiner_rn50x4_noft", "rn50x4_noft_unresolved", "fashioniq_original_split",
        {"observed_metrics": {"r10": 1.0, "r50": 2.0, "mean": 1.5}})
    assert outcome == benchmarks.SELECTION_NO_PUBLISHED_COMPARISON, outcome
    joined = " ".join(reasons)
    assert "no publishable comparison" in joined and "permanent" in joined, joined
    assert "forged" not in joined.lower() and "not a real experiment" not in joined, joined


def test_an_unbenchmarked_model_with_invalid_evidence_is_still_refused_for_that():
    """The D-4 allowance names a condition; it does not exempt the model from any other refusal.

    The same model id carrying a mock report is refused for declaring itself a non-experiment, and
    a report whose data kind is fine but whose declared state is inconsistent is refused too. The
    permanent condition is reported *beside* those reasons, never instead of them.
    """
    # The non-experiment refusal is the evidence-kind question, asked separately from the
    # reference question (the two are independent: a run can be a real experiment with no
    # publishable comparison, or a self-test with one).
    mock_kind = verify_run._evidence_kind_reasons({"data_kind": "mock"})
    assert any("not a real experiment" in reason for reason in mock_kind), mock_kind

    synthetic_kind = verify_run._evidence_kind_reasons(
        {"data_kind": "synthetic", "is_synthetic": True})
    assert any("declares itself synthetic" in reason for reason in synthetic_kind), synthetic_kind

    # The permanent-condition sentence is still reported beside them, not instead of them.
    reference_reasons, _ = verify_run.metric_evidence_reasons(
        "combiner_rn50x4_noft", "rn50x4_noft_unresolved", "fashioniq_original_split",
        {"data_kind": "mock"})
    assert any("no publishable comparison" in reason for reason in reference_reasons), \
        reference_reasons


def test_a_missing_verified_row_for_a_benchmarked_model_is_not_called_permanent():
    """The other half of D-4's distinction: a real gap in the table must read like a gap.

    PTHA+MTST's paper does print numbers and the registry records them, so its rows being
    unusable is a defect in the committed table; saying the authors published nothing would be a
    false statement about a paper that did.
    """
    gap = benchmarks.select_reference_outcome(
        "habit", checkpoint_training_noise_pct=99, protocol_id="fashioniq_full_gallery_ref_excluded")
    assert gap.outcome == benchmarks.SELECTION_NO_MATCHING_CONDITION
    assert gap.author_published_no_reference is False, gap
    assert benchmarks.unbenchmarked("ptha_mtst") is False


def test_a_synthetic_data_kind_is_representable_and_never_evidence():
    """D-3: the pipeline's own declaration must validate, and must not become research evidence.

    ``evaluate_models.py:185`` writes ``data_kind="synthetic"`` for a registry-declared synthetic
    source, and ``paper_gate.py:312`` refuses exactly that value. A schema that cannot represent it
    rejects the producer's honest declaration; a schema that accepts it must not soften either
    consumer.
    """
    from workbench.backend.schemas.results import DATA_KINDS, ResultRun
    from workbench.tests.mock_data import build_mock_runs

    assert "synthetic" in DATA_KINDS
    raw = build_mock_runs()[0].model_dump()
    raw["run"]["data_kind"] = "synthetic"
    validated = ResultRun.model_validate(raw)
    assert validated.run.data_kind == "synthetic"

    raw["run"]["data_kind"] = "fabricated"
    with pytest.raises(Exception):
        ResultRun.model_validate(raw)

    # The gate refuses it before any arithmetic, whatever the numbers say.
    reference, _ = benchmarks.select_reference(
        "csmcir", checkpoint_training_noise_pct=0, protocol_id="fashioniq_original_split")
    paper = {key: reference.value_for(key) for key in ("r10", "r50", "mean")}
    verdict = paper_gate.evaluate_gate(
        reference, dict(paper),
        run_meta={"protocol_id": "fashioniq_original_split", "checkpoint_id": "fashioniq",
                  "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0, "split": "val",
                  "dataset": "fashioniq"},
        data_kind="synthetic", real_execution=True)
    assert verdict.status == paper_gate.NOT_RUN and verdict.passed is False, verdict
    assert any("synthetic" in blocker for blocker in verdict.blockers), verdict

    # The verifier refuses it too, on the report's own declaration.
    kind_reasons = verify_run._evidence_kind_reasons({"data_kind": "synthetic"})
    assert any("not a real experiment" in reason for reason in kind_reasons), kind_reasons

    # And it is not quietly treated as an exempt development fixture: only `mock` is.
    from workbench.scripts.validate_results import _report_provenance

    lines: list[str] = []
    valid = _report_provenance([validated], strict_real=True, output=lines.append)
    assert valid is False and any("BLOCKED" in line for line in lines), lines


def test_autopilot_ledger_corroboration_contracts(env):
    """AP-1: autopilot requires corroboration against a valid evidence ledger.

    (a) a missing ledger and an HMAC-invalid ledger both REFUSE the run by name (not OFFICIAL_VERIFIED);
    (b) a ledger holding no entry for the run still refuses;
    (c) a legitimate entry still verifies.
    """
    from workbench.backend import autopilot

    plan = plan_for(env, reporting_evaluator(env["tmp"]))
    code, run_id = run_operator(env, plan)
    assert code == 0

    ap_cfg = autopilot.AutopilotConfig(
        path=env["tmp"] / "cfg.yaml",
        selected_model_ids=(MODEL,),
        paths={
            "artifacts": env["artifacts"],
            "checkpoints": env["tmp"] / "checkpoints",
            "results": env["tmp"] / "results",
            "dataset_root": env["tmp"] / "FashionIQ",
            "registry": env["tmp"],
            "third_party": env["tmp"] / "third_party",
        },
        budgets={"max_download_bytes": 0, "max_models": 1, "top_k": 10},
        permissions={},
        registry=registry_document(),
    )
    ap_plan = {
        "schema_version": 1,
        "solver_verified": True,
        "configuration_digest": "fixture",
        "models": [{"model_id": MODEL, "status": "NOT_RUN", "blockers": []}],
    }
    proof_path = em.completion_proof_path(env["config"], plan, run_id)
    eval_stage = {
        "model_id": MODEL,
        "name": "evaluation",
        "status": "COMPLETE",
        "run_id": run_id,
        "evidence": {"completion_proof": str(proof_path), "run_id": run_id},
    }
    execution = {"exit_code": 0, "stages": {f"evaluation:{MODEL}": eval_stage}}

    # (c) Legitimate entry verifies as OFFICIAL_VERIFIED
    rows, blockers = autopilot._classify(ap_plan, execution, ap_cfg)
    assert rows[0]["status"] == "COMPLETED"
    assert rows[0]["evaluation_kind"] == "OFFICIAL_VERIFIED"
    assert blockers == []

    # (b) Ledger holding no entry for this run still refuses
    ledger_path = em.evidence_ledger_path(env["config"])
    ledger_path.unlink()
    em.append_evidence_entry(env["config"], {
        "run_id": "other_run",
        "model_id": MODEL,
        "checkpoint_id": CHECKPOINT,
        "protocol_id": PROTOCOL,
        "report_digest": "0" * 64,
        "return_code": 0,
        "invocation_status": "INVOCATION_COMPLETED",
    })
    rows, blockers = autopilot._classify(ap_plan, execution, ap_cfg)
    assert rows[0]["evaluation_kind"] != "OFFICIAL_VERIFIED"
    assert rows[0]["status"] == "FAILED_OUTPUT_VALIDATION"
    assert any("evidence-ledger entry for this run" in r or "missing from evidence ledger" in r
               for b in blockers for r in b.get("reasons", []))

    # (a) HMAC-invalid ledger refuses by name
    ledger_path.write_text("{\"corrupted\": true}\n", encoding="utf-8")
    rows, blockers = autopilot._classify(ap_plan, execution, ap_cfg)
    assert rows[0]["evaluation_kind"] != "OFFICIAL_VERIFIED"
    assert rows[0]["status"] == "FAILED_OUTPUT_VALIDATION"
    assert any("the evidence ledger is missing or its HMAC chain does not verify" in r
               or "the evidence ledger is missing or its chain is broken" in r
               for b in blockers for r in b.get("reasons", []))

    # (a) Missing ledger refuses by name
    ledger_path.unlink()
    rows, blockers = autopilot._classify(ap_plan, execution, ap_cfg)
    assert rows[0]["evaluation_kind"] != "OFFICIAL_VERIFIED"
    assert rows[0]["status"] == "FAILED_OUTPUT_VALIDATION"
    assert any("the evidence ledger is missing or its HMAC chain does not verify" in r
               or "the evidence ledger is missing or its chain is broken" in r
               for b in blockers for r in b.get("reasons", []))


def test_unregistered_model_metric_evidence_contracts():
    """D-F1: an unregistered model requiring metric extraction must carry usable values.

    (d) an unregistered model with metric_extraction_available=True and no metrics is refused by name;
    (e) a run with real metrics still certifies.
    """
    no_metrics_report = {
        "metric_extraction_available": True,
        "extraction_status": "EXTRACTED_VERIFIED",
        "observed_metrics": None,
    }
    reasons_d, outcome_d = verify_run.metric_evidence_reasons(
        "unregistered_model", "ckpt", "fashioniq_original_split", no_metrics_report)
    assert outcome_d == benchmarks.SELECTION_UNREGISTERED
    assert any("claims metric extraction succeeded but carries no usable metrics" in r
               for r in reasons_d), reasons_d

    real_metrics_report = {
        "metric_extraction_available": True,
        "extraction_status": "EXTRACTED_VERIFIED",
        "observed_metrics": {"r10": 55.0, "r50": 75.0, "mean": 65.0},
    }
    reasons_e, outcome_e = verify_run.metric_evidence_reasons(
        "unregistered_model", "ckpt", "fashioniq_original_split", real_metrics_report)
    assert outcome_e == benchmarks.SELECTION_UNREGISTERED
    assert reasons_e == [], reasons_e


def test_partial_category_table_contracts():
    """D-R2-01: category table validation for non-LIMN vs LIMN models.

    (a) a non-LIMN run with a partial category table whose values are valid is ACCEPTED;
    (b) the same run with an out-of-range category value is REFUSED;
    (c) LIMN still requires all three categories.
    """
    report_partial_valid = {
        "metric_extraction_available": True,
        "extraction_status": "EXTRACTED_VERIFIED",
        "observed_metrics": {"r10": 55.0, "r50": 75.0, "mean": 65.0},
        "category_metrics": {"dress": {"r10": 50.0, "r50": 70.0}},
    }
    reasons_a, outcome_a = verify_run.metric_evidence_reasons(
        "unregistered_model", "ckpt", "fashioniq_original_split", report_partial_valid)
    assert outcome_a == benchmarks.SELECTION_UNREGISTERED
    assert reasons_a == [], reasons_a

    report_partial_bad = {
        "metric_extraction_available": True,
        "extraction_status": "EXTRACTED_VERIFIED",
        "observed_metrics": {"r10": 55.0, "r50": 75.0, "mean": 65.0},
        "category_metrics": {"dress": {"r10": 50.0, "r50": 150.0}},
    }
    reasons_b, outcome_b = verify_run.metric_evidence_reasons(
        "unregistered_model", "ckpt", "fashioniq_original_split", report_partial_bad)
    assert any("category dress r50 is not a finite Recall in [0, 100]: 150.0" in r
               for r in reasons_b), reasons_b

    reasons_c, outcome_c = verify_run.metric_evidence_reasons(
        "limn", "ckpt_v1", "fashioniq_original_split", report_partial_valid)
    assert any("category shirt is missing from the report" in r for r in reasons_c), reasons_c
    assert any("category toptee is missing from the report" in r for r in reasons_c), reasons_c


def test_ledger_key_conflict_appears_once_in_reasons(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """D-R2-02: a key conflict appears exactly once in reasons."""
    artifacts = tmp_path / "artifacts"
    reports = artifacts / "reports"
    reports.mkdir(parents=True)
    (reports / "csmcir_ckpt_r1_aggregate.json").write_text(json.dumps({
        "artifact_type": "official_aggregate_evaluation_report",
        "run_id": "r1",
        "model_id": "csmcir",
        "checkpoint_id": "ckpt",
        "protocol_id": "fashioniq_original_split",
    }), encoding="utf-8")

    monkeypatch.setenv("WORKBENCH_ARTIFACTS_ROOT", str(artifacts))
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

    verdict = verify_run.verify("r1", "csmcir", "ckpt", "fashioniq_original_split")
    key_reasons = [r for r in verdict["reasons"] if "the evidence ledger key is unusable" in r]
    assert len(key_reasons) == 1, key_reasons
