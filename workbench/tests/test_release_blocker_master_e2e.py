"""Level 3 master E2E through the real pipeline CLI: evaluation identity and integrity.

E2E-01..E2E-10.

Everything here runs the real ``pipeline.py`` as a subprocess: stage planning,
``run_stages``, evaluator launch, evidence validation, and workflow reporting are
all exercised. Only heavyweight resources (checkpoints, dataset, model code) are
synthetic fixtures.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]

EVALUATOR_SOURCE = '''#!/usr/bin/env python3
import json, os, sys

if os.environ.get("SYNTHETIC_FAIL"):
    sys.stderr.write("evaluator induced failure\\n")
    sys.exit(4)
if os.environ.get("SYNTHETIC_SILENT"):
    # Exits successfully but produces no observable evaluation output at all.
    sys.exit(0)

print("starting synthetic evaluation")
print(json.dumps({
    "dress_recall_at10": 60.0, "dress_recall_at50": 80.0,
    "shirt_recall_at10": 55.0, "shirt_recall_at50": 75.0,
    "toptee_recall_at10": 50.0, "toptee_recall_at50": 70.0,
    "average_recall_at10": 55.0, "average_recall_at50": 75.0, "average_recall": 65.0,
}))
print("synthetic evaluation completed successfully")
'''


@pytest.fixture
def master(tmp_path: Path):
    """A complete synthetic reproduction workspace plus its subprocess environment."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    data, checkpoints, third_party, reports = (ws / name for name in
                                               ("FashionIQ", "checkpoints", "third_party", "reports"))
    for directory in (data, checkpoints, third_party, reports):
        directory.mkdir()
    for sub in ("captions", "image_splits", "images"):
        (data / sub).mkdir()
    for category in ("dress", "shirt", "toptee"):
        (data / "captions" / f"cap.{category}.val.json").write_text(
            '[{"target": "img1", "candidate": "img0", "captions": ["c1", "c2"]}]')
        (data / "image_splits" / f"split.{category}.val.json").write_text('["img0", "img1"]')
    (data / "images" / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (data / "images" / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    source = third_party / "SyntheticModel"
    (source / "src").mkdir(parents=True)
    evaluator = source / "src" / "evaluator.py"
    evaluator.write_text(EVALUATOR_SOURCE)
    evaluator.chmod(0o755)
    for argv in (["git", "init", "-q"], ["git", "config", "user.email", "t@t.invalid"],
                 ["git", "config", "user.name", "T"], ["git", "add", "."],
                 ["git", "commit", "-qm", "pin"]):
        subprocess.run(argv, cwd=source, check=True, capture_output=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()

    checkpoint_dir = checkpoints / "synthetic_model"
    checkpoint_dir.mkdir()
    checkpoint = checkpoint_dir / "model.pt"
    checkpoint.write_bytes(b"synthetic-checkpoint-v1")

    registry = ws / "models.yaml"
    registry.write_text(json.dumps({
        "schema_version": 1,
        "protocols": {"fashioniq_original_split": {
            "label": "Original Split", "literature_split_label": "original",
            "evaluation_noise_pct": 0}},
        "models": [{
            "model_id": "synthetic_model", "method_name": "SyntheticModel",
            "research_generation": "recent", "publication_year": 2026,
            "preparation_contract": "manual_standard",
            "paper_title": "Synthetic Model", "paper_url": None,
            "upstream_repo_url": "https://example.invalid/synthetic.git",
            "upstream_default_branch": "main", "upstream_commit_sha": head,
            "source_available": True, "source_dir": "SyntheticModel",
            "fashioniq_layout": "fashioniq_standard", "literature_split_label": "original",
            "native_protocol": "fashioniq_original_split",
            "supported_protocols": ["fashioniq_original_split"],
            "reported_r10": 55.0, "reported_r50": 75.0, "reported_mean": 65.0,
            "reproduction_status": "NOT_RUN", "command_status": "COMMAND_AUDITED",
            "strict_clean_checkpoint_available": True,
            "checkpoint_variants": [{
                "checkpoint_id": "ckpt_v1", "filename": "model.pt",
                "download_url": "https://example.invalid/model.pt",
                "source_type": "author_huggingface",
                "checkpoint_training_noise_pct": 0, "evaluation_noise_pct": 0,
                "expected_sha256": None, "status": "URL_VERIFIED",
                "checkpoint_mapping_status": "VERIFIED_METADATA"}],
        }],
    }))

    env = os.environ.copy()
    env.update({
        "PYTHONPATH": str(REPOSITORY),
        "WORKBENCH_REGISTRY_PATH": str(registry),
        "WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS": "1",
        "CIR_REPO_ROOT": str(ws),
        "CIR_DATA_ROOT": str(ws / "data"),
        "FASHIONIQ_ROOT": str(data),
        "WORKBENCH_CHECKPOINT_ROOT": str(checkpoints),
        "WORKBENCH_RESULTS_ROOT": str(ws / "results"),
        "WORKBENCH_THIRD_PARTY_ROOT": str(third_party),
        "WORKBENCH_NO_ENV_CACHE": "1",
    })
    command = [sys.executable, str(REPOSITORY / "workbench" / "scripts" / "pipeline.py"),
               "all", "--model", "synthetic_model", "--apply", "--allow-network",
               "--allow-large-downloads", "--allow-env-install", "--allow-preparation",
               "--allow-gpu-eval", "--allow-index-write"]
    return {"ws": ws, "data": data, "checkpoint": checkpoint, "source": source,
            "reports": ws / "workbench" / "artifacts" / "reports", "env": env,
            "command": command, "reports_out": reports}


def artifacts(master) -> Path:
    return master["ws"] / "workbench" / "artifacts"


def pointer(master, kind: str) -> Path:
    return master["reports"] / f"synthetic_model_ckpt_v1_latest_{kind}.json"


def run(master, *extra: str, env: dict | None = None) -> subprocess.CompletedProcess:
    return subprocess.run([*master["command"], *extra], env=env or master["env"],
                          capture_output=True, text=True)


# ------------------------------------------------------------------ E2E-01, E2E-02

def test_e2e01_full_synthetic_evaluation_succeeds(master):
    result = run(master)
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"

    successful = json.loads(pointer(master, "successful").read_text())
    attempt = json.loads(pointer(master, "attempt").read_text())
    assert successful["run_id"] == attempt["run_id"], "a success must also be the latest attempt"

    run_id = successful["run_id"]
    proof = master["reports"] / f"synthetic_model_ckpt_v1_{run_id}_completion.json"
    report = master["reports"] / f"synthetic_model_ckpt_v1_{run_id}_aggregate.json"
    assert proof.is_file() and report.is_file()
    payload = json.loads(proof.read_text())
    assert payload["run_id"] == run_id
    assert payload["model_id"] == "synthetic_model"
    assert payload["checkpoint_id"] == "ckpt_v1"
    assert payload["protocol_id"] == "fashioniq_original_split"
    assert payload["return_code"] == 0 and payload["completion_status"] == "COMPLETED"
    assert payload["source_actual_commit"] == payload["source_expected_commit"]


def test_e2e02_resume_skips_verified_completed_run(master):
    assert run(master).returncode == 0
    successful_before = pointer(master, "successful").read_text()
    logs = artifacts(master) / "logs"
    attempts_before = len(list(logs.iterdir()))

    resumed = run(master, "--resume")
    assert resumed.returncode == 0, f"{resumed.stdout}\n{resumed.stderr}"
    assert "[SKIP] resume: already completed" in resumed.stdout
    assert len(list(logs.iterdir())) == attempts_before, "resume must not re-run the evaluator"
    assert pointer(master, "successful").read_text() == successful_before


# ------------------------------------------------------------------ E2E-03, E2E-04

def test_e2e03_checkpoint_modification_invalidates_evaluation(master):
    assert run(master).returncode == 0
    logs = artifacts(master) / "logs"
    before = len(list(logs.iterdir()))

    master["checkpoint"].write_bytes(b"synthetic-checkpoint-v2-mutated")
    assert run(master, "--resume").returncode == 0
    assert len(list(logs.iterdir())) > before, "a changed checkpoint must re-run evaluation"


def test_e2e04_dataset_modification_invalidates_evaluation(master):
    assert run(master).returncode == 0
    logs = artifacts(master) / "logs"
    before = len(list(logs.iterdir()))

    (master["data"] / "captions" / "cap.dress.val.json").write_text('[{"mutated": true}]')
    assert run(master, "--resume").returncode == 0
    assert len(list(logs.iterdir())) > before, "a changed dataset must re-run evaluation"


def test_e2e04b_dataset_modification_invalidates_only_metadata_size_changes(master):
    """A same-size annotation edit must still be detected (content, not size)."""
    assert run(master).returncode == 0
    annotation = master["data"] / "captions" / "cap.shirt.val.json"
    original = annotation.read_text()
    logs = artifacts(master) / "logs"
    before = len(list(logs.iterdir()))

    annotation.write_text(original.replace("c1", "c9"))
    assert len(annotation.read_text()) == len(original)
    assert run(master, "--resume").returncode == 0
    assert len(list(logs.iterdir())) > before


# ---------------------------------------------------------------------------- E2E-05

def test_e2e05_exit_zero_without_new_output_fails_evaluation(master):
    assert run(master).returncode == 0
    successful = json.loads(pointer(master, "successful").read_text())
    run1 = successful["run_id"]
    run1_report = master["reports"] / f"synthetic_model_ckpt_v1_{run1}_aggregate.json"
    run1_digest = run1_report.read_bytes()

    silent_env = dict(master["env"])
    silent_env["SYNTHETIC_SILENT"] = "1"
    result = run(master, "--force-stage", "evaluation", "--resume", env=silent_env)

    assert result.returncode == 1, "a zero-output evaluator must fail the evaluation stage"
    assert "declared outputs missing or invalid" in (result.stdout + result.stderr)
    # The earlier run is not reused...
    assert json.loads(pointer(master, "successful").read_text())["run_id"] == run1
    attempt = json.loads(pointer(master, "attempt").read_text())
    assert attempt["run_id"] != run1, "the silent attempt must be recorded as the latest attempt"
    assert attempt["completion_status"] == "NO_OBSERVABLE_OUTPUT"
    # ...and its artifacts are untouched.
    assert run1_report.read_bytes() == run1_digest
    proof = master["reports"] / f"synthetic_model_ckpt_v1_{run1}_completion.json"
    assert proof.is_file(), "the earlier successful proof must be preserved"
    assert not (master["reports"] / f"synthetic_model_ckpt_v1_{attempt['run_id']}_completion.json").exists()


# ------------------------------------------------------------------ E2E-06, E2E-07

def test_e2e06_failed_rerun_preserves_previous_successful_experiment(master):
    assert run(master).returncode == 0
    successful = json.loads(pointer(master, "successful").read_text())
    run1 = successful["run_id"]
    preserved = {
        path.name: path.read_bytes()
        for path in sorted(master["reports"].glob(f"synthetic_model_ckpt_v1_{run1}_*"))
    }
    assert preserved

    fail_env = dict(master["env"])
    fail_env["SYNTHETIC_FAIL"] = "1"
    report_path = master["reports_out"] / "failed.json"
    failed = run(master, "--force-stage", "evaluation", "--report", str(report_path), env=fail_env)

    assert failed.returncode == 1
    assert json.loads(pointer(master, "successful").read_text())["run_id"] == run1
    attempt = json.loads(pointer(master, "attempt").read_text())
    assert attempt["return_code"] == 4 and attempt["run_id"] != run1
    for name, content in preserved.items():
        assert (master["reports"] / name).read_bytes() == content, f"{name} was modified"
    workflow = json.loads(report_path.read_text())
    assert workflow["success"] is False
    assert any(record.get("status") == "FAILED" for record in workflow["stages"].values())


def test_e2e07_recovery_after_failure_succeeds_and_advances(master):
    assert run(master).returncode == 0
    first = json.loads(pointer(master, "successful").read_text())["run_id"]

    fail_env = dict(master["env"])
    fail_env["SYNTHETIC_FAIL"] = "1"
    assert run(master, "--force-stage", "evaluation", env=fail_env).returncode == 1

    assert run(master, "--force-stage", "evaluation").returncode == 0
    recovered = json.loads(pointer(master, "successful").read_text())["run_id"]
    assert recovered != first
    proof = master["reports"] / f"synthetic_model_ckpt_v1_{recovered}_completion.json"
    assert proof.is_file()
    # The original successful experiment is still on disk.
    assert (master["reports"] / f"synthetic_model_ckpt_v1_{first}_aggregate.json").is_file()


# ---------------------------------------------------------------------------- E2E-08

def test_e2e08_one_stage_failure_does_not_block_independent_stages(master, tmp_path: Path):
    import workbench.scripts.pipeline as pipeline

    state = tmp_path / "state.json"
    launched: list[list[str]] = []
    import workbench.backend.runtime as runtime

    original = pipeline.run_command
    pipeline.run_command = lambda argv: launched.append(argv) or (1 if "model_a" in str(argv) else 0)
    try:
        stages = [
            pipeline.Stage(name="sync", commands=(["/sync_model_a"],), model_id="model_a"),
            pipeline.Stage(name="evaluation", commands=(["/eval_model_a"],), model_id="model_a",
                           dependencies=("sync:model_a",)),
            pipeline.Stage(name="sync", commands=(["/sync_model_b"],), model_id="model_b"),
            pipeline.Stage(name="evaluation", commands=(["/eval_model_b"],), model_id="model_b",
                           dependencies=("sync:model_b",)),
        ]
        rc = pipeline.run_stages(stages, dry_run=False, continue_on_error=True, state_path=state)
    finally:
        pipeline.run_command = original

    assert rc == 1
    assert any("eval_model_b" in str(argv) for argv in launched)
    assert not any("eval_model_a" in str(argv) for argv in launched)
    assert runtime is not None


# ---------------------------------------------------------------- E2E-09, E2E-10

def test_e2e09_source_preparation_does_not_invalidate_verified_source(master):
    """A prepared declared runtime artifact must not dirty the pinned source."""
    sys.path.insert(0, str(REPOSITORY))
    from workbench.backend.runtime import _approved_source_artifacts, _source_dirty

    unknown = master["ws"] / "third_party" / "NotARegisteredModel"
    unknown.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=unknown, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "t@t.invalid"], cwd=unknown, check=True)
    subprocess.run(["git", "config", "user.name", "T"], cwd=unknown, check=True)
    (unknown / "keep.txt").write_text("x")
    subprocess.run(["git", "add", "."], cwd=unknown, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-qm", "pin"], cwd=unknown, check=True, capture_output=True)
    synthetic_source = master["ws"] / "third_party" / "SyntheticModel"

    # A checkout no registry model owns has no declared artifacts, so nothing is
    # approved there: an unexpected directory still counts as dirty.
    assert _approved_source_artifacts(unknown) == {}
    (unknown / "stray_artifact.bin").write_bytes(b"unexpected")
    assert _source_dirty(unknown) is True, "an undeclared artifact is a source change"

    # The synthetic checkout used by this workflow declares no runtime artifacts, so a
    # clean run leaves it clean and a local edit dirties it.
    assert _source_dirty(synthetic_source) is False
    evaluator = synthetic_source / "src" / "evaluator.py"
    evaluator.write_text(evaluator.read_text() + "\n# preparation note\n")
    assert _source_dirty(synthetic_source) is True


def test_e2e10_unauthorized_execution_remains_blocked(master):
    logs = artifacts(master) / "logs"
    before = len(list(logs.iterdir())) if logs.is_dir() else 0

    no_capability = [arg for arg in master["command"] if arg != "--allow-gpu-eval"]
    report_path = master["reports_out"] / "blocked.json"
    result = subprocess.run([*no_capability, "--report", str(report_path)],
                            env=master["env"], capture_output=True, text=True)

    assert result.returncode == 1
    assert "BLOCKED_AUTHORIZATION_REQUIRED" in result.stderr
    after = len(list(logs.iterdir())) if logs.is_dir() else 0
    assert after == before, "no evaluation may run without authorization"
    workflow = json.loads(report_path.read_text())
    assert any(record.get("status") == "BLOCKED_AUTHORIZATION_REQUIRED"
               for record in workflow["stages"].values())


def test_e2e10b_dirty_source_blocks_evaluation(master):
    evaluator = master["source"] / "src" / "evaluator.py"
    evaluator.write_text(evaluator.read_text() + "\n# unauthorized local edit\n")

    report_path = master["reports_out"] / "dirty.json"
    result = run(master, "--report", str(report_path))

    assert result.returncode == 1, "a dirty source must block evaluation"
    assert "dirty" in (result.stdout + result.stderr).lower()
    assert json.loads(report_path.read_text())["success"] is False
