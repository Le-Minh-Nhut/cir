"""True Level 3 Master E2E integration test executing pipeline.py via real subprocess."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPOSITORY = Path(__file__).resolve().parents[2]


@pytest.fixture
def true_e2e_workspace(tmp_path: Path):
    """Creates a fully isolated synthetic reproduction environment."""
    ws = tmp_path / "workspace"
    ws.mkdir()
    data = ws / "FashionIQ"
    checkpoints = ws / "checkpoints"
    results = ws / "results"
    third_party = ws / "third_party"
    reports = ws / "reports"
    artifacts = ws / "artifacts"

    for d in (data, checkpoints, results, third_party, reports, artifacts):
        d.mkdir(parents=True)

    # Synthetic FashionIQ dataset layout
    captions = data / "captions"
    splits = data / "image_splits"
    images = data / "images"
    captions.mkdir()
    splits.mkdir()
    images.mkdir()

    for cat in ("dress", "shirt", "toptee"):
        (captions / f"cap.{cat}.train.json").write_text('[{"target": "img1", "candidate": "img0", "captions": ["c1", "c2"]}]')
        (captions / f"cap.{cat}.val.json").write_text('[{"target": "img1", "candidate": "img0", "captions": ["c1", "c2"]}]')
        (splits / f"split.{cat}.val.json").write_text('["img0", "img1"]')
    (images / "img0.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (images / "img1.png").write_bytes(b"\x89PNG\r\n\x1a\n")

    # Synthetic upstream git repository for model
    source_dir = third_party / "SyntheticModel"
    source_dir.mkdir()
    subprocess.run(["git", "init"], cwd=source_dir, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Tester"], cwd=source_dir, check=True)
    subprocess.run(["git", "config", "user.email", "tester@test.invalid"], cwd=source_dir, check=True)

    src_sub = source_dir / "src"
    src_sub.mkdir()
    evaluator_py = src_sub / "evaluator.py"
    evaluator_py.write_text("""#!/usr/bin/env python3
import os, sys, json

if os.environ.get("SYNTHETIC_FAIL"):
    sys.stderr.write("Evaluator induced failure\\n")
    sys.exit(3)

print("Starting synthetic evaluation")
output = {
    "dress_recall_at10": 60.0,
    "dress_recall_at50": 80.0,
    "shirt_recall_at10": 55.0,
    "shirt_recall_at50": 75.0,
    "toptee_recall_at10": 50.0,
    "toptee_recall_at50": 70.0,
    "average_recall_at10": 55.0,
    "average_recall_at50": 75.0,
    "average_recall": 65.0
}
print(json.dumps(output, indent=4))
print("Synthetic evaluation completed successfully")
sys.exit(0)
""")
    evaluator_py.chmod(0o755)

    subprocess.run(["git", "add", "."], cwd=source_dir, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=source_dir, check=True)
    source_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source_dir, text=True).strip()

    # Checkpoint
    ckpt_dir = checkpoints / "synthetic_model"
    ckpt_dir.mkdir()
    ckpt_file = ckpt_dir / "model.pt"
    ckpt_file.write_bytes(b"synthetic-checkpoint-bytes-v1")

    # Synthetic registry pointing to our test model
    synthetic_registry = {
        "schema_version": 1,
        "protocols": {
            "fashioniq_original_split": {
                "label": "Original Split",
                "literature_split_label": "original",
                "evaluation_noise_pct": 0,
            }
        },
        "models": [
            {
                "model_id": "synthetic_model",
                "method_name": "SyntheticModel",
                "research_generation": "recent",
                "publication_year": 2026,
                "preparation_contract": "manual_standard",
                "paper_title": "Synthetic Model for Master E2E",
                "paper_url": None,
                "upstream_repo_url": "https://example.invalid/synthetic.git",
                "upstream_default_branch": "main",
                "upstream_commit_sha": source_head,
                "source_available": True,
                "source_dir": "SyntheticModel",
                "fashioniq_layout": "fashioniq_standard",
                "literature_split_label": "original",
                "native_protocol": "fashioniq_original_split",
                "supported_protocols": ["fashioniq_original_split"],
                "reported_r10": 55.0,
                "reported_r50": 75.0,
                "reported_mean": 65.0,
                "reproduction_status": "NOT_RUN",
                "command_status": "COMMAND_AUDITED",
                "strict_clean_checkpoint_available": True,
                "checkpoint_variants": [
                    {
                        "checkpoint_id": "ckpt_v1",
                        "filename": "model.pt",
                        "download_url": "https://example.invalid/model.pt",
                        "source_type": "author_huggingface",
                        "checkpoint_training_noise_pct": 0,
                        "evaluation_noise_pct": 0,
                        "expected_sha256": None,
                        "status": "URL_VERIFIED",
                        "checkpoint_mapping_status": "VERIFIED_METADATA",
                    }
                ],
            }
        ],
    }
    registry_yaml = ws / "models.yaml"
    registry_yaml.write_text(yaml.dump(synthetic_registry))

    # Environment dictionary for subprocess execution
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPOSITORY)
    env["WORKBENCH_REGISTRY_PATH"] = str(registry_yaml)
    env["WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS"] = "1"
    env["CIR_REPO_ROOT"] = str(ws)
    env["CIR_DATA_ROOT"] = str(ws / "data")
    env["FASHIONIQ_ROOT"] = str(data)
    env["WORKBENCH_CHECKPOINT_ROOT"] = str(checkpoints)
    env["WORKBENCH_RESULTS_ROOT"] = str(results)
    env["WORKBENCH_THIRD_PARTY_ROOT"] = str(third_party)

    # Register an adapter for synthetic_model dynamically in evaluate_models if needed
    from workbench.backend.adapters.base import EvalRequest
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter, register_test_adapter

    class SyntheticAdapter(OfficialScriptAdapter):
        model_id = "synthetic_model"
        script = "src/evaluator.py"

        def command(self, source, checkpoint, request: EvalRequest) -> list[str]:
            return [sys.executable, str(source / self.script)]

    register_test_adapter(SyntheticAdapter)

    return {
        "ws": ws,
        "data": data,
        "checkpoints": checkpoints,
        "ckpt_file": ckpt_file,
        "source_dir": source_dir,
        "source_head": source_head,
        "registry_yaml": registry_yaml,
        "env": env,
        "reports": reports,
    }


def test_true_master_pipeline_subprocess_e2e_complete_flow(true_e2e_workspace):
    """Full Level 3 Master E2E: Real CLI subprocess execution of pipeline.py through all stages."""
    w = true_e2e_workspace
    env = w["env"]
    report_file = w["reports"] / "workflow_report.json"
    pipeline_script = str(REPOSITORY / "workbench" / "scripts" / "pipeline.py")

    # RUN 1: Master 'all' workflow executed as a REAL subprocess
    cmd1 = [
        sys.executable,
        pipeline_script,
        "all",
        "--model", "synthetic_model",
        "--apply",
        "--allow-network",
        "--allow-large-downloads",
        "--allow-env-install",
        "--allow-preparation",
        "--allow-gpu-eval",
        "--allow-index-write",
        "--report", str(report_file),
    ]

    res1 = subprocess.run(cmd1, env=env, capture_output=True, text=True)
    assert res1.returncode == 0, f"Master pipeline failed:\nSTDOUT:\n{res1.stdout}\nSTDERR:\n{res1.stderr}"

    # Verify real workflow report was written
    assert report_file.is_file(), "Master workflow report was not created"
    rep1 = json.loads(report_file.read_text())
    assert rep1["success"] is True
    assert rep1["status"] == "SUCCESS"
    eval_key = next(k for k in rep1["stages"] if "evaluation" in k and "synthetic_model" in k)
    assert rep1["stages"][eval_key]["status"] == "COMPLETE"

    # RUN 2: Master 'all' with --resume -> must skip already completed stages
    cmd2 = cmd1 + ["--resume"]
    res2 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    assert res2.returncode == 0
    assert "[SKIP] resume: already completed" in res2.stdout, f"Resume did not skip completed work:\n{res2.stdout}"

    # RUN 3: Mutate checkpoint bytes -> --resume must invalidate and rerun evaluation
    logs_dir = w["ws"] / "workbench" / "artifacts" / "logs"
    logs_before = len(list(logs_dir.iterdir())) if logs_dir.is_dir() else 0
    w["ckpt_file"].write_bytes(b"mutated-checkpoint-bytes-v2")
    res3 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    assert res3.returncode == 0
    logs_after = len(list(logs_dir.iterdir()))
    assert logs_after > logs_before, f"Checkpoint mutation did not rerun evaluation (logs {logs_before} -> {logs_after})"

    # RUN 4: Mutate dataset annotation -> --resume must invalidate and rerun
    dress_ann = w["data"] / "captions" / "cap.dress.val.json"
    dress_ann.write_text('[{"mutated": true}]')
    logs_before = len(list(logs_dir.iterdir()))
    res4 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    assert res4.returncode == 0
    assert len(list(logs_dir.iterdir())) > logs_before, "Dataset mutation did not rerun evaluation"

    # RUN 5: Delete the run-bound completion proof -> --resume must detect that the
    # current invocation's evidence is gone and rerun (a stale report is not proof).
    reports_dir = w["ws"] / "workbench" / "artifacts" / "reports"
    successful = reports_dir / "synthetic_model_ckpt_v1_latest_successful.json"
    attempt = reports_dir / "synthetic_model_ckpt_v1_latest_attempt.json"
    assert successful.is_file(), "latest_successful pointer was not created"
    assert attempt.is_file(), "latest_attempt pointer was not created"
    proof = reports_dir / f"synthetic_model_ckpt_v1_{json.loads(successful.read_text())['run_id']}_completion.json"
    assert proof.is_file(), "run-bound completion proof was not created"
    proof.unlink()
    res5 = subprocess.run(cmd2, env=env, capture_output=True, text=True)
    assert res5.returncode == 0
    assert successful.is_file(), "latest_successful pointer missing after recovery"
    # The recovery run is the current latest success; a failing run must not beat it.
    last_run = json.loads(successful.read_text())["run_id"]
    # Run-scoped immutable reports accumulate; each run has its own file.
    run_reports = list((w["ws"] / "workbench" / "artifacts" / "reports").glob("synthetic_model_ckpt_v1_2*_aggregate.json"))
    assert len(run_reports) >= 2, f"expected multiple immutable run reports, got {run_reports}"

    # RUN 6: Induced evaluator failure -> must exit nonzero, report FAILED, preserve failure report
    fail_report = w["reports"] / "fail_report.json"
    fail_env = dict(env)
    fail_env["SYNTHETIC_FAIL"] = "1"
    cmd_fail = [
        sys.executable,
        pipeline_script,
        "all",
        "--model", "synthetic_model",
        "--apply",
        "--allow-network",
        "--allow-large-downloads",
        "--allow-env-install",
        "--allow-preparation",
        "--allow-gpu-eval",
        "--report", str(fail_report),
    ]
    res6 = subprocess.run(cmd_fail, env=fail_env, capture_output=True, text=True)
    assert res6.returncode == 1, f"Expected pipeline to exit 1 on failure, got {res6.returncode}\n{res6.stdout}\n{res6.stderr}"
    assert fail_report.is_file(), "Failure report was not preserved"
    rep_fail = json.loads(fail_report.read_text())
    assert rep_fail["success"] is False
    assert rep_fail["status"] in ("PARTIAL", "FAILED")
    assert any(s.get("status") == "FAILED" for s in rep_fail["stages"].values())

    # The failed invocation must not become the latest *successful* run.
    assert json.loads(successful.read_text())["run_id"] == last_run, (
        "a failed rerun replaced latest_successful")
    failed_attempt = json.loads(attempt.read_text())
    assert failed_attempt["return_code"] != 0, "failed attempt was not recorded as failed"
    assert failed_attempt["run_id"] != last_run


def test_true_master_dirty_source_blocks_evaluation(true_e2e_workspace):
    """E2E-05: a modified upstream source file (uncommitted) must block evaluation."""
    w = true_e2e_workspace
    evaluator = w["source_dir"] / "src" / "evaluator.py"
    evaluator.write_text(evaluator.read_text() + "\n# uncommitted local edit\n")

    report = w["reports"] / "dirty.json"
    cmd = [
        sys.executable, str(REPOSITORY / "workbench" / "scripts" / "pipeline.py"),
        "all", "--model", "synthetic_model",
        "--apply", "--allow-network", "--allow-large-downloads",
        "--allow-env-install", "--allow-preparation", "--allow-gpu-eval",
        "--report", str(report),
    ]
    res = subprocess.run(cmd, env=w["env"], capture_output=True, text=True)
    assert res.returncode == 1, f"dirty source did not fail the workflow:\n{res.stdout}\n{res.stderr}"
    assert "dirty" in (res.stdout + res.stderr).lower()
    data = json.loads(report.read_text())
    assert data["success"] is False


def test_true_master_missing_permission_blocks_side_effects(true_e2e_workspace):
    """E2E-11: without the GPU capability, no evaluation subprocess may run."""
    w = true_e2e_workspace
    logs_dir = w["ws"] / "workbench" / "artifacts" / "logs"
    before = len(list(logs_dir.iterdir())) if logs_dir.is_dir() else 0

    report = w["reports"] / "noperm.json"
    cmd = [
        sys.executable, str(REPOSITORY / "workbench" / "scripts" / "pipeline.py"),
        "all", "--model", "synthetic_model",
        "--apply", "--allow-network", "--allow-large-downloads",
        "--allow-env-install", "--allow-preparation",
        "--report", str(report),
    ]
    res = subprocess.run(cmd, env=w["env"], capture_output=True, text=True)
    assert res.returncode == 1
    assert "BLOCKED_AUTHORIZATION_REQUIRED" in res.stderr
    after = len(list(logs_dir.iterdir())) if logs_dir.is_dir() else 0
    assert after == before, "evaluation artifacts were produced without authorization"
    data = json.loads(report.read_text())
    assert any(s.get("status") == "BLOCKED_AUTHORIZATION_REQUIRED" for s in data["stages"].values())
