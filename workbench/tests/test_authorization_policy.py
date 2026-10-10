"""Regression tests demonstrating and preventing authorization bypasses across all modes."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import workbench.scripts.pipeline as pipeline
from workbench.backend.operator_config import WorkbenchConfig


@pytest.fixture
def test_cfg(tmp_path: Path):
    data = tmp_path / "FashionIQ"
    data.mkdir()
    (data / "captions").mkdir()
    (data / "image_splits").mkdir()
    (data / "images").mkdir()
    for cat in ("dress", "shirt", "toptee"):
        (data / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (data / "captions" / f"cap.{cat}.val.json").write_text("[]")

    third_party = tmp_path / "third_party"
    checkpoints = tmp_path / "checkpoints"
    results = tmp_path / "results"
    for d in (third_party, checkpoints, results):
        d.mkdir()

    return WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path / "repo",
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=data,
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=checkpoints,
        WORKBENCH_RESULTS_ROOT=results,
        WORKBENCH_THIRD_PARTY_ROOT=third_party,
    )


# A01: Legacy real --evaluate without --apply --allow-gpu-eval must NOT launch evaluation subprocess
def test_a01_real_evaluate_cannot_bypass_gpu_authorization(test_cfg, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "resolve_config", lambda: test_cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    # Calling real mode with --evaluate but without --apply --allow-gpu-eval
    rc = pipeline.main(["real", "--model", "csmcir", "--evaluate"])

    # Must be in planning mode or block; MUST NOT call run_command with evaluate_models.py
    eval_calls = [argv for argv in called if "evaluate_models.py" in str(argv)]
    assert eval_calls == [], f"Security bypass: evaluate_models was executed without authorization! Called: {eval_calls}"


# A02: Legacy real --sync-sources without --apply --allow-network must NOT launch sync subprocess
def test_a02_real_sync_cannot_bypass_network_authorization(test_cfg, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "resolve_config", lambda: test_cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    rc = pipeline.main(["real", "--model", "csmcir", "--sync-sources"])
    sync_calls = [argv for argv in called if "sync_upstreams.py" in str(argv)]
    assert sync_calls == [], f"Security bypass: sync_upstreams was executed without authorization! Called: {sync_calls}"


# A03: Legacy prepare without --apply --allow-preparation must NOT execute layout mutations
def test_a03_prepare_cannot_mutate_without_authorization(test_cfg, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "resolve_config", lambda: test_cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    rc = pipeline.main(["prepare", "--dataset-root", str(test_cfg.FASHIONIQ_ROOT)])
    prep_calls = [argv for argv in called if "--check-only" not in argv and ("prepare_dataset.py" in str(argv) or "prepare_model.py" in str(argv))]
    assert prep_calls == [], f"Security bypass: mutating prepare was executed without authorization! Called: {prep_calls}"


# A04: all --apply without capability flags does not grant capabilities
def test_a04_all_apply_without_capabilities_blocks(test_cfg, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "resolve_config", lambda: test_cfg)
    rc = pipeline.main(["all", "--model", "csmcir", "--dataset-root", str(test_cfg.FASHIONIQ_ROOT), "--apply"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "BLOCKED_AUTHORIZATION_REQUIRED" in err


# A05: Dry-run creates no state files, logs, or subprocess executions
def test_a05_dry_run_creates_zero_state_files_or_logs(test_cfg, monkeypatch, tmp_path):
    called = []
    state_file = tmp_path / "pipeline_state.json"
    report_file = tmp_path / "report.json"
    monkeypatch.setattr(pipeline, "resolve_config", lambda: test_cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    rc = pipeline.main(["all", "--model", "csmcir", "--dataset-root", str(test_cfg.FASHIONIQ_ROOT), "--dry-run", "--report", str(report_file)])
    assert rc == 0
    assert called == []
    assert not state_file.exists()
    assert not report_file.exists()


# A06: Unauthorized operations launch zero side-effecting subprocesses
def test_a06_unauthorized_stages_launch_zero_side_effecting_subprocesses(test_cfg, monkeypatch):
    called = []
    monkeypatch.setattr(pipeline, "resolve_config", lambda: test_cfg)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    # all mode without apply
    pipeline.main(["all", "--model", "csmcir", "--dataset-root", str(test_cfg.FASHIONIQ_ROOT)])
    assert called == [], f"Expected 0 subprocess calls in default planning mode, got {len(called)}: {called}"
