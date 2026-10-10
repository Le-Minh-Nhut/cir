"""Failing regression tests for DAG per-model execution graph, dependency enforcement, and failure isolation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

import workbench.scripts.pipeline as pipeline
from workbench.backend.operator_config import WorkbenchConfig


@pytest.fixture
def mock_cfg(tmp_path: Path):
    data = tmp_path / "FashionIQ"
    data.mkdir()
    (data / "captions").mkdir()
    (data / "image_splits").mkdir()
    (data / "images").mkdir()
    for cat in ("dress", "shirt", "toptee"):
        (data / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (data / "captions" / f"cap.{cat}.val.json").write_text("[]")

    return WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path / "repo",
        CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=data,
        WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000,
        WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
    )


# 1. Dependency enforcement: failed dependency must result in SKIPPED_DEPENDENCY, not execution
def test_failed_dependency_causes_skipped_dependency_not_execution(tmp_path: Path):
    state_file = tmp_path / "state.json"
    called = []

    def failing_run(argv):
        called.append(argv)
        if "failing.py" in str(argv):
            return 1
        return 0

    stage_dep = pipeline.Stage(
        name="sync",
        commands=([sys.executable, "failing.py"],),
        model_id="model_a",
    )
    stage_child = pipeline.Stage(
        name="evaluation",
        commands=([sys.executable, "child.py"],),
        model_id="model_a",
        dependencies=("sync:model_a",),
    )

    orig_run = pipeline.run_command
    try:
        pipeline.run_command = failing_run
        rc = pipeline.run_stages(
            [stage_dep, stage_child],
            dry_run=False,
            continue_on_error=True,
            state_path=state_file,
        )
    finally:
        pipeline.run_command = orig_run

    assert rc == 1
    # child.py must NOT be executed!
    child_calls = [argv for argv in called if "child.py" in str(argv)]
    assert child_calls == [], f"Child stage executed despite failed dependency! Calls: {called}"

    # State file must record SKIPPED_DEPENDENCY
    state = pipeline.load_pipeline_state(state_file)
    child_key = pipeline.stage_key(stage_child)
    assert state["stages"][child_key]["status"] == "SKIPPED_DEPENDENCY"


# 2. Multi-model failure isolation: Model A failure does not prevent independent Model B
def test_multi_model_failure_isolation_with_continue_on_error(tmp_path: Path):
    state_file = tmp_path / "state.json"
    called = []

    def mock_run(argv):
        called.append(argv)
        if "model_a" in str(argv):
            return 1
        return 0

    stage_a = pipeline.Stage(name="sync", commands=([sys.executable, "sync_model_a.py"],), model_id="model_a")
    stage_a_eval = pipeline.Stage(name="evaluation", commands=([sys.executable, "eval_model_a.py"],), model_id="model_a", dependencies=("sync:model_a",))

    stage_b = pipeline.Stage(name="sync", commands=([sys.executable, "sync_model_b.py"],), model_id="model_b")
    stage_b_eval = pipeline.Stage(name="evaluation", commands=([sys.executable, "eval_model_b.py"],), model_id="model_b", dependencies=("sync:model_b",))

    orig_run = pipeline.run_command
    try:
        pipeline.run_command = mock_run
        rc = pipeline.run_stages(
            [stage_a, stage_a_eval, stage_b, stage_b_eval],
            dry_run=False,
            continue_on_error=True,
            state_path=state_file,
        )
    finally:
        pipeline.run_command = orig_run

    # Model B sync and eval MUST have been executed!
    b_sync = [argv for argv in called if "sync_model_b.py" in str(argv)]
    b_eval = [argv for argv in called if "eval_model_b.py" in str(argv)]
    assert len(b_sync) == 1, "Model B sync was not executed"
    assert len(b_eval) == 1, "Model B evaluation was not executed"

    # Model A eval must NOT have been executed
    a_eval = [argv for argv in called if "eval_model_a.py" in str(argv)]
    assert len(a_eval) == 0, "Model A eval was executed despite failed sync"


# 3. all_stages with --all-models builds distinct per-model jobs
def test_all_models_builds_distinct_per_model_stages(mock_cfg):
    parser = pipeline.parser_for()
    args = parser.parse_args(["all", "--all-models", "--dry-run"])
    stages = pipeline.stages_for(args, mock_cfg)

    # Should contain distinct stages for registered models (csmcir, limn, dcnet, etc.)
    model_ids = {s.model_id for s in stages if s.model_id}
    assert "csmcir" in model_ids
    assert "limn" in model_ids
    assert "dcnet" in model_ids
    assert len(model_ids) >= 3, f"Expected multiple distinct model stages, got: {model_ids}"
