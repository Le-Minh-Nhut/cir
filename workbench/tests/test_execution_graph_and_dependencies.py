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


# 4. real mode: failed sync prevents real evaluation (dependency enforcement in legacy mode)
def test_real_mode_failed_sync_prevents_evaluation(tmp_path: Path, monkeypatch):
    from workbench.backend.operator_config import WorkbenchConfig
    data = tmp_path / "FashionIQ"
    (data / "captions").mkdir(parents=True)
    (data / "image_splits").mkdir()
    (data / "images").mkdir()
    for cat in ("dress", "shirt", "toptee"):
        (data / "image_splits" / f"split.{cat}.val.json").write_text("[]")
        (data / "captions" / f"cap.{cat}.val.json").write_text("[]")
    cfg = WorkbenchConfig(tmp_path / "repo", tmp_path / "d", data, "127.0.0.1", 8000, 5173,
                          tmp_path / "ck", tmp_path / "res", tmp_path / "tp")
    called = []

    def run(argv):
        called.append(argv)
        return 1 if "sync_upstreams.py" in str(argv) else 0

    monkeypatch.setattr(pipeline, "resolve_config", lambda: cfg)
    monkeypatch.setattr(pipeline, "run_command", run)

    rc = pipeline.main([
        "real", "--model", "csmcir", "--sync-sources", "--evaluate",
        "--apply", "--allow-network", "--allow-gpu-eval", "--allow-preparation",
        "--dataset-root", str(data), "--continue-on-error",
    ])
    assert not any("evaluate_models.py" in str(a) for a in called), "Evaluation launched after failed sync dependency"


# 5. A dependency persisted as FAILED in an EARLIER invocation must still block (F1/F2 regression)
def test_persisted_failed_dependency_blocks_later_invocation(tmp_path: Path, monkeypatch):
    state = tmp_path / "s.json"
    pipeline.save_pipeline_state({"stages": {"sync:m": {"status": "FAILED"}}}, state)
    launched: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)

    stages = [
        pipeline.Stage(name="environment", commands=(["/env"],), model_id="m",
                       dependencies=("sync:m",)),
        pipeline.Stage(name="evaluation", commands=(["/eval"],), model_id="m",
                       dependencies=("environment:m",)),
    ]
    rc = pipeline.run_stages(stages, dry_run=False, continue_on_error=True, resume=True, state_path=state)
    assert rc == 1
    assert launched == [], "a persisted FAILED dependency must not allow dependent execution"
    records = pipeline.load_pipeline_state(state)["stages"]
    assert records["environment:m"]["status"] == "SKIPPED_DEPENDENCY"


# 6. dataset-link declares a real output contract and can reach COMPLETE
def test_dataset_link_declares_output_contract():
    cfg = pipeline.resolve_config()
    args = pipeline.parser_for().parse_args(["real", "--model", "csmcir"])
    stage = next(s for s in pipeline.stages_for(args, cfg) if s.name == "dataset-link")
    assert stage.output_paths, "dataset-link must declare its CSMCIR dataset link output"

    # The proof must be satisfiable: create the symlink, then validate.
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        outer = Path(td) / "CSMCIR"
        outer.mkdir()
        target = Path(td) / "FashionIQ"
        target.mkdir()
        (target / "captions").mkdir()
        (outer / "fashionIQ_dataset").symlink_to(target, target_is_directory=True)
        probe = pipeline.Stage(name="dataset-link", commands=(["/x"],),
                               output_paths=(outer / "fashionIQ_dataset",))
        assert pipeline.validate_stage_outputs(probe) is True


# 7. LIMN evaluation proof filename must match the bundle id the evaluator writes
def test_limn_eval_output_uses_bundle_id():
    from workbench.backend.registry import model_by_id
    limn = model_by_id("limn")
    args = pipeline.parser_for().parse_args(["all", "--model", "limn"])
    outputs = pipeline.model_eval_outputs(limn, args, pipeline.resolve_config())
    assert any("base_iter0_all_categories" in str(p) for p in outputs), outputs


# 8. A provisioning stage with no artifact file must prove completion via a receipt
def test_provisioning_stage_requires_receipt(tmp_path: Path, monkeypatch):
    cfg = __import__("dataclasses").replace(
        pipeline.resolve_config(), CIR_REPO_ROOT=tmp_path)
    stage = pipeline.Stage(name="environment", commands=(["/env"],), model_id="m",
                           required_capabilities=frozenset({"allow_env_install"}), receipt=True)

    # No receipt yet -> not valid.
    assert pipeline.validate_stage_outputs(stage, config=cfg) is False

    launched: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)
    state = tmp_path / "s.json"
    assert pipeline.run_stages([stage], dry_run=False, apply=True,
                               authorized_capabilities={"allow_env_install"},
                               config=cfg, state_path=state) == 0
    assert launched == [["/env"]]
    # Receipt now exists and validates.
    assert pipeline.validate_stage_outputs(stage, config=cfg) is True
    receipt = pipeline.stage_receipt_path(cfg, stage)
    import json as _json
    assert _json.loads(receipt.read_text())["stage_id"] == "environment:m"


# 9. Persistent index writes require explicit authorization
def test_validation_index_requires_index_write_capability(tmp_path: Path, monkeypatch):
    launched: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)
    db = tmp_path / "workbench.duckdb"
    db.write_bytes(b"index")
    stages = [pipeline.Stage(name="validation-index",
                             commands=(["/validate"], ["/rebuild"]),
                             output_paths=(db,),
                             required_capabilities=frozenset({"allow_index_write"}))]

    # apply without the capability -> blocked, nothing launched
    rc = pipeline.run_stages(stages, dry_run=False, apply=True,
                             authorized_capabilities=set(), state_path=tmp_path / "s.json")
    assert rc == 1 and launched == []

    # with the capability -> runs
    rc = pipeline.run_stages(stages, dry_run=False, apply=True,
                             authorized_capabilities={"allow_index_write"},
                             state_path=tmp_path / "s2.json")
    assert rc == 0 and launched == [["/validate"], ["/rebuild"]]


# 10. Qualified dependency keys must be matched (F1 regression)
def test_qualified_dependency_key_is_matched(tmp_path: Path, monkeypatch):
    state = tmp_path / "s.json"
    # Recorded key carries the checkpoint qualifier; the dependency string does not.
    pipeline.save_pipeline_state(
        {"stages": {"checkpoint:limn:base_iter0_dress": {"status": "FAILED"}}}, state)
    launched: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: launched.append(argv) or 0)

    stage = pipeline.Stage(name="runtime-preflight", commands=(["/rp"],), model_id="limn",
                           dependencies=("checkpoint:limn",))
    rc = pipeline.run_stages([stage], dry_run=False, continue_on_error=True, resume=True,
                             state_path=state)
    assert rc == 1 and launched == [], "qualified recorded key did not block its dependency"
    assert pipeline.load_pipeline_state(state)["stages"]["runtime-preflight:limn"]["status"] == "SKIPPED_DEPENDENCY"


# 11. A dry-run must never mutate persistent workflow state (F3 regression)
def test_dry_run_never_mutates_state(tmp_path: Path, monkeypatch):
    state = tmp_path / "s.json"
    monkeypatch.setattr(pipeline, "run_command", lambda argv: 0)
    stages = [pipeline.Stage(name="checkpoint", commands=(["/dl"],),
                             required_capabilities=frozenset({"allow_large_downloads", "allow_network"}))]
    rc = pipeline.run_stages(stages, dry_run=True, apply=True,
                             authorized_capabilities={"allow_large_downloads"},
                             state_path=state)
    assert rc == 1
    assert not state.exists(), "dry-run must not write workflow state"
