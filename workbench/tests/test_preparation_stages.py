from __future__ import annotations

from pathlib import Path

from workbench.backend.operator_config import WorkbenchConfig
from workbench.backend.registry import load_registry
from workbench.scripts import pipeline


def _config(tmp_path: Path) -> WorkbenchConfig:
    return WorkbenchConfig(
        CIR_REPO_ROOT=tmp_path, CIR_DATA_ROOT=tmp_path / "data",
        FASHIONIQ_ROOT=tmp_path / "data" / "FashionIQ", WORKBENCH_HOST="127.0.0.1",
        WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173,
        WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints",
        WORKBENCH_RESULTS_ROOT=tmp_path / "results",
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party")


def test_preparation_stages_follow_declared_links_without_duplicate_writers(
        tmp_path: Path) -> None:
    models = load_registry()["models"]
    declaring = [model for model in models if model.get("canonical_dataset_link")]
    args = pipeline.parser_for().parse_args(
        ["all", "--all-models", "--apply", "--allow-preparation"])
    config = _config(tmp_path)

    stages = pipeline.all_stages(args, config)
    preparations = {stage.model_id: stage for stage in stages if stage.name == "preparation"}
    assert len(preparations) == len(models), "pipeline must retain per-model preparation status"
    assert len([stage for stage in preparations.values() if stage.output_paths]) == len(
        {model["source_dir"] + "/" + model["canonical_dataset_link"] for model in declaring})
    assert {stage.model_id for stage in preparations.values() if stage.output_paths} == {
        next(model["model_id"] for model in declaring
             if model["source_dir"] + "/" + model["canonical_dataset_link"] == path)
        for path in {model["source_dir"] + "/" + model["canonical_dataset_link"] for model in declaring}
    }

    for model in declaring:
        stage = preparations[model["model_id"]]
        expected = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] / model["canonical_dataset_link"]
        matching_writers = [item for item in preparations.values() if expected in item.output_paths]
        assert len(matching_writers) == 1, f"{expected} must have one writer"
        owner = matching_writers[0]
        if stage.model_id == owner.model_id:
            assert stage.output_paths == (expected,)
            assert "--execute" in stage.commands[0] and "--allow-preparation" in stage.commands[0]
        else:
            assert stage.output_paths == ()
            assert f"preparation:{owner.model_id}" in stage.dependencies
            assert "--execute" not in stage.commands[0]

    clip_models = [model for model in declaring if model["source_dir"] == "CLIP4Cir"]
    assert len(clip_models) == 2
    clip_link = config.WORKBENCH_THIRD_PARTY_ROOT / "CLIP4Cir" / clip_models[0]["canonical_dataset_link"]
    assert sum(clip_link in stage.output_paths for stage in preparations.values()) == 1


def test_preparation_status_stages_are_read_only_without_apply_authorization(tmp_path: Path) -> None:
    models = load_registry()["models"]
    args = pipeline.parser_for().parse_args(["all", "--all-models"])
    stages = pipeline.all_stages(args, _config(tmp_path))
    preparations = [stage for stage in stages if stage.name == "preparation"]

    declaring = [model for model in models if model.get("canonical_dataset_link")]
    assert len(preparations) == len(models)
    assert all(not stage.required_capabilities for stage in preparations)
    assert all("--execute" not in stage.commands[0] for stage in preparations)
    writers = [stage for stage in preparations if stage.output_paths]
    assert len(writers) == len({model["source_dir"] + "/" + model["canonical_dataset_link"]
                                for model in declaring})


def test_an_explicit_selection_scopes_the_planned_stages() -> None:
    """A caller that resolved a selection must not get stages for every registered model.

    The autopilot resolves `selected_models` itself, so without this the stage builder would plan
    all sixteen models for a run configured for one — and the run would then attempt work the
    operator never authorized.
    """
    from argparse import Namespace

    from workbench.backend.operator_config import resolve_config

    config = resolve_config()

    def planned(selection=None) -> set[str]:
        args = Namespace(
            mode="all", model=None, dataset_root=config.FASHIONIQ_ROOT, checkpoint=None,
            protocol=None, top_k=200, continue_on_error=True, allow_env_install=False,
            allow_preparation=False, apply=False, verify_gpu=False, sync_sources=False,
            download_checkpoints=False, download_auxiliary_downloads=False,
            download_auxiliary_assets=False, evaluate=False, rebuild_index=False, serve=False)
        if selection is not None:
            args.selected_model_ids = selection
        return {stage.model_id for stage in pipeline.all_stages(args, config) if stage.model_id}

    everything = planned()
    assert len(everything) > 2, "the registry must have several models for this to mean anything"

    one = planned(["limn"])
    assert one == {"limn"}, one

    two = planned(["limn", "pair"])
    assert two == {"limn", "pair"}, two

    import pytest as _pytest

    with _pytest.raises(ValueError, match="unknown"):
        planned(["not-a-registered-model"])


def test_an_unpinned_checkout_is_not_reproducible(monkeypatch) -> None:
    """A checkout that exists but has no recorded pin must be refused.

    The comparison previously ran only when a pin was present, so a registry entry with no
    `upstream_commit_sha` was accepted on whatever SHA happened to be checked out — the exact
    situation the pin exists to prevent.
    """
    from workbench.backend import runtime

    monkeypatch.setattr(runtime, "source_status", lambda source: {"tracked": [], "unauthorized": [],
                                                                 "invalid": []})
    source = Path("/repo")
    head = "7" * 40

    def git_run(*args, **kwargs):
        stdout = f"{source}\n" if "--show-toplevel" in args[0] else f"{head}\n"
        return type("R", (), {"stdout": stdout})()

    monkeypatch.setattr(runtime.subprocess, "run", git_run)

    clean, _, reason = runtime.source_clean_and_pinned({"upstream_commit_sha": None}, source)
    assert clean is False
    assert "pin missing" in reason

    clean, _, reason = runtime.source_clean_and_pinned({"upstream_commit_sha": head}, source)
    assert clean is True, reason


def test_solver_evidence_is_reported_from_the_planner_statuses(monkeypatch) -> None:
    """`solver_verified` must reflect what the planner recorded, not any non-"RUN" status.

    VERIFIED/CACHED_VERIFIED come from a real solve; NOT_RUN_REUSED means a runtime-verified
    interpreter was reused. Treating every other status as "not verified" mislabelled a successful
    solve, and treating reuse as a solve claimed evidence that does not exist.
    """
    import sys
    import types

    from workbench.backend import autopilot

    def report(groups: list[dict], overall: str) -> dict:
        module = types.ModuleType("workbench.backend.environment_planner")
        module.plan_environments = lambda models, artifacts_root=None: {
            "solver": "conda", "solver_verification_status": overall, "groups": groups}
        with monkeypatch.context() as patch:
            patch.setitem(sys.modules, "workbench.backend.environment_planner", module)
            return autopilot._environment_plan([{"model_id": "m"}], dry_run=False,
                                               artifacts_root=Path("/tmp/artifacts"))

    solved = report([{"solver_verification_status": "VERIFIED",
                      "runtime_verification_status": "NOT_CONFIGURED"}], "VERIFIED")
    assert solved["solver_verified"] is True and solved["solver_solved"] is True
    assert solved["source"] == "environment_planner"

    cached = report([{"solver_verification_status": "CACHED_VERIFIED",
                      "runtime_verification_status": "NOT_CONFIGURED"}], "VERIFIED")
    assert cached["solver_verified"] is True and cached["solver_solved"] is True

    reused = report([{"solver_verification_status": "NOT_RUN_REUSED",
                      "runtime_verification_status": "READY"}], "NOT_RUN")
    assert reused["solver_verified"] is True, "a runtime-verified reuse still covers the model"
    assert reused["solver_solved"] is False, "but it is not a solve"
    assert reused["source"] == "runtime_verified_reuse"

    unavailable = report([{"solver_verification_status": "UNAVAILABLE",
                           "runtime_verification_status": "NOT_CONFIGURED"}], "UNAVAILABLE")
    assert unavailable["solver_verified"] is False and unavailable["solver_solved"] is False


def test_offline_reuse_requires_no_network_permissions(tmp_path: Path, monkeypatch) -> None:
    config = _config(tmp_path)
    args = pipeline.parser_for().parse_args(["all", "--model", "limn"])

    # Case A: Present and verified -> frozenset(), --verify-only
    monkeypatch.setattr("workbench.backend.runtime.source_clean_and_pinned", lambda m, s: (True, "pin", None))
    monkeypatch.setattr("workbench.scripts.download_checkpoints.verify", lambda m, c, p: True)
    for f in ("0_dress_best_model.pt", "0_shirt_best_model.pt", "0_toptee_best_model.pt"):
        p = config.WORKBENCH_CHECKPOINT_ROOT / "limn" / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy")
    (config.WORKBENCH_THIRD_PARTY_ROOT / "LIMN").mkdir(parents=True, exist_ok=True)

    stages = pipeline.all_stages(args, config)
    sync_stage = next(s for s in stages if s.name == "sync")
    ckpt_stage = next(s for s in stages if s.name == "checkpoint")
    assert sync_stage.required_capabilities == frozenset()
    assert "--verify-only" in sync_stage.commands[0]
    assert ckpt_stage.required_capabilities == frozenset()
    assert "--verify-only" in ckpt_stage.commands[0]

    # Case B: Missing resources -> requires network permissions
    monkeypatch.setattr("workbench.backend.runtime.source_clean_and_pinned", lambda m, s: (False, None, "missing"))
    for p in (config.WORKBENCH_CHECKPOINT_ROOT / "limn").glob("*.pt"):
        p.unlink()

    stages_missing = pipeline.all_stages(args, config)
    sync_m = next(s for s in stages_missing if s.name == "sync")
    ckpt_m = next(s for s in stages_missing if s.name == "checkpoint")
    assert "allow_network" in sync_m.required_capabilities
    assert "allow_network" in ckpt_m.required_capabilities
    assert "allow_large_downloads" in ckpt_m.required_capabilities
    assert "--verify-only" not in sync_m.commands[0]
    assert "--verify-only" not in ckpt_m.commands[0]


def test_gpu_verification_stage_state_machine(tmp_path: Path, monkeypatch) -> None:
    import json
    import secrets
    from workbench.backend import runtime
    from workbench.tests.test_release_blocker_cuda_readiness import install_probe, SUCCESS

    config = _config(tmp_path)
    config.WORKBENCH_ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("WORKBENCH_LEDGER_KEY", secrets.token_hex(32))

    prefix = tmp_path / "env"
    site = prefix / "lib" / "python3.10" / "site-packages"
    site.mkdir(parents=True, exist_ok=True)
    interp = prefix / "bin" / "python"
    interp.parent.mkdir(parents=True, exist_ok=True)
    interp.write_text("#!/bin/sh\n")

    model = {
        "model_id": "csmcir",
        "environment": {"python": "3.10", "cuda": "12.4", "python_env_var": "WORKBENCH_CSMCIR_PYTHON"},
        "checkpoint_variants": [],
    }
    monkeypatch.setenv("WORKBENCH_CSMCIR_PYTHON", str(interp))
    monkeypatch.setattr("workbench.backend.registry.model_by_id", lambda mid: model)

    calls = []
    install_probe(SUCCESS, monkeypatch, install_prefix=prefix)
    orig_probe = runtime._GPU_PROBE_OVERRIDE
    def tracked_probe(m, rep):
        calls.append(True)
        return orig_probe(m, rep)
    runtime.set_gpu_probe_for_tests(tracked_probe)

    runtime.forget_gpu_capability()
    key = runtime._gpu_capability_key(model)
    cap_file = runtime.gpu_capability_path(config, key)
    assert not cap_file.exists()

    rep = runtime.EnvironmentReport(
        model_id="csmcir", tier=runtime.TIER_GPU_PENDING, status=runtime.TIER_GPU_PENDING,
        interpreter=interp, variable="WORKBENCH_CSMCIR_PYTHON",
    )
    monkeypatch.setattr(runtime, "verify_environment", lambda m, probe=True: rep)

    # 1. Probe runs when capability absent
    assert pipeline.probe_gpu_capability("csmcir", config) == 0
    assert len(calls) == 1
    assert cap_file.exists()

    # 2. Reused when valid
    assert pipeline.probe_gpu_capability("csmcir", config) == 0
    assert len(calls) == 1

    # 3. Refused and re-probed when stale/tampered
    payload = json.loads(cap_file.read_text())
    payload["signature"] = "tampered"
    cap_file.write_text(json.dumps(payload))
    runtime.forget_gpu_capability()
    assert pipeline.probe_gpu_capability("csmcir", config) == 0
    assert len(calls) == 2


def test_environment_plan_fingerprint_invalidated_by_selection_change(tmp_path: Path) -> None:
    config = _config(tmp_path)
    env_plan_path = tmp_path / "artifacts" / "environment_plan.json"

    cmd1 = pipeline.command("plan_environments.py", "--output", env_plan_path,
                            "--third-party-root", config.WORKBENCH_THIRD_PARTY_ROOT, "--model", "limn")
    stage1 = pipeline.run_stage("environment-plan", cmd1, output_paths=(env_plan_path,))
    fp1 = pipeline.compute_stage_input_fingerprint(stage1, config)

    cmd2 = pipeline.command("plan_environments.py", "--output", env_plan_path,
                            "--third-party-root", config.WORKBENCH_THIRD_PARTY_ROOT, "--model", "csmcir")
    stage2 = pipeline.run_stage("environment-plan", cmd2, output_paths=(env_plan_path,))
    fp2 = pipeline.compute_stage_input_fingerprint(stage2, config)

    assert fp1 != fp2, "Fingerprint must change when selection changes!"
