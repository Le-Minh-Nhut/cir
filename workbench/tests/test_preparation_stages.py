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

    for model in declaring:
        stage = preparations[model["model_id"]]
        expected = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] / model["canonical_dataset_link"]
        matching_writers = [item for item in preparations.values() if expected in item.output_paths]
        assert len(matching_writers) == 1, f"{expected} must have one writer"
        owner = matching_writers[0]
        if stage.model_id == owner.model_id:
            assert expected in stage.output_paths
            assert "--execute" in stage.commands[0] and "--allow-preparation" in stage.commands[0]
        else:
            assert expected not in stage.output_paths
            assert f"preparation:{owner.model_id}" in stage.dependencies
            assert "--execute" not in stage.commands[0]

    # A model that declares no link but does declare a generable artifact must still be prepared:
    # gating preparation on the link alone left LIMN's resize and dictionary unperformed.
    from workbench.backend.registry import preparation_contract_for_model
    artifact_only = [model for model in models
                     if not model.get("canonical_dataset_link")
                     and any(a.get("generator") in ("pinned_source_file", "fashioniq_resize_images")
                             for a in (preparation_contract_for_model(model) or {}).get("generated_artifacts", []))]
    assert artifact_only, "registry should declare at least one artifact-only preparation contract"
    for model in artifact_only:
        stage = preparations[model["model_id"]]
        assert stage.output_paths, f"{model['model_id']} must declare its produced paths"
        assert "--execute" in stage.commands[0] and "--allow-preparation" in stage.commands[0]

    clip_models = [model for model in declaring if model["source_dir"] == "CLIP4Cir"]
    assert len(clip_models) == 2
    clip_link = config.WORKBENCH_THIRD_PARTY_ROOT / "CLIP4Cir" / clip_models[0]["canonical_dataset_link"]
    assert sum(clip_link in stage.output_paths for stage in preparations.values()) == 1


def test_preparation_status_stages_are_read_only_without_apply_authorization(tmp_path: Path) -> None:
    models = load_registry()["models"]
    args = pipeline.parser_for().parse_args(["all", "--all-models"])
    stages = pipeline.all_stages(args, _config(tmp_path))
    preparations = [stage for stage in stages if stage.name == "preparation"]

    assert len(preparations) == len(models)
    # Without --apply nothing may be written: no capability is claimed and no execution is planned,
    # whatever a model's contract declares.
    assert all(not stage.required_capabilities for stage in preparations)
    assert all("--execute" not in stage.commands[0] for stage in preparations)


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


def test_csmcir_evaluation_command_parses_against_real_evaluator_parser(tmp_path: Path) -> None:
    import subprocess
    from argparse import Namespace
    from workbench.scripts.evaluate_models import parser_for

    config = _config(tmp_path)
    registry = load_registry()
    expected_ds = config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "fashionIQ_dataset"
    expected_canonical = config.FASHIONIQ_ROOT

    # 1. Per-model branch (all_stages)
    args_all = Namespace(
        mode="all", model=None, selected_model_ids=["csmcir"],
        checkpoint_ids={"csmcir": ["fashioniq"]},
        dataset_root=config.FASHIONIQ_ROOT, checkpoint=None,
        protocol="fashioniq_val_split", top_k=50, continue_on_error=True,
        allow_env_install=True, allow_preparation=True, apply=True,
        verify_gpu=True, sync_sources=True, download_checkpoints=True,
        download_auxiliary_assets=False, evaluate=True, rebuild_index=True, serve=False,
    )
    stages_all = pipeline.all_stages(args_all, config)
    eval_stage_all = next(s for s in stages_all if s.name == "evaluation" and s.model_id == "csmcir")
    cmd_all = eval_stage_all.commands[0]
    eval_argv_all = cmd_all[2:]
    parsed_all = parser_for(registry).parse_args(eval_argv_all)

    assert parsed_all.model == "csmcir"
    assert parsed_all.dataset_root == expected_ds
    assert parsed_all.canonical_dataset_root == expected_canonical
    assert parsed_all.checkpoint == ["fashioniq"]
    assert parsed_all.protocol == "fashioniq_val_split"
    assert parsed_all.top_k == 50
    assert eval_stage_all.required_capabilities == {"allow_gpu_eval", "allow_preparation"}

    smoke_all = subprocess.run([*cmd_all, "--dry-run"], capture_output=True, text=True)
    assert smoke_all.returncode in (0, 1)

    # 2. Single-model branch (real_stages)
    args_real = Namespace(
        mode="real", model="csmcir",
        dataset_root=config.FASHIONIQ_ROOT, checkpoint="fashioniq",
        protocol="fashioniq_val_split", top_k=50, continue_on_error=True,
        allow_env_install=True, allow_preparation=True, apply=True,
        verify_gpu=True, sync_sources=True, download_checkpoints=True,
        download_auxiliary_assets=False, evaluate=True, rebuild_index=True, serve=False,
    )
    stages_real = pipeline.real_stages(args_real, config)
    eval_stage_real = next(s for s in stages_real if s.name == "evaluation" and s.model_id == "csmcir")
    cmd_real = eval_stage_real.commands[0]
    eval_argv_real = cmd_real[2:]
    parsed_real = parser_for(registry).parse_args(eval_argv_real)

    assert parsed_real.model == "csmcir"
    assert parsed_real.dataset_root == expected_ds
    assert parsed_real.canonical_dataset_root == expected_canonical
    assert parsed_real.checkpoint == ["fashioniq"]
    assert parsed_real.protocol == "fashioniq_val_split"
    assert parsed_real.top_k == 50
    assert eval_stage_real.required_capabilities == {"allow_gpu_eval", "allow_preparation"}

    smoke_real = subprocess.run([*cmd_real, "--dry-run"], capture_output=True, text=True)
    assert smoke_real.returncode in (0, 1)


def test_fixed_dataset_link_authorization_semantics(tmp_path: Path, monkeypatch) -> None:
    from argparse import Namespace
    from workbench.scripts import prepare_dataset

    config = _config(tmp_path)
    config.FASHIONIQ_ROOT.mkdir(parents=True, exist_ok=True)
    csmcir_dir = config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR"
    csmcir_dir.mkdir(parents=True, exist_ok=True)
    link_path = csmcir_dir / "fashionIQ_dataset"
    source_marker = csmcir_dir / ".checkout-fixture"
    source_marker.write_text("fixture")
    monkeypatch.setattr("workbench.backend.runtime.source_clean_and_pinned",
                        lambda model, source: (True, "pin", None))

    # 1. VALID existing link: verified offline without network or preparation permissions
    link_path.symlink_to(config.FASHIONIQ_ROOT, target_is_directory=True)
    args_valid = Namespace(
        mode="all", model=None, selected_model_ids=["csmcir"],
        checkpoint_ids={"csmcir": ["fashioniq"]},
        dataset_root=config.FASHIONIQ_ROOT, checkpoint=None,
        protocol="fashioniq_val_split", top_k=50, continue_on_error=True,
        allow_env_install=False, allow_preparation=False, apply=True,
        verify_gpu=False, sync_sources=False, download_checkpoints=False,
        download_auxiliary_assets=False, evaluate=True, rebuild_index=False, serve=False,
    )
    monkeypatch.setattr(pipeline, "evaluation_evidence",
                        lambda model_dict, args, config, checkpoint_id=None: lambda record: (True, None, None))
    stages_valid = pipeline.all_stages(args_valid, config)
    prep_valid = next(s for s in stages_valid if s.name == "preparation" and s.model_id == "csmcir")
    assert prep_valid.required_capabilities == frozenset()
    # The command is stable across runs of a workflow (it is part of the stage fingerprint), so an
    # `--apply` invocation builds the executable form even when the link is already correct. The
    # stage claims no capability, which is what makes it read-only: the runner refuses to execute a
    # capability-bearing stage without authorization, and executes nothing here.
    assert "--execute" in prep_valid.commands[0]
    assert "--allow-preparation" in prep_valid.commands[0]
    sync_valid = next(s for s in stages_valid if s.name == "sync" and s.model_id == "csmcir")
    assert sync_valid.required_capabilities == frozenset()
    assert "--verify-only" in sync_valid.commands[0]
    ok, _ = prepare_dataset.prepare_canonical_link(
        config.FASHIONIQ_ROOT, csmcir_dir, "fashionIQ_dataset", check_only=True, dry_run=False)
    assert ok
    calls = []
    monkeypatch.setattr(pipeline, "validate_stage_outputs", lambda *args, **kwargs: True)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: calls.append(argv) or 0)
    eval_valid = next(s for s in stages_valid if s.name == "evaluation" and s.model_id == "csmcir")
    assert pipeline.run_stages(
        [sync_valid, prep_valid, eval_valid], dry_run=False, continue_on_error=False,
        apply=True, mode="all", authorized_capabilities={"allow_preparation", "allow_gpu_eval"},
        state_path=tmp_path / "valid-link-state.json", config=config) == 0
    assert [Path(argv[1]).name for argv in calls] == [
        "sync_upstreams.py", "prepare_model.py", "evaluate_models.py"]
    # The already-correct link was verified, not recreated: the stage was granted no capability, so
    # the runner treats it as read-only even though the template command carries --execute.
    assert not link_path.exists() or link_path.is_symlink()

    # 2. MISSING link: preparation authorized but network disabled does NOT require network
    link_path.unlink()
    args_missing = Namespace(
        mode="all", model=None, selected_model_ids=["csmcir"],
        checkpoint_ids={"csmcir": ["fashioniq"]},
        dataset_root=config.FASHIONIQ_ROOT, checkpoint=None,
        protocol="fashioniq_val_split", top_k=50, continue_on_error=True,
        allow_env_install=False, allow_preparation=True, apply=True,
        verify_gpu=False, sync_sources=False, download_checkpoints=False,
        download_auxiliary_assets=False, evaluate=True, rebuild_index=False, serve=False,
    )
    stages_missing = pipeline.all_stages(args_missing, config)
    prep_missing = next(s for s in stages_missing if s.name == "preparation" and s.model_id == "csmcir")
    assert prep_missing.required_capabilities == {"allow_preparation"}
    assert "allow_network" not in prep_missing.required_capabilities
    assert "--execute" in prep_missing.commands[0]
    calls.clear()
    assert pipeline.run_stages(
        [prep_missing], dry_run=False, continue_on_error=False, apply=True, mode="all",
        authorized_capabilities={"allow_preparation"},
        state_path=tmp_path / "missing-link-state.json", config=config) == 0
    ok, _ = prepare_dataset.prepare_canonical_link(
        config.FASHIONIQ_ROOT, csmcir_dir, "fashionIQ_dataset", check_only=False, dry_run=False)
    assert ok and link_path.is_symlink()

    # 3. MISSING source: sync stage still requires network
    link_path.unlink()
    source_marker.unlink()
    csmcir_dir.rmdir()
    stages_no_source = pipeline.all_stages(args_missing, config)
    sync_missing = next(s for s in stages_no_source if s.name == "sync" and s.model_id == "csmcir")
    assert sync_missing.required_capabilities == {"allow_network"}
    assert "allow_preparation" not in sync_missing.required_capabilities
    calls.clear()
    assert pipeline.run_stages(
        [sync_missing], dry_run=False, continue_on_error=False, apply=True, mode="all",
        authorized_capabilities={"allow_preparation"},
        state_path=tmp_path / "missing-source-state.json", config=config) == 1
    assert calls == []

    # 4. WRONG link target: rejected and not treated as valid
    csmcir_dir.mkdir(parents=True)
    source_marker.write_text("fixture")
    wrong_dir = tmp_path / "wrong_fashioniq"
    wrong_dir.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(wrong_dir, target_is_directory=True)
    stages_wrong = pipeline.all_stages(args_valid, config)
    prep_wrong = next(s for s in stages_wrong if s.name == "preparation" and s.model_id == "csmcir")
    assert prep_wrong.required_capabilities == {"allow_preparation"}
    ok, msgs = prepare_dataset.prepare_canonical_link(
        config.FASHIONIQ_ROOT, csmcir_dir, "fashionIQ_dataset", check_only=True, dry_run=False
    )
    assert ok is False
    assert any("resolves to" in m for m in msgs)


def test_environment_plan_fingerprint_invalidated_by_artifacts_root(tmp_path: Path) -> None:
    config = _config(tmp_path)
    env_plan_path = tmp_path / "artifacts" / "environment_plan.json"

    cmd1 = pipeline.command("plan_environments.py", "--output", env_plan_path,
                            "--third-party-root", config.WORKBENCH_THIRD_PARTY_ROOT,
                            "--artifacts-root", tmp_path / "artifacts1", "--model", "limn")
    stage1 = pipeline.run_stage("environment-plan", cmd1, output_paths=(env_plan_path,))
    fp1 = pipeline.compute_stage_input_fingerprint(stage1, config)

    cmd2 = pipeline.command("plan_environments.py", "--output", env_plan_path,
                            "--third-party-root", config.WORKBENCH_THIRD_PARTY_ROOT,
                            "--artifacts-root", tmp_path / "artifacts2", "--model", "limn")
    stage2 = pipeline.run_stage("environment-plan", cmd2, output_paths=(env_plan_path,))
    fp2 = pipeline.compute_stage_input_fingerprint(stage2, config)

    assert fp1 != fp2, "Fingerprint must change when artifacts-root changes!"


def test_evaluation_fingerprint_tracks_checkpoint_policy_and_overrides(tmp_path: Path) -> None:
    from argparse import Namespace

    config = _config(tmp_path)
    base = dict(
        mode="all", model=None, selected_model_ids=["csmcir"], checkpoint=None,
        checkpoint_ids={"csmcir": ["fashioniq"]}, checkpoint_overrides={},
        dataset_root=config.FASHIONIQ_ROOT, protocol=None, top_k=200,
        continue_on_error=True, allow_env_install=False, allow_preparation=False,
        apply=False, verify_gpu=False, sync_sources=False, download_checkpoints=False,
        download_auxiliary_assets=False, evaluate=True, rebuild_index=False, serve=False,
    )

    def fingerprint(policy: str, overrides: dict[str, list[str]] | None = None) -> str:
        args = Namespace(**(base | {"checkpoint_policy": policy,
                                    "checkpoint_overrides": overrides or {}}))
        stage = next(stage for stage in pipeline.all_stages(args, config)
                     if stage.name == "evaluation" and stage.model_id == "csmcir")
        return pipeline.compute_stage_input_fingerprint(stage, config)

    strict_digest = fingerprint("strict_clean")
    assert fingerprint("strict_clean") == strict_digest
    include_flagged_digest = fingerprint("include_flagged_conditions")
    assert include_flagged_digest != strict_digest
    override_digest = fingerprint("strict_clean", {"csmcir": ["fashioniq"]})
    assert override_digest != strict_digest


def test_every_registered_model_command_parses_with_its_real_script(tmp_path: Path, monkeypatch) -> None:
    import argparse
    import ast
    import importlib.util
    import inspect
    import sys

    config = _config(tmp_path)
    models = load_registry()["models"]
    monkeypatch.setattr("workbench.backend.runtime.source_clean_and_pinned",
                        lambda model, source: (False, None, "fixture"))
    monkeypatch.setattr("workbench.backend.runtime.verify_environment",
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fixture")))

    class Parsed(Exception):
        pass

    failures = []
    for model in models:
        model_id = model["model_id"]
        args = argparse.Namespace(
            mode="all", model=None, selected_model_ids=[model_id], checkpoint=None,
            checkpoint_ids={}, checkpoint_policy="strict_clean", checkpoint_overrides={},
            dataset_root=config.FASHIONIQ_ROOT, protocol=None, top_k=200,
            continue_on_error=True, allow_env_install=False, allow_preparation=False,
            apply=False, verify_gpu=False, sync_sources=False, download_checkpoints=True,
            download_auxiliary_assets=True, evaluate=True, rebuild_index=False, serve=False,
        )
        stages = pipeline.all_stages(args, config)
        real_args = argparse.Namespace(
            mode="real", model=model_id, dataset_root=config.FASHIONIQ_ROOT,
            checkpoint=None, protocol=None, top_k=200, continue_on_error=True,
            allow_env_install=False, allow_preparation=False, apply=False,
            verify_gpu=False, sync_sources=True, download_checkpoints=True,
            download_auxiliary_assets=True, evaluate=True, rebuild_index=False, serve=False,
        )
        stages.extend(pipeline.real_stages(real_args, config))
        for stage in stages:
            for argv in stage.commands:
                if len(argv) > 2 and argv[1] == "-c":
                    try:
                        ast.parse(argv[2])
                    except SyntaxError as error:
                        failures.append(f"{model_id}/{stage.name}: invalid Python command: {error}")
                    continue
                script_path = Path(argv[1])
                spec = importlib.util.spec_from_file_location(
                    f"pipeline_sweep_{script_path.stem}", script_path)
                if spec is None or spec.loader is None:
                    failures.append(f"{model_id}/{stage.name}: cannot load {script_path}")
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                parse_args = argparse.ArgumentParser.parse_args

                def capture(parser, parsed_argv=None, namespace=None):
                    raise Parsed(parse_args(parser, parsed_argv, namespace))

                try:
                    with monkeypatch.context() as patch:
                        patch.setattr(argparse.ArgumentParser, "parse_args", capture)
                        main = module.main
                        if "argv" in inspect.signature(main).parameters:
                            main(argv[2:])
                        else:
                            patch.setattr(sys, "argv", [str(script_path), *argv[2:]])
                            main()
                except Parsed:
                    pass
                except SystemExit as error:
                    failures.append(f"{model_id}/{stage.name}: {script_path.name} rejected "
                                    f"{argv[2:]} (exit {error.code})")

    assert not failures, "\n".join(failures)
