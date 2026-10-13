from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace



def load_pipeline():
    path = Path("workbench/scripts/pipeline.py")
    spec = importlib.util.spec_from_file_location("pipeline", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def stage_lines(output: str) -> list[str]:
    return [line for line in output.splitlines() if line.startswith("[") and len(line) > 1 and line[1].isdigit()]


def test_mock_dry_run_skips_serve_without_flag(monkeypatch, capsys) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["mock", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert stage_lines(output) == [
        "[1/5] doctor",
        "[2/5] load mock",
        "[3/5] validate",
        "[4/5] rebuild",
        "[5/5] serve",
    ]
    assert called == []
    assert "serve_workbench.py" not in output


def test_mock_serve_is_opt_in(monkeypatch) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["mock", "--serve"]) == 0

    assert [Path(argv[1]).name for argv in called] == ["doctor.py", "load_mock_results.py", "validate_results.py", "rebuild_index.py", "serve_workbench.py"]


def real_config() -> SimpleNamespace:
    return SimpleNamespace(
        FASHIONIQ_ROOT=Path("/configured/FashionIQ"),
        WORKBENCH_THIRD_PARTY_ROOT=Path("/configured/third_party"),
        WORKBENCH_CHECKPOINT_ROOT=Path("/configured/checkpoints"),
        WORKBENCH_RESULTS_ROOT=Path("/configured/results"),
    )


def test_real_csmcir_dry_run_orders_complete_preparation(monkeypatch, capsys) -> None:
    pipeline = load_pipeline()
    monkeypatch.setattr(pipeline, "resolve_config", real_config)

    assert pipeline.main(["real", "--model", "csmcir", "--sync-sources", "--download-checkpoints", "--download-auxiliary-assets", "--evaluate", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert stage_lines(output) == [
        "[1/10] doctor", "[2/10] dataset", "[3/10] sync", "[4/10] dataset-link", "[5/10] checkpoint",
        "[6/10] auxiliary-assets", "[7/10] runtime-preflight", "[8/10] evaluation", "[9/10] validation-index", "[10/10] serve",
    ]
    plans = [line for line in output.splitlines() if "[PLAN]" in line]
    assert "doctor.py --scope workbench" in plans[0]
    assert "prepare_dataset.py --dataset-root /configured/FashionIQ --check-only" in plans[1]
    assert "sync_upstreams.py --model csmcir" in plans[2]
    assert "prepare_dataset.py --dataset-root /configured/FashionIQ --model csmcir" in plans[3]
    assert "download_checkpoints.py --model csmcir" in plans[4]
    assert "download_auxiliary_assets.py --model csmcir" in plans[5]
    assert "doctor.py --scope real --dataset-root /configured/FashionIQ --model csmcir" in plans[6]
    assert "evaluate_models.py --model csmcir --dataset-root /configured/third_party/CSMCIR/fashionIQ_dataset --canonical-dataset-root /configured/FashionIQ" in plans[7]
    assert "validate_results.py" not in output


def test_mock_keeps_workbench_doctor(monkeypatch) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["mock"]) == 0
    assert called[0][2:] == ["--scope", "workbench"]


def test_encoder_auxiliary_selection_does_not_invoke_csmcir(monkeypatch) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "resolve_config", real_config)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--model", "encoder", "--download-auxiliary-assets"]) == 0
    assert [Path(argv[1]).name for argv in called] == ["doctor.py", "prepare_dataset.py"]


def test_all_auxiliary_selection_uses_registry_all_mode(monkeypatch) -> None:
    pipeline = load_pipeline()
    monkeypatch.setattr(pipeline, "resolve_config", real_config)

    args = pipeline.parser_for().parse_args(["real", "--download-auxiliary-assets"])
    stages = pipeline.stages_for(args, real_config())
    auxiliary = next(s for s in stages if s.name == "auxiliary-assets")
    assert auxiliary.commands, "auxiliary-assets stage should be planned"
    argv = auxiliary.commands[0]
    assert Path(argv[1]).name == "download_auxiliary_assets.py"
    assert argv[2:] == ["--all"]


def test_evaluation_does_not_validate_aggregate_only_output(monkeypatch) -> None:
    pipeline = load_pipeline()
    monkeypatch.setattr(pipeline, "resolve_config", real_config)

    args = pipeline.parser_for().parse_args(["real", "--model", "csmcir", "--evaluate"])
    stages = pipeline.stages_for(args, real_config())
    assert any(s.name == "evaluation" for s in stages)
    # A plain --evaluate run must not implicitly validate or index aggregate-only output:
    # the validation-index stage must exist but be skipped, and must not be executable.
    vi = next(s for s in stages if s.name == "validation-index")
    assert not vi.commands, "validation-index must not run without --rebuild-index"


def test_rebuild_index_stays_explicit(monkeypatch) -> None:
    """--rebuild-index is opt-in: absent unless requested; present and capability-gated when it is."""
    pipeline = load_pipeline()
    monkeypatch.setattr(pipeline, "resolve_config", real_config)

    # Not requested -> the stage is planned-as-skipped.
    args = pipeline.parser_for().parse_args(["real", "--model", "csmcir"])
    stages = pipeline.stages_for(args, real_config())
    vi = next(s for s in stages if s.name == "validation-index")
    assert not vi.commands

    # Requested -> validate+rebuild planned, gated on the index-write capability.
    args = pipeline.parser_for().parse_args(["real", "--model", "csmcir", "--rebuild-index"])
    stages = pipeline.stages_for(args, real_config())
    vi = next(s for s in stages if s.name == "validation-index")
    assert [Path(argv[1]).name for argv in vi.commands] == ["validate_results.py", "rebuild_index.py"]
    assert "allow_index_write" in vi.required_capabilities


def test_runtime_preflight_cannot_continue_into_evaluation(monkeypatch, tmp_path) -> None:
    """Without --continue-on-error, a failed preflight must stop the workflow (fail-fast)."""
    pipeline = load_pipeline()
    stages = [
        pipeline.run_stage("runtime-preflight", ["doctor.py"], model_id="m"),
        pipeline.run_stage("evaluation", ["evaluate_models.py"], model_id="m",
                           dependencies=("runtime-preflight:m",)),
    ]
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 2)

    # Fail-fast: the evaluation stage must never be launched after a failed preflight.
    # The workflow state is per-test: the live state path is globally locked, so using it
    # here would make the test depend on whatever else is running.
    assert pipeline.run_stages(stages, dry_run=False, continue_on_error=False,
                               state_path=tmp_path / "state.json") == 1
    assert called == [["doctor.py"]]

    # Even WITH --continue-on-error, a failing preflight must not let its dependent
    # evaluation run; it must be recorded as SKIPPED_DEPENDENCY instead.
    called.clear()
    state = tmp_path / "state.json"
    assert pipeline.run_stages(stages, dry_run=False, continue_on_error=True, state_path=state) == 1
    assert called == [["doctor.py"]]
    records = pipeline.load_pipeline_state(state)["stages"]
    assert records["runtime-preflight:m"]["status"] == "FAILED"
    assert records["evaluation:m"]["status"] == "SKIPPED_DEPENDENCY"


def test_unknown_model_is_rejected_before_stages(monkeypatch) -> None:
    pipeline = load_pipeline()
    monkeypatch.setattr(pipeline, "resolve_config", real_config)

    try:
        pipeline.main(["real", "--model", "unknown"])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError("unknown model accepted")


def test_csmcir_plan_uses_custom_canonical_root(monkeypatch, capsys) -> None:
    pipeline = load_pipeline()
    monkeypatch.setattr(pipeline, "resolve_config", real_config)
    custom = Path("/requested/FashionIQ")

    assert pipeline.main(["real", "--model", "csmcir", "--dataset-root", str(custom), "--evaluate", "--dry-run"]) == 0

    output = capsys.readouterr().out
    assert f"doctor.py --scope real --dataset-root {custom} --model csmcir" in output
    assert f"evaluate_models.py --model csmcir --dataset-root /configured/third_party/CSMCIR/fashionIQ_dataset --canonical-dataset-root {custom}" in output

def test_resume_skips_completed_stage(monkeypatch, tmp_path: Path, capsys) -> None:
    pipeline = load_pipeline()
    state_file = tmp_path / "pipeline_state.json"
    called = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    stages = [
        pipeline.run_stage("stage1", ["cmd1.py"]),
        pipeline.run_stage("stage2", ["cmd2.py"]),
    ]

    # First run without resume -> both run and state is saved
    rc = pipeline.run_stages(stages, dry_run=False, continue_on_error=False, resume=True, state_path=state_file)
    assert rc == 0
    assert len(called) == 2
    assert state_file.is_file()

    # Second run with resume -> both skipped
    called.clear()
    capsys.readouterr()
    rc = pipeline.run_stages(stages, dry_run=False, continue_on_error=False, resume=True, state_path=state_file)
    assert rc == 0
    assert len(called) == 0
    out = capsys.readouterr().out
    assert "resume: already completed" in out


def test_force_stage_overrides_resume(monkeypatch, tmp_path: Path) -> None:
    pipeline = load_pipeline()
    state_file = tmp_path / "pipeline_state.json"
    called = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    stages = [
        pipeline.run_stage("stage1", ["cmd1.py"]),
        pipeline.run_stage("stage2", ["cmd2.py"]),
    ]

    pipeline.run_stages(stages, dry_run=False, continue_on_error=False, resume=True, state_path=state_file)
    called.clear()

    # Force stage2 only
    rc = pipeline.run_stages(stages, dry_run=False, continue_on_error=False, resume=True, force_stages={"stage2"}, state_path=state_file)
    assert rc == 0
    assert len(called) == 1
    assert called[0] == ["cmd2.py"]


def test_resume_reruns_when_fingerprint_changes(monkeypatch, tmp_path: Path) -> None:
    pipeline = load_pipeline()
    state_file = tmp_path / "pipeline_state.json"
    called = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    stages1 = [pipeline.run_stage("stage1", ["cmd1.py", "--arg1"])]
    pipeline.run_stages(stages1, dry_run=False, continue_on_error=False, resume=True, state_path=state_file)
    called.clear()

    # Changing arguments changes fingerprint -> reruns!
    stages2 = [pipeline.run_stage("stage1", ["cmd1.py", "--arg2"])]
    pipeline.run_stages(stages2, dry_run=False, continue_on_error=False, resume=True, state_path=state_file)
    assert len(called) == 1
    assert called[0] == ["cmd1.py", "--arg2"]


def test_setup_reproduce_analyze_all_modes_accepted() -> None:
    pipeline = load_pipeline()
    for mode in ("setup", "reproduce", "analyze", "all", "report"):
        parser = pipeline.parser_for()
        args = parser.parse_args([mode, "--dry-run"])
        assert args.mode == mode


def test_all_stages_csmcir_evaluation_command_parses_with_evaluate_models() -> None:
    pipeline = load_pipeline()
    from workbench.backend.registry import load_registry
    from workbench.scripts.evaluate_models import parser_for

    registry = load_registry()
    args = pipeline.parser_for().parse_args([
        "all", "--model", "csmcir", "--dataset-root", "/custom/FashionIQ",
        "--apply", "--evaluate", "--checkpoint", "fashioniq", "--protocol", "fashioniq_val_split", "--top-k", "100",
    ])
    config = real_config()
    stages = pipeline.all_stages(args, config)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "csmcir")
    cmd = eval_stage.commands[0]
    # Ensure --dataset-root is preserved as flag and not replaced by path
    assert "--dataset-root" in cmd
    assert "--canonical-dataset-root" in cmd
    parsed = parser_for(registry).parse_args(cmd[2:])
    assert parsed.model == "csmcir"
    assert parsed.dataset_root == Path("/configured/third_party/CSMCIR/fashionIQ_dataset")
    assert parsed.canonical_dataset_root == Path("/custom/FashionIQ")
    assert parsed.checkpoint == ["fashioniq"]
    assert parsed.protocol == "fashioniq_val_split"
    assert parsed.top_k == 100
    assert eval_stage.required_capabilities == {"allow_gpu_eval", "allow_preparation"}


def test_selected_checkpoint_bytes_mutation_changes_fingerprint_unselected_does_not(tmp_path: Path):
    pipeline = load_pipeline()
    from dataclasses import replace
    from workbench.backend.operator_config import resolve_config

    ckpt_dir = tmp_path / "checkpoints" / "airknow"
    ckpt_dir.mkdir(parents=True)
    ckpt_a = ckpt_dir / "airknow_fiq_0.5.pt"
    ckpt_b = ckpt_dir / "airknow_fiq_0.8.pt"
    ckpt_a.write_bytes(b"initial bytes A")
    ckpt_b.write_bytes(b"initial bytes B")

    cfg = replace(resolve_config(), WORKBENCH_CHECKPOINT_ROOT=tmp_path / "checkpoints")
    args = SimpleNamespace(
        mode="all",
        model="airknow",
        selected_model_ids=["airknow"],
        dataset_root=None,
        checkpoint=None,
        checkpoint_ids={"airknow": ["fiq_n05"]},
        checkpoint_policy="strict_clean",
        checkpoint_overrides={},
        protocol=None,
        top_k=10,
        continue_on_error=True,
        verify_gpu=False,
        sync_sources=False,
        download_checkpoints=False,
        download_auxiliary_assets=False,
        evaluate=True,
        rebuild_index=False,
    )
    stages = pipeline.all_stages(args, cfg)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "airknow")

    fp1 = pipeline.compute_stage_input_fingerprint(eval_stage, cfg)

    # Mutate unselected checkpoint B
    ckpt_b.write_bytes(b"mutated bytes B")
    fp2 = pipeline.compute_stage_input_fingerprint(eval_stage, cfg)
    assert fp1 == fp2, "mutating unselected checkpoint B must not change evaluation input fingerprint"

    # Mutate selected checkpoint A
    ckpt_a.write_bytes(b"mutated bytes A")
    fp3 = pipeline.compute_stage_input_fingerprint(eval_stage, cfg)
    assert fp3 != fp1, "mutating selected checkpoint A must change evaluation input fingerprint"


def test_declared_external_backbone_asset_presence_in_model_eval_inputs(tmp_path: Path, monkeypatch):
    pipeline = load_pipeline()
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import load_registry

    registry = load_registry()
    model = next(m for m in registry["models"] if m["model_id"] == "clip4cir_rn50x4_fullft")
    weight_path = tmp_path / "RN50x4.pt"
    monkeypatch.setenv("WORKBENCH_CLIP4CIR_RN50X4_WEIGHTS", str(weight_path))

    cfg = resolve_config()
    args = SimpleNamespace(checkpoint=None, dataset_root=None)

    # Absent file -> absent from inputs
    inputs_absent = pipeline.model_eval_inputs(model, args, cfg)
    assert weight_path not in inputs_absent, "absent external asset file must not appear in model_eval_inputs"

    # Present file -> appears in inputs
    weight_path.write_bytes(b"pretrained clip backbone weights")
    inputs_present = pipeline.model_eval_inputs(model, args, cfg)
    assert weight_path in inputs_present, "present external asset file must appear in model_eval_inputs"


def test_empty_checkpoint_selection_excludes_all_checkpoint_inputs() -> None:
    pipeline = load_pipeline()
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import checkpoint_path, load_registry

    model = next(m for m in load_registry()["models"] if m["model_id"] == "airknow")
    cfg = resolve_config()
    args = SimpleNamespace(checkpoint=None, checkpoint_ids={"airknow": []}, dataset_root=None)

    inputs = pipeline.model_eval_inputs(model, args, cfg, checkpoint_ids=[])
    checkpoint_paths = [checkpoint_path("airknow", variant, cfg.WORKBENCH_CHECKPOINT_ROOT)
                        for variant in model["checkpoint_variants"]]
    assert not set(checkpoint_paths) & set(inputs), "explicit empty selection must not include any checkpoint paths"


def test_no_checkpoint_selection_includes_every_registered_variant() -> None:
    pipeline = load_pipeline()
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import checkpoint_path, load_registry

    model = next(m for m in load_registry()["models"] if m["model_id"] == "airknow")
    cfg = resolve_config()
    args = SimpleNamespace(checkpoint=None, checkpoint_ids=None, dataset_root=None)

    inputs = pipeline.model_eval_inputs(model, args, cfg)
    checkpoint_paths = [checkpoint_path("airknow", variant, cfg.WORKBENCH_CHECKPOINT_ROOT)
                        for variant in model["checkpoint_variants"]]
    assert set(checkpoint_paths) <= set(inputs), "no-selection regression: every registered checkpoint path must be fingerprinted"


def test_autopilot_checkpoint_selection_controls_resolved_output() -> None:
    pipeline = load_pipeline()
    from workbench.backend.registry import load_registry

    model = next(m for m in load_registry()["models"] if m["model_id"] == "airknow")
    assert model["checkpoint_variants"][0]["checkpoint_id"] == "fiq_n05"
    cfg = real_config()
    args = SimpleNamespace(checkpoint=None, checkpoint_ids={"airknow": ["fiq_n08"]})

    assert pipeline.resolved_checkpoint_id(model, args) == "fiq_n08"
    assert pipeline.model_eval_outputs(model, args, cfg)[0].name == "airknow_fiq_n08_latest_successful.json"
    assert pipeline.resolved_checkpoint_id(model, SimpleNamespace(checkpoint="fiq_n06")) == "fiq_n06"


def test_limn_bundle_selection_resolves_to_bundle_id() -> None:
    pipeline = load_pipeline()
    model = {
        "model_id": "limn",
        "checkpoint_bundles": [{"bundle_id": "limn_bundle", "required_checkpoint_ids": ["dress", "shirt"]}],
        "checkpoint_variants": [{"checkpoint_id": "dress"}, {"checkpoint_id": "shirt"}],
    }

    assert pipeline.resolved_checkpoint_id(
        model, SimpleNamespace(checkpoint=None, checkpoint_ids={"limn": ["limn_bundle"]})
    ) == "limn_bundle"
    assert pipeline.resolved_checkpoint_id(
        model, SimpleNamespace(checkpoint=None, checkpoint_ids={"limn": ["dress", "shirt"]})
    ) == "limn_bundle"


def test_limn_bundle_all_members_selection_resolves_to_bundle_id_and_output_paths() -> None:
    pipeline = load_pipeline()
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import model_by_id

    limn = model_by_id("limn")
    bundle = limn["checkpoint_bundles"][0]
    bundle_id = bundle["bundle_id"]
    member_ids = list(bundle["required_checkpoint_ids"])
    assert member_ids == ["base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"]

    args = SimpleNamespace(
        mode="all",
        model=None,
        selected_model_ids=["limn"],
        dataset_root=None,
        checkpoint=None,
        checkpoint_ids={"limn": member_ids},
        checkpoint_policy="strict_clean",
        checkpoint_overrides={},
        protocol=None,
        top_k=200,
        continue_on_error=True,
        verify_gpu=False,
        sync_sources=False,
        download_checkpoints=False,
        download_auxiliary_assets=False,
        evaluate=True,
        rebuild_index=False,
    )

    resolved = pipeline.resolved_checkpoint_id(limn, args)
    assert resolved == bundle_id

    cfg = resolve_config()
    stages = pipeline.all_stages(args, cfg)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "limn")
    assert eval_stage.output_paths
    assert all(bundle_id in path.name for path in eval_stage.output_paths)
def test_explicitly_empty_checkpoint_selection_refuses_evaluation_stage() -> None:
    pipeline = load_pipeline()
    from workbench.backend.operator_config import resolve_config
    cfg = resolve_config()
    args = SimpleNamespace(
        mode="all",
        model=None,
        selected_model_ids=["csmcir"],
        dataset_root=None,
        checkpoint=None,
        checkpoint_ids={"csmcir": []},
        checkpoint_policy="strict_clean",
        checkpoint_overrides={},
        protocol=None,
        top_k=200,
        continue_on_error=True,
        verify_gpu=False,
        sync_sources=False,
        download_checkpoints=False,
        download_auxiliary_assets=False,
        evaluate=True,
        rebuild_index=False,
    )
    stages = pipeline.all_stages(args, cfg)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "csmcir")
    assert eval_stage.skipped is not None
    assert "refusing evaluation run" in eval_stage.skipped
    assert not eval_stage.commands
    assert eval_stage.evidence is None


def test_multivariant_model_unselected_agrees_with_evaluator() -> None:
    import argparse
    pipeline = load_pipeline()
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import model_by_id
    from workbench.scripts.evaluate_models import candidates

    cfg = resolve_config()
    airknow = model_by_id("airknow")
    assert len(airknow["checkpoint_variants"]) > 1

    args = SimpleNamespace(
        mode="all",
        model=None,
        selected_model_ids=["airknow"],
        dataset_root=None,
        checkpoint=None,
        checkpoint_ids=None,
        checkpoint_policy="strict_clean",
        checkpoint_overrides={},
        protocol=None,
        top_k=200,
        continue_on_error=True,
        verify_gpu=False,
        sync_sources=False,
        download_checkpoints=False,
        download_auxiliary_assets=False,
        evaluate=True,
        rebuild_index=False,
    )
    stages = pipeline.all_stages(args, cfg)
    eval_stage = next(s for s in stages if s.name == "evaluation" and s.model_id == "airknow")
    declared_id = pipeline.resolved_checkpoint_id(airknow, args)

    assert eval_stage.checkpoint_id == declared_id
    assert any(declared_id in p.name for p in eval_stage.output_paths)
    cmd = eval_stage.commands[0]
    ckpt_idx = cmd.index("--checkpoint")
    eval_flag_id = cmd[ckpt_idx + 1]
    assert eval_flag_id == declared_id

    parser_eval = argparse.ArgumentParser()
    parser_eval.add_argument("--model")
    parser_eval.add_argument("--checkpoint", action="append")
    parsed_eval_args = parser_eval.parse_args(["--model", "airknow", "--checkpoint", eval_flag_id])
    eval_candidates = candidates(parsed_eval_args, [airknow])
    assert len(eval_candidates) == 1
    assert eval_candidates[0][1]["checkpoint_id"] == declared_id


def test_sequence_checkpoint_argument_and_resolved_checkpoint_id_string() -> None:
    pipeline = load_pipeline()
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import model_by_id

    cfg = resolve_config()
    airknow = model_by_id("airknow")

    for seq in (["fiq_n05", "fiq_n08"], ("fiq_n05", "fiq_n08"), {"fiq_n05", "fiq_n08"}):
        ns = SimpleNamespace(checkpoint=seq, checkpoint_ids=None)
        expanded = pipeline._checkpoint_ids(ns, "airknow", airknow)
        assert isinstance(expanded, list)
        assert set(expanded) == {"fiq_n05", "fiq_n08"}

        resolved = pipeline.resolved_checkpoint_id(airknow, ns)
        assert isinstance(resolved, str)
        assert resolved in ("fiq_n05", "fiq_n08")
        assert isinstance(pipeline.resolved_checkpoint_file(airknow, ns, cfg), Path)
    ns_custom = SimpleNamespace(checkpoint=["custom_a", "custom_b"], checkpoint_ids=None)
    resolved_custom = pipeline.resolved_checkpoint_id(airknow, ns_custom)
    assert isinstance(resolved_custom, str)
    assert resolved_custom == "custom_a"
    assert isinstance(pipeline.resolved_checkpoint_file(airknow, ns_custom, cfg), Path)
