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
    assert pipeline.run_stages(stages, dry_run=False, continue_on_error=False) == 1
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
