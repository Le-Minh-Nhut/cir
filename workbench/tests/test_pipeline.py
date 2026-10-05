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
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "resolve_config", real_config)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--download-auxiliary-assets"]) == 0
    auxiliary = next(argv for argv in called if Path(argv[1]).name == "download_auxiliary_assets.py")
    assert auxiliary[2:] == ["--all"]


def test_evaluation_does_not_validate_aggregate_only_output(monkeypatch) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "resolve_config", real_config)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--model", "csmcir", "--evaluate"]) == 0
    assert "validate_results.py" not in [Path(argv[1]).name for argv in called]
    assert "rebuild_index.py" not in [Path(argv[1]).name for argv in called]


def test_rebuild_index_stays_explicit(monkeypatch) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "resolve_config", real_config)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--rebuild-index"]) == 0
    assert [Path(argv[1]).name for argv in called][-2:] == ["validate_results.py", "rebuild_index.py"]


def test_runtime_preflight_cannot_continue_into_evaluation(monkeypatch) -> None:
    pipeline = load_pipeline()
    stages = [
        pipeline.run_stage("runtime-preflight", ["doctor.py"]),
        pipeline.run_stage("evaluation", ["evaluate_models.py"]),
    ]
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 2)

    assert pipeline.run_stages(stages, dry_run=False, continue_on_error=True) == 1
    assert called == [["doctor.py"]]


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
