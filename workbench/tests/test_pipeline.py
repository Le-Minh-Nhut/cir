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


def test_real_dry_run_orders_full_plan_without_heavy_actions(monkeypatch, capsys) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--dry-run"]) == 0

    assert stage_lines(capsys.readouterr().out) == [
        "[1/8] doctor",
        "[2/8] dataset",
        "[3/8] sync",
        "[4/8] checkpoint",
        "[5/8] auxiliary-assets",
        "[6/8] evaluation",
        "[7/8] validation-index",
        "[8/8] serve",
    ]
    assert called == []


def test_real_auxiliary_stage_uses_workbench_preflight(monkeypatch) -> None:
    pipeline = load_pipeline()
    config = SimpleNamespace(
        FASHIONIQ_ROOT=Path("/configured/FashionIQ"),
        WORKBENCH_THIRD_PARTY_ROOT=Path("/configured/third_party"),
        WORKBENCH_CHECKPOINT_ROOT=Path("/configured/checkpoints"),
        WORKBENCH_RESULTS_ROOT=Path("/configured/results"),
    )
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "resolve_config", lambda: config)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--model", "csmcir", "--download-auxiliary-assets"]) == 0

    assert called[0][2:] == ["--scope", "workbench"]
    assert [Path(argv[1]).name for argv in called] == ["doctor.py", "prepare_dataset.py", "download_auxiliary_assets.py"]

def test_bare_real_prepares_configured_dataset(monkeypatch) -> None:
    pipeline = load_pipeline()
    configured_root = Path("/configured/FashionIQ")
    config = SimpleNamespace(
        FASHIONIQ_ROOT=configured_root,
        WORKBENCH_THIRD_PARTY_ROOT=Path("/configured/third_party"),
        WORKBENCH_CHECKPOINT_ROOT=Path("/configured/checkpoints"),
        WORKBENCH_RESULTS_ROOT=Path("/configured/results"),
    )
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "resolve_config", lambda: config)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real"]) == 0

    assert [Path(argv[1]).name for argv in called] == ["doctor.py", "prepare_dataset.py"]
    assert called[1][2:] == ["--dataset-root", str(configured_root), "--check-only"]


def test_real_dataset_option_overrides_config(monkeypatch) -> None:
    pipeline = load_pipeline()
    config = SimpleNamespace(
        FASHIONIQ_ROOT=Path("/configured/FashionIQ"),
        WORKBENCH_THIRD_PARTY_ROOT=Path("/configured/third_party"),
        WORKBENCH_CHECKPOINT_ROOT=Path("/configured/checkpoints"),
        WORKBENCH_RESULTS_ROOT=Path("/configured/results"),
    )
    requested_root = Path("/requested/FashionIQ")
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "resolve_config", lambda: config)
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--dataset-root", str(requested_root)]) == 0

    assert called[1][2:] == ["--dataset-root", str(requested_root), "--check-only"]
