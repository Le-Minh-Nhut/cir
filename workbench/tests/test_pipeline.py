from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


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


def test_mock_dry_run_orders_all_stages_without_runner(monkeypatch, capsys) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["mock", "--dry-run"]) == 0

    assert stage_lines(capsys.readouterr().out) == [
        "[1/5] doctor",
        "[2/5] load mock",
        "[3/5] validate",
        "[4/5] rebuild",
        "[5/5] serve",
    ]
    assert called == []


def test_real_dry_run_orders_full_plan_without_heavy_actions(monkeypatch, capsys) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real", "--dry-run"]) == 0

    assert stage_lines(capsys.readouterr().out) == [
        "[1/8] doctor",
        "[2/8] dataset",
        "[3/8] sync",
        "[4/8] layout",
        "[5/8] checkpoint",
        "[6/8] evaluation",
        "[7/8] validation-index",
        "[8/8] serve",
    ]
    assert called == []


def test_bare_real_runs_only_doctor_without_explicit_actions(monkeypatch) -> None:
    pipeline = load_pipeline()
    called: list[list[str]] = []
    monkeypatch.setattr(pipeline, "run_command", lambda argv: called.append(argv) or 0)

    assert pipeline.main(["real"]) == 0

    assert [Path(argv[1]).name for argv in called] == ["doctor.py"]
