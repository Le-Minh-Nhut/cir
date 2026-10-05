from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from workbench.scripts import load_mock_results, rebuild_index, validate_results
from workbench.scripts.rebuild_index import main as rebuild_main
from workbench.scripts.validate_results import validate_file, validate_root
from workbench.tests.mock_data import build_mock_runs


def test_validator_accepts_valid_mock(tmp_path: Path) -> None:
    artifact = tmp_path / "mock.json"
    artifact.write_text(build_mock_runs()[0].model_dump_json())

    assert validate_file(artifact)
    assert validate_root(tmp_path)



def test_mock_loader_uses_configured_results_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "configured-results"
    monkeypatch.setattr(load_mock_results, "resolve_config", lambda: SimpleNamespace(WORKBENCH_RESULTS_ROOT=root))
    monkeypatch.setattr("sys.argv", ["load_mock_results.py"])

    load_mock_results.main()

    assert (root / "fashioniq_original_split" / "mock" / "mock-habit-fiq_n02.json").is_file()


def test_validator_all_uses_configured_results_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "configured-results"
    seen: list[Path] = []
    monkeypatch.setattr(validate_results, "resolve_config", lambda: SimpleNamespace(WORKBENCH_RESULTS_ROOT=root))
    monkeypatch.setattr(validate_results, "validate_root", lambda path, strict_real: seen.append(path) or True)

    assert validate_results.main(["--all"]) == 0
    assert seen == [root]


def test_rebuilder_uses_configured_results_root(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "configured-results"
    seen: list[Path] = []
    monkeypatch.setattr(rebuild_index, "resolve_config", lambda: SimpleNamespace(WORKBENCH_RESULTS_ROOT=root))
    monkeypatch.setattr(rebuild_index, "validate_root", lambda path: seen.append(path) or True)

    assert rebuild_index.main(["--check-only"]) == 0
    assert seen == [root]

def test_validator_rejects_malformed_json(tmp_path: Path) -> None:
    artifact = tmp_path / "broken.json"
    artifact.write_text("{")

    assert not validate_file(artifact)


def test_check_only_does_not_create_database(tmp_path: Path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    (root / "mock.json").write_text(build_mock_runs()[0].model_dump_json())
    database = tmp_path / "workbench.duckdb"

    assert rebuild_main(["--results-root", str(root), "--database", str(database), "--check-only"]) == 0
    assert not database.exists()
