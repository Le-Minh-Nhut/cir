from __future__ import annotations

from pathlib import Path

from workbench.scripts.rebuild_index import main as rebuild_main
from workbench.scripts.validate_results import validate_file, validate_root
from workbench.tests.mock_data import build_mock_runs


def test_validator_accepts_valid_mock(tmp_path: Path) -> None:
    artifact = tmp_path / "mock.json"
    artifact.write_text(build_mock_runs()[0].model_dump_json())

    assert validate_file(artifact)
    assert validate_root(tmp_path)


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
