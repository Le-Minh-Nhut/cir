from __future__ import annotations

from pathlib import Path

import pytest

from workbench.backend import operator_config


def test_environment_overrides_file_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    config_path = tmp_path / "workbench.env"
    config_path.write_text(
        "CIR_REPO_ROOT=repo\nCIR_DATA_ROOT=file-data\nWORKBENCH_BACKEND_PORT=8123\n",
        encoding="utf-8",
    )
    for key in operator_config._KEYS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(operator_config, "_CONFIG_PATH", config_path)
    monkeypatch.setattr(operator_config, "_DEFAULT_REPO_ROOT", tmp_path)
    monkeypatch.setenv("CIR_DATA_ROOT", "environment-data")
    monkeypatch.setenv("WORKBENCH_BACKEND_PORT", "9001")

    resolved = operator_config.resolve_config()

    assert resolved.CIR_REPO_ROOT == repo.resolve()
    assert resolved.CIR_DATA_ROOT == (repo / "environment-data").resolve()
    assert resolved.WORKBENCH_BACKEND_PORT == 9001


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("CIR_DATA_ROOT", "expected KEY=VALUE"),
        ("CIR DATA_ROOT=value", "expected KEY=VALUE"),
        ("UNKNOWN=value", "unknown setting"),
    ],
)
def test_config_parser_rejects_malformed_or_unknown_lines(tmp_path: Path, line: str, message: str) -> None:
    config_path = tmp_path / "workbench.env"
    config_path.write_text(f"{line}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        operator_config._read_env(config_path)
