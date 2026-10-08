from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manage_environment_list(capsys: pytest.CaptureFixture[str]) -> None:
    env_mod = load_script("manage_environment")
    rc = env_mod.main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "csmcir" in out
    assert "encoder" in out
    assert "dcnet" in out


def test_manage_environment_create_requires_authorization() -> None:
    env_mod = load_script("manage_environment")
    with pytest.raises(SystemExit):
        env_mod.main(["--model", "csmcir", "--create"])


def test_manage_environment_dry_run_authorized(capsys: pytest.CaptureFixture[str]) -> None:
    env_mod = load_script("manage_environment")
    # For a model without synced source spec, create should be blocked
    rc = env_mod.main(["--model", "csmcir", "--create", "--allow-env-install", "--dry-run"])
    out = capsys.readouterr()
    # Source not synced -> blocks safely without mutating
    assert rc == 1 or "[BLOCKED]" in out.err or "[BLOCKED]" in out.out


def test_prepare_model_list(capsys: pytest.CaptureFixture[str]) -> None:
    prep_mod = load_script("prepare_model")
    rc = prep_mod.main(["--list"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "csmcir" in out
    assert "dcnet" in out


def test_prepare_model_execute_requires_authorization() -> None:
    prep_mod = load_script("prepare_model")
    with pytest.raises(SystemExit):
        prep_mod.main(["--model", "csmcir", "--execute"])


def test_prepare_model_non_csmcir_automated_execution_blocked(capsys: pytest.CaptureFixture[str]) -> None:
    prep_mod = load_script("prepare_model")
    rc = prep_mod.main(["--model", "dcnet", "--execute", "--allow-preparation", "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no verified automated preparation step" in err
