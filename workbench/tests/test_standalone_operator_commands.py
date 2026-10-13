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
    rc = env_mod.main(["--model", "csmcir", "--create", "--allow-env-install", "--dry-run"])
    out = capsys.readouterr()
    # Shared environment groups are planned without inspecting unsynced source in dry-run.
    assert rc == 0
    assert "[OK] dry-run: would provision or reuse group" in out.out


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


def test_prepare_model_executes_only_declared_recipe_steps(capsys: pytest.CaptureFixture[str]) -> None:
    """`--execute` runs the steps a contract declares and reports the ones it cannot.

    DCNet declares a resize it can perform and a GloVe pickle generation it must not attempt (the
    author's generator samples unseeded vectors, so regenerating produces a different artifact).
    """
    prep_mod = load_script("prepare_model")
    rc = prep_mod.main(["--model", "dcnet", "--execute", "--allow-preparation", "--dry-run"])
    captured = capsys.readouterr()
    combined = captured.out + captured.err
    # Not the old blanket refusal: DCNet has a real declared step now.
    assert "no verified automated preparation step" not in captured.err
    # DCNet's GloVe PKL generation is nondeterministic, so its author artifact is required
    # while its resize may still be attempted in dry-run.
    assert rc == 0, (captured.out, captured.err)
    assert "[MANUAL]" in captured.out
    assert "NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT" in captured.out
    assert "would resize" in captured.out
    assert "dry-run" in combined, combined
