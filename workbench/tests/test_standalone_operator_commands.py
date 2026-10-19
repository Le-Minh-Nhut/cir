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


def test_prepare_model_executes_only_declared_recipe_steps(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """`--execute` runs the steps a contract declares and reports the ones it cannot.

    DCNet declares a resize it can perform and a GloVe pickle generation it must not attempt (the
    author's generator samples unseeded vectors, so regenerating produces a different artifact).

    The dataset root is staged HERE rather than taken from the configured default: without an
    explicit root the command resolves the operator's own FashionIQ tree, which a fresh clone does
    not have, so the test would fail for a reason that says nothing about the recipe steps.
    """
    prep_mod = load_script("prepare_model")
    # The full DCNet layout the contract declares, so the run reaches the RECIPE question this test
    # is about. A partial root would fail on missing inputs first and the recipe assertions would
    # never execute -- which is the failure mode the previous version had on a clean checkout.
    dataset = tmp_path / "FashionIQ"
    (dataset / "images").mkdir(parents=True)
    (dataset / "image_splits").mkdir(parents=True)
    (dataset / "captions").mkdir(parents=True)
    (dataset / "resized_images").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (dataset / "image_splits" / f"split.{category}.val.json").write_text("[]", encoding="utf-8")
        (dataset / "captions" / f"cap.{category}.val.json").write_text("[]", encoding="utf-8")
        # The TRAIN captions are part of the declared contract for this layout, and the step under
        # test is only reached once the raw inputs are complete. Missing them reported BLOCKED with
        # a raw-input list instead of the recipe lines this test asserts.
        (dataset / "captions" / f"cap.{category}.train.json").write_text("[]", encoding="utf-8")
        (dataset / "captions" / f"cap.{category}.glove.val.pkl").write_bytes(b"author-artifact")
    rc = prep_mod.main(["--model", "dcnet", "--execute", "--allow-preparation", "--dry-run",
                        "--dataset-root", str(dataset)])
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
