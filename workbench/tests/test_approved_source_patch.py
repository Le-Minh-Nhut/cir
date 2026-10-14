"""Tests for the approved-patch source overlay.

An upstream revision can be unusable as published. The workbench then needs a way to run a repaired
checkout without either disabling source integrity or silently editing the pinned tree. The rule these
tests pin is that approval is exact: one declared patch, one recorded base digest, one recorded derived
digest, and anything else still blocks.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import registry, runtime
from workbench.backend.operator_config import resolve_config
from workbench.backend.preflight import approved_patch_state, preflight_blockers
from workbench.backend.registry import load_registry

REPO = Path(__file__).resolve().parents[2]


def _csmcir() -> dict:
    return next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")


def _declared() -> dict:
    return _csmcir()["approved_source_patches"][0]


def _published_file() -> Path:
    return REPO / "workbench/third_party/CSMCIR" / _declared()["path"]


def _fixture(tmp_path: Path) -> Path:
    """A checkout holding only the one file the patch touches, at its published bytes."""
    target = tmp_path / "CSMCIR"
    (target / Path(_declared()["path"]).parent).mkdir(parents=True)
    shutil.copyfile(_published_file(), target / _declared()["path"])
    return target


def test_the_declaration_matches_the_committed_patch_and_the_real_file():
    patch = _declared()
    patch_file = REPO / patch["patch_file"]
    assert patch_file.is_file(), patch["patch_file"]
    assert hashlib.sha256(patch_file.read_bytes()).hexdigest() == patch["patch_sha256"]
    assert hashlib.sha256(_published_file().read_bytes()).hexdigest() == patch["base_sha256"]


def test_an_unpatched_checkout_reports_a_patch_explanation(tmp_path: Path):
    state = approved_patch_state(_csmcir(), _fixture(tmp_path))
    assert state["applied"] is False
    assert "unmodified published file" in state["detail"]


def test_the_approved_patch_makes_the_checkout_an_approved_variant(tmp_path: Path):
    target = _fixture(tmp_path)
    patch = _declared()
    result = subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                             str(REPO / patch["patch_file"])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    digest = hashlib.sha256((target / patch["path"]).read_bytes()).hexdigest()
    assert digest == patch["derived_sha256"]
    state = approved_patch_state(_csmcir(), target)
    assert state["applied"] is True


def test_an_extra_edit_to_the_patched_file_is_not_covered(tmp_path: Path):
    target = _fixture(tmp_path)
    patch = _declared()
    subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                    str(REPO / patch["patch_file"])], check=True, capture_output=True)
    file = target / patch["path"]
    file.write_text(file.read_text() + "\n# an edit no patch approved\n")
    state = approved_patch_state(_csmcir(), target)
    assert state["applied"] is False
    assert "neither the published file nor the approved variant" in state["detail"]


def test_a_missing_checkout_reports_rather_than_assuming(tmp_path: Path):
    state = approved_patch_state(_csmcir(), tmp_path / "absent")
    assert state["applied"] is False
    assert "not present" in state["detail"]


def test_a_model_without_a_declared_patch_has_no_patch_state():
    other = next(m for m in load_registry()["models"] if m["model_id"] == "habit")
    assert approved_patch_state(other, None) is None


def test_the_runtime_accepts_the_derived_file_and_rejects_other_bytes(tmp_path: Path):
    """The provenance classifier must accept exactly the approved delta."""
    target = _fixture(tmp_path)
    patch = _declared()
    subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                    str(REPO / patch["patch_file"])], check=True, capture_output=True)
    declaration = {"kind": "approved_source_patch", "path": patch["path"],
                   "base_sha256": patch["base_sha256"], "derived_sha256": patch["derived_sha256"]}
    file = target / patch["path"]
    assert runtime._verify_declared_artifact(target, patch["path"], declaration) is None

    file.write_text(file.read_text() + "# tamper\n")
    reason = runtime._verify_declared_artifact(target, patch["path"], declaration)
    assert reason and "modified source in an approved-patch checkout" in reason

    shutil.copyfile(_published_file(), file)
    reason = runtime._verify_declared_artifact(target, patch["path"], declaration)
    assert reason and "has not been applied" in reason


def test_the_blocker_is_reported_only_while_the_checkout_is_unrepaired(monkeypatch, tmp_path: Path):
    """A repaired checkout must not be blocked by a claim about its unrepaired original."""
    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=tmp_path)
    target = tmp_path / "CSMCIR"

    def blockers_for_checkout() -> list[str]:
        return preflight_blockers(_csmcir(), config)

    # The checkout does not exist yet: the declared blocker applies.
    monkeypatch.setattr("workbench.backend.runtime.source_clean_and_pinned", lambda *a, **k: (True, "pin", None))
    absent = blockers_for_checkout()
    assert any("source cannot run" in b and "lavis" in b for b in absent), absent

    # Now place a checkout holding the approved delta.
    (target / "src/lavis/models").mkdir(parents=True)
    shutil.copyfile(_published_file(), target / "src/lavis/models/__init__.py")
    subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                    str(REPO / _declared()["patch_file"])], check=True, capture_output=True)
    repaired = blockers_for_checkout()
    assert not any("source cannot run" in b and "lavis" in b for b in repaired), repaired
