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
    # The declared set is what makes a checkout an approved variant; this one is now complete.
    for remaining in _csmcir()["approved_source_patches"][1:]:
        destination = target / remaining["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "workbench/third_party/CSMCIR" / remaining["path"], destination)
        applied = subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                                  str(REPO / remaining["patch_file"])], capture_output=True, text=True)
        assert applied.returncode == 0, applied.stderr
    state = approved_patch_state(_csmcir(), target)
    assert state["applied"] is True
    assert all(entry["applied"] for entry in state["patches"])


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


def _stage_whole_patch_set(target: Path) -> None:
    """Every declared patch, applied in declaration order -- the definition of a repaired checkout."""
    for patch in _csmcir()["approved_source_patches"]:
        destination = target / patch["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "workbench/third_party/CSMCIR" / patch["path"], destination)
    for patch in _csmcir()["approved_source_patches"]:
        subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                        str(REPO / patch["patch_file"])], check=True, capture_output=True)


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

    # Now place a checkout holding the approved delta -- the whole set, because a checkout repaired
    # by only its first patch is not the approved variant and must still be reported as unrepaired.
    _stage_whole_patch_set(target)
    repaired = blockers_for_checkout()
    assert not any("source cannot run" in b and "lavis" in b for b in repaired), repaired


def test_a_patched_checkout_is_not_reported_as_a_dirty_source(tmp_path, monkeypatch):
    """Applying the approved patch must not make the checkout read as an unauthorized modification.

    This is the workflow the documentation gives an operator, and it did not work: the patch target is
    a tracked edit, and the classifier sent every tracked edit straight to `tracked` without consulting
    the declared-artifact allowlist, so the approved variant was reported dirty.

    The fixture holds only the file the patch touches. A full copy would also carry the staged caption
    aliases, whose verification needs an artifacts-root manifest this test does not build, and that is
    a different subject.
    """
    import hashlib
    import subprocess
    from dataclasses import replace

    from workbench.backend import runtime

    patch = _declared()
    target = tmp_path / "CSMCIR"
    (target / Path(patch["path"]).parent).mkdir(parents=True)
    shutil.copyfile(REPO / "workbench/third_party/CSMCIR" / patch["path"], target / patch["path"])
    # A git checkout is what the classifier reads, so the fixture must be one.
    subprocess.run(["git", "init", "-q"], cwd=target, check=True)
    subprocess.run(["git", "add", "-A"], cwd=target, check=True)
    subprocess.run(["git", "-c", "user.email=t@t.invalid", "-c", "user.name=T",
                    "commit", "-qm", "pin"], cwd=target, check=True)

    monkeypatch.setattr("workbench.backend.operator_config.resolve_config",
                        lambda: replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=tmp_path))

    before = runtime.source_status(target)
    assert runtime._source_dirty(target) is False, before
    assert runtime._approved_source_artifacts(target) != {}, "the allowlist must match this checkout"

    result = subprocess.run(["patch", "-p1", "--directory", str(target), "-i",
                             str(REPO / patch["patch_file"])], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert hashlib.sha256((target / patch["path"]).read_bytes()).hexdigest() == patch["derived_sha256"]

    after = runtime.source_status(target)
    assert after["tracked"] == [], after
    assert patch["path"] in after["approved"], after
    assert runtime._source_dirty(target) is False, after

    # A different file in the same checkout is still an unauthorized change.
    other = target / "src/another_module.py"
    other.write_text("x = 1\n")
    third = runtime.source_status(target)
    assert "src/another_module.py" in third["unauthorized"], third
    assert runtime._source_dirty(target) is True, third
