"""The vendored-source contract must honour the registry's approved patch, and nothing looser.

Background: CSMCIR's pinned revision cannot run as published — 98 unguarded imports of modules the
author never committed sit *before* the line that registers the class its evaluator asks for. The
registry therefore declares exactly one approved patch, with the published base digest and the
recorded post-patch digest, and the nested-Git path already accepted that delta.

The vendored layout did not. Applying the registry's own documented recipe left the file hashing to
`derived_sha256` while `source_provenance` still reported it as an unauthorized modification, so the
one supported way to run CSMCIR failed its own integrity check. These tests pin the corrected
behaviour in both directions: the approved delta is accepted byte-for-byte, and everything else —
an extra edit after the patch, a digest that matches neither side, an unrelated file — is refused.

They also assert something the two entry points had drifted on before: `runtime.source_clean_and_pinned`
and `replay.common.source_checkout_status` answer the *same* question with the same answer.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend.registry import model_by_id  # noqa: E402
from workbench.backend.runtime import source_clean_and_pinned  # noqa: E402
from workbench.backend.vendored_source import source_provenance  # noqa: E402
from workbench.replay.common import source_checkout_status  # noqa: E402

CSMCIR = REPO / "workbench" / "third_party" / "CSMCIR"
PATCH = REPO / "workbench" / "patches" / "csmcir_models_init_compat.patch"
TARGET = "src/lavis/models/__init__.py"

pytestmark = pytest.mark.skipif(
    not (CSMCIR / "provenance.json").is_file(),
    reason="vendored CSMCIR checkout not present")


def _model():
    return model_by_id("csmcir")


def _contract():
    patches = _model().get("approved_source_patches") or []
    assert patches, "the registry must declare the CSMCIR approved patch"
    return patches[0]


def _copy(tmp_path: Path) -> Path:
    root = tmp_path / "CSMCIR"
    shutil.copytree(CSMCIR, root)
    return root


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_registry_declaration_matches_the_shipped_patch_file(tmp_path: Path) -> None:
    """The patch file's own digest must be the one the registry approves."""
    contract = _contract()
    assert PATCH.is_file(), f"declared patch file is missing: {PATCH}"
    assert _sha(PATCH) == contract["patch_sha256"], "the shipped patch is not the approved one"


def test_pristine_source_is_accepted(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    assert _sha(root / TARGET) == _contract()["base_sha256"], (
        "the vendored file is not the published base any more; refresh the registry contract")
    assert source_provenance(_model(), root)["verified"] is True


def test_exact_approved_patch_is_accepted(tmp_path: Path) -> None:
    """Applying the declared patch must produce an APPROVED checkout, not a rejected one."""
    import subprocess

    root = _copy(tmp_path)
    result = subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=PATCH.open("rb"),
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert _sha(root / TARGET) == _contract()["derived_sha256"]

    provenance = source_provenance(_model(), root)
    assert provenance["verified"] is True, provenance["error"]
    assert provenance["error"] is None
    assert TARGET in provenance["approved_runtime_artifacts"]
    ok, _, reason = source_clean_and_pinned(_model(), root)
    assert (ok, reason) == (True, None)


def test_an_edit_after_the_approved_patch_is_rejected(tmp_path: Path) -> None:
    """Approval covers the declared delta only, not whatever is written next to it."""
    import subprocess

    root = _copy(tmp_path)
    subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=PATCH.open("rb"), check=True,
                   capture_output=True)
    (root / TARGET).write_text((root / TARGET).read_text() + "\n# an extra edit\n")

    provenance = source_provenance(_model(), root)
    assert provenance["verified"] is False
    assert "approved-patch" in provenance["error"]


def test_a_wrong_patch_digest_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """A declared patch whose recorded digest does not describe the file must not approve it."""
    import subprocess
    from workbench.backend import registry as registry_module

    root = _copy(tmp_path)
    subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=PATCH.open("rb"), check=True,
                   capture_output=True)

    original = registry_module.auxiliary_assets_for_model  # noqa: F841 - keeps the import honest
    real_model = model_by_id("csmcir")
    wrong = {**real_model,
             "approved_source_patches": [{**_contract(), "derived_sha256": "0" * 64}]}
    monkeypatch.setattr(registry_module, "model_by_id", lambda model_id: wrong if model_id == "csmcir" else real_model)

    provenance = source_provenance(wrong, root)
    assert provenance["verified"] is False
    assert "approved-patch" in provenance["error"]


def test_a_missing_patched_file_is_rejected(tmp_path: Path) -> None:
    root = _copy(tmp_path)
    (root / TARGET).unlink()
    provenance = source_provenance(_model(), root)
    assert provenance["verified"] is False
    assert TARGET in provenance["error"]


def test_an_unrelated_file_edit_is_still_rejected_in_a_patched_checkout(tmp_path: Path) -> None:
    """The approved delta must not become a blanket licence for the checkout."""
    import subprocess

    root = _copy(tmp_path)
    subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=PATCH.open("rb"), check=True,
                   capture_output=True)
    (root / "src" / "utils_csmcir.py").write_text("# tampered\n")
    provenance = source_provenance(_model(), root)
    assert provenance["verified"] is False
    assert "utils_csmcir.py" in provenance["error"]


@pytest.mark.parametrize("patched", [False, True])
def test_runtime_and_replay_agree_with_and_without_the_patch(tmp_path: Path, patched: bool) -> None:
    """Two entry points, one answer — they had drifted apart before this contract existed."""
    import subprocess

    root = _copy(tmp_path)
    if patched:
        subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=PATCH.open("rb"), check=True,
                       capture_output=True)
    model = _model()
    runtime_ok, _, _ = source_clean_and_pinned(model, root)
    # The replay path is given the MODEL, not a bare commit: the approved-patch declaration lives on
    # the registry entry, so a commit string alone cannot express it.
    replay_ok, _, _ = source_checkout_status(root, model["upstream_commit_sha"], model)
    assert runtime_ok is True and replay_ok is True

    (root / "src" / "utils_csmcir.py").write_text("# tampered\n")
    runtime_ok, _, _ = source_clean_and_pinned(model, root)
    replay_ok, _, reason = source_checkout_status(root, model["upstream_commit_sha"], model)
    assert runtime_ok is False, "the runtime path must refuse a modified checkout"
    assert replay_ok is False, "the replay path must agree with the runtime path"
    assert reason


def test_the_delivered_csmcir_checkout_still_verifies() -> None:
    """Guard the delivery: the committed checkout must satisfy the contract it declares."""
    provenance = source_provenance(_model(), CSMCIR)
    assert provenance["verified"] is True, provenance["error"]


def test_provenance_records_which_paths_are_approved(tmp_path: Path) -> None:
    """The record must say *why* a divergence is acceptable, not merely that it is."""
    import subprocess

    root = _copy(tmp_path)
    record = json.loads((root / "provenance.json").read_text())
    assert record["file_manifest"], "the manifest is the integrity baseline"
    subprocess.run(["patch", "-p1", "--quiet"], cwd=root, stdin=PATCH.open("rb"), check=True,
                   capture_output=True)
    provenance = source_provenance(_model(), root)
    # The patch target is approved, and so are the registry-declared auxiliary assets that live in
    # this checkout (CSMCIR's CoT captions are both a published asset and tracked source). What the
    # record must not do is report an unexplained divergence.
    assert TARGET in provenance["approved_runtime_artifacts"]
    assert provenance["tracked_modifications"] == []
    assert provenance["error"] is None
