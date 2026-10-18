"""The delivered vendored checkout is accepted, and a tampered one is not.

Why this test exists: the branch ships the pinned upstream source *inside* this repository with the
nested Git metadata removed, so the old `git status`-based source check cannot answer whether the
checkout still matches its pin — there is no `.git` there to ask. If nothing exercised the
replacement, the two failure modes below would both go unnoticed until a run was accepted (or
refused) for the wrong reason:

* a healthy delivered checkout reported as unverifiable, which refuses every model, and
* a checkout whose bytes drifted after the pin still reported as clean, which is the far worse
  direction: it silently evaluates code nobody reviewed.

The fixtures build a miniature vendored tree with `provenance.json` rather than copying a real one,
so the test states the contract instead of restating whichever checkout happens to be in the tree.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from workbench.backend.vendored_source import (  # noqa: E402
    PROVENANCE_FILENAME,
    live_digest,
    manifest_digest,
    source_clean_and_pinned,
    source_provenance,
)

PIN = "a" * 40


def _vendor(tmp_path: Path, model: dict | None = None) -> tuple[Path, dict]:
    checkout = tmp_path / "Demo"
    checkout.mkdir()
    (checkout / "model.py").write_text("VALUE = 1\n")
    (checkout / "LICENSE").write_text("MIT\n")
    manifest = live_digest(checkout, model)
    (checkout / PROVENANCE_FILENAME).write_text(json.dumps({
        "model_ids": ["demo"],
        "requested_url": "https://example.invalid/demo",
        "resolved_url": "https://example.invalid/demo",
        "upstream_commit": PIN,
        "upstream_commit_date": "2026-01-01T00:00:00+00:00",
        "license_file": "LICENSE",
        "destination": str(checkout),
        "vendored_at": "2026-01-01T00:00:00+00:00",
        "files_written": len(manifest),
        "nested_git_removed": [".git"],
        "preserved_local_files": [],
        "submodules_declared": "",
        "git_lfs_pointers": [],
        "file_manifest": manifest,
        "local_fixes": [],
    }) + "\n")
    return checkout, {**(model or {}), "model_id": "demo", "upstream_commit_sha": PIN,
                      "source_dir": "Demo"}


def test_a_pristine_vendored_checkout_is_clean_and_pinned(tmp_path: Path) -> None:
    checkout, model = _vendor(tmp_path)
    ok, commit, reason = source_clean_and_pinned(model, checkout)
    assert (ok, commit, reason) == (True, PIN, None)
    assert source_provenance(model, checkout)["verified"] is True


def test_a_modified_file_is_reported_as_a_source_change(tmp_path: Path) -> None:
    """The direction that matters: post-pin drift must never read as clean."""
    checkout, model = _vendor(tmp_path)
    (checkout / "model.py").write_text("VALUE = 2\n")  # same name, different bytes
    ok, _, reason = source_clean_and_pinned(model, checkout)
    assert ok is False
    assert "model.py" in reason


def test_a_deleted_file_is_reported(tmp_path: Path) -> None:
    checkout, model = _vendor(tmp_path)
    (checkout / "model.py").unlink()
    ok, _, reason = source_clean_and_pinned(model, checkout)
    assert ok is False
    assert "model.py" in reason


def test_an_undeclared_added_file_is_reported(tmp_path: Path) -> None:
    """A stray file in a committed checkout would ship with the branch, so it is not ignored."""
    checkout, model = _vendor(tmp_path)
    (checkout / "extra.py").write_text("import os\n")
    ok, _, reason = source_clean_and_pinned(model, checkout)
    assert ok is False
    assert "extra.py" in reason


def test_a_recorded_local_fix_is_approved_and_a_new_edit_to_it_still_is_not(tmp_path: Path) -> None:
    """A reviewed patch is legitimate; the same file edited again afterwards is not."""
    checkout, model = _vendor(tmp_path)
    declared = "start\n"
    (checkout / "patched.py").write_text(declared)
    record = json.loads((checkout / PROVENANCE_FILENAME).read_text())
    record["local_fixes"] = [{"path": "patched.py", "origin": "review",
                              "upstream_sha256": None,
                              "delivered_sha256": hashlib.sha256(declared.encode()).hexdigest()}]
    (checkout / PROVENANCE_FILENAME).write_text(json.dumps(record) + "\n")

    ok, _, reason = source_clean_and_pinned(model, checkout)
    assert (ok, reason) == (True, None), "a file the record declares as a fix must be approved"

    (checkout / "patched.py").write_text("edited again\n")
    ok, _, reason = source_clean_and_pinned(model, checkout)
    assert ok is False, "an edit after the fix was recorded is unreviewed content"


def test_a_declared_runtime_artifact_is_not_a_source_change(tmp_path: Path) -> None:
    """Preparation creates the canonical dataset link; that is not tampering."""
    model = {"model_id": "demo", "upstream_commit_sha": PIN, "source_dir": "Demo",
             "canonical_dataset_link": "fashionIQ_dataset"}
    checkout, model = _vendor(tmp_path, model)
    target = tmp_path / "canonical"
    target.mkdir()
    (checkout / "fashionIQ_dataset").symlink_to(target, target_is_directory=True)
    ok, _, reason = source_clean_and_pinned(model, checkout)
    assert (ok, reason) == (True, None)


def test_a_checkout_still_carrying_nested_git_is_refused(tmp_path: Path) -> None:
    """Vendoring exists to remove that metadata; leaving it means the path reported as a pin is
    whatever a nested repository happens to point at."""
    checkout, model = _vendor(tmp_path)
    (checkout / ".git").mkdir()
    (checkout / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    provenance = source_provenance(model, checkout)
    assert provenance["verified"] is False
    assert provenance["state"] == "vendored_with_nested_git"


def test_pin_mismatch_is_refused(tmp_path: Path) -> None:
    checkout, model = _vendor(tmp_path)
    record = json.loads((checkout / PROVENANCE_FILENAME).read_text())
    record["upstream_commit"] = "b" * 40
    (checkout / PROVENANCE_FILENAME).write_text(json.dumps(record) + "\n")
    ok, commit, reason = source_clean_and_pinned(model, checkout)
    assert ok is False
    assert commit == "b" * 40
    assert "pin mismatch" in reason


def test_missing_provenance_is_not_silently_trusted(tmp_path: Path) -> None:
    checkout = tmp_path / "Demo"
    checkout.mkdir()
    (checkout / "model.py").write_text("VALUE = 1\n")
    ok, _, reason = source_clean_and_pinned({"upstream_commit_sha": PIN}, checkout)
    assert ok is False
    assert PROVENANCE_FILENAME in reason


def test_manifest_digest_is_order_independent(tmp_path: Path) -> None:
    """The aggregate is what a run records; it must not depend on directory iteration order."""
    assert manifest_digest({"a": "1", "b": "2"}) == manifest_digest({"b": "2", "a": "1"})
    assert manifest_digest({"a": "1"}) != manifest_digest({"a": "2"})


@pytest.mark.parametrize("model_id", ["limn", "csmcir", "ptha_mtst"])
def test_the_delivered_checkouts_verify_against_their_recorded_pins(model_id: str) -> None:
    """Guards the delivery itself: the committed trees must satisfy the contract they declare."""
    from workbench.backend.registry import load_registry

    model = next(m for m in load_registry()["models"] if m["model_id"] == model_id)
    checkout = REPO / "workbench" / "third_party" / model["source_dir"]
    if not (checkout / PROVENANCE_FILENAME).is_file():
        pytest.skip(f"{model_id} checkout not present")
    ok, commit, reason = source_clean_and_pinned(model, checkout)
    assert ok is True, reason
    assert commit == model["upstream_commit_sha"]
