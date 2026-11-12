"""One ordered gallery must have ONE `gallery_identity_sha256`, in every branch that prints it.

The defect this file pins: `workbench/scripts/composition_eval.py` hashed
`{"protocol_id", "image_ids"}` (`0bfd43b9…` on this release) in the plan branch and in the
no-runnable-spec (blocked) branch, while `runner._gallery_fingerprint` hashed
`{"protocol_id", "ordered_gallery", "candidate_names"}` (`1db68c59…`) for the same 74,381-image
gallery. "Did this plan produce this run" was therefore unanswerable by comparing the two fields.

The acceptance check is stronger than "two sites agree", so all three states are built in one test
against one fixture release:

  * plan      — `--plan` (the `_plan` branch);
  * blocked   — every selected model unrunnable (the `if specs else {...}` branch);
  * completed — a real run, whose summary records the runner's fingerprint.

Then the gallery ORDER changes, and the digest must change with it: a constant would satisfy
parity and prove nothing. Order is not incidental here — a gallery position indexes into this list.

Ordering, pinned explicitly (nothing relies on incidental dict order):

  * `ordered_gallery` keeps the loader's release order ON PURPOSE (positions index into it);
  * `candidate_names` is `dict(sorted(...))`;
  * `_digest` serialises with `sort_keys=True`, so no payload KEY order is significant.

A completed run's summary digest is re-verified on read against the manifest it came with
(`backend/composition.py:_candidate_reader`), which is why the fix changes CALL SITES only and
never the fingerprint payload: artifacts written before the change keep verifying.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from workbench.composition import data as composition_data
from workbench.composition.runner import _gallery_fingerprint
from workbench.tests.test_circus_cli import _fixture, _run

#: The digest this release's gallery had before the fix, kept as the negative reference: no branch
#: may print it again, and no run ever recorded it.
LEGACY_PLAN_DIGEST = "0bfd43b9cc0162885f06c60cfc751602ee9f3bed2fb4d206ffcebe3f7b443462"

def _summary_path(report: dict) -> Path:
    return Path(report["models"][0]["summaries"]["full"])

def _reload(release: Path):
    return composition_data.load_release(release, verify_hashes=False, expect_released_counts=False)

def test_plan_blocked_and_completed_publish_one_digest_per_ordered_gallery(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d", "notready"))

    rc, plan, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                           "--plan", "--models", "selected")
    assert rc == 0, error
    rc, blocked, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                              "--models", "notready")
    assert rc == 1, error
    # `run_count` is recomputed after the blocked rows are appended, so the branch is identified by
    # the rows themselves: nothing ran, so no row carries a summary path.
    assert blocked["models"] and all(row["status"] == "SELECTED_BUT_BLOCKED"
                                     for row in blocked["models"]), blocked["models"]
    assert not any(row.get("summaries") for row in blocked["models"]), (
        "the `if specs else` report must be reached with no runnable spec")
    rc, completed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                "--models", "fixture3d")
    assert rc == 0, error
    summary = json.loads(_summary_path(completed).read_text(encoding="utf-8"))

    data = _reload(release)
    expected = _gallery_fingerprint(data)
    digests = {"plan": plan["gallery_identity_sha256"],
               "blocked": blocked["gallery_identity_sha256"],
               "completed": summary["gallery_identity_sha256"]}
    assert set(digests.values()) == {expected}, (
        f"one ordered gallery must have one digest, got {digests} (runner says {expected})")
    assert LEGACY_PLAN_DIGEST not in digests.values(), (
        "the pre-fix plan formula must not survive in any branch; it is what made the plan digest "
        "and the run digest incomparable")
    # The gallery is the same gallery in all three: parity is not a coincidence of two empty sets.
    assert plan["gallery_count"] == summary["gallery_count"] == len(data.gallery.image_ids)
    assert summary["gallery_count"] == len(data.gallery.image_ids) > 0

    # Stability inside ONE process: the digest is a function of the loaded release, not of dict
    # ordering (candidate_names is sorted, keys are serialised with sort_keys=True).
    assert _gallery_fingerprint(data) == expected
    rc, plan_again, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                 "--plan", "--models", "selected")
    assert rc == 0, error
    assert plan_again["gallery_identity_sha256"] == expected

def test_reordering_the_gallery_changes_the_digest_in_every_branch(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The negative: the digest is order-sensitive, so parity above is not a constant."""
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, before, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--plan", "--models", "selected")
    assert rc == 0, error
    ordered = before["gallery_identity_sha256"]
    assert ordered == _gallery_fingerprint(_reload(release))

    audit = release / "shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json"
    payload = json.loads(audit.read_text(encoding="utf-8"))
    assert len(payload["gallery_paths"]) > 1, "the fixture must have more than one gallery image"
    payload["gallery_paths"] = list(reversed(payload["gallery_paths"]))
    audit.write_text(json.dumps(payload), encoding="utf-8")

    reordered = _gallery_fingerprint(_reload(release))
    assert reordered != ordered, (
        "the digest must cover the gallery ORDER: positions index into this list, so two orderings "
        "are two different galleries")
    rc, after, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                            "--plan", "--models", "selected")
    assert rc == 0, error
    assert after["gallery_identity_sha256"] == reordered
    assert after["gallery_count"] == before["gallery_count"], (
        "the same IMAGES in a different order: the count cannot move, only the digest")

def test_the_fingerprint_payload_is_unchanged_so_recorded_artifacts_still_verify(tmp_path: Path) -> None:
    """Payload frozen: only the CALL SITES changed, so pre-fix artifacts keep verifying.

    `backend/composition.py:_candidate_reader` recomputes this digest on read and refuses positions
    recorded against a different gallery. A payload change would have made every existing artifact
    unreadable, which is the reason the fix reuses the helper instead of redefining it.
    """
    config, release, adapters, artifacts = _fixture(tmp_path / "payload", ("fixture3d",))
    data = _reload(release)
    import hashlib

    recorded = hashlib.sha256(json.dumps(
        {"protocol_id": data.gallery.protocol_id,
         "ordered_gallery": list(data.gallery.image_ids),
         "candidate_names": dict(sorted(data.candidate_names.items()))},
        sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()
    assert _gallery_fingerprint(data) == recorded
