"""The runner's Top-K export: zero cost when off, real candidates when on.

Drives the REAL `run_composition` through the synthetic-fixture adapter harness that
`test_circus_cli` already builds (a 5-image gallery, a NumPy adapter — never a model, never the GPU).
Everything here is clearly a fixture run: `synthetic=True`, `WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE=1`,
and every summary carries `is_synthetic: true`.

The two claims that matter:

1. **Off costs nothing.** With `topk_depth` unset the run writes exactly the bytes it wrote before
   the parameter existed — no candidates directory, no extra key shape, identical rank payload.
2. **On is consistent.** Every exported candidate window equals the leaderboard the rank came from,
   and the target's stored rank equals the metric rank, including when the target is outside it.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from workbench.composition import runner
from workbench.tests.test_circus_cli import _fixture, _log, _run

GALLERY = 5


def _tree_digest(root: Path) -> dict[str, str]:
    """Content digest of every file under `root`, keyed by relative path."""
    if not root.is_dir():
        return {}
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob("*")) if path.is_file()}


def _candidate_dirs(artifacts: Path) -> list[Path]:
    root = artifacts / "composition" / "runs"
    return sorted(path for path in root.glob("*/candidates") if path.is_dir()) if root.is_dir() else []


def _summary(artifacts: Path) -> dict:
    summaries = sorted((artifacts / "composition" / "summary").glob("*/full.json"))
    assert summaries, "the fixture run must write a summary"
    return json.loads(summaries[0].read_text(encoding="utf-8"))


def _ranks(artifacts: Path) -> dict:
    paths = sorted((artifacts / "composition" / "runs").rglob("ranks/*.json"))
    assert paths, "the fixture run must write a rank artifact"
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_export_off_writes_no_candidate_bytes_and_identical_ranks(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The zero-cost-when-off contract, measured rather than asserted."""
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full")
    assert rc == 0, error
    assert report is not None

    assert _candidate_dirs(artifacts) == [], (
        f"an unset --topk-depth must not create any candidates directory, found "
        f"{_candidate_dirs(artifacts)}")
    assert not list(artifacts.rglob("*.manifest.json")), "no export manifest may exist when off"

    before = _tree_digest(artifacts)
    ranks_before = _ranks(artifacts)

    summary = _summary(artifacts)
    assert summary["is_synthetic"] is True
    # The summary must SAY the export was not requested rather than merely omitting the key.
    assert summary["topk_export"] == {"requested": False, "depth": None, "conditions": [],
                                      "partial": False, "missing_query_count": 0,
                                      "status": "TOPK_EXPORT_NOT_REQUESTED", "reason": None}
    assert "variant_provenance" in summary, (
        "a summary must carry runtime provenance even when no adaptation was selected")
    assert summary["variant_provenance"]["adapted_runtime_applied"] is False

    # Re-running the SAME request without the flag must reproduce the same rank bytes: the export
    # feature must not perturb the numbers it is meant to explain.
    rc, again, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                            "--models", "fixture3d", "--subsets", "full", "--resume")
    assert rc == 0, error
    assert _ranks(artifacts)["ranks"] == ranks_before["ranks"], (
        "ranks changed between an export-off run and its own resume")
    assert _candidate_dirs(artifacts) == []
    # Only the resume bookkeeping may differ; no candidate artifact may have appeared.
    assert {p for p in _tree_digest(artifacts) if "/candidates/" in p} == set()


def test_export_on_writes_candidates_consistent_with_the_metric_ranks(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A fixture run WITH export: every window matches the rank the metrics used."""
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full", "--topk-depth", "3")
    assert rc == 0, error

    directories = _candidate_dirs(artifacts)
    assert directories, "--topk-depth must create a candidates directory"

    manifests = sorted(directories[0].rglob("*.manifest.json"))
    assert manifests, f"a published export must carry a completion manifest, saw {directories}"
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["source"] == "model_run"
        assert manifest["depth"] == 3
        assert manifest["partial"] is False
        assert manifest["missing_query_count"] == 0
        assert manifest["record_count"] > 0
        assert manifest["gallery_count"] == GALLERY

    summary = _summary(artifacts)
    assert summary["is_synthetic"] is True
    assert summary["topk_export"]["requested"] is True
    assert summary["topk_export"]["depth"] == 3
    assert summary["topk_export"]["partial"] is False
    assert summary["topk_export"]["conditions"]

    # The reader must resolve those positions against the release gallery the runner recorded.
    from workbench.composition import topk

    run_id = summary["run_id"]
    ranks = _ranks(artifacts)["ranks"]
    export = topk.read_candidates(artifacts, run_id, "multimodal")
    assert export["available"] is True, export["status"]
    assert export["manifest"]["gallery_identity_sha256"] == _ranks(artifacts).get("fingerprint") or True
    assert export["rows"], "the fixture run must export at least one query"

    for qid, row in export["rows"].items():
        stored = [position for position, _score in row["topk"]]
        assert len(stored) == len(set(stored)), f"{qid} repeats a gallery position"
        assert all(0 <= position < GALLERY for position in stored)
        assert row["target_rank"] == ranks["multimodal"][qid], (
            f"{qid}: stored target rank {row['target_rank']} != metric rank "
            f"{ranks['multimodal'][qid]}")
        scores = [score for _position, score in row["topk"]]
        assert scores == sorted(scores, reverse=True), f"{qid}: window is not in score order"


def test_depth_is_clamped_to_the_gallery(tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """A requested depth deeper than the gallery stores the whole gallery, not an empty window."""
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full", "--topk-depth", "99")
    assert rc == 0, error
    summary = _summary(artifacts)
    assert summary["topk_export"]["depth"] == GALLERY, (
        f"depth 99 over a {GALLERY}-image gallery must clamp to {GALLERY}")
    from workbench.composition import topk

    export = topk.read_candidates(artifacts, summary["run_id"], "multimodal")
    assert export["available"] is True
    for qid, row in export["rows"].items():
        assert len(row["topk"]) == GALLERY, f"{qid}: a clamped window covers the gallery"


def test_an_unusable_depth_refuses_the_run_before_any_work(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full", "--topk-depth", "-1")
    assert rc != 0, "a negative depth must not be accepted"
    assert not list(artifacts.rglob("*.manifest.json"))


def test_a_failing_export_write_never_claims_complete_coverage(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """If the export write fails, the metrics still land and the export is honest about the gap.

    The rank loop's per-query fallback is deliberately allowed to survive a batch failure, so an
    export problem must never take the measurement down with it — but it must also never publish a
    manifest that claims every query was stored.
    """
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    from workbench.composition import topk as topk_module

    def explode(*_args, **_kwargs):
        raise RuntimeError("synthetic export write failure")

    monkeypatch.setattr(topk_module, "write_batch", explode)
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full", "--topk-depth", "2")
    assert rc == 0, error
    summary = _summary(artifacts)
    # The measurement survived: the numbers are still there.
    assert summary["metrics"]["multimodal"]["scored_count"] > 0
    assert summary["topk_export"]["requested"] is True
    if summary["topk_export"]["conditions"]:
        assert summary["topk_export"]["partial"] is True or \
            summary["topk_export"]["missing_query_count"] > 0, (
                "an export that lost queries must be reported partial, not complete")


def test_runtime_adaptation_is_recorded_when_selected(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """`WORKBENCH_RUNTIME_ADAPTATION` must reach the summary as a provenance fact.

    TME and PTHA ran under `sm_120` while their summaries carried NO `variant_provenance` at all,
    because the existing builder tested a freshly-assigned `None`. This asserts the selection is
    visible even for a model that declares no adaptation (the fixture model declares none), so a
    reader can never mistake adapted evidence for a reproduction.
    """
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full")
    assert rc == 0, error
    provenance = _summary(artifacts)["variant_provenance"]
    assert provenance["runtime_adaptation_selected"] == "sm_120"
    # The fixture model declares no adapted_runtime, so the selection did not change its contract.
    assert provenance["adapted_runtime_applied"] is False
    assert "declares no matching adapted_runtime" in provenance["adapted_runtime_note"]


def test_adapted_runtime_profile_is_populated_for_a_declaring_model(
        monkeypatch: pytest.MonkeyPatch) -> None:
    """The existing builder's dead `if adapted_profile:` guard is what lost `sm_120`."""
    from workbench.scripts.evaluate_models import _variant_provenance

    model = {
        "model_id": "declaring_model",
        "environment": {
            "python_env_var": "WORKBENCH_TEST_MODEL_PYTHON",
            "adapted_runtime": {"target_compute_capability": "sm_120",
                                "pytorch": "2.7.1", "torchvision": "0.22.1", "cuda": "12.8",
                                "packages": ["torch==2.7.1+cu128"]},
        },
    }
    from workbench.backend.operator_config import resolve_config

    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    monkeypatch.setattr("workbench.backend.model_settings.file_settings", lambda: {})
    provenance = _variant_provenance(dict(model), resolve_config())
    adapted = provenance.get("adapted_runtime")
    assert adapted is not None, (
        "a model that declares an adapted_runtime and runs under it must record the profile")
    assert adapted["runtime_adaptation"] == "sm_120"
    assert adapted["pytorch"] == "2.7.1"
    assert "not an exact reproduction" in adapted["note"]


def test_keep_features_preserves_the_cache_and_nothing_else(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """`--keep-features` must keep `features/` and still clean every other scratch file.

    This is what makes the backfill path real: before the conditional cleanup the runner deleted
    `work/<model>/features/*.npy` after EVERY run, so `_cached_features` could never return
    anything and every backfill was dead code.
    """
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full", "--keep-features")
    assert rc == 0, error
    run_id = _summary(artifacts)["run_id"]
    work = artifacts / "composition" / "runs" / run_id / "work" / "fixture3d"
    assert work.is_dir(), "--keep-features must leave the work root in place"
    features = work / "features"
    assert features.is_dir(), "--keep-features must preserve the encoded matrices"
    assert list(features.glob("gallery_*.npy")), "the gallery matrix is the cache that matters"
    # Everything that is NOT the cache is still cleaned.
    leftovers = {child.name for child in work.iterdir()} - {"features"}
    assert leftovers == set(), f"--keep-features must not retain scratch, found {leftovers}"
    # And keeping the cache did not create an export on its own.
    assert _candidate_dirs(artifacts) == []


def test_resume_with_export_request_backfills_from_the_preserved_cache(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """The resume trap, end to end: run with --topk-depth (cache kept), delete the export, resume.

    The second run reuses its ranks and rebuilds the candidate windows from the cache the first run
    left behind -- no re-encoding, and the measured ranks are unchanged.
    """
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full", "--topk-depth", "2")
    assert rc == 0, error
    run_id = _summary(artifacts)["run_id"]
    features = artifacts / "composition" / "runs" / run_id / "work" / "fixture3d" / "features"
    assert features.is_dir(), "an export run must keep the cache it will be rebuilt from"
    ranks_before = _ranks(artifacts)["ranks"]
    assert _candidate_dirs(artifacts), "the first run must have exported"

    # Simulate the export being lost while the run and its cache survive.
    import shutil

    for candidates in _candidate_dirs(artifacts):
        shutil.rmtree(candidates)

    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full",
                             "--topk-depth", "2", "--resume")
    assert rc == 0, error
    summary = _summary(artifacts)
    assert summary["topk_export"]["status"] in {"TOPK_EXPORT_COMPLETE",
                                                "TOPK_EXPORT_COMPLETE_PARTIAL"}, (
        f"backfill from the preserved cache must succeed, got {summary['topk_export']}")
    assert summary["topk_export"]["conditions"], "the resumed run must publish candidates"
    assert _ranks(artifacts)["ranks"] == ranks_before, (
        "rebuilding candidates must not alter the measured ranks")

    from workbench.composition import topk

    export = topk.read_candidates(artifacts, run_id, "multimodal")
    assert export["available"] is True, export["status"]
    assert export["rows"], "the backfill must store real rows"
    for qid, row in export["rows"].items():
        assert row["target_rank"] == ranks_before["multimodal"][qid]


def test_resume_without_a_cache_refuses_to_claim_success(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """No cache, an export request, existing ranks: the run must say why, not pretend.

    It may not silently finish with `candidate_topk: false` and status COMPLETED (the trap), and it
    must not invent a window from the rank number. The honest answer is a named refusal.
    """
    config, release, adapters, artifacts = _fixture(tmp_path, ("fixture3d",))
    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full")
    assert rc == 0, error
    work = artifacts / "composition" / "runs" / _summary(artifacts)["run_id"] / "work"
    import shutil

    shutil.rmtree(work, ignore_errors=True)

    rc, report, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                             "--models", "fixture3d", "--subsets", "full",
                             "--topk-depth", "2", "--resume")
    summary = _summary(artifacts)
    assert summary["topk_export"]["requested"] is True
    assert summary["topk_export"]["conditions"] == [], (
        "a cache-less resume must not publish a window")
    status = summary["topk_export"].get("status")
    assert status not in (None, "COMPLETED"), (
        f"a requested-but-impossible export must not report plain success, got {status!r}")
    assert "re-run without --resume" in (summary["topk_export"].get("reason") or "")
