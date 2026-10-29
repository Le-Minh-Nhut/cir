"""A resume must not retry a deterministic per-query failure forever.

A recorded PARTIAL used to be retried whenever any per-query reason was not a category-scope
failure, so a run blocked by a deterministic condition (missing reference image, unresolvable
positive target, non-finite query features, invalid exact rank) re-encoded the whole gallery on
every resume. These checks pin the marker set and the end-to-end no-retry behavior; the CLI
fixture is reused from ``test_circus_cli`` rather than duplicated.
"""
from __future__ import annotations

import pytest

from workbench.composition import runner
from workbench.tests.test_circus_cli import _fixture, _log, _run

STABLE = (
    "limn category scope failure; batch_indices=(0,); qids=['q0']; limn category_scope_invalid",
    "query q3: category 'shirt' has no adapter embedding scope for query features",
    "no positive target resolved in exact released gallery",
    "query features unavailable",
    "invalid full-gallery exact rank: 0",
    "query reference image is missing",
)


@pytest.mark.parametrize("reason", STABLE)
def test_stable_failure_markers_are_recognised(reason: str) -> None:
    assert runner._is_stable_failure(reason)


def test_transient_reason_is_not_stable() -> None:
    assert not runner._is_stable_failure("RuntimeError: synthetic transient encoding failure")
    assert not runner._is_stable_failure("AdapterWorkerError: adapter_worker_exited: model=tme")


def test_category_scope_failure_resume_keeps_partial_without_reencoding(
        tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    config, release, adapters, artifacts = _fixture(tmp_path, ("scoped",))
    rc, first, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                            "--models", "scoped", "--resume")
    assert rc == 1, error
    assert first["models"][0]["status"] == "PARTIAL"
    encode_count = len([entry for entry in _log(tmp_path / "encodes.jsonl") if entry["kind"] != "checkpoint"])
    for _ in range(2):
        rc, resumed, error = _run(config, release, adapters, artifacts, monkeypatch, capsys,
                                  "--models", "scoped", "--resume")
        assert rc == 1, error
        assert resumed["models"][0]["status"] == "PARTIAL"
        assert len([entry for entry in _log(tmp_path / "encodes.jsonl")
                    if entry["kind"] != "checkpoint"]) == encode_count
