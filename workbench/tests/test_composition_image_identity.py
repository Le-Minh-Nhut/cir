"""The composition dataset identity must follow image bytes, not file metadata."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from workbench.composition import runner


def test_image_identity_follows_content_not_mtime(tmp_path: Path) -> None:
    """Same path, same size, restored mtime, different bytes: the identity must change."""
    image = tmp_path / "gallery.jpg"
    image.write_bytes(b"AAAA")
    stat = image.stat()
    atime_ns, mtime_ns = stat.st_atime_ns, stat.st_mtime_ns
    original = runner._image_identity([image])

    image.write_bytes(b"BBBB")
    os.utime(image, ns=(atime_ns, mtime_ns))
    assert image.stat().st_mtime_ns == mtime_ns
    assert image.stat().st_size == stat.st_size
    assert runner._image_identity([image]) != original, "changed bytes must change the identity"

    image.write_bytes(b"AAAA")
    os.utime(image, ns=(atime_ns, mtime_ns))
    assert runner._image_identity([image]) == original, "identical bytes must reproduce the identity"


def test_image_identity_rolls_up_per_file_digests(tmp_path: Path) -> None:
    """The identity carries one rolled-up digest, not a raw 64-char digest per file."""
    first, second = tmp_path / "a.jpg", tmp_path / "b.jpg"
    first.write_bytes(b"AAAA")
    second.write_bytes(b"BBBB")
    identity = runner._image_identity([first, second])

    assert isinstance(identity["content_digest"], str)
    assert not any(isinstance(value, list) and any(isinstance(item, str) and len(item) == 64 for item in value)
                   for value in identity.values()), "raw per-file digests must not be stored"
    assert not any(isinstance(value, list) and len(value) == 2 for value in identity.values())

    first.write_bytes(b"CCCC")
    assert runner._image_identity([first, second])["content_digest"] != identity["content_digest"]


def test_rank_payload_validation() -> None:
    qids = {"q1", "q2"}
    valid = {"condition": {"q1": 1, "q2": None}}
    assert runner._rank_payload_is_valid(valid, ["condition"], qids, 3)
    assert not runner._rank_payload_is_valid({"condition": {"q1": "1", "q2": None}}, ["condition"], qids, 3)
    assert not runner._rank_payload_is_valid({"condition": {"q1": 1}}, ["condition"], qids, 3)
    assert not runner._rank_payload_is_valid({"condition": {"q1": 9999, "q2": None}}, ["condition"], qids, 3)
    assert not runner._rank_payload_is_valid({"condition": {"q1": True, "q2": None}}, ["condition"], qids, 3)


def _take_model_lock(path: str, started, acquired) -> None:
    started.set()
    descriptor = runner._acquire_model_lock(Path(path))
    acquired.set()
    runner._release_model_lock(descriptor)


def test_model_lock_serializes_processes(tmp_path: Path) -> None:
    import multiprocessing

    context = multiprocessing.get_context("fork")
    path = tmp_path / "model.lock"
    first = runner._acquire_model_lock(path)
    started, acquired = context.Event(), context.Event()
    process = context.Process(target=_take_model_lock, args=(str(path), started, acquired))
    process.start()
    try:
        assert started.wait(5)
        assert not acquired.wait(0.2)
    finally:
        runner._release_model_lock(first)
    assert acquired.wait(5)
    process.join(5)
    assert process.exitcode == 0


def test_rank_validation_catches_old_permissive_guard(monkeypatch) -> None:
    monkeypatch.setattr(runner, "_rank_payload_is_valid",
                        lambda ranks, conditions, qids, gallery_count: set(ranks) == set(conditions))
    with pytest.raises(AssertionError):
        assert not runner._rank_payload_is_valid(
            {"condition": {"q1": "bad", "q2": None}}, ["condition"], {"q1", "q2"}, 3)


def test_model_lock_assertion_fails_when_flock_is_noop(tmp_path: Path, monkeypatch) -> None:
    import multiprocessing

    context = multiprocessing.get_context("fork")
    monkeypatch.setattr(runner.fcntl, "flock", lambda *_args: None)
    first = runner._acquire_model_lock(tmp_path / "unlocked.lock")
    started, acquired = context.Event(), context.Event()
    process = context.Process(target=_take_model_lock,
                              args=(str(tmp_path / "unlocked.lock"), started, acquired))
    process.start()
    try:
        assert started.wait(5)
        with pytest.raises(AssertionError):
            assert not acquired.wait(5)
    finally:
        runner._release_model_lock(first)
    process.join(5)
    assert process.exitcode == 0
