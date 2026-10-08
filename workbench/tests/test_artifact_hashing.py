from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from workbench.backend.registry import (
    checkpoint_missing_paths,
    checkpoint_is_present,
    sha256_file,
)
def test_single_file_hashing(tmp_path: Path) -> None:
    target = tmp_path / "weight.pt"
    target.write_bytes(b"checkpoint bytes")
    digest = sha256_file(target)
    assert isinstance(digest, str)
    assert len(digest) == 64
    # Deterministic
    assert sha256_file(target) == digest


def test_directory_hashing_no_is_a_directory_error(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"lr": 0.001}')
    (run_dir / "trained_model.pth").write_bytes(b"model-weights")

    digest = sha256_file(run_dir, required_files=["config.json", "trained_model.pth"])
    assert isinstance(digest, str)
    assert len(digest) == 64


def test_directory_hash_differs_from_file_hash_with_same_bytes(tmp_path: Path) -> None:
    content = b"same-content"
    single_file = tmp_path / "file.pt"
    single_file.write_bytes(content)

    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    (dir_path / "file.pt").write_bytes(content)

    file_digest = sha256_file(single_file)
    dir_digest = sha256_file(dir_path, required_files=["file.pt"])
    assert file_digest != dir_digest


def test_directory_hashing_scoped_to_required_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"lr": 0.001}')
    (run_dir / "trained_model.pth").write_bytes(b"model-weights")

    digest1 = sha256_file(run_dir, required_files=["config.json", "trained_model.pth"])

    # Adding an unrelated log or timestamp file should NOT change the hash
    (run_dir / "train.log").write_text("epoch 1 done at 12:34:56")
    digest2 = sha256_file(run_dir, required_files=["config.json", "trained_model.pth"])

    assert digest1 == digest2

    # But modifying a required file DOES change the hash
    (run_dir / "config.json").write_text('{"lr": 0.002}')
    digest3 = sha256_file(run_dir, required_files=["config.json", "trained_model.pth"])
    assert digest1 != digest3


def test_directory_checkpoint_missing_member_detected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"lr": 0.001}')
    # trained_model.pth is missing

    checkpoint = {
        "checkpoint_id": "test_ckpt",
        "artifact_type": "run_directory",
        "required_files": ["config.json", "trained_model.pth"],
    }

    missing = checkpoint_missing_paths(run_dir, checkpoint)
    assert len(missing) == 1
    assert missing[0] == run_dir / "trained_model.pth"
    assert not checkpoint_is_present(run_dir, checkpoint)


def test_directory_checkpoint_empty_member_detected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"lr": 0.001}')
    (run_dir / "trained_model.pth").write_bytes(b"")  # empty

    checkpoint = {
        "checkpoint_id": "test_ckpt",
        "artifact_type": "run_directory",
        "required_files": ["config.json", "trained_model.pth"],
    }

    missing = checkpoint_missing_paths(run_dir, checkpoint)
    assert len(missing) == 1
    assert missing[0] == run_dir / "trained_model.pth"


def test_directory_checkpoint_member_checksum_mismatch(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    (run_dir / "config.json").write_text('{"lr": 0.001}')
    (run_dir / "trained_model.pth").write_bytes(b"wrong-bytes")

    checkpoint = {
        "checkpoint_id": "test_ckpt",
        "artifact_type": "run_directory",
        "bundle_members": [
            {"filename": "config.json", "expected_sha256": None},
            {"filename": "trained_model.pth", "expected_sha256": "0" * 64},
        ],
    }

    missing = checkpoint_missing_paths(run_dir, checkpoint)
    assert run_dir / "trained_model.pth" in missing


def test_symlink_member_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_dir"
    run_dir.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("evil")

    symlink = run_dir / "link.txt"
    os.symlink(external, symlink)

    with pytest.raises(ValueError, match="symlink"):
        sha256_file(run_dir, required_files=["link.txt"])
