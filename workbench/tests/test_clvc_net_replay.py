"""Tests for CLVC-Net replay wrapper and preflight validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.replay.clvc_net import (
    EXPECTED_CATEGORY_FILES,
    MODEL,
    PROTOCOL,
    preflight,
    validate_checkpoint,
    validate_dataset,
    validate_source,
)


def test_clvc_net_protocol_is_val_split():
    assert PROTOCOL == "fashioniq_val_split"


def test_clvc_net_requires_six_checkpoint_files():
    assert len(EXPECTED_CATEGORY_FILES) == 3
    for cat in CATEGORIES:
        assert len(EXPECTED_CATEGORY_FILES[cat]) == 2
        assert EXPECTED_CATEGORY_FILES[cat] == (f"{cat}_local_model.pt", f"{cat}_global_model.pt")


def test_clvc_net_preflight_fails_on_empty_paths(tmp_path: Path):
    source_dir = tmp_path / "source"
    ckpt_dir = tmp_path / "checkpoints"
    data_dir = tmp_path / "dataset"
    source_dir.mkdir()
    ckpt_dir.mkdir()
    data_dir.mkdir()

    blockers = preflight(ckpt_dir, data_dir, source_dir)
    assert any("source" in b.lower() for b in blockers)
    assert any("checkpoint" in b.lower() for b in blockers)
    assert any("dataset" in b.lower() for b in blockers)
    # The old contract required a hard-coded "mapping unresolved" blocker on every call, which
    # refused a complete bundle (all six registry digests matching) with a string instead of the
    # bytes. The blocker now names the missing files, and a mapping-status blocker appears only when
    # the files are present and the registry status is not VERIFIED_METADATA.
    assert any("missing or empty" in b.lower() for b in blockers)
    # The replacement contract, both halves: the blocker must come from the BYTES, and a registry
    # that is not VERIFIED_METADATA must still block even when every file is present and hashed.
    assert not any("checkpoint mapping is unresolved" in b.lower() for b in blockers)


def test_clvc_net_preflight_fails_closed_on_a_wrong_digest(tmp_path: Path):
    """Present files with wrong bytes must be refused: a substituted bundle is not a bundle."""
    import hashlib
    from unittest.mock import patch

    from workbench.replay import clvc_net as module

    names = [name for cat in CATEGORIES for name in EXPECTED_CATEGORY_FILES[cat]]
    for name in names:
        (tmp_path / name).write_bytes(f"not {name}".encode())
    variant = dict(module.CHECKPOINT_VARIANT)
    variant["bundle_members"] = [
        {"filename": name, "expected_sha256": hashlib.sha256(b"something else").hexdigest()}
        for name in names
    ]
    with patch.object(module, "CHECKPOINT_VARIANT", variant):
        blockers = module.validate_checkpoint(tmp_path)
    assert sum("digest mismatch" in b for b in blockers) == len(names), blockers


def test_clvc_net_preflight_still_blocks_when_the_registry_is_not_verified(tmp_path: Path):
    """A verified digest plus an unverified registry status is still a refusal."""
    import hashlib
    from unittest.mock import patch

    from workbench.replay import clvc_net as module

    names = [name for cat in CATEGORIES for name in EXPECTED_CATEGORY_FILES[cat]]
    members = []
    for name in names:
        (tmp_path / name).write_bytes(f"bytes of {name}".encode())
        members.append({"filename": name,
                        "expected_sha256": hashlib.sha256(f"bytes of {name}".encode()).hexdigest()})
    variant = dict(module.CHECKPOINT_VARIANT)
    variant["bundle_members"] = members
    variant["checkpoint_mapping_status"] = "UNVERIFIED"
    with patch.object(module, "CHECKPOINT_VARIANT", variant):
        blockers = module.validate_checkpoint(tmp_path)
    assert any("mapping is unresolved" in b.lower() for b in blockers), blockers


def test_clvc_net_test_function_signature():
    """Verify CLVC-Net test.py test() signature is (params, local_model, global_model, testset, dataname) - 5 args."""
    clvc_source = Path(__file__).resolve().parents[2] / "workbench" / "third_party" / "CLVCNet"
    test_py = clvc_source / "test.py"
    assert test_py.is_file()
    for line in test_py.read_text().splitlines():
        if line.startswith("def test("):
            assert line.strip() == "def test(params, local_model, global_model, testset, dataname):"
            break
    else:
        pytest.fail("def test(...) not found in CLVCNet test.py")
