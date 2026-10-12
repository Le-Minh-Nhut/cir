"""A checkpoint must not be published until its bytes are verified."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_checkpoints.py"


def _module():
    spec = importlib.util.spec_from_file_location("checkpoint_download_publication", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run(module, tmp_path: Path, checkpoint: dict, payload: bytes, monkeypatch) -> int:
    destination_root = tmp_path / "checkpoints"
    model = {"model_id": "synthetic", "method_name": "Synthetic", "source_dir": None}
    monkeypatch.setattr(module, "resolve_config", lambda: SimpleNamespace(
        WORKBENCH_CHECKPOINT_ROOT=destination_root))

    def fake_download(url, partial: Path) -> None:
        partial.write_bytes(payload)

    monkeypatch.setattr(module, "download", fake_download)
    monkeypatch.setattr(module, "selected", lambda args, models: [(model, checkpoint)])
    monkeypatch.setattr(module, "load_registry", lambda: {"models": [model]})
    return module.main(["--model", "synthetic", "--output-root", str(destination_root)])


CHECKPOINT = {"checkpoint_id": "only", "filename": "weights.pt",
              "download_url": "https://example.invalid/weights.pt",
              "checkpoint_mapping_status": "VERIFIED_METADATA",
              "checkpoint_training_noise_pct": None,
              "expected_size_bytes": 4}

BODY = b"good"


def test_a_hash_mismatch_leaves_no_published_file(tmp_path: Path, monkeypatch) -> None:
    """The failure must not leave a destination the next run refuses to replace."""
    module = _module()
    checkpoint = dict(CHECKPOINT, expected_sha256="0" * 64)
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        _run(module, tmp_path, checkpoint, BODY, monkeypatch)
    assert not (tmp_path / "checkpoints" / "synthetic" / "weights.pt").exists()
    assert not list((tmp_path / "checkpoints").rglob("*.part"))


def test_a_size_mismatch_leaves_no_published_file(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    checkpoint = dict(CHECKPOINT, expected_size_bytes=99, expected_sha256=None)
    with pytest.raises(RuntimeError, match="size mismatch"):
        _run(module, tmp_path, checkpoint, BODY, monkeypatch)
    assert not (tmp_path / "checkpoints" / "synthetic" / "weights.pt").exists()


def test_a_valid_download_is_published_with_its_manifest(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    import hashlib

    checkpoint = dict(CHECKPOINT, expected_sha256=hashlib.sha256(BODY).hexdigest())
    assert _run(module, tmp_path, checkpoint, BODY, monkeypatch) == 0
    published = tmp_path / "checkpoints" / "synthetic" / "weights.pt"
    assert published.read_bytes() == BODY
    assert (tmp_path / "checkpoints" / "download_manifest.json").is_file()


def test_invalid_partial_does_not_block_recovery(tmp_path: Path, monkeypatch) -> None:
    """After a rejected download, a corrected source must be able to publish."""
    module = _module()
    import hashlib

    destination_root = tmp_path / "checkpoints"
    destination_root.mkdir(parents=True, exist_ok=True)
    model = {"model_id": "synthetic", "method_name": "Synthetic", "source_dir": None}
    monkeypatch.setattr(module, "resolve_config", lambda: SimpleNamespace(
        WORKBENCH_CHECKPOINT_ROOT=destination_root))
    monkeypatch.setattr(module, "load_registry", lambda: {"models": [model]})
    monkeypatch.setattr(module, "selected", lambda args, models: [(model, current)])
    checkpoint = dict(CHECKPOINT, expected_sha256=hashlib.sha256(BODY).hexdigest())
    current = dict(checkpoint, expected_sha256="1" * 64)
    monkeypatch.setattr(module, "download", lambda url, partial: partial.write_bytes(BODY))

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        module.main(["--model", "synthetic", "--output-root", str(destination_root)])
    assert not (destination_root / "synthetic" / "weights.pt").exists()

    current = checkpoint
    assert module.main(["--model", "synthetic", "--output-root", str(destination_root)]) == 0
    assert (destination_root / "synthetic" / "weights.pt").read_bytes() == BODY
