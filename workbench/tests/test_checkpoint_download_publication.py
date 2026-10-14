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

    def fake_download(url, partial: Path, limit=None) -> None:
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


def test_the_declared_size_is_passed_as_the_streaming_limit(tmp_path: Path, monkeypatch) -> None:
    """ISSUE A: a declared size must bound the transfer, not just be checked after it.

    Checking the size only after the body landed meant a runaway or wrong response was written to
    disk in full before being rejected. The declared size now bounds the write, so the check is a
    property of the transfer and not a post-mortem.
    """
    module = _module()
    seen: list[object] = []
    checkpoint = dict(CHECKPOINT, expected_sha256=None)
    destination_root = tmp_path / "checkpoints"
    model = {"model_id": "synthetic", "method_name": "Synthetic", "source_dir": None}
    monkeypatch.setattr(module, "resolve_config",
                        lambda: SimpleNamespace(WORKBENCH_CHECKPOINT_ROOT=destination_root))

    def capturing_download(url, partial: Path, limit=None) -> None:
        seen.append(limit)
        partial.write_bytes(BODY)

    monkeypatch.setattr(module, "download", capturing_download)
    monkeypatch.setattr(module, "selected", lambda args, models: [(model, checkpoint)])
    monkeypatch.setattr(module, "load_registry", lambda: {"models": [model]})
    assert module.main(["--model", "synthetic", "--output-root", str(destination_root)]) == 0
    assert seen == [4], "the registry's declared size must reach the downloader"

    # With no declared size the safety ceiling still applies, so an unlimited transfer is never
    # requested even for a checkpoint whose size the author never published.
    seen.clear()
    unbounded = {k: v for k, v in checkpoint.items() if k != "expected_size_bytes"}
    monkeypatch.setattr(module, "selected", lambda args, models: [(model, unbounded)])
    assert module.main(["--model", "synthetic", "--output-root", str(destination_root),
                        "--force-redownload"]) == 0
    assert seen and seen[0] == module._DOWNLOAD_CEILING

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


def test_failed_force_redownload_preserves_existing_checkpoint(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    destination_root = tmp_path / "checkpoints"
    import hashlib
    checkpoint = dict(CHECKPOINT, expected_sha256=hashlib.sha256(BODY).hexdigest())
    destination = destination_root / "synthetic" / "weights.pt"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(BODY)
    model = {"model_id": "synthetic", "method_name": "Synthetic", "source_dir": None}
    monkeypatch.setattr(module, "resolve_config", lambda: SimpleNamespace(
        WORKBENCH_CHECKPOINT_ROOT=destination_root))
    monkeypatch.setattr(module, "selected", lambda args, models: [(model, checkpoint)])
    monkeypatch.setattr(module, "load_registry", lambda: {"models": [model]})

    def failed_download(url, partial: Path, limit=None) -> None:
        raise OSError("network unavailable")

    monkeypatch.setattr(module, "download", failed_download)
    with pytest.raises(OSError, match="network unavailable"):
        module.main(["--model", "synthetic", "--output-root", str(destination_root),
                     "--force-redownload"])

    assert destination.read_bytes() == BODY

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
    monkeypatch.setattr(module, "download", lambda url, partial, limit=None: partial.write_bytes(BODY))

    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        module.main(["--model", "synthetic", "--output-root", str(destination_root)])
    assert not (destination_root / "synthetic" / "weights.pt").exists()

    current = checkpoint
    assert module.main(["--model", "synthetic", "--output-root", str(destination_root)]) == 0
    assert (destination_root / "synthetic" / "weights.pt").read_bytes() == BODY


def test_limn_repeatable_selection_runs_real_cli_offline(tmp_path: Path) -> None:
    import hashlib
    import os
    import subprocess
    import yaml
    from workbench.backend.registry import load_registry

    registry = load_registry()
    limn = next(model for model in registry["models"] if model["model_id"] == "limn")
    output_root = tmp_path / "checkpoints"
    limn_dir = output_root / "limn"
    limn_dir.mkdir(parents=True)
    ids = ["base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"]
    filenames = []
    for checkpoint in limn["checkpoint_variants"]:
        payload = f"synthetic {checkpoint['checkpoint_id']}".encode()
        checkpoint["expected_size_bytes"] = len(payload)
        checkpoint["expected_sha256"] = hashlib.sha256(payload).hexdigest()
        destination = limn_dir / checkpoint["filename"]
        destination.write_bytes(payload)
        filenames.append(destination.name)

    registry_path = tmp_path / "registry.yaml"
    registry_path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    env = os.environ.copy()
    env["WORKBENCH_REGISTRY_PATH"] = str(registry_path)
    command = [sys.executable, str(SCRIPT), "--model", "limn", "--output-root", str(output_root)]
    for checkpoint_id in ids:
        command.extend(("--checkpoint", checkpoint_id))
    result = subprocess.run(command, capture_output=True, text=True, env=env)

    assert result.returncode == 0, result.stderr
    checkpoint_lines = [line for line in result.stdout.splitlines() if line.startswith("Checkpoint:")]
    assert len(checkpoint_lines) == 3
    assert [line.split(" / ", 1)[0].removeprefix("Checkpoint: ") for line in checkpoint_lines] == ids
    assert "DOWNLOADED" not in result.stdout
    assert sorted(path.name for path in limn_dir.iterdir()) == sorted(filenames)

    for invalid_id, expected in (("unknown_checkpoint", "unknown checkpoint for limn"),
                                 ("fashioniq", "unknown checkpoint for limn")):
        rejected = subprocess.run(
            [sys.executable, str(SCRIPT), "--model", "limn", "--output-root", str(output_root),
             "--checkpoint", "base_iter0_dress", "--checkpoint", invalid_id],
            capture_output=True, text=True, env=env,
        )
        assert rejected.returncode == 2
        assert expected in rejected.stderr and invalid_id in rejected.stderr


def test_limn_three_category_checkpoints_selected_and_verified(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    from workbench.backend.registry import load_registry

    reg = load_registry()
    limn = next(m for m in reg["models"] if m["model_id"] == "limn")
    output_root = tmp_path / "checkpoints"
    limn_dir = output_root / "limn"
    limn_dir.mkdir(parents=True)

    expected_map = {}
    for variant in limn["checkpoint_variants"]:
        fn = limn_dir / variant["filename"]
        fn.write_bytes(b"synthetic valid content")
        expected_map[variant["filename"]] = variant["expected_sha256"]

    # Synthetic local files pass sha256 check
    orig_sha = module.sha256_file
    monkeypatch.setattr(module, "sha256_file", lambda p: expected_map.get(Path(p).name, orig_sha(p)))
    monkeypatch.setattr(module, "download", lambda url, partial, limit=None: pytest.fail("network download attempted"))

    cmd = ["--model", "limn", "--output-root", str(output_root),
           "--checkpoint", "base_iter0_dress",
           "--checkpoint", "base_iter0_shirt",
           "--checkpoint", "base_iter0_toptee"]

    parser_args = module.argparse.Namespace(
        model="limn",
        checkpoint=["base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"],
        all=False,
        list=False,
    )
    items = module.selected(parser_args, reg["models"])
    assert len(items) == 3
    selected_cids = [item[1]["checkpoint_id"] for item in items]
    assert selected_cids == ["base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"]

    code = module.main(cmd)
    assert code == 0


def test_limn_unknown_checkpoint_refused(tmp_path: Path) -> None:
    module = _module()
    from workbench.backend.registry import load_registry

    reg = load_registry()
    args = module.argparse.Namespace(
        model="limn",
        checkpoint=["unknown_ckpt"],
        all=False,
        list=False,
    )
    with pytest.raises(ValueError, match="unknown checkpoint for limn: unknown_ckpt"):
        module.selected(args, reg["models"])

    with pytest.raises(SystemExit) as exc:
        module.main(["--model", "limn", "--output-root", str(tmp_path), "--checkpoint", "unknown_ckpt"])
    assert exc.value.code == 2


def test_limn_wrong_category_checkpoint_refused(tmp_path: Path) -> None:
    module = _module()
    from workbench.backend.registry import load_registry

    reg = load_registry()
    args = module.argparse.Namespace(
        model="limn",
        checkpoint=["base_iter0_dress", "base_iter0_shirt", "base_iter0_shoes"],
        all=False,
        list=False,
    )
    with pytest.raises(ValueError, match="unknown checkpoint for limn: base_iter0_shoes"):
        module.selected(args, reg["models"])

    with pytest.raises(SystemExit) as exc:
        module.main(["--model", "limn", "--output-root", str(tmp_path),
                     "--checkpoint", "base_iter0_dress",
                     "--checkpoint", "base_iter0_shirt",
                     "--checkpoint", "base_iter0_shoes"])
    assert exc.value.code == 2


def test_limn_bundle_expands_to_three_declared_members() -> None:
    module = _module()
    from workbench.backend.registry import load_registry

    reg = load_registry()
    args = module.argparse.Namespace(
        model="limn",
        checkpoint=["base_iter0_all_categories"],
        all=False,
        list=False,
    )
    items = module.selected(args, reg["models"])
    assert len(items) == 3
    assert [c["checkpoint_id"] for _, c in items] == ["base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"]
