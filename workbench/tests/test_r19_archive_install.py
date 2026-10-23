"""Regression coverage for registry-declared checkpoint archives."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "download_checkpoints.py"


def _module():
    spec = importlib.util.spec_from_file_location("r19_archive_download", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _archive(payload: bytes, member_name: str = "best_model.pth") -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(member_name)
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


def _setup(module, tmp_path: Path, monkeypatch, variants: list[dict], payloads: dict[str, bytes]):
    root = tmp_path / "checkpoints"
    model = {"model_id": "tme", "method_name": "TME", "checkpoint_variants": variants}
    monkeypatch.setattr(module, "resolve_config", lambda: SimpleNamespace(WORKBENCH_CHECKPOINT_ROOT=root))
    monkeypatch.setattr(module, "load_registry", lambda: {"models": [model]})
    requests: list[str] = []
    limits: list[int] = []

    class Response:
        headers = {}

        def __init__(self, body: bytes):
            self.body = io.BytesIO(body)

        def getcode(self):
            return 200

        def read(self, size: int = -1):
            return self.body.read(size)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            self.body.close()

    def fake_urlopen(request):
        requests.append(request.full_url)
        return Response(payloads[request.full_url])

    original_write = module._write_response

    def record_write(response, partial: Path, mode: str, limit: int) -> None:
        limits.append(limit)
        original_write(response, partial, mode, limit)

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(module, "_write_response", record_write)
    return root, requests, limits
def test_real_download_builds_request_and_streams_response(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    body = b"network response bytes"
    seen = []

    class Response:
        headers = {}

        def __init__(self):
            self.reads = 0

        def getcode(self):
            return 200

        def read(self, size: int = -1):
            seen.append("read")
            self.reads += 1
            return body if self.reads == 1 else b""

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

    def fake_urlopen(request):
        assert isinstance(request, module.urllib.request.Request)
        assert request.full_url == "https://example.invalid/real-download"
        seen.append("urlopen")
        return Response()

    monkeypatch.setattr(module.urllib.request, "urlopen", fake_urlopen)
    partial = tmp_path / "checkpoint.part"
    module.download("https://example.invalid/real-download", partial, len(body))

    assert seen == ["urlopen", "read", "read"]
    assert partial.read_bytes() == body


def _variant(checkpoint_id: str, filename: str, url: str, payload: bytes, **fields) -> dict:
    return {
        "checkpoint_id": checkpoint_id,
        "filename": filename,
        "download_url": url,
        "checkpoint_mapping_status": "VERIFIED_METADATA",
        "checkpoint_training_noise_pct": 0,
        "archive_member": "best_model.pth",
        "archive_size_bytes": len(payload),
        "expected_size_bytes": None,
        "expected_sha256": None,
        **fields,
    }


def test_cli_extracts_exact_member_to_distinct_variant_destinations(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    fashioniq_weight = b"fashioniq distinctive extracted checkpoint bytes"
    cirr_weight = b"cirr different extracted checkpoint bytes"
    fashioniq_archive = _archive(fashioniq_weight)
    cirr_archive = _archive(cirr_weight)
    variants = [
        _variant("fashioniq_0", "fashioniq_0_best_model.pth", "https://example.invalid/fashioniq", fashioniq_archive,
                 expected_size_bytes=len(fashioniq_weight)),
        _variant("cirr_0", "cirr_0_best_model.pth", "https://example.invalid/cirr", cirr_archive,
                 expected_size_bytes=len(cirr_weight)),
    ]
    root, calls, limits = _setup(module, tmp_path, monkeypatch, variants,
                                 {"https://example.invalid/fashioniq": fashioniq_archive,
                                  "https://example.invalid/cirr": cirr_archive})

    assert module.main(["--model", "tme", "--checkpoint", "fashioniq_0", "--checkpoint", "cirr_0",
                        "--output-root", str(root)]) == 0
    fashioniq_path = root / "tme" / "fashioniq_0_best_model.pth"
    cirr_path = root / "tme" / "cirr_0_best_model.pth"
    assert fashioniq_path.read_bytes() == fashioniq_weight
    assert cirr_path.read_bytes() == cirr_weight
    assert fashioniq_path.read_bytes()[:2] != b"\x1f\x8b"
    assert fashioniq_path != cirr_path and fashioniq_path.read_bytes() != cirr_path.read_bytes()
    assert len(calls) == 2
    assert limits == [len(fashioniq_archive), len(cirr_archive)]

    manifest = json.loads((root / "download_manifest.json").read_text())
    fashioniq_record = next(item for item in manifest if item["checkpoint_id"] == "fashioniq_0")
    assert fashioniq_record["downloaded_compressed_archive_size_bytes"] == len(fashioniq_archive)
    assert fashioniq_record["local_compressed_archive_sha256"] == hashlib.sha256(fashioniq_archive).hexdigest()
    assert fashioniq_record["extracted_size_bytes"] == len(fashioniq_weight)
    assert fashioniq_record["local_extracted_sha256"] == hashlib.sha256(fashioniq_weight).hexdigest()
    assert "official_sha256" not in fashioniq_record


def test_verify_only_accepts_extracted_member_and_rejects_archive_masquerading_as_weight(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    module = _module()
    weight = b"valid extracted weight"
    archive = _archive(weight)
    variant = _variant("fashioniq_0", "fashioniq_0_best_model.pth", "https://example.invalid/archive", archive,
                       expected_size_bytes=len(weight), expected_sha256=hashlib.sha256(weight).hexdigest())
    root, _, _ = _setup(module, tmp_path, monkeypatch, [variant], {"https://example.invalid/archive": archive})
    args = ["--model", "tme", "--checkpoint", "fashioniq_0", "--output-root", str(root)]

    assert module.main(args) == 0
    assert module.main([*args, "--verify-only"]) == 0
    destination = root / "tme" / variant["filename"]
    destination.write_bytes(archive)
    assert module.main([*args, "--verify-only"]) == 1
    assert "INVALID ARCHIVE CONTAINER" in capsys.readouterr().out
    assert destination.read_bytes().startswith(b"\x1f\x8b")


@pytest.mark.parametrize("bad_field", ["expected_size_bytes", "expected_sha256"])
def test_extracted_size_or_hash_mismatch_leaves_no_published_file(
    tmp_path: Path, monkeypatch, bad_field: str
) -> None:
    module = _module()
    weight = b"checkpoint member payload"
    archive = _archive(weight)
    fields = {"expected_size_bytes": len(weight) + 1}
    if bad_field == "expected_sha256":
        fields = {"expected_size_bytes": len(weight), "expected_sha256": "0" * 64}
    variant = _variant("fashioniq_0", "fashioniq_0_best_model.pth", "https://example.invalid/archive", archive,
                       **fields)
    root, _, _ = _setup(module, tmp_path, monkeypatch, [variant], {"https://example.invalid/archive": archive})

    with pytest.raises(RuntimeError, match="size mismatch|sha256 mismatch"):
        module.main(["--model", "tme", "--checkpoint", "fashioniq_0", "--output-root", str(root)])
    assert not (root / "tme" / variant["filename"]).exists()
    assert not list(root.rglob("*.tmp"))
    assert not list(root.rglob("*.part"))

def test_archive_without_exact_declared_member_is_not_published(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    archive = _archive(b"must not install", "../best_model.pth")
    variant = _variant("fashioniq_0", "fashioniq_0_best_model.pth", "https://example.invalid/archive", archive)
    root, _, _ = _setup(module, tmp_path, monkeypatch, [variant], {"https://example.invalid/archive": archive})

    with pytest.raises(RuntimeError, match="exactly one regular member"):
        module.main(["--model", "tme", "--checkpoint", "fashioniq_0", "--output-root", str(root)])
    assert not (root / "tme" / variant["filename"]).exists()
    assert not list(root.rglob("*.tmp"))
    assert not list(root.rglob("*.part"))


def test_reuses_valid_extracted_checkpoint_without_redownloading(tmp_path: Path, monkeypatch) -> None:
    module = _module()
    weight = b"reusable extracted bytes"
    archive = _archive(weight)
    variant = _variant("fashioniq_0", "fashioniq_0_best_model.pth", "https://example.invalid/archive", archive,
                       expected_size_bytes=len(weight), expected_sha256=hashlib.sha256(weight).hexdigest())
    root, calls, _ = _setup(module, tmp_path, monkeypatch, [variant], {"https://example.invalid/archive": archive})
    args = ["--model", "tme", "--checkpoint", "fashioniq_0", "--output-root", str(root)]

    assert module.main(args) == 0
    assert module.main(args) == 0
    assert calls == ["https://example.invalid/archive"]
