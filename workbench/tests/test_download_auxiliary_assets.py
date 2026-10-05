from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import importlib.util
import sys
from pathlib import Path

from workbench.backend.operator_config import WorkbenchConfig


class Response(io.BytesIO):
    def __init__(self, payload: bytes, status: int = 200) -> None:
        super().__init__(payload)
        self.status = status

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def getcode(self) -> int:
        return self.status


def config_at(root: Path) -> WorkbenchConfig:
    return WorkbenchConfig(root, root / "data", root / "FashionIQ", "127.0.0.1", 8000, 5173, root / "checkpoints", root / "results", root / "third_party")


def qwen_assets(module, config: WorkbenchConfig):
    return tuple(asset for asset in module.assets(config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT) if asset.automatic_download_available)

def load_module():
    spec = importlib.util.spec_from_file_location("download_auxiliary_assets", Path("workbench/scripts/download_auxiliary_assets.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assets_list_qwen_and_unverified_cot(tmp_path: Path) -> None:
    module = load_module()
    items = module.assets(tmp_path / "third_party", tmp_path / "FashionIQ")
    qwen = tuple(asset for asset in items if asset.automatic_download_available)
    cot = tuple(asset for asset in items if not asset.automatic_download_available)

    assert len(qwen) == len(cot) == 3
    assert all(asset.required and asset.source_url and not asset.source_unverified for asset in qwen)
    assert all(asset.required and asset.source_url is None and asset.source_unverified for asset in cot)


def test_dry_run_reports_exact_qwen_urls_and_never_opens_network(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    config = config_at(tmp_path)
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: (_ for _ in ()).throw(AssertionError("network used")))

    assert module.main(["--model", "csmcir", "--dry-run"]) == 0

    output = capsys.readouterr().out
    for category in ("dress", "shirt", "toptee"):
        assert f"https://huggingface.co/peng12138/CSMCIR/resolve/main/fashioniq_qwen_captions/qwen_captions/{category}_cot_val.json" in output
        assert str(config.FASHIONIQ_ROOT / "qwen_captions" / f"{category}_cot_val.json") in output
    assert "no network request" in output


def test_downloads_qwen_atomically_and_writes_local_hash_manifest(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config)
    urls: list[str] = []
    renamed: list[tuple[Path, Path]] = []
    original_replace = module.os.replace

    def urlopen(url: str) -> Response:
        urls.append(url)
        return Response(f"payload-{len(urls)}".encode())

    def replace(partial: Path, destination: Path) -> None:
        assert partial.suffix == ".part" and partial.is_file()
        renamed.append((partial, destination))
        original_replace(partial, destination)

    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(module.os, "replace", replace)

    assert module.main(["--model", "csmcir"]) == 0
    assert urls == [asset.source_url for asset in items]
    assert [destination for _, destination in renamed] == [asset.destination for asset in items]
    assert [asset.destination.read_bytes() for asset in items] == [b"payload-1", b"payload-2", b"payload-3"]
    manifest = json.loads(module.manifest_path(config.CIR_REPO_ROOT).read_text())
    assert {entry["destination"] for entry in manifest} == {str(asset.destination) for asset in items}
    assert {entry["downloaded_sha256"] for entry in manifest} == {hashlib.sha256(asset.destination.read_bytes()).hexdigest() for asset in items}


def test_existing_qwen_are_preserved_without_force(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config)
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"existing")
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: (_ for _ in ()).throw(AssertionError("network used")))

    assert module.main(["--model", "csmcir"]) == 0
    assert all(asset.destination.read_bytes() == b"existing" for asset in items)


def test_force_redownload_replaces_existing_qwen(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config)
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"old")
    calls: list[str] = []
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url: (calls.append(url), Response(b"new"))[1])

    assert module.main(["--model", "csmcir", "--force-redownload"]) == 0
    assert calls == [asset.source_url for asset in items]
    assert all(asset.destination.read_bytes() == b"new" for asset in items)


def test_failed_download_leaves_no_corrupt_destination(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    first = qwen_assets(module, config)[0]
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: (_ for _ in ()).throw(urllib.error.URLError("offline")))

    assert module.main(["--model", "csmcir"]) == 1
    assert not first.destination.exists()
    assert not first.destination.with_suffix(first.destination.suffix + ".part").exists()


def test_verify_only_blocks_for_missing_cot_without_network(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    for asset in qwen_assets(module, config):
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"qwen")
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: (_ for _ in ()).throw(AssertionError("network used")))

    assert module.main(["--model", "csmcir", "--verify-only"]) == 1
