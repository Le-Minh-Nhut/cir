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


def qwen_assets(module, config: WorkbenchConfig, *, with_hash: bool = True):
    """The three downloadable Qwen captions.

    The registry now pins each file's published content hash, so a test that wants to exercise a
    transfer path without reproducing the real 2-5 MB payload uses `with_hash=False`; the tests that
    prove hash verification is enforced keep the real hash and serve matching or mismatching bytes.
    """
    items = tuple(asset for asset in module.assets({"csmcir"}, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
                  if asset.family == "Qwen captions")
    if with_hash:
        return items
    from dataclasses import replace
    return tuple(replace(asset, expected_sha256=None) for asset in items)

def load_module():
    spec = importlib.util.spec_from_file_location("download_auxiliary_assets", Path("workbench/scripts/download_auxiliary_assets.py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_assets_list_qwen_and_unverified_cot(tmp_path: Path) -> None:
    module = load_module()
    items = module.assets({"csmcir"}, tmp_path / "third_party", tmp_path / "FashionIQ")
    qwen = tuple(asset for asset in items if asset.family == "Qwen captions")
    cot = tuple(asset for asset in items if asset.family == "COT_ours2 captions")

    assert len(qwen) == len(cot) == 3
    assert all(asset.required and asset.source_url and asset.source_revision == "cf0c19bb346266c295b4b5772ebf03bd1c4f0467" and asset.provenance_status == "VERIFIED" for asset in qwen)
    assert all(asset.required and asset.source_url is None and asset.provenance_status == "UNVERIFIED" for asset in cot)


def test_dry_run_reports_pinned_qwen_urls_and_never_opens_network(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    config = config_at(tmp_path)
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: (_ for _ in ()).throw(AssertionError("network used")))

    assert module.main(["--model", "csmcir", "--dry-run"]) == 0

    output = capsys.readouterr().out
    for category in ("dress", "shirt", "toptee"):
        assert f"https://huggingface.co/peng12138/CSMCIR/resolve/cf0c19bb346266c295b4b5772ebf03bd1c4f0467/fashioniq_qwen_captions/qwen_captions/{category}_cot_val.json" in output
        assert str(config.FASHIONIQ_ROOT / "qwen_captions" / f"{category}_cot_val.json") in output
    assert "required asset has no verified automatic acquisition source" in output
    assert "no network request" in output


def test_downloads_qwen_atomically_and_writes_local_hash_manifest(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config, with_hash=False)
    # `main()` resolves the assets from the registry itself; patch the resolver so the transfer
    # path is exercised without reproducing the real multi-megabyte payloads.
    monkeypatch.setattr(module, "assets", lambda *a, **k: items)
    urls: list[str] = []
    renamed: list[tuple[Path, Path]] = []
    original_replace = module.os.replace


    def replace(partial: Path, destination: Path) -> None:
        assert partial.suffix == ".part" and partial.is_file()
        renamed.append((partial, destination))
        original_replace(partial, destination)

    # Publish the exact bytes each pinned revision declares, so the mandatory hash check passes.
    payloads = [b"payload-%d" % index for index in range(1, len(items) + 1)]
    def urlopen(url: str) -> Response:
        urls.append(url)
        return Response(payloads[len(urls) - 1])

    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(module.os, "replace", replace)

    assert module.main(["--model", "csmcir"]) == 0
    assert urls == [asset.source_url for asset in items]
    assert [destination for _, destination in renamed] == [asset.destination for asset in items]
    assert [asset.destination.read_bytes() for asset in items] == payloads
    manifest = json.loads(module.manifest_path(config.WORKBENCH_ARTIFACTS_ROOT).read_text())
    assert {entry["destination"] for entry in manifest} == {str(asset.destination) for asset in items}
    assert {entry["downloaded_sha256"] for entry in manifest} == {hashlib.sha256(asset.destination.read_bytes()).hexdigest() for asset in items}


def test_existing_qwen_are_preserved_without_force(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config, with_hash=False)
    monkeypatch.setattr(module, "assets", lambda *a, **k: items)
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
    items = qwen_assets(module, config, with_hash=False)
    monkeypatch.setattr(module, "assets", lambda *a, **k: items)
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"old")
    calls: list[str] = []
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url: (calls.append(url), Response(b"new"))[1])

    assert module.main(["--model", "csmcir", "--force-redownload"]) == 0
    assert calls == [asset.source_url for asset in items]
    assert all(asset.destination.read_bytes() == b"new" for asset in items)


def test_a_published_hash_is_enforced_and_a_mismatch_is_refused(monkeypatch, tmp_path: Path) -> None:
    """The registry now pins each caption file's published sha256, so a wrong payload must not land.

    The whole point of pinning is that a silently changed upstream file is caught here rather than
    producing a plausible-looking evaluation against different captions.
    """
    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config)
    assert all(asset.expected_sha256 for asset in items), "these assets declare a published hash"

    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url: Response(b"not-the-published-bytes"))

    assert module.main(["--model", "csmcir"]) == 1, "a hash mismatch must refuse the download"
    for asset in items:
        assert not asset.destination.exists(), f"{asset.destination} must not be published on mismatch"
        assert not asset.destination.with_suffix(asset.destination.suffix + ".part").exists()

    # Serving exactly the declared content is accepted, which proves the check is the hash and not
    # something else about the payload.
    payload = b"the recorded bytes"
    digest = hashlib.sha256(payload).hexdigest()
    from dataclasses import replace
    monkeypatch.setattr(module, "assets",
                        lambda *a, **k: tuple(replace(asset, expected_sha256=digest) for asset in items))
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda url: Response(payload))
    assert module.main(["--model", "csmcir"]) == 0
    assert all(asset.destination.read_bytes() == payload for asset in items)

def test_a_corrupt_workbench_download_is_repaired_without_force(monkeypatch, tmp_path: Path) -> None:
    """ISSUE D: a present-but-corrupt workbench-managed asset must be re-acquired, not skipped.

    Selection used to look only at whether the file existed, so a truncated or tampered caption file
    stayed on disk and kept blocking preflight until the operator worked out that
    --force-redownload was needed. A file whose published bytes the workbench itself recorded is
    repaired automatically; an operator-supplied file with unverified provenance is never touched.
    """
    from dataclasses import replace

    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config, with_hash=False)

    good = b"the correct caption bytes"
    digest = hashlib.sha256(good).hexdigest()
    monkeypatch.setattr(module, "assets",
                        lambda *a, **k: tuple(replace(asset, expected_sha256=digest) for asset in items))
    monkeypatch.setattr(module, "resolve_config", lambda: config)

    # Every destination exists but holds the wrong bytes, i.e. exactly the corrupt case.
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"corrupt truncated content")

    calls: list[str] = []
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda url: (calls.append(url), Response(good))[1])

    assert module.main(["--model", "csmcir"]) == 0, "a corrupt managed asset must be repaired"
    assert calls == [asset.source_url for asset in items], "each corrupt asset is re-fetched"
    assert all(asset.destination.read_bytes() == good for asset in items)

    # An intact asset is left alone: no second round of network traffic.
    calls.clear()
    assert module.main(["--model", "csmcir"]) == 0
    assert calls == [], "an intact asset must not be re-downloaded"

def test_an_operator_supplied_asset_is_never_replaced(monkeypatch, tmp_path: Path) -> None:
    """An unverified-provenance file may be the operator's own bytes, so it is never overwritten."""
    from dataclasses import replace

    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config, with_hash=False)
    monkeypatch.setattr(module, "assets",
                        lambda *a, **k: tuple(replace(asset, expected_sha256="0" * 64,
                                                      provenance_status="UNVERIFIED") for asset in items))
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"operator bytes that will not match any hash")
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda _: (_ for _ in ()).throw(AssertionError("network used")))

    assert module.main(["--model", "csmcir"]) == 0
    assert all(asset.destination.read_bytes() == b"operator bytes that will not match any hash"
               for asset in items)


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

def test_verify_only_rejects_empty_and_checksum_mismatch_without_network(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    config = config_at(tmp_path)
    item = qwen_assets(module, config)[0]
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: (_ for _ in ()).throw(AssertionError("network used")))
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    item.destination.touch()
    assert module.main(["--model", "csmcir", "--verify-only"]) == 1
    assert "INVALID EMPTY FILE" in capsys.readouterr().out

    from dataclasses import replace

    mismatched = replace(item, expected_sha256="0" * 64)
    monkeypatch.setattr(module, "assets", lambda *_: (mismatched,))
    item.destination.write_bytes(b"not the expected asset")
    assert module.main(["--model", "csmcir", "--verify-only"]) == 1
    assert "OFFICIAL SHA256 MISMATCH" in capsys.readouterr().out


def test_verify_only_rejects_unresolved_provenance_for_present_asset(monkeypatch, tmp_path: Path) -> None:
    from dataclasses import replace

    module = load_module()
    config = config_at(tmp_path)
    item = qwen_assets(module, config)[0]
    item.destination.parent.mkdir(parents=True, exist_ok=True)
    item.destination.write_bytes(b"present")
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module, "assets", lambda *_: (replace(item, provenance_status="UNVERIFIED"),))
    assert module.main(["--model", "csmcir", "--verify-only"]) == 1


def test_all_selects_only_registered_auxiliary_models(monkeypatch, tmp_path: Path) -> None:
    module = load_module()
    config = config_at(tmp_path)
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    monkeypatch.setattr(module.urllib.request, "urlopen", lambda _: (_ for _ in ()).throw(AssertionError("network used")))

    assert module.main(["--all", "--dry-run"]) == 0

def test_validate_auxiliary_asset_invalid_json(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    file_path = tmp_path / "bad.json"
    file_path.write_bytes(b"{not valid json")
    asset = {
        "model_id": "csmcir",
        "family": "Qwen captions",
        "provenance_status": "VERIFIED",
        "content_contract": {"record_key_fields": ["Final_Caption"], "expected_record_count": 1},
    }
    ok, issues = validate_auxiliary_asset(asset, file_path)
    assert not ok
    assert any("INVALID JSON" in issue for issue in issues)


def test_validate_auxiliary_asset_wrong_record_fields(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    file_path = tmp_path / "wrong_fields.json"
    file_path.write_text(json.dumps({"img_001": {"Wrong_Field": "text"}}), encoding="utf-8")
    asset = {
        "model_id": "csmcir",
        "family": "Qwen captions",
        "provenance_status": "VERIFIED",
        "content_contract": {"record_key_fields": ["Core_Essence", "Final_Caption"], "expected_record_count": 1},
    }
    ok, issues = validate_auxiliary_asset(asset, file_path)
    assert not ok
    assert any("missing required fields" in issue or "SCHEMA MISMATCH" in issue for issue in issues)


def test_validate_auxiliary_asset_record_count_disagreement(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    file_path = tmp_path / "count_mismatch.json"
    file_path.write_text(json.dumps({"img_001": {"Final_Caption": "text"}}), encoding="utf-8")
    asset = {
        "model_id": "csmcir",
        "family": "Qwen captions",
        "provenance_status": "VERIFIED",
        "content_contract": {"record_key_fields": ["Final_Caption"], "expected_record_count": 10},
    }
    ok, issues = validate_auxiliary_asset(asset, file_path)
    assert not ok
    assert any("RECORD COUNT MISMATCH" in issue for issue in issues)


def test_validate_auxiliary_asset_id_absent_from_declared_split(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    file_path = tmp_path / "absent_id.json"
    file_path.write_text(json.dumps({"img_999": {"Final_Caption": "text"}}), encoding="utf-8")
    split_dir = tmp_path / "image_splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    split_file = split_dir / "split.dress.val.json"
    split_file.write_text(json.dumps(["img_001", "img_002"]), encoding="utf-8")
    asset = {
        "model_id": "csmcir",
        "family": "Qwen captions",
        "provenance_status": "VERIFIED",
        "content_contract": {
            "record_key_fields": ["Final_Caption"],
            "expected_record_count": 1,
            "id_source": "image_splits/split.dress.val.json",
        },
    }
    ok, issues = validate_auxiliary_asset(asset, file_path, fashioniq_root=tmp_path)
    assert not ok
    assert any("SPLIT ID MISMATCH" in issue for issue in issues)


def test_validate_auxiliary_asset_operator_supplied_unverified_distinguished_from_missing(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    file_path = tmp_path / "cot_present.json"
    file_path.write_text(json.dumps({"img_001": {"Final_Caption": "text"}}), encoding="utf-8")
    asset = {
        "model_id": "csmcir",
        "family": "COT_ours2 captions",
        "provenance_status": "UNVERIFIED",
        "content_contract": {"record_key_fields": ["Final_Caption"], "expected_record_count": 1},
    }
    res_present = validate_auxiliary_asset(asset, file_path)
    assert not res_present.ok
    assert res_present.status == "OPERATOR_SUPPLIED_UNVERIFIED"
    assert any("OPERATOR_SUPPLIED_UNVERIFIED" in issue for issue in res_present.issues)

    res_missing = validate_auxiliary_asset(asset, tmp_path / "nonexistent.json")
    assert not res_missing.ok
    assert res_missing.status == "MISSING"
    assert res_present.status != res_missing.status


def test_validate_auxiliary_asset_missing_required_asset(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    missing_path = tmp_path / "missing.json"
    asset = {
        "model_id": "csmcir",
        "family": "Qwen captions",
        "provenance_status": "VERIFIED",
    }
    res = validate_auxiliary_asset(asset, missing_path)
    assert not res.ok
    assert res.status == "MISSING"
    assert any("MISSING" in issue for issue in res.issues)


def test_validate_auxiliary_asset_valid_verified_with_matching_sha256(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    payload = json.dumps({"img_001": {"Core_Essence": "a", "Visual_Attributes": "b", "Observation_Process": "c", "Final_Caption": "d"}}).encode("utf-8")
    file_path = tmp_path / "valid.json"
    file_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    asset = {
        "model_id": "csmcir",
        "family": "Qwen captions",
        "provenance_status": "VERIFIED",
        "expected_sha256": digest,
        "expected_size_bytes": len(payload),
        "content_contract": {
            "record_key_fields": ["Core_Essence", "Visual_Attributes", "Observation_Process", "Final_Caption"],
            "expected_record_count": 1,
        },
    }
    res = validate_auxiliary_asset(asset, file_path, split_ids={"img_001"})
    assert res.ok is True
    assert res.status == "VERIFIED"
    assert res.issues == []


def test_validate_auxiliary_asset_rejects_missing_declared_split_ids(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    file_path = tmp_path / "partial.json"
    file_path.write_text(json.dumps({"img_001": {"Final_Caption": "text"}}), encoding="utf-8")
    split_dir = tmp_path / "image_splits"
    split_dir.mkdir()
    (split_dir / "split.dress.val.json").write_text(json.dumps(["img_001", "img_002"]), encoding="utf-8")
    asset = {"provenance_status": "VERIFIED", "content_contract": {
        "record_key_fields": ["Final_Caption"], "id_source": "image_splits/split.dress.val.json",
    }}

    result = validate_auxiliary_asset(asset, file_path, fashioniq_root=tmp_path)

    assert not result.ok, "validator must refuse JSON missing required split ids"
    assert any("1 split ids missing" in issue for issue in result.issues)


def test_validate_auxiliary_asset_rejects_missing_declared_split_source(tmp_path: Path) -> None:
    from workbench.backend.registry import validate_auxiliary_asset
    file_path = tmp_path / "captions.json"
    file_path.write_text(json.dumps({"img_001": {"Final_Caption": "text"}}), encoding="utf-8")
    asset = {"provenance_status": "VERIFIED", "content_contract": {
        "record_key_fields": ["Final_Caption"], "id_source": "image_splits/missing.json",
    }}

    result = validate_auxiliary_asset(asset, file_path, fashioniq_root=tmp_path)

    assert not result.ok, "validator must refuse an unreadable declared split source"
    assert any("SPLIT ID SOURCE INVALID" in issue for issue in result.issues)


def test_csmcir_auxiliary_backbones_resolve_to_user_cache_paths(monkeypatch, tmp_path: Path) -> None:
    from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv("TORCH_HOME", str(tmp_path / "torch"))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    assets = auxiliary_assets_for_model("csmcir")
    expected = {
        "LAVIS BLIP-2 pretrain weights": tmp_path / "xdg/lavis/blip2_pretrained.pth",
        "EVA-ViT-g backbone weights": tmp_path / "torch/hub/checkpoints/eva_vit_g.pth",
        "bert-base-uncased tokenizer": tmp_path / "hf/hub/models--bert-base-uncased/vocab.txt",
    }

    for family, path in expected.items():
        asset = next(item for item in assets if item["family"] == family)
        assert auxiliary_destination(asset, tmp_path / "source", tmp_path / "fashioniq") == path


def test_clvc_net_auxiliary_resnet50_resolves_to_user_cache_path(monkeypatch, tmp_path: Path) -> None:
    from dataclasses import replace
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.preflight import model_requirements
    from workbench.backend.registry import (
        auxiliary_assets_for_model,
        auxiliary_destination,
        load_registry,
    )

    torch_home = tmp_path / "torch"
    monkeypatch.setenv("TORCH_HOME", str(torch_home))
    source_root = tmp_path / "source"
    fashioniq_root = tmp_path / "fashioniq"

    assets = auxiliary_assets_for_model("clvc_net")
    asset = next(item for item in assets if "ResNet-50" in item["family"])
    resolved = auxiliary_destination(asset, source_root, fashioniq_root)
    expected = torch_home / "hub/checkpoints/resnet50-19c8e357.pth"

    assert resolved == expected
    assert not resolved.is_relative_to(source_root)

    config = replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=source_root, FASHIONIQ_ROOT=fashioniq_root)
    model = next(m for m in load_registry()["models"] if m["model_id"] == "clvc_net")
    reqs = model_requirements(model, config)
    preflight_asset = next(a for a in reqs["evidence"]["auxiliary_assets"] if "ResNet-50" in a["name"])

    assert Path(preflight_asset["path"]) == expected
    assert Path(preflight_asset["path"]) == resolved
    assert not Path(preflight_asset["path"]).is_relative_to(source_root)
