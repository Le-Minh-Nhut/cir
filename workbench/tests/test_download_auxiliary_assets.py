from __future__ import annotations

import hashlib
import io
import json
import urllib.error
import importlib.util
import sys
from pathlib import Path

import pytest
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


def test_assets_list_qwen_and_published_cot(tmp_path: Path) -> None:
    module = load_module()
    items = module.assets({"csmcir"}, tmp_path / "third_party", tmp_path / "FashionIQ")
    qwen = tuple(asset for asset in items if asset.family == "Qwen captions")
    cot = tuple(asset for asset in items if asset.family == "COT_ours2 captions")

    assert len(qwen) == len(cot) == 3
    assert all(asset.required and asset.source_url and asset.source_revision == "cf0c19bb346266c295b4b5772ebf03bd1c4f0467" and asset.provenance_status == "VERIFIED" for asset in qwen)
    # The COT_ours2 captions ARE published: the author Hub serves `fashioniq_qwen_captions/
    # qwen_captions/{cat}_cot_val.json`, the local checkout copies are byte-identical to those
    # bytes, and the registry therefore declares a real acquisition source and digest for the
    # in-checkout path the evaluator opens. An earlier revision of this test asserted the opposite
    # ("no source URL, UNVERIFIED"); that assertion encoded a claim the Hub refutes.
    assert all(asset.required and asset.source_url and asset.provenance_status == "VERIFIED"
               and asset.expected_sha256 for asset in cot)
    assert {Path(asset.destination).name for asset in cot} == {
        f"{category}_cot_val.json" for category in ("dress", "shirt", "toptee")}


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
    # Every required CSMCIR asset now has a verified acquisition source, so nothing is reported
    # blocked: the COT_ours2 captions are published by the author Hub and are fetched to the
    # in-checkout path the evaluator opens.
    assert "required asset has no verified automatic acquisition source" not in output
    assert str(config.WORKBENCH_THIRD_PARTY_ROOT / "CSMCIR" / "COT_ours2" / "fashioniq") in output
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

    Ownership comes from the download manifest, which only ever records a file the workbench itself
    fetched. The recorded digest is what makes corruption detectable: the file on disk no longer
    matches what was verified at acquisition.
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

    # The workbench downloaded these, so the manifest records them and their verified digest; the
    # bytes on disk are then corrupted, which is exactly the case that must be repaired.
    manifest = module.manifest_path(config.WORKBENCH_ARTIFACTS_ROOT)
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"corrupt truncated content")
    module.save_manifest(manifest, [
        {"destination": str(asset.destination), "downloaded_sha256": digest,
         "model_id": "csmcir", "asset_family": "Qwen captions",
         "filename": asset.destination.name, "source_url": asset.source_url,
         "source_revision": None, "size_bytes": len(good), "downloaded_at": "2026-01-01T00:00:00Z",
         "official_hash_available": True, "official_sha256": digest}
        for asset in items])

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

def test_a_managed_download_is_repaired_even_without_a_published_hash(monkeypatch, tmp_path: Path) -> None:
    """AUX-2: a managed download stays managed even when the author published no hash.

    Tying repairability to the registry's `provenance_status` meant a corrupt file the workbench
    itself downloaded was kept forever whenever the upstream provenance was not VERIFIED — the
    operator had to notice and pass --force-redownload. The manifest digest is enough.
    """
    from dataclasses import replace

    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config, with_hash=False)
    good = b"verified managed bytes"
    digest = hashlib.sha256(good).hexdigest()
    # No published hash, and unverified upstream provenance: exactly the old blind spot.
    monkeypatch.setattr(module, "assets",
                        lambda *a, **k: tuple(replace(asset, expected_sha256=None,
                                                      provenance_status="UNVERIFIED") for asset in items))
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(b"corrupt")
    module.save_manifest(module.manifest_path(config.WORKBENCH_ARTIFACTS_ROOT), [
        {"destination": str(asset.destination), "downloaded_sha256": digest,
         "model_id": "csmcir", "asset_family": "Qwen captions",
         "filename": asset.destination.name, "source_url": asset.source_url,
         "source_revision": None, "size_bytes": len(good), "downloaded_at": "2026-01-01T00:00:00Z",
         "official_hash_available": False, "official_sha256": None}
        for asset in items])

    calls: list[str] = []
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda url: (calls.append(url), Response(good))[1])
    assert module.main(["--model", "csmcir"]) == 0
    assert calls == [asset.source_url for asset in items], "a corrupt managed file must be re-fetched"
    assert all(asset.destination.read_bytes() == good for asset in items)

def test_an_operator_supplied_asset_is_never_replaced(monkeypatch, tmp_path: Path) -> None:
    """AUX-1: a file the workbench never downloaded is the operator's, whatever the registry says.

    Ownership is decided by the manifest, not by `provenance_status`: an asset the registry calls
    VERIFIED but which has no manifest record was placed by the operator, so its bytes must survive
    even when they differ from every hash the registry declares.
    """
    from dataclasses import replace

    module = load_module()
    config = config_at(tmp_path)
    items = qwen_assets(module, config, with_hash=False)
    operator_bytes = b"operator bytes that will not match any hash"
    monkeypatch.setattr(module, "assets",
                        lambda *a, **k: tuple(replace(asset, expected_sha256="0" * 64,
                                                      provenance_status="VERIFIED") for asset in items))
    monkeypatch.setattr(module, "resolve_config", lambda: config)
    for asset in items:
        asset.destination.parent.mkdir(parents=True, exist_ok=True)
        asset.destination.write_bytes(operator_bytes)
    # No manifest: the workbench never fetched these.
    assert not module.manifest_path(config.WORKBENCH_ARTIFACTS_ROOT).exists()
    monkeypatch.setattr(module.urllib.request, "urlopen",
                        lambda _: (_ for _ in ()).throw(AssertionError("network used")))

    assert module.main(["--model", "csmcir"]) == 0
    assert all(asset.destination.read_bytes() == operator_bytes for asset in items), \
        "an operator's file must never be replaced without explicit force"


def test_openai_clip_weights_ignore_xdg_cache_home(monkeypatch, tmp_path: Path) -> None:
    """FA-05: the staged location must match the CONSUMER, not a convention.

    OpenAI's CLIP resolves its cache as `download_root or os.path.expanduser("~/.cache/clip")` and
    never reads XDG_CACHE_HOME (openai/CLIP `clip/clip.py:120`). Treating RN50x4 as an
    environment-driven cache staged it outside the operator's real cache whenever XDG_CACHE_HOME was
    set, so the evaluator would not find the weights and would fetch them itself.
    """
    from pathlib import Path as _Path

    from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination

    # A commandeer XDG_CACHE_HOME must not move these weights.
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "somewhere-else"))
    source_root = tmp_path / "source"
    fashioniq_root = tmp_path / "fashioniq"

    for model_id in ("combiner_rn50x4_noft", "clip4cir_rn50x4_fullft"):
        asset = next(item for item in auxiliary_assets_for_model(model_id)
                     if "RN50x4" in item["family"])
        resolved = auxiliary_destination(asset, source_root, fashioniq_root)
        assert resolved == _Path("~/.cache/clip/RN50x4.pt").expanduser(), (
            f"{model_id}: RN50x4 must land in OpenAI CLIP's own cache")
        assert not resolved.is_relative_to(tmp_path / "somewhere-else")

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
        # The hub REPO DIRECTORY, not a file inside it: `from_pretrained` resolves the repo id to
        # `hub/models--bert-base-uncased/` and reads `refs/` and `snapshots/` below that, so writing
        # a bare `vocab.txt` at this path produced a cache the loader never opens. A configured
        # `HF_HOME` is the cache ROOT; the repo directory is one level down and includes `hub/`.
        "bert-base-uncased tokenizer": tmp_path / "hf/hub/models--bert-base-uncased",
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


def _alias_fixture(tmp_path: Path, monkeypatch) -> tuple:
    """A config whose third-party root, dataset root and artifacts root are all throwaway.

    The published Qwen caption is written into the dataset root with the digest the registry
    declares, so alias staging has a real file to copy.

    The published files are 2-5 MB with thousands of records, so the fixture substitutes a
    three-record body and restates the asset's declared content contract and alias digest to match
    it. That contract lives in the registry, which both the downloader and preflight re-read on every
    call, so the fixture patches the loader they share rather than a copy — otherwise preflight would
    load the real 2.8 MB digest and reject the fixture's bytes, and the tests would be exercising a
    registry state no run ever sees.
    """
    from dataclasses import replace
    from workbench.backend import registry as registry_module
    from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination
    from workbench.backend.operator_config import resolve_config

    config = replace(
        resolve_config(),
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
        FASHIONIQ_ROOT=tmp_path / "FashionIQ",
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
    )
    (config.FASHIONIQ_ROOT / "qwen_captions").mkdir(parents=True)
    (config.FASHIONIQ_ROOT / "image_splits").mkdir(parents=True, exist_ok=True)
    body = json.dumps({f"ID{i}": {"Final_Caption": f"caption {i}"} for i in range(3)}).encode()
    digest = hashlib.sha256(body).hexdigest()
    fixture_ids = [f"ID{i}" for i in range(3)]

    patched = []
    for asset in auxiliary_assets_for_model("csmcir"):
        asset = dict(asset)
        if asset["family"] == "Qwen captions":
            assert asset["expected_sha256"]  # the fixture relies on a real declared digest
            asset["expected_sha256"] = digest
            asset["expected_size_bytes"] = len(body)
            (config.FASHIONIQ_ROOT / "qwen_captions" / Path(asset["destination_path"]).name).write_bytes(body)
        elif asset["family"] == "COT_ours2 captions":
            category = Path(asset["destination_path"]).name.split("_")[0]
            id_source = f"image_splits/split.{category}.val.json"
            (config.FASHIONIQ_ROOT / id_source).write_text(json.dumps(fixture_ids))
            asset["content_contract"] = {
                "record_key_fields": ["Final_Caption"],
                "expected_record_count": 3,
                "key_kind": "fashioniq_image_id",
                "id_source": id_source,
            }
            # The COT_ours2 destination is a *published* asset URL again (the author Hub serves the
            # same file under `fashioniq_qwen_captions/qwen_captions/`), while these tests exercise
            # the *alias* path: staging a published asset at a destination whose own URL is
            # unobtainable. The fixture therefore restates the record as alias-shaped, which is the
            # only way to reach `stage_caption_alias` now that the registry no longer uses that shape
            # for COT_ours2. `caption_alias_from` itself is still a live, exercised code path — see
            # `stage_caption_alias` in download_auxiliary_assets.py.
            alias_source = asset["source_path"]
            asset["source_path"] = None
            asset["source_repo"] = None
            asset["source_revision"] = None
            asset["automatic_download_available"] = False
            asset["provenance_status"] = "UNVERIFIED"
            asset["caption_alias_from"] = {
                "source_path": alias_source,
                "source_revision": "cf0c19bb346266c295b4b5772ebf03bd1c4f0467",
                "expected_sha256": digest,
                "expected_size_bytes": len(body),
            }
        patched.append(asset)

    monkeypatch.setattr(registry_module, "load_auxiliary_assets", lambda *a, **k: patched)
    cot = [asset for asset in patched if asset["family"] == "COT_ours2 captions"]
    return config, cot


def test_caption_alias_stages_a_copy_never_a_symlink(tmp_path: Path, monkeypatch) -> None:
    """The alias must be a real file inside the checkout: the runtime rejects a symlink there."""
    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)

    for asset in cot:
        staged = module.Asset(
            asset["model_id"], asset["family"], Path(asset["destination_path"]).name,
            module.auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT),
            True, asset["automatic_download_available"], None, None, None,
            asset["provenance_status"], None, None, asset.get("content_contract"),
            asset["caption_alias_from"],
        )
        ok, message = module.stage_caption_alias(staged, fashioniq_root=config.FASHIONIQ_ROOT)
        assert ok, message
        assert staged.destination.is_file()
        assert not staged.destination.is_symlink()
        assert staged.destination.is_relative_to(config.WORKBENCH_THIRD_PARTY_ROOT)


def test_caption_alias_refuses_a_verified_asset(tmp_path: Path, monkeypatch) -> None:
    """A checksum-verified asset is never replaced by an alias."""
    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = dict(cot[0])
    asset["provenance_status"] = "VERIFIED"
    staged = module.Asset(
        asset["model_id"], asset["family"], Path(asset["destination_path"]).name,
        module.auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT),
        True, False, None, None, None, "VERIFIED", None, None, None, asset["caption_alias_from"])
    ok, message = module.stage_caption_alias(staged, fashioniq_root=config.FASHIONIQ_ROOT)
    assert not ok
    assert "checksum-verified" in message
    assert not staged.destination.exists()


def test_caption_alias_refuses_bytes_that_are_not_the_published_file(tmp_path: Path, monkeypatch) -> None:
    """If the staged Qwen caption is not the declared published file, nothing is written."""
    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    published = config.FASHIONIQ_ROOT / "qwen_captions" / Path(asset["caption_alias_from"]["source_path"]).name
    published.write_bytes(b'{"X": {"Final_Caption": "not the published file"}}')
    staged = module.Asset(
        asset["model_id"], asset["family"], Path(asset["destination_path"]).name,
        module.auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT),
        True, False, None, None, None, asset["provenance_status"], None, None, None,
        asset["caption_alias_from"])
    ok, message = module.stage_caption_alias(staged, fashioniq_root=config.FASHIONIQ_ROOT)
    assert not ok
    assert "not the declared published file" in message
    assert not staged.destination.exists()


def test_preflight_accepts_a_manifest_recorded_alias_but_not_a_stray_file(tmp_path: Path, monkeypatch) -> None:
    """Only bytes the workbench staged as an alias are accepted; anything else stays a blocker."""
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.preflight import model_requirements
    from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination, load_registry

    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    destination = auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    published = (config.FASHIONIQ_ROOT / "qwen_captions"
                 / Path(asset["caption_alias_from"]["source_path"]).name).read_bytes()

    model = next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")
    cfg = WorkbenchConfig(
        config.FASHIONIQ_ROOT, config.FASHIONIQ_ROOT, config.FASHIONIQ_ROOT,
        "127.0.0.1", 8000, 5173, tmp_path / "ckpt", tmp_path / "results",
        config.WORKBENCH_THIRD_PARTY_ROOT, config.WORKBENCH_ARTIFACTS_ROOT)
    manifest = config.WORKBENCH_ARTIFACTS_ROOT / "auxiliary" / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    # A stray operator file, not in the manifest: it stays a blocker.
    destination.write_bytes(b'{"ID0": {"Final_Caption": "operator own"}}')
    stray = model_requirements(model, cfg)
    assert any(str(destination) in block or "no verified automatic acquisition" in block
               for block in stray["blocks"]), stray["blocks"]
    assert not stray["warnings"] or all("alias" not in w for w in stray["warnings"])

    # A manifest entry whose bytes are not the published file is still a blocker.
    destination.write_text(json.dumps({f"ID{i}": {"Final_Caption": f"not published {i}"} for i in range(3)}),
                           encoding="utf-8")
    manifest.write_text(json.dumps([{
        "destination": str(destination),
        "downloaded_sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "staged_as": "SOURCE_COMPATIBLE_CAPTION_ALIAS",
        "source_revision": asset["caption_alias_from"]["source_revision"],
        "alias_source_path": asset["caption_alias_from"]["source_path"],
    }]))
    forged = model_requirements(model, cfg)
    assert any(str(destination) in block for block in forged["blocks"]), forged["blocks"]

    # Recorded the way the staging path does, with the published bytes on disk: accepted, as a variant.
    destination.write_bytes(published)
    manifest.write_text(json.dumps([{
        "destination": str(destination),
        "downloaded_sha256": hashlib.sha256(published).hexdigest(),
        "staged_as": "SOURCE_COMPATIBLE_CAPTION_ALIAS",
        "source_revision": asset["caption_alias_from"]["source_revision"],
        "alias_source_path": asset["caption_alias_from"]["source_path"],
    }]))
    accepted = model_requirements(model, cfg)
    assert not any(str(destination) in block for block in accepted["blocks"]), accepted["blocks"]
    summary = [w for w in accepted["warnings"] if "source-compatible caption alias" in w]
    assert len(summary) == 1, accepted["warnings"]


def test_caption_alias_refuses_to_overwrite_an_operator_owned_file(tmp_path: Path, monkeypatch) -> None:
    """Asking for an alias does not authorize destroying data someone else put there.

    Ownership is decided by the download manifest, not by the filename: a file the manifest does not
    attribute to this destination is the operator's, and staging must refuse rather than replace it.
    """
    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    destination = module.auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    operator_bytes = b'{"OPERATOR": {"Final_Caption": "hand-made, do not touch"}}'
    destination.write_bytes(operator_bytes)

    staged = module.Asset(
        asset["model_id"], asset["family"], Path(asset["destination_path"]).name, destination,
        True, False, None, None, None, asset["provenance_status"], None, None, None,
        asset["caption_alias_from"])
    ok, message = module.stage_caption_alias(staged, fashioniq_root=config.FASHIONIQ_ROOT, managed={})
    assert not ok
    assert "operator-owned" in message
    assert destination.read_bytes() == operator_bytes, "the operator's file must survive the refusal"


def test_caption_alias_may_replace_a_file_the_manifest_attributes_to_it(tmp_path: Path, monkeypatch) -> None:
    """A file the workbench itself staged is fair game, so the alias stays idempotent."""
    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    destination = module.auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b'{"STALE": {"Final_Caption": "an older managed copy"}}')

    staged = module.Asset(
        asset["model_id"], asset["family"], Path(asset["destination_path"]).name, destination,
        True, False, None, None, None, asset["provenance_status"], None, None, None,
        asset["caption_alias_from"])
    ok, message = module.stage_caption_alias(
        staged, fashioniq_root=config.FASHIONIQ_ROOT, managed={str(destination): "whatever-was-recorded"})
    assert ok, message
    assert b"STALE" not in destination.read_bytes()


def test_caption_alias_refuses_a_destination_symlink(tmp_path: Path, monkeypatch) -> None:
    """A symlink at the destination is never followed, and its target stays untouched."""
    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    destination = module.auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_bytes(b'{"ELSEWHERE": {"Final_Caption": "do not follow me"}}')
    destination.symlink_to(elsewhere)

    staged = module.Asset(
        asset["model_id"], asset["family"], Path(asset["destination_path"]).name, destination,
        True, False, None, None, None, asset["provenance_status"], None, None, None,
        asset["caption_alias_from"])
    ok, message = module.stage_caption_alias(
        staged, fashioniq_root=config.FASHIONIQ_ROOT, managed={str(destination): "managed"})
    assert not ok
    assert "symlink" in message
    assert destination.is_symlink()
    assert elsewhere.read_bytes() == b'{"ELSEWHERE": {"Final_Caption": "do not follow me"}}'


def test_caption_alias_rejects_a_stand_in_that_cannot_serve_as_the_asset(tmp_path: Path, monkeypatch) -> None:
    """A published file that cannot cover the declared split is refused even though its digest is right."""
    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    alias = dict(asset["caption_alias_from"])
    published = config.FASHIONIQ_ROOT / "qwen_captions" / Path(alias["source_path"]).name
    # Valid JSON, correct digest, but no Final_Caption anywhere: the evaluator would read nothing.
    body = json.dumps({f"ID{i}": {"Core_Essence": "no final caption"} for i in range(3)}).encode()
    published.write_bytes(body)
    alias["expected_sha256"] = hashlib.sha256(body).hexdigest()

    staged = module.Asset(
        asset["model_id"], asset["family"], Path(asset["destination_path"]).name,
        module.auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT),
        True, False, None, None, None, asset["provenance_status"], None, None, None, alias)
    ok, message = module.stage_caption_alias(staged, fashioniq_root=config.FASHIONIQ_ROOT)
    assert not ok
    assert "unusable here" in message
    assert not staged.destination.exists()


def test_preflight_refuses_a_forged_manifest_that_matches_its_own_bytes(tmp_path: Path, monkeypatch) -> None:
    """A hand-written manifest cannot relabel arbitrary bytes as a variant.

    The manifest proves the workbench wrote the bytes; it cannot prove they are the published file.
    Only the registry's declared published digest can, so a file whose digest merely matches a
    self-consistent manifest entry is still a blocker.
    """
    from workbench.backend.registry import auxiliary_destination, caption_alias_evidence

    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    destination = auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b'{"FORGED": {"Final_Caption": "attacker-chosen bytes"}}')
    forged = hashlib.sha256(destination.read_bytes()).hexdigest()

    manifest = config.WORKBENCH_ARTIFACTS_ROOT / "auxiliary" / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([{
        "destination": str(destination), "downloaded_sha256": forged,
        "staged_as": "SOURCE_COMPATIBLE_CAPTION_ALIAS",
        "alias_source_path": asset["caption_alias_from"]["source_path"],
        "source_revision": asset["caption_alias_from"]["source_revision"],
    }]))

    assert caption_alias_evidence(asset, destination, config) is None


def test_preflight_accepts_a_manifest_alias_whose_bytes_are_the_published_file(tmp_path: Path, monkeypatch) -> None:
    """The honest path still works: manifest record plus the registered published digest."""
    from workbench.backend.registry import auxiliary_destination, caption_alias_evidence

    module = load_module()
    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    published_digest = asset["caption_alias_from"]["expected_sha256"]
    destination = auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    published_bytes = (config.FASHIONIQ_ROOT / "qwen_captions"
                       / Path(asset["caption_alias_from"]["source_path"]).name).read_bytes()
    real = hashlib.sha256(published_bytes).hexdigest()
    assert real == published_digest
    destination.write_bytes(published_bytes)
    manifest = config.WORKBENCH_ARTIFACTS_ROOT / "auxiliary" / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    # Bytes that are not the published file: refused.
    destination.write_text(json.dumps({f"ID{i}": {"Final_Caption": f"different caption {i}"} for i in range(3)}),
                           encoding="utf-8")
    different = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest.write_text(json.dumps([{
        "destination": str(destination), "downloaded_sha256": different,
        "staged_as": "SOURCE_COMPATIBLE_CAPTION_ALIAS",
    }]))
    assert different != published_digest
    assert caption_alias_evidence(asset, destination, config) is None
    # Bytes that are: accepted, and tied to the published digest.
    destination.write_bytes(published_bytes)
    manifest.write_text(json.dumps([{
        "destination": str(destination), "downloaded_sha256": real,
        "staged_as": "SOURCE_COMPATIBLE_CAPTION_ALIAS",
    }]))
    evidence = caption_alias_evidence(asset, destination, config)
    assert evidence is not None
    assert evidence["published_sha256"] == published_digest
    assert evidence["equivalence"] == "SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN"


def test_preflight_rejects_a_manifest_alias_whose_bytes_changed(tmp_path: Path, monkeypatch) -> None:
    """A managed alias that was edited afterwards stops being a variant and blocks again."""
    from workbench.backend.preflight import model_requirements
    from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination, load_registry

    config, cot = _alias_fixture(tmp_path, monkeypatch)
    asset = cot[0]
    destination = auxiliary_destination(asset, config.WORKBENCH_THIRD_PARTY_ROOT, config.FASHIONIQ_ROOT)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b'{"ID0": {"Final_Caption": "as staged"}}')
    good = hashlib.sha256(destination.read_bytes()).hexdigest()
    destination.write_bytes(b'{"ID0": {"Final_Caption": "edited afterwards"}}')

    manifest = config.WORKBENCH_ARTIFACTS_ROOT / "auxiliary" / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps([{
        "destination": str(destination), "downloaded_sha256": good,
        "staged_as": "SOURCE_COMPATIBLE_CAPTION_ALIAS",
    }]))

    model = next(m for m in load_registry()["models"] if m["model_id"] == "csmcir")
    cfg = WorkbenchConfig(
        config.FASHIONIQ_ROOT, config.FASHIONIQ_ROOT, config.FASHIONIQ_ROOT,
        "127.0.0.1", 8000, 5173, tmp_path / "ckpt", tmp_path / "results",
        config.WORKBENCH_THIRD_PARTY_ROOT, config.WORKBENCH_ARTIFACTS_ROOT)
    reqs = model_requirements(model, cfg)
    assert any("COT_ours2 captions" in block for block in reqs["blocks"]), reqs["blocks"]
    assert not any("source-compatible caption alias" in w for w in reqs["warnings"])

def test_tme_auxiliary_destinations_match_preparation_contract(tmp_path: Path, monkeypatch) -> None:
    from dataclasses import replace
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import (
        auxiliary_assets_for_model, auxiliary_destination, external_asset_statuses,
        load_registry, preparation_contract_for_model,
    )

    third_party = tmp_path / "third_party"
    monkeypatch.setattr(
        "workbench.backend.operator_config.resolve_config",
        lambda: replace(resolve_config(), WORKBENCH_THIRD_PARTY_ROOT=third_party),
    )
    model = next(model for model in load_registry()["models"] if model["model_id"] == "tme")
    contract_assets = preparation_contract_for_model(model)["external_assets"]
    statuses = {status["name"]: Path(status["path"]) for status in external_asset_statuses(model)}
    registered = auxiliary_assets_for_model("tme")
    # Two CHECKOUT-RELATIVE assets (the base weights the pinned loader opens by literal cwd path)
    # plus the Hugging Face hub cache, whose destination is the repo directory `from_pretrained`
    # resolves from the repo id rather than a file this contract names.
    checkout_assets = [asset for asset in registered if not asset.get("hub_repo_id")]
    hub_assets = [asset for asset in registered if asset.get("hub_repo_id")]
    assert len(checkout_assets) == 2
    assert len(hub_assets) == 1

    for asset in checkout_assets:
        contract_asset = next(item for item in contract_assets if item["name"] == asset["family"])
        expected = (third_party / model["source_dir"] / contract_asset["expected_local_location"]).resolve()
        destination = auxiliary_destination(asset, third_party, tmp_path / "FashionIQ")
        assert destination == expected
        assert statuses[contract_asset["name"]] == expected

    # The hub cache destination must be the directory the loader resolves AND the directory the
    # checker inspects: one resolver answers both, so a cache staged at one level cannot be
    # reported missing while the loader reads another.
    from workbench.backend.registry import hf_hub_repo_dir

    # The auxiliary registry names the family `bert-base-uncased Qformer payload` while the contract
    # keeps `bert-base-uncased tokenizer`, because that string is the KEY this asset takes inside the
    # TME precompute receipt's `base_assets` map. Renaming the contract entry to match a label would
    # invalidate a valid feature cache without any feature input having changed, so the two are
    # joined by what they point at -- the hub repo id -- rather than by string equality.
    hub_asset = hub_assets[0]
    contract_asset = next(item for item in contract_assets
                          if hub_asset["hub_repo_id"] in str(item.get("cache_path_suffix") or ""))
    destination = auxiliary_destination(hub_asset, third_party, tmp_path / "FashionIQ")
    assert destination == hf_hub_repo_dir(hub_asset["hub_repo_id"])
    assert statuses[contract_asset["name"]] == destination


def test_tme_model_source_destination_cannot_escape_checkout(tmp_path: Path) -> None:
    from workbench.backend.registry import auxiliary_assets_for_model, auxiliary_destination

    asset = dict(auxiliary_assets_for_model("tme")[0], destination_path="../outside.pth")
    with pytest.raises(ValueError, match="escapes root") as error:
        auxiliary_destination(asset, tmp_path / "third_party", tmp_path / "FashionIQ")
    assert str(tmp_path / "third_party" / "outside.pth") in str(error.value)


def test_tme_verify_reports_expected_size_and_rejects_wrong_size(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    config = config_at(tmp_path)
    item = next(asset for asset in module.assets({"tme"}, config.WORKBENCH_THIRD_PARTY_ROOT,
                                                 config.FASHIONIQ_ROOT)
                if asset.family == "BLIP-2 pretrain weights")
    monkeypatch.setattr(module, "sha256_file", lambda _: "a" * 64)
    item.destination.parent.mkdir(parents=True)
    item.destination.touch()
    with item.destination.open("r+b") as handle:
        handle.truncate(item.expected_size_bytes)

    assert not module.verify(item)
    output = capsys.readouterr().out
    assert "UNRESOLVED REQUIRED PROVENANCE" in output
    assert "expected " not in output

    with item.destination.open("r+b") as handle:
        handle.truncate(item.expected_size_bytes - 1)
    assert not module.verify(item)
    assert f"expected {item.expected_size_bytes} bytes" in capsys.readouterr().out


def test_auxiliary_model_ids_include_tme() -> None:
    from workbench.backend.registry import auxiliary_model_ids

    assert "tme" in auxiliary_model_ids()


def test_tme_hub_cache_offline_verification_and_pipeline_selection(monkeypatch, tmp_path: Path, capsys) -> None:
    module = load_module()
    from types import SimpleNamespace
    from workbench.backend import runtime
    from workbench.backend.operator_config import resolve_config
    from workbench.scripts import pipeline

    hf_home = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(module, "sha256_file", lambda _: "f" * 64)
    monkeypatch.setattr(module, "bootstrap_hub_asset", lambda *_: (_ for _ in ()).throw(AssertionError("unexpected network bootstrap")))
    monkeypatch.setattr(runtime, "verify_environment", lambda *a, **kw: SimpleNamespace())
    monkeypatch.setattr(runtime, "_cached_gpu_capability", lambda *a, **kw: None)
    monkeypatch.setattr("workbench.scripts.download_checkpoints.verify", lambda *a, **kw: True)

    repo_dir = hf_home / "hub" / "models--bert-base-uncased"

    # (a) missing / orphaned cache refused by --verify-only and reported with destination=
    assert module.main(["--model", "tme", "--verify-only"]) == 1
    out_missing = capsys.readouterr().out
    assert f"bert-base-uncased Qformer payload: destination={repo_dir}" in out_missing
    assert "no hub cache directory at" in out_missing

    (repo_dir / "refs").mkdir(parents=True)
    assert module.main(["--model", "tme", "--verify-only"]) == 1
    out_orphaned = capsys.readouterr().out
    assert f"bert-base-uncased Qformer payload: destination={repo_dir}" in out_orphaned
    assert "holds no active revision under refs/" in out_orphaned

    # (b) otherwise-valid but ref-less snapshot (files present, no refs/<rev>) is still refused
    (repo_dir / "refs").rmdir()
    snapshot_dir = repo_dir / "snapshots" / "rev123"
    snapshot_dir.mkdir(parents=True)
    (snapshot_dir / "pytorch_model.bin").write_bytes(b"dummy payload")
    assert module.main(["--model", "tme", "--verify-only"]) == 1
    out_ref_less = capsys.readouterr().out
    assert "holds no active revision under refs/" in out_ref_less

    # (c) cache with active ref and payload is accepted. The DECLARED payload, not one file: the
    # selection decision now asks whether every file the loader opens is present, so a fixture that
    # writes only `pytorch_model.bin` models a tokenizer-less cache -- which the pinned constructor
    # would fail on -- and must not be presented as a complete one.
    refs_dir = repo_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("rev123\n", encoding="utf-8")
    tme_asset = next(item for item in module.assets({"tme"}, Path("/tmp"), Path("/tmp"))
                     if item.hub_repo_id)
    for name in tme_asset.hub_files or ():
        (snapshot_dir / name).write_bytes(b"dummy payload")
    assert module.main(["--model", "tme", "--verify-only"]) == 0
    out_valid = capsys.readouterr().out
    assert "cache layout and declared payload verified" in out_valid
    assert "state=present" in out_valid

    # (d) the same cache missing ONE declared file is refused, and no acquisition is attempted:
    # a verification that accepted it would have told the operator the asset is ready while the
    # download path considered it owed.
    (snapshot_dir / "config.json").unlink()
    assert module.main(["--model", "tme", "--verify-only"]) == 1
    out_partial = capsys.readouterr().out
    assert "declared payload missing: config.json" in out_partial
    (snapshot_dir / "config.json").write_bytes(b"dummy payload")

    # (e) the stage command is STABLE across states, so a successful stage keeps its fingerprint and
    # a resume does not re-run it; only the capability depends on work actually being owed.
    cfg = resolve_config()
    args = pipeline.parser_for().parse_args(["all", "--model", "tme"])

    def aux_stage():
        return next(s for s in pipeline.all_stages(args, cfg)
                    if s.name == "auxiliary-assets" and s.model_id == "tme")

    # With the file-shaped assets staged but the cache pointed at an EMPTY root, work IS owed: the
    # command stays the same and the capability says the run needs a download.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    owed_stage = aux_stage()
    assert owed_stage.required_capabilities == frozenset({"allow_network", "allow_large_downloads"})

    # With everything staged, the COMMAND is byte-identical (the fingerprint covers it, so a state
    # change must not change it) and no capability is required.
    monkeypatch.setenv("HF_HOME", str(hf_home))
    staged_stage = aux_stage()
    assert staged_stage.commands == owed_stage.commands, (
        "the auxiliary command must not depend on staging state: it is part of the stage "
        "fingerprint, and a fingerprint that changes when the stage succeeds refuses its own resume")
    assert staged_stage.required_capabilities == frozenset()


def test_hub_bootstrap_publishes_the_ref_the_loader_resolves(monkeypatch, tmp_path: Path) -> None:
    """A bootstrap must produce a cache its own verification accepts, not only the bytes.

    Reproduced against `huggingface_hub`: `snapshot_download(revision=<sha>)` writes
    `snapshots/<sha>/` and NO `refs/<rev>` entry, and a snapshot no ref names cannot be reached by
    the repo-id lookup the pinned loader performs. The first real bootstrap therefore staged 421 MB
    of correct files and was then refused by the same function that staged them -- a software gap
    that would have read as a missing author artifact. The ref is what makes the two ends agree.
    """
    import huggingface_hub

    module = load_module()
    hf_home = tmp_path / "hf_home"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    revision = "86b5e0934494bd15c9632b12f734a8a67f723594"
    files = ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    repo_dir = hf_home / "hub" / "models--bert-base-uncased"
    snapshot_dir = repo_dir / "snapshots" / revision

    def fake_snapshot_download(*, repo_id, revision=None, allow_patterns=None, **kwargs):
        # The real library's observable effect for this call: bytes in `snapshots/<sha>/`, and no
        # ref. Writing exactly that is what makes this test fail if the bootstrap stops writing one.
        snapshot_dir.mkdir(parents=True)
        for name in allow_patterns or files:
            (snapshot_dir / name).write_bytes(b"payload " + name.encode())
        return str(snapshot_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    asset = module.Asset(
        model_id="tme", family="bert-base-uncased Qformer payload",
        filename="config.json", destination=repo_dir, required=True,
        automatic_download_available=True, source_url=None, source_revision=revision,
        expected_sha256=None, provenance_status="VERIFIED", expected_size_bytes=None,
        hub_repo_id="bert-base-uncased", hub_files=tuple(files))

    ok, message = module.bootstrap_hub_asset(asset)
    assert ok, message
    assert (repo_dir / "refs" / "main").read_text(encoding="utf-8") == revision
    from workbench.backend.registry import hub_cache_layout_problem

    assert hub_cache_layout_problem("bert-base-uncased", hf_home) is None
    assert module.hub_cache_layout_problem("bert-base-uncased") is None
