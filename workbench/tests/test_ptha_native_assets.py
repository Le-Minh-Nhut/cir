"""Regression tests for PTHA reachable native assets declaration."""
from __future__ import annotations

import os
from pathlib import Path
import pytest

from workbench.backend.registry import (
    auxiliary_assets_for_model,
    auxiliary_destination,
    external_asset_blockers,
    external_asset_statuses,
    hf_hub_repo_dir,
    model_by_id,
    required_runtime_assets,
)
from workbench.backend.autopilot import _enforce_download_budget, AutopilotConfig
from workbench.replay.ptha_mtst import preflight


@pytest.mark.parametrize("model_id", ["sprc", "airknow", "csmcir", "dcnet", "clvc_net", "ptha_mtst"])
def test_digestless_file_admission_matrix(tmp_path, model_id):
    from workbench.backend.registry import asset_admission_status, load_preparation_contracts

    model = model_by_id(model_id)
    contract = load_preparation_contracts().get(model.get("preparation_id"), {})
    declarations = [*contract.get("external_assets", []), *auxiliary_assets_for_model(model_id)]
    digestless = [asset for asset in declarations
                  if not asset.get("expected_sha256") and not asset.get("expected_sha256_prefix")
                  and not asset.get("hub_repo_id") and asset.get("location_type") != "directory"]
    if not digestless:
        assert model_id == "clvc_net"
        return

    for index, asset in enumerate(digestless):
        file_asset = {**asset, "destination_path": "asset.bin", "expected_size_bytes": 7}
        path = tmp_path / f"{index}.bin"
        assert asset_admission_status(file_asset, {"status": "CONFIGURED_MISSING", "path": str(path)}) == "CONFIGURED_MISSING"
        status = {"status": "EXISTS_UNVERIFIED_CHECKSUM", "path": str(path)}
        path.write_bytes(b"weights")
        assert asset_admission_status(file_asset, status) == "LOCALLY_RECORDED_ASSET"
        path.write_bytes(b"")
        assert asset_admission_status(file_asset, status) == "EMPTY_PLACEHOLDER"
        path.write_bytes(b"wrong-size")
        assert asset_admission_status(file_asset, status) == "SIZE_MISMATCH"



def test_digestless_size_match_rejects_zero_fill_and_accepts_plausible_large_file(tmp_path):
    from workbench.backend.registry import asset_admission_status, preparation_contract_for_model

    contract = preparation_contract_for_model(model_by_id("ptha_mtst"))
    asset = next(a for a in contract["external_assets"] if a["name"] == "EVA-ViT-g backbone weights")
    path = tmp_path / "eva_vit_g.pth"
    asset = {**asset, "expected_size_bytes": 2 * 1024 * 1024}
    status = {"status": "EXISTS_UNVERIFIED_CHECKSUM", "path": str(path)}

    with path.open("wb") as handle:
        handle.truncate(asset["expected_size_bytes"])
    assert asset_admission_status(asset, status) == "SUSPECT_PAYLOAD"

    with path.open("r+b") as handle:
        handle.seek(asset["expected_size_bytes"] - len(b"weights"))
        handle.write(b"weights")
    assert asset_admission_status(asset, status) == "LOCALLY_RECORDED_ASSET"


def test_clvc_net_digest_prefix_asset_not_treated_as_digestless():
    from workbench.backend.registry import preparation_contract_for_model

    asset = preparation_contract_for_model(model_by_id("clvc_net"))["external_assets"][0]
    assert asset["expected_sha256_prefix"] == "19c8e357"

def test_ptha_mtst_auxiliary_assets_declared():
    assets = auxiliary_assets_for_model("ptha_mtst")
    assert len(assets) == 2, f"expected 2 auxiliary assets, found {len(assets)}"
    families = {a["family"]: a for a in assets}

    eva = families["EVA-ViT-g backbone weights"]
    assert eva["destination_scope"] == "user_cache"
    assert eva["destination_path"] == "hub/checkpoints/eva_vit_g.pth"
    assert eva["cache_root_env_var"] == "TORCH_HOME"
    assert eva["expected_size_bytes"] == 2025249237
    assert eva["automatic_download_available"] is True
    assert eva["provenance_status"] == "UNVERIFIED"

    bert = families["bert-base-uncased tokenizer"]
    assert bert["destination_scope"] == "user_cache"
    assert bert["destination_path"] == "hub/models--bert-base-uncased"
    assert bert["cache_root_env_var"] == "HF_HOME"
    assert bert["hub_repo_id"] == "bert-base-uncased"
    assert bert["hub_files"] == ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]
    assert bert["expected_size_bytes"] == 441171321
    assert bert["automatic_download_available"] is True
    assert bert["provenance_status"] == "VERIFIED"


def test_ptha_mtst_required_runtime_assets_declared():
    m = model_by_id("ptha_mtst")
    assets = required_runtime_assets(m)
    names = {a["name"] for a in assets}
    assert "EVA-ViT-g backbone weights" in names
    assert "bert-base-uncased tokenizer" in names


def test_ptha_mtst_cold_preflight_blocks_by_name(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf"
    torch_home = tmp_path / "torch"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    m = model_by_id("ptha_mtst")
    blockers = external_asset_blockers(m)
    assert any("EVA-ViT-g backbone weights" in b and "missing" in b for b in blockers)
    assert any("bert-base-uncased tokenizer" in b and "unusable" in b for b in blockers)

    # Replay preflight test
    ckpt_root = tmp_path / "ckpt"
    ckpt_root.mkdir()
    (ckpt_root / "tuned_mtst_cirr.pt").write_bytes(b"dummy_bytes")

    ds_root = tmp_path / "ds"
    for cat in ["dress", "shirt", "toptee"]:
        (ds_root / f"captions/cap.{cat}.val.json").parent.mkdir(parents=True, exist_ok=True)
        (ds_root / f"captions/cap.{cat}.val.json").write_text("[]")
        (ds_root / f"image_splits/split.{cat}.val.json").parent.mkdir(parents=True, exist_ok=True)
        names = ["image_0"] if cat == "dress" else []
        (ds_root / f"image_splits/split.{cat}.val.json").write_text(__import__("json").dumps(names))

    source_root = Path(__file__).resolve().parents[2] / "workbench/third_party/MTST_PTHA"
    (ds_root / "images").mkdir(exist_ok=True)
    replay_blockers_no_images = preflight(source_root, ds_root, ckpt_root, transfer_cirr_to_fashioniq=True)
    assert any("native replay canonical FashionIQ image missing" in b and "image_0.png" in b
               for b in replay_blockers_no_images)

    # model_requirements also names missing raw benchmark input images

    (ds_root / "images" / "image_0.png").write_bytes(b"tiny image")
    replay_blockers = preflight(source_root, ds_root, ckpt_root, transfer_cirr_to_fashioniq=True)
    assert not any("native replay canonical FashionIQ image missing" in b for b in replay_blockers)
    assert not any("FashionIQ raw input missing:" in b and "images" in b for b in replay_blockers)
    assert any("EVA-ViT-g backbone weights" in b for b in replay_blockers)
    assert any("bert-base-uncased tokenizer" in b for b in replay_blockers)


def test_ptha_mtst_budget_charges_cold_and_waives_warm(tmp_path, monkeypatch):
    hf_home = tmp_path / "hf"
    torch_home = tmp_path / "torch"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    m = model_by_id("ptha_mtst")
    row = {"model_id": "ptha_mtst", "status": "NOT_RUN", "checkpoint_ids": []}

    # Cold: total auxiliary = 2,466,420,558 B
    # A 2 GB budget must fail
    cfg_small = AutopilotConfig(
        path=tmp_path / "cfg.yaml",
        selected_model_ids=("ptha_mtst",),
        paths={"checkpoints": tmp_path / "ckpt", "third_party": tmp_path / "tp", "dataset_root": tmp_path / "ds"},
        budgets={"max_download_bytes": 2_000_000_000},
        permissions={"allow_download": True},
        registry={"models": [m]},
    )
    _enforce_download_budget(cfg_small, [row])
    assert row["status"] == "BLOCKED_CHECKPOINT"
    assert any("download budget exceeded: requires 2466420558 bytes" in b for b in row["blockers"])

    # Warm: synthesize minimal hub cache and eva_vit_g.pth
    repo_dir = hf_hub_repo_dir("bert-base-uncased", hf_home)
    snap = repo_dir / "snapshots" / "86b5e0934494bd15c9632b12f734a8a67f723594"
    snap.mkdir(parents=True)
    refs = repo_dir / "refs"
    refs.mkdir(parents=True)
    (refs / "main").write_text("86b5e0934494bd15c9632b12f734a8a67f723594")
    for fname in ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]:
        (snap / fname).write_bytes(b"payload-" + fname.encode())
    payload_size = sum((snap / name).stat().st_size for name in ["config.json", "pytorch_model.bin", "tokenizer.json", "tokenizer_config.json", "vocab.txt"])
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_CONFIG_JSON", str(payload_size))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_EVA_VIT_G_PTH", "5")

    eva_path = torch_home / "hub/checkpoints/eva_vit_g.pth"
    eva_path.parent.mkdir(parents=True)
    eva_path.write_bytes(b"dummy")

    # Warm row with 10 byte budget should pass
    warm_row = {"model_id": "ptha_mtst", "status": "NOT_RUN", "checkpoint_ids": []}
    cfg_tiny = AutopilotConfig(
        path=tmp_path / "cfg.yaml",
        selected_model_ids=("ptha_mtst",),
        paths={"checkpoints": tmp_path / "ckpt", "third_party": tmp_path / "tp", "dataset_root": tmp_path / "ds"},
        budgets={"max_download_bytes": 10},
        permissions={"allow_download": True},
        registry={"models": [m]},
    )
    _enforce_download_budget(cfg_tiny, [warm_row])
    assert warm_row["status"] == "NOT_RUN"  # Passed without blocking!


@pytest.mark.parametrize("model_id,contract_id", [("tme", "tme_fashioniq"), ("ptha_mtst", "ptha_mtst_fashioniq")])
def test_hub_cache_pinned_revisions_match_across_registries(model_id, contract_id, tmp_path, monkeypatch):
    from workbench.backend.registry import (
        load_auxiliary_assets,
        load_preparation_contracts,
        validate_auxiliary_asset,
        cache_asset_blocker,
        external_asset_statuses,
        external_asset_blockers,
        auxiliary_destination,
    )

    contracts = load_preparation_contracts()
    contract = contracts[contract_id]
    aux_assets = [a for a in load_auxiliary_assets() if a["model_id"] == model_id and a.get("hub_repo_id")]
    assert aux_assets, f"no hub auxiliary assets for {model_id}"
    aux_asset = aux_assets[0]

    contract_assets = [a for a in contract["external_assets"] if "hub/models--" in str(a.get("cache_path_suffix") or "")]
    assert contract_assets, f"no hub contract assets for {contract_id}"
    contract_asset = contract_assets[0]

    # 1. Pinned revision must be declared on both sides and match exactly
    assert contract_asset.get("source_revision") is not None
    assert aux_asset.get("source_revision") is not None
    assert contract_asset["source_revision"] == aux_asset["source_revision"]

    # 2. Both consumers agree on unpinned ref mismatch vs valid pin
    hf_home = tmp_path / "hf"
    torch_home = tmp_path / "torch"
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("TORCH_HOME", str(torch_home))

    repo_dir = hf_hub_repo_dir(aux_asset["hub_repo_id"], hf_home)
    refs = repo_dir / "refs"
    refs.mkdir(parents=True)

    # Mismatch: refs/main -> unpinned_sha_1234
    unpinned_snap = repo_dir / "snapshots" / "unpinned_sha_1234"
    unpinned_snap.mkdir(parents=True)
    (refs / "main").write_text("unpinned_sha_1234")
    for fname in aux_asset.get("hub_files", ["config.json"]):
        (unpinned_snap / fname).write_text("dummy")

    dest = auxiliary_destination(aux_asset, tmp_path / "tp", tmp_path / "ds")
    val_bad = validate_auxiliary_asset(aux_asset, dest)
    assert val_bad.ok is False
    assert val_bad.status == "PINNED_REVISION_MISMATCH"

    m = model_by_id(model_id)
    statuses_bad = external_asset_statuses(m)
    hub_status_bad = next(s for s in statuses_bad if s["name"] == contract_asset["name"])
    reason_bad = cache_asset_blocker(contract_asset, hub_status_bad)
    assert reason_bad is not None
    assert "unpinned_sha_1234" in reason_bad
    assert contract_asset["source_revision"] in reason_bad
    assert any("unpinned_sha_1234" in b for b in external_asset_blockers(m))

    # Valid pin: refs/main -> declared revision
    pinned_rev = contract_asset["source_revision"]
    pinned_snap = repo_dir / "snapshots" / pinned_rev
    pinned_snap.mkdir(parents=True)
    (refs / "main").write_text(pinned_rev)
    import json
    (pinned_snap / "config.json").write_text(json.dumps({"model_type": "bert"}))
    (pinned_snap / "tokenizer_config.json").write_text(json.dumps({"do_lower_case": True}))
    (pinned_snap / "tokenizer.json").write_text(json.dumps({}))
    (pinned_snap / "vocab.txt").write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\n")
    (pinned_snap / "pytorch_model.bin").write_bytes(b"dummy")
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_CONFIG_JSON",
                       str(sum((pinned_snap / name).stat().st_size for name in aux_asset["hub_files"])))
    val_good = validate_auxiliary_asset(aux_asset, dest)
    assert val_good.ok is True
    assert val_good.status == "VERIFIED"

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: None)
    statuses_good = external_asset_statuses(m)
    hub_status_good = next(s for s in statuses_good if s["name"] == contract_asset["name"])
    reason_good = cache_asset_blocker(contract_asset, hub_status_good)
    assert reason_good is None  # Admitted!
