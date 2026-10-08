"""Tests for replay infrastructure and legacy model contracts."""
from __future__ import annotations

import inspect

import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import load_registry


def registry_models():
    return {model["model_id"]: model for model in load_registry()["models"]}


def test_three_protocols_unchanged():
    protocols = load_registry()["protocols"]
    assert set(protocols) == {
        "fashioniq_original_split",
        "fashioniq_full_gallery_ref_excluded",
        "fashioniq_val_split",
    }


def test_seven_legacy_entries_valid():
    models = registry_models()
    legacy_ids = ["clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc", "limn"]
    for model_id in legacy_ids:
        assert model_id in models, f"{model_id} missing from registry"
        assert models[model_id]["research_generation"] == "legacy"


def test_limn_requires_three_categories():
    from workbench.replay.common import ReplayResult, macro_aggregate

    results = [ReplayResult("limn", f"base_iter0_{category}", category, "fashioniq_val_split", r10=50, r50=70, complete=True) for category in ("dress", "shirt")]
    assert macro_aggregate(results)["complete"] is False


@pytest.mark.parametrize("missing_category", list(CATEGORIES))
def test_limn_rejects_missing_checkpoint(tmp_path: Path, missing_category: str):
    from workbench.replay.limn import validate_checkpoint_bundle

    for category in CATEGORIES:
        if category != missing_category:
            (tmp_path / f"0_{category}_best_model.pt").write_bytes(b"x" * 100)
    assert any(missing_category in blocker for blocker in validate_checkpoint_bundle(tmp_path))


def test_limn_rejects_sha_mismatch(tmp_path: Path):
    from workbench.replay.limn import validate_checkpoint_bundle

    for category in CATEGORIES:
        (tmp_path / f"0_{category}_best_model.pt").write_bytes(b"wrong content")
    blockers = validate_checkpoint_bundle(tmp_path)
    assert len(blockers) == 3
    assert all("mismatch" in blocker for blocker in blockers)


def test_limn_category_checkpoint_mapping():
    from workbench.replay.limn import LIMN_CHECKPOINTS

    assert set(LIMN_CHECKPOINTS) == set(CATEGORIES)
    for category in CATEGORIES:
        assert category in LIMN_CHECKPOINTS[category]["filename"]


def test_limn_no_model_averaging():
    from workbench.replay.limn import LIMN_CHECKPOINTS

    filenames = [LIMN_CHECKPOINTS[category]["filename"] for category in CATEGORIES]
    assert len(set(filenames)) == 3


def test_limn_macro_aggregation():
    from workbench.replay.common import ReplayResult, macro_aggregate

    results = [
        ReplayResult("limn", "base_iter0_dress", "dress", "fashioniq_val_split", r10=40.0, r50=60.0, complete=True),
        ReplayResult("limn", "base_iter0_shirt", "shirt", "fashioniq_val_split", r10=50.0, r50=70.0, complete=True),
        ReplayResult("limn", "base_iter0_toptee", "toptee", "fashioniq_val_split", r10=60.0, r50=80.0, complete=True),
    ]
    aggregate = macro_aggregate(results)
    assert aggregate["complete"] is True
    assert aggregate["macro_r10"] == 50.0
    assert aggregate["macro_r50"] == 70.0
    assert aggregate["macro_mean"] == 60.0


def test_limn_partial_not_complete():
    from workbench.replay.common import ReplayResult, macro_aggregate

    results = [ReplayResult("limn", "base_iter0_dress", "dress", "fashioniq_val_split", r10=40.0, r50=60.0, complete=True)]
    assert macro_aggregate(results)["complete"] is False


def test_limn_replay_uses_category_checkpoint_and_native_input_signature():
    from workbench.replay.limn import LIMN_CHECKPOINTS, replay_category

    assert set(inspect.signature(replay_category).parameters) == {
        "category", "checkpoint_root", "dataset_root", "source_root"
    }
    assert all(LIMN_CHECKPOINTS[category]["checkpoint_id"] == f"base_iter0_{category}" for category in CATEGORIES)


def test_encoder_is_shared_model():
    encoder = registry_models()["encoder"]
    assert encoder.get("category_model_policy", "shared") == "shared"
    assert len(encoder["checkpoint_variants"]) == 1
    assert not encoder.get("checkpoint_bundles")


def test_encoder_not_three_bundle():
    assert len(registry_models()["encoder"]["checkpoint_variants"]) == 1


def test_dcnet_run_directory_semantics():
    checkpoint = registry_models()["dcnet"]["checkpoint_variants"][0]
    assert checkpoint["artifact_type"] == "run_directory"
    assert set(checkpoint["required_files"]) == {"config.json", "trained_model.pth"}


def test_dcnet_missing_config_blocks(tmp_path: Path):
    from workbench.backend.registry import checkpoint_missing_paths

    checkpoint = registry_models()["dcnet"]["checkpoint_variants"][0]
    run_dir = tmp_path / "fashioniq_dcnet"
    run_dir.mkdir()
    (run_dir / "trained_model.pth").write_bytes(b"x")
    assert any("config.json" in str(path) for path in checkpoint_missing_paths(run_dir, checkpoint))


def test_dcnet_missing_weights_blocks(tmp_path: Path):
    from workbench.backend.registry import checkpoint_missing_paths

    checkpoint = registry_models()["dcnet"]["checkpoint_variants"][0]
    run_dir = tmp_path / "fashioniq_dcnet"
    run_dir.mkdir()
    (run_dir / "config.json").write_text("{}")
    assert any("trained_model.pth" in str(path) for path in checkpoint_missing_paths(run_dir, checkpoint))


def test_clip4cir_incomplete_bundle_blocks():
    checkpoint = registry_models()["clip4cir_rn50x4_fullft"]["checkpoint_variants"][0]
    assert checkpoint.get("artifact_type") == "bundle_placeholder"
    assert "combiner_state.pt" in checkpoint.get("required_files", [])


def test_sprc_checkpoint_mapping_blocks():
    checkpoint = registry_models()["sprc"]["checkpoint_variants"][0]
    assert checkpoint["checkpoint_mapping_status"] != "VERIFIED_METADATA"



def test_preparation_only_assets_not_runtime_blockers():
    from workbench.backend.registry import load_preparation_contracts

    dcnet_contract = load_preparation_contracts()["dcnet_fashioniq"]
    preparation_only = [asset for asset in dcnet_contract["external_assets"] if not asset["required_for_evaluation"]]
    assert len(preparation_only) == 2
    assert all(asset["required_for_preparation"] is True for asset in preparation_only)


def test_external_runtime_asset_resolution_fails_closed(tmp_path: Path, monkeypatch):
    import hashlib
    from workbench.backend import registry

    data = b"author-verified-weight-bytes"
    digest = hashlib.sha256(data).hexdigest()
    asset = {"name": "weight", "state": "SOURCE_HASH_AVAILABLE", "expected_local_location": None,
             "location_env_var": "WORKBENCH_TEST_WEIGHT", "expected_sha256": digest, "optional": False}
    monkeypatch.setattr(registry, "required_runtime_assets", lambda _: [asset])
    model = {"model_id": "test"}

    assert registry.external_asset_statuses(model)[0]["status"] == "UNCONFIGURED_LOCATION"
    path = tmp_path / "weight.pt"
    monkeypatch.setenv("WORKBENCH_TEST_WEIGHT", str(path))
    assert registry.external_asset_statuses(model)[0]["status"] == "CONFIGURED_MISSING"
    path.mkdir()
    assert registry.external_asset_statuses(model)[0]["status"] == "INVALID_FILE_TYPE"
    path.rmdir()
    path.write_bytes(b"")
    assert registry.external_asset_statuses(model)[0]["status"] == "EMPTY_PLACEHOLDER"
    path.write_bytes(b"wrong")
    assert registry.external_asset_statuses(model)[0]["status"] == "CHECKSUM_MISMATCH"
    path.write_bytes(data)
    status = registry.external_asset_statuses(model)[0]
    assert status["status"] == "VERIFIED_LOCAL_ASSET"
    assert status["actual_sha256"] == digest


def test_external_asset_hash_prefix_is_verified(tmp_path: Path, monkeypatch):
    import hashlib
    from workbench.backend import registry

    data = b"resnet weights"
    digest = hashlib.sha256(data).hexdigest()
    monkeypatch.setattr(registry, "required_runtime_assets", lambda _: [{
        "name": "resnet", "state": "SOURCE_HASH_PREFIX_AVAILABLE", "expected_local_location": None,
        "location_env_var": "WORKBENCH_TEST_PREFIX_WEIGHT", "expected_sha256_prefix": digest[:8], "optional": False,
    }])
    path = tmp_path / "resnet.pth"
    path.write_bytes(data)
    monkeypatch.setenv("WORKBENCH_TEST_PREFIX_WEIGHT", str(path))
    assert registry.external_asset_statuses({"model_id": "test"})[0]["status"] == "VERIFIED_LOCAL_ASSET"


def test_external_asset_uses_torch_cache_location(tmp_path: Path, monkeypatch):
    import hashlib
    from workbench.backend import registry

    digest = hashlib.sha256(b"torchvision weights").hexdigest()
    asset = {"name": "resnet", "state": "SOURCE_HASH_PREFIX_AVAILABLE", "expected_local_location": None,
             "cache_root_env_var": "TORCH_HOME", "cache_path_suffix": "checkpoints/resnet50.pth",
             "expected_sha256_prefix": digest[:8], "optional": False}
    monkeypatch.setattr(registry, "required_runtime_assets", lambda _: [asset])
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    weights = tmp_path / "torch" / "checkpoints" / "resnet50.pth"

    status = registry.external_asset_statuses({"model_id": "test"})[0]
    assert status["status"] == "CONFIGURED_MISSING"
    assert status["path"] == str(weights)
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"torchvision weights")
    assert registry.external_asset_statuses({"model_id": "test"})[0]["status"] == "VERIFIED_LOCAL_ASSET"




def test_missing_environment_documented():
    models = registry_models()
    for model_id in ["clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc", "limn"]:
        assert models[model_id].get("environment") is not None, f"{model_id} missing environment metadata"


def test_source_pin_required_for_legacy():
    models = registry_models()
    for model_id in ["clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc", "limn"]:
        assert models[model_id]["upstream_commit_sha"] is not None


def test_cross_protocol_prohibited():
    models = registry_models()
    for model_id in ["clvc_net", "tgcir", "limn"]:
        assert "fashioniq_original_split" not in models[model_id]["supported_protocols"]
    for model_id in ["combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "sprc"]:
        assert "fashioniq_val_split" not in models[model_id]["supported_protocols"]


def test_registry_alone_not_runnable():
    from workbench.backend.registry import checkpoint_availability

    models = registry_models()
    for model_id in ["clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc", "limn"]:
        for checkpoint in models[model_id]["checkpoint_variants"]:
            availability = checkpoint_availability(models[model_id], checkpoint)
            assert availability["runnable"] is False, f"{model_id}/{checkpoint['checkpoint_id']} should not be runnable"


def test_no_model_reproduced():
    models = registry_models()
    for model_id in ["clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc", "limn"]:
        assert models[model_id]["reproduction_status"] != "OFFICIAL_AGGREGATE_REPRODUCED"


def test_neucore_excluded():
    assert "neucore" not in registry_models()


def test_limn_independent_category_policy():
    assert registry_models()["limn"].get("category_model_policy") == "independent"


def test_legacy_replay_status_does_not_overstate_coverage():
    models = registry_models()

    assert models["clvc_net"]["replay_status"] == "BLOCKED_NO_SOURCE_REPLAY"
    assert models["tgcir"]["replay_status"] == "PREFLIGHT_ONLY"
    assert models["sprc"]["replay_status"] == "BLOCKED_CHECKPOINT_MAPPING"
    assert models["limn"]["replay_status"] == "REPLAY_CODE_IMPLEMENTED"


def test_limn_preflight_fails_on_empty(tmp_path: Path):
    from workbench.replay.limn import main

    code = main([
        "--checkpoint-root", str(tmp_path), "--dataset-root", str(tmp_path),
        "--source-root", str(tmp_path), "--preflight-only",
    ])
    assert code == 1


def test_checkpoint_serialization_metadata():
    models = registry_models()
    assert models["limn"].get("checkpoint_serialization") == "whole_model"
    assert models["dcnet"].get("checkpoint_serialization") == "run_directory"
