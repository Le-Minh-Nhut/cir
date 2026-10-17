"""Tests for replay infrastructure and legacy model contracts."""
from __future__ import annotations

import inspect
from pathlib import Path

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

def test_replay_and_runtime_agree_that_bytecode_is_not_a_source_change(tmp_path: Path):
    """A pinned checkout someone merely imported must not read as dirty.

    `runtime._source_dirty` excludes `__pycache__`/`*.pyc`; the replay path used raw `git status` and
    refused the LIMN checkout for that reason alone. Both must answer the same question the same way,
    otherwise a checkout is runnable through one entry point and blocked through another.
    """
    import subprocess

    from workbench.backend.runtime import source_clean_and_pinned
    from workbench.replay.common import source_checkout_status

    repo = tmp_path / "checkout"
    repo.mkdir()
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.com",
           "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.com"}
    import os

    run_env = {**os.environ, **env}
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / "src.py").write_text("x = 1\n")
    subprocess.run(["git", "add", "src.py"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True, env=run_env)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                          capture_output=True, text=True).stdout.strip()

    clean_ok, _, _ = source_checkout_status(repo, head)
    assert clean_ok, "a clean pinned checkout must pass"

    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "src.cpython-313.pyc").write_bytes(b"\x00")
    bytecode_ok, _, reason = source_checkout_status(repo, head)
    assert bytecode_ok, f"bytecode alone must not dirty the checkout: {reason}"

    (repo / "src.py").write_text("x = 2\n")
    edited_ok, _, reason = source_checkout_status(repo, head)
    assert not edited_ok and "dirty" in (reason or ""), "a real source edit must still be refused"


def test_the_export_entry_point_is_separate_from_the_traced_native_replay(monkeypatch, tmp_path, capsys):
    """The per-query exporter must not live in the module the environment planner traces.

    `_evaluator_imports` walks `workbench/replay/limn.py` to decide LIMN's interpreter. Putting the
    observer (numpy) and the canonical-result schema (pydantic) in that file made the planner
    require an interpreter for packages the native replay never imports, and a cold-start run
    stopped provisioning LIMN. The export therefore lives in its own module, and this test pins
    that separation -- it is the regression for that cold-start failure.
    """
    import workbench.replay.limn as limn
    import workbench.replay.limn_export as limn_export

    for name in ("observe_category", "publish_observed_categories", "ObservedCategory"):
        assert not hasattr(limn, name), f"{name} must not be reachable from the traced module"
    # And the export module itself is importable with both capabilities present.
    assert hasattr(limn_export, "export") and hasattr(limn_export, "observe_and_verify")


def test_export_refuses_when_instrumentation_parity_fails(monkeypatch, tmp_path, capsys):
    """PARITY 1 failure: the artifact is withheld, and the native aggregate is not invented."""
    import workbench.replay.limn_export as limn_export
    from workbench.replay.limn_observer import ObservedCategory, ObservedQuery

    query = ObservedQuery(annotation_index=0, reference_id="r", target_id="t",
                          raw_captions=("a", "b"), model_input_text="a and b",
                          target_rank=2, top_results=((1, "d", 0.5), (2, "t", 0.4)), category="dress")
    observation = ObservedCategory(category="dress", gallery_size=3, eligible_count=2, queries=(query,))

    def fake_observe(category, *_a, **_k):
        return 1.0, observation, ("dress_r1: observer 33.3 != native 1.0",)

    monkeypatch.setattr(limn_export, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(limn_export, "observe_and_verify", fake_observe)

    code = limn_export.export(tmp_path, tmp_path, tmp_path, tmp_path / "results")
    assert code == 1
    captured = capsys.readouterr()
    assert "QUERY_EXPORT_PARITY_MISMATCH" in captured.err
    assert "INSTRUMENTATION_PARITY_MISMATCH" in captured.out
    assert not (tmp_path / "results").exists()


def test_export_publishes_a_validated_artifact_when_parity_holds(monkeypatch, tmp_path, capsys):
    """POSITIVE CONTROL: clean parity publishes, and the file validates on its own from disk."""
    import workbench.replay.limn_export as limn_export
    from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
    from workbench.backend.schemas.results import ResultRun

    def fake_observe(category, *_a, **_k):
        # The export walks the full three-category bundle, so the stub must answer per category.
        queries = tuple(
            ObservedQuery(annotation_index=index, reference_id=f"{category}-r{index}",
                          target_id=f"{category}-t{index}", raw_captions=("a", "b"),
                          model_input_text="a and b", target_rank=1,
                          top_results=((1, f"{category}-t{index}", 0.9),), category=category)
            for index in range(2))
        observation = ObservedCategory(category=category, gallery_size=2, eligible_count=1,
                                       queries=queries)
        return 100.0, observation, ()

    monkeypatch.setattr(limn_export, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(limn_export, "observe_and_verify", fake_observe)

    results_root = tmp_path / "results"
    # With the orchestrator's identity supplied, the artifact is a real experiment bound to it.
    run_id = "20260101T000000000000Z_limn_base_iter0_all_categories_deadbeef"
    code = limn_export.export(tmp_path, tmp_path, tmp_path, results_root, top_k=1, run_id=run_id,
                              command_digest="c" * 64, environment_digest="d" * 64)
    assert code == 0, capsys.readouterr().err
    published = list(results_root.rglob("*.json"))
    assert len(published) == 1, published
    artifact = ResultRun.model_validate_json(published[0].read_text())
    assert artifact.run.protocol_id == "fashioniq_val_split"
    assert artifact.run.data_kind == "experiment"
    # Bound to the supplied invocation, not to an identity of its own invention.
    assert artifact.run.run_id == run_id
    # One artifact covering the whole bundle, not one per category.
    assert len(artifact.queries) == 6
    assert set(artifact.run.category_gallery_sizes) == {"dress", "shirt", "toptee"}
    # Every bundle member is recorded, so a substituted category weight changes the digest (F-12).
    assert set(artifact.run.checkpoint_bundle) == {"dress", "shirt", "toptee"}
    assert artifact.run.bundle_manifest_digest
    assert set(artifact.run.category_checkpoint_ids.values()) == {
        "base_iter0_dress", "base_iter0_shirt", "base_iter0_toptee"}


def test_export_without_a_run_id_is_marked_as_development(monkeypatch, tmp_path, capsys):
    """F-09: a standalone export must not be able to claim a recorded invocation.

    With no orchestrator identity there is nothing to bind to, so the artifact is declared
    development data -- which can never become research evidence -- rather than inventing an
    identity that looks like a real run.
    """
    import workbench.replay.limn_export as limn_export
    from workbench.backend.schemas.results import ResultRun
    from workbench.replay.limn_observer import ObservedCategory, ObservedQuery

    def fake_observe(category, *_a, **_k):
        queries = tuple(
            ObservedQuery(annotation_index=index, reference_id=f"{category}-r{index}",
                          target_id=f"{category}-t{index}", raw_captions=("a", "b"),
                          model_input_text="a and b", target_rank=1,
                          top_results=((1, f"{category}-t{index}", 0.9),), category=category)
            for index in range(2))
        return 100.0, ObservedCategory(category=category, gallery_size=2, eligible_count=1,
                                      queries=queries), ()

    monkeypatch.setattr(limn_export, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(limn_export, "observe_and_verify", fake_observe)
    results_root = tmp_path / "results"
    assert limn_export.export(tmp_path, tmp_path, tmp_path, results_root, top_k=1) == 0
    artifact = ResultRun.model_validate_json(list(results_root.rglob("*.json"))[0].read_text())
    assert artifact.run.data_kind == "mock", "a standalone export is development data"
    assert artifact.run.run_id.startswith("dev-")
    assert "PENDING" in capsys.readouterr().out


def test_export_refuses_before_inference_when_preflight_fails(monkeypatch, tmp_path, capsys):
    """A missing bundle is refused by name, and no inference starts."""
    import workbench.replay.limn_export as limn_export

    monkeypatch.setattr(limn_export, "preflight", lambda *a, **k: ["checkpoint missing: /x"])
    monkeypatch.setattr(limn_export, "observe_and_verify",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("no inference")))

    code = limn_export.export(tmp_path, tmp_path, tmp_path, tmp_path / "results")
    assert code == 1
    assert "checkpoint missing" in capsys.readouterr().err
