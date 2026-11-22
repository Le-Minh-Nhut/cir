"""Tests for replay infrastructure and legacy model contracts."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import load_registry


def registry_models():
    return {model["model_id"]: model for model in load_registry()["models"]}


def test_the_original_protocols_are_unchanged():
    """The three protocols the workbench shipped with must keep their exact identities.

    Asserted as a SUBSET rather than as the whole set: a later protocol (TME's
    `fashioniq_val_split_ref_included`) is a deliberate addition, and rewriting this assertion into an
    equality would make every such addition look like a regression of the originals.
    """
    protocols = load_registry()["protocols"]
    assert {
        "fashioniq_original_split",
        "fashioniq_full_gallery_ref_excluded",
        "fashioniq_val_split",
    } <= set(protocols)


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


def test_clip4cir_incomplete_bundle_blocks(tmp_path: Path):
    """The bundle is the AUTHOR's two files, and an incomplete or substituted one must block.

    The previous version asserted the old `bundle_placeholder` shape whose `required_files` were
    `combiner_state.pt` / `clip_state.pt` — spellings the adapter invented and that no file on disk
    ever had, so the check could only compare against names which do not exist. Asserting behaviour
    (a half-populated directory blocks; a name-complete one with foreign bytes also blocks) is what
    actually pins the requirement.
    """
    from workbench.backend.registry import checkpoint_missing_paths

    checkpoint = registry_models()["clip4cir_rn50x4_fullft"]["checkpoint_variants"][0]
    assert checkpoint.get("artifact_type") in {"bundle_directory", "bundle_placeholder"}
    assert checkpoint.get("required_files") == ["fiq_comb_RN50x4_fullft.pt",
                                                "fiq_clip_RN50x4_fullft.pt"]
    bundle = tmp_path / "RN50x4_fullft"
    bundle.mkdir()
    (bundle / "fiq_comb_RN50x4_fullft.pt").write_bytes(b"combiner")
    assert checkpoint_missing_paths(bundle, checkpoint), "half a bundle must not read as present"
    # Both names present, but these are not the author's bytes: the declared member digests are
    # enforced, so a substituted file is refused rather than silently evaluated.
    (bundle / "fiq_clip_RN50x4_fullft.pt").write_bytes(b"clip")
    assert checkpoint_missing_paths(bundle, checkpoint), "a substituted member must not read as present"


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
    assert hasattr(limn_export, "export")


def test_the_exporter_does_not_require_the_canonical_schema() -> None:
    """P1: the exporter runs under the MODEL interpreter, which has no pydantic.

    It therefore must not import the canonical schema -- not at module level and not inside a
    function, because the model environment is provisioned from the model's own requirements. The
    publisher (`limn_publish`, which imports pydantic) runs in the workbench interpreter instead.
    This is checked by AST so a future refactor cannot quietly reintroduce the dependency.
    """
    import ast
    from pathlib import Path

    source = (Path(__file__).resolve().parents[2] / "workbench/replay/limn_export.py").read_text()
    tree = ast.parse(source)
    required = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            required.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            required.add(node.module.split(".")[0])
    assert "pydantic" not in required, sorted(required)
    # And the schema module must not be reachable through a local import either.
    assert "workbench.backend.schemas.results" not in source, \
        "the exporter must not reach the canonical schema"
    assert "workbench.replay.publication" not in source, \
        "the exporter must not import the publication layer (it pulls pydantic)"


def test_the_export_stdout_is_exactly_one_json_object(monkeypatch, tmp_path, capsys):
    """The orchestrator parses exporter stdout as one JSON object."""
    import json as _json

    import workbench.replay.limn_export as limn_export
    from workbench.replay.limn_observer import ObservedCategory, ObservedQuery

    def observation(category):
        queries = tuple(ObservedQuery(annotation_index=index, reference_id=f"{category}-r{index}",
            target_id=f"{category}-t{index}", raw_captions=("a", "b"), model_input_text="a and b",
            target_rank=1, top_results=((1, f"{category}-t{index}", 0.9),), category=category)
            for index in range(2))
        return ObservedCategory(category, 100, 99, queries)

    monkeypatch.setattr(limn_export, "CATEGORIES", ("dress", "shirt", "toptee"))
    monkeypatch.setattr(limn_export, "_load_native_category", lambda category, *_: observation(category))
    monkeypatch.setattr(limn_export, "cross_check_with_native", lambda *_: [])
    code = limn_export.export(tmp_path, tmp_path, tmp_path / "results",
        {name: {"r1": 100, "r10": 100, "r50": 100} for name in ("dress", "shirt", "toptee")},
        top_k=1, run_id="20260101T000000000000Z_limn_ckpt_stream01",
        command_digest="c" * 64, environment_digest="d" * 64)
    assert code == 0
    captured = capsys.readouterr()
    payload = _json.loads(captured.out)
    assert payload["payload"].endswith(".json")
    assert payload["run_id"] == "20260101T000000000000Z_limn_ckpt_stream01"
    assert payload["instrumentation_parity"] == "INSTRUMENTATION_PARITY_VERIFIED"
    assert "" == captured.err


def test_the_export_reports_a_parity_mismatch_with_a_nonzero_code(monkeypatch, tmp_path, capsys):
    import json as _json

    import workbench.replay.limn_export as limn_export
    from workbench.replay.limn_observer import ObservedCategory, ObservedQuery

    query = ObservedQuery(annotation_index=0, reference_id="r", target_id="t",
        raw_captions=("a", "b"), model_input_text="a and b", target_rank=2,
        top_results=((1, "d", 0.5), (2, "t", 0.4)), category="dress")
    observation = ObservedCategory("dress", 3, 2, (query,))
    monkeypatch.setattr(limn_export, "CATEGORIES", ("dress", "shirt", "toptee"))
    monkeypatch.setattr(limn_export, "_load_native_category", lambda category, *_: observation)
    monkeypatch.setattr(limn_export, "cross_check_with_native",
        lambda observed, *_: ["dress_r1: 33 != 1"] if observed.category == "dress" else [])
    metrics = {name: {"r1": 1, "r10": 1, "r50": 1} for name in ("dress", "shirt", "toptee")}
    code = limn_export.export(tmp_path, tmp_path, tmp_path / "results", metrics, top_k=1)
    assert code == 1
    status = _json.loads(capsys.readouterr().out)
    assert status["instrumentation_parity"] == "INSTRUMENTATION_PARITY_MISMATCH"
    assert status["instrumentation_mismatches"]

def test_the_publisher_turns_a_staged_payload_into_a_validated_artifact(monkeypatch, tmp_path, capsys):
    import json as _json

    import workbench.replay.limn_export as limn_export
    import workbench.replay.limn_publish as limn_publish
    from workbench.backend.schemas.results import ResultRun
    from workbench.replay.limn_observer import ObservedCategory, ObservedQuery

    def fake_observe(category, *_args):
        queries = tuple(ObservedQuery(index, f"{category}-r{index}", f"{category}-t{index}",
            ("a", "b"), "a and b", 1, ((1, f"{category}-t{index}", 0.9),), category)
            for index in range(2))
        return ObservedCategory(category, 100, 99, queries)

    monkeypatch.setattr(limn_export, "CATEGORIES", ("dress", "shirt", "toptee"))
    monkeypatch.setattr(limn_export, "_load_native_category", fake_observe)
    monkeypatch.setattr(limn_export, "cross_check_with_native", lambda *_: [])
    metrics = {name: {"r1": 100, "r10": 100, "r50": 100} for name in limn_export.CATEGORIES}
    results_root = tmp_path / "results"
    run_id = "20260101T000000000000Z_limn_ckpt_stage001"
    assert limn_export.export(tmp_path, tmp_path, results_root, metrics, top_k=1, run_id=run_id,
        command_digest="c" * 64, environment_digest="d" * 64) == 0
    staged = _json.loads(capsys.readouterr().out)["payload"]
    destination = limn_publish.publish_payload(Path(staged), results_root)
    artifact = ResultRun.model_validate_json(destination.read_text())
    assert artifact.run.run_id == run_id
    assert artifact.run.protocol_id == "fashioniq_val_split"
    assert artifact.run.data_kind == "experiment"
    assert Path(staged).name.startswith(".observation-")


def test_the_publisher_refuses_missing_same_invocation_provenance(tmp_path):
    import json
    import pytest
    from workbench.replay.limn_publish import PayloadError, load_payload

    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({"artifact_type": "limn_observation_payload", "model_id": "limn"}))
    with pytest.raises(PayloadError, match="same-invocation native ranking provenance"):
        load_payload(payload)
def test_the_publisher_refuses_missing_or_substituted_declared_bundle_members(monkeypatch, tmp_path):
    import pytest
    import workbench.backend.registry as registry
    from workbench.replay.limn_publish import PayloadError, _require_declared_bundle

    monkeypatch.setattr(registry, "model_by_id", lambda _: {"checkpoint_bundles": [{
        "bundle_id": "all_categories", "required_checkpoint_ids": ["dress", "shirt", "toptee"]}]})
    path = tmp_path / "payload.json"
    for members in ({"dress": {"checkpoint_id": "dress"}},
                    {"dress": {"checkpoint_id": "dress"}, "shirt": {"checkpoint_id": "shirt"},
                     "toptee": {"checkpoint_id": "wrong"}}):
        with pytest.raises(PayloadError, match="declares the bundle"):
            _require_declared_bundle({"model_id": "limn", "checkpoint_id": "all_categories",
                                      "checkpoint_members": members}, path)



def test_the_publisher_refuses_a_parity_mismatch(monkeypatch, tmp_path, capsys):
    import json
    import pytest
    import workbench.replay.limn_export as limn_export
    import workbench.replay.limn_publish as limn_publish
    from workbench.replay.limn_observer import ObservedCategory, ObservedQuery

    def fake_observe(category, *_args):
        queries = tuple(ObservedQuery(index, f"{category}-r{index}", f"{category}-t{index}",
            ("a", "b"), "a and b", 1, ((1, f"{category}-t{index}", 0.9),), category)
            for index in range(2))
        return ObservedCategory(category, 100, 99, queries)

    monkeypatch.setattr(limn_export, "CATEGORIES", ("dress", "shirt", "toptee"))
    monkeypatch.setattr(limn_export, "_load_native_category", fake_observe)
    monkeypatch.setattr(limn_export, "cross_check_with_native",
        lambda observed, *_: ["native rank mismatch"] if observed.category == "dress" else [])
    metrics = {name: {"r1": 100, "r10": 100, "r50": 100} for name in limn_export.CATEGORIES}
    results_root = tmp_path / "results"
    assert limn_export.export(tmp_path, tmp_path, results_root, metrics, top_k=1,
        run_id="20260101T000000000000Z_limn_ckpt_parity99") == 1
    staged = json.loads(capsys.readouterr().out)["payload"]
    with pytest.raises(Exception):
        limn_publish.publish_payload(Path(staged), results_root)

def test_limn_batch_size_defaults_to_the_protocol_setting(monkeypatch: pytest.MonkeyPatch):
    """The native `test()` reads `params.batch_size`; 32 is the author's setting.

    A batch size is how many images are stacked per forward pass, not what the protocol computes:
    `test()` concatenates every batch's features and normalizes afterwards. The override exists so
    the declared protocol can run on a GPU smaller than the one the paper used, and it must default
    to 32 when the operator says nothing.
    """
    from workbench.replay import limn

    monkeypatch.delenv(limn.LIMN_BATCH_SIZE_ENV, raising=False)
    assert limn.LIMN_BATCH_SIZE_DEFAULT == 32
    assert limn.resolved_batch_size() == 32

    monkeypatch.setenv(limn.LIMN_BATCH_SIZE_ENV, "4")
    assert limn.resolved_batch_size() == 4

def test_limn_batch_size_refuses_a_value_that_is_not_a_positive_integer(monkeypatch: pytest.MonkeyPatch):
    """A malformed override must refuse, not silently fall back to a different setting."""
    from workbench.replay import limn

    for bad in ("0", "-1", "abc", "4.5", " "):
        monkeypatch.setenv(limn.LIMN_BATCH_SIZE_ENV, bad)
        if not bad.strip():
            assert limn.resolved_batch_size() == limn.LIMN_BATCH_SIZE_DEFAULT
            continue
        with pytest.raises(ValueError):
            limn.resolved_batch_size()

def test_limn_native_argsort_capture_preserves_actual_order_and_fp32_mask(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    import numpy as np
    import pytest
    import workbench.replay.limn_observer as observer
    from workbench.replay import limn

    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    dataset = SimpleNamespace(
        test_queries=[{"source_img_id": 0, "target_img_id": 2, "mod": {"str": "red shirt"}}],
        test_targets=[{"target_img_id": 0}, {"target_img_id": 2}],
    )
    monkeypatch.setattr(observer, "_image_ids", lambda *_: ["ref", "other", "target", "last"])
    monkeypatch.setattr(observer, "_raw_captions", lambda *_: [{
        "candidate": "ref", "target": "target", "captions": ["red", "shirt"]}])
    monkeypatch.setattr(observer, "assert_alignment", lambda *_: None)
    monkeypatch.setenv(limn.LIMN_NATIVE_RANKINGS_ENV, str(tmp_path / "rankings"))
    native = SimpleNamespace(np=np)
    values = np.array([1e11, -3.0], dtype=np.float32)
    assert values[0] == values.dtype.type(10e10)
    with limn._capture_native_rankings(native, dataset, "dress", dataset_root) as rows:
        order = native.np.argsort(values)
        assert order.tolist() == [1, 0]
    receipt = json.loads((tmp_path / "rankings" / "dress.json").read_text())
    assert len(rows) == 1
    assert receipt["queries"][0]["target_rank"] == 1
    assert receipt["queries"][0]["source_rank"] == 2
    assert [item["image_id"] for item in receipt["queries"][0]["top_results"]] == ["target"]
    assert native.np is np

    monkeypatch.setenv(limn.LIMN_NATIVE_RANKINGS_ENV, str(tmp_path / "bad-rankings"))
    native = SimpleNamespace(np=SimpleNamespace(argsort=lambda *_args, **_kwargs: np.array([0, 1])))
    with limn._capture_native_rankings(native, dataset, "dress", dataset_root) as rows:
        native.np.argsort(np.array([1e11, -3.0], dtype=np.float32))
    bad_receipt = json.loads((tmp_path / "bad-rankings" / "dress.json").read_text())
    assert rows == []
    assert "not last" in bad_receipt["capture_error"]
def test_limn_export_rejects_non_object_receipt(tmp_path):
    import pytest
    from workbench.replay.limn_export import _load_native_category

    root = tmp_path / "rankings"
    root.mkdir()
    (root / "dress.json").write_text("[]")
    with pytest.raises(ValueError, match="not a JSON object"):
        _load_native_category("dress", root, tmp_path / "dataset", 200)
