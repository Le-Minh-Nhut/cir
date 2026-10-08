from __future__ import annotations

import torch

from datasets.fashioniq import FashionIQAnnotation, build_pair_union_gallery
from evaluation.fashioniq import build_original_gallery, recall_at_k
from evaluation.fashioniq_encoder import fashioniq_encoder_recall_at_k


def test_original_split_uses_file_order_and_keeps_reference(tmp_path) -> None:
    (tmp_path / "split.dress.val.json").write_text('["reference", "target", "other"]')
    gallery = build_original_gallery(tmp_path, "dress", "val")
    assert gallery == ["reference", "target", "other"]
    assert recall_at_k(torch.tensor([[0.9, 0.8, 0.7]]), ["target"], gallery, 1) == 0.0


def test_val_split_union_order_and_reference_exclusion() -> None:
    annotations = [FashionIQAnnotation("ref-a", "target-a", ("one", "two"), "dress", 0), FashionIQAnnotation("ref-b", "target-a", ("three", "four"), "dress", 1)]
    gallery = build_pair_union_gallery(annotations)
    assert gallery == ["ref-a", "target-a", "ref-b"]
    assert fashioniq_encoder_recall_at_k(torch.tensor([[0.99, 0.90, 0.80]]), ["target-a"], ["ref-a"], gallery, 1) == 100.0

def test_registry_exposes_three_exact_workbench_protocols() -> None:
    from workbench.backend.registry import load_registry

    models = {model["model_id"]: model for model in load_registry()["models"]}

    assert list(load_registry()["protocols"]) == ["fashioniq_original_split", "fashioniq_full_gallery_ref_excluded", "fashioniq_val_split"]
    assert models["csmcir"]["native_protocol"] == "fashioniq_original_split"
    for model_id in ("airknow", "conesep", "habit", "intent"):
        assert models[model_id]["literature_split_label"] == "original"
        assert models[model_id]["native_protocol"] == "fashioniq_full_gallery_ref_excluded"
        assert models[model_id]["supported_protocols"] == ["fashioniq_full_gallery_ref_excluded"]
    for model_id in ("hint", "encoder", "pair"):
        assert models[model_id]["native_protocol"] == "fashioniq_val_split"
        assert models[model_id]["supported_protocols"] == ["fashioniq_val_split"]


def test_legacy_registry_protocols_and_blockers_are_explicit() -> None:
    from workbench.backend.registry import load_registry

    models = {model["model_id"]: model for model in load_registry()["models"]}

    assert models["dcnet"]["native_protocol"] == "fashioniq_full_gallery_ref_excluded"
    assert models["dcnet"]["supported_protocols"] == ["fashioniq_full_gallery_ref_excluded"]
    assert models["dcnet"]["literature_split_label"] == "original"
    assert models["airknow"]["native_protocol"] == models["dcnet"]["native_protocol"]
    for model_id in ("clvc_net", "tgcir", "limn"):
        assert models[model_id]["native_protocol"] == "fashioniq_val_split"
    for model_id in ("combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "sprc"):
        assert models[model_id]["native_protocol"] == "fashioniq_original_split"
    assert models["sprc"]["reported_r10"] is None
    assert "discrepancy" in models["sprc"]["reported_score_note"]
    assert all(checkpoint["checkpoint_mapping_status"] != "VERIFIED_METADATA" for model_id in ("clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc", "limn") for checkpoint in models[model_id]["checkpoint_variants"])
    assert models["tgcir"]["reported_mean"] is None
    assert "neucore" not in models


def test_dcnet_shares_only_reference_excluded_full_gallery_cohort() -> None:
    from workbench.backend.registry import load_registry

    models = {model["model_id"]: model for model in load_registry()["models"]}

    assert models["dcnet"]["native_protocol"] == models["airknow"]["native_protocol"]
    assert models["dcnet"]["native_protocol"] != models["csmcir"]["native_protocol"]
    assert models["dcnet"]["native_protocol"] != models["hint"]["native_protocol"]


def test_only_dcnet_legacy_adapter_is_command_audited() -> None:
    from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter

    legacy = {"clvc_net", "dcnet", "combiner_rn50x4_noft", "clip4cir_rn50x4_fullft", "tgcir", "sprc", "limn"}

    assert legacy <= ADAPTERS.keys()
    assert ADAPTERS["dcnet"].command is not OfficialScriptAdapter.command
    assert all(ADAPTERS[model_id].command is OfficialScriptAdapter.command for model_id in legacy - {"dcnet"})


def test_legacy_layout_contracts_require_native_inputs(tmp_path) -> None:
    from workbench.backend.registry import fashioniq_required_paths, load_registry

    models = {model["model_id"]: model for model in load_registry()["models"]}

    assert tmp_path / "resized_image" / "dress" in fashioniq_required_paths(models["clvc_net"], tmp_path)
    assert tmp_path / "captions" / "cap.dress.train.json" in fashioniq_required_paths(models["clvc_net"], tmp_path)
    assert tmp_path / "captions" / "cap.dress.glove.val.pkl" in fashioniq_required_paths(models["dcnet"], tmp_path)
    assert tmp_path / "resized_images" in fashioniq_required_paths(models["dcnet"], tmp_path)
    assert tmp_path / "captions" / "cap.dress.train.json" in fashioniq_required_paths(models["tgcir"], tmp_path)
    assert tmp_path / "captions" / "cap.dress.train.json" in fashioniq_required_paths(models["limn"], tmp_path)


def test_legacy_preparation_contracts_are_source_specific_and_stay_manual() -> None:
    from workbench.backend.registry import load_preparation_contracts

    contracts = load_preparation_contracts()
    dcnet = contracts["dcnet_fashioniq"]

    assert {path for artifact in dcnet["generated_artifacts"] for path in artifact["paths"]} >= {
        "captions/cap.dress.glove.val.pkl",
        "captions/cap.shirt.glove.val.pkl",
        "captions/cap.toptee.glove.val.pkl",
    }
    assert {asset["name"] for asset in dcnet["external_assets"] if asset["required_for_preparation"]} == {"spaCy en_vectors_web_lg", "NLTK punkt"}
    assert dcnet["deterministic_status"] == "NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT"
    assert dcnet["automation_policy"] == "AUTHOR_ARTIFACT_PREFERRED"
    assert all(contract["automation_policy"] == "MANUAL_REQUIRED" for key, contract in contracts.items() if key != "dcnet_fashioniq")


def test_all_legacy_external_assets_are_machine_readable_and_fail_closed() -> None:
    from workbench.backend.registry import load_preparation_contracts

    for contract in load_preparation_contracts().values():
        for asset in contract["external_assets"]:
            assert {"name", "kind", "state", "required_for_evaluation", "required_for_preparation", "expected_local_location", "source", "optional", "notes"} <= asset.keys()
        assert any(asset["required_for_evaluation"] for asset in contract["external_assets"])


def test_limn_requires_complete_category_checkpoint_bundle(tmp_path) -> None:
    from workbench.backend.registry import checkpoint_bundle_missing_paths, load_registry

    limn = next(model for model in load_registry()["models"] if model["model_id"] == "limn")
    root = tmp_path / "checkpoints"
    for checkpoint in limn["checkpoint_variants"][:2]:
        path = root / "limn" / checkpoint["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"checkpoint")

    missing = checkpoint_bundle_missing_paths(limn, root)

    assert list(missing) == ["base_iter0_all_categories"]
    assert missing["base_iter0_all_categories"] == [root / "limn" / "0_toptee_best_model.pt"]
