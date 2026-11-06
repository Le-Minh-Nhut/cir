"""TME feature-cache dtype: the historical contract stays the default, the adapted runtime is honest.

The pinned precompute saves whatever `vit_encode` returned. On the torcu128 adapted runtime that is
float32 (MEASURED; see `workbench/registry/preparation_contracts.yaml` `runtime_overrides`), while the
historical contract declares float16 from `convert_weights_to_fp16` in `eva_vit.py:452`.

The correction is therefore per-RUNTIME, not a global relaxation: with no adaptation selected the
contract must still be float16 and must still reject a float32 cache. These tests exercise the
production resolver and the production validator, not a copy of their logic.
"""
from __future__ import annotations

import json
import struct
from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import (
    generated_artifact_parameters,
    load_preparation_contracts,
    load_registry,
    preparation_contract_for_model,
)
from workbench.scripts import prepare_model

GENERATOR = "tme_precompute_features"
MODEL = next(model for model in load_registry()["models"] if model["model_id"] == "tme")
CONTRACT = preparation_contract_for_model(MODEL)

def _model(capability: str | None) -> dict:
    """The real TME entry, optionally declaring only the given adaptation target."""
    if capability is None:
        return {**MODEL, "environment": {k: v for k, v in MODEL["environment"].items()
                                         if k != "adapted_runtime"}}
    environment = dict(MODEL["environment"])
    environment["adapted_runtime"] = {
        **(environment.get("adapted_runtime") or {}),
        "target_compute_capability": capability,
    }
    return {**MODEL, "environment": environment}

# ------------------------------------------------------------------ default contract is preserved


def test_default_resolution_is_still_float16(monkeypatch):
    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    parameters = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    assert parameters["feature_dtype"] == "float16"
    assert parameters["tokens_per_image"] == 257
    assert parameters["feature_width"] == 1408

def test_no_adaptation_selected_ignores_the_override_even_for_the_adapted_model(monkeypatch):
    """The override is gated on selection, not on the model merely DECLARING the profile."""
    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    assert generated_artifact_parameters(
        CONTRACT, GENERATOR, _model("sm_120"))["feature_dtype"] == "float16"

def test_selecting_the_declared_adaptation_yields_float32(monkeypatch):
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    assert generated_artifact_parameters(
        CONTRACT, GENERATOR, MODEL)["feature_dtype"] == "float32"

def test_override_does_not_disturb_other_parameters(monkeypatch):
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    adapted = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    default = generated_artifact_parameters(CONTRACT, GENERATOR, None)
    assert {k: v for k, v in adapted.items() if k != "feature_dtype"} == \
           {k: v for k, v in default.items() if k != "feature_dtype"}

def test_a_model_declaring_a_different_adaptation_refuses_rather_than_defaulting(monkeypatch):
    """A model that DECLARES an adaptation must not silently verify against another target."""
    from workbench.backend.errors import WorkbenchError

    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    with pytest.raises(WorkbenchError, match="UNKNOWN_RUNTIME_ADAPTATION"):
        generated_artifact_parameters(CONTRACT, GENERATOR, _model("sm_999"))

def test_a_model_with_no_adaptation_declared_keeps_the_historical_default(monkeypatch):
    """A process-wide selection must not break models that declare no adaptation at all."""
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    assert generated_artifact_parameters(
        CONTRACT, GENERATOR, _model(None))["feature_dtype"] == "float16"

def test_resolution_returns_a_copy_not_the_shared_contract(monkeypatch):
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    first = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    first["feature_dtype"] = "float64"
    assert generated_artifact_parameters(
        CONTRACT, GENERATOR, MODEL)["feature_dtype"] == "float32"
    assert load_preparation_contracts()["tme_fashioniq"]["generated_artifacts"], "contract intact"

def test_unknown_generator_is_refused():
    with pytest.raises(ValueError, match="unknown generated artifact generator"):
        generated_artifact_parameters(CONTRACT, "no_such_generator", MODEL)

# ------------------------------------------------------------------ byte accounting follows dtype


def test_required_cache_bytes_double_under_the_adapted_profile(monkeypatch):
    """The preflight lower bound must be derived from the SAME dtype replay accepts."""
    gallery = 15536
    dimensions = gallery * 257 * 1408
    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    default_bits = struct.calcsize({"float16": "e", "float32": "f"}[
        generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)["feature_dtype"]])
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    adapted_bits = struct.calcsize({"float16": "e", "float32": "f"}[
        generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)["feature_dtype"]])
    assert adapted_bits == 2 * default_bits
    # 20.94 GiB vs 10.47 GiB: the figure the operator frees disk against.
    assert dimensions * adapted_bits / (1 << 30) == pytest.approx(20.94, abs=0.01)
    assert dimensions * default_bits / (1 << 30) == pytest.approx(10.47, abs=0.01)

# ------------------------------------------------------------------ the real validator, both ways


def _ephemeral_cache(tmp_path: Path, gallery: int, dtype, parameters: dict):
    """A cache + receipt pair shaped exactly like the native output, at tiny dimensions."""
    feature_root = tmp_path / "features" / "FashionIQ"
    feature_root.mkdir(parents=True)
    tokens, width = 2, 3
    features = torch.zeros(gallery, tokens, width, dtype=dtype)
    feature = feature_root / "val_images.pt"
    torch.save(features, feature)
    mapping = feature_root / "val_name_to_idx.json"
    names = [f"img_{index}" for index in range(gallery)]
    mapping.write_text(json.dumps({name: index for index, name in enumerate(names)}), encoding="utf-8")

    # Gallery identity is computed by the production helper so the receipt and the replay half
    # cannot disagree by construction.
    dataset = tmp_path / "FashionIQ"
    (dataset / "image_splits").mkdir(parents=True)
    for category in ("dress", "shirt", "toptee"):
        (dataset / "image_splits" / f"split.{category}.val.json").write_text(
            json.dumps(names if category == "dress" else []), encoding="utf-8")
        (dataset / "captions").mkdir(exist_ok=True)
        (dataset / "captions" / f"cap.{category}.val.json").write_text("[]", encoding="utf-8")

    inputs = {
        "dataset_root": str(dataset),
        "split_files": {category: prepare_model._file_sha256(
            dataset / "image_splits" / f"split.{category}.val.json")
            for category in ("dress", "shirt", "toptee")},
        "gallery_images": {"dress": "deadbeef"},
        "base_assets": {},
        "recipe": {"source_commit": MODEL["upstream_commit_sha"]},
    }
    shape = [gallery, tokens, width]
    receipt = feature_root / parameters["receipt_filename"]
    receipt.write_text(json.dumps({
        "kind": "workbench_tme_precompute_receipt", "model_id": "tme",
        "generator": GENERATOR, "source_script": parameters["source_script"],
        "source_commit": MODEL["upstream_commit_sha"], "dataset": parameters["eval_dataset"],
        "split": parameters["split"], "batch_size": parameters["batch_size"],
        "workers": parameters["workers"], "inputs": inputs,
        "files": {"features": {"size_bytes": feature.stat().st_size,
                               "sha256": prepare_model._file_sha256(feature)},
                  "name_to_idx": {"size_bytes": mapping.stat().st_size,
                                  "sha256": prepare_model._file_sha256(mapping)}},
        "feature_shape": shape, "feature_dtype": str(dtype),
    }, indent=2), encoding="utf-8")
    return feature_root, receipt, inputs, {**parameters, "tokens_per_image": tokens,
                                           "feature_width": width}


def test_validator_accepts_float32_only_when_the_contract_expects_float32(tmp_path, monkeypatch):
    monkeypatch.delenv("WORKBENCH_RUNTIME_ADAPTATION", raising=False)
    parameters = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    root, receipt, inputs, parameters = _ephemeral_cache(tmp_path, 4, torch.float32, parameters)

    # Historical contract: float32 is a real mismatch and must be refused, not coerced.
    assert prepare_model._tme_validate(root, parameters, inputs, MODEL["upstream_commit_sha"], receipt) is False

    # Adapted contract, resolved through the production resolver: the same bytes are correct.
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    adapted = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    parameters = {**parameters, "feature_dtype": adapted["feature_dtype"]}
    assert prepare_model._tme_validate(root, parameters, inputs, MODEL["upstream_commit_sha"], receipt) is True


def test_validator_still_rejects_a_cache_whose_dtype_matches_neither(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    parameters = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    root, receipt, inputs, parameters = _ephemeral_cache(tmp_path, 4, torch.float64, parameters)
    assert prepare_model._tme_validate(
        root, parameters, inputs, MODEL["upstream_commit_sha"], receipt) is False


def test_validator_still_rejects_a_wrong_shape(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    parameters = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    root, receipt, inputs, parameters = _ephemeral_cache(tmp_path, 4, torch.float32, parameters)
    wrong = {**parameters, "tokens_per_image": parameters["tokens_per_image"] + 1}
    assert prepare_model._tme_validate(
        root, wrong, inputs, MODEL["upstream_commit_sha"], receipt) is False


def test_validator_still_rejects_changed_gallery_order(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    parameters = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    root, receipt, inputs, parameters = _ephemeral_cache(tmp_path, 4, torch.float32, parameters)
    reordered = json.loads((root / "val_name_to_idx.json").read_text(encoding="utf-8"))
    flipped = dict(reversed(list(reordered.items())))
    (root / "val_name_to_idx.json").write_text(json.dumps(flipped), encoding="utf-8")
    assert prepare_model._tme_validate(
        root, parameters, inputs, MODEL["upstream_commit_sha"], receipt) is False


def test_validator_still_rejects_a_stale_receipt_digest(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")
    parameters = generated_artifact_parameters(CONTRACT, GENERATOR, MODEL)
    root, receipt, inputs, parameters = _ephemeral_cache(tmp_path, 4, torch.float32, parameters)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    payload["files"]["features"]["sha256"] = "0" * 64
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    assert prepare_model._tme_validate(
        root, parameters, inputs, MODEL["upstream_commit_sha"], receipt) is False
