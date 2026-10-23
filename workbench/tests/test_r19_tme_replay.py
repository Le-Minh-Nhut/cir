"""Regression coverage for TME replay metrics, cache sizing, and checkpoint paths."""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest
import torch

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import required_runtime_assets
from workbench.replay import tme


VALUES = {
    "dress": {"recall_at1": 10.0, "recall_at5": 15.0, "recall_at10": 20.0,
              "recall_at50": 40.0, "acc": 30.0, "mean": 30.0},
    "shirt": {"recall_at1": 15.0, "recall_at5": 25.0, "recall_at10": 35.0,
              "recall_at50": 55.0, "acc": 45.0, "mean": 45.0},
    "toptee": {"recall_at1": 20.0, "recall_at5": 40.0, "recall_at10": 50.0,
               "recall_at50": 70.0, "acc": 60.0, "mean": 60.0},
}


def _workspace(tmp_path: Path, monkeypatch, *, with_features: bool):
    checkout = tmp_path / "TME"
    source = checkout / "src"
    source.mkdir(parents=True)
    native_source = Path(tme.__file__).parents[1] / "third_party" / "TME" / "src"
    (source / "precompute_test.py").write_text("# fixture entry point\n")
    (source / "precompute_evaluation.py").symlink_to(native_source / "precompute_evaluation.py")
    (source / "data.py").symlink_to(native_source / "data.py")
    lavis = source / "lavis"
    lavis.mkdir()
    (lavis / "__init__.py").write_text("")
    (lavis / "models.py").write_text(
        "from types import SimpleNamespace\n"
        "class Model:\n"
        "    def load_state_dict(self, state, strict=False):\n"
        "        return SimpleNamespace(missing_keys=[], unexpected_keys=[])\n"
        "    def eval(self):\n"
        "        return self\n"
        "def load_model_and_preprocess(**kwargs):\n"
        "    return Model(), None, {'eval': lambda text: text}\n"
    )
    (source / "utility.py").write_text(
        "device = 'cpu'\n"
        "def targetpad_transform(**kwargs):\n"
        "    return lambda image: image\n"
    )

    dataset_root = tmp_path / "FashionIQ"
    (dataset_root / "captions").mkdir(parents=True)
    (dataset_root / "image_splits").mkdir()
    image_names = {category: [f"{category}_image"] for category in CATEGORIES}
    for category in CATEGORIES:
        (dataset_root / "captions" / f"cap.{category}.val.json").write_text("[]")
        (dataset_root / "image_splits" / f"split.{category}.val.json").write_text(
            json.dumps(image_names[category]))

    feature_root = checkout / "features" / "FashionIQ"
    feature_root.mkdir(parents=True)
    if with_features:
        names = [name for category in CATEGORIES for name in image_names[category]]
        mapping = {name: index for index, name in enumerate(names)}
        (feature_root / "val_name_to_idx.json").write_text(json.dumps(mapping))
        # Sparse fixture keeps the native [N, 257, 1408] shape without allocating the real cache.
        features = torch.sparse_coo_tensor(
            torch.empty((3, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float16),
            (3, 257, 1408))
        torch.save(features.coalesce(), feature_root / "val_images.pt")

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    # The registry's declared filename, not the archive member's own name: `fashioniq_0` publishes
    # `best_model.pth` INSIDE a container, and the downloader writes the extracted member under the
    # declared `fashioniq_0_best_model.pth` so the FashionIQ and CIRR variants cannot collide.
    torch.save({}, checkpoint_root / "fashioniq_0_best_model.pth")
    monkeypatch.chdir(checkout)
    return checkout, dataset_root, checkpoint_root, feature_root


def _install_native_metric(monkeypatch, checkout: Path, payload: dict):
    source = checkout / "src"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    for name in ("precompute_evaluation", "data", "utility", "lavis", "lavis.models"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    sys.path.insert(0, str(source))
    native = importlib.import_module("precompute_evaluation")
    calls: list[object] = []

    def fake_evaluate_features(model, dataset, text_preprocessor):
        calls.append(dataset)
        for category in CATEGORIES:
            dataset.set_specific_dress_type(category)
        return payload

    monkeypatch.setattr(native, "evaluate_features", fake_evaluate_features)
    return calls


def _asset_status(asset, path: Path, state: str, digest: str | None = None):
    return {
        "name": asset["name"], "status": state, "path": str(path),
        "expected_sha256": asset.get("expected_sha256"),
        "expected_sha256_prefix": asset.get("expected_sha256_prefix"),
        "actual_sha256": digest,
    }


@pytest.mark.parametrize(("state", "size", "digest", "expected", "blocked"), [
    ("EXISTS_UNVERIFIED_CHECKSUM", 4, "a" * 64, 4, False),
    ("EXISTS_UNVERIFIED_CHECKSUM", 3, "a" * 64, 4, True),
    ("CONFIGURED_MISSING", None, None, 4, True),
    ("CHECKSUM_MISMATCH", 4, "b" * 64, 4, True),
])
def test_tme_preflight_checks_unverified_assets_against_registry_size_and_digest(
        tmp_path, monkeypatch, capsys, state, size, digest, expected, blocked):
    checkout, dataset, checkpoints, _ = _workspace(tmp_path, monkeypatch, with_features=True)
    assets = required_runtime_assets(tme.MODEL)
    # All three of the registry's declared assets are required on the eval path. An earlier revision
    # marked BLIP-2 optional on the reasoning that `load_from_pretrained` is unreachable; that was
    # wrong (the pinned `blip2_pretrain.yaml` sets `load_finetuned: False` AND a `pretrained` URL, so
    # the pretrained branch really does run), and the corrected classification is what this asserts.
    assert {asset["name"] for asset in assets} == {
        "BLIP-2 pretrain weights", "EVA-ViT-g backbone weights", "bert-base-uncased tokenizer"}
    # Pick the FILE-shaped asset to exercise size/digest enforcement: the directory-shaped tokenizer
    # cache has no file size, and it gets its own assertion below.
    selected = next(asset for asset in assets if asset.get("location_type") != "directory")
    selected = {**selected, "expected_size_bytes": expected, "expected_sha256": None,
                "expected_sha256_prefix": None}
    asset_path = tmp_path / "tiny-asset.bin"
    if size is not None:
        asset_path.write_bytes(b"x" * size)
    # Every OTHER declared asset is supplied as genuinely available so these cases are about the
    # SELECTED asset and cannot pass or fail because an unrelated asset was missing. Their declared
    # size/digest are cleared for the stand-in: they are only a few bytes here, and an unrelated size
    # mismatch would mask the case under test. A `location_type: directory` asset needs a POPULATED
    # DIRECTORY, not a file - substituting a file for the tokenizer cache reproduced the very defect
    # the check exists to catch.
    populated = tmp_path / "hf-cache"
    populated.mkdir()
    (populated / "refs").write_text("main")
    others: list[dict] = []
    statuses = [_asset_status(selected, asset_path, state, digest)]
    for other in assets:
        if other["name"] == selected["name"]:
            continue
        stand_in = populated if other.get("location_type") == "directory" else asset_path
        others.append({**other, "expected_size_bytes": None, "expected_sha256": None,
                       "expected_sha256_prefix": None})
        statuses.append({
            "name": other["name"], "status": "VERIFIED_LOCAL_ASSET", "path": str(stand_in),
            "expected_sha256": None, "expected_sha256_prefix": None, "actual_sha256": None})
    monkeypatch.setattr(tme, "required_runtime_assets", lambda model: [selected, *others])
    monkeypatch.setattr(tme, "external_asset_statuses", lambda model: statuses)

    blockers = tme.preflight(checkout, dataset, checkpoints, "fashioniq_0",
                             runtime_assets=([], []))
    # `_runtime_asset_check` is what produces the asset verdicts, so it is called through the same
    # patched decorators preflight would use.
    blockers += tme._runtime_asset_check()[0]
    asset_blockers = [blocker for blocker in blockers if blocker.startswith("TME runtime asset")]
    stderr = capsys.readouterr().err

    if blocked:
        assert any(blocker.startswith(f"TME runtime asset {selected['name']}")
                   for blocker in asset_blockers), asset_blockers
    else:
        assert not asset_blockers, asset_blockers
        assert ("author digest unavailable; local digest recorded, not verified against a published value"
                in stderr), stderr


def test_tme_native_aggregate_runs_once_and_produces_complete_category_rows(
        tmp_path, monkeypatch, capsys):
    checkout, dataset, checkpoints, _ = _workspace(tmp_path, monkeypatch, with_features=True)
    calls = _install_native_metric(monkeypatch, checkout, {"acc": 45.0, **VALUES, "mean": 45.0})

    results, aggregate = tme.evaluate_categories(checkout, dataset, checkpoints)

    assert len(calls) == 1
    assert [result.category for result in results] == list(CATEGORIES)
    for result in results:
        assert result.complete is True
        assert all(math.isfinite(value) for value in (result.r1, result.r10, result.r50))
        assert result.r1 == VALUES[result.category]["recall_at1"]
        assert result.provenance["feature_cache_provenance"] == "UNVERIFIED"
    assert "TME feature cache provenance UNVERIFIED" in capsys.readouterr().err
    assert aggregate["macro_r10"] == sum(row["recall_at10"] for row in VALUES.values()) / 3
    assert aggregate["macro_r50"] == sum(row["recall_at50"] for row in VALUES.values()) / 3
    assert aggregate["macro_mean"] == sum(row["mean"] for row in VALUES.values()) / 3
    assert aggregate["macro_r10"] == sum(result.r10 for result in results) / 3
    assert aggregate["macro_r50"] == sum(result.r50 for result in results) / 3
    assert aggregate["macro_mean"] == sum((result.r10 + result.r50) / 2
                                           for result in results) / 3
@pytest.mark.parametrize("payload", [
    {"acc": 30.0, "dress": VALUES["dress"], "shirt": VALUES["shirt"], "mean": 30.0},
    {"acc": 30.0, **{
        **VALUES,
        "shirt": {**VALUES["shirt"], "recall_at10": float("nan")},
    }, "mean": 30.0},
])
def test_tme_refuses_partial_or_nonfinite_native_category_metrics(tmp_path, monkeypatch, payload):
    checkout, dataset, checkpoints, _ = _workspace(tmp_path, monkeypatch, with_features=True)
    calls = _install_native_metric(monkeypatch, checkout, payload)

    with pytest.raises(RuntimeError, match="omitted category|invalid recall_at10"):
        tme.evaluate_categories(checkout, dataset, checkpoints)

    assert len(calls) == 1


def test_tme_refuses_subset_request_without_single_category_native_api(tmp_path, monkeypatch):
    checkout, dataset, checkpoints, _ = _workspace(tmp_path, monkeypatch, with_features=True)
    calls = _install_native_metric(monkeypatch, checkout, {"acc": 45.0, **VALUES, "mean": 45.0})

    with pytest.raises(ValueError, match="only supports the full FashionIQ category set"):
        tme.evaluate_categories(checkout, dataset, checkpoints, categories=("dress",))

    assert calls == []


def test_tme_preflight_sizes_missing_native_features_from_fixture_gallery(tmp_path, monkeypatch):
    checkout, dataset, checkpoints, feature_root = _workspace(
        tmp_path, monkeypatch, with_features=False)
    tme_blockers = lambda: [
        blocker for blocker in tme.preflight(
            checkout, dataset, checkpoints, "fashioniq_0", runtime_assets=([], []))
        if blocker.startswith("TME feature cache file missing:")
    ]

    before = tme_blockers()
    assert len(before) == 2
    assert all(str(feature_root / name) in blocker for name, blocker in zip(
        ("val_images.pt", "val_name_to_idx.json"), before))
    expected_bytes = 3 * 257 * 1408 * 2
    assert all(f"required feature tensor lower bound: {expected_bytes} bytes" in blocker
               for blocker in before)
    assert all("3 gallery images x 257 tokens x 1408 features x 2 bytes" in blocker
               for blocker in before)
    assert all("free space at destination:" in blocker for blocker in before)
    other_before = [
        blocker for blocker in tme.preflight(
            checkout, dataset, checkpoints, "fashioniq_0", runtime_assets=([], []))
        if not blocker.startswith("TME feature cache file missing:")
    ]

    names = [f"{category}_image" for category in CATEGORIES]
    (feature_root / "val_name_to_idx.json").write_text(
        json.dumps({name: index for index, name in enumerate(names)}))
    torch.save(torch.sparse_coo_tensor(
        torch.empty((3, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float16),
        (3, 257, 1408)), feature_root / "val_images.pt")
    after = tme_blockers()
    other_after = [
        blocker for blocker in tme.preflight(
            checkout, dataset, checkpoints, "fashioniq_0", runtime_assets=([], []))
        if not blocker.startswith("TME feature cache file missing:")
    ]

    assert after == []
    assert other_after == other_before


def test_tme_checkpoint_member_root_has_no_duplicate_model_id(tmp_path, monkeypatch):
    checkout, dataset, checkpoints, _ = _workspace(tmp_path, monkeypatch, with_features=True)

    blockers = tme.preflight(checkout, dataset, checkpoints, "fashioniq_0", runtime_assets=([], []))

    assert not any(blocker.startswith("TME checkpoint") for blocker in blockers), blockers
    # The wrapper opens `<checkpoint-root>/<registry filename>`; a doubled model id would look for
    # `<checkpoint-root>/tme/<filename>`, which does not exist.
    assert (checkpoints / "fashioniq_0_best_model.pth").is_file()
    assert not (checkpoints / "tme" / "fashioniq_0_best_model.pth").exists()
