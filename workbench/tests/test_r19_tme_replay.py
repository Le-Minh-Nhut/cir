"""Regression coverage for TME replay metrics, cache sizing, and checkpoint paths."""
from __future__ import annotations

import importlib
import json
import math
import sys
from pathlib import Path

import pytest
from PIL import Image
import torch

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import required_runtime_assets
from workbench.replay import tme

#: The names `_install_native_metric` imports out of the per-test fixture checkout.
_NATIVE_MODULES = ("precompute_evaluation", "data", "utility", "lavis", "lavis.models")


@pytest.fixture(autouse=True)
def _purge_native_modules():
    """Drop the fixture checkout's modules around each test, then restore what was already there.

    `_install_native_metric` prepends a pytest tmpdir to `sys.path` and imports these names from
    it, and `importlib` caches them in `sys.modules` for the WHOLE pytest process -- the tmpdir is
    deleted when the test ends, but the module objects outlive it. Every test in this file
    re-deletes and re-imports, so the order dependence is invisible from inside the file and only
    shows from another one: `workbench/replay/tme.py:359` does the same bare `sys.path.insert` +
    `import data`, so a later consumer in the same process resolves the deleted fixture module.

    The purge is only a purge while a test is running. An earlier test module may legitimately
    have a REAL `lavis` (or one of the other names) in `sys.modules`, and deleting that
    permanently would leave the process in a state this fixture did not find -- the same defect
    pointing the other way. So the incoming objects are snapshotted and put back after the test,
    and the pop before that update is what keeps a fixture module from being restored in their
    place.
    """
    saved = {name: sys.modules[name] for name in _NATIVE_MODULES if name in sys.modules}
    for name in _NATIVE_MODULES:
        sys.modules.pop(name, None)
    yield
    for name in _NATIVE_MODULES:
        sys.modules.pop(name, None)
    sys.modules.update(saved)


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
    (dataset_root / "images").mkdir()
    image_names = {category: [f"{category}_image"] for category in CATEGORIES}
    for category in CATEGORIES:
        (dataset_root / "captions" / f"cap.{category}.val.json").write_text("[]")
        # CANONICAL images at the model's own declared layout (`fashioniq_standard` -> `.png`).
        # This fixture used to build `resized_image/<cat>/<id>.jpg` instead, which is iLearn-family
        # derived data: TME does not declare generating it and its own preprocess expects the
        # canonical bytes, so a fixture that required it was testing the wrong dependency.
        Image.new("RGB", (2, 2), (120, 40, 20)).save(
            dataset_root / "images" / f"{image_names[category][0]}.png")
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
        # Non-uniform bytes: a file whose entire length is one repeated byte is refused by the
        # payload check, so a single-byte fixture would exercise the wrong rule.
        asset_path.write_bytes(bytes(range(1, size + 1)))
    # Every OTHER declared asset is supplied as genuinely available so these cases are about the
    # SELECTED asset and cannot pass or fail because an unrelated asset was missing. Their declared
    # size/digest are cleared for the stand-in: they are only a few bytes here, and an unrelated size
    # mismatch would mask the case under test. A `location_type: directory` asset needs a POPULATED
    # DIRECTORY, not a file - substituting a file for the tokenizer cache reproduced the very defect
    # the check exists to catch - and it must be a cache its consumer can actually LOAD: an empty
    # `refs/` file is exactly the "populated directory that is not a cache" the shared check now
    # refuses, so the stand-in carries a real minimal BertTokenizer snapshot.
    populated = tmp_path / "hf-cache" / "hub" / "models--bert-base-uncased"
    tokenizer_asset = next((a for a in assets if a.get("location_type") == "directory"), {})
    pinned_rev = tokenizer_asset.get("source_revision") or "86b5e0934494bd15c9632b12f734a8a67f723594"
    snapshot = populated / "snapshots" / pinned_rev
    snapshot.mkdir(parents=True)
    (populated / "refs").mkdir()
    (populated / "refs" / "main").write_text(pinned_rev)
    (snapshot / "vocab.txt").write_text(
        "[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\n", encoding="utf-8")
    (snapshot / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"BertTokenizer","do_lower_case":true}', encoding="utf-8")
    (snapshot / "config.json").write_text(
        '{"model_type":"bert","vocab_size":7}', encoding="utf-8")
    tok_data = ('{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],'
                '"normalizer":{"type":"BertNormalizer","clean_text":true,"handle_chinese_chars":true,'
                '"strip_accents":null,"lowercase":true},"pre_tokenizer":{"type":"BertPreTokenizer"},'
                '"post_processor":null,"decoder":null,"model":{"type":"WordPiece","unk_token":"[UNK]",'
                '"continuing_subword_prefix":"##","max_input_chars_per_word":100,'
                '"vocab":{"[PAD]":0,"[UNK]":1,"[CLS]":2,"[SEP]":3,"[MASK]":4,"hello":5,"world":6}}}')
    (snapshot / "tokenizer.json").write_text(tok_data, encoding="utf-8")
    from workbench.backend.registry import hub_cache_declared_files

    for name in hub_cache_declared_files("bert-base-uncased"):
        if not (snapshot / name).is_file():
            (snapshot / name).write_bytes(b"stand-in payload for " + name.encode())
    payload_size = sum((snapshot / name).stat().st_size for name in hub_cache_declared_files("bert-base-uncased"))
    monkeypatch.setenv("WORKBENCH_TEST_ASSET_SIZE_CONFIG_JSON", str(payload_size))
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
