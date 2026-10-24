from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workbench.backend.fashioniq_layout import CATEGORIES, image_extension_for_layout
from workbench.backend.registry import load_auxiliary_assets, model_by_id, validate_auxiliary_asset
from workbench.replay.common import fashioniq_image_binding_blockers, isolated_cache_root


def _tme_asset() -> dict:
    asset = next(item for item in load_auxiliary_assets()
                 if item["model_id"] == "tme" and item["family"] == "BLIP-2 pretrain weights")
    return {**asset, "expected_size_bytes": 4, "expected_sha256": None,
            "expected_sha256_prefix": None, "provenance_status": "UNVERIFIED"}


def test_digestless_auxiliary_asset_admission_and_verify(tmp_path: Path, capsys) -> None:
    from workbench.backend import registry
    from workbench.backend.preflight import model_requirements
    from workbench.scripts.download_auxiliary_assets import Asset, verify

    path = tmp_path / "base.pth"
    path.write_bytes(b"TME!")
    declaration = _tme_asset()
    result = validate_auxiliary_asset(declaration, path)
    assert result.ok is True
    assert result.status == "LOCALLY_RECORDED_ASSET"
    assert result.issues == []

    downloader_asset = Asset(
        model_id="tme", family=declaration["family"], filename=path.name,
        destination=path, required=True, automatic_download_available=True,
        source_url=declaration["source_repo"], source_revision=None, expected_sha256=None,
        provenance_status="UNVERIFIED", expected_size_bytes=4,
        source_type=declaration["source_type"], source_repo=declaration["source_repo"],
    )
    assert verify(downloader_asset) is True
    assert "checksum unavailable; provenance not checksum-verified" in capsys.readouterr().out

    no_author_source = {**declaration, "source_repo": None}
    unlinked = validate_auxiliary_asset(no_author_source, path)
    assert unlinked.ok is False
    assert unlinked.status == "OPERATOR_SUPPLIED_UNVERIFIED"

    wrong_size = validate_auxiliary_asset({**declaration, "expected_size_bytes": 5}, path)
    assert wrong_size.ok is False

    # A file that is EXACTLY the declared length but holds no payload: a sparse or zero-filled
    # allocation is structurally a container of zeros, and since a digest-less asset has no
    # published value to compare against, the bytes themselves are the only evidence available.
    # Without this check the size test above is satisfied by `truncate()`, and a placeholder would
    # be admitted as a weight and handed to a model loader.
    placeholder = tmp_path / "placeholder.pth"
    placeholder.write_bytes(b"\0" * 4)
    suspect = validate_auxiliary_asset(declaration, placeholder)
    assert suspect.ok is False, "a uniform placeholder of the declared size is not an asset"
    assert suspect.status == "SUSPECT_PAYLOAD"
    assert any("SUSPECT_PAYLOAD" in issue for issue in suspect.issues)

    malformed = tmp_path / "captions.json"
    malformed.write_text("{broken", encoding="utf-8")
    bad_json = validate_auxiliary_asset(
        {**declaration, "content_contract": {"expected_record_count": 1}}, malformed)
    assert bad_json.ok is False
    assert any("INVALID JSON" in issue for issue in bad_json.issues)

    mismatch = validate_auxiliary_asset(
        {**declaration, "expected_sha256": hashlib.sha256(b"other").hexdigest()}, path)
    assert mismatch.ok is False
    assert any("OFFICIAL SHA256 MISMATCH" in issue for issue in mismatch.issues)

    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(registry, "auxiliary_assets_for_model", lambda _: [declaration])
        monkeypatch.setattr(registry, "auxiliary_destination", lambda *args: path)
        monkeypatch.setattr(registry, "preparation_contract_for_model", lambda _: None)
        monkeypatch.setattr(registry, "fashioniq_required_paths", lambda *args: ())
        monkeypatch.setattr(registry, "external_asset_statuses", lambda _: [])
        monkeypatch.setattr(registry, "required_runtime_assets", lambda _: [])
        model = {"model_id": "tme", "source_dir": "TME", "fashioniq_layout": "fashioniq_standard"}
        config = SimpleNamespace(
            FASHIONIQ_ROOT=tmp_path,
            WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
        )
        report = model_requirements(model, config, canonical_root=tmp_path)
        assert not any("unverified" in block for block in report["blocks"])
        assert any("locally recorded; author digest unavailable" in note for note in report["warnings"])
    finally:
        monkeypatch.undo()


def test_tokenizer_layout_can_be_certified_without_transformers(tmp_path: Path, monkeypatch) -> None:
    from workbench.backend.registry import cache_asset_blocker

    cache = tmp_path / "hub" / "models--bert-base-uncased"
    snapshot = cache / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    (snapshot / "vocab.txt").write_text("[UNK]\n", encoding="utf-8")
    monkeypatch.setitem(__import__("sys").modules, "transformers", None)
    status = {"path": str(cache)}
    assert cache_asset_blocker({"cache_path_suffix": "hub/models--bert-base-uncased"}, status) is None
    assert "load-test belongs to the model environment" in status["admission_note"]

    empty = tmp_path / "empty"
    (empty / "snapshots").mkdir(parents=True)
    assert cache_asset_blocker({"cache_path_suffix": "hub/models--bert-base-uncased"},
                               {"path": str(empty)}) is not None


def test_tme_flat_view_symlinks_nonsquare_canonical_image_without_resized_tree(tmp_path: Path) -> None:
    from PIL import Image

    root = tmp_path / "dataset"
    images = root / "images"
    splits = root / "image_splits"
    images.mkdir(parents=True)
    splits.mkdir()
    (root / "captions").mkdir()
    buffer = io.BytesIO()
    Image.new("RGB", (3, 2), (20, 40, 60)).save(buffer, format="PNG")
    canonical = buffer.getvalue()
    (images / "dress1.png").write_bytes(canonical)
    for category in CATEGORIES:
        (splits / f"split.{category}.val.json").write_text(
            json.dumps(["dress1"] if category == "dress" else []), encoding="utf-8")
    assert not (root / "resized_image").exists()

    extension = image_extension_for_layout("fashioniq_standard")
    assert extension == ".png"
    assert fashioniq_image_binding_blockers(root, "val", "images", extension) == []
    from workbench.scripts.prepare_model import _tme_dataset_root

    with _tme_dataset_root(SimpleNamespace(FASHIONIQ_ROOT=root),
                           {"fashioniq_layout": "fashioniq_standard"}) as staged:
        visible = staged / "images" / "dress1.jpg"
        assert visible.is_symlink()
        assert visible.read_bytes() == canonical
        assert Image.open(visible).size == (3, 2)

def test_native_tme_log_is_declared_but_python_edits_remain_dirty(tmp_path: Path) -> None:
    from workbench.backend.vendored_source import live_digest, source_provenance

    model = model_by_id("tme")
    checkout = tmp_path / "TME"
    script = checkout / "src" / "precompute.py"
    script.parent.mkdir(parents=True)
    script.write_text("native precompute source\n", encoding="utf-8")
    record = {"upstream_commit": model["upstream_commit_sha"],
              "file_manifest": live_digest(checkout, model), "local_fixes": []}
    (checkout / "provenance.json").write_text(json.dumps(record), encoding="utf-8")

    native_log = checkout / "log_TME" / "other" / "precomputation.log"
    native_log.parent.mkdir(parents=True)
    native_log.write_text("native runtime log\n", encoding="utf-8")
    (checkout / "src" / "__pycache__").mkdir()
    (checkout / "src" / "__pycache__" / "precompute.pyc").write_bytes(b"bytecode")
    assert source_provenance(model, checkout)["verified"] is True

    # The declaration names the log FILE the native generator writes, not the directory tree around
    # it. Admitting the whole `log_TME` subtree hid every file below it, so an injected executable
    # sitting next to the log was indistinguishable from the log line and no consumer could see it.
    injected = native_log.parent / "injected.py"
    injected.write_text("import os; os.system('echo injected')\n", encoding="utf-8")
    result = source_provenance(model, checkout)
    assert result["verified"] is False, (
        "an added .py under the log directory must be visible to the source gate")
    assert "injected.py" in (result.get("error") or "") or "injected.py" in json.dumps(result)
    injected.unlink()
    assert source_provenance(model, checkout)["verified"] is True

    script.write_text(script.read_text(encoding="utf-8") + "# unauthorized edit\n", encoding="utf-8")
    result = source_provenance(model, checkout)
    assert result["verified"] is False
    assert "precompute.py" in (result["error"] or "")


def _tme_input_fixture(tmp_path: Path, monkeypatch):
    from workbench.backend import registry
    from workbench.scripts.prepare_model import _file_sha256, _tme_inputs

    model = model_by_id("tme")
    dataset = tmp_path / "dataset"
    (dataset / "captions").mkdir(parents=True)
    (dataset / "image_splits").mkdir()
    (dataset / "images").mkdir()
    for category in CATEGORIES:
        name = f"{category}1"
        (dataset / "image_splits" / f"split.{category}.val.json").write_text(
            json.dumps([name]), encoding="utf-8")
        (dataset / "captions" / f"cap.{category}.val.json").write_text("[]", encoding="utf-8")
        (dataset / "images" / f"{name}.png").write_bytes(f"image-{name}".encode())
    checkout = tmp_path / "third_party" / "TME"
    (checkout / "src").mkdir(parents=True)
    script = checkout / "src" / "precompute.py"
    script.write_text("native transform source", encoding="utf-8")
    base_asset = tmp_path / "base.pth"
    base_asset.write_bytes(b"base-weights")
    config = SimpleNamespace(
        FASHIONIQ_ROOT=dataset,
        WORKBENCH_THIRD_PARTY_ROOT=tmp_path / "third_party",
        WORKBENCH_ARTIFACTS_ROOT=tmp_path / "artifacts",
    )
    contract = {"generated_artifacts": [{"generator": "tme_precompute_features", "parameters": {
        "split": "val", "source_script": "src/precompute.py", "eval_dataset": "FashionIQ",
        "batch_size": 3, "workers": 0, "tokens_per_image": 1, "feature_width": 2,
        "feature_dtype": "float16", "receipt_filename": ".workbench_tme_precompute_receipt.json"},
        "scope": "evaluator_cwd", "paths": ["features/FashionIQ"]}]}
    assets = [{"name": "test base weights", "expected_sha256": None,
               "expected_sha256_prefix": None, "location_type": "file"}]

    def statuses(_):
        digest = _file_sha256(base_asset)
        return [{"name": "test base weights", "status": "EXISTS_UNVERIFIED_CHECKSUM",
                 "path": str(base_asset), "actual_sha256": digest,
                 "actual_size_bytes": base_asset.stat().st_size}]

    monkeypatch.setattr(registry, "required_runtime_assets", lambda _: assets)
    monkeypatch.setattr(registry, "external_asset_statuses", statuses)
    return model, config, contract, base_asset


def _write_tme_receipt(tmp_path: Path, model, config, contract, inputs) -> Path:
    import torch

    from workbench.scripts.prepare_model import TME_FEATURE_GENERATOR, _file_sha256

    checkout = config.WORKBENCH_THIRD_PARTY_ROOT / "TME"
    output = checkout / "features" / "FashionIQ"
    output.mkdir(parents=True)
    feature = output / "val_images.pt"
    mapping = output / "val_name_to_idx.json"
    names = []
    for category in CATEGORIES:
        names.extend(json.loads((config.FASHIONIQ_ROOT / "image_splits"
                                 / f"split.{category}.val.json").read_text(encoding="utf-8")))
    torch.save(torch.zeros((len(names), 1, 2), dtype=torch.float16), feature)
    mapping.write_text(json.dumps({name: index for index, name in enumerate(names)}), encoding="utf-8")
    params = contract["generated_artifacts"][0]["parameters"]
    files = {"features": {"size_bytes": feature.stat().st_size, "sha256": _file_sha256(feature)},
             "name_to_idx": {"size_bytes": mapping.stat().st_size, "sha256": _file_sha256(mapping)}}
    receipt = config.WORKBENCH_ARTIFACTS_ROOT / "features" / "FashionIQ" / params["receipt_filename"]
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {"kind": "workbench_tme_precompute_receipt", "model_id": "tme",
               "generator": TME_FEATURE_GENERATOR, "source_script": params["source_script"],
               "source_commit": model["upstream_commit_sha"], "dataset": params["eval_dataset"],
               "split": params["split"], "batch_size": params["batch_size"], "workers": params["workers"],
               "inputs": inputs, "files": files, "feature_shape": [len(names), 1, 2],
               "feature_dtype": "torch.float16"}
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    return receipt


def test_tme_receipt_reuses_unchanged_inputs_and_rejects_changed_inputs(tmp_path: Path, monkeypatch) -> None:
    import workbench.replay.tme as replay_tme
    from workbench.replay.tme import _prior_feature_provenance
    from workbench.scripts.prepare_model import _tme_inputs, _tme_validate

    model, config, contract, base_asset = _tme_input_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(replay_tme, "preparation_contract_for_model", lambda _: contract)

    parameters = contract["generated_artifacts"][0]["parameters"]
    inputs = _tme_inputs(config, contract, model)
    assert inputs is not None
    receipt = _write_tme_receipt(tmp_path, model, config, contract, inputs)
    output = config.WORKBENCH_THIRD_PARTY_ROOT / "TME" / "features" / "FashionIQ"
    assert _tme_validate(output, parameters, inputs, model["upstream_commit_sha"], receipt)
    assert _prior_feature_provenance(config) is not None

    changed_image = config.FASHIONIQ_ROOT / "images" / "dress1.png"
    changed_image.write_bytes(b"changed same-id pixels")
    changed = _tme_inputs(config, contract, model)
    assert changed is not None and changed != inputs
    assert not _tme_validate(output, parameters, changed, model["upstream_commit_sha"], receipt)
    assert _prior_feature_provenance(config) is None

    changed_image.write_bytes(b"image-dress1")
    base_asset.write_bytes(b"changed base weights")
    changed_asset_inputs = _tme_inputs(config, contract, model)
    assert changed_asset_inputs is not None and changed_asset_inputs != inputs
    assert not _tme_validate(output, parameters, changed_asset_inputs, model["upstream_commit_sha"], receipt)

    base_asset.write_bytes(b"base-weights")
    split = config.FASHIONIQ_ROOT / "image_splits" / "split.dress.val.json"
    split.write_text(json.dumps(["dress2", "dress1"]), encoding="utf-8")
    (config.FASHIONIQ_ROOT / "images" / "dress2.png").write_bytes(b"image-dress2")
    changed_order = _tme_inputs(config, contract, model)
    assert changed_order is not None and changed_order != inputs
    assert not _tme_validate(output, parameters, changed_order, model["upstream_commit_sha"], receipt)

    legacy = json.loads(receipt.read_text(encoding="utf-8"))
    legacy["inputs"].pop("gallery_images")
    receipt.write_text(json.dumps(legacy), encoding="utf-8")
    assert not _tme_validate(output, parameters, inputs, model["upstream_commit_sha"], receipt)

    (output / "val_name_to_idx.json").unlink()
    assert not _tme_validate(output, parameters, inputs, model["upstream_commit_sha"], receipt)


def test_tme_replay_provenance_verifies_an_unchanged_cache_through_the_emitted_output(tmp_path: Path, monkeypatch) -> None:
    """Reader-not-None is not a verified run: the assertion belongs on the emitted provenance.

    What an operator consumes is the provenance that ships with the replay results, so this drives
    the real `evaluate_categories` and reads the verdict out of its rows. Only the native gallery
    loader and the metric function are stubbed; the receipt, the prior-run half, the comparison and
    the emitted fields are the production path.

    Reproduces the defect directly: the receipt side recorded ordered gallery IMAGE digests under
    `feature_gallery_sha256` while the replay side put split-JSON digests in that same field, so an
    unchanged cache could never match itself and every run emitted UNVERIFIED naming a difference
    that did not exist.
    """
    import torch
    from workbench.backend.fashioniq_layout import CATEGORIES
    from workbench.replay.common import fashioniq_gallery_digests
    import workbench.replay.tme as replay_tme
    from workbench.scripts.prepare_model import _tme_inputs

    model, config, contract, _ = _tme_input_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(replay_tme, "preparation_contract_for_model", lambda _: contract)
    parameters = contract["generated_artifacts"][0]["parameters"]

    inputs = _tme_inputs(config, contract, model)
    assert inputs is not None
    assert _write_tme_receipt(tmp_path, model, config, contract, inputs).is_file()
    prior = replay_tme._prior_feature_provenance(config)
    assert prior is not None, "a receipt bound to current inputs must be readable"

    # The real replay-side builder, on the real fixture: the model and the metric are not involved in
    # this verdict, the comparison is.
    output = config.WORKBENCH_THIRD_PARTY_ROOT / "TME" / "features" / "FashionIQ"
    feature = output / "val_images.pt"
    mapping = output / "val_name_to_idx.json"
    names = [name for category in CATEGORIES
             for name in json.loads((config.FASHIONIQ_ROOT / "image_splits"
                                     / f"split.{category}.val.json").read_text(encoding="utf-8"))]
    features = torch.load(feature, weights_only=True)
    provenance, verdict, differences = replay_tme.feature_cache_provenance(
        config.FASHIONIQ_ROOT, ".png", parameters, feature, mapping, features, prior)

    assert verdict == "VERIFIED_PRIOR_RUN", (
        f"an unchanged cache must verify; reported {differences}")
    assert differences == []
    # One meaning per field on BOTH sides: the emitted gallery identity is the ordered image bytes
    # the receipt recorded, and each side computes that same quantity from the dataset itself.
    assert provenance["feature_gallery_sha256"] == inputs["gallery_images"]
    assert provenance["feature_gallery_sha256"] == fashioniq_gallery_digests(
        config.FASHIONIQ_ROOT, parameters["split"], ".png")
    # The split-file digests remain a SEPARATE, still-emitted identity: separating the names must not
    # have deleted the second check.
    assert set(provenance["feature_split_files_sha256"]) == set(CATEGORIES)
    assert provenance["feature_split_files_sha256"] != provenance["feature_gallery_sha256"]

    # Genuine mutations still refuse to bind, under both identities.
    split = config.FASHIONIQ_ROOT / "image_splits" / "split.dress.val.json"
    original = split.read_text(encoding="utf-8")
    (config.FASHIONIQ_ROOT / "images" / "dress2.png").write_bytes(b"image-dress2")
    split.write_text(json.dumps(["dress2", "dress1"]), encoding="utf-8")
    try:
        assert replay_tme._prior_feature_provenance(config) is None, (
            "a reordered gallery must stop the receipt from binding")
    finally:
        split.write_text(original, encoding="utf-8")

    image = config.FASHIONIQ_ROOT / "images" / "dress1.png"
    pixels = image.read_bytes()
    image.write_bytes(b"different pixels for the same id")
    try:
        assert replay_tme._prior_feature_provenance(config) is None, (
            "changed image bytes must stop the receipt from binding")
    finally:
        image.write_bytes(pixels)
    assert replay_tme._prior_feature_provenance(config) is not None, (
        "restoring the original bytes must let the receipt bind again")
