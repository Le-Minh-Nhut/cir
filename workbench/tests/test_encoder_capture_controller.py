from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typing import Any

from workbench.replay import encoder_capture


def _receipt(root: Path) -> dict:
    plan = {"schema": "encoder-capture-plan-v1", "category": "dress"}
    return {
        "schema": "encoder-native-receipt-v1",
        "run_root": str(root.resolve()),
        "plan": plan,
        "plan_sha256": encoder_capture._canonical_sha(plan),
        "argv": [],
        "interpreter": "",
        "returncode": 0,
    }


@pytest.mark.parametrize("mutate", ["cross_run", "plan_tamper", "receipt_mutated"])
def test_reconcile_rejects_untrusted_receipt_before_adapter(tmp_path: Path, mutate: str) -> None:
    root = tmp_path / "capture"
    root.mkdir()
    receipt = _receipt(root)
    (root / "native-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    if mutate == "cross_run":
        other_root = tmp_path / "other"
        other_root.mkdir()
        (other_root / "native-receipt.json").write_text(json.dumps(_receipt(other_root)), encoding="utf-8")
        receipt["run_root"] = str(other_root)
    elif mutate == "plan_tamper":
        receipt["plan"]["category"] = "shirt"
    else:
        receipt["status"] = "forged"

    class MustNotConstructAdapter:
        def __init__(self, *_args, **_kwargs):
            pytest.fail("Adapter must not run before receipt verification")

    with pytest.raises(ValueError):
        encoder_capture.reconcile(root, receipt, adapter_factory=MustNotConstructAdapter)



def test_preflight_inputs_follow_adapter_configuration() -> None:
    config = encoder_capture.resolve_config()
    adapter = encoder_capture.Adapter(config)
    source, dataset, checkpoint = encoder_capture._inputs(config)

    assert source == adapter.source_root
    assert dataset == adapter.dataset_root
    assert checkpoint == adapter._checkpoint_path

def test_native_capture_uses_source_cwd_and_worker_pythonpath(tmp_path: Path) -> None:
    source = tmp_path / "ENCODER"
    source.mkdir()
    asset = source / encoder_capture.AUXILIARY_ASSET_NAME
    asset.write_bytes(b"verified fixture asset")
    run_root = tmp_path / "capture"
    run_root.mkdir()
    code = ("import pathlib; import workbench.replay.encoder; "
            f"assert pathlib.Path('{encoder_capture.AUXILIARY_ASSET_NAME}').read_bytes() == b'verified fixture asset'")

    result = encoder_capture._run_native([str(encoder_capture.CONTROLLER), "-c", code], source, run_root)

    assert result.returncode == 0

def test_capture_rejects_low_disk_before_creating_run_or_starting_native(tmp_path: Path,
                                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    from collections import namedtuple

    source = tmp_path / "ENCODER"
    run_root = tmp_path / "immutable-run"
    plan = {"dataset_root": str(tmp_path / "dataset"), "source_root": str(source),
            "native_argv_template": ["python", "-m", "encoder", "--output", "<RUN_ROOT>/native-result.json"]}
    monkeypatch.setattr(encoder_capture, "preflight", lambda *args: plan)
    monkeypatch.setattr(encoder_capture, "_staging_peak_bytes", lambda dataset: 100)
    usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(encoder_capture.shutil, "disk_usage", lambda path: usage(1000, 999, 1))
    native_started = False

    def run_native(*args: Any, **kwargs: Any) -> None:
        nonlocal native_started
        native_started = True

    monkeypatch.setattr(encoder_capture, "_run_native", run_native)
    with pytest.raises(OSError, match="insufficient disk for ENCODER capture"):
        encoder_capture.main(["--phase", "capture", "--run-root", str(run_root)])
    assert not run_root.exists()
    assert not native_started




def test_image_manifest_binds_canonical_bytes_after_cache_cleanup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import numpy as np
    import shutil

    dataset = tmp_path / "dataset"
    image_root = dataset / "resized_image" / "dress"
    image_root.mkdir(parents=True)
    images = [image_root / f"{name}.jpg" for name in ("reference", "target", "gallery")]
    for path in images:
        path.write_bytes(path.name.encode())
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "dress").symlink_to(image_root)
    assert (cache / "dress" / "reference.jpg").resolve() == images[0].resolve()

    root = tmp_path / "capture"
    score_root = root / "native-scores"
    import numpy as np

    capture = encoder_capture._NativeArgsortCapture(np, score_root, [0], [1],
                                                     [path.stem for path in images],
                                                     ["reference"], ["target"])
    capture.query_texts = ["edit"]
    capture.query_paths = [str(images[0].resolve())]
    capture.gallery_paths = [str(path.resolve()) for path in images]
    capture.image_manifest_paths = sorted(set(capture.query_paths) | set(capture.gallery_paths))
    capture.image_sha256 = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                            for path in capture.image_manifest_paths}
    capture.argsort(np.asarray([1e11, -3, -2], dtype=np.float32))
    capture.finish("dress", {"dress_r1": 100.0, "dress_r10": 100.0, "dress_r50": 100.0})
    g1 = root / "native-g1"
    g1.mkdir()
    np.savez(g1 / "query-00000.npz", pixels=np.zeros((1, 3, 1, 1)), image_paths=np.asarray(capture.query_paths),
             reference_ids=np.asarray(capture.reference_ids), target_ids=np.asarray(capture.target_ids),
             reference_positions=np.asarray([0]), target_positions=np.asarray([1]), texts=np.asarray(["edit"]),
             input_ids=np.zeros((1, 77), dtype=np.int64))
    np.savez(g1 / "gallery-00000.npz", pixels=np.zeros((3, 3, 1, 1)), image_paths=np.asarray(capture.gallery_paths),
             gallery_ids=np.asarray(capture.gallery_ids))
    relation = root / f"relation-seed{encoder_capture.RELATION_SEED}.npy"
    np.save(relation, np.asarray([1.0], dtype=np.float32))
    relation_digest = hashlib.sha256(np.asarray([1.0], dtype=np.float32).tobytes()).hexdigest()
    relation_metadata = {"sha256": relation_digest, "seed": encoder_capture.RELATION_SEED,
                         "tensor": "backbone.binding_decoder.learnable_relation", "shape": [1],
                         "dtype": "torch.float32", "in_state_dict": False}
    Path(f"{relation}.json").write_text(json.dumps(relation_metadata), encoding="utf-8")
    source = tmp_path / "ENCODER"
    source.mkdir()
    for name in ("test.py", "model_try2.py", "datasets.py", encoder_capture.AUXILIARY_ASSET_NAME):
        (source / name).write_bytes(name.encode())
    dataset_assets = dataset / "image_splits"
    dataset_assets.mkdir(parents=True)
    captions = dataset / "captions"
    captions.mkdir()
    for name in ("split.dress.val.json",):
        (dataset_assets / name).write_text("[]")
    for name in ("cap.dress.val.json", "correction_dict_dress.json"):
        (captions / name).write_text("[]")
    checkpoint = tmp_path / "encoder.pt"
    checkpoint.write_bytes(b"checkpoint")
    tokenizer = tmp_path / "bpe.gz"
    tokenizer.write_bytes(b"bpe fixture")
    worker_python = str(tmp_path / "encoder-python")
    worker_info = {"python": worker_python, "versions": "numpy torch open_clip", "tokenizer_asset": {
        "path": str(tokenizer), "sha256": hashlib.sha256(tokenizer.read_bytes()).hexdigest()}}
    class FakeAdapter:
        def __init__(self, _config):
            self.source_root, self.dataset_root, self._checkpoint_path = source, dataset, checkpoint

        def require_verified_interpreter(self):
            return worker_python

        @staticmethod
        def status(_config):
            return type("Status", (), {"ready": True, "checkpoint_ids": ("pinned",), "reason": None})()

    monkeypatch.setattr(encoder_capture, "Adapter", FakeAdapter)
    monkeypatch.setattr(encoder_capture, "resolve_config", lambda: object())
    monkeypatch.setattr(encoder_capture, "_worker", lambda _config: Path(worker_python))
    monkeypatch.setattr(encoder_capture, "_native_preflight", lambda *_args: {"argv": [], "exit_code": 0})
    monkeypatch.setattr(encoder_capture, "_worker_probe", lambda _worker: worker_info)
    monkeypatch.setattr(encoder_capture, "DEFAULT_CHECKPOINT", {
        "checkpoint_id": "pinned", "expected_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_mapping_status": "VERIFIED_METADATA"})
    plan = encoder_capture.preflight()
    (root / "frozen-plan.json").write_text(json.dumps(plan), encoding="utf-8")
    command = [*plan["native_argv_template"][:-1], str(root / "native-result.json")]
    result = {"results": [{"category": "dress", "provenance": {
        "native_score_capture": {"score_path": str(score_root / "native_scores.float32.npy"),
                                 "order_path": str(score_root / "native_orders.int32.npy"),
                                 "ids_path": str(score_root / "source_ids.npz")},
        "relation_state": {"status": "recorded", "path": str(relation), "sha256": relation_digest,
                           "seed": encoder_capture.RELATION_SEED}}}] }
    (root / "native-result.json").write_text(json.dumps(result), encoding="utf-8")

    shutil.rmtree(cache)
    receipt = encoder_capture.bind_receipt(root, plan, command, 0)
    encoder_capture.verify_receipt(root, receipt)
    original_interpreter = receipt["interpreter"]
    receipt["interpreter"] = str(tmp_path / "wrong-python")
    (root / "native-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="native argv/interpreter/exit mismatch"):
        encoder_capture.verify_receipt(root, receipt)
    receipt["interpreter"] = original_interpreter
    (root / "native-receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    tokenizer.write_bytes(b"changed BPE")
    with pytest.raises(ValueError, match="OpenCLIP tokenizer BPE asset changed"):
        encoder_capture.verify_receipt(root, receipt)
    tokenizer.write_bytes(b"bpe fixture")
    digests = [capture.image_sha256[path] for path in capture.image_manifest_paths]
    assert receipt["image_content_sha256"] == dict(zip(capture.image_manifest_paths, digests))
    target = images[1]
    target.write_bytes(b"tampered")
    (root / "native-receipt.json").unlink()
    with pytest.raises(ValueError, match="native input image content changed"):
        encoder_capture.bind_receipt(root, plan, command, 0)

def test_g1_comparison_checks_native_and_adapter_batch_partitions(tmp_path: Path) -> None:
    import numpy as np

    from workbench.composition.model_encoder import _write_g1_manifest, _write_g1_shard

    root = tmp_path
    native = root / "native-g1"
    g1 = root / "g1"
    native.mkdir()
    query_adapter = g1 / "adapter-queries"
    gallery_adapter = g1 / "adapter-gallery"
    query_adapter.mkdir(parents=True)
    gallery_adapter.mkdir()
    paths = ["/data/a.jpg", "/data/b.jpg", "/data/c.jpg"]
    pixels = np.arange(3, dtype=np.float32).reshape(3, 1, 1, 1).repeat(3, axis=1)
    query_texts = ["x", "y", "z"]
    token_ids = np.arange(3 * 77, dtype=np.int64).reshape(3, 77)
    query_ids = {"query_paths": paths, "gallery_paths": paths,
                 "query_texts": query_texts, "reference_ids": ["a", "a", "b"],
                 "target_ids": ["b", "c", "c"], "reference_positions": [0, 0, 1],
                 "target_positions": [1, 2, 2], "gallery_ids": ["a", "b", "c"]}
    query_shards = []
    gallery_shards = []
    for index in range(3):
        query_shards.append(_write_g1_shard(
            query_adapter, index, "queries", pixels[index:index + 1], paths[index:index + 1],
            query_texts[index:index + 1], token_ids[index:index + 1]))
        gallery_shards.append(_write_g1_shard(
            gallery_adapter, index, "targets", pixels[index:index + 1], paths[index:index + 1], [], None))
    _write_g1_manifest(query_adapter, "queries", paths, query_texts, [1, 1, 1], query_shards)
    _write_g1_manifest(gallery_adapter, "targets", paths, [], [1, 1, 1], gallery_shards)
    for phase in ("query", "gallery"):
        for batch_index, start in enumerate((0, 2)):
            stop = min(start + 2, len(paths))
            selection = list(range(start, stop))
            payload = {"pixels": pixels[selection], "image_paths": [paths[i] for i in selection]}
            if phase == "query":
                for key in ("reference_ids", "target_ids", "reference_positions", "target_positions"):
                    payload[key] = [query_ids[key][i] for i in selection]
                payload["texts"] = [query_texts[i] for i in selection]
                payload["input_ids"] = token_ids[selection]
            else:
                payload["gallery_ids"] = [query_ids["gallery_ids"][i] for i in selection]
            np.savez(native / f"{phase}-{batch_index:05d}.npz", **payload)
    compared = encoder_capture._compare_g1_inputs(root, query_ids, 2, 1)
    assert compared["query_pixels_equal"] is True
    assert compared["query_input_ids_equal"] is True
    assert compared["gallery_pixels_equal"] is True
    assert compared["native_query_batch_sizes"] == [2, 1]
    assert compared["adapter_query_batch_sizes"] == [1, 1, 1]
    shard_hashes = encoder_capture._adapter_g1_shard_hashes(query_adapter)
    assert set(shard_hashes) == {
        "adapter-queries/manifest.json",
        "adapter-queries/batch-00000.npz",
        "adapter-queries/batch-00001.npz",
        "adapter-queries/batch-00002.npz",
    }
    assert all(len(digest) == 64 for digest in shard_hashes.values())

    _write_g1_shard(query_adapter, 1, "queries", pixels[1:2] + 10, paths[1:2],
                    query_texts[1:2], token_ids[1:2] + 1000)
    compared = encoder_capture._compare_g1_inputs(root, query_ids, 2, 1)
    assert compared["query_pixels_equal"] is False
    assert compared["query_pixel_max_abs_delta"] == 10
    assert compared["query_input_ids_equal"] is False

    query_manifest = json.loads((query_adapter / "manifest.json").read_text())
    query_manifest["batch_sizes"] = [2, 1]
    (query_adapter / "manifest.json").write_text(json.dumps(query_manifest))
    with pytest.raises(ValueError, match="invalid ENCODER Adapter G1 shard manifest"): 
        encoder_capture._compare_g1_inputs(root, query_ids, 2, 1)
    query_manifest["batch_sizes"] = [1, 1, 1]
    query_manifest["image_paths"][0] = "/data/wrong.jpg"
    (query_adapter / "manifest.json").write_text(json.dumps(query_manifest))
    with pytest.raises(ValueError, match="query order"):
        encoder_capture._compare_g1_inputs(root, query_ids, 2, 1)
