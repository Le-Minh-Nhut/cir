from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
PRODUCER = REPO / "workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py"
PINNED_TEST = REPO / "workbench/third_party/HINT/test.py"
PINNED_DATASETS = REPO / "workbench/third_party/HINT/datasets.py"


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module



def test_hint_capture_row_ids_preserve_sample_and_full_annotation_identity():
    producer = load(PRODUCER, "hint_capture_contract_test_producer")
    query_ids = [f"dress:{index}" for index in range(2017)]
    inputs = {"query_annotation_ids": query_ids, "sample_indices": [0, 14, 18, 816, 2016],
              "reference_ids": ["ref-a", "ref-b", "ref-a", "ref-c", "ref-d"],
              "target_ids": ["target-a", "target-b", "target-a", "target-d", "target-e"]}
    full_ids, sample_ids = producer._hint_capture_row_ids(inputs, inputs["sample_indices"])
    assert full_ids == query_ids
    assert sample_ids == [query_ids[index] for index in inputs["sample_indices"]]
    assert len(sample_ids) == len(inputs["reference_ids"]) == len(inputs["target_ids"]) == 5
    assert len(set(inputs["reference_ids"])) == len(set(inputs["target_ids"])) == 4
    assert len(set(sample_ids)) == 5
    assert producer._hint_capture_row_ids(inputs) == (query_ids, query_ids)



def test_hint_plan_and_role_partitions_bind_annotation_rows_with_duplicate_references():
    producer = load(PRODUCER, "hint_partition_digest_test_producer")
    annotations = [f"dress:{index}:source:target:text" for index in range(5)]
    references = ["ref-a", "ref-b", "ref-a", "ref-c", "ref-a"]
    gallery = ["ref-a", "ref-b", "ref-c", "target-a", "target-b"]

    for batch_size in (1, 3):
        planned = producer._hint_execution_partitions(
            annotations, references, gallery, batch_size)
        native_role = producer._hint_execution_partitions(
            list(annotations), list(references), list(gallery), batch_size)
        adapter_role = producer._hint_execution_partitions(
            list(annotations), list(references), list(gallery), batch_size)

        assert len(set(annotations)) == len(annotations)
        assert len(set(references)) < len(references)
        assert planned["query_partition_sha256"] == producer.digest_ids(annotations)
        assert planned["query_order_sha256"] == producer.digest_ids(annotations)
        assert planned["reference_order_sha256"] == producer.digest_ids(references)
        assert planned["query_partition_sha256"] != planned["reference_order_sha256"]
        assert planned["query_partition_sha256"] == native_role["query_partition_sha256"]
        assert planned["query_partition_sha256"] == adapter_role["query_partition_sha256"]
        assert planned["batch_partition_sha256"] == native_role["batch_partition_sha256"]
        assert planned["batch_partition_sha256"] == adapter_role["batch_partition_sha256"]
        assert planned["gallery_partition_sha256"] == native_role["gallery_partition_sha256"]
        assert planned["gallery_partition_sha256"] == adapter_role["gallery_partition_sha256"]
        assert planned["batch_partition_sha256"] == producer._batch_partition_digest(
            annotations, gallery, batch_size)[2]
def test_hint_adapter_phase_reads_annotation_capture_and_binds_score_order(tmp_path, monkeypatch):
    producer = load(PRODUCER, "hint_adapter_phase_capture_test_producer")
    inputs = {
        "sample_indices": [0, 1], "query_annotation_ids": ["dress:row-0", "dress:row-1"],
        "query_reference_paths": ["/images/ref.jpg", "/images/ref.jpg"],
        "query_target_paths": ["/images/t0.jpg", "/images/t1.jpg"],
        "query_texts": ["red bag", "blue bag"], "reference_ids": ["ref", "ref"],
        "target_ids": ["t0", "t1"], "gallery_sample_paths": ["/images/t0.jpg", "/images/t1.jpg"],
        "gallery_sample_ids": ["t0", "t1"], "gallery_ids": ["ref", "t0", "t1"],
        "full_query_paths": ["/images/ref.jpg", "/images/ref.jpg"],
        "full_query_texts": ["red bag", "blue bag"], "full_reference_ids": ["ref", "ref"],
        "full_target_ids": ["t0", "t1"],
        "gallery_paths": ["/images/ref.jpg", "/images/t0.jpg", "/images/t1.jpg"],
    }
    root = tmp_path / "raw/roles/dress/native_adapter_native"
    root.mkdir(parents=True)
    (root / "sample_hint_dress_inputs.json").write_text(json.dumps(inputs))
    pixels = np.zeros((2, 3, 2, 2), dtype=np.float32)
    gallery_pixels = np.zeros((2, 3, 2, 2), dtype=np.float32)
    query = np.array([[1, 0], [0, 1]], dtype=np.float32)
    targets = np.array([[[1, 0]], [[0, 1]]], dtype=np.float32)
    gallery = np.array([[[1, 1]], [[1, 0]], [[0, 1]]], dtype=np.float32)
    native = {
        "native_query": query, "native_target": targets,
        "native_query_ids": np.array(inputs["query_annotation_ids"]),
        "native_target_ids": np.array(inputs["query_annotation_ids"]),
        "native_gallery_sample_ids": np.array(inputs["gallery_sample_ids"]),
        "native_query_paths": np.array(inputs["query_reference_paths"]),
        "native_target_paths": np.array(inputs["query_target_paths"]),
        "native_gallery_sample_paths": np.array(inputs["gallery_sample_paths"]),
        "native_query_pixels": pixels, "native_target_pixels": pixels.copy(),
        "native_gallery_sample_pixels": gallery_pixels.copy(),
        "native_query_processed_texts": np.array(inputs["query_texts"]),
        "native_query_token_ids": np.array([[1, 2], [3, 4]], dtype=np.int64),
        "native_query_token_mask": np.ones((2, 2), dtype=np.int64),
        "population_native_queries": query,
        "population_native_gallery": gallery,
        "population_native_query_paths": np.array(inputs["full_query_paths"]),
        "population_native_gallery_paths": np.array(inputs["gallery_paths"]),
    }
    np.savez(root / "sample_hint_dress_native.npz", **native)
    with np.load(root / "sample_hint_dress_native.npz") as captured_native:
        assert captured_native["native_query_processed_texts"].tolist() == inputs["query_texts"]
    for role in ("native_batch_1", "native_batch_n"):
        batch_root = tmp_path / f"raw/roles/dress/{role}"
        batch_root.mkdir(parents=True)
        (batch_root / "sample_hint_dress_inputs.json").write_text(json.dumps(inputs))
        np.savez(batch_root / "sample_hint_dress_native.npz", **native)

    class FakeAdapter:
        batch_size = 32
        mode = "exact"

        def __init__(self, config):
            self.work_root = None
            self.batch_size = 32

        def encode_queries(self, paths, texts, categories, *, capture_path=None, capture_ids=None,
                           selected_capture_ids=None):
            if capture_path is not None:
                all_ids: list[str] = list(capture_ids or ())
                selected = list(selected_capture_ids or all_ids)
                rows = [index for index, value in enumerate(all_ids) if value in selected]
                ids = [all_ids[index] for index in rows]
                if self.mode == "wrong-id" and ids:
                    ids[0] = "ref"
                processed = [texts[index] for index in rows]
                if self.mode == "wrong-text" and processed:
                    processed[0] = "conflicting processed text"
                np.savez(capture_path, pixels=pixels[rows], image_paths=np.array(paths)[rows],
                         capture_ids=np.array(ids), phase=np.array("queries"),
                         processed_texts=np.array(processed),
                         token_ids=native["native_query_token_ids"][rows],
                         token_mask=native["native_query_token_mask"][rows])
            return query[:1].copy() if len(paths) == 1 else query.copy()

        def encode_targets(self, paths, categories, *, capture_path=None, capture_ids=None,
                           selected_capture_ids=None):
            if capture_path is not None:
                capture_pixels = gallery_pixels if "gallery-sample" in str(capture_path) else pixels
                all_ids: list[str] = list(capture_ids or ())
                selected = list(selected_capture_ids or all_ids)
                rows = [index for index, value in enumerate(all_ids) if value in selected]
                ids = [all_ids[index] for index in rows]
                if self.mode == "wrong-id" and "targets.npz" in str(capture_path) and ids:
                    ids[0] = "ref"
                np.savez(capture_path, pixels=capture_pixels[rows], image_paths=np.array(paths)[rows],
                         capture_ids=np.array(ids), phase=np.array("targets"))
            return targets.copy() if len(paths) == 2 else gallery.copy()

        def score(self, queries, targets):
            return np.array([[0.9, 0.8, 0.1], [0.2, 0.1, 0.9]], dtype=np.float32)

        def rank(self, queries, targets, target_positions):
            return [1, 1]


        def restart(self):
            pass
        def close(self):
            pass

    class Hook:
        scope = producer.HintCapture.scope

    monkeypatch.setattr(producer, "_adapter_class", lambda hook: FakeAdapter)
    monkeypatch.setattr(producer, "RAW", tmp_path / "raw")

    def run(mode, role):
        FakeAdapter.mode = mode
        batch_size = 1 if role == "adapter_batch_1" else 32
        FakeAdapter.batch_size = batch_size
        monkeypatch.setattr("workbench.composition.adapters.resolved_batch_size", lambda: batch_size)
        return producer.adapter_phase(Hook(), "dress", object(), batch_size, role)

    exact = run("exact", "native_adapter_adapter")
    capture = exact["hint_input_capture"]
    assert capture["query_pixels_equal"] and capture["target_pixels_equal"]
    assert capture["text_equal"] and capture["token_ids_equal"] and capture["token_mask_equal"]
    assert exact["layer2_pixels"]["query_ids"] == inputs["query_annotation_ids"]
    assert exact["layer2_tokens"]["processed_texts_exact_match"]
    role_root = tmp_path / "raw/roles/dress/native_adapter_adapter"
    score = np.load(role_root / "adapter_scores.npy")
    order = np.load(role_root / "adapter_orders.npy")
    assert score[:, 0].tolist() == np.full(2, np.float32(-10e10)).tolist()
    assert exact["v3_population_scoring"]["score_order_sha256"] == producer._hint_score_order_digest(
        inputs["query_annotation_ids"], score, order)

    wrong_text = run("wrong-text", "adapter_batch_1")
    assert not wrong_text["hint_input_capture"]["text_equal"]
    assert not wrong_text["layer2_tokens"]["processed_texts_exact_match"]
    with pytest.raises(ValueError, match="full-population capture row identity/order mismatch"):
        run("wrong-id", "adapter_batch_n")

def test_one_category_source_loader_and_observer(monkeypatch, tmp_path):
    producer = load(PRODUCER, "hint_observer_cpu_test_producer")
    pinned = load(PINNED_TEST, "hint_observer_cpu_test_pinned")
    datasets = load(PINNED_DATASETS, "hint_observer_cpu_test_datasets")
    image_root = tmp_path / "resized_image" / "dress"
    image_root.mkdir(parents=True)
    (tmp_path / "image_splits").mkdir()
    (tmp_path / "captions").mkdir()
    ids = ["B000000001", "B000000002", "B000000003"]
    (tmp_path / "image_splits/split.dress.val.json").write_text(json.dumps(ids))
    (tmp_path / "captions/cap.dress.val.json").write_text(json.dumps([
        {"candidate": ids[0], "target": ids[1], "captions": ["one", "two"]},
        {"candidate": ids[1], "target": ids[2], "captions": ["three", "four"]},
        {"candidate": ids[0], "target": ids[2], "captions": ["five", "six"]},
    ]))
    (tmp_path / "captions/correction_dict_dress.json").write_text("{}")
    for index, image_id in enumerate(ids):
        Image.new("RGB", (2, 2), (index, 0, 0)).save(image_root / f"{image_id}.jpg")

    dataset = producer.load_hint_category_dataset(
        datasets, tmp_path, "dress",
        lambda image: torch.tensor([image.getpixel((0, 0))[0]], dtype=torch.float32))
    queries = dataset.test_queries_dress
    targets = dataset.test_targets_dress
    assert not hasattr(dataset, "test_queries_shirt")
    assert [row["target_img_id"] for row in targets] == [0, 1, 2]
    assert queries[-1] is dataset.test_queries_dress[-1]
    assert queries[0]["source_img_data"] is targets[0]["target_img_data"]
    assert queries[0]["target_img_data"] is queries[1]["source_img_data"]
    query_ids = [
        f"dress:{index}:{row['source_img_id']}:{row['target_img_id']}:"
        f"{__import__('hashlib').sha256(row['mod']['str'].encode('utf-8')).hexdigest()}"
        for index, row in enumerate(queries)]
    class Model:
        def __init__(self):
            self.forward_calls = 0

        def eval(self):
            return self
        def extract_retrieval_compose(self, images, texts):
            self.forward_calls += 1
            features = torch.zeros((len(texts), 1, 1, 3))
            features[0, 0, 0, 2 if len(texts) == 1 else 1] = 1
            if len(texts) > 1:
                features[1, 0, 0, 0] = 1
            return features

        def extract_retrieval_target(self, images):
            self.forward_calls += 1
            features = torch.zeros((len(images), 3, 32))
            for row, image in enumerate(images):
                features[row, int(image[0]), :] = 1
            return features

    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *args, **kwargs: self)
    gallery = ids
    references = [ids[0], ids[1], ids[0]]
    targets_ids = [ids[1], ids[2], ids[2]]
    paths = lambda values: [str(image_root / f"{value}.jpg") for value in values]
    original_numpy = pinned.np
    model = Model()
    observed = producer.observe_hint_evaluator(
        pinned, SimpleNamespace(batch_size=2, dataset="fashioniq"), model, dataset, "dress",
        {"eval": lambda text: text}, gallery, references, targets_ids,
        paths(gallery), paths(references), paths(targets_ids), tmp_path / "source-scores",
        query_ids)
    assert model.forward_calls > 0

    assert pinned.np is original_numpy
    assert observed["ranks"] == [1, 2, 1]
    assert observed["hit_counts"] == {"r1": 2, "r10": 3, "r50": 3}
    assert observed["duplicate_sort_calls_identical"] is True
    assert isinstance(observed["scores"], np.memmap) and observed["scores"].dtype == np.float32
    assert isinstance(observed["orders"], np.memmap) and observed["orders"].dtype == np.int32
    artifacts = producer.hint_source_artifacts(observed)
    assert artifacts["pinned_source_ranks"].tolist() == [1, 2, 1]
    assert artifacts["pinned_source_topk_target_hits"].tolist() == [[True] * 3, [False, True, True], [True] * 3]
    assert len(set(query_ids)) == len(queries)
    assert len(set(references)) < len(references)
    assert observed["score_order_sha256"] == producer._hint_score_order_digest(
        query_ids, observed["scores"], observed["orders"])
    assert observed["score_order_sha256"] != producer._hint_score_order_digest(
        references, observed["scores"], observed["orders"])
    with pytest.raises(ValueError, match="dimensions"):
        producer._hint_score_order_digest(query_ids, observed["scores"], observed["orders"][:, :-1])
    assert observed["scores"].shape == observed["orders"].shape == (3, 3)
    with pytest.raises(ValueError, match="paths/order"):
        producer.observe_hint_evaluator(
            pinned, SimpleNamespace(batch_size=2, dataset="fashioniq"), Model(), dataset, "dress",
            {"eval": lambda text: text}, gallery, references, targets_ids,
            paths(list(reversed(gallery))), paths(references), paths(targets_ids),
            tmp_path / "wrong-path-scores", query_ids)
    rejected_model = Model()
    queries[0]["mod"]["str"] = "wrong caption"
    with pytest.raises(ValueError, match="annotation source/target/text order"):
        producer.observe_hint_evaluator(
            pinned, SimpleNamespace(batch_size=2, dataset="fashioniq"), rejected_model,
            dataset, "dress", {"eval": lambda text: text}, gallery, references, targets_ids,
            paths(gallery), paths(references), paths(targets_ids), tmp_path / "wrong-annotation",
            query_ids)
    assert rejected_model.forward_calls == 0
    plan = {
        "identity": {"query_count": len(query_ids), "gallery_count": len(gallery),
                     "query_order_sha256": producer.digest_ids(query_ids),
                     "gallery_order_sha256": producer.digest_ids(gallery)},
        "scoring_labels": {
            "query_targets": [[query_id, target_id] for query_id, target_id in zip(query_ids, targets_ids)],
            "reference_exclusions": [[query_id, [reference]]
                                      for query_id, reference in zip(query_ids, references)],
        },
    }
    native_hash = producer._hint_score_order_digest(query_ids, observed["scores"], observed["orders"])
    assert native_hash == observed["score_order_sha256"]
    adapter_scores = np.asarray(observed["scores"]).copy()
    adapter_orders = np.asarray(observed["orders"]).copy()
    adapter_hash = producer._hint_score_order_digest(query_ids, adapter_scores, adapter_orders)
    assert adapter_hash == native_hash
    assert pinned.np is original_numpy
