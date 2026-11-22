from __future__ import annotations

import numpy as np
import pytest
import torch

from workbench.replay.run_fashioniq_v3 import ROOT, _author_validation_sort, _capture_ids, _labels, _observe_qformer_inputs, _save_input_capture, _source_metrics
from pathlib import Path
import hashlib
import json

from workbench.composition.model_sprc import Adapter, _target_transport
from workbench.composition.adapters import CompositionAdapter


@pytest.mark.parametrize("batch_size", [1, 3])
def test_sprc_target_transport_preserves_per_token_max_scoring(batch_size: int) -> None:
    raw = np.zeros((batch_size, 32, 256), dtype=np.float32)
    raw[:, 0, 0] = 1.0
    raw[:, 1, 1] = 1.0
    normalized = raw / np.maximum(np.linalg.norm(raw, axis=-1, keepdims=True), 1e-12)
    transported = _target_transport(normalized)
    assert transported.shape == (batch_size, 8192)
    np.testing.assert_array_equal(transported.reshape(batch_size, 32, 256), normalized)
    queries = np.zeros((2, 256), dtype=np.float32)
    queries[:, 0] = 1.0
    adapter = Adapter.__new__(Adapter)
    scores = adapter.score(queries, transported)
    np.testing.assert_array_equal(scores, np.ones((2, batch_size), dtype=np.float32))


def test_input_capture_is_bounded_and_persisted(tmp_path: Path) -> None:
    ids = ["dress:0000:ref:target"]
    assert _capture_ids({"input_capture": {"query_ids": ids}}, "query") == set(ids)
    with pytest.raises(ValueError, match="invalid bounded"):
        _capture_ids({"input_capture": {"query_ids": [str(index) for index in range(6)]}}, "query")
    arrays = {"transformed_pixels": np.zeros((1, 3, 2, 2), dtype=np.float32), "token_ids": np.array([[1, 2]])}
    manifest = _save_input_capture(tmp_path / "run.npz", "query", ids, arrays)
    capture = Path(manifest["path"])
    assert hashlib.sha256(capture.read_bytes()).hexdigest() == manifest["sha256"]
    with np.load(capture, allow_pickle=False) as saved:
        assert saved["ordered_ids"].tolist() == ids
        np.testing.assert_array_equal(saved["transformed_pixels"], arrays["transformed_pixels"])
        np.testing.assert_array_equal(saved["token_ids"], arrays["token_ids"])


def test_production_qformer_capture_skips_image_only_then_captures_multimodal_rows() -> None:
    query_ids = ["dress:0000:shared:target0", "dress:0001:shared:target1", "dress:0002:shared:target2"]
    selected_ids = {query_ids[0], query_ids[2]}
    tokens, masks = [], []
    inputs = torch.tensor([[1, 2], [3, 4], [5, 6]])
    mask = torch.tensor([[1, 1], [1, 0], [1, 1]])
    assert _observe_qformer_inputs((), {"query_embeds": torch.zeros((3, 1, 2))}, query_ids,
                                   selected_ids, tokens, masks) == []
    assert not tokens and not masks
    observed = _observe_qformer_inputs((inputs,), {"attention_mask": mask}, query_ids,
                                       selected_ids, tokens, masks)
    assert observed == [query_ids[0], query_ids[2]]
    assert tokens and masks
    assert tokens[0].tolist() == [[1, 2], [5, 6]]
    assert masks[0].tolist() == [[1, 1], [1, 1]]


def test_source_metric_uses_native_distance_sort_and_eligible_gallery() -> None:
    scores = np.array([[0.1, 0.9, 0.3], [0.7, 0.2, 0.4]], dtype=np.float32)
    ranks, labels, recalls = _source_metrics(scores, ["dress:0000:ref:a", "dress:0001:ref:b"], ["a", "b", "c"], torch)
    np.testing.assert_array_equal(ranks, [[1, 2, 0], [0, 2, 1]])
    np.testing.assert_array_equal(labels, [[False, False, True], [False, False, True]])
    assert recalls == {"recall10": 100.0, "recall50": 100.0}


def test_source_metric_rejects_missing_or_duplicate_positive() -> None:
    scores = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="one eligible positive"):
        _source_metrics(scores, ["dress:0000:ref:missing"], ["a", "b"], torch)
    with pytest.raises(ValueError, match="one eligible positive"):
        _source_metrics(scores, ["dress:0000:ref:a"], ["a", "a"], torch)

@pytest.mark.parametrize("model_id", ["csmcir", "sprc"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_native_sort_observes_author_score_device_and_final_order(device: str, model_id: str) -> None:
    pred_sim = torch.tensor([[0.7, 0.9, 0.9], [0.8, 0.1, 0.3]], device=device)
    source = ROOT / "workbench/third_party" / ("CSMCIR" if model_id == "csmcir" else "SPRC")
    ranks, score_device, sort_device = _author_validation_sort(source, model_id, pred_sim,
        ["c", "a"], ["a", "b", "c"], torch)
    expected = torch.argsort(1 - pred_sim, dim=-1).cpu()
    assert score_device == sort_device == str(pred_sim.device)
    torch.testing.assert_close(ranks, expected, rtol=0, atol=0)
    assert ranks.tolist() == [[1, 2, 0], [0, 2, 1]]


def test_v3_target_labels_are_gallery_positions_for_native_and_adapter_ranks() -> None:
    query_ids = ["dress:0000:ref:a", "dress:0001:ref:c"]
    gallery_ids = ["a", "b", "c"]
    labels = _labels(query_ids, gallery_ids)
    np.testing.assert_array_equal(labels, [0, 2])
    ranks = np.array([[1, 0, 2], [2, 1, 0]])
    np.testing.assert_array_equal(ranks == labels[:, None], [[False, True, False], [True, False, False]])
    target_positions = [[int(index)] for index in labels]
    assert target_positions == [[0], [2]]

def test_production_adapter_rank_and_topk_uses_one_score_matrix() -> None:
    class Scores(CompositionAdapter):
        def __init__(self) -> None:
            self.calls = 0

        def score(self, queries: object, targets: object, categories: object = None) -> np.ndarray:
            self.calls += 1
            return np.array([[0.9, 0.9, 0.1]], dtype=np.float32)

    adapter = Scores()
    scores = np.array([[0.9, 0.9, 0.1]], dtype=np.float32)
    ranks, candidates = adapter.rank_and_topk(np.zeros((1, 1)), np.zeros((3, 1)), [[1]], [None], 2)
    assert adapter.calls == 1
    assert ranks == [2]
    assert candidates == [[(0, float(np.float32(0.9))), (1, float(np.float32(0.9)))]]
    order = np.argsort(-scores[0], kind="stable")
    assert [(pos, float(scores[0, pos])) for pos in order[:2]] == candidates[0]


def test_native_gallery_caption_capture_preserves_selected_raw_text(tmp_path: Path) -> None:
    from workbench.replay.run_fashioniq_v3 import _gallery_caption_capture, _read_adapter_input_capture, _save_input_capture

    captions = ["caption zero", "caption one", "caption two"]
    selected = _gallery_caption_capture(captions, [0, 2])
    assert selected.tolist() == ["caption zero", "caption two"]
    capture_output = tmp_path / "native.npz"
    manifest = _save_input_capture(capture_output, "gallery", ["id0", "id2"], {
        "transformed_pixels": np.ones((2, 3, 2, 2), np.float32),
        "token_ids": np.ones((2, 4), np.int64),
        "attention_mask": np.ones((2, 36), np.int64),
        "raw_text": selected,
        "processed_text": selected.copy(),
    })
    assert manifest["path"] == str(tmp_path / "native.inputs.gallery.npz")
    arrays, ids = _read_adapter_input_capture(tmp_path / "native.inputs.gallery.npz", ["id0", "id2"], scope="gallery")
    assert ids == ["id0", "id2"]
    np.testing.assert_array_equal(arrays["raw_text"], selected)
def test_native_gallery_target_loop_captures_raw_text_and_appends_targets() -> None:
    from workbench.replay.run_fashioniq_v3 import _append_gallery_target

    class Native:
        def __init__(self) -> None:
            self.calls = []

        def extract_target_caption_features(self, chunk, captions, *, mode):
            self.calls.append((chunk, captions, mode))
            return torch.as_tensor([[float(len(text))] for text in captions]), None

    native = Native()
    chunks = [torch.zeros((2, 3)), torch.zeros((1, 3))]
    caption_batches = [["caption zero", "caption one"], ["caption two"]]
    raw_text, processed_text, target_parts = [], [], []
    for chunk, captions, keep in zip(chunks, caption_batches, [[0], [0]]):
        _append_gallery_target(native, "csmcir", chunk, captions, keep, raw_text, processed_text, target_parts)
    assert len(native.calls) == 2
    assert [call[1] for call in native.calls] == caption_batches
    assert [call[2] for call in native.calls] == ["mean", "mean"]
    assert raw_text == processed_text == ["caption zero", "caption two"]
    torch.testing.assert_close(torch.cat(target_parts), torch.tensor([[12.0], [11.0], [11.0]]))


def test_sprc_gallery_target_loop_appends_targets_without_captions() -> None:
    from workbench.replay.run_fashioniq_v3 import _append_gallery_target

    class Native:
        def extract_target_features(self, chunk, *, mode):
            return chunk.sum(dim=1, keepdim=True), None

    target_parts = []
    raw_text, processed_text = [], []
    _append_gallery_target(Native(), "sprc", torch.tensor([[1.0, 2.0], [3.0, 4.0]]), None,
                           [], raw_text, processed_text, target_parts)
    torch.testing.assert_close(torch.cat(target_parts), torch.tensor([[3.0], [7.0]]))
    assert raw_text == processed_text == []

def test_fashioniq_v3_aggregate_emits_gate_pair(tmp_path: Path, monkeypatch) -> None:
    from datetime import datetime, timezone

    from workbench.backend import parity_gate
    from workbench.replay import aggregate_fashioniq_v3 as aggregator
    from workbench.replay import prepare_csmcir_sprc_v3_plans as producer
    from workbench.replay.run_fashioniq_v3 import _v3_capture

    dataset = tmp_path / "dataset"
    (dataset / "captions").mkdir(parents=True)
    (dataset / "image_splits").mkdir()
    pairs = [{"candidate": f"ref{i}", "target": f"target{i}", "captions": ["blue", "soft"]}
             for i in range(70)]
    gallery = [f"ref{i}" for i in range(70)] + [f"target{i}" for i in range(70)]
    (dataset / "captions/cap.dress.val.json").write_text(json.dumps(pairs))
    (dataset / "dress_cot_val.json").write_text(json.dumps({
        item: {"Final_Caption": "blue soft garment"} for item in gallery}))
    (dataset / "image_splits/split.dress.val.json").write_text(json.dumps(gallery))
    monkeypatch.setattr(producer, "DATA", dataset)
    monkeypatch.setattr(producer, "CAPTIONS", dataset)
    sources, python, checkpoint, protocol, needs_captions = producer.source_info("csmcir")
    adapter_source = ROOT / "workbench/replay/prepare_csmcir_sprc_v3_plans.py"
    plan = producer.build_plan("csmcir", "dress", checkpoint, sources, adapter_source,
                               python, protocol, needs_captions,
                               {"python": "fixture", "torch": "fixture"})
    root = tmp_path / "evidence"
    root.mkdir()
    plan_path = root / "g1_g2.csmcir.dress.v3-plan.json"
    plan_sha = producer.write_plan(plan_path, plan)
    log = root / "runner.jsonl"
    query_ids = plan["population_scope"]["query_ids"]
    gallery_ids = plan["population_scope"]["gallery_ids"]
    with log.open("w", encoding="utf-8") as stream:
        for role in aggregator.ROLE_ORDER:
            run = plan["runs"][role]
            qids = query_ids if run["identity"]["query_count"] == len(query_ids) else plan["population_scope"]["control_query_ids"]
            gids = gallery_ids if run["identity"]["gallery_count"] == len(gallery_ids) else plan["population_scope"]["control_gallery_ids"]
            scores = np.tile(np.arange(len(gids), 0, -1, dtype=np.float32), (len(qids), 1))
            if role in ("native_adapter_native", "native_adapter_adapter"):
                scores[:, 0] = np.float32(0.8000000 if role == "native_adapter_native" else 0.8000002)
                scores[:, 1] = np.float32(0.8000002 if role == "native_adapter_native" else 0.8000000)
            ranks = (np.argsort(1 - scores, axis=1) if role == "native_adapter_native"
                     else np.argsort(-scores, axis=1, kind="stable")).astype(np.int64)
            labels = np.asarray([gids.index(qid.rsplit(":", 1)[-1]) for qid in qids], dtype=np.int64)
            arrays = {"query_features": (np.ones((len(qids), 256), np.float32), qids),
                      "target_features": (np.ones((len(gids), 8192), np.float32), gids),
                      "scores": (scores, qids), "ranks": (ranks, qids), "labels": (labels, qids)}
            archive = root / f"{plan_path.stem}.{role}.{run['source_id'].split(':', 1)[0]}.npz"
            np.savez_compressed(archive, **{name: value for name, (value, _) in arrays.items()},
                                query_ids=np.asarray(qids), gallery_ids=np.asarray(gids))
            receipt = _v3_capture(plan, role, run["source_id"].split(":", 1)[0],
                                  datetime.now(timezone.utc).isoformat(), plan_path, plan_sha, arrays)
            receipt["role"] = role
            if role in ("native_adapter_native", "native_adapter_adapter"):
                from workbench.replay.run_fashioniq_v3 import _identity_evidence
                role_texts = [f"{row['captions'][0].strip('.?, ').capitalize()} and {row['captions'][1].strip('.?, ')}"
                              for row in pairs]
                receipt["identity_evidence"] = _identity_evidence(
                    role=role, query_paths=[dataset / "images" / f"{item}.png"
                                            for item in plan["population_scope"]["reference_ids"]],
                    gallery_paths=[dataset / "images" / f"{item}.png" for item in gallery_ids],
                    query_ids=query_ids, gallery_ids=gallery_ids, labels=labels,
                    model_texts=role_texts)
                capture = {}
                for scope in ("query", "gallery"):
                    population_ids = plan["population_scope"]["query_ids" if scope == "query" else "gallery_ids"]
                    ids = [item for item in population_ids if item in plan["input_capture"][f"{scope}_ids"]]
                    text_key = "raw_text"
                    capture_arrays = {"transformed_pixels": np.ones((len(ids), 3, 2, 2), np.float32),
                                      "token_ids": np.ones((len(ids), 4), np.int64),
                                      "attention_mask": np.ones((len(ids), 36), np.int64),
                                      "processed_text": np.asarray(["processed"] * len(ids)),
                                      text_key: np.asarray(role_texts[:len(ids)] if scope == "query" else ["blue soft garment"] * len(ids))}
                    capture_path = root / f"{role}.{scope}.npz"
                    np.savez_compressed(capture_path, ordered_ids=np.asarray(ids), **capture_arrays)
                    capture[scope] = {"path": str(capture_path), "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                        "ordered_ids": ids, "arrays": {key: list(value.shape) for key, value in capture_arrays.items()},
                        "array_sha256": {key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                                         for key, value in capture_arrays.items()},
                        text_key: capture_arrays[text_key].tolist(), "processed_text": ["processed"] * len(ids)}
                receipt["input_capture"] = capture
            stream.write(json.dumps({"role": role, "v3_receipt": receipt}) + "\n")

    result = aggregator.aggregate(plan_path, [log])
    assert result["checks"]["all_five_control_feature_tensors_within_preregistered_tolerance"] is True
    assert result["checks"]["all_five_controls_componentwise_within_preregistered_tolerance"] is False
    assert result["checks"]["feature_tensor_component_failures"] == []
    assert result["checks"]["score_order_label_diagnostic_failures"] == ["native_adapter.ranks: exact sorted-index rows differ"]
    assert result["checks"]["legacy_mixed_diagnostic_failures"] == result["checks"]["component_failures"]
    assert result["scoring_proof_sha256"]
    native_artifact = json.loads((root / "g1_g2.csmcir.dress.native.json").read_text())
    adapter_artifact = json.loads((root / "g1_g2.csmcir.dress.adapter.json").read_text())
    proof = root / plan["scoring_proof_path"]
    assert native_artifact["parity_v3"]["plan_sha256"] == plan_sha
    assert native_artifact["parity_v3"]["scoring_proof_sha256"] == result["scoring_proof_sha256"]
    assert adapter_artifact["identity"] == native_artifact["identity"]
    assert len(native_artifact["parity_v3"]["receipts"]) == len(aggregator.ROLE_ORDER)
    assert proof.is_file() and hashlib.sha256(proof.read_bytes()).hexdigest() == result["scoring_proof_sha256"]
    assert len(proof.read_text().splitlines()) == len(query_ids) + 1

    verdict = parity_gate.evaluate_model("csmcir", categories=("dress",), root=root,
                                         standard=parity_gate.STANDARD_ABS_REL_V3)
    assert verdict.status == parity_gate.PASS, verdict.detail
