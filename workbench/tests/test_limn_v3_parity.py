from __future__ import annotations

import hashlib
import importlib.util
import json
import time
import sys
import shutil
from datetime import datetime, timezone
from pathlib import Path
import re
import pytest

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from workbench.backend import parity_gate as gate


def _module():
    path = REPO / "workbench/scripts/limn_v3_parity.py"
    spec = importlib.util.spec_from_file_location("limn_v3_parity_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import producer fixture target: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _seed(root: Path, producer, t4, category="dress"):
    qcount, gcount, dimension = 100, 200, 768
    query_ids = [f"q:{category}:{index}" for index in range(qcount)]
    gallery_ids = [f"g:{category}:{index}" for index in range(gcount)]
    references = gallery_ids[qcount:]
    targets = [gallery_ids[0]] * qcount
    exclusions = [json.dumps({"excluded_gallery_ids": [references[index]]}) for index in range(qcount)]
    protocol = {"queries": [{"reference_id": ref, "target_id": target, "text": f"caption {i}"}
                             for i, (ref, target) in enumerate(zip(references, targets))],
                "query_ids": query_ids, "gallery_ids": gallery_ids, "target_ids": targets}
    protocol["reference_exclusions"] = exclusions
    protocols = getattr(producer, "_fixture_protocols", {})
    protocols[category] = protocol
    producer._fixture_protocols = protocols
    producer._protocol = lambda config, value: producer._fixture_protocols[value]
    producer.resolve_config = lambda: object()
    labels = {"query_targets": [[qid, target] for qid, target in zip(query_ids, targets, strict=True)],
              "reference_exclusions": [[qid, [references[index]]]
                                       for index, qid in enumerate(query_ids)]}
    labels_sha = hashlib.sha256(json.dumps(labels, ensure_ascii=False,
                            separators=(",", ":")).encode()).hexdigest()
    protocol["labels"] = labels
    protocol["labels_sha"] = labels_sha

    checkpoint = hashlib.sha256(b"fixture checkpoint").hexdigest()
    identity = {"checkpoint": checkpoint, "configuration": "cfg", "environment": "env",
                "protocol": "fashioniq_val_split", "category": category, "cohort_id": "full-cohort",
                "precision": "float32", "seed": 0,
                "query_order_sha256": producer._ordered_sha(query_ids),
                "gallery_order_sha256": producer._ordered_sha(gallery_ids),
                "reference_exclusions_sha256": labels_sha,
                "query_count": qcount, "gallery_count": gcount}
    sample_q, sample_g = query_ids[:33], gallery_ids[:64]
    sample_labels = {"query_targets": labels["query_targets"][:33],
                     "reference_exclusions": labels["reference_exclusions"][:33]}
    sample_hash = hashlib.sha256(json.dumps(sample_labels, ensure_ascii=False,
                            separators=(",", ":")).encode()).hexdigest()
    sample_identity = dict(identity, cohort_id="sample-cohort", query_count=33, gallery_count=64,
                           query_order_sha256=producer._ordered_sha(sample_q),
                           gallery_order_sha256=producer._ordered_sha(sample_g),
                           reference_exclusions_sha256=sample_hash)
    definitions = {
        "native_repeat_1": ("native", sample_identity, sample_q, sample_g, 32),
        "native_repeat_2": ("native", sample_identity, sample_q, sample_g, 32),
        "adapter_repeat_1": ("adapter", sample_identity, sample_q, sample_g, 32),
        "adapter_repeat_2": ("adapter", sample_identity, sample_q, sample_g, 32),
        "native_adapter_native": ("native", identity, query_ids, gallery_ids, 32),
        "native_adapter_adapter": ("adapter", identity, query_ids, gallery_ids, 32),
        "native_batch_1": ("native", sample_identity, sample_q, sample_g, 1),
        "native_batch_n": ("native", sample_identity, sample_q, sample_g, 32),
        "adapter_batch_1": ("adapter", sample_identity, sample_q, sample_g, 1),
        "adapter_batch_n": ("adapter", sample_identity, sample_q, sample_g, 32),
    }
    runs = {role: {"run_id": f"run:{role}", "source_id": f"source:{role}", "identity": run_identity,
                    "execution": producer._execution(category, run_identity["cohort_id"], qids, gids, batch)}
            for role, (side, run_identity, qids, gids, batch) in definitions.items()}
    control_defs = {
        "native_repeat": ("native_repeat_1", "native_repeat_2", "native"),
        "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2", "adapter"),
        "native_adapter": ("native_adapter_native", "native_adapter_adapter", "native"),
        "native_batch_1_vs_n": ("native_batch_1", "native_batch_n", "native"),
        "adapter_batch_1_vs_n": ("adapter_batch_1", "adapter_batch_n", "adapter"),
    }
    controls = {}
    for name, (ref_role, actual_role, reference_side) in control_defs.items():
        full = name == "native_adapter"
        run_identity = runs[ref_role]["identity"]
        role_q, role_g = (query_ids, gallery_ids) if run_identity is identity else (sample_q, sample_g)
        coverage = "full" if run_identity is identity else "sample"
        scopes = {"query": producer._scope(coverage, qcount, role_q, run_identity["cohort_id"]),
                  "gallery": producer._scope(coverage, gcount, role_g, run_identity["cohort_id"])}
        batch = "batch" in name
        controls[name] = {"reference_run_id": runs[ref_role]["run_id"],
                          "actual_run_id": runs[actual_role]["run_id"], "reference_side": reference_side,
                          "dimension": "batch_parity" if batch else "forward_parity", "dtype": "float32",
                          "tolerance": {"atol": 1e-4 if batch else 1e-5,
                                        "rtol": 1e-3 if batch else 1e-4}, "scopes": scopes}
    plan = {"standard": producer.STANDARD, "model_id": "limn", "category": category,
            "identity": identity, "scoring_labels": labels,
            "scoring_proof_path": f"g1_g2.limn.{category}.native.score-ranks.jsonl",
            "supported_batch_policy": {"supported": True, "batch_sizes": [1, 32]},
            "population_scope": {"coverage": "full", "query_count": qcount, "gallery_count": gcount},
            "runs": runs, "controls": controls}
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / f"g1_g2.limn.{category}.v3-plan.json"
    plan_sha = t4.write_v3_plan(plan_path, plan)
    time.sleep(0.02)
    roles_root = root / "roles" / category
    roles_root.mkdir(parents=True)
    rng = np.random.default_rng(17)
    qfeatures = rng.normal(size=(qcount, dimension)).astype(np.float32)
    gfeatures = rng.normal(size=(gcount, dimension)).astype(np.float32)
    receipts = {}
    arrays_by_role = {}
    for role, (side, run_identity, qids, gids, _) in definitions.items():
        qidx, gidx = len(qids), len(gids)
        arrays_by_role[role] = {"query_features": (qfeatures[:qidx], qids),
                                "gallery_features": (gfeatures[:gidx], gids)}
    for role, (side, run_identity, qids, gids, _) in definitions.items():
        run = runs[role]
        captured = {}
        for name, control in controls.items():
            if run["run_id"] in (control["reference_run_id"], control["actual_run_id"]):
                captured[f"{name}.query"] = arrays_by_role[role]["query_features"]
                captured[f"{name}.gallery"] = arrays_by_role[role]["gallery_features"]
        role_arrays = arrays_by_role[role]
        tensors = {key: value[0] for key, value in role_arrays.items()}
        tensors.update({f"{key}_ids": np.asarray(value[1]) for key, value in role_arrays.items()})
        if role == "native_adapter_adapter":
            tensors["query_features_image_ids"] = np.asarray(references)
            tensors["gallery_features_image_ids"] = np.asarray(gids)
        np.savez_compressed(roles_root / f"{role}.npz", **tensors)
        receipts[role] = t4.v3_run_receipt(run_id=run["run_id"], side=side,
                                            source_id=run["source_id"], plan_sha256=plan_sha,
                                            started_at=datetime.now(timezone.utc).isoformat(), arrays=captured,
                                            execution=run["execution"], identity=run_identity)
        (roles_root / f"{role}.receipt.json").write_text(json.dumps(receipts[role]), encoding="utf-8")

    # Independent bounded captures mimic native and Adapter source artifacts, not model evidence.
    sample_indices = [0, 1, 2]
    sample_refs = [references[i] for i in sample_indices]
    sample_targets = [targets[i] for i in sample_indices]
    sample_texts = [f"caption {i}" for i in sample_indices]
    query_paths = [f"query-{category}-{i}" for i in sample_indices]
    target_paths = [f"target-{category}-{i}" for i in sample_indices]
    pixels = np.arange(3 * 3 * 2 * 2, dtype=np.float32).reshape(3, 3, 2, 2)
    target_pixels = pixels + 100
    tokens = np.asarray([[1, 2, 0, 0], [3, 4, 0, 0], [5, 6, 0, 0]], dtype=np.int64)
    padding = tokens != 0
    causal = np.zeros((4, 4), dtype=np.float32)
    native_root = root / "native" / category
    native_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(native_root / f"g2_{category}_native.npz",
                        source_tensor=pixels, target_tensor=target_pixels,
                        sample_token_ids=tokens, sample_query_texts=np.asarray(sample_texts),
                        sample_text_padding_mask=padding, causal_attn_mask_actual=causal,
                        sample_query_ids=np.asarray(sample_refs), sample_target_ids=np.asarray(sample_targets),
                        sample_indices=np.asarray(sample_indices, dtype=np.int32))
    adapter_g1 = root / "adapter" / category / "g1"
    adapter_g1.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(adapter_g1 / "query-0000.npz", pixels=pixels, paths=np.asarray(query_paths),
                        texts=np.asarray(sample_texts), token_ids=tokens,
                        token_padding_mask_derived=padding, causal_attn_mask_actual=causal)
    np.savez_compressed(adapter_g1 / "gallery-0000.npz", pixels=target_pixels,
                        paths=np.asarray(target_paths))
    (native_root / f"g2_{category}_inputs.json").write_text(json.dumps({
        "full_reference_ids": references, "full_target_ids": targets, "gallery_ids": gallery_ids,
        "full_query_paths": [f"query-{i}" for i in range(qcount)],
        "gallery_paths": [f"gallery-{i}" for i in range(gcount)],
        "queries": [[query_paths[i], target_paths[i]] for i in range(len(sample_indices))],
        "query_texts": sample_texts, "reference_ids": sample_refs, "target_ids": sample_targets,
        "sample_indices": sample_indices}), encoding="utf-8")
    (native_root / "native-observation.json").write_text(json.dumps({"native_result": {
        "query_count_native": qcount, "query_count_annotations": qcount,
        "gallery_count_native": gcount, "gallery_count_rebuilt_from_annotations": gcount,
        "gallery_order_identical": True, "reference_ids_identical": True,
        "target_ids_identical": True, "model_text_identical_to_author_composition": True,
        "native_metrics_recheck": {}}}), encoding="utf-8")
    scores = np.zeros((qcount, gcount), dtype=np.float32)
    scores[:, 0] = 1.0
    for index in range(qcount):
        scores[index, qcount + index] = -10e10
    orders = np.argsort(-scores, axis=1).astype(np.int32)
    np.savez(native_root / f"g2_{category}_native_ranking.npz",
             source_target_ranks=np.ones(qcount, dtype=np.int32), source_score_rows=scores,
             source_orders=orders, query_ids=np.asarray(references),
             target_ids=np.asarray(targets), gallery_ids=np.asarray(gallery_ids),
             reference_positions=np.arange(qcount, gcount, dtype=np.int32),
             target_positions=np.zeros(qcount, dtype=np.int32))
    np.savez(native_root.parent.parent / "adapter" / category / f"g2_{category}_adapter_ranking.npz",
             adapter_source_target_ranks=np.ones(qcount, dtype=np.int32),
             adapter_masked_score_rows=scores.copy(), adapter_source_orders=orders.copy(),
             query_ids=np.asarray(references), gallery_ids=np.asarray(gallery_ids))
    for role, archive in (("native_adapter_native", native_root / f"g2_{category}_native_ranking.npz"),
                          ("native_adapter_adapter", native_root.parent.parent / "adapter" / category /
                           f"g2_{category}_adapter_ranking.npz")):
        receipt_path = roles_root / f"{role}.receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["ranking_archive_sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    (root / "adapter" / category / "adapter-observation.json").write_text(
        json.dumps({"adapter_result": {"scoring_ranking": {
            "adapter_source_target_ranks": [1] * qcount,
            "score_max_abs_delta_vs_dot": 0.0, "score_pairs_compared": qcount * gcount,
            "score_max_abs_declared": 1e-6, "score_within_tolerance": True,
            "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
            "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                         "min_gap_abs": 0.0}}}}), encoding="utf-8")
    return root


def test_limn_producer_assembly_passes_real_gate_and_missing_receipt_is_malformed(tmp_path, monkeypatch):
    producer = _module()
    t4 = producer._t4()
    root = _seed(tmp_path / "assembled", producer, t4)
    monkeypatch.setattr(producer, "_t4", lambda: t4)
    result = producer.aggregate_category(root, "dress")
    stage = tmp_path / "gate"
    stage.mkdir()
    plan = root / "g1_g2.limn.dress.v3-plan.json"
    shutil.copy2(plan, stage / plan.name)
    proof_name = result["native"]["parity_v3"]["scoring_proof_path"]
    shutil.copy2(root / proof_name, stage / proof_name)
    for side in ("native", "adapter"):
        (stage / f"g1_g2.limn.dress.{side}.json").write_text(json.dumps(result[side]), encoding="utf-8")
    verdict = gate.evaluate_model("limn", categories=("dress",), root=stage,
                                  standard=gate.STANDARD_ABS_REL_V3)
    assert verdict.status == gate.PASS, verdict.detail
    broken = json.loads((stage / "g1_g2.limn.dress.native.json").read_text())
    del broken["parity_v3"]["receipts"]["native_repeat_1"]
    (stage / "g1_g2.limn.dress.native.json").write_text(json.dumps(broken), encoding="utf-8")
    malformed = gate.evaluate_model("limn", categories=("dress",), root=stage,
                                    standard=gate.STANDARD_ABS_REL_V3)
    assert malformed.status == gate.MALFORMED, malformed.to_dict()


@pytest.mark.parametrize(
    ("capture_name", "error_message"),
    (
        ("query-0000.npz", "G1 pixel mismatch: Adapter pixels differ from native validation decode"),
        ("gallery-0000.npz", "G1 target/gallery pixel mismatch: Adapter gallery pixels differ from native validation decode"),
    ),
)
def test_limn_producer_rejects_mutated_adapter_g1_pixels(tmp_path, monkeypatch, capture_name, error_message):
    producer = _module()
    t4 = producer._t4()
    root = _seed(tmp_path / "mismatch", producer, t4)
    monkeypatch.setattr(producer, "_t4", lambda: t4)
    capture = root / "adapter" / "dress" / "g1" / capture_name
    with np.load(capture, allow_pickle=False) as blob:
        arrays = {key: blob[key].copy() for key in blob.files}
    arrays["pixels"][0, 0, 0, 0] += 1
    np.savez_compressed(capture, **arrays)
    with pytest.raises(ValueError, match="^" + re.escape(error_message) + "$"):
        producer.aggregate_category(root, "dress")


def test_limn_publisher_archives_ranks_and_refuses_incomplete_second_publish(tmp_path, monkeypatch):
    producer = _module()
    t4 = producer._t4()
    root = tmp_path / "run"
    for category in producer.CATEGORIES:
        _seed(root, producer, t4, category=category)
    monkeypatch.setattr(producer, "_t4", lambda: t4)
    publication = tmp_path / "publication"
    generation = producer.publish(root, publication)
    verdict = gate.evaluate_model("limn", root=generation, standard=producer.STANDARD)
    assert verdict.status == gate.PASS, verdict.to_dict()
    manifest = json.loads((publication / "manifests" / f"{generation.name}.json").read_text())
    for category in producer.CATEGORIES:
        for side in ("native", "adapter"):
            artifact_name = f"g1_g2.limn.{category}.{side}.json"
            artifact = json.loads((generation / artifact_name).read_text())
            evidence = artifact["ranking_evidence"]
            evidence_path = publication / evidence["path"]
            assert evidence_path.is_file()
            assert hashlib.sha256(evidence_path.read_bytes()).hexdigest() == evidence["sha256"]
            assert evidence["path"] in manifest["raw_assets"]
    current_before = (publication / "CURRENT.json").read_bytes()
    (root / "roles" / "dress" / "native_repeat_1.receipt.json").unlink()
    try:
        producer.publish(root, publication)
    except FileNotFoundError as error:
        assert "missing actual role receipt" in str(error)
    else:
        raise AssertionError("incomplete second publication unexpectedly succeeded")
    assert (publication / "CURRENT.json").read_bytes() == current_before


def test_source_concat_observer_retains_post_normalized_gemm_operands():
    path = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py"
    spec = importlib.util.spec_from_file_location("limn_t4_parity_capture_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import scorer capture target: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    query_rows = [np.asarray([[3.0, 4.0]], dtype=np.float32),
                  np.asarray([[5.0, 12.0]], dtype=np.float32)]
    gallery_rows = [np.asarray([[8.0, 15.0]], dtype=np.float32),
                    np.asarray([[7.0, 24.0]], dtype=np.float32)]
    proxy, operands = module._source_concat_observer(
        np, np.argsort, query_count=2, gallery_count=2)
    query = proxy.concatenate(query_rows)
    gallery = proxy.concatenate(gallery_rows)
    for values in (query, gallery):
        for row in range(len(values)):
            values[row] /= np.linalg.norm(values[row])
    scores = proxy.matmul(query, gallery.T)

    assert operands["query"] is query
    assert operands["gallery"] is gallery
    assert np.array_equal(operands["query"], query)
    assert np.array_equal(operands["gallery"], gallery)
    assert np.array_equal(query @ gallery.T, scores)
    assert not np.array_equal(query, np.concatenate(query_rows))
