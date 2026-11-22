"""Measure one pinned TG-CIR native/Adapter FashionIQ category pair."""
from __future__ import annotations
from dataclasses import asdict

import argparse
import hashlib
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from workbench.backend.operator_config import resolve_config
from workbench.composition.model_tgcir import Adapter
from workbench.replay.tgcir import evaluate_categories
from workbench.replay.tensor_spool import disk_backed_images

G1_LIMIT = 5


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()

def scoring_query_ids(query_image_names: list[str]) -> list[str]:
    """Give repeated candidate images distinct stable identities as annotation query rows."""
    return [f"{name}#row-{index}" for index, name in enumerate(query_image_names)]


def write_score_rank_proof(output: Path, identity: dict[str, Any], query_ids: list[str],
                           gallery_ids: list[str], target_indices: list[int],
                           reference_indices: list[int], native_scores: np.ndarray,
                           adapter_scores: np.ndarray, native_orders: np.ndarray,
                           adapter_orders: np.ndarray) -> tuple[Path, str, str]:
    """Persist all observed scores and orders; result remains diagnostic until V3 roles are preregistered."""
    if not (len(query_ids) == len(target_indices) == len(reference_indices)):
        raise ValueError("TG-CIR score proof label counts differ from query population")
    if len(set(query_ids)) != len(query_ids) or len(set(gallery_ids)) != len(gallery_ids):
        raise ValueError("TG-CIR score proof IDs must be unique and ordered")
    labels = {
        "query_targets": [[qid, gallery_ids[target]] for qid, target in zip(query_ids, target_indices)],
        "reference_exclusions": [[qid, [gallery_ids[reference]]]
                                 for qid, reference in zip(query_ids, reference_indices)],
    }
    canonical = json.dumps(labels, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if _sha(canonical) != identity["reference_exclusions_sha256"]:
        raise RuntimeError("TG-CIR score proof exclusion identity mismatch")
    shape = (len(query_ids), len(gallery_ids))
    if any(array.shape != shape for array in (native_scores, adapter_scores, native_orders, adapter_orders)):
        raise RuntimeError("TG-CIR score proof arrays differ from declared population")
    expected_order = np.arange(len(gallery_ids))
    for side, scores, orders in (("native", native_scores, native_orders),
                                 ("adapter", adapter_scores, adapter_orders)):
        if not np.isfinite(scores).all():
            raise RuntimeError(f"TG-CIR {side} score proof contains non-finite score")
        if any(not np.array_equal(np.sort(order), expected_order) for order in orders):
            raise RuntimeError(f"TG-CIR {side} score proof contains invalid gallery order")
    header = {"schema": "cir.score-ranks/v1", "identity": identity, "query_ids": query_ids,
              "gallery_ids": gallery_ids, **labels}
    path = output / f"g1_g2.tgcir.{identity['category']}.native.score-ranks.jsonl"
    temporary = path.with_name(path.name + ".tmp")
    digest = hashlib.sha256()
    with temporary.open("wb") as stream:
        def write_line(record: dict[str, Any]) -> None:
            encoded = (json.dumps(record, ensure_ascii=False, separators=(",", ":"),
                                  allow_nan=False) + "\n").encode("utf-8")
            stream.write(encoded)
            digest.update(encoded)

        write_line(header)
        for index, query_id in enumerate(query_ids):
            write_line({"query_id": query_id,
                        "target_id": gallery_ids[target_indices[index]],
                        "target_position": target_indices[index],
                        "excluded_gallery_ids": [gallery_ids[reference_indices[index]]],
                        "native_scores": native_scores[index].tolist(),
                        "adapter_scores": adapter_scores[index].tolist(),
                        "native_order": native_orders[index].tolist(),
                        "adapter_order": adapter_orders[index].tolist()})
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return path, digest.hexdigest(), _sha(canonical)


def score_proof_identity(category: str, checkpoint_sha256: str, configuration: dict[str, Any],
                         environment: str, query_ids: list[str], gallery_ids: list[str],
                         target_ids: list[str], reference_ids: list[str]) -> dict[str, Any]:
    labels = {"query_targets": [[qid, target] for qid, target in zip(query_ids, target_ids)],
              "reference_exclusions": [[qid, [reference]]
                                       for qid, reference in zip(query_ids, reference_ids)]}
    digest = lambda value: _sha(json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                                           default=str).encode("utf-8"))
    return {"checkpoint": checkpoint_sha256, "configuration": digest(configuration),
            "environment": environment, "protocol": "workbench/third_party/TGCIR/test.py:test;fashioniq_val_split",
            "category": category, "cohort_id": f"fashioniq-val-{category}",
            "precision": "float32", "seed": "unset-by-pinned-evaluator",
            "query_order_sha256": digest(query_ids), "gallery_order_sha256": digest(gallery_ids),
            "reference_exclusions_sha256": digest(labels),
            "query_count": len(query_ids), "gallery_count": len(gallery_ids)}



def _atomic_npz(path: Path, **arrays: Any) -> None:
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez(temporary, **arrays)
    os.replace(temporary, path)


@contextmanager
def bounded_source_inputs(dataset_root: Path, cache_root: Path, datasets: Any,
                         selected_ids: set[str], artifact_root: Path) -> Iterator[dict[str, Any]]:
    """Spool native transformed images and persist selected pixels outside disposable cache."""
    artifact_root.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {"capture_limit": len(selected_ids), "selected": {}, "getter_calls": 0,
                              "_pixel_arrays": {}}
    with disk_backed_images(dataset_root, cache_root, datasets,
                            method_name="get_test_img", split="val") as spool:
        import torch
        original_stack = torch.stack
        cls = datasets.FashionIQ
        original = cls.get_test_img

        def stack_and_evict(*args: Any, **kwargs: Any) -> Any:
            batch = original_stack(*args, **kwargs)
            spool.evict()
            return batch
        def observe(instance: Any, image_name: str, *args: Any, **kwargs: Any) -> Any:
            tensor = original(instance, image_name, *args, **kwargs)
            receipt["getter_calls"] += 1
            if image_name in selected_ids and image_name not in receipt["selected"]:
                if len(receipt["selected"]) >= receipt["capture_limit"]:
                    raise RuntimeError("TG-CIR G1 input capture exceeds bounded endpoint limit")
                pixels = tensor.detach().cpu().contiguous().numpy()
                receipt["selected"][image_name] = {
                    "shape": list(pixels.shape), "dtype": str(pixels.dtype),
                    "sha256": _sha(pixels.tobytes()),
                }
                receipt["_pixel_arrays"][image_name.replace("/", "_")] = pixels.copy()
            return tensor

        torch.stack = stack_and_evict
        cls.get_test_img = observe
        try:
            yield receipt
        finally:
            cls.get_test_img = original
            torch.stack = original_stack
            arrays = receipt.pop("_pixel_arrays")
            if arrays:
                artifact = artifact_root / "tgcir-native-g1-inputs.npz"
                _atomic_npz(artifact, **arrays)
                receipt["pixel_artifact"] = str(artifact)
    receipt["spool"] = dict(spool)


@contextmanager
def native_feature_capture(model: Any, category: str) -> Iterator[dict[str, Any]]:
    """Capture exact native forward outputs, G1 text tokens, and source ranks."""
    from workbench.third_party.TGCIR import model as native_module
    receipt: dict[str, Any] = {"category": category, "queries": [], "targets": [],
                               "g1_text": [], "scores": [], "ranks": []}
    names = ("extract_retrieval_compose", "extract_retrieval_target")
    originals = {name: getattr(model, name) for name in names}
    original_tokenize = native_module.clip.tokenize
    def capture_tokenize(text: Any, *args: Any, **kwargs: Any) -> Any:
        tokens = original_tokenize(text, *args, **kwargs)
        remaining = G1_LIMIT - sum(len(item["texts"]) for item in receipt["g1_text"])
        if remaining > 0:
            texts = [text] if isinstance(text, str) else list(text)
            selected_texts = texts[:remaining]
            selected_tokens = tokens[:remaining].detach().cpu().numpy().copy()
            receipt["g1_text"].append({"texts": selected_texts, "tokens": selected_tokens,
                                       "sha256": _sha(selected_tokens.tobytes())})
        return tokens

    native_module.clip.tokenize = capture_tokenize
    original_argsort = np.argsort

    def captured_argsort(values: Any, *args: Any, **kwargs: Any) -> Any:
        order = original_argsort(values, *args, **kwargs)
        array = np.asarray(values)
        if array.ndim == 1 and array.shape[0] > 50:
            receipt["scores"].append(-array.copy())
            receipt["ranks"].append(np.asarray(order).copy())
        return order

    def call_query(*args: Any, **kwargs: Any) -> Any:
        output = originals["extract_retrieval_compose"](*args, **kwargs)
        receipt["queries"].append(output.detach().float().cpu().numpy().copy())
        return output

    def call_target(*args: Any, **kwargs: Any) -> Any:
        output = originals["extract_retrieval_target"](*args, **kwargs)
        receipt["targets"].append(output.detach().float().cpu().numpy().copy())
        return output

    setattr(model, names[0], call_query)
    setattr(model, names[1], call_target)
    np.argsort = captured_argsort
    try:
        yield receipt
    finally:
        np.argsort = original_argsort
        native_module.clip.tokenize = original_tokenize
        for name, original in originals.items():
            setattr(model, name, original)
        for key, count_key in (("queries", "query_forward_batches"), ("targets", "target_forward_batches")):
            chunks = receipt[key]
            receipt[count_key] = len(chunks)
            receipt[key] = np.concatenate(chunks, axis=0) if chunks else np.empty((0, 512), np.float32)

def _native_stage(dataset_root: Path, checkpoint_root: Path, source_root: Path,
                  output: Path, batch_size: int, category: str) -> None:
    sample = g1_sample(dataset_root, category, limit=G1_LIMIT)
    population = g1_sample(dataset_root, category, limit=None)
    captures: dict[str, Any] = {}

    @contextmanager
    def image_factory(root: Path, cache_root: Path, datasets: Any):
        with bounded_source_inputs(root, cache_root, datasets,
                                   set(sample["gallery_image_names"]), output) as receipt:
            yield receipt

    @contextmanager
    def model_factory(model: Any, _categories: tuple[str, ...]):
        with native_feature_capture(model, category) as captured:
            captures["native"] = captured
            yield captured

    rows, macro = evaluate_categories(
        checkpoint_root, dataset_root, source_root, categories=(category,),
        batch_size=batch_size, device="cuda", image_context_factory=image_factory,
        model_context_factory=model_factory)
    native = captures["native"]
    queries, targets = native["queries"], native["targets"]
    from workbench.replay.tensor_spool import expected_image_count
    expected_calls = expected_image_count(dataset_root, "val")
    spool = rows[0].provenance.get("image_spool", {})
    if spool.get("getter_calls") != expected_calls:
        raise RuntimeError("TG-CIR native source getter count differs from all-category FashionIQ constructor")
    if spool.get("spool", {}).get("stored_image_tensors") != expected_calls:
        raise RuntimeError("TG-CIR tensor spool did not observe every cold source transform")
    if queries.shape[0] != len(population["rows"]) or targets.shape[0] != len(population["gallery_paths"]):
        raise RuntimeError("TG-CIR native feature capture count differs from source population")
    arrays_path = output / "native-arrays.npz"
    _atomic_npz(arrays_path, queries=queries, targets=targets,
                scores=np.stack(native["scores"]), ranks=np.stack(native["ranks"]),
                g1_token_ids=np.concatenate([item["tokens"] for item in native["g1_text"]]))
    provenance = dict(rows[0].provenance)
    provenance.pop("native_model_observation", None)
    native_meta = {"metrics": {"r1": rows[0].r1, "r10": rows[0].r10, "r50": rows[0].r50},
                   "macro": macro, "provenance": provenance,
                   "native_observation": {"query_forward_batches": native["query_forward_batches"],
                                          "target_forward_batches": native["target_forward_batches"],
                                          "query_features_shape": list(queries.shape),
                                          "target_features_shape": list(targets.shape),
                                          "score_rows": len(native["scores"]),
                                          "rank_rows": len(native["ranks"]),
                                          "g1_text_count": len(native["g1_text"])},
                   "source_receipt": spool,
                   "g1_text": [text for item in native["g1_text"] for text in item["texts"]],
                   "query_ids": population["query_image_names"],
                   "gallery_ids": population["gallery_image_names"],
                   "reference_indices": population["reference_indices"],
                   "target_indices": population["target_indices"],
                   "native_runtime": {"python": sys.version, "torch": __import__("torch").__version__,
                                     "cuda_runtime": __import__("torch").version.cuda},
                   "native_pid": os.getpid(),
                   "artifact_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest()}
    temporary = output / "native-stage.json.tmp"
    temporary.write_text(json.dumps(native_meta, indent=2) + "\n")
    os.replace(temporary, output / "native-stage.json")


def _run_native_stage(dataset_root: Path, checkpoint_root: Path, source_root: Path,
                      output: Path, batch_size: int, category: str) -> dict[str, Any]:
    config = resolve_config()
    from workbench.backend.registry import model_by_id
    from workbench.backend.runtime import require_verified_model_interpreter
    interpreter = require_verified_model_interpreter(model_by_id("tgcir"), config=config)
    command = [interpreter, "-m", "workbench.replay.tgcir_pair", "--native-stage",
               "--dataset-root", str(dataset_root), "--checkpoint-root", str(checkpoint_root),
               "--source-root", str(source_root), "--output", str(output),
               "--native-batch-size", str(batch_size), "--category", category]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[2], check=True)
    metadata = json.loads((output / "native-stage.json").read_text())
    arrays_path = output / "native-arrays.npz"
    if not arrays_path.is_file():
        raise RuntimeError("TG-CIR native stage returned without complete atomic arrays")
    digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    if metadata.get("artifact_sha256") != digest:
        raise RuntimeError("TG-CIR native stage artifact digest mismatch")
    if metadata.get("native_pid") == os.getpid():
        raise RuntimeError("TG-CIR native stage unexpectedly shared Adapter orchestrator process")
    population = g1_sample(dataset_root, category, None)
    if metadata.get("query_ids") != population["query_image_names"]:
        raise RuntimeError("TG-CIR native stage query order differs from source annotations")
    if metadata.get("gallery_ids") != population["gallery_image_names"]:
        raise RuntimeError("TG-CIR native stage gallery order differs from source annotations")
    if metadata.get("reference_indices") != population["reference_indices"]:
        raise RuntimeError("TG-CIR native stage candidate positions differ from source annotations")
    if metadata.get("target_indices") != population["target_indices"]:
        raise RuntimeError("TG-CIR native stage target positions differ from source annotations")
    with np.load(arrays_path, allow_pickle=False) as archive:
        expected = {"queries", "targets", "scores", "ranks", "g1_token_ids"}
        if set(archive.files) != expected:
            raise RuntimeError("TG-CIR native stage artifact is incomplete")
        query_count = len(population["rows"])
        gallery_count = len(population["gallery_paths"])
        if archive["queries"].shape != (query_count, 512) or archive["targets"].shape != (gallery_count, 512):
            raise RuntimeError("TG-CIR native stage feature dimensions mismatch")
        if archive["scores"].shape != (query_count, gallery_count) or archive["ranks"].shape != (query_count, gallery_count):
            raise RuntimeError("TG-CIR native stage score/rank dimensions mismatch")
        if archive["g1_token_ids"].shape[0] != G1_LIMIT:
            raise RuntimeError("TG-CIR native stage G1 token capture dimensions mismatch")
    return metadata

def g1_sample(dataset_root: Path, category: str, limit: int | None = G1_LIMIT) -> dict[str, Any]:
    """Select annotation-ordered bounded queries, their pair-union gallery, and paths."""
    rows = json.loads((dataset_root / "captions" / f"cap.{category}.val.json").read_text())
    split = json.loads((dataset_root / "image_splits" / f"split.{category}.val.json").read_text())
    if limit is not None and not 1 <= limit <= G1_LIMIT:
        raise ValueError(f"TG-CIR bounded sample limit must be in [1,{G1_LIMIT}]")
    selected = rows if limit is None else rows[:limit]
    queries: list[str] = []
    gallery: list[str] = []
    for row in selected:
        query, target = f"{category}_{row['candidate']}", f"{category}_{row['target']}"
        queries.append(query)
        if query not in gallery:
            gallery.append(query)
        if target not in gallery:
            gallery.append(target)
    def path(name: str) -> str:
        return str(dataset_root / "resized_image" / category / f"{name.split('_', 1)[1]}.jpg")
    gallery_index = {name: i for i, name in enumerate(gallery)}
    return {"category": category, "rows": selected, "query_image_names": queries,
            "gallery_image_names": gallery, "query_paths": [path(name) for name in queries],
            "gallery_paths": [path(name) for name in gallery],
            "query_texts": [" and ".join(row["captions"]) for row in selected],
            "reference_indices": [gallery_index[f"{category}_{row['candidate']}"] for row in selected],
            "target_indices": [gallery_index[f"{category}_{row['target']}"] for row in selected],
            "split_count": len(split)}


def run_pair(dataset_root: Path, checkpoint_root: Path, source_root: Path,
             output: Path, *, category: str, batch_size: int = 16,
             adapter_batch_size: int | None = None, device: str = "cuda") -> dict[str, Any]:
    """Run full native category plus full Adapter ranking, retaining bounded G1 inputs."""
    from workbench.replay.tgcir import CHECKPOINT

    sample = g1_sample(dataset_root, category, limit=G1_LIMIT)
    population = g1_sample(dataset_root, category, limit=None)
    config = resolve_config()
    output.mkdir(parents=True, exist_ok=False)
    adapter_root = output / "adapter-work"
    capture_root = output / "adapter-g1"
    capture_root.mkdir()

    native_stage = _run_native_stage(dataset_root, checkpoint_root, source_root, output, batch_size, category)
    with np.load(output / "native-arrays.npz", allow_pickle=False) as archive:
        native_q = archive["queries"].copy()
        native_g = archive["targets"].copy()
        native_scores = archive["scores"].copy()
        native_rank = archive["ranks"].copy()
        native_g1_tokens = archive["g1_token_ids"].copy()
    native_summary = native_stage["metrics"]
    native_macro = native_stage["macro"]
    source_receipt = native_stage["source_receipt"]
    native_g1_text = native_stage["g1_text"]
    if native_q.shape[0] != len(population["rows"]) or native_g.shape[0] != len(population["gallery_paths"]):
        raise RuntimeError("TG-CIR native staged features differ from source population")
    effective_adapter_batch_size = batch_size if adapter_batch_size is None else adapter_batch_size
    adapter = Adapter(config)
    adapter.work_root = adapter_root
    adapter.batch_size = effective_adapter_batch_size
    try:
        identity = adapter.prepare_identity()
        adapter_q = adapter.encode_queries(population["query_paths"], population["query_texts"],
                                           [None] * len(population["query_paths"]), capture_path=None)
        adapter_g = adapter.encode_targets(population["gallery_paths"],
                                           [None] * len(population["gallery_paths"]), capture_path=None)
        g1_q = adapter.encode_queries(sample["query_paths"], sample["query_texts"],
                                      [None] * len(sample["query_paths"]),
                                      capture_path=capture_root / "queries.npz")
        g1_g = adapter.encode_targets(sample["gallery_paths"], [None] * len(sample["gallery_paths"]),
                                      capture_path=capture_root / "targets.npz")
        if not np.isfinite(adapter_q).all() or not np.isfinite(adapter_g).all():
            raise RuntimeError("TG-CIR Adapter produced non-finite features")
        score = adapter.score(adapter_q, adapter_g)
    finally:
        adapter.close()
    with np.load(output / "tgcir-native-g1-inputs.npz") as native_inputs, \
            np.load(capture_root / "queries.npz") as adapter_queries, \
            np.load(capture_root / "targets.npz") as adapter_targets:
        native_query = np.stack([native_inputs[name.replace("/", "_")]
                                 for name in sample["query_image_names"]])
        native_gallery = np.stack([native_inputs[name.replace("/", "_")]
                                   for name in sample["gallery_image_names"]])
        if not np.array_equal(native_query, adapter_queries["pixels"]):
            raise RuntimeError("TG-CIR native and Adapter G1 query pixels differ")
        if not np.array_equal(native_gallery, adapter_targets["pixels"]):
            raise RuntimeError("TG-CIR native and Adapter G1 gallery pixels differ")
    native_score = native_scores
    references = population["reference_indices"]
    if native_score.shape != score.shape:
        raise RuntimeError(f"TG-CIR native/Adapter score shape mismatch: {native_score.shape} != {score.shape}")
    targets_idx = population["target_indices"]
    score[np.arange(len(references)), references] = -10e10
    adapter_rank = np.argsort(-score, axis=1)
    topk = {}
    for k in (1, 10, 50):
        native_set = native_rank[:, :k]
        adapter_set = adapter_rank[:, :k]
        overlap = np.sum(np.any(native_set[:, :, None] == adapter_set[:, None, :], axis=2), axis=1)
        topk[str(k)] = {"queries_with_any_flip": int(np.count_nonzero(overlap != k)),
                        "mean_overlap": float(np.mean(overlap))}
    metrics = {}
    for label, ranks in (("native", native_rank), ("adapter", adapter_rank)):
        positions = np.where(ranks == np.asarray(targets_idx)[:, None])[1] + 1
        metrics[label] = {f"r{k}": float(np.mean(positions <= k) * 100) for k in (1, 10, 50)}
    native_summary = native_stage["metrics"]
    if any(value is None or not np.isclose(metrics["native"][f"r{k}"], value)
           for k, value in ((k, native_summary[f"r{k}"]) for k in (1, 10, 50))):
        raise RuntimeError("TG-CIR captured native rankings do not reproduce pinned evaluator metrics")
    sample_refs = sample["reference_indices"]
    sample_targets = sample["target_indices"]
    sample_score = adapter.score(g1_q, g1_g)
    sample_score[np.arange(len(sample_refs)), sample_refs] = -10e10
    sample_ranks = np.argsort(-sample_score, axis=1)
    sample_positions = np.where(sample_ranks == np.asarray(sample_targets)[:, None])[1] + 1
    bounded_metrics = {f"r{k}": float(np.mean(sample_positions <= k) * 100) for k in (1, 10, 50)}
    from workbench.backend.registry import checkpoint_path as resolve_checkpoint_path
    from workbench.replay.common import sha256_file
    checkpoint_path = resolve_checkpoint_path("tgcir", CHECKPOINT, checkpoint_root)
    checkpoint_sha = sha256_file(checkpoint_path)
    references = population["reference_indices"]
    query_ids = scoring_query_ids(population["query_image_names"])
    gallery_ids = population["gallery_image_names"]
    targets_idx = population["target_indices"]
    target_ids = [gallery_ids[index] for index in targets_idx]
    reference_ids = [gallery_ids[index] for index in references]
    source_files = [source_root / "datasets.py", source_root / "test.py", source_root / "model.py"]
    code_files = source_files + [Path(__file__).resolve(),
                                 Path(__file__).resolve().parents[1] / "composition" / "model_tgcir.py"]
    source_receipts = {path.name: sha256_file(path) for path in code_files}
    proof_config = asdict(config)
    proof_config.update(native_batch_size=batch_size,
                        adapter_batch_size=effective_adapter_batch_size,
                        device=device, transform="OpenAI CLIP ViT-B/16 preprocess",
                        code_sha256=source_receipts)
    score_identity = score_proof_identity(
        category, checkpoint_sha, proof_config,
        f"native={native_stage['native_runtime']};adapter={adapter.interpreter}",
        query_ids, gallery_ids, target_ids, reference_ids)
    proof_path, proof_sha, labels_sha = write_score_rank_proof(
        output, score_identity, query_ids, gallery_ids, targets_idx, references,
        native_score, score, native_rank, adapter_rank)
    paired_arrays_path = output / "paired-arrays.npz"
    _atomic_npz(paired_arrays_path, native_queries=native_q, native_gallery=native_g,
                adapter_queries=adapter_q, adapter_gallery=adapter_g,
                native_scores=native_score, adapter_scores=score,
                native_ranks=native_rank, adapter_ranks=adapter_rank,
                reference_indices=np.asarray(references, dtype=np.int64),
                target_indices=np.asarray(targets_idx, dtype=np.int64),
                query_ids=np.asarray(query_ids),
                source_query_ids=np.asarray(population["query_image_names"]),
                gallery_ids=np.asarray(gallery_ids))
    plan = {"status": "diagnostic_only", "scope": "adaptedlocalauthorarchive", "category": category,
            "native_batch_size": batch_size, "adapter_batch_size": effective_adapter_batch_size,
            "native_query_count": len(native_q), "native_gallery_count": len(native_g),
            "adapter_query_count": len(adapter_q), "adapter_gallery_count": len(adapter_g),
            "scoring_proof_path": proof_path.name,
            "scoring_identity": score_identity,
            "scoring_proof_sha256": proof_sha,
            "scoring_labels": {"query_targets": [[qid, target] for qid, target in zip(query_ids, target_ids)],
                               "reference_exclusions": [[qid, [reference]]
                                                        for qid, reference in zip(query_ids, reference_ids)]},
            "scoring_labels_sha256": labels_sha,
            "scoring_role_status": "diagnostic-only; not preregistered V3 acceptance evidence",
            "bounded_metrics": bounded_metrics,
            "bounded_input_artifacts": {"native": source_receipt.get("pixel_artifact"),
                                        "adapter_queries": str(capture_root / "queries.npz"),
                                        "adapter_targets": str(capture_root / "targets.npz")},
            "g1_raw_text": sample["query_texts"],
            "paired_arrays_sha256": _sha(paired_arrays_path.read_bytes()),
            "bounded_input_sha256": {"native": _sha(Path(source_receipt["pixel_artifact"]).read_bytes()),
                                     "adapter_queries": _sha((capture_root / "queries.npz").read_bytes()),
                                     "adapter_targets": _sha((capture_root / "targets.npz").read_bytes())},
            "g1_native_text": native_g1_text,
            "g1_query_paths": sample["query_paths"], "g1_gallery_paths": sample["gallery_paths"],
            "query_ids": query_ids, "source_query_ids": population["query_image_names"],
            "gallery_ids": gallery_ids,
            "reference_indices": references, "target_indices": targets_idx,
            "native_metrics": native_summary, "native_macro": native_macro,
            "native_stage_pid": native_stage["native_pid"],
            "native_stage_artifact_sha256": native_stage["artifact_sha256"],
            "native_stage_process_exited_before_adapter": True,
            "full_adapter_metrics": metrics["adapter"], "adapter_identity": identity,
            "rank_topk": topk,
            "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256_file(checkpoint_path),
            "native_source_sha256": source_receipts,
            "native_source_receipt": source_receipt,
            "native_provenance": native_stage["provenance"],
            "native_runtime": "verified tgcir39-la subprocess; torch 2.7.1+cu128; pinned OpenAI CLIP",
            "adapter_runtime": str(adapter.interpreter),
            "adapter_load_state": adapter.last_reply,
            "artifact": str(output / "paired-arrays.npz"),
            "caveat": "Adapted local author archive run; not paper reproduction."}
    manifest_path = output / "plan-and-results.json"
    manifest_tmp = manifest_path.with_name(manifest_path.name + ".tmp")
    manifest_tmp.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)
    return plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("dress", "shirt", "toptee"), default="dress")
    parser.add_argument("--native-batch-size", type=int, default=16)
    parser.add_argument("--adapter-batch-size", type=int, default=None)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    parser.add_argument("--native-stage", action="store_true")
    args = parser.parse_args()
    if args.native_stage:
        _native_stage(args.dataset_root, args.checkpoint_root, args.source_root,
                      args.output, args.native_batch_size, args.category)
        return 0
    plan = run_pair(args.dataset_root, args.checkpoint_root, args.source_root, args.output,
                    category=args.category, batch_size=args.native_batch_size,
                    adapter_batch_size=args.adapter_batch_size, device=args.device)
    print(json.dumps(plan, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
