"""Native-source vs production DCNet adapter scoring on original FashionIQ val.

This is a diagnostic parity producer: source FashionIQ query/gallery order, author-preprocessed
captions, reference mask, and native ranking metrics remain authoritative. The adapter only supplies
its real encoded features and fused scores for the exact same inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "workbench" / "third_party" / "DCNet"


def _preflight_adapter(config: Any) -> tuple[Any, Any]:
    """Resolve workbench dependencies and validate adapter identity before native GPU work."""
    from workbench.backend.operator_config import resolve_config
    from workbench.composition.model_dcnet import Adapter

    adapter = Adapter(resolve_config(), checkpoint_id="fashioniq_run_directory")
    adapter.prepare_identity()
    adapter.close()
    return Adapter, resolve_config


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_native_archive(path: Path, category: str, rows: dict[str, Any]) -> None:
    metadata = [{"category": category, "row": index, "candidate": row["candidate"],
                 "target": row["target"], "reference": row["candidate"]}
                for index, row in enumerate(rows["query_rows"])]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, composition=rows["composition"], correction=rows["correction"],
                            query_metadata_json=np.asarray(json.dumps(metadata)),
                            gallery_ids=np.asarray(rows["gallery_ids"]))
    os.replace(temporary, path)


def _source_trace(captured: dict[str, list[dict[str, Any]]], source_file: str | None = None,
                  capture_line: int = 241):
    source_file = source_file or str((SOURCE / "trainer" / "trainer_joint.py").resolve())

    def trace(frame: Any, event: str, arg: Any):
        if event == "call":
            if (frame.f_code.co_filename == source_file
                    and frame.f_code.co_name == "_valid_epoch"):
                return observe
            return None
        if (event == "line" and frame.f_code.co_name == "_valid_epoch"
                and frame.f_lineno == capture_line):
            local = frame.f_locals
            category = local["category"]
            captured.setdefault(category, []).append({
                "query_row": int(local["j"]),
                "scores": local["score_comb"].detach().cpu().numpy().copy(),
                "top50": local["topk"].detach().cpu().numpy().copy(),
            })
        return observe

    def observe(frame: Any, event: str, arg: Any):
        return trace(frame, event, arg)

    return trace
def _derived_order(scores: np.ndarray) -> np.ndarray:
    """Stable descending permutation derived from captured native score vectors."""
    return np.argsort(-scores, axis=1, kind="stable")



def _source_run(checkpoint: Path, output: Path) -> tuple[Any, dict[str, Any]]:
    os.chdir(SOURCE)
    sys.path.insert(0, str(SOURCE))
    sys.argv = ["test.py", "--resume", str(checkpoint), "--mode", "val"]
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--seeds", default="0", type=str)
    parser.add_argument("--mode", default="val", type=str)
    from parse_config import ConfigParser
    config = ConfigParser(parser, "eval")
    import random
    import torch
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    import data_loader.data_loaders as module_data
    import model.model as module_arch
    from trainer import TrainerJoint
    from utils.util import compute_dims

    dims = compute_dims(config)
    data_loaders = config.init(name="data_loader", module=module_data, expert_dims=dims,
                               text_feat=config["experts"]["text_feat"],
                               text_dim=config["experts"]["text_dim"])
    model = config.init(name="arch", module=module_arch, expert_dims=dims,
                        text_dim=config["experts"]["text_dim"],
                        same_dim=config["experts"]["ce_shared_dim"],
                        text_feat=config["experts"]["text_feat"])
    trainer = TrainerJoint(model, loss=None, optimizer=None, config=config,
                           data_loaders=data_loaders, lr_scheduler=None)
    final_capture: dict[str, list[dict[str, Any]]] = {}
    previous_trace = sys.gettrace()
    sys.settrace(_source_trace(final_capture))
    try:
        metrics = trainer._valid_epoch(save_textatt=False)
    finally:
        sys.settrace(previous_trace)
    native = {}
    for category in config["data_loader"]["args"]["categories"]:
        score = metrics["score"][category]
        final_rows = final_capture.get(category, [])
        query_rows = list(data_loaders[category].dataset.data)
        if [row["query_row"] for row in final_rows] != list(range(len(query_rows))):
            raise RuntimeError(f"{category}: native final source scores/ranks missing or out of query order")
        native[category] = {
            "query_rows": query_rows,
            "gallery_ids": list(score["ids"]),
            "composition": score["matrix_comp"].numpy().copy(),
            "correction": score["matrix_corr"].numpy().copy(),
            "native_final_scores": np.stack([row["scores"] for row in final_rows]),
            "native_final_top50": np.stack([row["top50"] for row in final_rows]),
        }


    output.parent.mkdir(parents=True, exist_ok=True)
    for category, rows in native.items():
        _write_native_archive(output.parent / f"native-{category}-scores.npz", category, rows)
        final_path = output.parent / f"native-{category}-final-scores.npz"
        final_tmp = final_path.with_suffix(final_path.suffix + ".tmp")
        native_order = _derived_order(rows["native_final_scores"])
        with final_tmp.open("wb") as stream:
            np.savez_compressed(stream, scores=rows["native_final_scores"],
                                top50=rows["native_final_top50"], derived_order=native_order)
        os.replace(final_tmp, final_path)
    provenance = {
        "status": "native_source_scores_complete",
        "source_root": str(SOURCE),
        "source_commit": "68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb",
        "checkpoint_sha256": _sha256_file(checkpoint / "trained_model.pth"),
        "categories": {category: {
            "query_count": len(rows["query_rows"]),
            "gallery_count": len(rows["gallery_ids"]),
            "query_order_sha256": __import__("hashlib").sha256(json.dumps([
                {"category": category, "row": index, "candidate": row["candidate"],
                 "target": row["target"], "reference": row["candidate"]}
                for index, row in enumerate(rows["query_rows"])],
                separators=(",", ":")).encode()).hexdigest(),
            "reference_ids": [row["candidate"] for row in rows["query_rows"]],
            "gallery_order_sha256": __import__("hashlib").sha256(json.dumps(
                rows["gallery_ids"], separators=(",", ":")).encode()).hexdigest(),
            "native_final_score_file": f"native-{category}-final-scores.npz",
            "native_final_score_sha256": _sha256_file(output.parent / f"native-{category}-final-scores.npz"),
            "native_final_capture": "scores and top50 captured from source _valid_epoch; full order is derived by stable descending argsort; source emits only topk50", 
        } for category, rows in native.items()},
    }
    receipt = output.parent / "native-source-receipt.json"
    receipt_tmp = receipt.with_suffix(".json.tmp")
    receipt_tmp.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(receipt_tmp, receipt)
    del metrics, trainer, model, data_loaders
    torch.cuda.empty_cache()
    return config, native



def _fused_native(composition: Any, correction: Any) -> np.ndarray:
    import torch.nn.functional as F
    fused = F.log_softmax(composition, dim=-1) + F.log_softmax(correction, dim=-1)
    return fused.float().cpu().numpy()


def _topk(scores: np.ndarray, k: int = 50) -> np.ndarray:
    import torch
    values, indices = torch.topk(torch.from_numpy(np.ascontiguousarray(scores)),
                                 k=k, dim=-1, largest=True, sorted=True)
    del values
    return indices.cpu().numpy()


def compare(checkpoint: Path, output: Path, category_filter: str | None = None) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    output = output.resolve()
    import torch
    Adapter, resolve_config = _preflight_adapter(None)
    config, native_rows = _source_run(checkpoint, output)
    adapter = Adapter(resolve_config(), checkpoint_id="fashioniq_run_directory")
    adapter.prepare_identity()
    adapter.work_root = output.parent / "adapter-work"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    categories = config["data_loader"]["args"]["categories"]
    if category_filter:
        if category_filter not in categories:
            raise ValueError(f"unknown category {category_filter!r}")
        categories = [category_filter]
    result: dict[str, Any] = {
        "protocol_id": "fashioniq_full_gallery_ref_excluded",
        "scope": "diagnostic_native_vs_production_adapter_exact_inputs",
        "native_only_benchmark": False,
        "checkpoint_id": "fashioniq_run_directory",
        "categories": {},
        "adapter_worker_identity": None,
    }
    try:
        for category in categories:
            rows = native_rows[category]["query_rows"]
            gallery_ids = native_rows[category]["gallery_ids"]
            query_ids = [row["candidate"] for row in rows]
            query_texts = [" and ".join(row["captions"]) for row in rows]
            query_paths = [str(Path(config["data_loader"]["args"]["data_dir"]) /
                               "resized_images" / f"{image_id}.jpg") for image_id in query_ids]
            gallery_paths = [str(Path(config["data_loader"]["args"]["data_dir"]) /
                                 "resized_images" / f"{image_id}.jpg") for image_id in gallery_ids]
            gallery_index = {image_id: i for i, image_id in enumerate(gallery_ids)}
            reference_indices = [gallery_index[image_id] for image_id in query_ids]
            adapter.batch_size = 8
            queries = adapter.encode_queries(query_paths, query_texts, [None] * len(query_paths))
            adapter.batch_size = 100
            targets = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
            adapter.batch_size = 8
            adapted = adapter.score(queries, targets, reference_indices=reference_indices)
            native = native_rows[category]["native_final_scores"]
            native_top = native_rows[category]["native_final_top50"]
            if native.shape != adapted.shape:
                raise AssertionError(f"{category}: native {native.shape} != adapter {adapted.shape}")
            adapted_top = _topk(adapted)
            expected = np.asarray([gallery_index[row["target"]] for row in rows], dtype=np.int64)
            native_hits = [int(np.count_nonzero(np.any(native_top[:, :k] == expected[:, None], axis=1)))
                           for k in (10, 50)]
            adapted_hits = [int(np.count_nonzero(np.any(adapted_top[:, :k] == expected[:, None], axis=1)))
                            for k in (10, 50)]
            score_path = output.with_name(f"{output.stem}-{category}-scores.npz")
            native_order = _derived_order(native)
            adapter_order = _derived_order(adapted)
            np.savez(score_path, native_composition=native_rows[category]["composition"],
                     native_correction=native_rows[category]["correction"], native_fused=native,
                     adapter_fused=adapted, native_top50=native_top, adapter_top50=adapted_top,
                     native_derived_order=native_order, adapter_derived_order=adapter_order,
                     query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
                     reference_ids=np.asarray(query_ids), target_ids=np.asarray([row["target"] for row in rows]))
            result["categories"][category] = {
                "adapter_query_batch_size": 8,
                "adapter_gallery_batch_size": 100,
                "queries": len(rows), "gallery": len(gallery_ids),
                "native_hits_at_10_50": native_hits, "adapter_hits_at_10_50": adapted_hits,
                "denominator": len(rows),
                "native_recall_at_10_50": [hits / len(rows) for hits in native_hits],
                "adapter_recall_at_10_50": [hits / len(rows) for hits in adapted_hits],
                "recall_equal": native_hits == adapted_hits,
                "top50_exact_query_count": int(np.count_nonzero(np.all(native_top == adapted_top, axis=1))),
                "top10_exact_query_count": int(np.count_nonzero(np.all(native_top[:, :10] == adapted_top[:, :10], axis=1))),
                "score_artifact": str(score_path),
                "max_abs_fused_score_error": float(np.max(np.abs(native - adapted))),
                "mean_abs_fused_score_error": float(np.mean(np.abs(native - adapted))),
                "gallery_order_sha256": __import__("hashlib").sha256(
                    json.dumps(gallery_ids, separators=(",", ":")).encode()).hexdigest(),
                "query_order_sha256": __import__("hashlib").sha256(json.dumps([
                    {"category": category, "row": index, "candidate": row["candidate"],
                     "target": row["target"], "reference": row["candidate"]}
                    for index, row in enumerate(rows)], separators=(",", ":")).encode()).hexdigest(),
                "native_scores": "source trainer _valid_epoch component scores; reference mask before component softmax",
                "adapter_scores": "production Adapter.score with reference_indices; same source inputs and order",
            }
            result["adapter_worker_identity"] = adapter.worker_identity
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    finally:
        adapter.close()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", choices=("dress", "shirt", "toptee"))
    args = parser.parse_args()
    result = compare(args.checkpoint, args.output, args.category)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
