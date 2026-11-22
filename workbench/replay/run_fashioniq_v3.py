"""Execute one preregistered FashionIQ V3 role or run CPU preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_plan(path: Path) -> dict[str, Any]:
    try:
        plan = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read plan {path}: {error}") from error
    if not isinstance(plan, dict) or plan.get("standard") != "parity-absolute-relative/v3":
        raise ValueError("plan must declare parity-absolute-relative/v3")
    for key in ("identity", "population_scope", "supported_batch_policy", "runs", "controls"):
        if not plan.get(key):
            raise ValueError(f"plan missing {key}")
    return plan


def digest(value: object) -> str:
    raw = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()

def _v3_capture(plan: dict[str, Any], role: str, side: str, started_at: str,
                plan_path: Path, plan_sha256: str,
                arrays: dict[str, tuple[np.ndarray, list[str]]]) -> dict[str, Any]:
    helper = ROOT / "workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("cir_t4_parity_v3", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V3 receipt helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = plan["runs"][role]
    query_ids = arrays["query_features"][1]
    gallery_ids = arrays["target_features"][1]
    expected = run["identity"]
    identity = dict(expected)
    identity.update({"query_order_sha256": digest(query_ids), "gallery_order_sha256": digest(gallery_ids),
                     "query_count": len(query_ids), "gallery_count": len(gallery_ids)})
    for key in ("query_order_sha256", "gallery_order_sha256", "query_count", "gallery_count"):
        if identity[key] != expected[key]:
            raise ValueError(f"observed V3 identity differs from preregistered {key}")
    execution = dict(run["execution"])
    execution["query_partition_sha256"] = identity["query_order_sha256"]
    execution["gallery_partition_sha256"] = identity["gallery_order_sha256"]
    helper_receipt = module.v3_run_receipt(
        run_id=run["run_id"], side=side, source_id=run["source_id"],
        plan_sha256=plan_sha256, started_at=started_at, arrays=arrays,
        execution=execution, identity=identity)
    if role in ("native_adapter_native", "native_adapter_adapter"): 
        scores, score_ids = arrays["scores"]
        orders, order_ids = arrays["ranks"]
        if score_ids != query_ids or order_ids != query_ids or scores.shape != orders.shape:
            raise ValueError("full-role scorer rows differ from ordered query population")
        hasher = hashlib.sha256()
        for index, query_id in enumerate(query_ids):
            row = json.dumps({"query_id": query_id, "scores": scores[index].tolist(),
                              "order": orders[index].tolist()}, ensure_ascii=False,
                             separators=(",", ":"))
            hasher.update(row.encode("utf-8"))
            hasher.update(b"\n")
        helper_receipt["score_order_sha256"] = hasher.hexdigest()
    return helper_receipt
def _identity_evidence(*, role: str, query_paths: list[Path], gallery_paths: list[Path],
                       query_ids: list[str], gallery_ids: list[str], labels: np.ndarray,
                       model_texts: list[str]) -> dict[str, Any]:
    if role not in ("native_adapter_native", "native_adapter_adapter"):
        return {}
    references = [path.stem for path in query_paths]
    ordered_gallery = [path.stem for path in gallery_paths]
    if (len(references) != len(query_ids) or len(model_texts) != len(query_ids)
            or ordered_gallery != gallery_ids or labels.shape != (len(query_ids),)):
        raise ValueError("identity evidence IDs/texts differ from executed scorer inputs")
    if np.any(labels < 0) or np.any(labels >= len(ordered_gallery)):
        raise ValueError("identity evidence target label falls outside actual gallery")
    return {"reference_ids": references,
            "target_ids": [ordered_gallery[int(index)] for index in labels],
            "gallery_order_ids": ordered_gallery, "model_texts": list(model_texts)}




def preflight(plan: dict[str, Any], path: Path) -> dict[str, Any]:
    identity = plan["identity"]
    population = plan["population_scope"]
    policy = plan["supported_batch_policy"]
    if policy.get("supported") is not True or policy.get("native_query_batch") != 32 or policy.get("native_gallery_batch") != 32 or policy.get("adapter_batch") != 32 or policy.get("control_batch") != 1 or policy.get("batch_sizes") != [1, 32]:
        raise ValueError("preflight requires matched native/adapter batch32 and batch1 control")
    if population.get("coverage") != "full":
        raise ValueError("population must be full")
    for key in ("query_ids", "reference_ids", "target_ids", "gallery_ids"):
        if not isinstance(population.get(key), list) or not population[key]:
            raise ValueError(f"population missing ordered {key}")
    if len(population["query_ids"]) != identity["query_count"] or len(population["gallery_ids"]) != identity["gallery_count"]:
        raise ValueError("population counts differ from identity")
    if plan["model_id"] == "csmcir" and not population.get("target_caption_source"):
        raise ValueError("CSMCIR plan needs category target captions")
    if plan["model_id"] == "sprc" and population.get("target_caption_source") not in (None, "not_applicable"):
        raise ValueError("SPRC image-only gallery must not depend on CSMCIR captions")
    expected = {"native_repeat_1", "native_repeat_2", "adapter_repeat_1", "adapter_repeat_2",
                "native_adapter_native", "native_adapter_adapter", "native_batch_1", "native_batch_n",
                "adapter_batch_1", "adapter_batch_n"}
    if set(plan["runs"]) != expected:
        raise ValueError("plan must predeclare all ten physical roles")
    return {"plan": str(path), "plan_sha256": digest(path.read_bytes()), "model_id": plan["model_id"],
            "category": plan["category"], "query_count": identity["query_count"],
            "gallery_count": identity["gallery_count"], "roles": sorted(expected),
            "native_metric_protocol": "torch.argsort(1-pred_sim); labels against all native index_names; references remain gallery eligible"}


def _paths(plan: dict[str, Any], role: str) -> tuple[list[Path], list[Path], list[str]]:
    category = plan["category"]
    population = plan["population_scope"]
    control = role not in ("native_adapter_native", "native_adapter_adapter")
    query_ids = population["control_query_ids"] if control else population["query_ids"]
    gallery_ids = population["control_gallery_ids"] if control else population["gallery_ids"]
    image_root = ROOT / "data/FashionIQ/images"
    query_paths = [image_root / f"{item.split(':')[-2]}.png" for item in query_ids] if control else [image_root / f"{item}.png" for item in population["reference_ids"]]
    gallery_paths = [image_root / f"{item}.png" for item in gallery_ids]
    pair_path = ROOT / "data/FashionIQ/captions" / f"cap.{category}.val.json"
    try:
        pairs = json.loads(pair_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load query pairs {pair_path}: {error}") from error
    selected = pairs[:len(query_paths)] if control else pairs
    texts = [f"{row['captions'][0].strip('.?, ').capitalize()} and {row['captions'][1].strip('.?, ')}" for row in selected]
    if len(query_paths) != len(texts) or any(not path.is_file() for path in query_paths + gallery_paths):
        raise ValueError("official FashionIQ image/text population is incomplete")
    return query_paths, gallery_paths, texts

def _population(plan: dict[str, Any], role: str) -> tuple[list[str], list[str]]:
    population = plan["population_scope"]
    if role in ("native_adapter_native", "native_adapter_adapter"):
        return population["query_ids"], population["gallery_ids"]
    return population["control_query_ids"], population["control_gallery_ids"]


def _ranks(scores: np.ndarray, torch: Any) -> np.ndarray:
    return torch.argsort(1 - torch.from_numpy(scores), dim=-1).cpu().numpy()

class _SorterObservation:
    def __init__(self, torch: Any) -> None:
        self.torch = torch
        self.original = torch.argsort
        self.rows = []

    def __enter__(self) -> "_SorterObservation":
        def observe(input: Any, *args: Any, **kwargs: Any) -> Any:
            result = self.original(input, *args, **kwargs)
            if input.ndim == 2 and input.shape[1] > 1:
                self.rows.append((input.detach(), result.detach(), str(input.device), str(result.device)))
            return result
        self.torch.argsort = observe
        return self

    def __exit__(self, _kind: Any, _value: Any, _traceback: Any) -> None:
        self.torch.argsort = self.original

def _author_validation_sort(source: Path, model_id: str, pred_sim: Any, target_ids: list[str],
                            gallery_ids: list[str], torch: Any) -> tuple[Any, str, str]:
    import ast
    from types import SimpleNamespace

    module_path = source / "src" / ("validate_blip_csmcir.py" if model_id == "csmcir" else "validate_blip.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "compute_fiq_val_metrics")
    isolated = ast.Module(body=[function], type_ignores=[])
    namespace: dict[str, Any] = {"torch": torch, "np": np, "Tuple": tuple, "List": list}
    exec(compile(isolated, str(module_path), "exec"), namespace)
    namespace["generate_fiq_val_predictions"] = lambda *_args, **_kwargs: (pred_sim, target_ids, [], [])
    with _SorterObservation(torch) as observation:
        namespace["compute_fiq_val_metrics"](SimpleNamespace(dress_types="FashionIQ"), None, None,
                                               gallery_ids, None)
    if len(observation.rows) != 1:
        raise ValueError(f"expected one author validation sort, observed {len(observation.rows)}")
    sort_input, indices, score_device, sort_device = observation.rows[0]
    if not torch.equal(sort_input, 1 - pred_sim):
        raise ValueError("author validator sorted unexpected score tensor")
    return indices.cpu(), score_device, sort_device

def _source_metrics(scores: np.ndarray, query_ids: list[str], gallery_ids: list[str], torch: Any) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    pred_sim = torch.from_numpy(scores)
    sorted_indices = torch.argsort(1 - pred_sim, dim=-1).cpu().numpy()
    sorted_names = np.asarray(gallery_ids, dtype=str)[sorted_indices]
    targets = np.asarray([query_id.rsplit(":", 1)[-1] for query_id in query_ids], dtype=str)
    labels = sorted_names == targets[:, None]
    if not np.all(labels.sum(axis=-1) == 1):
        raise ValueError("native FashionIQ metric requires one eligible positive per query")
    recalls = {f"recall{limit}": float(labels[:, :limit].sum() / len(labels) * 100) for limit in (10, 50)}
    return sorted_indices, labels, recalls

def _labels(query_ids: list[str], gallery_ids: list[str]) -> np.ndarray:
    labels = []
    for query_id in query_ids:
        target = query_id.rsplit(":", 1)[-1]
        if target not in gallery_ids:
            raise ValueError(f"positive target missing from role gallery: {target}")
        labels.append(gallery_ids.index(target))
    if len(labels) != len(query_ids):
        raise ValueError("role query/target denominator mismatch")
    return np.asarray(labels, dtype=np.int64)


def _transform(paths: list[Path], transform: Any, torch: Any, Image: Any, batch_size: int) -> Any:
    for start in range(0, len(paths), batch_size):
        values = []
        for path in paths[start:start + batch_size]:
            try:
                with Image.open(path) as image:
                    values.append(transform(image.convert("RGB")))
            except (OSError, ValueError) as error:
                raise ValueError(f"cannot preprocess image {path}: {error}") from error
        yield torch.stack(values).cuda()

def _capture_ids(plan: dict[str, Any], scope: str) -> set[str]:
    values = plan.get("input_capture", {}).get(f"{scope}_ids", [])
    if not isinstance(values, list) or len(values) > 5 or len(set(values)) != len(values):
        raise ValueError(f"invalid bounded {scope} input capture selector")
    return set(values)


def _save_input_capture(output: Path, scope: str, ids: list[str], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    capture_path = output.with_name(f"{output.stem}.inputs.{scope}.npz")
    np.savez_compressed(capture_path, ordered_ids=np.asarray(ids), **arrays)
    return {"path": str(capture_path), "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
            "ordered_ids": ids, "arrays": {name: list(value.shape) for name, value in arrays.items()},
            "array_sha256": {name: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
                             for name, value in arrays.items()}}

def _gallery_caption_capture(captions: list[str], keep: list[int]) -> np.ndarray:
    return np.asarray([captions[index] for index in keep], dtype=str)

def _read_adapter_input_capture(path: Path, expected_ids: list[str], *, scope: str) -> tuple[dict[str, np.ndarray], list[str]]:
    with np.load(path, allow_pickle=False) as observed:
        ids = observed["ordered_ids"].astype(str).tolist()
        if ids != expected_ids:
            raise ValueError(f"adapter input capture order mismatch: {path}")
        arrays = {name: observed[name].copy() for name in observed.files if name != "ordered_ids"}
    base = {"transformed_pixels"}
    if scope == "query":
        required = base | {"token_ids", "attention_mask", "raw_text", "processed_text"}
    elif scope == "gallery":
        required = base
        if "raw_text" in arrays or "processed_text" in arrays:
            required |= {"raw_text", "processed_text", "token_ids", "attention_mask"}
    else:
        raise ValueError(f"unknown adapter input capture scope: {scope}")
    if set(arrays) != required:
        raise ValueError(f"adapter input capture fields differ at {path}: {sorted(arrays)}")
    count = len(ids)
    if any(value.ndim == 0 or len(value) != count for value in arrays.values()):
        raise ValueError(f"adapter input capture row count mismatch: {path}")
    return arrays, ids

def _observe_qformer_inputs(args: tuple[Any, ...], kwargs: dict[str, Any], active_ids: list[str],
                            selected_ids: set[str], token_store: list[np.ndarray],
                            mask_store: list[np.ndarray]) -> list[str]:
    selected = [index for index, value in enumerate(active_ids) if value in selected_ids]
    if not selected:
        return []
    input_ids = args[0] if args else kwargs.get("input_ids")
    attention_mask = kwargs.get("attention_mask")
    if input_ids is None or attention_mask is None:
        return []
    token_store.append(input_ids[selected].detach().cpu().numpy())
    mask_store.append(attention_mask[selected].detach().cpu().numpy())
    return [active_ids[index] for index in selected]


def _append_gallery_target(model: Any, model_id: str, chunk: Any, captions: list[str] | None,
                           keep: list[int], raw_text: list[str], processed_text: list[str],
                           target_parts: list[Any]) -> None:
    if model_id == "csmcir":
        if captions is None:
            raise ValueError("CSMCIR gallery target captions are required")
        selected = _gallery_caption_capture(captions, keep).tolist()
        raw_text.extend(selected)
        processed_text.extend(selected)
        target, _ = model.extract_target_caption_features(chunk, captions, mode="mean")
    else:
        target, _ = model.extract_target_features(chunk, mode="mean")
    target_parts.append(target.float().cpu())

def _native(plan: dict[str, Any], output: Path, role: str, started_at: str,
            plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    import torch
    from PIL import Image
    from workbench.replay.lavis_processors import prepare
    model_id, category = plan["model_id"], plan["category"]
    source = ROOT / "workbench/third_party" / ("CSMCIR" if model_id == "csmcir" else "SPRC")
    prepare(source / "src")
    from lavis.models import load_model_and_preprocess
    if model_id == "csmcir":
        from data_utils_csmcir import targetpad_transform
        model, _, processors = load_model_and_preprocess(name="blip2_cir_align_prompt_csmcir", model_type="pretrain", is_eval=False, device="cuda")
    else:
        from data_utils import targetpad_transform
        model, _, processors = load_model_and_preprocess(name="blip2_cir_align_prompt", model_type="pretrain_vitL", is_eval=False, device="cuda")
    model.eval()
    state = torch.load(Path(plan["environment"]["checkpoint_path"]), map_location="cuda", weights_only=False)
    key = "Blip2QformerCirAlignPromptTargetImageCaptionshiyan92" if model_id == "csmcir" else model.__class__.__name__
    if key not in state:
        raise RuntimeError(f"expected native checkpoint key missing: {key}")
    report = model.load_state_dict(state[key], strict=(model_id == "sprc"))
    missing_keys = list(getattr(report, "missing_keys", ()))
    unexpected_keys = list(getattr(report, "unexpected_keys", ()))
    if missing_keys or (model_id == "sprc" and unexpected_keys) or (model_id == "csmcir" and set(unexpected_keys) - {"token_importance"}):
        raise RuntimeError(f"checkpoint key mismatch missing={missing_keys} unexpected={unexpected_keys}")
    query_paths, gallery_paths, texts = _paths(plan, role)
    transform = targetpad_transform(1.25, 224)
    captions = [processors["eval"](text) for text in texts]
    caption_map: dict[str, Any] = {}
    if model_id == "csmcir":
        caption_file = Path(plan["population_scope"]["target_caption_source"])
        try:
            loaded = json.loads(caption_file.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot load target captions {caption_file}: {error}") from error
        if not isinstance(loaded, dict):
            raise ValueError(f"target captions must be object: {caption_file}")
        caption_map = loaded
    query_ids, gallery_ids = _population(plan, role)
    gallery_captions: list[str] | None = None
    if model_id == "csmcir":
        gallery_captions = []
        for item in gallery_ids:
            value = caption_map.get(item, {}).get("Final_Caption")
            if not isinstance(value, str) or not value:
                raise ValueError(f"missing Final_Caption for {item}")
            gallery_captions.append(value)
    batch_size = int(plan["runs"][role]["execution"]["gallery_batch_size"])
    capture_queries, capture_gallery = _capture_ids(plan, "query"), _capture_ids(plan, "gallery")
    gallery_pixels: list[np.ndarray] = []
    query_pixels: list[np.ndarray] = []
    query_tokens: list[np.ndarray] = []
    query_masks: list[np.ndarray] = []
    gallery_tokens: list[np.ndarray] = []
    gallery_masks: list[np.ndarray] = []
    gallery_raw_text: list[str] = []
    gallery_processed_text: list[str] = []
    active_input_ids: list[str] = []
    active_input_scope = "query"
    query_observed_ids: list[str] = []
    gallery_observed_ids: list[str] = []
    active_capture_seen = False
    def observe_qformer(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        nonlocal active_capture_seen
        if active_capture_seen:
            return
        selected_ids = capture_queries if active_input_scope == "query" else capture_gallery
        token_store = query_tokens if active_input_scope == "query" else gallery_tokens
        mask_store = query_masks if active_input_scope == "query" else gallery_masks
        observed = _observe_qformer_inputs(args, kwargs, active_input_ids, selected_ids, token_store, mask_store)
        if observed:
            active_capture_seen = True
            (query_observed_ids if active_input_scope == "query" else gallery_observed_ids).extend(observed)
    observer = model.Qformer.bert.register_forward_pre_hook(observe_qformer, with_kwargs=True)
    gallery_chunks = _transform(gallery_paths, transform, torch, Image, batch_size)
    active_capture_seen = False
    target_parts = []
    with torch.no_grad():
        for chunk, start in zip(gallery_chunks, range(0, len(gallery_paths), batch_size)):
            active_forward = 0
            active_capture_seen = False
            ids = gallery_ids[start:start + len(chunk)]
            active_input_scope = "gallery"
            active_input_ids[:] = ids
            keep = [index for index, image_id in enumerate(ids) if image_id in capture_gallery]
            if keep:
                gallery_pixels.append(chunk[keep].detach().cpu().numpy())
            _append_gallery_target(model, model_id, chunk,
                                   gallery_captions[start:start + len(chunk)] if gallery_captions is not None else None,
                                   keep, gallery_raw_text, gallery_processed_text, target_parts)
    target = torch.cat(target_parts)
    target_gpu = target.to(model.device)
    query_chunks = _transform(query_paths, transform, torch, Image, int(plan["runs"][role]["execution"]["query_batch_size"]))
    scores = []
    score_devices = set()
    sort_devices = set()
    rank_parts = []
    query_ids, gallery_ids = _population(plan, role)
    hook_outputs = []
    hook = model.text_proj.register_forward_hook(lambda _module, _args, value: hook_outputs.append(value))
    with torch.no_grad():
        offset = 0
        for chunk in query_chunks:
            with model.maybe_autocast():
                reference = model.ln_vision(model.visual_encoder(chunk)).float()
            active_forward = 0
            active_capture_seen = False
            active_input_scope = "query"
            active_input_ids[:] = query_ids[offset:offset + len(chunk)]
            text_batch = captions[offset:offset + len(chunk)]
            keep = [index for index, query_id in enumerate(active_input_ids) if query_id in capture_queries]
            if keep:
                query_pixels.append(chunk[keep].detach().cpu().numpy())
            local = model.inference(reference, target_gpu, text_batch, None) if model_id == "csmcir" else model.inference(reference, target_gpu, text_batch)
            score_block = local.reshape(len(chunk), len(gallery_paths)).float()
            observed_ranks, score_device, sort_device = _author_validation_sort(source, model_id, score_block,
                [item.rsplit(":", 1)[-1] for item in query_ids[offset:offset + len(chunk)]], gallery_ids, torch)
            score_devices.add(score_device)
            sort_devices.add(sort_device)
            scores.append(score_block.cpu())
            rank_parts.append(observed_ranks)
            offset += len(chunk)
    hook.remove()
    observer.remove()
    query = torch.nn.functional.normalize(torch.cat(hook_outputs), dim=-1).float().cpu()
    scores_np = torch.cat(scores).numpy().astype(np.float32)
    target_np = target.numpy().astype(np.float32)
    query_np = query.numpy().astype(np.float32)
    ranks = torch.cat(rank_parts).numpy()
    labels = _labels(query_ids, gallery_ids)
    sorted_labels = ranks == labels[:, None]
    recalls = {f"recall{limit}": float(sorted_labels[:, :limit].sum() / len(labels) * 100) for limit in (10, 50)}
    if len(score_devices) != 1 or len(sort_devices) != 1:
        raise ValueError(f"native scorer changed score/sort device across query batches: scores={sorted(score_devices)}, sorts={sorted(sort_devices)}")
    if not np.all(sorted_labels.sum(axis=-1) == 1):
        raise ValueError("native FashionIQ metric did not preserve exact positive targets")
    np.savez_compressed(output, feature_layout="native_query256_gallery8192_v3", query_features=query_np, target_features=target_np,
                        scores=scores_np, ranks=ranks, labels=labels, query_ids=np.asarray(query_ids),
                        gallery_ids=np.asarray(gallery_ids), missing_keys=np.asarray(missing_keys), unexpected_keys=np.asarray(unexpected_keys))
    del model
    independent = np.full_like(scores_np, -np.inf)
    token_features = target_np.reshape(len(gallery_ids), 32, 256)
    for token_index in range(32):
        gallery_token = token_features[:, token_index, :]
        gallery_norm = gallery_token / np.maximum(np.linalg.norm(gallery_token, axis=1, keepdims=True), 1e-12)
        independent = np.maximum(independent, query_np @ gallery_norm.T)
    native_score_max_abs = float(np.max(np.abs(independent - scores_np)))
    if not np.allclose(independent, scores_np, atol=1e-5, rtol=1e-5):
        raise ValueError(f"native inference differs from independently scored stored features: max_abs={native_score_max_abs}")
    gallery_canonical = target_np.reshape(len(gallery_ids), -1)
    receipt = _v3_capture(plan, role, "native", started_at, plan_path, plan_sha256, {
        "query_features": (query_np, query_ids), "target_features": (gallery_canonical, gallery_ids),
        "scores": (scores_np, query_ids), "ranks": (ranks, query_ids), "labels": (labels, query_ids)})
    receipt["identity_evidence"] = _identity_evidence(
        role=role, query_paths=query_paths, gallery_paths=gallery_paths,
        query_ids=query_ids, gallery_ids=gallery_ids, labels=labels, model_texts=texts)
    if role in ("native_adapter_native", "native_adapter_adapter"):
        receipt["ranking_archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    capture_manifest: dict[str, Any] = {}
    if capture_gallery:
        expected_gallery_ids = [item for item in gallery_ids if item in capture_gallery]
        if (len(capture_gallery) > 5 or not gallery_pixels or
                (model_id == "csmcir" and (gallery_observed_ids != expected_gallery_ids or
                 len(gallery_tokens) != len(gallery_masks) or
                 sum(len(rows) for rows in gallery_tokens) != len(expected_gallery_ids)))):
            raise ValueError("bounded gallery input capture is empty, incomplete, or exceeds five selected IDs")
        gallery_arrays = {"transformed_pixels": np.concatenate(gallery_pixels)}
        if model_id == "csmcir":
            gallery_arrays.update({"token_ids": np.concatenate(gallery_tokens),
                                   "attention_mask": np.concatenate(gallery_masks),
                                   "raw_text": np.asarray(gallery_raw_text),
                                   "processed_text": np.asarray(gallery_processed_text)})
        capture_manifest["gallery"] = _save_input_capture(output, "gallery", [item for item in gallery_ids if item in capture_gallery], gallery_arrays)
    if capture_queries:
        expected_query_ids = [item for item in query_ids if item in capture_queries]
        captured_queries = sum(len(rows) for rows in query_tokens)
        if (len(capture_queries) > 5 or not query_pixels or not query_tokens or
                query_observed_ids != expected_query_ids or len(query_tokens) != len(query_masks) or
                captured_queries != len(expected_query_ids)):
            raise ValueError("bounded query input capture lacks complete transformed pixels or tokenizer observations")
        capture_manifest["query"] = _save_input_capture(output, "query", [item for item in query_ids if item in capture_queries],
                                                        {"transformed_pixels": np.concatenate(query_pixels),
                                                         "token_ids": np.concatenate(query_tokens),
                                                         "attention_mask": np.concatenate(query_masks),
                                                         "raw_composition_text": np.asarray([text for item, text in zip(query_ids, texts) if item in capture_queries]),
                                                         "processed_text": np.asarray([text for item, text in zip(query_ids, captions) if item in capture_queries])})
    if role in ("native_adapter_native", "native_adapter_adapter"):
        receipt["input_capture"] = capture_manifest
    return {"output": str(output), "query_shape": list(query_np.shape), "gallery_shape": list(target_np.shape), "metrics": recalls,
            "missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "v3_receipt": receipt,
            "input_capture": capture_manifest,
            "ranking_observation": {"sort": "torch.argsort(1-pred_sim, dim=-1)",
                                    "source_consumer": "CSMCIR/src/validate_blip_csmcir.py:compute_fiq_val_metrics" if model_id == "csmcir" else "SPRC/src/validate_blip.py:compute_fiq_val_metrics",
                                    "score_device": next(iter(score_devices)), "sort_device": next(iter(sort_devices))}}
def _adapter(plan: dict[str, Any], output: Path, role: str, started_at: str,
             plan_path: Path, plan_sha256: str) -> dict[str, Any]:
    import torch
    from workbench.backend.operator_config import resolve_config
    if plan["model_id"] == "csmcir":
        from workbench.composition.model_csmcir import Adapter
        adapter = Adapter(resolve_config(), checkpoint_id="fashioniq")
        scope = plan["category"]
    else:
        from workbench.composition.model_sprc import Adapter
        adapter = Adapter(checkpoint_id="sprc_fiq_vitl")
        scope = None
    root = output.parent / f"adapter-work-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    adapter.work_root = root
    query_paths, gallery_paths, texts = _paths(plan, role)
    query_ids, gallery_ids = _population(plan, role)
    capture = plan.get("input_capture", {})
    capture_root = output.with_name(f"{output.stem}.inputs")
    capture_root.mkdir(parents=True, exist_ok=True)
    query_positions = [index for index, query_id in enumerate(query_ids) if query_id in capture.get("query_ids", [])]
    selected_queries = [query_ids[index] for index in query_positions]
    selected_gallery = [image_id for image_id in gallery_ids if image_id in capture.get("gallery_ids", [])]
    setattr(adapter, "_v3_input_capture", {
        "queries": {"indices": query_positions, "selected_ids": selected_queries,
                    "output": str(capture_root / "query.npz"), "raw_composition_text": texts},
        "targets": {"ids": selected_gallery, "selected_ids": selected_gallery,
                    "output": str(capture_root / "gallery.npz")}})
    adapter.batch_size = int(plan["runs"][role]["execution"]["query_batch_size"])
    try:
        query = adapter.encode_queries(query_paths, texts, [scope] * len(texts))
        adapter.batch_size = int(plan["runs"][role]["execution"]["gallery_batch_size"])
        target = adapter.encode_targets(gallery_paths, [scope] * len(gallery_paths))
        query_ids, gallery_ids = _population(plan, role)
        labels = _labels(query_ids, gallery_ids)
        target_positions = [[int(index)] for index in labels]
        original_score = adapter.score
        score_observation: list[np.ndarray] = []
        def observe_score(*args: Any, **kwargs: Any) -> np.ndarray:
            value = original_score(*args, **kwargs)
            score_observation.append(value)
            return value
        adapter.score = observe_score
        try:
            production_ranks, candidate_rows = adapter.rank_and_topk(
                query, target, target_positions, [scope] * len(query), depth=min(50, len(gallery_ids)))
        finally:
            adapter.score = original_score
        if len(score_observation) != 1:
            raise ValueError(f"expected one adapter production scoring call, observed {len(score_observation)}")
        scores = np.asarray(score_observation[0], dtype=np.float32)
        ranks = np.argsort(-scores, axis=1, kind="stable")
        observed_target_ranks = np.argmax(ranks == np.asarray(target_positions), axis=1) + 1
        if not np.array_equal(np.asarray(production_ranks), observed_target_ranks):
            raise ValueError("adapter production rank differs from complete score ordering")
        for row, ranked in enumerate(candidate_rows):
            expected = [(int(pos), float(scores[row, pos])) for pos in ranks[row, :len(ranked)]]
            if [(int(pos), float(score)) for pos, score in ranked] != expected:
                raise ValueError("adapter production top-k differs from complete score ordering")
        sorted_labels = ranks == labels[:, None]
        recalls = {f"recall{limit}": float(sorted_labels[:, :limit].sum() / len(labels) * 100) for limit in (10, 50)}
        if not np.all(sorted_labels.sum(axis=-1) == 1):
            raise ValueError("adapter FashionIQ metric did not preserve exact positive targets")
        input_capture = {}
        for scope_name, expected in (("query", selected_queries), ("gallery", selected_gallery)):
            capture_path = capture_root / f"{scope_name}.npz"
            if expected:
                if not capture_path.is_file():
                    raise ValueError(f"adapter {scope_name} input capture was not produced")
                arrays, ordered_ids = _read_adapter_input_capture(capture_path, expected, scope=scope_name)
                names = sorted(arrays)
                shapes = {name: list(arrays[name].shape) for name in names}
                digests = {name: hashlib.sha256(np.ascontiguousarray(arrays[name]).tobytes()).hexdigest()
                           for name in names}
                entry = {"path": str(capture_path), "sha256": hashlib.sha256(capture_path.read_bytes()).hexdigest(),
                         "ordered_ids": ordered_ids, "arrays": shapes, "array_sha256": digests}
                entry["raw_composition_text" if scope_name == "query" else "raw_text"] = arrays["raw_text"].astype(str).tolist()
                entry["processed_text"] = arrays["processed_text"].astype(str).tolist()
                input_capture[scope_name] = entry
        np.savez_compressed(output, feature_layout="native_query256_gallery8192_v3", query_features=query, target_features=target,
                            scores=scores, ranks=ranks, labels=labels, query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids))
        receipt = _v3_capture(plan, role, "adapter", started_at, plan_path, plan_sha256, {
            "query_features": (query, query_ids), "target_features": (target, gallery_ids),
            "scores": (scores, query_ids), "ranks": (ranks, query_ids), "labels": (labels, query_ids)})
        receipt["identity_evidence"] = _identity_evidence(
            role=role, query_paths=query_paths, gallery_paths=gallery_paths,
            query_ids=query_ids, gallery_ids=gallery_ids, labels=labels, model_texts=texts)
        if role in ("native_adapter_native", "native_adapter_adapter"):
            receipt["input_capture"] = input_capture
            receipt["ranking_archive_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
        return {"output": str(output), "query_shape": list(query.shape), "gallery_shape": list(target.shape),
                "metrics": recalls, "v3_receipt": receipt, "input_capture": input_capture,
                "ranking_observation": {"sort": "CompositionAdapter.rank_and_topk/_ranks_from_scores/topk_from_scores",
                                        "score_device": "cpu", "sort_device": "cpu"}}
    finally:
        adapter.close()
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--role")
    parser.add_argument("--phase", choices=("native", "adapter"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    plan = read_plan(args.plan)
    if args.preflight:
        print(json.dumps(preflight(plan, args.plan), sort_keys=True))
        return 0
    if args.role not in plan["runs"] or args.phase is None:
        parser.error("--role and --phase required unless --preflight")
    if (args.phase == "native") != plan["runs"][args.role]["source_id"].startswith("native:"):
        parser.error("phase does not match preregistered role side")
    output = args.output or args.plan.with_name(f"{args.plan.stem}.{args.role}.{args.phase}.npz")
    plan_path = args.plan.resolve()
    plan_sha256 = digest(plan_path.read_bytes())
    started = datetime.now(timezone.utc).isoformat()
    result = (_native(plan, output, args.role, started, plan_path, plan_sha256) if args.phase == "native"
              else _adapter(plan, output, args.role, started, plan_path, plan_sha256))
    result.update({"run_id": plan["runs"][args.role]["run_id"], "phase": args.phase, "started_at": started, "role": args.role})
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
