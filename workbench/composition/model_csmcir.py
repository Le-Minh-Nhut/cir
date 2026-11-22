"""CSMCIR FashionIQ adapter with native 32-token gallery and text-query features.

Target encoding uses the author's category-specific ``Final_Caption`` and native
``extract_target_caption_features``. Query encoding reproduces ``inference``'s fused,
normalized text projection; scoring takes its maximum dot product over 32 gallery tokens.
Caption overlap requires an explicit FashionIQ category scope.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from workbench.composition.adapters import (
    AdapterStatus, AdapterWorkRootError, AdapterWorkerError, CompositionAdapter, worker_environment,
)

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
_TARGET_REASON = (
    "CSMCIR targets require category-specific author captions. Use one explicit FashionIQ category "
    "scope; an unscoped gallery is unsupported because caption IDs overlap across categories."
)


class Adapter(CompositionAdapter):
    """Run CSMCIR in its model-specific interpreter; never import torch in controller."""

    model_id = "csmcir"
    reference_eligible = True
    query_dimension: int | None = 256
    score_dimension: int = 8192
    category_scopes: tuple[str | None, ...] = ("dress", "shirt", "toptee")

    def __init__(self, config: Any, checkpoint_id: str | None = None) -> None:
        if CompositionAdapter is not object:
            super().__init__(config, checkpoint_id)
        else:
            self.config, self.checkpoint_id = config, checkpoint_id or "fashioniq"
        self.checkpoint_id = self.checkpoint_id or "fashioniq"
        self._model, self._checkpoint, self._interpreter = _resolve(config, self.checkpoint_id)
        #: Runner-owned root; each adapter removes only its private child on close.
        self.work_root: Path | None = None
        self._job_root: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any = None
        self._worker_counter = 0
        self._worker_failed: str | None = None
        self._interpreter_verified = False
        #: Optional FashionIQ parity caption merge order (e.g. ("dress", "shirt", "toptee")).
        #: Default None preserves single-category caption loading for CIRCUS.
        self.target_caption_merge_order: tuple[str, ...] | None = None

    def prepare_identity(self) -> dict[str, Any]:
        """Bind target-caption bytes and native preprocessing sources to feature-cache identity."""
        source = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self._model["source_dir"] / "src"
        caption_root = (Path(self.config.WORKBENCH_ARTIFACTS_ROOT) / "assets" / "csmcir" /
                       "cf0c19bb346266c295b4b5772ebf03bd1c4f0467" / "fashioniq_qwen_captions" /
                       "qwen_captions")
        categories = {}
        for category in self.category_scopes:
            path = caption_root / f"{category}_cot_val.json"
            categories[category] = {"path": str(path.resolve()), "sha256": _sha256_file(path)}
        config_path = source / "lavis/configs/models/blip2/blip2_pretrain.yaml"
        processor_path = source / "lavis/processors/blip_processors.py"
        model_path = source / "lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py"
        dataset_path = source / "data_utils_csmcir.py"
        checkpoint_path = Path(self._checkpoint["resolved_path"])
        raw_merge = getattr(self, "target_caption_merge_order", None)
        merge_order = list(raw_merge) if raw_merge else None
        target_protocol = (
            f"merged_{'_'.join(merge_order)}_Final_Caption_raw_tokenizer"
            if merge_order
            else "category_scoped_Final_Caption_raw_tokenizer"
        )
        return {
            "target_caption_assets": categories,
            "target_caption_merge_order": merge_order,
            "checkpoint": {"path": str(checkpoint_path.resolve()), "sha256": _sha256_file(checkpoint_path)},
            "native_configuration_sha256": _sha256_file(config_path),
            "text_processor_source_sha256": _sha256_file(processor_path),
            "image_preprocessor_source_sha256": _sha256_file(dataset_path),
            "query_text_protocol": "fashioniq_pair_join_strip_capitalize_and_eval_processor",
            "target_text_protocol": target_protocol,
            "target_category_scopes": list(self.category_scopes),
            "query_dimension": self.query_dimension,
            "score_dimension": self.score_dimension,
            "cache_policy": self.cache_policy,
            "batch_size": self.batch_size,
            "query_precision": "native_fp32_after_visual_autocast",
            "target_precision": "native_model_maybe_autocast",
        }
    @classmethod
    def status(cls, config: Any | None = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config
        return _status(cls, config or resolve_config())
    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        if condition not in CONDITIONS:
            return False, f"unknown composition condition: {condition!r}"
        return True, None


    def encode_queries(self, image_paths: Sequence[str | Path], texts: Sequence[str],
                        categories: Sequence[str | None]) -> np.ndarray:
        _required_work_root(self)
        _check_batch(image_paths, texts, categories)
        return _run(self, "queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str | Path], categories: Sequence[str | None]) -> np.ndarray:
        _required_work_root(self)
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        if not categories:
            return np.empty((0, self.score_dimension), dtype=np.float32)
        if len(set(categories)) != 1 or categories[0] not in self.category_scopes:
            raise ValueError("csmcir targets require one explicit FashionIQ category scope")
        return _run(self, "targets", image_paths, (), categories)

    def score(self, queries: Any, targets: Any, categories: Sequence[str | None] | None = None) -> np.ndarray:
        del categories
        query, target = np.asarray(queries, dtype=np.float32), np.asarray(targets, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != 256 or target.ndim != 2 or target.shape[1] != 8192:
            raise ValueError("expected native query (N,256) and projected target features (M,8192)")
        tokens = target.reshape(len(target), 32, 256)
        query = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
        scores = np.full((len(query), len(target)), -np.inf, dtype=np.float32)
        for token in range(32):
            target_token = tokens[:, token, :]
            target_token = target_token / np.maximum(
                np.linalg.norm(target_token, axis=1, keepdims=True), 1e-12)
            np.maximum(scores, query @ target_token.T, out=scores)
        return scores

    def rank(self, queries: Any, targets: Any, target_positions: Sequence[Sequence[int] | None] | None = None,
             categories: Sequence[str | None] | None = None) -> list[int | None]:
        return super().rank(queries, targets, target_positions, categories)

    def close(self) -> None:
        _stop_worker(self)
        _cleanup(self._job_root, self.work_root)
        self._job_root = None


def _stop_worker(adapter: Adapter) -> None:
    worker = adapter._worker
    if worker is not None:
        if worker.stdin is not None:
            try:
                worker.stdin.write(json.dumps({"stop": True}) + "\n")
                worker.stdin.flush()
            except OSError:
                pass
        try:
            worker.wait(timeout=5)
        except subprocess.TimeoutExpired:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait()
        adapter._worker = None
    if adapter._worker_log is not None:
        adapter._worker_log.close()
        adapter._worker_log = None


def _check_batch(image_paths: Sequence[str | Path], texts: Sequence[str], categories: Sequence[str | None]) -> None:
    if not (len(image_paths) == len(texts) == len(categories)):
        raise ValueError("image_paths, texts, and categories must have equal lengths")


def _resolve(config: Any, checkpoint_id: str) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("csmcir")
    checkpoint = checkpoint_by_id(model, checkpoint_id)
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)


def _run(adapter: Adapter, operation: str, image_paths: Sequence[str | Path], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    work_root = _required_work_root(adapter)
    if not image_paths:
        width = adapter.query_dimension if operation == "queries" else adapter.score_dimension
        if width is None:
            raise RuntimeError("CSMCIR query_dimension must be configured")
        return np.empty((0, width), dtype=np.float32)
    if not adapter._interpreter:
        raise RuntimeError("no interpreter environment resolved for WORKBENCH_CSMCIR_PYTHON")
    if not adapter._interpreter_verified:
        adapter.require_verified_interpreter()
        adapter._interpreter_verified = True
    checkpoint = Path(adapter._checkpoint["resolved_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"csmcir checkpoint not staged at {checkpoint}")
    job_root = work_root / f"csmcir-{adapter.checkpoint_id}"
    if not job_root.resolve().is_relative_to(work_root.resolve()):
        raise ValueError(f"unsafe job_root: {job_root}")
    job_root.mkdir(parents=True, exist_ok=True)
    adapter._job_root = job_root
    if adapter._worker_failed:
        raise AdapterWorkerError(adapter._worker_failed)
    adapter._worker_counter += 1
    output, job = job_root / f"{adapter._worker_counter}.npy", job_root / f"{adapter._worker_counter}.json"
    source = Path(adapter.config.WORKBENCH_THIRD_PARTY_ROOT) / adapter._model["source_dir"]
    asset_root = Path(adapter.config.WORKBENCH_ARTIFACTS_ROOT) / "assets" / "csmcir" / "cf0c19bb346266c295b4b5772ebf03bd1c4f0467" / "fashioniq_qwen_captions" / "qwen_captions"
    capture = getattr(adapter, "_v3_input_capture", {}).get(operation)
    raw_merge = getattr(adapter, "target_caption_merge_order", None)
    merge_order = list(raw_merge) if raw_merge else None
    job.write_text(json.dumps({"operation": operation, "image_paths": [str(path) for path in image_paths],
                               "texts": list(texts), "categories": list(categories), "output": str(output),
                               "source_root": str(source), "caption_root": str(asset_root),
                               "caption_merge_order": merge_order,
                               "checkpoint_path": str(checkpoint), "batch_size": adapter.batch_size,
                               "capture": capture}), encoding="utf-8")
    try:
        if adapter._worker is not None and adapter._worker.poll() is not None:
            raise AdapterWorkerError(f"CSMCIR worker exited with status {adapter._worker.returncode}")
        if adapter._worker is None:
            adapter._worker_log = (job_root / "worker.stderr.log").open("w", encoding="utf-8")
            adapter._worker = subprocess.Popen(
                [adapter._interpreter, "-m", "workbench.composition.model_csmcir", "--worker-job", str(job)],
                cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=adapter._worker_log, text=True, bufsize=1, env=worker_environment())
        else:
            assert adapter._worker.stdin is not None
            adapter._worker.stdin.write(json.dumps({"job_path": str(job)}) + "\n")
            adapter._worker.stdin.flush()
        assert adapter._worker.stdout is not None
        line = adapter._worker.stdout.readline()
        while line and not line.startswith("CIRCUS_WORKER:"):
            line = adapter._worker.stdout.readline()
        if not line:
            raise AdapterWorkerError(f"CSMCIR worker exited with status {adapter._worker.poll()}")
        reply = json.loads(line[len("CIRCUS_WORKER:"):])
        if not reply.get("ok"):
            raise AdapterWorkerError(f"CSMCIR worker failed: {reply.get('error_type')}: {reply.get('error')}")
        features = np.load(output, allow_pickle=False)
        width = adapter.query_dimension if operation == "queries" else adapter.score_dimension
        if width is None:
            raise RuntimeError("CSMCIR query_dimension must be configured")
        if features.dtype != np.float32 or features.shape != (len(image_paths), width):
            raise AdapterWorkerError(f"CSMCIR worker returned invalid matrix {features.shape}/{features.dtype}; expected width {width}")
        return features
    except (OSError, AdapterWorkerError) as error:
        adapter._worker_failed = str(error)
        _stop_worker(adapter)
        raise AdapterWorkerError(adapter._worker_failed) from error
    finally:
        job.unlink(missing_ok=True)
        output.unlink(missing_ok=True)


def _required_work_root(adapter: Adapter) -> Path:
    if adapter.work_root is None:
        raise AdapterWorkRootError("csmcir work_root was not set by runner")
    return adapter.work_root


def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.model_settings import setting
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import checkpoint_by_id, checkpoint_is_present, checkpoint_path, load_registry, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    config = config if config is not None else resolve_config()
    registry = getattr(config, "registry", None) or load_registry()
    model = model_by_id(cls.model_id, registry)
    checkpoint = checkpoint_by_id(model, "fashioniq")
    reasons = []
    if checkpoint is None:
        reasons.append("CSMCIR FashionIQ checkpoint mapping missing")
    else:
        path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        if not checkpoint_is_present(path, checkpoint, model_id=cls.model_id, config=config):
            reasons.append(f"CSMCIR FashionIQ checkpoint missing or invalid: {path}")
    interpreter = resolve_model_interpreter(model, config) or setting(model["environment"]["python_env_var"])
    if not interpreter or not Path(interpreter).is_file():
        reasons.append(f"CSMCIR model interpreter missing: {interpreter or 'environment not configured'}")
    else:
        from workbench.composition.adapters import _environment_gate_reason
        if reason := _environment_gate_reason(model, config):
            reasons.append(reason)
    return AdapterStatus(cls.model_id, not reasons, "; ".join(reasons) or None,
                         tuple(str(item["checkpoint_id"]) for item in model["checkpoint_variants"]),
                         {condition: (True, None) for condition in CONDITIONS}, _TARGET_REASON)


def _cleanup(path: Path | None, work_root: Path | None = None) -> None:
    if path is not None and path.exists():
        if work_root is not None and not path.resolve().is_relative_to(work_root.resolve()):
            raise ValueError(f"unsafe cleanup target: {path}")
        shutil.rmtree(path)

def _format_keys(keys: Sequence[str], limit: int = 5) -> str:
    if len(keys) <= limit:
        return ", ".join(repr(k) for k in keys)
    return f"{', '.join(repr(k) for k in keys[:limit])} (and {len(keys) - limit} more)"

def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _load_caption_map(cache: dict[tuple[str, str], dict[str, Any]], category: str,
                     caption_path: Path) -> dict[str, Any]:
    content = caption_path.read_bytes()
    key = (category, hashlib.sha256(content).hexdigest())
    captions = cache.get(key)
    if captions is None:
        captions = json.loads(content)
        cache.clear()
        cache[key] = captions
    return captions


def _load_merged_caption_map(cache: dict[tuple[str, str], dict[str, Any]],
                             caption_root: Path,
                             merge_order: Sequence[str]) -> dict[str, Any]:
    hasher = hashlib.sha256()
    raw_parts: list[bytes] = []
    for cat in merge_order:
        content = (caption_root / f"{cat}_cot_val.json").read_bytes()
        hasher.update(cat.encode("utf-8") + b":" + content)
        raw_parts.append(content)
    key = ("merged:" + ",".join(merge_order), hasher.hexdigest())
    captions = cache.get(key)
    if captions is None:
        merged: dict[str, Any] = {}
        for content in raw_parts:
            merged.update(json.loads(content))
        cache.clear()
        cache[key] = merged
        captions = merged
    return captions


def _csmcir_state_dict(state: Any) -> Any:
    class_key = "Blip2QformerCirAlignPromptCSMCIR"
    if class_key in state:
        return state[class_key]
    candidates = [value for key, value in state.items() if key != "epoch" and isinstance(value, dict)]
    if len(candidates) != 1:
        raise ValueError(f"expected one CSMCIR checkpoint state dictionary, found {len(candidates)}")
    return candidates[0]


def _load_csmcir_checkpoint(model: Any, state: Any) -> None:
    report = model.load_state_dict(state, strict=False)
    missing = list(getattr(report, "missing_keys", ()))
    unexpected = list(getattr(report, "unexpected_keys", ()))
    if missing:
        raise ValueError(
            "CSMCIR checkpoint is missing model keys: "
            f"{len(missing)} [{_format_keys(missing)}]; unexpected={len(unexpected)} "
            f"[{_format_keys(unexpected)}]"
        )

def _worker_main(job_path: str) -> int:
    import torch
    from PIL import Image
    from workbench.replay.lavis_processors import prepare

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    source = Path(job["source_root"]) / "src"
    prepare(source)
    from lavis.models import load_model_and_preprocess
    from data_utils_csmcir import targetpad_transform
    model, _, processors = load_model_and_preprocess(
        name="blip2_cir_align_prompt_csmcir", model_type="pretrain", is_eval=True, device="cuda")
    state = torch.load(job["checkpoint_path"], map_location="cuda", weights_only=False)
    _load_csmcir_checkpoint(model, _csmcir_state_dict(state))
    model.eval()
    transform = targetpad_transform(1.25, 224)

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    caption_cache: dict[tuple[str, str], dict[str, Any]] = {}
    def run(current: dict[str, Any]) -> None:
        all_features = []
        capture = current.get("capture") or {}
        size = max(1, int(current["batch_size"]))
        captions = None
        if current["operation"] == "targets":
            merge_order = current.get("caption_merge_order")
            if merge_order:
                captions = _load_merged_caption_map(caption_cache, Path(current["caption_root"]), merge_order)
            else:
                category = current["categories"][0]
                caption_path = Path(current["caption_root"]) / f"{category}_cot_val.json"
                captions = _load_caption_map(caption_cache, category, caption_path)
        forward_capture = {"active": False, "calls": 0, "keep": [], "raw_texts": [], "texts": []}
        def observe_qformer(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            if not forward_capture["active"]:
                return
            forward_capture["calls"] += 1
            if forward_capture["calls"] != 1 or not forward_capture["keep"]:
                return
            input_ids = args[0] if args else kwargs["input_ids"]
            capture.setdefault("tokens", []).append(input_ids[forward_capture["keep"]].detach().cpu().numpy())
            capture.setdefault("masks", []).append(kwargs["attention_mask"][forward_capture["keep"]].detach().cpu().numpy())
            capture.setdefault("text", []).extend(forward_capture["raw_texts"][index] for index in forward_capture["keep"])
            capture.setdefault("processed_text", []).extend(forward_capture["texts"][index] for index in forward_capture["keep"])
        observer = model.Qformer.bert.register_forward_pre_hook(observe_qformer, with_kwargs=True)
        try:
            for start in range(0, len(current["image_paths"]), size):
                paths = current["image_paths"][start:start + size]
                images = []
                for path in paths:
                    with Image.open(path) as image:
                        images.append(transform(image.convert("RGB")))
                batch = torch.stack(images).cuda()
                if current["operation"] == "queries":
                    positions = capture.get("indices", [])
                    keep = [index for index in range(len(paths)) if start + index in positions]
                else:
                    keep = [index for index, path in enumerate(paths) if Path(path).stem in capture.get("ids", [])]
                if keep:
                    capture.setdefault("pixels", []).append(batch[keep].detach().cpu().numpy())
                with torch.no_grad():
                    if current["operation"] == "queries":
                        with model.maybe_autocast():
                            visual = model.ln_vision(model.visual_encoder(batch)).float()
                        raw_texts = current["texts"][start:start + len(paths)]
                        texts = [processors["eval"](text) for text in raw_texts]
                        tokens = model.tokenizer(texts, padding="max_length", truncation=True,
                                                 max_length=model.max_txt_len, return_tensors="pt").to(visual.device)
                        query_tokens = model.query_tokens.expand(visual.shape[0], -1, -1)
                        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=visual.device)
                        image_atts = torch.ones(visual.size()[:-1], dtype=torch.long, device=visual.device)
                        attention_mask = torch.cat([query_atts, tokens.attention_mask], dim=1)
                        if keep:
                            capture.setdefault("tokens", []).append(tokens.input_ids[keep].detach().cpu().numpy())
                            capture.setdefault("masks", []).append(attention_mask[keep].detach().cpu().numpy())
                            capture.setdefault("text", []).extend(raw_texts[index] for index in keep)
                            capture.setdefault("processed_text", []).extend(texts[index] for index in keep)
                        fusion = model.Qformer.bert(tokens.input_ids, query_embeds=query_tokens,
                                                    attention_mask=attention_mask, encoder_hidden_states=visual,
                                                    encoder_attention_mask=image_atts, return_dict=True)
                        features = torch.nn.functional.normalize(model.text_proj(fusion.last_hidden_state[:, 32, :]), dim=-1)
                    elif current["operation"] == "targets":
                        ids = [Path(path).stem for path in paths]
                        if captions is None or any(image_id not in captions or not captions[image_id].get("Final_Caption") for image_id in ids):
                            raise ValueError("CSMCIR target captions missing for requested category or image IDs")
                        texts = [captions[image_id]["Final_Caption"] for image_id in ids]
                        forward_capture.update({"active": True, "calls": 0, "keep": keep,
                                                "raw_texts": texts, "texts": texts})
                        try:
                            features, _ = model.extract_target_caption_features(batch, texts, mode="mean")
                        finally:
                            forward_capture["active"] = False
                        features = features.reshape(len(ids), 8192)
                    else:
                        raise ValueError(f"unknown CSMCIR operation: {current['operation']!r}")
                all_features.append(features.float().cpu().numpy())
        finally:
            observer.remove()
        np.save(current["output"], np.concatenate(all_features, axis=0))
        if capture.get("pixels"):
            pixels = np.concatenate(capture["pixels"])
            tokens = np.concatenate(capture["tokens"])
            masks = np.concatenate(capture["masks"])
            if len(pixels) != len(capture["selected_ids"]) or len(tokens) != len(pixels) or len(masks) != len(pixels):
                raise ValueError("CSMCIR input capture row count mismatch")
            arrays = {"transformed_pixels": pixels, "raw_text": np.asarray(capture["text"]),
                      "processed_text": np.asarray(capture["processed_text"]),
                      "token_ids": tokens, "attention_mask": masks}
            if len(arrays["raw_text"]) != len(pixels) or len(arrays["processed_text"]) != len(pixels):
                raise ValueError("CSMCIR input capture text row count mismatch")
            np.savez_compressed(capture["output"], ordered_ids=np.asarray(capture["selected_ids"]), **arrays)
    try:
        run(job)
        reply(True)
        for line in sys.stdin:
            request = json.loads(line)
            if request.get("stop"):
                break
            try:
                run(json.loads(Path(request["job_path"]).read_text(encoding="utf-8")))
                reply(True)
            except Exception as error:
                reply(False, error_type=type(error).__name__, error=str(error))
                return 1
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return 1
    return 0


if __name__ == "__main__" and "--worker-job" in sys.argv:
    raise SystemExit(_worker_main(sys.argv[sys.argv.index("--worker-job") + 1]))


ADAPTER = Adapter
