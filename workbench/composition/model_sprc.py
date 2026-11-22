"""SPRC FashionIQ adapter for the author-released ViT-L checkpoint."""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from workbench.composition.adapters import (
    AdapterStatus, AdapterWorkerError, AdapterWorkRootError, CompositionAdapter, worker_environment,
)

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
CHECKPOINT_ID = "sprc_fiq_vitl"


def _target_transport(features: np.ndarray) -> np.ndarray:
    """Flatten already-normalized SPRC target tokens for adapter transport."""
    if features.ndim != 3 or features.shape[1:] != (32, 256):
        raise ValueError(f"expected normalized SPRC target tokens (N,32,256), got {features.shape}")
    return features.reshape(features.shape[0], 8192)

class Adapter(CompositionAdapter):
    model_id = "sprc"
    query_dimension: int | None = 256
    score_dimension: int = 8192

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config
        from workbench.backend.registry import checkpoint_by_id, load_registry
        from workbench.backend.model_settings import setting
        from workbench.backend.runtime import resolve_model_interpreter

        self.config = config if config is not None else resolve_config()
        checkpoint_id = checkpoint_id or CHECKPOINT_ID
        if checkpoint_id != CHECKPOINT_ID:
            raise ValueError(f"SPRC adapter requires {CHECKPOINT_ID!r}; ViT-G is a distinct, unverified variant")
        super().__init__(self.config, checkpoint_id)
        registry = getattr(self.config, "registry", None) or load_registry()
        model = next((row for row in registry["models"] if row["model_id"] == self.model_id), None)
        if model is None:
            raise ValueError("SPRC model registry row missing")
        checkpoint = checkpoint_by_id(model, checkpoint_id)
        if checkpoint is None:
            raise ValueError(f"SPRC checkpoint variant missing: {checkpoint_id}")
        self.model = model
        self.checkpoint = checkpoint
        self.checkpoint_id = checkpoint_id
        self.interpreter = resolve_model_interpreter(model, self.config) or setting(model["environment"]["python_env_var"])
        self.work_root: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any = None
        self._worker_counter = 0
        self._worker_failed: str | None = None
        self._job_root: Path | None = None

    def prepare_identity(self) -> dict[str, Any]:
        if self.model is None or self.checkpoint is None:
            raise ValueError("SPRC model/checkpoint mapping required for cache identity")
        from workbench.backend.registry import checkpoint_path
        source = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self.model["source_dir"] / "src"
        config_path = source / "lavis/configs/models/blip2/blip2_pretrain_vitL.yaml"
        model_path = source / "lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py"
        processor_path = source / "lavis/processors/blip_processors.py"
        dataset_path = source / "data_utils.py"
        checkpoint_file = checkpoint_path(self.model_id, self.checkpoint, self.config.WORKBENCH_CHECKPOINT_ROOT)
        return {
            "checkpoint": {"path": str(checkpoint_file.resolve()), "sha256": _sha256_file(checkpoint_file)},
            "query_text_protocol": "fashioniq_pair_join_strip_capitalize_eval_processor_native_two_stage_qformer",
            "target_feature_protocol": "native_extract_target_features_mean_32_tokens",
            "native_configuration": {"path": str(config_path), "sha256": _sha256_file(config_path)},
            "native_model_source_sha256": _sha256_file(model_path),
            "text_processor_source_sha256": _sha256_file(processor_path),
            "image_preprocessor_source_sha256": _sha256_file(dataset_path),
            "query_dimension": self.query_dimension,
            "score_dimension": self.score_dimension,
            "cache_policy": self.cache_policy,
            "batch_size": self.batch_size,
            "precision": "native_model_maybe_autocast; float32_feature_storage",
        }

    @classmethod
    def status(cls, config: Any = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config
        from workbench.backend.registry import checkpoint_by_id, checkpoint_is_present, checkpoint_path, load_registry
        from workbench.backend.model_settings import setting
        from workbench.backend.runtime import resolve_model_interpreter

        config = config if config is not None else resolve_config()
        registry = getattr(config, "registry", None) or load_registry()
        model = next((row for row in registry["models"] if row["model_id"] == cls.model_id), None)
        if model is None:
            return AdapterStatus(cls.model_id, False, "SPRC model registry row missing", (), {})
        checkpoint = checkpoint_by_id(model, CHECKPOINT_ID)
        if checkpoint is None:
            return AdapterStatus(cls.model_id, False, "SPRC ViT-L checkpoint mapping missing", (), {})
        path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        reasons = []
        if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
            reasons.append(f"SPRC checkpoint mapping unresolved: {CHECKPOINT_ID}")
        if not checkpoint_is_present(path, checkpoint, model_id=cls.model_id, config=config):
            reasons.append(f"SPRC ViT-L checkpoint missing or invalid: {path}")
        interpreter = resolve_model_interpreter(model, config) or setting(model["environment"]["python_env_var"])
        if not interpreter or not Path(interpreter).is_file():
            reasons.append(f"SPRC model interpreter missing: {interpreter or 'environment not configured'}")
        else:
            from workbench.composition.adapters import _environment_gate_reason
            if reason := _environment_gate_reason(model, config):
                reasons.append(reason)
        return AdapterStatus(cls.model_id, not reasons, "; ".join(reasons) or None, (CHECKPOINT_ID,),
                             {condition: (True, None) for condition in CONDITIONS},
                             "Author-released ViT-L/14 variant; not comparable to paper ViT-G results.")

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        return ((True, None) if condition in CONDITIONS else
                (False, f"unknown composition condition: {condition!r}"))

    def _run(self, operation: str, paths: Sequence[str | Path], texts: Sequence[str] = ()) -> np.ndarray:
        if self.work_root is None:
            raise AdapterWorkRootError("SPRC work_root was not set by runner")
        if self._worker_failed:
            raise AdapterWorkerError(self._worker_failed)
        if not paths:
            width = self.query_dimension if operation == "queries" else self.score_dimension
            assert width is not None
            return np.empty((0, width), dtype=np.float32)
        interpreter = self.require_verified_interpreter()
        if not self.interpreter:
            raise RuntimeError("no interpreter environment resolved for WORKBENCH_SPRC_PYTHON")
        from workbench.backend.registry import checkpoint_path
        model, checkpoint = self.model, self.checkpoint
        if model is None or checkpoint is None:
            raise ValueError("SPRC model or checkpoint registry mapping missing")
        if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
            raise ValueError(f"SPRC checkpoint mapping is unresolved: {self.checkpoint_id}")
        checkpoint_file = checkpoint_path(self.model_id, checkpoint, self.config.WORKBENCH_CHECKPOINT_ROOT)
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"SPRC ViT-L checkpoint not staged: {checkpoint_file}")
        source = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / model["source_dir"]
        root = self.work_root / f".sprc-{self.checkpoint_id}"
        root.mkdir(parents=True, exist_ok=True)
        if self._job_root is None:
            self._job_root = root
        elif self._job_root != root:
            raise ValueError("SPRC adapter work root changed during run")
        capture = getattr(self, "_v3_input_capture", {}).get(operation)
        job_path, output = root / f"job-{self._worker_counter}.json", root / f"features-{self._worker_counter}.npy"
        job_path.write_text(json.dumps({"operation": operation, "paths": [str(p) for p in paths],
                                       "texts": list(texts), "source": str(source),
                                       "checkpoint": str(checkpoint_file), "output": str(output),
                                       "batch_size": self.batch_size, "capture": capture}), encoding="utf-8")
        try:
            if self._worker is not None and self._worker.poll() is not None:
                raise AdapterWorkerError(f"SPRC worker exited with status {self._worker.returncode}")
            if self._worker is None:
                self._worker_log = (root / "worker.stderr.log").open("w", encoding="utf-8")
                self._worker = subprocess.Popen(
                    [interpreter, "-m", "workbench.composition.model_sprc", "--worker-job", str(job_path)],
                    cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=self._worker_log, text=True, bufsize=1, env=worker_environment())
            else:
                assert self._worker.stdin is not None
                self._worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
                self._worker.stdin.flush()
            assert self._worker.stdout is not None
            line = self._worker.stdout.readline()
            while line and not line.startswith("SPRC_WORKER:"):
                line = self._worker.stdout.readline()
            if not line:
                raise AdapterWorkerError(f"SPRC worker exited with status {self._worker.poll()}")
            response = json.loads(line[len("SPRC_WORKER:"):])
            if not response.get("ok"):
                raise AdapterWorkerError(f"SPRC worker failed: {response.get('error_type')}: {response.get('error')}")
            features = np.load(output, allow_pickle=False)
            width = self.query_dimension if operation == "queries" else self.score_dimension
            if features.dtype != np.float32 or features.shape != (len(paths), width):
                raise AdapterWorkerError(f"SPRC worker returned invalid matrix {features.shape}/{features.dtype}; expected width {width}")
            return features
        except (OSError, AdapterWorkerError) as error:
            self._worker_failed = str(error)
            _stop_worker(self)
            raise AdapterWorkerError(self._worker_failed) from error
        finally:
            job_path.unlink(missing_ok=True)
            output.unlink(missing_ok=True)

    def encode_queries(self, image_paths: Sequence[str | Path], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if not (len(image_paths) == len(texts) == len(categories)):
            raise ValueError("image_paths, texts, and categories must have equal lengths")
        return self._run("queries", image_paths, texts)

    def encode_targets(self, image_paths: Sequence[str | Path],
                       categories: Sequence[str | None]) -> np.ndarray:
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        return self._run("targets", image_paths)

    def score(self, queries: Any, targets: Any, categories: Sequence[str | None] | None = None) -> np.ndarray:
        del categories
        query, target = np.asarray(queries, dtype=np.float32), np.asarray(targets, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != 256 or target.ndim != 2 or target.shape[1] != 8192:
            raise ValueError("expected native SPRC query (N,256) and projected target features (M,8192)")
        q = query
        gallery = target.reshape(len(target), 32, 256)
        q = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
        scores = np.full((len(query), len(target)), -np.inf, dtype=np.float32)
        for token in range(32):
            gallery_token = gallery[:, token, :]
            gallery_token = gallery_token / np.maximum(
                np.linalg.norm(gallery_token, axis=1, keepdims=True), 1e-12)
            np.maximum(scores, q @ gallery_token.T, out=scores)
        return scores

    def close(self) -> None:
        _stop_worker(self)
        if self._job_root is not None:
            if self.work_root is None or not self._job_root.resolve().is_relative_to(self.work_root.resolve()):
                raise ValueError(f"unsafe cleanup target: {self._job_root}")
            shutil.rmtree(self._job_root, ignore_errors=True)
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


def _worker_main(job_file: str) -> int:
    import sys
    import torch
    from PIL import Image
    from workbench.replay.lavis_processors import prepare

    job = json.loads(Path(job_file).read_text(encoding="utf-8"))
    prepare(Path(job["source"]) / "src")
    from lavis.models import load_model_and_preprocess
    from data_utils import targetpad_transform

    model, _, txt_processors = load_model_and_preprocess(name="blip2_cir_align_prompt", model_type="pretrain_vitL",
                                                         is_eval=False, device="cuda")
    checkpoint = torch.load(job["checkpoint"], map_location="cuda", weights_only=False)
    model.load_state_dict(checkpoint[model.__class__.__name__], strict=True)
    model.eval()
    transform = targetpad_transform(1.25, 224)

    def reply(ok: bool, **values: Any) -> None:
        print("SPRC_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    def run(current: dict[str, Any]) -> None:
        capture = current.get("capture") or {}
        chunks = []
        chunk_size = max(1, int(current["batch_size"]))
        for start in range(0, len(current["paths"]), chunk_size):
            paths = current["paths"][start:start + chunk_size]
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).cuda()
            if current["operation"] == "queries":
                positions = capture.get("indices", [])
                keep = [index for index in range(len(paths)) if start + index in positions]
            else:
                selected_ids = capture.get("ids", [])
                keep = [index for index, path in enumerate(paths) if Path(path).stem in selected_ids]
            if keep:
                capture.setdefault("pixels", []).append(batch[keep].detach().cpu().numpy())
            with torch.no_grad():
                if current["operation"] == "targets":
                    features, _ = model.extract_target_features(batch, mode="mean")
                elif current["operation"] == "queries":
                    with model.maybe_autocast():
                        visual = model.ln_vision(model.visual_encoder(batch))
                    visual = visual.float()
                    raw_texts = current["texts"][start:start + len(paths)]
                    texts = [txt_processors["eval"](text) for text in raw_texts]
                    tokens = model.tokenizer(texts, padding="max_length", truncation=True,
                                             max_length=model.max_txt_len, return_tensors="pt").to(visual.device)
                    query_tokens = model.query_tokens.expand(visual.shape[0], -1, -1)
                    qatts = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=visual.device)
                    iatts = torch.ones(visual.size()[:-1], dtype=torch.long, device=visual.device)
                    mask = torch.cat([qatts, tokens.attention_mask], dim=1)
                    if keep:
                        capture.setdefault("text", []).extend(raw_texts[index] for index in keep)
                        capture.setdefault("processed_text", []).extend(texts[index] for index in keep)
                        capture.setdefault("tokens", []).append(tokens.input_ids[keep].detach().cpu().numpy())
                        capture.setdefault("masks", []).append(mask[keep].detach().cpu().numpy())
                    fusion = model.Qformer.bert(tokens.input_ids, query_embeds=query_tokens, attention_mask=mask,
                                                encoder_hidden_states=visual, encoder_attention_mask=iatts,
                                                return_dict=True)
                    query = model.Qformer.bert(tokens.input_ids,
                                               query_embeds=fusion.last_hidden_state[:, :query_tokens.size(1), :],
                                               attention_mask=mask, return_dict=True)
                    features = torch.nn.functional.normalize(model.text_proj(query.last_hidden_state[:, 32, :]), dim=-1)
                else:
                    raise ValueError(f"unknown SPRC operation: {current['operation']!r}")
            values = features.float().cpu().numpy()
            chunks.append(_target_transport(values) if current["operation"] == "targets" else values)
        np.save(current["output"], np.concatenate(chunks, axis=0), allow_pickle=False)
        if capture.get("pixels"):
            arrays = {"transformed_pixels": np.concatenate(capture["pixels"])}
            if capture.get("tokens"):
                arrays.update({"token_ids": np.concatenate(capture["tokens"]), "attention_mask": np.concatenate(capture["masks"]), "raw_text": np.asarray(capture["text"]), "processed_text": np.asarray(capture["processed_text"])})
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


if __name__ == "__main__" and "--worker-job" in __import__("sys").argv:
    _worker_main(__import__("sys").argv[__import__("sys").argv.index("--worker-job") + 1])



def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
ADAPTER = Adapter
