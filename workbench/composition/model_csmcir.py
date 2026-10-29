"""Subprocess client for CSMCIR's BLIP-2 Q-Former composition model.

Controller stays torch-free. Released CSMCIR gallery extraction needs per-image `COT_ours2` captions,
which the CIRCUS release does not provide. Local `qwen_captions` files are different Qwen-generated
captions, not the required model inputs, and are deliberately not substituted. `reference_eligible=True`
keeps references eligible; `category_scopes=(None,)` follows the evaluator's shared checkpoint
across FashionIQ categories (`workbench/third_party/CSMCIR/src/validate_blip_csmcir.py:594-614`).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

if TYPE_CHECKING:
    from workbench.composition.adapters import AdapterStatus, CompositionAdapter
else:
    try:
        from workbench.composition.adapters import (AdapterStatus, AdapterWorkRootError,
                                                     AdapterWorkerError, CompositionAdapter,
                                                     worker_environment)
    except ModuleNotFoundError:
        CompositionAdapter = object  # type: ignore[assignment,misc]
        AdapterStatus = Any  # type: ignore[misc,assignment]
        class AdapterWorkRootError(RuntimeError):
            pass
        class AdapterWorkerError(RuntimeError):
            pass
        def worker_environment() -> dict[str, str]:
            import os

            return os.environ.copy()

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
_TARGET_REASON = (
    "CSMCIR requires per-gallery-image COT_ours2 captions absent from released CIRCUS inputs; "
    "local qwen_captions are different Qwen-generated captions and are not substituted "
    "(workbench/third_party/CSMCIR/src/utils_csmcir.py:95-103,151-161; "
    "workbench/third_party/CSMCIR/src/lavis/models/blip2_models/"
    "blip2_qformer_cir_align_prompt_csmcir.py:498-535)"
)


class Adapter(CompositionAdapter):
    """Run CSMCIR in its model-specific interpreter; never import torch in controller."""

    model_id = "csmcir"
    reference_eligible = True
    category_scopes: tuple[str | None, ...] = (None,)

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

    @classmethod
    def status(cls, config: Any | None = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config
        return _status(cls, config or resolve_config())
    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        if condition not in CONDITIONS:
            return False, f"unknown composition condition: {condition!r}"
        return False, _TARGET_REASON


    def encode_queries(self, image_paths: Sequence[str], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        _required_work_root(self)
        _check_batch(image_paths, texts, categories)
        return _run(self, "queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str], categories: Sequence[str | None]) -> np.ndarray:
        _required_work_root(self)
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        if any(category is not None for category in categories):
            raise ValueError("csmcir category-agnostic targets require None scopes")
        raise RuntimeError(_TARGET_REASON)

    def score(self, queries: Any, targets: Any, categories: Sequence[str | None] | None = None) -> np.ndarray:
        del categories
        query, target = np.asarray(queries, dtype=np.float32), np.asarray(targets, dtype=np.float32)
        if query.ndim != 2 or target.ndim != 2 or query.shape[1] != 8192 or target.shape[1] != 8192:
            raise ValueError("expected expanded query and projected target features (N,8192)/(M,8192)")
        query = query.reshape(len(query), 32, 256)[:, 0, :]
        vectors = target.reshape(len(target), 32, 256)
        vectors = vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), np.finfo(np.float32).tiny)
        return np.einsum("nd,mkd->nmk", query, vectors, optimize=True).max(axis=-1)

    def rank(self, queries: Any, targets: Any, target_positions: Sequence[Sequence[int] | None] | None = None,
             categories: Sequence[str | None] | None = None) -> list[int | None]:
        return super().rank(queries, targets, target_positions, categories)

    def close(self) -> None:
        _cleanup(self._job_root)
        self._job_root = None


def _check_batch(image_paths: Sequence[str], texts: Sequence[str], categories: Sequence[str | None]) -> None:
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


def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    work_root = _required_work_root(adapter)
    if not image_paths:
        return np.empty((0, 8192), dtype=np.float32)
    if not adapter._interpreter:
        raise RuntimeError("no interpreter environment resolved for WORKBENCH_CSMCIR_PYTHON")
    checkpoint = Path(adapter._checkpoint["resolved_path"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"csmcir checkpoint not staged at {checkpoint}")
    job_root = work_root / f"csmcir-{adapter.checkpoint_id}"
    job_root.mkdir(parents=True, exist_ok=True)
    adapter._job_root = job_root
    token = uuid.uuid4().hex
    output, job = job_root / f"{token}.npy", job_root / f"{token}.json"
    source = Path(adapter.config.WORKBENCH_THIRD_PARTY_ROOT) / adapter._model["source_dir"]
    job.write_text(json.dumps({"operation": operation, "image_paths": list(image_paths),
                               "texts": list(texts), "categories": list(categories),
                               "output": str(output), "source_root": str(source),
                               "checkpoint_path": str(checkpoint)}), encoding="utf-8")
    scope = categories[0] if categories and len(set(categories)) == 1 else "mixed"
    result = subprocess.run([adapter._interpreter, "-m", "workbench.composition.model_csmcir",
                             "--worker-job", str(job)],
                            cwd=Path(__file__).resolve().parents[2], check=False,
                            env=worker_environment())
    if result.returncode:
        raise AdapterWorkerError(f"csmcir worker scope {scope!r} exited with status {result.returncode}")
    if not output.is_file():
        raise AdapterWorkerError(f"csmcir worker scope {scope!r} exited without feature matrix")
    features = np.load(output, allow_pickle=False).copy()
    if features.dtype != np.float32 or features.shape != (len(image_paths), 8192):
        raise AdapterWorkerError(f"csmcir worker scope {scope!r} returned invalid matrix {features.shape}/{features.dtype}")
    return features


def _required_work_root(adapter: Adapter) -> Path:
    if adapter.work_root is None:
        raise AdapterWorkRootError("csmcir work_root was not set by runner")
    return adapter.work_root


def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.registry import load_registry, model_by_id
    from workbench.composition.adapters import AdapterStatus as Status

    model = model_by_id(cls.model_id, getattr(config, "registry", None) or load_registry())
    return Status(cls.model_id, False, _TARGET_REASON,
                  tuple(str(item["checkpoint_id"]) for item in model["checkpoint_variants"]),
                  {condition: (False, _TARGET_REASON) for condition in CONDITIONS},
                  "Required COT_ours2 captions are absent from the CIRCUS release; local qwen_captions are different Qwen-generated captions and are not substituted.")


def _cleanup(path: Path | None) -> None:
    if path is not None and path.exists():
        shutil.rmtree(path)


def _worker_main(job_path: str) -> None:
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
    model.load_state_dict(state[model.__class__.__name__], strict=False)
    model.eval()
    transform = targetpad_transform(1.25, 224)
    images = []
    for path in job["image_paths"]:
        with Image.open(path) as image:
            images.append(transform(image.convert("RGB")))
    batch = torch.stack(images).cuda()
    with torch.no_grad():
        reference = model.ln_vision(model.visual_encoder(batch)).float()
        texts = [processors["eval"](text) for text in job["texts"]]
        _, features = model.inference_tsen(reference, torch.zeros((len(texts), 256, 1), device="cuda"),
                                           texts, [""] * len(texts))
    features = features.repeat(1, 32)
    np.save(job["output"], features.float().cpu().numpy())


if __name__ == "__main__" and "--worker-job" in sys.argv:
    _worker_main(sys.argv[sys.argv.index("--worker-job") + 1])


ADAPTER = Adapter
