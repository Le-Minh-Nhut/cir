"""Subprocess client for the released PTHA+MTST CIRR-to-FashionIQ transfer checkpoint.

The model interpreter owns torch and its pinned dependencies; this controller module stays
importable in the workbench Python. The selected `cirr_transfer` weights are not a FashionIQ
reproduction. `reference_eligible=True` retains references in the full CIRCUS gallery.
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
                                                     AdapterWorkerError, CompositionAdapter)
    except ModuleNotFoundError:
        CompositionAdapter = object  # type: ignore[assignment,misc]
        AdapterStatus = Any  # type: ignore[misc,assignment]
        class AdapterWorkRootError(RuntimeError):
            pass
        class AdapterWorkerError(RuntimeError):
            pass

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
_DEFAULT_CHECKPOINT = "cirr_transfer"


class Adapter(CompositionAdapter):
    """Use native PTHA+MTST query fusion and max cosine over 32 target tokens.

    `category_scopes=(None,)`: one model/checkpoint processes all categories; evaluator
    `validate_blip.py:596-614` iterates categories with the same loaded model. This is the CIRR →
    FashionIQ cross-dataset transfer checkpoint, not the paper's FashionIQ reproduction. Model code
    runs only in the configured PTHA interpreter.
    """

    model_id = "ptha_mtst"
    reference_eligible = True
    category_scopes: tuple[str | None, ...] = (None,)

    def __init__(self, config: Any, checkpoint_id: str | None = None) -> None:
        if CompositionAdapter is not object:
            super().__init__(config, checkpoint_id)
        else:
            self.config = config
            self.checkpoint_id = checkpoint_id or _DEFAULT_CHECKPOINT
        self._model, self._checkpoint, self._interpreter = _resolve(
            config, checkpoint_id or getattr(self, "checkpoint_id", None) or _DEFAULT_CHECKPOINT)
        self.work_root: Path | None = None
        self._job_root: Path | None = None

    @classmethod
    def status(cls, config: Any | None = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config
        return _status(cls, config or resolve_config())

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        if condition not in CONDITIONS:
            return False, f"unknown composition condition: {condition!r}"
        return True, None

    def encode_queries(self, image_paths: Sequence[str], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if not (len(image_paths) == len(texts) == len(categories)):
            raise ValueError("image_paths, texts, and categories must have equal lengths")
        return _run(self, "queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str], categories: Sequence[str | None]) -> np.ndarray:
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        return _run(self, "targets", image_paths, (), categories)
    def score(self, queries: Any, targets: Any, categories: Sequence[str | None] | None = None) -> np.ndarray:
        del categories
        query, target = np.asarray(queries, dtype=np.float32), np.asarray(targets, dtype=np.float32)
        if query.ndim != 2 or query.shape[1] != 32 * 256 or target.ndim != 2 or target.shape[1] != 32 * 256:
            raise ValueError("expected expanded normalized query and projected target features (N,8192)/(M,8192)")
        query = query.reshape(len(query), 32, 256)[:, 0, :]
        target_tokens = target.reshape(len(target), 32, 256)
        return np.einsum("nd,mkd->nmk", query, target_tokens, optimize=True).max(axis=-1)

    def rank(self, queries: Any, targets: Any, target_positions: Sequence[Sequence[int] | None] | None = None,
             categories: Sequence[str | None] | None = None) -> list[int | None]:
        return super().rank(queries, targets, target_positions, categories)

    def close(self) -> None:
        if self._job_root is not None and self._job_root.exists():
            shutil.rmtree(self._job_root)
        self._job_root = None


def _check_work_root(adapter: Adapter) -> Path:
    if adapter.work_root is None:
        raise AdapterWorkRootError("ptha_mtst work_root was not set by runner")
    return adapter.work_root
def _resolve(config: Any, checkpoint_id: str | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("ptha_mtst")
    checkpoint = checkpoint_by_id(model, checkpoint_id or _DEFAULT_CHECKPOINT)
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path("ptha_mtst", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)


def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    work_root = _check_work_root(adapter)
    if not image_paths:
        return np.empty((0, 8192), dtype=np.float32)
    interpreter = adapter._interpreter
    if not interpreter:
        raise RuntimeError("no interpreter environment resolved for WORKBENCH_PTHA_MTST_PYTHON")
    checkpoint_path = Path(adapter._checkpoint["resolved_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PTHA+MTST checkpoint not staged at {checkpoint_path}")
    source = Path(adapter.config.WORKBENCH_THIRD_PARTY_ROOT) / adapter._model["source_dir"]
    adapter_root = work_root / "ptha_mtst"
    adapter._job_root = adapter_root
    adapter_root.mkdir(parents=True, exist_ok=True)
    job_root = adapter_root / uuid.uuid4().hex
    job_root.mkdir()
    output, job = job_root / "features.npy", job_root / "job.json"
    job.write_text(json.dumps({"operation": operation, "image_paths": list(map(str, image_paths)),
                               "texts": list(texts), "categories": list(categories),
                               "output": str(output), "source_root": str(source),
                               "checkpoint_path": str(checkpoint_path)}), encoding="utf-8")
    try:
        result = subprocess.run([interpreter, "-m", "workbench.composition.model_ptha_mtst",
                                 "--worker-job", str(job)],
                                cwd=Path(__file__).resolve().parents[2], check=False)
        if result.returncode:
            raise AdapterWorkerError(f"ptha_mtst worker scope None exited with status {result.returncode}")
        if not output.is_file():
            raise AdapterWorkerError("ptha_mtst worker scope None exited without feature matrix")
        features = np.load(output, allow_pickle=False).copy()
        if features.dtype != np.float32 or features.shape != (len(image_paths), 8192):
            raise AdapterWorkerError(f"ptha_mtst worker scope None returned invalid matrix {features.shape}/{features.dtype}")
        return features
    finally:
        shutil.rmtree(job_root, ignore_errors=True)


def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_is_present, checkpoint_path, load_registry, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter
    from workbench.composition.adapters import AdapterStatus as Status

    model = model_by_id(cls.model_id, getattr(config, "registry", None) or load_registry())
    checkpoint_id = getattr(config, "checkpoint_id", None) or _DEFAULT_CHECKPOINT
    checkpoint = checkpoint_by_id(model, checkpoint_id)
    path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    variable = model["environment"]["python_env_var"]
    interpreter = resolve_model_interpreter(model, config) or setting(variable)
    reason = None
    if not interpreter:
        reason = f"no interpreter environment resolved for {variable}"
    elif not Path(interpreter).is_file():
        reason = f"model interpreter missing at {interpreter}"
    elif not checkpoint_is_present(path, checkpoint, cls.model_id, config):
        reason = f"checkpoint not staged at {path}"
    else:
        reason = _cuda_reason(interpreter)
    notes = "cirr_transfer is a CIRR-trained → FashionIQ transfer checkpoint, not a FashionIQ reproduction"
    return Status(cls.model_id, reason is None, reason,
                  tuple(item["checkpoint_id"] for item in model["checkpoint_variants"]),
                  {condition: (True, None) for condition in CONDITIONS}, notes)


def _cuda_reason(interpreter: str) -> str | None:
    probe = subprocess.run([interpreter, "-c", "import torch; print(torch.cuda.is_available())"],
                           capture_output=True, text=True, timeout=30, check=False)
    if probe.returncode:
        return f"model interpreter torch probe failed: {probe.stderr.strip()}"
    if probe.stdout.strip().splitlines()[-1:] != ["True"]:
        return "CUDA device unavailable in model interpreter"
    return None


def _worker_main(job_path: str) -> None:
    import torch

    from workbench.replay.ptha_mtst import source_tree

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    source = source_tree(Path(job["source_root"]))
    src = source / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    payload = torch.load(job["checkpoint_path"], map_location="cpu", weights_only=True)
    from omegaconf import OmegaConf
    from lavis.common.registry import registry
    from lavis.models import load_preprocess
    checkpoint_models = {
        "Blip2QformerCirCat": ("blip2_cir_cat", "pretrain"),
        "Blip2Qformer4TGIRCESIM": ("blip2_tgir_ce_sim", "pretrain_mtst"),
    }
    key = next((name for name in checkpoint_models if name in payload), None)
    if key is None:
        raise ValueError(f"unexpected PTHA+MTST checkpoint keys: {sorted(payload)}")
    model_name, model_type = checkpoint_models[key]
    model_class = registry.get_model_class(model_name)
    cfg = OmegaConf.load(model_class.default_config_path(model_type))
    cfg.model.load_pretrained = False
    model = model_class.from_config(cfg.model)
    model.load_state_dict(payload[key], strict=True)
    model = model.cuda().eval()
    _visual_processors, processors = load_preprocess(cfg.preprocess)
    from data_utils import targetpad_transform
    transform = targetpad_transform(1.25, 224)
    from PIL import Image

    outputs = []
    with torch.no_grad():
        for start in range(0, len(job["image_paths"]), 32):
            paths = job["image_paths"][start:start + 32]
            images = []
            for path in paths:
                with Image.open(path) as image:
                    images.append(transform(image.convert("RGB")))
            batch = torch.stack(images).cuda()
            if job["operation"] == "queries":
                visual = model.ln_vision(model.visual_encoder(batch)).float()
                text = [processors["eval"](value) for value in job["texts"][start:start + 32]]
                features = model.inference_reference(visual, text)
                if model_name == "blip2_cir_cat":
                    features = features / model.temp
                features = features.repeat(1, 32)
            else:
                extracted = model.extract_target_features(batch, mode="mean", target_only=True).to("cuda")
                features = model.inference_target(extracted).flatten(1)
            outputs.append(features.float().cpu().numpy())
    result = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 8192), dtype=np.float32)
    np.save(job["output"], result.reshape(len(result), -1))


if __name__ == "__main__" and "--worker-job" in sys.argv:
    _worker_main(sys.argv[sys.argv.index("--worker-job") + 1])


ADAPTER = Adapter
