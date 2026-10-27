"""Subprocess client for ConeSep's BLIP-2 composition model.

The model interpreter owns all torch imports. Category validation runs one model/checkpoint and one
feature space over all FashionIQ categories; references stay eligible in the CIRCUS full gallery.
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


class Adapter(CompositionAdapter):
    """Preserve ConeSep's Q-Former query and max target-token dot-product scoring.

    Runner assigns ``work_root: Path | None`` before encoding; reference stays eligible in the
    full CIRCUS gallery. Category selection uses one shared target feature space.
    """

    model_id = "conesep"
    reference_eligible = True
    category_scopes: tuple[str | None, ...] = (None,)

    def __init__(self, config: Any, checkpoint_id: str | None = None) -> None:
        if CompositionAdapter is not object:
            super().__init__(config, checkpoint_id)
        self._model, self._checkpoint, self._interpreter = _resolve(
            config, checkpoint_id or getattr(self, "checkpoint_id", None))
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
        if query.ndim != 2 or target.ndim != 2 or query.shape[1] != 8192 or target.shape[1] != 8192:
            raise ValueError("expected expanded query and target vectors (N,8192)/(M,8192)")
        query = query.reshape(len(query), 32, 256)[:, 0, :]
        target = target.reshape(len(target), 32, 256).transpose(0, 2, 1)
        return np.einsum("nd,mdk->nmk", query, target, optimize=True).max(axis=-1)

    def close(self) -> None:
        if self._job_root is not None and self._job_root.exists():
            shutil.rmtree(self._job_root)
        self._job_root = None


def _check_work_root(adapter: Adapter) -> Path:
    if adapter.work_root is None:
        raise AdapterWorkRootError("conesep work_root was not set by runner")
    return adapter.work_root



def _resolve(config: Any, checkpoint_id: str | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("conesep")
    checkpoint = checkpoint_by_id(model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path("conesep", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)


def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    work_root = _check_work_root(adapter)
    if not image_paths:
        return np.empty((0, 8192), dtype=np.float32)
    interpreter = adapter._interpreter
    if not interpreter:
        raise RuntimeError("no interpreter environment resolved for WORKBENCH_CONESEP_PYTHON")
    checkpoint = adapter._checkpoint
    checkpoint_path = Path(checkpoint["resolved_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"ConeSep checkpoint not staged at {checkpoint_path}")
    if checkpoint.get("checkpoint_training_noise_pct") not in {20, 50, 80}:
        raise ValueError(f"ConeSep checkpoint condition is not an author-released variant: {checkpoint['checkpoint_id']}")
    adapter_root = work_root / "conesep"
    adapter._job_root = adapter_root
    adapter_root.mkdir(parents=True, exist_ok=True)
    job_root = adapter_root / uuid.uuid4().hex
    job_root.mkdir()
    output, job = job_root / "features.npy", job_root / "job.json"
    source = Path(adapter.config.WORKBENCH_THIRD_PARTY_ROOT) / adapter._model["source_dir"]
    job.write_text(json.dumps({"operation": operation, "image_paths": list(map(str, image_paths)),
                              "texts": list(texts), "categories": list(categories),
                              "output": str(output), "source_root": str(source),
                              "checkpoint_path": str(checkpoint_path),
                              "checkpoint_id": checkpoint["checkpoint_id"]}), encoding="utf-8")
    scope = categories[0] if categories and len(set(categories)) == 1 else "mixed"
    try:
        result = subprocess.run([interpreter, "-m", "workbench.composition.model_conesep",
                                 "--worker-job", str(job)],
                                cwd=Path(__file__).resolve().parents[2], check=False)
        if result.returncode:
            raise AdapterWorkerError(f"conesep worker scope {scope!r} exited with status {result.returncode}")
        if not output.is_file():
            raise AdapterWorkerError(f"conesep worker scope {scope!r} exited without feature matrix")
        features = np.load(output, allow_pickle=False).copy()
        if features.dtype != np.float32 or features.shape != (len(image_paths), 8192):
            raise AdapterWorkerError(f"conesep worker scope {scope!r} returned invalid matrix {features.shape}/{features.dtype}")
        return features
    finally:
        shutil.rmtree(job_root, ignore_errors=True)


def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_is_present, checkpoint_path, load_registry, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter
    from workbench.composition.adapters import AdapterStatus as Status

    model = model_by_id(cls.model_id, getattr(config, "registry", None) or load_registry())
    checkpoint = checkpoint_by_id(model, getattr(config, "checkpoint_id", None) or model["checkpoint_variants"][0]["checkpoint_id"])
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
    return Status(cls.model_id, reason is None, reason,
                  tuple(item["checkpoint_id"] for item in model["checkpoint_variants"]),
                  {condition: (True, None) for condition in CONDITIONS},
                  "Category-agnostic single space; transfer/evaluation checkpoint follows selected registry variant.")


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

    from workbench.replay.common import load_trusted_whole_model
    from workbench.replay.conesep import KNOWN_AUTHOR_SHA256, PROCESSOR_CONFIGS, _load_replay_modules

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    source = Path(job["source_root"])
    _, _, data_utils, _, (txt_processors, _) = _load_replay_modules(source)
    checkpoint = Path(job["checkpoint_path"])
    expected = KNOWN_AUTHOR_SHA256.get(job["checkpoint_id"])
    model = load_trusted_whole_model(checkpoint, expected, torch).cuda().eval()
    transform = data_utils.targetpad_transform(1.25, 224)
    batch_size = 32
    outputs = []
    with torch.no_grad():
        for start in range(0, len(job["image_paths"]), batch_size):
            images = torch.stack([transform(_open_image(path)) for path in job["image_paths"][start:start + batch_size]]).cuda()
            if job["operation"] == "queries":
                texts = [txt_processors["eval"](text) for text in job["texts"][start:start + batch_size]]
                features = model.extract_retrieval_compose(images, texts)[:, 0, 0, :].repeat(1, 32)
            else:
                features = model.extract_retrieval_target(images).permute(0, 2, 1)
            outputs.append(features.float().cpu().numpy())
    dimension = 8192
    result = np.concatenate(outputs, axis=0) if outputs else np.empty((0, dimension), dtype=np.float32)
    np.save(job["output"], result.reshape(len(result), -1))


def _open_image(path: str) -> Any:
    from PIL import Image
    with Image.open(path) as image:
        return image.convert("RGB")


if __name__ == "__main__" and "--worker-job" in sys.argv:
    _worker_main(sys.argv[sys.argv.index("--worker-job") + 1])


ADAPTER = Adapter
