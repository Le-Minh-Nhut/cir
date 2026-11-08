"""Subprocess client for CSMCIR's BLIP-2 Q-Former composition model.

Controller stays torch-free. `reference_eligible=True` keeps references eligible;
`category_scopes=(None,)` follows the evaluator's shared checkpoint across FashionIQ categories
(`workbench/third_party/CSMCIR/src/validate_blip_csmcir.py:594-614`).

The adapter is not the path this model is measured on. Its scoring contract is a 256-d text embedding
expanded x32 into an 8192-d vector plus a max-over-query-tokens dot product, whereas the native
evaluator scores with `inference()`: attention-weighted token importance, then a max over 32 query
tokens of a 1x1x256 matmul. Those are different quantities, so routing the measured result through
this adapter would be a substitution, not a reuse. The native evaluator path is what produced
R@10 57.0507 / R@50 77.2986 (CHECKPOINT_SELECTION_REPORT.md 3.18).
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
# The previous reason said the `COT_ours2` captions were absent and that local `qwen_captions` were a
# different, unsubstitutable input. Both halves were measured false: the evaluation path opens
# `COT_ours2/fashioniq/{cat}_cot_val.json`, those three files are published at the pinned Hub revision
# with matching sha256, and `bert_captions`/`keywords_in_mods` -- the paths an earlier revision of the
# report demanded -- belong to `FashionIQDataset_imageedit`, which nothing instantiates. See
# CHECKPOINT_SELECTION_REPORT.md 3.18 and the call-path probe in the 20261107 evidence bundle.
_TARGET_REASON = (
    "CSMCIR cannot run through the composition adapter: the adapter's score() is a 256-d text "
    "embedding expanded x32 into an 8192-d dot product, whereas the native evaluator's inference() "
    "uses attention-weighted token importance and a max over 32 query tokens. Those are different "
    "quantities, so adapter scores would be a substitution, not the measured contract. The native "
    "evaluator path is measured (CHECKPOINT_SELECTION_REPORT.md 3.18); this adapter is not wired to it. "
    "Additionally, local captions cover only the FashionIQ validation split (15,415 of 74,381 CIRCUS "
    "gallery images, or 20.72%), leaving 58,966 images uncaptioned."
)


class Adapter(CompositionAdapter):
    """Run CSMCIR in its model-specific interpreter; never import torch in controller."""

    model_id = "csmcir"
    reference_eligible = True
    # score() reads a 32x256 token layout as an 8192-wide vector.
    score_dimension = 8192
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
        self._interpreter_verified = False

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
        _cleanup(self._job_root, self.work_root)
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
    if not getattr(adapter, "_interpreter_verified", False):
        # Same gate, same code path, as every other adapter: see
        # CompositionAdapter.require_verified_interpreter.
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
                  _TARGET_REASON)


def _cleanup(path: Path | None, work_root: Path | None = None) -> None:
    if path is not None and path.exists():
        if work_root is not None and not path.resolve().is_relative_to(work_root.resolve()):
            raise ValueError(f"unsafe cleanup target: {path}")
        shutil.rmtree(path)

def _format_keys(keys: Sequence[str], limit: int = 5) -> str:
    if len(keys) <= limit:
        return ", ".join(repr(k) for k in keys)
    return f"{', '.join(repr(k) for k in keys[:limit])} (and {len(keys) - limit} more)"


def _load_csmcir_checkpoint(model: Any, state: Any) -> None:
    report = model.load_state_dict(state, strict=False)
    _check_checkpoint_load(report)


def _check_checkpoint_load(report: Any) -> None:
    missing = list(getattr(report, "missing_keys", ()))
    unexpected = list(getattr(report, "unexpected_keys", ()))
    if unexpected:
        print(
            f"CSMCIR checkpoint load: {len(unexpected)} unexpected keys tolerated: "
            f"{_format_keys(unexpected)}",
            file=sys.stderr,
        )
    if missing:
        unexpected_part = (
            f", {len(unexpected)} unexpected keys [{_format_keys(unexpected)}]"
            if unexpected
            else f", {len(unexpected)} unexpected keys"
        )
        raise ValueError(
            "CSMCIR checkpoint does not match the model: "
            f"{len(missing)} missing [{_format_keys(missing)}]"
            f"{unexpected_part}"
        )


def _worker_main(job_path: str) -> None:
    import torch
    from PIL import Image
    from workbench.replay.lavis_processors import prepare

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    job = json.loads(Path(job_path).read_text(encoding="utf-8"))
    source = Path(job["source_root"]) / "src"
    prepare(source)
    from lavis.models import load_model_and_preprocess
    from data_utils_csmcir import targetpad_transform

    try:
        model, _, processors = load_model_and_preprocess(
            name="blip2_cir_align_prompt_csmcir", model_type="pretrain", is_eval=True, device="cuda")
        state = torch.load(job["checkpoint_path"], map_location="cuda", weights_only=False)
        _load_csmcir_checkpoint(model, state[model.__class__.__name__])
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return
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
