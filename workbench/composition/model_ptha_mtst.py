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
from typing import Any, Sequence

import numpy as np

from workbench.composition.adapters import (
    AdapterStatus, AdapterWorkRootError, AdapterWorkerError, CompositionAdapter,
)

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
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
        self._worker_counter = 0
        self._worker_failed: str | None = None

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
        if self._worker is not None:
            if self._worker.stdin is not None:
                try:
                    self._worker.stdin.write(json.dumps({"stop": True}) + "\n")
                    self._worker.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._worker.terminate()
                try:
                    self._worker.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._worker.kill()
                    self._worker.wait()
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None
        if self._job_root is not None:
            shutil.rmtree(self._job_root, ignore_errors=True)
            self._job_root = None

    def restart(self) -> None:
        self.close()


def _kill_worker(adapter: Adapter) -> None:
    worker = adapter._worker
    if worker is not None:
        if worker.stdin is not None:
            worker.stdin.close()
        if worker.poll() is None:
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
    if adapter._worker_failed is not None:
        raise AdapterWorkerError(adapter._worker_failed)
    interpreter = adapter._interpreter
    if not interpreter:
        raise RuntimeError("no interpreter environment resolved for WORKBENCH_PTHA_MTST_PYTHON")
    checkpoint_path = Path(adapter._checkpoint["resolved_path"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"PTHA+MTST checkpoint not staged at {checkpoint_path}")
    source = Path(adapter.config.WORKBENCH_THIRD_PARTY_ROOT) / adapter._model["source_dir"]
    if adapter._job_root is None:
        work_root.mkdir(parents=True, exist_ok=True)
        adapter._job_root = work_root / f".ptha_mtst-worker-{uuid.uuid4().hex}"
        adapter._job_root.mkdir()
    work = adapter._job_root
    adapter._worker_counter += 1
    job_path, output_path = (work / f"job-{adapter._worker_counter}.json",
                             work / f"output-{adapter._worker_counter}.npy")
    job_path.write_text(json.dumps({"operation": operation,
                                   "image_paths": [str(Path(path).resolve()) for path in image_paths],
                                   "texts": list(texts), "categories": list(categories),
                                   "output_path": str(output_path), "source_root": str(source),
                                   "checkpoint_path": str(checkpoint_path)}), encoding="utf-8")
    try:
        if adapter._worker is not None and adapter._worker.poll() is not None:
            status = adapter._worker.returncode
            adapter._worker_failed = (
                f"adapter_worker_exited: model=ptha_mtst scope=None phase={operation} exit_status={status}")
            _kill_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        if adapter._worker is None:
            adapter._worker_log = (work / "worker.stderr.log").open("w", encoding="utf-8")
            adapter._worker = subprocess.Popen(
                [interpreter, "-m", "workbench.composition.model_ptha_mtst", "--worker-job", str(job_path)],
                cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=adapter._worker_log, text=True, bufsize=1)
        else:
            assert adapter._worker.stdin is not None
            adapter._worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
            adapter._worker.stdin.flush()
        assert adapter._worker.stdout is not None
        line = adapter._worker.stdout.readline()
        while line and not line.startswith("CIRCUS_WORKER:"):
            line = adapter._worker.stdout.readline()
        if not line:
            status = adapter._worker.poll()
            _kill_worker(adapter)
            adapter._worker_failed = (
                f"adapter_worker_exited: model=ptha_mtst scope=None phase={operation} exit_status={status}")
            raise AdapterWorkerError(adapter._worker_failed)
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            adapter._worker_failed = (
                f"adapter_worker_job_failed: model=ptha_mtst scope=None phase={operation} "
                f"error={response.get('error_type')}: {response.get('error')}")
            _kill_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        features = np.load(output_path, allow_pickle=False)
        if features.dtype != np.float32 or features.shape != (len(image_paths), 8192):
            adapter._worker_failed = (
                f"adapter_worker_invalid_matrix: model=ptha_mtst scope=None phase={operation} "
                f"shape={features.shape}/{features.dtype} expected=({len(image_paths)}, 8192)/float32")
            _kill_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        return features
    except OSError as error:
        status = adapter._worker.poll() if adapter._worker is not None else None
        _kill_worker(adapter)
        adapter._worker_failed = (
            f"adapter_worker_exited: model=ptha_mtst scope=None phase={operation} "
            f"exit_status={status}: {error}")
        raise AdapterWorkerError(adapter._worker_failed) from error
    finally:
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)


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


def _worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import torch

    from workbench.replay.ptha_mtst import source_tree

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    job_path = Path(parser.parse_args(argv).worker_job)

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
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
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return 1

    def run_one(current: dict[str, Any]) -> None:
        outputs = []
        try:
            with torch.no_grad():
                for start in range(0, len(current["image_paths"]), 32):
                    paths = current["image_paths"][start:start + 32]
                    images = []
                    for path in paths:
                        with Image.open(path) as image:
                            images.append(transform(image.convert("RGB")))
                    batch = torch.stack(images).cuda()
                    if current["operation"] == "queries":
                        with model.maybe_autocast():
                            visual = model.ln_vision(model.visual_encoder(batch))
                        visual = visual.float()
                        text = [processors["eval"](value) for value in current["texts"][start:start + 32]]
                        features = model.inference_reference(visual, text)
                        if model_name == "blip2_cir_cat":
                            features = features / model.temp
                        features = features.repeat(1, 32)
                    else:
                        extracted = model.extract_target_features(batch, mode="mean", target_only=True).to("cuda")
                        features = model.inference_target(extracted).flatten(1)
                    outputs.append(features.float().cpu().numpy())
            result = np.concatenate(outputs, axis=0) if outputs else np.empty((0, 8192), dtype=np.float32)
            np.save(current["output_path"], result.reshape(len(current["image_paths"]), -1), allow_pickle=False)
            reply(True)
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))

    run_one(job)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("stop"):
            break
        try:
            run_one(json.loads(Path(request["job_path"]).read_text(encoding="utf-8")))
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(_worker_main())


ADAPTER = Adapter
