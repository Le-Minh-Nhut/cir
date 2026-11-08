"""Subprocess client for Air-Know's BLIP-2 composition model.

The model interpreter owns every torch import. Scoring is the native Air-Know operation:
a query is one normalized 256-d vector, a target is 32 normalized token vectors, and the
score is the maximum dot product over those tokens (``test_BLIP2.py:142-144``:
``np.matmul(all_queries, all_imgs).max(-1)``). No reference mask is applied -- the CIRCUS
full gallery keeps the reference eligible, which is Air-Know's own protocol.

Category is not a scoring axis here: one shared feature space serves all three FashionIQ
categories, so ``category_scopes`` is ``(None,)`` and target scopes must all be ``None``.
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
    AdapterStatus,
    AdapterWorkRootError,
    AdapterWorkerError,
    CompositionAdapter,
    limit_torch_threads,
    prefetch_rows,
    resolved_decode_workers,
    worker_environment,
)
from workbench.composition.inputs import CONDITIONS

#: ``extract_retrieval_target`` returns ``(B, 256, 32)`` permuted to ``(B, 32, 256)`` and
#: flattened to 8192 on disk, matching ConeSep's storage shape for the same BLIP-2 family.
TARGET_WIDTH = 8192
TOKENS = 32
DIMENSION = 256

class Adapter(CompositionAdapter):
    """Preserve Air-Know's max-over-target-tokens dot-product scoring.

    Runner assigns ``work_root`` before encoding; encode methods refuse with
    ``AdapterWorkRootError`` when it is unset. One resident worker serves every job so the
    whole-model checkpoint is unpickled once per run.
    """

    model_id = "airknow"
    reference_eligible = True
    # score() reduces a query (N,256) against targets (M,32,256); the probe uses one token.
    score_dimension = DIMENSION
    category_scopes: tuple[str | None, ...] = (None,)

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config

        self.config = config if config is not None else resolve_config()
        self._model, self._checkpoint, self._interpreter = _resolve(
            self.config, checkpoint_id)
        super().__init__(self.config, self._checkpoint["checkpoint_id"])
        self.checkpoint = self._checkpoint
        self.checkpoint_path = Path(self._checkpoint["resolved_path"])
        self.source_root = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self._model["source_dir"]
        self.work_root: Path | None = None
        self._job_root: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
        self._worker_counter = 0
        self._worker_failed: str | None = None
        self._interpreter_verified = False

    @classmethod
    def status(cls, config: Any = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config

        return _status(cls, config if config is not None else resolve_config())

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        if condition not in CONDITIONS:
            return False, f"unknown composition condition: {condition!r}"
        return True, None

    def encode_queries(self, image_paths: Sequence[str], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if not (len(image_paths) == len(texts) == len(categories)):
            raise ValueError("image_paths, texts, and categories must have equal lengths")
        return _run(self, "queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        if any(category is not None for category in categories):
            raise ValueError("Air-Know gallery is category-agnostic; target categories must all be None")
        return _run(self, "targets", image_paths, (), categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """max over the 32 target tokens of ``query . token``, as ``test_BLIP2.py:142-144``.

        Accepts the probe's flat 2-D targets (one token) as well as the real 8192-wide rows,
        so the base class can MEASURE reference eligibility offline instead of trusting the
        class literal.
        """
        del categories
        query = np.asarray(queries, dtype=np.float32)
        target = np.asarray(targets, dtype=np.float32)
        if query.ndim == 3:
            query = query[:, 0, :]
        if target.ndim == 2 and target.shape[1] == TARGET_WIDTH:
            target = target.reshape(len(target), TOKENS, DIMENSION)
        elif target.ndim == 2:
            target = target[:, None, :]
        if query.ndim != 2 or target.ndim != 3:
            raise ValueError("Air-Know expects query (N,256) and target (M,8192) or (M,32,256)")
        if query.shape[1] != target.shape[2]:
            raise ValueError("Air-Know query and target embedding dimensions differ")
        return np.einsum("nd,mgd->nmg", query, target).max(axis=-1)

    def close(self) -> None:
        _stop_worker(self)
        if self._job_root is not None:
            shutil.rmtree(self._job_root, ignore_errors=True)
            self._job_root = None

    def restart(self) -> None:
        self.close()

def _stop_worker(adapter: Adapter) -> None:
    worker = adapter._worker
    if worker is not None:
        if worker.stdin is not None:
            try:
                worker.stdin.write(json.dumps({"stop": True}) + "\n")
                worker.stdin.flush()
            except (BrokenPipeError, OSError):
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

def _resolve(config: Any, checkpoint_id: str | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("airknow")
    checkpoint = checkpoint_by_id(
        model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path("airknow", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    if adapter.work_root is None:
        raise AdapterWorkRootError("airknow work_root was not set by runner")
    width = DIMENSION if operation == "queries" else TARGET_WIDTH
    if not image_paths:
        return np.empty((0, width), dtype=np.float32)
    if adapter._worker_failed is not None:
        raise AdapterWorkerError(adapter._worker_failed)
    interpreter = adapter._interpreter
    if not interpreter:
        raise RuntimeError("no interpreter environment resolved for WORKBENCH_AIRKNOW_PYTHON")
    checkpoint = adapter._checkpoint
    if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        raise ValueError(
            f"Air-Know checkpoint mapping is not verified: {checkpoint['checkpoint_id']} "
            f"({checkpoint.get('checkpoint_mapping_status')})")
    if not adapter.checkpoint_path.is_file():
        raise FileNotFoundError(f"Air-Know checkpoint not staged at {adapter.checkpoint_path}")
    if not adapter._interpreter_verified:
        adapter.require_verified_interpreter()
        adapter._interpreter_verified = True
    work_root = Path(adapter.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    if adapter._job_root is None:
        adapter._job_root = work_root / f".airknow-worker-{uuid.uuid4().hex}"
        adapter._job_root.mkdir()
    work = adapter._job_root
    adapter._worker_counter += 1
    job_path = work / f"job-{adapter._worker_counter}.json"
    output_path = work / f"output-{adapter._worker_counter}.npy"
    scope = categories[0] if categories and len(set(categories)) == 1 else "global"
    job_path.write_text(json.dumps({
        "operation": operation,
        "image_paths": [str(Path(path).resolve()) for path in image_paths],
        "texts": list(texts), "categories": list(categories),
        "output_path": str(output_path),
        "source_root": str(adapter.source_root),
        "checkpoint_path": str(adapter.checkpoint_path),
        "checkpoint_id": checkpoint["checkpoint_id"],
        "batch_size": adapter.batch_size,
    }), encoding="utf-8")
    try:
        if adapter._worker is not None and adapter._worker.poll() is not None:
            status = adapter._worker.returncode
            adapter._worker_failed = (
                f"adapter_worker_exited: model=airknow scope={scope!r} phase={operation} exit_status={status}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        if adapter._worker is None:
            adapter._worker_log = (work / "worker.stderr.log").open("w", encoding="utf-8")
            adapter._worker = subprocess.Popen(
                [interpreter, "-m", "workbench.composition.model_airknow", "--worker-job", str(job_path)],
                cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=adapter._worker_log, text=True, bufsize=1,
                env=worker_environment())
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
            _stop_worker(adapter)
            adapter._worker_failed = (
                f"adapter_worker_exited: model=airknow scope={scope!r} phase={operation} exit_status={status}")
            raise AdapterWorkerError(adapter._worker_failed)
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            adapter._worker_failed = (
                f"adapter_worker_job_failed: model=airknow scope={scope!r} phase={operation} "
                f"error={response.get('error_type')}: {response.get('error')}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        features = np.load(output_path, allow_pickle=False)
        if features.dtype != np.float32 or features.shape != (len(image_paths), width):
            adapter._worker_failed = (
                f"adapter_worker_invalid_matrix: model=airknow scope={scope!r} phase={operation} "
                f"shape={features.shape}/{features.dtype} expected=({len(image_paths)}, {width})/float32")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        return features
    except OSError as error:
        status = adapter._worker.poll() if adapter._worker is not None else None
        _stop_worker(adapter)
        adapter._worker_failed = (
            f"adapter_worker_exited: model=airknow scope={scope!r} phase={operation} "
            f"exit_status={status}: {error}")
        raise AdapterWorkerError(adapter._worker_failed) from error
    finally:
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import (
        checkpoint_by_id, checkpoint_is_present, checkpoint_path, load_registry, model_by_id)
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id(cls.model_id, getattr(config, "registry", None) or load_registry())
    checkpoint = checkpoint_by_id(
        model, getattr(config, "checkpoint_id", None) or model["checkpoint_variants"][0]["checkpoint_id"])
    path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    variable = model["environment"]["python_env_var"]
    interpreter = resolve_model_interpreter(model, config) or setting(variable)
    reasons = []
    if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        reasons.append(f"checkpoint mapping unresolved: {checkpoint['checkpoint_id']}")
    if not checkpoint_is_present(path, checkpoint, cls.model_id, config):
        reasons.append(f"checkpoint not staged at {path}")
    if not interpreter:
        reasons.append(f"no interpreter environment resolved for {variable}")
    elif not Path(interpreter).is_file():
        reasons.append(f"model interpreter missing at {interpreter}")
    return AdapterStatus(cls.model_id, not reasons, "; ".join(reasons) or None,
                         tuple(item["checkpoint_id"] for item in model["checkpoint_variants"]),
                         {condition: (True, None) for condition in CONDITIONS},
                         "Category-agnostic single space; max over 32 target tokens, reference eligible.")

def _worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import torch

    from workbench.replay.common import load_trusted_whole_model
    from workbench.replay.airknow import KNOWN_AUTHOR_SHA256, _load_replay_modules

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    job_path = Path(parser.parse_args(argv).worker_job)

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        source = Path(job["source_root"])
        _, _, data_utils, _, (txt_processors, _) = _load_replay_modules(source)
        checkpoint = Path(job["checkpoint_path"])
        expected = KNOWN_AUTHOR_SHA256.get(job["checkpoint_id"])
        model = load_trusted_whole_model(checkpoint, expected, torch).cuda().eval()
        transform = data_utils.targetpad_transform(1.25, 224)
        limit_torch_threads()
        workers = resolved_decode_workers()
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return 1

    def run_one(current: dict[str, Any]) -> None:
        outputs = []
        try:
            with torch.no_grad():
                batch_size = max(1, int(current.get("batch_size") or 32))
                paths = current["image_paths"]
                batches = [paths[start:start + batch_size]
                           for start in range(0, len(paths), batch_size)]
                decoded = prefetch_rows(lambda chunk: torch.stack(
                    [transform(_open_image(path)) for path in chunk]), batches, workers)
                for index, images in enumerate(decoded):
                    images = images.cuda()
                    if current["operation"] == "queries":
                        first = index * batch_size
                        texts = [txt_processors["eval"](text) for text in
                                 current["texts"][first:first + len(images)]]
                        # `extract_retrieval_compose` returns (B,1,1,256).
                        features = model.extract_retrieval_compose(images, texts).reshape(len(images), DIMENSION)
                    else:
                        # `extract_retrieval_target` returns (B,256,32): column k is token k.
                        # Stored TOKEN-MAJOR as (B,32,256) so a flat 8192 row reshapes to
                        # (32,256) and `score` reduces over the token axis exactly as
                        # `test_BLIP2.py:142-144` does.
                        features = model.extract_retrieval_target(images).transpose(1, 2)
                    outputs.append(features.float().cpu().numpy())
                    del images
            result = np.concatenate(outputs, axis=0) if outputs else np.empty((0,), dtype=np.float32)
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

def _open_image(path: str) -> Any:
    from PIL import Image
    with Image.open(path) as image:
        return image.convert("RGB")

if __name__ == "__main__":
    raise SystemExit(_worker_main())

ADAPTER = Adapter
