"""Subprocess CIRCUS adapter for PAIR's FashionIQ composition space.

NOT A VERIFIED RESULT. Both registered PAIR variants carry
``checkpoint_mapping_status: UNVERIFIED`` and this adapter refuses to encode while that is
true: the released pickles were produced by an implementation the pinned repository does not
contain (``model_vssp.{Encoder,Backbone,FeatureWiseAffine}`` and
``MHTransformer_dis.{...}`` — neither module exists in ``third_party/PAIR/models``; rebuilt with
those names bound, ``PAIR()`` loads non-strictly at 455 checkpoint keys vs 381 model keys, 74
unexpected, 70 shape mismatches, 5.20 % of parameters unreachable). A number produced from those
bytes could not be labelled a PAIR reproduction, so this adapter exists to make that blocker
machine-readable, not to produce one. See ``workbench/registry/models.yaml`` (pair) for the full
measurement.

Native scoring reproduced, not invented (``third_party/PAIR/test.py``):

* ``:57-59`` — every query row and every target row is divided by its L2 norm;
* ``:60`` — ``sims = all_queries.dot(all_imgs.T)``: a plain cosine matrix, no max over tokens
  (``extract_retrieval_compose``/``extract_retrieval_target`` already mean-pool and normalize
  inside ``models/model.py:223-234``, so the vectors reaching the matmul are ``(B, 1024)``);
* ``:62-64`` — upstream then masks the reference row (``sims[i, ...] = -10e10``). That mask is
  **deliberately not inherited**: CIRCUS requires a reference-eligible full gallery, and the same
  choice is already made and documented by ``model_encoder.py`` for the identical upstream
  pattern. ``reference_eligible`` is therefore declared ``True`` and is MEASURED at runtime by
  the base class's score+rank probe.

Query transform caveat, stated rather than hidden: upstream's dataset applies
``preprocess_train`` (``RandomResizedCrop``) to query images, so the query population is
stochastic across constructions; ``replay/pair.py`` pins that with
``seeded_dataset_construction``. A raw-path adapter has no dataset object to seed, so this
adapter applies the deterministic ``preprocess_val`` transform to both sides. That is a
DOCUMENTED DIVERGENCE from upstream's stochastic query transform and is part of the same
mapping work that must land before any PAIR number is publishable.
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

#: ``models/model.py:210-216`` mean-pools ``(B, tokens, 1024)`` to ``(B, 1024)`` and L2-normalizes.
DIMENSION = 1024

class Adapter(CompositionAdapter):
    """PAIR query/target encoders in PAIR's own interpreter; cosine scoring, no reference mask."""

    model_id = "pair"
    reference_eligible = True
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
            raise ValueError("PAIR gallery is category-agnostic; target categories must all be None")
        return _run(self, "targets", image_paths, (), categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Cosine matrix: ``test.py:57-60`` normalize both sides, then one dot product.

        The upstream reference mask (``test.py:62-64``) is intentionally absent — see the module
        docstring. Chunked over the gallery so a 74k-row full gallery never materialises a second
        float32 copy of the target matrix.
        """
        del categories
        query_values = np.asarray(queries, dtype=np.float32)
        target_values = np.asarray(targets, dtype=np.float32)
        if query_values.ndim != 2 or target_values.ndim != 2:
            raise ValueError("PAIR expects query (N,1024) and target (M,1024) matrices")
        if query_values.shape[1] != target_values.shape[1]:
            raise ValueError("PAIR query and target embedding dimensions differ")
        tiny = np.finfo(np.float32).tiny
        query_values = query_values / np.maximum(
            np.linalg.norm(query_values, axis=1, keepdims=True), tiny)
        scores = np.empty((len(query_values), len(target_values)), dtype=np.float32)
        for start in range(0, len(target_values), 8192):
            stop = min(start + 8192, len(target_values))
            gallery = target_values[start:stop]
            gallery = gallery / np.maximum(
                np.linalg.norm(gallery, axis=1, keepdims=True), tiny)
            scores[:, start:stop] = query_values @ gallery.T
        return scores

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

    model = model_by_id("pair")
    checkpoint = checkpoint_by_id(
        model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path("pair", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)

def _mapping_refusal(checkpoint: dict[str, Any]) -> str | None:
    """The reason this variant may not be scored, or None.

    Shared by ``status()`` and ``_run()`` so a plan and a launch cannot disagree -- the same
    single-contract rule ``_environment_gate_reason`` exists for.
    """
    status = checkpoint.get("checkpoint_mapping_status")
    if status != "VERIFIED_METADATA":
        return (f"checkpoint mapping is {status} for {checkpoint['checkpoint_id']}: the released "
                "pickle names classes the pinned repository does not contain, so no number from "
                "these bytes can be labelled a PAIR reproduction")
    return None

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    if adapter.work_root is None:
        raise AdapterWorkRootError("pair work_root was not set by runner")
    if not image_paths:
        return np.empty((0, DIMENSION), dtype=np.float32)
    refusal = _mapping_refusal(adapter._checkpoint)
    if refusal is not None:
        raise ValueError(refusal)
    if adapter._worker_failed is not None:
        raise AdapterWorkerError(adapter._worker_failed)
    interpreter = adapter._interpreter
    if not interpreter:
        raise RuntimeError("no interpreter environment resolved for WORKBENCH_PAIR_PYTHON")
    if not adapter.checkpoint_path.is_file():
        raise FileNotFoundError(f"PAIR checkpoint not staged at {adapter.checkpoint_path}")
    if not adapter._interpreter_verified:
        adapter.require_verified_interpreter()
        adapter._interpreter_verified = True
    work_root = Path(adapter.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    if adapter._job_root is None:
        adapter._job_root = work_root / f".pair-worker-{uuid.uuid4().hex}"
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
        "checkpoint_id": adapter._checkpoint["checkpoint_id"],
        "expected_sha256": adapter._checkpoint.get("expected_sha256"),
        "batch_size": adapter.batch_size,
    }), encoding="utf-8")
    try:
        if adapter._worker is not None and adapter._worker.poll() is not None:
            status = adapter._worker.returncode
            adapter._worker_failed = (
                f"adapter_worker_exited: model=pair scope={scope!r} phase={operation} exit_status={status}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        if adapter._worker is None:
            adapter._worker_log = (work / "worker.stderr.log").open("w", encoding="utf-8")
            adapter._worker = subprocess.Popen(
                [interpreter, "-m", "workbench.composition.model_pair", "--worker-job", str(job_path)],
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
                f"adapter_worker_exited: model=pair scope={scope!r} phase={operation} exit_status={status}")
            raise AdapterWorkerError(adapter._worker_failed)
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            adapter._worker_failed = (
                f"adapter_worker_job_failed: model=pair scope={scope!r} phase={operation} "
                f"error={response.get('error_type')}: {response.get('error')}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        features = np.load(output_path, allow_pickle=False)
        if features.dtype != np.float32 or features.shape != (len(image_paths), DIMENSION):
            adapter._worker_failed = (
                f"adapter_worker_invalid_matrix: model=pair scope={scope!r} phase={operation} "
                f"shape={features.shape}/{features.dtype} expected=({len(image_paths)}, {DIMENSION})/float32")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        return features
    except OSError as error:
        status = adapter._worker.poll() if adapter._worker is not None else None
        _stop_worker(adapter)
        adapter._worker_failed = (
            f"adapter_worker_exited: model=pair scope={scope!r} phase={operation} "
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
    refusal = _mapping_refusal(checkpoint)
    if refusal is not None:
        reasons.append(refusal)
    if not checkpoint_is_present(path, checkpoint, cls.model_id, config):
        reasons.append(f"checkpoint not staged at {path}")
    if not interpreter:
        reasons.append(f"no interpreter environment resolved for {variable}")
    elif not Path(interpreter).is_file():
        reasons.append(f"model interpreter missing at {interpreter}")
    return AdapterStatus(cls.model_id, not reasons, "; ".join(reasons) or None,
                         tuple(item["checkpoint_id"] for item in model["checkpoint_variants"]),
                         {condition: (True, None) for condition in CONDITIONS},
                         "Cosine over mean-pooled 1024-d vectors; upstream reference mask NOT "
                         "inherited (CIRCUS needs a reference-eligible gallery); query transform "
                         "is the deterministic val transform, a documented divergence from "
                         "upstream's stochastic train transform.")

def _worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import torch

    from workbench.replay.common import load_trusted_whole_model
    from workbench.replay.pair import _load_replay_modules

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    job_path = Path(parser.parse_args(argv).worker_job)

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        source = Path(job["source_root"])
        _, open_clip, _, _ = _load_replay_modules(source)
        checkpoint = Path(job["checkpoint_path"])
        # `load_trusted_whole_model` unpickles only after the author digest matches. It will
        # still fail on the missing `model_vssp` / `MHTransformer_dis` classes, which is exactly
        # why `_mapping_refusal` refuses first.
        model = load_trusted_whole_model(checkpoint, job.get("expected_sha256"), torch)
        model = model.cuda().eval()
        # Transform-only use: no pretrained weights are requested here.
        _, _, preprocess_val = open_clip.create_model_and_transforms("ViT-B-32")
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
                    [preprocess_val(_open_image(path)) for path in chunk]), batches, workers)
                for index, images in enumerate(decoded):
                    images = images.cuda()
                    if current["operation"] == "queries":
                        first = index * batch_size
                        texts = list(current["texts"][first:first + len(images)])
                        features = model.extract_retrieval_compose(images, texts)
                    else:
                        features = model.extract_retrieval_target(images)
                    outputs.append(features.float().cpu().numpy().reshape(len(images), -1))
                    del images
            result = np.concatenate(outputs, axis=0) if outputs else np.empty((0, DIMENSION), dtype=np.float32)
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
