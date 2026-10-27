"""Subprocess CIRCUS adapter for HINT's FashionIQ composition space."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapters import AdapterStatus, AdapterWorkRootError, AdapterWorkerError, CompositionAdapter

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
CHECKPOINT_ID = "fashioniq"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PROBE = (
    "import json, torch; ok=torch.cuda.is_available(); "
    "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':ok,"
    "'capability':list(torch.cuda.get_device_capability()) if ok else None}))"
)




class Adapter(CompositionAdapter):
    """Run HINT's native query/gallery forwards under its model interpreter.

    ``test.py:37-47`` processes edit text and calls ``extract_retrieval_compose``;
    ``HINT.py:401-433`` performs reference-conditioned Q-former fusion and returns a
    normalized query vector. ``HINT.py:436-455`` returns normalized gallery vectors;
    ``test.py:73-76`` scores by maximum dot over target vectors. Transform is
    ``data_utils.py:92-105`` TargetPad(1.25), bicubic resize/crop 224, RGB and CLIP
    normalization. ``replay/hint.py:171-194`` uses one checkpoint across all categories,
    so gallery scope is category-agnostic. ``fashioniq`` has no documented training-noise
    label. Reference eligible. Child runs image forwards; controller NumPy computes exact
    multi-vector max-dot scores and ranks.
    """

    model_id = "hint"
    category_scopes = (None,)
    reference_eligible = True


    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config
        from workbench.backend.registry import checkpoint_by_id, load_registry
        from workbench.backend.runtime import resolve_model_interpreter

        self.config = config if config is not None else resolve_config()
        registry = getattr(self.config, "registry", None) or load_registry()
        self.model = next(row for row in registry["models"] if row["model_id"] == self.model_id)
        selected = checkpoint_id or CHECKPOINT_ID
        if selected != CHECKPOINT_ID:
            raise ValueError(f"HINT requires checkpoint {CHECKPOINT_ID!r}, got {selected!r}")
        self.checkpoint_id = selected
        self.checkpoint = checkpoint_by_id(self.model, selected)
        super().__init__(self.config, selected)
        self.model = next(row for row in registry["models"] if row["model_id"] == self.model_id)
        self.checkpoint = checkpoint_by_id(self.model, selected)
        self.checkpoint_path = self._checkpoint_path()
        self.source_root = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self.model["source_dir"]
        self.dataset_root = Path(self.config.FASHIONIQ_ROOT)
        self.interpreter = resolve_model_interpreter(self.model, self.config)
        if not self.interpreter:
            from workbench.backend.model_settings import setting

            variable = (self.model.get("environment") or {}).get("python_env_var")
            self.interpreter = setting(variable) if variable else None
        self._owned_work_dir: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
        self._worker_counter = 0
        self._worker_failed: str | None = None
        self._interpreter_verified = False

    def _checkpoint_path(self) -> Path:
        from workbench.backend.registry import checkpoint_path

        return checkpoint_path(self.model_id, self.checkpoint, self.config.WORKBENCH_CHECKPOINT_ROOT)

    @classmethod
    def status(cls, config: Any = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config
        from workbench.backend.registry import checkpoint_by_id, checkpoint_is_present, checkpoint_path, load_registry
        from workbench.backend.model_settings import setting
        from workbench.backend.runtime import resolve_model_interpreter, runtime_adaptation

        config = config if config is not None else resolve_config()
        registry = getattr(config, "registry", None) or load_registry()
        model = next(row for row in registry["models"] if row["model_id"] == cls.model_id)
        checkpoint = checkpoint_by_id(model, CHECKPOINT_ID)
        path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
        reasons = []
        if checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
            reasons.append(f"checkpoint mapping unresolved: {CHECKPOINT_ID}")
        if not checkpoint_is_present(path, checkpoint, model_id=cls.model_id, config=config):
            reasons.append(f"checkpoint missing or invalid: {path}")
        interpreter = resolve_model_interpreter(model, config)
        if not interpreter:
            variable = (model.get("environment") or {}).get("python_env_var")
            interpreter = setting(variable) if variable else None
        if not interpreter or not Path(interpreter).is_file():
            reasons.append(f"model interpreter missing: {interpreter or 'environment not configured'}")
        else:
            try:
                adaptation = runtime_adaptation(model)
                result = subprocess.run([interpreter, "-c", WORKER_PROBE], check=True,
                                        capture_output=True, text=True, cwd=REPO_ROOT)
                probe = json.loads(result.stdout.strip().splitlines()[-1])
                if not probe["available"]:
                    reasons.append(f"CUDA unavailable in HINT interpreter {interpreter}")
                elif adaptation:
                    capability = "sm_" + "".join(map(str, probe["capability"]))
                    target = str(adaptation.get("target_compute_capability", "")).replace("_blackwell", "")
                    if target != capability:
                        reasons.append(f"runtime adaptation targets {target}, but HINT interpreter GPU is {capability}")
            except Exception as error:
                reasons.append(f"HINT interpreter/runtime probe failed: {type(error).__name__}: {error}")
        return AdapterStatus(cls.model_id, not reasons, "; ".join(reasons) or None,
                             (CHECKPOINT_ID,), {condition: (True, None) for condition in CONDITIONS})

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        return (True, None) if condition in CONDITIONS else (False, f"unknown composition condition: {condition!r}")

    def _work_dir(self) -> Path:
        if self.work_root is None:
            raise AdapterWorkRootError(f"{self.model_id} work_root must be assigned before encoding")
        if self._owned_work_dir is None:
            root = Path(self.work_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self._owned_work_dir = root / f".hint-worker-{uuid.uuid4().hex}"
            self._owned_work_dir.mkdir()
        return self._owned_work_dir

    def _run_job(self, operation: str, image_paths: Sequence[str], texts: Sequence[str] = (),
                 categories: Sequence[str | None] = ()) -> np.ndarray:
        if not self.interpreter:
            raise RuntimeError("HINT model interpreter is not configured")
        if self._worker_failed is not None:
            raise AdapterWorkerError(self._worker_failed)
        if not self._interpreter_verified:
            from workbench.backend.runtime import require_verified_model_interpreter

            verified = require_verified_model_interpreter(self.model, config=self.config)
            if Path(verified).resolve() != Path(self.interpreter).resolve():
                raise RuntimeError(f"HINT resolved interpreter changed: {self.interpreter} -> {verified}")
            self._interpreter_verified = True
        work = self._work_dir()
        self._worker_counter += 1
        job_path = work / f"job-{self._worker_counter}.json"
        output_path = work / f"output-{self._worker_counter}.npy"
        scope = tuple(categories) or (None,)
        job_path.write_text(json.dumps({
            "operation": operation, "image_paths": [str(Path(path).resolve()) for path in image_paths],
            "texts": list(texts), "categories": list(categories), "output_path": str(output_path),
            "checkpoint_path": str(self.checkpoint_path), "checkpoint": self.checkpoint,
            "source_root": str(self.source_root), "dataset_root": str(self.dataset_root),
            "batch_size": self.batch_size,
        }), encoding="utf-8")
        try:
            if self._worker is not None and self._worker.poll() is not None:
                status = self._worker.returncode
                self._worker_failed = (
                    f"adapter_worker_exited: model=hint scope={scope} phase={operation} exit_status={status}")
                self._kill_worker()
                raise AdapterWorkerError(self._worker_failed)
            if self._worker is None:
                self._worker_log = (work / "worker.stderr.log").open("w", encoding="utf-8")
                self._worker = subprocess.Popen(
                    [self.interpreter, "-m", "workbench.composition.model_hint", "--worker-job", str(job_path)],
                    cwd=REPO_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=self._worker_log, text=True, bufsize=1)
            else:
                assert self._worker.stdin is not None
                self._worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
                self._worker.stdin.flush()
            assert self._worker.stdout is not None
            line = self._worker.stdout.readline()
            while line and not line.startswith("CIRCUS_WORKER:"):
                line = self._worker.stdout.readline()
            if not line:
                status = self._worker.poll()
                self._kill_worker()
                self._worker_failed = (
                    f"adapter_worker_exited: model=hint scope={scope} phase={operation} "
                    f"exit_status={status}")
                raise AdapterWorkerError(self._worker_failed)
            response = json.loads(line[len("CIRCUS_WORKER:"):])
            if not response.get("ok"):
                self._worker_failed = (
                    f"adapter_worker_job_failed: model=hint scope={scope} phase={operation} "
                    f"error={response.get('error_type')}: {response.get('error')}")
                self._kill_worker()
                raise AdapterWorkerError(self._worker_failed)
            result = np.load(output_path, allow_pickle=False).astype(np.float32, copy=False)
            expected = ((len(image_paths), 256) if operation == "queries" else
                        (len(image_paths), 32, 256))
            if result.shape != expected:
                self._worker_failed = (
                    f"adapter_worker_invalid_matrix: model=hint scope={scope} phase={operation} "
                    f"shape={result.shape} expected={expected}")
                self._kill_worker()
                raise AdapterWorkerError(self._worker_failed)
            return result
        except OSError as error:
            self._worker_failed = (
                f"adapter_worker_exited: model=hint scope={scope} phase={operation} "
                f"exit_status={self._worker.returncode if self._worker else None}: {error}")
            self._kill_worker()
            raise AdapterWorkerError(self._worker_failed) from error
        finally:
            job_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)

    def encode_queries(self, image_paths: Sequence[str], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if self.work_root is None:
            raise AdapterWorkRootError(f"{self.model_id} work_root must be assigned before encoding")
        if len(image_paths) != len(texts) or len(image_paths) != len(categories):
            raise ValueError("HINT image paths, texts, and categories must have equal lengths")
        return self._run_job("queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str], categories: Sequence[str | None]) -> np.ndarray:
        if self.work_root is None:
            raise AdapterWorkRootError(f"{self.model_id} work_root must be assigned before encoding")
        if len(image_paths) != len(categories):
            raise ValueError("HINT target paths and categories must have equal lengths")
        if any(category is not None for category in categories):
            raise ValueError("HINT gallery is category-agnostic; target categories must all be None")
        return self._run_job("targets", image_paths, categories=categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        query_values = np.asarray(queries, dtype=np.float32)
        target_values = np.asarray(targets, dtype=np.float32)
        if query_values.ndim != 2 or target_values.ndim not in (2, 3):
            raise ValueError("HINT expects query (N,D) and target (G,Q,D) matrices")
        if target_values.ndim == 2:
            target_values = target_values[:, None, :]
        if target_values.shape[-1] != query_values.shape[-1]:
            raise ValueError("HINT query and target embedding dimensions differ")
        if categories is not None and len(categories) != len(query_values):
            raise ValueError("HINT categories must match query count")
        return np.einsum("nd,gqd->ngq", query_values, target_values).max(axis=-1)

    def _kill_worker(self) -> None:
        worker = self._worker
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
            self._worker = None
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None

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
        if self._owned_work_dir is not None:
            shutil.rmtree(self._owned_work_dir, ignore_errors=True)
            self._owned_work_dir = None

    def restart(self) -> None:
        self.close()


def _worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    job_path = Path(parser.parse_args(argv).worker_job)
    import torch
    from PIL import Image
    from workbench.backend.registry import checkpoint_is_present
    from workbench.replay.common import load_trusted_whole_model
    from workbench.replay.lavis_processors import reset
    from workbench.replay.hint import _load_replay_modules

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        checkpoint = job["checkpoint"]
        if not checkpoint_is_present(Path(job["checkpoint_path"]), checkpoint, model_id="hint"):
            raise FileNotFoundError(f"HINT checkpoint missing or invalid: {job['checkpoint_path']}")
        source = Path(job["source_root"])
        reset()
        torch, _, pinned, processors = _load_replay_modules(source)
        data_utils, _ = pinned
        network = load_trusted_whole_model(
            Path(job["checkpoint_path"]), checkpoint.get("expected_sha256"), torch)
        if not torch.cuda.is_available():
            raise RuntimeError("HINT requires CUDA in its model interpreter")
        device = torch.device("cuda")
        network = network.to(device).eval()
        transform = data_utils.targetpad_transform(1.25, 224)
        text_processor = processors[0]["eval"]
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return 1

    def run_one(current: dict[str, Any]) -> None:
        outputs = []
        try:
            with torch.no_grad():
                for start in range(0, len(current["image_paths"]), current["batch_size"]):
                    paths = current["image_paths"][start:start + current["batch_size"]]
                    images = []
                    for path in paths:
                        with Image.open(path) as image:
                            images.append(transform(image.convert("RGB")))
                    pixels = torch.stack(images).float().to(device)
                    if current["operation"] == "queries":
                        texts = [text_processor(text) for text in
                                 current["texts"][start:start + current["batch_size"]]]
                        features = network.extract_retrieval_compose(pixels, texts)
                        outputs.append(features.detach().cpu().numpy().reshape(len(paths), -1))
                    else:
                        features = network.extract_retrieval_target(pixels)
                        outputs.append(features.detach().cpu().numpy().transpose(0, 2, 1))
            shape = (0, 256) if current["operation"] == "queries" else (0, 32, 256)
            result = np.concatenate(outputs, axis=0) if outputs else np.empty(shape, dtype=np.float32)
            np.save(current["output_path"], result.astype(np.float32, copy=False), allow_pickle=False)
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


ADAPTER = Adapter

if __name__ == "__main__":
    raise SystemExit(_worker_main())
