"""Subprocess CIRCUS adapter for TME's image-difference FashionIQ model."""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapters import AdapterStatus, AdapterWorkerError, AdapterWorkRootError, CompositionAdapter

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
CHECKPOINT_ID = "fashioniq_0"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PROBE = (
    "import json, torch; ok=torch.cuda.is_available(); "
    "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':ok,"
    "'capability':list(torch.cuda.get_device_capability()) if ok else None}))"
)



class Adapter(CompositionAdapter):
    """Run TME's native state-dict model forward on images opened from path.

    ``precompute_test.py:114,120`` selects TargetPad(1.25,224) and the ``blip_caption``
    eval processor; ``utility.py:130-144`` defines bicubic resize/crop, RGB conversion,
    tensor conversion and CLIP normalization. Native evaluator ``precompute.py:35-47``
    caches ViT features and ``precompute_evaluation.py:46-50`` reads them; this adapter
    intentionally bypasses that feature cache and computes image features from every
    provided path, including staged black references. ``precompute_evaluation.py:68-72``
    prepares edit text, then calls ``model.inference``. The adapter retains that native
    model inference: ``blip2_qformer_cir_image_diff_features.py:240-254`` tokenizes text,
    fuses reference tokens, projects/normalizes targets, and takes maximum target-token
    dot product. Input text remains runner-provided; eval processor prepends configured
    prompt (empty in released config), lowercases, removes punctuation, collapses spaces,
    strips and truncates to 50 words (``blip_processors.py:28-68``). Empty input stays
    empty before tokenization and has no edit-bearing auxiliary field. FashionIQ category
    iteration in ``precompute_evaluation.py:13-20`` uses same model for all categories.
    Scope is category-agnostic and reference eligible. Checkpoint is ``fashioniq_0`` at
    0% training noise. Controller-side NumPy performs exact full-gallery ranking.

    Loader mirrors ``replay/tme.py:358-376``: call the model's own
    ``load_model_and_preprocess``, load the registry-selected state dict with
    ``torch.load(weights_only=True)``, then check missing/unexpected keys. Registry
    publishes member size but no SHA-256; ``replay/tme.py:191-194`` checks file presence
    and non-empty size, then ``:367-375`` loads and checks state-dict compatibility.
    Thus checkpoint is author-published and size-verified, but hash-unverified.
    """

    model_id = "tme"
    category_scopes = (None,)
    reference_eligible = True
    work_root: Path | None = None

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config
        from workbench.backend.registry import checkpoint_by_id, load_registry
        from workbench.backend.runtime import resolve_model_interpreter

        self.config = config if config is not None else resolve_config()
        registry = getattr(self.config, "registry", None) or load_registry()
        self.model = next(row for row in registry["models"] if row["model_id"] == self.model_id)
        selected = checkpoint_id or CHECKPOINT_ID
        if selected != CHECKPOINT_ID:
            raise ValueError(f"TME requires checkpoint {CHECKPOINT_ID!r}, got {selected!r}")
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
        self._verified_interpreter: str | None = None
        self._counter = 0
        self._job_dir: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None

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
                    reasons.append(f"CUDA unavailable in TME interpreter {interpreter}")
                elif adaptation:
                    capability = "sm_" + "".join(map(str, probe["capability"]))
                    target = str(adaptation.get("target_compute_capability", "")).replace("_blackwell", "")
                    if target != capability:
                        reasons.append(f"runtime adaptation targets {target}, but TME interpreter GPU is {capability}")
            except Exception as error:
                reasons.append(f"TME interpreter/runtime probe failed: {type(error).__name__}: {error}")
        return AdapterStatus(
            cls.model_id, not reasons, "; ".join(reasons) or None, (CHECKPOINT_ID,),
            {condition: (True, None) for condition in CONDITIONS},
            notes="Checkpoint is author-published and size-verified; registry has no author SHA-256.",
        )

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        return (True, None) if condition in CONDITIONS else (False, f"unknown composition condition: {condition!r}")

    def _work_dir(self) -> Path:
        work_root = self.require_work_root().expanduser().resolve()
        if self._job_dir is None:
            adapter_root = work_root / self.model_id
            adapter_root.mkdir(parents=True, exist_ok=True)
            self._job_dir = adapter_root / f"worker-{uuid.uuid4().hex}"
            self._job_dir.mkdir()
        return self._job_dir

    def _run_job(self, operation: str, image_paths: Sequence[str], texts: Sequence[str] = (),
                 categories: Sequence[str | None] = ()) -> np.ndarray:
        if not self.interpreter or not Path(self.interpreter).is_file():
            raise RuntimeError("TME model interpreter is not configured")
        if self._verified_interpreter is None:
            from workbench.backend.runtime import require_verified_model_interpreter

            verified = require_verified_model_interpreter(self.model, config=self.config)
            if Path(verified).resolve() != Path(self.interpreter).resolve():
                raise RuntimeError(f"TME resolved interpreter changed: {self.interpreter} -> {verified}")
            self._verified_interpreter = verified
        self._counter += 1
        job_path = self._work_dir() / f"job-{self._counter}.json"
        output_path = self._job_dir / f"output-{self._counter}.npy"
        job_path.write_text(json.dumps({
            "operation": operation,
            "image_paths": [str(Path(path).expanduser().resolve()) for path in image_paths],
            "texts": list(texts), "categories": list(categories), "output_path": str(output_path.resolve()),
            "checkpoint_path": str(Path(self.checkpoint_path).expanduser().resolve()),
            "checkpoint": self.checkpoint,
            "source_root": str(Path(self.source_root).expanduser().resolve()),
            "dataset_root": str(Path(self.dataset_root).expanduser().resolve()),
            "batch_size": self.batch_size,
        }), encoding="utf-8")
        scope = tuple(categories)
        if self._worker is not None and self._worker.poll() is not None:
            status = self._worker.returncode
            self.restart()
            raise AdapterWorkerError(
                f"adapter_worker_exited: model=tme scope={scope} phase={operation} exit_status={status}")
        if self._worker is None:
            self._worker_log = (self._job_dir / "worker.stderr.log").open("w", encoding="utf-8")
            self._worker = subprocess.Popen(
                [self.interpreter, "-m", "workbench.composition.model_tme",
                 "--worker-job", str(job_path.resolve())], cwd=REPO_ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._worker_log,
                text=True, bufsize=1)
        else:
            assert self._worker.stdin is not None
            self._worker.stdin.write(json.dumps({"job_path": str(job_path.resolve())}) + "\n")
            self._worker.stdin.flush()
        assert self._worker.stdout is not None
        line = self._worker.stdout.readline()
        while line and not line.startswith("CIRCUS_WORKER:"):
            line = self._worker.stdout.readline()
        if not line:
            worker = self._worker
            self.restart()
            status = worker.returncode if worker is not None else None
            raise AdapterWorkerError(
                f"adapter_worker_exited: model=tme scope={scope} phase={operation} exit_status={status}")
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            job_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            self.restart()
            raise AdapterWorkerError(
                f"adapter_worker_job_failed: model=tme scope={scope} phase={operation} "
                f"error={response.get('error_type')}: {response.get('error')}")
        result = np.load(output_path, allow_pickle=False).astype(np.float32, copy=False)
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result

    def encode_queries(self, image_paths: Sequence[str], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if self.work_root is None:
            raise AdapterWorkRootError(f"{self.model_id} work_root must be assigned before encoding")
        if len(image_paths) != len(texts) or len(image_paths) != len(categories):
            raise ValueError("TME image paths, texts, and categories must have equal lengths")
        return self._run_job("queries", image_paths, texts, categories)
    def encode_targets(self, image_paths: Sequence[str], categories: Sequence[str | None]) -> np.ndarray:

        if self.work_root is None:
            raise AdapterWorkRootError(f"{self.model_id} work_root must be assigned before encoding")
        if len(image_paths) != len(categories):
            raise ValueError("TME target paths and categories must have equal lengths")
        if any(category is not None for category in categories):
            raise ValueError("TME gallery is category-agnostic; target categories must all be None")
        return self._run_job("targets", image_paths, categories=categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        query_values = np.asarray(queries, dtype=np.float32)
        target_values = np.asarray(targets, dtype=np.float32)
        if query_values.ndim != 2 or target_values.ndim not in (2, 3):
            raise ValueError("TME expects query (N,D) and target (G,Q,D) matrices")
        if target_values.ndim == 2:
            target_values = target_values[:, None, :]
        if target_values.shape[-1] != query_values.shape[-1]:
            raise ValueError("TME query and target embedding dimensions differ")
        if categories is not None and len(categories) != len(query_values):
            raise ValueError("TME query scopes must match query count")
        return np.einsum("nd,gqd->ngq", query_values, target_values).max(axis=-1)

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
        if self._job_dir is not None:
            shutil.rmtree(self._job_dir, ignore_errors=True)
            self._job_dir = None

    def restart(self) -> None:
        self.close()



def _prepare_worker_cwd(job: dict[str, Any], base_dir: Path | None = None) -> Path:
    """Resolve job paths against launch cwd, then enter the vendored TME source root."""
    from workbench.replay.tme import source_root_for

    base_dir = Path(base_dir or Path.cwd()).expanduser().resolve()
    resolve = lambda path: (base_dir / Path(path).expanduser()).resolve()
    source_root = source_root_for(resolve(job["source_root"]))
    job["source_root"] = str(source_root)
    job["checkpoint_path"] = str(resolve(job["checkpoint_path"]))
    job["dataset_root"] = str(resolve(job["dataset_root"]))
    job["image_paths"] = [str(resolve(path)) for path in job["image_paths"]]
    job["output_path"] = str(resolve(job["output_path"]))
    import os

    os.chdir(source_root)
    return source_root


def _worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import os
    import sys
    import torch
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    launch_cwd = Path.cwd()
    job_path = Path(parser.parse_args(argv).worker_job).expanduser().resolve()
    job = json.loads(job_path.read_text(encoding="utf-8"))
    source_root = _prepare_worker_cwd(job, launch_cwd)
    from PIL import Image
    from workbench.backend.registry import checkpoint_is_present

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        checkpoint = job["checkpoint"]
        path = Path(job["checkpoint_path"])
        if not checkpoint_is_present(path, checkpoint, model_id="tme"):
            raise FileNotFoundError(f"TME checkpoint missing or invalid: {path}")
        source = source_root / "src"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        os.environ["MTST_DATA_ROOT"] = str(Path(job["dataset_root"]).parent)
        from lavis.models import load_model_and_preprocess
        from utility import targetpad_transform
        from workbench.replay.tme import MODEL_NAME, MODEL_TYPE

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            raise RuntimeError("TME requires CUDA in its model interpreter")
        network, _, processors = load_model_and_preprocess(
            name=MODEL_NAME, model_type=MODEL_TYPE, is_eval=False, device=device)
        state = torch.load(path, map_location="cpu", weights_only=True)
        report = network.load_state_dict(state, strict=False)
        if report.missing_keys or report.unexpected_keys:
            raise RuntimeError(
                f"TME checkpoint did not load completely: {len(report.missing_keys)} missing, "
                f"{len(report.unexpected_keys)} unexpected keys")
        network.eval()
        transform = targetpad_transform(target_ratio=1.25, dim=224)
        text_processor = processors["eval"]
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return 0

    def run_one(current: dict[str, Any]) -> None:
        _prepare_worker_cwd(current)
        outputs = []
        try:
            with torch.no_grad():
                for start in range(0, len(current["image_paths"]), current["batch_size"]):
                    paths = current["image_paths"][start:start + current["batch_size"]]
                    image_tensors = []
                    for image_path in paths:
                        with Image.open(image_path) as image:
                            image_tensors.append(transform(image.convert("RGB")))
                    pixels = torch.stack(image_tensors).float().to(device)
                    with torch.cuda.amp.autocast():
                        vit_features = network.vit_encode(pixels)
                    reference = network.encode_image(vit_features)
                    if current["operation"] == "queries":
                        texts = [text_processor(text) for text in current["texts"][start:start + current["batch_size"]]]
                        tokens = network.tokenizer(
                            texts, padding="max_length", truncation=True,
                            max_length=network.max_txt_len, return_tensors="pt",
                        ).to(reference.device)
                        query = network.encode_fusion(reference, tokens)
                        outputs.append(query.detach().cpu().numpy())
                    else:
                        projected = network.vision_proj(reference)
                        outputs.append(torch.nn.functional.normalize(projected, dim=-1).detach().cpu().numpy())
            shape = (0, 256) if current["operation"] == "queries" else (0, 32, 256)
            result = np.concatenate(outputs, axis=0) if outputs else np.empty(shape, dtype=np.float32)
            np.save(current["output_path"], result.astype(np.float32, copy=False), allow_pickle=False)
            reply(True)
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))

    run_one(job)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        if request.get("stop"):
            break
        try:
            next_job_path = Path(request["job_path"]).expanduser().resolve()
            run_one(json.loads(next_job_path.read_text(encoding="utf-8")))
        except Exception as error:
            reply(False, error_type=type(error).__name__, error=str(error))
    del network
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


ADAPTER = Adapter

if __name__ == "__main__":
    raise SystemExit(_worker_main())
