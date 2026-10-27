"""Subprocess CIRCUS adapter for TME's image-difference FashionIQ model."""
from __future__ import annotations

import json
import shutil
import uuid
import subprocess
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .adapters import AdapterStatus, AdapterWorkRootError, CompositionAdapter

CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
CHECKPOINT_ID = "fashioniq_0"
REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER_PROBE = (
    "import json, torch; ok=torch.cuda.is_available(); "
    "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'available':ok,"
    "'capability':list(torch.cuda.get_device_capability()) if ok else None}))"
)


class AdapterWorkerError(RuntimeError):
    """A model worker exited before producing its complete feature matrix."""

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
        self._owned_work_dir: Path | None = None

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
        if self.work_root is None:
            raise AdapterWorkRootError(f"{self.model_id} work_root must be assigned before encoding")
        if self._owned_work_dir is None:
            root = Path(self.work_root).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            self._owned_work_dir = root / f".tme-worker-{uuid.uuid4().hex}"
            self._owned_work_dir.mkdir()
        return self._owned_work_dir

    def _run_job(self, operation: str, image_paths: Sequence[str], texts: Sequence[str] = (),
                 categories: Sequence[str | None] = ()) -> np.ndarray:
        if not self.interpreter:
            raise RuntimeError("TME model interpreter is not configured")
        from workbench.backend.runtime import require_verified_model_interpreter

        verified = require_verified_model_interpreter(self.model, config=self.config)
        if Path(verified).resolve() != Path(self.interpreter).resolve():
            raise RuntimeError(f"TME resolved interpreter changed: {self.interpreter} -> {verified}")
        work = self._work_dir()
        job_path = work / f"{uuid.uuid4().hex}.json"
        output_path = work / f"{uuid.uuid4().hex}.npy"
        job_path.write_text(json.dumps({
            "operation": operation, "image_paths": [str(Path(path).resolve()) for path in image_paths],
            "texts": list(texts), "categories": list(categories), "output_path": str(output_path),
            "checkpoint_path": str(self.checkpoint_path), "checkpoint": self.checkpoint,
            "source_root": str(self.source_root), "dataset_root": str(self.dataset_root),
            "batch_size": self.batch_size,
        }), encoding="utf-8")
        argv = [self.interpreter, "-m", "workbench.composition.model_tme", "--worker-job", str(job_path)]
        try:
            subprocess.run(argv, check=True, cwd=REPO_ROOT)
        except subprocess.CalledProcessError as error:
            raise AdapterWorkerError(
                f"TME worker failed for scope {tuple(categories)!r}: exit status {error.returncode}") from error
        return np.load(output_path, allow_pickle=False).astype(np.float32, copy=False)

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
        if self._owned_work_dir is not None:
            shutil.rmtree(self._owned_work_dir)
            self._owned_work_dir = None



def _worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    job = json.loads(Path(parser.parse_args(argv).worker_job).read_text(encoding="utf-8"))
    import os
    import sys
    import torch
    from PIL import Image
    from workbench.backend.registry import checkpoint_is_present

    checkpoint = job["checkpoint"]
    path = Path(job["checkpoint_path"])
    if not checkpoint_is_present(path, checkpoint, model_id="tme"):
        raise FileNotFoundError(f"TME checkpoint missing or invalid: {path}")
    source = Path(job["source_root"]) / "src"
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
    state = torch.load(path, map_location=device, weights_only=True)
    report = network.load_state_dict(state, strict=False)
    if report.missing_keys or report.unexpected_keys:
        raise RuntimeError(
            f"TME checkpoint did not load completely: {len(report.missing_keys)} missing, "
            f"{len(report.unexpected_keys)} unexpected keys")
    network.eval()
    transform = targetpad_transform(target_ratio=1.25, dim=224)
    text_processor = processors["eval"]
    outputs = []
    with torch.no_grad():
        for start in range(0, len(job["image_paths"]), job["batch_size"]):
            paths = job["image_paths"][start:start + job["batch_size"]]
            image_tensors = []
            for image_path in paths:
                with Image.open(image_path) as image:
                    image_tensors.append(transform(image.convert("RGB")))
            pixels = torch.stack(image_tensors).float().to(device)
            reference = network.encode_image(network.vit_encode(pixels))
            if job["operation"] == "queries":
                texts = [text_processor(text) for text in job["texts"][start:start + job["batch_size"]]]
                tokens = network.tokenizer(
                    texts, padding="max_length", truncation=True,
                    max_length=network.max_txt_len, return_tensors="pt",
                ).to(reference.device)
                query = network.encode_fusion(reference, tokens)
                outputs.append(query.detach().cpu().numpy())
            else:
                projected = network.vision_proj(reference)
                outputs.append(torch.nn.functional.normalize(projected, dim=-1).detach().cpu().numpy())
    shape = (0, 256) if job["operation"] == "queries" else (0, 32, 256)
    result = np.concatenate(outputs, axis=0) if outputs else np.empty(shape, dtype=np.float32)
    np.save(job["output_path"], result.astype(np.float32, copy=False), allow_pickle=False)
    del network
    torch.cuda.empty_cache()
    return 0


ADAPTER = Adapter

if __name__ == "__main__":
    raise SystemExit(_worker_main())
