"""CIRCUS adapter for DCNet native context-aware scoring.

Adapter packs source, text, composed, and gallery experts into a model-specific numeric cache;
worker applies DCNet's query-conditioned correction and global gallery log-softmax fusion.
FashionIQ reference exclusion is native evaluator policy; CIRCUS full-gallery scoring keeps reference eligibility.
"""
from __future__ import annotations

from os import environ
from pathlib import Path
from subprocess import Popen
from typing import Any, Sequence, TextIO

import numpy as np

from workbench.composition.adapters import (
    AdapterStatus,
    AdapterWorkRootError,
    CompositionAdapter,
)
from workbench.composition.inputs import CONDITIONS

#: The shared width is only for legacy synthetic probes; native packed contexts have distinct shapes.
DIMENSION = 1024

class Refused(RuntimeError):
    """Adapter launch is blocked by verified runtime or input prerequisites."""

class ScoringNotReproducible(RuntimeError):
    """Raised only when native DCNet context or score inputs are absent."""

#: Candidate-environment survey; availability does not constitute assignment.
INTERPRETER_SURVEY = (
    "dcnet-prepare (3.7.12): no torch; cir-005ef4e1b02999a5 (3.10.22, torch 2.7.1+cu128): "
    "strict checkpoint load 518/518; dcnet-377 (3.7.7, torch 1.4.0+cpu): weights_only argument "
    "unsupported"
)

def _mask_reference_scores(composition: Any, correction: Any,
                           reference_indices: Any, value: float = -10e10) -> None:
    """Apply source-native reference exclusion before either gallery softmax."""
    import torch

    references = torch.as_tensor(reference_indices, device=composition.device, dtype=torch.long)
    if references.ndim != 1 or len(references) != len(composition):
        raise ValueError("DCNet reference indices must contain one index per score row")
    if composition.shape != correction.shape or torch.any(references < 0) or torch.any(
            references >= composition.shape[1]):
        raise ValueError("DCNet reference indices or component score shapes are invalid")
    rows = torch.arange(len(references), device=composition.device)
    composition[rows, references] = value
    correction[rows, references] = value


class Adapter(CompositionAdapter):
    """DCNet adapter with context-aware native scoring."""

    model_id = "dcnet"
    reference_eligible = True
    query_dimension = 21504
    score_dimension = DIMENSION
    category_scopes: tuple[str | None, ...] = (None,)

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config

        self.config = config if config is not None else resolve_config()
        self._model, self._checkpoint, self._interpreter = _resolve(self.config, checkpoint_id)
        super().__init__(self.config, self._checkpoint["checkpoint_id"])
        self.checkpoint = self._checkpoint
        self.checkpoint_path = Path(self._checkpoint["resolved_path"])
        self.source_root = Path(self.config.WORKBENCH_THIRD_PARTY_ROOT) / self._model["source_dir"]
        self.work_root: Path | None = None
        self._interpreter_verified = False
        self._worker: Popen[str] | None = None
        self._worker_log: TextIO | None = None
        self.worker_identity: dict[str, Any] | None = None
        self._worker_failed: str | None = None
        self._worker_counter = 0
        self._job_root: Path | None = None
        self._packed_layout = _packed_layout()

    def prepare_identity(self) -> dict[str, Any]:
        import hashlib
        import json

        config_path = self.checkpoint_path / "config.json"
        native = json.loads(config_path.read_text(encoding="utf-8"))
        experts = native["experts"]
        arch = native["arch"]["args"]
        data_root = Path(self.config.FASHIONIQ_ROOT)
        artifacts_root = Path(self.config.WORKBENCH_ARTIFACTS_ROOT or
                              (Path(self.config.CIR_REPO_ROOT) / "workbench" / "artifacts"))
        nltk_data = artifacts_root / "assets" / "nltk_data"
        punkt = nltk_data / "tokenizers" / "punkt" / "english.pickle"
        if tuple(experts["modalities"]) != tuple(self._packed_layout["modalities"]):
            raise ValueError("DCNet checkpoint modalities do not match packed context layout")
        if experts["ce_shared_dim"] != DIMENSION or experts["text_dim"] != 300:
            raise ValueError("DCNet checkpoint dimensions do not match packed context layout")
        if native["data_loader"]["args"]["batch_size"] != self.batch_size:
            raise ValueError("DCNet runner batch size differs from checkpoint evaluation batch size")
        expected = {"composition": "tirg", "correction": "fd", "fusion": "hadamard",
                    "metric_fn": "ip", "backbone": "resnet", "norm_scale": 5.0,
                    "text_pooling_module": "sharedconv"}
        if any(arch[key] != value for key, value in expected.items()):
            raise ValueError(f"DCNet checkpoint architecture differs from worker factory: {arch}")
        if experts["text_feat"] != "glove":
            raise ValueError(f"unsupported DCNet checkpoint text feature: {experts['text_feat']}")
        if not punkt.is_file():
            raise FileNotFoundError(f"DCNet author tokenizer resource is missing: {punkt}")
        actual = {"composition": arch["composition"], "correction": arch["correction"],
                  "fusion": arch["fusion"], "metric": arch["metric_fn"],
                  "backbone": arch["backbone"], "norm_scale": arch["norm_scale"],
                  "text_pooling_module": arch["text_pooling_module"],
                  "text_feature": experts["text_feat"], "text_dim": experts["text_dim"],
                  "image_transform": "CenterCrop(224), ImageNet normalize",
                  "text_preprocess": "FashionIQ glove_vecs.pkl + nltk.word_tokenize"}
        resources = [data_root / "captions" / "glove_vecs.pkl"] + [
            data_root / "captions" / f"cap.{category}.glove.{split}.pkl"
            for category in ("dress", "shirt", "toptee")
            for split in ("train", "val", "test")
        ] + [punkt]
        missing = [str(path) for path in resources if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"DCNet text resources missing: {missing[0]}")
        def file_sha256(path: Path) -> str:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        checkpoint_resources = {
            str((self.checkpoint_path / "trained_model.pth").resolve()): self._checkpoint["sha256"],
            str(config_path.resolve()): file_sha256(config_path),
        }
        self.identity_payload = {
            "dcnet_packed_context": self._packed_layout,
            "checkpoint_id": self._checkpoint["checkpoint_id"],
            "checkpoint_resources": checkpoint_resources,
            "source_provenance": {
                "upstream_commit": json.loads((self.source_root / "provenance.json").read_text(
                    encoding="utf-8"))["upstream_commit"],
                "manifest_sha256": file_sha256(self.source_root / "provenance.json"),
            },
            "batch_size": self.batch_size,
            "text_resource_sha256": {str(path.resolve()): file_sha256(path) for path in resources},
        }
        return self.identity_payload

    @classmethod
    def status(cls, config: Any = None) -> AdapterStatus:
        from workbench.backend.operator_config import resolve_config

        return _status(cls, config if config is not None else resolve_config())

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        if condition not in CONDITIONS:
            return False, f"unknown composition condition: {condition!r}"
        reason = _refusal(self)
        return reason is None, reason

    def encode_queries(self, image_paths: Sequence[str | Path], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        if not (len(image_paths) == len(texts) == len(categories)):
            raise ValueError("image_paths, texts, and categories must have equal lengths")
        if any(category is not None for category in categories):
            raise ValueError("DCNet native global scoring requires category=None")
        return _run(self, "queries", image_paths, texts, categories)

    def encode_targets(self, image_paths: Sequence[str | Path],
                       categories: Sequence[str | None]) -> np.ndarray:
        if len(image_paths) != len(categories):
            raise ValueError("image_paths and categories must have equal lengths")
        if any(category is not None for category in categories):
            raise ValueError("DCNet gallery is category-agnostic; target categories must all be None")
        return _run(self, "targets", image_paths, (), categories)
    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None,
              reference_indices: Sequence[int] | None = None,
              capture_components_path: Path | None = None) -> np.ndarray:
        del categories
        query_rows = np.asarray(queries, dtype=np.float32)
        target_rows = (targets if isinstance(targets, np.memmap) and targets.dtype == np.float32
                       else np.asarray(targets, dtype=np.float32))
        if query_rows.ndim != 2 or query_rows.shape[1] != self._packed_layout["query_width"]:
            raise ValueError(f"DCNet query layout mismatch: {query_rows.shape}")
        if target_rows.ndim != 2 or target_rows.shape[1] != self._packed_layout["target_width"]:
            raise ValueError(f"DCNet gallery layout mismatch: {target_rows.shape}")
        if reference_indices is not None:
            reference_indices = tuple(reference_indices)
            if len(reference_indices) != len(query_rows) or any(
                    not isinstance(index, (int, np.integer)) or index < 0 or index >= len(target_rows)
                    for index in reference_indices):
                raise ValueError("DCNet reference_indices must contain one in-range gallery index per query")
        if not len(query_rows) or not len(target_rows):
            return np.empty((len(query_rows), len(target_rows)), dtype=np.float32)
        if capture_components_path is not None:
            capture_components_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(capture_components_path.with_suffix(".queries.npy"), query_rows, allow_pickle=False)
            np.save(capture_components_path.with_suffix(".targets.npy"), target_rows, allow_pickle=False)
        return _score(self, query_rows, target_rows, reference_indices, capture_components_path)
    def close(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            import subprocess
            try:
                if worker.poll() is None and worker.stdin is not None:
                    worker.stdin.write('{"stop":true}\n')
                    worker.stdin.flush()
                worker.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                if worker.poll() is None:
                    worker.kill()
                worker.wait()
        if self._worker_log is not None:
            self._worker_log.close()
            self._worker_log = None
        if self._job_root is not None:
            import shutil
            shutil.rmtree(self._job_root, ignore_errors=True)
            self._job_root = None


def _packed_layout() -> dict[str, Any]:
    modalities = ("im_feat", "inter-1", "spatial0", "spatial1", "spatial2", "spatial3", "spatial4")
    width = len(modalities) * DIMENSION
    return {"version": "dcnet-context-v1", "modalities": list(modalities), "expert_width": DIMENSION,
            "query_roles": {"source": [0, width], "text": [width, 2 * width],
                            "composed": [2 * width, 3 * width]},
            "gallery_role": [0, width], "query_width": 3 * width, "target_width": width,
            "dtype": "float32", "scale": "native CENet normalization and raw inner products",
            "score": "global gallery log_softmax(composition)+log_softmax(correction)"}

def _resolve(config: Any, checkpoint_id: str | None) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id("dcnet")
    checkpoint = checkpoint_by_id(
        model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    resolved_checkpoint_path = checkpoint_path(
        "dcnet", checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    checkpoint = dict(checkpoint, resolved_path=str(resolved_checkpoint_path))
    if "sha256" not in checkpoint:
        checkpoint = dict(checkpoint, sha256=next(
            member["expected_sha256"] for member in checkpoint.get("bundle_members", [])
            if member["filename"] == "trained_model.pth"))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)

def _blockers(model: dict[str, Any], checkpoint: dict[str, Any], path: Path,
              interpreter: str | None, config: Any) -> list[str]:
    """Every reason this model may not be scored, in one place for status() and _run()."""
    from workbench.backend.registry import checkpoint_is_present

    reasons: list[str] = []
    mapping = checkpoint.get("checkpoint_mapping_status")
    if mapping != "VERIFIED_METADATA":
        reasons.append(f"checkpoint mapping is {mapping} for {checkpoint['checkpoint_id']}")
    # The artifact is the run directory, so presence means config.json plus trained_model.pth.
    if not checkpoint_is_present(path, checkpoint, model_id=model["model_id"], config=config):
        reasons.append(f"checkpoint run directory incomplete at {path}")
    if not interpreter:
        reasons.append("no interpreter environment resolved for "
                       f"{model['environment']['python_env_var']} (unset, and no provisioning "
                       "assignment records one for dcnet)")
    elif not Path(interpreter).is_file():
        reasons.append(f"model interpreter missing at {interpreter}")
    return reasons

def _refusal(adapter: Adapter) -> str | None:
    reasons = _blockers(adapter._model, adapter._checkpoint, adapter.checkpoint_path,
                        adapter._interpreter, adapter.config)
    return "; ".join(reasons) if reasons else None

def _run(adapter: Adapter, operation: str, image_paths: Sequence[str | Path], texts: Sequence[str],
         categories: Sequence[str | None]) -> np.ndarray:
    if adapter.work_root is None:
        raise AdapterWorkRootError("dcnet work_root was not set by runner")
    width = adapter._packed_layout["query_width"] if operation == "queries" else adapter._packed_layout["target_width"]
    if not image_paths:
        return np.empty((0, width), dtype=np.float32)
    if adapter._worker_failed:
        raise Refused(adapter._worker_failed)
    if not adapter._interpreter_verified:
        adapter._interpreter = adapter.require_verified_interpreter()
        adapter._interpreter_verified = True
    if adapter._job_root is None:
        import uuid
        adapter._job_root = Path(adapter.work_root) / f".dcnet-worker-{uuid.uuid4().hex}"
        adapter._job_root.mkdir(parents=True)
    paths = []
    for image_path in image_paths:
        path = Path(image_path).expanduser()
        if not path.is_file():
            candidate = Path(adapter.config.FASHIONIQ_ROOT) / "resized_images" / path.name
            if candidate.is_file():
                path = candidate
            else:
                raise FileNotFoundError(f"DCNet image asset not found: {image_path}")
        paths.append(str(path.resolve()))
    adapter._worker_counter += 1
    output = adapter._job_root / f"features-{adapter._worker_counter}.npy"
    artifacts_root = Path(adapter.config.WORKBENCH_ARTIFACTS_ROOT or
                           (Path(adapter.config.CIR_REPO_ROOT) / "workbench" / "artifacts"))
    job = {"operation": operation, "image_paths": paths,
           "texts": list(texts), "categories": list(categories), "output_path": str(output),
           "source_root": str(adapter.source_root), "checkpoint_path": str(adapter.checkpoint_path),
           "fashioniq_root": str(adapter.config.FASHIONIQ_ROOT),
           "nltk_data": str(artifacts_root / "assets" / "nltk_data"),
           "batch_size": adapter.batch_size, "seed": 0, "autocast": False,
           "tf32": False, "cudnn_deterministic": True, "cudnn_benchmark": False}
    return _request(adapter, job, (len(image_paths), width))
def _score(adapter: Adapter, queries: np.ndarray, targets: np.ndarray,
           reference_indices: Sequence[int] | None = None,
           capture_components_path: Path | None = None) -> np.ndarray:
    if adapter.work_root is None:
        raise AdapterWorkRootError("dcnet work_root was not set by runner")
    if adapter._worker_failed:
        raise Refused(adapter._worker_failed)
    if not adapter._interpreter_verified:
        adapter._interpreter = adapter.require_verified_interpreter()
        adapter._interpreter_verified = True
    if adapter._job_root is None:
        import uuid
        adapter._job_root = Path(adapter.work_root) / f".dcnet-worker-{uuid.uuid4().hex}"
        adapter._job_root.mkdir(parents=True)
    adapter._worker_counter += 1
    q_path = adapter._job_root / f"queries-{adapter._worker_counter}.npy"
    target_filename = getattr(targets, "filename", None)
    owned_target = not (isinstance(targets, np.memmap) and target_filename and Path(target_filename).is_file())
    t_path = Path(target_filename) if not owned_target else adapter._job_root / f"targets-{adapter._worker_counter}.npy"
    np.save(q_path, queries, allow_pickle=False)
    if owned_target:
        np.save(t_path, targets, allow_pickle=False)
    if capture_components_path is not None:
        capture_components_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        out = adapter._job_root / f"scores-{adapter._worker_counter}.npy"
        artifacts_root = Path(adapter.config.WORKBENCH_ARTIFACTS_ROOT or
                              (Path(adapter.config.CIR_REPO_ROOT) / "workbench" / "artifacts"))
        job = {"operation": "score", "query_path": str(q_path), "target_path": str(t_path),
               "output_path": str(out), "source_root": str(adapter.source_root),
               "checkpoint_path": str(adapter.checkpoint_path),
               "fashioniq_root": str(adapter.config.FASHIONIQ_ROOT),
               "nltk_data": str(artifacts_root / "assets" / "nltk_data"),
               "reference_indices": list(reference_indices) if reference_indices is not None else None,
               "reference_mask_value": -10e10, "batch_size": adapter.batch_size, "seed": 0,
               "autocast": False, "tf32": False, "cudnn_deterministic": True,
               "cudnn_benchmark": False,
               "capture_components_path": str(capture_components_path) if capture_components_path else None}
        return _request(adapter, job, (len(queries), len(targets)))
    finally:
        q_path.unlink(missing_ok=True)
        if owned_target:
            t_path.unlink(missing_ok=True)
def _request(adapter: Adapter, job: dict[str, Any], expected: tuple[int, int]) -> np.ndarray:
    import json
    import subprocess
    job_root = adapter._job_root
    if job_root is None:
        raise AdapterWorkRootError("DCNet worker job root was not initialized")
    adapter._worker_counter += 1
    job_path = job_root / f"job-{adapter._worker_counter}.json"
    output = Path(job["output_path"])
    job_path.write_text(json.dumps(job), encoding="utf-8")
    try:
        worker = adapter._worker
        if worker is not None and worker.poll() is not None:
            raise Refused(f"DCNet worker exited with {worker.returncode}")
        if worker is None:
            interpreter = adapter._interpreter
            if not interpreter:
                raise Refused("DCNet model interpreter is unresolved")
            adapter._worker_log = (job_root / "worker.stderr.log").open("w", encoding="utf-8")
            worker = subprocess.Popen(
                [interpreter, "-m", "workbench.composition.model_dcnet", "--worker-job", str(job_path)],
                cwd=Path(__file__).resolve().parents[2], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=adapter._worker_log, text=True, bufsize=1)
            adapter._worker = worker
        else:
            if worker.stdin is None:
                raise Refused("DCNet worker stdin is closed")
            worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
            worker.stdin.flush()
        if worker.stdout is None:
            raise Refused("DCNet worker stdout is closed")
        line = worker.stdout.readline()
        while line and not line.startswith("DCNET_WORKER:"):
            line = worker.stdout.readline()
        if not line:
            raise Refused(f"DCNet worker exited with {worker.poll()}")
        response = json.loads(line[len("DCNET_WORKER:"):])
        worker_identity = {key: response[key] for key in (
            "executable", "prefix", "cwd", "argv", "model_module", "torch_module",
            "torchvision_module", "torch_version", "torchvision_version", "device")}
        expected_interpreter = adapter._interpreter
        if expected_interpreter is None or Path(worker_identity["executable"]).resolve() != Path(expected_interpreter).resolve():
            raise Refused("DCNet worker executable differs from verified interpreter")
        expected_source = (Path(job["source_root"]) / "model" / "model.py").resolve()
        if Path(worker_identity["model_module"]).resolve() != expected_source:
            raise Refused("DCNet worker imported model from unexpected source path")
        adapter.worker_identity = worker_identity
        if not response.get("ok"):
            raise Refused(f"DCNet worker operation={response.get('operation')!r} "
                          f"{response.get('error_type')}: {response.get('error')}\n"
                          f"{response.get('traceback', '')}")
        result = np.load(output, allow_pickle=False)
        if result.dtype != np.float32 or result.shape != expected:
            raise Refused(f"DCNet output {result.shape}/{result.dtype}; expected {expected}/float32")
        return result
    except Exception as error:
        adapter._worker_failed = f"{type(error).__name__}: {error}"
        raise
    finally:
        job_path.unlink(missing_ok=True)
        output.unlink(missing_ok=True)

def _worker_main(first_job: dict[str, Any]) -> int:
    environ["NLTK_DATA"] = first_job["nltk_data"]
    import json
    import sys
    import traceback
    import torch
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image
    seed = int(first_job["seed"])
    if seed != 0 or first_job["autocast"] or first_job["tf32"]:
        raise ValueError("DCNet worker requires seed=0, autocast=false, and tf32=false")
    if not first_job["cudnn_deterministic"] or first_job["cudnn_benchmark"]:
        raise ValueError("DCNet worker requires deterministic cuDNN and benchmark=false")
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    sys.path.insert(0, first_job["source_root"])
    from collections import OrderedDict
    from model.model import CENet
    modalities = _packed_layout()["modalities"]
    checkpoint_root = Path(first_job["checkpoint_path"])
    saved = torch.load(checkpoint_root / "trained_model.pth", map_location="cpu", weights_only=False)
    expert_dims = OrderedDict((name, 1024 if name == "inter-1" else 2048) for name in sorted(modalities))
    model = CENet(text_dim=300, composition="tirg", correction="fd", fusion="hadamard",
                  expert_dims=expert_dims, same_dim=1024, text_feat="glove", norm_scale=5.0,
                  backbone="resnet", text_pooling_module="sharedconv", metric_fn="ip")
    model.load_state_dict(saved["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    transform = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    import torchvision
    from model import model as dcnet_model_module
    runtime_identity = {"executable": sys.executable, "prefix": sys.prefix,
                        "cwd": str(Path.cwd()), "argv": sys.argv,
                        "model_module": str(Path(dcnet_model_module.__file__).resolve()),
                        "torch_module": str(Path(torch.__file__).resolve()),
                        "torchvision_module": str(Path(torchvision.__file__).resolve()),
                        "torch_version": torch.__version__,
                        "torchvision_version": torchvision.__version__,
                        "device": str(device), "seed": seed, "autocast": False,
                        "tf32": False, "cudnn_deterministic": True,
                        "cudnn_benchmark": False,
                        "matmul_precision": torch.get_float32_matmul_precision(),
                        "cuda_matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_tf32": torch.backends.cudnn.allow_tf32}
    def respond(ok: bool, **extra: Any) -> None:
        payload = dict(runtime_identity)
        payload.update(extra)
        payload["ok"] = ok
        print("DCNET_WORKER:" + json.dumps(payload, sort_keys=True), flush=True)
    def image_batch(paths: Sequence[str]) -> Any:
        rows = []
        for path in paths:
            with Image.open(path) as image:
                rows.append(transform(image.convert("RGB")))
        return torch.stack(rows).to(device)
    def pack(experts: dict[str, Any]) -> Any:
        return torch.cat([experts[name] for name in modalities], dim=-1)
    text_resources: tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]] | None = None
    batch_size = int(first_job["batch_size"])
    if batch_size < 1:
        raise ValueError("DCNet encoder batch_size must be positive")
    def run(job: dict[str, Any]) -> None:
        nonlocal text_resources
        if job["operation"] == "targets":
            pieces = []
            batch_size = int(job["batch_size"])
            if batch_size < 1:
                raise ValueError("DCNet encoder batch_size must be positive")
            for start in range(0, len(job["image_paths"]), batch_size):
                image = image_batch(job["image_paths"][start:start+batch_size])
                with torch.inference_mode():
                    pieces.append(pack(model.get_image_emb({"im_feat": image}, istgt=True)).float().cpu().numpy())
            result = np.concatenate(pieces, axis=0) if pieces else np.empty((0, len(modalities)*DIMENSION), np.float32)
        elif job["operation"] == "queries":
            if text_resources is None:
                text_resources = _load_circus_resources(Path(job["fashioniq_root"]), job["nltk_data"])
            rows, vocab = text_resources
            wv = _load_circus_wv(job["image_paths"], job["texts"], rows, vocab)
            pieces = []
            batch_size = int(job["batch_size"])
            if batch_size < 1:
                raise ValueError("DCNet encoder batch_size must be positive")
            for start in range(0, len(job["image_paths"]), batch_size):
                end = min(start+batch_size, len(job["image_paths"]))
                image = image_batch(job["image_paths"][start:end])
                words = [torch.as_tensor(x[:30], dtype=torch.float32, device=device).reshape(-1, 300) for x in wv[start:end]]
                words = [w if len(w) > 0 else torch.zeros((1, 300), dtype=torch.float32, device=device) for w in words]
                lengths = [len(x) for x in words]
                text = torch.zeros((len(words), max(map(len, words)), 300), device=device)
                for i, values in enumerate(words):
                    text[i, :len(values)] = values
                with torch.inference_mode():
                    source = model.get_image_emb({"im_feat": image}, istgt=False)
                    text_features = model.get_text_emb(text, lengths)
                    composed = model.get_composition(source, text_features)
                    pieces.append(torch.cat((pack(source),pack(text_features),pack(composed)),dim=-1).float().cpu().numpy())
            result = np.concatenate(pieces, axis=0) if pieces else np.empty((0, 3*len(modalities)*DIMENSION), np.float32)
        elif job["operation"] == "score":
            q = np.load(job["query_path"], mmap_mode="r", allow_pickle=False)
            t = np.load(job["target_path"], mmap_mode="r", allow_pickle=False)
            width = len(modalities)*DIMENSION
            def unpack(rows):
                return {name: torch.from_numpy(np.array(rows[:,i*DIMENSION:(i+1)*DIMENSION],copy=True)).to(device)
                        for i,name in enumerate(modalities)}
            capture_path = job.get("capture_components_path")
            result=np.lib.format.open_memmap(job["output_path"],mode="w+",dtype=np.float32,shape=(len(q),len(t)))
            component_paths = None if capture_path is None else (
                Path(capture_path + ".composition.npy"), Path(capture_path + ".correction.npy"))
            comp_memmap = None
            corr_memmap = None
            if component_paths is not None:
                component_paths[0].parent.mkdir(parents=True, exist_ok=True)
            if component_paths is not None:
                comp_memmap = np.lib.format.open_memmap(component_paths[0], mode="w+",
                                                        dtype=np.float32, shape=(len(q), len(t)))
                corr_memmap = np.lib.format.open_memmap(component_paths[1], mode="w+",
                                                        dtype=np.float32, shape=(len(q), len(t)))
            query_batch_size = int(job["batch_size"])
            for query_start in range(0, len(q), query_batch_size):
                query_stop = min(query_start + query_batch_size, len(q))
                query = q[query_start:query_stop]
                source,text,composed=(unpack(query[:,i*width:(i+1)*width]) for i in range(3))
                comp_scores = torch.empty((len(query), len(t)), dtype=torch.float32, device=device)
                corr_scores = torch.empty_like(comp_scores)
                with torch.inference_mode():
                    for target_start in range(0,len(t),512):
                        target_stop=min(target_start+512,len(t))
                        gallery=unpack(t[target_start:target_stop])
                        comp=model.get_score_matrix(trg_embds=gallery,src_embds=composed,subspaces=modalities,
                                                    l2renorm=False,dist=False,val=True)
                        correction=model.get_correction(source,gallery)
                        corr=model.get_score_matrix(trg_embds=correction,src_embds=text,subspaces=modalities,
                                                    l2renorm=False,dist=False,val=True)
                        comp_scores[:,target_start:target_stop]=comp
                        corr_scores[:,target_start:target_stop]=corr
                references = job.get("reference_indices")
                if references is not None:
                    refs = references[query_start:query_stop]
                    _mask_reference_scores(comp_scores, corr_scores, refs,
                                           job["reference_mask_value"])
                if comp_memmap is not None and corr_memmap is not None:
                    comp_memmap[query_start:query_stop] = comp_scores.cpu().numpy()
                    corr_memmap[query_start:query_stop] = corr_scores.cpu().numpy()
                fused=F.log_softmax(comp_scores,dim=-1)+F.log_softmax(corr_scores,dim=-1)
                result[query_start:query_stop]=fused.cpu().numpy()
            result.flush()
            if comp_memmap is not None and corr_memmap is not None:
                comp_memmap.flush()
                corr_memmap.flush()
        else:
            raise ValueError(f"unknown operation {job['operation']!r}")
        if job["operation"] != "score":
            np.save(job["output_path"],result,allow_pickle=False)
    try:
        run(first_job)
        respond(True)
    except Exception as error:
        respond(False,error_type=type(error).__name__,error=str(error), operation=first_job["operation"],
                traceback=traceback.format_exc())
    for line in sys.stdin:
        request=json.loads(line)
        if request.get("stop"): break
        try:
            run(json.loads(Path(request["job_path"]).read_text(encoding="utf-8")))
            respond(True)
        except Exception as error:
            respond(False,error_type=type(error).__name__,error=str(error),
                    operation=json.loads(Path(request["job_path"]).read_text(encoding="utf-8"))["operation"],
                    traceback=traceback.format_exc())
            break
    return 0


def _load_circus_resources(data_root: Path, nltk_data: str) -> tuple[dict[tuple[str, str], np.ndarray], dict[str, Any]]:
    import pickle
    import nltk

    nltk.data.path.insert(0, nltk_data)
    captions = data_root / "captions"
    rows: dict[tuple[str, str], np.ndarray] = {}
    for category in ("dress", "shirt", "toptee"):
        for split in ("train", "val", "test"):
            source = captions / f"cap.{category}.glove.{split}.pkl"
            with source.open("rb") as stream:
                for row in pickle.load(stream):
                    key = (row["candidate"], _canonical_caption(" and ".join(row["captions"])))
                    vector = np.asarray(row["wv"], dtype=np.float32)
                    prior = rows.get(key)
                    if prior is not None and not np.array_equal(prior, vector):
                        raise ValueError(f"ambiguous author-preprocessed DCNet text row: {key[0]} {key[1]!r}")
                    rows[key] = vector
    with (captions / "glove_vecs.pkl").open("rb") as stream:
        return rows, pickle.load(stream)


def _load_circus_wv(image_paths: Sequence[str], texts: Sequence[str],
                    rows: dict[tuple[str, str], np.ndarray], vocab: dict[str, Any]) -> list[np.ndarray]:
    import nltk

    result = []
    for image_path, text in zip(image_paths, texts):
        key = (Path(image_path).stem, _canonical_caption(text))
        cached = rows.get(key)
        candidates = []
        for left, right in _caption_pairs(text):
            tokens = []
            for caption in (left, right):
                tokens.extend(word for word in nltk.tokenize.word_tokenize(
                    caption.replace("-", " ").replace(".", "").lower()) if word in vocab)
            candidates.append(
                np.asarray([vocab[word] for word in tokens], dtype=np.float32).reshape(-1, 300)
                if tokens
                else np.zeros((0, 300), dtype=np.float32)
            )
        if cached is not None:
            if not any(np.array_equal(cached, value) for value in candidates):
                raise ValueError(f"CIRCUS text does not match author DCNet token cache: {key[0]} {text!r}")
            result.append(cached)
        elif len(candidates) == 1:
            result.append(candidates[0])
        else:
            raise ValueError(f"CIRCUS text has no unique author DCNet caption split: {key[0]} {text!r}")
    return result


def _caption_pairs(text: str) -> list[tuple[str, str]]:
    import re
    matches = list(re.finditer(r"\s+and\s+", text, re.IGNORECASE))
    return [(text[:match.start()], text[match.end():]) for match in matches] or [(text, "")]


def _canonical_caption(text: str) -> str:
    import re
    return " ".join(re.sub(r"[.-]", " ", text.lower()).split())


def _status(cls: type[Adapter], config: Any) -> AdapterStatus:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, load_registry, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id(cls.model_id, getattr(config, "registry", None) or load_registry())
    checkpoint = checkpoint_by_id(
        model, getattr(config, "checkpoint_id", None) or model["checkpoint_variants"][0]["checkpoint_id"])
    path = checkpoint_path(cls.model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    variable = model["environment"]["python_env_var"]
    interpreter = resolve_model_interpreter(model, config) or setting(variable)
    reasons = _blockers(model, checkpoint, path, interpreter, config)
    reason = "; ".join(reasons)
    return AdapterStatus(cls.model_id, not reasons, reason or None,
                          tuple(str(item["checkpoint_id"]) for item in model["checkpoint_variants"]),
                         {condition: (not reasons, reason or None) for condition in CONDITIONS},
                         "DCNet context worker encodes source, text, composed and gallery experts; "
                         "score preserves native global gallery log-softmax fusion. Runtime readiness "
                         "reflects verified checkpoint mapping, files, and interpreter resolution.")

ADAPTER = Adapter
if __name__ == "__main__":
    import argparse
    import json
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    args = parser.parse_args()
    raise SystemExit(_worker_main(json.loads(Path(args.worker_job).read_text(encoding="utf-8"))))
