"""LIMN FashionIQ adapter and model-interpreter worker.

Runner sets ``work_root`` after construction, before encoding; encode methods refuse with
``AdapterWorkRootError`` when unset. Each instance owns a unique
``<work_root>/limn/worker-*/`` directory, and ``close()``/``restart()`` remove only that
directory, never the run root. A query scope selects both its whole-model checkpoint and the
matching embedding space for the entire released gallery. Target matrices from scopes are never
mixed, candidates are never category-filtered, and checkpoint choice never consults positive targets.
Persistent worker caches one model per scope.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from workbench.backend.model_settings import setting
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import (
    checkpoint_by_id,
    checkpoint_is_present,
    checkpoint_path,
    model_by_id,
)
from workbench.backend.runtime import resolve_model_interpreter
from workbench.composition.adapters import (
    AdapterStatus,
    AdapterWorkerError,
    CompositionAdapter,
    ExactRankUnavailable,
)
from workbench.composition.inputs import CONDITIONS

BUNDLE_ID = "base_iter0_all_categories"


class CategoryScopeError(ValueError):
    """Scope refusal with stable caller-batch indices and per-index missing/out-of-scope reasons."""

    def __init__(self, reason: str, batch_indices: Sequence[int], detail: str,
                 reasons_by_index: Mapping[int, str] | None = None) -> None:
        self.model_id = "limn"
        self.reason = reason
        self.batch_indices = tuple(batch_indices)
        self.reasons_by_index = dict(reasons_by_index or {})
        super().__init__(f"limn {reason} at batch indices {self.batch_indices}: {detail}")


class Adapter(CompositionAdapter):
    """Encode a full gallery separately in each of LIMN's three learned category spaces."""

    model_id = "limn"
    category_scopes = ("dress", "shirt", "toptee")
    reference_eligible = True

    def __init__(self, config: Any, checkpoint_id: str | None = None) -> None:
        config = config or resolve_config()
        super().__init__(config, checkpoint_id)
        self._model_record = model_by_id(self.model_id)
        self._bundle_id = checkpoint_id or BUNDLE_ID
        self._bundle = checkpoint_by_id(self._model_record, self._bundle_id)
        self._members = {member["checkpoint_id"]: member
                         for member in self._bundle.get("bundle_members", ())}
        variable = self._model_record["environment"]["python_env_var"]
        self.interpreter = resolve_model_interpreter(self._model_record, config) or setting(variable)
        self.source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / self._model_record["source_dir"]
        self.dataset_root = Path(config.FASHIONIQ_ROOT)
        self._counter = 0
        self._job_dir: Path | None = None
        self._worker: subprocess.Popen[str] | None = None
        self._worker_log: Any | None = None
    @classmethod
    def status(cls, config: Any | None = None) -> AdapterStatus:
        """Check each bundle member, configured interpreter, its CUDA runtime, and bundle mapping."""
        config = config or resolve_config()
        model = model_by_id(cls.model_id)
        bundle = checkpoint_by_id(model, BUNDLE_ID)
        checkpoint_ids = tuple(bundle.get("required_checkpoint_ids", ()))
        variants = {item["checkpoint_id"]: item for item in model["checkpoint_variants"]}
        present: dict[str, bool] = {}
        for checkpoint_id in checkpoint_ids:
            member = checkpoint_by_id(model, checkpoint_id)
            path = checkpoint_path(cls.model_id, member, config.WORKBENCH_CHECKPOINT_ROOT)
            present[checkpoint_id] = checkpoint_is_present(
                path, member, model_id=cls.model_id, config=config)
        variable = model["environment"]["python_env_var"]
        setting_value = setting(variable)
        interpreter = resolve_model_interpreter(model, config) or setting_value
        interpreter_present = bool(interpreter and Path(interpreter).is_file())
        cuda_present = False
        if interpreter_present:
            try:
                probe = subprocess.run(
                    [str(interpreter), "-c",
                     "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"],
                    capture_output=True, text=True, check=True)
                cuda_present = probe.stdout.strip().split()[-1] == "True"
            except (OSError, subprocess.SubprocessError, IndexError):
                pass
        mapping = bundle.get("mapping_status")
        members_mapped = all(variants[item].get("checkpoint_mapping_status") == "VERIFIED_METADATA"
                             for item in checkpoint_ids)
        mapping_present = mapping == "VERIFIED_METADATA" and members_mapped
        checks = (f"checkpoints={'passed' if all(present.values()) and present else 'failed'} "
                  f"({present}); interpreter={'passed' if interpreter_present else 'failed'} "
                  f"({variable}={'set' if setting_value else 'unset'}); "
                  f"GPU={'passed' if cuda_present else 'failed'}; "
                  f"mapping={'passed' if mapping_present else 'failed'} ({mapping})")
        reason = None
        if not checkpoint_ids or not all(present.values()):
            missing = [item for item, is_present in present.items() if not is_present]
            reason = f"LIMN checkpoint bundle incomplete or integrity check failed: {missing or BUNDLE_ID}"
        elif not interpreter_present:
            reason = f"model interpreter unavailable: {variable} is {'set but missing' if setting_value else 'unset'}"
        elif not cuda_present:
            reason = "CUDA unavailable in LIMN model interpreter; native forward requires CUDA"
        elif not mapping_present:
            reason = f"checkpoint bundle mapping unresolved: {mapping}"
        return AdapterStatus(
            model_id=cls.model_id, ready=reason is None, reason=reason,
            checkpoint_ids=checkpoint_ids,
            per_mode={condition: (True, None) for condition in CONDITIONS}, notes=checks)

    def _run_job(self, phase: str, image_paths: Sequence[str | Path],
                 texts: Sequence[str] = (), categories: Sequence[str | None] = ()) -> np.ndarray:
        work_root = self.require_work_root()
        if not self.interpreter or not Path(self.interpreter).is_file():
            variable = self._model_record["environment"]["python_env_var"]
            raise RuntimeError(f"model interpreter unavailable: {variable} is unset or missing")
        paths = [str(Path(path).expanduser().resolve()) for path in image_paths]
        missing = [path for path in paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"LIMN input image missing: {missing[0]}")
        adapter_root = work_root / self.model_id
        adapter_root.mkdir(parents=True, exist_ok=True)
        if self._job_dir is None:
            self._job_dir = adapter_root / f"worker-{uuid.uuid4().hex}"
            self._job_dir.mkdir()
        self._counter += 1
        job_path = self._job_dir / f"job-{self._counter}.json"
        output_path = self._job_dir / f"output-{self._counter}.npy"
        scope = ",".join(sorted({str(value) for value in categories})) or "global"
        payload = {
            "phase": phase, "image_paths": paths, "texts": list(texts),
            "categories": list(categories), "output_path": str(output_path),
            "checkpoint_root": str(self.config.WORKBENCH_CHECKPOINT_ROOT),
            "bundle_id": self._bundle_id, "source_root": str(self.source_root),
            "dataset_root": str(self.dataset_root), "batch_size": max(1, int(self.batch_size)),
        }
        job_path.write_text(json.dumps(payload), encoding="utf-8")
        if self._worker is not None and self._worker.poll() is not None:
            status = self._worker.returncode
            self.restart()
            raise AdapterWorkerError(
                f"adapter_worker_exited: model=limn scope={scope} phase={phase} exit_status={status}")
        if self._worker is None:
            self._worker_error = self._job_dir / "worker.stderr.log"
            self._worker_log = self._worker_error.open("w", encoding="utf-8")
            self._worker = subprocess.Popen(
                [self.interpreter, "-m", "workbench.composition.model_limn",
                 "--worker-job", str(job_path)], cwd=self.config.CIR_REPO_ROOT,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._worker_log,
                text=True, bufsize=1)
        else:
            assert self._worker.stdin is not None
            self._worker.stdin.write(json.dumps({"job_path": str(job_path)}) + "\n")
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
                f"adapter_worker_exited: model=limn scope={scope} phase={phase} exit_status={status}")
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            job_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
            self.restart()
            raise AdapterWorkerError(
                f"adapter_worker_job_failed: model=limn scope={scope} phase={phase} "
                f"error={response.get('error_type')}: {response.get('error')}")
        result = np.load(output_path, allow_pickle=False).astype(np.float32, copy=False)
        if result.ndim != 2 or result.shape != (len(paths), 768):
            raise AdapterWorkerError(
                f"adapter_worker_invalid_matrix: model=limn scope={scope} phase={phase} "
                f"shape={result.shape} expected=({len(paths)}, 768)")
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        return result

    def encode_queries(self, images: Sequence[str | Path], texts: Sequence[str],
                       categories: Sequence[str | None]) -> np.ndarray:
        self.require_work_root()
        if len(images) != len(texts) or len(images) != len(categories):
            raise ValueError("LIMN query image, text, and category counts must match")
        invalid = [(index, category, "missing_category" if category is None else "category_outside_scopes")
                   for index, category in enumerate(categories)
                   if category not in self.category_scopes]
        if invalid:
            reasons = {index: reason for index, _category, reason in invalid}
            raise CategoryScopeError(
                "category_scope_invalid", tuple(reasons),
                f"query categories missing or outside declared scopes: {invalid!r}", reasons)
        return self._run_job("queries", images, texts, categories)

    def encode_targets(self, images: Sequence[str | Path],
                       categories: Sequence[str | None]) -> np.ndarray:
        self.require_work_root()
        if len(images) != len(categories):
            raise CategoryScopeError("target_scope_count_mismatch", (),
                                     "LIMN target image and category counts must match")
        scopes = set(categories)
        if len(scopes) != 1:
            raise CategoryScopeError("mixed_target_scopes", (),
                                     f"LIMN target call requires one repeated scope; got {scopes!r}")
        category = next(iter(scopes))
        if category not in self.category_scopes:
            reason = "missing_category" if category is None else "category_outside_scopes"
            raise CategoryScopeError(reason, (), f"LIMN target scope {category!r} is not declared")
        return self._run_job("targets", images, categories=categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Controller-side chunked normalized-dot scores in each query's matching space."""
        if not isinstance(targets, Mapping):
            raise ExactRankUnavailable("LIMN score requires scope-to-full-gallery matrices")
        if categories is None or len(categories) != len(queries):
            raise CategoryScopeError("missing_query_categories", tuple(range(len(queries))),
                                     "LIMN score requires one query scope per query")
        invalid = [(index, category, "missing_category" if category is None else "category_outside_scopes")
                   for index, category in enumerate(categories)
                   if category not in self.category_scopes or category not in targets]
        if invalid:
            reasons = {index: reason for index, _category, reason in invalid}
            raise CategoryScopeError(
                "category_scope_invalid", tuple(reasons),
                f"LIMN score query scopes lack matching gallery matrices: {invalid!r}", reasons)
        gallery_sizes = {np.shape(matrix)[0] for matrix in targets.values()}
        if len(gallery_sizes) != 1:
            raise CategoryScopeError("inconsistent_gallery_sizes", (),
                                     "LIMN scope matrices must each contain the same full gallery")
        gallery_size = next(iter(gallery_sizes), 0)
        scores = np.empty((len(queries), gallery_size), dtype=np.float32)
        for row, category in enumerate(categories):
            query = np.asarray(queries[row], dtype=np.float32)
            query_norm = np.linalg.norm(query)
            if query_norm:
                query = query / query_norm
            gallery = targets[category]
            for start in range(0, gallery_size, 8192):
                stop = min(start + 8192, gallery_size)
                scores[row, start:stop] = query @ np.asarray(gallery[start:stop]).T
        return scores

    def rank(self, queries: Any, targets: Any,
             target_positions: Sequence[Sequence[int] | None] | None = None,
             categories: Sequence[str | None] | None = None) -> list[int | None]:
        if target_positions is None:
            raise ExactRankUnavailable("LIMN exact rank requires positive target positions")
        scores = self.score(queries, targets, categories)
        if len(target_positions) != len(scores):
            raise ValueError("target position count does not match query count")
        ranks: list[int | None] = []
        for row, positions in zip(scores, target_positions):
            if not positions or not np.isfinite(row).all():
                ranks.append(None)
                continue
            valid = [position for position in positions if 0 <= position < row.size]
            if not valid:
                ranks.append(None)
                continue
            ranks.append(min(1 + int(np.count_nonzero(row > row[index]))
                             + int(np.count_nonzero(row[:index] == row[index])) for index in valid))
        return ranks

    def close(self) -> None:
        if self._worker is not None:
            if self._worker.stdin is not None:
                try:
                    self._worker.stdin.write(json.dumps({"stop": True}) + "\n")
                    self._worker.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass
            try:
                self._worker.wait(timeout=30)
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
        """Kill worker, unload cached scope models, and reload on next encode call."""
        self.close()

def _worker_main(job_path: Path) -> None:
    """Cache one verified LIMN whole model per scope while serving path-based jobs."""
    import importlib
    import sys

    import numpy as np
    import torch
    import open_clip
    from PIL import Image

    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.replay.common import load_trusted_whole_model

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        model_record = model_by_id("limn")
        bundle = checkpoint_by_id(model_record, job["bundle_id"])
        members = {member["checkpoint_id"]: member for member in bundle.get("bundle_members", ())}
        source_path = Path(job["source_root"]) / "LIMN"
        if str(source_path) not in sys.path:
            sys.path.insert(0, str(source_path))
        native_model = importlib.import_module("model")
        if Path(native_model.__file__).resolve() != (source_path / "model.py").resolve():
            raise ImportError(f"refusing non-pinned LIMN model module: {native_model.__file__}")
    except Exception as error:
        reply(False, error_type=type(error).__name__, error=str(error))
        return

    models: dict[str, Any] = {}
    transforms: dict[str, Any] = {}
    dataset_ids = {"dress": 0, "shirt": 1, "toptee": 2}

    def load_scope(category: str, current: dict[str, Any]) -> None:
        if category in models:
            return
        checkpoint_id = f"base_iter0_{category}"
        checkpoint = checkpoint_by_id(model_record, checkpoint_id)
        member = members[checkpoint_id]
        member_path = checkpoint_path("limn", checkpoint, Path(current["checkpoint_root"]))
        model = load_trusted_whole_model(member_path, checkpoint["expected_sha256"], torch)
        if not isinstance(model, native_model.LIMN):
            raise TypeError(f"LIMN {category} checkpoint does not deserialize to pinned model.LIMN")
        model.cuda().eval()
        visual = model.backbone.clip.visual
        models[category] = model
        transforms[category] = open_clip.image_transform(
            visual.image_size, is_train=False, mean=visual.image_mean, std=visual.image_std)

    def run_one(current: dict[str, Any]) -> None:
        try:
            paths, texts, categories = current["image_paths"], current["texts"], current["categories"]
            for category in dict.fromkeys(categories):
                load_scope(category, current)
            batch_size = max(1, int(current["batch_size"]))
            outputs: list[np.ndarray | None] = [None] * len(paths)
            start = 0
            while start < len(paths):
                category = categories[start]
                stop = start + 1
                while stop < len(paths) and categories[stop] == category:
                    stop += 1
                model = models[category]
                for batch_start in range(start, stop, batch_size):
                    batch_stop = min(batch_start + batch_size, stop)
                    images = []
                    for path in paths[batch_start:batch_stop]:
                        with Image.open(path) as image:
                            images.append(transforms[category](image.copy()))
                    batch = torch.stack(images).float().cuda()
                    dataset_type = torch.ones(len(images)).cuda() * dataset_ids[category]
                    with torch.no_grad():
                        if current["phase"] == "queries":
                            encoded = model.compose_query(
                                batch, texts[batch_start:batch_stop], dataset_type)[0]
                        else:
                            encoded = model.compose_target(batch, dataset_type)[0]
                    values = encoded.detach().float().cpu().numpy()
                    for offset, value in enumerate(values):
                        outputs[batch_start + offset] = value
                start = stop
            result = (np.stack(outputs).astype(np.float32, copy=False) if outputs
                      else np.empty((0, 768), dtype=np.float32))
            np.save(current["output_path"], result, allow_pickle=False)
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


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker-job":
        raise SystemExit("usage: python -m workbench.composition.model_limn --worker-job JOB.json")
    _worker_main(Path(sys.argv[2]))


ADAPTER = Adapter
