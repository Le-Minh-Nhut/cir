"""Shared subprocess CIRCUS implementation for the two CLIP4Cir variants.

Both registered variants are the same upstream evaluator (``CLIP4Cir/src/validate.py``) with a
different checkpoint shape, so the machinery lives here once and each variant is a small
registry-facing subclass (``model_combiner_rn50x4_noft.py``, ``model_clip4cir_rn50x4_fullft.py``).
That split exists because ``adapters.adapter_class`` requires ``module.ADAPTER.model_id`` to equal
the registry row's ``model_id`` — one module cannot serve two ids.

Native scoring reproduced, not invented (``CLIP4Cir/src/validate.py``):

* ``:41`` — ``index_features = F.normalize(index_features, dim=-1).float()``: gallery features
  are L2-normalized CLIP image features;
* ``:107`` — ``predicted_features = ... F.normalize(batch_predicted_features, dim=-1)``: the
  composed query is the Combiner output, L2-normalized;
* ``:44`` — ``distances = 1 - predicted_features @ index_features.T``, then ``:45``
  ``argsort(distances)``. Ranking by ascending distance is ranking by descending dot product of
  two unit vectors, so ``score()`` returns that dot product directly. There is **no max over
  tokens** and **no reference mask** on this path (unlike the BLIP-2 family), so
  ``reference_eligible = True`` is the upstream behaviour and is MEASURED by the base probe.

Combiner fusion is upstream's own ``combiner.py:50-68`` ``combine_features``: project image and
text, concat, MLP, then ``output_layer(combined) + g*text + (1-g)*image`` with
``g = dynamic_scalar(...)``, normalized. ``encode_queries`` therefore has to run the Combiner —
a CLIP4Cir query vector is NOT a dot of two independently encoded vectors.

TWO BLOCKERS STATED RATHER THAN PAPERED OVER:

1. **Upstream emits aggregate metrics only.** ``validate.py:47-52`` computes R@10/R@50 from
   ``labels``; it never writes per-query ranks. A per-query export path is genuinely new work.
   This adapter can produce features and an exact rank *given* a gallery, but it cannot claim
   upstream parity for the rank artifact — that is why ``status().notes`` says so and why the
   model is not ready regardless of the interpreter.
2. **The noft variant needs base OpenAI CLIP RN50x4 weights** (``clip.load("RN50x4")``,
   ``validate.py:279``) in addition to its staged Combiner state. The registry records that as
   an evaluation asset, not a checkpoint, so this adapter cannot verify its presence from the
   checkpoint registry and reports it in ``notes`` instead of pretending it is checked.
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

#: RN50x4 ``visual.output_dim`` — ``validate.py:281`` reads it from the loaded CLIP model; the
#: registry's Combiner geometry (640->2560, 5120->5120, 5120->640) pins it to 640.
DIMENSION = 640
#: Declared member roles the full-FT bundle must carry. Never a filename literal.
BUNDLE_ROLES = ("combiner", "clip")

class Clip4CirBase(CompositionAdapter):
    """CLIP4Cir Combiner scoring in the variant's own interpreter; cosine, no reference mask."""

    reference_eligible = True
    score_dimension = DIMENSION
    category_scopes: tuple[str | None, ...] = (None,)
    #: Overridden by each registry-facing subclass.
    model_id: str
    #: True when the variant is a two-member bundle resolved by role.
    bundle: bool = False

    def __init__(self, config: Any = None, checkpoint_id: str | None = None) -> None:
        from workbench.backend.operator_config import resolve_config

        self.config = config if config is not None else resolve_config()
        self._model, self._checkpoint, self._interpreter = _resolve(
            type(self).model_id, self.config, checkpoint_id)
        super().__init__(self.config, self._checkpoint["checkpoint_id"])
        self.checkpoint = self._checkpoint
        self.checkpoint_path = Path(self._checkpoint["resolved_path"])
        self.members = _members(self._checkpoint)
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
            raise ValueError("CLIP4Cir gallery is category-agnostic; target categories must all be None")
        return _run(self, "targets", image_paths, (), categories)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """``validate.py:41,44,107``: normalize both sides, one dot product (distance = 1 - dot)."""
        del categories
        query_values = np.asarray(queries, dtype=np.float32)
        target_values = np.asarray(targets, dtype=np.float32)
        if query_values.ndim != 2 or target_values.ndim != 2:
            raise ValueError("CLIP4Cir expects query (N,640) and target (M,640) matrices")
        if query_values.shape[1] != target_values.shape[1]:
            raise ValueError("CLIP4Cir query and target embedding dimensions differ")
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

    def _verify_interpreter_once(self) -> None:
        """Route through the ONE shared environment gate, exactly once per adapter.

        Defined here and overridden by each registry-facing subclass so the gate call is visible
        in the module a reader (and `test_circus_env_gate_omp.py`, which greps per registered
        module) actually opens. There is no bypass: the body is
        `CompositionAdapter.require_verified_interpreter()`, which refuses an unverified
        environment instead of returning a merely-present interpreter.
        """
        if not self._interpreter_verified:
            self.require_verified_interpreter()
            self._interpreter_verified = True

def _stop_worker(adapter: Clip4CirBase) -> None:
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

def _resolve(model_id: str, config: Any, checkpoint_id: str | None
             ) -> tuple[dict[str, Any], dict[str, Any], str | None]:
    from workbench.backend.model_settings import setting
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, model_by_id
    from workbench.backend.runtime import resolve_model_interpreter

    model = model_by_id(model_id)
    checkpoint = checkpoint_by_id(
        model, checkpoint_id or model["checkpoint_variants"][0]["checkpoint_id"])
    checkpoint = dict(checkpoint, resolved_path=str(
        checkpoint_path(model_id, checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)))
    variable = model["environment"]["python_env_var"]
    return model, checkpoint, resolve_model_interpreter(model, config) or setting(variable)

def _members(checkpoint: dict[str, Any]) -> dict[str, Path]:
    """Resolve a bundle's member files by DECLARED ROLE, reusing the one existing resolver.

    ``backend.adapters.models._declared_member_paths`` is the single implementation that reads
    ``bundle_members[*].role`` and cross-checks it against ``required_files``; importing it keeps
    this adapter from spelling a filename. A single-file variant (noft) has no roles: its staged
    artifact *is* the Combiner state, and the CLIP encoder is the stock OpenAI RN50x4 asset.
    """
    if not checkpoint.get("bundle_members"):
        return {"combiner": Path(checkpoint["resolved_path"])}
    from workbench.backend.adapters.models import _declared_member_paths

    return _declared_member_paths(
        Path(checkpoint["resolved_path"]), checkpoint, BUNDLE_ROLES)

def _refusal(adapter: Clip4CirBase) -> str | None:
    """Reasons this adapter may not produce a number, shared by status() and _run()."""
    if adapter._checkpoint.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        return (f"checkpoint mapping is {adapter._checkpoint.get('checkpoint_mapping_status')} "
                f"for {adapter._checkpoint['checkpoint_id']}")
    missing = [str(path) for path in adapter.members.values() if not Path(path).is_file()]
    if missing:
        return f"checkpoint member missing: {missing[0]}"
    return None

def _run(adapter: Clip4CirBase, operation: str, image_paths: Sequence[str],
         texts: Sequence[str], categories: Sequence[str | None]) -> np.ndarray:
    if adapter.work_root is None:
        raise AdapterWorkRootError(f"{adapter.model_id} work_root was not set by runner")
    if not image_paths:
        return np.empty((0, DIMENSION), dtype=np.float32)
    refusal = _refusal(adapter)
    if refusal is not None:
        raise ValueError(refusal)
    if adapter._worker_failed is not None:
        raise AdapterWorkerError(adapter._worker_failed)
    interpreter = adapter._interpreter
    if not interpreter:
        raise RuntimeError(
            f"no interpreter environment resolved for WORKBENCH_CLIP4CIR_PYTHON ({adapter.model_id})")
    if not adapter._interpreter_verified:
        adapter._verify_interpreter_once()
    work_root = Path(adapter.work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    if adapter._job_root is None:
        adapter._job_root = work_root / f".{adapter.model_id}-worker-{uuid.uuid4().hex}"
        adapter._job_root.mkdir()
    work = adapter._job_root
    adapter._worker_counter += 1
    job_path = work / f"job-{adapter._worker_counter}.json"
    output_path = work / f"output-{adapter._worker_counter}.npy"
    scope = categories[0] if categories and len(set(categories)) == 1 else "global"
    job_path.write_text(json.dumps({
        "operation": operation,
        "model_id": adapter.model_id,
        "image_paths": [str(Path(path).resolve()) for path in image_paths],
        "texts": list(texts), "categories": list(categories),
        "output_path": str(output_path),
        "source_root": str(adapter.source_root),
        "members": {role: str(path) for role, path in adapter.members.items()},
        "checkpoint_id": adapter._checkpoint["checkpoint_id"],
        "batch_size": adapter.batch_size,
    }), encoding="utf-8")
    try:
        if adapter._worker is not None and adapter._worker.poll() is not None:
            status = adapter._worker.returncode
            adapter._worker_failed = (
                f"adapter_worker_exited: model={adapter.model_id} scope={scope!r} "
                f"phase={operation} exit_status={status}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        if adapter._worker is None:
            adapter._worker_log = (work / "worker.stderr.log").open("w", encoding="utf-8")
            adapter._worker = subprocess.Popen(
                [interpreter, "-m", "workbench.composition.model_clip4cir",
                 "--worker-job", str(job_path)],
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
                f"adapter_worker_exited: model={adapter.model_id} scope={scope!r} "
                f"phase={operation} exit_status={status}")
            raise AdapterWorkerError(adapter._worker_failed)
        response = json.loads(line[len("CIRCUS_WORKER:"):])
        if not response.get("ok"):
            adapter._worker_failed = (
                f"adapter_worker_job_failed: model={adapter.model_id} scope={scope!r} "
                f"phase={operation} error={response.get('error_type')}: {response.get('error')}")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        features = np.load(output_path, allow_pickle=False)
        if features.dtype != np.float32 or features.shape != (len(image_paths), DIMENSION):
            adapter._worker_failed = (
                f"adapter_worker_invalid_matrix: model={adapter.model_id} scope={scope!r} "
                f"phase={operation} shape={features.shape}/{features.dtype} "
                f"expected=({len(image_paths)}, {DIMENSION})/float32")
            _stop_worker(adapter)
            raise AdapterWorkerError(adapter._worker_failed)
        return features
    except OSError as error:
        status = adapter._worker.poll() if adapter._worker is not None else None
        _stop_worker(adapter)
        adapter._worker_failed = (
            f"adapter_worker_exited: model={adapter.model_id} scope={scope!r} "
            f"phase={operation} exit_status={status}: {error}")
        raise AdapterWorkerError(adapter._worker_failed) from error
    finally:
        job_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

def _status(cls: type[Clip4CirBase], config: Any) -> AdapterStatus:
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
        reasons.append(f"checkpoint mapping unresolved: {checkpoint.get('checkpoint_mapping_status')}")
    if not checkpoint_is_present(path, checkpoint, cls.model_id, config):
        reasons.append(f"checkpoint not staged at {path}")
    else:
        try:
            missing = [str(p) for p in _members(dict(checkpoint, resolved_path=str(path))).values()
                       if not Path(p).is_file()]
            if missing:
                reasons.append(f"checkpoint member missing: {missing[0]}")
        except Exception as error:
            reasons.append(f"checkpoint members unresolved: {type(error).__name__}: {error}")
    if not interpreter:
        reasons.append(f"no interpreter environment resolved for {variable}")
    elif not Path(interpreter).is_file():
        reasons.append(f"model interpreter missing at {interpreter}")
    return AdapterStatus(
        cls.model_id, not reasons, "; ".join(reasons) or None,
        tuple(item["checkpoint_id"] for item in model["checkpoint_variants"]),
        {condition: (True, None) for condition in CONDITIONS},
        "Cosine over normalized CLIP features (validate.py:41,44,107); Combiner fusion is "
        "upstream's combine_features; no reference mask. Upstream emits AGGREGATE METRICS ONLY "
        "(validate.py:47-52) -- per-query rank export is new work, not an upstream parity claim. "
        "The noft variant additionally needs base OpenAI CLIP RN50x4 weights, which the "
        "checkpoint registry does not stage or verify.")

def _worker_main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import torch

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-job", required=True)
    job_path = Path(parser.parse_args(argv).worker_job)

    def reply(ok: bool, **values: Any) -> None:
        print("CIRCUS_WORKER:" + json.dumps({"ok": ok, **values}), flush=True)

    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        source = Path(job["source_root"])
        # The pinned tree's own modules, resolved the way validate.py expects them.
        for entry in (str(source / "src"),):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        import clip
        from combiner import Combiner
        from data_utils import targetpad_transform

        members = {role: Path(path) for role, path in job["members"].items()}
        # `clip.load` needs the stock RN50x4 architecture; `jit=False` matches validate.py:279.
        clip_model, _ = clip.load("RN50x4", device="cuda", jit=False)
        clip_model = clip_model.float().eval()
        feature_dim = clip_model.visual.output_dim
        if feature_dim != DIMENSION:
            raise ValueError(f"CLIP RN50x4 output_dim is {feature_dim}, expected {DIMENSION}")

        clip_state_path = members.get("clip")
        if clip_state_path is not None:
            saved = torch.load(clip_state_path, map_location="cuda")
            clip_model.load_state_dict(saved["CLIP"])

        combiner = Combiner(feature_dim, 2560, 5120).cuda()
        state = torch.load(members["combiner"], map_location="cuda")
        combiner.load_state_dict(state["Combiner"])
        combiner.eval()

        preprocess = targetpad_transform(1.25, clip_model.visual.input_resolution)
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
                    [preprocess(_open_image(path)) for path in chunk]), batches, workers)
                for index, images in enumerate(decoded):
                    images = images.cuda()
                    image_features = clip_model.encode_image(images)
                    if current["operation"] == "queries":
                        first = index * batch_size
                        texts = list(current["texts"][first:first + len(images)])
                        tokens = clip.tokenize(texts, context_length=77).cuda()
                        text_features = clip_model.encode_text(tokens)
                        # `validate.py:105-107`: combine, then L2-normalize the composed query.
                        composed = combiner.combine_features(image_features, text_features)
                        features = composed / composed.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    else:
                        # `validate.py:41`: gallery features are normalized CLIP image features.
                        features = image_features / image_features.norm(
                            dim=-1, keepdim=True).clamp_min(1e-12)
                    outputs.append(features.float().cpu().numpy())
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
