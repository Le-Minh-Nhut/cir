"""Dispatch registry-declared adapters for exact CIRCUS composition evaluation."""
from __future__ import annotations

import importlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def worker_environment() -> dict[str, str]:
    """Inherit environment and pass the effective HF_HOME as an absolute path."""
    import os

    from workbench.backend.model_settings import resolve_path_setting

    environment = os.environ.copy()
    # The shared resolver preserves environment > file precedence and anchors either relative
    # value at the repository root. Workers may chdir into a model source before importing HF.
    # Sibling cache variables pass through untouched and do not suppress HF_HOME.
    root = resolve_path_setting("HF_HOME")
    if root is not None:
        environment["HF_HOME"] = str(root)
    else:
        environment.pop("HF_HOME", None)
    # BLAS thread count is pinned here so the parent and the worker agree and neither spawns a
    # full-machine pool. Only the paths that actually reach BLAS matter: `np.einsum(...,
    # optimize=True)` lowers to a gemm (csmcir, ptha_mtst, conesep) and the `@` bodies do too
    # (encoder, limn). The naive `np.einsum("nd,gqd->ngq")` used by habit/intent/hint/tme has NO
    # BLAS path -- a hand-rolled numpy loop -- which is why these variables do not change its
    # throughput (measured flat at 1/2/4/8/10 threads) and why raising them is not a speedup for
    # those four; their speedup comes from query-axis parallelism.
    threads = str(resolved_decode_workers())
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
        environment.setdefault(variable, threads)
    return environment



class ExactRankUnavailable(RuntimeError):
    """The adapter cannot produce an exact positive rank over the entire eligible gallery."""


class AdapterWorkRootError(RuntimeError):
    """Adapter work_root must be assigned before any worker operation."""


class AdapterWorkerError(RuntimeError):
    """A named per-scope subprocess worker or job failed."""

#: Operator override for the forward-pass batch size. Unset keeps the historical 32, so a run that
#: does not opt in produces the same ``adapter_batch_size`` and therefore the same run id.
BATCH_SIZE_ENV = "WORKBENCH_COMPOSITION_BATCH_SIZE"
BATCH_SIZE_DEFAULT = 32
#: Fraction of total VRAM a batch must be projected to fit inside. A batch size is a request, not a
#: guarantee: refusing an oversized one up front is cheaper than an OOM 90 minutes into a gallery.
VRAM_SAFETY_FRACTION = 0.85

def resolved_batch_size(default: int = BATCH_SIZE_DEFAULT) -> int:
    """Forward-pass batch size from the operator override, or ``default`` when unset.

    Refuses anything that is not a positive integer rather than silently falling back to a
    different protocol setting than the one the operator asked for.
    """
    import os

    raw = os.environ.get(BATCH_SIZE_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{BATCH_SIZE_ENV} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{BATCH_SIZE_ENV} must be a positive integer, got {raw!r}")
    return value

def require_batch_fits(batch_size: int, per_image_bytes: int, total_bytes: int,
                       context: str = "") -> dict[str, int]:
    """Refuse a batch whose projected VRAM would not fit, instead of OOM-ing mid-run.

    ``per_image_bytes`` is MEASURED peak activation memory per image at the batch size actually
    used (see the throughput review under ``docs/reviews/20261108T-throughput``), not a guess. The
    projection is linear in batch size, which is what the measurements show for this encoder.
    """
    projected = int(batch_size) * int(per_image_bytes)
    budget = int(total_bytes * VRAM_SAFETY_FRACTION)
    report = {"batch_size": int(batch_size), "per_image_bytes": int(per_image_bytes),
              "projected_bytes": projected, "budget_bytes": budget,
              "total_bytes": int(total_bytes)}
    if projected > budget:
        where = f" for {context}" if context else ""
        raise ValueError(
            f"batch size {batch_size}{where} would need ~{projected / 2**30:.1f} GiB of VRAM, "
            f"above the {budget / 2**30:.1f} GiB safety budget of {total_bytes / 2**30:.1f} GiB "
            f"({VRAM_SAFETY_FRACTION:.0%} of total); lower {BATCH_SIZE_ENV}")
    return report


#: Worker pool size for image decode. The host is shared with another project, so the default is half
#: the box (AGENTS.md: never take more than half). Decode is I/O plus PIL; measured 2.4k img/s at 8
#: workers and no gain at 10.
DECODE_WORKERS_ENV = "WORKBENCH_DECODE_WORKERS"
DECODE_WORKERS_DEFAULT = min(10, (os.cpu_count() or 8) // 2)

def resolved_decode_workers(default: int = DECODE_WORKERS_DEFAULT) -> int:
    """Decode worker count from the operator override, or ``default`` when unset."""
    import os

    raw = os.environ.get(DECODE_WORKERS_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{DECODE_WORKERS_ENV} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{DECODE_WORKERS_ENV} must be a positive integer, got {raw!r}")
    return value

def prefetch_rows(map_fn: Any, batches: Sequence[Sequence[Any]], workers: int,
                  depth: int | None = None) -> Any:
    """Decode batches in parallel, YIELDING them in submission order.

    Only the DECODE stage is parallel: PIL work releases the GIL, so threads beat processes here and
    the transform objects stay shared. Submission order is preserved, which keeps encoded feature
    rows aligned with the released gallery.

    Genuinely bounded. ``ThreadPoolExecutor.map`` submits EVERY future up front, so completed
    batches pile up inside their futures: decode measured 2089 img/s against forward at ~84 img/s,
    so a ``map`` would run ~25x ahead and hold the whole gallery's tensors (~45 GB for 74381
    images) -- the exact blow-up this function exists to prevent. Instead only ``workers * 2``
    futures are outstanding at any moment: the oldest is yielded, then one more is submitted.
    """
    from collections import deque
    from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

    if workers <= 1 or len(batches) <= 1:
        for batch in batches:
            yield map_fn(batch)
        return

    window = max(1, workers * 2 if depth is None else int(depth))
    pending: deque[Any] = deque()
    remaining = iter(batches)
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        for batch in remaining:
            pending.append(pool.submit(map_fn, batch))
            if len(pending) >= window:
                break
        while pending:
            # `pending` is in submission order; wait only until that OLDEST future is done so
            # order is preserved, then retire it and submit the next batch.
            oldest = pending[0]
            wait([oldest], return_when=FIRST_COMPLETED)
            yield oldest.result()
            pending.popleft()
            for batch in remaining:
                pending.append(pool.submit(map_fn, batch))
                break
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True)

def limit_torch_threads(threads: int | None = None) -> int:
    """Pin torch's intra-op thread count for a data-parallel decode+forward loop.

    MEASURED pathology, not a micro-optimisation: with torch's default 20 intra-op threads the
    transform stage of one 224px image runs at 27 img/s, and at 1 thread the same transform runs at
    650 img/s. The per-image transform releases tiny ops whose thread-launch overhead dwarfs the
    work; parallelism belongs across images (the decode pool), not inside one transform. Pixel
    output is bit-identical between the two settings (verified: max abs delta 0.0).
    """
    import torch

    value = resolved_decode_workers() if threads is None else int(threads)
    torch.set_num_threads(max(1, value))
    return torch.get_num_threads()

@dataclass(frozen=True)
class AdapterStatus:
    """Readiness and condition support reported by one composition adapter."""

    model_id: str
    ready: bool
    reason: str | None
    checkpoint_ids: tuple[str, ...]
    per_mode: dict[str, tuple[bool, str | None]]
    notes: str = ""


def _model_record_of(adapter: Any) -> dict[str, Any] | None:
    """The registry row this adapter was constructed from, whatever attribute name it uses.

    Adapters historically stored it as `model` (HABIT/INTENT/HINT/TME) or `_model_record`
    (ENCODER/LIMN); the CSMCIR/ConeSep/PTHA family stores it as `_model`. One accessor keeps the
    environment gate independent of that naming, so adding `require_verified_interpreter()` to a
    new adapter cannot fail merely because it chose a different private attribute. When no
    attribute carries it, the row is looked up by the class's own `model_id` -- the registry is
    the authority the gate must consult anyway.
    """
    for name in ("model", "_model", "_model_record"):
        value = getattr(adapter, name, None)
        if isinstance(value, dict) and value.get("model_id"):
            return value
    model_id = getattr(type(adapter), "model_id", None)
    if not model_id:
        return None
    try:
        from workbench.backend.registry import load_registry

        registry = getattr(getattr(adapter, "config", None), "registry", None) or load_registry()
        return next(row for row in registry["models"] if row.get("model_id") == model_id)
    except Exception:
        return None


def _environment_gate_reason(model: dict[str, Any], config: Any = None) -> str | None:
    """The reason the execution gate WOULD refuse this model, or None if it would not.

    F1(c): `--plan` used to report READY from artifact/interpreter/CUDA presence alone, while the
    real run then died at `require_verified_model_interpreter` because the Python version or a
    declared dependency did not match. The plan and the gate were checking different contracts, so
    a plan could not be trusted. Calling the SAME probe the gate calls makes them one contract.

    The probe is the advisory pass of the SAME `verify_environment` the gate runs, so plan and gate
    cannot disagree. It executes the interpreter's identity + on-disk dependency inventory once
    (~1 s per model, measured), never the model and never a GPU probe.
    """
    try:
        from workbench.backend.runtime import READY_TIERS, verify_environment

        report = verify_environment(model, probe=True, advisory=True, config=config)
    except Exception as error:  # an unusable contract is itself a reason to refuse
        return f"environment inspection failed: {type(error).__name__}: {error}"
    if report.tier in READY_TIERS:
        return None
    detail = report.reason or "no reason reported"
    return f"environment gate would refuse the run [{report.tier}/{report.status}]: {detail}"


class CompositionAdapter:
    """Full-gallery CIRCUS adapter contract.

    Image inputs are local filesystem paths that this process can read; adapters may be
    subprocess clients and must not assume a shared in-process PIL object. The runner assigns
    ``work_root`` under its run's ``work/`` directory before the first encode call. Adapters
    must raise ``AdapterWorkRootError`` when it is unset, never fall back to a temporary root.
    ``close()`` removes only adapter-owned files below its own subdirectory and never removes
    the runner's work root; the runner owns final cleanup.
    """
    work_root: Path | None = None

    model_id: str
    reference_eligible = True
    category_scopes: tuple[str | None, ...] = (None,)

    def __init__(self, config: Any, checkpoint_id: str | None = None) -> None:
        self.config = config
        self.checkpoint_id = checkpoint_id
        self.model: dict[str, Any] | None = None
        self.checkpoint: dict[str, Any] | None = None
        self.batch_size = resolved_batch_size()

    @classmethod
    def status(cls, config: Any = None) -> AdapterStatus:
        """Return local readiness without downloading or initializing a model."""
        if config is None:
            from workbench.backend.operator_config import resolve_config

            config = resolve_config()
        try:
            from workbench.backend.registry import checkpoint_by_id, checkpoint_path, load_registry

            registry = getattr(config, "registry", None)
            registry = registry if registry is not None else load_registry()
            model = next(item for item in registry["models"] if item["model_id"] == cls.model_id)
            variants = model.get("checkpoint_variants") or []
            checkpoint_ids = tuple(str(item["checkpoint_id"]) for item in variants)
            checkpoint_id = getattr(config, "checkpoint_id", None)
            selected = checkpoint_by_id(model, checkpoint_id) if checkpoint_id else (variants[0] if variants else None)
            if selected is None:
                return AdapterStatus(cls.model_id, False, "no checkpoint is declared", checkpoint_ids, {})
            checkpoint_path_value = checkpoint_path(cls.model_id, selected, config.WORKBENCH_CHECKPOINT_ROOT)
            if not checkpoint_path_value.exists():
                return AdapterStatus(cls.model_id, False, f"checkpoint missing: {checkpoint_path_value}", checkpoint_ids, {})
            return AdapterStatus(cls.model_id, True, None, checkpoint_ids,
                                 {condition: (True, None) for condition in
                                  ("multimodal", "text_only_black_image", "image_only_empty_text")})
        except Exception as error:
            return AdapterStatus(cls.model_id, False, f"{type(error).__name__}: {error}", (), {})

    def mode_supported(self, condition: str) -> tuple[bool, str | None]:
        return True, None

    def require_verified_interpreter(self) -> str:
        """Return this adapter's model interpreter, proving its environment first.

        Every adapter that launches a model subprocess MUST call this before the first launch.
        The gate used to live in only five of the nine adapters, so the other four scored with
        nothing but an advisory environment record. It lives here, once, so a new adapter cannot
        silently skip it: `tests/test_circus_env_gate.py` asserts every `model_*.py` calls it.

        There is deliberately no bypass argument. The only supported way to run under a different
        contract is the registry's own `adapted_runtime` block, which changes what the gate checks
        rather than turning it off, and which badges the run as an adaptation.
        """
        from workbench.backend.runtime import require_verified_model_interpreter

        resolved = _model_record_of(self)
        if resolved is None:
            raise TypeError(f"{type(self).__name__} exposes no registry model record for the "
                            "environment gate; set `model` or `_model_record`")
        interpreter = getattr(self, "interpreter", None)
        if not interpreter:
            interpreter = getattr(self, "_interpreter", None)
        verified = require_verified_model_interpreter(resolved, config=getattr(self, "config", None))
        if interpreter and Path(verified).resolve() != Path(interpreter).resolve():
            raise RuntimeError(
                f"{self.model_id} resolved interpreter changed: {interpreter} -> {verified}")
        self._verified_interpreter = verified
        return verified

    def encode_queries(self, image_paths: Sequence[str | Path], texts: Sequence[str],
                       categories: Sequence[str | None]) -> Any:
        """Encode local readable image paths and adapted text into float32 query features."""
        raise NotImplementedError

    def encode_targets(self, image_paths: Sequence[str | Path],
                       categories: Sequence[str | None]) -> Any:
        """Encode local readable image paths in the single repeated embedding scope."""
        raise NotImplementedError

    def require_work_root(self) -> Path:
        if self.work_root is None:
            raise AdapterWorkRootError(f"{self.model_id} work_root must be assigned before encoding")
        return Path(self.work_root)

    def score(self, queries: Any, targets: Any,
              categories: Sequence[str | None] | None = None) -> np.ndarray:
        """Compute cosine scores; specialized adapters override native candidate scoring.

        ponytail: per-call gallery normalization re-reads full matrix each chunk; cache normalized targets
        on disk only if chunked scoring becomes bandwidth-bound.
        """
        query_values = np.asarray(queries, dtype=np.float32)
        if isinstance(targets, Mapping):
            if categories is None or len(categories) != len(query_values):
                raise ValueError("scoped targets require one category scope per query")
            output = np.empty((len(query_values), len(next(iter(targets.values())))), dtype=np.float32)
            for index, category in enumerate(categories):
                if category not in targets:
                    raise ValueError(f"no target gallery for category scope {category!r}")
                output[index] = self.score(query_values[index:index + 1], targets[category])[0]
            return output
        target_values = np.asarray(targets, dtype=np.float32)
        query_norms = np.linalg.norm(query_values, axis=1, keepdims=True)
        target_norms = np.linalg.norm(target_values, axis=1, keepdims=True)
        query_values = query_values / np.maximum(query_norms, np.finfo(np.float32).tiny)
        target_values = target_values / np.maximum(target_norms, np.finfo(np.float32).tiny)
        return query_values @ target_values.T

    @staticmethod
    def _ranks_from_scores(scores: np.ndarray,
                           target_positions: Sequence[Sequence[int] | None]) -> list[int | None]:
        """Exact 1-based best-positive rank: ``1 + #greater + #earlier-equal`` per gallery order."""
        if scores.ndim != 2 or scores.shape[0] != len(target_positions):
            raise ValueError("score output shape does not match query and target positions")
        ranks: list[int | None] = []
        for row, positions in zip(scores, target_positions):
            if not positions or not np.isfinite(row).all():
                ranks.append(None)
                continue
            valid = [index for index in positions if 0 <= index < row.shape[0]]
            if not valid:
                ranks.append(None)
                continue
            ranks.append(min(1 + int(np.count_nonzero(row > row[index])) +
                             int(np.count_nonzero(row[:index] == row[index])) for index in valid))
        return ranks

    def _score_matrix(self, queries: Any, targets: Any,
                      categories: Sequence[str | None] | None) -> np.ndarray:
        return self.score(queries, targets, categories) if isinstance(targets, Mapping) \
            else self.score(queries, targets)

    def rank_and_topk(self, queries: Any, targets: Any,
                      target_positions: Sequence[Sequence[int] | None] | None,
                      categories: Sequence[str | None] | None, depth: int) -> tuple[Any, Any]:
        """Metric ranks AND the leading ``depth`` candidates from ONE scoring call.

        This is the only supported way to obtain candidates: the score matrix that produced the
        rank also produced the window, so a stored candidate cannot come from a different
        nearest-neighbour method, gallery, preprocessing step or tie policy than the rank it
        accompanies. ``rank`` remains the metric path; both use ``_ranks_from_scores``.
        """
        from workbench.composition.topk import topk_from_scores

        if target_positions is None:
            raise ExactRankUnavailable("adapter rank requires positive target positions")
        scores = self._score_matrix(queries, targets, categories)
        ranks = self._ranks_from_scores(scores, target_positions)
        rows = [topk_from_scores(row, depth) for row in scores]
        return ranks, rows

    def rank(self, queries: Any, targets: Any,
             target_positions: Sequence[Sequence[int] | None] | None = None,
             categories: Sequence[str | None] | None = None) -> list[int | None]:
        """Return exact 1-based best-positive rank, breaking equal scores by gallery order."""
        if target_positions is None:
            raise ExactRankUnavailable("adapter rank requires positive target positions")
        return self._ranks_from_scores(self._score_matrix(queries, targets, categories),
                                       target_positions)

    def close(self) -> None:
        """Release model-owned resources."""

    def measured_reference_eligibility(self, *, probe_dimension: int | None = None) -> dict[str, Any]:
        """MEASURE whether this adapter's own scoring keeps the reference eligible.

        F7 exists because `reference_eligible` was a class-level literal (`= True` in all nine
        adapters) that the runner copied into the artifact as if it were a result. Nothing ever
        executed the claim, so an adapter that masked the reference row -- exactly what the
        per-characteristic protocols do upstream -- would still have written
        `reference_eligible: true` into a summary.

        This probe runs the adapter's OWN `score`+`rank` on a synthetic gallery of
        `probe_dimension`-wide vectors where the reference is the single, unambiguous best match.
        A reference-eligible scorer returns rank 1; a scorer that masks the reference cannot
        return 1 because the reference row is never the best-scoring row.

        It is used ONLY as a measured cross-check of the declared literal, never as a
        replacement for the real rank pass, and it never touches model weights. Adapters whose
        `score` requires a real gallery (a Mapping of scope matrices) or a live worker are
        reported as `unmeasurable` with the reason -- which is the honest answer, and is what
        stops a runner from asserting the value.
        """
        # The scorer decides its own input width (ENCODER 512, the BLIP-2 family 8192 because
        # their score reads a 32x256 token layout). Ask the adapter, then build the probe to fit;
        # a wrong width would be reported as an unmeasurable probe rather than a false verdict.
        dimension = probe_dimension or getattr(self, "score_dimension", None)
        if dimension is None:
            return {"measured": False, "reference_eligible": None,
                    "reason": (f"{type(self).__name__} declares no `score_dimension`, so its "
                               "scoring cannot be probed offline; the runner must not assert a "
                               "reference policy for it"),
                    "method": "score+rank probe"}
        probe = np.zeros((1, dimension), dtype=np.float32)
        probe[0, 0] = 1.0
        targets = np.stack([
            probe[0],                       # reference: identical to the query vector
            np.eye(dimension, dtype=np.float32)[1],
            np.eye(dimension, dtype=np.float32)[2 % dimension],
        ])
        try:
            ranks = self.rank(probe, targets, [[0]])
        except Exception as error:  # a live-worker or scoped adapter cannot answer offline
            return {"measured": False, "reference_eligible": None,
                    "reason": f"{type(error).__name__}: {error}", "method": "score+rank probe"}
        except NotImplementedError as error:  # pragma: no cover - defensive
            return {"measured": False, "reference_eligible": None,
                    "reason": f"{type(error).__name__}: {error}", "method": "score+rank probe"}
        eligible = bool(ranks and ranks[0] == 1)
        return {"measured": True, "reference_eligible": eligible,
                "reason": None if eligible else (
                    f"the reference row is not the top-ranked candidate (rank {ranks!r}); this "
                    "adapter's scoring excludes or masks the reference"),
                "reference_rank": ranks[0] if ranks else None,
                "method": "score+rank on a synthetic gallery whose reference is the exact match"}


def _models(registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if registry is None:
        from workbench.backend.registry import load_registry

        registry = load_registry()
    return registry.get("models", [])


def capability(model_id: str, registry: dict[str, Any] | None = None) -> str | None:
    """Return the registry-declared composition adapter module path, if present."""
    from workbench.backend.registry import model_by_id

    try:
        model = model_by_id(model_id, registry)
    except KeyError:
        return None
    value = model.get("composition_adapter")
    return value if isinstance(value, str) and value else None


def adapter_class(model_id: str, registry: dict[str, Any] | None = None) -> type[CompositionAdapter]:
    """Import the registry-declared module and validate its adapter class identity."""
    module_name = capability(model_id, registry)
    if module_name is None:
        raise ValueError(f"model {model_id!r} has no composition_adapter registry module")
    try:
        module = importlib.import_module(module_name)
    except Exception as error:
        raise ImportError(f"cannot import composition adapter module {module_name!r}: {error}") from error
    adapter = getattr(module, "ADAPTER", None)
    if not isinstance(adapter, type) or not issubclass(adapter, CompositionAdapter):
        raise TypeError(f"composition adapter module {module_name!r} must export ADAPTER subclassing CompositionAdapter")
    from workbench.backend.registry import model_by_id

    model = model_by_id(model_id, registry)
    if adapter.model_id != model["model_id"]:
        raise ValueError(f"composition adapter module {module_name!r} declares model_id {adapter.model_id!r}, expected {model_id!r}")
    return adapter


def adapter_status(model_id: str, registry: dict[str, Any] | None = None,
                   config: Any = None) -> AdapterStatus:
    try:
        return adapter_class(model_id, registry).status(config)
    except Exception as error:
        return AdapterStatus(model_id, False, str(error), (), {})


def all_adapter_statuses(registry: dict[str, Any] | None = None) -> tuple[AdapterStatus, ...]:
    return tuple(adapter_status(model["model_id"], registry) for model in _models(registry))
