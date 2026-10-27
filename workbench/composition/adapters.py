"""Dispatch registry-declared adapters for exact CIRCUS composition evaluation."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np



class ExactRankUnavailable(RuntimeError):
    """The adapter cannot produce an exact positive rank over the entire eligible gallery."""


class AdapterWorkRootError(RuntimeError):
    """Adapter work_root must be assigned before any worker operation."""


class AdapterWorkerError(RuntimeError):
    """A named per-scope subprocess worker or job failed."""


@dataclass(frozen=True)
class AdapterStatus:
    """Readiness and condition support reported by one composition adapter."""

    model_id: str
    ready: bool
    reason: str | None
    checkpoint_ids: tuple[str, ...]
    per_mode: dict[str, tuple[bool, str | None]]
    notes: str = ""


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
        self.batch_size = 32

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

    def rank(self, queries: Any, targets: Any,
             target_positions: Sequence[Sequence[int] | None] | None = None,
             categories: Sequence[str | None] | None = None) -> list[int | None]:
        """Return exact 1-based best-positive rank, breaking equal scores by gallery order."""
        if target_positions is None:
            raise ExactRankUnavailable("adapter rank requires positive target positions")
        scores = (self.score(queries, targets, categories) if isinstance(targets, Mapping)
                  else self.score(queries, targets))
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

    def close(self) -> None:
        """Release model-owned resources."""


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
