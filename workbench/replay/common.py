from __future__ import annotations

import hashlib
import inspect
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

CATEGORIES = ("dress", "shirt", "toptee")


@dataclass(frozen=True)
class ReplayResult:
    model_id: str
    checkpoint_id: str
    category: str
    protocol: str
    r1: float | None = None
    r10: float | None = None
    r50: float | None = None
    complete: bool = False
    provenance: dict[str, Any] = field(default_factory=dict)
    #: PARITY 1 (native instrumentation parity) mismatches between the native aggregate and the
    #: aggregate reconstructed from observed per-query rankings. Empty means the observation is
    #: the native ranking. Non-empty means QUERY_EXPORT_PARITY_MISMATCH: the native aggregate
    #: stands, and no per-query artifact may be published for this category.
    parity_mismatches: tuple[str, ...] = ()
    #: True only when per-query rankings were observed AND matched the native hit counts exactly.
    per_query_published: bool = False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trusted_whole_model(path: Path, expected_sha256: str | None, torch_module: Any) -> Any:
    """Unpickle only a whole model whose exact author hash was verified."""
    if expected_sha256 is None:
        raise ValueError(f"no author-verified SHA-256 for whole-model checkpoint: {path}")
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"whole-model checkpoint missing or empty: {path}")
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise ValueError(f"whole-model checkpoint SHA-256 mismatch: {path}")
    # `weights_only=False` is allowed only after matching trusted author bytes.
    if "weights_only" in inspect.signature(torch_module.load).parameters:
        return torch_module.load(path, map_location="cpu", weights_only=False)
    return torch_module.load(path, map_location="cpu")


def source_checkout_status(source_root: Path, expected_commit: str) -> tuple[bool, str | None, str | None]:
    """The runtime's answer, so the two entry points cannot disagree.

    This used to be a second implementation of the same question, and the copy silently lost the
    dirtiness check: it compared the pin and nothing else, so a *modified* source file was accepted
    by the replay path while `runtime.source_clean_and_pinned` refused it. Delegating means one
    implementation covers both layouts — the vendored digest manifest, and a nested Git checkout
    including "is it dirty" — and `test_replay_and_runtime_agree_that_bytecode_is_not_a_source_change`
    holds the two together.
    """
    from workbench.backend.runtime import source_clean_and_pinned

    return source_clean_and_pinned({"upstream_commit_sha": expected_commit}, source_root)


def source_pin_status(source_root: Path, expected_commit: str) -> tuple[bool, str | None]:
    ok, _, reason = source_checkout_status(source_root, expected_commit)
    return ok, reason


@contextmanager
def isolated_cache_root(dataset_root: Path, directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image")) -> Iterator[Path]:
    """Use native caches in a fresh temp root, symlinking immutable inputs only."""
    with tempfile.TemporaryDirectory(prefix="cir-replay-cache-") as temporary:
        cache_root = Path(temporary)
        for name in directory_names:
            source = dataset_root / name
            if not source.exists():
                raise FileNotFoundError(f"native replay input missing: {source}")
            (cache_root / name).symlink_to(source.resolve(), target_is_directory=source.is_dir())
        yield cache_root


def macro_aggregate(results: list[ReplayResult]) -> dict[str, float | None]:
    """Compute source-compatible arithmetic category macro metrics, never query-weighted."""
    if len(results) != len(CATEGORIES) or not all(result.complete for result in results):
        return {"macro_r10": None, "macro_r50": None, "macro_mean": None, "complete": False}
    if tuple(result.category for result in results) != CATEGORIES:
        return {"macro_r10": None, "macro_r50": None, "macro_mean": None, "complete": False}
    if any(result.r10 is None or result.r50 is None for result in results):
        return {"macro_r10": None, "macro_r50": None, "macro_mean": None, "complete": False}
    r10 = sum(result.r10 for result in results) / len(CATEGORIES)
    r50 = sum(result.r50 for result in results) / len(CATEGORIES)
    return {"macro_r10": r10, "macro_r50": r50, "macro_mean": (r10 + r50) / 2, "complete": True}
