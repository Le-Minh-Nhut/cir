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
    if not source_root.is_dir():
        return False, None, "source is not a readable Git checkout"
    try:
        top_level = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=source_root,
                                   check=True, capture_output=True, text=True).stdout.strip()
        if Path(top_level).resolve() != source_root.resolve():
            return False, None, "source path is not a Git checkout root"
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source_root, check=True,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=source_root, check=True,
                               capture_output=True, text=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, None, "source is not a readable Git checkout"
    if dirty:
        return False, head, "source checkout is dirty"
    if head != expected_commit:
        return False, head, f"source pin mismatch: expected {expected_commit}, found {head}"
    return True, head, None


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
