from __future__ import annotations

import hashlib
import inspect
import os
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

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


def source_checkout_status(source_root: Path, expected_commit: str,
                           model: dict[str, Any] | None = None) -> tuple[bool, str | None, str | None]:
    """The runtime's answer, so the two entry points cannot disagree.

    This used to be a second implementation of the same question, and the copy silently lost the
    dirtiness check: it compared the pin and nothing else, so a *modified* source file was accepted
    by the replay path while `runtime.source_clean_and_pinned` refused it. Delegating means one
    implementation covers both layouts — the vendored digest manifest, and a nested Git checkout
    including "is it dirty" — and `test_replay_and_runtime_agree_that_bytecode_is_not_a_source_change`
    holds the two together.

    ``model`` is the registry entry when the caller has it. Passing it matters for a checkout under an
    **approved patch**: the declaration that makes the patched bytes acceptable lives on the model, so
    a bare commit string cannot express it, and a patched CSMCIR checkout was reported dirty here
    while the runtime accepted it. Callers holding only the commit still work; they simply cannot
    authorize a patch, which is the fail-closed direction.
    """
    from workbench.backend.runtime import source_clean_and_pinned

    if model is None:
        model = {"upstream_commit_sha": expected_commit}
    return source_clean_and_pinned(model, source_root)


def source_pin_status(source_root: Path, expected_commit: str,
                      model: dict[str, Any] | None = None) -> tuple[bool, str | None]:
    """Pin check for replay callers; ``model`` authorizes an approved patch when supplied."""
    ok, _, reason = source_checkout_status(source_root, expected_commit, model)
    return ok, reason


#: The parent sets this to a directory the whole RUN shares, so the aggregate and the per-query
#: observer construct the native dataset cache once between them instead of once each. See
#: `reusable_cache_root` for why reuse is sound and what still refuses a bad cache.
CACHE_ROOT_ENV = "WORKBENCH_REPLAY_CACHE_ROOT"


@contextmanager
def run_cache_root(dataset_root: Path,
                   directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image")) -> Iterator[Path]:
    """The cache root for THIS run: the shared one when the parent named it, else a fresh temp root.

    A replay invoked directly (an operator running the wrapper by hand, or a test) has no parent to
    extend its lifetime, so it keeps the fully isolated behaviour.
    """
    shared = os.environ.get(CACHE_ROOT_ENV, "").strip()
    if shared:
        with reusable_cache_root(dataset_root, Path(shared), directory_names) as root:
            yield root
        return
    with isolated_cache_root(dataset_root, directory_names) as root:
        yield root


@contextmanager
def reusable_cache_root(dataset_root: Path, cache_root: Path,
                        directory_names: tuple[str, ...] = ("captions", "image_splits", "resized_image")) -> Iterator[Path]:
    """A cache root that survives between the processes of one run.

    `isolated_cache_root` deliberately yields a FRESH directory every time, so the native
    constructor always takes its cold path and writes its own cache. That is right for isolation and
    wrong for cost: the aggregate and the per-query observer build the SAME three-category cache --
    measured at an 11.87 GB peak for one construction -- so a run needs room for two or for the first
    one to be deleted before the second starts.

    On a host where one cache fits and two do not, neither of those is acceptable. This root is
    scoped to ONE run instead: it is created under the run's own directory, so concurrent runs still
    cannot see each other's caches, and the second construction reuses the first's files. The
    constructor's own cold/warm branch decides: with `fashion_iq_data.json` present it loads the
    pickles rather than rebuilding them.

    Reuse is only sound because the inputs are pinned -- the seed, the transform and the annotation
    files are the same for both children -- and because the callback that reads the cache is
    `assert_alignment`, which refuses a cache whose identities do not match the annotations. A stale
    or foreign cache is a refusal, not a silently wrong ranking.
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    for name in directory_names:
        source = dataset_root / name
        if not source.exists():
            raise FileNotFoundError(f"native replay input missing: {source}")
        link = cache_root / name
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=source.is_dir())
    yield cache_root


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


def constructor_input_blockers(dataset_root: Path, message: Callable[[Path], str],
                               categories: tuple[str, ...] = CATEGORIES,
                               train_captions: bool = True) -> list[str]:
    """The inputs the native FashionIQ constructor opens, independent of the evaluated categories.

    Every wrapper in this family calls `datasets.FashionIQ(...)`, whose `__init__` runs
    `train_init_process()` and then `get_test_data()` for **all three** categories
    (`HINT/datasets.py:35-47,59-73`) with no category argument anywhere in its signature. So
    `--category dress` selects what is *scored*, never what is *opened*: a root that satisfies the
    dress files alone loads a checkpoint and only then dies on `cap.shirt.train.json`.

    `isolated_cache_root` makes this reachable on every run -- it yields an empty cache root, so
    `fashion_iq_data.json` never exists there and the constructor always takes the cold path. The
    evaluated subset and the constructor's input set are therefore two different questions, and this
    is the second one; the caller keeps asking the first about the categories it will score.

    `message` is the caller's own phrasing for one missing path, so the two questions produce
    identical text for the paths both of them cover and the caller can deduplicate.
    """
    blockers: list[str] = []
    for category in categories:
        required = [f"captions/cap.{category}.val.json",
                    f"captions/correction_dict_{category}.json",
                    f"image_splits/split.{category}.val.json"]
        if train_captions:
            required.append(f"captions/cap.{category}.train.json")
        for relative in required:
            path = dataset_root / relative
            if not path.is_file():
                blockers.append(message(path))
        image_directory = dataset_root / "resized_image" / category
        if not image_directory.is_dir():
            blockers.append(message(image_directory))
    return blockers


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


#: Fixed seed for the dataset constructors whose query pipeline is STOCHASTIC.
#:
#: The iLearn-family constructors bake the transformed query image INTO the dataset at construction
#: time: `datasets.py::get_test_data` calls `get_img(..., stage=0)`, and the replay supplies
#: `transform = [preprocess_train, preprocess_val]`, so the query inputs are drawn through
#: `preprocess_train` -- whose first step is `RandomResizedCrop`. Every construction therefore draws
#: a different crop, and a per-query export that rebuilds the dataset in its own process observes a
#: DIFFERENT input population than the aggregate it is supposed to reconstruct.
#:
#: Measured on this host with the pinned transform and the real annotations: two constructions of the
#: dress category produced 3994 differing query tensors out of 4034, while every gallery input
#: (stage=1, the deterministic validation transform) was byte-identical. That is the whole cause of
#: the ENCODER artifact-versus-aggregate disagreement, and it is a preprocessing-population
#: difference, not the GPU nondeterminism it was previously attributed to: normalization, the dot
#: product and `argsort` all happen in NumPy after the features return to CPU.
#:
#: Seeding global RNG immediately before construction makes each construction reproduce the same
#: draws. It must be applied at the construction site and not merely at process start, because the
#: constructor consumes randomness from the same global generators that a model load or a CUDA
#: context can also touch first.
DATASET_CONSTRUCTION_SEED = 0


@contextmanager
def seeded_dataset_construction(seed: int = DATASET_CONSTRUCTION_SEED) -> Iterator[None]:
    """Make a stochastic dataset construction reproducible across processes.

    Seed state is saved and restored, so seeding a construction does not leak into whatever the
    caller does next (a differently seeded evaluation, or a caller that seeded deliberately).
    """
    import random

    python_state = random.getstate()
    numpy_state = None
    try:
        import numpy as np

        numpy_state = np.random.get_state()
    except Exception:
        np = None  # type: ignore[assignment]

    torch_state = None
    try:
        import torch

        torch_state = torch.random.get_rng_state()
    except Exception:
        torch = None  # type: ignore[assignment]

    random.seed(seed)
    if numpy_state is not None:
        np.random.seed(seed)
    if torch_state is not None:
        torch.manual_seed(seed)
    try:
        yield
    finally:
        random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)
