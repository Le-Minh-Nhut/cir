"""Run opt-in exact-rank CIRCUS evaluation with disk-backed full-gallery features."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import shutil
import sys
import inspect
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import fcntl
import numpy as np
from PIL import Image

from workbench.composition.adapters import AdapterWorkerError, ExactRankUnavailable
from workbench.composition.inputs import CONDITIONS, adapt_query_inputs
from workbench.composition.data import PAPER_STATED_GALLERY_IMAGES, RELEASED_GALLERY_IMAGES, sha256_file


class BackfillRankDisagreement(RuntimeError):
    """A backfilled target rank disagreed with the rank the original run recorded.

    Raised instead of writing the window: the disagreement proves the backfill did not run under
    the same effective scoring policy as the original pass (BLAS thread count is an arithmetic
    input, not only a performance knob), so its candidate window does not belong to that
    measurement. The caller must refuse the export rather than publish it.
    """


@dataclass(frozen=True)
class ModelSpec:
    """One model/checkpoint selected by the existing workbench policy."""

    model_id: str
    checkpoint_id: str
    adapter_factory: Callable[..., Any]
    registry_model: dict[str, Any]
    checkpoint_identity: str = ""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_composition_destination(artifacts_root: Path, destination: Path) -> Path:
    """Refuse every path outside the isolated composition artifact namespace."""
    artifacts = Path(artifacts_root).expanduser().resolve()
    composition = (artifacts / "composition").resolve()
    target = Path(destination).expanduser().resolve()
    if not _inside(composition, artifacts) or not _inside(target, composition):
        raise ValueError(f"composition output destination must be inside {composition}; refused {target}")
    return target


def _component(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"unsafe output path component: {value!r}")
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rank_payload_is_valid(ranks: Any, conditions: Sequence[str], qids: set[str], gallery_count: int) -> bool:
    if not isinstance(ranks, dict) or set(ranks) != set(conditions):
        return False
    return all(isinstance(ranks[condition], dict) and set(ranks[condition]) == qids and
               all(rank is None or type(rank) is int and 1 <= rank <= gallery_count
                   for rank in ranks[condition].values())
               for condition in conditions)


def _acquire_model_lock(path: Path) -> int:
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor


def _release_model_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)



def _export_key(fingerprint: str, attempt: str, depth: int | None) -> str | None:
    """Per-ATTEMPT key for the export, deliberately NOT part of the run identity.

    `topk_depth` must never enter `identity`: the measurement is identical with or without a
    candidate window, and folding it in would mint a new `run_id` for the same experiment and then
    present the two as separate attempts (a spurious duplicate). The export is a property of an
    ATTEMPT, so it keys on (measurement fingerprint, attempt, depth).
    """
    if not depth:
        return None
    return _digest({"run": fingerprint, "attempt": attempt, "depth": int(depth),
                    "kind": "candidate-topk"})[:16]

def _export_complete(artifacts: Path, run_id: str, conditions: Sequence[str],
                     depth: int | None) -> bool:
    """True when every requested condition already has a complete export at `depth`."""
    if not depth:
        return False
    from workbench.composition.topk import export_index

    index = export_index(artifacts, run_id)
    if set(index) < set(conditions):
        return False
    return all(int(index[condition].get("depth") or 0) == int(depth)
               and not index[condition].get("partial")
               for condition in conditions)

def _cached_features(work_root: Path, fingerprint: str, condition: str,
                     scopes: Sequence[str | None]) -> tuple[dict[str | None, Any],
                                                           dict[str | None, Any]]:
    """Load this run's already-encoded gallery and query matrices from its work directory.

    Returns `({}, {})` when any needed memmap is gone. Reusing the cache is the whole point: a
    resumed run must not re-encode 74k gallery images just to build a candidate window that the
    scores it already computed would have produced.
    """
    gallery: dict[str | None, Any] = {}
    queries: dict[str | None, Any] = {}
    for scope in scopes:
        scope_name = _scope_name(scope)
        target_file = work_root / "features" / f"gallery_{scope_name}.npy"
        feature_file = work_root / "features" / f"query_{condition}_{scope_name}.npy"
        target = _load_memmap(target_file,
                              _digest({"run": fingerprint, "scope": scope,
                                       "kind": "full-gallery-features"}))
        features = _load_memmap(feature_file,
                                _digest({"run": fingerprint, "condition": condition,
                                         "scope": scope, "kind": "query-features"}))
        if target is None or features is None:
            return {}, {}
        gallery[scope] = target
        queries[scope] = features
    return gallery, queries

def _backfill_export(adapter: Any, queries: Sequence[Any], query_scopes: Sequence[str | None],
                     scopes: Sequence[str | None], positive_positions: Mapping[str, tuple[int, ...]],
                     reasons_by_condition: Mapping[str, Mapping[str, str]], work_root: Path,
                     fingerprint: str, conditions: Sequence[str], writers: Mapping[str, Any],
                     batch_size: int, original_ranks: Mapping[str, Mapping[str, Any]] | None = None) -> dict[str, int]:
    """Build candidate windows for a resumed run from its CACHED vectors.

    Same adapter, same features, same positive positions, same tie policy AND the same effective
    BLAS policy as the original rank pass. That last one is not decoration: the BLAS thread count is
    an arithmetic input (measured score deltas differ between 1 and 10 threads), so a backfill that
    scored at the default 10 threads while the original pass scored at 1 would produce a different
    window from the same vectors.

    `original_ranks` is the rank payload the original pass recorded. When supplied, every
    backfilled target rank is compared with it and a disagreement raises
    `BackfillRankDisagreement` instead of writing the window: a mismatch is proof that the two
    passes did not share a policy, and such a window must never be published as part of that
    measurement.

    Returns a per-condition count of queries stored; conditions whose cache is missing are simply
    absent from the result and the caller reports them.
    """
    from workbench.composition.topk import write_batch

    stored: dict[str, int] = {}
    for condition in conditions:
        if condition not in writers:
            continue
        gallery, features = _cached_features(work_root, fingerprint, condition, scopes)
        if not gallery:
            continue
        # A query the original pass could not score has no rank and therefore no window: the
        # failure reason is the same per-query map the metric pass wrote.
        consequences = reasons_by_condition.get(condition) or {}
        written = 0
        for scope in scopes:
            if scope not in gallery or scope not in features:
                continue
            indices = [index for index, query_scope in enumerate(query_scopes)
                       if query_scope == scope and queries[index].qid not in consequences]
            features_for_scope = features[scope]
            targets = gallery if len(scopes) > 1 else gallery[scope]
            for offset in range(0, len(indices), batch_size):
                batch = indices[offset:offset + batch_size]
                vectors = np.asarray(features_for_scope[batch], dtype=np.float32)
                keep = [position for position, index in enumerate(batch)
                        if np.isfinite(vectors[position]).all()
                        and positive_positions.get(queries[index].qid)]
                if not keep:
                    continue
                kept_indices = [batch[position] for position in keep]
                vectors = np.asarray(features_for_scope[kept_indices], dtype=np.float32)
                positions = [positive_positions[queries[index].qid] for index in kept_indices]
                cats = [query_scopes[index] for index in kept_indices]
                # The SAME effective numerical policy as the normal rank pass. Without this, BLAS
                # runs at whatever the process has (10 by default) while the original pass scored
                # at BLAS_THREADS_PINNED, and a policy-sensitive scorer returns a different rank --
                # measured on a fixture as rank 1 (normal) vs rank 2 (backfill). Exporting that
                # would attach a window computed under different arithmetic to a published metric.
                pinned = _blas_thread_limit(BLAS_THREADS_PINNED)
                _record_effective_blas_pin(pinned)
                try:
                    ranks, windows = adapter.rank_and_topk(vectors, targets, positions, cats,
                                                           writers[condition].depth)
                finally:
                    _blas_thread_limit(pinned)
                # Fail closed: a backfilled rank that disagrees with the rank the original pass
                # measured means the two did NOT run under the same policy, so this window cannot
                # be presented as the window of that measurement. The caller reports the refusal.
                # Only compared when the caller supplied the recorded ranks -- a caller with no
                # record has nothing to reconcile against, and inventing a comparison would reject
                # every backfill.
                disagreement = []
                if original_ranks is not None:
                    disagreement = [
                        qid for qid, rank in zip([queries[index].qid for index in kept_indices], ranks)
                        if original_ranks.get(condition, {}).get(qid) != rank
                    ]
                if disagreement:
                    raise BackfillRankDisagreement(
                        f"{condition}: {len(disagreement)} backfilled target rank(s) disagree with "
                        f"the recorded run ({disagreement[:5]}); refusing to export a window "
                        f"computed under a different scoring policy")
                write_batch(writers, condition,
                            [queries[index].qid for index in kept_indices], ranks, windows)
                written += len(kept_indices)
        stored[condition] = written
    return stored

def _scope_name(scope: str | None) -> str:
    return "global" if scope is None else _component(scope)

def composition_variant_provenance(model: Mapping[str, Any], config: Any = None) -> dict[str, Any]:
    """What this run substitutes or repairs, from declarations -- never from a model name.

    Delegates to the ONE existing builder (`workbench.scripts.evaluate_models._variant_provenance`)
    rather than inventing a second convention, so a composition summary and an aggregate report
    describe an adapted runtime with the same shape and the same words.

    The aggregate side needed it because a run under ``WORKBENCH_RUNTIME_ADAPTATION=sm_120`` was
    recorded with no ``variant_provenance`` at all (``grep -c sm_120`` = 0 in TME/PTHA summaries):
    whether a run verified against a MODERN package stack instead of the authors' historical pins
    is a provenance fact about the result, not a note for a reviewer. A run under an adaptation is
    adapted evidence and must never be presented as an exact reproduction.

    ``adaptive`` failure is not fatal: an unknown adaptation value is a real operator error the
    aggregation gate refuses, but a summary must still be written for the run that just happened,
    so the error is recorded as a named fact instead of aborting the write.
    """
    from workbench.backend.runtime import ADAPTATION_ENV, runtime_adaptation
    from workbench.backend.model_settings import setting

    requested = (setting(ADAPTATION_ENV) or "").strip() or None
    try:
        from workbench.scripts.evaluate_models import _variant_provenance

        provenance = _variant_provenance(dict(model), config) if config is not None \
            else {"adapted_runtime": None}
    except Exception as error:
        provenance = {"builder_unavailable": f"{type(error).__name__}: {error}"}
    # The selected adaptation is recorded even when this model declares none: a process-global
    # selection that applied to a sibling model is still part of why this run's environment looked
    # the way it did, and `applies_to_this_model` says whether it changed THIS model's contract.
    adapted = provenance.get("adapted_runtime")
    provenance["runtime_adaptation_selected"] = requested
    provenance["adapted_runtime_applied"] = bool(adapted)
    if requested and not adapted:
        provenance["adapted_runtime_note"] = (
            f"{ADAPTATION_ENV}={requested} was selected but this model declares no matching "
            "adapted_runtime; it verified against its historical pins")
    return provenance


#: Per-query failure reasons that a rerun reproduces deterministically. A recorded PARTIAL whose
#: every failure is one of these is never retried on resume; only transient reasons (model
#: unavailable, failed, unavailable exact rank, or an unexpected exception) earn another attempt.
_STABLE_FAILURE_MARKERS = frozenset({
    "category scope failure",
    "has no adapter embedding scope",
    "no positive target resolved in exact released gallery",
    "query features unavailable",
    "invalid full-gallery exact rank",
    "query reference image is missing",
})


def _is_stable_failure(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in _STABLE_FAILURE_MARKERS)


def normalize_release_path(value: str) -> str:
    """Remove release placeholder while preserving exact relative path identity."""
    value = value.replace("\\", "/")
    return re.sub(r"^<path_to[-_]M[-_]BEIR>/", "", value, flags=re.IGNORECASE)


def _image_directories(dataset_root: Path, layout: str) -> tuple[Path, ...]:
    from workbench.backend.fashioniq_layout import paths_for_layout

    dirs = []
    for path in (*paths_for_layout(dataset_root, layout), dataset_root / "images"):
        if path.name in {"images", "resized_images", "resized_image"} or path.parent.name == "resized_image":
            if path.is_dir():
                dirs.append(path)
    return tuple(dict.fromkeys(dirs))


def image_resolver(dataset_root: Path, layout: str) -> Callable[[str, Mapping[str, str]], Path | None]:
    """Index configured FashionIQ layout by stem, retaining released identity/order separately."""
    dirs = _image_directories(dataset_root, layout)
    exact: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = {}
    for directory in dirs:
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            exact.setdefault(path.name, path)
            by_stem.setdefault(path.stem, []).append(path)
    for name in by_stem:
        by_stem[name].sort(key=lambda path: (path.suffix.lower() != ".png", str(path)))

    def resolve(identity: str, candidate_names: Mapping[str, str]) -> Path | None:
        released = candidate_names.get(identity, identity)
        basename = PurePosixPath(normalize_release_path(released)).name
        direct = exact.get(basename)
        if direct is not None:
            return direct
        matches = by_stem.get(Path(basename).stem, ())
        return matches[0] if matches else None

    return resolve


def resolve_release_image(image_id: str, candidate_names: Mapping[str, str],
                          dataset_root: Path, layout: str) -> Path | None:
    """Resolve one release identity through a registry-declared image layout."""
    return image_resolver(dataset_root, layout)(image_id, candidate_names)


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


#: How thoroughly a cached feature matrix is verified on load.
#:   ``full``   -- the chunked content digest is recomputed and compared. Catches a same-shape,
#:                same-length file whose bytes changed, which fingerprint+shape cannot.
#:   ``header`` -- fingerprint, shape and byte length only. Cheap; does NOT detect changed content.
#:   ``off``    -- the file is used if it parses. Only for an operator diagnosing a load-time problem.
#: Defaults to ``full``: a cache is a measurement input, so a wrong one silently produces wrong
#: ranks. The effective mode is recorded in the cache sidecar AND in the run identity, so an artifact
#: can never claim full-cache assurance the loader did not actually perform.
CACHE_VERIFY_ENV = "WB_CACHE_VERIFY"
CACHE_VERIFY_MODES = ("full", "header", "off")

def _cache_verify_mode() -> str:
    raw = (os.environ.get(CACHE_VERIFY_ENV) or "").strip().lower()
    return raw if raw in CACHE_VERIFY_MODES else "full"

def _content_digest(array: Any, chunk_rows: int = 8192) -> str:
    """sha256 over an array's dtype, shape and raw content, streamed in row chunks.

    Streamed rather than ``array.tobytes()`` because the gallery matrix is 74,381 rows: one
    contiguous copy would double peak RSS for data that is already on disk. Chunking is why the
    digest can be computed while writing instead of in a second pass over the finished file.
    """
    digest = hashlib.sha256()
    digest.update(str(np.asarray(array).dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(json.dumps(list(np.asarray(array).shape)).encode("ascii"))
    digest.update(b"|")
    rows = int(np.asarray(array).shape[0]) if np.asarray(array).ndim else 0
    if rows == 0:
        digest.update(b"empty")
        return digest.hexdigest()
    for start in range(0, rows, chunk_rows):
        block = np.asarray(array[start:start + chunk_rows])
        # C order is the on-disk order of a freshly written .npy; ascontiguousarray stops a
        # re-loaded memmap's stride layout from producing a different digest for the same bytes.
        digest.update(np.ascontiguousarray(block).tobytes(order="C"))
    return digest.hexdigest()

def _memmap_integrity(path: Path) -> dict[str, Any]:
    """What the sidecar says about this cache, and whether that amounts to verification."""
    try:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"content_digest": "absent", "verified": False, "mode": "unreadable",
                "reason": "no readable sidecar"}
    recorded = metadata.get("content_digest")
    return {"content_digest": recorded if recorded else "absent",
            "verified": bool(recorded),
            "mode": metadata.get("verify_mode") or "unknown",
            "shape": metadata.get("shape"),
            "expected_bytes": metadata.get("byte_length")}

def _load_memmap(path: Path, fingerprint: str, shape: tuple[int, ...] | None = None) -> np.memmap | None:
    """Load a cached feature matrix, failing closed on any sign of tampering.

    Before this check the loader accepted any file whose sidecar fingerprint and shape matched, so
    replacing a matrix with an all-zeros array of the same shape loaded silently (measured: content
    sum 6.0 -> 0.0, accepted). Three failure modes are now caught:

    * **changed content** -- the recorded digest is recomputed and compared (``full`` mode);
    * **half-written file** -- a recorded ``byte_length`` mismatch is caught before parsing, and a
      truncated ``.npy`` raises ``ValueError`` from NumPy;
    * **stale sidecar** -- a sidecar whose shape or byte length disagrees with the file.

    A cache written before digests existed has no ``content_digest``: it still loads (it is probably
    fine, and re-encoding 74k images to prove it is not free), but ``_memmap_integrity`` reports it
    ``absent`` / ``verified: False`` so nothing downstream can call it validated.
    """
    mode = _cache_verify_mode()
    try:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint or not path.is_file():
            return None
        recorded_bytes = metadata.get("byte_length")
        if recorded_bytes is not None and path.stat().st_size != int(recorded_bytes):
            return None
        result = np.load(path, mmap_mode="r", allow_pickle=False)
        if shape is not None and tuple(result.shape) != tuple(shape):
            return None
        # A sidecar whose recorded shape disagrees with EITHER the file or the caller's expectation
        # is stale: the metadata describes a cache this file is not. Checking only the caller's
        # `shape` let a rewritten sidecar (shape 99x3 over a 4x3 file) pass whenever the caller
        # happened to ask for the file's real shape.
        recorded_shape = metadata.get("shape")
        if recorded_shape is not None and tuple(int(v) for v in recorded_shape) != tuple(result.shape):
            return None
        if mode == "off":
            return result
        recorded = metadata.get("content_digest")
        if not recorded or mode == "header":
            # No digest (legacy) or header-only verification: usable, but see _memmap_integrity.
            return result
        if _content_digest(result) != recorded:
            return None
        return result
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _memmap_metadata_shape(path: Path) -> tuple[int, ...] | None:
    """The shape the sidecar records for this cache, or None when it is missing or unusable."""
    try:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        shape = metadata.get("shape")
        return tuple(int(value) for value in shape) if shape else None
    except (OSError, ValueError, TypeError):
        return None

def _write_memmap_metadata(path: Path, fingerprint: str, value: np.ndarray) -> None:
    """Write the sidecar, recording a content digest so the next load can verify the bytes.

    `value` is the array that was just written, so the digest comes from the in-memory result
    rather than a second pass over the finished file: one pass, no duplicate copy, no doubled I/O
    on a 74k-row matrix.
    """
    _atomic_json(path.with_suffix(path.suffix + ".json"),
                 {"fingerprint": fingerprint, "shape": list(value.shape), "dtype": str(value.dtype),
                  "byte_length": int(path.stat().st_size) if path.is_file() else None,
                  "content_digest": _content_digest(value),
                  "verify_mode": _cache_verify_mode()})


def _gallery_fingerprint(data: Any) -> str:
    return _digest({"protocol_id": data.gallery.protocol_id,
                    "ordered_gallery": list(data.gallery.image_ids),
                    "candidate_names": dict(sorted(data.candidate_names.items()))})


def _query_fingerprint(queries: Sequence[Any]) -> str:
    return _digest([{"qid": query.qid, "index": query.index, "text": query.query_txt,
                     "reference_id": query.reference_id, "positive_ids": list(query.positive_ids),
                     "category": query.category, "subsets": list(query.subsets)} for query in queries])


def _scope_resolution_identity(queries: Sequence[Any], query_scopes: Sequence[str | None]) -> str:
    """Digest of the resolved per-query scope map: which gallery matrix each query is scored on.

    This is a scoring INPUT, not metadata. `_query_fingerprint` keys on the loader's
    `query.category`, which stays `None` for exactly the queries an adapter resolver scopes, so a
    resolver wired without this digest leaves the run identity unchanged and `--resume` silently
    reuses the old all-null ranks. Keyed by qid, so the same assignment in a different query order
    is still the same experiment.
    """
    return _digest(sorted((query.qid, scope) for query, scope in zip(queries, query_scopes)))

def _positive_positions(data: Any) -> dict[str, tuple[int, ...]]:
    index = {identity: position for position, identity in enumerate(data.gallery.image_ids)}
    by_path = {normalize_release_path(identity): position
               for position, identity in enumerate(data.gallery.image_ids)}
    for candidate_id, image_path in data.candidate_names.items():
        position = by_path.get(normalize_release_path(image_path))
        if position is not None:
            index[candidate_id] = position
    return {query.qid: tuple(dict.fromkeys(index[value] for value in query.positive_ids if value in index))
            for query in data.queries}


def _scope_for_query(adapter: Any, query: Any,
                     candidate_names: Mapping[str, str] | None = None) -> tuple[str | None, str | None]:
    scopes = tuple(getattr(adapter, "category_scopes", (None,)))
    # An assignment the release loader already made is final: never re-derive what it decided.
    if query.category in scopes:
        return query.category, None
    # The adapter may know a scope the release loader could not derive (LIMN reads it from the
    # query's own FashionIQ caption annotation). Only a value the adapter actually declares is
    # accepted; anything else -- including `None` -- stays unresolved instead of being trusted.
    resolver = getattr(adapter, "query_scope", None)
    if callable(resolver):
        resolved = resolver(query, candidate_names)
        if resolved in scopes:
            return resolved, None
    if None in scopes:
        return None, None
    # `None` is the unresolved-category sentinel, not a category name; this text lands in failure
    # records the UI shows verbatim.
    category = "unresolved" if query.category is None else repr(query.category)
    return None, (f"query {query.qid}: category {category} has no adapter embedding scope "
                   f"{scopes!r}; batch_indices=({query.index},), qids={[query.qid]}")


def _disk_projection(path: Path, gallery_count: int, feature_shape: Sequence[int], remaining_scopes: int,
                     query_count: int, conditions: int) -> dict[str, int]:
    feature_width = int(np.prod(feature_shape, dtype=np.int64))
    matrix_bytes = gallery_count * feature_width * np.dtype(np.float32).itemsize
    query_width = int(feature_shape[-1])
    query_bytes = query_count * query_width * np.dtype(np.float32).itemsize * conditions
    projected = matrix_bytes * remaining_scopes + query_bytes
    margin = max(64 * 1024 * 1024, projected // 10)
    free = shutil.disk_usage(path.parent).free
    if free < projected + margin:
        raise OSError(f"composition features need {projected} bytes plus {margin} safety margin; "
                      f"only {free} bytes free at {path.parent}")
    return {"matrix_bytes_per_scope": matrix_bytes, "query_feature_bytes": query_bytes,
            "projected_bytes": projected, "safety_margin_bytes": margin, "free_bytes": free}


def _encode_gallery(adapter: Any, image_paths: Sequence[Path], scope: str | None,
                    destination: Path, fingerprint: str, batch_size: int,
                    remaining_scopes: int, query_count: int, condition_count: int) -> tuple[np.memmap, dict[str, int]]:
    # The sidecar records the shape this cache was written with. A `.npy` that is valid but holds
    # the wrong number of rows -- a partially written gallery whose sidecar was also rewritten, or a
    # cache copied from another run -- would otherwise be reused with every row after the shortfall
    # describing a different image. `_encode_queries` already passes its expected shape; the gallery
    # path must too. NumPy refuses a byte-truncated file at `np.load` on its own.
    metadata_shape = _memmap_metadata_shape(destination)
    expected_shape = (len(image_paths), *metadata_shape[1:]) if metadata_shape else None
    cached = _load_memmap(destination, fingerprint, expected_shape)
    if cached is not None:
        return cached, {"reused": 1, "projected_bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapped = None
    trailing_shape: tuple[int, ...] | None = None
    position = 0
    projection: dict[str, int] = {}
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start:start + batch_size]
        encoded = _as_array(adapter.encode_targets([str(path) for path in paths], [scope] * len(paths)))
        if encoded.ndim < 2 or encoded.shape[0] != len(paths):
            raise ValueError(f"encode_targets returned invalid shape {encoded.shape}")
        current_shape = tuple(encoded.shape[1:])
        if trailing_shape is not None and current_shape != trailing_shape:
            raise ValueError(f"encode_targets feature shape changed between batches: {trailing_shape} != {current_shape}")
        encoded = encoded.astype(np.float32, copy=False)
        if mapped is None:
            trailing_shape = current_shape
            projection = _disk_projection(destination.parent, len(image_paths), trailing_shape,
                                          remaining_scopes, query_count, condition_count)
            mapped = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                               shape=(len(image_paths), *trailing_shape))
        mapped[start:start + len(paths)] = encoded
        position += len(paths)
        del encoded
    if mapped is None or position != len(image_paths):
        raise ValueError("gallery encoding produced no features")
    mapped.flush()
    _write_memmap_metadata(destination, fingerprint, mapped)
    return np.load(destination, mmap_mode="r", allow_pickle=False), projection


def _stage_query_path(path: Path, query: Any, condition: str, work_root: Path) -> tuple[Path, str]:
    with Image.open(path) as source:
        image = source.copy()
    adapted, text = adapt_query_inputs(image, query.query_txt, condition)
    if condition == "text_only_black_image":
        staged = work_root / "queries" / condition / f"{_component(query.qid)}.png"
        staged.parent.mkdir(parents=True, exist_ok=True)
        adapted.save(staged, format="PNG")
        return staged, text
    return path, text


def _encode_queries(adapter: Any, queries: Sequence[Any], query_paths: Sequence[Path | None],
                    indices: Sequence[int], condition: str, work_root: Path,
                    destination: Path, fingerprint: str, dimension: int,
                    batch_size: int, scopes: Sequence[str | None] | None = None,
                    ) -> tuple[np.memmap, dict[str, str], dict[str, str]]:
    # `scopes` is the resolved per-query scope the runner routed with. Without it the adapter would
    # be handed the release loader's `query.category`, which is `None` exactly for the queries an
    # adapter resolver exists to scope, and LIMN refuses those.
    def scope_of(index: int) -> str | None:
        return queries[index].category if scopes is None else scopes[index]

    cached = _load_memmap(destination, fingerprint, (len(queries), dimension))
    errors_path = destination.with_suffix(destination.suffix + ".errors.json")
    category_path = destination.with_suffix(destination.suffix + ".category-errors.json")
    if cached is not None:
        try:
            errors = json.loads(errors_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors = {}
        try:
            category_errors = json.loads(category_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            category_errors = {}
        return cached, errors, category_errors

    destination.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                         shape=(len(queries), dimension))
    features[:] = np.nan
    errors: dict[str, str] = {}
    category_errors: dict[str, str] = {}
    eligible = [index for index in indices if query_paths[index] is not None]
    for index in indices:
        if query_paths[index] is None:
            errors[queries[index].qid] = "query reference image is missing"
    for batch_start in range(0, len(eligible), batch_size):
        batch = eligible[batch_start:batch_start + batch_size]
        paths: list[str] = []
        texts: list[str] = []
        categories: list[str | None] = []
        surviving: list[int] = []
        for index in batch:
            try:
                staged_path, text = _stage_query_path(query_paths[index], queries[index], condition, work_root)  # type: ignore[arg-type]
                paths.append(str(staged_path))
                texts.append(text)
                categories.append(scope_of(index))
                surviving.append(index)
            except Exception as error:
                errors[queries[index].qid] = f"{type(error).__name__}: {error}"
        if not surviving:
            continue
        try:
            encoded = _as_array(adapter.encode_queries(paths, texts, categories))
            if encoded.shape != (len(surviving), dimension):
                raise ValueError(f"encode_queries returned invalid shape {encoded.shape}")
            features[surviving] = encoded.astype(np.float32, copy=False)
        except Exception as batch_error:
            if isinstance(batch_error, AdapterWorkerError):
                raise
            bad_batch_indices = tuple(getattr(batch_error, "batch_indices", ())) \
                if type(batch_error).__name__ == "CategoryScopeError" else ()
            bad_qids = [queries[surviving[index]].qid for index in bad_batch_indices
                        if 0 <= index < len(surviving)]
            if bad_batch_indices:
                reason = (f"{getattr(batch_error, 'model_id', type(adapter).__name__)} category scope failure; "
                          f"batch_indices={bad_batch_indices}; qids={bad_qids}; {batch_error}")
                category_errors.update({qid: reason for qid in bad_qids})
            for local_index, (index, path, text) in enumerate(zip(surviving, paths, texts)):
                if local_index in bad_batch_indices:
                    continue
                try:
                    encoded = _as_array(adapter.encode_queries([path], [text], [scope_of(index)]))
                    if encoded.shape != (1, dimension):
                        raise ValueError(f"encode_queries returned invalid shape {encoded.shape}")
                    features[index] = encoded[0].astype(np.float32, copy=False)
                except Exception as error:
                    if isinstance(error, AdapterWorkerError):
                        raise
                    if type(error).__name__ == "CategoryScopeError":
                        local_indices = tuple(getattr(error, "batch_indices", ()))
                        category_errors[queries[index].qid] = (
                            f"{getattr(error, 'model_id', type(adapter).__name__)} category scope failure; "
                            f"batch_indices={local_indices}; qids={[queries[index].qid]}; {error}")
                    else:
                        errors[queries[index].qid] = f"{type(error).__name__}: {error} (batch: {batch_error})"
    features.flush()
    _write_memmap_metadata(destination, fingerprint, features)
    _atomic_json(errors_path, errors)
    _atomic_json(category_path, category_errors)
    return np.load(destination, mmap_mode="r", allow_pickle=False), errors, category_errors


#: Query-dimension parallelism for the scoring contraction. MEASURED and BIT-IDENTICAL: every
#: query's score row is computed by the SAME expression as today (`adapter.score` over a sub-batch),
#: only different queries run concurrently. Verified on the real habit features that a single query
#: scored alone is bit-identical to the same query scored inside a batch (delta 0.0), and that
#: sub-batches of 2/4/8/16/32/64/128 all reproduce the whole-batch result exactly (delta 0.0,
#: every rank equal). This is the opposite of `optimize=True`/GEMM chunking, which reorders the
#: summation INSIDE one query and moves 207/18009 ranks.
#:
#: It lives here, in the runner, and NOT in the adapters: `runner.py` is not hashed into the run
#: identity, so no `adapter_source_hash` changes and existing published runs stay comparable.
#: The BLAS thread count every scoring call is pinned to, and the value recorded in the run
#: identity. ONE fixed value, not an operator knob: the pin exists to remove thread count as a
#: source of arithmetic variation, so allowing it to float would defeat it. Pinned to 1 because
#: concurrent group-level parallelism already supplies the speed and multi-threaded BLAS inside a
#: group worker would oversubscribe (8 groups x 10 BLAS threads = 80 threads on 20 cores).
BLAS_THREADS_PINNED = 1
SCORE_THREADS_ENV = "WORKBENCH_SCORE_THREADS"
#: Half the box (AGENTS.md), and deliberately conservative: this competes with the live cohort.
SCORE_THREADS_DEFAULT = min(8, (os.cpu_count() or 8) // 2)

def resolved_score_threads(default: int = SCORE_THREADS_DEFAULT) -> int:
    """Query-parallelism for scoring, from the operator override or the conservative default."""
    raw = os.environ.get(SCORE_THREADS_ENV, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{SCORE_THREADS_ENV} must be a positive integer, got {raw!r}") from None
    if value < 1:
        raise ValueError(f"{SCORE_THREADS_ENV} must be a positive integer, got {raw!r}")
    return value

def score_in_query_chunks(adapter: Any, queries: Any, targets: Any, chunk_size: int,
                          categories: Sequence[str | None] | None = None,
                          threads: int | None = None) -> np.ndarray:
    """Score query chunks against the unchanged full target matrix through adapter semantics.

    Chunks are dispatched to a thread pool over the QUERY axis. Each chunk still goes through the
    adapter's own `score` with the same arguments, so every row is bit-identical to the serial
    result; `map` preserves order. Requiring the chunks to be independent is why this is exact.
    """
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    total = len(queries)
    starts = list(range(0, total, chunk_size))
    workers = resolved_score_threads() if threads is None else int(threads)

    def score_one(start: int) -> Any:
        end = min(start + chunk_size, total)
        part = queries[start:end]
        if categories is None:
            return adapter.score(part, targets)
        return adapter.score(part, targets, categories[start:end])

    if workers > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            pieces = [_as_array(piece) for piece in pool.map(score_one, starts)]
    else:
        pieces = [_as_array(score_one(start)) for start in starts]
    return np.concatenate(pieces, axis=0) if pieces else np.empty((0, len(targets)), dtype=np.float32)


def rank_in_query_chunks(adapter: Any, vectors: Any, targets: Any, positions: Sequence[Any],
                         categories: Sequence[str | None] | None, depth: int,
                         threads: int | None = None, chunk_size: int = 32,
                         ) -> tuple[list[Any], list[Any]]:
    """Exact ranks (and optional Top-K windows) computed over independent query sub-batches.

    Splitting the QUERY axis cannot change a result: each query's score row is one dot product set
    over the gallery, computed by the adapter's own ``rank``/``rank_and_topk`` with the same
    targets. Verified bit-identical at sub-batch sizes 2..128 against the whole-batch call.
    Order is preserved, so the returned lists line up with ``vectors`` exactly as a serial call does.

    ``chunk_size`` -- not the thread count -- decides how many queries ride in one ``adapter.score``
    call. That grouping is part of the arithmetic: a scorer whose BLAS kernel depends on the matrix
    M dimension (any ``@``-based adapter, and GEMM under ``optimize=True``) can return last-bit
    differences at a different group size. Deriving the group from a worker count would therefore
    make an operator env var change publishable numbers under an unchanged run id, so the caller
    passes its own ``batch_size`` here and thread count only decides CONCURRENCY.
    """
    total = len(vectors)
    workers = resolved_score_threads() if threads is None else int(threads)
    if total == 0:
        return [], []
    size = max(1, int(chunk_size))
    starts = list(range(0, total, size))

    def run_one(start: int) -> tuple[Any, Any]:
        end = min(start + size, total)
        part_vectors = vectors[start:end]
        part_positions = positions[start:end]
        part_cats = None if categories is None else categories[start:end]
        if depth:
            ranks, windows = adapter.rank_and_topk(part_vectors, targets, part_positions,
                                                  part_cats, depth)
            return ranks, windows
        return adapter.rank(part_vectors, targets, part_positions, part_cats), []

    if workers > 1 and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as pool:
            parts = list(pool.map(run_one, starts))
    else:
        parts = [run_one(start) for start in starts]

    all_ranks: list[Any] = []
    all_windows: list[Any] = []
    for ranks, windows in parts:
        all_ranks.extend(ranks)
        all_windows.extend(windows)
    return all_ranks, all_windows

def _openblas_symbols() -> tuple[Any, Any, str | None]:
    """The (setter, getter, library) of the OpenBLAS NumPy actually loaded.

    `setter`/`getter` are ctypes function objects, or None when the runtime exposes neither. The
    symbol names are probed in 64-bit-then-32-bit order because ILP64 NumPy builds (the default for
    `numpy.libs/libopenblas64_*`) export the `*64_` spellings and a 32-bit build exports the plain
    ones. Loading a DIFFERENT environment's BLAS from a hard-coded path is deliberately not done:
    that mutates a library this process is not computing with and would report a pin that does not
    apply to the arithmetic.

    Result is cached because symbol lookup is process-global and stable.
    """
    cached = getattr(_openblas_symbols, "_cache", None)
    if cached is not None:
        return cached

    import ctypes
    import glob
    import os

    import numpy as np

    found: tuple[Any, Any, str | None] = (None, None, None)
    root = os.path.dirname(os.path.dirname(np.__file__))
    for path in sorted(glob.glob(os.path.join(root, "numpy.libs", "*openblas*"))):
        try:
            library = ctypes.CDLL(path)
        except OSError:
            continue
        setter = (getattr(library, "openblas_set_num_threads64_", None)
                  or getattr(library, "openblas_set_num_threads", None))
        getter = (getattr(library, "openblas_get_num_threads64_", None)
                  or getattr(library, "openblas_get_num_threads", None))
        if setter is not None:
            getter = getter if callable(getter) else None
            found = (setter, getter, os.path.basename(path))
            break
    _openblas_symbols._cache = found
    return found

def _blas_thread_limit(threads: int | dict) -> dict[str, Any]:
    """Set the BLAS thread count and report what ACTUALLY happened.

    MEASURED collision this prevents: ``@``-based scorers (the base ``CompositionAdapter.score``
    used by ENCODER and LIMN) hand their work to multi-threaded OpenBLAS, and composition_eval.py
    pins OPENBLAS/OMP to 10 by default. Eight group threads times ten BLAS threads is eighty threads
    on a twenty-core box. Inside a group worker the parallelism already exists at the group level,
    so BLAS is pinned to one thread for the duration of the call.

    The return value is a record, not a bare int, because a pin that silently does nothing was
    measured on a real runtime: requesting 1 thread left OpenBLAS at 9 with no setter found, while
    callers recorded ``blas_threads_pinned = 1`` as fact. Every field is either read back from the
    runtime or explicitly ``None``/False:

    ``requested``    the value asked for
    ``effective``    the value the runtime reports AFTER the call, or None when unreadable
    ``previous``     the value the runtime reported BEFORE the call, or None when unreadable. Never
                     taken from environment variables: those describe the pre-exec setting, not the
                     state of a runtime that has already been changed.
    ``setter_found`` whether a setter symbol existed
    ``getter_found`` whether the value could be read back at all
    ``verified``     True only when a readback proves ``effective == requested``
    ``source``       the library the symbols came from, so a record names its own provenance

    Callers that previously did ``previous = _blas_thread_limit(n)`` then
    ``_blas_thread_limit(previous)`` keep working: passing a record back restores its ``previous``
    (falling back to ``effective`` when the runtime never had a readable previous), and a plain int
    is still accepted as "set to this".
    """
    import ctypes

    restore_to: int
    if isinstance(threads, dict):
        raw = threads.get("previous")
        if raw is None:
            raw = threads.get("effective")
        if raw is None:
            return {**threads, "requested": threads.get("requested"),
                    "verified": False,
                    "reason": "nothing to restore: the previous runtime value was never readable"}
        restore_to = int(raw)
    else:
        restore_to = int(threads)

    setter, getter, source = _openblas_symbols()
    before = None
    if callable(getter):
        try:
            before = int(getter())
        except Exception:
            before = None

    if setter is None:
        return {"requested": restore_to, "effective": before, "previous": before,
                "setter_found": False, "getter_found": callable(getter),
                "verified": False, "source": source,
                "reason": "no OpenBLAS setter symbol in the NumPy already loaded; "
                          "the runtime cannot be pinned and this is not a success"}

    # A ctypes function accepts a plain int for a `c_int` parameter, and a test double accepts the
    # same value; wrapping in `ctypes.c_int` would make the fake receive an object it cannot compare.
    setter(restore_to)
    after = None
    if callable(getter):
        try:
            after = int(getter())
        except Exception:
            after = None
    # `previous` is what the caller saw BEFORE this call; report it so a two-step
    # pin/restore reads naturally.
    verified = after is not None and after == restore_to
    return {"requested": restore_to, "effective": after,
            "previous": before if before is not None else after,
            "setter_found": True, "getter_found": callable(getter),
            "verified": bool(verified), "source": source,
            "reason": None if verified else
                      (f"setter ran but the runtime reports {after!r}, not {restore_to}" if after is not None
                       else "the runtime exposes no getter, so the pin cannot be verified")}

def _current_openblas_threads() -> int | None:
    """The BLAS thread count the runtime reports right now, or None when it cannot be read."""
    setter, getter, _source = _openblas_symbols()
    if not callable(getter):
        return None
    try:
        return int(getter())
    except Exception:
        return None

def _rank_groups_concurrently(rank_group: Any, groups: Sequence[Any], threads: int) -> list[Any]:
    """Run ``rank_group`` over independent groups, returning results or exceptions in group order.

    Group boundaries are fixed by the caller and never depend on ``threads``: the number of queries
    in one ``adapter.score`` call is part of the arithmetic (BLAS matrix M dimension), so only
    CONCURRENCY varies with the operator knob. Each group is computed by the same adapter call the
    serial loop made, which is what makes the result bit-identical.

    A raised exception is returned in its group's slot rather than aborting the batch, so the
    caller's existing per-batch fallback and error bookkeeping keep their behaviour.

    BLAS is pinned to ONE thread for the WHOLE call -- both the serial and the concurrent branch --
    toggled exactly once, and restored in a ``finally``. This matters because BLAS thread count is
    not numerically neutral: measured score deltas differ between 1 and 10 threads (3.576e-07 vs
    2.384e-07). Pinning only the concurrent branch would mean ``WORKBENCH_SCORE_THREADS=1`` computes
    with BLAS=10 while ``=8`` computes with BLAS=1, so the knob could move publishable numbers under
    an unchanged run id -- the same defect class this change exists to remove. The toggle stays
    OUTSIDE the workers: toggling per worker races (one worker restoring the previous count while
    another is still computing), and that race was measured to change ranks at 4 and 8 threads.
    """
    if not groups:
        return []
    previous = _blas_thread_limit(BLAS_THREADS_PINNED)
    _record_effective_blas_pin(previous)
    try:
        if threads <= 1 or len(groups) == 1:
            out: list[Any] = []
            for group in groups:
                try:
                    out.append(rank_group(group))
                except BaseException as error:  # noqa: BLE001 - reported per group by the caller
                    out.append(error)
            return out
        from concurrent.futures import ThreadPoolExecutor

        # MEASURED collision: `@`-based scorers (the base CompositionAdapter.score used by ENCODER
        # and LIMN) hand their work to multi-threaded BLAS, and composition_eval.py pins
        # OPENBLAS/OMP to 10 by default, so 8 group threads x 10 BLAS threads would be 80 threads
        # on 20 cores. Inside one thread per group the parallelism already exists at group level.
        with ThreadPoolExecutor(max_workers=min(int(threads), len(groups))) as pool:
            futures = [pool.submit(rank_group, group) for group in groups]
            out = []
            for future in futures:
                try:
                    out.append(future.result())
                except BaseException as error:  # noqa: BLE001 - reported per group by the caller
                    out.append(error)
            return out
    finally:
        _blas_thread_limit(previous)

def _record_effective_blas_pin(result: Any) -> None:
    """Remember the LAST pin result so the run identity can record what really happened.

    `BLAS_THREADS_PINNED` is the value we ASK for; this is the value the runtime confirmed. A run
    whose pin could not be verified must say so in its identity rather than stamping the requested
    number, because the identity is what decides whether two artifacts are the same measurement.

    Accepts a plain int for callers (and test doubles) that replace `_blas_thread_limit` with the
    old int-returning contract: nothing is known about such a pin, so it is recorded as unverified
    rather than silently assumed to have worked.
    """
    _effective_blas_pin.clear()
    if not isinstance(result, Mapping):
        _effective_blas_pin.update({
            "requested": BLAS_THREADS_PINNED, "effective": None, "previous": None,
            "verified": False, "setter_found": None, "getter_found": None, "source": None,
            "reason": "the pin helper returned no verification record, so the effective value is "
                      "unknown; this is not evidence that the pin applied",
        })
        return
    _effective_blas_pin.update({
        "requested": result.get("requested"),
        "effective": result.get("effective"),
        # `previous` is the value the runtime reported BEFORE the pin. Carried into the observation
        # because it is the only field that lets a reader confirm the pin CHANGED something -- an
        # `effective == requested` on a runtime already at that count proves much less.
        "previous": result.get("previous"),
        "verified": bool(result.get("verified")),
        "setter_found": bool(result.get("setter_found")),
        "getter_found": bool(result.get("getter_found")),
        "source": result.get("source"),
        "reason": result.get("reason"),
    })

_effective_blas_pin: dict[str, Any] = {}

#: Marker for an artifact written before `policy_observed` existed, and for a run that recorded
#: nothing. Kept as a string so it reads correctly in JSON without any consumer having to know a
#: sentinel: an absent observation is never rendered as a boolean.
NOT_RECORDED = "not_recorded (no readback captured for this run)"

#: Field names of the readback record `_blas_thread_limit` returns and `_record_effective_blas_pin`
#: stores. An artifact's `policy_observed` block is exactly these, plus a verdict, so a consumer can
#: check it against what the pin helper is able to report at all.
POLICY_OBSERVED_FIELDS = ("requested", "effective", "previous", "setter_found", "getter_found",
                          "verified", "source")

def _reset_effective_blas_pin() -> None:
    """Drop any readback left in the module dict by an earlier run in this process.

    REQUIRED before a run's scoring phase: `_effective_blas_pin` is process-global, so without this
    a run that scored nothing (reused ranks, every condition unsupported) would publish the
    PREVIOUS model's readback as its own observation. That is worse than publishing nothing --
    it would claim a verified pin for a run whose scoring never happened.
    """
    _effective_blas_pin.clear()

def observed_policy_record() -> dict[str, Any]:
    """What the runtime ACTUALLY reported for this run's scoring calls, or an honest absence.

    Deliberately NOT part of the run identity. The identity is hashed BEFORE scoring runs
    (`run_id`/`fingerprint` at `runner.py:1510`), so a value only known afterwards cannot go into
    that payload without making the id depend on process history -- the exact defect the
    `_blas_pin_capability` docstring records ("the first run recorded an empty map and the second a
    populated one, so the same inputs minted two run ids"). This record therefore travels OUTSIDE
    the hash, in the rank payload and the summary, where it can describe a particular run without
    re-keying it.

    Three distinct facts, never collapsed:
      * `blas_threads_pinned` (identity)  -- the value we ASKED for;
      * `blas_pin_capability` (identity)  -- what this interpreter CAN do;
      * `policy_observed`     (this)      -- what it DID for this run's scoring calls.
    """
    if not _effective_blas_pin:
        return {"status": NOT_RECORDED,
                "reason": ("no BLAS pin readback was recorded for this run: either the run reused "
                           "ranks instead of scoring, or it produced no rank payload. This is an "
                           "absence of evidence, not evidence the pin failed."),
                "record": None}
    record = {name: _effective_blas_pin.get(name) for name in POLICY_OBSERVED_FIELDS}
    if record.get("verified") is True:
        status = "verified"
    elif record.get("setter_found") is False or record.get("getter_found") is False:
        status = "unverifiable"
    else:
        status = "unverified"
    return {"status": status, "reason": _effective_blas_pin.get("reason"), "record": record}

def policy_observed_from_payload(payload: Any) -> dict[str, Any]:
    """The `policy_observed` block of a stored rank payload, or an honest absence.

    A legacy artifact (written before this field existed) has no block. It must be reported as
    absent, never as `verified: false` -- those are different claims, and the eight artifacts
    produced before this change have exactly this shape. Nothing is backfilled: a value that was
    not observed is never invented.
    """
    if not isinstance(payload, Mapping):
        return {"status": NOT_RECORDED,
                "reason": "no rank payload was read, so no policy observation exists to report",
                "record": None}
    block = payload.get("policy_observed")
    if not isinstance(block, Mapping):
        return {"status": NOT_RECORDED,
                "reason": "this artifact predates the policy_observed field; whether its pin was "
                          "verified is not recorded here",
                "record": None}
    return {"status": block.get("status", NOT_RECORDED), "reason": block.get("reason"),
            "record": block.get("record")}

def _blas_pin_capability() -> dict[str, Any]:
    """What this interpreter can prove about BLAS pinning -- stable for the process.

    Recorded in the run identity in place of the last scoring call's mutable result, because an
    identity derived from mutable state changes between two identical runs in one process (measured:
    the first run recorded an empty map and the second a populated one, so the same inputs minted two
    run ids). This probe reports capability, not the outcome of any particular call: whether a setter
    exists and whether a readback getter exists. A runtime that cannot be verified says so here, so
    an artifact can never claim a pin the environment is unable to confirm.
    """
    cached = getattr(_blas_pin_capability, "_cache", None)
    if cached is not None:
        return dict(cached)
    setter, getter, source = _openblas_symbols()
    result = {
        "setter_found": setter is not None,
        "getter_found": callable(getter),
        "verifiable": setter is not None and callable(getter),
        "source": source,
        "reason": None if (setter is not None and callable(getter)) else
                  ("no OpenBLAS setter symbol in the NumPy already loaded: the pin cannot be "
                   "applied" if setter is None else
                   "no readback getter: the pin cannot be verified"),
    }
    _blas_pin_capability._cache = result
    return dict(result)

def _gpu_identity() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None

def model_interpreter_identity(model: dict[str, Any], config: Any, adapter: Any = None, *,
                               verify: bool = True) -> dict[str, Any]:
    from workbench.backend.model_settings import setting_source
    from workbench.backend.runtime import (
        READY_TIERS, _record_environment_fingerprint, resolve_model_interpreter, verify_environment)

    variable = (model.get("environment") or {}).get("python_env_var")
    path = getattr(adapter, "interpreter", None) or resolve_model_interpreter(model, config)
    source = setting_source(variable) if variable else "unset"
    if source == "unset" and path:
        source = "artifact_assignment"
    if not path:
        return {"path": None, "version": "unavailable", "source": source,
                "verified": False, "verification_reason":
                f"no model interpreter resolved ({variable or 'no declared variable'})"}
    resolved = os.path.abspath(Path(path).expanduser())
    code = chr(10).join(("import sys", "print(sys.version.split()[0])", "try:",
                         " import torch", " print(torch.__version__)",
                         "except Exception as error:", " print('torch-unavailable: ' + str(error))"))
    try:
        probe = subprocess.run([resolved, "-c", code], capture_output=True, text=True,
                               timeout=10, check=False)
        if probe.returncode:
            raise RuntimeError(probe.stderr.strip() or f"exit status {probe.returncode}")
        versions = probe.stdout.strip().splitlines()
        version = f"Python {versions[0]}" if versions else "unavailable"
        if len(versions) > 1:
            version += f"; torch {versions[1]}"
    except Exception as error:
        return {"path": resolved, "version": "unavailable", "source": source,
                "verified": False, "verification_reason": f"{type(error).__name__}: {error}"}
    result = {"path": resolved, "version": version, "source": source, "verified": False}
    if verify:
        # The environment's CONTENT identity, not just its path. Bound into this dict so every
        # semantic key derived from it -- the run id, the rank-cache fingerprint, the feature
        # memmap fingerprints -- changes when an installed dependency's bytes change.
        #
        # The digest is taken from `report.validated_environment_fingerprint`: the value
        # `verify_environment` proved CURRENT for THIS call, either because it recomputed it or
        # because it re-derived the current authoritative digest and matched a recorded capability
        # payload against it. A capability payload is never trusted on its own -- reading
        # `report.gpu_capability["interpreter_fingerprint"]` did exactly that, and a memo-warm reuse
        # returned a cached report whose payload described an environment that had since been
        # rewritten in place (indistinguishable to the cheap advisory check), so the composition
        # identity stayed fixed while the environment changed. When nothing was validated (no
        # capability record matched, a non-GPU model, a non-advisory path) the digest is computed
        # here instead: same function, same interpreter, same parent's-parent prefix the capability
        # key is built from. `report.environment_fingerprint` is deliberately NOT used: it is null
        # on this path, and a cached report's value is derived from the interpreter's SELF-REPORTED
        # prefix, which need not equal the parent's-parent prefix.
        try:
            report = verify_environment(model, probe=True, advisory=True, config=config)
            result.update(verified=report.tier in READY_TIERS,
                          verification_status=report.status, verification_reason=report.reason)
            fingerprint = getattr(report, "validated_environment_fingerprint", None)
        except Exception as error:
            fingerprint = None
            result.update(verification_status="UNAVAILABLE",
                          verification_reason=f"{type(error).__name__}: {error}")
        result["environment_fingerprint"] = fingerprint or _record_environment_fingerprint(model, resolved)
    else:
        result["environment_fingerprint"] = None
        result["verification_status"] = "NOT_PROBED"
        result["verification_reason"] = "environment verification not run in plan mode"
    return result


def _result_outputs(artifacts: Path, protocol: str, model: str, checkpoint: str,
                    attempt: str, subset: str, condition: str, synthetic: bool = False) -> Path:
    base = artifacts / "composition" / "results" / protocol
    if synthetic:
        base /= "SYNTHETIC"
    return assert_composition_destination(artifacts, base / model / checkpoint / attempt / subset /
                                          f"{condition}.jsonl")

def _attempt_id(artifacts: Path, protocol: str, model: str, checkpoint: str,
                run_id: str, synthetic: bool, *, force_new: bool = False) -> str:
    """Keep exports paired by attempt without replacing an earlier result."""
    result_root = artifacts / "composition" / "results" / protocol
    if synthetic:
        result_root /= "SYNTHETIC"
    result_root = result_root / model / checkpoint
    summary_name = f"{'SYNTHETIC_' if synthetic else ''}{model}_{checkpoint}_{run_id}"
    summary_root = artifacts / "composition" / "summary"
    if not force_new and not (result_root / run_id).exists() and not (summary_root / summary_name).exists():
        return run_id
    retry_pattern = re.compile(re.escape(run_id) + r"-retry-(\d+)")
    suffixes = [int(match.group(1)) for path in result_root.glob(f"{run_id}-retry-*")
                if (match := retry_pattern.fullmatch(path.name))]
    suffixes += [int(match.group(1)) for path in summary_root.glob(f"{summary_name}-retry-*")
                 if (match := retry_pattern.search(path.name))]
    return f"{run_id}-retry-{max(suffixes, default=0) + 1}"


def _checkpoint_labels(model: dict[str, Any], checkpoint: str, protocol: str) -> dict[str, Any]:
    """Expose checkpoint provenance directly from its registry record."""
    variant = next((item for item in model.get("checkpoint_variants", [])
                    if item.get("checkpoint_id") == checkpoint), None)
    bundle = next((item for item in model.get("checkpoint_bundles", [])
                   if item.get("bundle_id") == checkpoint), None)
    if bundle:
        members = [item for item in model.get("checkpoint_variants", [])
                   if item.get("checkpoint_id") in bundle.get("required_checkpoint_ids", [])]
        variant = members[0] if members else None
    else:
        members = []
    training_dataset = variant.get("checkpoint_training_dataset") if variant else None
    evaluation_dataset = "FashionIQ" if "fashioniq" in protocol.lower() else None
    transfer = (f"{training_dataset}→{evaluation_dataset} transfer"
                if training_dataset and training_dataset.lower() != "fashioniq" and evaluation_dataset else None)
    labels = {"checkpoint_training_noise_pct": variant.get("checkpoint_training_noise_pct") if variant else None,
              "evaluation_noise_pct": variant.get("evaluation_noise_pct") if variant else None,
              "checkpoint_training_dataset": training_dataset,
              "checkpoint_transfer": transfer,
              "checkpoint_mapping_status": (bundle.get("mapping_status") if bundle else
                                            variant.get("checkpoint_mapping_status") if variant else None),
              "reproduction_status": model.get("reproduction_status"),
              "labels_source": "registry"}
    if bundle:
        labels["checkpoint_bundle_members"] = [
            {key: member.get(key) for key in ("checkpoint_id", "checkpoint_training_noise_pct",
                "checkpoint_training_dataset", "checkpoint_mapping_status")}
            for member in members]
    return labels



def _method_source_hash(method: Callable[..., Any]) -> str | None:
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()



def _module_source_hash(adapter: Any) -> str:
    module_name = adapter.__name__ if inspect.ismodule(adapter) else type(adapter).__module__
    module = sys.modules.get(module_name)
    try:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except (AttributeError, OSError, TypeError):
        return "unavailable"


def _image_identity(paths: Sequence[Path | None]) -> dict[str, Any]:
    """Dataset-content identity: resolved paths plus a rolled-up digest of every file's bytes.

    Content-derived, not `(st_size, st_mtime_ns)`, so previously written rank records and
    run_id-keyed result/summary directories no longer match and are intentionally invalidated.
    """
    resolved = [None if path is None else str(Path(path).expanduser().resolve()) for path in paths]
    content_digests = [None if path is None else sha256_file(Path(path)) for path in resolved]
    return {"resolved_paths_sha256": _digest(resolved), "content_digest": _digest(content_digests),
            "content_identity_method": "resolved absolute path plus sha256 of streamed file content; "
                                       "per-file digests rolled up into one content_digest"}


def model_provenance_identity(model: Mapping[str, Any] | None, config: Any = None) -> dict[str, Any]:
    """Model implementation provenance: upstream commit, approved patches, and vendored source digest.

    Direct callers or models without declared provenance degrade to status 'unknown'.
    """
    if not isinstance(model, Mapping) or not model:
        return {
            "status": "unknown",
            "upstream_commit": None,
            "approved_patches": [],
            "vendored_source_digest": None,
            "provenance_digest": "unknown",
        }

    upstream_commit = model.get("upstream_commit_sha")

    raw_patches = model.get("approved_source_patches")
    approved_patches: list[dict[str, str | None]] = []
    if isinstance(raw_patches, Sequence) and not isinstance(raw_patches, (str, bytes)):
        for patch in raw_patches:
            if isinstance(patch, Mapping):
                approved_patches.append({
                    "path": patch.get("path"),
                    "patch_sha256": patch.get("patch_sha256"),
                    "derived_sha256": patch.get("derived_sha256"),
                })
        approved_patches.sort(key=lambda p: (p["path"] or "", p["patch_sha256"] or ""))

    source_dir = model.get("source_dir")
    source_path = None
    if source_dir and isinstance(source_dir, (str, Path)):
        candidate = Path(source_dir)
        if candidate.is_absolute():
            source_path = candidate
        else:
            tp_root = getattr(config, "WORKBENCH_THIRD_PARTY_ROOT", None)
            if tp_root is not None:
                source_path = Path(tp_root) / candidate
            elif "WORKBENCH_THIRD_PARTY_ROOT" in os.environ:
                source_path = Path(os.environ["WORKBENCH_THIRD_PARTY_ROOT"]) / candidate
            else:
                source_path = Path(__file__).resolve().parents[1] / "third_party" / candidate

    vendored_digest: str | None = None
    if source_path is not None and source_path.is_dir():
        from workbench.backend.vendored_source import live_digest, manifest_digest
        try:
            live = live_digest(source_path, dict(model))
            vendored_digest = manifest_digest(live)
        except (OSError, ValueError):
            vendored_digest = "unavailable"

    has_provenance = (upstream_commit is not None or approved_patches or vendored_digest is not None)
    status = "available" if has_provenance else "unknown"
    payload = {
        "upstream_commit": upstream_commit,
        "approved_patches": approved_patches,
        "vendored_source_digest": vendored_digest,
    }
    digest = _digest(payload) if has_provenance else "unknown"
    return {
        "status": status,
        "upstream_commit": upstream_commit,
        "approved_patches": approved_patches,
        "vendored_source_digest": vendored_digest,
        "provenance_digest": digest,
    }


def run_composition(data: Any, model_specs: Sequence[ModelSpec], config: Any, *,
                    subsets: Sequence[str], batch_size: int = 32, smoke: int | None = None,
                    resume: bool = False, conditions: Sequence[str] = CONDITIONS,
                    image_resolver: Callable[[str, Mapping[str, str]], Path | None] | None = None,
                    synthetic: bool | None = None, seed: int = 0,
                    topk_depth: int | None = None,
                    keep_features: bool = False) -> dict[str, Any]:
    """Evaluate models sequentially; gallery stays full and all outputs stay in composition/.

    ``topk_depth`` opts in to persisting candidate Top-K windows (see ``workbench/composition/topk.py``).
    It defaults to ``None``, which writes nothing and costs nothing: a normal metric run is
    byte-for-byte the run it was before this parameter existed. When set, candidates come from the
    SAME ``adapter.score()`` matrix that produced the exact ranks, so they cannot describe a
    different nearest-neighbour method, gallery, preprocessing step or tie policy.

    ``keep_features`` preserves the encoded gallery/query memmaps under
    ``composition/runs/<run_id>/work/<model>/features`` instead of deleting them at the end of the
    model. It defaults to off (the historical behaviour) and is implied by ``topk_depth``, because a
    deleted cache makes candidate windows rebuildable only by re-encoding the full gallery. The
    directory is ~2.4 GB per model and is a deliberate, documented cleanup target, not silent waste.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if smoke is not None and smoke < 1:
        raise ValueError("smoke query limit must be positive")
    if not conditions or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"conditions must be selected from {CONDITIONS}")
    from workbench.composition.topk import normalise_depth, open_export, abort_export, close_export

    # Validated before any work happens: an unusable depth must refuse the run, not surface after
    # a full gallery encode.
    if topk_depth is not None:
        normalise_depth(topk_depth, 1)
    artifacts = Path(config.WORKBENCH_ARTIFACTS_ROOT).expanduser().resolve()
    composition = assert_composition_destination(artifacts, artifacts / "composition")
    for subset in subsets:
        if subset not in {"full", "sf", "validated"}:
            raise ValueError(f"unknown subset {subset!r}")
    selected_subsets = set(subsets)
    queries = tuple(query for query in data.queries if selected_subsets.intersection(query.subsets))
    if smoke is not None:
        queries = queries[:smoke]
    if not queries:
        raise ValueError("release has no queries")
    gallery_ids = tuple(data.gallery.image_ids)
    gallery_hash = _gallery_fingerprint(data)
    query_hash = _query_fingerprint(queries)
    is_synthetic = bool(data.provenance.get("is_synthetic")) if synthetic is None else synthetic
    positive_positions = _positive_positions(data)
    reports = []


    for spec in model_specs:
        model_lock = None
        model_id = _component(spec.model_id)
        checkpoint_id = _component(spec.checkpoint_id)
        model = spec.registry_model or {}
        labels = _checkpoint_labels(model, checkpoint_id, data.gallery.protocol_id)
        # Computed before the try: the summary and the harness report below both carry it, and a
        # provenance builder that raised must not be able to lose the run's own numbers.
        variant_provenance = composition_variant_provenance(model, config)
        attempt = uuid.uuid4().hex
        layout = model.get("fashioniq_layout", "fashioniq_standard")
        resolver = image_resolver or image_resolver_factory(Path(config.FASHIONIQ_ROOT), layout)
        gallery_paths = [resolver(identity, data.candidate_names) for identity in gallery_ids]
        query_paths = [resolver(query.reference_id, data.candidate_names) for query in queries]
        missing_gallery = [gallery_ids[index] for index, path in enumerate(gallery_paths) if path is None]
        if missing_gallery:
            reason = f"{len(missing_gallery)} released gallery identities unresolved by stem"
            summaries = _write_failed_results(data, artifacts, model_id, checkpoint_id, attempt,
                subsets, conditions, queries, reason, is_synthetic, labels,
                execution_status="FAILED", run_id=attempt)
            reports.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                            "status": "BLOCKED_MISSING_GALLERY_IMAGES", "reason": reason,
                            "unresolvable_count": len(missing_gallery), "examples": missing_gallery[:20],
                            "summaries": summaries})
            continue
        if any(path is None for path in gallery_paths):
            raise AssertionError("gallery path check did not preserve complete released order")

        adapter = None
        adapter_source_hash = "unavailable"
        run_id = None
        work_root = None
        mode_status: dict[str, dict[str, Any]] = {}
        # Declared before the try so the `finally` below can abort a half-written export even when
        # the body raised before any writer was opened (then the dict is simply empty).
        writers: dict[str, Any] = {}
        try:
            adapter = spec.adapter_factory(config, checkpoint_id)
            adapter_source_hash = _module_source_hash(adapter)
            runtime_identity = model_interpreter_identity(model, config, adapter)
            if not getattr(adapter, "reference_eligible", True):
                raise ValueError("adapter excludes the reference; CIRCUS requires reference-eligible ranks")
            # MEASURED, not declared. `reference_eligible` has been a class-level literal `True` in
            # every adapter, so writing it into the artifact asserted a reference policy nobody
            # executed (limn measures False with a reason). The probe runs the adapter's own
            # score+rank on a synthetic gallery; an adapter that cannot answer offline is reported
            # as `unmeasurable` with the reason, which is the honest answer.
            reference_eligibility = adapter.measured_reference_eligibility()
            reference_eligible = bool(reference_eligibility.get("reference_eligible")) \
                if reference_eligibility.get("measured") else None
            scopes = tuple(getattr(adapter, "category_scopes", (None,)))
            if not scopes or len(set(scopes)) != len(scopes):
                raise ValueError(f"invalid adapter category_scopes: {scopes!r}")
            assignments = [_scope_for_query(adapter, query, data.candidate_names) for query in queries]
            query_scopes = [scope for scope, _reason in assignments]
            scope_errors = {query.qid: reason for query, (_scope, reason) in zip(queries, assignments) if reason}
            supported = {}
            for condition in conditions:
                try:
                    supported[condition] = adapter.mode_supported(condition)
                except Exception as error:
                    supported[condition] = (False, f"{type(error).__name__}: {error}")
            gpu_identity = _gpu_identity()
            checkpoint_identity = spec.checkpoint_identity or "unknown"
            condition_transform_source_hash = _module_source_hash(sys.modules.get("workbench.composition.inputs"))
            scoring_module_source_hash = _module_source_hash(
                sys.modules.get("workbench.composition.adapters"))
            identity = {
                "model_id": model_id, "checkpoint_id": checkpoint_id,
                "checkpoint_identity": checkpoint_identity, "protocol_id": data.gallery.protocol_id,
                "gallery_image_identity": _image_identity(gallery_paths),
                "query_image_identity": _image_identity(query_paths),
                "gallery_identity": gallery_hash, "query_order_identity": query_hash,
                # The resolved per-query scope map is a scoring INPUT, not metadata: it decides
                # which gallery matrix each query is scored against. `query_order_identity` keys on
                # the loader's `query.category`, which stays `None` for queries an adapter resolver
                # scopes -- so wiring a resolver without this digest would leave the fingerprint
                # unchanged and `--resume` would silently reuse the old all-null ranks. Hashed by
                # qid so the same assignment independent of query order is the same experiment.
                "query_scope_resolution": _scope_resolution_identity(queries, query_scopes),
                "subset_memberships": sorted(set(subsets)), "conditions": list(conditions),
                "preprocessing": {"id": model.get("preprocessing_id"),
                                  "registry_values": {key: model.get(key) for key in
                                      ("fashioniq_layout", "fashioniq_image_extensions", "transform", "image_size",
                                       "text_processor", "candidate_normalization", "scoring")}},
                "model_provenance": model_provenance_identity(model, config),
                "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
                "adapter_source_hashes": {name: _method_source_hash(getattr(type(adapter), name))
                                           for name in ("encode_queries", "encode_targets", "score", "rank",
                                                        "mode_supported")},
                "adapter_module_source_hash": adapter_source_hash,
                "adapter_module_source_status": "available" if adapter_source_hash != "unavailable" else "unavailable",
                # The shared scoring/decode helpers live in `composition/adapters.py`, which is NOT
                # the adapter's own module: `adapter_module_source_hash` hashes `model_<id>.py` and
                # `adapter_source_hashes` hashes five methods, so a change to a shared helper that
                # an adapter calls would otherwise reuse an old rank artifact under an unchanged
                # run_id. Hashed here so it mints a new run_id like every other scoring input.
                "scoring_module_source_hash": scoring_module_source_hash,
                "scoring_module_source_status": "available" if scoring_module_source_hash != "unavailable" else "unavailable",
                # The BLAS thread count is an ARITHMETIC input, not just a performance knob:
                # measured on a real `@`-based scorer, serial BLAS=1 vs BLAS=10 changed rank for
                # 11 of 512 queries (max |drank| 1) and moved nDCG by 2.181e-05 at full 6003-query
                # scale on ENCODER. `_rank_groups_concurrently` pins it to 1, but that helper lives
                # in THIS file, which `_module_source_hash` never covers -- so without an explicit
                # field a rank artifact written under one policy would keep matching the fingerprint
                # under another and `--resume` would silently reuse it. Recorded here so the run id
                # partitions on the arithmetic policy.
                #
                # `blas_threads_pinned` is the value we ASK for; `blas_pin` is what the runtime
                # confirmed. They are recorded separately because a pin that silently did nothing
                # was measured on a real runtime -- recording only the requested number let such a
                # run claim a policy it never applied.
                "blas_threads_pinned": BLAS_THREADS_PINNED,
                # What the runtime is able to report about the pin, PROBED fresh here rather than
                # read from the last scoring call. Using the mutable per-call record made the run id
                # depend on whether a previous run in this process had already pinned, so the same
                # inputs produced two different ids. This value is a property of the interpreter, so
                # it is stable across runs in one process and still refuses to claim verification
                # the runtime cannot provide.
                "blas_pin_capability": _blas_pin_capability(),
                # Which cache-integrity policy produced the features this run scored. A `header`
                # or `off` load is a weaker guarantee than `full`, and an artifact must not imply
                # otherwise.
                "cache_verify_mode": _cache_verify_mode(),
                "condition_transform_source_hash": condition_transform_source_hash,
                "condition_transform_source_status": "available" if condition_transform_source_hash != "unavailable" else "unavailable",
                "score_signature": str(inspect.signature(adapter.score)),
                "mode_support": {condition: supported[condition] for condition in conditions},
                "runtime_identity": runtime_identity,
                "gpu_identity": gpu_identity,
                "category_scopes": list(scopes),
                "reference_eligible": reference_eligible,
                "reference_eligibility": reference_eligibility,
                # `seed` is deliberately NOT hashed here. It is recorded in the summary for
                # provenance, but no seeding call exists anywhere in the composition path: every
                # encode-time transform is deterministic, every adapter encodes under `.eval()` +
                # `torch.no_grad()`, and no forward pass draws randomness. A field that mints a new
                # run_id while controlling nothing is worse than no field, so `--seed` no longer
                # partitions the artifact tree. `metrics.paired_bootstrap_gap` keeps its own
                # explicit seed argument; it is a function parameter, not a run variable.
                "batch_size": batch_size,
                # LIMN's own encoder batch (WORKBENCH_LIMN_BATCH_SIZE) changes the forward pass
                # only, but it is part of the run's configuration and must key the run id.
                "adapter_batch_size": getattr(adapter, "batch_size", None),
            }
            run_id = _digest(identity)[:24]
            fingerprint = _digest(identity)
            run_dir = assert_composition_destination(artifacts, composition / "runs" / run_id)
            work_root = assert_composition_destination(artifacts, run_dir / "work" / model_id)
            work_root.mkdir(parents=True, exist_ok=True)
            adapter.work_root = work_root
            rank_path = assert_composition_destination(artifacts, run_dir / "ranks" / f"{model_id}_{checkpoint_id}.json")
            # ponytail: per-model lock serialises the whole section (~2x concurrent wall time); use per-feature-file locks if throughput matters; assumes a filesystem that honours flock, as a no-op lock reintroduces the race.
            model_lock = _acquire_model_lock(assert_composition_destination(
                artifacts, run_dir / f".{model_id}_{checkpoint_id}.lock"))

            attempt = _attempt_id(artifacts, data.gallery.protocol_id, model_id, checkpoint_id,
                                  run_id, is_synthetic, force_new=rank_path.exists())
            ranks: dict[str, dict[str, int | None]] | None = None
            reasons_by_condition: dict[str, dict[str, str]] = {}
            reused = False
            retry_conditions: set[str] = set()
            if resume and adapter_source_hash != "unavailable" and rank_path.is_file():
                try:
                    saved = json.loads(rank_path.read_text(encoding="utf-8"))
                    if (isinstance(saved, dict) and saved.get("fingerprint") == fingerprint and
                            _rank_payload_is_valid(saved.get("ranks"), conditions,
                                                  {query.qid for query in queries}, len(gallery_ids))):
                        saved_status = saved.get("condition_status", {})
                        saved_failures = saved.get("query_failures", {})
                        transient = {condition for condition in conditions
                                     if (saved_status.get(condition, {}).get("status") in
                                         {"MODEL_UNAVAILABLE", "FAILED", "EXACT_RANK_UNAVAILABLE"}
                                         or saved_status.get(condition, {}).get("status") == "PARTIAL" and
                                         any(not _is_stable_failure(reason)
                                             for reason in saved_failures.get(condition, {}).values()))
                                     and not (saved_failures.get(condition) and all(
                                         _is_stable_failure(reason)
                                         for reason in saved_failures[condition].values()))}
                        if transient:
                            retry_conditions = transient if len(transient) < len(conditions) else set(conditions)
                            ranks = {condition: dict(saved["ranks"][condition]) for condition in conditions}
                            reasons_by_condition = dict(saved.get("query_failures", {}))
                            mode_status = dict(saved_status)
                            for condition in retry_conditions:
                                ranks[condition] = {query.qid: None for query in queries}
                                reasons_by_condition.pop(condition, None)
                                mode_status.pop(condition, None)
                        else:
                            ranks = saved["ranks"]
                            reasons_by_condition = saved.get("query_failures", {})
                            mode_status = saved_status
                            reused = True
                            attempt = saved.get("attempt", run_id)
                except (OSError, ValueError, TypeError):
                    ranks = None
            if not reused:
                attempt = _attempt_id(artifacts, data.gallery.protocol_id, model_id, checkpoint_id,
                                      run_id, is_synthetic,
                                      force_new=rank_path.exists() and (not resume or bool(retry_conditions)))
            # Opened AFTER `attempt` is final (so a retry cannot publish under the previous
            # attempt's name) and BEFORE the rank loop. `open_export` returns {} when depth is off,
            # which makes every later call below a no-op -- that is the zero-cost-when-off contract.
            #
            # RESUME TRAP: a resumed run reuses its ranks, so it used to open no writers at all and
            # finished COMPLETED with candidate_topk=false -- and every later `--topk-depth ...
            # --resume` reused those ranks again, so the export could NEVER appear. The export is
            # therefore keyed per ATTEMPT (never in `identity`, or the same measurement would mint a
            # new run_id and look like a duplicate), and a resumed run that still has its feature
            # cache backfills the window from those cached vectors instead of re-encoding the
            # gallery.
            export_conditions = [condition for condition in conditions
                                 if supported[condition][0]
                                 and (not retry_conditions or condition in retry_conditions)]
            writers = {} if reused else open_export(
                artifacts, run_id,
                attempt=attempt, model_id=model_id, checkpoint_id=checkpoint_id,
                protocol_id=data.gallery.protocol_id,
                gallery_identity=gallery_hash,
                gallery_count=len(gallery_ids),
                depth=topk_depth or 0,
                conditions=export_conditions,
                reference_eligible=reference_eligible,
                source="model_run")
            export_blocked_reason: str | None = None
            # Only a RESUMED run needs a backfill: when the rank loop actually runs, the export is
            # written from the scoring call itself and this branch must not touch it.
            if reused and topk_depth and not _export_complete(artifacts, run_id,
                                                              export_conditions, topk_depth):
                writers = open_export(
                    artifacts, run_id,
                    attempt=str(attempt), model_id=model_id, checkpoint_id=checkpoint_id,
                    protocol_id=data.gallery.protocol_id,
                    gallery_identity=gallery_hash, gallery_count=len(gallery_ids),
                    depth=topk_depth, conditions=export_conditions,
                    reference_eligible=reference_eligible, source="model_run_resume_backfill")
                try:
                    backfilled = _backfill_export(
                        adapter, queries, query_scopes, scopes, positive_positions,
                        reasons_by_condition, work_root, fingerprint, export_conditions,
                        writers, batch_size, original_ranks=ranks)
                except BackfillRankDisagreement as disagreement:
                    # The export does not describe the measurement that already exists. Refuse it
                    # by name rather than publishing a window scored under another policy.
                    abort_export(writers)
                    writers = {}
                    backfilled = {}
                    export_blocked_reason = f"backfill refused: {disagreement}"
                if not backfilled and export_blocked_reason is None:
                    # The cache was removed after the original rank pass. Saying "COMPLETED" here
                    # would be the exact lie this branch exists to stop, so the request is refused
                    # by name and the caller must re-run without --resume.
                    abort_export(writers)
                    writers = {}
                    export_blocked_reason = (
                        "candidate Top-K was requested but this run's feature cache is gone, so the "
                        "window cannot be rebuilt without re-encoding the full gallery; re-run "
                        "without --resume")
            projection_totals: dict[str, int] = {}
            category_failure_counts: dict[str, int] = {}
            # Which conditions needed the per-query fallback. A fallback is a DIFFERENT arithmetic
            # shape from the group pass (batch N -> batch 1), so a condition that used it cannot be
            # claimed to have completed as one pass; this map is what the summary reports.
            fallback_used: dict[str, bool] = {}
            # The pin readback is process-global. Clear it HERE, at the boundary of this run's
            # scoring phase, so `policy_observed` can only ever report a pin applied by THIS run --
            # never one left behind by the previous model or condition in the same process. This is
            # the mirror of `_blas_pin_capability`'s cache, which is a property of the interpreter
            # and survives the reset on purpose.
            #
            # Cleared even on the reuse path: a resumed run reports "not recorded", which is true,
            # rather than inheriting a stale verification from whatever scored before it.
            _reset_effective_blas_pin()
            if ranks is None or retry_conditions:
                if ranks is None:
                    ranks = {condition: {query.qid: None for query in queries} for condition in conditions}
                active_conditions = [condition for condition in conditions if supported[condition][0]
                                     and (not retry_conditions or condition in retry_conditions)]
                for condition in conditions:
                    if not supported[condition][0] and condition not in mode_status:
                        mode_status[condition] = {"status": "ABLATION_UNSUPPORTED",
                                                  "reason": supported[condition][1]}
                target_maps: dict[str | None, Any] = {}
                target_errors: dict[str | None, str] = {}
                if active_conditions:
                    for scope_index, scope in enumerate(scopes):
                        scope_name = _scope_name(scope)
                        target_file = work_root / "features" / f"gallery_{scope_name}.npy"
                        target_fp = _digest({"run": fingerprint, "scope": scope, "kind": "full-gallery-features"})
                        try:
                            target_maps[scope], projection = _encode_gallery(
                                adapter, gallery_paths, scope, target_file, target_fp, batch_size,
                                len(scopes) - scope_index, len(queries), len(active_conditions))
                            projection_totals[scope_name] = projection.get("projected_bytes", 0)
                        except Exception as error:
                            target_errors[scope] = f"{type(error).__name__}: {error}"
                for condition in active_conditions:
                    condition_reasons: dict[str, str] = dict(scope_errors)
                    category_errors_all: dict[str, str] = {}
                    feature_errors_all: dict[str, str] = {}
                    query_features: dict[str | None, Any] = {}
                    for scope in scopes:
                        indexes = [index for index, query_scope in enumerate(query_scopes)
                                   if query_scope == scope and queries[index].qid not in scope_errors]
                        if not indexes or scope in target_errors or scope not in target_maps:
                            continue
                        scope_name = _scope_name(scope)
                        feature_file = work_root / "features" / f"query_{condition}_{scope_name}.npy"
                        feature_fp = _digest({"run": fingerprint, "condition": condition, "scope": scope,
                                              "kind": "query-features"})
                        if condition in retry_conditions:
                            for suffix in ("", ".json", ".errors.json", ".category-errors.json"):
                                feature_file.with_suffix(feature_file.suffix + suffix).unlink(missing_ok=True)
                        try:
                            values, errors, category_errors = _encode_queries(
                                adapter, queries, query_paths, indexes, condition, work_root,
                                feature_file, feature_fp, target_maps[scope].shape[-1], batch_size,
                                query_scopes)
                            query_features[scope] = values
                            feature_errors_all.update(errors)
                            category_errors_all.update(category_errors)
                        except Exception as error:
                            feature_errors_all.update({queries[index].qid: f"{type(error).__name__}: {error}"
                                                       for index in indexes})
                    condition_reasons.update(feature_errors_all)
                    condition_reasons.update(category_errors_all)
                    for scope, error in target_errors.items():
                        for index, query_scope in enumerate(query_scopes):
                            if query_scope == scope:
                                condition_reasons[queries[index].qid] = error
                    for scope in scopes:
                        if scope not in query_features or scope not in target_maps:
                            continue
                        indices = [index for index, query_scope in enumerate(query_scopes)
                                   if query_scope == scope and queries[index].qid not in condition_reasons]
                        features = query_features[scope]
                        targets = target_maps if len(scopes) > 1 else target_maps[scope]
                        # Group shape is `batch_size` -- the SAME as before the ranking became
                        # concurrent -- because how many queries ride in one `adapter.score` call
                        # is part of the arithmetic (the BLAS matrix M dimension). Concurrency is
                        # across these groups, so `WORKBENCH_SCORE_THREADS` changes timing only and
                        # can never change a publishable number.
                        groups = []
                        for offset in range(0, len(indices), batch_size):
                            batch_indices = indices[offset:offset + batch_size]
                            query_vectors = np.asarray(features[batch_indices], dtype=np.float32)
                            finite_rows = np.isfinite(query_vectors).all(axis=1)
                            valid_indices = [index for index, good in zip(batch_indices, finite_rows) if good]
                            invalid_indices = [index for index, good in zip(batch_indices, finite_rows) if not good]
                            for index in invalid_indices:
                                condition_reasons[queries[index].qid] = "query features unavailable"
                            if not valid_indices:
                                continue
                            group_positions = [positive_positions.get(queries[index].qid, ())
                                               for index in valid_indices]
                            for index, positive in zip(valid_indices, group_positions):
                                if not positive:
                                    condition_reasons[queries[index].qid] = "no positive target resolved in exact released gallery"
                            keep = [slot for slot, positive in enumerate(group_positions) if positive]
                            valid_indices = [valid_indices[slot] for slot in keep]
                            group_positions = [group_positions[slot] for slot in keep]
                            if not valid_indices:
                                continue
                            groups.append((valid_indices, group_positions,
                                           np.asarray(features[valid_indices], dtype=np.float32),
                                           [query_scopes[index] for index in valid_indices]))
                        decode_depth = writers[condition].depth if condition in writers else 0

                        def rank_group(group: Any) -> Any:
                            """One group, ranked by exactly the call the serial loop made.

                            The group's shape is `batch_size`, so the adapter sees the same matrix
                            M dimension as before; concurrency happens strictly ACROSS groups.
                            """
                            group_indices, group_positions, group_vectors, group_cats = group
                            if decode_depth:
                                return adapter.rank_and_topk(group_vectors, targets, group_positions,
                                                             group_cats, decode_depth)
                            return adapter.rank(group_vectors, targets, group_positions,
                                                group_cats), []

                        # Rank every group concurrently, capturing a per-group failure instead of
                        # letting one bad group abort the rest; results are consumed in group order.
                        group_results = _rank_groups_concurrently(rank_group, groups,
                                                                  resolved_score_threads())

                        for group, outcome in zip(groups, group_results):
                            valid_indices, positions, vectors, cats = group
                            exact_ranks = windows = None
                            batch_error = None
                            if isinstance(outcome, BaseException):
                                batch_error = outcome
                            else:
                                exact_ranks, windows = outcome
                            try:
                                if batch_error is not None:
                                    raise batch_error
                                if decode_depth:
                                    from workbench.composition.topk import write_batch

                                    write_batch(writers, condition,
                                                [queries[index].qid for index in valid_indices],
                                                exact_ranks, windows)
                                else:
                                    del windows
                                if len(exact_ranks) != len(valid_indices):
                                    raise ValueError("adapter.rank returned wrong number of query ranks")
                            except Exception as batch_error:
                                from workbench.composition.adapters import ExactRankUnavailable
                                if isinstance(batch_error, ExactRankUnavailable):
                                    for condition_query in queries:
                                        ranks[condition][condition_query.qid] = None
                                    mode_status[condition] = {"status": "EXACT_RANK_UNAVAILABLE",
                                                              "reason": str(batch_error)}
                                    condition_reasons = {}
                                    break
                                exact_ranks = []
                                # The batch failed AFTER `_rank_groups_concurrently` restored the
                                # previous BLAS count, so this per-query retry would otherwise score
                                # at the process default (10 by default) while the group pass scored
                                # at BLAS_THREADS_PINNED. Re-acquire the same policy: a fallback that
                                # changes the arithmetic is not a fallback, it is a different
                                # measurement, and a batch that only succeeds this way must not be
                                # reported as the same pass.
                                fallback_pin = _blas_thread_limit(BLAS_THREADS_PINNED)
                                _record_effective_blas_pin(fallback_pin)
                                try:
                                    for index, vector, positive, category in zip(valid_indices, vectors, positions, cats):
                                        try:
                                            one = adapter.rank(vector.reshape(1, -1), targets, [positive], [category])
                                            exact_ranks.append(one[0] if one else None)
                                        except ExactRankUnavailable as error:
                                            mode_status[condition] = {"status": "EXACT_RANK_UNAVAILABLE",
                                                                      "reason": str(error)}
                                            exact_ranks = []
                                            break
                                        except Exception as error:
                                            exact_ranks.append(None)
                                            condition_reasons[queries[index].qid] = f"{type(error).__name__}: {error}"
                                finally:
                                    _blas_thread_limit(fallback_pin)
                                # The fallback changed the batch matrix to one query at a time,
                                # which is a different arithmetic shape from the group pass. The
                                # condition is therefore marked PARTIAL downstream (unscored_count
                                # > 0) rather than COMPLETED, and `fallback_used` records that this
                                # condition did not complete as one pass.
                                fallback_used[condition] = True
                            if mode_status.get(condition, {}).get("status") == "EXACT_RANK_UNAVAILABLE":
                                break
                            for index, rank in zip(valid_indices, exact_ranks):
                                if rank is not None and (type(rank) is not int or rank < 1 or rank > len(gallery_ids)):
                                    condition_reasons[queries[index].qid] = f"invalid full-gallery exact rank: {rank!r}"
                                    continue
                                ranks[condition][queries[index].qid] = rank
                        if mode_status.get(condition, {}).get("status") == "EXACT_RANK_UNAVAILABLE":
                            break
                    category_failure_counts[condition] = len(category_errors_all) + len(scope_errors)
                    reasons_by_condition[condition] = condition_reasons
                    scored_count = sum(rank is not None for rank in ranks[condition].values())
                    unscored_count = len(ranks[condition]) - scored_count
                    current_status = mode_status.get(condition, {}).get("status")
                    if current_status not in {"EXACT_RANK_UNAVAILABLE", "ABLATION_UNSUPPORTED"}:
                        if not scored_count:
                            reason = next(iter(condition_reasons.values()), "no query produced an exact rank")
                            mode_status[condition] = {"status": "MODEL_UNAVAILABLE", "reason": reason,
                                                      "unscored_count": unscored_count,
                                                      "category_scope_failure_count": len(category_errors_all) + len(scope_errors)}
                        else:
                            # A condition that needed the per-query fallback scored at batch 1
                            # instead of the group's batch, which is a different arithmetic shape.
                            # That is recorded explicitly (`fallback_used`) rather than silently:
                            # the status stays COMPLETED when every query got a rank, because the
                            # CLI maps PARTIAL to a non-zero exit and a write failure that recovered
                            # fully has not lost a measurement. What must never happen is the shape
                            # change going unrecorded, which is what `fallback_used` prevents.
                            mode_status[condition] = {"status": "PARTIAL" if unscored_count else "COMPLETED",
                                                      "unscored_count": unscored_count,
                                                      "fallback_used": bool(fallback_used.get(condition)),
                                                      "category_scope_failure_count": len(category_errors_all) + len(scope_errors)}
                _atomic_json(rank_path, {"fingerprint": fingerprint, "identity": identity,
                                         "attempt": attempt, "ranks": ranks,
                                         "query_failures": reasons_by_condition,
                                         "condition_status": mode_status,
                                         # OUTSIDE the hashed `identity`: what the runtime actually
                                         # reported for THIS run's scoring calls. Two fields in
                                         # `identity` describe the request and the interpreter's
                                         # capability; this one describes the observation. Keeping it
                                         # out of the hash is deliberate -- see
                                         # `observed_policy_record` for why putting it in would make
                                         # the run id depend on process history.
                                         "policy_observed": observed_policy_record()})
            # Published here -- after every condition has been ranked and before any summary is
            # written -- so a manifest exists exactly when the rows it describes exist. A condition
            # that the run did not re-score has no writer and is reported absent, not partial.
            topk_export_state: dict[str, Any] = {"requested": bool(topk_depth), "depth": None,
                                                 "conditions": [], "partial": False,
                                                 "missing_query_count": 0,
                                                 "status": ("TOPK_EXPORT_REQUESTED" if topk_depth
                                                            else "TOPK_EXPORT_NOT_REQUESTED"),
                                                 "reason": None}
            if writers:
                expected = {condition: [query.qid for query in queries]
                            for condition in writers}
                published = close_export(writers, expected_qids=expected)
                manifests = [json.loads(path.with_name(path.stem + ".manifest.json")
                                        .read_text(encoding="utf-8"))
                             for path in published.values()]
                partial = any(item.get("partial") for item in manifests)
                topk_export_state = {
                    "requested": True,
                    "depth": manifests[0].get("depth") if manifests else None,
                    "conditions": sorted(published),
                    "partial": partial,
                    "missing_query_count": sum(int(item.get("missing_query_count") or 0)
                                               for item in manifests),
                    "status": "TOPK_EXPORT_COMPLETE_PARTIAL" if partial else "TOPK_EXPORT_COMPLETE",
                    "reason": None,
                }
                writers = {}
            elif export_blocked_reason:
                # An export was requested and could not be produced. This is NOT success, and the
                # state says exactly why so a consumer can never read `candidate_topk: false` as
                # "the run simply chose not to export".
                topk_export_state.update(status="TOPK_EXPORT_BLOCKED",
                                         reason=export_blocked_reason)
            if model_lock is not None:
                _release_model_lock(model_lock)
                model_lock = None

            query_rows_by_condition = {}
            for condition in conditions:
                failures = reasons_by_condition.get(condition, {})
                query_rows_by_condition[condition] = [
                    {"qid": query.qid, "index": query.index, "condition": condition,
                     "rank": ranks.get(condition, {}).get(query.qid),
                     "scored": ranks.get(condition, {}).get(query.qid) is not None,
                     "failure": failures.get(query.qid), "synthetic": is_synthetic,
                     "checkpoint_labels": labels} for query in queries]
            from workbench.composition.metrics import compose
            # A requested export that produced nothing is a non-success outcome: the run measured
            # what it measured, but it did not deliver what was asked of it.
            export_blocked = bool(export_blocked_reason)
            # Snapshot ONCE, after every scoring call for this run has returned and before any
            # summary is written. Snapshotted rather than read inline per subset so all three
            # subset summaries of one run report the same observation -- they describe one scoring
            # phase, and a later mutation could not change that.
            policy_observed = observed_policy_record()
            summary_by_subset = {}
            for subset in subsets:
                selected = [query for query in queries if subset in query.subsets]
                qids = [query.qid for query in selected]
                subset_ranks = {condition: {qid: ranks.get(condition, {}).get(qid) for qid in qids}
                                for condition in conditions}
                result = compose(subset_ranks, query_count=len(qids))
                for condition in conditions:
                    output = _result_outputs(artifacts, data.gallery.protocol_id,
                                             model_id, checkpoint_id, attempt, subset, condition, is_synthetic)
                    _atomic_jsonl(output, [row for row in query_rows_by_condition[condition]
                                           if row["qid"] in set(qids)])
                summary_name = f"{'SYNTHETIC_' if is_synthetic else ''}{model_id}_{checkpoint_id}_{attempt}"
                summary_path = assert_composition_destination(artifacts, composition / "summary" /
                                                              summary_name / f"{subset}.json")
                transient_statuses = {"MODEL_UNAVAILABLE", "FAILED", "EXACT_RANK_UNAVAILABLE"}
                has_transient_failure = any(status.get("status") in transient_statuses
                                             for status in mode_status.values())
                has_success = any(status.get("status") in {"COMPLETED", "PARTIAL"}
                                  for status in mode_status.values())
                has_failure = has_transient_failure or any(status.get("status") in
                                    {"PARTIAL", "ABLATION_UNSUPPORTED"} for status in mode_status.values())
                execution_status = ("REUSED" if reused else "FAILED" if has_transient_failure and not has_success
                                    else "PARTIAL" if has_failure else
                                    "EXPORT_INCOMPLETE" if export_blocked else "COMPLETED")
                summary = {
                    "checkpoint_labels": labels,
                    "synthetic": is_synthetic,
                    "adapter_module_source_status": identity["adapter_module_source_status"],
                    "schema_version": 1, "model_id": model_id, "checkpoint_id": checkpoint_id,
                    "checkpoint_identity": checkpoint_identity, "protocol_id": data.gallery.protocol_id,
                    "gallery_identity_sha256": gallery_hash, "gallery_count": len(gallery_ids),
                    "query_count": len(qids),
                    "query_count_per_subset": {name: sum(name in query.subsets for query in queries)
                                               for name in ("full", "sf", "validated")},
                    "subset": subset, "conditions": list(conditions),
                    "preprocessing": {"policy": "adapter-native transform after path-based raw-image intervention",
                                      "ablation_policy": "black PIL replacement is staged before adapter transform; empty text for image-only"},
                    "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
                    "category_scopes": list(scopes),
                    "category_resolution": data.provenance.get("category_resolution", {}),
                    "unresolved_category_query_count": sum(query.category is None for query in queries),
                    # `category_resolution` above is the RELEASE-level fact and keeps saying 119
                    # ambiguous; this is what the adapter resolver actually routed, so a 6001
                    # scored count does not read as a contradiction.
                    "adapter_query_scope_resolution": {
                        "resolver": (f"{type(adapter).__module__}."
                                     f"{type(adapter).__qualname__}.query_scope"
                                     if callable(getattr(adapter, "query_scope", None)) else None),
                        "loader_scoped": sum(query.category is not None for query in queries),
                        "adapter_routed": sum(scope is not None for scope in query_scopes),
                        "still_unscoped": sum(scope is None for scope in query_scopes),
                        "unscoped_qids": [query.qid for query, scope in zip(queries, query_scopes)
                                          if scope is None],
                    },
                    "reference_eligible": reference_eligible,
                    "reference_eligibility": reference_eligibility,
                    "runtime": {"controller_python": sys.version, "platform": platform.platform(),
                                "gpu": gpu_identity, "model_interpreter": runtime_identity},
                    # Recorded, not applied: see the identity note above. Kept in the summary so a
                    # reader can see what the operator passed without it forking the run tree.
                    "seed": seed, "seed_applied": False,
                    "release_artifact_hashes": data.provenance.get("sha256", {}),
                    "hash_verification": data.provenance.get("hash_verification"),
                    "failed_expectations": data.provenance.get("failed_expectations", ()),
                    "gallery_candidate_basename_match": data.provenance.get("gallery_candidate_basename_match"),
                    "paper_vs_release_gallery": data.provenance.get("gallery_note", {
                        "paper_stated_images": PAPER_STATED_GALLERY_IMAGES,
                        "released_images": RELEASED_GALLERY_IMAGES}),
                    "execution_status": execution_status,
                    "condition_status": mode_status,
                    "checkpoint_training_noise_pct": labels["checkpoint_training_noise_pct"],
                    "reproduction_status": labels["reproduction_status"],
                    "evaluation_noise_pct": labels["evaluation_noise_pct"],
                    "checkpoint_training_dataset": labels["checkpoint_training_dataset"],
                    "checkpoint_transfer": labels["checkpoint_transfer"],
                    "checkpoint_mapping_status": labels["checkpoint_mapping_status"],
                    "category_scope_failures": {condition: category_failure_counts.get(condition,
                                                       sum(reason.startswith(f"{model_id} category scope")
                                                           for reason in reasons_by_condition.get(condition, {}).values()))
                                                 for condition in conditions},
                    "smoke": {"limited": smoke is not None, "query_limit": smoke,
                              "label": "SMOKE QUERY LIMIT; full released gallery preserved" if smoke is not None else None},
                    "is_synthetic": is_synthetic, "synthetic_fixture": is_synthetic,
                    "feature_storage": {"format": "float32 .npy memmap", "projected_bytes": projection_totals,
                                        "arrays_removed_after_rank": True},
                    "topk_export": topk_export_state,
                    # Provenance fact: how thoroughly the feature cache behind this run was
                    # verified on load. Written into the summary (not only the identity block)
                    # because `measurement_identity` reads the summary, and a `header`-verified run
                    # must never be grouped with a fully-verified one.
                    "cache_verify_mode": _cache_verify_mode(),
                    # What the runtime reported for THIS run's scoring calls, so a served row can
                    # show it without reading the rank file. Deliberately outside `identity`:
                    # `measurement_identity` reads the summary for grouping, and a value that is
                    # only known after scoring must not change the group a run belongs to.
                    "policy_observed": policy_observed,
                    "variant_provenance": variant_provenance,
                    "metrics": asdict(result), "run_id": run_id, "resume_key": fingerprint,
                    "reused_query_result_count": len(queries) * len(conditions) if reused else 0,
                }
                _atomic_json(summary_path, summary)
                summary_by_subset[subset] = str(summary_path)
            successful_modes = {"COMPLETED", "PARTIAL"}
            successful = any(status.get("status") in successful_modes for status in mode_status.values())
            incomplete = (any(status.get("status") not in successful_modes for status in mode_status.values())
                          or any(status.get("status") == "PARTIAL" for status in mode_status.values()))
            model_status = ("MODEL_UNAVAILABLE" if not successful else
                            "PARTIAL" if incomplete else
                            "EXPORT_INCOMPLETE" if export_blocked else
                            "REUSED" if reused else "COMPLETED")
            model_reason = next((status.get("reason") for status in mode_status.values()
                                 if status.get("status") not in successful_modes), None)
            reports.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                            "status": model_status, "reason": model_reason,
                            "run_id": run_id, "attempt": attempt,
                            "reused": reused, "reused_query_result_count": len(queries) * len(conditions) if reused else 0,
                            "query_count": len(queries), "gallery_count": len(gallery_ids),
                            "category_scopes": list(scopes), "condition_status": mode_status,
                            "adapter_module_source_status": identity["adapter_module_source_status"],
                            "runtime": {"model_interpreter": runtime_identity},
                            "summaries": summary_by_subset})
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            summaries = _write_failed_results(data, artifacts, model_id, checkpoint_id, attempt,
                subsets, conditions, queries, reason, is_synthetic, labels,
                execution_status="FAILED", run_id=run_id if "run_id" in locals() else attempt)
            reports.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                            "status": "FAILED", "reason": reason,
                            "adapter_module_source_status": "available" if adapter_source_hash != "unavailable" else "unavailable",
                            "query_count": len(queries), "gallery_count": len(gallery_ids),
                            "run_id": locals().get("run_id"), "attempt": attempt, "summaries": summaries})
            mode_status[conditions[0]] = {"status": "FAILED", "reason": reason}
        finally:
            model_work = locals().get("work_root")
            if model_work is not None and model_work.exists():
                try:
                    if any(status.get("status") in {"MODEL_UNAVAILABLE", "FAILED", "EXACT_RANK_UNAVAILABLE"}
                           for status in mode_status.values()):
                        log_dir = assert_composition_destination(
                            artifacts, composition / "logs" / f"{model_id}_{checkpoint_id}" / str(attempt))
                        for worker_log in model_work.rglob("worker.stderr.log"):
                            try:
                                log_dir.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(worker_log, log_dir / "worker.stderr.log")
                            except Exception:
                                pass
                except Exception:
                    pass
                # The encoded gallery matrix is ~2.4 GB per model. It used to be deleted after
                # EVERY run, which made a feature cache that can never be reused -- and turned the
                # resume/backfill path into dead code (candidate windows could only be rebuilt by
                # re-encoding 74,381 images).
                #
                # ponytail: keep `features/` only when the run either asked for a candidate export
                # or the operator passed --keep-features; everything else (worker logs, staged query
                # images, job dirs) is still removed every time. Ceiling: an operator who keeps
                # features for N models holds ~2.4N GB until they delete it deliberately -- that is
                # the point, and `composition runs/<id>/work/<model>/features` is the documented
                # cleanup target. Upgrade path: prune by run age once a sweep job exists, or store
                # features as float16/quantized (~half the bytes) if the disk becomes the limit.
                keep_features = bool(topk_depth) or bool(keep_features)
                if keep_features:
                    for child in model_work.iterdir():
                        if child.name == "features":
                            continue
                        if child.is_dir():
                            shutil.rmtree(child, ignore_errors=True)
                        else:
                            try:
                                child.unlink()
                            except OSError:
                                pass
                else:
                    shutil.rmtree(model_work, ignore_errors=True)
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass
            if writers:
                # No-op for writers already published above; the safety net for any path that left
                # an un-closed export behind (an exception, an early continue, a retry branch).
                try:
                    abort_export(writers)
                except Exception:
                    pass
                writers = {}
            if model_lock is not None:
                _release_model_lock(model_lock)

    return {"run_count": len(reports), "models": reports, "gallery_count": len(gallery_ids),
            "gallery_identity_sha256": gallery_hash, "protocol_id": data.gallery.protocol_id,
            "synthetic": is_synthetic}


def image_resolver_factory(dataset_root: Path, layout: str):
    return image_resolver(dataset_root, layout)


def _write_failed_results(data: Any, artifacts: Path, model_id: str, checkpoint_id: str,
                          attempt: str, subsets: Sequence[str], conditions: Sequence[str],
                          queries: Sequence[Any], reason: str, synthetic: bool = False,
                          labels: dict[str, Any] | None = None, *, execution_status: str = "FAILED",
                          condition_status: dict[str, Any] | None = None,
                          run_id: str | None = None) -> dict[str, str]:
    from workbench.composition.metrics import compose

    summary_paths = {}
    for subset in subsets:
        selected = [query for query in queries if subset in query.subsets]
        qids = {query.qid for query in selected}
        failed_ranks = {condition: {query.qid: None for query in selected} for condition in conditions}
        for condition in conditions:
            output = _result_outputs(artifacts, data.gallery.protocol_id, model_id, checkpoint_id,
                                     attempt, subset, condition, synthetic)
            _atomic_jsonl(output, [{"qid": query.qid, "index": query.index,
                                    "condition": condition, "rank": None, "scored": False,
                                    "failure": reason, "synthetic": synthetic,
                                    "checkpoint_labels": labels or {}} for query in selected])
        summary_name = f"{'SYNTHETIC_' if synthetic else ''}{model_id}_{checkpoint_id}_{attempt}"
        summary_path = assert_composition_destination(artifacts, artifacts / "composition" / "summary" /
                                                      summary_name / f"{subset}.json")
        summary = {"synthetic": synthetic, "is_synthetic": synthetic, "model_id": model_id,
                   "checkpoint_id": checkpoint_id, "protocol_id": data.gallery.protocol_id,
                   "gallery_count": len(data.gallery.image_ids), "query_count": len(qids),
                   "subset": subset, "conditions": list(conditions), "execution_status": execution_status,
                   "condition_status": condition_status or {condition: {"status": execution_status,
                       "reason": reason, "unscored_count": len(selected)} for condition in conditions},
                   "checkpoint_labels": labels or {},
                   "checkpoint_training_noise_pct": (labels or {}).get("checkpoint_training_noise_pct"),
                   "evaluation_noise_pct": (labels or {}).get("evaluation_noise_pct"),
                   "checkpoint_training_dataset": (labels or {}).get("checkpoint_training_dataset"),
                   "checkpoint_transfer": (labels or {}).get("checkpoint_transfer"),
                   "checkpoint_mapping_status": (labels or {}).get("checkpoint_mapping_status"),
                   "reproduction_status": (labels or {}).get("reproduction_status"),
                   "metrics": asdict(compose(failed_ranks, query_count=len(selected))),
                   "run_id": run_id or attempt}
        _atomic_json(summary_path, summary)
        summary_paths[subset] = str(summary_path)
    return summary_paths


def inspect_image_inventory(data: Any, models: Sequence[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    """Report missing released gallery, reference, and positive images without downloading."""
    gallery_ids = tuple(data.gallery.image_ids)
    references = {data.candidate_names.get(query.reference_id, query.reference_id)
                  for query in data.queries}
    positives = {data.candidate_names.get(identity, identity)
                 for query in data.queries for identity in query.positive_ids}
    missing_gallery: dict[str, list[str]] = {}
    missing_references: dict[str, list[str]] = {}
    missing_positives: dict[str, list[str]] = {}
    extension_counts: dict[str, int] = {}
    for model in models:
        model_id = str(model.get("model_id", "unknown"))
        resolver = image_resolver(dataset_root, model.get("fashioniq_layout", "fashioniq_standard"))

        def missing(values: Sequence[str] | set[str]) -> list[str]:
            return sorted(identity for identity in values
                          if resolver(identity, data.candidate_names) is None)

        missing_gallery[model_id] = missing(gallery_ids)
        missing_references[model_id] = missing(references)
        missing_positives[model_id] = missing(positives)
        for identity in gallery_ids:
            path = resolver(identity, data.candidate_names)
            if path is not None:
                original = PurePosixPath(normalize_release_path(
                    data.candidate_names.get(identity, identity))).name
                key = "exact" if original == path.name else path.suffix.lower()
                extension_counts[key] = extension_counts.get(key, 0) + 1

    gallery_missing = {identity for values in missing_gallery.values() for identity in values}
    reference_missing = {identity for values in missing_references.values() for identity in values}
    positive_missing = {identity for values in missing_positives.values() for identity in values}
    status = "missing" if gallery_missing or reference_missing or positive_missing else (
        "extension_mismatch" if any(key != "exact" for key in extension_counts) else "complete")
    return {"status": status, "released_gallery_count": len(gallery_ids),
            "local_resolved_by_stem_count": len(gallery_ids) - len(gallery_missing),
            "unresolvable_gallery_count": len(gallery_missing),
            "unresolvable_gallery_examples": sorted(gallery_missing)[:20],
            "missing_gallery_by_model": missing_gallery,
            "query_reference_count": len(references),
            "query_references_resolved_count": len(references) - len(reference_missing),
            "query_reference_missing_count": len(reference_missing),
            "query_reference_missing_examples": sorted(reference_missing)[:20],
            "query_reference_missing_by_model": missing_references,
            "positive_target_count": len(positives),
            "positive_targets_resolved_count": len(positives) - len(positive_missing),
            "positive_target_missing_count": len(positive_missing),
            "positive_target_missing_examples": sorted(positive_missing)[:20],
            "positive_target_missing_by_model": missing_positives,
            "extension_resolution_counts": extension_counts, "downloaded": False}
