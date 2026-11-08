"""Bounded candidate Top-K export for CIRCUS composition runs.

The candidates written here come from the SAME ranking that produced the metrics: one
``adapter.score()`` call yields both the exact target rank and these rows
(``CompositionAdapter.rank_and_topk``), so a stored candidate can never come from a different
nearest-neighbour method, gallery, preprocessing step or tie policy than the number it explains.

Two facts are stored side by side and never derived from each other:

* ``target_rank`` -- the exact 1-based position of the best positive in the complete eligible
  gallery. It survives even when it is far outside the stored depth.
* ``topk`` -- the leading ``depth`` gallery positions with their scores, in the scorer's own
  order. It is a bounded window, never a renumbering of the target.

ponytail: JSONL, one line per query, gallery POSITIONS as integers, whole file parsed when a
query page is served. Ceiling: a 6003x50x3 export is ~10 MB per run. Upgrade path: move the row
store to Parquet/DuckDB keyed (run_id, condition, qid) once a run needs depth > 200 or thousands
of runs must be browsed at once.
"""
from __future__ import annotations

import json
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCHEMA_VERSION = 1
ARTIFACT_TYPE = "circus_candidate_topk"
#: Depth a run exports when it opts in without naming one. The UI shows Top-10 and Top-50, so 50
#: is the smallest depth that can serve both without a second export.
DEFAULT_EXPORT_DEPTH = 50
#: Hard ceiling; a deeper window costs a linear share of the gallery per query and explains
#: nothing the exact target rank does not already say.
MAX_EXPORT_DEPTH = 500

TIE_POLICY = "descending score, ties broken by earlier gallery position"
CANDIDATES_DIR = "candidates"


class TopKExportError(RuntimeError):
    """The export is requested with parameters that cannot describe a real ranking."""


def gallery_digest(protocol_id: str, image_ids: Sequence[str],
                   candidate_names: Mapping[str, str]) -> str:
    """Digest of the exact gallery identity a candidate position indexes into.

    Byte-identical to the runner's ``_gallery_fingerprint`` (pinned by
    ``tests/test_composition_topk_export.py``): a position is only meaningful against the ordered
    gallery that produced it, so the export records this digest and the reader refuses to resolve
    positions against a different one.
    """
    payload = json.dumps({"protocol_id": protocol_id,
                          "ordered_gallery": list(image_ids),
                          "candidate_names": dict(sorted(candidate_names.items()))},
                         sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalise_depth(depth: int | None, gallery_count: int) -> int | None:
    """Validate an opt-in export depth, or ``None`` when the export is not requested."""
    if depth is None or depth == 0:
        return None
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 1:
        raise TopKExportError(f"top-k export depth must be a positive integer or None, got {depth!r}")
    if depth > MAX_EXPORT_DEPTH:
        raise TopKExportError(f"top-k export depth {depth} exceeds the supported maximum {MAX_EXPORT_DEPTH}")
    if gallery_count < 1:
        raise TopKExportError("top-k export needs a non-empty gallery")
    return min(depth, gallery_count)


def topk_from_scores(scores: Any, depth: int) -> list[tuple[int, float]]:
    """Leading ``depth`` ``(position, score)`` pairs in the scorer's own order.

    The tie policy is "descending score, earlier gallery position first" -- exactly the rule
    ``CompositionAdapter._ranks_from_scores`` counts (``1 + #greater + #earlier-equal``), so a
    candidate rank read off this list equals the metric rank for the same query by construction.
    A full stable argsort is correct but sorts the whole 74k-row gallery per query; this selects
    the boundary score first and then keeps every strictly-greater row plus the earliest ties, which
    is the same prefix for a fraction of the work.
    """
    import numpy as np

    row = np.asarray(scores, dtype=np.float32)
    if row.ndim != 1:
        raise TopKExportError(f"expected one score row, got shape {row.shape}")
    if depth < 1 or row.size == 0:
        return []
    if depth >= row.size:
        order = np.argsort(-row, kind="stable")
        return [(int(position), float(row[position])) for position in order]
    boundary = np.partition(row, row.size - depth)[row.size - depth]
    above = np.flatnonzero(row > boundary)
    tied = np.flatnonzero(row == boundary)  # ascending position == "earlier first"
    order = np.concatenate([above, tied[:depth - len(above)]])
    # `above` is in position order already; re-sort by descending score, then position.
    order = order[np.argsort(-row[order], kind="stable")]
    return [(int(position), float(row[position])) for position in order]


def candidates_dir(artifacts_root: Path, run_id: str) -> Path:
    return Path(artifacts_root).expanduser().resolve() / "composition" / "runs" / _component(run_id) / CANDIDATES_DIR


def _component(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise TopKExportError(f"unsafe output path component: {value!r}")
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class CandidateTopKWriter:
    """Stream one condition's candidates to ``runs/<run_id>/candidates/<model>_<ckpt>/``.

    Bound to one (run, condition, gallery) tuple at construction: a row written through this
    object cannot describe a different query set, gallery or model than its manifest declares.
    """

    directory: Path
    run_id: str
    attempt: str
    model_id: str
    checkpoint_id: str
    condition: str
    protocol_id: str
    gallery_identity: str
    gallery_count: int
    depth: int
    score_semantics: str = "adapter.score native candidate score"
    #: The adapter's MEASURED reference policy, or None when it could not be probed offline. A
    #: literal True here would be the same false provenance claim the runner used to write.
    reference_eligible: bool | None = None
    source: str = "model_run"

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise TopKExportError("candidate writer needs a positive depth")
        if self.gallery_count < 1:
            raise TopKExportError("candidate writer needs a non-empty gallery")
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._path = self.directory / f"{_component(self.condition)}.jsonl"
        # Left over from an interrupted attempt in the same run directory: the manifest is the
        # completion marker, so a stale file without one is discarded rather than appended to.
        self._path.unlink(missing_ok=True)
        (self.directory / f"{_component(self.condition)}.manifest.json").unlink(missing_ok=True)
        self._temporary = self._path.with_name(f".{self._path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
        self._stream = self._temporary.open("w", encoding="utf-8")
        self._written: set[str] = set()
        self._closed = False

    def write(self, qid: str, target_rank: int | None, ranked: Sequence[tuple[int, float]]) -> None:
        """Append one query's candidates. ``ranked`` is ``topk_from_scores`` output."""
        if self._closed:
            raise TopKExportError("candidate writer is already closed")
        if qid in self._written:
            raise TopKExportError(f"duplicate query id {qid!r} in one condition export")
        for position, _score in ranked:
            if not isinstance(position, int) or not 0 <= position < self.gallery_count:
                raise TopKExportError(f"candidate position {position!r} is outside the gallery")
        if len(ranked) > self.depth:
            raise TopKExportError(f"{len(ranked)} candidates exceed the declared depth {self.depth}")
        if target_rank is not None and (isinstance(target_rank, bool) or not isinstance(target_rank, int)
                                        or not 1 <= target_rank <= self.gallery_count):
            raise TopKExportError(f"target rank {target_rank!r} is outside the gallery")
        row = {"qid": qid, "target_rank": target_rank,
               "topk": [[position, score] for position, score in ranked]}
        self._stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self._written.add(qid)

    def close(self, *, expected_qids: Iterable[str] | None = None) -> Path:
        """Flush, publish atomically, and write the manifest that marks the export complete."""
        if self._closed:
            return self._path
        self._stream.close()
        self._closed = True
        os.replace(self._temporary, self._path)
        expected = set(expected_qids) if expected_qids is not None else None
        missing = sorted(expected - self._written) if expected is not None else []
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "artifact_type": ARTIFACT_TYPE,
            "run_id": self.run_id,
            "attempt": self.attempt,
            "model_id": self.model_id,
            "checkpoint_id": self.checkpoint_id,
            "condition": self.condition,
            "protocol_id": self.protocol_id,
            "gallery_identity_sha256": self.gallery_identity,
            "gallery_count": self.gallery_count,
            "depth": self.depth,
            "tie_policy": TIE_POLICY,
            "score_semantics": self.score_semantics,
            "score_present": True,
            "reference_eligible": self.reference_eligible,
            "reference_eligibility_measured": self.reference_eligible is not None,
            "source": self.source,
            "file": self._path.name,
            "file_sha256": _sha256_file(self._path),
            "file_bytes": self._path.stat().st_size,
            "record_count": len(self._written),
            "expected_record_count": len(expected) if expected is not None else None,
            "missing_query_count": len(missing),
            "missing_query_examples": missing[:20],
            # A partial export is honest about it instead of looking like a run that scored
            # fewer queries: the per-query fallback path runs without a candidate window.
            "partial": bool(missing),
        }
        _atomic_json(self.directory / f"{self._path.stem}.manifest.json", manifest)
        self.manifest = manifest
        return self._path

    def abort(self) -> None:
        if self._closed:
            return
        self._stream.close()
        self._closed = True
        self._temporary.unlink(missing_ok=True)


def open_export(artifacts_root: Path, run_id: str, *, attempt: str, model_id: str,
                checkpoint_id: str, protocol_id: str, gallery_identity: str,
                gallery_count: int, depth: int, conditions: Sequence[str],
                score_semantics: str = "adapter.score native candidate score",
                reference_eligible: bool | None = None,
                source: str = "model_run") -> dict[str, CandidateTopKWriter]:
    """Open one writer per condition. Returns an empty mapping when ``depth`` opts out.

    ``source`` records WHERE the rows came from. ``"model_run"`` is the scorer that produced the
    run's own metrics; anything else (an imported released artifact) is a different provenance and
    the reader/UI must be able to say so.
    """
    resolved = normalise_depth(depth, gallery_count)
    if resolved is None:
        return {}
    if not source or not isinstance(source, str):
        raise TopKExportError(f"candidate export needs a non-empty source, got {source!r}")
    directory = candidates_dir(artifacts_root, run_id) / f"{_component(model_id)}_{_component(checkpoint_id)}"
    return {condition: CandidateTopKWriter(
        directory=directory, run_id=run_id, attempt=attempt, model_id=model_id,
        checkpoint_id=checkpoint_id, condition=condition, protocol_id=protocol_id,
        gallery_identity=gallery_identity, gallery_count=gallery_count, depth=resolved,
        score_semantics=score_semantics, reference_eligible=reference_eligible, source=source)
        for condition in conditions}


def close_export(writers: Mapping[str, CandidateTopKWriter],
                 expected_qids: Mapping[str, Iterable[str]] | None = None) -> dict[str, Path]:
    """Close every writer, aborting all of them if any one cannot publish."""
    published: dict[str, Path] = {}
    try:
        for condition, writer in writers.items():
            expected = expected_qids.get(condition) if expected_qids else None
            published[condition] = writer.close(expected_qids=expected)
    except BaseException:
        for writer in writers.values():
            writer.abort()
        raise
    return published


def write_batch(writers: Mapping[str, CandidateTopKWriter], condition: str,
                qids: Sequence[str], ranks: Sequence[int | None],
                windows: Sequence[Sequence[tuple[int, float]]]) -> None:
    """Store one already-scored batch: ``adapter.rank_and_topk`` output, zipped to its qids.

    This is the single call a runner makes at its rank site. It refuses a length mismatch rather
    than pairing a query with another query's window, which is the one failure mode that would put
    a plausible-looking but wrong candidate under a real rank.
    """
    writer = writers.get(condition)
    if writer is None:
        return
    if not (len(qids) == len(ranks) == len(windows)):
        raise TopKExportError(
            f"candidate batch shape mismatch: {len(qids)} qids, {len(ranks)} ranks, "
            f"{len(windows)} windows")
    for qid, rank, window in zip(qids, ranks, windows, strict=True):
        writer.write(qid, rank, window)


def abort_export(writers: Mapping[str, CandidateTopKWriter]) -> None:
    for writer in writers.values():
        writer.abort()


_ROWS_CACHE: dict[tuple[str, float, int], tuple[dict[str, Any], dict[str, dict[str, Any]]]] = {}


def read_candidates(artifacts_root: Path, run_id: str, condition: str = "multimodal", *,
                    verify_digest: bool = True) -> dict[str, Any]:
    """Read one condition's export for a run.

    Returns ``{"available": bool, "status": str, "reason": str|None, "manifest": dict|None,
    "rows": {qid: row}, "path": str|None, "digest_verified": bool|None}``. Every failure is a
    named status, never an empty dict pretending the run stored nothing.
    """
    base = candidates_dir(artifacts_root, run_id)
    if not base.is_dir():
        return _unavailable("TOPK_EXPORT_NOT_REQUESTED",
                            "no candidate export directory for this run", None)
    manifests = sorted(base.glob(f"*/{_component(condition)}.manifest.json"))
    if not manifests:
        partial = sorted(base.glob(f"*/{_component(condition)}.jsonl"))
        if partial:
            return _unavailable("TOPK_EXPORT_INCOMPLETE",
                                "candidate rows exist without a completion manifest", partial[0])
        return _unavailable("TOPK_EXPORT_NOT_REQUESTED",
                            "no candidate export for this condition", None)
    manifest_path = manifests[0]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return _unavailable("TOPK_EXPORT_UNREADABLE", f"{type(error).__name__}: {error}", manifest_path)
    if not isinstance(manifest, dict) or manifest.get("artifact_type") != ARTIFACT_TYPE:
        return _unavailable("TOPK_EXPORT_UNREADABLE", "manifest is not a candidate top-k export",
                            manifest_path)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return _unavailable("TOPK_EXPORT_UNREADABLE",
                            f"unsupported schema_version {manifest.get('schema_version')!r}",
                            manifest_path)
    if str(manifest.get("run_id")) != run_id:
        return _unavailable("TOPK_EXPORT_MISMATCH",
                            f"manifest describes run {manifest.get('run_id')!r}", manifest_path)
    rows_path = manifest_path.parent / str(manifest.get("file") or f"{condition}.jsonl")
    try:
        stat = rows_path.stat()
    except OSError:
        return _unavailable("TOPK_EXPORT_INCOMPLETE", "manifest has no row file", rows_path)

    key = (str(rows_path), stat.st_mtime, stat.st_size)
    cached = _ROWS_CACHE.get(key)
    if cached is not None:
        cached_manifest, rows = cached
        return {"available": True, "status": "AVAILABLE", "reason": None,
                "manifest": cached_manifest, "rows": rows, "path": str(rows_path),
                "digest_verified": cached_manifest.get("_digest_verified")}

    digest_verified: bool | None = None
    expected_digest = manifest.get("file_sha256")
    if verify_digest and expected_digest:
        digest_verified = _sha256_file(rows_path) == expected_digest
        if not digest_verified:
            return _unavailable("TOPK_EXPORT_CORRUPT",
                                "row file digest does not match its manifest", rows_path)

    rows: dict[str, dict[str, Any]] = {}
    try:
        with rows_path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("qid"), str) \
                        or not isinstance(row.get("topk"), list):
                    raise ValueError(f"{rows_path}:{line_number} is not a candidate row")
                if row["qid"] in rows:
                    raise ValueError(f"{rows_path}:{line_number} repeats query {row['qid']!r}")
                rows[row["qid"]] = row
    except (OSError, ValueError) as error:
        return _unavailable("TOPK_EXPORT_UNREADABLE", f"{type(error).__name__}: {error}", rows_path)

    manifest = dict(manifest, _digest_verified=digest_verified)
    _ROWS_CACHE[key] = (manifest, rows)
    return {"available": True, "status": "AVAILABLE", "reason": None, "manifest": manifest,
            "rows": rows, "path": str(rows_path), "digest_verified": digest_verified}


def _unavailable(status: str, reason: str, path: Path | None) -> dict[str, Any]:
    return {"available": False, "status": status, "reason": reason, "manifest": None,
            "rows": {}, "path": str(path) if path is not None else None,
            "digest_verified": None}


def export_index(artifacts_root: Path, run_id: str) -> dict[str, dict[str, Any]]:
    """Cheap per-condition index (manifest only, no rows) for run headers and list endpoints."""
    base = candidates_dir(artifacts_root, run_id)
    index: dict[str, dict[str, Any]] = {}
    if not base.is_dir():
        return index
    for manifest_path in sorted(base.glob("*/*.manifest.json")):
        try:
            stat = manifest_path.stat()
        except OSError:
            continue
        key = (str(manifest_path), stat.st_mtime, stat.st_size)
        cached = _MANIFEST_CACHE.get(key)
        if cached is None:
            try:
                cached = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(cached, dict) or cached.get("artifact_type") != ARTIFACT_TYPE:
                continue
            if str(cached.get("run_id")) != run_id:
                continue
            _MANIFEST_CACHE[key] = cached
        index[str(cached.get("condition"))] = dict(cached, path=str(manifest_path))
    return index


_MANIFEST_CACHE: dict[tuple[str, float, int], dict[str, Any]] = {}
