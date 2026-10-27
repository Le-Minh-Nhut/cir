"""Load released CIRCUS/FashionIQ identities without changing their order."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

RELEASE_PROTOCOL_ID = "circus_fashioniq_release_v1"
PAPER_STATED_GALLERY_IMAGES = 18853
RELEASED_GALLERY_IMAGES = 74381
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
SUBSET_ORDER = ("full", "sf", "validated")
EXPECTED_QUERY_COUNTS = {"full": 6003, "sf": 4069, "validated": 586}
VALIDATED_PARTS = ("validated_solved", "validated_unsolved")
EXPECTED_VALIDATED_PART_COUNTS = {"validated_solved": 368, "validated_unsolved": 218}
EXPECTED_SHA256 = {
    "mbeir_query": "b601ef6a5411bf9385cff41e64bea863871ab1094cd09d78683de6a5141165ef",
    "mbeir_candidates": "121ec284832c33289ee13bfac19d5d59dc3f8aeddf672d784fd5eb8bcdf79eef",
    "sf": "95a93883334e351e4d3aaaf25e47e89b48d4daf02bdb786e8f00d58165ace37d",
    "validated_solved": "18f829da4d3b1ee8439f43998bae43dbbe8924c2eaafba84473ae849f564f5c7",
    "validated_unsolved": "ba019a30d3b45f4b29c34d2d48638dbeeed2e2a42409f625922c65de5b2a7efa",
    "aggregated_audit": "27bdbef041200dc393b5478d3e57121bc16c70a9acc2e98a19fb48408a99533c",
}


@dataclass(frozen=True)
class ReleasePaths:
    root: Path
    mbeir_query: Path
    mbeir_candidates: Path
    sf: Path
    validated_solved: Path
    validated_unsolved: Path
    aggregated_audit: Path
def resolve_release_paths(root: Path, *, validated_root: Path | None = None) -> ReleasePaths:
    """Resolve release paths; `validated_root` overrides validation directory.

    Released validation files are nested at
    `final_dataset/query_jsonl/fashioniq/<part>/mbeir_fashioniq_task7_test_<part>.jsonl`.
    Direct downloaded artifact basenames are also accepted for query/candidate/validated files.
    """
    validation_dir = validated_root or root / "final_dataset/query_jsonl/fashioniq"

    def choose(primary: Path, downloaded_name: str) -> Path:
        downloaded = root / downloaded_name
        return primary if primary.exists() or not downloaded.exists() else downloaded

    def validated_path(part: str) -> Path:
        nested = validation_dir / part / f"mbeir_fashioniq_task7_test_{part}.jsonl"
        flat = validation_dir / f"{part}.jsonl"
        downloaded = root / f"{part}.jsonl"
        return next((path for path in (nested, flat, downloaded) if path.exists()), nested)

    return ReleasePaths(
        root,
        choose(root / "mbeir_fashioniq_task7_test.jsonl", "mbeir_query.jsonl"),
        choose(root / "mbeir_fashioniq_task7_cand_pool.jsonl", "mbeir_cand_pool.jsonl"),
        root / "shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl"
        if (root / "shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl").exists()
        else root / "mbeir_fashioniq_task7_test_multimodal.jsonl",
        validated_path("validated_solved"),
        validated_path("validated_unsolved"),
        choose(root / "shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json",
               "retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json"),
    )


def sha256_file(path: Path) -> str:
    """Hash a file incrementally using 1 MiB reads."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class HashReport:
    path: Path
    expected: str | None
    actual: str

    @property
    def matches(self) -> bool | None:
        return None if self.expected is None else self.expected == self.actual


def verify_release_hashes(paths: ReleasePaths) -> tuple[HashReport, ...]:
    """Report file digests against locally pinned release-byte hashes."""
    return tuple(HashReport(path, EXPECTED_SHA256.get(name), sha256_file(path))
                 for name, path in _path_items(paths))


@dataclass(frozen=True)
class Query:
    qid: str
    index: int
    query_txt: str
    reference_id: str
    positive_ids: tuple[str, ...]
    category: str | None
    subsets: tuple[str, ...]


@dataclass(frozen=True)
class Gallery:
    protocol_id: str
    image_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.image_ids)


@dataclass(frozen=True)
class ReleaseData:
    gallery: Gallery
    queries: tuple[Query, ...]
    candidate_names: dict[str, str]
    provenance: dict[str, object]


def _path_items(paths: ReleasePaths) -> tuple[tuple[str, Path], ...]:
    return tuple((name, getattr(paths, name)) for name in EXPECTED_SHA256)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected JSON object in {path}:{line_number}")
            rows.append(row)
    return rows


def _release_fields(row: dict[str, Any], *, kind: str, source: str,
                    row_number: int = 0) -> tuple[Any, ...]:
    """Parse verified candidate (`did`, `img_path`) and query release fields."""
    if kind == "candidate":
        candidate_id, filename = row.get("did"), row.get("img_path")
        if candidate_id is None or filename is None:
            raise ValueError(f"candidate row {row_number} lacks did or img_path")
        return str(candidate_id), str(filename)
    qid, query_txt = row.get("qid"), row.get("query_txt")
    reference, positives = row.get("query_img_path"), row.get("pos_cand_list")
    if qid is None or query_txt is None or reference is None or positives is None:
        missing = [key for key, value in (("qid", qid), ("query_txt", query_txt),
                   ("query_img_path", reference), ("pos_cand_list", positives)) if value is None]
        raise ValueError(f"{source} row is missing required fields {missing}")
    positives = (positives,) if isinstance(positives, str) else tuple(positives)
    if not all(isinstance(value, (str, int)) for value in positives):
        raise ValueError(f"query {qid!r} in {source} has invalid positive candidate ids")
    return str(qid), str(query_txt), str(reference), tuple(map(str, positives))


def _resolve_candidate_id(value: str, candidate_names: dict[str, str], filenames: dict[str, str], *, qid: str) -> str:
    if value in candidate_names:
        return value
    if value in filenames:
        return filenames[value]
    raise ValueError(f"query {qid!r} refers to unresolved candidate id {value!r}")


def _normalize_gallery_path(path: str) -> str:
    """Remove literal released placeholder root and preserve relative path identity/order."""
    prefix = "<path_to_M-BEIR>/"
    if not isinstance(path, str) or not path.startswith(prefix):
        raise ValueError(f"unexpected gallery path root in {path!r}")
    return path[len(prefix):]


def _by_query_id(rows: list[tuple[Any, ...]], source: str) -> dict[str, tuple[Any, ...]]:
    result = {}
    for row in rows:
        if row[0] in result:
            raise ValueError(f"duplicate query id {row[0]!r} in {source}")
        result[row[0]] = row
    return result


def _require_count(label: str, rows: list[object], expected: int) -> None:
    if len(rows) != expected:
        raise ValueError(f"{label} count {len(rows)} != expected {expected}")


def load_release(root: Path, *, verify_hashes: bool = True,
                 expect_released_counts: bool = True) -> ReleaseData:
    """Load release rows; set `expect_released_counts=False` only for synthetic fixtures.

    Synthetic loads skip pinned hash verification, preserve structural identity checks, and mark
    provenance with `is_synthetic` and every unmet released count/membership expectation.
    """
    paths = resolve_release_paths(root)
    hash_reports = verify_release_hashes(paths) if verify_hashes and expect_released_counts else ()
    candidate_rows = _read_jsonl(paths.mbeir_candidates)
    full_rows = _read_jsonl(paths.mbeir_query)
    sf_rows = _read_jsonl(paths.sf)
    validated_rows = {part: _read_jsonl(getattr(paths, part)) for part in VALIDATED_PARTS}
    with paths.aggregated_audit.open(encoding="utf-8") as stream:
        audit = json.load(stream)
    gallery_paths = audit.get("gallery_paths") if isinstance(audit, dict) else None
    if not isinstance(gallery_paths, list):
        raise ValueError(f"{paths.aggregated_audit} has no gallery_paths list")
    gallery_ids = tuple(_normalize_gallery_path(path) for path in gallery_paths)

    failed_expectations = []
    counts = {
        "candidate": (len(candidate_rows), RELEASED_GALLERY_IMAGES),
        "gallery": (len(gallery_ids), RELEASED_GALLERY_IMAGES),
        "full": (len(full_rows), EXPECTED_QUERY_COUNTS["full"]),
        "sf": (len(sf_rows), EXPECTED_QUERY_COUNTS["sf"]),
        **{part: (len(validated_rows[part]), EXPECTED_VALIDATED_PART_COUNTS[part])
           for part in VALIDATED_PARTS},
    }
    for name, (actual, expected) in counts.items():
        if actual != expected:
            failed_expectations.append(f"{name}_count:{actual}!={expected}")
            if expect_released_counts:
                raise ValueError(f"{name} count {actual} != expected {expected}")

    candidate_names = {}
    for index, row in enumerate(candidate_rows, 1):
        candidate_id, filename = _release_fields(row, kind="candidate", source="candidate table", row_number=index)
        if candidate_id in candidate_names:
            raise ValueError(f"duplicate candidate id {candidate_id!r} in candidate table")
        candidate_names[candidate_id] = filename
    filename_ids = {filename: candidate_id for candidate_id, filename in candidate_names.items()}

    parsed_full = [_release_fields(row, kind="query", source="full") for row in full_rows]
    parsed_sf = [_release_fields(row, kind="query", source="sf") for row in sf_rows]
    parsed_parts = {part: [_release_fields(row, kind="query", source=part) for row in validated_rows[part]]
                    for part in VALIDATED_PARTS}
    full_by_id = _by_query_id(parsed_full, "full")
    sf_ids = set(_by_query_id(parsed_sf, "sf"))
    solved_ids = set(_by_query_id(parsed_parts["validated_solved"], "validated_solved"))
    unsolved_ids = set(_by_query_id(parsed_parts["validated_unsolved"], "validated_unsolved"))
    overlap = solved_ids & unsolved_ids
    if overlap:
        raise ValueError(f"validated parts overlap by identity ({len(overlap)} ids): {sorted(overlap)[:10]}")
    validated_ids = solved_ids | unsolved_ids
    for label, missing in (
        ("validated_subset_of_sf", validated_ids - sf_ids),
        ("sf_subset_of_full", sf_ids - set(full_by_id)),
        ("validated_subset_of_full", validated_ids - set(full_by_id)),
    ):
        if missing:
            failed_expectations.append(f"{label}:{len(missing)} missing ids {sorted(missing)[:10]}")
            raise ValueError(f"{label.replace('_', ' ')}; missing {len(missing)} ids: {sorted(missing)[:10]}")
    if len(validated_ids) != EXPECTED_QUERY_COUNTS["validated"]:
        failed_expectations.append(
            f"validated_union_count:{len(validated_ids)}!={EXPECTED_QUERY_COUNTS['validated']}"
        )
        if expect_released_counts:
            raise ValueError(
                f"validated identity union count {len(validated_ids)} != {EXPECTED_QUERY_COUNTS['validated']}"
            )

    gallery_basenames = {Path(path).name for path in gallery_ids}
    candidate_basenames = {Path(path).name for path in candidate_names.values()}
    if (gallery_basenames != candidate_basenames
            or len(gallery_basenames) != len(gallery_ids)
            or len(candidate_basenames) != len(candidate_names)):
        raise ValueError(
            "gallery/candidate basename mismatch: "
            f"gallery={len(gallery_basenames)}, candidates={len(candidate_basenames)}, "
            f"gallery_only={sorted(gallery_basenames - candidate_basenames)[:10]}, "
            f"candidate_only={sorted(candidate_basenames - gallery_basenames)[:10]}"
        )

    memberships = {
        qid: ("full",) + (("sf",) if qid in sf_ids else ()) +
        (("validated",) if qid in validated_ids else ())
        for qid in full_by_id
    }
    queries = []
    for index, (qid, query_txt, reference, positives) in enumerate(parsed_full):
        reference_id = _resolve_candidate_id(reference, candidate_names, filename_ids, qid=qid)
        positive_ids = tuple(_resolve_candidate_id(value, candidate_names, filename_ids, qid=qid)
                             for value in positives)
        queries.append(Query(qid, index, query_txt, reference_id, positive_ids, None, memberships[qid]))

    hashes = {name: report.actual for (name, _), report in zip(_path_items(paths), hash_reports)}
    verification = "skipped" if not hash_reports else (
        "matched" if all(report.matches for report in hash_reports) else "mismatch"
    )
    provenance: dict[str, object] = {
        "protocol_id": RELEASE_PROTOCOL_ID,
        "is_synthetic": not expect_released_counts,
        "failed_expectations": tuple(failed_expectations),
        "hash_verification": verification,
        "sha256": hashes,
        "hash_reports": tuple(hash_reports),
        "counts": {"candidate": len(candidate_rows), "gallery": len(gallery_ids),
                   "full": len(full_rows), "sf": len(sf_rows), "validated": len(validated_ids),
                   **{part: len(validated_rows[part]) for part in VALIDATED_PARTS}},
        "gallery_note": {
            "paper_stated_images": PAPER_STATED_GALLERY_IMAGES,
            "released_images": RELEASED_GALLERY_IMAGES,
            "discrepancy": "paper states 18,853; pinned released audit artifact contains 74,381",
            "identity_normalization": "removed <path_to_M-BEIR>/; retained released order",
        },
        "gallery_candidate_basename_match": True,
        "gallery_candidate_basename_count": len(gallery_basenames),
    }
    return ReleaseData(Gallery(RELEASE_PROTOCOL_ID, gallery_ids), tuple(queries), candidate_names, provenance)


def assign_garment_categories(data: ReleaseData, fashioniq_root: Path) -> ReleaseData:
    """Assign garment category by reference identity; shared images remain unscoped.

    Reads local validation splits and test-caption annotations. Ambiguous references retain
    category=None and are recorded in provenance; absent annotation files leave categories unset.
    Positive/target IDs and caption text never determine garment category.
    """
    categories = ("dress", "shirt", "toptee")
    identities: dict[str, set[str]] = {category: set() for category in categories}
    missing = []
    for category in categories:
        split = fashioniq_root / "image_splits" / f"split.{category}.val.json"
        captions = fashioniq_root / "captions" / f"cap.{category}.test.json"
        if not split.is_file() or not captions.is_file():
            missing.extend(path for path in (split, captions) if not path.is_file())
            continue
        split_values = json.loads(split.read_text(encoding="utf-8"))
        caption_values = json.loads(captions.read_text(encoding="utf-8"))
        if (not isinstance(split_values, list)
                or not all(isinstance(value, str) and value for value in split_values)
                or len({Path(value).stem for value in split_values}) != len(split_values)):
            raise ValueError(f"malformed or duplicate image identities in {split}")
        if not isinstance(caption_values, list):
            raise ValueError(f"malformed caption annotation list in {captions}")
        caption_ids = []
        for row in caption_values:
            if not isinstance(row, dict) or not isinstance(row.get("candidate"), str) or not row["candidate"]:
                raise ValueError(f"malformed candidate identity in {captions}")
            identities[category].add(Path(row["candidate"]).stem)
        identities[category].update(Path(value).stem for value in split_values)

    provenance = dict(data.provenance)
    query_count = len(data.queries)
    if missing:
        provenance["category_resolution"] = {
            "status": "unavailable", "considered_count": query_count,
            "resolved_count": 0, "ambiguous_count": 0, "unresolved_count": query_count,
            "ambiguous_examples": (), "missing_files": tuple(str(path) for path in missing),
        }
        return replace(data, queries=tuple(replace(query, category=None) for query in data.queries),
                       provenance=provenance)

    assigned = {}
    ambiguous = []
    unresolved = []
    for query in data.queries:
        image = Path(data.candidate_names[query.reference_id]).stem
        matches = [category for category in categories if image in identities[category]]
        if len(matches) > 1:
            ambiguous.append({"qid": query.qid, "categories": tuple(matches)})
        elif matches:
            assigned[query.qid] = matches[0]
        else:
            unresolved.append(query.qid)
    resolved_count, ambiguous_count, unresolved_count = len(assigned), len(ambiguous), len(unresolved)
    if resolved_count + ambiguous_count + unresolved_count != query_count:
        raise AssertionError("category resolution counts do not cover every query")
    provenance["category_resolution"] = {
        "status": ("resolved" if resolved_count else "ambiguous" if ambiguous_count else "unresolved"),
        "considered_count": query_count,
        "resolved_count": resolved_count,
        "ambiguous_count": ambiguous_count,
        "unresolved_count": unresolved_count,
        "ambiguous_qids": tuple(row["qid"] for row in ambiguous),
        "ambiguous_examples": tuple(ambiguous[:20]),
        "unresolved_qids": tuple(unresolved[:20]),
        "identity_source": "reference in FashionIQ validation splits or test-caption annotations",
    }
    return replace(data, queries=tuple(replace(query, category=assigned.get(query.qid))
                                       for query in data.queries), provenance=provenance)


def select_queries(data: ReleaseData, subset: str) -> tuple[Query, ...]:
    """Select queries only; gallery identity and order stay unchanged."""
    if subset not in SUBSET_ORDER:
        raise ValueError(f"unknown subset {subset!r}; expected one of {SUBSET_ORDER}")
    selected = tuple(query for query in data.queries if subset in query.subsets)
    if subset in EXPECTED_QUERY_COUNTS and len(selected) != EXPECTED_QUERY_COUNTS[subset]:
        if data.provenance.get("is_synthetic"):
            return selected
        raise ValueError(f"{subset} query count {len(selected)} != expected {EXPECTED_QUERY_COUNTS[subset]}")
    return selected
