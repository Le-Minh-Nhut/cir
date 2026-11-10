"""M-02: recount the "117 historically recovered" queries from raw annotation bytes.

Two independent paths are kept side by side and compared:

* ``resolve_query_category`` / ``caption_scope_index`` (the shipped resolver, used for routing);
* ``raw_pairs`` here, which re-reads ``cap.<category>.val.json`` bytes and rebuilds
  ``(stem, normalized caption text)`` pairs without touching any cached index.

Usage: workbench/envs/limn310-adapted/bin/python scripts/m02/recount_recovered.py
Writes raw/m02/m02_recount.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]          # workbench/docs/reviews/<audit>
REPO = HERE.parents[3]                              # repository root
sys.path.insert(0, str(REPO))

from workbench.composition import data as release_data  # noqa: E402
from workbench.composition.model_limn import (  # noqa: E402
    Adapter,
    caption_scope_index,
    normalize_query_text,
    query_reference_stem,
    resolve_query_category,
)

RELEASE = REPO / "workbench/artifacts/circus_release"
FASHIONIQ = REPO / "data/FashionIQ"
OUT = HERE / "raw/m02/m02_recount.json"
SCOPES = Adapter.category_scopes
CATEGORIES = ("dress", "shirt", "toptee")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cap_raw(category: str) -> Path:
    return FASHIONIQ / "captions" / f"cap.{category}.val.json"


def raw_pairs(category: str) -> set[tuple[str, str]]:
    """``(image stem, normalized joined caption text)`` straight from the file bytes.

    Mirrors ``caption_scope_index``'s join/filter rules (non-empty strings only, ``" and "``)
    but shares no code with it, so a cached or mis-keyed index cannot hide here.
    """
    rows = json.loads(cap_raw(category).read_bytes().decode("utf-8"))
    pairs = set()
    for row in rows if isinstance(rows, list) else ():
        if not isinstance(row, dict) or not isinstance(row.get("candidate"), str):
            continue
        captions = row.get("captions")
        if not isinstance(captions, list):
            continue
        joined = " and ".join(c for c in captions if isinstance(c, str) and c.strip())
        text = " ".join(re.sub(r"[^a-z0-9 ]+", " ", joined.strip().lower()).split())
        if text:
            pairs.add((Path(row["candidate"]).stem, text))
    return pairs


def degraded_routing(ambiguous: list, candidate_names: dict) -> list[dict]:
    """Re-run the shipped resolver on the real 119 with one annotation file removed or corrupted.

    This is what the fixture probe predicts, measured on the actual cohort: a source that cannot
    be read stops narrowing the index, so the surviving category silently wins.
    """
    import shutil
    import tempfile

    results = []
    for category in CATEGORIES:
        for mode in ("absent", "corrupt"):
            with tempfile.TemporaryDirectory(prefix="m02-degrade-") as tmp:
                root = Path(tmp)
                (root / "captions").mkdir()
                for other in CATEGORIES:
                    source = cap_raw(other)
                    target = root / "captions" / source.name
                    if other == category:
                        if mode == "corrupt":
                            target.write_text('{"rows": [', encoding="utf-8")
                        continue
                    shutil.copyfile(source, target)
                index = caption_scope_index(root)
                resolved = {q.qid: resolve_query_category(SCOPES, index, q, candidate_names)
                            for q in ambiguous}
                results.append({
                    "removed": category,
                    "mode": mode,
                    "resolved_count": sum(v is not None for v in resolved.values()),
                    "unresolved": sorted(k for k, v in resolved.items() if v is None),
                    "scope_counts": dict(sorted(Counter(v for v in resolved.values() if v).items())),
                    "7:3038": resolved["7:3038"],
                    "7:3790": resolved["7:3790"],
                })
    return results


def main() -> int:
    raw = {category: raw_pairs(category) for category in CATEGORIES}
    sha = {category: sha256(cap_raw(category)) for category in CATEGORIES}
    loaded = release_data.load_release(RELEASE)
    assigned = release_data.assign_garment_categories(loaded, FASHIONIQ)
    resolution = assigned.provenance["category_resolution"]
    index = caption_scope_index(FASHIONIQ)

    ambiguous = [q for q in assigned.queries if q.category is None]
    report: list[dict] = []
    for query in ambiguous:
        stem = query_reference_stem(query, assigned.candidate_names)
        key = normalize_query_text(query.query_txt)
        routed = resolve_query_category(SCOPES, index, query, assigned.candidate_names)
        hits = [c for c in CATEGORIES if (stem, key) in raw[c]]
        report.append({
            "qid": query.qid,
            "reference_stem": stem,
            "reference_id": query.reference_id,
            "query_txt": query.query_txt,
            "normalized_text": key,
            "loader_category": query.category,
            "index_categories": sorted(index.get((stem, key), ())),
            "raw_file_categories": hits,
            "resolved": routed,
            "raw_rederived_scope": hits[0] if len(hits) == 1 else None,
        })

    resolved = [row for row in report if row["resolved"] is not None]
    unresolved = [row for row in report if row["resolved"] is None]

    def verdict(row: dict) -> str:
        if row["resolved"] is None:
            return "UNRESOLVED_raw_duplicate" if len(row["raw_file_categories"]) > 1 else "UNRESOLVED_raw_absent"
        deciding = row["resolved"]
        if (row["reference_stem"], row["normalized_text"]) not in raw[deciding]:
            return f"FAIL_deciding_file_{deciding}_lacks_pair"
        others = [c for c in CATEGORIES if c != deciding
                  and (row["reference_stem"], row["normalized_text"]) in raw[c]]
        if others:
            return f"FAIL_other_file_also_has_pair:{others}"
        return f"PASS_{deciding}_only"

    for row in report:
        row["rederivation"] = verdict(row)

    checks = {
        "ambiguous_count_is_119": len(ambiguous) == 119,
        "resolved_count_is_117": len(resolved) == 117,
        "unresolved_are_exactly_7:3038_7:3790": sorted(r["qid"] for r in unresolved) == ["7:3038", "7:3790"],
        "resolved_scopes_subset_of_scopes": {r["resolved"] for r in resolved} <= set(SCOPES),
        "resolver_matches_raw_rederivation": all(
            row["resolved"] == row["raw_rederived_scope"] for row in report),
        "index_matches_raw_files": all(
            sorted(row["index_categories"]) == sorted(row["raw_file_categories"]) for row in report),
        "all_rederivations_pass": all(row["rederivation"].startswith(("PASS", "UNRESOLVED"))
                                      for row in report),
    }

    payload = {
        "source": {
            "release_root": str(RELEASE.relative_to(REPO)),
            "release_query_file": str(release_data.resolve_release_paths(RELEASE).mbeir_query.relative_to(REPO)),
            "release_query_sha256": sha256(release_data.resolve_release_paths(RELEASE).mbeir_query),
            "fashioniq_root": str(FASHIONIQ.relative_to(REPO)),
            "caption_file_sha256": sha,
            "loader": "workbench/composition/data.py::load_release + assign_garment_categories",
            "loader_hash_verification": loaded.provenance["hash_verification"],
            "resolver": "workbench/composition/model_limn.py::resolve_query_category + caption_scope_index",
            "provenance_category_resolution": resolution,
        },
        "ambiguous_qid_count": len(ambiguous),
        "resolved_count": len(resolved),
        "unresolved_count": len(unresolved),
        "resolved_scope_counts": dict(sorted(Counter(r["resolved"] for r in resolved).items())),
        "unresolved_qids": [r["qid"] for r in unresolved],
        "unresolved_detail": unresolved,
        "checks": checks,
        "degraded_real_data": degraded_routing(ambiguous, assigned.candidate_names),
        "queries": report,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    summary = {key: payload[key] for key in
               ("ambiguous_qid_count", "resolved_count", "unresolved_count", "resolved_scope_counts",
                "unresolved_qids", "checks")}
    print(json.dumps(summary, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
