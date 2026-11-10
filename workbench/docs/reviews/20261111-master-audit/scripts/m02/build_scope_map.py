"""M-02: build raw/m02/qid_scope_map.json for the whole 6003-query full cohort.

Every query gets exactly one scope decision and one provenance:
``loader_identity`` (release loader pinned the garment category) or ``caption_annotation``
(loader left it None, the author caption pair names exactly one scope). The 119 ambiguous
queries whose caption pair names two scopes are listed separately, unresolved.

Rows are grouped by (reason, scope) because the provenance is identical within a group.

Usage: workbench/envs/limn310-adapted/bin/python scripts/m02/build_scope_map.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

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
OUT = HERE / "raw/m02/qid_scope_map.json"
SCOPES = Adapter.category_scopes
CATEGORIES = ("dress", "shirt", "toptee")
LOADER_DECIDER = "workbench/composition/data.py::assign_garment_categories"
CAPTION_DECIDER = "workbench/composition/model_limn.py::resolve_query_category+caption_scope_index"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(REPO))


def main() -> int:
    sha = {category: sha256(FASHIONIQ / "captions" / f"cap.{category}.val.json")
           for category in CATEGORIES}
    query_file = release_data.resolve_release_paths(RELEASE).mbeir_query
    loaded = release_data.load_release(RELEASE)
    assigned = release_data.assign_garment_categories(loaded, FASHIONIQ)
    index = caption_scope_index(FASHIONIQ)
    resolution = assigned.provenance["category_resolution"]

    groups: dict[tuple[str, str | None], dict[str, Any]] = {}
    unresolved: list[dict[str, Any]] = []
    scope_by_qid: dict[str, str] = {}
    unresolved_qids: list[str] = []

    for query in assigned.queries:
        routed = resolve_query_category(SCOPES, index, query, assigned.candidate_names)
        if query.category is not None:
            reason, decider = "loader_identity", LOADER_DECIDER
            sources = [rel(FASHIONIQ / "image_splits" / f"split.{query.category}.val.json"),
                       rel(FASHIONIQ / "captions" / f"cap.{query.category}.test.json")]
        else:
            reason = "caption_annotation" if routed is not None else "ambiguous_reference_and_caption"
            decider = CAPTION_DECIDER
            stem = query_reference_stem(query, assigned.candidate_names)
            key = normalize_query_text(query.query_txt)
            holders = sorted(index.get((stem, key), ()))
            sources = [rel(FASHIONIQ / "captions" / f"cap.{category}.val.json")
                       for category in holders]

        if routed is None:
            unresolved_qids.append(query.qid)
            unresolved.append({
                "qid": query.qid,
                "scope": None,
                "reason": reason,
                "source_files": sources,
                "caption_file_sha256": sha,
                "decided_by": decider,
            })
            continue

        scope_by_qid[query.qid] = routed
        group = groups.setdefault((reason, routed), {
            "reason": reason,
            "scope": routed,
            "source_files": sources,
            "caption_file_sha256": sha,
            "decided_by": decider,
            "qids": [],
        })
        group["qids"].append(query.qid)

    records = [groups[key] for key in sorted(groups, key=lambda item: (item[0], item[1]))]
    for record in records:
        record["count"] = len(record["qids"])

    counts = Counter(record["reason"] for record in records for _ in record["qids"])
    payload = {
        "cohort": "full",
        "cohort_size": len(assigned.queries),
        "records": records,
        "unresolved": unresolved,
        "scope_by_qid": dict(sorted(scope_by_qid.items())),
        "counts": {
            "by_reason": dict(sorted(counts.items())),
            "by_reason_scope": {f"{r['reason']}:{r['scope']}": r["count"] for r in records},
            "scored": len(scope_by_qid),
            "unresolved": len(unresolved),
        },
        "sha256": {
            "release_query_file": rel(query_file),
            "release_query_sha256": sha256(query_file),
            "caption_files": {rel(FASHIONIQ / "captions" / f"cap.{c}.val.json"): h
                              for c, h in sha.items()},
            "loader_identity_support_files": {
                rel(path): sha256(path)
                for category in CATEGORIES
                for path in (FASHIONIQ / "image_splits" / f"split.{category}.val.json",
                             FASHIONIQ / "captions" / f"cap.{category}.test.json")
            },
        },
        "loader_provenance": {
            "category_resolution": resolution,
            "hash_verification": assigned.provenance.get("hash_verification"),
            "counts": assigned.provenance.get("counts"),
        },
        "spec": {
            "reason:loader_identity": "release loader assigned query.category from reference identity",
            "reason:caption_annotation": "loader left None; the (reference stem, normalized two-caption text) pair exists under exactly one cap.<category>.val.json",
            "reason:ambiguous_reference_and_caption": "loader left None and the same (stem, text) pair exists under two category files, so no scope is chosen",
        },
    }

    assert sum(counts.values()) + len(unresolved) == len(assigned.queries), "accounting must close"
    assert set(scope_by_qid.values()) <= set(SCOPES)
    assert sorted(unresolved_qids) == ["7:3038", "7:3790"], unresolved_qids

    OUT.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=1) + "\n"
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({"path": rel(OUT), "bytes": len(text.encode()), "counts": payload["counts"],
                      "unresolved_qids": unresolved_qids}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
