"""Phase A: recompute protocol cohort identities from the files the evaluators read.

Writes raw/protocol_cohorts.json. No GPU, no model load: reads the FashionIQ manifests and the
CIRCUS release, and hashes the exact ordered identities each protocol ranks against.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from datasets.fashioniq import build_pair_union_gallery, load_fashioniq_annotations
from evaluation.fashioniq import FASHIONIQ_PROTOCOLS, build_fashioniq_gallery

OUT = ROOT / "workbench/docs/reviews/20261110-e2e-repair"
FI_ROOT = ROOT / "data/FashionIQ"
CATEGORIES = ("dress", "shirt", "toptee")


def sha256_ids(names) -> str:
    h = hashlib.sha256()
    for name in names:
        h.update(str(name).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def main() -> int:
    report = {"source": "recomputed from data/FashionIQ + artifacts/circus_release",
              "protocols": {}, "fashioniq_root": str(FI_ROOT)}
    for name, protocol in FASHIONIQ_PROTOCOLS.items():
        per_cat = {}
        for category in CATEGORIES:
            annotations = load_fashioniq_annotations(FI_ROOT / "captions", "val", category)
            gallery = build_fashioniq_gallery(name, FI_ROOT / "image_splits", category, annotations, "val")
            split = json.loads((FI_ROOT / "image_splits" / f"split.{category}.val.json").read_text())
            targets = [a.target_id for a in annotations]
            refs = [a.reference_id for a in annotations]
            per_cat[category] = {
                "queries": len(annotations),
                "gallery": len(gallery),
                "gallery_sha256_ordered": sha256_ids(gallery),
                "gallery_duplicates": len(gallery) - len(set(gallery)),
                "split_images": len(split),
                "split_sha256": hashlib.sha256((FI_ROOT / "image_splits" / f"split.{category}.val.json").read_bytes()).hexdigest(),
                "query_ids_sha256": sha256_ids(a.index for a in annotations),
                "reference_ids_sha256": sha256_ids(refs),
                "target_ids_sha256": sha256_ids(targets),
                "targets_in_gallery": sum(1 for t in targets if t in set(gallery)),
                "references_in_gallery": sum(1 for r in refs if r in set(gallery)),
                "targets_inside_reference_set": sum(1 for t, r in zip(targets, refs) if t == r),
                "query_identity_sha256": sha256_ids(f"{c}:{a.index}:{a.reference_id}:{a.target_id}"
                                                    for a in annotations for c in (category,)),
            }
        report["protocols"][name] = {
            "gallery_kind": protocol.gallery,
            "reference_excluded": protocol.reference_excluded,
            "aliases": list(protocol.aliases),
            "categories": per_cat,
        }

    release = ROOT / "workbench/artifacts/circus_release"
    audit_path = release / "shortcut_audit/retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    gallery_paths = audit["gallery_paths"]
    prefix = "<path_to_M-BEIR>/"
    normalized = [p[len(prefix):] if p.startswith(prefix) else p for p in gallery_paths]
    cand = release / "mbeir_fashioniq_task7_cand_pool.jsonl"
    cand_rows = [json.loads(line) for line in cand.read_text(encoding="utf-8").splitlines() if line.strip()]
    queries = release / "mbeir_fashioniq_task7_test.jsonl"
    query_rows = [json.loads(line) for line in queries.read_text(encoding="utf-8").splitlines() if line.strip()]
    report["circus_release"] = {
        "gallery": len(normalized),
        "gallery_sha256_ordered": sha256_ids(normalized),
        "gallery_duplicates": len(normalized) - len(set(normalized)),
        "candidates": len(cand_rows),
        "queries": len(query_rows),
        "query_ids_sha256": sha256_ids(row["qid"] for row in query_rows),
        "query_has_multi_positive": sum(1 for row in query_rows if len(
            [row["pos_cand_list"]] if isinstance(row["pos_cand_list"], str) else row["pos_cand_list"]) > 1),
        "query_reference_equals_positive": sum(
            1 for row in query_rows
            if Path(str(row["query_img_path"])).stem in {str(p) for p in (
                [row["pos_cand_list"]] if isinstance(row["pos_cand_list"], str) else row["pos_cand_list"])}),
        "gallery_sha256_source_file": str(audit_path.relative_to(ROOT)),
        "file_sha256": {
            "mbeir_query": hashlib.sha256(queries.read_bytes()).hexdigest(),
            "mbeir_candidates": hashlib.sha256(cand.read_bytes()).hexdigest(),
            "aggregated_audit": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        },
    }
    (OUT / "raw" / "protocol_cohorts.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: {c: v["gallery"] for c, v in val["categories"].items()}
                      for k, val in report["protocols"].items()}, indent=1))
    print("circus", report["circus_release"]["gallery"], report["circus_release"]["queries"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
