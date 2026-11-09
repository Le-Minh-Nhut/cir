"""Task 1: provenance for the two genuinely ambiguous CIRCUS task-7 queries.

Reads only released files + the FashionIQ annotation corpus, prints a JSON report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path("/home/heheboiz/data/cir")
RELEASE = ROOT / "workbench/artifacts/circus_release"
FI = ROOT / "data/FashionIQ"
QIDS = ("7:3038", "7:3790")
CATS = ("dress", "shirt", "toptee")


def norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(text).strip().lower()).split())


def main() -> int:
    queries = {}
    with (RELEASE / "mbeir_fashioniq_task7_test.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            if row["qid"] in QIDS:
                queries[row["qid"]] = row
    pool = {}
    with (RELEASE / "mbeir_fashioniq_task7_cand_pool.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            pool[row["did"]] = row

    audit = json.loads((RELEASE / "shortcut_audit" /
                        "retrieval_data_fashioniq_task7_aggregated_best_rank_mm_any.json").read_text())
    audit_ids = [row.get("qid") or row.get("query_id") for row in audit] if isinstance(audit, list) else []
    audit_keys = sorted({k for row in audit[:5] for k in row}) if isinstance(audit, list) and audit else []

    caps = {cat: json.loads((FI / "captions" / f"cap.{cat}.val.json").read_text()) for cat in CATS}
    splits = {cat: json.loads((FI / "image_splits" / f"split.{cat}.val.json").read_text()) for cat in CATS}
    test_splits = {cat: json.loads((FI / "image_splits" / f"split.{cat}.test.json").read_text()) for cat in CATS}

    report = {"audit_schema_keys": audit_keys, "audit_rows": len(audit) if isinstance(audit, list) else None,
              "queries": {}}
    for qid in QIDS:
        q = queries[qid]
        ref_path = q["query_img_path"]
        ref_stem = Path(ref_path).stem
        target_did = q["pos_cand_list"][0]
        target_stem = Path(pool[target_did]["img_path"]).stem
        entry = {
            "release_record": q,
            "query_src_content": json.loads(q["query_src_content"]),
            "reference_stem_from_path": ref_stem,
            "target": {"did": target_did, "img_path": pool[target_did]["img_path"],
                       "stem": target_stem, "src_content": json.loads(pool[target_did]["src_content"])},
            "caption_rows": [], "split_membership": {}, "test_split_membership": {},
            "normalized_text": norm(q["query_txt"]),
            "audit_present": qid in audit_ids,
            "audit_row": next((r for r in audit if (r.get("qid") or r.get("query_id")) == qid), None)
            if isinstance(audit, list) else None,
        }
        for cat in CATS:
            for index, row in enumerate(caps[cat]):
                if Path(row["candidate"]).stem == ref_stem:
                    joined_raw = " and ".join(row["captions"])
                    entry["caption_rows"].append({
                        "category": cat, "annotation_index": index,
                        "candidate": row["candidate"], "target": row["target"],
                        "captions_raw": row["captions"], "captions_joined_raw": joined_raw,
                        "captions_joined_normalized": norm(joined_raw),
                        "matches_release_text": norm(joined_raw) == norm(q["query_txt"]),
                    })
            entry["split_membership"][cat] = ref_stem in splits[cat]
            entry["test_split_membership"][cat] = ref_stem in test_splits[cat]
        report["queries"][qid] = entry
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
