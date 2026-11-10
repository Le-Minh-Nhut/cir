
"""Phase A: CIRCUS release cohort identities: subsets, membership hashes, category resolution."""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.composition.data import (load_release, assign_garment_categories, select_queries,
    CONDITIONS, EXPECTED_QUERY_COUNTS, RELEASED_GALLERY_IMAGES)
from workbench.composition.topk import gallery_digest

OUT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261110-e2e-repair/raw/circus_cohorts.json")
root = Path("/home/heheboiz/data/cir/workbench/artifacts/circus_release")
data = load_release(root)
def h(items):
    d = hashlib.sha256()
    for i in items:
        d.update(str(i).encode()); d.update(b"\n")
    return d.hexdigest()

report = {"gallery": {
    "protocol_id": data.gallery.protocol_id,
    "size": data.gallery.size,
    "sha256_ordered": h(data.gallery.image_ids),
    "topk_gallery_digest": gallery_digest(data.gallery.protocol_id, data.gallery.image_ids, data.candidate_names),
    "duplicates": data.gallery.size - len(set(data.gallery.image_ids)),
}, "queries": {}, "provenance": {
    "hash_verification": data.provenance.get("hash_verification"),
    "sha256": data.provenance.get("sha256"),
    "counts": data.provenance.get("counts"),
    "gallery_note": data.provenance.get("gallery_note"),
    "gallery_candidate_basename_count": data.provenance.get("gallery_candidate_basename_count"),
}}
gallery_set = set(data.gallery.image_ids)
names = data.candidate_names
for subset in ("full", "sf", "validated"):
    qs = select_queries(data, subset)
    report["queries"][subset] = {
        "count": len(qs),
        "qid_sha256": h(q.qid for q in qs),
        "identity_sha256": h(f"{q.qid}|{q.reference_id}|{'/'.join(q.positive_ids)}|{q.query_txt}" for q in qs),
        "multi_positive": sum(1 for q in qs if len(q.positive_ids) > 1),
        "reference_is_positive": sum(1 for q in qs if q.reference_id in q.positive_ids),
        "reference_in_gallery": sum(1 for q in qs if names[q.reference_id] in gallery_set),
        "positives_all_in_gallery": sum(1 for q in qs if {names[p] for p in q.positive_ids} <= gallery_set),
        "all_one_token_query": sum(1 for q in qs if not q.query_txt.strip()),
        "query_txt_sha256": h(q.query_txt for q in qs),
    }
by_cat = assign_garment_categories(data, Path("/home/heheboiz/data/cir/data/FashionIQ"))
report["category_resolution"] = dict(by_cat.provenance.get("category_resolution") or {})
counts = {}
for q in by_cat.queries:
    counts[str(q.category)] = counts.get(str(q.category), 0) + 1
report["category_counts"] = counts
OUT.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=1)[:2000])
