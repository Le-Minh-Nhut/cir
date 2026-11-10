"""Phase B inventory: one record per execution artifact, per surface, no interpretation.

Collects, for every composition run directory and every native evidence file:
execution status, validation evidence present (verify_run receipt / validation_status),
export presence and depth (candidates/), serving availability (rank files loadable),
cache identity (resume_key/fingerprint), and the rerun command the artifact itself names.
"""
import json, os, glob, hashlib
from pathlib import Path

ROOT = Path("/home/heheboiz/data/cir")
OUT = ROOT / "workbench/docs/reviews/20261110-e2e-repair/raw/run_inventory.json"
comp = ROOT / "workbench/artifacts/composition"

def sha(p):
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            d.update(c)
    return d.hexdigest()

records = []
for summary_dir in sorted((comp / "summary").iterdir()):
    if not summary_dir.is_dir():
        continue
    run_id = summary_dir.name
    subsets = {}
    for path in sorted(summary_dir.glob("*.json")):
        d = json.loads(path.read_text())
        subsets[path.stem] = {
            "execution_status": d.get("execution_status"),
            "reproduction_status": d.get("reproduction_status"),
            "is_synthetic": d.get("is_synthetic"),
            "query_count": d.get("query_count"),
            "gallery_count": d.get("gallery_count"),
            "gallery_identity_sha256": d.get("gallery_identity_sha256"),
            "hash_verification": d.get("hash_verification"),
            "failed_expectations": d.get("failed_expectations"),
            "reference_eligible": d.get("reference_eligible"),
            "reference_eligibility_measured": (d.get("reference_eligibility") or {}).get("measured")
                if isinstance(d.get("reference_eligibility"), dict) else None,
            "adapter_module_source_status": d.get("adapter_module_source_status"),
            "condition_status": {k: v.get("status") for k, v in (d.get("condition_status") or {}).items()},
            "unscored": {k: v.get("unscored_count") for k, v in (d.get("condition_status") or {}).items()},
            "gap_cohort_count": (d.get("metrics") or {}).get("gap_cohort_count"),
            "coverage": {k: (d.get("metrics", {}).get(k) or {}).get("coverage")
                         for k in ("multimodal", "text_only", "image_only")},
            "metrics_r10": (d.get("metrics") or {}).get("recall_at_10_multimodal"),
            "gap": (d.get("metrics") or {}).get("composition_gap_ndcg"),
            "resume_key": d.get("resume_key"),
            "feature_storage": d.get("feature_storage"),
            "model_id": d.get("model_id"), "checkpoint_id": d.get("checkpoint_id"),
            "protocol_id": d.get("protocol_id"),
        }
    # the run directory holding ranks/candidates: summary hashes are the same id
    digest = run_id.rsplit("_", 1)[-1] if run_id.count("_") else run_id
    run_dir = comp / "runs" / digest
    cand_dir = run_dir / "candidates"
    ranks_dir = run_dir / "ranks"
    records.append({
        "run_id": run_id, "kind": "composition",
        "summary_dir": str(summary_dir.relative_to(ROOT)),
        "run_dir": str(run_dir.relative_to(ROOT)) if run_dir.is_dir() else None,
        "subsets": subsets,
        "export_files": sorted(str(p.relative_to(cand_dir)) for p in cand_dir.rglob("*") if p.is_file()) if cand_dir.is_dir() else [],
        "rank_files": sorted(p.name for p in ranks_dir.glob("*")) if ranks_dir.is_dir() else [],
        "summary_digest": {p.name: sha(p) for p in sorted(summary_dir.glob("*.json"))},
    })

native = []
for path in sorted((ROOT / "workbench/docs/reviews/20261109-native15-composition-limn/raw/native").glob("*.json")):
    try:
        d = json.loads(path.read_text())
    except Exception as e:
        native.append({"file": str(path.relative_to(ROOT)), "unreadable": type(e).__name__})
        continue
    native.append({
        "file": str(path.relative_to(ROOT)),
        "model_id": d.get("model_id"),
        "protocol": d.get("protocol") or d.get("protocol_id"),
        "complete_benchmark": d.get("complete_benchmark"),
        "aggregate": d.get("aggregate"),
        "exit_code": (d.get("runtime") or {}).get("exit_code"),
        "sha256": sha(path),
    })
for extra in ("limn_native.json",):
    p = ROOT / "workbench/docs/reviews/20261109-native15-composition-limn" / extra
    if p.is_file():
        d = json.loads(p.read_text())
        native.append({"file": extra, "model_id": "limn", "protocol": d.get("protocol"),
                       "aggregate": d.get("aggregate"), "all_parity_verified": d.get("all_parity_verified"),
                       "sha256": sha(p)})

OUT.write_text(json.dumps({"composition": records, "native": native,
                           "generated_from": str(ROOT)}, indent=2))
print(f"composition={len(records)} native={len(native)}")
