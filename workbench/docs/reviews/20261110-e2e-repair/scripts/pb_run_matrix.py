"""Phase B: RUN_MATRIX.csv — one row per run, one column per surface.

Surfaces are deliberately separate columns because they fail independently: a run can execute and
never be validated, execute and export no candidates, or serve ranks while its cache identity is
unknown. A single 'status' column would hide exactly those differences.
"""
import csv, json
from pathlib import Path

BASE = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261110-e2e-repair")
inv = json.loads((BASE / "raw/run_inventory.json").read_text())
ROOT = Path("/home/heheboiz/data/cir")

COLS = ["row_id", "kind", "model_id", "checkpoint_id", "protocol_id", "subset", "execution_status",
        "execution_evidence", "validation_status", "validation_evidence", "export_status",
        "export_depth_files", "export_evidence", "serving_status", "serving_evidence",
        "cache_identity", "cache_evidence", "rerun_action", "rerun_command", "notes"]

rows = []

def add(**kw):
    r = {c: "" for c in COLS}
    r.update(kw)
    rows.append(r)

# duplicate detection: identical (model, checkpoint, gap, gap_cohort_count, query_count)
groups = {}
for rec in inv["composition"]:
    for subset, s in rec["subsets"].items():
        key = (s["model_id"], s["checkpoint_id"], subset, s.get("gap"), s.get("gap_cohort_count"),
               s.get("query_count"))
        groups.setdefault(key, []).append(rec["run_id"])

for rec in inv["composition"]:
    for subset, s in sorted(rec["subsets"].items()):
        key = (s["model_id"], s["checkpoint_id"], subset, s.get("gap"), s.get("gap_cohort_count"),
               s.get("query_count"))
        twins = [r for r in groups[key] if r != rec["run_id"]]
        export_files = rec["export_files"]
        has_candidates = [f for f in export_files if f.endswith(".jsonl")]
        manifests = [f for f in export_files if f.endswith(".manifest.json")]
        serving = "RANKS_PRESENT" if rec["rank_files"] else "NO_RANKS"
        if rec["run_id"].startswith("released_audit"):
            serving = "IMPORTED_ARTIFACT (no adapter ranks expected)"
        if s["is_synthetic"]:
            execution = "FIXTURE (synthetic summary, not an execution)"
        else:
            execution = s["execution_status"]
        if s["is_synthetic"]:
            validation = "N/A - fixture"
        elif rec["run_id"].startswith("released_audit"):
            validation = "NOT_APPLICABLE (released artifact import)"
        else:
            validation = "NOT_RUN - no verify_run receipt, validation_status absent in every summary"
        if s["is_synthetic"] or rec["run_id"].startswith("released_audit"):
            export_status = "N/A"
        elif has_candidates and subset == "full":
            export_status = "EXPORTED"
        elif has_candidates:
            export_status = "COVERED BY RUN EXPORT (the run's 6003-record export contains this subset's qids)"
        else:
            export_status = "NOT_EXPORTED (no candidates/ file for this run)"
        if twins:
            rerun = ("DO_NOT_RERUN - duplicate of " + ", ".join(sorted(twins)[:3]) +
                     " (identical model/checkpoint/subset/query_count/gap); keep one, mark the other superseded")
        elif s["is_synthetic"]:
            rerun = "NEVER (fixture)"
        elif s["execution_status"] == "FAILED":
            rerun = "RERUN after fixing the declared failure"
        elif s["execution_status"] == "PARTIAL":
            rerun = "RESUME only after the partial cause is named; do not restart from scratch"
        elif not has_candidates:
            rerun = "EXPORT ONLY (re-rank is not needed; add the candidate window)"
        else:
            rerun = "KEEP - do not rerun; use for validation"
        add(
            row_id=f"comp:{rec['run_id']}:{subset}", kind="composition",
            model_id=s["model_id"], checkpoint_id=s["checkpoint_id"], protocol_id=s["protocol_id"],
            subset=subset, execution_status=execution,
            execution_evidence=rec["summary_dir"] + f"/{subset}.json (sha256 in raw/run_inventory.json)",
            validation_status=validation, validation_evidence="none found under workbench/artifacts/receipts",
            export_status=export_status, export_depth_files=str(len(has_candidates)) + " jsonl + " + str(len(manifests)) + " manifest",
            export_evidence=(rec["run_dir"] or "") + "/candidates",
            serving_status=serving, serving_evidence=(rec["run_dir"] or "") + "/ranks",
            cache_identity=s["resume_key"],
            cache_evidence=f"resume_key in summary; gallery_identity_sha256={s['gallery_identity_sha256']}",
            rerun_action=rerun,
            rerun_command=("python -m workbench.scripts.composition_eval --models <model_id> --subset "
                           f"{subset}  (registry-resolved checkpoint; do NOT reuse the old run_id)"),
            notes=("query_count=" + str(s["query_count"]) + " gap_cohort_count=" + str(s["gap_cohort_count"]) +
                   " coverage=" + json.dumps(s["coverage"]) +
                   " reference_eligible=" + str(s["reference_eligible"]) +
                   " measured=" + str(s["reference_eligibility_measured"]) +
                   " hash_verification=" + str(s["hash_verification"]) +
                   " failed_expectations=" + json.dumps(s["failed_expectations"])))

for n in inv["native"]:
    if "unreadable" in n:
        add(row_id="native:" + n["file"], kind="native", execution_status="UNREADABLE_JSON",
            execution_evidence=n["file"], notes=n["unreadable"])
        continue
    agg = n.get("aggregate") or {}
    add(row_id="native:" + n["file"].split("/")[-1].replace(".json", ""), kind="native",
        model_id=n.get("model_id"), protocol_id=n.get("protocol"),
        execution_status=("EXECUTION_COMPLETED" if n.get("exit_code") in (0, None) and agg else "NO_AGGREGATE"),
        execution_evidence=n["file"] + f" (sha256 {n['sha256'][:16]}…)",
        validation_status=("PARITY_VERIFIED" if n.get("all_parity_verified") else
                           "NOT_RUN - native_verified=N in NATIVE_EVAL_MATRIX.csv for every executed row"),
        validation_evidence="workbench/docs/reviews/20261109-native15-composition-limn/NATIVE_EVAL_MATRIX.csv",
        export_status="N/A (aggregate-only upstream evaluator; no per-query artifact)",
        serving_status="N/A", cache_identity="N/A (no workbench cache on the native path)",
        rerun_action=("KEEP unless a protocol/seed defect is proven" if agg else "RERUN when prerequisites exist"),
        rerun_command="see raw/native/<model>.command.sh (native) or workbench/replay/<model>.py (shell)",
        notes=("aggregate=" + json.dumps(agg))[:300])

OUT = BASE / "RUN_MATRIX.csv"
with OUT.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(r)
print("wrote", OUT, "rows", len(rows))
