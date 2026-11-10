"""R1 verification: re-run the label migration on a COPY with the cited citation correction.

The original artifact and its historical migration sidecar are never touched. A fresh copy is made
from the pre-migration source, migrated with the verified `protocol_source`, and the result is checked:

* `protocol` is `fashioniq_original_split` (full-split gallery semantics, ref NOT excluded);
* the active `protocol_source` no longer claims a FashionIQ reference mask;
* the false text survives verbatim in `superseded_protocol_source`;
* the `protocol_source_correction` block carries the file:line that disproves it;
* every measured number is byte-identical to the source.

Usage:
  workbench/envs/ctl312h/bin/python .../raw/guards/r1_rerun_migration.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
SOURCE = REPO / "workbench" / "docs" / "reviews" / "20261109-native15-composition-limn" / "raw" / "native" / "clip4cir_fullft.json"
OUT_DIR = REPO / "workbench" / "docs" / "reviews" / "20261110-e2e-repair" / "raw" / "guards"
sys.path.insert(0, str(REPO))

from workbench.scripts.migrate_protocol_label import migrate, sha256

VETTED_SOURCE = (
    "third_party/CLIP4Cir/src/validate.py::compute_fiq_val_metrics — index_features over the whole "
    "split.{category}.val.json gallery (the full released gallery, no category filter, no reference "
    "mask); recall@10/@50 counted directly from the sorted index. The reference-mask branch cited by "
    "the superseded text exists only in compute_cirr_val_metrics (validate.py:162-166), which the "
    "FashionIQ branch never calls")

EVIDENCE = {
    "verdict": "MISLABELED",
    "reason": ("The recorded protocol excludes the reference, but the evaluator branch that ran has "
               "no reference mask at all. The candidate universe is identical either way, so only the "
               "label (and the name a reader files the number under) is wrong."),
    "citations": [
        "workbench/third_party/CLIP4Cir/src/validate.py:21-57 compute_fiq_val_metrics has no "
        "reference write; it normalizes index features (:41), sorts ascending over 1 - cos (:44-45), "
        "labels vs targets only (:49-50)",
        "workbench/third_party/CLIP4Cir/src/validate.py:162-166 reference_mask exists ONLY inside "
        "compute_cirr_val_metrics",
    ],
    "gallery_counts": {"dress": 3817, "shirt": 6346, "toptee": 5373},
    "protocol_source_correction": {
        "protocol_source": VETTED_SOURCE,
        "reason": ("the superseded citation asserts a reference mask that compute_fiq_val_metrics "
                   "does not contain; freezing it would propagate a false statement into the "
                   "migrated artifact"),
        "citations": [
            "workbench/third_party/CLIP4Cir/src/validate.py:21-57 — no reference write in the "
            "FashionIQ branch (normalize :41, distances :44, argsort :45, labels :49-50)",
            "workbench/third_party/CLIP4Cir/src/validate.py:162-166 — the only reference_mask is in "
            "compute_cirr_val_metrics, a different metric function",
        ],
    },
}

MEASURED = ("aggregate", "categories", "checkpoint", "source", "runtime", "side_effects",
            "complete_benchmark", "model_id")


def main() -> int:
    if not SOURCE.is_file():
        raise SystemExit(f"missing migration source: {SOURCE}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    copy = OUT_DIR / "clip4cir_fullft.migrated.r1.json"
    shutil.copyfile(SOURCE, copy)
    source_digest = sha256(SOURCE)
    copy_digest_before = sha256(copy)
    if source_digest != copy_digest_before:
        raise SystemExit("copy differs from source before migrating")

    before = json.loads(copy.read_text(encoding="utf-8"))
    false_citation = before["protocol_source"]

    result = migrate(copy, "fashioniq_original_split", evidence=EVIDENCE, dry_run=False)
    after = json.loads(copy.read_text(encoding="utf-8"))

    checks = {
        "protocol_is_original_split": after["protocol"] == "fashioniq_original_split",
        "from_label_was_full_gallery": before["protocol"] == "fashioniq_full_gallery_ref_excluded",
        "active_source_drops_the_mask_claim": "masking its own index entry" not in after["protocol_source"],
        "active_source_cites_cirr_lines": "162-166" in after["protocol_source"],
        "active_source_states_full_split": "whole split" in after["protocol_source"],
        "false_text_preserved_verbatim": after.get("superseded_protocol_source") == false_citation,
        "correction_block_present": (after.get("protocol_source_correction") or {}).get("field")
        == "protocol_source",
        "correction_cites_the_disproof": any(
            "21-57" in cite and "validate.py" in cite
            for cite in (after.get("protocol_source_correction") or {}).get("citations") or []),
        "measured_fields_identical": all(after.get(name) == before.get(name) for name in MEASURED),
        "sidecar_records_correction": json.loads(
            Path(result["sidecar"]).read_text(encoding="utf-8")).get(
            "protocol_source_corrected") is True,
        "source_artifact_untouched": sha256(SOURCE) == source_digest,
    }

    report = {
        "source_artifact": str(SOURCE),
        "source_sha256": source_digest,
        "source_untouched": sha256(SOURCE) == source_digest,
        "migrated_copy": str(copy),
        "sha256_before": result["sha256_before"],
        "sha256_after": result["sha256_after"],
        "superseded_protocol_source": false_citation,
        "active_protocol_source": after["protocol_source"],
        "protocol_before": before["protocol"],
        "protocol_after": after["protocol"],
        "protocol_source_correction": after.get("protocol_source_correction"),
        "measured_fields_identical": checks["measured_fields_identical"],
        "checks": checks,
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "verdict": "R1_CONFIRMED" if not all(checks.values()) else "R1_FIXED_AND_VERIFIED",
        "note": ("the migration was re-run on a COPY. The historical artifact and its sidecar under "
                 "raw/native/ are not modified; only the copied file was migrated."),
    }
    (OUT_DIR / "r1_migration_rerun.json").write_text(json.dumps(report, indent=1) + "\n",
                                                     encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("sha256_before", "sha256_after", "protocol_before", "protocol_after",
                       "verdict", "failed_checks")}, indent=1))
    for name, ok in checks.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    return 0 if not report["failed_checks"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
