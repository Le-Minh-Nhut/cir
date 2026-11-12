"""Measure what the LIMN diagnostic reader does with the CLVC dual-scope artifact.

Not a fix -- evidence. Four stages, each additive, each printing the exact refusal or acceptance
the current `workbench/backend/limn_diagnostic.py` gives:

  S1  point `EVIDENCE_RELATIVE` at raw/clvc_diag, leave file names as they are
  S2  copy to the LIMN file names, leave `artifact_type` alone
  S3  also rewrite `artifact_type`, leave `canonical_cohort` alone
  S4  also add `canonical_cohort`, leave `conditions` (3 supported conditions) alone
  S5  also relabel the third condition to `image_only_empty_text`, leave checkpoint block alone
  S6  also add a scope-level `sha256` to `checkpoints` (LIMN has one, CLVC does not)

S5/S6 are the two schema deltas that are CLVC-specific: its third condition is a named ablation,
and its checkpoint block records per-member digests instead of one `sha256` per scope.

Run: workbench/envs/ctl312h/bin/python \
       workbench/docs/reviews/20261113-final-acceptance/raw/clvc-dualscope/scripts/probe_clvc_reader.py
"""
from __future__ import annotations

import collections
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[7]
sys.path.insert(0, str(REPO))
SRC = REPO / "workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag"
COND = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")
LIMN_COND = ("multimodal", "text_only_black_image", "image_only_empty_text")
COHORT = {"subset": "full", "query_count": 6003, "query_count_per_subset": {"full": 6003,
          "sf": 4069, "validated": 586}, "scored_count": 6001, "unscored_count": 2,
          "multimodal_recall_at_10": None, "source": "probe"}


def attempt(label: str, root: Path) -> None:
    import workbench.backend.limn_diagnostic as limn

    limn._RECORD_CACHE.clear()
    limn.evidence_root = lambda config=None, root=root: root
    for name, call in (("load_diagnostic", lambda: limn.load_diagnostic()),
                       ("load_candidates", lambda: limn.load_candidates(
                           "7:3038", "shirt", limn.CONDITIONS[2]))):
        try:
            out = call()
            if name == "load_diagnostic":
                codes = dict(collections.Counter(d["code"] for d in out["defects"]))
                print(f"{label} {name}: OK status={out['status']} complete={out['complete']} "
                      f"defects={codes or 0}")
            else:
                print(f"{label} {name}: OK depth_stored={out['depth_stored']} "
                      f"served={len(out['candidates'])} defects={len(out['defects'])}")
        except Exception as error:  # noqa: BLE001 - the refusal text IS the evidence
            print(f"{label} {name}: REFUSED {type(error).__name__}: {error}")


def payload() -> dict:
    return json.loads((SRC / "clvc_dual_scope.json").read_text(encoding="utf-8"))


def stage(root: Path, *, rename: bool, artifact_type: bool, cohort: bool,
          relabel_condition: bool, checkpoint_sha: bool) -> None:
    data = payload()
    if artifact_type:
        data["artifact_type"] = "limn_dual_scope_diagnostic"
    if cohort:
        data["canonical_cohort"] = COHORT
    conditions = LIMN_COND if relabel_condition else COND
    data["conditions"] = list(conditions)
    for record in data["queries"].values():
        for cells in record["scopes"].values():
            if relabel_condition:
                cells[conditions[2]] = cells.pop(COND[2])
                cells[conditions[2]]["condition"] = conditions[2]
    if checkpoint_sha:
        for block in data["checkpoints"].values():
            block["sha256"] = block["local_sha256_recomputed"]
    (root / "limn_dual_scope.json").write_text(json.dumps(data), encoding="utf-8")

    for path in SRC.glob("clvc_dual_scope.*.jsonl"):
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            if relabel_condition and row["condition"] == COND[2]:
                row["condition"] = conditions[2]
            rows.append(json.dumps(row))
        name = path.name.replace("clvc_dual_scope", "limn_dual_scope")
        if rename or relabel_condition:
            name = name.replace(COND[2], conditions[2])
        (root / name).write_text("\n".join(rows) + "\n", encoding="utf-8")


def main() -> int:
    print("# S1: EVIDENCE_RELATIVE pointed at raw/clvc_diag, CLVC file names untouched")
    import workbench.backend.limn_diagnostic as limn
    limn._RECORD_CACHE.clear()
    limn.EVIDENCE_RELATIVE = ("workbench", "docs", "reviews", "20261112-parity-repair",
                              "raw", "clvc_diag")
    outcome = limn.load_diagnostic()
    print(f"S1 load_diagnostic: status={outcome['status']} available={outcome['available']} "
          f"reason={outcome.get('reason')}")

    for label, kwargs in (
        ("S2", dict(rename=True, artifact_type=False, cohort=False,
                    relabel_condition=False, checkpoint_sha=False)),
        ("S3", dict(rename=True, artifact_type=True, cohort=False,
                    relabel_condition=False, checkpoint_sha=False)),
        ("S4", dict(rename=True, artifact_type=True, cohort=True,
                    relabel_condition=False, checkpoint_sha=False)),
        ("S5", dict(rename=True, artifact_type=True, cohort=True,
                    relabel_condition=True, checkpoint_sha=False)),
        ("S6", dict(rename=True, artifact_type=True, cohort=True,
                    relabel_condition=True, checkpoint_sha=True)),
    ):
        root = Path(tempfile.mkdtemp())
        try:
            stage(root, **kwargs)
            attempt(f"\n# {label} {kwargs}", root)
        finally:
            shutil.rmtree(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
