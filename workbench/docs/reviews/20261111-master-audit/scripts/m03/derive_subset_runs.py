"""Derive LIMN `sf` and `validated` rows from the canonical run's own per-query artifacts.

The subsets are selections of the same 6003-query cohort the `full` run already scored. So the
derivation is: read the parent's `ranks` payload (per-qid exact rank per condition), select the qids
in the declared subset membership, recompute metrics on the selected rows with the SAME metric
functions the runner uses (`workbench.composition.metrics.compose`), and write a derived artifact
that names its parent, the parent's digest, the membership definition and its sha256, the coverage and
the denominator actually used.

No GPU, no re-encoding, no new scoring: every number is the parent's own rank for that qid.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

from workbench.backend.operator_config import resolve_config  # noqa: E402
from workbench.composition.data import (  # noqa: E402
    EXPECTED_QUERY_COUNTS, assign_garment_categories, load_release, select_queries)
from workbench.composition.metrics import compose  # noqa: E402

PARENT = "9e4918a6b5c243412cbc58c9"
MODEL = "limn"
CHECKPOINT = "base_iter0_all_categories"
ARTIFACTS = REPO / "workbench/artifacts/composition"
OUT = REPO / "workbench/docs/reviews/20261111-master-audit/raw/m03"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def membership_digest(qids) -> str:
    return hashlib.sha256("\n".join(sorted(qids)).encode()).hexdigest()


def main() -> int:
    config = resolve_config()
    data = assign_garment_categories(
        load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT)), Path(config.FASHIONIQ_ROOT))
    rank_path = ARTIFACTS / "runs" / PARENT / "ranks" / f"{MODEL}_{CHECKPOINT}.json"
    payload = json.loads(rank_path.read_text(encoding="utf-8"))
    parent_ranks = payload["ranks"]
    parent_summary_path = ARTIFACTS / "summary" / f"{MODEL}_{CHECKPOINT}_{PARENT}" / "full.json"
    parent_summary = json.loads(parent_summary_path.read_text(encoding="utf-8"))

    report = {
        "parent_run_id": PARENT,
        "parent_rank_artifact": str(rank_path.relative_to(REPO)),
        "parent_rank_artifact_sha256": sha256_file(rank_path),
        "parent_summary": str(parent_summary_path.relative_to(REPO)),
        "parent_summary_sha256": sha256_file(parent_summary_path),
        "parent_fingerprint": payload.get("fingerprint"),
        "parent_conditions": sorted(parent_ranks),
        "derivation": ("select the parent's per-qid exact rank for every qid in the declared subset "
                       "membership and recompute metrics with workbench.composition.metrics.compose; "
                       "no query is re-encoded and no score is recomputed"),
        "subsets": {},
        "missing_per_query_artifacts": [],
    }

    for subset in ("full", "sf", "validated"):
        declared = {query.qid for query in select_queries(data, subset)}
        taken = {condition: {qid: rank for qid, rank in parent_ranks.get(condition, {}).items()
                             if qid in declared}
                 for condition in parent_ranks}
        missing = sorted(qid for qid in declared
                         if not any(qid in taken[condition] for condition in taken))
        if missing:
            report["missing_per_query_artifacts"].append(
                {"subset": subset, "qids": missing, "reason": "no rank row in the parent payload"})
        result = compose(taken, query_count=len(declared))
        metrics = asdict(result)
        report["subsets"][subset] = {
            "declared_query_count": EXPECTED_QUERY_COUNTS[subset],
            "membership_count": len(declared),
            "membership_sha256": membership_digest(declared),
            "membership_definition": (f"workbench.composition.data.select_queries(subset={subset!r}) "
                                      "over the pinned CIRCUS release membership columns"),
            "coverage": metrics["multimodal"]["coverage"],
            "denominator": metrics["multimodal"]["scored_count"],
            "metrics": metrics,
            "agrees_with_parent_summary": (
                metrics["multimodal"]["recall_at_10"]
                == ((parent_summary.get("metrics") or {}).get("multimodal") or {}).get("recall_at_10")
                if subset == "full" else None),
        }

    # One derived run id per subset: a function of the PARENT identity, the membership bytes and the
    # metric implementation. Mutating the parent's directory would make two different measurements
    # share an id, which is the failure mode this derivation exists to avoid.
    schema = "circus-derived-subset/v1"
    derived_root = OUT / "derived_runs"
    report["derived_runs"] = {}
    for subset, row in report["subsets"].items():
        if subset == "full":
            continue
        identity = {"parent_run_id": PARENT, "parent_rank_artifact_sha256":
                    report["parent_rank_artifact_sha256"],
                    "subset": subset, "membership_sha256": row["membership_sha256"],
                    "schema": schema}
        derived_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:24]
        target = derived_root / derived_id
        target.mkdir(parents=True, exist_ok=True)
        record = {
            "derived_run_id": derived_id,
            "schema": schema,
            "subset": subset,
            "parent_run_id": PARENT,
            "parent_rank_artifact": report["parent_rank_artifact"],
            "parent_rank_artifact_sha256": report["parent_rank_artifact_sha256"],
            "parent_summary": report["parent_summary"],
            "parent_summary_sha256": report["parent_summary_sha256"],
            "membership_definition": row["membership_definition"],
            "membership_sha256": row["membership_sha256"],
            "membership_count": row["membership_count"],
            "declared_query_count": row["declared_query_count"],
            "coverage": row["coverage"],
            "denominator": row["denominator"],
            "metrics": row["metrics"],
            "derivation": report["derivation"],
            "gpu_used": False,
            "installed_into_served_summaries": False,
            "install_blocker": ("`workbench/artifacts/composition/summary/` is read by the results "
                                "index owned by another worker; this artifact carries everything the "
                                "install needs, and the copy is left to that owner"),
        }
        (target / f"{subset}.json").write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
        report["derived_runs"][subset] = {"derived_run_id": derived_id,
                                          "artifact": str((target / f"{subset}.json").relative_to(REPO))}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "derived_subset_runs.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for subset, row in report["subsets"].items():
        print(f"{subset}: declared {row['declared_query_count']} membership {row['membership_count']} "
              f"scored {row['denominator']} coverage {row['coverage']:.6f} "
              f"mm_r10 {row['metrics']['multimodal']['recall_at_10']:.4f}")
    print("full agrees with parent summary:", report["subsets"]["full"]["agrees_with_parent_summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
