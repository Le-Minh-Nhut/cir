"""Where a CIRCUS subset row came from, derived from per-query artifacts.

R7: the release serves one row per (run, subset) — `full`, `sf`, `validated` — and nothing on the row
said WHICH parent run produced it, which queries the subset contains, or whether the number could be
recomputed. Three consequences, all of them silent:

* a subset could be a different run generation under the same `run_id` (nothing compared them);
* a subset could hold queries the parent does not (nothing compared the memberships);
* a served metric could be un-derivable from the artifacts (nothing recomputed it).

This module answers all three from the per-query files that ARE the measurement
(``composition/results/<protocol>/<model>/<checkpoint>/<run>/<subset>/<condition>.jsonl``), and
attaches the answers to the served rows. It never changes a served number: a recomputation either
agrees (and is reported as agreement) or disagrees (and is reported as disagreement), because the
number in the summary is the artifact and this is a cross-check of it.

Deliberately NOT done here:

* no metric is replaced by a recomputed one — a disagreement is a finding, not a correction;
* `ndcg_full` is not recomputed: the served value is nDCG over the full ranking depth, not @10, and a
  local @10 recomputation would "disagree" with it by construction;
* a subset with no per-query artifact is reported as having no membership evidence, never as empty.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

#: Subset order as the protocol defines it, widest first. A subset's queries must be a slice of
#: `full`: that is the containment the release documents, and it is what is checked here.
SUBSET_ORDER = ("full", "sf", "validated")

_LOCK = threading.Lock()
_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}


def _digest(values: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(values)).encode()).hexdigest()


def _results_root(artifacts_root: Path) -> Path:
    return artifacts_root / "composition" / "results"


def _run_dirs(artifacts_root: Path, run_id: str) -> list[Path]:
    """`results/<protocol>/<model>/<checkpoint>/<run_id>` — four components under the root."""
    root = _results_root(artifacts_root)
    if not root.is_dir():
        return []
    return sorted(path for path in root.glob(f"*/*/*/{run_id}") if path.is_dir())


def _read_subset(path: Path) -> dict[str, Any] | None:
    """qids, scored count and R@10 numerator of one per-query file, or None when absent."""
    if not path.is_file():
        return None
    stamp = (path.stat().st_mtime_ns, path.stat().st_size)
    cache_key = (str(path), "row", f"{stamp[0]}:{stamp[1]}")
    with _LOCK:
        cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached
    qids: list[str] = []
    scored = 0
    hits_at_10 = 0
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        qid = record.get("qid")
        if not isinstance(qid, str):
            continue
        qids.append(qid)
        rank = record.get("rank")
        if record.get("scored") and isinstance(rank, int):
            scored += 1
            if rank <= 10:
                hits_at_10 += 1
    result = {"qids": qids, "scored_count": scored, "hits_at_10": hits_at_10,
              "query_count": len(qids), "bytes": path.stat().st_size}
    with _LOCK:
        _CACHE[cache_key] = result
    return result


def _subset_row(artifacts_root: Path, run_id: str, subset: str, condition: str,
                summary: dict[str, Any] | None) -> dict[str, Any]:
    dirs = _run_dirs(artifacts_root, run_id)
    row: dict[str, Any] = {
        "subset": subset,
        "parent_run_id": run_id,
        "condition": condition,
        "artifact_path": None,
        "membership_sha256": None,
        "query_count": None,
        "scored_count": None,
        "unscored_count": None,
        "evidence": "NO_PER_QUERY_ARTIFACT",
    }
    if not dirs:
        return row
    path = dirs[0] / subset / f"{condition}.jsonl"
    data = _read_subset(path)
    if data is None:
        return row
    metrics = (summary or {}).get("metrics") or {}
    bucket = metrics.get(condition) if isinstance(metrics.get(condition), dict) else {}
    served_recall = bucket.get("recall_at_10")
    served_scored = bucket.get("scored_count")
    recomputed = None
    if data["scored_count"] and isinstance(served_recall, (int, float)):
        recomputed = 100.0 * data["hits_at_10"] / data["scored_count"]
    # `agrees` is tri-state on purpose: None means "not recomputable from this artifact" (a failed
    # run with zero scored queries), which is not the same as a disagreement.
    agrees = None if recomputed is None else abs(recomputed - float(served_recall)) <= 1e-9
    row.update({
        "artifact_path": str(path.relative_to(artifacts_root)),
        "membership_sha256": _digest(data["qids"]),
        "query_count": data["query_count"],
        "scored_count": data["scored_count"],
        "unscored_count": data["query_count"] - data["scored_count"],
        "evidence": "PER_QUERY_ARTIFACT",
        "metric_check": {
            "metric": f"{condition}.recall_at_10",
            "served": served_recall,
            "served_scored_count": served_scored,
            "recomputed": recomputed,
            "agrees": agrees,
            "denominator": "scored queries of this subset's own per-query artifact",
        },
    })
    return row


def subset_provenance(artifacts_root: Path, run_id: str, subset: str,
                      summary: dict[str, Any] | None = None,
                      condition: str = "multimodal") -> dict[str, Any]:
    """The derivation facts of one (run, subset) row, plus its containment against `full`."""
    own = _subset_row(artifacts_root, run_id, subset, condition, summary)
    full = own if subset == "full" else _subset_row(artifacts_root, run_id, "full", condition, None)
    facts: dict[str, Any] = {
        "parent_run_id": run_id,
        "subset": subset,
        "own": own,
        "full": None if subset == "full" else full,
        "containment": None,
        "consistent": None,
        "reasons": [],
    }
    if own["evidence"] != "PER_QUERY_ARTIFACT":
        facts["reasons"].append(
            "no per-query artifact for this subset, so its membership cannot be stated")
        facts["consistent"] = None
        return facts
    if subset != "full":
        facts["full"] = full
        if full["evidence"] != "PER_QUERY_ARTIFACT":
            facts["reasons"].append("the parent (full) membership is not recorded either")
            facts["consistent"] = None
            return facts
        own_qids = set(_read_subset(artifacts_root / own["artifact_path"])["qids"])
        full_qids = set(_read_subset(artifacts_root / full["artifact_path"])["qids"])
        extra = sorted(own_qids - full_qids)
        facts["containment"] = {
            "subset_of_full": not extra,
            "extra_query_count": len(extra),
            "extra_query_examples": extra[:5],
            # A subset cannot hold more unscored queries than its parent: unscored is a per-query
            # fact, and a slice of the parent's queries cannot contain more of them.
            "unscored_exceeds_full": (own["unscored_count"] or 0) > (full["unscored_count"] or 0),
            "query_count_exceeds_full": (own["query_count"] or 0) > (full["query_count"] or 0),
        }
        if not facts["containment"]["subset_of_full"]:
            facts["reasons"].append(
                f"{len(extra)} queries in subset {subset} are not in the parent run's full subset")
        if facts["containment"]["unscored_exceeds_full"]:
            facts["reasons"].append(
                f"subset {subset} has more unscored queries than its parent — the two rows do not "
                "describe one generation")
        facts["consistent"] = (facts["containment"]["subset_of_full"]
                               and not facts["containment"]["unscored_exceeds_full"]
                               and not facts["containment"]["query_count_exceeds_full"])
    else:
        facts["consistent"] = True
    check = own.get("metric_check") or {}
    if check and check.get("agrees") is False:
        facts["reasons"].append(
            f"served {check['metric']} {check['served']} is not what the per-query artifact gives "
            f"({check['recomputed']:.6f} over {own['scored_count']} scored queries)")
        facts["consistent"] = False
    return facts


def decorate_run_rows(artifacts_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach `subset_provenance` to each served composition row, in place.

    One call per row, each of which is cached by file stamp, so a warm process re-reads nothing. A row
    whose artifact is missing gets a named absence rather than a fabricated membership.
    """
    for row in rows:
        run_id = row.get("run_id")
        subset = row.get("subset")
        if not isinstance(run_id, str) or not isinstance(subset, str):
            continue
        row["subset_provenance"] = subset_provenance(artifacts_root, run_id, subset, row)
    return rows
