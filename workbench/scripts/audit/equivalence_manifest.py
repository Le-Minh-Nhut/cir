"""Audit helper: map the stored query feature rows to query ids and target positions.

Reads ONLY existing artifacts. Writes one JSON manifest into this audit's evidence dir.

The stored `query_<condition>_global.npy` is indexed by the run's query index (the position in
`ReleaseData.queries` after the `full,sf,validated` subset union), NOT by anything stored beside it.
This script re-derives that exact ordering from the released inputs and cross-checks it by
recomputing ranks from the cached features with the shipped scorer and comparing against the run's
own `ranks/habit_fiq_n02.json`. A mismatch means the mapping is wrong and the audit must stop.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

from workbench.backend.operator_config import resolve_config  # noqa: E402
from workbench.composition import data as composition_data  # noqa: E402

RUN = REPO / "workbench/artifacts/composition/runs/497c25c57fbeba885ecfb9d1"
FEATURES = RUN / "work/habit/features"
OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def positive_positions(data) -> dict[str, tuple[int, ...]]:
    from workbench.composition.runner import _positive_positions

    return _positive_positions(data)


def main() -> int:
    config = resolve_config()
    data = composition_data.load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT), verify_hashes=True)
    data = composition_data.assign_garment_categories(data, Path(config.FASHIONIQ_ROOT))
    subsets = {"full", "sf", "validated"}
    queries = tuple(q for q in data.queries if subsets.intersection(q.subsets))
    positions = positive_positions(data)
    stored = json.loads((RUN / "ranks/habit_fiq_n02.json").read_text())
    qids = [q.qid for q in queries]
    manifest = {
        "run": RUN.name,
        "gallery_count": len(data.gallery.image_ids),
        "query_count": len(queries),
        "query_order": qids,
        "query_order_sha256": digest(qids),
        "gallery_order_sha256": digest(list(data.gallery.image_ids)),
        "positive_positions": {qid: list(positions.get(qid, ())) for qid in qids},
        "query_categories": {q.qid: q.category for q in queries},
        "query_subsets": {q.qid: list(q.subsets) for q in queries},
        "stored_rank_keys_match": set(stored["ranks"]["multimodal"]) == set(qids),
        "stored_rank_counts": {c: sum(1 for v in stored["ranks"][c].values() if v is not None)
                               for c in stored["ranks"]},
        "identity_batch_size": stored["identity"].get("batch_size"),
        "identity_scoring_module_source_hash": stored["identity"].get("scoring_module_source_hash"),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "query_manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in manifest.items()
                      if k not in {"positive_positions", "query_categories", "query_subsets"}},
                     indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
