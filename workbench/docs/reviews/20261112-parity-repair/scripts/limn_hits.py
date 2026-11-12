"""LIMN `metrics_parity`: hit counts derived INDEPENDENTLY on each side from stored data.

The gap this closes: `g1_g2.{category}.native.json` carried no `native_metrics_match_feature_rerank`,
so `parity_gate._metrics_checks` returned `[]`, the gate reported
`missing=['metrics_parity']` and the verdict could never reach PASS on measured evidence.

Neither side's count is read to produce the other:

  native   `limn_native.{category}.jsonl` -> `target_rank`, written by the native aggregate
           (`LIMN/test.py::test`). Certified against the artifact's own `native_metrics_recheck`.
  adapter  `raw/g2_{category}_population.npz` (adapter worker features) scored HERE, with the
           native protocol applied: reference masked to -1e10, `argsort(-sims)`, identical cutoffs.
           `raw/g2_{category}_inputs.json` supplies the gallery order and the reference/target ids.

The mask is NOT a free parameter: the native metric makes the reference unretrievable, so the same
mask is applied on both sides. The reference-eligible production ranking is recorded beside it as
information, never as the comparison.

Writes only additive keys into each `g1_g2.{category}.native.json`; the whole existing key set is
preserved and the gate's own `_metrics_checks` shape is mirrored (one real bool + a detail block).
`t4_parity.py` rewrites those artifacts from scratch; it calls `refresh()` after its phases, so the
block survives a re-run. Nothing here needs a GPU or the adapter worker.

    python limn_hits.py            # recompute and write
    python limn_hits.py --negative # prove the check can fail (writes negative_test.json)
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
EVIDENCE = REPO / "workbench/docs/reviews/20261109-native15-composition-limn"
RAW = EVIDENCE / "raw"
DEST = REPO / "workbench/docs/reviews/20261112-parity-repair/raw/limn_hits"
CATEGORIES = ("dress", "shirt", "toptee")
CUTOFFS = (1, 10, 50)

sys.path.insert(0, str(REPO))
import numpy as np  # noqa: E402


def _hits(ranks: list[int]) -> dict[str, int]:
    return {f"r{k}": sum(1 for rank in ranks if rank <= k) for k in CUTOFFS}


def _order_digest(names: list[str]) -> str:
    """sha256 over an id sequence, newline-joined in ORDER.

    Order is the whole point: two sides that ranked the same set in a different order are not the same
    comparison, so the digest covers position, not just membership. `HINT`'s producer digests its
    sequences the same way (`digest_ids` in `third_model_parity.py`), so the two are comparable.
    """
    return hashlib.sha256("\n".join(str(name) for name in names).encode("utf-8")).hexdigest()


def provenance(category: str) -> dict:
    """The identifiers a hit-count comparison needs, from the stored protocol inputs.

    `parity_gate._metrics_checks` refuses to certify a hit-count comparison without these, because
    equal counts over different queries, galleries or checkpoints measure nothing. LIMN's artifacts
    carried NONE of them, so the gate could not tell whether the two count vectors described the same
    population. Nothing here is remeasured: the query and gallery sequences are the ones the native
    phase recorded in `raw/g2_<category>_inputs.json` and the ranks were computed from, so the digests
    bind the comparison that was actually run rather than a fresh one.

    `protocol`/`checkpoint` are the declaration the native aggregate itself carries
    (`raw/native_metrics.csv`, which records the protocol id and the checkpoint id it scored), and the
    same values the other two producers declare.
    """
    inputs = json.loads((RAW / f"g2_{category}_inputs.json").read_text())
    row = next(line for line in (RAW / "native_metrics.csv").read_text().splitlines()
               if line.startswith("limn,") and f",{category}," in line)
    fields = row.split(",")
    protocol, checkpoint_id = fields[1], fields[3]
    return {"protocol": protocol,
            "checkpoint_id": checkpoint_id,
            "model_id": "limn",
            "query_order_sha256": _order_digest(inputs["full_reference_ids"]),
            "gallery_order_sha256": _order_digest(inputs["gallery_ids"])}


def _pct(hits: dict[str, int], total: int) -> dict[str, float]:
    return {key: 100.0 * value / total for key, value in hits.items()}


def native_side(category: str) -> dict:
    """The native aggregate's own per-query ranks, cross-checked against its recorded recall."""
    rows = [json.loads(line) for line in
            (EVIDENCE / f"limn_native.{category}.jsonl").read_text().splitlines() if line.strip()]
    ranks = [int(row["target_rank"]) for row in rows]
    hits = _hits(ranks)
    recorded = json.loads((EVIDENCE / f"g1_g2.{category}.native.json").read_text())
    recheck = recorded["layer1_identity"]["native_metrics_recheck"]
    # Certification only: if the stored ranks did not reproduce the recorded macro, the stored ranks
    # are not the ones the aggregate used and neither side's count can be trusted.
    reproduces = all(abs(_pct(hits, len(ranks))[key] - recheck[f"{category}_{key}"]) < 1e-9
                     for key in _pct(hits, len(ranks)))
    return {
        "source": f"limn_native.{category}.jsonl:{'target_rank'} (native LIMN/test.py::test)",
        "query_count": len(ranks),
        "hits": hits,
        "recall_pct": _pct(hits, len(ranks)),
        "ranks": ranks,
        "recorded_macro_reproduced_by_stored_ranks": reproduces,
    }


def adapter_side(category: str) -> dict:
    """The adapter's stored population features, ranked under the native protocol."""
    blob = np.load(RAW / f"g2_{category}_population.npz")
    inputs = json.loads((RAW / f"g2_{category}_inputs.json").read_text())
    # float32 end to end: pinned test.py normalizes and dots float32, so a float64 recompute would
    # confound the ranking convention with a dtype difference.
    queries = blob["queries"].astype(np.float32)
    gallery = blob["gallery"].astype(np.float32)
    queries = queries / np.linalg.norm(queries, axis=1, keepdims=True)
    gallery = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    sims = queries.dot(gallery.T)

    position = {name: index for index, name in enumerate(inputs["gallery_ids"])}
    reference_positions = [position[name] for name in inputs["full_reference_ids"]]
    target_positions = [position[name] for name in inputs["full_target_ids"]]
    for row, reference in enumerate(reference_positions):
        sims[row, reference] = -10e10
    ranks = [int(np.flatnonzero(np.argsort(-sims[row]) == target)[0]) + 1
             for row, target in enumerate(target_positions)]
    hits = _hits(ranks)
    return {
        "source": (f"raw/g2_{category}_population.npz queries/gallery (adapter worker features) + "
                   f"raw/g2_{category}_inputs.json"),
        "query_count": len(ranks),
        "gallery_count": len(inputs["gallery_ids"]),
        "hits": hits,
        "recall_pct": _pct(hits, len(ranks)),
        "ranks": ranks,
        "reference_positions": reference_positions,
        "target_positions": target_positions,
    }


def build(category: str) -> dict:
    native = native_side(category)
    adapter = adapter_side(category)
    native_rows = [json.loads(line) for line in
                   (EVIDENCE / f"limn_native.{category}.jsonl").read_text().splitlines() if line.strip()]
    eligible_identical = (
        native["query_count"] == adapter["query_count"]
        and [row["reference_id"] for row in native_rows]
        == [name for name in json.loads((RAW / f"g2_{category}_inputs.json").read_text())["full_reference_ids"]]
        and [row["target_id"] for row in native_rows]
        == [name for name in json.loads((RAW / f"g2_{category}_inputs.json").read_text())["full_target_ids"]])
    equality = {f"r{k}": native["hits"][f"r{k}"] == adapter["hits"][f"r{k}"] for k in CUTOFFS}
    stored_ranking = json.loads(
        (EVIDENCE / f"g1_g2.{category}.adapter.json").read_text())["layer2_4_forward"]["scoring_ranking"]
    return {
        "category": category,
        "declared_rule": "exact integer hit-count equality at every cutoff",
        "cutoffs": list(CUTOFFS),
        "reference_mask": "native test.py:92 convention applied on BOTH sides "
                          "(sims[row, reference_position] = -10e10)",
        "ranking_convention": "native test.py:92 (np.argsort(-sims), 1-based target index)",
        "eligible_queries": native["query_count"],
        "eligible_query_set_identical": bool(eligible_identical),
        "eligible_query_set_evidence": (
            "limn_native.<category>.jsonl reference_id/target_id sequences equal "
            "raw/g2_<category>_inputs.json full_reference_ids/full_target_ids, same length, same order"),
        "provenance": provenance(category),
        "native": {key: native[key] for key in ("source", "query_count", "hits", "recall_pct",
                                                "recorded_macro_reproduced_by_stored_ranks")},
        "adapter": {key: adapter[key] for key in ("source", "query_count", "gallery_count", "hits",
                                                  "recall_pct")},
        "equality": equality,
        "mask_is_native_protocol_not_a_free_parameter": (
            "the reference is unretrievable in the native metric, so the same mask is applied to the "
            "adapter features; the reference-eligible production ranking is reported below as "
            "information and is never the comparison"),
        "informational": {
            "adapter_production_ranking_recall_pct": stored_ranking.get("adapter_style_recall"),
            "adapter_production_ranking_note": (
                "unmasked + tie-counted, i.e. what CIRCUS reports: reference-eligible and tie-safe. "
                "It differs from the native metric by protocol, which the adapter artifact already "
                "quantifies as mask_explained_rank_shift_count="
                f"{stored_ranking.get('mask_explained_rank_shift_count')}. Not a hit-count mismatch."),
            "rank_rows_identical_to_native_aggregate": (
                adapter["ranks"] == native["ranks"]),
            "stored_rank_differences_in_adapter_artifact": stored_ranking.get("rank_differences"),
        },
        "_ranks": {"native": native["ranks"], "adapter": adapter["ranks"]},
    }


def write_artifacts(results: dict[str, dict]) -> list[str]:
    written = []
    for category, result in results.items():
        path = EVIDENCE / f"g1_g2.{category}.native.json"
        artifact = json.loads(path.read_text(encoding="utf-8"))
        before = set(artifact)
        # Per-query ranks stay in the review's raw dir, not in the artifact: a reviewer reading the
        # artifact sees the counts and how to reproduce them; 2017 ranks per side would triple every
        # file for data already on disk.
        detail = {key: value for key, value in result.items()
                  if key not in ("_ranks", "category", "native", "adapter")}
        detail["native"] = {k: v for k, v in result["native"].items()}
        detail["adapter"] = {k: v for k, v in result["adapter"].items()}
        artifact["native_metrics_match_feature_rerank"] = all(result["equality"].values())
        detail.pop("provenance", None)
        artifact["native_metrics_parity_detail"] = detail
        artifact.update(result["provenance"])
        # Same declared tolerance key ENCODER carries, on BOTH sides so neither artifact is the only
        # one declaring the rule this field certifies.
        artifact.setdefault("tolerances", {})["metrics"] = "exact hit-count equality"
        adapter_path = EVIDENCE / f"g1_g2.{category}.adapter.json"
        adapter = json.loads(adapter_path.read_text(encoding="utf-8"))
        adapter.setdefault("tolerances", {})["metrics"] = "exact hit-count equality"
        adapter.update(result["provenance"])
        adapter_path.write_text(json.dumps(adapter, indent=2) + "\n", encoding="utf-8")
        path.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        added = sorted(set(artifact) - before)
        assert not (before - set(artifact)), f"{path.name}: a key was dropped"
        written.append(f"{path.name}: added {added}, kept {len(before)} existing keys")
    return written


def negative() -> dict:
    """Prove the gate's `hit_count_equality` check fails when the recorded claims fail."""
    from workbench.backend.parity_gate import evaluate_model

    outcomes = {}
    for label, mutate in (
        ("flip_one_side_hit_count", lambda doc: (
            doc["native_metrics_parity_detail"]["adapter"]["hits"].__setitem__("r10", 0),
            doc["native_metrics_parity_detail"]["equality"].__setitem__("r10", False),
            doc.__setitem__("native_metrics_match_feature_rerank", False))),
        ("non_boolean_flag", lambda doc: (
            doc.__setitem__("native_metrics_match_feature_rerank", "true"))),
    ):
        tmp = Path(tempfile.mkdtemp(prefix=f"limn-hits-{label}-"))
        for category in CATEGORIES:
            for side in ("native", "adapter"):
                name = f"g1_g2.{category}.{side}.json"
                shutil.copy2(EVIDENCE / name, tmp / name)
        target = tmp / "g1_g2.dress.native.json"
        document = json.loads(target.read_text())
        mutate(document)
        target.write_text(json.dumps(document, indent=2) + "\n")
        verdict = evaluate_model("limn", root=tmp)
        relevant = [(check.name, check.ok, check.measured) for check in verdict.checks
                    if check.dimension == "metrics_parity" and check.name.startswith("dress.")]
        outcomes[label] = {
            "status": verdict.status,
            "detail": verdict.detail,
            "dress_metrics_checks": relevant,
            "root": str(tmp),
        }
        shutil.rmtree(tmp)
    return outcomes


def refresh(categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    """Recompute and rewrite the metrics block for `categories` from stored data only.

    Called by `t4_parity.py` after its adapter phase, because that script rebuilds each artifact
    document from scratch and would otherwise drop these keys.
    """
    DEST.mkdir(parents=True, exist_ok=True)
    results = {category: build(category) for category in categories}
    for category, result in results.items():
        (DEST / f"{category}.json").write_text(
            json.dumps({key: value for key, value in result.items() if key != "_ranks"},
                       indent=2) + "\n", encoding="utf-8")
    return write_artifacts(results)


def _selfcheck(results: dict[str, dict]) -> None:
    """One runnable check: the recorded boolean must be a real bool that matches the counts.

    Catches the two ways this evidence could lie: a bool that disagrees with the hit counts beside
    it, and counts derived from a query set that is not the native one.
    """
    for category, result in results.items():
        equality = result["equality"]
        flag = all(equality.values())
        assert set(equality) == {f"r{k}" for k in CUTOFFS}, (category, equality)
        assert isinstance(flag, bool)
        if flag:
            assert result["native"]["hits"] == result["adapter"]["hits"], (category, "bool true, counts differ")
            assert result["_ranks"]["native"] == result["_ranks"]["adapter"], (
                category, "equal counts but different per-query ranks: inspect before recording")
        assert result["eligible_query_set_identical"], (category, "query sets differ")
        assert result["native"]["recorded_macro_reproduced_by_stored_ranks"], (
            category, "stored native ranks do not reproduce the recorded macro")


def main(argv: list[str]) -> int:
    summary = refresh()
    results = {category: build(category) for category in CATEGORIES}
    _selfcheck(results)
    if "--negative" in argv:
        outcomes = negative()
        (DEST / "negative_test.json").write_text(json.dumps(outcomes, indent=2) + "\n",
                                                 encoding="utf-8")
        for label, outcome in outcomes.items():
            print(label, outcome["status"], outcome["dress_metrics_checks"])
    print(json.dumps({
        "summary": summary,
        "hits": {category: {"native": result["native"]["hits"],
                            "adapter": result["adapter"]["hits"],
                            "equality": result["equality"],
                            "match": all(result["equality"].values())}
                 for category, result in results.items()},
    }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
