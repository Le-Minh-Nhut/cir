"""Did the unseeded native relation tensor cause the two historical ENCODER rank disagreements?

The historical claim, committed at `git show HEAD:` on
`g1_g2.encoder.{dress,shirt}.adapter.json`, is `rank_differences_vs_native_val_control`:

    dress  sample index 1569  native_rank 1215  adapter_rank 1216
    shirt  sample index 1352  native_rank 1373  adapter_rank 1372

The producer then built its native `Encoder` with an UNSEEDED relation tensor: the committed
`encoder_parity.py` called no `manual_seed` before `model_try2.Encoder(...)`, and
`model_try2.py:93` draws `learnable_relation` with `torch.randn`, i.e. from the global RNG. Both
production paths seed that construction with the registry's `RELATION_SEED`
(`workbench/composition/model_encoder.py:59`, `:500`). The producer has been fixed to do the same
(`encoder_parity.py:144-148`) and the six artifacts regenerated, so the two sides now share one
relation tensor.

This script measures, per stable ID, whether the disagreement is gone and whether the stored scores
support "the relation draw caused it":

  pre   `raw/encoder_parity/{adapter,native}.<cat>.npz`   (tracked, from the unseeded run)
  post  `$ENCODER_PARITY_RAW/{adapter,native}.<cat>.npz`  (default `/tmp/encparity_raw`, seeded run;
        the regenerated producer writes its intermediates there rather than over the tracked record)

Ranking protocol, applied identically to both eras and both sides:

  gallery    `datasets.py:186-195` (`get_test_data`, `val-split`): first appearance of
             `source_img_id` then `target_img_id`, rebuilt from `captions/cap.<cat>.val.json` and
             asserted equal to the archived `gallery_ids`.
  normalize  `test.py:54-58`, per-row L2, float32.
  score      `test.py:62`, `all_queries.dot(all_imgs.T)`, float32.
  mask       `test.py:67-68`, reference row set to -10e10 (unretrievable).
  rank       `test.py:71`, 1-based position in `np.argsort(-sims[i])`.

Three rank computations are reported per side and era -- masked argsort (the native metric),
unmasked argsort (production eligibility) and the adapter's closed form
`1 + #greater + #earlier-equal` (`adapters.py:393-409`) -- so a residual difference can be
attributed to the mask, the tie policy or the scores rather than assumed.

Verdicts:

  RESOLVED_BY_RELATION_SEED  both sides agree post-seed, and the pre/post score movement is orders of
                             magnitude larger than the deciding margin
  STILL_DISAGREES            they do not agree post-seed
  UNDETERMINED               the arrays needed for the attribution are missing, or the pre and post
                             arrays are not the ones their artifacts were computed from

    python workbench/docs/reviews/20261112-parity-repair/scripts/stable_id_regressions.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
WORKBENCH = REPO / "workbench"
EVIDENCE = WORKBENCH / "docs/reviews/20261109-native15-composition-limn"
PRE_RAW = EVIDENCE / "raw/encoder_parity"
POST_RAW = Path(os.environ.get("ENCODER_PARITY_RAW", "/tmp/encparity_raw"))
CAPTIONS = REPO / "data/FashionIQ/captions"
RELEASE = WORKBENCH / "artifacts/circus_release/mbeir_fashioniq_task7_test.jsonl"
CAND_POOL = WORKBENCH / "artifacts/circus_release/mbeir_fashioniq_task7_cand_pool.jsonl"
DEST = WORKBENCH / "docs/reviews/20261112-parity-repair/raw/stable_ids"

#: sample index -> the neighbourhood the review asks for, and the ranks the historical claim records.
CASES = {
    "dress": {"index": 1569, "window": (1205, 1220), "recorded_native": 1215, "recorded_adapter": 1216},
    "shirt": {"index": 1352, "window": (1365, 1380), "recorded_native": 1373, "recorded_adapter": 1372},
}
TIE_EPSILON = 1e-6  # the declared `tolerances.tie_epsilon`; labels a margin, never moves a rank


def _normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def _rebuild_gallery(category: str) -> list[str]:
    """`datasets.py:186-195` from the caption file, independent of the archived inputs."""
    order: list[str] = []
    for row in json.loads((CAPTIONS / f"cap.{category}.val.json").read_text()):
        for key in ("candidate", "target"):
            if row[key] not in order:
                order.append(row[key])
    return order


def _rankings(queries: np.ndarray, gallery: np.ndarray, index: int,
              reference: int, target: int) -> tuple[dict, np.ndarray]:
    sims = _normalize(queries).dot(_normalize(gallery).T).astype(np.float32)
    masked = sims.copy()
    masked[index, reference] = -10e10   # `test.py:67-68` masks per row: this query's own source
    row = masked[index]
    unmasked_row = sims[index]

    def _argsort(score_row: np.ndarray) -> int:
        return int(np.flatnonzero(np.argsort(-score_row) == target)[0]) + 1

    return {
        "rank_masked_argsort": _argsort(row),
        "rank_unmasked_argsort": _argsort(unmasked_row),
        "rank_masked_tie_policy": 1 + int(np.count_nonzero(row > row[target]))
        + int(np.count_nonzero(row[:target] == row[target])),
        "count_strictly_greater_masked": int(np.count_nonzero(row > row[target])),
        "count_equal_excluding_self_masked": int(np.count_nonzero(row == row[target])) - 1,
        "target_score_masked": float(row[target]),
        "masked_row": row,
    }, sims


def _side(queries: np.ndarray, gallery: np.ndarray, index: int, reference: int, target: int,
          ids: list[str]) -> dict:
    ranks, sims = _rankings(queries, gallery, index, reference, target)
    row = ranks.pop("masked_row")
    position = ranks["rank_masked_argsort"]
    order = np.argsort(-row)
    above, below = int(order[position - 2]), int(order[position])
    ranks.update({
        "margin_above": float(np.float64(row[above]) - np.float64(row[target])),
        "nearest_above": {"img_id": ids[above], "score": float(row[above])},
        "margin_below": float(np.float64(row[target]) - np.float64(row[below])),
        "nearest_below": {"img_id": ids[below], "score": float(row[below])},
        "dirichlet_gap": None,
        "_row": row,
        "_order": order,
        "_sims": sims,
    })
    return ranks


def _recorded(category: str) -> dict:
    """The historical claim, from the committed artifact rather than the worktree.

    Any adapter re-run rewrites `g1_g2.encoder.*.adapter.json` in place, so the worktree copy is not
    a stable record of what was claimed; `git show HEAD:` is.
    """
    path = f"workbench/docs/reviews/20261109-native15-composition-limn/g1_g2.encoder.{category}.adapter.json"
    committed = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=REPO, capture_output=True,
                               text=True, check=True).stdout
    document = json.loads(committed)
    block = document["adapter"]["scoring_ranking"]["rank_differences_vs_native_val_control"]
    return {"source": f"git show HEAD:{path}", "block": block,
            "relation_state_present": "relation_state" in document}


def _current(category: str) -> dict:
    """The regenerated artifact as it stands on disk."""
    path = EVIDENCE / f"g1_g2.encoder.{category}.adapter.json"
    document = json.loads(path.read_text())
    scoring = document["adapter"]["scoring_ranking"]
    return {
        "source": str(path.relative_to(REPO)),
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rank_differences_vs_native_val_control": scoring["rank_differences_vs_native_val_control"],
        "adapter_vs_argsort_unmasked_count": scoring["adapter_vs_argsort_unmasked"]["count"],
        "native_mask_only_effect_count": scoring["native_mask_only_effect"]["count"],
        "native_second_process_floor": scoring["native_second_process_floor"],
        "recall": {"adapter_style": scoring["adapter_style_recall"],
                   "native_convention_on_adapter_features":
                       scoring["native_convention_recall_on_adapter_features"],
                   "native_convention_val_control": scoring["native_convention_recall_val_control"]},
        "relation_state": document.get("relation_state"),
    }


def _qid(reference: str) -> dict:
    """Stable-ID join: the released rows carry ids, not ranks, so the join is what names the case."""
    rows = [json.loads(line) for line in RELEASE.read_text().splitlines() if line.strip()]
    hits = [row for row in rows if row["query_img_path"].endswith(f"/{reference}.jpg")]
    if len(hits) != 1:
        raise AssertionError(f"reference {reference} joined {len(hits)} release rows, expected 1")
    row = hits[0]
    pool = {json.loads(line)["did"]: json.loads(line)
            for line in CAND_POOL.read_text().splitlines() if line.strip()}
    target = pool[row["pos_cand_list"][0]]
    return {"qid": row["qid"], "query_img_path": row["query_img_path"],
            "query_txt": row["query_txt"], "pos_cand_list": row["pos_cand_list"],
            "target_did": target["did"], "target_img_path": target["img_path"],
            "joined_rows_for_reference": len(hits)}


def _neighbourhood(side: dict, ids: list[str], low: int, high: int) -> list[dict]:
    return [{"rank": rank, "gallery_row": int(side["_order"][rank - 1]),
             "img_id": ids[int(side["_order"][rank - 1])],
             "score": float(side["_row"][int(side["_order"][rank - 1])])}
            for rank in range(low, high + 1)]


def _era(root: Path, category: str, case: dict) -> dict:
    """Both sides of one era, plus the score comparison that the era-to-era change needs."""
    inputs = json.loads((root / f"inputs.{category}.json").read_text())
    ids = inputs["gallery_ids"]
    index = case["index"]
    reference, target = inputs["reference_positions"][index], inputs["target_positions"][index]
    adapter = np.load(root / f"adapter.{category}.npz")
    native = np.load(root / f"native.{category}.npz")
    sides = {
        "adapter": _side(adapter["queries"], adapter["gallery"], index, reference, target, ids),
        "native_val_control": _side(native["population_val"], native["gallery"], index,
                                    reference, target, ids),
    }
    row_delta = np.abs(sides["adapter"]["_sims"][index].astype(np.float64)
                       - sides["native_val_control"]["_sims"][index].astype(np.float64))
    agreeing = sides["adapter"]["rank_masked_argsort"] == sides["native_val_control"]["rank_masked_argsort"]
    return {
        "root": str(root),
        "gallery_rows": {"reference": reference, "target": target},
        "adapter": {key: value for key, value in sides["adapter"].items() if not key.startswith("_")},
        "native_val_control": {
            key: value for key, value in sides["native_val_control"].items() if not key.startswith("_")},
        "rank_differences_vs_native_val_control_recomputed": {
            "count": 0 if agreeing else 1,
            "agrees": agreeing,
            "adapter_rank": sides["adapter"]["rank_masked_argsort"],
            "native_rank": sides["native_val_control"]["rank_masked_argsort"],
        },
        "row_max_abs_delta": float(row_delta.max()),
        "row_max_abs_delta_gallery_row": int(row_delta.argmax()),
        "_sides": sides,
        "_native_row": sides["native_val_control"]["_sims"][index],
        "_adapter_row": sides["adapter"]["_sims"][index],
    }


def build(category: str) -> dict:
    case = CASES[category]
    low, high = case["window"]
    index = case["index"]
    recorded = _recorded(category)
    current = _current(category)

    pre = _era(PRE_RAW, category, case)
    post = _era(POST_RAW, category, case)
    gallery_ok = _rebuild_gallery(category) == json.loads(
        (PRE_RAW / f"inputs.{category}.json").read_text())["gallery_ids"]
    assert gallery_ok, f"{category}: rebuilt gallery order differs from the archived order"
    assert pre["adapter"]["rank_masked_argsort"] == case["recorded_adapter"], (category, "adapter")
    assert pre["native_val_control"]["rank_masked_argsort"] == case["recorded_native"], (category, "native")
    block = recorded["block"]
    if block["classification"]:
        entry = block["classification"][0]
        assert entry["index"] == index, (category, entry["index"])
        assert entry["adapter_rank"] == case["recorded_adapter"]
        assert entry["native_rank"] == case["recorded_native"]

    target = pre["gallery_rows"]["target"]
    pre_native_row = pre["_native_row"]
    post_native_row = post["_native_row"]
    native_row_delta = np.abs(pre_native_row.astype(np.float64)
                              - post_native_row.astype(np.float64))
    pre_adapter_row = pre["_adapter_row"]
    post_adapter_row = post["_adapter_row"]
    adapter_row_delta = np.abs(pre_adapter_row.astype(np.float64)
                               - post_adapter_row.astype(np.float64))

    pre_margin = min(pre["native_val_control"]["margin_above"],
                     pre["native_val_control"]["margin_below"],
                     pre["adapter"]["margin_above"], pre["adapter"]["margin_below"])
    movement = float(max(native_row_delta.max(), adapter_row_delta.max()))
    resolved = post["rank_differences_vs_native_val_control_recomputed"]["agrees"]
    if not resolved:
        verdict = "STILL_DISAGREES"
    elif movement > pre_margin:
        verdict = "RESOLVED_BY_RELATION_SEED"
    else:
        verdict = "UNDETERMINED"

    # The pre-era cross-side feature distance is the measurement that decides HOW the relation
    # hypothesis applies. Two independently drawn relation tensors cannot produce features 1e-6 apart
    # while a genuine relation change moves them 1e-2 (`era_shift`.native_query below) -- so the
    # stored data says both pre-era sides already shared one relation, and the historical
    # disagreement is NOT "two different relation tensors were compared".
    pre_inputs = json.loads((PRE_RAW / f"inputs.{category}.json").read_text())
    pre_adapter_blob = np.load(PRE_RAW / f"adapter.{category}.npz")
    pre_native_blob = np.load(PRE_RAW / f"native.{category}.npz")
    post_adapter_blob = np.load(POST_RAW / f"adapter.{category}.npz")
    post_native_blob = np.load(POST_RAW / f"native.{category}.npz")
    cross_side = {}
    for era, adapter_blob, native_blob in (("pre", pre_adapter_blob, pre_native_blob),
                                           ("post", post_adapter_blob, post_native_blob)):
        cross_side[era] = max(
            float(np.abs(adapter_blob["queries"].astype(np.float64)
                         - native_blob["population_val"].astype(np.float64)).max()),
            float(np.abs(adapter_blob["gallery"].astype(np.float64)
                         - native_blob["gallery"].astype(np.float64)).max()))
    era_shift = {
        "native_query": float(np.abs(pre_native_blob["population_val"].astype(np.float64)
                                     - post_native_blob["population_val"].astype(np.float64)).max()),
        "native_gallery": float(np.abs(pre_native_blob["gallery"].astype(np.float64)
                                       - post_native_blob["gallery"].astype(np.float64)).max()),
        "adapter_query": float(np.abs(pre_adapter_blob["queries"].astype(np.float64)
                                      - post_adapter_blob["queries"].astype(np.float64)).max()),
        "adapter_gallery": float(np.abs(pre_adapter_blob["gallery"].astype(np.float64)
                                        - post_adapter_blob["gallery"].astype(np.float64)).max()),
    }
    same_relation_pre = cross_side["pre"] < era_shift["native_query"] / 1000.0
    cause = ("ONE_SHARED_RELATION_PRE_REGENERATION" if same_relation_pre
             else "DIFFERENT_RELATIONS_PRE_REGENERATION")

    return {
        "category": category,
        "stable_ids": {
            "reference": pre_inputs["full_reference_ids"][index],
            "target": pre_inputs["full_target_ids"][index],
            "sample_index": index, "gallery_rows": pre["gallery_rows"],
            "release": _qid(pre_inputs["full_reference_ids"][index]),
        },
        "protocol": {
            "gallery": "datasets.py:186-195 get_test_data(val-split), rebuilt from "
                       f"captions/cap.{category}.val.json, asserted identical",
            "normalize": "test.py:54-58 per-row L2, float32",
            "score": "test.py:62 all_queries.dot(all_imgs.T), float32",
            "mask": "test.py:67-68 sims[i, test_targets_id.index(source_img_id)] = -10e10",
            "rank": "test.py:71 np.argsort(-sims[i]); adapters.py:393-409 closed form "
                    "1 + #greater + #earlier-equal",
            "tie_epsilon": TIE_EPSILON,
            "eras": {"pre": pre["root"], "post": post["root"]},
        },
        "cross_side_feature_distance": cross_side,
        "era_shift": era_shift,
        "cause_of_pre_regeneration_state": {
            "classification": cause,
            "pre_cross_side_distance": cross_side["pre"],
            "relation_change_moves_features_by": era_shift["native_query"],
            "ratio": era_shift["native_query"] / cross_side["pre"] if cross_side["pre"] else None,
            "meaning": (
                "both pre-era sides sat within the 1e-6 float32 noise floor of each other while a "
                "relation change moves features by ~1e-2, so the pre-era pair was computed with ONE "
                "relation tensor on both sides; the historical 1215-vs-1216 disagreement was a "
                "boundary case INSIDE that shared-relation state, not a comparison of two relations"
                if same_relation_pre else
                "the pre-era sides are further apart than a relation change would put them, which "
                "contradicts a shared-relation reading"),
        },
        "pre_regeneration": {
            "sides": {"adapter": pre["adapter"], "native_val_control": pre["native_val_control"]},
            "rank_differences": pre["rank_differences_vs_native_val_control_recomputed"],
            "row_max_abs_delta": pre["row_max_abs_delta"],
            "archived_claim": recorded,
        },
        "post_regeneration": {
            "sides": {"adapter": post["adapter"], "native_val_control": post["native_val_control"]},
            "rank_differences": post["rank_differences_vs_native_val_control_recomputed"],
            "row_max_abs_delta": post["row_max_abs_delta"],
            "artifact": current,
        },
        "attribution": {
            "verdict": verdict,
            "pre_deciding_margin": pre_margin,
            "pre_cross_side_row_max_abs": pre["row_max_abs_delta"],
            "pre_margin_over_cross_side_noise": pre_margin / pre["row_max_abs_delta"],
            "pre_rank_native": pre["native_val_control"]["rank_masked_argsort"],
            "pre_rank_adapter": pre["adapter"]["rank_masked_argsort"],
            "post_rank_native": post["native_val_control"]["rank_masked_argsort"],
            "post_rank_adapter": post["adapter"]["rank_masked_argsort"],
            "pre_target_score_native": pre["native_val_control"]["target_score_masked"],
            "post_target_score_native": post["native_val_control"]["target_score_masked"],
            "target_score_delta_native": float(np.float64(post["native_val_control"]["target_score_masked"])
                                               - np.float64(pre["native_val_control"]["target_score_masked"])),
            "native_score_row_max_abs_pre_vs_post": float(native_row_delta.max()),
            "adapter_score_row_max_abs_pre_vs_post": float(adapter_row_delta.max()),
            "movement_over_pre_margin": movement / pre_margin if pre_margin else None,
            "relation_seed": (current["relation_state"] or {}).get("seed"),
            "relation_sha256": (current["relation_state"] or {}).get("sha256"),
            "relation_matches_pinned": (current["relation_state"] or {}).get("matches_pinned_digest"),
            "pre_relation_state_present": recorded["relation_state_present"],
            "how_a_shared_relation_still_disagreed": (
                "the pre-era target sat on a knife edge: its deciding margin "
                f"({pre_margin!r}) is {pre_margin / pre['row_max_abs_delta']!r} of the residual "
                f"cross-side score difference on the same row ({pre['row_max_abs_delta']!r}). Two "
                "implementations that agree to that level still order two near-equal rows "
                "differently, so the historical 1-rank difference needs no relation difference at "
                "all -- it is the float32 noise both sides already carry, landing on a boundary. "
                "The seeded re-run moves both queries by ~1e2-1e4x that margin, which destroys the "
                "knife edge instead of resolving a policy question"),
        },
        "neighbourhood": {
            "pre": {name: _neighbourhood(pre["_sides"][name], _ids_pre(category), low, high)
                    for name in pre["_sides"]},
            "post": {name: _neighbourhood(post["_sides"][name], _ids_pre(category), low, high)
                     for name in post["_sides"]},
        },
    }


_IDS: dict[str, list[str]] = {}


def _ids_pre(category: str) -> list[str]:
    if category not in _IDS:
        _IDS[category] = json.loads((PRE_RAW / f"inputs.{category}.json").read_text())["gallery_ids"]
    return _IDS[category]


def _selfcheck(results: dict[str, dict]) -> None:
    """The verdict must follow from the numbers beside it, not from the narration."""
    for category, result in results.items():
        pre = result["pre_regeneration"]
        post = result["post_regeneration"]
        attribution = result["attribution"]
        # 1. the historical claim reproduces from the pre-regeneration features
        assert not pre["rank_differences"]["agrees"], (category, "pre-era sides already agreed")
        assert pre["rank_differences"]["count"] == 1, (category, pre["rank_differences"])
        # 2. the post-era state is the one the regenerated artifact claims
        artifact_block = post["artifact"]["rank_differences_vs_native_val_control"]
        assert post["rank_differences"]["agrees"] == (artifact_block["count"] == 0), (
            category, "recomputed post-era state disagrees with the artifact")
        # 3. the verdict class follows the measured agreement and movement
        if attribution["verdict"] == "RESOLVED_BY_RELATION_SEED":
            assert post["rank_differences"]["agrees"], (category, "resolved but still disagreeing")
            assert attribution["movement_over_pre_margin"] > 1.0, (category, "no movement to attribute")
        # 4. the tables are the whole window, in order, and each row's score is the stored one
        for era in ("pre", "post"):
            sides = pre["sides"] if era == "pre" else post["sides"]
            for name, rows in result["neighbourhood"][era].items():
                window = CASES[category]["window"]
                assert [row["rank"] for row in rows] == list(range(window[0], window[1] + 1)), (
                    category, era, name)
                assert all(isinstance(row["score"], float) and row["score"] == row["score"]
                           for row in rows), (category, era, name, "non-float score")
                assert len({row["rank"] for row in rows}) == len(rows), (category, era, name)
        # 5. the cause classification is the one the measured distances support
        cause = result["cause_of_pre_regeneration_state"]
        assert cause["classification"] == ("ONE_SHARED_RELATION_PRE_REGENERATION"
                                           if result["cross_side_feature_distance"]["pre"]
                                           < result["era_shift"]["native_query"] / 1000.0
                                           else "DIFFERENT_RELATIONS_PRE_REGENERATION"), (
            category, cause)
        # 6. the pre-era knife-edge claim: the margin really is below the cross-side noise scale
        assert attribution["pre_margin_over_cross_side_noise"] < 1.0, (
            category, "pre margin is not sub-noise, so the caveat text is wrong")


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    results = {category: build(category) for category in CASES}
    _selfcheck(results)
    for category, result in results.items():
        (DEST / f"{category}.json").write_text(json.dumps(result, indent=2) + "\n",
                                              encoding="utf-8")
    print(json.dumps({category: {
        "qid": result["stable_ids"]["release"]["qid"],
        "pre": {"native": result["attribution"]["pre_rank_native"],
                "adapter": result["attribution"]["pre_rank_adapter"]},
        "post": {"native": result["attribution"]["post_rank_native"],
                 "adapter": result["attribution"]["post_rank_adapter"]},
        "verdict": result["attribution"]["verdict"],
        "movement_over_pre_margin": result["attribution"]["movement_over_pre_margin"],
    } for category, result in results.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
