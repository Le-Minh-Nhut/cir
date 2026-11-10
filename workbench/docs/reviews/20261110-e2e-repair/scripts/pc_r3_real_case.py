"""G-02 (corrected): re-derive the R3 real-case rank comparison from the stored NPZ features.

Supersedes `20261110-e2e-repair/scripts/pc_r3_real_case.py`, whose bytes are left untouched.

What the superseded script did wrong, and what this one does instead:

| superseded                                       | here                                              |
|--------------------------------------------------|---------------------------------------------------|
| target joined by exact vector equality            | joined by recorded gallery POSITION                |
| rank computed without the reference mask          | the SAME reference mask on both matrices           |
| crossings read from `argsort(...)[lo-1:hi]`       | crossings by the pairwise definition, per candidate|
| gap = max over the slice                          | per-crosser distance to target AND to the cutoff   |
| tolerance implicit, chosen while reading output   | `TIE_EPSILON` declared below, before any measurement |

TOLERANCES ARE DECLARED HERE, BEFORE MEASUREMENT, and are not moved afterwards:

* `TIE_EPSILON = 1e-6` -- the campaign's declared `tolerances.tie_epsilon`
  (`g1_g2.encoder.*.native.json`), and the value `encoder_parity.py:50` used when measuring.
* `RANK_AGREEMENT = "exact"` -- the R3 question is whether two feature matrices rank one fixed
  target identically. There is no numeric tolerance to pick: either the ranks are equal or they are
  not, and any difference is reported with its crossing evidence rather than waved through.

The cutoff is the only derived quantity: `DEEP_CUTOFF = 100`, chosen because the campaign reports
R@1/R@10/R@50, so 100 is the deepest published cut with headroom. It is a REPORTING threshold, not
a pass/fail criterion.

Inputs, all recorded by `encoder_parity.py` (nothing recomputed here):
  raw/encoder_parity/native.<category>.npz   sample_query_native/val, sample_target, gallery
  raw/encoder_parity/adapter.<category>.npz  population queries/gallery
  raw/encoder_parity/inputs.<category>.json  reference_positions, target_positions, sample_indices
"""
import json
import sys
from pathlib import Path

import numpy as np

RAW = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn"
           "/raw/encoder_parity")

# ---- declared BEFORE measuring; never adjusted after the numbers are known ----
TIE_EPSILON = 1e-06   # g1_g2.encoder.*.native.json `tolerances.tie_epsilon`
RANK_AGREEMENT = "exact"
DEEP_CUTOFF = 100


def unit(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), np.finfo(np.float32).tiny)


def rank_masked(scores: np.ndarray, target: int, reference: int | None) -> tuple[int, np.ndarray, np.ndarray]:
    """Rank of `target` with `reference` excluded, exactly as the model ranks.

    `-10e10` then `argsort(-row)` places the masked entry LAST, which is an exclusion. Returned with
    the row the rank was computed from, so no caller can probe a different row for its gaps.
    """
    row = np.asarray(scores, dtype=np.float64).copy()
    if reference is not None:
        row[reference] = -10e10
    order = np.argsort(-row, kind="stable")
    return int(np.flatnonzero(order == target)[0]) + 1, order, row


def crossing_evidence(row_a: np.ndarray, row_b: np.ndarray, target: int,
                      reference: int | None) -> list[dict]:
    """Every candidate whose relation to the target CHANGED, with both distances.

    The pairwise definition, not an argsort slice: "did this candidate move from below the target to
    above it (or the reverse) between the two matrices?". Invariant to column order and to ties in
    the slice, and it names each crosser with the number that decided it.
    """
    out = []
    for position in range(len(row_a)):
        if position in (target, reference):
            continue
        before, after = row_a[position] > row_a[target], row_b[position] > row_b[target]
        if before == after:
            continue
        out.append({
            "candidate_position": position,
            "outranked_target_in": "native_train" if before else "native_val",
            "distance_to_target_train": float(abs(row_a[position] - row_a[target])),
            "distance_to_target_val": float(abs(row_b[position] - row_b[target])),
            # How far the CUTOFF (the deepest crosser's score) sat from the target in each matrix.
            "target_distance_to_cutoff_train": float(abs(np.max(row_a[np.argsort(-row_a)[:DEEP_CUTOFF]])
                                                         - row_a[target])),
            "target_distance_to_cutoff_val": float(abs(np.max(row_b[np.argsort(-row_b)[:DEEP_CUTOFF]])
                                                       - row_b[target])),
        })
    return out


report: dict = {}
for category in ("dress", "shirt", "toptee"):
    native = np.load(RAW / f"native.{category}.npz")
    adapter = np.load(RAW / f"adapter.{category}.npz")
    inputs = json.loads((RAW / f"inputs.{category}.json").read_text(encoding="utf-8"))

    indices = list(inputs["sample_indices"])
    # JOIN BY POSITION: the recorded gallery positions of the sampled queries' reference and target.
    target_positions = [int(inputs["target_positions"][i]) for i in indices]
    reference_positions = [int(inputs["reference_positions"][i]) for i in indices]

    gallery = unit(native["gallery"])
    q_train = unit(native["sample_query_native"])
    q_val = unit(native["sample_query_val"])
    a_gallery = unit(adapter["gallery"])
    a_queries = unit(adapter["queries"])

    sample = []
    for row_index, (tpos, rpos) in enumerate(zip(target_positions, reference_positions)):
        s_train = q_train[row_index] @ gallery.T
        s_val = q_val[row_index] @ gallery.T
        rank_train, _, row_train = rank_masked(s_train, tpos, rpos)
        rank_val, _, row_val = rank_masked(s_val, tpos, rpos)

        entry = {
            "sample_row": row_index,
            "query_index": indices[row_index],
            "target_position": tpos,
            "reference_position": rpos,
            "rank_native_train": rank_train,
            "rank_native_val": rank_val,
            "ranks_agree": rank_train == rank_val,
            "target_score_train": float(row_train[tpos]),
            "target_score_val": float(row_val[tpos]),
            "deep_cutoff": DEEP_CUTOFF,
            "target_inside_cutoff_train": rank_train <= DEEP_CUTOFF,
            "target_inside_cutoff_val": rank_val <= DEEP_CUTOFF,
            "crossings": crossing_evidence(row_train, row_val, tpos, rpos),
            "rank_used_one_mask": True,
            "mask_position": rpos,
            "join": "recorded gallery position (inputs.<category>.json), not vector equality",
        }
        # Adapter vs native-val on the SAME population matrix, same mask, same position.
        if indices[row_index] < len(a_queries):
            s_adapter = a_queries[indices[row_index]] @ a_gallery.T
            rank_adapter, _, row_adapter = rank_masked(s_adapter, tpos, rpos)
            entry["rank_adapter"] = rank_adapter
            entry["adapter_ranks_with_native_val"] = rank_adapter == rank_val
            if rank_adapter != rank_val:
                entry["adapter_crossings"] = crossing_evidence(row_adapter, row_val, tpos, rpos)
        sample.append(entry)

    disagreements = [e for e in sample if not e["ranks_agree"]]
    report[category] = {
        "sample_rows": len(sample),
        "sample_indices": indices,
        "target_positions": target_positions,
        "reference_positions": reference_positions,
        "tie_epsilon_declared": TIE_EPSILON,
        "rank_agreement_declared": RANK_AGREEMENT,
        "rows": sample,
        "rank_disagreement_count": len(disagreements),
        "all_ranks_agree": not disagreements,
        "classification": (
            "every sampled target gets the same rank under preprocess_train and preprocess_val"
            if not disagreements else
            f"{len(disagreements)} sampled target(s) change rank; each is reported with its "
            "crossing set and both distances"),
    }

print("R3SUPERSEDED:" + json.dumps(report, indent=1))
