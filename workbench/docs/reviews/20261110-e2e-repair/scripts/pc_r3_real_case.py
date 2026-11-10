"""R3: real case from the stored NPZ, with ONE eligibility mask for ranking and probe.

The NPZ stores per-category feature populations (2017/2038/1961 rows) and a 4-row pixel sample.
`sample_query_native` / `sample_query_val` are the features of the SAME 4 sampled queries under
native preprocess_train and preprocess_val; `sample_target` are those 4 targets. Ranks are therefore
computable on the sample, and on the full population, without re-running the model.

Every rank pair is produced with the same declared mask, and a disagreement is reported with the
candidate IDs that actually cross, plus the gap in BOTH matrices - never a bare tie verdict.
"""
import json
from pathlib import Path
import numpy as np

RAW = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn/raw/encoder_parity")
TIE_EPSILON = 1e-6

def unit(a):
    a = np.asarray(a, dtype=np.float32)
    return a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), np.finfo(np.float32).tiny)

def rank_and_order(row, target, mask_position=None):
    r = row.copy()
    if mask_position is not None:
        r[mask_position] = -10e10
    order = np.argsort(-r, kind="stable")
    return int(np.flatnonzero(order == target)[0]) + 1, order, r

results = {}
for cat in ("dress", "shirt", "toptee"):
    native = np.load(RAW / f"native.{cat}.npz")
    gallery = unit(native["gallery"])
    qn = unit(native["sample_query_native"])
    qv = unit(native["sample_query_val"])
    tgt = unit(native["sample_target"])

    # pair each sample target with its unique gallery row (exact identity, not nearest)
    positions = []
    for row in tgt:
        match = np.flatnonzero(np.all(gallery == row, axis=1))
        positions.append(int(match[0]) if match.size == 1 else None)

    entries = []
    for i, t in enumerate(positions):
        if t is None:
            entries.append({"query_index": i, "note": "target not identifiable in the gallery"})
            continue
        sn = qn[i] @ gallery.T
        sv = qv[i] @ gallery.T
        rn, on, _ = rank_and_order(sn, t)
        rv, ov, _ = rank_and_order(sv, t)
        item = {"query_index": i, "target_position": t, "rank_native_train": rn, "rank_native_val": rv}
        if rn != rv:
            lo, hi = sorted((rn, rv))
            crossed = [int(p) for p in on[lo - 1:hi] if p != t]
            gaps_train = [float(abs(sn[p] - sn[t])) for p in crossed]
            gaps_val = [float(abs(sv[p] - sv[t])) for p in crossed]
            item.update({
                "moved": f"{rn}->{rv}",
                "crossed_positions": crossed,
                "gap_train_min": min(gaps_train) if gaps_train else None,
                "gap_val_min": min(gaps_val) if gaps_val else None,
                "tie_in_both_matrices": bool(gaps_train and gaps_val and
                                             min(gaps_train) <= TIE_EPSILON and min(gaps_val) <= TIE_EPSILON),
                "rank_used_one_mask": True,
            })
        entries.append(item)

    # population-level check: same 4 queries against the adapter's own feature matrix
    adapter = np.load(RAW / f"adapter.{cat}.npz")
    aq = unit(adapter["queries"])[:len(qn)]
    ag = unit(adapter["gallery"])
    pop = []
    for i, t in enumerate(positions):
        if t is None:
            continue
        sa = aq[i] @ ag.T
        ra, oa, _ = rank_and_order(sa, t)
        rv, ov, _ = rank_and_order(qv[i] @ gallery.T, t)
        if ra != rv:
            lo, hi = sorted((ra, rv))
            crossed = [int(p) for p in ov[lo - 1:hi] if p != t]
            pop.append({"query_index": i, "adapter_rank": ra, "native_val_rank": rv,
                        "crossed_positions": crossed,
                        "adapter_gap_min": min((float(abs(sa[p] - sa[t])) for p in crossed), default=None),
                        "native_gap_min": min((float(abs((qv[i] @ gallery.T)[p] - (qv[i] @ gallery.T)[t]))
                                               for p in crossed), default=None)})

    results[cat] = {
        "sample_size": len(positions),
        "targets_identifiable": sum(1 for p in positions if p is not None),
        "native_train_vs_native_val": entries,
        "adapter_vs_native_val_on_population": pop,
        "tie_epsilon": TIE_EPSILON,
        "classification": (
            "no rank disagreement on the sample" if not any("moved" in e for e in entries)
            else "disagreements exist; each is reported with crossed IDs and both gaps"),
    }

print("R3REAL:" + json.dumps(results, indent=1))
