"""R3: reproduce the harness's tie/mask logic on the campaign's toy case.

The claim under test: a masked-rank comparison can move a target's rank and change the reported
'candidate gap' purely because ranking uses one eligibility mask and the tie probe another.
"""
import hashlib, json
import numpy as np

TIE_EPSILON = 1e-6   # the harness's declared tolerance (encoder_parity.py:50) - NOT moved here

def build_scores():
    # reference 1.0, another image 0.8, target 0.5, candidate 0.5000001 (the campaign's toy case)
    return np.array([[1.0, 0.8, 0.5, 0.5000001]], dtype=np.float32)

def ranks(sims, target, mask_reference=None):
    row = sims[0].copy()
    if mask_reference is not None:
        row[mask_reference] = -10e10
    order = np.argsort(-row, kind="stable")
    return int(np.flatnonzero(order == target)[0]) + 1, order

def probe_gap(row, target, lo, hi):
    window = np.argsort(-row)[lo - 1:hi]
    return float(np.abs(row[window] - row[target]).max()) if len(window) else 0.0

sims = build_scores()
TARGET, REFERENCE, CANDIDATE = 2, 0, 3
unmasked_rank, _ = ranks(sims, TARGET)
masked_rank, order = ranks(sims, TARGET, mask_reference=REFERENCE)
row_unmasked = sims[0].copy()
row_masked = sims[0].copy(); row_masked[REFERENCE] = -10e10

lo, hi = sorted((unmasked_rank, masked_rank))
gap_unmasked_probe = probe_gap(row_unmasked, TARGET, lo, hi)
gap_masked_probe = probe_gap(row_masked, TARGET, lo, hi)

# Which candidate actually crossed the boundary, and by how much?
crossed = [int(i) for i in order[lo - 1:hi] if i != TARGET]
deltas = {int(i): float(abs(row_unmasked[i] - row_unmasked[TARGET])) for i in (CANDIDATE,)}

# same case with the 0.8 image removed: the operator's 3-element construction
sims3 = sims[:, [0, 2, 3]]
t3, ref3 = 1, 0
row3u = sims3[0].copy(); row3m = sims3[0].copy(); row3m[ref3] = -10e10
o3u = np.argsort(-row3u, kind="stable"); o3m = np.argsort(-row3m, kind="stable")
rank3_unmasked = int(np.flatnonzero(o3u == t3)[0]) + 1
rank3_masked = int(np.flatnonzero(o3m == t3)[0]) + 1

print("R3JSON:" + json.dumps({
    "scores": [float(v) for v in sims[0]],
    "rank_unmasked": unmasked_rank,
    "rank_masked": masked_rank,
    "rank_move": f"{unmasked_rank}->{masked_rank}",
    "probe_gap_unmasked_matrix": gap_unmasked_probe,
    "probe_gap_masked_matrix": gap_masked_probe,
    "probe_gap_change": gap_masked_probe - gap_unmasked_probe,
    "crossed_positions": crossed,
    "target_vs_candidate_delta": deltas,
    "all_gaps_below_tie_epsilon": gap_masked_probe <= TIE_EPSILON and gap_unmasked_probe <= TIE_EPSILON,
    "tie_epsilon": TIE_EPSILON,
    "three_element": {"scores": [float(v) for v in sims3[0]],
                       "rank_unmasked": rank3_unmasked, "rank_masked": rank3_masked,
                       "rank_move": f"{rank3_unmasked}->{rank3_masked}",
                       "target_vs_candidate_delta": float(abs(sims3[0][2] - sims3[0][1]))},
    "verdict_supported": ("the two ranks and the two probe gaps come from DIFFERENT eligibility masks; "
                          "the reported 'gap change' is an artifact of that, not a score change"),
}, indent=1))
