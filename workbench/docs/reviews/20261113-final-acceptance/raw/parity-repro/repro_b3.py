"""B3: where 375 vs 376 could come from. Count dress R@1 under three conventions on the SAME
stored adapter features, and under the native masks, so the audit's number is either located or
refuted."""
import json, sys
from pathlib import Path
import numpy as np
REPO = Path("/home/heheboiz/data/cir")
RAW = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/raw"
E = REPO / "workbench/docs/reviews/20261109-native15-composition-limn"
C = "dress"
TIE = 1e-06

blob = np.load(RAW / f"g2_{C}_population.npz")
inputs = json.loads((RAW / f"g2_{C}_inputs.json").read_text())
q = blob["queries"].astype(np.float32); g = blob["gallery"].astype(np.float32)
q = q / np.linalg.norm(q, axis=1, keepdims=True); g = g / np.linalg.norm(g, axis=1, keepdims=True)
sims = q.dot(g.T)
pos = {n: i for i, n in enumerate(inputs["gallery_ids"])}
refs = [pos[n] for n in inputs["full_reference_ids"]]
tgts = [pos[n] for n in inputs["full_target_ids"]]

def ranks(masked):
    s = sims.copy()
    if masked:
        for row, r in enumerate(refs):
            s[row, r] = -10e10
    return np.argsort(-s, axis=1)

def strict(r, tgt):  return sum(1 for row, t in enumerate(tgt) if r[row, 0] == t)
def tiecount(r, s, tgt):
    n = 0
    for row, t in enumerate(tgt):
        top = s[row, r[row, 0]]
        if s[row, t] >= top - TIE:
            n += 1
    return n

rm = ranks(True); ru = ranks(False)
print("dress queries:", len(refs), "gallery:", len(pos))
print("masked,   strict hard-hit R@1      :", strict(rm, tgts))
sm = sims.copy()
for row, r in enumerate(refs): sm[row, r] = -10e10
print("masked,   tie-counted R@1         :", tiecount(rm, sm, tgts))
print("unmasked, strict hard-hit R@1     :", strict(ru, tgts))
print("unmasked, tie-counted R@1         :", tiecount(ru, sims, tgts))
print("artifact recorded native hits     :", json.loads((E/f"g1_g2.{C}.native.json").read_text())["native_metrics_parity_detail"]["native"]["hits"])
print("artifact recorded adapter hits    :", json.loads((E/f"g1_g2.{C}.native.json").read_text())["native_metrics_parity_detail"]["adapter"]["hits"])
