"""A3: what the v2 reader validates vs trusts. Each probe starts from the writer's own chunked
record and changes ONE thing a producer (or a hand-edited artifact) could change."""
import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))
from workbench.backend import parity_gate as gate

SCRIPTS = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/scripts"
spec = importlib.util.spec_from_file_location("producer_t4", SCRIPTS / "t4_parity.py")
t4 = importlib.util.module_from_spec(spec)
sys.modules["producer_t4"] = t4
spec.loader.exec_module(t4)

TOL = {"identity": "exact, zero tolerance", "forward_max_abs": 1e-05, "forward_max_rel": 1e-04,
       "forward_min_cosine": 0.9999999, "batch_max_abs": 1e-04, "batch_max_rel": 1e-03,
       "score_max_abs": 1e-06, "tie_epsilon": 1e-06, "metrics": "exact hit-count equality"}
TOTAL, SCOPE, OFFENDER = 4146, "sample_forward.query", 3600

reference = np.full(TOTAL, 0.5, dtype=np.float32)
actual = reference + np.float32(1e-9)
actual[OFFENDER] = np.float32(0.5 + 1e-2)
clean_actual = reference + np.float32(1e-9)

offender_block = t4.component_block({SCOPE: t4.component_scope(
    SCOPE, "forward_parity", actual, reference,
    atol=TOL["forward_max_abs"], rtol=TOL["forward_max_rel"])})
clean_block = t4.component_block({SCOPE: t4.component_scope(
    SCOPE, "forward_parity", clean_actual, reference,
    atol=TOL["forward_max_abs"], rtol=TOL["forward_max_rel"])})


def native():
    return {"category": "shirt", "phase": "native", "model_id": "testmodel",
            "tolerances": dict(TOL), "standard": gate.STANDARD_ABS_REL_V2,
            "native": {"layer1_identity": {
                "query_count_native": 10, "query_count_annotations": 10,
                "gallery_count_native": 20, "gallery_count_rebuilt_from_annotations": 20,
                "gallery_order_identical": True, "reference_ids_identical": True,
                "target_ids_identical": True,
                "model_text_identical_to_author_composition": True,
                "gallery_has_duplicates": False},
                "native_metrics_match_feature_rerank": True}}


def fwd(v=0.0):
    return {"max_abs": v, "max_rel": v, "min_cosine": 1.0, "bitwise_equal": False,
            "shape": [2, 4], "within_tolerance": v <= TOL["forward_max_abs"]}


def adapter(components):
    return {"category": "shirt", "phase": "adapter", "model_id": "testmodel",
            "tolerances": dict(TOL), "standard": gate.STANDARD_ABS_REL_V2,
            "adapter": {"layer2_pixels": {"adapter_decode_vs_native_val_decode": {
                            "max_abs": 0.0, "max_rel": 0.0, "bitwise_equal": True,
                            "min_cosine": 1.0, "shape": [2, 4]}},
                        "batch_size_1_vs_n": {"max_abs": 3e-07, "max_rel": 1e-06,
                                              "min_cosine": 0.9999999, "bitwise_equal": False,
                                              "shape": [2, 4], "within_tolerance": True},
                        "scoring_ranking": {
                            "score_max_abs_delta_vs_dot": 1.19e-07, "score_pairs_compared": 8,
                            "score_max_abs_declared": 1e-06, "score_within_tolerance": True,
                            "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
                            "rank_differences_vs_native_val_control": {"count": 0,
                                                                       "non_tie_count": 0,
                                                                       "min_gap_abs": 0.0}}},
            "component_parity": components}


def evaluate(components):
    root = Path(tempfile.mkdtemp(prefix="repro-a3-"))
    for name, doc in (("g1_g2.testmodel.shirt.native.json", native()),
                      ("g1_g2.testmodel.shirt.adapter.json", adapter(components))):
        (root / name).write_text(json.dumps(doc), encoding="utf-8")
    v = gate.evaluate_model("testmodel", root=root, categories=("shirt",))
    derived = [c for c in v.checks if c.name.startswith("shirt.components.")]
    return v, (derived[0].measured if derived else None)


def probe(label, mutate):
    components = copy.deepcopy(clean_block)
    mutate(components["scopes"][SCOPE])
    v, measured = evaluate(components)
    print(f"{label:<64} -> {v.status:<32} "
          f"compared/total={measured['components_compared'] if measured else '-'}"
          f"/{measured['components_total'] if measured else '-'} "
          f"cov={measured['coverage'] if measured else '-'}")


# P1: emit only 100 of 4146 components, chunked, no offenders — head absent, nothing refuses it.
probe("P1 chunked, only 100 of 4146 emitted (no head guarantee)",
      lambda r: ([r.__setitem__(k, r[k][:100]) for k in ("indices", "reference", "actual")],
                 r.__setitem__("components_emitted", 100)))

# P2: same, but the emitted indices are 3000..3099 — no head at all, still inside range.
probe("P2 chunked, indices 3000..3099 only (no head, no gap check)",
      lambda r: ([r.__setitem__(k, r[k][-100:]) for k in ("indices", "reference", "actual")],
                 r.__setitem__("components_emitted", 100)))

# P3: a HOLE inside the head: indices 0..255 + 400..511.
def hole(r):
    keep = list(range(0, 256)) + list(range(400, 512))
    r["indices"] = keep
    r["reference"] = [0.5] * len(keep)
    r["actual"] = [0.5] * len(keep)
    r["components_emitted"] = len(keep)
probe("P3 chunked, head has a 144-index hole (no completeness check)", hole)

# P4: recorded worst_normalized > 1 with zero offenders emitted and no offenders list.
def worst_high(r):
    r["worst_normalized"] = 5.0
probe("P4 chunked, worst_normalized=5.0 recorded, mismatch_count=0", worst_high)

# P5: all three digests replaced by arbitrary 64-hex — nothing is recomputed (no arrays held).
def fake_digests(r):
    r["sha256"] = {k: "a" * 64 for k in ("reference", "actual", "delta")}
probe("P5 all three sha256 replaced by 'aaaa…' (digests never verified)", fake_digests)

# P6: shape/order swapped so the chunk claims a different layout, same product.
def fake_shape(r):
    r["shape"] = [6, 691]
probe("P6 shape rewritten to [6,691] (only the product is checked)", fake_shape)

# P7 (negative control): duplicate index IS refused.
def dup(r):
    r["indices"] = r["indices"][:-1] + [r["indices"][1]]
probe("P7 duplicate index (control: refused)", dup)

# P8 (negative control): non-finite IS refused.
def nan(r):
    r["actual"][3] = float("nan")
probe("P8 NaN component (control: refused)", nan)

# P9: coverage "full" with total lying small — product + emitted check forces the lie into shape.
def shrink(r):
    r["components_total"] = 512
    r["shape"] = [512]
probe("P9 components_total+shape shrunk to 512 while 4146 measured", shrink)
