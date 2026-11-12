"""Repro A: a chunked v2 record whose only offender is outside the 512 head, then the same
record with that offender's rows removed from the emitted preview. Writer = the repo's own."""
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
TOTAL = 4146
SCOPE = "sample_forward.query"

reference = np.full(TOTAL, 0.5, dtype=np.float32)
actual = reference + np.float32(1e-9)
OFFENDER = 3600                              # outside the first 512, ascending
actual[OFFENDER] = np.float32(0.5 + 1e-2)    # 1e-2 > 1e-5 + 1e-4*0.5 = 6e-5 -> violation

block = t4.component_block({SCOPE: t4.component_scope(
    SCOPE, "forward_parity", actual, reference,
    atol=TOL["forward_max_abs"], rtol=TOL["forward_max_rel"])})
rec = block["scopes"][SCOPE]
print("WRITER scope:", json.dumps({
    "components_total": rec["components_total"], "components_emitted": rec["components_emitted"],
    "coverage": rec["coverage"], "mismatch_count": rec["mismatch_count"],
    "max_abs": rec["max_abs"], "worst_normalized": rec["worst_normalized"],
    "offenders": [o["index"] for o in rec["offenders"]],
    "head_last_index": rec["indices"][511], "offender_emitted": OFFENDER in rec["indices"]}))


def native():
    return {"category": "shirt", "phase": "native", "model_id": "testmodel",
            "tolerances": dict(TOL), "standard": gate.STANDARD_ABS_REL_V2,
            "native": {
                "layer1_identity": {
                    "query_count_native": 10, "query_count_annotations": 10,
                    "gallery_count_native": 20, "gallery_count_rebuilt_from_annotations": 20,
                    "gallery_order_identical": True, "reference_ids_identical": True,
                    "target_ids_identical": True,
                    "model_text_identical_to_author_composition": True,
                    "gallery_has_duplicates": False},
                "native_metrics_match_feature_rerank": True}}


def fwd(value=0.0):
    return {"max_abs": value, "max_rel": value, "min_cosine": 1.0, "bitwise_equal": False,
            "shape": [2, 4], "within_tolerance": value <= TOL["forward_max_abs"]}


def adapter(components):
    body = {"layer2_pixels": {"adapter_decode_vs_native_val_decode": {
                "max_abs": 0.0, "max_rel": 0.0, "bitwise_equal": True, "min_cosine": 1.0,
                "shape": [2, 4]}},
            # No `sample_forward`/`population` aggregate for the component scope's comparison:
            # the component record is this pair's ONLY forward evidence (LIMN's shape, per
            # parity_gate.py:1275 "the LIMN artifact records no population forward delta at all").
            "batch_size_1_vs_n": {"max_abs": 3e-07, "max_rel": 1e-06, "min_cosine": 0.9999999,
                                  "bitwise_equal": False, "shape": [2, 4],
                                  "within_tolerance": True},
            "scoring_ranking": {
                "score_max_abs_delta_vs_dot": 1.19e-07, "score_pairs_compared": 8,
                "score_max_abs_declared": 1e-06, "score_within_tolerance": True,
                "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
                "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                           "min_gap_abs": 0.0}}}
    return {"category": "shirt", "phase": "adapter", "model_id": "testmodel",
            "tolerances": dict(TOL), "standard": gate.STANDARD_ABS_REL_V2,
            "adapter": body, "component_parity": components}


def run(label, components):
    root = Path(tempfile.mkdtemp(prefix="repro-a-"))
    for name, doc in (("g1_g2.testmodel.shirt.native.json", native()),
                      ("g1_g2.testmodel.shirt.adapter.json", adapter(components))):
        (root / name).write_text(json.dumps(doc), encoding="utf-8")
    v = gate.evaluate_model("testmodel", root=root, categories=("shirt",))
    derived = [c for c in v.checks if c.name.startswith("shirt.components.")]
    print(f"\n=== {label} ===")
    print("status           :", v.status)
    print("passed           :", v.passed)
    print("applied_standard :", v.applied_standard)
    print("detail           :", v.detail)
    print("component checks :", [(c.name, c.ok, c.measured["mismatch_count"],
                                  c.measured["components_compared"],
                                  c.measured["components_total"],
                                  c.measured["coverage"],
                                  c.measured["worst_normalized"],
                                  c.measured["offending_coordinates"]) for c in derived])
    print("failures         :", [c.name for c in v.failures])
    print("missing          :", v.missing)
    return v


run("A1 as-written record (offender inside the emitted chunk)", block)

# A2a: delete ONLY the offender's rows from the preview, leave mismatch_count/offenders as recorded.
mutated = json.loads(json.dumps(block))
m = mutated["scopes"][SCOPE]
drop = m["indices"].index(OFFENDER)
for key in ("indices", "reference", "actual"):
    del m[key][drop]
m["components_emitted"] = len(m["indices"])
run("A2a offender rows deleted, mismatch_count/offenders left as recorded", mutated)

# A2b: also silence the two summary fields that contradict (offenders list, mismatch_count).
#       coverage stays "chunked", totals/digests/worst_normalized/max_abs untouched.
mutated2 = json.loads(json.dumps(mutated))
m2 = mutated2["scopes"][SCOPE]
m2["mismatch_count"] = 0
m2["offenders"] = []
run("A2b offender rows deleted + mismatch_count/offenders zeroed (digests+worst kept)", mutated2)
print("\nrecorded worst_normalized still:", mutated2["scopes"][SCOPE]["worst_normalized"],
      "max_abs:", mutated2["scopes"][SCOPE]["max_abs"],
      "coverage:", mutated2["scopes"][SCOPE]["coverage"],
      "components_emitted/total:", mutated2["scopes"][SCOPE]["components_emitted"],
      "/", mutated2["scopes"][SCOPE]["components_total"])
