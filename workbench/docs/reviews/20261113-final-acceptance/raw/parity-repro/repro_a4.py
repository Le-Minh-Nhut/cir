"""A4: `components_total` is an unverified claim. A chunked 4146-component record relabelled as a
full 512-component scope keeps the sha256 of the REAL 4146-element arrays and passes."""
import copy, importlib.util, json, sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path("/home/heheboiz/data/cir"); sys.path.insert(0, str(REPO))
from workbench.backend import parity_gate as gate
SCRIPTS = REPO / "workbench/docs/reviews/20261109-native15-composition-limn/scripts"
spec = importlib.util.spec_from_file_location("producer_t4", SCRIPTS / "t4_parity.py")
t4 = importlib.util.module_from_spec(spec); sys.modules["producer_t4"] = t4
spec.loader.exec_module(t4)
TOL = {"identity": "x", "forward_max_abs": 1e-05, "forward_max_rel": 1e-04,
       "forward_min_cosine": 0.9999999, "batch_max_abs": 1e-04, "batch_max_rel": 1e-03,
       "score_max_abs": 1e-06, "tie_epsilon": 1e-06, "metrics": "exact hit-count equality"}
TOTAL, SCOPE = 4146, "sample_forward.query"
reference = np.full(TOTAL, 0.5, dtype=np.float32)
block = t4.component_block({SCOPE: t4.component_scope(
    SCOPE, "forward_parity", reference + np.float32(1e-9), reference,
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
    root = Path(tempfile.mkdtemp(prefix="repro-a4-"))
    for name, doc in (("g1_g2.testmodel.shirt.native.json", native()),
                      ("g1_g2.testmodel.shirt.adapter.json", adapter(components))):
        (root / name).write_text(json.dumps(doc), encoding="utf-8")
    v = gate.evaluate_model("testmodel", root=root, categories=("shirt",))
    d = [c for c in v.checks if c.name.startswith("shirt.components.")]
    return v, (d[0].measured if d else None)

comps = copy.deepcopy(block)
r = comps["scopes"][SCOPE]
r["components_total"] = 512
r["shape"] = [512]
r["coverage"] = "full"
r["worst_normalized"] = 0.0
r["max_abs"] = r["max_abs"]
v, m = evaluate(comps)
print("A4 chunked->full relabel, total claim 512, sha256 still of the real 4146 arrays, "
      "no offenders")
print("   status:", v.status, "| compared/total:", m["components_compared"], "/",
      m["components_total"], "| coverage:", m["coverage"])
print("   detail:", v.detail[:150])
