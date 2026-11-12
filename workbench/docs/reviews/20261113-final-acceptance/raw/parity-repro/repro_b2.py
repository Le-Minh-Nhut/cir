"""B2: (i) the repo's own writer recomputes the flag from counts; (ii) the gate never does, so a
true flag beside differing per-side counts certifies metrics_parity and can carry a full PASS."""
import json, sys, tempfile
from pathlib import Path
REPO = Path("/home/heheboiz/data/cir"); sys.path.insert(0, str(REPO))
from workbench.backend import parity_gate as gate
sys.path.insert(0, str(REPO / "workbench/docs/reviews/20261112-parity-repair/scripts"))
import limn_hits

print("(i) writer `limn_hits.build('dress')` — the flag it would write:")
res = limn_hits.build("dress")
print("   native hits :", res["native"]["hits"], "queries", res["native"]["query_count"])
print("   adapter hits:", res["adapter"]["hits"], "queries", res["adapter"]["query_count"])
print("   equality    :", res["equality"], "-> flag all(equality) =", all(res["equality"].values()))
print("   WRITER derives the flag from the counts: parity_gate is not involved.")

TOL = {"identity": "exact, zero tolerance", "forward_max_abs": 1e-05, "forward_max_rel": 1e-04,
       "forward_min_cosine": 0.9999999, "batch_max_abs": 1e-04, "batch_max_rel": 1e-03,
       "score_max_abs": 1e-06, "tie_epsilon": 1e-06, "metrics": "exact hit-count equality"}

def native(flag=True, detail=None):
    doc = {"category": "dress", "phase": "native", "model_id": "limn",
           "tolerances": dict(TOL), "layer1_identity": {
               "query_count_native": 2017, "query_count_annotations": 2017,
               "gallery_count_native": 2628, "gallery_count_rebuilt_from_annotations": 2628,
               "gallery_order_identical": True, "reference_ids_identical": True,
               "target_ids_identical": True,
               "model_text_identical_to_author_composition": True,
               "gallery_has_duplicates": False},
           "native_metrics_recheck": {"dress_r1": 18.591968, "dress_r10": 51.165097,
                                      "dress_r50": 74.268715},
           "native_rank_recall_from_features": {"r1": 18.591968, "r10": 51.165097,
                                                "r50": 74.268715}}
    if flag is not None:
        doc["native_metrics_match_feature_rerank"] = flag
    if detail is not None:
        doc["native_metrics_parity_detail"] = detail
    return doc

def fwd(v=0.0):
    return {"max_abs": v, "max_rel": v, "min_cosine": 1.0, "bitwise_equal": False,
            "shape": [2, 4], "within_tolerance": v <= TOL["forward_max_abs"]}

def adapter():
    return {"category": "dress", "phase": "adapter", "model_id": "limn",
            "tolerances": dict(TOL),
            "layer2_4_forward": {
                "layer2_pixels": {"adapter_decode_vs_native_val_decode": {
                    "max_abs": 0.0, "max_rel": 0.0, "bitwise_equal": True, "min_cosine": 1.0,
                    "shape": [2, 4]}},
                "sample_forward": {"query_val_pixels": fwd(), "target": fwd()},
                "batch_size_1_vs_n": {"max_abs": 3e-07, "max_rel": 1e-06, "min_cosine": 0.9999999,
                                      "bitwise_equal": False, "shape": [2, 4],
                                      "within_tolerance": True},
                "scoring_ranking": {
                    "score_max_abs_delta_vs_dot": 1.19e-07, "score_pairs_compared": 8,
                    "score_max_abs_declared": 1e-06, "score_within_tolerance": True,
                    "adapter_vs_argsort_unmasked": {"non_tie_count": 0},
                    "rank_differences_vs_native_val_control": {"count": 0, "non_tie_count": 0,
                                                               "min_gap_abs": 0.0}}}}

DETAIL_375_376 = {
    "declared_rule": "exact integer hit-count equality at every cutoff", "cutoffs": [1, 10, 50],
    "eligible_queries": 2017, "eligible_query_set_identical": True,
    "equality": {"r1": False, "r10": True, "r50": True},
    "native":  {"query_count": 2017, "hits": {"r1": 375, "r10": 1032, "r50": 1498}},
    "adapter": {"query_count": 2017, "hits": {"r1": 376, "r10": 1032, "r50": 1498}}}

def run(label, native_doc):
    root = Path(tempfile.mkdtemp(prefix="b2-"))
    (root / "g1_g2.limn.dress.native.json").write_text(json.dumps(native_doc))
    (root / "g1_g2.limn.dress.adapter.json").write_text(json.dumps(adapter()))
    v = gate.evaluate_model("limn", root=root, categories=("dress",))
    checks = [(c.name, c.ok, c.measured) for c in v.checks
              if c.dimension == "metrics_parity"]
    print(f"\n{label}")
    print("   status          :", v.status)
    print("   passed          :", v.passed)
    print("   metrics_parity  :", v.dimensions.get("metrics_parity", "<dimension absent>"))
    print("   metrics checks  :", checks)
    print("   detail block read by the gate? ", any("parity_detail" in (c.note or "")
                                                  for c in v.checks))

run("B2a flag=true WITH a detail block saying native 375 / adapter 376, equality r1=False",
    native(True, DETAIL_375_376))
run("B2b flag=true, NO detail block at all (a bare assertion)",
    native(True, None))
run("B2c flag=false (control: the reader does react to the boolean)",
    native(False, DETAIL_375_376))
run("B2d flag absent (LIMN before limn_hits: absence is INCOMPLETE, not a pass)",
    native(None, DETAIL_375_376))
