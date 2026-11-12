"""B1: the flag's reader path, and a real-artifact case where counts differ while the flag is true."""
import json, shutil, sys, tempfile
from pathlib import Path
REPO = Path("/home/heheboiz/data/cir"); sys.path.insert(0, str(REPO))
from workbench.backend import parity_gate as gate
E = REPO / "workbench/docs/reviews/20261109-native15-composition-limn"

def copy(tmp):
    for c in ("dress", "shirt", "toptee"):
        for side in ("native", "adapter"):
            shutil.copy2(E / f"g1_g2.{c}.{side}.json", tmp / f"g1_g2.{c}.{side}.json")

def metrics_checks(verdict, category):
    return [(c.name, c.ok, c.measured, c.mandatory) for c in verdict.checks
            if c.dimension == "metrics_parity" and c.name.startswith(f"{category}.")]

print("REAL dress pair, unmodified:")
tmp = Path(tempfile.mkdtemp(prefix="b1-real-"))
copy(tmp)
v = gate.evaluate_model("limn", root=tmp)
print("  status:", v.status)
print("  dress metrics:", metrics_checks(v, "dress"))
print("  dress detail counts:", {k: v_ for k, v_ in
      json.load(open(tmp / "g1_g2.dress.native.json"))["native_metrics_parity_detail"]["native"]["hits"].items()})

print("\nREAL dress pair, adapter r1 raised 375 -> 376 (a rerun under a different mask), flag untouched:")
tmp2 = Path(tempfile.mkdtemp(prefix="b1-376-"))
copy(tmp2)
path = tmp2 / "g1_g2.dress.native.json"
doc = json.loads(path.read_text())
doc["native_metrics_parity_detail"]["adapter"]["hits"]["r1"] = 376
doc["native_metrics_parity_detail"]["adapter"]["recall_pct"]["r1"] = 100.0 * 376 / 2017
doc["native_metrics_parity_detail"]["equality"]["r1"] = False   # producer's own per-cutoff verdict
path.write_text(json.dumps(doc, indent=2))
print("  flag kept:", doc["native_metrics_match_feature_rerank"],
      "| detail native r1:", doc["native_metrics_parity_detail"]["native"]["hits"]["r1"],
      "| detail adapter r1:", doc["native_metrics_parity_detail"]["adapter"]["hits"]["r1"],
      "| equality:", doc["native_metrics_parity_detail"]["equality"])
v2 = gate.evaluate_model("limn", root=tmp2)
print("  status:", v2.status)
print("  dress metrics:", metrics_checks(v2, "dress"))
print("  dress dimension metrics_parity:", v2.dimensions["metrics_parity"])
print("  failures:", sorted({c.name for c in v2.failures}))

print("\nSame but with the flag flipped to False (proves the flag alone decides):")
tmp3 = Path(tempfile.mkdtemp(prefix="b1-flip-"))
copy(tmp3)
path = tmp3 / "g1_g2.dress.native.json"
doc = json.loads(path.read_text())
doc["native_metrics_parity_detail"]["adapter"]["hits"]["r1"] = 376
doc["native_metrics_match_feature_rerank"] = False
path.write_text(json.dumps(doc, indent=2))
v3 = gate.evaluate_model("limn", root=tmp3)
print("  status:", v3.status)
print("  dress metrics:", metrics_checks(v3, "dress"))
