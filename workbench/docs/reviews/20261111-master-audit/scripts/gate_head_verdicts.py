"""The gate's own verdicts on the campaign artifacts, before/after comparison surface."""
import json, sys
sys.path.insert(0, "/home/heheboiz/data/cir")
from workbench.backend import parity_gate as gate

result = gate.evaluate_all()
print(json.dumps({"standard": result["standard"],
                  "requested_standard": result["requested_standard"],
                  "advisory_standards": result["advisory_standards"],
                  "passed_models": result["passed_models"],
                  "models": {k: {"status": v["status"], "passed": v["passed"],
                                 "complete": v["complete"], "failure_count": v["failure_count"],
                                 "missing": v["missing"],
                                 "applied_standard": v["applied_standard"]}
                             for k, v in result["models"].items()}}, indent=1))
v2 = gate.evaluate_all(standard=gate.STANDARD_ABS_REL_V2)
print(json.dumps({"v2_request_standard": v2["standard"],
                  "v2_request_advisory": v2["advisory_standards"],
                  "v2_request_passed": v2["passed_models"]}, indent=1))
