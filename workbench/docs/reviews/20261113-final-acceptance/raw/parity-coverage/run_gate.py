"""Step 3 of the paired-parity coverage lane: run `parity_gate.evaluate_all()` in-process.

Why in-process: the shipped service on :8000 was started 2026-11-12 23:22 without `--reload` and
several modules changed after it (LEDGER.md L27), so its parity payload is a stale snapshot. The gate
that decides these verdicts is the module on disk, and this calls it directly -- no HTTP.

Writes `gate.json`: the aggregate plus, per model, the status, the failure count, EVERY failure's
check name with its class, and the `missing` dimensions. The class is derived here, from the check's
own fields, and the derivation is printed so a reader can audit it:

  measured_disagreement  -- the artifact recorded a value and it is outside the declared tolerance
                            (`reports_absent_evidence is False`, mandatory, not ok)
  absent_evidence_refusal-- the check reports NO claim was recorded (`reports_absent_evidence`), so it
                            is a refusal to certify, not a failed measurement
  derived_invariant      -- the gate derived the value itself (a componentwise re-derivation, a
                            cross-side identity mismatch) rather than reading one recorded boolean

The third class is not guessable from `ParityCheck` alone, so the model is asked directly: the
component checks, the identity cross-checks and the malformed path are the gate's own derivations.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

OUT = Path(__file__).resolve().parent

#: Check-name prefixes the gate DERIVES rather than reads. Kept as a named list so the classifier is
#: auditable: `component.` is the v2 recomputation over emitted values, `input.identity_agreement.` is
#: a cross-side comparison of two documents, and `input.relation.` compares a digest to its pin.
DERIVED_PREFIXES = ("component.", "input.identity_agreement.", "input.relation.")


def classify(check: dict) -> str:
    if check.get("reports_absent_evidence"):
        return "absent_evidence_refusal"
    if any(str(check.get("name", "")).startswith(prefix) for prefix in DERIVED_PREFIXES):
        return "derived_invariant"
    return "measured_disagreement"


def main() -> int:
    from workbench.backend import parity_gate

    aggregate = parity_gate.evaluate_all()
    root = parity_gate.evidence_root()
    models = {}
    for model_id, verdict in aggregate["models"].items():
        failures = [dict(check, **{"class": classify(check)}) for check in verdict["failures"]]
        models[model_id] = {
            "status": verdict["status"],
            "passed": verdict["passed"],
            "complete": verdict["complete"],
            "applied_standard": verdict["applied_standard"],
            "failure_count": verdict["failure_count"],
            "failures": failures,
            "failure_classes": sorted({item["class"] for item in failures}),
            "missing": verdict["missing"],
            "missing_by_category": verdict["missing_by_category"],
            "dimensions": verdict["dimensions"],
            "coverage_counts": {key: len(value) for key, value in verdict["coverage"].items()},
            "categories": list(verdict.get("categories") or ()),
            "evidence": verdict["evidence"],
        }
    document = {
        "generated_by": "workbench/docs/reviews/20261113-final-acceptance/raw/parity-coverage/"
                        "run_gate.py",
        "interpreter": sys.executable,
        "evidence_root": str(root),
        "discovered_models": parity_gate._discover_models(root),
        "standard": aggregate["standard"],
        "standards": aggregate["standards"],
        "passed_models": aggregate["passed_models"],
        "classification_rule": {
            "absent_evidence_refusal": "check.reports_absent_evidence is True",
            "derived_invariant": f"check.name starts with one of {list(DERIVED_PREFIXES)}",
            "measured_disagreement": "otherwise: a recorded value outside its declared tolerance",
        },
        "models": models,
    }
    (OUT / "gate.json").write_text(json.dumps(document, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"standard": document["standard"],
                      "standards": document["standards"],
                      "passed_models": document["passed_models"],
                      "models": {key: {"status": value["status"],
                                       "failure_count": value["failure_count"],
                                       "failure_classes": value["failure_classes"],
                                       "missing": value["missing"]}
                                 for key, value in models.items()}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
