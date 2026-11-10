"""R9 after-evidence: what the coded gate says about each model, derived from the artifacts.

Read-only. Writes r9_after.txt beside itself so the shell cannot garble it.
Usage: workbench/envs/ctl312h/bin/python .../raw/guards/r9_after.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
OUT = Path(__file__).with_suffix(".txt")
sys.path.insert(0, str(REPO))

from workbench.backend.parity_gate import evaluate_all, evaluate_model

MODELS = ("encoder", "limn", "hint")


def main() -> int:
    lines: list[str] = []
    for model_id in MODELS:
        verdict = evaluate_model(model_id)
        lines.append(f"== {model_id} {verdict.status} passed={verdict.passed} "
                     f"failures={len(verdict.failures)} nchecks={len(verdict.checks)}")
        lines.append(f"   dims {verdict.dimensions}")
        for check in verdict.failures:
            lines.append(f"   FAIL {check.name} declared={check.declared!r} "
                         f"measured={check.measured!r} dim={check.dimension}")
        for item in verdict.label_disagreements:
            lines.append(f"   LABEL-DISAGREEMENT {item['comparison']}.{item['subcheck']} "
                         f"declared={item['declared']!r} measured={item['measured']!r} "
                         f"harness_recorded={item['harness_recorded']!r} "
                         f"gate_verdict={item['gate_verdict']!r}")
        if verdict.advisory:
            lines.append(f"   advisory {verdict.advisory}")
    combined = evaluate_all()
    lines.append(f"ALL standard={combined['standard']} "
                 f"passed_models={combined['passed_models']}")
    for model_id, verdict in combined["models"].items():
        lines.append(f"   {model_id}: status={verdict['status']} passed={verdict['passed']} "
                     f"failure_count={verdict['failure_count']}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT} ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
