"""D2 -- isCertifiedEvidence disagrees with isCertifiedAggregate.

Frontend logic is tested via Python / pytest following the convention established in
workbench/tests/test_frontend_staleness.py: the frontend package.json has no test runner,
so pure TS modules are checked via pytest using node / tsc against the shipping source.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "workbench" / "frontend"
GATE_TS = FRONTEND / "gate.ts"
EVIDENCE_TS = FRONTEND / "evidence.ts"


def test_gate_and_evidence_certification_predicates_are_consistent(tmp_path: Path) -> None:
    if shutil.which("npx") is None or shutil.which("node") is None:
        pytest.skip("npx or node is not installed")

    compiled = tmp_path / "ts_out"
    compiled.mkdir(parents=True, exist_ok=True)

    emitted = subprocess.run(
        ["npx", "tsc", "gate.ts", "evidence.ts", "--ignoreConfig", "--outDir", str(compiled),
         "--module", "commonjs", "--target", "es2022"],
        cwd=FRONTEND, capture_output=True, text=True, timeout=180
    )
    assert emitted.returncode == 0, f"tsc compilation failed: {emitted.stdout} {emitted.stderr}"

    script = f"""
    const {{ isCertifiedEvidence }} = require({json.dumps(str(compiled / "gate.js"))});
    const {{ evidenceState }} = require({json.dumps(str(compiled / "evidence.js"))});

    // Case 1: Development certificate with development_mode: true
    const devReceipt1 = {{
        certified: true,
        is_research_evidence: true,
        development_mode: true,
    }};

    // Case 2: Development certificate with certification_mode: "development"
    const devReceipt2 = {{
        certified: true,
        is_research_evidence: true,
        certification_mode: "development",
    }};

    // Case 3: Genuine certified research evidence
    const validReceipt = {{
        certified: true,
        is_research_evidence: true,
        development_mode: false,
        certification_mode: "strict",
    }};

    const gate1 = isCertifiedEvidence({{ verification: devReceipt1 }});
    const gate2 = isCertifiedEvidence({{ verification: devReceipt2 }});
    const gateValid = isCertifiedEvidence({{ verification: validReceipt }});

    const evState1 = evidenceState([], [{{ run_id: "r1", verification: devReceipt1 }}]);
    const evState2 = evidenceState([], [{{ run_id: "r2", verification: devReceipt2 }}]);
    const evStateValid = evidenceState([], [{{ run_id: "r3", verification: validReceipt }}]);

    const results = {{
        gate1,
        gate2,
        gateValid,
        ev1Certified: evState1.certifiedAggregates,
        ev2Certified: evState2.certifiedAggregates,
        evValidCertified: evStateValid.certifiedAggregates,
    }};

    console.log(JSON.stringify(results));
    """

    res = subprocess.run(
        ["node", "-e", script],
        capture_output=True, text=True, timeout=60
    )
    assert res.returncode == 0, f"node execution failed: {res.stdout} {res.stderr}"

    data = json.loads(res.stdout)
    # RED verification:
    # devReceipt1 must NOT be certified by gate: gate1 must be false, matching ev1Certified == 0.
    assert data["ev1Certified"] == 0, "evidence.ts correctly rejects development_mode"
    assert data["ev2Certified"] == 0, "evidence.ts correctly rejects certification_mode development"
    assert data["evValidCertified"] == 1, "evidence.ts accepts valid certified receipt"

    assert data["gateValid"] is True, "gate.ts must accept valid certified receipt"
    # The defect: gate.ts dropped development guards, so gate1 was True while ev1Certified was 0.
    assert data["gate1"] is False, f"gate.ts classified development_mode as certified: {data}"
    assert data["gate2"] is False, f"gate.ts classified certification_mode development as certified: {data}"
