"""Verify CSMCIR aggregate API response equals raw report artifact bytes."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from workbench.backend.main import benchmark_runs
from workbench.backend.operator_config import resolve_config

REPORTS_DIR = resolve_config().WORKBENCH_ARTIFACTS_ROOT / "reports"
CSMCIR_RUNS = [
    "20261107T054933277898Z_csmcir_fashioniq_f03887d2",
    "20261107T054343058424Z_csmcir_fashioniq_f564570e",
]


@pytest.mark.parametrize("run_id", CSMCIR_RUNS)
def test_csmcir_aggregate_api_matches_artifact_bytes(run_id: str):
    matches = list(REPORTS_DIR.glob(f"*{run_id}*_aggregate.json"))
    if not matches:
        pytest.skip(f"CSMCIR aggregate report for {run_id} absent")

    artifact_path = matches[0]
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))

    api_runs = benchmark_runs()
    target = next((r for r in api_runs if r["run_id"] == run_id), None)
    assert target is not None, f"Run {run_id} missing from /api/benchmark-runs response"

    # 1. Observed macro metrics
    assert target["observed_metrics"] == artifact["observed_metrics"]
    for m in ("r10", "r50", "mean"):
        assert target["observed_metrics"][m] == artifact["observed_metrics"][m]

    # 2. Per-category metrics
    assert target["category_metrics"] == artifact["category_metrics"]
    for cat in ("dress", "shirt", "toptee"):
        assert target["category_metrics"][cat]["r10"] == artifact["category_metrics"][cat]["r10"]
        assert target["category_metrics"][cat]["r50"] == artifact["category_metrics"][cat]["r50"]

    # 3. Paper sanity metrics
    assert target["paper_sanity_metrics"] == artifact["paper_sanity_metrics"]
    for m in ("r10", "r50", "mean"):
        assert target["paper_sanity_metrics"][m] == artifact["paper_sanity_metrics"][m]

    # 4. Gate comparisons
    art_comps = artifact.get("benchmark_gate", {}).get("comparisons", [])
    target_comps = target.get("benchmark_gate", {}).get("comparisons", [])
    assert len(target_comps) == len(art_comps)
    for t_c, a_c in zip(target_comps, art_comps):
        assert t_c["metric"] == a_c["metric"]
        assert t_c["paper"] == a_c["paper"]
        assert t_c["reproduced"] == a_c["reproduced"]
        assert t_c["relative_error_pct"] == a_c["relative_error_pct"]
        assert t_c["within_tolerance"] == a_c["within_tolerance"]

    # 5. Capability and certification state
    assert target["per_query_state"] == artifact["per_query_state"]
    assert target["per_query_state"] == "CAPABILITY_ABSENT"
    assert target["verification"]["certified"] is True
    assert target["verification"]["is_research_evidence"] is True
