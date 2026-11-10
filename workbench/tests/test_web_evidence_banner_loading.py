"""D2: the evidence banner must not claim an absence while a source is still loading.

Before this file, `evidenceState([])` produced `noRealEvidence: true` and
`evidenceBanner` rendered "MOCK DEVELOPMENT DATA — no real evaluation results are loaded". On a cold
load that string is printed for as long as `/api/benchmark-runs` takes, which is a false statement
about data that has not arrived. The first test fails against the old signature/behaviour; the rest pin
the new contract so the branch cannot be dropped again.
"""
from __future__ import annotations

import json
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("evidence.ts", Path("/tmp/cir_frontend_evidence_loading"))
SRC = json.dumps(str(MODULE))


def _run(body: str) -> dict:
    return support.run_json(f"import * as E from {SRC};\n{body}")


def test_a_loading_source_is_never_reported_as_no_evidence() -> None:
    """FAILING-FIRST: this is the defect. An unsettled region is not an empty deployment."""
    out = _run("""
const state = E.evidenceState([], [], [], null,
  {runs: "loading", benchmarks: "loading", composition: "loading"});
console.log('__R__' + JSON.stringify({state, banner: E.evidenceBanner(state)}));
""")
    state, banner = out["state"], out["banner"]
    assert state["loading"] is True, state
    assert sorted(state["pendingSources"]) == ["benchmarks", "composition", "runs"], state
    assert "no real evaluation results" not in banner, (
        f"the banner claims an absence while every source is in flight: {banner!r}")
    assert "MOCK" not in banner, banner
    assert "Đang tải" in banner, banner


def test_an_idle_source_also_counts_as_unsettled() -> None:
    """`idle` is "no request has been made for this context yet" — also not evidence of absence."""
    out = _run("""
const state = E.evidenceState([], [], [], null,
  {runs: "idle", benchmarks: "idle", composition: "idle"});
console.log('__R__' + JSON.stringify({loading: state.loading, banner: E.evidenceBanner(state)}));
""")
    assert out["loading"] is True, out
    assert "no real evaluation results" not in out["banner"], out["banner"]


def test_a_failed_source_is_reported_as_a_failure_not_as_zero() -> None:
    out = _run("""
const state = E.evidenceState([], [], [], null,
  {runs: "ready", benchmarks: "error", composition: "ready"});
console.log('__R__' + JSON.stringify({state, banner: E.evidenceBanner(state)}));
""")
    assert out["state"]["failedSources"] == ["benchmarks"], out["state"]
    assert "Không tải được benchmarks" in out["banner"], out["banner"]
    assert "no real evaluation results" not in out["banner"], out["banner"]


def test_a_settled_empty_deployment_still_says_so() -> None:
    """The loading branch must not swallow the real claim: an empty, settled workspace has no data."""
    out = _run("""
const state = E.evidenceState([], [], [], null,
  {runs: "ready", benchmarks: "ready", composition: "ready"});
console.log('__R__' + JSON.stringify({state, banner: E.evidenceBanner(state)}));
""")
    assert out["state"]["loading"] is False, out["state"]
    assert out["state"]["noRealEvidence"] is True, out["state"]
    assert "no real evaluation results are loaded" in out["banner"], out["banner"]


def test_data_that_arrived_while_another_source_loads_is_counted_not_claimed_absent() -> None:
    out = _run("""
const reports = [{run_id: "r1", verification: {certified: true, is_research_evidence: true,
                  development_mode: false, certification_mode: "deployment", state: "certified"}}];
const state = E.evidenceState([], reports, [], null,
  {runs: "loading", benchmarks: "ready", composition: "loading"});
console.log('__R__' + JSON.stringify({banner: E.evidenceBanner(state), state}));
""")
    assert "Đang tải" in out["banner"], out["banner"]
    assert "1 đã chứng nhận" in out["banner"], out["banner"]
    assert "no real evaluation results" not in out["banner"], out["banner"]


def test_a_caller_that_states_no_statuses_keeps_the_old_meaning() -> None:
    """Every non-shell caller (tests, CLI readers) passes data only and must not start "loading".

    This is the contract `test_e2e_release_chain.py` asserts through `npx tsx`: a certified aggregate
    with no region state must still read `AGGREGATE-ONLY`, never "no real evaluation results".
    """
    out = _run("""
const state = E.evidenceState([], []);
console.log('__R__' + JSON.stringify({loading: state.loading, banner: E.evidenceBanner(state)}));
""")
    assert out["loading"] is False, out
    assert "no real evaluation results are loaded" in out["banner"], out["banner"]


def test_a_certified_aggregate_without_region_state_still_reads_aggregate_only() -> None:
    """The release-chain banner contract, re-asserted here so a change to the loading branch breaks it."""
    out = _run("""
const row = {run_id: "r", verification: {certified: true, is_research_evidence: true,
  execution_authenticated: true, state: "certified", certification_mode: "deployment",
  development_mode: false}};
const banner = E.evidenceBanner(E.evidenceState([], [row]));
console.log('__R__' + JSON.stringify({banner}));
""")
    banner = out["banner"]
    assert "AGGREGATE-ONLY" in banner, banner
    assert "no real evaluation results" not in banner.lower(), banner
    assert "1 certified research result" in banner, banner


def test_the_shell_passes_the_region_statuses() -> None:
    """A branch nobody calls would be the same defect with better prose."""
    app = (support.FRONTEND / "App.tsx").read_text(encoding="utf-8")
    assert "runs: data.indexRuns.region.status" in app, (
        "App.tsx does not pass the runs region status, so the loading branch cannot fire")
    assert "benchmarks: data.aggregates.region.status" in app, app
    assert "composition: data.compositionRuns.region.status" in app, app
    assert "data-evidence-loading" in app, "the banner does not expose its own state to a harness"
