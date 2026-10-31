"""A02 — a response must never be applied to state it does not describe.

The frontend has no DOM test runner, so the ordering rule is factored into
`workbench/frontend/staleness.ts` as a pure function and its ENUMERATED transitions are checked
here, with the expectations written from the rule rather than from the implementation. The wiring
is checked separately: the explorer must actually consult the guard, and the frontend must be built
from the same source, or a passing primitive would be a guard nobody uses.

The interactive half of this defect was reproduced in a real browser against the built frontend and
a real HTTP API (see workbench/docs/reviews/FINAL_DEFECT_LEDGER.md, A02).
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "workbench" / "frontend"
APP_TSX = FRONTEND / "App.tsx"
STALENESS_TS = FRONTEND / "staleness.ts"

#: The transition set from the finding: state the request was issued for -> state live when the
#: answer arrives. The rule is equality of the identity, so the empty-to-empty row is TRUE -- the
#: answer describes the same (empty) state it always did, and applying it changes nothing. That row
#: was the one the first version of this table got wrong, in the direction that would have made
#: "switch between two protocols that both lack runs" look like a refusal.
TRANSITIONS = [
    # A response may be applied to the live state, and to nothing else. Every row is a state the
    # explorer can be in when an answer arrives; `may_apply` is true only when the answer is about
    # the state on screen.
    #
    # (requested protocol, requested run, live protocol, live run, may_apply)
    ("A", "run1", "A", "run1", True),    # unchanged: the answer is about what is on screen
    ("A", "run1", "A", "run2", False),   # run A -> run B, same protocol
    ("A", "run1", "A", "", False),       # the protocol's run was removed
    ("A", "", "A", "run1", False),       # a protocol that had no run gained one
    ("A", "", "A", "", True),            # no run then and no run now: the same (empty) state
    ("A", "run1", "B", "run1", False),   # same run id under another protocol
    ("A", "run1", "B", "run2", False),   # protocol A -> protocol B
    ("A", "run1", "B", "", False),       # a protocol with a run -> one with none
    ("A", "", "B", "run1", False),       # a protocol with no run -> one that has a run
    ("A", "", "B", "", False),           # one empty protocol -> another empty protocol
]


def _frontend_sources() -> dict[str, str]:
    return {path.name: path.read_text(encoding="utf-8")
            for path in sorted(FRONTEND.glob("*.ts")) + sorted(FRONTEND.glob("*.tsx"))}


def test_the_frontend_actually_consults_the_staleness_guard() -> None:
    """A pure function nobody calls is not a fix.

    Without this, deleting the guard from the explorer would leave every staleness assertion
    passing, because they would be testing an imported module.
    """
    app = APP_TSX.read_text(encoding="utf-8")
    assert 'from "./staleness"' in app, "App.tsx does not import the guard"
    assert "mayApply(" in app, "App.tsx never asks the guard whether a response may be applied"
    assert app.count("mayApply(") >= 2, (
        "the query page and the jump-to-query path must BOTH be guarded")
    # And the guard must be the thing that decides, not a second copy of the rule.
    assert not re.search(r"token\s*!==\s*pageRequest\.current", app), (
        "App.tsx still compares the token inline: the rule has two implementations and they can "
        "diverge")
    assert "pageKey" in app and "requestKey(" in app, (
        "the request identity must carry the protocol, or the valid -> empty transition cannot be "
        "distinguished from an unchanged request")


def test_the_request_key_separates_every_state_this_explorer_can_be_in() -> None:
    """Every distinct (protocol, run) pair must produce a distinct key.

    A key built by concatenation without a separator would make ("ab", "") and ("a", "b") the same
    request, and the guard would then apply a response from one to the other.
    """
    source = STALENESS_TS.read_text(encoding="utf-8")
    assert "\\u0000" in source, (
        "the separator must be a character that cannot occur in a protocol id or a run id")

    # The rule itself, executed. Node is a build dependency of this repository (the frontend is
    # built with it), so this is not a new one, and it exercises the SHIPPING source rather than a
    # re-implementation.
    compiled = Path("/tmp/cir_staleness_check")
    compiled.mkdir(exist_ok=True)
    if shutil.which("npx") is None:
        pytest.skip("npx is not installed: the pure module cannot be compiled here")
    emitted = subprocess.run(
        ["npx", "tsc", "staleness.ts", "--ignoreConfig", "--outDir", str(compiled),
         "--module", "esnext", "--target", "es2022", "--moduleResolution", "bundler"],
        cwd=FRONTEND, capture_output=True, text=True, timeout=180)
    # tsc reports type errors on STDOUT and leaves stderr empty, so asserting on stderr alone
    # turned a broken compile into "tsc unavailable" — a skip concealing the defect, with no
    # diagnostic. Only an unresolvable tool may skip; a non-zero exit with the tool present is
    # a compile failure and must fail here.
    assert emitted.returncode == 0, (
        f"staleness.ts does not compile: {emitted.stdout[-200:]} {emitted.stderr[-200:]}")

    program = compiled / "staleness.js"
    assert program.is_file(), "the guard did not compile"

    rows = json.dumps(TRANSITIONS)
    script = f"""
import {{mayApply, requestKey}} from {json.dumps(str(program))};
const rows = {rows};
const failures = [];
for (const [rp, rr, cp, cr, expected] of rows) {{
  const got = mayApply({{token: 1, key: requestKey(rp, rr)}},
                       {{token: 1, key: requestKey(cp, cr)}});
  if (got !== expected) failures.push([rp, rr, cp, cr, expected, got]);
}}
if (failures.length) {{
  console.error(JSON.stringify(failures));
  process.exit(1);
}}
// A newer request always wins, even when the key is unchanged.
if (mayApply({{token: 1, key: requestKey("A", "r")}},
             {{token: 2, key: requestKey("A", "r")}}) !== false) {{
  console.error("a superseded token was applied");
  process.exit(2);
}}
console.log("ok");
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"the transition table failed: {result.stdout} {result.stderr}")


def test_every_request_family_that_can_outlive_its_subject_is_guarded() -> None:
    """A guard on three of four fetches is not a guard.

    `Compare` fetches `/api/compare/<query_id>` for the selected query, and the query can change
    while that request is in flight -- exactly the hazard the other three families are guarded
    against. It was the one family left unguarded, and the same race applied a previous query's
    comparison rows under the new query's header and images.
    """
    app = APP_TSX.read_text(encoding="utf-8")
    # The component is a one-liner covering many lines? No: it is one physical line, and the next
    # declaration starts on a new one. The window is therefore from its own declaration to the next
    # top-level `function` at the START of a line.
    start = app.index("function Compare(")
    following = app.find("\nfunction ", start)
    end = len(app) if following == -1 else following
    body = app[start:end]
    assert "api<" in body, "the extraction no longer finds the Compare fetch"
    # Any of the three equivalent guards is acceptable -- the shared pure guard, an AbortController,
    # or this component's own counter -- but SOME request identity must be consulted before the
    # answer is applied. What is not acceptable is an unguarded `then(setRows)`.
    guarded = ("mayApply(" in body) or ("AbortController" in body) or (
        "request.current" in body and ".then(" in body and "current)return" in body.replace(" ", ""))
    assert guarded, (
        "the Compare fetch can still be answered after its query changed:\n" + body[:400])
    # Dropping below two runs must clear the previous comparison rather than leave it on screen.
    if "runIds.length>1" in body:
        assert "setRows([])" in body, (
            "deselecting runs leaves the previous comparison rendered")


def test_every_registered_native_protocol_is_selectable_in_the_frontend() -> None:
    """A cross-check between the registry and the UI, which is where the defect lived.

    The protocol selector filters by `run.protocol_id`; a protocol the registry uses but the UI
    cannot select would hide those runs entirely, and a protocol the UI offers but no run carries is
    exactly the "no compatible run" state this pass had to make safe.
    """
    from workbench.backend.registry import load_registry

    sources = _frontend_sources()
    protocols_ts = sources["protocols.ts"]
    declared = set(re.findall(r'id:\s*"([^"]+)"', protocols_ts))
    assert declared, "no protocol ids found in protocols.ts"

    registry_protocols = {model.get("native_protocol") for model in load_registry()["models"]}
    registry_protocols.discard(None)
    missing = sorted(registry_protocols - declared)
    assert not missing, (
        f"the registry evaluates under protocols the UI cannot select: {missing}")


def test_the_built_bundle_is_not_older_than_the_sources() -> None:
    """The release verifier checks that `frontend/dist` exists; it does not check that it is current.

    A stale bundle is how a fixed source ships unfixed UI, and the Docker image builds the frontend
    from source, so the host bundle and the served bundle must both come from this source.
    """
    dist = FRONTEND / "dist" / "index.html"
    if not dist.is_file():
        # A missing bundle is exactly what the release verifier fails on (release_verify.sh J10),
        # so skipping here would conceal the defect the test names. Only a host that cannot build
        # at all (`npx` is what runs `vite build`) may skip.
        if shutil.which("npx") is None:
            pytest.skip("frontend/dist is unbuilt and no build toolchain (npx) is available")
        pytest.fail("frontend/dist is not built: run `npx vite build`")
    newest_source = max(path.stat().st_mtime for path in list(FRONTEND.glob("*.ts"))
                        + list(FRONTEND.glob("*.tsx")))
    assert dist.stat().st_mtime >= newest_source, (
        "frontend/dist is older than the sources it was built from: run `npx vite build`")


def test_availability_claims_are_derived_from_the_selection_not_from_a_global_counter() -> None:
    """A deployment-wide boolean cannot answer a question about one run.

    `evidence.queryLevelUnavailable` is true only when the deployment holds aggregates and NO
    per-query run. In a deployment holding both families it is false, and the pages used to fall
    through to a table — correct there by accident — while the pager printed the page count for a
    run with no rows; in the mock-only case it is true, and the pager suppressed the page count for a
    mock run that HAS rows. Both are claims about the wrong subject. The UI now asks the selection.
    """
    app = APP_TSX.read_text(encoding="utf-8")
    assert "evidence.queryLevelUnavailable" not in app, (
        "a deployment-wide flag is deciding a per-selection claim again")
    assert "activeRunId ? `" in app, (
        "the pager must branch on whether a run is selected")
    # Both failure pages must keep telling the operator WHY there is nothing, rather than showing an
    # empty table next to a Jaccard matrix that also has no input.
    assert app.count("aggregate-only models hold none by design") == 2, (
        "both Failure Analytics and the failure pages must explain an empty result")


def test_a_development_certificate_is_not_counted_as_a_certified_result() -> None:
    """F-4 — `certified: true` is necessary and not sufficient.

    Under a development certificate the API relaxes the EXECUTION question because there is no
    evaluator-written proof to validate; the receipt still says `certified` when the verifier's own
    metric and evidence-kind rules pass. A report judged that way is an execution record, not a
    research result, and counting it made the dashboard banner claim certified results for runs whose
    proof was never checked.

    Executed through the shipping module, with receipts shaped exactly as `_development_certification`
    returns them (`workbench/backend/main.py:563-565`).
    """
    import json as _json
    import subprocess as _subprocess

    program = FRONTEND / "evidence.ts"
    compiled = Path("/tmp/cir_evidence_check")
    compiled.mkdir(exist_ok=True)
    if shutil.which("npx") is None:
        pytest.skip("npx is not installed: the pure module cannot be compiled here")
    emitted = _subprocess.run(
        ["npx", "tsc", "evidence.ts", "--ignoreConfig", "--outDir", str(compiled),
         "--module", "esnext", "--target", "es2022", "--moduleResolution", "bundler"],
        cwd=FRONTEND, capture_output=True, text=True, timeout=180)
    # As above: tsc's diagnostics go to stdout, so a non-zero exit is a compile failure even
    # when stderr is empty.
    assert emitted.returncode == 0, (
        f"evidence.ts does not compile: {emitted.stdout[-200:]} {emitted.stderr[-200:]}")

    server = compiled / "evidence.js"
    assert server.is_file(), "the module did not compile"

    development = {"run_id": "r", "verification": {
        "certified": True, "is_research_evidence": True, "execution_authenticated": True,
        "state": "certified", "certification_mode": "development", "development_mode": True}}
    deployed = {"run_id": "r", "verification": {
        "certified": True, "is_research_evidence": True, "execution_authenticated": True,
        "state": "certified", "certification_mode": "deployment", "development_mode": False}}

    script = f"""
import {{evidenceState}} from {_json.dumps(str(server))};
const state = evidenceState([], [{_json.dumps(development)}, {_json.dumps(deployed)}]);
console.log(JSON.stringify({{aggregates: state.aggregateRuns, certified: state.certifiedAggregates}}));
"""
    result = _subprocess.run(["node", "--input-type=module", "-e", script],
                             capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    counts = _json.loads(result.stdout)
    assert counts["aggregates"] == 2, counts
    assert counts["certified"] == 1, (
        "the development-certified report was counted as a certified research result: "
        f"{counts}")
