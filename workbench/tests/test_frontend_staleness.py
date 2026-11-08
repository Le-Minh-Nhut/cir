"""A02 — a response must never be applied to state it does not describe.

The frontend has no DOM test runner, so the ordering rule was factored into a pure module and its
ENUMERATED transitions are checked here, with the expectations written from the rule rather than
from the implementation. The wiring is checked separately: the shell must actually consult the
guard, and the frontend must be built from the same source, or a passing primitive would be a guard
nobody uses.

**History, stated so this file is not mistaken for a weakened test.** The first version of this
guard was `workbench/frontend/staleness.ts` (`requestKey` + `mayApply(token, key)`), consulted by a
single-component `App.tsx`. This pass replaced that component with keyed data regions
(`requests.ts` `RequestRunner` + `useRegion.ts`), which carry the same two conditions — newest
generation token AND unchanged context key — plus an `AbortController` for bytes nobody wants. With
the old component gone, `staleness.ts` had **zero importers**: dead code pinned by a test that
pinned dead code. The primitive was therefore deleted and this file was repointed at the module the
shell actually uses. Every enumerated transition below still expresses a distinct rule, and the
mechanism-level rows (token ordering, abort, per-family isolation) are covered in
`test_frontend_requests.py`, so no rule was dropped in the move.

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
REQUESTS_TS = FRONTEND / "requests.ts"
USE_REGION_TS = FRONTEND / "useRegion.ts"

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

    The guard moved from `staleness.ts` called by `App.tsx` to `requests.ts` called by
    `useRegion.ts`, which every data region goes through. The assertion follows it: the module must
    be wired into the data layer (not into one page), and no page may apply a fetch answer itself.
    """
    hook = USE_REGION_TS.read_text(encoding="utf-8")
    assert 'from "./requests"' in hook, "useRegion.ts does not import the guard"
    assert "new RequestRunner(" in hook, "useRegion.ts never constructs the guard"
    assert ".run(" in hook, "useRegion.ts never asks the guard whether a response may be applied"
    # The guard must be the thing that decides, not a second copy of the rule.
    assert not re.search(r"token\s*!==\s*\w+\.current", hook), (
        "useRegion.ts compares the token inline: the rule has two implementations and they can "
        "diverge")

    # The shell must render from that layer, not from its own fetches.
    app = APP_TSX.read_text(encoding="utf-8")
    assert "useWorkbenchData" in app, "the shell does not use the guarded data layer"
    assert "fetch(" not in app, "App.tsx still fetches directly instead of through a keyed region"

    # No page may own an unguarded fetch either.
    for page in ("OverviewPage.tsx", "ExperimentsPage.tsx", "EvaluationPage.tsx", "SamplesPage.tsx",
                 "FailuresPage.tsx", "NotebookPage.tsx", "QueryInspector.tsx", "data.ts"):
        source = (FRONTEND / page).read_text(encoding="utf-8")
        bare = re.findall(r"\.then\(\s*\(?\s*\w*\s*\)?\s*=>\s*\{?\s*set", source)
        assert not bare, f"{page} applies a fetch answer without a context guard: {bare[:2]}"

    # The retired primitive must stay retired: a dead module pinned by a test is worse than neither.
    assert not (FRONTEND / "staleness.ts").exists(), (
        "staleness.ts is back but nothing imports it; the rule lives in requests.ts")


def test_the_request_key_separates_every_state_this_explorer_can_be_in() -> None:
    """Every distinct (protocol, run) pair must produce a distinct key.

    A key built by concatenation without a separator would make ("ab", "") and ("a", "b") the same
    request, and the guard would then apply a response from one to the other.
    """
    source = REQUESTS_TS.read_text(encoding="utf-8")
    # One separator rule for the whole codebase: a character that cannot occur in an id, so no two
    # distinct contexts can serialize to the same key.
    assert "\\u001f" in source, (
        "the separator must be a character that cannot occur in a protocol id or a run id")
    assert source.count("contextKey({") >= 1 and "return contextKey({ protocol, run: runId })" in source, (
        "requestKey must delegate to contextKey, or the codebase has two separator rules")

    # The rule itself, executed. Node is a build dependency of this repository (the frontend is
    # built with it), so this is not a new one, and it exercises the SHIPPING source rather than a
    # re-implementation.
    compiled = Path("/tmp/cir_staleness_check")
    compiled.mkdir(exist_ok=True)
    if shutil.which("npx") is None:
        pytest.skip("npx is not installed: the pure module cannot be compiled here")
    emitted = subprocess.run(
        ["npx", "tsc", "requests.ts", "--ignoreConfig", "--outDir", str(compiled),
         "--module", "esnext", "--target", "es2022", "--moduleResolution", "bundler",
         "--skipLibCheck"],
        cwd=FRONTEND, capture_output=True, text=True, timeout=180)
    # tsc reports type errors on STDOUT and leaves stderr empty, so asserting on stderr alone
    # turned a broken compile into "tsc unavailable" — a skip concealing the defect, with no
    # diagnostic. Only an unresolvable tool may skip; a non-zero exit with the tool present is
    # a compile failure and must fail here.
    assert emitted.returncode == 0, (
        f"requests.ts does not compile: {emitted.stdout[-200:]} {emitted.stderr[-200:]}")

    program = compiled / "requests.js"
    assert program.is_file(), "the guard did not compile"

    rows = json.dumps(TRANSITIONS)
    script = f"""
import {{requestKey, contextKey, openRequest, isCurrent}} from {json.dumps(str(program))};
const rows = {rows};
const failures = [];
for (const [rp, rr, cp, cr, expected] of rows) {{
  // A ticket minted for the REQUESTED state, then judged against the LIVE state. The live family
  // state is whatever the operator moved to; the ticket is the answer that just arrived. Same
  // token in both (nothing newer was issued), so only the KEY decides -- which is the rule.
  const ticket = {{family: "", token: 1, key: requestKey(rp, rr)}};
  const live = {{token: 1, key: requestKey(cp, cr)}};
  const got = isCurrent(ticket, live);
  if (got !== expected) failures.push([rp, rr, cp, cr, expected, got]);
}}
if (failures.length) {{
  console.error(JSON.stringify(failures));
  process.exit(1);
}}
// A newer request always wins, even when the key is unchanged: same context, newer generation.
const key = requestKey("A", "r");
const live2 = {{token: 0, key}};
const old = openRequest(live2, key);
openRequest(live2, key);
if (isCurrent(old, live2) !== false) {{
  console.error("a superseded token was applied");
  process.exit(2);
}}
// And the two-field key is exactly the general form, so the separator rule cannot drift.
if (requestKey("ab", "") !== contextKey({{protocol: "ab", run: ""}})) {{
  console.error("requestKey and contextKey disagree on the same fields");
  process.exit(3);
}}
if (requestKey("ab", "") === requestKey("a", "b")) {{
  console.error('("ab","") and ("a","b") produced the same request identity');
  process.exit(4);
}}
console.log("ok");
"""
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (
        f"the transition table failed: {result.stdout} {result.stderr}")


def test_every_request_family_that_can_outlive_its_subject_is_guarded() -> None:
    """A guard on three of four fetches is not a guard.

    `ComparePanel` fetches candidate ranks for the selected query, and the query can change while
    that request is in flight -- exactly the hazard the other regions are guarded against. It was
    the one family left unguarded, and the same race applied a previous query's comparison rows
    under the new query's header and images.
    """
    inspector = (FRONTEND / "QueryInspector.tsx").read_text(encoding="utf-8")
    start = inspector.index("function ComparePanel(")
    following = inspector.find("\nfunction ", start)
    body = inspector[start:] if following == -1 else inspector[start:following]
    assert "fetchJson" in body, "the extraction no longer finds the ComparePanel fetch"
    # The component must go through the keyed region hook, not hold its own AbortController and
    # token: two implementations of the rule can diverge (and did).
    guarded = ('from "./useRegion"' in inspector and "useRegion<" in body)
    assert guarded, (
        "the ComparePanel fetch can still be answered after its query changed:\n" + body[:400])
    # Dropping below two runs must render the reason rather than leave the previous rows on screen.
    assert "EmptyState" in body, (
        "deselecting runs leaves the previous comparison rendered instead of explaining why not")


def test_every_registered_native_protocol_is_selectable_in_the_frontend() -> None:
    """A cross-check between the registry and the UI, which is where the defect lived.

    The protocol selector filters by `protocol_id`; a protocol the registry uses but the UI cannot
    select would hide those runs entirely, and a protocol the UI offers but no run carries is
    exactly the "no compatible run" state this pass had to make safe.

    The list moved from the retired `protocols.ts` into `routes.ts` (`KNOWN_PROTOCOL_IDS`), which is
    the parser's acceptance set as well as the selector's, so there is still exactly one place to
    compare against the registry.
    """
    from workbench.backend.registry import load_registry

    sources = _frontend_sources()
    routes = sources["routes.ts"]
    declared = set(re.findall(r'^\s*"([a-z0-9_]+)",\s*$', routes, re.MULTILINE))
    assert "KNOWN_PROTOCOL_IDS" in routes, "routes.ts no longer names the protocol set"
    assert {"fashioniq_original_split", "fashioniq_val_split"} <= declared, (
        f"the registry's protocols are not in KNOWN_PROTOCOL_IDS: {sorted(declared)}")

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
    # The bundle must be built from the CURRENT sources, not a leftover artifact. Minification
    # removes identifiers, so the probe is a user-visible literal that only the current UI renders:
    # the fixed-cutoff label and the capability reason shown when candidates were not exported.
    bundles = sorted((FRONTEND / "dist" / "assets").glob("*.js"))
    assert bundles, "no JavaScript bundle under frontend/dist/assets"
    text = bundles[0].read_text(encoding="utf-8")
    assert "c\u1ed1 \u0111\u1ecbnh theo protocol" in text or "cố định theo protocol" in text, (
        "the served bundle predates the fixed-cutoff labelling: rebuild the frontend")
    assert "ch\u01b0a \u0111\u01b0\u1ee3c export" in text or "chưa được export" in text, (
        "the served bundle predates the capability-gap state: rebuild the frontend")


def test_availability_claims_are_derived_from_the_selection_not_from_a_global_counter() -> None:
    """A deployment-wide boolean cannot answer a question about one run.

    `evidence.queryLevelUnavailable` is true only when the deployment holds aggregates and NO
    per-query run. In a deployment holding both families it is false, and the pages used to fall
    through to a table — correct there by accident — while the pager printed the page count for a
    run with no rows; in the mock-only case it is true, and the pager suppressed the page count for a
    mock run that HAS rows. Both are claims about the wrong subject. The UI now asks the selection,
    through the per-run capability flags.
    """
    app = APP_TSX.read_text(encoding="utf-8")
    assert "evidence.queryLevelUnavailable" not in app, (
        "a deployment-wide flag is deciding a per-selection claim again")
    # The selection-scoped claim is a capability of the runs actually chosen.
    assert "per_query_target_ranks" in app, (
        "the shell must decide what a selection can answer from the runs' own capability flag")
    assert "selectedEntries" in app, (
        "the availability claim must be computed over the selected runs, not the deployment")

    # The pager reports a real denominator from the region it belongs to, and a region with no
    # request says so instead of printing a page count for another context's data.
    samples = (FRONTEND / "SamplesPage.tsx").read_text(encoding="utf-8")
    assert 'data-testid="pager-total"' in samples, "the pager must expose its denominator"
    assert "data-total={total}" in samples, (
        "the pager total must come from the filtered set, not from the visible page")
    assert 'kind="none-selected"' in samples, (
        "no selection must render an explicit state rather than an empty table")

    # Both failure surfaces must keep telling the operator WHY there is nothing.
    failures = (FRONTEND / "FailuresPage.tsx").read_text(encoding="utf-8")
    assert failures.count('kind="none-selected"') >= 1, (
        "the failure area must explain a missing selection")
    assert 'kind="capability"' in failures, (
        "the failure area must name a missing capability rather than show an empty table")
    assert 'kind="no-match"' in failures, (
        "a filter that matches nothing must be distinguished from having no data")


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
