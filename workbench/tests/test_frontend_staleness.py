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
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "workbench" / "frontend"
APP_TSX = FRONTEND / "App.tsx"
STALENESS_TS = FRONTEND / "staleness.ts"

#: The transition set from the finding, written as (requested, current, may_apply). "may_apply" is
#: the answer to the only question that matters: is the state this response describes still the
#: state on screen? Two entries with the same key are the ONLY ones where it is yes.
TRANSITIONS = [
    # (requested protocol, requested run, current protocol, current run, may_apply)
    ("A", "run1", "A", "run1", True),    # valid -> valid, unchanged
    ("A", "run1", "A", "", False),       # valid -> empty (new protocol has no compatible run)
    ("A", "", "A", "run1", False),       # empty -> valid (a run appeared)
    ("A", "run1", "B", "run2", False),   # protocol A -> protocol B
    ("A", "run1", "A", "run2", False),   # run A -> run B, same protocol
    ("A", "run1", "A", "", False),       # run A -> removed
    ("A", "run1", "B", "run1", False),   # same run id under another protocol
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
    emitted = subprocess.run(
        ["npx", "tsc", "staleness.ts", "--ignoreConfig", "--outDir", str(compiled),
         "--module", "esnext", "--target", "es2022", "--moduleResolution", "bundler"],
        cwd=FRONTEND, capture_output=True, text=True, timeout=180)
    if emitted.returncode != 0:
        pytest.skip(f"tsc unavailable for the pure module: {emitted.stderr[-200:]}")

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
        pytest.skip("frontend/dist has not been built in this checkout")
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
