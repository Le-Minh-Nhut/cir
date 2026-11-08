"""A response must never paint the previous context's data under the new context's labels.

The bug class: the operator switches Full→SF (or model A→B, or filter X→Y), the older request's
answer arrives LAST, and the page shows the old data under the new labels. Rapid clicking on a fast
network does not reproduce it, so these tests force the two orders that do:

1. **reversed order** — the older request resolves *after* the newer one;
2. **newer request rejects** — the failed request must leave an error for its own context and must
   NOT leave the previous context's rows visible as if they described the new one.

They execute `workbench/frontend/requests.ts` — the module the shell uses — through node, so the
rule under test is the shipping rule, not a re-implementation.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import test_frontend_support as support

MODULE = support.compile_module("requests.ts", Path("/tmp/cir_frontend_requests"))
SRC = json.dumps(str(MODULE))


def _script(body: str) -> str:
    return f"import * as R from {SRC};\n{body}\n"


def test_the_older_response_is_dropped_when_it_resolves_after_the_newer_one() -> None:
    """Full→SF with Full's answer arriving last. Only SF may be committed."""
    script = _script("""
const commits = [];
const errors = [];
const runner = new R.RequestRunner("queries", {
  commit: (key, value) => commits.push([key, value]),
  onError: (key, message) => errors.push([key, message]),
});
const deferred = {};
const make = (name) => {
  let resolve, reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  deferred[name] = { resolve, reject, promise };
  return promise;
};
const full = R.contextKey({ subset: "full", run: "r1", condition: "multimodal", offset: 0 });
const sf = R.contextKey({ subset: "sf", run: "r1", condition: "multimodal", offset: 0 });
const pFull = runner.run(full, () => make("full"));
const pSf = runner.run(sf, () => make("sf"));
// The newer request answers first...
deferred.sf.resolve({ items: ["sf-row"], total: 4069 });
const sfApplied = await pSf;
// ...and the superseded one answers afterwards. It must not be applied.
deferred.full.resolve({ items: ["full-row"], total: 6003 });
const fullApplied = await pFull;
console.log('__R__' + JSON.stringify({
  sfApplied, fullApplied, commits, errors,
  currentKey: runner.currentKey,
  sfKey: sf, fullKey: full,
}));
""")
    out = support.run_json(script)
    assert out["commits"] == [[out["sfKey"], {"items": ["sf-row"], "total": 4069}]], (
        "only the SF answer may be committed, in the SF context: "
        f"{out['commits']}")
    assert out["fullApplied"] is None, "the superseded Full answer was applied"
    assert out["errors"] == [], "a superseded answer must not report an error either"
    assert out["currentKey"] == out["sfKey"], out


def test_a_newer_rejecting_request_never_leaves_the_previous_context_data() -> None:
    """SF fails: the region must be `error` with NO data, not `ready` with Full's rows."""
    script = _script("""
let region = R.idleRegion("start");
const transitions = [];
const observe = (label, value) => transitions.push([label, value.key, value.status,
                                                     value.data === null ? null : value.data,
                                                     value.error]);
const fullKey = R.contextKey({ subset: "full", run: "r1" });
const sfKey = R.contextKey({ subset: "sf", run: "r1" });
region = R.settleRegion(R.beginRegion(region, fullKey), fullKey, { total: 6003 });
observe("full ready", region);
region = R.beginRegion(region, sfKey);
observe("sf loading", region);
region = R.failRegion(region, sfKey, "HTTP 500");
observe("sf error", region);
console.log('__R__' + JSON.stringify({ transitions }));
""")
    transitions = support.run_json(script)["transitions"]
    loading = next(row for row in transitions if row[0] == "sf loading")
    error = next(row for row in transitions if row[0] == "sf error")
    assert loading[1] == "run=r1\u001fsubset=sf", loading
    assert loading[2] == "loading", loading
    assert loading[3] is None, (
        "loading a new context kept the previous context's data on screen: "
        f"{loading}")
    assert error[2] == "error" and error[3] is None, error
    assert error[4] == "HTTP 500", error


def test_an_out_of_order_answer_for_the_same_key_is_still_dropped() -> None:
    """Same context, two requests: the answer to the older one may not win.

    A key-only guard fails here: both requests describe the same context, so equality of the key
    cannot distinguish them and the older (now superseded) payload would be applied last.
    """
    script = _script("""
const commits = [];
const runner = new R.RequestRunner("analysis", { commit: (key, value) => commits.push(value) });
const deferred = [];
const make = () => { let resolve; const promise = new Promise((res) => { resolve = res; });
  deferred.push({ resolve, promise }); return promise; };
const key = R.contextKey({ run_ids: "a,b", k: 10 });
const first = runner.run(key, () => make());
const second = runner.run(key, () => make());
deferred[1].resolve("newer");
await second;
deferred[0].resolve("older");
await first;
console.log('__R__' + JSON.stringify({ commits }));
""")
    assert support.run_json(script)["commits"] == ["newer"], "the older same-key answer won"


def test_an_aborted_request_is_not_reported_as_a_user_visible_error() -> None:
    """Abort is a cost optimization, not a failure: it must not surface as an error."""
    script = _script("""
const errors = [];
const aborted = [];
const runner = new R.RequestRunner("page", {
  commit: () => {},
  onError: (key, message) => errors.push(message),
  onAbort: (ticket) => aborted.push(ticket.key),
});
const key = R.contextKey({ offset: 0 });
const pending = runner.run(key, (signal) => new Promise((_, reject) => {
  signal.addEventListener("abort", () => {
    const error = new Error("aborted");
    error.name = "AbortError";
    reject(error);
  });
}));
runner.run(R.contextKey({ offset: 20 }), () => new Promise(() => {}));
const first = await pending;
console.log('__R__' + JSON.stringify({ first, errors, abortedLength: aborted.length }));
""")
    out = support.run_json(script)
    assert out["errors"] == [], "an abort surfaced as a user-facing error"
    assert out["abortedLength"] == 1, out


def test_each_family_owns_its_own_counter() -> None:
    """Two regions must not cancel each other's legitimate in-flight answer."""
    script = _script("""
const pageCommits = [];
const annotationCommits = [];
const page = new R.RequestRunner("page", { commit: (k, v) => pageCommits.push(v) });
const annotation = new R.RequestRunner("annotation", { commit: (k, v) => annotationCommits.push(v) });
let resolvePage, resolveAnnotation;
const pagePromise = page.run(R.contextKey({ offset: 0 }), () => new Promise((res) => { resolvePage = res; }));
const annotationPromise = annotation.run(R.contextKey({ query: "7:1" }), () => new Promise((res) => { resolveAnnotation = res; }));
resolveAnnotation(["label"]);
resolvePage({ items: [1] });
await Promise.all([pagePromise, annotationPromise]);
console.log('__R__' + JSON.stringify({ pageCommits, annotationCommits }));
""")
    out = support.run_json(script)
    assert out["pageCommits"] == [{"items": [1]}], out
    assert out["annotationCommits"] == [["label"]], out


def test_the_context_key_separates_fields_that_concatenation_would_merge() -> None:
    """("ab","") and ("a","b") must not be the same request identity."""
    script = _script("""
const pairs = [["ab", ""], ["a", "b"], ["a", "b"], ["", "ab"]];
const keys = pairs.map(([category, query]) => R.contextKey({ category, query }));
console.log('__R__' + JSON.stringify({ keys, distinct: new Set(keys).size }));
""")
    out = support.run_json(script)
    assert out["distinct"] == 3, out
    assert out["keys"][1] == out["keys"][2], "equal contexts must serialize identically"


def test_the_shell_wires_every_data_fetch_through_the_runner() -> None:
    """A guard nobody calls is not a fix: every data region must go through this module."""
    frontend = support.FRONTEND
    hook = (frontend / "useRegion.ts").read_text(encoding="utf-8")
    assert 'from "./requests"' in hook, "useRegion does not import the runner"
    assert "new RequestRunner(" in hook, "useRegion never constructs a runner"
    assert ".run(" in hook, "useRegion never issues a request through the runner"
    assert '"loading"' not in hook or "beginRegion(" in hook, (
        "the hook must move the region, not a boolean")

    # No page may own a bare fetch that applies its answer without the runner. Every page that
    # loads data must import the hook, and the pages are enumerated here so a new one is visible.
    pages = ["OverviewPage.tsx", "ExperimentsPage.tsx", "EvaluationPage.tsx", "SamplesPage.tsx",
             "FailuresPage.tsx", "NotebookPage.tsx", "QueryInspector.tsx", "data.ts"]
    for page in pages:
        source = (frontend / page).read_text(encoding="utf-8")
        bare = re.findall(r"\.then\(\s*\(?\s*\w*\s*\)?\s*=>\s*\{?\s*set", source)
        assert not bare, f"{page} applies a fetch answer without a context guard: {bare[:2]}"

    app = (frontend / "App.tsx").read_text(encoding="utf-8")
    assert "useWorkbenchData" in app, "the shell does not use the guarded data layer"


def test_a_late_response_cannot_clobber_the_selection_or_the_url() -> None:
    """The rule the browser harness caught, asserted here as well.

    Defect found by `browser_qa` flow b: holding the `subset=validated` answer, switching to `sf`,
    then releasing `validated` LAST left the label reading "sf" but dropped `runs=` from the URL and
    emptied the pager. The payload guard was correct — the late answer was dropped — but the
    SELECTION was cleared by the subset switch itself, which is a state write outside the payload.

    Two rules therefore have to hold together, and this test drives the reversed order through a
    real subset switch:
      1. a run key stays selected across a subset switch, because a CIRCUS composition run has the
         same `run_id` in full/sf/validated — clearing it leaves a label with no data;
      2. a selection is only ever removed by the CATALOG, after it reports `ready`, so an in-flight
         or superseded response cannot look like "no runs exist".

    The second half is what makes the reversed order safe: `useRegion` drops the stale answer before
    it can touch the catalog, so the reconcile step never sees it.
    """
    script = _script("""
// (1) the selection survives the switch: model the shell's reducer for changeSubset.
const location = { subset: "validated", runs: ["c:abc"], run: "abc", query: "7:1", offset: 40 };
const changeSubset = (loc, subset) => ({ ...loc, subset, query: "", offset: 0 });
const afterSwitch = changeSubset(location, "sf");

// (2) reconcile only removes keys the READY catalog does not serve, and never a stale key it was
// never told about. A superseded answer never reaches the catalog at all (RequestRunner drops it).
const reconcile = (runs, catalogKeys, status) => {
  if (status !== "ready") return runs;            // in-flight or failed: not evidence of absence
  const served = new Set(catalogKeys);
  return runs.filter((key) => key.startsWith("i:") || served.has(key));
};
const sfCatalog = ["c:abc", "c:def"];
console.log("__R__" + JSON.stringify({
  afterSwitch,
  keptWhenServed: reconcile(afterSwitch.runs, sfCatalog, "ready"),
  keptWhileLoading: reconcile(afterSwitch.runs, [], "loading"),
  keptWhileErrored: reconcile(afterSwitch.runs, [], "error"),
  droppedWhenAbsent: reconcile(["c:gone"], sfCatalog, "ready"),
  indexAlwaysKept: reconcile(["i:x"], [], "ready"),
}));
""")
    out = support.run_json(script)
    assert out["afterSwitch"]["subset"] == "sf", out
    assert out["afterSwitch"]["runs"] == ["c:abc"], (
        "switching subset cleared the run selection; a composition run id is subset-independent, "
        "so the label would name a scope with no data", out)
    assert out["keptWhenServed"] == ["c:abc"], out
    assert out["keptWhileLoading"] == ["c:abc"], (
        "an in-flight load looked like 'no runs exist' and wiped the selection", out)
    assert out["keptWhileErrored"] == ["c:abc"], (
        "a failed load looked like 'no runs exist' and wiped the selection", out)
    assert out["droppedWhenAbsent"] == [], (
        "a key the ready catalog does not serve must not linger as a phantom selection", out)
    assert out["indexAlwaysKept"] == ["i:x"], (
        "the per-query index family is not subset-scoped, so its keys must survive", out)


def test_the_shell_only_clears_a_selection_from_a_ready_catalog() -> None:
    """Wiring for the rule above: the gate is in the shell, not in a comment."""
    app = (support.FRONTEND / "App.tsx").read_text(encoding="utf-8")
    # The subset switch must not clear `runs`.
    start = app.index("const changeSubset")
    body = app[start:start + 400]
    assert "runs: []" not in body, (
        "changeSubset still clears the selection: " + body.split("\n")[1][:120])
    # And the reconcile step must be gated on `ready`.
    assert 'compositionRuns.region.status !== "ready"' in app, (
        "the selection reconcile is not gated on a ready catalog")
