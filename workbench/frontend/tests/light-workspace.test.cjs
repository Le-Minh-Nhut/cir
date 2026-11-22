// Contract tests for the light workspace's URL, candidate-request and stale-response behaviour.
//
// Pure modules only: `routes.ts`, `candidateRequest.ts` and `requests.ts` are compiled with the
// repo's own tsc into a temp dir and required from there, so the assertions run the SAME code the
// app ships. No DOM, no fetch interception, no fixtures — the real app never sees any of this.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "cir-ui-check-"));
execFileSync(
  path.join(root, "node_modules/.bin/tsc"),
  ["routes.ts", "candidateRequest.ts", "requests.ts", "analysis.ts", "evidence.ts", "--ignoreConfig",
   "--module", "commonjs", "--target", "es2022", "--skipLibCheck", "--outDir", temp],
);
const load = (name) => require(path.join(temp, `${name}.js`));
process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));

const { candidateRequestPath, candidateDisplayOptions, selectExactQuery, effectiveCandidateK } = load("candidateRequest");
const { parseLocation, toSearch, KNOWN_PROTOCOL_IDS, defaultLocation } = load("routes");
const { RequestRunner } = load("requests");
const { buildScope, withCutoff, filterRows } = load("analysis");

const CIRCUS = "circus_fashioniq_release_v1";

const { evidenceState, evidenceBanner } = load("evidence");

test("composition banner ignores unrelated mock index and reflects rendered composition rows", () => {
  const sources = { runs: "ready", benchmarks: "ready", composition: "ready" };
  const actualComposition = [{ protocol_id: CIRCUS, subset: "full", is_synthetic: false }];
  const mockIndex = [{ data_kind: "mock" }];
  const state = evidenceState(mockIndex, [], actualComposition,
    { protocolId: CIRCUS, subset: "full" }, sources, "samples");
  assert.equal(evidenceBanner(state), "");
  assert.match(evidenceBanner(evidenceState([], [], [{ ...actualComposition[0], is_synthetic: true }],
    { protocolId: CIRCUS, subset: "full" }, sources, "samples")), /MOCK DEVELOPMENT DATA/);
  assert.match(evidenceBanner(evidenceState(mockIndex, [], actualComposition,
    { protocolId: CIRCUS, subset: "full" }, sources, "experiments")), /MOCK DEVELOPMENT DATA/);
});

// --------------------------------------------------------------------- scope round-trip

for (const subset of ["full", "sf", "validated"]) {
  test(`URL preserves subset=${subset} through reload and copy-link`, () => {
    const input = parseLocation(
      `?area=samples&protocol=${CIRCUS}&subset=${subset}&runs=c:abc&query=7:2179`, KNOWN_PROTOCOL_IDS);
    assert.deepEqual(input.problems, []);
    const output = parseLocation(toSearch(input.location), KNOWN_PROTOCOL_IDS);
    assert.equal(output.location.subset, subset);
    assert.equal(output.location.query, "7:2179");
    assert.equal(output.location.protocol, CIRCUS);
    assert.deepEqual(output.location.runs, ["c:abc"]);
    assert.deepEqual(output.problems, []);
  });
}

test("Full is written to the URL even though it is the parser's non-default subset", () => {
  const location = { ...defaultLocation(), subset: "sf", area: "evaluation", view: "composition" };
  assert.match(toSearch(location), /subset=sf/);
  assert.match(toSearch({ ...location, subset: "full" }), /subset=full/);
  // The serializer's output must parse back to the same subset for every accepted value.
  for (const subset of ["full", "sf", "validated"]) {
    const round = parseLocation(toSearch({ ...location, subset }), KNOWN_PROTOCOL_IDS);
    assert.equal(round.location.subset, subset);
    assert.deepEqual(round.problems, []);
  }
});

test("Back/forward over two subsets restores each scope from its own URL", () => {
  const base = { ...defaultLocation(), area: "evaluation", view: "composition", protocol: CIRCUS };
  const full = toSearch({ ...base, subset: "full" });
  const sf = toSearch({ ...base, subset: "sf" });
  assert.notEqual(full, sf);
  // Walk the history the browser would: sf -> full -> sf.
  assert.equal(parseLocation(full, KNOWN_PROTOCOL_IDS).location.subset, "full");
  assert.equal(parseLocation(sf, KNOWN_PROTOCOL_IDS).location.subset, "sf");
  assert.equal(parseLocation(full, KNOWN_PROTOCOL_IDS).location.subset, "full");
});

// --------------------------------------------------------------------- candidate requests

const BASE_REQUEST = { runId: "run/a", subset: "sf", condition: "text_only_black_image", queryId: "7:2179&x=y", k: 10 };

test("Changing display K produces a distinct request with the same query, subset and condition", () => {
  const a = new URL(candidateRequestPath(BASE_REQUEST), "https://example.test");
  const b = new URL(candidateRequestPath({ ...BASE_REQUEST, k: 50 }), "https://example.test");
  assert.notEqual(a.href, b.href);
  assert.equal(b.searchParams.get("top_k"), "50");
  for (const key of ["subset", "condition", "query_id"]) {
    assert.equal(a.searchParams.get(key), b.searchParams.get(key));
  }
  // The query id is carried as ONE value, never split on its own separators.
  assert.equal(b.searchParams.get("query_id"), BASE_REQUEST.queryId);
  assert.equal(b.searchParams.get("x"), null);
  assert.equal(b.searchParams.get("limit"), "1");
  assert.match(b.pathname, /run%2Fa/);
});

test("Condition, query and subset changes each use a different request identity", () => {
  const paths = [
    candidateRequestPath(BASE_REQUEST),
    candidateRequestPath({ ...BASE_REQUEST, condition: "image_only_empty_text" }),
    candidateRequestPath({ ...BASE_REQUEST, queryId: "7:2180" }),
    candidateRequestPath({ ...BASE_REQUEST, subset: "full" }),
    candidateRequestPath({ ...BASE_REQUEST, runId: "run/b" }),
  ];
  assert.equal(new Set(paths).size, paths.length);
});

test("The display menu never promises more than the exported depth", () => {
  assert.deepEqual(candidateDisplayOptions(null), []);
  assert.deepEqual(candidateDisplayOptions(0), []);
  assert.deepEqual(candidateDisplayOptions(7), [1, 5, 7]);
  assert.deepEqual(candidateDisplayOptions(50), [1, 5, 10, 20, 50]);
  assert.equal(Math.max(...candidateDisplayOptions(400)), 100);
  // Every offered size is requestable, and the request is bounded by the depth as well.
  for (const depth of [7, 50, 400]) {
    for (const option of candidateDisplayOptions(depth)) {
      assert.ok(option <= depth);
      assert.ok(effectiveCandidateK(option, depth) <= Math.min(depth, 100));
    }
  }
  assert.equal(effectiveCandidateK(100, 50), 50);
  assert.equal(effectiveCandidateK(10, 0), 10);
});

// --------------------------------------------------------------------- stale responses

test("A slow older Top-K answer cannot overwrite the most recent choice", async () => {
  const commits = [];
  let resolveOld;
  let resolveNew;
  const runner = new RequestRunner("candidate", { commit: (key, data) => commits.push([key, data]) });
  const older = runner.run("K10", () => new Promise((resolve) => { resolveOld = resolve; }));
  const newer = runner.run("K50", () => new Promise((resolve) => { resolveNew = resolve; }));
  resolveNew(["latest"]);
  await newer;
  resolveOld(["stale"]);
  await older;
  assert.deepEqual(commits, [["K50", ["latest"]]]);
});

test("Leaving a candidate query invalidates its pending response", async () => {
  const commits = [];
  let resolveOld;
  const runner = new RequestRunner("candidate", { commit: (key, data) => commits.push([key, data]) });
  const pending = runner.run("old-query", () => new Promise((resolve) => { resolveOld = resolve; }));
  runner.reset("new-query");
  resolveOld(["stale"]);
  await pending;
  assert.deepEqual(commits, []);
});

test("A re-issued identical request still commits its fresh answer", async () => {
  const commits = [];
  const runner = new RequestRunner("candidate", { commit: (key, data) => commits.push([key, data]) });
  await runner.run("K10", () => Promise.resolve(["first"]));
  await runner.run("K10", () => Promise.resolve(["again"]));
  assert.deepEqual(commits, [["K10", ["first"]], ["K10", ["again"]]]);
});

// --------------------------------------------------------------------- absent queries

test("A selected query missing from the current page is never replaced with its first row", () => {
  assert.equal(selectExactQuery([{ query_id: "7:1" }], "7:2179"), null);
  assert.equal(selectExactQuery([], "7:2179"), null);
  // A prefix is not a match.
  assert.equal(selectExactQuery([{ query_id: "7:21790" }], "7:2179"), null);
  assert.equal(selectExactQuery([{ query_id: "7:2179", rank: 4 }], "7:2179").rank, 4);
});

test("consensus presets exclude partially scored qids but all-cohort keeps them", () => {
  const query = (query_id, target_rank) => ({
    query_id, category: "dress", query_txt: "", reference_id: "", positive_ids: [],
    ranks: { multimodal: target_rank }, target_rank, outcome: "ground_truth_unretrieved",
  });
  const scope = withCutoff(buildScope([
    { runKey: "full", runLabel: "full", items: [query("7:1", 11), query("7:2", 30)] },
    { runKey: "partial", runLabel: "partial", items: [query("7:1", 15)] },
  ]), 10);
  assert.equal(scope.cohort, 1);
  assert.equal(scope.partial, 1);
  assert.deepEqual(filterRows(scope, { preset: "allfail", k: 10, category: "", outcome: "", scoring: "" })
    .map((row) => row.query_id), ["7:1"]);
  assert.deepEqual(filterRows(scope, { preset: "all", k: 10, category: "", outcome: "", scoring: "" })
    .map((row) => row.query_id), ["7:1", "7:2"]);
});

test("An unknown query id in the URL is reported, not silently dropped", () => {
  const parsed = parseLocation(`?query=7:2179&condition=BAD!`, KNOWN_PROTOCOL_IDS);
  assert.equal(parsed.location.query, "7:2179");
  assert.equal(parsed.location.condition, "multimodal");
  assert.equal(parsed.problems.length, 1);
  assert.equal(parsed.problems[0].param, "condition");
});
