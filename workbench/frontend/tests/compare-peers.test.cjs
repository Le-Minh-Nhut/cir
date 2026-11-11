// Compare-panel peer reachability + duplicate-label contract.
//
// Defect 1: the peers were `served.filter(same protocol+subset).slice(0, 4)` in API order, so the
// four columns were always `limn, conesep, ptha_mtst, intent` and TME could never be compared.
// Defect 2: `model / checkpoint` is not unique in a served scope (two `hint / fashioniq`, two
// `tme / fashioniq_0` in subset full), and only one of each pair exported Top-K.
//
// Pure module only: `catalog.ts` is compiled with the repo's own tsc and required from there, so the
// assertions run the SAME code the app ships. Rows are the ones `/api/composition/runs?subset=full`
// served, minus fields these decisions do not read.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "cir-compare-check-"));
execFileSync(
  path.join(root, "node_modules/.bin/tsc"),
  ["catalog.ts", "--ignoreConfig", "--module", "commonjs", "--target", "es2022",
   "--skipLibCheck", "--outDir", temp],
  { cwd: root, stdio: "pipe" },
);
const C = require(path.join(temp, "catalog.js"));
process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));

const CIRCUS = "circus_fashioniq_release_v1";

function row(model, runId, subset = "full", topk = null) {
  return {
    run_id: runId, model_id: model, checkpoint_id: "fashioniq", protocol_id: CIRCUS, subset,
    capabilities: { aggregate_metrics: true, per_query_target_ranks: true,
                    candidate_topk: topk !== null, image_availability: true },
    capability_detail: topk === null
      ? { candidate_topk: { status: "TOPK_EXPORT_NOT_REQUESTED", depth: null } }
      : { candidate_topk: { status: "AVAILABLE", depth: topk } },
  };
}

const HINT_A = "f461f391c6d4696011be8067";
const HINT_B = "4bc9893a0e3d4b8c9a1f2d3e4c5b6a70";
const TME_A = "ad477312111111111111111111111111";
const TME_B = "cd1a696d222222222222222222222222";

const served = [
  row("limn", "9e4918a6b5c243412cbc58c9", "full", 100),
  row("conesep", "aaaa1111", "full", 100),
  row("ptha_mtst", "bbbb2222", "full", 100),
  row("intent", "cccc3333", "full", 100),
  row("hint", HINT_A, "full", 100),
  row("hint", HINT_B, "full", 50),
  row("tme", TME_A, "full", null),
  row("tme", TME_B, "full", 100),
  row("limn", "dddd4444", "sf", 100),
];

test("the selection the URL holds is the first peer, so the requested comparison is on screen", () => {
  const peers = C.comparePeers(served, {
    currentRunId: HINT_A, protocol: CIRCUS, subset: "full", selected: [`c:${HINT_A}`, `c:${TME_B}`],
  });
  assert.equal(peers[0].run_id, TME_B, "the compared run comes first, not API order");
  assert.equal(peers.length, 7, "every other row of the scope is a reachable peer");
});

test("no peer is cut off: tme and the rows past the fourth are all reachable", () => {
  const peers = C.comparePeers(served, {
    currentRunId: HINT_A, protocol: CIRCUS, subset: "full", selected: [],
  });
  const ids = peers.map((peer) => peer.run_id);
  assert.equal(peers.length, served.filter((r) => r.subset === "full" && r.run_id !== HINT_A).length);
  assert.ok(ids.includes(TME_A) && ids.includes(TME_B), "both tme rows are offered");
  assert.ok(!ids.includes(HINT_A), "the inspected run is never its own peer");
  assert.ok(!ids.includes("dddd4444"), "another subset is never a peer");
});

test("a selection naming a run outside the scope adds nothing", () => {
  const peers = C.comparePeers(served, {
    currentRunId: HINT_A, protocol: CIRCUS, subset: "full", selected: ["c:nope"],
  });
  assert.equal(peers[0].run_id, "9e4918a6b5c243412cbc58c9");
});

test("duplicate labels are disambiguated by a run-id stub", () => {
  const a = C.compositionLabel(served.find((r) => r.run_id === HINT_A));
  const b = C.compositionLabel(served.find((r) => r.run_id === HINT_B));
  const ta = C.compositionLabel(served.find((r) => r.run_id === TME_A));
  const tb = C.compositionLabel(served.find((r) => r.run_id === TME_B));
  assert.notEqual(a, b, "two `hint / fashioniq` rows must not print the same label");
  assert.notEqual(ta, tb, "two `tme / fashioniq` rows must not print the same label");
  assert.match(a, new RegExp(HINT_A.slice(0, 8)));
  assert.match(b, new RegExp(HINT_B.slice(0, 8)));
});

test("the label says whether a candidate export exists, and how deep", () => {
  const withTopk = C.compositionLabel(served.find((r) => r.run_id === TME_B));
  const without = C.compositionLabel(served.find((r) => r.run_id === TME_A));
  assert.match(withTopk, /Top-K 100/);
  assert.match(without, /không có Top-K/);
  assert.doesNotMatch(without, /Top-K \d/);
});

test("the label never alters a served number", () => {
  const sample = { ...served[0], metrics: { ndcg_multimodal: 0.1498, coverage: 1.0 } };
  C.compositionLabel(sample);
  assert.equal(sample.metrics.ndcg_multimodal, 0.1498);
  assert.equal(sample.metrics.coverage, 1.0);
});

// A5: the comparison TABLE labelled peers by bare `model_id`, so two distinct LIMN runs both printed
// "limn #193" while one had Top-K 100 and the other TOPK_EXPORT_NOT_REQUESTED. Same defect class as
// the run-picker label, so the same fix: the composition label, which is unique and states the
// export state.
const LIMN_TOPK = "9e4918a6b5c243412cbc58c9";
const LIMN_NOTO = "10173dc45dea8c90fd2a284b";

test("two LIMN runs in the compare table cannot render identically", () => {
  const withTopk = C.compositionLabel(row("limn", LIMN_TOPK, "full", 100));
  const without = C.compositionLabel(row("limn", LIMN_NOTO, "full", null));
  assert.notEqual(withTopk, without, "the two limn columns must be separable");
  assert.match(withTopk, new RegExp(LIMN_TOPK.slice(0, 8)));
  assert.match(without, new RegExp(LIMN_NOTO.slice(0, 8)));
  assert.match(withTopk, /Top-K 100/);
  assert.match(without, /không có Top-K/);
  // The old rendering was the bare model id — the thing that produced two identical cells.
  assert.equal(row("limn", LIMN_TOPK, "full", 100).model_id,
               row("limn", LIMN_NOTO, "full", null).model_id);
});

test("the compare table's peer column prints the composition label, not the bare model id", () => {
  const source = fs.readFileSync(path.join(root, "QueryInspector.tsx"), "utf8");
  const table = source.slice(source.indexOf("const rows = ranksRegion.region.data"));
  assert.match(table, /compositionLabel\(peer\)/,
    "the peer column must use the disambiguated label");
  assert.doesNotMatch(table, /peers\.find\(\(peer\) => peer\.run_id === row\.key\)\?\.model_id/,
    "a bare model_id in the peer column is what made two LIMN runs indistinguishable");
  assert.match(table, /compositionLabel\(run\)/,
    "the viewed run prints the same label shape, so the columns read alike");
});
