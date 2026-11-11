// R06/R10 contract tests: which condition a candidate-window sentence may quote, and which native
// rows count as a measurement in the ACTIVE protocol.
//
// Pure module only: `scope.ts` is compiled with the repo's own tsc and required from there, so the
// assertions run the SAME code the app ships. No DOM, no fetch, no fixtures.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "cir-inspector-check-"));
execFileSync(
  path.join(root, "node_modules/.bin/tsc"),
  ["scope.ts", "--ignoreConfig", "--module", "commonjs", "--target", "es2022",
   "--skipLibCheck", "--outDir", temp],
  { cwd: root, stdio: "pipe" },
);
const S = require(path.join(temp, "scope.js"));
process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));

// The review's counter-example: MM rank 50, text-only rank 5, K=10, MM window on screen.
const conditions = [
  { id: "multimodal", rank: 50 },
  { id: "text_only_black_image", rank: 5 },
  { id: "image_only_empty_text", rank: 900 },
];

test("the window sentence quotes the VIEWED condition's rank, never the best one across conditions", () => {
  const verdict = S.candidateWindowVerdict({
    conditions, viewed: "multimodal", viewedLabel: "Ảnh + văn bản (MM)",
    effectiveK: 10, depth: 20, targetInShown: false, status: "ready",
  });
  assert.equal(verdict.kind, "ranked");
  assert.equal(verdict.conditionId, "multimodal");
  assert.equal(verdict.rank, 50, "the text-only condition's #5 leaked into the MM window");
  assert.equal(verdict.insideK, false);
  assert.equal(verdict.conditionLabel, "Ảnh + văn bản (MM)");

  // Switching the viewed condition switches the quoted rank with it.
  const text = S.candidateWindowVerdict({
    conditions, viewed: "text_only_black_image", viewedLabel: "Chỉ văn bản",
    effectiveK: 10, depth: 20, targetInShown: false, status: "ready",
  });
  assert.equal(text.rank, 5);
  assert.equal(text.insideK, true);
});

test("nothing conclusive is said before the window request has answered", () => {
  for (const status of ["idle", "loading"]) {
    const pending = S.candidateWindowVerdict({
      conditions, viewed: "multimodal", viewedLabel: "MM",
      effectiveK: 10, depth: 20, targetInShown: false, status,
    });
    assert.equal(pending.kind, "pending", `${status} was treated as an empty window`);
  }
  const failed = S.candidateWindowVerdict({
    conditions, viewed: "multimodal", viewedLabel: "MM",
    effectiveK: 10, depth: 20, targetInShown: false, status: "error",
  });
  assert.equal(failed.kind, "unavailable");

  // The viewed condition having no exact rank is a stated reason, not another condition's rank.
  const unscored = S.candidateWindowVerdict({
    conditions: [{ id: "multimodal", rank: null }, ...conditions.slice(1)],
    viewed: "multimodal", viewedLabel: "MM", effectiveK: 10, depth: 20,
    targetInShown: false, status: "ready",
  });
  assert.equal(unscored.kind, "unavailable");
  assert.match(unscored.reason, /MM/);
});

test("no conclusion when the target is in the window or the export stored nothing", () => {
  const base = { conditions, viewed: "multimodal", viewedLabel: "MM", effectiveK: 10, depth: 20 };
  assert.equal(S.candidateWindowVerdict({ ...base, targetInShown: true, status: "ready" }).kind, "hidden");
  assert.equal(S.candidateWindowVerdict({ ...base, depth: 0, targetInShown: false, status: "ready" }).kind, "hidden");
});

// R06/R01: only an exact (model, protocol) native row measures the model HERE. A protocol-less row is
// still reachable under its own label, and is never counted as measured in a selected protocol.
test("a protocol-less native row is not measured in a selected protocol but stays reachable", () => {
  const rows = [
    { model_id: "conesep", protocol_id: "p1", metrics: { r10: 54.76 } },
    { model_id: "tme", protocol_id: "p2", metrics: { r10: 31 } },
    { model_id: "habit", protocol_id: "p1", metrics: null },
    { model_id: "airknow", protocol_id: null, metrics: { r10: 12 } },
  ];
  // Exact (model, protocol) rows measure that model in that protocol, and nowhere else.
  assert.deepEqual([...S.nativeMeasuredIds(rows, "p1")], ["conesep"]);
  assert.deepEqual([...S.nativeMeasuredIds(rows, "p2")], ["tme"]);
  assert.equal(S.nativeRowsFor(rows, "conesep", "p2").length, 0, "a p1 row was borrowed for p2");
  assert.equal(S.nativeRowsFor(rows, "conesep", "p1").length, 1);
  assert.equal(S.nativeRowsFor(rows, "habit", "p1").length, 0, "a blocker document counted as a number");
  // The protocol-less row is NOT a measurement of any scope...
  for (const protocol of ["p1", "p2", "p3"]) {
    assert.equal(S.nativeMeasuredIds(rows, protocol).has("airknow"), false,
      `protocol-less row counted as measured in ${protocol}`);
    assert.equal(S.nativeRowsFor(rows, "airknow", protocol).length, 0);
  }
  // ...but it is not lost: it is served under its own, unattributed answer.
  assert.equal(S.nativeUnattributedFor(rows, "airknow").length, 1);
  assert.equal(S.nativeUnattributedFor(rows, "conesep").length, 0);
  assert.equal(S.nativeRowsFor(rows, null, "p1").length, 0);
});

// (b) an exact (model, protocol) native row IS measured there.
test("an exact (model, protocol) native row makes the model measured in that protocol", () => {
  const rows = [{ model_id: "csmcir", protocol_id: "fashioniq_original_split", metrics: { r10: 57.05 } }];
  assert.equal(S.nativeMeasuredIds(rows, "fashioniq_original_split").has("csmcir"), true);
  assert.equal(S.nativeMeasuredIds(rows, "fashioniq_val_split").has("csmcir"), false);
  assert.equal(S.nativeRowsFor(rows, "csmcir", "fashioniq_original_split").length, 1);
  assert.equal(S.nativeUnattributedFor(rows, "csmcir").length, 0);
});

// (c) the deferral stays the status, and it does not conceal the native number behind it.
test("a deferred model with a native row reports the deferral AND the number", () => {
  const base = {
    deferred: true, hasRun: false, runHasAggregateMetrics: false, runMissingScored: false,
    entryFamily: null, entryHasAggregateMetrics: false, nativeUnattributed: 0,
    usableCount: 0, measuredEntryCount: 0, elsewhereCount: 0, checkpointDownloaded: true,
  };
  assert.equal(S.inventoryStatus({ ...base, nativeInScope: 1 }), "Tạm gác",
    "the campaign status must not be replaced by a measurement label");
  assert.equal(S.inventoryStatus({ ...base, nativeInScope: 0 }), "Tạm gác");
  // The number is reachable, not hidden: the caller prints it beside the status (nativeInScope).
  assert.equal(S.inventoryStatus({ ...base, nativeInScope: 3 }), "Tạm gác");
});

test("a label never contradicts a row the API serves in the same scope", () => {
  const base = {
    deferred: false, hasRun: false, runHasAggregateMetrics: false, runMissingScored: false,
    entryFamily: null, entryHasAggregateMetrics: false, nativeInScope: 0, nativeUnattributed: 0,
    usableCount: 0, measuredEntryCount: 0, elsewhereCount: 0, checkpointDownloaded: true,
  };
  // Two composition runs here plus a native macro: the runs are the answer, not "native only".
  assert.match(S.inventoryStatus({ ...base, usableCount: 2, measuredEntryCount: 2, nativeInScope: 1 }),
    /2 lượt.*đã có số đo/);
  assert.doesNotMatch(S.inventoryStatus({ ...base, usableCount: 2, measuredEntryCount: 2, nativeInScope: 1 }),
    /Có dữ liệu native/);
  // An aggregate row without aggregate metrics must not read "Có số đo" (the counter excludes it).
  assert.equal(S.inventoryStatus({ ...base, entryFamily: "aggregate" }), "Chỉ có lượt chạy, chưa có số");
  assert.equal(S.inventoryStatus({ ...base, entryFamily: "aggregate", entryHasAggregateMetrics: true }), "Có số đo");
  // A run without aggregate metrics is ranks-only, not measured.
  assert.equal(S.inventoryStatus({ ...base, hasRun: true }), "Chỉ có hạng từng query");
  // An index row is never called native.
  assert.notEqual(S.inventoryStatus({ ...base, entryFamily: "index" }), "Có dữ liệu native");
  // A protocol-less native row does not make the selected scope measured.
  assert.equal(S.inventoryStatus({ ...base, nativeUnattributed: 1 }), "Số đo native không ghi protocol");
});
