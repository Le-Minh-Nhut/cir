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

// R06: a native row in the ACTIVE protocol measures the model HERE; a row naming another protocol
// does not, and a blocker document (no metrics) is never a measurement.
test("native rows measure the model in their own protocol, and only there", () => {
  const rows = [
    { model_id: "conesep", protocol_id: "p1", metrics: { r10: 54.76 } },
    { model_id: "tme", protocol_id: "p2", metrics: { r10: 31 } },
    { model_id: "habit", protocol_id: "p1", metrics: null },
    { model_id: "unstated", protocol_id: null, metrics: { r10: 12 } },
  ];
  assert.deepEqual([...S.nativeMeasuredIds(rows, "p1")].sort(), ["conesep", "unstated"]);
  assert.deepEqual([...S.nativeMeasuredIds(rows, "p2")].sort(), ["tme", "unstated"]);
  assert.equal(S.nativeRowsFor(rows, "conesep", "p2").length, 0, "a p1 row was borrowed for p2");
  assert.equal(S.nativeRowsFor(rows, "conesep", "p1").length, 1);
  assert.equal(S.nativeRowsFor(rows, "habit", "p1").length, 0, "a blocker document counted as a number");
  assert.equal(S.nativeRowsFor(rows, "unstated", "p2").length, 1, "an unstated protocol is the fallback");
  assert.equal(S.nativeRowsFor(rows, null, "p1").length, 0);
});
