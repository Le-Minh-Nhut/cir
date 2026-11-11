// Derived-subset lineage contract: a run re-derived from a parent's ranks must print WHY it is not a
// fresh measurement and what ITS OWN coverage is. `execution_status` is inherited from the parent
// (option b: the enum keeps COMPLETED|PARTIAL|REUSED|FAILED), so LIMN's validated subset reads
// PARTIAL while its own numbers are 586/586 — the two facts must both be on screen, and the
// denominator printed must be the SUBSET's, never the parent's.
//
// Pure module only: `scope.ts` is compiled with the repo's own tsc and required from there, so the
// assertions run the SAME code the app ships. Fixtures are the two rows `/api/composition/runs`
// served for parent `9e4918a6b5c243412cbc58c9`.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "cir-derived-check-"));
execFileSync(
  path.join(root, "node_modules/.bin/tsc"),
  ["scope.ts", "--ignoreConfig", "--module", "commonjs", "--target", "es2022",
   "--skipLibCheck", "--outDir", temp],
  { cwd: root, stdio: "pipe" },
);
const S = require(path.join(temp, "scope.js"));
process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));

const PARENT = "9e4918a6b5c243412cbc58c9";

/** Served verbatim, minus the digest fields this decision does not read. */
const validated = {
  run_id: "aee2e8d39214d877a84d7c08", model_id: "limn", subset: "validated",
  execution_status: "PARTIAL",
  metrics: { multimodal: { query_count: 586, scored_count: 586, coverage: 1.0 } },
  derived_from: {
    schema: "circus-derived-subset/v1", parent_run_id: PARENT, subset: "validated",
    membership_count: 586, coverage: 1.0, scored_count: 586,
    parent_execution_status: "PARTIAL", parent_coverage: 0.9996668332500417,
    own_coverage: 1.0, own_scored_count: 586, own_unscored_count: 0, own_complete: true,
    own_status_label: "DERIVED_COMPLETE",
  },
};
const sf = {
  run_id: "6a96c5c199526b61c1470b0e", model_id: "limn", subset: "sf",
  execution_status: "PARTIAL",
  derived_from: {
    parent_run_id: PARENT, subset: "sf", membership_count: 4069,
    parent_execution_status: "PARTIAL", own_coverage: 0.9997542393708528,
    own_scored_count: 4068, own_unscored_count: 1, own_complete: false,
    own_status_label: "DERIVED_PARTIAL",
  },
};

test("a derived row carries the DERIVED marker and names its parent", () => {
  const lineage = S.derivedLineage(validated, (id) => id.slice(0, 8));
  assert.ok(lineage, "a row with derived_from must produce a lineage line");
  assert.equal(lineage.marker, "DERIVED");
  assert.equal(lineage.parentRunId, PARENT);
  assert.match(lineage.text, /9e4918a6/, "the parent run id is printed");
  assert.doesNotMatch(lineage.text, new RegExp(PARENT), "short id, not the full parent id");
  assert.match(lineage.text, /không chạy lại GPU/, "a derived row is not a fresh measurement");
  assert.match(lineage.text, /execution_status là của parent/,
    "the inherited status must be attributed to the parent on screen");
});

test("the validated subset shows its OWN 586/586 while the parent stays partial", () => {
  const lineage = S.derivedLineage(validated, (id) => id.slice(0, 8));
  assert.equal(lineage.ownComplete, true);
  assert.match(lineage.text, /586\/586 đã chấm/, "own coverage, not the parent's 6001/6003");
  assert.match(lineage.text, /DERIVED_COMPLETE/);
  assert.doesNotMatch(lineage.text, /6,001|6003/, "the parent denominator must not be printed as this subset's");
});

test("the sf subset stays partial on its own numbers", () => {
  const lineage = S.derivedLineage(sf, (id) => id.slice(0, 8));
  assert.equal(lineage.ownComplete, false);
  assert.match(lineage.text, /4[,.]?068\/4[,.]?069 đã chấm/);
  assert.match(lineage.text, /DERIVED_PARTIAL/);
});

test("a plain run gets no lineage line", () => {
  assert.equal(S.derivedLineage({ run_id: "x", execution_status: "COMPLETED" }), null);
  assert.equal(S.derivedLineage({ derived_from: null }), null);
  assert.equal(S.derivedLineage({ derived_from: { subset: "sf" } }), null,
    "no parent run id means no lineage claim");
  assert.equal(S.derivedLineage(null), null);
});
