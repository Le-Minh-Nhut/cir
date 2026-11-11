// R06-follow-up contract test: "this row names no protocol" and "this model has numbers in another
// scope" are INDEPENDENT facts. The inventory used `nativeUnattributed.length` as a blocker for the
// cross-scope link, so a model whose selected protocol holds only protocol-less native rows was
// told "a number exists but its scope is unknown" and never shown the scope that has its counted row.
//
// Pure module only: `scope.ts` is compiled with the repo's own tsc and required from there, so the
// assertions run the SAME code the app ships.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "cir-cross-scope-check-"));
execFileSync(
  path.join(root, "node_modules/.bin/tsc"),
  ["scope.ts", "--ignoreConfig", "--module", "commonjs", "--target", "es2022",
   "--skipLibCheck", "--outDir", temp],
  { cwd: root, stdio: "pipe" },
);
const S = require(path.join(temp, "scope.js"));
process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));

const SELECTED = "fashioniq_full_gallery_ref_excluded";
const OTHER = "fashioniq_val_split";

/** Native rows as `/api/native-measurements` served them for tgcir: a macro that names no protocol,
 *  plus a counted, artifact-backed row attributed to another protocol. */
const nativeRows = [
  { model_id: "tgcir", protocol_id: null, metrics: { r10: 1.0, r50: 2.0 }, counted: false,
    source_kind: "rollup:all_measured_summary.json" },
  { model_id: "tgcir", protocol_id: OTHER, metrics: { r10: 40.2, r50: 60.5 }, unit: "percent",
    counted: true, artifact_backed: true, source_kind: "artifact" },
];
/** The catalog index the inventory hands `otherScopeFor`: every row that can hold a number, with the
 *  family that decides WHICH table lists it. */
const allRows = nativeRows
  .filter((row) => row.model_id && row.metrics && row.protocol_id)
  .map((row) => ({ model_id: row.model_id, protocol_id: row.protocol_id, subset: "",
                   metrics: row.metrics, family: "native" }));

function factsFor(modelId) {
  const nativeUnattributed = S.nativeUnattributedFor(nativeRows, modelId);
  const nativeHere = S.nativeRowsFor(nativeRows, modelId, SELECTED);
  const nativeOnly = nativeHere.length > 0;
  const allowed = S.crossScopeAllowed({ hasEntry: false, deferred: false, nativeOnly });
  const elsewhere = allowed
    ? S.otherScopeFor(allRows, modelId).filter((scope) =>
        scope.protocol !== SELECTED
        || (S.protocolHasSubsets(SELECTED) ? scope.subset !== "" : false))
    : [];
  const status = S.inventoryStatus({
    deferred: false, hasRun: false, runHasAggregateMetrics: false, runMissingScored: false,
    entryFamily: null, entryHasAggregateMetrics: false, nativeInScope: nativeHere.length,
    nativeUnattributed: nativeUnattributed.length, usableCount: 0, measuredEntryCount: 0,
    elsewhereCount: elsewhere.length, checkpointDownloaded: true,
  });
  return { nativeUnattributed, nativeHere, nativeOnly, elsewhere, status };
}

test("a protocol-less native row does NOT hide the scope where the model's counted row lives", () => {
  const { nativeUnattributed, elsewhere, status } = factsFor("tgcir");
  assert.equal(nativeUnattributed.length, 1, "the unattributed row must still be surfaced");
  // The link is present, and it names the scope that actually holds the attributed number.
  assert.equal(elsewhere.length, 1, "nativeUnattributed blocked the cross-scope link");
  assert.equal(elsewhere[0].protocol, OTHER);
  assert.equal(elsewhere[0].family, "native");
  // ...and the status still says the served file does not name a protocol: both facts are visible.
  assert.equal(status, "Số đo native không ghi protocol");

  // The pre-fix condition, verbatim, for the same row: it suppresses the link entirely. This is the
  // defect the assertion above pins — against that line the same model produced an empty list.
  const beforeFix = (false || false || false || nativeUnattributed.length)
    ? []
    : S.otherScopeFor(allRows, "tgcir").filter((scope) =>
        scope.protocol !== SELECTED
        || (S.protocolHasSubsets(SELECTED) ? scope.subset !== "" : false));
  assert.deepEqual(beforeFix, [], "pre-fix condition hid the scope that has tgcir's counted row");
  assert.notDeepEqual(beforeFix, elsewhere);
});

test("a native number in the CURRENT scope still suppresses the cross-scope link", () => {
  const rowsHere = [...nativeRows,
    { model_id: "tgcir", protocol_id: SELECTED, metrics: { r10: 33.3 }, unit: "percent", counted: true }];
  const nativeRowsBefore = nativeRows.length;
  // Re-run the same two decisions with a row in the selected protocol added to the read-set.
  const nativeHere = S.nativeRowsFor(rowsHere, "tgcir", SELECTED);
  assert.equal(nativeHere.length, 1);
  assert.equal(S.crossScopeAllowed({ hasEntry: false, deferred: false, nativeOnly: true }), false,
    "a model measured in the CURRENT scope must not be told to go elsewhere");
  assert.equal(rowsHere.length, nativeRowsBefore + 1);
  // An entry or a deferral likewise means this scope already has its answer.
  assert.equal(S.crossScopeAllowed({ hasEntry: true, deferred: false, nativeOnly: false }), false);
  assert.equal(S.crossScopeAllowed({ hasEntry: false, deferred: true, nativeOnly: false }), false);
  // A model with nothing anywhere is allowed to ask the question; it simply finds no rows.
  assert.equal(S.crossScopeAllowed({ hasEntry: false, deferred: false, nativeOnly: false }), true);
});
