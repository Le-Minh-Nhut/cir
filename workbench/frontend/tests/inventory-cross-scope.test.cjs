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

/** Native rows as `/api/native-measurements` served them: tgcir's only protocol-less row is a ROLL-UP
 *  (`counted: false`), airknow has a genuine counted protocol-less macro plus a `duplicate_of` repeat
 *  draw in the selected protocol, and each model has one counted row somewhere. */
const nativeRows = [
  { model_id: "tgcir", protocol_id: null, metrics: { r10: 1.0, r50: 2.0 }, counted: false,
    source_kind: "rollup:all_measured_summary.json" },
  { model_id: "airknow", protocol_id: null, metrics: { r10: 3.0, r50: 4.0 }, counted: true,
    source_kind: "rollup:all_measured_summary.json" },
  { model_id: "tgcir", protocol_id: OTHER, metrics: { r10: 40.2, r50: 60.5 }, unit: "percent",
    counted: true, artifact_backed: true, source_kind: "artifact" },
  { model_id: "airknow", protocol_id: SELECTED, metrics: { r10: 5.0 }, unit: "percent",
    counted: false, duplicate_of: "airknow_run3cat_gpu_run1.json", source_kind: "artifact" },
  { model_id: "clvc_net", protocol_id: SELECTED, metrics: { r10: 6.0 }, unit: "percent",
    counted: true, source_kind: "artifact" },
];

/** The pre-fix predicates, verbatim: `statesNumber` and nothing about `counted`. */
const statesNumber = (row) => row.metrics !== null && row.metrics !== undefined;
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

// R08 at the scope layer: `counted: false` rows state a number, are still listed, and are not
// evidence. Pre-fix, all three helpers filtered on `statesNumber` alone.
const preFixUnattributed = (modelId) =>
  nativeRows.filter((row) => statesNumber(row) && row.model_id === modelId && !row.protocol_id);

test("(a) an uncounted protocol-less ROLL-UP is not an unattributed measurement", () => {
  // tgcir's only protocol-less rows are `all_measured_summary.json` / `selection_decisions.json`.
  assert.equal(preFixUnattributed("tgcir").length, 1, "fixture: pre-fix counted the roll-up");
  const { nativeUnattributed, elsewhere, status } = factsFor("tgcir");
  assert.equal(nativeUnattributed.length, 0, "a roll-up restating another row is not evidence");
  // ...so no "không ghi protocol" claim, and the link to the counted row stands on its own.
  assert.equal(status, "Có kết quả ở scope khác");
  assert.equal(elsewhere.length, 1, "nativeUnattributed blocked the cross-scope link");
  assert.equal(elsewhere[0].protocol, OTHER);
  assert.equal(elsewhere[0].family, "native");
});

test("(b) a protocol-less COUNTED row is still an unattributed measurement", () => {
  const { nativeUnattributed, status } = factsFor("airknow");
  assert.equal(nativeUnattributed.length, 1);
  assert.equal(nativeUnattributed[0].counted, true);
  assert.equal(status, "Số đo native không ghi protocol");
});

test("(c) a duplicate_of repeat is not counted as measured in its own protocol", () => {
  // airknow's `run3cat_gpu_run2..4` repeats sit at SELECTED with counted: false.
  assert.equal(nativeRows.filter((row) => row.model_id === "airknow" && row.protocol_id === SELECTED
    && statesNumber(row)).length, 1, "fixture: pre-fix counted the repeat");
  assert.deepEqual([...S.nativeMeasuredIds(nativeRows, SELECTED)].sort(), ["clvc_net"],
    "a repeat draw measured its protocol");
  assert.deepEqual([...S.nativeRowsFor(nativeRows, "airknow", SELECTED)], []);
  // The row is still served and still visible: only the scope decision excludes it.
  assert.equal(nativeRows.filter((row) => row.duplicate_of).length, 1);
  // A counted row in the same protocol DOES measure it.
  const withKeeper = [...nativeRows, { model_id: "airknow", protocol_id: SELECTED,
    metrics: { r10: 5.0 }, counted: true, source_kind: "artifact" }];
  assert.ok(S.nativeMeasuredIds(withKeeper, SELECTED).has("airknow"));
  assert.equal(S.nativeRowsFor(withKeeper, "airknow", SELECTED).length, 1);
});

test("(d) uncounted protocol-null rows yield elsewhere and no unattributed count", () => {
  const { nativeUnattributed, elsewhere, status } = factsFor("tgcir");
  assert.equal(nativeUnattributed.length, 0);
  assert.equal(elsewhere.length, 1, "the cross-scope link must survive");
  assert.equal(status, "Có kết quả ở scope khác");
  // The pre-fix condition, verbatim, for the same row: the unattributed count suppressed the link.
  const beforeFix = (false || false || false || preFixUnattributed("tgcir").length)
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
