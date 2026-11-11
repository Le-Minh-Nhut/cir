// R16 contract tests: a counted native number is not settled when another file disagrees with it,
// and a keeper row standing for repeated draws must say the evaluator is nondeterministic.
//
// Pure module only: `scope.ts` is compiled with the repo's own tsc and required from there, so the
// assertions run the SAME code the app ships. Fixtures are the real served numbers.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "cir-native-r16-check-"));
execFileSync(
  path.join(root, "node_modules/.bin/tsc"),
  ["scope.ts", "--ignoreConfig", "--module", "commonjs", "--target", "es2022",
   "--skipLibCheck", "--outDir", temp],
  { cwd: root, stdio: "pipe" },
);
const S = require(path.join(temp, "scope.js"));
process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));

const CLVC_ARTIFACT = "checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json";
const CLVC_REVIEW = "workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json";

const clvcArtifactRow = {
  model_id: "clvc_net", protocol_id: "fashioniq_val_split", unit: "percent", counted: true,
  metrics: { r10: 29.970746517160162, r50: 58.62171212901515 },
  source: { path: CLVC_ARTIFACT }, source_kind: "artifact", conflict_with: [CLVC_REVIEW],
};
const clvcReviewRow = {
  model_id: "clvc_net", protocol_id: "fashioniq_val_split", unit: "percent", counted: true,
  metrics: { r10: 30.02079703540087, r50: 58.67189400437203 },
  source: { path: CLVC_REVIEW }, source_kind: "review_run", conflict_with: [CLVC_ARTIFACT],
};

test("a counted row with conflict_with cannot be read without the second number", () => {
  const rows = [clvcArtifactRow, clvcReviewRow];
  const chips = S.nativeConflictChips(clvcArtifactRow, rows);
  assert.equal(chips.length, 1, "one chip per conflicting file");
  assert.equal(chips[0].file, "clvc_run3cat_gpu.json");
  // Both numbers, formatted on the row's own scale, and the other file's basename.
  assert.match(chips[0].label, /29\.97%/);
  assert.match(chips[0].label, /30\.02%/);
  assert.match(chips[0].label, /clvc_run3cat_gpu\.json/);
  assert.match(chips[0].label, /CHƯA chốt|CHƯA CHỐT/i);
  // ...and the numeric cells carry the other value, so no cell shows one number alone.
  assert.match(chips[0].perMetric.r10, /30\.02%/);
  assert.match(chips[0].perMetric.r50, /58\.67%/);
  // A row with no conflict has nothing to show.
  assert.deepEqual(S.nativeConflictChips({ ...clvcArtifactRow, conflict_with: null }, rows), []);
  assert.deepEqual(S.nativeConflictChips(clvcArtifactRow, rows).length, 1);
});

const airknowKeeper = {
  model_id: "airknow", protocol_id: "fashioniq_full_gallery_ref_excluded", unit: "percent",
  counted: true, metrics: { r10: 52.924086629607395 },
  source: { path: "workbench/docs/reviews/20261111-master-audit/raw/airknow_run3cat_gpu_run1.json" },
  review_repeat_index: 1,
  variance_group: {
    metric: "r10", n: 4,
    values: [52.924086629607395, 52.44973964180556, 52.29879994438645, 52.51441404945451],
    mean: 52.54676006631348, sd: 0.26728186433775697, spread: 0.6252866852209422,
  },
};

test("a variance_group with n=4 states the spread and the upstream nondeterminism", () => {
  const summary = S.nativeVarianceSummary(airknowKeeper);
  assert.equal(summary.kind, "group");
  assert.equal(summary.n, 4);
  assert.equal(summary.mean, 52.54676006631348);
  assert.equal(summary.sd, 0.26728186433775697);
  assert.equal(summary.spread, 0.6252866852209422);
  assert.match(summary.label, /n|4 draw/);
  assert.match(summary.label, /52\.55%/, "mean must be printed");
  assert.match(summary.label, /0\.27/, "sd must be printed (a distance, no percent sign)");
  assert.match(summary.label, /0\.63/, "spread must be printed");
  assert.match(summary.label, /nondeterministic by upstream design/);
});

test("insufficient_draws says only that one draw exists", () => {
  const single = S.nativeVarianceSummary({
    ...airknowKeeper,
    review_repeat_index: 1,
    variance_group: { metric: "r10", n: 1, mean: 52.9, sd: null, spread: 0,
                      insufficient_draws: true },
  });
  assert.equal(single.kind, "single-draw");
  assert.match(single.label, /chỉ có 1 draw/);
  assert.doesNotMatch(single.label, /nondeterministic/, "one draw says nothing about stability");
  assert.doesNotMatch(single.label, /sd /);
});

test("a plain row yields no chip, no variance line and no repeat label", () => {
  const plain = {
    model_id: "csmcir", protocol_id: "fashioniq_original_split", unit: "percent", counted: true,
    metrics: { r10: 57.05, r50: 72.1 }, source: { path: "a/plain.json" },
  };
  assert.deepEqual(S.nativeConflictChips(plain, [plain]), []);
  assert.deepEqual(S.nativeVarianceSummary(plain), { kind: "none" });
  assert.equal(S.nativeRepeatLabel(plain, [plain]), null);
});

test("a repeat row names its draw index out of the keeper's group size", () => {
  const repeat = {
    model_id: "airknow", protocol_id: "fashioniq_full_gallery_ref_excluded", unit: "percent",
    counted: false, metrics: { r10: 52.44973964180556 },
    source: { path: "workbench/docs/reviews/20261111-master-audit/raw/airknow_run3cat_gpu_run2.json" },
    review_repeat_index: 2, review_repeat_count: null,
    duplicate_of: airknowKeeper.source.path, source_kind: "review_run",
  };
  const rows = [airknowKeeper, repeat];
  assert.equal(S.nativeRepeatLabel(repeat, rows), "repeat 2 of 4");
  assert.equal(S.nativeRepeatLabel(airknowKeeper, rows), "repeat 1 of 4");
  // Without the keeper in the row set the index still prints; the size does not get invented.
  assert.equal(S.nativeRepeatLabel(repeat, [repeat]), "repeat 2");
});
