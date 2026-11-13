// `model` must be part of the location model, not a parameter read straight off `window.location`.
//
// The defect, measured in the deployed stack (`937f2ec`): `EvaluationPage.tsx` read
// `new URLSearchParams(window.location.search).get("model")` while `routes.ts` neither parsed nor
// serialized `model`. The landing-scope effect's `replaceState` therefore dropped it, so
// `?area=evaluation&view=native&model=pair` rendered an empty native table with NO deferral notice —
// the exact misreading `webDeferralNotice` exists to prevent. It only survived when every other
// parameter was explicit and no correction ran.
//
// A parameter the UI reads must survive every surface that rewrites the URL. These tests pin the
// three properties that make that true, and the one that keeps existing links byte-identical.
const assert = require("node:assert/strict");
const test = require("node:test");
const fs = require("node:fs");
const path = require("node:path");
const os = require("node:os");
const { execFileSync } = require("node:child_process");

const root = path.resolve(__dirname, "..");
const temp = fs.mkdtempSync(path.join(os.tmpdir(), "cir-model-param-check-"));
execFileSync(
  path.join(root, "node_modules/.bin/tsc"),
  ["routes.ts", "--ignoreConfig", "--module", "commonjs", "--target", "es2022",
   "--skipLibCheck", "--outDir", temp],
  { cwd: root, stdio: "pipe" },
);
const { parseLocation, toSearch, defaultLocation, KNOWN_PROTOCOL_IDS } = require(
  path.join(temp, "routes.js"));
process.on("exit", () => fs.rmSync(temp, { recursive: true, force: true }));

const CIRCUS = "circus_fashioniq_release_v1";

test("a deferred model survives parse -> toSearch, area and view implied", () => {
  // The failing URL from the live check, verbatim.
  const url = "?area=evaluation&view=native&model=pair";
  const parsed = parseLocation(url, KNOWN_PROTOCOL_IDS);
  assert.deepEqual(parsed.problems, []);
  assert.equal(parsed.location.model, "pair", "parseLocation dropped `model`");

  const serialized = toSearch(parsed.location);
  assert.match(serialized, /model=pair/, `toSearch dropped \`model\`: ${serialized}`);

  // The round trip a replaceState performs: what lands back in the address bar must still name it.
  const again = parseLocation(serialized, KNOWN_PROTOCOL_IDS);
  assert.equal(again.location.model, "pair");
  assert.equal(again.location.view, "native");
  assert.equal(again.location.area, "evaluation");
});

test("`?view=native&model=pair` with no area keeps the model through the correction", () => {
  const parsed = parseLocation("?view=native&model=pair", KNOWN_PROTOCOL_IDS);
  assert.equal(parsed.location.model, "pair");
  assert.match(toSearch(parsed.location), /model=pair/);
});

test("the fully explicit URL still works", () => {
  const url = `?area=evaluation&view=native&protocol=${CIRCUS}&subset=full&model=pair`;
  const parsed = parseLocation(url, KNOWN_PROTOCOL_IDS);
  assert.equal(parsed.location.model, "pair");
  assert.equal(parsed.location.protocol, CIRCUS);
  assert.equal(parsed.location.subset, "full");
  assert.match(toSearch(parsed.location), /model=pair/);
});

test("`?view=native&model=pair` with no area lands on the area that owns `native`", () => {
  // `overview` (the default area) declares no views, so an unnamed area used to drop the view and
  // land on the dashboard — the link rendered nothing it asked for.
  const parsed = parseLocation("?view=native&model=pair", KNOWN_PROTOCOL_IDS);
  assert.deepEqual(parsed.problems, []);
  assert.equal(parsed.location.area, "evaluation", "the named view did not select its area");
  assert.equal(parsed.location.view, "native");
  assert.equal(parsed.location.model, "pair");
  assert.match(toSearch(parsed.location), /view=native/);
  assert.match(toSearch(parsed.location), /model=pair/);
});

test("an explicitly stated area still wins over the view's owner", () => {
  // `queue` belongs to experiments; naming `evaluation` must report the mismatch, not silently
  // move the reader to another area.
  const parsed = parseLocation("?area=evaluation&view=queue", KNOWN_PROTOCOL_IDS);
  assert.equal(parsed.location.area, "evaluation");
  assert.equal(parsed.problems.length, 1, parsed.problems);
  assert.equal(parsed.problems[0].param, "view");
});

test("a view id that belongs to no area does not take the model with it", () => {
  // NOTE: no `problems` entry is asserted here, because the default area (`overview`) declares no
  // views and the parser's guard is `views.length` — an unknown view under a view-less area has
  // nothing to be validated against. That is pre-existing behaviour outside this change's scope.
  // What must hold is the risk this change introduces: an unresolvable view may not cost the
  // caller the parameter the notice reads.
  const parsed = parseLocation("?view=not_a_view&model=pair", KNOWN_PROTOCOL_IDS);
  assert.equal(parsed.location.area, "overview", "an unknown view must not move the reader to another area");
  assert.equal(parsed.location.model, "pair", "an unknown view must not take the model with it");
  assert.match(toSearch(parsed.location), /model=pair/);
});

test("an ABSENT model writes no `model=`", () => {
  const serialized = toSearch(defaultLocation());
  assert.ok(!serialized.includes("model="), `empty model leaked into the URL: ${serialized}`);
  const parsed = parseLocation("?area=evaluation&view=native", KNOWN_PROTOCOL_IDS);
  assert.equal(parsed.location.model, "");
  assert.ok(!toSearch(parsed.location).includes("model="));
});

test("an unknown model id is PRESERVED so the notice can name it", () => {
  // Not a closed enum: the id is a registry key the page cannot validate, and dropping it would
  // leave a reader on a silently truncated page — the same failure, one layer down.
  const parsed = parseLocation("?area=evaluation&view=native&model=not_a_registered_model", KNOWN_PROTOCOL_IDS);
  assert.deepEqual(parsed.problems, [], "an unknown id must not be a URL problem");
  assert.equal(parsed.location.model, "not_a_registered_model");
  assert.match(toSearch(parsed.location), /model=not_a_registered_model/);
});

test("existing links stay byte-identical: `model` is written last, only when set", () => {
  const location = { ...defaultLocation(), area: "evaluation", view: "native" };
  const without = toSearch(location);
  const withModel = toSearch({ ...location, model: "pair" });
  // Appending is the only change: a link that never named a model is untouched.
  assert.equal(withModel, `${without}&model=pair`);
  // And the whole no-model string is unchanged from what this serializer produced before the field
  // existed (the trailing param is still `query_state`, which defaultLocation leaves empty).
  assert.ok(without.endsWith("subset=sf"));
});
