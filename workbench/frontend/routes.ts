/** Work areas, URL state, and the legacy-route map.
 *
 *  The URL is the single source of truth for what the operator is looking at: area, view,
 *  protocol, subset, selected runs, inspected query, filters and page. Refresh, deep links and
 *  Back/Forward therefore restore the same context, and a link can be shared.
 *
 *  Two rules this module exists to enforce:
 *
 *  * Every value read from the URL is validated, and an invalid value is REPORTED (returned as a
 *    problem the shell renders) instead of being silently coerced. A URL that says
 *    `subset=bogus` must not quietly show Full data as if that were asked for.
 *  * The 12 legacy tabs keep working. `?page=Common+Failures` lands on the failure area's
 *    consensus view, `?tab=Composition` on evaluation/composition, and the old `comp_*` params
 *    are read into the same normalized state.
 *
 *  Pure and dependency-free: the mapping is verified by `workbench/tests/test_frontend_routes.py`.
 */

export const AREA_IDS = ["overview", "experiments", "evaluation", "samples", "failures", "notebook"] as const;
export type AreaId = (typeof AREA_IDS)[number];

export interface ViewDef { id: string; label: string }
export interface AreaDef {
  id: AreaId;
  label: string;
  /** One line the header may show: what this area answers. */
  question: string;
  views: ViewDef[];
  /** Legacy `?page=`/`?tab=` values that map onto this area. */
  legacy: string[];
}

export const AREAS: AreaDef[] = [
  {
    id: "overview",
    label: "Tổng quan",
    question: "Kết quả thật đang có gì, thiếu gì, và đi tiếp từ đâu",
    views: [],
    legacy: ["Dashboard", "Overview"],
  },
  {
    id: "experiments",
    label: "Thí nghiệm",
    question: "Lượt chạy, mô hình, checkpoint và trạng thái sẵn sàng",
    views: [{ id: "runs", label: "Lượt chạy" }, { id: "models", label: "Mô hình" }, { id: "queue", label: "Hàng đợi job" }],
    legacy: ["Evaluation Runs", "Models", "Runs"],
  },
  {
    id: "evaluation",
    label: "Đánh giá",
    question: "Chỉ số đo được theo protocol và subset, kèm mẫu số",
    views: [{ id: "composition", label: "CIRCUS composition" }, { id: "benchmark", label: "Benchmark tổng hợp" }],
    legacy: ["Composition", "Benchmark"],
  },
  {
    id: "samples",
    label: "Khám phá mẫu",
    question: "Từng query: ảnh, văn bản sửa, hạng theo điều kiện, ứng viên",
    views: [{ id: "explorer", label: "Inspector" }, { id: "compare", label: "So sánh mô hình" }],
    legacy: ["Sample Explorer", "Compare Models"],
  },
  {
    id: "failures",
    label: "Phân tích lỗi",
    question: "Lỗi lặp lại, unscored tách riêng, cohort lưu được cả tập",
    views: [{ id: "analytics", label: "Tổng hợp lỗi" }, { id: "consensus", label: "Đồng thuận" }, { id: "disagreement", label: "Bất đồng" }],
    legacy: ["Failure Analytics", "Common Failures", "Disagreement"],
  },
  {
    id: "notebook",
    label: "Sổ nghiên cứu",
    question: "Ghi chú gắn đúng query, cohort và giả thuyết",
    views: [{ id: "annotations", label: "Ghi chú" }, { id: "cohorts", label: "Cohort & giả thuyết" }],
    legacy: ["Annotations", "Hypotheses"],
  },
];

/** The fixed outcome cutoff, named once so no selector can pretend to control it. */
export const FIXED_CUTOFF_LABEL = "@10 (cố định theo protocol)";

/** Every legacy tab name, so the shell can prove the old route space is covered. */
export const LEGACY_PAGES = AREAS.flatMap((area) => area.legacy);

export const SUBSETS = ["full", "sf", "validated"] as const;
export type Subset = (typeof SUBSETS)[number];
export const SUBSET_LABEL: Record<Subset, string> = {
  full: "Full",
  sf: "Shortcut-Free (SF)",
  validated: "Validated",
};

/** Query conditions in the CIRCUS input-ablation release. The cutoff of the outcome
 *  classification is FIXED at 10 by the backend (`determine_query_outcome(k=10)`); it is not a
 *  user control, which is why it is named here and never bound to a selector. */
export const CONDITIONS = [
  { id: "multimodal", label: "Multimodal (MM)" },
  { id: "text_only_black_image", label: "Text-only (ảnh đen)" },
  { id: "image_only_empty_text", label: "Image-only (text rỗng)" },
] as const;
export type ConditionId = (typeof CONDITIONS)[number]["id"];

export const OUTCOMES = [
  { id: "composition_required", label: "Composition required" },
  { id: "unresolved", label: "Unresolved" },
  { id: "text_only", label: "Text-only" },
  { id: "image_only", label: "Image-only" },
  { id: "both", label: "Both" },
  { id: "unscored", label: "Unscored (pipeline/mapping)" },
] as const;
export type OutcomeId = (typeof OUTCOMES)[number]["id"];

export const ORDERS = [
  { id: "worst", label: "Hạng xấu nhất trước (bỏ qua unscored)" },
  { id: "best", label: "Hạng tốt nhất trước" },
  { id: "asc", label: "Hạng tăng dần" },
  { id: "desc", label: "Hạng giảm dần" },
] as const;
export type OrderId = (typeof ORDERS)[number]["id"];

/** Consensus cutoffs for the per-query failure analysis. These are real controls: `/api/analysis`
 *  takes `k`. Distinct from the fixed @10 composition outcome cutoff. */
export const ANALYSIS_K = [1, 5, 10, 20, 50, 100, 200];

export type ProtocolId = string;

export interface Location {
  area: AreaId;
  view: string;
  protocol: ProtocolId;
  subset: Subset;
  /** Composition release run under inspection, when the protocol is the CIRCUS release. */
  run: string;
  /** Selected per-query index runs (ordered). */
  runs: string[];
  query: string;
  condition: ConditionId;
  outcome: string;
  category: string;
  order: OrderId;
  offset: number;
  /** Query-id search inside the explorer. */
  q: string;
  /** Candidate-display depth. NEVER the composition outcome cutoff. */
  k: number;
  preset: string;
  /** `exclude` (default) hides imported released-audit rows; `include` shows them, labelled. */
  imported: "exclude" | "include";
  /** Authoritative per-query state filter from the backend contract. */
  queryState: "" | "unscored" | "retrieval_failure" | "hit";
}

export interface UrlProblem { param: string; value: string; message: string }

/** Protocols this UI knows how to label and link, independent of what is served right now.
 *  `protocolsWithData` (data.ts) narrows the SELECTOR to protocols with results; this list is the
 *  parser's acceptance set, so a bookmark or a link naming a registry protocol still resolves
 *  before the first fetch lands. One list, two uses — the registry cross-check test reads it. */
export const KNOWN_PROTOCOL_IDS = [
  "circus_fashioniq_release_v1",
  "fashioniq_original_split",
  "fashioniq_val_split",
  "fashioniq_full_gallery_ref_excluded",
] as const;

const DEFAULT_PROTOCOL = "circus_fashioniq_release_v1";

export function defaultLocation(): Location {
  return {
    area: "overview",
    view: "",
    protocol: DEFAULT_PROTOCOL,
    subset: "sf",
    run: "",
    runs: [],
    query: "",
    condition: "multimodal",
    outcome: "",
    category: "",
    order: "worst",
    offset: 0,
    q: "",
    k: 10,
    preset: "all",
    imported: "exclude",
    queryState: "",
  };
}

export function areaDef(id: AreaId): AreaDef {
  return AREAS.find((area) => area.id === id) ?? AREAS[0];
}

function oneOf<T extends string>(values: readonly T[], raw: string): T | null {
  return (values as readonly string[]).includes(raw) ? (raw as T) : null;
}

function legacyArea(page: string): { area: AreaId; view: string } | null {
  for (const area of AREAS) {
    if (area.legacy.some((name) => name.toLowerCase() === page.toLowerCase())) {
      const exact = Object.keys(LEGACY_VIEW).find((name) => name.toLowerCase() === page.toLowerCase());
      // Prefer the tab's own view; fall back to the area's first only when the map has none.
      const mapped = exact ? LEGACY_VIEW[exact] : "";
      const view = mapped && area.views.some((candidate) => candidate.id === mapped)
        ? mapped
        : (area.views[0]?.id ?? "");
      return { area: area.id, view };
    }
  }
  return null;
}

/** Which view each legacy tab belongs to.
 *
 *  `AREAS[].legacy` answers "which area", which is enough for navigation but NOT for where a
 *  bookmarked URL should land: `?page=Benchmark` was resolving to the area's FIRST view
 *  (composition), and `?page=Disagreement` to the consensus table. Both silent misroutes. This map
 *  is the authoritative tab -> view table, and it is the one the test asserts against.
 */
export const LEGACY_VIEW: Record<string, string> = {
  Dashboard: "",
  Models: "models",
  "Evaluation Runs": "runs",
  Benchmark: "benchmark",
  Composition: "composition",
  "Sample Explorer": "explorer",
  "Compare Models": "compare",
  "Common Failures": "consensus",
  Disagreement: "disagreement",
  "Failure Analytics": "analytics",
  Annotations: "annotations",
  Hypotheses: "cohorts",
};

/** Parse a `location.search` string into normalized state plus every rejected value. */
export function parseLocation(search: string, knownProtocols: string[]): { location: Location; problems: UrlProblem[] } {
  const params = new URLSearchParams(search.startsWith("?") ? search.slice(1) : search);
  const location = defaultLocation();
  const problems: UrlProblem[] = [];

  const legacyPage = params.get("page") ?? params.get("tab");
  const rawArea = params.get("area");
  if (rawArea !== null) {
    const area = oneOf(AREA_IDS, rawArea);
    if (area) location.area = area;
    else problems.push({ param: "area", value: rawArea, message: `không phải khu vực hợp lệ (${AREA_IDS.join(", ")})` });
  } else if (legacyPage !== null) {
    const mapped = legacyArea(legacyPage);
    if (mapped) location.area = mapped.area;
    else problems.push({ param: "page", value: legacyPage, message: "không thuộc 12 tab cũ nên không ánh xạ được" });
  }

  const rawView = params.get("view");
  const views = areaDef(location.area).views.map((view) => view.id);
  if (rawView !== null && views.length) {
    const view = oneOf(views, rawView);
    if (view) location.view = view;
    else problems.push({ param: "view", value: rawView, message: `khu vực ${location.area} chỉ có: ${views.join(", ")}` });
  }
  if (!location.view) {
    const mapped = legacyPage !== null ? legacyArea(legacyPage) : null;
    location.view = mapped && mapped.area === location.area
      ? mapped.view
      : (areaDef(location.area).views[0]?.id ?? "");
  }

  const rawProtocol = params.get("protocol");
  if (rawProtocol !== null) {
    if (knownProtocols.includes(rawProtocol)) location.protocol = rawProtocol;
    else problems.push({ param: "protocol", value: rawProtocol, message: `không có trong registry (${knownProtocols.join(", ")})` });
  }
  // The legacy composition links carried no protocol at all; their content is CIRCUS release data.
  if (rawProtocol === null && legacyPage !== null && /composition/i.test(legacyPage)) {
    location.protocol = DEFAULT_PROTOCOL;
  }

  const rawSubset = params.get("subset") ?? params.get("comp_subset");
  if (rawSubset !== null) {
    const subset = oneOf(SUBSETS, rawSubset);
    if (subset) location.subset = subset;
    else problems.push({ param: "subset", value: rawSubset, message: `chỉ nhận ${SUBSETS.join(", ")}` });
  }

  location.run = (params.get("run") ?? params.get("comp_run") ?? "").trim();

  const rawRuns = params.get("runs");
  if (rawRuns) {
    location.runs = rawRuns.split(",").map((id) => id.trim()).filter(Boolean);
  }

  location.query = (params.get("query") ?? params.get("comp_qid") ?? "").trim();

  const rawCondition = params.get("condition") ?? params.get("comp_cond");
  if (rawCondition !== null) {
    const condition = oneOf(CONDITIONS.map((item) => item.id), rawCondition);
    if (condition) location.condition = condition;
    else problems.push({ param: "condition", value: rawCondition, message: `chỉ nhận ${CONDITIONS.map((item) => item.id).join(", ")}` });
  }
  const rawConditionAll = oneOf(CONDITIONS.map((item) => item.id), location.condition);
  if (!rawConditionAll) location.condition = "multimodal";

  const rawOutcome = params.get("outcome") ?? params.get("comp_outcome");
  if (rawOutcome !== null && rawOutcome !== "") {
    const outcome = oneOf(OUTCOMES.map((item) => item.id), rawOutcome);
    if (outcome) location.outcome = outcome;
    else problems.push({ param: "outcome", value: rawOutcome, message: `chỉ nhận ${OUTCOMES.map((item) => item.id).join(", ")}` });
  }

  location.category = (params.get("category") ?? params.get("comp_cat") ?? "").trim();
  if (location.category && !["dress", "shirt", "toptee"].includes(location.category)) {
    problems.push({ param: "category", value: location.category, message: "chỉ nhận dress, shirt, toptee" });
    location.category = "";
  }

  const rawOrder = params.get("order") ?? params.get("comp_order");
  if (rawOrder !== null) {
    const order = oneOf(ORDERS.map((item) => item.id), rawOrder);
    if (order) location.order = order;
    else problems.push({ param: "order", value: rawOrder, message: `chỉ nhận ${ORDERS.map((item) => item.id).join(", ")}` });
  }

  const rawOffset = params.get("offset") ?? params.get("comp_offset");
  if (rawOffset !== null && rawOffset !== "") {
    const offset = Number(rawOffset);
    if (Number.isInteger(offset) && offset >= 0) location.offset = offset;
    else problems.push({ param: "offset", value: rawOffset, message: "phải là số nguyên ≥ 0" });
  }

  location.q = (params.get("q") ?? "").trim();

  const rawK = params.get("k");
  if (rawK !== null && rawK !== "") {
    const k = Number(rawK);
    if (Number.isInteger(k) && k > 0 && k <= 200) location.k = k;
    else problems.push({ param: "k", value: rawK, message: "phải là số nguyên trong 1..200" });
  }

  const rawImported = params.get("imported");
  if (rawImported !== null) {
    const imported = oneOf(["exclude", "include"] as const, rawImported);
    if (imported) location.imported = imported;
    else problems.push({ param: "imported", value: rawImported, message: "chỉ nhận exclude, include" });
  }

  const rawState = params.get("query_state");
  if (rawState !== null && rawState !== "") {
    const state = oneOf(["unscored", "retrieval_failure", "hit"] as const, rawState);
    if (state) location.queryState = state;
    else problems.push({ param: "query_state", value: rawState, message: "chỉ nhận unscored, retrieval_failure, hit" });
  }

  const rawPreset = params.get("preset");
  if (rawPreset !== null && rawPreset !== "") {
    const preset = oneOf(["all", "allfail", "80", "50", "winner", "easy"], rawPreset);
    if (preset) location.preset = preset;
    else problems.push({ param: "preset", value: rawPreset, message: "chỉ nhận all, allfail, 80, 50, winner, easy" });
  }

  return { location, problems };
}

/** Canonical search string. Only meaningful values are written, so two URLs that mean the same
 *  state serialize identically (a shared link is stable). */
export function toSearch(location: Location): string {
  const params = new URLSearchParams();
  params.set("area", location.area);
  if (location.view) params.set("view", location.view);
  params.set("protocol", location.protocol);
  if (location.subset !== "full") params.set("subset", location.subset);
  if (location.run) params.set("run", location.run);
  if (location.runs.length) params.set("runs", location.runs.join(","));
  if (location.query) params.set("query", location.query);
  if (location.condition !== "multimodal") params.set("condition", location.condition);
  if (location.outcome) params.set("outcome", location.outcome);
  if (location.category) params.set("category", location.category);
  if (location.order !== "worst") params.set("order", location.order);
  if (location.offset > 0) params.set("offset", String(location.offset));
  if (location.q) params.set("q", location.q);
  if (location.k !== 10) params.set("k", String(location.k));
  if (location.preset !== "all") params.set("preset", location.preset);
  if (location.imported === "include") params.set("imported", "include");
  if (location.queryState) params.set("query_state", location.queryState);
  return `?${params.toString()}`;
}

/** A cache-busting suffix for the canonical URL. Not part of the parsed state; it exists so a
 *  `replaceState` that changes only the hash-less search still differs from the previous URL. */
export const TO_SEARCH_VERSION = "";

/** A shareable link for one context. Used by the "copy link" affordance. */
export function hrefFor(location: Location): string {
  return `${window.location.origin}${window.location.pathname}${toSearch(location)}`;
}
