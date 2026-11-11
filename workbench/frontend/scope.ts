/** Scope selection and result labels — the two things the web surface got wrong at once.
 *
 *  W1: the shell opened on a protocol with no composition rows and a subset that protocol does not
 *  have, so every count read 0 next to a page full of measurements in another scope. Two rules:
 *
 *  * an unnamed scope is chosen from REAL data (the protocol with the most rows, then the subset with
 *    the most rows inside it), never from a hardcoded default;
 *  * a subset belongs to a protocol that declares subsets. Printing "Subset SF" beside a FashionIQ
 *    Original protocol is a claim about a cohort that does not exist.
 *
 *  §6: one badge cannot carry five independent facts. A row's measurement state, its verification,
 *  its coverage, its rerun blocker and whether its samples are viewable are separate answers, and
 *  each is rendered as its own labelled part so no reader has to infer one from another.
 *
 *  Pure (no imports): executed by `workbench/tests/test_web_scope_and_labels.py` through node.
 */

export interface ScopeCapabilities {
  aggregate_metrics?: boolean | null;
  per_query_target_ranks?: boolean | null;
  candidate_topk?: boolean | null;
  image_availability?: boolean | null;
}

export interface ScopeRow {
  model_id?: string | null;
  /** Composition rows only carry these; the index/aggregate families are counted elsewhere. */
  protocol_id?: string | null;
  subset?: string | null;
  execution_status?: string | null;
  is_failed?: boolean | null;
  capabilities?: ScopeCapabilities | null;
  metrics?: Record<string, unknown> | null;
  subset_provenance?: { own?: { evidence?: string | null } | null } | null;
  provenance_source?: string | null;
  is_synthetic?: boolean | null;
  synthetic_fixture?: boolean | null;
  smoke?: { limited?: boolean | null } | null;
  /** Which served table this row came from. Optional so existing callers keep working; when it is
   *  absent the row is treated as a composition row, which is what every pre-existing caller meant. */
  family?: "composition" | "native" | "aggregate";
}

/** Subset order the release defines, widest first. */
export const SUBSET_SEQUENCE = ["full", "sf", "validated"] as const;

/** Protocols whose circuit defines `full` / `sf` / `validated`. The others have ONE cohort, so a
 *  subset control over them would be a control that changes nothing and a label that misdescribes
 *  the data. The list is data about protocols, not a UI preference. */
export const PROTOCOLS_WITH_SUBSETS: readonly string[] = ["circus_fashioniq_release_v1"];

export function protocolHasSubsets(protocolId: string): boolean {
  return PROTOCOLS_WITH_SUBSETS.includes(protocolId);
}

export function rowIsResearch(row: ScopeRow): boolean {
  if (row.is_synthetic === true || row.synthetic_fixture === true) return false;
  if (row.provenance_source) return false;
  return true;
}

export interface ScopeChoice {
  protocol: string;
  subset: string;
  /** How many comparable rows the choice was made on, so the reason is inspectable. */
  rows: number;
  /** Which served table actually lists these rows. A native macro measurement and an aggregate
   *  report can share a protocol id but live in different views, so following a link to "the
   *  protocol" is not enough: it has to be the table that holds the row (W-02). */
  family?: "composition" | "native" | "aggregate";
}

/** The scope an unnamed URL should open: the one this deployment actually measured most of.
 *
 *  Ties break on `SUBSET_SEQUENCE` for the subset (width first) and on the protocol id for the
 *  protocol, so the result is deterministic — a landing page that changes between reloads is a
 *  landing page nobody can cite.
 */
export function chooseInitialScope(rows: ScopeRow[]): ScopeChoice | null {
  const research = rows.filter(rowIsResearch);
  if (!research.length) return null;
  const byProtocol = new Map<string, ScopeRow[]>();
  for (const row of research) {
    const protocol = row.protocol_id ?? "";
    if (!protocol) continue;
    byProtocol.set(protocol, [...(byProtocol.get(protocol) ?? []), row]);
  }
  if (!byProtocol.size) return null;
  const [protocol, protocolRows] = [...byProtocol.entries()].sort((left, right) =>
    right[1].length - left[1].length || left[0].localeCompare(right[0]))[0];
  const measureable = protocolRows.filter((row) => (row.metrics?.coverage ?? null) !== 0
    && row.execution_status !== "FAILED" && row.is_failed !== true);
  const pool = measureable.length ? measureable : protocolRows;
  const bySubset = new Map<string, number>();
  for (const row of pool) {
    const subset = row.subset ?? "";
    if (!subset) continue;
    bySubset.set(subset, (bySubset.get(subset) ?? 0) + 1);
  }
  const ranked = [...bySubset.entries()].sort((left, right) =>
    right[1] - left[1]
    || SUBSET_SEQUENCE.indexOf(left[0] as never) - SUBSET_SEQUENCE.indexOf(right[0] as never));
  return { protocol, subset: ranked[0]?.[0] ?? "", rows: ranked[0]?.[1] ?? 0 };
}

/** The best subset for ONE protocol, for when the URL named a protocol but no subset.
 *
 *  Same rule as `chooseInitialScope`, restricted to the protocol the reader asked for: an explicit
 *  protocol is honoured, and only the subset he left unstated is filled from real rows.
 */
export function chooseSubsetFor(rows: ScopeRow[], protocol: string): string | null {
  const scoped = rows.filter((row) => rowIsResearch(row) && row.protocol_id === protocol);
  if (!scoped.length) return null;
  const measureable = scoped.filter((row) => (row.metrics?.coverage ?? null) !== 0
    && row.execution_status !== "FAILED" && row.is_failed !== true);
  const pool = measureable.length ? measureable : scoped;
  const counts = new Map<string, number>();
  for (const row of pool) {
    const subset = row.subset ?? "";
    if (!subset) continue;
    counts.set(subset, (counts.get(subset) ?? 0) + 1);
  }
  const ranked = [...counts.entries()].sort((left, right) =>
    right[1] - left[1]
    || SUBSET_SEQUENCE.indexOf(left[0] as never) - SUBSET_SEQUENCE.indexOf(right[0] as never));
  return ranked[0]?.[0] ?? null;
}

export interface ScopeCounts {
  /** Rows the API served for this protocol+subset, of any state. */
  served: number;
  /** Rows whose evaluator reported completion (a number was measured). */
  completed: number;
  /** Rows whose per-query artifact exists on disk (`capabilities.per_query_target_ranks`). */
  artifactPresent: number;
  /** Rows whose subset membership was re-derived from that artifact. */
  membershipDerived: number;
  /** Rows that hold a verifier receipt. Composition rows hold none: they are run summaries, not
   *  certified reports, and claiming otherwise is the defect this field exists to prevent. */
  verified: number;
  failed: number;
}

/** Six separate counts, never collapsed: "9 rows" says nothing about how many of them can be
 *  opened, re-derived, or trusted. */
export function scopeCounts(rows: ScopeRow[], protocol: string, subset: string): ScopeCounts {
  const scoped = rows.filter((row) => row.protocol_id === protocol
    && (protocolHasSubsets(protocol) ? row.subset === subset : true));
  return {
    served: scoped.length,
    completed: scoped.filter((row) => row.execution_status === "COMPLETED").length,
    artifactPresent: scoped.filter((row) => row.capabilities?.per_query_target_ranks === true).length,
    membershipDerived: scoped.filter((row) =>
      row.subset_provenance?.own?.evidence === "PER_QUERY_ARTIFACT").length,
    verified: 0,
    failed: scoped.filter((row) => row.execution_status === "FAILED" || row.is_failed === true).length,
  };
}

/** Where a model DOES have results, when it has none in the scope on screen.
 *
 *  The old inventory rendered "Chưa có số đo / Cần chạy" for a model whose rows simply lived in
 *  another protocol or subset, which reads as "nobody ran it". A real row elsewhere is a link, and
 *  the link names the scope so the reader can see the cohort change before they follow it.
 */
export function otherScopeFor(rows: ScopeRow[], model?: string | null): ScopeChoice[] {
  const research = rows.filter(rowIsResearch);
  const byScope = new Map<string, ScopeRow[]>();
  for (const row of research) {
    const protocol = row.protocol_id ?? "";
    if (!protocol) continue;
    // `family` is part of the identity: one model can hold a native macro AND an aggregate report
    // under the same protocol, and those are listed in two different tables. Collapsing them into
    // one link sent the reader to whichever table happened to be grouped first (W-02).
    const key = `${protocol}\u001f${row.subset ?? ""}\u001f${row.family ?? "composition"}`;
    byScope.set(key, [...(byScope.get(key) ?? []), row]);
  }
  if (!model) return [];
  return [...byScope.entries()]
    .filter(([, scoped]) => scoped.some((row) => row.model_id === model))
    .map(([key, scoped]) => {
      const [protocol, subset, family] = key.split("\u001f");
      return {
        protocol, subset, rows: scoped.length,
        family: family as ScopeChoice["family"],
      };
    })
    .sort((left, right) => left.protocol.localeCompare(right.protocol)
      || SUBSET_SEQUENCE.indexOf(left.subset as never) - SUBSET_SEQUENCE.indexOf(right.subset as never));
}

/** The native-row fields a scope decision needs: exactly what `/api/native-measurements` serves per
 *  measurement, and what a native row carries after it has been mapped into a `ScopeRow`.
 *
 *  R16: two served facts make a counted number provisional, and neither is a scope decision, so they
 *  live here as the fields the pure functions below READ (`nativeConflictChips`,
 *  `nativeVarianceSummary`, `nativeRepeatLabel`) — the wire contract that types the whole row is
 *  `NativeMeasurement` in `data.ts`. Keeping the read-set here is what lets a test execute the
 *  decisions on plain objects, with no React and no catalog. */
export interface NativeRowLike {
  model_id?: string | null;
  protocol_id?: string | null;
  metrics?: { r10?: number; r50?: number; mean?: number } | null;
  /** The scale `metrics` is on; needed to print a conflicting number the way its own row prints it. */
  unit?: "fraction" | "percent" | null;
  /** The file this row came from: `conflict_with` names OTHER files by this same path. */
  source?: { path: string } | null;
  /** Files whose number for the same identity disagrees with this one. Non-empty = not settled. */
  conflict_with?: string[] | null;
  /** Statistics of the repeat draws the keeper of a nondeterministic config stands for. */
  variance_group?: {
    metric?: string | null;
    n: number;
    values?: number[];
    mean: number;
    sd?: number | null;
    spread?: number | null;
    note?: string | null;
    insufficient_draws?: boolean;
  } | null;
  /** The draw index a repeat row states about itself (1-based), and the group size when served. */
  review_repeat_index?: number | null;
  review_repeat_count?: number | null;
  /** Set on a repeat row: the file it duplicates. */
  duplicate_of?: string | null;
  /** `artifact` | `review_run` | `rollup:<file>` — ancestry, which decides what a repeat is. */
  source_kind?: string | null;
  /** `false` on a row that states no number of its own (repeat, roll-up, unbacked claim). */
  counted?: boolean;
}

/** One number on the row's own scale, 2 decimals. Local because this module is import-free by
 *  contract (see the file header): the tests compile `scope.ts` alone. */
function recallText(value: number | null | undefined, unit: "fraction" | "percent" | null | undefined,
                    digits = 2): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  if (unit === "fraction") return `${(value * 100).toFixed(digits)}%`;
  if (unit === "percent") return `${value.toFixed(digits)}%`;
  return `${value.toFixed(digits)}`;
}

/** Same scaling WITHOUT a percent sign: sd and spread are distances, not recall values. */
function spreadText(value: number | null | undefined, unit: "fraction" | "percent" | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return (unit === "fraction" ? value * 100 : value).toFixed(2);
}

export interface NativeConflictChip {
  /** The other file's path — the chip's key. */
  key: string;
  file: string;
  /** The other row's value, formatted on ITS scale, ready for the numeric cell. */
  perMetric: { r10: string; r50: string };
  /** Full sentence: both numbers, the other file, and that the value is not settled. */
  label: string;
}

/** R16: a counted row whose number a second file contradicts.
 *
 *  The table printed both rows as plain numbers, so a reader saw two measurements where the backend
 *  says one disagreement. Every chip names BOTH numbers and the file the other one is in, and says
 *  the value is not settled; `perMetric` is what the numeric cell prints beside its own value, so
 *  neither number can be read without the other appearing.
 */
export function nativeConflictChips(row: NativeRowLike, rows: readonly NativeRowLike[]): NativeConflictChip[] {
  const paths = row.conflict_with ?? [];
  return paths.map((path) => {
    const file = path.split("/").pop() ?? path;
    const other = rows.find((peer) => peer.source?.path === path);
    const mine = { r10: recallText(row.metrics?.r10, row.unit), r50: recallText(row.metrics?.r50, row.unit) };
    if (!other || !other.metrics) {
      return {
        key: path, file,
        perMetric: { r10: `≠ số trong ${file}`, r50: `≠ số trong ${file}` },
        label: `Giá trị CHƯA chốt: file ${file} cho một số khác cho cùng mô hình + protocol, `
             + `nhưng dòng đó không có trong catalog nên chưa đối chiếu được hai số.`,
      };
    }
    const theirs = { r10: recallText(other.metrics?.r10, other.unit),
                     r50: recallText(other.metrics?.r50, other.unit) };
    return {
      key: path, file, perMetric: { r10: `≠ ${theirs.r10} (${file})`, r50: `≠ ${theirs.r50} (${file})` },
      label: `Giá trị CHƯA chốt: hai file cùng mô hình + protocol cho hai số khác nhau — `
           + `${mine.r10} (dòng này) vs ${theirs.r10} (${file}). `
           + `Đây là một bất đồng giữa hai nguồn, không phải hai số đo độc lập.`,
    };
  });
}

export type NativeVarianceSummary =
  | { kind: "none" }
  | { kind: "single-draw"; n: number; label: string }
  | { kind: "group"; n: number; mean: number; sd: number | null; spread: number; label: string };

/** R16: what a keeper row's `variance_group` says about the number on screen.
 *
 *  Four draws of one configuration are one measurement with a spread, not four measurements — and
 *  the spread is the only place the evaluator's nondeterminism is visible. `sd`/`spread` are printed
 *  without a percent sign (they are distances); one draw makes no statement about stability at all,
 *  so that case says only that one draw exists.
 */
export function nativeVarianceSummary(row: NativeRowLike): NativeVarianceSummary {
  const group = row.variance_group;
  if (!group || group.n === undefined) return { kind: "none" };
  const metric = (group.metric ?? "r10").replace("r", "R@");
  const n = group.n;
  const sd = group.sd ?? null;
  if (group.insufficient_draws === true || n <= 1 || sd === null
      || group.spread === null || group.spread === undefined) {
    return { kind: "single-draw", n,
      label: `chỉ có 1 draw cho cấu hình này (${metric}) — chưa đủ draw để nói gì về độ ổn định.` };
  }
  const mean = group.mean;
  return { kind: "group", n, mean, sd, spread: group.spread,
    label: `${n} draw của MỘT cấu hình (${metric}): mean ${recallText(mean, row.unit)}, `
         + `sd ${spreadText(sd, row.unit)}, spread ${spreadText(group.spread, row.unit)} — `
         + `evaluator nondeterministic by upstream design, nên con số ở dòng này là một draw, `
         + `không phải hằng số; ${n} draw không phải ${n} số đo.` };
}

/** "repeat 2 of 4" for a repeat row. The group size is served on the KEEPER (`variance_group.n`),
 *  which is the row `duplicate_of` names; `review_repeat_count` is the row's own claim when present. */
export function nativeRepeatLabel(row: NativeRowLike, rows: readonly NativeRowLike[]): string | null {
  if (row.review_repeat_index === null || row.review_repeat_index === undefined) return null;
  const total = row.review_repeat_count
    ?? row.variance_group?.n
    ?? rows.find((peer) => row.duplicate_of && peer.source?.path === row.duplicate_of)?.variance_group?.n
    ?? null;
  return total === null
    ? `repeat ${row.review_repeat_index}`
    : `repeat ${row.review_repeat_index} of ${total}`;
}

/** A native row is a scope fact only when it STATES a number. A blocker document (`metrics: null`)
 *  is a reason, not a measurement, and must not make its model read as measured. */
function statesNumber(row: NativeRowLike): boolean {
  return row.metrics !== null && row.metrics !== undefined;
}

/** R08 at the scope layer: a row the backend marks `counted: false` (a repeat draw, a roll-up that
 *  restates another row, a superseded selection) still STATES a number and is still displayed with
 *  its `duplicate_of` label — but it is not evidence in a scope question. `statesNumber` answers
 *  "is there a number at all"; this answers "may a scope claim it". airknow showed six native rows at
 *  `fashioniq_full_gallery_ref_excluded`, three of them `duplicate_of` repeats, and printed "6 số đo
 *  native"; clvc_net/tgcir's only protocol-less rows were roll-ups, and the inventory told the reader
 *  a number existed whose scope was unknown. */
function counts(row: NativeRowLike): boolean {
  return statesNumber(row) && row.counted !== false;
}

/** Native rows for one model IN one scope.
 *
 *  R06: the inventory consulted native rows only to build links to OTHER scopes, and filtered those
 *  to scopes DIFFERENT from the current one — so a model whose only number was a native macro in
 *  the protocol on screen read "Chưa có số đo" while the row sat in the catalog, and the link
 *  appeared only after switching protocol. Scope resolution is a function of its own, and the
 *  inventory asks it about the ACTIVE scope.
 *
 *  R01 at the UI layer: the ONLY admissible match is an exact `(model_id, protocol_id)`. A row that
 *  states no protocol cannot be attributed to a protocol — it is not evidence about this scope and
 *  lending it to whatever the reader happens to have selected is the same defect as attributing an
 *  unidentified measurement to a protocol. `otherScopeFor` already refuses such rows; this is
 *  consistent with it. Unattributed rows are surfaced separately (`nativeUnattributedFor`) so the
 *  reader still sees the file instead of losing it.
 */
export function nativeRowsFor(rows: readonly NativeRowLike[], model: string | null | undefined,
                              protocol: string): NativeRowLike[] {
  if (!model) return [];
  return rows.filter((row) => counts(row) && row.model_id === model && row.protocol_id === protocol);
}

/** Native rows for one model that state a number but name NO protocol. They are real rows the tree
 *  serves; what they are not is a measurement of any scope, so they never enter `nativeMeasuredIds`
 *  and never render as "measured here". The inventory shows them under their own label. */
export function nativeUnattributedFor(rows: readonly NativeRowLike[], model: string | null | undefined): NativeRowLike[] {
  if (!model) return [];
  return rows.filter((row) => counts(row) && row.model_id === model && !row.protocol_id);
}

/** Models with a native number in EXACTLY `protocol`. A protocol-less row is deliberately excluded:
 *  "measured in this protocol" is a claim a row without a protocol cannot support. */
export function nativeMeasuredIds(rows: readonly NativeRowLike[], protocol: string): Set<string> {
  const ids = new Set<string>();
  for (const row of rows) {
    if (!row.model_id || !counts(row) || row.protocol_id !== protocol) continue;
    ids.add(row.model_id);
  }
  return ids;
}

/** One inventory row's status, from the served facts alone.
 *
 *  The chain is ordered, and the order is the claim: a campaign deferral outranks a run, a run
 *  outranks a row in another table, and "measured" is only ever said about a row that carries a
 *  number. Extracted so the ordering is one function a test can execute rather than a ternary tower
 *  where a later branch can silently contradict the counter above it.
 *
 *  Two contradictions the tower used to have, both now impossible:
 *  * `nativeOnly` was `!entry && nativeInScope > 0`, so two composition runs in this scope plus one
 *    native macro read "Có dữ liệu native" — a label denying the runs. It now requires
 *    `usableCount === 0`.
 *  * every `entry` that was not a composition row was labelled "Có dữ liệu native", including index
 *    and aggregate rows; and an aggregate row with `aggregate_metrics: false` read "Có số đo" while
 *    the measured counter (which requires that flag) excluded it. Both now follow the flag.
 *
 *  A native row that names no protocol is not a measurement of any scope; it gets its own status so
 *  the row is visible without the scope being able to claim it.
 */
export interface InventoryFacts {
  deferred: boolean;
  /** Composition run selected for this row in the active protocol/subset. */
  hasRun: boolean;
  /** `capabilities.flags.aggregate_metrics` on the selected row. */
  runHasAggregateMetrics: boolean;
  runMissingScored: boolean;
  /** Which served table the selected catalog entry came from; null when there is no entry. */
  entryFamily: "composition" | "index" | "aggregate" | null;
  entryHasAggregateMetrics: boolean;
  /** Native rows with a number for (model, this protocol). */
  nativeInScope: number;
  /** Native rows with a number for this model that name NO protocol. */
  nativeUnattributed: number;
  /** Usable composition rows for this model in this scope. */
  usableCount: number;
  /** Usable entries in this scope whose row reports aggregate metrics — i.e. rows the measured
   *  counter ("Có số đo") already counts. A multi-run row that the counter counts must not read
   *  "cần chọn" as if nothing were measured. */
  measuredEntryCount: number;
  elsewhereCount: number;
  checkpointDownloaded: boolean;
}

/** Whether a model with nothing in THIS scope may be told where its numbers actually are.
 *
 *  R06-follow-up: the condition was `entry || deferredReason || nativeOnly || nativeUnattributed.length`
 *  — so a model whose selected protocol holds only protocol-less native rows was left with "a number
 *  exists but its scope is unknown" and never the link to the scope that DOES have its attributed,
 *  counted row. "This row names no protocol" and "this model has numbers in another scope" are
 *  independent facts; neither may erase the other. Only the three reasons that mean the CURRENT scope
 *  already has something to show suppress the link: a selected entry, a campaign deferral, and a
 *  native number in this very scope. `nativeUnattributed` deliberately does not.
 *
 *  The caller still filters `otherScopeFor` to scopes different from the one on screen, and renders
 *  the unattributed label and the links side by side when both apply.
 */
export function crossScopeAllowed(facts: {
  /** A catalog entry selected for this row in the active protocol/subset. */
  hasEntry: boolean;
  /** An operator campaign deferral; the reason is printed on the row instead. */
  deferred: boolean;
  /** Native rows with a number in the ACTIVE protocol — this scope's own answer. */
  nativeOnly: boolean;
}): boolean {
  return !facts.hasEntry && !facts.deferred && !facts.nativeOnly;
}

export function inventoryStatus(facts: InventoryFacts): string {
  if (facts.deferred) return "Tạm gác";
  if (facts.hasRun) {
    if (!facts.runHasAggregateMetrics) return "Chỉ có hạng từng query";
    return facts.runMissingScored ? "Chấm thiếu" : "Có số đo";
  }
  if (facts.entryFamily) {
    if (facts.entryFamily === "composition") return "Chưa có số đo";
    if (!facts.entryHasAggregateMetrics) return "Chỉ có lượt chạy, chưa có số";
    return "Có số đo";
  }
  if (facts.nativeInScope > 0 && facts.usableCount === 0) return "Có dữ liệu native";
  if (facts.nativeUnattributed > 0) return "Số đo native không ghi protocol";
  if (facts.usableCount > 1) {
    return facts.measuredEntryCount > 0
      ? `${facts.usableCount} lượt trong phạm vi, đã có số đo — chọn một`
      : `${facts.usableCount} lượt cần chọn`;
  }
  if (facts.elsewhereCount > 0) return "Có kết quả ở scope khác";
  return facts.checkpointDownloaded ? "Chưa có số đo" : "Chưa có checkpoint xác nhận";
}

/** What the inspector may say about the target and the candidate window ON SCREEN.
 *
 *  R10: the sentence used `Math.min` over all three conditions' ranks, so MM rank 50 + text rank 5
 *  with K=10 explained the MULTIMODAL window with the TEXT condition's #5 and declared the target
 *  "outside Top-10" — a correct number about a different condition. A window belongs to ONE
 *  condition, so the rank quoted is that condition's, and the sentence names it.
 *
 *  It also concluded before the request had an answer: while the window was pending, `shown` was
 *  empty and "outside the window" was printed about a window that did not exist yet. Pending is its
 *  own verdict, and the component prints no conclusion for it.
 *
 *  Lives here (not in `inspector.ts`) so the choice is a pure export the tests can execute.
 */
export type WindowVerdict =
  | { kind: "hidden" }
  | { kind: "pending" }
  | { kind: "unavailable"; reason: string }
  | { kind: "ranked"; conditionId: string; conditionLabel: string; rank: number; insideK: boolean };

export function candidateWindowVerdict(input: {
  /** All three condition readings for the selected query, in protocol order. */
  conditions: ReadonlyArray<{ id: string; rank: number | null }>;
  /** The condition whose candidate window is rendered (the URL's `condition`). */
  viewed: string;
  /** The label that condition is printed under, so the sentence can name whose rank it quotes. */
  viewedLabel: string;
  effectiveK: number;
  /** Rows the export stored for this run; 0 means there is no window to describe at all. */
  depth: number;
  targetInShown: boolean;
  /** `useRegion` status of the candidate request for THIS run+subset+condition+query+K. A region
   *  still holding another key's data is `loading` here by construction of its caller. */
  status: "idle" | "loading" | "ready" | "error";
}): WindowVerdict {
  if (!input.depth || input.targetInShown) return { kind: "hidden" };
  if (input.status === "idle" || input.status === "loading") return { kind: "pending" };
  if (input.status === "error") {
    return { kind: "unavailable", reason: `không tải được cửa sổ Top-${input.effectiveK}` };
  }
  const viewed = input.conditions.find((item) => item.id === input.viewed);
  if (!viewed || viewed.rank === null) {
    return { kind: "unavailable",
             reason: `điều kiện ${input.viewedLabel} chưa có hạng chính xác, nên chưa kết luận được` };
  }
  return { kind: "ranked", conditionId: viewed.id, conditionLabel: input.viewedLabel,
           rank: viewed.rank, insideK: viewed.rank <= input.effectiveK };
}

/** The five independent parts of one result label. Each is a separate sentence because each can be
 *  true while its neighbour is false: a run can be measured and unverified, or verified with no
 *  exported candidates, and a single combined badge cannot say that. */
export interface ResultLabels {
  measurement: string;
  verification: string;
  coverage: string;
  blocker: string;
  samples: string;
}

export interface LabelInput {
  execution_status?: string | null;
  is_failed?: boolean | null;
  aggregate_metrics?: boolean | null;
  per_query_target_ranks?: boolean | null;
  candidate_topk?: boolean | null;
  scored?: number | null;
  total?: number | null;
  parsed_scored?: number | null;
  is_transfer?: boolean | null;
  verified?: boolean | null;
  verification_state?: string | null;
  blocker_reason?: string | null;
  reproduction_status?: string | null;
}

export function resultLabels(input: LabelInput): ResultLabels {
  const failed = input.is_failed === true || (input.execution_status ?? "").toUpperCase() === "FAILED";
  const measured = !failed && input.aggregate_metrics === true;
  const measurement = failed
    ? "Lượt lỗi · không có số đo"
    : measured ? "Đã đo" : "Chỉ có số tổng hợp chưa đủ";
  const verification = input.verified === true
    ? `Đã xác minh${input.verification_state ? ` (${input.verification_state})` : ""}`
    : input.verification_state
      ? `Chưa xác minh (${input.verification_state})`
      : "Chưa xác minh";
  const scored = input.scored ?? null;
  const total = input.total ?? null;
  const coverage = scored != null && total != null
    ? `Đã chấm ${scored.toLocaleString("en-US")}/${total.toLocaleString("en-US")}`
    : "Chưa có mẫu số";
  const blocker = failed
    ? (input.blocker_reason ? `Chặn: ${input.blocker_reason}` : "Chặn: lượt không hoàn tất")
    : input.reproduction_status
      ? `Rerun: ${input.reproduction_status}`
      : "Rerun: chưa ghi blocker";
  const samples = !input.per_query_target_ranks
    ? "Không có hạng từng query"
    : input.candidate_topk
      ? "Có hạng · đã xuất ứng viên"
      : "Có hạng · chưa xuất ứng viên";
  return { measurement, verification, coverage, blocker, samples };
}

/** The subset label a protocol is allowed to print. A protocol without subsets prints nothing,
 *  because "" is a statement ("one cohort") and "Full" would be a false one. */
export function subsetLabelFor(protocolId: string, subset: string, labels: Record<string, string>): string {
  return protocolHasSubsets(protocolId) ? (labels[subset] ?? subset) : "";
}

/** The explorer chip row: the four authoritative query states, in one control.
 *
 *  `query_state` is the backend's own classification (`hit` / `retrieval_failure` / `unscored`), so
 *  the chips filter on the contract rather than on a UI-side re-derivation of it. `all` is not a
 *  state and is spelled as the empty string, which is also what the URL omits. */
export const QUERY_STATE_CHIPS: ReadonlyArray<{ id: string; label: string; hint: string }> = [
  { id: "", label: "Tất cả", hint: "không lọc theo trạng thái" },
  { id: "hit", label: "Correct@10", hint: "hạng đích trong @10" },
  { id: "retrieval_failure", label: "Miss@10", hint: "có hạng thật nhưng ngoài @10" },
  { id: "unscored", label: "Unscored", hint: "không có hạng chính xác (pipeline/ánh xạ)" },
];
