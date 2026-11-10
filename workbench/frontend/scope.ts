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
