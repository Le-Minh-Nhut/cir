/** Per-query analysis across several runs of the same protocol and subset.
 *
 *  Two rules this module exists to enforce, both of which the previous UI broke:
 *
 *  * **Unscored is not a failure.** A query with no exact rank in a condition is a pipeline /
 *    mapping fact (`neither_scored`). It has no rank, so it cannot be "the worst" — floating null
 *    ranks to the top of the Worst sort put 119 unscored queries above rank 70,016 failures and made
 *    the failure list meaningless. Unscored is its own bucket, with its own counter and filter.
 *
 *  * **The cutoff is the protocol's.** Outcome classification is fixed at @10 by the release
 *    (`determine_query_outcome(k=10)`). `k` here is the *displayed candidate depth* and, where a
 *    control offers it, the consensus cutoff — the two are named separately so a selector cannot
 *    pretend to change the protocol.
 *
 *  Everything is derived from the rows the API returns, so the cohort is exactly the intersection
 *  of the runs' query universes and its size is published with the number.
 */

export type Condition = "multimodal" | "text_only_black_image" | "image_only_empty_text";

export interface AnalysisQueryRow {
  query_id: string;
  category: string | null;
  query_txt: string;
  reference_id: string;
  positive_ids: string[];
  ranks: Record<string, number | null>;
  target_rank: number | null;
  outcome: string;
  failure?: string | null;
}

export interface RunPage {
  runKey: string;
  runLabel: string;
  items: AnalysisQueryRow[];
}

export interface ConsensusRow {
  query_id: string;
  category: string | null;
  query_txt: string;
  reference_id: string;
  positive_ids: string[];
  /** rank per run key; null means "this run has no rank for this query" */
  ranks: Record<string, number | null>;
  /** ranks per condition, only for runs that have it */
  conditionRanks: Record<Condition, Record<string, number | null>>;
  /** fraction of the scored cohort whose rank exceeds `k` */
  failFraction: number;
  minRank: number | null;
  maxRank: number | null;
  medianRank: number | null;
  rankRange: number | null;
  /** Runs that produced a rank for this query. */
  scoredRuns: number;
  /** Runs in the cohort that did not score it: the "unscored" bucket. */
  unscoredRuns: number;
  /** Outcome as classified at the protocol's fixed @10 cutoff. */
  outcome: string;
  unscored: boolean;
}

export interface AnalysisScope {
  rows: ConsensusRow[];
  /** Queries present in every selected run. */
  cohort: number;
  /** Queries present in at least one run but not all — excluded from consensus, counted. */
  partial: number;
  /** Queries with no exact rank anywhere in this selection. */
  unscored: number;
  runs: number;
}

function median(values: number[]): number | null {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Join the pages by query id.
 *
 *  The cohort is the INTERSECTION of the runs' query ids: comparing models on different query sets
 *  would publish a number whose denominator nobody stated. Queries outside the intersection are
 *  counted in `partial`, not silently dropped.
 */
export function buildScope(pages: RunPage[]): AnalysisScope {
  // Keyed by run as well as query: a run whose page lacks the query must not inherit another
  // run's row, or the per-condition columns would silently show the wrong run's ranks.
  const ids = new Map<string, Map<string, AnalysisQueryRow>>();
  for (const page of pages) {
    for (const item of page.items) {
      const bucket = ids.get(item.query_id);
      if (bucket) bucket.set(page.runKey, item);
      else ids.set(item.query_id, new Map([[page.runKey, item]]));
    }
  }
  const rows: ConsensusRow[] = [];
  let partial = 0;
  let unscoredCount = 0;
  for (const [queryId, byRun] of ids) {
    if (byRun.size !== pages.length) partial += 1;
    const ranks: Record<string, number | null> = {};
    for (const page of pages) ranks[page.runKey] = byRun.get(page.runKey)?.target_rank ?? null;
    const scored = Object.values(ranks).filter((rank): rank is number => typeof rank === "number");
    if (!scored.length) unscoredCount += 1;
    const conditionRanks = { multimodal: {}, text_only_black_image: {}, image_only_empty_text: {} } as Record<Condition, Record<string, number | null>>;
    for (const page of pages) {
      const copy = byRun.get(page.runKey);
      for (const condition of Object.keys(conditionRanks) as Condition[]) {
        conditionRanks[condition][page.runKey] = copy?.ranks?.[condition] ?? null;
      }
    }
    const first = [...byRun.values()][0];
    rows.push({
      query_id: queryId,
      category: first.category,
      query_txt: first.query_txt,
      reference_id: first.reference_id,
      positive_ids: first.positive_ids,
      ranks,
      conditionRanks,
      failFraction: 0, // filled below: needs k
      minRank: scored.length ? Math.min(...scored) : null,
      maxRank: scored.length ? Math.max(...scored) : null,
      medianRank: median(scored),
      rankRange: scored.length ? Math.max(...scored) - Math.min(...scored) : null,
      scoredRuns: scored.length,
      unscoredRuns: pages.length - scored.length,
      outcome: first.outcome,
      unscored: !scored.length,
    });
  }
  return { rows, cohort: rows.filter((row) => row.unscoredRuns === 0).length, partial, unscored: unscoredCount, runs: pages.length };
}

/** Recompute the fail fraction at a stated cutoff. `k` here is a real choice (consensus), not the
 *  protocol's classification, and it is rendered next to the column header. */
export function withCutoff(scope: AnalysisScope, k: number): AnalysisScope {
  const rows = scope.rows.map((row) => {
    const scored = Object.values(row.ranks).filter((rank): rank is number => typeof rank === "number");
    const failing = scored.filter((rank) => rank > k).length;
    return { ...row, failFraction: scored.length ? failing / scored.length : 0 };
  });
  return { ...scope, rows };
}

export const PRESETS = [
  { id: "all", label: "Tất cả query" },
  { id: "allfail", label: "Mọi run đều trượt" },
  { id: "80", label: "≥80% run trượt" },
  { id: "50", label: "≥50% run trượt" },
  { id: "winner", label: "Có run đạt trong @K" },
  { id: "easy", label: "Mọi run đạt trong @K" },
] as const;
export type PresetId = (typeof PRESETS)[number]["id"];

export interface FilterInput {
  preset: PresetId;
  k: number;
  category: string;
  outcome: string;
  /** "" = all; "scored" hides unscored; "unscored" shows only unscored. */
  scoring: "" | "scored" | "unscored";
  query?: string;
}

/** The filter that decides what "saved" means. Kept pure and separate from rendering so the save
 *  action saves exactly the published set — never just the current page. */
export function filterRows(scope: AnalysisScope, filter: FilterInput): ConsensusRow[] {
  return scope.rows.filter((row) => {
    if (filter.query && !row.query_id.includes(filter.query.trim())) return false;
    if (filter.category && row.category !== filter.category) return false;
    if (filter.outcome && row.outcome !== filter.outcome) return false;
    if (filter.scoring === "scored" && row.unscored) return false;
    if (filter.scoring === "unscored" && !row.unscored) return false;
    if (filter.preset === "all") return true;
    // Every preset is about the SCORED cohort; an unscored row has no rank to fail with, so it is
    // not a consensus failure and must not be recruited into one.
    if (row.unscored) return filter.scoring === "unscored";
    switch (filter.preset) {
      case "allfail": return row.failFraction === 1;
      case "80": return row.failFraction >= 0.8;
      case "50": return row.failFraction >= 0.5;
      case "winner": return row.minRank !== null && row.minRank <= filter.k;
      case "easy": return row.maxRank !== null && row.maxRank <= filter.k;
      default: return true;
    }
  });
}

export type SortKey = "worst" | "best" | "consensus" | "range" | "query";

/** Ordering. Unscored rows sort to the END for every rank-based order, in both directions: they
 *  have no rank, so "worst first" cannot mean them. */
export function sortRows(rows: ConsensusRow[], key: SortKey): ConsensusRow[] {
  const rank = (row: ConsensusRow, field: "minRank" | "maxRank" | "rankRange"): number | null => row[field];
  const compareRank = (a: ConsensusRow, b: ConsensusRow, field: "minRank" | "maxRank" | "rankRange", dir: 1 | -1) => {
    const left = rank(a, field);
    const right = rank(b, field);
    if (left === null && right === null) return 0;
    if (left === null) return 1;
    if (right === null) return -1;
    return (left - right) * dir;
  };
  const copy = [...rows];
  copy.sort((a, b) => {
    if (a.unscored !== b.unscored) return a.unscored ? 1 : -1;
    switch (key) {
      case "worst": return compareRank(a, b, "maxRank", -1);
      case "best": return compareRank(a, b, "minRank", 1);
      case "range": return compareRank(a, b, "rankRange", -1);
      case "consensus": return b.failFraction - a.failFraction || compareRank(a, b, "maxRank", -1);
      default: return a.query_id.localeCompare(b.query_id);
    }
  });
  return copy;
}

export interface Page<T> { items: T[]; offset: number; limit: number; total: number }

export function paginate<T>(rows: T[], offset: number, limit: number): Page<T> {
  return { items: rows.slice(offset, offset + limit), offset, limit, total: rows.length };
}
