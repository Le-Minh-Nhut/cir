/** One candidate-list request, identified by everything that can change its answer.
 *
 *  The old inspector sliced `query.top_results`, which the page endpoint had already cut to its own
 *  `top_k` (default 10). Choosing "Top 50" then showed 10 rows forever, because no request was made
 *  and a 10-element array cannot be sliced into 50. Here K is part of the request identity, so a new
 *  K is a new fetch — and the region key (run/subset/condition/query/K) is what makes a late answer
 *  for an old K unable to overwrite the current one.
 */
export interface CandidateRequest {
  runId: string;
  subset: string;
  condition: string;
  queryId: string;
  k: number;
}

export function candidateRequestPath(input: CandidateRequest): string {
  const params = new URLSearchParams({
    subset: input.subset, condition: input.condition, query_id: input.queryId,
    top_k: String(input.k), limit: "1", offset: "0",
  });
  return `/api/composition/runs/${encodeURIComponent(input.runId)}/queries?${params}`;
}

/** The display sizes the served export can actually honour, plus the depth itself.
 *
 *  A menu may not offer a number the artifact cannot produce: an export of 50 renders Top 100 as a
 *  promise the data breaks, and an empty menu is the honest answer when no depth is known.
 */
export function candidateDisplayOptions(depth: number | null): number[] {
  if (!depth || depth < 1) return [];
  return [...new Set([1, 5, 10, 20, 50, 100].filter((n) => n <= depth).concat(Math.min(depth, 100)))]
    .sort((a, b) => a - b);
}

/** An explicit deep link must never silently display a different sample.
 *  Exact match on `query_id`, never `items[0]`: a query on another page, or one absent from this
 *  run, is reported as missing rather than replaced by whatever row happened to be first. */
export function selectExactQuery<T extends { query_id: string }>(items: T[], queryId: string): T | null {
  return items.find((item) => item.query_id === queryId) ?? null;
}

/** Bounded display K for a request: never more rows than were exported. */
export function effectiveCandidateK(k: number, depth: number): number {
  return depth > 0 ? Math.min(Math.max(1, k), depth, 100) : 10;
}
