/** What the header may say about the evidence this deployment actually holds.
 *
 *  Two independent sources answer different questions and neither implies the other:
 *
 *  * `/api/runs` is the DuckDB index of PER-QUERY results. A run there has `data_kind`
 *    (`mock` for development data, `experiment` for a real evaluation).
 *  * `/api/benchmark-runs` is the AGGREGATE report list -- the official macro result. An
 *    aggregate can exist for a run whose per-query export was never saved, which is the normal
 *    case after a benchmark run.
 *
 *  The banner used to be derived from `/api/runs` alone. A legitimate aggregate-only LIMN
 *  evaluation therefore coexisted with an empty query-level index, and the header announced
 *  "no real evaluation results are loaded" over a certified real benchmark -- a false statement
 *  about the deployment's own evidence. This module keeps the two sources separate and names
 *  every state the operator can actually be in, so the label never claims more (or less) than
 *  the API reported. It is pure so the states are testable without a browser.
 */

export type RunDataKind = "experiment" | "mock" | "synthetic" | string;

/** The fields of `/api/runs` rows this module reads. */
export interface EvidenceRun {
  run_id?: string | null;
  data_kind?: RunDataKind | null;
}

/** The fields of `/api/benchmark-runs` rows this module reads. */
export interface EvidenceBenchmarkRun {
  run_id?: string | null;
  /** The API's four-question receipt, when the endpoint served it. */
  verification?: {
    certified?: boolean | null;
    is_research_evidence?: boolean | null;
    execution_authenticated?: boolean | null;
    state?: string | null;
  } | null;
}

export interface EvidenceState {
  /** Real runs in the per-query index. */
  experimentRuns: number;
  /** Labelled development runs in the per-query index. */
  mockRuns: number;
  /** Aggregate benchmark reports served by the API, of any certification state. */
  aggregateRuns: number;
  /** Aggregate reports the API presented as valid research evidence. */
  certifiedAggregates: number;
  /** Aggregate reports whose execution is authenticated but which are not evidence. */
  uncertifiedAggregates: number;
  /** True when a real experiment exists ONLY as an aggregate, with no per-query export. */
  aggregateOnly: boolean;
  /** True when this deployment holds no real evidence at all, at either level. */
  noRealEvidence: boolean;
  /** True when labelled development data is present in the per-query index. */
  hasMockEvidence: boolean;
  /** The per-query level is unavailable while real aggregate evidence exists. */
  queryLevelUnavailable: boolean;
}

export function evidenceState(runs: EvidenceRun[] | null | undefined,
                              benchmarkRuns: EvidenceBenchmarkRun[] | null | undefined): EvidenceState {
  const safeRuns = Array.isArray(runs) ? runs : [];
  const safeBenchmarks = Array.isArray(benchmarkRuns) ? benchmarkRuns : [];

  const experimentRuns = safeRuns.filter((run) => run?.data_kind === "experiment").length;
  const mockRuns = safeRuns.filter((run) => run?.data_kind === "mock").length;
  const aggregateRuns = safeBenchmarks.length;
  const certifiedAggregates = safeBenchmarks.filter(
    (run) => run?.verification?.certified === true
      && run?.verification?.is_research_evidence === true).length;
  const uncertifiedAggregates = safeBenchmarks.filter(
    (run) => !(run?.verification?.certified === true
      && run?.verification?.is_research_evidence === true)).length;
  // "Aggregate-only" is a statement about the per-query export, not about certification: a
  // refusal is still a real thing that happened, and it must not be reported as absent.
  const aggregateOnly = aggregateRuns > 0 && experimentRuns === 0;
  return {
    experimentRuns,
    mockRuns,
    aggregateRuns,
    certifiedAggregates,
    uncertifiedAggregates,
    aggregateOnly,
    noRealEvidence: aggregateRuns === 0 && experimentRuns === 0,
    hasMockEvidence: mockRuns > 0,
    queryLevelUnavailable: aggregateOnly,
  };
}

/** The header label for a state, or "" when nothing needs saying.
 *
 *  Precedence: the most severe true statement about the deployment wins, and no branch may
 *  assert the absence of real evidence while an aggregate report exists.
 */
export function evidenceBanner(state: EvidenceState): string {
  if (state.noRealEvidence) {
    return state.hasMockEvidence
      ? "MOCK DEVELOPMENT DATA — only labelled development runs are loaded; no real evaluation results"
      : "MOCK DEVELOPMENT DATA — no real evaluation results are loaded";
  }
  if (state.aggregateOnly) {
    const certification = state.certifiedAggregates > 0
      ? `${state.certifiedAggregates} certified research result${state.certifiedAggregates === 1 ? "" : "s"}`
      : (state.uncertifiedAggregates > 0
        ? `${state.uncertifiedAggregates} uncertified aggregate result${state.uncertifiedAggregates === 1 ? "" : "s"}`
        : `${state.aggregateRuns} aggregate result${state.aggregateRuns === 1 ? "" : "s"}`);
    return `AGGREGATE-ONLY REAL EVALUATION — ${certification}; per-query Top-K analysis unavailable`
      + (state.hasMockEvidence
        ? " (development runs also loaded and excluded from research conclusions)" : "");
  }
  if (state.hasMockEvidence) {
    return "Mock and real experiment runs are loaded. Mock results are excluded from research conclusions.";
  }
  if (state.aggregateRuns > 0 && state.uncertifiedAggregates > 0 && state.certifiedAggregates === 0) {
    return `Real evaluation present (${state.experimentRuns} indexed) with only uncertified aggregate reports.`;
  }
  return "";
}
