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
  /** The backend's admission state for this artifact. Absent means unverified. */
  evidence_status?: string | null;
}

/** The fields of `/api/benchmark-runs` rows this module reads. */
export interface EvidenceBenchmarkRun {
  run_id?: string | null;
  /** The evaluator's own declaration that this run is a self-test, not an experiment. */
  is_synthetic?: boolean | null;
  data_kind?: RunDataKind | null;
  /** The API's four-question receipt, when the endpoint served it. */
  verification?: {
    certified?: boolean | null;
    is_research_evidence?: boolean | null;
    execution_authenticated?: boolean | null;
    state?: string | null;
  } | null;
}

/** Whether a per-query run may be presented as verified research evidence.
 *
 *  `evidence_status` is computed by the backend's admission policy, which only reports
 *  `PER_QUERY_VERIFIED` when an authoritative verifier corroborated the artifact. Anything else is
 *  an artifact whose numbers nobody has checked, and the UI must say so rather than reading
 *  `data_kind === "experiment"` -- which is what the producer DECLARED about itself.
 */
export function isPerQueryVerified(run: EvidenceRun): boolean {
  return run?.evidence_status === "PER_QUERY_VERIFIED";
}

export interface EvidenceState {
  /** Real runs in the per-query index. */
  experimentRuns: number;
  /** Per-query runs the authoritative verifier corroborated. */
  verifiedQueryRuns: number;
  /** Runs that declare themselves experiments but carry no corroboration. */
  unverifiedExperimentRuns: number;
  /** Labelled development runs in the per-query index. */
  mockRuns: number;
  /** Aggregate benchmark reports served by the API, of any certification state. */
  aggregateRuns: number;
  /** Aggregate reports the API presented as valid research evidence. */
  certifiedAggregates: number;
  /** Aggregate reports whose execution is authenticated but which are not evidence. */
  uncertifiedAggregates: number;
  /** Aggregate reports that declare themselves synthetic or development self-tests. */
  syntheticAggregates: number;
  /** True when a real experiment exists ONLY as an aggregate, with no per-query export. */
  aggregateOnly: boolean;
  /** True when this deployment holds no real evidence at all, at either level. */
  noRealEvidence: boolean;
  /** True when labelled development data is present in the per-query index. */
  hasMockEvidence: boolean;
  /** The per-query level is unavailable while real aggregate evidence exists. */
  queryLevelUnavailable: boolean;
}

/** Whether a served report is a self-test rather than an experiment about the paper. */
function isSyntheticAggregate(run: EvidenceBenchmarkRun): boolean {
  return run?.is_synthetic === true || run?.data_kind === "mock" || run?.data_kind === "synthetic";
}

function isCertifiedAggregate(run: EvidenceBenchmarkRun): boolean {
  return run?.verification?.certified === true && run?.verification?.is_research_evidence === true;
}

export function evidenceState(runs: EvidenceRun[] | null | undefined,
                              benchmarkRuns: EvidenceBenchmarkRun[] | null | undefined): EvidenceState {
  const safeRuns = Array.isArray(runs) ? runs : [];
  const safeBenchmarks = Array.isArray(benchmarkRuns) ? benchmarkRuns : [];

  const experimentRuns = safeRuns.filter((run) => run?.data_kind === "experiment").length;
  // A declared experiment is NOT verified evidence. Counted separately so the banner can say
  // "declared, unverified" instead of implying a checked result.
  const verifiedQueryRuns = safeRuns.filter(isPerQueryVerified).length;
  const unverifiedExperimentRuns = experimentRuns - verifiedQueryRuns;
  const mockRuns = safeRuns.filter((run) => run?.data_kind === "mock").length;
  const aggregateRuns = safeBenchmarks.length;
  const certifiedAggregates = safeBenchmarks.filter(isCertifiedAggregate).length;
  const syntheticAggregates = safeBenchmarks.filter(isSyntheticAggregate).length;
  const uncertifiedAggregates = aggregateRuns - certifiedAggregates;
  // "Aggregate-only" is a statement about the per-query export, not about certification: a
  // refusal is still a real thing that happened, and it must not be reported as absent.
  const aggregateOnly = aggregateRuns > 0 && experimentRuns === 0;
  return {
    experimentRuns,
    verifiedQueryRuns,
    unverifiedExperimentRuns,
    mockRuns,
    aggregateRuns,
    certifiedAggregates,
    uncertifiedAggregates,
    syntheticAggregates,
    aggregateOnly,
    noRealEvidence: aggregateRuns === 0 && experimentRuns === 0,
    hasMockEvidence: mockRuns > 0,
    queryLevelUnavailable: aggregateOnly,
  };
}

/** The header label for a state, or "" when nothing needs saying.
 *
 *  Precedence: the most severe true statement about the deployment wins, and no branch may
 *  assert the presence of real evidence it does not have, nor the absence of evidence that
 *  exists. A synthetic or refused aggregate is never described as a real evaluation (I03/I14/I16).
 */
export function evidenceBanner(state: EvidenceState): string {
  if (state.noRealEvidence) {
    return state.hasMockEvidence
      ? "MOCK DEVELOPMENT DATA — only labelled development runs are loaded; no real evaluation results"
      : "MOCK DEVELOPMENT DATA — no real evaluation results are loaded";
  }
  const onlySynthetic = state.syntheticAggregates === state.aggregateRuns && state.experimentRuns === 0;
  if (onlySynthetic) {
    return `MOCK DEVELOPMENT DATA — ${state.syntheticAggregates} synthetic aggregate `
      + `run${state.syntheticAggregates === 1 ? "" : "s"} loaded; no real evaluation results`;
  }
  if (state.aggregateOnly) {
    // "REAL EVALUATION" is only earned by evidence that is neither synthetic nor refused.
    const head = state.certifiedAggregates > 0
      ? `AGGREGATE-ONLY REAL EVALUATION — ${state.certifiedAggregates} certified research `
        + `result${state.certifiedAggregates === 1 ? "" : "s"}`
      : `AGGREGATE-ONLY RESULTS — ${state.uncertifiedAggregates} uncertified aggregate `
        + `run${state.uncertifiedAggregates === 1 ? "" : "s"}, no certified research evidence`;
    const trailer = state.uncertifiedAggregates > 0 && state.certifiedAggregates > 0
      ? ` (${state.uncertifiedAggregates} further aggregate `
        + `run${state.uncertifiedAggregates === 1 ? "" : "s"} uncertified)`
      : "";
    return `${head}${trailer}; per-query Top-K analysis unavailable`
      + (state.hasMockEvidence
        ? " (development runs also loaded and excluded from research conclusions)" : "");
  }
  if (state.unverifiedExperimentRuns > 0) {
    // The strongest true statement available: these runs call themselves experiments and nobody has
    // corroborated them. Saying nothing (the old behaviour) presented them as checked evidence.
    return `DECLARED EXPERIMENT (UNVERIFIED) — ${state.unverifiedExperimentRuns} per-query `
      + `run${state.unverifiedExperimentRuns === 1 ? "" : "s"} declare themselves experiments but `
      + `carry no verifier corroboration; they are not research evidence`
      + (state.verifiedQueryRuns > 0
        ? ` (${state.verifiedQueryRuns} verified run${state.verifiedQueryRuns === 1 ? "" : "s"} also loaded)`
        : "")
      // The aggregate counters are a separate truth and must survive this branch: the dashboard
      // says those runs exist, so the banner may not omit them.
      + (state.uncertifiedAggregates > 0
        ? `; ${state.uncertifiedAggregates} further aggregate `
          + `run${state.uncertifiedAggregates === 1 ? "" : "s"} not certified research evidence`
        : "");
  }
  if (state.hasMockEvidence) {
    return "Mock and real experiment runs are loaded. Mock results are excluded from research conclusions.";
  }
  // Per-query real results exist, so the mock alarm is not the right one -- but an uncertified
  // aggregate must still be announced rather than silently suppressed (I17's UI half).
  if (state.uncertifiedAggregates > 0) {
    return `Real evaluation present (${state.experimentRuns} indexed); `
      + `${state.uncertifiedAggregates} aggregate run${state.uncertifiedAggregates === 1 ? "" : "s"} `
      + `not certified research evidence`
      + (state.syntheticAggregates > 0
        ? ` (${state.syntheticAggregates} synthetic)` : ".");
  }
  return "";
}
