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
export function isCertifiedEvidence(run: { verification?: EvidenceBenchmarkRun["verification"] | null } | null | undefined): boolean {
  const receipt = run?.verification;
  if (receipt?.certified !== true || receipt?.is_research_evidence !== true) return false;
  return receipt?.development_mode !== true && receipt?.certification_mode !== "development";
}

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
    /** The policy that judged this report. `development` means the completion proof was NOT
     *  validated, so the receipt is an execution-level answer and not a research certification. */
    certification_mode?: string | null;
    development_mode?: boolean | null;
  } | null;
}

/** The fields of `/api/composition/runs` rows this module reads. */
export interface EvidenceCompositionRun {
  run_id?: string | null;
  protocol_id?: string | null;
  subset?: string | null;
  is_synthetic?: boolean | null;
  capabilities?: { candidate_topk?: boolean | null } | null;
}

export type EvidenceLane = "overview" | "experiments" | "evaluation-composition" |
  "evaluation-benchmark" | "evaluation-native" | "samples" | "failures" | "notebook";

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

/** What the shell knows about each source's own request state.
 *
 *  `evidenceState` used to infer "no real evidence" from an EMPTY ARRAY, which is also what a region
 *  holds before its first answer arrives. The banner therefore announced "MOCK DEVELOPMENT DATA — no
 *  real evaluation results are loaded" over data that had not been fetched yet. The fix is not a
 *  guess at a default: the caller states each source's status, and any unsettled source makes the
 *  banner a LOADING statement rather than a claim about content.
 */
export type EvidenceSourceStatus = "idle" | "loading" | "ready" | "error";

export interface EvidenceSources {
  runs?: EvidenceSourceStatus;
  benchmarks?: EvidenceSourceStatus;
  composition?: EvidenceSourceStatus;
}

export interface EvidenceState {
  /** True while any source that feeds this state has not settled. When true, every count below is a
   *  lower bound read from whatever arrived, and the banner must not state an absence. */
  loading: boolean;
  /** Sources that failed to load. A failed source is also not evidence of absence. */
  failedSources: string[];
  /** Sources still in flight. */
  pendingSources: string[];
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
  /** Composition rows served for this deployment, and how many really carry candidate Top-K.
   *  The banner may not call Top-K unavailable while these rows have it: the aggregate endpoints
   *  report the per-query export level, which is not the same question as what the rows on screen
   *  can show. */
  compositionRows: number;
  rowsWithCandidateTopk: number;
}

/** Whether a served report is a self-test rather than an experiment about the paper. */
function isSyntheticAggregate(run: EvidenceBenchmarkRun): boolean {
  return run?.is_synthetic === true || run?.data_kind === "mock" || run?.data_kind === "synthetic";
}


/** The scope the composition table is showing. The banner must count inside it, not deployment-wide,
 *  or it states a number the reader cannot reconcile with the rows directly beneath it. */
export interface CompositionScope {
  protocolId?: string | null;
  subset?: string | null;
}

export function evidenceState(runs: EvidenceRun[] | null | undefined,
                              benchmarkRuns: EvidenceBenchmarkRun[] | null | undefined,
                              compositionRuns?: EvidenceCompositionRun[] | null,
                              scope?: CompositionScope | null,
                              sources?: EvidenceSources | null,
                              lane: EvidenceLane = "overview"): EvidenceState {
  const allRuns = Array.isArray(runs) ? runs : [];
  const allBenchmarks = Array.isArray(benchmarkRuns) ? benchmarkRuns : [];
  const allComposition = Array.isArray(compositionRuns) ? compositionRuns : [];
  const compositionLane = ["evaluation-composition", "samples", "failures", "notebook"].includes(lane);
  const benchmarkLane = lane === "evaluation-benchmark";
  const experimentsLane = lane === "experiments";
  const overviewLane = lane === "overview";
  const safeRuns = overviewLane || experimentsLane ? allRuns : [];
  const safeBenchmarks = overviewLane || benchmarkLane ? allBenchmarks : [];
  const safeComposition = overviewLane || compositionLane ? allComposition : [];
  const activeSources = overviewLane
    ? (sources ?? {})
    : experimentsLane ? { runs: sources?.runs }
      : benchmarkLane ? { benchmarks: sources?.benchmarks }
        : compositionLane ? { composition: sources?.composition }
          : {};
  const pendingSources = (Object.entries(activeSources) as Array<[keyof EvidenceSources, EvidenceSourceStatus | undefined]>)
    .filter(([, status]) => status === "idle" || status === "loading")
    .map(([name]) => name);
  const failedSources = (Object.entries(activeSources) as Array<[keyof EvidenceSources, EvidenceSourceStatus | undefined]>)
    .filter(([, status]) => status === "error")
    .map(([name]) => name);
  const loading = pendingSources.length > 0;
  const experimentRuns = safeRuns.filter((run) => run?.data_kind === "experiment").length;
  const verifiedQueryRuns = safeRuns.filter(isPerQueryVerified).length;
  const unverifiedExperimentRuns = experimentRuns - verifiedQueryRuns;
  const indexMockRuns = safeRuns.filter((run) => run?.data_kind === "mock").length;
  const aggregateRuns = safeBenchmarks.length;
  const certifiedAggregates = safeBenchmarks.filter(isCertifiedEvidence).length;
  const syntheticAggregates = safeBenchmarks.filter(isSyntheticAggregate).length;
  const uncertifiedAggregates = aggregateRuns - certifiedAggregates;
  const aggregateOnly = aggregateRuns > 0 && experimentRuns === 0;
  const scoped = (scope?.protocolId || scope?.subset)
    ? safeComposition.filter((run) =>
        (!scope?.protocolId || run?.protocol_id === scope.protocolId)
        && (!scope?.subset || run?.subset === scope.subset))
    : safeComposition;
  const scopedSynthetic = scoped.filter((run) => run?.is_synthetic === true).length;
  const scopedReal = scoped.length - scopedSynthetic;
  const mockRuns = indexMockRuns + (compositionLane ? scopedSynthetic : 0);
  const compositionRealEvidence = compositionLane && scopedReal > 0;
  const noRealEvidence = aggregateRuns === 0 && experimentRuns === 0 && !compositionRealEvidence;
  const rowsWithCandidateTopk = scoped
    .filter((run) => run?.capabilities?.candidate_topk === true).length;
  return {
    loading,
    failedSources,
    pendingSources,
    experimentRuns,
    verifiedQueryRuns,
    unverifiedExperimentRuns,
    mockRuns,
    aggregateRuns,
    certifiedAggregates,
    uncertifiedAggregates,
    syntheticAggregates,
    aggregateOnly,
    noRealEvidence,
    hasMockEvidence: mockRuns > 0,
    queryLevelUnavailable: aggregateOnly,
    compositionRows: scoped.length,
    rowsWithCandidateTopk,
  };
}

/** The header label for a state, or "" when nothing needs saying.
 *
 *  Precedence: the most severe true statement about the deployment wins, and no branch may
 *  assert the presence of real evidence it does not have, nor the absence of evidence that
 *  exists. A synthetic or refused aggregate is never described as a real evaluation (I03/I14/I16).
 */
export function evidenceBanner(state: EvidenceState): string {
  if (state.loading || state.failedSources.length > 0) {
    const parts: string[] = [];
    if (state.loading) parts.push("Đang tải bằng chứng");
    if (state.failedSources.length) {
      parts.push(`Không tải được ${state.failedSources.join(", ")} — đây là lỗi yêu cầu, `
        + "không phải 'không có kết quả'");
    }
    const arrived = state.experimentRuns + state.aggregateRuns;
    if (arrived > 0) {
      parts.push(`${state.certifiedAggregates} đã chứng nhận trong ${state.aggregateRuns} report đã về`);
    }
    return parts.join(" · ");
  }
  if (state.compositionRows > 0 && state.noRealEvidence) {
    return "MOCK DEVELOPMENT DATA — only synthetic composition runs are loaded";
  }
  if (state.compositionRows > 0 && !state.noRealEvidence
      && state.experimentRuns === 0 && state.aggregateRuns === 0) return "";
  if (state.noRealEvidence) {
    return state.hasMockEvidence
      ? "MOCK DEVELOPMENT DATA — only labelled development runs are loaded; no real evaluation results"
      : "MOCK DEVELOPMENT DATA — no real evaluation results are loaded";
  }
  const onlySynthetic = state.syntheticAggregates === state.aggregateRuns
    && state.experimentRuns === 0 && state.compositionRows === 0;
  if (onlySynthetic) {
    return `MOCK DEVELOPMENT DATA — ${state.syntheticAggregates} synthetic aggregate `
      + `run${state.syntheticAggregates === 1 ? "" : "s"} loaded; no real evaluation results`;
  }
  if (state.aggregateOnly) {
    const head = state.certifiedAggregates > 0
      ? `AGGREGATE-ONLY REAL EVALUATION — ${state.certifiedAggregates} certified research `
        + `result${state.certifiedAggregates === 1 ? "" : "s"}`
      : `AGGREGATE-ONLY RESULTS — ${state.uncertifiedAggregates} uncertified aggregate `
        + `run${state.uncertifiedAggregates === 1 ? "" : "s"}, no certified research evidence`;
    const trailer = state.uncertifiedAggregates > 0 && state.certifiedAggregates > 0
      ? ` (${state.uncertifiedAggregates} further aggregate `
        + `run${state.uncertifiedAggregates === 1 ? "" : "s"} uncertified)`
      : "";
    const topk = state.rowsWithCandidateTopk > 0
      ? `; per-query candidate Top-K available on ${state.rowsWithCandidateTopk} of `
        + `${state.compositionRows} composed run${state.compositionRows === 1 ? "" : "s"} in view`
      : "; per-query Top-K analysis unavailable";
    return `${head}${trailer}${topk}`
      + (state.hasMockEvidence
        ? " (development runs also loaded and excluded from research conclusions)" : "");
  }
  if (state.unverifiedExperimentRuns > 0) {
    return `DECLARED EXPERIMENT (UNVERIFIED) — ${state.unverifiedExperimentRuns} per-query `
      + `run${state.unverifiedExperimentRuns === 1 ? "" : "s"} declare themselves experiments but `
      + `carry no verifier corroboration; they are not research evidence`
      + (state.verifiedQueryRuns > 0
        ? ` (${state.verifiedQueryRuns} verified run${state.verifiedQueryRuns === 1 ? "" : "s"} also loaded)`
        : "")
      + (state.uncertifiedAggregates > 0
        ? `; ${state.uncertifiedAggregates} further aggregate `
          + `run${state.uncertifiedAggregates === 1 ? "" : "s"} not certified research evidence`
        : "");
  }
  if (state.hasMockEvidence) {
    return "Mock and real experiment runs are loaded. Mock results are excluded from research conclusions.";
  }
  if (state.uncertifiedAggregates > 0) {
    return `Real evaluation present (${state.experimentRuns} indexed); `
      + `${state.uncertifiedAggregates} aggregate run${state.uncertifiedAggregates === 1 ? "" : "s"} `
      + `not certified research evidence`
      + (state.syntheticAggregates > 0 ? ` (${state.syntheticAggregates} synthetic)` : ".");
  }
  return "";
}
