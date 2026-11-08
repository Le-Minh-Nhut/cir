/** What a served run can actually answer, and what it must say when it cannot.
 *
 *  Every contradiction this module removes came from one mistake: a single boolean (or a global
 *  string like "aggregate-only") standing in for a set of independent questions. A run can hold
 *  aggregate metrics and no per-query ranks, or ranks and no candidate list, or ranks whose images
 *  are not installed here. So capabilities are separate flags, and — per the backend contract in
 *  `workbench/docs/reviews/frontend-backend-requests.md` — each absent flag carries a NAMED status
 *  plus a reason. A false flag with a status is a capability the run does not have; it is not an
 *  error and not a broken run.
 *
 *  The served fields (`capabilities`, `capability_detail`, `measurement`, `transfer`,
 *  `duplicate_status`) are authoritative and read directly. The `*Derived` helpers exist only for
 *  rows the backend has not annotated yet (the older per-query index payload) and are marked as
 *  fallbacks in the UI, so a derived number is never mistaken for a measured one.
 *
 *  Nothing here matches on a model NAME. Synthetic, development and refused artifacts are
 *  separated by declared provenance (`is_synthetic`, `synthetic_fixture`, `data_kind`,
 *  `verification.state`), because a name is not a schema and a rename is not a provenance change.
 *
 *  Pure: executed by `workbench/tests/test_frontend_catalog.py` through node.
 */

import type {
  CapabilityDetail, CapabilityFlags, CompositionRun, DuplicateStatus, Measurement, TransferInfo,
} from "./types";

export type { CapabilityFlags } from "./types";

export interface CapabilityReport {
  flags: CapabilityFlags;
  /** Why a false flag is false, per flag. A disabled control names this. */
  reasons: Partial<Record<keyof CapabilityFlags, string>>;
  /** The backend's own status word, so the UI shows its vocabulary rather than inventing one. */
  statuses?: Partial<Record<keyof CapabilityFlags, string>>;
  /** True when these flags were derived here rather than served; the UI labels such a row. */
  derived?: boolean;
}

export interface CompositionRunLike {
  run_id: string;
  model_id: string;
  checkpoint_id: string;
  protocol_id: string;
  subset?: string;
  execution_status?: string;
  is_failed?: boolean | null;
  is_synthetic?: boolean | null;
  synthetic_fixture?: boolean | null;
  checkpoint_transfer?: unknown;
  checkpoint_training_dataset?: string | null;
  training_or_transfer_label?: string | null;
  reproduction_status?: string | null;
  /** Backend-derived contraction per run: `einsum` | `einsum-optimized` | `matmul`. */
  scoring_impl?: { impl?: string | null } | null;
  checkpoint_identity?: string | null;
  gallery_count?: number | null;
  query_count?: number | null;
  conditions?: string[];
  condition_status?: Record<string, { status?: string; unscored_count?: number; reason?: string }> | null;
  metrics?: CompositionRun["metrics"];
  capabilities?: CapabilityFlags | null;
  capability_detail?: Partial<Record<keyof CapabilityFlags, CapabilityDetail>> | null;
  measurement?: Measurement | null;
  transfer?: TransferInfo | null;
  duplicate_status?: DuplicateStatus | null;
  duplicate_attempt_count?: number | null;
  measurement_identity_confidence?: "full" | "partial" | null;
  measurement_identity_missing_inputs?: string[] | null;
  provenance_source?: string | null;
}

export interface BenchmarkRunLike {
  run_id: string;
  model_id?: string | null;
  checkpoint_id?: string | null;
  protocol_id?: string | null;
  is_synthetic?: boolean | null;
  data_kind?: string | null;
  per_query_state?: string | null;
  per_query_capability?: string | null;
  observed_metrics?: Record<string, number | null> | null;
  verification?: { state?: string | null; certified?: boolean | null; reasons?: string[] } | null;
  capabilities?: CapabilityFlags | null;
  measurement?: Measurement | null;
  transfer?: TransferInfo | null;
  duplicate_status?: DuplicateStatus | null;
}

export interface IndexRunLike {
  run_id: string;
  model_id?: string | null;
  protocol_id?: string | null;
  checkpoint_id?: string | null;
  data_kind?: string | null;
  top_k_saved?: number | null;
  evidence_status?: string | null;
}

/** Why a flag is false, when the backend named a status. Falls back to a plain statement. */
function reasonFor(detail: CapabilityDetail | undefined, fallback: string): string {
  if (detail?.reason) return detail.reason;
  if (detail?.status) return detail.status.replace(/_/g, " ").toLowerCase();
  return fallback;
}

const NO_CANDIDATES = "this run saved exact ranks but no candidate image lists";
const NO_RANKS = "this run recorded an aggregate metric only, not per-query ranks";
const NO_IMAGES = "the images this run refers to are not installed in this deployment";

/** Capabilities of a CIRCUS release composition run, from the served fields. */
export function compositionCapabilities(run: CompositionRunLike): CapabilityReport {
  const served = run.capabilities;
  const detail = run.capability_detail ?? undefined;
  const metrics = run.metrics ?? {};
  const failed = run.is_failed === true || (run.execution_status ?? "").toUpperCase() === "FAILED";

  // Served flags win. The derivation below exists only for a payload that predates them.
  const derived: CapabilityFlags = {
    aggregate_metrics: !failed && (metrics.coverage ?? 0) > 0 && metrics.ndcg_multimodal != null,
    per_query_target_ranks: !failed && (metrics.multimodal?.scored_count ?? 0) > 0,
    candidate_topk: false,
    image_availability: (run.query_count ?? 0) > 0,
  };
  const flags: CapabilityFlags = served ? { ...derived, ...served } : derived;

  const reasons: CapabilityReport["reasons"] = {};
  const statuses: CapabilityReport["statuses"] = {};
  for (const key of Object.keys(flags) as Array<keyof CapabilityFlags>) {
    if (detail?.[key]?.status) statuses[key] = detail[key]!.status!;
  }
  if (!flags.aggregate_metrics) {
    reasons.aggregate_metrics = reasonFor(detail?.aggregate_metrics, failed
      ? "the run did not complete, so no metric was measured"
      : "no aggregate metric was recorded for this run");
  }
  if (!flags.per_query_target_ranks) {
    reasons.per_query_target_ranks = reasonFor(detail?.per_query_target_ranks, failed
      ? "the run did not complete, so no per-query rank exists"
      : NO_RANKS);
  }
  if (!flags.candidate_topk) reasons.candidate_topk = reasonFor(detail?.candidate_topk, NO_CANDIDATES);
  if (!flags.image_availability) reasons.image_availability = reasonFor(detail?.image_availability, NO_IMAGES);
  return { flags, reasons, statuses, derived: !served };
}

/** Capabilities of an aggregate benchmark report, from the served fields when present. */
export function benchmarkCapabilities(run: BenchmarkRunLike): CapabilityReport {
  const metrics = run.observed_metrics ?? {};
  const aggregate = Object.values(metrics).some((value) => typeof value === "number");
  const perQuery = (run.per_query_state ?? "").toUpperCase() !== "CAPABILITY_ABSENT"
    && (run.per_query_capability ?? "").toUpperCase() !== "CAPABILITY_ABSENT";
  const derived: CapabilityFlags = {
    aggregate_metrics: aggregate,
    per_query_target_ranks: perQuery,
    candidate_topk: false,
    image_availability: perQuery,
  };
  const flags: CapabilityFlags = run.capabilities ? { ...derived, ...run.capabilities } : derived;
  const reasons: CapabilityReport["reasons"] = {};
  if (!flags.aggregate_metrics) reasons.aggregate_metrics = "the report records no measured metric";
  if (!flags.per_query_target_ranks) {
    reasons.per_query_target_ranks = `${NO_RANKS} (per_query_state: ${run.per_query_state ?? "unknown"})`;
  }
  if (!flags.candidate_topk) reasons.candidate_topk = NO_CANDIDATES;
  if (!flags.image_availability) reasons.image_availability = NO_IMAGES;
  return { flags, reasons, derived: !run.capabilities };
}

/** Capabilities of one row of the per-query DuckDB index. These carry no capability_detail, so the
 *  report is marked `derived` and the UI says so rather than implying the backend declared it. */
export function indexCapabilities(run: IndexRunLike): CapabilityReport {
  const hasTopK = (run.top_k_saved ?? 0) > 0;
  const flags: CapabilityFlags = {
    aggregate_metrics: true,
    per_query_target_ranks: hasTopK,
    candidate_topk: hasTopK,
    image_availability: hasTopK,
  };
  const reasons: CapabilityReport["reasons"] = {};
  if (!hasTopK) {
    reasons.per_query_target_ranks = NO_RANKS;
    reasons.candidate_topk = NO_CANDIDATES;
    reasons.image_availability = NO_IMAGES;
  }
  return { flags, reasons, derived: true };
}

/** Why an artifact must stay out of default research results, or null when it may be shown.
 *
 *  Provenance only. A run that declares itself synthetic, development or refused is excluded; a
 *  run whose provenance is merely UNKNOWN is not excluded, it is flagged, because "we could not
 *  determine provenance" and "we know it is a fixture" are different facts.
 *
 *  Imported rows (`provenance_source: "released_audit_retriever"`) are real released data but they
 *  are not evidence about a model we ran, so they are their own bucket and the UI must name the
 *  source whenever it shows them.
 */
export type Provenance = {
  bucket: "research" | "development" | "imported";
  label: string;
  undetermined: boolean;
};

export function provenanceOf(run: BenchmarkRunLike | CompositionRunLike | IndexRunLike): Provenance {
  const anyRun = run as CompositionRunLike & BenchmarkRunLike & IndexRunLike;
  if (anyRun.is_synthetic === true || anyRun.synthetic_fixture === true) {
    return { bucket: "development", label: "SYNTHETIC / FIXTURE", undetermined: false };
  }
  if (anyRun.data_kind === "mock" || anyRun.data_kind === "synthetic") {
    return { bucket: "development", label: `DEVELOPMENT (${anyRun.data_kind})`, undetermined: false };
  }
  const source = anyRun.provenance_source;
  if (source) {
    return { bucket: "imported", label: `IMPORTED · ${source}`, undetermined: false };
  }
  const verification = anyRun.verification;
  if (verification?.state === "refused") {
    return { bucket: "development", label: "REFUSED (not research evidence)", undetermined: false };
  }
  if (verification?.certified === false || verification?.state === "uncertified") {
    return { bucket: "development", label: "UNCERTIFIED", undetermined: false };
  }
  const reproduction = (anyRun.reproduction_status ?? "").toUpperCase();
  if (reproduction.startsWith("BLOCKED")) {
    return { bucket: "research", label: reproduction, undetermined: true };
  }
  if (verification === undefined || verification === null) {
    return { bucket: "research", label: "provenance undetermined", undetermined: true };
  }
  return { bucket: "research", label: verification.state?.toUpperCase() || "CERTIFIED", undetermined: false };
}

/** Whether a run is a transfer: trained on another dataset, evaluated here. Labelled TRANSFER and
 *  never presented as a paper row for this benchmark. Reads the served `transfer` object first. */
export function transferLabel(run: CompositionRunLike): { isTransfer: boolean; label: string } {
  const served = run.transfer;
  if (served) {
    if (!served.is_transfer) return { isTransfer: false, label: served.label ?? "trained on FashionIQ" };
    const parts = [served.training_dataset, served.evaluation_dataset].filter(Boolean);
    return {
      isTransfer: true,
      label: served.label ? `TRANSFER · ${served.label}` : parts.length
        ? `TRANSFER · ${parts.join("→")}` : "TRANSFER",
    };
  }
  const declared = run.training_or_transfer_label ?? null;
  const dataset = (run.checkpoint_training_dataset ?? "").toUpperCase();
  const flag = run.checkpoint_transfer === true
    || (typeof run.checkpoint_transfer === "string" && run.checkpoint_transfer.trim() !== "");
  if (flag || (dataset && dataset !== "FASHIONIQ")) {
    const source = (run.checkpoint_transfer as string | undefined)?.trim();
    return {
      isTransfer: true,
      label: typeof run.checkpoint_transfer === "string" && source
        ? `TRANSFER · ${source}`
        : declared ? `TRANSFER · ${declared}` : "TRANSFER",
    };
  }
  return { isTransfer: false, label: declared ?? "trained on FashionIQ" };
}

/** A composition-gap reading.
 *
 *  `Composition Gap = 1 − max(mean_nDCG_image, mean_nDCG_text) / mean_nDCG_multimodal`, with each
 *  mean taken over the cohort of queries that have exact ranks in all three conditions. The value
 *  is a dimensionless FRACTION in (−∞, 1]: a negative gap means the ablated conditions retrieved
 *  BETTER than the multimodal query, which is a real result and is kept, not clamped.
 *
 *  It is not a percentage-point difference and it is not "accuracy"; the caller renders it with the
 *  formula so the unit cannot be read as a gain.
 */
export type GapReading =
  | { state: "available"; value: number; cohort: number; formula: string }
  | { state: "unavailable"; reason: string };

export const COMPOSITION_GAP_FORMULA = "1 − max(mean nDCG_image, mean nDCG_text) / mean nDCG_multimodal";

export function compositionGap(run: CompositionRunLike): GapReading {
  const metrics = run.metrics ?? {};
  const cohort = metrics.gap_cohort_count ?? 0;
  const mm = metrics.gap_multimodal_ndcg;
  const text = metrics.gap_text_only_ndcg;
  const image = metrics.gap_image_only_ndcg;
  if (metrics.gap_unavailable_reason) {
    return { state: "unavailable", reason: metrics.gap_unavailable_reason };
  }
  if (mm == null || text == null || image == null) {
    return { state: "unavailable", reason: "one or more of the three cohort means was not recorded" };
  }
  if (!(mm > 0)) {
    return { state: "unavailable", reason: `mean multimodal nDCG is ${mm}, so the gap has no denominator` };
  }
  if (cohort <= 0) {
    return { state: "unavailable", reason: "no query has exact ranks in all three conditions" };
  }
  const derived = 1 - Math.max(image, text) / mm;
  const recorded = metrics.composition_gap_ndcg;
  // The recorded value is the backend's own arithmetic on the same cohort. A disagreement means
  // one of the two is describing a different cohort, which must be surfaced, not averaged over.
  if (recorded != null && Math.abs(recorded - derived) > 1e-6) {
    return {
      state: "unavailable",
      reason: `recorded gap ${recorded.toFixed(6)} disagrees with the formula's ${derived.toFixed(6)} on this cohort`,
    };
  }
  return { state: "available", value: derived, cohort, formula: COMPOSITION_GAP_FORMULA };
}

/** Scored vs unscored. `unscored` is a pipeline fact (no exact rank was produced), not a retrieval
 *  failure, and mixing them is what floated null ranks to the top of "worst".
 *
 *  Reads `measurement` when the backend provides it (the authoritative pair of counters), falling
 *  back to the per-condition metric bucket for older payloads. */
export interface ScoredCounts {
  scored: number;
  total: number;
  unscored: number;
  coverage: number | null;
  /** Set when the backend reported the unscored breakdown by kind. */
  unscoredByKind?: Record<string, number> | null;
  derived: boolean;
}

export function scoredCounts(run: CompositionRunLike, condition: string): ScoredCounts {
  const metrics0 = run.metrics ?? {};
  const bucket0 = condition === "text_only_black_image" ? metrics0.text_only
    : condition === "image_only_empty_text" ? metrics0.image_only : metrics0.multimodal;
  const measurement = run.measurement;
  // The served `measurement` block is not always subset-scoped: on this deployment its
  // `scored_count`/`unscored_query_count` are WHOLE-RUN totals while `sample_count`/`denominator`
  // follow the requested subset, which rendered "5,884 / 4,069 (144.6%)" — a numerator larger than
  // its own denominator. The per-condition metric bucket IS subset-scoped, so the two are
  // cross-checked and the bucket wins when they disagree. A served count that cannot be reconciled
  // with its own denominator is not usable evidence.
  const reconcilable = measurement != null
    && typeof measurement.sample_count === "number"
    && typeof measurement.scored_count === "number"
    && measurement.scored_count <= measurement.sample_count
    && (bucket0?.query_count == null || bucket0.query_count === measurement.sample_count);
  if (measurement && typeof measurement.sample_count === "number" && reconcilable) {
    const total = measurement.sample_count;
    const scored = measurement.scored_count ?? 0;
    const unscored = measurement.unscored_query_count ?? Math.max(total - scored, 0);
    return {
      scored,
      total,
      unscored,
      coverage: measurement.coverage ?? (total > 0 ? scored / total : null),
      unscoredByKind: measurement.unscored_by_kind ?? null,
      derived: false,
    };
  }
  const metrics = run.metrics ?? {};
  const bucket = condition === "text_only_black_image" ? metrics.text_only
    : condition === "image_only_empty_text" ? metrics.image_only : metrics.multimodal;
  const total = bucket?.query_count ?? run.query_count ?? 0;
  const scored = bucket?.scored_count ?? 0;
  return {
    scored,
    total,
    unscored: Math.max(total - scored, 0),
    coverage: total > 0 ? scored / total : null,
    derived: true,
  };
}

/** One measurement identity, for grouping repeated attempts without inventing a "unique
 *  measurement" from a missing field.
 *
 *  The served `duplicate_status` / `duplicate_attempt_count` / `measurement_identity_*` fields are
 *  authoritative; `groupMeasurements` is only for rows that arrive without them (the per-query
 *  index), and it marks the group `undetermined` whenever a component is absent. Absence of
 *  `evaluator_environment_knobs` is NOT evidence that two runs are duplicates, and it is not
 *  evidence that they are distinct either.
 */
export interface AttemptInput {
  run_id: string;
  model_id?: string | null;
  checkpoint_id?: string | null;
  checkpoint_identity?: string | null;
  protocol_id?: string | null;
  checkpoint_training_dataset?: string | null;
  checkpoint_training_noise_pct?: number | null;
  evaluation_noise_pct?: number | null;
  subset?: string | null;
  duplicate_status?: DuplicateStatus | null;
  duplicate_attempt_count?: number | null;
  measurement_identity_confidence?: "full" | "partial" | null;
  measurement_identity_missing_inputs?: string[] | null;
}

export interface MeasurementAttempt {
  run_id: string;
  identity: string;
  undetermined: boolean;
  missing: string[];
}

export interface MeasurementGroup {
  identity: string;
  undetermined: boolean;
  attempts: MeasurementAttempt[];
  missing: string[];
  /** The backend's own verdict when it supplied one. */
  served?: DuplicateStatus | null;
}

export function measurementIdentity(run: AttemptInput): MeasurementAttempt {
  const missing: string[] = [];
  const checkpoint = run.checkpoint_identity ?? run.checkpoint_id ?? null;
  if (!checkpoint) missing.push("checkpoint identity");
  if (!run.model_id) missing.push("model");
  if (!run.protocol_id) missing.push("protocol");
  if (run.checkpoint_training_noise_pct == null) missing.push("training noise");
  for (const served of run.measurement_identity_missing_inputs ?? []) {
    if (!missing.includes(served)) missing.push(served);
  }
  const parts = [
    run.model_id ?? "?",
    checkpoint ?? "?",
    run.protocol_id ?? "?",
    run.checkpoint_training_dataset ?? "?",
    run.subset ?? "all",
    `train=${run.checkpoint_training_noise_pct ?? "?"}`,
    `eval=${run.evaluation_noise_pct ?? "?"}`,
  ];
  return {
    run_id: run.run_id ?? "",
    identity: parts.join("\u001f"),
    undetermined: missing.length > 0 || run.measurement_identity_confidence === "partial",
    missing,
  };
}

export function groupMeasurements(runs: AttemptInput[]): MeasurementGroup[] {
  const groups = new Map<string, MeasurementGroup>();
  for (const run of runs) {
    const attempt = measurementIdentity(run);
    const existing = groups.get(attempt.identity);
    if (existing) {
      existing.attempts.push(attempt);
      for (const field of attempt.missing) {
        if (!existing.missing.includes(field)) existing.missing.push(field);
      }
      existing.undetermined = existing.undetermined || attempt.undetermined;
      existing.served = existing.served ?? run.duplicate_status ?? null;
    } else {
      groups.set(attempt.identity, {
        identity: attempt.identity,
        undetermined: attempt.undetermined,
        attempts: [attempt],
        missing: [...attempt.missing],
        served: run.duplicate_status ?? null,
      });
    }
  }
  return [...groups.values()];
}

/** What a duplicate status means, in the reader's words. `undetermined` is stated as such: it is
 *  not a synonym for "unique" and not a synonym for "duplicate". */
export function duplicateLabel(status: DuplicateStatus | null | undefined, attempts: number | null | undefined): string | null {
  if (!status) return null;
  const count = attempts ?? 1;
  switch (status) {
    case "duplicate":
      return `${count} attempts of the same measurement`;
    case "undetermined":
      return "cannot tell: identity inputs missing";
    case "synthetic_fixture":
      return "fixture (not a measurement)";
    default:
      return count > 1 ? `${count} attempts, one measurement` : "unique measurement";
  }
}

/** Where an artifact was produced, when the served environment knobs prove it.
 *
 *  The aggregate store legitimately holds rows written by the test suite (each `pytest` run
 *  publishes into `WORKBENCH_RESULTS_ROOT`), and those rows are real API rows that must stay
 *  visible. They are not, however, evidence about a model this deployment trained, and ten of them
 *  sit side by side with two real reports. The UI already labels them REFUSED; it did not say WHY
 *  they exist, so ten rows looked like duplicates of one another. This reads the served
 *  `evaluator_environment_knobs` — the paths the evaluator actually ran with — and names the test
 *  workspace. It never matches on `model_id`: a name is not a schema.
 */
export function testWorkspaceOrigin(run: {
  evaluator_environment_knobs?: Record<string, string> | null;
  is_synthetic?: boolean | null;
  synthetic_fixture?: boolean | null;
  data_kind?: string | null;
}): string | null {
  if (run.is_synthetic === true || run.synthetic_fixture === true) return "fixture";
  if (run.data_kind === "mock" || run.data_kind === "synthetic") return `development (${run.data_kind})`;
  for (const value of Object.values(run.evaluator_environment_knobs ?? {})) {
    const match = /(\/[^\s"']*pytest-of-[^/\s]+\/pytest-\d+)/.exec(String(value));
    if (match) return match[1];
  }
  return null;
}

/** Reported protocols, most reports first. A protocol with zero reports is not listed, because this
 *  is the affordance a reader sees INSTEAD of a dead end: it must name only places with data. */
export function protocolsWithReports(aggregates: readonly BenchmarkRunLike[]): Array<{ protocol_id: string; count: number }> {
  const counts = new Map<string, number>();
  for (const run of aggregates) {
    const id = run.protocol_id;
    if (!id) continue;
    counts.set(id, (counts.get(id) ?? 0) + 1);
  }
  return [...counts.entries()]
    .map(([protocol_id, count]) => ({ protocol_id, count }))
    .sort((left, right) => right.count - left.count || left.protocol_id.localeCompare(right.protocol_id));
}

/** Which protocol the benchmark view should show when the operator did not name one.
 *
 *  The deployment's first protocol (CIRCUS release) has composition rows but ZERO aggregate
 *  reports; the 13 real reports live under `fashioniq_original_split`. Landing on the former showed
 *  an empty card and read as broken. This only redirects when the current protocol has NO report at
 *  all AND another one does, and only when navigating into the view — an explicit `?protocol=…`
 *  selection is left alone, because a reader who asked to see an empty protocol deserves to see it.
 */
export function protocolOrDefault(current: string, served: readonly { protocol_id: string }[]): string {
  if (served.some((item) => item.protocol_id === current)) return current;
  return served[0]?.protocol_id ?? current;
}

/** One column of the CIRCUS composition table.
 *
 *  The table overflowed its 1148 px container by 104 px at 1440x900 once the 11 columns existed,
 *  which clipped the last two. With `table-layout: fixed` every column's width is declared here
 *  rather than left to content, so the fit is a fact that can be asserted
 *  (`workbench/tests/test_frontend_layout_columns.py`) instead of something only a screenshot knows.
 *  Fixed columns are `px` because their content is (a checkbox, a badge, four digits); the two
 *  flexible ones split the remainder, so the model cell can wrap instead of pushing the table wide.
 */
export interface CompositionColumn { key: string; label: string; px?: number; flex?: boolean }

export const COMPOSITION_COLUMNS: readonly CompositionColumn[] = [
  { key: "select", label: "", px: 34 },
  { key: "model", label: "Mô hình / checkpoint", flex: true },
  { key: "source", label: "Nguồn / transfer", flex: true },
  { key: "status", label: "Trạng thái", px: 116 },
  { key: "ndcg", label: "MM nDCG", px: 90 },
  { key: "r10", label: "MM R@10", px: 112 },
  { key: "text", label: "Text nDCG", px: 82 },
  { key: "image", label: "Image nDCG", px: 88 },
  // Gap holds the literal "unavailable" when the cohort is empty: 92 px broke it mid-word
  // ("unavailab / le"), which reads as a typo rather than a state.
  { key: "gap", label: "Gap", px: 104 },
  { key: "scored", label: "Đã chấm", px: 112 },
  // The toggle must not wrap: "Chi / tiết" made the row taller than its content.
  { key: "detail", label: "Chi tiết", px: 104 },
];

/** The narrowest a flexible column may be before its header wraps mid-word.
 *  148 px fit the old 216 px sidebar; the light shell uses a 260 px sidebar and 32 px content
 *  padding, so the same container is ~80 px narrower and the declared minimum overflowed it. */
export const COMPOSITION_FLEX_MIN = 112;

/** Total width the table needs at least: every fixed column plus the flexible minimum. */
export function compositionMinWidth(columns: readonly CompositionColumn[] = COMPOSITION_COLUMNS): number {
  const fixed = columns.reduce((sum, column) => sum + (column.px ?? 0), 0);
  const flexible = columns.filter((column) => column.flex).length;
  return fixed + flexible * COMPOSITION_FLEX_MIN;
}

/** Fixed cutoffs that the protocol itself defines. A UI control must not pretend to change these. */
export const FIXED_CUTOFFS = {
  compositionOutcomeK: 10,
  compositionOutcomeNote: "Outcome (composition_required / unresolved / …) is classified at @10 by the release protocol; it is fixed and not controlled from the UI.",
  recallCutoffs: [1, 10, 50],
  candidateDisplayNote: "Top-K below is how many saved candidate images are DISPLAYED, not the cutoff that classified the outcome.",
} as const;

/** The distinct scoring implementations among the visible rows, sorted.
 *
 *  `einsum`, `einsum(..., optimize=True)` and `matmul` sum in different orders, so their raw scores
 *  are not on a common scale and score differences must not be read as calibrated distances. The
 *  evaluation contract (gallery, query set, cutoff, metric definitions) is the same, so ranks and
 *  metrics of rows under one protocol remain comparable; the table names the implementations rather
 *  than implying they were measured by one. A row whose backend derivation is missing counts as its
 *  own `unknown` — never silently folded into a real one.
 */
export function scoringImplsPresent(rows: readonly CompositionRunLike[]): string[] {
  return [...new Set(rows.map((run) => run.scoring_impl?.impl?.trim() || "unknown"))].sort();
}
