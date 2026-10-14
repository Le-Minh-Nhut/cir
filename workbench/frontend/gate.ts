import type { MetricComparison } from "./gate_types";

export const DEFAULT_THRESHOLD_PCT = 5.0;
export const thresholdStorageKey = "cir-workbench.thresholdPct";

export type RunGate = {
  benchmark_gate?: Record<string, unknown> | null;
  parity_status?: string | null;
  variant_provenance?: Record<string, unknown> | null;
  category_metrics?: Record<string, Record<string, number>> | null;
  observed_metrics?: Record<string, number> | null;
  execution_status?: string | null;
  is_synthetic?: boolean | null;
};

export type GateVerdict = {
  status?: string;
  passed?: boolean;
  variant?: boolean;
  comparisons?: MetricComparison[];
  blockers?: string[];
  detail?: string;
};

export type BenchmarkRun = {
  run_id: string;
  model_id: string | null;
  checkpoint_id: string | null;
  protocol_id: string | null;
} & RunGate;

/** Re-flag recorded comparisons against a user-chosen threshold. Same arithmetic as the backend's
 *  `deviations()` helper, applied to the verdict the backend already computed and recorded. */
export function reflag(comparisons: MetricComparison[] | undefined, thresholdPct: number): MetricComparison[] {
  return (comparisons || []).filter(
    (c) => c.relative_error_pct !== null && c.relative_error_pct > thresholdPct,
  );
}

/** The two warning labels, and only these two. Anything else a run can say belongs to the gate
 *  detail text, not to a new label colour. */
export type WarningLabel =
  | { kind: "caption-substitute"; text: "CAPTION THAY THẾ"; tone: "amber" }
  | { kind: "paper-deviation"; text: string; tone: "red" };

export function warningLabels(gate: RunGate | undefined, thresholdPct: number): WarningLabel[] {
  const labels: WarningLabel[] = [];
  if ((gate?.variant_provenance || {}).caption_alias_staged) {
    labels.push({ kind: "caption-substitute", text: "CAPTION THAY THẾ", tone: "amber" });
  }
  const deviating = reflag(((gate?.benchmark_gate || {}) as GateVerdict).comparisons, thresholdPct);
  if (deviating.length) {
    const worst = deviating.reduce((a, b) =>
      (b.relative_error_pct ?? 0) > (a.relative_error_pct ?? 0) ? b : a);
    labels.push({
      kind: "paper-deviation",
      text: `LỆCH PAPER ${Math.max(worst.relative_error_pct ?? 0, thresholdPct).toFixed(2)}% (${worst.metric})`,
      tone: "red",
    });
  }
  return labels;
}
