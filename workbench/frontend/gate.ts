import type { MetricComparison } from "./gate_types";

export const DEFAULT_THRESHOLD_PCT = 5.0;
export const thresholdStorageKey = "cir-workbench.thresholdPct";

/** Read the stored deviation threshold. `Number(null)` is 0, so a raw cast silently turned a first
 *  visit (no saved value) into a 0% threshold -- every deviation flagged. A missing entry falls back
 *  to the default; an unparsable or out-of-range value does too. */
export function readThresholdPct(storage: Pick<Storage, "getItem"> = localStorage): number {
  const raw = storage.getItem(thresholdStorageKey);
  const saved = raw === null || raw.trim() === "" ? NaN : Number(raw);
  return Number.isFinite(saved) && saved >= 0 && saved <= 20 ? saved : DEFAULT_THRESHOLD_PCT;
}

export type VerificationInfo = {
  artifact_exists?: boolean;
  execution_authenticated?: boolean;
  is_research_evidence?: boolean;
  certified?: boolean;
  development_mode?: boolean;
  certification_mode?: string;
  state?: string;
  report_digest?: string | null;
  reasons?: string[];
  evidence_reasons?: string[];
};

export type RunGate = {
  benchmark_gate?: Record<string, unknown> | null;
  parity_status?: string | null;
  variant_provenance?: Record<string, unknown> | null;
  category_metrics?: Record<string, Record<string, number | null>> | null;
  observed_metrics?: Record<string, number | null> | null;
  execution_status?: string | null;
  is_synthetic?: boolean | null;
  verification?: VerificationInfo | null;
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

/** The warning labels a run can carry. Anything else belongs to the gate detail text, not to a
 *  new label colour. */
export type WarningLabel =
  | { kind: "caption-substitute"; text: "CAPTION THAY THẾ"; tone: "amber" }
  | { kind: "paper-deviation"; text: string; tone: "red" }
  | { kind: "proof-unavailable"; text: "CHƯA CÓ PROOF (PROOF UNAVAILABLE)"; tone: "amber" }
  | { kind: "uncorroborated"; text: "CHƯA ĐỐI SOÁT LEDGER (UNCORROBORATED)"; tone: "red" }
  | { kind: "refused"; text: "TỪ CHỐI CHỨNG NHẬN (REFUSED)"; tone: "red" }
  | { kind: "uncertified"; text: "CHƯA CHỨNG NHẬN (UNCERTIFIED)"; tone: "red" };

export function isCertifiedEvidence(gate: RunGate | undefined): boolean {
  return Boolean(gate?.verification?.certified && gate?.verification?.is_research_evidence);
}
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
  // Fail closed: `undefined` is not a certified run, it is a run whose certification was never
  // stated. Treating it as clean made every `/api/runs` row without an aggregate gate record
  // render with no warning at all, so an uncorroborated run looked valid.
  if (!isCertifiedEvidence(gate)) {
    const state = gate?.verification?.state || gate?.verification?.certification_mode
      || (gate?.verification ? "uncertified" : "uncorroborated");
    if (state === "proof_unavailable") {
      labels.push({ kind: "proof-unavailable", text: "CHƯA CÓ PROOF (PROOF UNAVAILABLE)", tone: "amber" });
    } else if (state === "refused") {
      // Distinct from "uncertified": the verifier looked at this run and rejected it, which is a
      // different fact from "its certification was never established".
      labels.push({ kind: "refused", text: "TỪ CHỐI CHỨNG NHẬN (REFUSED)", tone: "red" });
    } else if (state === "uncorroborated") {
      labels.push({ kind: "uncorroborated", text: "CHƯA ĐỐI SOÁT LEDGER (UNCORROBORATED)", tone: "red" });
    } else {
      labels.push({ kind: "uncertified", text: "CHƯA CHỨNG NHẬN (UNCERTIFIED)", tone: "red" });
    }
  }
  return labels;
}
