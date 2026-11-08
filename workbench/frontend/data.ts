import { useCallback, useEffect, useMemo, useRef } from "react";
import { fetchJson, useRegion, type RegionHandle } from "./useRegion";
import type { BenchmarkRun, RunGate } from "./gate";
import type { Cohort, CompositionQuery, CompositionRun, DuplicateStatus, Job, Measurement, Model, Query, Run } from "./types";
import type { CapabilityFlags, CapabilityReport, Provenance } from "./catalog";
import { compositionCapabilities, benchmarkCapabilities, indexCapabilities, provenanceOf, transferLabel } from "./catalog";
import type { Subset } from "./routes";

/** Every source the workbench reads, each as its own keyed region.
 *
 *  The previous shell fetched once on mount and refreshed on a 10-second timer, so a page could not
 *  say "this failed" or "this is loading" for the region it was rendering, and a slow answer could
 *  land after the context changed. Each source here has its own request identity; a failure is
 *  local to the region that needed it.
 */

/** Whether imported released-audit rows are in scope. Off by default: they are real released data
 *  but they are not evidence about a model this deployment ran, so they must be asked for and are
 *  labelled with their source whenever shown. */
export type ImportedScope = "exclude" | "include";

export interface RunCatalogEntry {
  /** Stable key used in URLs and selection: prefixed so a composition run and an index run can
   *  never be confused even if two hashes collide. */
  key: string;
  family: "composition" | "index" | "aggregate";
  runId: string;
  label: string;
  modelId: string;
  checkpointId: string;
  protocolId: string;
  subset: Subset | "all";
  status: string;
  capabilities: CapabilityReport;
  provenance: Provenance;
  isTransfer: boolean;
  transferLabel: string | null;
  duplicateStatus: DuplicateStatus | null;
  duplicateAttempts: number | null;
  measurement: Measurement | null;
  importedSource: string | null;
  composition?: CompositionRun;
  indexRun?: Run;
  aggregate?: BenchmarkRun;
}

const PROTOCOL_LABELS: Record<string, string> = {
  circus_fashioniq_release_v1: "CIRCUS FashionIQ Release v1",
  fashioniq_original_split: "FashionIQ — Original Split (Full Gallery)",
  fashioniq_val_split: "FashionIQ — Val Split (Pair-Union Gallery)",
  fashioniq_full_gallery_ref_excluded: "FashionIQ — Full Gallery, Reference Excluded",
};

export function protocolLabel(id: string): string {
  return PROTOCOL_LABELS[id] ?? id;
}

/** The protocols this deployment can offer, derived from the data that is actually present.
 *  A protocol the registry knows but nothing was served for must not be selectable as if it had
 *  results, and a protocol with results must never be hidden behind a label for a different one. */
export function protocolsWithData(compositionRuns: CompositionRun[], indexRuns: Run[], aggregates: BenchmarkRun[]): string[] {
  const ids = new Set<string>();
  for (const run of compositionRuns) if (run.protocol_id) ids.add(run.protocol_id);
  for (const run of indexRuns) if (run.protocol_id) ids.add(run.protocol_id);
  for (const run of aggregates) if (run.protocol_id) ids.add(run.protocol_id);
  return [...ids];
}

export function buildCatalog(compositionRuns: CompositionRun[], indexRuns: Run[], aggregates: BenchmarkRun[], subset: Subset, imported: ImportedScope = "exclude"): RunCatalogEntry[] {
  const entries: RunCatalogEntry[] = [];
  const seenComposition = new Set<string>();
  for (const run of compositionRuns) {
    // The API returns one summary per (run, subset); only the active subset's rows are selectable,
    // so a "selected run" always means "this run at this subset" and cannot silently be another.
    if (run.subset !== subset || seenComposition.has(run.run_id)) continue;
    const provenance = provenanceOf(run);
    // Imported released-audit rows are opt-in: they are real released data, but not evidence about
    // a model this deployment ran, so they never enter the default list.
    if (provenance.bucket === "imported" && imported === "exclude") continue;
    seenComposition.add(run.run_id);
    const capabilities = compositionCapabilities(run);
    entries.push({
      key: `c:${run.run_id}`,
      family: "composition",
      runId: run.run_id,
      label: `${run.model_id} / ${run.checkpoint_id}`,
      modelId: run.model_id,
      checkpointId: run.checkpoint_id,
      protocolId: run.protocol_id,
      subset,
      status: run.execution_status || "UNKNOWN",
      capabilities,
      provenance,
      isTransfer: transferLabel(run).isTransfer,
      transferLabel: transferLabel(run).label,
      duplicateStatus: run.duplicate_status ?? null,
      duplicateAttempts: run.duplicate_attempt_count ?? null,
      measurement: run.measurement ?? null,
      importedSource: run.provenance_source ?? null,
      composition: run,
    });
  }
  for (const run of indexRuns) {
    entries.push({
      key: `i:${run.run_id}`,
      family: "index",
      runId: run.run_id,
      label: `${run.model_id} / ${run.method_name} / ${run.checkpoint_id}`,
      modelId: run.model_id,
      checkpointId: run.checkpoint_id,
      protocolId: run.protocol_id,
      subset: "all",
      status: run.evidence_status || run.data_kind.toUpperCase(),
      capabilities: indexCapabilities(run),
      provenance: provenanceOf(run),
      isTransfer: false,
      transferLabel: null,
      duplicateStatus: null,
      duplicateAttempts: null,
      measurement: null,
      importedSource: null,
      indexRun: run,
    });
  }
  for (const run of aggregates) {
    entries.push({
      key: `a:${run.run_id}`,
      family: "aggregate",
      runId: run.run_id,
      label: `${(run.model_id || "?").toUpperCase()} / ${run.checkpoint_id}`,
      modelId: run.model_id || "",
      checkpointId: run.checkpoint_id || "",
      protocolId: run.protocol_id || "",
      subset: "all",
      status: run.verification?.state?.toUpperCase() || "UNCERTIFIED",
      capabilities: benchmarkCapabilities(run),
      provenance: provenanceOf(run),
      isTransfer: transferLabel(run as never).isTransfer,
      transferLabel: transferLabel(run as never).label,
      duplicateStatus: run.duplicate_status ?? null,
      duplicateAttempts: run.duplicate_attempt_count ?? null,
      measurement: run.measurement ?? null,
      importedSource: null,
      aggregate: run,
    });
  }
  return entries;
}

export interface CompositionPagePayload {
  items: CompositionQuery[];
  total: number;
  queries_unavailable_reason?: string;
}

export interface UniverseEntry<T> {
  key: string;
  items: T[];
  total?: number;
}

/** `GET /api/composition/capabilities?subset=…` — what the deployment can do right now. */
export interface CapabilitySummary {
  run_count: number;
  capability_counts: Record<keyof CapabilityFlags, number>;
  artifact_count: number;
  measured_artifact_count: number;
  unique_measurement_count: number;
  undetermined_artifact_count: number;
  confidence: "full" | "partial";
  image_availability_is_sample: boolean;
}

/** `GET /api/evidence/summary` — replaces the dashboard's hardcoded "(1 unique measurement)". */
export interface EvidenceSummary {
  aggregate_run_count: number;
  certified_aggregate_count: number;
  unique_measurement_count: number;
  undetermined_measurement_count: number;
  duplicate_attempt_count: number;
  confidence: "full" | "partial";
}

export interface WorkbenchData {
  models: RegionHandle<Model[]>;
  indexRuns: RegionHandle<Run[]>;
  aggregates: RegionHandle<BenchmarkRun[]>;
  compositionRuns: RegionHandle<CompositionRun[]>;
  cohorts: RegionHandle<Cohort[]>;
  jobs: RegionHandle<Job[]>;
  gates: RegionHandle<Record<string, RunGate>>;
  /** Composition queries for the active (run, subset, condition, filters, order, page). */
  compositionPage: RegionHandle<CompositionPagePayload>;
  /** Whole-universe query lists for the selected composition runs: analysis input. */
  compositionUniverse: RegionHandle<Array<UniverseEntry<CompositionQuery>>>;
  /** Per-query index run pages, keyed by the selected run ids. */
  indexUniverse: RegionHandle<Array<UniverseEntry<Query>>>;
  compositionCapabilitiesSummary: RegionHandle<CapabilitySummary>;
  evidenceSummary: RegionHandle<EvidenceSummary>;
  protocolIds: string[];
  catalog: RunCatalogEntry[];
  refreshAll: () => void;
}

export interface DataQuery {
  protocol: string;
  subset: Subset;
  selectedKeys: string[];
  condition: string;
  outcome: string;
  category: string;
  order: string;
  offset: number;
  limit: number;
  query: string;
  imported?: ImportedScope;
  queryState?: string;
}

export function useWorkbenchData(query: DataQuery): WorkbenchData {
  const imported: ImportedScope = query.imported ?? "exclude";
  const models = useRegion<Model[]>("models", "models", (signal) => fetchJson("/api/models", signal));
  const indexRuns = useRegion<Run[]>("index-runs", "index", (signal) => fetchJson("/api/runs", signal));
  const aggregates = useRegion<BenchmarkRun[]>("aggregates", "aggregates", (signal) => fetchJson("/api/benchmark-runs", signal));
  const compositionRuns = useRegion<CompositionRun[]>("composition-runs", `composition\u001f${imported}`,
    (signal) => fetchJson(`/api/composition/runs${imported === "include" ? "?include_imported=true" : ""}`, signal));
  // Deployment-wide capability and evidence counters. The dashboard reads these instead of printing
  // any literal, so a computed count and a stated count can never disagree on screen.
  const compositionCapabilitiesSummary = useRegion<CapabilitySummary>("composition-capabilities",
    `capabilities\u001f${query.subset}`,
    (signal) => fetchJson(`/api/composition/capabilities?subset=${query.subset}`, signal));
  const evidenceSummary = useRegion<EvidenceSummary>("evidence-summary", "evidence",
    (signal) => fetchJson("/api/evidence/summary", signal));
  const cohorts = useRegion<Cohort[]>("cohorts", "cohorts", (signal) => fetchJson("/api/cohorts", signal));
  const jobs = useRegion<Job[]>("jobs", "jobs", (signal) => fetchJson("/api/jobs", signal));
  const gates = useRegion<Record<string, RunGate>>("gates", "gates", (signal) => fetchJson("/api/runs/gate", signal));

  const safeComposition = useMemo(() => (Array.isArray(compositionRuns.region.data) ? compositionRuns.region.data : []), [compositionRuns.region.data]);
  const safeIndex = useMemo(() => (Array.isArray(indexRuns.region.data) ? indexRuns.region.data : []), [indexRuns.region.data]);
  const safeAggregates = useMemo(() => (Array.isArray(aggregates.region.data) ? aggregates.region.data : []), [aggregates.region.data]);

  const protocolIds = useMemo(
    () => protocolsWithData(safeComposition, safeIndex, safeAggregates),
    [safeComposition, safeIndex, safeAggregates]);
  const catalog = useMemo(
    () => buildCatalog(safeComposition, safeIndex, safeAggregates, query.subset, imported),
    [safeComposition, safeIndex, safeAggregates, query.subset, imported]);

  const selectedComposition = query.selectedKeys.filter((key) => key.startsWith("c:")).map((key) => key.slice(2));
  const selectedIndex = query.selectedKeys.filter((key) => key.startsWith("i:")).map((key) => key.slice(2));

  const compositionPageKey = [
    "composition-page", query.subset, selectedComposition[0] ?? "", query.condition,
    query.outcome, query.category, query.order, query.offset, query.limit, query.query,
    query.queryState ?? "", query.imported ?? "exclude",
  ].join("\u001f");
  const compositionPage = useRegion<CompositionPagePayload>(
    "composition-page",
    compositionPageKey,
    useMemo(() => {
      const runId = selectedComposition[0];
      if (!runId) return null;
      return async (signal: AbortSignal) => {
        const params = new URLSearchParams({
          subset: query.subset, condition: query.condition, order: query.order,
          limit: String(query.limit), offset: String(query.offset),
        });
        if (query.category) params.set("category", query.category);
        if (query.outcome) params.set("outcome", query.outcome);
        if (query.query) params.set("query_id", query.query);
        if (query.queryState) params.set("query_state", query.queryState);
        return fetchJson(`/api/composition/runs/${encodeURIComponent(runId)}/queries?${params}`, signal);
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedComposition[0], query.subset, query.condition, query.outcome, query.category,
        query.order, query.offset, query.limit, query.query, query.queryState]),
  );

  // The analysis input needs the WHOLE universe of each selected run, not the visible page: a
  // consensus computed over the first 20 rows would be a different, and unstated, cohort.
  const compositionUniverse = useRegion<Array<UniverseEntry<CompositionQuery>>>(
    "composition-universe",
    `universe\u001f${query.subset}\u001f${selectedComposition.join(",")}`,
    useMemo(() => {
      if (!selectedComposition.length) return null;
      return async (signal: AbortSignal) => Promise.all(selectedComposition.map(async (runId) => {
        const params = new URLSearchParams({ subset: query.subset, condition: "multimodal", order: "worst", limit: "20000", offset: "0" });
        const payload = await fetchJson<{ items: CompositionQuery[] }>(`/api/composition/runs/${encodeURIComponent(runId)}/queries?${params}`, signal);
        return { key: `c:${runId}`, items: payload.items ?? [] };
      }));
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedComposition.join(","), query.subset]),
  );

  const indexUniverse = useRegion<Array<UniverseEntry<Query>>>(
    "index-universe",
    `universe\u001f${selectedIndex.join(",")}`,
    useMemo(() => {
      if (!selectedIndex.length) return null;
      return async (signal: AbortSignal) => Promise.all(selectedIndex.map(async (runId) => {
        const params = new URLSearchParams({ run_id: runId, limit: "20000", offset: "0", order: "worst" });
        const payload = await fetchJson<{ items: Query[]; total: number }>(`/api/queries?${params}`, signal);
        return { key: `i:${runId}`, items: payload.items ?? [], total: payload.total ?? 0 };
      }));
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [selectedIndex.join(",")]),
  );

  // Jobs settle on their own; poll while any is live rather than on a fixed 2s tick forever.
  const liveJobs = (jobs.region.data ?? []).some((job) => ["queued", "preparing", "running"].includes(job.status));
  const pollRef = useRef<number | null>(null);
  useEffect(() => {
    if (!liveJobs) return;
    const timer = window.setInterval(() => jobs.retry(), 3000);
    pollRef.current = timer;
    return () => window.clearInterval(timer);
  }, [liveJobs, jobs.retry]);

  const refreshAll = useCallback(() => {
    models.retry(); indexRuns.retry(); aggregates.retry(); compositionRuns.retry();
    cohorts.retry(); jobs.retry(); gates.retry();
    compositionCapabilitiesSummary.retry(); evidenceSummary.retry();
  }, [models.retry, indexRuns.retry, aggregates.retry, compositionRuns.retry, cohorts.retry, jobs.retry,
      gates.retry, compositionCapabilitiesSummary.retry, evidenceSummary.retry]);

  return { models, indexRuns, aggregates, compositionRuns, cohorts, jobs, gates,
    compositionPage, compositionUniverse, indexUniverse, compositionCapabilitiesSummary,
    evidenceSummary, protocolIds, catalog, refreshAll };
}
