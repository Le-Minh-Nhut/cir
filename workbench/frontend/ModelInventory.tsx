import { useMemo, useState } from "react";
import { protocolLabel, type WorkbenchData } from "./data";
import type { Location } from "./routes";
import type { Model } from "./types";
import { Badge, Detail, ErrorState, LoadingState } from "./ui";
import { fmtDenominator, fmtInt, fmtNdcg, fmtPercentValue } from "./format";
import { scoredCounts } from "./catalog";
import { crossScopeAllowed, inventoryStatus, nativeMeasuredIds, nativeRowsFor, nativeUnattributedFor, otherScopeFor, protocolHasSubsets, type ScopeRow } from "./scope";
import { SUBSET_LABEL } from "./routes";

/** Operator scope for this review. Not a registry fact and not a measurement: the two models whose
 *  auxiliary data or checkpoints are missing stay LISTED with this reason attached, so "why is it
 *  not in the numbers" has an answer on the row instead of an absence.
 *  ponytail: keyed by model_id, so it must be revised when the registry changes; the honest general
 *  form is a `deferred_reason` field served per model. */
export const DEFERRED_MODELS: Record<string, string> = {
  // CSMCIR was deferred for a missing auxiliary asset; it was then MEASURED natively on FashionIQ
  // (macro R@10 57.05 / R@50 77.30, run `f03887d2`). Only its CIRCUS composition stays deferred, so
  // the row must not read as "nothing measured" while a counted native row exists.
  csmcir: "FashionIQ native đã đo · CIRCUS composition tạm gác",
  sprc: "Tạm gác theo yêu cầu: chưa tải checkpoint",
};

/** Every model in `/api/models`, with its run status in the current protocol/subset.
 *
 *  The old models view listed one row per API entry but said nothing about whether that model had a
 *  run here, so a reader could not tell "not run" from "ran and failed" from "deferred". Rows are
 *  the whole registry, always: a model with no result is a row with a reason and a next step, not a
 *  missing row. Counts are separated (models / runs / pending / deferred) because "17" as a single
 *  number conflated three different questions.
 */
export function ModelInventory({ data, location, onGoto }: {
  data: WorkbenchData; location: Location; onGoto: (patch: Partial<Location>) => void;
}) {
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const models = data.models.region.data ?? [];

  const entries = useMemo(
    () => data.catalog.filter((entry) => entry.protocolId === location.protocol && entry.provenance.bucket !== "development"),
    [data.catalog, location.protocol]);
  const compositionMeasuredIds = useMemo(() => new Set(entries
    .filter((entry) => !entry.composition?.is_failed && entry.status !== "FAILED" && entry.capabilities.flags.aggregate_metrics)
    .map((entry) => entry.modelId)), [entries]);
  const nativeRows = useMemo(
    () => data.nativeMeasurements.region.data?.measurements ?? [],
    [data.nativeMeasurements.region.data]);
  // R06: a native macro in the ACTIVE protocol is a measurement of that model in this scope. Before
  // this, native rows were consulted only to build links to OTHER scopes and filtered to scopes
  // different from the current one, so a model whose only number was native here read "Chưa có số
  // đo" while its row sat in the catalog. Native rows carry no subset: the number is whole-protocol.
  const nativeMeasured = useMemo(
    () => nativeMeasuredIds(nativeRows, location.protocol),
    [nativeRows, location.protocol]);
  const measuredIds = useMemo(
    () => new Set([...compositionMeasuredIds, ...nativeMeasured]),
    [compositionMeasuredIds, nativeMeasured]);
  const active = models.filter((model) => !DEFERRED_MODELS[model.model_id]);
  const pending = active.filter((model) => !measuredIds.has(model.model_id));
  const deferred = models.filter((model) => DEFERRED_MODELS[model.model_id]);
  const measuredCount = models.length - pending.length - deferred.length;
  const runCount = entries.filter((entry) => entry.family === "composition" && entry.status !== "FAILED").length;
  // W-02: five different questions, five different numbers. The old header printed one "17" that a
  // reader could only interpret by guessing which of these it meant.
  //  * attempts: rows the API published, duplicates included — a rerun is an attempt, not a run.
  //  * measurements: native macro blocks with a number, the only rows this deployment MEASURED.
  //  * blockers: files that only signpost a blocker, i.e. models with a named reason and no number.
  const attemptCount = entries.length + nativeRows.length;
  // R06: the backend census (`counts.with_metrics`) counts counted AND artifact-backed rows. This
  // used to filter on `!rollup`, which is a different set: a roll-up replaced by an artifact is
  // still counted, and an unbacked claim is not. Same predicate, stated once.
  const measurementCount = nativeRows
    .filter((row) => row.metrics && row.counted !== false && row.artifact_backed === true).length;
  const blockerCount = data.nativeMeasurements.region.data?.counts.signposted_blockers ?? 0;

  const ordered = useMemo(() => [...models].sort((a, b) => {
    const priority = (id: string) => DEFERRED_MODELS[id] ? 2 : measuredIds.has(id) ? 0 : 1;
    return priority(a.model_id) - priority(b.model_id) || a.model_id.localeCompare(b.model_id);
  }), [models, measuredIds]);

  // EVERY row the API served that can hold a model's numbers, across all protocols and subsets: the
  // input `otherScopeFor` needs to find where a model absent HERE actually has numbers. Three
  // sources, because this deployment measures in three places: composition runs (CIRCUS subsets),
  // native measurements (macro numbers, unverified) and aggregate reports (FashionIQ protocols).
  const allRows = useMemo(() => {
    const composition = (data.compositionRuns.region.data ?? [])
      .map((row): ScopeRow => ({ ...(row as ScopeRow), family: "composition" }));
    const native = (data.nativeMeasurements.region.data?.measurements ?? [])
      .filter((row) => row.model_id && row.metrics && row.protocol_id)
      .map((row): ScopeRow => ({ model_id: row.model_id, protocol_id: row.protocol_id,
                                 subset: "", metrics: row.metrics as Record<string, unknown>,
                                 // A native macro is listed in the native table, NOT in the
                                 // benchmark table: the link has to name the family that holds it.
                                 family: "native" }));
    const aggregate = (data.aggregates.region.data ?? [])
      .filter((row) => row.model_id && row.protocol_id && !row.is_synthetic)
      .map((row): ScopeRow => ({ model_id: row.model_id, protocol_id: row.protocol_id,
                                 subset: "", metrics: (row.observed_metrics ?? {}) as Record<string, unknown>,
                                 family: "aggregate" }));
    return [...composition, ...native, ...aggregate];
  }, [data.compositionRuns.region.data, data.nativeMeasurements.region.data, data.aggregates.region.data]);

  const matching = ordered.filter((model) => {
    const text = `${model.model_id} ${model.method_name}`.toLowerCase();
    const measured = measuredIds.has(model.model_id);
    const isDeferred = !!DEFERRED_MODELS[model.model_id];
    return text.includes(search.toLowerCase())
      && (filter === "all" || (filter === "measured" && measured)
        || (filter === "pending" && !measured && !isDeferred) || (filter === "deferred" && isDeferred));
  });

  if (data.models.region.status === "error") {
    return <ErrorState message={data.models.region.error ?? "Không tải được danh sách model"} onRetry={data.models.retry} />;
  }
  if (!models.length && data.models.region.status === "loading") return <LoadingState label="Đang đọc danh sách mô hình…" />;
  // "Chưa có số đo" is a claim about the served runs. While those summaries are still in flight (or
  // failed) that claim would be unearned: every row would read "not measured" because nothing had
  // arrived yet. The inventory waits for the runs region instead of guessing.
  if (data.compositionRuns.region.status === "loading" && !data.compositionRuns.region.data) {
    return <LoadingState label="Đang đọc lượt chạy của protocol này…" />;
  }
  if (data.compositionRuns.region.status === "error") {
    return <ErrorState message={data.compositionRuns.region.error ?? "Không tải được danh sách lượt chạy"}
                       onRetry={data.compositionRuns.retry} />;
  }
  // R06: native rows also decide "measured" now, so the same reasoning applies to their region: a
  // native catalog still in flight would read as "no native number" and re-earn the wrong label.
  if (data.nativeMeasurements.region.status === "loading" && !data.nativeMeasurements.region.data) {
    return <LoadingState label="Đang đọc catalog số đo native…" />;
  }
  if (data.nativeMeasurements.region.status === "error") {
    return <ErrorState message={data.nativeMeasurements.region.error ?? "Không tải được catalog số đo native"}
                       onRetry={data.nativeMeasurements.retry} />;
  }

  return (
    <section className="model-inventory" aria-label="Toàn bộ mô hình" data-testid="model-inventory">
      <div className="inventory-stats" data-testid="inventory-stats">
        <div><strong data-testid="count-models">{models.length}</strong><span>Mô hình đăng ký</span></div>
        <div><strong data-testid="count-runs">{runCount}</strong><span>Lượt CIRCUS có dữ liệu ở phạm vi này</span></div>
        <div><strong data-testid="count-composition-rows">{entries.filter((entry) => entry.family === "composition").length}</strong><span>Dòng composition (mọi trạng thái)</span></div>
        <div><strong data-testid="count-attempts">{attemptCount}</strong><span>Lần thử ghi nhận (kể cả trùng)</span></div>
        <div><strong data-testid="count-measurements">{measurementCount}</strong><span>Số đo native có macro</span></div>
        <div><strong data-testid="count-blockers">{blockerCount}</strong><span>File chỉ ghi blocker</span></div>
        <div><strong data-testid="count-active-measured">{measuredCount}</strong><span>Có số đo trong phạm vi này</span></div>
        <div><strong data-testid="count-pending">{pending.length}</strong><span>Cần chạy hoặc xử lý blocker</span></div>
        <div><strong data-testid="count-deferred">{deferred.length}</strong><span>Tạm gác theo yêu cầu</span></div>
      </div>

      <div className="inventory-toolbar">
        <label className="search-field">Tìm mô hình
          <input type="search" name="model-search" autoComplete="off" value={search}
                 placeholder="Ví dụ: LIMN, ENCODER…" onChange={(event) => setSearch(event.target.value)} />
        </label>
        <label>Trạng thái
          <select value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="all">Tất cả mô hình ({models.length})</option>
            <option value="measured">Có số đo ({measuredCount})</option>
            <option value="pending">Cần xử lý ({pending.length})</option>
            <option value="deferred">Tạm gác ({deferred.length})</option>
          </select>
        </label>
        <p className="hint inventory-count" aria-live="polite">{matching.length} / {models.length} mô hình. Số lượt chạy được đếm riêng.</p>
      </div>

      <div className="table-wrap">
        <table className="inventory-table">
          <caption className="sr-only">Danh sách đầy đủ mô hình và trạng thái trong protocol, subset hiện tại</caption>
          <thead>
            <tr>
              <th scope="col">Mô hình / checkpoint</th>
              <th scope="col">Trạng thái</th>
              <th scope="col" className="num">MM R@10 · nDCG · đã chấm</th>
              <th scope="col">Ứng viên xếp hạng</th>
              <th scope="col">Chặn bởi</th>
            </tr>
          </thead>
          <tbody>
            {matching.map((model) => {
              const runs = entries.filter((entry) => entry.modelId === model.model_id);
              const usable = runs.filter((entry) => entry.status !== "FAILED" && entry.status !== "REFUSED" && !entry.composition?.is_failed);
              const selected = usable.find((entry) => location.runs.includes(entry.key));
              const entry = selected ?? (usable.length === 1 ? usable[0] : undefined);
              const run = entry?.composition;
              const counts = run ? scoredCounts(run, "multimodal") : null;
              const deferredReason = DEFERRED_MODELS[model.model_id];
              const checkpoints = model.checkpoints ?? [];
              const downloaded = checkpoints.filter((checkpoint) => checkpoint.checkpoint_downloaded).length;
              const blockers = [...new Set(checkpoints.flatMap((checkpoint) => checkpoint.blocking_reasons ?? []))];
              const missingScored = counts != null && counts.scored < counts.total;
              // R06: only an EXACT (model, protocol) native row measures this model here. A row that
              // names no protocol is not evidence about the selected scope (R01 in the UI layer), so
              // it gets its own label and its own link instead of being lent to whatever protocol is
              // on screen.
              const nativeHere = nativeRowsFor(nativeRows, model.model_id, location.protocol);
              const nativeUnattributed = nativeUnattributedFor(nativeRows, model.model_id);
              const nativeOnly = !entry && nativeHere.length > 0 && usable.length === 0;
              // Scopes this model DOES have numbers in, from composition rows, native measurements and
              // aggregate reports. Never let "no run here" be the whole answer when a real row exists
              // elsewhere: the reader gets the scope, its size, and a link that opens it.
              const elsewhere = crossScopeAllowed({ hasEntry: !!entry, deferred: !!deferredReason, nativeOnly })
                ? otherScopeFor(allRows, model.model_id).filter((scope) =>
                    scope.protocol !== location.protocol
                    || (protocolHasSubsets(location.protocol) ? scope.subset !== location.subset : false))
                : [];
              // A deferred row still has real numbers somewhere: the counted native rows name their
              // own protocol, so the row links to the table that lists them (never to a scope it
              // does not have).
              const nativeScopes = deferredReason
                ? otherScopeFor(allRows, model.model_id).filter((scope) => scope.family === "native")
                : [];
              // One decision, in `scope.ts`, so the label and the counter above cannot disagree.
              const status = inventoryStatus({
                deferred: !!deferredReason,
                hasRun: !!run,
                runHasAggregateMetrics: !!entry?.capabilities.flags.aggregate_metrics,
                runMissingScored: missingScored,
                entryFamily: entry?.family ?? null,
                entryHasAggregateMetrics: !!entry?.capabilities.flags.aggregate_metrics,
                nativeInScope: nativeHere.length,
                nativeUnattributed: nativeUnattributed.length,
                usableCount: usable.length,
                measuredEntryCount: usable.filter((item) => item.capabilities.flags.aggregate_metrics).length,
                elsewhereCount: elsewhere.length,
                checkpointDownloaded: downloaded > 0,
              });
              const openNative = () => onGoto({ area: "evaluation", view: "native", runs: [], run: "", query: "", offset: 0 });
              const openRun = () => {
                if (nativeOnly || nativeUnattributed.length) {
                  openNative();
                  return;
                }
                if (!entry) return;
                onGoto(entry.family === "composition"
                  ? { area: "samples", view: "explorer", runs: [entry.key], run: entry.runId, query: "", q: "", offset: 0 }
                  : { area: "evaluation", view: "benchmark", runs: [entry.key], run: entry.runId });
              };
              return (
                <tr key={model.model_id} data-testid={`model-row-${model.model_id}`}>
                  <td>
                    <span className="cell-strong">{model.method_name || model.model_id}</span>
                    <span className="cell-sub">{entry?.checkpointId ?? model.model_id}</span>
                    {entry
                      ? <button className="inspect-button" type="button" onClick={openRun}>
                          {entry.family === "composition" ? "Xem query" : "Xem kết quả"}
                        </button>
                      : nativeOnly
                        ? <button className="inspect-button" type="button" onClick={openRun}
                                  data-testid={`native-open-${model.model_id}`}>
                            Xem số đo native
                          </button>
                      : deferredReason
                        ? <div className="cell-sub" style={{ marginTop: 6 }} data-testid={`deferred-${model.model_id}`}>
                            <span>{deferredReason}</span>
                            {/* The deferral is a campaign decision about THIS scope, not a statement
                                that the model has no number. The counted native rows live under their
                                own protocol, so the link names it and opens the table that lists them. */}
                            {nativeScopes.length > 0 && (
                              <div className="scope-links">
                                {nativeScopes.map((scope) => (
                                  <button key={`${scope.protocol}:${scope.family ?? ""}`} type="button" className="ghost"
                                          style={{ marginTop: 6 }}
                                          data-testid={`deferred-native-${model.model_id}-${scope.protocol}`}
                                          onClick={() => onGoto({ area: "evaluation", view: "native",
                                                                 protocol: scope.protocol, runs: [], run: "",
                                                                 query: "", offset: 0 })}>
                                    Mở số đo native · {protocolLabel(scope.protocol)}
                                    {protocolHasSubsets(scope.protocol) && scope.subset
                                      ? ` · ${SUBSET_LABEL[scope.subset as keyof typeof SUBSET_LABEL] ?? scope.subset}` : ""}
                                    {" · "}{fmtInt(scope.rows)}
                                  </button>
                                ))}
                              </div>
                            )}
                          </div>
                      : nativeUnattributed.length || elsewhere.length
                        // R06-follow-up: the two facts are independent, so both are printed. The
                        // label says this file's scope is unknown; the links say where the model's
                        // ATTRIBUTED rows are. One used to hide the other.
                        ? <>
                            {nativeUnattributed.length > 0 && (
                              <div className="cell-sub" style={{ marginTop: 6 }} data-testid={`native-unattributed-${model.model_id}`}>
                                <span>
                                  {fmtInt(nativeUnattributed.length)} số đo native nhưng file không ghi protocol —
                                  không quy được cho phạm vi đang chọn.
                                </span>
                                <button type="button" className="ghost" style={{ marginTop: 6 }}
                                        data-testid={`native-unattributed-open-${model.model_id}`} onClick={openNative}>
                                  Xem trong bảng native
                                </button>
                              </div>
                            )}
                            {elsewhere.length > 0 && (
                              <div className="cell-sub" data-testid={`other-scope-${model.model_id}`} style={{ marginTop: 6 }}>
                                <span>Có kết quả ở {elsewhere.length} phạm vi khác — mở đúng phạm vi đó:</span>
                                <div className="scope-links">
                                  {elsewhere.map((scope) => (
                                    <button key={`${scope.protocol}:${scope.subset}:${scope.family ?? ""}`} type="button"
                                            className="ghost"
                                            data-testid={`other-scope-${model.model_id}-${scope.protocol}-${scope.family ?? "composition"}-${scope.subset || "all"}`}
                                            // The view that actually LISTS these rows. A composition
                                            // subset opens the composition table; a native macro opens
                                            // the native table (which the benchmark table never lists);
                                            // an aggregate report opens the benchmark table. Opening
                                            // "the protocol" was not enough: two families can share one
                                            // protocol id and only one of them has a row there (W-02).
                                            onClick={() => onGoto(scope.family === "native"
                                              ? { area: "evaluation", view: "native", protocol: scope.protocol,
                                                  runs: [], run: "", query: "", offset: 0 }
                                              : scope.subset
                                                ? { area: "evaluation", view: "composition", protocol: scope.protocol,
                                                    subset: scope.subset as Location["subset"], runs: [], run: "", query: "", offset: 0 }
                                                : { area: "evaluation", view: "benchmark", protocol: scope.protocol,
                                                    runs: [], run: "", query: "", offset: 0 })}>
                                      {protocolLabel(scope.protocol)}
                                      {protocolHasSubsets(scope.protocol) && scope.subset
                                        ? ` · ${SUBSET_LABEL[scope.subset as keyof typeof SUBSET_LABEL] ?? scope.subset}` : ""}
                                      {scope.family === "native" ? " · số đo native" : ""}
                                      {" · "}{fmtInt(scope.rows)} {scope.subset ? "lượt" : "số đo (toàn phạm vi)"}
                                    </button>
                                  ))}
                                </div>
                              </div>
                            )}
                          </>
                        : <div className="cell-sub" data-testid={`todo-${model.model_id}`} style={{ marginTop: 6 }}>
                            <span>{usable.length > 1
                              ? `${usable.length} lượt trong phạm vi này; chọn một lượt ở khu vực Lượt chạy.`
                              : nextStep(model)}</span>
                            <button type="button" style={{ marginTop: 6 }}
                                    onClick={() => onGoto({ area: "experiments", view: "runs" })}>Mở các lượt chạy</button>
                          </div>}
                  </td>
                  <td>
                    <Badge tone={deferredReason ? "neutral" : run && !missingScored ? "ok" : nativeOnly ? "info" : "warn"}>{status}</Badge>
                    <span className="cell-sub">
                      {deferredReason
                        ? `${deferredReason}${nativeHere.length
                            ? ` · ${fmtInt(nativeHere.length)} số đo native ở phạm vi này`
                            : ""}`
                        : entry
                          ? (missingScored ? `${fmtInt(counts?.unscored)} / ${fmtInt(counts?.total)} query chưa chấm` : "mọi query trong mẫu đều có hạng")
                          : nativeOnly
                            ? `${fmtInt(nativeHere.length)} số đo native · macro toàn protocol, chưa xác minh`
                            : nativeUnattributed.length
                              ? `${fmtInt(nativeUnattributed.length)} số đo native không ghi protocol · chưa quy được cho phạm vi này`
                              : `${downloaded} / ${checkpoints.length} checkpoint đã có · ${model.reproduction_status ?? "chưa rõ"}`}
                    </span>
                  </td>
                  <td className="num">
                    {run
                      ? <>
                          {fmtPercentValue(run.metrics?.recall_at_10_multimodal ?? run.metrics?.multimodal?.recall_at_10)}
                          {" · "}{fmtNdcg(run.metrics?.ndcg_multimodal ?? run.metrics?.multimodal?.ndcg_full)}
                          <span className="cell-sub num">{counts ? fmtDenominator(counts.scored, counts.total) : "—"}</span>
                        </>
                      : "—"}
                  </td>
                  <td>
                    {entry
                      ? <span className={entry.capabilities.flags.candidate_topk ? "hint-ok" : "hint-bad"}>
                          {entry.capabilities.flags.candidate_topk ? "Đã xuất ứng viên" : "Cần xuất Top-K"}
                        </span>
                      : <span className="hint">{deferredReason ? "Ngoài đợt chạy này"
                          : nativeOnly ? "Macro native — không có ứng viên"
                          : nativeUnattributed.length ? "Macro native — không ghi protocol"
                          : "Chưa có kết quả"}</span>}
                  </td>
                  <td>
                    {/* Registry's own reasons, verbatim; the summary is the count. An empty list says the
                        registry recorded none, which is a different fact from "no blocker exists". */}
                    {blockers.length === 0
                      ? <span className="hint">không có lý do chặn được ghi</span>
                      : <Detail summary={`${blockers.length} lý do`} testId={`blockers-${model.model_id}`}>
                          <ul style={{ margin: 0, paddingLeft: 18 }}>
                            {blockers.map((reason) => <li key={reason}><small>{reason}</small></li>)}
                          </ul>
                        </Detail>}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {!matching.length && (
        <p className="empty" role="status">
          Không có mô hình khớp bộ lọc.
          <button type="button" onClick={() => { setSearch(""); setFilter("all"); }}>Xóa bộ lọc</button>
        </p>
      )}
      <p className="hint">
        Cột MM dành cho lượt CIRCUS ở protocol/subset đang chọn; số đo native xem trong chi tiết.
        “Có số đo” chỉ nói về lượt CIRCUS/composition đã chạy; “Có dữ liệu native” là macro trong protocol
        đang chọn, mở trong bảng Số đo native và chưa thay thế kiểm chứng pipeline (không có subset: số đo
        phủ toàn protocol). Cả hai đều được tính là đã đo trong phạm vi này.
        CSMCIR và SPRC vẫn có trong danh sách để theo dõi phạm vi.
      </p>
    </section>
  );
}

/** The next real step for a model with no run here, from what the registry actually declares. */
function nextStep(model: Model): string {
  const checkpoints = model.checkpoints ?? [];
  if (!checkpoints.length) return "Registry không khai báo checkpoint nào cho mô hình này.";
  const downloaded = checkpoints.filter((checkpoint) => checkpoint.checkpoint_downloaded);
  if (!downloaded.length) return "Chưa có checkpoint nào được tải về host; cần tải hoặc trỏ đúng đường dẫn.";
  const ready = downloaded.filter((checkpoint) => checkpoint.adapter_command_ready);
  if (!ready.length) return `Có ${downloaded.length} checkpoint đã tải nhưng chưa adapter nào ở trạng thái chạy được.`;
  const verified = ready.filter((checkpoint) => checkpoint.runtime_verified);
  if (!verified.length) return "Adapter đã khai báo nhưng runtime chưa được xác minh trên host này.";
  return "Checkpoint và runtime đã sẵn sàng nhưng chưa có lượt evaluation ở protocol/subset này.";
}
