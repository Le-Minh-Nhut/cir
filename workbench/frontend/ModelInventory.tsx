import { useMemo, useState } from "react";
import { protocolLabel, type WorkbenchData } from "./data";
import type { Location } from "./routes";
import type { Model } from "./types";
import { Badge, Detail, ErrorState, LoadingState } from "./ui";
import { fmtDenominator, fmtInt, fmtNdcg, fmtPercentValue } from "./format";
import { scoredCounts } from "./catalog";
import { otherScopeFor, protocolHasSubsets, type ScopeRow } from "./scope";
import { SUBSET_LABEL } from "./routes";

/** Operator scope for this review. Not a registry fact and not a measurement: the two models whose
 *  auxiliary data or checkpoints are missing stay LISTED with this reason attached, so "why is it
 *  not in the numbers" has an answer on the row instead of an absence.
 *  ponytail: keyed by model_id, so it must be revised when the registry changes; the honest general
 *  form is a `deferred_reason` field served per model. */
export const DEFERRED_MODELS: Record<string, string> = {
  csmcir: "Tạm gác theo yêu cầu: thiếu dữ liệu phụ trợ",
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
  const measuredIds = useMemo(() => new Set(entries
    .filter((entry) => !entry.composition?.is_failed && entry.status !== "FAILED" && entry.capabilities.flags.aggregate_metrics)
    .map((entry) => entry.modelId)), [entries]);
  const active = models.filter((model) => !DEFERRED_MODELS[model.model_id]);
  const pending = active.filter((model) => !measuredIds.has(model.model_id));
  const deferred = models.filter((model) => DEFERRED_MODELS[model.model_id]);
  const runCount = entries.filter((entry) => entry.family === "composition" && entry.status !== "FAILED").length;

  const ordered = useMemo(() => [...models].sort((a, b) => {
    const priority = (id: string) => DEFERRED_MODELS[id] ? 2 : measuredIds.has(id) ? 0 : 1;
    return priority(a.model_id) - priority(b.model_id) || a.model_id.localeCompare(b.model_id);
  }), [models, measuredIds]);

  // EVERY row the API served that can hold a model's numbers, across all protocols and subsets: the
  // input `otherScopeFor` needs to find where a model absent HERE actually has numbers. Three
  // sources, because this deployment measures in three places: composition runs (CIRCUS subsets),
  // native measurements (macro numbers, unverified) and aggregate reports (FashionIQ protocols).
  const allRows = useMemo(() => {
    const composition = (data.compositionRuns.region.data ?? []) as ScopeRow[];
    const native = (data.nativeMeasurements.region.data?.measurements ?? [])
      .filter((row) => row.model_id && row.metrics && row.protocol_id)
      .map((row): ScopeRow => ({ model_id: row.model_id, protocol_id: row.protocol_id,
                                 subset: "", metrics: row.metrics as Record<string, unknown> }));
    const aggregate = (data.aggregates.region.data ?? [])
      .filter((row) => row.model_id && row.protocol_id && !row.is_synthetic)
      .map((row): ScopeRow => ({ model_id: row.model_id, protocol_id: row.protocol_id,
                                 subset: "", metrics: (row.observed_metrics ?? {}) as Record<string, unknown> }));
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

  return (
    <section className="model-inventory" aria-label="Toàn bộ mô hình" data-testid="model-inventory">
      <div className="inventory-stats" data-testid="inventory-stats">
        <div><strong>{models.length}</strong><span>Mô hình đăng ký</span></div>
        <div><strong>{active.filter((model) => measuredIds.has(model.model_id)).length}</strong><span>Có số đo trong phạm vi này</span></div>
        <div><strong>{runCount}</strong><span>Lượt CIRCUS có dữ liệu ở phạm vi này</span></div>
        <div><strong>{pending.length}</strong><span>Cần chạy hoặc xử lý blocker</span></div>
        <div><strong>{deferred.length}</strong><span>Tạm gác theo yêu cầu</span></div>
      </div>

      <div className="inventory-toolbar">
        <label className="search-field">Tìm mô hình
          <input type="search" name="model-search" autoComplete="off" value={search}
                 placeholder="Ví dụ: LIMN, ENCODER…" onChange={(event) => setSearch(event.target.value)} />
        </label>
        <label>Trạng thái
          <select value={filter} onChange={(event) => setFilter(event.target.value)}>
            <option value="all">Tất cả mô hình ({models.length})</option>
            <option value="measured">Có số đo ({models.length - pending.length - deferred.length})</option>
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
              // Scopes this model DOES have numbers in, from composition rows, native measurements and
              // aggregate reports. Never let "no run here" be the whole answer when a real row exists
              // elsewhere: the reader gets the scope, its size, and a link that opens it.
              const elsewhere = entry || deferredReason
                ? []
                : otherScopeFor(allRows, model.model_id).filter((scope) =>
                    scope.protocol !== location.protocol
                    || (protocolHasSubsets(location.protocol) ? scope.subset !== location.subset : false));
              const status = deferredReason ? "Tạm gác"
                : run ? (missingScored ? "Chấm thiếu" : "Có số đo")
                : entry ? "Có dữ liệu native"
                : usable.length > 1 ? `${usable.length} lượt cần chọn`
                : elsewhere.length ? "Có kết quả ở scope khác"
                : downloaded ? "Chưa có số đo" : "Chưa có checkpoint xác nhận";
              const openRun = () => {
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
                      : deferredReason
                        ? <span className="cell-sub">Ngoài đợt chạy này</span>
                        : elsewhere.length
                          ? <div className="cell-sub" data-testid={`other-scope-${model.model_id}`} style={{ marginTop: 6 }}>
                              <span>Có kết quả ở {elsewhere.length} phạm vi khác — mở đúng phạm vi đó:</span>
                              <div className="scope-links">
                                {elsewhere.map((scope) => (
                                  <button key={`${scope.protocol}:${scope.subset}`} type="button"
                                          className="ghost"
                                          data-testid={`other-scope-${model.model_id}-${scope.protocol}-${scope.subset || "all"}`}
                                          // A composition subset opens the composition table; an empty
                                          // subset means the rows came from a source without subsets
                                          // (native measurements / aggregate reports), so it opens the
                                          // view that actually lists them.
                                          onClick={() => onGoto(scope.subset
                                            ? { area: "evaluation", view: "composition", protocol: scope.protocol,
                                                subset: scope.subset as Location["subset"], runs: [], run: "", query: "", offset: 0 }
                                            : { area: "evaluation", view: "benchmark", protocol: scope.protocol,
                                                runs: [], run: "", query: "", offset: 0 })}>
                                    {protocolLabel(scope.protocol)}
                                    {protocolHasSubsets(scope.protocol) && scope.subset
                                      ? ` · ${SUBSET_LABEL[scope.subset as keyof typeof SUBSET_LABEL] ?? scope.subset}` : ""}
                                    {" · "}{fmtInt(scope.rows)} {scope.subset ? "lượt" : "số đo (toàn phạm vi)"}
                                  </button>
                                ))}
                              </div>
                            </div>
                          : <div className="cell-sub" data-testid={`todo-${model.model_id}`} style={{ marginTop: 6 }}>
                              <span>{usable.length > 1
                                ? `${usable.length} lượt trong phạm vi này; chọn một lượt ở khu vực Lượt chạy.`
                                : nextStep(model)}</span>
                              <button type="button" style={{ marginTop: 6 }}
                                      onClick={() => onGoto({ area: "experiments", view: "runs" })}>Mở các lượt chạy</button>
                            </div>}
                  </td>
                  <td>
                    <Badge tone={deferredReason ? "neutral" : run && !missingScored ? "ok" : "warn"}>{status}</Badge>
                    <span className="cell-sub">
                      {deferredReason
                        ?? (entry
                          ? (missingScored ? `${fmtInt(counts?.unscored)} / ${fmtInt(counts?.total)} query chưa chấm` : "mọi query trong mẫu đều có hạng")
                          : `${downloaded} / ${checkpoints.length} checkpoint đã có · ${model.reproduction_status ?? "chưa rõ"}`)}
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
                      : <span className="hint">{deferredReason ? "Ngoài đợt chạy này" : "Chưa có kết quả"}</span>}
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
        “Có số đo” mô tả dữ liệu đã ghi, chưa thay thế kiểm chứng pipeline.
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
