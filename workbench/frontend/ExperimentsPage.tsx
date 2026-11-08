import type { WorkbenchData, RunCatalogEntry } from "./data";
import { protocolLabel } from "./data";
import type { Location } from "./routes";
import { Badge, Card, Detail, EmptyState, ErrorState, KeyValue, LoadingState, SortHeader, sortRows, useSort } from "./ui";
import { fmtDenominator, fmtDateStamp, fmtInt, fmtNdcg, fmtPercentValue, shortId } from "./format";
import { duplicateLabel, scoredCounts } from "./catalog";
import { SUBSET_LABEL } from "./routes";

/** Thí nghiệm: the runs and models that exist, what each can answer, and the job queue.
 *
 *  Runs from all three families are listed together but never merged: a CIRCUS release run, a
 *  per-query index run and an aggregate benchmark report are three different artifacts, and the
 *  "Khả năng" column says which questions each can answer. Blocked models show their blocking
 *  reason and a disabled queue control with the reason attached, not a silent no-op.
 */
export function ExperimentsPage({ data, location, view, onSelectRun, onGoto }: {
  data: WorkbenchData;
  location: Location;
  view: string;
  onSelectRun: (entry: RunCatalogEntry, additive?: boolean) => void;
  onGoto: (patch: Partial<Location>) => void;
}) {
  if (view === "models") return <ModelsView data={data} />;
  if (view === "queue") return <QueueView data={data} />;
  return <RunsView data={data} location={location} onSelectRun={onSelectRun} onGoto={onGoto} />;
}

function RunsView({ data, location, onSelectRun, onGoto }: {
  data: WorkbenchData;
  location: Location;
  onSelectRun: (entry: RunCatalogEntry, additive?: boolean) => void;
  onGoto: (patch: Partial<Location>) => void;
}) {
  const { sort, toggle } = useSort({ key: "model", dir: "asc" });
  const catalog = data.catalog;
  const selected = new Set(location.runs);
  const rows = sortRows(catalog, sort, (entry, key) => {
    switch (key) {
      case "model": return entry.modelId;
      case "status": return entry.status;
      case "ndcg": return entry.composition?.metrics?.ndcg_multimodal ?? null;
      case "covered": {
        const counts = entry.composition ? scoredCounts(entry.composition, "multimodal") : null;
        return counts ? counts.coverage : null;
      }
      default: return entry.label;
    }
  });
  const failedCount = catalog.filter((entry) => entry.status === "FAILED").length;
  const capabilityMissing = catalog.filter((entry) => !entry.capabilities.flags.per_query_target_ranks).length;

  if (data.compositionRuns.region.status === "error" && data.indexRuns.region.status === "error") {
    return <ErrorState message={data.compositionRuns.region.error ?? ""} onRetry={data.refreshAll} />;
  }

  return (
    <div className="stack" data-testid="area-experiments">
      <Card
        title="Lượt chạy"
        subtitle={`${fmtInt(catalog.length)} lượt cho ${protocolLabel(location.protocol)} · subset ${SUBSET_LABEL[location.subset]} · ${failedCount} lượt lỗi · ${capabilityMissing} lượt không có rank từng query`}
        actions={(
          <>
            <button type="button" onClick={data.refreshAll}>Làm mới</button>
            <button type="button" onClick={() => onGoto({ area: "evaluation", view: "composition" })}>Bảng so sánh</button>
          </>
        )}
      >
        {data.compositionRuns.region.status === "loading" && !data.compositionRuns.region.data
          ? <LoadingState label="Đang tải danh sách lượt…" />
          : rows.length === 0
            ? <EmptyState kind="no-results" detail="Không có lượt nào cho protocol và subset đang chọn."
                          next={<button type="button" onClick={() => onGoto({ subset: "full" })}>Đổi sang subset Full</button>} />
            : (
              <div className="table-wrap">
                <table>
                  <caption className="sr-only">Các lượt chạy</caption>
                  <thead>
                    <tr>
                      <th scope="col" style={{ width: 34 }} aria-label="Chọn" />
                      <SortHeader label="Mô hình / checkpoint" column="model" sort={sort} onSort={toggle} />
                      <th scope="col">Nguồn</th>
                      <SortHeader label="Trạng thái" column="status" sort={sort} onSort={toggle} />
                      <SortHeader label="MM nDCG" column="ndcg" sort={sort} onSort={toggle} numeric unit="[0,1]" />
                      <SortHeader label="Đã chấm" column="covered" sort={sort} onSort={toggle} numeric />
                      <th scope="col">Khả năng</th>
                      <th scope="col">Chi tiết</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((entry) => {
                      const isSelected = selected.has(entry.key);
                      const counts = entry.composition ? scoredCounts(entry.composition, "multimodal") : null;
                      const disabledReason = entry.capabilities.reasons.per_query_target_ranks;
                      return (
                        <tr key={entry.key} aria-selected={isSelected} data-testid={`run-row-${entry.key}`}>
                          <td>
                            <input type="checkbox" checked={isSelected}
                                   disabled={!entry.capabilities.flags.per_query_target_ranks}
                                   title={disabledReason}
                                   aria-label={`Chọn ${entry.label}`}
                                   data-testid={`run-option-${entry.key}`}
                                   onChange={(event) => onSelectRun(entry, event.target.checked)} />
                          </td>
                          <td>
                            <span className="cell-strong">{entry.modelId}</span>
                            <span className="cell-sub">
                              {entry.checkpointId} · {entry.family === "composition" ? "CIRCUS release" : entry.family === "index" ? "chỉ mục per-query" : "báo cáo tổng hợp"}
                              {entry.isTransfer ? " · TRANSFER" : ""}
                            </span>
                          </td>
                          <td><Badge tone={entry.family === "composition" ? "info" : entry.family === "index" ? "neutral" : "warn"}>
                            {entry.family === "composition" ? "release" : entry.family === "index" ? "index" : "aggregate"}
                          </Badge></td>
                          <td>
                            {duplicateLabel(entry.duplicateStatus, entry.duplicateAttempts) && (
                              <span className="cell-sub" data-testid={`duplicate-${entry.key}`}>
                                {duplicateLabel(entry.duplicateStatus, entry.duplicateAttempts)}
                              </span>
                            )}
                            <Badge tone={entry.status === "FAILED" || entry.status === "REFUSED" ? "bad"
                              : ["COMPLETED", "PARTIAL", "certified", "CERTIFIED"].includes(entry.status) ? "ok"
                              : entry.provenance.bucket === "development" ? "warn" : "neutral"}>
                              {entry.status}
                            </Badge>
                            {entry.provenance.bucket === "development" && <Badge tone="warn" title="Kết quả không dùng cho kết luận nghiên cứu">dev</Badge>}
                          </td>
                          <td className="num">{entry.composition?.is_failed ? "—" : fmtNdcg(entry.composition?.metrics?.ndcg_multimodal)}</td>
                          <td className="num">{counts ? fmtDenominator(counts.scored, counts.total) : "—"}</td>
                          <td>
                            <span className="row" style={{ gap: 4 }}>
                              <Badge tone={entry.capabilities.flags.aggregate_metrics ? "ok" : "neutral"}
                                     title={entry.capabilities.reasons.aggregate_metrics}>aggregate</Badge>
                              <Badge tone={entry.capabilities.flags.per_query_target_ranks ? "ok" : "bad"}
                                     title={entry.capabilities.reasons.per_query_target_ranks}>rank</Badge>
                              <Badge tone={entry.capabilities.flags.candidate_topk ? "ok" : "warn"}
                                     title={entry.capabilities.statuses?.candidate_topk
                                       ? `${entry.capabilities.statuses.candidate_topk} — ${entry.capabilities.reasons.candidate_topk}`
                                       : entry.capabilities.reasons.candidate_topk}>top-k</Badge>
                              <Badge tone={entry.capabilities.flags.image_availability ? "ok" : "neutral"}
                                     title={entry.capabilities.reasons.image_availability}>ảnh</Badge>
                            </span>
                          </td>
                          <td>
                            <RunDetail entry={entry} />
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
      </Card>
      <p className="hint" data-testid="selected-run-count">
        Đã chọn <b>{location.runs.length}</b> lượt. Phân tích lỗi và so sánh dùng đúng các lượt này; mọi
        phép so sánh đều nêu cohort và mẫu số.
      </p>
    </div>
  );
}

function RunDetail({ entry }: { entry: RunCatalogEntry }) {
  const run = entry.composition;
  if (entry.family === "aggregate") {
    const agg = entry.aggregate!;
    return (
      <Detail summary="Chi tiết">
        <KeyValue rows={[
          ["metric", <span key="m">R@10 {fmtPercentValue(agg.observed_metrics?.r10)} · R@50 {fmtPercentValue(agg.observed_metrics?.r50)} · mean {fmtPercentValue(agg.observed_metrics?.mean)}</span>],
          ["per-query", <span key="p"><code>{agg.per_query_state ?? "unknown"}</code> — {entry.capabilities.reasons.per_query_target_ranks}</span>],
          ["receipt", <span key="v">{agg.verification?.state ?? "unknown"}{(agg.verification?.reasons ?? []).slice(0, 2).join("; ")}</span>],
          ["run_id", <code key="r">{agg.run_id}</code>],
        ]} />
      </Detail>
    );
  }
  if (entry.family === "index") {
    const run = entry.indexRun!;
    return (
      <Detail summary="Chi tiết">
        <KeyValue rows={[
          ["data_kind", run.data_kind],
          ["evidence", run.evidence_status ?? "undetermined"],
          ["top_k_saved", fmtInt(run.top_k_saved)],
          ["gallery", fmtInt(run.gallery_size)],
          ["run_id", <code key="r">{run.run_id}</code>],
        ]} />
      </Detail>
    );
  }
  if (!run) return null;
  const counts = scoredCounts(run, "multimodal");
  return (
    <Detail summary="Chi tiết">
      <KeyValue rows={[
        ["trạng thái", run.execution_status || "unknown"],
        ["subset có", (run.available_subsets ?? ["full", "sf", "validated"]).join(", ")],
        ["gallery", fmtInt(run.gallery_count)],
        ["query", fmtInt(run.query_count)],
        ["đã chấm MM", fmtDenominator(counts.scored, counts.total)],
        ["unscored", counts.unscored === 0
          ? "0 (mọi query đều có hạng chính xác)"
          : `${fmtInt(counts.unscored)} query không có hạng chính xác — lỗi pipeline/ánh xạ, không phải lỗi truy hồi`
            + (counts.unscoredByKind
              ? ` (${Object.entries(counts.unscoredByKind).map(([kind, n]) => `${kind} ${fmtInt(n)}`).join(", ")})`
              : "")],
        ["coverage", fmtPercentValue((counts.coverage ?? 0) * 100)],
        ["danh tính measurement", entry.duplicateStatus
          ? `${entry.duplicateStatus}${entry.duplicateAttempts ? ` · ${fmtInt(entry.duplicateAttempts)} lần thử` : ""}`
          : "chưa được backend công bố cho lượt này"],
        ["chuyển giao", run.checkpoint_transfer ? String(run.checkpoint_transfer) : "không (huấn luyện trên FashionIQ)"],
        ["checkpoint", <code key="c">{run.checkpoint_identity ?? run.checkpoint_id}</code>],
        ["interpreter", run.interpreter_verification_status ?? "unknown"],
        ["tái lập", run.reproduction_status ?? "unknown"],
        ["run_id", <code key="r">{run.run_id}</code>],
        ["lý do unscored", run.metrics?.gap_unavailable_reason ?? "—"],
      ]} />
    </Detail>
  );
}

function ModelsView({ data }: { data: WorkbenchData }) {
  const region = data.models.region;
  if (region.status === "error") return <ErrorState message={region.error ?? ""} onRetry={data.models.retry} />;
  if (region.status === "loading" && !region.data) return <LoadingState label="Đang tải registry…" />;
  const models = region.data ?? [];
  return (
    <div className="stack" data-testid="area-models">
      <Card title="Mô hình trong registry" subtitle={`${fmtInt(models.length)} entry · trạng thái tái lập và lý do chặn`}>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th scope="col">Mô hình</th>
                <th scope="col">Protocol native</th>
                <th scope="col">Tái lập</th>
                <th scope="col">Checkpoint</th>
                <th scope="col">Chặn bởi</th>
              </tr>
            </thead>
            <tbody>
              {models.map((model) => {
                const checkpoints = model.checkpoints ?? [];
                const runnable = checkpoints.filter((checkpoint) => checkpoint.runnable);
                const blockers = [...new Set(checkpoints.flatMap((checkpoint) => checkpoint.blocking_reasons ?? []))];
                return (
                  <tr key={model.model_id} data-testid={`model-row-${model.model_id}`}>
                    <td>
                      <span className="cell-strong">{model.method_name || model.model_id}</span>
                      <span className="cell-sub">{model.model_id} · {model.paper_title ?? "—"}</span>
                    </td>
                    <td><code>{model.native_protocol ?? "—"}</code></td>
                    <td>
                      <Badge tone={(model.reproduction_status || "").startsWith("BLOCKED") ? "warn" : model.reproduction_status === "NOT_RUN" ? "neutral" : "ok"}>
                        {model.reproduction_status ?? "unknown"}
                      </Badge>
                    </td>
                    <td className="num">{fmtInt(runnable.length)} / {fmtInt(checkpoints.length)}</td>
                    <td>
                      {blockers.length === 0
                        ? <span className="hint">không có lý do chặn được ghi</span>
                        : <Detail summary={`${blockers.length} lý do`}>
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
      </Card>
    </div>
  );
}

function QueueView({ data }: { data: WorkbenchData }) {
  const region = data.jobs.region;
  const jobs = region.data ?? [];
  return (
    <div className="stack">
      <Card title="Hàng đợi job" subtitle="Chỉ lệnh đã preflight mới vào hàng đợi; mỗi lúc một evaluator."
            actions={<button type="button" onClick={data.jobs.retry}>Làm mới</button>}>
        {region.status === "error"
          ? <ErrorState message={region.error ?? ""} onRetry={data.jobs.retry} />
          : region.status === "loading" && !region.data
            ? <LoadingState />
            : jobs.length === 0
              ? <EmptyState kind="no-results" detail="Chưa có job nào được xếp hàng." />
              : (
                <div className="stack">
                  {jobs.map((job) => (
                    <article key={job.job_id} className="card" style={{ padding: 10 }}>
                      <div className="row">
                        <b>{job.model_id} / {job.checkpoint_id}</b>
                        <Badge tone={job.status === "completed" ? "ok" : job.status === "failed" ? "bad" : job.status === "blocked" ? "warn" : "info"}>{job.status}</Badge>
                        <span className="hint">{fmtDateStamp(job.created_at)}</span>
                        {["queued", "preparing", "running"].includes(job.status) && (
                          <button type="button" className="ghost" style={{ marginLeft: "auto" }}
                                  onClick={() => void fetch(`/api/jobs/${job.job_id}/cancel`, { method: "POST" }).then(() => data.jobs.retry())}>
                            Huỷ
                          </button>
                        )}
                      </div>
                      {job.error && <pre style={{ marginTop: 8 }}>{job.error}</pre>}
                      <Detail summary="Lệnh và log">
                        <pre>{job.command.join(" ")}</pre>
                        {job.stderr_tail && <pre>{job.stderr_tail}</pre>}
                      </Detail>
                    </article>
                  ))}
                </div>
              )}
      </Card>
    </div>
  );
}
