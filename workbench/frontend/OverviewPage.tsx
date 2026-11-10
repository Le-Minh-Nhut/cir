import type { WorkbenchData, RunCatalogEntry } from "./data";
import { protocolLabel } from "./data";
import { Badge, Card, EmptyState, ErrorState, KeyValue, LoadingState, Progress, Detail } from "./ui";
import { fmtDateStamp, fmtDenominator, fmtFractionPercent, fmtGap, fmtInt, fmtNdcg, fmtPercentValue, fmtRank } from "./format";
import { compositionGap, scoredCounts } from "./catalog";
import { protocolHasSubsets, scopeCounts } from "./scope";
import type { Location } from "./routes";
import { SUBSET_LABEL } from "./routes";
import { ModelInventory } from "./ModelInventory";

/** Tổng quan: what results exist, what does not, and how to continue.
 *
 *  Counts come from the two served summaries (`/api/composition/capabilities?subset=…` and
 *  `/api/evidence/summary`), never from a literal and never from one source standing in for another.
 *
 *  The previous dashboard printed "0 experiment runs" next to a page full of CIRCUS metrics, and
 *  announced "aggregate-only" over a context that did have per-query ranks. Both came from counting
 *  `/api/runs` only. Here every count is stated with the source that produced it, and the banner is
 *  about the SELECTION, not the deployment.
 */
export function OverviewPage({ data, location, onOpenRun, onGoto }: {
  data: WorkbenchData;
  location: Location;
  onOpenRun: (entry: RunCatalogEntry) => void;
  onGoto: (patch: Partial<Location>) => void;
}) {
  const composition = data.compositionRuns.region;
  const models = data.models.region;
  const capabilities = data.compositionCapabilitiesSummary.region;
  const evidence = data.evidenceSummary.region;
  const caps = capabilities.data;
  const ev = evidence.data;

  const compositionRuns = composition.data ?? [];
  // One protocol, one subset: the header must never say "FashionIQ Original" over CIRCUS rows.
  const activeSubset = compositionRuns.filter((run) => run.subset === location.subset && run.protocol_id === location.protocol);
  const failed = activeSubset.filter((run) => run.is_failed || run.execution_status === "FAILED");

  const browsable = data.catalog.filter((entry) => entry.family === "composition"
    && entry.protocolId === location.protocol && entry.subset === location.subset
    && entry.provenance.bucket !== "development" && entry.status !== "FAILED");
  const browse = browsable.find((entry) => entry.capabilities.flags.candidate_topk)
    ?? browsable.find((entry) => entry.capabilities.flags.per_query_target_ranks);

  const served = scopeCounts(compositionRuns, location.protocol, location.subset).served;
  const native = data.nativeMeasurements;
  const nativeRegion = native.region;
  const nativeRows = (nativeRegion.data?.measurements ?? []).filter((row) =>
    row.protocol_id === location.protocol && row.metrics != null);
  const aggregateReports = (data.aggregates.region.data ?? [])
    .filter((run) => run.protocol_id === location.protocol).length;

  const registry = models.data ?? [];
  const blockedModels = registry.filter((model) => (model.reproduction_status || "").startsWith("BLOCKED")
    || (model.checkpoints || []).every((checkpoint) => !checkpoint.runnable));


  return (
    <div className="stack" data-testid="area-overview">
      <div className="page-intro">
        <div>
          <h2>Từ kết quả đến từng ảnh truy hồi.</h2>
          <p>Đối chiếu mô hình, mở một query và xem thứ tự ứng viên trong cùng một protocol và subset.</p>
        </div>
        {browse && (
          <button className="primary" type="button"
                  onClick={() => onGoto({ area: "samples", view: "explorer", runs: [browse.key], run: browse.runId, query: "", q: "", offset: 0 })}>
            Khám phá query · {browse.modelId}
          </button>
        )}
      </div>

      <div className="scope-strip" data-testid="scope-strip">
        <strong>{protocolLabel(location.protocol)}</strong>
        {/* A protocol with one cohort must not print a subset: "Shortcut-Free (SF)" over FashionIQ
            Original named a cohort this protocol does not have. */}
        {protocolHasSubsets(location.protocol) && <span>{SUBSET_LABEL[location.subset]}</span>}
        {/* Three states the old strip merged into one "0": still loading, failed, and genuinely
            empty. A zero printed while the request is in flight is the W1 defect in miniature. */}
        <span data-testid="scope-counts" data-scope-state={composition.status}>
          {composition.status === "error"
            ? "Không tải được danh sách lượt — đây là lỗi yêu cầu, không phải 0 lượt"
            : composition.status !== "ready"
              ? "Đang tải danh sách lượt…"
              : served === 0 && data.aggregates.region.status !== "ready"
                // Both halves of the scope are still arriving. Claiming "0 lượt" here is the same
                // defect one layer down: a zero printed for a scope that has not been counted yet.
                ? "0 lượt composition · đang tải báo cáo tổng hợp…"
              : served === 0 && aggregateReports > 0
                // A protocol whose data is aggregate reports has ZERO composition rows — and saying
                // only "0 lượt" over 13 measured reports is the W1 complaint itself. Name what
                // this protocol DOES have, and give the link that opens it.
                ? `${served} lượt composition (protocol này không có) · ${aggregateReports} báo cáo tổng hợp có số đo`
                : `${served} lượt phục vụ · ${activeSubset.length} có dữ liệu ở phạm vi này, chưa gồm lượt lỗi`}
        </span>
        {composition.status === "ready" && data.aggregates.region.status === "ready"
          && served === 0 && aggregateReports > 0 && (
          <button type="button" className="ghost" data-testid="scope-strip-open-benchmark"
                  onClick={() => onGoto({ area: "evaluation", view: "benchmark", runs: [], run: "", query: "", offset: 0 })}>
            Mở {aggregateReports} báo cáo tổng hợp
          </button>
        )}
      </div>

      <ModelInventory data={data} location={location} onGoto={onGoto} />

      <div className="grid-3" data-testid="overview-cards">
        <Card title={protocolHasSubsets(location.protocol) ? "Lượt CIRCUS release" : "Số đo native (chưa xác minh)"}
              subtitle={protocolHasSubsets(location.protocol)
                ? `${protocolLabel(location.protocol)} · subset ${SUBSET_LABEL[location.subset]}`
                : `${protocolLabel(location.protocol)} · ${nativeRegion.status === "ready" ? `${nativeRows.length} file có số` : "đang đọc"} · KHÔNG vào leaderboard đã xác minh`}>
          {!protocolHasSubsets(location.protocol)
            ? (nativeRegion.status === "error"
              ? <ErrorState message={nativeRegion.error ?? ""} onRetry={data.nativeMeasurements.retry} />
              : nativeRegion.status !== "ready"
                ? <LoadingState label="Đang đọc measurement native…" />
                : nativeRows.length === 0
                  ? <EmptyState kind="no-results"
                                detail={`Không có measurement native nào cho ${protocolLabel(location.protocol)}.`} />
                  : (
                    <>
                      <KeyValue rows={nativeRows.slice(0, 8).map((row): [string, React.ReactNode] => [
                        String(row.model_id ?? row.source.path.split("/").pop() ?? ""),
                        <span key={row.measurement_id}>
                          R@10 {fmtPercentValue(row.metrics?.r10 ?? null, 2)} · R@50 {fmtPercentValue(row.metrics?.r50 ?? null, 2)}
                          <span className="cell-sub mono">{row.source.path.split("/").pop()}</span>
                        </span>,
                      ])} />
                      <p className="hint" style={{ marginTop: 10 }}>
                        Nguồn: <code>/api/native-measurements</code> — {nativeRegion.data?.counts.with_metrics ?? 0} file có số,
                        {" "}{nativeRegion.data?.counts.signposted_blockers ?? 0} file chỉ ghi blocker,
                        {" "}{nativeRegion.data?.counts.verified ?? 0} đã xác minh. {nativeRegion.data?.note ?? ""}
                      </p>
                    </>
                  ))
            : capabilities.status === "error"
            ? <ErrorState message={capabilities.error ?? ""} onRetry={data.compositionCapabilitiesSummary.retry} />
            : capabilities.status === "loading" && !capabilities.data ? <LoadingState />
            : !caps
              ? <EmptyState kind="no-results" detail="Chưa có summary nào cho subset này." />
              : (
                <>
                  {/* A definition list, not a flex row of big numbers: four wrapped `.stat` boxes
                      put two unrelated figures on one visual line, which read as one number. */}
                  <KeyValue rows={[
                    ["artifact", fmtInt(caps.artifact_count)],
                    ["measurement", fmtInt(caps.unique_measurement_count)],
                    ["chưa xác định danh tính", caps.undetermined_artifact_count > 0
                      ? fmtInt(caps.undetermined_artifact_count)
                      : "0 — mọi artifact đều xác định được"],
                    ["có rank từng query", fmtInt(caps.capability_counts?.per_query_target_ranks ?? 0)],
                  ]} />
                  <p className="hint" style={{ marginTop: 10 }}>
                    Nguồn: <code>/api/composition/capabilities?subset={location.subset}</code>.
                    {caps.image_availability_is_sample ? " Số liệu ảnh lấy trên mẫu, không phải đếm toàn bộ." : ""}
                  </p>
                </>
              )}
        </Card>

        <Card title="Bằng chứng tổng hợp" subtitle="Báo cáo macro; không có rank từng query">
          {evidence.status === "error"
            ? <ErrorState message={evidence.error ?? ""} onRetry={data.evidenceSummary.retry} />
            : evidence.status === "loading" && !evidence.data ? <LoadingState />
            : !ev
              ? <EmptyState kind="no-results" detail="Chưa có report tổng hợp nào." />
              : (
                <>
                  <KeyValue rows={[
                    ["unique measurement", fmtInt(ev.unique_measurement_count)],
                    ["undetermined", ev.undetermined_measurement_count > 0
                      ? `${fmtInt(ev.undetermined_measurement_count)} — thiếu đầu vào danh tính`
                      : "0"],
                    ["được chứng nhận", fmtInt(ev.certified_aggregate_count)],
                    ["report", fmtInt(ev.aggregate_run_count)],
                  ]} />
                  <p className="hint" style={{ marginTop: 10 }}>
                    Nguồn: <code>/api/evidence/summary</code>.
                    {ev.duplicate_attempt_count > 0
                      ? ` ${fmtInt(ev.duplicate_attempt_count)} lượt là lần thử lặp của cùng một measurement.`
                      : " Không có lần thử lặp nào được phát hiện."}
                  </p>
                </>
              )}
        </Card>

        <Card title="Registry mô hình" subtitle="Mô hình chạy được hay bị chặn, kèm lý do">
          {models.status === "error"
            ? <ErrorState message={models.error ?? ""} onRetry={data.models.retry} />
            : models.status === "loading" && !models.data ? <LoadingState />
            : (
              <div className="row" style={{ gap: 24 }}>
                <div className="stat">
                  <span className="stat-value">{fmtInt((models.data ?? []).length)}</span>
                  <span className="stat-label">entry</span>
                </div>
                <div className="stat">
                  <span className="stat-value" style={{ color: blockedModels.length ? "var(--warn)" : undefined }}>{fmtInt(blockedModels.length)}</span>
                  <span className="stat-label">bị chặn</span>
                </div>
              </div>
            )}
        </Card>
      </div>

      <Card title="Lượt có kết quả và lượt bị chặn" subtitle="Chọn một lượt để mở inspector; chỉ số kèm mẫu số."
            actions={<button type="button" onClick={() => onGoto({ area: "evaluation", view: "composition" })}>Mở bảng đánh giá</button>}>
        {composition.status === "loading" && !composition.data ? <LoadingState label="Đang tải summary…" />
          : activeSubset.length === 0
            ? <EmptyState kind="no-results"
                          detail="Không có lượt CIRCUS nào cho subset này."
                          next={<button type="button" onClick={() => onGoto({ subset: "full" })}>Xem subset Full</button>} />
            : (
              <div className="table-wrap">
                <table>
                  <caption className="sr-only">Lượt CIRCUS, subset {SUBSET_LABEL[location.subset]}</caption>
                  <thead>
                    <tr>
                      <th scope="col">Mô hình / checkpoint</th>
                      <th scope="col">Trạng thái</th>
                      <th scope="col" className="num">MM nDCG</th>
                      <th scope="col" className="num">MM R@10</th>
                      <th scope="col" className="num">Composition gap</th>
                      <th scope="col" className="num">Đã chấm / mẫu</th>
                      <th scope="col">Hành động</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activeSubset.map((run) => {
                      const gap = compositionGap(run);
                      const counts = scoredCounts(run, "multimodal");
                      const key = `c:${run.run_id}`;
                      const capabilities = data.catalog.find((entry) => entry.key === key)?.capabilities;
                      return (
                        <tr key={key}>
                          <td>
                            <span className="cell-strong">{run.model_id}</span>
                            <span className="cell-sub">{run.checkpoint_id}{run.checkpoint_transfer ? " · TRANSFER" : ""}</span>
                          </td>
                          <td>
                            <Badge tone={run.is_failed ? "bad" : run.execution_status === "COMPLETED" ? "ok" : "warn"}>
                              {run.is_failed ? "FAILED" : run.execution_status || "UNKNOWN"}
                            </Badge>
                          </td>
                          <td className="num">{run.is_failed ? "—" : fmtNdcg(run.metrics?.ndcg_multimodal)}</td>
                          <td className="num">{run.is_failed ? "—" : fmtPercentValue(run.metrics?.recall_at_10_multimodal)}</td>
                          <td className="num">
                            {gap.state === "available"
                              ? <span title={gap.formula}>{fmtGap(gap.value)}</span>
                              : <span className="hint" title={gap.reason}>unavailable</span>}
                          </td>
                          <td className="num">{fmtDenominator(counts.scored, counts.total)}</td>
                          <td>
                            {/* `runs` is what the explorer selects by; `run` alone left it on
                                "— chọn lượt —" and the inspector empty. */}
                            <button type="button" className="ghost"
                                    onClick={() => onGoto({ area: "samples", view: "explorer", runs: [key], run: run.run_id, query: "", q: "", offset: 0 })}
                                    disabled={!capabilities?.flags.per_query_target_ranks}
                                    title={capabilities?.flags.per_query_target_ranks ? "Mở inspector" : capabilities?.reasons.per_query_target_ranks}>
                              Inspector
                            </button>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
      </Card>

      <div className="grid-2">
        <Card title="Đi tiếp từ đây" subtitle="Ba bước dùng được ngay với dữ liệu đang có">
          <div className="stack">
            <button type="button" className="primary" style={{ justifyContent: "flex-start" }}
                    onClick={() => onGoto({ area: "evaluation", view: "composition" })}>
              So sánh các lượt trên cùng subset và đọc composition gap
            </button>
            <button type="button" style={{ justifyContent: "flex-start" }}
                    onClick={() => onGoto({ area: "samples", view: "explorer" })}>
              Mở một query: ảnh tham chiếu, văn bản sửa, hạng MM / text / image
            </button>
            <button type="button" style={{ justifyContent: "flex-start" }}
                    onClick={() => onGoto({ area: "failures", view: "consensus" })}>
              Lọc query trượt ở mọi lượt rồi lưu thành cohort
            </button>
          </div>
        </Card>

        <Card title="Giới hạn của dữ liệu này" subtitle="Điều KHÔNG suy ra được từ những gì đang có">
          <div className="stack">
            <p className="hint">
              Ứng viên Top-K ảnh <b>chưa được export</b> cho mọi lượt CIRCUS release trên máy này:
              mỗi query có đúng hạng nhưng danh sách ứng viên rỗng. Inspector sẽ nói rõ điều đó thay vì
              vẽ ảnh giả.
            </p>
            <p className="hint">
              Outcome (<code>composition_required</code>, <code>unresolved</code>, …) được phân loại ở
              <b> @10 cố định</b> theo protocol release. Top-K chỉ đổi số ứng viên hiển thị.
            </p>
            <p className="hint">
              Composition gap là <b>phân số không đơn vị</b> tính trên cohort có đủ ba hạng chính xác;
              gap âm là kết quả thật (điều kiện bị cắt bỏ tốt hơn), không bị kẹp về 0.
            </p>
            <Detail summary="Vì sao các số này khác bảng paper">
              <p className="hint">
                Gallery release chứa 74,381 ảnh trong khi paper nêu 18,853 — đây là khác biệt của
                artifact release, không phải lỗi đánh giá. Tỉ lệ trượt paper hiển thị ở khu vực Đánh giá.
              </p>
            </Detail>
          </div>
        </Card>
      </div>
    </div>
  );
}
