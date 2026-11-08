import type { WorkbenchData, RunCatalogEntry } from "./data";
import { protocolLabel } from "./data";
import { Badge, Card, EmptyState, ErrorState, KeyValue, LoadingState, Progress, Detail } from "./ui";
import { fmtDateStamp, fmtDenominator, fmtFractionPercent, fmtGap, fmtInt, fmtNdcg, fmtPercentValue, fmtRank } from "./format";
import { compositionGap, scoredCounts } from "./catalog";
import type { Location } from "./routes";
import { SUBSET_LABEL } from "./routes";

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
  const activeSubset = compositionRuns.filter((run) => run.subset === location.subset);
  const failed = activeSubset.filter((run) => run.is_failed || run.execution_status === "FAILED");

  const registry = models.data ?? [];
  const blockedModels = registry.filter((model) => (model.reproduction_status || "").startsWith("BLOCKED")
    || (model.checkpoints || []).every((checkpoint) => !checkpoint.runnable));


  return (
    <div className="stack" data-testid="area-overview">
      <div className="grid-3" data-testid="overview-cards">
        <Card title="Lượt CIRCUS release" subtitle={`Subset ${SUBSET_LABEL[location.subset]} · ${protocolLabel(location.protocol)}`}>
          {capabilities.status === "error"
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
                            <button type="button" className="ghost"
                                    onClick={() => onGoto({ area: "samples", view: "explorer", run: run.run_id, query: "" })}
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
