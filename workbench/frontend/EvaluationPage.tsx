import type { WorkbenchData, RunCatalogEntry } from "./data";
import { protocolLabel } from "./data";
import type { Location } from "./routes";
import { SUBSET_LABEL, SUBSETS, CONDITIONS, FIXED_CUTOFF_LABEL } from "./routes";
import { Badge, Card, Detail, EmptyState, ErrorState, KeyValue, LoadingState, SortHeader, sortRows, useSort } from "./ui";
import {
  COMPOSITION_GAP_FORMULA, FIXED_CUTOFFS, compositionGap, scoredCounts, transferLabel,
} from "./catalog";
import { fmtDenominator, fmtFractionPercent, fmtGap, fmtInt, fmtNdcg, fmtPercentValue, fmtPp, shortId } from "./format";
import type { CompositionRun } from "./types";
import type { WarningLabel } from "./gate";
import { warningLabels } from "./gate";

/** Đánh giá: measured metrics for a stated protocol and subset, with denominators.
 *
 *  Two controls used to lie here:
 *  * a global Top-K selector implied it changed the CIRCUS outcome classification, which is fixed
 *    at @10 by the protocol — so it now appears only where it really acts (a label on the fixed
 *    cutoff, and separately a candidate-depth control inside the inspector);
 *  * selecting a protocol labelled "FashionIQ Original" rendered CIRCUS release content. The table
 *    header now names the protocol the DATA carries, and a mismatch is impossible by construction:
 *    the rows are filtered by the selected protocol and the same protocol is printed above them.
 */
export function EvaluationPage({ data, location, view, onGoto, onSelectRun }: {
  data: WorkbenchData;
  location: Location;
  view: string;
  onGoto: (patch: Partial<Location>) => void;
  onSelectRun: (entry: RunCatalogEntry, additive?: boolean) => void;
}) {
  if (view === "benchmark") return <BenchmarkTable data={data} location={location} onGoto={onGoto} />;
  return <CompositionTable data={data} location={location} onGoto={onGoto} onSelectRun={onSelectRun} />;
}

function subsetRows(runs: CompositionRun[], subset: string): CompositionRun[] {
  return runs.filter((run) => run.subset === subset);
}

function CompositionTable({ data, location, onGoto, onSelectRun }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
  onSelectRun: (entry: RunCatalogEntry, additive?: boolean) => void;
}) {
  const { sort, toggle } = useSort({ key: "ndcg", dir: "desc" });
  const region = data.compositionRuns.region;
  const all = region.data ?? [];
  const forProtocol = all.filter((run) => run.protocol_id === location.protocol);
  const rows = subsetRows(forProtocol, location.subset);
  const selected = new Set(location.runs);
  const sorted = sortRows(rows, sort, (run, key) => {
    switch (key) {
      case "model": return run.model_id;
      case "ndcg": return run.metrics?.ndcg_multimodal ?? null;
      case "r10": return run.metrics?.recall_at_10_multimodal ?? null;
      case "gap": {
        const gap = compositionGap(run);
        return gap.state === "available" ? gap.value : null;
      }
      case "coverage": return scoredCounts(run, "multimodal").coverage;
      default: return run.run_id;
    }
  });

  const transferred = rows.filter((run) => transferLabel(run).isTransfer);
  const failed = rows.filter((run) => run.is_failed);

  return (
    <div className="stack" data-testid="area-evaluation">
      <Card
        title="CIRCUS composition"
        subtitle={`${protocolLabel(location.protocol)} · ${rows.length} lượt ở subset ${SUBSET_LABEL[location.subset]}`}
        actions={(
          <>
            <div className="viewtabs" role="tablist" aria-label="Subset">
              {SUBSETS.map((subset) => (
                <button key={subset} type="button" role="tab" className="viewtab"
                        aria-selected={location.subset === subset}
                        data-testid={`subset-pill-${subset}`}
                        onClick={() => onGoto({ subset, offset: 0 })}>
                  {SUBSET_LABEL[subset]}
                </button>
              ))}
            </div>
          </>
        )}
      >
        <div className="formula" style={{ marginBottom: 10 }}>
          Composition gap = {COMPOSITION_GAP_FORMULA}. Giá trị là <b>phân số không đơn vị</b>, tính trên
          cohort có đủ ba hạng chính xác. Outcome phân loại ở <b>{FIXED_CUTOFF_LABEL}</b> theo protocol
          release — không đổi bằng Top-K.
        </div>

        {region.status === "error"
          ? <ErrorState message={region.error ?? ""} onRetry={data.compositionRuns.retry} />
          : region.status === "loading" && !region.data
            ? <LoadingState label="Đang tải summary…" />
            : rows.length === 0
              ? <EmptyState kind="no-results"
                            detail={`Protocol ${protocolLabel(location.protocol)} không có summary nào cho subset ${SUBSET_LABEL[location.subset]}.`}
                            next={<button type="button" onClick={() => onGoto({ subset: "full" })}>Xem subset Full</button>} />
              : (
                <div className="table-wrap">
                  <table data-testid="composition-table">
                    <caption className="sr-only">Chỉ số composition theo lượt</caption>
                    <thead>
                      <tr>
                        <th scope="col" style={{ width: 34 }} aria-label="Chọn" />
                        <SortHeader label="Mô hình / checkpoint" column="model" sort={sort} onSort={toggle} />
                        <th scope="col" className="t-col-narrow">Nguồn / transfer</th>
                        <th scope="col">Trạng thái</th>
                        <SortHeader label="MM nDCG" column="ndcg" sort={sort} onSort={toggle} numeric unit="[0,1]" />
                        <SortHeader label="MM R@10" column="r10" sort={sort} onSort={toggle} numeric unit="%" />
                        <th scope="col" className="num t-col-narrow">Text nDCG</th>
                        <th scope="col" className="num t-col-narrow">Image nDCG</th>
                        <SortHeader label="Gap" column="gap" sort={sort} onSort={toggle} numeric />
                        <SortHeader label="Đã chấm" column="coverage" sort={sort} onSort={toggle} numeric />
                        <th scope="col" className="t-col-narrow">Chi tiết</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((run) => {
                        const key = `c:${run.run_id}`;
                        const gap = compositionGap(run);
                        const counts = scoredCounts(run, "multimodal");
                        const transfer = transferLabel(run);
                        const entry = data.catalog.find((item) => item.key === key);
                        return (
                          <tr key={`${run.run_id}-${run.subset}`} aria-selected={selected.has(key)}
                              data-testid={`composition-row-${run.run_id}`}>
                            <td>
                              <input type="checkbox" checked={selected.has(key)}
                                     disabled={!entry?.capabilities.flags.per_query_target_ranks}
                                     title={entry?.capabilities.reasons.per_query_target_ranks}
                                     aria-label={`Chọn ${run.model_id}`}
                                     data-testid={`run-option-${key}`}
                                     onChange={(event) => entry && onSelectRun(entry, event.target.checked)} />
                            </td>
                            <td>
                              <span className="cell-strong">{run.model_id}</span>
                              <span className="cell-sub">{shortId(run.checkpoint_id, 26)}</span>
                            </td>
                            <td className="t-col-narrow">
                              <Badge tone={transfer.isTransfer ? "warn" : "neutral"}
                                     title={transfer.isTransfer ? "Checkpoint huấn luyện trên dataset khác, đánh giá tại đây — không phải dòng paper" : undefined}>
                                {transfer.label}
                              </Badge>
                            </td>
                            <td>
                              <Badge tone={run.is_failed ? "bad" : run.execution_status === "COMPLETED" ? "ok" : "warn"}>
                                {run.is_failed ? "FAILED" : run.execution_status}
                              </Badge>
                            </td>
                            <td className="num">{run.is_failed ? "—" : fmtNdcg(run.metrics?.ndcg_multimodal)}</td>
                            <td className="num">{run.is_failed ? "—" : fmtPercentValue(run.metrics?.recall_at_10_multimodal)}</td>
                            <td className="num t-col-narrow">{run.is_failed ? "—" : fmtNdcg(run.metrics?.ndcg_text_only)}</td>
                            <td className="num t-col-narrow">{run.is_failed ? "—" : fmtNdcg(run.metrics?.ndcg_image_only)}</td>
                            <td className="num" title={gap.state === "available" ? `${gap.formula}; cohort ${fmtInt(gap.cohort)}` : gap.reason}>
                              {gap.state === "available"
                                ? <span style={{ color: gap.value < 0 ? "var(--ok)" : undefined }}>{fmtGap(gap.value)}</span>
                                : <span className="hint">unavailable</span>}
                            </td>
                            <td className="num">{fmtDenominator(counts.scored, counts.total)}</td>
                            <td className="t-col-narrow">
                              <RunMetricsDetail run={run} />
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
      </Card>

      {failed.length > 0 && (
        <EmptyState kind="partial" title={`${failed.length} lượt không đo được gì`}
                    detail={<span>{failed.map((run) => `${run.model_id}: ${run.condition_status?.multimodal?.reason ?? "không có lý do được ghi"}`).join(" · ")}</span>}
                    next={<button type="button" onClick={() => onGoto({ area: "experiments", view: "runs" })}>Xem lý do trong danh sách lượt</button>} />
      )}
      {transferred.length > 0 && (
        <p className="hint">
          {transferred.length} lượt là <b>transfer</b> ({transferred.map((run) => `${run.model_id}: ${transferLabel(run).label}`).join("; ")}).
          Chúng không phải dòng paper của FashionIQ và không được so trực tiếp với bảng paper.
        </p>
      )}
    </div>
  );
}

function RunMetricsDetail({ run }: { run: CompositionRun }) {
  const gap = compositionGap(run);
  const warnings: WarningLabel[] = warningLabels(run as never, 5);
  return (
    <Detail summary="Chỉ số & nguồn">
      <KeyValue rows={[
        ["gap", gap.state === "available"
          ? <span key="g">{fmtGap(gap.value)} · cohort {fmtInt(gap.cohort)} · {gap.formula}</span>
          : <span key="g" className="hint">unavailable — {gap.reason}</span>],
        ["mean cohort", <span key="m">MM {fmtNdcg(run.metrics?.gap_multimodal_ndcg)} · text {fmtNdcg(run.metrics?.gap_text_only_ndcg)} · image {fmtNdcg(run.metrics?.gap_image_only_ndcg)}</span>],
        ["recall", <span key="r">R@10 {fmtPercentValue(run.metrics?.recall_at_10_multimodal)} · R@50 {fmtPercentValue(run.metrics?.recall_at_50_multimodal)} · cắt cố định trong protocol</span>],
        ["outcome", <span key="o">{Object.entries(run.metrics?.outcomes ?? {}).map(([name, count]) => `${name} ${fmtInt(count)}`).join(" · ") || "—"} (tại @{FIXED_CUTOFFS.compositionOutcomeK})</span>],
        ["unscored", <span key="u">{fmtInt(scoredCounts(run, "multimodal").unscored)} query không có hạng chính xác — lỗi pipeline/ánh xạ, không phải lỗi truy hồi
          {scoredCounts(run, "multimodal").unscoredByKind
            ? ` (${Object.entries(scoredCounts(run, "multimodal").unscoredByKind!).map(([kind, n]) => `${kind} ${fmtInt(n)}`).join(", ")})`
            : ""}</span>],
        ["ứng viên", <span key="c">{run.capability_detail?.candidate_topk?.status ?? "chưa công bố"}
          {run.capability_detail?.candidate_topk?.reason ? ` — ${run.capability_detail.candidate_topk.reason}` : ""}</span>],
        ["danh tính", run.duplicate_status
          ? `${run.duplicate_status}${run.duplicate_attempt_count ? ` · ${fmtInt(run.duplicate_attempt_count)} lần thử` : ""}`
          : "chưa công bố"],
        ["coverage", fmtFractionPercent(run.metrics?.coverage)],
        ["gallery", `${fmtInt(run.gallery_count)} ảnh · ${fmtInt(run.query_count)} query`],
        ["checkpoint id", <code key="c">{run.checkpoint_identity ?? run.checkpoint_id}</code>],
        ["run_id", <code key="r2">{run.run_id}</code>],
        ["cảnh báo", <span key="w">{warnings.length ? warnings.map((label) => label.text).join(" · ") : "không có"}</span>],
      ]} />
    </Detail>
  );
}

function BenchmarkTable({ data, location, onGoto }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
}) {
  const region = data.aggregates.region;
  const rows = (region.data ?? []).filter((run) => run.protocol_id === location.protocol);
  const { sort, toggle } = useSort({ key: "r10", dir: "desc" });
  const sorted = sortRows(rows, sort, (run, key) => {
    switch (key) {
      case "model": return run.model_id ?? "";
      case "r10": return run.observed_metrics?.r10 ?? null;
      case "r50": return run.observed_metrics?.r50 ?? null;
      case "delta": return run.observed_metrics?.r10 != null && run.paper_sanity_metrics?.r10 != null
        ? run.observed_metrics.r10 - run.paper_sanity_metrics.r10 : null;
      default: return run.run_id;
    }
  });
  return (
    <div className="stack">
      <Card title="Báo cáo tổng hợp" subtitle={`${rows.length} report cho ${protocolLabel(location.protocol)} · chỉ số macro, không có rank từng query`}>
        {region.status === "error"
          ? <ErrorState message={region.error ?? ""} onRetry={data.aggregates.retry} />
          : region.status === "loading" && !region.data
            ? <LoadingState />
            : rows.length === 0
              ? <EmptyState kind="capability" detail="Protocol này không có báo cáo tổng hợp nào."
                            next={<button type="button" onClick={() => onGoto({ view: "composition" })}>Xem bảng composition</button>} />
              : (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <SortHeader label="Mô hình" column="model" sort={sort} onSort={toggle} />
                        <th scope="col">Chứng nhận</th>
                        <SortHeader label="R@10" column="r10" sort={sort} onSort={toggle} numeric unit="%" />
                        <SortHeader label="R@50" column="r50" sort={sort} onSort={toggle} numeric unit="%" />
                        <th scope="col" className="num">Paper R@10</th>
                        <SortHeader label="Δ vs paper" column="delta" sort={sort} onSort={toggle} numeric unit="pp" />
                        <th scope="col">Khả năng</th>
                        <th scope="col" className="t-col-narrow">Chi tiết</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((run) => {
                        const entry = data.catalog.find((item) => item.key === `a:${run.run_id}`);
                        const certified = run.verification?.certified && run.verification?.is_research_evidence;
                        return (
                          <tr key={run.run_id}>
                            <td>
                              <span className="cell-strong">{(run.model_id || "?").toUpperCase()}</span>
                              <span className="cell-sub">{run.checkpoint_id}</span>
                            </td>
                            <td>
                              <Badge tone={certified ? "ok" : run.verification?.state === "refused" ? "bad" : "warn"}>
                                {run.verification?.state ?? "unknown"}
                              </Badge>
                            </td>
                            <td className="num">{fmtPercentValue(run.observed_metrics?.r10, 2)}</td>
                            <td className="num">{fmtPercentValue(run.observed_metrics?.r50, 2)}</td>
                            <td className="num">{fmtPercentValue(run.paper_sanity_metrics?.r10, 2)}</td>
                            <td className="num">{fmtPp(run.observed_metrics?.r10, run.paper_sanity_metrics?.r10)}</td>
                            <td>
                              <span className="row" style={{ gap: 4 }}>
                                <Badge tone={entry?.capabilities.flags.aggregate_metrics ? "ok" : "neutral"}>aggregate</Badge>
                                <Badge tone={entry?.capabilities.flags.per_query_target_ranks ? "ok" : "bad"}
                                       title={entry?.capabilities.reasons.per_query_target_ranks}>rank</Badge>
                                <Badge tone="warn" title={entry?.capabilities.reasons.candidate_topk}>top-k</Badge>
                              </span>
                            </td>
                            <td>
                              <Detail summary="Chi tiết">
                                <KeyValue rows={[
                                  ["per_query_state", <code key="p">{run.per_query_state ?? "unknown"}</code>],
                                  ["evidence", (run.verification?.reasons ?? []).join("; ") || "—"],
                                  ["danh tính", run.duplicate_status
                                    ? `${run.duplicate_status}${run.duplicate_attempt_count ? ` · ${fmtInt(run.duplicate_attempt_count)} lần thử` : ""}`
                                    : "chưa được công bố — không suy ra được là trùng hay khác"],
                                  ["knobs", run.evaluator_environment_knobs
                                    ? Object.entries(run.evaluator_environment_knobs).map(([name, value]) => `${name}=${shortId(String(value), 40)}`).join(", ")
                                    : "không được ghi (đây KHÔNG phải bằng chứng trùng lặp)"],
                                  ["run_id", <code key="r">{run.run_id}</code>],
                                ]} />
                              </Detail>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
      </Card>
      <p className="hint">
        Paper R@10 là <b>tham chiếu văn liệu</b>, không phải số đo. Δ tính bằng điểm phần trăm (pp) so với
        tham chiếu, không phải phần trăm tương đối.
      </p>
    </div>
  );
}
