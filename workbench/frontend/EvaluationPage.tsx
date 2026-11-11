import { useState } from "react";
import type { WorkbenchData, RunCatalogEntry } from "./data";
import { protocolLabel } from "./data";
import type { Location } from "./routes";
import { SUBSET_LABEL, SUBSETS, CONDITIONS, FIXED_CUTOFF_LABEL } from "./routes";
import { Badge, Card, Detail, DetailToggle, EmptyState, ErrorState, KeyValue, LoadingState, SortHeader, sortRows, useSort } from "./ui";
import {
  COMPOSITION_COLUMNS, COMPOSITION_GAP_FORMULA, FIXED_CUTOFFS, compositionGap, compositionMinWidth,
  protocolsWithReports,
  scoredCounts, scoringImplsPresent, testWorkspaceOrigin, transferLabel,
} from "./catalog";
import { fmtDateStamp, fmtDenominator, fmtFractionPercent, fmtGap, fmtInt, fmtNdcg, fmtPercentValue, fmtPp, fmtRecall, shortId } from "./format";
import type { CompositionRun } from "./types";
import type { NativeMeasurement } from "./data";
import type { WarningLabel } from "./gate";
import { warningLabels } from "./gate";
import { derivedLineage, nativeConflictChips, nativeRepeatLabel, nativeVarianceSummary, protocolHasSubsets } from "./scope";

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
  if (view === "native") return <NativeMeasurementTable data={data} location={location} onGoto={onGoto} />;
  return <CompositionTable data={data} location={location} onGoto={onGoto} onSelectRun={onSelectRun} />;
}

function subsetRows(runs: CompositionRun[], subset: string): CompositionRun[] {
  return runs.filter((run) => run.subset === subset);
}

/** The G1/G2 verdict on the web, derived from the artifacts instead of quoted from prose.
 *
 *  An empty `passed_models` is rendered as the verdict it is: NOT a failure to load, and not a
 *  placeholder. Each model prints its own status, the dimensions that held, and the mandatory
 *  checks that did not — with declared vs measured, because "failed" without the numbers is
 *  unarguable and unfixable.
 */
function ParityGateCard({ data }: { data: WorkbenchData }) {
  const region = data.parityGate.region;
  if (region.status === "error") {
    return (
      <Card title="G1/G2 — cổng chẵn lẻ (từ artifact)" testId="parity-gate">
        <ErrorState message={region.error ?? "Không tải được cổng G1/G2"} onRetry={data.parityGate.retry} />
      </Card>
    );
  }
  if (!region.data) {
    return <Card title="G1/G2 — cổng chẵn lẻ (từ artifact)" testId="parity-gate"><LoadingState label="Đang tính cổng từ artifact…" /></Card>;
  }
  const gate = region.data;
  const models = Object.values(gate.models ?? {});
  const failed = models.filter((model) => !model.passed);
  return (
    <Card
      title="G1/G2 — cổng chẵn lẻ (từ artifact)"
      testId="parity-gate"
      subtitle={`Chuẩn ${gate.standard} · tính từ artifact đo, không đọc từ văn bản`}
      actions={(
        <Badge tone={gate.passed_models.length ? "ok" : "bad"} testId="parity-gate-verdict">
          {gate.passed_models.length
            ? `PASS: ${gate.passed_models.join(", ")}`
            : `KHÔNG mô hình nào PASS (0/${models.length})`}
        </Badge>
      )}
    >
      {/* The empty list is the answer, said out loud, so nobody reads it as "not loaded". */}
      <p className="pipeline-notice" data-testid="parity-gate-summary">
        {gate.passed_models.length === 0
          ? <>Không mô hình nào đạt: {failed.length}/{models.length} mô hình có subcheck bắt buộc FAIL. Đây là kết luận từ artifact, không phải lỗi tải.</>
          : <>{gate.passed_models.length}/{models.length} mô hình đạt mọi subcheck bắt buộc.</>}
      </p>
      <div className="table-wrap">
        <table data-testid="parity-gate-table">
          <caption className="sr-only">Kết luận G1/G2 theo mô hình, kèm subcheck bắt buộc thất bại</caption>
          <thead>
            <tr>
              <th scope="col">Mô hình</th>
              <th scope="col">Kết luận</th>
              <th scope="col" className="num">Subcheck FAIL</th>
              <th scope="col">Chiều đạt</th>
            </tr>
          </thead>
          <tbody>
            {models.map((model) => (
              <tr key={model.model_id} data-testid={`parity-row-${model.model_id}`} data-passed={model.passed}>
                <th scope="row">{model.model_id}</th>
                <td><Badge tone={model.passed ? "ok" : "bad"}>{model.status}</Badge></td>
                <td className="num">
                  {model.failure_count === 0 ? "0" : <Detail summary={`${model.failure_count} subcheck`} testId={`parity-failures-${model.model_id}`}>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {model.failures.map((failure) => (
                        <li key={failure.name}>
                          <code>{failure.name}</code> ({failure.gate}, {failure.dimension}): khai báo{" "}
                          <b>{String(failure.declared)}</b>, đo <b>{String(typeof failure.measured === "number" ? failure.measured.toPrecision(6) : failure.measured)}</b>
                        </li>
                      ))}
                    </ul>
                  </Detail>}
                </td>
                <td>
                  <span className="cell-sub">
                    {Object.entries(model.dimensions ?? {}).map(([dimension, ok]) => (
                      <span key={dimension} className={ok ? "hint-ok" : "hint-bad"} style={{ marginRight: 8 }}>
                        {dimension} {ok ? "✓" : "✗"}
                      </span>
                    ))}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="hint">{gate.note}</p>
    </Card>
  );
}

function CompositionTable({ data, location, onGoto, onSelectRun }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
  onSelectRun: (entry: RunCatalogEntry, additive?: boolean) => void;
}) {
  const { sort, toggle } = useSort({ key: "ndcg", dir: "desc" });
  const [expandedDetail, setExpandedDetail] = useState<string | null>(null);
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
  const expandedRow = expandedDetail
    ? sorted.map((run) => ({ run, key: `${run.run_id}-${run.subset}` }))
        .find((candidate) => candidate.key === expandedDetail) ?? null
    : null;

  return (
    <div className="stack" data-testid="area-evaluation">
      <ParityGateCard data={data} />
      <Card
        title="CIRCUS composition"
        subtitle={`${protocolLabel(location.protocol)} · ${rows.length} lượt${protocolHasSubsets(location.protocol) ? ` ở subset ${SUBSET_LABEL[location.subset]}` : ""}`}
        actions={protocolHasSubsets(location.protocol) ? (
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
        ) : undefined}
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
                <>
                  <ScoringImplNotice rows={sorted} />
                  <div className="table-wrap">
                  <table className="fixed-table" data-testid="composition-table"
                         style={{ minWidth: compositionMinWidth() }}>
                    <caption className="sr-only">Chỉ số composition theo lượt</caption>
                    <colgroup>
                      {COMPOSITION_COLUMNS.map((column) => (
                        <col key={column.key} data-col={column.key}
                             style={column.px ? { width: column.px } : undefined} />
                      ))}
                    </colgroup>
                    <thead>
                      <tr>
                        <th scope="col" aria-label="Chọn" />
                        <SortHeader label="Mô hình / checkpoint" column="model" sort={sort} onSort={toggle} />
                        <th scope="col">Nguồn / transfer</th>
                        <th scope="col">Trạng thái</th>
                        <SortHeader label="MM nDCG" column="ndcg" sort={sort} onSort={toggle} numeric unit="[0,1]" />
                        <SortHeader label="MM R@10" column="r10" sort={sort} onSort={toggle} numeric unit="%" />
                        <th scope="col" className="num">Text nDCG</th>
                        <th scope="col" className="num">Image nDCG</th>
                        <SortHeader label="Gap" column="gap" sort={sort} onSort={toggle} numeric />
                        <SortHeader label="Đã chấm" column="coverage" sort={sort} onSort={toggle} numeric />
                        <th scope="col">Chi tiết</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((run) => {
                        const key = `c:${run.run_id}`;
                        const gap = compositionGap(run);
                        const counts = scoredCounts(run, "multimodal");
                        const transfer = transferLabel(run);
                        const entry = data.catalog.find((item) => item.key === key);
                        const rowKey = `${run.run_id}-${run.subset}`;
                        const expanded = expandedDetail === rowKey;
                        return (
                          <tr aria-selected={selected.has(key)}
                              data-testid={`composition-row-${run.run_id}`} data-expanded={expanded || undefined}>
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
                              {/* A derived subset carries its PARENT's execution_status, so the
                                  status column alone cannot say this subset is 586/586 while the
                                  parent is 6001/6003. The lineage and its own coverage are printed
                                  here, from `derived_from`; no metric or denominator is recomputed. */}
                              {(() => {
                                const lineage = derivedLineage(run, (id) => shortId(id, 8));
                                if (!lineage) return null;
                                return (
                                  <span className="cell-sub" data-testid={`derived-lineage-${run.run_id}`}>
                                    <Badge tone={lineage.ownComplete ? "ok" : "warn"}>{lineage.marker}</Badge>
                                    {" "}{lineage.text}
                                  </span>
                                );
                              })()}
                            </td>
                            <td>
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
                            <td className="num">{run.is_failed ? "—" : fmtNdcg(run.metrics?.ndcg_text_only)}</td>
                            <td className="num">{run.is_failed ? "—" : fmtNdcg(run.metrics?.ndcg_image_only)}</td>
                            <td className="num" title={gap.state === "available" ? `${gap.formula}; cohort ${fmtInt(gap.cohort)}` : gap.reason}>
                              {gap.state === "available"
                                ? <span style={{ color: gap.value < 0 ? "var(--ok)" : undefined }}>{fmtGap(gap.value)}</span>
                                : <span className="hint">unavailable</span>}
                            </td>
                            {/* Two lines, not one: "6,003 / 6,003 (100.0%)" is ~150 px of
                                nowrap digits and cannot fit a 104 px track. Same three numbers,
                                same meaning; the coverage share sits under its own denominator. */}
                            <td className="num">
                              {fmtInt(counts.scored)} / {fmtInt(counts.total)}
                              <span className="cell-sub num">{fmtFractionPercent(counts.coverage)}</span>
                            </td>
                            <td>
                              <DetailToggle open={expanded}
                                            onToggle={() => setExpandedDetail(expanded ? null : rowKey)} />
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                  </div>
                  {expandedRow && (
                    <section className="row-detail" data-testid="composition-detail">
                      <header className="row-detail-head">
                        <b>{expandedRow.run.model_id} / {shortId(expandedRow.run.checkpoint_id, 26)}</b>
                        <span className="hint">{expandedRow.run.run_id}</span>
                        <button type="button" className="ghost" onClick={() => setExpandedDetail(null)}>Đóng</button>
                      </header>
                      <RunMetricsDetail run={expandedRow.run} />
                    </section>
                  )}
                </>
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

/** One line naming the scoring implementations present when they are NOT all the same.
 *
 *  What differs is the arithmetic (summation order, precision), so the raw scores are not on a
 *  common scale and must not be read as one. What does NOT differ is the evaluation contract: the
 *  same gallery, query set, cutoff and metric definitions, so ranks and metrics stay comparable
 *  across these rows. Declaring them incomparable "at query level" was too strong — it would forbid
 *  exactly the same-protocol comparison this table exists for.
 */
function ScoringImplNotice({ rows }: { rows: CompositionRun[] }) {
  const present = scoringImplsPresent(rows);
  if (present.length <= 1) return null;
  return (
    <p className="hint" data-testid="scoring-impl-notice">
      Các lượt dưới đây chấm bằng <b>{present.length} cách khác nhau</b> ({present.join(", ")}). Điểm
      thô không cùng thang đo vì thứ tự cộng dồn khác nhau, nên đừng đọc chênh lệch điểm như một
      khoảng cách chính xác. Hạng và metric vẫn so sánh được trong cùng protocol, cùng tập query và
      cùng quy tắc đánh giá; cột điểm ở đây là số tổng hợp, không phải bảng xếp hạng độ chính xác.
    </p>
  );
}

function RunMetricsDetail({ run }: { run: CompositionRun }) {
  const gap = compositionGap(run);
  const warnings: WarningLabel[] = warningLabels(run as never, 5);
  const lineage = derivedLineage(run, (id) => shortId(id, 8));
  return (
    <KeyValue rows={[
        ...(lineage ? [[`dẫn xuất (${lineage.marker})`,
          <span key="dv" data-testid="derived-detail">{lineage.text} · parent run_id <code>{lineage.parentRunId}</code></span>] as [string, React.ReactNode]] : []),
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
        ["scoring impl", <span key="si">
          {run.scoring_impl?.impl
            ? <><code>{run.scoring_impl.impl}</code>{run.scoring_impl.basis ? <span className="hint"> — {run.scoring_impl.basis}</span> : null}</>
            : <span className="hint">unknown — backend did not record a derivation for this row</span>}
        </span>],
        ["checkpoint id", <code key="c">{run.checkpoint_identity ?? run.checkpoint_id}</code>],
        ["run_id", <code key="r2">{run.run_id}</code>],
        ["cảnh báo", <span key="w">{warnings.length ? warnings.map((label) => label.text).join(" · ") : "không có"}</span>],
      ]} />
  );
}

function BenchmarkTable({ data, location, onGoto }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
}) {
  const region = data.aggregates.region;
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const served = protocolsWithReports(region.data ?? []);
  const rows = (region.data ?? []).filter((run) => run.protocol_id === location.protocol);
  const { sort, toggle } = useSort({ key: "r10", dir: "desc" });
  const expandedReport = expandedRow
    ? (region.data ?? []).find((run) => run.run_id === expandedRow) ?? null
    : null;
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
    <div className="stack" data-testid="area-benchmark">
      <Card title="Báo cáo tổng hợp" subtitle={`${rows.length} report cho ${protocolLabel(location.protocol)} · chỉ số macro, không có rank từng query`}>
        {region.status === "error"
          ? <ErrorState message={region.error ?? ""} onRetry={data.aggregates.retry} />
          : region.status === "loading" && !region.data
            ? <LoadingState />
            : rows.length === 0
              ? <EmptyState kind="capability"
                            testId="benchmark-empty"
                            title={`Protocol ${protocolLabel(location.protocol)} không có báo cáo tổng hợp nào`}
                            detail={served.length
                              ? <>Đây là dữ liệu, không phải lỗi tải. {served.length} protocol khác có báo cáo — chọn một cái ở dưới để xem ngay.</>
                              : "Không protocol nào trong deployment này có báo cáo tổng hợp."}
                            next={<>
                              {served.map((item) => (
                                <button key={item.protocol_id} type="button"
                                        data-testid={`benchmark-goto-${item.protocol_id}`}
                                        onClick={() => onGoto({ protocol: item.protocol_id, runs: [], run: "", query: "", offset: 0 })}>
                                  {protocolLabel(item.protocol_id)} ({fmtInt(item.count)} report)
                                </button>
                              ))}
                              <button type="button" className="ghost" onClick={() => onGoto({ view: "composition" })}>Xem bảng composition</button>
                            </>} />
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
                        <th scope="col">Chi tiết</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sorted.map((run) => {
                        const entry = data.catalog.find((item) => item.key === `a:${run.run_id}`);
                        const certified = run.verification?.certified && run.verification?.is_research_evidence;
                        const origin = testWorkspaceOrigin(run);
                        return (
                          <tr key={run.run_id}>
                            <td>
                              <span className="cell-strong">{(run.model_id || "?").toUpperCase()}</span>
                              <span className="cell-sub">{run.checkpoint_id}</span>
                              {origin && (
                                <span className="cell-sub" data-testid={`benchmark-origin-${run.run_id}`}
                                      title={`Evaluator ran with paths under ${origin}: a test/development run, not evidence about a trained model`}>
                                  TEST WORKSPACE · {origin.split("/").pop()}
                                </span>
                              )}
                              <span className="cell-sub mono">{fmtDateStamp(run.run_id)}</span>
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
                              <DetailToggle open={expandedRow === run.run_id}
                                            onToggle={() => setExpandedRow(expandedRow === run.run_id ? null : run.run_id)} />
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
        {expandedReport && (
          <section className="row-detail" data-testid="benchmark-detail">
            <header className="row-detail-head">
              <b>{(expandedReport.model_id || "?").toUpperCase()} / {expandedReport.checkpoint_id}</b>
              <span className="hint mono">{expandedReport.run_id}</span>
              <button type="button" className="ghost" onClick={() => setExpandedRow(null)}>Đóng</button>
            </header>
            <KeyValue rows={[
              ["per_query_state", <code key="p">{expandedReport.per_query_state ?? "unknown"}</code>],
              ["evidence", (expandedReport.verification?.reasons ?? []).join("; ") || "—"],
              ["danh tính", expandedReport.duplicate_status
                ? `${expandedReport.duplicate_status}${expandedReport.duplicate_attempt_count ? ` · ${fmtInt(expandedReport.duplicate_attempt_count)} lần thử` : ""}`
                : "chưa được công bố — không suy ra được là trùng hay khác"],
              ["knobs", expandedReport.evaluator_environment_knobs
                ? <span key="k">{Object.entries(expandedReport.evaluator_environment_knobs)
                    .map(([name, value]) => `${name}=${value}`).join(", ")}</span>
                : "không được ghi (đây KHÔNG phải bằng chứng trùng lặp)"],
            ]} />
          </section>
        )}
      </Card>
      <p className="hint">
        Paper R@10 là <b>tham chiếu văn liệu</b>, không phải số đo. Δ tính bằng điểm phần trăm (pp) so với
        tham chiếu, không phải phần trăm tương đối.
      </p>
    </div>
  );
}

/** Số đo native — the served table for rows `/api/native-measurements` holds.
 *
 *  W-02: a model whose only numbers are native macros had no view to open. Its "elsewhere" link
 *  went to the benchmark table, which lists `aggregate` reports and therefore showed an EMPTY page
 *  under a label promising results. This view is the destination of that link, and the anchor a
 *  native row can be opened at (`?view=native&model=…`).
 *
 *  W-01: the unit travels with each row. `fmtRecall` renders a `fraction` row as a percent of the
 *  same value and a `percent` row as-is, so `0.2793` and `27.93` print as one number. A row whose
 *  artifact never declares a unit prints its raw value with "đơn vị không ghi" instead of a `%` the
 *  artifact never promised, and `counts.unit_not_declared` names every such row at the top.
 */
function NativeMeasurementTable({ data, location, onGoto }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
}) {
  const region = data.nativeMeasurements.region;
  const [expanded, setExpanded] = useState<string | null>(null);
  const all = region.data?.measurements ?? [];
  const rows = all.filter((row) => row.metrics && row.protocol_id === location.protocol);
  const undeclared = region.data?.undeclared_unit_rows ?? [];
  const protocols = [...new Set(all.filter((row) => row.metrics && row.protocol_id)
    .map((row) => row.protocol_id as string))];
  const model = new URLSearchParams(window.location.search).get("model");

  return (
    <div className="stack" data-testid="area-native">
      <Card
        title="Số đo native (chưa xác minh)"
        subtitle={region.status === "ready"
          // A zero printed for a region that FAILED or is still loading is a count of nothing that
          // was counted. Name the state instead of the number.
          ? `${rows.length} dòng có số cho ${protocolLabel(location.protocol)} · không vào leaderboard đã xác minh`
          : region.status === "error"
            ? `không đọc được catalog native cho ${protocolLabel(location.protocol)} — đây là lỗi yêu cầu, không phải 0 dòng`
            : `đang đọc catalog native cho ${protocolLabel(location.protocol)}…`}
      >
        {region.status === "error"
          ? <ErrorState message={region.error ?? "Không tải được catalog native"} onRetry={data.nativeMeasurements.retry} />
          : region.status !== "ready"
            ? <LoadingState label="Đang đọc catalog native…" />
            : rows.length === 0
              ? <EmptyState kind="no-results" testId="native-empty"
                           detail={`Không có số đo native nào cho ${protocolLabel(location.protocol)}. Đây là dữ liệu, không phải lỗi tải.`}
                           next={<>
                             {protocols.map((id) => (
                               <button key={id} type="button" data-testid={`native-goto-${id}`}
                                       onClick={() => onGoto({ protocol: id, runs: [], run: "", query: "", offset: 0 })}>
                                 {protocolLabel(id)}
                               </button>
                             ))}
                           </>} />
              : (
                <>
                  <div className="pipeline-notice" data-testid="native-unit-notice">
                    <strong>{fmtInt(region.data?.counts.unit_not_declared ?? 0)} dòng không khai đơn vị trong artifact.</strong>
                    <p className="hint">
                      <b>{fmtInt(region.data?.counts.duplicate_rows ?? 0)}</b> dòng trùng bị bỏ khỏi đếm,{" "}
                      <b>{fmtInt(region.data?.counts.unbacked_rows ?? 0)}</b> dòng không có artifact chống lưng.
                    </p>
                    <p className="hint">
                      Đơn vị đọc từ chính file (<code>metric_unit</code>/<code>unit</code>…). Không file nào khai, nên mọi giá trị
                      dưới đây được render đúng thang đã suy theo độ lớn:{" "}
                      {fmtInt(region.data?.counts.unit_inferred ?? 0)} dòng suy là percent,{" "}
                      {fmtInt(region.data?.counts.unit_undeclared ?? 0)} dòng không suy được.
                      Một fraction <code>0.2793</code> và một percent <code>27.93</code> là cùng một số đo, và cả hai render
                      thành <b>27.93 %</b> ở đây.
                    </p>
                    {undeclared.length > 0 && (
                      <Detail summary={`${fmtInt(undeclared.length)} dòng thiếu đơn vị khai báo`} testId="native-undeclared-list">
                        <ul style={{ margin: 0, paddingLeft: 18 }}>
                          {undeclared.map((row) => (
                            <li key={row.measurement_id}>
                              <code>{row.model_id ?? "?"}</code> · {row.metric_path ?? "(không rõ block)"} ·{" "}
                              R@10 {fmtRecall(row.metrics?.r10 ?? null, row.unit, 2)} ({row.unit_source ?? "không ghi"}) ·{" "}
                              <span className="mono">{row.source.split("/").pop()}</span>
                            </li>
                          ))}
                        </ul>
                      </Detail>
                    )}
                  </div>

                  <div className="table-wrap">
                    <table data-testid="native-table">
                      <thead>
                        <tr>
                          <th scope="col">Mô hình</th>
                          <th scope="col">Nguồn</th>
                          <th scope="col" className="num">R@10</th>
                          <th scope="col" className="num">R@50</th>
                          <th scope="col">Đơn vị</th>
                          <th scope="col">Mapping</th>
                          <th scope="col">Chi tiết</th>
                        </tr>
                      </thead>
                      <tbody>
                        {rows.map((row) => {
                          // R16: three served facts the table used to print as if they were absent.
                          // The DECISIONS live in `scope.ts` so the tests execute them; this file only
                          // renders their output.
                          const conflicts = nativeConflictChips(row, rows);
                          const variance = nativeVarianceSummary(row);
                          const repeat = nativeRepeatLabel(row, rows);
                          return (
                          <tr key={row.measurement_id}
                              data-testid={`native-row-${row.measurement_id}`}
                              aria-selected={model ? row.model_id === model : undefined}>
                            <td>
                              <span className="cell-strong">{row.model_id ?? "?"}</span>
                              <span className="cell-sub">{row.metric_path ?? "(không rõ block)"}</span>
                              {row.rollup && <span className="cell-sub">roll-up: lặp lại số của file khác</span>}
                            </td>
                            <td>
                              <span className="mono cell-sub">{row.source.path.split("/").pop()}</span>
                              <span className="cell-sub mono">{row.source.sha256?.slice(0, 16) ?? "không digest"}…</span>
                              {/* W-01: a roll-up states a number; it does not back one. Saying which
                                  file the served value came from is what stops a roll-up's number
                                  reading as a measurement. */}
                              <span className="cell-sub">
                                <Badge tone={row.artifact_backed ? "ok" : "bad"}>
                                  {row.artifact_backed ? "có artifact" : "không artifact"}
                                </Badge>
                              </span>
                              {row.backed_by && (
                                <span className="cell-sub mono" title={row.backed_by_sha256 ?? ""}>
                                  ← {row.backed_by.split("/").pop()}
                                </span>
                              )}
                              {row.raw_metrics && (() => {
                                // R07: `row.unit` is the REPLACING report's scale, not the scale
                                // the roll-up wrote. Reading the pre-replacement number on the new
                                // row's unit printed DCNet's 0.2793 fraction as "0.2793 %". The
                                // backend keeps `raw_unit`/`raw_unit_source` beside `raw_metrics`;
                                // `row.unit` is the fallback ONLY for a row served without them, and
                                // the line says which scale is on screen and whether it was inferred
                                // — the same rule the unit badge one column over follows.
                                const hasRawUnit = row.raw_unit !== undefined;
                                const rawUnit = hasRawUnit ? row.raw_unit : row.unit;
                                const rawSource = hasRawUnit ? row.raw_unit_source : row.unit_source;
                                return (
                                  <span className="cell-sub" data-testid={`native-replaced-${row.measurement_id}`}>
                                    roll-up ghi R@10 {fmtRecall(row.raw_metrics.r10 ?? null, rawUnit, 4)} — đã thay
                                    bằng số của artifact. Số gốc theo thang{" "}
                                    {rawUnit ?? "không ghi"}
                                    {hasRawUnit ? "" : " (lấy từ dòng đã thay: backend không kèm thang gốc)"}
                                    {rawSource === "inferred_from_magnitude"
                                      ? " · suy theo độ lớn, không phải khai báo"
                                      : ` · ${rawSource ?? "không ghi"}`}
                                  </span>
                                );
                              })()}
                              {/* Two different reasons a row is not counted, told apart: a repeat
                                  names the row it duplicates, an unbacked claim names no artifact. */}
                              {row.counted === false && (
                                <span className="cell-sub">
                                  {row.duplicate_of
                                    ? `trùng với ${row.duplicate_of.split("/").pop()} · không đếm`
                                    : "không có artifact chống lưng · không đếm"}
                                </span>
                              )}
                              {/* R16: "repeat 2 of 4" — the draw index the file states, so a reader
                                  sees WHICH repeat is on screen, not just that it repeats. */}
                              {repeat && (
                                <span className="cell-sub" data-testid={`native-repeat-${row.measurement_id}`}>
                                  {repeat}
                                </span>
                              )}
                              {conflicts.map((chip) => (
                                <span className="cell-sub" key={chip.key}
                                      data-testid={`native-conflict-${row.measurement_id}`}
                                      title={chip.label}>
                                  <Badge tone="warn">CHƯA CHỐT · {chip.file}</Badge>
                                </span>
                              ))}
                              {variance.kind !== "none" && (
                                <span className="cell-sub" data-testid={`native-variance-${row.measurement_id}`}>
                                  {variance.label}
                                </span>
                              )}
                            </td>
                            <td className="num">
                              {fmtRecall(row.metrics?.r10 ?? null, row.unit, 2)}
                              {conflicts.map((chip) => (
                                <span className="cell-sub" key={chip.key} data-testid={`native-conflict-r10-${row.measurement_id}`}>
                                  {chip.perMetric.r10}
                                </span>
                              ))}
                            </td>
                            <td className="num">
                              {fmtRecall(row.metrics?.r50 ?? null, row.unit, 2)}
                              {conflicts.map((chip) => (
                                <span className="cell-sub" key={chip.key} data-testid={`native-conflict-r50-${row.measurement_id}`}>
                                  {chip.perMetric.r50}
                                </span>
                              ))}
                            </td>
                            <td>
                              <Badge tone={row.unit_source === "declared"
                                || row.unit_source === "sibling_key" ? "ok" : "warn"}>
                                {row.unit ?? "không ghi"}
                              </Badge>
                              <span className="cell-sub">{row.unit_source ?? "không ghi"}</span>
                            </td>
                            <td>
                              {/* A mapping status is why an exploratory number is not a reproduction:
                                  without it the row reads as a measurement of the paper's protocol. */}
                              {row.mapping_status
                                ? <Badge tone="warn">{row.mapping_status}</Badge>
                                : <span className="hint">không ghi</span>}
                            </td>
                            <td><DetailToggle open={expanded === row.measurement_id}
                                              onToggle={() => setExpanded(expanded === row.measurement_id ? null : row.measurement_id)} /></td>
                          </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  <NativeMeasurementDetail measurement={rows.find((row) => row.measurement_id === expanded) ?? null} />
                </>
              )}
        <p className="hint" style={{ marginTop: 10 }}>
          Nguồn: <code>/api/native-measurements</code>. {region.data?.note ?? ""}
        </p>
      </Card>
    </div>
  );
}

/** One native row, opened: the record itself, not a re-statement of the table. */
function NativeMeasurementDetail({ measurement }: { measurement: NativeMeasurement | null }) {
  if (!measurement) return null;
  return (
    <section className="row-detail" data-testid="native-detail">
      <header className="row-detail-head">
        <b>{measurement.model_id ?? "?"} / {measurement.checkpoint_id ?? "không checkpoint"}</b>
        <span className="hint mono">{measurement.measurement_id}</span>
      </header>
      <KeyValue rows={[
        ["đường dẫn số đo", <code key="p">{measurement.metric_path ?? "—"}</code>],
        ["file", <span key="f" className="mono">{measurement.source.path}</span>],
        ["digest", <span key="d" className="mono">{measurement.source.sha256 ?? "không có (file không đọc được)"}</span>],
        ["bytes", fmtInt(measurement.source.bytes)],
        ["R@10", fmtRecall(measurement.metrics?.r10 ?? null, measurement.unit, 2)],
        ["R@50", fmtRecall(measurement.metrics?.r50 ?? null, measurement.unit, 2)],
        ["đơn vị", `${measurement.unit ?? "không ghi"} (${measurement.unit_source ?? "không ghi"})`],
        ["mapping", measurement.mapping_status ?? "không ghi"],
        ["xác minh", `${measurement.verification.state} — ${measurement.verification.reason}`],
        ["category", measurement.categories.length ? measurement.categories.join(", ") : "không ghi"],
        ["roll-up", measurement.rollup ? `có — entry ${measurement.rollup_entry ?? "?"}` : "không"],
      ]} />
    </section>
  );
}
