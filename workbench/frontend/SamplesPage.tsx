import { useMemo } from "react";
import type { CompositionQuery, CompositionRun } from "./types";
import type { WorkbenchData } from "./data";
import { protocolLabel } from "./data";
import type { Location } from "./routes";
import { CONDITIONS, ORDERS, OUTCOMES, SUBSET_LABEL } from "./routes";
import { Card, Detail, EmptyState, ErrorState, Field, LoadingState } from "./ui";
import { fmtInt, fmtRank } from "./format";
import { QueryInspector } from "./QueryInspector";
import { QUERY_STATE_CHIPS, protocolHasSubsets } from "./scope";
import { LimnScopeDiagnostic } from "./LimnScopeDiagnostic";
import { fetchJson, useRegion } from "./useRegion";
import { candidateRequestPath, selectExactQuery } from "./candidateRequest";

/** Khám phá mẫu: browse queries of one run, then inspect one.
 *
 *  The list and the inspector are two regions with two request identities: paging the list must not
 *  blank the inspector, and jumping to a query must not be overridden by the list's answer.
 *
 *  The inspected query is fetched BY ITS ID, never looked up in the current page. The old
 *  `items.find(...) ?? items[0]` silently showed a different sample whenever the selected query sat
 *  on another page or another run — the exact bug a deep link or a paging click exposed.
 *
 *  Every filter is in the URL, so a link reproduces the exact view.
 */
export function SamplesPage({ data, location, onGoto, onAnnotate }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
  onAnnotate: () => void;
}) {
  // One protocol per page: a run from another protocol must not be selectable here, or the header
  // would name a protocol the rows beneath it do not belong to.
  const catalog = data.catalog.filter((item) => item.protocolId === location.protocol);
  const runKey = location.runs[0];
  const entry = catalog.find((item) => item.key === runKey) ?? null;
  const run: CompositionRun | null = entry?.composition ?? null;
  const page = data.compositionPage.region;
  const items = page.data?.items ?? [];
  const total = page.data?.total ?? 0;
  const limit = 20;

  // The inspected query is its own read, keyed by run + subset + condition + query id.
  const inspectorPath = run && location.query
    ? candidateRequestPath({ runId: run.run_id, subset: location.subset, condition: location.condition,
                             queryId: location.query, k: 10 })
    : "none";
  const selectedRegion = useRegion<{ items: CompositionQuery[] }>(
    "selected-query", inspectorPath,
    useMemo(() => (inspectorPath === "none"
      ? null
      : (signal: AbortSignal) => fetchJson<{ items: CompositionQuery[] }>(inspectorPath, signal)),
      [inspectorPath]));
  const selectedCurrent = selectedRegion.region.key === inspectorPath;
  const selectedReady = selectedCurrent && selectedRegion.region.status === "ready";
  const selectedQuery = location.query
    ? selectExactQuery(selectedReady ? selectedRegion.region.data?.items ?? [] : [], location.query)
    : page.status === "ready" ? items[0] ?? null : null;
  const inspectorPending = !!(run && location.query
    && (!selectedCurrent || selectedRegion.region.status === "loading" || selectedRegion.region.status === "idle"));
  const inspectorFailed = !!(run && location.query && selectedCurrent && selectedRegion.region.status === "error");

  // Changing the comparison cohort rewrites ONLY `runs=`. Protocol, subset, the inspected run and the
  // query stay exactly as they are, so the comparison is still about the same cohort and the same
  // sample — a peer choice must never navigate the reader somewhere else.
  const changePeers = (keys: string[]) => onGoto({ runs: keys });

  if (location.view === "compare") {
    return (
      <div className="stack">
        <Card title="So sánh mô hình" subtitle="So sánh một query trên nhiều lượt cùng protocol và subset">
          {!selectedQuery
            ? <EmptyState kind="none-selected" detail="Chọn một query ở Inspector trước; bảng so sánh dùng chính query đó."
                          next={<button type="button" onClick={() => onGoto({ view: "explorer" })}>Mở inspector</button>} />
            : <QueryInspector run={run} query={selectedQuery} condition={location.condition}
                              selectedCount={location.runs.length}
                              selectedRuns={location.runs}
                              onPeerChange={changePeers}
                              onAnnotate={() => onGoto({ area: "notebook", view: "annotations", query: selectedQuery.query_id })} />}
        </Card>
      </div>
    );
  }

  const selectQuery = (queryId: string) => onGoto({ query: queryId });

  return (
    <div className="stack" data-testid="area-samples">
      <Card
        title="Bộ lọc mẫu"
        subtitle={`${run ? `${run.model_id} / ${run.checkpoint_id}` : "chưa chọn lượt"} · ${protocolLabel(location.protocol)}${protocolHasSubsets(location.protocol) ? ` · ${SUBSET_LABEL[location.subset]}` : ""}`}
        actions={<button type="button" className="ghost"
                         onClick={() => onGoto({ category: "", outcome: "", query: "", q: "", queryState: "", order: "worst", offset: 0 })}>Xoá bộ lọc</button>}
      >
        <div className="toolbar sample-primary-filters">
          <Field label="Lượt" htmlFor="run-select">
            <select id="run-select" value={runKey ?? ""} data-testid="active-run"
                    onChange={(event) => onGoto({ runs: event.target.value ? [event.target.value] : [], query: "", offset: 0 })}>
              <option value="">— chọn lượt —</option>
              {catalog.filter((item) => item.family === "composition").map((item) => (
                <option key={item.key} value={item.key} disabled={!item.capabilities.flags.per_query_target_ranks}>
                  {item.label}{item.isTransfer ? " (TRANSFER)" : ""}{item.capabilities.flags.per_query_target_ranks ? "" : " — không có rank"}
                </option>
              ))}
            </select>
          </Field>
          <Field label="Điều kiện" htmlFor="condition-select" hint="Đổi điều kiện sẽ tải xếp hạng ứng viên riêng của điều kiện đó">
            <select id="condition-select" value={location.condition} data-testid="filter-condition"
                    onChange={(event) => onGoto({ condition: event.target.value as Location["condition"], offset: 0 })}>
              {CONDITIONS.map((condition) => <option key={condition.id} value={condition.id}>{condition.label}</option>)}
            </select>
          </Field>
          <Field label="Mã query" htmlFor="query-search">
            <input id="query-search" type="search" placeholder="vd 7:603" value={location.q}
                   data-testid="filter-query-id"
                   onChange={(event) => onGoto({ q: event.target.value, offset: 0 })} />
          </Field>
        </div>

        <Detail summary="Bộ lọc nâng cao và nguồn dữ liệu">
          <div className="toolbar">
            <Field label="Outcome (@10 cố định)" htmlFor="outcome-select">
              <select id="outcome-select" value={location.outcome} data-testid="filter-outcome"
                      onChange={(event) => onGoto({ outcome: event.target.value, offset: 0 })}>
                <option value="">Tất cả</option>
                {/* The count comes from the backend's `outcome_counts`, computed over the same
                    predicate the rows' own `outcome` uses. A menu that listed all 6,003 under
                    "Unscored" while every visible row read a real rank is the defect this fixes. */}
                {OUTCOMES.map((outcome) => (
                  <option key={outcome.id} value={outcome.id}>
                    {outcome.label}{page.data?.outcome_counts?.[outcome.id] != null
                      ? ` (${fmtInt(page.data.outcome_counts[outcome.id])})` : ""}
                  </option>
                ))}
              </select>
            </Field>
            <Field label="Danh mục" htmlFor="category-select">
              <select id="category-select" value={location.category} data-testid="filter-category"
                      onChange={(event) => onGoto({ category: event.target.value, offset: 0 })}>
                <option value="">Tất cả</option>
                {["dress", "shirt", "toptee"].map((category) => <option key={category} value={category}>{category}</option>)}
              </select>
            </Field>
            <Field label="Sắp xếp" htmlFor="order-select">
              <select id="order-select" value={location.order} data-testid="filter-order"
                      onChange={(event) => onGoto({ order: event.target.value as Location["order"], offset: 0 })}>
                {ORDERS.map((order) => <option key={order.id} value={order.id}>{order.label}</option>)}
              </select>
            </Field>
          </div>
          <div className="row" style={{ marginTop: 10 }}>
            <label className="badge" style={{ cursor: "pointer", textTransform: "none" }}>
              <input type="checkbox" checked={location.imported === "include"}
                     data-testid="include-imported"
                     onChange={(event) => onGoto({ imported: event.target.checked ? "include" : "exclude",
                                                   runs: [], query: "", offset: 0 })} />
              Hiện cả artifact import từ bản release
            </label>
            <span className="hint">
              Mặc định ẩn: artifact import là dữ liệu release thật nhưng không phải đầu ra của mô hình
              chạy ở đây. Khi bật, mỗi hàng/ứng viên đều ghi rõ nguồn.
            </span>
          </div>
          {location.category && run && (
            <p className="hint" style={{ marginTop: 8 }}>
              Lưu ý: release CIRCUS không gán <code>category</code> cho query (mọi query có category rỗng),
              nên bộ lọc danh mục sẽ trả 0 kết quả ở protocol này. Đây là dữ liệu, không phải lỗi bộ lọc.
            </p>
          )}
        </Detail>
      </Card>

      <div className="samples-layout">
        <Card title="Danh sách query"
              subtitle={page.status === "ready" ? `${fmtInt(total)} query khớp bộ lọc` : undefined}>
          {/* The backend's own three query states plus "all", as one control. Counts come from the
              page response and are computed BEFORE the state filter, so each chip shows its own
              total even while another chip is active. The selection lives in the URL
              (`query_state=`), so a reload and Back restore the same view; paging below keeps it. */}
          <div className="chiprow" role="group" aria-label="Trạng thái query" data-testid="query-state-chips"
               data-active={location.queryState || "all"}>
            {QUERY_STATE_CHIPS.map((chip) => {
              const counts = page.data?.state_counts;
              const count = chip.id === ""
                ? (counts ? counts.hit + counts.retrieval_failure + counts.unscored : (page.status === "ready" ? total : null))
                : counts?.[chip.id as keyof typeof counts] ?? null;
              return (
                <button key={chip.id || "all"} type="button" className="ghost chip"
                        aria-pressed={location.queryState === chip.id}
                        title={chip.hint}
                        data-testid={`chip-${chip.id || "all"}`}
                        disabled={!run}
                        onClick={() => onGoto({ queryState: chip.id as Location["queryState"], offset: 0 })}>
                  {chip.label}
                  {count != null && <span className="chip-count" data-testid={`chip-count-${chip.id || "all"}`}>{fmtInt(count)}</span>}
                </button>
              );
            })}
            <span className="chip-spacer" />
            {location.queryState && (
              <button type="button" className="ghost" data-testid="chip-clear"
                      onClick={() => onGoto({ queryState: "", offset: 0 })}>
                Xoá lọc trạng thái
              </button>
            )}
          </div>
          {!run
            ? <EmptyState kind="none-selected" detail="Chọn một lượt ở trên. Chỉ lượt có rank từng query mới duyệt được." />
            : page.status === "error"
              ? <ErrorState message={page.error ?? ""} onRetry={data.compositionPage.retry} />
              : page.status === "loading"
                ? <LoadingState label="Đang tải query…" />
                : items.length === 0
                  ? (
                    <EmptyState kind="no-match" testId="composition-empty"
                                detail="Không query nào khớp bộ lọc hiện tại."
                                next={<button type="button"
                                              onClick={() => onGoto({ category: "", outcome: "", q: "", query: "", offset: 0 })}>Xoá bộ lọc</button>} />
                  )
                  : (
                    <div className="stack">
                      <ul className="query-list" data-testid="query-list">
                        {items.map((item) => (
                          <li key={item.query_id}>
                            <button className="query-list-item" type="button"
                                    aria-current={selectedQuery?.query_id === item.query_id ? "true" : undefined}
                                    onClick={() => selectQuery(item.query_id)}>
                              <span className="query-list-heading">
                                <b data-testid="query-id">{item.query_id}</b>
                                {/* A rank plus the outcome word: "unresolved" here means the ranks
                                    exist and all sit outside @10, so it can never read as "chưa chấm". */}
                                <span>{item.outcome === "unscored" ? "Chưa chấm" : `MM ${fmtRank(item.ranks?.multimodal ?? null)}`}</span>
                              </span>
                              <span className="query-list-caption">{item.query_txt}</span>
                              <span className="query-list-ranks">
                                Text {fmtRank(item.ranks?.text_only_black_image ?? null)} ·
                                {" "}Ảnh {fmtRank(item.ranks?.image_only_empty_text ?? null)}
                              </span>
                            </button>
                          </li>
                        ))}
                      </ul>
                      <div className="pager" data-testid="pager-total" data-total={total} data-shown={items.length}>
                        <span>{total === 0 ? "0 query" : `${location.offset + 1}–${Math.min(location.offset + items.length, total)} / ${fmtInt(total)}`}</span>
                        <span className="spacer" />
                        <button type="button" disabled={location.offset <= 0}
                                onClick={() => onGoto({ offset: Math.max(0, location.offset - limit) })}>Trước</button>
                        <button type="button" data-testid="pager-next" disabled={location.offset + limit >= total}
                                onClick={() => onGoto({ offset: location.offset + limit })}>Sau</button>
                      </div>
                    </div>
                  )}
        </Card>

        <div className="stack">
          {inspectorPending
            ? <Card title="Duyệt ảnh truy hồi"><LoadingState label={`Đang tải đúng query ${location.query}…`} /></Card>
            : inspectorFailed
              ? <Card title="Duyệt ảnh truy hồi"><ErrorState message={selectedRegion.region.error ?? "Không tải được query"} onRetry={selectedRegion.retry} /></Card>
              : run && selectedQuery
                ? <QueryInspector run={run} query={selectedQuery} condition={location.condition}
                                  selectedCount={location.runs.length}
                                  selectedRuns={location.runs}
                                  onPeerChange={changePeers}
                                  onAnnotate={() => onGoto({ area: "notebook", view: "annotations", query: selectedQuery.query_id })} />
                : (
                  <Card title="Duyệt ảnh truy hồi">
                    <EmptyState kind="no-results"
                                detail={location.query
                                  ? `Không tìm thấy đúng query ${location.query} ở lượt này hoặc subset đang chọn. Không tự thay bằng query khác.`
                                  : "Chưa có query nào để hiển thị."}
                                next={<button type="button" onClick={() => onGoto({ area: "evaluation", view: "composition" })}>Chọn lượt khác</button>} />
                  </Card>
                )}
        </div>
      </div>
      {/* Not a per-query row: the two queries it covers have NO canonical row anywhere on this page.
          It is rendered here so a reader looking for them finds the comparison instead of assuming
          the query simply does not exist, and it states the canonical count it does not change. */}
      <LimnScopeDiagnostic pageSubset={location.subset} />
    </div>
  );
}
