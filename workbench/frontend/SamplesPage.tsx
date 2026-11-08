import { useMemo } from "react";
import type { CompositionQuery, CompositionRun } from "./types";
import type { WorkbenchData } from "./data";
import { protocolLabel } from "./data";
import type { Location } from "./routes";
import { CONDITIONS, ORDERS, OUTCOMES, SUBSET_LABEL } from "./routes";
import { Card, Detail, EmptyState, ErrorState, Field, LoadingState } from "./ui";
import { fmtInt, fmtRank } from "./format";
import { QueryInspector } from "./QueryInspector";
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

  if (location.view === "compare") {
    return (
      <div className="stack">
        <Card title="So sánh mô hình" subtitle="So sánh một query trên nhiều lượt cùng protocol và subset">
          {!selectedQuery
            ? <EmptyState kind="none-selected" detail="Chọn một query ở Inspector trước; bảng so sánh dùng chính query đó."
                          next={<button type="button" onClick={() => onGoto({ view: "explorer" })}>Mở inspector</button>} />
            : <QueryInspector run={run} query={selectedQuery} condition={location.condition}
                              selectedCount={location.runs.length}
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
        subtitle={`${run ? `${run.model_id} / ${run.checkpoint_id}` : "chưa chọn lượt"} · ${protocolLabel(location.protocol)} · ${SUBSET_LABEL[location.subset]}`}
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
            <Field label="Trạng thái query" htmlFor="state-select"
                   hint="Phân loại của backend, tách unscored khỏi lỗi truy hồi">
              <select id="state-select" value={location.queryState} data-testid="filter-query-state"
                      onChange={(event) => onGoto({ queryState: event.target.value as Location["queryState"], offset: 0 })}>
                <option value="">Tất cả</option>
                <option value="hit">Hit (trong @10)</option>
                <option value="retrieval_failure">Retrieval failure (có hạng, ngoài @10)</option>
                <option value="unscored">Unscored (pipeline/ánh xạ)</option>
              </select>
            </Field>
            <Field label="Outcome (@10 cố định)" htmlFor="outcome-select">
              <select id="outcome-select" value={location.outcome} data-testid="filter-outcome"
                      onChange={(event) => onGoto({ outcome: event.target.value, offset: 0 })}>
                <option value="">Tất cả</option>
                {OUTCOMES.map((outcome) => <option key={outcome.id} value={outcome.id}>{outcome.label}</option>)}
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
    </div>
  );
}
