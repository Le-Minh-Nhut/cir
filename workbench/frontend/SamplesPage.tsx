import type { CompositionRun } from "./types";
import type { WorkbenchData } from "./data";
import { protocolLabel } from "./data";
import type { Location } from "./routes";
import { CONDITIONS, ORDERS, OUTCOMES, SUBSET_LABEL } from "./routes";
import { Badge, Card, EmptyState, ErrorState, Field, LoadingState } from "./ui";
import { fmtInt, fmtRank } from "./format";
import { QueryInspector } from "./QueryInspector";

/** Khám phá mẫu: browse queries of one run, then inspect one.
 *
 *  The list and the inspector are two regions with two request identities: paging the list must not
 *  blank the inspector, and jumping to a query must not be overridden by the list's answer.
 *  Every filter is in the URL, so a link reproduces the exact view.
 */
export function SamplesPage({ data, location, onGoto, onAnnotate }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
  onAnnotate: () => void;
}) {
  const catalog = data.catalog;
  const runKey = location.runs[0];
  const entry = catalog.find((item) => item.key === runKey) ?? null;
  const run: CompositionRun | null = entry?.composition ?? null;
  const page = data.compositionPage.region;
  const items = page.data?.items ?? [];
  const total = page.data?.total ?? 0;
  const selectedQuery = items.find((item) => item.query_id === location.query) ?? items[0] ?? null;
  const limit = 20;

  if (location.view === "compare") {
    return (
      <div className="stack">
        <Card title="So sánh mô hình" subtitle="So sánh một query trên nhiều lượt cùng protocol và subset">
          {!selectedQuery
            ? <EmptyState kind="none-selected" detail="Chọn một query ở Inspector trước; bảng so sánh dùng chính query đó."
                          next={<button type="button" onClick={() => onGoto({ view: "explorer" })}>Mở inspector</button>} />
            : <QueryInspector run={run} query={selectedQuery} selectedCount={location.runs.length} onAnnotate={onAnnotate} />}
        </Card>
      </div>
    );
  }

  const reasons: string[] = [];
  if (!run) reasons.push("chưa chọn lượt có rank từng query");

  return (
    <div className="stack" data-testid="area-samples">
      <Card
        title="Bộ lọc mẫu"
        subtitle={`${run ? `${run.model_id} / ${run.checkpoint_id}` : "chưa chọn lượt"} · ${protocolLabel(location.protocol)} · ${SUBSET_LABEL[location.subset]}`}
        actions={<button type="button" className="ghost" onClick={() => onGoto({ category: "", outcome: "", query: "", order: "worst", offset: 0 })}>Xoá bộ lọc</button>}
      >
        <div className="toolbar">
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
          <Field label="Điều kiện" htmlFor="condition-select" hint="Hạng dùng để sắp xếp và phân loại outcome">
            <select id="condition-select" value={location.condition} data-testid="filter-condition"
                    onChange={(event) => onGoto({ condition: event.target.value as Location["condition"], offset: 0 })}>
              {CONDITIONS.map((condition) => <option key={condition.id} value={condition.id}>{condition.label}</option>)}
            </select>
          </Field>
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
          <Field label="Mã query" htmlFor="query-search">
            <input id="query-search" type="search" placeholder="vd 7:603" value={location.q}
                   data-testid="filter-query-id"
                   onChange={(event) => onGoto({ q: event.target.value, offset: 0 })} />
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
            Lưu ý: release CIRCUS không gán <code>category</code> cho query (mọi query có category rỗng), nên
            bộ lọc danh mục sẽ trả 0 kết quả ở protocol này. Đây là dữ liệu, không phải lỗi bộ lọc.
          </p>
        )}
      </Card>

      <div className="samples-layout">
        <Card title="Danh sách query" subtitle={page.status === "ready" ? `${fmtInt(total)} query khớp bộ lọc` : undefined}>
          {!run
            ? <EmptyState kind="none-selected" detail="Chọn một lượt ở trên. Chỉ lượt có rank từng query mới duyệt được." />
            : page.status === "error"
              ? <ErrorState message={page.error ?? ""} onRetry={data.compositionPage.retry} />
              : page.status === "loading"
                ? <LoadingState label="Đang tải query…" />
                : items.length === 0
                  ? <EmptyState kind="no-match" testId="composition-empty"
                                detail="Không query nào khớp bộ lọc hiện tại."
                                next={<button type="button" onClick={() => onGoto({ category: "", outcome: "", q: "", query: "", offset: 0 })}>Xoá bộ lọc</button>} />
                  : (
                    <div className="stack">
                      <div className="table-wrap">
                        <table data-testid="query-list">
                          <caption className="sr-only">Query trong lượt đang chọn</caption>
                          <thead>
                            <tr>
                              <th scope="col">Query</th>
                              <th scope="col">Outcome</th>
                              <th scope="col" className="num">MM</th>
                              <th scope="col" className="num">Text</th>
                              <th scope="col" className="num">Image</th>
                            </tr>
                          </thead>
                          <tbody>
                            {items.map((item) => (
                              <tr key={item.query_id} aria-selected={selectedQuery?.query_id === item.query_id}
                                  onClick={() => onGoto({ query: item.query_id })}>
                                <td><button type="button" className="rowbtn" data-testid="query-id">{item.query_id}</button></td>
                                <td><Badge tone={item.outcome === "unscored" ? "neutral" : item.outcome === "composition_required" ? "ok" : "warn"}>{item.outcome}</Badge></td>
                                <td className="num">{fmtRank(item.ranks?.multimodal ?? null)}</td>
                                <td className="num">{fmtRank(item.ranks?.text_only_black_image ?? null)}</td>
                                <td className="num">{fmtRank(item.ranks?.image_only_empty_text ?? null)}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                      <div className="pager" data-testid="pager-total"
                           data-total={total} data-shown={items.length}>
                        <span>{total === 0 ? "0 query" : `${location.offset + 1}–${Math.min(location.offset + items.length, total)} / ${fmtInt(total)}`}</span>
                        <span className="spacer" />
                        <button type="button" disabled={location.offset <= 0}
                                onClick={() => onGoto({ offset: Math.max(0, location.offset - limit) })}>Trước</button>
                        <button type="button" data-testid="pager-next"
                                disabled={location.offset + limit >= total}
                                onClick={() => onGoto({ offset: location.offset + limit })}>Sau</button>
                      </div>
                    </div>
                  )}
        </Card>

        <div className="stack">
          {selectedQuery
            ? <QueryInspector run={run} query={selectedQuery} selectedCount={location.runs.length} onAnnotate={onAnnotate} />
            : <Card title="Inspector">
                <EmptyState kind="no-results" detail="Chưa có query nào để hiển thị."
                            next={<button type="button" onClick={() => onGoto({ area: "evaluation", view: "composition" })}>Chọn lượt khác</button>} />
              </Card>}
        </div>
      </div>
    </div>
  );
}
