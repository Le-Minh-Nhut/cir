import { useMemo, useState } from "react";
import type { WorkbenchData } from "./data";
import { ANALYSIS_K, type Location } from "./routes";
import { PRESETS, buildScope, filterRows, paginate, sortRows, withCutoff, type PresetId, type RunPage } from "./analysis";
import { Badge, Card, Detail, EmptyState, ErrorState, Field, KeyValue, LoadingState, SaveFeedback, type SaveState } from "./ui";
import { fmtFractionPercent, fmtInt, fmtRank } from "./format";
import { FIXED_CUTOFFS } from "./catalog";

/** Phân tích lỗi: consensus across the selected runs, with the cohort stated.
 *
 *  The page has to answer four different questions that used to look identical:
 *  * nothing selected → say so and offer the selectable runs;
 *  * no per-query ranks in this scope → name the capability that is missing;
 *  * a filter that matches nothing → say it is the filter, and offer to clear it;
 *  * a failed request → an error with retry.
 *
 *  "Save filtered set" writes the WHOLE filtered set, not the visible page: the count is shown
 *  before and after the save, and a failure is reported rather than swallowed.
 */
export function FailuresPage({ data, location, onGoto, onSaved }: {
  data: WorkbenchData;
  location: Location;
  onGoto: (patch: Partial<Location>) => void;
  onSaved: () => void;
}) {
  const universe = data.compositionUniverse.region;
  const indexUniverse = data.indexUniverse.region;
  const selected = location.runs;
  const [page, setPage] = useState(0);
  const [state, setState] = useState<SaveState>({ status: "idle" });
  const limit = 25;
  const preset = (location.preset || "all") as PresetId;

  const pages = useMemo<RunPage[]>(() => {
    const label = (key: string) => data.catalog.find((item) => item.key === key)?.label ?? key;
    const composition: RunPage[] = (universe.data ?? []).map((entry) => ({
      runKey: entry.key,
      runLabel: label(entry.key),
      items: entry.items,
    }));
    // The per-query index speaks a different vocabulary: `top_results[].rank` order, no conditions.
    // Its target rank is the only common quantity, so only that is joined; the condition columns
    // stay empty rather than being filled with a guess.
    const index: RunPage[] = (indexUniverse.data ?? []).map((entry) => ({
      runKey: entry.key,
      runLabel: label(entry.key),
      items: entry.items.map((query) => ({
        query_id: query.query_id,
        category: query.category,
        query_txt: query.raw_captions.join(" / "),
        reference_id: query.reference_id,
        positive_ids: [query.target_id],
        ranks: { multimodal: query.target_rank },
        target_rank: query.target_rank,
        outcome: query.target_rank <= FIXED_CUTOFFS.compositionOutcomeK ? "composition_required" : "unresolved",
      })),
    }));
    return [...composition, ...index];
  }, [universe.data, indexUniverse.data, data.catalog]);

  const scope = useMemo(() => withCutoff(buildScope(pages), location.k), [pages, location.k]);
  const filtered = useMemo(() => sortRows(filterRows(scope, {
    preset,
    k: location.k,
    category: location.category,
    outcome: location.outcome,
    scoring: location.outcome === "unscored" ? "unscored" : "",
    query: location.q,
  }), location.order === "best" ? "best" : location.order === "desc" ? "worst" : "consensus"),
  [scope, preset, location.k, location.category, location.outcome, location.q, location.order]);
  const pageWindow = paginate(filtered, Math.min(page, Math.max(0, Math.ceil(filtered.length / limit) - 1)) * limit, limit);

  const loading = (universe.status === "loading" && selected.length > 0) || (indexUniverse.status === "loading" && selected.length > 0);
  const failure = universe.status === "error" ? universe.error : indexUniverse.status === "error" ? indexUniverse.error : null;

  async function saveCohort() {
    if (!filtered.length) return;
    const cohortId = `cohort-${new Date().toISOString().replace(/[-:.]/g, "").slice(0, 15)}`;
    setState({ status: "saving" });
    try {
      const runKeys = pages.map((entry) => entry.runKey);
      const hasComposition = runKeys.some((key) => key.startsWith("c:"));
      const hasIndex = runKeys.some((key) => key.startsWith("i:"));
      if (hasComposition && hasIndex) {
        // The backend refuses this too (`mixed_data_kind`), and it must: a CIRCUS rank and an index
        // top-K are different quantities. Saying so here is the same rule stated before the round
        // trip, not a second rule.
        throw new Error("Không thể trộn lượt CIRCUS với lượt của chỉ mục per-query trong một cohort: hai loại báo hai đại lượng khác nhau.");
      }
      // A composition reference keeps its `c:` prefix: that prefix is what tells the backend which
      // arm validates it. An index reference is sent bare, exactly as before, so every cohort saved
      // through this page still loads.
      const runIds = runKeys.map((key) => (key.startsWith("c:") ? key : key.slice(2)));
      const response = await fetch("/api/cohorts", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cohort_id: cohortId,
          protocol_id: location.protocol,
          query_ids: filtered.map((row) => row.query_id),
          run_ids: runIds,
          definition: { preset, k: location.k, category: location.category, outcome: location.outcome,
                        subset: location.subset,
                        scoring: "unscored-separate", cohort: scope.cohort, partial: scope.partial },
          notes: `saved from failure analysis; ${filtered.length} of ${scope.rows.length} rows`,
        }),
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.message || payload?.detail?.message || `${response.status}`);
      }
      setState({ status: "saved" });
      data.cohorts.retry();
      onSaved();
    } catch (cause) {
      setState({ status: "error", message: cause instanceof Error ? cause.message : String(cause) });
    }
  }

  return (
    <div className="stack" data-testid="area-failures">
      <Card
        title="Phân tích lỗi"
        subtitle={selected.length === 0
          ? "Chưa chọn lượt nào"
          : `${selected.length} lượt · cohort ${fmtInt(scope.cohort)} query có mặt ở mọi lượt · ${fmtInt(scope.partial)} query chỉ có ở một phần · ${fmtInt(scope.unscored)} unscored`}
        actions={(
          <>
            <Field label="Preset" htmlFor="preset">
              <select id="preset" data-testid="failure-preset" value={preset}
                      onChange={(event) => { setPage(0); onGoto({ preset: event.target.value }); }}>
                {PRESETS.map((item) => <option key={item.id} value={item.id}>{item.label}</option>)}
              </select>
            </Field>
            <Field label="Cutoff @K" htmlFor="analysis-k" hint="Ngưỡng đồng thuận; khác với @10 cố định của outcome">
              <select id="analysis-k" value={location.k}
                      onChange={(event) => { setPage(0); onGoto({ k: Number(event.target.value) }); }}>
                {ANALYSIS_K.map((value) => <option key={value} value={value}>@{value}</option>)}
              </select>
            </Field>
          </>
        )}
      >
        {selected.length === 0 ? (
          <EmptyState kind="none-selected"
                      detail="Chọn ít nhất hai lượt có rank từng query ở khu vực Thí nghiệm (cùng protocol và subset) để phân tích lỗi."
                      next={<button type="button" onClick={() => onGoto({ area: "experiments", view: "runs" })}>Chọn lượt</button>} />
        ) : failure ? (
          <ErrorState message={failure} onRetry={() => { data.compositionUniverse.retry(); data.indexUniverse.retry(); }} />
        ) : loading ? (
          <LoadingState label="Đang tải toàn bộ query của các lượt đã chọn…" />
        ) : scope.rows.length === 0 ? (
          <EmptyState kind="capability"
                      detail="Các lượt đang chọn không có rank từng query (lượt lỗi hoặc chỉ có chỉ số tổng hợp)."
                      next={<button type="button" onClick={() => onGoto({ area: "evaluation", view: "composition" })}>Xem bảng đánh giá</button>} />
        ) : filtered.length === 0 ? (
          <EmptyState kind="no-match"
                      detail={`Bộ lọc hiện tại (preset "${preset}", @${location.k}${location.outcome ? `, outcome ${location.outcome}` : ""}) không khớp query nào trong cohort.`}
                      next={<>
                        <button type="button" onClick={() => onGoto({ preset: "all", outcome: "", category: "", q: "" })}>Xoá bộ lọc</button>
                        <button type="button" className="ghost" onClick={() => onGoto({ outcome: "unscored" })}>Xem {fmtInt(scope.unscored)} unscored</button>
                      </>} />
        ) : (
          <div className="stack">
            <div className="toolbar">
              <Field label="Lưu cohort" htmlFor="cohort-id" hint="ID tự sinh; ghi chú kèm phạm vi và preset">
                <button type="button" id="cohort-id" className="primary" data-testid="cohort-save"
                        data-scope-count={filtered.length}
                        data-scope-cohort={scope.cohort}
                        onClick={() => void saveCohort()} disabled={state.status === "saving"}>
                  Lưu tập đã lọc ({fmtInt(filtered.length)} query)
                </button>
              </Field>
              <SaveFeedback state={state} count={filtered.length} />
            </div>
            <div className="table-wrap">
              <table>
                <caption className="sr-only">Query lỗi theo đồng thuận</caption>
                <thead>
                  <tr>
                    <th scope="col">Query</th>
                    <th scope="col">Outcome</th>
                    <th scope="col" className="num">Trượt</th>
                    <th scope="col" className="num">Min</th>
                    <th scope="col" className="num">Median</th>
                    <th scope="col" className="num">Max</th>
                  </tr>
                </thead>
                <tbody>
                  {pageWindow.items.map((row) => (
                    <tr key={row.query_id} data-testid="analysis-row" data-query-id={row.query_id}
                        onClick={() => onGoto({ area: "samples", view: "explorer", query: row.query_id })}>
                      <td><button type="button" className="rowbtn">{row.query_id}</button></td>
                      <td>
                        <Badge tone={row.unscored ? "neutral" : row.outcome === "composition_required" ? "ok" : "warn"}>
                          {row.unscored ? "unscored" : row.outcome}
                        </Badge>
                      </td>
                      <td className="num">{row.unscored ? "—" : fmtFractionPercent(row.failFraction, 0)}</td>
                      <td className="num">{fmtRank(row.minRank)}</td>
                      <td className="num">{fmtRank(row.medianRank)}</td>
                      <td className="num">{fmtRank(row.maxRank)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="pager" data-testid="pager-total" data-total={filtered.length} data-shown={pageWindow.items.length}>
              <span>{pageWindow.offset + 1}–{Math.min(pageWindow.offset + pageWindow.items.length, filtered.length)} / {fmtInt(filtered.length)}</span>
              <span className="spacer" />
              <button type="button" disabled={page === 0} onClick={() => setPage(page - 1)}>Trước</button>
              <button type="button" data-testid="pager-next"
                      disabled={pageWindow.offset + limit >= filtered.length}
                      onClick={() => setPage(page + 1)}>Sau</button>
            </div>
            <Detail summary="Cách tính và giới hạn của trang này">
              <KeyValue rows={[
                ["cohort", `${fmtInt(scope.cohort)} query có mặt trong mọi lượt đã chọn`],
                ["partial", `${fmtInt(scope.partial)} query chỉ có ở một phần các lượt — không vào consensus, không bị bỏ im lặng`],
                ["unscored", `${fmtInt(scope.unscored)} query không có hạng chính xác — tách riêng, không xếp "worst"`],
                ["cutoff đồng thuận", `@${location.k} (điều khiển được ở trên)`],
                ["outcome", `@${FIXED_CUTOFFS.compositionOutcomeK} cố định theo protocol — không đổi khi đổi @K ở trên`],
                ["thứ tự", "unscored luôn ở cuối cho mọi kiểu xếp theo hạng"],
              ]} />
            </Detail>
          </div>
        )}
      </Card>
    </div>
  );
}
