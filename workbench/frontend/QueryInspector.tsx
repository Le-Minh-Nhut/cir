import { useCallback, useEffect, useMemo, useState } from "react";
import { BASE } from "./client";
import { fetchJson, useRegion } from "./useRegion";
import { conditionsOf, outcomeTone } from "./inspector";
import { Badge, Card, Detail, EmptyState, ErrorState, KeyValue, LoadingState } from "./ui";
import { fmtInt, fmtRank } from "./format";

/** How an imported candidate source is named to a reader. Never the model whose row it sits on. */
const IMPORTED_LABEL: Record<string, string> = {
  released_audit_retriever: "released CIRCUS audit retriever",
  released_artifact_import: "released artifact import",
  external_artifact: "external artifact",
};
import { FIXED_CUTOFFS } from "./catalog";
import type { CompositionQuery, CompositionRun } from "./types";

/** One query, in one run: reference image, edit text, target image, the three condition ranks, and
 *  the candidate list when the run actually exported one.
 *
 *  The honesty rules that matter here:
 *  * a missing condition rank is `unscored` and says WHY it has no rank — it is never shown as a
 *    loss at rank 0 and never mixed into the failure statistics;
 *  * when `candidate_topk` is false the candidate region says "candidates not exported for this
 *    run" with the capability reason. No placeholder images, no synthesis;
 *  * images that 404 in this deployment render an explicit missing tile with the image id, so a
 *    coverage gap is visible instead of an empty box.
 */
export function QueryInspector({ run, query, selectedCount, onAnnotate }: {
  run: CompositionRun | null;
  query: CompositionQuery | null;
  selectedCount: number;
  onAnnotate: () => void;
}) {
  const [k, setK] = useState(10);
  const candidates = query?.top_results ?? [];
  const hasCandidates = candidates.length > 0;
  // The exact rank is a full-gallery fact and survives outside the stored window. It is never
  // recomputed or clamped from `top_results`; this only says whether it lies outside it.
  const depth = query?.candidate_depth ?? null;
  const importedSource = run?.provenance_source ?? null;
  const candidateReason = query?.candidate_reason
    ?? run?.capability_detail?.candidate_topk?.reason
    ?? (run?.capability_detail?.candidate_topk?.status
      ? `Trạng thái do backend báo: ${run.capability_detail.candidate_topk.status}.`
      : "Lượt này không lưu danh sách ứng viên.");

  useEffect(() => setK(10), [query?.query_id, run?.run_id]);

  // Candidate images from the backend arrive as `/api/queries`-style `top_results` on the
  // composition query payload. They are consumed only when present; absence is a stated capability
  // gap, so nothing here fabricates a list.
  const shown = useMemo(() => candidates.slice(0, k), [candidates, k]);

  if (!run) {
    return (
      <Card title="Inspector">
        <EmptyState kind="none-selected" detail="Chọn một lượt ở khu vực Đánh giá hoặc Thí nghiệm để xem query."
                    next={<button type="button" onClick={onAnnotate}>Mở sổ nghiên cứu</button>} />
      </Card>
    );
  }
  if (!query) {
    return (
      <Card title="Inspector" subtitle={`${run.model_id} / ${run.checkpoint_id}`}>
        <EmptyState kind="none-selected" detail="Chọn một query ở danh sách bên trái, hoặc nới bộ lọc để có query." />
      </Card>
    );
  }

  const conditions = conditionsOf(query);
  const targetId = query.positive_ids[0] ?? "";
  const category = query.category || "circus";

  return (
    <div className="inspector" data-testid="query-inspector">
      <Card
        title={<span data-testid="query-title">{query.query_id}</span>}
        subtitle={`${run.model_id} / ${run.checkpoint_id} · ${run.protocol_id} · subset ${run.subset}`}
        actions={(
          <>
            <Badge tone={outcomeTone(query.outcome)}>{query.outcome}</Badge>
            {query.category && <Badge>{query.category}</Badge>}
          </>
        )}
      >
        <div className="stack">
          <div>
            <div className="field-label">Văn bản sửa (edit text)</div>
            <p style={{ marginTop: 4 }}>{query.query_txt}</p>
          </div>

          <div className="shots">
            <Shot category={category} id={query.reference_id} label="Ảnh tham chiếu" />
            <Shot category={category} id={targetId} label="Ảnh đích (ground truth)" target />
            {query.positive_ids.slice(1).map((id) => (
              <Shot key={id} category={category} id={id} label="Ảnh đích phụ" target />
            ))}
          </div>

          <div>
            <div className="field-label" style={{ marginBottom: 6 }}>
              Hạng theo điều kiện · phân loại outcome tại @{FIXED_CUTOFFS.compositionOutcomeK} (cố định)
            </div>
            <div className="rankgrid">
              {conditions.map((condition) => (
                <div key={condition.id}
                     className={`rankcell ${condition.rank === null ? "unscored-cell" : condition.rank <= FIXED_CUTOFFS.compositionOutcomeK ? "hit" : "miss"}`}
                     data-testid={`rank-${condition.id}`}>
                  <small>{condition.short}</small>
                  <b>{fmtRank(condition.rank)}</b>
                  <span className="hint">
                    {condition.rank === null
                      ? (query.unscored_kind
                        ? `không có hạng chính xác (${query.unscored_kind}) — lỗi pipeline/ánh xạ, không phải lỗi truy hồi`
                        : "không có hạng chính xác — lỗi pipeline/ánh xạ, không phải lỗi truy hồi")
                      : condition.rank <= FIXED_CUTOFFS.compositionOutcomeK ? `trong @${FIXED_CUTOFFS.compositionOutcomeK}` : `ngoài @${FIXED_CUTOFFS.compositionOutcomeK}`}
                    {condition.rank != null && depth != null && condition.rank > depth
                      ? ` · ngoài cửa sổ ${fmtInt(depth)} ứng viên đã lưu`
                      : ""}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {query.failure && (
            <p className="hint-bad">Lý do do evaluator ghi: {query.failure}</p>
          )}

          <div>
            <div className="toolbar" style={{ marginBottom: 8 }}>
              <div className="field">
                <label className="field-label" htmlFor="candidate-depth">Số ứng viên hiển thị</label>
                <select id="candidate-depth" data-testid="topk" value={k} onChange={(event) => setK(Number(event.target.value))}>
                  {[1, 5, 10, 20, 50, 100].map((value) => <option key={value} value={value}>Top {value}</option>)}
                </select>
              </div>
              <span className="hint" style={{ paddingBottom: 8 }}>{FIXED_CUTOFFS.candidateDisplayNote}</span>
            </div>

            {!hasCandidates ? (
              <EmptyState kind="capability" testId="candidates-unavailable"
                          title="Ứng viên chưa được export cho lượt này"
                          detail={<>
                            <span className="mono" style={{ fontSize: 11.5 }}>{candidateReason}</span>
                            <br />
                            Lượt này vẫn có hạng chính xác cho từng query, nên đây là giới hạn của dữ liệu
                            chứ không phải lỗi tải hay một lượt hỏng.
                          </>}
                          next={<span className="hint">Hạng MM / text / image ở trên vẫn đầy đủ cho query này.</span>} />
            ) : (
              <>
                {importedSource && (
                  <p className="hint" data-testid="candidates-provenance" style={{ marginBottom: 8 }}>
                    <Badge tone="warn">NGUỒN: {importedSource}</Badge>{" "}
                    Ứng viên dưới đây là đầu ra của <b>{IMPORTED_LABEL[importedSource] ?? importedSource}</b>,
                    không phải đầu ra của mô hình <b>{run.model_id}</b>. Chỉ dùng để đọc hạng và thứ tự.
                  </p>
                )}
                <div className="candidates" data-testid="topk-grid">
                  {shown.map((candidate) => (
                    <div key={candidate.rank}
                         className={`candidate ${candidate.image_id === targetId ? "is-target" : ""}`}>
                      <Shot category={category} id={candidate.image_id} label={`#${candidate.rank}`} compact target={candidate.image_id === targetId} />
                      <small>
                        <span>#{candidate.rank}</span>
                        <span>{"gallery_position" in candidate && candidate.gallery_position != null
                          ? `g ${candidate.gallery_position.toLocaleString("en-US")}`
                          : ""}</span>
                        {/* A stored order without scores must read "—", never 0.0. */}
                        <span>{candidate.score == null ? "—" : candidate.score.toFixed(4)}</span>
                      </small>
                    </div>
                  ))}
                </div>
                <p className="hint" style={{ marginTop: 6 }}>
                  Hiển thị {shown.length} / {candidates.length} ứng viên trong cửa sổ đã lưu
                  {depth ? ` (độ sâu lưu: ${fmtInt(depth)})` : ""}. Danh sách này KHÔNG phải ngưỡng phân
                  loại outcome.
                  {query.target_rank != null && depth != null && query.target_rank > depth && (
                    <> Hạng đích <b>{fmtRank(query.target_rank)}</b> nằm ngoài cửa sổ hiển thị — đó là hạng
                      trên toàn gallery, không phải vị trí trong danh sách này.</>
                  )}
                </p>
              </>
            )}
          </div>
        </div>
      </Card>

      <div className="inspector-secondary">
        <Card title="So sánh trên cùng query" subtitle="Cùng protocol, cùng subset: cohort và mẫu số được nêu">
          <ComparePanel run={run} query={query} />
        </Card>
        <Card title="Ghi chú nghiên cứu" subtitle="Gắn đúng query + protocol, lưu qua reload">
          <p className="hint">
            Query <code>{query.query_id}</code> · protocol <code>{run.protocol_id}</code> · lượt <code>{run.model_id}</code>.
            {selectedCount > 1 ? ` Đang chọn ${selectedCount} lượt để phân tích.` : ""}
          </p>
          <button type="button" className="primary" style={{ marginTop: 10 }} onClick={onAnnotate}>
            Mở sổ nghiên cứu cho query này
          </button>
        </Card>
      </div>
    </div>
  );
}

function Shot({ category, id, label, target, compact }: {
  category: string;
  id: string;
  label: string;
  target?: boolean;
  compact?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const clean = id.split("/").pop() || id;
  useEffect(() => setFailed(false), [id]);
  return (
    <figure className={`shot ${target ? "target" : ""} ${compact ? "compact" : ""}`} style={{ margin: 0 }}>
      {failed
        ? <div className="shot-missing" data-testid="query-image" data-image-id={clean}>Ảnh không có trong deployment<br />{clean}</div>
        : <img src={`${BASE}/api/images/${category}/${encodeURIComponent(clean)}`}
               alt={`${label}: ${clean}`} loading="lazy" data-testid="query-image"
               data-image-id={clean}
               onError={() => setFailed(true)} />}
      <figcaption className="shot-label">
        <span>{label}</span>
        <span className="mono" title={clean}>{clean.length > 16 ? `${clean.slice(0, 14)}…` : clean}</span>
      </figcaption>
    </figure>
  );
}

/** Cross-run comparison for one query, restricted to the same protocol and subset.
 *  A different cohort would make the columns incomparable, so it is refused with the reason.
 *  Both reads are keyed regions (protocol+subset for the peer list, query+selection for the ranks),
 *  so a query change cannot leave the previous query's comparison on screen. */
function ComparePanel({ run, query }: { run: CompositionRun; query: CompositionQuery }) {
  const peerKey = `${run.protocol_id}\u001f${run.subset}`;
  const peersRegion = useRegion<CompositionRun[]>(
    "compare-peers", peerKey,
    useCallback((signal: AbortSignal) => fetchJson<CompositionRun[]>(`/api/composition/runs?subset=${run.subset}`, signal),
      [run.subset]));

  const peers = useMemo(
    () => (peersRegion.region.data ?? [])
      .filter((candidate) => candidate.protocol_id === run.protocol_id && candidate.subset === run.subset)
      .filter((candidate) => candidate.run_id !== run.run_id)
      .slice(0, 4),
    [peersRegion.region.data, run.protocol_id, run.subset, run.run_id]);

  const rankKey = `${run.run_id}\u001f${query.query_id}\u001f${peers.map((peer) => peer.run_id).join(",")}`;
  const ranksRegion = useRegion<Array<{ key: string; rank: number | null }>>(
    "compare-ranks", rankKey,
    useCallback(async (signal: AbortSignal) => {
      const keys = [run.run_id, ...peers.map((peer) => peer.run_id)];
      return Promise.all(keys.map(async (key) => {
        const params = new URLSearchParams({
          subset: run.subset, condition: "multimodal", order: "worst", limit: "1", offset: "0",
          query_id: query.query_id,
        });
        const payload = await fetchJson<{ items: CompositionQuery[] }>(
          `/api/composition/runs/${encodeURIComponent(key)}/queries?${params}`, signal);
        return { key, rank: payload.items[0]?.ranks?.multimodal ?? null };
      }));
    }, [run.run_id, run.subset, query.query_id, peers.map((peer) => peer.run_id).join(",")]));

  if (peersRegion.region.status === "error") {
    return <ErrorState message={peersRegion.region.error ?? ""} onRetry={peersRegion.retry} />;
  }
  if (peersRegion.region.status === "loading") return <LoadingState label="Đang tải các lượt cùng scope…" />;
  if (!peers.length) {
    return <EmptyState kind="capability" detail="Chỉ có một lượt trong protocol và subset này, nên không có gì để so sánh." />;
  }
  if (ranksRegion.region.status === "error") {
    return <ErrorState message={ranksRegion.region.error ?? ""} onRetry={ranksRegion.retry} />;
  }
  if (ranksRegion.region.status !== "ready" || !ranksRegion.region.data) {
    return <LoadingState label="Đang so sánh…" />;
  }
  const rows = ranksRegion.region.data;

  return (
    <div className="stack">
      <div className="table-wrap">
        <table>
          <caption className="sr-only">Hạng multimodal của cùng query trên các lượt cùng scope</caption>
          <thead>
            <tr><th scope="col">Lượt</th><th scope="col" className="num">Hạng MM</th></tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const label = row.key === run.run_id
                ? `${run.model_id} / ${run.checkpoint_id} (đang xem)`
                : peers.find((peer) => peer.run_id === row.key)?.model_id ?? row.key;
              return (
                <tr key={row.key} aria-selected={row.key === run.run_id}>
                  <td>{label}</td>
                  <td className="num">{fmtRank(row.rank)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="hint">
        Cohort: {rows.length} lượt, cùng protocol <code>{run.protocol_id}</code> và subset <code>{run.subset}</code>,
        mỗi lượt đều có hạng cho đúng query này. Ảnh và rank của query không đổi khi so sánh.
      </p>
      <Detail summary="Mẫu số và giới hạn">
        <KeyValue rows={[
          ["query", <code key="q">{query.query_id}</code>],
          ["protocol", <code key="p">{run.protocol_id}</code>],
          ["subset", run.subset],
          ["lượt so sánh", fmtInt(rows.length)],
          ["ứng viên", (query.top_results ?? []).length ? `${fmtInt((query.top_results ?? []).length)} đã lưu` : "không có (chưa export)"],
        ]} />
      </Detail>
    </div>
  );
}
