import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BASE } from "./client";
import { fetchJson, useRegion } from "./useRegion";
import { conditionsOf, outcomeTone } from "./inspector";
import { candidateWindowVerdict } from "./scope";
import { Badge, Card, Detail, EmptyState, ErrorState, Field, KeyValue, LoadingState } from "./ui";
import { fmtInt, fmtRank } from "./format";
import { FIXED_CUTOFFS, comparePeers, compositionLabel } from "./catalog";
import { candidateDisplayOptions, candidateRequestPath, effectiveCandidateK } from "./candidateRequest";
import { outcomeLabel } from "./routes";
import type { CompositionQuery, CompositionRun } from "./types";

/** How an imported candidate source is named to a reader. Never the model whose row it sits on. */
const IMPORTED_LABEL: Record<string, string> = {
  released_audit_retriever: "released CIRCUS audit retriever",
  released_artifact_import: "released artifact import",
  external_artifact: "external artifact",
};

const CONDITION_LABELS: Record<string, string> = {
  multimodal: "Ảnh + văn bản (MM)",
  text_only_black_image: "Chỉ văn bản",
  image_only_empty_text: "Chỉ ảnh",
};

/** One query, in one run: reference image, edit text, target image, the three condition ranks, and
 *  the candidate list the run actually exported for the selected condition.
 *
 *  The honesty rules that matter here:
 *  * a missing condition rank is `unscored` and says WHY it has no rank — it is never shown as a
 *    loss at rank 0 and never mixed into the failure statistics; "every rank is present" is never
 *    claimed when one is null;
 *  * candidates come from a request that names run + subset + condition + query + K. The page
 *    endpoint's own `top_k` used to cap the list at 10, so choosing "Top 50" could not add rows;
 *    now K is part of the request identity and a superseded answer cannot overwrite a newer one;
 *  * the menu never offers more than the export's depth, and no export is a stated reason with a
 *    next step, never placeholder images;
 *  * images that 404 in this deployment render an explicit missing tile with the image id, so a
 *    coverage gap is visible instead of an empty box.
 */
export function QueryInspector({ run, query, condition = "multimodal", selectedCount, selectedRuns = [], onPeerChange, onAnnotate }: {
  run: CompositionRun | null;
  query: CompositionQuery | null;
  condition?: string;
  selectedCount: number;
  /** Selected run keys (`c:<run_id>`) from the URL: the comparison cohort the reader already named. */
  selectedRuns?: readonly string[];
  /** Replace the comparison cohort. The caller updates `runs=` and keeps protocol/subset/query. */
  onPeerChange?: (keys: string[]) => void;
  onAnnotate: () => void;
}) {
  const [k, setK] = useState(10);
  const depth = query?.candidate_depth
    ?? run?.capability_detail?.candidate_topk?.depth
    ?? query?.top_results?.length
    ?? 0;
  const options = candidateDisplayOptions(depth);
  const effectiveK = effectiveCandidateK(k, depth);
  const canRequest = !!(run && query && depth > 0 && (run.capabilities?.candidate_topk || (query.top_results?.length ?? 0) > 0));
  const request = run && query
    ? candidateRequestPath({ runId: run.run_id, subset: run.subset, condition, queryId: query.query_id, k: effectiveK })
    : "none";
  // The FULL identity, protocol included: two protocols can serve the same run id shape, and an
  // answer is only valid for the protocol whose export produced it.
  const requestKey = `${canRequest}:${run?.protocol_id ?? ""}:${request}`;
  const candidatesRegion = useRegion<{ items: CompositionQuery[] }>(
    "inspector-candidates", requestKey,
    useMemo(() => (canRequest
      ? (signal: AbortSignal) => fetchJson<{ items: CompositionQuery[] }>(request, signal)
      : null), [request, canRequest]));
  const current = candidatesRegion.region.key === requestKey;
  const fetched = current && candidatesRegion.region.status === "ready"
    ? candidatesRegion.region.data?.items.find((item) => item.query_id === query?.query_id) ?? null
    : null;
  // Never reuse another query/condition/K answer: a rank is not a candidate list.
  const candidates = fetched?.top_results ?? [];
  const shown = candidates.slice(0, effectiveK);
  // W-03: presence is decided on the window that is RENDERED, never on a rank comparison against a
  // depth the export advertises. `shown` can be shorter than `effectiveK` (a run that exported
  // fewer rows, or one whose export starts late), and in that case a target at rank <= effectiveK
  // is still NOT in the list a reader is looking at.
  const servedDepth = shown.length;
  const targetInShown = shown.some((candidate) => query ? query.positive_ids.includes(candidate.image_id) : false);
  const loading = canRequest && (!current || candidatesRegion.region.status === "idle" || candidatesRegion.region.status === "loading");

  useEffect(() => setK(10), [query?.query_id, run?.run_id, condition]);

  if (!run) {
    return (
      <Card title="Duyệt ảnh truy hồi">
        <EmptyState kind="none-selected" detail="Chọn một mô hình có kết quả để bắt đầu."
                    next={<button type="button" onClick={onAnnotate}>Mở sổ nghiên cứu</button>} />
      </Card>
    );
  }
  if (!query) {
    return (
      <Card title="Duyệt ảnh truy hồi" subtitle={`${run.model_id} / ${run.checkpoint_id}`}>
        <EmptyState kind="none-selected" detail="Chọn một query ở danh sách bên trái, hoặc nới bộ lọc để có query." />
      </Card>
    );
  }

  const conditions = conditionsOf(query);
  const unscored = conditions.some((item) => item.rank === null);
  const category = query.category || "circus";
  const reason = fetched?.candidate_reason ?? query.candidate_reason
    ?? run.capability_detail?.candidate_topk?.reason
    ?? (run.capability_detail?.candidate_topk?.status
      ? `Trạng thái do backend báo: ${run.capability_detail.candidate_topk.status}.`
      : "Lượt này không lưu danh sách ứng viên.");
  const isTarget = (imageId: string) => query.positive_ids.includes(imageId);
  const hasExport = canRequest;

  return (
    <div className="inspector" data-testid="query-inspector">
      <Card
        title={<span data-testid="query-title">Query {query.query_id}</span>}
        subtitle={`${run.model_id} · ${run.checkpoint_id} · ${run.protocol_id} · subset ${run.subset}`}
        actions={(
          <>
            {/* The label, not the id: a bare `unresolved` next to three numbered ranks reads as
                `unscored`, the opposite of what it means. */}
            <Badge tone={unscored ? "warn" : outcomeTone(query.outcome)} testId="query-outcome">
              {unscored ? "Có điều kiện chưa chấm" : outcomeLabel(query.outcome)}
            </Badge>
            {query.category && <Badge>{query.category}</Badge>}
          </>
        )}
      >
        <div className="query-brief">
          {/* DOM order is the reading order the evidence requires: the reference image, then the
              edit caption, THEN the target. A reader who sees the target first reads the caption as
              a post-hoc description of it. */}
          <div className="query-sequence" data-testid="query-sequence">
            <div className="shots">
              <Shot category={category} id={query.reference_id} label="Ảnh tham chiếu" />
            </div>
            <div className="query-instruction-caption">
              <div className="eyebrow">Yêu cầu chỉnh sửa</div>
              <p className="edit-caption" data-testid="query-caption">{query.query_txt}</p>
            </div>
            <div className="shots" data-testid="query-targets">
              {query.positive_ids.map((id, index) => (
                <Shot key={id} category={category} id={id} target
                      label={index === 0 ? "Ảnh đích" : "Ảnh đích bổ sung"} />
              ))}
            </div>
            {(() => {
              // The target image is rendered from the query record, never from the candidate window,
              // so it survives a rank outside the exported depth. Say so where the image is, and
              // name BOTH windows: what the export stored and what is on screen.
              //
              // W-03: the decision is PRESENCE IN THE RENDERED WINDOW, not `best > K`. Those differ
              // exactly where it matters — a run that exported 50 rows while the menu offers Top-100
              // has a target at #43 that is inside the export and outside the served list, and a
              // bare rank comparison called that "outside the 100 stored rows".
              //
              // R10: the window on screen belongs to `condition` (the URL's), so the rank quoted is
              // THAT condition's. The old code took `Math.min` across all three and explained the
              // MM window with the text-only condition's rank — a true number about another
              // condition. The sentence names the condition, and nothing is concluded until the
              // request for this window has answered.
              const label = CONDITION_LABELS[condition] ?? condition;
              const verdict = candidateWindowVerdict({
                conditions, viewed: condition, viewedLabel: label,
                effectiveK, depth, targetInShown,
                status: !canRequest ? "idle" : loading ? "loading" : candidatesRegion.region.status,
              });
              if (verdict.kind === "hidden") return null;
              if (verdict.kind === "pending") {
                return (
                  <p className="hint" data-testid="target-window-pending" aria-live="polite">
                    Đang tải cửa sổ Top-{fmtInt(effectiveK)} của {label}… Chưa kết luận ảnh đích nằm
                    trong hay ngoài cửa sổ này.
                  </p>
                );
              }
              if (verdict.kind === "unavailable") {
                return (
                  <p className="hint" data-testid="target-window-unknown">
                    Chưa mô tả được cửa sổ ứng viên: {verdict.reason}. Ảnh đích vẫn hiện ở đây vì nó
                    lấy từ bản ghi query, không từ danh sách ứng viên.
                  </p>
                );
              }
              return (
                <p className="hint" data-testid="target-outside-window">
                  Ảnh đích ở hạng {fmtRank(verdict.rank)} <b>theo điều kiện {label}</b> —{" "}
                  <b>ngoài cửa sổ Top-{fmtInt(effectiveK)} đang hiện</b>
                  {" "}({fmtInt(servedDepth)} ứng viên được serve, export lưu {fmtInt(depth)} dòng).{" "}
                  {verdict.insideK
                    ? `Hạng ${fmtRank(verdict.rank)} của ${label} nằm trong K nhưng export của lượt này không chứa đúng ảnh đó.`
                    : `Hạng ${fmtRank(verdict.rank)} của ${label} đã vượt K, nên với K=10 hay K=20 ảnh đích không nằm trong danh sách.`}
                  {" "}Đây là hạng toàn gallery, không phải vị trí trong danh sách ứng viên.
                </p>
              );
            })()}
          </div>

          <div className="query-instruction">
            <div>
              <div className="eyebrow" style={{ marginBottom: 8 }}>
                Hạng theo điều kiện · outcome phân loại tại @{FIXED_CUTOFFS.compositionOutcomeK} (cố định)
              </div>
              <div className="rankgrid">
                {conditions.map((item) => (
                  <div key={item.id}
                       className={`rankcell ${item.rank === null ? "unscored-cell" : item.rank <= FIXED_CUTOFFS.compositionOutcomeK ? "hit" : "miss"}`}
                       data-testid={`rank-${item.id}`}
                       data-rank-state={item.rank === null ? "missing-rank" : item.rank <= FIXED_CUTOFFS.compositionOutcomeK ? "hit" : "miss"}>
                    <small>{CONDITION_LABELS[item.id] ?? item.short}</small>
                    <b>{item.rank === null ? "Chưa chấm" : fmtRank(item.rank)}</b>
                    <span className="hint">
                      {item.rank === null
                        ? (query.unscored_kind ? `Không có hạng chính xác (${query.unscored_kind})` : "Không có hạng chính xác")
                        : item.rank <= FIXED_CUTOFFS.compositionOutcomeK
                          ? `trong @${FIXED_CUTOFFS.compositionOutcomeK}`
                          : `ngoài @${FIXED_CUTOFFS.compositionOutcomeK}`}
                      {item.rank != null && depth > 0 && item.rank > depth
                        ? ` · ngoài ${fmtInt(depth)} dòng export đã lưu`
                        : item.rank != null && item.rank > effectiveK
                          ? ` · ngoài Top-${fmtInt(effectiveK)} đang hiện`
                          : ""}
                    </span>
                  </div>
                ))}
              </div>
              <p className="hint" style={{ marginTop: 8 }}>
                Hạng đích trên toàn gallery, theo từng điều kiện. Số ứng viên hiển thị bên dưới không
                đổi ngưỡng phân loại này.
              </p>
            </div>
          </div>
        </div>

        {unscored && (
          <div className="pipeline-notice" data-testid="query-unscored-notice">
            <strong>Query này chưa được chấm đầy đủ.</strong>
            <p>
              {query.failure?.includes("embedding scope")
                ? "Evaluator chưa xác định được nhánh category để tạo embedding, nên chưa có hạng hợp lệ cho điều kiện bị ảnh hưởng. Chưa thể kết luận model truy hồi sai trên query này."
                : "Một hoặc nhiều điều kiện chưa có hạng chính xác. Phần chưa chấm không phải lỗi truy hồi."}
            </p>
            <Detail summary="Nguyên nhân và bước xử lý">
              <p className="diagnostic">{query.failure ?? query.unscored_kind ?? "Evaluator chưa cung cấp nguyên nhân."}</p>
              <p>
                Cần đối soát mapping/adapter với dữ liệu nguồn rồi chạy lại lượt bị ảnh hưởng. Sửa code
                không tự cập nhật artifact cũ.
              </p>
            </Detail>
          </div>
        )}
        {!unscored && query.failure && (
          <Detail summary="Chẩn đoán evaluator"><p className="diagnostic">{query.failure}</p></Detail>
        )}
      </Card>

      <Card
        title={`Ứng viên xếp hạng · ${CONDITION_LABELS[condition] ?? condition}`}
        subtitle="Thứ tự do lượt chạy xuất ra cho đúng điều kiện này; ảnh đích được đánh dấu."
        actions={(
          <label className="candidate-control">Hiển thị
            <select data-testid="topk" aria-label="Số ứng viên hiển thị"
                    value={hasExport ? effectiveK : "unavailable"}
                    disabled={!hasExport}
                    onChange={(event) => setK(Number(event.target.value))}>
              {hasExport
                ? options.map((value) => <option key={value} value={value}>Top {value}</option>)
                : <option value="unavailable">Chưa có ứng viên</option>}
            </select>
          </label>
        )}
      >
        {run.provenance_source && (
          <p className="pipeline-notice" data-testid="candidates-provenance">
            Nguồn ứng viên: <b>{IMPORTED_LABEL[run.provenance_source] ?? run.provenance_source}</b>.
            Đây là artifact nhập vào, không phải đầu ra của model <b>{run.model_id}</b> chạy tại đây.
            Chỉ dùng để đọc hạng và thứ tự.
          </p>
        )}

        {loading ? <LoadingState label={`Đang tải Top ${effectiveK} cho query ${query.query_id}…`} />
          : current && candidatesRegion.region.status === "error"
            ? <ErrorState message={candidatesRegion.region.error ?? "Không tải được ứng viên"} onRetry={candidatesRegion.retry} />
            : shown.length > 0 ? (
              <>
                <p className="candidate-count" aria-live="polite">
                  Đang hiển thị <b>{shown.length}</b> ứng viên · export lưu {fmtInt(depth)} dòng
                  {candidates.length !== shown.length ? `, API serve ${fmtInt(candidates.length)}` : ""} ·
                  {" "}{CONDITION_LABELS[condition] ?? condition}
                  {targetInShown
                    ? " · ảnh đích nằm trong danh sách"
                    : " · ảnh đích KHÔNG nằm trong danh sách này"}
                </p>
                <div className="candidates" data-testid="topk-grid">
                  {shown.map((candidate) => {
                    const target = isTarget(candidate.image_id);
                    return (
                      <article key={`${candidate.rank}:${candidate.image_id}`}
                               className={`candidate ${target ? "is-target" : ""}`}>
                        <div className="candidate-rank">
                          <b>#{candidate.rank}</b>
                          {target && <span>Ảnh đích</span>}
                        </div>
                        <Shot category={category} id={candidate.image_id} compact target={target}
                              label={`Ứng viên ${candidate.rank}`} />
                        <small>
                          <span>Điểm</span>
                          {/* A stored order without scores must read "—", never 0.0. */}
                          <b>{candidate.score == null ? "—" : candidate.score.toFixed(4)}</b>
                        </small>
                        {"gallery_position" in candidate && candidate.gallery_position != null && (
                          <small><span>Vị trí gallery</span><b>{candidate.gallery_position.toLocaleString("en-US")}</b></small>
                        )}
                      </article>
                    );
                  })}
                </div>
                <p className="hint candidate-footnote">
                  Đổi Top-K sẽ tải số ứng viên tương ứng, tối đa độ sâu đã xuất. Ảnh đích nằm ngoài
                  danh sách này vẫn giữ nguyên hạng trên toàn gallery.
                </p>
              </>
            ) : (
              <EmptyState kind="capability" testId="candidates-unavailable"
                          title={unscored ? "Chưa đủ kết quả để xem ứng viên" : "Lượt này chưa có danh sách ứng viên"}
                          detail={unscored
                            ? "Có điều kiện chưa được chấm. Cần sửa nguyên nhân rồi chạy lại để có cả hạng lẫn danh sách ứng viên."
                            : "Hạng đích không chứa thứ tự của các ảnh khác. Cần xuất Top-K từ đúng scoring path của lượt này; nếu cache không xác minh được thì phải chạy lại."}
                          next={reason ? <Detail summary="Chi tiết dữ liệu"><span className="diagnostic">{reason}</span></Detail> : undefined} />
            )}
      </Card>

      <div className="inspector-secondary">
        <Card title="So sánh cùng query" subtitle="Cùng protocol, cùng subset: cohort và mẫu số được nêu">
          <ComparePanel run={run} query={query} selected={selectedRuns}
                        onPeerChange={onPeerChange ?? (() => undefined)} />
        </Card>
        <Card title="Ghi chú nghiên cứu" subtitle="Gắn đúng query + protocol, lưu qua reload">
          <p className="hint">
            Query <code>{query.query_id}</code> · protocol <code>{run.protocol_id}</code> · lượt{" "}
            <code>{run.model_id}</code>.{selectedCount > 1 ? ` Đang chọn ${selectedCount} lượt để phân tích.` : ""}
          </p>
          <button type="button" className="primary" style={{ marginTop: 10 }} onClick={onAnnotate}>
            Ghi chú query này
          </button>
        </Card>
      </div>
    </div>
  );
}

/** One image tile: explicit missing state, and a dialog that Escape closes.
 *
 *  Exported because the LIMN diagnostic gallery renders the SAME tile: two image renderers would be
 *  two answers to "does this image exist in this deployment", and the missing-asset state has to be
 *  identical in both.
 */
export function Shot({ category, id, label, target, compact }: {
  category: string;
  id: string;
  label: string;
  target?: boolean;
  compact?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const clean = id.split("/").pop() || id;
  const src = `${BASE}/api/images/${category}/${encodeURIComponent(clean)}`;
  useEffect(() => {
    setFailed(false);
    dialog.current?.close();
  }, [id, category]);
  return (
    <figure className={`shot ${target ? "target" : ""} ${compact ? "compact" : ""}`}>
      {failed || !id
        ? <div className="shot-missing" data-testid="query-image" data-image-id={clean}
               data-asset-state="missing">Ảnh không có trong deployment<br />{clean}</div>
        : (
          <button type="button" className="shot-open" aria-label={`Phóng to ${label}: ${clean}`}
                  onClick={() => dialog.current?.showModal()}>
            <img src={src} alt={`${label}: ${clean}`} width="240" height="240" loading="lazy"
                 data-testid="query-image" data-image-id={clean} data-asset-state="present"
                 onError={() => setFailed(true)} />
          </button>
        )}
      <figcaption className="shot-label">
        <span>{label}</span>
        <span className="mono" title={clean}>{clean.length > 16 ? `${clean.slice(0, 14)}…` : clean}</span>
      </figcaption>
      <dialog className="image-dialog" ref={dialog} aria-label={`Ảnh phóng to: ${clean}`}>
        <form method="dialog">
          <span>{label} · {clean}</span>
          <button type="submit" autoFocus>Đóng ảnh</button>
        </form>
        <img src={src} width="800" height="800" alt={`${label}: ${clean}`} />
      </dialog>
    </figure>
  );
}

/** Cross-run comparison for one query, restricted to the same protocol and subset.
 *  A different cohort would make the columns incomparable, so it is refused with the reason.
 *  Both reads are keyed regions (protocol+subset for the peer list, query+selection for the ranks),
 *  so a query change cannot leave the previous query's comparison on screen.
 *
 *  Peers come from the SERVED rows of the same scope, ordered by the selection the URL already holds
 *  (`runs=` names both runs of the comparison the reader asked for) and then unsliced — plus an
 *  explicit picker, so any row of the scope is reachable. The old list sliced the first four rows in
 *  API order, which made `tme` and every other run impossible to compare. */
function ComparePanel({ run, query, selected, onPeerChange }: {
  run: CompositionRun;
  query: CompositionQuery;
  /** Selected run keys (`c:<run_id>`) straight from the URL, so a peer choice survives navigation. */
  selected: readonly string[];
  onPeerChange: (runIds: string[]) => void;
}) {
  const peerKey = `${run.protocol_id}\u001f${run.subset}`;
  const peersRegion = useRegion<CompositionRun[]>(
    "compare-peers", peerKey,
    useCallback((signal: AbortSignal) => fetchJson<CompositionRun[]>(`/api/composition/runs?subset=${run.subset}`, signal),
      [run.subset]));

  const selectedRunIds = useMemo(
    () => selected.filter((key) => key.startsWith("c:")).map((key) => key.slice(2)),
    [selected.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps
  const candidates = useMemo(
    () => comparePeers(peersRegion.region.data ?? [], {
      currentRunId: run.run_id, protocol: run.protocol_id, subset: run.subset, selected: selectedRunIds,
    }),
    [peersRegion.region.data, run.run_id, run.protocol_id, run.subset, selectedRunIds]);
  // The comparison shows the OTHER runs the URL selects; when it names none besides this one, the
  // first few scoped rows are shown so the panel is never empty — and the picker reaches all of them.
  const picked = selectedRunIds.filter((id) => id !== run.run_id);
  const peers = useMemo(
    () => (picked.length ? candidates.filter((row) => picked.includes(row.run_id)) : candidates.slice(0, 4)),
    [candidates, picked.join(",")]); // eslint-disable-line react-hooks/exhaustive-deps

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
  if (!candidates.length) {
    return <EmptyState kind="capability" detail="Chỉ có một lượt trong protocol và subset này, nên không có gì để so sánh." />;
  }
  const picker = (
    <div className="toolbar" data-testid="compare-peer-picker">
      <Field label="Thêm lượt so sánh" htmlFor="compare-peer"
             hint="Cùng protocol, cùng subset; mọi lượt của phạm vi này đều chọn được">
        <select id="compare-peer" data-testid="compare-peer-select" value=""
                onChange={(event) => {
                  const peer = candidates.find((row) => row.run_id === event.target.value);
                  if (!peer) return;
                  // Only the comparison cohort changes: protocol, subset, inspected run and query stay.
                  const next = [...new Set([...picked, peer.run_id])].map((id) => `c:${id}`);
                  onPeerChange([`c:${run.run_id}`, ...next]);
                  event.target.value = "";
                }}>
          <option value="">— chọn lượt —</option>
          {candidates.map((peer) => (
            <option key={peer.run_id} value={peer.run_id} disabled={picked.includes(peer.run_id)}>
              {compositionLabel(peer)}{picked.includes(peer.run_id) ? " · đang so sánh" : ""}
            </option>
          ))}
        </select>
      </Field>
      {picked.length > 0 && (
        <button type="button" className="ghost" data-testid="compare-peer-clear"
                onClick={() => onPeerChange([`c:${run.run_id}`])}>Bỏ lượt đã thêm</button>
      )}
    </div>
  );
  if (!peers.length) {
    return (
      <div className="stack">
        {picker}
        <EmptyState kind="capability" detail="Chưa chọn lượt nào để so sánh — chọn một lượt ở ô trên." />
      </div>
    );
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
      {picker}
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
        Cohort: {rows.length} lượt, cùng protocol <code>{run.protocol_id}</code> và subset{" "}
        <code>{run.subset}</code>; hạng thiếu được ghi riêng là chưa chấm.
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
