import { useMemo, useState } from "react";
import { fetchJson, useRegion } from "./useRegion";
import { Badge, Card, Detail, EmptyState, ErrorState, LoadingState } from "./ui";
import { fmtInt, fmtRank } from "./format";
import { CONDITIONS, SUBSET_LABEL, type Subset } from "./routes";

/** Diagnostic comparison for the two CIRCUS queries whose garment category the release cannot pin.
 *
 *  `7:3038` and `7:3790` have a reference image plus a normalized caption pair that appear in BOTH
 *  the `shirt` and the `toptee` FashionIQ validation captions. The canonical evaluator refuses to
 *  guess, so both stay unscored and the canonical coverage stays 6001/6003. This panel shows what
 *  each candidate scope WOULD retrieve, side by side, because "we did not score it" is a weaker
 *  answer than "here is what both checkpoints do".
 *
 *  The rules this component exists to keep:
 *  * the canonical unscored state is rendered with the comparison, in the card, at all times --
 *    neither scope's numbers can be read as a canonical rank;
 *  * the two scopes are columns of one table, not a single "best" number: no scope is chosen, no
 *    difference is averaged, and no denominator is reduced;
 *  * every cell is a link to the candidate window it came from, so a rank can be checked instead of
 *    trusted;
 *  * an absent or inconsistent diagnostic renders a named status, never an empty comparison.
 */
interface ScopeReading {
  target_rank: number;
  target_score: number;
  score_at_rank_1: number | null;
  target_margin_vs_rank1: number | null;
  gallery_size_scored: number | null;
  model_text: string;
  export: string | null;
}

interface DiagnosticPayload {
  available: boolean;
  status: string;
  label?: string;
  reason?: string;
  canonical_status: {
    status: string;
    scored_count: number;
    unscored_count: number;
    unscored_qids: string[];
    note: string;
  } | null;
  /** WHICH population the canonical counts count. Present because this panel is rendered on every
   *  subset page, and the row above it is scoped: `sf` shows ~4,000/4,069 and `validated` ~581/586.
   *  A bare "6,001/6,003" beside those reads as this page's number, which it is not. */
  canonical_cohort?: {
    subset: string | null;
    query_count: number | null;
    query_count_per_subset: Record<string, number> | null;
    scored_count: number | null;
    unscored_count: number | null;
    multimodal_recall_at_10: number | null;
    source: string | null;
  } | null;
  conditions?: string[];
  scopes?: string[];
  gallery?: { count: number; protocol_id: string; note: string } | null;
  checkpoints?: Record<string, { checkpoint_id: string; sha256: string; filename: string }>;
  rank_convention_crosscheck?: { checked: number; all_agree: boolean; method?: string };
  queries?: Record<string, {
    qid: string;
    release: { query_txt: string; reference_id: string; positive_ids: string[]; loader_category: string | null };
    target_gallery_id: string;
    target_position_in_gallery: number[];
    scopes: Record<string, Record<string, ScopeReading>>;
  }>;
  evidence_files?: { per_scope?: string[] };
}

interface DiagnosticCandidates {
  qid: string;
  scope: string;
  condition: string;
  target_rank: number;
  target_score: number;
  checkpoint_sha256: string | null;
  gallery_size: number | null;
  depth_stored: number | null;
  candidates: Array<{ rank: number; gallery_position: number; gallery_id: string; score: number; is_target: boolean }>;
  canonical_status_note: string;
}

const CONDITION_SHORT: Record<string, string> = {
  multimodal: "MM",
  text_only_black_image: "Text-only",
  image_only_empty_text: "Image-only",
};

export function LimnScopeDiagnostic({ pageSubset }: { pageSubset?: string }) {
  const region = useRegion<DiagnosticPayload>(
    "limn-dual-scope", "limn-dual-scope",
    useMemo(() => (signal: AbortSignal) =>
      fetchJson<DiagnosticPayload>("/api/composition/diagnostics/limn-dual-scope", signal), []));

  if (region.region.status === "loading" || region.region.status === "idle") {
    return <LoadingState label="Đang tải so sánh hai scope…" />;
  }
  if (region.region.status === "error") {
    return <ErrorState message={region.region.error ?? "Không tải được so sánh hai scope"} onRetry={region.retry} />;
  }
  const payload = region.region.data;
  if (!payload || !payload.available) {
    return (
      <EmptyState kind="capability" testId="limn-diagnostic-unavailable"
                  title="Chưa có so sánh hai scope"
                  detail={payload?.canonical_status
                    ? `Chẩn đoán chưa được xuất bản (${payload.status}). Trạng thái canonical không đổi.`
                    : `Chẩn đoán chưa được xuất bản (${payload?.status ?? "không rõ"}).`} />
    );
  }

  const canonical = payload.canonical_status!;
  const cohort = payload.canonical_cohort ?? null;
  const cohortSubset = cohort?.subset ?? null;
  const cohortLabel = cohortSubset
    ? (SUBSET_LABEL[cohortSubset as Subset] ?? cohortSubset)
    : "không ghi cohort";
  const cohortMismatch = !!cohortSubset && !!pageSubset && cohortSubset !== pageSubset;
  // Only the published per-subset DENOMINATOR is quoted here. The scored count of a subset is NOT
  // derivable by subtracting the full-cohort unscored count: measured, the `sf` row on this page
  // reads 4,000/4,069 (69 unscored) while the validated row reads 581/586 (5), so subtracting 2
  // would print 4,067/4,069 -- a number the panel never measured and no artifact contains.
  const ownTotal = cohortMismatch ? cohort?.query_count_per_subset?.[pageSubset] ?? null : null;
  const scopes = payload.scopes ?? [];
  const conditions = payload.conditions ?? CONDITIONS.map((item) => item.id);
  const queries = Object.values(payload.queries ?? {});

  return (
    <Card
      title={<span data-testid="limn-diagnostic-title">Ambiguous category — diagnostic comparison</span>}
      subtitle={payload.reason}
      testId="limn-diagnostic"
      actions={(
        <Badge tone="warn" testId="limn-diagnostic-canonical">
          Canonical ({cohortLabel} cohort): {fmtInt(canonical.scored_count)}/
          {fmtInt(canonical.scored_count + canonical.unscored_count)} scored
        </Badge>
      )}
    >
      <div className="pipeline-notice" data-testid="limn-diagnostic-canonical-notice">
        <strong>Đây là chẩn đoán, không phải metric canonical.</strong>
        <p>
          Hai query {canonical.unscored_qids.map((qid) => <code key={qid}>{qid}</code>).reduce<React.ReactNode[]>(
            (acc, node, index) => (index === 0 ? [node] : [...acc, ", ", node]), [])}{" "}
          vẫn <b>chưa được chấm</b> trong lượt canonical: coverage giữ nguyên{" "}
          <b>{fmtInt(canonical.scored_count)}/{fmtInt(canonical.scored_count + canonical.unscored_count)}</b>{" "}
          của cohort <b>{cohortLabel}</b>. Bảng dưới chấm cả hai checkpoint như nhau và không chọn
          scope nào; không có số nào ở đây được cộng vào nDCG, Recall hay coverage canonical.
        </p>
        {cohortMismatch && (
          <p data-testid="limn-diagnostic-cohort-mismatch">
            <b>Số canonical ở trên là của cohort {cohortLabel}, không phải của trang này.</b>{" "}
            Trang đang xem là subset <code>{pageSubset}</code>
            {ownTotal != null ? ` (${fmtInt(ownTotal)} query)` : ""}, còn số canonical ở trên là của
            toàn bộ {fmtInt(cohort?.query_count ?? 0)} query. Hai mẫu số khác nhau là đúng thiết kế —
            không được đọc số {fmtInt(canonical.scored_count)}/
            {fmtInt(canonical.scored_count + canonical.unscored_count)} như của subset này. Số đã chấm
            của riêng subset này do bảng phía trên tự báo; panel này không suy ra nó.
          </p>
        )}
        <p className="hint">{canonical.note}</p>
      </div>

      <div className="table-wrap" style={{ marginTop: 16 }}>
        <table data-testid="limn-diagnostic-table">
          <caption className="sr-only">
            Hạng đích của hai query mơ hồ dưới từng scope và từng điều kiện; đây là chẩn đoán, không phải
            metric canonical
          </caption>
          <thead>
            <tr>
              <th scope="col">Query</th>
              <th scope="col">Điều kiện</th>
              {scopes.map((scope) => (
                <th scope="col" className="num" key={scope}>
                  scope <code>{scope}</code>
                  <div className="cell-sub mono" title={payload.checkpoints?.[scope]?.sha256}>
                    {payload.checkpoints?.[scope]?.checkpoint_id ?? "checkpoint không rõ"}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {queries.map((query) => (
              conditions.map((condition, index) => (
                <tr key={`${query.qid}:${condition}`} data-testid={`limn-diagnostic-row-${query.qid}-${condition}`}>
                  {index === 0 && (
                    <th scope="rowgroup" rowSpan={conditions.length}>
                      <code>{query.qid}</code>
                      <div className="cell-sub">loader category: {query.release.loader_category ?? "không xác định"}</div>
                    </th>
                  )}
                  <td>{CONDITION_SHORT[condition] ?? condition}</td>
                  {scopes.map((scope) => {
                    const reading = query.scopes[scope]?.[condition];
                    if (!reading) {
                      return <td key={scope} className="num"><Badge tone="warn">thiếu</Badge></td>;
                    }
                    return (
                      <td key={scope} className="num">
                        <b>{fmtRank(reading.target_rank)}</b>
                        <div className="cell-sub mono">score {reading.target_score.toFixed(4)}</div>
                        <div className="cell-sub">
                          {reading.gallery_size_scored != null
                            ? `trên ${fmtInt(reading.gallery_size_scored)} ảnh`
                            : "gallery size không ghi"}
                        </div>
                        {reading.target_margin_vs_rank1 != null && (
                          <div className="cell-sub">
                            cách #1 {reading.target_margin_vs_rank1.toFixed(4)}
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))
            ))}
          </tbody>
        </table>
      </div>

      <Detail summary="Vì sao hai query này mơ hồ, và bằng chứng đi kèm">
        <p>
          Mỗi query có <b>cùng một cặp caption validation</b> nộp dưới cả <code>shirt</code> và{" "}
          <code>toptee</code>, nên annotation không nói checkpoint nào là đúng. Vì vậy evaluator từ chối
          đoán thay vì chọn một scope.
        </p>
        <ul>
          {queries.map((query) => (
            <li key={query.qid}>
              <code>{query.qid}</code>: ảnh tham chiếu <code>{query.release.reference_id}</code>, đích{" "}
              <code>{query.target_gallery_id}</code> ở vị trí gallery{" "}
              {fmtInt(query.target_position_in_gallery[0])}; text “{query.release.query_txt}”.
            </li>
          ))}
        </ul>
        <p className="hint">
          Gallery dùng cho MỌI scope: {payload.gallery ? `${fmtInt(payload.gallery.count)} ảnh, protocol ${payload.gallery.protocol_id}` : "không ghi"} —
          không lọc theo category, nên một checkpoint chuyên biệt vẫn phải truy hồi trên toàn bộ gallery.
        </p>
        {payload.rank_convention_crosscheck && (
          <p className="hint">
            Quy ước hạng đã đối chứng: {fmtInt(payload.rank_convention_crosscheck.checked)} hạng canonical
            được tái lập từ chính feature cache của lượt, khớp{" "}
            <b>{payload.rank_convention_crosscheck.all_agree ? "toàn bộ" : "KHÔNG toàn bộ"}</b>.
          </p>
        )}
        {cohort && (
          <p className="hint" data-testid="limn-diagnostic-cohort-source">
            Cohort canonical: <code>{cohort.subset ?? "không ghi"}</code>, {fmtInt(cohort.query_count)} query
            {cohort.query_count_per_subset
              ? ` (full ${fmtInt(cohort.query_count_per_subset.full)}, sf ${fmtInt(cohort.query_count_per_subset.sf)}, validated ${fmtInt(cohort.query_count_per_subset.validated)})`
              : ""}
            {" "}— đọc từ <code>{cohort.source ?? "không ghi nguồn"}</code>, không suy ra từ trang này.
          </p>
        )}
        <p className="hint">
          Bằng chứng: <code>limn_dual_scope.json</code> + 6 file per-scope trong{" "}
          <code>workbench/docs/reviews/20261109-native15-composition-limn/</code>.
        </p>
      </Detail>

      <CandidateProbe queries={queries.map((query) => query.qid)} scopes={scopes} conditions={conditions} />
    </Card>
  );
}

/** One query × one scope × one condition, exported Top-100, so a rank can be inspected not trusted. */
function CandidateProbe({ queries, scopes, conditions }: {
  queries: string[];
  scopes: string[];
  conditions: string[];
}) {
  const [qid, setQid] = useState(queries[0] ?? "");
  const [scope, setScope] = useState(scopes[0] ?? "");
  const [condition, setCondition] = useState(conditions[0] ?? "multimodal");
  const key = `${qid}\u001f${scope}\u001f${condition}`;
  const region = useRegion<DiagnosticCandidates>(
    "limn-diagnostic-candidates", key,
    useMemo(() => ((signal: AbortSignal) =>
      fetchJson<DiagnosticCandidates>(
        `/api/composition/diagnostics/limn-dual-scope/candidates?qid=${encodeURIComponent(qid)}`
        + `&scope=${encodeURIComponent(scope)}&condition=${encodeURIComponent(condition)}&depth=20`,
        signal)), [qid, scope, condition]));
  const current = region.region.key === key;

  return (
    <div className="stack" style={{ marginTop: 16 }} data-testid="limn-diagnostic-probe">
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
        <label className="field">Query
          <select value={qid} onChange={(event) => setQid(event.target.value)} data-testid="limn-probe-qid">
            {queries.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label className="field">Scope
          <select value={scope} onChange={(event) => setScope(event.target.value)} data-testid="limn-probe-scope">
            {scopes.map((value) => <option key={value} value={value}>{value}</option>)}
          </select>
        </label>
        <label className="field">Điều kiện
          <select value={condition} onChange={(event) => setCondition(event.target.value)} data-testid="limn-probe-condition">
            {conditions.map((value) => <option key={value} value={value}>{CONDITION_SHORT[value] ?? value}</option>)}
          </select>
        </label>
      </div>
      {!current || region.region.status === "loading"
        ? <LoadingState label="Đang tải Top-20 đã xuất…" />
        : region.region.status === "error"
          ? <ErrorState message={region.region.error ?? "Không tải được ứng viên chẩn đoán"} onRetry={region.retry} />
          : region.region.data && (
            <>
              <p className="hint" data-testid="limn-probe-note">
                Hạng đích <b>{fmtRank(region.region.data.target_rank)}</b> trên{" "}
                {fmtInt(region.region.data.gallery_size)} ảnh · checkpoint{" "}
                <code>{region.region.data.checkpoint_sha256?.slice(0, 16) ?? "không ghi"}…</code> ·{" "}
                {region.region.data.canonical_status_note}
              </p>
              <div className="table-wrap">
                <table>
                  <caption className="sr-only">20 ứng viên đầu tiên của một scope và một điều kiện</caption>
                  <thead>
                    <tr>
                      <th scope="col" className="num">#</th>
                      <th scope="col">Ảnh</th>
                      <th scope="col" className="num">Score</th>
                      <th scope="col" className="num">Vị trí gallery</th>
                    </tr>
                  </thead>
                  <tbody>
                    {region.region.data.candidates.map((candidate) => (
                      <tr key={`${candidate.rank}:${candidate.gallery_id}`} aria-selected={candidate.is_target}>
                        <td className="num">{candidate.rank}</td>
                        <td>
                          <code>{candidate.gallery_id}</code>
                          {candidate.is_target && <> <Badge tone="ok">ảnh đích</Badge></>}
                        </td>
                        <td className="num mono">{candidate.score.toFixed(4)}</td>
                        <td className="num mono">{fmtInt(candidate.gallery_position)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="hint">
                Ảnh không được render ở đây: đây là chẩn đoán, và một tile ảnh cạnh metric canonical dễ bị
                đọc như một hàng canonical. Đường dẫn ảnh nằm trong file JSONL của scope tương ứng.
              </p>
            </>
          )}
    </div>
  );
}
