import { useMemo, useState } from "react";
import { fetchJson, useRegion } from "./useRegion";
import { Badge, Card, Detail, EmptyState, ErrorState, LoadingState } from "./ui";
import { fmtInt, fmtRank } from "./format";
import { CONDITIONS, SUBSET_LABEL, type Subset } from "./routes";
import { Shot } from "./QueryInspector";

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
  /** CLVC adds these; LIMN omits them, so every read is optional. */
  condition_kind?: "protocol" | "ablation";
  condition_label?: string;
  raw_edit_caption?: string;
  encoded_text_is_empty?: boolean;
  target_within_export_depth?: boolean;
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
  /** Present only on the CLVC payload (LIMN's bytes are frozen). */
  model?: string;
  model_name?: string;
  diagnostic?: boolean;
  canonical?: boolean;
  condition_kinds?: Record<string, "protocol" | "ablation">;
  condition_labels?: Record<string, string>;
  canonical_cohort_provenance?: {
    source: "artifact" | "run_summary";
    run_id?: string | null;
    path?: string | null;
    sha256?: string | null;
  } | null;
  reference_match_note?: string;
  queries?: Record<string, {
    qid: string;
    reference_asset?: string;
    release: { query_txt: string; reference_id: string; positive_ids: string[]; loader_category: string | null };
    target_gallery_id: string;
    target_position_in_gallery: number[];
    scopes: Record<string, Record<string, ScopeReading>>;
  }>;
  evidence_files?: { summary?: string; per_scope?: string[] };
  /** Whether the diagnostic evidence is internally consistent. False means the comparison below
   *  contradicts itself somewhere, so it is rendered with the reason rather than silently. */
  complete?: boolean;
  defects?: Array<{ code: string; qid: string | null; scope: string | null; condition: string | null; detail: string }>;
}

interface DiagnosticCandidate {
  rank: number;
  gallery_position: number;
  gallery_id: string;
  image_path: string;
  image_stem: string;
  /** Resolved by the backend against the real gallery; `null` means this deployment cannot serve
   *  that image, which is a fact the tile states instead of rendering an empty box. */
  image_url: string | null;
  image_category: string | null;
  score: number;
  is_target: boolean;
  /** CLVC only: this tile IS the query's own reference image, i.e. a self-match. */
  is_reference?: boolean;
  match_kind?: "reference_self_match" | "candidate";
}

interface DiagnosticCandidates {
  qid: string;
  scope: string;
  condition: string;
  /** "diagnostic — <scope> scope, <condition> (not canonical)": rendered as the block title so no
   *  tile can be read as a canonical per-query row. */
  label: string;
  identity: { qid: string; scope: string; condition: string; standard: string; manifest_digest: string | null };
  complete: boolean;
  defects: Array<{ code: string; qid: string | null; scope: string | null; condition: string | null; detail: string }>;
  target: {
    rank: number;
    score: number;
    gallery_id: string;
    gallery_position: number;
    in_window: boolean;
    /** Whether the target was inside the FULL stored export, independent of the depth served. */
    in_stored_window?: boolean;
    window_served?: number;
    window_stored?: number;
    delta_to_rank_1: number | null;
    image_url: string | null;
    image_category: string | null;
    image_stem: string;
  };
  target_did: string | null;
  target_asset: string | null;
  target_rank: number;
  target_score: number;
  checkpoint_sha256: string | null;
  gallery_size: number | null;
  model_text: string;
  depth_requested: number;
  depth_served: number;
  depth_stored: number;
  depth_truncated: boolean;
  candidates: DiagnosticCandidate[];
  staged_query_image_is_evaluator_input: boolean;
  canonical_status_note: string;
  /* CLVC adds these; LIMN omits them. */
  model?: string;
  model_name?: string;
  diagnostic?: boolean;
  canonical?: boolean;
  condition_kind?: "protocol" | "ablation";
  condition_label?: string;
  raw_edit_caption?: string;
  query_image_path?: string;
  reference_image_path?: string;
  reference_asset?: string;
  score_at_rank_1?: number | null;
  target_within_export_depth?: boolean;
  effectively_encoded?: { text: string; text_is_empty: boolean; note: string | null };
  rank_1_is_reference_self_match?: boolean;
  target_outside_stored_window?: boolean;
}

const CONDITION_SHORT: Record<string, string> = {
  multimodal: "MM",
  text_only_black_image: "Text-only",
  image_only_empty_text: "Image-only",
  clvc_image_encoder_only: "Image only (encoder)",
};

/** The diagnostic endpoints this panel can switch between, without a reload.
 *
 *  Both models answer the same payload shape from the same reader, so the panel is one component with
 *  a model axis rather than two copies that drift. `slug` is the path segment the backend declares
 *  literally for each model. */
const DIAGNOSTIC_MODELS = [
  { slug: "limn", label: "LIMN" },
  { slug: "clvc", label: "CLVC-Net" },
] as const;
type DiagnosticModelSlug = typeof DIAGNOSTIC_MODELS[number]["slug"];

/** Which published dual-scope diagnostic the panel is showing. Switching is a fetch, not a reload. */
function ModelPicker({ value, onChange }: {
  value: DiagnosticModelSlug;
  onChange: (next: DiagnosticModelSlug) => void;
}) {
  return (
    <label className="field">Model
      <select value={value} aria-label="Model chẩn đoán"
              onChange={(event) => onChange(event.target.value as DiagnosticModelSlug)}
              data-testid="dual-diagnostic-model">
        {DIAGNOSTIC_MODELS.map((item) => (
          <option key={item.slug} value={item.slug}>{item.label}</option>
        ))}
      </select>
    </label>
  );
}

export function LimnScopeDiagnostic({ pageSubset }: { pageSubset?: string }) {
  // The model is a control, not a build-time choice: both diagnostics are published, both answer the
  // same shape, and a reader comparing them must not reload the page to do it.
  const [modelSlug, setModelSlug] = useState<DiagnosticModelSlug>("limn");
  const region = useRegion<DiagnosticPayload>(
    "dual-scope", `dual-scope:${modelSlug}`,
    useMemo(() => (signal: AbortSignal) =>
      fetchJson<DiagnosticPayload>(`/api/composition/diagnostics/${modelSlug}-dual-scope`, signal),
      [modelSlug]));
  // A payload for another model may never paint under this selector's label.
  const payloadIsCurrent = region.region.key === `dual-scope:${modelSlug}`;

  if (region.region.status === "loading" || region.region.status === "idle") {
    return <LoadingState label="Đang tải so sánh hai scope…" />;
  }
  if (region.region.status === "error") {
    return <ErrorState message={region.region.error ?? "Không tải được so sánh hai scope"} onRetry={region.retry} />;
  }
  const payload = payloadIsCurrent ? region.region.data : null;
  if (!payload || !payload.available) {
    return (
      <div className="stack" data-testid="dual-diagnostic-model-bar">
        <ModelPicker value={modelSlug} onChange={setModelSlug} />
        <EmptyState kind="capability" testId="limn-diagnostic-unavailable"
                  title="Chưa có so sánh hai scope"
                  detail={payload?.canonical_status
                    ? `Chẩn đoán chưa được xuất bản (${payload.status}). Trạng thái canonical không đổi.`
                    : `Chẩn đoán chưa được xuất bản (${payload?.status ?? "không rõ"}).`} />
      </div>
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
      title={<span data-testid="limn-diagnostic-title">
        {modelSlug === "clvc" ? "CLVC-Net — " : ""}Ambiguous category — diagnostic comparison
      </span>}
      subtitle={payload.reason}
      testId="limn-diagnostic"
      actions={(
        <>
          <ModelPicker value={modelSlug} onChange={setModelSlug} />
          {payload.complete === false && (
            <Badge tone="bad" testId="limn-diagnostic-incomplete-badge"
                   title={(payload.defects ?? []).map((defect) => defect.code)
                     .filter((code, index, all) => all.indexOf(code) === index).join(", ")}>
              Chưa hoàn tất — {fmtInt((payload.defects ?? []).length)} defect
            </Badge>
          )}
          <Badge tone="warn" testId="limn-diagnostic-canonical">
            Canonical ({cohortLabel} cohort): {fmtInt(canonical.scored_count)}/
            {fmtInt(canonical.scored_count + canonical.unscored_count)} scored
          </Badge>
        </>
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
        {payload.complete === false && (
          <div className="pipeline-notice" data-testid="limn-diagnostic-defects">
            <strong>
              Bằng chứng chẩn đoán chưa hoàn tất — {fmtInt((payload.defects ?? []).length)} defect.
            </strong>
            <p>
              Mã: <code>{(payload.defects ?? []).map((defect) => defect.code)
                .filter((code, index, all) => all.indexOf(code) === index).join(", ")}</code>.
              Các hạng và điểm bên dưới vẫn là số đo thật từ file bằng chứng, nhưng chúng{" "}
              <b>chưa được coi là đầy đủ</b>: mỗi ô bị ảnh hưởng được nêu chi tiết ở phần chọn query
              bên dưới. Trạng thái canonical không đổi.
            </p>
          </div>
        )}
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
                  <td>
                    {payload.condition_labels?.[condition] ?? CONDITION_SHORT[condition] ?? condition}
                    {payload.condition_kinds?.[condition] === "ablation" && (
                      <div className="cell-sub">
                        <Badge tone="warn">ablation</Badge>
                      </div>
                    )}
                  </td>
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
        {payload.canonical_cohort_provenance && (
          <p className="hint" data-testid="dual-diagnostic-cohort-provenance">
            Cohort canonical đọc từ <b>run summary</b>{" "}
            {payload.canonical_cohort_provenance.run_id
              ? <code>{payload.canonical_cohort_provenance.run_id}</code> : null}
            {" "}(sha256 <code>{payload.canonical_cohort_provenance.sha256?.slice(0, 16) ?? "không ghi"}…</code>),
            vì artifact chẩn đoán không tự ghi cohort. Số đã chấm vẫn là của lượt canonical, không phải
            của panel này.
          </p>
        )}
        {payload.evidence_files?.summary && (
          <p className="hint" data-testid="dual-diagnostic-evidence">
            Bằng chứng: <code>{payload.evidence_files.summary.split("/").pop()}</code> +{" "}
            {fmtInt(payload.evidence_files.per_scope?.length ?? 0)} file per-scope trong{" "}
            <code>{payload.evidence_files.summary.replace(/\/[^/]+$/, "/")}</code>.
          </p>
        )}
        {!payload.evidence_files?.summary && (
          <p className="hint">
            Bằng chứng: <code>limn_dual_scope.json</code> + 6 file per-scope trong{" "}
            <code>workbench/docs/reviews/20261109-native15-composition-limn/</code>.
          </p>
        )}
      </Detail>

      <CandidateProbe modelSlug={modelSlug}
                      referenceAssets={Object.fromEntries(
                        queries.map((query) => [query.qid, query.reference_asset ?? null]))}
                      queries={queries.map((query) => query.qid)}
                      scopes={scopes} conditions={conditions}
                      conditionLabels={payload.condition_labels ?? {}} />
    </Card>
  );
}

/** One query × one scope × one condition, exported Top-100, so a rank can be inspected not trusted. */
function CandidateProbe({ modelSlug, referenceAssets, queries, scopes, conditions, conditionLabels }: {
  modelSlug: DiagnosticModelSlug;
  referenceAssets: Record<string, string | null>;
  queries: string[];
  scopes: string[];
  conditions: string[];
  conditionLabels: Record<string, string>;
}) {
  const [qid, setQid] = useState(queries[0] ?? "");
  const [scope, setScope] = useState(scopes[0] ?? "");
  const [condition, setCondition] = useState(conditions[0] ?? "multimodal");
  // W-03: K is a real control here, not a constant. Serving the whole stored export and calling the
  // rank-43 target "in window" is TRUE for 100 rows and would be misread as "in Top-10": the finding
  // is that it is outside K=10 and K=20. Default 10 = the protocol's own outcome cutoff.
  const [depth, setDepth] = useState(10);
  const key = `${qid}\u001f${scope}\u001f${condition}\u001f${depth}`;
  // `depth` is part of BOTH the request and its identity: two K values are two different windows,
  // and a late answer for K=100 must not paint under a K=10 label. The backend validates depth
  // rather than clamping it, and reports `depth_served` / `depth_stored` separately.
  const region = useRegion<DiagnosticCandidates>(
    "dual-diagnostic-candidates", `${modelSlug}\u001f${key}`,
    useMemo(() => ((signal: AbortSignal) =>
      fetchJson<DiagnosticCandidates>(
        `/api/composition/diagnostics/${modelSlug}-dual-scope/candidates?qid=${encodeURIComponent(qid)}`
        + `&scope=${encodeURIComponent(scope)}&condition=${encodeURIComponent(condition)}`
        + `&depth=${depth}`,
        signal)), [modelSlug, qid, scope, condition, depth]));
  // A response for another qid/scope/condition may never paint under these labels.
  const current = region.region.key === `${modelSlug}\u001f${key}`;
  const data = current ? region.region.data : null;

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
            {conditions.map((value) => (
              <option key={value} value={value}>
                {conditionLabels[value] ?? CONDITION_SHORT[value] ?? value}
              </option>
            ))}
          </select>
        </label>
        <label className="field">Top-K hiển thị
          <select value={depth} onChange={(event) => setDepth(Number(event.target.value))}
                  data-testid="limn-probe-depth" aria-label="Số ứng viên hiển thị">
            {[10, 20, 50, 100].map((value) => <option key={value} value={value}>Top {value}</option>)}
          </select>
        </label>
      </div>
      {!current || region.region.status === "loading"
        ? <LoadingState label="Đang tải cửa sổ ứng viên đã xuất…" />
        : region.region.status === "error"
          ? <ErrorState message={region.region.error ?? "Không tải được ứng viên chẩn đoán"} onRetry={region.retry} />
          : data && (
            <>
              <p className="hint" data-testid="limn-probe-note">
                {data.label} · hạng đích <b>{fmtRank(data.target_rank)}</b> trên{" "}
                {data.gallery_size == null ? "gallery không ghi" : `${fmtInt(data.gallery_size)} ảnh`} · cửa sổ đã
                xuất <b>{fmtInt(data.depth_served)}</b>/{fmtInt(data.depth_stored)} (yêu cầu {fmtInt(data.depth_requested)}) ·
                đích <b>{data.target.in_window ? "nằm trong" : "ngoài"}</b> cửa sổ này
                {data.target.in_stored_window !== data.target.in_window
                  ? ` (trong toàn bộ ${fmtInt(data.depth_stored)} dòng đã lưu thì ${data.target.in_stored_window ? "có" : "không"})`
                  : ""}
                {data.depth_truncated ? " · bị cắt theo độ sâu đã lưu" : ""} · checkpoint{" "}
                <code>{data.checkpoint_sha256?.slice(0, 16) ?? "không ghi"}…</code>
              </p>
              <p className="hint" data-testid="limn-probe-canonical-note">{data.canonical_status_note}</p>
              {data.rank_1_is_reference_self_match && (
                <div className="pipeline-notice" data-testid="dual-probe-self-match">
                  <strong>Hạng 1 ở điều kiện này là chính ảnh tham chiếu của query.</strong>
                  <p>
                    Đây là <b>self-match</b> (score ≈ {data.score_at_rank_1?.toFixed(6) ?? "1.0"}), không phải
                    một kết quả truy hồi: gallery có chứa ảnh tham chiếu nên hàng của nó tự khớp với chính
                    nó. Ảnh đích thật của query nằm ở hạng <b>{fmtRank(data.target_rank)}</b>.
                  </p>
                </div>
              )}

              {!data.complete && (
                <div className="pipeline-notice" data-testid="limn-probe-incomplete">
                  <strong>Bằng chứng chẩn đoán chưa hoàn tất.</strong>
                  {data.defects.length
                    ? <>
                        <span> {fmtInt(data.defects.length)} defect liên quan tới ô này:</span>
                        <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
                          {data.defects.map((defect, index) => (
                            <li key={`${defect.code}-${index}`}>
                              <code>{defect.code}</code>
                              {defect.detail ? <> — {defect.detail}</> : null}
                            </li>
                          ))}
                        </ul>
                      </>
                    : <span> Không defect nào áp riêng cho ô này; file bằng chứng đánh dấu
                        <code> complete=false</code>, nên cửa sổ dưới đây là số đo thật nhưng chưa được
                        coi là đầy đủ.</span>}
                </div>
              )}

              {/* The query itself: reference image, the released caption, and what the model was
                  ACTUALLY given. The ablation transports no text, so `model_text` is empty and saying
                  only "caption: ..." would misreport the input. */}
              <div className="stack" data-testid="dual-probe-query">
                <div className="eyebrow">Query — ảnh tham chiếu, caption gốc, và text thực sự đưa vào model</div>
                <div className="shots">
                  {data.reference_image_path && (
                    <Shot category={data.target.image_category ?? data.scope}
                          id={data.reference_image_path}
                          label={`ảnh tham chiếu ${referenceAssets[data.qid] ?? ""}`.trim()} />
                  )}
                </div>
                <p className="hint" data-testid="dual-probe-caption">
                  Caption gốc: “{data.raw_edit_caption ?? "không ghi"}”
                </p>
                <p className="hint" data-testid="dual-probe-effective-text">
                  Text thực sự mã hoá:{" "}
                  {data.effectively_encoded
                    ? (data.effectively_encoded.text_is_empty
                        ? <b>rỗng</b>
                        : <code>{data.effectively_encoded.text}</code>)
                    : <code>{data.model_text}</code>}
                  {data.effectively_encoded?.text_is_empty && <span> (điều kiện này không truyền text)</span>}
                  {data.condition_kind === "ablation" && <span> · <Badge tone="warn">ablation</Badge></span>}
                </p>
                {data.effectively_encoded?.text_is_empty && (
                  <p className="hint" data-testid="dual-probe-empty-text-note">
                    {data.effectively_encoded.note}
                  </p>
                )}
              </div>

              {/* The target ALWAYS renders, separately from the window, because it frequently is not
                  in it: at rank 6891 no Top-100 window contains it, and a gallery that can only show
                  the target when it happens to be a candidate cannot show the finding. */}
              <div className="stack" data-testid="limn-probe-target">
                <div className="eyebrow">Ảnh đích của query — luôn hiện, tách khỏi cửa sổ</div>
                <div className="shots">
                  <Shot category={data.target.image_category ?? data.scope}
                        id={data.target.gallery_id}
                        label={`ảnh đích ${data.target_did ?? ""} (hạng ${fmtRank(data.target.rank)})`.trim()}
                        target />
                </div>
                {data.target.in_window
                  ? <p className="hint" data-testid="limn-probe-target-in-window">
                      Ảnh đích nằm trong cửa sổ {fmtInt(data.depth_served)} ứng viên đang hiện (hạng {fmtRank(data.target.rank)}).
                    </p>
                  : <p className="hint" data-testid="limn-probe-target-outside">
                      Ảnh đích ở hạng {fmtRank(data.target.rank)} — <b>ngoài cửa sổ Top-{fmtInt(data.depth_served)}
                      {" "}đang hiện</b> (export lưu {fmtInt(data.depth_stored)} dòng). Với K=10 hay K=20 thì
                      ảnh đích <b>không</b> nằm trong cửa sổ; hạng {fmtRank(data.target.rank)} là hạng toàn gallery,
                      không phải vị trí trong danh sách ứng viên.
                    </p>}
              </div>

              {data.candidates.length
                ? (
                  <>
                    <p className="candidate-count">
                      {fmtInt(data.candidates.length)} ứng viên theo đúng thứ tự export · {data.label}
                    </p>
                    <div className="candidates" data-testid="limn-probe-grid">
                      {data.candidates.map((candidate) => (
                        <article key={`${candidate.rank}:${candidate.gallery_id}`}
                                 className={`candidate ${candidate.is_target ? "is-target" : ""}`
                                   + `${candidate.is_reference ? " is-reference" : ""}`}
                                 data-match-kind={candidate.match_kind ?? "candidate"}>
                          <div className="candidate-rank">
                            <b>#{candidate.rank}</b>
                            {candidate.is_target && <span>Ảnh đích</span>}
                            {candidate.is_reference && (
                              <span data-testid={`dual-probe-ref-match-${candidate.rank}`}>
                                Ảnh tham chiếu (self-match)
                              </span>
                            )}
                          </div>
                          <Shot category={candidate.image_category ?? data.scope}
                                id={candidate.gallery_id} compact target={candidate.is_target}
                                label={`Ứng viên ${candidate.rank}`} />
                          <small><span>Điểm</span><b>{candidate.score.toFixed(4)}</b></small>
                          <small><span>Vị trí gallery</span><b>{fmtInt(candidate.gallery_position)}</b></small>
                          <small className="mono"><span>stem</span><b>{candidate.image_stem}</b></small>
                        </article>
                      ))}
                    </div>
                  </>
                )
                : <EmptyState kind="capability" testId="limn-probe-no-window"
                              detail="Export của scope/điều kiện này không lưu ứng viên nào cho query này. Ảnh đích ở trên vẫn là hạng thật trên toàn gallery." />}
            </>
          )}
    </div>
  );
}
