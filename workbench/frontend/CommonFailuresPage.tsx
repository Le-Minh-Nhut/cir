import { useEffect, useMemo, useState } from "react";
import { api } from "./client";
import { Badge, Card, EmptyState, ErrorState, LoadingState } from "./ui";
import type { BenchmarkRegime, InspectorSeed } from "./benchmarksTypes";

interface ProtocolOption {
  id: string;
  group: string;
  label: string;
  dataset: "edir" | "pinpoint" | "fashioniq";
  regime: string;
  ready_models: number;
}

interface ModelChip {
  key: string;
  display: string;
  family: string;
  protocol_id: string;
}

interface PerModelRank {
  key: string;
  display: string;
  rank: number | null;
  failed: boolean;
  ref_copy: boolean;
  score: number;
}

interface CommonFailureRow {
  query_id: string;
  text: string;
  category: string;
  l1_interest?: string | null;
  ref_image_id: string;
  ref_image_url: string;
  gt_image_id: string;
  gt_image_url: string;
  scored_models: number;
  fail_count: number;
  hit_count: number;
  fail_rate: number;
  ref_copy_count: number;
  best_rank: number | null;
  median_rank: number | null;
  mean_score: number;
  per_model: PerModelRank[];
}

interface CommonFailuresResponse {
  protocol: string;
  protocol_label: string;
  dataset: "edir" | "pinpoint" | "fashioniq";
  regime: string;
  k: number;
  preset: string;
  available_protocols: ProtocolOption[];
  all_models: ModelChip[];
  selected_models: string[];
  selected_model_count: number;
  slices: Array<{ id: string; label: string; count: number }>;
  categories: Array<{ id: string; count: number }>;
  category_breakdown: Array<{
    category: string;
    total: number;
    all_fail: number;
    fail80: number;
    fail80_rate: number;
  }>;
  preset_counts: Record<string, number>;
  filtered_total: number;
  offset: number;
  limit: number;
  items: CommonFailureRow[];
}

interface RankedCandidate {
  rank: number;
  image_id: string;
  image_url: string;
  score: number | null;
  is_gt: boolean;
  is_ref: boolean;
  is_hard_neg: boolean;
}

interface ModelRankingBlock {
  key: string;
  display: string;
  family: string;
  best_gt_rank: number | null;
  top1_is_ref: boolean;
  candidates: RankedCandidate[];
}

interface CommonFailureDetailResponse {
  protocol: string;
  dataset: "edir" | "pinpoint" | "fashioniq";
  query: {
    query_id: string;
    text: string;
    category: string;
    l1_interest?: string | null;
    ref_image_id: string;
    ref_image_url: string;
  };
  ground_truths: Array<{ image_id: string; image_url: string }>;
  hard_negatives: Array<{ image_id: string; image_url: string; is_ref?: boolean }>;
  models: ModelRankingBlock[];
}

const DATASET_TABS: Array<{
  id: "fashioniq" | "pinpoint" | "edir";
  label: string;
  subtitle: string;
  defaultProtocol: string;
}> = [
  {
    id: "fashioniq",
    label: "FashionIQ",
    subtitle: "3 protocol chính chủ · 16 runs",
    defaultProtocol: "fashioniq_original_split",
  },
  {
    id: "pinpoint",
    label: "PinPoint",
    subtitle: "1,337 queries · 8 lát cắt Fashion",
    defaultProtocol: "pinpoint_indomain",
  },
  {
    id: "edir",
    label: "EDIR",
    subtitle: "1,003 queries · 15 phép chỉnh sửa",
    defaultProtocol: "edir_indomain",
  },
];

const PRESET_TABS: Array<{ id: string; label: string; desc: string }> = [
  { id: "fail80", label: "Fail chung ≥80%", desc: "Mặc định: ít nhất 80% mô hình cùng trượt" },
  { id: "all_fail", label: "100% cùng trượt", desc: "Toàn bộ mô hình đều trượt" },
  { id: "fail50", label: "≥50% cùng trượt", desc: "Ít nhất một nửa mô hình trượt" },
  { id: "disagree", label: "Bất đồng mạnh", desc: "Đa số trượt nhưng 1–2 mô hình vẫn trúng" },
  { id: "ref_copy50", label: "Ref Copy ≥50%", desc: "≥50% mô hình copy lại ảnh gốc ở Top-1" },
  { id: "all", label: "Tất cả", desc: "Toàn bộ query" },
];

function fmtRank(r: number | null): string {
  return r === null ? ">50" : `#${r}`;
}

function datasetForProtocol(proto: string): "fashioniq" | "pinpoint" | "edir" {
  if (proto.startsWith("pinpoint")) return "pinpoint";
  if (proto.startsWith("edir")) return "edir";
  return "fashioniq";
}

function buildDetailHref(protocol: string, queryId: string, k: number, modelsParam: string): string {
  const params = new URLSearchParams();
  params.set("area", "common_failures");
  params.set("cf_protocol", protocol);
  params.set("cf_query", queryId);
  params.set("cf_k", String(k));
  if (modelsParam) params.set("cf_models", modelsParam);
  return `${window.location.pathname}?${params.toString()}`;
}

function readDetailFromUrl(): { protocol: string; queryId: string; k: number; models: string } | null {
  const params = new URLSearchParams(window.location.search);
  const qid = (params.get("cf_query") ?? "").trim();
  if (!qid) return null;
  const proto = (params.get("cf_protocol") ?? "fashioniq_original_split").trim();
  const kRaw = Number(params.get("cf_k") ?? "10");
  const k = Number.isInteger(kRaw) && kRaw >= 1 && kRaw <= 50 ? kRaw : 10;
  const models = (params.get("cf_models") ?? "").trim();
  return { protocol: proto, queryId: qid, k, models };
}

/* ========================================================================= */
/* STANDALONE QUERY DETAIL VIEW (in-app or opened in a dedicated browser tab) */
/* ========================================================================= */

function QueryDetailPage({
  protocol,
  queryId,
  k,
  modelsParam,
  onBack,
  onInspectBenchmarkQuery,
}: {
  protocol: string;
  queryId: string;
  k: number;
  modelsParam: string;
  onBack: () => void;
  onInspectBenchmarkQuery: (seed: InspectorSeed) => void;
}) {
  const [detail, setDetail] = useState<CommonFailureDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [err, setErr] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"single" | "stacked">("single");
  const [activeModel, setActiveModel] = useState<string>("");
  const [compareModel, setCompareModel] = useState<string>("");
  const [topK, setTopK] = useState<number>(10);
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setLightbox(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox]);

  useEffect(() => {
    const ctrl = new AbortController();
    const params = new URLSearchParams({
      protocol,
      query_id: queryId,
      top_k: String(topK),
    });
    if (modelsParam) params.set("models", modelsParam);
    setLoading(true);
    setErr(null);
    api<CommonFailureDetailResponse>(`/api/benchmarks/common-failures/detail?${params.toString()}`, {
      signal: ctrl.signal,
    })
      .then((res) => {
        setDetail(res);
        setLoading(false);
        setActiveModel((prev) =>
          prev && res.models.some((m) => m.key === prev) ? prev : (res.models[0]?.key ?? ""),
        );
      })
      .catch((e: unknown) => {
        if ((e as { name?: string })?.name === "AbortError") return;
        setErr(e instanceof Error ? e.message : String(e));
        setLoading(false);
      });
    return () => ctrl.abort();
  }, [protocol, queryId, modelsParam, topK]);

  const standaloneHref = buildDetailHref(protocol, queryId, k, modelsParam);

  if (loading && !detail) return <LoadingState label="Đang tải chi tiết query & lưới ảnh Top-K của các mô hình…" />;
  if (err) {
    return (
      <div className="stack">
        <div>
          <button type="button" className="secondary" onClick={onBack}>← Quay lại danh sách lỗi chung</button>
        </div>
        <ErrorState message={err} />
      </div>
    );
  }
  if (!detail) return null;

  const primaryBlock = detail.models.find((m) => m.key === activeModel) ?? detail.models[0] ?? null;
  const compareBlock = compareModel ? detail.models.find((m) => m.key === compareModel) ?? null : null;
  const failCount = detail.models.filter((m) => m.best_gt_rank === null || m.best_gt_rank > k).length;
  const refCopyCount = detail.models.filter((m) => m.top1_is_ref).length;

  const renderCandidate = (c: RankedCandidate, compact = false) => {
    const cls = c.is_gt
      ? "bm-cand bm-cand-gt"
      : c.is_hard_neg
        ? "bm-cand bm-cand-hneg"
        : c.is_ref
          ? "bm-cand bm-cand-ref"
          : "bm-cand";
    return (
      <button
        key={`${c.rank}:${c.image_id}`}
        type="button"
        className={compact ? `${cls} bm-cand-compact` : cls}
        onClick={() => setLightbox({ src: c.image_url, caption: `Rank #${c.rank} · ${c.image_id}` })}
        title={`Rank #${c.rank} · ${c.image_id}${c.score !== null ? ` (score ${c.score.toFixed(3)})` : ""}`}
      >
        <div className="bm-cand-img-wrap">
          <img src={c.image_url} alt={c.image_id} />
          <span className="bm-cand-rank">#{c.rank}</span>
          {c.is_gt && <span className="bm-cand-flag bm-flag-gt">GT</span>}
          {!c.is_gt && c.is_hard_neg && <span className="bm-cand-flag bm-flag-hneg">HNeg</span>}
          {!c.is_gt && !c.is_hard_neg && c.is_ref && <span className="bm-cand-flag bm-flag-ref">Ref</span>}
        </div>
        <span className="bm-cand-id mono">
          {c.score !== null ? `${c.score.toFixed(3)} · ${c.image_id}` : c.image_id}
        </span>
      </button>
    );
  };

  return (
    <div className="stack" data-testid="cf-detail-page">
      {/* Clean Top Navigation Header */}
      <div className="bm-detail-topbar">
        <button type="button" className="secondary" style={{ minHeight: 36, padding: "5px 12px" }} onClick={onBack}>
          ← Quay lại danh sách
        </button>
        <div className="row" style={{ gap: 6 }}>
          <Badge tone="info">{detail.dataset.toUpperCase()}</Badge>
          <span className="mono cell-strong">{detail.query.query_id}</span>
          <Badge tone="neutral">{detail.query.category}</Badge>
          {detail.query.l1_interest && <Badge tone="neutral">{detail.query.l1_interest}</Badge>}
          <Badge tone={failCount === detail.models.length ? "bad" : "warn"}>
            {failCount}/{detail.models.length} trượt @{k}
          </Badge>
          {refCopyCount > 0 && <Badge tone="warn">Ref Copy #1: {refCopyCount}</Badge>}
        </div>
        <div className="row" style={{ marginLeft: "auto" }}>
          {detail.dataset !== "fashioniq" && (
            <button
              type="button"
              className="secondary"
              style={{ minHeight: 36, padding: "5px 12px" }}
              onClick={() =>
                onInspectBenchmarkQuery({
                  dataset: detail.dataset === "edir" ? "edir" : "pinpoint",
                  regime: (protocol.replace(/^(edir|pinpoint)_/, "") || "indomain") as BenchmarkRegime,
                  queryId: detail.query.query_id,
                })
              }
            >
              Mở Benchmark Inspector
            </button>
          )}
          <a
            className="bm-btn-link"
            href={standaloneHref}
            target="_blank"
            rel="noopener noreferrer"
            data-testid="cf-open-new-tab"
          >
            Mở tab mới ↗
          </a>
        </div>
      </div>

      {/* Hero Brief Card: Reference -> Ground Truth + Caption + Per-Model Rank Summary */}
      <Card>
        <div className="bm-detail-hero">
          <div className="bm-detail-pair">
            <button
              type="button"
              className="bm-detail-shot bm-shot-ref"
              onClick={() => setLightbox({ src: detail.query.ref_image_url, caption: `Ảnh gốc · ${detail.query.ref_image_id}` })}
            >
              <span className="bm-shot-tag bm-shot-tag-ref">Ảnh gốc (Reference)</span>
              <img src={detail.query.ref_image_url} alt="Reference" />
              <span className="mono bm-shot-id">{detail.query.ref_image_id}</span>
            </button>

            <div className="bm-detail-arrow" aria-hidden="true">→</div>

            {detail.ground_truths.slice(0, 3).map((gt, idx) => (
              <button
                key={gt.image_id}
                type="button"
                className="bm-detail-shot bm-shot-gt"
                onClick={() => setLightbox({ src: gt.image_url, caption: `Đáp án đúng #${idx + 1} · ${gt.image_id}` })}
              >
                <span className="bm-shot-tag bm-shot-tag-gt">
                  Đáp án đúng {detail.ground_truths.length > 1 ? `#${idx + 1}/${detail.ground_truths.length}` : "(GT)"}
                </span>
                <img src={gt.image_url} alt={gt.image_id} />
                <span className="mono bm-shot-id">{gt.image_id}</span>
              </button>
            ))}
          </div>

          <div className="bm-detail-right">
            <div className="bm-caption-box">
              <span className="field-label">Câu lệnh chỉnh sửa (Modification Text)</span>
              <p className="bm-caption-text">“{detail.query.text}”</p>
            </div>

            <div>
              <span className="field-label" style={{ display: "block", marginBottom: 8 }}>
                Thứ hạng Ground Truth theo từng mô hình (bấm để xem Top-K ảnh bên dưới)
              </span>
              <div className="bm-consensus-grid">
                {detail.models.map((m) => {
                  const isAct = viewMode === "single" && m.key === primaryBlock?.key;
                  const failed = m.best_gt_rank === null || m.best_gt_rank > k;
                  return (
                    <button
                      key={m.key}
                      type="button"
                      className={isAct ? "bm-cons-chip active" : "bm-cons-chip"}
                      onClick={() => {
                        setActiveModel(m.key);
                        setViewMode("single");
                      }}
                    >
                      <b>{m.display}</b>
                      <Badge tone={failed ? "bad" : "ok"}>{fmtRank(m.best_gt_rank)}</Badge>
                      {m.top1_is_ref && <span className="bm-tag bm-tag-warn">Ref#1</span>}
                    </button>
                  );
                })}
              </div>
            </div>

            {detail.hard_negatives.length > 0 && (
              <div>
                <span className="field-label" style={{ display: "block", marginBottom: 6 }}>
                  Bẫy khó (Hard Negatives · {detail.hard_negatives.length})
                </span>
                <div className="bm-mini-strip">
                  {detail.hard_negatives.slice(0, 6).map((hn) => (
                    <button
                      key={hn.image_id}
                      type="button"
                      className="bm-mini-card bm-cand-hneg"
                      style={{ width: 82 }}
                      onClick={() => setLightbox({ src: hn.image_url, caption: `Hard Negative · ${hn.image_id}` })}
                    >
                      <img src={hn.image_url} alt={hn.image_id} />
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Top-K Ranked Images Card */}
      <Card
        title={
          viewMode === "single" && primaryBlock
            ? `Ảnh xếp hạng Top-${primaryBlock.candidates.length} · ${primaryBlock.display}`
            : `So sánh xếp tầng Top-5 của toàn bộ ${detail.models.length} mô hình`
        }
        subtitle="Bấm ảnh bất kỳ để phóng to · Khung xanh = Đáp án đúng (GT) · Khung vàng = Copy ảnh gốc (Ref) · Khung đỏ = Hard Negative"
        actions={
          <div className="row">
            <div className="bm-seg" role="group" aria-label="Chế độ xem Top-K">
              <button
                type="button"
                className={viewMode === "single" ? "bm-seg-btn active" : "bm-seg-btn"}
                onClick={() => setViewMode("single")}
              >
                Chi tiết Top-{topK}
              </button>
              <button
                type="button"
                className={viewMode === "stacked" ? "bm-seg-btn active" : "bm-seg-btn"}
                onClick={() => setViewMode("stacked")}
              >
                Xếp tầng Top-5 ({detail.models.length})
              </button>
            </div>

            {viewMode === "single" && (
              <>
                <select
                  aria-label="So sánh thêm mô hình thứ 2"
                  value={compareModel}
                  onChange={(ev) => setCompareModel(ev.target.value)}
                  style={{ minHeight: 36, padding: "4px 10px", fontSize: ".84rem" }}
                >
                  <option value="">So sánh thêm: Không</option>
                  {detail.models
                    .filter((m) => m.key !== primaryBlock?.key)
                    .map((m) => (
                      <option key={m.key} value={m.key}>
                        vs {m.display} ({fmtRank(m.best_gt_rank)})
                      </option>
                    ))}
                </select>
                {detail.dataset !== "fashioniq" && (
                  <select
                    aria-label="Số lượng Top-K"
                    value={topK}
                    onChange={(ev) => setTopK(Number(ev.target.value))}
                    style={{ minHeight: 36, padding: "4px 10px", fontSize: ".84rem" }}
                  >
                    <option value={10}>Top-10</option>
                    <option value={15}>Top-15</option>
                    <option value={20}>Top-20</option>
                    <option value={50}>Top-50</option>
                  </select>
                )}
              </>
            )}
          </div>
        }
      >
        {viewMode === "single" && primaryBlock && (
          <div className={compareBlock ? "bm-compare-grids" : "stack"}>
            <div>
              {compareBlock && (
                <div className="row" style={{ marginBottom: 10 }}>
                  <b>{primaryBlock.display}</b>
                  <Badge tone={primaryBlock.best_gt_rank !== null && primaryBlock.best_gt_rank <= k ? "ok" : "bad"}>
                    GT {fmtRank(primaryBlock.best_gt_rank)}
                  </Badge>
                </div>
              )}
              <div className="bm-cand-grid">
                {primaryBlock.candidates.map((c) => renderCandidate(c, false))}
              </div>
            </div>

            {compareBlock && (
              <div>
                <div className="row" style={{ marginBottom: 10 }}>
                  <b>{compareBlock.display}</b>
                  <Badge tone={compareBlock.best_gt_rank !== null && compareBlock.best_gt_rank <= k ? "ok" : "bad"}>
                    GT {fmtRank(compareBlock.best_gt_rank)}
                  </Badge>
                </div>
                <div className="bm-cand-grid">
                  {compareBlock.candidates.map((c) => renderCandidate(c, false))}
                </div>
              </div>
            )}
          </div>
        )}

        {viewMode === "stacked" && (
          <div className="bm-stacked-models">
            {detail.models.map((m) => {
              const failed = m.best_gt_rank === null || m.best_gt_rank > k;
              return (
                <div key={m.key} className="bm-stacked-row">
                  <div className="bm-stacked-meta">
                    <b>{m.display}</b>
                    <span className="bm-sub">{m.family}</span>
                    <div className="row" style={{ marginTop: 6 }}>
                      <Badge tone={failed ? "bad" : "ok"}>{fmtRank(m.best_gt_rank)}</Badge>
                      {m.top1_is_ref && <Badge tone="warn">Ref#1</Badge>}
                    </div>
                  </div>
                  <div className="bm-stacked-cands">
                    {m.candidates.slice(0, 5).map((c) => renderCandidate(c, true))}
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </Card>

      {lightbox && (
        <div className="bm-lightbox" role="dialog" aria-modal="true" aria-label={lightbox.caption} onClick={() => setLightbox(null)}>
          <div className="bm-lightbox-inner" onClick={(ev) => ev.stopPropagation()}>
            <img src={lightbox.src} alt={lightbox.caption} />
            <div className="bm-lightbox-bar">
              <span className="mono">{lightbox.caption}</span>
              <button type="button" className="secondary" onClick={() => setLightbox(null)}>Đóng (Esc)</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ========================================================================= */
/* MAIN COMMON FAILURES PAGE (List + Quick GT Preview + Dedicated Detail Tab) */
/* ========================================================================= */

export function CommonFailuresPage({
  onInspectBenchmarkQuery,
}: {
  onInspectBenchmarkQuery: (seed: InspectorSeed) => void;
}) {
  const initialDetail = useMemo(() => readDetailFromUrl(), []);
  const [detailRoute, setDetailRoute] = useState<{ protocol: string; queryId: string; k: number; models: string } | null>(
    initialDetail,
  );

  const [activeDataset, setActiveDataset] = useState<"fashioniq" | "pinpoint" | "edir">(
    initialDetail ? datasetForProtocol(initialDetail.protocol) : "fashioniq",
  );
  const [protocol, setProtocol] = useState<string>(initialDetail?.protocol ?? "fashioniq_original_split");
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);
  const [preset, setPreset] = useState<string>("fail80");
  const [k, setK] = useState<number>(initialDetail?.k ?? 10);
  const [slice, setSlice] = useState<string>("all");
  const [category, setCategory] = useState<string>("");
  const [sort, setSort] = useState<string>("fail_desc");
  const [searchInput, setSearchInput] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [offset, setOffset] = useState<number>(0);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);

  const [data, setData] = useState<CommonFailuresResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // Sync browser Back/Forward with detail view
  useEffect(() => {
    const onPop = () => setDetailRoute(readDetailFromUrl());
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  useEffect(() => {
    const t = window.setTimeout(() => {
      setSearchQuery(searchInput.trim());
      setOffset(0);
    }, 220);
    return () => window.clearTimeout(t);
  }, [searchInput]);

  useEffect(() => {
    if (!lightbox) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setLightbox(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox]);

  const modelsParam = selectedModels.join(",");

  useEffect(() => {
    if (detailRoute) return;
    const ctrl = new AbortController();
    const params = new URLSearchParams({
      protocol,
      preset,
      k: String(k),
      sort,
      offset: String(offset),
      limit: "25",
    });
    if (modelsParam) params.set("models", modelsParam);
    if (slice && slice !== "all") params.set("slice", slice);
    if (category) params.set("category", category);
    if (searchQuery) params.set("q", searchQuery);

    setLoading(true);
    setError(null);
    api<CommonFailuresResponse>(`/api/benchmarks/common-failures?${params.toString()}`, {
      signal: ctrl.signal,
    })
      .then((res) => {
        setData(res);
        setLoading(false);
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
    return () => ctrl.abort();
  }, [detailRoute, protocol, modelsParam, preset, k, slice, category, sort, searchQuery, offset]);

  const openDetailInApp = (qid: string) => {
    const next = { protocol, queryId: qid, k, models: modelsParam };
    const url = buildDetailHref(protocol, qid, k, modelsParam);
    window.history.pushState({ at: Date.now() }, "", url);
    setDetailRoute(next);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const closeDetailToList = () => {
    const url = `${window.location.pathname}?area=common_failures`;
    window.history.pushState({ at: Date.now() }, "", url);
    setDetailRoute(null);
  };

  if (detailRoute) {
    return (
      <QueryDetailPage
        protocol={detailRoute.protocol}
        queryId={detailRoute.queryId}
        k={detailRoute.k}
        modelsParam={detailRoute.models}
        onBack={closeDetailToList}
        onInspectBenchmarkQuery={onInspectBenchmarkQuery}
      />
    );
  }

  const switchDataset = (nextDs: "fashioniq" | "pinpoint" | "edir") => {
    const tab = DATASET_TABS.find((t) => t.id === nextDs)!;
    setActiveDataset(nextDs);
    switchProtocol(tab.defaultProtocol);
  };

  const switchProtocol = (nextProto: string) => {
    setProtocol(nextProto);
    setSelectedModels([]);
    setSlice("all");
    setCategory("");
    setOffset(0);
    setExpandedId(null);
    if (nextProto.startsWith("fashioniq") && preset === "ref_copy50") setPreset("fail80");
    if (nextProto.startsWith("fashioniq") && sort === "ref_copy_desc") setSort("fail_desc");
  };

  const toggleModel = (key: string) => {
    if (!data) return;
    const active = selectedModels.length ? selectedModels : data.all_models.map((m) => m.key);
    const next = active.includes(key) ? active.filter((x) => x !== key) : [...active, key];
    if (next.length === 0) return;
    setSelectedModels(next.length === data.all_models.length ? [] : next);
    setOffset(0);
  };

  const applyFamilyPreset = (family: string) => {
    if (!data) return;
    if (family === "all") setSelectedModels([]);
    else if (family === "top5") setSelectedModels(data.all_models.slice(0, 5).map((m) => m.key));
    else {
      const matched = data.all_models.filter((m) => m.family === family).map((m) => m.key);
      if (matched.length) setSelectedModels(matched);
    }
    setOffset(0);
  };

  const datasetProtocols = (data?.available_protocols ?? []).filter((p) => p.dataset === activeDataset);
  const activeModelSet = new Set(
    selectedModels.length ? selectedModels : (data?.all_models ?? []).map((m) => m.key),
  );
  const activeExtraFilterCount =
    (selectedModels.length > 0 ? 1 : 0) + (category ? 1 : 0) + (slice !== "all" ? 1 : 0) + (sort !== "fail_desc" ? 1 : 0);

  const exportUrl = (() => {
    const params = new URLSearchParams({ protocol, preset, k: String(k), sort });
    if (modelsParam) params.set("models", modelsParam);
    if (slice && slice !== "all") params.set("slice", slice);
    if (category) params.set("category", category);
    if (searchQuery) params.set("q", searchQuery);
    return `/api/benchmarks/common-failures/export.csv?${params.toString()}`;
  })();

  return (
    <div className="stack">
      {/* 1. Dataset Tabs (FashionIQ default · PinPoint · EDIR) */}
      <div className="bm-ds-tabs" role="tablist" aria-label="Chọn bộ dữ liệu">
        {DATASET_TABS.map((tab) => {
          const isAct = activeDataset === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              role="tab"
              aria-selected={isAct}
              data-testid={`cf-ds-${tab.id}`}
              className={isAct ? "bm-ds-tab active" : "bm-ds-tab"}
              onClick={() => switchDataset(tab.id)}
            >
              <span className="bm-ds-tab-title">{tab.label}</span>
              <span className="bm-ds-tab-sub">{tab.subtitle}</span>
            </button>
          );
        })}
      </div>

      {/* 2. Clean Compact Control Panel */}
      <Card>
        <div className="bm-cf-ctrl-top">
          {/* Protocol pills */}
          <div className="bm-slice-grid" role="group" aria-label="Chọn protocol">
            {datasetProtocols.map((opt) => {
              const shortLabel = opt.label.replace(/^(FashionIQ|PinPoint|EDIR)\s*·\s*/i, "");
              return (
                <button
                  key={opt.id}
                  type="button"
                  disabled={opt.ready_models === 0}
                  data-testid={`cf-proto-${opt.id}`}
                  className={protocol === opt.id ? "bm-slice-pill active" : "bm-slice-pill"}
                  onClick={() => switchProtocol(opt.id)}
                >
                  <span>{shortLabel}</span>
                  <span className="bm-slice-cnt">{opt.ready_models}</span>
                </button>
              );
            })}
          </div>

          <div className="bm-cf-ctrl-right">
            {loading && <Badge tone="info">Đang cập nhật…</Badge>}
            <select
              id="cf-cutoff"
              aria-label="Ngưỡng trượt Top-K"
              value={k}
              onChange={(ev) => {
                setK(Number(ev.target.value));
                setOffset(0);
              }}
              style={{ minHeight: 36, padding: "4px 10px", fontSize: ".85rem" }}
            >
              <option value={1}>Trượt @1</option>
              <option value={10}>Trượt @10</option>
              <option value={50}>Trượt @50</option>
            </select>
            <input
              id="cf-search"
              type="search"
              placeholder="Tìm caption / ID…"
              value={searchInput}
              onChange={(ev) => setSearchInput(ev.target.value)}
              style={{ minHeight: 36, padding: "4px 10px", width: 210, fontSize: ".85rem" }}
            />
            <button
              type="button"
              className={showAdvanced || activeExtraFilterCount > 0 ? "primary" : "secondary"}
              style={{ minHeight: 36, padding: "4px 12px", fontSize: ".84rem", whiteSpace: "nowrap" }}
              onClick={() => setShowAdvanced((v) => !v)}
            >
              Lọc thêm{activeExtraFilterCount > 0 ? ` (${activeExtraFilterCount})` : ""} {showAdvanced ? "▴" : "▾"}
            </button>
            <a className="bm-btn-link" href={exportUrl} download>
              CSV
            </a>
          </div>
        </div>

        {/* Preset pills */}
        <div className="bm-outcome-pills" role="group" aria-label="Mức độ fail chung">
          {PRESET_TABS.filter((p) => p.id !== "ref_copy50" || data?.dataset !== "fashioniq").map((p) => {
            const cnt = data?.preset_counts?.[p.id];
            return (
              <button
                key={p.id}
                type="button"
                className={preset === p.id ? "bm-pill active" : "bm-pill"}
                title={p.desc}
                onClick={() => {
                  setPreset(p.id);
                  setOffset(0);
                }}
              >
                <span>{p.label}</span>
                {cnt !== undefined && <span className="bm-pill-count">{cnt.toLocaleString()}</span>}
              </button>
            );
          })}
          {data && (
            <span className="hint" style={{ marginLeft: "auto", alignSelf: "center" }}>
              Gộp {data.selected_model_count}/{data.all_models.length} model
            </span>
          )}
        </div>

        {protocol === "fashioniq_all_protocols" && (
          <div className="bm-insight-box bm-insight-warn" style={{ marginTop: 10, marginBottom: 0, padding: "8px 12px" }}>
            <b>Lưu ý:</b> Đang gộp 16 model từ 3 protocol FashionIQ khác nhau (khác kích thước gallery). Chọn riêng từng protocol ở trên nếu cần so sánh công bằng tuyệt đối.
          </div>
        )}

        {/* Collapsible Optional Filters (Category / Slice / Sort / Model Picker) */}
        {showAdvanced && data && (
          <div className="bm-model-picker-body">
            <div className="toolbar" style={{ marginBottom: 10 }}>
              {data.slices.length > 0 && (
                <div className="field">
                  <label className="field-label" htmlFor="cf-slice">Lát cắt PinPoint</label>
                  <select id="cf-slice" value={slice} onChange={(ev) => { setSlice(ev.target.value); setOffset(0); }} style={{ minHeight: 36 }}>
                    {data.slices.map((s) => (
                      <option key={s.id} value={s.id}>{s.label} ({s.count})</option>
                    ))}
                  </select>
                </div>
              )}
              {data.categories.length > 0 && (
                <div className="field">
                  <label className="field-label" htmlFor="cf-cat">
                    {data.dataset === "edir" ? "Phép chỉnh sửa" : "Nhóm / Category"}
                  </label>
                  <select id="cf-cat" value={category} onChange={(ev) => { setCategory(ev.target.value); setOffset(0); }} style={{ minHeight: 36 }}>
                    <option value="">Tất cả ({data.categories.length})</option>
                    {data.categories.map((c) => (
                      <option key={c.id} value={c.id}>{c.id} ({c.count})</option>
                    ))}
                  </select>
                </div>
              )}
              <div className="field">
                <label className="field-label" htmlFor="cf-sort">Sắp xếp</label>
                <select id="cf-sort" value={sort} onChange={(ev) => { setSort(ev.target.value); setOffset(0); }} style={{ minHeight: 36 }}>
                  <option value="fail_desc">Nhiều model trượt nhất</option>
                  {data.dataset !== "fashioniq" && <option value="ref_copy_desc">Nhiều Ref Copy nhất</option>}
                  <option value="best">Ít model trượt nhất</option>
                  <option value="id">Theo Query ID</option>
                </select>
              </div>
              {activeExtraFilterCount > 0 && (
                <button
                  type="button"
                  className="ghost"
                  style={{ minHeight: 36 }}
                  onClick={() => {
                    setSelectedModels([]);
                    setCategory("");
                    setSlice("all");
                    setSort("fail_desc");
                    setOffset(0);
                  }}
                >
                  Đặt lại bộ lọc
                </button>
              )}
            </div>

            <div className="row" style={{ marginBottom: 8 }}>
              <span className="field-label">Chọn nhanh model:</span>
              <button type="button" className="bm-pill" onClick={() => applyFamilyPreset("all")}>Tất cả ({data.all_models.length})</button>
              <button type="button" className="bm-pill" onClick={() => applyFamilyPreset("top5")}>Top-5</button>
              {Array.from(new Set(data.all_models.map((m) => m.family))).map((fam) => (
                <button key={fam} type="button" className="bm-pill" onClick={() => applyFamilyPreset(fam)}>Nhóm {fam}</button>
              ))}
            </div>
            <div className="bm-consensus-grid">
              {data.all_models.map((m) => {
                const checked = activeModelSet.has(m.key);
                return (
                  <button key={m.key} type="button" className={checked ? "bm-cons-chip active" : "bm-cons-chip"} onClick={() => toggleModel(m.key)}>
                    <span>{checked ? "✓" : "○"}</span>
                    <b>{m.display}</b>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </Card>

      {error && <ErrorState message={error} />}
      {loading && !data && <LoadingState label="Đang tổng hợp các lỗi chung giữa các mô hình…" />}

      {/* 3. Compact Category Failure Distribution */}
      {data && data.category_breakdown.length > 0 && (
        <Card
          title={`Phân bố Fail chung (≥80% trượt @${data.k}) theo nhóm`}
          subtitle="Bấm vào thanh bất kỳ để lọc nhanh theo nhóm đó"
        >
          <div className="bm-cat-bars-grid">
            {data.category_breakdown.map((cb) => {
              const active = category === cb.category;
              return (
                <button
                  key={cb.category}
                  type="button"
                  className={active ? "bm-cat-bar-row active" : "bm-cat-bar-row"}
                  onClick={() => {
                    setCategory(active ? "" : cb.category);
                    setOffset(0);
                  }}
                >
                  <span className="bm-cat-name">{cb.category}</span>
                  <div className="bm-cat-track" aria-hidden="true">
                    <span style={{ width: `${Math.min(100, cb.fail80_rate * 100)}%` }} />
                  </div>
                  <span className="bm-cat-stat num">
                    <b>{(cb.fail80_rate * 100).toFixed(1)}%</b>{" "}
                    <span className="hint">({cb.fail80}/{cb.total})</span>
                  </span>
                </button>
              );
            })}
          </div>
        </Card>
      )}

      {/* 4. Clean Query List with Quick GT Preview + Dedicated Detail Page / New Tab */}
      {data && (
        <Card
          title={`${data.protocol_label} · ${data.filtered_total.toLocaleString()} query`}
          subtitle={`Trang ${Math.floor(offset / 25) + 1} / ${Math.max(1, Math.ceil(data.filtered_total / 25))} · Bấm vào câu lệnh để mở trang chi tiết Top-K ảnh, hoặc bấm "Mở tab mới ↗"`}
        >
          {data.items.length === 0 ? (
            <EmptyState kind="no-match" detail="Không có query nào khớp bộ lọc hiện tại." />
          ) : (
            <>
              <div className="bm-cf-list">
                {data.items.map((row) => {
                  const isOpen = expandedId === row.query_id;
                  const tone = row.fail_count === row.scored_models ? "bad" : row.fail_rate >= 0.8 ? "warn" : "info";
                  const detailHref = buildDetailHref(data.protocol, row.query_id, data.k, modelsParam);

                  return (
                    <div key={row.query_id} className={isOpen ? "bm-cf-item open" : "bm-cf-item"}>
                      <div className="bm-cf-head">
                        {row.ref_image_url && (
                          <button
                            type="button"
                            className="bm-cf-thumb-btn"
                            onClick={() => setLightbox({ src: row.ref_image_url, caption: `Reference · ${row.query_id}` })}
                            title="Phóng to ảnh gốc"
                          >
                            <img src={row.ref_image_url} alt="" loading="lazy" />
                          </button>
                        )}

                        <div
                          className="bm-cf-main bm-cf-clickable"
                          role="button"
                          tabIndex={0}
                          onClick={() => openDetailInApp(row.query_id)}
                          onKeyDown={(ev) => {
                            if (ev.key === "Enter" || ev.key === " ") {
                              ev.preventDefault();
                              openDetailInApp(row.query_id);
                            }
                          }}
                          title="Mở trang chi tiết Top-K ảnh của query này"
                        >
                          <div className="row" style={{ gap: 6 }}>
                            <span className="mono cell-strong" style={{ fontSize: ".83rem" }}>{row.query_id}</span>
                            <Badge tone="info">{row.category}</Badge>
                            {row.l1_interest && data.dataset === "pinpoint" && <Badge tone="neutral">{row.l1_interest}</Badge>}
                            <Badge tone={tone}>
                              {row.fail_count}/{row.scored_models} trượt @{data.k} ({(row.fail_rate * 100).toFixed(0)}%)
                            </Badge>
                            {row.ref_copy_count > 0 && <Badge tone="warn">Ref#1: {row.ref_copy_count}</Badge>}
                            <span className="hint tabular">Best {fmtRank(row.best_rank)} · Med {fmtRank(row.median_rank)}</span>
                          </div>
                          <p className="bm-cf-caption">“{row.text}”</p>
                          <div className="bm-dot-strip">
                            {row.per_model.map((pm) => (
                              <span
                                key={pm.key}
                                className={pm.failed ? (pm.ref_copy ? "bm-dot bm-dot-ref" : "bm-dot bm-dot-fail") : "bm-dot bm-dot-hit"}
                                title={`${pm.display}: ${fmtRank(pm.rank)}`}
                              >
                                {pm.display} <b>{fmtRank(pm.rank)}</b>
                              </span>
                            ))}
                          </div>
                        </div>

                        <div className="bm-cf-actions">
                          <div className="row" style={{ gap: 6, flexWrap: "nowrap" }}>
                            <button
                              type="button"
                              className="primary"
                              style={{ minHeight: 34, padding: "4px 11px", fontSize: ".82rem", whiteSpace: "nowrap" }}
                              onClick={() => openDetailInApp(row.query_id)}
                            >
                              Chi tiết Top-K
                            </button>
                            <a
                              className="bm-btn-link"
                              style={{ minHeight: 34, padding: "4px 10px", fontSize: ".82rem", whiteSpace: "nowrap" }}
                              href={detailHref}
                              target="_blank"
                              rel="noopener noreferrer"
                              title="Mở trang chi tiết trong tab trình duyệt mới"
                            >
                              Tab mới ↗
                            </a>
                          </div>
                          <button
                            type="button"
                            className="rowbtn"
                            style={{ fontSize: ".8rem", color: "var(--ink-3)" }}
                            onClick={() => setExpandedId(isOpen ? null : row.query_id)}
                          >
                            {isOpen ? "Ẩn ảnh GT ▴" : "Xem nhanh GT ▾"}
                          </button>
                        </div>
                      </div>

                      {/* Lightweight inline Ref + GT + Rank table preview (reverted clean version) */}
                      {isOpen && (
                        <div className="bm-cf-detail">
                          <div className="bm-cf-imgs">
                            {row.ref_image_url && (
                              <button
                                type="button"
                                className="bm-mini-card"
                                style={{ width: 128 }}
                                onClick={() => setLightbox({ src: row.ref_image_url, caption: `Ảnh gốc · ${row.ref_image_id}` })}
                              >
                                <img src={row.ref_image_url} alt="Reference" />
                                <span className="hint">Ảnh gốc (Ref)</span>
                              </button>
                            )}
                            {row.gt_image_url && (
                              <button
                                type="button"
                                className="bm-mini-card bm-cand-gt"
                                style={{ width: 128 }}
                                onClick={() => setLightbox({ src: row.gt_image_url, caption: `Đáp án đúng (GT) · ${row.gt_image_id}` })}
                              >
                                <img src={row.gt_image_url} alt="Ground Truth" />
                                <span className="hint-ok">Đáp án đúng (GT)</span>
                              </button>
                            )}
                          </div>
                          <div className="table-wrap" style={{ flex: 1 }}>
                            <table className="bm-compact-table">
                              <thead>
                                <tr>
                                  <th>Mô hình</th>
                                  <th className="num">Best GT Rank</th>
                                  <th>Kết quả @{data.k}</th>
                                  <th>Ref Copy #1</th>
                                </tr>
                              </thead>
                              <tbody>
                                {row.per_model.map((pm) => (
                                  <tr key={pm.key}>
                                    <td><span className="cell-strong">{pm.display}</span></td>
                                    <td className="num mono">{fmtRank(pm.rank)}</td>
                                    <td>
                                      <Badge tone={pm.failed ? "bad" : "ok"}>
                                        {pm.failed ? `Trượt @${data.k}` : `Trúng @${data.k}`}
                                      </Badge>
                                    </td>
                                    <td>{pm.ref_copy ? <Badge tone="warn">Ref Copy #1</Badge> : <span className="hint">—</span>}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="bm-pager" style={{ justifyContent: "flex-end", gap: 12 }}>
                <button type="button" className="secondary" disabled={offset <= 0} onClick={() => setOffset(Math.max(0, offset - 25))}>
                  ← Trước
                </button>
                <span className="hint tabular">
                  Trang {Math.floor(offset / 25) + 1} / {Math.max(1, Math.ceil(data.filtered_total / 25))}
                </span>
                <button type="button" className="secondary" disabled={offset + 25 >= data.filtered_total} onClick={() => setOffset(offset + 25)}>
                  Sau →
                </button>
              </div>
            </>
          )}
        </Card>
      )}

      {lightbox && (
        <div className="bm-lightbox" role="dialog" aria-modal="true" aria-label={lightbox.caption} onClick={() => setLightbox(null)}>
          <div className="bm-lightbox-inner" onClick={(ev) => ev.stopPropagation()}>
            <img src={lightbox.src} alt={lightbox.caption} />
            <div className="bm-lightbox-bar">
              <span className="mono">{lightbox.caption}</span>
              <button type="button" className="secondary" onClick={() => setLightbox(null)}>Đóng (Esc)</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
