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

const PRESET_TABS: Array<{ id: string; label: string; desc: string }> = [
  { id: "fail80", label: "Fail chung ≥80%", desc: "Mặc định: ít nhất 80% mô hình cùng trượt" },
  { id: "all_fail", label: "100% cùng trượt", desc: "Toàn bộ mô hình đều không tìm ra đáp án" },
  { id: "fail50", label: "≥50% cùng trượt", desc: "Hơn một nửa số mô hình bị trượt" },
  { id: "disagree", label: "Bất đồng mạnh", desc: "Phần lớn trượt nhưng 1–2 mô hình vẫn trúng" },
  { id: "ref_copy50", label: "Ref Copy ≥50%", desc: "≥50% mô hình trả lại chính ảnh gốc ở Top-1" },
  { id: "all", label: "Tất cả query", desc: "Toàn bộ truy vấn (xếp từ dễ hỏng nhất)" },
];

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

function fmtRank(r: number | null): string {
  return r === null ? ">50" : `#${r}`;
}

function InlineQueryRankings({
  protocol,
  queryId,
  modelsParam,
  k,
  onZoom,
}: {
  protocol: string;
  queryId: string;
  modelsParam: string;
  k: number;
  onZoom: (src: string, caption: string) => void;
}) {
  const [detail, setDetail] = useState<CommonFailureDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [err, setErr] = useState<string | null>(null);
  const [viewMode, setViewMode] = useState<"single" | "stacked">("single");
  const [activeModel, setActiveModel] = useState<string>("");
  const [compareModel, setCompareModel] = useState<string>("");
  const [topK, setTopK] = useState<number>(10);

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

  if (loading && !detail) return <LoadingState label="Đang tải Top-K ảnh từng mô hình…" />;
  if (err) return <ErrorState message={err} />;
  if (!detail) return null;

  const primaryBlock = detail.models.find((m) => m.key === activeModel) ?? detail.models[0] ?? null;
  const compareBlock = compareModel ? detail.models.find((m) => m.key === compareModel) ?? null : null;

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
        onClick={() => onZoom(c.image_url, `Rank #${c.rank} · ${c.image_id}`)}
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
    <div className="bm-cf-detail-rich">
      {/* Reference + Ground Truths + Hard Negatives row */}
      <div className="bm-cf-refgt-strip">
        <div className="bm-cf-strip-group">
          <span className="field-label">Ảnh gốc (Reference)</span>
          <div className="bm-mini-strip">
            <button
              type="button"
              className="bm-mini-card"
              style={{ width: 124, borderColor: "var(--accent)" }}
              onClick={() => onZoom(detail.query.ref_image_url, `Ảnh gốc · ${detail.query.ref_image_id}`)}
            >
              <img src={detail.query.ref_image_url} alt="Reference" />
              <span className="hint mono" style={{ fontSize: ".68rem", overflow: "hidden", textOverflow: "ellipsis" }}>
                {detail.query.ref_image_id}
              </span>
            </button>
          </div>
        </div>

        <div className="bm-cf-strip-group">
          <span className="field-label">Đáp án đúng (Ground Truth · {detail.ground_truths.length})</span>
          <div className="bm-mini-strip">
            {detail.ground_truths.slice(0, 6).map((gt) => (
              <button
                key={gt.image_id}
                type="button"
                className="bm-mini-card bm-cand-gt"
                style={{ width: 124 }}
                onClick={() => onZoom(gt.image_url, `Ground Truth · ${gt.image_id}`)}
              >
                <img src={gt.image_url} alt={gt.image_id} />
                <span className="hint-ok mono" style={{ fontSize: ".68rem", overflow: "hidden", textOverflow: "ellipsis" }}>
                  {gt.image_id}
                </span>
              </button>
            ))}
          </div>
        </div>

        {detail.hard_negatives.length > 0 && (
          <div className="bm-cf-strip-group">
            <span className="field-label">Bẫy khó (Hard Neg · {detail.hard_negatives.length})</span>
            <div className="bm-mini-strip">
              {detail.hard_negatives.slice(0, 4).map((hn) => (
                <button
                  key={hn.image_id}
                  type="button"
                  className="bm-mini-card bm-cand-hneg"
                  style={{ width: 96 }}
                  onClick={() => onZoom(hn.image_url, `Hard Negative · ${hn.image_id}`)}
                >
                  <img src={hn.image_url} alt={hn.image_id} />
                  <span className="hint mono" style={{ fontSize: ".66rem" }}>{hn.is_ref ? "Ref" : "HNeg"}</span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Ranking viewer toolbar */}
      <div className="bm-cf-rank-bar">
        <div className="bm-seg" role="group" aria-label="Chế độ xem Top-K">
          <button
            type="button"
            className={viewMode === "single" ? "bm-seg-btn active" : "bm-seg-btn"}
            onClick={() => setViewMode("single")}
          >
            Chi tiết Top-{topK} theo model
          </button>
          <button
            type="button"
            className={viewMode === "stacked" ? "bm-seg-btn active" : "bm-seg-btn"}
            onClick={() => setViewMode("stacked")}
          >
            Xếp tầng Top-5 tất cả ({detail.models.length} model)
          </button>
        </div>

        {viewMode === "single" && (
          <div className="row" style={{ marginLeft: "auto" }}>
            <label className="hint" htmlFor={`cmp-${queryId}`}>So sánh thêm:</label>
            <select
              id={`cmp-${queryId}`}
              value={compareModel}
              onChange={(ev) => setCompareModel(ev.target.value)}
              style={{ minHeight: 34, padding: "4px 8px", fontSize: ".82rem" }}
            >
              <option value="">— 1 model —</option>
              {detail.models
                .filter((m) => m.key !== primaryBlock?.key)
                .map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.display} ({fmtRank(m.best_gt_rank)})
                  </option>
                ))}
            </select>
            {detail.dataset !== "fashioniq" && (
              <select
                aria-label="Số lượng Top-K"
                value={topK}
                onChange={(ev) => setTopK(Number(ev.target.value))}
                style={{ minHeight: 34, padding: "4px 8px", fontSize: ".82rem" }}
              >
                <option value={10}>Top-10</option>
                <option value={15}>Top-15</option>
                <option value={20}>Top-20</option>
              </select>
            )}
          </div>
        )}
      </div>

      {/* Mode 1: Single / Side-by-side Model Top-K Image Grid */}
      {viewMode === "single" && primaryBlock && (
        <div className="stack">
          <div className="bm-consensus-grid">
            {detail.models.map((m) => {
              const isAct = m.key === primaryBlock.key;
              const failed = m.best_gt_rank === null || m.best_gt_rank > k;
              return (
                <button
                  key={m.key}
                  type="button"
                  className={isAct ? "bm-cons-chip active" : "bm-cons-chip"}
                  onClick={() => setActiveModel(m.key)}
                >
                  <b>{m.display}</b>
                  <Badge tone={failed ? "bad" : "ok"}>{fmtRank(m.best_gt_rank)}</Badge>
                  {m.top1_is_ref && <span className="bm-tag bm-tag-warn">Ref#1</span>}
                </button>
              );
            })}
          </div>

          <div className={compareBlock ? "bm-compare-grids" : "stack"}>
            <div className="bm-cf-grid-box">
              <div className="row" style={{ marginBottom: 8 }}>
                <b>Top-{primaryBlock.candidates.length} · {primaryBlock.display}</b>
                <Badge tone={primaryBlock.best_gt_rank !== null && primaryBlock.best_gt_rank <= k ? "ok" : "bad"}>
                  GT Rank: {fmtRank(primaryBlock.best_gt_rank)}
                </Badge>
                <span className="hint" style={{ marginLeft: "auto" }}>
                  Khung xanh = GT · Vàng = Ref Copy · Đỏ = Hard Neg
                </span>
              </div>
              {primaryBlock.candidates.length > 0 ? (
                <div className="bm-cand-grid">
                  {primaryBlock.candidates.map((c) => renderCandidate(c, false))}
                </div>
              ) : (
                <EmptyState kind="no-results" detail="Lượt chạy này không lưu danh sách Top-K ảnh ứng viên." />
              )}
            </div>

            {compareBlock && (
              <div className="bm-cf-grid-box">
                <div className="row" style={{ marginBottom: 8 }}>
                  <b>Top-{compareBlock.candidates.length} · {compareBlock.display}</b>
                  <Badge tone={compareBlock.best_gt_rank !== null && compareBlock.best_gt_rank <= k ? "ok" : "bad"}>
                    GT Rank: {fmtRank(compareBlock.best_gt_rank)}
                  </Badge>
                </div>
                <div className="bm-cand-grid">
                  {compareBlock.candidates.map((c) => renderCandidate(c, false))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Mode 2: Stacked Top-5 across ALL models simultaneously */}
      {viewMode === "stacked" && (
        <div className="bm-stacked-models">
          {detail.models.map((m) => {
            const failed = m.best_gt_rank === null || m.best_gt_rank > k;
            return (
              <div key={m.key} className="bm-stacked-row">
                <div className="bm-stacked-meta">
                  <b>{m.display}</b>
                  <span className="bm-sub">{m.family}</span>
                  <div className="row" style={{ marginTop: 4 }}>
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
    </div>
  );
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
    subtitle: "3 protocol chính chủ (Original / Full Gallery / Val Split)",
    defaultProtocol: "fashioniq_original_split",
  },
  {
    id: "pinpoint",
    label: "PinPoint",
    subtitle: "1,337 test queries · 8 lát cắt Fashion & Non-Fashion",
    defaultProtocol: "pinpoint_indomain",
  },
  {
    id: "edir",
    label: "EDIR",
    subtitle: "1,003 test queries · 15 phép chỉnh sửa độc lập",
    defaultProtocol: "edir_indomain",
  },
];

export function CommonFailuresPage({
  onInspectBenchmarkQuery,
}: {
  onInspectBenchmarkQuery: (seed: InspectorSeed) => void;
}) {
  const [activeDataset, setActiveDataset] = useState<"fashioniq" | "pinpoint" | "edir">("fashioniq");
  const [protocol, setProtocol] = useState<string>("fashioniq_original_split");
  const [selectedModels, setSelectedModels] = useState<string[]>([]); // empty = ALL ready models by default!
  const [showModelPicker, setShowModelPicker] = useState<boolean>(false);
  const [preset, setPreset] = useState<string>("fail80");
  const [k, setK] = useState<number>(10);
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
  }, [protocol, modelsParam, preset, k, slice, category, sort, searchQuery, offset]);

  const switchDataset = (nextDs: "fashioniq" | "pinpoint" | "edir") => {
    const tab = DATASET_TABS.find((t) => t.id === nextDs)!;
    setActiveDataset(nextDs);
    switchProtocol(tab.defaultProtocol);
  };

  const switchProtocol = (nextProto: string) => {
    setProtocol(nextProto);
    setSelectedModels([]); // Reset to ALL models of the new protocol automatically
    setSlice("all");
    setCategory("");
    setOffset(0);
    setExpandedId(null);
    if (nextProto.startsWith("fashioniq") && preset === "ref_copy50") {
      setPreset("fail80");
    }
    if (nextProto.startsWith("fashioniq") && sort === "ref_copy_desc") {
      setSort("fail_desc");
    }
  };

  const toggleModel = (key: string) => {
    if (!data) return;
    const active = selectedModels.length ? selectedModels : data.all_models.map((m) => m.key);
    const has = active.includes(key);
    const next = has ? active.filter((x) => x !== key) : [...active, key];
    // Never allow 0 models selected; if user clicks all back on, normalize to [] (= all)
    if (next.length === 0) return;
    setSelectedModels(next.length === data.all_models.length ? [] : next);
    setOffset(0);
  };

  const applyFamilyPreset = (family: string) => {
    if (!data) return;
    if (family === "all") {
      setSelectedModels([]);
    } else if (family === "top5") {
      setSelectedModels(data.all_models.slice(0, 5).map((m) => m.key));
    } else {
      const matched = data.all_models.filter((m) => m.family === family).map((m) => m.key);
      if (matched.length) setSelectedModels(matched);
    }
    setOffset(0);
  };

  const datasetProtocols = useMemo(
    () => (data?.available_protocols ?? []).filter((p) => p.dataset === activeDataset),
    [data?.available_protocols, activeDataset],
  );

  const exportUrl = useMemo(() => {
    const params = new URLSearchParams({ protocol, preset, k: String(k), sort });
    if (modelsParam) params.set("models", modelsParam);
    if (slice && slice !== "all") params.set("slice", slice);
    if (category) params.set("category", category);
    if (searchQuery) params.set("q", searchQuery);
    return `/api/benchmarks/common-failures/export.csv?${params.toString()}`;
  }, [protocol, modelsParam, preset, k, slice, category, sort, searchQuery]);

  const activeModelSet = useMemo(() => {
    if (!data) return new Set<string>();
    return new Set(selectedModels.length ? selectedModels : data.all_models.map((m) => m.key));
  }, [data, selectedModels]);

  return (
    <div className="stack">
      {/* Dataset switcher bar: strictly separates FashionIQ (default), PinPoint, and EDIR */}
      <div className="bm-ds-tabs" role="tablist" aria-label="Chọn bộ dữ liệu (Dataset)">
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

      {/* Top control bar: zero-click default, optional protocol / model / category filters */}
      <Card
        title={
          <span className="row">
            <Badge tone="info">{activeDataset === "fashioniq" ? "FashionIQ" : activeDataset === "pinpoint" ? "PinPoint" : "EDIR"}</Badge>
            <span>Lỗi chung giữa các mô hình (hiện sẵn mặc định — lọc tuỳ chọn)</span>
          </span>
        }
        subtitle="Tự động gộp toàn bộ mô hình đã hoàn tất của protocol thuộc bộ dữ liệu đang chọn và hiển thị ngay các query fail chung (≥80% cùng trượt)."
        actions={
          <div className="row">
            {loading && <Badge tone="info">Đang cập nhật…</Badge>}
            <a className="bm-btn-link" href={exportUrl} download>
              Tải CSV
            </a>
          </div>
        }
      >
        {/* Protocol / Regime pills strictly scoped to the selected dataset */}
        {datasetProtocols.length > 0 && (
          <div className="bm-slice-grid" role="group" aria-label="Chọn protocol / chế độ" style={{ marginBottom: 12 }}>
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
                  <span className="bm-slice-cnt">{opt.ready_models} model</span>
                </button>
              );
            })}
          </div>
        )}

        <div className="toolbar">
          <div className="field">
            <label className="field-label" htmlFor="cf-cutoff">Ngưỡng trượt (Top-K)</label>
            <select
              id="cf-cutoff"
              value={k}
              onChange={(ev) => {
                setK(Number(ev.target.value));
                setOffset(0);
              }}
            >
              <option value={1}>Trượt Top-1 (R@1)</option>
              <option value={10}>Trượt Top-10 (R@10)</option>
              <option value={50}>Trượt Top-50 (R@50)</option>
            </select>
          </div>

          {data && data.slices.length > 0 && (
            <div className="field">
              <label className="field-label" htmlFor="cf-slice">Lát cắt PinPoint</label>
              <select
                id="cf-slice"
                value={slice}
                onChange={(ev) => {
                  setSlice(ev.target.value);
                  setOffset(0);
                }}
              >
                {data.slices.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label} ({s.count})
                  </option>
                ))}
              </select>
            </div>
          )}

          {data && data.categories.length > 0 && (
            <div className="field">
              <label className="field-label" htmlFor="cf-cat">
                {data.dataset === "edir" ? "Phép chỉnh sửa" : "Nhóm / Category"}
              </label>
              <select
                id="cf-cat"
                value={category}
                onChange={(ev) => {
                  setCategory(ev.target.value);
                  setOffset(0);
                }}
              >
                <option value="">Tất cả ({data.categories.length})</option>
                {data.categories.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.id} ({c.count})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="field">
            <label className="field-label" htmlFor="cf-sort">Sắp xếp</label>
            <select
              id="cf-sort"
              value={sort}
              onChange={(ev) => {
                setSort(ev.target.value);
                setOffset(0);
              }}
            >
              <option value="fail_desc">Nhiều model trượt nhất</option>
              {data?.dataset !== "fashioniq" && <option value="ref_copy_desc">Nhiều Ref Copy nhất</option>}
              <option value="best">Ít model trượt nhất</option>
              <option value="id">Theo Query ID</option>
            </select>
          </div>

          <div className="field grow">
            <label className="field-label" htmlFor="cf-search">Tìm caption / ID</label>
            <input
              id="cf-search"
              type="search"
              placeholder="Lọc nhanh theo chữ trong câu lệnh hoặc query ID…"
              value={searchInput}
              onChange={(ev) => setSearchInput(ev.target.value)}
            />
          </div>
        </div>

        {/* Preset pills with live counts */}
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
        </div>

        {/* Optional Model Filter Drawer (already auto-selected to ALL ready models!) */}
        {data && data.all_models.length > 0 && (
          <div className="bm-model-filter-bar">
            <div className="row">
              <Badge tone={selectedModels.length === 0 ? "ok" : "info"}>
                {selectedModels.length === 0
                  ? `Đang gộp tất cả ${data.all_models.length}/${data.all_models.length} mô hình (mặc định)`
                  : `Đang lọc ${data.selected_model_count}/${data.all_models.length} mô hình`}
              </Badge>
              <button
                type="button"
                className="secondary"
                style={{ minHeight: 32, padding: "4px 10px", fontSize: ".82rem" }}
                onClick={() => setShowModelPicker((v) => !v)}
              >
                {showModelPicker ? "Ẩn bộ lọc model ▴" : "Lọc chọn model cụ thể ▾"}
              </button>
              {selectedModels.length > 0 && (
                <button
                  type="button"
                  className="ghost"
                  style={{ minHeight: 32, padding: "4px 8px", fontSize: ".82rem" }}
                  onClick={() => applyFamilyPreset("all")}
                >
                  Đặt lại tất cả
                </button>
              )}
              {protocol === "fashioniq_all_protocols" && (
                <Badge tone="warn">Lưu ý: gộp 3 protocol FashionIQ có gallery khác nhau — xem chi tiết từng protocol nếu cần so công bằng 100%</Badge>
              )}
            </div>

            {showModelPicker && (
              <div className="bm-model-picker-body">
                <div className="row" style={{ marginBottom: 8 }}>
                  <span className="hint">Chọn nhanh:</span>
                  <button type="button" className="bm-pill" onClick={() => applyFamilyPreset("all")}>Tất cả ({data.all_models.length})</button>
                  <button type="button" className="bm-pill" onClick={() => applyFamilyPreset("top5")}>Top-5</button>
                  {Array.from(new Set(data.all_models.map((m) => m.family))).map((fam) => (
                    <button key={fam} type="button" className="bm-pill" onClick={() => applyFamilyPreset(fam)}>
                      Nhóm {fam}
                    </button>
                  ))}
                </div>
                <div className="bm-consensus-grid">
                  {data.all_models.map((m) => {
                    const checked = activeModelSet.has(m.key);
                    return (
                      <button
                        key={m.key}
                        type="button"
                        className={checked ? "bm-cons-chip active" : "bm-cons-chip"}
                        onClick={() => toggleModel(m.key)}
                      >
                        <span>{checked ? "✓" : "○"}</span>
                        <b>{m.display}</b>
                        <small className="hint">{m.family}</small>
                      </button>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}
      </Card>

      {error && <ErrorState message={error} />}
      {loading && !data && <LoadingState label="Đang tổng hợp các lỗi chung giữa các mô hình…" />}

      {/* Category Failure Concentration Bar */}
      {data && data.category_breakdown.length > 0 && (
        <Card
          title={`Phân bố Fail chung (≥80% trượt Top-${data.k}) theo nhóm — bấm thanh bất kỳ để lọc`}
          subtitle="Cho thấy ngay nhóm câu lệnh / phép chỉnh sửa nào khiến số đông mô hình cùng gãy nhiều nhất"
        >
          <div className="bm-cat-bars">
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
                    <span className="hint">({cb.fail80}/{cb.total} · all-fail {cb.all_fail})</span>
                  </span>
                </button>
              );
            })}
          </div>
        </Card>
      )}

      {/* Main Common Failure List */}
      {data && (
        <Card
          title={`Danh sách lỗi chung · ${data.protocol_label} (${data.filtered_total.toLocaleString()} query)`}
          subtitle={`Hiển thị ${data.filtered_total === 0 ? 0 : data.offset + 1}–${Math.min(data.offset + data.limit, data.filtered_total)} · Bấm trực tiếp vào bất kỳ query nào để xem ảnh gốc, ảnh GT và lưới Top-K ảnh từng mô hình xếp hạng`}
        >
          {data.items.length === 0 ? (
            <EmptyState
              kind="no-match"
              detail="Không có query nào khớp ngưỡng lọc hiện tại. Hãy thử đổi mức độ fail chung (vd ≥50%) hoặc xoá bộ lọc."
            />
          ) : (
            <>
              <div className="bm-cf-list">
                {data.items.map((row) => {
                  const isOpen = expandedId === row.query_id;
                  const tone = row.fail_count === row.scored_models ? "bad" : row.fail_rate >= 0.8 ? "warn" : "info";
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
                          onClick={() => setExpandedId(isOpen ? null : row.query_id)}
                          onKeyDown={(ev) => {
                            if (ev.key === "Enter" || ev.key === " ") {
                              ev.preventDefault();
                              setExpandedId(isOpen ? null : row.query_id);
                            }
                          }}
                        >
                          <div className="row" style={{ gap: 6 }}>
                            <span className="mono cell-strong" style={{ fontSize: ".84rem" }}>{row.query_id}</span>
                            <Badge tone="info">{row.category}</Badge>
                            {row.l1_interest && <Badge tone="neutral">{row.l1_interest}</Badge>}
                            <Badge tone={tone}>
                              {row.fail_count}/{row.scored_models} trượt @{data.k} ({(row.fail_rate * 100).toFixed(0)}%)
                            </Badge>
                            {row.ref_copy_count > 0 && (
                              <Badge tone="warn">Ref Copy #1: {row.ref_copy_count}</Badge>
                            )}
                            <span className="hint tabular">Best: {fmtRank(row.best_rank)} · Med: {fmtRank(row.median_rank)}</span>
                          </div>
                          <p className="bm-cf-caption">“{row.text}”</p>
                          {/* Compact visual hit/miss dot bar across all models */}
                          <div className="bm-dot-strip" aria-label="Trạng thái từng mô hình">
                            {row.per_model.map((pm) => (
                              <span
                                key={pm.key}
                                className={pm.failed ? (pm.ref_copy ? "bm-dot bm-dot-ref" : "bm-dot bm-dot-fail") : "bm-dot bm-dot-hit"}
                                title={`${pm.display}: ${fmtRank(pm.rank)}${pm.ref_copy ? " (Ref Copy #1)" : ""}`}
                              >
                                {pm.display} <b>{fmtRank(pm.rank)}</b>
                              </span>
                            ))}
                          </div>
                        </div>

                        <div className="bm-cf-actions">
                          <button
                            type="button"
                            className={isOpen ? "primary" : "secondary"}
                            style={{ minHeight: 34, padding: "4px 10px", fontSize: ".82rem", whiteSpace: "nowrap" }}
                            onClick={() => setExpandedId(isOpen ? null : row.query_id)}
                          >
                            {isOpen ? "Thu gọn ▴" : "Xem Top-K ảnh ▾"}
                          </button>
                          {data.dataset !== "fashioniq" && (
                            <button
                              type="button"
                              className="rowbtn"
                              style={{ color: "var(--accent)", fontWeight: 600, whiteSpace: "nowrap" }}
                              onClick={() =>
                                onInspectBenchmarkQuery({
                                  dataset: data.dataset === "edir" ? "edir" : "pinpoint",
                                  regime: data.regime as BenchmarkRegime,
                                  queryId: row.query_id,
                                })
                              }
                            >
                              Mở Inspector →
                            </button>
                          )}
                        </div>
                      </div>

                      {isOpen && (
                        <InlineQueryRankings
                          protocol={data.protocol}
                          queryId={row.query_id}
                          modelsParam={modelsParam}
                          k={data.k}
                          onZoom={(src, caption) => setLightbox({ src, caption })}
                        />
                      )}
                    </div>
                  );
                })}
              </div>

              <div className="bm-pager" style={{ justifyContent: "flex-end", gap: 12 }}>
                <button
                  type="button"
                  className="secondary"
                  disabled={offset <= 0}
                  onClick={() => setOffset(Math.max(0, offset - 25))}
                >
                  ← Trước
                </button>
                <span className="hint tabular">
                  Trang {Math.floor(offset / 25) + 1} / {Math.max(1, Math.ceil(data.filtered_total / 25))}
                </span>
                <button
                  type="button"
                  className="secondary"
                  disabled={offset + 25 >= data.filtered_total}
                  onClick={() => setOffset(offset + 25)}
                >
                  Sau →
                </button>
              </div>
            </>
          )}
        </Card>
      )}

      {lightbox && (
        <div
          className="bm-lightbox"
          role="dialog"
          aria-modal="true"
          aria-label={lightbox.caption}
          onClick={() => setLightbox(null)}
        >
          <div className="bm-lightbox-inner" onClick={(ev) => ev.stopPropagation()}>
            <img src={lightbox.src} alt={lightbox.caption} />
            <div className="bm-lightbox-bar">
              <span className="mono">{lightbox.caption}</span>
              <button type="button" className="secondary" onClick={() => setLightbox(null)}>
                Đóng (Esc)
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
