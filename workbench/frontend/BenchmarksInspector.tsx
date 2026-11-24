import { useEffect, useMemo, useState } from "react";
import { api } from "./client";
import { Badge, Card, EmptyState, ErrorState, LoadingState } from "./ui";
import type {
  BenchmarkDataset,
  BenchmarkQueriesResponse,
  BenchmarkQueryDetailResponse,
  BenchmarkRegime,
  CandidateImageItem,
  EdirSummaryResponse,
  InspectorSeed,
  PinPointSummaryResponse,
} from "./benchmarksTypes";

const REGIME_LABELS: Record<BenchmarkRegime, string> = {
  indomain: "In-Domain Fine-tuned",
  fiq_transfer: "FashionIQ Transfer",
  cirr_transfer: "CIRR Transfer",
};

const OUTCOME_OPTIONS: Array<{ id: string; label: string; needsCompare?: boolean }> = [
  { id: "all", label: "Tất cả" },
  { id: "miss10", label: "Trượt Top-10" },
  { id: "miss50", label: "Trượt Top-50" },
  { id: "hit1", label: "Trúng Top-1" },
  { id: "hit10", label: "Trúng Top-10" },
  { id: "ref_copy", label: "Ref Copy (#1)" },
  { id: "improved", label: "Cải thiện", needsCompare: true },
  { id: "regressed", label: "Giảm hạng", needsCompare: true },
];

function fmtScore(dataset: BenchmarkDataset, score: number | undefined | null): string {
  if (score === undefined || score === null || Number.isNaN(score)) return "—";
  return dataset === "edir" ? `${(score * 100).toFixed(0)}%` : score.toFixed(3);
}

function fmtRank(rank: number | null | undefined): string {
  if (rank === null || rank === undefined) return ">50";
  return `#${rank}`;
}

function CandidateCard({
  item,
  onZoom,
}: {
  item: CandidateImageItem;
  onZoom: (src: string, caption: string) => void;
}) {
  const cls = item.is_gt
    ? "bm-cand bm-cand-gt"
    : item.is_hard_neg
      ? "bm-cand bm-cand-hneg"
      : item.is_ref
        ? "bm-cand bm-cand-ref"
        : "bm-cand";

  return (
    <button
      type="button"
      className={cls}
      onClick={() => onZoom(item.image_url, `Rank #${item.rank} · ${item.image_id}`)}
      title={`Rank #${item.rank} · ${item.image_id}`}
    >
      <div className="bm-cand-img-wrap">
        <img src={item.image_url} alt={item.image_id} />
        <span className="bm-cand-rank">#{item.rank}</span>
        {item.is_gt && <span className="bm-cand-flag bm-flag-gt">GT</span>}
        {!item.is_gt && item.is_hard_neg && <span className="bm-cand-flag bm-flag-hneg">Hard Neg</span>}
        {!item.is_gt && !item.is_hard_neg && item.is_ref && (
          <span className="bm-cand-flag bm-flag-ref">Ref Copy</span>
        )}
      </div>
      <span className="bm-cand-id mono">{item.image_id}</span>
    </button>
  );
}

export function BenchmarksInspector({
  edirSummary,
  pinpointSummary,
  seed,
}: {
  edirSummary: EdirSummaryResponse | null;
  pinpointSummary: PinPointSummaryResponse | null;
  seed: InspectorSeed | null;
}) {
  const [dataset, setDataset] = useState<BenchmarkDataset>(seed?.dataset ?? "pinpoint");
  const [model, setModel] = useState<string>(seed?.model ?? "");
  const [regime, setRegime] = useState<BenchmarkRegime>(seed?.regime ?? "indomain");
  const [compareModel, setCompareModel] = useState<string>(seed?.compareModel ?? "");
  const [compareRegime, setCompareRegime] = useState<BenchmarkRegime>(seed?.compareRegime ?? "fiq_transfer");
  const [slice, setSlice] = useState<string>(seed?.slice ?? "all");
  const [category, setCategory] = useState<string>(seed?.category ?? "");
  const [outcome, setOutcome] = useState<string>(seed?.outcome ?? "all");
  const [sort, setSort] = useState<string>("worst");
  const [searchInput, setSearchInput] = useState<string>("");
  const [searchQuery, setSearchQuery] = useState<string>("");
  const [offset, setOffset] = useState<number>(0);
  const [topK, setTopK] = useState<number>(12);
  const [selectedQueryId, setSelectedQueryId] = useState<string>("");
  const [lightbox, setLightbox] = useState<{ src: string; caption: string } | null>(null);

  const [listData, setListData] = useState<BenchmarkQueriesResponse | null>(null);
  const [listLoading, setListLoading] = useState<boolean>(false);
  const [listError, setListError] = useState<string | null>(null);

  const [detailData, setDetailData] = useState<BenchmarkQueryDetailResponse | null>(null);
  const [detailLoading, setDetailLoading] = useState<boolean>(false);
  const [detailError, setDetailError] = useState<string | null>(null);

  // Apply external drilldown seed whenever the user clicks a heatmap cell or "Soi lỗi →".
  useEffect(() => {
    if (!seed) return;
    setDataset(seed.dataset);
    if (seed.model !== undefined) setModel(seed.model);
    if (seed.regime !== undefined) setRegime(seed.regime);
    setCategory(seed.category ?? "");
    setSlice(seed.slice ?? "all");
    setOutcome(seed.outcome ?? "all");
    setCompareModel(seed.compareModel ?? "");
    if (seed.compareRegime) setCompareRegime(seed.compareRegime);
    setOffset(0);
    setSelectedQueryId("");
  }, [seed]);

  // Debounce search text
  useEffect(() => {
    const t = window.setTimeout(() => {
      setSearchQuery(searchInput.trim());
      setOffset(0);
    }, 220);
    return () => window.clearTimeout(t);
  }, [searchInput]);

  // Escape closes lightbox
  useEffect(() => {
    if (!lightbox) return;
    const onKey = (ev: KeyboardEvent) => {
      if (ev.key === "Escape") setLightbox(null);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox]);

  const activeSummary = dataset === "edir" ? edirSummary : pinpointSummary;

  const readyOptions = useMemo(() => {
    if (!activeSummary) return [] as Array<{ model: string; display: string; regime: BenchmarkRegime; label: string }>;
    const out: Array<{ model: string; display: string; regime: BenchmarkRegime; label: string }> = [];
    for (const m of activeSummary.models) {
      if (m.indomain.status === "ready" && m.indomain.has_predictions) {
        out.push({ model: m.key, display: m.display, regime: "indomain", label: `${m.display} · In-Domain` });
      }
      if (dataset === "edir" && "cirr_transfer" in m && m.cirr_transfer.status === "ready" && m.cirr_transfer.has_predictions) {
        out.push({ model: m.key, display: m.display, regime: "cirr_transfer", label: `${m.display} · CIRR Transfer` });
      }
      if (m.fiq_transfer.status === "ready" && m.fiq_transfer.has_predictions) {
        out.push({ model: m.key, display: m.display, regime: "fiq_transfer", label: `${m.display} · FIQ Transfer` });
      }
    }
    return out;
  }, [activeSummary, dataset]);

  // Ensure valid default primary selection whenever dataset or loaded models change
  useEffect(() => {
    if (!readyOptions.length) return;
    const exists = readyOptions.some((o) => o.model === model && o.regime === regime);
    if (!exists) {
      setModel(readyOptions[0].model);
      setRegime(readyOptions[0].regime);
    }
  }, [readyOptions, model, regime]);

  // Ensure compare selection stays valid for active dataset
  useEffect(() => {
    if (!compareModel) return;
    const exists = readyOptions.some((o) => o.model === compareModel && o.regime === compareRegime);
    if (!exists) {
      setCompareModel("");
    }
  }, [readyOptions, compareModel, compareRegime]);

  const effectivePrimaryKey = model && regime ? `${model}:${regime}` : "";
  const effectiveCompareKey = compareModel && compareRegime ? `${compareModel}:${compareRegime}` : "";

  // Fetch paginated query list with AbortController
  useEffect(() => {
    if (!model || !regime) return;
    const ctrl = new AbortController();
    const params = new URLSearchParams({
      model,
      regime,
      sort,
      outcome,
      offset: String(offset),
      limit: "20",
    });
    if (dataset === "pinpoint" && slice) params.set("slice", slice);
    if (category) params.set("category", category);
    if (searchQuery) params.set("q", searchQuery);
    if (compareModel && compareRegime) {
      params.set("compare_model", compareModel);
      params.set("compare_regime", compareRegime);
    }

    setListLoading(true);
    setListError(null);
    api<BenchmarkQueriesResponse>(`/api/benchmarks/${dataset}/queries?${params.toString()}`, {
      signal: ctrl.signal,
    })
      .then((res) => {
        setListData(res);
        setListLoading(false);
        setSelectedQueryId((prev) => {
          if (prev && res.items.some((it) => it.query_id === prev)) return prev;
          return res.items[0]?.query_id ?? "";
        });
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        setListError(err instanceof Error ? err.message : String(err));
        setListLoading(false);
      });

    return () => ctrl.abort();
  }, [dataset, model, regime, compareModel, compareRegime, slice, category, outcome, sort, searchQuery, offset]);

  // Fetch selected query detail with AbortController
  useEffect(() => {
    if (!selectedQueryId || !model || !regime) {
      setDetailData(null);
      return;
    }
    const ctrl = new AbortController();
    const params = new URLSearchParams({
      model,
      regime,
      top_k: String(topK),
    });
    if (compareModel && compareRegime) {
      params.set("compare_model", compareModel);
      params.set("compare_regime", compareRegime);
    }

    setDetailLoading(true);
    setDetailError(null);
    api<BenchmarkQueryDetailResponse>(
      `/api/benchmarks/${dataset}/queries/${encodeURIComponent(selectedQueryId)}?${params.toString()}`,
      { signal: ctrl.signal },
    )
      .then((res) => {
        setDetailData(res);
        setDetailLoading(false);
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        setDetailError(err instanceof Error ? err.message : String(err));
        setDetailLoading(false);
      });

    return () => ctrl.abort();
  }, [dataset, selectedQueryId, model, regime, compareModel, compareRegime, topK]);

  const categories = dataset === "edir"
    ? (edirSummary?.subcategories ?? [])
    : (pinpointSummary?.categories ?? []);

  const canQuickCompareTransfer = useMemo(() => {
    const preferred: BenchmarkRegime = dataset === "edir" ? "cirr_transfer" : "fiq_transfer";
    if (readyOptions.some((o) => o.model === model && o.regime === preferred)) return preferred;
    if (readyOptions.some((o) => o.model === model && o.regime === "fiq_transfer")) return "fiq_transfer";
    return null;
  }, [readyOptions, model, dataset]);

  const switchDataset = (next: BenchmarkDataset) => {
    if (next === dataset) return;
    setDataset(next);
    setModel("");
    setRegime("indomain");
    setCompareModel("");
    setCategory("");
    setSlice("all");
    setOutcome("all");
    setOffset(0);
    setSelectedQueryId("");
  };

  return (
    <div className="stack">
      <Card
        title="Khám phá truy vấn, ảnh gốc & đối chiếu Top-K"
        subtitle="Mở từng test query thực tế để kiểm tra ảnh gốc, câu lệnh chỉnh sửa, Ground Truth, Hard Negative và lỗi Ref Copy across 15 models"
        actions={
          <div className="bm-seg" role="group" aria-label="Chọn benchmark">
            <button
              type="button"
              className={dataset === "pinpoint" ? "bm-seg-btn active" : "bm-seg-btn"}
              onClick={() => switchDataset("pinpoint")}
            >
              PinPoint ({pinpointSummary?.test_query_count ?? 1337})
            </button>
            <button
              type="button"
              className={dataset === "edir" ? "bm-seg-btn active" : "bm-seg-btn"}
              onClick={() => switchDataset("edir")}
            >
              EDIR ({edirSummary?.test_query_count ?? 1003})
            </button>
          </div>
        }
      >
        <div className="toolbar">
          <div className="field">
            <label className="field-label" htmlFor="bm-insp-primary">Mô hình chính</label>
            <select
              id="bm-insp-primary"
              value={effectivePrimaryKey}
              onChange={(ev) => {
                const [m, r] = ev.target.value.split(":");
                setModel(m);
                setRegime(r as BenchmarkRegime);
                setOffset(0);
              }}
            >
              {readyOptions.map((o) => (
                <option key={`${o.model}:${o.regime}`} value={`${o.model}:${o.regime}`}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label className="field-label" htmlFor="bm-insp-compare">So sánh đối chiếu (tuỳ chọn)</label>
            <select
              id="bm-insp-compare"
              value={effectiveCompareKey}
              onChange={(ev) => {
                if (!ev.target.value) {
                  setCompareModel("");
                  if (outcome === "improved" || outcome === "regressed") setOutcome("all");
                  return;
                }
                const [m, r] = ev.target.value.split(":");
                setCompareModel(m);
                setCompareRegime(r as BenchmarkRegime);
                setOffset(0);
              }}
            >
              <option value="">— Không so sánh —</option>
              {readyOptions.map((o) => (
                <option key={`cmp:${o.model}:${o.regime}`} value={`${o.model}:${o.regime}`}>
                  {o.label}
                </option>
              ))}
            </select>
          </div>

          {canQuickCompareTransfer && regime === "indomain" && (
            <button
              type="button"
              className="secondary"
              onClick={() => {
                setCompareModel(model);
                setCompareRegime(canQuickCompareTransfer);
                setSort("delta_desc");
                setOffset(0);
              }}
            >
              So sánh với {canQuickCompareTransfer === "cirr_transfer" ? "CIRR" : "FIQ"} Transfer
            </button>
          )}

          {dataset === "pinpoint" && pinpointSummary && (
            <div className="field">
              <label className="field-label" htmlFor="bm-insp-slice">Lát cắt PinPoint</label>
              <select
                id="bm-insp-slice"
                value={slice}
                onChange={(ev) => {
                  setSlice(ev.target.value);
                  setOffset(0);
                }}
              >
                {pinpointSummary.slices.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.label} ({s.count})
                  </option>
                ))}
              </select>
            </div>
          )}

          <div className="field">
            <label className="field-label" htmlFor="bm-insp-cat">
              {dataset === "edir" ? "Phép chỉnh sửa" : "Nhóm truy vấn"}
            </label>
            <select
              id="bm-insp-cat"
              value={category}
              onChange={(ev) => {
                setCategory(ev.target.value);
                setOffset(0);
              }}
            >
              <option value="">Tất cả ({categories.length})</option>
              {categories.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} ({c.count})
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label className="field-label" htmlFor="bm-insp-sort">Sắp xếp</label>
            <select
              id="bm-insp-sort"
              value={sort}
              onChange={(ev) => {
                setSort(ev.target.value);
                setOffset(0);
              }}
            >
              <option value="worst">Tệ nhất trước</option>
              <option value="best">Tốt nhất trước</option>
              {compareModel && <option value="delta_desc">Cải thiện mạnh nhất</option>}
              {compareModel && <option value="delta_asc">Giảm hạng mạnh nhất</option>}
              <option value="id">Theo Query ID</option>
            </select>
          </div>

          <div className="field grow">
            <label className="field-label" htmlFor="bm-insp-q">Tìm nhanh</label>
            <input
              id="bm-insp-q"
              type="search"
              placeholder="Tìm caption, query ID, lĩnh vực..."
              value={searchInput}
              onChange={(ev) => setSearchInput(ev.target.value)}
            />
          </div>
        </div>

        <div className="bm-outcome-pills" role="group" aria-label="Lọc theo kết quả">
          {OUTCOME_OPTIONS.filter((o) => !o.needsCompare || Boolean(compareModel)).map((opt) => {
            const cnt = listData?.outcome_counts?.[opt.id];
            return (
              <button
                key={opt.id}
                type="button"
                className={outcome === opt.id ? "bm-pill active" : "bm-pill"}
                onClick={() => {
                  setOutcome(opt.id);
                  setOffset(0);
                }}
              >
                <span>{opt.label}</span>
                {cnt !== undefined && <span className="bm-pill-count">{cnt}</span>}
              </button>
            );
          })}
        </div>
      </Card>

      {listError && <ErrorState message={listError} />}

      <div className="samples-layout">
        {/* Left column: Query List */}
        <Card
          title={`Danh sách query (${listData ? listData.filtered_total : "…"})`}
          subtitle={listData ? `Hiển thị ${listData.filtered_total === 0 ? 0 : listData.offset + 1}–${Math.min(listData.offset + listData.limit, listData.filtered_total)}` : ""}
          className="bm-list-card"
        >
          {listLoading && !listData && <LoadingState label="Đang tải danh sách query…" />}
          {listData && listData.items.length === 0 && (
            <EmptyState kind="no-match" detail="Không có query nào khớp bộ lọc hiện tại. Hãy nới lỏng bộ lọc hoặc xoá ô tìm kiếm." />
          )}
          {listData && listData.items.length > 0 && (
            <>
              <div className="bm-query-list">
                {listData.items.map((it) => {
                  const selected = it.query_id === selectedQueryId;
                  const tone = it.primary.hit1 ? "ok" : it.primary.hit10 ? "warn" : "bad";
                  return (
                    <button
                      key={it.query_id}
                      type="button"
                      className={selected ? "bm-qrow selected" : "bm-qrow"}
                      onClick={() => setSelectedQueryId(it.query_id)}
                    >
                      <img className="bm-qrow-thumb" src={it.ref_image_url} alt="" loading="lazy" />
                      <div className="bm-qrow-body">
                        <div className="bm-qrow-top">
                          <span className="mono bm-qrow-id">{it.query_id}</span>
                          <Badge tone={tone}>
                            {dataset === "edir"
                              ? fmtRank(it.primary.best_gt_rank)
                              : `AP ${fmtScore(dataset, it.primary.primary_score)}`}
                          </Badge>
                        </div>
                        <div className="bm-qrow-text">{it.text}</div>
                        <div className="bm-qrow-meta">
                          <span className="bm-tag">{it.category}</span>
                          {it.l1_interest && <span className="bm-tag">{it.l1_interest}</span>}
                          {it.primary.top1_is_ref && <span className="bm-tag bm-tag-warn">Ref #1</span>}
                          {it.delta !== null && Math.abs(it.delta) > 1e-6 && (
                            <span className={it.delta > 0 ? "bm-delta-pos" : "bm-delta-neg"}>
                              {it.delta > 0 ? "+" : ""}
                              {dataset === "edir" ? `${(it.delta * 100).toFixed(0)}%` : it.delta.toFixed(2)}
                            </span>
                          )}
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              <div className="bm-pager">
                <button
                  type="button"
                  className="secondary"
                  disabled={offset <= 0}
                  onClick={() => setOffset(Math.max(0, offset - 20))}
                >
                  ← Trước
                </button>
                <span className="hint tabular">
                  {Math.floor(offset / 20) + 1} / {Math.max(1, Math.ceil(listData.filtered_total / 20))}
                </span>
                <button
                  type="button"
                  className="secondary"
                  disabled={offset + 20 >= listData.filtered_total}
                  onClick={() => setOffset(offset + 20)}
                >
                  Sau →
                </button>
              </div>
            </>
          )}
        </Card>

        {/* Right column: Detailed Visual Inspector */}
        <div className="stack">
          {detailLoading && !detailData && <LoadingState label="Đang tải ảnh & top-K…" />}
          {detailError && <ErrorState message={detailError} />}
          {!detailLoading && !detailData && !detailError && (
            <EmptyState kind="none-selected" detail="Chọn một truy vấn ở cột bên trái để xem ảnh gốc, Ground Truth và bảng xếp hạng Top-K." />
          )}

          {detailData && (
            <>
              <Card
                title={
                  <span className="row">
                    <span className="mono">{detailData.query.query_id}</span>
                    <Badge tone="info">{detailData.query.category}</Badge>
                    {detailData.query.l1_interest && <Badge tone="neutral">{detailData.query.l1_interest}</Badge>}
                    {detailData.query.is_imperative && <Badge tone="neutral">Imperative</Badge>}
                    {detailData.query.is_question && <Badge tone="neutral">Question</Badge>}
                  </span>
                }
                actions={
                  <div className="row">
                    <label className="hint" htmlFor="bm-topk-sel">Hiển thị</label>
                    <select
                      id="bm-topk-sel"
                      value={topK}
                      onChange={(ev) => setTopK(Number(ev.target.value))}
                      style={{ minHeight: 34, padding: "4px 8px" }}
                    >
                      <option value={8}>Top-8</option>
                      <option value={12}>Top-12</option>
                      <option value={20}>Top-20</option>
                      <option value={50}>Top-50</option>
                    </select>
                  </div>
                }
              >
                <div className="bm-hero-query">
                  <button
                    type="button"
                    className="bm-ref-box"
                    onClick={() =>
                      setLightbox({
                        src: detailData.query.ref_image_url,
                        caption: `Reference · ${detailData.query.ref_image_id}`,
                      })
                    }
                  >
                    <img src={detailData.query.ref_image_url} alt="Reference" />
                    <span className="bm-ref-label">Ảnh gốc (Reference)</span>
                  </button>

                  <div className="bm-hero-meta">
                    <div className="bm-caption-box">
                      <span className="field-label">Câu lệnh chỉnh sửa (Modification Text)</span>
                      <p className="bm-caption-text">“{detailData.query.text}”</p>
                    </div>

                    <div className="bm-scorecards">
                      <div className="bm-scorecard">
                        <span className="field-label">
                          Chính: {detailData.primary.display} · {REGIME_LABELS[detailData.primary.regime]}
                        </span>
                        <div className="bm-scorecard-vals">
                          <b>
                            {dataset === "edir"
                              ? `Rank ${fmtRank(detailData.primary.metrics.best_gt_rank)}`
                              : `AP@10 ${fmtScore(dataset, detailData.primary.metrics.ap10)}`}
                          </b>
                          <Badge tone={detailData.primary.metrics.hit1 ? "ok" : detailData.primary.metrics.hit10 ? "warn" : "bad"}>
                            {detailData.primary.metrics.hit1 ? "Hit@1" : detailData.primary.metrics.hit10 ? "Hit@10" : "Miss@10"}
                          </Badge>
                          {detailData.primary.metrics.top1_is_ref && <Badge tone="warn">Top-1 = Ref Copy</Badge>}
                          {detailData.primary.metrics.top1_is_hneg && <Badge tone="bad">Top-1 = Hard Neg</Badge>}
                          {dataset === "pinpoint" && detailData.primary.metrics.neg_recall10 !== undefined && (
                            <span className="hint">NegRec@10: {detailData.primary.metrics.neg_recall10.toFixed(2)}</span>
                          )}
                        </div>
                      </div>

                      {detailData.compare && (
                        <div className="bm-scorecard bm-scorecard-cmp">
                          <span className="field-label">
                            Đối chiếu: {detailData.compare.display} · {REGIME_LABELS[detailData.compare.regime]}
                          </span>
                          <div className="bm-scorecard-vals">
                            <b>
                              {dataset === "edir"
                                ? `Rank ${fmtRank(detailData.compare.metrics.best_gt_rank)}`
                                : `AP@10 ${fmtScore(dataset, detailData.compare.metrics.ap10)}`}
                            </b>
                            <Badge tone={detailData.compare.metrics.hit1 ? "ok" : detailData.compare.metrics.hit10 ? "warn" : "bad"}>
                              {detailData.compare.metrics.hit1 ? "Hit@1" : detailData.compare.metrics.hit10 ? "Hit@10" : "Miss@10"}
                            </Badge>
                            {detailData.compare.metrics.top1_is_ref && <Badge tone="warn">Top-1 = Ref Copy</Badge>}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {/* Ground Truth + Hard Negatives strip */}
                <div className="bm-gt-section">
                  <div>
                    <h3 className="bm-subhead">
                      Đáp án đúng (Ground Truth · {detailData.ground_truths.length})
                    </h3>
                    <div className="bm-mini-strip">
                      {detailData.ground_truths.map((gt) => (
                        <button
                          key={gt.image_id}
                          type="button"
                          className="bm-mini-card bm-cand-gt"
                          onClick={() => setLightbox({ src: gt.image_url, caption: `Ground Truth · ${gt.image_id}` })}
                        >
                          <img src={gt.image_url} alt={gt.image_id} />
                          <div className="bm-mini-foot">
                            <Badge tone={gt.primary_rank && gt.primary_rank <= 10 ? "ok" : "bad"}>
                              {fmtRank(gt.primary_rank)}
                            </Badge>
                            {detailData.compare && (
                              <span className="hint">vs {fmtRank(gt.compare_rank)}</span>
                            )}
                          </div>
                        </button>
                      ))}
                    </div>
                  </div>

                  {detailData.hard_negatives.length > 0 && (
                    <div>
                      <h3 className="bm-subhead">
                        Bẫy khó (Hard Negatives · {detailData.hard_negatives.length})
                      </h3>
                      <div className="bm-mini-strip">
                        {detailData.hard_negatives.slice(0, 8).map((hn) => (
                          <button
                            key={hn.image_id}
                            type="button"
                            className="bm-mini-card bm-cand-hneg"
                            onClick={() => setLightbox({ src: hn.image_url, caption: `Hard Negative · ${hn.image_id}` })}
                          >
                            <img src={hn.image_url} alt={hn.image_id} />
                            <div className="bm-mini-foot">
                              <Badge tone={hn.primary_rank && hn.primary_rank <= 10 ? "bad" : "neutral"}>
                                {fmtRank(hn.primary_rank)}
                              </Badge>
                              {hn.is_ref && <span className="hint">Ref</span>}
                            </div>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </Card>

              {/* Top-K Retrieval Grids (Primary vs Optional Compare) */}
              <div className={detailData.compare ? "bm-compare-grids" : "stack"}>
                <Card
                  title={`Top-${detailData.primary.candidates.length} · ${detailData.primary.display} (${REGIME_LABELS[detailData.primary.regime]})`}
                  subtitle="Khung xanh = Đáp án đúng (GT) · Khung đỏ = Hard Negative · Khung vàng = Copy ảnh gốc (Ref Copy)"
                >
                  <div className="bm-cand-grid">
                    {detailData.primary.candidates.map((c) => (
                      <CandidateCard key={c.rank} item={c} onZoom={(src, caption) => setLightbox({ src, caption })} />
                    ))}
                  </div>
                </Card>

                {detailData.compare && (
                  <Card
                    title={`Top-${detailData.compare.candidates.length} · ${detailData.compare.display} (${REGIME_LABELS[detailData.compare.regime]})`}
                    subtitle="Đặt cạnh nhau để đối chiếu trực quan sự khác biệt giữa 2 mô hình hoặc In-Domain vs Transfer"
                  >
                    <div className="bm-cand-grid">
                      {detailData.compare.candidates.map((c) => (
                        <CandidateCard key={c.rank} item={c} onZoom={(src, caption) => setLightbox({ src, caption })} />
                      ))}
                    </div>
                  </Card>
                )}
              </div>

              {/* Cross-Model Consensus on this exact query */}
              <Card
                title={`Đồng thuận trên truy vấn này (${detailData.all_models.length} lượt đã hoàn tất)`}
                subtitle="Bấm vào bất kỳ model nào để chuyển làm mô hình chính — nhìn ngay lỗi hệ thống (toàn bộ 15 model cùng trượt) hay lỗi riêng lẻ"
              >
                <div className="bm-consensus-grid">
                  {detailData.all_models.map((m) => {
                    const isActive = m.model === model && m.regime === regime;
                    const tone = m.hit1 ? "ok" : m.hit10 ? "warn" : "bad";
                    return (
                      <button
                        key={`${m.model}:${m.regime}`}
                        type="button"
                        className={isActive ? "bm-cons-chip active" : "bm-cons-chip"}
                        onClick={() => {
                          setModel(m.model);
                          setRegime(m.regime);
                        }}
                      >
                        <span className="bm-cons-name">
                          <b>{m.display}</b>
                          <small>{m.regime === "indomain" ? "In-Dom" : m.regime === "cirr_transfer" ? "CIRR" : "FIQ"}</small>
                        </span>
                        <Badge tone={tone}>
                          {dataset === "edir" ? fmtRank(m.best_gt_rank) : fmtScore(dataset, m.primary_score)}
                        </Badge>
                        {m.top1_is_ref && <span className="bm-tag bm-tag-warn">Ref#1</span>}
                      </button>
                    );
                  })}
                </div>
              </Card>
            </>
          )}
        </div>
      </div>

      {/* Lightbox modal */}
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
