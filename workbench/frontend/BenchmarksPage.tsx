import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./client";
import { Badge, Card, DetailToggle, EmptyState, ErrorState, KeyValue, LoadingState, Progress } from "./ui";
import { BenchmarksInspector } from "./BenchmarksInspector";
import type {
  BenchmarkDataset,
  BenchmarkRegime,
  BenchmarksOverviewResponse,
  EdirModelRow,
  EdirSummaryResponse,
  InspectorSeed,
  ModelStatus,
  PinPointModelRow,
  PinPointSummaryResponse,
} from "./benchmarksTypes";

function fmtPct(v: number | null | undefined, digits = 2): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtDec(v: number | null | undefined, digits = 4): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

function fmtDeltaPct(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2)}%`;
}

function fmtDeltaDec(v: number | null | undefined): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return `${v >= 0 ? "+" : ""}${v.toFixed(4)}`;
}

function statusBadge(status: ModelStatus, variant?: string | null, reason?: string | null) {
  if (status === "ready") {
    return <Badge tone="ok" title={variant ?? "Đã hoàn tất"}>{variant ? `Xong · ${variant}` : "Hoàn tất"}</Badge>;
  }
  if (status === "unavailable") {
    return <Badge tone="neutral" title={reason ?? "Không phát hành"}>Không có ckpt</Badge>;
  }
  return <Badge tone="warn">Đang chạy / chờ</Badge>;
}

function heatBg(val: number | null | undefined, maxRef: number, isDelta = false): string {
  if (val === null || val === undefined || Number.isNaN(val)) return "var(--surface-2)";
  if (isDelta) {
    const t = Math.min(1, Math.abs(val) / Math.max(0.05, maxRef));
    const alpha = 0.08 + t * 0.34;
    return val >= 0 ? `rgba(8, 113, 90, ${alpha.toFixed(2)})` : `rgba(173, 48, 69, ${alpha.toFixed(2)})`;
  }
  const ratio = Math.max(0, Math.min(1, val / Math.max(1e-6, maxRef)));
  if (ratio < 0.28) {
    const a = 0.10 + (0.28 - ratio) * 0.7;
    return `rgba(173, 48, 69, ${Math.min(0.32, a).toFixed(2)})`;
  }
  if (ratio < 0.55) {
    return `rgba(196, 130, 14, ${(0.08 + (ratio - 0.28) * 0.35).toFixed(2)})`;
  }
  return `rgba(8, 113, 90, ${(0.10 + (ratio - 0.55) * 0.65).toFixed(2)})`;
}

function MetricBarCell({
  primary,
  secondary,
  ratio,
  highlight = false,
}: {
  primary: string;
  secondary?: string;
  ratio: number;
  highlight?: boolean;
}) {
  const pct = Math.max(0, Math.min(100, ratio * 100));
  return (
    <div className="bm-bar-cell">
      <div className="bm-bar-nums">
        <span className={highlight ? "cell-strong" : ""}>{primary}</span>
        {secondary && <span className="cell-sub">{secondary}</span>}
      </div>
      <div className="bm-microbar" aria-hidden="true">
        <span style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

export function BenchmarksPage({
  view,
  onChangeView,
  externalSeed,
}: {
  view: string;
  onChangeView: (nextView: string) => void;
  externalSeed?: InspectorSeed | null;
}) {
  const [overview, setOverview] = useState<BenchmarksOverviewResponse | null>(null);
  const [edir, setEdir] = useState<EdirSummaryResponse | null>(null);
  const [pinpoint, setPinpoint] = useState<PinPointSummaryResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>("");
  const [autoRefresh, setAutoRefresh] = useState<boolean>(false);
  const [inspectorSeed, setInspectorSeed] = useState<InspectorSeed | null>(externalSeed ?? null);

  useEffect(() => {
    if (externalSeed) setInspectorSeed({ ...externalSeed });
  }, [externalSeed]);

  const loadAll = useCallback((silent = false) => {
    const ctrl = new AbortController();
    if (!silent) setLoading(true);
    setError(null);
    Promise.all([
      api<BenchmarksOverviewResponse>("/api/benchmarks/overview", { signal: ctrl.signal }),
      api<EdirSummaryResponse>("/api/benchmarks/edir/summary", { signal: ctrl.signal }),
      api<PinPointSummaryResponse>("/api/benchmarks/pinpoint/summary", { signal: ctrl.signal }),
    ])
      .then(([ov, ed, pp]) => {
        setOverview(ov);
        setEdir(ed);
        setPinpoint(pp);
        setLastUpdated(new Date().toLocaleTimeString("vi-VN"));
        setLoading(false);
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === "AbortError") return;
        setError(err instanceof Error ? err.message : String(err));
        setLoading(false);
      });
    return () => ctrl.abort();
  }, []);

  useEffect(() => loadAll(false), [loadAll]);

  // Live auto-refresh every 20s when enabled (helpful while transfer suite is running)
  useEffect(() => {
    if (!autoRefresh) return;
    const id = window.setInterval(() => {
      loadAll(true);
    }, 20000);
    return () => window.clearInterval(id);
  }, [autoRefresh, loadAll]);

  const openInspector = useCallback(
    (seed: InspectorSeed) => {
      setInspectorSeed({ ...seed });
      onChangeView("bm_inspector");
    },
    [onChangeView],
  );

  const anyPending = useMemo(() => {
    if (!overview) return false;
    const ec = overview.edir.completion;
    const pc = overview.pinpoint.completion;
    return (
      ec.completed.indomain < ec.expected.indomain ||
      ec.completed.cirr_transfer < ec.expected.cirr_transfer ||
      ec.completed.fiq_transfer < ec.expected.fiq_transfer ||
      pc.completed.indomain < pc.expected.indomain ||
      pc.completed.fiq_transfer < pc.expected.fiq_transfer
    );
  }, [overview]);

  if (loading && !overview) {
    return <LoadingState label="Đang tải toàn bộ kết quả EDIR & PinPoint (In-Domain + Transfer)…" />;
  }
  if (error && !overview) {
    return <ErrorState message={error} onRetry={() => loadAll(false)} />;
  }

  return (
    <div className="stack">
      {/* Live status & refresh bar */}
      <div className="bm-status-strip">
        <div className="row">
          <Badge tone="ok">EDIR In-Domain {overview?.edir.completion.completed.indomain}/{overview?.edir.completion.expected.indomain}</Badge>
          <Badge tone="ok">PinPoint In-Domain {overview?.pinpoint.completion.completed.indomain}/{overview?.pinpoint.completion.expected.indomain}</Badge>
          <Badge tone={overview && overview.pinpoint.completion.completed.fiq_transfer >= overview.pinpoint.completion.expected.fiq_transfer ? "ok" : "warn"}>
            FIQ → PinPoint {overview?.pinpoint.completion.completed.fiq_transfer}/{overview?.pinpoint.completion.expected.fiq_transfer}
          </Badge>
          <Badge tone={overview && overview.edir.completion.completed.cirr_transfer >= overview.edir.completion.expected.cirr_transfer ? "ok" : "warn"}>
            CIRR → EDIR {overview?.edir.completion.completed.cirr_transfer}/{overview?.edir.completion.expected.cirr_transfer}
          </Badge>
          <Badge tone={overview && overview.edir.completion.completed.fiq_transfer >= overview.edir.completion.expected.fiq_transfer ? "ok" : "warn"}>
            FIQ → EDIR {overview?.edir.completion.completed.fiq_transfer}/{overview?.edir.completion.expected.fiq_transfer}
          </Badge>
        </div>
        <div className="row" style={{ marginLeft: "auto" }}>
          {lastUpdated && <span className="hint">Cập nhật: {lastUpdated}</span>}
          {anyPending && (
            <label className="bm-toggle-inline">
              <input
                type="checkbox"
                checked={autoRefresh}
                onChange={(ev) => setAutoRefresh(ev.target.checked)}
              />
              <span>Tự động làm mới (20s)</span>
            </label>
          )}
          <button type="button" className="secondary" style={{ minHeight: 34, padding: "4px 12px" }} onClick={() => loadAll(false)}>
            Làm mới
          </button>
        </div>
      </div>

      {view === "bm_overview" && overview && (
        <OverviewSection overview={overview} onChangeView={onChangeView} onOpenInspector={openInspector} />
      )}
      {view === "edir" && edir && (
        <EdirSection data={edir} onOpenInspector={openInspector} />
      )}
      {view === "pinpoint" && pinpoint && (
        <PinPointSection data={pinpoint} onOpenInspector={openInspector} />
      )}
      {view === "bm_inspector" && (
        <BenchmarksInspector edirSummary={edir} pinpointSummary={pinpoint} seed={inspectorSeed} />
      )}
    </div>
  );
}

/* ========================================================================= */
/* 1. OVERVIEW SECTION                                                       */
/* ========================================================================= */

function OverviewSection({
  overview,
  onChangeView,
  onOpenInspector,
}: {
  overview: BenchmarksOverviewResponse;
  onChangeView: (v: string) => void;
  onOpenInspector: (seed: InspectorSeed) => void;
}) {
  return (
    <div className="stack">
      <div className="bm-hero-grid">
        <Card
          title="EDIR · Image Editing Retrieval"
          subtitle={`Corpus chuẩn ${overview.edir.corpus_size.toLocaleString()} ảnh · ${overview.edir.test_query_count.toLocaleString()} test queries · 15 phép chỉnh sửa độc lập`}
          actions={
            <button type="button" className="primary" onClick={() => onChangeView("edir")}>
              Mở bảng 4 cột & Heatmap →
            </button>
          }
        >
          <div className="bm-podium">
            {overview.edir.top3.map((m, idx) => (
              <div key={m.key} className="bm-podium-item">
                <span className="bm-rank-medal">#{idx + 1}</span>
                <div>
                  <b>{m.display}</b>
                  <span className="bm-sub">{m.family}</span>
                </div>
                <div className="bm-podium-score">
                  <b>{fmtPct(m.recall1)}</b>
                  <span className="bm-sub">R@1 · MRR {fmtDec(m.mrr, 3)}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="bm-insight-box bm-insight-warn">
            <b>Phát hiện cốt lõi trên EDIR:</b> Toàn bộ <b>15/15</b> mô hình suy sụp trên phép chỉnh sửa{" "}
            <code>remove</code> (R@1 chỉ{" "}
            <b>
              {fmtPct(overview.edir.remove_r1_min, 1)} – {fmtPct(overview.edir.remove_r1_max, 1)}
            </b>
            , sát mức ngẫu nhiên 1/6 = 16.7%) dù Recall@10 đạt 95–98%.{" "}
            <button
              type="button"
              className="rowbtn"
              style={{ color: "var(--accent)", fontWeight: 600 }}
              onClick={() =>
                onOpenInspector({
                  dataset: "edir",
                  model: overview.edir.top3[0]?.key ?? "mtst_ptha",
                  regime: "indomain",
                  category: "remove",
                  outcome: "miss10",
                })
              }
            >
              Soi trực tiếp các ca lỗi remove →
            </button>
          </div>
          <div className="bm-progress-list">
            <div className="bm-prog-row">
              <span>In-Domain EDIR</span>
              <Progress value={overview.edir.completion.completed.indomain} max={overview.edir.completion.expected.indomain} label="EDIR In-Domain" />
              <span className="num">{overview.edir.completion.completed.indomain}/{overview.edir.completion.expected.indomain}</span>
            </div>
            <div className="bm-prog-row">
              <span>CIRR → EDIR Transfer</span>
              <Progress value={overview.edir.completion.completed.cirr_transfer} max={overview.edir.completion.expected.cirr_transfer} label="CIRR to EDIR" />
              <span className="num">{overview.edir.completion.completed.cirr_transfer}/{overview.edir.completion.expected.cirr_transfer}</span>
            </div>
            <div className="bm-prog-row">
              <span>FashionIQ → EDIR Transfer</span>
              <Progress value={overview.edir.completion.completed.fiq_transfer} max={overview.edir.completion.expected.fiq_transfer} label="FIQ to EDIR" />
              <span className="num">{overview.edir.completion.completed.fiq_transfer}/{overview.edir.completion.expected.fiq_transfer}</span>
            </div>
          </div>
        </Card>

        <Card
          title="PinPoint · Multi-Domain & Fashion CIR"
          subtitle={`Corpus chuẩn ${overview.pinpoint.corpus_size.toLocaleString()} ảnh · ${overview.pinpoint.test_query_count.toLocaleString()} test queries · 23 lĩnh vực & 5 loại truy vấn`}
          actions={
            <button type="button" className="primary" onClick={() => onChangeView("pinpoint")}>
              Mở bảng 3 cột & lát cắt Fashion →
            </button>
          }
        >
          <div className="bm-podium">
            {overview.pinpoint.top3.map((m, idx) => (
              <div key={m.key} className="bm-podium-item">
                <span className="bm-rank-medal">#{idx + 1}</span>
                <div>
                  <b>{m.display}</b>
                  <span className="bm-sub">{m.family}</span>
                </div>
                <div className="bm-podium-score">
                  <b>{fmtDec(m.map10)}</b>
                  <span className="bm-sub">mAP@10 · P@1 {fmtDec(m.p1, 3)}</span>
                </div>
              </div>
            ))}
          </div>
          <div className="bm-insight-box bm-insight-ok">
            <b>Phát hiện cốt lõi trên PinPoint:</b> Có{" "}
            <b>{overview.pinpoint.models_beating_gpt5}/15</b> mô hình In-Domain vượt baseline công bố{" "}
            <b>GPT-5 reranked ({overview.pinpoint.gpt5_reference_map10.toFixed(4)})</b> — dẫn đầu đều là các mô hình
            BLIP-2 chống nhiễu nhãn (<code>ConeSep</code>, <code>CSMCIR</code>, <code>INTENT</code>, <code>AirKnow</code>).{" "}
            <button
              type="button"
              className="rowbtn"
              style={{ color: "var(--accent)", fontWeight: 600 }}
              onClick={() =>
                onOpenInspector({
                  dataset: "pinpoint",
                  model: overview.pinpoint.top3[0]?.key ?? "conesep",
                  regime: "indomain",
                  slice: "fashion_swap",
                })
              }
            >
              Khám phá lát cắt PinPoint-Fashion →
            </button>
          </div>
          <div className="bm-progress-list">
            <div className="bm-prog-row">
              <span>In-Domain PinPoint</span>
              <Progress value={overview.pinpoint.completion.completed.indomain} max={overview.pinpoint.completion.expected.indomain} label="PinPoint In-Domain" />
              <span className="num">{overview.pinpoint.completion.completed.indomain}/{overview.pinpoint.completion.expected.indomain}</span>
            </div>
            <div className="bm-prog-row">
              <span>FashionIQ → PinPoint Transfer</span>
              <Progress value={overview.pinpoint.completion.completed.fiq_transfer} max={overview.pinpoint.completion.expected.fiq_transfer} label="FIQ to PinPoint" />
              <span className="num">{overview.pinpoint.completion.completed.fiq_transfer}/{overview.pinpoint.completion.expected.fiq_transfer}</span>
            </div>
          </div>
        </Card>
      </div>

      {/* Transparent 15-model availability & provenance matrix */}
      <Card
        title="Ma trận đầy đủ trạng thái & tính khả dụng checkpoint của 15 mô hình"
        subtitle="Minh bạch hoàn toàn: các ô 'Không có ckpt' là checkpoint chính chủ tác giả không phát hành (không giả lập, không chấm 0%); các biến thể noise (N0.2/N0.5) & 3-head ensemble được ghi rõ."
      >
        <div className="table-wrap">
          <table className="bm-compact-table">
            <thead>
              <tr>
                <th>Mô hình</th>
                <th>Họ / Backbone</th>
                <th>EDIR In-Domain</th>
                <th>PinPoint In-Domain</th>
                <th>FIQ → PinPoint</th>
                <th>CIRR → EDIR</th>
                <th>FIQ → EDIR</th>
              </tr>
            </thead>
            <tbody>
              {overview.availability.map((row) => (
                <tr key={row.key}>
                  <td>
                    <span className="cell-strong">{row.display}</span>
                    <span className="cell-sub mono">{row.key}</span>
                  </td>
                  <td>
                    <Badge tone="info">{row.family}</Badge>
                    <span className="cell-sub">{row.backbone}</span>
                  </td>
                  <td>{statusBadge(row.edir_indomain)}</td>
                  <td>{statusBadge(row.pinpoint_indomain)}</td>
                  <td>
                    {statusBadge(row.fiq_to_pinpoint, row.fiq_variant, row.fiq_reason)}
                    {row.fiq_reason && <span className="cell-sub">{row.fiq_reason}</span>}
                  </td>
                  <td>
                    {statusBadge(row.cirr_to_edir, row.cirr_variant, row.cirr_reason)}
                    {row.cirr_reason && <span className="cell-sub">{row.cirr_reason}</span>}
                  </td>
                  <td>{statusBadge(row.fiq_to_edir, row.fiq_variant, row.fiq_reason)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================= */
/* 2. EDIR 4-COLUMN + 15-EDIT HEATMAP SECTION                                */
/* ========================================================================= */

type EdirSortKey = "indomain_r1" | "cirr_r1" | "fiq_r1" | "stage0_r1" | "mrr" | "macro_r1" | "delta_cirr" | "display";

function EdirSection({
  data,
  onOpenInspector,
}: {
  data: EdirSummaryResponse;
  onOpenInspector: (seed: InspectorSeed) => void;
}) {
  const [family, setFamily] = useState<string>("all");
  const [search, setSearch] = useState<string>("");
  const [sortKey, setSortKey] = useState<EdirSortKey>("indomain_r1");
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [heatRegime, setHeatRegime] = useState<"indomain" | "cirr_transfer" | "fiq_transfer" | "delta_cirr">("indomain");
  const [heatMetric, setHeatMetric] = useState<"Recall@1" | "Recall@10" | "MRR">("Recall@1");

  const toggleSort = (k: EdirSortKey) => {
    if (sortKey === k) setSortAsc((p) => !p);
    else {
      setSortKey(k);
      setSortAsc(k === "display");
    }
  };

  const filteredModels = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = data.models.filter((m) => {
      if (family !== "all" && m.family !== family) return false;
      if (q && !`${m.display} ${m.key} ${m.backbone}`.toLowerCase().includes(q)) return false;
      return true;
    });
    const valOf = (m: EdirModelRow, k: EdirSortKey): number | string => {
      switch (k) {
        case "indomain_r1": return m.indomain.overall?.["Recall@1"] ?? -1;
        case "cirr_r1": return m.cirr_transfer.overall?.["Recall@1"] ?? -1;
        case "fiq_r1": return m.fiq_transfer.overall?.["Recall@1"] ?? -1;
        case "stage0_r1": return m.stage0?.["Recall@1"] ?? -1;
        case "mrr": return m.indomain.overall?.MRR ?? -1;
        case "macro_r1": return m.indomain.macro_subcategory?.["Recall@1"] ?? -1;
        case "delta_cirr": return m.deltas.indomain_minus_cirr_r1 ?? -9;
        case "display": return m.display.toLowerCase();
      }
    };
    return [...list].sort((a, b) => {
      const va = valOf(a, sortKey);
      const vb = valOf(b, sortKey);
      const cmp = typeof va === "string" && typeof vb === "string" ? va.localeCompare(vb) : Number(va) - Number(vb);
      return sortAsc ? cmp : -cmp;
    });
  }, [data.models, family, search, sortKey, sortAsc]);

  const expandedModel = useMemo(
    () => data.models.find((m) => m.key === expandedKey) ?? null,
    [data.models, expandedKey],
  );

  const maxR1 = useMemo(
    () => Math.max(0.4, ...data.models.map((m) => m.indomain.overall?.["Recall@1"] ?? 0)),
    [data.models],
  );

  const sortBtn = (k: EdirSortKey, label: string, unit?: string) => (
    <button
      type="button"
      className={sortKey === k ? "sort-head active" : "sort-head"}
      onClick={() => toggleSort(k)}
    >
      <span>{label}</span>
      {unit && <span className="unit">{unit}</span>}
      <span className="sort-arrow">{sortKey === k ? (sortAsc ? "▲" : "▼") : "⇅"}</span>
    </button>
  );

  return (
    <div className="stack">
      <Card
        title="Bảng đối chiếu 4 cột trên EDIR (Test 1,003 queries · Corpus 28,645 ảnh)"
        subtitle="So sánh trực tiếp Base Zero-Shot → FashionIQ Transfer → CIRR Transfer → In-Domain EDIR trên cùng một split chuẩn 70_10_20_seed42"
        actions={
          <a className="bm-btn-link" href="/api/benchmarks/edir/export.csv" download>
            Tải CSV
          </a>
        }
      >
        <div className="toolbar" style={{ marginBottom: 12 }}>
          <div className="bm-seg" role="group" aria-label="Lọc họ mô hình">
            {["all", "BLIP-2", "CLIP", "Classic"].map((fam) => (
              <button
                key={fam}
                type="button"
                className={family === fam ? "bm-seg-btn active" : "bm-seg-btn"}
                onClick={() => setFamily(fam)}
              >
                {fam === "all" ? `Tất cả (${data.models.length})` : fam}
              </button>
            ))}
          </div>
          <input
            type="search"
            placeholder="Lọc theo tên model hoặc backbone…"
            value={search}
            onChange={(ev) => setSearch(ev.target.value)}
            style={{ minWidth: 240, minHeight: 38 }}
          />
        </div>

        <div className="table-wrap">
          <table className="bm-compact-table">
            <thead>
              <tr>
                <th>#</th>
                <th>{sortBtn("display", "Mô hình")}</th>
                <th className="num">{sortBtn("stage0_r1", "Base ZS", "R@1")}</th>
                <th className="num">{sortBtn("fiq_r1", "FIQ Transfer", "R@1")}</th>
                <th className="num">{sortBtn("cirr_r1", "CIRR Transfer", "R@1")}</th>
                <th className="num">{sortBtn("indomain_r1", "In-Domain EDIR", "R@1")}</th>
                <th className="num">{sortBtn("mrr", "MRR")}</th>
                <th className="num">{sortBtn("macro_r1", "Macro-15")}</th>
                <th className="num">{sortBtn("delta_cirr", "Δ In−CIRR")}</th>
                <th>Thao tác</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m, idx) => {
                const inR1 = m.indomain.overall?.["Recall@1"];
                const inR10 = m.indomain.overall?.["Recall@10"];
                const cirrR1 = m.cirr_transfer.overall?.["Recall@1"];
                const cirrR10 = m.cirr_transfer.overall?.["Recall@10"];
                const fiqR1 = m.fiq_transfer.overall?.["Recall@1"];
                const fiqR10 = m.fiq_transfer.overall?.["Recall@10"];
                const delta = m.deltas.indomain_minus_cirr_r1;
                const isExp = expandedKey === m.key;
                const epTxt = m.indomain.best_epoch !== null && m.indomain.best_epoch !== undefined ? ` · ep${m.indomain.best_epoch}` : "";

                return (
                  <tr key={m.key} data-expanded={isExp ? "true" : undefined}>
                    <td className="num hint">{idx + 1}</td>
                    <td>
                      <span className="cell-strong">{m.display}</span>
                      <span className="cell-sub">{m.family}</span>
                    </td>
                    <td className="num">{fmtPct(m.stage0?.["Recall@1"])}</td>
                    <td className="num">
                      {m.fiq_transfer.status === "ready" ? (
                        <MetricBarCell
                          primary={fmtPct(fiqR1)}
                          secondary={`R@10 ${fmtPct(fiqR10, 1)}`}
                          ratio={(fiqR1 ?? 0) / maxR1}
                        />
                      ) : (
                        statusBadge(m.fiq_transfer.status, m.fiq_transfer.variant, m.fiq_transfer.unavailable_reason)
                      )}
                    </td>
                    <td className="num">
                      {m.cirr_transfer.status === "ready" ? (
                        <MetricBarCell
                          primary={fmtPct(cirrR1)}
                          secondary={`R@10 ${fmtPct(cirrR10, 1)}`}
                          ratio={(cirrR1 ?? 0) / maxR1}
                        />
                      ) : (
                        statusBadge(m.cirr_transfer.status, m.cirr_transfer.variant, m.cirr_transfer.unavailable_reason)
                      )}
                    </td>
                    <td className="num">
                      {m.indomain.status === "ready" ? (
                        <MetricBarCell
                          primary={fmtPct(inR1)}
                          secondary={`R@10 ${fmtPct(inR10, 1)}${epTxt}`}
                          ratio={(inR1 ?? 0) / maxR1}
                          highlight
                        />
                      ) : (
                        statusBadge(m.indomain.status)
                      )}
                    </td>
                    <td className="num">{fmtDec(m.indomain.overall?.MRR)}</td>
                    <td className="num">{fmtPct(m.indomain.macro_subcategory?.["Recall@1"])}</td>
                    <td className="num">
                      {delta !== null ? (
                        <Badge tone={delta >= 0 ? "ok" : "bad"}>{fmtDeltaPct(delta)}</Badge>
                      ) : (
                        <span className="hint">—</span>
                      )}
                    </td>
                    <td>
                      <div className="row" style={{ flexWrap: "nowrap" }}>
                        <DetailToggle
                          open={isExp}
                          onToggle={() => setExpandedKey(isExp ? null : m.key)}
                        />
                        {m.indomain.has_predictions && (
                          <button
                            type="button"
                            className="rowbtn"
                            style={{ color: "var(--accent)", fontWeight: 600 }}
                            onClick={() =>
                              onOpenInspector({
                                dataset: "edir",
                                model: m.key,
                                regime: "indomain",
                                compareModel: m.cirr_transfer.has_predictions ? m.key : undefined,
                                compareRegime: m.cirr_transfer.has_predictions ? "cirr_transfer" : undefined,
                              })
                            }
                          >
                            Soi ảnh →
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {expandedModel && (
          <div className="row-detail">
            <div className="row-detail-head">
              <b>Chi tiết đầy đủ: {expandedModel.display}</b>
              <Badge tone="info">{expandedModel.backbone}</Badge>
              <button type="button" className="ghost" onClick={() => setExpandedKey(null)}>
                Đóng
              </button>
            </div>
            <KeyValue
              rows={[
                ["Run ID In-Domain", <code key="r">{expandedModel.run_id}</code>],
                [
                  "In-Domain R@1 / R@5 / R@10 / R@50 / MRR",
                  `${fmtPct(expandedModel.indomain.overall?.["Recall@1"])} / ${fmtPct(expandedModel.indomain.overall?.["Recall@5"])} / ${fmtPct(expandedModel.indomain.overall?.["Recall@10"])} / ${fmtPct(expandedModel.indomain.overall?.["Recall@50"])} / ${fmtDec(expandedModel.indomain.overall?.MRR)}`,
                ],
                [
                  "CIRR Transfer",
                  expandedModel.cirr_transfer.status === "ready"
                    ? `${expandedModel.cirr_transfer.variant} · R@1=${fmtPct(expandedModel.cirr_transfer.overall?.["Recall@1"])}, R@10=${fmtPct(expandedModel.cirr_transfer.overall?.["Recall@10"])}, MRR=${fmtDec(expandedModel.cirr_transfer.overall?.MRR)}`
                    : expandedModel.cirr_transfer.unavailable_reason ?? "Đang chờ chạy",
                ],
                [
                  "FashionIQ Transfer",
                  expandedModel.fiq_transfer.status === "ready"
                    ? `${expandedModel.fiq_transfer.variant} · R@1=${fmtPct(expandedModel.fiq_transfer.overall?.["Recall@1"])}, R@10=${fmtPct(expandedModel.fiq_transfer.overall?.["Recall@10"])}, MRR=${fmtDec(expandedModel.fiq_transfer.overall?.MRR)}`
                    : expandedModel.fiq_transfer.unavailable_reason ?? "Đang chờ chạy",
                ],
              ]}
            />
          </div>
        )}
      </Card>

      {/* Interactive 15-Edit-Operation Heatmap */}
      <Card
        title="Bản đồ nhiệt 15 phép chỉnh sửa trên EDIR (bấm vào bất kỳ ô nào để soi ảnh lỗi)"
        subtitle="Màu đỏ nhấn mạnh điểm gãy (điển hình cột remove ở ~10–21%); bấm ô bất kỳ mở thẳng Query Inspector cho đúng (model, phép chỉnh sửa)"
        actions={
          <div className="row">
            <div className="bm-seg">
              {(
                [
                  ["indomain", "In-Domain"],
                  ["cirr_transfer", "CIRR Transfer"],
                  ["fiq_transfer", "FIQ Transfer"],
                  ["delta_cirr", "Δ InDom−CIRR"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={heatRegime === id ? "bm-seg-btn active" : "bm-seg-btn"}
                  onClick={() => setHeatRegime(id)}
                >
                  {label}
                </button>
              ))}
            </div>
            <select
              aria-label="Chọn chỉ số heatmap"
              value={heatMetric}
              onChange={(ev) => setHeatMetric(ev.target.value as "Recall@1" | "Recall@10" | "MRR")}
              style={{ minHeight: 36, padding: "4px 10px" }}
            >
              <option value="Recall@1">Recall@1</option>
              <option value="Recall@10">Recall@10</option>
              <option value="MRR">MRR</option>
            </select>
          </div>
        }
      >
        <div className="table-wrap">
          <table className="bm-heatmap">
            <thead>
              <tr>
                <th className="bm-sticky-col">Mô hình</th>
                {data.subcategories.map((sc) => (
                  <th key={sc.id} className="num" title={`${sc.id} (${sc.count} queries)`}>
                    <div>{sc.id}</div>
                    <div className="unit" style={{ fontSize: ".7rem", fontWeight: 400 }}>n={sc.count}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m) => {
                const block =
                  heatRegime === "indomain"
                    ? m.indomain
                    : heatRegime === "cirr_transfer"
                      ? m.cirr_transfer
                      : heatRegime === "fiq_transfer"
                        ? m.fiq_transfer
                        : m.indomain;
                const cirrBlock = m.cirr_transfer;
                const isDelta = heatRegime === "delta_cirr";
                const canDrill =
                  !isDelta && block.status === "ready" && block.has_predictions;

                return (
                  <tr key={m.key}>
                    <td className="bm-sticky-col">
                      <span className="cell-strong">{m.display}</span>
                    </td>
                    {data.subcategories.map((sc) => {
                      let val: number | null = null;
                      if (isDelta) {
                        const a = m.indomain.per_subcategory?.[sc.id]?.[heatMetric];
                        const b = cirrBlock.per_subcategory?.[sc.id]?.[heatMetric];
                        if (a !== undefined && b !== undefined) val = a - b;
                      } else {
                        const v = block.per_subcategory?.[sc.id]?.[heatMetric];
                        if (v !== undefined) val = v;
                      }
                      const maxRef = heatMetric === "Recall@10" ? 1.0 : 0.65;
                      const txt =
                        val === null
                          ? "—"
                          : heatMetric === "MRR"
                            ? isDelta
                              ? `${val >= 0 ? "+" : ""}${val.toFixed(2)}`
                              : val.toFixed(2)
                            : isDelta
                              ? `${val >= 0 ? "+" : ""}${(val * 100).toFixed(0)}%`
                              : `${(val * 100).toFixed(0)}%`;
                      return (
                        <td
                          key={sc.id}
                          className="num bm-heat-cell"
                          style={{ background: heatBg(val, isDelta ? 0.3 : maxRef, isDelta) }}
                        >
                          {canDrill && val !== null ? (
                            <button
                              type="button"
                              className="bm-heat-btn"
                              title={`Mở Query Inspector: ${m.display} · ${sc.id}`}
                              onClick={() =>
                                onOpenInspector({
                                  dataset: "edir",
                                  model: m.key,
                                  regime: heatRegime as BenchmarkRegime,
                                  category: sc.id,
                                })
                              }
                            >
                              {txt}
                            </button>
                          ) : (
                            <span>{txt}</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}

/* ========================================================================= */
/* 3. PINPOINT 3-COLUMN + 8 FASHION SLICES + 5-CATEGORY HEATMAP SECTION      */
/* ========================================================================= */

type PinPointSortKey = "indomain_map10" | "fiq_map10" | "stage0_map10" | "delta" | "p1" | "neg_rec10" | "no_neg" | "display";

function PinPointSection({
  data,
  onOpenInspector,
}: {
  data: PinPointSummaryResponse;
  onOpenInspector: (seed: InspectorSeed) => void;
}) {
  const [slice, setSlice] = useState<string>("all");
  const [family, setFamily] = useState<string>("all");
  const [search, setSearch] = useState<string>("");
  const [sortKey, setSortKey] = useState<PinPointSortKey>("indomain_map10");
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [heatRegime, setHeatRegime] = useState<"indomain" | "fiq_transfer" | "delta">("indomain");

  const activeSliceMeta = useMemo(
    () => data.slices.find((s) => s.id === slice) ?? data.slices[0],
    [data.slices, slice],
  );

  const toggleSort = (k: PinPointSortKey) => {
    if (sortKey === k) setSortAsc((p) => !p);
    else {
      setSortKey(k);
      setSortAsc(k === "display");
    }
  };

  const filteredModels = useMemo(() => {
    const q = search.trim().toLowerCase();
    const list = data.models.filter((m) => {
      if (family !== "all" && m.family !== family) return false;
      if (q && !`${m.display} ${m.key} ${m.backbone}`.toLowerCase().includes(q)) return false;
      return true;
    });
    const valOf = (m: PinPointModelRow, k: PinPointSortKey): number | string => {
      const ind = m.indomain.slices[slice];
      const fiq = m.fiq_transfer.slices[slice];
      switch (k) {
        case "indomain_map10": return ind?.["mAP@10"] ?? -1;
        case "fiq_map10": return fiq?.["mAP@10"] ?? -1;
        case "stage0_map10": return m.stage0?.["mAP@10"] ?? -1;
        case "delta": return m.slice_deltas[slice] ?? -9;
        case "p1": return ind?.["precision@1"] ?? -1;
        case "neg_rec10": return ind?.["NegRecall@10"] ?? 99;
        case "no_neg": return ind?.["mAP@10_noNeg"] ?? -1;
        case "display": return m.display.toLowerCase();
      }
    };
    return [...list].sort((a, b) => {
      const va = valOf(a, sortKey);
      const vb = valOf(b, sortKey);
      const cmp = typeof va === "string" && typeof vb === "string" ? va.localeCompare(vb) : Number(va) - Number(vb);
      return sortAsc ? cmp : -cmp;
    });
  }, [data.models, slice, family, search, sortKey, sortAsc]);

  const maxMap10 = useMemo(() => {
    const vals = data.models.flatMap((m) => [
      m.indomain.slices[slice]?.["mAP@10"] ?? 0,
      m.fiq_transfer.slices[slice]?.["mAP@10"] ?? 0,
    ]);
    return Math.max(0.22, ...vals);
  }, [data.models, slice]);

  const expandedModel = useMemo(
    () => data.models.find((m) => m.key === expandedKey) ?? null,
    [data.models, expandedKey],
  );

  const sortBtn = (k: PinPointSortKey, label: string, unit?: string) => (
    <button
      type="button"
      className={sortKey === k ? "sort-head active" : "sort-head"}
      onClick={() => toggleSort(k)}
    >
      <span>{label}</span>
      {unit && <span className="unit">{unit}</span>}
      <span className="sort-arrow">{sortKey === k ? (sortAsc ? "▲" : "▼") : "⇅"}</span>
    </button>
  );

  return (
    <div className="stack">
      {/* 8-slice interactive selector */}
      <Card
        title="Chọn lát cắt đánh giá PinPoint (cùng corpus 109,598 ảnh · 0 lệch protocol)"
        subtitle={activeSliceMeta ? `${activeSliceMeta.label} (${activeSliceMeta.count.toLocaleString()} queries): ${activeSliceMeta.description}` : ""}
        actions={
          <a className="bm-btn-link" href={`/api/benchmarks/pinpoint/export.csv?slice=${encodeURIComponent(slice)}`} download>
            Tải CSV ({slice})
          </a>
        }
      >
        <div className="bm-slice-grid" role="tablist" aria-label="Lát cắt PinPoint">
          {data.slices.map((s) => (
            <button
              key={s.id}
              type="button"
              role="tab"
              aria-selected={slice === s.id}
              className={slice === s.id ? "bm-slice-pill active" : "bm-slice-pill"}
              onClick={() => setSlice(s.id)}
              title={s.description}
            >
              <span className="bm-slice-name">{s.label}</span>
              <span className="bm-slice-cnt">{s.count.toLocaleString()}</span>
            </button>
          ))}
        </div>

        <div className="toolbar" style={{ marginTop: 12, marginBottom: 12 }}>
          <div className="bm-seg" role="group" aria-label="Lọc họ mô hình">
            {["all", "BLIP-2", "CLIP", "Classic"].map((fam) => (
              <button
                key={fam}
                type="button"
                className={family === fam ? "bm-seg-btn active" : "bm-seg-btn"}
                onClick={() => setFamily(fam)}
              >
                {fam === "all" ? `Tất cả (${data.models.length})` : fam}
              </button>
            ))}
          </div>
          <input
            type="search"
            placeholder="Lọc theo tên model hoặc backbone…"
            value={search}
            onChange={(ev) => setSearch(ev.target.value)}
            style={{ minWidth: 240, minHeight: 38 }}
          />
          {slice === "all" && (
            <div className="row" style={{ marginLeft: "auto" }}>
              {data.published_baselines.map((b) => (
                <Badge key={b.label} tone="info">
                  {b.label}: {b.map10.toFixed(4)}
                </Badge>
              ))}
            </div>
          )}
        </div>

        <div className="table-wrap">
          <table className="bm-compact-table">
            <thead>
              <tr>
                <th>#</th>
                <th>{sortBtn("display", "Mô hình")}</th>
                {slice === "all" && <th className="num">{sortBtn("stage0_map10", "Base ZS")}</th>}
                <th className="num">{sortBtn("fiq_map10", "FIQ Transfer")}</th>
                <th className="num">{sortBtn("indomain_map10", "In-Domain")}</th>
                <th className="num">{sortBtn("delta", "Δ In−FIQ")}</th>
                <th className="num">{sortBtn("p1", "P@1")}</th>
                <th className="num">{sortBtn("neg_rec10", "NegRec@10")}</th>
                <th className="num">{sortBtn("no_neg", "noNeg")}</th>
                <th>Thao tác</th>
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m, idx) => {
                const indS = m.indomain.slices[slice];
                const fiqS = m.fiq_transfer.slices[slice];
                const delta = m.slice_deltas[slice];
                const beatsGpt5 = slice === "all" && (indS?.["mAP@10"] ?? 0) > 0.1837;
                const isExp = expandedKey === m.key;
                const epTxt = m.indomain.best_epoch !== null && m.indomain.best_epoch !== undefined ? `ep${m.indomain.best_epoch} · ` : "";

                return (
                  <tr key={m.key} data-expanded={isExp ? "true" : undefined}>
                    <td className="num hint">{idx + 1}</td>
                    <td>
                      <div className="row" style={{ gap: 5 }}>
                        <span className="cell-strong">{m.display}</span>
                        {beatsGpt5 && <Badge tone="ok">&gt;GPT-5</Badge>}
                      </div>
                      <span className="cell-sub">{m.family}</span>
                    </td>
                    {slice === "all" && <td className="num">{fmtDec(m.stage0?.["mAP@10"])}</td>}
                    <td className="num">
                      {m.fiq_transfer.status === "ready" && fiqS ? (
                        <MetricBarCell
                          primary={fmtDec(fiqS["mAP@10"])}
                          secondary={`${m.fiq_transfer.variant ?? "fiq"} · NR ${fmtDec(fiqS["NegRecall@10"], 3)}`}
                          ratio={fiqS["mAP@10"] / maxMap10}
                        />
                      ) : (
                        statusBadge(m.fiq_transfer.status, m.fiq_transfer.variant, m.fiq_transfer.unavailable_reason)
                      )}
                    </td>
                    <td className="num">
                      {m.indomain.status === "ready" && indS ? (
                        <MetricBarCell
                          primary={fmtDec(indS["mAP@10"])}
                          secondary={`${epTxt}P@10 ${fmtDec(indS["precision@10"], 3)}`}
                          ratio={indS["mAP@10"] / maxMap10}
                          highlight
                        />
                      ) : (
                        statusBadge(m.indomain.status)
                      )}
                    </td>
                    <td className="num">
                      {delta !== null && delta !== undefined ? (
                        <Badge tone={delta >= 0 ? "ok" : "bad"}>{fmtDeltaDec(delta)}</Badge>
                      ) : (
                        <span className="hint">—</span>
                      )}
                    </td>
                    <td className="num">{fmtDec(indS?.["precision@1"])}</td>
                    <td className="num">{fmtDec(indS?.["NegRecall@10"])}</td>
                    <td className="num">{fmtDec(indS?.["mAP@10_noNeg"])}</td>
                    <td>
                      <div className="row" style={{ flexWrap: "nowrap" }}>
                        <DetailToggle
                          open={isExp}
                          onToggle={() => setExpandedKey(isExp ? null : m.key)}
                        />
                        {m.indomain.has_predictions && (
                          <button
                            type="button"
                            className="rowbtn"
                            style={{ color: "var(--accent)", fontWeight: 600 }}
                            onClick={() =>
                              onOpenInspector({
                                dataset: "pinpoint",
                                model: m.key,
                                regime: "indomain",
                                slice,
                                compareModel: m.fiq_transfer.has_predictions ? m.key : undefined,
                                compareRegime: m.fiq_transfer.has_predictions ? "fiq_transfer" : undefined,
                              })
                            }
                          >
                            Soi ảnh →
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Full-width 8-slice breakdown for expanded model */}
        {expandedModel && (
          <div className="row-detail">
            <div className="row-detail-head">
              <b>Phân rã đầy đủ 8 lát cắt: {expandedModel.display}</b>
              <Badge tone="info">{expandedModel.backbone}</Badge>
              <button type="button" className="ghost" onClick={() => setExpandedKey(null)}>
                Đóng
              </button>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Lát cắt</th>
                    <th className="num">Queries</th>
                    <th className="num">FIQ Transfer mAP@10</th>
                    <th className="num">In-Domain mAP@10</th>
                    <th className="num">Δ (InDom − FIQ)</th>
                    <th className="num">In-Domain P@1</th>
                    <th className="num">In-Domain NegRec@10</th>
                  </tr>
                </thead>
                <tbody>
                  {data.slices.map((s) => {
                    const ind = expandedModel.indomain.slices[s.id];
                    const fiq = expandedModel.fiq_transfer.slices[s.id];
                    const d = expandedModel.slice_deltas[s.id];
                    return (
                      <tr key={s.id}>
                        <td><span className="cell-strong">{s.label}</span></td>
                        <td className="num">{s.count}</td>
                        <td className="num">{fmtDec(fiq?.["mAP@10"])}</td>
                        <td className="num"><b>{fmtDec(ind?.["mAP@10"])}</b></td>
                        <td className="num">
                          {d !== null && d !== undefined ? (
                            <Badge tone={d >= 0 ? "ok" : "bad"}>{fmtDeltaDec(d)}</Badge>
                          ) : "—"}
                        </td>
                        <td className="num">{fmtDec(ind?.["precision@1"])}</td>
                        <td className="num">{fmtDec(ind?.["NegRecall@10"])}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </Card>

      {/* 5-Category PinPoint Heatmap */}
      <Card
        title="Ma trận mAP@10 theo 5 nhóm truy vấn PinPoint (bấm vào bất kỳ ô nào để soi ảnh)"
        subtitle="Cho thấy rõ Swap & Context Fit mạnh nhất, trong khi Complement & Explore là điểm nghẽn chung của toàn bộ 15 mô hình"
        actions={
          <div className="bm-seg">
            {(
              [
                ["indomain", "In-Domain (15/15)"],
                ["fiq_transfer", "FIQ Transfer"],
                ["delta", "Δ InDom−FIQ"],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                type="button"
                className={heatRegime === id ? "bm-seg-btn active" : "bm-seg-btn"}
                onClick={() => setHeatRegime(id)}
              >
                {label}
              </button>
            ))}
          </div>
        }
      >
        <div className="table-wrap">
          <table className="bm-heatmap">
            <thead>
              <tr>
                <th className="bm-sticky-col">Mô hình</th>
                {data.categories.map((c) => (
                  <th key={c.id} className="num">
                    <div>{c.id}</div>
                    <div className="unit" style={{ fontSize: ".7rem", fontWeight: 400 }}>n={c.count}</div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredModels.map((m) => {
                const isDelta = heatRegime === "delta";
                const block = heatRegime === "fiq_transfer" ? m.fiq_transfer : m.indomain;
                const canDrill = !isDelta && block.status === "ready" && block.has_predictions;
                return (
                  <tr key={m.key}>
                    <td className="bm-sticky-col">
                      <span className="cell-strong">{m.display}</span>
                    </td>
                    {data.categories.map((c) => {
                      let val: number | null = null;
                      if (isDelta) {
                        const a = m.indomain.per_category?.[c.id]?.["mAP@10"];
                        const b = m.fiq_transfer.per_category?.[c.id]?.["mAP@10"];
                        if (a !== undefined && b !== undefined) val = a - b;
                      } else {
                        const v = block.per_category?.[c.id]?.["mAP@10"];
                        if (v !== undefined) val = v;
                      }
                      const txt =
                        val === null ? "—" : isDelta ? fmtDeltaDec(val) : val.toFixed(4);
                      return (
                        <td
                          key={c.id}
                          className="num bm-heat-cell"
                          style={{ background: heatBg(val, isDelta ? 0.15 : 0.25, isDelta) }}
                        >
                          {canDrill && val !== null ? (
                            <button
                              type="button"
                              className="bm-heat-btn"
                              title={`Mở Query Inspector: ${m.display} · ${c.id}`}
                              onClick={() =>
                                onOpenInspector({
                                  dataset: "pinpoint",
                                  model: m.key,
                                  regime: heatRegime as BenchmarkRegime,
                                  category: c.id,
                                })
                              }
                            >
                              {txt}
                            </button>
                          ) : (
                            <span>{txt}</span>
                          )}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  );
}
