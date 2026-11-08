import { useEffect, useMemo, useState } from "react";
import { api, BASE } from "./client";
import type { CompositionQuery, CompositionRun } from "./types";

const SUBSET_INFO: Record<string, { label: string; queries: number; gallery: number }> = {
  full: { label: "Full", queries: 6003, gallery: 74381 },
  sf: { label: "Shortcut-Free (SF)", queries: 4069, gallery: 74381 },
  validated: { label: "Validated", queries: 586, gallery: 74381 },
};

function formatNdcg(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return val.toFixed(4);
}

function formatPercent(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return `${(val > 0 ? "+" : "") + (val * 100).toFixed(2)}%`;
}

function formatRecall(val: number | null | undefined): string {
  if (val === null || val === undefined || isNaN(val)) return "—";
  return `${val.toFixed(1)}%`;
}

export function CompositionView({ onError }: { onError: (msg: string) => void }) {
  // Sync state with URL params
  const initialParams = useMemo(() => new URLSearchParams(window.location.search), []);
  const [subset, setSubset] = useState<"full" | "sf" | "validated">(
    (initialParams.get("comp_subset") as "full" | "sf" | "validated") || "full"
  );
  const [runs, setRuns] = useState<CompositionRun[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedRunId, setSelectedRunId] = useState<string>(initialParams.get("comp_run") || "");

  // Query drilldown filters
  const [condition, setCondition] = useState<string>(initialParams.get("comp_cond") || "multimodal");
  const [category, setCategory] = useState<string>(initialParams.get("comp_cat") || "");
  const [outcome, setOutcome] = useState<string>(initialParams.get("comp_outcome") || "");
  const [order, setOrder] = useState<string>(initialParams.get("comp_order") || "worst");
  const [querySearch, setQuerySearch] = useState<string>(initialParams.get("comp_qid") || "");
  const [offset, setOffset] = useState<number>(parseInt(initialParams.get("comp_offset") || "0", 10) || 0);

  const [queries, setQueries] = useState<CompositionQuery[]>([]);
  const [totalQueries, setTotalQueries] = useState<number>(0);
  const [queriesUnavailableReason, setQueriesUnavailableReason] = useState<string | null>(null);
  const [loadingQueries, setLoadingQueries] = useState<boolean>(false);

  // Sync state back to URL query parameters
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    params.set("comp_subset", subset);
    if (selectedRunId) params.set("comp_run", selectedRunId);
    else params.delete("comp_run");
    if (condition) params.set("comp_cond", condition);
    if (category) params.set("comp_cat", category);
    else params.delete("comp_cat");
    if (outcome) params.set("comp_outcome", outcome);
    else params.delete("comp_outcome");
    if (order) params.set("comp_order", order);
    if (querySearch) params.set("comp_qid", querySearch);
    else params.delete("comp_qid");
    if (offset > 0) params.set("comp_offset", String(offset));
    else params.delete("comp_offset");

    const newUrl = `${window.location.pathname}?${params.toString()}`;
    window.history.replaceState({}, "", newUrl);
  }, [subset, selectedRunId, condition, category, outcome, order, querySearch, offset]);

  // Load composition runs for the active subset
  useEffect(() => {
    setLoading(true);
    api<CompositionRun[]>(`/api/composition/runs?subset=${subset}`)
      .then((data) => {
        const safeRuns = Array.isArray(data) ? data : [];
        setRuns(safeRuns);
        if (safeRuns.length > 0 && !selectedRunId) {
          setSelectedRunId(safeRuns[0].run_id);
        } else if (safeRuns.length > 0 && !safeRuns.some((r) => r.run_id === selectedRunId)) {
          setSelectedRunId(safeRuns[0].run_id);
        }
      })
      .catch((err) => onError(`Composition runs: ${err.message}`))
      .finally(() => setLoading(false));
  }, [subset]);

  // Load paginated queries when a run is selected
  useEffect(() => {
    if (!selectedRunId) {
      setQueries([]);
      setTotalQueries(0);
      setQueriesUnavailableReason(null);
      return;
    }
    setLoadingQueries(true);
    const p = new URLSearchParams({
      subset,
      condition,
      order,
      limit: "20",
      offset: String(offset),
    });
    if (category) p.set("category", category);
    if (outcome) p.set("outcome", outcome);
    if (querySearch) p.set("query_id", querySearch);

    api<{ items: CompositionQuery[]; total: number; queries_unavailable_reason?: string }>(
      `/api/composition/runs/${encodeURIComponent(selectedRunId)}/queries?${p.toString()}`
    )
      .then((data) => {
        setQueries(data.items || []);
        setTotalQueries(data.total || 0);
        setQueriesUnavailableReason(data.queries_unavailable_reason || null);
      })
      .catch((err) => onError(`Composition queries: ${err.message}`))
      .finally(() => setLoadingQueries(false));
  }, [selectedRunId, subset, condition, category, outcome, order, querySearch, offset]);

  const activeRun = runs.find((r) => r.run_id === selectedRunId);

  return (
    <section className="composition-view">
      <div className="composition-header">
        <div>
          <h2>CIRCUS Composition Evaluation</h2>
          <p className="subtitle">
            Input-ablation dependence on CIRCUS FashionIQ release protocol (<code>circus_fashioniq_release_v1</code>).
            Candidate gallery: 74,381 images.
          </p>
        </div>
      </div>

      {/* Subset Selector Pills */}
      <div className="subset-selector" role="tablist" aria-label="Dataset Subsets">
        {(["full", "sf", "validated"] as const).map((subKey) => {
          const info = SUBSET_INFO[subKey];
          const active = subset === subKey;
          return (
            <button
              key={subKey}
              role="tab"
              aria-selected={active}
              className={`subset-pill ${active ? "active" : ""}`}
              onClick={() => {
                setSubset(subKey);
                setOffset(0);
              }}
            >
              <span className="subset-title">{info.label}</span>
              <span className="subset-counts">
                {info.queries.toLocaleString()}&nbsp;queries · {info.gallery.toLocaleString()}&nbsp;gallery
              </span>
            </button>
          );
        })}
      </div>

      {/* Runs Table */}
      <div className="table-container">
        <h3>Model Runs ({runs.length})</h3>
        {loading ? (
          <p>Loading composition runs…</p>
        ) : runs.length === 0 ? (
          <p className="empty-notice">No composition run summaries found for subset {subset}.</p>
        ) : (
          <table className="composition-table">
            <thead>
              <tr>
                <th scope="col">Model / Run</th>
                <th scope="col">Condition / Transfer</th>
                <th scope="col">Status</th>
                <th scope="col" className="num-col">MM nDCG</th>
                <th scope="col" className="num-col">MM R@10</th>
                <th scope="col" className="num-col">MM R@50</th>
                <th scope="col" className="num-col">Text nDCG</th>
                <th scope="col" className="num-col">Image nDCG</th>
                <th scope="col" className="num-col">Comp Gap</th>
                <th scope="col" className="num-col">Coverage</th>
                <th scope="col">Action</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => {
                const isSelected = r.run_id === selectedRunId;
                const m = r.metrics || {};
                const gapVal = m.composition_gap_ndcg;
                const isPositiveGap = gapVal != null && gapVal > 0;
                return (
                  <tr key={r.run_id} className={isSelected ? "selected-row" : ""}>
                    <td>
                      <div className="model-cell">
                        <b>{r.model_id}</b>
                        <small>{r.checkpoint_id}</small>
                        {r.is_synthetic && <span className="badge badge-warning">FIXTURE</span>}
                        {r.smoke?.limited && <span className="badge badge-smoke">SMOKE ({r.smoke.query_limit})</span>}
                      </div>
                      <small className="provenance-note" style={{ color: "#8b949e", fontSize: "11px", marginTop: "2px" }}>
                        env: {r.interpreter_verification_status || "unknown"} · {r.reproduction_status || "unknown"}
                      </small>
                    </td>
                    <td>
                      <span className="condition-label">{r.training_or_transfer_label || "0% noise"}</span>
                    </td>
                    <td>
                      <span className={`status-badge status-${(r.execution_status || "").toLowerCase()}`}>
                        {r.execution_status || "UNKNOWN"}
                      </span>
                    </td>
                    <td className="num-col font-mono">{formatNdcg(m.ndcg_multimodal)}</td>
                    <td className="num-col font-mono">{formatRecall(m.recall_at_10_multimodal)}</td>
                    <td className="num-col font-mono">{formatRecall(m.recall_at_50_multimodal)}</td>
                    <td className="num-col font-mono">{formatNdcg(m.ndcg_text_only)}</td>
                    <td className="num-col font-mono">{formatNdcg(m.ndcg_image_only)}</td>
                    <td className="num-col font-mono">
                      <span className={isPositiveGap ? "gap-positive" : "gap-neutral"}>
                        {formatPercent(gapVal)}
                      </span>
                    </td>
                    <td className="num-col font-mono">
                      {m.coverage != null && !r.is_failed ? `${(m.coverage * 100).toFixed(1)}%` : "—"}
                    </td>
                    <td>
                      <button
                        className={`btn-drill ${isSelected ? "btn-active" : ""}`}
                        onClick={() => {
                          setSelectedRunId(r.run_id);
                          setOffset(0);
                        }}
                      >
                        {isSelected ? "Inspecting" : "Drill Down"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Drill-down Section */}
      {activeRun && (
        <div className="drilldown-section">
          <div className="drilldown-header">
            <h3>
              Drill-down: {activeRun.model_id} ({activeRun.checkpoint_id})
            </h3>
            <span className="run-id-tag">run: {activeRun.run_id}</span>
          </div>
          <div className="drilldown-provenance" style={{ fontSize: "12px", color: "#8b949e" }}>
            <b>provenance:</b> env: {activeRun.interpreter_verification_status || "unknown"} · {activeRun.reproduction_status || "unknown"}
          </div>
          {/* Outcome metrics overview */}
          {activeRun.metrics?.outcomes && (
            <div className="outcomes-bar">
              <span className="outcome-pill outcome-comp">
                Comp Required: {activeRun.metrics.outcomes.composition_required.toLocaleString()}
              </span>
              <span className="outcome-pill outcome-text">
                Text Only: {activeRun.metrics.outcomes.text_only.toLocaleString()}
              </span>
              <span className="outcome-pill outcome-image">
                Image Only: {activeRun.metrics.outcomes.image_only.toLocaleString()}
              </span>
              <span className="outcome-pill outcome-both">
                Both: {activeRun.metrics.outcomes.both.toLocaleString()}
              </span>
              <span className="outcome-pill outcome-unres">
                Unresolved: {activeRun.metrics.outcomes.unresolved.toLocaleString()}
              </span>
              <span className="outcome-pill outcome-unscored">
                Unscored: {activeRun.metrics.outcomes.neither_scored.toLocaleString()}
              </span>
            </div>
          )}

          {/* Query Filters */}
          <div className="drilldown-filters">
            <label>
              Condition
              <select value={condition} onChange={(e) => { setCondition(e.target.value); setOffset(0); }}>
                <option value="multimodal">Multimodal</option>
                <option value="text_only_black_image">Text-Only (Black Image)</option>
                <option value="image_only_empty_text">Image-Only (Empty Text)</option>
              </select>
            </label>

            <label>
              Outcome
              <select value={outcome} onChange={(e) => { setOutcome(e.target.value); setOffset(0); }}>
                <option value="">All Outcomes</option>
                <option value="composition_required">Composition Required</option>
                <option value="text_only">Text Only</option>
                <option value="image_only">Image Only</option>
                <option value="both">Both</option>
                <option value="unresolved">Unresolved</option>
                <option value="unscored">Unscored</option>
              </select>
            </label>

            <label>
              Category
              <select value={category} onChange={(e) => { setCategory(e.target.value); setOffset(0); }}>
                <option value="">All Categories</option>
                <option value="dress">dress</option>
                <option value="shirt">shirt</option>
                <option value="toptee">toptee</option>
              </select>
            </label>

            <label>
              Order
              <select value={order} onChange={(e) => { setOrder(e.target.value); setOffset(0); }}>
                <option value="worst">Worst Rank First</option>
                <option value="best">Best Rank First</option>
                <option value="asc">Ascending Rank</option>
                <option value="desc">Descending Rank</option>
              </select>
            </label>

            <label>
              Query ID
              <input
                type="text"
                placeholder="e.g. 7:1…"
                value={querySearch}
                onChange={(e) => { setQuerySearch(e.target.value); setOffset(0); }}
              />
            </label>
          </div>

          {/* Queries List */}
          {queriesUnavailableReason ? (
            <div className="notice-banner">{queriesUnavailableReason}</div>
          ) : loadingQueries ? (
            <p>Loading queries…</p>
          ) : queries.length === 0 ? (
            <p className="empty-notice">No queries match the selected filters.</p>
          ) : (
            <div className="queries-list">
              <div className="pagination-bar">
                <span>
                  Showing {offset + 1}–{Math.min(offset + queries.length, totalQueries)} of {totalQueries.toLocaleString()} queries
                </span>
                <div className="pager-buttons">
                  <button
                    disabled={offset <= 0}
                    onClick={() => setOffset((o) => Math.max(0, o - 20))}
                  >
                    Previous
                  </button>
                  <button
                    disabled={offset + 20 >= totalQueries}
                    onClick={() => setOffset((o) => o + 20)}
                  >
                    Next
                  </button>
                </div>
              </div>

              {queries.map((q) => (
                <article key={q.query_id} className="query-card">
                  <header className="query-card-header">
                    <span className="query-id font-mono"><b>{q.query_id}</b></span>
                    {q.category && <span className="badge category-badge">{q.category}</span>}
                    <span className={`outcome-badge outcome-${q.outcome}`}>{q.outcome}</span>
                  </header>

                  <div className="query-content">
                    {/* Visual Images Side-by-Side */}
                    <div className="query-images">
                      <CompositionImage category={q.category} id={q.reference_id} kind="Reference" />
                      <div className="target-images">
                        {q.positive_ids.map((posId) => (
                          <CompositionImage key={posId} category={q.category} id={posId} kind="Target" />
                        ))}
                      </div>
                    </div>

                    {/* Query Text and Condition Ranks */}
                    <div className="query-meta">
                      <div className="query-text">
                        <b>Edit text:</b> <span>{q.query_txt}</span>
                      </div>

                      <div className="ranks-row">
                        <div className="rank-item">
                          <small>Multimodal</small>
                          <span className="rank-value font-mono">
                            {q.ranks.multimodal != null ? `#${q.ranks.multimodal.toLocaleString()}` : <span className="unscored">unscored</span>}
                          </span>
                        </div>
                        <div className="rank-item">
                          <small>Text-Only</small>
                          <span className="rank-value font-mono">
                            {q.ranks.text_only_black_image != null ? `#${q.ranks.text_only_black_image.toLocaleString()}` : <span className="unscored">unscored</span>}
                          </span>
                        </div>
                        <div className="rank-item">
                          <small>Image-Only</small>
                          <span className="rank-value font-mono">
                            {q.ranks.image_only_empty_text != null ? `#${q.ranks.image_only_empty_text.toLocaleString()}` : <span className="unscored">unscored</span>}
                          </span>
                        </div>
                      </div>

                      {q.failure && <div className="query-failure-note">Failure: {q.failure}</div>}
                    </div>
                  </div>
                </article>
              ))}

              <div className="pagination-bar bottom-pager">
                <div className="pager-buttons">
                  <button
                    disabled={offset <= 0}
                    onClick={() => { setOffset((o) => Math.max(0, o - 20)); window.scrollTo({ top: 300, behavior: "smooth" }); }}
                  >
                    Previous
                  </button>
                  <button
                    disabled={offset + 20 >= totalQueries}
                    onClick={() => { setOffset((o) => o + 20); window.scrollTo({ top: 300, behavior: "smooth" }); }}
                  >
                    Next
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function CompositionImage({ category, id, kind }: { category: string | null; id: string; kind: string }) {
  const [failed, setFailed] = useState<boolean>(false);
  const cat = category || "circus";
  const cleanId = id.split("/").pop() || id;

  if (failed) {
    return (
      <figure className="image-figure image-missing">
        <div className="image-placeholder">{kind}: {cleanId}</div>
        <figcaption><small>{kind}</small></figcaption>
      </figure>
    );
  }

  return (
    <figure className="image-figure">
      <img
        src={`${BASE}/api/images/${cat}/${encodeURIComponent(cleanId)}`}
        alt={`${kind}: ${cleanId}`}
        loading="lazy"
        width={100}
        height={100}
        onError={() => setFailed(true)}
      />
      <figcaption><small>{kind}: {cleanId}</small></figcaption>
    </figure>
  );
}
