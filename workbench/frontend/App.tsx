import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AREAS, KNOWN_PROTOCOL_IDS, SUBSET_LABEL, TO_SEARCH_VERSION, areaDef, hrefFor, parseLocation, toSearch, type AreaId, type Location } from "./routes";
import { useWorkbenchData, protocolLabel, type RunCatalogEntry } from "./data";
import { OverviewPage } from "./OverviewPage";
import { ExperimentsPage } from "./ExperimentsPage";
import { EvaluationPage } from "./EvaluationPage";
import { SamplesPage } from "./SamplesPage";
import { FailuresPage } from "./FailuresPage";
import { NotebookPage } from "./NotebookPage";
import { Card, CopyLink, EmptyState } from "./ui";
import { fmtInt } from "./format";
import { evidenceBanner, evidenceState } from "./evidence";

/** The shell: URL → context → area.
 *
 *  The URL is the source of truth for area, view, protocol, subset, selected runs, inspected query
 *  and every shareable filter. Every navigation is a `pushState`, so Back/Forward walk the operator's
 *  own trail, and a `popstate` listener restores the state a URL describes. Invalid values are
 *  reported in the context bar (and the message names the accepted values) instead of being
 *  silently replaced by a default: a reader must never see one subset's numbers under another
 *  subset's label because a URL was quietly coerced.
 */
export default function App() {
  const [knownProtocols, setKnownProtocols] = useState<string[]>([]);
  const initial = useMemo(() => parseLocation(window.location.search, knownProtocolsForParse()), []);
  const [location, setLocation] = useState<Location>(initial.location);
  const [problems, setProblems] = useState(initial.problems);
  const [menuOpen, setMenuOpen] = useState(false);
  const [dismissed, setDismissed] = useState(false);

  const data = useWorkbenchData({
    protocol: location.protocol,
    subset: location.subset,
    selectedKeys: location.runs,
    condition: location.condition,
    outcome: location.outcome,
    category: location.category,
    order: location.order,
    offset: location.offset,
    limit: 20,
    query: location.q,
    imported: location.imported,
  });

  // Protocols this deployment actually serves, from its own data. Kept in state so the URL parser
  // can validate later navigations against the same list the selectors use.
  useEffect(() => {
    if (data.protocolIds.length && data.protocolIds.join(",") !== knownProtocols.join(",")) {
      setKnownProtocols(data.protocolIds);
    }
  }, [data.protocolIds, knownProtocols]);

  // Write the canonical URL. `replaceState` is used only when nothing the operator did changed the
  // area (a background default landing), so Back never skips an area they chose.
  const writeUrl = useCallback((next: Location, mode: "push" | "replace") => {
    const search = toSearch(next);
    const url = `${window.location.pathname}${search}${TO_SEARCH_VERSION}`;
    if (`${window.location.pathname}${window.location.search}` === url) return;
    if (mode === "push") window.history.pushState({ at: Date.now() }, "", url);
    else window.history.replaceState({ at: Date.now() }, "", url);
  }, []);

  const goto = useCallback((patch: Partial<Location>, mode: "push" | "replace" = "push") => {
    setLocation((current) => {
      const next = { ...current, ...patch };
      // A different area starts at its own first view (or the explicit one in the patch).
      if (patch.area && patch.view === undefined) {
        next.view = areaDef(patch.area).views[0]?.id ?? "";
      }
      writeUrl(next, mode);
      return next;
    });
  }, [writeUrl]);

  // Select a run for analysis: composition runs are exclusive within a subset, and choosing one
  // clears the inspected query so a query from another run cannot be shown under the new run.
  const selectRun = useCallback((entry: RunCatalogEntry, additive?: boolean) => {
    setLocation((current) => {
      const key = entry.key;
      const has = current.runs.includes(key);
      const runs = additive
        ? (has ? current.runs.filter((item) => item !== key) : [...current.runs, key])
        : [key];
      const next: Location = {
        ...current,
        runs,
        run: entry.family === "composition" ? entry.runId : current.run,
        subset: entry.family === "composition" ? current.subset : current.subset,
        query: "",
        offset: 0,
      };
      writeUrl(next, "push");
      return next;
    });
  }, [writeUrl]);

  // Back/Forward. The URL is re-parsed with the protocol list now known, so a restored link with a
  // protocol this deployment does not serve is reported rather than silently swapped.
  useEffect(() => {
    const onPop = () => {
      const parsed = parseLocation(window.location.search, knownProtocols.length ? knownProtocols : knownProtocolsForParse());
      setLocation(parsed.location);
      setProblems(parsed.problems);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [knownProtocols]);

  // First load: if the URL said nothing, adopt the protocol the data actually has, so the header
  // never claims "FashionIQ Original" while the served rows are CIRCUS release content.
  useEffect(() => {
    if (window.location.search) return;
    const fallback = data.protocolIds[0];
    if (fallback && fallback !== location.protocol) {
      setLocation((current) => {
        const next = { ...current, protocol: fallback };
        writeUrl(next, "replace");
        return next;
      });
    }
  }, [data.protocolIds, location.protocol, writeUrl]);

  // A subset switch KEEPS the run selection.
  //
  // Clearing it was a defect the browser harness caught: a CIRCUS composition run has the same
  // `run_id` in full/sf/validated (one summary row per subset), so the selection is still valid
  // after the switch — but dropping it left the label reading "SF" over an empty pager and no
  // `runs=` in the URL, which is the same class of lie as painting the wrong rows. The inspected
  // query is cleared because a query id from another subset's page is not the same reading.
  //
  // Selection validity is not assumed here: `reconcileSelection` below removes any key the serving
  // catalog no longer contains, so a run that genuinely does not exist in the new scope cannot
  // linger as a phantom selection.
  const changeSubset = (subset: Location["subset"]) => {
    goto({ subset, query: "", offset: 0 });
  };

  // Drop selected runs the current catalog does not serve.
  //
  // This is the ONLY place that rewrites the selection because of served data, and it reads the
  // catalog, which is itself currency-guarded (`useRegion`): a superseded response never reaches
  // it, so a late answer cannot remove a selection it knows nothing about. It only runs once the
  // composition summaries are READY, so an in-flight or failed load cannot look like "no runs
  // exist" and wipe the operator's choice.
  useEffect(() => {
    if (data.compositionRuns.region.status !== "ready") return;
    if (!location.runs.length) return;
    const served = new Set(data.catalog.map((entry) => entry.key));
    // The per-query index family is not filtered by subset, so its keys always remain valid.
    const kept = location.runs.filter((key) => key.startsWith("i:") || served.has(key));
    if (kept.length === location.runs.length) return;
    const nextRuns = kept;
    setLocation((current) => {
      const next: Location = { ...current, runs: nextRuns, query: nextRuns.length ? current.query : "" };
      writeUrl(next, "replace");
      return next;
    });
  }, [data.catalog, data.compositionRuns.region.status, location.runs, writeUrl]);

  useEffect(() => {
    if (!location.view && areaDef(location.area).views.length) {
      goto({ view: areaDef(location.area).views[0].id }, "replace");
    }
  }, [location.area, location.view, goto]);

  const evidence = evidenceState(
    (data.indexRuns.region.data ?? []).map((run) => ({ run_id: run.run_id, data_kind: run.data_kind, evidence_status: run.evidence_status })),
    (data.aggregates.region.data ?? []).map((run) => ({ run_id: run.run_id, is_synthetic: run.is_synthetic, verification: run.verification })));
  const banner = evidenceBanner(evidence);

  // The selection-scoped claim that replaces the old app-wide banner: it is about the runs on
  // screen, not about the deployment, and it disappears when the selection does have per-query ranks.
  const selectedEntries = data.catalog.filter((entry) => location.runs.includes(entry.key));
  const selectionNotice = selectedEntries.length === 0
    ? "Chưa chọn lượt nào — số liệu bên dưới là của toàn bộ dữ liệu đang có."
    : selectedEntries.every((entry) => !entry.capabilities.flags.per_query_target_ranks)
      ? `${selectedEntries.length} lượt đang chọn không có rank từng query, nên không phân tích được query cá nhân.`
      : null;

  const area = areaDef(location.area);
  const openArea = (id: AreaId) => {
    goto({ area: id, view: undefined });
    setMenuOpen(false);
  };

  return (
    <div className="shell">
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} role="presentation" />}
      <nav className="sidebar" data-open={menuOpen} aria-label="Khu vực" data-testid="workbench-nav">
        <div className="brand">
          <b>CIR Workbench</b>
          <span>research</span>
        </div>
        {AREAS.map((item) => (
          <button key={item.id} type="button" className="nav-item"
                  aria-current={location.area === item.id ? "page" : undefined}
                  data-testid={`area-nav-${item.id}`}
                  onClick={() => openArea(item.id)}>
            <span className="nav-glyph" aria-hidden="true">{GLYPHS[item.id]}</span>
            <span>{item.label}</span>
            {COUNTS[item.id] ? <span className="nav-count">{COUNTS[item.id]!(data, location)}</span> : null}
          </button>
        ))}
        <div className="sidebar-foot">
          <small>{protocolLabel(location.protocol)}</small>
          <small>Subset {SUBSET_LABEL[location.subset]}</small>
          <small>{data.catalog.length} lượt khả dụng</small>
        </div>
      </nav>

      <div className="workspace">
        <header className="topbar">
          <button type="button" className="menu-toggle ghost" aria-expanded={menuOpen}
                  aria-label="Mở điều hướng" onClick={() => setMenuOpen(true)}>☰</button>
          <div>
            <h1>{area.label}</h1>
            <p className="question">{area.question}</p>
          </div>
          <div className="topbar-actions">
            <CopyLink href={hrefFor(location)} />
          </div>
        </header>

        <div className="contextbar">
          <div className="ctx">
            <label htmlFor="protocol-select">Protocol</label>
            <select id="protocol-select" value={location.protocol}
                    onChange={(event) => goto({ protocol: event.target.value, runs: [], run: "", query: "", offset: 0 })}>
              {(knownProtocols.length ? knownProtocols : [location.protocol]).map((id) => (
                <option key={id} value={id}>{protocolLabel(id)}</option>
              ))}
            </select>
          </div>
          {(location.area === "evaluation" || location.area === "samples" || location.area === "failures") && (
            <div className="ctx">
              <label htmlFor="subset-select">Subset</label>
              <select id="subset-select" value={location.subset}
                      data-testid="subset-select"
                      onChange={(event) => changeSubset(event.target.value as Location["subset"])}>
                {(["full", "sf", "validated"] as const).map((subset) => (
                  <option key={subset} value={subset}>{SUBSET_LABEL[subset]}</option>
                ))}
              </select>
            </div>
          )}
          <div className="ctx">
            <span>Lượt chọn</span><b data-testid="selected-run-count">{location.runs.length}</b>
          </div>
          {location.query && (
            <div className="ctx"><span>Query</span><b className="mono">{location.query}</b></div>
          )}
          <span className="spacer" />
          {area.views.length > 0 && (
            <div className="viewtabs" role="tablist" aria-label={`Chế độ xem ${area.label}`}>
              {area.views.map((view) => (
                <button key={view.id} type="button" role="tab" className="viewtab"
                        aria-selected={location.view === view.id}
                        data-testid={`view-${view.id}`}
                        onClick={() => goto({ view: view.id })}>{view.label}</button>
              ))}
            </div>
          )}
        </div>

        {problems.length > 0 && (
          <div className="content" style={{ paddingBottom: 0 }}>
            <Card title="Giá trị URL không hợp lệ" subtitle="Không giá trị nào bị âm thầm thay đổi; hãy chọn lại">
              <ul style={{ margin: 0, paddingLeft: 18 }}>
                {problems.map((problem) => (
                  <li key={`${problem.param}:${problem.value}`}>
                    <code>{problem.param}={problem.value}</code> — {problem.message}
                  </li>
                ))}
              </ul>
            </Card>
          </div>
        )}

        {(banner || selectionNotice) && !dismissed && (
          <div className="content" style={{ paddingBottom: 0 }}>
            <div className={`empty ${banner ? "empty-partial" : "empty-capability"}`}
                 data-testid="error-banner" data-notice-kind={banner ? "evidence" : "selection"}
                 style={{ flexDirection: "row", alignItems: "center", gap: 12 }}>
              <div style={{ flex: 1 }}>
                {banner && <b>{banner}</b>}
                {selectionNotice && <span className="empty-detail" style={{ display: "block" }}>{selectionNotice}</span>}
              </div>
              <button type="button" className="ghost" data-testid="error-dismiss"
                      onClick={() => setDismissed(true)} aria-label="Ẩn thông báo">Ẩn</button>
            </div>
          </div>
        )}

        <main className="content">
          {location.area === "overview" && (
            <OverviewPage data={data} location={location}
                          onOpenRun={(entry) => selectRun(entry)}
                          onGoto={goto} />
          )}
          {location.area === "experiments" && (
            <ExperimentsPage data={data} location={location} view={location.view}
                             onSelectRun={selectRun} onGoto={goto} />
          )}
          {location.area === "evaluation" && (
            <EvaluationPage data={data} location={location} view={location.view}
                            onGoto={goto} onSelectRun={selectRun} />
          )}
          {location.area === "samples" && (
            <SamplesPage data={data} location={location} onGoto={goto}
                         onAnnotate={() => goto({ area: "notebook", view: "annotations" })} />
          )}
          {location.area === "failures" && (
            <FailuresPage data={data} location={location} onGoto={goto} onSaved={() => data.cohorts.retry()} />
          )}
          {location.area === "notebook" && (
            <NotebookPage data={data} location={location} view={location.view} onGoto={goto} />
          )}
        </main>
      </div>
    </div>
  );
}

const GLYPHS: Record<AreaId, string> = {
  overview: "◎",
  experiments: "▤",
  evaluation: "◑",
  samples: "▦",
  failures: "△",
  notebook: "✎",
};

/** Sidebar counts are real counts from the loaded data, or nothing at all — never a placeholder. */
const COUNTS: Partial<Record<AreaId, (data: ReturnType<typeof useWorkbenchData>, location: Location) => string>> = {
  experiments: (data) => data.catalog.length ? fmtInt(data.catalog.length) : "",
  evaluation: (data, location) => {
    const runs = (data.compositionRuns.region.data ?? []).filter((run) => run.subset === location.subset);
    return runs.length ? fmtInt(runs.length) : "";
  },
  failures: (data) => {
    const selected = data.catalog.filter((entry) => entry.capabilities.flags.per_query_target_ranks);
    return selected.length ? fmtInt(selected.length) : "";
  },
};

/** Before the data lands the served protocol list is unknown; the parser accepts the known ids so a
 *  valid deep link is not reported as invalid on first paint. Same list the registry test reads. */
function knownProtocolsForParse(): string[] {
  return [...KNOWN_PROTOCOL_IDS];
}
