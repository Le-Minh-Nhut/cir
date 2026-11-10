import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AREAS, KNOWN_PROTOCOL_IDS, SUBSET_LABEL, TO_SEARCH_VERSION, areaDef, hrefFor, parseLocation, toSearch, type AreaId, type Location } from "./routes";
import { useWorkbenchData, protocolLabel, type RunCatalogEntry } from "./data";
import { protocolsWithReports, protocolOrDefault } from "./catalog";
import { chooseInitialScope, chooseSubsetFor, protocolHasSubsets, SUBSET_SEQUENCE } from "./scope";
import { OverviewPage } from "./OverviewPage";
import { ExperimentsPage } from "./ExperimentsPage";
import { EvaluationPage } from "./EvaluationPage";
import { SamplesPage } from "./SamplesPage";
import { FailuresPage } from "./FailuresPage";
import { NotebookPage } from "./NotebookPage";
import { Card, CopyLink, EmptyState } from "./ui";
import { fmtInt } from "./format";
import { evidenceBanner, evidenceState } from "./evidence";

/** The one protocol whose circuit supports full/sf/validated subsets. Named once, from the parser's
 *  own acceptance list, so the subset control cannot appear over a protocol that has no subset. */
const CIRCUS_PROTOCOL = KNOWN_PROTOCOL_IDS[0];

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
  // Which halves of the scope the URL itself named. A landing correction may fill an unstated half
  // and must never overwrite a stated one.
  const [namedScope, setNamedScope] = useState(initial.scope);
  const [menuOpen, setMenuOpen] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  // Reading size is a reader preference, not a browser zoom setting: 17px default, 19px on request,
  // remembered across reloads. `data-density` on <html> is what style.css keys off.
  const [density, setDensity] = useState(() => {
    try { return localStorage.getItem("cir-density") === "large" ? "large" : "comfortable"; }
    catch { return "comfortable"; }
  });
  useEffect(() => {
    document.documentElement.dataset.density = density;
    try { localStorage.setItem("cir-density", density); } catch { /* private browsing */ }
  }, [density]);

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
    queryState: location.queryState,
    // The whole-universe read is analysis input only; the explorer does not need 20k rows.
    loadAnalysis: location.area === "failures",
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
        // A run is an artifact OF one protocol: opening it must carry that protocol, or the page
        // header keeps describing a different one while the rows come from this run.
        protocol: entry.protocolId,
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
      setNamedScope(parsed.scope);
    };
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [knownProtocols]);

  // Landing scope, chosen from the rows the API actually served.
  //
  // Two defects this replaces. First, `data.protocolIds[0]` is insertion order, and the deployment's
  // FIRST protocol (CIRCUS release) has composition rows while its reports live under
  // `fashioniq_original_split` -- so "first" is not "most". Second, and worse: a bare URL adopted a
  // protocol but KEPT the default subset `sf`, which that protocol may not have. Opening the app then
  // printed "FashionIQ Original / SF / 0 lượt" beside a deployment holding 13 aggregate reports, and
  // it looked like a workbench with no results.
  //
  // So: each half of the scope is filled only when the URL did not name it, from the same data the
  // page is about to count. `replaceState`, because a background correction must not become a history
  // entry the operator has to walk back, and never when the reader stated the value.
  useEffect(() => {
    if (data.compositionRuns.region.status !== "ready") return;
    const rows = data.compositionRuns.region.data ?? [];
    if (!rows.length) return;
    const choice = chooseInitialScope(rows);
    if (!choice) return;
    const nextProtocol = namedScope.protocolNamed ? location.protocol : choice.protocol;
    const subset = namedScope.subsetNamed
      ? location.subset
      : (chooseSubsetFor(rows, nextProtocol) ?? location.subset);
    if (nextProtocol === location.protocol && subset === location.subset) return;
    setLocation((current) => {
      const next: Location = { ...current, protocol: nextProtocol, subset: subset as Location["subset"] };
      writeUrl(next, "replace");
      return next;
    });
  }, [data.compositionRuns.region.status, data.compositionRuns.region.data,
      location.protocol, location.subset, namedScope.protocolNamed, namedScope.subsetNamed, writeUrl]);

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

  // The benchmark view must not open on a dead end. The deployment's first protocol has composition
  // rows but zero aggregate reports, so landing there showed an empty card that read as broken.
  // This redirects ONLY when (a) the benchmark view is open, (b) the selected protocol has no report
  // at all while another does, and (c) the URL did not name a protocol — an explicit choice is left
  // alone, because a reader who asked for an empty protocol deserves to see it. `replace`, not
  // `push`: a background correction must not become a history entry the operator has to walk back.
  useEffect(() => {
    if (location.area !== "evaluation" || location.view !== "benchmark") return;
    if (data.aggregates.region.status !== "ready") return;
    if (new URLSearchParams(window.location.search).has("protocol")) return;
    const served = protocolsWithReports(data.aggregates.region.data ?? []);
    if (!served.length) return;
    const target = protocolOrDefault(location.protocol, served);
    if (target === location.protocol) return;
    setLocation((current) => {
      const next = { ...current, protocol: target, runs: [], run: "", query: "", offset: 0 };
      writeUrl(next, "replace");
      return next;
    });
  }, [location.area, location.view, location.protocol, data.aggregates.region.status, data.aggregates.region.data, writeUrl]);

  useEffect(() => {
    if (!location.view && areaDef(location.area).views.length) {
      goto({ view: areaDef(location.area).views[0].id }, "replace");
    }
  }, [location.area, location.view, goto]);

  const evidence = evidenceState(
    (data.indexRuns.region.data ?? []).map((run) => ({ run_id: run.run_id, data_kind: run.data_kind, evidence_status: run.evidence_status })),
    (data.aggregates.region.data ?? []).map((run) => ({ run_id: run.run_id, is_synthetic: run.is_synthetic, verification: run.verification })),
    (data.compositionRuns.region.data ?? []).map((run) => ({
      run_id: run.run_id, protocol_id: run.protocol_id, subset: run.subset, capabilities: run.capabilities,
    })),
    { protocolId: location.protocol, subset: location.subset },
    // The banner is a claim about what this deployment HOLDS, so it needs each source's own request
    // state: an empty array means "nothing arrived yet" as often as it means "nothing exists".
    {
      runs: data.indexRuns.region.status,
      benchmarks: data.aggregates.region.status,
      composition: data.compositionRuns.region.status,
    });
  const banner = evidenceBanner(evidence);

  // The selection-scoped claim that replaces the old app-wide banner: it is about the runs on
  // screen, not about the deployment, and it disappears when the selection does have per-query ranks.
  const selectedEntries = data.catalog.filter((entry) => location.runs.includes(entry.key));
  const selectionNotice = selectedEntries.length > 0
    && selectedEntries.every((entry) => !entry.capabilities.flags.per_query_target_ranks)
    ? "Các lượt đang chọn chỉ có số tổng hợp. Chọn một lượt có hạng từng query để khám phá ảnh."
    : null;

  const area = areaDef(location.area);
  const openArea = (id: AreaId) => {
    goto({ area: id, view: undefined });
    setMenuOpen(false);
  };

  return (
    <div className="shell" data-shell-state={data.compositionRuns.region.status}>
      <a className="skip-link" href="#main-content">Đến nội dung chính</a>
      {menuOpen && <div className="scrim" onClick={() => setMenuOpen(false)} role="presentation" />}
      <nav className="sidebar" data-open={menuOpen} aria-label="Khu vực" data-testid="workbench-nav">
        <div className="brand">
          <b>CIR<span className="brand-dot">.</span></b>
          <span>Research workbench</span>
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
        <div className="sidebar-foot" data-testid="sidebar-scope">
          <small>{protocolLabel(location.protocol)}</small>
          {protocolHasSubsets(location.protocol) && <small>Subset {SUBSET_LABEL[location.subset]}</small>}
          {/* The catalog is per-subset, so a deployment-wide number here contradicted the scoped
              count printed on the page directly beside it. */}
          <small data-testid="sidebar-catalog-count">{data.catalog.length} lượt khả dụng ở phạm vi này</small>
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
            <label className="density-control">Cỡ giao diện
              <select aria-label="Cỡ giao diện" value={density}
                      data-testid="density-select"
                      onChange={(event) => setDensity(event.target.value)}>
                <option value="comfortable">17px</option>
                <option value="large">19px</option>
              </select>
            </label>
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
          {/* The subset control exists only over a protocol whose circuit HAS subsets. Over
              FashionIQ Original it used to print "Shortcut-Free (SF)" while every row beneath was
              that protocol's single cohort — a label for a cohort that does not exist. */}
          {protocolHasSubsets(location.protocol) && (
            <div className="ctx">
              <label htmlFor="subset-select">Subset</label>
              <select id="subset-select" value={location.subset}
                      data-testid="subset-select"
                      onChange={(event) => changeSubset(event.target.value as Location["subset"])}>
                {SUBSET_SEQUENCE.map((subset) => (
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
            <Card title="Giá trị URL không hợp lệ" testId="url-problems"
                  subtitle="Không giá trị nào bị âm thầm thay đổi; hãy chọn lại">
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
            <div className={`empty banner ${banner ? "empty-partial" : "empty-capability"}`}
                 data-testid="error-banner" data-notice-kind={banner ? "evidence" : "selection"}
                 data-evidence-loading={evidence.loading || evidence.failedSources.length > 0 || undefined}>
              <div className="banner-text">
                {banner && <b>{banner}</b>}
                {selectionNotice && <span className="empty-detail" style={{ display: "block" }}>{selectionNotice}</span>}
              </div>
              <button type="button" className="ghost" data-testid="error-dismiss"
                      onClick={() => setDismissed(true)} aria-label="Ẩn thông báo">Ẩn</button>
            </div>
          </div>
        )}

        <main className="content" id="main-content" tabIndex={-1}>
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
  experiments: (data) => data.models.region.data?.length ? fmtInt(data.models.region.data.length) : "",
  evaluation: (data, location) => {
    const runs = (data.compositionRuns.region.data ?? [])
      .filter((run) => run.subset === location.subset && run.protocol_id === location.protocol);
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
