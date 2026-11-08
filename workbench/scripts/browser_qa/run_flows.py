"""Durable browser QA harness for the CIR Failure Analysis Workbench.

Runs eight independent flows against the *real served app* and writes a machine-readable
`qa_report.json` plus screenshots. Selectors prefer `data-testid`, fall back to role+name and then
legacy CSS, and report a miss as its own failure category so the harness survives the UI redesign
happening in parallel.

    workbench/envs/ctl312h/bin/python workbench/scripts/browser_qa/run_flows.py \
        --out workbench/docs/reviews/<timestamp>-qa

Every assertion is checked against a ground truth that is not the UI talking to itself: the HTTP API
and the raw artifacts on disk. "No console errors" is recorded as evidence and is never itself a
pass criterion.

Flows
  a  end-to-end     two runs -> disagreement query -> images/ranks/Top-K -> note -> cohort -> reload
  b  race           reversed response order under rapid scope/filter switching
  c  degraded       failing request+retry, empty filter, no query, aggregate-only, rank-only,
                    partial coverage, failed run, blocked model
  d  navigation     refresh, deep link per area, Back/Forward, invalid URL values
  e  artifacts      raw rank artifact -> API -> DOM for a fixed query set
  f  cohorts        saved set equals the full published scope, not the visible page
  g  annotations    save, reload, still bound to the same query
  h  layout         overflow, <24px targets, distorted images, focus, sticky occlusion
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import traceback
import urllib.parse
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

import harness as H  # noqa: E402  (plain module, not a package: keeps invocation path independent)

DEFAULT_UI = "http://127.0.0.1:5173"
DEFAULT_API = "http://127.0.0.1:8000"
VIEWPORTS = ("1440x900", "1920x1080", "390x844")
CARDS_PER_PAGE = 20

#: Area aliases: the redesigned Vietnamese names and today's English tabs both resolve.
AREAS: dict[str, list[str]] = {
    "overview": ["Tổng quan", "Overview", "Dashboard"],
    "experiments": ["Thí nghiệm", "Experiments", "Evaluation Runs"],
    "evaluation": ["Đánh giá", "Evaluation", "Composition", "Benchmark"],
    "samples": ["Khám phá mẫu", "Samples", "Sample Explorer"],
    "failures": ["Phân tích lỗi", "Failures", "Failure Analytics", "Common Failures"],
    "notebook": ["Sổ nghiên cứu", "Notebook", "Annotations", "Hypotheses"],
    "composition": ["Composition"],
    "models": ["Models", "Mô hình"],
}

#: Accessible/visible names of the three composition subsets.
SUBSET_LABELS: dict[str, list[str]] = {
    "full": ["Full"],
    "sf": ["Shortcut-Free (SF)", "Shortcut-Free", "SF"],
    "validated": ["Validated"],
}




#: The six work areas the redesign collapses today's twelve tabs into.
REDESIGN_AREAS = ("overview", "experiments", "evaluation", "samples", "failures", "notebook")

#: The testids the redesign should add; grouped by the selector key that wanted them. Kept next to
#: the registry so `testid-requests.md` is generated from what the run actually missed.
TESTID_CATALOG: dict[str, str] = {
    "nav": "nav container: `data-testid=\"workbench-nav\"`",
    "nav.item": "each area link: `data-testid=\"area-nav-<area>\"` where area is the stable key "
                "(overview, experiments, evaluation, samples, failures, notebook)",
    "nav.active": "active area: `data-testid=\"area-nav-active\"` or `aria-current=\"page\"` on the link",
    "area.root": "area root: `data-testid=\"area-<area>\"` on the outer element of each area",
    "error.banner": "`data-testid=\"error-banner\"` on the guard/API error region",
    "error.dismiss": "`data-testid=\"error-dismiss\"` on its dismiss control",
    "error.retry": "`data-testid=\"error-retry\"` on any retry/reload control for a failed request",
    "empty.state": "`data-testid=\"empty-state\"` on a zero-result placeholder",
    "notice": "`data-testid=\"notice\"` on an explanatory (non-error) notice such as aggregate-only",
    "overview.cards": "`data-testid=\"overview-cards\"` on the summary card container",
    "runs.checkbox": "per-run selector checkbox: `data-testid=\"run-option-<run_id>\"`",
    "runs.row": "per-run row: `data-testid=\"run-row-<run_id>\"`",
    "runs.selected_count": "`data-testid=\"selected-run-count\"` with the count as its text",
    "models.section": "`data-testid=\"area-models\"` on the models list; per-model row "
                      "`data-testid=\"model-row-<model_id>\"` with the reproduction status as text",
    "comp.root": "`data-testid=\"composition\"` on the composition view",
    "comp.subset": "subset pill: `data-testid=\"subset-pill-<full|sf|validated>\"` plus "
                   "`role=\"tab\"` + `aria-selected` (already present, keep it)",
    "comp.table": "`data-testid=\"composition-table\"` on the runs table",
    "comp.row": "per-run row: `data-testid=\"composition-row-<run_id>\"`",
    "comp.drill": "`data-testid=\"drill-down\"` on the drill-down region",
    "comp.pager": "`data-testid=\"pager-total\"` carrying `data-total` and `data-shown` attributes "
                  "so the harness can read the denominator without parsing prose",
    "comp.pager.next": "`data-testid=\"pager-next\"` on the next-page control",
    "comp.cards": "`data-testid=\"query-list\"` on the query card list",
    "comp.qid": "`data-testid=\"query-id\"` on each card's id (with the raw id as text)",
    "comp.rank": "`data-testid=\"rank-<condition>\"` per condition rank (condition is one of "
                 "multimodal, text_only_black_image, image_only_empty_text)",
    "comp.empty": "`data-testid=\"composition-empty\"` when no query matches the filters",
    "comp.filter": "filter control: `data-testid=\"filter-<condition|outcome|category|order>\"`",
    "comp.search": "`data-testid=\"filter-query-id\"` on the query-id search input",
    "comp.topk": "`data-testid=\"topk-grid\"` on the candidate grid when a run publishes Top-K",
    "comp.image": "`data-testid=\"query-image\"` on each query image (with the image id as "
                  "`data-image-id`)",
    "explorer.active_run": "`data-testid=\"active-run\"` on the active run selector",
    "explorer.query_title": "`data-testid=\"query-title\"` on the inspected query heading",
    "explorer.topk": "`data-testid=\"topk-grid\"` on the candidate grid",
    "explorer.image": "`data-testid=\"query-image\"` on each query image",
    "explorer.jump": "`data-testid=\"jump-query-id\"` on the jump-to-query input",
    "explorer.jump.button": "`data-testid=\"jump-submit\"` on the jump button",
    "annot.note": "`data-testid=\"annotation-note\"` on the note textarea",
    "annot.save": "`data-testid=\"annotation-save\"` on the save control",
    "annot.query": "`data-testid=\"annotation-query-id\"` on the text naming the bound query",
    "fail.section": "`data-testid=\"area-failures\"` on the failure-analysis area",
    "fail.row": "`data-testid=\"analysis-row\"` per analysis row, with `data-query-id`",
    "fail.preset": "`data-testid=\"failure-preset\"` on the consensus preset selector",
    "cohort.name": "`data-testid=\"cohort-id\"` on the cohort id input",
    "cohort.save": "`data-testid=\"cohort-save\"` on the save control",
    "cohort.item": "`data-testid=\"cohort-<cohort_id>\"` per saved cohort",
}


# --------------------------------------------------------------------------- helpers

def now_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%S")


def area_url(ui: str, area: str, view: str | None = None, **params: Any) -> str:
    """Canonical deep link. `?area=` is the redesign's route space; the legacy `?page=` name is a
    separate, explicitly-labelled entry point (flow d checks both)."""
    query = {"area": area}
    if view:
        query["view"] = view
    query.update({key: value for key, value in params.items() if value not in (None, "")})
    return f"{ui}/?{urllib.parse.urlencode(query)}"


def legacy_url(ui: str, page: str) -> str:
    return f"{ui}/?page={urllib.parse.quote(page)}"


class Ctx:
    """Per-viewport browser context plus the data the flows assert against."""

    def __init__(self, page, capture, api: H.Api, ui_url: str, api_url: str, out_dir: Path,
                 viewport: str, read_only: bool):
        self.page = page
        self.capture = capture
        self.api = api
        self.ui = ui_url.rstrip("/")
        self.api_url = api_url.rstrip("/")
        self.out = out_dir
        self.viewport = viewport
        self.read_only = read_only
        self._comp_runs: list[dict] | None = None
        self._runs: list[dict] | None = None
        self._nav_labels: list[str] | None = None

    # -- data, read once per run
    @property
    def runs(self) -> list[dict]:
        if self._runs is None:
            self._runs = self.api.get("/api/runs") or []
        return self._runs

    @property
    def comp_runs(self) -> list[dict]:
        """One record per run_id.

        `/api/composition/runs` answers one row per (run, subset) and only some rows carry
        `available_subsets`, so the raw list repeats a run and would make "pick two runs" pick the
        same one twice. The merged record keeps the union of the subsets.
        """
        if self._comp_runs is None:
            raw = self.api.get("/api/composition/runs") or []
            merged: dict[str, dict] = {}
            for row in raw:
                run_id = row.get("run_id") or ""
                entry = merged.get(run_id)
                subset = row.get("subset")
                if entry is None:
                    entry = merged[run_id] = {**row, "available_subsets": [],
                                              "metrics_by_subset": {}}
                seen = set(entry["available_subsets"])
                seen.update(row.get("available_subsets") or [])
                if subset:
                    seen.add(subset)
                entry["available_subsets"] = sorted(seen)
                if subset and row.get("metrics"):
                    entry["metrics_by_subset"][subset] = row["metrics"]
                for field in ("category_resolution", "query_count"):
                    if entry.get(field) is None and row.get(field) is not None:
                        entry[field] = row[field]
            for entry in merged.values():
                if not entry.get("metrics") and entry["metrics_by_subset"]:
                    entry["metrics"] = next(iter(entry["metrics_by_subset"].values()))
            self._comp_runs = list(merged.values())
        return self._comp_runs

    def comp_run(self, *, status: str | None = None, failed: bool | None = None,
                 subsets: tuple[str, ...] = ("full", "sf", "validated")) -> dict | None:
        for run in self.comp_runs:
            if run.get("is_synthetic"):
                continue
            if status and (run.get("execution_status") or "").upper() != status:
                continue
            if failed is not None and bool(run.get("is_failed")) is not failed:
                continue
            if not set(subsets).issubset(set(run.get("available_subsets") or [])):
                continue
            return run
        return None

    @staticmethod
    def metrics_for(run: dict, subset: str) -> dict:
        """The summary metrics of one (run, subset) — not of whichever row came first."""
        return (run.get("metrics_by_subset") or {}).get(subset) or {}

    def run_key(self, run_id: str) -> str:
        """The redesign prefixes a run key with its family: `c:<run_id>` (composition release) or
        `i:<run_id>` (per-query index)."""
        return f"c:{run_id}"

    # -- navigation
    def goto(self, url: str, settle_ms: int = 1200, attempts: int = 4) -> None:
        """Navigate, tolerating a server restart.

        The frontend and backend are rebuilt by other agents while this harness runs, so a refused
        connection is a deployment event, not a UI defect. Each attempt re-checks the API first, so
        a run that ends in a timeout can state that the server never came back.
        """
        last: Exception | None = None
        for attempt in range(attempts):
            try:
                self.page.goto(url, wait_until="domcontentloaded", timeout=H.NAV_TIMEOUT_MS)
                self.settle(settle_ms)
                return
            except Exception as exc:  # ERR_CONNECTION_REFUSED / timeout during a rebuild
                last = exc
                if attempt + 1 >= attempts:
                    break
                deadline = time.time() + 20
                while time.time() < deadline:
                    if self.api.get("/api/benchmark-runs") is not None:
                        break
                    time.sleep(1)
        raise RuntimeError(f"could not load {url} after {attempts} attempts: {last}")

    def settle(self, ms: int = 1200) -> None:
        """Waits for the app to look settled without demanding a quiet network: this UI polls jobs
        and re-fetches regions, so `networkidle` can never arrive on some areas."""
        try:
            self.page.wait_for_load_state("networkidle", timeout=2500)
        except Exception:
            pass
        self.page.wait_for_timeout(ms)

    def nav_labels(self) -> list[str]:
        if self._nav_labels is None:
            loc = H.optional(self.page, "nav")
            labels: list[str] = []
            if loc is not None:
                for chunk in H.texts(loc):
                    labels.extend(part.strip() for part in chunk.splitlines() if part.strip())
            self._nav_labels = labels
        return self._nav_labels

    def active_area(self) -> str | None:
        loc = H.optional(self.page, "nav.active", timeout=1200)
        if loc is None or not loc.count():
            return None
        testid = loc.first.get_attribute("data-testid") or ""
        if testid.startswith("area-nav-"):
            return testid[len("area-nav-"):]
        text = " ".join(H.texts(loc)).lower()
        for area, aliases in AREAS.items():
            if any(alias.lower() in text for alias in aliases):
                return area
        return None

    def open_area(self, area: str) -> None:
        loc = H.optional(self.page, "nav.item", {"name": [area] + AREAS.get(area, [])},
                         timeout=2500)
        if loc is not None and loc.count():
            loc.first.click()
            self.settle(1000)
        else:
            self.goto(area_url(self.ui, area))

    def shot(self, flow: H.Flow, name: str, full_page: bool = True) -> str:
        path = self.out / "screens" / f"{flow.flow_id}-{flow.viewport}-{name}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.page.screenshot(path=str(path), full_page=full_page)
            rel = str(path.relative_to(self.out))
            flow.screenshots.append(rel)
            return rel
        except Exception as exc:  # a screenshot must never abort a flow
            flow.note(f"screenshot {name} failed: {exc}")
            return ""

    # -- the data surface this checkout can actually serve
    def comp_query_page(self, run_id: str, subset: str, condition: str = "multimodal",
                        order: str = "worst", limit: int = CARDS_PER_PAGE,
                        offset: int = 0, qid: str | None = None,
                        outcome: str | None = None) -> dict:
        params = {"subset": subset, "condition": condition, "order": order,
                  "limit": str(limit), "offset": str(offset)}
        if qid:
            params["query_id"] = qid
        if outcome:
            params["outcome"] = outcome
        return self.api.get(
            f"/api/composition/runs/{urllib.parse.quote(run_id)}/queries?"
            f"{urllib.parse.urlencode(params)}") or {}

    def explorer_url(self, run_id: str, subset: str = "sf", condition: str = "multimodal",
                     order: str = "worst", query: str = "", offset: int = 0,
                     outcome: str = "") -> str:
        return area_url(self.ui, "samples", "explorer", protocol="circus_fashioniq_release_v1",
                        subset=subset, runs=self.run_key(run_id), condition=condition,
                        order=order, query=query,
                        offset=str(offset) if offset else "", outcome=outcome)

    def evaluation_url(self, run_id: str, subset: str = "sf",
                       protocol: str = "circus_fashioniq_release_v1") -> str:
        return area_url(self.ui, "evaluation", "composition", protocol=protocol,
                        subset=subset, run=run_id)

    def browser_evidence(self, flow: H.Flow, *, assert_clean: bool = True) -> None:
        snap = self.capture.snapshot()
        flow.evidence.update(snap)
        if not assert_clean:
            return
        injected = len(snap["injected_failures"])
        real_failed = [r for r in snap["failed_requests"]
                       if not r["url"].split("?")[0].endswith(H.IGNORE_REQUEST_SUFFIXES)]
        # Image 404s are a data-coverage fact checked by flow (e); everything else is a defect.
        real_http = [r for r in snap["http_ge_400"] if "/api/images/" not in r["url"]]
        flow.check("browser.page_errors", not snap["page_errors"],
                   f"{len(snap['page_errors'])} uncaught page error(s)"
                   + (f": {snap['page_errors'][0][:160]}" if snap["page_errors"] else ""))
        flow.check("browser.failed_requests", not real_failed,
                   f"{len(real_failed)} failed request(s)"
                   + (f": {real_failed[0]['url'][:120]} {real_failed[0]['failure'][:80]}"
                      if real_failed else "")
                   + (f"; {injected} expected injected failure(s)" if injected else ""))
        flow.check("browser.http_ge_400", not real_http,
                   f"{len(real_http)} HTTP >= 400 response(s)"
                   + (f": {real_http[0]['status']} {real_http[0]['url'][:120]}"
                      if real_http else ""))
        flow.note(f"console errors: {len(snap['console_errors'])} "
                  "(recorded as evidence, never a pass criterion)")
        flow.evidence["injected_failure_count"] = injected


def pager_state(flow: H.Flow, ctx: Ctx) -> tuple[int | None, int | None, str]:
    """(shown, total, text) from the pager of whichever list is on screen."""
    loc = H.optional(ctx.page, "comp.pager")
    if loc is None or not loc.count():
        return None, None, ""
    total: int | None = None
    attr = loc.first.get_attribute("data-total")
    if attr is not None:
        try:
            total = int(attr)
        except ValueError:
            total = None
    text = " ".join(H.texts(loc))
    shown, parsed_total = H.parse_pager_total(text)
    if total is None:
        total = parsed_total
    if shown is None:
        attr_shown = loc.first.get_attribute("data-shown")
        shown = int(attr_shown) if attr_shown and attr_shown.isdigit() else None
    return shown, total, text


def card_ids(flow: H.Flow, ctx: Ctx) -> list[str]:
    loc = H.optional(ctx.page, "comp.qid")
    if loc is None:
        return []
    out = []
    for text in H.texts(loc):
        match = re.match(r"([\w\-]+:\d+)", text)
        out.append(match.group(1) if match else text.strip())
    return out


def _visible_subset(ctx: Ctx) -> list[str]:
    """What the UI currently says the scope is, whichever control it uses: the selected pill's text
    or the `<select>`'s current value. A tablist can nest headings, so short noise is filtered out."""
    # Only subset controls, never a view tab: "Inspector" is a tab too, and reporting it as the
    # scope label would make the assertion meaningless.
    select = H.optional(ctx.page, "subset.select")
    if select is not None and select.count():
        try:
            return [select.first.input_value()]
        except Exception:
            pass
    pills = ctx.page.locator("[data-testid^='subset-pill-'][aria-selected='true'], "
                            ".subset-pill[aria-selected='true']")
    labels = [t.replace("\n", " ") for t in H.texts(pills) if len(t.strip()) > 3]
    if labels:
        return labels
    select = H.optional(ctx.page, "subset.select")
    if select is not None and select.count():
        try:
            return [select.first.input_value()]
        except Exception:
            return []
    return []


def set_subset(ctx: Ctx, subset: str) -> bool:
    """Change the subset through whichever control this area renders.

    The evaluation area uses pills (one button per subset); the samples area uses a
    `<select data-testid="subset-select">`. Both are legitimate, so the flow drives either.
    """
    pill = subset_locator(ctx, subset)
    if pill is not None and pill.count():
        pill.first.click()
        return True
    select = H.optional(ctx.page, "subset.select")
    if select is not None and select.count():
        try:
            select.first.select_option(subset)
            return True
        except Exception:
            return False
    return False


def subset_locator(ctx: Ctx, subset: str):
    """Resolve a subset pill by testid first, then by its (changeable) label.

    Retried once: a route change re-renders the tablist, so a single 2 s attempt can land between
    the old and the new DOM and report a miss that is only a re-render.
    """
    fmt = {"subset": subset, "label": SUBSET_LABELS.get(subset, [subset])}
    loc = H.optional(ctx.page, "comp.subset", fmt, timeout=4000)
    if loc is None or not loc.count():
        ctx.page.wait_for_timeout(700)
        loc = H.optional(ctx.page, "comp.subset", fmt, timeout=4000)
    return loc


def require(flow: H.Flow, cond: bool, ident: str, detail: str) -> bool:
    flow.check(ident, cond, detail, category="prerequisite" if not cond else "assertion")
    return cond


def open_query_in_explorer(ctx: Ctx, run_id: str, qid: str) -> None:
    """Deep link straight to one query; the URL is the app's own contract, so this exercises the
    same path a shared link does."""
    ctx.goto(ctx.explorer_url(run_id, query=qid))


# --------------------------------------------------------------------------- flow (a)

#: How many runs the end-to-end flow wants. Two, because a disagreement needs two runs.
NEEDS_RUNS = 2


def flow_a(ctx: Ctx, flow: H.Flow) -> None:
    """Pick two runs -> disagreement query -> images/ranks/Top-K -> note -> cohort -> reload."""
    per_query = [r for r in ctx.runs if (r.get("top_k_saved") or 0) > 0]
    flow.evidence["indexed_runs"] = len(ctx.runs)
    flow.evidence["per_query_runs"] = [r["run_id"] for r in per_query]
    ctx.goto(area_url(ctx.ui, "overview"))
    flow.check("a.overview_reachable", ctx.active_area() == "overview",
               f"active area after navigation: {ctx.active_area()}")
    ctx.shot(flow, "overview")

    # --- the per-query path needs the DuckDB index; this deployment serves none.
    if not require(flow, len(per_query) >= NEEDS_RUNS, "a.two_per_query_runs",
                   f"/api/runs exposes {len(ctx.runs)} run(s), {len(per_query)} with stored Top-K; "
                   f"{NEEDS_RUNS} protocol-compatible per-query runs are required for a "
                   "disagreement query"):
        for step in ("a.select_two_runs", "a.open_disagreement_query", "a.images", "a.ranks",
                     "a.topk", "a.write_note", "a.save_cohort", "a.reload_same_data"):
            flow.blocked(step, "blocked: the per-query index (/api/runs) is empty, so no query can "
                               "be selected, inspected or annotated through the index path")

    # --- the surface that exists: the CIRCUS composition inspector. Exercise it end to end and
    #     record what it does and does not carry, rather than reporting a fake success.
    run = ctx.comp_run(status="COMPLETED", failed=False)
    if not require(flow, run is not None, "a.composition_run_available",
                   "a completed composition run with full/sf/validated summaries is required"):
        return
    rid = run["run_id"]
    page = ctx.comp_query_page(rid, "sf")
    items = page.get("items") or []
    if not require(flow, bool(items), "a.composition_query_available",
                   f"run {rid[:12]} publishes queries for subset sf"):
        return
    qid = items[0]["query_id"]
    flow.evidence["substitute"] = {
        "run_id": rid, "query_id": qid, "subset": "sf",
        "why": "the per-query index is empty, so the composition inspector is the only query "
               "surface this deployment can serve",
        "top_k_available": bool(items[0].get("top_results")),
    }
    ctx.goto(ctx.explorer_url(rid, "sf", query=qid))
    ids = card_ids(flow, ctx)
    flow.check("a.query_opens_in_inspector", qid in ids,
               f"query {qid} present in the inspector list: {ids[:4]}")
    title = H.optional(ctx.page, "explorer.query_title")
    flow.check("a.query_title", title is not None and title.count() > 0,
               f"inspector title: {H.texts(title)[:1] if title is not None else 'missing'}")
    imgs = H.optional(ctx.page, "explorer.image")
    if imgs is None:
        flow.miss(H.SelectorMiss("explorer.image", ["no image locator matched"]), "query images")
    else:
        srcs = [i.get_attribute("src") or "" for i in imgs.all()]
        flow.check("a.query_images", len(srcs) >= 2,
                   f"{len(srcs)} query image(s): {srcs[:3]}")
    ranks = {el.get_attribute("data-testid"): el.inner_text()
             for el in (H.optional(ctx.page, "comp.rank.any") or ctx.page.locator("nothing")).all()} \
        if H.optional(ctx.page, "comp.rank.any") is not None else {}
    flow.evidence["rendered_ranks"] = ranks
    flow.check("a.ranks_rendered", bool(ranks), f"condition ranks rendered: {list(ranks)}")
    topk = H.optional(ctx.page, "comp.topk")
    unavailable = H.optional(ctx.page, "explorer.candidates_unavailable")
    has_candidates = bool(items[0].get("top_results"))
    rendered_candidates = H.count(topk) if topk is not None else 0
    flow.check("a.topk_honest", has_candidates or rendered_candidates == 0,
               f"API publishes {len(items[0].get('top_results') or [])} candidate(s) and the DOM "
               f"renders {rendered_candidates} candidate grid(s)"
               + (" (capability notice shown)" if unavailable is not None else ""))
    protocol = "circus_fashioniq_release_v1"
    ctx.goto(area_url(ctx.ui, "notebook", "annotations", protocol=protocol, query=qid))
    ctx.settle(1500)
    note = H.optional(ctx.page, "annot.note")
    save = H.optional(ctx.page, "annot.save")
    if note is None or save is None:
        flow.miss(H.SelectorMiss("annot.note", ["no annotation form for a deep-linked query"]),
                  "write and save a note")
        flow.blocked("a.note_roundtrip", "blocked: the annotation form did not render for the "
                                        "deep-linked query")
    else:
        note_text = f"qa note {now_stamp()} {random.choice(['alpha', 'bravo', 'charlie'])}"
        note.first.fill(note_text)
        save.first.click()
        ctx.settle(1800)
        server = ctx.api.get(f"/api/annotations?query_id={urllib.parse.quote(qid)}"
                             f"&protocol_id={protocol}") or []
        saved = any(r.get("note") == note_text for r in server)
        feedback = " ".join(H.texts(ctx.page.locator("[role='alert'],[role='status']")))
        flow.evidence["note_feedback"] = feedback[:300]
        # The note must either persist, or the save must say why it could not. Silence is the bug.
        flow.check("a.note_written", saved or bool(feedback),
                   f"note {'persisted' if saved else 'refused with a message'} for ({qid}, "
                   f"{protocol}): {feedback[:180]!r}")
        if saved:
            ctx.page.reload(wait_until="domcontentloaded")
            ctx.settle(2000)
            # `innerText` on a <textarea> is ALWAYS empty in Chromium: a textarea keeps its content
            # in `value`, not in a text child, so `all_inner_texts()` cannot read it. Read the value.
            restored = [ctx.page.locator("textarea").first.input_value()]
            flow.check("a.note_survives_reload", note_text in restored,
                       f"note textarea value after reload: {restored[:1]}")
            bound = H.optional(ctx.page, "annot.query")
            flow.check("a.note_still_bound", bound is not None and qid in " ".join(H.texts(bound)),
                       f"reloaded note is bound to {qid}: {H.texts(bound)[:1]}")
    # --- cohort over a scope larger than one page
    selectable = [r["run_id"] for r in ctx.comp_runs
                  if not r.get("is_synthetic")
                  and (r.get("execution_status") or "").upper() == "COMPLETED"
                  and "sf" in (r.get("available_subsets") or [])]
    ctx.goto(area_url(ctx.ui, "failures", "analytics", protocol=protocol, subset="sf",
                      runs=",".join(ctx.run_key(rid_) for rid_ in selectable[:2])))
    ctx.settle(2500)
    _, total, _ = pager_state(flow, ctx)
    scope = ctx.comp_query_page(rid, "sf").get("total")
    flow.check("a.cohort_scope_is_full_scope", total == scope,
               f"failure analysis cohort size {total} vs published sf scope {scope}")
    ctx.shot(flow, "failure-analytics")
    ctx.browser_evidence(flow)


# --------------------------------------------------------------------------- flow (b)

def _hold_routes(ctx: Ctx, pattern: str, marker: str) -> dict:
    """Intercept API responses whose URL contains `marker`, but only once armed.

    Arming matters: the page's *first* load must be allowed to complete normally, otherwise the
    scenario cannot tell "the abandoned response overwrote good data" from "nothing ever rendered".
    """
    state: dict = {"held": [], "seen": [], "armed": False}

    def handler(route):
        url = route.request.url
        state["seen"].append(url)
        if state["armed"] and marker in url:
            response = route.fetch()
            state["held"].append((route, response, time.time()))
            return
        route.continue_()

    ctx.page.route(pattern, handler)
    return state


def _arm(state: dict) -> None:
    state["armed"] = True


def _release(state: dict, reverse: bool = True) -> list[str]:
    routes = list(state["held"])
    if reverse:
        routes.reverse()
    released = []
    for route, response, _ in routes:
        try:
            route.fulfill(response=response)
            released.append(urllib.parse.unquote(route.request.url.split("?", 1)[-1])[:90])
        except Exception as exc:
            released.append(f"release failed: {exc}")
    state["held"].clear()
    return released


def _unroute_all(ctx: Ctx) -> None:
    try:
        ctx.page.unroute_all(behavior="ignoreErrors")
    except Exception:
        try:
            ctx.page.unroute("**/*")
        except Exception:
            pass


def flow_b(ctx: Ctx, flow: H.Flow) -> None:
    """Rapid switching with deliberately reversed response order.

    The label must stay consistent with the data: when the header says subset SF, the pager
    denominator and the first row must be SF's — even though the abandoned Validated response is
    released *after* SF's. A scenario that intercepted nothing is reported as a failure, because a
    fast network proves nothing.
    """
    run = ctx.comp_run(status="COMPLETED", failed=False)
    if not require(flow, run is not None, "b.composition_run_available",
                   "a completed composition run with full/sf/validated summaries is required"):
        return
    rid = run["run_id"]
    sf_page = ctx.comp_query_page(rid, "sf")
    # Validated is the abandoned scope: it differs from SF in BOTH the denominator (586 vs 4,069)
    # and the first row (7:3477 vs 7:603), so a leaked response is visible either way.
    val_page = ctx.comp_query_page(rid, "validated")
    sf_total, val_total = sf_page.get("total"), val_page.get("total")
    if not require(flow, bool(sf_total and val_total and sf_total != val_total),
                   "b.distinguishable_scopes",
                   f"sf total={sf_total}, validated total={val_total} (must differ)"):
        return
    sf_first = (sf_page.get("items") or [{}])[0].get("query_id")
    val_first = (val_page.get("items") or [{}])[0].get("query_id")
    if not require(flow, bool(sf_first and val_first and sf_first != val_first),
                   "b.distinguishable_first_page",
                   f"sf first={sf_first}, validated first={val_first} (must differ)"):
        return
    flow.evidence["scopes"] = {"sf": sf_total, "validated": val_total,
                               "sf_first": sf_first, "validated_first": val_first}

    ctx.capture.expect_failure("subset=validated")
    ctx.capture.expect_failure("condition=multimodal")
    state = _hold_routes(ctx, "**/api/**", "subset=validated")
    ctx.goto(ctx.explorer_url(rid, "sf"))
    _, total0, text0 = pager_state(flow, ctx)
    flow.check("b.initial_scope_sf", total0 == sf_total,
               f"initial pager before any switching: {text0!r}")

    pills = H.count(H.optional(ctx.page, "comp.subset_any"))
    select = H.optional(ctx.page, "subset.select")
    has_select = bool(select is not None and select.count())
    if not require(flow, pills >= 2 or has_select, "b.subset_control_present",
                   f"subset control on this surface: {pills} pill(s), select={has_select}"):
        _unroute_all(ctx)
        return
    _arm(state)
    if not set_subset(ctx, "validated"):
        flow.miss(H.SelectorMiss("comp.subset", ["the subset control did not accept 'validated'"]),
                  "switch subset to validated")
        _unroute_all(ctx)
        return
    ctx.page.wait_for_timeout(600)
    set_subset(ctx, "sf")
    ctx.page.wait_for_timeout(1800)
    held_count = len(state["held"])
    flow.check("b.reversed_order_injected", held_count >= 1,
               f"{held_count} abandoned Validated response(s) held back while SF was requested "
               f"({len(state['seen'])} API request(s) seen)")
    released = _release(state, reverse=True)
    _unroute_all(ctx)
    # The abandoned response is applied, then the app re-requests the current scope. Give it the
    # chance to settle before judging, so "empty while reloading" is not read as "wrong data".
    try:
        ctx.page.wait_for_function(
            "() => document.querySelector('[data-testid=pager-total]') !== null", timeout=15000)
    except Exception:
        pass
    ctx.page.wait_for_timeout(2000)

    _, total_after, text_after = pager_state(flow, ctx)
    labels = _visible_subset(ctx)
    ids = card_ids(flow, ctx)
    selected_runs = urllib.parse.parse_qs(urllib.parse.urlparse(ctx.page.url).query).get("runs")
    flow.evidence["race"] = {"released_late": released, "url": ctx.page.url,
                             "selected_runs_in_url": selected_runs,
                             "pager": text_after, "label": labels,
                             "first_row": ids[0] if ids else None,
                             "expected": {"total": sf_total, "first": sf_first}}
    if selected_runs is None:
        flow.note("the abandoned response removed `runs=` from the URL: the run selection was "
                  "dropped by the late answer, so the scope the label still names has no data")
    flow.check("b.label_says_sf",
               any("shortcut" in lab.lower() or lab.strip().lower() == "sf" for lab in labels),
               f"visible subset label: {labels}")
    lost_selection = selected_runs is None
    flow.check("b.data_matches_label", total_after == sf_total,
               f"label {labels} but the pager shows {text_after!r}"
               + ("" if total_after is not None else
                  " (no data at all: the late Validated answer dropped `runs=` from the URL)")
               + f" (SF total {sf_total}, abandoned Validated total {val_total})")
    flow.check("b.selection_survives_late_response", not lost_selection,
               f"run selection after the abandoned response: "
               f"{selected_runs if selected_runs is not None else 'lost'}")
    flow.check("b.first_row_matches_label", bool(ids) and ids[0] == sf_first,
               f"first visible query {ids[0] if ids else None} vs SF first {sf_first} "
               f"(the abandoned scope starts at {val_first})")
    ctx.shot(flow, "race-subset")

    # --- condition switch, same reversal
    img_page = ctx.comp_query_page(rid, "sf", condition="image_only_empty_text")
    img_first = (img_page.get("items") or [{}])[0].get("query_id")
    flow.evidence["condition_first"] = {"image_only_empty_text": img_first,
                                       "multimodal": sf_first}
    state2 = _hold_routes(ctx, "**/api/**", "condition=multimodal")
    # Start from image_only so the abandoned request is a real transition; re-selecting the current
    # condition issues no request, and a scenario that never switched would prove nothing.
    ctx.goto(ctx.explorer_url(rid, "sf", condition="image_only_empty_text"))
    cond = H.optional(ctx.page, "comp.filter",
                      {"name": ["condition"], "label": ["Điều kiện", "Condition"]})
    if cond is None:
        flow.miss(H.SelectorMiss("comp.filter", ["no condition select matched"]),
                  "switch condition")
        _unroute_all(ctx)
        return
    _arm(state2)
    cond.first.select_option("multimodal")
    ctx.page.wait_for_timeout(500)
    cond.first.select_option("image_only_empty_text")
    ctx.page.wait_for_timeout(1800)
    held2 = len(state2["held"])
    flow.check("b.condition_reversed_order_injected", held2 >= 1,
               f"{held2} abandoned multimodal response(s) held back while image_only was requested")
    _release(state2, reverse=True)
    _unroute_all(ctx)
    ctx.page.wait_for_timeout(2000)
    ids2 = card_ids(flow, ctx)
    try:
        visible_cond = cond.first.input_value()
    except Exception:
        visible_cond = None
    flow.evidence["race_condition"] = {"visible_condition": visible_cond,
                                      "first_row": ids2[0] if ids2 else None,
                                      "expected_first": img_first}
    flow.check("b.condition_label", visible_cond == "image_only_empty_text",
               f"visible condition: {visible_cond!r}")
    flow.check("b.condition_data_matches_label", bool(ids2) and ids2[0] == img_first,
               f"the selector says image_only_empty_text but the first row is "
               f"{ids2[0] if ids2 else None} (image-only order starts at {img_first})")
    ctx.shot(flow, "race-condition")
    ctx.browser_evidence(flow)


# --------------------------------------------------------------------------- flow (c)

def flow_c(ctx: Ctx, flow: H.Flow) -> None:
    """Every degraded state, one assertion group each."""
    run = ctx.comp_run(status="COMPLETED", failed=False)
    if not require(flow, run is not None, "c.composition_run_available",
                   "a completed composition run is required"):
        return
    rid = run["run_id"]
    protocol = "circus_fashioniq_release_v1"

    # 1. a failing request, and recovery through the app's own retry control
    ctx.capture.expect_failure("/queries?")
    ctx.page.route("**/api/composition/runs/*/queries*", lambda route: route.abort("failed"))
    ctx.goto(ctx.explorer_url(rid, "sf"))
    ctx.settle(1200)
    state = H.optional(ctx.page, "error.retry")
    retry = H.optional(ctx.page, "error.retry")
    text = " ".join(H.texts(H.optional(ctx.page, "empty.state") or ctx.page.locator("body")))
    flow.check("c.error_surfaced", retry is not None and retry.count() > 0,
               f"a failed query request renders a retry affordance; page text: {text[:140]!r}")
    ctx.shot(flow, "error-state")
    flow.evidence["retry_control_present"] = bool(retry is not None and retry.count())
    H_unroute = getattr(ctx.page, "unroute_all", None)
    _unroute_all(ctx)
    if retry is not None and retry.count():
        retry.first.click()
    ctx.settle(2500)
    _, total_recovered, text_recovered = pager_state(flow, ctx)
    expected = ctx.comp_query_page(rid, "sf").get("total")
    flow.check("c.recovery_after_retry", total_recovered == expected,
               f"after retry the list shows {text_recovered!r} (expected {expected})")
    ctx.capture.clear()
    ctx.shot(flow, "after-retry")

    # 2. a filter that matches nothing
    ctx.goto(ctx.explorer_url(rid, "sf"))
    search = H.optional(ctx.page, "comp.search")
    if search is None:
        flow.miss(H.SelectorMiss("comp.search", ["no query-id filter matched"]), "empty filter")
    else:
        search.first.fill("7:99999999")
        ctx.settle(1800)
        cards = H.count(H.optional(ctx.page, "comp.cards"))
        empty = H.optional(ctx.page, "comp.empty")
        flow.check("c.empty_filter_no_cards", cards == 0,
                   f"{cards} card(s) for a query id that cannot exist")
        flow.check("c.empty_filter_message", empty is not None and empty.count() > 0,
                   f"zero-result filter shows an explicit empty state: "
                   f"{H.texts(empty)[:1] if empty is not None else 'none'}")
        ctx.shot(flow, "empty-filter")

    # 3. no query selected -> the notebook names what is missing and what to do next
    ctx.goto(area_url(ctx.ui, "notebook", "annotations"))
    ctx.settle(1200)
    body = ctx.page.inner_text("main")
    flow.check("c.no_query_state_message", "query" in body.lower(),
               f"annotation area without a query explains the requirement: {body[-220:]!r}")
    ctx.shot(flow, "no-query")

    # 4. aggregate-only run: the index protocol has only aggregate reports
    ctx.goto(area_url(ctx.ui, "samples", "explorer",
                      protocol="fashioniq_original_split", subset="sf"))
    ctx.settle(2000)
    body = ctx.page.inner_text("body")
    explicit = H.optional(ctx.page, "empty.state")
    empty_kind = explicit.first.get_attribute("data-empty-kind") if explicit is not None else None
    flow.evidence["aggregate_only_empty_kind"] = empty_kind
    flow.check("c.aggregate_only_explained",
               empty_kind in ("capability", "none-selected")
               or "aggregate" in body.lower() or "CAPABILITY_ABSENT" in body,
               f"an aggregate-only protocol states why per-query inspection is unavailable "
               f"(empty-state kind {empty_kind!r})")
    ctx.shot(flow, "aggregate-only")

    # 5. rank-only run: no candidate list may be fabricated
    qid = ((ctx.comp_query_page(rid, "sf").get("items") or [{}])[0]).get("query_id") or ""
    api_items = (ctx.comp_query_page(rid, "sf", qid=qid).get("items") or [{}])[0]
    ctx.goto(ctx.explorer_url(rid, "sf", query=qid))
    ctx.settle(1500)
    grid = H.optional(ctx.page, "comp.topk")
    rendered = H.count(grid) if grid is not None else 0
    flow.check("c.rank_only_no_fake_topk",
               bool(api_items.get("top_results")) or rendered == 0,
               f"API publishes {len(api_items.get('top_results') or [])} candidate(s) for {qid}; the "
               f"DOM renders {rendered} candidate grid(s)")
    flow.evidence["rank_only"] = {"query_id": qid,
                                 "api_top_results": len(api_items.get("top_results") or []),
                                 "dom_grids": rendered}
    ctx.shot(flow, "rank-only")

    # 6. partial coverage
    partial = ctx.comp_run(status="PARTIAL", failed=False)
    if partial is None:
        flow.blocked("c.partial_coverage", "no PARTIAL composition run is served")
    else:
        ctx.goto(ctx.evaluation_url(partial["run_id"], "sf"))
        ctx.settle(1500)
        row = H.optional(ctx.page, "comp.row", {"run_id": partial["run_id"]})
        row_text = " ".join(H.texts(row)) if row is not None else ""
        flow.check("c.partial_badge", "PARTIAL" in row_text,
                   f"the PARTIAL run's row is labelled PARTIAL: {row_text[:120]!r}")
        coverage = (ctx.metrics_for(partial, "sf").get("multimodal") or {}).get("coverage")
        flow.check("c.partial_coverage_stated", coverage is None or 0 < coverage <= 1,
                   f"partial run reports coverage {coverage}")
        flow.evidence["partial"] = {"run_id": partial["run_id"], "coverage": coverage}
        ctx.shot(flow, "partial-coverage")

    # 7. failed run
    failed = ctx.comp_run(failed=True)
    if failed is None:
        flow.blocked("c.failed_run", "no FAILED composition run is served")
    else:
        ctx.goto(ctx.evaluation_url(failed["run_id"], "sf"))
        ctx.settle(1500)
        row = H.optional(ctx.page, "comp.row", {"run_id": failed["run_id"]})
        row_text = " ".join(H.texts(row)) if row is not None else ""
        flow.check("c.failed_run_label", "FAILED" in row_text,
                   f"the FAILED run's row is labelled FAILED: {row_text[:120]!r}")
        flow.check("c.failed_run_no_numbers",
                   not re.search(r"\b0\.\d{4}\b", row_text),
                   "a FAILED run does not present fabricated metric numbers")
        flow.evidence["failed"] = {"run_id": failed["run_id"],
                                  "execution_status": failed.get("execution_status")}
        ctx.shot(flow, "failed-run")

    # 8. blocked model
    ctx.goto(area_url(ctx.ui, "experiments", "models"))
    ctx.settle(2000)
    models_text = ctx.page.inner_text("main")
    api_models = ctx.api.get("/api/models") or []
    if not api_models:
        # Read twice: a single empty answer during a backend restart is not a fact about the data.
        ctx.page.wait_for_timeout(1500)
        api_models = ctx.api.get("/api/models") or []
    blocked_api = [m for m in api_models
                   if re.search(r"BLOCKED", str(m.get("reproduction_status", "")))]
    flow.evidence["blocked_models"] = [m["model_id"] for m in blocked_api]
    if not require(flow, bool(blocked_api), "c.blocked_model_exists",
                   f"/api/models returned {len(api_models)} model(s); "
                   f"{len(blocked_api)} carry a BLOCKED_* status"):
        return
    sample = blocked_api[0]
    row = H.optional(ctx.page, "models.row", {"model_id": sample["model_id"]})
    row_text = " ".join(H.texts(row)) if row is not None else ""
    flow.check("c.blocked_model_listed", bool(row_text),
               f"blocked model {sample['model_id']} has its own row "
               f"({'data-testid=model-row-' + sample['model_id']})")
    flow.check("c.blocked_model_status_shown", "BLOCKED" in row_text,
               f"the row states the block reason (status {sample['reproduction_status']}): "
               f"{row_text[:140]!r}")
    ctx.shot(flow, "blocked-model")
    ctx.browser_evidence(flow)


# --------------------------------------------------------------------------- flow (d)

def flow_d(ctx: Ctx, flow: H.Flow) -> None:
    """Refresh, deep link per area, Back/Forward, invalid URL values."""
    labels = ctx.nav_labels()
    flow.evidence["nav_labels"] = labels
    flow.check("d.nav_rendered", len(labels) >= 1, f"{len(labels)} nav entr(ies): {labels[:8]}")
    present = [area for area in REDESIGN_AREAS
               if any(alias.lower() == lab.lower() or alias.lower() in lab.lower()
                      for alias in AREAS[area] for lab in labels)]
    flow.evidence["areas_detected"] = present
    flow.check("d.six_area_nav_present", len(present) == len(REDESIGN_AREAS),
               f"{len(present)}/{len(REDESIGN_AREAS)} areas present: {present}")

    # deep link per area, each a fresh load (a shared link must reproduce the view)
    for area in REDESIGN_AREAS:
        ctx.goto(area_url(ctx.ui, area))
        active = ctx.active_area()
        flow.check(f"d.deep_link_{area}", active == area,
                   f"?area={area} -> active area {active!r}")
    ctx.shot(flow, "deep-link-notebook")

    # refresh keeps the area
    ctx.goto(area_url(ctx.ui, "failures"))
    before = ctx.active_area()
    ctx.page.reload(wait_until="domcontentloaded")
    ctx.settle(1500)
    after = ctx.active_area()
    flow.check("d.refresh_keeps_area", before == after,
               f"active area before refresh {before!r}, after {after!r}")

    # Back / Forward over three areas
    ctx.goto(area_url(ctx.ui, "overview"))
    ctx.goto(area_url(ctx.ui, "samples"))
    ctx.goto(area_url(ctx.ui, "failures"))
    ctx.page.go_back()
    ctx.settle(1400)
    back_area = ctx.active_area()
    flow.check("d.back_returns_previous_area", back_area == "samples",
               f"after Back the active area is {back_area!r} (expected samples)")
    ctx.page.go_forward()
    ctx.settle(1400)
    fwd_area = ctx.active_area()
    flow.check("d.forward_returns_next_area", fwd_area == "failures",
               f"after Forward the active area is {fwd_area!r} (expected failures)")

    # invalid URL values: each must be rejected visibly, not crash or silently pretend
    ctx.goto(f"{ctx.ui}/?area=__no_such_area__")
    ctx.settle(1500)
    # The URL-problem card renders above <main>, so the whole page body is the correct scope.
    body = ctx.page.inner_text("body")
    flow.check("d.invalid_area_explained",
               "__no_such_area__" in body and "không phải khu vực hợp lệ" in body,
               f"unknown ?area is named and explained: {body[:200]!r}")
    flow.check("d.invalid_area_falls_back", ctx.active_area() == "overview",
               f"unknown ?area falls back to {ctx.active_area()!r} instead of a blank page")
    ctx.shot(flow, "invalid-area")

    ctx.goto(f"{ctx.ui}/?area=evaluation&view=__no_such_view__")
    ctx.settle(1500)
    flow.check("d.invalid_view_explained", "__no_such_view__" in ctx.page.inner_text("body"),
               "an unknown ?view value is named on the page")
    ctx.shot(flow, "invalid-view")

    run = ctx.comp_run(status="COMPLETED", failed=False)
    if run is not None:
        ctx.goto(area_url(ctx.ui, "evaluation", "composition",
                          protocol="circus_fashioniq_release_v1", subset="bogus_subset"))
        ctx.settle(1500)
        body = ctx.page.inner_text("body")
        flow.check("d.invalid_subset_explained", "subset=bogus_subset" in body,
                   f"invalid subset is named on the page: {body[:200]!r}")
        ctx.shot(flow, "invalid-subset")

    # legacy route space stays deep-linkable
    for page_name, area in (("Dashboard", "overview"), ("Evaluation Runs", "experiments"),
                            ("Composition", "evaluation"), ("Sample Explorer", "samples"),
                            ("Failure Analytics", "failures"), ("Annotations", "notebook")):
        ctx.goto(legacy_url(ctx.ui, page_name))
        active = ctx.active_area()
        flow.check(f"d.legacy_page_{page_name.replace(' ', '_')}", active == area,
                   f"?page={page_name} -> active area {active!r} (expected {area})")
    ctx.goto(legacy_url(ctx.ui, "__not_a_tab__"))
    ctx.settle(1500)
    body = ctx.page.inner_text("body")
    flow.check("d.legacy_unknown_tab_explained", "__not_a_tab__" in body,
               "an unknown legacy tab is named and explained")
    ctx.browser_evidence(flow)


# --------------------------------------------------------------------------- flow (e)

def _artifact_root() -> Path:
    env = os.environ.get("WORKBENCH_ARTIFACTS_ROOT")
    return Path(env) if env else REPO / "workbench/artifacts"


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def flow_e(ctx: Ctx, flow: H.Flow) -> None:
    """Raw rank artifact -> API -> DOM, for a fixed small query set. Fails on any mismatch.

    The three layers are independent: the JSON on disk is the canonical record, the API is what the
    server derives from it, and the DOM is what the operator reads. Any disagreement is a defect
    even when all three look plausible on their own.
    """
    run = ctx.comp_run(status="COMPLETED", failed=False)
    if not require(flow, run is not None, "e.composition_run_available",
                   "a completed composition run is required"):
        return
    rid = run["run_id"]
    root = _artifact_root()
    ranks_dir = root / "composition/runs" / rid / "ranks"
    artifact = None
    for candidate in sorted(ranks_dir.glob("*.json")):
        content = _read_json(candidate)
        if content and "ranks" in content:
            artifact = content
            flow.evidence["rank_artifact"] = str(candidate)
            break
    if not require(flow, bool(artifact and artifact.get("ranks")), "e.rank_artifact_readable",
                   f"ranks map read from {flow.evidence.get('rank_artifact') or ranks_dir}"):
        return
    summary_dir = root / "composition/summary" / f"{run.get('checkpoint_id')}_{rid}"
    if not summary_dir.is_dir():
        matches = sorted((root / "composition/summary").glob(f"*_{rid}"))
        summary_dir = matches[0] if matches else summary_dir
    summary = _read_json(summary_dir / "sf.json") or {}
    art_ranks = artifact["ranks"]
    conditions = ("multimodal", "text_only_black_image", "image_only_empty_text")

    # fixed query set: first three of the SF scope in worst order + one validated-subset query
    page = ctx.comp_query_page(rid, "sf")
    targets = [{"subset": "sf", "item": item} for item in (page.get("items") or [])[:3]]
    v_first = (ctx.comp_query_page(rid, "validated").get("items") or [{}])[0]
    if v_first.get("query_id") and v_first["query_id"] not in [
            t["item"].get("query_id") for t in targets]:
        targets.append({"subset": "validated", "item": v_first})
    if not require(flow, bool(targets), "e.fixed_query_set",
                   f"{len(targets)} query id(s) selected for the cross-check"):
        return
    flow.evidence["targets"] = [
        {"subset": t["subset"], "query_id": t["item"].get("query_id"),
         "api_ranks": t["item"].get("ranks")} for t in targets]

    label_map = {"multimodal": "MULTIMODAL", "text_only_black_image": "TEXT-ONLY",
                 "image_only_empty_text": "IMAGE-ONLY"}

    for entry in targets:
        subset, item = entry["subset"], entry["item"]
        qid = item.get("query_id") or ""
        api_ranks = item.get("ranks") or {}
        art = {cond: (art_ranks.get(cond) or {}).get(qid) for cond in conditions}
        flow.eq(f"e.{qid}.artifact_vs_api", art, api_ranks,
                f"rank artifact vs API for {qid} ({subset})")
        ctx.goto(ctx.explorer_url(rid, subset, query=qid))
        ctx.settle(1800)
        title = H.optional(ctx.page, "explorer.query_title")
        if title is None or not title.count():
            flow.blocked(f"e.{qid}.dom", f"blocked: the API serves {qid} for subset {subset} but the "
                                         "query inspector does not render it, so there is no DOM to "
                                         "compare against the artifact")
            continue
        title_text = " ".join(H.texts(title)) if title is not None else ""
        flow.check(f"e.{qid}.dom_id", qid in title_text,
                   f"inspector title names {qid}: {title_text[:60]!r}")
        # images: the inspector's srcs must name the API's reference/target basenames
        ref = str(item.get("reference_id") or "").split("/")[-1]
        tgt = str((item.get("positive_ids") or [""])[0]).split("/")[-1]
        imgs = H.optional(ctx.page, "explorer.image")
        srcs = [i.get_attribute("src") or "" for i in imgs.all()] if imgs is not None else []
        flow.check(f"e.{qid}.dom_reference_image", bool(ref) and any(ref in s for s in srcs),
                   f"reference {ref} appears in the inspector's image srcs: {srcs[:3]}")
        flow.check(f"e.{qid}.dom_target_image", bool(tgt) and any(tgt in s for s in srcs),
                   f"target {tgt} appears in the inspector's image srcs: {srcs[:3]}")
        # rendered condition ranks, read from the per-condition testids
        rendered: dict[str, str] = {}
        for cond in conditions:
            block = H.optional(ctx.page, "comp.rank", {"condition": cond})
            if block is None:
                continue
            text = " ".join(H.texts(block))
            found = re.search(r"#([\d,\.]+)", text)
            rendered[cond] = found.group(1).replace(",", "") if found else "unscored"
        expected_text = {cond: (str(api_ranks[cond]) if api_ranks.get(cond) is not None
                                else "unscored") for cond in conditions}
        flow.eq(f"e.{qid}.dom_ranks", expected_text, rendered,
                f"rendered condition ranks vs artifact for {qid} "
                f"(labels: {list(label_map.values())})")
        ctx.shot(flow, f"artifact-{qid.replace(':', '-')}")

    # candidate ordering, denominator and one aggregate metric, all against the artifact
    ctx.goto(ctx.explorer_url(rid, "sf"))
    ctx.settle(1800)
    dom_order = card_ids(flow, ctx)
    api_order = [i.get("query_id") for i in (page.get("items") or [])]
    flow.eq("e.order", api_order, dom_order, "first-page query ordering API vs DOM")
    _, total, text = pager_state(flow, ctx)
    art_count = ((summary.get("metrics") or {}).get("multimodal") or {}).get("query_count")
    flow.eq("e.denominator", art_count, total,
            f"pager denominator vs artifact query_count for subset sf ({text!r})")

    # the condition selector must re-order from the artifact, not only re-label
    best = ctx.comp_query_page(rid, "sf", order="best") or {}
    best_order = [i.get("query_id") for i in (best.get("items") or [])]
    ctx.goto(ctx.explorer_url(rid, "sf", order="best"))
    ctx.settle(1800)
    flow.eq("e.order_control", best_order, card_ids(flow, ctx),
            "order=best in the URL: DOM order vs API order")

    # metric cell vs the summary artifact
    ndcg_art = ((summary.get("metrics") or {}).get("multimodal") or {}).get("ndcg_full")
    ctx.goto(ctx.evaluation_url(rid, "sf"))
    ctx.settle(1800)
    row = H.optional(ctx.page, "comp.row", {"run_id": rid})
    row_text = " ".join(H.texts(row)) if row is not None else ""
    if ndcg_art is None or not row_text:
        flow.blocked("e.metric_matches_artifact",
                     "summary artifact or evaluation-table row not available")
    else:
        flow.check("e.metric_matches_artifact", f"{ndcg_art:.4f}" in row_text,
                   f"MM nDCG cell vs summary artifact {ndcg_art:.4f}: {row_text[:140]!r}")
    flow.evidence["artifact_summary"] = str(summary_dir / "sf.json")
    ctx.shot(flow, "artifact-table")
    ctx.browser_evidence(flow)


# --------------------------------------------------------------------------- flow (f)

def flow_f(ctx: Ctx, flow: H.Flow) -> None:
    """Save a filtered set larger than one page; the saved set must be the whole published scope."""
    complete = [r for r in ctx.comp_runs
                if not r.get("is_synthetic")
                and (r.get("execution_status") or "").upper() == "COMPLETED"
                and (r.get("available_subsets") or [])]
    flow.evidence["candidate_runs"] = [r["run_id"] for r in complete]
    if not require(flow, len(complete) >= 2, "f.two_runs_selectable",
                   f"{len(complete)} selectable run(s); failure analysis needs two to build a "
                   "cohort over a per-query-universe"):
        flow.blocked("f.save_cohort", "blocked: cohort scope needs two runs with query-level ranks")
        flow.blocked("f.saved_scope_equals_published",
                     "blocked: nothing to compare until a cohort can be saved")
        return
    rid = complete[0]["run_id"]
    keys = [ctx.run_key(r["run_id"]) for r in complete[:2]]
    protocol = "circus_fashioniq_release_v1"
    subset = "sf"
    expected_total = ctx.comp_query_page(rid, subset).get("total")
    ctx.goto(area_url(ctx.ui, "failures", "analytics", protocol=protocol, subset=subset,
                      runs=",".join(keys)))
    ctx.settle(3000)
    shown, total, text = pager_state(flow, ctx)
    flow.eq("f.cohort_scope_equals_published", expected_total, total,
            f"analysis scope vs published {subset} scope ({text!r})")
    flow.check("f.scope_exceeds_one_page", (total or 0) > CARDS_PER_PAGE,
               f"{total} query/ies in scope, one page is {CARDS_PER_PAGE}: a saved set larger than "
               "the visible page is what makes the check meaningful")

    save = H.optional(ctx.page, "cohort.save")
    if save is None:
        flow.miss(H.SelectorMiss("cohort.save", ["no cohort save control matched"]), "save cohort")
        return
    flow.check("f.save_button_states_scope", str(total) in re.sub(r"\D", "", save.first.inner_text())
               or f"{total:,}" in save.first.inner_text(),
               f"the save control names the full scope: {save.first.inner_text()[:80]!r}")
    save.first.click()
    ctx.settle(3000)
    feedback = " ".join(H.texts(ctx.page.locator("[role='alert'],[role='status']")))
    flow.evidence["save_feedback"] = feedback[:300]
    cohorts = ctx.api.get("/api/cohorts") or []
    flow.evidence["cohorts_on_server"] = len(cohorts)
    flow.check("f.save_outcome_is_reported", bool(feedback),
               f"the save reports its outcome instead of failing silently: {feedback[:160]!r}")
    # Whether it persisted or not, the scope claim must be truthful: either the cohort holds the
    # full page-overflowing set, or the UI said it could not save. A silent partial save is the bug.
    named = [c for c in cohorts if (c.get("notes") or "").find(str(total)) >= 0]
    saved_ok = bool(named)
    flow.check("f.saved_scope_equals_published_or_refused",
               saved_ok or "thất bại" in feedback.lower() or "lỗi" in feedback.lower()
               or "không" in feedback.lower(),
               f"cohort persisted with the full scope ({len(named)} record(s) mentioning {total}) "
               f"or the save refused loudly: {feedback[:140]!r}",
               category="assertion")
    ctx.page.reload(wait_until="domcontentloaded")
    ctx.settle(2500)
    body = ctx.page.inner_text("main")
    flow.check("f.state_survives_reload", str(total) in re.sub(r"[^0-9]", "", body) or
               str(expected_total) in body or "query" in body.lower(),
               "the failure-analysis scope is still shown after a reload")
    ctx.shot(flow, "cohort-save")


# --------------------------------------------------------------------------- flow (g)

def flow_g(ctx: Ctx, flow: H.Flow) -> None:
    """Save an annotation, reload, confirm it is still bound to the same query."""
    run = ctx.comp_run(status="COMPLETED", failed=False)
    if not require(flow, run is not None, "g.composition_run", "a completed composition run"):
        return
    rid = run["run_id"]
    qid = ((ctx.comp_query_page(rid, "sf").get("items") or [{}])[0]).get("query_id") or ""
    if not require(flow, bool(qid), "g.query_available", f"a query id is served for {rid[:12]}"):
        return
    protocol = "circus_fashioniq_release_v1"
    flow.evidence["annotating"] = {"run_id": rid, "query_id": qid, "protocol": protocol}

    ctx.goto(area_url(ctx.ui, "notebook", "annotations", protocol=protocol, query=qid))
    ctx.settle(2000)
    note = H.optional(ctx.page, "annot.note")
    save = H.optional(ctx.page, "annot.save")
    if note is None or save is None:
        flow.miss(H.SelectorMiss("annot.note", ["no annotation form matched"]), "save annotation")
        flow.blocked("g.annotation_bound_after_reload",
                     "blocked: the annotation form did not render")
        return
    note_text = f"qa annotation {now_stamp()}"
    note.first.fill(note_text)
    save.first.click()
    ctx.settle(2500)
    server = ctx.api.get(f"/api/annotations?query_id={urllib.parse.quote(qid)}"
                         f"&protocol_id={protocol}") or []
    persisted = any(r.get("note") == note_text for r in server)
    feedback = " ".join(H.texts(ctx.page.locator("[role='alert'],[role='status']")))
    flow.evidence["annotation_save_feedback"] = feedback[:300]
    flow.evidence["annotation_server_rows"] = len(server)
    # Save either persists, or says why it refused. A silent no-op is the only unacceptable outcome.
    flow.check("g.annotation_persisted_or_refused", persisted or bool(feedback),
               f"save persisted ({persisted}) or explained its refusal: {feedback[:180]!r}")
    if not persisted:
        for ident in ("g.annotation_note_restored", "g.annotation_bound_after_reload"):
            flow.blocked(ident, "blocked: the backend refuses annotations for a CIRCUS release "
                                "query id, so there is nothing to re-read after a reload")
        bound = H.optional(ctx.page, "annot.query")
        bound_text = " ".join(H.texts(bound)) if bound is not None else ""
        flow.check("g.annotation_form_bound_to_query", qid in bound_text,
                   f"the form names the query it would annotate: {bound_text[:110]!r}")
        ctx.shot(flow, "annotation-refused")
        ctx.browser_evidence(flow)
        return

    ctx.page.reload(wait_until="domcontentloaded")
    ctx.settle(2500)
    # Same probe defect as flow a: `innerText` is always "" for a <textarea> in Chromium, so the
    # note must be read from the control's VALUE, which is where a textarea keeps its content.
    restored = [ctx.page.locator("textarea").first.input_value()]
    flow.check("g.annotation_note_restored", note_text in restored,
               f"note textarea value after reload: {str(restored[:1])[:120]}")
    bound = H.optional(ctx.page, "annot.query")
    bound_text = " ".join(H.texts(bound)) if bound is not None else ""
    flow.check("g.annotation_bound_after_reload", qid in bound_text,
               f"reloaded note names query {qid}: {bound_text[:120]!r}")
    flow.check("g.annotation_protocol_bound", protocol in bound_text,
               f"reloaded note names its protocol (a query id alone is ambiguous between the "
               f"CIRCUS release and the FashionIQ index): {bound_text[:160]!r}")
    ctx.shot(flow, "annotation-saved")
    ctx.browser_evidence(flow)


# --------------------------------------------------------------------------- flow (h)

#: Areas checked for layout: the shell plus every data-bearing surface.
LAYOUT_SURFACES = ("overview", "experiments", "evaluation", "samples", "failures", "notebook")


def layout_urls(ctx: Ctx) -> list[tuple[str, str]]:
    urls = [(area, area_url(ctx.ui, area)) for area in LAYOUT_SURFACES]
    run = ctx.comp_run(status="COMPLETED", failed=False)
    if run is not None:
        qid = ((ctx.comp_query_page(run["run_id"], "sf").get("items") or [{}])[0]).get("query_id")
        urls.append(("explorer-query",
                     ctx.explorer_url(run["run_id"], "sf", query=qid or "")))
    return urls


def flow_h(ctx: Ctx, flow: H.Flow) -> None:
    """Layout checks at the current viewport."""
    for name, url in layout_urls(ctx):
        ctx.goto(url)
        ctx.settle(1500)
        overflow = ctx.page.evaluate(H.PROBE_OVERFLOW)
        flow.evidence.setdefault("overflow", {})[name] = overflow
        flow.overflow_px = overflow.get("overflowPx")
        flow.check(f"h.{name}.no_page_overflow", overflow["overflowPx"] <= 1,
                   f"{overflow['overflowPx']}px horizontal overflow at {ctx.viewport}; "
                   f"offenders: {[(o['el'], o['over']) for o in overflow['offenders'][:3]]}")
        targets = ctx.page.evaluate(H.PROBE_TARGETS)
        flow.evidence.setdefault("click_targets", {})[name] = {
            "total": targets["total"], "small_count": targets["smallCount"],
            "examples": targets["small"][:8]}
        examples = [(t["tag"], t["label"][:18], t["w"], t["h"]) for t in targets["small"][:4]]
        flow.check(f"h.{name}.click_targets", targets["smallCount"] == 0,
                   f"{targets['smallCount']} of {targets['total']} interactive element(s) under "
                   f"24x24 CSS px: {examples}")
        images = ctx.page.evaluate(H.PROBE_IMAGES)
        distorted = [i for i in images if i["ratioDrift"] is not None and i["ratioDrift"] > 0.02]
        broken = [i for i in images if i["visible"] and not i["loaded"]]
        flow.evidence.setdefault("images", {})[name] = {
            "count": len(images), "distorted": len(distorted), "unloaded_onscreen": len(broken)}
        flow.check(f"h.{name}.images_not_distorted", not distorted,
                   f"{len(distorted)} of {len(images)} image(s) distorted by object-fit:fill "
                   f"beyond 2% aspect drift: "
                   f"{[(i['src'][:40], i['natural'], i['rendered']) for i in distorted[:3]]}")
        flow.check(f"h.{name}.images_loaded", not broken,
                   f"{len(broken)} on-screen image(s) failed to load: "
                   f"{[i['src'][:50] for i in broken[:3]]}", category="evidence")
        sticky = ctx.page.evaluate(H.PROBE_STICKY)
        flow.evidence.setdefault("sticky", {})[name] = sticky
        flow.check(f"h.{name}.sticky_no_occlusion", not sticky["occluded"],
                   f"{len(sticky['occluded'])} text node(s) covered by a sticky/fixed element: "
                   f"{[o['text'][:30] for o in sticky['occluded'][:3]]}")

    # focus visibility on tab-through
    ctx.goto(area_url(ctx.ui, "evaluation", "composition"))
    ctx.settle(1500)
    ctx.page.keyboard.press("Tab")
    invisible, stops = [], []
    for _ in range(30):
        info = ctx.page.evaluate(H.PROBE_FOCUS)
        stops.append(info)
        if info.get("tag") and not info.get("visible"):
            invisible.append(info)
        ctx.page.keyboard.press("Tab")
    flow.evidence["focus"] = {"stops": len(stops),
                              "invisible": invisible[:6],
                              "labels": [s.get("label") for s in stops[:10]]}
    flow.check("h.focus_visible", not invisible,
               f"{len(invisible)} of {len(stops)} tab stop(s) have no visible focus indicator: "
               f"{[(s['tag'], s['label'][:20]) for s in invisible[:4]]}")
    ctx.shot(flow, "focus", full_page=False)
    ctx.browser_evidence(flow)


# --------------------------------------------------------------------------- baseline

#: Screenshot set for the "before" record. Entries are (name, area, view) so the capture keeps
#: working when a view is renamed; the composition/inspector pair also needs the selected run.
BASELINE_AREAS = (
    ("dashboard", "overview", None),
    ("evaluation-runs", "experiments", "runs"),
    ("composition-table", "evaluation", "composition"),
    ("query-inspector", "samples", "explorer"),
)


def capture_baseline(ctx: Ctx, out: Path) -> list[dict]:
    """Real screenshots of the served app at the current viewport: before-<page>-<viewport>.png."""
    shots = []
    run = ctx.comp_run(status="COMPLETED", failed=False)
    rid = run["run_id"] if run else ""
    qid = ""
    if run:
        qid = ((ctx.comp_query_page(rid, "sf").get("items") or [{}])[0]).get("query_id") or ""
    for page_name, area, view in BASELINE_AREAS:
        if area == "samples":
            if not run:
                shots.append({"page": page_name, "viewport": ctx.viewport, "skipped":
                              "no completed composition run"})
                continue
            url = ctx.explorer_url(rid, "sf", query=qid)
        elif area == "evaluation":
            if not run:
                shots.append({"page": page_name, "viewport": ctx.viewport, "skipped":
                              "no completed composition run"})
                continue
            url = ctx.evaluation_url(rid, "sf")
        else:
            url = area_url(ctx.ui, area, view)
        path = out / f"before-{page_name}-{ctx.viewport}.png"
        try:
            ctx.goto(url, settle_ms=1800)
            ctx.page.screenshot(path=str(path), full_page=True)
            probe = ctx.page.evaluate(H.PROBE_OVERFLOW)
            shots.append({"page": page_name, "viewport": ctx.viewport, "url": url,
                          "screenshot": str(path.relative_to(out)),
                          "overflow_px": probe["overflowPx"]})
        except Exception as exc:
            shots.append({"page": page_name, "viewport": ctx.viewport, "url": url,
                          "screenshot": None, "error": str(exc)[:300]})
    return shots


# --------------------------------------------------------------------------- main

FLOWS = {
    "a": ("end-to-end", flow_a),
    "b": ("race", flow_b),
    "c": ("degraded", flow_c),
    "d": ("navigation", flow_d),
    "e": ("artifacts", flow_e),
    "f": ("cohorts", flow_f),
    "g": ("annotations", flow_g),
    "h": ("layout", flow_h),
}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ui-url", default=os.environ.get("QA_UI_URL", DEFAULT_UI))
    p.add_argument("--api-url", default=os.environ.get("QA_API_URL", DEFAULT_API))
    p.add_argument("--out", type=Path, default=None,
                   help="evidence directory (default workbench/docs/reviews/<timestamp>-qa)")
    p.add_argument("--flows", default="a,b,c,d,e,f,g,h",
                   help="comma-separated subset of a,b,c,d,e,f,g,h")
    p.add_argument("--viewports", default=",".join(VIEWPORTS))
    p.add_argument("--baseline", action="store_true",
                   help="also capture before-*.png for every viewport")
    p.add_argument("--headed", action="store_true")
    p.add_argument("--read-only", action="store_true",
                   help="skip flows f and g (they write a cohort/annotation when reachable)")
    p.add_argument("--timeout", type=int, default=H.NAV_TIMEOUT_MS)
    return p


def write_report(report: dict, out: Path) -> None:
    """Recompute the summary and persist `qa_report.json` (callable after every flow)."""
    results = report.get("results", [])
    by_flow: dict[str, dict[str, Any]] = {}
    for item in results:
        bucket = by_flow.setdefault(item["flow"], {"name": item["name"], "viewports": {},
                                                   "statuses": []})
        bucket["viewports"][item["viewport"]] = item["status"]
        bucket["statuses"].append(item["status"])
    report["summary"] = {
        "total": len(results),
        "pass": sum(1 for r in results if r["status"] == "pass"),
        "fail": sum(1 for r in results if r["status"] == "fail"),
        "selector_miss": sum(1 for r in results if r["status"] == "selector_miss"),
        "blocked": sum(1 for r in results if r["status"] == "blocked"),
        "by_flow": by_flow,
        "failed_assertions": [
            {"flow": r["flow"], "viewport": r["viewport"],
             "id": a["id"], "detail": a["detail"]}
            for r in results for a in r["failed_assertions"]
        ],
        "blocked_items": [
            {"flow": r["flow"], "viewport": r["viewport"], "id": c["id"], "reason": c["detail"]}
            for r in results for c in r["assertions"] if c["ok"] is None
        ],
        "console_error_total": sum(len(r["evidence"].get("console_errors", []))
                                   for r in results),
        "page_error_total": sum(len(r["evidence"].get("page_errors", [])) for r in results),
        "http_ge_400_total": sum(len(r["evidence"].get("http_ge_400", [])) for r in results),
    }
    H.write_json(out / "qa_report.json", report)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    H.ensure_browser_path()
    stamp = now_stamp()
    out = args.out or (REPO / "workbench/docs/reviews" / f"{stamp}-qa")
    out.mkdir(parents=True, exist_ok=True)
    flows_wanted = [f.strip() for f in args.flows.split(",") if f.strip()]
    unknown = [f for f in flows_wanted if f not in FLOWS]
    if unknown:
        print(f"unknown flow id(s): {unknown}", file=sys.stderr)
        return 2
    if args.read_only:
        flows_wanted = [f for f in flows_wanted if f not in ("f", "g")]

    api = H.Api(args.api_url)
    probe = api.get("/api/benchmark-runs")
    if probe is None:
        print(f"backend not reachable at {args.api_url}", file=sys.stderr)
        return 2

    report: dict = {
        "generated_at": stamp,
        "ui_url": args.ui_url,
        "api_url": args.api_url,
        "viewports": [v for v in args.viewports.split(",") if v],
        "flows_requested": flows_wanted,
        "browser": None,
        "environment": {},
        "baseline": [],
        "results": [],
        "selector_usage": {},
        "selector_misses": {},
        "testid_requests": {},
        "summary": {},
    }

    from playwright.sync_api import sync_playwright  # imported here so --help needs no browser

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=not args.headed)
        report["browser"] = {"name": "chromium", "version": browser.version,
                             "headless": not args.headed,
                             "browsers_path": os.environ.get("PLAYWRIGHT_BROWSERS_PATH")}
        report["environment"] = {
            "ui_h1_title": None,
            "nav_labels": None,
            "api_benchmark_runs": len(probe) if isinstance(probe, list) else None,
            "api_runs": len(api.get("/api/runs") or []),
            "api_composition_runs": len(api.get("/api/composition/runs") or []),
        }
        for viewport in report["viewports"]:
            width, height = H.viewport_size(viewport)
            context = browser.new_context(viewport={"width": width, "height": height},
                                          device_scale_factor=1)
            page = context.new_page()
            page.set_default_timeout(15_000)
            capture = H.PageCapture(page)
            ctx = Ctx(page, capture, api, args.ui_url, args.api_url, out, viewport,
                      args.read_only)
            if args.baseline:
                try:
                    report["baseline"].extend(capture_baseline(ctx, out))
                except Exception as exc:
                    report["baseline"].append({"viewport": viewport, "screenshot": None,
                                              "error": f"baseline capture aborted: {exc}"[:300]})
            if report["environment"]["ui_h1_title"] is None:
                ctx.goto(f"{ctx.ui}/", settle_ms=1200)
                h1 = page.locator("h1").first
                report["environment"]["ui_h1_title"] = (
                    h1.inner_text() if h1.count() else None)
                report["environment"]["nav_labels"] = ctx.nav_labels()
            for flow_id in flows_wanted:
                name, fn = FLOWS[flow_id]
                flow = H.Flow(flow_id, name, viewport)
                capture.clear()
                SELECTOR_START = dict(H.SELECTOR_USED)
                try:
                    fn(ctx, flow)
                except Exception:
                    flow.check("harness.exception", False,
                               "harness raised: " + traceback.format_exc(limit=4)[-700:],
                               category="harness_error")
                finally:
                    if not flow.screenshots:
                        ctx.shot(flow, "final")
                    report["results"].append(flow.to_json())
                    # Written after every flow: a browser or server that dies mid-run must not cost
                    # the evidence already collected.
                    write_report(report, out)
                    for key, used in H.SELECTOR_USED.items():
                        if key not in SELECTOR_START:
                            report["selector_usage"][key] = used
                    for miss in flow.misses:
                        entry = report["selector_misses"].setdefault(miss["key"], {
                            "count": 0, "tried": miss["tried"], "purposes": []})
                        entry["count"] += 1
                        entry["purposes"].append(f"{flow.flow_id}/{viewport}: {miss['purpose']}")
            try:
                context.close()
            except Exception:
                pass
        try:
            browser.close()
        except Exception:
            pass

    for key in report["selector_misses"]:
        report["testid_requests"][key] = TESTID_CATALOG.get(
            key, f"`data-testid` hook for selector `{key}`")
    write_report(report, out)
    print(json.dumps({"out": str(out),
                      "summary": {k: v for k, v in report["summary"].items()
                                  if k not in ("by_flow", "failed_assertions",
                                               "blocked_items")}}, ensure_ascii=False))
    for flow_id, bucket in report["summary"]["by_flow"].items():
        print(f"  {flow_id} {bucket['name']:<12} " +
              " ".join(f"{vp}={st}" for vp, st in bucket["viewports"].items()))
    return 1 if report["summary"]["fail"] else 0


if __name__ == "__main__":
    sys.exit(main())
