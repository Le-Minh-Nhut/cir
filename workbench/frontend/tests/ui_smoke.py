#!/usr/bin/env python3
"""Dev-only DOM-verified UI smoke for the CIR Failure Analysis Workbench (4e61f2f page split).

Never reads pixels: every check is a locator, textContent, computed style or boundingBox.
Screenshots are SAVED for the human operator and never opened here.

Modes
  --build-fixture DIR   write two protocol-compatible mock results + real small PNGs, then exit
  --url URL             run the smoke against an already-served workbench
                        (--session cold | fixture | composition | mobile | all; default all)

Navigation goes through the shell's own affordances -- `[data-testid=area-nav-<area>]` and
`[data-testid=view-<view>]`, the ids routes.ts AREAS declares -- never through a text label, so a
relabelled area cannot silently re-point a session at another page.

Sessions and what they need
  cold         any stack; read-only. Registry count == len(/api/models), one model row each, a
               blockers disclosure whose summary is "<n> lý do" when the API declares reasons.
  fixture      the fixture stack (protocol fashioniq_val_split). Writes nothing. Requires the two
               published index runs to be LISTED as `index` rows with their real run_ids/model_ids
               from /api/runs even though no composition row exists. Then the injected-failure +
               recovery flow.
  composition  a stack that serves CIRCUS composition rows (production). Table fits its
               `.table-wrap` at 1440x900 and 1920x1080, expanding a detail neither widens the table
               nor triples the document height nor produces a 60px cell holding 200+ chars, and the
               benchmark view never dead-ends. Prints SKIP with the reason when the stack serves 0
               composition rows for the selected protocol/subset.
  mobile       390x844. Nav reachable, `body.scrollWidth <= innerWidth + 2`, every model row's 5th
               cell (the blocker column) inside its scroll box, and on a stack with composition
               data the composition table contained in its scroll box with no page overflow.

Flows removed with the 4e61f2f page split, and why
  * `open_query` / `wait_for_pager_ready` on index runs, `div.filters`, `div.pager`, `div.labels`,
    `aside.error`, "Save filtered set", and the two-run comparison article flow all named DOM that
    no longer exists (the explorer is now a URL-state page with `data-testid` controls).
  * The COHORT flow is dropped: the cohort write needs a per-query universe, and on the fixture
    stack `/api/queries?limit=20000` answers 400 `invalid_pagination` (storage.py caps limit at
    200), so Phân tích lỗi renders an error state and `[data-testid=cohort-save]` never mounts.
    On production the same button refuses CIRCUS release runs by contract (FailuresPage.saveCohort).
    There is no live stack on which the flow is drivable end-to-end today.
  * The ANNOTATION flow is dropped for the same reason: the notebook writes are reachable only
    through an inspector query, and the inspector needs a selected composition run, which the
    fixture stack does not serve. Re-add it when a stack serves both a composition run and its
    images; it is one URL, one textarea, one save button.

Browsers: PLAYWRIGHT_BROWSERS_PATH must point at the installed chromium.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

PROTOCOL = "fashioniq_val_split"
CIRCUS_PROTOCOL = "circus_fashioniq_release_v1"
CATEGORIES = ("dress", "shirt", "toptee")
TOP_K_SAVED = 20
VIEWPORT_DESKTOP = {"width": 1440, "height": 900}
VIEWPORT_WIDE = {"width": 1920, "height": 1080}
VIEWPORT_MOBILE = {"width": 390, "height": 844}
NAV_TIMEOUT_MS = 120_000

# --------------------------------------------------------------------------- fixture

def _queries(category: str, ranks: list[int], top_depth: int) -> list:
    from workbench.replay.limn_observer import ObservedQuery

    out = []
    for index, rank in enumerate(ranks):
        reference, target = f"{category}ref{index}", f"{category}tgt{index}"
        top = []
        for position in range(1, top_depth + 1):
            if position == rank:
                image_id = target
            elif position == 1:
                image_id = "shareddistractor"
            else:
                image_id = f"{category}d{index}_{position}"
            top.append((position, image_id, round(1.0 - position / 100.0, 6)))
        out.append(ObservedQuery(
            annotation_index=index, reference_id=reference, target_id=target,
            raw_captions=("change one", "detail two"),
            model_input_text=f"{category} change one and detail two",
            target_rank=rank, top_results=tuple(top), category=category,
        ))
    return out

def _image_ids(category: str, top_depth: int) -> list[str]:
    ids = ["shareddistractor"]
    for index in range(12):
        ids += [f"{category}ref{index}", f"{category}tgt{index}"]
        ids += [f"{category}d{index}_{position}" for position in range(2, top_depth + 1)]
    return ids

def build_fixture(results_root: Path, images_root: Path) -> dict:
    """Two compatible mock runs (identical query universe, different ranks) + real PNGs.

    Twelve queries per category (24 per run) so the explorer's 20-row page really has a second page.
    """
    from PIL import Image

    from workbench.replay.limn_observer import ObservedCategory
    from workbench.replay.publication import build_result_run, publish

    ranks_a = {
        "dress": [1, 2, 12, 30, 4, 9, 45, 7, 21, 3, 40, 11],
        "shirt": [2, 15, 59, 5, 20, 1, 33, 8, 44, 6, 17, 28],
    }
    ranks_b = {
        "dress": [1, 7, 12, 30, 45, 3, 2, 9, 4, 21, 11, 40],
        "shirt": [59, 2, 4, 20, 15, 33, 1, 8, 6, 44, 28, 17],
    }
    sizes = {"dress": 50, "shirt": 60}
    published = {}
    for label, ranks in (("fixture-a", ranks_a), ("fixture-b", ranks_b)):
        categories = [
            ObservedCategory(category=category, gallery_size=sizes[category],
                             eligible_count=sizes[category] - 1,
                             queries=tuple(_queries(category, ranks[category], TOP_K_SAVED)))
            for category in ("dress", "shirt")
        ]
        run = build_result_run(
            categories, run_id=f"dev-{label}-{PROTOCOL}-1", model_id=label,
            method_name=label.upper(), checkpoint_id=f"{label}-ckpt", protocol_id=PROTOCOL,
            top_k_saved=TOP_K_SAVED, data_kind="mock")
        published[label] = str(publish(run, results_root))

    for category in ("dress", "shirt"):
        (images_root / category).mkdir(parents=True, exist_ok=True)
        for image_id in _image_ids(category, TOP_K_SAVED):
            Image.new("RGB", (8, 8), color=(32, 120, 64)).save(
                images_root / category / f"{image_id}.png", format="PNG")
    return published

# --------------------------------------------------------------------------- helpers

class Recorder:
    """Console errors / page errors / failed requests, with the injected failure separable.

    Browser console text for an aborted fetch is a generic `net::ERR_FAILED` with no URL, so the
    events raised *while a route abort is armed* are tagged at the moment they arrive.
    """

    def __init__(self) -> None:
        self.console: list[str] = []
        self.console_injected: list[str] = []
        self.page_errors: list[str] = []
        self.failed_requests: list[str] = []
        self.failed_requests_injected: list[str] = []
        self.injected: list[str] = []
        self.injecting = False
        self.observations: list[str] = []
        self.checks = 0

    def attach(self, page) -> None:
        page.on("console", lambda m: self._console(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: self.page_errors.append(str(e)))
        page.on("requestfailed", lambda r: self._request(
            f"{r.method} {r.url} :: {r.failure}"))

    def _console(self, text: str) -> None:
        (self.console_injected if self.injecting else self.console).append(text)

    def _request(self, text: str) -> None:
        (self.failed_requests_injected if self.injecting else self.failed_requests).append(text)

    def note(self, text: str, check: bool = True) -> None:
        self.observations.append(text)
        if check:
            self.checks += 1
        print(f"[DOM-VERIFIED] {text}", flush=True)

    def skip(self, text: str) -> None:
        print(f"[SKIP] {text}", flush=True)

def api_json(base_url: str, path: str):
    with urllib.request.urlopen(f"{base_url}{path}", timeout=NAV_TIMEOUT_MS // 1000) as response:
        return json.loads(response.read().decode("utf-8"))

def open_page(page, area: str, view: str | None = None) -> None:
    """Open (area, view) through the shell's own test ids, then wait for the area to render.

    On a narrow viewport the sidebar is an off-canvas drawer: the nav button exists but is
    translated out of the viewport, so it must be opened first. `.menu-toggle` is the shell's
    affordance and is display:none on desktop, so asking for it is free there.
    """
    drawer = page.evaluate("""() => {
        const nav = document.querySelector('[data-testid=workbench-nav]');
        const rect = nav.getBoundingClientRect();
        return {open: nav.dataset.open === 'true', offscreen: rect.right <= 1 || rect.left < -1};
    }""")
    if drawer["offscreen"] and not drawer["open"]:
        page.locator(".menu-toggle").first.click()
        page.wait_for_function(
            "() => document.querySelector('[data-testid=workbench-nav]')?.dataset.open === 'true'",
            timeout=NAV_TIMEOUT_MS)
    page.locator(f"[data-testid=area-nav-{area}]").first.click()
    page.wait_for_function(
        "area => new URLSearchParams(location.search).get('area') === area",
        arg=area, timeout=NAV_TIMEOUT_MS)
    if view is not None:
        page.locator(f"[data-testid=view-{view}]").first.click()
        page.wait_for_function(
            "view => new URLSearchParams(location.search).get('view') === view",
            arg=view, timeout=NAV_TIMEOUT_MS)

def goto_view(page, base_url: str, area: str, view: str | None = None,
              query: str = "") -> None:
    """Deep-link one view through the URL, then open it through the nav (proving the nav agrees)."""
    page.goto(f"{base_url}{query}", wait_until="domcontentloaded")
    open_page(page, area, view)

def set_protocol(page, protocol_id: str, timeout_ms: int = NAV_TIMEOUT_MS) -> bool:
    """Pick the protocol in the global control. Returns False when the stack does not serve it."""
    select = page.locator("#protocol-select")
    select.wait_for(timeout=timeout_ms)
    page.wait_for_function(
        "id => [...document.querySelectorAll('#protocol-select option')].some(o => o.value === id)",
        arg=protocol_id, timeout=timeout_ms)
    select.select_option(protocol_id)
    page.wait_for_function(
        "id => document.querySelector('#protocol-select').value === id",
        arg=protocol_id, timeout=timeout_ms)
    return True

def set_subset(page, subset: str) -> None:
    select = page.locator("#subset-select")
    select.wait_for(timeout=NAV_TIMEOUT_MS)
    select.select_option(subset)
    page.wait_for_function(
        "value => document.querySelector('#subset-select').value === value",
        arg=subset, timeout=NAV_TIMEOUT_MS)

def registry_stat(page) -> dict:
    """The overview registry card's own counts: {entry, blocked}, read as numbers.

    The value must be a real digit string, never a placeholder: the card renders `—` while the
    fetch is in flight, and a placeholder that happens to parse is exactly the lie this waits out.
    """
    return page.evaluate("""() => {
        const out = {};
        for (const stat of document.querySelectorAll('[data-testid=overview-cards] .stat')) {
            const label = (stat.querySelector('.stat-label')?.textContent || '').trim();
            const value = (stat.querySelector('.stat-value')?.textContent || '').trim();
            out[label] = value;
        }
        return out;
    }""")

def shot(page, shots: Path, name: str) -> str:
    shots.mkdir(parents=True, exist_ok=True)
    path = shots / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return str(path)

def overflow(page) -> dict:
    return page.evaluate("""() => ({
        scrollWidth: document.body.scrollWidth, innerWidth: window.innerWidth,
        documentScrollWidth: document.documentElement.scrollWidth})""")

def composition_fit(page) -> dict | None:
    """The composition table's own scroll width against the box it lives in."""
    return page.evaluate("""() => {
        const table = document.querySelector('[data-testid=composition-table]');
        if (!table) return null;
        const wrap = table.closest('.table-wrap');
        const wide = [...table.querySelectorAll('td')]
            .filter(td => td.getBoundingClientRect().width < 60 && td.textContent.length > 200).length;
        return {tableScroll: table.scrollWidth, tableClient: table.clientWidth,
                wrapClient: wrap ? wrap.clientWidth : null, wrapScroll: wrap ? wrap.scrollWidth : null,
                rows: table.querySelectorAll('tbody tr').length,
                narrowLongCells: wide, documentHeight: document.documentElement.scrollHeight};
    }""")

def open_row_detail(page) -> str:
    """Expand the first row's detail. Current build uses a `DetailToggle` outside the table; the
    build still serving `<details>` inside the last cell is driven through its summary."""
    toggle = page.locator("[data-testid=composition-table] tbody tr button[aria-expanded]").first
    if toggle.count():
        toggle.click()
        page.wait_for_selector("[data-testid=composition-detail]", timeout=NAV_TIMEOUT_MS)
        return "DetailToggle row"
    summary = page.locator("[data-testid=composition-table] tbody details summary").first
    summary.wait_for(timeout=NAV_TIMEOUT_MS)
    summary.click()
    return "details summary in cell"

def recover_from_injected_failure(page, base_url: str, recorder: Recorder, shots: Path) -> None:
    """One bounded injected API failure -> visible error region -> retry -> recovery.

    The registry card is the region re-fetched: it is present on every stack (no composition row
    needed), it renders `ErrorState` with a retry control when its fetch fails, and the retry
    re-issues the same request, so a cached answer cannot fake the recovery.
    """
    pattern = "**/api/models"

    def abort_models(route):
        recorder.injected.append(f"INJECTED abort {route.request.url}")
        route.abort()

    open_page(page, "overview")
    expected = registry_stat(page)["entry"]
    recorder.injecting = True
    page.route(pattern, abort_models)
    page.reload(wait_until="domcontentloaded")
    error = page.locator("[data-testid=overview-cards] [data-testid=empty-state][data-empty-kind=error]").first
    error.wait_for(timeout=NAV_TIMEOUT_MS)
    recorder.injecting = False
    error_text = error.inner_text()
    retry = page.locator("[data-testid=overview-cards] [data-testid=error-retry]").first
    recorder.note(f"injected failure surfaced as {error_text.splitlines()[0]!r}; "
                  f"retry control present={retry.count() == 1}")
    assert retry.count() == 1, "the failed region offers no retry control"
    shot(page, shots, "error-state")

    page.unroute(pattern, abort_models)
    retry.click()
    page.wait_for_function(
        """() => document.querySelectorAll('[data-testid=overview-cards] [data-testid=empty-state]').length === 0
              && [...document.querySelectorAll('[data-testid=overview-cards] .stat')]
                   .some(s => (s.querySelector('.stat-label')?.textContent || '').trim() === 'entry'
                              && /^[0-9][0-9,]*$/.test((s.querySelector('.stat-value')?.textContent || '').trim()))""",
        timeout=NAV_TIMEOUT_MS)
    recovered = registry_stat(page)["entry"]
    recorder.note(f"after unroute+retry the error region is gone and the registry re-fetched: "
                  f"entry {expected!r} -> {recovered!r}")
    assert recovered == expected, (expected, recovered)

# --------------------------------------------------------------------------- sessions

def session_cold(page, base_url: str, shots: Path, recorder: Recorder) -> list[str]:
    """Any stack, read-only: registry count == /api/models, one row each, truthful blockers."""
    saved = []
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector("[data-testid=area-overview]", timeout=NAV_TIMEOUT_MS)
    models = api_json(base_url, "/api/models")
    # The registry card is the UI's own count: wait for a DIGIT, not for a placeholder.
    page.wait_for_function(
        """() => [...document.querySelectorAll('[data-testid=overview-cards] .stat')]
                 .some(s => (s.querySelector('.stat-label')?.textContent || '').trim() === 'entry'
                            && /^[0-9][0-9,]*$/.test((s.querySelector('.stat-value')?.textContent || '').trim()))""",
        timeout=NAV_TIMEOUT_MS)
    stats = registry_stat(page)
    rendered = int(stats["entry"].replace(",", ""))
    recorded = int((stats.get("bị chặn") or "0").replace(",", ""))
    recorder.note(f"cold overview: registry card reads entry={stats['entry']!r} "
                  f"bị chặn={stats.get('bị chặn')!r}; API /api/models returned {len(models)} entries")
    assert rendered == len(models), f"overview count {rendered} != API {len(models)}"

    open_page(page, "experiments", "models")
    rows = page.locator("[data-testid^=model-row-]")
    rows.first.wait_for(timeout=NAV_TIMEOUT_MS)
    page.wait_for_function(
        "n => document.querySelectorAll('[data-testid^=model-row-]').length === n",
        arg=len(models), timeout=NAV_TIMEOUT_MS)
    row_count = rows.count()
    recorder.note(f"cold models: {row_count} rendered rows for {len(models)} API entries")
    assert row_count == len(models), (row_count, len(models))

    blocked = {}
    for model in models:
        reasons = []
        for checkpoint in model.get("checkpoints") or []:
            reasons += checkpoint.get("blocking_reasons") or []
        if reasons:
            blocked[model["model_id"]] = reasons
    summaries = page.evaluate("""() => [...document.querySelectorAll('[data-testid^=model-row-]')]
        .map(row => { const d = row.querySelector('details > summary');
                      return {id: row.dataset.testid, has: !!d, text: d ? d.textContent.trim() : null}; })""")
    shown = [item for item in summaries if item["has"]]
    recorder.note(f"cold models: {len(blocked)} of {len(models)} models declare blocking_reasons; "
                  f"{len(shown)} rows render a blockers disclosure, e.g. "
                  f"{[item['text'] for item in shown][:3]}")
    if blocked:
        assert shown, "the API declares blocking reasons but no row renders a blockers disclosure"
        import re
        for item in shown:
            assert re.fullmatch(r"\d+\s+lý do", item["text"] or ""), (
                f"{item['id']}: disclosure summary is {item['text']!r}, expected '<n> lý do'")
    else:
        recorder.skip("cold models: no model declares blocking_reasons on this stack")

    saved.append(shot(page, shots, "desktop-inventory"))
    return saved

def session_fixture(page, base_url: str, shots: Path, recorder: Recorder) -> list[str]:
    """Fixture stack only: index runs are catalogued without composition rows; then error recovery."""
    saved = []
    runs = api_json(base_url, "/api/runs")
    models = api_json(base_url, "/api/models")
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector("[data-testid=area-overview]", timeout=NAV_TIMEOUT_MS)
    set_protocol(page, PROTOCOL)
    recorder.note(f"fixture: /api/runs holds {len(runs)} runs "
                  f"({[r['run_id'] for r in runs]}), all protocol {PROTOCOL}")

    open_page(page, "experiments", "runs")
    page.wait_for_function(
        "n => document.querySelectorAll('[data-testid^=run-row-]').length >= n",
        arg=len(runs), timeout=NAV_TIMEOUT_MS)
    listed = page.evaluate("""() => [...document.querySelectorAll('[data-testid^=run-row-]')]
        .map(row => ({id: row.dataset.testid,
                      badges: [...row.querySelectorAll('.badge')].map(b => b.textContent.trim()),
                      text: row.textContent}))""")
    for run in runs:
        match = next((item for item in listed if item["id"] == f"run-row-i:{run['run_id']}"), None)
        assert match is not None, f"{run['run_id']} is not listed on experiments/runs: {listed}"
        assert "index" in match["badges"], f"{run['run_id']} carries badges {match['badges']}"
        assert run["model_id"] in match["text"], f"{run['run_id']}: model_id not rendered: {match}"
    recorder.note(f"fixture runs: {len(runs)} index runs listed with badge 'index', real run_ids "
                  f"and model_ids {[r['model_id'] for r in runs]}")
    assert len(runs) == 2, f"expected 2 published fixture index runs, got {len(runs)}"
    saved.append(shot(page, shots, "desktop-runs"))

    # Index runs have no per-query page in the explorer: SamplesPage lists only `composition`
    # catalog entries, so selecting one here would be a flow the UI does not offer.
    recover_from_injected_failure(page, base_url, recorder, shots)

    recorder.note(f"fixture errors: unexpected console={len(recorder.console)} "
                  f"page_errors={len(recorder.page_errors)} failed_requests={len(recorder.failed_requests)}"
                  f" (injected window: console={len(recorder.console_injected)} "
                  f"failed_requests={len(recorder.failed_requests_injected)}, "
                  f"{len(recorder.injected)} route aborts)")
    assert not recorder.console, recorder.console
    assert not recorder.page_errors, recorder.page_errors
    assert not recorder.failed_requests, recorder.failed_requests
    return saved

def session_composition(browser, base_url: str, shots: Path, recorder: Recorder) -> list[str]:
    """Production stack: the composition table fits, expanding a row does not deform it, and the
    benchmark view always names a next step. SKIPs (with its reason) on a stack with 0 rows."""
    saved = []
    runs = api_json(base_url, "/api/composition/runs")
    served = [run for run in runs
              if run.get("protocol_id") == CIRCUS_PROTOCOL and run.get("subset") == "full"]
    if not served:
        recorder.skip(f"composition: {base_url} serves 0 composition rows for "
                      f"{CIRCUS_PROTOCOL}/full (it has {len(runs)} rows in other scopes), so the "
                      f"fit and row-expansion checks do not apply here")
        return saved

    for viewport in (VIEWPORT_DESKTOP, VIEWPORT_WIDE):
        name = f"{viewport['width']}x{viewport['height']}"
        context = browser.new_context(viewport=viewport)
        page = context.new_page()
        recorder.attach(page)
        try:
            goto_view(page, base_url, "evaluation", "composition",
                      f"/?area=evaluation&view=composition&protocol={CIRCUS_PROTOCOL}&subset=full")
            page.wait_for_selector("[data-testid=composition-table]", timeout=NAV_TIMEOUT_MS)
            page.wait_for_function(
                "n => document.querySelectorAll('[data-testid=composition-table] tbody tr').length === n",
                arg=len(served), timeout=NAV_TIMEOUT_MS)
            fit = composition_fit(page)
            recorder.note(f"composition {name}: {fit['rows']} rows, table scrollWidth "
                          f"{fit['tableScroll']} vs .table-wrap clientWidth {fit['wrapClient']} "
                          f"(narrow cells with 200+ chars: {fit['narrowLongCells']})")
            assert fit["tableScroll"] <= fit["wrapClient"], (
                f"{name}: composition table {fit['tableScroll']}px exceeds its box "
                f"{fit['wrapClient']}px -- the last columns are clipped, not scrollable")

            collapsed_fit, collapsed = fit, fit["documentHeight"]
            how = open_row_detail(page)
            page.wait_for_timeout(400)
            expanded = composition_fit(page)
            recorder.note(f"composition {name}: expanded one row via {how}; table scrollWidth "
                          f"{fit['tableScroll']} -> {expanded['tableScroll']}, document height "
                          f"{collapsed} -> {expanded['documentHeight']} "
                          f"({expanded['documentHeight'] / max(collapsed, 1):.2f}x), "
                          f"narrow 200+ char cells: {expanded['narrowLongCells']}")
            assert expanded["tableScroll"] <= collapsed_fit["tableScroll"], (
                f"{name}: expanding the detail widened the table "
                f"{collapsed_fit['tableScroll']} -> {expanded['tableScroll']}")
            assert expanded["documentHeight"] <= 3 * collapsed, (
                f"{name}: expanding the detail grew the document "
                f"{collapsed['documentHeight']} -> {expanded['documentHeight']} (>3x)")
            assert expanded["narrowLongCells"] == 0, (
                f"{name}: {expanded['narrowLongCells']} cells narrower than 60px hold 200+ chars")
            saved.append(shot(page, shots, f"composition-{name}"))

            open_page(page, "evaluation", "benchmark")
            page.wait_for_function(
                "() => !!document.querySelector('[data-testid=area-benchmark]')",
                timeout=NAV_TIMEOUT_MS)
            state = page.evaluate("""() => {
                const rows = document.querySelectorAll('[data-testid=area-benchmark] tbody tr').length;
                const empty = document.querySelector('[data-testid=benchmark-empty]');
                const named = [...document.querySelectorAll('[data-testid^=benchmark-goto-]')]
                    .map(b => b.textContent.trim());
                return {rows, empty: empty ? empty.textContent : null, named};
            }""")
            recorder.note(f"composition {name}: benchmark view has {state['rows']} report rows; "
                          f"protocols named in the empty state: {state['named']}")
            assert state["rows"] > 0 or any(
                item.strip().endswith("report)") or " report" in item for item in state["named"]), (
                "the benchmark view dead-ends: no report rows and no protocol named with a count")
            saved.append(shot(page, shots, f"benchmark-{name}"))
        finally:
            context.close()
    assert not recorder.console, recorder.console
    assert not recorder.page_errors, recorder.page_errors
    return saved

def session_mobile(browser, base_url: str, shots: Path, recorder: Recorder) -> list[str]:
    """390x844: nav reachable, no page overflow, model blocker column inside its scroll box."""
    context = browser.new_context(viewport=VIEWPORT_MOBILE, is_mobile=True, has_touch=True)
    page = context.new_page()
    recorder.attach(page)
    saved = []
    try:
        page.goto(base_url, wait_until="domcontentloaded")
        page.wait_for_selector("[data-testid=area-overview]", timeout=NAV_TIMEOUT_MS)
        metrics = overflow(page)
        recorder.note(f"mobile: body.scrollWidth={metrics['scrollWidth']} "
                      f"innerWidth={metrics['innerWidth']}")
        assert metrics["scrollWidth"] <= metrics["innerWidth"] + 2, metrics

        # The drawer must actually come into view when opened: a nav button that exists but sits at
        # x=-252 is unreachable, and a bounding box is how that is proved.
        page.locator(".menu-toggle").first.click()
        page.wait_for_function(
            "() => document.querySelector('[data-testid=workbench-nav]')?.dataset.open === 'true'",
            timeout=NAV_TIMEOUT_MS)
        page.wait_for_timeout(300)
        nav = page.evaluate("""() => {
            const items = [...document.querySelectorAll('[data-testid^=area-nav-]')]
                .map(el => { const r = el.getBoundingClientRect();
                             return {id: el.dataset.testid, left: Math.round(r.left),
                                     right: Math.round(r.right), w: Math.round(r.width),
                                     h: Math.round(r.height)}; });
            return {items, innerWidth: window.innerWidth};
        }""")
        off = [item for item in nav["items"]
               if item["left"] < -1 or item["right"] > nav["innerWidth"] + 2
               or item["w"] <= 0 or item["h"] <= 0]
        recorder.note(f"mobile nav: {len(nav['items'])} area buttons, off-viewport={off}")
        assert len(nav["items"]) >= 6, nav
        assert not off, f"nav buttons off-viewport: {off}"

        open_page(page, "experiments", "models")
        models = api_json(base_url, "/api/models")
        page.wait_for_function(
            "n => document.querySelectorAll('[data-testid^=model-row-]').length === n",
            arg=len(models), timeout=NAV_TIMEOUT_MS)
        cells = page.evaluate("""() => [...document.querySelectorAll('[data-testid^=model-row-]')]
            .map(row => { const td = row.children[4]; const wrap = row.closest('.table-wrap');
                const r = td.getBoundingClientRect(); const w = wrap.getBoundingClientRect();
                return {cells: row.children.length, left: Math.round(r.left), right: Math.round(r.right),
                        width: Math.round(r.width), textLength: td.textContent.trim().length,
                        wrapLeft: Math.round(w.left), wrapScroll: wrap.scrollWidth,
                        innerWidth: window.innerWidth}; })""")
        metrics = overflow(page)
        recorder.note(f"mobile models: {len(cells)} rows from /api/models, "
                      f"body.scrollWidth={metrics['scrollWidth']} innerWidth={metrics['innerWidth']}, "
                      f"first blocker cell left={cells[0]['left']} right={cells[0]['right']} "
                      f"width={cells[0]['width']} wrapScroll={cells[0]['wrapScroll']}")
        assert len(cells) == len(models), (len(cells), len(models))
        assert metrics["scrollWidth"] <= metrics["innerWidth"] + 2, metrics
        for index, cell in enumerate(cells):
            assert cell["cells"] == 5, f"model row {index}: {cell['cells']} cells, expected 5"
            # The blocker column must be reachable: inside the horizontal scroll box of its table.
            assert cell["wrapLeft"] <= cell["left"] and \
                cell["right"] - cell["wrapLeft"] <= cell["wrapScroll"] + 1, (
                f"model row {index}: blocker cell is outside its scroll box: {cell}")
            assert cell["width"] > 40 and cell["textLength"] > 0, (
                f"model row {index}: blocker cell collapsed or empty: {cell}")
        hidden = [cell for cell in cells if cell["right"] > cell["innerWidth"] + 2]
        if hidden:
            recorder.skip(f"mobile models: {len(hidden)}/{len(cells)} blocker cells are not inside "
                          f"the 390px viewport at rest (first: left={hidden[0]['left']} "
                          f"right={hidden[0]['right']}); they are reachable by scrolling their "
                          f"table box (scrollWidth={hidden[0]['wrapScroll']}), which is asserted above")
        saved.append(shot(page, shots, "mobile-inventory"))

        runs = api_json(base_url, "/api/composition/runs")
        served = [run for run in runs
                  if run.get("protocol_id") == CIRCUS_PROTOCOL and run.get("subset") == "full"]
        if not served:
            recorder.skip("mobile composition: this stack serves 0 composition rows for "
                          f"{CIRCUS_PROTOCOL}/full, so there is no table to contain")
        else:
            goto_view(page, base_url, "evaluation", "composition",
                      f"/?area=evaluation&view=composition&protocol={CIRCUS_PROTOCOL}&subset=full")
            page.wait_for_selector("[data-testid=composition-table]", timeout=NAV_TIMEOUT_MS)
            fit = composition_fit(page)
            metrics = overflow(page)
            recorder.note(f"mobile composition: table scrollWidth {fit['tableScroll']} inside "
                          f".table-wrap clientWidth {fit['wrapClient']} / scrollWidth "
                          f"{fit['wrapScroll']}; page body.scrollWidth={metrics['scrollWidth']} "
                          f"innerWidth={metrics['innerWidth']}")
            assert fit["wrapScroll"] >= fit["tableScroll"], fit
            assert metrics["scrollWidth"] <= metrics["innerWidth"] + 2, metrics
            saved.append(shot(page, shots, "mobile-composition"))
        assert not recorder.console, recorder.console
        assert not recorder.page_errors, recorder.page_errors
    finally:
        context.close()
    return saved

# --------------------------------------------------------------------------- main

def launch(playwright, mobile: bool = False):
    chrome = os.environ.get("WORKBENCH_CHROME_PATH")
    if not chrome and Path("/usr/bin/google-chrome").exists():
        chrome = "/usr/bin/google-chrome"
    kwargs = {"executable_path": chrome} if chrome else {"channel": "chrome"}
    return playwright.chromium.launch(**kwargs)

def run_sessions(url: str, shots: Path, which: str) -> Recorder:
    from playwright.sync_api import sync_playwright

    recorder = Recorder()
    saved: list[str] = []
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            if which in ("cold", "fixture", "all"):
                context = browser.new_context(viewport=VIEWPORT_DESKTOP)
                page = context.new_page()
                recorder.attach(page)
                try:
                    if which in ("cold", "all"):
                        saved += session_cold(page, url, shots, recorder)
                    if which in ("fixture", "all"):
                        saved += session_fixture(page, url, shots, recorder)
                finally:
                    context.close()
            if which in ("composition", "all"):
                saved += session_composition(browser, url, shots, recorder)
            if which in ("mobile", "all"):
                saved += session_mobile(browser, url, shots, recorder)
        finally:
            browser.close()
    recorder.note(f"screenshots saved: {saved}", check=False)
    return recorder

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build-fixture", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--shots", type=Path, default=Path("/tmp/ui-e2e/shots"))
    parser.add_argument("--session",
                        choices=("cold", "fixture", "composition", "mobile", "all"), default="all")
    args = parser.parse_args(argv)

    if args.build_fixture is not None:
        published = build_fixture(args.build_fixture, args.images_root or args.build_fixture.parent / "fashioniq")
        print(f"[FIXTURE] published {published}")
        return 0

    if not args.url:
        parser.error("either --build-fixture or --url is required")
    recorder = run_sessions(args.url, args.shots, args.session)
    summary = {
        "session": args.session,
        "checks": recorder.checks,
        "console_errors_unexpected": recorder.console,
        "console_errors_from_injected_failure": recorder.console_injected,
        "page_errors": recorder.page_errors,
        "failed_requests_unexpected": recorder.failed_requests,
        "failed_requests_from_injected_failure": recorder.failed_requests_injected,
        "injected_failures": recorder.injected,
    }
    print("[SUMMARY] " + json.dumps(summary, indent=2))
    if recorder.console or recorder.failed_requests or recorder.page_errors:
        print("[WARNING] unexpected browser errors present (outside the injected window)",
              file=sys.stderr)
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
