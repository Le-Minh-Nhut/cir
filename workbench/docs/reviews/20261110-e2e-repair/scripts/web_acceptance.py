"""Web surface acceptance: screenshots + machine-checkable assertions.

Run: workbench/envs/ctl312h/bin/python workbench/docs/reviews/20261110-e2e-repair/scripts/web_acceptance.py
Writes: workbench/docs/reviews/20261110-e2e-repair/web_evidence/*.png and raw/web_acceptance.json

Every assertion is about something the page RENDERS, checked against the same number the API serves.
Nothing here restarts the service or writes to the repo outside the review directory.
"""
from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

CHROME = "/home/heheboiz/.cache/ms-playwright/chromium-1254/chrome-linux64/chrome"
BASE = "http://127.0.0.1:5173"
API = "http://127.0.0.1:8000"
ROOT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261110-e2e-repair")
SHOTS = ROOT / "web_evidence"
RAW = ROOT / "raw"
CIRCUS = "circus_fashioniq_release_v1"
DEEP_RUN = "9e4918a6b5c243412cbc58c9"   # limn / full / depth 100
QUERY = "7:3045"                          # target rank 73369 -> outside every window
NAV_MS = 60_000

results: dict[str, object] = {"shots": [], "checks": {}, "timing": {}, "errors": []}
console_errors: list[str] = []


def api(path: str):
    with urllib.request.urlopen(f"{API}{path}", timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def note(key: str, value) -> None:
    results["checks"][key] = value
    print(f"  {key}: {value}", flush=True)


def shot(page, name: str, full: bool = True) -> str:
    path = SHOTS / f"{name}.png"
    page.screenshot(path=str(path), full_page=full)
    results["shots"].append(str(path.relative_to(ROOT.parent.parent.parent.parent)))
    print(f"  shot {name} ({path.stat().st_size} bytes)", flush=True)
    return str(path)


def wait_ready(page) -> None:
    """Readiness = the run list is on screen. NEVER /api/models (its cold start is ~900 s)."""
    page.wait_for_function(
        "() => document.querySelector('.shell')?.getAttribute('data-shell-state') === 'ready'",
        timeout=NAV_MS)


def wait_landed(page) -> None:
    """Readiness PLUS the landing correction: when the URL named no scope, the shell rewrites the URL
    from the served rows with `replaceState`. Reading a selector before that lands compares two
    different scopes; and a bare URL landing must be checked after it, not before."""
    wait_ready(page)
    page.wait_for_function(
        "() => new URLSearchParams(location.search).has('protocol')"
        " && new URLSearchParams(location.search).has('subset')", timeout=NAV_MS)

def main() -> int:
    SHOTS.mkdir(parents=True, exist_ok=True)
    route_errors: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=CHROME, args=["--no-sandbox"])

        def new_page(width: int, height: int, mobile: bool = False, error_log: list[str] | None = None):
            context = browser.new_context(viewport={"width": width, "height": height},
                                          is_mobile=mobile, has_touch=mobile)
            page = context.new_page()
            page.on("console", lambda msg: error_log.append(f"console {msg.type}: {msg.text}")
                    if msg.type == "error" and error_log is not None else None)
            page.on("pageerror", lambda err: error_log.append(f"pageerror: {err}") if error_log is not None else None)
            return context, page

        # ---------------------------------------------------------------- 1. landing / scopes
        context, page = new_page(1920, 1080, error_log=console_errors)
        cold_start = time.time()
        page.goto(BASE + "/", wait_until="domcontentloaded")
        wait_landed(page)
        landing_ms = round((time.time() - cold_start) * 1000)
        state = page.evaluate("""() => ({
            search: location.search,
            protocol: document.querySelector('#protocol-select')?.value,
            subset: document.querySelector('#subset-select')?.value ?? null,
            subsetControl: !!document.querySelector('#subset-select'),
            sidebar: document.querySelector('[data-testid=sidebar-scope]')?.innerText ?? '',
            strip: document.querySelector('[data-testid=scope-strip]')?.innerText ?? '',
        })""")
        note("landing", state)
        results["timing"]["page_load_to_numbers_ms"] = landing_ms
        assert "protocol=circus_fashioniq_release_v1" in state["search"], state
        assert "subset=full" in state["search"], state
        assert state["subsetControl"] and state["subset"] == "full", state
        assert "0 lượt" not in state["strip"], state["strip"]
        shot(page, "1920-01-landing-bare-url")

        # FashionIQ Original: one cohort, NO subset control, nonzero counts.
        page.goto(BASE + "/?area=overview&protocol=fashioniq_original_split", wait_until="domcontentloaded")
        wait_ready(page)
        fiq = page.evaluate("""() => ({
            subsetControl: !!document.querySelector('#subset-select'),
            strip: document.querySelector('[data-testid=scope-strip]')?.innerText ?? '',
            sidebar: document.querySelector('[data-testid=sidebar-scope]')?.innerText ?? '',
        })""")
        note("fashiq_original", fiq)
        assert not fiq["subsetControl"], "FashionIQ Original must not render a subset control"
        assert "Subset" not in fiq["sidebar"], fiq["sidebar"]
        shot(page, "1920-02-fashioniq-original")

        # CIRCUS Full / SF / Validated, with counts read from the API for cross-check.
        rows = api("/api/composition/runs")
        for subset in ("full", "sf", "validated"):
            page.goto(BASE + f"/?area=evaluation&view=composition&protocol={CIRCUS}&subset={subset}",
                      wait_until="domcontentloaded")
            wait_ready(page)
            page.wait_for_selector("[data-testid=composition-table]", timeout=NAV_MS)
            rendered = page.locator("[data-testid^=composition-row-]").count()
            served = len([r for r in rows if r["protocol_id"] == CIRCUS and r["subset"] == subset])
            note(f"subset_{subset}", {"rendered_rows": rendered, "api_rows": served})
            assert rendered == served, (subset, rendered, served)
            shot(page, f"1920-03-subset-{subset}")

        # Scope with no data: a protocol whose circuit has no rows in this deployment.
        page.goto(BASE + "/?area=evaluation&view=composition&protocol=fashioniq_val_split",
                  wait_until="domcontentloaded")
        wait_ready(page)
        page.wait_for_timeout(500)
        empty = page.evaluate("""() => {
            const card = [...document.querySelectorAll('[data-testid=area-evaluation] .card')]
              .find(n => n.querySelector('[data-testid=composition-table], [data-testid=empty-state]'))
              ?? document.querySelector('[data-testid=area-evaluation]');
            const state = card?.querySelector('[data-testid=empty-state]');
            return {
              kind: state?.getAttribute('data-empty-kind') ?? 'table',
              text: (card?.innerText ?? '').split('\\n').filter(Boolean).join(' ').slice(0, 300),
            };
        }""")
        note("scope_no_data", empty)
        shot(page, "1920-04-scope-no-data")

        # API error state, two ways. First: the API itself returns 500 for the run list, and the page
        # must say "request failed" rather than print 0 rows.
        # Its own log: this context deliberately serves a 500, so its console errors are expected.
        forced_errors: list[str] = []
        err_context, err_page = new_page(1920, 1080, error_log=forced_errors)
        err_page.route("**/api/composition/runs?*",
                       lambda route: route.fulfill(status=500, content_type="application/json",
                                                   body='{"error":"forced"}'))
        err_page.route("**/api/composition/runs",
                       lambda route: route.fulfill(status=500, content_type="application/json",
                                                   body='{"error":"forced"}'))
        err_page.goto(BASE + "/?area=evaluation&view=composition&protocol=" + CIRCUS + "&subset=full",
                      wait_until="domcontentloaded")
        err_page.wait_for_function(
            "() => document.querySelector('[data-testid=area-evaluation] [data-testid=empty-state]')"
            "?.getAttribute('data-empty-kind') === 'error'", timeout=NAV_MS)
        api_err = err_page.evaluate("""() => ({
            kind: document.querySelector('[data-testid=area-evaluation] [data-testid=empty-state]')
                    ?.getAttribute('data-empty-kind'),
            text: document.querySelector('[data-testid=area-evaluation] [data-testid=empty-state]')
                  ?.innerText.replace(/\\n/g, ' ').slice(0, 200),
        })""")
        note("api_error_state", api_err)
        assert api_err["kind"] == "error", api_err
        shot(err_page, "1920-05-error-state")
        err_context.close()

        # Second: an invalid URL value is REFUSED by the parser and reported, never silently coerced.
        page.goto(BASE + "/?area=samples&view=explorer&subset=bogus", wait_until="domcontentloaded")
        page.wait_for_timeout(1500)
        err = page.evaluate("""() => ({
            problems: document.querySelector('[data-testid=url-problems]')?.innerText ?? null,
        })""")
        note("url_problem_state", err)
        assert err["problems"] and "subset=bogus" in err["problems"], (
            "an invalid URL value must be reported, not silently coerced", err)
        shot(page, "1920-05b-url-problem")

        # Loading state: throttle-free capture by intercepting the runs request and delaying it.
        slow_context, slow_page = new_page(1920, 1080, error_log=console_errors)

        def delay_runs(route):
            time.sleep(2.5)
            route.continue_()

        slow_page.route("**/api/composition/runs*", delay_runs)
        slow_page.goto(BASE + "/?area=overview", wait_until="domcontentloaded")
        slow_page.wait_for_selector("[data-testid=scope-strip]", timeout=NAV_MS)
        loading = slow_page.evaluate("""() => ({
            state: document.querySelector('[data-testid=scope-counts]')?.getAttribute('data-scope-state'),
            text: document.querySelector('[data-testid=scope-strip]')?.innerText ?? '',
            emptyKinds: [...document.querySelectorAll('[data-testid=empty-state]')].map(n => n.getAttribute('data-empty-kind')),
        })""")
        note("loading_state", loading)
        shot(slow_page, "1920-06-loading-state")
        slow_context.close()

        # ---------------------------------------------------------------- 2. inspector + Top-K
        url = (BASE + f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full"
               f"&runs=c:{DEEP_RUN}&run={DEEP_RUN}&condition=multimodal")
        page.goto(url, wait_until="domcontentloaded")
        wait_ready(page)
        page.wait_for_selector("[data-testid=query-inspector]", timeout=NAV_MS)
        select_start = time.time()
        page.fill("[data-testid=filter-query-id]", QUERY)
        page.wait_for_timeout(200)
        page.wait_for_selector(f"[data-testid=query-list] [data-testid=query-id]:text-is('{QUERY}')", timeout=NAV_MS)
        page.click(f"[data-testid=query-list] [data-testid=query-id]:text-is('{QUERY}')")
        page.wait_for_function(
            "(q) => (document.querySelector('[data-testid=query-title]')?.innerText || '').includes(q)",
            arg=QUERY, timeout=NAV_MS)
        page.wait_for_function(
            "() => document.querySelectorAll('[data-testid=query-sequence] [data-testid=query-image][data-asset-state=present]').length >= 2",
            timeout=NAV_MS)
        query_to_image_ms = round((time.time() - select_start) * 1000)
        results["timing"]["query_select_to_first_image_ms"] = query_to_image_ms

        images = page.evaluate("""() => {
            const seq = document.querySelector('[data-testid=query-sequence]');
            const nodes = [...seq.querySelectorAll('[data-testid=query-image]')];
            return {
              order: [...seq.children].map(n => n.getAttribute('data-testid') || n.className.trim()),
              tiles: nodes.map(n => ({id: n.getAttribute('data-image-id'),
                                      state: n.getAttribute('data-asset-state')})),
              caption: seq.querySelector('[data-testid=query-caption]')?.innerText ?? null,
              outside: document.querySelector('[data-testid=target-outside-window]')?.innerText ?? null,
            };
        }""")
        note("inspector_sequence", images)
        assert images["tiles"] and all(t["state"] == "present" for t in images["tiles"]), images
        real = api(f"/api/composition/runs/{DEEP_RUN}/queries?subset=full&condition=multimodal"
                   f"&query_id={QUERY}&limit=1&offset=0&top_k=10")["items"][0]
        note("query_export", {"query_id": real["query_id"], "target_rank": real["target_rank"],
                             "reference": real["reference_id"], "positives": real["positive_ids"]})
        assert images["caption"] == real["query_txt"], (images["caption"], real["query_txt"])
        # reference first, caption second, targets third: the evidence reading order.
        assert images["order"][:3] == ["shots", "query-instruction-caption", "query-targets"], images["order"]
        assert len(images["tiles"]) == 1 + len(real["positive_ids"]), images
        assert images["outside"], "target rank 73369 is outside the window and must be stated"
        shot(page, "1920-07-inspector-query-images")

        counts = {}
        for k in (10, 50, 100):
            page.select_option("[data-testid=topk]", str(k))
            page.wait_for_function(
                f"() => document.querySelectorAll('[data-testid=topk-grid] .candidate').length === {k}",
                timeout=NAV_MS)
            tile = page.evaluate("""() => {
                const cards = [...document.querySelectorAll('[data-testid=topk-grid] .candidate')];
                return cards.map(card => ({
                    id: card.querySelector('[data-testid=query-image]')?.getAttribute('data-image-id'),
                    rank: card.querySelector('.candidate-rank b')?.innerText,
                    score: card.querySelectorAll('small b')[0]?.innerText,
                }));
            }""")
            served = api(f"/api/composition/runs/{DEEP_RUN}/queries?subset=full&condition=multimodal"
                         f"&query_id={QUERY}&limit=1&offset=0&top_k={k}")["items"][0]["top_results"]
            served_ids = [row["image_id"].split("/")[-1] for row in served]
            ids = [t["id"] for t in tile]
            assert len(tile) == k, (k, len(tile))
            assert ids == served_ids[:k], (k, ids[:5], served_ids[:5])
            assert [t["rank"] for t in tile] == [f"#{row['rank']}" for row in served[:k]], k
            assert all(float(t["score"]) == round(row["score"], 4) for t, row in zip(tile, served)), k
            counts[k] = {"tiles": len(tile), "first": ids[0], "last": ids[-1],
                         "url": page.evaluate("() => location.search")}
            shot(page, f"1920-08-topk-{k}")
        note("topk_tiles", counts)
        assert "topk=" not in counts[100]["url"], "display K is not a URL control; the query stays the request identity"

        # Target outside K, stated on the panel and its image rendered.
        panel = page.evaluate("""() => ({
            targetTiles: [...document.querySelectorAll('[data-testid=query-targets] [data-testid=query-image]')]
              .map(n => ({id: n.getAttribute('data-image-id'), state: n.getAttribute('data-asset-state')})),
            outside: document.querySelector('[data-testid=target-outside-window]')?.innerText ?? null,
            ranks: [...document.querySelectorAll('[data-testid^=rank-]')].map(n => ({
                id: n.getAttribute('data-testid'), state: n.getAttribute('data-rank-state'),
                text: n.innerText.replace(/\\n/g, ' ')})),
        })""")
        note("target_states", panel)
        assert panel["targetTiles"] and panel["targetTiles"][0]["id"] == real["positive_ids"][0].split("/")[-1], panel
        assert "ngoài cửa sổ" in (panel["outside"] or ""), panel["outside"]
        # Three DISTINCT states, exercised on the two queries that have them:
        #   miss           -> a real rank outside @10 and outside the window (7:3045, above)
        #   missing-rank   -> no rank exists at all (7:3038, unscored)
        #   missing-export -> the run exported no candidate window for that query
        assert [r["state"] for r in panel["ranks"]] == ["miss", "miss", "miss"], panel["ranks"]
        page.fill("[data-testid=filter-query-id]", "7:3038")
        page.wait_for_selector("[data-testid=query-list] [data-testid=query-id]:text-is('7:3038')", timeout=NAV_MS)
        page.click("[data-testid=query-list] [data-testid=query-id]:text-is('7:3038')")
        page.wait_for_function(
            "(q) => (document.querySelector('[data-testid=query-title]')?.innerText || '').includes(q)",
            arg="7:3038", timeout=NAV_MS)
        page.wait_for_selector("[data-testid=candidates-unavailable]", timeout=NAV_MS)
        unscored_state = page.evaluate("""() => ({
            ranks: [...document.querySelectorAll('[data-testid^=rank-]')]
              .map(n => n.getAttribute('data-rank-state')),
            exportState: document.querySelector('[data-testid=candidates-unavailable]')
              ?.getAttribute('data-empty-kind') ?? null,
            targets: [...document.querySelectorAll('[data-testid=query-targets] [data-testid=query-image]')]
              .map(n => n.getAttribute('data-asset-state')),
        })""")
        note("unscored_states", unscored_state)
        assert set(unscored_state["ranks"]) == {"missing-rank"}, unscored_state
        assert unscored_state["exportState"] == "capability", unscored_state
        shot(page, "1920-07b-inspector-missing-rank")
        page.fill("[data-testid=filter-query-id]", QUERY)
        page.wait_for_selector(f"[data-testid=query-list] [data-testid=query-id]:text-is('{QUERY}')", timeout=NAV_MS)
        page.click(f"[data-testid=query-list] [data-testid=query-id]:text-is('{QUERY}')")
        page.wait_for_function(
            "(q) => (document.querySelector('[data-testid=query-title]')?.innerText || '').includes(q)",
            arg=QUERY, timeout=NAV_MS)

        # Missing asset: an id that does not exist must render the explicit missing tile.
        # Asked of the API directly: a deliberate 404 fetched from the page would appear as a console
        # error on the page whose console must stay clean, for an assertion about the server.
        try:
            urllib.request.urlopen(API + "/api/images/circus/NOPE_NOT_AN_IMAGE.jpg", timeout=30)
            missing = 200
        except urllib.error.HTTPError as error:
            missing = error.code
        results["intentional_console_errors"] = forced_errors
        note("missing_asset_http", missing)
        assert missing == 404, missing

        # ---------------------------------------------------------------- 3. chip row
        page.goto(BASE + f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full&runs=c:{DEEP_RUN}",
                  wait_until="domcontentloaded")
        wait_ready(page)
        page.wait_for_selector("[data-testid=query-state-chips]", timeout=NAV_MS)
        chips = page.evaluate("""() => [...document.querySelectorAll('[data-testid^=chip-]')]
            .filter(n => n.getAttribute('data-testid').startsWith('chip-')
                         && !n.getAttribute('data-testid').startsWith('chip-count'))
            .map(n => ({id: n.getAttribute('data-testid'), pressed: n.getAttribute('aria-pressed'),
                        text: n.innerText.replace(/\\n/g, ' ')}))""")
        note("chips", chips)
        api_counts = api(f"/api/composition/runs/{DEEP_RUN}/queries?subset=full&limit=1&offset=0")["state_counts"]
        note("chip_api_counts", api_counts)
        assert len(chips) == 4, chips
        for chip in chips:
            if chip["id"] == "chip-all":
                continue
            key = chip["id"].replace("chip-", "")
            assert f"{api_counts[key]:,}" in chip["text"], (chip, api_counts)

        # A chip filters, and the URL restores the same view; paging preserves it.
        page.click("[data-testid=chip-hit]")
        page.wait_for_function("() => location.search.includes('query_state=hit')", timeout=NAV_MS)
        page.wait_for_function(
            "() => document.querySelector('[data-testid=pager-total]')?.getAttribute('data-total')"
            f" === '{api_counts['hit']}'", timeout=NAV_MS)
        filtered_url = page.evaluate("() => location.search")
        shot(page, "1920-09-chip-correct-at-10")
        page.click("[data-testid=pager-next]")
        page.wait_for_function("() => location.search.includes('offset=20')", timeout=NAV_MS)
        paged_url = page.evaluate("() => location.search")
        assert "query_state=hit" in paged_url, paged_url
        page.reload(wait_until="domcontentloaded")
        wait_ready(page)
        assert page.evaluate("() => location.search") == paged_url, "reload must restore the full URL"
        page.go_back()
        page.wait_for_timeout(800)
        assert page.evaluate("() => location.search") == filtered_url, "Back must restore the previous URL"
        page.wait_for_selector("[data-testid=query-state-chips]", timeout=NAV_MS)
        assert api_counts["hit"] == int(page.evaluate(
            "() => document.querySelector('[data-testid=pager-total]').getAttribute('data-total')")), "Back restores the filter"

        # Advanced disclosure carries category/outcome/sort/imported.
        advanced = page.evaluate("""() => {
            const detail = [...document.querySelectorAll('details')]
              .find(d => d.querySelector('[data-testid=filter-outcome]'));
            if (!detail) return null;
            return {open: detail.open,
                    tests: [...detail.querySelectorAll('[data-testid^=filter-]')].map(n => n.getAttribute('data-testid'))
                           .concat([...detail.querySelectorAll('input[type=checkbox]')].map(n => n.getAttribute('data-testid')))};
        }""")
        note("advanced_disclosure", advanced)
        assert advanced and not advanced["open"], advanced
        for testid in ("filter-outcome", "filter-category", "filter-order", "include-imported"):
            assert testid in advanced["tests"], (testid, advanced)
        shot(page, "1920-10-advanced-collapsed")

        # ---------------------------------------------------------------- 4. stale responses
        stale = {"lag": []}
        race_context, race_page = new_page(1920, 1080, error_log=console_errors)
        delay = {"query": 3.0}

        def slow_query(route):
            time.sleep(delay["query"])
            route.continue_()

        race_page.route("**/api/composition/runs/**/queries?*query_id=*", slow_query)
        race_page.goto(BASE + f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full&runs=c:{DEEP_RUN}",
                       wait_until="domcontentloaded")
        wait_ready(race_page)
        race_page.wait_for_selector("[data-testid=query-inspector]", timeout=NAV_MS)
        delay["query"] = 3.0
        race_page.evaluate("""() => { history.pushState({}, '', location.search + '&query=7:3045'); }""")
        race_page.evaluate("() => dispatchEvent(new PopStateEvent('popstate'))")
        race_page.wait_for_timeout(250)
        delay["query"] = 0
        race_page.evaluate("""() => {
            const params = new URLSearchParams(location.search);
            params.set('query', '7:1466');
            history.pushState({}, '', '?' + params.toString());
            dispatchEvent(new PopStateEvent('popstate'));
        }""")
        race_page.wait_for_function(
            "() => { const t = document.querySelector('[data-testid=query-title]');"
            " return !!t && t.innerText.includes('7:1466'); }", timeout=NAV_MS)
        race_page.wait_for_timeout(4000)   # let the SLOW first answer land
        race_title = race_page.evaluate(
            "() => document.querySelector('[data-testid=query-title]')?.innerText ?? null")
        stale["after_slow_arrives"] = race_title
        note("stale_response_guard", stale)
        assert race_title and "7:1466" in race_title, f"a stale answer overwrote the newer query: {race_title}"
        shot(race_page, "1920-11-stale-response-guard")
        race_context.close()

        # ---------------------------------------------------------------- 5. LIMN diagnostics
        page.goto(BASE + f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full&runs=c:{DEEP_RUN}",
                  wait_until="domcontentloaded")
        wait_ready(page)
        page.wait_for_selector("[data-testid=limn-diagnostic-probe]", timeout=NAV_MS)
        page.locator("[data-testid=limn-diagnostic]").scroll_into_view_if_needed()
        combos = []
        for qid in ("7:3038", "7:3790"):
            for scope in ("shirt", "toptee"):
                for condition in ("multimodal", "text_only_black_image", "image_only_empty_text"):
                    page.select_option("[data-testid=limn-probe-qid]", qid)
                    page.select_option("[data-testid=limn-probe-scope]", scope)
                    page.select_option("[data-testid=limn-probe-condition]", condition)
                    page.wait_for_function(
                        "() => document.querySelectorAll('[data-testid=limn-probe-grid] .candidate').length === 100",
                        timeout=NAV_MS)
                    page.wait_for_function(
                        "() => { const t = document.querySelector('[data-testid=limn-probe-target] [data-testid=query-image]');"
                        " return !!t; }", timeout=NAV_MS)
                    page.wait_for_timeout(250)
                    state = page.evaluate("""() => {
                        const grid = [...document.querySelectorAll('[data-testid=limn-probe-grid] .candidate')];
                        const target = document.querySelector('[data-testid=limn-probe-target] [data-testid=query-image]');
                        return {
                          tiles: grid.length,
                          targetId: target?.getAttribute('data-image-id'),
                          targetState: target?.getAttribute('data-asset-state'),
                          outside: document.querySelector('[data-testid=limn-probe-target-outside]')?.innerText ?? null,
                          inWindow: document.querySelector('[data-testid=limn-probe-target-in-window]')?.innerText ?? null,
                          incomplete: document.querySelector('[data-testid=limn-probe-incomplete]')?.innerText ?? null,
                          note: document.querySelector('[data-testid=limn-probe-note]')?.innerText ?? null,
                          shown: grid.filter(c => c.querySelector('[data-testid=query-image]')
                                                   ?.getAttribute('data-asset-state') === 'present').length,
                          firstRank: grid[0]?.querySelector('.candidate-rank b')?.innerText,
                        };
                    }""")
                    served = api(f"/api/composition/diagnostics/limn-dual-scope/candidates"
                                 f"?qid={qid}&scope={scope}&condition={condition}")
                    assert state["tiles"] == len(served["candidates"]), (qid, scope, condition, state["tiles"])
                    assert state["shown"] == state["tiles"], f"images did not decode: {state}"
                    assert state["targetState"] == "present", state
                    assert state["targetId"] == served["target"]["gallery_id"].split("/")[-1], (
                        state["targetId"], served["target"]["gallery_id"])
                    assert (state["outside"] is not None) != served["target"]["in_window"], (state, served["target"]["in_window"])
                    combos.append({"qid": qid, "scope": scope, "condition": condition,
                                   "tiles": state["tiles"], "target": state["targetId"],
                                   "in_window": served["target"]["in_window"],
                                   "target_rank": served["target"]["rank"],
                                   "complete": served["complete"], "defects": len(served["defects"]),
                                   "incomplete_notice": bool(state["incomplete"])})
        note("limn_combinations", combos)
        assert len(combos) == 12 and all(c["tiles"] == 100 for c in combos), combos
        page.select_option("[data-testid=limn-probe-qid]", "7:3790")
        page.select_option("[data-testid=limn-probe-scope]", "shirt")
        page.select_option("[data-testid=limn-probe-condition]", "text_only_black_image")
        page.wait_for_timeout(1200)
        page.locator("[data-testid=limn-diagnostic]").scroll_into_view_if_needed()
        shot(page, "1920-12-limn-outside-window")
        shot(page, "1920-13-limn-defects")
        # The paragraph that used to explain why images are not rendered must be gone.
        body = page.inner_text("body")
        assert "Ảnh không được render ở đây" not in body, "the 'images are not rendered' paragraph must be deleted"

        # ---------------------------------------------------------------- 6. parity gate
        page.goto(BASE + f"/?area=evaluation&view=composition&protocol={CIRCUS}&subset=full",
                  wait_until="domcontentloaded")
        wait_ready(page)
        page.wait_for_selector("[data-testid=parity-gate-table]", timeout=NAV_MS)
        gate_state = page.evaluate("""() => ({
            verdict: document.querySelector('[data-testid=parity-gate-verdict]')?.innerText ?? null,
            summary: document.querySelector('[data-testid=parity-gate-summary]')?.innerText ?? null,
            rows: [...document.querySelectorAll('[data-testid^=parity-row-]')].map(n => ({
                id: n.getAttribute('data-testid'), passed: n.getAttribute('data-passed'),
                text: n.innerText.replace(/\\n/g, ' ').slice(0, 160)})),
        })""")
        gate_api = api("/api/runs/parity-gate")
        note("parity_gate", {"ui": gate_state, "api_passed": gate_api["passed_models"],
                             "api_status": {k: v["status"] for k, v in gate_api["models"].items()}})
        assert gate_api["passed_models"] == [], gate_api["passed_models"]
        assert set(gate_state["rows"][0]) and len(gate_state["rows"]) == len(gate_api["models"]), gate_state
        assert all(row["passed"] == "false" for row in gate_state["rows"]), gate_state
        shot(page, "1920-14-parity-gate")

        # ---------------------------------------------------------------- 7. model inventory
        page.goto(BASE + f"/?area=experiments&view=models&protocol={CIRCUS}&subset=validated",
                  wait_until="domcontentloaded")
        wait_ready(page)
        # `/api/models` hashes every checkpoint in the registry; its cold start is minutes, which is
        # exactly why readiness above never waits on it. The inventory is the ONE view that needs it,
        # so it gets its own (long) budget here and is pre-warmed by the runner.
        page.wait_for_selector("[data-testid=model-inventory]", timeout=900_000)
        inv = page.evaluate("""() => ({
            rows: document.querySelectorAll('[data-testid^=model-row-]').length,
            scopeLinks: [...document.querySelectorAll('[data-testid^=other-scope-]')].map(n => ({
                id: n.getAttribute('data-testid'), text: n.innerText.replace(/\\n/g, ' ').slice(0, 120)})),
            todo: document.querySelectorAll('[data-testid^=todo-]').length,
        })""")
        note("model_inventory", {"rows": inv["rows"], "scope_link_groups": len(inv["scopeLinks"]),
                                 "links": inv["scopeLinks"][:6], "bare_todo_rows": inv["todo"]})
        shot(page, "1920-15-model-inventory-validated")

        # ---------------------------------------------------------------- 8. viewports
        for width, height, name in ((2560, 1440, "2560"), (390, 844, "390")):
            vcontext, vpage = new_page(width, height, mobile=width < 500, error_log=console_errors)
            vpage.goto(BASE + f"/?area=samples&view=explorer&protocol={CIRCUS}&subset=full"
                              f"&runs=c:{DEEP_RUN}&query={QUERY}", wait_until="domcontentloaded")
            wait_ready(vpage)
            vpage.wait_for_function(
                "() => document.querySelectorAll('[data-testid^=query-image][data-asset-state=present]').length >= 3",
                timeout=NAV_MS)
            overflow = vpage.evaluate("""() => ({
                docWidth: document.documentElement.scrollWidth,
                viewport: window.innerWidth,
                navVisible: !!document.querySelector('[data-testid=workbench-nav]'),
            })""")
            note(f"viewport_{name}", overflow)
            assert overflow["docWidth"] <= overflow["viewport"] + 2, overflow
            shot(vpage, f"{name}-16-inspector")
            vcontext.close()

        # ---------------------------------------------------------------- 9. warm timing
        warm_start = time.time()
        page.goto(BASE + f"/?area=evaluation&view=composition&protocol={CIRCUS}&subset=full",
                  wait_until="domcontentloaded")
        wait_ready(page)
        page.wait_for_selector("[data-testid^=composition-row-]", timeout=NAV_MS)
        warm_ms = round((time.time() - warm_start) * 1000)
        results["timing"]["warm_page_load_to_numbers_ms"] = warm_ms
        note("warm_load", warm_ms)

        context.close()
        browser.close()

    results["console_errors"] = console_errors
    assert not console_errors, console_errors
    RAW.mkdir(parents=True, exist_ok=True)
    (RAW / "web_acceptance.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results["timing"], indent=2))
    print("ACCEPTANCE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
