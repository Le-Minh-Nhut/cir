#!/usr/bin/env python3
"""Element-level zoom captures: crops one component at 2x DSF for pixel inspection.

The whole-page screenshots show what a reviewer sees; these cropped, 2x-DSF shots show what the
defect actually looks like (a stretched pill, a header that wraps, a table whose first column
scrolled away). Regenerates the `screens/zoom_*.png` set.

    PLAYWRIGHT_BROWSERS_PATH=workbench/envs/ctl312h/.playwright-browsers \
    workbench/envs/ctl312h/bin/python <bundle>/browser_qa/zooms.py --out <bundle>/browser_qa

Requires COMP_WITH_FIXTURES=1 to replay the fixtures opt-in (route injection), which is how the
FIXTURE/SMOKE badge layout is exposed: the app itself never requests the fixtures.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "workbench/frontend/tests"))
import browser_qa as hq  # noqa: E402

UI = "http://127.0.0.1:5173"
REAL_RUN = "6a528e877c328772e2c7bb28"
VIEWPORTS = ((1440, 900), (768, 1024), (390, 844))

INJECT = "**/api/composition/runs?**"


def opt_in(route):
    url = route.request.url
    if "include_fixtures" not in url:
        url += "&include_fixtures=true&include_smoke=true"
    route.continue_(url=url)


def clip(page, out: pathlib.Path, name: str, selector: str, limit: int | None = None) -> bool:
    """Screenshot every match of `selector` as its own file. Returns whether any was captured."""
    handles = [h for h in page.query_selector_all(selector) if h.is_visible()]
    if not handles:
        return False
    for i, handle in enumerate(handles[: limit or len(handles)]):
        handle.scroll_into_view_if_needed()
        page.wait_for_timeout(200)
        suffix = f"_{i}" if len(handles) > 1 else ""
        handle.screenshot(path=str(out / f"zoom_{name}{suffix}.png"))
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--viewport", action="append", default=None)
    args = parser.parse_args()
    out = args.out / "screens"
    out.mkdir(parents=True, exist_ok=True)
    vps = VIEWPORTS if not args.viewport else [v for v in VIEWPORTS if f"{v[0]}x{v[1]}" in args.viewport]

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = hq.launch(p)
        for w, h in vps:
            tag = f"_{w}x{h}"
            ctx = browser.new_context(viewport={"width": w, "height": h}, device_scale_factor=2)
            page = ctx.new_page()

            page.goto(f"{UI}/?page=Composition&comp_subset=full", wait_until="load")
            page.wait_for_timeout(2000)
            clip(page, out, f"subset_selector{tag}", ".subset-selector")
            clip(page, out, f"composition_table{tag}", ".composition-view .table-container", 1)
            clip(page, out, f"outcomes_bar{tag}", ".outcomes-bar")
            clip(page, out, f"drilldown_filters{tag}", ".drilldown-filters")
            clip(page, out, f"drilldown_card{tag}", ".query-card", 1)

            # Row identity after a horizontal scroll, the case the Model column leaves first.
            page.evaluate("""() => {
              const wrap = document.querySelector('.composition-table')?.closest('.table-container');
              if (wrap) wrap.scrollLeft = wrap.scrollWidth;
            }""")
            page.wait_for_timeout(300)
            clip(page, out, f"table_scrolled_right{tag}", ".composition-view .table-container", 1)

            # The app itself never opts into fixtures; replay that request so the FIXTURE/SMOKE
            # badges exist on screen at all. Both the pre-fix and post-fix table are kept by the
            # bundle's history, so this file only records the current build.
            page.route(INJECT, opt_in)
            page.goto(f"{UI}/?page=Composition&comp_subset=full", wait_until="load")
            page.wait_for_timeout(2000)
            clip(page, out, f"table_fixtures{tag}", ".composition-view .table-container", 1)
            page.unroute(INJECT)

            page.goto(f"{UI}/?page=Benchmark", wait_until="load")
            page.wait_for_timeout(1500)
            clip(page, out, f"benchmark_card{tag}", ".benchmark-run-card", 1)
            clip(page, out, f"benchmark_table{tag}", ".benchmark-run-card .table-container", 3)
            if w >= 1440:
                for details in page.query_selector_all("details.gate-detail"):
                    details.evaluate("el => el.open = true")
                page.wait_for_timeout(400)
                clip(page, out, f"gate_details{tag}", "details.gate-detail", 1)

            page.goto(f"{UI}/?page=Dashboard", wait_until="load")
            page.wait_for_timeout(1800)
            clip(page, out, f"dash_card{tag}", ".benchmark-summary-card", 1)
            clip(page, out, f"dashboard_bench{tag}", ".dashboard-benchmarks", 1)

            ctx.close()
            print(f"captured {w}x{h}")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
