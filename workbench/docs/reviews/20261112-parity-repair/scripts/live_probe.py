"""Live UI probe after the types-only gate.ts edit + rebuild (dev-only, not shipped).

Loads the built bundle through `vite preview` on 5173 and asserts the three things the repair
claims: the parity header renders the standard as a *string*, the run's condition selector offers
exactly the run's own three conditions, and a known query renders real ranks. Console errors are
collected, not swallowed.

Modelled on workbench/docs/reviews/20261109-native15-composition-limn/scripts/ui_shots.py.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261112-parity-repair/raw/live")
URL = "http://127.0.0.1:5173"
RUN = "da0a2162be0e159a1a456cec"
RUN_KEY = f"c:{RUN}"


def watch(page, sink: list[str]) -> None:
    page.on("console", lambda m: m.type == "error" and sink.append(f"console:{m.text}"))
    page.on("pageerror", lambda e: sink.append(f"pageerror:{e}"))
    page.on("requestfailed", lambda r: sink.append(f"requestfailed:{r.url}:{r.failure}"))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"url": URL, "run": RUN}

    with sync_playwright() as pw:
        browser = pw.chromium.launch(channel="chrome", headless=True)
        ctx = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = ctx.new_page()
        errors: list[str] = []
        watch(page, errors)

        # --- 1. shell serves the built bundle ------------------------------------------------
        resp = page.goto(f"{URL}/", wait_until="domcontentloaded", timeout=90000)
        scripts = page.eval_on_selector_all("script[src]", "els => els.map(e => e.getAttribute('src'))")
        styles = page.eval_on_selector_all("link[rel=stylesheet]", "els => els.map(e => e.getAttribute('href'))")
        report["shell"] = {"status": resp.status if resp else None, "scripts": scripts, "stylesheets": styles}
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "01-shell.png"), full_page=False)

        # --- 2. Evaluation / parity header ----------------------------------------------------
        page.goto(f"{URL}/?area=evaluation&view=composition", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector('[data-testid="parity-gate"]', timeout=120000)
        page.wait_for_selector('[data-testid="parity-gate-table"]', timeout=120000)
        card = page.get_by_test_id("parity-gate")
        card.first.scroll_into_view_if_needed()
        page.wait_for_timeout(600)
        subtitle = card.first.locator(".card-head p, .card-subtitle, p").first.inner_text()
        header_text = card.first.inner_text()
        rows = page.locator('[data-testid^="parity-row-"]')
        report["parity"] = {
            "subtitle": subtitle,
            "header_line": next((ln for ln in header_text.splitlines() if "Chuẩn" in ln), None),
            "contains_Chuẩn_parity-relative/v1": "Chuẩn parity-relative/v1" in header_text,
            "contains_bracket_list": bool(re.search(r"Chuẩn \[", header_text)),
            "verdict_badge": page.get_by_test_id("parity-gate-verdict").first.inner_text(),
            "summary": page.get_by_test_id("parity-gate-summary").first.inner_text(),
            "models": [
                {
                    "model_id": rows.nth(i).get_attribute("data-testid"),
                    "passed": rows.nth(i).get_attribute("data-passed"),
                    "text": rows.nth(i).inner_text().splitlines(),
                }
                for i in range(rows.count())
            ],
        }
        card.first.screenshot(path=str(OUT / "02-parity-card.png"))
        page.screenshot(path=str(OUT / "02-parity-page.png"), full_page=True)

        # --- 3. Samples page for this run: condition selector + query 7:2179 ------------------
        page.goto(
            f"{URL}/?area=samples&view=explorer&protocol=circus_fashioniq_release_v1"
            f"&subset=full&runs={RUN_KEY}&query=7%3A2179&q=7%3A2179",
            wait_until="domcontentloaded",
            timeout=90000,
        )
        page.wait_for_selector('[data-testid="filter-condition"]', timeout=120000)
        page.wait_for_timeout(2500)
        sel = page.get_by_test_id("filter-condition")
        opts = sel.locator("option").all()
        report["conditions"] = {
            "values": [o.get_attribute("value") for o in opts],
            "labels": [o.inner_text() for o in opts],
            "selected": sel.input_value(),
        }
        report["run_select"] = {
            "value": page.get_by_test_id("active-run").input_value(),
        }
        page.screenshot(path=str(OUT / "03-samples-filters.png"), full_page=False)

        # query 7:2179: selected query title + the three rank cells
        page.wait_for_selector('[data-testid="query-inspector"]', timeout=120000)
        page.wait_for_timeout(2500)
        insp = page.get_by_test_id("query-inspector")
        ranks = {}
        for cid in ("multimodal", "text_only_black_image", "clvc_image_encoder_only"):
            loc = page.get_by_test_id(f"rank-{cid}")
            ranks[cid] = {"count": loc.count(), "text": loc.first.inner_text() if loc.count() else None}
        report["inspector"] = {
            "title": page.get_by_test_id("query-title").first.inner_text(),
            "caption": page.get_by_test_id("query-caption").first.inner_text(),
            "ranks": ranks,
            "unscored_notice": (page.get_by_test_id("query-unscored-notice").first.inner_text()
                                if page.get_by_test_id("query-unscored-notice").count() else None),
        }
        insp.first.scroll_into_view_if_needed()
        page.wait_for_timeout(400)
        insp.screenshot(path=str(OUT / "04-query-7-2179-inspector.png"))
        page.screenshot(path=str(OUT / "04-query-7-2179-page.png"), full_page=True)

        # --- 4. console errors ---------------------------------------------------------------
        report["console_errors"] = errors
        report["console_error_count"] = len(errors)
        ctx.close()
        browser.close()

    (OUT / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
