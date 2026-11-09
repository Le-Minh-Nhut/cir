"""Screenshots of the live workbench for the LIMN dual-scope diagnostic (dev-only, not shipped).

Uses the installed Chrome through playwright's `channel="chrome"`. Writes PNGs plus a JSON of the
DOM facts the screenshots must agree with, so a claim about the page is checked against the page.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261109-native15-composition-limn/raw/ui")
URL = "http://127.0.0.1:5173"
SIZES = {"1920x1080": (1920, 1080), "2560x1440": (2560, 1440), "390x844": (390, 844)}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: dict = {"url": URL, "sizes": {}}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="chrome", headless=True)
        for name, (width, height) in SIZES.items():
            context = browser.new_context(viewport={"width": width, "height": height})
            page = context.new_page()
            page.goto(f"{URL}/?area=samples&view=explorer", wait_until="domcontentloaded", timeout=90000)
            # `networkidle` never settles here: the shell keeps a live region fetching. Wait for the
            # card itself instead, which is the thing being inspected.
            page.wait_for_selector('[data-testid="limn-diagnostic"], [data-testid="limn-diagnostic-unavailable"]',
                                   timeout=120000)
            page.wait_for_timeout(1200)
            facts: dict = {"viewport": [width, height]}
            # The diagnostic card is served independently of any run selection.
            card = page.get_by_test_id("limn-diagnostic")
            facts["diagnostic_card_count"] = card.count()
            if card.count():
                card.first.scroll_into_view_if_needed()
                page.wait_for_timeout(500)
                facts["title"] = page.get_by_test_id("limn-diagnostic-title").first.inner_text()
                facts["canonical_badge"] = page.get_by_test_id("limn-diagnostic-canonical").first.inner_text()
                facts["canonical_notice"] = page.get_by_test_id("limn-diagnostic-canonical-notice").first.inner_text()
                table = page.get_by_test_id("limn-diagnostic-table")
                facts["table_visible"] = table.is_visible()
                rows = page.locator('[data-testid^="limn-diagnostic-row-"]')
                facts["row_count"] = rows.count()
                facts["row_texts"] = [rows.nth(i).inner_text() for i in range(rows.count())]
                facts["body_scroll_width"] = page.evaluate("document.body.scrollWidth")
                facts["inner_width"] = page.evaluate("window.innerWidth")
                table_box = table.bounding_box()
                facts["table_box"] = table_box
                # Any element overflowing the viewport horizontally?
                facts["horizontal_overflow"] = page.evaluate(
                    "Array.from(document.querySelectorAll('body *'))"
                    ".filter(e => e.getBoundingClientRect().right > window.innerWidth + 2)"
                    ".slice(0, 5).map(e => e.className || e.tagName)")
            page.screenshot(path=str(OUT / f"diagnostic-{name}.png"), full_page=True)
            facts["screenshot"] = f"diagnostic-{name}.png"
            report["sizes"][name] = facts
            context.close()
        # A crop at full resolution around the table, so a text claim is not read from a thumbnail.
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        page.goto(f"{URL}/?area=samples&view=explorer", wait_until="domcontentloaded", timeout=90000)
        page.wait_for_selector('[data-testid="limn-diagnostic"]', timeout=120000)
        page.wait_for_timeout(1200)
        card = page.get_by_test_id("limn-diagnostic")
        if card.count():
            card.first.scroll_into_view_if_needed()
            page.wait_for_timeout(400)
            card.first.screenshot(path=str(OUT / "diagnostic-card-1920.png"))
            report["card_crop"] = "diagnostic-card-1920.png"
        context.close()
        browser.close()
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in facts.items() if k != "row_texts"}
                      for name, facts in report["sizes"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
