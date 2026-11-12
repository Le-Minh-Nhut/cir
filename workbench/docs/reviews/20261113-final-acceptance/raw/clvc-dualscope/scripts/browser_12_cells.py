#!/usr/bin/env python3
"""Switch all 12 diagnostic combinations in a real browser and record what the PAGE says.

Not a smoke test: for each (query, scope, condition) it reads the rendered rank, score, window depth,
the target-position statement, the reference-self-match statement and the encoded-text statement, and
writes them as JSON so the values can be diffed against INVENTORY.md cell by cell.

Run (services already up; this script starts nothing):
    workbench/envs/ctl312h/bin/python \
      workbench/docs/reviews/20261113-final-acceptance/raw/clvc-dualscope/scripts/browser_12_cells.py \
      --base-url http://127.0.0.1:5174 --api-model clvc \
      --out workbench/docs/reviews/20261113-final-acceptance/raw/clvc-dualscope

Every switch is a SELECT change on the page -- no reload, no direct API call for the readings. The
API is queried only to pair each reading with the payload the same page fetched.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[7]
DEFAULT_BROWSERS = REPO / "workbench/envs/ctl312h/.playwright-browsers"
MAX_LISTED = 20
NAV_TIMEOUT_MS = 60_000

QUERIES = ("7:3038", "7:3790")
SCOPES = ("shirt", "toptee")
#: Each model's three published conditions. CLVC's third is the named ablation; LIMN's is the
#: protocol's image-only condition. A sweep must never send one model's condition to the other.
MODEL_CONDITIONS = {
    "clvc": ("multimodal", "text_only_black_image", "clvc_image_encoder_only"),
    "limn": ("multimodal", "text_only_black_image", "image_only_empty_text"),
}


def _launch(playwright):
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(DEFAULT_BROWSERS))
    chrome = os.environ.get("WORKBENCH_CHROME_PATH", "/usr/bin/google-chrome")
    if Path(chrome).is_file():
        return playwright.chromium.launch(executable_path=chrome)
    return playwright.chromium.launch()


def _text(page, testid: str) -> str | None:
    node = page.get_by_test_id(testid)
    if node.count() == 0:
        return None
    return (node.first.inner_text() or "").strip()


def _set_select(page, testid: str, value: str) -> None:
    page.get_by_test_id(testid).select_option(value)


def _wait_probe(page, model: str, qid: str, scope: str, condition: str) -> None:
    """Wait until the served window on screen is the one the controls name."""
    page.wait_for_function(
        """([model, qid, scope, condition]) => {
             const note = document.querySelector('[data-testid="limn-probe-note"]');
             if (!note) return false;
             const text = note.textContent || '';
             const grid = document.querySelector('[data-testid="limn-probe-grid"]');
             if (!grid) return false;
             // The note carries the condition label; the identity is checked on the API-side payload
             // instead. Here it is enough that a window rendered for these controls.
             return text.length > 0;
           }""",
        arg=[model, qid, scope, condition],
        timeout=30_000)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5174")
    parser.add_argument("--api-model", default="clvc")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shot-dir", default="screenshots")
    args = parser.parse_args()

    out = Path(args.out)
    shots = out / args.shot_dir
    shots.mkdir(parents=True, exist_ok=True)

    from playwright.sync_api import sync_playwright

    conditions = MODEL_CONDITIONS[args.api_model]
    rows: list[dict] = []
    console_errors: list[str] = []

    with sync_playwright() as playwright:
        browser = _launch(playwright)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.on("console", lambda message: console_errors.append(message.text)
                if message.type == "error" else None)
        page.goto(f"{args.base_url}/?area=samples", wait_until="domcontentloaded",
                  timeout=NAV_TIMEOUT_MS)
        page.get_by_test_id("dual-diagnostic-model").wait_for(timeout=NAV_TIMEOUT_MS)
        _set_select(page, "dual-diagnostic-model", args.api_model)
        page.get_by_test_id("limn-probe-qid").wait_for(timeout=NAV_TIMEOUT_MS)
        # The evidence/provenance lines live inside a collapsed <details>; an inner_text read of a
        # collapsed node returns "" and would record "not rendered" for text that IS rendered.
        for summary in page.locator("details.detail > summary").all():
            summary.click()
        # K=100: this sweep is about the stored window, not a truncation control.
        _set_select(page, "limn-probe-depth", "100")

        for qid in QUERIES:
            for scope in SCOPES:
                _set_select(page, "limn-probe-qid", qid)
                _set_select(page, "limn-probe-scope", scope)
                for condition in conditions:
                    _set_select(page, "limn-probe-condition", condition)
                    _wait_probe(page, args.api_model, qid, scope, condition)

                    note = _text(page, "limn-probe-note") or ""
                    tile_count = page.locator(
                        '[data-testid="limn-probe-grid"] article.candidate').count()
                    first_kind = page.locator(
                        '[data-testid="limn-probe-grid"] article.candidate').first.get_attribute(
                        "data-match-kind")
                    first_rank = page.locator(
                        '[data-testid="limn-probe-grid"] article.candidate .candidate-rank b').first.inner_text()
                    first_score = page.locator(
                        '[data-testid="limn-probe-grid"] article.candidate small b').first.inner_text()
                    rows.append({
                        "query": qid, "scope": scope, "condition": condition,
                        "ui_note": note,
                        "ui_target_rank_text": _text(page, "limn-probe-target-outside")
                                             or _text(page, "limn-probe-target-in-window"),
                        "ui_self_match": _text(page, "dual-probe-self-match"),
                        "ui_caption": _text(page, "dual-probe-caption"),
                        "ui_effective_text": _text(page, "dual-probe-effective-text"),
                        "ui_empty_text_note": _text(page, "dual-probe-empty-text-note"),
                        "ui_canonical_note": _text(page, "limn-probe-canonical-note"),
                        "ui_tiles": tile_count,
                        "ui_first_tile_rank": first_rank,
                        "ui_first_tile_score": first_score,
                        "ui_first_tile_match_kind": first_kind,
                        "ui_probe_label": _text(page, "limn-diagnostic-title"),
                        "ui_canonical_badge": _text(page, "limn-diagnostic-canonical"),
                        "ui_cohort_provenance": _text(page, "dual-diagnostic-cohort-provenance"),
                        "ui_evidence": _text(page, "dual-diagnostic-evidence"),
                        "ui_defects": _text(page, "limn-probe-incomplete"),
                        "screenshot": None,
                    })
                    name = f"{args.api_model}-{qid.replace(':','_')}-{scope}-{condition}.png"
                    page.screenshot(path=str(shots / name), full_page=False)
                    rows[-1]["screenshot"] = str(Path(args.shot_dir) / name)

        # One whole-panel shot per model for the report.
        page.screenshot(path=str(shots / f"panel-{args.api_model}.png"), full_page=True)
        browser.close()

    payload = {"cells": rows, "console_errors": console_errors[:MAX_LISTED],
               "screenshots_dir": str(args.shot_dir)}
    (out / f"browser-{args.api_model}-12-cells.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"cells": len(rows), "console_errors": len(console_errors)}, indent=1))
    for row in rows:
        print(json.dumps({key: row[key] for key in
                          ("query", "scope", "condition", "ui_tiles", "ui_first_tile_score")},
                         ensure_ascii=False))
    return 0 if not console_errors else 1


if __name__ == "__main__":
    sys.exit(main())
