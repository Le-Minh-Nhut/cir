#!/usr/bin/env python3
"""The browser half of the 20261113 acceptance check, run against the DEPLOYED stack.

Every reading below comes from the rendered DOM in headless Chrome at 127.0.0.1:5173 — playbook
selects and text reads, never an API call standing in for the page. The API is consulted only to
pair a rendered number with the payload the same page fetched, or to assert something the UI is
required to *omit* (a PAIR row).

Run (services already up):
    workbench/envs/ctl312h/bin/python \
      workbench/docs/reviews/20261113-final-acceptance/raw/live-verify/probe_live_verify.py

Outputs into this directory: live-verify.json plus screenshots/.
Exit 0 only when there are no mismatches, no console errors, no page errors and no failed requests.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
BROWSERS = REPO / "workbench/envs/ctl312h/.playwright-browsers"
SHOTS = HERE / "screenshots"
BASE = "http://127.0.0.1:5173"
API = "http://127.0.0.1:8000"
NAV_MS = 60_000

QUERIES = ("7:3038", "7:3790")
SCOPES = ("shirt", "toptee")
CLVC_CONDITIONS = ("multimodal", "text_only_black_image", "clvc_image_encoder_only")

#: raw/clvc-dualscope/INVENTORY.md §"The 12 cells", transcribed as (qid, scope, condition) -> rank.
INVENTORY = {
    ("7:3038", "shirt", "multimodal"): 848,
    ("7:3038", "shirt", "text_only_black_image"): 304,
    ("7:3038", "shirt", "clvc_image_encoder_only"): 8702,
    ("7:3038", "toptee", "multimodal"): 1253,
    ("7:3038", "toptee", "text_only_black_image"): 1549,
    ("7:3038", "toptee", "clvc_image_encoder_only"): 1267,
    ("7:3790", "shirt", "multimodal"): 412,
    ("7:3790", "shirt", "text_only_black_image"): 1466,
    ("7:3790", "shirt", "clvc_image_encoder_only"): 5481,
    ("7:3790", "toptee", "multimodal"): 259,
    ("7:3790", "toptee", "text_only_black_image"): 568,
    ("7:3790", "toptee", "clvc_image_encoder_only"): 4535,
}


def _rank_from_note(note: str) -> int | None:
    """The probe note prints `· hạng đích <b>848</b> trên 74 381 ảnh ·`. Read that number."""
    match = re.search(r"hạng đích\s*#?\s*([\d.,\s\u00a0]+)", note or "")
    if not match:
        return None
    digits = re.sub(r"[^\d]", "", match.group(1).split("trên")[0])
    return int(digits) if digits else None


def _text(page, testid: str) -> str | None:
    node = page.get_by_test_id(testid)
    if node.count() == 0:
        return None
    return (node.first.inner_text() or "").strip()


def main() -> int:
    from playwright.sync_api import sync_playwright

    SHOTS.mkdir(parents=True, exist_ok=True)
    console_errors: list[str] = []
    page_errors: list[str] = []
    failed_requests: list[str] = []
    result: dict = {"base": BASE}

    with sync_playwright() as playwright:
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(BROWSERS))
        chrome = os.environ.get("WORKBENCH_CHROME_PATH", "/usr/bin/google-chrome")
        browser = (playwright.chromium.launch(executable_path=chrome)
                   if Path(chrome).is_file() else playwright.chromium.launch())
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda m: console_errors.append(f"{m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: page_errors.append(str(e)))
        page.on("requestfailed",
                lambda r: failed_requests.append(f"{r.method} {r.url} :: {r.failure}"))

        # In-flight bookkeeping for the diagnostic windows. A control change that lands while the
        # previous window is still downloading makes the app cancel that superseded fetch, which
        # shows up as ERR_ABORTED — a behaviour of the probe's pace, not of the page. Waiting for
        # zero in flight before each change means any abort recorded below is the app's own.
        inflight: set = set()
        page.on("request", lambda r: inflight.add(r.url) if "/candidates?" in r.url else None)
        succeeded: set = set()
        page.on("requestfinished", lambda r: (inflight.discard(r.url), succeeded.add(r.url)))
        page.on("requestfailed", lambda r: inflight.discard(r.url))

        def quiesce() -> None:
            page.wait_for_function(
                "() => true", timeout=NAV_MS)  # flush pending events
            for _ in range(600):
                if not inflight:
                    # A control change also re-renders React, so a request can be issued a tick
                    # after the last one finished. Dwell so the NEXT change supersedes nothing.
                    page.wait_for_timeout(300)
                    if not inflight:
                        return
                    continue
                page.wait_for_timeout(50)

        # ---- 1. overview: BOTH counts, each with its own label -------------------------------
        page.goto(f"{BASE}/?area=overview", wait_until="domcontentloaded", timeout=NAV_MS)
        page.get_by_test_id("overview-published-count").wait_for(timeout=NAV_MS)
        published = _text(page, "overview-published-count")
        # registry_count comes from /api/models/publication (a separate region); it renders an
        # em dash until that request lands, so reading it immediately records loading as a value.
        page.wait_for_function(
            """() => {
                 const el = document.querySelector('[data-testid="overview-registry-count"]');
                 return el && /\\d/.test(el.textContent || "");
               }""", timeout=NAV_MS)
        registry = _text(page, "overview-registry-count")
        labels = page.evaluate("""() => {
            const grab = (id) => {
                const v = document.querySelector(`[data-testid="${id}"]`);
                return v ? (v.parentElement?.innerText || "").replace(/\\s+/g, " ").trim() : null;
            };
            return {published: grab("overview-published-count"),
                    registry: grab("overview-registry-count")};
        }""")
        result["overview"] = {"published": published, "registry": registry, "label_blocks": labels,
                              "distinct_labels": labels["published"] != labels["registry"]}
        page.screenshot(path=str(SHOTS / "01-overview-1440x900.png"))

        # ---- 2. inventory: 16 published ----------------------------------------------------
        page.goto(f"{BASE}/?area=experiments&view=models", wait_until="domcontentloaded",
                  timeout=NAV_MS)
        page.get_by_test_id("count-models").wait_for(timeout=NAV_MS)
        inventory_count = _text(page, "count-models")
        inventory_rows = page.locator('[data-testid="model-inventory"] tbody tr').count()
        # A ROW whose model id is `pair`, not the substring somewhere in the panel: the panel also
        # carries protocol labels such as "Pair-Union Gallery", so a text search answers the wrong
        # question. Row identity comes from the model-id cell.
        inventory_ids = page.evaluate("""() => Array.from(
            document.querySelectorAll('[data-testid="model-inventory"] tbody tr'))
            .map((row) => (row.querySelector('th, td')?.innerText || '').trim())""")
        result["inventory"] = {"count_models": inventory_count, "row_count": inventory_rows,
                               "model_ids": inventory_ids,
                               "pair_row_present": "pair" in inventory_ids}
        page.screenshot(path=str(SHOTS / "02-inventory-1440x900.png"))

        # ---- 3. native table shows no PAIR rows (every protocol that has rows) --------------
        # `?view=native` with no protocol opens the CIRCUS release protocol, which has zero native
        # rows, so "no PAIR rows" would be vacuously true there. The catalog's rows live under three
        # other protocols; the claim is checked on each of them.
        native_tables = {}
        for protocol in ("fashioniq_val_split", "fashioniq_original_split",
                         "fashioniq_full_gallery_ref_excluded"):
            page.goto(f"{BASE}/?area=evaluation&view=native&protocol={protocol}&subset=full",
                      wait_until="domcontentloaded", timeout=NAV_MS)
            page.wait_for_selector(
                '[data-testid="native-table"], [data-testid="native-empty"]', timeout=NAV_MS)
            page.wait_for_timeout(800)
            native_tables[protocol] = {
                "rows": page.locator('[data-testid="native-table"] tbody tr').count(),
                "pair_rows": page.locator(
                    '[data-testid="native-table"] tbody tr', has_text="pair").count(),
            }
        page.screenshot(path=str(SHOTS / "03-native-table-1440x900.png"))
        result["native_table"] = native_tables
        result["native_table_pair_rows"] = sum(v["pair_rows"] for v in native_tables.values())

        # ---- 3b. deep link to a deferred model ---------------------------------------------
        # The notice is derived from the served catalog's `deferred_models`, so it appears only
        # after that request lands. Wait for the region to settle rather than race it.
        page.goto(f"{BASE}/?area=evaluation&view=native&model=pair", wait_until="domcontentloaded",
                  timeout=NAV_MS)
        page.wait_for_function(
            """() => {
                 const area = document.querySelector('[data-testid="area-native"]');
                 if (!area) return false;
                 if (document.querySelector('[data-testid="native-deferral-notice"]')) return true;
                 return !!document.querySelector(
                     '[data-testid="native-table"], [data-testid="native-empty"]');
               }""", timeout=NAV_MS)
        page.wait_for_timeout(1500)
        bare_deep_link_url = page.url
        bare_notice = _text(page, "native-deferral-notice")
        bare_rows = page.locator('[data-testid="native-table"] tbody tr').count()
        bare_area = page.locator('[data-testid="area-native"]').inner_text()
        page.screenshot(path=str(SHOTS / "03b-pair-deep-link-bare-1440x900.png"))

        # Control: the same page with BOTH scope halves named. The landing-scope effect rewrites the
        # URL only when the URL left protocol/subset unstated, so this isolates `model=`.
        page.goto(f"{BASE}/?area=evaluation&view=native"
                  f"&protocol=circus_fashioniq_release_v1&subset=full&model=pair",
                  wait_until="domcontentloaded", timeout=NAV_MS)
        page.wait_for_function(
            """() => !!document.querySelector(
                 '[data-testid="native-deferral-notice"], [data-testid="native-table"], [data-testid="native-empty"]')""",
            timeout=NAV_MS)
        page.wait_for_timeout(800)
        scoped_notice = _text(page, "native-deferral-notice")
        scoped_url = page.url
        page.screenshot(path=str(SHOTS / "03c-pair-deep-link-scoped-1440x900.png"))

        result["pair_deep_link"] = {
            "bare_url_requested": f"{BASE}/?area=evaluation&view=native&model=pair",
            "bare_url_final": bare_deep_link_url,
            "bare_notice": bare_notice,
            "bare_native_rows": bare_rows,
            "bare_area_renders": "Số đo native" in bare_area,
            "scoped_url_final": scoped_url,
            "scoped_notice": (scoped_notice or "")[:160],
            "scoped_notice_renders": bool(scoped_notice),
            "notice_param_survives_only_when_scope_is_named":
                bool(scoped_notice) and not bare_notice,
        }

        # ---- 4. parity card: `Chuẩn <string>` and the per-model verdicts ---------------------
        page.goto(f"{BASE}/?area=evaluation", wait_until="domcontentloaded", timeout=NAV_MS)
        page.get_by_test_id("parity-gate").wait_for(timeout=NAV_MS)
        page.wait_for_function(
            """() => {
                 const card = document.querySelector('[data-testid="parity-gate"]');
                 return card && !/Đang tính/.test(card.innerText);
               }""", timeout=NAV_MS)
        subtitle = page.evaluate("""() => {
            const card = document.querySelector('[data-testid="parity-gate"]');
            const sub = card?.querySelector(".card-sub, .subtitle, h2 + p, p");
            return (sub?.innerText || card?.innerText || "").slice(0, 300);
        }""")
        verdict = _text(page, "parity-gate-verdict")
        summary = _text(page, "parity-gate-summary")
        parity_rows = page.evaluate("""() => Array.from(
            document.querySelectorAll('[data-testid^="parity-row-"]')).map((row) => {
                const cells = Array.from(row.querySelectorAll("td"));
                // Cells: [0] verdict badge, [1] subcheck-FAIL count, [2] dimensions held.
                const failCell = (cells[1]?.innerText || "").trim();
                // The failure column renders either "0" or "<n> subcheck"; the count is the number
                // the API's `failures.length` reports, read off the page.
                const digits = (failCell.match(/^[0-9]+/) || [""])[0];
                return {
                    model: row.getAttribute("data-testid").replace("parity-row-", ""),
                    passed: row.getAttribute("data-passed"),
                    status: (cells[0]?.innerText || "").trim(),
                    failure_count: digits ? Number(digits) : 0,
                    failures: failCell,
                    dimensions: (cells[2]?.innerText || "").replace(/\\s+/g, " ").trim(),
                };
            })""")
        result["parity_failure_counts_dom"] = {
            row["model"]: row["failure_count"] for row in parity_rows}
        standard_text = next((line for line in (subtitle or "").split("\n") if "Chuẩn" in line), None)
        result["parity"] = {
            "standard_line": standard_text,
            "standard_is_string": bool(standard_text and "Chuẩn parity-relative/v1" in standard_text
                                       and "[" not in standard_text),
            "verdict_badge": verdict,
            "summary": summary,
            "rows": parity_rows,
        }
        page.screenshot(path=str(SHOTS / "04-parity-1440x900.png"))

        # ---- 5. CLVC diagnostic: all 12 cells, read from the DOM ----------------------------
        page.goto(f"{BASE}/?area=samples&view=explorer", wait_until="domcontentloaded",
                  timeout=NAV_MS)
        page.get_by_test_id("dual-diagnostic-model").wait_for(timeout=NAV_MS)
        quiesce()  # the default LIMN window is still in flight; superseding it is the probe's doing
        page.get_by_test_id("dual-diagnostic-model").select_option("clvc")
        page.get_by_test_id("limn-probe-qid").wait_for(timeout=NAV_MS)
        page.get_by_test_id("limn-probe-depth").select_option("100")
        for summary_node in page.locator("details.detail > summary").all():
            try:
                summary_node.click()
            except Exception:
                pass
        diag_title = _text(page, "limn-diagnostic-title")
        diag_canonical = _text(page, "limn-diagnostic-canonical")

        cells: list[dict] = []
        for qid in QUERIES:
            for scope in SCOPES:
                quiesce()
                page.get_by_test_id("limn-probe-qid").select_option(qid)
                page.get_by_test_id("limn-probe-scope").select_option(scope)
                for condition in CLVC_CONDITIONS:
                    quiesce()
                    page.get_by_test_id("limn-probe-condition").select_option(condition)
                    # Identity from the DOM, not from timing: the probe note opens with the served
                    # payload's own label (`diagnostic — <scope> scope, <condition> (not canonical)`),
                    # so waiting for THAT string proves the window on screen belongs to this cell.
                    # Waiting for "some new text" instead would accept the previous condition's
                    # window, and arming on a request would hang on the cell whose selection is
                    # already current and therefore fetches nothing.
                    page.wait_for_function(
                        """([scope, condition]) => {
                             const note = document.querySelector('[data-testid="limn-probe-note"]');
                             const grid = document.querySelector('[data-testid="limn-probe-grid"]');
                             if (!note || !grid) return false;
                             const text = note.textContent || '';
                             return text.includes(`${scope} scope, ${condition}`)
                                    && grid.querySelectorAll('article.candidate').length > 0;
                           }""", arg=[scope, condition], timeout=NAV_MS)
                    quiesce()
                    note = _text(page, "limn-probe-note") or ""
                    outside = _text(page, "limn-probe-target-outside")
                    inside = _text(page, "limn-probe-target-in-window")
                    self_match = _text(page, "dual-probe-self-match")
                    tile_count = page.locator(
                        '[data-testid="limn-probe-grid"] article.candidate').count()
                    ref_tile = page.locator(
                        '[data-testid="limn-probe-grid"] article.candidate.is-reference').count()
                    expected = INVENTORY[(qid, scope, condition)]
                    rendered = _rank_from_note(note)
                    cells.append({
                        "query": qid, "scope": scope, "condition": condition,
                        "inventory_rank": expected,
                        "rendered_rank": rendered,
                        "match": rendered == expected,
                        "target_statement": (outside or inside or "")[:200],
                        "target_outside_window": bool(outside),
                        "self_match_banner": (self_match or "")[:160],
                        "tiles": tile_count,
                        "reference_tiles": ref_tile,
                        "screenshot": None,
                    })
                    name = f"clvc-{qid.replace(':', '_')}-{scope}-{condition}.png"
                    page.screenshot(path=str(SHOTS / name))
                    cells[-1]["screenshot"] = f"screenshots/{name}"

        result["clvc_diagnostic"] = {
            "title": diag_title, "canonical_badge": diag_canonical,
            "cells": cells,
            "mismatches": [c for c in cells if not c["match"]],
            "cells_outside_window": sum(1 for c in cells if c["target_outside_window"]),
            "ablation_cells_with_self_match": sum(
                1 for c in cells if c["condition"] == "clvc_image_encoder_only"
                and c["self_match_banner"]),
            "ablation_cells": sum(1 for c in cells if c["condition"] == "clvc_image_encoder_only"),
        }
        quiesce()  # the last cell's window must land before the page is left
        page.screenshot(path=str(SHOTS / "05-clvc-diagnostic-panel-1440x900.png"), full_page=True)

        # ---- 6. the same page at 390x844, to record responsiveness --------------------------
        narrow = browser.new_page(viewport={"width": 390, "height": 844})
        narrow_console: list[str] = []
        narrow.on("console", lambda m: narrow_console.append(f"{m.type}: {m.text}")
                  if m.type == "error" else None)
        narrow.goto(f"{BASE}/?area=overview", wait_until="domcontentloaded", timeout=NAV_MS)
        narrow.get_by_test_id("overview-published-count").wait_for(timeout=NAV_MS)
        narrow.wait_for_function(
            """() => {
                 const el = document.querySelector('[data-testid="overview-registry-count"]');
                 return el && /\\d/.test(el.textContent || "");
               }""", timeout=NAV_MS)
        result["mobile"] = {
            "viewport": "390x844",
            "published": _text(narrow, "overview-published-count"),
            "registry": _text(narrow, "overview-registry-count"),
            "horizontal_overflow_px": narrow.evaluate(
                "() => document.documentElement.scrollWidth - window.innerWidth"),
            "console_errors": narrow_console,
        }
        narrow.screenshot(path=str(SHOTS / "06-overview-390x844.png"), full_page=True)
        narrow.goto(f"{BASE}/?area=samples&view=explorer", wait_until="domcontentloaded",
                    timeout=NAV_MS)
        narrow.get_by_test_id("dual-diagnostic-model").wait_for(timeout=NAV_MS)
        narrow.screenshot(path=str(SHOTS / "07-diagnostic-390x844.png"), full_page=True)
        narrow.close()

        browser.close()

    # An abort of a URL that ALSO completed for the same page is a superseded duplicate: the app
    # issued the same window twice (a control re-render) and the currency guard cancelled the older
    # copy. Distinguishing that from a real failure is the point of recording it this way.
    superseded = [text for text in failed_requests
                  if "ERR_ABORTED" in text and any(
                      url in succeeded for url in [text.split(" :: ")[0].split(" ", 1)[1]])]
    result["failed_requests_classification"] = {
        "total": len(failed_requests),
        "superseded_duplicate": len(superseded),
        "genuine": len(failed_requests) - len(superseded),
        "superseded": superseded,
    }
    result["console_errors"] = console_errors
    result["page_errors"] = page_errors
    result["failed_requests"] = failed_requests
    result["counts"] = {"console_errors": len(console_errors), "page_errors": len(page_errors),
                        "failed_requests": len(failed_requests)}

    clvc = result["clvc_diagnostic"]
    genuine_failures = result["failed_requests_classification"]["genuine"]
    criteria = {
        "overview_prints_both_counts_with_distinct_labels":
            result["overview"]["published"] == "16" and result["overview"]["registry"] == "17"
            and result["overview"]["distinct_labels"],
        "inventory_shows_16_and_no_pair_row":
            result["inventory"]["count_models"] == "16"
            and not result["inventory"]["pair_row_present"],
        "native_table_has_no_pair_rows": result["native_table_pair_rows"] == 0,
        # THE CRITERION AS WRITTEN: the bare deep link must render the deferral notice.
        "bare_deep_link_view_native_model_pair_shows_notice":
            bool(result["pair_deep_link"]["bare_notice"]),
        "parity_card_renders_standard_as_a_string":
            result["parity"]["standard_is_string"],
        # The names the acceptance prompt fixes, read from the rendered table.
        "parity_per_model_failure_counts_match":
            result["parity_failure_counts_dom"] == {
                "conesep": 0, "encoder": 9, "habit": 2, "hint": 2, "intent": 0,
                "limn": 3, "tgcir": 3},
        "clvc_12_cells_match_inventory": not clvc["mismatches"],
        "ablation_cells_show_self_match_and_outside_window":
            clvc["ablation_cells_with_self_match"] == clvc["ablation_cells"] == 4
            and clvc["cells_outside_window"] == 12,
        "no_console_errors": not console_errors,
        "no_page_errors": not page_errors,
        "no_genuine_failed_requests": not genuine_failures,
    }
    result["acceptance_criteria"] = criteria
    result["failed_criteria"] = [name for name, held in criteria.items() if not held]
    ok = all(criteria.values())
    result["verdict"] = "PASS" if ok else "FAIL"
    (HERE / "live-verify.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"verdict": result["verdict"], "counts": result["counts"],
                      "mismatches": len(clvc["mismatches"]),
                      "request_failures": result["failed_requests_classification"],
                      "overview": [result["overview"]["published"],
                                   result["overview"]["registry"]],
                      "inventory": result["inventory"]["count_models"],
                      "pair_notice_scoped": result["pair_deep_link"]["scoped_notice_renders"],
                      "pair_notice_bare": bool(result["pair_deep_link"]["bare_notice"]),
                      "native_pair_rows": result["native_table_pair_rows"],
                      "standard_line": result["parity"]["standard_line"],
                      "failed_criteria": result["failed_criteria"]},
                     ensure_ascii=False, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
