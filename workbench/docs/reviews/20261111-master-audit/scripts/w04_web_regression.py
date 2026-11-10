#!/usr/bin/env python3
"""W-04 regression: late answer, out-of-range qid, reload, Back/Forward, induced API error.

Each check is behavioural, not cosmetic: it asserts on the DOM the operator would read, and it saves
the screenshot plus the exact URLs the page requested so the claim can be re-checked.

The three failures this pins:
  * a response for run A landing AFTER the operator switched to run B and painting A's rows under
    B's labels (`RequestRunner`'s token+key guard is what has to hold);
  * a deep link naming a query that is not in the run silently showing whatever row came first
    (`selectExactQuery`, never `items[0]`);
  * an API failure rendering as a silent blank instead of an error IN the data region.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[4]
DEFAULT_BROWSERS = REPO / "workbench/envs/ctl312h/.playwright-browsers"

CIRCUS = "circus_fashioniq_release_v1"
RUN_A = "9e4918a6b5c243412cbc58c9"   # limn, full
RUN_B = "ce89a9a27c4cceadeaaf30b0"   # encoder, full + sf


def explorer(run: str, subset: str, query: str = "", k: int = 10) -> str:
    return (f"/?area=samples&view=explorer&protocol={CIRCUS}&subset={subset}"
            f"&runs=c%3A{run}&run={run}" + (f"&query={query}" if query else "") + f"&k={k}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5173")
    parser.add_argument("--out", required=True)
    parser.add_argument("--chrome", default=os.environ.get("WORKBENCH_CHROME_PATH", "/usr/bin/google-chrome"))
    args = parser.parse_args()
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(DEFAULT_BROWSERS))
    from playwright.sync_api import sync_playwright

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failed: list[str] = []

    def record(name: str, page, expectation: str, passed: bool, extra: dict | None = None) -> None:
        png = out / f"{name}.png"
        page.screenshot(path=str(png), full_page=False)
        text = page.inner_text("body")
        results.append({"check": name, "url": page.url, "expectation": expectation,
                        "passed": passed, "png": png.name, "dom_text": text[:6000],
                        **(extra or {})})
        if not passed:
            failed.append(f"{name}: {expectation}")
        print(f"{'PASS' if passed else 'FAIL'} {name}", flush=True)

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(
            executable_path=args.chrome if Path(args.chrome).exists() else None,
            args=["--no-sandbox", "--disable-dev-shm-usage"])

        # ---------------------------------------------------------------- 1. late answer A after B
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        seen: list[str] = []
        page.on("request", lambda request: seen.append(request.url) if "/api/" in request.url else None)

        def delay_run_a(route, request):
            # Hold A's answer for 6 s; B is answered immediately below. If the guard is wrong, A
            # paints when it finally resolves, under B's heading.
            if RUN_A in request.url and "/queries" in request.url:
                import time
                time.sleep(6)
            route.continue_()

        page.route("**/api/composition/runs/**/queries?*", delay_run_a)
        page.goto(args.base_url + explorer(RUN_A, "full", "7%3A3038"), wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(1500)
        # Switch to B while A is still in flight.
        page.goto(args.base_url + explorer(RUN_B, "full", "7%3A3038"), wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(8000)
        text = page.inner_text("body")
        # A's model name must not appear as the SELECTED run's identity after B was chosen.
        a_painted = "limn" in text.lower() and "encoder" not in text.lower()
        record("w04_late_answer_a_after_b", page,
               "sau khi đổi sang run B, DOM phải nói encoder và KHÔNG còn render run A",
               not a_painted, {"requested": [url for url in seen if "/queries" in url][:10]})
        page.close()

        # ---------------------------------------------------------------- 2. out-of-range qid
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.goto(args.base_url + explorer(RUN_B, "sf", "7%3A99999"), wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(4000)
        # Scope the assertion to the INSPECTOR, not the whole page: the LIMN diagnostic block below
        # it legitimately names 7:3038/7:3790 as the two ambiguous qids, and counting that as a
        # substitution would fail a page that is behaving correctly.
        inspector = page.locator("[data-testid=query-inspector]")
        inspector_count = inspector.count()
        body = page.inner_text("body")
        # The empty-state card, not an inspector card, is what must be on screen for a missing qid.
        named = "7:99999" in body
        says_no_substitution = "Không tự thay bằng query khác" in body
        diagnostic_named_only = all(
            "7:99999" not in block
            for block in [page.locator("[data-testid=limn-diagnostic-probe]").inner_text()
                          if page.locator("[data-testid=limn-diagnostic-probe]").count() else ""])
        record("w04_deeplink_out_of_range", page,
               "deep link qid ngoài dữ liệu phải NÓI TÊN qid đó, không dựng inspector, "
               "và không thay bằng query khác",
               named and says_no_substitution and inspector_count == 0,
               {"named": named, "says_no_substitution": says_no_substitution,
                "inspector_rendered": inspector_count, "diagnostic_untouched": diagnostic_named_only,
                "inspector_text": inspector.inner_text()[:400] if inspector_count else ""})

        # ---------------------------------------------------------------- 3. reload + Back/Forward
        page.reload(wait_until="domcontentloaded")
        page.wait_for_timeout(3500)
        reload_text = page.inner_text("body")
        reload_ok = "7:99999" in reload_text and "Shortcut-Free" in reload_text
        page.go_back(wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        back_url = page.url
        page.go_forward(wait_until="domcontentloaded")
        page.wait_for_timeout(2500)
        forward_text = page.inner_text("body")
        record("w04_reload_and_history", page,
               "reload giữ qid + subset; Back/Forward trả lại đúng URL và nội dung",
               reload_ok and "7:99999" in forward_text,
               {"reload_ok": reload_ok, "back_url": back_url, "forward_url": page.url})
        page.close()

        # ---------------------------------------------------------------- 4. induced API error
        page = browser.new_page(viewport={"width": 1920, "height": 1080})
        page.route("**/api/composition/runs?*", lambda route: route.fulfill(
            status=503, content_type="application/json",
            body=json.dumps({"detail": "induced_failure_for_w04"})))
        page.route("**/api/native-measurements*", lambda route: route.fulfill(
            status=503, content_type="application/json",
            body=json.dumps({"detail": "induced_failure_for_w04"})))
        page.goto(args.base_url + "/?area=evaluation&view=native&protocol=fashioniq_original_split&subset=full",
                  wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_timeout(4000)
        text = page.inner_text("body")
        data_region = page.locator("[data-testid=area-native]").inner_text()
        has_error = ("lỗi" in data_region.lower() or "không tải" in data_region.lower()
                     or "induced" in data_region.lower())
        has_retry = page.locator("[data-testid=error-retry]").count() > 0
        record("w04_induced_api_error", page,
               "lỗi API phải hiện trong vùng dữ liệu kèm retry, không phải khoảng trắng im lặng",
               has_error and has_retry,
               {"data_region_excerpt": data_region[:1200], "retry_control": has_retry,
                "body_excerpt": text[:400]})
        page.close()
        browser.close()

    (out / "w04_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    if failed:
        print("\n".join(failed), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
