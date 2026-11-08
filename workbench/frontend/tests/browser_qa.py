#!/usr/bin/env python3
"""Reusable real-browser QA sweep: screenshot + overflow/console/image report.

For each --viewport: load --base-url, wait for network idle, save a PNG, and record
console errors, page errors, failed requests, DOM node count, document scrollWidth vs
viewport width, and elements wider than the viewport.

Prints one JSON document to stdout. Exit code 1 if any viewport has console errors,
page errors, failed requests, or whole-page horizontal overflow. Exit 2 if a page
could not be loaded at all.

Dependencies: playwright (project-local) + its managed chromium.
    workbench/envs/ctl312h/bin/python workbench/frontend/tests/browser_qa.py \
        --base-url http://127.0.0.1:5173 --viewport 1440x900 --viewport 390x844 \
        --out /tmp/browser-qa

Browser location: PLAYWRIGHT_BROWSERS_PATH is set from the project dir when unset, and
WORKBENCH_CHROME_PATH overrides the binary with any installed chrome (falls back to
managed chromium).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DEFAULT_BROWSERS = REPO / "workbench/envs/ctl312h/.playwright-browsers"
DEFAULT_CHROME = Path("/usr/bin/google-chrome")
MAX_LISTED = 20
NAV_TIMEOUT_MS = 60_000
NETWORK_IDLE_MS = 15_000
# Browsers auto-request /favicon.ico; a missing one is not a UI defect and would fail every
# app that does not ship one. ponytail: only favicon is exempt; add no other known-noise list.
IGNORE_URL_SUFFIXES = ("/favicon.ico",)

def ignorable(url: str) -> bool:
    return url.split("?")[0].endswith(IGNORE_URL_SUFFIXES)

# --------------------------------------------------------------------------- helpers

def parse_viewport(text: str) -> tuple[int, int, str]:
    width, _, height = text.lower().partition("x")
    if not (width.isdigit() and height.isdigit()) or int(width) < 1 or int(height) < 1:
        raise argparse.ArgumentTypeError(f"viewport must be WxH, got {text!r}")
    return int(width), int(height), f"{width}x{height}"

JS_PROBE = """
() => {
  const vw = window.innerWidth;
  const path = (el) => {
    const parts = [];
    for (let n = el; n && n.nodeType === 1 && parts.length < 6; n = n.parentElement) {
      let s = n.tagName.toLowerCase();
      if (n.id) { parts.unshift(s + '#' + n.id); break; }
      const cls = (typeof n.className === 'string' ? n.className : '').trim().split(/\\s+/).filter(Boolean);
      if (cls.length) s += '.' + cls.slice(0, 2).join('.');
      parts.unshift(s);
    }
    return parts.join(' > ');
  };
  // Inner text of a node, so a report line names the offending content, not just a div.
  const label = (el) => (el.textContent || '').replace(/\\s+/g, ' ').trim().slice(0, 60);

  const wide = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;      // display:none / detached
    if (r.width <= vw + 1 && r.right <= vw + 1) continue;
    wide.push({ selector: path(el), width: Math.round(r.width), right: Math.round(r.right),
                text: label(el) });
    if (wide.length >= 200) break;                      // cap the walk on pathological pages
  }

  const images = [...document.images].map((img) => ({
    src: (img.currentSrc || img.src || '').slice(0, 120),
    naturalWidth: img.naturalWidth, naturalHeight: img.naturalHeight,
    complete: img.complete, selector: path(img),
  }));

  return {
    innerWidth: vw,
    innerHeight: window.innerHeight,
    scrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body ? document.body.scrollWidth : null,
    scrollHeight: document.documentElement.scrollHeight,
    nodeCount: document.getElementsByTagName('*').length,
    title: document.title,
    wideElements: wide,
    images,
  };
}
"""

def attach_recorders(page) -> dict:
    """Collect console/page/request failures while the page is alive."""
    seen = {"console": [], "pageErrors": [], "failedRequests": [], "httpErrors": [], "console_errors": 0}

    def on_console(msg):
        if msg.type not in ("error", "warning"):
            return
        if ignorable(msg.location.get("url") or ""):
            return
        seen["console"].append({"type": msg.type, "text": msg.text[:500], "location": msg.location})
        if msg.type == "error":
            seen["console_errors"] += 1

    def on_page_error(error):
        seen["pageErrors"].append(str(error)[:500])

    def on_request_failed(request):
        if ignorable(request.url):
            return
        failure = request.failure
        seen["failedRequests"].append({
            "url": request.url[:200], "resourceType": request.resource_type,
            "error": (failure or "unknown")[:300],
        })

    def on_response(response):
        if response.status >= 400 and not ignorable(response.url):
            seen["httpErrors"].append({"url": response.url[:200], "status": response.status})

    page.on("console", on_console)
    page.on("pageerror", on_page_error)
    page.on("requestfailed", on_request_failed)
    page.on("response", on_response)
    return seen

def first(items: list) -> list:
    return items[:MAX_LISTED]

def png_size(path: Path) -> list | None:
    """Width/height from the PNG IHDR chunk. Stdlib only, no image decoding."""
    try:
        head = path.read_bytes()[:33]
    except OSError:
        return None
    if head[:8] != b"\x89PNG\r\n\x1a\n" or head[12:16] != b"IHDR":
        return None
    return [int.from_bytes(head[16:20], "big"), int.from_bytes(head[20:24], "big")]

def capture(browser, base_url: str, viewport: tuple[int, int, str], out: Path) -> dict:
    from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

    width, height, name = viewport
    context = browser.new_context(viewport={"width": width, "height": height})
    page = context.new_page()
    recorded = attach_recorders(page)
    result: dict = {"viewport": name, "url": base_url, "screenshot": None, "loadState": None}
    try:
        # A hard navigation failure (DNS, refused port, bad TLS) must be fatal: there is nothing to
        # QA. networkidle is only a settle hint, so a timeout there degrades to "load".
        page.goto(base_url, wait_until="load", timeout=NAV_TIMEOUT_MS)
        try:
            page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_MS)
            result["loadState"] = "networkidle"
        except PlaywrightTimeoutError:
            result["loadState"] = f"load (networkidle not reached in {NETWORK_IDLE_MS // 1000}s)"
        page.wait_for_timeout(250)  # let post-load paint/font/layout settle

        shot = out / f"{name}.png"
        page.screenshot(path=str(shot), full_page=False)
        result["screenshot"] = str(shot)
        result["screenshotBytes"] = shot.stat().st_size if shot.exists() else 0
        result["screenshotSize"] = png_size(shot)

        probe = page.evaluate(JS_PROBE)
        result.update({k: probe[k] for k in ("innerWidth", "innerHeight", "scrollWidth", "bodyScrollWidth",
                                             "scrollHeight", "nodeCount", "title")})
        overflow = max(probe["scrollWidth"], probe["bodyScrollWidth"] or 0)
        result["horizontalOverflow"] = overflow > probe["innerWidth"] + 1
        result["overflowPixels"] = max(0, overflow - probe["innerWidth"])
        result["wideElements"] = first(probe["wideElements"])
        result["wideElementCount"] = len(probe["wideElements"])

        broken = [i for i in probe["images"] if i["complete"] and i["naturalWidth"] == 0]
        pending = [i for i in probe["images"] if not i["complete"]]
        result["imageCount"] = len(probe["images"])
        result["brokenImages"] = first(broken)
        result["brokenImageCount"] = len(broken)
        result["unloadedImages"] = first(pending)
    except Exception as exc:
        result["fatal"] = f"{type(exc).__name__}: {str(exc)[:500]}"
    finally:
        result["console"] = first(recorded["console"])
        result["consoleErrorCount"] = recorded["console_errors"]
        result["consoleWarningCount"] = sum(1 for c in recorded["console"] if c["type"] == "warning")
        result["pageErrors"] = first(recorded["pageErrors"])
        result["pageErrorCount"] = len(recorded["pageErrors"])
        result["failedRequests"] = first(recorded["failedRequests"])
        result["failedRequestCount"] = len(recorded["failedRequests"])
        result["httpErrors"] = first(recorded["httpErrors"])
        result["httpErrorCount"] = len(recorded["httpErrors"])
        context.close()

    failures = []
    if result.get("fatal"):
        failures.append("fatal")
    if recorded["console_errors"]:
        failures.append(f"{recorded['console_errors']} console error(s)")
    if recorded["pageErrors"]:
        failures.append(f"{len(recorded['pageErrors'])} page error(s)")
    if recorded["failedRequests"]:
        failures.append(f"{len(recorded['failedRequests'])} failed request(s)")
    if recorded["httpErrors"]:
        failures.append(f"{len(recorded['httpErrors'])} HTTP >=400 response(s)")
    if result.get("horizontalOverflow"):
        failures.append(f"horizontal overflow +{result['overflowPixels']}px")
    if result.get("brokenImageCount"):
        failures.append(f"{result['brokenImageCount']} image(s) with naturalWidth 0")
    result["pass"] = not failures
    result["failures"] = failures
    return result

# --------------------------------------------------------------------------- main

def launch(playwright):
    chrome = os.environ.get("WORKBENCH_CHROME_PATH")
    if chrome is None and DEFAULT_CHROME.exists():
        chrome = str(DEFAULT_CHROME)
    if chrome:
        return playwright.chromium.launch(executable_path=chrome)
    return playwright.chromium.launch()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--viewport", type=parse_viewport, action="append", default=None,
                        help="WxH, repeatable (default 1440x900 and 390x844)")
    parser.add_argument("--out", type=Path, default=Path("/tmp/browser-qa"))
    args = parser.parse_args(argv)

    viewports = args.viewport or [parse_viewport("1440x900"), parse_viewport("390x844")]
    args.out.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH") and DEFAULT_BROWSERS.exists():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(DEFAULT_BROWSERS)
    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as playwright:
        browser = launch(playwright)
        try:
            for viewport in viewports:
                results.append(capture(browser, args.base_url, viewport, args.out))
        finally:
            browser.close()

    summary = {
        "baseUrl": args.base_url,
        "out": str(args.out),
        "viewports": results,
        "ok": all(r["pass"] for r in results),
        "fatal": any(r.get("fatal") for r in results),
    }
    print(json.dumps(summary, indent=2))
    if summary["fatal"]:
        return 2
    return 0 if summary["ok"] else 1

if __name__ == "__main__":
    sys.exit(main())
