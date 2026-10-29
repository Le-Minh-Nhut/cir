#!/usr/bin/env python3
"""Dev-only DOM-verified UI smoke for the CIR Failure Analysis Workbench.

Never reads pixels: every check is a locator, textContent, computed style or boundingBox.
Screenshots are SAVED for the human operator and never opened here.

Modes
  --build-fixture DIR   write two protocol-compatible mock results + real small PNGs, then exit
  --url URL             run the smoke against an already-served workbench
                        (--session cold | fixture | all; default all)

Browsers: PLAYWRIGHT_BROWSERS_PATH must point at the installed chromium.
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

PROTOCOL = "fashioniq_val_split"
CATEGORIES = ("dress", "shirt", "toptee")
TOP_K_SAVED = 20
VIEWPORT_DESKTOP = {"width": 1440, "height": 900}
VIEWPORT_MOBILE = {"width": 390, "height": 844}
NAV_TIMEOUT_MS = 90_000


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
    events raised *while a route abort is armed* are tagged at the moment they arrive. Comparing
    message text against the injected URL cannot work: the console entry does not carry one.
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

    def attach(self, page) -> None:
        page.on("console", lambda m: self._console(m.text) if m.type == "error" else None)
        page.on("pageerror", lambda e: self.page_errors.append(str(e)))
        page.on("requestfailed", lambda r: self._request(
            f"{r.method} {r.url} :: {r.failure}"))

    def _console(self, text: str) -> None:
        (self.console_injected if self.injecting else self.console).append(text)

    def _request(self, text: str) -> None:
        (self.failed_requests_injected if self.injecting else self.failed_requests).append(text)

    def note(self, text: str) -> None:
        self.observations.append(text)
        print(f"[DOM-VERIFIED] {text}", flush=True)


def api_json(base_url: str, path: str):
    with urllib.request.urlopen(f"{base_url}{path}", timeout=NAV_TIMEOUT_MS // 1000) as response:
        return json.loads(response.read().decode("utf-8"))


def goto(page, base_url: str, recorder: Recorder) -> None:
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector("nav button", timeout=NAV_TIMEOUT_MS)
    # The header reports API failures in this aside; an error here is a real defect, not a probe.
    try:
        page.wait_for_function(
            "() => !document.querySelector('aside.error')", timeout=20_000)
    except Exception:
        recorder.note(f"WARNING: an API error aside was present after load: "
                      f"{page.locator('aside.error').inner_text()!r}")


def open_page(page, name: str) -> None:
    page.locator("nav button", has_text=name).first.click()
    page.locator("section h2", has_text=name).first.wait_for(timeout=NAV_TIMEOUT_MS)


def select_protocol(page, protocol_id: str) -> None:
    """Pick the protocol in the global control; the explorer only lists compatible runs.

    The run list is fetched once on mount, so the caller MUST have waited for it first
    (`wait_for_runs_loaded`) -- switching the protocol before then leaves `compatible` empty and
    every downstream control disabled, which looks like a UI failure but is a test race.
    """
    select = page.locator("section.controls select").first
    select.wait_for(timeout=NAV_TIMEOUT_MS)
    select.select_option(protocol_id)
    page.wait_for_function(
        "id => document.querySelector('section.controls select').value === id",
        arg=protocol_id, timeout=NAV_TIMEOUT_MS)


def wait_for_runs_loaded(page, expected_runs: int) -> None:
    """The dashboard's run counters come from `/api/runs`; wait for the fetch to have landed."""
    page.locator("div.cards b", has_text="mock runs").first.wait_for(timeout=NAV_TIMEOUT_MS)
    page.wait_for_function(
        "n => [...document.querySelectorAll('div.cards b')].some(b => "
        "b.textContent.trim() === `${n} mock runs`)",
        arg=expected_runs, timeout=NAV_TIMEOUT_MS)


def wait_for_pager_ready(page) -> str:
    """Wait until the pager reports a real, settled window.

    On mount `activeRunId` is "" until `/api/runs` lands, and the explorer's `[activeRunId]` effect
    resets the query page. Acting in that window sees a rendered query beside a dead
    `1–0 of 0` pager; the UI converges on its own within a moment, so this is a wait, not a defect.
    """
    page.wait_for_function(
        "() => /^\\d+–\\d+ of \\d+$/.test(document.querySelector('div.pager span').textContent.trim()) "
        "&& (document.querySelectorAll('div.pager span')[0].textContent.match(/of (\\d+)$/)||[])[1]"
        " !== '0'",
        timeout=NAV_TIMEOUT_MS)
    return page.locator("div.pager span").first.inner_text()


def open_query(page, query_id: str) -> None:
    open_page(page, "Sample Explorer")
    wait_for_pager_ready(page)
    page.get_by_placeholder("Paste query ID").fill(query_id)
    page.locator("div.filters button", has_text="Jump").click()
    page.locator("h3", has_text=query_id).wait_for(timeout=NAV_TIMEOUT_MS)
    wait_for_pager_ready(page)


def image_evidence(page) -> list[dict]:
    return page.evaluate("""() => [...document.querySelectorAll('figure img')].map(img => ({
        src: img.getAttribute('src'), alt: img.getAttribute('alt'),
        complete: img.complete, naturalWidth: img.naturalWidth,
        visible: getComputedStyle(img).display !== 'none',
        width: Math.round(img.getBoundingClientRect().width)}))""")


def shot(page, shots: Path, name: str) -> str:
    shots.mkdir(parents=True, exist_ok=True)
    path = shots / f"{name}.png"
    page.screenshot(path=str(path), full_page=True)
    return str(path)


def overflow(page) -> dict:
    return page.evaluate("""() => ({
        scrollWidth: document.body.scrollWidth, innerWidth: window.innerWidth,
        widest: [...document.querySelectorAll('main *')].map(el => el.getBoundingClientRect().width)
                .reduce((a, b) => Math.max(a, b), 0)})""")


# --------------------------------------------------------------------------- sessions

def session_cold(page, base_url: str, shots: Path, recorder: Recorder) -> list[str]:
    """Session A: empty temp roots. 17 model entries, loading completes, truthful blocked reasons."""
    saved = []
    goto(page, base_url, recorder)
    # The dashboard card is the UI's own count of the registry: loading must COMPLETE, not just
    # render a placeholder. Wait for the count itself, then read it.
    card = page.locator("div.cards b", has_text="registered models").first
    card.wait_for(timeout=NAV_TIMEOUT_MS)
    page.wait_for_function(
        "() => /^17 registered models$/.test(document.querySelector('div.cards b').textContent.trim())",
        timeout=NAV_TIMEOUT_MS)
    dashboard = page.locator("div.cards").first.inner_text()
    recorder.note(f"cold inventory: dashboard cards read {dashboard!r}")
    open_page(page, "Models")

    models = api_json(base_url, "/api/models")
    rows = page.locator("section table tbody tr")
    rows.first.wait_for(timeout=NAV_TIMEOUT_MS)
    row_count = rows.count()
    recorder.note(f"cold inventory: API /api/models returned {len(models)} entries, "
                  f"DOM rendered {row_count} model rows")
    assert len(models) == 17, f"expected 17 model entries from the API, got {len(models)}"
    assert row_count == 17, f"expected 17 rendered model rows, got {row_count}"

    blocked = {m["model_id"]: [r for c in m["checkpoints"] for r in (c.get("blocking_reasons") or [])]
               for m in models}
    blocked = {k: v for k, v in blocked.items() if v}
    dom_text = [rows.nth(i).inner_text() for i in range(row_count)]
    shown = sum(1 for text in dom_text if "Blockers:" in text)
    recorder.note(f"cold inventory: {len(blocked)} models declare blocking_reasons in the API; "
                  f"{shown} rendered rows show a 'Blockers:' line")
    for model_id, reasons in list(blocked.items())[:3]:
        row = next((text for text in dom_text if reasons[0] in text), None)
        assert row is not None, f"{model_id}: API reason {reasons[0]!r} not rendered"
    recorder.note(f"cold inventory: sample truthful reason rendered, e.g. "
                  f"{list(blocked.values())[0][0]!r}")
    saved.append(shot(page, shots, "desktop-inventory"))
    return saved


def session_fixture(page, base_url: str, shots: Path, recorder: Recorder) -> list[str]:
    saved = []
    goto(page, base_url, recorder)
    # The default protocol (fashioniq_original_split) has no run here, so the compatible run set is
    # empty until the protocol control names the fixture's protocol. This is the UI's own routing.
    runs = api_json(base_url, "/api/runs")
    compatible = [r for r in runs if r["protocol_id"] == PROTOCOL]
    # Wait for the dashboard's mock-run counter to reach the API's own count, i.e. the run list
    # fetch has landed and `compatible` is populated. Only then is the protocol switch meaningful.
    wait_for_runs_loaded(page, len(runs))
    select_protocol(page, PROTOCOL)
    recorder.note(f"fixture: {len(runs)} runs in the index, {len(compatible)} compatible with "
                  f"{PROTOCOL} ({[r['run_id'] for r in compatible]})")
    assert len(compatible) == 2, f"expected 2 compatible fixture runs, got {len(compatible)}"

    query_id = "dress:0:dressref0:dresstgt0"

    # --- Flow 1: open a query, reference/target/Top-K image elements present with correct src.
    open_query(page, query_id)
    page.locator("figure img").first.wait_for(timeout=NAV_TIMEOUT_MS)
    page.wait_for_function(
        "() => [...document.querySelectorAll('figure img')].every(i => i.complete)",
        timeout=NAV_TIMEOUT_MS)
    images = image_evidence(page)
    recorder.note(f"flow 1: {len(images)} img elements rendered; srcs={[i['src'] for i in images]}")
    for image in images:
        assert image["src"].startswith("/api/images/"), f"unexpected src {image['src']!r}"
    reference = [i for i in images if i["alt"] == "Reference: dressref0"]
    target = [i for i in images if i["alt"] == "Target: dresstgt0"]
    assert reference, f"reference image missing; alts={[i['alt'] for i in images]}"
    assert target, f"target image missing; alts={[i['alt'] for i in images]}"
    assert reference[0]["src"] == "/api/images/dress/dressref0", reference[0]["src"]
    assert target[0]["src"] == "/api/images/dress/dresstgt0", target[0]["src"]
    topk = [i for i in images if (i["alt"] or "").startswith("#")]
    live = [i for i in topk if i["naturalWidth"] > 0 and i["visible"]]
    recorder.note(f"flow 1: reference+target srcs exact; {len(topk)} Top-K img elements, "
                  f"{len(live)} loaded (naturalWidth>0, displayed)")
    assert len(topk) == 10, f"expected 10 Top-K images at default K=10, got {len(topk)}"
    assert len(live) == len(topk), (
        "some Top-K images did not load: "
        f"{[i for i in topk if not (i['naturalWidth'] > 0 and i['visible'])]}")
    assert reference[0]["naturalWidth"] > 0 and target[0]["naturalWidth"] > 0, images[:2]
    saved.append(shot(page, shots, "desktop-query"))

    # --- Flow 2: a filter change and one pagination step change the visible result set.
    pager = wait_for_pager_ready(page)
    page.locator("div.pager button", has_text="Next page").click()
    page.locator("div.pager span").first.wait_for(timeout=NAV_TIMEOUT_MS)
    page.wait_for_function(
        "() => document.querySelector('div.pager span').textContent.trim().startsWith('21')",
        timeout=NAV_TIMEOUT_MS)
    pager_after = page.locator("div.pager span").first.inner_text()
    recorder.note(f"flow 2: pagination {pager!r} -> {pager_after!r}")
    assert pager != pager_after, "pagination step did not move the window"

    page.locator("div.filters select").first.select_option("shirt")
    page.wait_for_function(
        "() => document.querySelector('div.pager span').textContent.trim().startsWith('1')",
        timeout=NAV_TIMEOUT_MS)
    category_total = page.locator("div.pager span").first.inner_text()
    open_query(page, "shirt:0:shirtref0:shirttgt0")
    recorder.note(f"flow 2: category filter 'shirt' -> pager {category_total!r}; "
                  f"query switched to shirt universe")

    # --- Flow 3: annotation save -> reload -> persists.
    open_query(page, query_id)
    open_page(page, "Annotations")
    page.locator("div.labels label").first.wait_for(timeout=NAV_TIMEOUT_MS)
    label = page.locator("div.labels label").first
    label_text = label.inner_text()
    label.locator("input").check()
    page.locator("textarea").fill("dom-verified smoke note")
    page.locator("section button", has_text="Save / update").click()
    page.wait_for_function(
        "() => document.querySelector('textarea').value === 'dom-verified smoke note'",
        timeout=NAV_TIMEOUT_MS)
    stored = api_json(base_url, f"/api/annotations?query_id={query_id}&protocol_id={PROTOCOL}")
    recorder.note(f"flow 3: saved label {label_text!r} via the UI; API now holds "
                  f"{len(stored)} record(s), labels={stored[0]['labels'] if stored else None}")

    page.reload(wait_until="domcontentloaded")
    page.wait_for_selector("nav button", timeout=NAV_TIMEOUT_MS)
    wait_for_runs_loaded(page, 2)
    select_protocol(page, PROTOCOL)
    open_query(page, query_id)
    open_page(page, "Annotations")
    page.locator("div.labels label").first.wait_for(timeout=NAV_TIMEOUT_MS)
    checked = page.locator("div.labels label", has_text=label_text).first.locator("input").is_checked()
    note_value = page.locator("textarea").input_value()
    recorder.note(f"flow 3: after reload the label checkbox is checked={checked} "
                  f"and the note reads {note_value!r}")
    assert checked, "annotation label did not survive a reload"
    assert note_value == "dom-verified smoke note", note_value

    # --- Flow 4: one bounded injected API failure -> visible error state, then recovery.
    open_page(page, "Sample Explorer")
    wait_for_pager_ready(page)

    def abort_queries(route):
        recorder.injected.append(f"INJECTED abort {route.request.url}")
        route.abort()

    recorder.injecting = True
    page.route("**/api/queries*", abort_queries)
    page.locator("div.pager button", has_text="Next page").click()
    page.locator("aside.error").wait_for(timeout=30_000)
    recorder.injecting = False
    error_text = page.locator("aside.error").inner_text()
    spinner = page.evaluate(
        "() => [...document.querySelectorAll('*')].filter(e => "
        "/spinner|loading/i.test(e.className || '')).length")
    recorder.note(f"flow 4: injected failure surfaced as {error_text!r}; "
                  f"loading-class elements on screen = {spinner}")
    assert "Guard/API error" in error_text, error_text
    saved.append(shot(page, shots, "error-state"))

    page.unroute("**/api/queries*", abort_queries)
    # The aborted click still advanced the offset (React's onClick ran before the fetch failed), so
    # the pager already reads the last window: the enabled navigation is Previous, not Next.
    page.locator("aside.error button", has_text="Dismiss").click()
    page.locator("aside.error").wait_for(state="detached", timeout=NAV_TIMEOUT_MS)
    page.locator("div.pager button", has_text="Previous page").click()
    page.wait_for_function(
        "() => document.querySelector('div.pager span').textContent.trim().startsWith('1')",
        timeout=NAV_TIMEOUT_MS)
    recovered = page.locator("div.pager span").first.inner_text()
    errors_after = page.locator("aside.error").count()
    recorder.note(f"flow 4: after unroute+Dismiss a real request succeeded (pager {recovered!r}), "
                  f"error asides on screen = {errors_after} (recovered)")
    assert errors_after == 0, "error aside reappeared after recovery"
    return saved


def session_mobile(browser, base_url: str, shots: Path, recorder: Recorder) -> list[str]:
    """Mobile 390x844: controls reachable, no horizontal page overflow, images bounded."""
    context = browser.new_context(viewport=VIEWPORT_MOBILE)
    page = context.new_page()
    recorder.attach(page)
    saved = []
    try:
        goto(page, base_url, recorder)
        wait_for_runs_loaded(page, 2)
        select_protocol(page, PROTOCOL)
        open_page(page, "Models")
        page.locator("section table tbody tr").first.wait_for(timeout=NAV_TIMEOUT_MS)
        controls = page.locator("nav button").count()
        reachable = page.evaluate("""() => [...document.querySelectorAll('nav button, .controls select')]
            .every(el => { const r = el.getBoundingClientRect();
                           return r.width > 0 && r.height > 0 && r.left >= -1 &&
                                  r.right <= window.innerWidth + 2; })""")
        metrics = overflow(page)
        recorder.note(f"mobile inventory: {controls} nav controls, all reachable={reachable}, "
                      f"body.scrollWidth={metrics['scrollWidth']} innerWidth={metrics['innerWidth']}")
        assert reachable, "a nav/control element is off-viewport"
        assert metrics["scrollWidth"] <= metrics["innerWidth"] + 2, metrics
        saved.append(shot(page, shots, "mobile-inventory"))

        open_query(page, "dress:0:dressref0:dresstgt0")
        page.locator("figure img").first.wait_for(timeout=NAV_TIMEOUT_MS)
        page.wait_for_function(
            "() => [...document.querySelectorAll('figure img')].every(i => i.complete)",
            timeout=NAV_TIMEOUT_MS)
        mobile_metrics = overflow(page)
        images = image_evidence(page)
        widest = max((i["width"] for i in images), default=0)
        recorder.note(f"mobile query: body.scrollWidth={mobile_metrics['scrollWidth']} "
                      f"innerWidth={mobile_metrics['innerWidth']}, widest img={widest}px, "
                      f"{sum(1 for i in images if i['naturalWidth'] > 0)}/{len(images)} loaded")
        assert mobile_metrics["scrollWidth"] <= mobile_metrics["innerWidth"] + 2, mobile_metrics
        assert widest <= mobile_metrics["innerWidth"] + 2, widest
        saved.append(shot(page, shots, "mobile-query"))
    finally:
        context.close()
    return saved


# --------------------------------------------------------------------------- main

def run_sessions(url: str, shots: Path, which: str) -> Recorder:
    from playwright.sync_api import sync_playwright

    recorder = Recorder()
    saved: list[str] = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport=VIEWPORT_DESKTOP)
        page = context.new_page()
        recorder.attach(page)
        try:
            if which in ("cold", "all"):
                saved += session_cold(page, url, shots, recorder)
            if which in ("fixture", "all"):
                saved += session_fixture(page, url, shots, recorder)
            if which in ("mobile", "all"):
                saved += session_mobile(browser, url, shots, recorder)
        finally:
            context.close()
            browser.close()
    recorder.note(f"screenshots saved: {saved}")
    return recorder


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-fixture", type=Path, default=None)
    parser.add_argument("--images-root", type=Path, default=None)
    parser.add_argument("--url", default=None)
    parser.add_argument("--shots", type=Path, default=Path("/tmp/ui-e2e/shots"))
    parser.add_argument("--session", choices=("cold", "fixture", "mobile", "all"), default="all")
    args = parser.parse_args(argv)

    if args.build_fixture is not None:
        published = build_fixture(args.build_fixture, args.images_root or args.build_fixture.parent / "fashioniq")
        print(f"[FIXTURE] published {published}")
        return 0

    if not args.url:
        parser.error("either --build-fixture or --url is required")
    recorder = run_sessions(args.url, args.shots, args.session)
    summary = {
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
