#!/usr/bin/env python3
"""Real-browser end-to-end + performance test harness (4 viewports: 1440x900, 1280x800, 390x844, 360x800).

Requires running backend and frontend servers (or manages them via ServerManager).
Env vars:
  WORKBENCH_CHROME_PATH: optional explicit path to Chrome executable.
  WORKBENCH_E2E_BASE_URL: optional base URL when pointing at existing servers.
  WORKBENCH_ARTIFACTS_ROOT: isolated scratch artifacts directory (NEVER touch real research assets).

Uses system Google Chrome with zero browser downloads.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from workbench.frontend.tests.ui_smoke import (
    build_fixture, Recorder, goto, open_page, select_protocol,
    wait_for_runs_loaded, wait_for_pager_ready, open_query,
    set_active_run, select_active_run, visible_query_id, wait_held,
    image_evidence, shot, overflow, model_detail_geometry, api_json,
    PROTOCOL, STRADDLE_QUERY, NAV_TIMEOUT_MS
)
from playwright.sync_api import sync_playwright

VIEWPORTS = [
    {"name": "1440x900", "width": 1440, "height": 900, "is_mobile": False},
    {"name": "1280x800", "width": 1280, "height": 800, "is_mobile": False},
    {"name": "390x844", "width": 390, "height": 844, "is_mobile": True},
    {"name": "360x800", "width": 360, "height": 800, "is_mobile": True},
]

CHROME_BIN = os.environ.get("WORKBENCH_CHROME_PATH", "/usr/bin/google-chrome")
DEFAULT_TMPDIR = Path("/dev/shm/a40-e2e")

def port_is_listening(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        try:
            s.connect((host, port))
            return True
        except (OSError, TimeoutError):
            return False

def wait_for_port(port: int, timeout: float = 30.0) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout:
        if port_is_listening(port):
            return (time.perf_counter() - t0) * 1000.0
        time.sleep(0.05)
    raise TimeoutError(f"Port {port} not ready within {timeout}s")

class ServerManager:
    def __init__(self, tmp_dir: Path, backend_port: int = 8765, frontend_port: int = 5765):
        self.tmp_dir = tmp_dir
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.backend_proc: subprocess.Popen | None = None
        self.frontend_proc: subprocess.Popen | None = None

    def start(self) -> tuple[float, float]:
        env = os.environ.copy()
        env["WORKBENCH_ARTIFACTS_ROOT"] = str(self.tmp_dir / "artifacts")
        env["WORKBENCH_RESULTS_ROOT"] = str(self.tmp_dir / "artifacts" / "results")
        env["WORKBENCH_CHECKPOINT_ROOT"] = str(self.tmp_dir / "artifacts" / "checkpoints")
        env["FASHIONIQ_ROOT"] = str(self.tmp_dir / "fashioniq")
        env["WORKBENCH_DEVELOPMENT_MODE"] = "1"
        env["TMPDIR"] = str(self.tmp_dir)

        self.backend_proc = subprocess.Popen(
            [str(REPO / ".venv" / "bin" / "python"), "-m", "uvicorn",
             "workbench.backend.main:app", "--host", "127.0.0.1", "--port", str(self.backend_port)],
            cwd=str(REPO), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True
        )
        b_time = wait_for_port(self.backend_port)

        fe_env = os.environ.copy()
        fe_env["WORKBENCH_BACKEND_URL"] = f"http://127.0.0.1:{self.backend_port}"
        self.frontend_proc = subprocess.Popen(
            ["npm", "--prefix", str(REPO / "workbench" / "frontend"), "run", "dev",
             "--", "--host", "127.0.0.1", "--port", str(self.frontend_port), "--strictPort"],
            cwd=str(REPO), env=fe_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            start_new_session=True
        )
        f_time = wait_for_port(self.frontend_port)
        return b_time, f_time

    def stop(self) -> None:
        for proc in (self.frontend_proc, self.backend_proc):
            if proc is not None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except OSError:
                    pass
        deadline = time.monotonic() + 4.0
        for proc in (self.frontend_proc, self.backend_proc):
            if proc is not None:
                try:
                    proc.wait(timeout=max(0.1, deadline - time.monotonic()))
                except (subprocess.TimeoutExpired, OSError):
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except OSError:
                        pass
                    proc.wait()
        self.backend_proc = None
        self.frontend_proc = None

def run_journeys_for_viewport(
    page, base_url: str, vp: dict[str, Any], shots_dir: Path, recorder: Recorder
) -> dict[str, Any]:
    vp_name = vp["name"]
    vp_shots = shots_dir / vp_name
    vp_shots.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"viewport": vp_name, "journeys": {}, "screenshots": {}}

    # Journey A: Initial Load
    t0 = time.perf_counter()
    goto(page, base_url, recorder)
    page.wait_for_function(
        "() => /^17 registered models$/.test(document.querySelector('div.cards b')?.textContent?.trim() || '')",
        timeout=NAV_TIMEOUT_MS
    )
    # Ensure runs index loaded and protocol is set for subsequent flows
    wait_for_runs_loaded(page, 2)
    select_protocol(page, PROTOCOL)
    j_a_time = (time.perf_counter() - t0) * 1000.0
    shot_a = shot(page, vp_shots, "A_initial_load")
    report["journeys"]["A"] = {"status": "PASS", "timing_ms": round(j_a_time, 2)}
    report["screenshots"]["A"] = shot_a

    # Journey B: Model Inventory
    t0 = time.perf_counter()
    open_page(page, "Models")
    models = api_json(base_url, "/api/models")
    rows = page.locator("section table tbody tr")
    rows.first.wait_for(timeout=NAV_TIMEOUT_MS)
    row_count = rows.count()
    assert row_count == 17, f"expected 17 model rows, got {row_count}"
    # Verify blocker info truthful
    dom_text = [rows.nth(i).inner_text() for i in range(row_count)]
    shown_blockers = sum(1 for text in dom_text if "Blockers:" in text)
    assert shown_blockers > 0, "expected blockers to be shown in DOM"
    # Verify disabled actions stay disabled (no false readiness)
    queue_btns = page.locator("section table tbody tr button", has_text="Queue official evaluation")
    for i in range(queue_btns.count()):
        assert queue_btns.nth(i).is_disabled(), f"action button {i} should be disabled"
    j_b_time = (time.perf_counter() - t0) * 1000.0
    shot_b = shot(page, vp_shots, "B_model_inventory")
    report["journeys"]["B"] = {"status": "PASS", "timing_ms": round(j_b_time, 2)}
    report["screenshots"]["B"] = shot_b

    # Journey C: Query + Top-K
    t0 = time.perf_counter()
    query_id = "dress:0:dressref0:dresstgt0"
    open_query(page, query_id)
    page.locator("figure img").first.wait_for(timeout=NAV_TIMEOUT_MS)
    page.wait_for_function("() => [...document.querySelectorAll('figure img')].every(i => i.complete)", timeout=NAV_TIMEOUT_MS)
    images = image_evidence(page)
    assert len(images) == 12, f"expected 12 image elements, got {len(images)}"
    ref = [i for i in images if i["alt"] == "Reference: dressref0"]
    tgt = [i for i in images if i["alt"] == "Target: dresstgt0"]
    assert ref and ref[0]["naturalWidth"] > 0, "reference image not loaded"
    assert tgt and tgt[0]["naturalWidth"] > 0, "target image not loaded"
    topk = [i for i in images if (i["alt"] or "").startswith("#")]
    assert len(topk) == 10, f"expected 10 Top-K images, got {len(topk)}"
    assert all(i["naturalWidth"] > 0 for i in topk), "Top-K image naturalWidth check failed"
    # Test pagination
    pager_before = wait_for_pager_ready(page)
    page.locator("div.pager button", has_text="Next page").click()
    page.wait_for_function("() => document.querySelector('div.pager span').textContent.trim().startsWith('21')", timeout=NAV_TIMEOUT_MS)
    pager_after = page.locator("div.pager span").first.inner_text()
    assert pager_before != pager_after, "pagination did not advance"
    # Return to page 1
    page.locator("div.pager button", has_text="Previous page").click()
    wait_for_pager_ready(page)
    j_c_time = (time.perf_counter() - t0) * 1000.0
    shot_c = shot(page, vp_shots, "C_query_topk")
    report["journeys"]["C"] = {"status": "PASS", "timing_ms": round(j_c_time, 2)}
    report["screenshots"]["C"] = shot_c

    # Journey D: Model Comparison
    t0 = time.perf_counter()
    open_query(page, STRADDLE_QUERY)
    open_page(page, "Compare Models")
    articles = page.locator("section > article")
    articles.first.wait_for(timeout=NAV_TIMEOUT_MS)
    page.wait_for_function("() => document.querySelectorAll('section > article').length >= 2", timeout=NAV_TIMEOUT_MS)
    ranks = page.evaluate("() => [...document.querySelectorAll('section > article')].map(a => { const m = a.textContent.match(/target rank (\\d+)/); return m ? Number(m[1]) : null; })")
    assert ranks == [4, 45], f"expected target ranks [4, 45], got {ranks}"
    j_d_time = (time.perf_counter() - t0) * 1000.0
    shot_d = shot(page, vp_shots, "D_model_comparison")
    report["journeys"]["D"] = {"status": "PASS", "timing_ms": round(j_d_time, 2)}
    report["screenshots"]["D"] = shot_d

    # Journey E: Failure Analysis
    t0 = time.perf_counter()
    open_page(page, "Common Failures")
    page.locator("div.filters select").first.select_option("all")
    page.locator("table tbody tr").first.wait_for(timeout=NAV_TIMEOUT_MS)
    rows_shown = page.locator("table tbody tr").count()
    assert rows_shown == 24, f"expected 24 failure rows, got {rows_shown}"
    cohort_id = f"cohort-{vp_name}"
    page.get_by_placeholder("cohort ID").fill(cohort_id)
    page.locator("section button", has_text="Save filtered set").click()
    stored_cohorts = []
    for _ in range(50):
        stored_cohorts = api_json(base_url, "/api/cohorts")
        if any(c["cohort_id"] == cohort_id for c in stored_cohorts):
            break
        time.sleep(0.1)
    saved_cohort = next((c for c in stored_cohorts if c["cohort_id"] == cohort_id), None)
    assert saved_cohort is not None, f"cohort {cohort_id} not found in API"
    assert len(saved_cohort["query_ids"]) == rows_shown, "cohort size mismatch"
    j_e_time = (time.perf_counter() - t0) * 1000.0
    shot_e = shot(page, vp_shots, "E_failure_analysis")
    report["journeys"]["E"] = {"status": "PASS", "timing_ms": round(j_e_time, 2)}
    report["screenshots"]["E"] = shot_e

    # Journey F: Annotation
    t0 = time.perf_counter()
    open_query(page, query_id)
    open_page(page, "Annotations")
    page.locator("div.labels label").first.wait_for(timeout=NAV_TIMEOUT_MS)
    label_el = page.locator("div.labels label").first
    label_text = label_el.inner_text().strip()
    label_el.locator("input").check()
    test_note = f"annotation-{vp_name}"
    page.locator("textarea").fill(test_note)
    page.locator("section button", has_text="Save / update").click()
    stored_ann = []
    for _ in range(50):
        stored_ann = api_json(base_url, f"/api/annotations?query_id={query_id}&protocol_id={PROTOCOL}")
        if stored_ann:
            break
        time.sleep(0.1)
    assert stored_ann and label_text in stored_ann[0]["labels"], "annotation not saved in API"
    # Verify reload persistence
    page.reload(wait_until="domcontentloaded")
    wait_for_runs_loaded(page, 2)
    select_protocol(page, PROTOCOL)
    open_query(page, query_id)
    open_page(page, "Annotations")
    page.locator("div.labels label").first.wait_for(timeout=NAV_TIMEOUT_MS)
    is_checked = page.locator("div.labels label", has_text=label_text).first.locator("input").is_checked()
    saved_note = page.locator("textarea").input_value()
    assert is_checked, "annotation checkbox did not persist after reload"
    assert saved_note == test_note, f"expected note {test_note!r}, got {saved_note!r}"
    j_f_time = (time.perf_counter() - t0) * 1000.0
    shot_f = shot(page, vp_shots, "F_annotation")
    report["journeys"]["F"] = {"status": "PASS", "timing_ms": round(j_f_time, 2)}
    report["screenshots"]["F"] = shot_f

    # Journey G: Error Recovery
    t0 = time.perf_counter()
    open_page(page, "Sample Explorer")
    wait_for_pager_ready(page)
    def abort_queries(route):
        recorder.injected.append(f"INJECTED abort {route.request.url}")
        route.abort()
    recorder.injecting = True
    page.route("**/api/queries*", abort_queries)
    page.locator("div.pager button", has_text="Next page").click()
    page.locator("aside.error").wait_for(timeout=30000)
    recorder.injecting = False
    err_text = page.locator("aside.error").inner_text()
    assert "Guard/API error" in err_text, f"unexpected error text: {err_text!r}"
    page.unroute("**/api/queries*", abort_queries)
    page.locator("aside.error button", has_text="Dismiss").click()
    page.locator("aside.error").wait_for(state="detached", timeout=NAV_TIMEOUT_MS)
    page.locator("div.pager button", has_text="Previous page").click()
    page.wait_for_function("() => document.querySelector('div.pager span').textContent.trim().startsWith('1')", timeout=NAV_TIMEOUT_MS)
    assert page.locator("aside.error").count() == 0, "error aside still present after recovery"
    j_g_time = (time.perf_counter() - t0) * 1000.0
    shot_g = shot(page, vp_shots, "G_error_recovery")
    report["journeys"]["G"] = {"status": "PASS", "timing_ms": round(j_g_time, 2)}
    report["screenshots"]["G"] = shot_g

    # Journey H: Mobile Navigation / Responsiveness
    t0 = time.perf_counter()
    open_page(page, "Models")
    page.wait_for_function(
        "() => document.querySelectorAll('section > table > tbody > tr').length === 17",
        timeout=NAV_TIMEOUT_MS
    )
    metrics = overflow(page)
    controls = page.locator("nav button").count()
    assert controls >= 5, f"expected at least 5 nav buttons, got {controls}"
    reachable = page.evaluate("""() => [...document.querySelectorAll('nav button, .controls select')]
        .every(el => { const r = el.getBoundingClientRect();
                       return r.width > 0 && r.height > 0 && r.left >= -1 &&
                              r.right <= window.innerWidth + 2; })""")
    assert reachable, "nav control element is off-viewport"
    assert metrics["scrollWidth"] <= metrics["innerWidth"] + 2, f"body overflow: {metrics}"
    detail = model_detail_geometry(page)
    assert len(detail) == 17, f"expected 17 detail cells, measured {len(detail)}"
    for idx, cell in enumerate(detail):
        assert cell["left"] >= 0 and cell["right"] <= metrics["innerWidth"] + 2, f"cell {idx} off-viewport: {cell}"
        assert cell["scrollHeight"] > 0 and cell["width"] > 35, f"cell {idx} collapsed: {cell}"
    # Verify query view bounded
    open_query(page, query_id)
    page.locator("figure img").first.wait_for(timeout=NAV_TIMEOUT_MS)
    q_metrics = overflow(page)
    q_imgs = image_evidence(page)
    widest = max((i["width"] for i in q_imgs), default=0)
    assert q_metrics["scrollWidth"] <= q_metrics["innerWidth"] + 2, q_metrics
    assert widest <= q_metrics["innerWidth"] + 2, f"image too wide: {widest} > {q_metrics['innerWidth']}"
    j_h_time = (time.perf_counter() - t0) * 1000.0
    shot_h = shot(page, vp_shots, "H_mobile_nav")
    report["journeys"]["H"] = {"status": "PASS", "timing_ms": round(j_h_time, 2)}
    report["screenshots"]["H"] = shot_h

    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tmpdir", type=Path, default=DEFAULT_TMPDIR)
    parser.add_argument("--backend-port", type=int, default=8765)
    parser.add_argument("--frontend-port", type=int, default=5765)
    args = parser.parse_args(argv)

    tmp_dir = args.tmpdir
    shots_dir = tmp_dir / "shots"
    shots_dir.mkdir(parents=True, exist_ok=True)
    results_dir = tmp_dir / "artifacts" / "results"
    images_dir = tmp_dir / "fashioniq"
    if not (results_dir / "fashioniq_val_split" / "fixture-a").is_dir():
        build_fixture(results_dir, images_dir)

    sm = ServerManager(tmp_dir, args.backend_port, args.frontend_port)
    b_ms, f_ms = sm.start()
    base_url = f"http://127.0.0.1:{args.frontend_port}"
    print(f"[OK] Backend ready in {b_ms:.1f}ms, Frontend ready in {f_ms:.1f}ms")

    full_results: dict[str, Any] = {
        "server_timing": {"backend_startup_ms": b_ms, "frontend_startup_ms": f_ms},
        "viewports": {},
    }

    try:
        with sync_playwright() as playwright:
            launch_kwargs: dict[str, Any] = {
                "args": ["--no-sandbox", "--disable-gpu", "--disable-dev-shm-usage"],
            }
            if os.environ.get("WORKBENCH_CHROME_PATH"):
                launch_kwargs["executable_path"] = os.environ["WORKBENCH_CHROME_PATH"]
            elif Path(CHROME_BIN).exists():
                launch_kwargs["executable_path"] = CHROME_BIN
            else:
                launch_kwargs["channel"] = "chrome"
            browser = playwright.chromium.launch(**launch_kwargs)
            for vp in VIEWPORTS:
                vp_name = vp["name"]
                print(f"\n--- Testing Viewport {vp_name} ({vp['width']}x{vp['height']}) ---")
                rec = Recorder()
                ctx = browser.new_context(
                    viewport={"width": vp["width"], "height": vp["height"]},
                    is_mobile=vp["is_mobile"],
                    has_touch=vp["is_mobile"]
                )
                page = ctx.new_page()
                rec.attach(page)

                try:
                    vp_report = run_journeys_for_viewport(page, base_url, vp, shots_dir, rec)
                    vp_report["recorder"] = {
                        "console_errors_unexpected": rec.console,
                        "console_errors_injected": rec.console_injected,
                        "page_errors": rec.page_errors,
                        "failed_requests_unexpected": rec.failed_requests,
                        "failed_requests_injected": rec.failed_requests_injected,
                    }
                    full_results["viewports"][vp_name] = vp_report
                    print(f"[PASS] Viewport {vp_name}: all 8 journeys PASSED")
                except Exception as ex:
                    print(f"[FAIL] Viewport {vp_name} error: {ex}", file=sys.stderr)
                    full_results["viewports"][vp_name] = {"status": "FAIL", "error": str(ex)}
                    raise
                finally:
                    ctx.close()
            browser.close()
    finally:
        sm.stop()
        print("[OK] Servers stopped cleanly.")

    out_json = tmp_dir / "g2_e2e_results.json"
    out_json.write_text(json.dumps(full_results, indent=2))
    print(f"\nResults saved to {out_json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
