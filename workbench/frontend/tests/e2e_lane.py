#!/usr/bin/env python3
"""Real-browser end-to-end lane: the ui_smoke journeys, re-run at 4 viewports.

Ports: this lane never borrows the live stacks. It starts its own backend on 8775 and its own
frontend on 5775 (was 8765/5765, which collided with the running stacks), or reuses an already
running workbench when `WORKBENCH_E2E_BASE_URL` names one.

Env vars:
  WORKBENCH_E2E_BASE_URL: base URL of an existing workbench; skips the server start entirely.
  WORKBENCH_CHROME_PATH:  optional explicit path to Chrome.
  WORKBENCH_ARTIFACTS_ROOT: isolated scratch artifacts directory (NEVER touch real research assets).

Journeys kept, all re-derived from the repaired harness:
  A  overview: the registry card's entry count equals len(/api/models) (a digit, not a placeholder).
  B  experiments/models: one row per API model, and a blockers disclosure reading "<n> lý do" for
     every model whose checkpoints declare blocking reasons.
  H  containment: body.scrollWidth <= innerWidth + 2, every area-nav button inside the viewport
     once the drawer is open, and every model row's 5th (blocker) cell inside its own scroll box.

Journeys removed by the 4e61f2f page split:
  C  query + Top-K drove `div.pager`/"Next page" and `figure img` alts — `open_query` no longer
     exists, and the explorer needs a selected CIRCUS composition run, which this lane does not serve.
  D  two-run comparison drove `section > article` per-run columns and a "target rank" string; the
     compare view is now the inspector's own ComparePanel, same blocker (needs a composition run).
  E  failure analysis / "Save filtered set" drove `div.filters` + a cohort POST that the fixture
     stack cannot satisfy (`/api/queries?limit=20000` answers 400 invalid_pagination, so the page
     shows an error and the save control never mounts).
  F  annotation save/reload drove `div.labels label` + "Save / update"; the notebook is reachable
     only through an inspector query, i.e. through a composition run this lane does not serve.
  G  error recovery drove `aside.error`/Dismiss; the current UI reports failures with
     `[data-testid=empty-state][data-empty-kind=error]` + `[data-testid=error-retry]`, which the
     `ui_smoke.py --session fixture` session covers on the registry card.

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
import struct
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from workbench.frontend.tests.ui_smoke import (
    build_fixture, Recorder, api_json, launch, open_page, overflow, registry_stat,
    set_protocol, shot, PROTOCOL, NAV_TIMEOUT_MS,
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
    def __init__(self, tmp_dir: Path, backend_port: int = 8775, frontend_port: int = 5775):
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
    models = api_json(base_url, "/api/models")

    # Journey A: initial load -- the registry card is the UI's own count of /api/models.
    t0 = time.perf_counter()
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector("[data-testid=area-overview]", timeout=NAV_TIMEOUT_MS)
    page.wait_for_function(
        """() => [...document.querySelectorAll('[data-testid=overview-cards] .stat')]
                 .some(s => (s.querySelector('.stat-label')?.textContent || '').trim() === 'entry'
                            && /^[0-9][0-9,]*$/.test((s.querySelector('.stat-value')?.textContent || '').trim()))""",
        timeout=NAV_TIMEOUT_MS)
    stats = registry_stat(page)
    rendered = int(stats["entry"].replace(",", ""))
    assert rendered == len(models), f"overview count {rendered} != /api/models {len(models)}"
    set_protocol(page, PROTOCOL)
    report["journeys"]["A"] = {"status": "PASS",
                               "detail": f"registry entry={rendered} == /api/models {len(models)}",
                               "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
    report["screenshots"]["A"] = shot(page, vp_shots, "A_initial_load")

    # Journey B: model inventory -- one row per API entry, blockers truthful.
    t0 = time.perf_counter()
    open_page(page, "experiments", "models")
    page.wait_for_function(
        "n => document.querySelectorAll('[data-testid^=model-row-]').length === n",
        arg=len(models), timeout=NAV_TIMEOUT_MS)
    rows = page.locator("[data-testid^=model-row-]").count()
    assert rows == len(models), f"expected {len(models)} model rows, got {rows}"
    blocked = {}
    for model in models:
        reasons = [reason for checkpoint in model.get("checkpoints") or []
                   for reason in checkpoint.get("blocking_reasons") or []]
        if reasons:
            blocked[model["model_id"]] = reasons
    disclosures = page.evaluate("""() => [...document.querySelectorAll('[data-testid^=model-row-]')]
        .map(row => { const d = row.querySelector('details > summary');
                      return d ? d.textContent.trim() : null; }).filter(Boolean)""")
    if blocked:
        assert disclosures, "the API declares blocking reasons but no row shows a disclosure"
        import re
        for text in disclosures:
            assert re.fullmatch(r"\d+\s+lý do", text), f"blocker summary {text!r} is not '<n> lý do'"
    report["journeys"]["B"] = {
        "status": "PASS",
        "detail": f"{rows} rows, {len(blocked)} API models blocked, {len(disclosures)} disclosures",
        "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
    report["screenshots"]["B"] = shot(page, vp_shots, "B_model_inventory")

    # Journey H: containment at this viewport.
    t0 = time.perf_counter()
    metrics = overflow(page)
    assert metrics["scrollWidth"] <= metrics["innerWidth"] + 2, f"body overflow: {metrics}"
    cells = page.evaluate("""() => [...document.querySelectorAll('[data-testid^=model-row-]')]
        .map(row => { const td = row.children[4]; const wrap = row.closest('.table-wrap');
            const r = td.getBoundingClientRect(); const w = wrap.getBoundingClientRect();
            return {cells: row.children.length, left: Math.round(r.left), right: Math.round(r.right),
                    width: Math.round(r.width), textLength: td.textContent.trim().length,
                    wrapLeft: Math.round(w.left), wrapScroll: wrap.scrollWidth}; })""")
    assert len(cells) == len(models), f"expected {len(models)} blocker cells, measured {len(cells)}"
    for index, cell in enumerate(cells):
        assert cell["cells"] == 5, f"model row {index}: {cell['cells']} cells, expected 5"
        assert cell["wrapLeft"] <= cell["left"] and \
            cell["right"] - cell["wrapLeft"] <= cell["wrapScroll"] + 1, (
            f"model row {index}: blocker cell outside its scroll box: {cell}")
        assert cell["width"] > 40 and cell["textLength"] > 0, (
            f"model row {index}: blocker cell collapsed or empty: {cell}")

    # Nav reachability: on a narrow viewport the sidebar is a drawer, so open it first.
    nav = page.evaluate("""() => {
        const el = document.querySelector('[data-testid=workbench-nav]');
        const rect = el.getBoundingClientRect();
        return {open: el.dataset.open === 'true', offscreen: rect.right <= 1 || rect.left < -1};
    }""")
    if nav["offscreen"] and not nav["open"]:
        page.locator(".menu-toggle").first.click()
        page.wait_for_function(
            "() => document.querySelector('[data-testid=workbench-nav]')?.dataset.open === 'true'",
            timeout=NAV_TIMEOUT_MS)
        page.wait_for_timeout(250)
    items = page.evaluate("""() => [...document.querySelectorAll('[data-testid^=area-nav-]')]
        .map(el => { const r = el.getBoundingClientRect();
                     return {id: el.dataset.testid, left: Math.round(r.left),
                             right: Math.round(r.right), w: Math.round(r.width),
                             h: Math.round(r.height)}; })""")
    inner = page.evaluate("() => window.innerWidth")
    off = [item for item in items if item["left"] < -1 or item["right"] > inner + 2
           or item["w"] <= 0 or item["h"] <= 0]
    assert len(items) >= 6, items
    assert not off, f"nav buttons off-viewport: {off}"

    viewport_metrics = page.evaluate("""() => ({
        document_client_width: document.documentElement.clientWidth,
        window_inner_width: window.innerWidth,
        device_pixel_ratio: window.devicePixelRatio,
        body_scroll_width: document.body.scrollWidth})""")
    shot_h = shot(page, vp_shots, "H_containment")
    screenshot_pixel_width, screenshot_pixel_height = struct.unpack(">II", Path(shot_h).read_bytes()[16:24])
    viewport_metrics["screenshot_pixel_width"] = screenshot_pixel_width
    viewport_metrics["screenshot_pixel_height"] = screenshot_pixel_height
    assert viewport_metrics["window_inner_width"] == vp["width"], viewport_metrics
    assert viewport_metrics["body_scroll_width"] <= vp["width"] + 2, viewport_metrics
    assert abs(screenshot_pixel_width - vp["width"]) <= 2, viewport_metrics
    report["viewport_metrics"] = viewport_metrics
    report["journeys"]["H"] = {
        "status": "PASS",
        "detail": f"{len(cells)} blocker cells inside their scroll boxes, {len(items)} nav buttons "
                  f"inside {inner}px, body {viewport_metrics['body_scroll_width']}px",
        "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
    report["screenshots"]["H"] = shot_h
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tmpdir", type=Path, default=DEFAULT_TMPDIR)
    parser.add_argument("--backend-port", type=int, default=8775)
    parser.add_argument("--frontend-port", type=int, default=5775)
    args = parser.parse_args(argv)

    existing = os.environ.get("WORKBENCH_E2E_BASE_URL", "").rstrip("/")
    tmp_dir = args.tmpdir
    shots_dir = tmp_dir / "shots2"
    shots_dir.mkdir(parents=True, exist_ok=True)
    results_dir = tmp_dir / "artifacts" / "results"
    images_dir = tmp_dir / "fashioniq"
    if not existing and not (results_dir / "fashioniq_val_split" / "fixture-a").is_dir():
        # A self-started backend serves an empty scratch root; without the fixture the registry
        # count is still assertable but the run catalog would be empty, so publish it first.
        build_fixture(results_dir, images_dir)

    sm = None
    server_timing: dict[str, Any] = {}
    if existing:
        base_url = existing
        print(f"[OK] Reusing the running workbench at {base_url}")
    else:
        sm = ServerManager(tmp_dir, args.backend_port, args.frontend_port)
        b_ms, f_ms = sm.start()
        base_url = f"http://127.0.0.1:{args.frontend_port}"
        server_timing = {"backend_startup_ms": b_ms, "frontend_startup_ms": f_ms}
        print(f"[OK] Backend ready in {b_ms:.1f}ms, Frontend ready in {f_ms:.1f}ms at {base_url}")

    full_results: dict[str, Any] = {"server_timing": server_timing, "base_url": base_url,
                                    "viewports": {}}
    failures = 0
    try:
        with sync_playwright() as playwright:
            browser = launch(playwright)
            try:
                for vp in VIEWPORTS:
                    vp_name = vp["name"]
                    print(f"\n--- Viewport {vp_name} ({vp['width']}x{vp['height']}) ---")
                    rec = Recorder()
                    ctx = browser.new_context(
                        viewport={"width": vp["width"], "height": vp["height"]},
                        is_mobile=vp["is_mobile"], has_touch=vp["is_mobile"])
                    page = ctx.new_page()
                    rec.attach(page)
                    try:
                        vp_report = run_journeys_for_viewport(page, base_url, vp, shots_dir, rec)
                        vp_report["recorder"] = {
                            "console_errors_unexpected": rec.console,
                            "page_errors": rec.page_errors,
                            "failed_requests_unexpected": rec.failed_requests,
                        }
                        assert not rec.console and not rec.page_errors and not rec.failed_requests, \
                            vp_report["recorder"]
                        full_results["viewports"][vp_name] = vp_report
                        print(f"[PASS] Viewport {vp_name}: journeys "
                              f"{sorted(vp_report['journeys'])} on {len(api_json(base_url, '/api/models'))} models")
                    except Exception as ex:
                        failures += 1
                        print(f"[FAIL] Viewport {vp_name}: {ex}", file=sys.stderr)
                        full_results["viewports"][vp_name] = {"status": "FAIL", "error": str(ex)}
                    finally:
                        ctx.close()
            finally:
                browser.close()
    finally:
        if sm is not None:
            sm.stop()
            print("[OK] Servers stopped cleanly.")

    out_json = tmp_dir / "g2_e2e_results.json"
    out_json.write_text(json.dumps(full_results, indent=2))
    print(f"\nResults saved to {out_json}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
