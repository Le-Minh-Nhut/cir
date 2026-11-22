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
  A  overview: rendered published and registry counts match their API populations; deferred IDs are absent.
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
import shutil
import csv
import io
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from workbench.frontend.tests.ui_smoke import (
    build_fixture, Recorder, api_json, launch, open_page, overflow, registry_stat,
    assert_registry_counts, set_protocol, shot, PROTOCOL, NAV_TIMEOUT_MS,
)
from playwright.sync_api import sync_playwright
PARITY_FIXTURE = {
    "standard": "parity-absolute-relative/v3",
    "standards": {"parity-relative/v1": ["fixture-incomplete"], "parity-absolute-relative/v3": ["fixture-failed"]},
    "passed_models": [], "note": "Synthetic UI fixture only; no model evidence.",
    "models": {
        "fixture-incomplete": {
            "model_id": "fixture-incomplete", "status": "PARITY_INCOMPLETE", "passed": False,
            "complete": False, "applied_standard": "parity-relative/v1", "requested_standard": "parity-absolute-relative/v3",
            "dimensions": {"forward_parity": False, "batch_parity": True},
            "categories": ["dress", "shirt"], "missing": ["forward_parity", "batch_parity"],
            "missing_by_category": {"dress": ["forward_parity", "batch_parity"]}, "mandatory_by_category": {"dress": []},
            "failure_count": 0, "failures": [], "checks": [
                {"name": "query control", "dimension": "forward_parity", "check_type": "forward_parity", "gate": "G1", "category": "dress", "declared": {"coverage": "full", "population_count": 100}, "measured": {"control": "control-a", "coverage_scope": "sample", "sample_count": 10, "population_count": 100, "ordered_ids_sha256": "a" * 64, "tensor_shape": [10, 32, 768], "dtype": "float32", "batch_policy": {"partition_id": "part-a", "batch_sizes": [1, 8]}}, "ok": False, "mandatory": True, "standard": "parity-relative/v1", "note": "fixture"},
                {"name": "gallery control", "dimension": "batch_parity", "check_type": "batch_parity", "gate": "G2", "category": "dress", "declared": {"coverage": "full", "population_count": 100}, "measured": {"control": "control-a", "coverage_scope": "full", "population_count": 100, "ordered_ids_sha256": "b" * 64, "tensor_shape": [100, 768], "dtype": "float32", "batch_policy": None}, "ok": True, "mandatory": True, "standard": "parity-relative/v1", "note": "fixture"},
            ],
        },
        "fixture-failed": {"model_id": "fixture-failed", "status": "PARITY_MANDATORY_SUBCHECK_FAILED", "passed": False, "complete": True,
            "applied_standard": "parity-absolute-relative/v3", "dimensions": {"input_parity": False}, "categories": [], "missing": [],
            "failure_count": 1, "failures": [{"name": "measured fail", "dimension": "input_parity", "check_type": "input_parity", "gate": "G1", "declared": {}, "measured": {}, "ok": False, "mandatory": True, "standard": "parity-absolute-relative/v3", "note": "fixture"}], "checks": []},
        "fixture-malformed": {"model_id": "fixture-malformed", "status": "PARITY_EVIDENCE_MALFORMED", "passed": False, "complete": False,
            "applied_standard": "parity-relative/v1", "dimensions": {}, "categories": [], "missing": [], "failure_count": 0, "failures": [], "checks": []},
    },
}

def run_parity_fixture(page, base_url: str, recorder: Recorder) -> None:
    page.route("**/api/runs/parity-gate**", lambda route: route.fulfill(status=200, json=PARITY_FIXTURE))
    page.goto(f"{base_url}/?area=evaluation&view=composition", wait_until="domcontentloaded")
    page.wait_for_selector('[data-testid="parity-gate-table"]', timeout=NAV_TIMEOUT_MS)
    page.locator('[data-testid="parity-row-fixture-incomplete"]').wait_for()
    row = page.locator('[data-testid="parity-row-fixture-incomplete"]')
    if page.locator('[data-testid="parity-gate-summary"]').get_attribute("data-missing-count") != "1":
        raise AssertionError("incomplete model was counted as mandatory failure or omitted")
    if page.locator('[data-testid="parity-gate-summary"]').get_attribute("data-failed-count") != "1":
        raise AssertionError("mandatory failure count does not match fixture")
    if row.get_attribute("data-passed") != "false":
        raise AssertionError("incomplete model incorrectly counted as passed")
    for dim in ("forward_parity", "batch_parity"):
        state = row.locator(f'[data-testid="parity-dimension-fixture-incomplete-{dim}"]')
        if state.get_attribute("data-state") != "uncertain" or "?" not in state.inner_text():
            raise AssertionError(f"missing {dim} dimension not marked uncertain")
    if row.locator('[data-testid="parity-category-fixture-incomplete-shirt"]').get_attribute("data-state") != "unknown":
        raise AssertionError("unrecorded shirt coverage was treated as measured")
    if row.locator('[data-testid="parity-category-fixture-incomplete-dress"]').get_attribute("data-state") != "missing":
        raise AssertionError("missing dress dimension not reflected")
    if row.locator('[data-testid="parity-applied-standard-fixture-incomplete"]').count() != 1:
        raise AssertionError("applied standard is not visible")
    if row.locator('[data-testid="parity-requested-standard-fixture-incomplete"]').count() != 1:
        raise AssertionError("requested standard is not visible")
    scopes = row.locator('[data-testid="parity-scope-fixture-incomplete"]')
    if scopes.count() != 2:
        raise AssertionError("query and gallery check scopes were aggregated")
    query = row.locator('[data-testid="parity-scope-fixture-incomplete"][data-check-type="forward_parity"]')
    if query.count() != 1:
        raise AssertionError("query scope is missing or duplicated")
    query = query.first
    if query.get_attribute("data-declared-scope") != "full" or query.get_attribute("data-declared-population") != "100":
        raise AssertionError("declared query scope or denominator is wrong")
    if query.get_attribute("data-measured-scope") != "sample" or query.get_attribute("data-sample-count") != "10" or query.get_attribute("data-measured-population") != "100":
        raise AssertionError("measured query sample denominator is wrong")
    gallery = row.locator('[data-testid="parity-scope-fixture-incomplete"][data-check-type="batch_parity"]')
    if gallery.count() != 1:
        raise AssertionError("gallery scope is missing or duplicated")
    gallery = gallery.first
    if gallery.get_attribute("data-measured-scope") != "full" or gallery.get_attribute("data-measured-population") != "100":
        raise AssertionError("measured gallery scope or denominator is wrong")
    if page.evaluate("document.documentElement.scrollWidth <= window.innerWidth + 2") is not True:
        raise AssertionError("parity details overflow viewport")
    if recorder.page_errors:
        raise AssertionError(recorder.page_errors)


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

    def __init__(self, tmp_dir: Path, backend_port: int = 8775, frontend_port: int = 5775,
                 source_artifacts: Path | None = None, fashioniq_root: Path | None = None,
                 circus_release_root: Path | None = None):
        self.tmp_dir = tmp_dir
        self.backend_port = backend_port
        self.frontend_port = frontend_port
        self.source_artifacts = source_artifacts
        self.fashioniq_root = fashioniq_root
        self.circus_release_root = circus_release_root
        self.backend_proc: subprocess.Popen | None = None
        self.frontend_proc: subprocess.Popen | None = None
        self.log_handles = []

    def _log(self, name: str):
        path = self.tmp_dir / f"{name}.log"
        handle = path.open("w", encoding="utf-8")
        self.log_handles.append(handle)
        return handle
    def start(self) -> tuple[float, float]:
        env = os.environ.copy()
        env["WORKBENCH_ARTIFACTS_ROOT"] = str(self.tmp_dir / "artifacts")
        env["WORKBENCH_RESULTS_ROOT"] = str(self.tmp_dir / "artifacts" / "results")
        env["WORKBENCH_CHECKPOINT_ROOT"] = str(self.tmp_dir / "artifacts" / "checkpoints")
        env["FASHIONIQ_ROOT"] = str(self.fashioniq_root or (self.tmp_dir / "fashioniq"))
        if self.circus_release_root is not None:
            env["WORKBENCH_CIRCUS_RELEASE_ROOT"] = str(self.circus_release_root)
        # Keep source artifacts read-only: stage_imported_run copied the selected run into
        # `tmp_dir/artifacts`; backend writes (cohorts/index state) must stay in that scratch tree.
        # `source_artifacts` is only the input tree, never WORKBENCH_ARTIFACTS_ROOT.
        env["WORKBENCH_DEVELOPMENT_MODE"] = "1"
        env["TMPDIR"] = str(self.tmp_dir)
        env["PYTHONUNBUFFERED"] = "1"
        self.backend_proc = subprocess.Popen(
            [str(REPO / ".venv" / "bin" / "python"), "-u", "-m", "uvicorn",
             "workbench.backend.main:app", "--host", "127.0.0.1", "--port", str(self.backend_port)],
            cwd=str(REPO), env=env, stdout=self._log("backend.stdout.log"),
            stderr=self._log("backend.stderr.log"), start_new_session=True
        )
        try:
            b_time = wait_for_port(self.backend_port)
            fe_env = os.environ.copy()
            fe_env["WORKBENCH_BACKEND_URL"] = f"http://127.0.0.1:{self.backend_port}"
            fe_env["PYTHONUNBUFFERED"] = "1"
            self.frontend_proc = subprocess.Popen(
                ["npm", "--prefix", str(REPO / "workbench" / "frontend"), "run", "dev",
                 "--", "--host", "127.0.0.1", "--port", str(self.frontend_port), "--strictPort"],
                cwd=str(REPO), env=fe_env, stdout=self._log("frontend.stdout.log"),
                stderr=self._log("frontend.stderr.log"), start_new_session=True
            )
            f_time = wait_for_port(self.frontend_port)
            return b_time, f_time
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        processes = [proc for proc in (self.frontend_proc, self.backend_proc) if proc is not None]
        pgids = [proc.pid for proc in processes]
        for pgid in pgids:
            try:
                os.killpg(pgid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 4.0
        for proc in processes:
            try:
                proc.wait(timeout=max(0.1, deadline - time.monotonic()))
            except (subprocess.TimeoutExpired, OSError):
                pass
        for pgid in pgids:
            try:
                os.killpg(pgid, 0)
            except ProcessLookupError:
                continue
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for proc in processes:
            if proc.poll() is None:
                proc.wait()
        self.backend_proc = None
        self.frontend_proc = None
        for handle in self.log_handles:
            handle.close()
        self.log_handles.clear()

IMPORTED_RUN_ID = "released_audit_task7_mm_any"
IMPORTED_SUMMARY_DIR = "released_audit_task7_mm_any_released_audit_task7_mm_any"

def stage_imported_run(source: Path, isolated: Path) -> Path:
    """Expose only the existing imported run through a read-only isolated artifact tree."""
    summary_source = source / "composition" / "summary" / IMPORTED_SUMMARY_DIR
    run_source = source / "composition" / "runs" / IMPORTED_RUN_ID
    if not (summary_source / "full.json").is_file() or not run_source.is_dir():
        raise FileNotFoundError("the retained imported CIRCUS run artifacts are incomplete")
    composition = isolated / "composition"
    summary_dir = composition / "summary"
    runs_dir = composition / "runs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(summary_source, summary_dir / IMPORTED_SUMMARY_DIR, symlinks=False)
    shutil.copytree(run_source, runs_dir / IMPORTED_RUN_ID, symlinks=False)
    return isolated
def run_imported_query(page, base_url: str, shots: Path) -> dict[str, Any]:
    query_id = "7:4436"
    run_key = f"c:{IMPORTED_RUN_ID}"
    url = (f"{base_url}/?area=samples&view=explorer&protocol=circus_fashioniq_release_v1"
           f"&subset=full&runs={run_key}&imported=include&query={query_id}&condition=multimodal")
    page.goto(url, wait_until="domcontentloaded")
    page.locator('[data-testid="include-imported"]').wait_for(state="attached")
    if not page.locator('[data-testid="include-imported"]').is_checked():
        raise AssertionError("imported release scope was not retained in the URL")
    active_run = page.locator('[data-testid="active-run"]')
    if active_run.input_value() != run_key:
        active_run.select_option(run_key)
    page.locator('[data-testid="query-inspector"]').wait_for(timeout=NAV_TIMEOUT_MS)
    page.locator('[data-testid="query-title"]').wait_for()
    if page.locator('[data-testid="query-title"]').inner_text() != f"Query {query_id}":
        raise AssertionError("selected query identity changed")
    run = api_json(base_url, f"/api/composition/runs/{IMPORTED_RUN_ID}?subset=full")
    query = api_json(base_url, f"/api/composition/runs/{IMPORTED_RUN_ID}/queries?subset=full&condition=multimodal&query_id={query_id}&limit=1&top_k=20")
    item = query["items"][0]
    top_results = item["top_results"]
    if item["target_rank"] != 31987 or item["candidate_depth"] != 20 or len(top_results) != 20:
        raise AssertionError("selected query rank or exported candidate depth changed")
    if run["provenance_source"] != "released_audit_retriever" or run["reproduction_status"] != "RELEASED_ARTIFACT_IMPORT":
        raise AssertionError("run lost imported-artifact provenance")
    if run["protocol_id"] != "circus_fashioniq_release_v1" or run["subset"] != "full" or run["metrics"]["common_cohort_count"] != 6003:
        raise AssertionError("run protocol, subset, or common-cohort denominator changed")
    provenance = page.locator('[data-testid="candidates-provenance"]')
    provenance.wait_for()
    if not provenance.is_visible():
        raise AssertionError("imported provenance warning is not visible")
    ref = Path(item["reference_id"]).name
    positive = Path(item["positive_ids"][0]).name
    rank = page.locator('[data-testid="rank-multimodal"]')
    rank.wait_for()
    if rank.get_attribute("data-rank-state") != "miss" or rank.get_attribute("data-rank") != str(item["target_rank"]):
        raise AssertionError("query rank or full-gallery miss state differs from served record")
    if page.locator('[data-testid="query-caption"]').inner_text() != item["query_txt"]:
        raise AssertionError("query caption differs from served query record")
    reference_image = page.locator('[data-testid="query-sequence"] [data-testid="query-image"]').first
    target_image = page.locator('[data-testid="query-targets"] [data-testid="query-image"]')
    for image, expected in ((reference_image, ref), (target_image, positive)):
        image.wait_for()
        if image.get_attribute("data-image-id") != expected or image.get_attribute("data-asset-state") != "present":
            raise AssertionError(f"query image identity or availability differs: {expected}")
        image.evaluate("img => img.decode()")
        if image.evaluate("img => img.naturalWidth") <= 0:
            raise AssertionError(f"image did not render: {expected}")
    page.locator('[data-testid="topk-grid"]').wait_for(timeout=NAV_TIMEOUT_MS)
    topk = page.locator('[data-testid="topk"]')
    topk.select_option("20")
    first = page.locator('[data-testid="topk-grid"] .candidate').first
    first.wait_for()
    first_result = top_results[0]
    if first.locator('[data-testid="query-image"]').get_attribute("data-image-id") != Path(first_result["image_id"]).name:
        raise AssertionError("Top-K candidate image ID differs from exported rank one")
    if first.locator(".candidate-rank").inner_text().splitlines()[0] != f"#{first_result['rank']}":
        raise AssertionError("Top-K does not preserve exported candidate rank order")
    if first.locator("small b").first.inner_text() != "—" or any(c.get("score") is not None for c in top_results):
        raise AssertionError("rank-only export was presented as a scored artifact")
    return {"run_id": IMPORTED_RUN_ID, "model_id": run["model_id"],
            "provenance_source": run["provenance_source"], "reproduction_status": run["reproduction_status"],
            "protocol_id": run["protocol_id"], "subset": run["subset"], "condition": "multimodal",
            "query_id": query_id, "category": item["category"], "query_text": item["query_txt"],
            "reference_id": item["reference_id"], "positive_id": item["positive_ids"][0],
            "target_rank": item["target_rank"], "outcome": item["outcome"],
            "candidate_depth": item["candidate_depth"], "common_cohort_count": run["metrics"]["common_cohort_count"],
            "candidate_rank_1": first_result["image_id"], "candidate_scores_present": False}

def run_imported_controls(page, base_url: str, shots: Path, narrow: dict[str, Any]) -> dict[str, Any]:
    """Exercise reachable filters, paging, condition/subset identity and cohort export.

    All reads come from the retained imported CIRCUS artifact. The cohort write is confined to
    this lane's isolated artifact root; it carries one real query and the backend's own artifact
    provenance, never fabricated scores or model output.
    """
    run_id = IMPORTED_RUN_ID
    run_key = f"c:{run_id}"
    query_id = narrow["query_id"]
    base_query = (f"{base_url}/?area=samples&view=explorer&protocol=circus_fashioniq_release_v1"
                  f"&subset=full&runs={run_key}&imported=include&query={query_id}&condition=multimodal")
    active_run = page.locator('[data-testid="active-run"]')
    if active_run.input_value() != run_key:
        active_run.select_option(run_key)
    query_screenshot = shot(page, shots, "I_imported_query")

    # Query search and outcome filters must constrain the same served list, not the inspector's
    # already selected query. Category is intentionally empty for this release artifact.
    page.locator('[data-testid="filter-query-id"]').fill(query_id)
    page.wait_for_function("""() => new URLSearchParams(location.search).get('q') === '7:4436'""", timeout=NAV_TIMEOUT_MS)
    if page.locator('[data-testid="query-id"]').first.inner_text() != query_id:
        raise AssertionError("query-id filter did not preserve exact imported query")
    advanced = page.locator("details").filter(has_text="Bộ lọc nâng cao và nguồn dữ liệu")
    advanced.locator("summary").click()
    category_filter = page.locator('[data-testid="filter-category"]')
    category_filter.select_option("dress")
    response = api_json(base_url, f"/api/composition/runs/{run_id}/queries?subset=full&condition=multimodal&category=dress&limit=1")
    if response.get("total") != 0:
        raise AssertionError(f"source category-null filter should return zero rows: {response.get('total')!r}")
    page.locator('[data-testid="composition-empty"]').wait_for(state="visible", timeout=NAV_TIMEOUT_MS)
    category_filter.select_option("")
    page.locator('[data-testid="filter-outcome"]').select_option("unresolved")
    page.locator('[data-testid="query-list"]').wait_for(timeout=NAV_TIMEOUT_MS)
    unresolved_total = int(page.locator('[data-testid="pager-total"]').get_attribute("data-total") or "0")
    if unresolved_total <= 0:
        raise AssertionError("unresolved outcome filter returned no genuine imported rows")

    # Clear filters, turn condition, and verify the condition-specific rank remains tied to the
    # same query/run/protocol/subset. The three rank values are from the retained artifact.
    page.locator('[data-testid="filter-query-id"]').fill(query_id)
    query_button = page.locator('[data-testid="query-list"] button').filter(has_text=query_id)
    query_button.wait_for(state="visible", timeout=NAV_TIMEOUT_MS)
    if query_button.locator('[data-testid="query-id"]').inner_text() != query_id:
        raise AssertionError("query-id filter did not return target query")
    query_button.click()
    page.wait_for_function("""() => document.querySelector('[data-testid="query-title"]')?.textContent === 'Query 7:4436'""", timeout=NAV_TIMEOUT_MS)
    page.locator('[data-testid="filter-condition"]').select_option("text_only_black_image")
    page.wait_for_function("""() => {
      const card = document.querySelector('[data-testid="query-inspector"]');
      return new URLSearchParams(location.search).get('condition') === 'text_only_black_image'
        && card?.querySelector('[data-testid="query-title"]')?.textContent === 'Query 7:4436'
        && card?.querySelector('[data-testid="rank-text_only_black_image"]')?.getAttribute('data-rank') === '38489';
    }""", timeout=NAV_TIMEOUT_MS)
    inspector = page.locator('[data-testid="query-inspector"]')
    text_rank = inspector.locator('[data-testid="rank-text_only_black_image"]').get_attribute("data-rank")
    if text_rank != "38489":
        raise AssertionError(f"condition switch changed query identity or rank: {text_rank!r}")
    page.locator('[data-testid="filter-condition"]').select_option("image_only_empty_text")
    page.wait_for_function("""() => {
      const card = document.querySelector('[data-testid="query-inspector"]');
      return new URLSearchParams(location.search).get('condition') === 'image_only_empty_text'
        && card?.querySelector('[data-testid="query-title"]')?.textContent === 'Query 7:4436'
        && card?.querySelector('[data-testid="rank-image_only_empty_text"]')?.getAttribute('data-rank') === '14250';
    }""", timeout=NAV_TIMEOUT_MS)
    image_rank = inspector.locator('[data-testid="rank-image_only_empty_text"]').get_attribute("data-rank")
    if image_rank != "14250":
        raise AssertionError(f"image condition rank mismatch: {image_rank!r}")

    # Paging uses the complete list after clearing filters; the exact query inspector remains
    # independently keyed, so page two cannot silently replace query 7:4436.
    page.goto(base_query.replace(f"&query={query_id}", ""), wait_until="domcontentloaded")
    page.locator('[data-testid="query-list"]').wait_for(timeout=NAV_TIMEOUT_MS)
    page.locator('[data-testid="pager-next"]').wait_for(timeout=NAV_TIMEOUT_MS)
    total = int(page.locator('[data-testid="pager-total"]').get_attribute("data-total") or "0")
    if total < 40:
        raise AssertionError(f"genuine imported list too short to verify paging: {total}")
    first_page_query = page.locator('[data-testid="query-id"]').first.inner_text()
    page.locator('[data-testid="pager-next"]').click()
    page.wait_for_function("""() => new URLSearchParams(location.search).get('offset') === '20'""", timeout=NAV_TIMEOUT_MS)
    second_page_query = page.locator('[data-testid="query-id"]').first.inner_text()
    if second_page_query == first_page_query:
        raise AssertionError("next page did not change the served query page")
    page.locator('[data-testid="pager-total"]').locator("button").first.click()
    page.wait_for_function("""() => (new URLSearchParams(location.search).get('offset') || '0') === '0'""", timeout=NAV_TIMEOUT_MS)

    # URL scope identity survives subset navigation; the run's summary exposes the same genuine
    # run under Full/SF/Validated, so this is a read-only scope-preservation check.
    page.goto(base_query.replace("subset=full", "subset=sf"), wait_until="domcontentloaded")
    active_run = page.locator('[data-testid="active-run"]')
    if active_run.input_value() != run_key:
        active_run.select_option(run_key)
    page.locator('[data-testid="query-list"]').wait_for(timeout=NAV_TIMEOUT_MS)
    if "subset=sf" not in page.url or active_run.input_value() != run_key:
        raise AssertionError("subset navigation dropped selected imported run identity")
    page.goto(base_query, wait_until="domcontentloaded")
    if active_run.input_value() != run_key:
        active_run.select_option(run_key)
    page.locator('[data-testid="query-list"]').wait_for(timeout=NAV_TIMEOUT_MS)

    # Save one real query as a composition cohort in isolated storage, then exercise every
    # supported export format through the actual Notebook UI controls.
    cohort_id = "e2e-imported-controls-20261117"
    payload = {"cohort_id": cohort_id, "protocol_id": narrow["protocol_id"],
               "query_ids": [query_id], "run_ids": [run_key],
               "definition": {"preset": "all", "k": 10, "subset": "full",
                              "cohort": narrow["common_cohort_count"], "partial": 0},
               "notes": "isolated genuine imported-query UI evidence"}
    saved = page.request.put(f"{base_url}/api/cohorts", data=json.dumps(payload),
                             headers={"Content-Type": "application/json"})
    if not saved.ok:
        raise AssertionError(f"genuine composition cohort save failed: {saved.status} {saved.text()}")
    page.goto(f"{base_url}/?area=notebook&view=cohorts&protocol={narrow['protocol_id']}&subset=full",
              wait_until="domcontentloaded")
    cohort_card = page.locator(f'[data-testid="cohort-{cohort_id}"]')
    cohort_card.wait_for(timeout=NAV_TIMEOUT_MS)
    cohort_screenshot = shot(page, shots, "I_imported_cohort_exports")
    exports: dict[str, str] = {}
    for fmt in ("json", "csv", "markdown"):
        with page.expect_download(timeout=NAV_TIMEOUT_MS) as download_info:
            cohort_card.locator(f'[data-testid="cohort-export-{cohort_id}-{fmt}"]').click()
        target = shots / f"{cohort_id}.{fmt if fmt != 'markdown' else 'md'}"
        download_info.value.save_as(str(target))
        exports[fmt] = str(target)
        content = target.read_text(encoding="utf-8")
        if fmt == "json":
            data = json.loads(content)
            if data.get("protocol_id") != narrow["protocol_id"] or data.get("query_ids") != [query_id] or data.get("run_ids") != [run_key]:
                raise AssertionError("JSON cohort export lost query/protocol/run identity")
            row = next(iter(data.get("metrics", [])), {})
            if any(row.get(key) != value for key, value in (("query_count", 1), ("scored_count", 1), ("mean_target_rank", 31987), ("r10", 0))):
                raise AssertionError(f"JSON cohort metrics differ from selected query: {row!r}")
        elif fmt == "csv":
            rows = list(csv.DictReader(io.StringIO(content)))
            if len(rows) != 1 or rows[0].get("run_id") != run_id or rows[0].get("query_count") != "1" or rows[0].get("scored_count") != "1" or rows[0].get("mean_target_rank") != "31987" or rows[0].get("r10") != "0.00":
                raise AssertionError(f"CSV cohort metric row mismatch: {rows!r}")
        elif narrow["protocol_id"] not in content or run_id not in content or "| 1 | 0.00 |" not in content or "| 31987 | 31987 |" not in content:
            raise AssertionError("Markdown cohort export lost protocol/run/query-count/rank metrics")

    # Common cohort is the run's recorded 6,003-query denominator. Overlap is computed only from
    # real rank rows: compare top-20 image sets across the three served conditions for qid7:4436.
    overlap: dict[str, float] = {}
    sets: dict[str, set[str]] = {}
    for condition_name in ("multimodal", "text_only_black_image", "image_only_empty_text"):
        result = api_json(base_url, f"/api/composition/runs/{run_id}/queries?subset=full&condition={condition_name}&query_id={query_id}&limit=1&top_k=20")
        sets[condition_name] = {row["image_id"] for row in result["items"][0]["top_results"]}
    for left, right in (("multimodal", "text_only_black_image"),
                        ("multimodal", "image_only_empty_text"),
                        ("text_only_black_image", "image_only_empty_text")):
        union = sets[left] | sets[right]
        overlap[f"{left}:{right}"] = len(sets[left] & sets[right]) / len(union) if union else 1.0
    return {"filters": {"query_id": query_id, "category_empty_reason": "release CIRCUS category is null",
                         "unresolved_total": unresolved_total},
            "condition_ranks": {"multimodal": narrow["target_rank"],
                                "text_only_black_image": int(text_rank),
                                "image_only_empty_text": int(image_rank)},
            "pagination": {"total": total, "first_page_query": first_page_query,
                            "second_page_query": second_page_query, "page_size": 20},
            "scope": {"protocol_id": narrow["protocol_id"], "run_id": run_id,
                      "full_common_cohort": narrow["common_cohort_count"], "sf_run_preserved": True},
            "overlap_top20_jaccard_api_diagnostic": overlap, "cohort_id": cohort_id, "exports": exports,
            "query_screenshot": query_screenshot, "cohort_screenshot": cohort_screenshot,
            "warning": "isolated development catalog; genuine RELEASED_ARTIFACT_IMPORT, not model output; Top-20 overlap is API-derived diagnostic, not UI overlap"}
def run_journeys_for_viewport(
    page, base_url: str, vp: dict[str, Any], shots_dir: Path, recorder: Recorder,
    protocol_id: str = PROTOCOL,
) -> dict[str, Any]:
    vp_name = vp["name"]
    vp_shots = shots_dir / vp_name
    vp_shots.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"viewport": vp_name, "journeys": {}, "screenshots": {}}
    models = api_json(base_url, "/api/models")
    publication = api_json(base_url, "/api/models/publication")

    # Journey A: rendered published/registry totals and deferred population match served APIs.
    t0 = time.perf_counter()
    page.goto(base_url, wait_until="domcontentloaded")
    page.wait_for_selector("[data-testid=area-overview]", timeout=NAV_TIMEOUT_MS)
    page.wait_for_function(
        """() => ['published-count', 'registry-count'].every(name =>
            /^[0-9][0-9,]*$/.test(document.querySelector(`[data-testid=overview-${name}]`)?.textContent?.trim() || ''))""",
        timeout=NAV_TIMEOUT_MS)
    stats = registry_stat(page)
    assert_registry_counts(stats, models, publication)
    rendered = int(stats["published-count"].replace(",", ""))
    set_protocol(page, protocol_id)
    report["journeys"]["A"] = {"status": "PASS",
                               "detail": f"published={rendered}, registry={stats['registry-count']}, "
                                         f"deferred={publication['deferred_count']} match served APIs",
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
        .map(row => { const d = row.querySelector('details');
                      return d ? d.innerText.trim() : null; }).filter(Boolean)""")
    if blocked and not disclosures:
        raise AssertionError(f"API declares blockers for {len(blocked)} models but no disclosure is rendered")
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
    # Journey P uses synthetic API data solely to verify parity UI rendering.
    t0 = time.perf_counter()
    run_parity_fixture(page, base_url, recorder)
    report["journeys"]["P"] = {"status": "PASS", "detail": "UI-only incomplete, malformed, mandatory-failure, coverage and mixed-standard fixture", "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
    report["screenshots"]["P"] = shot(page, vp_shots, "P_parity_fixture")
    if vp_name == "1440x900" and os.environ.get("WORKBENCH_E2E_IMPORTED_RUN") == "1":
        t0 = time.perf_counter()
        report["imported_run"] = run_imported_query(page, base_url, vp_shots)
        report["imported_controls"] = run_imported_controls(page, base_url, vp_shots, report["imported_run"])
        report["screenshots"]["I_query"] = report["imported_controls"]["query_screenshot"]
        report["journeys"]["I"] = {"status": "PASS", "detail": "real imported query filters, paging, conditions, scope and exports; Top-20 overlap is API-derived, not a UI overlap view; not model acceptance", "timing_ms": round((time.perf_counter() - t0) * 1000.0, 2)}
        report["screenshots"]["I_cohort_exports"] = report["imported_controls"]["cohort_screenshot"]
    return report

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tmpdir", type=Path, default=DEFAULT_TMPDIR)
    parser.add_argument("--viewport", choices=[vp["name"] for vp in VIEWPORTS])
    parser.add_argument("--backend-port", type=int, default=8775)
    parser.add_argument("--frontend-port", type=int, default=5775)
    parser.add_argument("--source-artifacts", type=Path)
    parser.add_argument("--fashioniq-root", type=Path)
    parser.add_argument("--circus-release-root", type=Path)
    args = parser.parse_args(argv)

    existing = os.environ.get("WORKBENCH_E2E_BASE_URL", "").rstrip("/")
    tmp_dir = args.tmpdir
    shots_dir = tmp_dir / "shots2"
    shots_dir.mkdir(parents=True, exist_ok=True)
    results_dir = tmp_dir / "artifacts" / "results"
    images_dir = args.fashioniq_root or tmp_dir / "fashioniq"
    isolated_artifacts = None
    if args.source_artifacts is not None:
        isolated_artifacts = stage_imported_run(args.source_artifacts, tmp_dir / "artifacts")
    elif not existing:
        build_fixture(results_dir, images_dir)

    def terminate(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    sm = None
    server_timing: dict[str, Any] = {}
    full_results: dict[str, Any] = {"server_timing": server_timing, "base_url": existing,
                                    "viewports": {}}
    failures = 0
    try:
        if existing:
            base_url = existing
            print(f"[OK] Reusing the running workbench at {base_url}")
        else:
            sm = ServerManager(tmp_dir, args.backend_port, args.frontend_port,
                               isolated_artifacts, args.fashioniq_root,
                               args.circus_release_root)
            b_ms, f_ms = sm.start()
            base_url = f"http://127.0.0.1:{args.frontend_port}"
            server_timing = {"backend_startup_ms": b_ms, "frontend_startup_ms": f_ms}
            print(f"[OK] Backend ready in {b_ms:.1f}ms, Frontend ready in {f_ms:.1f}ms at {base_url}")
        full_results["base_url"] = base_url
        full_results["server_timing"] = server_timing
        with sync_playwright() as playwright:
            browser = launch(playwright)
            try:
                viewports = [vp for vp in VIEWPORTS if args.viewport is None or vp["name"] == args.viewport]
                for vp in viewports:
                    vp_name = vp["name"]
                    print(f"\\n--- Viewport {vp_name} ({vp['width']}x{vp['height']}) ---")
                    rec = Recorder()
                    ctx = browser.new_context(
                        viewport={"width": vp["width"], "height": vp["height"]},
                        is_mobile=vp["is_mobile"], has_touch=vp["is_mobile"])
                    page = ctx.new_page()
                    rec.attach(page)
                    try:
                        protocol_id = "circus_fashioniq_release_v1" if isolated_artifacts is not None else PROTOCOL
                        vp_report = run_journeys_for_viewport(page, base_url, vp, shots_dir, rec, protocol_id)
                        navigation_cancellations = [failure for failure in rec.failed_requests
                                                    if "net::ERR_ABORTED" in failure]
                        unexplained_failed_requests = [failure for failure in rec.failed_requests
                                                       if "net::ERR_ABORTED" not in failure]
                        vp_report["recorder"] = {
                            "console_errors_unexpected": rec.console,
                            "failed_requests": rec.failed_requests,
                            "navigation_cancellations": navigation_cancellations,
                            "failed_requests_unexpected": unexplained_failed_requests,
                        }
                        assert not rec.console and not rec.page_errors and not unexplained_failed_requests, vp_report["recorder"]
                        full_results["viewports"][vp_name] = vp_report
                        print(f"[PASS] Viewport {vp_name}: journeys "
                              f"{sorted(vp_report['journeys'])} on {len(api_json(base_url, '/api/models'))} models")
                    except Exception as ex:
                        failures += 1
                        error_shot = shots_dir / vp_name / "failure.png"
                        error_shot.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            page.screenshot(path=str(error_shot), full_page=True)
                        except Exception:
                            error_shot = None
                        print(f"[FAIL] Viewport {vp_name}: {ex}", file=sys.stderr)
                        full_results["viewports"][vp_name] = {
                            "status": "FAIL", "error": str(ex),
                            "screenshot": str(error_shot) if error_shot else None,
                            "recorder": {"console_errors": rec.console, "page_errors": rec.page_errors,
                                         "failed_requests": rec.failed_requests},
                        }
                    finally:
                        ctx.close()
            finally:
                browser.close()
    except KeyboardInterrupt as ex:
        failures += 1
        full_results["interrupted"] = str(ex)
    except Exception as ex:
        failures += 1
        full_results["setup_error"] = str(ex)
        print(f"[FAIL] E2E setup: {ex}", file=sys.stderr)
    finally:
        if sm is not None:
            sm.stop()
        out_json = tmp_dir / "g2_e2e_results.json"
        out_json.write_text(json.dumps(full_results, indent=2))
        print(f"\\nResults saved to {out_json}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(main())
