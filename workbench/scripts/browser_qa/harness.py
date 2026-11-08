"""Plumbing for the browser QA harness.

Four pieces, all deliberately small:

* a **selector registry** that prefers `data-testid`, falls back to role+name and then legacy CSS,
  and reports every miss as a distinct failure category instead of crashing;
* a **flow ledger** that records pass/fail/blocked per assertion plus the browser-side evidence
  (console errors, page errors, failed requests, HTTP >= 400, overflow pixels, screenshots);
* **DOM probes** (overflow, click-target size, image distortion, focus visibility, sticky occlusion)
  as plain JS so they survive a redesign that keeps the DOM semantics;
* a **stdlib API client** used as ground truth, so no assertion depends on the UI agreeing with
  itself.

The UI is being redesigned underneath this file. Nothing here asserts a specific route, tab name or
class name: every locator goes through `SELECTORS` and a miss is data, not a crash.
"""
from __future__ import annotations

import itertools
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
DEFAULT_BROWSERS = REPO / "workbench/envs/ctl312h/.playwright-browsers"
NAV_TIMEOUT_MS = 60_000
RESOLVE_TIMEOUT_MS = 2_000
# Browsers auto-request /favicon.ico; a missing one is not a UI defect.
IGNORE_REQUEST_SUFFIXES = ("/favicon.ico",)

# --------------------------------------------------------------------------- selectors

#: Logical name -> ordered locator strategies. Each strategy is a tuple whose first element is the
#: kind: `testid`, `role`, `label`, `css`, `css_text`, `text`, `tab`. `{placeholders}` are filled
#: with any exceptions; a list value expands into one strategy per element, so a renamed area is
#: found whether it is called "Overview" or "Tổng quan".
SELECTORS: dict[str, list[tuple[str, ...]]] = {
    # --- shell / navigation
    "nav": [("testid", "workbench-nav"), ("role", "navigation"), ("css", "nav")],
    "nav.item": [("testid", "area-nav-{name}"), ("css_text", "nav button", "{name}"),
                 ("css_text", "nav a", "{name}"), ("css", "nav [aria-current='page']")],
    "nav.active": [("testid", "area-nav-active"), ("css", "nav [aria-current='page']"),
                   ("css", "nav button.active")],
    "area.root": [("testid", "area-{name}"), ("css", "main")],
    # --- shared chrome
    "error.banner": [("testid", "error-banner"), ("css", ".error")],
    "error.dismiss": [("testid", "error-dismiss"), ("role", "button", "Dismiss"),
                      ("testid", "notice-dismiss")],
    "error.retry": [("testid", "error-retry"), ("role", "button", "Thử lại"),
                    ("role", "button", "Retry")],
    "error.state": [("testid", "error-state"), ("testid", "empty-state")],
    "empty.state": [("testid", "empty-state"),
                    ("css", ".empty-notice, .notice-aggregate-only, .notice-banner")],
    "notice": [("testid", "notice"), ("css", ".notice-aggregate-only, .notice-banner")],
    # --- overview (today: Dashboard)
    "overview.cards": [("testid", "overview-cards"), ("css", ".cards")],
    # --- run selection (both route families)
    "runs.checkbox": [("testid", "run-option-{key}"), ("testid", "run-option-{run_id}"),
                      ("css", "[data-testid^='run-option-']")],
    "runs.row": [("testid", "run-row-{key}"), ("testid", "run-row-{run_id}"),
                 ("css", ".run")],
    "runs.selected_count": [("testid", "selected-run-count"), ("css", ".controls span")],
    "runs.any_option": [("css", "[data-testid^='run-option-']")],
    # --- composition / evaluation table
    "comp.root": [("testid", "area-evaluation"), ("css", ".composition-view")],
    "comp.subset": [("testid", "subset-pill-{subset}"), ("tab_sub", "{label}"),
                    ("css_text", ".subset-pill", "{label}")],
    "subset.select": [("testid", "subset-select"), ("label_sub", "Subset")],
    "comp.subset_any": [("css", "[data-testid^='subset-pill-']"), ("css", ".subset-pill")],
    "comp.table": [("testid", "composition-table"), ("css", "table.composition-table")],
    "comp.row": [("testid", "composition-row-{run_id}"),
                 ("css", "table.composition-table tbody tr")],
    "comp.drill": [("testid", "drill-down"), ("testid", "query-inspector"),
                   ("css", ".drilldown-section")],
    "comp.pager": [("testid", "pager-total"), ("css", ".pagination-bar span")],
    "comp.pager.next": [("testid", "pager-next"), ("role", "button", "Next"),
                        ("role", "button", "Sau")],
    "comp.cards": [("testid", "query-list"), ("css", ".query-card")],
    "comp.qid": [("testid", "query-id"), ("css", ".query-card .query-id")],
    "comp.rank": [("testid", "rank-{condition}"), ("css", ".rank-value")],
    "comp.rank.any": [("css", "[data-testid^='rank-']"), ("css", ".rank-value")],
    "comp.empty": [("testid", "composition-empty"), ("testid", "empty-state"),
                   ("css", ".empty-notice")],
    "comp.filter": [("testid", "filter-{name}"), ("label_sub", "{label}")],
    "comp.search": [("testid", "filter-query-id"), ("placeholder", "e.g. 7:1…"),
                    ("placeholder", "vd 7:603")],
    "comp.topk": [("testid", "topk-grid"), ("css", ".grid")],
    "comp.image": [("testid", "query-image"), ("css", ".query-card img"),
                   ("css", ".shots img")],
    # --- sample explorer / query inspector
    "explorer.active_run": [("testid", "active-run"), ("label", "Active model/run")],
    "explorer.query_title": [("testid", "query-title"), ("css", "section h3")],
    "explorer.topk": [("testid", "topk-grid"), ("css", ".grid article")],
    "explorer.image": [("testid", "query-image"), ("css", ".images img")],
    "explorer.jump": [("testid", "jump-query-id"), ("placeholder", "Paste query ID")],
    "explorer.jump.button": [("testid", "jump-submit"), ("role", "button", "Jump")],
    "explorer.candidates_unavailable": [("testid", "candidates-unavailable")],
    # --- annotations
    "annot.note": [("testid", "annotation-note"), ("css", "textarea")],
    "annot.save": [("testid", "annotation-save"), ("role", "button", "Save / update"),
                   ("role", "button", "Lưu / cập nhật")],
    "annot.query": [("testid", "annotation-query-id"), ("css", "section p")],
    # --- failure analytics / cohorts
    "fail.section": [("testid", "area-failures"), ("css", "section table")],
    "fail.row": [("testid", "analysis-row"), ("css", "section table tbody tr")],
    "fail.preset": [("testid", "failure-preset"), ("css", "section .filters select")],
    "cohort.name": [("testid", "cohort-id"), ("placeholder", "cohort ID")],
    "cohort.save": [("testid", "cohort-save"), ("role", "button", "Save filtered set")],
    "cohort.item": [("testid", "cohort-{cohort_id}"), ("css", "section article")],
    # --- models
    "models.section": [("testid", "area-models"), ("css", "section h2")],
    "models.row": [("testid", "model-row-{model_id}")],
}

SELECTOR_USED: dict[str, str] = {}


class SelectorMiss(Exception):
    def __init__(self, key: str, tried: list[str]):
        super().__init__(f"selector miss: {key} (tried {len(tried)} strategies)")
        self.key = key
        self.tried = tried


def _locator(scope: Any, strategy: tuple[str, ...]) -> Any:
    kind = strategy[0]
    if kind == "testid":
        return scope.get_by_test_id(strategy[1])
    if kind == "role":
        return scope.get_by_role(strategy[1], name=strategy[2], exact=True)
    if kind == "role_sub":
        # Accessible names often carry counts ("Full 6,003 queries · 74,381 gallery"); a substring
        # match is the only stable form while the redesign changes copy.
        return scope.get_by_role(strategy[1], name=strategy[2], exact=False)
    if kind == "tab":
        return scope.get_by_role("tab", name=strategy[1], exact=True)
    if kind == "tab_sub":
        return scope.get_by_role("tab", name=strategy[1], exact=False)
    if kind == "label":
        return scope.get_by_label(strategy[1], exact=True)
    if kind == "label_sub":
        # A wrapping <label> whose text is "Condition\nMultimodal\n…" has no exact accessible name;
        # a substring match is what survives both a redesign and a copied control.
        return scope.get_by_label(strategy[1], exact=False)
    if kind == "placeholder":
        return scope.get_by_placeholder(strategy[1], exact=True)
    if kind == "text":
        return scope.get_by_text(strategy[1], exact=True)
    if kind == "css":
        return scope.locator(strategy[1])
    if kind == "css_text":
        return scope.locator(strategy[1], has_text=strategy[2])
    raise ValueError(f"unknown selector kind {kind!r}")


def strategies(key: str, fmt: dict[str, Any] | None = None) -> list[tuple[str, ...]]:
    fmt = fmt or {}
    raw = SELECTORS[key]
    keys = sorted(fmt)
    expansions = [fmt[k] if isinstance(fmt[k], list) else [fmt[k]] for k in keys]
    out: list[tuple[str, ...]] = []
    for combo in itertools.product(*expansions) if keys else [()]:
        values = dict(zip(keys, combo))
        for strat in raw:
            try:
                out.append(tuple(part.format(**values) if isinstance(part, str) else part
                                 for part in strat))
            except KeyError:
                continue  # a strategy needing a placeholder the caller did not supply
    return out


def resolve(scope: Any, key: str, fmt: dict[str, Any] | None = None,
            timeout: int | None = None) -> Any:
    """First matching strategy wins. Raises `SelectorMiss` when none matches.

    A miss is a first-class outcome: the caller records it and continues, and the harness emits it
    as its own failure category plus a `data-testid` request for the redesign.
    """
    tried: list[str] = []
    for strat in strategies(key, fmt):
        tried.append(":".join(str(part) for part in strat))
        try:
            loc = _locator(scope, strat)
            loc.first.wait_for(state="visible", timeout=timeout or RESOLVE_TIMEOUT_MS)
            SELECTOR_USED[key] = tried[-1]
            return loc
        except Exception:
            continue
    raise SelectorMiss(key, tried)


def optional(scope: Any, key: str, fmt: dict[str, Any] | None = None,
             timeout: int = 800) -> Any | None:
    try:
        return resolve(scope, key, fmt, timeout)
    except SelectorMiss:
        return None


def count(loc: Any) -> int:
    try:
        return loc.count()
    except Exception:
        return 0


def texts(loc: Any) -> list[str]:
    try:
        return [t.strip() for t in loc.all_inner_texts()]
    except Exception:
        return []


# --------------------------------------------------------------------------- DOM probes

#: Horizontal overflow, distinguishing real page overflow from content that lives inside an
#: intentionally scrollable container (a wide table in `overflow-x:auto` is not a page defect).
PROBE_OVERFLOW = r"""
() => {
  const vw = window.innerWidth;
  const de = document.documentElement, bd = document.body;
  const pageOverflow = Math.round(Math.max(de.scrollWidth, bd ? bd.scrollWidth : 0) - vw);
  const label = (el) => (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 60);
  const path = (el) => {
    const parts = [];
    for (let n = el; n && n.nodeType === 1 && parts.length < 5; n = n.parentElement) {
      let s = n.tagName.toLowerCase();
      if (n.id) { parts.unshift(s + '#' + n.id); break; }
      const cls = (typeof n.className === 'string' ? n.className : '').trim().split(/\s+/).filter(Boolean);
      if (cls.length) s += '.' + cls.slice(0, 2).join('.');
      parts.unshift(s);
    }
    return parts.join('>');
  };
  const contained = (el) => {
    for (let n = el.parentElement; n; n = n.parentElement) {
      const ox = getComputedStyle(n).overflowX;
      if (ox === 'auto' || ox === 'scroll' || ox === 'hidden') return true;
    }
    return false;
  };
  const offenders = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;
    const over = Math.round(Math.max(0, r.right - vw) + Math.max(0, -r.left));
    if (over <= 1) continue;
    if (contained(el)) continue;
    offenders.push({ el: path(el), over, label: label(el) });
  }
  offenders.sort((a, b) => b.over - a.over);
  return { overflowPx: pageOverflow, offenders: offenders.slice(0, 10) };
}
"""

#: Interactive elements whose box is smaller than the 24x24 CSS px minimum.
PROBE_TARGETS = r"""
() => {
  const sel = 'a[href],button,input,select,textarea,[role="button"],[role="tab"],[role="checkbox"],[onclick]';
  const label = (el) => (el.getAttribute('aria-label') || el.textContent || el.value || el.name || '')
    .replace(/\s+/g, ' ').trim().slice(0, 60);
  // A native checkbox inside its <label> is hit through the label's box, which is the real target:
  // measuring the 13px glyph would report every accessible checkbox form as a defect.
  const target = (el) => {
    if (el.tagName === 'INPUT' && (el.type === 'checkbox' || el.type === 'radio')) {
      const wrap = el.closest('label');
      if (wrap) {
        const wr = wrap.getBoundingClientRect();
        if (wr.width >= 24 && wr.height >= 24) return [wrap, wr, 'label'];
      }
      if (el.id) {
        const ext = document.querySelector('label[for="' + CSS.escape(el.id) + '"]');
        if (ext) {
          const er = ext.getBoundingClientRect();
          if (er.width >= 24 && er.height >= 24) return [ext, er, 'label[for]'];
        }
      }
    }
    return [el, el.getBoundingClientRect(), 'self'];
  };
  const small = [];
  let total = 0;
  for (const el of document.querySelectorAll(sel)) {
    if (el.type === 'hidden') continue;
    const cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') continue;
    const [host, r, via] = target(el);
    if (r.width === 0 || r.height === 0) continue;
    total += 1;
    if (r.width < 24 || r.height < 24) {
      small.push({ tag: host.tagName.toLowerCase(), label: label(el), via,
                   w: Math.round(r.width), h: Math.round(r.height) });
    }
  }
  return { total, small: small.slice(0, 40), smallCount: small.length };
}
"""

#: Image distortion. Only `object-fit: fill` can distort; `cover`/`contain` crop or letterbox by
#: design, so they are reported as information and never as a failure.
PROBE_IMAGES = r"""
() => {
  const out = [];
  for (const img of document.querySelectorAll('img')) {
    const r = img.getBoundingClientRect();
    const cs = getComputedStyle(img);
    const loaded = img.complete && img.naturalWidth > 0;
    const naturalRatio = img.naturalWidth / img.naturalHeight;
    const renderedRatio = r.width / r.height;
    const drift = (cs.objectFit === 'fill' && loaded && Number.isFinite(renderedRatio) &&
                   renderedRatio > 0) ? Math.abs(naturalRatio - renderedRatio) / naturalRatio : null;
    // `loading="lazy"` images below the fold are not broken; only count one that is on screen.
    const onScreen = r.width > 0 && r.height > 0 && cs.display !== 'none' &&
                     r.bottom > 0 && r.top < window.innerHeight;
    out.push({
      src: (img.getAttribute('src') || '').slice(0, 120), alt: (img.alt || '').slice(0, 60),
      loaded, fit: cs.objectFit, lazy: img.getAttribute('loading') === 'lazy',
      natural: [img.naturalWidth, img.naturalHeight],
      rendered: [Math.round(r.width), Math.round(r.height)],
      visible: onScreen,
      ratioDrift: drift,
    });
  }
  return out;
}
"""

#: Sticky/fixed elements that cover text that is not inside them.
PROBE_STICKY = r"""
() => {
  const stickies = [];
  for (const el of document.querySelectorAll('body *')) {
    const cs = getComputedStyle(el);
    if (cs.position !== 'sticky' && cs.position !== 'fixed') continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    stickies.push([el, r]);
  }
  if (!stickies.length) return { stickies: 0, occluded: [] };
  const textish = 'p,td,th,h1,h2,h3,h4,li,span,small,b,code,label,figcaption,a';
  const occluded = [];
  for (const el of document.querySelectorAll(textish)) {
    if (!(el.textContent || '').trim()) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    for (const [sticky, sr] of stickies) {
      if (sticky === el || sticky.contains(el)) continue;
      if (r.bottom < sr.top || r.top > sr.bottom || r.right < sr.left || r.left > sr.right) continue;
      const x = Math.min(Math.max(r.left + r.width / 2, 0), window.innerWidth - 1);
      const y = Math.min(Math.max(r.top + r.height / 2, 0), window.innerHeight - 1);
      const top = document.elementFromPoint(x, y);
      if (top && sticky.contains(top) && !el.contains(top)) {
        occluded.push({ text: (el.textContent || '').replace(/\s+/g, ' ').trim().slice(0, 60),
                        sticky: sticky.tagName.toLowerCase() + '.' +
                                (typeof sticky.className === 'string' ? sticky.className.split(/\s+/)[0] : '') });
        break;
      }
    }
  }
  return { stickies: stickies.length, occluded: occluded.slice(0, 10) };
}
"""

#: Focus visibility for whatever currently has focus.
PROBE_FOCUS = r"""
() => {
  const el = document.activeElement;
  if (!el || el === document.body) return { tag: null };
  const cs = getComputedStyle(el);
  const outline = cs.outlineStyle !== 'none' && parseFloat(cs.outlineWidth || '0') > 0;
  const shadow = cs.boxShadow && cs.boxShadow !== 'none';
  return {
    tag: el.tagName.toLowerCase(),
    label: (el.getAttribute('aria-label') || el.textContent || el.value || el.name || '')
      .replace(/\s+/g, ' ').trim().slice(0, 50),
    outline, shadow, visible: outline || shadow,
    outlineStyle: cs.outlineStyle, outlineWidth: cs.outlineWidth,
  };
}
"""


# --------------------------------------------------------------------------- API client

class Api:
    """Direct backend client: the ground truth the DOM is compared against."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.calls: list[dict[str, Any]] = []

    def request(self, method: str, path: str, body: Any | None = None) -> tuple[int, Any]:
        url = f"{self.base_url}{path}"
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                payload = json.loads(raw) if raw else None
                self.calls.append({"method": method, "path": path, "status": resp.status})
                return resp.status, payload
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                payload = json.loads(raw) if raw else None
            except ValueError:
                payload = raw.decode("utf-8", "replace")[:400]
            self.calls.append({"method": method, "path": path, "status": exc.code,
                               "error": str(payload)[:200]})
            return exc.code, payload
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self.calls.append({"method": method, "path": path, "status": None,
                               "error": repr(exc)[:200]})
            return 0, None

    def get(self, path: str, attempts: int = 3) -> Any | None:
        """GET with a couple of retries: the backend is restarted by other agents while this runs,
        and an empty answer would otherwise be reported as a data fact."""
        for attempt in range(attempts):
            status, payload = self.request("GET", path)
            if status == 200:
                return payload
            if status not in (0, None) or attempt + 1 >= attempts:
                return None
            time.sleep(1.5)
        return None

    def put(self, path: str, body: Any) -> tuple[int, Any]:
        return self.request("PUT", path, body)


# --------------------------------------------------------------------------- flow ledger

class Check:
    __slots__ = ("id", "ok", "detail", "category")

    def __init__(self, ident: str, ok: bool | None, detail: str, category: str = "assertion"):
        self.id = ident
        self.ok = ok
        self.detail = detail
        self.category = category

    def to_json(self) -> dict[str, Any]:
        return {"id": self.id, "ok": self.ok, "detail": self.detail, "category": self.category}


class Flow:
    """One flow at one viewport: assertions, misses, browser evidence, screenshots."""

    def __init__(self, flow_id: str, name: str, viewport: str):
        self.flow_id = flow_id
        self.name = name
        self.viewport = viewport
        self.checks: list[Check] = []
        self.misses: list[dict[str, Any]] = []
        self.screenshots: list[str] = []
        self.started = time.time()
        self.notes: list[str] = []
        self.overflow_px: int | None = None
        self.evidence: dict[str, Any] = {}

    # -- assertions
    def check(self, ident: str, ok: bool, detail: str, category: str = "assertion") -> bool:
        self.checks.append(Check(ident, bool(ok), detail, category))
        return bool(ok)

    def eq(self, ident: str, expected: Any, actual: Any, note: str = "",
           category: str = "assertion") -> bool:
        ok = expected == actual
        detail = f"{note}: expected {expected!r}, got {actual!r}".strip()
        return self.check(ident, ok, detail, category)

    def blocked(self, ident: str, reason: str) -> None:
        self.checks.append(Check(ident, None, reason, "blocked"))

    def miss(self, exc: SelectorMiss, purpose: str = "") -> None:
        self.misses.append({"key": exc.key, "purpose": purpose, "tried": exc.tried})
        self.checks.append(Check(f"selector:{exc.key}", False,
                                 f"{purpose or 'resolve'} — no strategy matched "
                                 f"({len(exc.tried)} tried)", "selector_miss"))

    def note(self, text: str) -> None:
        self.notes.append(text)

    # -- status
    def status(self) -> str:
        """fail > blocked > selector_miss > pass.

        `evidence` checks record a fact that is not by itself a defect (a gallery image that is
        genuinely absent is data coverage, and flow (e) checks image ids against the API instead).
        `prerequisite` checks describe what the app cannot do yet; when one fails, the selector
        misses downstream are consequences, not new information, so `blocked` outranks them.
        """
        hard = [c for c in self.checks
                if c.ok is False and c.category not in
                ("selector_miss", "prerequisite", "evidence")]
        if hard:
            return "fail"
        if any(c.ok is None or (c.ok is False and c.category == "prerequisite")
               for c in self.checks):
            return "blocked"
        if any(c.category == "selector_miss" for c in self.checks):
            return "selector_miss"
        return "pass"

    def to_json(self) -> dict[str, Any]:
        return {
            "flow": self.flow_id,
            "name": self.name,
            "viewport": self.viewport,
            "status": self.status(),
            "passed": self.status() == "pass",
            "duration_ms": int((time.time() - self.started) * 1000),
            "assertions": [c.to_json() for c in self.checks],
            "failed_assertions": [c.to_json() for c in self.checks if c.ok is False],
            "selector_misses": self.misses,
            "overflow_px": self.overflow_px,
            "notes": self.notes,
            "evidence": self.evidence,
            "screenshots": self.screenshots,
        }


class PageCapture:
    """Browser-side evidence: console errors, page errors, failed requests, HTTP >= 400.

    Requests the harness itself aborts (to simulate a broken backend) are expected and are listed
    separately, so an injected failure never masquerades as a real one.
    """

    def __init__(self, page: Any):
        self.page = page
        self.console: list[str] = []
        self.page_errors: list[str] = []
        self.failed: list[dict[str, str]] = []
        self.http: list[dict[str, Any]] = []
        self.expected: list[str] = []
        self.injected: list[dict[str, str]] = []
        page.on("console", self._on_console)
        page.on("pageerror", self._on_pageerror)
        page.on("requestfailed", self._on_failed)
        page.on("response", self._on_response)

    def expect_failure(self, pattern: str) -> None:
        self.expected.append(pattern)

    def _is_expected(self, url: str) -> bool:
        return any(p in url for p in self.expected)

    def _on_console(self, msg: Any) -> None:
        if msg.type == "error":
            self.console.append(msg.text[:400])

    def _on_pageerror(self, exc: Any) -> None:
        self.page_errors.append(str(exc)[:400])

    def _on_failed(self, request: Any) -> None:
        url = request.url
        if url.split("?")[0].endswith(IGNORE_REQUEST_SUFFIXES):
            return
        record = {"url": url[:200], "failure": str(request.failure)[:200]}
        (self.injected if self._is_expected(url) else self.failed).append(record)

    def _on_response(self, response: Any) -> None:
        if response.status < 400:
            return
        url = response.url
        if url.split("?")[0].endswith(IGNORE_REQUEST_SUFFIXES):
            return
        record = {"status": response.status, "url": url[:200]}
        if self._is_expected(url):
            self.injected.append(record)
        else:
            self.http.append(record)

    def snapshot(self) -> dict[str, Any]:
        return {
            "console_errors": list(self.console),
            "page_errors": list(self.page_errors),
            "failed_requests": list(self.failed),
            "http_ge_400": list(self.http),
            "injected_failures": list(self.injected),
        }

    def clear(self) -> None:
        self.console.clear()
        self.page_errors.clear()
        self.failed.clear()
        self.http.clear()
        self.injected.clear()


def ignorable_image_404(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if "/api/images/" in str(r.get("url", ""))]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def viewport_size(text: str) -> tuple[int, int]:
    width, _, height = text.partition("x")
    return int(width), int(height)


def parse_pager_total(text: str) -> tuple[int | None, int | None]:
    """`Showing 1–20 of 4,069 queries` -> (shown_count, total)."""
    match = re.search(r"of\s+([\d,\.]+)", text)
    total = int(re.sub(r"[^\d]", "", match.group(1))) if match else None
    first = re.search(r"Showing\s+(\d+)[–-](\d+)", text)
    shown = (int(first.group(2)) - int(first.group(1)) + 1) if first else None
    return shown, total


def ensure_browser_path() -> None:
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", str(DEFAULT_BROWSERS))
