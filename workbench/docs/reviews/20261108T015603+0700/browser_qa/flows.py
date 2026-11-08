#!/usr/bin/env python3
"""Interactive browser QA flows against the real workbench stack.

Reuses `workbench/frontend/tests/browser_qa.py` for browser launch, console/page/request
recorders and PNG sizing -- no second convention for the same job.

Every number asserted here is read back from the live API first, so a pass means "the DOM
agrees with the backend", never "the DOM agrees with a constant in this file". The one
exception is BRIEF, which pins the values the review brief promised; it is asserted against
the API (proof the fixture data is the real encoder run) and separately against the DOM.

    PLAYWRIGHT_BROWSERS_PATH=workbench/envs/ctl312h/.playwright-browsers \
    workbench/envs/ctl312h/bin/python <bundle>/browser_qa/flows.py --out <bundle>/browser_qa
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO / "workbench/frontend/tests"))
import browser_qa as hq  # noqa: E402

API = "http://127.0.0.1:8000"
UI = "http://127.0.0.1:5173"
REAL_RUN = "6a528e877c328772e2c7bb28"
SPARSE_RUN = "eda30a37c0bc17ac43844f09"  # real run, 8 ranked queries out of the subset
VIEWPORTS = [(1440, 900, "1440x900"), (768, 1024, "768x1024"), (390, 844, "390x844")]

# What the review brief promised the UI shows for the real encoder run.
BRIEF = {
    "full": dict(queries=6003, mm_ndcg=0.2959, mm_r10=31.0, text=0.1561, image=0.1073, gap=47.23, coverage=100.0),
    "sf": dict(queries=4069, mm_ndcg=0.2189, mm_r10=18.7, text=0.1077, image=0.0977, gap=50.77, coverage=100.0),
    "validated": dict(queries=586, mm_ndcg=0.2965, mm_r10=34.6, text=0.1161, image=0.1078, gap=60.84, coverage=100.0),
}


def api(path: str, attempts: int = 40):
    """GET the backend, retrying while the stack is being restarted under us.

    Another worker owns this repo and restarts uvicorn/vite at will; a connection refusal here
    is that restart, not a QA finding. Waits long enough to outlast a rebuild, then raises."""
    last = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(f"{API}{path}", timeout=30) as resp:
                return json.load(resp)
        except Exception as exc:  # URLError, JSONDecodeError from a half-written response
            last = exc
            time.sleep(min(1.0 + attempt * 0.5, 5.0))
    raise RuntimeError(f"backend unavailable for {path}: {last}")


def wait_for_stack(timeout_s: float = 180) -> None:
    """Block until both ports answer 200 for the two endpoints every flow needs."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            api("/api/benchmark-runs", attempts=1)
            api("/api/composition/runs?subset=full", attempts=1)
            with urllib.request.urlopen(f"{UI}/?page=Composition", timeout=10) as resp:
                if resp.status == 200:
                    return
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError(f"stack not ready after {timeout_s}s")


def check(checks: list, label, ok, detail="", defect=False) -> bool:
    """`defect=True` marks an assertion known to fail because of a product defect rather than
    harness drift. It still fails -- the flag only keeps the report honest about the cause."""
    checks.append({"label": label, "ok": bool(ok), "detail": detail, "defect": bool(defect)})
    return bool(ok)


# --------------------------------------------------------------------- DOM readers

READ_TABLE = """
() => {
  const table = document.querySelector('.composition-table');
  if (!table) return null;
  const rows = [...table.querySelectorAll('tbody tr')].map((tr) => {
    const cells = [...tr.children].map((td) => td.innerText.replace(/\\s+/g, ' ').trim());
    return {
      cells,
      fixture: !!tr.querySelector('.badge-warning'),
      real: !tr.querySelector('.badge-warning'),
      selected: tr.classList.contains('selected-row'),
      cellCount: cells.length,
    };
  });
  return {headers: [...table.querySelectorAll('thead th')].map((th) => th.innerText.trim()), rows};
}
"""

READ_CARD_SUMMARY = """
() => {
  const cards = [...document.querySelectorAll('.benchmark-run-card')];
  return cards.map((card) => ({
    title: card.querySelector('h3')?.innerText.trim() || '',
    meta: card.querySelector('.run-meta')?.innerText.replace(/\\s+/g, ' ').trim() || '',
    badges: [...card.querySelectorAll('.run-badges .badge')].map((b) => b.innerText.trim()),
    rows: [...card.querySelectorAll('table.benchmark-table tr')].map((tr) =>
      [...tr.children].map((c) => ({
        text: c.innerText.replace(/\\s+/g, ' ').trim(),
        muted: c.classList.contains('text-muted'),
      }))),
  }));
}
"""

READ_TARGETS = """
() => {
  const sel = '.subset-selector button, nav button, .drilldown-filters select, .pager-buttons button';
  return [...document.querySelectorAll(sel)].map((el) => {
    const r = el.getBoundingClientRect();
    return {text: (el.innerText || el.value || '').replace(/\\s+/g, ' ').trim().slice(0, 40),
            w: Math.round(r.width), h: Math.round(r.height),
            left: Math.round(r.left), right: Math.round(r.right)};
  });
}
"""


def subset_json(subset: str) -> dict:
    return {r["run_id"]: r for r in api(f"/api/composition/runs?subset={subset}")}


def expected_cells(run: dict) -> dict:
    """The exact strings CompositionView.tsx renders, derived from the API row. Null metrics
    render as an em-dash there (`formatNdcg`/`formatRecall`/`formatPercent`), never as 0."""
    m = run["metrics"]

    def ndcg(v):
        return "—" if v is None else f"{v:.4f}"

    def pct(v, places):
        return "—" if v is None else f"{v:.{places}f}%"

    gap = m.get("composition_gap_ndcg")
    return {
        "mm_ndcg": ndcg(m.get("ndcg_multimodal")),
        "mm_r10": pct(m.get("recall_at_10_multimodal"), 1),
        "text": ndcg(m.get("ndcg_text_only")),
        "image": ndcg(m.get("ndcg_image_only")),
        "gap": "—" if gap is None else ("+" if gap > 0 else "") + f"{gap * 100:.2f}%",
        "coverage": pct(None if m.get("coverage") is None else m["coverage"] * 100, 1),
    }


def expected_row(run: dict) -> list[str]:
    """The six numeric cells CompositionView.tsx renders for a run, in column order:
    MM nDCG | MM R@10 | Text nDCG | Image nDCG | Comp Gap | Coverage."""
    e = expected_cells(run)
    return [e["mm_ndcg"], e["mm_r10"], e["text"], e["image"], e["gap"], e["coverage"]]


def brief_row(subset: str) -> list[str]:
    """The same six cells, formatted exactly as the brief's table states them."""
    b = BRIEF[subset]
    return [f"{b['mm_ndcg']:.4f}", f"{b['mm_r10']:.1f}%", f"{b['text']:.4f}",
            f"{b['image']:.4f}", f"+{b['gap']:.2f}%", f"{b['coverage']:.1f}%"]


def table_matches(table: dict, subset: str) -> tuple[bool, str]:
    """Every rendered row must equal one API row for that subset, and every API row must be
    rendered. Rows are matched on their exact six numeric cells plus the FIXTURE badge (two
    real runs share model_id `encoder`, so the name cannot identify a row)."""
    runs = subset_json(subset)
    problems = []
    unused = {rid: (run, expected_row(run)) for rid, run in runs.items()}
    for row in table["rows"]:
        got = row["cells"][3:9]
        hit = next((rid for rid, (run, want) in unused.items()
                    if want == got and bool(run.get("is_synthetic")) == row["fixture"]), None)
        if hit is None:
            problems.append(f"row {row['cells'][0]!r} cells {got} match no API run (fixture={row['fixture']})")
            continue
        unused.pop(hit)
        if row["cellCount"] != 10:
            problems.append(f"{hit}: {row['cellCount']} cells, expected 10")
    if unused:
        problems.append(f"API runs not rendered: {sorted(unused)}")
    return (not problems), "; ".join(problems)


def settle(page, ms=1200):
    page.wait_for_timeout(ms)


def shot(page, out: Path, name: str) -> dict:
    path = out / "screens" / f"{name}.png"
    page.screenshot(path=str(path), full_page=False)
    full = out / "screens" / f"{name}_full.png"
    page.screenshot(path=str(full), full_page=True)
    return {"viewport": str(path), "fullPage": str(full),
            "viewportSize": hq.png_size(path), "fullPageSize": hq.png_size(full)}


def run_flow(browser, vp, out: Path, flow: str, url: str, body):
    width, height, name = vp
    ctx = browser.new_context(viewport={"width": width, "height": height})
    page = ctx.new_page()
    rec = hq.attach_recorders(page)
    checks: list = []
    try:
        page.goto(url, wait_until="load", timeout=60_000)
        try:
            page.wait_for_load_state("networkidle", timeout=15_000)
        except Exception:
            pass
        extra = body(page, checks) or {}
        shots = extra.pop("screens", {})
        result = {"flow": flow, "viewport": name, "url": url, "checks": checks,
                  "pass": all(c["ok"] for c in checks), "screens": shots, **extra}
    except Exception as exc:  # a flow that cannot even run is a failure, never a skip
        result = {"flow": flow, "viewport": name, "url": url, "checks": checks,
                  "pass": False, "fatal": f"{type(exc).__name__}: {exc}"[:600], "screens": {}}
    result.update({
        "consoleErrors": rec["console"], "pageErrors": rec["pageErrors"],
        "failedRequests": rec["failedRequests"], "httpErrors": rec["httpErrors"],
        "consoleErrorCount": rec["console_errors"], "pageErrorCount": len(rec["pageErrors"]),
        "failedRequestCount": len(rec["failedRequests"]), "httpErrorCount": len(rec["httpErrors"]),
    })
    ctx.close()
    return result


# ------------------------------------------------------------------------ flows

def flow_benchmark(page, checks):
    shots = {"benchmark": shot(page, OUT, f"benchmark_{VP[2]}")}
    runs = api("/api/benchmark-runs")
    check(checks, "api: exactly 2 aggregate benchmark runs", len(runs) == 2, f"{len(runs)} runs")
    cards = page.evaluate(READ_CARD_SUMMARY)
    check(checks, "2 benchmark cards rendered", len(cards) == len(runs), f"{len(cards)} cards")
    for card, run in zip(cards, runs):
        tag = run["run_id"][:24]
        check(checks, f"{tag}: model/checkpoint in title",
              run["model_id"].upper() in card["title"] and run["checkpoint_id"] in card["title"], card["title"])
        check(checks, f"{tag}: run id + protocol in meta",
              run["run_id"] in card["meta"] and run["protocol_id"] in card["meta"], card["meta"])
        check(checks, f"{tag}: CERTIFIED badge", "CERTIFIED" in card["badges"], str(card["badges"]))
        check(checks, f"{tag}: aggregate-only badge states CAPABILITY_ABSENT",
              any("aggregate-only" in b and "CAPABILITY_ABSENT" in b for b in card["badges"]), str(card["badges"]))
        macro = [r for r in card["rows"] if r and r[0]["text"].startswith("Recall@")]
        measured = {r[0]["text"]: r[1]["text"] for r in macro}
        paper = {r[0]["text"]: r[2]["text"] for r in macro}
        obs, ref = run["observed_metrics"], run["paper_sanity_metrics"]
        check(checks, f"{tag}: measured R@10 = {obs['r10']}",
              measured.get("Recall@10", "").startswith(f"{obs['r10']:.2f}%"), measured.get("Recall@10", ""))
        check(checks, f"{tag}: measured R@50 = {obs['r50']}",
              measured.get("Recall@50", "").startswith(f"{obs['r50']:.2f}%"), measured.get("Recall@50", ""))
        check(checks, f"{tag}: paper ref R@10 = {ref['r10']}",
              paper.get("Recall@10", "").startswith(f"{ref['r10']:.2f}%"), paper.get("Recall@10", ""))
        check(checks, f"{tag}: measured != paper cells (distinct values)",
              measured.get("Recall@10") != paper.get("Recall@10"),
              f"measured={measured.get('Recall@10')} paper={paper.get('Recall@10')}")
        check(checks, f"{tag}: paper cells carry muted styling",
              all(r[2]["muted"] and not r[1]["muted"] for r in macro), "text-muted on paper column only")
        cats = {r[0]["text"].lower(): r for r in card["rows"]
                if r and r[0]["text"].lower() in ("dress", "shirt", "toptee")}
        check(checks, f"{tag}: per-category rows dress/shirt/toptee",
              set(cats) == {"dress", "shirt", "toptee"}, str(sorted(cats)))
        for cat, catrow in cats.items():
            api_cat = run["category_metrics"][cat]
            check(checks, f"{tag}/{cat}: measured R@10 {api_cat['r10']:.4f} rendered",
                  catrow[1]["text"].startswith(f"{api_cat['r10']:.2f}%"), catrow[1]["text"])

    # per-query actions: disabled with a stated reason
    page.goto(f"{UI}/?page=Evaluation%20Runs", wait_until="load")
    settle(page, 800)
    disabled = page.evaluate("""
    () => [...document.querySelectorAll('label.aggregate-only')].map((l) => {
      const box = l.querySelector('input[type=checkbox]');
      return {disabled: !!box?.disabled, title: box?.title || '',
              text: l.innerText.replace(/\\s+/g, ' ').trim().slice(0, 400),
              badge: [...l.querySelectorAll('.badge')].map((b) => b.innerText.trim())};
    })
    """)
    shots["evaluation_runs"] = shot(page, OUT, f"evaluation_runs_{VP[2]}")
    check(checks, "evaluation runs: 2 aggregate-only rows", len(disabled) == 2, f"{len(disabled)}")
    for item in disabled:
        check(checks, "per-query action disabled", item["disabled"], item["text"][:60])
        check(checks, f"disabled reason stated ({item['title'][:40]!r})",
              "aggregate-only" in item["title"] and "CAPABILITY_ABSENT" in item["title"], item["title"])
        check(checks, "aggregate-only badge visible on row",
              any("CAPABILITY_ABSENT" in b for b in item["badge"]), str(item["badge"]))
        check(checks, "SYNTHETIC SOURCE not claimed for real run", "SYNTHETIC SOURCE" not in item["text"], "")

    # Dashboard model count is a real number. `/api/models` is fetched after first paint, so read
    # the first-paint value for the record, then REQUIRE the card to resolve within 3s: an em-dash
    # that never clears is the defect the mission names, a sub-second placeholder is a separate
    # (cosmetic) finding.
    page.goto(f"{UI}/?page=Dashboard", wait_until="load")
    first_paint = page.evaluate("() => document.querySelector('.cards b')?.innerText.trim() || ''")
    try:
        page.wait_for_function(
            "() => !/^\u2014/.test(document.querySelector('.cards b')?.innerText.trim() || '\u2014')",
            timeout=3000)
    except Exception:
        pass
    settle(page, 150)
    cards_txt = page.evaluate("() => [...document.querySelectorAll('.cards b')].map(b => b.innerText.trim())")
    models = api("/api/models")
    shots["dashboard"] = shot(page, OUT, f"dashboard_{VP[2]}")
    check(checks, "dashboard: model count is a number, not an em-dash",
          bool(cards_txt) and "—" not in cards_txt[0], cards_txt[0] if cards_txt else "")
    # The requirement is that the card shows a real number. It does, once `/api/models` lands.
    # The loading placeholder is an em-dash -- the same glyph a genuinely absent value would use --
    # so it is RECORDED as an observation (see README defects) rather than gated: a 300ms loading
    # glyph is a cosmetic finding, while a count that never resolves is the failure the mission
    # names, and that is what the check above tests.
    check(checks, f"dashboard: model count == /api/models ({len(models)})",
          cards_txt and cards_txt[0].startswith(str(len(models))), cards_txt[0] if cards_txt else "")
    check(checks, "dashboard: certified aggregate count present",
          any("certified research results" in c for c in cards_txt), str(cards_txt))

    # sample explorer states the aggregate-only scope
    page.goto(f"{UI}/?page=Sample%20Explorer", wait_until="load")
    settle(page, 800)
    notice = page.evaluate("() => document.querySelector('.notice-aggregate-only')?.innerText || ''")
    shots["sample_explorer"] = shot(page, OUT, f"sample_explorer_{VP[2]}")
    check(checks, "sample explorer: explicit aggregate-only notice",
          "CAPABILITY_ABSENT" in notice and "cannot be displayed" in notice, notice[:120].replace("\n", " "))
    return {"screens": shots, "dashboardFirstPaint": first_paint}


def flow_composition_switch(page, checks):
    page.goto(f"{UI}/?page=Composition", wait_until="load")
    settle(page, 900)
    shots = {}
    seen_any_real = False
    for subset, label in (("full", "Full"), ("sf", "Shortcut-Free (SF)"), ("validated", "Validated")):
        page.get_by_role("tab", name=re.compile(label.replace("(", r"\(").replace(")", r"\)"))).click()
        settle(page, 900)
        table = page.evaluate(READ_TABLE)
        shots[subset] = shot(page, OUT, f"composition_{subset}_{VP[2]}")
        ok, detail = table_matches(table, subset)
        check(checks, f"{subset}: every rendered row equals the subset's API JSON", ok, detail)
        active = page.evaluate("() => document.querySelector('.subset-pill.active')?.innerText || ''")
        check(checks, f"{subset}: pill active and labelled", label.split("(")[0].strip() in active, active[:60])
        real_pool = {k: v for k, v in subset_json(subset).items() if not v.get("is_synthetic")}
        real_encoder = real_pool.get(REAL_RUN)
        if real_encoder:
            seen_any_real = True
            brief = BRIEF[subset]
            m = real_encoder["metrics"]
            check(checks, f"{subset}: API matches brief (queries={brief['queries']})",
                  real_encoder["query_count"] == brief["queries"], str(real_encoder["query_count"]))
            # Compare at the precision the review brief states and the UI renders.
            for field, api_val, brief_val, places in (
                ("mm_ndcg", m["ndcg_multimodal"], brief["mm_ndcg"], 4),
                ("mm_r10", m["recall_at_10_multimodal"], brief["mm_r10"], 1),
                ("text", m["ndcg_text_only"], brief["text"], 4),
                ("image", m["ndcg_image_only"], brief["image"], 4),
                ("gap", m["composition_gap_ndcg"] * 100, brief["gap"], 2),
                ("coverage", m["coverage"] * 100, brief["coverage"], 1),
            ):
                check(checks, f"{subset}: API {field} == brief {brief_val}",
                      round(api_val, places) == brief_val, f"api={round(api_val, places)}")
            want_row = brief_row(subset)
            dom_row = [r for r in table["rows"] if r["cells"][3:9] == want_row]
            check(checks, f"{subset}: DOM shows the brief's exact six values for the encoder run",
                  len(dom_row) == 1, f"wanted {want_row}, DOM had {[r['cells'][3:9] for r in table['rows']]}")
            # Default path is measurements only, by design: the backend defaults
            # include_fixtures=false so a screenshot of "real composition results" can never open
            # with invented rows. Assert THAT here, not the presence of fixtures.
            fixtures = [r for r in table["rows"] if r["fixture"]]
            reals = [r for r in table["rows"] if r["real"]]
            default_runs = subset_json(subset)
            check(checks, f"{subset}: default list renders exactly the measured runs",
                  len(table["rows"]) == len(default_runs),
                  f"{len(table['rows'])} rows vs {len(default_runs)} API runs")
            check(checks, f"{subset}: no FIXTURE badge and no SYNTHETIC_ text on the default path",
                  not fixtures and not any("synthetic" in c.lower() for r in table["rows"] for c in r["cells"]),
                  str([r["cells"][0] for r in table["rows"]]))
            check(checks, f"{subset}: real encoder row rendered as measured",
                  any("encoder" in r["cells"][0].lower() for r in reals),
                  f"{len(reals)} real rows: {[r['cells'][0] for r in reals]}")
    check(checks, "real encoder row present across subsets", seen_any_real)

    # The two SYNTHETIC_* fixture summaries exist on disk and the API serves them on request,
    # but the view's own fetch omits `include_fixtures` and the backend now defaults it to
    # false. Prove the defect is the missing parameter and not the renderer: replay the exact
    # request with the parameter added and check the FIXTURE label.
    api_default = api("/api/composition/runs?subset=full")
    api_optin = api("/api/composition/runs?subset=full&include_fixtures=true&include_smoke=true")
    check(checks, "api: default list is measurements only",
          not any(r.get("is_synthetic") for r in api_default), f"{len(api_default)} runs")
    check(checks, "api: fixtures available via include_fixtures/include_smoke opt-in",
          sum(1 for r in api_optin if r.get("is_synthetic")) == 2, f"{len(api_optin)} runs")
    injected: list[str] = []

    def opt_in(route):
        url = route.request.url
        if "/api/composition/runs?" in url and "include_fixtures" not in url:
            url += "&include_fixtures=true&include_smoke=true"
            injected.append(url)
        route.continue_(url=url)

    page.route("**/api/composition/runs?**", opt_in)
    page.goto(f"{UI}/?page=Composition", wait_until="load")
    settle(page, 1200)
    with_fixtures = page.evaluate(READ_TABLE)
    page.unroute("**/api/composition/runs?**")
    shots["fixtures_injected"] = shot(page, OUT, f"composition_fixtures_injected_{VP[2]}")
    fixture_rows = [r for r in with_fixtures["rows"] if r["fixture"]]
    real_rows = [r for r in with_fixtures["rows"] if r["real"]]
    check(checks, "opt-in: request carried include_fixtures", bool(injected), str(injected[:1]))
    check(checks, "opt-in: two SYNTHETIC_* fixture rows appear and are labelled FIXTURE",
          len(fixture_rows) == 2, f"{len(fixture_rows)} labelled rows of {len(with_fixtures['rows'])}")
    check(checks, "opt-in: fixture rows are named SYNTHETIC_*",
          all("fixture" in r["cells"][0].lower() for r in fixture_rows),
          str([r["cells"][0] for r in fixture_rows]))
    check(checks, "opt-in: fixture rows carry different numbers than the measured rows",
          all(r["cells"][3:9] not in [x["cells"][3:9] for x in real_rows] for r in fixture_rows),
          str([r["cells"][3:9] for r in fixture_rows]))
    # The whole point of the default: measured results sort above invented ones.
    first = with_fixtures["rows"][0] if with_fixtures["rows"] else None
    check(checks, "opt-in: the real encoder run sorts FIRST, above the fixtures",
          first is not None and not first["fixture"] and "encoder" in first["cells"][0].lower(),
          str(first["cells"][0]) if first else "no rows")
    real_idx = [i for i, r in enumerate(with_fixtures["rows"]) if not r["fixture"]]
    fix_idx = [i for i, r in enumerate(with_fixtures["rows"]) if r["fixture"]]
    check(checks, "opt-in: every measured row precedes every fixture row",
          bool(real_idx) and bool(fix_idx) and max(real_idx) < min(fix_idx),
          f"real at {real_idx}, fixture at {fix_idx}")
    return {"screens": shots}


LOAD_ALL_IMAGES = """
async () => {
  const imgs = [...document.querySelectorAll('.query-card img')];
  const outp = [];
  for (const img of imgs) {
    if (!img.complete || img.naturalWidth === 0) {
      img.scrollIntoView({block: 'center'});
      await new Promise((resolve) => {
        let done = false;
        const fin = () => { if (!done) { done = true; resolve(); } };
        img.addEventListener('load', fin, {once: true});
        img.addEventListener('error', fin, {once: true});
        setTimeout(fin, 8000);
      });
    }
    outp.push({alt: img.alt, naturalWidth: img.naturalWidth, complete: img.complete,
               src: (img.currentSrc || img.src).slice(0, 160)});
  }
  return outp;
}
"""


def flow_drilldown(page, checks):
    page.goto(f"{UI}/?page=Composition&comp_run={REAL_RUN}&comp_subset=full&comp_cond=multimodal",
              wait_until="load")
    settle(page, 1200)
    page.evaluate("() => document.querySelector('.drilldown-section')?.scrollIntoView()")
    settle(page, 400)
    imgs = page.evaluate(LOAD_ALL_IMAGES)
    page.evaluate("() => document.querySelector('.drilldown-section')?.scrollIntoView()")
    settle(page, 300)
    shots = {"drilldown": shot(page, OUT, f"drilldown_{VP[2]}")}
    check(checks, "drilldown: images present", len(imgs) > 0, f"{len(imgs)} imgs")
    check(checks, "drilldown: every query image decoded (naturalWidth > 0)",
          imgs and all(i["naturalWidth"] > 0 for i in imgs),
          str([(i["alt"], i["naturalWidth"], i["src"][-60:]) for i in imgs if i["naturalWidth"] == 0][:5]))
    kinds = {i["alt"].split(":")[0] for i in imgs}
    check(checks, "drilldown: reference and target images both load",
          "Reference" in kinds and "Target" in kinds, str(sorted(kinds)))
    check(checks, "drilldown: no image fell back to the missing placeholder",
          page.evaluate("() => document.querySelectorAll('.image-figure.image-missing').length") == 0,
          f"{page.evaluate('() => document.querySelectorAll(\".image-figure.image-missing\").length')} placeholders")
    txt = page.evaluate("() => document.querySelector('.drilldown-section')?.innerText || ''")
    check(checks, "drilldown: edit text rendered", "Edit text:" in txt, "")
    check(checks, "drilldown: run id tag shows real run", REAL_RUN in txt, "")
    check(checks, "drilldown: pagination total == API total",
          "of 6,003 queries" in txt, txt[txt.find("Showing"):txt.find("Showing") + 40])

    # A rank that does not exist must render an explicit state, never 0. The sparse real run is
    # a SMOKE run, which the backend's default list now omits; opt in for this page only so the
    # view can select it. The missing-rank renderer is what is under test here.
    page.route("**/api/composition/runs?**", lambda route: route.continue_(
        url=route.request.url + ("&" if "?" in route.request.url else "?") + "include_smoke=true&include_fixtures=true"
        if "include_smoke" not in route.request.url else route.request.url))
    page.goto(f"{UI}/?page=Composition&comp_run={SPARSE_RUN}&comp_subset=full&comp_order=worst",
              wait_until="load")
    settle(page, 1800)
    page.evaluate("() => document.querySelector('.drilldown-section')?.scrollIntoView()")
    settle(page, 400)
    shots["missing_rank"] = shot(page, OUT, f"drilldown_missing_rank_{VP[2]}")
    ranks = page.evaluate("""
    () => [...document.querySelectorAll('.query-card')].map((card) => ({
      id: card.querySelector('.query-id')?.innerText.trim() || '',
      values: [...card.querySelectorAll('.rank-value')].map((v) => v.innerText.trim()),
      unscored: card.querySelectorAll('.rank-value .unscored').length,
    }))
    """)
    shown_run = page.evaluate("() => document.querySelector('.run-id-tag')?.innerText || ''")
    check(checks, "missing rank: sparse run actually selected (not silently swapped)",
          SPARSE_RUN in shown_run, shown_run, defect=True)
    check(checks, "missing rank: cards rendered for sparse run", len(ranks) > 0, f"{len(ranks)} cards")
    check(checks, "missing rank: unscored state rendered explicitly",
          any(r["unscored"] > 0 for r in ranks),
          str([(r["id"], r["values"]) for r in ranks[:2]]))
    zero_like = [r for r in ranks for v in r["values"] if v in ("#0", "0", "0.0000")]
    check(checks, "missing rank: never rendered as 0 / #0", not zero_like, str(zero_like[:3]))
    # API agrees these queries have no rank in this run
    q = api(f"/api/composition/runs/{SPARSE_RUN}/queries?subset=full&order=worst&limit=5")
    check(checks, "missing rank: API reports null target_rank for these queries",
          all(item["target_rank"] is None for item in q["items"]),
          str([item["target_rank"] for item in q["items"]]))
    page.unroute("**/api/composition/runs?**")
    return {"screens": shots, "sparseTotal": q["total"]}


def flow_rapid_switching(page, checks):
    page.goto(f"{UI}/?page=Composition", wait_until="load")
    settle(page, 900)
    bursts = [
        ([0, 1, 2], "validated"),
        ([2, 0, 1], "sf"),
        ([0, 2, 1, 0], "full"),
        ([1, 2, 0, 1, 2], "validated"),
    ]
    shots = {}
    for n, (sequence, final) in enumerate(bursts):
        page.evaluate(
            "(seq) => { const b = [...document.querySelectorAll('.subset-pill')]; seq.forEach((i) => b[i].click()); }",
            sequence)
        settle(page, 1500)
        table = page.evaluate(READ_TABLE)
        label = (table["rows"] and page.evaluate("() => document.querySelector('.subset-pill.active')?.innerText || ''")) or ""
        ok, detail = table_matches(table, final)
        check(checks, f"burst {n} {sequence} -> {final}: rendered data == {final} JSON", ok, detail)
        check(checks, f"burst {n}: active pill is {final}", BRIEF[final] and
              {"full": "Full", "sf": "Shortcut-Free", "validated": "Validated"}[final] in label, label[:40])
        shots[f"rapid_{n}"] = shot(page, OUT, f"rapid_{n}_{VP[2]}")
    return {"screens": shots}


def flow_mobile(page, checks):
    """Viewport-specific: whole-page overflow, table-only scrolling, tap targets, clipping.

    Also asserts the table keeps row identity when it is scrolled horizontally: with a 1193px
    table in a 366px wrapper the Model column leaves the screen first, so the numeric cells a
    reviewer is comparing lose their row label."""
    page.goto(f"{UI}/?page=Composition&comp_run={REAL_RUN}&comp_subset=full", wait_until="load")
    settle(page, 1200)
    metrics = page.evaluate("""
    () => {
      const vw = window.innerWidth;
      const table = document.querySelector('.composition-table');
      const wrap = table?.closest('.table-container');
      const nav = document.querySelector('nav');
      const sections = [...document.querySelectorAll('section')].map((s) => ({
        cls: s.className, scrollWidth: s.scrollWidth, clientWidth: s.clientWidth,
      }));
      return {
        innerWidth: vw,
        docScrollWidth: document.documentElement.scrollWidth,
        bodyScrollWidth: document.body.scrollWidth,
        tableScrolls: wrap ? wrap.scrollWidth > wrap.clientWidth + 1 : null,
        wrapOverflowX: wrap ? getComputedStyle(wrap).overflowX : null,
        wrapClientWidth: wrap ? wrap.clientWidth : null,
        tableWidth: table ? Math.round(table.getBoundingClientRect().width) : null,
        navScrollWidth: nav ? nav.scrollWidth : null,
        navClientWidth: nav ? nav.clientWidth : null,
        sections,
      };
    }
    """)
    shots = {"mobile": shot(page, OUT, f"mobile_composition_{VP[2]}")}
    check(checks, "no whole-page horizontal overflow",
          metrics["docScrollWidth"] <= metrics["innerWidth"] + 1
          and metrics["bodyScrollWidth"] <= metrics["innerWidth"] + 1,
          f"doc={metrics['docScrollWidth']} body={metrics['bodyScrollWidth']} vw={metrics['innerWidth']}")
    needs_scroll = metrics["tableWidth"] is not None and metrics["wrapClientWidth"] is not None \
        and metrics["tableWidth"] > metrics["wrapClientWidth"]
    check(checks, "wide table scrolls inside its wrapper (never the page)",
          (not needs_scroll) or metrics["tableScrolls"] is True,
          f"table={metrics['tableWidth']} wrapper={metrics['wrapClientWidth']} scrolls={metrics['tableScrolls']}")
    check(checks, "table overflow contained by an overflow-x:auto wrapper",
          metrics["wrapOverflowX"] in ("auto", "scroll"), str(metrics["wrapOverflowX"]))

    wide = page.evaluate(hq.JS_PROBE)["wideElements"]
    outside = [w for w in wide if "table-container" not in w["selector"] and "composition-table" not in w["selector"]]
    check(checks, "no elements escape the viewport outside the table wrapper", not outside,
          str([(w["selector"], w["width"]) for w in outside[:4]]))

    targets = page.evaluate(READ_TARGETS)
    # WCAG 2.5.8 (AA) needs 24x24 CSS px; 44x44 is the comfortable-mobile recommendation.
    small = [t for t in targets if t["h"] < 24 and t["w"] > 0]
    clipped = [t for t in targets if t["w"] > 0 and (t["right"] > metrics["innerWidth"] + 1 or t["left"] < -1)]
    check(checks, "controls stay inside the viewport", not clipped, str(clipped[:4]))
    check(checks, "tap targets meet WCAG 2.5.8 (>= 24px tall)", not small,
          str([(t["text"], t["h"]) for t in small[:6]]))

    # Row identity while scrolled: the Model column must not scroll out of view.
    scroll_probe = page.evaluate("""() => {
      const wrap = document.querySelector('.composition-table')?.closest('.table-container');
      if (!wrap) return null;
      const before = [...document.querySelectorAll('.composition-table tbody tr')]
        .map((tr) => tr.firstElementChild.getBoundingClientRect().left);
      wrap.scrollLeft = wrap.scrollWidth;
      const after = [...document.querySelectorAll('.composition-table tbody tr')]
        .map((tr) => tr.firstElementChild.getBoundingClientRect().left);
      const wrapLeft = wrap.getBoundingClientRect().left;
      const visibleAfter = after.every((x) => x >= wrapLeft - 1);
      return {before: before.map(Math.round), after: after.map(Math.round),
              wrapLeft: Math.round(wrapLeft), visibleAfter,
              scrollLeft: wrap.scrollLeft, maxScroll: wrap.scrollWidth - wrap.clientWidth};
    }""")
    check(checks, "row identity survives horizontal scroll (Model column pinned or in view)",
          scroll_probe is not None and scroll_probe["visibleAfter"], str(scroll_probe), defect=True)

    # nav row must not be clipped at the left/right edge
    check(checks, "nav row not clipped", metrics["navScrollWidth"] is not None
          and metrics["navScrollWidth"] <= metrics["navClientWidth"] + 2,
          f"nav scrollWidth={metrics['navScrollWidth']} clientWidth={metrics['navClientWidth']}")
    return {"screens": shots, "mobileMetrics": metrics, "tapTargets": targets}


FLOWS = {
    "benchmark_native": (f"{UI}/?page=Benchmark", flow_benchmark),
    "composition_switch": (f"{UI}/?page=Composition", flow_composition_switch),
    "composition_drilldown": (f"{UI}/?page=Composition&comp_run={REAL_RUN}", flow_drilldown),
    "rapid_filter_switching": (f"{UI}/?page=Composition", flow_rapid_switching),
    "mobile_layout": (f"{UI}/?page=Composition&comp_subset=full", flow_mobile),
}


def main() -> int:
    global OUT, VP
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--only", default="", help="comma-separated flow names")
    parser.add_argument("--viewport", action="append", default=None, help="WxH, repeatable")
    args = parser.parse_args()
    OUT = args.out
    (OUT / "screens").mkdir(parents=True, exist_ok=True)
    if not os.environ.get("PLAYWRIGHT_BROWSERS_PATH"):
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(hq.DEFAULT_BROWSERS)

    vps = VIEWPORTS
    if args.viewport:
        wanted = set(args.viewport)
        vps = [v for v in VIEWPORTS if v[2] in wanted]
    names = [n for n in (args.only.split(",") if args.only else FLOWS) if n in FLOWS]

    wait_for_stack()
    from playwright.sync_api import sync_playwright

    results = []
    with sync_playwright() as playwright:
        browser = hq.launch(playwright)
        try:
            for vp in vps:
                VP = vp
                for name in names:
                    url, fn = FLOWS[name]
                    if results:  # first flow already proved the stack; later restarts need a recheck
                        wait_for_stack()
                    res = run_flow(browser, vp, OUT, name, url, fn)
                    results.append(res)
                    print(f"{vp[2]:>9} {name:<24} {'PASS' if res['pass'] else 'FAIL'} "
                          f"checks={sum(c['ok'] for c in res['checks'])}/{len(res['checks'])} "
                          f"console={res['consoleErrorCount']} page={res['pageErrorCount']} "
                          f"req={res['failedRequestCount']} http={res['httpErrorCount']}")
                    for c in res["checks"]:
                        if not c["ok"]:
                            print(f"    FAIL {c['label']} :: {c['detail']}")
        finally:
            browser.close()

    summary = {
        "ui": UI,
        "api": API,
        "viewports": [v[2] for v in vps],
        "flowCount": len(results),
        "ok": all(r["pass"] for r in results),
        "flows": results,
    }
    (OUT / "flows.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nflows: {sum(1 for r in results if r['pass'])}/{len(results)} pass -> {OUT / 'flows.json'}")
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
