# Browser QA — real stack, real data

Bundle: `workbench/docs/reviews/20261108T015603+0700/browser_qa/`
Run date: 2026-11-08 (+0700). Branch `research/cir-failure-workbench`.

## Stack

URL: **http://127.0.0.1:5173** (production frontend via `vite preview`) with the backend proxied at
**http://127.0.0.1:8000**.

```bash
cd /home/heheboiz/data/cir
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release \
workbench/envs/ctl312h/bin/python workbench/scripts/serve_workbench.py --production-frontend \
  --host 127.0.0.1 --backend-port 8000 --frontend-port 5173

# after any frontend source change (dist is a build artifact, gitignored):
npm run build --prefix workbench/frontend
```

`vite` proxies `/api` to `WORKBENCH_BACKEND_URL`, which only `serve_workbench.py` sets. Running
`vite preview` alone makes `/api` 404 — that is the proxy, not an app bug.

## Pre-flight API confirmation (both required, both confirmed)

```bash
curl -s http://127.0.0.1:8000/api/benchmark-runs | jq length          # -> 2
curl -s "http://127.0.0.1:8000/api/composition/runs?subset=full" | jq '.[0].metrics'
```

- `/api/benchmark-runs` → **2 CSMCIR runs**, both `verification.state = certified`,
  `certified: true`, `parity_status: PARITY_VERIFIED`, measured R@10 57.0507 / R@50 77.2986.
- `/api/composition/runs?subset=full` → includes real encoder run **`6a528e877c328772e2c7bb28`**
  with populated metrics; all six values match the review brief at the precision it states.

`/api/runs` is `[]` (no schema-v2 per-query artifacts). That is correct, not a defect.

## Commands

```bash
# Reused project harness: screenshot + console/page/request/overflow/image report per viewport.
PLAYWRIGHT_BROWSERS_PATH=workbench/envs/ctl312h/.playwright-browsers \
workbench/envs/ctl312h/bin/python workbench/frontend/tests/browser_qa.py \
  --base-url "http://127.0.0.1:5173/?page=Benchmark" \
  --viewport 1440x900 --viewport 768x1024 --viewport 390x844 \
  --out <bundle>/browser_qa/harness_benchmark

# Interactive flows: subset switching, drill-down, missing rank, rapid switching, mobile layout.
PLAYWRIGHT_BROWSERS_PATH=workbench/envs/ctl312h/.playwright-browsers \
workbench/envs/ctl312h/bin/python <bundle>/browser_qa/flows.py --out <bundle>/browser_qa

# Element-level 2x-DSF crops for pixel inspection.
PLAYWRIGHT_BROWSERS_PATH=workbench/envs/ctl312h/.playwright-browsers \
workbench/envs/ctl312h/bin/python <bundle>/browser_qa/zooms.py --out <bundle>/browser_qa
```

Both scripts reuse `browser_qa.py` for launch, recorders and PNG sizing — one convention, no fork.

## Result: 13/15 flows pass

`flows.json` is the raw output. Per viewport, per flow:

| viewport | benchmark_native | composition_switch | composition_drilldown | rapid_filter_switching | mobile_layout |
|---|---|---|---|---|---|
| 1440x900 | PASS 41/41 | PASS 48/48 | PASS 12/12 | PASS 8/8 | PASS 8/8 |
| 768x1024 | PASS 41/41 | PASS 48/48 | PASS 12/12 | PASS 8/8 | **FAIL 7/8** |
| 390x844 | PASS 41/41 | PASS 48/48 | PASS 12/12 | PASS 8/8 | **FAIL 7/8** |

Console errors **0**, page errors **0**, failed requests **0**, HTTP ≥400 **0** in every flow at
every viewport.

### Harness sweeps (`harness_*.json`), all `ok: true`

| page | nodes | overflow px | console/page/req/http | images | broken |
|---|---|---|---|---|---|
| Benchmark | 405 | 0 / 0 / 0 | 0/0/0/0 | 0 | 0 |
| Composition | 756 | 0 / 0 / 0 | 0/0/0/0 | 40 | 0 |
| Drill-down (validated) | 756 | 0 / 0 / 0 | 0/0/0/0 | 40 | 0 |
| Dashboard | 106 | 0 / 0 / 0 | 0/0/0/0 | 0 | 0 |
| Evaluation Runs | 278 | 0 / 0 / 0 | 0/0/0/0 | 0 | 0 |

Whole-page horizontal overflow is 0 at all three viewports. Wide-element counts above 0 on narrow
viewports are the data tables *inside* their `overflow-x: auto` wrappers — contained, not page
overflow.

## Verified against real data

- **Benchmark**: the 2 real CSMCIR aggregate runs. Model/checkpoint/protocol, run id, CERTIFIED
  badge, aggregate-only + `CAPABILITY_ABSENT` marker, measured R@10/R@50/mean, paper-sanity column
  visually distinct (`text-muted`, verified via computed class, not eyeballing), per-category
  dress/shirt/toptee measured values matching `category_metrics` to 2dp.
- **Evaluation Runs**: per-query checkbox `disabled` with `title` stating
  "Per-query actions disabled: aggregate-only run (CAPABILITY_ABSENT)".
- **Sample Explorer**: explicit "unavailable for current protocol" notice naming
  `per_query_state: CAPABILITY_ABSENT`.
- **Composition**: Full / SF / Validated each repopulate with the real encoder row
  (`6003 / 4069 / 586` queries) and every rendered numeric cell is compared to that subset's live
  API JSON — not to a constant.
- **Drill-down**: real image bytes from `/api/images/...` (`naturalWidth > 0`, 40 images, 0
  broken); real ranks `#50,729 / #19,161 / #37,597`.
- **Missing rank**: the sparse real run `eda30a37c0bc17ac43844f09` has 8 ranked of 6003 queries;
  unranked queries render the explicit `unscored` state, never `0` or `#0`.
- **Rapid switching**: 4 bursts of 3–5 clicks; after each, the rendered table equals the
  finally-selected subset's JSON and the active pill agrees. No stale data.

## Verified against fixtures (opt-in only)

The two `SYNTHETIC_*` fixture summaries are exercised **only** through the explicit opt-in path
(`?include_fixtures=true&include_smoke=true`, replayed by request interception in `flows.py`). There
the assertions are: two rows appear, both carry the `FIXTURE` badge, they are named `SYNTHETIC_*`,
their numbers differ from every measured row, the real encoder run sorts **first**, and every
measured row precedes every fixture row.

The **default** path asserts the opposite, which is the intended contract: the rendered table
equals the default API list exactly, contains no `FIXTURE` badge and no `SYNTHETIC_` text.

## Defects

### 1. CONFIRMED — Model column scrolls out of view on narrow screens (2 viewports)

Screenshots: `screens/zoom_table_scrolled_right_390x844.png`,
`screens/zoom_table_scrolled_right_768x1024.png`, `screens/mobile_composition_390x844.png`.

The composition table is 1193px wide in a 366px wrapper. The first column is `position: static`,
so scrolling right moves the Model cell to **x = −827** (390) and **x = −461** (768) while the
numeric cells stay on screen. A reader comparing `12.5% | 0.1176` cannot tell which run the row
belongs to. Measured by `flows.py` at `maxScroll = 853` (390) and `499` (768); `visibleAfter: false`.

Reported only — `workbench/frontend/style.css` and `CompositionView.tsx` are owned by another
worker. A `position: sticky; left: 0` first column fixes it; I verified this locally (Model cell
stayed at x=13 after a full right scroll) and reverted, since the fix is not mine to land.

### 2. Cosmetic — Dashboard loading placeholder is the same em-dash as an absent value

Screenshots: `screens/dashboard_1440x900.png`. First paint of `.cards b` reads
"— registered models" until `/api/models` resolves (~300ms), then "17 registered models". The
required end state holds (verified against `/api/models` length, 17); the loading glyph is
indistinguishable from a genuinely missing count. Recorded as an observation, not gated.

### Inspected and found clean

- **Numeric formatting** — all numeric cells `ui-monospace` with `tabular-nums`, right-aligned,
  consistent 4dp nDCG / 1dp percentages; columns line up in
  `screens/zoom_composition_table_1440x900.png`.
- **Contrast** — every token pair measured, lowest 4.57:1 (FIXTURE badge), all ≥ WCAG AA normal
  text.
- **Focus order** — 31 tabbables on Composition, in DOM order, no invisible stops, 2px
  `rgb(88,166,255)` focus ring on every one.
- **Tap targets** — nav buttons 31px, subset pills 47px, filter controls 36px; all ≥ WCAG 2.5.8
  (24px).
- **Benchmark tables at 390** — scroll inside their wrappers, no clipped text; badges wrap rather
  than truncate (`scrollWidth == clientWidth`).
- **Gate detail panel** — opens, provenance table readable, `PAPER_REPORTED` per row.

### Fixed

Nothing. My `style.css` edits (`.model-cell { align-items: flex-start }` and a sticky first
column) were **reverted** at the user's direction; `dist/` was rebuilt so the served CSS hash is
back to the original `index-CY7KtFqe.css`.

## Artifacts

```
browser_qa/
  README.md                  this file
  flows.json                 raw interactive-flow output (13/15, per-check detail)
  flows.py                   flow driver (reuses browser_qa.py)
  zooms.py                   element-crop capture
  harness_benchmark.json     \  raw browser_qa.py output,
  harness_composition.json    |  3 viewports each
  harness_dashboard.json      |
  harness_evaluation_runs.json|
  harness_drilldown.json     /
  harness_*/                 screenshots from those sweeps
  screens/                   146 PNGs: <flow>_<viewport>.png + _full.png + zoom_*.png (2x DSF)
```
