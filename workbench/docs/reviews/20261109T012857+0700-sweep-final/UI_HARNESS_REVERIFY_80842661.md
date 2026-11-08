# Browser harness re-verification at 80842661

`FINAL_PC_RELEASE_VERDICT.md` row "Build/Playwright/image evidence" claimed the harness was still stale
against the `4e61f2f` page split. `80842661` rewrote `workbench/frontend/tests/ui_smoke.py` (+618/−733)
and `e2e_lane.py` (+439 changed lines). This file records running them rather than assuming either way.

Date: 2026-11-09. Revision `80842661e672736c3e2cf4bb4c49a9473db708f8`. Interpreter `.venv/bin/python`
(playwright present; browser is the system `/usr/bin/google-chrome`, no download).

## Commands and results

| command | target | exit | checks |
|---|---|---:|---|
| `.venv/bin/python workbench/frontend/tests/ui_smoke.py --url http://127.0.0.1:5173 --shots /tmp/ui-smoke-80842661/cold --session cold` | live production stack | **0** | 3 |
| `... --session composition` | live production stack | **0** | 6 |
| `... --session mobile` | live production stack | **0** | 4 + 1 documented SKIP |
| `... --url http://127.0.0.1:5765 --session fixture` | fixture stack (`/tmp/fixture_stack.sh`, already running) | **0** | 5 |
| `... --url http://127.0.0.1:5173 --session all` | production | **1** | — |
| `.venv/bin/python workbench/frontend/tests/e2e_lane.py --tmpdir /dev/shm/e2e-80842661` | own isolated `:8775`/`:5775` servers | **0** | A/B/H × 4 viewports |

`--session all` exits 1 on the production stack **by design, not a UI defect**: it chains cold →
fixture → composition → mobile, and `fixture` requires `fashioniq_val_split` runs. Production serves
`/api/runs` = 0 entries, so `set_protocol` waits for an `<option>` that cannot exist and raises
`playwright._impl._errors.TimeoutError: Page.wait_for_function: Timeout 120000ms exceeded.` at
`ui_smoke.py:223`. Run individually, cold/composition/mobile all pass on production and fixture passes
on the fixture stack — the four sessions cover what `all` would have.

## What the checks asserted

- **cold (3):** registry card reads `entry='17' bị chặn='10'` matching `/api/models` = 17 entries;
  17 rendered model rows for 17 API entries; 14 of 17 models declare `blocking_reasons` and all 14
  render a disclosure reading `"<n> lý do"`.
- **composition (6):** 9 composition rows at 1440×900 and 1920×1080 with `table.scrollWidth ==
  .table-wrap.clientWidth` (no overflow, 0 narrow cells holding 200+ chars); expanding a row leaves
  table width unchanged and document height at 1.31×/1.32×; benchmark view never dead-ends.
- **mobile (4 + SKIP):** `body.scrollWidth=390 == innerWidth=390`; 6 nav area buttons all inside the
  viewport; 17 model rows with the blocker column reachable inside its scroll box
  (`scrollWidth=861`) — recorded as an explicit SKIP rather than a false pass; composition table
  contained in a 330px box with no page overflow.
- **fixture (5):** 2 index runs listed with real `run_id`s/badges; injected `/api/models` abort
  surfaced as `Yêu cầu thất bại` with a retry control; after retry the error region is gone and the
  registry re-fetched; 0 unexpected console/page/request errors outside the injected window.
- **e2e_lane (A/B/H × 4):** registry count equals `/api/models`; one row per model with a `"<n> lý do"`
  disclosure where declared; containment at 1440×900, 1280×800, 390×844, 360×800.

## Coverage limit, stated exactly

`routes.ts` declares six areas: `overview`, `experiments`, `evaluation`, `samples`, `failures`,
`notebook`. **Covered: three** — `overview`, `experiments` (models + runs), `evaluation` (composition +
benchmark). **Not covered: `samples`, `failures`, `notebook`** — each needs a selected CIRCUS
composition run, which the fixture stack does not serve, and `/api/queries?limit=20000` answers 400
`invalid_pagination` (storage caps limit at 200), so the cohort/annotation flows cannot be driven end to
end on any live stack today. The rewritten harness states this in its own docstring and drops those
flows explicitly rather than silently skipping them.

## Live stack untouched

`127.0.0.1:5173` and `127.0.0.1:8000` both 200 before and after. `serve_workbench.py` PID 1267083 and
its uvicorn PID 1267084 were not restarted (same process, uninterrupted uptime). `e2e_lane.py` bound
its own `:5775`/`:8775` and reported "Servers stopped cleanly."
