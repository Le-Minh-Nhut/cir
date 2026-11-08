# The browser harness no longer matches the UI it drives

> **STATUS — FIXED by `80842661`, 2026-11-09. This document describes a state that no longer exists.**
> `80842661` rewrote both harnesses against the `4e61f2f` page split (`ui_smoke.py` +618/−733,
> `e2e_lane.py` +439 changed lines), replacing label-based navigation with the `data-testid` ids
> `routes.ts` declares. Re-verified at `80842661`: `ui_smoke.py` cold 3 checks, composition 6, mobile 4
> (+1 documented SKIP), fixture 5 — all exit 0; `e2e_lane.py` exit 0 with journeys A/B/H at all four
> viewports. Read everything below as the diagnosis that led to the rewrite, not as current status.
> One limitation survives and is documented in the harness itself rather than hidden: of the six
> `routes.ts` areas, `overview`, `experiments` and `evaluation` are covered; `samples`, `failures` and
> `notebook` are not, because every flow there needs a selected CIRCUS composition run that no live
> stack serves end to end. See `FINAL_PC_RELEASE_VERDICT.md` row "Build/Playwright/image evidence".

Date: 2026-11-08. Branch `research/cir-failure-workbench`. Written after the release-guard round; no
production change accompanies it.

## What is broken

`workbench/frontend/tests/ui_smoke.py` and `e2e_lane.py` were written against the single-page UI. The
page split (`4e61f2f`, "Split the frontend into pages and let the payload, not the UI, decide status")
replaced that UI and did not update them, so both suites now fail immediately on the
default session, before they assert anything about the product:

| harness expectation | what the UI renders now |
|---|---|
| `div.cards b` reading `"17 registered models"` | removed; the count moved to `[data-testid="area-overview"] .stat-value` inside the "Registry mô hình" card |
| `div.cards b` reading `"N mock runs"` | removed; the same `/api/runs` count is the sidebar badge `[data-testid="area-nav-experiments"] .nav-count` |
| `open_page(page, "Models")`, `"Sample Explorer"`, `"Annotations"`, `"Compare Models"`, `"Common Failures"`, `"Hypotheses"` | nav labels are Vietnamese now: `Mô hình`, `Khám phá mẫu`, `Sổ nghiên cứu`… and most flows became *views* under an area (`routes.ts`), not top-level buttons |
| `div.pager button` `"Next page"`, `div.filters button` `"Jump"`, `"Save / update"`, `"Save filtered set"`, `"Dismiss"` | unchanged strings only by luck; several of these live in different containers after the split |

Measured, not inferred: `ui_smoke.py --url http://127.0.0.1:5173 --session cold` failed twice —
first at `Locator.wait_for: Timeout 90000ms exceeded … div.cards b … "registered models"`, then, after
locally repointing the two overview waits at the current selectors, at
`Locator.click: Timeout 30000ms exceeded … nav button … "Models"`. The second failure is the same class
of staleness one layer deeper, which is why the whole expectation set needs re-deriving rather than
patching selector by selector.

## Why this round did not fix it

`workbench/frontend/tests/` is **not** in the release guard's exclusion list
(`DOCS_AND_TESTS_EXCLUDED_PATHS = (":(exclude)workbench/docs", ":(exclude)workbench/tests")`), so any
edit there counts as a **production** change for `TESTED_SOURCE_SHA`: it invalidates the recorded sweep
and demands a fresh ~58-minute serial run. Fixing the harness properly means re-deriving six page names,
four control selectors and the fixture expectations for a split that changed the navigation model — and
a half-fixed harness that still cannot run would buy nothing while costing a full re-measurement.

Partial edits made while investigating were reverted (`git checkout --`) so the tree carries no
half-finished harness.

## What this means for the verification this round claims

The `scoring_impl` work was verified by something other than this harness: a targeted Playwright probe
of the evaluation area against the live stack, recorded in the round's report — 9 rows rendered, the
mixed-implementation notice visible and naming `einsum, einsum-optimized, matmul`, the `scoring impl`
detail row present, and zero console errors. That is narrower than the harness's coverage (which spans
cold load, pager, annotations, comparison and injected-failure recovery), and it should be described
that way: **the evaluation area was browser-verified; the other areas were not re-verified this round
because their harness is stale.**

## How to fix it, in order

1. Re-derive the navigation model from `routes.ts`: `AREAS` × `views`. Add one helper that opens
   `(area, view)` by `data-testid="area-nav-<area>"` plus the view control, and delete `open_page`.
2. Replace the two overview waits with the selectors named in the table above; keep the
   "loading must COMPLETE, not merely render a placeholder" intent by waiting for a non-empty value.
3. Re-point the control selectors (`pager`, `filters`, `Save …`, `Dismiss`) at their current
   containers, or add `data-testid`s to the controls so the harness stops depending on English copy.
4. Run `--session cold`, then `--session fixture`, then the viewport lane, against a served production
   build; only then treat the harness as covering the UI again.

Landing that is a production change: it needs its own serial sweep and a `TESTED_SOURCE_SHA` advance,
exactly like any other production commit.
