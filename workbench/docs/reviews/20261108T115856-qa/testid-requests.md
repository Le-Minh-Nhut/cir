# data-testid hooks the browser QA harness needs

The harness resolves every element through `workbench/scripts/browser_qa/harness.py:SELECTORS`:
`data-testid` first, then role+accessible-name, then a legacy CSS/text fallback. A resolution that
fails on every strategy is recorded as the distinct failure category `selector_miss` in
`qa_report.json` and **never crashes the run**, so a missing hook degrades the report instead of the
harness.

That means this file is a request list, not a blocker: everything below is either already satisfied
by the redesign or is a hook that would replace a brittle fallback.

Legend: **HAVE** = present in `workbench/frontend/**` today · **WANT** = add it ·
**WANT ATTRIBUTE** = the element exists, the machine-readable attribute does not.

## Shell / navigation

| Selector key | Hook | State |
|---|---|---|
| `nav` | `data-testid="workbench-nav"` | **HAVE** |
| `nav.item` | `data-testid="area-nav-<area>"` for `overview, experiments, evaluation, samples, failures, notebook` | **HAVE** |
| `nav.active` | `aria-current="page"` on the active area button, or `data-testid="area-nav-active"` | **HAVE** (`aria-current`) |
| `area.root` | `data-testid="area-<area>"` on each area's outer element | **HAVE** for overview/experiments/evaluation/samples/failures; **WANT** for `notebook` (no `area-notebook` node) |
| `error.banner` / `error.dismiss` | `data-testid="error-banner"` / `"error-dismiss"` | **WANT** — today the guard error is a `.error` aside in the legacy shell and `notice-dismiss` in the redesign; one stable hook would make "an error is visible" independent of which shell renders it |
| `error.retry` | `data-testid="error-retry"` on the retry control of a failed region | **HAVE** (`ErrorState`) |
| `empty.state` | `data-testid="empty-state"` + `data-empty-kind` | **HAVE** — `data-empty-kind` is what lets the harness tell "no match" from "capability absent"; please keep it |
| `notice` | `data-testid="notice"` | **HAVE** |

## Overview / experiments

| Selector key | Hook | State |
|---|---|---|
| `overview.cards` | `data-testid="overview-cards"` | **HAVE** |
| `runs.any_option` | `data-testid="run-option-<key>"` where key is `c:<run_id>` or `i:<run_id>` | **HAVE** |
| `runs.row` | `data-testid="run-row-<key>"` | **HAVE** |
| `runs.selected_count` | `data-testid="selected-run-count"` with the count as its text | **HAVE** |
| `models.row` | `data-testid="model-row-<model_id>"` | **HAVE** (models view) |

## Evaluation (composition) and the query inspector

| Selector key | Hook | State |
|---|---|---|
| `comp.subset` | `data-testid="subset-pill-<full\|sf\|validated>"` + `role="tab"` + `aria-selected` | **HAVE** |
| `comp.table` | `data-testid="composition-table"` | **HAVE** |
| `comp.row` | `data-testid="composition-row-<run_id>"` | **HAVE** |
| `comp.pager` | `data-testid="pager-total"` with `data-total` and `data-shown` | **HAVE** — this is the single most valuable hook in the whole app for QA; the denominator comparison in flow (e) depends on it |
| `comp.pager.next` | `data-testid="pager-next"` | **HAVE** |
| `comp.qid` | `data-testid="query-id"` on each row's id button | **HAVE** |
| `explorer.query_title` | `data-testid="query-title"` on the inspected query heading | **HAVE** |
| `comp.rank` | `data-testid="rank-<condition>"` per condition (`multimodal`, `text_only_black_image`, `image_only_empty_text`) | **HAVE** |
| `explorer.image` | `data-testid="query-image"` on each query image | **WANT** — today the images are plain `<img>` inside `.shots`, so the harness falls back to a class selector; adding `data-image-id` would also let flow (e) compare ids instead of parsing `src` |
| `comp.topk` | `data-testid="topk-grid"` | **HAVE** |
| `explorer.candidates_unavailable` | `data-testid="candidates-unavailable"` | **HAVE** |
| `comp.filter` | `data-testid="filter-<condition\|outcome\|category\|order>"` | **HAVE** |
| `comp.search` | `data-testid="filter-query-id"` | **HAVE** |
| `comp.empty` | `data-testid="composition-empty"` | **HAVE** |

## Failure analytics and cohorts

| Selector key | Hook | State |
|---|---|---|
| `fail.row` | `data-testid="analysis-row"` + `data-query-id` | **HAVE** |
| `fail.preset` | `data-testid="failure-preset"` | **HAVE** |
| `cohort.save` | `data-testid="cohort-save"` | **HAVE** |
| `cohort.item` | `data-testid="cohort-<cohort_id>"` | **HAVE** (cohorts view) |
| `cohort.name` | `data-testid="cohort-id"` on the cohort id input | **WANT** — the id is auto-generated and there is no input; the harness now reads the id from the API instead. Add the hook only if the id becomes editable |
| `save.feedback` | `data-testid="save-feedback"` with `data-status="idle\|saving\|saved\|error"` on the save result | **WANT ATTRIBUTE** — the harness currently reads `role="alert"`/`role="status"`, which is correct but forces it to parse Vietnamese prose to tell "saved" from "refused" |

## Notebook (annotations)

| Selector key | Hook | State |
|---|---|---|
| `annot.note` | `data-testid="annotation-note"` | **HAVE** |
| `annot.save` | `data-testid="annotation-save"` | **HAVE** |
| `annot.query` | `data-testid="annotation-query-id"` | **HAVE** — needed because a bare `7:603` exists in both dataset identities; the harness asserts the protocol is shown as well |

## What the misses in the last run actually meant

Two categories were reported as `selector_miss` in earlier runs and are worth naming because they
are UI facts, not harness bugs:

1. `explorer.image` — the inspector renders `<img>` with no testid. Fixed by the fallback above;
   the requested hook removes the fallback.
2. `area-notebook` — the notebook area renders no outer `data-testid`, so `area.root` cannot
   identify it. The six-area nav is still verifiable through `area-nav-notebook` + `aria-current`.

## One request that is about assertions, not selectors

Flow (f) needs to prove that a saved cohort equals the **full published scope**, not the page.
That requires the save request's payload to be observable. `data-testid="cohort-save"` with
`data-scope-count="<n>"` (the number of queries about to be written) would make the assertion
direct instead of inferring it from the button's label text.
