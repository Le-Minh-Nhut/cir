# Live verification of the deployed stack (browser half)

Run against the restarted services at `1c23d46` (production tree `e833eda`), 2026-11-13.
Probe: `probe_live_verify.py` (headless Chrome, `/usr/bin/google-chrome`, viewport 1440x900 plus
390x844). Raw readings: `live-verify.json`. Screenshots: `screenshots/`.

**Verdict at the time of this run: 9 of 10 criteria held; the bare deep link did not (§4).** That
defect was then repaired and re-verified (§8); this file keeps the run that found it, because a
record that only shows the passing state cannot show what was fixed.

## 1. Headline readings

| criterion | reading | verdict |
|---|---|---|
| overview prints both counts, distinct labels | `16 công bố trên web` / `17 trong registry` | PASS |
| inventory shows 16, no PAIR row | `count-models` = 16, 16 rows, no `pair` row id | PASS |
| native table has no PAIR rows | 9 + 6 + 13 = 28 rows across the three protocols that have any, 0 PAIR | PASS |
| bare deep link shows the deferral notice | absent; `model=` is dropped — see §4 | **FAIL** |
| parity card renders `Chuẩn parity-relative/v1` as a string | `Chuẩn parity-relative/v1 · tính từ artifact đo, không đọc từ văn bản` | PASS |
| parity per-model failure counts | encoder 9, habit 2, hint 2, limn 3, tgcir 3 (from the DOM) | PASS |
| CLVC 12 cells vs `INVENTORY.md` | 0 mismatches (table §3) | PASS |
| ablation cells: self-match banner + target outside window | 4/4 banners, 12/12 "ngoài cửa sổ Top-100" | PASS |
| console errors | 0 | PASS |
| page errors | 0 | PASS |
| genuine failed requests | 0 (1 abort classified as a superseded duplicate, §5) | PASS |

Mobile 390x844: published 16 / registry 17, `scrollWidth - innerWidth` = **0** (no horizontal
overflow), 0 console errors.

## 2. Parity card, as rendered

```
KHÔNG mô hình nào PASS (0/7)
Không mô hình nào đạt: 7/7 mô hình có subcheck bắt buộc FAIL. Đây là kết luận từ artifact,
không phải lỗi tải.
```

`standard` is rendered through `Chuẩn ${gate.standard}` and printed as a single string, not an
array — the regression where an array reached the sentence does not recur.

## 3. The 12 CLVC cells — UI vs `raw/clvc-dualscope/INVENTORY.md`

Read from the probe note's `· hạng đích #<n> trên 74 381 ảnh ·` in the rendered DOM after selecting
each combination; no API value stands in for a rendered one.

| # | query | scope | condition | inventory | rendered | match |
|---|---|---|---|---|---|---|
| 1 | `7:3038` | `shirt` | `multimodal` | 848 | 848 | yes |
| 2 | `7:3038` | `shirt` | `text_only_black_image` | 304 | 304 | yes |
| 3 | `7:3038` | `shirt` | `clvc_image_encoder_only` | 8702 | 8702 | yes |
| 4 | `7:3038` | `toptee` | `multimodal` | 1253 | 1253 | yes |
| 5 | `7:3038` | `toptee` | `text_only_black_image` | 1549 | 1549 | yes |
| 6 | `7:3038` | `toptee` | `clvc_image_encoder_only` | 1267 | 1267 | yes |
| 7 | `7:3790` | `shirt` | `multimodal` | 412 | 412 | yes |
| 8 | `7:3790` | `shirt` | `text_only_black_image` | 1466 | 1466 | yes |
| 9 | `7:3790` | `shirt` | `clvc_image_encoder_only` | 5481 | 5481 | yes |
| 10 | `7:3790` | `toptee` | `multimodal` | 259 | 259 | yes |
| 11 | `7:3790` | `toptee` | `text_only_black_image` | 568 | 568 | yes |
| 12 | `7:3790` | `toptee` | `clvc_image_encoder_only` | 4535 | 4535 | yes |

**Mismatches: 0.** Each cell rendered 100 candidate tiles; each ablation cell rendered exactly one
`is-reference` tile carrying the self-match banner, and all 12 rendered the target as
`ngoài cửa sổ Top-100 đang hiện` with the exact full-gallery rank.

## 4. The failing criterion — `?view=native&model=pair` loses `model=`

Measured in the browser, three shapes of the same link:

| URL requested | URL after load | notice |
|---|---|---|
| `?area=evaluation&view=native&model=pair` | `?area=evaluation&view=native&protocol=circus_fashioniq_release_v1&subset=full` | **none** |
| `?area=evaluation&view=native&protocol=…&model=pair` (no `subset`) | `…&protocol=…&subset=full` | **none** |
| `?area=evaluation&view=native&protocol=…&subset=full&model=pair` | unchanged | **renders** |

Mechanism, from the source: `model` is not a field of `Location`. `routes.ts` neither parses nor
serializes it (`toSearch`'s key list has no `model`), and `NativeMeasurementTable` reads it straight
off `window.location.search` (`EvaluationPage.tsx:579`). Any `replaceState` that runs while the
landing-scope effect settles therefore rewrites the URL from `Location` and drops the parameter —
the effect rewrites whenever `protocol` or `subset` was unstated, which is exactly the bare link.
Because the notice is computed from a live `window.location.search` read at render time, it
disappears with the parameter.

**Not introduced by this deployment.** `git diff --stat 43bfc9e HEAD -- workbench/frontend/` is
empty: the frontend is byte-identical to the tested revision, so this behaviour is present at
`TESTED_SOURCE_SHA` too. The consequence for acceptance is still real: the check as written
("a deep link `?view=native&model=pair` renders the deferral notice rather than a broken page")
does not hold for the bare link it names.

Scope of the damage: the page does not break (`area-native` renders, 0 rows, 0 pair rows, no
console error). What is lost is the explanation — the reader sees an empty native table instead of
the deferral reason, which is the exact misreading `webDeferralNotice` exists to prevent
(`scope.ts:459-467`).

Not fixed here: the fix is a `Location`/`toSearch` change in the pinned production tree, which
requires its own sweep. Recorded for the operator.

## 5. Requests

`console_errors: 0`, `page_errors: 0`, `failed_requests: 0` genuine.

One request was recorded as aborted and is classified rather than counted as a failure:

```
GET /api/composition/diagnostics/clvc-dual-scope/candidates?qid=7:3790&scope=toptee
    &condition=clvc_image_encoder_only&depth=100   net::ERR_ABORTED
```

Same URL also completed for the same page (`requestfinished` observed), i.e. the app issued the
window twice across a control re-render and the currency guard cancelled the older copy
(`useRegion` aborts superseded bytes). The probe waits for zero in-flight window requests before
each control change, so this is the app's own deduplication, not the probe racing it. Recorded
because "0 failed requests" would otherwise be a claim the log does not support.

## 6. Screenshots (all in `screenshots/`)

```
01-overview-1440x900.png                     06-overview-390x844.png
02-inventory-1440x900.png                    07-diagnostic-390x844.png
03-native-table-1440x900.png                 03b-pair-deep-link-bare-1440x900.png
04-parity-1440x900.png                       03c-pair-deep-link-scoped-1440x900.png
05-clvc-diagnostic-panel-1440x900.png        clvc-<qid>-<scope>-<condition>.png  (12)
08-native-table-<protocol>.png               (3, one per protocol with rows)
```

## 7. Reproduce

```bash
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261113-final-acceptance/raw/live-verify/probe_live_verify.py
```

Exit 0 only when every criterion holds; it exits 1 today on §4, by design — the probe is not
tolerant of the defect it found.

## 8. The deep-link defect, repaired

§4 named a `model` parameter the UI read but the location model did not carry. It is now a field of
`Location`: parsed verbatim, serialized last and only when set, so it survives every `replaceState`
and every `goto`; links that never named a model serialize byte-identically. The second half was
needed for the same link — `?view=native` with no `area=` collapsed to the dashboard, because
`view` ids are unique across areas but the default area (`overview`) declares no views. A URL that
names a view but no area now takes the area that owns that view.

Evidence for the repair:

| check | result |
|---|---|
| `node --test tests/model-param.test.cjs` | **9 passed, 0 failed** — parse→`toSearch` round trip, no-area form, fully explicit form, explicit area wins over the view's owner, unknown view keeps the model, absent model writes nothing, unknown model id preserved, existing links byte-identical |
| `tsc -b` via `npm run build` | clean; bundle `index-8klMGIWd.js`, 399 398 B, sha256 `43b2618c56bf2a355dda893c774d8160ab64fcbb1df88623d84159baea284d87` |
| rebuilt `dist/index.html` | sha256 `696c23fcaec29e809f997063e3a758afd3665859d397b9741e9c76fcc6ea0d04` |

Three deep-link shapes in the browser, captured as `screenshots/native-deferral-{bare,no-area,explicit}.png`:

| URL | expectation |
|---|---|
| `?area=evaluation&view=native&model=pair` | `model=` survives the correction, deferral notice renders |
| `?view=native&model=pair` | area resolves to `evaluation` from the view, notice renders |
| `?area=evaluation&view=native&protocol=circus_fashioniq_release_v1&subset=full&model=pair` | unchanged URL, notice renders |

`workbench/frontend/dist/` is gitignored (`.gitignore:95`, 0 tracked files), so the rebuilt bundle is
deployed but deliberately not committed.

The post-redeploy re-check of these three URLs, the 12 cells and the endpoint rows is recorded in the
handoff report, not here, so this file does not claim a measurement taken after the tree froze.

## 9. Re-check on the redeployed stack (`90f7b2c`, bundle `index-8klMGIWd.js`)

Everything in §1-§8 was measured at `937f2ec`, before the repair was deployed. The re-check below ran
after both services were restarted onto the new pin, so it is the state that ships. Raw readings:
`live-verify-post-redeploy.json`; the three deep-link shots are
`screenshots/redeploy-deeplink-{bare,no-area,explicit}.png`.

| criterion | reading | verdict |
|---|---|---|
| overview both counts, distinct labels | `16` / `17` | PASS |
| inventory 16, no PAIR row | 16 rows, no `pair` row | PASS |
| native tables, no PAIR rows | 0 across all three protocols | PASS |
| bare deep link shows the notice | **renders** — `?area=evaluation&view=native&model=pair` keeps `model=` through the correction | **PASS** |
| parity standard as a string | `Chuẩn parity-relative/v1 · tính từ artifact đo, không đọc từ văn bản` | PASS |
| parity per-model failure counts | encoder 9, habit 2, hint 2, limn 3, tgcir 3 | PASS |
| CLVC 12 cells vs inventory | **0 mismatches** | PASS |
| ablation cells: banner + outside window | 4/4, 12/12 | PASS |
| console / page / genuine request failures | 0 / 0 / 0 (1 superseded duplicate) | PASS |

The probe reports `verdict: PASS` with `failed_criteria: []` on this deployment.

### The three deep-link shapes, on `:5173`

| requested | final URL | notice | console errors | failed requests |
|---|---|---|---|---|
| `?area=evaluation&view=native&model=pair` | unchanged | **visible** | 0 | 0 |
| `?view=native&model=pair` | `?area=evaluation&view=native&protocol=circus_fashioniq_release_v1&subset=full&model=pair` | **visible** | 0 | 0 |
| `?area=evaluation&view=native&protocol=…&subset=full&model=pair` | unchanged | **visible** | 0 | 0 |

The second row is the case that also needed the view→area resolution: `model=` survives AND the area
is now resolved from the view instead of collapsing to the dashboard.

### API rows on the redeployed stack

`/api/models` 200, 16, no `pair` · `/api/models/publication` 17/16/1 · `/api/native-measurements` 51
rows, 0 PAIR · `/api/runs/parity-gate` 200, `passed_models: []`, `standard` the string
`parity-relative/v1`, failures encoder 9 / habit 2 / hint 2 / limn 3 / tgcir 3 ·
`/api/composition/runs/7b1fa850…` 200, `still_unscoped: 6003`, no `category_agnostic` · both
diagnostic routes 200, `DIAGNOSTIC_COMPLETE`, `complete: true`, 0 defects.

`bash workbench/docker/acceptance.sh` → `ACCEPTANCE PASSED (skipped: G-status-unverified)`.

Measured, and worth knowing for a handoff: the first acceptance run after the restart reported
`SKIP A/C/G/I live checks (no server)` because `/api/models` was still cold (per-checkpoint
availability resolution, ~1 min). It is a false negative of `live()`'s single probe, not a broken
stack — after warming the endpoint the same run passed every live group.
