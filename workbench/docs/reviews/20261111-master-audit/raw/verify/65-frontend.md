# 65 — frontend: R06 (native inventory), R07 (raw unit), R10 (inspector rank)

## Follow-up (live UI verification): two precedence defects

### D1 — a protocol-less native row was lent to every protocol
`nativeRowsFor`'s second tier (`own.filter(row => !row.protocol_id)`) made a row with no protocol
identity match whichever protocol the reader had selected: `airknow` read "Có dữ liệu native" under
`fashioniq_original_split`, where `/api/native-measurements` has 0 rows with metrics. That is R01 in
the UI layer — `otherScopeFor` already refuses `!protocol` rows, and these helpers now agree with it.

* `nativeRowsFor` / `nativeMeasuredIds` match **exact `(model_id, protocol_id)` only**.
* `nativeUnattributedFor(rows, model)` surfaces the protocol-less rows separately.
* `inventoryStatus` gives them `"Số đo native không ghi protocol"` with the native table as the open
  action, so the file stays visible and the scope cannot claim it.

### D2 — the deferral label concealed real numbers
`csmcir` read "Tạm gác" under `fashioniq_original_split` while the native table serves `57.05%` for it
there. The campaign status is correct and stays; the numbers are now named beside it:
status cell `Tạm gác · 1 số đo native ở phạm vi này` + a `deferred-native-<model>` button opening the
native table.

### Full precedence audit (`inventoryStatus` in `scope.ts`, single source)

Three further contradictions the ternary tower had, all fixed and covered by
`a label never contradicts a row the API serves in the same scope`:

1. `nativeOnly` was `!entry && nativeInScope > 0` → a model with **two** composition runs in scope
   plus a native macro read "Có dữ liệu native", denying the runs it had. Now requires
   `usableCount === 0`.
2. Every non-composition `entry` (index **and** aggregate) was labelled "Có dữ liệu native" — a
   pointer at the wrong table. Index rows now say `Chỉ có lượt chạy, chưa có số`.
3. An aggregate/composition entry with `aggregate_metrics: false` read "Có số đo" while the
   measured counter (which requires that flag) excluded it — label and counter disagreed on the same
   row. Both now read the flag: `Chỉ có hạng từng query` / `Chỉ có lượt chạy, chưa có số`.
4. Multi-run (`usableCount > 1`) rows that DO have aggregate metrics read "N lượt cần chọn" while the
   counter already counted them as measured; they now read `${n} lượt trong phạm vi, đã có số đo — chọn một`.

Remaining pairs audited and found consistent: `deferred → run` (a deferred model with a run keeps the
deferral and adds the native count), `entry → usable>1` (entry resolved first, so a multi-run row with
a selected run never says "cần chọn"), `elsewhere` is only consulted when no entry, no native row and
not deferred, and `downloaded` only distinguishes two *absence* labels, neither of which claims a
number the API serves.

Original pass below.

Owned files only: `frontend/data.ts`, `frontend/ModelInventory.tsx`, `frontend/EvaluationPage.tsx`,
`frontend/QueryInspector.tsx`, `frontend/scope.ts`, plus `frontend/tests/inspector-window.test.cjs`.
No backend/composition/replay file touched; `inspector.ts` is **not** mine — an initial edit there was
reverted (`git status` clean for it) and the pure decision moved into `scope.ts`.

## R06 — "Chưa có số đo" for a scope with a native measurement

Root cause: `measuredIds` was built from `data.catalog` alone. Native rows fed only
`otherScopeFor(...)`, and the result was filtered to scopes **different** from the current one, so a
native macro in the selected protocol was structurally unable to make its model measured there — and
the cross-scope link that proved the row existed only appeared after switching protocol.

New pure helpers in `scope.ts` (no imports → node-executable, same module the app ships):

* `nativeMeasuredIds(rows, protocol)` → models with a native row that **states a number** in exactly
  this protocol (exact `(model_id, protocol_id)`; the protocol-less fallback tier was removed — D1).
* `nativeRowsFor(rows, model, protocol)` → that model's rows, exact `(model_id, protocol_id)` only.
* `nativeUnattributedFor(rows, model)` → number-stating rows for that model that name NO protocol;
  reported separately, never counted as measured — see D1.

`ModelInventory.tsx`:

* `measuredIds = compositionMeasuredIds ∪ nativeMeasuredIds(...)` → a native macro in the ACTIVE
  protocol now satisfies "measured" in the current scope.
* status label: native-only rows read `"Có dữ liệu native"`; `"Có số đo"` stays reserved for a
  composition/aggregate run with aggregate metrics. Before, `"Có dữ liệu native"` was unreachable in
  the current protocol (it keyed off a catalog entry, which is never native).
* open action for a native-only row: `onGoto({ area: "evaluation", view: "native", … })` — the same
  destination the cross-scope native link uses; never a composition run that does not exist.
* the inventory now also waits on the native region (`loading`/`error`), for the same reason it
  already waited on the composition region: an unarrived catalog is not "no native number".
* `measurementCount` was `metrics && !rollup`. Backend census `counts.with_metrics` is
  `len(per_model)` = measured rows that are `counted` **and** `artifact_backed`. Aligned to exactly
  that predicate (`metrics && counted !== false && artifact_backed === true`); no third definition.
* toolbar truthfulness: `measuredCount = models.length - pending.length - deferred.length` is computed
  once and used by both the stat tile and the `Có số đo (n)` option, so the two cannot drift.

Probe (`/tmp/r06probe/inventory_probe.mjs`, adapted calls only; real `buildCatalog` + real helpers),
native row for `conesep` in `fashioniq_full_gallery_ref_excluded`, `catalogEntries=0`:

| | measured | elsewhere | status |
|---|---|---|---|
| before (current protocol) | `false` | `[]` | `Chưa có số đo` |
| after (current protocol) | `true` | `[]` | `Có dữ liệu native` |
| after (other protocol) | `false` | 1 link | `Có kết quả ở scope khác` |

R06 probe re-run after the precedence fix: unchanged for exact-protocol rows (`measured: true`,
`Có dữ liệu native`); the protocol-less fallback tier is gone, so no protocol can borrow such a row.

Navigation for the native-only row: `evaluation / view=native` (probe reports
`open: "evaluation/view=native"`).

## R07 — raw disclosure printed on the replaced row's unit

Before: `fmtRecall(row.raw_metrics.r10 ?? null, row.unit, 4)`
After: `fmtRecall(row.raw_metrics.r10 ?? null, row.raw_unit !== undefined ? row.raw_unit : row.unit, 4)`
plus the line now names the scale shown (`Số gốc theo thang …`), says when the fallback was used
(`backend không kèm thang gốc`), and when `raw_unit_source === "inferred_from_magnitude"` it says
`suy theo độ lớn, không phải khai báo` — same rule as the unit badge one column over.

`NativeMeasurement` gained `raw_unit` / `raw_unit_source` (backend already serves them beside
`raw_metrics`).

## R10 — inspector explained a window with another condition's rank

Before (pure source read):
`const ranks = conditions.map(c => c.rank).filter(...); const best = Math.min(...ranks);` then the
MM window explained with `best` — MM rank 50 + text-only rank 5, K=10 → "ảnh đích ở hạng tốt nhất 5
— ngoài cửa sổ Top-10" while describing MM.

After — `candidateWindowVerdict(...)` in `scope.ts`, keyed on the **viewed** condition (the URL's
`condition`, which is also the condition the candidate request names). Quoted rank is that
condition's rank; the sentence names it ("theo điều kiện Ảnh + văn bản (MM)"). Verdicts:

* `hidden` — target in the rendered window, or depth 0.
* `pending` — the region for THIS window is `idle`/`loading`: renders "Đang tải cửa sổ … Chưa kết
  luận", no conclusion. Previously an empty list while loading read as "outside the window".
* `unavailable` — region `error`, or the viewed condition has no exact rank (`null`): states that
  reason instead of substituting another condition's rank.
* `ranked` — the target is outside the rendered window; body distinguishes "rank ≤ K but the export
  does not contain that image" from "rank > K".

Reference image, edit text, target tiles and the ranked candidate grid with rank/score are unchanged.

## Verification

* `node --test tests/inspector-window.test.cjs tests/light-workspace.test.cjs` → 17 pass, 0 fail
  (4 new: viewed-condition rank, pending/failed/unscored no-conclusion, hidden cases, native scope).
* `workbench/tests` python slice (`test_web_scope_and_labels`, `test_frontend_catalog`,
  `test_frontend_requests`, `test_frontend_staleness`, `test_frontend_layout_provenance`,
  `test_derived_subset_status`) → 66 passed.
* `npm run build --prefix workbench/frontend` → `✓ built in 249ms` (tsc -b + vite).
