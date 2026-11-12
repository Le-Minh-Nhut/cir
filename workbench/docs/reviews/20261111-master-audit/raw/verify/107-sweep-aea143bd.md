# 107 — stationary sweep at `aea143bd`: 20 failed / 3126 passed / 47 skipped, pin ADVANCED

Final pin round. Production moved after the `117fc172` advance (a UI condition-list fix touching
`routes.ts`, `SamplesPage.tsx`, `data.ts`, `backend/composition.py`, `test_frontend_routes.py` and a
rebuilt `dist`), so the `117fc172` pin no longer described the tree and the guard was red — correct
behaviour, not a regression.

**Verdict: the sweep is valid and the pin is advanced to `aea143bd8894881ad7d5b341c21dbeff25cfaaaf`.**
`TESTED_SOURCE_SHA` is that revision in both release documents; `test_handoff_commands.py` is **8 passed**.

---

## 1. Preflight (`107-sweep-preflight-aea143bd.txt`)

| check | value | verdict |
|---|---|---|
| `pgrep -af "pytest workbench/tests"` | empty | PASS |
| `pgrep -af composition_eval.py` | empty | PASS |
| `nvidia-smi --query-compute-apps` | empty | PASS |
| `git status --porcelain` | `?? CLEANUP_REPORT.md.save` only | PASS — **zero modified tracked files** |
| `git rev-parse HEAD` | `aea143bd8894881ad7d5b341c21dbeff25cfaaaf` | the sha pinned |
| `5b5a24a` in history | **YES** (`merge-base --is-ancestor`) | the local unpushed commit is included in the swept tree |
| production diff vs `117fc172` | **5 files, +123 / −25** | recorded |
| frontend freshness (`dist/assets/*.js` ≥ newest `*.ts`/`*.tsx`) | src `1794460770.047`, dist js `1794460856.749` | **FRESH** — no rebuild |
| `dist/assets` | `js=1` (`index-NnjrdZLG.js`), `css=1` (`index-DecQnta9.css`) | PASS |
| served bundle on `:5173` | `assets/index-NnjrdZLG.js` — matches `dist` | PASS |
| `df -h /` free | **87 GiB** | PASS |
| `--collect-only -q` | **3193** | reference for accounting |

Note on disk: the gate was initially **27 GiB**, below the 45 GiB floor. The cause was this
workstream's own scratch — `/tmp/sweep_basetemp_117fc172` (31 GiB) and `/tmp/sweep_basetemp_1d5a66c1`
(30 GiB), both already recorded in `105-…md` / `106-…md`. Removing them restored 87 GiB. Nothing else
was deleted.

The production diff, all 5 files:

```
workbench/backend/composition.py     | 19 +++++++++-
workbench/frontend/SamplesPage.tsx   | 26 ++++++++++----
workbench/frontend/data.ts           | 18 ++++++++--
workbench/frontend/routes.ts         | 68 ++++++++++++++++++++++++++++++------
workbench/frontend/tests/ui_smoke.py | 17 ++++++++---
5 files changed, 123 insertions(+), 25 deletions(-)
```

## 2. The sweep

```bash
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_aea143bd \
  > /tmp/sweep_aea143bd.txt 2>&1 < /dev/null &
```

Launched 2026-11-12T12:30, PID 1764977, finished 13:31.

```text
20 failed, 3126 passed, 47 skipped, 350 warnings in 3684.01s (1:01:24)
```

| check | value |
|---|---|
| `grep -c '^ERROR '` | **0** |
| `grep -c "Errno 28"` | **0** |
| passed + failed + skipped | 3126 + 20 + 47 = **3193** |
| `--collect-only` count | **3193** |
| accounting verdict | **EXACT — 0 unaccounted** |

The predicted collect count was **3191**; the measured count is **3193** (+2). The prediction is
superseded by the measurement, not the other way round.

## 3. Failure set — diff against the `117fc172` sweep

| direction | ids |
|---|---|
| **new** (in this sweep, absent from the 20) | **none** |
| **gone** (in the 20, absent from this) | **none** |

The set is the **identical 20 ids**. That is what the pin round expects: the sweep runs *before* the
advance, so `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` is
still red **during** the sweep (it compares against a pin that does not yet name this tree) and goes
green **after** the advance — verified in §5. Every other id is the verified pre-existing set from
`105-…md` §3.1-§3.2 and `106-…md` §4:

* 18 documented baseline ids (blip ×4, `circus_cli`, `tme_lifecycle` ×2, `tme_precision`, `f08_dcnet`,
  `handoff_commands` SHA guard, `release_blocker_gpu_records` ×7, `prepare_model`);
* 2 undocumented-but-pre-existing (`test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present`,
  `test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers`).

## 4. Pin advance

| document | change |
|---|---|
| `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md` | `TESTED_SOURCE_SHA` → `aea143bd…`; the row's own numbers → 3126/20/47 = 3193, 3684.01 s (1:01:24), record `107-…md`; `Suite accounting` and `Backend suite` rows updated; the plain `TESTED_SOURCE_SHA` row's "latest is X" pointer → `aea143bd`; **new** non-bold `Superseded tested source` row for `117fc172` keeping 3123/20/47 = 3190; the checkout block and its `must print` line → `aea143bd…` |
| `workbench/docs/PC_PILOT_HANDOFF.md` | `SHA ĐÃ KIỂM THỬ` → `aea143bd…`; the sweep sentence → 3126/20/47 = 3193, 3684.01 s, record `107-…md`; the failure-set sentence → **0 mới, 0 rơi, 20 y nguyên** against the `117fc172` sweep; the `git checkout` line → `aea143bd…` |

No old numbers were rewritten: `117fc172`'s 3123/20/47 = 3190 is preserved as a superseded row, and
`96d86010`'s 2939/18/47 = 3004 remains in the history rows.

Guard, after the advance:

```bash
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_handoff_commands.py \
  -q -p no:cacheprovider --basetemp=/tmp/pin_aea
# 8 passed in 14.01s
```

The count predicate holds: all three bold `**N passed, M failed**` lines in the verdict read
`3126 passed, 20 failed`, the handoff's one reads the same, the first `skipped` match in each is `47`,
and `3126 + 20 + 47 = 3193` equals what the pinned revision's own test tree collects. The
supersession guard is green because both the checkout line and the `must print` line in **every**
release document now name the pin — the extension from `a0c08c0` is what makes that a check rather
than a coincidence.

## 5. Safety

No GPU job, no checkpoint load, no forward pass. Live units untouched. `/tmp/cir_gpu.lock` never read for
a decision and never deleted. No `git add -A`; `CLEANUP_REPORT.md.save` never staged. Only this
workstream's own sweep scratch was removed from `/tmp`.
