# Sweep at `90f7b2c`: 8 failed / 3275 passed / 47 skipped

The serial sweep that lets `TESTED_SOURCE_SHA` advance from `e833eda` to `90f7b2c`. One run, one
process, the tree stationary for the whole thing.

## 1. Why this revision exists

`90f7b2c` is `e833eda` plus two commits, and it changes production:

| commit | path | why |
|---|---|---|
| `406c755` | `workbench/frontend/routes.ts`, `EvaluationPage.tsx`, `tests/model-param.test.cjs` | the live check found that `?view=native&model=pair` rendered no deferral notice: `model` was read straight off `window.location.search` while the location model neither parsed nor serialized it, so the landing-scope `replaceState` deleted it. `model` is now a `Location` field, and a URL naming a view but no area takes the area that owns the view (`?view=native` alone used to collapse to the dashboard). |
| `90f7b2c` | `docs/reviews/20261113-final-acceptance/raw/live-verify/**` | the repair's evidence: record §8 plus the three deep-link screenshots |

`workbench/frontend/dist/` is gitignored (`.gitignore:95`, 0 tracked paths), so the rebuilt bundle
`index-8klMGIWd.js` (399 398 B, sha256 `43b2618c56bf2a355dda893c774d8160ab64fcbb1df88623d84159baea284d87`)
is deployed but is not part of the tracked tree by design.

## 2. Preflight

| check | value | verdict |
|---|---|---|
| `git rev-parse HEAD` | `90f7b2c1e1f6b8474570961ae5a2e08ecf2ee36c` | the revision under test |
| `git status --porcelain --untracked-files=no` | empty | PASS — no tracked edit during the run |
| untracked | `CLEANUP_REPORT.md.save`, three `population_habit_*.npz` | another lane's, untouched |
| `pgrep -af 'pytest\|composition_eval\|gpu_job\|clvc_\|third_model_parity'` | empty | PASS |
| `nvidia-smi --query-compute-apps` | header only | PASS — no GPU job |
| `workbench/frontend/dist/assets` | `index-8klMGIWd.js` + `index-DecQnta9.css` | PASS |
| frontend freshness (`.ts`/`.tsx` newer than `dist/index.html`) | 0 | FRESH |
| stale `/tmp/sweep_basetemp_*` | **none present** (nothing removed) | PASS |
| `df -h /` free | **84 GiB** | PASS (floor 45 GiB) |
| `--collect-only` | **3330** | reference for accounting |

## 3. The sweep

```
mkdir -p /tmp/sweep_basetemp_90f7b2c
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_90f7b2c \
  > /tmp/sweep_90f7b2c.txt 2>&1 < /dev/null &
```

| check | value |
|---|---|
| result | **3275 passed, 8 failed, 47 skipped** |
| wall time | **3808.40 s** (1:03:28) |
| exit | 1 (8 failures) |
| `grep -c '^ERROR '` | **0** |
| `grep -c "Errno 28"` | **0** |
| passed + failed + skipped | 3275 + 8 + 47 = **3330** |
| `--collect-only` count | **3330** |
| accounting verdict | **EXACT — 0 unaccounted** |

The collected count **did not move** from `SWEEP_e833eda.md` (3330 → 3330). The new
`workbench/frontend/tests/model-param.test.cjs` is a **node** test run by `node --test`, not collected
by pytest, and the frontend repair added no pytest case. Confirmed both ways: `--collect-only` says
3330 before and after, and this run accounts for exactly 3330.

## 4. Failure set — diff against the 8 ids in `SWEEP_e833eda.md` §4

| direction | count |
|---|---|
| **new** | **0** |
| gone | 0 |
| unchanged (identical id set) | **8** |

```
workbench/tests/test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation
workbench/tests/test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]
workbench/tests/test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[limn]
workbench/tests/test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present
workbench/tests/test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers
workbench/tests/test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution
workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha
workbench/tests/test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps
```

Same classes as before: order dependency, unset `WORKBENCH_*_PYTHON` under the suite's operator-state
isolation, the host-provided torchvision asset, the host-staged FashionIQ symlink, and the pin guard
(red **by construction** during a sweep whose pin is still the previous revision). **No new failure.**

## 5. Production diff this sweep covers

```
git diff e833eda 90f7b2c -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
```

is `workbench/frontend/EvaluationPage.tsx` and `workbench/frontend/routes.ts` — the frontend repair.
The node test and the evidence directory are outside the guard's path set.

## 6. Safety

No GPU job, no checkpoint loaded, no forward pass. No service restarted by this sweep, no systemd unit
written, `workbench/config/workbench.env` never touched, `CLEANUP_REPORT.md.save` neither read nor
modified, the three `population_habit_*.npz` never staged. Every commit in this cycle staged explicit
paths and compared `git diff --cached --name-only` first; no `git add -A`.
