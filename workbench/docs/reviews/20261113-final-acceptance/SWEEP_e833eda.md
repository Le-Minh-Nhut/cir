# Sweep at `e833eda`: 8 failed / 3275 passed / 47 skipped

The serial sweep that lets `TESTED_SOURCE_SHA` advance from `43bfc9e` to `e833eda`. `e833eda` changes
production (`workbench/docker/acceptance.sh`) plus one test module, so the previous pin no longer
covered the tree on disk and the release guard said so. One run, one process, the tree stationary.

## 1. Why this revision exists

`acceptance.sh` could not pass on `43bfc9e`. Two of its groups asserted properties the tested
revision deliberately stopped having:

* **H** grepped source and bundle for `phán quyết audit chuẩn`, prose `4e61f2f` DELETED on purpose —
  the same defect `release_verify.sh` J10 had already been repaired for. H now mirrors J10: the
  bundle named by `dist/index.html` must exist and carry both labels `gate.ts` emits
  (`CAPTION THAY THẾ`, `LỆCH PAPER`).
* **C** asserted the served ids EQUAL the whole registry. `393de24` makes `/api/models` withhold the
  `web_publication.status: DEFERRED` entry, so the registry's 17 entries serve as 16 and the
  assertion was unsatisfiable. The expectation is now the registry minus its deferrals, and a
  deferred entry that IS served still FAILS.

Both are graded by new tests in `workbench/tests/test_final_hardening_regressions.py` (3 added, 48 in
the file, all green). The H test drives the group against a real `dist` copy and proves it fails for a
bundle missing a label, an `index.html` naming an absent bundle and an empty `dist/assets`.

## 2. Preflight

| check | value | verdict |
|---|---|---|
| `git rev-parse HEAD` | `e833eda6944bda1a6d7fc7045d3d38de9154a044` | the revision under test |
| `git status --porcelain` | 4 untracked only (`CLEANUP_REPORT.md.save`, three `population_habit_*.npz`) | PASS — no tracked edit during the run |
| `pgrep -af 'pytest\|composition_eval\|gpu_job\|clvc_\|third_model_parity'` | empty | PASS |
| `nvidia-smi --query-compute-apps` | header only | PASS — no GPU job |
| `workbench/frontend/dist/assets` | `index-CZlMxyEn.js` 399 247 B, `index-DecQnta9.css` | PASS |
| frontend freshness (`.ts`/`.tsx` newer than `dist/index.html`) | 0 | FRESH |
| `df -h /` free | **84 GiB** (stale `/tmp/sweep_basetemp_b4d25c70`, 31 GiB, removed first) | PASS |
| `--collect-only` | **3330** | reference for accounting |

## 3. The sweep

```
mkdir -p /tmp/sweep_basetemp_e833eda
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_e833eda \
  > /tmp/sweep_e833eda.txt 2>&1 < /dev/null &
```

| check | value |
|---|---|
| result | **3275 passed, 8 failed, 47 skipped** |
| wall time | **3608.26 s** (1:00:08) |
| exit | 1 (8 failures) |
| `grep -c '^ERROR '` | **0** |
| `grep -c "Errno 28"` | **0** |
| passed + failed + skipped | 3275 + 8 + 47 = **3330** |
| `--collect-only` count | **3330** |
| accounting verdict | **EXACT — 0 unaccounted** |

The 3 new tests are the only count delta against `SWEEP_b4d25c70.md`: 3327 + 3 = 3330, and the
passed delta is 3272 + 3 = 3275.

## 4. Failure set — diff against the 8 ids in `SWEEP_b4d25c70.md` §3

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

The classes are unchanged from that record §3 (order dependency, unset `WORKBENCH_*_PYTHON` under the
suite's operator-state isolation, host-provided torchvision asset, host-staged FashionIQ symlink, and
the pin guard, which is red *by construction* during a sweep whose pin is still the previous
revision). No new failure.

## 5. Production diff this sweep covers

`git diff 43bfc9e e833eda -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` is
`workbench/docker/acceptance.sh` alone; the test module is outside the guard's path set.

## 6. Safety

No GPU job. No checkpoint loaded. No service restarted by this sweep, no systemd unit written, no
`workbench/config/workbench.env` touch, `CLEANUP_REPORT.md.save` neither read nor modified. Both
commits staged explicit paths and compared `git diff --cached --name-only` first.
