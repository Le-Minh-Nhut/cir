# Sweep at `82b0b3f`: 8 failed / 3283 passed / 47 skipped

The serial sweep that lets `TESTED_SOURCE_SHA` advance from `90f7b2c` to `82b0b3f`. One run, one
process, the tree stationary the whole time.

## 1. Why this revision exists

`82b0b3f` is `90f7b2c` plus one production change: the parity gate's cost. `/api/runs/parity-gate` is
fetched on every mount of the parity page, and deriving it cost 4.37 s of CPU and 480.6 MiB peak RSS.

1. `_component_scope_check` no longer builds a per-component `values` dict (~2.4M entries per ENCODER
   category; ~195 MiB and ~0.55 s) that was consulted at most five times and then walked again.
   `derived_max_abs`, `worst` and the offender list accumulate inside the loop that already computes
   `delta`; `offending_values` reads a bounded five-entry `shown_values`.
2. `evaluate_all` is memoised on `(str(base), requested_standard, _artifact_stamp(base))`, where the
   stamp is the sorted `(name, mtime_ns, size)` of every `g1_g2.*.json` under the root. One entry,
   `deepcopy` on return, no TTL. An absent/unlistable root raises and the caller derives uncached.

Cold 4.37 s → 2.90 s, peak RSS 480.6 → 270.4 MiB, warm 4.2 s → **1.5 ms**. No `main.py` change.

**Verdict bytes unchanged**, re-verified against the pre-change module loaded from `/tmp` before the
commit: `md5(json.dumps(payload, sort_keys=True))` = `da701905419b8b9833966f637ac521a8` for old,
`evaluate_all_uncached`, cold and warm alike; `evaluate_model('encoder')` =
`2edf794cf460cf14abab02b2ba48fafe` on both. Cache safety measured separately: deleting an artifact
changes the served payload and restoring it returns to baseline; a caller mutating the returned dict
cannot poison the cache; an absent root yields `PARITY_NOT_RUN`, not a cached empty answer.

## 2. Preflight

| check | value | verdict |
|---|---|---|
| `git rev-parse HEAD` | `82b0b3fb72e9181a8835c07a0974a1279f21aeef` | the revision under test |
| `git status --porcelain --untracked-files=no` | empty | PASS — stationary |
| untracked | `CLEANUP_REPORT.md.save`, three `population_habit_*.npz` | another lane's, untouched |
| `pgrep -af 'pytest\|composition_eval\|gpu_job'` | empty | PASS |
| `nvidia-smi --query-compute-apps` | header only | PASS — no GPU job |
| stale `/tmp/sweep_basetemp_*` | `/tmp/sweep_basetemp_90f7b2c` — **31 GiB, removed** | PASS (53 → 84 GiB free) |
| `df -h /` free | **84 GiB** | PASS (floor 45 GiB) |
| `--collect-only` | **3338** | reference for accounting |

## 3. The sweep

```
mkdir -p /tmp/sweep_basetemp_82b0b3f
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_82b0b3f \
  > /tmp/sweep_82b0b3f.txt 2>&1 < /dev/null &
```

| check | value |
|---|---|
| result | **3283 passed, 8 failed, 47 skipped** |
| wall time | **3969.99 s** (1:06:09) |
| exit | 1 (8 failures) |
| `grep -c '^ERROR '` | **0** |
| `grep -c "Errno 28"` | **0** |
| passed + failed + skipped | 3283 + 8 + 47 = **3338** |
| `--collect-only` count | **3338** |
| accounting verdict | **EXACT — 0 unaccounted** |

The collected count grew 3330 → 3338, by the **8 new memo/stamp `test_` functions** in
`test_parity_absolute_relative_v2.py`, and the passed count grew 3275 → 3283 by the same 8. No other
count moved.

## 4. Failure set — diff against the 8 ids in `SWEEP_90f7b2c.md` §4

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
(red **by construction** during a sweep whose pin is still the previous revision). No new failure.

## 5. Production diff this sweep covers

```
git diff 90f7b2c 82b0b3f -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
```

is `workbench/backend/parity_gate.py` alone; the new tests and the cost evidence are outside the
guard's path set.

## 6. Safety

No GPU job, no checkpoint loaded. No service restarted by this sweep, no systemd unit written,
`workbench/config/workbench.env` never touched, `CLEANUP_REPORT.md.save` neither read nor modified, the
three `population_habit_*.npz` never staged. The commit staged 11 explicit paths and compared
`git diff --cached --name-only` first; no `git add -A`.
