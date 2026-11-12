# Sweep at `cf6ba468`: 20 failed / 3159 passed / 47 skipped — pin advanced to this revision

Third stationary sweep of the `20261112` parity-repair campaign, run on a tree that had stopped
receiving writes from every lane. It is the sweep that justified the pin advance, and that advance
**was made** by `58175d3` (`release: advance TESTED_SOURCE_SHA to cf6ba468 after a stationary
sweep`): `TESTED_SOURCE_SHA` is now `cf6ba468` in both release documents, and the release guard
went from `1 failed, 7 passed` (before the advance, measured while the pin still named
`aea143bd`) to **`8 passed, 0 failed`** (after).

**Verdict: the sweep is valid.** Zero collection errors, zero `Errno 28`, accounting exact at 3226.
The failure set is the **identical 20 ids** of `107-sweep-aea143bd.md` — no new failure, none gone.

---

## 1. Preflight

| check | value | verdict |
|---|---|---|
| `pgrep -af 'pytest\|composition_eval.py\|clvc_native_ranks\|gpu_job'` | empty | PASS |
| `nvidia-smi --query-compute-apps` | empty | PASS — all GPU capture lanes finished |
| `git status --porcelain` | `?? CLEANUP_REPORT.md.save` only | PASS — zero modified tracked files |
| `git rev-parse HEAD` | `cf6ba468af5b3a6f47174bdd0d50866f23b7006f` | the swept sha |
| frontend freshness (`dist/index.html` ≥ newest `*.ts`/`*.tsx`) | dist `23:22:36`, `gate.ts` `23:20:00`, 0 newer sources | **FRESH** — rebuilt by the UI lane, no rebuild needed here |
| `dist/assets` | `js=1` (`index-NnjrdZLG.js`), `css=1` (`index-DecQnta9.css`) | PASS |
| `df -h /` free | **86 GiB** | PASS (floor 45 GiB) |
| no stale `/tmp/sweep_basetemp_*` | none | PASS |
| `--collect-only` | **3226** | reference for accounting |

Trees checked before launch: the dist rebuild a UI lane performed is gitignored, so it does not
dirty the tracked tree and did not require a wait beyond its own completion, which had already
happened.

## 2. The sweep

```bash
mkdir -p /tmp/sweep_basetemp_cf6ba468
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_cf6ba468 \
  > /tmp/sweep_cf6ba468.txt 2>&1 < /dev/null &
```

Launched 2026-11-12T23:23, PID 2164384, finished 2026-11-13T00:30.

```text
20 failed, 3159 passed, 47 skipped, 350 warnings in 4037.65s (1:07:17)
```

| check | value |
|---|---|
| `grep -c '^ERROR '` | **0** |
| `grep -c "Errno 28"` | **0** |
| passed + failed + skipped | 3159 + 20 + 47 = **3226** |
| `--collect-only` count | **3226** |
| accounting verdict | **EXACT — 0 unaccounted** |

Raw log: `/tmp/sweep_cf6ba468.txt`. Collected moved 3193 → 3226 (+33: the new and rewritten parity
and composition tests from `cb0b228`, `23e5b8f` and the v2 lane).

## 3. Failure set — diff against the `107` sweep at `aea143bd`

| direction | ids |
|---|---|
| **new** (in this sweep, absent from the 20) | **none** |
| **gone** (in the 20, absent from this) | **none** |
| identical | **all 20** |

The set in full:

```
workbench/tests/test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]
workbench/tests/test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_ptha_mtst-ptha_mtst]
workbench/tests/test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_conesep-conesep]
workbench/tests/test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_ptha_mtst-ptha_mtst]
workbench/tests/test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation
workbench/tests/test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]
workbench/tests/test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[limn]
workbench/tests/test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines
workbench/tests/test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present
workbench/tests/test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers
workbench/tests/test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution
workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha
workbench/tests/test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass
workbench/tests/test_release_blocker_gpu_records.py::test_offline_verdict_names_deployment_reason
workbench/tests/test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0]
workbench/tests/test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>1]
workbench/tests/test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>2]
workbench/tests/test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>3]
workbench/tests/test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>4]
workbench/tests/test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps
```

(All five `[<lambda>N]` ids are parametrisations of one test.)

The expected member is present and explained: `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`
is **red during** the sweep by construction — it asserts that `TESTED_SOURCE_SHA` contains every
production byte on disk, and the pin is still `aea143bd` while production moved 400 lines. Its
assertion text is exactly that:

```
AssertionError: TESTED_SOURCE_SHA aea143bd8894881ad7d5b341c21dbeff25cfaaaf does not contain the
production code currently on disk: production files changed after the revision the release documents
```

It goes green after the advance, as it did in `107`. The other 19 are the documented baseline set
(`105-…md` §3.1-§3.2, `106-…md` §4): blip ×4, `circus_cli`, `tme_lifecycle` ×2, `tme_precision`,
`doctor`, `evaluate_models`, `f08_dcnet`, `release_blocker_gpu_records` ×7,
`standalone_operator_commands`. No test that pinned a pre-seed ENCODER value appears — the v2 lane's
rewrites hold.

## 4. Production diff this sweep covers

```
workbench/backend/composition.py | 117 +++++++++++----
workbench/backend/parity_gate.py | 317 +++++++++++++++++++++++++++++++++++++--
workbench/frontend/gate.ts       |  11 ++
3 files changed, 400 insertions(+), 45 deletions(-)
```

Against `aea143bd`, production only (`workbench/docs` and `workbench/tests` excluded, as the release
guard does). The pin advance named `cf6ba468`, whose tree these numbers describe; it was committed
as `58175d3`, so `git diff --stat cf6ba468 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`
is now empty and these numbers are readable from the pin itself.

## 5. Safety

No GPU job launched, no checkpoint loaded, no forward pass. No systemd unit touched. No
`git add -A`; `CLEANUP_REPORT.md.save` never staged and never read-modified. The tree was not edited
while the sweep ran.
