# Sweep at the 20261113 acceptance tree: 8 failed / 3272 passed / 47 skipped

The authoritative serial sweep for the 20261113 acceptance campaign. One run, one process, the tree
stationary for the whole thing. Its result is what `FINAL_PC_RELEASE_VERDICT.md` and
`PC_PILOT_HANDOFF.md` quote, and what `TESTED_SOURCE_SHA` was advanced to.

## 1. Preflight

| check | value | verdict |
|---|---|---|
| `git rev-parse HEAD` | `b4d25c70382b193ae20be59e52e163a0ea4c94ad` | the revision the worktree was based on; the campaign's changes were uncommitted when the sweep ran |
| `git rev-list --left-right --count origin/...HEAD` | `0 0` | nothing committed, nothing unpushed |
| `git status --porcelain` | 76 modified, 200 untracked | the campaign's work, all of it |
| `pgrep -af 'pytest\|composition_eval\|gpu_job\|clvc_\|third_model_parity'` | empty | PASS |
| `nvidia-smi --query-compute-apps` | empty | PASS — no GPU job in this campaign |
| `workbench/frontend/dist/assets` | `js=1` (`index-CZlMxyEn.js`, 399 247 B), `css=1` (`index-DecQnta9.css`) | PASS |
| frontend freshness (`dist/index.html` ≥ newest `*.ts`/`*.tsx`) | dist `03:56:19`, newest source `ModelInventory.tsx` `02:13:57`, 0 newer | FRESH |
| `df -h /` free | **54 GiB**, raised to **84 GiB** | PASS (floor 45 GiB) |
| stale `/tmp/sweep_basetemp_*` | `/tmp/sweep_basetemp_cf6ba468` — **31 GiB**, removed | PASS after cleanup |
| `--collect-only` | **3327** | reference for accounting |

## 2. The sweep

```
mkdir -p /tmp/sweep_basetemp_b4d25c70
setsid nohup workbench/envs/ctl312h/bin/python -m pytest workbench/tests \
  -q -p no:cacheprovider -rfE --tb=short --basetemp=/tmp/sweep_basetemp_b4d25c70 \
  > /tmp/sweep_b4d25c70.txt 2>&1 < /dev/null &
```

| check | value |
|---|---|
| result | **3272 passed, 8 failed, 47 skipped** |
| wall time | **3473.27 s** (0:57:53) |
| exit | 1 (8 failures) |
| `grep -c '^ERROR '` | **0** |
| `grep -c "Errno 28"` | **0** |
| passed + failed + skipped | 3272 + 8 + 47 = **3327** |
| `--collect-only` count | **3327** |
| accounting verdict | **EXACT — 0 unaccounted** |

The tree was not edited while it ran: `CLEANUP_REPORT.md.save` was never read or modified, and
nothing was staged until the run finished.

## 3. Failure set — diff against the 20 ids in `20261112-parity-repair/SWEEP_cf6ba468.md` §4

| direction | count | ids |
|---|---|---|
| **new** (here, absent from the 20) | **0** | — |
| **gone** (in the 20, absent here) | **12** | see below |
| unchanged (still red) | **8** | see below |

**0 new failures.** The advance is allowed on that alone; the 12 gone ids are the campaign's repairs
landing, and each was predicted by `raw/ops-gaps/TASK2_TRIAGE.md`.

### Still red (8) — every one explained, none new

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

| id | class | assertion | why it stays red |
|---|---|---|---|
| `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | **order/state dependency** | `assert changed["models"][0]["status"] == "COMPLETED"` → `assert 'REUSED' == 'COMPLETED'` | one of the three "passes alone, fails in the full serial run" ids recorded by `107-sweep-aea143bd.md` §3.3: `_module_source_hash` hashes the **already-imported** module file, so a same-process `--resume` after an earlier test edited a fixture source sees different bytes in one run order and not another. Green in a targeted re-run (`TASK2_TRIAGE.md` §2), red in a full sweep. Order-sensitive, not a routing or identity defect. |
| `test_circus_tme_lifecycle.py::…[encoder]` | `ENVIRONMENT` | `RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_ENCODER_PYTHON=None` | `tests/conftest.py::isolate_operator_state` deletes every `WORKBENCH_*`/`CIR_*` variable so the suite runs as a fresh clone; refusing an unverified interpreter is the product guarantee (`adapters.py:325`), not a defect. |
| `test_circus_tme_lifecycle.py::…[limn]` | `ENVIRONMENT` | same, `WORKBENCH_LIMN_PYTHON=None` | as above. |
| `test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present` | `EXTERNAL` | `assert checks["external-assets:dcnet"]["status"] == "BLOCKED"` → `assert 'OK' == 'BLOCKED'` | the torchvision ResNet-50 weights are present on this host, so the doctor reports `OK` and no blocker names them. The product is right and the assertion is stale against a better host state; it is decided by a downloaded artifact the repo does not ship. |
| `test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers` | `EXTERNAL` | `assert any(reason.startswith("external runtime asset file missing: torchvision ResNet-50 …"))` → `False` | same asset. What still blocks a dcnet launch is reported and is a different blocker (`WORKBENCH_DCNET_PYTHON` unset). |
| `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | `ENVIRONMENT` | `assert PosixPath('/home/heheboiz/data/cir/data/FashionIQ') == PosixPath('…/third_party/DCNet') / 'dataset' / 'fashioniq'` | `prepare_dataset.prepare_canonical_link` created that symlink on this host, so `.resolve()` follows it; the link is gitignored (`third_party/DCNet/.gitignore`), i.e. host state by construction. On a clone with no staged link the test passes. |
| `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `PRE_EXISTING_BY_DESIGN` | `AssertionError: TESTED_SOURCE_SHA cf6ba468… does not contain the production code currently on disk …` | **red during the sweep by construction.** It asserts the recorded source contains every production byte on disk; the pin was still `cf6ba468` while production had moved. Green after the advance — the run that proves it is quoted in `LEDGER.md` L18's successor row and in the release-metadata table. |
| `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | `ENVIRONMENT` | `[BLOCKED] dataset link resolves to /home/heheboiz/data/cir/data/FashionIQ, expected /tmp/sweep_basetemp_b4d25c70/…/FashionIQ` | the same staged symlink. The test points the fixture at a temp dataset; the host's canonical link still resolves to the real tree. Missing thing: the absence of a staged link, which is the opposite of an installed deployment. |

### Gone (12) — the campaign's repairs, measured

```
test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[conesep]      fixture now runs the real base __init__
test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[ptha_mtst]   "
test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[conesep]                               "
test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[ptha_mtst]                            "
test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines                                           stub gained set/get_num_threads
test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass         J10 asserts the served bundle
test_release_blocker_gpu_records.py::test_offline_verdict_names_deployment_reason                          "
test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]  "
```

The four `blip` ids passed in the whole-file Phase-2 run (18/18) exactly as the fixture repair
predicted; the five `[<lambda>N]` ids are one test under five documented invocations, and all five
are green now that J10 asks whether the bundle was built from this source instead of grepping for
copy that `4e61f2f` deleted on purpose.

## 4. Production diff this sweep covers

Nothing: the sweep ran on the uncommitted worktree, so the diff it covers is
`git diff cf6ba468 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`, i.e. exactly the
15 production paths listed in `TASK2_TRIAGE.md` §7 plus the two test-side repairs and the six new
test modules. After the campaign's slices were committed, `git diff <pin> HEAD -- . ':(exclude)
workbench/docs' ':(exclude)workbench/tests'` is empty because the pin names the last commit that
touched production.

## 5. Safety

No GPU job launched, no checkpoint loaded, no forward pass. No systemd unit touched, no service
restarted — **deployment is pending**. No `git add -A`: every commit staged an explicit path list
and compared `git diff --cached --name-only` against it before committing. `CLEANUP_REPORT.md.save`
was never staged and never modified. Three untracked `population_habit_*.npz` (121.3 / 200.3 /
169.8 MiB) were deliberately left out of every commit; see the evidence commit's message.
