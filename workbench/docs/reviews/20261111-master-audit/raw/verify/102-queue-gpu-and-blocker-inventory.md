# 102 — `gpu_may_be_held` through the real queue, and the launcher backoff

HEAD `82cb1044` (= `origin/research/cir-failure-workbench`), branch `research/cir-failure-workbench`,
2026-11-12. Control interpreter `workbench/envs/ctl312h/bin/python` (CPython 3.12.11). CPU only: every
fake job is a `python -c` / `sleep` shape under `/tmp/q102/`, **not** the live queue, and the live
systemd units on 8000/5173 were never touched, restarted, stopped or signalled. `/tmp/cir_gpu.lock` was
never read for a decision, never created, never deleted.

## Verdict

| question | answer | proof type |
|---|---|---|
| Is the `gpu_may_be_held` blocker kept or cleared? | **KEPT** (not cleared) | measured: a `setsid` evaluator survives a SIGKILL of its queue owner on a bare supervisor; the systemd unit reaps it, but only the *unit* path does, and the queue cannot see which path it is on |
| Does the detached child survive a hard kill of the queue owner? | **YES on a bare supervisor** (state `S`, alive after `SIGKILL`), **NO inside a systemd unit with `KillMode=mixed`** (dead at t+0.050 s, before the replacement starts) | two real-process experiments, §1 and §2 |
| Is the release path proven? | **YES**, but only as a *procedural* release, and that limit is stated in §4 | real queue: blocked → restart → accepted |
| Backoff assertion | **PASS**: `restart_delay(1025) == 8.0`, ladder `0.2/0.4/0.8/1.6/3.2/6.4/8.0` | §5 |

**No change to `workbench/backend/jobs.py`.** Nothing was reproduced that the guard gets wrong on the
path it is written for; §4 records one real defect *outside* the guard's own claim, which is reported
rather than patched because the fix belongs in `main.py`/`composition` (not owned here).

---

## 1. The scenario, on a bare supervisor: the child survives

Owner = a process that constructs the real `SingleGpuQueue`, enqueues one CPU job, and blocks once the
job is `running`. The job command forks a child that calls `os.setsid()` — the shape of the operator's
evaluator after a backend crash (`86 §cgroup containment`).

Harness: `/tmp/q102/fake_eval.py`, `/tmp/q102/owner.py`, `/tmp/q102/run_bare.py`. Log
`/tmp/q102/bare.log`:

```text
owner_pid=1444748
job_status=running child_pid=1444753
child_cmdline=.../ctl312h/bin/python /tmp/q102/fake_eval.py /tmp/q102/ws/child.pid /tmp/q102/ws/child.marker
child_cgroup=0::/user.slice/.../app-org.gnome.Terminal.slice/vte-spawn-8d9e615d-....scope
owner_cgroup=0::/user.slice/.../app-org.gnome.Terminal.slice/vte-spawn-8d9e615d-....scope
child_state=S child_pgid=1444753 owner_pgid=1444747
child_in_owner_session=False
owner_hard_killed_at=0.061 rc=-9
child_alive_after_owner_sigkill=True
child_state_after=S
```

So on a supervisor that is **not** a cgroup-owning systemd unit, the evaluator is alive and running
after the crash. This is the premise the guard exists for. Note what the record contains: §3 shows the
persisted `queue_state.json` has no pid, no pgid and no process group — so there is no id on disk that
could be proven to refer to this child, which is why "kill the id from the record" is not an option.

The child in this experiment was killed by the experiment itself, by the pid the child wrote for the
experiment, after checking `/proc/<pid>/cmdline` contains the experiment's own script path. No signal
in this document was ever addressed to an id read from a job record.

## 2. The same scenario inside a systemd user unit with the real `KillMode=mixed`

Transient unit, `systemd-run --user --unit=q102-probe-…` with the properties the real unit renders
(`workbench/systemd/cir-workbench-backend.service.in`): `Type=simple`, `KillMode=mixed`, `Restart=always`,
`RestartSec=1` (the real unit uses 2), `TimeoutStopSec=30`. It ran the same `owner.py`.

Containment before the kill — the `setsid` child is inside the unit's cgroup even though it left the
backend's session:

```text
unit=q102-probe-1794444580 main=1445451 child=1445460
child_cgroup=0::/user.slice/user-1000.slice/user@1000.service/app.slice/q102-probe-1794444580.service
unit_cgroup=/user.slice/user-1000.slice/user@1000.service/app.slice/q102-probe-1794444580.service
pre_kill child_alive=True child_state=S
sent_SIGKILL_to_1445451 at 2026-11-12T00:49:49.439974+00:00 t+0.000
child_absent_at t+0.050 (poll 1)
child_alive_final=False
```

Journal, verbatim (`journalctl --user -u q102-probe-1794444580`):

```text
07:49:49 systemd[1735]: q102-probe-....service: Main process exited, code=killed, status=9/KILL
07:49:49 systemd[1735]: q102-probe-....service: Killing process 1445457 (python) with signal SIGKILL.
07:49:49 systemd[1735]: q102-probe-....service: Killing process 1445460 (python) with signal SIGKILL.
07:49:49 systemd[1735]: q102-probe-....service: Failed with result 'signal'.
07:49:50 systemd[1735]: q102-probe-....service: Scheduled restart job, restart counter is at 1.
07:49:50 systemd[1735]: Started q102-probe-....service - ... owner.py ...
07:49:50 python[1445541]: replacement_start=2026-11-12T00:49:50.528184+00:00 mainpid=1445541 child=1445460 child_alive=False
```

The replacement's **first instruction** observed `child_alive=False`. The detached child was SIGKILLed
by systemd 0.05 s after the main process died, 1.09 s before the replacement existed. That reproduces
`86 §cgroup containment` on an independent unit (86 measured 23:46:07.15 / 23:46:07.37 / 23:46:10.69).

**Why this does not justify clearing the guard.** The proof is conditional on *being inside a
cgroup-owning systemd unit*. `jobs.py` cannot observe that: it has no handle on the unit, no
`ControlGroup` path, and `recover_interrupted` runs before any HTTP handler exists. The same code runs
under the supervisor (`workbench/scripts/serve_workbench.py`, the documented non-systemd entrypoint) and
under a plain container, where §1 shows the child lives on. Clearing the flag on a cgroup inference the
queue cannot verify would be the blind dispatch the guard was written to prevent, and the honest limit is
that the *mechanism* is proven while the *applicability* is not. The blocker stays; §3 proves the
refusal and §4 the release.

The transient unit was stopped and is gone: `systemctl --user is-active` → `inactive`, `reset-failed` →
"Unit … not loaded", and no `q102`/`fake_eval`/`owner.py` process remains (`pgrep` empty).

## 3. Blocked state, through the real queue, with the specific message

Crash → replacement (`/tmp/q102/replacement.py`, log `/tmp/q102/rep1.log`). The replacement calls
`SingleGpuQueue.recover_interrupted` + `adopt_recovered` — the exact startup path of
`main.py:52-70` — then enqueues a second CPU job.

```text
recovered=[('e96d8b6e-1fba-4f70-bd2c-a61b8dd0206e', 'failed', True)]
artifacts_byte_identical=True files=5
gpu_orphans=['e96d8b6e-1fba-4f70-bd2c-a61b8dd0206e']
dispatch_status=blocked
dispatch_error='a job recovered from a crash may still hold the GPU (e96d8b6e-1fba-4f70-bd2c-a61b8dd0206e); release the device and restart the workbench before dispatching'
job_command_ran=False
stderr_tail='[BLOCKED] a job recovered from a crash may still hold the GPU (e96d8b6e-...); release the device and restart the workbench before dispatching\n'
```

Four facts, each measured:

1. the record became `failed` **with `gpu_may_be_held: true`** (`jobs.py:502-541`);
2. the artifact tree is **byte-identical** across recovery — 5 files, sha256 compared with only
   `queue_state.json` removed from both sides:
   `artifacts/payload.bin fdeab9ac…1108`, `artifacts/result.json bc7043f7…c183`,
   `queue_command.json 24e5806e…ef44`, `queue_stderr.log e3b0c442…b855` (empty),
   `queue_stdout.log 95f376d9…07cd`;
3. the dispatch is `blocked` with the message naming the job id, and the job command **did not run**
   (`ran-*` marker absent);
4. the refusal is also in `stderr_tail`, so it is visible through the status API like every other named
   refusal (`jobs.py:553-573`).

Same result with the detached child still alive (`/tmp/q102/restart_clears.log`, `child_alive_during_restart1=True`),
so the block does not depend on the child having died.

## 4. Release path — proven, and its real limit

`/tmp/q102/rep2.log`, second startup over the same artifacts root:

```text
recovered=[]
gpu_orphans=[]
dispatch_status=completed
dispatch_error=None
job_command_ran=True
```

The release works because the recovered record is already terminal, so a later startup recovers nothing
and adopts nothing into `_gpu_orphans`; the next dispatch is accepted. Two things must be stated
precisely:

* **The release is procedural, not verified.** `/tmp/q102/restart_clears.log` ran the second restart with
  the orphan **still alive** (`ORPHAN_STILL_ALIVE=True`, `child_alive_during_restart2=True`) and the
  dispatch was **accepted** (`dispatch_status=completed`, `job_command_ran=True`). So "release the device
  and restart" is enforced only by the operator's honesty; the code cannot tell a free device from an
  occupied one. This is a real weakness, and it is the same weakness the guard already documents ("not
  self-clearing, because a self-clearing flag would be the same blind dispatch with extra steps",
  `84 §3 Finding 6`). It is **not** a regression and **not** something `jobs.py` can fix without a device
  probe, which needs the GPU. Recorded as a limitation, not fixed.
* **The flag is never cleared by a signal.** `test_the_flag_survives_further_restarts_and_is_never_self_clearing`
  asserts the persisted record keeps `gpu_may_be_held: true` and `status: failed`, and that a second
  `recover_interrupted` returns `[]` without touching it.

## 5. Backoff assertion — present, not re-broken, not re-patched

`workbench/scripts/serve_workbench.py:103-135` is the fixed form: `BACKOFF_BASE = 0.2`,
`BACKOFF_CAP = 8.0`, `BACKOFF_SATURATING_EXPONENT` derived as
`next(e for e in range(64) if BACKOFF_BASE * 2 ** e >= BACKOFF_CAP)` → `6`, and `restart_delay` clamps
the exponent **before** the multiply. Measured (`/tmp/q102/backoff.log`):

```text
python 3.12.11
BACKOFF_BASE 0.2 BACKOFF_CAP 8.0 SATURATING_EXPONENT 6
ladder(1..7) [0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 8.0]
clamp(0,-1)=0.2 ok
saturation at 8,9,1025,1e6,1e12 = 8.0 ok
restart_delay(1025) = 8.0
1000 calls at 1e6: 0.000156s
monotone 1..2000 ok; all <= cap: True
```

No `OverflowError`, no giant-int arithmetic. The ladder matches `84 §1` exactly.

## 6. Regression tests

`workbench/tests/test_job_gpu_guard_through_queue.py` — 4 tests, all CPU, ~0.8 s:

| test | asserts |
|---|---|
| `test_the_detached_evaluator_outlives_a_hard_kill_of_the_queue_owner` | the persisted record carries **no** pid/pgid/process-group; the child left the session and leads its own group; it is alive after `SIGKILL` of the owner |
| `test_guard_blocks_then_the_documented_release_accepts_the_dispatch` | both directions in one run: `failed` + `gpu_may_be_held`, byte-identical artifacts, `blocked` with the verbatim message, command did not run, child not signalled by the queue; then restart → `completed`, command ran |
| `test_the_flag_survives_further_restarts_and_is_never_self_clearing` | flag persisted, terminal record not re-recovered, flag never rewritten to false |
| `test_launcher_backoff_is_the_measured_ladder_and_saturates` | `restart_delay(1025) == 8.0`, ladder exact, saturating exponent == 6, monotone over 1..2000, < 1 s |

```bash
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_job_gpu_guard_through_queue.py \
  -p no:cacheprovider --basetemp=/tmp/queue_q102final -q
# 4 passed in 0.80s
```

The pre-existing `workbench/tests/test_job_recovery_ownership.py` (11 tests) was **not** modified.

---

## 7. Blocker inventory — every remaining blocker, precise label

Sources read for this table: `FINAL_REPORT_20261112.vi.md` §5/§6/§7, `FINAL_REPORT.vi.md` §5/§6,
`INVENTORY.csv` (17 rows), `COMPOSITION_BLOCKERS.vi.md`, `REMAINING_WORK.md`,
`HANDOFF_20261111.vi.md` §7, `CENSUS_20261111.vi.md` §1/§1.2, `BLOCKED_MODELS_VERIFICATION.vi.md` §3,
`INDEPENDENT_REVIEW_RESPONSE_20261111.vi.md` §17. Every row was re-measured here where the artefact is
on disk; where it is not, the row says which document is the only source.

### 7.1 Rows that are now **FALSE / STALE** (label must change)

| id | current label | what is actually true now | evidence |
|---|---|---|---|
| **R16(b)** — "two composition adapters have no encoder" (`REMAINING_WORK.md:41-52`, `COMPOSITION_BLOCKERS.vi.md` §3.1, `HANDOFF_20261111.vi.md:494-503`) | `BLOCKED` — `raise Refused` unconditional at `model_clvc_net.py:205`, `model_tgcir.py:177` | **CLOSED.** Both adapters now have `encode_queries`/`encode_targets` (`model_clvc_net.py:268,274`, `model_tgcir.py:354,360`), both delegate via `subprocess` (6 and 7 occurrences), and the unconditional refusal is gone — the surviving `raise Refused` sites are all *conditional* (`model_clvc_net.py:401,407,421,468,472,475`; `model_tgcir.py:527`). Both Full runs completed, which is impossible through an unconditional refusal. | `grep -n 'def encode_queries'`; the two Full summaries in §7.4 |
| **"TG-CIR Full does not exist"** — `FINAL_REPORT_20261112.vi.md:485`, `:488-525`, `:684` | `CHƯA ĐẠT` because "TG-CIR Full chưa tồn tại (chỉ có smoke 5 query)" | **FALSE.** `tgcir` Full `7b1fa85007bffdab931f7f19` exists: `execution_status COMPLETED`, `query_count 6003`, `smoke {label: null, limited: false, query_limit: null}`, all three conditions `COMPLETED`/`unscored_count 0`, `topk_export TOPK_EXPORT_COMPLETE` depth 100 missing 0, `composition_gap_ndcg 0.43853620852911723` over 6003 qids. | `workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json` (109 793 B, mtime 05:16); `raw/verify/98-tgcir-full-run.md` §2 |
| **"the rerun is still running"** — `FINAL_REPORT_20261112.vi.md:594-606`, `§0` lines 33-38 | "ĐANG CHẠY, chưa xong", PID `1366364` | **FALSE.** That process exited; its run produced `…_eab10106…-retry-2/full.json` (17 029 B, mtime 04:53:22), `execution_status PARTIAL`, and no `composition_eval.py` is running. | `ls` of the two summaries; `raw/verify/98-…md` §1 |
| **CLVC Full metrics cite the retry-0 summary** — `:580-581` | metrics read from `…_eab10106…/full.json` | the *same* numbers, but the completed attempt is `…-retry-2`. `:580-581` names the superseded attempt. | `raw/verify/98-…md` §1 |

### 7.2 Genuinely open blockers

| id | label (precise) | exact missing input / unproven step | what would close it |
|---|---|---|---|
| **PAIR-SRC** | `BLOCKED_MISSING_SOURCE_CLASSES` (not "blocked", not "mapping unverified") | Upstream `iLearn-Lab/ICASSP25-PAIR` **never** published two modules the checkpoint pickles require: `model_vssp` (`Encoder`, `Backbone`, `FeatureWiseAffine`) and `MHTransformer_dis` (`Transformer`, `TransformerLayer`, `MultiHeadAttention`, `MlpTransformer`). Verified here against a full clone at `/tmp/pair_up` (= `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`): `git rev-list --all --count` → **21**; the 21 commits are `4fa4de5, 800bde2, 5991d55, bca9871, f1cef93, a6f6566, 2540ddd, 816a340, 76fa92e, 4549718, 86f7dce, 3ed877c, 77cb056, 7b02e6a, 8ff9864, b33095c, 5d163ab, 16a3f47, fc61c67, 847c7a8, bfb0b98`; every path that ever existed (**31** distinct) contains no `model_vssp*` and no `MHTransformer_dis*` — the only match is `models/MHTransformer.py`, whose classes are `MlpTransformer, MultiHeadAttention, MultiHeadAttentionRev, TransformerLayer, Transformer`; `git grep -E 'model_vssp\|MHTransformer_dis'` over **all 21 revs** returns **0** hits. Evidence path: `raw/pair_absence_evidence.json` (`verdict: BLOCKED_MISSING_SOURCE_CLASSES`, `required_globals_present_as_module_or_package` all `false`), `raw/pair_globals_from_pickle.log`, `raw/pair_class_census_tree.log`, script `workbench/replay/pair_absence.py`. | Upstream publishing the four items in `what_upstream_must_publish`: the `model_vssp` module, the `MHTransformer_dis` module (distinct from `MHTransformer.py` — `why_public_mapping_is_not_proof` records `transformer_a` 2 layers × 4 heads vs public 1 × 8, plus five undeclared tensors `transformer_c/transformer_d/cross_t2i/weight_chanelImg/weight_chanelImg_t`), the training entrypoint that instantiates `model_vssp.Encoder`, and a statement of which category/split `pair-B1.pt` and `pair-B2.pt` each scores. Forbidden shortcuts, per that file: `strict=False`, dropping layers, treating a near-paper score as mapping proof. |
| **G1G2-15** | 15 models `NOT_RUN` for G1/G2 (13 active + 2 deferred `csmcir`, `sprc`) | No differential native↔adapter artifact. Only `encoder` and `limn` have a parity artifact, and both are `MANDATORY_SUBCHECK_FAILED` (`encoder.failure_count 16`, `limn` 3 `batch_1_vs_n.max_rel`), so `passed_models: []`. | One fixed-sample differential per model, tolerance declared before measuring, plus a GPU per model. `CENSUS_20261111.vi.md` §1 is the only source of the 15. |
| **RERUN-5** | 5 rerun cases: `clip4cir_rn50x4_noft` (roll-up only, `artifact_backed: false`), `airknow` (3 counted rows; `fiq_n05` has two conflicting counted rows `52.54661498980954` vs `52.924086629607395`, `variance_group` n=4 sd `0.26728186433775697`), `clvc_net` (see CLVC-NUM below), `pair` (4 rows B1/B2 × H_normal/H_rev), `habit` `e0dd04a4` (artifact says `FAILED` because the interpreter was `airknow-adapted` 3.11.15 ≠ 3.8.10; env now correct) | ENCODER is **not** in the 5 — canonical `acc9b20e…`. | rerun, or a canonical choice per model. |
| **CLVC-NUM** | two `counted: true` rows, same `model_id: clvc_net`, same `protocol_id: fashioniq_val_split`, each naming the other in `conflict_with`; **not resolved by recency** (`native_measurements.py` `_mark_duplicates` branch `# 3d`) | row A `29.970746517160162` from `workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json`; row B `30.02079703540087` from `raw/clvc_run3cat_gpu.json`. **Provenance traced here, and it changes the label:** the two files score the **same** checkpoint bytes (`provenance.source_commit` identical `bd9b6889…79cb`; `realpath` of A's `local`/`global` == B's `local_checkpoint`/`global_checkpoint`, `same_file=True` for all three categories), over the **same** protocol and **same** 2017/2038/1961 val queries. Every value in both files is an **exact integer count** over the val size (max residual < 2e-6 queries), and the deltas are one to two queries per category: dress R@1 137→138, R@10 592→594, R@50 1146→1145; shirt 131→132, 566→566, 1124→1126; toptee 154→154, 643→644, 1253→1255. So this is **not** a provenance or checkpoint difference — it is a **tie-break / ordering** difference of 1-2 queries. `per_query_published: false` in **both** files, so the disagreement cannot be localised from the artifacts. | A rerun is **warranted**, but not as a third number: the run must publish **per-query** ranks (`per_query_published: true`) so the 1-2 differing qids can be named, and the tie-break rule (`argsort` kind, `workbench/composition/topk.py:94,101` uses `kind="stable"`) must be recorded as part of run identity. A third macro number without per-query data would repeat the same unresolvable conflict. |
| **LIMN-2** | `7:3038`, `7:3790` cannot be resolved to a canonical category | Upstream preprocessor never wrote a garment-category key; both qids exist in **both** shirt and toptee (`7:3038` shirt MM rank 3 / text 88 / image 152, toptee MM 43 / 389 / 67; `7:3790` shirt 65 / 6891 / 2200, toptee 43 / 4730 / 4047 — `MODEL_LIMN_ENCODER_VERIFICATION.vi.md:57-63`). 119 ambiguous, 117 resolved, **2** remain (`:209`). | upstream category metadata. Not a code fix. |
| **UNIT-46** | 46 of 50 metric rows use a **magnitude-inferred** unit | `unit_declared 4`, `unit_sibling_key 2`, `unit_inferred 46`, `unit_not_declared 46`, over `blocks_with_metrics 50` (re-measured here at HEAD: `counts` identical to `358c3fc`). Row-level: 46 `inferred_from_magnitude` + 2 `declared` + 2 `sibling_key` = 50. Each row audited: 0 missed declarations. `*_pct` keys describe different quantities and are not macro-scale declarations. | fix the **producer** of each artifact to declare its own scale. Cannot be fixed at the API layer (`native_measurements.py:1158-1167`). |
| **SHA-PIN** | `TESTED_SOURCE_SHA` = `96d86010e85048d82f2dacf26e17d7b7206d31ab`, guard **red by design** | 14 production files moved past the pin (`git diff --stat 96d86010 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'`): `backend/jobs.py` 45, `composition/metrics.py` 7, `model_clvc_net.py` 525, `model_tgcir.py` 632, `runner.py` 218, `registry/auxiliary_assets.yaml` 24, `registry/models.yaml` 25, `registry/preparation_contracts.yaml` 12, `scripts/backfill_recall_at_1.py` 187, `scripts/serve_workbench.py` 9, `systemd/__init__.py`, `systemd/cir-workbench-backend.service.in` 33, `systemd/cir-workbench-frontend.service.in` 28, `systemd/render_units.py` 215 — 1805 insertions, 155 deletions. Guard failure text, verbatim: `TESTED_SOURCE_SHA 96d86010… does not contain the production code currently on disk`. **Not advanced here** — the lead schedules the stationary sweep. | Exact preconditions from the guard's own code (`workbench/tests/test_handoff_commands.py:437-530`): (1) `git rev-parse <sha>` resolves; (2) `<sha>` is an ancestor of HEAD (`merge-base --is-ancestor`); (3) `git diff --quiet <sha> -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests` exits 0, i.e. the pin contains **every production byte now on disk**; (4) `**TESTED_SOURCE_SHA** \| **\`<sha>\`**` is present in `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md`; (5) every `git checkout <rev>` in `workbench/docs/PC_PILOT_HANDOFF.md` is a fixed 7-40 hex revision whose `_production_tree` equals the pin's; (6) every `**\`<sha>\`**` in the five `RELEASE_DOCS` (`PC_PILOT_HANDOFF.md`, `PC_ONE_COMMAND_GUIDE.vi.md`, `FINAL_PC_RELEASE_VERDICT.md`, `FINAL_DEFECT_LEDGER.md`, `AUTOMATION_RUNBOOK.md`) resolves when it is a commit. Precondition (3) can only be satisfied by a sweep measured on a tree at or after the current production bytes. |
| **SIGTERM-SENDER** | historical SIGTERM sender **unidentified** | The receive path is known (`serve_workbench.py:152-166`, `forward_stop` → `os.kill(child.pid, SIGTERM)` per child → `raise_signal(SIGINT)`); `jobs.py` excluded (`start_new_session=True`), repo tests/scripts excluded, session reaping excluded. Four death signatures are committed: `raw/service-death-log{,-2,-3,-4}.txt` (PIDs 3691345 / 4040574 / 4062957 / 4082632), all `[WARN] npm exited with status -15`. | a sender identity. Cannot be closed by inference (`86 §"Historical SIGTERM sender"`: "that inference is for a future investigation, not evidence about the past"). |
| **ENV-CACHE-API** | `/api/models` serves a stale registry snapshot for the process lifetime | `main._MODELS_CACHE` is a module-level list with no key (`main.py:170-176`); `POST /api/jobs` recomputes `preflight_blockers` per request and does **not** go stale (`raw/verify/95-external-asset-unblock.md`). | patch in `main.py` (not owned here). Spec: key the cache on `(models.yaml path, st_size, st_mtime_ns)` like `registry._load_yaml_cached` already does, or drop the module-level list. |
| **FALLBACK-ENCODE** | per-query encode fallback can mix batch sizes within one condition **unrecorded** | `runner.py:700-709` (`_encode_queries`). For CLVC-Net this changes features (batch N ≠ batch 1, `87 §3b`), so a run that partly used the fallback is not comparable with one that did not. The **scoring** fallback *is* recorded (`condition_status.<cond>.fallback_used`). | record the fallback at the encode layer, or make it fatal for batch-sensitive models. Not patched. |
| **S/IMG-5x** | gallery encode `0.0138 s/img` in-run vs `0.0027 s/img` standalone, **~5× unexplained** | attributed to a 609 MB memmap flush but **not proven** (`96-staged-path-and-tgcir-worker-exit.md` §6.2: "Recording the discrepancy rather than explaining it away … Both readings are real"). Note the two completed Full runs now give independent per-image rates: `clvc_net` 0.0202 s/img (92 390 images / 1867.2 s) and `tgcir` 0.00603 s/img (92 390 / 556.9 s) — neither reproduces 0.0027. | per-scope instrumentation inside one run. |
| **CLVC-VRAM** | CLVC Full peak VRAM **never sampled** | the run predates the sampling window (`98-…md` §5). Only `tgcir` has GPU evidence (22 samples, util 97-100 %, 2172 MiB, 235.02-260.04 W). | one sampled CLVC Full. |
| **R1-ABSENT** | R@1 is **not computed**, not zero | `ConditionMetrics` declares only `recall_at_10`/`recall_at_50` (`workbench/composition/metrics.py:38-49`); `compute_condition_metrics` returns exactly those (`:60-78`); no `recall_at_1` in `workbench/composition/`. The one `recall_at_1_multimodal` is a fixture placeholder (`backend/composition.py:1470`, `r10 / 2.0`). | add R@1 to `metrics.py` and re-rank from stored `ranks/*.json`, or leave absent and never quote it as 0. |
| **STATS-QCOUNT** | `statistics.query_count` serves `6003` beside a per-condition denominator of `5` on a smoke | `backend/composition.py:1372` passes `len(subset_queries)` while the adjacent `conditions` block counts the 5 ranked qids. `statistics_whole_run.query_count` and run-detail `query_count` both say `5`. **OBSERVATION, not scored** (`97 §7(a)`). | use the ranked qid count in the `statistics` block. Not patched. |
| **SUBSET-PROV** | `subset_provenance` appears only after the list route warms the shared summary cache | cold detail 20 053 B has no key; later detail and the proxy fetch (20 841 B) have it. Cause: `read_summary_file` caches the parsed dict by path and hands the **same object** to every caller, and `list_composition_summaries` mutates it in place via `decorate_run_rows`. When present it is correct (`consistent: true`). | decorate a copy, not the cached object. Not patched. |
| **TG-FIRSTJOB-GUARD** | first-job guard **shipped**, proven only on CPU, **no real-run evidence** | guard at `model_tgcir.py:653` (hunk 653-663) shipped in `be9de58`; repro before `exit_status: 1, replies: 0`, after `exit_status: 0, replies: 2`. No real run reaches it while 6003/6003 qids resolve. The latch change was **analysed and reverted** (`96 §5`) and must stay reverted. | a real run that enters the guard. |
| **9REPORTS** | cross-check of 9 reports **incomplete** | the worker was killed; `raw/verify/42-*` does not exist; 6 places were fixed by hand (`FINAL_REPORT.vi.md` §6.1), the rest unverified. | redo the cross-check. |
| **SWEEP-18** | 18 pre-existing test failures, `NEW = 0` | baseline `full_suite_68ad75e.txt`; BLIP resident worker, circus CLI resume, TME lifecycle/precision, F08 DCNet config, handoff SHA, release-blocker GPU records, `prepare_model` recipe. | not this round's work, but still red. |

### 7.3 Count

**21 blockers inventoried**, of which **4 are stale labels that must be corrected** (§7.1: R16(b) now
closed, TG-CIR Full exists, the retry is no longer running, the CLVC metrics cite the superseded
attempt) and **17 are genuinely open** (§7.2). One of the 17 (`CLVC-NUM`) changed label through this
tracing: it is a tie-break/ordering difference of 1-2 queries on identical checkpoint bytes, **not** a
provenance difference — and that is what makes a per-query rerun the only closure.

### 7.4 Both Full runs, verified on disk at HEAD

| run | `execution_status` | `query_count` | `smoke.limited` | gap | file |
|---|---|---|---|---|---|
| `tgcir` `7b1fa85007bffdab931f7f19` | **`COMPLETED`** | 6003 | `false` | `0.43853620852911723` (cohort 6003) | `…/summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json`, 109 793 B, 05:16 |
| `clvc_net` `eab10106b2df8e97fb7a5ce5` `-retry-2` | **`PARTIAL`** | 6003 | `false` | `null` (`no queries have exact ranks in all three conditions`) | `…/summary/clvc_net_author_drive_unresolved_eab10106b2df8e97fb7a5ce5-retry-2/full.json`, 17 029 B, 04:53 |

CLVC is `PARTIAL` for a structural reason, not a failed metric: `image_only_empty_text` is
`ABLATION_UNSUPPORTED` (`text_model.py:84-86` tokenises to zero tokens and indexes `x[0][0]`), so no qid
can carry all three conditions and `metrics.py:126` builds an empty cohort. Both conditions that do run
score 5884/6003 (119 unscored, `coverage 0.980176578377478`), and `topk_export` is
`TOPK_EXPORT_COMPLETE_PARTIAL` with `missing_query_count 238` = 119 × 2.

---

## 8. What this document does **not** claim

* It does not claim the guard is wrong. It claims the guard's premise is real on the supervisor path
  (§1) and that the unit path reaps (§2), and that the queue cannot tell the two apart, so the refusal
  stands.
* It does not claim the release verifies the device. §4 proves the opposite: a restart clears the flag
  with the orphan still alive.
* It does not claim any GPU work. No job in this document touched a model, a checkpoint or a device.
* It does not advance `TESTED_SOURCE_SHA`, does not rewrite `FINAL_REPORT_20261112.vi.md`, and does not
  patch `jobs.py`, `main.py`, `composition/**`, `frontend/**`, `systemd/**` or `config/workbench.env`.
