# ops-gaps lane — report, 2026-11-13

Lane owner: operational gaps. Control interpreter `workbench/envs/ctl312h/bin/python` throughout.
No GPU, no `gpu_job.sh`. Shipped units on :8000/:5173 **never started, stopped or signalled** —
read only. Nothing committed; nothing staged. `workbench/config/workbench.env`,
`CLEANUP_REPORT.md.save` and the `g1_g2.*` artifacts untouched.

## Files I changed

| path | why |
|---|---|
| `workbench/backend/composition.py` | Task 1 fix: `run_conditions` three-state + `conditions_declaration` in both payloads |
| `workbench/tests/test_declared_empty_conditions.py` | **new**, 9 tests for Task 1 |
| `workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/**` | this lane's evidence |

I touched nothing else. The `batch_size` fixture defect (§1 of `TASK2_TRIAGE.md`) and the incomplete
`torch` stub (§4) are in files I do not own — reported, not patched.

## Task 1 — `run_conditions()` could not represent "declared empty"

**Reproduced first** (`task1-before-after.txt`, pre-fix module loaded from `git show HEAD:`):

```
BEFORE  field absent       run_conditions({})                   -> None    displayed -> (multimodal, text_only_black_image, image_only_empty_text)
BEFORE  declared empty     run_conditions({"conditions": []})   -> None    displayed -> (multimodal, text_only_black_image, image_only_empty_text)
BEFORE  declared non-empty -> ('multimodal','text_only_black_image','clvc_image_encoder_only')
        absent is empty -> True     <-- the conflation
```

**Fix** — three states, pairwise distinct, and the payload says which one it is:

```
AFTER   field absent       -> None  displayed (protocol triple)  conditions_declaration=UNKNOWN
AFTER   declared empty     -> ()    displayed ()                 conditions_declaration=DECLARED_EMPTY
AFTER   declared non-empty -> names displayed names              conditions_declaration=DECLARED
        absent is empty -> False
```

`conditions_declaration` is served on **both** routes that answer "which conditions": the run detail
(`decorate_summary`, beside `conditions_display`) and the per-query page
(`get_composition_run_queries`, beside `conditions`). `UNKNOWN` is what stops an empty list from
meaning two things; it is emitted, never inferred by the reader.

**Tests** — `workbench/tests/test_declared_empty_conditions.py`, **9 passed**:
`test_the_field_absent_is_unknown_not_empty`, `test_a_declared_empty_list_is_an_empty_tuple_not_none`,
`test_a_declared_non_empty_list_is_those_conditions`, `test_the_three_states_are_pairwise_distinct`,
`test_a_non_list_conditions_value_is_unknown_rather_than_empty`,
`test_the_detail_payload_serves_a_declared_empty_run_as_an_empty_list`,
`test_a_summary_the_reader_refuses_is_not_served_under_any_declaration`,
`test_the_queries_payload_serves_a_declared_empty_run_as_an_empty_list`,
`test_the_clvc_full_run_still_serves_exactly_its_three_conditions`.

**`da0a2162be0e159a1a456cec`, served verbatim (after the fix):**

```json
"conditions_display": [
  {"id": "multimodal",               "label": "Image + text"},
  {"id": "text_only_black_image",    "label": "Text only"},
  {"id": "clvc_image_encoder_only",  "label": "Image only (image encoder)"}
],
"conditions_declaration": "DECLARED",
"conditions": ["multimodal", "text_only_black_image", "clvc_image_encoder_only"]
```

Exactly `[multimodal, text_only_black_image, clvc_image_encoder_only]`, unchanged — the positive
control proving the fix does not move a real payload. (36/36 on-disk summaries are non-empty, so the
defect was latent, as the ledger said.)

## Task 2 — the 20 baseline ids

Full per-id table, assertion and verdict: **`TASK2_TRIAGE.md`**. Summary:

| class | ids | action |
|---|---|---|
| `REPRODUCIBLE_CODE_DEFECT` — in a **test fixture** | blip ×4 (§1), tme_precision (§4) | reported, not patched (files outside my ownership) |
| `ENVIRONMENT` | tme_lifecycle ×2 (§3), f08_dcnet (§6), prepare_model (§9) | none — host state the suite deliberately clears |
| `EXTERNAL` | doctor dcnet (§5), evaluate_models dcnet (§5), release_blocker_gpu_records ×7 (§8) | none — stale assertion vs. present asset (§5); J10 copy the verifier greps was deliberately deleted in `4e61f2f` (§8) |
| `PRE_EXISTING_BY_DESIGN` | handoff SHA guard (§7) | none — red because the tree is unpinned; clears when the lead advances `TESTED_SOURCE_SHA` |
| now green | circus_cli (§2) | none |

Re-run verdict: **19 failed, 45 passed** over the same selection; **18 of the 20 ids still fail**,
`circus_cli` is green, and the SHA guard is red for a *correct* reason.

### The `batch_size` attribution — verified, and it is NOT an adapter-contract gap

```
CompositionAdapter.__init__ sets batch_size: True
model_conesep:    __new__ has batch_size = False | real construction batch_size = 32
model_ptha_mtst:  __new__ has batch_size = False | real construction batch_size = 32
```

`adapters.py:273` assigns `self.batch_size = resolved_batch_size()` in `CompositionAdapter.__init__`,
and **every** `model_*.py` `__init__` calls `super().__init__(…)`. So no adapter fails to expose it.
The fixture does:

```python
workbench/tests/test_circus_adapters_blip.py:118
    adapter = module.Adapter.__new__(module.Adapter)   # __init__ never runs -> no batch_size
```

There is **nothing to close inside `composition.py`**. I deliberately rejected the one change that
*would* have made it green from production code — a class-level default
`batch_size = BATCH_SIZE_DEFAULT` — because `resolved_batch_size()` honours
`WORKBENCH_COMPOSITION_BATCH_SIZE`, and `batch_size` is a **measurement-identity** input (batch 1 vs
32 moves 1548/2017 dress ranks). A `__new__`-built adapter would then write a batch size the real run
never used, trading a red fixture test for a silently wrong identity. And I did **not** edit the test
to expect the breakage.

## Task 3 — shutdown, measured in isolation

Full detail, quoted properties and the limitation statement: **`TASK3_SHUTDOWN.md`**.

| | backend | frontend |
|---|---|---|
| `Type` | `simple` | `simple` |
| `ExecStop` | *(none declared)* | *(none declared)* |
| `KillMode` | `mixed` | `mixed` |
| `TimeoutStopSec` | `30` (`TimeoutStopUSec=30s`) | `10` (`TimeoutStopUSec=10s`) |
| `Restart` | `always`, `RestartUSec=2s` | `always`, `RestartUSec=2s` |
| `KillSignal`/`SendSIGKILL` | `15` / `yes` | `15` / `yes` |

Measured on isolated equivalents (same argv, cwd, `Environment=`, spare ports 8795/8796):

```
backend  SIGTERM  stop=170 ms  children=[]  port released  restart same port=YES
backend  SIGINT   stop=168 ms  children=[]  port released  restart same port=YES
frontend SIGTERM  stop=22 ms (npm)  node SURVIVES, HOLDS PORT 10 s  restart same port=NO
frontend SIGINT   stop=50086 ms (npm ignores)  child survives  port held  restart same port=NO
```

**The finding:** the backend's idle stop is 170 ms against a 30 s budget — comfortable. The
**frontend's `TimeoutStopSec=10` is fully consumed on every normal stop**: `npm` exits in 22 ms and
`node` (the vite listener) keeps :5173 until systemd's cgroup SIGKILL at the timeout. That is exactly
what `KillMode=mixed` exists for, and the margin is **zero** — a `node` that ignores SIGKILL would
exceed the timeout and the port would stay held. SIGINT is not a substitute for SIGTERM on the
frontend.

**What this does not prove:** it is not `systemctl stop` on the real unit. It measures the same process
tree receiving the same first signal systemd sends. Not exercised: the cgroup SIGKILL phase, cgroup
containment of the surviving `node`, `Restart=always` bookkeeping, and `TimeoutStopSec` enforcement.
The **worst case is still unmeasured**: no live job existed, so `GRACE_SECONDS=10` + 5 s SIGKILL +
`join(10)` = 25 s against a 30 s budget remains arithmetic (`jobs.py:325,404,433`), 5 s inside the
limit and not falsified.

**Cleanup proof** (both harness passes): ports 8795/8796 free, `pgrep`/`ss` show no instance of mine,
no transient `opsgaps-*` unit left, no temp lock or pid file behind. The one orphan I created during
harness development was killed and the ports re-verified free.

## Task 4 — historical SIGTERM sender

**`UNDETERMINED`.** Detail: **`TASK4_SIGTERM_SENDER.md`**. Two facts this pass adds:

1. The retained unit journal **cannot** answer it: the units' first entry is `2026-11-11T23:48:44`,
   hours *after* all four deaths (02:25:51, ~03:06, 03:53:06, 07:29:33) — the stack was a hand-launched
   `serve_workbench.py` then, so it left no unit journal at all. Absence is a coverage fact.
2. Both plausible service senders are excluded by measurement: `systemd-oomd` is `inactive` with zero
   entries 11-11→11-13, and `systemd-logind` logged no session close in any of the four windows.
   `auditd` is not installed, so no syscall audit trail exists or can be created retroactively.

What would settle it: an audit rule on kill-family syscalls before the next occurrence; or running the
stack as the already-written units so every stop becomes a journal line; or logging `signum` in
`forward_stop`. No culprit invented.

## Test counts

| command | result |
|---|---|
| `pytest workbench/tests/test_declared_empty_conditions.py -q` | **9 passed** in 1.46 s |
| `pytest test_declared_empty_conditions.py test_composition_api_contracts.py test_api_outcome_triple.py -q` | **28 passed** in 4.30 s |
| `pytest workbench/tests -k "composition or api or service" -q` | **4 failed, 391 passed, 1 skipped, 2927 deselected** in 461.46 s |

The 4 failures are the baseline `test_circus_adapters_blip.py` ids (conesep/ptha_mtst ×
`reuse_and_cleanup`/`failure_uses_shared_error_and_stays_failed`), id-for-id from
`SWEEP_cf6ba468.md` §4 — diagnosed in `TASK2_TRIAGE.md` §1 as a fixture defect, not a contract gap.
**No new failure from this lane's change**, and `test_declared_empty_conditions.py` is inside that
selection (it contributes 9 of the 391 passes).

## Changed paths, for explicit staging

```
workbench/backend/composition.py
workbench/tests/test_declared_empty_conditions.py            (new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/REPORT.md                (new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/TASK2_TRIAGE.md         (new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/TASK3_SHUTDOWN.md       (new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/TASK4_SIGTERM_SENDER.md (new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/task1-before-after.txt  (new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/task3-measure.sh        (new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/task3-signal-direct.json(new)
workbench/docs/reviews/20261113-final-acceptance/raw/ops-gaps/triage-20-ids.log       (new)
```

Nothing staged, nothing committed. `workbench/config/workbench.env`, `CLEANUP_REPORT.md.save`, the
`g1_g2.*` artifacts, and every NOT-mine source path are untouched by me.

## Defects found in paths I do not own (reported, not edited)

1. `workbench/tests/test_circus_adapters_blip.py:118` — `_resident_adapter` builds adapters with
   `__new__`, so `batch_size` (set in `CompositionAdapter.__init__`) never exists. One-line fixture
   fix; keeps 4 baseline ids red until then. **Do not** "fix" this with a class-level default: that
   would corrupt a measurement-identity input.
2. `workbench/tests/test_circus_tme_precision.py:90-96` — the fake `torch` module lacks
   `set_num_threads`, so `model_tme.py:370` `limit_torch_threads()` raises and the worker writes no
   `.npy`.
3. `workbench/docker/release_verify.sh:448` (J10) greps the built bundle for `phán quyết audit chuẩn`,
   which `4e61f2f` deliberately deleted from `App.tsx`; the copy now exists in neither source nor
   bundle, so J10 — and therefore the 7 `release_blocker_gpu_records` ids — cannot pass. Owner of
   `frontend/` + `release_verify.sh` must decide: restore the copy or retire the check.
4. `workbench/frontend/service` units: frontend `TimeoutStopSec=10` is consumed in full on every stop
   because `node` outlives `npm`. Not a defect today (SIGKILL frees it), but zero margin — worth a
   `ExecStop` that stops the vite child, or `KillMode=control-group`.
