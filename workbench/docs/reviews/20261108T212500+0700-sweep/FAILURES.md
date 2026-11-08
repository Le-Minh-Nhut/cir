# The 18 failures at `c2d8cdeb` — attributed, with the correction history

Serial sweep, no `-n`. Log: `full_suite_c2d8cdeb.txt` in this directory (header with revision,
command, interpreter, duration, exit code).

```
18 failed, 2463 passed, 47 skipped, 290 warnings in 3464.03s (0:57:44)
pytest_exit=1
```

2463 + 18 + 47 = **2528 collected**. HEAD was `c2d8cdeb0eabc3232b28070f55b5a10d63ccffcb` before and
after the run, so the log describes these bytes.

> **This file's first version was wrong, and an independent agent disproved three of its rows by
> re-running them.** It had mis-tallied its own categories, called two impossibility-by-construction
> failures "operator state", called a this-round ordering interaction "pre-existing", overstated the
> capability-record counts, and claimed a zero-regression result over a range it had not checked. The
> 18-id table was accurate throughout; the *attribution layer* was not. Every row below was re-verified
> by running the test or by re-reading the artifacts after that review. Where verification changed a
> category it says so. Where a category still cannot be established, the row says **unresolved** and
> names what would settle it.

## Categories

- **(a)** caused by a production or test change **inside `b0f38bd..HEAD`**
- **(b)** environment/host state (a real machine condition, not a test's own construction)
- **(c)** pre-existing at `b0f38bd` — reproduces when the range's changes are absent
- **(d)** test pins obsolete behaviour — the test's own assertion no longer matches intended behaviour
- **(h)** harness/test defect — the test guarantees its own failure by construction, independent of
  operator state

The range is stated on every `(a)`/`(c)` claim because **"this round" was ambiguous in the first
version**: I checked only `bf6a448`/`30f0c07`/`7dcc082`/`6eb4434` and then wrote "0 caused by this
round", while the recorded comparison range is `b0f38bd..HEAD` and `aa0f530` (also production, also
inside it) is the origin of two of these failures. The narrow claim was true; the broad one was not.

| # | test | category | evidence |
|---|---|---|---|
| 1 | `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_conesep-conesep]` | **(a)** in `aa0f530` | `AttributeError: 'Adapter' object has no attribute 'batch_size'`. `adapter.batch_size` is read at `model_conesep.py:183`, added by **`aa0f530`** — verified `git merge-base --is-ancestor aa0f530 b0f38bd` → **not** an ancestor, so it is inside the range. The test builds the adapter with `Adapter.__new__` and never sets it. Fails identically at `1a4cd32` (which is also inside the range), so it is not caused by the four commits I originally enumerated. |
| 2 | same, `[workbench.composition.model_ptha_mtst-ptha_mtst]` | **(a)** in `aa0f530` | same |
| 3 | `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]` | **(a)** in `aa0f530` | same |
| 4 | same, `[workbench.composition.model_ptha_mtst-ptha_mtst]` | **(a)** in `aa0f530` | same |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | **(a)** — ordering interaction, introduced in `9b6b5b6` | `assert 'REUSED' == 'COMPLETED'`. Passes alone (21 passed). Fails when `test_composition_topk_runner.py` runs first: `1 failed, 30 passed`. **`test_composition_topk_runner.py` does not exist at `b0f38bd`** (`git cat-file -e b0f38bd:…` → exit 128) — it was added by `9b6b5b6`, inside the range. The first version called this "(c) pre-existing, reproduced at `1a4cd32`": the reproduction was real but it cannot make the cause pre-existing, because at `b0f38bd` the interacting file is absent. Mechanism is test ordering, not either test's own content. |
| 6 | `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]` | **(a)** — test unsatisfiable by construction, inside the range | Four independent reasons, each proven; see "Why #6/#7 cannot pass" below. |
| 7 | same, `[limn]` | **(a)** | same, via `model_limn.py:352` → `:267` → `adapters.py:322` → `runtime.py:2309` |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | **(a)** in `aa0f530` | The first version said "missing `output.npy`, operator state". The file is missing *because the worker returned no output*: captured stdout is `CIRCUS_WORKER:{"ok": false, "error_type": "AttributeError", "error": "module 'torch' has no attribute 'set_num_threads'"}`. The test stubs `torch` as a `SimpleNamespace`; `adapters.py:182` now calls `torch.set_num_threads`, added by **`aa0f530`**. Verified `b0f38bd:model_tme.py` contains no such call (count 0). So: production change inside the range met a stdlib-only stub. |
| 9 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | **(b)** host state | Untracked symlink `workbench/third_party/DCNet/dataset/fashioniq -> /home/heheboiz/data/cir/data/FashionIQ`, created 13:43 by another agent, makes `.resolve()` return the real tree. **Proven**: passes in a clean worktree of `c2d8cdeb`. |
| 10 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | **(d)**, closed | It diffed the tree against the then-recorded `b0f38bd` while four production commits had landed. Closed by advancing `TESTED_SOURCE_SHA`: `8 passed` on re-run. |
| 11 | `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass` | **(b)** host state | `release_verify.sh` → `verdict=RELEASE_BLOCKED`. Cause: the shared, gitignored `workbench/artifacts/gpu_capabilities/` is polluted with probe records. **Exact breakdown of all 632** (recounted): **465** `probe_source: "override"`; **363** `device_name: "Synthetic GPU"`; **34** `error: "CUDA out of memory"`; **34** with *both* Synthetic GPU and CUDA-OOM; **329** Synthetic GPU with `error: null`; **50** with no `device_name`. The first version said "465 override *with* Synthetic GPU *and* CUDA OOM" — that conflated three counts. The conclusion (a polluted shared artifact directory, not this revision's code) stands on the corrected numbers, and on the fact that `release_verify` reports the same `RELEASE_BLOCKED` family at `b0f38bd`. |
| 12-16 | same file, `test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` | **(b)** | same root cause; the five invocations differ only in flag order |
| 17 | same file, `test_offline_verdict_names_deployment_reason` | **(b)** | same root cause |
| 18 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | **(b)** | `[MANUAL] dcnet: policy=AUTHOR_ARTIFACT_PREFERRED deterministic=NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT` — the DCNet symlink of #9 changes the recipe path. **Proven**: passes in a clean worktree (`2 passed` together with #9). |

## Why #6/#7 cannot pass — the traced mechanism

The first version called these "operator/machine state". The second called them "harness/test defect
that fails on a fresh clone". Both were understatements: **the test is unsatisfiable as written**, and
the trace below shows why exporting the variable cannot help.

### The raising path (encoder; LIMN is the same through `model_limn.py:352` → `:267`)

```
test_circus_tme_lifecycle.py:213  adapter.encode_queries(...)
model_encoder.py:214              self._run_job("queries", ...)
model_encoder.py:139              self.require_verified_interpreter()
adapters.py:322                   require_verified_model_interpreter(...)
runtime.py:2305                   verify_environment(model, probe=True, ...)
runtime.py:1604                   resolve_model_interpreter(...) -> None
runtime.py:1606                   EnvironmentReport(..., UNCONFIGURED, "UNCONFIGURED")
runtime.py:2309                   raise RuntimeError
```

### Reason 1 — the fixture sets an attribute the gate never reads

`test_circus_tme_lifecycle.py:181` does `adapter.interpreter = sys.executable` (LIMN `:190`). The gate
**re-resolves from the registry row** via `setting(python_env_var)`
(`adapters.py:322` → `runtime.py:1604`); it consults `adapter.interpreter` only *afterwards*, to compare.
The fixture never sets `_verified_interpreter` / `_interpreter_verified`, which is exactly what the
**TME fixture at `test_circus_tme_lifecycle.py:25-27` does** — and that one passes.

### Reason 2 — the autouse fixture removes the env the gate does read

`conftest.py:31-43` `isolate_operator_state` is `autouse=True` and `monkeypatch.delenv`s every `CIR_*`/
`WORKBENCH_*` name for every test. Proven directly: an untracked probe test printed
`WORKBENCH_ vars visible: {}` **with the variable exported**. Exporting it therefore changes nothing:
`WORKBENCH_ENCODER_PYTHON=…/encoder310-adapted/bin/python pytest '[encoder]'` produces the identical
`WORKBENCH_ENCODER_PYTHON=None` message.

### Reason 3 — the test patches stdlib `subprocess` globally

`test_circus_tme_lifecycle.py:209` does `monkeypatch.setattr(module.subprocess, "Popen", start)`.
`module` is the adapter module, and that module does `import subprocess` (`:10` of the test file shows the
same binding), so `module.subprocess` **is** the stdlib module: the patch is process-global for the test.
`runtime._run_probe` → `subprocess.run` → the patched `Popen`, so the gate's own identity probe receives
the test's `sys.exit(9)` child. Measured: `_interpreter_identity()` returns `None` inside the test and a
valid dict outside it; `clean identity: True / patched global subprocess.Popen is start: True / identity
after patch: None`.

### Reason 4 — the post-check rejects every interpreter the fixture could supply

`adapters.py:323-325` compares the gate's resolved interpreter against `adapter.interpreter` and raises on
a mismatch. With a working probe:

- `WORKBENCH_ENCODER_PYTHON=…/encoder310-adapted/bin/python` →
  `RuntimeError: encoder resolved interpreter changed: …/ctl312h/bin/python -> …/encoder310-adapted/bin/python`
- `WORKBENCH_ENCODER_PYTHON=…/ctl312h/bin/python` →
  `[BLOCKED/UNSAFE_INTERPRETER] cannot use active workbench environment`

There is **no value** that satisfies both the gate and the equality check in this fixture.

### Verdict

**(a) — a defect inside the delivered range.** These two tests **cannot pass on a fresh clone as
written**. Minimal fix, described and deliberately NOT applied in this pass: the fixture should pass a
`config` whose `setting()` resolves a real interpreter and set `_interpreter_verified` to it, and the
`subprocess` patch should be scoped to the adapter module's own reference (e.g. patch the module's
`_popen`/worker-spawn helper) rather than the stdlib module. **Loosening the gate would be wrong** — the
gate is doing its job; the fixture is lying to it.

## Tally (corrected; the first version's prose said "11" for a range naming 12 ids)

| category | n | ids |
|---|---:|---|
| **(a)** caused inside `b0f38bd..HEAD` | **8** | #1–#4 (in `aa0f530`), #5 (ordering, in `9b6b5b6`), #6–#7 (test unsatisfiable by construction, traced below), #8 (in `aa0f530`) |
| **(h)** harness/test defect | **0** | — (reclassified to (a); see the trace) |
| **(b)** host state | **9** | #9, #11–#17, #18 |
| **(c)** pre-existing at `b0f38bd` | **0** | — |
| **(d)** obsolete test (now closed) | **1** | #10 |
| **total** | **18** | |

8 + 0 + 9 + 0 + 1 = 18, which is both the table's row count and the log's failure count. The earlier
"11 operator-state / 1 pre-existing / 5 obsolete / 0 caused" split is withdrawn, and so is the interim
"6 caused / 2 harness" split.

**Release impact, stated plainly:** 8 of the 18 are defects inside the delivered range (6 production-
caused: #1–#4, #5, #8; 2 tests that are unsatisfiable by construction: #6, #7). **#6/#7 cannot pass on a
fresh clone** — not because of operator state, but because the fixture sets an attribute the gate never
reads, the autouse scrub removes the env the gate does read, the test patches stdlib `subprocess`, and
the post-check rejects the only interpreter the fixture could supply. None of them blocks the *numbers* this round recorded — the 18 failures do
not include any measurement path — but "operator state explains the failures" is false, and the earlier
note that said so is withdrawn.

## Unresolved

**#11–#17** are attributed **(b)** on the strength of the polluted directory and the fact that the same
`RELEASE_BLOCKED` family appears at `b0f38bd`. What would settle it definitively: moving
`workbench/artifacts/gpu_capabilities/` aside and re-running the seven tests — if they pass with an empty
directory, the cause is confirmed as probe pollution; if they still fail, the cause is in the verifier's
logic and the category becomes (a) or (c). Not done here because the directory belongs to another
agent's run and moving it was out of scope for a docs-only correction.

**#9 and #18** are marked proven because removing the symlink (a clean worktree) makes both pass. That
identifies the symlink as *sufficient*; whether it is the only trigger was not tested.

## Against the previous serial sweep at `b0f38bd`

| | `b0f38bd` | `c2d8cdeb` |
|---|---:|---:|
| collected | 2266 | 2528 |
| passed | 2186 | 2463 |
| failed | 33 | 18 |
| skipped | 47 | 47 |
| duration | 2211.38 s | 3464.03 s |
| exit | 1 | 1 |

Per-file breakdown of the 33, **recounted from the raw log** (the first version said 7/4/9/2 = 31):

| file | ids |
|---|---:|
| `test_pc_launcher.py` | **11** |
| `test_release_blocker_master_e2e.py` | **10** |
| `test_phase1_acceptance_scenarios.py` | **4** |
| `test_release_blocker_source_integrity.py` | **2** |
| `test_circus_adapters_blip.py` | 2 |
| `test_true_master_e2e.py` | 1 |
| `test_http_api_layer.py` | 1 |
| `test_handoff_commands.py` | 1 |
| `test_circus_adapters_rest.py` | 1 |
| **total** | **33** |

The 27 ids across `test_pc_launcher`/`phase1`/`master_e2e`/`source_integrity` are the ones that sweep
itself attributed wholesale to `workbench/config/workbench.env` leaking into sandboxed tests (its A/B
table). They pass here because `9b6b5b6` added the `isolate_operator_state` fixture — the same fixture
that, used by a test which *needs* a configured interpreter, is now the cause of #6/#7.

**Id-level overlap between the two sweeps is zero**: no id from the 33 appears among the 18.

**New tests:** `6eb4434` adds **30** `def test_*`; over `b0f38bd..HEAD` the suite grows **2266 → 2528 =
+262 ids**, measured by collecting both revisions. (The first version said "38 ids added this round",
which was a five-file subset; and the sentence before it said 46, which was simply unmeasured.) All new
ids pass.

The honest summary: **33 → 18 failures sharing no ids, and the current 18 break down as 6 inside the
range, 2 harness defects, 9 host state, 1 closed.** A smaller count is not evidence the delivered code
improved — the test-isolation change removed 27 operator-state ids while the range's own production
changes introduced 6 of the 18 that remain.

## The G6 prose-only warning (downgraded, not archived)

`INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md` §8 reports a decode baseline/after pair (21.2 → 2089.2
img/s) and the whole query-axis table as measurements. **None of it has a raw file** — verified by
grepping every `.txt`/`.json` under `20261108T-throughput/`: `21.2`, `2089.2` and `2192` each appear in
**zero** of them. They are therefore marked **prose-only, unverified** in §8.8 and in the G6 gate row.

What G6 *can* cite, from raw files: `batch.txt` (17.6/18.0/17.9/18.0 img/s at bs 64/128/256/512 —
saturated), `ein_threads.txt` (einsum flat at 6.5/6.0/6.2/5.6/6.2 q/s across 1/2/4/8/10 threads), and
`vdecode.txt` (512 real images, decode+transform only: serial 109.1 img/s → **682.7 at 4 workers**,
588.5 at 8, delta 0.0). **No raw file backs a full-pipeline throughput**, so G6 is PARTIAL for every
variant.

An earlier version of this paragraph claimed `query.txt` "covers the 512-query rank identity". It does
not: `query.txt` contains stage timings and forward throughput only, with no rank data. The real
full-coverage rank evidence is `d2_gate_full_6003x3.txt` (18,009/18,009 identical), which backs G3, not
G6. No number in this document is used for a release claim.
