# 96 — the staged-path commit, and why the TG-CIR worker EXITED for conditions 2 and 3

Branch `research/cir-failure-workbench`, 2026-11-12. Control interpreter
`workbench/envs/ctl312h/bin/python`. Every GPU command ran under `workbench/scripts/gpu_job.sh` with
`MAX_WAIT=3600`.

Two separate defects produced the same symptom class in run `9f115afe`. They are different: the first is
fixed and committed, the second is fixed for the PROVEN case (the Popen-time job) and analysed but not
changed for the unproven one (§5). Run `9f115afe` reported `adapter_worker_exited ... exit_status=1` for
conditions 2 and 3 alike.

---

## 1. The staged-path fix, committed `be9de58`

`runner._stage_query_path` wrote the staged black replacement as `<qid>.png`. It now writes
`<Path(path).stem>.png`. The defect: an adapter may resolve a caption's category from its reference
image's stem, and TG-CIR does exactly that (`Corrections._candidates`, `model_tgcir.py:161-175`:
`(reference stem, raw text)` → raw text alone → stem alone). Replacing the stem with the qid destroyed
the only input that could scope those queries.

Measured, on the real release and the real correction dictionaries (no model, no GPU):

```
queries 6003
with qid-only (no stem): {'ok': 6000, 'unresolvable': 3}
with source stem       : {'ok': 6003}
```

The three stem-only queries, verbatim:

```json
{"qid": "7:3938", "text": "Is grey with a design on the back.", "stem": "B005PQ02G6"}
{"qid": "7:4723", "text": "Is an off the shoulder top.",        "stem": "B00A13DXIW"}
{"qid": "7:5123", "text": "Fades from red to orange.",          "stem": "B00BM9J4TG"}
```

Collision safety, which is why changing the name does not trade one bug for another: the staged image is
`inputs.black_image_like`, i.e. `Image.new(mode, size, 0)`, a function of the SOURCE's mode and size
alone. Two queries sharing a reference therefore write the same filename with the same bytes, so a second
write cannot change what the first query encoded. Both properties are pinned by
`workbench/tests/test_composition_topk_runner.py::test_the_staged_text_only_image_is_named_by_the_source_stem_not_the_qid`
and `::test_two_queries_sharing_a_reference_collide_safely`.

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_composition_topk_runner.py -q
12 passed in 0.49s
```

---

## 2. The second defect — the worker EXITS on a failed first job

### 2.1 The quoted failure, verbatim from the retry summary

`workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_9f115afeb58eb1adf75d88ba-retry-1/full.json`,
`condition_status`:

```json
"image_only_empty_text": {
  "category_scope_failure_count": 0,
  "reason": "AdapterWorkerError: adapter_worker_exited: model=tgcir phase=queries exit_status=1 log=/home/heheboiz/data/cir/workbench/artifacts/composition/runs/9f115afeb58eb1adf75d88ba/work/tgcir/tgcir/worker-d5caa664d6b448d290fe063f9f5a160c/worker.stderr.log",
  "status": "MODEL_UNAVAILABLE",
  "unscored_count": 6003
},
"multimodal": {
  "category_scope_failure_count": 0, "fallback_used": false,
  "status": "COMPLETED", "unscored_count": 0
},
"text_only_black_image": {
  "category_scope_failure_count": 0,
  "reason": "AdapterWorkerError: adapter_worker_exited: model=tgcir phase=queries exit_status=1 <same log path>",
  "status": "MODEL_UNAVAILABLE",
  "unscored_count": 6003
}
```

`query_failures` is `null` — NOT a per-query degradation. The whole condition was declared unavailable
with an identical reason, 6003 unscored each, which is what a latched worker failure looks like from the
outside.

The log path the summary names (`runs/9f115afe/work/tgcir/tgcir/worker-d5caa664…`) no longer exists; the
run's work files were cleaned. The surviving copy, quoted in §2.2, is
`workbench/artifacts/composition/logs/tgcir_tgcir_archive_unresolved/9f115afeb58eb1adf75d88ba-retry-1/worker.stderr.log`
(14 lines, 1040 bytes).

### 2.2 The traceback, verbatim

```
Traceback (most recent call last):
  File ".../runpy.py", line 197, in _run_module_as_main
  File ".../runpy.py", line 87, in _run_code
  File "/home/heheboiz/data/cir/workbench/composition/model_tgcir.py", line 667, in <module>
    _worker_main(Path(sys.argv[2]))
  File "/home/heheboiz/data/cir/workbench/composition/model_tgcir.py", line 653, in _worker_main
    run_one(job)
  File "/home/heheboiz/data/cir/workbench/composition/model_tgcir.py", line 616, in run_one
    corrected = corrections.correct(current["texts"], current["categories"], paths)
  File "/home/heheboiz/data/cir/workbench/composition/model_tgcir.py", line 198, in correct
    raise ValueError(
ValueError: tgcir caption text 3937 ('Is grey with a design on the back.') resolves to no category, so no correction dictionary can be chosen
```

`text 3937` is the position inside the 4096-path call window, and `7:3938` is that query — one of the
three in §1. So the staged-path bug made the worker raise; §2 is why a single raised job killed the whole
condition instead of costing one query.

### 2.3 The code trace, with the pre-fix line numbers

```
653	    run_one(job)                                    <-- NO try. A raise here leaves _worker_main.
654	    for line in sys.stdin:
655	        request = json.loads(line)
656	        if request.get("stop"):
657	            break
658	        try:
659	            run_one(json.loads(Path(request["job_path"]).read_text(encoding="utf-8")))
660	        except Exception as error:
661	            reply(False, error_type=type(error).__name__, error=str(error))
```

The asymmetry is the defect. The Popen-time job at 653 was the ONLY job in the protocol that ran outside
a `try`; jobs arriving on stdin at 658-661 were already guarded, and `run_one` itself already catches
everything around the forward pass and replies `ok=False` (650-651). So the loop was structured to
survive a bad job and the first job was not.

How that exit became a dead condition — `_exchange`:

```
278	        if self._worker_failed is not None:
279	            raise AdapterWorkerError(self._worker_failed)      <-- latch consulted first
286	        if self._worker is not None and self._worker.poll() is not None:
294	            self._worker_failed = (
295	                f"adapter_worker_exited: model=tgcir phase={phase} exit_status={status} ...")
297	            raise AdapterWorkerError(self._worker_failed)
312	        line = self._worker.stdout.readline()          <-- EOF, because the child died
317	            self._worker_failed = ( ... exit_status={status} ...)   <-- LATCHED
320	            raise AdapterWorkerError(self._worker_failed)
```

Sequence: first batch raises at 653 → process exits 1 → parent's `readline` returns `""` →
`_worker_failed` set (317) → every later batch, in every condition, short-circuits at 278 with the SAME
string. `_encode_queries` re-raises `AdapterWorkerError` unconditionally (`runner.py:713-714`), so the
condition dies as `MODEL_UNAVAILABLE` with 6003 unscored. `multimodal` was already COMPLETED and cached
before this, which is exactly the observed pattern: 1 of 3 conditions measured, 2 dead.

Answering the brief's question — *can a second job be served by a worker whose first job raised?* **No,
and the worker never gets the chance:** it is already gone. `_exchange` treats it as latchable, not
restartable (278-279), so the adapter does not retry with a fresh process either.

### 2.4 Reproduced, on CPU, no GPU, no weights

Two runs of the SAME harness, before and after the guard. The harness (`/tmp/repro/parent.py` +
`/tmp/repro/driver.py`) launches the real `model_tgcir._worker_main` from the real model interpreter
(`workbench/envs/tgcir39-la/bin/python`, `CUDA_VISIBLE_DEVICES=""` so it takes the CPU path), the
verbatim body lines 560-677, with ONLY the weights replaced: `workbench.replay.tgcir._load_tgcir`
returns a fake torch/clip and a fake model whose `extract_retrieval_*` return zeros, and
`_load_verified_model` returns that fake. The real `Corrections` is built from the real
`data/FashionIQ` dictionaries. Two jobs: job 1 is the exact defect trigger (caption of `7:3938` with the
pre-fix qid-named staged path), job 2 is the same caption with the real source stem.

**Before the guard:**

```
exit_status: 1
replies: 0
job2 output exists: False
=== worker stderr:
... model_tgcir.py, line 653, in _worker_main       run_one(job)
... model_tgcir.py, line 616, in run_one            corrected = corrections.correct(...)
... model_tgcir.py, line 198, in correct            raise ValueError(
ValueError: tgcir caption text 0 ('Is grey with a design on the back.') resolves to no category,
so no correction dictionary can be chosen
```

Identical to §2.2 modulo the call-window position. **Zero replies** — the parent got EOF, not an error
reply. Job 2 was never served and never could be.

**After the guard (`be9de58` + the guard):**

```
exit_status: 0
replies: 2
  {"ok": false, "error_type": "ValueError", "error": "tgcir caption text 0 ('Is grey with a design
   on the back.') resolves to no category, so no correction dictionary can be chosen"}
  {"ok": true, "error_type": null, "error": null,
   "corrected_first": "is grey with a design on the back",
   "corrected_digest": "66ba9ebf9db63afaa2a8b471bc36b59586ff0fc8a3e97a57c7197e68af3efc9f"}
job2 output exists: True
=== worker stderr: (empty)
```

The bad job replies with a named error; the next job is served normally, with a real correction and a
real digest. One caption costs one job, not the condition.

### 2.5 Verdict: design defect, not OOM

**A consequence of the worker's own error handling.** Deciding evidence, in order:

1. The stderr log holds a Python `ValueError` traceback — not a `MemoryError`, not a
   `torch.cuda.OutOfMemoryError`, not a signal-terminated process. An OOM kill leaves no traceback
   (SIGKILL) or a traceback naming an allocation; neither is present.
2. `exit_status=1` is Python's own exit code for an uncaught exception. A SIGKILL or an OOM-killer kill
   reports `-9`, and `_exchange` interpolates `poll()`'s returncode — it reported a positive `1`.
3. The three affected captions are resolvable by a pure dictionary lookup on the CPU with no model
   loaded (§1), which is how the failure was reproduced twice with no GPU and no weights.
4. Deterministic and reproduced: two runs, same traceback, same reply count.

The co-tenant-GPU/OOM hypothesis is refuted by (1)-(2), not merely unsupported.

---

## 3. The fix that shipped — first-job guard ONLY

`workbench/composition/model_tgcir.py`, one hunk at 653-663; the first job is guarded exactly like the
loop's jobs.

```python
    try:
        run_one(job)
    except Exception as error:
        # The FIRST job is the one `_exchange` gave the worker at Popen time, so it must fail the
        # same way the loop's jobs do: reply, then keep serving. Unguarded, a `Corrections.correct`
        # ValueError here propagated out of `_worker_main`, the process exited 1, the parent's
        # `readline` saw EOF and reported `adapter_worker_exited ... exit_status=1` -- and because
        # `_exchange` latches `_worker_failed`, every later batch in that condition got the latched
        # string, so one bad caption zeroed all 6003 queries (run 9f115afe, both text conditions).
        # `_run` turns the reply into the same per-query degradation any other failure gets.
        reply(False, error_type=type(error).__name__, error=str(error))
    for line in sys.stdin:
```

`git diff -U0 -- workbench/composition/model_tgcir.py` is exactly one hunk,
`@@ -653 +653,11 @@ def _worker_main(job_path: Path) -> None:` — guard only, no other semantics touched.

Deliberately NOT changed:

* `Corrections.correct`'s `ValueError` for a genuinely unresolvable or ambiguous caption. That is the
  correct signal; what was wrong was letting it kill the process. With both fixes in, no release query
  reaches it (§1: 6003/6003 resolve), so it is a guard, not a live path.
* `_exchange`'s latch and stop at 322-329. See §5 — analysed, reverted, not shipped.
* Scoring, `runner.py`, the gallery path.

New test,
`workbench/tests/test_tgcir_composition_parity.py::test_a_failed_first_job_replies_and_keeps_the_worker_serving`:
drives the real `_worker_main` with a fake model and `FlakyCorrections` whose first call raises; asserts
two replies, the first `ok=False` carrying the caption error, the second `ok=True`. Verified as a REAL
regression test — with the guard reverted by hand the test FAILS (the `ValueError` escapes
`_worker_main`); with it, it passes. Both states were observed.

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests/test_tgcir_composition_parity.py \
  workbench/tests/test_clvc_net_composition_parity.py \
  workbench/tests/test_composition_topk_runner.py -q
50 passed in 2.00s
```

---

## 4. Post-fix smoke — all three conditions COMPLETE

Command, exactly:

```
MAX_WAIT=3600 workbench/scripts/gpu_job.sh workbench/envs/ctl312h/bin/python \
  workbench/scripts/composition_eval.py --models tgcir --smoke 5 --topk-depth 100 \
  --subsets full --config ~/cir_rerun_20261109.yaml
```

| field | value |
|---|---|
| run_id | `1eee4c0adbc93f7708205a5e` |
| summary | `workbench/artifacts/composition/summary/tgcir_tgcir_archive_unresolved_1eee4c0adbc93f7708205a5e/full.json` |
| execution_status | **`COMPLETED`** |
| queries / gallery | 5 / 74381 |
| interpreter gate | `GPU_RUNTIME_READY`, `verified: true`, Python 3.9.25; torch 2.7.1+cu128 |

`condition_status`, verbatim — this is the field the defect showed as `MODEL_UNAVAILABLE`,
`unscored_count: 6003`:

```json
{
  "image_only_empty_text": {"category_scope_failure_count": 0, "fallback_used": false,
                            "status": "COMPLETED", "unscored_count": 0},
  "multimodal":            {"category_scope_failure_count": 0, "fallback_used": false,
                            "status": "COMPLETED", "unscored_count": 0},
  "text_only_black_image": {"category_scope_failure_count": 0, "fallback_used": false,
                            "status": "COMPLETED", "unscored_count": 0}
}
```

`topk_export`:

```json
{"conditions": ["image_only_empty_text", "multimodal", "text_only_black_image"],
 "depth": 100, "missing_query_count": 0, "partial": false, "reason": null,
 "requested": true, "status": "TOPK_EXPORT_COMPLETE"}
```

`metrics` (5 queries, so these are a liveness check, NOT a measurement):

| condition | scored | coverage | R@10 | R@50 | nDCG_full | MRR |
|---|---|---|---|---|---|---|
| `multimodal` | 5 | 1.0 | 20.0 | 40.0 | 0.18061210310965697 | 0.044481057352368764 |
| `text_only_black_image` | 5 | 1.0 | 0.0 | 0.0 | 0.10082312072434303 | 0.002228127690261614 |
| `image_only_empty_text` | 5 | 1.0 | 0.0 | 0.0 | 0.07693918292263288 | 0.00019029794641650075 |

`composition_gap_ndcg` is `0.4417698538002782` over `gap_cohort_count: 5`, `gap_unavailable_reason:
null` — so the gap is computable for the first time, where run `9f115afe` had
`"no queries have exact ranks in all three conditions"`. `shortcut_outcomes`:
`{"both": 0, "composition_required": 1, "image_only": 0, "neither_scored": 0, "text_only": 0,
"unresolved": 4}`; `unresolved: 4` is the 5-query sample, not a defect.

**What this smoke does and does not prove.** It proves the three conditions now run end-to-end and
that both text conditions are servable — the outcome `9f115afe` denied. It does NOT exercise the
first-job guard: with the staged-path fix in place no release query is unscopeable (§1), so the guard is
never entered at 5 queries or at 6003. Its evidence is the CPU reproduction in §2.4, which is the only
way to make the first job the bad one on purpose.

---

## 5. The latch question: analysed, NOT shipped

A second mechanism was found and deliberately **not** changed. Recording it because it is real and a
future reader will meet it: `_exchange` latches a job-level refusal just like a dead worker —

```
model_tgcir.py:322  if not response.get("ok"):
model_tgcir.py:325      self._worker_failed = (f"adapter_worker_job_failed: ...")
model_tgcir.py:328      self._stop_worker()
model_tgcir.py:329      raise AdapterWorkerError(self._worker_failed)
```

and `runner._encode_queries` re-raises `AdapterWorkerError` unconditionally (`runner.py:713-714`), so one
such reply zeroes the remaining queries of that condition.

`model_encoder.py:362-370` and `model_clvc_net.py:433-438` do exactly the same, so `model_tgcir` is not
an outlier — the behaviour is a three-adapter convention, not a TG-CIR bug. `clvc_net` only degrades per
query because its refusal is raised PARENT-side (`model_clvc_net.py:299-319`, `Refused(RuntimeError)`)
before any worker call, which is a different mechanism than the reply path.

**Why no change shipped:** the scenario needs a caption the worker refuses, and after §1 no release query
is refused (6003/6003 resolve). It was seen exactly once, in `9f115afe-retry-1`, and cannot be
reproduced now. A semantics change to `_worker_failed` justified only by a scenario that no longer
reproduces is a precaution, not a fix; it was reverted and the original latch/stop behaviour restored
verbatim. If a future condition can hand the worker an input it legitimately cannot scope, the fix is
already written down here: reply-with-error and keep serving for a LIVE worker, keep latching strictly
for a dead one.

## Còn chặn

1. **Full runs not started.** No `--subsets full` run of either model was launched from this section.
   §4 is a 5-query smoke; the fix removes the mechanism that killed conditions 2 and 3, and produces no
   aggregate metric.
2. **The fixes were not exercised at 6003 scale.** `text_only_black_image` and `image_only_empty_text`
   have never scored more than 5 queries for `tgcir`. §4 shows they are servable; it does not show the
   full-run aggregates.
3. **The first-job guard has no live-run evidence**, only the CPU reproduction (§2.4). No full-run path
   enters it while §1 holds.
4. **The latch behaviour described in §5 remains.** A live worker's job-level refusal still latches and
   still zeroes that condition. Not changed on purpose (unreproducible), and it is why §1 matters: as
   long as every query resolves, the latch is unreachable.
5. **`clvc_net`'s equivalent path was not re-checked.** The staged-path change is in shared
   `runner._stage_query_path` so `clvc_net` inherits it, but `clvc_net` does no stem-based caption
   resolution and its two conditions already completed before the change.
6. **No `browser`/API cross-check in this section.** Not requested here; 90 §0.3 owns that surface.

## Note on the `96-` filename prefix

Two files in this directory start with `96-`, written by two different workers in the same window:

* `96-staged-path-and-tgcir-worker-exit.md` — **this file**. The staged-path defect (`be9de58`), the
  TG-CIR worker-exit trace and CPU reproduction, and §5's unshipped latch analysis.
* `96-staged-path-identity.md` — NOT this file's author; it was created independently and is left
  untouched by this section.

They overlap only in that both touch the staging name. Neither renames the other.
