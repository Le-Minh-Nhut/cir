# Rerun impact set — what the lane-A fixes invalidate, and in what order to re-measure

Analysis only. No GPU job, no model run, no evaluation, no full suite. Produced by reading
`workbench/artifacts/composition/`, the interpreters, and the source at
`ffb6ae8a072f4f92ddd203bbebac8fa2cb4e3702`.

## 0. The finding that dominates this document

**Every CIRCUS run on this machine predates the BLAS pin. Measured, only `encoder` is actually
affected in its numbers.**

Evidence for the timing:

* `git log -S blas_threads_pinned` → introduced by **`bf6a448`**, committed
  **2026-11-08 20:33:11 +0700** ("Parallelise scoring across groups and pin BLAS to one thread").
* Latest composition summary on disk: **2026-11-08 17:42** (`conesep`). Summaries written after
  20:33: **0**.
* A real identity block (`runs/37ee72aa…/ranks/intent_fiq_n02.json`) has **no**
  `blas_threads_pinned` key.

So all 36 non-synthetic summary rows came from a runner that never pinned, and the machinery
`bf6a448` added has never governed a published CIRCUS number.

### Whether that matters is MEASURED, not inferred from which contraction is in the source

An earlier version of this document argued from the source: "only `encoder` and `limn` use `@`, the
rest use `einsum`, therefore only two can be affected". That reasoning was wrong in both directions
and has been replaced by measurement. Two corrections it missed:

1. **`optimize=True` does reach a threaded path.** `model_conesep.py` / `model_ptha_mtst.py` use
   `np.einsum("nd,mdk->nmk", …, optimize=True)`, which NumPy optimises into a tensordot-style
   contraction and runs ~2.1× faster at 20 BLAS threads than at 1 (18.2 → 8.5 ms/iter measured).
   So "einsum therefore no BLAS" was false.
2. **Being threaded does not imply the result changes.** Despite that speedup, the einsum scores came
   back **bit-identical** across thread counts.

The real mechanism, isolated on this host: a **batched** GEMM (M > 1) is thread-variant, a
single-query matrix-vector product is not.

```
8x640 @ 640x8192   identical=False  max|delta| 4.20e-05     <- batched, varies
1x640 @ 640x8192   identical=True   max|delta| 0.00e+00     <- single query, invariant
4x640 @ 640x8192   identical=False  max|delta| 4.01e-05     <- batched, varies
```

`np.linalg.norm(..., axis=1)` was checked separately and is thread-invariant here (bit-identical), so
the norm is not the source. `encoder` batches 8 queries against an 8192-row gallery chunk → varies.
`limn` loops one query at a time → invariant.

### Measured sensitivity table

Probe: each published adapter's own `score()`, synthetic float32 inputs of the shapes it validates,
gallery 3000, 8 queries, 6 planted near-tie gallery items per query (perturbed ~1e-7, a few float32
ulps), BLAS set to 1 then 20 through the runtime setter and restored after. Seed 20261109.
Rank changes are counted over each query's top-50.

| model | scorer expression | max abs(dscore) | ranks changed | max abs(drank) | tie clusters reordered | BLAS-sensitive? |
|---|---|---:|---:|---:|---:|---|
| conesep | einsum(optimize=True) → tensordot/dot | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | **no** |
| ptha_mtst | einsum(optimize=True) → tensordot/dot | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | **no** |
| habit | einsum (C loop) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | **no** |
| intent | einsum (C loop) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | **no** |
| hint | einsum (C loop) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | **no** |
| tme | einsum (C loop) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | **no** |
| **encoder** | matmul / norm (batched, M=8) | **1.788e-07** | **29 / 400** | **5** | **6 / 8** | **YES** |
| limn | matmul / norm (single query, M=1) | 0.000e+00 | 0 / 400 | 0 | 0 / 8 | no |

Restored to 20 threads after the probe; `blas_pin_capability` reported setter+getter present and
verifiable.

**Consequence:** the unpinned-BLAS defect can have altered numbers for **`encoder` only**. For the
other seven it is a provenance gap — their identity cannot prove a policy whose absence demonstrably
did not move their scores — not a numeric defect. Note the distinction from the earlier draft, which
wrongly called `limn` sensitive too: its scorer is thread-invariant, and its rerun is required for
coverage/audit reasons instead.

**Scope of this measurement:** synthetic inputs, one host, one OpenBLAS build. It establishes that
six scorers are numerically invariant to the thread count *on these shapes*, and that encoder's is
not. It is not a claim that every possible input is invariant.

## 1. Per-interpreter BLAS capability — measured, not assumed

`_openblas_symbols()` / `_blas_thread_limit(1)` from the **fixed** runner, executed inside each
interpreter with `OPENBLAS_NUM_THREADS`/`OMP_NUM_THREADS`/`MKL_NUM_THREADS` unset so the runtime
default is visible:

| interpreter | python | numpy | setter | getter | previous | effective | verified | source |
|---|---|---:|---|---|---:|---:|---|---|
| airknow-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | libopenblas64_p-r0-0cf96a72.3.23.dev.so |
| clip4cir-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| ctl312 | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| ctl312h | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| dl-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| encoder310-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| ilearn39-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| limn-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| limn310-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| pair-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| sprc-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| test-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| tme-adapted | 3.12 | 1.26.4 | yes | yes | 20 | 1 | **true** | same |
| **dcnet-prepare** | **3.7.12** | **1.21.6** | yes | yes | 20 | 1 | **true** (probed via plain ctypes; `runner.py` cannot be imported on 3.7 — see note) | libopenblas.so.0 |

**No interpreter fails to pin.** Every environment exposes `openblas_set_num_threads*_` and
`openblas_get_num_threads*_`, the setter takes effect, and the readback confirms it. All 13 pandas-era
envs share one NumPy/OpenBLAS build; `dcnet-prepare` is a genuinely separate Python 3.7 / NumPy 1.21
stack and still pins.

**The handoff review's BLAS probe result (`setter_found: false`, runtime stuck at 9) does not
reproduce on this host's interpreters.** It was measured in the reviewer's own runtime
(`/opt/codex/...`, OpenBLAS 0.3.30, `libscipy_openblas64_`). So on this machine the *capability* was
never the problem — the problem was that no artifact was produced by a runner that *used* it.

Note on `dcnet-prepare`: `runner.py` uses walrus operators and cannot be imported under Python 3.7.
That is a pre-existing compatibility fact about that env, not something my change introduced. DCNet
has no CIRCUS artifact and is out of the current cohort, so it does not block anything here.

## 2. Which published runs the fixes actually invalidate

Two different things are being fixed, and they have different consequences:

* **Numeric path changes** (can change a metric): the BLAS pin now applying on the backfill and
  fallback paths, and the verified pin itself. Only matters for **BLAS-sensitive scorers**.
* **Bookkeeping / provenance changes** (cannot change a metric): identity v2, `blas_pin_capability`
  in the identity, cache content digests, `scoring_impl` provenance, `fallback_used`.

A run whose *numbers* were produced by an unchanged arithmetic path does not need re-evaluation just
because its provenance record would now look different. Conversely, a run whose arithmetic path was
unpinned where it mattered does need a rerun regardless of how good its bookkeeping is.

| model | run id | export | measured BLAS-sensitive | numbers moved by my fixes? | verdict |
|---|---|---|---|---|---|
| **encoder** | `6a528e877c3287` | none | **YES** | **YES — 29/400 ranks move** | **RERUN_FULL + depth 100** |
| limn | `10173dc45dea8c` | none | no | no (scorer invariant) — rerun is for PARTIAL coverage + audit | **RERUN_FULL + depth 100** |
| conesep | `30f6de423861cc1` | depth 50 | no | **no** | **RERUN_FULL + depth 100** (export/provenance only) |
| ptha_mtst | `315dd590eeb03d2c` | none | no | **no** | **RERUN_FULL + depth 100** (export/provenance only) |
| habit | `497c25c57fbeba88` | depth 50 | no | **no** | **RERUN_FULL + depth 100** (export/provenance only) |
| intent | `37ee72aa298b276d` | depth 50 | no | **no** | **RERUN_FULL + depth 100** (export/provenance only) |
| hint | `4bc9893a977e5018` | depth 50 | no | **no** | **RERUN_FULL + depth 100** (export/provenance only) |
| tme | `ad477312aa57c1b9` | none | no | **no** | **RERUN_FULL + depth 100** (export/provenance only) |

Excluded: `habit e0dd04a46` (historical FAILED attempt, superseded), `encoder eda30a37` and
`limn 3f86f584` (8-query smoke runs), `released_audit` (imported release baseline, not a model run).

**Corrected counts: 8 models need a GPU pass. 0 export-only. 0 reusable as-is.**

Every one of the eight is `RERUN_FULL` because **no historical run can be backfilled** (§3,
fingerprint keying), not because all eight are numerically suspect. Only `encoder` is numerically
suspect; `limn` needs coverage/audit repair; the other six need a 100-deep export and a recorded
policy. "Rerun" for those six must not be read as "their numbers were wrong".

### Denominator warning, stated explicitly

limn's artifact records 5,884 of 6,003. The resolver now assigns 6,001 with `7:3038` and `7:3790`
unresolved. **6,001/6,003 is not 100%**, and no rerun may quietly adopt 6,001 as the benchmark size.
A rerun must publish 6,001/6,003 with the unscored list, or a separately approved denominator change.

## 3. Reachability: no historical run can be backfilled — every one is a full rerun

**The depth-100 backfill path is not addressable for any existing run.** Verified in code and by
computation, not assumed:

* `_cached_features` keys every memmap on `_digest({"run": fingerprint, …})` (`runner.py:145-152`).
* `fingerprint = _digest(identity)` (`runner.py:1512`).
* The identity now includes `blas_threads_pinned`, `blas_pin_capability`, `cache_verify_mode`,
  `query_scope_resolution` and `query_order_identity` (`runner.py:1485-1493`).
* Every historical artifact's recorded identity lacks all of those, so its fingerprint differs.

Computed on the real `conesep` artifact: recorded identity → fingerprint `30f6de4238…` (which is the
run id itself); adding the identity fields the current runner always writes → `cb92b4b4fd…`. The
per-file gallery cache key moves with it (`439d49b3…` → `4f5e49f0…`). `_load_memmap` therefore fails
its fingerprint check, `_cached_features` returns `{},{}`, and `--resume` refuses with
"feature cache is gone … re-run without --resume".

**So the `.npy` files that physically exist for `conesep`/`habit`/`intent`/`hint` are unreachable.**
Presence on disk is not usability.

**Consequence for the operator, stated plainly:** any change to an identity input orphans every
existing feature cache. That is the *correct* behaviour — a cache produced under a different
arithmetic policy must not be silently reused — but it means `--resume` cannot bridge an identity
schema change. Nobody should attempt `--resume --topk-depth` against a pre-`bf6a448` run expecting a
cheap export. The backfill code remains correct and valuable for *future* runs that need an export
added to a still-current run, and it is the thing that now fails closed on rank disagreement; it
simply has no historical application here.

### Corrected decision table

Every run below needs **one GPU pass with `--topk-depth 100`**, which produces clean ranks *and*
Top-100 together — strictly better than rerun-then-backfill.

| model | run | prior state | measured BLAS-sensitive | why it reruns | decision |
|---|---|---|---|---|---|
| **encoder** | `6a528e877c3287` | COMPLETED, no export | **YES** (29/400 ranks move) | numbers not reproducible | **RERUN_FULL + depth 100** |
| **limn** | `10173dc45dea8c` | PARTIAL 5884/6003, no export | no (single-query path) | PARTIAL coverage + audit conditions + export | **RERUN_FULL + depth 100** |
| conesep | `30f6de423861cc1` | exported depth 50 | no | depth 100 + recorded policy | **RERUN_FULL + depth 100** |
| ptha_mtst | `315dd590eeb03d2c` | no export | no | export + recorded policy | **RERUN_FULL + depth 100** |
| habit | `497c25c57fbeba88` | exported depth 50 | no | depth 100 + recorded policy | **RERUN_FULL + depth 100** |
| intent | `37ee72aa298b276d` | exported depth 50 | no | depth 100 + recorded policy | **RERUN_FULL + depth 100** |
| hint | `4bc9893a977e5018` | exported depth 50 | no | depth 100 + recorded policy | **RERUN_FULL + depth 100** |
| tme | `ad477312aa57c1b9` | no export | no | export + recorded policy | **RERUN_FULL + depth 100** |

**Corrected count: 8 models need a GPU pass. 0 are export-only. 0 are reusable as-is.**

### What "rerun" does and does not mean, per model

* **encoder** — the rerun exists because its numbers may genuinely differ. Measured: 29 of 400
  top-50 ranks changed between BLAS=1 and BLAS=20, max |Δrank| 5, and 6 of 8 planted near-tie
  clusters reordered. Previous encoder numbers are **not reproducible**, so treat them as superseded.
* **limn** — the rerun is required for its PARTIAL coverage (5,884/6,003) and its audit-condition
  defect. Its *scorer* measured insensitive (single-query matmul is thread-invariant), so BLAS is not
  the reason. Previous limn numbers are suspect only through the coverage/audit defect.
* **conesep, ptha_mtst, habit, intent, hint, tme** — the rerun is **for export depth and recorded
  policy provenance only**. Their numbers did not move under the measurement: max |Δscore| = 0.0,
  0/400 ranks changed. **"Rerun" must not be read as "their previous numbers were wrong."** They were
  produced by a scorer that is thread-invariant, so the unpinned BLAS cannot have altered them; what
  they lack is a 100-deep export and an identity that records the policy.

## 4. Recommended execution order (do not run yet)

All eight are full reruns with `--topk-depth 100`. Cheapest/most-informative first, one model at a
time, single GPU owner.

**Step 1 — encoder.** The only model whose published numbers the measurement says may have moved, so
it is the one that actually answers a scientific question rather than adding an export.

```
workbench/envs/ctl312h/bin/python -m workbench.scripts.composition_eval \
  --models encoder --subsets full,sf,validated --topk-depth 100
```

**Step 2 — limn.** Needs the most repair (PARTIAL coverage + audit conditions). Its preserved
`preserved_features/limn/` cache (654 MB) is **also unreachable** for the same fingerprint reason, so
do not budget on skipping its gallery encode.

```
workbench/envs/ctl312h/bin/python -m workbench.scripts.composition_eval \
  --models limn --subsets full,sf,validated --topk-depth 100
```

Before trusting limn's ablation numbers, verify the three condition inputs are genuinely different —
the old audit passed the same `paths`/`query_txt`/`category` three times.

**Step 3 — the six export-depth reruns**, batchable in one invocation each because their numbers are
not in question and only the artifact changes:

```
workbench/envs/ctl312h/bin/python -m workbench.scripts.composition_eval \
  --models conesep,ptha_mtst,habit,intent,hint,tme --subsets full --topk-depth 100
```

**Step 4 — nothing else.** The remaining seven active models (airknow, pair, clvc_net, dcnet,
combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir) have **no CIRCUS artifact at all** — new
integration work, not a rerun, outside this impact set.

### Denominator warning, unchanged

limn's artifact records 5,884/6,003. The resolver now assigns 6,001 with `7:3038` and `7:3790`
unresolved. **6,001/6,003 is not 100%**, and no rerun may quietly adopt 6,001 as the benchmark size.

## 5. What I deliberately did not decide

* **How far encoder's published numbers actually moved.** The measurement shows 29/400 top-50 ranks
  change on synthetic near-tie data between BLAS=1 and BLAS=20. Whether that translates to a visible
  metric delta on the real gallery needs the rerun; the probe establishes sensitivity, not the size of
  the real effect.
* **Whether the six invariant models need rerunning at all, on cost grounds.** Their numbers are
  demonstrably unaffected, so a rerun buys only a 100-deep export and a recorded policy. They cannot be
  backfilled (§3), so the choice is "pay for eight passes" or "accept depth-50 exports and an
  `unknown (predates identity v2)` label". That is a cost/benefit call, not a correctness one.
* **dcnet-prepare's Python 3.7 incompatibility.** Reported, not fixed: `runner.py` cannot be imported
  there. DCNet has no artifact, so nothing is blocked today, but any future DCNet run via the shared
  runner would hit it.
