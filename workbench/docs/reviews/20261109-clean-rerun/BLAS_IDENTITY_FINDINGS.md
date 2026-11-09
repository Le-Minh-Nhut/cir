# BLAS identity findings for the 2026-11-09 clean rerun

Written by the GPU lane. Evidence is the rank artifacts under
`workbench/artifacts/composition/runs/<run_id>/ranks/*.json`, compared query by query with
`rank_diff.py` in this directory. Nothing here is inferred from aggregate metrics alone.

## Per-query rank identity, pinned rerun vs pre-pin run

| model | condition | dR@10 | dR@50 | dndcg | changed ranks | max abs d | bit-identical |
|---|---|---:|---:|---:|---:|---:|---|
| encoder | multimodal | +0.016658 | -0.099950 | +2.069e-04 | 2819/6003 | 805 | no |
| encoder | image_only | 0.000000 | -0.016658 | +4.490e-05 | 5214/6003 | 852 | no |
| encoder | text_only | +0.166583 | +0.116608 | +8.297e-04 | 4694/6003 | 1706 | no |
| conesep | multimodal | 0.000000 | 0.000000 | -3.077e-07 | 3/6003 | 1 | no |
| conesep | image_only | 0.000000 | 0.000000 | -2.055e-09 | 10/6003 | 1 | no |
| conesep | text_only | 0.000000 | 0.000000 | +5.142e-10 | 4/6003 | 1 | no |
| habit | all three | 0.000000 | 0.000000 | +0.000e+00 | 0/6003 | 0 | yes |
| intent | all three | 0.000000 | 0.000000 | +0.000e+00 | 0/6003 | 0 | yes |
| hint | all three | 0.000000 | 0.000000 | +0.000e+00 | 0/6003 | 0 | yes |
| ptha_mtst | multimodal | 0.000000 | 0.000000 | +2.681e-08 | 12/6003 | 2 | no |
| ptha_mtst | image_only | 0.000000 | 0.000000 | +2.512e-09 | 20/6003 | 1 | no |
| ptha_mtst | text_only | 0.000000 | 0.000000 | +2.902e-08 | 14/6003 | 1 | no |
| tme | all three | 0.000000 | 0.000000 | +0.000e+00 | 0/6003 | 0 | yes |

`composition_gap_ndcg`: conesep 0.3942551173824983 -> 0.39425455009745813 (-5.673e-07);
encoder 0.47230210227676583 -> 0.469868464471065 (-2.434e-03); ptha_mtst
0.24362219946484542 -> 0.24362216061027542 (-3.885e-08); habit, intent, hint, tme unchanged.
limn has no usable pair: the previous run scored 5884 queries and this one 6001, different cohorts.

**Raw evidence.** Every count in the table above comes from one file under `rank_diffs/`,
produced by `rank_diff.py <model> <old_run> <new_run> <checkpoint>` reading
`workbench/artifacts/composition/runs/<run_id>/ranks/<model>_<checkpoint>.json` for both runs.
No number here is hand-entered.

| model | raw evidence file |
|---|---|
| encoder | `rank_diffs/encoder_6a528e87_vs_5dee7425.json` |
| conesep | `rank_diffs/conesep_30f6de42_vs_a4f786d8.json` |
| habit | `rank_diffs/habit_497c25c5_vs_c6eb9d57.json` |
| intent | `rank_diffs/intent_37ee72aa_vs_4901da13.json` |
| hint | `rank_diffs/hint_4bc9893a_vs_f461f391.json` |
| ptha_mtst | `rank_diffs/ptha_mtst_315dd590_vs_f54a7745.json` |
| tme | `rank_diffs/tme_ad477312_vs_cd1a696d.json` |
| limn | `rank_diffs/limn_cohort_note.json` (not a diff: cohorts 5884 vs 6001 differ) |

Per-file totals, summed over the three conditions: `queries_compared` is 18009 for every model
(6003 x 3), and `changed_rank_queries` sums to encoder 12727, conesep 17, ptha_mtst 46,
habit/intent/hint/tme 0. The per-condition breakdowns quoted above are the `conditions` blocks of
those same files.

## limn coverage: 6001/6003, never 100%

`category_resolution` reports 5884 resolved / 0 unresolved, which is NOT the scored count. Two
queries hit the adapter's embedding-scope ceiling and are unscored with explicit reasons in
`ranks/limn_base_iter0_all_categories.json:query_failures`:

```
7:3038: category unresolved has no adapter embedding scope ('dress','shirt','toptee')
7:3790: same
```

Scored 6001 per condition, coverage 0.9996668332500417, `gap_cohort_count` 6001, candidate
manifests `partial: true` with `record_count` 6001 against `expected_record_count` 6003. The
previous run (`10173dc4`) scored 5884, so the new run is strictly better coverage but its metrics
are not comparable to the older one.

## ptha_mtst: scoring path pinned, encode path unpinnable

`score()` for ptha_mtst is `np.einsum(..., optimize=True)` and runs in the CONTROLLER process
(`ctl312h`, real OpenBLAS, pin verifiable). The model env
(`artifacts/environments/cir-005ef4e1b02999a5`) has **no OpenBLAS**: `libblas.so.3` resolves to
`libmkl_rt.so.2` (MKL 2022.1), `numpy.libs/` is empty. Consequences measured:

* `_openblas_symbols()` finds no setter -> `setter_found: false` if the pin were attempted there.
* `mkl_set_num_threads(1)` **segfaults** that interpreter (exit 139, `Fatal Python error:
  Segmentation fault`) via `ctypes.CDLL("libmkl_rt.so.2")`, via `ctypes.CDLL(None)`, via
  `mkl_domain_set_num_threads`, with `LD_PRELOAD=libiomp5.so`, with `LD_PRELOAD=libomp.so`, before
  and after NumPy import. `mkl_get_max_threads()` works (returns 20).
* `MKL_NUM_THREADS=1` DOES pin it (max reads 1, gemm succeeds, and the gemm result differs from the
  20-thread result: 15302833 vs 15377137 on 600x256 @ 400x256) — but the env var only takes effect
  before load and cannot be applied mid-process.

So: ptha_mtst's ENCODE forward passes ran under MKL with no runtime pin and no way to verify one.
Its scoring ran pinned. This run must not be presented as equivalent to an OpenBLAS-pinned run.

Gap cohort and `reference_eligible` (true, measured, reference_rank 1) are identical on both sides
of every pair, so none of the deltas above come from a denominator change.

## Correction to an earlier claim in RUN_LOG.csv

The conesep entry originally said all three conditions were bit-identical while quoting a gap that
differed. That was wrong. The per-query diff shows 17 of 18009 conesep ranks moved by exactly 1.
The claim is corrected in RUN_LOG.csv; the numbers above are the record.

## Mechanism: conesep is weakly BLAS-sensitive, not score-neutral

Directly observed in the export for multimodal `7:3744` (target rank 30 -> 31):

```
[29] old 27939:0.6219767332077026   new 67612:0.6219767928123474
[30] old 67612:0.6219766139984131   new 27939:0.6219766736030579
```

Two gallery items 1.2e-7 apart swap order. The other 16 conesep moves have target ranks
1007..65087, i.e. outside the stored top-100 window, so their window is identical and only the
count of higher-scoring items shifts by one. Same 1e-7 reordering; not directly visible.

Consequence for the earlier 0/400 measurement: sampling 400 queries per model did not hit a query
holding such a near-tie in its top-50, so "0 ranks changed" was true of that sample and not of the
6003-query population. R@10/R@50 are unchanged for conesep because no rank crossed a k boundary —
the exact case where equal aggregates hide reordered ranks.

## Encoder

CORRECTION 2026-11-09: the 12727 of 18009 rank movement is NOT BLAS pinning. A controlled A/B
(`d3b_encoder_aggregate_scan.txt`, same vectors, only threads 1 vs 10) moves 1 rank/condition with
r10/r50 delta 0.0. The 12727 delta is caused by a changed adapter RANK method (adapter_module_source_hash 1a8b0dd2->bf60d855, adapter_source_hashes.rank 5f828b13->f0914b12, score hash identical) plus an interpreter change (pair-adapted py3.11.15 unverified -> encoder310 py3.10.20 verified).
R@10 rose (31.0178->31.0345) while R@50 fell (49.5752->49.4753), impossible for a thread-count change. Movement roughly symmetric
(multimodal 1414 improved / 1405 worsened, image_only 2443 / 2771, text_only 2173 / 2521), so it is
reordering rather than a shift in one direction. The pre-pin and pinned encoder runs are different
measurements and must not be mixed.

## Open defect (not fixed): the artifact cannot prove its own BLAS policy

**Defect.** `runner._record_effective_blas_pin()` (`workbench/composition/runner.py:1020`, dict at
`:1049`) is called at `:218`, `:992` and `:1782`, and it faithfully stores the runtime readback in the
module-level `_effective_blas_pin`. **No code path ever writes that dict into the run identity.**
Grepping for the name outside those lines returns nothing but the calls and the definition.

**Why it matters.** Every artifact therefore records two things and neither is the observed value:
`blas_threads_pinned: 1` (what we ASKED for) and `blas_pin_capability` (a probe that a setter and a
getter exist). An artifact cannot show that *this* run's arithmetic was pinned — only that this
*interpreter* is capable of being pinned. The evidence that the pin applied to the eight runs in this
report is external: the gate probe in `blas_gate.py` (20 → 1 → readback 1 → restored 20) plus the
observed metric movement on encoder. Both are neighbouring evidence, not the artifact's own record.
This is the same defect class the lane-A fix removed for the *fallback* path: recording a requested
number while the runtime might be doing something else.

**Minimal fix** (deliberately not applied here):
1. add `"blas_pin_effective": dict(_effective_blas_pin)` to the identity payload in `runner.py`,
   next to `blas_pin_capability` (around `:1485`);
2. it is mutable-per-process state, so it must be snapshotted at the same point the capability probe
   is, and the payload must exclude nothing — a record that omits `verified` is the old bug again;
3. **because it adds a hashed field, the group key changes: bump `identity_schema_version` to 3** in
   `workbench/backend/composition.py` (`measurement_identity`, ~`:729`), and extend `V2_MARKERS` /
   `_identity_schema` so a v2 artifact is labelled `unknown (predates identity v3)` rather than being
   regrouped under v3.

**Why it is not applied now.** It re-mints every run id and orphans the eight artifacts this report is
based on — the run directories, the candidate exports and every `rank_diffs` pairing would stop
matching. That is a deliberate re-measurement decision, not a patch. Recorded as the next task.
