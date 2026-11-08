# DIFFERENTIAL_TESTS — axes, tolerances declared before the run, observed errors

Tolerances were written into the script **before** any measurement and were not relaxed after seeing
results. Scripts are in `scripts/`; raw JSON in `logs/`.

## Declared tolerances

| Output class | Tolerance | Why this number |
|---|---|---|
| Discrete (ranks, gallery positions, candidate order, token ids, query ids) | **exact equality** | no floating freedom exists in these values |
| Repeat of the same call, same process, same input | `atol = 0`, `rtol = 0` | a deterministic expression must be bit-repeatable; anything else is a defect |
| Score matrix, batch 1 vs batch 32 vs all-at-once | `atol = 1e-6`, `rtol = 0` | float32 BLAS may reorder additions across a different call shape; 1e-6 is ~2 ulp at unit norm and was fixed before measuring. Observed error was 6.6e-7, i.e. inside it by luck of magnitude, not by adjustment |

## Environment

CPU only, `OMP_NUM_THREADS=4` / `OPENBLAS_NUM_THREADS=4` to stay well under the shared-host budget
(the live `intent` cohort holds the GPU and some cores). Node under test:
`workbench/composition/adapters.py::CompositionAdapter.score` and `_ranks_from_scores` — these are the
scoring and rank stages **every** adapter funnels through (`model_habit.py`, `model_intent.py`,
`model_hint.py`, `model_tme.py`, `model_encoder.py` all delegate to them; `limn` re-implements the same
tie rule in `model_limn.py:400-416`).

Data: the real LIMN dress gallery matrix, 74,381 × 768 float32, unit-norm rows
(`artifacts/composition/preserved_features/limn/gallery_dress.npy`), plus the real 6,003-query HABIT
rank artifact for the metric axis.

## Axes and results

### A1 — batch size 1 vs 32 vs all-at-once (score stage)

`logs/differential_stages.json`

| comparison | max abs | exact equal | verdict |
|---|---:|---|---|
| batch 32 vs all 8 rows at once | 0.0 | yes | identical |
| batch 1 vs all 8 rows at once | 6.56e-7 | no | divergent, **within** declared 1e-6 |

**First divergent stage: the score stage, at the BLAS reduction — nothing downstream.** Confirmed
non-propagating by A1b.

### A1b — does that float noise reach the rank stage?

`logs/rank_batch200.txt`, 200 real gallery rows used as queries, each its own positive:

```
queries 200   max_abs_score_delta 2.38e-7
ranks differing: 0        max rank delta 0
```

**First divergent stage: none.** The score noise is absorbed; ranks are unchanged.

### A2 — `_ranks_from_scores` vs brute-force stable argsort

8/8 identical (`logs/differential_stages.json`). Brute force = order by `(-score, position)` then take
the best positive's 1-based index, computed independently of the implementation under test.

### A3 — exact-duplicate tie behaviour (the systematic-tie axis)

`logs/tie_rank_stability.json`, `logs/duplicate_embeddings.json`

The gallery genuinely contains byte-identical duplicate rows: **851 groups, 2,386 rows, 1,535 excess**.
They are **byte-identical source files under different product IDs** — verified by hashing the `.png`s
(`logs/dup_pixel_check.txt`: sha256 equal for 8/8 groups, pixel correlation 1.0). Not a resolver bug: an
independent resolution pass found 74,381 identities → 74,381 distinct files, 0 collisions.

Test: 25 largest duplicate groups used as queries, positive placed at the **last** duplicate so an
exact score tie must be broken by gallery position.

```
score max abs delta between batch sizes: 5.36e-7
ranks that flipped: 0
```

Largest group was 54 members, all 54 with identical scores; rank still identical under batch 1 and 32.

**First divergent stage: none.** The tie rule is correct *and* stable under batching.

### A4 — repeat determinism

Same call twice, same process: `max_abs 0.0`, `exact_equal true`. Meets the declared `atol = 0`.

### A5 — metric aggregation vs dict ordering (real 6,003-query ranks)

`compose()` on the real HABIT rank table, forward vs reversed condition and qid ordering:

```
gap_forward = gap_reversed = 0.388704727214254   (bit-identical)
cohort      = 6003 both ways
```

**First divergent stage: none on this data.** Note the theoretical exposure: `metrics.py:104` builds
`qids` as a `set` and iterates it to form the common cohort, so summation order is hash-seed
dependent in principle. Not observed; carried as `SUSPECTED` (FINDINGS S1), not claimed as a defect.

## Axes not run

| Axis | Status | Reason |
|---|---|---|
| batch 1 vs 32 through a **real model forward** | `NOT_RUN` | needs GPU; busy with the live `intent` cohort for the whole audit, and a second evaluator is forbidden |
| a query count **not divisible by 32** through `_encode_queries` on real data | `NOT_RUN` | needs a model worker; the CPU `CompositionAdapter` path has no query-encode loop. Code-level review instead: NaN prefill + per-index writes make row loss/misalignment structurally impossible (`FINDINGS.md` D7) |
| cold vs warm cache, cache enabled vs disabled, end to end | `NOT_RUN` | same reason. Code-level review found the gallery cache is shape-blind (FINDINGS F3, demonstrated on a synthetic memmap) |
| token/mask stage differential across conditions on a real model | `NOT_RUN` | needs GPU. Substituted with executed tokenizer-level checks (FINDINGS D3) |
| determinism across **two separate processes** | `NOT_RUN` | needs a GPU encode; the repo's own replay notes two processes can differ in feature hashes even with a fixed seed (`replay/encoder.py:297-302`), which is why that path pins deterministic kernels |

## Interpretation

The scoring and ranking stages are the most trustworthy part of this pipeline: exact-duplicate ties
are handled with a stable, documented rule; chunking does not change ranks; repeated calls are
bit-identical; and the metric aggregation is order-insensitive on real data. **No divergence was found
at or after the rank stage on any axis that could be run.**

One cross-check arrived from a parallel trace and is worth recording here because it is an
*equivalence* result I did not run myself: `docs/reviews/20261108T153045-audit-equivalence/decode_encoder.json`
records `bit_identical: true, bytes_differing: 0` for the ENCODER **decode/transform** stage across
batches {1, 32, 64, 256} on 512 real released images, and the throughput review records a 0.0 delta for
LIMN's transform across 2/4/8 decode workers. That corroborates A1 at the stage where A1 was measured
(scores, not pixels) and extends it upstream to the transform. It does **not** cover the model forward,
which remains the unmeasured gate (G4 in `docs/reviews/20261108T153045-audit-equivalence/TOLERANCES.md:34-36`,
no result file present).

Also unexecuted for the same reason: three separate trace agents independently reported that no
forward-pass batch-invariance result exists in the tree, so my A1/A1b results (controller-side score
stage) are the only batch-invariance evidence that has actually been produced.

The unexecuted stages are exactly the ones where a preprocessing or tokenization defect would live
(transform, tokens, masks, model input). Those are covered statically in `PIPELINE_TRACE.md` /
`UPSTREAM_PARITY.md`, and the two spots where a real behavioural question exists are answered by
execution at the tokenizer level (empty text) and the transform level (black image, ENCODER
train-vs-val) rather than left as source-reading claims.
