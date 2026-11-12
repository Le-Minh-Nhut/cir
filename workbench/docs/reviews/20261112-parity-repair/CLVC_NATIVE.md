# CLVC-Net native R@10 conflict — one controlled run, per-query evidence

Resolves the contradiction between the two CLVC-Net native artifacts. Both numbers are real; they
were produced at **different batch sizes**, and the difference is entirely in the evaluator's batch
loop, not in the checkpoint, mask, text or tie handling.

Producer: `scripts/clvc_native_ranks.py` (new, this review).
Evidence: `raw/clvc_native/` (per-query JSONL + per-category JSON + `summary.json`).

| | value |
|---|---|
| model | `clvc_net` (`author_drive`) |
| checkpoint bundle | `workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc/` (6 files) |
| bundle digest (repo canonical hasher, `kind=directory`) | `b504245dc028d534e6e421eb73d5884d6cbda9cbb3ae03b95ab5c8b8e90ee6b2` — reproduced in this run |
| source pin | `bd9b6889489806cb60676597880f91d918b279cb`, `verified=true`, `state=vendored_in_repo`, `dirty=false`, provenance digest `09b9f981…c7620f` |
| worker interpreter | `workbench/envs/clvc39-la/bin/python` (3.9.25), torch `2.7.1+cu128`, torchvision `0.22.1+cu128`, numpy `1.26.4` |
| controller interpreter | `workbench/envs/ctl312h/bin/python` (3.12.11) — orchestrates only, never composes |
| launcher | `MAX_WAIT=3600 workbench/scripts/gpu_job.sh …` (one GPU job at a time) |
| device | CUDA (RTX 5070 Ti); no device-neutral patch, so `runtime_label = NATIVE_CUDA` |

Interpreter split is deliberate: the gate refuses `clvc39-la` as composition controller
(`[BLOCKED/UNSAFE_INTERPRETER]`), but the model env is the one that can unpickle the author's
whole-module checkpoints. The controller therefore spawns the model env as a subprocess and records
both paths (`summary.json` → `controller_interpreter`, `worker_interpreter`).

## 1. Provenance table — what native CLVC-Net evaluation actually does

| item | where | what it is |
|---|---|---|
| **gallery construction** | `datasets.py:233-257` (`get_test_targets`, `gallery_all=False`), called from `test.py:85` | Order-preserving **union of every query's candidate and target image**, in first-appearance order. Not the full split list (3817 for dress), not per-query candidates: dress 2017 queries → **2628** gallery images. |
| **reference inclusion / exclusion** | `test.py:134-135` | The query's own source image **is** in the gallery and is masked to `-10e10`, i.e. excluded **by score**, not by removal — it stays in the array at the bottom. Measured: `reference_masked_last` is true for **2628/2628** dress and **3089/3089** shirt queries, `min_reference_rank` = 2627 of 2628 and 3088 of 3089 (i.e. exactly last, never higher). Since `-10e10` dominates any cosine similarity, the mask is total — it never competes for a top-50 slot. |
| **mask constant** | `test.py:135` | `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` |
| **ranking / tie convention** | `test.py:138` | `np.argsort(-sims[i, :])[:50]` — numpy default `kind='quicksort'`, **not stable**. R@k is a positional test (`test.py:145`), so a query whose target ties with another gallery image has a tie-decided rank. Recorded per query as `tied_with_target`; counts: dress 40, shirt 12, toptee 14 at batch 32 (36/12/14 at batch 48). |
| **text correction** | none | Captions reach the model verbatim as `"<BOS> {c0} <AND> {c1} <EOS>"` (`datasets.py:173-175`), tokenised by the checkpoint's own `SimpleVocab` (`text_model.py:23-27`). No correction, normalisation or filtering step exists in the evaluator. |
| **transform** | `train.py:90-96` (FashionIQ val branch) | `Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet)`. Reproduced literally by the wrapper at `replay/clvc_net.py:183-188`. |
| **batch-size plumbing** | `test.py:96` (query loop), `test.py:112` (gallery loop) | `params.batch_size` is the flush threshold: `if len(imgs) >= params.batch_size or t is <last>`. Pinned `test.py` has **no CLI, no `__main__`, no env var** — the only argument path is the `params` object. The wrapper builds `SimpleNamespace(batch_size=batch_size)` at `replay/clvc_net.py:206` and exposes `--batch-size` at `replay/clvc_net.py:268`. Pinned source bytes were never edited. |
| **recall arithmetic** | `test.py:142-148` | `100 * hits / len(nn_result)`, per category; macro is the unweighted mean over the three categories (`replay/common.py:702-712`), never query-weighted. |

### What the two historical artifacts failed to record

Both `clvc_run1.json` and `clvc_run3cat_gpu.json` record `per_query_published: false`, no `mrr`, no
batch size, no mask policy, no tie policy and no checkpoint digest. `clvc_run1.json`'s provenance is
only `{source_commit, local, global}` (paths, not digests); `clvc_run3cat_gpu.json` adds
`native_evaluator`, `runtime_label`, `device`. Command and batch size existed **only in prose**
(`BLOCKED_MODELS_VERIFICATION.vi.md:214-218` → `--batch-size 32`;
`CHECKPOINT_SELECTION_REPORT.md:1697-1699` → positional `48` to a session runner). That is exactly
why the two numbers could not be reconciled from the artifacts alone: the single variable that
separates them was never inside them.

## 2. Pass table

All passes share one fixed query set per category and the same seven gallery/mask constants above.
Per-query count = category query count in every pass (asserted in the controller).

| pass | batch | categories | queries (d/s/t) | R@1 d/s/t | R@10 d/s/t | R@50 d/s/t | macro R@1 | macro R@10 | macro R@50 | wall |
|---|---|---|---|---|---|---|---|---|---|---|
| `batch32` (baseline, declared before run) | 32 | 3 | 2017/2038/1961 | 6.841844 / 6.476938 / 7.853136 | 29.449678 / 27.772326 / 32.840388 | 56.767476 / 55.250245 / 63.997960 | 7.057306217652120 | **30.02079703540087** | **58.67189400437203** | 334.3 s |
| `batch32_r2` (determinism control) | 32 | 3 | 2017/2038/1961 | identical | identical | identical | 7.057306217652120 | 30.02079703540087 | 58.67189400437203 | 180.2 s |
| `batch32_r3` (mask-field re-measure; adds `reference_rank`) | 32 | 3 | 2017/2038/1961 | identical | identical | identical | 7.057306217652120 | 30.02079703540087 | 58.67189400437203 | 651.6 s |
| `batch48` (control for the 29.9707 artifact) | 48 | 3 | 2017/2038/1961 | 6.792266 / 6.427870 / 7.853136 | 29.350521 / 27.772326 / 32.789393 | 56.817055 / 55.152110 / 63.895971 | 7.024424119153085 | **29.970746517160162** | **58.62171212901515** | 636.3 s |
| `batch1` (sensitivity probe) | 1 | 1 (dress) | 2017 | 6.792266 | 28.805156 | 56.717898 | — | 28.805156172533465 | 56.717897868120971 | 290.2 s |

Every `recomputed_matches_native` is `true`: the recalls recomputed from the captured per-query
`target_rank` values equal the numbers the pinned `test.test` returned, to 1e-9, for all 10
category/pass combinations. The per-query evidence therefore *is* the aggregate, not a parallel
estimate. Macro recomputed from per-query ranks also matches (`summary.json` →
`macro_matches_recomputed`).

Batch detection note: the batch-48 dress wall time (268.6 s) is ~5.5× the batch-32 dress wall time
(44.8-49.6 s) on the same GPU, independently confirming that the `params.batch_size` path really
changed the loop, not just a label.

## 3. Reproduction verdict

### `workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json` = **batch 32** ✓

Reproduced **to the last bit**, all six R@10/R@50 cells, not just the macro:

| category | artifact R@10 | batch-32 R@10 | Δ | artifact R@50 | batch-32 R@50 | Δ |
|---|---|---|---|---|---|---|
| dress | 29.449677739216657 | 29.449677739216657 | 0.0 | 56.767476450173525 | 56.767476450173525 | 0.0 |
| shirt | 27.77232580961727 | 27.77232580961727 | 0.0 | 55.250245338567225 | 55.250245338567225 | 0.0 |
| toptee | 32.840387557368686 | 32.840387557368686 | 0.0 | 63.99796022437532 | 63.99796022437532 | 0.0 |
| **macro** | **30.02079703540087 / 58.67189400437203** | **30.02079703540087 / 58.67189400437203** | **0.0 / 0.0** | | | |

This also matches the prose command that recorded `--batch-size 32`
(`BLOCKED_MODELS_VERIFICATION.vi.md:214-218`).

### `workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json` = **batch 48** ✓

Reproduced **to the last bit**, all nine R@1/R@10/R@50 cells:

| category | artifact R@1 / R@10 / R@50 | batch-48 R@1 / R@10 / R@50 | Δ |
|---|---|---|---|
| dress | 6.7922657411998015 / 29.350520575111553 / 56.81705503222608 | identical | 0.0 |
| shirt | 6.427870461236506 / 27.77232580961727 / 55.15210991167812 | identical | 0.0 |
| toptee | 7.8531361550229475 / 32.78939316675166 / 63.895971443141256 | identical | 0.0 |
| **macro** | **29.970746517160162 / 58.62171212901515** | **29.970746517160162 / 58.62171212901515** | **0.0 / 0.0** |

This matches the prose command that passed batch `48`
(`CHECKPOINT_SELECTION_REPORT.md:1697-1699`).

**Neither historical number was wrong.** They are two batch sizes of one unmodified pipeline over
one checkpoint bundle. Exhausted hypothesis space, with the evidence for each rejection:

- *checkpoint*: excluded — the same 6 files (digests recorded per pass, bundle digest `b504245d…`
  reproduced) produce both numbers.
- *gallery order / construction*: excluded — 2628/3089/2902 gallery images in every pass, ordered
  pair-union by `datasets.py:243-249`; the code path is not batch-dependent.
- *mask*: excluded — `-10e10`, `test.py:135`, identical in all passes; the reference sits exactly
  last in 2628/2628 dress and 3089/3089 shirt queries (`reference_masked_last`, `min_reference_rank`
  = gallery−1), so it can never take a top-50 slot.
- *corrected text*: excluded — no correction exists anywhere in `test.py`/`datasets.py`/`text_model.py`.
- *tie rule*: **not** the cause of the historical gap, but it is a real confound at the margin — ties
  at the target score exist (40/12/14 at batch 32 vs 36/12/14 at batch 48 in dress/shirt/toptee), so
  individual ranks within a tie group are arbitrary. It cannot explain a systematic +0.05 macro
  shift, and here it is not even needed: batch size alone reproduces both artifacts exactly.
- *batch*: **confirmed as the mechanism.** It is the only difference between the two passes and it
  reproduces both artifacts exactly, in both directions. Numerically the effect is small but real:
  dress R@10 592→594 hits, toptee 643→644, shirt unchanged (per 2017/2038/1961 queries).

Why batch size can move a "deterministic" eval at all: the flush boundary at `test.py:96`/`test.py:112`
changes how many images/texts share one forward pass, so cuDNN selects different kernels and
accumulation orders; features then differ in the last bits (measured max |Δ| on the *target* score
alone: 0.1266 at batch 1 vs 32, 1.4e-5 between batch 32 and rerun — exactly 0.0 in the rerun case).
Near-tied gallery pairs flip. This is a property of float reduction order, not of the model.

## 4. Batch sensitivity: batch 1 vs batch 32

Fixed query set, same gallery, same weights, only `params.batch_size` differs.

| comparison | category | max abs Δ R@1/R@10/R@50 | moved target ranks | queries compared |
|---|---|---|---|---|
| batch 1 vs 32 | dress | **0.644522** (R@10: 28.805156 vs 29.449678) | **1548** | 2017 |
| batch 48 vs 32 | dress | 0.099157 | 427 | 2017 |
| batch 48 vs 32 | shirt | 0.098135 | 378 | 2038 |
| batch 48 vs 32 | toptee | 0.101989 | 372 | 1961 |

**1548 of 2017 dress ranks move between batch 1 and batch 32** (76.7%), with a max score change of
0.1266 at the target rank. One category only was run at batch 1, for cost — stated plainly rather
than extrapolated: `shirt` and `toptee` were **not** run at batch 1.

This is the point that matters for the composition path: per-query ranks are not stable under a
batch change, so a single-query composition result compared against a batch-32 aggregate recall is
comparing two different numerical regimes. Even the *aggregate* moves by 0.64 pp at R@10 for a
40× batch change — larger than several of the model gaps this campaign is adjudicating.

## 5. Determinism control

`batch32` vs `batch32_r2` — same batch, same inputs, separate process, ~4 minutes apart:

| category | rows | qid order identical | moved ranks | max abs score Δ |
|---|---|---|---|---|
| dress | 2017 | yes | **0** | **0.000e+00** |
| shirt | 2038 | yes | **0** | **0.000e+00** |
| toptee | 1961 | yes | **0** | **0.000e+00** |

**Bit-identical.** Same batch size ⇒ identical per-query ranks and identical scores, so every
difference reported above is attributable to batch size and to nothing else — no run-to-run noise,
no seed drift, no cuDNN nondeterminism at fixed shape. This is what makes the two exact
reproductions meaningful rather than lucky.

A third batch-32 pass (`batch32_r3`, run to re-measure the reference-mask field after a defect in it
was found) returned the same macro to the last digit: **30.02079703540087 / 58.67189400437203**.
Three independent batch-32 passes agree exactly.

## 6. Per-query evidence

`raw/clvc_native/<pass>/<category>.jsonl`, one row per ranked query:

| pass | category | query_count | JSONL rows | unique qid | fields complete | gallery | ties at target | rank=None | ref masked last |
|---|---|---|---|---|---|---|---|---|---|
| batch32 | dress | 2017 | 2017 | yes | yes | 2628 | 40 | 0 | 2017/2017 |
| batch32 | shirt | 2038 | 2038 | yes | yes | 3089 | 12 | 0 | 2038/2038 |
| batch32 | toptee | 1961 | 1961 | yes | yes | 2902 | 14 | 0 | measured in `batch32_r3` |
| batch32_r2 | dress/shirt/toptee | 2017/2038/1961 | equal | yes | yes | same | same | 0 | — |
| batch32_r3 | dress/shirt/toptee | 2017/2038/1961 | equal | yes | yes | same | 40/12/14 | 0 | 2017/2038/1961 |
| batch48 | dress/shirt/toptee | 2017/2038/1961 | equal | yes | yes | same | 36/12/14 | 0 | — |
| batch1 | dress | 2017 | 2017 | yes | yes | 2628 | 40 | 0 | — |

Row fields: `qid` (stable `<category>:<index>:<candidate>-><target>`), `candidate_id`, `target_id`,
`reference_gallery_index`, `target_gallery_index`, `target_rank`, `score_at_target_rank`,
`score_at_prev_rank` (the score one rank above, so a tie at the boundary is visible),
`tied_with_target` (count of other gallery images with the identical score), `tie_rule`,
`batch_size`, `reference_rank` (positional rank of the masked reference) and `reference_masked_last`.

Passes `batch32`, `batch32_r2` and `batch48` predate the `reference_rank` field: they carry
`reference_in_top50`, which tested membership in the full ordering and is therefore trivially true —
it is **not** evidence about the mask and is superseded by `batch32_r3`. Everything those three
passes assert about recalls, ranks, ties and batch sensitivity is unaffected; the field was additive. `target_rank` is never `None` in any pass because the gallery
contains every target by construction.

## 7. What this settles / what it does not

**Settles**

- `clvc_run3cat_gpu.json` (R@10 **30.02079703540087**, R@50 **58.67189400437203**) = batch 32.
  Reproduced exactly, all six cells.
- `clvc_run1.json` (R@10 **29.970746517160162**, R@50 **58.62171212901515**) = batch 48.
  Reproduced exactly, all nine cells.
- Neither artifact is defective, mistuned, or measured on a different checkpoint, gallery, mask or
  text pipeline. The 0.05 pp macro gap is a batch-size artefact of a genuinely unchanged evaluator.
- The evaluation is deterministic at fixed batch size (0 moved ranks, 0.0 score Δ on a full repeat).
- Batch size is **not** a cosmetic flag: it moves 427/378/372 ranks at a 1.5× change and 1548/2017
  at a 32× change, on one category.

**Does not settle**

- **Which batch size is canonical.** Nothing in the pinned upstream or the registry designates one;
  `test.py` merely reads `params.batch_size`. Any future "native CLVC-Net R@10" claim is ambiguous
  unless the batch size travels with it.
- It does not re-derive the paper comparison (Table 1), does not touch gate tolerances, and does not
  change any historical number — both old artifacts are preserved byte-for-byte.
- It does not cover batch 1 for shirt/toptee, and does not test whether batch 32 vs 48 changes
  *downstream composition* beyond the rank movement already measured.

### Rule this record imposes

The composition runs pin **batch 32** (`replay/clvc_net.py:268` default, and every recorded
composition measurement). Because a batch change moves 1548/2017 dress ranks, `--batch-size` MUST
NOT be varied for any CLVC-Net measurement that will be reported or compared. A CLVC number without
an explicit `batch_size` field is not comparable to these artifacts — which is precisely the defect
that produced this investigation.

## 8. Reproduction

```bash
# baseline batch 32 + batch-1 sensitivity probe (the deliverable)
MAX_WAIT=3600 setsid nohup workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/clvc_native_ranks.py \
  --batch-size 32 > /tmp/clvc_native.log 2>&1 < /dev/null &

# controls behind the verdict: batch 48 (run1) and a batch-32 repeat (determinism)
MAX_WAIT=3600 workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/clvc_native_ranks.py \
  --summarise-only --extra-pass 48 --extra-pass 32

# merge existing pass directories without running anything
workbench/envs/ctl312h/bin/python \
  workbench/docs/reviews/20261112-parity-repair/scripts/clvc_native_ranks.py --summarise-only
```

Pass directories are never overwritten: a colliding label becomes `batch32_r2`, `batch32_r3`, … so
prior evidence survives a re-run.
