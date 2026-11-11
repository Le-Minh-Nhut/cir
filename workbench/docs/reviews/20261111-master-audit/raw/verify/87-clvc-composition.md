# 87 — CLVC-Net composition adapter: worker, parity, and what CIRCUS changes

Branch `research/cir-failure-workbench`, 2026-11-11/12. Measurement interpreter
`workbench/envs/clvc39-la/bin/python` (CPython 3.9.25, torch 2.7.1+cu128, torchvision
0.22.1+cu128). Repo-side test interpreter `workbench/envs/ctl312h/bin/python`.
Every GPU command ran under `workbench/scripts/gpu_job.sh` (one holder campaign-wide).
Live service on 8000/5173 was never contacted; ports 8000/5173 never bound.

Checkpoint root (via `workbench.backend.registry.checkpoint_path()`, no path guessed):
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc/`.

---

## 0. Locked sample — written BEFORE any forward pass

Frozen in `/tmp/c87/sample.json` by `/tmp/c87/sample.py`, whose only inputs are
`data/FashionIQ/captions/cap.<cat>.val.json` and `image_splits/split.<cat>.val.json`
(no model, no GPU, no checkpoint). Selection rule, so it is reproducible rather than hand-picked:
the **first 5 val captions in file order**, gallery = the 10 distinct candidate/target identities of
those 5 queries in query order, then the first 10 unused `split.<cat>.val.json` entries in file
order (20 total). Native edit string is the pinned template from `datasets.py:174`:
`"<BOS> {cap0} <AND> {cap1} <EOS>"`.

### dress (query_index 0,1,2,3,4 of `cap.dress.val.json`)
| # | reference (candidate) | positive (target) | native edit |
|---|---|---|---|
| 0 | `B005X4PL1G` | `B0084Y8XIU` | `<BOS> is shiny and silver with shorter sleeves <AND> fit and flare <EOS>` |
| 1 | `B008XODTD0` | `B00AKLK08G` | `<BOS> is grey with black design <AND> is a light printed short dress <EOS>` |
| 2 | `B00BPYP69K` | `B00CMPE0C0` | `<BOS> is a solid red color <AND> shorter and tighter with more blue and white <EOS>` |
| 3 | `B000QSGNOI` | `B004UO3XYC` | `<BOS> is a plain white feminine t shirt <AND> is a tan shirt. <EOS>` |
| 4 | `B00FQANLX2` | `B007U6KROG` | `<BOS> has thin straps and different pattern <AND> more autumn colored and longer <EOS>` |

gallery (20, order is the score-matrix column order and the tie-break order):
`B005X4PL1G, B0084Y8XIU, B008XODTD0, B00AKLK08G, B00BPYP69K, B00CMPE0C0, B000QSGNOI, B004UO3XYC,
B00FQANLX2, B007U6KROG, B009PMCJLW, B007HNF4QI, B004KHOPDW, B0083E6W7U, B007L3WP5W, B0091PLEKA,
B004P7TNIY, B0077SM1Z0, B007UYY43I, B00762GDB0`

### shirt (query_index 0..4 of `cap.shirt.val.json`)
| # | reference | positive | native edit |
|---|---|---|---|
| 0 | `B00CZ7QJUG` | `B005AD7WZI` | `<BOS> is solid white <AND> is a lighter color <EOS>` |
| 1 | `B0083I6W08` | `B00BPD4N5E` | `<BOS> is green with a four leaf clover <AND> is green and has no text <EOS>` |
| 2 | `B0083WOL16` | `B008VNAKSU` | `<BOS> Is a brown tee shirt with diseal logo <AND> is darker and has short sleeves <EOS>` |
| 3 | `B005Y4KFPM` | `B005Y4JJ0Y` | `<BOS> is dark blue <AND> is blue with a different character. <EOS>` |
| 4 | `B006QHPYIY` | `B005NYBUF2` | `<BOS> is red and a tshirt <AND> more warmer colors <EOS>` |

gallery: `B00CZ7QJUG, B005AD7WZI, B0083I6W08, B00BPD4N5E, B0083WOL16, B008VNAKSU, B005Y4KFPM,
B005Y4JJ0Y, B006QHPYIY, B005NYBUF2, B000KENMD8, B0013EO152, B00DI9YY8O, B00C40R3FY, B008LIH5VU,
B0081SF8U0, B007KFQVGK, B0051UW60Q, B004DJ0A0Y, B000GYAKW8`

### toptee (query_index 0..4 of `cap.toptee.val.json`)
| # | reference | positive | native edit |
|---|---|---|---|
| 0 | `B008CFZW76` | `B008CG1JJ0` | `<BOS> is the same <AND> appears to be exactly the same <EOS>` |
| 1 | `B0088WRQVS` | `B001AS562I` | `<BOS> i taank top <AND> has spaghetti straps <EOS>` |
| 2 | `B00BM27KWQ` | `B007TGJOMS` | `<BOS> is yellow with fringe <AND> is yellow with shorter sleeves <EOS>` |
| 3 | `B00H03TS7Q` | `B0085YC1G4` | `<BOS> The shirt is black with the cat in hat. <AND> has the Cat in the Hat on it <EOS>` |
| 4 | `B00B02GQ1E` | `B00BB1UAMU` | `<BOS> has longer lace sleeves in white <AND> Is longer sleeved <EOS>` |

gallery: `B008CFZW76, B008CG1JJ0, B0088WRQVS, B001AS562I, B00BM27KWQ, B007TGJOMS, B00H03TS7Q,
B0085YC1G4, B00B02GQ1E, B00BB1UAMU, B00D2EVETW, B009P4JF1W, B004C1XF96, B003R7235S, B000F7SSDO,
B0015O8JLC, B005CTBNN2, B00528E466, B009CC9S5A, B00BSWGT18`

Every string above is copied from `sample.json`, which is generated from the annotation files
alone; the model-derived numbers in §2-§5 are keyed on it, not hand-typed.

---

## 1. Declared tolerances — written BEFORE the final measurement run

The native path and the composition path must perform the SAME operations on the SAME float32
tensors: same transform, same `F.normalize` per half, same `torch.cat`, same device. So:

| comparison | target | pass criterion |
|---|---|---|
| local 1024-d half | bit-identical | `atol=0, rtol=0`, accept `Δ ≤ 1e-6` |
| global 1024-d half | bit-identical | `atol=0, rtol=0`, accept `Δ ≤ 1e-6` |
| concatenated 2048-d row | bit-identical | `atol=0, rtol=0`, accept `Δ ≤ 1e-6` |
| target row 2048-d | bit-identical | `atol=0, rtol=0`, accept `Δ ≤ 1e-6` |
| masked score matrix | bit-identical | `atol=0, rtol=0`, accept `Δ ≤ 1e-6` |
| rank vector | identical | exact list equality |

Score equivalence identity (`test.py:100-129` vs `score()`): the claim is **exact** float32
arithmetic up to one rounding, so the same `Δ ≤ 1e-6` criterion applies, and a value above it means
the docstring is wrong and `score()` must be replaced by what upstream actually computes.

Human-visible disclosure: these thresholds were written into this file after a first exploratory
run (whose numbers matched them) and before the final run reported in §2–§5. No threshold was moved
after seeing a number; the one place a first guess was wrong is §4, where "batch 1 == batch N" turned
out to be **false upstream**, and that is reported as a finding rather than hidden.

---

## 2. Same-input parity (native `test.py` vs the composition worker)

Method. Both paths get byte-identical inputs: a synthetic 3-file mini FashionIQ root per category
(`/tmp/c87/mini/<cat>/`) whose `resized_image/<cat>/*.jpg` are **symlinks** to the real
`data/FashionIQ` files, so no image is re-encoded. The native side is the pinned evaluator itself —
`datasets.FashionIQ(gallery_all=True)` + `test.test(params, local, global, testset, category)`
through `workbench/replay/clvc_net.py`'s `_load_replay_modules` — so its own transform, its own
batch loop and its own reference mask are exercised. The composition side is `_run(adapter, ...)`
against the resident worker plus `Adapter.score`. Both matrices then receive the SAME reference
policy (reference row masked to `-10e10`, which is what upstream does) before ranks are computed.

Native recalls on the 20-image gallery (sanity that the mask wiring is right — 5 queries only):
dress `{r1 80.0, r10 100.0, r50 100.0}`, shirt `{r1 40.0, r10 100.0, r50 100.0}`,
toptee `{r1 80.0, r10 100.0, r50 100.0}`.

### max |Δ| per comparison (float32, 5 queries × 20-gallery)

| comparison | dress | shirt | toptee |
|---|---|---|---|
| local 1024-d half | **0.0** | **0.0** | **0.0** |
| global 1024-d half | **0.0** | **0.0** | **0.0** |
| concatenated 2048-d query row | **0.0** | **0.0** | **0.0** |
| concatenated 2048-d target row | **0.0** | **0.0** | **0.0** |
| masked score matrix | **8.94e-08** | **1.19e-07** | **1.79e-07** |

All five are at or below the declared `1e-6`; four of five are exactly zero. The score row is not
bit-identical because it is the one place the two paths use different arithmetic: upstream divides
each whole row by its own L2 norm and dots (`test.py:122-129`), `score()` normalizes and dots too but
the matrix product is not required to round identically to upstream's `np.dot` on a different
memory layout. The residual is 1 ULP of a value near 1, which is what float32 gives.

### Rank vector

| category | native ranks | composition ranks | identical | top-10 list identical |
|---|---|---|---|---|
| dress | `[1, 1, 10, 1, 1]` | `[1, 1, 10, 1, 1]` | **yes** | yes |
| shirt | `[1, 2, 2, 1, 2]` | `[1, 2, 2, 1, 2]` | **yes** | yes |
| toptee | `[1, 2, 1, 1, 1]` | `[1, 2, 1, 1, 1]` | **yes** | yes |

The whole ranked list (positions 0-9 of `argsort(-row)`) is identical too, not just the metric rank,
so the 1e-07 score residual does not move any ordering on this sample.

### Score-equivalence identity, measured on real embeddings

`score(row_q, row_g)` vs `(dot_local + dot_global) / 2` on the worker's own rows:

| category | max abs difference | query row L2 norms |
|---|---|---|
| dress | **1.1920929e-07** | 1.4142135 ×5 |
| shirt | **1.1920929e-07** | 1.4142135 ×5 |
| toptee | **1.1920929e-07** | 1.4142135 ×5 (one 1.4142134) |

Row norms are `sqrt(2)` to float32 precision, i.e. each half is unit-norm as the identity requires.
The difference is one ULP and is **not** exact: the two expressions associate the sum differently
(`(a+b)/2` vs a normalized dot of the concatenation). So the docstring's "exactly" is a float32
identity to 1 ULP, not a bitwise one — reported as such rather than rounded up to "exact", and the
declared `1e-6` criterion **passes**. `score()` is kept: replacing it with the two-dot form would
change nothing a rank can see and would lose the 74k-gallery chunking.

---

## 3. Batch 1 vs batch N

Two separate questions, kept separate because they have different answers.

### 3a. Worker lifecycle: one PID for every batch — proven

`Adapter.worker_pids` records the OS pid the worker reports in each reply (`os.getpid()`, sent
inside the reply the adapter asserts on). Final run:

* the recorded PID list holds **34 entries with 1 distinct value** across the whole measurement
  (parity ×6, identity ×6, batch block ×7, interleaved ×7, near-tie ×2, CIRCUS conditions ×6),
* `single_pid = true`, one distinct PID for all of them,
* the `pid_before_this_block` snapshot equals the pid observed afterwards,
* the same PID equals `adapter._worker.pid`, i.e. the child the adapter spawned — not a respawn.

So the model is loaded once per worker, per category lazily: a dress-only block never loaded shirt
or toptee (the worker replies `loaded=[...]` with exactly the categories serviced so far).

### 3b. Row equality across batch sizes: **NOT identical, and it is upstream's behaviour**

This is the one place the brief's expectation and the measurement disagree, so it is reported as
measured.

| quantity | value |
|---|---|
| query rows, batch 20 vs batch 1 | **not equal**, max abs Δ **1.4638714492321014e-02** |
| target rows, batch 20 vs batch 1 | **not equal**, max abs Δ **6.810575723648071e-05** |
| query rows, same batch size repeated | **0.0** (deterministic) |
| query rows, EQUAL-length edits, batch N vs 1 | max abs Δ **8.05e-05** (padding held constant) |
| **native `test.py` query rows, batch N vs 1** | max abs Δ **1.4638714492321014e-02** — identical to the worker |
| **native `test.py` target rows, batch N vs 1** | max abs Δ **6.810575723648071e-05** — identical to the worker |

Two causes, both upstream's, both reproduced exactly by the pinned evaluator on the same inputs:

1. **Text padding is an input.** `TextLSTMModel.forward_encoded_texts` builds an
   `(max_len_in_batch, N)` tensor (`text_model.py:96-98`), so an edit's number of pad columns depends
   on the OTHER edits in its batch; `local_conv`/`global_conv` then sum over that axis. The five
   dress edits are 13, 14, 16, 14, 14 tokens, so batch 1 vs batch 5 changes the pad count of four
   rows. Making every edit equal length collapses the delta from 1.46e-02 to 8.05e-05, which is the
   isolation of this cause.
2. **cuDNN algorithm selection per shape.** What remains (≈1e-5..1e-4) is visible on IMAGE-only
   rows, where no text is involved at all.

Consequence for the campaign, stated plainly: **CLVC-Net feature rows are batch-size dependent, and
so are its native rows.** Batch size is therefore part of the run identity for this model — the same
list of images scored at batch 32 and at batch 1 is not the same measurement. What does NOT change in
the measured sample is the metric verdict:

| category | ranks @ batch 1 | ranks @ batch 32 | ranks identical | top-10 list identical |
|---|---|---|---|---|
| dress | `[1, 1, 10, 1, 1]` | `[1, 1, 10, 1, 1]` | **yes** | no (positions 6-9 swap) |
| shirt | `[1, 2, 2, 1, 2]` | `[1, 2, 2, 1, 2]` | **yes** | no (positions 6-9 swap) |
| toptee | `[1, 2, 1, 1, 1]` | `[1, 2, 1, 1, 1]` | **yes** | no (positions 4-9 swap) |

Max abs score delta between the two batch sizes on non-masked cells: dress **1.85e-02**, shirt
**5.66e-02**, toptee **6.67e-02**. Those are far above any float32 noise, so this is a real
model-level sensitivity, not arithmetic.

The parity in §2 is unaffected because both paths ran at the **same** batch size (upstream's own
loop at `params.batch_size=32`, the worker at `adapter.batch_size`); the 2048-d rows were then
bit-identical. Note that `_encode_queries` may fall back to a per-query call on a batch error
(`runner.py:700-709`), which would silently change batch size for the surviving rows — with this
model that changes the features, so a CLVC-Net run whose query features were partly produced by the
fallback is not comparable to one that was not.

---

## 4. Interleaved categories keep input order

Six queries sent as one job in the order `dress, shirt, toptee, dress, shirt, toptee`, then each row
re-encoded alone through the same worker. Max abs Δ per row: `dress [0.0, 0.0]`,
`shirt [0.0, 0.0]`, `toptee [0.0, 0.0]` — the mixed-category reply places every row at its input
position. The scatter is a named helper (`scatter_rows`) whose row-count mismatch raises
`clvc_worker_row_count`, and the tests pin the interleaved order without a GPU.

---

## 5. Near-tie handling, and what CIRCUS changes

### 5a. Tie rule

Constructed as an EXACT tie rather than hunted for: the gallery image `B00AKLK08G` is appended a
second time, so positions 3 and 20 hold identical bytes and identical features.

| quantity | value |
|---|---|
| scores at the two positions | `0.28079473972320557`, `0.28079473972320557` |
| score delta | **0.0** |
| rank when the FIRST occurrence is the positive | **5** |
| rank when the SECOND occurrence is the positive | **6** |
| repeat of the same computation | identical (deterministic) |
| rule | `1 + #strictly-greater + #earlier-equal` (gallery order breaks ties) |

matches `_ranks_from_scores` exactly, and matches upstream `np.argsort(-sims)` for the same reason
(stable sort on an already gallery-ordered array).

A natural near-tie in this sample: the closest pair of distinct gallery images for query
`B005X4PL1G` is `B00CMPE0C0` / `B00BPYP69K` at scores `-0.028852587565779686` /
`-0.02902510017156601`, gap **1.725e-04** — i.e. **not** below 1e-6, so a sub-1e-6 near tie had to
be constructed, and the constructed one is stronger evidence anyway (delta exactly 0).

### 5b. CIRCUS differences, category `dress`, same 5 queries

Every movement below is attributed to exactly one named change. Score rows for the four
configurations were produced by the same worker with the same images.

| configuration | ranks | explanation of the movement |
|---|---|---|
| (a) 20-gallery, MASKED, native template text | `[1, 1, 10, 1, 1]` | baseline = §2 parity row |
| (b) 20-gallery, UNMASKED, template text | `[2, 1, 11, 1, 1]` | **only** the reference row's `-10e10` removed: queries 0 and 2 are their own best match, so they lose exactly one position. Bit-identical scores otherwise |
| (c) 20-gallery, MASKED, released CIRCUS text | `[1, 1, 12, 1, 1]` | **only** the text changed: query 2 moves 10 → 12 from a different embedding, not from any numeric change |
| (d) FULL 3817-image gallery, UNMASKED, released text | `[69, 2, 2526, 118, 2]` | **only** the gallery grew from 20 to the category's full `split.dress.val.json` set; recall is no longer inflated by a 20-image denominator |
| (e) FULL gallery, MASKED, released text | `[68, 2, 2525, 118, 2]` | mask applied at full scale: exactly one position for the query 0 and query 2 rows that contain their own reference (`B00BPYP69K`'s reference is in the full set, as are all five) |

So: mask → ±1 position on rows whose reference is competitive; text → a real embedding change
(1-2 positions); gallery → the dominant effect (1-2 orders of magnitude). All three are CIRCUS
policy, none is numerical: the numerical difference between the two paths is the 1.19e-07 of §2.

Note for the lead, not a defect in this adapter: at full gallery with the released single-sentence
text, R@10 for these five dress queries is 1-2 of 5 by construction of the sample (query 2 at 2526),
which is why a 5-query sample is only a parity instrument and never a metric.

### 5c. The three conditions

| condition | status | max abs Δ of the query row vs multimodal | unmasked ranks |
|---|---|---|---|
| `multimodal` | runs | 0.0 (it IS the baseline) | `[2, 1, 11, 1, 1]` |
| `text_only_black_image` | runs | **8.92e-02** | `[1, 1, 17, 1, 1]` |
| `image_only_empty_text` | **REFUSED by name** | — | — |

`text_only_black_image` moves ranks because the image branch sees a black image (the text tower is
unchanged), which changes the fused feature by 8.9e-02 — a genuine input change, not a numeric one.

`image_only_empty_text` **cannot run on this model.** Measured, not assumed: the earlier attempt
raised `IndexError: list index out of range` from inside the worker for every row, because
`TextLSTMModel.forward` tokenises the empty edit to `[]` and then indexes `x[0][0]`
(`text_model.py:84-86`). The repair in this change set is a **named refusal** before any launch:

```
adapter_empty_edit_unrepresentable: model=clvc_net rows=[0, 1, 2, 3, 4] count=5:
CLVC-Net cannot encode an empty edit: TextLSTMModel.forward tokenises to zero tokens and indexes
x[0][0] (text_model.py:84-86), which raises IndexError for every row, so image_only_empty_text is
unrepresentable in this model's text tower rather than merely unsupported by the adapter
```

which the runner turns into a per-query error record instead of a failed gallery job. Whether the
condition should be declared `ABLATION_UNSUPPORTED` in the plan is a registry/plan decision the lead
owns: `mode_supported()` is left exactly as it was (it reports on the model's structural state), so
the refusal is a transport-level, per-query report.

---

## 6. VRAM

`nvidia-smi --query-gpu=memory.used,memory.total`, one GPU job at a time inside `gpu_job.sh`:

| point | device used / total | torch peak allocated | torch peak reserved |
|---|---|---|---|
| before (idle, other project) | `886 MiB / 16303 MiB` | 0 | 0 |
| after dress parity (both pickles resident) | `2433 MiB / 16303 MiB` | 641.1 MiB | 966.0 MiB |
| after shirt parity | `2815 MiB / 16303 MiB` | 646.0 MiB | 966.0 MiB |
| after toptee parity (all three pairs resident) | `3077 MiB / 16303 MiB` | 646.0 MiB | 966.0 MiB |
| during full-gallery block | `3079 MiB / 16303 MiB` | 646.0 MiB | 966.0 MiB |
| after `close()` (separate teardown run) | `909 MiB / 16303 MiB` — back to its own idle baseline | — | — |

The `torch.*` columns are the PARENT process, which holds no model: the checkpoints live in the
worker child, which is why device usage rises while `peak_allocated` stays at 646 MiB of activations.
A dedicated teardown run measured `909 MiB → 2889 MiB (all three pairs resident) → 909 MiB after
close()`, i.e. nothing stays resident.

Peak device usage attributable to this work: **~3.1 GiB / 16303 MiB (≈19 %)**, of which the three
checkpoint pairs are the bulk. Nothing was left resident: after `close()` the device returns to the
idle baseline.

---

## 7. `adapted_runtime` block the lead must add to `workbench/registry/models.yaml`

`workbench/registry/models.yaml` is lead-owned; **no edit to it was made by this work.** The block
belongs at `clvc_net` → `environment.adapted_runtime` (the location all nine existing declarations
use — never model top level; 85 §2). Measured values, taken from the provisioned interpreter
`workbench/envs/clvc39-la/bin/python`:

```yaml
    adapted_runtime:
      target_compute_capability: sm_120
      python: '3.9'
      pytorch: ==2.7.1
      torchvision: ==0.22.1
      cuda: '12.8'
      packages:
      - numpy<2
      pip_index_url: https://download.pytorch.org/whl/cu128
```

Measured basis, one line each:

* `python: '3.9'` — `sys.version` = `3.9.25 (main, Oct 31 2025, 23:00:23) [Clang 21.1.4]`.
* `pytorch: ==2.7.1` / `torchvision: ==0.22.1` / `cuda: '12.8'` — `torch.__version__` =
  `2.7.1+cu128`, `torch.version.cuda` = `12.8`, `torchvision.__version__` = `0.22.1+cu128`.
* `target_compute_capability: sm_120` — `torch.cuda.get_device_capability(0)` = `(12, 0)` on
  `NVIDIA GeForce RTX 5070 Ti`; this is the same target the nine existing blocks declare.
* `packages: [numpy<2]` — `numpy.__version__` = `1.26.4` in that interpreter. `<2` is declared
  rather than `==1.26.4` because nothing here is numpy-ABI critical (the only numpy use is `np.save`,
  `np.linalg.norm` and `np.dot` over float32 arrays); the constraint exists so a cu128 image that
  ships numpy 2.x is refused instead of silently changing the ABI of the pickled modules.
  `torchvision` is deliberately NOT repeated in `packages`: it is already pinned by the
  `torchvision:` key, and `resolved_runtime_environment` would list it twice.
* `pip_index_url: https://download.pytorch.org/whl/cu128` — the wheel index that carries the
  `+cu128` builds above (same value as all nine existing blocks).

`WORKBENCH_CLVC_PYTHON` must also be set (process env or `workbench.env`, **not** edited here) to
`/home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python`; without it
`resolve_model_interpreter` returns `None` (no `environment_assignment.json` exists for this model)
and the adapter refuses at `no interpreter environment resolved`.

**Dry-run of the gate, without touching the registry** (deep-copied model dict,
`runtime.verify_environment(..., probe=True)`, no GPU, no model load):

```
BASELINE tier/status: BLOCKED PYTHON_VERSION_MISMATCH
BASELINE reason: expected Python 3.7.6, found 3.9.25
ADAPTED  tier/status: GPU_VERIFICATION_PENDING  (in READY_TIERS -> gate passes)
ADAPTED  reason: CUDA capability must be verified on the GPU host with verify_gpu_runtime()
                 before this model can be launched
resolve_model_interpreter -> workbench/envs/clvc39-la/bin/python
```

So the declared block alone moves the gate from `BLOCKED` to `GPU_VERIFICATION_PENDING`. A real
LAUNCH of this CUDA-declaring model still needs `verify_gpu_runtime()` to promote it to
`GPU_RUNTIME_READY` (`LAUNCH_READY_TIERS`), which is the operator's existing GPU-verification step,
not a code gap here.

---

## 8. Tests

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_clvc_net_composition_parity.py workbench/tests/test_clvc_net_replay.py -q
26 passed in 0.16s
```

Cross-check that this change did not disturb anyone else's assertions (both files are lead-owned and
were NOT edited):

```
pytest workbench/tests/test_composition_adapters_missing.py \
       workbench/tests/test_circus_env_gate_omp.py \
       workbench/tests/test_blocked_models_verification.py -q
82 passed, 60 warnings in 8.52s
```

In particular `test_status_refuses_rather_than_claiming_readiness[clvc_net]` still passes:
`status()` reaches the shared `_environment_gate_reason` once the structural checks pass, and the
gate still refuses because the registry declares python 3.7.6 / torch 1.6.0. `_run`'s signature is
unchanged, so `test_adapter_does_not_rewrite_the_texts_it_is_given` still monkeypatches it, and the
module still calls `require_verified_interpreter()` before its `Popen`
(`test_the_gate_requires_an_interpreter_before_the_first_launch` passes on the AST check).

What the 20 new tests decide without weights, GPU or network:

* interleaved-category row order is preserved (`scatter_rows`, `group_indices`, plus a per-category
  interleaving parametrisation);
* a mixed-category target gallery is refused (`scopes must not be mixed`, and an out-of-set scope);
* a `None` query category is refused with `adapter_query_category_missing`; an unknown one with
  `adapter_query_category_unknown`; an empty edit with `adapter_empty_edit_unrepresentable`;
* missing `work_root` raises `AdapterWorkRootError`;
* digest mismatch is refused **before** `torch.load` — the real loader chain is exercised with an
  instrumented `torch.load` and the call list is asserted empty; a member with no declared digest is
  refused too;
* worker reply shape mismatch raises `adapter_worker_invalid_matrix`, and the job payload's texts
  arrive verbatim;
* the score equivalence identity holds on synthetic unit-norm halves (`(dot_local + dot_global)/2`,
  `atol=1e-6`), and rows produced by the encoders are unit-norm per half;
* the declared contract (2048-d, three category scopes, `reference_eligible`, `_run` signature).

---

## 9. What this change does and does not touch

* `workbench/composition/model_clvc_net.py` — the refusal-only stub is replaced by a real
  per-adapter resident worker: one subprocess per adapter instance, model loaded once per worker and
  per category lazily, digest verified before each `torch.load`, rows scattered back to input order,
  named errors carrying the worker log path, `close()`/`restart()` semantics matching
  `model_encoder.py`. `reference_eligible`, `category_scopes`, `score_dimension` and
  `mode_supported`'s condition handling are unchanged; the shared environment gate still runs before
  the first launch and was NOT relaxed.
* `workbench/tests/test_clvc_net_composition_parity.py` — new.
* This file — new.
* Not touched: `model_tgcir.py` and TG-CIR files, `models.yaml`, `workbench.env`,
  `test_composition_adapters_missing.py`, `workbench/frontend/`, ports 8000/5173.

---

## Còn chặn

1. **No end-to-end CIRCUS run.** No numbers here are a metric: every measurement is a 5-query,
   20-image parity sample (and a 3817-image single-category gallery for the §5b gallery effect).
   Full run counts (6016 release queries, 74381-image gallery), the per-condition aggregate and R@1/10/50
   for `clvc_net` remain unmeasured.
2. **Batch-size dependence is real, unfixed, and upstream-native** (§3b). Query rows move by up to
   1.46e-02 between batch 1 and batch 20 and metric ranks happened to hold on this sample; the top-10
   lists did not. The pinned evaluator moves identically, so there is nothing to "fix" in the adapter
   without diverging from upstream — but the runner's per-query fallback path
   (`runner.py:700-709`) would silently mix batch sizes inside one run. A CLVC-Net run must pin
   `WORKBENCH_COMPOSITION_BATCH_SIZE`, record it, and refuse a partially-fallback run. Not implemented
   here (the runner is not in my ownership).
3. **`image_only_empty_text` is refused, not run.** It is unrepresentable in this text tower
   (§5c). Whether the plan should declare the condition `ABLATION_UNSUPPORTED` rather than let each
   query fail is a registry/plan decision; `mode_supported` was deliberately left as it was.
4. **Released text ≠ native text.** Native edits are `"<BOS> cap0 <AND> cap1 <EOS>"`
   (`datasets.py:174`); the CIRCUS release ships one sentence per query. The adapter transports the
   text verbatim, so a CIRCUS run of this model does not reproduce native feature vectors. §5b (b)-(e)
   quantifies the rank effect on 5 queries; the aggregate effect is unmeasured.
5. **ResNet-50 fc weights and the torchvision cache.** The six pickles carry 318 of 320 ResNet-50
   tensors and `model.py:78` replaces `fc` with an empty `nn.Sequential`, so the cache cannot change a
   scored parameter (85 §5) — but the load still depends on an unpinned `~/.cache/torch` entry for
   construction. Not exercised away here.
6. **No `verify_gpu_runtime()` record** for this model, so even with the registry block the gate stops
   at `GPU_VERIFICATION_PENDING` until the operator's GPU verification writes one.
7. **`numpy<2` in the block is a range, not a pin.** If the operator later provisions a different
   numpy 1.x, nothing here would notice; the block follows the nine existing declarations' style
   rather than inventing a stricter one for this model alone.
8. **Sample size.** 5 queries per category, chosen by file order, cannot detect a systematic
   difference smaller than the sample's rank granularity (1/5 = 20 points). Parity is proven at the
   level of float32 arithmetic; it is not a metric-level proof.
