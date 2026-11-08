# G2 / G3 / G4 equivalence audit — OPTIMIZED path vs verified sequential baseline

Date 2026-11-08. Branch `research/cir-failure-workbench`. No commit, no push, no git state touched.
Tolerances declared in `TOLERANCES.md` **before** any measurement. No tolerance changed after a result.

**Overall verdict: `PARTIALLY_VERIFIED`.** The decode/throughput work is bit-identical and safe. The
proposed *scoring* change (`optimize=True` / chunked GEMM) is **not** rank-identical: 207 of 18,009
query-conditions change target rank, and 19 Top-K windows change order though not membership. Zero
queries cross K=10 or K=50, so R@10/R@50 are bit-identical, but per-query ranks and Top-K ordering
are not.

Anchor run: `497c25c57fbeba885ecfb9d1` (habit / `fiq_n02`), gallery `74381×32×256`, 6003 queries ×
3 conditions, `ranks/habit_fiq_n02.json` + published depth-50 `candidates/habit_fiq_n02/*.jsonl`.

**Anchor correctness verified, not assumed.** The harness recomputes the shipped
`np.einsum("nd,gqd->ngq")` scorer for sampled queries and requires exact agreement with the stored
artifact: **12/12 sampled queries per condition, 0 mismatches, all three conditions**
(`fastpath_vs_stored.json` → `anchor_verification`).

---

## Gate matrix

| Gate | Question | Verdict | Numbers |
|---|---|---|---|
| **G2 habit decode** | Does parallel decode change the pixel matrix for `targetpad_transform(1.25,224)`? | **PASS** | 512 real gallery images; workers {1,2,4,8} × batch {1,32,64,256}; **16/16 bit-identical**, `max_abs_delta = 0.0`, `bytes_differing = 0`, single sha256 `5fc8002d…` |
| **G2 encoder decode** | Same for `open_clip` `preprocess_val` (ViT-B-32) | **PASS** | Same grid, **16/16 bit-identical**, `0.0`, single sha256 `4f6a14f9…` |
| **G2 forward features** | Do the *encoded* feature matrices agree? | **NOT_RUN** | GPU held by the live `intent` cohort for the entire audit window (`nvidia-smi`: pid 267905, ~3.6 GiB). Pixel equality does not imply feature equality. |
| **G3 scoring, same arithmetic order** | Does regrouping the query axis (batch size) change ranks? | **PASS on the shipped path** | 0 rank mismatches at batch 1 / 7 / 256 vs 32 over a declared 96-query slice per condition; 0 at repeat; incomplete-final-batch grouping leaves the first 32 rows unchanged. Structurally each query's score row is an independent dense reduction, so row grouping cannot alter it. |
| **G3b `optimize=True` / chunked GEMM** | Does a changed summation order change ranks? | **FAIL on rank identity; PASS on R@10 / R@50 / candidate SET** | 4 + 35 + 168 = **207 / 18,009 rank changes**; **0** K=10 crossings; **0** K=50 crossings; Top-K set differs **0/18,009**; ordering differs **7 + 7 + 5 = 19** |
| **G4a batch 1 vs 32** | — | **FAIL on the GEMM fast path; PASS on the shipped path** | GEMM: **6/6003** multimodal rank changes at batch 1 vs 32 (`7:666`, `7:1988`, `7:2003`, `7:2142`, `7:2253`, `7:4730`); batch 256 vs 32: 0. Shipped path: 0 mismatches at batch 1/7/256 over a 96-query slice. `opt`: 0 mismatches over a 600-query slice. |
| **G4b incomplete final batch** | — | **PARTIAL** | 6003 % 32 = 19; the tail's existence provably cannot change earlier rows (each batch's scores depend only on that batch) |
| **G4c repeat determinism** | — | **PASS** | Two identical full runs of the fast path agree exactly; `divergence_detail.json` shows the two fast paths agree on every changed query and on every new rank |
| **G4d cold vs warm cache** | — | **NOT_RUN** | See below. |
| **G4e cache identity completeness** | Can a cache be reused under a different computation? | **DEFECT, partially remediated** | Recorded verdict: 7 inputs absent from `identity`. Closed since: BLAS/OMP/MKL thread count (`blas_threads_pinned`) and `scoring_module_source_hash`. Still open: TF32/cuDNN flags, GPU capability/driver/CUDA version, transform source (registry labels only), and byte-level array verification — the sidecar checks the fingerprint, never the bytes. Also open: 851 duplicate gallery feature rows over 2,386 positions (largest group 54). |
| **G3c tie policy** | Do systematic ties make ranks order-dependent? | **DEFECT (reported, not fixed)** | 851 duplicate gallery feature rows over 2386 positions; largest group 54 |

---

## G3b — the full 6003 × 3 result (the decision)

Two fast paths were measured: `optimize=True` and chunked GEMM (1024-row gallery chunks). They agree
with each other on **every** changed query and on **every** new rank value
(`divergence_detail.json` → `fastpath_agreement` → `identical_changed_query_sets_and_new_ranks: true`
for all three conditions). Speed: shipped ≈ 5.6 q/s, `optimize=True` 27–43 q/s, chunked GEMM
70–75 q/s under `OPENBLAS_NUM_THREADS=2`.

| condition | rank changes / 6003 | max \|Δrank\| | K=10 crossings | K=50 crossings | Top-K set differs | Top-K order differs | R@10 Δ | R@50 Δ | nDCG Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `multimodal` | **4** | 1 | **0** | **0** | 0 | **7** | **0.0** | **0.0** | +1.4044e-09 |
| `text_only_black_image` | **35** | 2 | **0** | **0** | 0 | **7** | **0.0** | **0.0** | +1.5525e-09 |
| `image_only_empty_text` | **168** | 5 | **0** | **0** | 0 | **5** | **0.0** | **0.0** | −1.5752e-08 |
| **total** | **207 / 18,009** | **5** | **0** | **0** | **0** | **19** | **0.0** | **0.0** | — |

Aggregate metrics, stored vs fast (full precision, `gap_and_axes.json`):

| metric | stored | fast (`opt` == `gemm1024`) | delta |
|---|---:|---:|---:|
| MM nDCG | 0.32828280509551133 | 0.3282828064999273 | +1.4044159724058147e-09 |
| text nDCG | 0.2006777268917305 | 0.20067772844423498 | +1.5525044860797976e-09 |
| image nDCG | 0.11373588650412819 | 0.11373587075219026 | −1.5751937930693848e-08 |
| **composition gap nDCG** | 0.388704727214254 | 0.38870472510024856 | **−2.1140054151658205e-09** |
| MM R@10 (%) | 35.61552557054806 | 35.61552557054806 | **0.0** |
| MM R@50 (%) | 54.3728135932034 | 54.3728135932034 | **0.0** |
| MM MRR | 0.20496768403872728 | 0.20496768410229743 | +6.357e-11 |
| mrr_gap | 0.11678778274193043 | 0.11678778259623385 | −1.457e-10 |
| scored counts (mm/text/image) | 6003 / 6003 / 6003 | 6003 / 6003 / 6003 | **0** |

**Rank-change depth.** Every change is deep. Depth histogram of `min(old_rank, new_rank)`:

| condition | 1–10 | 11–50 | 51–100 | 101–1000 | 1001–5000 | 5001–20000 | 20001–74381 |
|---|---:|---:|---:|---:|---:|---:|---:|
| multimodal | **0** | **0** | **0** | 0 | 1 | 1 | 2 |
| text_only_black_image | **0** | **0** | **0** | 1 | 2 | 2 | 30 |
| image_only_empty_text | **0** | **0** | **0** | 1 | 3 | 4 | 160 |

**A rank change confined to rank ~1k–74k cannot alter R@10 or R@50**: both cutoffs are booleans on
`r <= K`, and zero changed ranks cross them. It *does* alter nDCG, because nDCG uses the exact rank;
the measured nDCG movement is 1e-9 to 1.6e-8.

### Mechanism: 1-ULP gap between near-duplicate gallery rows

Every changed rank sits on a pair of gallery rows that are near-duplicates in feature space. Two
distinct mechanisms occur and must not be conflated (from `margins` and `divergence_detail.json`):

1. **Exact tie split by 1 ULP.** The shipped path scores the positive *exactly equal* to another row;
   `_ranks_from_scores` adds the `#earlier-equal` term. The fast path's different accumulation order
   moves the positive by ~3e-08, the equality disappears, and the rank moves by 1.
   `7:574`: shipped 0.3319489359855652, tie multiplicity **2**, rank 1628; fast 0.3319489657878876,
   multiplicity **1**, rank 1627; margin −2.98e-08. `tie_interaction: true`.
2. **Genuine near-tie decided at the last bit.** Both paths have multiplicity 1 but the two
   neighbours differ by <5e-08, so the rounding decides the order.
   `7:690`: shipped 0.08333621919155121 (rank 63717) vs fast 0.08333626389503479 (rank 63716),
   margin −4.47e-08. `tie_interaction: false`.

Measured tie interactions among changed ranks: 2/4 (multimodal), 16/35 (text-only), 59/120 sampled
(image-only; the sample cap is 120 of 168). Max |score margin| observed: **8.94e-08**.

### Top-K ordering: 19 windows, membership identical

`set_differs = 0` for all three conditions: **no query's depth-50 candidate ID *set* changes**.
`order_differs_same_set = 7 / 7 / 5`: 19 windows present the same 50 ids in a different order. These
are adjacent swaps of a near-duplicate pair, and in **most** of them the shipped path did *not* have
an exact tie, i.e. the swap is a genuine 1-ULP reordering, not only a tie-policy artefact:

| qid | position | shipped order (scores) | fast order (scores) | shipped exact tie? |
|---|---:|---|---|---|
| `7:1947` | 4 | [21911, 12676] (0.7409913539886475, 0.7409912943840027) | [12676, 21911] | no |
| `7:2586` | 2 | [4048, 11801] (0.8123618960380554, 0.8123618364334106) | [11801, 4048] | no |
| `7:4437` | 27 | [19609, 19666] (0.6810510158538818, equal) | [19666, 19609] | **yes** |
| `7:2534` | 45 | [435, 34017] (equal) | [34017, 435] | **yes** |
| `7:56` | 21 | [43765, 55217] (0.5519279837608337, 0.551927924156189) | [55217, 43765] | no |

So `G3`'s "identical candidate ordering" requirement **fails** under the fast path: 19/18,009.
Membership passes.

---

## Verdict on adopting the fast path

**Not acceptable as a silent swap. Acceptable only as a declared scoring-function change with a new
`run_id` and a relabelled artifact.**

What is preserved: R@10, R@50 (bit-identical), the scored cohort and denominators, and the Top-K
candidate *set* for every query.

What is **not** preserved: 207 per-query target ranks (up to 5 positions), 19 Top-K orderings, nDCG
to O(1e-8), the composition gap to −2.1e-09, and MRR to O(1e-11).

Under the spec's own framing this is legal *only* as a provenance-visible change: the spec forbids
altering the official computation to make comparison easier, and it requires every summary to carry
protocol, preprocessing, checkpoint and runtime identity. A different summation order is a different
computation for the quantities the spec calls exact integers. Therefore:

1. The change must mint a new `run_id` (it does: `score()` source feeds
   `adapter_source_hashes`/`adapter_module_source_hash`). Old rows stay visible as
   **pre-scoring-change artifacts**.
2. Every published row must be re-run so the whole table uses one scoring function — a mixed table
   is scientifically invalid, as stated.
3. The summary must record the scoring path explicitly (e.g. `scoring_impl: einsum_optimize` /
   `gemm_chunk1024`), not merely the source hash, plus a note that per-query ranks differ from the
   einsum baseline by up to 5 positions while K-hit metrics are unchanged.
4. Any downstream artifact holding per-query ranks across models (e.g. a cross-model rank table)
   must not mix pre- and post-change runs. The pre-existing asymmetry — CONESEP/CSMCIR/PTHA already
   use `optimize=True`, HABIT/INTENT/HINT/TME do not — is the same defect in the current cohort and
   is the strongest argument for the uniform path, *provided* it is declared.

My judgement: **adopt it, declared, and re-run everything.** 18× wall time for a 1e-8 nDCG movement
and zero K-boundary crossings is a good trade for a research table, and the current state — two
different summation orders in one published cohort — is worse than either uniform choice.

---

## G4 axes

| axis | result | scope |
|---|---|---|
| batch 1 vs 32 (GEMM fast path, full 6003) | **6 rank mismatches**; qids in FINDINGS F8a | full |
| batch 1 vs 32 (`opt`, 600-query head) | 0 mismatches | 600-query declared slice |
| batch 1 / 7 / 256 vs 32 (shipped path) | **0 mismatches each** | 96-query declared slice |
| incomplete final batch | 6003 % 32 = 19; no earlier row can change, batches are independent | reasoning + slice |
| repeated identical run | **0 rank mismatches, 0 window mismatches** (`divergence_detail.json`) | full |
| batch 256 vs 32 (GEMM, full) | 0 mismatches | full |
| chunk 512 / 4096 vs 1024 (GEMM, full) | 0 mismatches each | full |
| chunk 512 vs 1024 vs 4096 | **0 mismatches each** | full fast path |
| **batch axes for text_only / image_only** | **NOT_RUN** | the axes job was killed after the multimodal condition to stop competing with the live cohort |
| cold vs warm cache | **NOT_RUN** | see below |

---

## G4b — cache identity: the defect

Cache mechanics, read from the code: `runner._load_memmap` accepts a cache when the **sidecar
`.json`** `fingerprint` equals the expected digest and the `.npy` exists with the right shape. The
sidecar holds exactly `dtype`, `fingerprint`, `shape`. **Nothing hashes the array contents.** So a
copied `.npy` plus a rewritten sidecar is accepted, and the fingerprint is of the *identity*, not of
the numbers.

For run `497c25c57fbeba885ecfb9d1` the identity contains
`adapter_module_source_hash`, `adapter_source_hashes` (5 methods),
`condition_transform_source_hash`, `gallery_identity`, `gallery_image_identity`,
`query_image_identity`, `query_order_identity`, `checkpoint_identity`, `runtime_identity`,
`preprocessing`, `batch_size` (32), `adapter_batch_size` (32), `seed` (0), `protocol_id`,
`subset_memberships`, `conditions`, `score_signature`.

**Inputs that affect a cached feature matrix but are NOT in this identity** (7):

| missing input | why it matters |
|---|---|
| `scoring_module_source_hash` | **absent from this run entirely** (added later). The shared helpers in `composition/adapters.py` were unhashed when this artifact was written. Even when present it hashes only that one file. |
| BLAS / OMP / MKL intra-op thread count | `ein_threads.txt` shows it changes fast-path score bits (2.38e-07 at 10 threads vs 3.58e-07 elsewhere), so it can move a score-derived rank. Not recorded by name. |
| GPU compute capability / driver / CUDA version | only the device *name* is recorded (`gpu_identity`); the capability that selects kernels is not a named field. |
| tf32 / cudnn benchmark / deterministic flags | not recorded; can change GPU forward bits. |
| decode worker count | deliberately excluded because decode is bit-identical; **this audit re-measured that for two transform families** (16/16). Legitimately absent, but it *is* an input. |
| the transform's actual code and parameters | `preprocessing.registry_values` records registry **labels** (`fashioniq_layout`, `image_size`, `transform`), not the third-party transform's source. A change inside the vendor's transform function is undetected. |
| feature array **content** | no byte-level verification anywhere; the sidecar fingerprint covers identity, not bytes. |

Additional identity weaknesses worth naming: `torch`/`numpy` versions appear only indirectly, via
`runtime_identity.environment_fingerprint` (a digest of an environment record, not of the imported
versions), and `batch_size` is recorded but `keep_features`/`topk_depth` are not — those do not
change features, so that one is correct.

**A feature cache can therefore be reused under a different computation.** That is a defect by the
audit's own definition. Concrete exposures: (a) rewriting a sidecar makes stale numbers look fresh;
(b) a scoring-path change that only alters `score()` inside `model_*.py` mints a new `run_id`, but a
change to a shared helper outside `adapters.py` does not; (c) BLAS thread count changes can alter
fast-path scores with an unchanged identity.

---

## G3c — tie policy

`_ranks_from_scores` counts exact equality, so rank depends on gallery order at a tie. The real
gallery features contain **851 byte-identical duplicate row groups covering 2386 positions**, the
largest group holding **54** identical rows. Examples: `[292, 293, 294, 295, 296]` (5-way),
`[164, 11026, 11027]`, `[185, 186]`.

In a 200-query census per condition: **all 200 queries have at least one exact duplicate score
pattern** (~2500–7000 duplicate values per score row, which is expected at float32 over 74,381 rows);
the positive is exactly tied in 2/200 (multimodal), 2/200 (text-only), 7/200 (image-only); and the
rank depends on tie order in 2, 2 and 5 of 200 respectively. Extrapolating that rate to 6003 queries
puts roughly 60–150 ranks per condition on tie order — consistent with the 207 changed ranks being
concentrated in exactly these near-duplicate groups. No top-1 is tied in any sample.

Reported, not fixed, as instructed.

---

## Reproducing this audit

Scripts: `workbench/scripts/audit/equivalence_*.py` (manifest, decode, fastpath, divergence detail,
cache/tie census, encoder transform, encoder rank impact, batch sweep). Each writes one JSON into
this directory and is runnable standalone under the interpreter named in its docstring.

`workbench/tests/test_equivalence_invariants.py` — 5 tests, **5 passed**. It pins the local rank-rule
copy against `CompositionAdapter._ranks_from_scores`, pins the tie-order semantics, pins that
`topk_from_scores` breaks a tie by earlier gallery position, pins `prefetch_rows` order preservation,
and asserts that `optimize=True` is not bit-equal to the plain einsum (the premise that forced this
audit). Synthetic data only; no weights, gallery or GPU.

## Honest scope statement

Every number in this document is from **one model** (habit) **one checkpoint** (`fiq_n02`) on **one
host**, from cached artifacts that already existed. Six of the nine adapters use `@`/`optimize=True`
code paths this audit did not rank-compare, and no forward-pass equivalence was measured. The
decode bit-identity result covers two transform families out of three; LIMN's `open_clip.image_transform`
was not measured.

## NOT_RUN list

| item | reason |
|---|---|
| **G2 forward** (encoded feature matrices per transform family) | GPU owned by the live `intent` cohort all window; starting a competing run was forbidden and would not have produced comparable numbers. |
| **G4d cold vs warm cache** | Only one warm instance of the habit cache exists on disk (all other models' caches deleted); producing a cold counterpart needs a full 74,381-image GPU re-encode. Simulating it would measure this script, not the runner. |
| **Query-axis parallelism (`WORKBENCH_SCORE_THREADS`, 8 threads, bit-identical claim)** | **Not measured by me.** No evidence for or against. I neither endorse nor dispute the claim; see FINDINGS. |
| G3 scoring batching on the *shipped* path over all 6003×3 | Shipped einsum is 5.6 q/s → ~55 min per configuration; the invariant is also structurally guaranteed (each query's dense row is independent). Partial slice evidence only. |
| ENCODER / LIMN full-gallery rank impact of `preprocess_train` | No ENCODER gallery feature cache survives on this host. Proxy measured; labelled PROXY. |
