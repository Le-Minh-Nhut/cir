# FINDINGS — divergences, defects, and what is not proven

Date 2026-11-08. Companion to `EQUIVALENCE.md`; tolerances in `TOLERANCES.md`.

## F1. `optimize=True` / chunked GEMM changes 207 of 18,009 target ranks — not rank-identical

**Stage where divergence first appears:** the score matrix, not the rank rule. `score()` returns
different float32 values (max abs delta 8.94e-08 between the positive's score under the two paths);
`_ranks_from_scores` is the shipped function and is unchanged.

| condition | changed | max \|Δrank\| | K=10 | K=50 | set | order |
|---|---:|---:|---:|---:|---:|---:|
| multimodal | 4 | 1 | 0 | 0 | 0 | 7 |
| text_only_black_image | 35 | 2 | 0 | 0 | 0 | 7 |
| image_only_empty_text | 168 | 5 | 0 | 0 | 0 | 5 |

Changed qids (multimodal, all four): `7:574` 1628→1627, `7:666` 42456→42455, `7:690` 63717→63716,
`7:2253` 17475→17474. Full lists for the other two conditions are in
`fastpath_vs_stored.json` → `conditions.<c>.paths.opt.changed_all`.

**Score margins at the changed ranks:** |margin| ≤ 8.94e-08, i.e. 1–2 ULP at these magnitudes.
Tie interactions: `7:574` and `7:666` had an exact tie (multiplicity 2) in the shipped path and
multiplicity 1 in the fast path; `7:690` and `7:2253` had multiplicity 1 both ways and the ordering
was decided by rounding.

**A rank change confined to ranks 1k–74k cannot change R@10 or R@50** (both are `r <= K`), and none
does: **0 crossings out of 18,009**. nDCG does move, because it reads the exact rank: −1.58e-08 to
+1.55e-09 per condition, composition gap −2.11e-09.

## F2. 19 Top-K windows change ORDER with identical membership

Not a set difference: `set_differs = 0` for all 18,009. But 19 windows order the same 50 ids
differently, which violates G3's "identical candidate ordering". Examples with the swap pair and the
scores under each path are in `EQUIVALENCE.md`; the machine-readable list is
`divergence_detail.json` → `order_only_divergences`.

Important distinction: **4 of the 19 are exact-tie artefacts** (shipped multiplicity 2, fast 1 or
vice versa), the other 15 are genuine 1-ULP reorderings of near-duplicate rows. A reader who treats
all 19 as "tie policy" would be wrong.

Near-duplicate gallery rows are the mechanism. `7:56` and `7:1798` swap the *same* pair (43765,
55217) at the *same* position (21) in the image-only condition, and in the fast path those two rows
score **exactly equal** (0.5519279837608337 both) while the shipped path does not. That is a pair of
near-identical gallery images, not a query pathology.

## F3. Feature cache identity omits 7 inputs; the sidecar protects nothing

See `EQUIVALENCE.md` G4b for the table. Two exposures are severe enough to restate:

1. `scoring_module_source_hash` is **absent from run `497c25c57fbeba885ecfb9d1`** — the key was added
   after this artifact was written. This is exactly the run the user is reasoning about, and its
   identity cannot distinguish a change to the shared scoring helpers.
2. `_load_memmap` compares only the sidecar fingerprint. A copied 2.3 GiB `.npy` with a rewritten
   `{"fingerprint": <expected>, "shape": [...], "dtype": "float32"}` sidecar is accepted with no
   inspection of the numbers. Feature arrays are never verified by content.

Not tested by execution (would require writing a fake cache into a run directory, which the audit
rules forbid). The claim is read from `runner._load_memmap` and from the four sidecar files' exact
key sets (`dtype`, `fingerprint`, `shape`).

## F4. Systematic ties: 851 duplicate gallery rows

2386 of 74,381 gallery positions hold byte-identical feature rows; largest group 54. These are
duplicated/identical FashionIQ images. Since `_ranks_from_scores` counts exact equality, every rank
inside such a group depends on gallery order, and the more so when a query's score vector contains
many exact duplicate values (every sampled query did: ~2.5k–7k duplicates per row).

Consequence for the fast-path decision: the 207 rank changes are not random noise, they are
concentrated where the float32 score surface has 1-ULP gaps. Any future scoring change will hit the
same places. Reported only; no fix.

## F5. ENCODER reference transform is a real protocol divergence (numerical side)

Upstream `third_party/ENCODER/datasets.py::FashionIQ.get_test_data` encodes the reference with
`get_img(..., stage=0)` and gallery rows with `stage=1`, and `evaluate_model.py` builds
`transform=[preprocess_train, preprocess_val]`. So **upstream reference = `preprocess_train`,
gallery = `preprocess_val`**; the workbench adapter applies `preprocess_val` to both
(`model_encoder.py:321,340`). Another agent settled the source reading; these are my numbers.

`preprocess_train` is `RandomResizedCrop(224, scale=(0.9,1.0))` — **stochastic**:

| measurement | value |
|---|---|
| same image, 3 draws, pairwise max abs delta | **3.8269808292388916** (all three pairs) |
| `preprocess_val` same image twice | **0.0** (deterministic) |
| same torch seed, two encodes | **identical, 0.0**; different seed → 0.3572 |

So upstream ENCODER is reproducible **only** with an explicit seed, which the author's evaluation
script never sets. ENCODER's published number is therefore not reproducible upstream as written.

Scale of the change on 64 real reference images:

| level | max abs delta | mean abs delta |
|---|---:|---:|
| normalized pixel tensors | **3.8269808292388916** | 0.2487 |
| ViT-B-32 image features | **1.2528** | 0.1350 |
| cosine(val, train) per reference | min **0.8261**, mean 0.9276, max 1.0 |

Rank impact, two independent proxies (no ENCODER gallery cache survives, so the 74,381 rank cannot
be recomputed):

| proxy | result |
|---|---|
| self-consistent 512-image mini-gallery | 25/64 queries' top-1 changes; 63/64 full order changes; max top-1 shift 234 positions |
| real (reference, positive) pairs over a 128-row mini-gallery | **48/64 ranks change**, max \|Δrank\| **51**, R@10 23→24 of 64, R@50 48→48, nDCG 0.29443→0.29472 |

Assessment requested: **the delta is large** — features move by 1.25 in absolute terms and per-query
ranks move by up to 51 in a 128-row gallery. This is not a rounding effect and cannot be dismissed as
tie policy. It **would** move R@10/R@50 materially if the gallery were the full one and references were
encoded stochastically, because unlike the fast-path case these changes are **not** confined to deep
ranks. The published 31.02% is internally consistent with the declared (`val`) transform but is **not**
an upstream reproduction; the row's provenance must say so, and the `preprocess_train` reference path
must not be silently adopted either, because unseeded it is not reproducible at all.

## F6. Query-axis parallelism (`WORKBENCH_SCORE_THREADS`) — NOT_RUN

Another agent reports splitting the 6003 queries across 8 threads is bit-identical (delta exactly 0.0,
0 rank changes for sub-batch sizes 2–128). **I did not measure this and have no independent evidence
for or against it.**

What I can contribute without running it, all of which is consistent with the claim and none of which
proves it:

- The claim is structurally plausible: each query's score row is computed by a dense operation over
  the full gallery, so splitting the *query* axis changes which rows are computed together but not the
  per-row reduction order. That is a materially different operation from `optimize=True`/GEMM
  chunking, which changes the reduction order along the 256-dim axis and the token-max path.
- My own measurements show the two *order-changing* paths agree with each other bit-for-bit (`opt` vs
  `gemm1024` on every one of 207 changed ranks), which supports the idea that divergence here comes
  from reduction order, not from thread count.
- Decode parallelism at 2/4/8 workers is bit-identical for two transform families (16/16), so
  "parallelism per se is safe" has supporting evidence at the pixel stage.

None of that is a measurement of `WORKBENCH_SCORE_THREADS`. It should be `NOT_RUN` in the release
documentation until someone runs the same 18,009-row comparison anchored on the stored artifact — the
harness in `scripts/audit/equivalence_fastpath.py --verify-sample` is that comparison and would settle
it in ~15 minutes.

## F7. Scope of the loudest warning

The 207/18,009 figure is one model (habit) with one 32×256 token layout. CONESEP/CSMCIR/PTHA use a
different subscript (`nmk`) and already ship `optimize=True`; ENCODER and LIMN use `@` with 8192-row
gallery chunks in the controller. A uniform scoring function therefore does **not** currently exist
across the nine adapters, and the per-model rank sensitivity to a summation-order change has been
measured for exactly one of them. Adopting the fast path uniformly without per-model re-verification
would generalise a single measurement across six unmeasured code paths.
