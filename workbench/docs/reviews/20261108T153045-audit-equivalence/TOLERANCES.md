# Declared tolerances — G2/G3/G4 equivalence audit

Written 2026-11-08 BEFORE any measurement in this directory. No tolerance below was
changed after a result was seen. If a later file disagrees with this one, this one wins
and the disagreement is a defect in the later file.

Scope: `research/cir-failure-workbench`, run `497c25c57fbeba885ecfb9d1` (habit/fiq_n02),
gallery `74381×32×256`, query sets `6003×256` × 3 conditions.

## Classes of comparison

| Class | Objects | Tolerance | Rationale |
|---|---|---|---|
| A. Exact identity | query ids, gallery positions, target ranks, Top-K candidate ids, Top-K ORDER, scored/unscored membership, denominators, hit/miss bits, counts | **0 differences.** Any single difference is FAIL. | These are the quantities the spec's metric definitions are stated over (`r` is an integer 1-based rank). A rank is not a float to be approximated: `hit_at_10` reads `r <= 10`, so one rank unit at the boundary is one metric unit. |
| B. Floating-point features | encoded feature rows, score matrices | **bit equality expected** for decode-only changes (worker count, batch size, thread pinning). Report `max abs delta`, `max rel delta`, and `#bytes differing`. Any nonzero byte is a FAIL for a decode-only claim and must be explained. | The throughput review claims decode parallelism is bit-identical. "Bit-identical" is checkable exactly; a tolerance would only hide a violation. |
| C. Floating-point scores under a changed summation order | score matrices from `optimize=True`, chunked GEMM, different BLAS thread counts | A numeric tolerance is NOT a pass criterion. Report the distribution (`max abs delta`, `max rel delta`, ULP scale). Scores are only an intermediate; the verdict comes from class A on the ranks derived from them. | Loosening a float tolerance cannot make a rank change acceptable, and tightening it cannot make an exact tie immune. Only class A decides. |
| D. Aggregate metrics | R@10, R@50, nDCG, composition gap | Report **exact** values and deltas to full float precision. For a claim of "unchanged", require bit-equal aggregate values given identical ranks. For a scoring-function change, any nonzero delta is reported as-is, never rounded to "no change". | Recall is percent, nDCG a fraction; the spec forbids mixing units, so deltas are reported in the metric's own unit. |
| E. Timing | q/s, wall clock | **Not a pass criterion.** Reported as measured. | This audit asks whether numbers changed, not whether they got faster. |

## Specific gates

- **G2 (decode/preprocess):** exact bit equality of the feature matrix across
  {1 worker, default threads} × {8 workers, pinned threads} × workers {2,4,8} × batch
  {1,32,64,256}, per transform family, using each model's own interpreter and transform.
  Verdict PASS requires `#bytes differing == 0` **and** `max abs delta == 0.0`.
- **G3 (scoring batching/chunking, same arithmetic order):** exact equality of ranks,
  K-hit bits, R@10/R@50/nDCG, Top-K id sets, Top-K order, denominators. Splitting the
  query axis must not change any of these; if it does, that is a real defect because the
  per-query sum order is unchanged.
- **G3b (changed summation order: `optimize=True`, chunked GEMM):** ranks are compared
  exactly; **any** rank change is reported with qid, both ranks and the score margin.
  Whether a rank change is *acceptable* is not a tolerance: it is the declared verdict in
  `EQUIVALENCE.md`, and it is FAIL for any gate phrased as "identical".
- **G4 (axes):** batch 1 vs 32, an incomplete final batch, cold vs warm cache, repeated
  run. Exact equality wherever the code path is deterministic; the FIRST divergence is
  reported with its stage.
- **G4b / G3c:** descriptive, no tolerance. Defects are named, not scored.

## Pre-declared expectations that a result may falsify

1. Decode-only changes are bit-identical for **every** transform family. HABIT's 0.0 does
   not transfer to `open_clip`; both are measured separately.
2. Changing the summation order (`optimize=True`, GEMM chunking, BLAS thread count)
   **will** change some ranks, because `_ranks_from_scores` counts exact equality and the
   data contains exact ties. Predicted rate from the prior review: ~14/1200 for the fast
   paths.
3. Predicted changed ranks are deep (`~2k–64k`), so R@10/R@50 are expected to be
   **unchanged** and nDCG to move by O(1e-5). These are predictions, not tolerances: the
   measured numbers replace them, and a K-boundary crossing is a FAIL of the "no metric
   movement" claim, not a tolerance question.
4. The gallery contains exact duplicate score patterns (FashionIQ has duplicated images),
   so systematic ties exist and some ranks depend on tie order. Quantified in G3c.

## What is never acceptable

- Reporting a PASS implied by a smaller sample when the required sample was larger.
- Reporting `NOT_RUN` as anything other than `NOT_RUN`.
- Rounding a nonzero rank-change count to zero, or describing a per-query rank change as
  "no change" because the aggregate moved by less than a printing precision.

---

## Post-run attestation (appended after all measurements)

No tolerance was changed, loosened, or reinterpreted after a result. Specifically:

- The decode gates were declared as **exact bit equality** and are reported as PASS on that basis for
  both transform families, 16 configurations each.
- The scoring gates were declared as "any rank change is FAIL for a gate phrased as identical", and
  the 207/18,009 rank changes are reported as **FAIL on rank identity**, not softened into a float
  tolerance after the fact.
- The K-boundary crossings were declared as the mechanism that decides metric movement, and **0
  crossings** was reported as measured, with R@10/R@50 shown bit-equal.
- The batch axis was declared "exact equality wherever the code path is deterministic". The GEMM path
  turned out **not** to be deterministic in the batch size (6/6003). That is reported as a FAIL of the
  axis, and the tolerance was not widened to accommodate it.
- Where the sample was smaller than the full set (shipped-path batch axes: 96 queries; `opt` batch-1:
  600 queries; ENCODER rank impact: proxy galleries), the sample size is stated in the result itself
  and in `EQUIVALENCE.md`, never implied to be the full set.
- `NOT_RUN` items are listed in `EQUIVALENCE.md` with reasons and are nowhere described as passes.
