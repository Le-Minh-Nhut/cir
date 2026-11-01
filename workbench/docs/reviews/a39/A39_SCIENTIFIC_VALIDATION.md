# A39 Scientific Validation

## Evidence and limits

All claims below retain the evidence classes used by the A39 source notes. No target-GPU evaluation is claimed.

| Validation | Result | Evidence class | Source |
|---|---|---|---|
| Independent metric recomputation | Recomputed from raw ranks for 8 queries and 3 conditions; all values match repository summary within `<1e-15`. Includes Recall@10/50, MRR, nDCG, common-cohort composition gap and MRR gap. | `REAL_ASSET_VERIFIED` | `.scratch-tmp/a39-sci/scientific-validation.md:97-128` |
| Composition condition tensors | Real query `7:1`: multimodal uses original image and prompt; text-only uses same prompt plus synthesized black `(444,342,3)` image (all zero pixels; normalized channel constants recorded in source); image-only uses original image and empty text. | `REAL_ASSET_VERIFIED` | `.scratch-tmp/a39-sci/scientific-validation.md:132-153` |
| Dataset tree / alignment | `data/FashionIQ` symlink resolves to populated `data/fashionIQ_dataset`; source reports 74,381 PNG images, and all 74,381 candidate-pool entries resolve via `.jpg` to `.png` stem fallback with no missing images. Counts/subsets: full 6,003; shortcut-free 4,069; validated 586 (368 solved, 218 unsolved). | `REAL_ASSET_VERIFIED` | `.scratch-tmp/a39-sci/scientific-validation.md:5-46` |
| Ranking edge fixtures | Verified exact 1-based ordering, gallery-order tie-break, minimum rank across multiple positives, NaN/Inf -> `None`, empty and out-of-bounds targets -> `None`; reference remains eligible. | `FIXTURE_VERIFIED` | `.scratch-tmp/a39-sci/scientific-validation.md:50-73` |
| Nine active adapters | 1 `REAL_ASSET_VERIFIED` (`encoder`); 1 `FIXTURE_VERIFIED` (`habit`); 7 `STATICALLY_VERIFIED` (`csmcir`, `conesep`, `ptha_mtst`, `intent`, `hint`, `limn`, `tme`). Other host readiness blockers are per-adapter in matrix. | Mixed classes as tallied | `.scratch-tmp/a39-sci/adapter-matrix.md:5-19` |
| CSMCIR checkpoint compatibility | Official 2.72 GB checkpoint absent; real state-dict key compatibility unverified. | `BLOCKED` | `.scratch-tmp/a39-sci/csmcir-investigation.md`; `.scratch-tmp/a39-core/open-questions.md:5-10` |
| RTX 5070 Ti | Host has RTX 3050 Laptop `sm_86`; no RTX 5070 Ti `sm_120` device run. | `BLOCKED` | `.scratch-tmp/a39-core/open-questions.md:12-16` |

Metric definitions: Recall@K is percentage of scored queries with rank ≤K; MRR and nDCG are fractions. Composition gap uses common scored cohort and ratio of cohort means, not mean of query-level ratios. Per-condition coverage uses that condition's own scored count; unscored queries are not assigned synthetic ranks. The A39 export fix now reports requested `query_count` separately from `scored_count`.

Real dataset and fixture validation does not establish model quality or cross-hardware equivalence. No GPU job was run in this reporting task.