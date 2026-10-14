Independent adversarial review of the CIR Failure Analysis Workbench (branch
`research/cir-failure-workbench`). Read-only: no file under `/home/heheboiz/data/cir` was modified;
scratch went to /tmp. No GPU run, no install, no download.

## SCOPE

The shared evaluation, provenance and orchestration code, plus the behaviour the test suite claims to
pin:

- `src/evaluation/fashioniq.py`, `src/evaluation/fashioniq_encoder.py`, `src/datasets/fashioniq.py`
  — the three declared protocols, gallery construction, reference policy, Recall@K, aggregation.
- `workbench/backend/runtime.py` — `source_status`, `_source_dirty`, `_approved_source_artifacts`,
  `_verify_declared_artifact`: can a real source divergence hide, and can a checkout borrow another
  model's allowlist?
- `workbench/backend/registry.py` — `caption_alias_evidence`, `validate_auxiliary_asset`: can
  arbitrary bytes be accepted as a variant?
- `workbench/scripts/pipeline.py` — resume, fingerprints, stage receipts, failure propagation and
  exit codes: can a failed stage be recorded as complete?
- `workbench/backend/paper_gate.py` and `workbench/backend/metrics_extraction.py` — the acceptance
  rule: can a run pass on a partial or mismatched reference?

## METHOD

Static reading plus hand-computed counterexamples and throwaway scratch repositories. For the
protocol work, expected Recall@K values were computed by hand from small rankings and compared with
the implementation rather than the other way round. For provenance, two attacks were constructed in
scratch git repositories and run. For the pipeline, receipts and fingerprints were exercised against
a throwaway artifacts root. Every finding below was reproduced by hand before it was reported; one
claim the implementer had made was refuted and is recorded as withdrawn.

## FINDINGS

This document is the cross-cutting summary. The per-model evidence, the specific P0/P1 findings and
their resolution state are in `OPEN_BLOCKERS.md` and in the 16 files under
`workbench/docs/reviews/models/`. The items a cross-cutting reviewer raised and that this engagement
acted on:

| Finding | Severity | Resolution |
|---|---|---|
| A modified committed `.pyc` changed executed code while the `.py` stayed pinned; a `.py`→`.pyc` rename hid a module's removal | P0 | fixed in `runtime.py`; covered by `test_src11`-`test_src13`; the live AirKnow checkout was already diverging (602 deleted tracked `.pyc`) and reported clean |
| A stage receipt from an earlier invocation could certify a later one | P1 | fixed in `pipeline.py`; receipts now record and require the input fingerprint |
| A partial paper row was an automatic pass | P1 | fixed in both `_check_parity` and `paper_gate`; SPRC's row (no R@10) now yields `METRIC_REFERENCE_UNVERIFIED` |
| `replay/common.py` refused a checkout that had merely been imported | P2 | fixed; runtime and replay share the bytecode rule |
| The claim "the iLearn evaluator raises before ranking on a shape mismatch" | — | **refuted by this review and withdrawn**; `np.matmul` broadcasts and `squeeze().max(-1)` yields the expected `(queries, gallery)` |

## CONTRACT THE REVIEW LEAVES IN PLACE

- A `VALID` verdict requires a genuine run; no code path in this repository can produce one today,
  which is why all 16 models are `BLOCKED` or `VARIANT` and none is `VERIFIED`.
- Provenance is checked against git and against declared digests, never against a filename.
- The 5 percent rule is relative, per metric, and cannot be satisfied by a partial reference.

## VERDICT

The shared code no longer contains a known path by which a result could be attributed to a pinned
tree that does not match what executed, or accepted against a reference it does not actually meet.
The blockers that remain are upstream and resource-bound, not workbench defects.
