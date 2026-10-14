# Backend/runtime review

**Overall verdict: UNSOUND** — parser can accept ambiguous metric output and can raise on malformed averages; resize receipt validation also fails to enforce resampling during image reuse.

## Findings

### P1 — Duplicate metric lines are silently accepted
`workbench/backend/metrics_extraction.py:326-333` assigns each match into dictionaries, so repeated category or average keys overwrite earlier values. The parser then validates only the surviving values at `:335-352`. Read-only probe: conflicting duplicate category and average lines both returned `EXTRACTED_VERIFIED`. Ambiguous output can therefore supply fabricated values while passing macro checks. Reject duplicate keys before extraction.

### P1 — Malformed numeric average escapes parser error handling
`workbench/backend/metrics_extraction.py:332-333` converts averages with `float()` before entering the `try` at `:334`. A matched malformed token such as `average recall10 = 1e` raises `ValueError` out of `parse_combiner_output` instead of returning the unparsed result. `workbench/scripts/evaluate_models.py:1375-1398` calls extraction directly before report construction, with no local exception handling. The read-only probe reproduced the exception.

### P2 — Resampling mismatch does not force regenerated images
`workbench/scripts/prepare_model.py:192-193` correctly rejects receipts whose `resampling` differs from `RESAMPLING`; `write_resize_receipt` writes that value at `:141-145`. But `_run_resize` still loads the prior receipt at `:517`, and `_derived_is_bound_to_source` at `:397-438` validates source/output hashes, size, and extension without checking receipt `resampling`. Its reuse path at `:576-585` can therefore reuse images produced with a different kernel, then write a fresh receipt claiming current `RESAMPLING`. The new check detects an invalid receipt for readiness but does not enforce the promised regeneration. Include resampling in reuse eligibility, or otherwise refuse reuse when prior receipt lacks/mismatches it.

**Legacy receipt note:** No resize receipt exists under the configured FashionIQ root, so there is no live receipt to confirm. The previous writer did not record `resampling`; such receipts fail the new completeness check (`prepare_model.py:192-193`). This is not by itself a harmful invalidation, but their subsequent reuse still needs the guard above.

## Checked changes

- **Fullft adapter path is consistent.** `workbench/backend/adapters/models.py:97-111` now passes `<checkpoint bundle directory>/combiner_state.pt` and `clip_state.pt`. `OfficialScriptAdapter` resolves the same checkpoint root at `:16-21`; `workbench/backend/registry.py:269-291` treats `bundle_placeholder` as that directory and checks its `required_files`. Pinned validator loads the supplied paths (`workbench/third_party/CLIP4Cir/src/validate.py:283-287,304-308`).
- **Paper identity override and fallback are correct.** `workbench/scripts/build_paper_registry.py:88-103` takes non-null experiment `paper_*` values and falls back to paper-block values; empty/missing title also falls back at `:101-102`. Rebuilding in memory produced the same 33 reference IDs as committed `papers.yaml`; checked `paper_id`, `paper_title`, model IDs, statuses, and metrics with no mismatches. ENCODER/PAIR identities are correctly committed at `workbench/registry/paper_benchmarks/papers.yaml:1102-1105,1165-1168`.
- **Parser range and missing-average handling otherwise fail closed.** `_is_valid_recall` rejects booleans, non-finite values and values outside 0–100 (`metrics_extraction.py:64-71`); missing categories/averages reach `KeyError` handling at `:335-370`. Negative, >100, NaN, and missing-average probes returned unparsed. Normalized averages and macro consistency checks are at `:339-357`.
- **SPRC routing matches its emitted schema.** Dispatch reuses CSMCIR parser at `metrics_extraction.py:397-400`; SPRC emits the same category and average JSON keys at `workbench/third_party/SPRC/src/blip_validate.py:88-98`.
- **Registry YAMLs load successfully.** Read-only `load_registry`, `load_auxiliary_assets`, and benchmark registry probes succeeded: 16 models, 14 auxiliary assets, 33 paper references. Preparation contracts load through `load_registry`; PAIR’s `preparation_id` resolves to `pair_fashioniq` (`models.yaml:484-485`, `preparation_contracts.yaml:302-338`). The ResNet cache suffix is `hub/checkpoints/resnet50-19c8e357.pth` for both relevant contracts (`preparation_contracts.yaml:52,131`), matching the Torch Hub cache layout used by the asset checker (`backend/registry.py:213-216`).

**Verification:** read-only import/registry probes and targeted parser probes only; no tests or files modified.
