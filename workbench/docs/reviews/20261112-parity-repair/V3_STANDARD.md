# `parity-absolute-relative/v3` evidence and publication contract

Status: **implemented contract; no real-model V3 capture published by this change**. V1/V2 rules and historical files stay unchanged.

## Numerical rule

For each emitted numeric component, fail when

```text
abs(actual - reference) > atol + rtol * abs(reference)
```

`native` is the reference for native/adapter comparisons. V3 requires complete per-component evidence or an explicitly incomplete preview. A measured violation in a preview fails; a clean preview is incomplete. Aggregate `max_abs`/`max_rel` remain diagnostics, not replacement numeric vetoes. Identity, scoring, rank/tie, metrics, and other mandatory checks remain independent.

## Preregistered plan and run receipts

Before any named run, persist immutable plan sidecar with `write_v3_plan(path, plan)`. Canonical identity has exactly `checkpoint`, `configuration`, `environment`, `protocol`, `category`, `cohort_id`, `precision`, `seed`, `query_order_sha256`, `gallery_order_sha256`, `reference_exclusions_sha256`, `query_count`, and `gallery_count`; `model_id` remains artifact routing identity. Both final artifacts carry same top-level identity; every preregistered role carries actual `source_id` and role identity, repeated identically in embedded receipt. Receipts bind run ID, plan SHA-256, native/adapter side, actual start/capture timestamps, captured array dtype/shape/content/order digests, partition ID, batch sizes, and partition digests.

Full native/adapter scope must match both actual artifact identities and their native layer1 and metric denominators, including query/gallery counts and order. Diagnostic repeats and B1/BN controls may use declared subsets, but role/receipt identities, cohort IDs, tensor shapes, ordered IDs, and global denominators must agree. Identity mismatch makes evidence malformed. Changing preregistered scope creates a new plan and run ID; never edits plan after execution.

The plan preregisters five controls: native repeat, adapter repeat, matched native/adapter, native B1/BN, adapter B1/BN. Each control binds actual role IDs, sides, dimension, dtype, tolerances and separate query/gallery scopes. Matched comparisons share IDs/order; independent diagnostic samples need not equal full ranking scope. Batch sensitivity is diagnostic by default; when the model/method requires batch invariance, preregister `supported_batch_policy.batch_invariant: true` to make mismatch a mandatory veto.

Each final artifact embeds `parity_v3.plan_path`, `plan_sha256`, component scopes, and actual receipts. Plan names are deterministic siblings: `g1_g2.{model}.{category}.v3-plan.json`.

## Fixed publication and API

The trusted publication root is `workbench/artifacts/parity/v3/`. `CURRENT.json` contains only a generation ID; the gate validates containment beneath `generations/`. Publish with `publish_v3_generation(...)`: writers serialize with a file lock, stage an immutable generation, validate complete native/adapter artifacts and plans for all three categories per new model, then atomically replace `CURRENT.json`. Existing generations and V1/V2 records are carried unchanged; hardlinks avoid repeated bulk evidence copies and staging links are unlinked before replacement. No request supplies a filesystem path. The API discovers the active generation alongside the historical root; explicit old-root evaluation remains reproducible.

`GET /api/runs/parity-gate?standard=...` keeps the requested selector advisory to the artifact-applied standard. Unsupported selectors return HTTP 400. Results expose applied/requested standard, categories, per-check `check_type` (semantic dimension), category, and nested declared/measured scope metadata. Measured fields include coverage scope, population and sample counts, tensor shape, dtype, ordered-ID digest, control, and batch policy; consumers retain each check independently.

## Implementation and verification

Writer and publication helpers: `workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py:182-412`. Gate and trusted root: `workbench/backend/parity_gate.py`; route: `workbench/backend/main.py`. Focused regression file: `workbench/tests/test_parity_absolute_relative_v2.py`.

After identity/population binding corrections, `workbench/envs/ctl312h/bin/python -m pytest -n10 workbench/tests/test_parity_absolute_relative_v2.py -q` exited 0: **85 passed, 10 warnings in 4.02s**. Durable record: `workbench/docs/reviews/20261112-parity-repair/raw/v3-contract/pytest-n10-identity-bound.txt`. Earlier selected V3 run remains historical only; it used the invalid 2-vs-10 full-query fixture.

This verifies contract logic and synthetic fixtures only. No native/adapter model run or full-population measurement is asserted. Producer lanes must preregister, capture real execution receipts, and publish complete category bundles before making measured claims.

## V3 consumer decision — raw score/rank proof

Decision: **MODIFY** V3 acceptance to validate consumer-readable, source-pinned per-query raw scores, ordered IDs, reference/target labels, and captured orders/ranks. Derive score agreement, rank differences, tie qualifications under preregistered `score_max_abs`/`tie_epsilon`, sorted-output identity, top-K membership/order, per-query hit flips, integer hit counts, and their aggregate equality from these captures; treat stored summaries only as cross-checks. Keep existing preregistered score/tie/integer-metric rules as the only acceptance policy: do not require zero exact Top-K/order differences or zero per-query hit flips. Equal aggregate counts with offsetting per-query flips remain aggregate-equal, but must be reported as non-identical query outcomes and cannot establish failure-overlap equivalence. Report tie-qualified order changes separately from identical sorted output. Missing or corrupt raw proof is incomplete/malformed, never a scalar-based pass. External raw-proof bytes enter the gate cache dependency stamp. Targeted consumer regressions cover forged summaries with non-tie changes, exact/near ties, offsetting hit flips, missing/swapped/corrupt proof, and unchanged V1/V2 behavior.

Scope caveat: HINT producer→gate path is repaired and verified by a CPU toy fixture; fresh physical HINT receipts are still required for model acceptance. Fresh LIMN scorer evidence is also still required; retained 2026-11-20 physical run records shirt R@1 failure (native 516, Adapter 517), unchanged and not acceptance.

## Per-side sorting and model-specific eligibility

Plans may optionally seal `score_order_conventions` per side. Missing values retain legacy `fp32_descending_score`; supported declarations are `fp32_descending_score` and `fp32_one_minus_score_ascending`. The latter validates exact FP32 `float32(1 - float32(score))` keys and ascending order, matching pinned sources that sort `1-score`. No epsilon relaxes validity of captured ordering. Unsupported convention is malformed. Producers must declare only the observed source/production sorter.

`metric_cutoffs` seals model-specific integer retrieval cutoffs and must match artifact metric details. Preregistered `scoring_labels` bind targets and reference exclusions: target must remain eligible, declared exclusions are omitted before rank, Top-K, and hit derivation. For models whose native protocol keeps reference images eligible, exclusions are empty. These plan fields do not alter legacy plans, V1/V2 behavior, or ranking policy.

## Focused verification — 2026-11-21

Gate, LIMN producer, and HINT producer compile passed. Five focused nodes passed: `test_v3_scoring_proof_order_must_follow_scores`, `test_fp32_one_minus_score_order_allows_rounded_key_tie`, `test_v3_raw_rows_derive_tied_order_and_hit_diagnostics`, `test_v3_rejects_non_target_inversion_outside_tie_epsilon`, and `test_limn_producer_assembly_passes_real_gate_and_missing_receipt_is_malformed`. Exact output: `workbench/docs/reviews/20261117-fix-all/raw/scientific-parity-final-verification-20261121-confirm.log`. HINT toy producer→gate fixture passed `PARITY_PASS`, exit 0; receipt: `workbench/docs/reviews/20261112-parity-repair/raw/third_model/20261121-hint-dress-aggregate-gate-cpu-proof.log`. Contract and toy-fixture evidence are not model acceptance.

## Work checkpoint — 2026-11-17

Paused by user direction. Latest edits are unverified: `workbench/backend/parity_gate.py` serializer now adds `harness_recorded_within_tolerance`; `evaluate_all_uncached` advisory list changed to V2-only. Last actual verification predates these edits: focused parity file reported 94 passed, 10 warnings; retained log `/tmp/cir-parity-file-full.txt`. Do not treat that result as verification of current tree. Re-read serializer before resuming: current edit may duplicate `reports_absent_evidence` in `ParityCheck.to_dict()`. No tests run after current edits. Owned files touched this turn: `workbench/backend/parity_gate.py`, this V3 contract note. No locks or queued jobs owned.

