# Final audit consolidation — 16 models

All 16 final audits are stored verbatim under `models/`. Every audit is read-only and static; no
GPU run, no download, no install. Verdict normalization: `FINAL VERDICT:` values as delivered.

## Verdicts

| Model | Final verdict |
|---|---|
| limn | BLOCKED |
| csmcir | VARIANT |
| airknow | BLOCKED |
| conesep | BLOCKED |
| habit | BLOCKED |
| intent | BLOCKED |
| hint | VARIANT |
| encoder | BLOCKED |
| pair | BLOCKED |
| clvc_net | BLOCKED |
| dcnet | BLOCKED |
| clip4cir_rn50x4_fullft | BLOCKED |
| combiner_rn50x4_noft | BLOCKED |
| tgcir | BLOCKED |
| sprc | BLOCKED |
| ptha_mtst | BLOCKED |

BLOCKED here means "cannot be reproduced end-to-end today" — the dominant causes are upstream
(unpublished or unmappable checkpoints, dead upstream entrypoints, missing modules, unobtainable
corpora), not workbench gate defects. VARIANT means the model's evidence chain carries deliberate,
labelled deviations (caption alias, provenance) that a reproduction must preserve.

## Disposition of every P0/P1 finding

### CONFIRMED_FIXED (workbench-side, verified)

1. **limn P0** — reference row asserted `checkpoint_training_noise_pct: 0` the paper never states.
   Bundle row changed to `null` with evidence; the gate grades LIMN runs again
   (`PASS_WITHIN_5_PERCENT` probe on paper-exact metrics, 3 comparisons).
2. **hint/encoder bundle rows + variants noise 0-to-null** — same defect class.
3. **hint/encoder/pair misfiled paper identity** — `build_record` now lets an experiment's own
   `paper_*` identity override the seed bundle's paper block; rows rebuilt (`encoder_aaai25`,
   `pair_icassp25`).
4. **Parser gaps (combiner C-3, sprc C7)** — `parse_combiner_output` (CLIP4Cir `validate.py`
   recall lines, shared by combiner_rn50x4_noft and clip4cir_rn50x4_fullft) added; sprc routed
   onto the csmcir JSON parser (identical key set, verified against `blip_validate.py:88-98`).
   Adversarial tests added: fabricated averages, incomplete output, malformed lines all rejected.
5. **clvc_net M1 / dcnet / clip4cir headline nulls** — sanity rows transcribed from verified
   benchmark rows with explicit derived-vs-reported provenance comments.
6. **clvc_net M2** — backbone string corrected: two-LSTM textual branch; the paper contains no
   "CLIP" occurrence.
7. **pair C3 labeling** — `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`,
   `strict_clean_checkpoint_available: false`, Hub URLs + LFS sizes recorded (identity still
   `CHECKPOINT_MAPPING_UNVERIFIED`).
8. **pair C6** — preparation contract `pair_fashioniq` added; the declared
   `manual_ilearn_resized` contract was inert (`preparation_id` missing); `prepare_model` now
   reports the missing resize instead of NONE_REQUIRED.
9. **dcnet F5 / pair M5 receipt claim** — `resize_is_complete` now validates the receipt's
   `resampling` field; the "kernel change invalidates receipts" sentence is enforced.
10. **clip4cir fullft P1** — variant noise 0 recorded (paper section 4.1 task-oriented
    fine-tuning, no noise injection); headline row transcribed from the three-way-verified
    Table 9 row.
11. **combiner M-1 / C-4** — noft variant noise 0 (noise-free by construction; committed rows
    assert 0); the Drive folder's documented filename recorded as provenance; bytes stay unmapped.
12. **combiner M-9 / clip4cir P1 bundle-path disagreement** — `CLIP4CirFullftAdapter` now reads
    bundle members under the `bundle_placeholder` path itself, agreeing with
    `registry.checkpoint_missing_paths`.
13. **ResNet-50 cache path (clvc C5, dcnet contract)** — preparation-contract
    `cache_path_suffix` corrected to `hub/checkpoints/...`, matching what the consumers actually
    read via `torch.hub`; the asset status now probes the true location.
14. **sprc C3 / M4** — registry note no longer asserts an unregistered model class;
    `lavis` corrected to `salesforce-lavis==1.0.2`.
15. **sprc C5** — BLIP-2 pretrain + EVA-ViT-g asset sizes recorded (HEAD-observed content
    lengths); decoy non-empty files at those paths are now size-rejectable.
16. **sprc C8** — SPRC2 attribution conflict recorded in the row's `metric_provenance`.
17. **tgcir P1-1 / ptha C4** — reference-row noise 0-to-null (papers state no noise protocol;
    same class as limn P0), each with re-derivation evidence.
18. **combiner M-6** — OpenAI RN50x4 asset size recorded; `provenance_status: VERIFIED` scoped
    to "URL + published digest".
19. **tgcir P2 clip pin** — `clip==0.2.0` documented as a known-wrong pin (PyPI resolves to a
    clipboard CLI; openai/CLIP publishes no tags).

### EXTERNAL_BLOCKER (upstream; not fixable in the workbench)

- **No runnable upstream path**: airknow CF-1/CF-3, conesep CF-1, habit C-2, intent, pair C1/C3
  (`MHTransformer_rev` absent from every one of 21 upstream commits; `test.py` has no
  `__main__`), clvc_net C1/C3, tgcir P1-3, dcnet (evaluator data root relative and unreachable),
  ptha C2 (`blip2_qformer_tgir_concatembedding_sim_zyn` absent from all 8 upstream commits),
  sprc C1/C2 (`import clip` unlisted; adapter emits no command).
- **Checkpoint identity unresolvable**: conesep CF-2, habit C-1 (0%-noise weights unpublished),
  clvc_net C2, dcnet P1 (folder-only URL), pair C2 (two blobs, three datasets, no binding), tgcir
  (no filename published), sprc C4 (URL 403), ptha C3 (FashionIQ_MTST corpus unobtainable),
  clip4cir fullft (bundle pairing genuinely open), combiner C-4 (no published checksum).
- **Missing support artifacts**: habit/intent/hint/pair correction dictionaries
  (`OFFICIAL_ASSET_UNAVAILABLE`), dcnet GloVe PKLs (unseeded generator), ptha corpus.

### DISPUTED / REJECTED (with rationale)

- **encoder C4** — originally disputed (the 3fc0c1a repair kept `VERIFIED_METADATA` so a complete
  local install was not refused as `mapping unresolved`). The holistic science review's argument
  stands: no author-linked direct URL, no size/hash, and no local artifact bind the Drive file to
  the paper's FashionIQ variant, and local availability cannot establish scientific identity. The
  mapping is now `UNVERIFIED` (and the registry coherence rule is one-directional: an unresolved
  URL no longer forces a verified mapping). Disposition changed to CONFIRMED_FIXED.
- **hint C5 category_unchecked bypass** — originally rejected as frozen policy; the holistic
  science review showed a plain `PASS_WITHIN_5_PERCENT` could stand while every paper-published
  category metric went unchecked. The gate now returns `PASS_WITH_UNCHECKED_CATEGORIES` — an
  explicitly incomplete pass — when published categories were never observed. Disposition changed
  to CONFIRMED_FIXED.
- **limn P3 dataset-root dual identity** — `run_meta["dataset"]` is a family name, not a path;
  symlink repointing is caught by dataset preflight.
- **ptha C1 ambiguous rows** — `select_reference` refuses when two verified rows match the same
  condition. This refusal is the correct fail-closed behavior: a run cannot yet declare which
  pretraining it reproduces because the checkpoint mapping is unverified. Marking one row
  non-selectable would be arbitrary. Kept as-is. (Holistic science review concurs: defensible
  refusal, not closure.)
- **tgcir P1-2 parser** — the replay is preflight-only and no command exists; a parser now would
  parse output no run can produce. Terminal state documented instead.

### DEFERRED (needs new evidence work)

- **combiner C-1** — add a `VERIFIED` noft reference row from the CVPR22 paper. The audit's
  numbers are secondhand through the audit document; registering them requires the same
  three-route verification every committed row carries.

## Cross-cutting observations

- Every model's reproduction status today is dominated by upstream blockers, not by workbench
  gate defects. The gate, selection, parsing and preparation layers now fail closed with named,
  accurate blockers.
- The recurrent workbench defect class was **unsupported assertions** — rows claiming a 0
  training-noise condition the paper never states (limn, hint, encoder, tgcir, ptha) and misfiled
  paper identities (encoder/pair under the HINT bundle). All were corrected at the bundle or
  build-script root cause, never by patching the gate.
- Several audit documents carry stale claims (dcnet BLOCKED_SOURCE, ptha "no source", sprc
  "shape audited", four docs citing a nonexistent `prepare_csmcir`). Documentation debt recorded
  for the holistic review round.

## What changed in this round (files)

- `workbench/scripts/build_paper_registry.py` — per-experiment `paper_*` override.
- `workbench/backend/metrics_extraction.py` — `parse_combiner_output`; sprc routing.
- `workbench/backend/adapters/models.py` — fullft bundle member paths.
- `workbench/scripts/prepare_model.py` — receipt `resampling` validation.
- `workbench/registry/models.yaml` — headline fills, noise fields, command/provenance labels,
  pair contract id, sprc corrections, tgcir package note.
- `workbench/registry/preparation_contracts.yaml` — `pair_fashioniq`; ResNet-50 suffix fix.
- `workbench/registry/paper_benchmarks/extracted/*.json` + rebuilt `papers.yaml` — noise nulls,
  backbone correction, attribution note; new `registry_version`.
- `workbench/registry/auxiliary_assets.yaml` — observed sizes for CLIP RN50x4, BLIP-2, EVA-ViT-g.
- `workbench/tests/test_adversarial_metrics.py` — combiner-parser adversarial tests.
- `workbench/tests/test_protocols.py` — pair contract exemption (no external assets).


## Holistic review round (3 reviewers)

Reports under `holistic/` (`scientific.md`, `backend-runtime.md`, `frontend.md`). All three
reviewers returned UNSOUND on delivery; every P1 finding is fixed in this round. Science reviewer
additionally confirmed: derived-metric arithmetic (9 formulas) checks; PTHA C1, LIMN aggregate
rejection, TG-CIR parser deferral and combiner noft deferral are defensible; no sampled benchmark
metric is untraceable.

1. **P1 duplicate metric lines accepted** — `parse_combiner_output` now rejects any repeated
   category or average line before extraction (the evaluator prints each recall exactly once).
2. **P1 malformed average escaped error handling** — average parsing moved inside the guarded
   conversion; `average recall50 = 1e` now returns `METRICS_UNPARSED` instead of raising.
3. **P2 resampling mismatch did not force regeneration** — `_derived_is_bound_to_source` now
   requires the receipt's `resampling` to equal the current kernel; files produced with a
   different (or unrecorded) kernel are regenerated, not reused under a fresh receipt.

Frontend reviewer findings:

- Synthetic evaluator output could reach the warning path unlabelled → `is_synthetic` is now
  projected by `/api/runs/gate` and `/api/benchmark-runs`, and aggregate-only runs render a
  `SYNTHETIC SOURCE` badge. No third warning label was added (frozen vocabulary).
- Slider is a UI warning threshold, not a backend gate control — **by design**: the gate verdict
  is recorded at the canonical 5% tolerance and the frontend re-flags the recorded comparisons
  without recomputing a percentage. Rejected as a defect; it is the documented contract.
- Mock runs visible but explicitly badged — development data path, not a substitution. Accepted.

Additional science-review fixes in this round:

4. **Gate could pass with every published category unchecked** — new gate status
   `PASS_WITH_UNCHECKED_CATEGORIES`: an explicitly incomplete pass when the paper publishes
   category metrics the run never observed. This supersedes the earlier hint C5 rejection.
5. **ENCODER checkpoint mapping overclaimed** — mapping flipped to `UNVERIFIED`; the registry
   coherence rule is now one-directional (an unresolved URL no longer forces a verified mapping).
   This supersedes the earlier encoder C4 dispute.
6. **Evidence honesty** — CSMCIR category/average internal disagreement (R@10 0.027 pp, R@50
   0.087 pp) recorded in `csmcir_acl26.json`; AirKnow AVG 65.73 rounding sentence corrected
   (half-up gives 65.74; the printed cell is transcribed, not arithmetically justified);
   `checkpoint_training_noise_pct: 0` on csmcir/clip4cir rows annotated as WORKBENCH_INFERENCE.
   Null-paper-noise handling (caveat recorded, not a hard blocker) kept as-is with rationale: a
   hard blocker would disable every row whose paper states no noise protocol.

Full suite after the fixes above (collected before the two new parser tests): **897 passed,
4 warnings**. Full-suite collection after the additions is 899 tests; the two new parser tests
pass in their file (25 passed in `test_adversarial_metrics.py`).
