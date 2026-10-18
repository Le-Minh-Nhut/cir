# R8 — SIXTEEN-MODEL STATUS: ACTUAL IMPLEMENTATION STATE

**Companions**: `FINAL_16_MODEL_REPRODUCIBILITY_REASSESSMENT.md` (the evidence matrix with per-model
citations), `FINAL_16_MODEL_RECOVERY_PRIORITY.md` (the ranked plan), and 16 per-model reports under
`model_reassessment/`. This document states only what is **implemented in this tree**; anything
planned but not implemented is labelled as a plan.

## 1. What the columns mean

`native_evaluator` is read from the adapter that exists for the model in
`workbench/backend/adapters/models.py`, not from a registry field: the registry has no
`command_status` key at all, and the registry's `command_status`-like labels used elsewhere in this
document's companions are audit prose, not a lookup. It says which model has an adapter that
constructs the upstream command, not that the workbench can reproduce the model.

`parser` says whether a metric parser branch exists for the model in
`workbench/backend/metrics_extraction.py`. **Five models have one**: `limn`, `csmcir`, `sprc`
(shares the CSMCIR shape) and the two `CLIP4Cir` variants. The remaining eleven return
`AGGREGATE_PARSER_UNAVAILABLE`. `observer` describes code that exists in this tree.

**Runtime verification level is 0 for every model, including LIMN.** No run of any model has been
executed on the target GPU in this pass, so no model is GPU-verified and no model's Recall has been
reproduced here. The static level counts what was established by reading the pinned source and the
artifact metadata; it is not evidence of execution.

## 2. Matrix

| model_id | protocol | src_avail | src_pinned | ckpt_variants | mapping_verified | adapter_command | parser | observer | static_lvl | runtime_lvl | GPU | remaining_blockers | readiness | next step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `csmcir` | `fashioniq_original_split` | yes | yes | 1 | yes | yes | yes | NONE | 2 | 0 | **no** | see `model_reassessment/csmcir.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `airknow` | `fashioniq_full_gallery_ref_excluded` | yes | yes | 2 | yes | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/airknow.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `conesep` | `fashioniq_full_gallery_ref_excluded` | yes | yes | 3 | yes | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/conesep.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `ptha_mtst` | `UNRESOLVED` | **no** | **no** | 0 | no | **none** | no | NONE | 2 | 0 | **no** | see `model_reassessment/ptha_mtst.md` | **NEEDS_AUTHOR_INPUT** | Author source exists but is not integrated; no FashionIQ weight and no verified protocol. |
| `habit` | `fashioniq_full_gallery_ref_excluded` | yes | yes | 3 | yes | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/habit.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `intent` | `fashioniq_full_gallery_ref_excluded` | yes | yes | 3 | yes | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/intent.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `hint` | `fashioniq_val_split` | yes | yes | 1 | yes | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/hint.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `encoder` | `fashioniq_val_split` | yes | yes | 1 | no | yes | no | NONE | 2 | 0 | **no** | see `model_reassessment/encoder.md` | **READY_AFTER_SMALL_FIX** | Apply the approved import patch, register the LAION ViT-B/32 asset, add the metric parser. **An adapter and a `command()` override already exist** (`adapters/models.py:36-41`); what is missing is the alias patch, the asset and the parser. |
| `pair` | `fashioniq_val_split` | yes | yes | 2 | no | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/pair.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `clvc_net` | `fashioniq_val_split` | yes | yes | 1 | no | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/clvc_net.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `dcnet` | `fashioniq_full_gallery_ref_excluded` | yes | yes | 1 | no | yes | no | NONE | 2 | 0 | **no** | see `model_reassessment/dcnet.md` | **NEEDS_UPSTREAM_ASSET** | Author preprocessing cache unavailable; original parity not established. |
| `combiner_rn50x4_noft` | `fashioniq_original_split` | yes | yes | 1 | no | yes | yes | NONE | 2 | 0 | **no** | see `model_reassessment/combiner_rn50x4_noft.md` | **NEEDS_CHECKPOINT_MAPPING** | Fetch the published `weights` release, map members, then load. |
| `clip4cir_rn50x4_fullft` | `fashioniq_original_split` | yes | yes | 1 | no | yes | yes | NONE | 2 | 0 | **no** | see `model_reassessment/clip4cir_rn50x4_fullft.md` | **NEEDS_CHECKPOINT_MAPPING** | Fetch the published `weights` release, map members, then load. |
| `tgcir` | `fashioniq_val_split` | yes | yes | 1 | no | inherited | no | NONE | 2 | 0 | **no** | see `model_reassessment/tgcir.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `sprc` | `fashioniq_original_split` | yes | yes | 1 | no | inherited | yes | NONE | 2 | 0 | **no** | see `model_reassessment/sprc.md` | **NEEDS_WRAPPER** | Audit the native evaluator and write the wrapper. |
| `limn` | `fashioniq_val_split` | yes | yes | 3 | yes | yes | yes | LIMN_QUERY_OBSERVER_V1 | 2 | 0 | **no** | see `model_reassessment/limn.md` | **READY_FOR_PC_PILOT** | Run `./pc.sh` on the RTX 5070 Ti; the environment, bundle and observer are in tree. |

## 3. Capabilities that exist in this tree

| Capability | Models | Evidence |
|---|---|---|
| A per-query (Top-K) observer, published as a canonical artifact and verifiable end to end | **`limn` only** | `workbench/replay/limn_export.py`, `limn_publish.py`, `REPLAY_COMMAND_IMPLEMENTED` in the registry |
| Aggregate-only evaluation (no per-query artifact required by design) | the other **15** | the gate reads `per_query_observer` from the registry and requires an artifact exactly when it is declared |
| A metric parser branch for the native output | **5 of 16**: `limn`, `csmcir`, `sprc`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` | `workbench/backend/metrics_extraction.py:402-411`; the other eleven return `AGGREGATE_PARSER_UNAVAILABLE` |
| An adapter that constructs the upstream command | **6 of 16**: `limn`, `csmcir`, `dcnet`, `encoder`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` | `workbench/backend/adapters/models.py` — these are the classes that override `command()` |

**### What the adapter column does and does not say

`adapter_command` distinguishes **yes** (the class overrides `command()` and builds an argument list
for the upstream script) from **inherited** (the class names a script but takes the base class's
command construction) and **none** (no adapter at all -- `ptha_mtst`). The distinguishing fact for
`encoder` is that its adapter **does** override `command()` (`adapters/models.py:36-41`), so the claim
that encoder has no command construction is wrong; what it lacks is the compatibility patch, the
backbone asset and the metric parser.** Its recovery is `RECOVERY_PLAN_IDENTIFIED` and no further: the
compatibility alias is identified but not applied, the OpenCLIP backbone asset is not registered, no
metric parser for its native output exists, and no GPU run has been attempted. It remains
**aggregate-only**; no observer for it is implemented or planned in code.

## 4. Readiness vocabulary used above

`READY_FOR_PC_PILOT` · `READY_AFTER_SMALL_FIX` · `READY_AFTER_ASSET_SETUP` · `NEEDS_WRAPPER` ·
`NEEDS_CHECKPOINT_MAPPING` · `NEEDS_UPSTREAM_ASSET` · `NEEDS_AUTHOR_INPUT` ·
`UNRESOLVED_SCIENTIFIC_PROTOCOL` · `GPU_VALIDATION_PENDING`.

Every row above that is not `READY_FOR_PC_PILOT` is also `GPU_VALIDATION_PENDING`: the two axes are
independent, and a model can be blocked on an asset and unverified on hardware at the same time.

## 5. Protocol invariants preserved in this pass

Protocol A `fashioniq_original_split` (full gallery, reference eligible) — `csmcir`, `sprc`,
`combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`.
Protocol B `fashioniq_full_gallery_ref_excluded` — `airknow`, `conesep`, `habit`, `intent`, `dcnet`.
Protocol C `fashioniq_val_split` (ordered pair-union, reference excluded) — `hint`, `encoder`, `pair`,
`clvc_net`, `tgcir`, `limn`.
`ptha_mtst` — **UNRESOLVED**, unchanged.

No protocol, reference-eligibility rule, gallery construction or Recall calculation was modified in
this pass. No checkpoint variant was substituted. No upstream revision was moved. `pc.sh` and the
evaluation path were changed only for argument carriage, the certification policy and the published
bundle binding; none of those read or alter a model's scientific contract.
