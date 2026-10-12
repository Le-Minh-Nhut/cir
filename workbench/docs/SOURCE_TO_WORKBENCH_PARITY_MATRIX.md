# Source-to-workbench parity matrix

One row per registered model, built from the four source-audit documents. Group A
(`PHASE2_SOURCES_A.md`) and Group B (`PHASE2_SOURCES_B.md`) carry a per-component matrix with the
full column set; Groups C and D recorded their comparisons per component in prose. This document
collects the model-level verdicts so the whole registry is visible in one place. It adds no claim
that the underlying audits did not establish.

**Status vocabulary.** EXACT_MATCH | EQUIVALENT_WITH_EVIDENCE | MISMATCH_CONFIRMED |
INCOMPLETE_IMPLEMENTATION | SOURCE_EVIDENCE_MISSING | EXTERNAL_ASSET_REQUIRED | NOT_APPLICABLE |
NOT_YET_VERIFIED.

**No model is EXACT_MATCH.** Every audit states this explicitly: workbench adapters mostly route
to the official evaluator rather than reimplementing it, and routed execution plus artifact parity
have not been reproduced. `EQUIVALENT_WITH_EVIDENCE` means the named component's behaviour agrees
with the source at that boundary, never that a score was reproduced.

| model_id | pin verified | protocol correct | preparation contract | caption stage | image stage | evaluator command | overall status |
|---|---|---|---|---|---|---|---|
| csmcir | yes | EQUIVALENT_WITH_EVIDENCE | MISMATCH_CONFIRMED (`.png` vs official `.jpg`; base BLIP-2 and COT provenance) | EQUIVALENT_WITH_EVIDENCE (native BLIP processor; not the local generic composer) | EQUIVALENT_WITH_EVIDENCE (targetpad owned by native evaluator; torchvision version unpinned) | IMPLEMENTED | MISMATCH_CONFIRMED |
| limn | yes | EQUIVALENT_WITH_EVIDENCE | EQUIVALENT_WITH_EVIDENCE (train-inclusive contract, isolated cache root) | EQUIVALENT_WITH_EVIDENCE (native tokenizer retained) | EQUIVALENT_WITH_EVIDENCE (model-derived transform, versions checked) | IMPLEMENTED (native replay) | EQUIVALENT_WITH_EVIDENCE |
| encoder | yes | EQUIVALENT_WITH_EVIDENCE | EXTERNAL_ASSET_REQUIRED (base OpenCLIP binary; train captions/caches) | EQUIVALENT_WITH_EVIDENCE | EQUIVALENT_WITH_EVIDENCE | IMPLEMENTED | EXTERNAL_ASSET_REQUIRED |
| pair | yes | EQUIVALENT_WITH_EVIDENCE | EXTERNAL_ASSET_REQUIRED | NOT_YET_VERIFIED | NOT_YET_VERIFIED | NOT IMPLEMENTED (inherits the abstract command) | INCOMPLETE_IMPLEMENTATION |
| hint | yes | EQUIVALENT_WITH_EVIDENCE (pair union, reference masked before ranking) | INCOMPLETE_IMPLEMENTATION (val-only layout misses fresh-constructor train inputs) | EQUIVALENT_WITH_EVIDENCE | EQUIVALENT_WITH_EVIDENCE | NOT IMPLEMENTED | INCOMPLETE_IMPLEMENTATION |
| clvc_net | yes | EQUIVALENT_WITH_EVIDENCE (source-order endpoint union, mask before ranking) | INCOMPLETE_IMPLEMENTATION (per-category whole models vs `category_model_policy: shared`) | NOT_YET_VERIFIED | NOT_YET_VERIFIED | NOT IMPLEMENTED | INCOMPLETE_IMPLEMENTATION |
| tgcir | yes | EQUIVALENT_WITH_EVIDENCE | EXTERNAL_ASSET_REQUIRED (correction dictionaries at the loader's path; no archive identity) | NOT_YET_VERIFIED | NOT_YET_VERIFIED | NOT IMPLEMENTED | INCOMPLETE_IMPLEMENTATION |
| airknow | yes | EQUIVALENT_WITH_EVIDENCE (full split, reference masked) | EXTERNAL_ASSET_REQUIRED (LAVIS BLIP-2 weights, `/root/bert-base-uncased`, correction dictionaries, cache PKLs) | EQUIVALENT_WITH_EVIDENCE (pre-LAVIS string stage only) | EQUIVALENT_WITH_EVIDENCE | NOT IMPLEMENTED | INCOMPLETE_IMPLEMENTATION |
| conesep | yes | EQUIVALENT_WITH_EVIDENCE | EXTERNAL_ASSET_REQUIRED (same LAVIS/BERT/dictionary set) | EQUIVALENT_WITH_EVIDENCE | EQUIVALENT_WITH_EVIDENCE | NOT IMPLEMENTED | INCOMPLETE_IMPLEMENTATION |
| habit | yes | EQUIVALENT_WITH_EVIDENCE | EXTERNAL_ASSET_REQUIRED | EQUIVALENT_WITH_EVIDENCE | EQUIVALENT_WITH_EVIDENCE | NOT IMPLEMENTED | INCOMPLETE_IMPLEMENTATION |
| intent | yes | EQUIVALENT_WITH_EVIDENCE (default `original-split` branch, reference masked) | EXTERNAL_ASSET_REQUIRED | EQUIVALENT_WITH_EVIDENCE | EQUIVALENT_WITH_EVIDENCE | NOT IMPLEMENTED | INCOMPLETE_IMPLEMENTATION |
| dcnet | yes | EQUIVALENT_WITH_EVIDENCE (ordered `val_trg`, mask before dual-branch fusion) | EXTERNAL_ASSET_REQUIRED (three GloVe val PKLs, 256-resized JPEGs, ResNet-50) | NOT_YET_VERIFIED | NOT_YET_VERIFIED | IMPLEMENTED (`test.py --resume <dir>`) | EXTERNAL_ASSET_REQUIRED |
| combiner_rn50x4_noft | yes | EQUIVALENT_WITH_EVIDENCE (full split, reference eligible) | MISMATCH_CONFIRMED (validator hardcodes `<source>/fashionIQ_dataset`; canonical root alone is insufficient) | NOT_YET_VERIFIED | NOT_YET_VERIFIED | IMPLEMENTED | MISMATCH_CONFIRMED |
| clip4cir_rn50x4_fullft | yes | EQUIVALENT_WITH_EVIDENCE | MISMATCH_CONFIRMED (same hardcoded root) | NOT_YET_VERIFIED | NOT_YET_VERIFIED | IMPLEMENTED | MISMATCH_CONFIRMED |
| sprc | yes | EQUIVALENT_WITH_EVIDENCE (full split, reference eligible) | MISMATCH_CONFIRMED (source-fixed `fashionIQ_dataset` + PNGs; no SPRC link step) | EQUIVALENT_WITH_EVIDENCE (pre-LAVIS string stage) | EQUIVALENT_WITH_EVIDENCE | NOT IMPLEMENTED (default model name unregistered) | MISMATCH_CONFIRMED |
| ptha_mtst | no source | NOT_APPLICABLE | SOURCE_EVIDENCE_MISSING | NOT_APPLICABLE | NOT_APPLICABLE | NOT_APPLICABLE | SOURCE_EVIDENCE_MISSING |

## What this matrix is not

- It is not a reproduction table. No checkpoint was downloaded, no environment built, no
  evaluator executed, and no Recall computed for any row.
- It does not rank models by quality. `EQUIVALENT_WITH_EVIDENCE` on a component says the workbench
  behaves like the source at that boundary; it says nothing about the reported score.
- It does not replace the per-component matrices. Group A and Group B list every component with
  its own status, severity, downstream impact and required correction; Groups C and D carry the
  same information per model in prose. This table is an index into those.

## Registry-wide findings

| Finding | Consequence | Status |
|---|---|---|
| The third protocol had no implementation (`src/evaluation/fashioniq.py`) | Five registered models named a protocol the evaluator could not build | **Fixed**: one protocol table defines gallery and reference policy, and unknown names are refused |
| `fashioniq_val_split` was evaluated with the reference eligible | An excluding cohort scored as if the reference were a valid answer | **Fixed**: the reference rule is a property of the resolved protocol |
| Only internal protocol aliases were accepted | A caller holding a registry id could not build the two older families | **Fixed**: both spellings resolve |
| CSMCIR opens `{id}.png`; the official release ships `.jpg` | A `.jpg` layout failed only as a missing file on the GPU host | **Fixed**: the registry declares `fashioniq_image_extensions` and both preflight views check every validation id |
| Nine of fifteen adapters inherit the abstract `command()` | Those models cannot be launched at all, which matches their `BLOCKED` command status | Open, and expected: it reflects source reality (callable-only evaluators) |
| No iLearn method publishes a category macro | The workbench macro is a workbench aggregate, never a source metric | Recorded in the audits and in `FASHIONIQ_PROTOCOL_PARITY.md` |
| SPRC's native inference ignores its reference argument | Local SPRC-shaped scores are text-only, unlike the workbench retrieval path | Recorded as a model-path mismatch, not a protocol mismatch |
| `category_model_policy: shared` vs CLVC-Net's per-category whole models | One registered pair cannot cover three categories | Recorded as an open contract defect |
