# Open blockers

Every entry below is evidence-backed and unresolved. Nothing here is a wish list: each line is either
a defect the code cannot fix by itself, or a prerequisite that does not exist on this host. The 16
per-model audits are in `workbench/docs/reviews/models/`.

## Deciding constraint: no genuine evaluation can run here

| Prerequisite | State | Evidence |
|---|---|---|
| GPU inference | **not possible** | `nvidia-smi` reports an RTX 3050 Laptop with 4 GiB (compute 8.6), but every environment on this host has a CPU-only torch build: the workbench venv is `torch 2.14.0+cpu`. `torch.cuda.is_available()` is `False`. |
| Model checkpoints | **none staged** | `workbench/artifacts/checkpoints/` holds only `.gitkeep`. |
| Model environments | **none provisioned** | `workbench/artifacts/environments/` does not exist; the single assignment record (`environment_assignment.json`) names one conda path that was never created. |
| BLIP-2 / EVA-ViT-g backbone caches | **corrupt** | `~/.cache/lavis/blip2_pretrained.pth` is **22 bytes** containing the text `qwen`; `~/.cache/torch/hub/checkpoints/eva_vit_g.pth` is **4 bytes** containing `corrupt nonempty bytes`. Both are placeholders a loader would reject or load as garbage. `convnext_base` and `resnet18` in the same directory are real sizes, so the corruption is specific to the files these models need. |
| Disk headroom | **15 GiB free of 173 GiB** | `df -h /`, after the `uv` package cache was cleared. The CSMCIR checkpoint alone is 2.7 GB and the LIMN bundle is 5.4 GB, so staging one model's checkpoints is now possible and staging all of them is not. |

Consequence: **every model is `NOT_RUN` and no metric in this repository is a reproduced metric.**
The numbers in the registry are paper values, entered as paper values, and the gate treats them as the
reference to be reproduced, never as the result.

The target machine is an RTX 5070 Ti (16 GB, sm_120). Blackwell needs a torch build whose arch list
includes `sm_120`; a CUDA 12.x wheel without it raises at the first kernel launch rather than at
import. `python workbench/scripts/pc_readiness.py` reports the device, the torch build and its arch
list, and labels every otherwise-ready model `PC_GPU_VALIDATION_PENDING` until a real run exists.

## P0 — would invalidate a scientific result

### B0b. The approved-patch overlay did not work on its own documented workflow (FIXED)

The registry declares one approved patch for CSMCIR and the runtime verifies it, but applying that
patch produces a *tracked* edit, and the provenance classifier sent every tracked edit straight to
`tracked` without consulting its declared-artifact allowlist. The exact file the registry had just
approved was therefore reported as an unauthorized source change, and the model landed below its
unpatched classification.

Confirmed on the documented recipe before the fix: `source_status` -> `tracked:
['src/lavis/models/__init__.py']`, `approved: []`, `source_clean_and_pinned` -> `(False, ...)`.

A tracked path now reaches the same allowlist every other declared artifact uses. A declared patch
whose derived digest matches is recorded as approved; a modified committed `.pyc`, and any tracked edit
with no declaration, are still reported as source changes.

### B0. A committed bytecode cache could change the evaluated code (FIXED)

Four of the pinned checkouts commit `__pycache__`: AirKnow (602 files), CSMCIR (498), INTENT (312),
SPRC (160). The provenance classifier skipped every `*.pyc`, so two divergences were invisible:

- **A modified committed `.pyc`.** CPython imports a cache whose header matches the source, so the
  executed code changes while every `.py` stays byte-identical to the pin. Observed on a scratch
  repo: tampered cache, `git status` reporting it, and `source_clean_and_pinned` still returning
  `(True, …)`.
- **`git mv real.py real.pyc`.** The rename is a tracked status line, but the path parser returns the
  new name, which the bytecode filter then dropped — so the removal of a module was invisible.

On the live host the same blind spot was already active: AirKnow's checkout reported **602 deleted
committed `.pyc`** files while `source_status` said `{tracked: 0}` and `_source_dirty` said `False`.

Fixed in `workbench/backend/runtime.py`: a *deleted* committed `.pyc` is still ignored, because
Python removes stale caches just by being run and that must not refuse a checkout, but a *modified*
one and a `.py`→`.pyc` rename are now tracked divergences like any other. Both attacks and the
benign deletion are covered by `test_src11`-`test_src13` in
`workbench/tests/test_release_blocker_source_integrity.py`.

Residual, not fixed: bytecode staged *outside* the checkout and selected through
`PYTHONPYCACHEPREFIX` is invisible to any `git status`-based check. The workbench does not set that
variable (`_EVALUATOR_ENV_ALLOWLIST`), so this is an operator-environment concern rather than a
workbench defect.

### B1. WITHDRAWN — the iLearn family's similarity computation is fine

An earlier revision of this document claimed `INTENT/test.py:141` raises on a tensor-shape mismatch.
**That claim was wrong.** Measured against the extractors' real output shapes — query
`(B, 1, 1, 256)` from `blip2_cir.py:769`, gallery `(B, 256, 32)` from `:792`:

```
np.matmul((N,1,1,256), (G,256,32))  ->  (N, G, 1, 32)
.squeeze().max(-1)                  ->  (N, G)          # exactly the (queries, gallery) the indexing expects
```

`np.matmul` broadcasts rather than contracting, and the following `squeeze().max(-1)` produces the
matrix the reference-masking and `argsort` code indexes. The registry entries that carried the
withdrawn claim have been corrected, and no workbench patch is warranted. This is recorded rather
than deleted because the withdrawal is itself the evidence that an independent reviewer checked a
claim the implementer had made.

### B2. The 0 %-noise paper row has no published checkpoint (AirKnow, ConeSep, HABIT, INTENT)

The registered `reported_r10/r50/mean` for all four are the paper's **0 % noise** rows, verified
against the papers by the audits. The author Hubs publish only noise-trained weights:

| Model | Published weights | Paper row | Source |
|---|---|---|---|
| Air-Know | 50 %, 80 % | 0 % | Hub API listing: exactly two `.pt` files |
| ConeSep | 20 %, 50 %, 80 % | 0 % | Hub listing: exactly three |
| HABIT | 20 %, 50 %, 80 % (+3 CIRR) | 0 % | Hub listing: exactly six, no `N0.0` |
| INTENT | 20 %, 50 %, 80 % (+1 CIRR) | 0 % | Hub listing: exactly four |

`strict_clean_checkpoint_available: false` is therefore correct for all four, and the workbench
refuses them under `strict_clean` (`select_checkpoints` observed returning `BLOCKED_CHECKPOINT`).
Any run would be a **variant** comparison, and must carry `checkpoint_training_noise_pct` next to
the delta. The gate can enforce that (`COMPARABILITY_FIELDS` includes the field) but **no production
caller passes `paper_meta`/`run_meta` yet**, so today nothing would trigger it.

### B3. CSMCIR's pinned package cannot import (98 unguarded imports)

`src/lavis/models/__init__.py` lines 44-151 contain 98 uncommented imports of modules absent from the
pinned public revision, and they sit *before* line 152, where the class the evaluator needs is
registered. Pristine checkout, pinned environment:

```
ModuleNotFoundError: No module named 'lavis.models.blip2_models.blip2_qformer_cir_align_prompt_caption_query'
```

A reproducible patch exists at `workbench/patches/csmcir_models_init_compat.patch`; applied to a
derived copy, the same environment reports `IMPORT_OK True` with
`blip2_cir_align_prompt_csmcir` registered, and the patched file's 332 statements are byte-identical
in sequence to the original's. It is **not applied** to the registered checkout: changing which
classes LAVIS registers is a modification of the official implementation and stays a deliberate
human decision.

### B4. The COT_ours2 gallery captions were never published

`extract_index_blip_caption_features` (`utils_csmcir.py:100`) reads
`COT_ours2/fashioniq/{cat}_cot_val.json`, which appears in no public tree (author GitHub at the pin,
every git ref, the pinned HF revision, the author's only HF repository). The published Qwen captions
are source-compatible — same basename, exact gallery-id set equality (3817/6346/5373), every caption
equal to the `Final_Caption` the extractor consumes — which is why an opt-in labelled alias exists,
but byte identity with what the authors evaluated with is **not** established. See
`workbench/docs/COT_OURS2_AUDIT.md`; the classification is `SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN`.

## P1 — would make a result wrong if it were reported

### B4b. A stage receipt from an earlier invocation could certify a later one (FIXED)

`stage_receipt_path` is a pure function of the stage key and the receipt recorded no input
fingerprint, so `validate_stage_outputs` accepted any receipt with the right `stage_id` and
`return_code: 0` — including one written by an unrelated earlier run. A stage could therefore be
finalized `COMPLETE` on the strength of an invocation whose inputs no longer applied. Verified on a
throwaway artifacts root: a receipt with an old fingerprint, and one with none at all, both validated
as satisfied.

Fixed in `workbench/scripts/pipeline.py`: `write_stage_receipt` records the stage's input fingerprint
(which already covers the commands and the model's preparation recipe) and validation requires it to
match the recorded fingerprint. A legacy receipt without the field is not trusted, so such a stage
simply runs again. Both negative cases and the matching case are exercised in this change.

### B5. Air-Know evaluates with the model in training mode

`AirKnow/test_BLIP2.py:10` calls `model.train()` on entry to `test(...)`, which the FashionIQ branch
uses. Its three siblings (INTENT, ConeSep, HABIT) call `model.eval()` at the same position. Dropout
and batch-norm therefore behave differently from the paper's protocol, and the RNG is unseeded.

### B6. Four evaluators have no CLI or loader at all

| Model | State | Evidence |
|---|---|---|
| HABIT | `test.py` has no `argparse`, no `__main__`; its only caller is the training loop | AST of `test.py`: only `load_obj`, `test`, `test_cirr_valset` |
| INTENT | same, plus `--restore_file` is declared and never read | `grep` finds the declaration only |
| TG-CIR | no source replay CLI; `replay/tgcir.py` is preflight-only | `replay_status: PREFLIGHT_ONLY` |
| CLVC-Net | `test.py` is one function taking constructed models | 148 lines, `def test(...)` only |

For each, the workbench adapter deliberately raises rather than inventing a command
(`adapters/models.py:24-25`), and preflight reports `command not runnable: COMMAND_UNAUDITED`.

### B7. Correction dictionaries are mandatory, unpinned and unpublished

HABIT, INTENT, AirKnow, ConeSep, ENCODER, TGCIR and LIMN all read
`captions/correction_dict_{cat}.json`, whose content is substituted into the query string. They are
absent locally, published nowhere in the pinned trees, and recorded as `OFFICIAL_ASSET_UNAVAILABLE`.
`registry/auxiliary_assets.yaml` has **no entry for any of those models**, so nothing declares,
checks or hashes them. A substituted correction changes the query text and therefore every score.

### B8. `resized_image/<cat>/*.jpg` does not exist

The iLearn layout requires JPEGs under `resized_image/`; the canonical dataset is 74,381 `.png` files
under `images/`. No generator is tracked upstream, and the layout is not derivable from the pinned
tree alone. Renaming `.png` to `.jpg` would be a data change, not a fix.

## P2 — reproducibility gaps

- **B9.** `paper_gate.py` has no production caller. `grep -rn paper_meta` finds only the module and
  its test, so the comparability guard cannot fire in a real report yet. **Related defect, fixed:**
  a *partial* paper row used to be an automatic pass. `_check_parity`
  (`workbench/backend/metrics_extraction.py`) and `evaluate_gate` both skipped any metric the paper
  row omitted, so SPRC's registered row — which publishes R@50 and mean but no R@10 — reported
  `PARITY_VERIFIED` / `PASS_WITHIN_5_PERCENT` for a reproduced R@10 of 0, 5 or 500 alike. Both now
  return `PARITY_NOT_EVALUATED` / `METRIC_REFERENCE_UNVERIFIED` when a metric has no paper value to
  compare against, because an absent reference is unverified, not exempt.
- **B10.** Neither `workbench/tests/test_download_auxiliary_assets.py`'s alias fixture nor any test
  exercises the real 2-3 MB caption end to end; the fixture patches the registry loader, so the
  integration between the downloader's manifest and preflight's reader is only covered by tests that
  share one fixture.
- **B11.** `workbench/replay/common.py` used raw `git status`, so a pinned checkout that had merely
  been imported read as dirty and the LIMN replay refused it. Fixed in this change; the runtime and
  the replay path now share the bytecode rule.
- **B12.** `environment_required` is unset on the iLearn models, so `environment_blockers` returns
  `[]` for environments that do not exist — the absence is silent rather than reported.
- **B13.** ptha_mtst's registry `notes` said no author repository exists; one does
  (`github.com/zyinan99/MTST_PTHA`, first-authored by the paper's first author). Corrected in this
  change to record the repository as evidence without claiming a synced checkout.
- **B14.** HINT, Air-Know, ConeSep, HABIT and INTENT publish LFS `sha256` and sizes on their Hubs,
  but the registry records `expected_sha256: null` / `expected_size_bytes: null` for them, so no
  download could be verified even if one were attempted.

## Not blockers, recorded so they are not mistaken for them

- `workbench/docs/reviews/models/hint.md` verdict is **VARIANT**, not BLOCKED: its pin, cohort
  assignment and paper numbers all check out; what it lacks is a runnable command and the same
  dataset prerequisites above.
- The 121-image shirt/toptee gallery-id overlap inside CSMCIR's merge is genuine upstream behaviour
  (`dict.update` over three category files, later category wins for shared ids). It is preserved, not
  fixed, and is a labelled ablation candidate only after an official evaluation exists.
