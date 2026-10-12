# FashionIQ protocol parity

Which protocol each registered model's pinned evaluator actually implements, traced to source,
and whether the workbench can express it. Static source evidence only: no dataset, checkpoint,
environment or GPU evaluation was run for this document.

## The three protocol families

| Protocol ID | Candidate universe | Reference | Literature label |
|---|---|---|---|
| `fashioniq_original_split` | complete `image_splits/split.{category}.val.json`, in file order | eligible (may outrank the target) | `original` |
| `fashioniq_full_gallery_ref_excluded` | the same complete split file, in file order | excluded from that query's ranking | `original` |
| `fashioniq_val_split` | first-seen ordered union of each validation annotation's reference then target | excluded from that query's ranking | `val` |

"Excluded" is the cohort property, which is what a comparison depends on. The pinned evaluators
implement it by masking the reference score to `-10e10` and sorting; the workbench removes the
column. Both make the reference unretrievable, by different means — see the note below.

These are three different cohorts. A target's rank — and therefore Recall@K — changes when the
reference is removed, and changes again when the gallery shrinks to the pair union. No
cross-protocol aggregate, consensus, Jaccard, rank-disagreement or common-distractor request is
valid, and the workbench refuses them.

## Workbench implementation, after this pass

`src/evaluation/fashioniq.py` holds one authoritative table, `FASHIONIQ_PROTOCOLS`, mapping each
protocol id (and its accepted alias) to its gallery source and its reference policy.
`resolve_fashioniq_protocol()` raises for an unknown name, so no misspelling can default to
reference-eligible evaluation, and every entry point resolves through the table rather than
deciding the reference rule from a caller argument.

- `build_fashioniq_gallery(protocol, ...)` resolves the protocol and branches on its declared
  gallery. The two full-gallery
  families share a candidate universe — the exception is applied per query, not by shrinking the
  gallery — while `fashioniq_val` builds the pair union.
- `recall_at_k` is reference-eligible. `recall_at_k_reference_excluded` **removes** the reference
  column from the ranking. The pinned evaluators instead *mask* the reference score to `-10e10`
  and then sort (`sims[i, targets.index(source)] = -10e10` before `argsort`). The two agree on the
  top-k hit test whenever every other score clears the mask, but they are different operations,
  and the mask relies on that bound; the workbench removes the column explicitly and its tests pin
  the cohort's behaviour rather than asserting numeric equivalence with the mask.
- `evaluate_fashioniq_recall(..., reference_ids=None | Sequence[str])` requires references exactly
  when exclusion is active. A protocol mix-up therefore cannot be silently evaluated as a plain
  one.
- Inputs that could not produce an interpretable number are refused rather than averaged: a
  non-finite score, an empty query set, a gallery smaller than k, and a reference or target absent
  from the gallery it is ranked against. Ties remain an ambiguity of the ranking itself, which the
  workbench does not claim to pin.
- The denominator is always the query count. `macro_average_fashioniq` takes the equal-weight
  mean of category recalls, and the reported mean is the arithmetic mean of macro R@10 and R@50.

Two gaps existed before this pass. `fashioniq_full_gallery_ref_excluded` — the native protocol of
five registered models — had no branch at all and raised `Unsupported FashionIQ protocol`. And
the two older families were reachable only through the shorter internal spellings
(`fashioniq_original`, `fashioniq_val`), so a caller holding a **registry** protocol id could not
build either of them. Both spellings are now accepted for each family, and a test asserts that
every protocol id the registry declares is buildable.

## Per-model evidence

| Registered model | Registry protocol | Pinned evaluator evidence | Gallery | Reference | Expressed by the workbench? |
|---|---|---|---|---|---|
| csmcir | `fashioniq_original_split` | `CSMCIR/src/validate_blip_csmcir.py:compute_fiq_val_metrics`; `data_utils_csmcir.py:FashionIQDataset.__init__` | full ordered split | eligible | yes (native evaluator runs; generic route now also expressible) |
| limn | `fashioniq_val_split` | `LIMN/LIMN/datasets.py:FashionIQ.get_test_data`; `LIMN/LIMN/test.py:test` | pair union, first-seen | masked before ranking | yes |
| encoder | `fashioniq_val_split` | `ENCODER/datasets.py:FashionIQ`; `evaluate_model.py` | pair union | masked before ranking | yes |
| pair | `fashioniq_val_split` | pinned `test.py` (callable only) | pair union | masked before ranking | yes, but no runnable command |
| hint | `fashioniq_val_split` | `HINT/datasets.py:FashionIQ.get_test_data`; `test.py:test` | pair union, ordered | masked before ranking | yes, but no runnable command |
| clvc_net | `fashioniq_val_split` | `CLVCNet/datasets.py:FashionIQ.get_test_targets`; `test.py:test` | source-order endpoint union | masked before ranking | yes, but no runnable command |
| tgcir | `fashioniq_val_split` | `TGCIR/datasets.py`; `test.py` | source-order endpoint union | masked before ranking | yes, but no runnable command |
| airknow | `fashioniq_full_gallery_ref_excluded` | `AirKnow/datasets.py:FashionIQ.get_test_data`; `test_BLIP2.py:test` | full ordered split | masked to `-10e10` before ranking | yes (was `Unsupported`) |
| conesep | `fashioniq_full_gallery_ref_excluded` | `ConeSep/datasets.py:FashionIQ.get_test_data`; `test.py:test` | full ordered split | masked before ranking | yes (was `Unsupported`) |
| habit | `fashioniq_full_gallery_ref_excluded` | `HABIT/datasets.py:FashionIQ.get_test_data`; `test.py:test` | full ordered split | masked before ranking | yes (was `Unsupported`) |
| intent | `fashioniq_full_gallery_ref_excluded` | `INTENT/datasets.py:FashionIQ.get_test_data`; `test.py:test` | full ordered split (`original-split` default; the `val-split` branch is optional) | masked before ranking | yes (was `Unsupported`) |
| dcnet | `fashioniq_full_gallery_ref_excluded` | `DCNet/data_loader/CE_dataset.py:CE.__init__` (`val_trg`); `trainer/trainer_joint.py:TrainerJoint._valid_epoch` | full ordered split | masked before ranking, both branches, before log-softmax fusion | yes (was `Unsupported`) |
| combiner_rn50x4_noft | `fashioniq_original_split` | `CLIP4Cir/src/validate.py` | full ordered split | eligible | yes |
| clip4cir_rn50x4_fullft | `fashioniq_original_split` | `CLIP4Cir/src/validate.py` | full ordered split | eligible | yes |
| sprc | `fashioniq_original_split` | `SPRC/src/validate_blip.py:compute_fiq_val_metrics`; `data_utils.py:FashionIQDataset` | full ordered split | eligible | yes, but the native inference ignores its reference argument — see below |

## Findings that change what can be claimed

1. **The third protocol was advertised but unimplemented.** Fixed: the branch exists, with the
   reference rule enforced by signature rather than by convention.

2. **INTENT's registry protocol is its default, not its only branch.** `datasets.py` has both
   `original-split` and `val-split` gallery construction; the launcher defaults to
   `original-split`. The registry label therefore describes the default path, and the presence of
   a pair-union branch in the source must not be read as "INTENT is val-split".

3. **SPRC's native similarity is not reference-conditioned.** `blip2_qformer_cir_rerank.py`'s
   `inference` accepts `reference_embeds` and never reads it: the score is text-to-gallery. The
   workbench's own retrieval path *is* reference-conditioned, so local SPRC-shaped scores cannot
   stand in for the official ones even when the gallery and the reference rule match. This is a
   model-path mismatch, not a protocol mismatch, and it is recorded as such.

4. **No iLearn method publishes a category macro.** Their evaluation functions return
   per-category R@1/R@10/R@50, and the training loops select checkpoints on a *sum* of R@10 across
   categories. The workbench's macro R@10/R@50 is a workbench aggregate of source per-category
   numbers; it must never be presented as a source-reported metric.

5. **Several methods leave the reference eligible but never use it.** CSMCIR, the CLIP4Cir pair and
   SPRC are `fashioniq_original_split`; for SPRC the reference is passed to a model path that
   ignores it, so "reference eligible" describes the ranking pool, not the query construction.

6. **Gallery ordering is source-file order everywhere.** No inspected evaluator sorts or dedups
   the split file; the workbench preserves declared order and asserts uniqueness rather than
   sorting, so an official split with a surprising order is reproduced rather than corrected.

## Pending on the GPU host

Nothing in this document is a reproduction. Still owed, per model, in the audit documents: fetch
and hash the author checkpoint, load it with source-compatible code, run the native evaluator,
capture per-category and per-query rankings, and compare against the paper-reported numbers as
sanity references only. For methods whose evaluator is a callable with no CLI (PAIR, CLVC-Net,
TG-CIR, Air-Know, ConeSep, HABIT, INTENT, HINT, SPRC, DCNet), reaching a runnable command under
the pinned source is itself an unresolved item, recorded in `PHASE2_SOURCES_*.md`.
