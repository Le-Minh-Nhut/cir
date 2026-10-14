# Final reproducibility audit

Independent, adversarial audit of all 16 registered models plus the shared workbench code. 16
model audits are in `workbench/docs/reviews/models/`; the unresolved items are in
`OPEN_BLOCKERS.md`. This file is the cross-cutting summary.

## What was actually executed

| Claim | Evidence |
|---|---|
| No model was evaluated | `workbench/artifacts/checkpoints/` holds only `.gitkeep`; no result JSON exists outside `results/*/mock/` |
| No metric was reproduced | every registry number is a paper value; the gate exists but has no real run to judge |
| The GPU cannot be used | `nvidia-smi` shows an RTX 3050 (4 GiB) but every environment here has a CPU-only torch build |
| Three real evaluations are impossible for a second, independent reason | two backbone caches are corrupt placeholders: `blip2_pretrained.pth` is 22 bytes of text, `eva_vit_g.pth` is 4 bytes of text |

Every audit is `STATIC_AUDIT` unless it names a command and its output. The single genuine
`RUNTIME_VERIFIED` result in this engagement is the CSMCIR import, described below.

## Model verdicts

| model_id | verdict | paper row | published checkpoint condition | decisive blocker |
|---|---|---|---|---|
| csmcir | BLOCKED | 57.07 / 77.27 / 67.17 | clean (verified) | pinned package cannot import; the gallery caption input was never published |
| airknow | BLOCKED | 55.75 / 75.72 / 65.74 | 50 %, 80 % | paper row is 0 %-noise; no clean checkpoint; evaluates in training mode |
| conesep | BLOCKED | 55.58 / 75.88 / 65.73 | 20 %, 50 %, 80 % | paper row is 0 %-noise; no clean checkpoint; no runnable evaluator |
| ptha_mtst | BLOCKED | 54.99 / 76.28 / 65.64 | none published | no FashionIQ checkpoint and no MTST corpus; an author repo exists but is unsynced |
| habit | BLOCKED | 55.38 / 75.20 / 65.29 | 20 %, 50 %, 80 % | paper row is 0 %-noise; no clean checkpoint; no CLI |
| intent | BLOCKED | 55.31 / 75.16 / 65.24 | 20 %, 50 %, 80 % | paper row is 0 %-noise; no clean checkpoint; no CLI or restore path |
| hint | VARIANT | 58.48 / 77.93 / 68.21 | clean | pin, cohort and paper numbers verified; no runnable command and the dataset contract is unmet |
| encoder | BLOCKED | 56.13 / 77.59 / 66.86 | clean | evaluator imports a module absent from the pinned tree |
| pair | BLOCKED | 52.76 / 74.51 / 63.64 | B1 / B2, mapping unverified | which blob is the FashionIQ model cannot be established from the pinned tree |
| clvc_net | BLOCKED | none recorded | unresolved | no CLI or loader; placeholder checkpoint; ambiguous ResNet-50 path |
| dcnet | BLOCKED | none recorded | unresolved run directory | legacy stack, no verified mapping, unseeded PKL regeneration risk |
| combiner_rn50x4_noft | BLOCKED | none recorded | unresolved | registry names a folder, not a file |
| clip4cir_rn50x4_fullft | BLOCKED | none recorded | unresolved bundle | two artefacts genuinely required, neither identified |
| tgcir | BLOCKED | 51.32 / 73.09 / — | unresolved archive | no source replay CLI; whole-model pickle with no author digest |
| sprc | BLOCKED | — / 74.97 / 64.85 | unresolved | `sprc_fiq.pt` backbone mapping unresolved; registry row partial |
| limn | BLOCKED | none recorded | base iteration 0, not verified clean | checkpoint bundle absent; replay refuses a dirty checkout (fixed) |

One `VARIANT`, fifteen `BLOCKED`, zero `VERIFIED`. That is the honest result: this repository
contains no reproduced metric.

## What the audits found that was not known before

- **The registry's "no source exists" for ptha_mtst was wrong.** An author repository exists
  (`github.com/zyinan99/MTST_PTHA`), first-authored by the paper's first author, README title
  identical to the paper. Corrected in the registry as evidence, without claiming a synced checkout.
- **PAIR's two checkpoints are published** on an author Hugging Face repository with LFS digests,
  which the registry did not record. The *mapping* to the FashionIQ row remains unverified.
- **Combiner's Google Drive folder is machine-readable** via `embeddedfolderview`, which names the
  file behind each variant — the "does not document which file" claim was stale.
- **Four evaluators have no CLI at all** (HABIT, INTENT, TG-CIR, CLVC-Net), so `COMMAND_UNAUDITED`
  is not a workbench gap but an upstream one.
- **Air-Know evaluates with `model.train()`** while its three siblings call `model.eval()`.
- **The correction dictionaries are mandatory and unpublished** for seven models, and no
  auxiliary-asset entry declares them for any of them.

## Defects fixed in this engagement

| Defect | Severity | Evidence of fix |
|---|---|---|
| A modified committed `.pyc` changed the code that runs while the source stayed pinned | P0 | `test_src11`-`test_src13`; the live AirKnow checkout was reporting 602 deleted tracked `.pyc` as clean |
| A stage receipt from an earlier invocation could certify a later one | P1 | stale and fingerprint-less receipts now refused; matching one accepted |
| A partial paper row was an automatic pass (SPRC's row omits R@10) | P1 | reproduced R@10 of 0, 5, 500 now returns `METRIC_REFERENCE_UNVERIFIED` |
| Alias staging could overwrite an operator's own file | P1 | operator file preserved, exit nonzero |
| Alias acceptance could be forged with a self-consistent manifest | P1 | bytes must equal the registry's published digest |
| `replay/common.py` refused any checkout that had merely been imported | P2 | LIMN replay now passes; runtime and replay share the rule |

## What an operator should do next

1. Reclaim disk space; 5.2 GiB free cannot hold even one checkpoint bundle.
2. Obtain a CUDA-enabled torch in a per-model environment built from the pinned
   `requirements.txt`.
3. Stage one non-corrupt backbone cache and re-run the CSMCIR import smoke test.
4. For CSMCIR only, decide whether to apply `workbench/patches/csmcir_models_init_compat.patch`
   and run the official evaluation with the labelled caption variant. That is the single
   highest-information experiment available, and it is the only path that could turn a BLOCKED into
   a VARIANT-with-numbers.
