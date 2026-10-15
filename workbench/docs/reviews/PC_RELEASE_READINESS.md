# PC release readiness

Assessed on the preparation host. **Nothing here is a claim that any model reproduces its paper**: no
GPU evaluation has run anywhere. Read this together with `OPEN_BLOCKERS.md` and
`workbench/docs/reviews/models/*.md`.

## The one command to start with

```bash
python workbench/scripts/pc_readiness.py            # per model: what stops it, exactly
python workbench/scripts/pc_readiness.py --json     # the same, machine-readable
```

It is read-only: it writes nothing, downloads nothing and runs no evaluator. Run it on the PC before
preparing anything, and again after each step, to see the classification move.

## What is true today

| Area | State |
|---|---|
| Repository | branch `research/cir-failure-workbench`, tests green, no absolute host path in any runnable file |
| Papers | 33 committed reference rows covering all 16 registered models; every row built from the paper's own text and independently re-read by a second pass |
| Benchmark gate | single canonical implementation, plain Python, wired into the evaluation report, no model consulted at runtime |
| Data preparation | reads the PNG canonical distribution, produces the evaluator's JPEG layout, and refuses to reuse a derived file without evidence it came from the current source |
| CSMCIR | import repaired by a verified approved patch; caption alias available opt-in and labelled; the overlay workflow is tested end to end |
| LIMN | source-faithful replay intact; the resize recipe is corrected; the checkpoint bundle is registered with hashes |
| Environment | no model environment exists on the preparation host; every model is blocked on that or on an upstream defect |
| GPU | absent on the preparation host; the PC target is an RTX 5070 Ti (sm_120) |

## Classification of the 16 registered models

States are from `pc_readiness.py`. `PC_GPU_VALIDATION_PENDING` means preparation is complete but the
model's own interpreter has not recorded a verified GPU capability (a CUDA-declaring model stays
pending until `verify_gpu_runtime()` has run for it on the GPU host) — it is never reported as ready.
Each registered checkpoint also carries its own blocker list (presence, hash, bundle, mapping).

| Model | State on the preparation host | What would change it |
|---|---|---|
| `limn` | `BLOCKED_ENVIRONMENT` | provision the model environment and stage the three category checkpoints |
| `csmcir` | `BLOCKED_SOURCE` | apply the approved patch to a derived checkout and point the config at it |
| `airknow`, `conesep` | `BLOCKED_SOURCE` | the pinned evaluator has no CLI; a source-faithful wrapper would be needed, and the paper's 0 % row has no published weight |
| `habit`, `intent` | `BLOCKED_IMPLEMENTATION` / `BLOCKED_SOURCE` | a source-faithful replay exists for neither; both publish only noise-trained weights |
| `hint` | `BLOCKED_IMPLEMENTATION` | an audited replay command |
| `encoder` | `BLOCKED_DATASET` | the `resized_image/` tree, plus the `datasets1` module absent from the pinned source |
| `pair` | `BLOCKED_IMPLEMENTATION` | the checkpoint-to-paper mapping cannot be established |
| `clvc_net` | `BLOCKED_IMPLEMENTATION` | no CLI; the ResNet-50 asset path is contradictory |
| `dcnet` | `BLOCKED_ENVIRONMENT` | the unresolved run directory and its two required files |
| `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` | `BLOCKED_ENVIRONMENT` | the two-artefact bundle for fullft, and the base CLIP cache |
| `tgcir` | `BLOCKED_IMPLEMENTATION` | no source replay CLI; the archive's FashionIQ member is unidentified |
| `sprc` | `BLOCKED_IMPLEMENTATION` | the `sprc_fiq.pt` backbone mapping |
| `ptha_mtst` | `BLOCKED_SOURCE` | the author repository exists but is unsynced, and no FashionIQ checkpoint or corpus is published |

The states above were re-verified against `pc_readiness.py` after round 2 merged the
per-checkpoint blockers (presence, hash, bundle, mapping) into the model classification: the
table matches the tool's output on the preparation host at regeneration time, and the per-checkpoint
blocker lists live in the `--json` report.

## What the preparation host could and could not establish

**Established here:**
- the CSMCIR import defect, its patch, and that the patch changes no evaluator semantics (the patched
  file's 332 statements are identical in sequence to the original's);
- that the canonical dataset is PNG while the pinned readers open JPEG under a derived tree, and a
  recipe that produces that tree;
- that the committed benchmark rows match their sources, through a second independent read;
- that the gate refuses mismatched protocols, checkpoints and scales.

**Not established here, and owed to the PC:**
- any Recall number at all;
- that a checkpoint loads, that a backbone cache is valid, or that a model's forward pass runs;
- that a CUDA build executes on `sm_120`;
- that the numbers agree with the papers.

## The order that wastes least time

1. `pc_readiness.py` — see the starting classifications.
2. LIMN. It is the only model whose replay is source-faithful *and* whose checkpoints are hashed, so it
   is the cheapest real evaluation and the first real test of the environment.
3. CSMCIR as a **declared variant**: build the derived checkout, apply the approved patch, stage the
   caption alias, and run. It is the second pilot because it exercises the hardest parts (LAVIS import,
   caption alias, fixed dataset link) and its reference row is verified.
4. Everything else, in whatever order its blocker clears.
