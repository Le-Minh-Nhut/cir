# IAG-SRME V2

Clean research implementation of the recurrent IAG-SRME model for composed image
retrieval. The architecture follows the
[V2 canonical specification](doc/CIR_IAG_SRME_UNIFIED_CANONICAL_ARCHITECTURE_AND_TRAINING_SPEC_V2_2026-09-06.md).

The current implementation is Track B (FG-CLIP v1 Base) with two controlled global
readout modes:

- `R0-QG` / `learned_qg`: canonical learned recurrent global query;
- `R0-NCLS` / `native_cls`: immutable image-specific penultimate CLS and current
  patches feed the exact CLS row of the native final vision block; the full-token call
  remains a parity oracle.

Both modes keep the following architecture fixed:

- persistent state is the penultimate patch representation, without CLS;
- proposals are regenerated from current visual state and instruction tokens;
- grounding uses native dense cosine evidence with separate softmax read and sigmoid write maps;
- actions use entity-conditioned independent gates;
- all candidates are masked local residual previews from one parent state;
- one shared ScoreNet predicts absolute marginal utility against KEEP/STOP = 0;
- rollout uses hard argmax/gather, with no Gumbel, straight-through estimator, or state mixture;
- training targets appear only in the external retrieval teacher and terminal loss;
- instruction concepts supervise the pooled proposal set at `t=0` only;
- relation binding uses a prototype bank separate from proposal queries;
- Functional DPP operates on retrieval consequences `delta_q` and detached executed history;
- correspondence is disabled.

The complete architecture is in
[`src/models/iag_srme/model.py`](src/models/iag_srme/model.py). FG-CLIP-specific
state/readout code and teacher retrieval helpers are the only extracted utilities.
The current implementation/verification snapshot is recorded separately in the
[V2 R0 experiment status](doc/CIR_IAG_SRME_V2_R0_EXPERIMENT_IMPLEMENTATION_STATUS_2026-09-06.md).

## Setup

```bash
python -m pip install -e '.[dev]'
```

The default config pins `qihoo360/fg-clip-base` at revision
`454d76372c2cf5eb48fa0d871fd0534481484d97`.

## Test

```bash
pytest -q
ruff check src tests
```

## FashionIQ smoke update

Expected data layout:

```text
data/fashionIQ_dataset/
├── captions/
├── image_splits/
└── images/
```

Run one small target-firewalled train update:

```bash
python src/canary_train_iag_srme.py \
  --dataset-root data/fashionIQ_dataset \
  --steps 1 --batch-size 2 --precision fp32
```

Add `--global-readout-mode native_cls` for the R0-NCLS canary.
On CUDA, the canary JSON also reports peak allocated and reserved memory in GiB.

## Train

```bash
# R0-QG
python src/train.py backbone=fgclip_base_full_qg

# R0-NCLS: the only intended difference is global_readout_mode
python src/train.py backbone=fgclip_base_full_native_cls
```

The default objective config enables all requested A6 auxiliaries. Each term can be
ablated independently with Hydra overrides such as
`objective.concept_enabled=false` or `objective.dpp_enabled=false`.

Dataset loading, FashionIQ evaluation, AMP policy, checkpointing, optimizer ownership
checks, and stable `CIRSample.target_id` handling are retained from the clean-rewrite
branch.
