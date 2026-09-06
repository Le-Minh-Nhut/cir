# IAG-SRME V2

Clean research implementation of the recurrent IAG-SRME model for composed image
retrieval. The architecture follows the
[V2 canonical specification](doc/CIR_IAG_SRME_UNIFIED_CANONICAL_ARCHITECTURE_AND_TRAINING_SPEC_V2_2026-09-06.md).

The current implementation is canonical Track B (FG-CLIP v1 Base):

- persistent state is the penultimate patch representation, without CLS;
- proposals are regenerated from current visual state and instruction tokens;
- grounding uses native dense cosine evidence with separate softmax read and sigmoid write maps;
- actions use entity-conditioned independent gates;
- all candidates are masked local residual previews from one parent state;
- one shared ScoreNet predicts absolute marginal utility against KEEP/STOP = 0;
- rollout uses hard argmax/gather, with no Gumbel, straight-through estimator, or state mixture;
- training targets appear only in the external retrieval teacher and terminal loss.

The complete architecture is in
[`src/models/iag_srme/model.py`](src/models/iag_srme/model.py). FG-CLIP-specific
state/readout code and teacher retrieval helpers are the only extracted utilities.

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

## Train

```bash
python src/train.py \
  backbone=fgclip_base_full \
  model=iag_srme \
  objective=core \
  experiment=iag_srme_base_full
```

Dataset loading, FashionIQ evaluation, AMP policy, checkpointing, optimizer ownership
checks, and stable `CIRSample.target_id` handling are retained from the clean-rewrite
branch.
