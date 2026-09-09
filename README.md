# IAG-SRME V2 Sequential R0

Clean sequential-context-edit experiment for composed image retrieval with the existing
FG-CLIP backbone and learned ProposalNet context slots.

```text
Reference + instruction
        ↓
ProposalNet (once at V0)
        ↓
C0  C1  C2  C3
        ↓
C0 applied to V0 → V1
C1 applied to V1 → V2
C2 applied to V2 → V3
C3 applied to V3 → V4
        ↓
terminal retrieval from V4
```

The four context edits are learned latent slots; this experiment does not claim semantic
specialization for any slot. Grounder, ActionFusion, and one shared Executor recompute their
visual consequences from the evolving state at every slot.

Architecture constraints:

- ProposalNet runs exactly once and produces four fixed ordered context edits.
- Every context edit executes exactly once in fixed order.
- Grounding and execution for slot `j` use state `Vj`.
- One Executor instance and one parameter set are shared across all slots.
- Only final state `V4` is returned for retrieval and supervised during training.
- No ScoreNet, hard selector, learned STOP, aWTA, DPP, prefix loss, or state mixture.

The FG-CLIP readout and fine-tuning policies remain available as orthogonal backbone choices:

- `learned_qg` or `native_cls` global readout;
- `full` or `text_only` fine-tuning.

## Setup and tests

```bash
python -m pip install -e '.[dev]'
pytest -q
ruff check src tests
```

## One-batch FashionIQ canary

```bash
python src/canary_train_iag_srme.py \
  --dataset-root data/fashionIQ_dataset \
  --steps 1 --batch-size 2 --precision fp32
```

The canary checks four transitions, finite loss/gradients, required gradient paths, and reports
basic CUDA peak memory when CUDA is available.

## Train

```bash
# Learned global query, full fine-tuning (default)
python src/train.py backbone=fgclip_base_full_qg

# Native CLS readout, full fine-tuning
python src/train.py backbone=fgclip_base_full_native_cls

# Frozen vision, trainable text-side model
python src/train.py backbone=fgclip_base_text_qg

# Native CLS with frozen vision
python src/train.py backbone=fgclip_base_text_native_cls
```

Official FashionIQ evaluation uses only the final `V4` query against the gallery and reports
R@10, R@50, and mean recall. It does not ensemble or choose among prefix states.
