# IAG-SRME OpenCLIP ViT-B/16 controlled backbone ablation

This branch changes exactly two scientific variables relative to the current FG-CLIP Base
IAG-SRME experiment:

1. the paired vision/text backbone is OpenCLIP ViT-B/16;
2. validation uses the repository's `fashioniq_val` protocol.

The IAG-SRME architecture, recurrent horizon, selector, STOP behavior, grounding equation,
editor, readout, scorer, objectives, optimizer, learning rate, precision, caption policy, and
all other training hyperparameters are unchanged.

## Pinned pretrained model

- OpenCLIP architecture: `ViT-B-16`
- pretrained tag: `laion2b_s34b_b88k`
- package: `open-clip-torch==3.3.0`
- weight repository: `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`
- immutable repository revision: `7288da5a0d6f0b51c4a2b27c624837a9236d0112`
- weight file: `open_clip_model.safetensors`
- native input: 224 × 224
- patch grid: 14 × 14 = 196
- retrieval dimension: 512

The reference path calls OpenCLIP `forward_intermediates` once. The anchor uses the final
contextual spatial tokens after the visual transformer and final visual normalization, drops
the CLS token through the public spatial-token API, applies the pretrained OpenCLIP visual
projection token-wise, then applies only `Linear(512,256) + LayerNorm(256)` for the existing
IAG width contract. The reference global and gallery embeddings use OpenCLIP's pretrained
pooled/projected retrieval representation.

For the pinned OpenCLIP 3.3.0 API, `image_indices=1` means “take the last one transformer
block,” `normalize_intermediates=True` applies `visual.ln_post`, and the default
`image_output_extra_tokens=False` (passed explicitly) removes the CLS prefix from the NLC
intermediates. The
adapter asserts the resulting native tensor is `[B,196,768]` before applying the pretrained
visual projection. A real-checkpoint smoke invariant compares the reference global from this
single-pass path with `encode_image(..., normalize=True)` for the same pixels.

The text path uses the matching OpenCLIP text tower. Final contextual token states are mapped
to width 256 with `Linear + LayerNorm`; the official projected global text feature remains the
auxiliary semantic global. No FG-CLIP text representation is mixed into this experiment.
The collator-validity mask follows OpenCLIP's own EOT argmax contract: it includes the
contiguous SOT-through-EOT sequence, then the generic collator excludes SOT and EOT to form
the IAG content mask. It does not interpret token value zero as padding semantics.

## Trainability

The policy matches FG-CLIP Base full fine-tuning:

- OpenCLIP visual transformer and pretrained visual projection: trainable;
- OpenCLIP token embedding, positional embedding, text transformer, and final text norm:
  trainable;
- IAG image/text compatibility projections: trainable;
- OpenCLIP text retrieval projection: frozen, matching the baseline's frozen auxiliary-only
  retrieval text projection;
- OpenCLIP logit scale: frozen and unused by IAG-SRME.

No LoRA, partial unfreezing, layer-wise decay, new optimizer, scheduler, or warmup is added.

## FashionIQ VAL protocol

Queries are the deterministic `ordered_and` FashionIQ validation queries. For each category,
the `fashioniq_val` gallery is the ordered duplicate-free union of reference and target image
IDs in that category's validation annotations, as implemented by
`build_pair_union_gallery`. Retrieval keeps the existing rule that removes the query's
reference image from that query's ranking unless reference and target are the same image.
Targets never enter model forward and are used only for retrieval metrics or offline oracle
diagnostics.

## Commands

Install the pinned dependency:

```bash
python -m pip install -e '.[dev]'
```

Run the real pretrained integration smoke:

```bash
PYTHONPATH=src python src/smoke_openclip_integration.py --device cpu
```

Run the CUDA FP16 canary before full training:

```bash
python src/canary_train_iag_srme.py \
  --backbone openclip_b16 \
  --dataset-root data/FashionIQ \
  --steps 100 \
  --precision fp16
```

Launch the matched experiment:

```bash
python src/train.py \
  backbone=openclip_b16_full \
  experiment=iag_srme_openclip_b16_valsplit \
  protocol=fashioniq_val \
  objective=core \
  dataset.root=data/FashionIQ
```

Evaluate a checkpoint:

```bash
python src/evaluate.py \
  backbone=openclip_b16_full \
  experiment=iag_srme_openclip_b16_valsplit \
  protocol=fashioniq_val \
  dataset.root=data/FashionIQ \
  checkpoint=/absolute/path/to/best_val.pt
```

Training evaluates the same model state after every epoch on both `fashioniq_original` and
`fashioniq_val`. It encodes each category's validation queries once, encodes the full original
gallery once, and selects pair-union embeddings in the exact `fashioniq_val` ID order. The
independent outputs are `best_original.pt`, `best_val.pt`, and `last.pt`; no ambiguous
`best.pt` is written. Every checkpoint stores both current protocol metric groups.

Run checkpoint diagnostics. The runner reconstructs OpenCLIP from checkpoint metadata and
accepts the gallery protocol explicitly:

```bash
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint outputs/openclip-b16/best_val.pt \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_val \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/iag_srme_openclip_b16_best.json
```

Pass `--protocol fashioniq_original` to diagnose the exact same checkpoint against the full
original gallery. If omitted, diagnostics use an unambiguous checkpoint selection protocol;
`last.pt` and legacy checkpoints without such provenance fall back to `fashioniq_original`.
