# A8.0 — FG-CLIP2 Entity–Action Binding

## Hypothesis

A8.0 tests whether one shared set of learned relation queries can bind paired
information from two frozen FG-CLIP2 streams:

```text
                         shared relation queries R[K,1024]
                         /                         \
frozen reference dense tokens                 frozen text tokens
             |                                      |
             v                                      v
      entity[K,1024]                         action[K,1024]
             \                                      /
              paired feature-wise affine fusion
                              |
                    mean + L2 normalization
                              |
                      composed CIR query
```

The image branch is also applied to target/gallery frozen dense features, so
the learned retrieval target is `normalize(mean([image_global, entities]))`.
Gallery embeddings are recomputed once per validation category/evaluation call;
only frozen global/dense FG-CLIP2 features are cached.

## Isolation and attribution

This experiment adapts the shared Entity–Action relation-binding intuition from
[AAAI25-ENCODER](https://github.com/iLearn-Lab/AAAI25-ENCODER). It does not copy
the implementation verbatim: A8.0 uses standard masked cross-attention, does not
softmax the feature dimension, and omits ENCODER LFF/ScoreNet/FactorNet.

Unlike TAPER, A8.0 has no QASA, competitive ownership, NULL slot, primitive
bank, router, recurrent executor, teacher, or CSMCIR path. It keeps the frozen
FG-CLIP2-Large model and immutable revision used by A3.2.

## Frozen cache contracts

Global image caches remain unchanged (`images.npy [N,1,1024]`). A new sibling
`dense_images/` cache stores:

```text
values.npy          [total_visual_tokens,1024], float16 by default
offsets.npy         [num_images+1], int64
spatial_shapes.npy  [num_images,2], int32
name_to_idx.json
manifest.json
```

For each image, only the official `real_h * real_w` prefix from
`get_image_dense_feature()` is retained. Runtime gathering pads only the current
batch and emits a boolean real-token mask. It never moves the whole cache to the
GPU.

Text caches remain backward-compatible. With `--save-global`, they additionally
contain normalized `global.npy [Q,1024]` from official short-walk
`get_text_features`. A8.0 fails loudly if this file is absent; old consumers do
not require it. Cached captions/sample IDs and content masks retain A3.2 policy.

## Shapes and architecture

With `B` samples, `K=4`, text length `Nt=64`, and ragged padded visual length
`Nv`:

```text
reference_global, text_global, target_global  [B,1024]
reference_dense, target_dense                  [B,Nv,1024]
text_states                                    [B,Nt,1024]
entity, action                                 [B,K,1024]
vision_attention                              [B,K,Nv]
text_attention                                [B,K,Nt]
image_tokens, text_tokens                      [B,K+1,1024]
composed query, target embedding               [B,1024]
```

The affine module predicts `[gamma,beta]` from each paired token and computes
`gamma * reference + beta * text`. Its final layer starts with zero weights,
gamma bias 1, and beta bias 0, making initialization approximately reference
preserving without detaching either learned path.

All global, entity, and action tokens are L2-normalized before paired affine
fusion. This representation-interface contract prevents relation tokens from
dominating frozen normalized FG-CLIP2 global tokens purely through norm
magnitude. It removes a magnitude shortcut; it does not prevent relation
collapse or guarantee functional specialization.

## Objectives

The total objective is:

```text
L = 1.0 L_retrieval + 1.0 L_entity_action + 1.0 L_relation_ortho
```

- `L_retrieval` is the A3.2 query-to-target log-sum-exp contrastive objective;
  all batch rows with the same target ID are positives.
- `L_entity_action` uses symmetric unweighted set matching. It averages
  action-to-entity and entity-to-action max matching, then applies bidirectional
  batch InfoNCE. No semantic labels are used.
- `L_relation_ortho = MSE(normalize(R) normalize(R)^T, I)` is a weak geometric
  regularizer on the one shared query tensor only.

## Diagnostics and forensic interventions

Training reports relation off-diagonal cosine; per-relation image/text attention
entropy, normalized entropy, maximum probability, and effective support; entity
and action cross-slot cosine; and same-index versus off-index entity/action
cosines.

`diagnose_entity_action_binding.py` reports FULL, GLOBAL_ONLY, DROP-k, SINGLE-k,
REPEAT-k, best SINGLE/FULL, and best REPEAT/FULL using the unchanged FashionIQ
gallery protocol. These are diagnostics, not additional objectives.

## Commands

```bash
python3 src/precompute_fgclip2_dense_images.py \
  --dataset-root data/fashionIQ_dataset \
  --cache-root features/fashioniq/fgclip2-large \
  --storage-dtype float16

python3 src/precompute_fgclip2_text.py \
  --dataset-root data/fashionIQ_dataset \
  --cache-root features/fashioniq/fgclip2-large \
  --correction-policy fashioniq \
  --text-cache-subdir text \
  --save-global

python3 src/train_entity_action_binding.py experiment=encoder_binding_e2e \
  experiment.num_epochs=1 experiment.batch_size=8

python3 src/train_entity_action_binding.py experiment=encoder_binding_e2e

python3 src/diagnose_entity_action_binding.py \
  --checkpoint outputs/<run>/best.pt \
  --dataset-root data/fashionIQ_dataset \
  --cache-root features/fashioniq/fgclip2-large
```

## Known limitations

Shared relation queries plus orthogonality do **not** guarantee functional
specialization. A8.0 is designed to empirically test whether this inductive bias
is sufficient before adding stronger anti-collapse mechanisms. Set matching uses
in-batch negatives and is therefore batch-composition dependent. Dense caches are
larger than global caches, and learned gallery binding adds validation compute.
