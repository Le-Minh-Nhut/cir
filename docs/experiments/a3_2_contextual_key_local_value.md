# A3.2: Contextual key, hard-private local value

Parent branch: `exp/e2e-a3.1-qasa-slot-filter-eval-winner`

Experiment branch: `exp/e2e-a3.2-contextual-key-local-value`

## Hypothesis

Current Edit Slots may clone one near-global edit because contextual,
reference-conditioned text states and teacher counterfactual effects can leak
near-global edit information into every slot value.

## Intervention

This experiment changes only the information path that constructs the Edit
Slot latent:

```text
contextual/reference-conditioned Q-Former text states
                         |
                         v
                ownership KEY + QASA
                         |
                 soft slot masks
                         |
                token-wise argmax
                         |
             hard exclusive support
                         |
                         v
raw CSMCIR Q-Former word embeddings (teacher_states.npy)
                         |
                 mass-aware pooling
                         |
                         v
                      slot_mlp
                         |
                         v
                     Edit Slots
```

Every valid content token contributes VALUE to exactly one candidate slot in
the forward pass. Padding and special tokens contribute to no slot. The hard
mask uses a straight-through estimator: its forward value is exactly the
one-hot partition, while its backward derivative follows the original soft
competitive ownership. This retains gradients for the slot queries and KEY
projections without allowing continuous soft routing weights to encode VALUE.

The counterfactual teacher quantities `q_full`, `q_minus`, and `slot_effects`
are still computed from the original soft ownership and returned for existing
diagnostics. `slot_effects` is not an input to `slot_mlp` or any other Edit-Slot
latent construction. QASA also remains on its independent FP32 contextual soft
attention path and does not gate VALUE ownership.

Experiment provenance is explicit:

```yaml
slot_value_source: teacher_raw
slot_effect_in_value: false
slot_value_assignment: hard_st_exclusive
```

The first `slot_mlp` layer therefore has `teacher_text_dim` inputs (768 in the
experiment config), rather than `text_dim + teacher_query_dim` inputs. A3.1
TAPER checkpoints are shape-incompatible and must not be reused. The frozen
CSMCIR teacher checkpoint and the existing aligned CSMCIR image/text caches are
reused unchanged.

## Preserved controls

The contextual KEY, soft competitive ownership diagnostics, content mask, QASA
algorithm and thresholds, frozen teacher, Executor, Router, Primitive Bank,
retrieval loss, optimizer, schedule, captions, and FashionIQ evaluation
protocol are unchanged. No new loss, supervision, regularizer, balancing
constraint, quota, sequential claiming, or semantic label is introduced.

## Falsification criterion

This intervention does not guarantee specialization. A giant slot may still
win every valid token, leaving all other VALUE slots exactly empty; this is an
intended falsification mode rather than an error to balance away.

After retraining, if functional Phi effective rank remains near one and both
REPEAT/FULL and MEANxK/FULL remain near one, hard information isolation alone
is not sufficient to produce functional specialization. Stop at that result;
do not add another mechanism inside this ablation.

## Commands

Train from scratch:

```bash
python src/train.py experiment=taper_e2e dataset.root=data/FashionIQ
```

Run a quick QASA hard-partition evaluation:

```bash
python src/evaluate_qasa_inference.py \
  --checkpoint outputs/REPLACE_WITH_RUN/best.pt \
  --dataset-root data/FashionIQ \
  --cache-root features \
  --config conf/experiment/taper_e2e.yaml \
  --device cuda \
  --max-queries-per-category 256 \
  --json-output reports/a3_2_contextual_key_local_value_qasa_quick.json
```

Set `--max-queries-per-category 0` for the complete validation set.

Run a quick P0 functional audit:

```bash
python src/audit_taper_merit_p0.py \
  --checkpoint outputs/REPLACE_WITH_RUN/best.pt \
  --dataset-root data/FashionIQ \
  --cache-root features \
  --config conf/experiment/taper_e2e.yaml \
  --device cuda \
  --max-queries-per-category 256 \
  --json-output reports/a3_2_hard_private_value_p0_quick.json
```

Run the full P0 functional audit:

```bash
python src/audit_taper_merit_p0.py \
  --checkpoint outputs/REPLACE_WITH_RUN/best.pt \
  --dataset-root data/FashionIQ \
  --cache-root features \
  --config conf/experiment/taper_e2e.yaml \
  --device cuda \
  --max-queries-per-category 0 \
  --json-output reports/a3_2_hard_private_value_p0_full.json
```
