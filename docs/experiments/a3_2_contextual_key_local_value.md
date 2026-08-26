# A3.2: Contextual key, local VALUE assignment ablation

Parent branch: `exp/e2e-a3.1-qasa-slot-filter-eval-winner`

Experiment branch: `exp/e2e-a3.2-contextual-key-local-value`

## Hypothesis

This comparison tests one question in isolation: is soft VALUE sharing, or the
continuous routing-weight information it exposes, a meaningful cause of slot
functional collapse?

Both modes keep contextual/reference-conditioned KEY states, raw teacher word
embeddings as VALUE, and exclude teacher counterfactual effects from the slot
latent. Only VALUE assignment differs.

## Intervention

| Mode | KEY | VALUE source | VALUE assignment |
| --- | --- | --- | --- |
| `soft_shared` | contextual | raw teacher embedding | soft shared |
| `hard_st_exclusive` | contextual | raw teacher embedding | hard one-owner forward + ST backward |

The shared information path is:

```text
contextual/reference-conditioned Q-Former text states
                         |
                         v
                ownership KEY + QASA
                         |
                 soft slot masks
                         |
             assignment switch
                 /       \
      soft weights       token-wise argmax + ST
                 \       /
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

In `soft_shared`, `value_slot_masks` is exactly the soft competitive ownership,
so one raw token may contribute to multiple slots. In `hard_st_exclusive`, each
valid token contributes VALUE to exactly one candidate slot in the forward
pass. Its straight-through backward derivative follows the original soft
ownership, preserving gradients for slot queries and KEY projections. Padding
and special tokens contribute to no slot in either mode.

`value_hard_slot_masks` is returned in both modes so hard-winner diagnostics
remain comparable. In `soft_shared` it is diagnostic-only and is never used by
VALUE pooling; in `hard_st_exclusive` it is the actual forward VALUE support.

The counterfactual teacher quantities `q_full`, `q_minus`, and `slot_effects`
are still computed from the original soft ownership and returned for existing
diagnostics. `slot_effects` is not an input to `slot_mlp` or any other Edit-Slot
latent construction. QASA also remains on its independent FP32 contextual soft
attention path and does not gate VALUE ownership.

Experiment provenance is explicit:

```yaml
slot_value_source: teacher_raw
slot_effect_in_value: false
slot_value_assignment: hard_st_exclusive  # branch default; soft_shared is valid
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

The hard intervention does not guarantee specialization. A giant slot may
still win every valid token, leaving all other VALUE slots exactly empty; this
is an intended falsification mode rather than an error to balance away.

After retraining, if functional Phi effective rank remains near one and both
REPEAT/FULL and MEANxK/FULL remain near one, hard information isolation alone
is not sufficient to produce functional specialization. Stop at that result;
do not add another mechanism inside this ablation. This experiment does not
test multi-error loss, residual pursuit, NULL slots, load balancing, or semantic
supervision.

## Commands

Train the soft control from scratch:

```bash
python src/train.py \
  experiment=taper_e2e \
  dataset.root=data/FashionIQ \
  experiment.model.slot_value_assignment=soft_shared
```

Train the hard-ST intervention from scratch:

```bash
python src/train.py \
  experiment=taper_e2e \
  dataset.root=data/FashionIQ \
  experiment.model.slot_value_assignment=hard_st_exclusive
```

Run a quick QASA hard-partition evaluation:

```bash
python src/evaluate_qasa_inference.py \
  --checkpoint outputs/REPLACE_WITH_RUN/best.pt \
  --dataset-root data/FashionIQ \
  --cache-root features \
  --config conf/experiment/taper_e2e.yaml \
  --slot-value-assignment hard_st_exclusive \
  --device cuda \
  --max-queries-per-category 256 \
  --json-output reports/a3_2_contextual_key_local_value_qasa_quick.json
```

Use `--slot-value-assignment soft_shared` for a soft checkpoint. Evaluation and
P0 loading fail loudly when this selection disagrees with checkpoint
provenance.

Set `--max-queries-per-category 0` for the complete validation set.

Run a quick P0 functional audit:

```bash
python src/audit_taper_merit_p0.py \
  --checkpoint outputs/REPLACE_WITH_RUN/best.pt \
  --dataset-root data/FashionIQ \
  --cache-root features \
  --config conf/experiment/taper_e2e.yaml \
  --slot-value-assignment hard_st_exclusive \
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
  --slot-value-assignment hard_st_exclusive \
  --device cuda \
  --max-queries-per-category 0 \
  --json-output reports/a3_2_hard_private_value_p0_full.json
```
