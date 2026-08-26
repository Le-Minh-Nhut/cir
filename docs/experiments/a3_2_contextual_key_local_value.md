# A3.2: Contextual key, local/private value

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
                    slot masks
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

The counterfactual teacher quantities `q_full`, `q_minus`, and `slot_effects`
are still computed and returned for existing diagnostics. `slot_effects` is not
an input to `slot_mlp` or any other Edit-Slot latent construction.

Experiment provenance is explicit:

```yaml
slot_value_source: teacher_raw
slot_effect_in_value: false
```

The first `slot_mlp` layer therefore has `teacher_text_dim` inputs (768 in the
experiment config), rather than `text_dim + teacher_query_dim` inputs. A3.1
TAPER checkpoints are shape-incompatible and must not be reused. The frozen
CSMCIR teacher checkpoint and the existing aligned CSMCIR image/text caches are
reused unchanged.

## Preserved controls

The contextual KEY, competitive ownership, content mask, QASA algorithm and
thresholds, frozen teacher, Executor, Router, Primitive Bank, retrieval loss,
optimizer, schedule, captions, and FashionIQ evaluation protocol are unchanged.
No new loss, supervision, regularizer, routing algorithm, or semantic label is
introduced.

## Falsification criterion

After retraining, if functional Phi effective rank remains near one and both
REPEAT/FULL and MEANxK/FULL remain near one, isolating local token values is not
sufficient to produce functional specialization. That result would motivate a
later multi-error functional-ownership experiment; it is not corrected inside
this ablation.

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

The requested `src/audit_taper_merit_p0.py` is not present in this branch or
its parent. The older `src/forensic_taper_a3.py` is also incompatible with the
parent's current TAPER/QASA API (`slot_gates`/`null_probs` assumptions), so it
is not presented as a valid P0 command. A full P0 audit remains blocked until
the actual current audit implementation is supplied or restored.
