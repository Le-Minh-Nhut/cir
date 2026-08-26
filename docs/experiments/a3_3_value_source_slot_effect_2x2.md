# A3.3: VALUE source × slot-effect 2×2 ablation

## Question

A3.1 to A3.2 changed both the token representation pooled into each Edit Slot
and the inclusion of the CSMCIR counterfactual slot effect. This experiment
separates those factors while holding soft VALUE assignment and all downstream
training behavior fixed.

| Cell | VALUE source | `slot_effects` in latent | Meaning |
| --- | --- | --- | --- |
| A | contextual Q-Former tokens | ON | A3.1-like slot construction |
| B | raw teacher word embeddings | ON | isolates replacement of contextual VALUE while retaining effects |
| C | contextual Q-Former tokens | OFF | isolates removal of the effect input while retaining contextual VALUE |
| D | raw teacher word embeddings | OFF | A3.2-soft-like slot construction |

Cell A is described as A3.1-like, not bit-identical to historical A3.1, because
the branch contains later shared changes. All four cells use
`slot_value_assignment=soft_shared`.

## Fixed dataflow

Contextual/reference-conditioned `text_states` always provide the KEY for
competitive ownership and the input to QASA. The VALUE source is selected only
after ownership. `teacher_raw` means the cached raw CSMCIR word embeddings before
Q-Former contextualization; `contextual` means the Q-Former token states.

The counterfactual signal is unchanged:

```text
slot_effects = q_full.unsqueeze(1) - q_minus
```

When enabled it is concatenated with pooled `slot_semantics`; when disabled the
MLP receives only `slot_semantics`. The Executor, QASA, teacher, cache, retrieval
loss, optimizer, schedule, and FashionIQ evaluation protocol are unchanged.

## Comparisons

- A → B: replace contextual VALUE with raw/pre-Q-Former VALUE while retaining effects.
- A → C: remove effects while retaining contextual VALUE.
- B → D: remove effects under raw VALUE.
- C → D: replace contextual VALUE with raw VALUE when effects are off.
- Interaction: `(D - C) - (B - A)`.

Use the same seed and budget for every cell. One seed is not sufficient for a
strong interaction claim; repeat across seeds if resources permit.

## Scope

This ablation adds no anti-collapse mechanism, balancing, NULL slot, new loss,
Entmax, Sinkhorn/OT, residual pursuit, or semantic supervision. Diagnostic hard
argmax partitions remain available in `soft_shared`, but they are not the actual
soft VALUE support.

## Training overrides

```bash
# A: contextual + effect ON
python src/train.py experiment=taper_e2e \
  experiment.model.slot_value_assignment=soft_shared \
  experiment.model.slot_value_source=contextual \
  experiment.model.slot_effect_in_value=true

# B: teacher_raw + effect ON
python src/train.py experiment=taper_e2e \
  experiment.model.slot_value_assignment=soft_shared \
  experiment.model.slot_value_source=teacher_raw \
  experiment.model.slot_effect_in_value=true

# C: contextual + effect OFF
python src/train.py experiment=taper_e2e \
  experiment.model.slot_value_assignment=soft_shared \
  experiment.model.slot_value_source=contextual \
  experiment.model.slot_effect_in_value=false

# D: teacher_raw + effect OFF (branch default)
python src/train.py experiment=taper_e2e \
  experiment.model.slot_value_assignment=soft_shared \
  experiment.model.slot_value_source=teacher_raw \
  experiment.model.slot_effect_in_value=false
```

Train every cell from scratch. Evaluation and audit commands must pass matching
`--slot-value-source`, `--slot-effect-in-value`, and
`--slot-value-assignment` values; checkpoint provenance mismatches fail loudly.
