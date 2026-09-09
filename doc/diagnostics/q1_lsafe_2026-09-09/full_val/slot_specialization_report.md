# IAG-SRME Slot Specialization Diagnostic

> Read-only checkpoint evidence. Gradient exposure and auxiliary alignment at a final checkpoint do not establish temporal causality.

## Cohort

- split: `val`
- processed samples: `6016`
- valid t0 teacher rows: `6016`
- sample fingerprint: `97994df35e76d7166c5dff66d610ba782067b0ce94c37086012f54ce67718c40`

## Functional specialization at t=0

| slot | selected/execution | mean utility | positive | harmful | oracle/execution | unique wins | Shapley |
|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 30.790% | 0.087088 | 50.332% | 49.668% | 30.577% | 27.111% | 0.118739 |
| C1 | 14.338% | 0.020862 | 38.963% | 61.021% | 13.555% | 12.018% | 0.027576 |
| C2 | 46.101% | 0.004968 | 45.479% | 54.521% | 38.170% | 33.843% | 0.206473 |
| C3 | 8.771% | 0.003417 | 38.597% | 61.403% | 17.698% | 15.691% | 0.055194 |

## Coalition oracle

- all-K value: `0.407982`
- C3-only value: `0.09104307729037518`
- all-K minus C3: `0.316939078271389`
- oracle exact-tie fraction (given execute): `0.0`
- mean oracle tie size (given execute): `1.0`
- effective functional K: `3.1792028379820763`
- Shapley efficiency error: `0.000e+00`
- Oracle occupancy in the table is tie-aware; deterministic argmax-tiebroken occupancy remains in JSON for backward comparison only.

### Teacher-batch cluster-bootstrap intervals

- method: `teacher_batch_cluster`
- replicates: `1000`
- confidence: `0.95`
- teacher clusters: `752`
- all-K value CI: `[0.39313096693677313, 0.42164058005734484]`
- all-K minus C3 CI: `[0.30427505245947456, 0.3289591628760892]`
- relative all-K minus C3: `{'estimate': 0.7768454427497816, 'ci_low': 0.7634941776123342, 'ci_high': 0.7905986404777269, 'valid_bootstrap_replicates': 1000}`

## Conditional proposal geometry

PR by slot and functional subset (`n/a` means insufficient support).

| slot | all | utility > 0 | Shapley > 0 | oracle winner |
|---:|---:|---:|---:|---:|
| C0 | 21.448 | 21.598 | 21.598 | 24.053 |
| C1 | 5.067 | 5.277 | 5.277 | 5.404 |
| C2 | 27.529 | 26.185 | 26.185 | 26.110 |
| C3 | 7.308 | 8.602 | 8.602 | 10.145 |

## Current gradient exposure

Unavailable: no gradient batches were processed

## Concept-MIL responsibility

- mean responsibility per slot: `0.2050, 0.2617, 0.3448, 0.1885`
- top-responsibility occupancy: `11.949%, 35.047%, 51.041%, 1.963%`
- mean responsibility entropy: `1.1667`

## Semantic conditioning

- concepts meeting support threshold: `186`
- concepts below threshold: `7139`
- CI method: `teacher_batch_cluster`
- Multi-label conditionals are explanatory, not causal semantic expertise.

## Automatic descriptive flags

- **NON_C3_COMPLEMENTARITY_PRESENT** — Teacher-batch cluster-bootstrap evidence supports positive all-K over C3-only complementarity.

## Interpretation guardrails

- High C3 occupancy is not itself collapse.
- Low proposal PR is not itself a dead-slot diagnosis.
- Shapley and all-K-minus-C3 quantify functional complementarity.
- Functional and semantic CIs resample whole teacher batches because rows share an in-batch negative pool.
- Gradient attribution measures current exposure, not what caused training history.
- Concept-MIL responsibility measures current semantic routing pressure, not causality.
- Automatic flags are descriptive heuristics, not hypothesis tests.

## Limitations

- Static-checkpoint gradients measure current exposure, not temporal causality.
- Semantic conditioning is multi-label and explanatory, not causal expertise.
- Canonical utilities are conditioned on the matched in-batch negative pool.
- No fixed larger negative-bank robustness view is implemented in this patch.
- A 160-row manifest is underpowered for rare-specialist claims; use full VAL or 2000+.
- OLD and STRONG must be run separately with the same manifest and settings.
- Mutual information, stratified permutation tests, and FDR tests are not emitted; deterministic bootstrap confidence intervals are the implemented robustness view.
- The configurable relative-complementarity threshold is a descriptive flagging heuristic, not a universal scientific effect-size threshold.