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
| C0 | 20.731% | 0.404659 | 75.399% | 24.601% | 22.482% | 17.104% | 0.122873 |
| C1 | 25.013% | 0.404772 | 75.382% | 24.618% | 24.962% | 18.983% | 0.122905 |
| C2 | 19.745% | 0.405597 | 75.515% | 24.485% | 30.151% | 22.922% | 0.123161 |
| C3 | 34.511% | 0.405082 | 75.399% | 24.584% | 22.406% | 17.038% | 0.122825 |

## Coalition oracle

- all-K value: `0.491764`
- C3-only value: `0.4860452323200855`
- all-K minus C3: `0.005718741030927621`
- oracle exact-tie fraction (given execute): `0.00043696744592527855`
- mean oracle tie size (given execute): `1.0004369674459253`
- effective functional K: `3.99999775954226`
- Shapley efficiency error: `5.551e-17`
- Oracle occupancy in the table is tie-aware; deterministic argmax-tiebroken occupancy remains in JSON for backward comparison only.

### Teacher-batch cluster-bootstrap intervals

- method: `teacher_batch_cluster`
- replicates: `1000`
- confidence: `0.95`
- teacher clusters: `752`
- all-K value CI: `[0.4756220716091388, 0.5089099661060709]`
- all-K minus C3 CI: `[0.005414974731095946, 0.006021841124512632]`
- relative all-K minus C3: `{'estimate': 0.011629036165375368, 'ci_low': 0.011014807267508177, 'ci_high': 0.01225589309131103, 'valid_bootstrap_replicates': 1000}`

## Conditional proposal geometry

PR by slot and functional subset (`n/a` means insufficient support).

| slot | all | utility > 0 | Shapley > 0 | oracle winner |
|---:|---:|---:|---:|---:|
| C0 | 24.896 | 23.672 | 23.672 | 23.780 |
| C1 | 24.976 | 23.731 | 23.731 | 22.568 |
| C2 | 24.915 | 23.690 | 23.690 | 23.330 |
| C3 | 24.983 | 23.754 | 23.754 | 23.305 |

## Current gradient exposure

Full-objective gradient energy with respect to t0 proposals:

| loss | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| terminal | 22.298% | 29.600% | 17.730% | 30.371% |
| concept | 25.786% | 27.529% | 25.172% | 21.512% |
| bind | 24.960% | 25.623% | 24.253% | 25.164% |
| dpp | 9.380% | 22.656% | 7.439% | 60.525% |
| pair | 0.000% | 0.000% | 0.000% | 0.000% |
| gain | 0.000% | 0.000% | 0.000% | 0.000% |

Pair/gain upstream-detach controls:

- `pair`: passed=`True`, max |grad|=`0.000e+00`
- `gain`: passed=`True`, max |grad|=`0.000e+00`

This is the full configured component gradient with respect to the t0 proposal tensor, not a local t0-only loss gradient; downstream recurrent effects may contribute.
Parameter-level proposal-query observations each represent one diagnostic batch gradient, not one sample.

## Concept-MIL responsibility

- mean responsibility per slot: `0.2499, 0.2495, 0.2503, 0.2503`
- top-responsibility occupancy: `25.770%, 19.852%, 29.843%, 24.534%`
- mean responsibility entropy: `1.3862`

## Semantic conditioning

- concepts meeting support threshold: `186`
- concepts below threshold: `7139`
- CI method: `teacher_batch_cluster`
- Multi-label conditionals are explanatory, not causal semantic expertise.

## Automatic descriptive flags

- **NON_C3_COMPLEMENTARITY_PRESENT** — Teacher-batch cluster-bootstrap evidence supports positive all-K over C3-only complementarity.
- **NEGATIVE_TASK_AUX_ALIGNMENT** — At least half of valid task/auxiliary gradient cosines are negative.

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