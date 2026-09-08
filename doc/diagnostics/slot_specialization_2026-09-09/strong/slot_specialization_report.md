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
| C0 | 16.886% | -0.077250 | 32.929% | 67.071% | 17.853% | 14.096% | 0.045129 |
| C1 | 25.568% | -0.102786 | 41.489% | 58.511% | 25.242% | 19.930% | 0.071594 |
| C2 | 29.380% | -0.009679 | 32.430% | 67.570% | 19.453% | 15.359% | 0.035699 |
| C3 | 28.166% | -0.384116 | 35.588% | 64.412% | 37.453% | 29.571% | 0.300617 |

## Coalition oracle

- all-K value: `0.453040`
- C3-only value: `0.3477715141476786`
- all-K minus C3: `0.10526824346248137`
- oracle exact-tie fraction (given execute): `0.0`
- mean oracle tie size (given execute): `1.0`
- effective functional K: `2.7011704275929604`
- Shapley efficiency error: `0.000e+00`
- Oracle occupancy in the table is tie-aware; deterministic argmax-tiebroken occupancy remains in JSON for backward comparison only.

### Teacher-batch cluster-bootstrap intervals

- method: `teacher_batch_cluster`
- replicates: `1000`
- confidence: `0.95`
- teacher clusters: `752`
- all-K value CI: `[0.4333718761989966, 0.4711786693898327]`
- all-K minus C3 CI: `[0.1002927857322341, 0.11001699540486681]`
- relative all-K minus C3: `{'estimate': 0.23235983529963067, 'ci_low': 0.21927081047529, 'ci_high': 0.24476137340421422, 'valid_bootstrap_replicates': 1000}`

## Conditional proposal geometry

PR by slot and functional subset (`n/a` means insufficient support).

| slot | all | utility > 0 | Shapley > 0 | oracle winner |
|---:|---:|---:|---:|---:|
| C0 | 4.640 | 6.937 | 6.937 | 6.963 |
| C1 | 8.278 | 10.148 | 10.148 | 11.436 |
| C2 | 2.863 | 3.482 | 3.482 | 4.529 |
| C3 | 27.938 | 27.728 | 27.728 | 27.851 |

## Current gradient exposure

Full-objective gradient energy with respect to t0 proposals:

| loss | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| terminal | 5.075% | 14.230% | 16.712% | 63.982% |
| concept | 18.108% | 12.122% | 18.041% | 51.728% |
| bind | 33.110% | 15.876% | 39.857% | 11.157% |
| dpp | 10.306% | 2.155% | 82.162% | 5.378% |
| pair | 0.000% | 0.000% | 0.000% | 0.000% |
| gain | 0.000% | 0.000% | 0.000% | 0.000% |

Pair/gain upstream-detach controls:

- `pair`: passed=`True`, max |grad|=`0.000e+00`
- `gain`: passed=`True`, max |grad|=`0.000e+00`

This is the full configured component gradient with respect to the t0 proposal tensor, not a local t0-only loss gradient; downstream recurrent effects may contribute.
Parameter-level proposal-query observations each represent one diagnostic batch gradient, not one sample.

## Concept-MIL responsibility

- mean responsibility per slot: `0.2254, 0.1761, 0.2056, 0.3929`
- top-responsibility occupancy: `16.867%, 0.865%, 18.075%, 64.194%`
- mean responsibility entropy: `1.1806`

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