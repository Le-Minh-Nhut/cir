# Candidate / Selector Diagnostic

- checkpoint: `outputs/r0_ncls_text_strong_aux/best.pt`
- epoch: `18`
- metric: `28.244677186012268`

## Automatic flags

- **WARN — SELECTOR_SLOT_COLLAPSE**: C3 is selected 72.7% of decisions.
- **WARN — ORACLE_SLOT_BIAS**: Oracle itself prefers C3 63.2% of decisions.
- **WARN — HARMFUL_EXECUTIONS**: 25.2% of selected executions are teacher-negative.
- **WARN — DPP_STARVED_FOR_GOOD_CANDIDATES**: Only 0.99 useful candidates on average; diversity exists but candidate quality is insufficient.

## Slot statistics

| Slot | selected % | oracle % | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |
|---|---:|---:|---:|---:|---:|---:|---:|
| C0 | 4.228% | 7.400% | -0.15812 | -0.03418 | 32.770% | 45.000% | 13.108% |
| C1 | 10.571% | 10.994% | -0.10797 | -0.02950 | 45.877% | 40.000% | 18.816% |
| C2 | 11.628% | 8.879% | -0.08015 | -0.00773 | 35.307% | 58.182% | 9.725% |
| C3 | 72.727% | 63.214% | 0.33910 | 0.03119 | 67.230% | 16.860% | 57.082% |
| STOP | 0.846% | 9.514% | 0.00000 | 0.00000 | 0.000% | 0.000% | 0.000% |

## Selector summary

- exact oracle accuracy: `0.6850`
- stop/execute accuracy: `0.9006`
- harmful execution rate: `0.2516`
- missed-opportunity STOP rate: `0.0063`
- selected teacher utility: `0.23775`
- oracle teacher utility: `0.30610`
- oracle regret: `0.06836`
- ScoreNet/teacher Pearson: `0.6124`

### Selector vs oracle confusion

Rows = selector, columns = oracle.

| | C0 | C1 | C2 | C3 | STOP |
|---|---:|---:|---:|---:|---:|
| C0 | 0.0148 | 0.0063 | 0.0063 | 0.0063 | 0.0085 |
| C1 | 0.0085 | 0.0444 | 0.0063 | 0.0211 | 0.0254 |
| C2 | 0.0169 | 0.0169 | 0.0338 | 0.0148 | 0.0338 |
| C3 | 0.0338 | 0.0381 | 0.0402 | 0.5899 | 0.0254 |
| STOP | 0.0000 | 0.0042 | 0.0021 | 0.0000 | 0.0021 |

## Proposal cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.7392 | 0.7900 | 0.0807 |
| C1 | 0.7392 | 1.0000 | 0.4305 | 0.4240 |
| C2 | 0.7900 | 0.4305 | 1.0000 | -0.2079 |
| C3 | 0.0807 | 0.4240 | -0.2079 | 1.0000 |

## Action cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.7334 | 0.7292 | 0.0625 |
| C1 | 0.7334 | 1.0000 | 0.3635 | 0.2598 |
| C2 | 0.7292 | 0.3635 | 1.0000 | -0.0676 |
| C3 | 0.0625 | 0.2598 | -0.0676 | 1.0000 |

## delta_q cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.1013 | 0.1694 | -0.1101 |
| C1 | 0.1013 | 1.0000 | -0.0235 | 0.0589 |
| C2 | 0.1694 | -0.0235 | 1.0000 | -0.0958 |
| C3 | -0.1101 | 0.0589 | -0.0958 | 1.0000 |

## Write-mask Jaccard matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.9979 | 0.2740 | 0.6187 | 0.1125 |
| C1 | 0.2740 | 0.9771 | 0.0883 | 0.0948 |
| C2 | 0.6187 | 0.0883 | 1.0000 | 0.1650 |
| C3 | 0.1125 | 0.0948 | 0.1650 | 0.9958 |

## Candidate-state L2 matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.0000 | 37.5616 | 22.6337 | 180.5305 |
| C1 | 37.5616 | 0.0000 | 43.3593 | 164.3354 |
| C2 | 22.6337 | 43.3593 | 0.0000 | 181.6286 |
| C3 | 180.5305 | 164.3354 | 181.6286 | 0.0000 |

## Learned proposal-query prior cosine

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | -0.0070 | 0.0176 | -0.0072 |
| C1 | -0.0070 | 1.0000 | -0.0781 | 0.0356 |
| C2 | 0.0176 | -0.0781 | 1.0000 | -0.0966 |
| C3 | -0.0072 | 0.0356 | -0.0966 | 1.0000 |

## DPP / candidate quality

- mean useful candidate count: `0.9873`
- rows with >=2 useful candidates: `0.2114`
- positive candidate fraction: `0.4530`

> Target-derived teacher utility is diagnostic only and is never an inference input.