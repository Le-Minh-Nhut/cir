# Candidate / Selector Diagnostic

- checkpoint: `outputs/2026-09-21/residual_ln_score_stream/score_stream.pt`
- epoch: `3`
- metric: `None`

## Automatic flags

- **WARN — SELECTOR_SLOT_COLLAPSE**: C0 is selected in 78.5% of executed edits.
- **WARN — SELECTOR_BIAS_NOT_ORACLE_BIAS**: Selector prefers C0, while oracle most often prefers C3.
- **WARN — HARMFUL_EXECUTIONS**: 47.9% of executed actions are teacher-negative.
- **WARN — LOW_EXACT_ORACLE_ACCURACY**: Exact selector-vs-oracle accuracy is only 23.0%.
- **WARN — DPP_STARVED_FOR_GOOD_CANDIDATES**: Only 0.56 useful candidates on average; diversity exists but candidate quality is insufficient.

## Slot statistics

| Slot | selected/all | selected/execute | oracle/all | oracle/oracle-execute | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 36.965% | 78.512% | 26.848% | 30.531% | -0.00137 | -0.00034 | 39.300% | 54.737% | 16.732% |
| C1 | 0.389% | 0.826% | 15.564% | 17.699% | -0.01361 | -0.00472 | 33.463% | 0.000% | 14.397% |
| C2 | 0.778% | 1.653% | 15.953% | 18.142% | -0.02408 | -0.00869 | 32.685% | 0.000% | 13.619% |
| C3 | 8.949% | 19.008% | 29.572% | 33.628% | -0.00879 | -0.00448 | 43.191% | 26.087% | 10.895% |
| STOP | 52.918% | 0.000% | 12.062% | 0.000% | 0.00000 | 0.00000 | 0.000% | 0.000% | 0.000% |

## Selector summary

- exact oracle accuracy: `0.2296`
- stop/execute accuracy: `0.4825`
- execute rate: `0.4708`
- harmful / executions: `0.4793`
- harmful / all decisions: `0.2257`
- missed-opportunity STOP rate: `0.4630`
- selected teacher utility: `0.00317`
- oracle teacher utility: `0.03794`
- oracle regret: `0.03477`
- ScoreNet/teacher Pearson: `0.1077`
- selected/oracle agreement: `0.2296`
- selected utility: `0.00317`
- oracle utility: `0.03794`
- regret: `0.03477`
- STOP precision / recall / F1: `0.1250` / `0.5484` / `0.2036`
- harmful executions: `58` / `121` = `0.4793`

### Selector vs oracle confusion

Rows = selector, columns = oracle.

| | C0 | C1 | C2 | C3 | STOP |
|---|---:|---:|---:|---:|---:|
| C0 | 0.1206 | 0.0467 | 0.0584 | 0.0895 | 0.0545 |
| C1 | 0.0000 | 0.0000 | 0.0039 | 0.0000 | 0.0000 |
| C2 | 0.0000 | 0.0039 | 0.0039 | 0.0000 | 0.0000 |
| C3 | 0.0272 | 0.0156 | 0.0078 | 0.0389 | 0.0000 |
| STOP | 0.1206 | 0.0895 | 0.0856 | 0.1673 | 0.0661 |

## Proposal cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.6066 | 0.6248 | 0.6236 |
| C1 | 0.6066 | 1.0000 | 0.6035 | 0.6236 |
| C2 | 0.6248 | 0.6035 | 1.0000 | 0.6161 |
| C3 | 0.6236 | 0.6236 | 0.6161 | 1.0000 |

## Action cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.4821 | 0.4252 | 0.0785 |
| C1 | 0.4821 | 1.0000 | 0.5324 | 0.0869 |
| C2 | 0.4252 | 0.5324 | 1.0000 | 0.1239 |
| C3 | 0.0785 | 0.0869 | 0.1239 | 1.0000 |

## delta_q cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.1819 | -0.0395 | -0.6694 |
| C1 | 0.1819 | 1.0000 | -0.0217 | -0.1676 |
| C2 | -0.0395 | -0.0217 | 1.0000 | -0.0501 |
| C3 | -0.6694 | -0.1676 | -0.0501 | 1.0000 |

## Write-mask Jaccard matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.1889 | 0.1921 | 0.6892 |
| C1 | 0.1889 | 1.0000 | 0.1394 | 0.1785 |
| C2 | 0.1921 | 0.1394 | 1.0000 | 0.2257 |
| C3 | 0.6892 | 0.1785 | 0.2257 | 1.0000 |

## Candidate-state L2 matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.0000 | 14.7399 | 16.4921 | 22.6479 |
| C1 | 14.7399 | 0.0000 | 17.7924 | 25.4750 |
| C2 | 16.4921 | 17.7924 | 0.0000 | 20.8194 |
| C3 | 22.6479 | 25.4750 | 20.8194 | 0.0000 |

## Learned proposal-query prior cosine

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | -0.0181 | -0.0026 | 0.0193 |
| C1 | -0.0181 | 1.0000 | -0.0663 | 0.0365 |
| C2 | -0.0026 | -0.0663 | 1.0000 | -0.0259 |
| C3 | 0.0193 | 0.0365 | -0.0259 | 1.0000 |

## DPP / candidate quality

- mean useful candidate count: `0.5564`
- rows with >=2 useful candidates: `0.1479`
- positive candidate fraction: `0.3716`

> Target-derived teacher utility is diagnostic only and is never an inference input.

## Correct vs shuffled-caption sensitivity

- `best_candidate_utility_*` compares raw best candidates and ignores STOP.
- `oracle_policy_utility_*` applies the checkpoint's `epsilon_stop`; its value is zero when the oracle chooses STOP.

- `actions_cosine`: mean `0.993173` (n=528)
- `actions_norm_difference`: mean `1.352409` (n=528)
- `best_candidate_utility_correct_minus_shuffled`: mean `0.000108` (n=132)
- `candidate_utility_correct_minus_shuffled`: mean `0.000098` (n=528)
- `delta_q_cosine`: mean `0.999687` (n=528)
- `delta_q_norm_difference`: mean `0.003994` (n=528)
- `oracle_policy_utility_correct_minus_shuffled`: mean `0.000128` (n=132)
- `proposals_cosine`: mean `0.995140` (n=528)
- `proposals_norm_difference`: mean `2.660934` (n=528)
- `selected_utility_correct`: mean `0.004960` (n=132)
- `selected_utility_correct_minus_shuffled`: mean `0.003112` (n=132)
- `selected_utility_shuffled`: mean `0.001848` (n=132)
- `terminal_retrieval_correct_advantage`: mean `0.002672` (n=132)

## Score / teacher calibration

- Pearson: `0.107671`
- bias (score - utility): `-0.007408`
- MAE: `0.034699`
- RMSE: `0.069721`
- sign agreement at zero: `0.634241`

| bin | n | teacher mean | score mean | bias |
|---:|---:|---:|---:|---:|
| 0 | 103 | -0.117453 | -0.015791 | 0.101662 |
| 1 | 103 | -0.040377 | -0.018270 | 0.022107 |
| 2 | 103 | -0.015455 | -0.013143 | 0.002312 |
| 3 | 103 | -0.006115 | -0.013497 | -0.007383 |
| 4 | 103 | -0.002623 | -0.010360 | -0.007737 |
| 5 | 103 | -0.000732 | -0.010833 | -0.010101 |
| 6 | 103 | 0.000290 | -0.006015 | -0.006304 |
| 7 | 103 | 0.003816 | -0.008875 | -0.012690 |
| 8 | 102 | 0.018565 | -0.011757 | -0.030323 |
| 9 | 102 | 0.115913 | -0.011089 | -0.127002 |