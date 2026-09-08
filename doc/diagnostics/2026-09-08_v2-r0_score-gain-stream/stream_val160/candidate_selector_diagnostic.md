# Candidate / Selector Diagnostic

- checkpoint: `outputs/r0_ncls_text_strong_aux_score_gain_stream/score_gain_stream.pt`
- epoch: `1`
- metric: `None`

## Automatic flags

- **WARN — SELECTOR_SLOT_COLLAPSE**: C3 is selected in 74.5% of executed edits.
- **WARN — HARMFUL_EXECUTIONS**: 32.5% of executed actions are teacher-negative.
- **WARN — DPP_STARVED_FOR_GOOD_CANDIDATES**: Only 1.08 useful candidates on average; diversity exists but candidate quality is insufficient.

## Slot statistics

| Slot | selected/all | selected/execute | oracle/all | oracle/oracle-execute | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 3.791% | 4.124% | 12.796% | 14.477% | -0.06805 | -0.03267 | 37.204% | 37.500% | 19.905% |
| C1 | 10.427% | 11.340% | 11.137% | 12.601% | -0.05832 | -0.04798 | 40.284% | 34.091% | 21.564% |
| C2 | 9.242% | 10.052% | 11.611% | 13.137% | -0.02081 | -0.00343 | 36.493% | 46.154% | 16.114% |
| C3 | 68.483% | 74.485% | 52.844% | 59.786% | 0.12032 | -0.03623 | 56.635% | 30.104% | 50.237% |
| STOP | 8.057% | 0.000% | 11.611% | 0.000% | 0.00000 | 0.00000 | 0.000% | 0.000% | 0.000% |

## Selector summary

- exact oracle accuracy: `0.5592`
- stop/execute accuracy: `0.8365`
- execute rate: `0.9194`
- harmful / executions: `0.3247`
- harmful / all decisions: `0.2986`
- missed-opportunity STOP rate: `0.0640`
- selected teacher utility: `0.22290`
- oracle teacher utility: `0.38140`
- oracle regret: `0.15850`
- ScoreNet/teacher Pearson: `0.5132`
- selected/oracle agreement: `0.5592`
- selected utility: `0.22290`
- oracle utility: `0.38140`
- regret: `0.15850`
- STOP precision / recall / F1: `0.2059` / `0.1429` / `0.1687`
- harmful executions: `126` / `388` = `0.3247`

### Selector vs oracle confusion

Rows = selector, columns = oracle.

| | C0 | C1 | C2 | C3 | STOP |
|---|---:|---:|---:|---:|---:|
| C0 | 0.0071 | 0.0071 | 0.0142 | 0.0071 | 0.0024 |
| C1 | 0.0142 | 0.0427 | 0.0000 | 0.0308 | 0.0166 |
| C2 | 0.0166 | 0.0024 | 0.0355 | 0.0142 | 0.0237 |
| C3 | 0.0687 | 0.0545 | 0.0474 | 0.4573 | 0.0569 |
| STOP | 0.0213 | 0.0047 | 0.0190 | 0.0190 | 0.0166 |

## Proposal cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.7375 | 0.7935 | 0.0730 |
| C1 | 0.7375 | 1.0000 | 0.4318 | 0.4226 |
| C2 | 0.7935 | 0.4318 | 1.0000 | -0.2148 |
| C3 | 0.0730 | 0.4226 | -0.2148 | 1.0000 |

## Action cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.7177 | 0.7373 | 0.0575 |
| C1 | 0.7177 | 1.0000 | 0.3533 | 0.2610 |
| C2 | 0.7373 | 0.3533 | 1.0000 | -0.0741 |
| C3 | 0.0575 | 0.2610 | -0.0741 | 1.0000 |

## delta_q cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.0979 | 0.1610 | -0.1195 |
| C1 | 0.0979 | 1.0000 | -0.0291 | 0.0279 |
| C2 | 0.1610 | -0.0291 | 1.0000 | -0.1092 |
| C3 | -0.1195 | 0.0279 | -0.1092 | 1.0000 |

## Write-mask Jaccard matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.9958 | 0.2912 | 0.5965 | 0.1021 |
| C1 | 0.2912 | 0.9854 | 0.0782 | 0.0986 |
| C2 | 0.5965 | 0.0782 | 1.0000 | 0.1542 |
| C3 | 0.1021 | 0.0986 | 0.1542 | 1.0000 |

## Candidate-state L2 matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.0000 | 38.8757 | 23.7703 | 184.2542 |
| C1 | 38.8757 | 0.0000 | 45.7794 | 168.4245 |
| C2 | 23.7703 | 45.7794 | 0.0000 | 186.0214 |
| C3 | 184.2542 | 168.4245 | 186.0214 | 0.0000 |

## Learned proposal-query prior cosine

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | -0.0070 | 0.0176 | -0.0072 |
| C1 | -0.0070 | 1.0000 | -0.0781 | 0.0356 |
| C2 | 0.0176 | -0.0781 | 1.0000 | -0.0966 |
| C3 | -0.0072 | 0.0356 | -0.0966 | 1.0000 |

## DPP / candidate quality

- mean useful candidate count: `1.0782`
- rows with >=2 useful candidates: `0.2773`
- positive candidate fraction: `0.4265`

> Target-derived teacher utility is diagnostic only and is never an inference input.

## Correct vs shuffled-caption sensitivity

- `best_candidate_utility_*` compares raw best candidates and ignores STOP.
- `oracle_policy_utility_*` applies the checkpoint's `epsilon_stop`; its value is zero when the oracle chooses STOP.

- `actions_cosine`: mean `0.639044` (n=544)
- `actions_norm_difference`: mean `5.862059` (n=544)
- `best_candidate_utility_correct_minus_shuffled`: mean `0.037780` (n=136)
- `candidate_utility_correct_minus_shuffled`: mean `-0.041517` (n=544)
- `delta_q_cosine`: mean `0.578354` (n=544)
- `delta_q_norm_difference`: mean `0.395476` (n=544)
- `oracle_policy_utility_correct_minus_shuffled`: mean `0.040975` (n=136)
- `proposals_cosine`: mean `0.660204` (n=544)
- `proposals_norm_difference`: mean `9.307417` (n=544)
- `selected_utility_correct`: mean `0.291850` (n=136)
- `selected_utility_correct_minus_shuffled`: mean `0.454451` (n=136)
- `selected_utility_shuffled`: mean `-0.162601` (n=136)
- `terminal_retrieval_correct_advantage`: mean `1.030368` (n=136)

## Score / teacher calibration

- Pearson: `0.513213`
- bias (score - utility): `0.023361`
- MAE: `0.262946`
- RMSE: `0.566371`
- sign agreement at zero: `0.617891`

| bin | n | teacher mean | score mean | bias |
|---:|---:|---:|---:|---:|
| 0 | 169 | -1.170538 | -0.617597 | 0.552941 |
| 1 | 169 | -0.182302 | -0.013092 | 0.169210 |
| 2 | 169 | -0.054227 | -0.003194 | 0.051033 |
| 3 | 169 | -0.020261 | -0.002920 | 0.017341 |
| 4 | 169 | -0.006779 | -0.012714 | -0.005935 |
| 5 | 169 | -0.000834 | -0.011002 | -0.010168 |
| 6 | 169 | 0.005543 | 0.016492 | 0.010949 |
| 7 | 169 | 0.028298 | 0.055164 | 0.026867 |
| 8 | 168 | 0.142643 | 0.162718 | 0.020075 |
| 9 | 168 | 0.964580 | 0.362156 | -0.602424 |