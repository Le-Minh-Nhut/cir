# Candidate / Selector Diagnostic

- checkpoint: `outputs/r0_ncls_text_strong_aux/best.pt`
- epoch: `18`
- metric: `28.244677186012268`

## Automatic flags

- **WARN — SELECTOR_SLOT_COLLAPSE**: C3 is selected in 75.8% of executed edits.
- **WARN — ORACLE_SLOT_BIAS**: Oracle itself prefers C3 in 62.4% of oracle-execute decisions.
- **WARN — HARMFUL_EXECUTIONS**: 32.2% of executed actions are teacher-negative.
- **WARN — DPP_STARVED_FOR_GOOD_CANDIDATES**: Only 1.10 useful candidates on average; diversity exists but candidate quality is insufficient.

## Slot statistics

| Slot | selected/all | selected/execute | oracle/all | oracle/oracle-execute | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 4.526% | 4.626% | 12.500% | 13.976% | -0.16206 | -0.03029 | 36.422% | 47.619% | 19.828% |
| C1 | 7.759% | 7.930% | 10.560% | 11.807% | -0.11490 | -0.04008 | 40.733% | 33.333% | 21.121% |
| C2 | 11.422% | 11.674% | 10.560% | 11.807% | -0.08503 | -0.00419 | 37.931% | 45.283% | 15.517% |
| C3 | 74.138% | 75.771% | 55.819% | 62.410% | 0.42115 | 0.02476 | 59.698% | 29.070% | 53.448% |
| STOP | 2.155% | 0.000% | 10.560% | 0.000% | 0.00000 | 0.00000 | 0.000% | 0.000% | 0.000% |

## Selector summary

- exact oracle accuracy: `0.5927`
- stop/execute accuracy: `0.8815`
- execute rate: `0.9784`
- harmful / executions: `0.3216`
- harmful / all decisions: `0.3147`
- missed-opportunity STOP rate: `0.0172`
- selected teacher utility: `0.23913`
- oracle teacher utility: `0.39876`
- oracle regret: `0.15963`
- ScoreNet/teacher Pearson: `0.4860`
- selected/oracle agreement: `0.5927`
- selected utility: `0.23913`
- oracle utility: `0.39876`
- regret: `0.15963`
- STOP precision / recall / F1: `0.2000` / `0.0408` / `0.0678`
- harmful executions: `146` / `454` = `0.3216`

### Selector vs oracle confusion

Rows = selector, columns = oracle.

| | C0 | C1 | C2 | C3 | STOP |
|---|---:|---:|---:|---:|---:|
| C0 | 0.0129 | 0.0086 | 0.0108 | 0.0108 | 0.0022 |
| C1 | 0.0086 | 0.0345 | 0.0022 | 0.0216 | 0.0108 |
| C2 | 0.0151 | 0.0065 | 0.0409 | 0.0194 | 0.0323 |
| C3 | 0.0819 | 0.0560 | 0.0474 | 0.5000 | 0.0560 |
| STOP | 0.0065 | 0.0000 | 0.0043 | 0.0065 | 0.0043 |

## Proposal cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.7384 | 0.7924 | 0.0648 |
| C1 | 0.7384 | 1.0000 | 0.4316 | 0.4125 |
| C2 | 0.7924 | 0.4316 | 1.0000 | -0.2228 |
| C3 | 0.0648 | 0.4125 | -0.2228 | 1.0000 |

## Action cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.7235 | 0.7306 | 0.0498 |
| C1 | 0.7235 | 1.0000 | 0.3507 | 0.2518 |
| C2 | 0.7306 | 0.3507 | 1.0000 | -0.0808 |
| C3 | 0.0498 | 0.2518 | -0.0808 | 1.0000 |

## delta_q cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.0919 | 0.1578 | -0.1243 |
| C1 | 0.0919 | 1.0000 | -0.0237 | 0.0284 |
| C2 | 0.1578 | -0.0237 | 1.0000 | -0.1080 |
| C3 | -0.1243 | 0.0284 | -0.1080 | 1.0000 |

## Write-mask Jaccard matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.9958 | 0.3011 | 0.5959 | 0.1020 |
| C1 | 0.3011 | 0.9854 | 0.0831 | 0.0966 |
| C2 | 0.5959 | 0.0831 | 1.0000 | 0.1552 |
| C3 | 0.1020 | 0.0966 | 0.1552 | 1.0000 |

## Candidate-state L2 matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.0000 | 38.2994 | 23.3862 | 185.9624 |
| C1 | 38.2994 | 0.0000 | 45.0809 | 170.3242 |
| C2 | 23.3862 | 45.0809 | 0.0000 | 187.6273 |
| C3 | 185.9624 | 170.3242 | 187.6273 | 0.0000 |

## Learned proposal-query prior cosine

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | -0.0070 | 0.0176 | -0.0072 |
| C1 | -0.0070 | 1.0000 | -0.0781 | 0.0356 |
| C2 | 0.0176 | -0.0781 | 1.0000 | -0.0966 |
| C3 | -0.0072 | 0.0356 | -0.0966 | 1.0000 |

## DPP / candidate quality

- mean useful candidate count: `1.0991`
- rows with >=2 useful candidates: `0.2802`
- positive candidate fraction: `0.4370`

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
- `selected_utility_correct`: mean `0.269509` (n=136)
- `selected_utility_correct_minus_shuffled`: mean `0.422889` (n=136)
- `selected_utility_shuffled`: mean `-0.153380` (n=136)
- `terminal_retrieval_correct_advantage`: mean `1.313929` (n=136)

## Score / teacher calibration

- Pearson: `0.486029`
- bias (score - utility): `0.027243`
- MAE: `0.390277`
- RMSE: `0.667652`
- sign agreement at zero: `0.637931`

| bin | n | teacher mean | score mean | bias |
|---:|---:|---:|---:|---:|
| 0 | 186 | -1.071974 | -0.586554 | 0.485420 |
| 1 | 186 | -0.149250 | 0.015894 | 0.165144 |
| 2 | 186 | -0.046206 | -0.052214 | -0.006009 |
| 3 | 186 | -0.017881 | -0.083270 | -0.065389 |
| 4 | 186 | -0.005923 | -0.120000 | -0.114077 |
| 5 | 186 | -0.000368 | -0.109100 | -0.108732 |
| 6 | 185 | 0.007018 | -0.058195 | -0.065213 |
| 7 | 185 | 0.030539 | 0.071579 | 0.041040 |
| 8 | 185 | 0.145009 | 0.370989 | 0.225980 |
| 9 | 185 | 0.991102 | 0.704329 | -0.286773 |