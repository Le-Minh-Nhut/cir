# Candidate / Selector Diagnostic

- checkpoint: `outputs/2026-09-21/20-02-45/best.pt`
- epoch: `1`
- metric: `14.446642498175304`

## Automatic flags

- **WARN — SELECTOR_SLOT_COLLAPSE**: C3 is selected in 98.1% of executed edits.
- **WARN — SELECTOR_BIAS_NOT_ORACLE_BIAS**: Selector prefers C3, while oracle most often prefers C0.
- **WARN — HARMFUL_EXECUTIONS**: 69.4% of executed actions are teacher-negative.
- **WARN — LOW_EXACT_ORACLE_ACCURACY**: Exact selector-vs-oracle accuracy is only 22.2%.
- **WARN — DPP_STARVED_FOR_GOOD_CANDIDATES**: Only 0.63 useful candidates on average; diversity exists but candidate quality is insufficient.

## Slot statistics

| Slot | selected/all | selected/execute | oracle/all | oracle/oracle-execute | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 1.794% | 1.869% | 36.099% | 40.351% | 0.00685 | 0.00241 | 54.036% | 75.000% | 20.404% |
| C1 | 0.000% | 0.000% | 16.816% | 18.797% | -0.03110 | -0.00378 | 43.049% | nan% | 14.798% |
| C2 | 0.000% | 0.000% | 13.229% | 14.787% | -0.02789 | -0.00493 | 38.117% | nan% | 14.126% |
| C3 | 94.170% | 98.131% | 23.318% | 26.065% | 0.02411 | -0.00599 | 31.614% | 69.286% | 13.677% |
| STOP | 4.036% | 0.000% | 10.538% | 0.000% | 0.00000 | 0.00000 | 0.000% | 0.000% | 0.000% |

## Selector summary

- exact oracle accuracy: `0.2220`
- stop/execute accuracy: `0.8587`
- execute rate: `0.9596`
- harmful / executions: `0.6939`
- harmful / all decisions: `0.6659`
- missed-opportunity STOP rate: `0.0381`
- selected teacher utility: `-0.00801`
- oracle teacher utility: `0.03834`
- oracle regret: `0.04635`
- ScoreNet/teacher Pearson: `-0.0031`
- selected/oracle agreement: `0.2220`
- selected utility: `-0.00801`
- oracle utility: `0.03834`
- regret: `0.04635`
- STOP precision / recall / F1: `0.0556` / `0.0213` / `0.0308`
- harmful executions: `297` / `428` = `0.6939`

### Selector vs oracle confusion

Rows = selector, columns = oracle.

| | C0 | C1 | C2 | C3 | STOP |
|---|---:|---:|---:|---:|---:|
| C0 | 0.0022 | 0.0045 | 0.0045 | 0.0045 | 0.0022 |
| C1 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C2 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| C3 | 0.3430 | 0.1592 | 0.1211 | 0.2175 | 0.1009 |
| STOP | 0.0157 | 0.0045 | 0.0067 | 0.0112 | 0.0022 |

## Proposal cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.6072 | 0.6250 | 0.6231 |
| C1 | 0.6072 | 1.0000 | 0.6036 | 0.6231 |
| C2 | 0.6250 | 0.6036 | 1.0000 | 0.6153 |
| C3 | 0.6231 | 0.6231 | 0.6153 | 1.0000 |

## Action cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.4878 | 0.4312 | 0.0752 |
| C1 | 0.4878 | 1.0000 | 0.5348 | 0.0878 |
| C2 | 0.4312 | 0.5348 | 1.0000 | 0.1224 |
| C3 | 0.0752 | 0.0878 | 0.1224 | 1.0000 |

## delta_q cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.3401 | 0.1350 | -0.6242 |
| C1 | 0.3401 | 1.0000 | 0.1312 | -0.3768 |
| C2 | 0.1350 | 0.1312 | 1.0000 | -0.2026 |
| C3 | -0.6242 | -0.3768 | -0.2026 | 1.0000 |

## Write-mask Jaccard matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.2735 | 0.2074 | 0.6815 |
| C1 | 0.2735 | 1.0000 | 0.1457 | 0.2393 |
| C2 | 0.2074 | 0.1457 | 1.0000 | 0.2481 |
| C3 | 0.6815 | 0.2393 | 0.2481 | 1.0000 |

## Candidate-state L2 matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.0000 | 15.0954 | 17.8324 | 23.5880 |
| C1 | 15.0954 | 0.0000 | 19.1711 | 26.6679 |
| C2 | 17.8324 | 19.1711 | 0.0000 | 22.2532 |
| C3 | 23.5880 | 26.6679 | 22.2532 | 0.0000 |

## Learned proposal-query prior cosine

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | -0.0181 | -0.0026 | 0.0193 |
| C1 | -0.0181 | 1.0000 | -0.0663 | 0.0365 |
| C2 | -0.0026 | -0.0663 | 1.0000 | -0.0259 |
| C3 | 0.0193 | 0.0365 | -0.0259 | 1.0000 |

## DPP / candidate quality

- mean useful candidate count: `0.6300`
- rows with >=2 useful candidates: `0.1659`
- positive candidate fraction: `0.4170`

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
- `selected_utility_correct`: mean `-0.007238` (n=132)
- `selected_utility_correct_minus_shuffled`: mean `-0.000542` (n=132)
- `selected_utility_shuffled`: mean `-0.006696` (n=132)
- `terminal_retrieval_correct_advantage`: mean `-0.004196` (n=132)

## Score / teacher calibration

- Pearson: `-0.003085`
- bias (score - utility): `-0.003931`
- MAE: `0.043082`
- RMSE: `0.068601`
- sign agreement at zero: `0.512892`

| bin | n | teacher mean | score mean | bias |
|---:|---:|---:|---:|---:|
| 0 | 179 | -0.109535 | -0.006667 | 0.102867 |
| 1 | 179 | -0.039098 | -0.009075 | 0.030022 |
| 2 | 179 | -0.015567 | -0.005370 | 0.010197 |
| 3 | 179 | -0.006378 | -0.009060 | -0.002682 |
| 4 | 178 | -0.002132 | -0.006145 | -0.004013 |
| 5 | 178 | -0.000366 | -0.006640 | -0.006274 |
| 6 | 178 | 0.001305 | -0.008264 | -0.009569 |
| 7 | 178 | 0.006757 | -0.005212 | -0.011969 |
| 8 | 178 | 0.023921 | -0.007647 | -0.031568 |
| 9 | 178 | 0.111241 | -0.005959 | -0.117200 |