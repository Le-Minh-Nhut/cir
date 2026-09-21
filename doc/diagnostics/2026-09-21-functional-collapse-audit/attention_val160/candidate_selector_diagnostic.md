# Candidate / Selector Diagnostic

- checkpoint: `outputs/2026-09-21/18-26-04/best.pt`
- epoch: `1`
- metric: `15.13077262789011`

## Automatic flags

- **WARN — HARMFUL_EXECUTIONS**: 61.7% of executed actions are teacher-negative.
- **WARN — LOW_EXACT_ORACLE_ACCURACY**: Exact selector-vs-oracle accuracy is only 25.4%.
- **WARN — DPP_STARVED_FOR_GOOD_CANDIDATES**: Only 0.00 useful candidates on average; diversity exists but candidate quality is insufficient.
- **FAIL — FUNCTIONAL_COLLAPSE**: Mean sibling delta_q cosine is 0.9926.

## Slot statistics

| Slot | selected/all | selected/execute | oracle/all | oracle/oracle-execute | mean score | mean teacher utility | positive utility % | harmful when selected % | useful-for-DPP % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C0 | 30.241% | 44.898% | 10.309% | 25.424% | 0.00072 | 0.00028 | 37.457% | 57.955% | 0.000% |
| C1 | 2.749% | 4.082% | 9.966% | 24.576% | 0.00064 | 0.00029 | 38.832% | 12.500% | 0.000% |
| C2 | 13.058% | 19.388% | 10.653% | 26.271% | 0.00069 | 0.00029 | 38.144% | 76.316% | 0.000% |
| C3 | 21.306% | 31.633% | 9.622% | 23.729% | 0.00070 | 0.00028 | 37.801% | 64.516% | 0.000% |
| STOP | 32.646% | 0.000% | 59.450% | 0.000% | 0.00000 | 0.00000 | 0.000% | 0.000% | 0.000% |

## Selector summary

- exact oracle accuracy: `0.2543`
- stop/execute accuracy: `0.4914`
- execute rate: `0.6735`
- harmful / executions: `0.6173`
- harmful / all decisions: `0.4158`
- missed-opportunity STOP rate: `0.1203`
- selected teacher utility: `0.00018`
- oracle teacher utility: `0.00084`
- oracle regret: `0.00066`
- ScoreNet/teacher Pearson: `-0.0334`
- selected/oracle agreement: `0.2543`
- selected utility: `0.00018`
- oracle utility: `0.00084`
- regret: `0.00066`
- STOP precision / recall / F1: `0.6316` / `0.3468` / `0.4478`
- harmful executions: `121` / `196` = `0.6173`

### Selector vs oracle confusion

Rows = selector, columns = oracle.

| | C0 | C1 | C2 | C3 | STOP |
|---|---:|---:|---:|---:|---:|
| C0 | 0.0309 | 0.0275 | 0.0515 | 0.0275 | 0.1649 |
| C1 | 0.0069 | 0.0000 | 0.0103 | 0.0069 | 0.0034 |
| C2 | 0.0137 | 0.0103 | 0.0034 | 0.0069 | 0.0962 |
| C3 | 0.0206 | 0.0344 | 0.0206 | 0.0137 | 0.1237 |
| STOP | 0.0309 | 0.0275 | 0.0206 | 0.0412 | 0.2062 |

## Proposal cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.9997 | 0.9998 | 0.9998 |
| C1 | 0.9997 | 1.0000 | 0.9998 | 0.9998 |
| C2 | 0.9998 | 0.9998 | 1.0000 | 0.9998 |
| C3 | 0.9998 | 0.9998 | 0.9998 | 1.0000 |

## Action cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| C3 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## delta_q cosine matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.9925 | 0.9925 | 0.9926 |
| C1 | 0.9925 | 1.0000 | 0.9928 | 0.9927 |
| C2 | 0.9925 | 0.9928 | 1.0000 | 0.9928 |
| C3 | 0.9926 | 0.9927 | 0.9928 | 1.0000 |

## Write-mask Jaccard matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | 0.9873 | 0.9861 | 0.9853 |
| C1 | 0.9873 | 1.0000 | 0.9950 | 0.9765 |
| C2 | 0.9861 | 0.9950 | 1.0000 | 0.9725 |
| C3 | 0.9853 | 0.9765 | 0.9725 | 1.0000 |

## Candidate-state L2 matrix

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 0.0000 | 0.0036 | 0.0029 | 0.0031 |
| C1 | 0.0036 | 0.0000 | 0.0022 | 0.0033 |
| C2 | 0.0029 | 0.0022 | 0.0000 | 0.0035 |
| C3 | 0.0031 | 0.0033 | 0.0035 | 0.0000 |

## Learned proposal-query prior cosine

| | C0 | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| C0 | 1.0000 | -0.0159 | 0.0002 | 0.0214 |
| C1 | -0.0159 | 1.0000 | -0.0649 | 0.0468 |
| C2 | 0.0002 | -0.0649 | 1.0000 | -0.0224 |
| C3 | 0.0214 | 0.0468 | -0.0224 | 1.0000 |

## DPP / candidate quality

- mean useful candidate count: `0.0000`
- rows with >=2 useful candidates: `0.0000`
- positive candidate fraction: `0.3806`

> Target-derived teacher utility is diagnostic only and is never an inference input.

## Correct vs shuffled-caption sensitivity

- `best_candidate_utility_*` compares raw best candidates and ignores STOP.
- `oracle_policy_utility_*` applies the checkpoint's `epsilon_stop`; its value is zero when the oracle chooses STOP.

- `actions_cosine`: mean `0.996924` (n=528)
- `actions_norm_difference`: mean `1.489470` (n=528)
- `best_candidate_utility_correct_minus_shuffled`: mean `0.000006` (n=132)
- `candidate_utility_correct_minus_shuffled`: mean `0.000017` (n=528)
- `delta_q_cosine`: mean `0.989337` (n=528)
- `delta_q_norm_difference`: mean `0.000849` (n=528)
- `oracle_policy_utility_correct_minus_shuffled`: mean `0.000012` (n=132)
- `proposals_cosine`: mean `0.983802` (n=528)
- `proposals_norm_difference`: mean `2.124638` (n=528)
- `selected_utility_correct`: mean `0.000003` (n=132)
- `selected_utility_correct_minus_shuffled`: mean `0.000064` (n=132)
- `selected_utility_shuffled`: mean `-0.000061` (n=132)
- `terminal_retrieval_correct_advantage`: mean `0.000248` (n=132)

## Score / teacher calibration

- Pearson: `-0.033421`
- bias (score - utility): `0.000400`
- MAE: `0.005574`
- RMSE: `0.007186`
- sign agreement at zero: `0.471649`

| bin | n | teacher mean | score mean | bias |
|---:|---:|---:|---:|---:|
| 0 | 117 | -0.003166 | 0.001536 | 0.004702 |
| 1 | 117 | -0.001046 | 0.000860 | 0.001906 |
| 2 | 117 | -0.000485 | -0.000005 | 0.000481 |
| 3 | 117 | -0.000185 | 0.002716 | 0.002901 |
| 4 | 116 | -0.000099 | 0.002087 | 0.002186 |
| 5 | 116 | -0.000036 | -0.000729 | -0.000692 |
| 6 | 116 | 0.000027 | -0.003646 | -0.003673 |
| 7 | 116 | 0.000440 | 0.000582 | 0.000141 |
| 8 | 116 | 0.001657 | 0.002386 | 0.000729 |
| 9 | 116 | 0.005797 | 0.001047 | -0.004751 |