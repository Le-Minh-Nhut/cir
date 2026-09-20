# IAG-SRME V2 R0 Diagnostic Report

## Checkpoint

- epoch: `6`; stored metric: `17.563009758790336`
- identity: `R0-QG-FULL`; readout: `learned_qg`; fine-tune: `full`

## Automatic health flags

- **WARN — DPP_MOSTLY_GATED_OFF**: useful candidates=0.00; DPP needs >=2, so lambda_dpp may not matter.
- **WARN — DPP_LOW_ACTIVATION**: DPP valid rate=0.000.
- **WARN — HARMFUL_EXECUTIONS**: 34.9% selected executions have negative teacher utility.
- **WARN — LOW_CONCEPT_RECALL**: Concept positive recall=0.083.
- **WARN — DPP_NOT_REACHING_EXECUTOR_PROBE**: DPP gradient to representative Executor parameter is zero on this batch; usually the DPP guard is inactive.

## Loss decomposition

| Loss | Raw | Weighted | |Contribution| fraction |
| --- | --- | --- | --- |
| terminal | 0.035598 | 0.035598 | 0.091295 |
| pair | 0.448494 | 0.224247 | 0.649340 |
| gain | 0.136004 | 0.068002 | 0.189681 |
| candidate_credit | 0.024576 | 0.024576 | 0.063182 |
| concept | 0.003458 | 0.002075 | 0.006197 |
| bind | 1.037e-04 | 1.037e-04 | 3.050e-04 |
| rel_ortho | 1.938e-08 | 1.163e-08 | 3.479e-08 |
| dpp | 0.000000 | 0.000000 | 0.000000 |

## Retrieval / rollout

| Metric | Value |
| --- | --- |
| retrieval/initial_terminal_loss | 0.251099 |
| retrieval/final_terminal_loss | 0.035598 |
| retrieval/terminal_loss_improvement | 0.215501 |
| retrieval/positive_similarity_gain | 0.092016 |
| retrieval/inbatch_rank_improvement | 0.171875 |
| rollout/final_state_relative_change | 0.094061 |
| rollout/final_initial_query_cosine | 0.865441 |
| objective/mean_rollout_length | 3.000000 |
| objective/stop_rate | 0.000000 |

## Collapse / diversity

| Metric | Value |
| --- | --- |
| step/proposal/pairwise_cosine_mean | 0.672498 |
| step/entity/pairwise_cosine_mean | 0.993547 |
| step/action/pairwise_cosine_mean | 0.731002 |
| objective/functional_pairwise_cosine | 0.370331 |
| objective/functional_rank | 2.255726 |
| objective/mean_delta_q_norm | 0.340736 |
| step/effect/delta_q_effective_rank | 2.255725 |
| step/effect/delta_q_near_zero_fraction | 0.000000 |
| step/executor/candidate_state_pairwise_l2 | 29.495183 |

## Grounding / write masks

| Metric | Value |
| --- | --- |
| step/grounding/alpha_entropy | 0.903556 |
| step/grounding/alpha_peak | 0.028574 |
| step/grounding/exec_mask_mean | 0.590180 |
| step/grounding/exec_mask_fraction_gt_0_5 | 0.627289 |
| step/grounding/exec_mask_fraction_lt_0_05 | 0.001611 |
| step/grounding/exec_mask_fraction_gt_0_95 | 0.043311 |
| step/grounding/exec_mask_jaccard | 0.611282 |
| step/fusion/gamma_mean | 0.482605 |
| step/fusion/beta_mean | 0.515027 |
| step/fusion/gate_saturation | 5.387e-04 |

## ScoreNet / teacher

| Metric | Value |
| --- | --- |
| score/global_teacher_pearson | 0.482325 |
| step/score/pairwise_teacher_accuracy | 0.828252 |
| step/decision/selected_teacher_utility | 0.088735 |
| step/decision/oracle_teacher_utility | 0.107654 |
| step/decision/oracle_regret | 0.018919 |
| step/decision/harmful_execution_rate | 0.348958 |
| step/decision/missed_opportunity_stop_rate | 0.000000 |
| step/decision/stop_execute_accuracy | 0.812500 |
| step/decision/exact_oracle_action_accuracy | 0.067708 |

## Semantic auxiliaries / DPP

| Metric | Value |
| --- | --- |
| objective/concept_positive_recall | 0.082622 |
| objective/concept_negative_false_positive_rate | 9.954e-05 |
| objective/instruction_concept_coverage | 0.000000 |
| objective/prototype_occupancy | 1.000000 |
| objective/prototype_entropy | 2.079241 |
| objective/prototype_pairwise_cosine | 1.289e-04 |
| objective/useful_candidate_count | 0.000000 |
| objective/dpp_valid_rate | 0.000000 |

## DAC responsibility

| Metric | Value |
| --- | --- |
| objective/dac_stage | 3.000000 |
| objective/dac_num_groups | 8.000000 |
| objective/dac_group_size | 1.000000 |
| objective/dac_split_interval_steps | 2000.000000 |
| objective/dac_gradient_candidate_fraction | 0.125000 |
| objective/dac_responsibility_concentration | 0.244764 |

## Total gradient by module

| Module | grad L2 | max |grad| | nonzero elem frac | no-grad tensor frac |
| --- | --- | --- | --- | --- |
| action_fusion | 0.046195 | 0.003750 | 0.909897 | 0.000000 |
| backbone_q_g | 0.044846 | 0.004614 | 1.000000 | 0.000000 |
| backbone_text | 0.596007 | 0.047145 | 0.598322 | 0.010101 |
| backbone_text_adapter | 0.229079 | 0.005562 | 0.999947 | 0.000000 |
| backbone_vision | 11.210421 | 0.290283 | 0.997219 | 0.000000 |
| backbone_visual_projection | 0.985290 | 0.044067 | 0.999947 | 0.000000 |
| executor | 1.198707 | 0.035858 | 0.999400 | 0.000000 |
| grounder | 0.266121 | 0.095649 | 0.999625 | 0.000000 |
| objective_concept | 0.005031 | 9.978e-05 | 0.996902 | 0.000000 |
| objective_relation | 0.071576 | 0.003952 | 0.997201 | 0.000000 |
| proposal | 0.347068 | 0.011299 | 0.996910 | 0.000000 |
| score_net | 0.711117 | 0.061829 | 0.999691 | 0.000000 |

## Per-loss gradient routing probes

| Loss | action_fusion | backbone_q_g | backbone_text | backbone_text_adapter | backbone_vision | backbone_visual_projection | executor | grounder | objective_concept | objective_relation | proposal | score_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| terminal | 0.025551 | 0.034319 | 0.238441 | 0.216133 | 0.232089 | 0.848158 | 0.202232 | 0.002608 | 0.000000 | 0.000000 | 0.084951 | 0.000000 |
| pair | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.436260 |
| gain | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.397023 |
| candidate_credit | 0.009485 | 0.016224 | 0.021547 | 0.058355 | 0.098238 | 0.303420 | 0.023884 | 1.482e-04 | 0.000000 | 0.000000 | 0.045839 | 0.000000 |
| concept | 0.000000 | 1.704e-05 | 0.002149 | 0.003869 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.007148 | 0.000000 | 3.919e-04 | 0.000000 |
| bind | 0.000000 | 3.622e-06 | 0.001450 | 0.004789 | 2.748e-04 | 8.903e-04 | 0.000000 | 5.676e-05 | 0.000000 | 0.056216 | 1.641e-04 | 0.000000 |
| rel_ortho | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 1.352e-04 | 0.000000 | 0.000000 |
| dpp | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

## Proposal query per-slot gradient L2

| Slot | Total | Terminal | Candidate credit |
| --- | --- | --- | --- |
| 0 | 1.453e-04 | 0.000000 | 1.627e-04 |
| 1 | 1.104e-04 | 0.000000 | 0.000000 |
| 2 | 0.014644 | 0.015032 | 5.366e-04 |
| 3 | 0.062307 | 0.061874 | 6.770e-04 |
| 4 | 0.043498 | 0.000000 | 0.043548 |
| 5 | 0.008966 | 0.009603 | 0.014242 |
| 6 | 0.005224 | 0.004907 | 0.001062 |
| 7 | 0.055142 | 0.055191 | 2.186e-04 |

## Approximate checkpoint movement from fresh initialization

| Group | relative L2 change | mean abs change | max abs change | numel |
| --- | --- | --- | --- | --- |
| action_fusion | 0.108188 | 0.001548 | 0.011518 | 2361600 |
| backbone_q_g | 0.005833 | 7.742e-04 | 0.007283 | 768 |
| backbone_text | 0.015733 | 3.583e-04 | 0.006694 | 63419904 |
| backbone_text_adapter | 0.015125 | 6.062e-04 | 0.003882 | 131840 |
| backbone_text_projection | 0.000000 | 0.000000 | 0.000000 | 262144 |
| backbone_vision | 0.020476 | 5.705e-04 | 0.010555 | 85799424 |
| backbone_visual_projection | 0.064629 | 6.654e-04 | 0.007892 | 393216 |
| executor | 0.053645 | 0.001462 | 0.014744 | 1581312 |
| grounder | 0.030280 | 8.372e-04 | 0.007379 | 985859 |
| other_model | 0.000000 | 0.000000 | 0.000000 | 262147 |
| proposal | 0.048848 | 7.406e-04 | 0.012339 | 10632960 |
| score_net | 0.056564 | 0.001566 | 0.014074 | 1643777 |

> Interpret this only when seed/code/config reproduce the same fresh initialization.

## Selection histogram

`{"2": 10, "4": 2, "5": 14, "6": 81, "7": 85}`

## All aggregate metrics

| Metric | Mean |
| --- | --- |
| loss/total | 0.354601 |
| loss_abs_fraction/bind | 3.050e-04 |
| loss_abs_fraction/candidate_credit | 0.063182 |
| loss_abs_fraction/concept | 0.006197 |
| loss_abs_fraction/dpp | 0.000000 |
| loss_abs_fraction/gain | 0.189681 |
| loss_abs_fraction/pair | 0.649340 |
| loss_abs_fraction/rel_ortho | 3.479e-08 |
| loss_abs_fraction/terminal | 0.091295 |
| loss_raw/bind | 1.037e-04 |
| loss_raw/candidate_credit | 0.024576 |
| loss_raw/concept | 0.003458 |
| loss_raw/dpp | 0.000000 |
| loss_raw/gain | 0.136004 |
| loss_raw/pair | 0.448494 |
| loss_raw/rel_ortho | 1.938e-08 |
| loss_raw/terminal | 0.035598 |
| loss_weighted/bind | 1.037e-04 |
| loss_weighted/candidate_credit | 0.024576 |
| loss_weighted/concept | 0.002075 |
| loss_weighted/dpp | 0.000000 |
| loss_weighted/gain | 0.068002 |
| loss_weighted/pair | 0.224247 |
| loss_weighted/rel_ortho | 1.163e-08 |
| loss_weighted/terminal | 0.035598 |
| objective/concept_negative_false_positive_rate | 9.954e-05 |
| objective/concept_positive_recall | 0.082622 |
| objective/dac_gradient_candidate_fraction | 0.125000 |
| objective/dac_group_size | 1.000000 |
| objective/dac_group_winning_frequency_0 | 0.083347 |
| objective/dac_group_winning_frequency_1 | 0.036453 |
| objective/dac_group_winning_frequency_2 | 0.083321 |
| objective/dac_group_winning_frequency_3 | 0.140610 |
| objective/dac_group_winning_frequency_4 | 0.130211 |
| objective/dac_group_winning_frequency_5 | 0.348923 |
| objective/dac_group_winning_frequency_6 | 0.104164 |
| objective/dac_group_winning_frequency_7 | 0.072918 |
| objective/dac_num_groups | 8.000000 |
| objective/dac_responsibility_concentration | 0.244764 |
| objective/dac_responsibility_frequency_0 | 0.083347 |
| objective/dac_responsibility_frequency_1 | 0.036453 |
| objective/dac_responsibility_frequency_2 | 0.083321 |
| objective/dac_responsibility_frequency_3 | 0.140610 |
| objective/dac_responsibility_frequency_4 | 0.130211 |
| objective/dac_responsibility_frequency_5 | 0.348923 |
| objective/dac_responsibility_frequency_6 | 0.104164 |
| objective/dac_responsibility_frequency_7 | 0.072918 |
| objective/dac_split_interval_steps | 2000.000000 |
| objective/dac_stage | 3.000000 |
| objective/dpp_valid_rate | 0.000000 |
| objective/dpp_valid_timestep_count | 0.000000 |
| objective/functional_pairwise_cosine | 0.370331 |
| objective/functional_rank | 2.255726 |
| objective/instruction_concept_coverage | 0.000000 |
| objective/mean_delta_q_norm | 0.340736 |
| objective/mean_rollout_length | 3.000000 |
| objective/prototype_entropy | 2.079241 |
| objective/prototype_occupancy | 1.000000 |
| objective/prototype_pairwise_cosine | 1.289e-04 |
| objective/stop_rate | 0.000000 |
| objective/teacher_invalid_rows | 0.000000 |
| objective/useful_candidate_count | 0.000000 |
| retrieval/final_inbatch_recall1 | 1.000000 |
| retrieval/final_terminal_loss | 0.035598 |
| retrieval/inbatch_rank_improvement | 0.171875 |
| retrieval/initial_terminal_loss | 0.251099 |
| retrieval/positive_similarity_gain | 0.092016 |
| retrieval/terminal_loss_improvement | 0.215501 |
| rollout/final_initial_query_cosine | 0.865441 |
| rollout/final_state_relative_change | 0.094061 |
| score/global_teacher_pearson | 0.482325 |
| step/action/norm | 7.517118 |
| step/action/pairwise_cosine_max | 0.999136 |
| step/action/pairwise_cosine_mean | 0.731002 |
| step/candidate_query/pairwise_cosine_max | 0.999997 |
| step/candidate_query/pairwise_cosine_mean | 0.877488 |
| step/decision/exact_oracle_action_accuracy | 0.067708 |
| step/decision/harmful_execution_rate | 0.348958 |
| step/decision/missed_opportunity_stop_rate | 0.000000 |
| step/decision/oracle_regret | 0.018919 |
| step/decision/oracle_teacher_utility | 0.107654 |
| step/decision/selected_teacher_utility | 0.088735 |
| step/decision/stop_execute_accuracy | 0.812500 |
| step/delta_q/pairwise_cosine_max | 0.999838 |
| step/delta_q/pairwise_cosine_mean | 0.370364 |
| step/effect/delta_q_effective_rank | 2.255725 |
| step/effect/delta_q_near_zero_fraction | 0.000000 |
| step/effect/delta_q_norm | 0.340736 |
| step/entity/norm | 28.506786 |
| step/entity/pairwise_cosine_max | 0.999999 |
| step/entity/pairwise_cosine_mean | 0.993547 |
| step/executor/candidate_state_pairwise_l2 | 29.495183 |
| step/executor/delta_l2 | 24.195886 |
| step/executor/delta_relative_parent | 0.073811 |
| step/finite/actions | 1.000000 |
| step/finite/alpha_read | 1.000000 |
| step/finite/candidate_queries | 1.000000 |
| step/finite/candidate_states | 1.000000 |
| step/finite/delta | 1.000000 |
| step/finite/delta_q | 1.000000 |
| step/finite/entities | 1.000000 |
| step/finite/exec_mask | 1.000000 |
| step/finite/fuse_beta | 1.000000 |
| step/finite/fuse_gamma | 1.000000 |
| step/finite/grounding | 1.000000 |
| step/finite/proposals | 1.000000 |
| step/finite/scores | 1.000000 |
| step/fusion/beta_mean | 0.515027 |
| step/fusion/beta_std | 0.127671 |
| step/fusion/gamma_mean | 0.482605 |
| step/fusion/gamma_std | 0.137919 |
| step/fusion/gate_saturation | 5.387e-04 |
| step/grounding/alpha_entropy | 0.903556 |
| step/grounding/alpha_peak | 0.028574 |
| step/grounding/exec_mask_fraction_gt_0_5 | 0.627289 |
| step/grounding/exec_mask_fraction_gt_0_95 | 0.043311 |
| step/grounding/exec_mask_fraction_lt_0_05 | 0.001611 |
| step/grounding/exec_mask_jaccard | 0.611282 |
| step/grounding/exec_mask_mean | 0.590180 |
| step/grounding/exec_mask_std | 0.253488 |
| step/grounding/raw_mean | 0.051420 |
| step/grounding/raw_std | 0.136051 |
| step/proposal/norm | 12.370618 |
| step/proposal/pairwise_cosine_max | 0.998789 |
| step/proposal/pairwise_cosine_mean | 0.672498 |
| step/score/mean | -0.051932 |
| step/score/pairwise_teacher_accuracy | 0.828252 |
| step/score/range | 1.110937 |
| step/score/std | 0.489355 |
| step/teacher/dpp_useful_candidate_count | 0.953125 |
| step/teacher/positive_candidate_fraction | 0.415365 |
| step/teacher/utility_mean | 0.023829 |
| step/teacher/utility_std | 0.412490 |

> In-batch retrieval and target-privileged teacher statistics are diagnostics only, not official FashionIQ validation metrics.