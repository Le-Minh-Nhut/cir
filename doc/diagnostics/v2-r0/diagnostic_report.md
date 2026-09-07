# IAG-SRME V2 R0 Diagnostic Report

## Checkpoint

- epoch: `18`; stored metric: `28.244677186012268`
- identity: `R0-NCLS-TEXT`; readout: `native_cls`; fine-tune: `text_only`

## Automatic health flags

- **WARN — DPP_MOSTLY_GATED_OFF**: useful candidates=0.98; DPP needs >=2, so lambda_dpp may not matter.
- **WARN — HARMFUL_EXECUTIONS**: 27.0% selected executions have negative teacher utility.
- **WARN — LOW_CONCEPT_RECALL**: Concept positive recall=0.163.

## Loss decomposition

| Loss | Raw | Weighted | |Contribution| fraction |
| --- | --- | --- | --- |
| terminal | 0.152376 | 0.152376 | 0.124507 |
| pair | 0.382655 | 0.191328 | 0.163980 |
| gain | 0.128422 | 0.064211 | 0.054430 |
| concept | 0.003068 | 0.001841 | 0.001598 |
| bind | 4.084e-05 | 4.084e-05 | 3.567e-05 |
| rel_ortho | 2.106e-09 | 1.264e-09 | 1.097e-09 |
| dpp | -1.277746 | -0.766648 | 0.655449 |

## Retrieval / rollout

| Metric | Value |
| --- | --- |
| retrieval/initial_terminal_loss | 0.906618 |
| retrieval/final_terminal_loss | 0.152376 |
| retrieval/terminal_loss_improvement | 0.754242 |
| retrieval/positive_similarity_gain | -0.025272 |
| retrieval/inbatch_rank_improvement | 0.625000 |
| rollout/final_state_relative_change | 1.037115 |
| rollout/final_initial_query_cosine | 0.599247 |
| objective/mean_rollout_length | 2.900000 |
| objective/stop_rate | 0.013437 |

## Collapse / diversity

| Metric | Value |
| --- | --- |
| step/proposal/pairwise_cosine_mean | 0.371042 |
| step/entity/pairwise_cosine_mean | 0.669788 |
| step/action/pairwise_cosine_mean | 0.342257 |
| objective/functional_pairwise_cosine | 0.013252 |
| objective/functional_rank | 1.916181 |
| objective/mean_delta_q_norm | 0.322646 |
| step/effect/delta_q_effective_rank | 1.916180 |
| step/effect/delta_q_near_zero_fraction | 0.000000 |
| step/executor/candidate_state_pairwise_l2 | 103.979122 |

## Grounding / write masks

| Metric | Value |
| --- | --- |
| step/grounding/alpha_entropy | 0.883210 |
| step/grounding/alpha_peak | 0.034469 |
| step/grounding/exec_mask_mean | 0.439264 |
| step/grounding/exec_mask_fraction_gt_0_5 | 0.422620 |
| step/grounding/exec_mask_fraction_lt_0_05 | 0.040640 |
| step/grounding/exec_mask_fraction_gt_0_95 | 0.010658 |
| step/grounding/exec_mask_jaccard | 0.219320 |
| step/fusion/gamma_mean | 0.371301 |
| step/fusion/beta_mean | 0.464961 |
| step/fusion/gate_saturation | 0.131738 |

## ScoreNet / teacher

| Metric | Value |
| --- | --- |
| score/global_teacher_pearson | 0.612227 |
| step/score/pairwise_teacher_accuracy | 0.783222 |
| step/decision/selected_teacher_utility | 0.257590 |
| step/decision/oracle_teacher_utility | 0.359605 |
| step/decision/oracle_regret | 0.102015 |
| step/decision/harmful_execution_rate | 0.269643 |
| step/decision/missed_opportunity_stop_rate | 0.012500 |
| step/decision/stop_execute_accuracy | 0.908333 |
| step/decision/exact_oracle_action_accuracy | 0.642262 |

## Semantic auxiliaries / DPP

| Metric | Value |
| --- | --- |
| objective/concept_positive_recall | 0.163380 |
| objective/concept_negative_false_positive_rate | 4.593e-04 |
| objective/instruction_concept_coverage | 0.012500 |
| objective/prototype_occupancy | 1.000000 |
| objective/prototype_entropy | 2.079440 |
| objective/prototype_pairwise_cosine | 3.958e-05 |
| objective/useful_candidate_count | 0.983547 |
| objective/dpp_valid_rate | 0.206204 |

## Total gradient by module

| Module | grad L2 | max |grad| | nonzero elem frac | no-grad tensor frac |
| --- | --- | --- | --- | --- |
| action_fusion | 0.012503 | 4.101e-04 | 0.976558 | 0.000000 |
| backbone_text | 3.425079 | 0.083496 | 0.599281 | 0.010101 |
| backbone_text_adapter | 0.588648 | 0.045380 | 0.999985 | 0.000000 |
| executor | 7.340750 | 0.187012 | 0.998858 | 0.000000 |
| grounder | 0.723278 | 0.078064 | 0.999886 | 0.000000 |
| objective_concept | 0.003677 | 5.984e-05 | 0.996755 | 0.000000 |
| objective_relation | 0.029707 | 0.002078 | 0.996951 | 0.000000 |
| proposal | 0.778067 | 0.024292 | 0.998680 | 0.000000 |
| score_net | 0.897744 | 0.035095 | 0.999831 | 0.000000 |

## Per-loss gradient routing probes

| Loss | action_fusion | backbone_text | backbone_text_adapter | executor | grounder | objective_concept | objective_relation | proposal | score_net |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| terminal | 0.005111 | 1.103767 | 0.588422 | 1.067882 | 0.001019 | 0.000000 | 0.000000 | 0.210360 | 0.000000 |
| pair | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.643411 |
| gain | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.550528 |
| concept | 0.000000 | 0.006758 | 0.004203 | 0.000000 | 0.000000 | 0.005929 | 0.000000 | 8.874e-04 | 0.000000 |
| bind | 0.000000 | 8.596e-04 | 9.304e-04 | 0.000000 | 5.970e-07 | 0.000000 | 0.024847 | 1.092e-04 | 0.000000 |
| rel_ortho | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | 4.465e-05 | 0.000000 | 0.000000 |
| dpp | 0.002384 | 0.069580 | 0.040839 | 0.054641 | 1.373e-05 | 0.000000 | 0.000000 | 0.028217 | 0.000000 |

## Approximate checkpoint movement from fresh initialization

| Group | relative L2 change | mean abs change | max abs change | numel |
| --- | --- | --- | --- | --- |
| action_fusion | 0.249836 | 0.003454 | 0.039571 | 2361600 |
| backbone_q_g | 0.000000 | 0.000000 | 0.000000 | 768 |
| backbone_text | 0.026073 | 5.947e-04 | 0.009228 | 63419904 |
| backbone_text_adapter | 0.020557 | 8.286e-04 | 0.005908 | 131840 |
| backbone_text_projection | 0.000000 | 0.000000 | 0.000000 | 262144 |
| backbone_vision | 0.000000 | 0.000000 | 0.000000 | 85799424 |
| backbone_visual_projection | 0.000000 | 0.000000 | 0.000000 | 393216 |
| executor | 0.202667 | 0.004669 | 0.040648 | 1581312 |
| grounder | 0.049782 | 0.001341 | 0.048512 | 985859 |
| other_model | 0.000000 | 0.000000 | 0.000000 | 262147 |
| proposal | 0.066841 | 0.001051 | 0.010524 | 10629888 |
| score_net | 0.089206 | 0.002374 | 0.029128 | 1643777 |

> Interpret this only when seed/code/config reproduce the same fresh initialization.

## Selection histogram

`{"0": 8, "1": 22, "2": 24, "3": 178, "4": 3}`

## All aggregate metrics

| Metric | Mean |
| --- | --- |
| loss/total | -0.356852 |
| loss_abs_fraction/bind | 3.567e-05 |
| loss_abs_fraction/concept | 0.001598 |
| loss_abs_fraction/dpp | 0.655449 |
| loss_abs_fraction/gain | 0.054430 |
| loss_abs_fraction/pair | 0.163980 |
| loss_abs_fraction/rel_ortho | 1.097e-09 |
| loss_abs_fraction/terminal | 0.124507 |
| loss_raw/bind | 4.084e-05 |
| loss_raw/concept | 0.003068 |
| loss_raw/dpp | -1.277746 |
| loss_raw/gain | 0.128422 |
| loss_raw/pair | 0.382655 |
| loss_raw/rel_ortho | 2.106e-09 |
| loss_raw/terminal | 0.152376 |
| loss_weighted/bind | 4.084e-05 |
| loss_weighted/concept | 0.001841 |
| loss_weighted/dpp | -0.766648 |
| loss_weighted/gain | 0.064211 |
| loss_weighted/pair | 0.191328 |
| loss_weighted/rel_ortho | 1.264e-09 |
| loss_weighted/terminal | 0.152376 |
| objective/concept_negative_false_positive_rate | 4.593e-04 |
| objective/concept_positive_recall | 0.163380 |
| objective/dpp_valid_rate | 0.206204 |
| objective/dpp_valid_timestep_count | 4.800000 |
| objective/functional_pairwise_cosine | 0.013252 |
| objective/functional_rank | 1.916181 |
| objective/instruction_concept_coverage | 0.012500 |
| objective/mean_delta_q_norm | 0.322646 |
| objective/mean_rollout_length | 2.900000 |
| objective/prototype_entropy | 2.079440 |
| objective/prototype_occupancy | 1.000000 |
| objective/prototype_pairwise_cosine | 3.958e-05 |
| objective/stop_rate | 0.013437 |
| objective/teacher_invalid_rows | 0.000000 |
| objective/useful_candidate_count | 0.983547 |
| retrieval/final_inbatch_recall1 | 0.975000 |
| retrieval/final_terminal_loss | 0.152376 |
| retrieval/inbatch_rank_improvement | 0.625000 |
| retrieval/initial_terminal_loss | 0.906618 |
| retrieval/positive_similarity_gain | -0.025272 |
| retrieval/terminal_loss_improvement | 0.754242 |
| rollout/final_initial_query_cosine | 0.599247 |
| rollout/final_state_relative_change | 1.037115 |
| score/global_teacher_pearson | 0.612227 |
| step/action/norm | 8.725333 |
| step/action/pairwise_cosine_max | 0.907127 |
| step/action/pairwise_cosine_mean | 0.342257 |
| step/candidate_query/pairwise_cosine_max | 0.994754 |
| step/candidate_query/pairwise_cosine_mean | 0.800035 |
| step/decision/exact_oracle_action_accuracy | 0.642262 |
| step/decision/harmful_execution_rate | 0.269643 |
| step/decision/missed_opportunity_stop_rate | 0.012500 |
| step/decision/oracle_regret | 0.102015 |
| step/decision/oracle_teacher_utility | 0.359605 |
| step/decision/selected_teacher_utility | 0.257590 |
| step/decision/stop_execute_accuracy | 0.908333 |
| step/delta_q/pairwise_cosine_max | 0.579882 |
| step/delta_q/pairwise_cosine_mean | 0.013250 |
| step/effect/delta_q_effective_rank | 1.916180 |
| step/effect/delta_q_near_zero_fraction | 0.000000 |
| step/effect/delta_q_norm | 0.319389 |
| step/entity/norm | 14.531639 |
| step/entity/pairwise_cosine_max | 0.992020 |
| step/entity/pairwise_cosine_mean | 0.669788 |
| step/executor/candidate_state_pairwise_l2 | 103.979122 |
| step/executor/delta_l2 | 68.334124 |
| step/executor/delta_relative_parent | 0.224091 |
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
| step/fusion/beta_mean | 0.464961 |
| step/fusion/beta_std | 0.291418 |
| step/fusion/gamma_mean | 0.371301 |
| step/fusion/gamma_std | 0.273159 |
| step/fusion/gate_saturation | 0.131738 |
| step/grounding/alpha_entropy | 0.883210 |
| step/grounding/alpha_peak | 0.034469 |
| step/grounding/exec_mask_fraction_gt_0_5 | 0.422620 |
| step/grounding/exec_mask_fraction_gt_0_95 | 0.010658 |
| step/grounding/exec_mask_fraction_lt_0_05 | 0.040640 |
| step/grounding/exec_mask_jaccard | 0.219320 |
| step/grounding/exec_mask_mean | 0.439264 |
| step/grounding/exec_mask_std | 0.276514 |
| step/grounding/raw_mean | -0.031282 |
| step/grounding/raw_std | 0.144900 |
| step/proposal/norm | 12.429876 |
| step/proposal/pairwise_cosine_max | 0.877135 |
| step/proposal/pairwise_cosine_mean | 0.371042 |
| step/score/mean | 0.030491 |
| step/score/pairwise_teacher_accuracy | 0.783222 |
| step/score/range | 1.302896 |
| step/score/std | 0.628160 |
| step/teacher/dpp_useful_candidate_count | 0.979762 |
| step/teacher/positive_candidate_fraction | 0.462202 |
| step/teacher/utility_mean | 0.011236 |
| step/teacher/utility_std | 0.422635 |

> In-batch retrieval and target-privileged teacher statistics are diagnostics only, not official FashionIQ validation metrics.