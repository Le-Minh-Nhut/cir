# Frozen T0 Score Feature-Sufficiency Probe

- source checkpoint: `outputs/2026-09-21/20-02-45/best.pt`
- source SHA256: `75e47f40e4cd3c3367889b18d39da878668b0eac1d6a72a57ec2e7de06f93de0`
- train manifest: `${REPO_ROOT}/outputs/diagnostics/2026-09-21/score_feature_probe_train.json`
- true VAL manifest: `outputs/diagnostics/2026-09-21/shared_val160_true.json`
- t0 only; teacher utility is a label and never a probe input.

## TRUE VAL oracle sanity check

- passed: `True`
- actual: `{'oracle_slot_counts': (37, 34, 27, 45, 17), 'oracle_utility': 0.0607275553047657, 'oracle_stop_rate': 0.10625000298023224}`

## Comparison

| probe | Pearson | Spearman | sign@0 | exact oracle | selected utility | oracle utility | regret | harmful / execute | STOP | oracle STOP | dominant slot |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| legacy_independent | 0.0426 | 0.0795 | 0.5703 | 0.2188 | -0.00310 | 0.06073 | 0.06382 | 0.5455 | 0.4500 | 0.1063 | STOP |
| legacy_set_relative | 0.0660 | 0.0779 | 0.5844 | 0.2875 | 0.00464 | 0.06073 | 0.05609 | 0.4494 | 0.4437 | 0.1063 | STOP |
| retrieval_augmented_independent | 0.0537 | 0.1153 | 0.5828 | 0.2188 | 0.00514 | 0.06073 | 0.05559 | 0.4706 | 0.4688 | 0.1063 | STOP |

## Known production/refit TRUE VAL t0 baselines

- `source`: {'pearson': 0.0026, 'exact_oracle_accuracy': 0.3, 'selected_utility': 0.0048, 'oracle_utility': 0.06073, 'regret': 0.05592, 'harmful_execute': 0.5342, 'stop_rate': 0.0875, 'oracle_stop_rate': 0.1063}
- `gain_only_refit`: {'pearson': 0.0682, 'exact_oracle_accuracy': 0.1875, 'selected_utility': 0.0062, 'oracle_utility': 0.06073, 'regret': 0.05452, 'harmful_execute': 0.4035, 'stop_rate': 0.6438, 'oracle_stop_rate': 0.1063}
- `pair_gain_refit`: {'pearson': 0.1023, 'exact_oracle_accuracy': 0.2812, 'selected_utility': 0.0072, 'oracle_utility': 0.06073, 'regret': 0.05353, 'harmful_execute': 0.4938, 'stop_rate': 0.0, 'oracle_stop_rate': 0.1063}