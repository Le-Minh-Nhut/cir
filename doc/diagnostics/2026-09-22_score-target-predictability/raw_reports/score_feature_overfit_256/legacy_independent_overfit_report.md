# legacy_independent 256-row frozen-cohort overfit

- compact cache: `outputs/diagnostics/2026-09-21/score_feature_sufficiency_t0/compact_cache`
- source checkpoint SHA256: `75e47f40e4cd3c3367889b18d39da878668b0eac1d6a72a57ec2e7de06f93de0`
- TRAIN only; first 8 stored 32-row teacher groups; t0 only; no VAL or probe-dev.

| epoch | loss | Pearson | Spearman | sign@0 | exact oracle | selected utility | oracle utility | regret | STOP | oracle STOP |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00452 | 0.06403 | 0.05977 | 0.58594 | 0.17188 | 0.00184 | 0.05658 | 0.05474 | 0.59375 | 0.12500 |
| 2 | 0.00420 | 0.12839 | 0.11908 | 0.62207 | 0.17188 | 0.00520 | 0.05658 | 0.05138 | 0.79688 | 0.12500 |
| 5 | 0.00315 | 0.29416 | 0.24772 | 0.50000 | 0.28906 | 0.00784 | 0.05658 | 0.04875 | 0.07812 | 0.12500 |
| 10 | 0.00248 | 0.48069 | 0.38715 | 0.61816 | 0.30078 | 0.01813 | 0.05658 | 0.03845 | 0.25781 | 0.12500 |
| 20 | 0.00171 | 0.69993 | 0.57229 | 0.66406 | 0.44922 | 0.03555 | 0.05658 | 0.02103 | 0.15234 | 0.12500 |
| 50 | 0.00065 | 0.89572 | 0.79683 | 0.80078 | 0.56641 | 0.04562 | 0.05658 | 0.01096 | 0.21094 | 0.12500 |
| 100 | 0.00007 | 0.99096 | 0.97435 | 0.92480 | 0.80859 | 0.05549 | 0.05658 | 0.00109 | 0.12891 | 0.12500 |

## Interpretation

Frozen-cohort capacity/optimization can fit this mapping; poor full-TRAIN/DEV/VAL performance is primarily a generalization/predictability concern.