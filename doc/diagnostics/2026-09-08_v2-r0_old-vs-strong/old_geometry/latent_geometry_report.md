# CIR IAG-SRME V2 R0 — Latent Geometry Diagnostic

- checkpoint: `outputs/r0_ncls_text/best.pt`
- epoch: `15`
- validation metric: `30.824426313241325`
- batches: `20`
- batch size: `8`

## Automatic flags

- **OK — NO_STRONG_LATENT_COLLAPSE_FLAG**: No conservative automatic latent-collapse threshold was crossed. Compare OLD vs STRONG reports before concluding geometry is healthy.

## Geometry summary

| Representation | N | mean cos | eff rank(PR) | stable rank | top1 var | top5 var | top10 var | mean dim std | near-zero dims | mean norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0_pooled | 160 | 0.88543 | 12.30015 | 4.07139 | 0.24562 | 0.48336 | 0.59550 | 0.10204 | 0.00000 | 8.73857 |
| reference_global | 160 | 0.28619 | 30.07287 | 7.15237 | 0.13981 | 0.29795 | 0.41639 | 0.77032 | 0.00000 | 26.21841 |
| initial_query | 160 | 0.35008 | 28.88312 | 7.05324 | 0.14178 | 0.30494 | 0.42784 | 0.03460 | 0.00000 | 1.00000 |
| target_query | 160 | 0.35478 | 29.58519 | 7.26136 | 0.13772 | 0.30064 | 0.42270 | 0.03440 | 0.00000 | 1.00000 |
| t0/actions_all | 640 | 0.43805 | 19.29548 | 9.11780 | 0.10968 | 0.43938 | 0.64916 | 0.22507 | 0.00000 | 8.52067 |
| t0/actions_pooled | 640 | 0.43805 | 19.29548 | 9.11780 | 0.10968 | 0.43938 | 0.64916 | 0.22507 | 0.00000 | 8.52067 |
| t0/candidate_global_all | 640 | 0.26853 | 22.40009 | 6.91167 | 0.14468 | 0.38527 | 0.54357 | 0.76418 | 0.00000 | 25.70283 |
| t0/candidate_global_pooled | 640 | 0.26853 | 22.40009 | 6.91167 | 0.14468 | 0.38527 | 0.54357 | 0.76418 | 0.00000 | 25.70283 |
| t0/candidate_query_all | 640 | 0.21802 | 20.60557 | 6.69995 | 0.14925 | 0.40885 | 0.57026 | 0.03788 | 0.00000 | 1.00000 |
| t0/candidate_query_pooled | 640 | 0.21802 | 20.60557 | 6.69995 | 0.14925 | 0.40885 | 0.57026 | 0.03788 | 0.00000 | 1.00000 |
| t0/candidate_state_pooled_all | 640 | 0.76788 | 13.04098 | 5.72947 | 0.17454 | 0.55650 | 0.71349 | 0.15293 | 0.00000 | 9.05369 |
| t0/candidate_state_pooled_pooled | 640 | 0.76788 | 13.04098 | 5.72947 | 0.17454 | 0.55650 | 0.71349 | 0.15293 | 0.00000 | 9.05369 |
| t0/committed_global | 158 | 0.26457 | 22.46389 | 6.93416 | 0.14421 | 0.38395 | 0.54313 | 0.76434 | 0.00000 | 25.69963 |
| t0/committed_query | 158 | 0.21427 | 20.68528 | 6.72118 | 0.14878 | 0.40732 | 0.56961 | 0.03788 | 0.00000 | 1.00000 |
| t0/committed_state_after_decision_pooled | 160 | 0.76130 | 12.48402 | 5.29995 | 0.18868 | 0.56425 | 0.71847 | 0.15424 | 0.00000 | 9.05010 |
| t0/committed_state_pooled | 158 | 0.76604 | 12.98201 | 5.70682 | 0.17523 | 0.55739 | 0.71478 | 0.15310 | 0.00000 | 9.04964 |
| t0/current_global | 160 | 0.28619 | 30.07287 | 7.15237 | 0.13981 | 0.29795 | 0.41639 | 0.77032 | 0.00000 | 26.21841 |
| t0/current_query | 160 | 0.35008 | 28.88312 | 7.05324 | 0.14178 | 0.30494 | 0.42784 | 0.03460 | 0.00000 | 1.00000 |
| t0/current_state_pooled | 160 | 0.88543 | 12.30015 | 4.07139 | 0.24562 | 0.48336 | 0.59550 | 0.10204 | 0.00000 | 8.73857 |
| t0/delta_q_all | 640 | 0.36941 | 15.36039 | 7.10419 | 0.14076 | 0.52213 | 0.67500 | 0.02665 | 0.00000 | 0.77443 |
| t0/delta_q_pooled | 640 | 0.36941 | 15.36039 | 7.10419 | 0.14076 | 0.52213 | 0.67500 | 0.02665 | 0.00000 | 0.77443 |
| t0/proposals_all | 640 | 0.46585 | 19.78103 | 9.45067 | 0.10581 | 0.43228 | 0.64188 | 0.34791 | 0.00000 | 13.25076 |
| t0/proposals_pooled | 640 | 0.46585 | 19.78103 | 9.45067 | 0.10581 | 0.43228 | 0.64188 | 0.34791 | 0.00000 | 13.25076 |
| t1/actions_all | 632 | 0.30154 | 17.21383 | 7.38652 | 0.13538 | 0.46264 | 0.68042 | 0.24466 | 0.00000 | 8.31086 |
| t1/actions_pooled | 632 | 0.30154 | 17.21383 | 7.38652 | 0.13538 | 0.46264 | 0.68042 | 0.24466 | 0.00000 | 8.31086 |
| t1/candidate_global_all | 632 | 0.18640 | 22.40755 | 9.19746 | 0.10873 | 0.39685 | 0.59314 | 0.80452 | 0.00000 | 25.80163 |
| t1/candidate_global_pooled | 632 | 0.18640 | 22.40755 | 9.19746 | 0.10873 | 0.39685 | 0.59314 | 0.80452 | 0.00000 | 25.80163 |
| t1/candidate_query_all | 632 | 0.11178 | 20.89193 | 9.32026 | 0.10729 | 0.41307 | 0.62252 | 0.04028 | 0.00000 | 1.00000 |
| t1/candidate_query_pooled | 632 | 0.11178 | 20.89193 | 9.32026 | 0.10729 | 0.41307 | 0.62252 | 0.04028 | 0.00000 | 1.00000 |
| t1/candidate_state_pooled_all | 632 | 0.62417 | 10.76073 | 5.67641 | 0.17617 | 0.62477 | 0.79367 | 0.21887 | 0.00000 | 10.04774 |
| t1/candidate_state_pooled_pooled | 632 | 0.62417 | 10.76073 | 5.67641 | 0.17617 | 0.62477 | 0.79367 | 0.21887 | 0.00000 | 10.04774 |
| t1/committed_global | 149 | 0.18268 | 22.15425 | 8.77737 | 0.11393 | 0.40083 | 0.59187 | 0.80376 | 0.00000 | 25.79148 |
| t1/committed_query | 149 | 0.10852 | 20.71740 | 8.94179 | 0.11183 | 0.41747 | 0.62027 | 0.04025 | 0.00000 | 1.00000 |
| t1/committed_state_after_decision_pooled | 158 | 0.62553 | 10.82245 | 5.68758 | 0.17582 | 0.62224 | 0.79011 | 0.21698 | 0.00000 | 10.00919 |
| t1/committed_state_pooled | 149 | 0.62150 | 10.69012 | 5.51717 | 0.18125 | 0.62547 | 0.79373 | 0.21851 | 0.00000 | 10.03115 |
| t1/current_global | 158 | 0.26457 | 22.46392 | 6.93417 | 0.14421 | 0.38395 | 0.54313 | 0.76434 | 0.00000 | 25.69964 |
| t1/current_query | 158 | 0.21427 | 20.68530 | 6.72119 | 0.14878 | 0.40733 | 0.56960 | 0.03788 | 0.00000 | 1.00000 |
| t1/current_state_pooled | 158 | 0.76604 | 12.98201 | 5.70682 | 0.17523 | 0.55739 | 0.71478 | 0.15310 | 0.00000 | 9.04964 |
| t1/delta_q_all | 632 | 0.27918 | 11.32462 | 4.15229 | 0.24083 | 0.52982 | 0.70012 | 0.01855 | 0.00000 | 0.48632 |
| t1/delta_q_pooled | 632 | 0.27918 | 11.32462 | 4.15229 | 0.24083 | 0.52982 | 0.70012 | 0.01855 | 0.00000 | 0.48632 |
| t1/proposals_all | 632 | 0.28050 | 17.66266 | 8.18783 | 0.12213 | 0.45778 | 0.67845 | 0.37604 | 0.00000 | 12.35658 |
| t1/proposals_pooled | 632 | 0.28050 | 17.66266 | 8.18783 | 0.12213 | 0.45778 | 0.67845 | 0.37604 | 0.00000 | 12.35658 |
| t2/actions_all | 596 | 0.28714 | 17.37214 | 7.44533 | 0.13431 | 0.46025 | 0.67468 | 0.23941 | 0.00000 | 8.02585 |
| t2/actions_pooled | 596 | 0.28714 | 17.37214 | 7.44533 | 0.13431 | 0.46025 | 0.67468 | 0.23941 | 0.00000 | 8.02585 |
| t2/candidate_global_all | 596 | 0.16964 | 21.43490 | 9.25456 | 0.10805 | 0.40443 | 0.61186 | 0.81837 | 0.00000 | 26.03347 |
| t2/candidate_global_pooled | 596 | 0.16964 | 21.43490 | 9.25456 | 0.10805 | 0.40443 | 0.61186 | 0.81837 | 0.00000 | 26.03347 |
| t2/candidate_query_all | 596 | 0.10509 | 20.31161 | 9.24447 | 0.10817 | 0.41695 | 0.63487 | 0.04033 | 0.00000 | 1.00000 |
| t2/candidate_query_pooled | 596 | 0.10509 | 20.31161 | 9.24447 | 0.10817 | 0.41695 | 0.63487 | 0.04033 | 0.00000 | 1.00000 |
| t2/candidate_state_pooled_all | 596 | 0.52100 | 9.72989 | 4.84969 | 0.20620 | 0.64710 | 0.82292 | 0.28281 | 0.00000 | 11.39260 |
| t2/candidate_state_pooled_pooled | 596 | 0.52100 | 9.72989 | 4.84969 | 0.20620 | 0.64710 | 0.82292 | 0.28281 | 0.00000 | 11.39260 |
| t2/committed_global | 108 | 0.16613 | 20.69439 | 8.79173 | 0.11374 | 0.41494 | 0.61220 | 0.81715 | 0.00000 | 26.00827 |
| t2/committed_query | 108 | 0.09983 | 19.79152 | 9.09111 | 0.11000 | 0.42574 | 0.63427 | 0.04024 | 0.00000 | 1.00000 |
| t2/committed_state_after_decision_pooled | 149 | 0.54205 | 10.15326 | 4.95858 | 0.20167 | 0.63389 | 0.81088 | 0.26916 | 0.00000 | 11.04943 |
| t2/committed_state_pooled | 108 | 0.51629 | 9.60634 | 4.60983 | 0.21693 | 0.64488 | 0.82571 | 0.28598 | 0.00000 | 11.51194 |
| t2/current_global | 149 | 0.18268 | 22.15428 | 8.77737 | 0.11393 | 0.40083 | 0.59187 | 0.80376 | 0.00000 | 25.79149 |
| t2/current_query | 149 | 0.10852 | 20.71740 | 8.94182 | 0.11183 | 0.41747 | 0.62027 | 0.04025 | 0.00000 | 1.00000 |
| t2/current_state_pooled | 149 | 0.62150 | 10.69012 | 5.51717 | 0.18125 | 0.62547 | 0.79373 | 0.21851 | 0.00000 | 10.03115 |
| t2/delta_q_all | 596 | 0.16196 | 13.01113 | 4.32306 | 0.23132 | 0.46751 | 0.62490 | 0.01049 | 0.00000 | 0.25052 |
| t2/delta_q_pooled | 596 | 0.16196 | 13.01113 | 4.32306 | 0.23132 | 0.46751 | 0.62490 | 0.01049 | 0.00000 | 0.25052 |
| t2/proposals_all | 596 | 0.26317 | 17.37068 | 7.64351 | 0.13083 | 0.46116 | 0.67824 | 0.36616 | 0.00000 | 11.88206 |
| t2/proposals_pooled | 596 | 0.26317 | 17.37068 | 7.64351 | 0.13083 | 0.46116 | 0.67824 | 0.36616 | 0.00000 | 11.88206 |
| terminal_query | 160 | 0.10013 | 21.46213 | 9.78823 | 0.10216 | 0.39864 | 0.62031 | 0.04039 | 0.00000 | 1.00000 |
| terminal_state_pooled | 160 | 0.54627 | 10.50314 | 5.14338 | 0.19442 | 0.62381 | 0.80234 | 0.26564 | 0.00000 | 10.94593 |

## Candidate geometry decomposition

| Representation | pooled PR | slot-centered PR | mean per-slot PR | min per-slot PR | between-slot variance |
|---|---:|---:|---:|---:|---:|
| t0/actions | 19.29548 | 19.29299 | 19.28338 | 19.25647 | 0.00009 |

- `t0/actions/slot_0`: PR=19.26826, entropy-rank=37.57676, mean-std=0.22514, zero-variance=False
- `t0/actions/slot_1`: PR=19.25647, entropy-rank=37.56567, mean-std=0.22497, zero-variance=False
- `t0/actions/slot_2`: PR=19.30054, entropy-rank=37.59622, mean-std=0.22475, zero-variance=False
- `t0/actions/slot_3`: PR=19.30825, entropy-rank=37.62459, mean-std=0.22538, zero-variance=False
| t0/candidate_global | 22.40009 | 22.40008 | 22.39898 | 22.36149 | 0.00001 |

- `t0/candidate_global/slot_0`: PR=22.39428, entropy-rank=49.34225, mean-std=0.76382, zero-variance=False
- `t0/candidate_global/slot_1`: PR=22.36149, entropy-rank=49.31134, mean-std=0.76452, zero-variance=False
- `t0/candidate_global/slot_2`: PR=22.42167, entropy-rank=49.39768, mean-std=0.76440, zero-variance=False
- `t0/candidate_global/slot_3`: PR=22.41849, entropy-rank=49.37742, mean-std=0.76398, zero-variance=False
| t0/candidate_query | 20.60557 | 20.60555 | 20.60459 | 20.57234 | 0.00001 |

- `t0/candidate_query/slot_0`: PR=20.60335, entropy-rank=44.76493, mean-std=0.03787, zero-variance=False
- `t0/candidate_query/slot_1`: PR=20.57234, entropy-rank=44.73331, mean-std=0.03789, zero-variance=False
- `t0/candidate_query/slot_2`: PR=20.62384, entropy-rank=44.81141, mean-std=0.03789, zero-variance=False
- `t0/candidate_query/slot_3`: PR=20.61883, entropy-rank=44.78901, mean-std=0.03788, zero-variance=False
| t0/candidate_state_pooled | 13.04098 | 13.04141 | 13.04031 | 13.01445 | 0.00002 |

- `t0/candidate_state_pooled/slot_0`: PR=13.03935, entropy-rank=27.44589, mean-std=0.15294, zero-variance=False
- `t0/candidate_state_pooled/slot_1`: PR=13.01445, entropy-rank=27.38966, mean-std=0.15298, zero-variance=False
- `t0/candidate_state_pooled/slot_2`: PR=13.06909, entropy-rank=27.47114, mean-std=0.15281, zero-variance=False
- `t0/candidate_state_pooled/slot_3`: PR=13.03834, entropy-rank=27.43486, mean-std=0.15296, zero-variance=False
| t0/delta_q | 15.36039 | 15.36044 | 15.35924 | 15.30560 | 0.00002 |

- `t0/delta_q/slot_0`: PR=15.34937, entropy-rank=32.99165, mean-std=0.02667, zero-variance=False
- `t0/delta_q/slot_1`: PR=15.30560, entropy-rank=32.88968, mean-std=0.02665, zero-variance=False
- `t0/delta_q/slot_2`: PR=15.38213, entropy-rank=32.97343, mean-std=0.02663, zero-variance=False
- `t0/delta_q/slot_3`: PR=15.39986, entropy-rank=33.01834, mean-std=0.02666, zero-variance=False
| t0/proposals | 19.78103 | 19.77822 | 19.76764 | 19.74097 | 0.00010 |

- `t0/proposals/slot_0`: PR=19.74886, entropy-rank=38.07694, mean-std=0.34793, zero-variance=False
- `t0/proposals/slot_1`: PR=19.74097, entropy-rank=38.07162, mean-std=0.34773, zero-variance=False
- `t0/proposals/slot_2`: PR=19.78902, entropy-rank=38.09943, mean-std=0.34747, zero-variance=False
- `t0/proposals/slot_3`: PR=19.79169, entropy-rank=38.12954, mean-std=0.34843, zero-variance=False
| t1/actions | 17.21383 | 17.21001 | 17.19749 | 17.13391 | 0.00022 |

- `t1/actions/slot_0`: PR=17.20444, entropy-rank=34.02162, mean-std=0.24468, zero-variance=False
- `t1/actions/slot_1`: PR=17.25130, entropy-rank=34.10013, mean-std=0.24501, zero-variance=False
- `t1/actions/slot_2`: PR=17.13391, entropy-rank=33.90785, mean-std=0.24400, zero-variance=False
- `t1/actions/slot_3`: PR=17.20030, entropy-rank=34.03053, mean-std=0.24483, zero-variance=False
| t1/candidate_global | 22.40755 | 22.40777 | 22.40586 | 22.38268 | 0.00002 |

- `t1/candidate_global/slot_0`: PR=22.43485, entropy-rank=43.66505, mean-std=0.80439, zero-variance=False
- `t1/candidate_global/slot_1`: PR=22.42206, entropy-rank=43.61460, mean-std=0.80451, zero-variance=False
- `t1/candidate_global/slot_2`: PR=22.38268, entropy-rank=43.62556, mean-std=0.80466, zero-variance=False
- `t1/candidate_global/slot_3`: PR=22.38387, entropy-rank=43.58371, mean-std=0.80450, zero-variance=False
| t1/candidate_query | 20.89193 | 20.89211 | 20.89035 | 20.86908 | 0.00002 |

- `t1/candidate_query/slot_0`: PR=20.91384, entropy-rank=39.45495, mean-std=0.04029, zero-variance=False
- `t1/candidate_query/slot_1`: PR=20.90065, entropy-rank=39.40427, mean-std=0.04029, zero-variance=False
- `t1/candidate_query/slot_2`: PR=20.87784, entropy-rank=39.43701, mean-std=0.04028, zero-variance=False
- `t1/candidate_query/slot_3`: PR=20.86908, entropy-rank=39.38161, mean-std=0.04028, zero-variance=False
| t1/candidate_state_pooled | 10.76073 | 10.76083 | 10.76028 | 10.74635 | 0.00003 |

- `t1/candidate_state_pooled/slot_0`: PR=10.76581, entropy-rank=20.22226, mean-std=0.21885, zero-variance=False
- `t1/candidate_state_pooled/slot_1`: PR=10.76394, entropy-rank=20.19613, mean-std=0.21914, zero-variance=False
- `t1/candidate_state_pooled/slot_2`: PR=10.76501, entropy-rank=20.23055, mean-std=0.21836, zero-variance=False
- `t1/candidate_state_pooled/slot_3`: PR=10.74635, entropy-rank=20.17976, mean-std=0.21909, zero-variance=False
| t1/delta_q | 11.32462 | 11.32637 | 11.32428 | 11.19716 | 0.00010 |

- `t1/delta_q/slot_0`: PR=11.38785, entropy-rank=25.99229, mean-std=0.01858, zero-variance=False
- `t1/delta_q/slot_1`: PR=11.40598, entropy-rank=25.99495, mean-std=0.01859, zero-variance=False
- `t1/delta_q/slot_2`: PR=11.19716, entropy-rank=25.70295, mean-std=0.01846, zero-variance=False
- `t1/delta_q/slot_3`: PR=11.30613, entropy-rank=25.87928, mean-std=0.01858, zero-variance=False
| t1/proposals | 17.66266 | 17.65837 | 17.64475 | 17.58883 | 0.00024 |

- `t1/proposals/slot_0`: PR=17.64937, entropy-rank=33.91612, mean-std=0.37590, zero-variance=False
- `t1/proposals/slot_1`: PR=17.70293, entropy-rank=34.01698, mean-std=0.37620, zero-variance=False
- `t1/proposals/slot_2`: PR=17.58883, entropy-rank=33.78030, mean-std=0.37547, zero-variance=False
- `t1/proposals/slot_3`: PR=17.63788, entropy-rank=33.91133, mean-std=0.37637, zero-variance=False
| t2/actions | 17.37214 | 17.36666 | 17.35098 | 17.31183 | 0.00025 |

- `t2/actions/slot_0`: PR=17.34764, entropy-rank=34.08517, mean-std=0.23959, zero-variance=False
- `t2/actions/slot_1`: PR=17.36226, entropy-rank=34.07524, mean-std=0.23978, zero-variance=False
- `t2/actions/slot_2`: PR=17.31183, entropy-rank=34.03629, mean-std=0.23860, zero-variance=False
- `t2/actions/slot_3`: PR=17.38221, entropy-rank=34.14437, mean-std=0.23953, zero-variance=False
| t2/candidate_global | 21.43490 | 21.43492 | 21.43399 | 21.41409 | 0.00001 |

- `t2/candidate_global/slot_0`: PR=21.45307, entropy-rank=40.12025, mean-std=0.81830, zero-variance=False
- `t2/candidate_global/slot_1`: PR=21.41409, entropy-rank=40.05978, mean-std=0.81845, zero-variance=False
- `t2/candidate_global/slot_2`: PR=21.45275, entropy-rank=40.13425, mean-std=0.81832, zero-variance=False
- `t2/candidate_global/slot_3`: PR=21.41607, entropy-rank=40.06586, mean-std=0.81838, zero-variance=False
| t2/candidate_query | 20.31161 | 20.31163 | 20.31081 | 20.29256 | 0.00001 |

- `t2/candidate_query/slot_0`: PR=20.32678, entropy-rank=36.75397, mean-std=0.04033, zero-variance=False
- `t2/candidate_query/slot_1`: PR=20.29256, entropy-rank=36.69790, mean-std=0.04033, zero-variance=False
- `t2/candidate_query/slot_2`: PR=20.32874, entropy-rank=36.77104, mean-std=0.04032, zero-variance=False
- `t2/candidate_query/slot_3`: PR=20.29515, entropy-rank=36.70501, mean-std=0.04032, zero-variance=False
| t2/candidate_state_pooled | 9.72989 | 9.72986 | 9.72960 | 9.71811 | 0.00001 |

- `t2/candidate_state_pooled/slot_0`: PR=9.73318, entropy-rank=17.44588, mean-std=0.28271, zero-variance=False
- `t2/candidate_state_pooled/slot_1`: PR=9.72192, entropy-rank=17.41654, mean-std=0.28337, zero-variance=False
- `t2/candidate_state_pooled/slot_2`: PR=9.74518, entropy-rank=17.47183, mean-std=0.28207, zero-variance=False
- `t2/candidate_state_pooled/slot_3`: PR=9.71811, entropy-rank=17.41845, mean-std=0.28311, zero-variance=False
| t2/delta_q | 13.01113 | 13.00967 | 13.00517 | 12.90155 | 0.00009 |

- `t2/delta_q/slot_0`: PR=12.96934, entropy-rank=31.76947, mean-std=0.01051, zero-variance=False
- `t2/delta_q/slot_1`: PR=13.04842, entropy-rank=31.90723, mean-std=0.01053, zero-variance=False
- `t2/delta_q/slot_2`: PR=12.90155, entropy-rank=31.60621, mean-std=0.01041, zero-variance=False
- `t2/delta_q/slot_3`: PR=13.10137, entropy-rank=31.89606, mean-std=0.01051, zero-variance=False
| t2/proposals | 17.37068 | 17.36468 | 17.34803 | 17.32237 | 0.00027 |

- `t2/proposals/slot_0`: PR=17.34580, entropy-rank=33.21801, mean-std=0.36630, zero-variance=False
- `t2/proposals/slot_1`: PR=17.35241, entropy-rank=33.19854, mean-std=0.36651, zero-variance=False
- `t2/proposals/slot_2`: PR=17.32237, entropy-rank=33.14378, mean-std=0.36517, zero-variance=False
- `t2/proposals/slot_3`: PR=17.37154, entropy-rank=33.25131, mean-std=0.36645, zero-variance=False

## Matched-survivor temporal geometry

Native timestep geometry remains in the table above; temporal flags use these stable-ID-aligned rows.

| baseline | current | matched N | baseline PR | current PR | rank drop | baseline cosine | current cosine | cosine increase |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 158 | 30.10931 | 22.46392 | 0.25392 | 0.28532 | 0.26457 | -0.02075 |
| 0 | 2 | 149 | 29.17532 | 22.15428 | 0.24065 | 0.28347 | 0.18268 | -0.10079 |

## Recurrent drift

| Comparison | relative L2 drift | cosine to anchor |
|---|---:|---:|
| t0/committed_global_vs_initial_global | 0.77202 | 0.69002 |
| t0/current_state_pooled_vs_V0 | 0.00000 | 1.00000 |
| t0/paired_executed_state_after_vs_before | 0.81466 | 0.67727 |
| t1/committed_global_vs_initial_global | 0.88884 | 0.59279 |
| t1/current_state_pooled_vs_V0 | 0.81466 | 0.67727 |
| t1/paired_executed_state_after_vs_before | 0.42965 | 0.91237 |
| t2/committed_global_vs_initial_global | 0.96498 | 0.52491 |
| t2/current_state_pooled_vs_V0 | 0.99108 | 0.58817 |
| t2/paired_executed_state_after_vs_before | 0.27195 | 0.97391 |

## Distribution-shift proxies

| Comparison | mean shift L2 | std shift L2 | norm mean ratio |
|---|---:|---:|---:|
| reference_global_vs_reference_global_distribution | 0.00000 | 0.00000 | 1.00000 |
| t0/candidate_global_pooled_vs_live_reference_global_aligned | 13.04519 | 2.16506 | 0.98034 |
| t0/candidate_global_slot_0_vs_live_reference_global_aligned | 13.07802 | 2.16875 | 0.98033 |
| t0/candidate_global_slot_1_vs_live_reference_global_aligned | 13.01789 | 2.16785 | 0.98040 |
| t0/candidate_global_slot_2_vs_live_reference_global_aligned | 13.02305 | 2.16043 | 0.98032 |
| t0/candidate_global_slot_3_vs_live_reference_global_aligned | 13.06227 | 2.16371 | 0.98029 |
| t0/committed_global_vs_reference_global_distribution | 13.00951 | 2.16460 | 0.98021 |
| t0/current_global_vs_reference_global_distribution | 0.00000 | 0.00000 | 1.00000 |
| t1/candidate_global_pooled_vs_live_reference_global_aligned | 11.92699 | 3.17829 | 0.98418 |
| t1/candidate_global_slot_0_vs_live_reference_global_aligned | 11.94193 | 3.17782 | 0.98412 |
| t1/candidate_global_slot_1_vs_live_reference_global_aligned | 11.94662 | 3.18293 | 0.98414 |
| t1/candidate_global_slot_2_vs_live_reference_global_aligned | 11.89651 | 3.17143 | 0.98427 |
| t1/candidate_global_slot_3_vs_live_reference_global_aligned | 11.92475 | 3.18102 | 0.98419 |
| t1/committed_global_vs_reference_global_distribution | 11.97256 | 3.20336 | 0.98372 |
| t1/current_global_vs_reference_global_distribution | 13.00949 | 2.16460 | 0.98021 |
| t2/candidate_global_pooled_vs_live_reference_global_aligned | 11.34842 | 3.69074 | 0.99323 |
| t2/candidate_global_slot_0_vs_live_reference_global_aligned | 11.35572 | 3.68884 | 0.99320 |
| t2/candidate_global_slot_1_vs_live_reference_global_aligned | 11.35943 | 3.69677 | 0.99329 |
| t2/candidate_global_slot_2_vs_live_reference_global_aligned | 11.33617 | 3.68540 | 0.99314 |
| t2/candidate_global_slot_3_vs_live_reference_global_aligned | 11.34296 | 3.69204 | 0.99327 |
| t2/committed_global_vs_reference_global_distribution | 11.59923 | 3.62147 | 0.99199 |
| t2/current_global_vs_reference_global_distribution | 11.97257 | 3.20333 | 0.98372 |
| terminal_query_vs_target_query_aligned | 0.38665 | 0.17588 | 1.00000 |

## Retrieval geometry

### t0

- `committed_positive_similarity_mean`: `0.50711`
- `committed_positive_similarity_std`: `0.10750`
- `current_positive_similarity_mean`: `0.57487`
- `current_positive_similarity_std`: `0.14596`

### t1

- `committed_positive_similarity_mean`: `0.56080`
- `committed_positive_similarity_std`: `0.06961`
- `current_positive_similarity_mean`: `0.50712`
- `current_positive_similarity_std`: `0.10750`

### t2

- `committed_positive_similarity_mean`: `0.56659`
- `committed_positive_similarity_std`: `0.06053`
- `current_positive_similarity_mean`: `0.56080`
- `current_positive_similarity_std`: `0.06961`

### terminal

- `terminal_positive_similarity_mean`: `0.57678`
- `terminal_positive_similarity_std`: `0.06238`

## Interpretation rules

- Sibling diversity and global representation health are different questions.
- A low sibling cosine does **not** prove the global embedding space is healthy.
- Strong evidence of recurrent collapse would be a large rank drop, rising cross-sample cosine, or rapidly increasing variance concentration from early to late timesteps.
- Healthy rank but large real-vs-synthetic distribution shift would indicate off-manifold drift without classical dimensional collapse.
- Always compare this report between OLD and STRONG checkpoints before changing the objective.