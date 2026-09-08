# CIR IAG-SRME V2 R0 — Latent Geometry Diagnostic

- checkpoint: `outputs/r0_ncls_text_strong_aux/best.pt`
- epoch: `18`
- validation metric: `28.244677186012268`
- batches: `20`
- batch size: `8`

## Automatic flags

- **WARN — PATCH_STATE_RANK_DROP**: pooled-patch-state effective rank is 33.0% lower (V0_pooled 12.30 -> terminal_state_pooled 8.24).
- **WARN — RECURRENT_RANK_COLLAPSE**: Current-global effective rank for the same 154 survivors drops 30.8% from t0 to t2.
- **WARN — LARGE_RECURRENT_STATE_DRIFT**: t2/current_state_pooled_vs_V0 relative L2 drift is 1.122.

## Geometry summary

| Representation | N | mean cos | eff rank(PR) | stable rank | top1 var | top5 var | top10 var | mean dim std | near-zero dims | mean norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0_pooled | 160 | 0.88543 | 12.30015 | 4.07139 | 0.24562 | 0.48336 | 0.59550 | 0.10204 | 0.00000 | 8.73857 |
| reference_global | 160 | 0.28619 | 30.07287 | 7.15237 | 0.13981 | 0.29795 | 0.41639 | 0.77032 | 0.00000 | 26.21841 |
| initial_query | 160 | 0.35008 | 28.88312 | 7.05324 | 0.14178 | 0.30494 | 0.42784 | 0.03460 | 0.00000 | 1.00000 |
| target_query | 160 | 0.35478 | 29.58519 | 7.26136 | 0.13772 | 0.30064 | 0.42270 | 0.03440 | 0.00000 | 1.00000 |
| t0/actions_all | 640 | 0.34875 | 3.75873 | 2.01858 | 0.49540 | 0.71935 | 0.78277 | 0.20652 | 0.00000 | 7.28079 |
| t0/actions_pooled | 640 | 0.34875 | 3.75873 | 2.01858 | 0.49540 | 0.71935 | 0.78277 | 0.20652 | 0.00000 | 7.28079 |
| t0/candidate_global_all | 640 | 0.22062 | 17.93756 | 5.53505 | 0.18067 | 0.42113 | 0.53162 | 0.80792 | 0.00000 | 26.22444 |
| t0/candidate_global_pooled | 640 | 0.22062 | 17.93756 | 5.53505 | 0.18067 | 0.42113 | 0.53162 | 0.80792 | 0.00000 | 26.22444 |
| t0/candidate_query_all | 640 | 0.22801 | 16.99472 | 5.33586 | 0.18741 | 0.43042 | 0.54425 | 0.03730 | 0.00000 | 1.00000 |
| t0/candidate_query_pooled | 640 | 0.22801 | 16.99472 | 5.33586 | 0.18741 | 0.43042 | 0.54425 | 0.03730 | 0.00000 | 1.00000 |
| t0/candidate_state_pooled_all | 640 | 0.72440 | 3.14585 | 1.82188 | 0.54888 | 0.76582 | 0.83241 | 0.19111 | 0.00000 | 9.90863 |
| t0/candidate_state_pooled_pooled | 640 | 0.72440 | 3.14585 | 1.82188 | 0.54888 | 0.76582 | 0.83241 | 0.19111 | 0.00000 | 9.90863 |
| t0/committed_global | 155 | 0.24363 | 10.72804 | 3.76134 | 0.26586 | 0.51065 | 0.61167 | 0.78659 | 0.00000 | 26.11947 |
| t0/committed_query | 155 | 0.20651 | 10.81974 | 3.83780 | 0.26057 | 0.51807 | 0.62310 | 0.03738 | 0.00000 | 1.00000 |
| t0/committed_state_after_decision_pooled | 160 | 0.73393 | 2.73746 | 1.69327 | 0.59057 | 0.80520 | 0.85778 | 0.20851 | 0.00000 | 10.55168 |
| t0/committed_state_pooled | 155 | 0.73770 | 2.69387 | 1.67790 | 0.59598 | 0.80710 | 0.85892 | 0.20855 | 0.00000 | 10.61957 |
| t0/current_global | 160 | 0.28619 | 30.07287 | 7.15237 | 0.13981 | 0.29795 | 0.41639 | 0.77032 | 0.00000 | 26.21841 |
| t0/current_query | 160 | 0.35008 | 28.88312 | 7.05324 | 0.14178 | 0.30494 | 0.42784 | 0.03460 | 0.00000 | 1.00000 |
| t0/current_state_pooled | 160 | 0.88543 | 12.30015 | 4.07139 | 0.24562 | 0.48336 | 0.59550 | 0.10204 | 0.00000 | 8.73857 |
| t0/delta_q_all | 640 | 0.08700 | 5.57787 | 2.63280 | 0.37982 | 0.68726 | 0.75157 | 0.02444 | 0.00000 | 0.56929 |
| t0/delta_q_pooled | 640 | 0.08700 | 5.57787 | 2.63280 | 0.37982 | 0.68726 | 0.75157 | 0.02444 | 0.00000 | 0.56929 |
| t0/proposals_all | 640 | 0.35679 | 3.48228 | 1.90592 | 0.52468 | 0.70889 | 0.79126 | 0.33228 | 0.00000 | 11.67761 |
| t0/proposals_pooled | 640 | 0.35679 | 3.48228 | 1.90592 | 0.52468 | 0.70889 | 0.79126 | 0.33228 | 0.00000 | 11.67761 |
| t1/actions_all | 620 | 0.28406 | 4.01873 | 2.20427 | 0.45366 | 0.76764 | 0.82682 | 0.26241 | 0.00000 | 9.21861 |
| t1/actions_pooled | 620 | 0.28406 | 4.01873 | 2.20427 | 0.45366 | 0.76764 | 0.82682 | 0.26241 | 0.00000 | 9.21861 |
| t1/candidate_global_all | 620 | 0.24024 | 14.32796 | 4.57657 | 0.21850 | 0.45285 | 0.57072 | 0.79194 | 0.00000 | 26.07659 |
| t1/candidate_global_pooled | 620 | 0.24024 | 14.32796 | 4.57657 | 0.21850 | 0.45285 | 0.57072 | 0.79194 | 0.00000 | 26.07659 |
| t1/candidate_query_all | 620 | 0.19542 | 14.36798 | 4.69113 | 0.21317 | 0.46172 | 0.58382 | 0.03803 | 0.00000 | 1.00000 |
| t1/candidate_query_pooled | 620 | 0.19542 | 14.36798 | 4.69113 | 0.21317 | 0.46172 | 0.58382 | 0.03803 | 0.00000 | 1.00000 |
| t1/candidate_state_pooled_all | 620 | 0.72033 | 3.40833 | 1.91163 | 0.52311 | 0.77327 | 0.83660 | 0.21325 | 0.00000 | 10.78222 |
| t1/candidate_state_pooled_pooled | 620 | 0.72033 | 3.40833 | 1.91163 | 0.52311 | 0.77327 | 0.83660 | 0.21325 | 0.00000 | 10.78222 |
| t1/committed_global | 154 | 0.29604 | 20.79744 | 6.64388 | 0.15051 | 0.41144 | 0.54664 | 0.75728 | 0.00000 | 25.93275 |
| t1/committed_query | 154 | 0.21190 | 18.95927 | 6.51749 | 0.15343 | 0.43681 | 0.57884 | 0.03796 | 0.00000 | 1.00000 |
| t1/committed_state_after_decision_pooled | 155 | 0.79259 | 6.95104 | 3.04419 | 0.32850 | 0.64671 | 0.75628 | 0.19044 | 0.00000 | 11.23547 |
| t1/committed_state_pooled | 154 | 0.79440 | 6.98960 | 3.05541 | 0.32729 | 0.64473 | 0.75512 | 0.19011 | 0.00000 | 11.24868 |
| t1/current_global | 155 | 0.24363 | 10.72801 | 3.76133 | 0.26586 | 0.51065 | 0.61167 | 0.78659 | 0.00000 | 26.11945 |
| t1/current_query | 155 | 0.20651 | 10.81970 | 3.83778 | 0.26057 | 0.51807 | 0.62310 | 0.03738 | 0.00000 | 1.00000 |
| t1/current_state_pooled | 155 | 0.73770 | 2.69387 | 1.67790 | 0.59598 | 0.80710 | 0.85892 | 0.20855 | 0.00000 | 10.61957 |
| t1/delta_q_all | 620 | 0.08595 | 9.41559 | 3.94609 | 0.25342 | 0.59114 | 0.72921 | 0.01644 | 0.00000 | 0.23088 |
| t1/delta_q_pooled | 620 | 0.08595 | 9.41559 | 3.94609 | 0.25342 | 0.59114 | 0.72921 | 0.01644 | 0.00000 | 0.23088 |
| t1/proposals_all | 620 | 0.28565 | 2.87581 | 1.73336 | 0.57691 | 0.77421 | 0.83427 | 0.37171 | 0.00000 | 12.64091 |
| t1/proposals_pooled | 620 | 0.28565 | 2.87581 | 1.73336 | 0.57691 | 0.77421 | 0.83427 | 0.37171 | 0.00000 | 12.64091 |
| t2/actions_all | 616 | 0.33449 | 3.76814 | 2.10560 | 0.47493 | 0.75107 | 0.80578 | 0.26704 | 0.00000 | 9.81413 |
| t2/actions_pooled | 616 | 0.33449 | 3.76814 | 2.10560 | 0.47493 | 0.75107 | 0.80578 | 0.26704 | 0.00000 | 9.81413 |
| t2/candidate_global_all | 616 | 0.26300 | 23.41153 | 7.44416 | 0.13433 | 0.38322 | 0.53189 | 0.77796 | 0.00000 | 25.95053 |
| t2/candidate_global_pooled | 616 | 0.26300 | 23.41153 | 7.44416 | 0.13433 | 0.38322 | 0.53189 | 0.77796 | 0.00000 | 25.95053 |
| t2/candidate_query_all | 616 | 0.17781 | 21.30711 | 7.32329 | 0.13655 | 0.40768 | 0.56427 | 0.03894 | 0.00000 | 1.00000 |
| t2/candidate_query_pooled | 616 | 0.17781 | 21.30711 | 7.32329 | 0.13655 | 0.40768 | 0.56427 | 0.03894 | 0.00000 | 1.00000 |
| t2/candidate_state_pooled_all | 616 | 0.76438 | 7.51121 | 3.26077 | 0.30668 | 0.64175 | 0.75509 | 0.19986 | 0.00000 | 11.17282 |
| t2/candidate_state_pooled_pooled | 616 | 0.76438 | 7.51121 | 3.26077 | 0.30668 | 0.64175 | 0.75509 | 0.19986 | 0.00000 | 11.17282 |
| t2/committed_global | 154 | 0.21790 | 24.58693 | 9.16819 | 0.10907 | 0.37828 | 0.55199 | 0.79894 | 0.00000 | 25.98605 |
| t2/committed_query | 154 | 0.12049 | 22.17962 | 9.06369 | 0.11033 | 0.40310 | 0.58873 | 0.04024 | 0.00000 | 1.00000 |
| t2/committed_state_after_decision_pooled | 154 | 0.73535 | 7.98199 | 3.35634 | 0.29794 | 0.61629 | 0.74337 | 0.20544 | 0.00000 | 10.91818 |
| t2/committed_state_pooled | 154 | 0.73535 | 7.98199 | 3.35634 | 0.29794 | 0.61629 | 0.74337 | 0.20544 | 0.00000 | 10.91818 |
| t2/current_global | 154 | 0.29604 | 20.79747 | 6.64389 | 0.15051 | 0.41144 | 0.54664 | 0.75728 | 0.00000 | 25.93274 |
| t2/current_query | 154 | 0.21190 | 18.95939 | 6.51758 | 0.15343 | 0.43681 | 0.57884 | 0.03796 | 0.00000 | 1.00000 |
| t2/current_state_pooled | 154 | 0.79440 | 6.98960 | 3.05541 | 0.32729 | 0.64473 | 0.75512 | 0.19011 | 0.00000 | 11.24868 |
| t2/delta_q_all | 616 | 0.14083 | 9.23715 | 3.58822 | 0.27869 | 0.56978 | 0.74035 | 0.01260 | 0.00000 | 0.17361 |
| t2/delta_q_pooled | 616 | 0.14083 | 9.23715 | 3.58822 | 0.27869 | 0.56978 | 0.74035 | 0.01260 | 0.00000 | 0.17361 |
| t2/proposals_all | 616 | 0.32740 | 3.09951 | 1.80965 | 0.55259 | 0.76364 | 0.82753 | 0.36443 | 0.00000 | 12.90923 |
| t2/proposals_pooled | 616 | 0.32740 | 3.09951 | 1.80965 | 0.55259 | 0.76364 | 0.82753 | 0.36443 | 0.00000 | 12.90923 |
| terminal_query | 160 | 0.12260 | 22.92541 | 9.14738 | 0.10932 | 0.39477 | 0.57790 | 0.04019 | 0.00000 | 1.00000 |
| terminal_state_pooled | 160 | 0.71269 | 8.24180 | 3.55285 | 0.28146 | 0.63364 | 0.75364 | 0.20965 | 0.00000 | 10.83021 |

## Candidate geometry decomposition

| Representation | pooled PR | slot-centered PR | mean per-slot PR | min per-slot PR | between-slot variance |
|---|---:|---:|---:|---:|---:|
| t0/actions | 3.75873 | 7.44233 | 8.77028 | 2.06279 | 0.42020 |

- `t0/actions/slot_0`: PR=3.95591, entropy-rank=15.28522, mean-std=0.15187, zero-variance=False
- `t0/actions/slot_1`: PR=5.32322, entropy-rank=20.41029, mean-std=0.14887, zero-variance=False
- `t0/actions/slot_2`: PR=2.06279, entropy-rank=6.47595, mean-std=0.15636, zero-variance=False
- `t0/actions/slot_3`: PR=23.73921, entropy-rank=49.22833, mean-std=0.14967, zero-variance=False
| t0/candidate_global | 17.93756 | 28.89654 | 20.46919 | 10.99499 | 0.15956 |

- `t0/candidate_global/slot_0`: PR=20.70098, entropy-rank=53.43587, mean-std=0.79095, zero-variance=False
- `t0/candidate_global/slot_1`: PR=22.47081, entropy-rank=56.16188, mean-std=0.75805, zero-variance=False
- `t0/candidate_global/slot_2`: PR=27.70997, entropy-rank=62.05708, mean-std=0.78991, zero-variance=False
- `t0/candidate_global/slot_3`: PR=10.99499, entropy-rank=38.28957, mean-std=0.60756, zero-variance=False
| t0/candidate_query | 16.99472 | 27.34180 | 19.89456 | 9.34310 | 0.16084 |

- `t0/candidate_query/slot_0`: PR=20.94591, entropy-rank=51.93047, mean-std=0.03671, zero-variance=False
- `t0/candidate_query/slot_1`: PR=22.19883, entropy-rank=53.95245, mean-std=0.03633, zero-variance=False
- `t0/candidate_query/slot_2`: PR=27.09041, entropy-rank=59.26986, mean-std=0.03592, zero-variance=False
- `t0/candidate_query/slot_3`: PR=9.34310, entropy-rank=33.08965, mean-std=0.02740, zero-variance=False
| t0/candidate_state_pooled | 3.14585 | 9.55375 | 7.16132 | 4.59033 | 0.44883 |

- `t0/candidate_state_pooled/slot_0`: PR=6.66584, entropy-rank=20.73103, mean-std=0.13474, zero-variance=False
- `t0/candidate_state_pooled/slot_1`: PR=10.85002, entropy-rank=28.80557, mean-std=0.12384, zero-variance=False
- `t0/candidate_state_pooled/slot_2`: PR=6.53907, entropy-rank=20.80840, mean-std=0.13127, zero-variance=False
- `t0/candidate_state_pooled/slot_3`: PR=4.59033, entropy-rank=13.48230, mean-std=0.18623, zero-variance=False
| t0/delta_q | 5.57787 | 12.20437 | 7.42266 | 3.04483 | 0.34779 |

- `t0/delta_q/slot_0`: PR=3.15997, entropy-rank=8.81798, mean-std=0.01794, zero-variance=False
- `t0/delta_q/slot_1`: PR=4.42072, entropy-rank=15.03590, mean-std=0.01861, zero-variance=False
- `t0/delta_q/slot_2`: PR=3.04483, entropy-rank=7.49244, mean-std=0.01371, zero-variance=False
- `t0/delta_q/slot_3`: PR=19.06512, entropy-rank=49.00842, mean-std=0.02672, zero-variance=False
| t0/proposals | 3.48228 | 7.78811 | 9.93600 | 2.64158 | 0.45334 |

- `t0/proposals/slot_0`: PR=4.46198, entropy-rank=16.60626, mean-std=0.25889, zero-variance=False
- `t0/proposals/slot_1`: PR=7.58422, entropy-rank=25.82174, mean-std=0.23819, zero-variance=False
- `t0/proposals/slot_2`: PR=2.64158, entropy-rank=8.63937, mean-std=0.24999, zero-variance=False
- `t0/proposals/slot_3`: PR=25.05623, entropy-rank=48.83697, mean-std=0.23871, zero-variance=False
| t1/actions | 4.01873 | 12.82508 | 9.15680 | 3.92746 | 0.60097 |

- `t1/actions/slot_0`: PR=5.22197, entropy-rank=16.15026, mean-std=0.15001, zero-variance=False
- `t1/actions/slot_1`: PR=5.43120, entropy-rank=17.50910, mean-std=0.14938, zero-variance=False
- `t1/actions/slot_2`: PR=3.92746, entropy-rank=13.10723, mean-std=0.15385, zero-variance=False
- `t1/actions/slot_3`: PR=22.04657, entropy-rank=46.96719, mean-std=0.18419, zero-variance=False
| t1/candidate_global | 14.32796 | 14.24632 | 13.40868 | 10.82825 | 0.02533 |

- `t1/candidate_global/slot_0`: PR=10.89147, entropy-rank=34.65282, mean-std=0.78762, zero-variance=False
- `t1/candidate_global/slot_1`: PR=10.82825, entropy-rank=34.74269, mean-std=0.78574, zero-variance=False
- `t1/candidate_global/slot_2`: PR=10.99557, entropy-rank=34.72976, mean-std=0.78825, zero-variance=False
- `t1/candidate_global/slot_3`: PR=20.91941, entropy-rank=48.08351, mean-std=0.75694, zero-variance=False
| t1/candidate_query | 14.36798 | 14.36807 | 13.02523 | 10.97331 | 0.02320 |

- `t1/candidate_query/slot_0`: PR=10.97331, entropy-rank=33.21027, mean-std=0.03737, zero-variance=False
- `t1/candidate_query/slot_1`: PR=10.98376, entropy-rank=33.39638, mean-std=0.03733, zero-variance=False
- `t1/candidate_query/slot_2`: PR=11.09330, entropy-rank=33.31305, mean-std=0.03728, zero-variance=False
- `t1/candidate_query/slot_3`: PR=19.05055, entropy-rank=43.05071, mean-std=0.03795, zero-variance=False
| t1/candidate_state_pooled | 3.40833 | 3.30094 | 3.79659 | 2.70306 | 0.08225 |

- `t1/candidate_state_pooled/slot_0`: PR=2.70994, entropy-rank=7.58314, mean-std=0.20892, zero-variance=False
- `t1/candidate_state_pooled/slot_1`: PR=2.71451, entropy-rank=7.63005, mean-std=0.20848, zero-variance=False
- `t1/candidate_state_pooled/slot_2`: PR=2.70306, entropy-rank=7.55040, mean-std=0.20800, zero-variance=False
- `t1/candidate_state_pooled/slot_3`: PR=7.05883, entropy-rank=19.02108, mean-std=0.18978, zero-variance=False
| t1/delta_q | 9.41559 | 10.74164 | 5.87556 | 1.42176 | 0.12411 |

- `t1/delta_q/slot_0`: PR=1.42176, entropy-rank=2.64786, mean-std=0.00350, zero-variance=False
- `t1/delta_q/slot_1`: PR=2.21043, entropy-rank=3.97077, mean-std=0.00507, zero-variance=False
- `t1/delta_q/slot_2`: PR=9.54194, entropy-rank=28.82155, mean-std=0.00133, zero-variance=False
- `t1/delta_q/slot_3`: PR=10.32809, entropy-rank=25.81969, mean-std=0.03009, zero-variance=False
| t1/proposals | 2.87581 | 23.80244 | 15.65626 | 10.82287 | 0.68290 |

- `t1/proposals/slot_0`: PR=13.01858, entropy-rank=31.48254, mean-std=0.18435, zero-variance=False
- `t1/proposals/slot_1`: PR=15.62571, entropy-rank=34.82133, mean-std=0.20837, zero-variance=False
- `t1/proposals/slot_2`: PR=10.82287, entropy-rank=25.36103, mean-std=0.15771, zero-variance=False
- `t1/proposals/slot_3`: PR=23.15789, entropy-rank=47.52337, mean-std=0.28814, zero-variance=False
| t2/actions | 3.76814 | 28.70581 | 14.72486 | 9.50262 | 0.66367 |

- `t2/actions/slot_0`: PR=10.73696, entropy-rank=30.24885, mean-std=0.13233, zero-variance=False
- `t2/actions/slot_1`: PR=15.43433, entropy-rank=35.45395, mean-std=0.12804, zero-variance=False
- `t2/actions/slot_2`: PR=9.50262, entropy-rank=26.76468, mean-std=0.14566, zero-variance=False
- `t2/actions/slot_3`: PR=23.22553, entropy-rank=48.13333, mean-std=0.19406, zero-variance=False
| t2/candidate_global | 23.41153 | 23.31196 | 21.72871 | 20.73078 | 0.02311 |

- `t2/candidate_global/slot_0`: PR=20.75159, entropy-rank=48.02758, mean-std=0.75632, zero-variance=False
- `t2/candidate_global/slot_1`: PR=20.73078, entropy-rank=47.87853, mean-std=0.75854, zero-variance=False
- `t2/candidate_global/slot_2`: PR=20.84554, entropy-rank=48.02190, mean-std=0.75770, zero-variance=False
- `t2/candidate_global/slot_3`: PR=24.58693, entropy-rank=48.90298, mean-std=0.79894, zero-variance=False
| t2/candidate_query | 21.30711 | 21.21494 | 19.76734 | 18.92375 | 0.02243 |

- `t2/candidate_query/slot_0`: PR=18.95906, entropy-rank=43.06721, mean-std=0.03781, zero-variance=False
- `t2/candidate_query/slot_1`: PR=18.92375, entropy-rank=42.91478, mean-std=0.03793, zero-variance=False
- `t2/candidate_query/slot_2`: PR=19.00694, entropy-rank=43.03828, mean-std=0.03781, zero-variance=False
- `t2/candidate_query/slot_3`: PR=22.17962, entropy-rank=43.36350, mean-std=0.04024, zero-variance=False
| t2/candidate_state_pooled | 7.51121 | 7.48862 | 7.23468 | 6.96572 | 0.06459 |

- `t2/candidate_state_pooled/slot_0`: PR=6.98664, entropy-rank=18.87170, mean-std=0.18945, zero-variance=False
- `t2/candidate_state_pooled/slot_1`: PR=6.96572, entropy-rank=18.82737, mean-std=0.18993, zero-variance=False
- `t2/candidate_state_pooled/slot_2`: PR=7.00438, entropy-rank=18.91891, mean-std=0.18948, zero-variance=False
- `t2/candidate_state_pooled/slot_3`: PR=7.98199, entropy-rank=20.54403, mean-std=0.20544, zero-variance=False
| t2/delta_q | 9.23715 | 16.14875 | 16.17010 | 12.88206 | 0.21367 |

- `t2/delta_q/slot_0`: PR=18.31820, entropy-rank=38.04015, mean-std=0.00139, zero-variance=False
- `t2/delta_q/slot_1`: PR=12.88206, entropy-rank=30.79556, mean-std=0.00094, zero-variance=False
- `t2/delta_q/slot_2`: PR=17.49088, entropy-rank=38.68196, mean-std=0.00107, zero-variance=False
- `t2/delta_q/slot_3`: PR=15.98925, entropy-rank=30.42565, mean-std=0.02248, zero-variance=False
| t2/proposals | 3.09951 | 26.35243 | 16.67341 | 11.13676 | 0.68813 |

- `t2/proposals/slot_0`: PR=13.69269, entropy-rank=34.56260, mean-std=0.16970, zero-variance=False
- `t2/proposals/slot_1`: PR=16.63653, entropy-rank=37.23179, mean-std=0.19160, zero-variance=False
- `t2/proposals/slot_2`: PR=11.13676, entropy-rank=28.82250, mean-std=0.15041, zero-variance=False
- `t2/proposals/slot_3`: PR=25.22764, entropy-rank=49.23423, mean-std=0.29462, zero-variance=False

## Matched-survivor temporal geometry

Native timestep geometry remains in the table above; temporal flags use these stable-ID-aligned rows.

| baseline | current | matched N | baseline PR | current PR | rank drop | baseline cosine | current cosine | cosine increase |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 155 | 30.16276 | 10.72801 | 0.64433 | 0.28946 | 0.24363 | -0.04582 |
| 0 | 2 | 154 | 30.03952 | 20.79747 | 0.30766 | 0.28967 | 0.29604 | 0.00637 |

## Recurrent drift

| Comparison | relative L2 drift | cosine to anchor |
|---|---:|---:|
| t0/committed_global_vs_initial_global | 0.75166 | 0.70520 |
| t0/current_state_pooled_vs_V0 | 0.00000 | 1.00000 |
| t0/paired_executed_state_after_vs_before | 0.76368 | 0.77161 |
| t1/committed_global_vs_initial_global | 0.87966 | 0.60290 |
| t1/current_state_pooled_vs_V0 | 0.76368 | 0.77161 |
| t1/paired_executed_state_after_vs_before | 0.56200 | 0.84833 |
| t2/committed_global_vs_initial_global | 0.91392 | 0.57324 |
| t2/current_state_pooled_vs_V0 | 1.12218 | 0.55377 |
| t2/paired_executed_state_after_vs_before | 0.33712 | 0.93279 |

## Distribution-shift proxies

| Comparison | mean shift L2 | std shift L2 | norm mean ratio |
|---|---:|---:|---:|
| reference_global_vs_reference_global_distribution | 0.00000 | 0.00000 | 1.00000 |
| t0/candidate_global_pooled_vs_live_reference_global_aligned | 7.16909 | 3.14001 | 1.00023 |
| t0/candidate_global_slot_0_vs_live_reference_global_aligned | 7.62688 | 2.74360 | 0.99746 |
| t0/candidate_global_slot_1_vs_live_reference_global_aligned | 10.10097 | 2.33763 | 0.98956 |
| t0/candidate_global_slot_2_vs_live_reference_global_aligned | 5.33715 | 1.79472 | 0.99821 |
| t0/candidate_global_slot_3_vs_live_reference_global_aligned | 18.93644 | 5.64054 | 1.01569 |
| t0/committed_global_vs_reference_global_distribution | 10.63741 | 3.97175 | 0.99623 |
| t0/current_global_vs_reference_global_distribution | 0.00000 | 0.00000 | 1.00000 |
| t1/candidate_global_pooled_vs_live_reference_global_aligned | 10.76418 | 3.37946 | 0.99455 |
| t1/candidate_global_slot_0_vs_live_reference_global_aligned | 10.37484 | 3.94465 | 0.99558 |
| t1/candidate_global_slot_1_vs_live_reference_global_aligned | 10.45305 | 3.91292 | 0.99561 |
| t1/candidate_global_slot_2_vs_live_reference_global_aligned | 10.36545 | 3.91949 | 0.99796 |
| t1/candidate_global_slot_3_vs_live_reference_global_aligned | 13.83645 | 2.59129 | 0.98906 |
| t1/committed_global_vs_reference_global_distribution | 13.73672 | 2.61392 | 0.98910 |
| t1/current_global_vs_reference_global_distribution | 10.63742 | 3.97176 | 0.99623 |
| t2/candidate_global_pooled_vs_live_reference_global_aligned | 12.73883 | 2.46298 | 0.98980 |
| t2/candidate_global_slot_0_vs_live_reference_global_aligned | 13.60789 | 2.58218 | 0.98876 |
| t2/candidate_global_slot_1_vs_live_reference_global_aligned | 13.52939 | 2.57222 | 0.98906 |
| t2/candidate_global_slot_2_vs_live_reference_global_aligned | 13.51337 | 2.57702 | 0.99024 |
| t2/candidate_global_slot_3_vs_live_reference_global_aligned | 12.00522 | 2.79295 | 0.99116 |
| t2/committed_global_vs_reference_global_distribution | 11.93961 | 2.78833 | 0.99114 |
| t2/current_global_vs_reference_global_distribution | 13.73670 | 2.61395 | 0.98910 |
| terminal_query_vs_target_query_aligned | 0.36468 | 0.16382 | 1.00000 |

## Retrieval geometry

### t0

- `committed_positive_similarity_mean`: `0.45757`
- `committed_positive_similarity_std`: `0.10971`
- `current_positive_similarity_mean`: `0.57487`
- `current_positive_similarity_std`: `0.14596`

### t1

- `committed_positive_similarity_mean`: `0.49135`
- `committed_positive_similarity_std`: `0.08072`
- `current_positive_similarity_mean`: `0.45757`
- `current_positive_similarity_std`: `0.10971`

### t2

- `committed_positive_similarity_mean`: `0.56514`
- `committed_positive_similarity_std`: `0.06374`
- `current_positive_similarity_mean`: `0.49136`
- `current_positive_similarity_std`: `0.08072`

### terminal

- `terminal_positive_similarity_mean`: `0.56485`
- `terminal_positive_similarity_std`: `0.06413`

## Interpretation rules

- Sibling diversity and global representation health are different questions.
- A low sibling cosine does **not** prove the global embedding space is healthy.
- Strong evidence of recurrent collapse would be a large rank drop, rising cross-sample cosine, or rapidly increasing variance concentration from early to late timesteps.
- Healthy rank but large real-vs-synthetic distribution shift would indicate off-manifold drift without classical dimensional collapse.
- Always compare this report between OLD and STRONG checkpoints before changing the objective.