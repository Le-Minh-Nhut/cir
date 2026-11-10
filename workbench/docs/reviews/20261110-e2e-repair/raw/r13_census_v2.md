# Census v2 — CORRECTION of the earlier census

v1 (`raw/native_measurement_census.txt`) scanned only `**/measurements/*.json` and only the shape
`aggregate.macro_r10`. That missed: `pair_compat/measurements/`, `macro.macro_r10`,
`macro_metrics`, top-level `macro_r10`, and `clip4cir_parsed.json`. v1 undercounted.

v2 scans every JSON under checkpoint_selection/ (excluding checkpoints/downloads/site-packages)
and looks for macro_r10/macro_r50 at any depth.

[OWN]      conesep_fiq_n02.json             mid=conesep                {'aggregate.macro_r10': 54.76130615726771, 'aggregate.macro_r50': 75.21514359343843}
[OWN]      encoder_fashioniq.json           mid=encoder                {'aggregate.macro_r10': 56.223101675386054, 'aggregate.macro_r50': 77.66738539271432}
[OWN]      habit_fiq_n02.json               mid=habit                  {'aggregate.macro_r10': 54.87332239571265, 'aggregate.macro_r50': 74.50337396905957}
[OWN]      hint_fashioniq.json              mid=hint                   {'aggregate.macro_r10': 59.80338494972219, 'aggregate.macro_r50': 80.13076097500765}
[OWN]      intent_fiq_n02.json              mid=intent                 {'aggregate.macro_r10': 54.86624335289025, 'aggregate.macro_r50': 75.43191079283865}
[OWN]      limn_base_iter0.json             mid=limn                   {'aggregate.macro_r10': 55.79193333333333, 'aggregate.macro_r50': 77.6378}
[OWN]      tme_fashioniq_0.json             mid=tme                    {'aggregate.macro_r10': 54.922901590665184, 'aggregate.macro_r50': 74.83842174212138}
[OWN]      airknow_fiq_n05_run1.json        mid=airknow                {'aggregate.macro_r10': 52.54661498980954, 'aggregate.macro_r50': 72.9026186457433}
[OWN]      airknow_fiq_n08_run1.json        mid=airknow                {'aggregate.macro_r10': 50.64166565555885, 'aggregate.macro_r50': 71.45559570455528}
[roll-up]  all_measured_summary.json        mid=None                   {'airknow_n05.macro.macro_r10': 52.5466, 'airknow_n05.macro.macro_r50': 72.9026, 'airknow_n08.macro.macro_r10': 50.6417, 'airknow_n08.macro.macro_r50': 71.4556, 'clvc.macro.macro_r10': 29.9707, 'clvc.macro.macro_r50': 58.6217, 'tgcir.macro.macro_r10': 49.7208, 'tgcir.macro.macro_r50': 71.1023}
[roll-up]  clip4cir_parsed.json             mid=None                   {'clip4cir_noft.macro_r10': 35.14068226019541, 'clip4cir_noft.macro_r50': 58.44364364941915, 'clip4cir_fullft.macro_r10': 43.81821850935618, 'clip4cir_fullft.macro_r50': 66.95005893707275}
[OWN]      clvc_run1.json                   mid=clvc_net               {'aggregate.macro_r10': 29.970746517160162, 'aggregate.macro_r50': 58.62171212901515}
[OWN]      dcnet_run1.json                  mid=dcnet                  {'aggregate.macro_r10': 27.929730914548923, 'aggregate.macro_r50': 53.72723331879208, 'paper.macro_r10': 27.78, 'paper.macro_r50': 53.89}
[roll-up]  selection_decisions.json         mid=None                   {'selected_2026-11-06.airknow.metrics.macro_r10': 52.5466, 'selected_2026-11-06.airknow.metrics.macro_r50': 72.9026, 'selected_2026-11-06.airknow_variant_80.metrics.macro_r10': 50.6417, 'selected_2026-11-06.airknow_variant_80.metrics.macro_r50': 71.4556, 'selected_2026-11-06.combiner_rn50x4_noft.metrics.macro_r10': 35.1407, 'selected_2026-11-06.combiner_rn50x4_noft.metrics.macro_r50': 58.4436, 'selected_2026-11-06.clip4cir_rn50x4_fullft.metrics.macro_r10': 43.8182, 'selected_2026-11-06.clip4cir_rn50x4_fullft.metrics.macro_r50': 66.9501, 'selected_2026-11-06.dcnet.metrics.macro_r10': 27.9297, 'selected_2026-11-06.dcnet.metrics.macro_r50': 53.7272, 'selected_2026-11-06.clvc_net.metrics.macro_r10': 29.9707, 'selected_2026-11-06.clvc_net.metrics.macro_r50': 58.6217, 'selected_2026-11-06.tgcir.metrics.macro_r10': 49.7208, 'selected_2026-11-06.tgcir.metrics.macro_r50': 71.1023, 'selected_2026-11-06.sprc_vitl.metrics.macro_r10': 51.0386, 'selected_2026-11-06.sprc_vitl.metrics.macro_r50': 73.3263}
[OWN]      sprc_vitl_run1.json              mid=sprc                   {'aggregate.macro_r10': 51.038571198781334, 'aggregate.macro_r50': 73.32633932431538}
[OWN]      tgcir_run1.json                  mid=tgcir                  {'aggregate.macro_r10': 49.720772235486656, 'aggregate.macro_r50': 71.10231942031965}
[roll-up]  pair_compat_verdict.json         mid=None                   {'measured.B1/H_rev.macro_r10': 35.0147, 'measured.B1/H_rev.macro_r50': 57.5753, 'measured.B1/H_normal.macro_r10': 34.035, 'measured.B1/H_normal.macro_r50': 56.5957, 'measured.B2/H_rev.macro_r10': 20.0868, 'measured.B2/H_rev.macro_r50': 41.4988, 'measured.B2/H_normal.macro_r10': 22.1564, 'measured.B2/H_normal.macro_r50': 43.4864, 'paper_reference.macro_r10': 52.76, 'paper_reference.macro_r50': 74.51}
[OWN]      pair_b1_H_normal.json            mid=None                   {'macro.macro_r10': 34.03503453772374, 'macro.macro_r50': 56.59567753134161, 'paper_reference.macro_r10': 52.76, 'paper_reference.macro_r50': 74.51}
[OWN]      pair_b1_H_rev.json               mid=None                   {'macro.macro_r10': 35.014736041655844, 'macro.macro_r50': 57.57531573029066, 'paper_reference.macro_r10': 52.76, 'paper_reference.macro_r50': 74.51}
[OWN]      pair_b2_H_normal.json            mid=None                   {'macro.macro_r10': 22.156393011046543, 'macro.macro_r50': 43.486402136659365, 'paper_reference.macro_r10': 52.76, 'paper_reference.macro_r50': 74.51}
[OWN]      pair_b2_H_rev.json               mid=None                   {'macro.macro_r10': 20.086769084793715, 'macro.macro_r50': 41.49879132733318, 'paper_reference.macro_r10': 52.76, 'paper_reference.macro_r50': 74.51}
[roll-up]  summary.json                     mid=None                   {'B1/H_rev.macro_r10': 35.0147, 'B1/H_rev.macro_r50': 57.5753, 'B1/H_normal.macro_r10': 34.035, 'B1/H_normal.macro_r50': 56.5957, 'B2/H_rev.macro_r10': 20.0868, 'B2/H_rev.macro_r50': 41.4988, 'B2/H_normal.macro_r10': 22.1564, 'B2/H_normal.macro_r50': 43.4864}
[OWN]      hint_doc_command.json            mid=hint                   {'aggregate.macro_r10': 59.80338494972219, 'aggregate.macro_r50': 80.13076097500765}
[OWN]      tme_normal_path_full.json        mid=tme                    {'aggregate.macro_r10': 54.922901590665184, 'aggregate.macro_r50': 74.83842174212138}
[OWN]      measurement.json                 mid=csmcir                 {'macro_r10': 57.05073873202006, 'macro_r50': 77.29859352111816}
