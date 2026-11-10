# pair / ptha / clip4cir — no macro check

## pair_b1_H_normal.json -> workbench/artifacts/checkpoint_selection/20261106T002617+0700/pair_compat/measurements/pair_b1_H_normal.json
keys: ['batch_size', 'candidate', 'checkpoint', 'class_mapping', 'complete', 'device', 'hypothesis', 'macro', 'mapping_status', 'mode', 'model', 'paper_reference', 'per_category', 'precision', 'protocol', 'protocol_note', 'torch']
macro: {"macro_r1": 11.429797333174463, "macro_r10": 34.03503453772374, "macro_r50": 56.59567753134161, "macro_mean": 45.31535603453267}

## ptha_transfer_run1.json -> workbench/artifacts/checkpoint_selection/20261106T233409+0700/measurements/ptha_transfer_run1.json
keys: ['backbone_config', 'category_metrics', 'checkpoint_id', 'checkpoint_sha256', 'checkpoint_top_level_model_key', 'checkpoint_training_noise', 'evaluation_dataset', 'label', 'load_report', 'macro_metrics', 'model_id', 'model_name', 'paper_fashioniq_context_only', 'protocol_id', 'serialization_inspection', 'target_feature_contract', 'train_dataset']
category_metrics: {"dress": {"r10": 33.11849236488342, "r50": 56.02380037307739}, "shirt": {"r10": 38.861629366874695, "r50": 59.02845859527588}, "toptee": {"r10": 41.71341061592102, "r50": 64.45690989494324}}

## clip4cir_parsed.json -> workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clip4cir_parsed.json
keys: ['clip4cir_fullft', 'clip4cir_noft']
clip4cir_fullft: {"dress_r10": 39.11750018596649, "dress_r50": 64.15468454360962, "shirt_r10": 44.50441598892212, "shirt_r50": 65.4563307762146, "toptee_r10": 47.83273935317993, "toptee_r50": 71.23916149139404, "macro_r10": 43.81821850935618, "macro_r50": 6

## pair_blocker.json -> workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/pair_blocker.json
keys: ['candidates', 'investigation', 'model_id', 'pickle_globals_required', 'structural_facts', 'verdict']
