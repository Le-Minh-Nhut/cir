# Production source coverage manifest

Generated from the frozen revision `8140cc570acaf27fd3d7b72909c8216600402cdb`. Every production Python file under `workbench/` and
`src/` is listed with its content digest at that revision, so a reviewer can confirm the reviewed
bytes are the shipped bytes.

Review coverage by round: R1-R6 (control flow, scientific, data/assets, conda/GPU, security, test
quality), F1-F3 (cross-module, scientific/assets, release integrity), plus the fix agents E1-E4,
V1-V5, W1-W5, X1, Y1-Y2, A1, B1, C1, D1, E1, G1. Scientific correctness returned clean in every
round.

| File | Subsystem | Lines | sha256[:16] | Status |
|---|---|---|---|---|
| `src/evaluation/__init__.py` | legacy evaluation | 0 | `e3b0c44298fc1c14` | REVIEWED_REACHABLE_PATHS |
| `src/evaluation/cirr.py` | legacy evaluation | 203 | `77992a19b3fcce83` | REVIEWED_REACHABLE_PATHS |
| `src/evaluation/cirr_encoder.py` | legacy evaluation | 368 | `7d67a0ccf30c0954` | REVIEWED_REACHABLE_PATHS |
| `src/evaluation/fashioniq.py` | legacy evaluation | 328 | `49f4b048ca2904ca` | REVIEWED_REACHABLE_PATHS |
| `src/evaluation/fashioniq_encoder.py` | legacy evaluation | 149 | `f8ba120dbbe77f22` | REVIEWED_REACHABLE_PATHS |
| `workbench/backend/__init__.py` | backend | 0 | `e3b0c44298fc1c14` | FULLY_REVIEWED |
| `workbench/backend/adapters/__init__.py` | adapters | 0 | `e3b0c44298fc1c14` | FULLY_REVIEWED |
| `workbench/backend/adapters/base.py` | adapters | 43 | `e704de37bfe83e3c` | FULLY_REVIEWED |
| `workbench/backend/adapters/models.py` | adapters | 195 | `daec03077f860305` | FULLY_REVIEWED |
| `workbench/backend/analysis.py` | backend | 93 | `041b09a0a9380eca` | FULLY_REVIEWED |
| `workbench/backend/autopilot.py` | backend | 1199 | `9281a61629a271ad` | FULLY_REVIEWED |
| `workbench/backend/checkpoint_selection.py` | backend | 263 | `92dc74a507d23497` | FULLY_REVIEWED |
| `workbench/backend/environment_planner.py` | backend | 1812 | `0af73cf9d07cc946` | FULLY_REVIEWED |
| `workbench/backend/errors.py` | backend | 14 | `d1d038737778b3e1` | FULLY_REVIEWED |
| `workbench/backend/fashioniq_layout.py` | backend | 140 | `4e8d3b77ec7dca49` | FULLY_REVIEWED |
| `workbench/backend/index.py` | backend | 110 | `6c72f8b8f7b95d6f` | FULLY_REVIEWED |
| `workbench/backend/jobs.py` | backend | 333 | `cd3d1d066c615a4b` | FULLY_REVIEWED |
| `workbench/backend/main.py` | backend | 272 | `7c9911751484fb68` | FULLY_REVIEWED |
| `workbench/backend/metrics_extraction.py` | backend | 320 | `e45524a2239f9429` | FULLY_REVIEWED |
| `workbench/backend/operator_config.py` | backend | 108 | `46585f87a66db836` | FULLY_REVIEWED |
| `workbench/backend/preflight.py` | backend | 131 | `aba473962c259aee` | FULLY_REVIEWED |
| `workbench/backend/registry.py` | backend | 670 | `f9346ea50b65b061` | FULLY_REVIEWED |
| `workbench/backend/runtime.py` | backend | 2113 | `62e8775914affe8b` | FULLY_REVIEWED |
| `workbench/backend/schemas/__init__.py` | backend | 0 | `e3b0c44298fc1c14` | FULLY_REVIEWED |
| `workbench/backend/schemas/results.py` | backend | 127 | `c145a75b38651da8` | FULLY_REVIEWED |
| `workbench/backend/storage.py` | backend | 204 | `392aee56f29e1138` | FULLY_REVIEWED |
| `workbench/replay/__init__.py` | replay | 0 | `e3b0c44298fc1c14` | FULLY_REVIEWED |
| `workbench/replay/common.py` | replay | 100 | `b3c427496ef47809` | FULLY_REVIEWED |
| `workbench/replay/limn.py` | replay | 216 | `161c74038e447e0c` | FULLY_REVIEWED |
| `workbench/replay/tgcir.py` | replay | 95 | `8c75201fc83268ed` | FULLY_REVIEWED |
| `workbench/scripts/autopilot.py` | scripts | 12 | `6cce50aa2774bb11` | FULLY_REVIEWED |
| `workbench/scripts/clean_generated.py` | scripts | 104 | `6fa2ca32f607a64d` | FULLY_REVIEWED |
| `workbench/scripts/doctor.py` | scripts | 318 | `76587b4a67387f0d` | FULLY_REVIEWED |
| `workbench/scripts/download_auxiliary_assets.py` | scripts | 226 | `c44cdf5dee82e446` | FULLY_REVIEWED |
| `workbench/scripts/download_checkpoints.py` | scripts | 233 | `182ba7c31529f2e0` | FULLY_REVIEWED |
| `workbench/scripts/evaluate_models.py` | scripts | 1739 | `80a9ffae0f3c7f2c` | FULLY_REVIEWED |
| `workbench/scripts/load_mock_results.py` | scripts | 31 | `a03be7c4e947b7cb` | FULLY_REVIEWED |
| `workbench/scripts/manage_environment.py` | scripts | 358 | `2492d3780eb65231` | FULLY_REVIEWED |
| `workbench/scripts/pipeline.py` | scripts | 2317 | `8465b352edd23553` | FULLY_REVIEWED |
| `workbench/scripts/plan_environments.py` | scripts | 14 | `4de84777f64e07a3` | FULLY_REVIEWED |
| `workbench/scripts/prepare_dataset.py` | scripts | 133 | `e695e333ffa78f69` | FULLY_REVIEWED |
| `workbench/scripts/prepare_model.py` | scripts | 591 | `1ecdef89fb97f516` | FULLY_REVIEWED |
| `workbench/scripts/readiness_matrix.py` | scripts | 212 | `d8db778a888667d8` | FULLY_REVIEWED |
| `workbench/scripts/rebuild_index.py` | scripts | 35 | `b512573520fa4729` | FULLY_REVIEWED |
| `workbench/scripts/run_eval.py` | scripts | 29 | `8fbae131d3db035f` | FULLY_REVIEWED |
| `workbench/scripts/serve_workbench.py` | scripts | 166 | `8a3990343272a510` | FULLY_REVIEWED |
| `workbench/scripts/sync_upstreams.py` | scripts | 203 | `7327515b0026e6cd` | FULLY_REVIEWED |
| `workbench/scripts/validate_results.py` | scripts | 75 | `dc092e114354a474` | FULLY_REVIEWED |
