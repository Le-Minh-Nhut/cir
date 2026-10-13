# Source coverage manifest

Every production file reviewed in this release cycle, with its content digest at the reviewed
revision, the subsystem it belongs to, who reviewed it, and the review status.

Statuses: `FULLY_REVIEWED`, `REVIEWED_REACHABLE_PATHS`, `EXCLUDED_NONEXECUTABLE_WITH_REASON`,
`NOT_REVIEWED`.

| File | Subsystem | Reviewer(s) | Status |
|---|---|---|---|
| `workbench/backend/autopilot.py` | orchestration | R1, E3, W1 | FULLY_REVIEWED |
| `workbench/backend/environment_planner.py` | environments | R4, W1, X1 | FULLY_REVIEWED |
| `workbench/backend/runtime.py` | runtime/GPU/verify | R4, R5 | FULLY_REVIEWED |
| `workbench/backend/registry.py` | registry/asset validation | R3, W3 | FULLY_REVIEWED |
| `workbench/backend/preflight.py` | readiness | R3, W3, Y2 | FULLY_REVIEWED |
| `workbench/backend/checkpoint_selection.py` | checkpoint policy | R1, R2 | FULLY_REVIEWED |
| `workbench/backend/fashioniq_layout.py` | dataset layouts | R2, R3 | FULLY_REVIEWED |
| `workbench/backend/metrics_extraction.py` | metric parsing | R2 | FULLY_REVIEWED |
| `workbench/backend/analysis.py` | failure analysis | R2, R6 | FULLY_REVIEWED |
| `workbench/backend/index.py` | results index | R5, R6 | FULLY_REVIEWED |
| `workbench/backend/storage.py` | DuckDB queries | R5 | FULLY_REVIEWED |
| `workbench/backend/main.py` | HTTP API | R5, R6 | FULLY_REVIEWED |
| `workbench/backend/jobs.py` | job queue | R5 | FULLY_REVIEWED |
| `workbench/backend/operator_config.py` | path resolution | R4, orchestrator | FULLY_REVIEWED |
| `workbench/backend/errors.py` | error types | orchestrator | FULLY_REVIEWED |
| `workbench/backend/adapters/models.py` | evaluator commands | R1, R2, orchestrator | FULLY_REVIEWED |
| `workbench/backend/adapters/base.py` | adapter contract | R1 | FULLY_REVIEWED |
| `workbench/backend/schemas/results.py` | result schema | R2, R6 | FULLY_REVIEWED |
| `workbench/scripts/pipeline.py` | stage DAG/resume | R1, E3, W5, Y1 | FULLY_REVIEWED |
| `workbench/scripts/evaluate_models.py` | evaluation/ledger | R2, R5, Y2 | FULLY_REVIEWED |
| `workbench/scripts/prepare_model.py` | preparation | R3, R5, W2 | FULLY_REVIEWED |
| `workbench/scripts/prepare_dataset.py` | dataset link | R3 | FULLY_REVIEWED |
| `workbench/scripts/download_checkpoints.py` | checkpoint acquisition | R5, W4 | FULLY_REVIEWED |
| `workbench/scripts/download_auxiliary_assets.py` | asset acquisition | R3, R5, W3 | FULLY_REVIEWED |
| `workbench/scripts/sync_upstreams.py` | source pins | R5, E4 | FULLY_REVIEWED |
| `workbench/scripts/manage_environment.py` | env CLI | R4, W1 | FULLY_REVIEWED |
| `workbench/scripts/plan_environments.py` | env planning CLI | R4 | FULLY_REVIEWED |
| `workbench/scripts/doctor.py` | readiness CLI | R4, R6 | FULLY_REVIEWED |
| `workbench/scripts/readiness_matrix.py` | registry matrix | orchestrator | FULLY_REVIEWED |
| `workbench/scripts/clean_generated.py` | deletion safety | R5, orchestrator | FULLY_REVIEWED |
| `workbench/scripts/rebuild_index.py` | index rebuild | R5 | FULLY_REVIEWED |
| `workbench/scripts/serve_workbench.py` | server launch | R6 | FULLY_REVIEWED |
| `workbench/scripts/validate_results.py` | result validation | R2, R6 | FULLY_REVIEWED |
| `workbench/scripts/load_mock_results.py` | mock loader | R6 | REVIEWED_REACHABLE_PATHS |
| `workbench/scripts/run_eval.py` | legacy shim | orchestrator | REVIEWED_REACHABLE_PATHS |
| `workbench/scripts/autopilot.py` | autopilot CLI | R1, E4 | FULLY_REVIEWED |
| `workbench/replay/limn.py` | LIMN native replay | R2, R3 | FULLY_REVIEWED |
| `workbench/replay/common.py` | replay helpers | R2, R3 | FULLY_REVIEWED |
| `workbench/replay/tgcir.py` | TG-CIR preflight | R2, R3 | FULLY_REVIEWED |
| `workbench/run.sh` | launcher | R1 | FULLY_REVIEWED |
| `workbench/registry/models.yaml` | model registry | R2, R4 | FULLY_REVIEWED |
| `workbench/registry/auxiliary_assets.yaml` | asset registry | R3, W3 | FULLY_REVIEWED |
| `workbench/registry/preparation_contracts.yaml` | preparation contracts | R3, W2 | FULLY_REVIEWED |
| `workbench/registry/assets/cluster_b1_csmcir_limn.yaml` | asset audit | B1, R3 | FULLY_REVIEWED |
| `workbench/registry/assets/cluster_b2_encoder_family.yaml` | asset audit | B2, R3 | FULLY_REVIEWED |
| `workbench/registry/assets/cluster_b3_ilearn.yaml` | asset audit | B3, R3 | FULLY_REVIEWED |
| `workbench/registry/assets/cluster_b4_classic.yaml` | asset audit | B4, R3 | FULLY_REVIEWED |
| `workbench/registry/assets/cluster_b5_ptha_mtst.yaml` | asset audit | orchestrator | FULLY_REVIEWED |
| `workbench/configs/pc_autopilot.pilot_csmcir.yaml` | first-GPU-pilot | E4, R6 | FULLY_REVIEWED |
| `workbench/configs/pc_autopilot.all_available.yaml` | full-cohort profile | E4, R6 | FULLY_REVIEWED |
| `workbench/configs/pc_autopilot.example.yaml` | template | E4, R6 | FULLY_REVIEWED |
| `src/evaluation/fashioniq.py` | original-split protocol | R2 | REVIEWED_REACHABLE_PATHS |
| `src/evaluation/fashioniq_encoder.py` | val-split protocol | R2 | REVIEWED_REACHABLE_PATHS |
| `src/evaluation/cirr.py` | CIRR (non-FashionIQ) | R2 | EXCLUDED_NONEXECUTABLE_WITH_REASON |
| `src/evaluation/cirr_encoder.py` | CIRR encoder (non-FashionIQ) | R2 | EXCLUDED_NONEXECUTABLE_WITH_REASON |

## Upstream evaluator paths

The pinned checkouts are *not* committed (they are gitignored and acquired by
`sync_upstreams.py` at a recorded SHA). Their reachable evaluation paths were reviewed via the
per-cluster audits in `workbench/registry/assets/`, which cite file and line for every asset read.
Each audit records the exact entrypoint, its call chain, and the assets it opens.

| Cluster | Repos | Models | Reviewer |
|---|---|---|---|
| B1 | CSMCIR, LIMN | csmcir, limn | B1, R2, R3 |
| B2 | HINT, ENCODER, PAIR, TGCIR | hint, encoder, pair, tgcir | B2, R2 |
| B3 | AirKnow, ConeSep, HABIT, INTENT | airknow, conesep, habit, intent | B3, R2 |
| B4 | DCNet, CLVCNet, SPRC, CLIP4Cir | dcnet, clvc_net, sprc, combiner_rn50x4_noft, clip4cir_rn50x4_fullft | B4, R2, R3 |
| B5 | (no source) | ptha_mtst | orchestrator |

`src/evaluation/cirr*.py` are excluded because they implement the CIRR benchmark, not FashionIQ,
and no registered model's native protocol is CIRR. They are retained as the upstream reference for
the CIRR protocol family only.

## Reviewer assignments

| Reviewer | Specialty | Independence |
|---|---|---|
| R1 | execution and control flow | reviewed the integrated tree, no part in the fixes |
| R2 | scientific correctness | reviewed the integrated tree, no part in the fixes |
| R3 | data and external assets | reviewed the integrated tree, no part in the fixes |
| R4 | conda / GPU / dependencies | reviewed the integrated tree, no part in the fixes |
| R5 | security / filesystem / concurrency | reviewed the integrated tree, no part in the fixes |
| R6 | test quality and release claims | reviewed the integrated tree, no part in the fixes |
