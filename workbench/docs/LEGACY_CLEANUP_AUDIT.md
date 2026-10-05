# Legacy custom-architecture cleanup audit

Audit date: 2026-10-06. Scope: tracked executable source, configs, tests, root documentation, and history for candidate paths. `git log -- <path>` confirms the candidate surface predates the workbench and belongs to TAPER/CSMCIR internal experiments. Historical Markdown is retained unless its commands describe deleted executables.

## KEEP

- `src/datasets/fashioniq.py`, `src/datasets/common.py`, `src/evaluation/fashioniq.py`, `src/evaluation/fashioniq_encoder.py`, and their direct generic dependency `src/cache/features.py`: canonical FashionIQ loading and both protocol families. Workbench protocol tests import these modules.
- `src/datasets/cirr.py`, `src/evaluation/cirr.py`, `src/evaluation/cirr_encoder.py`, related CIRR metric tests, and generic dataset/image helpers: generic benchmark utilities rather than confirmed custom-architecture implementations.
- `workbench/`: current research tool and all its tests, registry, scripts, protocol docs, and mock generator.
- `CLEANUP_REPORT.md`: research-history record. It is amended only where it would otherwise claim removed code remains current.

## DELETE

- `src/models/taper.py`, `src/models/__init__.py`; `src/teachers/csmcir.py`, `src/teachers/csmcir_compose.py`, `src/teachers/__init__.py`; `src/training/engine.py`, `src/training/__init__.py`.
  - Reason: TAPER model, frozen CSMCIR teacher wrappers, and its training loop are one coherent abandoned internal architecture.
  - References: all references are in the TAPER scripts/configs/tests listed below, root README, and prior cleanup report; no workbench or canonical FashionIQ protocol module imports them.
  - Safe removal: workbench executes official upstream repositories by subprocess and preserves generic FashionIQ evaluators.
- `src/train.py`, `src/train_cirr.py`, `src/train_cirr_encoder.py`, `src/train_fashioniq_encoder.py`, `src/evaluate_cirr_encoder.py`, `src/evaluate_fashioniq_encoder.py`, `src/predict_cirr_encoder_test.py`.
  - Reason: TAPER-only train/evaluation/submission entrypoints.
  - References: root README, `CLEANUP_REPORT.md`, `tests/unit/test_encoder_mode_isolation.py`, and `tests/unit/test_fashioniq_encoder.py`; tests are updated to test protocol behavior directly.
  - Safe removal: neither workbench adapters nor canonical FashionIQ protocol modules invoke local model entrypoints.
- `src/precompute_cirr.py`, `src/precompute_csmcir_stage1.py`, `src/precompute_taper_e2e_text.py`, `src/check_csmcir_compose_parity.py`, `src/check_taper_chunk_parity.py`, `src/check_taper_text_cache_parity.py`.
  - Reason: CSMCIR feature-cache and TAPER parity tooling for the deleted architecture.
  - References: root README and `CLEANUP_REPORT.md` only outside that coherent surface.
  - Safe removal: workbench does not create model feature caches or import external CSMCIR code.
- `conf/experiment/taper_e2e.yaml`, `conf/experiment/taper_e2e_ortho07.yaml`, `tests/unit/test_taper_slot_query_ortho.py`.
  - Reason: TAPER hyperparameters and architecture-only orthogonal-loss test.
  - References: deleted entrypoints, root README, and prior cleanup report.
  - Safe removal: active FashionIQ protocol definitions live in workbench registry and protocol audit; no current executable selects legacy Hydra configs.
- TAPER-dependent fragments in `tests/unit/test_cirr_encoder.py`, `tests/unit/test_encoder_mode_isolation.py`, and `tests/unit/test_fashioniq_encoder.py`.
  - Reason: TAPER score equivalence, import-isolation, and training-constant checks; replace only with generic protocol assertions.
  - Safe removal: retained tests still cover CIRR and FashionIQ protocol metrics.
- `conf/config.yaml`, `conf/dataset/`, `conf/protocol/`, `conf/experiment/default.yaml`, and `src/runtime.py`.
  - Reason: only configured or supported removed custom-model workflows.
  - Safe removal: no retained source imports or reads them.

## UNCERTAIN

- `src/cache/features.py`: supports generic evaluators as well as deleted training code. Keep.
- `src/datasets/cirr.py`, `src/evaluation/cirr.py`, `src/evaluation/cirr_encoder.py`, and CIRR metric tests: generic benchmark support with possible future reuse. Keep.
- `teacher/README.md`: historical external-teacher contract. Keep as research documentation even though its executable consumers are deleted.
