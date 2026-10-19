# Test Suite Production Parity and Verification Copy Audit

## 1. Candidate Test Files Inventory and Classification

Classification schema:
- **(a) Production Entry Point**: Calls production code imported from `workbench/...`.
- **(b) Test Copy / Re-implementation**: Re-implements logic under test within the test file and asserts against that local copy.
- **(c) Source-Text Assertion**: Asserts on raw source text or document text via regex, string matching, or AST scan.
- **(d) Fixture / Constant**: Asserts only on constants, schemas, or static test fixture dictionaries without executing production logic.

### 1.1 `workbench/tests/test_f05_f08_f09_verification.py` (11 tests)
- `workbench/tests/test_f05_f08_f09_verification.py:26` `test_f05_hint_constructor_reads_train_captions_for_all_categories`: **(c)** Reads `workbench/third_party/HINT/datasets.py` text; asserts `"def train_init_process(self):"` in content.
- `workbench/tests/test_f05_f08_f09_verification.py:34` `test_f05_encoder_constructor_reads_train_captions_for_all_categories`: **(c)** Reads `workbench/third_party/ENCODER/datasets.py` text; asserts train initialization process.
- `workbench/tests/test_f05_f08_f09_verification.py:42` `test_f05_pair_constructor_reads_train_captions_for_all_categories`: **(c)** Reads `workbench/third_party/PAIR/data/datasets.py` text; asserts train initialization process.
- `workbench/tests/test_f05_f08_f09_verification.py:50` `test_f05_pair_orchestrator_contract_already_requires_train`: **(c)** Reads `workbench/registry/preparation_contracts.yaml` text; asserts caption path entries.
- `workbench/tests/test_f05_f08_f09_verification.py:58` `test_f05_isolated_cache_root_starts_fresh_without_cached_tensors`: **(a)** Imports and executes `isolated_cache_root` from `workbench.replay.common`.
- `workbench/tests/test_f05_f08_f09_verification.py:78` `test_f05_extended_validator_detects_missing_train_captions`: **(b)** Defines ad-hoc validator `validate_dataset_with_train` locally at line 79; asserts against that local function.
- `workbench/tests/test_f05_f08_f09_verification.py:117` `test_f08_dcnet_config_data_dir_resolution`: **(d)** Parses `workbench/third_party/DCNet/configs/ce/fashioniq_dcnet.json` and evaluates raw path arithmetic.
- `workbench/tests/test_f05_f08_f09_verification.py:132` `test_f08_preflight_refusal_on_data_dir_mismatch`: **(b)** Defines ad-hoc validator `validate_dcnet_data_dir_alignment` locally at line 133; asserts against that local function.
- `workbench/tests/test_f05_f08_f09_verification.py:279` `test_f09_dcnet_parser_extracts_combined_metrics_and_multiplies_by_100`: **(b)** Calls local function `parse_dcnet_native_output` defined at line 174; never invokes production parser.
- `workbench/tests/test_f05_f08_f09_verification.py:308` `test_f09_dcnet_parser_fallback_for_single_branch`: **(b)** Calls local function `parse_dcnet_native_output` defined at line 174.
- `workbench/tests/test_f05_f08_f09_verification.py:320` `test_f09_dcnet_parser_rejects_inconsistent_macro_mean`: **(b)** Calls local function `parse_dcnet_native_output` defined at line 174.

### 1.2 `workbench/tests/test_verified_declarations.py` (15 tests)
- `workbench/tests/test_verified_declarations.py:51` `test_f05_a_val_only_root_is_reported_as_missing_train`: **(a)** Dynamically imports `workbench.replay.{hint,encoder,pair}` and executes `module.validate_dataset`.
- `workbench/tests/test_verified_declarations.py:67` `test_f05_the_declared_layout_requires_the_train_paths`: **(a)** Calls `model_by_id` and `paths_for_layout` from `workbench.backend.fashioniq_layout`.
- `workbench/tests/test_verified_declarations.py:79` `test_f05_the_native_constructors_really_read_train`: **(c)** Greps text of third-party source files (`HINT/datasets.py`, `ENCODER/datasets.py`, `PAIR/data/datasets.py`).
- `workbench/tests/test_verified_declarations.py:89` `test_f07_tgcir_pins_the_module_its_evaluator_imports`: **(c)** Greps `TGCIR/model.py` source text for `clip.load(`; checks registry pin dictionary.
- `workbench/tests/test_verified_declarations.py:119` `test_f09_dcnet_fractions_are_converted_once`: **(a)** Calls `extract_aggregate_metrics("dcnet", ...)` from `workbench.backend.metrics_extraction`.
- `workbench/tests/test_verified_declarations.py:129` `test_f09_the_combined_branch_is_the_one_read`: **(a)** Calls production `parse_dcnet_output` from `workbench.backend.metrics_extraction`.
- `workbench/tests/test_verified_declarations.py:135` `test_f09_composition_and_correction_are_not_merged_into_the_combined_result`: **(a)** Calls production `parse_dcnet_output`.
- `workbench/tests/test_verified_declarations.py:143` `test_f09_the_two_cell_branch_is_accepted_only_when_complete`: **(a)** Calls production `parse_dcnet_output`.
- `workbench/tests/test_verified_declarations.py:155` `test_f09_a_stream_that_contradicts_its_own_average_is_refused`: **(a)** Calls production `parse_dcnet_output`.
- `workbench/tests/test_verified_declarations.py:163` `test_f09_ranges_and_completeness_still_hold`: **(a)** Calls production `parse_dcnet_output` and `extract_aggregate_metrics`.
- `workbench/tests/test_verified_declarations.py:171` `test_f10_the_encoder_checkpoint_is_pinned_to_an_author_revision`: **(d)** Calls `model_by_id("encoder")` and asserts registry metadata fields.
- `workbench/tests/test_verified_declarations.py:185` `test_f10_the_loader_reports_what_strict_false_hid`: **(c)** Reads `workbench/replay/encoder.py` text and asserts substring presence.
- `workbench/tests/test_verified_declarations.py:197` `test_f14_sprc_backbone_is_declared_not_inferred`: **(d)** Asserts metadata keys in `model_by_id("sprc")`.
- `workbench/tests/test_verified_declarations.py:207` `test_f14_the_adapter_uses_the_declared_backbone`: **(a)** Invokes `ADAPTERS["sprc"]().command(...)` with mocked Python executable.
- `workbench/tests/test_verified_declarations.py:220` `test_f14_an_undeclared_backbone_is_refused_rather_than_guessed`: **(a)** Invokes `ADAPTERS["sprc"]().command(...)` with invalid variant.

### 1.3 `workbench/tests/test_handoff_commands.py` (7 tests)
- `workbench/tests/test_handoff_commands.py:160` `test_every_document_in_the_handoff_set_has_run_commands_or_says_why_not`: **(c)** Globs `workbench/docs/*.md` text; asserts shell blocks are either audited or in hardcoded `exempt` set.
- `workbench/tests/test_handoff_commands.py:197` `test_every_documented_command_parses_and_pipeline_has_required_authorization`: **(a)** Extracts shell commands from markdown and parses them via production argument parser `build_arg_parser`.
- `workbench/tests/test_handoff_commands.py:346` `test_no_release_document_points_at_a_superseded_source_sha`: **(c)** Greps commit hashes in `FINAL_PC_RELEASE_VERDICT.md` and `PC_PILOT_HANDOFF.md`; runs `git diff`.
- `workbench/tests/test_handoff_commands.py:442` `test_a_release_document_cannot_quote_a_production_tree_that_never_ran_the_gates`: **(c)** Executes `git show` against older git commits and greps strings.
- `workbench/tests/test_handoff_commands.py:474` `test_the_release_documents_quote_the_test_count_they_actually_ran`: **(c)** Regex greps `**(\d+) passed, 0 failed**` in `FINAL_PC_RELEASE_VERDICT.md`; compares against `pytest --collect-only`.
- `workbench/tests/test_handoff_commands.py:513` `test_no_release_document_describes_verify_only_as_verifying_a_run`: **(c)** Regex greps prose containing `--verify-only` across release documentation.
- `workbench/tests/test_handoff_commands.py:535` `test_the_verdict_names_the_same_test_count_the_handoff_does`: **(c)** Compares regex counts between `FINAL_PC_RELEASE_VERDICT.md` and `PC_PILOT_HANDOFF.md`.

### 1.4 `workbench/tests/test_dcnet_replay.py` (4 tests)
- `workbench/tests/test_dcnet_replay.py:19` `test_dcnet_protocol_and_model_id`: **(d)** Asserts imported module constants (`MODEL`, `PROTOCOL`, `REQUIRED_RUN_FILES`).
- `workbench/tests/test_dcnet_replay.py:26` `test_dcnet_validate_source_pinned`: **(a)** Calls `workbench.replay.dcnet.validate_source`.
- `workbench/tests/test_dcnet_replay.py:32` `test_dcnet_run_dir_fails_on_missing_files`: **(a)** Calls `workbench.replay.dcnet.validate_run_directory`.
- `workbench/tests/test_dcnet_replay.py:40` `test_dcnet_dataset_fails_on_missing_glove_and_resized_images`: **(a)** Calls `workbench.replay.dcnet.validate_dataset`.

### 1.5 `workbench/tests/test_csmcir_source_blocker.py` (4 tests)
- `workbench/tests/test_csmcir_source_blocker.py:45` `test_the_entrypoint_really_imports_the_model_package`: **(c)** Greps source text of `workbench/third_party/CSMCIR/src/validate_blip_csmcir.py`.
- `workbench/tests/test_csmcir_source_blocker.py:54` `test_the_model_package_references_files_that_do_not_exist`: **(c)** Parses imports in `workbench/third_party/CSMCIR/src/lavis/models/__init__.py`; checks disk existence.
- `workbench/tests/test_csmcir_source_blocker.py:68` `test_the_missing_imports_are_not_guarded`: **(c)** Greps lines preceding registration in `CSMCIR/src/lavis/models/__init__.py` for `try:`/`except`.
- `workbench/tests/test_csmcir_source_blocker.py:78` `test_the_needed_class_is_the_one_registered_under_the_expected_name`: **(c)** Greps `CSMCIR/src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py` for decorator string.

### 1.6 `workbench/tests/test_hint_replay.py` (4 tests)
- `workbench/tests/test_hint_replay.py:32` `test_hint_stale_cache_in_canonical_root_not_reused`: **(a)** Calls `workbench.replay.hint.evaluate_categories`.
- `workbench/tests/test_hint_replay.py:73` `test_hint_clean_root_works`: **(a)** Calls `workbench.replay.hint.evaluate_categories`.
- `workbench/tests/test_hint_replay.py:83` `test_hint_canonical_root_left_unmodified`: **(a)** Calls `workbench.replay.hint.evaluate_categories`.
- `workbench/tests/test_hint_replay.py:119` `test_hint_isolated_root_lives_through_evaluation`: **(a)** Calls `workbench.replay.hint.evaluate_categories`.

### 1.7 `workbench/tests/test_hint_metric_parser.py` (11 tests)
- `workbench/tests/test_hint_metric_parser.py:58` `test_a_valid_wrapper_result_is_parsed`: **(a)** Calls `workbench.backend.metrics_extraction.extract_aggregate_metrics`.
- `workbench/tests/test_hint_metric_parser.py:67` `test_the_native_evaluator_print_is_parsed`: **(a)** Calls `workbench.backend.metrics_extraction.extract_aggregate_metrics`.
- `workbench/tests/test_hint_metric_parser.py:74` `test_all_three_categories_are_required`: **(a)** Calls `workbench.backend.metrics_extraction.parse_hint_output`.
- `workbench/tests/test_hint_metric_parser.py:82` `test_recall_values_must_be_numeric_and_in_range`: **(a)** Calls `workbench.backend.metrics_extraction.parse_hint_output`.
- `workbench/tests/test_hint_metric_parser.py:90` `test_a_declared_macro_that_disagrees_with_its_categories_is_rejected`: **(a)** Calls `workbench.backend.metrics_extraction.parse_hint_output`.
- `workbench/tests/test_hint_metric_parser.py:98` `test_malformed_and_empty_output_is_rejected`: **(a)** Calls `workbench.backend.metrics_extraction.parse_hint_output`.
- `workbench/tests/test_hint_metric_parser.py:105` `test_a_failed_process_yields_no_metrics`: **(a)** Calls `workbench.backend.metrics_extraction.extract_aggregate_metrics`.
- `workbench/tests/test_hint_metric_parser.py:111` `test_the_macro_is_recomputed_not_copied`: **(a)** Calls `workbench.backend.metrics_extraction.parse_hint_output`.
- `workbench/tests/test_hint_metric_parser.py:124` `test_a_result_embedded_in_other_output_is_found`: **(a)** Calls `workbench.backend.metrics_extraction.parse_hint_output`.
- `workbench/tests/test_hint_metric_parser.py:131` `test_paper_metrics_produce_a_parity_verdict_not_a_substitution`: **(a)** Calls `workbench.backend.metrics_extraction.parse_hint_output`.
- `workbench/tests/test_hint_metric_parser.py:145` `test_adding_hint_did_not_disturb_the_existing_parsers`: **(a)** Calls `workbench.backend.metrics_extraction.extract_aggregate_metrics`.

### 1.8 `workbench/tests/test_release_blocker_*.py` (10 files, 297 tests)
All 297 tests across the 10 release blocker files are classified as **(a) Production Entry Point**. They import and execute production functions from `workbench.backend.runtime`, `workbench.backend.jobs`, `workbench.backend.operator_config`, `workbench.backend.paper_gate`, `workbench.backend.registry`, and `workbench.scripts.evaluate_models`:
- `workbench/tests/test_release_blocker_source_integrity.py` (18 tests): Lines 72-405: **(a)**. Calls `source_provenance`, `source_clean_and_pinned`, `_source_dirty`, `sha256_file`.
- `workbench/tests/test_release_blocker_evaluation_output.py` (68 tests): Lines 133-1655: **(a)**. Calls `evaluate_models`, ledger verification, output contracts.
- `workbench/tests/test_release_blocker_gpu_records.py` (31 tests): Lines 247-1261: **(a)**. Calls runtime corroboration, GPU readiness, and release verification.
- `workbench/tests/test_release_blocker_cuda_readiness.py` (33 tests): Lines 117-946: **(a)**. Calls `verify_gpu_readiness`, `inspect_environment`, `SingleGpuQueue`.
- `workbench/tests/test_release_blocker_invocation_proof.py` (40 tests): Lines 257-1154: **(a)**. Calls `verify_run`, `SingleGpuQueue`, `paper_gate`.
- `workbench/tests/test_release_blocker_master_e2e.py` (16 tests): Lines 165-592: **(a)**. Calls `run_pc`, sandbox harness, execution graph.
- `workbench/tests/test_release_blocker_adversarial_final.py` (12 tests): Lines 83-498: **(a)**. Calls release verification under corrupted logs and missing proofs.
- `workbench/tests/test_release_blocker_production_stage_fingerprint.py` (13 tests): Lines 88-287: **(a)**. Calls stage resumption and input hashing.
- `workbench/tests/test_release_blocker_mutations.py` (13 tests): Lines 112-620: **(a)**. Calls `python_executable`, `check_environment_isolation`, and executes live mutation checks against production modules.
- `workbench/tests/test_release_blocker_interpreter.py` (51 tests): Lines 91-1175: **(a)**. Calls `inspect_environment`, `require_verified_model_interpreter`, `version_satisfies`.

### 1.9 Test Files Added in Recent Commits (commit `119df5f`, 5 tests)
- `workbench/tests/test_blip2_text_processor_parity.py:20` `test_all_five_models_resolve_identical_blip_caption_eval_processor`: **(d)** Parses YAML configs in `workbench/third_party/<model>/lavis/configs/models/blip2/blip2_pretrain.yaml`.
- `workbench/tests/test_blip2_text_processor_parity.py:39` `test_the_five_wrappers_agree_on_the_text_processor_the_configs_declare`: **(c)** Parses YAML configs and greps `blip_processors.py` text for class and method names.
- `workbench/tests/test_encoder_pair_clvc_verification.py:24` `test_f10_encoder_checkpoint_schema_and_verification`: **(b)** Defines ad-hoc function `verify_encoder_checkpoint_keys` locally at line 28; asserts against that local copy.
- `workbench/tests/test_encoder_pair_clvc_verification.py:86` `test_f11_clvc_net_assumed_names_and_whole_model_loader`: **(a)** Calls `workbench.replay.common.load_trusted_whole_model`.
- `workbench/tests/test_encoder_pair_clvc_verification.py:101` `test_f12_pair_reconstruction_labeling_and_behavior`: **(c)** Greps `MHTransformer_rev.py` header for `"EXPERIMENTAL_RECONSTRUCTION"`; asserts attention matrix numerical inversion.

---

## 2. Concrete List: Tests That Would STILL PASS Under Reintroduced Defects

| Test Name (`file:line`) | Exact Reintroduction Code Change | Why Test Stays Green |
|---|---|---|
| `workbench/tests/test_f05_f08_f09_verification.py:78`<br>`test_f05_extended_validator_detects_missing_train_captions` | In `workbench/replay/hint.py:99-106`, `workbench/replay/encoder.py:105-112`, and `workbench/replay/pair.py:99-106`, delete the loop checking `f"captions/cap.{category}.train.json"`. | The test executes its own local helper `validate_dataset_with_train` defined at `line 79`. It never calls `workbench.replay.{hint,encoder,pair}.validate_dataset`. |
| `workbench/tests/test_f05_f08_f09_verification.py:132`<br>`test_f08_preflight_refusal_on_data_dir_mismatch` | In `workbench/replay/dcnet.py:32`, change `validate_dataset_binding` body to `return []` (or remove `validate_dataset_binding(checkpoint_root, dataset_root)` from `workbench/replay/dcnet.py:114`). | The test executes its own local helper `validate_dcnet_data_dir_alignment` defined at `line 133`. It never calls `workbench.replay.dcnet.validate_dataset_binding` or `preflight`. |
| `workbench/tests/test_f05_f08_f09_verification.py:279`<br>`test_f09_dcnet_parser_extracts_combined_metrics_and_multiplies_by_100` | In `workbench/backend/metrics_extraction.py:583`, revert `parse_dcnet_output` to return `unparsed`, omit `* 100.0`, or remove `"dcnet"` dispatch from `extract_aggregate_metrics:800-801`. | The test invokes the local duplicate `parse_dcnet_native_output` defined at `line 174`. It never calls `workbench.backend.metrics_extraction.parse_dcnet_output`. |
| `workbench/tests/test_f05_f08_f09_verification.py:308`<br>`test_f09_dcnet_parser_fallback_for_single_branch` | In `workbench/backend/metrics_extraction.py:640-646`, delete the `elif single_pairs:` block in `parse_dcnet_output`. | The test invokes local duplicate `parse_dcnet_native_output` defined at `line 174`. |
| `workbench/tests/test_f05_f08_f09_verification.py:320`<br>`test_f09_dcnet_parser_rejects_inconsistent_macro_mean` | In `workbench/backend/metrics_extraction.py:657-660`, delete the `comb_avg` cross-check `if abs(decl_mean - macro_mean) > 0.05: return unparsed`. | The test invokes local duplicate `parse_dcnet_native_output` defined at `line 174`. |
| `workbench/tests/test_encoder_pair_clvc_verification.py:24`<br>`test_f10_encoder_checkpoint_schema_and_verification` | In `workbench/replay/encoder.py:192-198`, delete `if missing or unexpected: raise ValueError(...)`. | The test executes its private closure `verify_encoder_checkpoint_keys` defined at `line 28`. It never calls `workbench.replay.encoder`. |
| `workbench/tests/test_verified_declarations.py:185`<br>`test_f10_the_loader_reports_what_strict_false_hid` | In `workbench/replay/encoder.py:192`, change `if missing or unexpected:` to `if False and (missing or unexpected):`. | The test performs only string matching on `encoder.py`. All 4 asserted substrings (`"load_state_dict(ckpt_data, strict=False)"`, `'getattr(report, "missing_keys"'`, `"Refusing to evaluate a partially-loaded model."`, `'!= "loss_T"'`) remain in the file, keeping the test green while the runtime validation is completely bypassed. |
| `workbench/tests/test_dcnet_replay.py:32`<br>`test_dcnet_run_dir_fails_on_missing_files` | In `workbench/replay/dcnet.py:114`, remove `validate_dataset_binding(checkpoint_root, dataset_root)` from `preflight()`. | The test only calls `validate_run_directory`; it does not test dataset binding or call `preflight()`. |
| `workbench/tests/test_dcnet_replay.py:40`<br>`test_dcnet_dataset_fails_on_missing_glove_and_resized_images` | In `workbench/replay/dcnet.py:114`, remove `validate_dataset_binding(checkpoint_root, dataset_root)` from `preflight()`. | The test only calls `validate_dataset`; it does not test dataset binding or call `preflight()`. |
| `workbench/tests/test_handoff_commands.py:160`<br>`test_every_document_in_the_handoff_set_has_run_commands_or_says_why_not` | In `workbench/docs/MODEL_STATUS_SPEC.md`, insert broken bash blocks or invalid CLI commands. | `MODEL_STATUS_SPEC.md` is hardcoded in the `exempt` set at `line 181`, bypassing all command parsing. |
| `workbench/tests/test_handoff_commands.py:474`<br>`test_the_release_documents_quote_the_test_count_they_actually_ran` | In `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md`, change the recorded pass count to any value within 5 of actual (e.g. `1605 passed, 0 failed` when 1610 collect). | The assertion explicitly permits up to 5 tests of drift (`assert collected - recorded <= 5`), allowing inaccurate counts to pass. |
| `workbench/tests/test_handoff_commands.py:513`<br>`test_no_release_document_describes_verify_only_as_verifying_a_run` | In `workbench/docs/PC_PILOT_HANDOFF.md`, state: `Chạy --verify-only để xác minh kết quả benchmark trên GPU mà không cần chạy lại.` | The regex requires words matching `(re-runs?|re-runs verification|rebuild|dựng lại|xác thực)[^.]*(run|lượt chạy|index|chỉ mục)`. Misrepresentations that omit those specific words bypass the regex completely. |
| `workbench/tests/test_handoff_commands.py:535`<br>`test_the_verdict_names_the_same_test_count_the_handoff_does` | In both `FINAL_PC_RELEASE_VERDICT.md` and `PC_PILOT_HANDOFF.md`, set an unverified pass count (e.g. `**2000 passed, 0 failed**`). | The test only checks equality between the two document regex matches (`assert set(counts) == set(handoff_counts)`). |

---

## 3. Deep Dive: `test_f05_f08_f09_verification.py`

`workbench/tests/test_f05_f08_f09_verification.py` is the prime suspect for verifying private test copies instead of production logic.

### 3.1 Production Imports Made
Quoted verbatim from `workbench/tests/test_f05_f08_f09_verification.py`:
- `workbench/tests/test_f05_f08_f09_verification.py:18`:
  ```python
  from workbench.backend.fashioniq_layout import CATEGORIES, ilearn_resized_training_paths
  ```
  *(Note: `ilearn_resized_training_paths` is imported but never called or referenced in the file).*
- `workbench/tests/test_f05_f08_f09_verification.py:19`:
  ```python
  from workbench.backend.metrics_extraction import MetricExtractionResult, _check_parity, _is_valid_recall
  ```
- `workbench/tests/test_f05_f08_f09_verification.py:59`:
  ```python
  from workbench.replay.common import isolated_cache_root
  ```

No other production modules or functions are imported. Crucially, neither `parse_dcnet_output`, nor `workbench.replay.dcnet.validate_dataset_binding`, nor `workbench.replay.{hint,encoder,pair}.validate_dataset` are imported.

### 3.2 Locally-Defined Functions Duplicating Production Logic

#### 1. `validate_dataset_with_train`
- **Location**: `workbench/tests/test_f05_f08_f09_verification.py:79-93` (nested inside `test_f05_extended_validator_detects_missing_train_captions`).
- **Verbatim Code**:
  ```python
      def validate_dataset_with_train(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
          blockers: list[str] = []
          for cat in CATEGORIES:
              for rel in (
                  f"captions/cap.{cat}.train.json",
                  f"captions/cap.{cat}.val.json",
                  f"captions/correction_dict_{cat}.json",
                  f"image_splits/split.{cat}.val.json",
              ):
                  if not (dataset_root / rel).is_file():
                      blockers.append(f"dataset input missing: {dataset_root / rel}")
              cat_resized = dataset_root / "resized_image" / cat
              if not cat_resized.is_dir():
                  blockers.append(f"dataset input missing: {cat_resized}")
          return blockers
  ```
- **Production Function Shadowed**:
  Shadows `validate_dataset` in:
  - `workbench/replay/hint.py:83-110`
  - `workbench/replay/encoder.py:89-115`
  - `workbench/replay/pair.py:83-110`
  as well as `workbench.backend.fashioniq_layout.paths_for_layout`.

#### 2. `validate_dcnet_data_dir_alignment`
- **Location**: `workbench/tests/test_f05_f08_f09_verification.py:133-149` (nested inside `test_f08_preflight_refusal_on_data_dir_mismatch`).
- **Verbatim Code**:
  ```python
      def validate_dcnet_data_dir_alignment(checkpoint_root: Path, source_root: Path, dataset_root: Path) -> list[str]:
          cfg_file = checkpoint_root / "config.json"
          if not cfg_file.is_file():
              return [f"DCNet config missing: {cfg_file}"]
          cfg = json.loads(cfg_file.read_text(encoding="utf-8"))
          data_dir_str = cfg.get("data_loader", {}).get("args", {}).get("data_dir")
          if not data_dir_str:
              return ["DCNet config missing data_loader.args.data_dir"]
          configured_path = (source_root / data_dir_str).resolve() if not Path(data_dir_str).is_absolute() else Path(data_dir_str).resolve()
          if configured_path != dataset_root.resolve():
              return [
                  f"DCNet dataset root mismatch: checkpoint config.json data_loader.args.data_dir '{data_dir_str}' "
                  f"resolves to '{configured_path}', but evaluator received dataset_root '{dataset_root.resolve()}'. "
                  f"To resolve without modifying author config.json, stage a symlink at {source_root / data_dir_str} -> {dataset_root.resolve()}."
              ]
          return []
  ```
- **Production Function Shadowed**:
  Shadows `validate_dataset_binding` in `workbench/replay/dcnet.py:32-68`.
  Production `validate_dataset_binding` refuses relative paths and tells the operator how to bind them. The local test function instead evaluates path resolution against a caller-passed `source_root` and checks symlinks.

#### 3. `parse_dcnet_native_output`
- **Location**: `workbench/tests/test_f05_f08_f09_verification.py:174-272` (module-level function, 99 lines long).
- **Verbatim Signature**:
  ```python
  def parse_dcnet_native_output(stdout_text: str, paper_metrics: dict[str, float | None] | None = None) -> MetricExtractionResult:
      """Proposed parser for DCNet native stdout (test.py --mode val)."""
      parser_id = "dcnet_native_logger"
  ```
- **Production Function Shadowed**:
  Shadows `parse_dcnet_output` in `workbench/backend/metrics_extraction.py:583-662`.
  It duplicates the regex parsing of the `comb:` line, `comp corr r@10, r@50:` component lines, single-branch fallback, and the `[comb_avg]` macro consistency check. Tests at lines 279, 308, and 320 assert exclusively against this local copy.

---

## 4. Tests Asserting on `MODEL_STATUS_SPEC.md` or Handoff Prose

### 4.1 Exemption of `MODEL_STATUS_SPEC.md`
In `workbench/tests/test_handoff_commands.py:160-184`:
```python
    exempt = {
        "AUTOMATED_DATA_PREPARATION.md", "AUTOPILOT_GUIDE.md", "AUXILIARY_ASSET_AUDIT.md",
        "COT_OURS2_AUDIT.md", "ENGINEERING_FINAL_ACCEPTANCE.md", "FASHIONIQ_PROTOCOL_PARITY.md",
        "FINAL_PC_READINESS.md", "FIRST_GPU_EXPERIMENT.md", "LAPTOP_RELEASE_ACCEPTANCE.md",
        "LEGACY_CLEANUP_AUDIT.md", "LEGACY_REPRODUCTION_RUNBOOK.md", "LEGACY_SOURCE_AUDIT.md",
        "PC_DOCKER_QUICKSTART.md", "PC_TRANSFER_CHECKLIST.md", "PROTOCOL_AUDIT.md",
        "READINESS_MATRIX.md", "RESULT_SCHEMA.md", "UPSTREAM_SOURCE_AUDIT.md",
        # MODEL_STATUS_SPEC.md carries the PC continuation commands (clone, verify from a fresh
        # checkout, download assets, preflight, run). They cannot be executed as an audit here for the
        # same reason the model rows are PENDING: staging a checkpoint needs multi-GB downloads and
        # running any model needs a GPU, neither of which this host has. The document is the operator
        # handoff, so it is exempt rather than silently unaudited.
        "MODEL_STATUS_SPEC.md",
    }
```
**What this fails to catch**:
`MODEL_STATUS_SPEC.md` contains documented operator commands for all 16 models. Because it is explicitly exempt from `test_every_documented_command_parses_and_pipeline_has_required_authorization`, this test fails to catch:
- Invalid or hallucinated CLI flags in `MODEL_STATUS_SPEC.md`.
- Malformed command invocations.
- Broken asset download URLs or non-existent script arguments.

### 4.2 Handoff Prose and Text Invariant Checks

#### 1. `--verify-only` Prose Regex
In `workbench/tests/test_handoff_commands.py:513-529`:
```python
            lowered = line.lower()
            # The forbidden claim: the flag verifies, rebuilds, or re-runs something about a RUN.
            if re.search(r"verify-only[^.]*(re-runs?|re-runs verification|rebuild|dựng lại|xác thực)"
                         r"[^.]*(run|lượt chạy|index|chỉ mục)", lowered):
                if "không" not in lowered and "does not" not in lowered and "not " not in lowered:
                    offenders.append(f"{document.relative_to(ROOT)}:{number}")
```
**What this fails to catch**: Fails to catch semantic errors in documentation that don't match the regex pattern. For instance, claiming `--verify-only` runs benchmarks or validates model weights passes without error.

#### 2. Test Pass Count Matching
In `workbench/tests/test_handoff_commands.py:474-510`:
```python
    verdict = (ROOT / "workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md").read_text(encoding="utf-8")
    counts = re.findall(r"\*\*(\d+) passed, 0 failed\*\*", verdict)
    ...
    assert collected - recorded <= 5
```
and `workbench/tests/test_handoff_commands.py:535-541`:
```python
    verdict = (ROOT / "workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md").read_text(encoding="utf-8")
    counts = re.findall(r"\*\*(\d+) passed, 0 failed\*\*", verdict)
    ...
    handoff = (ROOT / "workbench/docs/PC_PILOT_HANDOFF.md").read_text(encoding="utf-8")
    handoff_counts = re.findall(r"\*\*(\d+) passed, 0 failed\*\*", handoff)
    assert counts == handoff_counts
```
**What this fails to catch**:
- Fails to verify that any test actually ran or passed; it only compares regex counts.
- Allows up to 5 tests of silent drift (`collected - recorded <= 5`).
- If both markdown files are edited to claim any identical arbitrary number, the test passes.

#### 3. `pc.sh` Flags Listing
In `workbench/tests/test_pc_launcher.py:1793-1804`:
```python
    doc = (REPO / "workbench/docs/PC_PILOT_HANDOFF.md").read_text(encoding="utf-8")
    marker = "Cờ đã tồn tại"
    assert marker in doc, "the handoff document no longer names its flags"
    listing = doc.split(marker, 1)[1].split("
", 1)[0]
    for flag in ("--auto-pilot", "--execute", "--verify-only", "--plan-only", "--model",
                 "--skip-preflight", "--force-clean", "--help"):
        assert f"`{flag}`" in listing, f"{flag} missing from handoff documentation"
```
**What this fails to catch**: Only checks that the flag names are listed on the specific line in `PC_PILOT_HANDOFF.md`. It does not execute `pc.sh` or verify that the flags operate correctly.

---

## 5. End-to-End Production Tests for the Four Defects

### 5.1 DCNet Binding (F08)
- **Production Implementation**: `workbench/replay/dcnet.py:32-68` (`validate_dataset_binding`) and `workbench/replay/dcnet.py:114` (`preflight`).
- **Tests Exercising Production Code**: **NONE**.
  - `workbench/tests/test_dcnet_replay.py:14` imports `preflight`, but lines 19-48 test only `validate_source`, `validate_run_directory`, and `validate_dataset`. It never calls `validate_dataset_binding` or `preflight`.
  - `workbench/tests/test_f05_f08_f09_verification.py:132` tests its private helper `validate_dcnet_data_dir_alignment`, crossing no production boundary.
- **Boundary Actually Crossed**: None. Production DCNet dataset binding has **0% test coverage**.

### 5.2 DCNet Parser Order (F09)
- **Production Implementation**: `workbench/backend/metrics_extraction.py:583-662` (`parse_dcnet_output`) and `workbench/backend/metrics_extraction.py:800-801` (`extract_aggregate_metrics`).
- **Tests Exercising Production Code**:
  - `workbench/tests/test_verified_declarations.py:119` `test_f09_dcnet_fractions_are_converted_once`: Crosses the public metric extraction entry point `extract_aggregate_metrics("dcnet", ...)`.
  - `workbench/tests/test_verified_declarations.py:129` `test_f09_the_combined_branch_is_the_one_read`: Crosses `parse_dcnet_output`.
  - `workbench/tests/test_verified_declarations.py:135` `test_f09_composition_and_correction_are_not_merged_into_the_combined_result`: Crosses `parse_dcnet_output`.
  - `workbench/tests/test_verified_declarations.py:143` `test_f09_the_two_cell_branch_is_accepted_only_when_complete`: Crosses `parse_dcnet_output`.
  - `workbench/tests/test_verified_declarations.py:155` `test_f09_a_stream_that_contradicts_its_own_average_is_refused`: Crosses `parse_dcnet_output`.
  - `workbench/tests/test_verified_declarations.py:163` `test_f09_ranges_and_completeness_still_hold`: Crosses `parse_dcnet_output` and `extract_aggregate_metrics`.
- **Boundary Actually Crossed**: Metric extraction parser boundary.
- **Vulnerability**: All test fixtures in `test_verified_declarations.py` pass stdout in the standard category order (`dress`, `shirt`, `toptee`). In `parse_dcnet_output` (`workbench/backend/metrics_extraction.py:642-645`), `comb:` values are mapped via `for index, category in enumerate(named)` where `named = [c for c in ("dress", "shirt", "toptee") if c in combined_pairs]`. If the evaluator prints categories in a different order, `comb:` cells are silently transposed because `comb:` has no category names and `named` is sorted by the fixed tuple. **No test tests this non-standard order condition**.

### 5.3 CSMCIR Patch Set (F04)
- **Production Implementation**:
  - `workbench/patches/csmcir_models_init_compat.patch`
  - `workbench/patches/csmcir_validate_visualization_opt_in.patch`
  - `workbench/registry/models.yaml:370-388` (`approved_source_patches`)
  - `workbench/backend/source_provenance.py` (`source_provenance`)
- **Tests Exercising Production Code**:
  - `workbench/tests/test_component_boundaries.py:251` `test_f04_the_visualization_patch_is_declared_and_matches_its_file`: Crosses registry patch declaration and patch file sha256 verification.
  - `workbench/tests/test_component_boundaries.py:259` `test_f04_the_patch_makes_the_diagnostic_dump_opt_in`: Applies patch using real `patch -p1` CLI on a copy of `workbench/third_party/CSMCIR`; verifies AST position of `print(json.dumps(...))` before `if not visualize:` before `vis_fiq(...)`. Crosses real patch application on third-party source.
  - `workbench/tests/test_component_boundaries.py:283` `test_f04_the_patched_checkout_is_approved_and_an_edit_after_it_is_not`: Applies patch and calls production `source_provenance(model, root)`. Crosses runtime source provenance verification boundary.
  - `workbench/tests/test_component_boundaries.py:304` `test_f04_both_csmcir_patches_apply_together`: Sequentially applies both `csmcir_models_init_compat.patch` and `csmcir_validate_visualization_opt_in.patch` with `patch -p1`; verifies `source_provenance` approves both runtime artifacts. Crosses full patch set provenance boundary.
  - `workbench/tests/test_vendored_approved_patch.py:65, 79, 97, 153`: Crosses production runtime provenance verification under patched and unpatched states.

### 5.4 Category Constructor Inputs (F05)
- **Production Implementation**:
  - `workbench/replay/hint.py:83-110` (`validate_dataset`)
  - `workbench/replay/encoder.py:89-115` (`validate_dataset`)
  - `workbench/replay/pair.py:83-110` (`validate_dataset`)
  - `workbench/backend/fashioniq_layout.py:paths_for_layout`
- **Tests Exercising Production Code**:
  - `workbench/tests/test_verified_declarations.py:51` `test_f05_a_val_only_root_is_reported_as_missing_train`: Parametrized for `hint`, `encoder`, `pair`. Calls `module.validate_dataset(root)`. Verifies that a val-only root returns blockers requiring `.train.json`, and adding `.train.json` causes `validate_dataset` to return `[]`. Crosses the production replay preflight dataset validation boundary.
  - `workbench/tests/test_verified_declarations.py:67` `test_f05_the_declared_layout_requires_the_train_paths`: Calls `paths_for_layout(Path("/FashionIQ"), model["fashioniq_layout"])`. Crosses the layout resolution engine boundary.
  - `workbench/tests/test_f05_f08_f09_verification.py:58` `test_f05_isolated_cache_root_starts_fresh_without_cached_tensors`: Calls `isolated_cache_root` from `workbench.replay.common`. Crosses replay temporary cache isolation boundary.

---

## 6. Pytest Collection Count

Execution command:
```bash
timeout 300 .venv/bin/python -m pytest workbench/tests --collect-only -q -p no:cacheprovider 2>/dev/null | tail -2
```

Observed output:
```
-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
1610 tests collected in 1.85s
```

The test suite collects exactly **1610 tests**.

---

## Verified claims

| Claim | Command / Verification Route | Observed Output |
|---|---|---|
| Pytest suite collects 1610 tests | `timeout 300 .venv/bin/python -m pytest workbench/tests --collect-only -q -p no:cacheprovider 2>/dev/null \| tail -2` | `1610 tests collected in 1.85s` |
| Zero test files added in last 3 commits | `git log --name-status -3` | Only `PC_PILOT_HANDOFF.md`, `FINAL_PC_RELEASE_VERDICT.md`, and `MODEL_STATUS_SPEC.md` modified |
| `test_f05_f08_f09_verification.py` defines 3 duplicate functions locally | AST scan and inspection of `workbench/tests/test_f05_f08_f09_verification.py` | `validate_dataset_with_train` (line 79), `validate_dcnet_data_dir_alignment` (line 133), `parse_dcnet_native_output` (line 174) |
| `parse_dcnet_native_output` is 99 lines of shadowed parser logic | Line range check on `workbench/tests/test_f05_f08_f09_verification.py:174-272` | Defines local copy of DCNet stdout parser matching `parse_dcnet_output` |
| `dcnet.validate_dataset_binding` has 0 tests calling it | Repo grep: `grep -rn "validate_dataset_binding" workbench/tests` | `No matches found` |
| `MODEL_STATUS_SPEC.md` is explicitly exempt from command audit | Code inspection `workbench/tests/test_handoff_commands.py:181` | `"MODEL_STATUS_SPEC.md"` in `exempt` set |
| `test_f10_the_loader_reports_what_strict_false_hid` passes when logic is bypassed | String containment test against bypassed guard `if False and (missing or unexpected):` | All 4 substring assertions in `test_verified_declarations.py:185-194` evaluate `True` |
| Handoff `--verify-only` regex fails to match valid misrepresentations | Python `re.search` on `"Chạy --verify-only để xác minh kết quả benchmark trên GPU mà không cần chạy lại."` | Returns `None` (passes `test_handoff_commands.py:513` undetected) |
| Release documents pass count check allows 5 tests of drift | Code inspection `workbench/tests/test_handoff_commands.py:507` | `assert collected - recorded <= 5` |
| Real production F04 CSMCIR patch tests execute in `test_component_boundaries.py` | Inspection of `workbench/tests/test_component_boundaries.py:251-320` | Tests execute `patch -p1`, verify sha256 hashes, and call `source_provenance` |
| Real production F05 constructor input tests execute in `test_verified_declarations.py` | Inspection of `workbench/tests/test_verified_declarations.py:51-78` | Tests call `module.validate_dataset` for HINT, ENCODER, PAIR and `paths_for_layout` |
