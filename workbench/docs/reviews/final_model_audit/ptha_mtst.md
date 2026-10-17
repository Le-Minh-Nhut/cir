# Final Model Audit: ptha_mtst

## A. MODEL IDENTITY

- **Model ID**: `ptha_mtst` (`workbench/registry/models.yaml:217`)
- **Method Name**: `PTHA + MTST` (`workbench/registry/models.yaml:218`)
- **Research Generation**: `recent` (`workbench/registry/models.yaml:219`)
- **Publication Year**: `2025` (`workbench/registry/models.yaml:220`)
- **Paper Title**: `Scale Up Composed Image Retrieval Learning via Modification Text Generation` (`workbench/registry/models.yaml:222`)
- **Paper URL**: `https://arxiv.org/abs/2504.05316` (`workbench/registry/models.yaml:223`)
- **Upstream Repository URL**: `https://github.com/zyinan99/MTST_PTHA` (`workbench/registry/models.yaml:228`)
- **Upstream Commit SHA**: `null` (`workbench/registry/models.yaml:230`) — UNPINNED
- **Source Directory**: `null` (`workbench/registry/models.yaml:232`)
- **Source Available**: `false` (`workbench/registry/models.yaml:231`)
- **Native Protocol**: `null` (`workbench/registry/models.yaml:234`) — UNRESOLVED
- **Supported Protocols**: `[]` (`workbench/registry/models.yaml:235`)
- **Reproduction Status**: `BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` (`workbench/registry/models.yaml:239`)
- **Strict Clean Checkpoint Available**: `false` (`workbench/registry/models.yaml:240`)
- **Checkpoint Variants**: `[]` (`workbench/registry/models.yaml:242`)
- **Reported Metrics**: `reported_r10: 54.99`, `reported_r50: 76.28`, `reported_mean: 65.64` (`workbench/registry/models.yaml:236-238`)

### Registry Claims Audit

| Registry Field | Declared Value | Status | Evidence / Verification |
|---|---|---|---|
| `model_id` | `ptha_mtst` | PRESENT | Unique key in `workbench/registry/models.yaml:217`. |
| `method_name` | `PTHA + MTST` | PRESENT | Matches paper section I & IV terminology ("Prototypical Two-Hop Alignment" + "Modification Text-oriented Synthetic Triplets"). |
| `research_generation` | `recent` | PRESENT | Conforms to `workbench/backend/registry.py:66-67` schema (`publication_year: 2025` >= 2020). |
| `publication_year` | `2025` | PRESENT | arXiv:2504.05316 submission date is 2025-02-21 (`[v1]`). |
| `preparation_contract` | `unavailable` | VERIFIED_ABSENT | Sentinel string accepted by `registry.py:73`, but no contract exists in `workbench/registry/preparation_contracts.yaml`. `preparation_contract_for_model()` returns `None`. |
| `paper_title` | `Scale Up Composed Image Retrieval Learning via Modification Text Generation` | PRESENT | Verified against arXiv:2504.05316 metadata. |
| `paper_url` | `https://arxiv.org/abs/2504.05316` | PRESENT | Resolves to HTTP 200 on arXiv. |
| `upstream_repo_url` | `https://github.com/zyinan99/MTST_PTHA` | PRESENT_BUT_UNPINNED | Remote repository exists on GitHub, but no checkout exists in `workbench/third_party/MTST_PTHA`. |
| `upstream_default_branch` | `main` | PRESENT_BUT_UNPINNED | Branch exists remotely on GitHub; unpinned. |
| `upstream_commit_sha` | `null` | VERIFIED_ABSENT | No git commit SHA is pinned in the repository. |
| `source_available` | `false` | PRESENT | Accurately records that local source is absent. |
| `source_dir` | `null` | VERIFIED_ABSENT | No directory exists in `workbench/third_party/`. |
| `source_blocker` | String | PRESENT | Factual note explaining author repository exists at unpinned HEAD `6a02ee4763fc2260b6abfb18be78853facb91d86` but is not synced, pinned, or verified. |
| `native_protocol` | `null` | VERIFIED_ABSENT | Truthful representation that evaluation protocol remains unresolved. |
| `supported_protocols` | `[]` | VERIFIED_ABSENT | Empty list. |
| `checkpoint_variants` | `[]` | VERIFIED_ABSENT | Zero checkpoints exist in repo or declared in registry. |
| `environment` | (Omitted) | VERIFIED_ABSENT | No environment block declared for `ptha_mtst` in `models.yaml`. |
| `synthetic_source` | (Omitted / False) | VERIFIED_ABSENT | Model is not declared synthetic. |

### Paper Reference Identity

The paper benchmark registry (`workbench/registry/paper_benchmarks/papers.yaml`) contains two entries for `ptha_mtst`:

1. `reference_id: ptha_mtst_fiq_table3_fashiqmtst_blip2` (`papers.yaml:1660-1725`):
   - **Source**: arXiv:2504.05316v1, Table III ("Comparison on FashionIQ validation set"), page 7, line 54 of pdftotext extraction.
   - **Exact Row Quoted**: `PTHA (Ours) | BLIP2 | FashionIQMTST | 50.77 73.78 | 55.91 75.36 | 58.28 79.70 | 54.99 76.28 | 65.64`
   - **Metrics**: Dress R@10=50.77, R@50=73.78; Shirt R@10=55.91, R@50=75.36; Toptee R@10=58.28, R@50=79.70; Average R@10=54.99, Average R@50=76.28, Avg=65.64.
   - **Condition**: PTHA model with BLIP2 backbone pretrained on synthetic corpus `FashionIQ_MTST`. Corresponds to registry values in `models.yaml:236-238`.

2. `reference_id: ptha_mtst_fiq_no_pretrain_table3` (`papers.yaml:1726-1774`):
   - **Source**: arXiv:2504.05316v1, Table III, page 7.
   - **Exact Row Quoted**: `PTHA (Ours) | BLIP2 | - | 49.54 72.81 | 55.50 73.97 | 57.96 78.96 | 54.33 75.24 | 64.79`
   - **Metrics**: Dress R@10=49.54, R@50=72.81; Shirt R@10=55.50, R@50=73.97; Toptee R@10=57.96, R@50=78.96; Average R@10=54.33, Average R@50=75.24, Avg=64.79.
   - **Condition**: Control baseline without `FashionIQ_MTST` pretraining.

---

## B. SOURCE COVERAGE

- **Verdict Axis**: `BLOCKED_SOURCE`
- **Audit Type**: STATIC_AUDIT (source code completely absent from repo).
- **Filesystem Inventory**:
  - `workbench/third_party/` contains 14 directories (`AirKnow`, `CLIP4Cir`, `CLVCNet`, `CSMCIR`, `ConeSep`, `DCNet`, `ENCODER`, `HABIT`, `HINT`, `INTENT`, `LIMN`, `PAIR`, `SPRC`, `TGCIR`).
  - No `MTST_PTHA` directory exists in `workbench/third_party/`.
  - No checkout exists in git submodules or local cache.

### Upstream Repository Status
- Remote URL: `https://github.com/zyinan99/MTST_PTHA`
- Remote HEAD: `6a02ee4763fc2260b6abfb18be78853facb91d86` (commit date 2025-02-09, author Yinan Zhou).
- Repository contains 8 total commits:
  `a6efd88`, `2254834`, `3584a05`, `f8bcca3`, `f41ed83`, `36979a4`, `e064beb`, `6a02ee4`.
- **Fatal Upstream Bug (P0 External)**:
  In `MTST_PTHA/src/lavis/models/__init__.py:54`, the code imports:
  ```python
  from lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn import Blip2Qformer4TGIRCESIMZYN
  ```
  Module `blip2_qformer_tgir_concatembedding_sim_zyn.py` does not exist in commit `6a02ee4`, nor has it existed in any of the 8 git commits in the repository's history. Any import of `lavis.models` immediately fails with `ModuleNotFoundError`.
  Reachable evaluator entrypoint `MTST_PTHA/src/blip_validate.py:16` imports `from lavis.models import load_model_and_preprocess`, so the evaluator cannot even be imported.

---

## C. NATIVE EVALUATION

- **Verdict Axis**: `PROTOCOL_UNVERIFIED`

### Analysis of Protocol Uncertainty
The CIR Failure Analysis Workbench recognizes three distinct evaluation protocols for FashionIQ:
- **PROTOCOL A (`fashioniq_original_split`)**: Full ordered validation split gallery (`split.{cat}.val.json`); reference images REMAIN ELIGIBLE (not excluded from ranking).
- **PROTOCOL B (`fashioniq_full_gallery_ref_excluded`)**: Full ordered validation split gallery (`split.{cat}.val.json`); reference images EXCLUDED per query.
- **PROTOCOL C (`fashioniq_val_split`)**: Unique union of validation references and targets (`cap.{cat}.val.json`); reference images EXCLUDED per query.

### Why Protocol Remains Unresolved
1. **Paper Silence**: arXiv:2504.05316 Section V-A-2 and V-B states only that evaluation is conducted using "Recall@K (K = 10, 50) on Fashion-IQ validation set across three subsets". It does not document gallery construction (full split vs ref+target union) or reference masking policy.
2. **Unpinned, Broken Upstream**: The upstream code cannot run. While unpinned remote file `MTST_PTHA/src/validate_blip.py:38-55` builds similarity matrices against `target_names` without reference exclusion (contrasted with CIRR at line 257 which masks references), this unpinned code has never been validated against checkpoints, cannot produce reproduction numbers, and cannot be established as the authentic experimental configuration without execution evidence.
3. **Registry Inconsistency (CONFIRMED_BUG)**:
   - `workbench/registry/models.yaml:234-235` correctly and honestly sets `native_protocol: null` and `supported_protocols: []`.
   - In contrast, `workbench/registry/paper_benchmarks/papers.yaml:1678` and `:1741` assert `gallery_protocol: fashioniq_original_split` and `reference_excluded: false`.
   - `workbench/registry/paper_benchmarks/extracted/ptha_mtst_scaleup_mtst.json:42` and `:96` also assert `reference_excluded: false`.
   This is a confirmed bug: `papers.yaml` and its extracted benchmark record have assigned Protocol A without source-backed runtime evidence, contradicting the model registry.

---

## D. PREPARATION

- **Preparation Contract**: `unavailable` (`models.yaml:221`).
- **Preparation Registry**: `workbench/registry/preparation_contracts.yaml` contains no contract ID or configuration for `ptha_mtst`.
- **Preflight Paths**: `fashioniq_required_paths(model, ...)` returns empty tuple `()`. Preflight raises no dataset requirements because `fashioniq_layout` and `preparation_id` are absent.
- **Missing Synthetic Corpus (P0 External)**:
  - Table III headline metrics (`54.99 / 76.28 / 65.64`) rely on the `FashionIQ_MTST` corpus: 579,114 synthetic triplets across dress, shirt, and toptee.
  - This corpus is not hosted on Hugging Face (author repository `yinanzhou1/mtst_ptha` contains only CIRR JSON files).
  - Upstream script `blip_fine_tune_2.py:400-401` rejects any dataset argument outside `['fashioniq', 'cirr']`, and `data_utils.py` contains no `FashionIQGen` class.
  - The pretraining data that produced the paper's primary row is neither published nor loadable.

---

## E. CHECKPOINT

- **Verdict Axis**: `CHECKPOINT_BLOCKED`
- **Registry Declaration**: `checkpoint_variants: []`, `strict_clean_checkpoint_available: false`.
- **Public Artifact Investigation**:
  - Author Hugging Face repository `https://huggingface.co/yinanzhou1/mtst_ptha` hosts only 5 files:
    1. `.gitattributes`
    2. `README.md`
    3. `cirr_modifier_generated_5p_completed_rev.json` (298 MB)
    4. `cirr_train_modifier_rev_long.json` (15.6 MB)
    5. `tuned_mtst_cirr.pt` (2.72 GB, SHA256 `1c4dab86a9f66dfe10aff75b39790432eb6cc746cf27b6cfab74b66d93b7588f`)
  - All hosted artifacts are CIRR-specific. No FashionIQ checkpoint exists.
  - Probes for candidate FashionIQ checkpoint filenames on the Hugging Face repo all returned HTTP 404:
    - `.../resolve/main/fashioniq.pt` -> 404
    - `.../resolve/main/fashioniq_mtst.pt` -> 404
    - `.../resolve/main/ptha_fashioniq.pt` -> 404
    - `.../resolve/main/tuned_mtst_fashioniq.pt` -> 404
    - `.../resolve/main/mtst_fashioniq.pt` -> 404
- **Local Artifacts**: None. `workbench/artifacts/checkpoints/` contains zero weights for `ptha_mtst`.

---

## F. WORKBENCH INTEGRATION

- **Verdict Axis**: `COMMAND_BLOCKED`
- **Adapter**: Absent. `workbench/backend/adapters/models.py:146` (`ADAPTERS`) lists 15 model adapters; `ptha_mtst` is not present.
- **Readiness Execution Trace**:
  1. `workbench/scripts/evaluate_models.py`:
     - Function `candidates()` iterates `model["checkpoint_variants"]`, which is empty `[]`.
     - When invoked via `python3 -m workbench.scripts.evaluate_models --model ptha_mtst`, it returns empty `items = []`.
     - The evaluation loop `for model, checkpoint in items:` executes 0 times.
     - **Observed Behavior**: Silently terminates with exit code 0; produces zero output, zero evaluation plans, and zero logs.
  2. `workbench/backend/preflight.py`:
     - `preflight_blockers(model, ...)` returns 3 blockers:
       - `source cannot run: author repository https://github.com/zyinan99/MTST_PTHA exists and is reachable...`
       - `upstream source unavailable`
       - `command not runnable: COMMAND_UNAUDITED`
  3. `workbench/scripts/doctor.py`:
     - Exits with returncode 2 (`BLOCKED`).
     - Reports `[BLOCKED] source:ptha_mtst: source unavailable, dirty, or not pinned`.
     - Reports `[BLOCKED] runtime:ptha_mtst: command unavailable`.
     - Reports `[BLOCKED] command:ptha_mtst: COMMAND_AUDITED` (misleading detail string due to fallback at `doctor.py:169`).
  4. `workbench/scripts/pc_readiness.py`:
     - Exits 0, reporting `models: ptha_mtst BLOCKED_SOURCE`.
  5. `workbench/backend/paper_gate.py`:
     - Fails closed on any non-experiment data kind (`NOT_RUN`).
     - Fails closed if no real execution occurred.
  6. `workbench/backend/benchmark_references.py`:
     - Calling `select_reference_outcome("ptha_mtst", checkpoint_training_noise_pct=None, protocol_id="fashioniq_original_split")` returns `SELECTION_AMBIGUOUS` because both committed rows (`ptha_mtst_fiq_table3_fashiqmtst_blip2` and `ptha_mtst_fiq_no_pretrain_table3`) match identical conditions (`noise=None`, `protocol=fashioniq_original_split`).
  7. `workbench/scripts/verify_run.py`:
     - Fails closed: refuses any run of `ptha_mtst` with `verified: False` due to reference selection ambiguity.

---

## G. OBSERVATION CAPABILITY & SCIENTIFIC RESTRICTIONS

- **Query Exporter Verdict**: `QUERY_EXPORT_BLOCKED`
  - No native evaluator can be executed.
  - No adapter or runner exists to attach per-query hooks.
  - `ResultRun` per-query records cannot be produced.

### Scientific Comparison Restrictions
- **Mandatory Refusals**:
  1. The workbench must refuse cross-model comparison between `ptha_mtst` and models operating under Protocol B (`fashioniq_full_gallery_ref_excluded`) or Protocol C (`fashioniq_val_split`).
  2. The workbench must refuse cross-model comparison against Protocol A until `ptha_mtst` has a verified, source-backed native protocol.
  3. The workbench must refuse failure overlap (Jaccard similarity via `/api/analysis/failure-jaccard`) and consensus failure analysis involving `ptha_mtst`.
- **Enforcement Mechanism in Code**:
  - `workbench/backend/storage.py:150-152` (`validate_analysis_selection`) validates `len(protocols) == 1`. Runs under differing protocols are rejected with `WorkbenchError("cross_protocol", ...)`.
  - `workbench/backend/storage.py:153-155` enforces `len(data_kinds) == 1`.
  - **Enforcement Gap**: There is NO explicit filter by `model_id == "ptha_mtst"` in `validate_analysis_selection`. Protection currently relies on the fact that `ptha_mtst` has no runnable adapter, no checkpoints, and cannot produce a `ResultRun`. If a fabricated run tagged `fashioniq_original_split` were inserted into DuckDB, `validate_analysis_selection` would not reject it by model ID.

---

## H. DEFECTS

### C1 (P1, Confirmed Bug) — Reference Selection Ambiguity Refuses All Runs
- **File:Line**: `workbench/registry/paper_benchmarks/papers.yaml:1660-1774`, `workbench/backend/benchmark_references.py:277-305`
- **Precondition**: `select_reference_outcome()` queried for `ptha_mtst`.
- **Actual**: Both reference rows (`ptha_mtst_fiq_table3_fashiqmtst_blip2` and `ptha_mtst_fiq_no_pretrain_table3`) declare `checkpoint_id: null`, `checkpoint_training_noise_pct: null`, and `gallery_protocol: fashioniq_original_split`. Selector matches both and returns `SELECTION_AMBIGUOUS`.
- **Expected**: Disambiguated reference criteria (e.g. pretraining condition) allowing selection of the intended benchmark row.

### C2 (P0 External, Confirmed Bug) — Upstream Author Repository Cannot Be Imported
- **File:Line**: `zyinan99/MTST_PTHA@main` `MTST_PTHA/src/lavis/models/__init__.py:54`
- **Precondition**: Attempting to import `lavis.models` in the upstream codebase.
- **Actual**: Unguarded import of nonexistent module `lavis.models.blip2_models.blip2_qformer_tgir_concatembedding_sim_zyn` raises `ModuleNotFoundError`. Absent in all 8 git commits.
- **Expected**: Clean import of official model definitions.

### C3 (P0 External, Confirmed Bug) — Unpublished Pretraining Corpus and Checkpoints
- **File:Line**: `zyinan99/MTST_PTHA` `README.md:82-95`, `blip_fine_tune_2.py:400-401`
- **Precondition**: Operator attempts to reproduce Table III headline metrics.
- **Actual**: `FashionIQ_MTST` pretraining corpus (579,114 triplets) is unpublished; code rejects `--dataset FashionIQGen`; zero FashionIQ checkpoints exist on Hugging Face.
- **Expected**: Public checkpoint and pretraining dataset availability.

### C4 (P1, Confirmed Bug) — Premature Protocol Assignment in `papers.yaml`
- **File:Line**: `workbench/registry/paper_benchmarks/papers.yaml:1678, 1741`, `workbench/registry/paper_benchmarks/extracted/ptha_mtst_scaleup_mtst.json:42, 96`
- **Precondition**: Inspecting protocol assignment for `ptha_mtst`.
- **Actual**: Records assign `gallery_protocol: fashioniq_original_split` and `reference_excluded: false` without source-backed runtime evidence, contradicting `workbench/registry/models.yaml:234-235` where protocol is `null`.
- **Expected**: Benchmark registry should record protocol as `null` / `UNRESOLVED` until source verification is established.

### C5 (P2, Confirmed Bug) — `evaluate_models.py` Exits 0 Silently Without Action
- **File:Line**: `workbench/scripts/evaluate_models.py:212-261`
- **Precondition**: Running `python3 -m workbench.scripts.evaluate_models --model ptha_mtst`.
- **Actual**: Empty `checkpoint_variants` causes `candidates()` to return `[]`. The script loops 0 times and exits 0 with no stdout/stderr or indication of failure.
- **Expected**: Fail closed with explicit message indicating no runnable checkpoint variants exist.

### C6 (P2, Confirmed Bug) — Misleading `COMMAND_AUDITED` Detail String in `doctor.py`
- **File:Line**: `workbench/scripts/doctor.py:168-169`
- **Precondition**: Running `doctor.py --model ptha_mtst`.
- **Actual**: Fallback logic `command_status or "COMMAND_AUDITED"` outputs `[BLOCKED] command:ptha_mtst: COMMAND_AUDITED` despite `ptha_mtst` having no adapter or script.
- **Expected**: Detail string should report `COMMAND_UNAUDITED` or `COMMAND_ABSENT`.

### C7 (P2, Inconsistency) — Stale "No Source Exists" Notes in Documentation Assets
- **File:Line**: `workbench/registry/assets/cluster_b5_ptha_mtst.yaml:3, 32-33`, `workbench/docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:19, 38`
- **Precondition**: Comparing documentation notes against model registry.
- **Actual**: Files assert that no author repository exists ("the workbench does not invent one"), contradicting `models.yaml:228` which now records `https://github.com/zyinan99/MTST_PTHA`.
- **Expected**: Documentation aligned with current registry evidence.

---

## I. EVIDENCE

1. **Silent evaluate_models Execution**:
   ```bash
   $ python3 -m workbench.scripts.evaluate_models --model ptha_mtst
   $ echo $?
   0
   ```
   *Observed*: Zero output emitted, clean exit 0.

2. **Preflight Blockers**:
   ```python
   from workbench.backend.operator_config import resolve_config
   from workbench.backend.preflight import preflight_blockers
   from workbench.backend.registry import load_registry

   model = [
       m for m in load_registry()["models"] if m["model_id"] == "ptha_mtst"
   ][0]
   print(preflight_blockers(model, resolve_config()))
```
   *Observed Output*:
   `["source cannot run: author repository https://github.com/zyinan99/MTST_PTHA exists and is reachable (README title matches the paper, commits authored by the paper's first author Yinan Zhou, HEAD 6a02ee4763fc2260b6abfb18be78853facb91d86 at 2025-02-09) but is not synced, pinned or audited here, and no FashionIQ fine-tuned checkpoint is verified; the earlier 'no source exists' note was incomplete", 'upstream source unavailable', 'command not runnable: COMMAND_UNAUDITED']`

3. **PC Readiness Output**:
   ```bash
   $ python3 workbench/scripts/pc_readiness.py --model ptha_mtst
   ```
   *Observed Output*:
   `ptha_mtst BLOCKED_SOURCE`
   `summary: {"BLOCKED_SOURCE": 1}`

4. **Doctor Command Detail Fallback Bug**:
   ```bash
   $ python3 workbench/scripts/doctor.py --model ptha_mtst
   ```
   *Observed Output*:
   `[BLOCKED] command:ptha_mtst: COMMAND_AUDITED`

5. **Reference Selection Ambiguity Reproduction**:
   ```python
   from workbench.backend import benchmark_references as br

   ref = br.select_reference_outcome(
       "ptha_mtst",
       checkpoint_training_noise_pct=None,
       protocol_id="fashioniq_original_split",
   )
   print("Outcome:", ref.outcome)
   print("Reason:", ref.reason)
```
   *Observed Output*:
   `Outcome: SELECTION_AMBIGUOUS`
   `Reason: ptha_mtst: the registry documents this exact experiment more than once -- 2 verified rows match the run's condition (protocol fashioniq_original_split) and disagree on which one this run reproduces: ['ptha_mtst_fiq_no_pretrain_table3', 'ptha_mtst_fiq_table3_fashiqmtst_blip2']...`

---

## J. VERDICT

- **SOURCE_COVERAGE**: `BLOCKED_SOURCE`
- **PROTOCOL**: `PROTOCOL_UNVERIFIED`
- **COMMAND**: `COMMAND_BLOCKED`
- **CHECKPOINT**: `CHECKPOINT_BLOCKED`
- **QUERY_EXPORT**: `QUERY_EXPORT_BLOCKED`
- **RELEASE_READINESS**: `BLOCKED_EXTERNAL_DEPENDENCY`
