# AUDIT_D — Round 02 Verification Audit (Slice C + D)

Audited slice: `workbench/scripts/verify_run.py`, `workbench/backend/paper_gate.py`, `workbench/backend/benchmark_references.py`, `workbench/backend/checkpoint_selection.py`, `workbench/backend/schemas/results.py`, `workbench/registry/paper_benchmarks/papers.yaml`, `workbench/replay/` (`limn.py`, `tgcir.py`, `common.py`), `workbench/scripts/evaluate_models.py`.
Cross-slice references audited: `pc.sh`, `workbench/docker/release_verify.sh`, `workbench/backend/main.py`, `workbench/backend/runtime.py`.
Branch: `research/cir-failure-workbench` at `92ecbd9` + 3 uncommitted fix waves.
Method: Direct static source analysis, AST/logic path tracing, 10 executed bidirectional attack probes on verifier, ledger/proof verification harness.

---

## 1. Resolution of Round-01 Open Leads

### D-F1: Unregistered Model with `observed_metrics=None` Certifies
- **Verdict**: `CONFIRMED` | **Severity**: `P1`
- **Location**: `workbench/scripts/verify_run.py:427-445`.
- **Mechanism**:
  For `outcome == benchmarks.SELECTION_UNREGISTERED`, `honest = list(_extraction_reasons(report))`. The metric validation loop runs:
  `honest.extend(... for key, value in observed.items() if key in paper_gate.METRIC_KEYS and not _is_recall(value))`
  When `observed_metrics` is `None`, `{}`, or contains only non-`METRIC_KEYS` (e.g. `{"other": 1.0}`), `observed.items()` yields nothing. `honest` remains `[]`.
  `verify()` receives `metric_reasons = []`. `corroborated` and `is_research_evidence` evaluate to `True`.
  A report containing ZERO extracted metric numbers certifies as authentic scientific evidence.
- **Observed Execution**:
  Command:
  ```bash
  .venv/bin/python -c "
  import workbench.scripts.verify_run as vr
  rep = {'metric_extraction_available': True, 'extraction_status': 'EXTRACTED_VERIFIED', 'observed_metrics': None}
  reasons, outcome = vr.metric_evidence_reasons('unregistered_model', 'ckpt1', 'fashioniq_original_split', rep)
  print('outcome:', outcome, 'reasons:', reasons)
  "
  ```
  Output:
  `outcome: UNREGISTERED reasons: []`
- **Correct Contract**:
  Every report claiming `extraction_status in (EXTRACTED_VERIFIED, EXTRACTED_SYNTHETIC)` MUST contain at least one finite Recall in `paper_gate.METRIC_KEYS` (`r10`, `r50`, `mean`) or complete category metrics. An empty dictionary or `None` must be refused.

### A9: `OPERATIONS` PENDING on Empty Queue Yields `ENGINEERING_INCOMPLETE`
- **Verdict**: `CONFIRMED` (EXPECTED_BEHAVIOR for J8 / P2 false release blocker)
- **Location**: `workbench/docker/release_verify.sh:370-377, 531, 811-820`; `workbench/backend/jobs.py:505`.
- **Mechanism**:
  Check J8 queries `/api/jobs`. On clean deployment or after backend restart, job queue is empty (`jobs == []`).
  J8 exits with code 3 (`NO_JOBS_OBSERVED`), setting J8 to `PENDING`.
  Dimension `OPERATIONS` requires J8 (`mandatory=True`) and becomes `PENDING`.
  `release_verify.sh:817-818` checks if mandatory pending dimensions are in `("GPU_RUNTIME", "MODEL_INFERENCE", "MODEL_EVALUATION", "SCIENTIFIC_OUTPUT")`. `OPERATIONS` is NOT in this list.
  Final verdict is forced to `"ENGINEERING_INCOMPLETE"`.
- **Target PC Impact**: Clean host with passing GPU and existing verified runs reports `ENGINEERING_INCOMPLETE` until operator manually enqueues and runs a pilot job.

### B-1 / A5: Divergent `_gpu_capability_key` Causes Cross-Process False FAIL
- **Verdict**: `CONFIRMED` | **Severity**: `P1`
- **Location**: `workbench/backend/runtime.py:1030-1047`; `workbench/docker/release_verify.sh:581-583, 637-648`.
- **Mechanism**:
  `_gpu_capability_key` derives key solely from `os.environ.get(env.get("python_env_var") or "", "")`. It never consults `environment_assignment.json`.
  `verify_environment` adopts interpreter path from `environment_assignment.json` when env var is unset.
  Writer (with env var) hashes interpreter path; Reader (in container without env var) hashes `interpreter=""`.
- **Observed Execution**:
  `Key with env var: 6e5b4be7... | Key without env var: 28c012ba... | Keys match: False`
  `release_verify.sh:637` rejects genuine capability file with `"the record's capability key is not this environment's key"`.
- **Target PC Impact**: False FAIL. RTX 5070 Ti capability record rejected by release verifier; `GPU_RUNTIME` pinned to `PENDING`.

### B-2: Advisory Read Omits Interpreter Fingerprint Comparison
- **Verdict**: `CONFIRMED` | **Severity**: `P2` (KNOWN_LIMITATION / EXPECTED_BEHAVIOR)
- **Location**: `workbench/backend/runtime.py:1192-1220`.
- **Mechanism**: Advisory pass (`authoritative=False`) skips byte-level fingerprint hashing for latency. Drifted packages return cached `GPU_RUNTIME_READY`. Authoritative path retains full verification.

### F4: `_proof_stamp == "absent"` with Development Opt-In Certifies Report
- **Verdict**: `EXPECTED_BEHAVIOR` | **Severity**: `P3`
- **Location**: `workbench/backend/main.py:208-245, 332-359`.
- **Analysis**:
  `_certification_mode` requires BOTH `_development_process()` (`PYTEST_CURRENT_TEST` or `WORKBENCH_DEVELOPMENT_MODE`) AND `_development_opt_in()` (`WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`).
  Production process sets neither flag, returning `"proof_unavailable"` and `certified=False`. Sound fail-closed behavior in production.

### F8-release: Substring Proxy in `release_verify.sh` Passes `MODEL_INFERENCE` on Refuted Runs
- **Verdict**: `CONFIRMED` | **Severity**: `P1`
- **Location**: `workbench/docker/release_verify.sh:752, 764-765, 782-785`.
- **Mechanism**:
  `_has_proof = not any("completion proof" in r for r in _reasons)`.
  When a run is refuted for invalid macro arithmetic, unparsed metrics, out-of-range Recall, or missing ledger entry, none of the refusal strings contain the literal substring `"completion proof"`.
  `_has_proof` evaluates to `True`. Line 764 appends `_run_id` to `_inference_evidence`, and lines 782-785 promote `MODEL_INFERENCE` to `PASS`.
- **Observed Execution**:
  Simulating report with macro discrepancy `r10: 99.0` (refused by `_macro_reasons`):
  `has_proof: True | inference_evidence: ['run_tampered_macro'] | MODEL_INFERENCE status: PASS`

---

## 2. Regressions Introduced in Recent Fix Waves

### D-R2-01: `_category_reasons` Rejects Partial Category Tables When `required=False`
- **Verdict**: `CONFIRMED` | **Severity**: `P2`
- **Location**: `workbench/scripts/verify_run.py:205-222, 380-385, 441-444`.
- **Mechanism**:
  `_category_reasons(categories, required=False)` uses `required` only to allow empty/None `categories`.
  Once `categories` is a non-empty dictionary, the loop `for category in LIMN_CATEGORIES:` unconditionally requires all 3 categories:
  `if not isinstance(metrics, dict): reasons.append(f"category {category} is missing from the report")`.
  `_non_limn_reasons:384` contains guard: `if not category_reasons and all(category in categories for category in LIMN_CATEGORIES):`.
  Because `category_reasons` always fails on missing categories, partial category tables for non-LIMN or unregistered models can never pass, and macro re-derivation on partial tables is dead code.
- **Observed Execution**:
  Command:
  ```bash
  .venv/bin/python -c "
  import workbench.scripts.verify_run as vr
  print(vr._category_reasons({'dress': {'r10': 10.0, 'r50': 20.0}}, required=False))
  "
  ```
  Output:
  `['category shirt is missing from the report', 'category toptee is missing from the report']`

### D-R2-02: Duplicate `LedgerKeyError` Appended to `reasons`
- **Verdict**: `CONFIRMED` | **Severity**: `P3` (Message hygiene regression)
- **Location**: `workbench/scripts/verify_run.py:111-120`; `workbench/scripts/evaluate_models.py:1374-1375`.
- **Mechanism**:
  Wave 3 added `em.ledger_hmac_key(config)` check in `verify_run.py:111-114`.
  Wave 3 also made `read_evidence_ledger` re-raise `LedgerKeyError` (`evaluate_models.py:1374`).
  `verify_run.py:117-119` catches `LedgerKeyError` from `read_evidence_ledger` and appends identical text again.
  When two conflicting keys exist and ledger file exists, `reasons` contains the identical 240-character error message twice.
- **Observed Output**:
  `the evidence ledger key is unusable: two different ledger keys exist (...); keep one file and remove the other...` (printed twice back-to-back).

---

## 3. Bidirectional Execution Attack Probes on `verify_run.py`

Harness: `.venv/bin/python` under temporary operator sandbox with live config resolution.

| Probe | Target Condition | Exact Command / Input | Observed Result | Classification |
|---|---|---|---|---|
| **P1** | Tampered report post-ledger | Mutated `observed_metrics.r10` from `27.93` to `99.9` in aggregate JSON | `verified: False`. Reasons: ledger digest mismatch, proof report digest mismatch, macro r10 inconsistent | `CONFIRMED_PROTECTED` |
| **P2** | Replayed proof from another run | Renamed valid proof from `r_orig` to `r_replay_completion.json` | `verified: False`. Reason: `declared output belongs to run 'r_orig', not the current run 'r_replay'` | `CONFIRMED_PROTECTED` |
| **P3** | Inconsistent macro arithmetic | Report with category recalls ~28.0 and macro `r10: 99.0, mean: 78.45` | `verified: False`. Reasons: macro r10 and mean inconsistent with categories (tolerance 0.05 pp) | `CONFIRMED_PROTECTED` |
| **P4a** | Extraction available = False | `metric_extraction_available: False` | `verified: False`. Reason: `metric_extraction_available=False: no metric extraction was performed` | `CONFIRMED_PROTECTED` |
| **P4b** | Extraction status unparsed | `extraction_status: "EVALUATION_COMPLETED_METRICS_UNPARSED"` | `verified: False`. Reason: `extraction_status is 'EVALUATION_COMPLETED_METRICS_UNPARSED', not 'EXTRACTED_VERIFIED'` | `CONFIRMED_PROTECTED` |
| **P5a** | Declared synthetic marker | `is_synthetic: True` | `evidence_reasons`: `the report declares itself synthetic (is_synthetic), so it is a pipeline self-test` | `CONFIRMED_PROTECTED` |
| **P5b** | Data kind synthetic | `data_kind: "synthetic"` | `evidence_reasons`: `the report's data_kind is 'synthetic', which is not a real experiment` | `CONFIRMED_PROTECTED` |
| **P6a** | Metric NaN | `category_metrics.dress.r10: float("nan")` | `verified: False`. Reason: `category dress r10 is not a finite Recall in [0, 100]: nan` | `CONFIRMED_PROTECTED` |
| **P6b** | Metric infinite | `category_metrics.dress.r10: float("inf")` | `verified: False`. Reason: `category dress r10 is not a finite Recall in [0, 100]: inf` | `CONFIRMED_PROTECTED` |
| **P6c** | Metric out-of-range (>100) | `category_metrics.dress.r10: 150.0` | `verified: False`. Reason: `category dress r10 is not a finite Recall in [0, 100]: 150.0` | `CONFIRMED_PROTECTED` |
| **P6d** | Metric out-of-range (<0) | `category_metrics.dress.r10: -5.0` | `verified: False`. Reason: `category dress r10 is not a finite Recall in [0, 100]: -5.0` | `CONFIRMED_PROTECTED` |
| **P6e** | Metric boolean | `category_metrics.dress.r10: True` | `verified: False`. Reason: `category dress r10 is not a finite Recall in [0, 100]: True` | `CONFIRMED_PROTECTED` |
| **P7** | Exact boundary values | Category and macro recalls at exactly `0.0` and `100.0` | `verified: True, reasons: []`. Boundary values accepted cleanly | `CONFIRMED_PROTECTED` |
| **P8a** | Non-LIMN valid report (CSMCIR) | CSMCIR with valid extracted recalls matching published Table 1 | `outcome: MATCHED, reasons: []`. Verified cleanly | `CONFIRMED_PROTECTED` |
| **P8b** | Non-LIMN bad report (CSMCIR) | CSMCIR with `r10: nan, r50: 150.0` | `outcome: MATCHED`. Reason: `reference publishes r10, r50, mean but report carries none as finite Recall` | `CONFIRMED_PROTECTED` |
| **P9** | `NO_PUBLISHED_COMPARISON` model | `combiner_rn50x4_noft` with extracted valid metrics | `outcome: NO_PUBLISHED_COMPARISON`. Reason: `permanent property of what author published` | `CONFIRMED_PROTECTED` |
| **P10** | Unregistered model, no metrics (D-F1) | `model_id: "unregistered_model"`, `observed_metrics: None` | `outcome: UNREGISTERED, reasons: []`. Certified with no numbers | **`CONFIRMED_DEFECT` (P1)** |

---

## 4. Subsystem Completeness & Contract Verification

### 4.1 LIMN Verifier Contract (`verify_run.py:291-305`)
- Requires `metric_extraction_available is True` and `extraction_status in (EXTRACTED_VERIFIED, EXTRACTED_SYNTHETIC)`.
- Enforces presence of all three FashionIQ categories (`dress`, `shirt`, `toptee`) with finite Recalls in [0, 100].
- Computes macro averages: `abs(macro_r10 - mean(cat_r10)) <= 0.05`, `abs(macro_r50 - mean(cat_r50)) <= 0.05`, and `abs(macro_mean - (macro_r10 + macro_r50)/2) <= 0.05`.
- Sound fail-closed behavior verified.

### 4.2 Non-LIMN Reference Selection & Verification (`benchmark_references.py:236-313`)
- `MATCHED`: Exactly 1 verified row matches run condition (`noise`, `protocol_id`). Compares against published metrics.
- `AMBIGUOUS`: Multiple verified rows match condition (`ptha_mtst` Table III pretrain vs no-pretrain). Fails closed naming competing rows.
- `NO_PUBLISHED_COMPARISON`: Authors never published numbers (`combiner_rn50x4_noft`). Distinct permanent reason returned.
- `NO_MATCHING_CONDITION`: Condition unrepresented in registry. Fails closed.
- `UNREGISTERED`: Model not in `papers.yaml`. Fails to enforce non-empty metrics (Defect D-F1).

### 4.3 Paper Gate (`paper_gate.py:26-38, 170-220, 294-475`)
- Enforces 5.0% relative error tolerance: `abs(reproduced - paper) / abs(paper) * 100 <= 5.0 * (1 + 1e-12)`.
- Zero-baseline policy: `if paper == 0.0: ok = (reproduced == 0.0)`.
- `_comparability_blockers`: Strictly matches `protocol_id`, `checkpoint_id`, `noise`, `split`, `dataset`.
- Decoupled from verifier: Metric deviation produces `ALERT_METRIC_DEVIATION` (`passed=False`), but verifier outputs `is_research_evidence: True` (authentic empirical evidence preserved).

### 4.4 Checkpoint Selection Policies (`checkpoint_selection.py:32-71, 109-254`)
- `strict_clean`: Selects only variants with `noise == 0`, `checkpoint_mapping_status == VERIFIED_METADATA`, and obtainable source.
- `include_flagged_conditions`: Allows variants with unverified or non-zero noise; marks selection as flagged secondary cohort.
- Bundle resolution: Bundle selected only when all required members satisfy policy.

### 4.5 Schema V2 Contracts (`schemas/results.py:8-123`)
- Validates protocols: `fashioniq_original_split`, `fashioniq_full_gallery_ref_excluded`, `fashioniq_val_split`.
- `data_kind`: `Literal["experiment", "mock", "synthetic"]` representable without Pydantic validation failure.
- Strict query checks: Contiguous ranks from 1, unique image IDs, canonical query ID formatting (`category:index:ref:target`).

### 4.6 Replay Modules (`workbench/replay/`)
- `limn.py`: Validates 3-category bundle checksums, adapted runtime pins (`sm_120`), dataset JSON paths, and whole-model weights hash before unpickling.
- `tgcir.py`: Fails closed; preflight detects unverified checkpoint mapping and forbids whole-model unpickling.
- `common.py:macro_aggregate`: Computes true unweighted category arithmetic mean, never query-weighted average.

---

## 5. Scope & Inspection Exclusions

### Inspected First-Party Files (Slice C+D)
1. `workbench/scripts/verify_run.py` (471 lines)
2. `workbench/backend/paper_gate.py` (475 lines)
3. `workbench/backend/benchmark_references.py` (329 lines)
4. `workbench/backend/checkpoint_selection.py` (263 lines)
5. `workbench/backend/schemas/results.py` (141 lines)
6. `workbench/registry/paper_benchmarks/papers.yaml` (1837 lines)
7. `workbench/replay/common.py` (108 lines)
8. `workbench/replay/limn.py` (257 lines)
9. `workbench/replay/tgcir.py` (95 lines)
10. `workbench/scripts/evaluate_models.py` (2061 lines)

### NOT Inspected in this Slice
1. Frontend React components (`workbench/frontend/App.tsx`, `workbench/frontend/types.ts`).
2. Docker build stages and multi-arch Dockerfiles (`workbench/docker/Dockerfile`).
3. DuckDB indexing and SQL query routines (`workbench/backend/index.py`, `workbench/backend/storage.py`).
4. Autopilot state machine internals (`workbench/backend/autopilot.py`).
5. Real CUDA Blackwell hardware execution (host lacks RTX 5070 Ti and Nvidia Container Toolkit).

---

## 6. Target PC Hardware Feasibility Verdict

**Can this fail on the real first run on an RTX 5070 Ti PC?**

**YES.**

Three concrete execution failure modes exist on the target hardware:
1. **GPU Capability Key Divergence (B-1 / A5)**: `_gpu_capability_key` in `runtime.py:1031` computes the environment hash without reading `environment_assignment.json`. When `release_verify.sh` executes inside the container, `WORKBENCH_LIMN_PYTHON` is not present in container environment. The verifier recomputes an empty-interpreter key, rejecting the RTX 5070 Ti capability file with `"the record's capability key is not this environment's key"`, pinning `GPU_RUNTIME` to `PENDING`.
2. **Release Blocker on Clean Deployment (A9)**: On a freshly initialized system, `/api/jobs` returns `[]`. Check J8 exits with code 3, setting `OPERATIONS` to `PENDING`. Because `OPERATIONS` is mandatory and not in the GPU/research exemption list, `release_verify.sh` marks the overall verdict as `ENGINEERING_INCOMPLETE`, blocking acceptance despite healthy services.
3. **Refuted Inference Masking (F8-release)**: In `release_verify.sh:752`, `_has_proof = not any("completion proof" in r for r in _reasons)`. If a real execution fails extraction or macro arithmetic, `_has_proof` evaluates to `True`, improperly certifying `MODEL_INFERENCE` as `PASS`.
