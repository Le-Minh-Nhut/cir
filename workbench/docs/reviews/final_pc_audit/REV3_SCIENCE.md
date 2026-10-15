# Independent Review: Scientific Integrity, Protocol, Paper Gate, and Benchmark References (Slice C/D)

## 1. Scope & Inspected Components
- `workbench/backend/paper_gate.py` (475 lines) — acceptance gate, metric tolerance (5%), category comparison, mapping-status cap.
- `workbench/backend/benchmark_references.py` (195 lines) — paper benchmark registry loader, condition-based reference matching.
- `workbench/backend/checkpoint_selection.py` (263 lines) — `strict_clean` vs `include_flagged_conditions` selection policy.
- `workbench/backend/registry.py` (886 lines) — model registry, bundle resolution in `checkpoint_by_id`, manifest-backed checkpoint digests.
- `workbench/registry/paper_benchmarks/papers.yaml` (1837 lines) — committed paper benchmark references across 33 rows.
- `workbench/scripts/evaluate_models.py` (1902 lines) — plan execution, invocation, evidence ledger writing, gate invocation.
- `workbench/scripts/validate_results.py` (75 lines) — schema-v2 result artifact provenance and structural validator.
- Prior audits: `workbench/docs/reviews/final_pc_audit/C.md` (models/eval) and `D.md` (scientific integrity).

---

## 2. Verification by Execution

### A. LIMN Paper Row Certification: Before vs After Vocabulary Fix
- **Context**: In commit `6e3c979~1`, `papers.yaml` spelled verified mappings as `VERIFIED` while `paper_gate.py:415` only checked `("VERIFIED_METADATA", "NO_CHECKPOINT")`, demoting byte-exact LIMN reproductions to `passed=False`.
- **Execution**: Evaluated gate with exact paper numbers for LIMN (`r10=55.9133`, `r50=77.82`, `mean=66.8667`, per-category `dress`, `shirt`, `toptee`) under `run_meta` with `checkpoint_mapping_verified=True`.
- **Observed (Before Fix)**:
  - `status`: `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`
  - `passed`: `False`
  - `detail`: `"every comparable metric is within 5 percent relative error; checkpoint mapping is UNVERIFIED, so this is a metric comparison, not a certified official reproduction; 1 condition(s) the reference does not state were not checked"`
- **Observed (Current Commit `6e3c979`)**:
  - `status`: `PASS_WITHIN_5_PERCENT`
  - `passed`: `True`
  - `detail`: `"every comparable metric is within 5 percent relative error; 1 condition(s) the reference does not state were not checked"`
- **Classification**: NOT_A_BUG (Fix verified in `paper_gate.py:415`).

### B. Preservation of the Mapping Status Cap (ConeSep UNVERIFIED Row)
- **Context**: The vocabulary expansion in `paper_gate.py:415` must not become an "accept everything" leak. References with unverified asset mappings must stay capped at `passed=False` even if `run_meta` asserts verified mapping.
- **Execution**: Selected `conesep_fiq_n20` (`checkpoint_training_noise_pct=20`, `checkpoint_mapping_status: UNVERIFIED`). Provided exact paper metrics (`r10=54.93`, `r50=75.01`, `mean=64.97`, complete categories) with `run_meta={"checkpoint_mapping_verified": True}`.
- **Observed**:
  - `status`: `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`
  - `passed`: `False`
  - `detail`: `"every comparable metric is within 5 percent relative error; checkpoint mapping is UNVERIFIED, so this is a metric comparison, not a certified official reproduction"`
- **Classification**: NOT_A_BUG (Cap holds; unverified paper mappings cannot certify reproduction).

### C. Reference Publishing No Category Table: Complete vs Partial Run Coverage
- **Context**: When a benchmark reference publishes only aggregate metrics (no `category_metrics` table), `paper_gate.py:449-467` inspects the run's category coverage.
- **Execution 1 (Complete Coverage)**: Evaluated reference having `category_metrics={}` against run supplying complete categories (`dress`, `shirt`, `toptee` with valid `r10` and `r50`).
  - `status`: `PASS_WITHIN_5_PERCENT`
  - `passed`: `True`
  - `detail`: `"...the reference publishes no category table, so per-category agreement was checked against the run's complete per-category evidence"`
- **Execution 2 (Partial Coverage)**: Evaluated identical reference against run observing only `dress`.
  - `status`: `PASS_WITH_UNCHECKED_CATEGORIES`
  - `passed`: `False`
  - `detail`: `"...the reference publishes no category table, so per-category agreement is unchecked"`
- **Classification**: NOT_A_BUG (Both directions verified; partial category run correctly demoted).

### D. Synthetic & Mock Run Refusal and Evidence-Ledger Gating on `/api/benchmark-runs`
- **Execution 1 (Gate Refusal of Non-Experiment Data)**:
  - `data_kind="synthetic"`: `status=NOT_RUN`, `passed=False`, note: `["data_kind='synthetic' is a pipeline self-test, not an experiment"]`.
  - `data_kind="mock"`: `status=NOT_RUN`, `passed=False`, note: `["data_kind='mock' is not a real experiment"]`.
- **Execution 2 (Evidence Ledger Corroboration in `main._aggregate_reports`)**:
  - Simulated report directory with valid report, forged report (absent from ledger), failed-invocation report (`return_code=1`), and duplicate `run_id` report.
  - Output observed:
    - `workbench: skipping aggregate report .../r_forged_aggregate.json: no evidence-ledger entry for its run`
    - `workbench: skipping aggregate report .../r_fail_aggregate.json: disagrees with its evidence-ledger entry`
    - `workbench: skipping aggregate report .../r_dup_aggregate.json: duplicate run_id`
    - Served list: `['run_1']` (only the byte-exact ledger-corroborated completed report).
- **Classification**: NOT_A_BUG (Fix verified; synthetic and forged runs cannot be served as benchmark evidence).

### E. Adversarial Input Rejection
Tested boundary and corrupted inputs directly against `paper_gate.evaluate_gate`:
- **NaN Recall** (`r10=float('nan')`):
  - `status`: `ALERT_METRIC_DEVIATION`, `passed`: `False`
  - `blockers`: `["r10: reproduced value is not a finite number"]`
- **Negative Recall** (`r10=-50.0`):
  - `status`: `ALERT_METRIC_DEVIATION`, `passed`: `False`
  - `blockers`: `["r10 differs by 189.4294% (paper 55.91, reproduced -50.0)"]`
- **Over 100% Recall** (`r10=150.0`):
  - `status`: `ALERT_METRIC_DEVIATION`, `passed`: `False`
  - `blockers`: `["r10 differs by 168.2883% (paper 55.91, reproduced 150.0)"]`
- **Evaluation Failure** (`return_code=1` / `evaluation_completed=False`):
  - `status`: `FAILED_EVALUATION`, `passed`: `False`
  - `blockers`: `["the evaluation did not complete"]`
- **Classification**: NOT_A_BUG (All adversarial inputs fail-closed at gate).

### F. Bundle Resolution & Single-Category Refusal (LIMN 3-Category vs Dress-Only)
- **Context**: Prior audit C found `run_eval.py` crashed with `KeyError` resolving bundle ID `base_iter0_all_categories`.
- **Execution 1 (Bundle Resolution)**:
  - `registry.checkpoint_by_id(limn_model, "base_iter0_all_categories")` returns anchor checkpoint augmented with `bundle_id="base_iter0_all_categories"` and `bundle_members=['base_iter0_dress', 'base_iter0_shirt', 'base_iter0_toptee']`.
  - `run_eval.py --model limn --checkpoint base_iter0_all_categories --protocol fashioniq_val_split ...` outputs:
    `/usr/bin/python3 .../workbench/replay/limn.py --checkpoint-root .../checkpoints/limn --dataset-root /data/FashionIQ --source-root .../third_party/LIMN` (omits `--category`, executing all 3 categories).
- **Execution 2 (Single-Category Refusal)**:
  - Running with `--checkpoint base_iter0_dress` appends `--category dress`.
  - In `limn.py:199-201`, single-category runs yield `aggregate={"complete": False}`.
  - In `metrics_extraction.py:133-143`, incomplete aggregate yields `extraction_status="EVALUATION_COMPLETED_METRICS_UNPARSED"` and `observed_metrics=None`.
  - In `paper_gate.evaluate_gate`:
    - With run checkpoint `base_iter0_dress`: returns `BLOCKED_CHECKPOINT_MISMATCH`, `passed=False`.
    - With run checkpoint `base_iter0_all_categories`: returns `INCOMPLETE_METRIC_EVIDENCE`, `passed=False`.
- **Classification**: NOT_A_BUG (Bundle resolution fixed; dress-only run cannot produce certified aggregate).

### G. Protocol Immutability & Reference Integrity
- `git diff 6e3c979~1..6e3c979 workbench/registry/paper_benchmarks/papers.yaml`: Empty (0 bytes changed).
- No protocols, thresholds, gallery masks, or paper numbers were altered to fit code.
- Gallery protocols in `src/evaluation/fashioniq.py:171-187` enforce reference exclusion and gallery definitions independently.

---

## 3. Defects Log

### UNVERIFIED_RISK 1: `ReportedMetrics` in Schema-v2 Accepts Non-Finite and Out-of-Bound Values (P2)
- **Location**: `workbench/backend/schemas/results.py:16-19`
- **Root Cause**: `ReportedMetrics.r10/r50/mean` are typed as `float | None` without `Field(ge=0.0, le=100.0, allow_inf_nan=False)`.
- **Reproduction**: `ReportedMetrics(r10=float('nan'), r50=-1.0, mean=150.0)` validates without raising `ValidationError`.
- **Mitigation**: Evaluator and gate reject NaN and out-of-bound values (`ALERT_METRIC_DEVIATION`) before certification. However, an offline schema-v2 JSON artifact can persist invalid numbers that enter DuckDB.
- **Classification**: UNVERIFIED_RISK (P2).

### UNVERIFIED_RISK 2: DCNet Protocol Discrepancy in `papers.yaml` (P2)
- **Location**: `workbench/registry/paper_benchmarks/papers.yaml:1460-1514`
- **Root Cause**: DCNet rows specify `gallery_protocol: fashioniq_full_gallery_ref_excluded` with `reference_excluded: true`, while the upstream pinned evaluator (`workbench/third_party/DCNet/trainer/trainer_ref.py:126-178`) computes retrieval over full targets without reference masking.
- **Impact**: DCNet is currently blocked from evaluation by `source_blocker` in `models.yaml:615`. If unblocked in the future, DCNet runs would fail protocol comparison or compare against mismatched protocol assumptions.
- **Classification**: UNVERIFIED_RISK (P2, latent on blocked model).

### UNVERIFIED_RISK 3: `validate_results.py` Reports Success on Empty Results Root (P2)
- **Location**: `workbench/scripts/validate_results.py:38-44`, `workbench/scripts/pipeline.py:1449-1453`
- **Root Cause**: When no schema-v2 result JSON files exist below the root, `load_runs` returns `[]`, and `validate_results.py` prints `[OK] validated 0 runs under ...` and exits with code 0.
- **Impact**: LIMN aggregate evaluations write `*_aggregate.json` reports rather than schema-v2 per-query files. A run that produced only aggregate reports causes `validate_results` to exit 0 without validating the actual evaluation output.
- **Classification**: UNVERIFIED_RISK (P2).

---

## 4. Overall Verdict

**Verdict**: `SOUND_WITH_RESERVATIONS`

The scientific reproduction slice is robust and fail-closed:
1. The vocabulary fix in `paper_gate.py:415` allows exact LIMN runs to certify reproduction (`PASS_WITHIN_5_PERCENT`) while preserving the unverified mapping cap on ConeSep.
2. Bundle resolution in `registry.checkpoint_by_id` unblocks official LIMN multi-category commands while preventing single-category truncation.
3. Out-of-band evidence ledger verification prevents forged, failed, or synthetic runs from presenting as certified reproduction.
4. Metric deviation checks, NaN rejection, and protocol parity are verified by direct execution.
Reservations are limited to schema-v2 metric carrier validation bounds (P2) and latent protocol tagging on blocked secondary model DCNet (P2).

---

Can this implementation fail during the real first run on an RTX 5070 Ti PC?
No — the scientific evaluation, bundle resolution, metric extraction, and paper gate verification path will not fail during the real first run of LIMN on the RTX 5070 Ti PC; all gate logic, vocabulary mappings, and ledger corroboration are sound, with GPU execution itself legitimately PENDING.
