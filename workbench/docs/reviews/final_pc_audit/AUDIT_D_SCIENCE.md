# Final Source Audit Report: Slice D (Scientific Integrity, FashionIQ Protocols, Benchmark References, Paper Gate, Evidence Ledger, and Verifier Contracts)

## 1. Scope & Target Files Audited
Audited files:
- `workbench/backend/paper_gate.py` (475 lines) — numerical acceptance gate, 5% relative error, zero-baseline equality, comparability fields, mapping-status cap.
- `workbench/backend/benchmark_references.py` (195 lines) — benchmark registry loader, condition-based reference matching (`select_reference`).
- `workbench/backend/checkpoint_selection.py` (263 lines) — checkpoint selection policies (`strict_clean` vs `include_flagged_conditions`), requested checkpoint resolution.
- `workbench/backend/schemas/results.py` (132 lines) — canonical schema-v2 contracts, protocol IDs, `RunMetadata`, `QueryResult`, `TopResult`, `ResultRun`.
- `workbench/backend/schemas/__init__.py` (1 line) — schema package export.
- `workbench/scripts/verify_run.py` (342 lines) — end-to-end verifier CLI, report identity, HMAC ledger corroboration, completion proof validation, metric evidence reasons.
- `workbench/registry/paper_benchmarks/papers.yaml` (1837 lines, 33 rows) — committed benchmark reference registry.
- `workbench/scripts/evaluate_models.py` (2003 lines) — report generation, HMAC evidence ledger writer (`append_evidence_entry`), proof writer, completion validator (`validate_run_manifest`).
- `workbench/replay/common.py` (108 lines) — replay harness, unweighted category macro aggregation (`macro_aggregate`), trusted whole model unpickling, source pin check.
- Supporting protocol files: `src/evaluation/fashioniq.py`, `src/datasets/fashioniq.py`, `workbench/backend/metrics_extraction.py`, `workbench/backend/registry.py`.

---

## 2. Executive Summary & Findings Matrix

| Finding ID | Classification | Severity | Target File:Line | Summary |
|---|---|---|---|---|
| **D-1** | **CONFIRMED** | **HIGH** | `workbench/scripts/verify_run.py:313-316`, `:266-301` | **Completeness contract bypass on all non-LIMN models**: `_published_reasons` ignores `extraction_status`, ignores `metric_extraction_available`, ignores category metrics, ignores macro consistency, and uses `any()` so a non-LIMN report with unextracted metrics, missing categories, contradictory macro, or `NaN` values is certified as `verified: True`. |
| **D-2** | **CONFIRMED** | **HIGH** | `workbench/backend/benchmark_references.py:187-188`, `workbench/registry/paper_benchmarks/papers.yaml:1660-1773`, `workbench/scripts/verify_run.py:285-294` | **Legitimate run refusal on `ptha_mtst` due to reference ambiguity**: `papers.yaml` holds two verified rows for `ptha_mtst` with `noise=null` under `fashioniq_original_split`. `select_reference` matches both and returns `None`, causing `verify_run.py` to unconditionally reject genuine executions of `ptha_mtst`. |
| **D-3** | **CONFIRMED** | **MEDIUM** | `workbench/backend/schemas/results.py:79` | **Schema rejection of `"synthetic"` data kind**: `RunMetadata.data_kind` only allows `Literal["experiment", "mock"] = "experiment"`. Pipeline code (`evaluate_models.py:185`) and gate (`paper_gate.py:312`) declare `data_kind="synthetic"`, which fails Pydantic schema-v2 validation if serialized. |
| **D-4** | **CONFIRMED** | **MEDIUM** | `workbench/backend/benchmark_references.py:168-171`, `workbench/registry/paper_benchmarks/papers.yaml:293-354`, `workbench/scripts/verify_run.py:291-294` | **Legitimate run refusal on `combiner_rn50x4_noft`**: `combiner_rn50x4_noft` has implemented extraction parser and registry entry, but all committed references are `UNVERIFIED` (authors never reported noft FashionIQ numbers). `verify_run.py` unconditionally refuses all genuine runs of this model. |
| **D-5** | **NOT_A_BUG** | **INFORMATIONAL** | `workbench/scripts/verify_run.py:277-280` | **Verifier architecture deliberately decouples evidence verification from paper gate pass**: `verify_run.py` certifies authentic execution and evidence integrity, not paper score reproduction. A large deviation produces `GateVerdict(ALERT_METRIC_DEVIATION, passed=False)` but exits 0 with `verified: True`. |

---

## 3. Detailed Findings

### Finding D-1: Completeness Contract Bypass on Non-LIMN Models
- **Classification**: CONFIRMED
- **Severity**: HIGH
- **File:Line**: `workbench/scripts/verify_run.py:313-316`, `workbench/scripts/verify_run.py:266-301`
- **Root Cause**:
  In `verify_run.py`:
  ```python
  def metric_evidence_reasons(model_id: str, checkpoint_id: str, protocol_id: str,
                              report: dict | None) -> list[str]:
      reasons = _evidence_kind_reasons(report)
      if model_id == "limn":
          return reasons + _limn_reasons(report)
      return reasons + _published_reasons(model_id, checkpoint_id, protocol_id, report)
  ```
  `_limn_reasons` enforces that:
  1. `metric_extraction_available is True`
  2. `extraction_status == EXTRACTION_COMPLETED`
  3. All three categories (`dress`, `shirt`, `toptee`) are present with finite `r10` and `r50`
  4. Macro `r10`, `r50`, and `mean` equal the mean of category recalls within 0.05 pp.

  However, for every other model (`csmcir`, `clip4cir_rn50x4_fullft`, `sprc`, etc.), `_limn_reasons` is completely skipped! Instead, `_published_reasons` runs:
  ```python
  published = ([key for key in paper_gate.METRIC_KEYS if reference.value_for(key) is not None]
               if reference is not None else [])
  ...
  observed = report.get("observed_metrics")
  if not isinstance(observed, dict) or not any(_is_recall(observed.get(key))
                                               for key in published):
      return [...]
  return []
  ```
  `_published_reasons` checks ONLY that `any(_is_recall(observed.get(key)) for key in published)` is true. It never checks `extraction_status`, never checks `metric_extraction_available`, never checks `category_metrics`, never checks macro consistency, and allows other published metrics to be `NaN` or out-of-range as long as at least one metric is valid.
- **Reproduction**:
  Executed probe with CSMCIR report where `metric_extraction_available: False`, `extraction_status: "EVALUATION_COMPLETED_METRICS_UNPARSED"`, category metrics set to 10.0%, and `observed_metrics` set to `{"r10": 57.07, "r50": float("nan"), "mean": 999.9}`:
  Command:
  ```bash
  .venv/bin/python -c "
  import tempfile, os, json, hashlib
  from pathlib import Path
  import workbench.scripts.verify_run as vr
  from workbench.backend.operator_config import WorkbenchConfig
  from workbench.scripts.evaluate_models import append_evidence_entry

  with tempfile.TemporaryDirectory() as tmp:
      sandbox = Path(tmp); run_id = 'run_csmcir_attack'
      model_id = 'csmcir'; checkpoint_id = 'csmcir_fiq'; protocol_id = 'fashioniq_original_split'
      pin = '774f94e2076ff17ea91703a6239d2a08f0e1a44e'
      reports_dir = sandbox / 'workbench' / 'artifacts' / 'reports'; reports_dir.mkdir(parents=True, exist_ok=True)
      rep_file = reports_dir / f'{model_id}_{checkpoint_id}_{run_id}_aggregate.json'
      rep_data = {
          'run_id': run_id, 'model_id': model_id, 'checkpoint_id': checkpoint_id, 'protocol_id': protocol_id,
          'execution_status': 'EXECUTION_COMPLETED', 'return_code': 0, 'invocation_status': 'INVOCATION_COMPLETED',
          'checkpoint_sha256': '123456', 'is_synthetic': False, 'data_kind': 'experiment',
          'metric_extraction_available': False,
          'extraction_status': 'EVALUATION_COMPLETED_METRICS_UNPARSED',
          'category_metrics': {'dress': {'r10': 10.0, 'r50': 10.0}, 'shirt': {'r10': 10.0, 'r50': 10.0}, 'toptee': {'r10': 10.0, 'r50': 10.0}},
          'observed_metrics': {'r10': 57.07, 'r50': float('nan'), 'mean': 999.9}
      }
      rep_bytes = json.dumps(rep_data, indent=2).encode('utf-8')
      rep_file.write_bytes(rep_bytes)
      actual_digest = hashlib.sha256(rep_bytes).hexdigest()
      config = WorkbenchConfig(CIR_REPO_ROOT=sandbox, CIR_DATA_ROOT=sandbox/'data', FASHIONIQ_ROOT=sandbox/'data'/'FashionIQ', WORKBENCH_HOST='127.0.0.1', WORKBENCH_BACKEND_PORT=8000, WORKBENCH_FRONTEND_PORT=5173, WORKBENCH_CHECKPOINT_ROOT=sandbox/'workbench'/'artifacts'/'checkpoints', WORKBENCH_RESULTS_ROOT=sandbox/'workbench'/'artifacts'/'results', WORKBENCH_THIRD_PARTY_ROOT=sandbox/'workbench'/'third_party')
      os.environ['WORKBENCH_LEDGER_KEY'] = 'mock_ledger_key_for_test_harness_32b!'
      rec = append_evidence_entry(config, {'run_id': run_id, 'model_id': model_id, 'checkpoint_id': checkpoint_id, 'protocol_id': protocol_id, 'return_code': 0, 'invocation_status': 'INVOCATION_COMPLETED', 'report_digest': actual_digest})
      run_dir = sandbox / 'workbench' / 'artifacts' / 'logs' / f'{model_id}__{checkpoint_id}__{run_id}'; run_dir.mkdir(parents=True, exist_ok=True)
      logs = {'stdout': 'out', 'stderr': ''}; ev_digest = hashlib.sha256(json.dumps(logs, sort_keys=True).encode('utf-8')).hexdigest()
      proof_file = reports_dir / f'{model_id}_{checkpoint_id}_{run_id}_completion.json'
      proof_data = {
          'artifact_type': 'official_evaluation_completion_proof', 'schema_version': 1, 'completion_status': 'COMPLETED',
          'run_id': run_id, 'model_id': model_id, 'checkpoint_id': checkpoint_id, 'protocol_id': protocol_id,
          'source_expected_commit': pin, 'source_actual_commit': pin, 'checkpoint_artifact_type': 'file',
          'checkpoint_sha256': '123456', 'bundle_manifest_digest': None, 'invocation_status': 'INVOCATION_COMPLETED',
          'return_code': 0, 'ledger_entry_digest': rec['entry_digest'], 'report': str(rep_file), 'report_digest': actual_digest,
          'report_relative': rep_file.name, 'environment_digest': 'env123', 'command_digest': 'cmd123', 'evidence_digest': ev_digest,
          'run_directory': str(run_dir), 'stdout_log': 'stdout.log', 'stderr_log': 'stderr.log', 'completed_at': '2026-10-16T00:00:00Z'
      }
      proof_file.write_text(json.dumps(proof_data, indent=2))
      (run_dir / 'stdout.log').write_text('out'); (run_dir / 'stderr.log').write_text(''); (run_dir / 'command.json').write_text(json.dumps(proof_data))
      os.environ['CIR_REPO_ROOT'] = str(sandbox); os.environ['WORKBENCH_ARTIFACTS_ROOT'] = str(sandbox / 'workbench' / 'artifacts')
      os.environ['WORKBENCH_CHECKPOINT_ROOT'] = str(sandbox / 'workbench' / 'artifacts' / 'checkpoints'); os.environ['WORKBENCH_RESULTS_ROOT'] = str(sandbox / 'workbench' / 'artifacts' / 'results')
      verdict = vr.verify(run_id, model_id, checkpoint_id, protocol_id)
      print('verified:', verdict['verified'], 'reasons:', verdict['reasons'])
  "
  ```
  Observed Output:
  `verified: True reasons: []`
- **Minimal Corrective Action**:
  In `workbench/scripts/verify_run.py`:
  1. Extend `_limn_reasons` checks (or extract a shared FashionIQ metric evidence validator) to all models that evaluate FashionIQ: require `metric_extraction_available is True` and `extraction_status == EXTRACTION_COMPLETED`.
  2. In `_published_reasons`, require that ALL published metrics present in `observed_metrics` must be valid finite recalls (`all(_is_recall(observed.get(key)) for key in published)`).
  3. When a non-LIMN report carries `category_metrics`, validate that macro metrics in `observed_metrics` agree with the mean of the categories.

---

### Finding D-2: Reference Selection Ambiguity Refuses Legitimate `ptha_mtst` Runs
- **Classification**: CONFIRMED
- **Severity**: HIGH
- **File:Line**: `workbench/backend/benchmark_references.py:187-188`, `workbench/registry/paper_benchmarks/papers.yaml:1660-1773`, `workbench/scripts/verify_run.py:285-294`
- **Root Cause**:
  `workbench/registry/paper_benchmarks/papers.yaml` contains two committed `reference_status: VERIFIED` rows for `ptha_mtst`:
  - `ptha_mtst_fiq_table3_fashiqmtst_blip2`: `gallery_protocol: fashioniq_original_split`, `checkpoint_training_noise_pct: null`.
  - `ptha_mtst_fiq_no_pretrain_table3`: `gallery_protocol: fashioniq_original_split`, `checkpoint_training_noise_pct: null`.

  In `benchmark_references.py:172-188`:
  ```python
  def matches(candidate: BenchmarkReference) -> bool:
      record = candidate.payload
      if checkpoint_training_noise_pct is not None:
          recorded = record.get("checkpoint_training_noise_pct")
          if recorded is None or int(recorded) != int(checkpoint_training_noise_pct):
              return False
      if protocol_id is not None:
          recorded = record.get("gallery_protocol")
          if recorded is not None and recorded != protocol_id:
              return False
      return True

  exact = [c for c in verified if matches(c)]
  if len(exact) == 1:
      return exact[0], "exact condition match"
  if len(exact) > 1:
      return None, f"more than one verified reference matches this run: {sorted(c.reference_id for c in exact)}"
  ```
  Because both rows declare `checkpoint_training_noise_pct: null` and `protocol_id: fashioniq_original_split`, `exact` contains 2 candidates. `select_reference` returns `(None, f"more than one verified reference matches this run: ...")`.
  Consequently, `verify_run.py:291-294` unconditionally refuses ANY genuine run of `ptha_mtst`.
- **Reproduction**:
  Command:
  ```bash
  .venv/bin/python -c "
  from workbench.backend import benchmark_references as br
  ref, reason = br.select_reference('ptha_mtst', checkpoint_training_noise_pct=None, protocol_id='fashioniq_original_split')
  print('ref:', ref, '| reason:', reason)
  "
  ```
  Observed Output:
  `ref: None | reason: more than one verified reference matches this run: ['ptha_mtst_fiq_no_pretrain_table3', 'ptha_mtst_fiq_table3_fashiqmtst_blip2']`

  End-to-end verification probe against `verify_run.py`:
  Observed Output:
  `verified: False reasons: ["ptha_mtst: more than one verified reference matches this run: ['ptha_mtst_fiq_no_pretrain_table3', 'ptha_mtst_fiq_table3_fashiqmtst_blip2']; no verified committed reference publishes a metric for checkpoint None under protocol fashioniq_original_split, so this run cannot carry the published-vs-reproduced comparison its protocol would need"]`
- **Minimal Corrective Action**:
  Disambiguate reference rows for `ptha_mtst` by adding distinguishing metadata in `papers.yaml` (e.g. `checkpoint_id: fashioniq_mtst_blip2` vs `fashioniq_no_pretrain`), or extend `select_reference` matching criteria to include pretraining condition or checkpoint variant ID.

---

### Finding D-3: Pydantic Schema `data_kind` Literal Rejects `"synthetic"`
- **Classification**: CONFIRMED
- **Severity**: MEDIUM
- **File:Line**: `workbench/backend/schemas/results.py:79`
- **Root Cause**:
  `workbench/backend/schemas/results.py:79` defines:
  `data_kind: Literal["experiment", "mock"] = "experiment"`
  However, both `evaluate_models.py:185` and `paper_gate.py:312-316` explicitly use `data_kind="synthetic"` for pipeline validation runs. If a synthetic run is passed to schema-v2 validation (`ResultRun` / `RunMetadata`), Pydantic raises `pydantic_core._pydantic_core.ValidationError: Input should be 'experiment' or 'mock'`.
- **Reproduction**:
  Command:
  ```bash
  .venv/bin/python -c "
  from datetime import datetime, timezone
  from workbench.backend.schemas.results import RunMetadata
  try:
      meta = RunMetadata(
          run_id='syn_1', dataset='FashionIQ', split='val', protocol_id='fashioniq_val_split',
          literature_split_label='val', evaluation_noise_pct=0, model_id='limn', method_name='LIMN',
          checkpoint_id='base_iter0_all_categories', top_k_saved=50, gallery_size=2628,
          timestamp=datetime.now(timezone.utc), data_kind='synthetic'
      )
      print('Validated:', meta.data_kind)
  except Exception as e:
      print('Caught:', type(e).__name__, str(e)[:100])
  "
  ```
  Observed Output:
  `Caught: ValidationError 1 validation error for RunMetadata\ndata_kind\n  Input should be 'experiment' or 'mock' [type=literal_error`
- **Minimal Corrective Action**:
  In `workbench/backend/schemas/results.py:79`, update the field definition:
  `data_kind: Literal["experiment", "mock", "synthetic"] = "experiment"`

---

### Finding D-4: Legitimate Run Refusal on `combiner_rn50x4_noft`
- **Classification**: CONFIRMED
- **Severity**: MEDIUM
- **File:Line**: `workbench/backend/benchmark_references.py:168-171`, `workbench/registry/paper_benchmarks/papers.yaml:293-354`, `workbench/scripts/verify_run.py:291-294`
- **Root Cause**:
  `combiner_rn50x4_noft` is registered in `models.yaml:660-719`, has an implemented extraction parser in `metrics_extraction.py:309` (`parse_combiner_output`), and can be executed via its adapter. However, the original authors of CLIP4Cir TOMM 2023 never reported numerical FashionIQ validation metrics for the `noft` variant (only for `fullft`). All three rows in `papers.yaml:293-354` are marked `reference_status: UNVERIFIED`. `select_reference` requires `reference_status == REFERENCE_VERIFIED`, so it returns `None, "combiner_rn50x4_noft has references but none is verified: ['UNVERIFIED']"`.
  When `verify_run.py` checks the run, `published` is empty, causing unconditional refusal.
- **Reproduction**:
  Command:
  ```bash
  .venv/bin/python -c "
  from workbench.backend import benchmark_references as br
  ref, reason = br.select_reference('combiner_rn50x4_noft', checkpoint_training_noise_pct=0, protocol_id='fashioniq_original_split')
  print('ref:', ref, '| reason:', reason)
  "
  ```
  Observed Output:
  `ref: None | reason: combiner_rn50x4_noft has references but none is verified: ['UNVERIFIED']`
- **Minimal Corrective Action**:
  Document in the model registry and operator runbook that `combiner_rn50x4_noft` is an unbenchmarked reference variant by design, or provide an explicit verification mode for baseline variants without paper references.

---

### Finding D-5: Distinction Between Evidence Verification and 5% Paper Gate Acceptance
- **Classification**: NOT_A_BUG (ARCHITECTURAL DESIGN)
- **Severity**: INFORMATIONAL
- **File:Line**: `workbench/scripts/verify_run.py:277-280`
- **Analysis**:
  `verify_run.py` evaluates execution authenticity, ledger witness corroboration, and completeness contract, but **deliberately does not assert that `benchmark_gate.passed` is True**.
  As documented in `verify_run.py:277-280`:
  *"A measured deviation is deliberately not a refusal here. It is a research finding about a real execution, and the recipe for it is already on the report as the gate's own ALERT_METRIC_DEVIATION; republishing a finding as a failed invocation would say the run did not happen when it did. What this refuses is a run that cannot be compared at all."*
  Therefore, an honest execution observing Recall of 10.0% against a published 55.91% is certified as `verified: True` (authentic scientific evidence), while the gate status `ALERT_METRIC_DEVIATION` and `passed: False` are faithfully preserved in the report.

---

## 4. Adversarial Attack Probes on `verify_run.py` (Both Directions)

### Direction 1: Attacks Attempting to Make the Verifier Certify an Invalid Run

#### Probe 1.1: Tampered Report (Altered Report Post-Ledger)
- **Target**: Evidence ledger digest check & completion proof checksum.
- **Probe Command**:
  ```bash
  .venv/bin/python -c "
  import tempfile, os, json, hashlib
  from pathlib import Path
  from workbench.tests.test_pc_launcher import _create_corroborated_run
  import workbench.scripts.verify_run as vr

  with tempfile.TemporaryDirectory() as tmp:
      sandbox = Path(tmp); run_id = 'test_tampered_report'
      _create_corroborated_run(sandbox, run_id=run_id)
      rep_file = sandbox / 'workbench' / 'artifacts' / 'reports' / f'limn_base_iter0_all_categories_{run_id}_aggregate.json'
      data = json.loads(rep_file.read_text())
      data['tampered'] = True
      rep_file.write_text(json.dumps(data, indent=2))
      os.environ['CIR_REPO_ROOT'] = str(sandbox)
      os.environ['WORKBENCH_ARTIFACTS_ROOT'] = str(sandbox / 'workbench' / 'artifacts')
      os.environ['WORKBENCH_CHECKPOINT_ROOT'] = str(sandbox / 'workbench' / 'artifacts' / 'checkpoints')
      os.environ['WORKBENCH_RESULTS_ROOT'] = str(sandbox / 'workbench' / 'artifacts' / 'results')
      os.environ['WORKBENCH_LEDGER_KEY'] = 'mock_ledger_key_for_test_harness_32b!'
      verdict = vr.verify(run_id, 'limn', 'base_iter0_all_categories', 'fashioniq_val_split')
      print('verified:', verdict['verified'], 'reasons:', verdict['reasons'])
  "
  ```
- **Observed Result**:
  `verified: False reasons: ["the evidence ledger's report digest does not match the report file", 'completion proof rejected: declared output report digest does not match its report']`
- **Result**: REFUSED (Pass). Tampering detected by dual ledger & proof digest verification.

---

#### Probe 1.2: Replayed Completion Proof Across Runs
- **Target**: Replaying proof from legitimate run A to masquerade as run B.
- **Probe 1.2a (Direct Copy)**:
  Replayed proof from `run_legit_a` to `run_replayed_b` without modifying fields:
  Observed Result:
  `verified: False reasons: ['the evidence ledger holds no entry for run run_replayed_b', "completion proof rejected: declared output belongs to run 'run_legit_a', not the current run 'run_replayed_b'"]`
- **Probe 1.2b (Modified `run_id` in proof, pointing to run A report)**:
  Observed Result:
  `verified: False reasons: ["completion proof rejected: declared output run ('run_replayed_b') disagrees with its report ('run_legit_a')"]`
- **Probe 1.2c (Modified `run_id` and report, keeping run A ledger digest)**:
  Observed Result:
  `verified: False reasons: ['completion proof rejected: declared output names a different evidence-ledger entry than the ledger holds']`
- **Probe 1.2d (Modified `run_id`, report, and ledger digest, but replaying run A log directory)**:
  Observed Result:
  `verified: False reasons: ["completion proof rejected: declared output belongs to run directory 'limn__base_iter0_all_categories__run_legit_a', expected this invocation's 'limn__base_iter0_all_categories__run_replayed_b'"]`
- **Result**: REFUSED (Pass). Run directory naming convention, ledger entry chaining, and report bindings prevent cross-run replay.

---

#### Probe 1.3: Macro Inconsistent with Category Averages
- **Target**: Macro arithmetic verification in `_macro_reasons` (`verify_run.py:185-209`).
- **Probe 1.3a (Macro R@10 = 99.0 vs Category Mean = 27.9333)**:
  Observed Result:
  `verified: False reasons: ["the report's macro r10 (99.0) is inconsistent with its categories (their mean is 27.933333333333334, tolerance 0.05 pp)", "the report's macro mean (78.45) is inconsistent with its categories (their mean is 42.916666666666664, tolerance 0.05 pp)"]`
- **Probe 1.3b (Macro R@50 = 99.0 vs Category Mean = 57.9)**:
  Observed Result:
  `verified: False reasons: ["the report's macro r50 (99.0) is inconsistent with its categories (their mean is 57.9, tolerance 0.05 pp)", "the report's macro mean (63.4667) is inconsistent with its categories (their mean is 42.916666666666664, tolerance 0.05 pp)"]`
- **Probe 1.3c (Macro Mean = 99.0 vs `(r10 + r50) / 2 = 42.9167`)**:
  Observed Result:
  `verified: False reasons: ["the report's macro mean (99.0) is inconsistent with its categories (their mean is 42.916666666666664, tolerance 0.05 pp)"]`
- **Probe 1.3d (`observed_metrics: None`)**:
  Observed Result:
  `verified: False reasons: ['the report records no macro aggregate to check against its categories']`
- **Result**: REFUSED (Pass for LIMN). Verifier re-derives unweighted category mean and enforces `MACRO_TOLERANCE_PP = 0.05`.

---

#### Probe 1.4: Metric Extraction Not Performed
- **Target**: Verification of extraction execution in `_limn_reasons` (`verify_run.py:230-253`).
- **Probe 1.4a (`metric_extraction_available: False`, `extraction_status: "EXTRACTED_VERIFIED"`)**:
  Observed Result:
  `verified: False reasons: ['the report records metric_extraction_available=False: no metric extraction was performed for this run']`
- **Probe 1.4b (`metric_extraction_available: True`, `extraction_status: "EVALUATION_COMPLETED_METRICS_UNPARSED"`)**:
  Observed Result:
  `verified: False reasons: ["the report's extraction_status is 'EVALUATION_COMPLETED_METRICS_UNPARSED', not 'EXTRACTED_VERIFIED': the run's metrics were not extracted"]`
- **Probe 1.4c (`extraction_status: None`)**:
  Observed Result:
  `verified: False reasons: ["the report's extraction_status is None, not 'EXTRACTED_VERIFIED': the run's metrics were not extracted"]`
- **Result**: REFUSED (Pass for LIMN). Unextracted metrics reliably refused on LIMN.

---

#### Probe 1.5: Synthetic Run Presenting as Real Experiment
- **Target**: `_evidence_kind_reasons` (`verify_run.py:212-228`).
- **Probe 1.5a (`is_synthetic: True`)**:
  Observed Result:
  `verified: False reasons: ['the report declares itself synthetic (is_synthetic), so it is a pipeline self-test and not evidence about reproduction']`
- **Probe 1.5b (`data_kind: "synthetic"`)**:
  Observed Result:
  `verified: False reasons: ["the report's data_kind is 'synthetic', which is not a real experiment"]`
- **Probe 1.5c (`data_kind: "mock"`)**:
  Observed Result:
  `verified: False reasons: ["the report's data_kind is 'mock', which is not a real experiment"]`
- **Probe 1.5d (`data_kind: "arbitrary_string"`)**:
  Observed Result:
  `verified: False reasons: ["the report's data_kind is 'arbitrary_string', which is not a real experiment"]`
- **Result**: REFUSED (Pass). Synthetic and mock runs cannot pass verifier.

---

#### Probe 1.6: Metric Bounds & Type Invalidity
- **Target**: `_is_recall` function (`verify_run.py:61-65`).
- **Probe 1.6a (`category dress r10 = NaN`)**:
  Observed: `verified: False reasons: ['category dress r10 is not a finite Recall in [0, 100]: nan']`
- **Probe 1.6b (`category dress r10 = -0.01`)**:
  Observed: `verified: False reasons: ['category dress r10 is not a finite Recall in [0, 100]: -0.01']`
- **Probe 1.6c (`category dress r10 = 100.01`)**:
  Observed: `verified: False reasons: ['category dress r10 is not a finite Recall in [0, 100]: 100.01']`
- **Probe 1.6d (`category dress r10 = True` boolean)**:
  Observed: `verified: False reasons: ['category dress r10 is not a finite Recall in [0, 100]: True']`
- **Probe 1.6e (`category dress r10 = False` boolean)**:
  Observed: `verified: False reasons: ['category dress r10 is not a finite Recall in [0, 100]: False']`
- **Probe 1.6f (`macro r10 = True` boolean)**:
  Observed: `verified: False reasons: ["the report's macro r10 is not a finite Recall in [0, 100]: True"]`
- **Probe 1.6g (`macro r10 = NaN`)**:
  Observed: `verified: False reasons: ["the report's macro r10 is not a finite Recall in [0, 100]: nan"]`
- **Probe 1.6h (`macro r10 = 100.05`)**:
  Observed: `verified: False reasons: ["the report's macro r10 is not a finite Recall in [0, 100]: 100.05"]`
- **Result**: REFUSED (Pass for LIMN). Strict boundary and type enforcement blocks boolean and non-finite values.

---

#### Probe 1.7: Non-LIMN Completeness Contract Bypass (CONFIRMED DEFECT D-1)
- **Target**: `_published_reasons` (`verify_run.py:266-301`).
- **Probe 1.7a (CSMCIR with unparsed extraction and empty categories)**:
  Report payload: `metric_extraction_available: False`, `extraction_status: "EVALUATION_COMPLETED_METRICS_UNPARSED"`, `category_metrics: {}`, `observed_metrics: {"r10": 57.07, "r50": 77.27, "mean": 67.17}`.
  Observed Output: `verified: True reasons: []`
  **FAIL**: Fabricated metrics with failed extraction certified as authentic evidence.
- **Probe 1.7b (CSMCIR with contradictory category metrics)**:
  Report payload: Category recalls = 10.0%, Macro observed = 57.07%.
  Observed Output: `verified: True reasons: []`
  **FAIL**: Blatant contradiction between category recalls and macro aggregate certified as authentic evidence.
- **Probe 1.7c (CSMCIR with NaN and out-of-range 999.9 in `observed_metrics`)**:
  Report payload: `observed_metrics: {"r10": 57.07, "r50": float("nan"), "mean": 999.9}`.
  Observed Output: `verified: True reasons: []`
  **FAIL**: Because `any(_is_recall(...))` matched `r10`, the presence of `NaN` and `999.9` was ignored and certified.

---

### Direction 2: Attacks Attempting to Make the Verifier Refuse a Fully Consistent Legitimate Run

#### Probe 2.1: Legitimate LIMN Run with Boundary Recalls at Exactly 0.0
- **Scenario**: Fully corroborated LIMN run where all category recalls and macro aggregate sit exactly on `0.0` (zero retrieval hits across all categories).
- **Execution**:
  ```python
  category_metrics = {
      'dress': {'r10': 0.0, 'r50': 0.0},
      'shirt': {'r10': 0.0, 'r50': 0.0},
      'toptee': {'r10': 0.0, 'r50': 0.0},
  }
  observed_metrics = {'r10': 0.0, 'r50': 0.0, 'mean': 0.0}
  ```
- **Observed Result**:
  `verified: True reasons: []`
- **Result**: ACCEPTED (Pass). `_is_recall(0.0)` evaluates `0.0 <= 0.0 <= 100.0` as True; macro recomputation `sum(...) / 3 = 0.0` agrees within tolerance.

---

#### Probe 2.2: Legitimate LIMN Run with Boundary Recalls at Exactly 100.0
- **Scenario**: Fully corroborated LIMN run where all category recalls and macro aggregate sit exactly on `100.0` (perfect retrieval across all categories).
- **Execution**:
  ```python
  category_metrics = {
      'dress': {'r10': 100.0, 'r50': 100.0},
      'shirt': {'r10': 100.0, 'r50': 100.0},
      'toptee': {'r10': 100.0, 'r50': 100.0},
  }
  observed_metrics = {'r10': 100.0, 'r50': 100.0, 'mean': 100.0}
  ```
- **Observed Result**:
  `verified: True reasons: []`
- **Result**: ACCEPTED (Pass). `_is_recall(100.0)` evaluates `0.0 <= 100.0 <= 100.0` as True; macro recomputation agrees within tolerance.

---

#### Probe 2.3: Legitimate LIMN Run with Mixed Boundary Recalls (0.0, 50.0, 100.0)
- **Scenario**: Fully corroborated LIMN run with Dress `(r10=0.0, r50=100.0)`, Shirt `(r10=50.0, r50=50.0)`, Toptee `(r10=100.0, r50=0.0)`.
  Macro re-derivation:
  `r10 = (0 + 50 + 100) / 3 = 50.0`
  `r50 = (100 + 50 + 0) / 3 = 50.0`
  `mean = (50.0 + 50.0) / 2 = 50.0`
- **Observed Result**:
  `verified: True reasons: []`
- **Result**: ACCEPTED (Pass). Internal consistency check correctly handles extreme dispersion.

---

#### Probe 2.4: Legitimate Non-LIMN Model with Published Metrics (CSMCIR)
- **Scenario**: Real CSMCIR execution under `fashioniq_original_split`, checkpoint `csmcir_fiq`, with upstream commit pin `774f94e2076ff17ea91703a6239d2a08f0e1a44e`, ledger witness, completion proof, and published metrics `r10: 57.07, r50: 77.27, mean: 67.17`.
- **Observed Result**:
  `verified: True reasons: []`
- **Result**: ACCEPTED (Pass). Legitimate non-LIMN execution with valid reference correctly certifies.

---

#### Probe 2.5: Legitimate Non-LIMN Run Refused by Reference Ambiguity (`ptha_mtst` — CONFIRMED DEFECT D-2)
- **Scenario**: Fully consistent, proof-backed execution of `ptha_mtst` under `fashioniq_original_split`.
- **Observed Result**:
  `verified: False reasons: ["ptha_mtst: more than one verified reference matches this run: ['ptha_mtst_fiq_no_pretrain_table3', 'ptha_mtst_fiq_table3_fashiqmtst_blip2']; no verified committed reference publishes a metric for checkpoint None under protocol fashioniq_original_split, so this run cannot carry the published-vs-reproduced comparison its protocol would need"]`
- **Result**: REFUSED (False Failure). Verifier rejects legitimate run due to reference selection ambiguity.

---

#### Probe 2.6: Legitimate Non-LIMN Run Refused by Unverified Reference (`combiner_rn50x4_noft` — CONFIRMED DEFECT D-4)
- **Scenario**: Genuine execution of `combiner_rn50x4_noft` under `fashioniq_original_split`.
- **Observed Result**:
  `verified: False reasons: ["combiner_rn50x4_noft: combiner_rn50x4_noft has references but none is verified: ['UNVERIFIED']; no verified committed reference publishes a metric for checkpoint rn50x4_noft_unresolved under protocol fashioniq_original_split, so this run cannot carry the published-vs-reproduced comparison its protocol would need"]`
- **Result**: REFUSED (False Failure). Registered model cannot be verified because paper published no baseline numbers.

---

## 5. Scientific Protocol & Methodology Audit

### 5.1 FashionIQ Protocol Correctness & Candidate Universes
Audit of `src/evaluation/fashioniq.py`, `src/datasets/fashioniq.py`, and actual dataset files on disk at `data/FashionIQ`:
- **Protocol 1: `fashioniq_val_split`**
  - **Gallery Construction**: `build_pair_union_gallery` (`src/datasets/fashioniq.py:186-209`). Constructs ordered, duplicate-free union of reference IDs and target IDs from `cap.{category}.val.json`.
  - **Reference Policy**: `reference_excluded = True`. Each query's reference ID is removed from the ranked candidate list before Top-K evaluation (`ranking = ranking[ranking != reference_column]`).
  - **Measured Dimensions on Disk**:
    * `dress`: 2,017 queries; gallery = 2,628 images.
    * `shirt`: 2,038 queries; gallery = 3,089 images.
    * `toptee`: 1,961 queries; gallery = 2,902 images.
    * Total: 6,016 queries.
  - **Models**: CLVC-Net, HINT, ENCODER, PAIR, TG-CIR, LIMN.
- **Protocol 2: `fashioniq_full_gallery_ref_excluded`**
  - **Gallery Construction**: `build_original_gallery` (`src/evaluation/fashioniq.py:121-129`). Uses full split from `split.{category}.val.json`.
  - **Reference Policy**: `reference_excluded = True`.
  - **Measured Dimensions on Disk**:
    * `dress`: 2,017 queries; gallery = 3,817 images (+45.2% vs pair_union).
    * `shirt`: 2,038 queries; gallery = 6,346 images (+105.4% vs pair_union).
    * `toptee`: 1,961 queries; gallery = 5,373 images (+85.1% vs pair_union).
  - **Models**: Air-Know, ConeSep, HABIT, INTENT, DCNet (the "iLearn" model family).
- **Protocol 3: `fashioniq_original_split`**
  - **Gallery Construction**: `build_original_gallery` (full split: 3,817 / 6,346 / 5,373 images).
  - **Reference Policy**: `reference_excluded = False`. Reference image remains in the gallery and can be retrieved, occupying rank 1 if scored highest.
  - **Models**: CLIP4Cir (fullft, noft), CSMCIR, PTHA_MTST, SPRC.

**Scientific Assessment**: The three protocols represent fundamentally distinct search spaces and retrieval tasks. Cross-protocol comparison is mathematically invalid. The repository's explicit typing in `FashionIQProtocol` (`slots=True`), registry separation, and storage guard (`storage.py:152` refusing cross-protocol cohort queries) correctly enforce protocol boundaries.

---

### 5.2 Query & Target Identity Mechanics
- Canonical query ID format: `{category}:{annotation_index}:{reference_id}:{target_id}` (`schemas/results.py:45-47`).
- Target rank constraints:
  - If `target_rank <= effective_depth`: `top_results[target_rank - 1].image_id == target_id` (`schemas/results.py:109-110`).
  - If `target_rank > effective_depth`: target ID must not appear in `top_results` (`schemas/results.py:111-112`).
  - `target_rank` must not exceed `gallery_size` (`schemas/results.py:107-108`).
- Result: Structural consistency checks in Pydantic schema-v2 guarantee query-target integrity.

---

### 5.3 Recall@10 and Recall@50 Computation
- Helper: `evaluate_fashioniq_recall` (`src/evaluation/fashioniq.py:80-119`).
- Hit definition: Query hit occurs if target index appears within top $k$ retrieved indices.
- Finite check: `_assert_comparable` verifies `torch.isfinite(scores).all()`.
- Reference exclusion implementation:
  - `recall_at_k_reference_excluded` (`src/evaluation/fashioniq.py:35-77`) removes the reference column explicitly (`ranking[ranking != reference_column]`).
  - Upstream evaluators (e.g. LIMN) mask reference score (`sims[i, target_index(source)] = -10e10`). Both operations are mathematically equivalent when valid scores exceed mask, but explicit column removal avoids numerical underflow/overflow artifacts.

---

### 5.4 Category & Macro Aggregation Arithmetic
- Aggregation rule across literature:
  $$\text{Macro } R@K = \frac{R@K_{\text{dress}} + R@K_{\text{shirt}} + R@K_{\text{toptee}}}{3}$$
  $$\text{Macro Mean} = \frac{\text{Macro } R@10 + \text{Macro } R@50}{2}$$
- Arithmetic verification:
  - Query counts per category are unequal (2,017 dress, 2,038 shirt, 1,961 toptee).
  - A query-weighted average would yield $\frac{2017 R_d + 2038 R_s + 1961 R_t}{6016}$.
  - The repository implementations in `macro_average_fashioniq` (`src/evaluation/fashioniq.py:238-255`), `macro_aggregate` (`workbench/replay/common.py:98-108`), `metrics_extraction.py:167-169`, and `verify_run.py:195-197` **strictly implement the unweighted arithmetic category mean**.
  - All verified reference rows in `papers.yaml` match this unweighted formula cell-for-cell.

---

### 5.5 Benchmark Reference Conditions & Training Noise Filtering
- `papers.yaml` contains 33 committed rows across 3 protocol IDs.
- Conditions tracked: `gallery_protocol`, `split`, `dataset`, `checkpoint_training_noise_pct`, `evaluation_noise_pct`.
- Selection logic (`benchmark_references.py:172-182`):
  Matches by condition rather than metric values.
  Noise filtering: if run specifies `checkpoint_training_noise_pct=50`, 0% or 20% rows are rejected.
  Defect noted in D-2: When multiple verified rows exist with `noise=null` (`ptha_mtst`), selection raises ambiguity and refuses to choose.

---

### 5.6 Result Identity & Immutability
- Report path: `{model_id}_{checkpoint_id}_{run_id}_aggregate.json`.
- Completion proof path: `{model_id}_{checkpoint_id}_{run_id}_completion.json`.
- Immutability guards:
  - `evaluate_models.py:1440-1441`: Refuses to launch if report or proof already exists on disk.
  - `evaluate_models.py:1668-1671`: Proof written via atomic exclusive open (`open("x")`), raising `FileExistsError` on collision.
  - `evaluate_models.py:1291-1292`: Evidence ledger raises `LedgerDuplicateRunError` if `run_id` already exists in ledger.
  - `storage.py:84-88`: Result loader rejects duplicate run IDs.

---

### 5.7 HMAC Evidence Ledger Chain & Proof Verification
- Location: `workbench/artifacts/evidence_ledger.jsonl`.
- Key management: `WORKBENCH_LEDGER_KEY` environment variable or 32-byte secret generated at 0600 permissions in `~/.config/cir/workbench_ledger_key` (container: `/artifacts/config/cir/workbench_ledger_key`).
- Cryptographic chaining:
  - Canonical JSON: `json.dumps(record, sort_keys=True, separators=(',', ':'))`.
  - Authentication: `hmac.new(key, canonical, hashlib.sha256).hexdigest()`.
  - Chain linkage: `record["previous_digest"] = previous_entry["entry_digest"]`.
  - Verified attacks: Probed tampering with `previous_digest`, mid-line truncation, content alteration, and key corruption. All 4 attacks were detected and returned `None` (refusal).

---

### 5.8 Verifier Completeness Contract
- The verifier (`workbench/scripts/verify_run.py`) provides rigorous regression protection for LIMN:
  - Verifies report file presence and identity match (`run_id`, `model_id`, `checkpoint_id`, `protocol_id`).
  - Verifies HMAC ledger entry presence, report SHA-256 digest match, `return_code == 0`, and `invocation_status == 'INVOCATION_COMPLETED'`.
  - Calls evaluator's `validate_run_manifest` with checksum, environment identity, and log directory verification.
  - Refuses synthetic declarations (`is_synthetic` and non-experiment `data_kind`).
  - Refuses incomplete extraction on LIMN (`metric_extraction_available != True` or `extraction_status != 'EXTRACTED_VERIFIED'`).
  - Enforces three complete FashionIQ categories with finite recalls in `[0, 100]`.
  - Recomputes unweighted category means and rejects macro deviations exceeding 0.05 pp.
- **Critical Limitation**: As identified in Finding D-1, this completeness contract is applied exclusively to `limn`. Non-LIMN models bypass extraction status verification, category completeness, and macro consistency checks.

---

## 6. Coverage Note

### What Was Directly Inspected & Executed:
1. All target Python source files: `workbench/backend/paper_gate.py`, `workbench/backend/benchmark_references.py`, `workbench/backend/checkpoint_selection.py`, `workbench/backend/schemas/results.py`, `workbench/scripts/verify_run.py`, `workbench/replay/common.py`.
2. Every committed benchmark row in `workbench/registry/paper_benchmarks/papers.yaml` (1837 lines, 33 rows).
3. Report, ledger, and completion proof writers in `workbench/scripts/evaluate_models.py`.
4. Protocol evaluation and dataset loader helpers in `src/evaluation/fashioniq.py` and `src/datasets/fashioniq.py`.
5. Exact disk counts of validation queries, pair-union gallery IDs, and split image IDs on the local filesystem (`data/FashionIQ`).
6. All 14 adversarial verification probes executed directly via Python runtime with exact outputs recorded.
7. Targeted test suites executed via `.venv/bin/python -m pytest`:
   - `workbench/tests/test_release_blocker_invocation_proof.py` (24 passed)
   - `workbench/tests/test_paper_gate.py`, `test_checkpoint_selection.py`, `test_protocols.py` (93 passed)
   - `workbench/tests/test_scientific_input_identity.py` (14 passed)

### What Was Not Inspected / Non-Goals:
1. Live GPU inference executions (RTX 5070 Ti Blackwell `sm_120` CUDA kernels): GPU inference was not run in this audit slice.
2. The LIMN per-query Top-K gap: explicitly excluded per user instructions as a known limitation.
3. Frontend React rendering lifecycle and Vite CSS styles (inspected only schema types and protocol constants).
4. Unrelated test files outside science, paper gate, benchmark references, and verification harness.
