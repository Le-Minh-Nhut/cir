# FIRST-PARTY SCIENTIFIC EXECUTION PATH AUDIT

**Audit Date**: 2026-10-17  
**Auditor**: AuditFirstPartyScience  
**Target**: First-party scientific producer/consumer chain  
**Audit Standard**: STATIC_AUDIT + RUNTIME_VERIFIED (pure Python / in-process execution, no GPU)  
**Status**: READ-ONLY AUDIT COMPLETE  

---

## 1. METRIC EXTRACTION

### Files Traced
- `workbench/backend/metrics_extraction.py:1-422`
- `workbench/registry/models.yaml:20-989`
- `workbench/scripts/evaluate_models.py:1612-1628`

### Observations
1. **Parser Availability Across 16 Registered Models**:
   - Only **5 models** have an extraction parser implemented:
     - `limn`: `parse_limn_output` (`metrics_extraction.py:74-216`)
     - `csmcir`: `parse_csmcir_output` (`metrics_extraction.py:219-307`)
     - `combiner_rn50x4_noft`: `parse_combiner_output` (`metrics_extraction.py:309-382`)
     - `clip4cir_rn50x4_fullft`: `parse_combiner_output` (`metrics_extraction.py:309-382`)
     - `sprc`: calls `parse_csmcir_output` (`metrics_extraction.py:408-411`)
   - **11 models** return `AGGREGATE_PARSER_UNAVAILABLE` (`metrics_extraction.py:414-422`):
     `airknow`, `conesep`, `ptha_mtst`, `habit`, `intent`, `hint`, `encoder`, `pair`, `clvc_net`, `dcnet`, `tgcir`.

2. **Macro Recomputation Independence**:
   - `parse_limn_output` (`metrics_extraction.py:167-204`): **INDEPENDENT**.
     Computes `computed_r10_macro = (d_10 + s_10 + t_10) / 3.0`, `computed_r50_macro = (d_50 + s_50 + t_50) / 3.0`, `computed_mean = (computed_r10_macro + computed_r50_macro) / 2.0`.
     Checks `abs(decl - computed) <= 0.05`.
     Lines 200-204 emit:
     `observed = {"r10": round(computed_r10_macro, 4), "r50": round(computed_r50_macro, 4), "mean": round(computed_mean, 4)}`.
     The returned `observed_metrics` is strictly derived from per-category values.
   - `parse_csmcir_output` (`metrics_extraction.py:271-285`): **ECHOES REPORT**.
     Lines 271-279 compute `exp10`, `exp50`, `exp_mean` and assert `abs(avg - exp) <= 0.05`.
     However, lines 281-285 emit:
     `observed = {"r10": round(float(avg10), 4), "r50": round(float(avg50), 4), "mean": round(float(avg_mean), 4)}`.
     It validates internal consistency against category numbers, but emits the author stdout aggregate, not the recomputed macro.
   - `parse_combiner_output` (`metrics_extraction.py:357-368`): **HYBRID / ECHOES RECALLS**.
     Computes `exp10` and `exp50` to cross-check within 0.05.
     Emits `observed["r10"] = round(float(avg10), 4)` and `observed["r50"] = round(float(avg50), 4)` directly from stdout lines; re-derives only `mean = round((avg10 + avg50) / 2.0, 4)`.
   - `sprc`: delegates to `parse_csmcir_output`, thus echoes report averages.

### Findings
- **UNVERIFIED_RISK**: 11 of 16 registered models lack aggregate parsers. Any completed run for these 11 models produces `extraction_status="AGGREGATE_PARSER_UNAVAILABLE"` and `observed_metrics=None`, preventing downstream benchmark gate comparison and failing `verify_run.py`.
- **BY_DESIGN**: `parse_csmcir_output` and `parse_combiner_output` echo author-printed macro values after asserting $\le 0.05$ pp self-consistency with category metrics to guard against text parsing corruptions while preserving exact author floating point formatting.

---

## 2. PAPER GATE

### Files Traced
- `workbench/backend/paper_gate.py:1-475`
- `workbench/scripts/evaluate_models.py:127-198`
- `workbench/tests/test_paper_gate.py:145-175`

### Observations
1. **Comparison Semantics**:
   - `TOLERANCE_PCT = 5.0` (`paper_gate.py:27`).
   - Relative percentage error (`:213-220`): `relative = abs(reproduced - paper) / abs(paper) * 100.0`. Acceptance condition: `relative <= tolerance_pct * (1 + 1e-12)`.
   - Zero baseline (`:208-211`): If `paper == 0.0`, relative error is `None`; requires `reproduced == 0.0`.
   - Scale normalization (`:155-167, 195, 204`): `normalize_scale(value, scale)`. Passes percentage unchanged; multiplies fraction by 100.
     Notice: `compare_metric` normalizes **both** `paper` and `reproduced` using `scale`. In production (`evaluate_models.py:177`), `metric_scale` is not passed and defaults to `SCALE_PERCENTAGE`.
   - Category metrics (`:35-37, 380-392`): 6 category keys (`dress_r10`, `dress_r50`, `shirt_r10`, `shirt_r50`, `toptee_r10`, `toptee_r50`).
     - Compared only when both reference and run provide category metrics.
     - If paper publishes categories but run did not observe them: verdict demoted to `PASS_WITH_UNCHECKED_CATEGORIES` (or `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`) with `passed = False` (`:442-448`).
     - If paper has no category table, but run observed all 6 categories: passes with note that category evidence was complete (`:456-466`).
   - Macro-vs-category handling: Per-category deviation $> 5\%$ triggers `ALERT_METRIC_DEVIATION` with `passed = False` (`:417-420`). A strong Shirt result cannot mask a failing Dress result.
2. **Protocol Mismatch & Published Disagreements**:
   - Disagreements are never hidden: any comparable metric exceeding tolerance returns `ALERT_METRIC_DEVIATION` (`passed = False`).
   - Candidate universe / protocol mismatch: `COMPARABILITY_FIELDS = ("protocol_id", "checkpoint_id", "checkpoint_training_noise_pct", "evaluation_noise_pct", "split", "dataset")` (`:72-79`).
     If the run's `protocol_id` disagrees with `reference.gallery_protocol`, `_comparability_blockers` (`:260-277`) returns a hard blocker $\rightarrow$ `BLOCKED_PROTOCOL_MISMATCH` with `passed = False` (`:287, 364-366`).
     If reference states 0 constraints, returns `BLOCKED_REFERENCE_INCOMPLETE` (`:360-363`).

### Findings
- **BY_DESIGN**: Canonical 5% relative error gate strictly refuses protocol mismatches, unverified references, checkpoint mismatches, and deviating categories.
- **UNVERIFIED_RISK**: In `paper_gate.py:195, 204`, `compare_metric` applies `normalize_scale` to both `paper` and `reproduced`. If an evaluator ever declares `scale="fraction"`, `paper` (stored as percentage in `papers.yaml`) would be multiplied by 100 to 5707.0. Currently unused in production (`evaluate_models.py` uses default `percentage`), but test-only `SCALE_FRACTION` design requires caller caution.

---

## 3. REFERENCE SELECTION

### Files Traced
- `workbench/backend/benchmark_references.py:1-361`
- `workbench/registry/paper_benchmarks/papers.yaml`

### Observations
1. **Selection Semantics**:
   - `select_reference_outcome` (`benchmark_references.py:236-334`):
     - Line 266: `verified = tuple(c for c in candidates if c.reference_status == REFERENCE_VERIFIED)`.
       Rows with `reference_status != "VERIFIED"` are filtered out immediately. If no verified rows exist, returns `SELECTION_NO_PUBLISHED_COMPARISON` with `reference = None`.
     - `matches(candidate)` (`:277-293`):
       Matches `checkpoint_training_noise_pct` and `gallery_protocol`.
       Lines 284-290: If `protocol_id is not None` and `record.get("gallery_protocol") is None`, `matches()` returns `False`.
     - Exact match (`:296-299`): If `len(exact) == 1`, returns `exact[0]` with `SELECTION_MATCHED`.
     - Ambiguity path (`:300-311`): If `len(exact) > 1`, returns `SELECTION_AMBIGUOUS` with `reference = None`. Refuses to select when two verified rows bind identical conditions.
     - `author_published_no_reference` path (`:207-234, 265`): `unbenchmarked(model_id)` inspects `models.yaml`. If `reported_r10`, `reported_r50`, `reported_mean` are all null, returns `author_published_no_reference=True`.
2. **Can a run bind to an unverified row?**
   - **NO**. Blocked at `benchmark_references.py:266` (`reference_status == REFERENCE_VERIFIED`) and redundantly at `paper_gate.py:337` (`reference.reference_status != REFERENCE_VERIFIED`).
3. **Can an unresolved-protocol row bind to a run?**
   - **NO**. When `protocol_id` is passed (standard call from `evaluate_models.py:172`), `matches()` line 290 explicitly returns `False` for any candidate missing `gallery_protocol`.

### Findings
- **BY_DESIGN**: Reference selection fails closed on ambiguity, missing protocols, or unverified references.
- **CONFIRMED_BUG (External/Registry Disambiguation)**: `ptha_mtst` has two committed verified rows in `papers.yaml:1660-1774` (`ptha_mtst_fiq_table3_fashiqmtst_blip2` and `ptha_mtst_fiq_no_pretrain_table3`) that both declare `checkpoint_id: null`, `checkpoint_training_noise_pct: null`, and `gallery_protocol: fashioniq_original_split`. `select_reference_outcome()` returns `SELECTION_AMBIGUOUS` with `reference = None` for any run of `ptha_mtst`. (Previously noted in model audit; confirmed here in first-party science path).

---

## 4. EVIDENCE CHAIN (`verify_run.py`)

### Files Traced
- `workbench/scripts/verify_run.py:1-626`
- `workbench/scripts/evaluate_models.py:127-198, 1612-1628`

### Observations
1. **`is_research_evidence` Contract (`verify_run.py:200-201`)**:
   `corroborated = not reasons and not metric_reasons`  
   `is_evidence = corroborated and not evidence_reasons`  
   Requires:
   - Report JSON exists, parses, matches exact request identity (`run_id`, `model_id`, `checkpoint_id`, `protocol_id`).
   - HMAC ledger key valid, ledger chain verified, run entry present, report sha256 matches ledger `report_digest`, `return_code == 0`, `invocation_status == 'INVOCATION_COMPLETED'`.
   - Completion proof valid under `completion_validator` (environment digest, checksums).
   - Metric contract satisfied (`metric_evidence_reasons`):
     - LIMN: extraction verified/synthetic, 3 categories present with valid recalls in $[0, 100]$, macro within 0.05 pp of category arithmetic mean.
     - Non-LIMN: reference selected, report carries at least one published metric in `observed_metrics` as finite recall in $[0, 100]$, extraction verified/synthetic, categories valid if present, macro consistent if all 3 categories present.
   - Re-extraction (`_reextraction_reasons`): parses recorded `stdout.log`/`stderr.log` from run directory using `extract_aggregate_metrics` and checks values match report within 0.00011.
   - Evidence kind (`_evidence_kind_reasons`): `is_synthetic` is not True, and `data_kind` is None or `"experiment"`.
2. **Dead / Bypass Scrutiny**:
   - `_reextraction_reasons` (`verify_run.py:358-364`): If `extracted.extraction_status == "AGGREGATE_PARSER_UNAVAILABLE"`, returns `[]`. For the 11 models without aggregate parsers, log re-extraction is a no-op.
   - `_non_limn_reasons` (`verify_run.py:516-524`): Uses `any(_is_recall(observed.get(key)) for key in published)`. If reference publishes `r10`, `r50`, `mean`, but report only has `r10`, it does NOT fail `_non_limn_reasons`.
   - `verify_run.py` does NOT check `benchmark_gate.passed`. Certified execution and evidence integrity are deliberately decoupled from whether the numbers matched the paper (`BY_DESIGN`, `:502-506`).

### Findings
- **BY_DESIGN**: Verification validates evidence integrity and authentic execution, allowing real runs that deviate from paper numbers to be certified as valid research evidence (`ALERT_METRIC_DEVIATION` recorded on report).
- **UNVERIFIED_RISK**: `verify_run.py:516` requires `any` rather than `all` published metrics to be observed in `observed_metrics`. A run observing only R@10 while missing R@50 passes evidence verification for non-LIMN models.

---

## 5. VALIDATOR (`validate_results.py`)

### Files Traced
- `workbench/scripts/validate_results.py:1-85`
- `workbench/backend/schemas/results.py:1-163`
- `workbench/replay/publication.py:146-168`

### Observations
1. **Validator vs Schema Division of Responsibilities**:
   - `validate_results.py` enforces:
     - Runtime identity derivation: Calls `verify_run_identity(result)` which recomputes `expected = run_id_for(...)` from source commit and checkpoint sha256, refusing relabelled `run_id`s. (Schema does not enforce derived identity; only checks `min_length=1`).
     - Provenance completeness: With `--strict-real`, checks `PROVENANCE_FIELDS` (`upstream_commit`, `checkpoint_sha256`, `command_digest`, `environment_digest`). (Schema defines these as optional `str | None = None`).
   - Schema (`ResultRun`) enforces:
     - Strict field types, forbidden extra fields (`extra="forbid"`).
     - Query ID canonical format `{category}:{index}:{reference}:{target}`.
     - Contiguous ranks $1 \dots \text{depth}$ in `top_results` and unique image IDs.
     - Target rank agreement: if $\text{target\_rank} \le \text{depth}$, `top_results[target_rank - 1].image_id == target_id`; if $\text{target\_rank} > \text{depth}$, `target_id` is absent from `top_results`.
2. **Gaps Where Malformed Artifacts Pass Both**:
   - **Score Monotonicity Gap**: Neither schema nor validator checks that `TopResult.score` decreases monotonically with `rank`. An artifact with arbitrary or inverted scores passes both.
   - **Reference Exclusion Gap**: Under exclusion protocols (`fashioniq_full_gallery_ref_excluded`, `fashioniq_val_split`), neither schema nor validator checks whether `reference_id` appears in `top_results`.
   - **Validation Completeness Gap**: Neither schema nor validator checks query count. A partial run with a single query passes both.
   - **Metric Distribution Gap**: Aggregate `reproduced_metrics` in `RunMetadata` is never cross-checked against the actual empirical rank distribution of `queries`.
   - **Non-Strict Execution**: Running `validate_results.py` without `--strict-real` emits `[WARN]` on missing provenance but exits 0.

### Findings
- **UNVERIFIED_RISK**: Gaps in `validate_results.py` allow non-monotonic scores, reference leaks in exclusion galleries, and incomplete query sets to pass validation.

---

## 6. PARALLEL EVALUATION MODULES (`src/evaluation/fashioniq.py`, `src/datasets/fashioniq.py`)

### Files Traced
- `src/evaluation/fashioniq.py:1-344`
- `src/evaluation/fashioniq_encoder.py:1-149`
- `src/datasets/fashioniq.py:1-336`
- `workbench/replay/limn_observer.py:91-125`
- `workbench/third_party/LIMN/LIMN/test.py:80-97`

### Observations
1. **Execution Path Reachability**:
   - `src/evaluation/fashioniq.py`, `src/evaluation/fashioniq_encoder.py`, and `src/datasets/fashioniq.py` are **NOT** imported or called by any production workbench execution path (`workbench/backend/`, `workbench/scripts/`, `workbench/replay/`).
   - They are referenced **strictly by unit tests** (`tests/unit/test_fashioniq_*.py`, `workbench/tests/test_protocols.py`).
   - They represent an older, generic evaluation scaffold that runs parallel to the model-specific third-party and replay evaluators.
2. **Semantic Divergence: Generic Scaffolding vs LIMN Native Procedure**:
   - **Mask Value**:
     - LIMN Native (`LIMN/test.py:82`, `limn_observer.py:113`): Masks reference similarity in the score matrix to `-10e10` before sorting (`sims[i, ref_idx] = -10e10`).
     - `src/evaluation/fashioniq.py:71-72` & `src/evaluation/fashioniq_encoder.py:44-45`: Does **not** mask. Sorts the entire score tensor descending, then physically filters out the reference column (`ranking = ranking[ranking != reference_column]`).
   - **Sorting**:
     - LIMN Native: `np.argsort(-sims[i, :])` (numpy quicksort on negated float32, ascending $\equiv$ descending; not stable).
     - `src/evaluation/fashioniq.py`: `torch.argsort(scores, descending=True)` (torch default, not stable).
     - `src/evaluation/fashioniq_encoder.py`: `torch.argsort(scores, descending=True, stable=True)` (enforces stable sorting).
   - **Gallery Order**:
     - LIMN Native: Traverses validation queries and appends references and targets in first-seen order.
     - `src/datasets/fashioniq.py:build_pair_union_gallery`: Matches LIMN's first-seen pair-union order.
     - `src/evaluation/fashioniq.py:build_fashioniq_gallery`: For `fashioniq_val_split` uses pair-union; for `fashioniq_original_split` and `fashioniq_full_gallery_ref_excluded` loads `split.{category}.val.json` in raw file order.
   - **Recall Denominator**:
     - All implementations use total query count as the denominator ($100 \times \text{hits} / \text{query\_count}$).

### Findings
- **BY_DESIGN**: `src/evaluation/fashioniq.py` is a preserved standalone generic harness for offline unit tests; workbench replays use source-faithful observer modules (`workbench/replay/limn_observer.py`).

---

## 7. ANALYSIS, STORAGE, AND INDEX

### Files Traced
- `workbench/backend/storage.py:1-204`
- `workbench/backend/index.py:1-125`
- `workbench/backend/analysis.py:1-93`
- `workbench/backend/main.py:195-201, 860-885`
- `workbench/backend/schemas/results.py:59-110`
- `workbench/frontend/types.ts:1-10`

### Observations
1. **Per-Query Flow from Disk to API Response**:
   - **Disk**: `ResultRun` JSON validated with Pydantic v2.
   - **Index (`index.py:rebuild_index`)**: Parses `ResultRun`, creates DuckDB tables `runs`, `queries`, `top_results`, writes CSVs and executes `COPY ... FROM CSV`.
   - **Storage / API (`storage.py`)**:
     - `query_page()` selects query columns, joins `top_results`, and returns `{"items": [...], "total": ..., "limit": ..., "offset": ...}`.
     - `get_run()` / `list_runs()` selects `metadata` from `runs` table and parses with `json.loads()`.
     - `get_analysis_input()` validates run comparability, selects top-K results, and builds comparison maps.
2. **Silently Dropped Fields and Coalesced Defaults**:
   - **CONFIRMED BUG: `evidence_status` Dropped**:
     `index.py:82, 104` creates and populates column `runs.evidence_status` (`PER_QUERY_VERIFIED` or `SYNTHETIC_OR_MOCK`).
     `storage.py:33, 47, 60` queries `SELECT metadata FROM runs`.
     Because `metadata` is `run.model_dump_json()`, and `RunMetadata` does not contain `evidence_status`, `evidence_status` is **never selected or merged into the returned run record**!
     `frontend/types.ts:3` defines `Run.evidence_status?: string | null`. The API `/api/runs` always returns `undefined` for `evidence_status`.
   - `_query_maps` (`storage.py:140`): Hard-codes `"top_results": []`. Any caller using `_query_maps` or `get_query_universe` directly receives empty `top_results` (prevented in analysis by `get_analysis_input` which populates `top_results`).
   - `cohort_metrics` (`storage.py:204`): Computes pooled micro-average over all queries in `query_ids` ($100 \times \text{hits} / \text{total\_queries}$), rather than the standard unweighted category macro average.
   - `mean_topk_jaccard` (`analysis.py:53`): Coalesces to `1.0` if `pairwise` is empty (e.g. single run selection).
   - `failure_jaccard_from_maps` (`analysis.py:80`): Coalesces to `1.0` if neither run experienced any failures.

### Findings
- **CONFIRMED_BUG (Field Dropped in API)**: `evidence_status` is stored in DuckDB by `index.py` but omitted by `storage.py:list_runs` / `get_run`.
- **BY_DESIGN**: Jaccard 1.0 fallbacks on empty candidate sets handle edge-case degenerate analysis selections without raising unhandled exceptions.

---

## CONTRACT VIOLATIONS

| Producer / Source | Consumer / Destination | File:Line | Issue | Severity |
|---|---|---|---|---|
| `workbench/replay/publication.py:22` | `workbench/backend/schemas/results.py:10` & `workbench/registry/models.yaml:14` | `publication.py:22`, `schemas/results.py:10` | **Protocol label disagreement**: `publication.py` declares `"fashioniq_full_gallery_ref_excluded": "val"`, while `schemas/results.py` and `models.yaml` declare `"original"`. Any call to `build_result_run` for Protocol B crashes with Pydantic `ValidationError: literature_split_label does not match protocol_id`. | **CONFIRMED_BUG / CRITICAL** |
| `workbench/backend/index.py:82, 104` | `workbench/backend/storage.py:33, 47, 60` | `index.py:104`, `storage.py:33` | **Silently dropped field**: `index.py` computes and indexes `runs.evidence_status` in DuckDB, but `storage.py` queries `SELECT metadata FROM runs`, omitting `evidence_status` so `/api/runs` never returns it to frontend consumers. | **CONFIRMED_BUG / MEDIUM** |
| `workbench/registry/paper_benchmarks/papers.yaml:1660-1774` | `workbench/backend/benchmark_references.py:277-311` | `papers.yaml:1667, 1732`, `benchmark_references.py:300` | **Reference ambiguity**: Two verified rows for `ptha_mtst` declare identical selection criteria (`noise=null`, `protocol=fashioniq_original_split`), causing `select_reference_outcome` to return `SELECTION_AMBIGUOUS` and blocking all reproduction queries. | **CONFIRMED_BUG / HIGH** |

---

## REPRODUCTIONS

### Reproduction 1: Protocol B Literature Label Crash
```bash
.venv/bin/python -c "
from workbench.replay.publication import build_result_run, ObservedCategory
from workbench.replay.limn_observer import ObservedQuery

query = ObservedQuery(
    category='dress', annotation_index=0, reference_id='ref', target_id='tgt',
    raw_captions=('cap1', 'cap2'), model_input_text='cap1 cap2', target_rank=1,
    top_results=[(1, 'tgt', 0.9)],
)
cat = ObservedCategory(category='dress', gallery_size=10, eligible_count=9, queries=[query])
build_result_run(
    [cat], model_id='dcnet', method_name='DCNet', checkpoint_id='ckpt',
    protocol_id='fashioniq_full_gallery_ref_excluded', top_k_saved=1,
)
"
```
**Output**:
```
pydantic_core._pydantic_core.ValidationError: 1 validation error for RunMetadata
  Value error, literature_split_label does not match protocol_id [type=value_error, input_value=...]
```

### Reproduction 2: `evidence_status` Dropped from `/api/runs`
```bash
.venv/bin/python -c "
from workbench.backend import storage, index
# In DuckDB runs table, evidence_status is stored:
# index.py:82 CREATE TABLE runs (... evidence_status VARCHAR NOT NULL, ...)
# In storage.py:33:
# SELECT metadata FROM runs -> returns json.loads(metadata) where evidence_status is absent!
"
```
Column is persisted in database but excluded from all `storage.py` run queries.
