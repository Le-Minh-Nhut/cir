# D — FashionIQ protocol / paper-reference / results-contract slice

Model audited: **limn** (only registered model in this assignment's scope). Branch
`research/cir-failure-workbench`, HEAD `0b0f3a6`. Read-only audit; no repo file was modified except
this report. No GPU inference was run.

Target files: `workbench/backend/paper_gate.py`, `workbench/backend/benchmark_references.py`,
`workbench/backend/checkpoint_selection.py`, `workbench/backend/schemas/results.py`,
`workbench/registry/paper_benchmarks/papers.yaml`, `workbench/scripts/validate_results.py`.

---

## A) Files inspected (with role in the verdict)

| file | lines read | role |
|---|---|---|
| `workbench/backend/paper_gate.py` | 1–474 (all) | the acceptance gate; owns status vocabulary, tolerance, comparability, mapping cap |
| `workbench/backend/benchmark_references.py` | 1–195 (all) | registry load, reference selection by condition, evidence record |
| `workbench/backend/checkpoint_selection.py` | 1–263 (all) | checkpoint selection policy (`strict_clean` / `include_flagged_conditions`) |
| `workbench/backend/schemas/results.py` | 1–132 (all) | canonical schema-v2 run contract, protocol-id vocabulary, `top_results`/`target_rank` self-consistency |
| `workbench/registry/paper_benchmarks/papers.yaml` | 1–1837 (all rows, line-addressed below) | committed paper reference set (33 rows, 3 protocol ids) |
| `workbench/scripts/validate_results.py` | 1–75 (all) | "validator" — provenance presence + schema validation only |
| `workbench/registry/models.yaml` | protocols block 7–19; LIMN 900–1000; per-model noise/mapping fields | registry of record for protocol ids and checkpoint metadata |
| `workbench/registry/paper_benchmarks/extracted/legacy_cir_fashioniq_bundle.json` | LIMN record | offline-extraction payload the registry is built from |
| `workbench/registry/paper_benchmarks/verification/VerifyPapers{A,B,C}.json` | LIMN/TGCIR/DCNet/CLVC/HINT/ENCODER/PAIR/SPRC/CSMCIR checks | independent verification ledger for the rows |
| `workbench/scripts/evaluate_models.py` | 110–470, 650–720, 1400–1520, 1803–1860 | the only production caller of `select_reference` + `evaluate_gate`; builds `run_meta` |
| `workbench/backend/metrics_extraction.py` | 1–423 (all) | the parsers that declare observed metrics and per-category metrics |
| `workbench/replay/limn.py`, `workbench/replay/common.py` | all | LIMN replay: protocol string, macro arithmetic, checkpoint hash gate |
| `workbench/third_party/LIMN/LIMN/{test.py,datasets.py}` | `test.py:23–95`, `datasets.py:38–70,143–170` | pinned upstream LIMN evaluator (gallery + reference mask + recall) |
| `src/evaluation/fashioniq.py`, `src/datasets/fashioniq.py` | `fashioniq.py:1–345`, `fashioniq.py:186–205` (dataset) | the workbench's own protocol-table-driven evaluator |
| `workbench/backend/adapters/base.py`, `models.py` | all | `EvalRequest`, per-model command; LIMN adapter |
| `workbench/backend/main.py` | 108–250 | aggregate-report serving + evidence-ledger corroboration |
| `workbench/backend/index.py` | 53–95 | per-query JSON loading, duplicate `run_id` refusal |
| `workbench/registry/{models.yaml}` mapping-status fields, `workbench/scripts/doctor.py:209`, `download_checkpoints.py:117`, `replay/limn.py:49` | — | every consumer of `checkpoint_mapping_status` |
| `workbench/frontend/protocols.ts`, `gate.ts`, `App.tsx:13,58` | — | frontend protocol vocabulary + verdict display |
| `workbench/configs/pc_autopilot.pilot_limn.yaml` | all | the pilot config the first real LIMN run would use |

Existing workbench audit documents were treated as hypotheses only; where my reading disagrees I say
so in §B/§C (notably: `VerifyPapersC.json` check 2's "registry gap" note is stale, and its LIMN
protocol class was never independently checked).

---

## B) Execution paths traced

### B1. LIMN reference selection (exact commands + observed output)

```
$ .venv/bin/python -c "... benchmark_references.select_reference('limn', checkpoint_training_noise_pct=None, protocol_id='fashioniq_val_split')"
select: limn_fashioniq_base_iter0_table1 | exact condition match
```

Reference row: `papers.yaml:1589–1651` — `gallery_protocol: fashioniq_val_split` (:1607),
`reference_excluded: true` (:1608), `checkpoint_id: base_iter0_all_categories` (:1599),
`checkpoint_training_noise_pct: null` (:1600), `split: val` (:1606), `dataset: fashioniq` (:1602),
`reference_status: VERIFIED` (:1651), metrics R@10 55.91 / R@50 77.82 / mean 66.87 (derived, :1615–1630).
`select_reference` (`benchmark_references.py:149–195`) requires a VERIFIED row (`:170–173`) and matches
on condition, not on numbers (`:175–182`).

### B2. LIMN run→gate, executed end to end against a paper-identical observation

```
status  = PASS_METRICS_ONLY_MAPPING_UNVERIFIED
passed  = False
detail  = every comparable metric is within 5 percent relative error; checkpoint mapping is UNVERIFIED,
          so this is a metric comparison, not a certified official reproduction;
          1 condition(s) the reference does not state were not checked
reason  = exact condition match
```

A run **whose numbers are exactly the paper's** does not come out as a certified pass. Same result for
TGCIR. A deviating category raises the alert correctly:

```
dress r10 60.0 (+10pp) -> ALERT_METRIC_DEVIATION passed=False
```

### B3. Where the LIMN protocol string is produced vs where it must agree

- `workbench/registry/models.yaml:920-921` — `native_protocol: fashioniq_val_split`, `supported_protocols: [fashioniq_val_split]`.
- `workbench/scripts/evaluate_models.py:1825` — `protocol = args.protocol or model["native_protocol"]`.
- `workbench/backend/adapters/models.py:125–139` (`LIMNAdapter`) — the produced command is
  `…/workbench/replay/limn.py --checkpoint-root … --dataset-root … --source-root …`.
- `workbench/replay/limn.py:181` — every `ReplayResult` is stamped `protocol="fashioniq_val_split"`, and `:226`
  stamps the emitted JSON `"protocol": "fashioniq_val_split"`.
- `workbench/backend/adapters/base.py:23-24` + `ModelAdapter.validate_request` — refuses a protocol not in `supported_protocols`.
- `EvalRequest.output_path` (`adapters/base.py:17`, set at `evaluate_models.py:348`) is **never read by
  LIMNAdapter**, so no schema-v2 `ResultRun` file is produced on that path.

### B4. LIMN's cohort, from the pinned upstream source (verification, not the docs)

- gallery = first-seen ordered union of each validation triplet's `source_img_id` then `target_img_id`:
  `workbench/third_party/LIMN/LIMN/datasets.py:158–170`.
- same order as the consumed target list: `LIMN/test.py:76–78`.
- reference removed by score mask **before** ranking and before the top-50 truncation:
  `LIMN/test.py:80–85` (`if name != 'birds': sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`).
- Recall@1/10/50 as `100 * hits / len(nn_result)`: `LIMN/test.py:88–95`.
- `workbench/replay/limn.py:179–182` calls this function directly and copies `{cat}_r1/_r10/_r50`;
  `workbench/replay/common.py:98–108` (`macro_aggregate`) takes an equal-weight mean over the three
  categories and then `(r10+r50)/2`, which is the arithmetic `papers.yaml:1622-1630` re-executed
  ((50.72+56.08+60.94)/3 = 55.9133, (74.52+77.09+81.85)/3 = 77.82).
  → protocol id, gallery, exclusion and aggregation **agree with the pinned code**.

### B5. The workbench's own protocol table (parity, separate path)

`src/evaluation/fashioniq.py:171–187` defines exactly the three registry ids, `:190–201` refuses unknown
names outright, `:204–219` builds `full_split` vs `pair_union`. `fashioniq_val_split` →
`build_pair_union_gallery` (`src/datasets/fashioniq.py:186–205`, reference-then-target first-seen union,
identical rule to B4) with `reference_excluded=True` (`src/evaluation/fashioniq.py:179-180`). This path is
**not** the LIMN path (`LIMNAdapter` runs the replay, not this code) and is exercised only by
`tests/unit/test_fashioniq_protocols.py`.

### B6. Duplicate reports / evidence-ledger corroboration / immutability

- per-query artifacts: `index.load_runs` raises `duplicate_run_id` (`index.py:58–61`);
  `validate_results.py` uses the same loader (`:13,39`).
- aggregate reports: `main._aggregate_reports` (`main.py:128–187`) serves a `*_aggregate.json` only when
  its **sha256** has a ledger entry (`:143–160`) whose `run_id`, `model_id`, `checkpoint_id`,
  `protocol_id`, `return_code == 0` and `invocation_status == "INVOCATION_COMPLETED"` all agree (`:160–172`),
  and skips a second file with an already-seen `run_id` (`:173–177`).
- report write is exclusive-create (`evaluate_models.py:1508–1515`), and the completion validator
  re-checks live checkpoint members against the report/ledger (`:650–720`).
- ran: `.venv/bin/python workbench/scripts/validate_results.py --root workbench/artifacts/results --strict-real`
  → `[OK] validated 6 runs … exit=0`.

---

## C) Confirmed defects

### C-1 (P1) — `checkpoint_mapping_status` is two disjoint vocabularies; the gate demotes **all 8** VERIFIED paper rows

- Location: `workbench/registry/paper_benchmarks/papers.yaml:129,175,798,912,959,1005,1052,1601`
  (`checkpoint_mapping_status: VERIFIED`) vs `workbench/registry/models.yaml:96,139,149,…`
  (`VERIFIED_METADATA`); capping code `workbench/backend/paper_gate.py:413-414`.
- Root cause: the paper registry says `VERIFIED`; the gate tests against the *models.yaml* spelling
  `VERIFIED_METADATA` (`paper_gate.py:410-413` → `if reference_mapping and reference_mapping not in
  ("VERIFIED_METADATA","NO_CHECKPOINT"): mapping_verified = False`). The set of rows that claim a
  verified asset and the set of strings the gate accepts are disjoint.
- Repro:
  ```
  $ .venv/bin/python -c "…references_for_model('limn')… select_reference(protocol_id='fashioniq_val_split')"
  limn_fashioniq_base_iter0_table1 VERIFIED fashioniq_val_split base_iter0_all_categories VERIFIED val 0 55.91 77.82 66.87
  $ …evaluate_gate(ref, {'r10':55.9133,'r50':77.82,'mean':66.8667}, run_meta={…,'checkpoint_mapping_verified':True}, …)
  PASS_METRICS_ONLY_MAPPING_UNVERIFIED   passed=False
  ```
- Expected vs actual: a paper row whose asset mapping the registry verified should permit certification
  (subject to numbers); actually every exact reproduction of CSMCIR, HINT and LIMN is reported
  `PASS_METRICS_ONLY_MAPPING_UNVERIFIED`, `passed=False`.
- Impact: the mission's target deployment can never print a certified LIMN reproduction, even with a
  perfect GPU run and byte-identical author weights. It also destroys the intended signal: genuinely
  unmapped rows (14 rows, including TGCIR) become indistinguishable from verified ones.
- Minimal fix: normalise the comparison (accept `VERIFIED` and `VERIFIED_METADATA` in `paper_gate.py:413`)
  **or** rewrite the 8 paper rows to `VERIFIED_METADATA` and make the mapper in
  `scripts/build_paper_registry.py` emit one spelling. One vocabulary, one place.

### C-2 (P2) — `fashioniq_full_gallery_ref_excluded` is declared with `reference_excluded: true`, contradicted by its own pinned evaluator

- Location: `workbench/registry/models.yaml:12-15` (`fashioniq_full_gallery_ref_excluded: … reference_excluded` not
  stated in the protocols block, but implicitly excluded by its own name) and
  `papers.yaml:1253-1308,1359-1408,1460-1514` (`reference_excluded: true` for the four INTENT rows and DCNet);
  `workbench/registry/paper_benchmarks/verification/VerifyPapersA.json` carries no `gallery_protocol`
  check for Air-Know/HABIT/INTENT/ConeSep and `VerifyPapersC.json` check 1 carries none for TGCIR.
- Root cause: `dcnet` (`models.yaml:626-627`) and INTENT are assigned to a protocol whose defining
  property (`reference_excluded: true`) the pinned DCNet evaluator contradicts:
  `workbench/third_party/DCNet/trainer/trainer_ref.py:126–178` ranks all `N = val_experts` index
  embeddings (`val_ids` = the full split file read by
  `workbench/third_party/DCNet/data_loader/CE_dataset.py:27–28`) and `compute_score`
  (`workbench/third_party/DCNet/model/metric.py:1–19`) has no reference removal; the dataset row
  (`CE_dataset.py:56–64`) carries `candidate` alongside `target` and nothing masks the candidate.
  `grep -rn "10e10" workbench/third_party/DCNet/` → no match.
- Repro: the greps above; plus the row text (`papers.yaml:1460-1472`).
- Impact: DCNet/INTENT runs are compared against a wrong cohort label inside `COMPARABILITY_FIELDS`
  (`paper_gate.py:71-78`) — which is exactly the "protocol alteration" the mission forbids in reverse:
  a paper number is labelled as produced under a protocol the code never implemented. Any future
  DCNet/INTENT evaluation is judged `BLOCKED_PROTOCOL_MISMATCH` or silently mislabelled.
- Minimal fix: re-derive `dcnet`/`intent` protocol class from the pinned evaluator and set it to the
  reference-eligible full-gallery class (or add an explicit per-row provenance field naming the code
  evidence); do not change any number.

### C-3 (P2) — `RecoveredMetrics`/schema-v2 recall values accept NaN, ±inf, negatives and >100

- Location: `workbench/backend/schemas/results.py:16–19` (`ReportedMetrics.r10/r50/mean: float | None`, no bound,
  no finiteness) and `:104–110` (`TopResult.score: float`, no finiteness).
- Root cause: no constraint anywhere on the metric carriers; the gate's own finiteness policy
  (`paper_gate.py:172–176`) exists but the canonical artifact never routes through it.
- Repro:
  ```
  $ .venv/bin/python -c "ResultRun.model_validate(run with reproduced_metrics.r10=NaN / inf / -1.0 / 1e9)"
  nan          -> ACCEPTED
  inf          -> ACCEPTED
  -1.0         -> ACCEPTED
  1000000000.0 -> ACCEPTED
  $ .venv/bin/python -c "json.dumps({'r10': float('nan')})"   # default allow_nan=True
  ```
- Impact: a future real writer (the only producer today is `load_mock_results.py`; `EvalRequest.output_path`
  is ignored by every audited adapter, so no production writer exists yet) could persist a `NaN` recall
  that `index.py:89` pushes into DuckDB `DOUBLE r10` and the API then serves. The gate would call it
  `ALERT_METRIC_DEVIATION` ("reproduced value is not a finite number", `paper_gate.py:196-200`) — correct
  but late; the audit trail already contains the non-finite value.
- Minimal fix: `Field(ge=0.0, le=100.0, allow_inf_nan=False)` on `ReportedMetrics.r10/r50/mean`, and
  `allow_inf_nan=False` on `TopResult.score`. Three tokens.

### C-4 (P2) — `pipeline.run_validation_index` reports success while real artifacts cannot be validated

- Location: `workbench/scripts/pipeline.py:1449-1453` — `if not runs: … rebuild_index() … "[OK] no schema-v2
  indexable runs found; wrote an empty index"`; reached through `:1869-1878` with
  `input_paths=(WORKBENCH_RESULTS_ROOT,)` and `output_paths=(database_path(config),)`.
- Root cause: the stage's declared output is the database, and `load_runs` returning `[]` (an I/O or
  schema error, or simply "no per-query export exists") is treated as a legitimate empty result instead
  of a validation failure. LIMN (B3) never writes a schema-v2 file, so this is the state a real LIMN run
  lands in.
- Repro: read `pipeline.py:1449-1453`; `load_runs` failure path is caught at `:1414-1418` and converted to
  `runs = []`. Ran the equivalent stage in isolation against an empty temp root:
  `validate_results.py --root <empty>` → exit 0 by the same design (`validate_results.py:39-43`).
- Impact: after the real RTX 5070 Ti evaluation, the pipeline prints `[OK]` for "validation-index" while
  the only artifact that exists (the aggregate report) was never validated; the per-query evidence is
  absent by design and the stage says nothing about it.
- Minimal fix: when `WORKBENCH_RESULTS_ROOT` is non-empty but `load_runs` raises, return 1; and have the
  evaluation stage emit at least one schema-v2 file (or have `validate_results` accept and report the
  aggregate-only case explicitly) so `[OK]` means "validation ran".

### C-5 (P3) — gate exports/reports statuses it can never emit; frontend keys off them

- Location: `workbench/backend/paper_gate.py:129-155` (`BLOCKED_SOURCE`, `BLOCKED_ENVIRONMENT` defined, never
  returned anywhere in the file) vs `workbench/scripts/pc_readiness.py:31–55` which re-defines the same
  names for a different vocabulary. Frontend: `workbench/frontend/gate.ts:29-74` renders only
  `CAPTION THAY THẾ` and `LỆCH PAPER …`; `App.tsx:58` prints ``(verdict.passed===false ? " (not a certified pass)" : "")``.
- Root cause: two status vocabularies with the same spelling and different owners, no cross-file
  assertion that they mean the same thing.
- Impact: a reader of either file cannot tell which statuses are reachable; `BLOCKED_SOURCE`/
  `BLOCKED_ENVIRONMENT` look like gate outcomes and are not.
- Minimal fix: delete the two dead constants from `paper_gate.py`, or add one test asserting the emitted
  status set equals the declared set.

### C-6 (P3) — `validate_results.py` name and `--strict-real` semantics overstate what is checked

- Location: `workbench/scripts/validate_results.py:18-32` — only `upstream_commit`, `checkpoint_sha256`,
  `command_digest`, `environment_digest` presence; `data_kind == "mock"` is exempt (`:25`); `:39-43`
  returns success when zero files are loaded. Every protocol/gallery/dedup invariant lives in imports
  (`:13-14 ` → `index.load_runs`, `operator_config.resolve_config`).
- Evidence: `.venv/bin/python workbench/scripts/validate_results.py --root workbench/artifacts/results --strict-real`
  → `[OK] validated 6 runs under workbench/artifacts/results`, `exit=0` — all six are mocks, so nothing was
  actually checked; the LIMN run directory
  (`workbench/artifacts/logs/limn__ckpt__20261011T114609437143Z_050e1b05/command.json`, `return_code: 0`)
  has no corresponding result artifact at all.
- Impact: `[OK]` is read by operators as "protocols validated"; the only cohort check that runs is
  "same `protocol_id` string within one analysis" (`storage.py:150-155`, `analysis.py:12-14`), which
  cannot detect that two protocols describe the same or different cohorts.
- Minimal fix: rename to `validate_result_artifacts.py`, and print the checked count (already available
  as `len(runs)`) plus the exempt mock count so `[OK]` cannot be read as a cohort check.

### C-7 (P3) — `LIMN/replay` `protocol` string is constant; nothing validates it against upstream evidence

- Location: `workbench/replay/limn.py:181,196,226` (three hard-coded `"fashioniq_val_split"`),
  `:193-197` (equality check only against the same three constants).
- Verdict: the class is **correct** (B4) but the check at `:193-197` is circular — it compares the
  replay's own constant to itself. Any future change to `LIMN/datasets.py` gallery construction (e.g.
  switching the LIMN+ variant, or a pin bump) would keep the stamp while the cohort changed.
- Minimal fix: assert the gallery against `src/datasets/fashioniq.build_pair_union_gallery` output for
  the same annotations (one test), or derive the stamp from the resolved protocol object.

---

## D) Probable defects needing reproduction (not moved into C)

- **D-1 (P2 probable).** DCNet/INTENT wrong cohort (C-2) is established for DCNet from `trainer_ref.py`
  and `model/metric.py`; INTENT's `workbench/third_party/INTENT` evaluator was not opened in this slice,
  so the INTENT half of the label is PROBABLE, not CONFIRMED. Verification: grep INTENT's evaluator for
  a reference mask.
- **D-2 (P2 probable).** `papers.yaml:1975-…`-class rows: no `gallery_protocol` check exists in
  `VerifyPapersA.json` for Air-Know/HABIT/INTENT/ConeSep or in `VerifyPapersC.json` check 1 for TGCIR;
  HINT/ENCODER/PAIR/SPRC/CSMCIR do have one (`VerifyPapersB.json` checks 22/25/28 and 4/12/19).
  PROBABLE: of the 33 committed rows, only 6 carry a `gallery_protocol` check in the verification ledger (`sprc`, `csmcir`, `hint`, `encoder`, `pair`, plus the shared `clip4cir_rn50x4_fiq_gallery_protocol` record); the remaining 27 rows, including the LIMN row itself and DCNet/INTENT, state a `gallery_protocol` with no independent verification entry.
  Verification: extend the verifier ledger and re-run it.
- **D-3 (P2 probable).** `papers.yaml:1600` (`checkpoint_training_noise_pct: null`) vs
  `models.yaml:1000` for the same three LIMN checkpoints disagree. `select_reference` handles `None`
  correctly (`benchmark_references.py:175-178`), and `checkpoint_selection.py` refuses `strict_clean`
  under its own documented rule, so there is no wrong verdict — but the two files answer "what
  condition is this weight?" differently, and `papers.yaml` for DCNet/CLIP4Cir/Combiner asserts `0`
  where `models.yaml:407,465,520,531,600,654,715,779,837` records `null` or a comment-inferred `0`
  (the `VerifyPapersC.json` recommendations already flag this). UNVERIFIED which side is right.
- **D-4 (UNKNOWN, blocked on hardware).** Whether a real LIMN run reproduces 55.91/77.82 at all. Cannot
  be answered statically; no run exists (`workbench/artifacts/reports/` is empty and the single ledger
  line is a test fixture).

---

## E) Test coverage gaps (relative to this slice)

1. No test asserts that the *allocated* mapping-status vocabulary in `papers.yaml` is one the gate
   accepts. `test_final_hardening_regressions.py:335-349` builds a synthetic `SimpleNamespace` reference
   carrying `"checkpoint_mapping_status": "VERIFIED_METADATA"` — a value no committed row uses — so the
   C-1 defect is invisible to the suite. `test_paper_gate.py` (65 passed) likewise uses fixtures.
2. No test asserts `native_protocol == gallery_protocol of the selected reference` across the registry;
   `test_protocols.py:26-37` asserts ids exist and are conventional, not that they agree with the
   committed rows.
3. No test asserts `reference_excluded` agreement between a paper row and the pinned evaluator for the
   affected models (the DCNet/INTENT case).
4. No test round-trips a real evaluator's artifact through `validate_results.py`; all coverage uses mocks
   (`test_final_hardening_regressions.py`, `test_workbench.py`).
5. No test pins the `papers.yaml` ↔ `models.yaml` (`reported_r10/r50/mean`) agreement; it currently holds
   (verified below) but nothing keeps it holding.
6. `ReportedMetrics` finiteness/uniqueness: no test (C-3).
7. `pipeline.run_validation_index` empty-runs success path: no test (C-4).

---

## F) Recommended minimal fixes (ordered, smallest first)

1. **One mapping vocabulary.** Accept `VERIFIED`/`VERIFIED_METADATA` at `paper_gate.py:413` **or**
   rewrite the 8 rows — pick one; one line either way. This is the only thing standing between a perfect
   LIMN run and a certified verdict.
2. **Re-derive DCNet/INTENT protocol class** from their pinned evaluators and correct the
   `gallery_protocol`/`reference_excluded` fields in `papers.yaml` and `models.yaml` (no metric changes).
3. **Bound the metric carriers**: `Field(ge=0.0, le=100.0, allow_inf_nan=False)` on
   `schemas/results.py:16-19`; `allow_inf_nan=False` on `TopResult.score`.
4. **Make validation-index honest**: fail the stage when `load_runs` raises on a non-empty results root
   (`pipeline.py:1414-1453`).
5. **Delete the two unreachable gate statuses** (`paper_gate.py:129,131`) or assert reachability.
6. **Emit at least one schema-v2 artifact** for the LIMN replay (honour `EvalRequest.output_path`) so the
   per-query immutability/duplicate machinery has something to protect, or make `validate_results.py`
   report "aggregate-only" explicitly instead of `[OK] validated N runs`.

Explicitly **not** recommended: any change to the tolerance (`5.0`, `paper_gate.py:28`), to the recall
definition, to the gallery construction, or to any published number. The protocol math is correct (B4);
the defects are labelling, vocabulary and contract completeness.

---

## G) Exact regression tests required

| # | test | asserts |
|---|---|---|
| G1 | `test_every_committed_reference_mapping_status_is_gate_acceptable` — load `papers.yaml`, feed each row's `checkpoint_mapping_status` through the gate's cap logic | no VERIFIED-claiming row is silently demoted (fails today: 8 rows) |
| G2 | `test_registry_and_reference_protocols_agree` — for each model, every VERIFIED row's `gallery_protocol` ∈ `{native_protocol} ∪ supported_protocols` | C-1/C-2 class; passes today |
| G3 | `test_reference_excluded_claim_matches_pinned_evaluator` — for `dcnet` (and `intent` once opened) assert the pinned evaluator masks/removes the reference | fails today for DCNet (C-2) |
| G4 | `test_reported_metrics_are_finite_and_bounded` — `ResultRun.model_validate` with `r10 ∈ {NaN, inf, -0.1, 100.1}` | raises for all four (fails today, C-3) |
| G5 | `test_pipeline_validation_index_fails_when_results_root_is_unreadable` — unreadable/unparseable file under results root | stage returns ≠ 0 (fails today, C-4) |
| G6 | `test_limn_replay_gallery_matches_workbench_pair_union` — same annotations through `build_pair_union_gallery` and the replay's derived gallery list | identical order and content (guards C-7) |
| G7 | `test_limn_reference_row_permits_certified_pass` — synthetic run equal to `limn_fashioniq_base_iter0_table1` with `checkpoint_mapping_verified=True` | status is a single `PASS_*` that is **not** `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` (fails today, C-1) |
| G8 | `test_gate_status_vocabulary_is_reachable` — collect the statuses `evaluate_gate` can return, compare with the module constants | no unreachable/undeclared status (C-5) |

---

## Checklist answers

**Q1. Does `fashioniq_val_split` exist and agree across registry, reference set and validators?**
**YES.** The id exists and agrees wherever it appears in the registry, the reference set and the schema-level validator. It is not a "spelling" question: the four places that must agree do agree, and the disagreements that do exist are on other protocol labels (listed below).



The exact answer:

- `fashioniq_val_split` — **exists** in `models.yaml:16-19`, in `schemas/results.py:8-13`
  (`PROTOCOL_LITERATURE_LABELS`/`PROTOCOL_IDS`), in `frontend/protocols.ts:3`, in
  `src/evaluation/fashioniq.py:179`, in `workbench/replay/limn.py:181,226`, in `models.yaml:920-921`
  (LIMN `native_protocol` and `supported_protocols`) and in `papers.yaml:1607` for the LIMN row.
  No disagreement among those. `literature_split_label` is `val` in both registries
  (`models.yaml:919`, `papers.yaml:1606`) and matches `PROTOCOL_LITERATURE_LABELS` (`schemas/results.py:11`).
- **Every protocol name in code or registry that does not exist or disagrees:**
  1. `fashioniq_val_split` — no disagreement (control).
  2. `fashioniq_full_gallery_ref_excluded` — exists as an id, but its **semantics disagree with its own
     pinned evaluator for `dcnet` (and probably `intent`)**: `papers.yaml:1460-1472` /
     `models.yaml:626-627` say `reference_excluded: true`; `third_party/DCNet/trainer/trainer_ref.py:126-178`
     + `model/metric.py:1-19` never remove the reference. C-2.
  3. `fashioniq_val` / `fashioniq_original` — accepted as aliases by the evaluator table
     (`src/evaluation/fashioniq.py:174,180`) and used only in `tests/unit/test_fashioniq_{dispatch,protocols}.py`;
     **rejected** by the canonical schema (`schemas/results.py:85-88`) — verified: feeding
     `protocol_id: "fashioniq_val"` into `ResultRun` raises `unsupported protocol_id: fashioniq_val`.
     No committed artifact uses them, so it is a latent, not a live, inconsistency.
  4. `checkpoint_mapping_status: VERIFIED` (papers.yaml ×8) vs `VERIFIED_METADATA` (models.yaml ×16) — two
     vocabularies for one field; this is the disagreement that costs the certified pass. C-1.
- **Verdict per the mission's three-way test: `fashioniq_val_split` is the required LIMN id and it agrees
  across registry, reference set and (schema-level) validator. VERIFIED.** The registry↔validator
  agreement is limited to string identity: `validate_results.py` validates nothing protocol-specific
  (C-6).

**Q2. What does a truthful verdict require after a real LIMN run?**
A verdict may be called truthful only when all of the following are on record:
(a) the aggregate + per-category R@10/R@50 from `workbench/replay/limn.py` via the pinned
`LIMN/test.py:test` (STATIC_AUDIT of that path is what §B4 establishes); (b) the three checkpoint
SHA-256s verified against the author LFS metadata (`replay/limn.py:41-50`, currently satisfied on this
host — `validate_checkpoint_bundle` returned `[]`); (c) the pinned commit `7d7bc9b1…` on a clean
checkout (`replay/common.py:52-79`); (d) `run_meta` carrying `checkpoint_training_noise_pct: null` and
`checkpoint_mapping_verified: true` (`evaluate_models.py:155-166`), so the gate's condition matching is
honest; (e) the report written with the evidence-ledger entry (`evaluate_models.py:1508-1540`,
`main.py:143-177`); (f) the observed per-category values forwarded
(`evaluate_models.py:192`, `paper_gate.py:377-393`) so a broken Dress cannot hide behind the macro.
Today a truthful verdict is still capped at `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` by C-1. Until C-1 is
fixed, the truthful sentence is "numerically comparable to the paper within 5 % relative error; not
certified as an official reproduction, for a registry-vocabulary reason that has nothing to do with the
run".

**Q3. Is a Recall below the paper value an infrastructure failure or a research finding?**
Neither, by default — it is a **measurement that must first survive the comparability gate**, and the
code already models this correctly:
- `≤5 %` relative below the paper → `direction: "lower"` recorded (`paper_gate.py:206-207`) and it can
  still be a numeric pass (verified: 53.2/74.0/63.6 → `rel%` 4.85/4.91/4.89, `PASS_*`, `direction=lower`).
- `>5 %` relative → `ALERT_METRIC_DEVIATION` with a per-metric `_failure_note` (`paper_gate.py:415-418`).
- No reproduced value where the paper has one → `INCOMPLETE_METRIC_EVIDENCE`, explicitly **not** a
  deviation (`paper_gate.py:419-423`; verified: dress-only observation → that status, no deviation label).
- Non-zero exit → `FAILED_EVALUATION` (`paper_gate.py:322-325`).
It is an **infrastructure failure** only when one of the precondition states fires first — `NOT_RUN`
(no real execution), `FAILED_EVALUATION`, `BLOCKED_PROTOCOL_MISMATCH`/`BLOCKED_CHECKPOINT_MISMATCH`
(`paper_gate.py:287-292`), or the whole run is unusable because the eval never produced output. It is a
**research finding** only after those are excluded *and* the cohort is evidenced — and for LIMN the
cohort evidence is exactly the piece D-2 says is missing for the TGCIR-style rows (it is present for
LIMN: `papers.yaml:1607-1608` matches `LIMN/test.py:80-85` + `datasets.py:158-170`). Note the
asymmetry the gate deliberately keeps: a run **above** the paper by >5 % is a deviation too
(`_is_number`/`compare_metric` use `abs`), so a suspiciously high number cannot be reported as a
reproduction.

**Q4. Can `./pc.sh` even be judged?** `pc.sh` **does not exist** at the repository root
(`ls pc.sh` → `No such file or directory`), and no document in `workbench/docs` mentions it. The
documented entrypoint is `workbench/run.sh` (`run.sh:2-4`, `autopilot` → `scripts/autopilot.py`) and the
LIMN pilot config documents `./workbench/run.sh autopilot --config workbench/configs/pc_autopilot.pilot_limn.yaml --apply --resume`
(`pc_autopilot.pilot_limn.yaml:58-59`). The *feasibility* of what `pc.sh` is supposed to do is VERIFIED
as far as static evidence goes: LIMN is the only registry model with `checkpoint_mapping_status:
VERIFIED_METADATA` for a whole bundle, its three checkpoints are present and hash-verified on this host,
`data/FashionIQ/{captions,image_splits,resized_image}` are present, and the source pin matches
(`7d7bc9b116f594a65ac22457491edf28a88d3c3e`). The run remains **NOT_RUN**: no GPU inference was executed
and `workbench/artifacts/reports/` is empty.

---

## Execution trace (this audit)

```
git log --oneline -3                       -> 0b0f3a6 · e4868bb · e9cafaf
pc.sh                                      -> absent
read paper_gate.py / benchmark_references.py / checkpoint_selection.py
     schemas/results.py / validate_results.py / papers.yaml (all 1837 lines)
select_reference(limn, noise=null, protocol=fashioniq_val_split)
                                           -> limn_fashioniq_base_iter0_table1 | exact condition match
evaluate_gate(paper-identical metrics)      -> PASS_METRICS_ONLY_MAPPING_UNVERIFIED, passed=False
evaluate_gate(dress +10pp)                  -> ALERT_METRIC_DEVIATION
evaluate_gate(r10=NaN)                      -> ALERT_METRIC_DEVIATION ("not a finite number")
evaluate_gate(dress-only partial)           -> INCOMPLETE_METRIC_EVIDENCE (no deviation claim)
ResultRun.model_validate(NaN/inf/-1/1e9)    -> ACCEPTED (all four)
ResultRun.model_validate(protocol_id=fashioniq_val) -> rejected
grep 10e10 third_party/DCNet               -> no match
grep 10e10 third_party/LIMN/LIMN/test.py   -> :82 (mask present)
grep 10e10 third_party/TGCIR/test.py       -> :67 (mask present)
validate_results.py --root workbench/artifacts/results --strict-real -> [OK] validated 6 runs, exit=0
pytest workbench/tests/test_paper_gate.py -q                        -> 65 passed
pytest workbench/tests/test_checkpoint_selection.py -q              -> 16 passed
git -C workbench/third_party/LIMN rev-parse HEAD                    -> 7d7bc9b1… (== registry pin)
LIMN validate_checkpoint_bundle(artifacts/checkpoints/limn)         -> [] (three SHA-256s match)
```

## Final verdict

**VARIANT.** The LIMN reproduction path is protocol- and metric-faithful to the pinned upstream
(VERIFIED: `datasets.py:158-170` + `test.py:76-95`, registry id `fashioniq_val_split` agreeing across
`models.yaml`, `schemas/results.py`, `frontend/protocols.ts`, `replay/limn.py` and `papers.yaml:1607`),
but the *verdict* it can emit is wrong: `paper_gate.py:413-414` compares a `VERIFIED` paper-registry
value against the `VERIFIED_METADATA` column vocabulary and therefore demotes every exact CSMCIR/HINT/LIMN
reproduction to `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` (CONFIRMED, P1). One further cohort label is
contradicted by its own pinned evaluator (`reference_excluded: true` for DCNet; CONFIRMED, P2), the
canonical metric carriers admit NaN/±inf (CONFIRMED, P2), and `validation-index` reports success on an
empty/unreadable results root (CONFIRMED, P2). Runtime reproduction remains **NOT_RUN**; `./pc.sh` does
not exist.

**Can this implementation fail during the real first run on an RTX 5070 Ti PC?** Yes — the run itself is
statically feasible (checkpoints hash-verified, dataset present, pin matched, adapted sm_120 runtime
declared), but after it succeeds the pipeline will report `[OK]` for validation-index while
`./pc.sh`-driven validation never checked a real artifact, and the LIMN verdict will be
`PASS_METRICS_ONLY_MAPPING_UNVERIFIED` with `passed=false` even for a perfect reproduction, for a
registry-vocabulary reason unrelated to the run; additionally `pc.sh` is not present in the repository at
HEAD `0b0f3a6`.
