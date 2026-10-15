# Slice C — SCIENTIFIC CORRECTNESS — independent source audit

Auditor: independent read-only reviewer (slice C). Branch `research/cir-failure-workbench`, HEAD
`e9cafafd0a13b0d9084ef16c75e7cf57c8260f96` (verified: `git rev-parse HEAD`). Read-only: no file under
the repo was written except this report; no pytest run, no docker, no network fetch, no GPU inference.

Revision audited (md5, read-only):

```
b572e2282a91eaebc8665e0e128cd1d5  workbench/scripts/evaluate_models.py
378dc806c07028b3270dc8e2fb8d13bd  workbench/backend/paper_gate.py
c05d498921629339b50c6780eb40860e  workbench/backend/benchmark_references.py
5f859155b260ad67ab6e2abafe06d0d8  workbench/backend/checkpoint_selection.py
193bd25455c4263f9faeb1ae427625cc  workbench/backend/metrics_extraction.py
5650234aa551706fe927337754844960  workbench/backend/adapters/base.py
ae89abc8c3374f58193db2497929a8e3  workbench/backend/adapters/models.py
b4adee7a1f7b0f14253ee43096629a4c  workbench/replay/limn.py
24c795da907e9c7da44b377fa372d3e7  workbench/replay/common.py
5918603fa3514a9cb39dab57ae41f68f  workbench/replay/tgcir.py
393db8f3b42e092102285100f16ace54  workbench/backend/main.py   (later re-read as
   1040642a790e2059c6aac91a3e8a4ec3 — the main agent added a duplicate-job guard and queue
   recovery after this audit; the audited regions `_aggregate_reports` :112, `_aggregate_report_is_well_formed`
   :143, `/api/benchmark-runs` :189-194, `/api/runs/gate` :196-199 are byte-unchanged,
   re-verified at the new hash)
a017f87ccb96f07f3c1f6f4d0acb43ad  workbench/backend/autopilot.py
e033a7a0a46cb14bf12c9a98939baefc  workbench/registry/paper_benchmarks/papers.yaml
```

Every claim is `STATIC_AUDIT` (source read + in-process reproduction of pure logic) unless a command and
its observed output are quoted. **No GPU inference was run; no runtime result is claimed as observed.**
Local interpreter for the in-process reproductions: `python3` 3.14.4 under
`WORKBENCH_NO_ENV_CACHE=1`. Upstream checkouts are read from `workbench/third_party/<dir>` at the
registry-declared pins (LIMN `7d7bc9b1…`, CSMCIR `774f94e2…`, checkouts verified clean of tracked edits;
only untracked `__pycache__` present). Existing workbench audits were treated as hypotheses only.

Vocabulary: verdict ∈ {VERIFIED, VARIANT, BLOCKED, NOT_RUN, UNKNOWN}; class ∈ {CONFIRMED, PROBABLE,
UNVERIFIED, FALSE_POSITIVE}; severity P0 = breaks research workflow / invalidates scientific evidence,
P1 = deployment-runtime-integration defect blocking a user journey, P2 = workaround exists, P3 = cosmetic.
Severity `NONE` where the concern was checked and is a non-issue.

---

## 1. Checklist answers

### C1. Checkpoint identity binding — does the result record bind the sha256-verified file, or only the id?

**Verdict: VARIANT — the run record binds the file bytes whenever the registry publishes an author hash;
when it publishes none, the binding is a self-referential digest.**

- Single-file path: `plan.checkpoint_sha256 = sha256_file(checkpoint_file)`
  (`evaluate_models.py:367-368`) is written to the report as `checkpoint_sha256`
  (`:1478`) and to the completion proof (`:1540`), and the proof's value is re-checked against a fresh
  hash of the live file at validation time (`_hash_checkpoint()` `:653-663`, compare `:824-829`:
  `live_digest = _hash_checkpoint() or plan.checkpoint_sha256`).
- Bundle path: `_members_live()` (`:676-748`) recomputes the member manifest from live files and
  compares it to the digest the proof records; I reproduced both accept and reject states
  (STATIC_AUDIT, in-process):

  ```
  clean bundle: ACCEPTED
  tampered non-anchor member: rejected -> checkpoint members no longer match the recorded artifact manifest
  ```

  so a changed `0_toptee_best_model.pt` does invalidate a LIMN bundle run, not just the anchor.
  `_bundle_members_live()` (`:770-813`) is dead code (never called) — its non-anchor per-member digest
  fallback is therefore never exercised: when the proof carries no `bundle_manifest_digest`, `_members_live`
  falls back to `plan.directory_members` digests, which are hashed from disk at plan time, so the
  comparison degenerates to "hash equals itself". Not reachable from `execute()` (the proof always records
  a manifest) but reachable from `completion_validator` (`:1636-1639` via `_resolve_bundle`). P3, dead code.
- **Binding strength is data-dependent.** Only 11 of 25 registry variants carry `expected_sha256`
  (airknow n05/n08, conesep n02/n05/n08, encoder, pair b1/b2, clvc_net, dcnet, combiner, clip4cir,
  tgcir, sprc have none). For those, `checkpoint_is_present` (`registry.py:273-301`) only checks
  size > 0, and the recorded digest is whatever bytes happen to be on disk — a substituted file is
  recorded faithfully but never detected as a substitution. The report carries no field saying
  "author hash not available", and neither does the gate: the `reference_evidence` of a
  `PASS_*` verdict contains no checkpoint-digest field at all (observed output below).

  Observed (in-process), the very best case — csmcir, author hash present, every metric paper-exact:

  ```
  status: PASS_WITHIN_5_PERCENT passed: True
  reference_evidence keys: ['category_metrics','checkpoint_training_noise_pct','derived_metrics',
    'gallery_protocol','metric_provenance','model_id','paper_id','paper_title','reference_id',
    'reference_status','registry_digest','registry_file','registry_version','reported_metrics']
  has any checkpoint-hash/sha field: ['registry_digest']
  ```

  `checkpoint_mapping_verified=True` is the only identity signal in `run_meta`, and it is derived from
  the registry's `checkpoint_mapping_status`, not from bytes.

- **Severity P2**, recommended action: (a) add an explicit `author_sha256_available: bool` (and the
  recorded `local_sha256`) to the gate's `run_meta`/verdict so a `PASS` cannot be read as byte-level
  identity when no author hash exists; (b) delete `_bundle_members_live()` or call it.

### C2. Bundle completeness enforcement — `base_iter0_all_categories` must yield all 3 categories; can a run end with 2 and still be certified?

**Verdict: VARIANT — for the LIMN bundle path the completeness contract is enforced at three
independent layers and I could not break it; for the *gate* the contract is weaker than "all three
categories observed": a run that observes 1 or even 0 of the 3 categories can still return
`passed=True` when the reference row prints no per-category table.**

- Enforced (all STATIC_AUDIT, code cited):
  1. `macro_aggregate` refuses unless exactly the three canonical categories, in order, all
     `complete` (`common.py:90-108`); `evaluate_categories` only calls it when
     `categories == CATEGORIES` (`limn.py:152-157`).
  2. `parse_limn_output` returns `EVALUATION_COMPLETED_METRICS_UNPARSED` unless
     `set(cat_metrics) == {"dress","shirt","toptee"}` **and** `agg["complete"]` (`metrics_extraction.py:132-141`).
     Observed on the real emitted shape:

     ```
     3-cat pretty-printed JSON: EXTRACTED_VERIFIED {...} PARITY_NOT_EVALUATED
     2-cat  pretty-printed JSON: EVALUATION_COMPLETED_METRICS_UNPARSED None cats= ['dress', 'shirt']
     ```
  3. Preflight refuses a category that is missing/empty/hash-mismatched (`limn.py:29-49`), and a
     single-category run is explicitly documented as "never a complete benchmark result"
     (`replay/limn.py:211`; behaviour `replay/limn.py:214-215`).

- **Gap (P1).** `paper_gate.evaluate_gate` treats a reference that publishes no `category_metrics` as
  having nothing to compare per category (`paper_gate.py:372-386`), and then the only coverage signal is
  `category_unchecked`, which can only be non-empty when the paper *does* publish categories. All 28 committed verified rows publish per-category cells (checked); a row that prints only `Avg.` (or a future row added without
  `category_metrics`) yields an unconditional pass on any observed coverage. Reproduced in-process
  against a reference whose `reported_metrics` match and which publishes no category table but *does*
  state all six comparability fields (so `BLOCKED_REFERENCE_INCOMPLETE` does not fire):

  ```
  1/3 categories   -> PASS_WITHIN_5_PERCENT        passed=True
  0/3 categories   -> PASS_WITHIN_5_PERCENT        passed=True
  3/3 categories   -> PASS_WITHIN_5_PERCENT        passed=True
  ```

  Against the *committed* rows the incomplete cases are caught, because every one of the 28 verified committed rows
  publishes `dress/shirt/toptee` (checked: 0 rows lack them) — `PASS_WITH_UNCHECKED_CATEGORIES`,
  `passed=False`, observed for 2/3, 1/3 and 0/3 coverage. So the defect is a latent hole in the gate, not
  a currently-fireable false pass; the LIMN/CSMCIR reports cannot end with 2 categories certified today.
  Recommended action: make the gate refuse an *observed* run whose category coverage is smaller than the
  coverage the run's own protocol declares (or than the reference's category count when non-zero),
  independent of what the paper prints — i.e. a completeness floor sourced from the run/bundle, not from
  the paper row.

- Extra facet worth noting (P2): a **single-category LIMN diagnostic** is a first-class command
  (`--checkpoint base_iter0_dress` survives `candidates()` unmodified — observed:
  `--checkpoint=['base_iter0_dress'] -> [('limn', 'base_iter0_dress')]`), and `LIMNAdapter.command`
  emits `--category dress` (`adapters/models.py:127-131`), which `replay/limn.py:214` writes as a
  **create-only standalone JSON** (`limn.py:231-245`) carrying `complete_benchmark: false`. That file
  uses the same `model_id/protocol/categories/aggregate` vocabulary as the bundle report; nothing scopes
  it to `workbench/artifacts`, so a diagnostic placed beside a certified report is a shape-compatible
  neighbour. It cannot pass the parser (no macro, `complete: false`), so the risk is operational, not
  a false pass.

### C3. Category aggregation math — macro mean over present categories vs declared

**Verdict: VERIFIED (with one P2 attribution gap).**

- The number used for comparison is the **recomputed** macro, never the declared one:
  `parse_limn_output` re-derives `(d10+s10+t10)/3`, `(d50+s50+t50)/3` and `(r10m+r50m)/2` and rejects the
  payload when the declared values disagree by more than 0.05 (`metrics_extraction.py:167-186`).
  `parse_csmcir_output` does the same against `average_recall_at10/at50/average_recall`
  (`metrics_extraction.py:264-283`). Observed: a payload with a self-consistent declared macro yields
  `EXTRACTED_VERIFIED {'r10': 55.9133, 'r50': 77.82, 'mean': 66.8667}`.
- The aggregation is a *macro* only when all three categories are present; a partial set never produces a
  macro (`common.py:92-95`; `limn.py:152-157`). Never query-weighted.
- CSMCIR's native evaluator computes exactly this macro
  (`CSMCIR/src/validate_blip_csmcir.py:624-628`), so reading `average_recall_at10/50` is faithful, and
  the parser's independent recomputation restates it rather than inventing it.
- P2: the workbench-derived macro for a model whose paper prints no macro is labelled
  `WORKBENCH_DERIVED` on the *reference* side (`benchmark_references.py:44-64,72-80`) but the run side
  records only `observed_metrics`; the autopilot table prints `Macro R@10` with no derived/published
  marker (`autopilot.py:1052-1108`). The paper's `Avg.` for the committed rows is itself the rounded
  macro (verified in the registry's own `verification_method` text), so the values should agree — the
  risk is only presentation of provenance.
- Derived-mean comparison is sound, and I checked the failure mode rather than assuming it. If a run's
  `r10`/`r50` are each within 5 % of the paper's, then `(r10+r50)/2` is within 5 % of the paper's mean
  by construction, so a `WORKBENCH_DERIVED` mean can never *create* a deviation on a consistent run.
  Observed on `clip4cir_rn50x4_fullft_fiq_tab9` where the paper prints no scalar mean:
  ```
  mean provenance: WORKBENCH_DERIVED | value: 55.36
  verdict: ALERT_METRIC_DEVIATION passed= False
     r10: paper=43.78 prov=PAPER_REPORTED rel=0.0 ok=True
     r50: paper=66.93 prov=PAPER_REPORTED rel=0.0 ok=True
     mean: paper=55.36 prov=WORKBENCH_DERIVED rel=7.48 ok=False
  ```
  The triggering run in that reproduction reported `mean=59.5`, i.e. a mean **inconsistent with its own
  r10/r50**; every parser rejects that before the gate sees it (`metrics_extraction.py:186-198,277-283`,
  recomputation with a 0.05 guard), so the comparison acts as a consistency check, not a false
  positive. The provenance of the paper side is recorded per comparison (`paper_provenance`) and
  displayed, so a reader can see the mean was not printed by the authors.

### C4. Query construction / reference-target mapping — correct columns, no train/val leakage into test splits

**Verdict: VARIANT — the CSMCIR caption path does resolve (initial hypothesis withdrawn); the
workbench's own dataset contracts and the LIMN path are VERIFIED; one provenance caveat remains
(the target-caption file of `utils_csmcir.py:100` is the UNVERIFIED opt-in alias), and one UNVERIFIED
residual (the DataLoader ordering in CSMCIR's index walk), both recorded below.**

- **CSMCIR caption read paths (P3, FALSE_POSITIVE for the missing-file hypothesis; one documented
  provenance caveat).** My first reading was that the staged asset and the read path disagree — they do
  not, and the correction is recorded here rather than dropped. The canonical dataset link is a
  directory symlink (`prepare_dataset.py:65`
  `destination.symlink_to(root, target_is_directory=True)`), so `data_utils_csmcir.py:16`
  `base_path = <checkout>` and `:408` `base_path/'fashionIQ_dataset'/'qwen_captions'/f'{cat}_cot_val.json'`
  resolve to `<fashioniq_root>/qwen_captions/{cat}_cot_val.json` — exactly the registry's
  `destination_scope: fashioniq_root`, `destination_path: qwen_captions/{cat}_cot_val.json`
  (`auxiliary_assets.yaml:10-11,33-34,56-57`). The second reader,
  `CSMCIR/src/utils_csmcir.py:100`, reads `<checkout>/COT_ours2/fashioniq/{cat}_cot_val.json`, which is
  the `destination_scope: model_source` alias (`auxiliary_assets.yaml:86-87,109-110,132-133`), and the
  alias is opt-in (`download_auxiliary_assets.py:129 --stage-caption-aliases`) with
  `provenance_status: UNVERIFIED` plus a registry note citing `COT_OURS2_AUDIT.md`. Both files are
  therefore required and both are stageable; the reader uses `['Final_Caption']`
  (`data_utils_csmcir.py:309,311,435`) and both files carry that key per image id (the content contract
  asserts `record_key_fields` incl. `Final_Caption`, keyed by every id of `split.{cat}.val.json`), so
  reference-id and target-id lookups both exist. The dict's name (`target_image_captions`, used for the
  **reference** caption at `:309,435`) is misleading but the content is per-image, so no target-derived
  query signal is introduced. Remaining caveat, already documented and labelled: the alias is
  `SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN`, so a CSMCIR run is a variant, and the run record says so
  (`caption_alias_staged` → `PASS_WITHIN_5_PERCENT_VARIANT`).
- The CSMCIR *index* walk is deterministic in source: `extract_index_blip_caption_features`
  (`CSMCIR/src/utils_csmcir.py:89-90`) uses `DataLoader(dataset, batch_size=32, num_workers=2,
  pin_memory=True, collate_fn=collate_fn)` — `shuffle` defaults to False — `collate_fn`
  (`utils_csmcir.py:396-403`) is `default_collate` on the non-`None` batch in order, and
  `classic_val_dataset` is built per category (`validate_blip_csmcir.py:597-604`), so `index_names`
  extends in dataset order and `sorted_index_names` is consistent between the two extractors. That is
  STATIC_AUDIT; whether the pinned `FashionIQDataset.__getitem__` ever returns `None` for a val id is
  **UNKNOWN** here (no data was inspected), so the multi-worker-`None` ordering hazard is
  **UNVERIFIED, not PASS**.
- Reference exclusion is genuinely absent: `compute_fiq_val_metrics` (`validate_blip_csmcir.py:38-57`)
  only `argsort`s `1 - pred_sim`; the reference name is retrieved from the index but never masked.
- **No train/val leakage in the declared layouts (VERIFIED).** `fashioniq_layout.py:12-23`
  (`standard_paths`) requires `cap.{cat}.val.json` + `split.{cat}.val.json` only;
  `ilearn_resized_training_paths` (`:37-41`) *adds* `cap.{cat}.train.json`, which LIMN's
  `get_test_data` (`LIMN/LIMN/datasets.py:129-168`) does read for the correction dictionary and
  `split.{cat}.val.json` for the gallery, and which `train_init_process` also reads — the train file is
  a vocabulary/dictionary source, and the val gallery comes only from the val split. No test-set query is
  drawn from `train`.
- **Gallery/ranking consistency in the LIMN path (VERIFIED, with a latent P2).** `test.py:76-95`
  masks the reference image (`sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`) and ranks the
  **union** gallery built from `image_splits/split.{cat}.val.json`
  (`datasets.py:158-166`), i.e. exactly the full val split. Latent P2: `test_targets_id` dedups ids in
  first-encounter order and only **queries** are encoded on the fly (`test.py:26-46`); each unique target
  is encoded once in that dedup order (`test.py:47-63`), so if any target image were ever missing from
  the `test_queries` union the positional `all_imgs` mapping would silently shift. No duplicate in the
  union can shift it (the dedup `if … not in` makes a second occurrence a no-op). Not reachable with a
  consistent `split.*.val.json`.

### C5. Retrieval ranking semantics — argpartition/topk off-by-one, self-match exclusion

**Verdict: VERIFIED for the workbench (it produces no rankings); the upstream ranking semantics are
faithful to the sources it cites, with one protocol-level discrepancy (P1) recorded in C8.**

- The workbench never computes a ranking. The only ranking-shaped code in the audited slice is
  `_top_results` (`storage.py:89-104`), which *reads* `top_results` ordered by `rank` and validates
  rank/image-id/target consistency in the schema (`schemas/results.py:38-51,95-108`): ranks must be
  contiguous from 1, image ids unique, `target_rank <= gallery_size`, and `top_results[target_rank-1]`
  must be the target when the depth covers it. An off-by-one in a producer would be rejected, not
  averaged. Conversely, no production producer of that shape exists (C14).
- Upstream: every FashionIQ evaluator in the slice masks the query image with `-10e10` before
  `argsort` and slices `[:50]` (`LIMN/LIMN/test.py:80-95`, `TGCIR/test.py:67-79`, `HABIT/test.py:64-65`,
  `INTENT/test.py:64-65`, `AirKnow/test_BLIP2.py:65-66`, `PAIR/test.py:68`, `HINT/test.py:83`,
  `CLVCNet/test.py:64-76`, `ENCODER/test.py:68-73`). The recall predicate counts
  `test_targets_id.index(target) in nns[:k]`, i.e. 0-based slicing of a `[:50]` prefix — no off-by-one,
  and the masked reference can never occupy a slot. `-10e10` is a legitimate dense-mask value.
- The replay harness passes exactly the native objects (`SimpleNamespace(local_rank=0, batch_size=32)`,
  `limn.py:178`), so nothing in the ranking is reimplemented.

### C6. Metric parsing — R@10/R@50 extraction vs actual LIMN/CSMCIR output format

**Verdict: VERIFIED for both pinned formats in the slice (P2 for the shape/robustness of the LIMN
discovery path).**

I reconstructed each evaluator's native stdout from source and ran the workbench parser on it:

- **LIMN** (`replay/limn.py:226-235` prints `json.dumps(output, indent=2)`): parser discovery is a
  `line.startswith("{") and line.endswith("}")` scan (`metrics_extraction.py:76-85`) — which can never
  match a pretty-printed object — followed by `re.search(r'\{\s*"model_id":\s*"limn".*\}',
  stdout_text, re.DOTALL)` (`metrics_extraction.py:87-93`). Observed:

  ```
  LIMN native stdout (json + trailing [OK]) -> EXTRACTED_VERIFIED {'r10': 55.9133, 'r50': 77.82, 'mean': 66.8667}
  without --output -> EXTRACTED_VERIFIED
  LIMN with preceding '{' line -> EVALUATION_COMPLETED_METRICS_UNPARSED
  ```

  So the greedy regex happens to work for the emitted order; a single earlier brace-bearing stdout line
  makes the extraction fail closed (a blocked run, not a wrong number). P2: replace the greedy regex
  with the same brace-scanning used for CSMCIR.
- **CSMCIR** (`CSMCIR/src/validate_blip_csmcir.py:630` prints `json.dumps(results_dict, indent=4)` between
  other prints): the parser's `\{[^{}]*"average_recall"[^{}]*\}` regex (`metrics_extraction.py:222-236`)
  matches the flat object only because indistinguishable-key comments explain the `[^{}]` class; the
  visualisation block below the print adds no braces. Observed:

  ```
  CSMCIR pretty stdout -> EXTRACTED_VERIFIED {'r10': 57.0967, 'r50': 77.1833, 'mean': 67.14} PARITY_VERIFIED
  ```

  The category keys are read as `{cat}_recall_at10/at50` (`:255-262`), matching
  `validate_blip_csmcir.py:622-623`. **Note (P3):** `results_dict` has no `mean` key; the parser's
  `observed["mean"]` is `average_recall` (`:267`), which is the *mean of the two macro recalls* —
  arithmetically identical to `(r10+r50)/2`, so no numeric error, but the mapping is incidental.
- Adjacent format verified as a byproduct: **CLIP4Cir** prints f-string assignments for FashionIQ
  (`CLIP4Cir/src/validate.py:344-354`), and the `clip4cir_stdout_recalls` parser reads them
  (`metrics_extraction.py:309-380`). Observed on the native text:
  ```
  combiner -> EXTRACTED_VERIFIED {'r10': 43.7667, 'r50': 66.9333, 'mean': 55.35} {...per category...}
  ```
  Model → parser routing is `metrics_extraction.py:403-412`; the scale convention is declared, not
  inferred (`paper_gate.py:155-176`), and the only production caller uses the default percentage scale
  (`evaluate_models.py:127-190` never passes `metric_scale`), so `SCALE_FRACTION` is test-only.
- **`_check_parity` contradiction (P2, CONFIRMED).** `metrics_extraction._check_parity` still runs a
  ±0.05 **absolute** tolerance (`metrics_extraction.py:28-61`) while the canonical gate uses 5 % **relative**
  (`paper_gate.py:177-228`). Observed divergence on the same numbers:

  ```
  legacy _check_parity: PARITY_VERIFIED
  canonical gate r10: 1.6295777115822667 within_tolerance= True
  ```
  Both are visible in the same report (`evaluate_models.py:1194,1200`: `parity_status`,
  `benchmark_gate`) and both are surfaced in the UI (`App.tsx:60`, `gate.ts`). For a published value
  near 57 the two point opposite ways (`PARITY_MISMATCH` vs `within tolerance`) whenever the gap is
  between 1 % and 5 % relative. Its docstring claims it "can never contradict the gate" — that claim is
  FALSE as written.

### C7. Partial-result handling — 2/3 categories: BLOCKED or silently averaged?

**Verdict: VERIFIED for the run path (blocked, never averaged); VARIANT for the gate (see C2) and P1 for
the fact that only two models have any parser at all.**

- LIMN: no macro is emitted for a partial set (`common.py:92-95`), and even if a partial payload were
  assembled by hand the parser refuses it (`metrics_extraction.py:132-141`). Observed:
  `2-cat pretty-printed JSON: EVALUATION_COMPLETED_METRICS_UNPARSED`. With no `observed_metrics` the gate
  reaches `FAILED_OUTPUT_VALIDATION` (`paper_gate.py:320-323`) or, when only the category rows survive,
  `INCOMPLETE_METRIC_EVIDENCE` (`:404-407`). Observed:
  `LIMN aggregate=None 1-cat: INCOMPLETE_METRIC_EVIDENCE False`.
- CSMCIR/SPRC: `parse_csmcir_output` requires all three category key pairs and rejects on any missing
  (`metrics_extraction.py:255-262`) → `EVALUATION_COMPLETED_METRICS_UNPARSED`.
- **Everything else has no parser** (`metrics_extraction.py:414-423` → `AGGREGATE_PARSER_UNAVAILABLE`),
  and produces a `mean` averaged over whatever was extracted (`parse_combiner_output` `metrics_extraction.py:364-367` sums
  three categories from native output, so this one is safe). The general risk statement stands:
  a model with a parser but a partial native report cannot be averaged silently.

### C8. Synthetic evidence separation — can mock data enter the real report?

**Verdict: VARIANT — one path is real and unreachable in production; one path is a CONFIRMED
mis-certification route for a registry-declared synthetic model.**

- The `data_kind` guard is real but is supplied as a hard-coded literal:
  `evaluate_models.py:181` passes `data_kind="experiment"` unconditionally, and `real_execution=True`
  (`:186`). `paper_gate.py:311-315` therefore cannot ever see `"mock"` from this caller; no production
  caller passes anything else (grep: `data_kind` appears in `paper_gate.py`, `schemas/results.py`,
  `storage.py`, `main.py`, `evaluate_models.py:181` only). The gate's mock rejection is exercised only by
  tests (`test_paper_gate.py:220`). That is by construction ("the report is only reached after an
  invocation was made"), but it means the *only* thing separating mock from experiment on this path is
  the registry's `synthetic_source` flag, not `data_kind`.
- **`synthetic_source` does not neutralise the verdict (P1, CONFIRMED).** `evaluate_models.py:1426-1437`
  marks such a run `extraction_status="EXTRACTED_SYNTHETIC"` and records `is_synthetic: True`
  (`evaluate_models.py:1474`), but the gate is still called with `data_kind="experiment"` and `real_execution=True`.
  Reproduced in-process with a plan carrying `synthetic_source=True`, a synthetic extraction, and
  paper-exact metrics against csmcir's committed row:

  ```
  {"status": "PASS_WITHIN_5_PERCENT", "passed": true, "variant": false}
  ```

  The JSON that wrote those numbers was produced by the *synthetic* fixture evaluator. The only
  registry-level separation is `synthetic_source` (declared on **no** model in `models.yaml`, and gated
  behind `WORKBENCH_ALLOW_SYNTHETIC_ADAPTERS=1`, `adapters/models.py:157-190`), so today this is latent;
  but a registry entry that sets `synthetic_source: true` plus a paper row becomes a `passed=True`
  verdict. The UI does show a `SYNTHETIC SOURCE` badge for aggregate-only rows (`App.tsx:37`), which
  mitigates presentation, not the verdict. Recommended action: pass
  `data_kind="mock" if plan.synthetic_source else "experiment"` (or a distinct `data_kind`), so a
  synthetic evaluator can never mint a pass.
- The per-query separation *is* enforced where it applies: `validate_analysis_selection`
  (`storage.py:153-155`) refuses a mixed-`data_kind` selection, and `load_runs` rejects duplicate
  `run_id` across files (`index.py:56-60`).

### C9. Paper comparison provenance

**Verdict: VERIFIED for the run record; VARIANT for the registry's per-reference provenance.**

- Every verdict carries `reference_evidence` with `reference_id`, `paper_id`, `paper_title`,
  `registry_version`, `registry_file`, `registry_digest`, `reference_status`, the reported and derived
  metrics, per-metric provenance (`PAPER_REPORTED` vs `WORKBENCH_DERIVED`, with the derivation formula
  recorded in the grid — e.g. `dcnet_fashioniq_val_table1`'s `mean(28.95, 23.95, 30.44)`), the category
  table, the checkpoint noise condition and the gallery protocol
  (`benchmark_references.py:82-103`). Reference selection records its reason
  (`evaluate_models.py:192-195`, observed `"exact condition match"`).
- Immutability of a judgement: the reference is selected once and its values plus the **digest of the
  whole registry file** are copied into the verdict (`benchmark_references.py:82-103`), so editing
  `papers.yaml` later cannot retroactively change an existing verdict — the recorded
  `registry_digest` (observed `96986e864905a0e3…` for `extract-79c32139b606`) becomes stale and
  detectable.
- P2: the *per-reference* provenance is a free-text `verification_method` string inside the registry
  (28 verified rows checked; every one carries one, e.g. `"pdftotext -layout of arXiv 2305.09979v2 page
  8"`), and nothing in the runtime validates it, links it to a digest, or records it in the verdict.
  A row's `source_digest`/`evidence` fields do not exist (observed: both `None`). The verification
  artefacts live under `registry/paper_benchmarks/verification/` and `…/extracted/` but are not bound
  to the rows by hash. Recommended action: add `source_digest` (per verification artefact) and surface
  it in `as_evidence`.

### C10. Report immutability — append-only? can a later run overwrite?

**Verdict: VERIFIED for the run scope; VARIANT at two boundaries (a non-immutable ledger key location
and an unauthenticated API surface).**

- Run reports and completion proofs are exclusive-create: `run_report.open("x")`
  (`evaluate_models.py:1494-1498`, raising on `FileExistsError`), `proof_path.open("x")`
  (`:1566-1570`), with a pre-check that refuses an existing pair for the same run id (`:1349-1351`).
  The log directory is claimed atomically (`:517-527`) and never reused. `latest_attempt` /
  `latest_successful` are separate pointers and a failed rerun cannot advance `latest_successful`
  (`:1573-1600`). The evidence ledger is HMAC-chained and append-only with an fsync per entry
  (`:1258-1279`) and duplicate-run-id refusal (`:1263-1265`).
- The replay entrypoint is also create-only (`replay/limn.py:237` `open("x")`).
- **P2, ledger key.** `ledger_hmac_key` (`:1102-1188`) prefers a key at `<repo_root>/.workbench_ledger_key`
  if that legacy file exists, and the repository root is forwarded to the evaluator child
  (`WORKBENCH_*` env allowlist, `:459-487`). An actor able to write inside the repo — the same actor the
  ledger is meant to be independent of — can therefore read (and, if it also rewrote the `.jsonl`,
  re-sign) the chain. The private `$XDG_CONFIG_HOME/cir/workbench_ledger_key` fallback is the good state;
  the legacy path is a documented migration convenience with a real ceiling. Also: an append-only
  `jsonl` is not tamper-evident against truncation of a *suffix* by an attacker holding the key — inside
  this threat model that is acceptable, but it should be stated.
- **P2, API surface.** `/api/benchmark-runs` and `/api/runs/gate` (`main.py:189-199`) serve every
  `*_aggregate.json` in the reports root that merely parses and carries four string fields plus an
  object `benchmark_gate` (`_aggregate_report_is_well_formed`, `main.py:143-155`; scan `:112-141`).
  They never consult the
  completion proof, the ledger, `evaluate_models.validate_run_manifest`, or the report digest.
  Reproduced in-process: a hand-written report with no proof, no ledger entry and no run directory is
  served as a gate-bearing run —

  ```
  accepted by /api/benchmark-runs filter: ['forged'] | gate passed: True
  ```

  `completion_validator` (`:1629-1752`) *is* wired into the autopilot (`autopilot.py:747-752`), so the
  canonical consumer verifies; the API is the unverified surface. Since the UI's warning labels for
  aggregate-only runs come from exactly this endpoint (`App.tsx:37`), a dropped-in file renders as a
  labelled pass. Recommended action: have the API verify the proof/ledger (or serve only reports whose
  proof validates) before listing them.

### C11. Wrong-protocol detection — `protocol_id` vs benchmark reference key mismatch refused

**Verdict: VERIFIED.**

- Selection is by `gallery_protocol` (`benchmark_references.py:139-190`): a row whose `gallery_protocol`
  differs is not a candidate, and the caller receives a named refusal. Observed:

  ```
  select_reference csmcir: csmcir_fiq_val_ours_vitg exact condition match
  wrong protocol: None no verified reference for csmcir under training noise 0%, protocol fashioniq_val_split; committed rows are ['csmcir_fiq_val_ours_vitg']
  ```
  End-to-end at the gate's own entry point:
  ```
  wrong-protocol run -> METRIC_REFERENCE_UNVERIFIED | reason: no verified reference for csmcir under training noise 0%, protocol fashioniq_val_split; ...
  third-protocol run -> METRIC_REFERENCE_UNVERIFIED | reason: ... protocol fashioniq_full_gallery_ref_excluded; ...
  ```
- The gate independently re-checks the condition with the reference's own `_reference_expected`
  (`paper_gate.py:237-260`) against the run record, producing `BLOCKED_PROTOCOL_MISMATCH`
  (`paper_gate.py:280-292`), and refuses a verified row that states no condition at all
  (`BLOCKED_REFERENCE_INCOMPLETE`, `paper_gate.py:354-357`).
- Ambiguity fails closed: `ptha_mtst` has two verified rows with identical stated conditions and
  selection refuses (`observed: more than one verified reference matches this run: [...]`).
- P2: the registry declares every protocol's `evaluation_noise_pct: 0` (`models.yaml:8-19`) and the run
  side reads the protocol definition (`evaluate_models.py:147`); no code path can ask for a noisy
  *evaluation* yet, so the noisy-reference rows (20/50/80 %) can only be selected by the checkpoint
  condition. That is correct for the papers (their noise is training-time), but it means a future noisy
  evaluation protocol would be silently compared against a training-noise row.

### C12. Checkpoint policy vs paper reference identity — extra facet

**Verdict: CONFIRMED mismatch (P1) for `conesep`; the other noisy models are consistent.**

Observed, cross-checking every committed reference's `checkpoint_id` against the registry variant ids:

```
MISMATCH conesep_fiq_n20  ref.checkpoint_id='fiq_n20' noise=20 registry={'fiq_n02':20,'fiq_n05':50,'fiq_n08':80}
MISMATCH conesep_fiq_n50  ref.checkpoint_id='fiq_n50' noise=50 registry={'fiq_n02':20,'fiq_n05':50,'fiq_n08':80}
MISMATCH conesep_fiq_n80  ref.checkpoint_id='fiq_n80' noise=80 registry={'fiq_n02':20,'fiq_n05':50,'fiq_n08':80}
MISMATCH limn_fashioniq_base_iter0_table1 ref.checkpoint_id='base_iter0_all_categories' noise=None registry={…three category ids…}
```

`airknow`, `habit`, `intent` use their registry variant ids and pair correctly. The consequence for
`conesep`: `include_flagged_conditions` selects `fiq_n02`/`fiq_n05`/`fiq_n08`
(`checkpoint_selection.py:148-160`), and the gate then compares `checkpoint_id` as a comparability
field against the reference's `fiq_n20`/`fiq_n50`/`fiq_n80`, so every conesep run is
`BLOCKED_CHECKPOINT_MISMATCH` — observed:

```
selected: conesep_fiq_n20 | reason: exact condition match
STATUS: BLOCKED_CHECKPOINT_MISMATCH passed: False
   blocker: checkpoint_id mismatch: reference='fiq_n20' run='fiq_n02'
```

Fail-closed, so no wrong number is produced (severity P1, blocking the conesep user journey, not
mis-certifying). Note the reference *id* was named after the noise level, and the registry id after a
different indexing; one of the two must change, ideally with a mismatch assertion in
`load_registry` (`registry.py:83-109`), which today only checks bundle membership.

### C13. Author-hash availability and what it protects

**Verdict: VARIANT (P2) — 14 of 25 variants have no author hash; the download path asks the right
question but cannot enforce what is not declared, and the gate never sees it.**

Observed audit of `models.yaml`:

```
airknow fiq_n05/fiq_n08   NO AUTHOR HASH   (status NO_CLEAN_CHECKPOINT, download_url present)
conesep fiq_n02/n05/n08   NO AUTHOR HASH
encoder fashioniq         NO AUTHOR HASH   (mapping UNVERIFIED)
pair pair_b1/b2           NO AUTHOR HASH   (mapping UNVERIFIED)
clvc_net/dcnet/combiner/clip4cir/tgcir/sprc  NO AUTHOR HASH  (mapping UNVERIFIED)
csmcir/habit(×3)/intent(×3)/hint/limn(×3)   hash present  → 11 of 25
```

`download_checkpoints.py:241-256` enforces `expected_size_bytes` and `expected_sha256` **when
declared**, and `registry.checkpoint_missing_paths` (`registry.py:273-301`) treats "no expected hash" as
"present if non-empty". `checkpoint_availability` (`registry.py:714-745`) reports
`official_sha256_known: False`, `official_sha256_match: None`, but `runnable` does not depend on it
(`airknow` observed `runnable=False` only because its command/entrypoint is blocked). `replay/common.py`
is the only place that *refuses* to load an unhashable whole-model pickle
(`load_trusted_whole_model`, `common.py:36-48`, raising for `expected_sha256 is None`) — the correct
stance, but it applies to LIMN alone. Recommended action: make `validate_request`
(`adapters/base.py:27-33`) refuse a whole-model-pickle checkpoint with no author hash, and surface
`official_sha256_known` in the gate's run_meta (see C1).

### C14. Per-query results, ranking semantics and failure analytics (extra scope facet)

**Verdict: NOT_RUN — the schema and the analysis are complete and self-consistent; no production
producer exists, so every Research-tab number today rests on `tests/mock_data.py` rows.**

- `ResultRun`/`QueryResult` are strict (`schemas/results.py:28-108` with `extra="forbid"`, canonical
  `query_id = category:annotation_index:reference_id:target_id`, contiguous ranks, unique image ids,
  consistency between `target_rank` and the saved top-k). `analysis.py` requires aligned query universes,
  refuses cross-protocol selections (`storage.py:150-152`), and computes statistics with plain
  statistics functions (`analysis.py:51-93`).
- The only production writer of `schema_version: 2` is absent: `load_mock_results.py:24` writes each row to
  `<results_root>/<protocol>/mock/<run_id>.json` with `data_kind: "mock"` (`tests/mock_data.py:26`), and the report path explicitly sets
  `per_query_export_available: False` and `index_available: False` for aggregate-only runs
  (`evaluate_models.py:1455`, `autopilot.py:989-991`); `result_files()` skips
  `artifact_type == "official_aggregate_evaluation_report"` (`index.py:45-48`). Observed:
  `find workbench/artifacts/results -name '*.json' | grep -v mock` → empty; every UI Research-tab metric
  in a real run is `null` (`cohort_metrics` returns `None`s for empty query sets, `storage.py:192-204`).
  This is honestly labelled (`aggregate_only`, `RESULT_SCHEMA.md:54`) and is therefore a completeness
  gap (documented prerequisite: "a future reviewable observation-only export"), not a false claim.
- Ranking semantics: the canonical R@k derivation is `rank <= k` (`storage.py:203-204`) and the failure
  boundary is `rank > k` (`analysis.py:62,79`) — mutually consistent with the native evaluators'
  0-based `nns[:k]` prefix counts (C5). No off-by-one found.

### C15. LIMN `docs/reviews/models/limn.md` hypothesis check

**Verdict: VARIANT — the document's central scientific claims re-check cleanly; two of its statements
are stale.**

- Re-verified positive: `test.py:88-97` per-category R@1/R@10/R@50 with the query-count denominator
  (`test.py:94`); no cross-category macro in the pinned tree (grep over `LIMN/*.py`); the reference
  image masked before `argsort` (`test.py:80-85`); whole-model `torch.save` site and the three
  checkpoint filenames; the replay imports the pinned modules and enforces `__file__` identity
  (`limn.py:157-172`); the macro is workbench-authored only (`common.py:90-108`).
- Stale: (a) the document's `metrics_extraction.py:121-186` / `common.py:90-100` line citations no longer
  match the current tree (`parse_limn_output` is `metrics_extraction.py:74-198`; `macro_aggregate` is
  `common.py:90-108`) — cosmetic, but it means the document was not re-run against `e9cafaf`;
  (b) it states the parity check is "structurally `METRIC_REFERENCE_UNVERIFIED` (paper_gate.py:177-179)
  because the registry records no paper metric (`models.yaml:854-856` all null)". Observed today:
  `models.yaml` for LIMN records no `reported_*`, but the **gate consults
  `registry/paper_benchmarks/papers.yaml`, not the registry transcription**, and
  `limn_fashioniq_base_iter0_table1` is VERIFIED with paper-exact category metrics — so a LIMN run with
  paper-exact metrics now returns `PASS_WITHIN_5_PERCENT`, `passed=True`
  (observed: `LIMN exact paper metrics -> PASS_WITHIN_5_PERCENT passed= True`, blocker list empty). The
  document's conclusion "no evaluation or paper comparison has ever been executed" remains true for the
  *runtime*, but its claim about what the gate would say is out of date.
- Its Q40 ("a partial-category run cannot emit a macro") is verified by me independently (C2/C7).

---

## 2. Execution trace (what was actually run)

All commands read-only. `WORKBENCH_NO_ENV_CACHE=1` on every Python invocation. Scratch under `/tmp/cslice`.

1. `git rev-parse HEAD`, `git status --porcelain`, `git -C workbench/third_party/LIMN rev-parse HEAD` —
   HEAD `e9cafaf…`, LIMN checkout at the registry pin, only untracked `__pycache__`.
2. `md5sum` over the audited files (hashes at the top of this report).
3. `read` of: `scripts/evaluate_models.py` (full, in 5 ranges), `backend/paper_gate.py` (full),
   `backend/benchmark_references.py` (full), `backend/checkpoint_selection.py` (full),
   `backend/metrics_extraction.py` (full), `backend/adapters/base.py`, `backend/adapters/models.py`,
   `replay/limn.py`, `replay/common.py`, `replay/tgcir.py`, `backend/registry.py` (ranges),
   `backend/index.py`, `backend/storage.py`, `backend/schemas/results.py`, `backend/analysis.py`,
   `backend/main.py`, `backend/autopilot.py` (ranges), `backend/runtime.py` (ranges),
   `backend/fashioniq_layout.py`, `frontend/gate.ts`, `frontend/App.tsx` (grep), `registry/models.yaml`
   (dump of all variants/bundles/environments), `registry/paper_benchmarks/papers.yaml` (dump of all 33
   rows (28 verified / 5 unverified)), `registry/auxiliary_assets.yaml` (dump).
4. Upstream sources read at their pins: `LIMN/LIMN/test.py`, `LIMN/LIMN/datasets.py`,
   `CSMCIR/src/validate_blip_csmcir.py`, `CSMCIR/src/data_utils_csmcir.py`, `CSMCIR/src/utils_csmcir.py`,
   `CSMCIR/src/blip_validate.py`, `CLIP4Cir/src/validate.py`, `SPRC/src/blip_validate.py`,
   `TGCIR/test.py`, `HABIT/test.py`, `INTENT/test.py`, `AirKnow/test_BLIP2.py`, `CLVCNet/test.py`,
   `PAIR/test.py`, `HINT/test.py`, `ENCODER/evaluate_model.py`, `ENCODER/test.py`, `DCNet/test.py`.
5. In-process reproductions (pure Python, the outputs quoted inline above): `paper_gate.evaluate_gate`
   on 2/3, 1/3, 0/3 and deviating-category coverage; `benchmark_references.select_reference` for
   protocol/condition/ambiguity cases; `_benchmark_gate_verdict` end-to-end for a wrong-protocol run and
   for a `synthetic_source` plan; `parse_limn_output` on the real emitted shape (3-cat, 2-cat,
   trailing-line, preceding-brace cases); `parse_csmcir_output` on the real pretty-printed object;
   `parse_combiner_output` on the real f-string text; `validate_run_manifest` accept/reject on a bundle
   with a tampered non-anchor member; `_recorded_report_manifest`/`manifest_digest`/`_plan_members`;
   `checkpoint_availability` for airknow; a replica of `main._aggregate_reports`'s filter on a forged
   report; `checkpoint_selection.select_checkpoints` and registry dumps for every model; `replay.limn.candidates`
   via `evaluate_models.candidates` for LIMN bundle/member/single selection; `LIMNAdapter.command` argv.
6. Explicitly NOT run: pytest, docker, any evaluator, any GPU probe, any network fetch (so every
   registry claim about download URLs, Hub LFS oids and byte-level checkpoint identity is **UNKNOWN**
   here, not PASS).

---

## 3. Findings summary (severity ordered)

| # | Severity | Class | Finding | Evidence |
|---|---|---|---|---|
| C-1 | P3 | FALSE_POSITIVE | CSMCIR caption read paths match the staging destinations (dir symlink at `prepare_dataset.py:65`); the initial missing-file/keyerror hypothesis is withdrawn, only the documented UNVERIFIED alias caveat remains | C4 |
| C-2 | P1 | CONFIRMED | `synthetic_source` runs are still gated with `data_kind="experiment"`/`real_execution=True`; a synthetic evaluator's declared JSON can mint `PASS_WITHIN_5_PERCENT passed=true` (`evaluate_models.py:1426-1437,181`) | observed JSON |
| C-3 | P1 | CONFIRMED | `conesep` reference rows use `fiq_n20/n50/n80` while the registry variants are `fiq_n02/05/08`; every conesep run is `BLOCKED_CHECKPOINT_MISMATCH` | observed |
| C-4 | P1 | CONFIRMED | `/api/benchmark-runs` + `/api/runs/gate` serve any parseable `*_aggregate.json` without proof/ledger verification; a forged report is served as `passed=True` | observed |
| C-5 | P1 | CONFIRMED | Gate's completeness floor comes from the paper row, not the run: a reference without `category_metrics` lets a 1/3- or 0/3-category run pass (`paper_gate.py:372-386,396-402,422-425,437`) | observed 1/3, 0/3 |
| C-6 | P2 | CONFIRMED | `_check_parity` (±0.05 absolute) contradicts the canonical gate (5 % relative) and both are displayed; its "can never contradict the gate" docstring is false | observed |
| C-7 | P2 | CONFIRMED | 14/25 registry variants carry no author hash; no report or verdict field discloses that, so a `PASS` does not imply byte-verified checkpoint identity | observed dump |
| C-8 | P2 | CONFIRMED | Legacy repo-root ledger key is preferred when present, and the repo root is exported to the evaluator child (`evaluate_models.py:1102-1188,459-487`) | source |
| C-9 | P2 | CONFIRMED | `_bundle_members_live` (`:770-813`) is dead; the non-anchor digest fallback it implements is never exercised | grep |
| C-10 | P2 | CONFIRMED | LIMN parser discovery can never match the pretty-printed object (line scan) and depends on a greedy `.*` regex; one earlier brace-bearing line → UNPARSED | observed |
| C-11 | P2 | CONFIRMED | Per-reference paper provenance is unvalidated free text; no `source_digest` per row, not surfaced in the verdict | observed dump |
| C-12 | P2 | CONFIRMED | Single-category LIMN diagnostic writes an unbounded-location JSON with the same report vocabulary (`replay/limn.py:210,231-245`) | source |
| C-13 | P2 | CONFIRMED | `parse_csmcir_output` maps `observed["mean"]` to `average_recall` (numerically equal, but the key is not a `mean`) | source |
| C-14 | P2 | CONFIRMED | LIMN `docs/reviews/models/limn.md` line citations are stale and its `METRIC_REFERENCE_UNVERIFIED` claim is superseded by the committed reference registry | observed |
| C-15 | P2 | CONFIRMED | Every registered protocol pins `evaluation_noise_pct: 0`, so a future noisy evaluation mode would match a training-noise reference row | source |
| C-16 | P2 | UNVERIFIED | LIMN `test.py` target-feature indexing assumes every target is reachable from the query union; unreachable with a consistent val split | C4 |
| C-18 | P2 | UNVERIFIED | CSMCIR index/query DataLoaders run `num_workers=2` with `collate_fn` filtering `None`; any `None` item could couple per-worker RNG and reorder `index_names` positionally (no data inspected) | C4 |
| C-17 | P3 | CONFIRMED | No production schema-v2 per-query producer; Research-tab analytics are fixture-only and honestly labelled | C14 |

Checked and clean (severity NONE): every registered adapters' `script` exists in its pinned checkout and
is the right file (`synthetic_model` entry has no `source_dir`, so `blocked` is correct); `LIMNAdapter`
argv equals the harness signature; per-category `DataLoader`s use `shuffle=False`; the declared
FashionIQ layouts contain no `train` file for a standard/val-split model; every verified committed
reference row (28/28) carries per-category metrics and all six comparability fields (no committed
`PASS_WITH_UNCHECKED_CATEGORIES` today); `ptha_mtst` ambiguity fails closed; rank/`target_rank`
semantics agree between the native evaluators, the schema and the failure analytics; no rollup code
anywhere reads `PASS_*` or `passed` as an acceptance decision (grep: `paper_gate.py:146` is the only
occurrence) — gate verdicts are display/audit labels only on a completed run, which is what makes C-2
and C-5 "evidence that is wrong" rather than "a pipeline that is broken".

---

## 4. Final verdict

**VARIANT.** The scientific core of this slice is honest and mostly fail-closed: reference selection by
condition, protocol-mismatch refusal, ambiguity refusal, macro recomputation, partial-payload refusal,
exclusive-create artifacts, a chained evidence ledger, and a bundle identity check that really does
reject a tampered non-anchor member. But three defects can produce *wrong research conclusions* rather
than a blocked run — the `synthetic_source` route to a `passed=True` paper match, the gate's
paper-sourced completeness floor, plus two integrity surfaces (the unverified report API and the paper-row
missing-metric status that outranks a real category deviation in presentation). No defect found here
requires GPU inference to demonstrate; several (C-2, C-4, C-5) are reproducible in-process and
should be fixed before any run's output is treated as evidence.

Cross-checks that stayed **UNKNOWN** on purpose: byte-level equality of every staged checkpoint against
its author hash (no download, no fetch), and any runtime behaviour of an evaluator (no GPU). Those are
not PASSes; they are simply out of scope for a static, read-only audit.
