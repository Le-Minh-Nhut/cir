# EVIDENCE_INDEX — every source directory, what it proves, and what it does not

Companion to `INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md` and `GATE_MATRIX.md`.
All paths relative to the repo root `/home/heheboiz/data/cir`.

---

## 1. `workbench/docs/reviews/20261108T152852-audit-inventory/` — gate G0

| file | proves | does not prove |
|---|---|---|
| `REPOSITORY_INVENTORY.md` (48,909 B) | 16 unique upstream URLs across 17 `model_id`s; equal to the 16 dirs under `workbench/third_party/`; 29 declared checkpoint variants; 21 staged; per-file manifest re-hash for every vendored tree; every staged checkpoint `sha256` computed in that session; every `WORKBENCH_*_PYTHON` probed with `CUDA_VISIBLE_DEVICES=""` | whether any model *runs*; float equivalence; anything about `results/**` |
| `inventory.json` (76,822 B) | same data, machine-readable. Keys: `counts`, `sources`, `variants`, `repositories`, `environments`, `checkpoint_identities`, `patches`, `protocols`, `gaps`, `reconciliation` | — |
| `INVENTORY_GAPS.md` (9,112 B) | A1 interpreter unresolved for 8 of 17 models; A2 three artifacts with verified digest but unread internals; A3 four variants with neither digest nor bytes; B1 12/16 repos have no `.git`; B2 TME fork mismatch; B3 CSMCIR third state; B4 two non-model checkouts; C1 no training-run metadata in any checkpoint; C2 registry dirty vs HEAD; C3 GPU capability not re-probed | — |
| `_registry_dump.json` | the registry flattened, one object per `model_id` | — |
| `_ckpt_keys.txt` | per-checkpoint pickle structure from a stubbed unpickler | DCNet's legacy non-zip pickle (unreadable without executing it) |

Key numbers to cite from `inventory.json` → `counts`:
`unique_repositories_workbench_scope: 16`, `registered_model_variants: 17`,
`declared_checkpoint_variants: 29`, `staged_checkpoint_variants: 21`,
`absent_checkpoint_variants: 8`, `models_with_composition_adapter: 9`,
`models_with_native_launcher: 17`, `models_with_resolvable_interpreter: 9`,
`non_model_git_checkouts: 2`, `source_trees_on_disk_including_non_model: 18`.

---

## 2. `workbench/docs/reviews/20261108T082857-audit-pipeline/` — static trace + differential

| file | proves | does not prove |
|---|---|---|
| `PIPELINE_TRACE.md` (19,167 B) | per-family transform, tokenizer, pooling and scoring, with `file:line`; the cross-model table (resize geometry, target layout `mdk` vs `mkd`, feature width 256/512/768/8192, `strict=True` only for `ptha_mtst`) | anything on GPU (`NOT_RUN` throughout) |
| `UPSTREAM_PARITY.md` (15,679 B) | 28-row/22-row parity tables for HABIT and ENCODER with a per-row executed-vs-`STATIC_ONLY` verdict; 4 intentional divergences named | upstream *numbers*; two protocols agreeing |
| `DIFFERENTIAL_TESTS.md` (7,163 B) | axes A1–A5 with tolerances declared before measurement; `atol 1e-6` for batch, `atol 0.0` for repeat | model forward; cold/warm cache |
| `FINDINGS.md` (31,556 B) | F1 seed inert; F2 ENCODER stochastic-vs-deterministic; F3 gallery cache shape; F4 ENCODER mapping literal; F5 auxiliary weight presence-only; F6 LIMN comment; F7 per-category gallery sizes 3817/6346/5373; F8 BERT vocab unresolved; S1–S5; §6b three corrections the audit made to itself | again: no GPU |
| `ENCODER_LIMN_PREPROCESSING_VERDICT.md` (17,882 B) | upstream `stage=0`/`stage=1` wiring read verbatim; resolved open_clip transforms; measured pixel divergence 0.164/0.503/0.205 mean abs, max 3.83, same-seed identical | rank impact on the real 74,381 gallery (no cache survives) |
| `logs/differential_stages.json`, `logs/tie_rank_stability.json`, `logs/duplicate_embeddings.json`, `logs/dup_pixel_check.txt`, `logs/empty_text_tokens.txt`, `logs/encoder_empty_text_tokens.txt`, `logs/encoder_preprocess_train_vs_val.txt`, `logs/encoder_transform_divergence.txt`, `logs/stale_cache_demo.txt`, `logs/limn_checkpoint_norm_probe.txt`, `logs/open_clip_default_norm.txt` | the raw measurements behind every number above | — |

---

## 3. `workbench/docs/reviews/20261108T153045-audit-equivalence/` — gates G2/G3/G4

| file | proves | does not prove |
|---|---|---|
| `TOLERANCES.md` (4,847 B) | five comparison classes (A exact / B bit / C float-under-reordering / D aggregate / E timing) declared **before** measurement, with "if a later file disagrees, this one wins" | — |
| `EQUIVALENCE.md` (15,862 B) | gate table; 207/18,009 rank changes; 0 K crossings; 19 order-only; cache-identity defect; tie policy; verdict `PARTIALLY_VERIFIED` | forward features; cold/warm; `WORKBENCH_SCORE_THREADS` (explicitly not measured there) |
| `FINDINGS.md` (8,686 B) | F1 207 changes with qids and margins; F2 19 order-only with the 4-vs-15 tie split; F3 7 missing identity inputs; F4 851 duplicate groups; F5 ENCODER proxy impact; F6 query-axis NOT_RUN; F7 scope = one model only | — |
| `decode_habit.json` | 4 rows (workers/batch grid at batch 32), all `bit_identical: true`, one sha256 `5fc8002d…` | the other 12 grid cells `EQUIVALENCE.md` claims |
| `decode_habit_batch.json` | 6 further rows (batch 1/64/256 × workers 1/8), same sha256 | — |
| `decode_encoder.json` | 16 rows, full 4×4 grid, `bit_identical: true`, sha256 `4f6a14f9…` | — |
| `fastpath_vs_stored.json` (285,082 B) | anchor run `497c25c57fbeba885ecfb9d1`; anchor re-verification 12/12 per condition, 0 mismatches; per-condition changed sets | the shipped-path 6003×3 sweep |
| `divergence_detail.json` (29,604 B) | `order_only_divergences` 7+7+5 = 19 rows; `changed_rank_margins`; `fastpath_agreement.identical_changed_query_sets_and_new_ranks: true` | — |
| `cache_and_ties.json` (18,014 B) | 7 inputs absent from identity; sidecar holds only `dtype`/`fingerprint`/`shape`; 851 duplicate groups / 2,386 positions / largest 54; tie census per condition | cache content verification never executed |
| `gap_and_axes.json` (3,874 B) | stored vs `opt` vs `gemm1024` aggregates at full precision; `axes: {}` (empty — the axes were never filled) | — |
| `encoder_transform.json` | `preprocess_train` stochasticity: pairwise max abs 3.8269808292388916; same-seed 0.0; different seed 0.3572; cosine mean 0.9276, min 0.8261 | — |
| `encoder_rank_impact.json` | PROXY: 128-row mini-gallery, 48/64 ranks change, max \|Δrank\| 51, R@10 24→23 of 64 | the real gallery |
| `query_manifest.json` (569,505 B) | per-query identity for the 6,003-query cohort | — |

---

## 4. `workbench/docs/reviews/20261108T155252-protocol-audit/` — gate G1

| file | proves | does not prove |
|---|---|---|
| `PROTOCOL_MATRIX.md` (23,949 B) | per-repo gallery rule, reference policy, entrypoint, with quoted source and line; executed counts 3817/6346/5373/15536 and 2628/3089/2902/8619; 0 MISMATCH against the registry over 17 variants | reported paper numbers |
| `PROTOCOL_FINDINGS.md` (18,534 B) | 4 genuinely wrong citations (C1–C4), 3 block-head citations (C5–C7); 3 entrypoint misstatements (MTST_PTHA, SPRC CIRR-only `main()`; CLVC-Net `python test.py` inert); per-model protocol bound to the model not the run; 7 vendored files + 2 added files undeclared in `approved_source_patches` for MTST_PTHA/AirKnow/ENCODER/PAIR | — |
| `CLONE_MANIFEST.tsv` | 16 clones at full 40-char SHAs, all `CHECKOUT_OK`, all `git status --porcelain` empty | — |
| `REPRODUCE.md` (5,393 B) | exact reproduction recipe including our own `recall_at_k` vs `recall_at_k_reference_excluded` check on `[REF, WRONG, TARGET]` → rank 3 kept / rank 2 excluded | — |
| `raw/` | `clone.log`, `pins.tsv`, `cite_check.json`, `group{A,B,C}_quotes.txt`, `vendored_diffs.txt` (15 diff blocks), `provenance_local_fixes.txt` | — |
| `REGISTRY_CORRECTIONS.patch` | proposed, **not applied**: protocol section empty, only the two missing `approved_source_patches` declarations | — |

---

## 5. `workbench/docs/reviews/20261108T155805-audit-fixes/` — defect fixes

| file | proves | does not prove |
|---|---|---|
| `FIXES.md` (25,183 B) | F1 applied (seed removed from `identity`, `seed_applied: false` added) with the no-RNG-draws proof; F3 applied (`_memmap_metadata_shape`); F4 applied via patch; F6 applied via patch; F7 (missing imports) found and fixed; test counts 7 failed / 321→328→328→348 passed, **same 7 failing ids in all four runs** | numerical change of any fix (none: all four are provenance/namespace or load-path) |
| `PRE_F1_ARTIFACTS_AND_CACHE_VERDICT.md` (9,235 B) | 8 pre-F1 run dirs listed with sizes; caches cannot be reused under a post-F1 id (two independent keys: path and sidecar fingerprint); measured against the real 2.3 GB cache; pre-F1 detection via `seed_applied: null` | — |
| `MODEL_REPRODUCIBILITY_VERDICT.md` (7,290 B) | ENCODER and PTHA+MTST raised `NameError` in `_worker_main`; 9-adapter static audit; revert-proof 3 failed / 4 failed / 23 passed | — |
| `ENCODER_MAPPING_STATUS.patch`, `LIMN_OBSERVER_COMMENT.patch` | change records, both now applied | — |
| `baseline_subset.txt`, `final_subset*.txt`, `helpers_test.txt` | the four pytest runs | — |

---

## 6. `workbench/docs/reviews/20261108T-throughput/` — gate G6

| file | proves | does not prove |
|---|---|---|
| `THROUGHPUT.vi.md` (24,272 B) | batch saturation at ~84 img/s from batch 32; bf16 no gain; transform 27 img/s at 20 threads vs 650 at 1 thread; decode A/B 21.2 → 2089.2 img/s; query-axis 1/2/4/8 → 4.8/8.7/15.7/27.4 q/s, 5.67× at 8 threads; `optimize=True` and GEMM chunk rejected on rank identity | see traceability caveats in the main doc §8 |
| `batch.txt` | 7 lines: 2000-image slice, fp32 17.6–18.0 img/s at bs 64–512, bf16 16.2–17.7; peak alloc 3484→9544 MiB | — |
| `ein_threads.txt` | einsum timing grid at `OPENBLAS_THREADS` 1/2/4/8/10; `old` 4.89–5.76 s, `optimize=True` 0.94–1.45 s, `gemm_chunk4096` 0.32–1.03 s; maxdelta `3.576e-07` (and `2.384e-07`/`2.980e-07` at 10 threads) | — |
| `eq_fast.txt` | 400-query slice: rank_mismatch 0 / 3 / 11 = 14 of 1200 | full cohort |
| `eq_opt.txt` | 400-query slice, first two conditions only | — |
| `metric_gate.json`, `metric_gate_gemm_vs_optimize.json`, `metric_gate_gemm_vs_optimize.txt`, `metric_decision_optimize_true.txt` | full 6,003×3 metric comparison; `accept: true`; exact r10/r50/ndcg/seconds per condition | — |
| `query.txt` | query-stage timings, forward 81.0 q/s at b=32 and 79.1 at b=64 | the thread table |
| `vdecode.txt` | 512 images: serial 4.69 s (109.1 img/s), 2/4/8 workers 4.6×/6.2×/5.4×, `bitwise_equal=True`, `maxabs_delta=0.0` | the 21.2 img/s baseline |
| `cohort_full.txt` | the cohort driver log for the `habit` attempt: `NO PROGRESS for 900s` repeatedly, `TIMEOUT after 7200s`, `END habit exit=-15`, `NO_CANDIDATE_FILES` | — |

---

## 7. Run and UI state

| dir | proves | does not prove |
|---|---|---|
| `20261108T113610-eval/` — `RUNS.vi.md`, `MODEL_INVENTORY.md`, `MODEL_RUN_MATRIX.md`, `BLOCKERS.vi.md`, `ENVIRONMENTS.vi.md`, `circus_cohort.yaml` | cohort setup: profile path bug (8/9 `checkpoint not staged` under the shipped profile); GPU probes for habit/intent/hint/conesep at capability 12.0; the interpreter map; B1–B9 blockers; pre-existing artifacts `ad477…`/`6a528…`/`315dd…`/`10173…` with their metrics | any NEW_RUN completed in that session |
| `20261108T115509+0700-backend/` — `BACKEND_CHANGES.vi.md` | real candidate Top-K from the same `score()` matrix; four independent capabilities; `unscored` vs `retrieval_failure`; 5/5 old runs are `TOPK_EXPORT_NOT_REQUESTED` | per-query Top-K for any model without an observer |
| `20261108T140435+0700-native-runs/` — `NATIVE_RUNS.vi.md` | 3 CPU envs built at their exact pins; `combiner_rn50x4_noft` executed end-to-end exit 0 in ~81 min with R@10 35.1078 / R@50 58.4756; DCNet failed with `TypeError: 'weights_only'`; `/api/runs` structurally cannot show these 4 models | paper comparison (`METRIC_REFERENCE_UNVERIFIED`) |
| `20261108T042931Z/` — `UI_ACCEPTANCE.vi.md`, `QA_flow_g_textarea_probe.md`, `capture-log.txt`, `frontend-before-after.md`, `shots/` (33 PNG) | frontend build exit 0; 51 frontend tests; flows b/h pass at 3 viewports; 18/18 viewport×screen overflow clean; flow g's single red is a `innerText`-on-`<textarea>` harness defect, proven by `input_value` returning the note | backend behaviour |

---

## 8. Artifacts read directly (not from a review doc)

`workbench/artifacts/composition/summary/*/full.json`, 14 directories. Values quoted in the main doc
§2/§6 come from these files, not from a summary document:

| run dir | status | queries | MM R@10 | MM R@50 | `seed_applied` |
|---|---|---|---|---|---|
| `habit_fiq_n02_497c25c57fbeba885ecfb9d1` | COMPLETED | 6003 | 35.61552557054806 | 54.3728135932034 | `null` (pre-F1) |
| `intent_fiq_n02_37ee72aa298b276d1bfda0fa` | COMPLETED | 6003 | 34.39946693320007 | 54.00633016824921 | `null` (pre-F1) |
| `hint_fashioniq_4bc9893a977e501b82378e76` | COMPLETED | 6003 | 32.816924870897886 | 51.77411294352824 | `false` (post-F1) |
| `tme_fashioniq_0_ad477312aa57c1b9114f9e51` | COMPLETED | 6003 | 34.01632517074796 | 53.95635515575545 | `null` |
| `ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a` | COMPLETED | 6003 | 19.473596535065802 | 34.44944194569382 | `null` |
| `encoder_fashioniq_6a528e877c328772e2c7bb28` | COMPLETED | 6003 | 31.01782442112277 | 49.5752123938031 | `null` |
| `limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b` | **PARTIAL** | 6003 | 30.064581917063222 | 48.759347382732834 | `null` |
| `habit_fiq_n02_e0dd04a46ef7a58334b9e353` | **FAILED** | 6003 | — | — | `null` |
| `conesep_fiq_n02_30f6de423861cc1d6c78c361` | COMPLETED | 6003 | 35.48225887056472 | 54.13959686823255 | `false` (post-F1) |
| `encoder_fashioniq_eda30a37c0bc17ac43844f09` | COMPLETED | **8** (smoke) | 12.5 | 25.0 | `null` |
| `limn_base_iter0_all_categories_3f86f58483b368377e2c151e` | COMPLETED | **8** (smoke) | 25.0 | 37.5 | `null` |
| `released_audit_task7_mm_any_…` | COMPLETED | 6003 | 52.84024654339497 | 71.2976844910878 | `null` |
| `SYNTHETIC_encoder_fashioniq_fixture001`, `SYNTHETIC_habit_fiq_n02_fixture002` | COMPLETED | 6003 | fixtures | fixtures | `null` |

The two `SYNTHETIC_*` rows are test fixtures and the two `query_count: 8` rows are smoke-scale; both
classes are excluded from every metric claim, as `MODEL_INVENTORY.md` §7 requires.

---

## 9. What no evidence directory contains

- A **forward-feature** bit-identity measurement (G2-fwd) for any model. GPU was held by the `intent`
  cohort in every audit window.
- A **cold-vs-warm cache** comparison (G4d).
- A raw file for the **21.2 img/s** decode baseline or the **5.67×** query-axis table; both live only
  in `THROUGHPUT.vi.md` prose (see §8 of the main doc).
- Any measurement of **`WORKBENCH_SCORE_THREADS`** by the equivalence audit — it says so explicitly.
- `conesep_fiq_n02_30f6de423861cc1d6c78c361`, `intent_fiq_n02_37ee72aa298b276d1bfda0fa` and
  `hint_fashioniq_4bc9893a977e501b82378e76` in any audit document: all three exist on disk with full
  metrics, but none appears in `RUNS.vi.md`, `MODEL_RUN_MATRIX.md`, or any of the six audit
  directories. `intent` and `hint` are named there as QUEUED/NOT_RUN.
- Any **paper-number** reproduction for the 8 non-composition models, or for the 9 composition models
  against their published values.
- A definition of the gates **G1, G5, G6, G7** anywhere in the corpus.
