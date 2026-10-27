# CIRCUS composition evaluation spec

## Research question and interpretation

This opt-in evaluation asks whether specialized CIR models benefit from composing the reference image and modification text on the released CIRCUS subsets, compared across multimodal and two input-ablation conditions. Report any difference as **input-ablation dependence under this protocol**, not proof of intrinsic compositional reasoning. Specialized models were not trained on blank images or empty text, so a large gap may partly reflect distribution shift. Report multimodal (MM) retrieval quality next to every composition gap.

## Protocol and gallery

Primary protocol: `circus_fashioniq_release_v1`. Its gallery is the exact ordered list of 74,381 image identities in the pinned aggregated audit artifact. The paper's dataset paragraph states 18,853 images; the released artifact contains 74,381, matching M-BEIR's candidate pool. Stored ranks exceed 18,853: the maximum aggregated multimodal rank observed is 31,987 at sample index 4435. The discrepancy is therefore not an unused metadata field.

This workbench uses the released-artifact protocol. It does **not** claim exact reproduction of the paper's stated 18,853-image setting. No 18,853-image subset is invented, and the local native gallery is not silently substituted. The reference is eligible in ranking: the author retrieval code ranks without masking it. The native `-10e10` reference mask from ENCODER/LIMN val-split replays is deliberately not inherited.

## Conditions and upstream ablation naming

| Semantic condition | Input intervention |
|---|---|
| `multimodal` | Original reference image and edit text. |
| `text_only_black_image` | Replace reference with a solid black image of the same size **before** the model's own image transform; retain original text. |
| `image_only_empty_text` | Retain reference; remove edit text before tokenization. Ordinary tokenizer special tokens may remain. |

The upstream filenames are misleading. `ablation_configs/image_only_zero_image.json` contains `{"name": "img_black"}` but `retrieval/generate_retrieval_data.py` uses it for **text-only**. `ablation_configs/text_only_drop_text.json` contains `{"name": "text_drop_all"}` but is used for **image-only**. Names describe the removed input; this spec uses semantic mode names.

The black-image intervention follows `subset_evaluation/eval_subset.py::apply_ablation`: it replaces the RAW image via `_black_image_same_size`. A zero tensor after normalization or a zeroed feature vector is a different intervention and is not used here. Query text is passed as supplied; it is never re-split on `and`.

## Release data contract

The pinned author revision for these release inputs is `b7ed55104abd987e60f7099037e4cd8a69966428`.

| Set or artifact | Path / identity | Rows or count |
|---|---|---:|
| Full queries | M-BEIR `query/test/mbeir_fashioniq_task7_test.jsonl` | 6,003 |
| Shortcut-free (SF) | `shortcut_free_subsets/fashioniq/shortcut_free/mbeir_fashioniq_task7_test_multimodal.jsonl` | 4,069 |
| Validated solved | `final_dataset/query_jsonl/fashioniq/validated_solved/mbeir_fashioniq_task7_test_validated_solved.jsonl` | 368 |
| Validated unsolved | `final_dataset/query_jsonl/fashioniq/validated_unsolved/mbeir_fashioniq_task7_test_validated_unsolved.jsonl` | 218 |
| Validated union | Disjoint solved + unsolved query IDs | 586 |
| Gallery | Ordered `gallery_paths` from pinned aggregated audit artifact | 74,381 |
| Candidate pool | M-BEIR candidate-pool rows | 74,381 |

The validated parts have intersection 0 and `validated ⊆ sf ⊆ full` by query identity. `validated_unsolved` means unresolved by the ORIGINAL GME pool, not invalid; it remains in the cohort. The supplied `query_txt` is used unchanged.

These are locally computed SHA-256 values for the inspected bytes, not author-signed digests:

| File | SHA-256 |
|---|---|
| Aggregated audit | `27bdbef041200dc393b5478d3e57121bc16c70a9acc2e98a19fb48408a99533c` |
| SF | `95a93883334e351e4d3aaaf25e47e89b48d4daf02bdb786e8f00d58165ace37d` |
| Validated solved | `18f829da4d3b1ee8439f43998bae43dbbe8924c2eaafba84473ae849f564f5c7` |
| Validated unsolved | `ba019a30d3b45f4b29c34d2d48638dbeeed2e2a42409f625922c65de5b2a7efa` |
| M-BEIR query | `b601ef6a5411bf9385cff41e64bea863871ab1094cd09d78683de6a5141165ef` |
| M-BEIR candidate pool | `121ec284832c33289ee13bfac19d5d59dc3f8aeddf672d784fd5eb8bcdf79eef` |

## Local data availability

On this host, the released gallery names images `.jpg`, while the local FashionIQ root is flat and contains 74,381 `.png` files. All released gallery identities resolve by filename stem; zero need fetching. Query references (4,315 distinct basenames) and all 6,003 positive targets are present. Cached `data/fashionIQ_dataset/features/FashionIQ/val_images.pt` and `val_name_to_idx.json` are 1-byte / `{}` placeholders, not a reusable feature cache.

The garment-category resolver uses local annotations. It resolves 5,884 of 6,003 queries; 119 are `AMBIGUOUS` because a reference is shared across categories, so they remain unscoped rather than being assigned. There are 0 unresolved category identities. LIMN has separate whole-model checkpoints for `dress`, `shirt`, and `toptee`; it cannot score those 119 ambiguous-category queries. This is a coverage limitation, not a retrieval miss.

## Metrics and outcome labels

For each scored query, `r` is the 1-based exact target rank over the FULL eligible gallery:

```text
hit_at_10(q) = 1[r <= 10];  hit_at_50(q) = 1[r <= 50]
ndcg_full(q) = 1 / log2(r + 1)
reciprocal_rank(q) = 1 / r
MM = mean(ndcg_full) over the selected queries in multimodal mode; T and I likewise
composition_gap_ndcg = 1 - max(I, T) / MM      # ratio computed AFTER averaging the same cohort
```

The gap uses a common cohort with exact ranks in all three conditions. Compute the ratio after averaging, not by averaging per-query ratios. Retain negative gaps. Recall is **PERCENT**; nDCG and gap are **FRACTIONS**; never mix these units. An empty cohort, missing condition, or zero MM denominator produces an explicit unavailable value and reason, never 0, infinity, or NaN. An unscored query reduces coverage, is excluded from means, and is **not** a miss.

If reported, the reciprocal-rank difference is separately named `mrr_gap`. The paper's normalized nDCG gap is primary. A README-style Recall subtraction is not the paper's Composition Gap and must not be called `compgap`.

For K=10, report mutually exclusive outcomes: `text_only`, `image_only`, `both`, `composition_required` (MM succeeds and neither unimodal condition succeeds), and `unresolved` (all three fail). `unscored` is separate; a query missing any condition's exact rank is excluded from outcome bins, not counted as an unresolved retrieval miss. The current summary serializes this separate count under `neither_scored`.

## Checkpoint and training-condition labels

Every result carries the selected checkpoint and its actual training condition separately from evaluation annotation noise. Do not lump supervised FashionIQ checkpoints, noise-trained checkpoints, transfer checkpoints, and zero-shot embedding models together.

- HABIT, INTENT, and ConeSep retain selected `fiq_n02`: **training noise 20%**, evaluation annotation noise **0%**. ConeSep's registry notes only 20/50/80 noise variants exist.
- TME retains `fashioniq_0`, labelled **training noise 0%**. No 20% checkpoint exists.
- PTHA+MTST `cirr_transfer` is a **CIRR→FashionIQ transfer** checkpoint, not a FashionIQ reproduction.
- ENCODER's checkpoint-mapping uncertainty remains visible; successful composition execution resolves no unrelated mapping uncertainty.
- For models whose training noise is unstated in the existing status/registry, retain that label as unstated; do not infer 0% from evaluation noise.

## 17-model composition capability and status

“Supported” means the adapter declares the mode's input path; it does not imply full evaluation. Nine adapters are scaffolded; all remain `TARGET_GPU_PENDING` for full composition evaluation. ENCODER has a passing real-data GPU smoke on RTX 3050 (`sm_86`); no other composition GPU execution is recorded. “Selected checkpoint / training” preserves the existing registry/status scientific label. For rows without adapters, “not implemented” says nothing about checkpoint availability.

| Model id | Composition adapter | Adapter module path | MM | Text-only black image | Image-only empty text | Category scopes | Selected checkpoint and training condition | Composition status / concrete blocker |
|---|---|---|---|---|---|---|---|---|
| `csmcir` | yes | `workbench.composition.model_csmcir` | blocked | blocked | blocked | shared, category-agnostic | `fashioniq`; 0% training noise, eval noise 0% | `TARGET_GPU_PENDING`; gallery target features require per-image `COT_ours2` captions absent from released CIRCUS inputs. Source basis: `workbench/third_party/CSMCIR/src/utils_csmcir.py:95-103` and `workbench/third_party/CSMCIR/src/lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py:498-535`. Local `qwen_captions` are a different generator and are not substituted. |
| `airknow` | not implemented | — | — | — | — | not applicable (no adapter) | `fiq_n05` / `fiq_n08`; training noise 50% / 80%, eval noise 0%; requested 20% unavailable | Existing `CODE_TESTED` / `BLOCKED_CHECKPOINT_CONDITION`; only 50/80 variants exist, never substitute for requested 20%. |
| `conesep` | yes | `workbench.composition.model_conesep` | yes | yes | yes | shared, category-agnostic | `fiq_n02`; training noise 20%, eval annotation noise 0% | `TARGET_GPU_PENDING`; scaffolded, not yet exercised on a GPU; selected noisy condition still requires `include_flagged_conditions`. |
| `ptha_mtst` | yes | `workbench.composition.model_ptha_mtst` | yes | yes | yes | shared, category-agnostic | `cirr_transfer`; CIRR→FashionIQ cross-dataset transfer, not a FashionIQ reproduction | `TARGET_GPU_PENDING`; scaffolded, not yet exercised on a GPU; requires isolated Python 3.9 / torch 2.0.1 runtime and declared inputs. |
| `habit` | yes | `workbench.composition.model_habit` | yes | yes | yes | shared, category-agnostic | `fiq_n02`; training noise 20%, eval annotation noise 0% | `TARGET_GPU_PENDING`; scaffolded, not yet exercised on a GPU; selected noisy condition still requires `include_flagged_conditions`. |
| `intent` | yes | `workbench.composition.model_intent` | yes | yes | yes | shared, category-agnostic | `fiq_n02`; training noise 20%, eval annotation noise 0% | `TARGET_GPU_PENDING`; scaffolded, not yet exercised on a GPU; selected noisy condition still requires `include_flagged_conditions`. |
| `hint` | yes | `workbench.composition.model_hint` | yes | yes | yes | shared, category-agnostic | `fashioniq`; training noise unstated, eval noise 0% | `TARGET_GPU_PENDING`; scaffolded, not yet exercised on a GPU; existing `PUBLIC_ASSET_SETUP` blocker remains. |
| `encoder` | yes | `workbench.composition.model_encoder` | yes | yes | yes | shared, category-agnostic | `fashioniq`; training condition unknown, eval noise 0%; checkpoint mapping remains `UNVERIFIED` | Real-data GPU smoke passed on RTX 3050 (`sm_86`); full composition evaluation remains `TARGET_GPU_PENDING`; composition does not resolve checkpoint-mapping uncertainty. |
| `pair` | not implemented | — | — | — | — | not applicable (no adapter) | `pair_b1` / `pair_b2`; FashionIQ/training-condition mapping unverified | Existing `BLOCKED_SOURCE_ENTRYPOINT`; required pickle classes `model_vssp` / `MHTransformer_dis` are unpublished, and available checkpoint files do not establish FashionIQ mapping. |
| `clvc_net` | not implemented | — | — | — | — | not applicable (no adapter) | `author_drive_unresolved`; mapping unverified | Existing `BLOCKED_NO_SOURCE_REPLAY`; checkpoint mapping and source replay/entrypoint unresolved; ResNet-50 weights absent. |
| `dcnet` | not implemented | — | — | — | — | not applicable (no adapter) | `fashioniq_run_directory`; mapping unverified | Existing `AGGREGATE_PARSED`; run-directory mapping, validation-data preparation, GloVe validation PKLs, and ResNet-50 weights remain blocked. |
| `combiner_rn50x4_noft` | not implemented | — | — | — | — | not applicable (no adapter) | `rn50x4_noft_unresolved`; artifact mapping unverified | Existing `BLOCKED_CHECKPOINT_MAPPING`; author checkpoint mapping, RN50x4 base weights, and compatible data remain unresolved. |
| `clip4cir_rn50x4_fullft` | not implemented | — | — | — | — | not applicable (no adapter) | `rn50x4_fullft_unresolved`; paired Combiner/CLIP mapping unverified | Existing `BLOCKED_CHECKPOINT_BUNDLE`; author bundle pairing, RN50x4 base weights, and compatible data remain unresolved. |
| `tgcir` | not implemented | — | — | — | — | not applicable (no adapter) | `tgcir_archive_unresolved`; archive member/hash unverified | Existing `PREFLIGHT_IMPLEMENTED_BLOCKED`; archive mapping and evaluator entrypoint unresolved. |
| `sprc` | not implemented | — | — | — | — | not applicable (no adapter) | `sprc_fiq_unresolved` / separate `sprc_fiq_vitl`; both mappings unverified | Existing `BLOCKED_MODEL_MAPPING`; verify checkpoint/model mapping and assets; ViT-L is not the ViT-g paper-row variant. |
| `limn` | yes | `workbench.composition.model_limn` | yes | yes | yes | `dress`, `shirt`, `toptee` whole-model checkpoints | `base_iter0_all_categories`; training noise unstated, eval noise 0% | `TARGET_GPU_PENDING`; scaffolded, not yet exercised on a GPU; 119 ambiguous-category queries are out of scope and reduce coverage, not misses. |
| `tme` | yes | `workbench.composition.model_tme` | yes | yes | yes | shared, category-agnostic | `fashioniq_0`; training noise 0%, eval noise 0%; no 20% checkpoint exists | `TARGET_GPU_PENDING`; scaffolded, not yet exercised on a GPU; retain 0% checkpoint and do not substitute another model's noisy variant. |

The `csmcir` blocker is specifically the target-feature dependency, not the query-mode input wiring. The adapter reports all three conditions unavailable while those gallery target features cannot be produced. The seven other deferred entries retain their existing checkpoint/source/data reasons above; they have no composition adapter.

## Summary provenance and output isolation

Every summary must carry: protocol ID; gallery count and identity; subset and query count; model/checkpoint identity; condition set; preprocessing and ablation policy; adapter identity; category scopes; actual interpreter/runtime/GPU; seed; release artifact hashes; a synthetic-fixture label where applicable; and the paper-vs-release gallery note. Synthetic runs must remain labelled as fixtures. Results are written under `<artifacts>/composition/`; composition paths are checked to remain within that namespace and cannot overwrite native results.

The inspected runner summary records controller Python version, platform, and GPU identity, alongside the seed, hashes, adapter, category scopes, protocol/gallery/query identity, conditions, preprocessing/ablation policy, synthetic label, and paper-vs-release note. `UNVERIFIED: the current summary does not expose the selected model-interpreter path/version; record that runtime identity alongside each summary before interpreting an actual run.` No real composition result exists yet.

## PC command surface

The CLI exists at `workbench/scripts/composition_eval.py`; its parser defines `--config`, `--models`, `--subsets`, `--resume`, `--smoke [QUERY_LIMIT]`, `--plan`, `--prepare-data`, and `--seed`. These commands use parser-confirmed flags; no real composition evaluation has run:

```bash
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --plan
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --prepare-data
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder,limn --smoke
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models selected --subsets full,sf,validated --resume
```

Schedule one GPU-heavy model at a time; the runner serializes model runs. Smoke mode limits queries (default 1) and preserves the full released gallery. Released data root is configured by `WORKBENCH_CIRCUS_RELEASE_ROOT`.
