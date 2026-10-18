# Sixteen-model recovery — final state

Supersedes `VENDORED_RECOVERY_16_MODEL_STATUS.md`. This file states what is **in the tree**, what was
**executed**, and what remains — separately, because "implemented" and "ran" are different claims.

## 1. Verification vocabulary

Levels are cumulative and never merged. `GPU_TARGET_VERIFIED` requires an RTX 5070 Ti run; nothing here
has reached it, because no target GPU has executed any model.

`SOURCE_VERIFIED` → `IMPORT_VERIFIED` → `CHECKPOINT_LOADED` → `ONE_BATCH_FORWARD_VERIFIED` →
`NATIVE_EVALUATION_VERIFIED` → `METRIC_PARSER_VERIFIED` → `PER_QUERY_PARITY_VERIFIED` →
`GPU_TARGET_VERIFIED`

Blocker classes: `UNFINISHED_ENGINEERING` · `PUBLIC_ASSET_SETUP` · `ENVIRONMENT_RESOURCE_LIMIT` ·
`AUTHOR_ONLY_INFORMATION` · `SCIENTIFIC_VARIANT_NEEDS_LABEL`.

## 2. Matrix

| model_id | source_delivery | environment_setup | asset_acquisition | checkpoint_mapping | checkpoint_loading | native_wrapper | adapter | metric_parser | per_query_export | parity | GPU_validation | tests_executed | remaining_blocker | exact_next_command |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `limn` | vendored `LIMN` @`7d7bc9b` | declared (README) | 3 ckpts, Hub LFS sha256 | VERIFIED | not reached | `replay/limn.py` (existing) | real command | `limn_structured_json` | `limn_export.py` + observer + publish | aggregate parser verified | **PENDING** | `test_vendored_source_integrity`, `test_replay`, replay tests | `ENVIRONMENT_RESOURCE_LIMIT`: 5.4 GB ckpts unstaged, no torch here | `python workbench/replay/limn.py --checkpoint-root <dir> --dataset-root <FIQ> --source-root workbench/third_party/LIMN` |
| `encoder` | vendored `ENCODER` @`29a2a31` | declared | LAION ViT-B/32 registered | variant declared, file unverified | analysed, not reached | `replay/encoder.py` | real command | shared replay parser | not implemented | not reached | **PENDING** | `test_encoder_replay` (5) | `PUBLIC_ASSET_SETUP`: 1.5 GB backbone + ckpt unstaged | `python workbench/scripts/download_auxiliary_assets.py --model encoder` |
| `hint` | vendored `HINT` @`bec50b6` | declared | `fashioniq.pt` 2 724 451 182 B, sha256 `bf9c72d4…` | VERIFIED | not reached | `replay/hint.py` | real command | `hint_native_recall` | not implemented | parser parity verified | **PENDING** | `test_hint_replay` (4), `test_hint_metric_parser` (11) | `PUBLIC_ASSET_SETUP`: 2.7 GB ckpt unstaged | `python workbench/scripts/download_checkpoints.py --model hint` |
| `pair` | vendored `PAIR` @`bfb0b98` | declared | `pair-B1.pt`/`pair-B2.pt` published | **UNRESOLVED** — no dataset/category label | not reached | `replay/pair.py` | real command | shared replay parser | not implemented | not reached | **PENDING** | `test_pair_replay` (5) | `AUTHOR_ONLY_INFORMATION` + `SCIENTIFIC_VARIANT_NEEDS_LABEL`: `MHTransformer_rev.py` is a labelled EXPERIMENTAL_RECONSTRUCTION | author confirmation of B1/B2 mapping |
| `clvc_net` | vendored `CLVCNet` @`bd9b688` | declared | ResNet-50 ImageNet weights registered | archive interior unknown | not reached | `replay/clvc_net.py` | real command | shared replay parser | not implemented | not reached | **PENDING** | `test_clvc_net_replay` (4) | `PUBLIC_ASSET_SETUP`: `CLVCNET.zip` (6 whole-model files) unmapped | inspect `CLVCNET.zip` members |
| `tgcir` | vendored `TGCIR` @`65fa78e` | pin corrected in report | none published | archive mapping unknown | not reached | `replay/tgcir.py` (preflight) | real command | none | not implemented | not reached | **PENDING** | `test_tgcir_replay` (4) | `PUBLIC_ASSET_SETUP` + `KNOWN-WRONG_PIN` pending registry edit | apply the pinned OpenAI CLIP source, inspect the ZIP |
| `habit` | vendored `HABIT` @`6ddc2cd` | declared | 3 noisy variants, sha256 | VERIFIED (20/50/80 %) | not reached | `replay/habit.py` | real command | shared replay parser | not implemented | not reached | **PENDING** | `test_habit_replay` | `PUBLIC_ASSET_SETUP`: no clean ckpt published; noisy eval on clean val must stay labelled | `download_checkpoints.py --model habit` |
| `intent` | vendored `INTENT` @`2a75fc6` | declared | 3 noisy variants, sha256 | VERIFIED | not reached | `replay/intent.py` | real command | shared replay parser | not implemented | not reached | **PENDING** | `test_intent_replay` | `PUBLIC_ASSET_SETUP`: no clean ckpt | `download_checkpoints.py --model intent` |
| `conesep` | vendored `ConeSep` @`bd4bf62` (redirects to `Lee-zixu/ConeSep`) | declared | 3 noisy variants, sha256 | VERIFIED | not reached | `replay/conesep.py` | real command | shared replay parser | not implemented | not reached | **PENDING** | `test_conesep_replay` | `PUBLIC_ASSET_SETUP`: no clean ckpt | `download_checkpoints.py --model conesep` |
| `airknow` | vendored `AirKnow` @`6ab6fa9` | declared | 2 noisy variants, sha256 | VERIFIED | not reached | `replay/airknow.py` | real command | shared replay parser | not implemented | not reached | **PENDING** | `test_airknow_replay` | `SCIENTIFIC_VARIANT_NEEDS_LABEL`: native path calls `model.train()`; eval-mode alternative is a labelled variant | author decision on the eval-mode variant |
| `dcnet` | vendored `DCNet` @`68c79d3` | declared | `trained_model.pth` (Drive) | PENDING | not reached | `replay/dcnet.py` (preflight) | real command | none | not implemented | not reached | **PENDING** | `test_dcnet_replay` | `AUTHOR_ONLY_INFORMATION`: GloVe caches unpublished; deterministic reconstruction specified, not implemented | author caches, or opt into the reconstruction |
| `csmcir` | vendored `CSMCIR` @`774f94e` | declared | CoT captions published | VERIFIED | not reached | `src/validate_blip_csmcir.py` (has `__main__`) | real command | `csmcir_stdout_json` | not implemented | not reached | **PENDING** | `test_csmcir_source_blocker`, `test_vendored_approved_patch` (11) | `PUBLIC_ASSET_SETUP`: 2.7 GB ckpt + BLIP-2/EVA-ViT-g caches | `patch -p1 -d workbench/third_party/CSMCIR < workbench/patches/csmcir_models_init_compat.patch` then download |
| `combiner_rn50x4_noft` | vendored `CLIP4Cir` @`dfed9f7` | declared | RN50x4 base registered | part digests recorded, members unmapped | not reached | `src/validate.py` (has `__main__`) | real command | `combiner_stdout_recall` | not implemented | not reached | **PENDING** | `test_clip4cir_preflight` | `PUBLIC_ASSET_SETUP`: 8.6 GB archive not downloaded | reconstruct the 5-part archive, extract members |
| `clip4cir_rn50x4_fullft` | vendored `CLIP4Cir` @`dfed9f7` | declared | RN50x4 base registered | part digests recorded, both members required | not reached | `src/validate.py` | real command | `combiner_stdout_recall` | not implemented | not reached | **PENDING** | `test_clip4cir_preflight` | `PUBLIC_ASSET_SETUP`: fullft needs the matching fine-tuned CLIP **and** Combiner | same archive; verify pairing |
| `sprc` | vendored `SPRC` @`2935a53` | declared | ViT-L reachable; ViT-g OneDrive 403 | PENDING | not reached | `src/blip_validate.py` | real command (registered class + backbone) | `csmcir_stdout_json` | not implemented | not reached | **PENDING** | `test_sprc_preflight` | `SCIENTIFIC_VARIANT_NEEDS_LABEL`: only ViT-L is public; using it changes the backbone | author mirror for the ViT-g weights |
| `ptha_mtst` | vendored `MTST_PTHA` @`6a02ee4` | declared | **none published** | none | not reached | `replay/ptha_mtst.py` (preflight) | none | none | not implemented | not reached | **PENDING** | `test_ptha_mtst_preflight` | `AUTHOR_ONLY_INFORMATION`: no FashionIQ checkpoint exists; CIRR weights are not a substitute | author must publish FashionIQ weights |

## 3. Delivery defects closed in this pass

`B1` manifest ≠ tree. 35 manifest paths were absent from the commit: 34 IDE/notebook by-products
excluded during copy but not during manifest generation, and **one required source file**,
`PAIR/data/datasets.py`, blocked by the top-level `data/` ignore rule. Both now derive from one
predicate (`vendored_source.is_unwanted`), and the ignore negation is scoped to vendored trees with
bytecode explicitly re-ignored (Git's last-match-wins made the negation outrank `*.py[codz]`).

`B2` approved patch rejected. Applying the registry's own CSMCIR patch produced exactly
`derived_sha256` and the checkout was still reported modified. The vendored contract now honours
`approved_source_patches` as the nested-Git path did; an edit after the patch, a digest matching
neither side, and an unrelated edit are all still refused.

`B3` HINT cache reuse. The wrapper handed the canonical root to a dataset that reuses
`test_queries_*.pkl`, which hold already-transformed tensors. It now runs inside the existing
`isolated_cache_root`.

`B4` HINT unparsed. A valid result with exit 0 reported `AGGREGATE_PARSER_UNAVAILABLE`. Both real
output shapes now parse, with the macro recomputed and all three categories required.

Also closed: seven adapters that inherited `NotImplementedError`; eight wrapper models with no parser;
`command()` reading the checkpoint record where it receives a resolved path.

## 4. Files

Vendored: 15 checkouts, every manifest path present in the tree with matching bytes, no nested `.git`,
no gitlinks.

New in this pass: `vendor_upstreams.py` exclusion predicate alignment; `vendored_source.py`
approved-patch contract + `is_unwanted`; `metrics_extraction.py` `parse_hint_output` and
`parse_workbench_replay_output`; `replay/common.py` pin check carrying the model;
`replay/{pair,encoder,clvc_net,intent,habit,conesep,airknow,dcnet,ptha_mtst}.py`; adapters for all
fifteen entries; tests `test_fresh_clone_delivery.py`, `test_vendored_approved_patch.py`,
`test_hint_metric_parser.py`, `test_replay_output_parser.py`, and the per-model replay suites.

## 5. Running a recovered workflow

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git checkout research/cir-failure-workbench
grep -H upstream_commit workbench/third_party/*/provenance.json
```

Wrapper models take the same three paths, because every wrapper accepts the same contract:

```bash
python workbench/replay/<model>.py \
  --checkpoint-root <dir> --dataset-root <canonical FashionIQ> \
  --source-root workbench/third_party/<SourceDir> [--preflight-only]
```

`--preflight-only` reports the exact missing prerequisite without loading anything, which is the
honest way to inspect a model on a host that has no weights.

Script evaluators (`csmcir`, `sprc`, `clip4cir_*`, `dcnet`) run from their own checkout per the
registry's command; `sprc` needs the registered model name and a backbone matching its checkpoint,
both of which the adapter passes explicitly.
