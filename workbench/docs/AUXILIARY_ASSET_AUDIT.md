# Auxiliary Asset Audit -- every registered model, from pinned source

Generated from the per-cluster source audits in `workbench/registry/audits/`. Each row
cites the pinned source line that reads the asset, so the claim is checkable against the
checkout rather than against a README.

## Classification vocabulary

| Code | Meaning |
|---|---|
| `BUNDLED_WITH_SOURCE` | already shipped inside the pinned checkout |
| `OFFICIAL_PUBLIC_DOWNLOAD` | an author-published artifact with a verified source |
| `SOURCE_REGENERABLE` | an exact, source-documented generation procedure exists |
| `EXTERNAL_PRETRAINED_ASSET` | an official third-party backbone/tokenizer/model asset |
| `OPERATOR_PROVIDED` | the canonical dataset or a derived layout the operator supplies |
| `OFFICIAL_ASSET_UNAVAILABLE` | a required official artifact cannot currently be obtained |
| `SOURCE_EVIDENCE_MISSING` | the source does not establish the asset or its recipe |
| `OPTIONAL_NOT_REQUIRED_FOR_EVALUATION` | training-only or otherwise outside the eval path |

## hint

- pinned source: `workbench/third_party/HINT` at `bec50b6c8c19111893b617979502b948d1cea5b2`
- official eval entrypoint: `test.py`
- invocation evidence: train.py:199 -- t = test.test(args, model, trainset, category, txt_processors)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `hint_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:29 |  |
| `hint_correction_dict` | correction_dict | `SOURCE_EVIDENCE_MISSING` | UNVERIFIED | datasets.py:137 | Not bundled in pinned HINT checkout; no official download URL or generation script provided in repository. |
| `hint_bert_base_uncased` | tokenizer | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/models/blip2_models/blip2.py:32 |  |
| `hint_eva_vit_g_weights` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | AVAILABLE | lavis/models/eva_vit.py:442 |  |

Not required for evaluation: `hint_blip2_pretrained_weights`, `hint_fashioniq_cache`, `hint_query_target_pickles`

Unresolved:
- test.py contains no __main__ CLI block and train.py contains no evaluation-only CLI flag (train.py:59 defines --restore_file but ignores it); running official evaluation without training requires wrapping test.test().
- captions/correction_dict_{dress,shirt,toptee}.json is required by datasets.py:63, 137 but not bundled in HINT source; no download URL or generation script is provided in the repository.
- Blip2Base hardcodes '/root/bert-base-uncased' at blip2.py:32, 48, 54, requiring local bert-base-uncased weights or source patching.

## encoder

- pinned source: `workbench/third_party/ENCODER` at `29a2a31d6a56f677bf450c3be7cdaef423fb7018`
- official eval entrypoint: `evaluate_model.py`
- invocation evidence: evaluate_model.py:73-75 -- for ci, category in enumerate(['dress', 'shirt', 'toptee']): t = test.test(args, Encoder_model, dataset_list[0], category)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `encoder_datasets1_module` | annotation | `SOURCE_EVIDENCE_MISSING` | BLOCKED | evaluate_model.py:7 | datasets1.py is missing from the pinned checkout across all git branches/commits; evaluate_model.py fails with ModuleNotFoundError: No module named 'datasets1'. |
| `encoder_open_clip_vit_b32_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | evaluate_model.py:40 |  |
| `encoder_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:68 |  |
| `encoder_correction_dict` | correction_dict | `SOURCE_EVIDENCE_MISSING` | UNVERIFIED | datasets.py:163 | Not bundled in pinned ENCODER checkout; no official download link or generation script provided in repository. |

Not required for evaluation: `encoder_fashioniq_cache`, `encoder_query_target_pickles`

Unresolved:
- evaluate_model.py:7 imports datasets1, but datasets1.py is missing from the pinned checkout across all git branches/commits; running evaluate_model.py fails immediately with ModuleNotFoundError.
- evaluate_model.py:39 and model_try2.py:134 hardcode clip_path = '.' looking for open_clip_pytorch_model.bin in the working directory; official download URL is not documented in the repository.
- captions/correction_dict_{dress,shirt,toptee}.json is required by datasets.py:105, 163 but not bundled in ENCODER source; no official download URL is provided.
- Official evaluation checkpoint fashioniq.pt mapping and direct URL remain unverified; README links an unpinned Google Drive folder (README.md:249).

## pair

- pinned source: `workbench/third_party/PAIR` at `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`
- official eval entrypoint: `test.py`
- invocation evidence: test.py:6 -- def test(params, model, testset, category):

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `pair_mhtransformer_rev` | backbone_weights | `SOURCE_EVIDENCE_MISSING` | BLOCKED | models/model.py:13 | MHTransformer_rev.py is imported at models/model.py:13 and instantiated at line 47, but missing from pinned checkout. |
| `pair_open_clip_vit_b32_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | models/model.py:38 |  |
| `pair_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | data/datasets.py:27 |  |
| `pair_correction_dict` | correction_dict | `SOURCE_EVIDENCE_MISSING` | UNVERIFIED | data/datasets.py:122 | Not bundled in pinned PAIR checkout; no official download link or generation script provided in repository. |

Not required for evaluation: `pair_fashioniq_cache`, `pair_query_target_pickles`

Unresolved:
- models/model.py:13 imports MHTransformer_rev and line 47 instantiates MHTransformer_rev.Transformer, but MHTransformer_rev.py is completely missing from the pinned checkout, causing ModuleNotFoundError upon importing model.
- train.py is incomplete/truncated at line 154 without training or evaluation loop, and test.py lacks a __main__ CLI entrypoint; no official executable FashionIQ evaluation command is provided.
- captions/correction_dict_{dress,shirt,toptee}.json is required by data/datasets.py:65, 122 but not bundled in PAIR source; no official download URL is provided.
- Official checkpoint files pair-B1.pt and pair-B2.pt lack verified download URLs and explicit FashionIQ mapping in the repository.

## tgcir

- pinned source: `workbench/third_party/TGCIR` at `65fa78eaf8cabe8197fcc22ccab42207c591bdee`
- official eval entrypoint: `test.py`
- invocation evidence: train.py:200 -- t = test.test(args, model, trainset, category)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `tgcir_bundled_correction_dict` | correction_dict | `BUNDLED_WITH_SOURCE` | AVAILABLE | datasets.py:59 |  |
| `tgcir_clip_vit_b16_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | train.py:88 |  |
| `tgcir_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:25 |  |

Not required for evaluation: `tgcir_fashioniq_cache`, `tgcir_query_target_pickles`

Unresolved:
- test.py contains no __main__ CLI entrypoint and train.py contains no checkpoint-only evaluation command, preventing standalone evaluation invocation without running training or wrapping test.test().
- Checkpoints link in README.md:15 points to an unverified Google Drive zip file (1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi); exact FashionIQ member file and checksum remain unverified.
- Correction dictionaries are bundled under correction_dict/ in the checkout but datasets.py:59, 137 expects them at captions/ under the FashionIQ root, requiring operator placement.

## airknow

- pinned source: `workbench/third_party/AirKnow` at `6ab6fa95b746f85b08eea4cb66eb51448bce99b7`
- official eval entrypoint: `test_BLIP2.py`
- invocation evidence: AirKnow/train_BLIP2.py:298 -- 't = test.test(args, model, trainset, category, txt_processors)' in train_and_evaluate(); invoked via 'python train_BLIP2.py --dataset fashioniq --fashioniq_path <path>/ ...' (AirKnow/README.md:205-213)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `lavis_blip2_pretrained_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/configs/models/blip2/blip2_pretrain.yaml:10 |  |
| `bert_base_uncased_tokenizer_and_config` | tokenizer | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/models/blip2_models/blip2.py:32 | Upstream source hardcodes absolute host path /root/bert-base-uncased in blip2.py lines 32, 48, and 55. |
| `fashioniq_correction_dict_dress` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:187 | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_shirt` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:187 | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_toptee` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:187 | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `fashioniq_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:154 |  |
| `fashioniq_val_captions` | caption_source | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:185 |  |
| `fashioniq_val_splits` | annotation | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:183 |  |

Not required for evaluation: `fashioniq_cache_data_json`, `fashioniq_cache_test_queries_and_targets_pkl`, `fashioniq_train_captions_on_cache_miss`

Unresolved:
- No standalone evaluation entrypoint or CLI exists in AirKnow/test_BLIP2.py; test() is only callable as an internal training subroutine.
- No checkpoint loading routine exists to evaluate official author Hugging Face checkpoints (airknow_fiq_0.5.pt, airknow_fiq_0.8.pt); training script initializes a fresh model and saves best_model.pt.
- Official Hugging Face repository releases only 50% and 80% noise-trained weights; no clean (0% noise) trained checkpoint is published.
- Model is left in model.train() during FashionIQ validation in test_BLIP2.py:10.
- Hardcoded absolute path /root/bert-base-uncased in lavis/models/blip2_models/blip2.py:32,48,55.
- Missing provenance for captions/correction_dict_{dress,shirt,toptee}.json.

## conesep

- pinned source: `workbench/third_party/ConeSep` at `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b`
- official eval entrypoint: `test.py`
- invocation evidence: ConeSep/train.py:216 -- 't = test.test(args, model, trainset, category, txt_processors)' in train_and_evaluate(); invoked via 'python train.py --dataset fashioniq --fashioniq_path <path>/ ...' (ConeSep/README.md:213-221)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `lavis_blip2_pretrained_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/configs/models/blip2/blip2_pretrain.yaml:10 |  |
| `bert_base_uncased_tokenizer_and_config` | tokenizer | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/models/blip2_models/blip2.py:32 | Upstream source hardcodes absolute host path /root/bert-base-uncased in blip2.py lines 32, 48, and 55. |
| `fashioniq_correction_dict_dress` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:170 | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_shirt` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:170 | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_toptee` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:170 | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `fashioniq_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:136 |  |
| `fashioniq_val_captions` | caption_source | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:168 |  |
| `fashioniq_val_splits` | annotation | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:166 |  |

Not required for evaluation: `fashioniq_cache_data_json`, `fashioniq_cache_test_queries_and_targets_pkl`, `fashioniq_train_captions_on_cache_miss`

Unresolved:
- No standalone evaluation entrypoint or CLI exists in ConeSep/test.py; test() is only callable as an internal training subroutine.
- train.py defines argument --restore_file (line 76), but never implements restoring weights from it.
- No checkpoint loading routine exists to evaluate official author Hugging Face checkpoints (ConeSep-FIQ-N0.2.pt, ConeSep-FIQ-N0.5.pt, ConeSep-FIQ_N0.8.pt).
- Official Hugging Face repository releases only 20%, 50%, and 80% noise-trained weights; no clean (0% noise) trained checkpoint is published.
- Hardcoded absolute path /root/bert-base-uncased in lavis/models/blip2_models/blip2.py:32,48,55.
- Missing provenance for captions/correction_dict_{dress,shirt,toptee}.json.

## habit

- pinned source: `workbench/third_party/HABIT` at `6ddc2cd63eac1f65693f4e821c34fb3b42aee306`
- official eval entrypoint: `test.py`
- invocation evidence: HABIT/train.py:283 -- 't = test.test(args, model, trainset, category, txt_processors)' in train_and_evaluate(); invoked via 'python train.py --dataset fashioniq --fashioniq_path <path>/ ...' (HABIT/README.md)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `lavis_blip2_pretrained_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/configs/models/blip2/blip2_pretrain.yaml:10 |  |
| `bert_base_uncased_tokenizer_and_config` | tokenizer | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/models/blip2_models/blip2.py:32 | Upstream source hardcodes absolute host path /root/bert-base-uncased in blip2.py lines 32, 48, and 55. |
| `fashioniq_correction_dict_dress` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:174 | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_shirt` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:174 | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_toptee` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:174 | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `fashioniq_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:140 |  |
| `fashioniq_val_captions` | caption_source | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:172 |  |
| `fashioniq_val_splits` | annotation | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:170 |  |

Not required for evaluation: `fashioniq_cache_data_json`, `fashioniq_cache_test_queries_and_targets_pkl`, `fashioniq_train_captions_on_cache_miss`

Unresolved:
- No standalone evaluation entrypoint or CLI exists in HABIT/test.py; test() is only callable as an internal training subroutine.
- train.py defines argument --restore_file (line 79), but never implements restoring weights from it.
- No checkpoint loading routine exists to evaluate official author Hugging Face checkpoints (HABIT-FIQ_N0.2.pt, HABIT-FIQ_N0.5.pt, HABIT-FIQ_N0.8.pt).
- Official Hugging Face repository releases only 20%, 50%, and 80% noise-trained weights; no clean (0% noise) trained checkpoint is published.
- Hardcoded absolute path /root/bert-base-uncased in lavis/models/blip2_models/blip2.py:32,48,55.
- Missing provenance for captions/correction_dict_{dress,shirt,toptee}.json.

## intent

- pinned source: `workbench/third_party/INTENT` at `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3`
- official eval entrypoint: `test.py`
- invocation evidence: INTENT/train_INTENT.py:284 -- 't = test.test(args, model, trainset, category, txt_processors)' in train_and_evaluate(); invoked via 'python train_INTENT.py --dataset fashioniq --fashioniq_path <path>/ ...' (INTENT/README.md)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `lavis_blip2_pretrained_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/configs/models/blip2/blip2_pretrain.yaml:10 |  |
| `bert_base_uncased_tokenizer_and_config` | tokenizer | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | lavis/models/blip2_models/blip2.py:32 | Upstream source hardcodes absolute host path /root/bert-base-uncased in blip2.py lines 32, 48, and 55. |
| `fashioniq_correction_dict_dress` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:172 | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_shirt` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:172 | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `fashioniq_correction_dict_toptee` | correction_dict | `OFFICIAL_ASSET_UNAVAILABLE` | BLOCKED | datasets.py:172 | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `fashioniq_resized_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:138 |  |
| `fashioniq_val_captions` | caption_source | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:170 |  |
| `fashioniq_val_splits` | annotation | `OPERATOR_PROVIDED` | AVAILABLE | datasets.py:168 |  |

Not required for evaluation: `fashioniq_cache_data_json`, `fashioniq_cache_test_queries_and_targets_pkl`, `fashioniq_train_captions_on_cache_miss`

Unresolved:
- No standalone evaluation entrypoint or CLI exists in INTENT/test.py; test() is only callable as an internal training subroutine.
- train_INTENT.py defines argument --restore_file, but never implements restoring weights from it.
- No checkpoint loading routine exists to evaluate official author Hugging Face checkpoints (intent_fiq_0.2.pt, intent_fiq_0.5.pt, intent_fiq_0.8.pt).
- Official Hugging Face repository releases only 20%, 50%, and 80% noise-trained weights; no clean (0% noise) trained checkpoint is published.
- Hardcoded absolute path /root/bert-base-uncased in lavis/models/blip2_models/blip2.py:32,48,55.
- Missing provenance for captions/correction_dict_{dress,shirt,toptee}.json.

## combiner_rn50x4_noft

- pinned source: `workbench/third_party/CLIP4Cir` at `dfed9f748a8a4d05abf613164e06d59b22dbdf13`
- official eval entrypoint: `src/validate.py`
- invocation evidence: src/validate.py:263-356 and README.md:258-269 -- python src/validate.py --dataset fashionIQ --combining-function combiner --combiner-path <combiner_path> --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `clip_rn50x4_backbone` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | src/validate.py:279 |  |
| `clip4cir_fashioniq_link` | dataset_symlink | `OPERATOR_PROVIDED` | AVAILABLE | src/data_utils.py:11 |  |

Unresolved:
- Author Google Drive folder https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing does not document which file maps to the RN50x4 noft Combiner checkpoint, nor does it publish a checksum.

## clip4cir_rn50x4_fullft

- pinned source: `workbench/third_party/CLIP4Cir` at `dfed9f748a8a4d05abf613164e06d59b22dbdf13`
- official eval entrypoint: `src/validate.py`
- invocation evidence: src/validate.py:263-356 and README.md:258-269 -- python src/validate.py --dataset fashionIQ --combining-function combiner --combiner-path <combiner_path> --clip-model-path <clip_path> --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `clip_rn50x4_backbone` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | src/validate.py:279 |  |
| `clip4cir_fashioniq_link` | dataset_symlink | `OPERATOR_PROVIDED` | AVAILABLE | src/data_utils.py:11 |  |

Unresolved:
- Author Google Drive folder https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing does not identify which fine-tuned CLIP state pairs with which Combiner state for the RN50x4 fullft run, requiring bundle-level pairing verification.

## clvc_net

- pinned source: `workbench/third_party/CLVCNet` at `bd9b6889489806cb60676597880f91d918b279cb`
- official eval entrypoint: `test.py`
- invocation evidence: test.py:6 and train.py:259 -- test.test(args, local_model, global_model, dataset, args.dataset); README.md:70 cites 'python test.py' but test.py defines test(...) with no CLI parser

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `torchvision_resnet50_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | model.py:72 |  |
| `clvc_resized_images` | resized_images | `SOURCE_REGENERABLE` | AVAILABLE | datasets.py:195 |  |
| `clvc_category_checkpoint_bundle` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | UNVERIFIED | train.py:274 | Author Google Drive archive member structure and mapping to the six per-category model slots has not been verified |

Not required for evaluation: `clvc_simplevocab_fallback`

Unresolved:
- CLVCNet/test.py does not define an executable CLI or __main__ block; standalone evaluation requires an external replay wrapper or running through train.py.
- The author Google Drive archive CLVCNET.zip (https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing) member layout and category mapping remain unverified.
- datasets.py:157 relies on string concatenation self.path + 'resized_image', requiring self.path to end with a trailing slash to avoid malformed paths.

## dcnet

- pinned source: `workbench/third_party/DCNet` at `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`
- official eval entrypoint: `test.py`
- invocation evidence: test.py:96-117 and README.md:62 -- python test.py --resume logdir/fashioniq_dcnet/

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `dcnet_glove_val_pkls` | caption_source | `SOURCE_REGENERABLE` | AVAILABLE | data_loader/CE_dataset.py:34 |  |
| `dcnet_resized_images` | resized_images | `SOURCE_REGENERABLE` | AVAILABLE | data_loader/CE_dataset.py:22 |  |
| `torchvision_resnet50_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | model/model.py:283 |  |
| `dcnet_checkpoint_run_dir` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | UNVERIFIED | test.py:98 | Author Google Drive run directory contents and checkpoint hashes have not been audited or verified |

Not required for evaluation: `spacy_en_vectors_web_lg`, `nltk_punkt_tokenizer`

Unresolved:
- preprocess/process_cap.py:31 samples unseeded np.random.normal(0, 0.3, (300,)) for out-of-vocabulary words exceeding frequency 2, creating non-deterministic word vectors on regeneration.
- The author Google Drive run directory at https://drive.google.com/drive/folders/1wgygqF095Di67EaHaGOXbwh3wEzk9izB?usp=sharing has unverified file hashes and configuration compatibility.

## sprc

- pinned source: `workbench/third_party/SPRC` at `2935a5397732260d1db6fa577e5926f963e36f0f`
- official eval entrypoint: `src/blip_validate.py`
- invocation evidence: src/blip_validate.py:141-155 and README.md:108-112 -- python src/blip_validate.py --dataset FashionIQ --blip-model-name blip2_cir_align_prompt --backbone pretrain --model-path <model_path> (default blip-model-name blip2_cir_rerank_learn is unregistered in LAVIS)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `lavis_blip2_base_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | src/lavis/configs/models/blip2/blip2_pretrain.yaml:10 |  |
| `hf_bert_base_uncased_tokenizer` | tokenizer | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | src/lavis/models/blip2_models/blip2.py:32 |  |
| `sprc_fashioniq_link` | dataset_symlink | `OPERATOR_PROVIDED` | AVAILABLE | src/data_utils.py:12 |  |
| `sprc_fashioniq_png_images` | resized_images | `OPERATOR_PROVIDED` | AVAILABLE | src/data_utils.py:185 |  |
| `sprc_fiq_checkpoint` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | UNVERIFIED | src/blip_validate.py:48 | OneDrive link has not been audited or verified for key structure, and blip_validate.py:49 uses strict=False which silently ignores missing or unexpected keys |

Unresolved:
- CLI argument --blip-model-name defaults to blip2_cir_rerank_learn at src/blip_validate.py:144, but no model with that name is registered in LAVIS registry, raising KeyError unless overridden at CLI.
- src/data_utils.py:185 hardcodes .png image extension, while canonical FashionIQ provides .jpg; images must be converted or symlinked with .png.
- Checkpoint loading in src/blip_validate.py:49 executes strict=False on checkpoint[blip_model.__class__.__name__], which masks missing or mismatched layer keys if the checkpoint does not match the chosen architecture.
- In src/lavis/models/blip2_models/blip2_qformer_cir_rerank.py:Blip2QformerCirRerank.inference, reference_embeds is accepted but never used in similarity computation (similarity is text-to-gallery), creating a model-level departure from reference-conditioned CIR.

## csmcir

- pinned source: `workbench/third_party/CSMCIR` at `774f94e2076ff17ea91703a6239d2a08f0e1a44e`
- official eval entrypoint: `src/validate_blip_csmcir.py`
- invocation evidence: src/validate_blip_csmcir.py:515 -- blip_validate_fashioniq(args.blip_model_name, args.blip_model_path, args.transform, args.target_ratio); invoked as: python src/validate_blip_csmcir.py --dataset fashionIQ --blip-model-name blip2_cir_align_prompt_csmcir --blip-model-path <checkpoint_path>

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `csmcir_qwen_captions` | caption_source | `OFFICIAL_PUBLIC_DOWNLOAD` | AVAILABLE | src/data_utils_csmcir.py:408 |  |
| `csmcir_cot_ours2_captions` | caption_source | `SOURCE_EVIDENCE_MISSING` | BLOCKED | src/utils_csmcir.py:100 | Author did not publish COT_ours2/ to GitHub or HuggingFace repo; no public download URL or checksum exists |
| `csmcir_checkpoint_fashioniq` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | AVAILABLE | src/validate_blip_csmcir.py:570 |  |
| `csmcir_blip2_base_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | src/lavis/configs/models/blip2/blip2_pretrain.yaml:10 |  |
| `csmcir_eva_vit_g_weights` | backbone_weights | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | src/lavis/models/eva_vit.py:442 |  |
| `csmcir_bert_tokenizer` | tokenizer | `EXTERNAL_PRETRAINED_ASSET` | AVAILABLE | src/lavis/models/blip2_models/blip2.py:32 |  |

Not required for evaluation: `csmcir_keywords_in_mods`, `csmcir_bert_captions`

Unresolved:
- COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json is required by extract_index_blip_caption_features (src/utils_csmcir.py:100-103), but is absent from upstream GitHub and HuggingFace repositories; acquisition source is unverified.
- Image loading in FashionIQDataset.__getitem__ (src/data_utils_csmcir.py:468) hardcodes the .png extension, whereas standard FashionIQ distribution provides .jpg files.
- base_path in src/data_utils_csmcir.py:16 and src/utils_csmcir.py:18 resolves to <checkout_root> (workbench/third_party/CSMCIR), so fashionIQ_dataset must be a symlink or directory at the checkout root, and COT_ours2 must be placed at <checkout_root>/COT_ours2/.

## limn

- pinned source: `workbench/third_party/LIMN` at `7d7bc9b116f594a65ac22457491edf28a88d3c3e`
- official eval entrypoint: `LIMN/test.py`
- invocation evidence: LIMN/train.py:231 -- t = test.test(args, model, dataset_list[_id], _name[1]); official test function is test.py:23 test(params, model, testset, category); invoked via workbench wrapper workbench/replay/limn.py:142 -- native_test.test(SimpleNamespace(local_rank=0, batch_size=32), model, dataset, category)

| Asset | Role | Classification | Acquisition | Source line | Blocker |
|---|---|---|---|---|---|
| `limn_checkpoint_dress` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | AVAILABLE | LIMN/train.py:263 |  |
| `limn_checkpoint_shirt` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | AVAILABLE | LIMN/train.py:263 |  |
| `limn_checkpoint_toptee` | backbone_weights | `OFFICIAL_PUBLIC_DOWNLOAD` | AVAILABLE | LIMN/train.py:263 |  |
| `limn_resized_images` | resized_images | `SOURCE_REGENERABLE` | AVAILABLE | LIMN/datasets.py:44 |  |
| `limn_correction_dictionaries` | correction_dict | `BUNDLED_WITH_SOURCE` | AVAILABLE | LIMN/datasets.py:70 |  |
| `limn_train_data_cache` | annotation | `SOURCE_REGENERABLE` | AVAILABLE | LIMN/datasets.py:49 |  |
| `limn_test_queries_targets_cache` | annotation | `SOURCE_REGENERABLE` | AVAILABLE | LIMN/datasets.py:58 |  |

Not required for evaluation: `limn_openclip_pretrained_weights`

Unresolved:
- Upstream LIMN has no standalone eval CLI script (test.py defines test() with no __main__ block; evaluation is called in train.py:231), requiring a wrapper harness such as workbench/replay/limn.py.
- LIMN/datasets.py:44 concatenates self.path + 'resized_image' without os.path.join, strictly requiring self.path to end with a trailing slash '/'.
- FashionIQ.__init__ (LIMN/datasets.py:49-56) unconditionally checks and generates {category}_train_data.json even during evaluation, creating a prerequisite on captions/cap.{category}.train.json when the cache is absent.

## Blocking assets, by model

| Model | Asset | Classification | Blocker |
|---|---|---|---|
| `hint` | `hint_correction_dict` | `SOURCE_EVIDENCE_MISSING` | Not bundled in pinned HINT checkout; no official download URL or generation script provided in repository. |
| `encoder` | `encoder_datasets1_module` | `SOURCE_EVIDENCE_MISSING` | datasets1.py is missing from the pinned checkout across all git branches/commits; evaluate_model.py fails with ModuleNotFoundError: No module named 'datasets1'. |
| `encoder` | `encoder_correction_dict` | `SOURCE_EVIDENCE_MISSING` | Not bundled in pinned ENCODER checkout; no official download link or generation script provided in repository. |
| `pair` | `pair_mhtransformer_rev` | `SOURCE_EVIDENCE_MISSING` | MHTransformer_rev.py is imported at models/model.py:13 and instantiated at line 47, but missing from pinned checkout. |
| `pair` | `pair_correction_dict` | `SOURCE_EVIDENCE_MISSING` | Not bundled in pinned PAIR checkout; no official download link or generation script provided in repository. |
| `airknow` | `fashioniq_correction_dict_dress` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `airknow` | `fashioniq_correction_dict_shirt` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `airknow` | `fashioniq_correction_dict_toptee` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `conesep` | `fashioniq_correction_dict_dress` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `conesep` | `fashioniq_correction_dict_shirt` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `conesep` | `fashioniq_correction_dict_toptee` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `habit` | `fashioniq_correction_dict_dress` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `habit` | `fashioniq_correction_dict_shirt` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `habit` | `fashioniq_correction_dict_toptee` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `intent` | `fashioniq_correction_dict_dress` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_dress.json nor provides a download URL or generation script. |
| `intent` | `fashioniq_correction_dict_shirt` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_shirt.json nor provides a download URL or generation script. |
| `intent` | `fashioniq_correction_dict_toptee` | `OFFICIAL_ASSET_UNAVAILABLE` | Upstream repository neither tracks captions/correction_dict_toptee.json nor provides a download URL or generation script. |
| `csmcir` | `csmcir_cot_ours2_captions` | `SOURCE_EVIDENCE_MISSING` | Author did not publish COT_ours2/ to GitHub or HuggingFace repo; no public download URL or checksum exists |

Models audited: 15. A blocking asset here is not an engineering failure: it is
the exact, evidence-backed reason a model cannot be evaluated, and the workbench reports it
instead of substituting something similar.
