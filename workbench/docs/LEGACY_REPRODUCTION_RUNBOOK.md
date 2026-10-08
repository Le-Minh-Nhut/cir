# Legacy FashionIQ reproduction runbook

This guide records source-faithful launch and prerequisite details for seven legacy baselines. It does not authorize downloading weights, generating datasets, creating environments, or changing upstream source. Use pinned source commits in `workbench/registry/models.yaml`; keep each model in its documented native protocol. Missing hashes, artifact mappings, or assets remain blockers. A paper score or successful preflight is not evidence of reproduction.

## Reproduction stages

1. **Pin source and protocol.** Use exact `upstream_commit_sha` and `native_protocol`; do not compare across protocols.
2. **Prepare inputs manually.** Stage FashionIQ files and model-specific derived artifacts as documented. Preparation contracts are requirements, not permission to run preprocessing. Preserve provenance and generated-file hashes.
3. **Establish isolated environment.** Match version data below. Unknown Python/CUDA details remain unresolved; do not silently substitute a different stack.
4. **Validate artifacts.** Confirm required file structure, SHA-256 where published, and author mapping. Never deserialize a whole-model pickle until provenance is trusted. Verify external runtime assets separately.
5. **Preflight then evaluate.** Run only the native source evaluator or explicit Workbench replay wrapper after all blockers clear. Preserve native transforms, gallery and reference handling.
6. **Record results.** Keep raw category metrics, checkpoint/source/asset hashes, environment, command and protocol. Aggregate FashionIQ category results only after all three are complete. Reproduction status changes only after validated end-to-end evaluation.

## CLVC-Net

- **Protocol / source:** `fashioniq_val_split`; `iLearn-Lab/SIGIR21-CLVC-Net@bd9b6889489806cb60676597880f91d918b279cb`.
- **Checkpoint:** source saves whole-model `local_model.pt` and `global_model.pt`. Author `CLVCNET.zip` member/category mapping is unresolved; do not assume the two files' pairing.
- **Native evaluation:** `test.py` exposes `test(params, local_model, global_model, testset, dataname)`; it has no CLI/model construction/checkpoint loader. Source-faithful construction requires `compose_local(texts, T, dropout_p)` and `compose_global(texts, T, dropout_p)` with `texts=trainset.get_all_texts()`.
- **Replay command:** none is currently source-backed; do not use README's bare `python test.py` as a replay command.
- **Required assets:** FashionIQ val annotations/splits/images and category resized images; torchvision ResNet-50 ImageNet weights at `$TORCH_HOME/checkpoints/resnet50-19c8e357.pth` (default `~/.cache/torch/checkpoints/resnet50-19c8e357.pth`); verify published filename hash prefix `19c8e357`.
- **Environment:** Python 3.7.6, PyTorch 1.6.0; torchvision and CUDA versions unknown.
- **Preparation:** start-kit resize creates `resized_image/{dress,shirt,toptee}`. Manual only; no auto-preparation.
- **Current blockers:** unresolved checkpoint mapping, missing source-backed executable replay, ResNet-50 asset provenance, and manual preparation.
- **Expected output:** three category recall results from the native endpoint-union, reference-excluded evaluator; no trusted aggregate until full set is complete. Workbench has no runnable source entrypoint for this model.

## DCNet

- **Protocol / source:** `fashioniq_full_gallery_ref_excluded`; `ozmig77/dcnet@68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`.
- **Checkpoint:** run directory containing exactly the required `config.json` and `trained_model.pth`; loader restores state dict from run checkpoint. Preserve run directory layout.
- **Native replay command:** from pinned repository root, `python test.py --resume <run-directory>`.
- **Required assets:** raw FashionIQ captions/splits/images, prepared `captions/cap.{category}.glove.val.pkl`, resized `resized_images/`, and torchvision ResNet-50 ImageNet weights at `$TORCH_HOME/checkpoints/resnet50-19c8e357.pth` (default `~/.cache/torch/checkpoints/resnet50-19c8e357.pth`); verify published filename hash prefix `19c8e357`. spaCy `en_vectors_web_lg` and NLTK `punkt` are only needed to regenerate caption PKLs, not evaluation when author/generated PKLs already exist.
- **Environment:** Python 3.7.7, PyTorch 1.4.0, torchvision 0.5.0; NLTK 3.5, spaCy 2.3.0. CUDA unknown.
- **Preparation:** upstream `preprocess/process_cap.py` and `resize_img.py` generate artifacts. OOV vectors are sampled without a seed; prefer author PKLs and do not regenerate silently.
- **Current blockers:** Drive run-directory-to-baseline mapping/hash unresolved, FashionIQ preparation state and ImageNet asset provenance unresolved.
- **Expected output:** source evaluator recall metrics for reference-excluded full gallery; preserve per-category values and source output.

## Combiner RN50x4 noft

- **Protocol / source:** `fashioniq_original_split`; `ABaldrati/CLIP4Cir@dfed9f748a8a4d05abf613164e06d59b22dbdf13`.
- **Checkpoint:** one Combiner state dict checkpoint, key `Combiner`. Author Drive artifact mapping for the no-finetuning RN50x4 baseline is unresolved; do not substitute an arbitrary state.
- **Native replay command:** from CLIP4Cir root, `python src/validate.py --dataset fashionIQ --combining-function combiner --combiner-path <combiner-state> --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad`.
- **Required assets:** FashionIQ validation annotations/splits/images and OpenAI CLIP RN50x4 base weights. CLIP source cache defaults to `~/.cache/clip`; independently verify provenance.
- **Environment:** Python 3.8, PyTorch 1.11.0, torchvision 0.12.0, OpenAI CLIP; CUDA unspecified.
- **Preparation:** native TargetPad ratio 1.25, bicubic resize, center crop, RGB tensor, CLIP normalization; no generated dataset artifacts documented.
- **Current blockers:** noft checkpoint identity/mapping/hash and base CLIP asset provenance.
- **Expected output:** native original-split retrieval metrics; full gallery retains reference as source defines.

## CLIP4Cir RN50x4 fullft

- **Protocol / source:** `fashioniq_original_split`; same pinned CLIP4Cir source as Combiner above.
- **Checkpoint:** paired files `combiner_state.pt` (key `Combiner`) and fine-tuned `clip_state.pt` (key `CLIP`). They are separate states and must remain a verified author pair.
- **Native replay command:** from CLIP4Cir root, `python src/validate.py --dataset fashionIQ --combining-function combiner --combiner-path <combiner-state> --clip-model-path <clip-state> --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad`.
- **Required assets:** FashionIQ validation inputs, OpenAI CLIP RN50x4 base weights and both paired checkpoint files.
- **Environment:** Python 3.8, PyTorch 1.11.0, torchvision 0.12.0, OpenAI CLIP; CUDA unspecified.
- **Preparation:** same native TargetPad/CLIP preprocessing as noft; no generated dataset artifacts documented.
- **Current blockers:** author Drive pairing and checkpoint mapping/hash unresolved; base CLIP asset provenance unresolved. Never collapse pair into a synthetic one-file checkpoint.
- **Expected output:** native original-split metrics. Keep the full gallery reference-eligible as source specifies.

## TG-CIR

- **Protocol / source:** `fashioniq_val_split`; `iLearn-Lab/MM23-TG-CIR@65fa78eaf8cabe8197fcc22ccab42207c591bdee`.
- **Checkpoint:** whole model saved as `best_model.pt`; FashionIQ archive member/hash/checkpoint loader mapping unresolved.
- **Native evaluation:** `test.py` defines `test(params, model, testset, category)` but has no standalone checkpoint replay CLI. `train.py` evaluation is not a substitute for inference replay.
- **Replay command:** no source-backed command currently available.
- **Required assets:** FashionIQ train/val captions and splits, resized category JPEGs, manually staged `captions/correction_dict_{category}.json`, source test caches if needed, OpenAI CLIP ViT-B/16 weights/cache. Caches have no freshness validation.
- **Environment:** PyTorch 1.7.0, torchvision 0.8.0, `clip==0.2.0`; Python/CUDA unspecified.
- **Preparation:** category resized images and correction dictionaries required. Do not copy unprovenanced caches from another repository.
- **Current blockers:** absent source replay entrypoint, unresolved checkpoint mapping, ViT-B/16 asset provenance and manual preparation.
- **Expected output:** native category recall results, reference masked over endpoint-union gallery. No Workbench executable replay currently.

## SPRC

- **Protocol / source:** `fashioniq_original_split`; `chunmeifeng/SPRC@2935a5397732260d1db6fa577e5926f963e36f0f`.
- **Checkpoint:** `sprc_fiq.pt`; loader reads class-named entry from checkpoint and applies `strict=False`. File's model-class key/backbone mapping remains unverified; non-strict loading does not verify identity.
- **Evaluator entrypoint:** `src/blip_validate.py`; exact model-name/backbone flags remain blocked until author checkpoint class-key mapping is verified. Do not run a guessed command.
- **Required assets:** FashionIQ validation annotations/splits/images, LAVIS BLIP-2 pretrained ViT-g weights and `sprc_fiq.pt`.
- **Environment:** Python 3.9, PyTorch 2.0.1, torchvision 0.15.2; transformers 4.36.2, timm 0.9.12, spaCy 3.7.2, LAVIS; CUDA unspecified.
- **Preparation:** source evaluation performs TargetPad ratio 1.25 and CLIP normalization; no separate generated assets listed.
- **Current blockers:** `sprc_fiq.pt` model/backbone/class-key mapping unresolved; LAVIS pretrain cache and provenance unresolved.
- **Expected output:** native original-split retrieval metrics with reference retained. Preserve reported score discrepancy; it is not reproduction evidence.

## LIMN base iteration 0

- **Protocol / source:** `fashioniq_val_split`; `iLearn-Lab/TPAMI24-LIMN@7d7bc9b116f594a65ac22457491edf28a88d3c3e`.
- **Checkpoint:** exactly three author-associated whole-model files: `0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt`, with registry-published SHA-256 values. Verify all hashes before deserialization. These are three independent category models, not one shared model; each category uses only its matching checkpoint. Never average or merge parameters.
- **Replay command:** `PYTHONPATH=. python -m workbench.replay.limn --checkpoint-root <checkpoint-model-dir> --dataset-root <fashioniq-root> --source-root <pinned-repo-root> [--category dress|shirt|toptee] [--preflight-only] [--output <new-json-path>]`. Omit `--category` for all three categories; only complete three-category run emits macro metrics. Output path is create-only; existing files are never overwritten.
- **Required assets:** all three verified checkpoints, FashionIQ train/val captions/splits, resized category images, and manually staged correction dictionaries.
- **Environment:** PyTorch 1.12.1, torchvision 0.13.1, CUDA 12.4, `open_clip_torch==2.20.0`; Python unspecified.
- **Preparation:** start-kit resize and manual placement of pinned-source correction dictionaries. Native caches are created under an isolated temporary root; source caches are not reused. Whole-model checkpoints already serialize their OpenCLIP backbone.
- **Current blockers:** no verified local source/data/checkpoints/environment have been exercised.
- **Expected output:** one native result per category and macro R@10/R@50/mean only when all three category results are complete. Partial category runs are incomplete, not aggregate scores.

## ENCODER contrast


## Operator commands for legacy baselines

Every legacy baseline can be inspected and guarded through standalone operator tools:

```bash
# Check prerequisites and blockers for DCNet:
python workbench/scripts/doctor.py --model dcnet

# Inspect environment requirements:
python workbench/scripts/manage_environment.py --model dcnet

# Inspect required raw and generated preparation artifacts:
python workbench/scripts/prepare_model.py --model dcnet

# Verify checkpoint directory integrity without network access:
python workbench/scripts/download_checkpoints.py --model dcnet --verify-only

# Dry-run execution plan:
python workbench/scripts/evaluate_models.py --model dcnet --dry-run

# Pipeline reproduction with stage resume:
python workbench/scripts/pipeline.py reproduce --model dcnet --resume
```
ENCODER remains one shared FashionIQ model with one checkpoint variant. It is not a three-checkpoint category bundle. LIMN's independent category checkpoints must not be generalized to ENCODER or other shared-model baselines.
