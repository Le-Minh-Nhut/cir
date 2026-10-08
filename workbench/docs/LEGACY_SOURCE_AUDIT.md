# Legacy FashionIQ source audit

Audit date: 2026-10-08. Static audit of immutable author source pins, author README/requirements, and author-linked artifact metadata. No source clone, checkpoint, dataset, environment, preprocessing, patch, model execution, or GPU evaluation was created or run. Paper scores are sanity references, never reproductions. `preparation_contracts.yaml` describes requirements only; it authorizes no preprocessing.

| Model | Target / protocol | Command | Checkpoint structure | Environment | Runtime status |
| --- | --- | --- | --- | --- | --- |
| CLVC-Net | FashionIQ / `fashioniq_val_split` | **BLOCKED:** pinned `test.py` only defines `test(...)`; README bare `python test.py` has no source-backed standalone loader. | Author Drive `CLVCNET.zip`; source saves separate `local_model.pt` and `global_model.pt`, but archive member/category mapping is unverified. | README: Python 3.7.6, Torch 1.6.0; torchvision version/CUDA unknown. | `BLOCKED_CHECKPOINT_MAPPING`, source entrypoint, ResNet-50 external weights, preparation. |
| DCNet | standard FashionIQ / `fashioniq_full_gallery_ref_excluded` | **AUDITED:** repo root `python test.py --resume <run-directory>`. | Run directory, exactly `config.json` plus `trained_model.pth`; state dict loader. Drive association exists; artifact hash/content mapping unresolved. | Python 3.7.7, Torch 1.4.0, torchvision 0.5.0; requirements pin NLTK 3.5 and spaCy 2.3.0. | `BLOCKED_CHECKPOINT_MAPPING`, prepared artifacts, ImageNet ResNet-50 weights, source/environment. |
| Combiner RN50x4 noft | RN50x4 noft / `fashioniq_original_split` | **BLOCKED:** validator shape audited, but no author mapping identifies requested Combiner state or proves noft selection. | One Combiner state with `Combiner` key; author Drive file mapping/hash unresolved. | README: Python 3.8, Torch 1.11.0, torchvision 0.12.0, OpenAI CLIP. | `BLOCKED_CHECKPOINT_MAPPING`, RN50x4 base CLIP asset. |
| CLIP4Cir RN50x4 fullft | RN50x4 fullft / `fashioniq_original_split` | **BLOCKED:** required artifact paths/pairing unresolved. | Two states: `Combiner` and fine-tuned `CLIP`; never collapse to one file. Author pairing/hash unresolved. | Same CLIP4Cir README environment. | `BLOCKED_CHECKPOINT_BUNDLE`, RN50x4 base CLIP asset. |
| TG-CIR | FashionIQ / `fashioniq_val_split` | **BLOCKED:** `test.py:test(...)` is a function; no source-backed replay CLI/checkpoint loader. | Training saves a whole model; `TG-CIR.zip` FashionIQ member/hash/loader mapping unresolved. | README: Torch 1.7.0, torchvision 0.8.0, `clip==0.2.0`; Python/CUDA unspecified. | `BLOCKED_COMMAND`, checkpoint mapping, ViT-B/16 asset, manual preparation/caches. |
| SPRC BLIP-2 | BLIP-2 / `fashioniq_original_split` | **BLOCKED_MODEL_MAPPING:** repo-root CLI is structurally audited as `python src/blip_validate.py --dataset FashionIQ --blip-model-name <name> --backbone <type> --model-path <file>`, but `sprc_fiq.pt` has no verified model-name/backbone/class-key mapping. | `torch.load`; class-named state dict loaded `strict=False`. OneDrive `sprc_fiq.pt` structure is unverified. | Python 3.9, Torch 2.0.1, torchvision 0.15.2; requirements include transformers 4.36.2, timm 0.9.12, spaCy 3.7.2. | `BLOCKED_COMMAND_AND_CHECKPOINT_MAPPING`, LAVIS BLIP-2 asset. |
| LIMN base iteration 0 | base only / `fashioniq_val_split` | **BLOCKED:** README commands train category models; no inference-only replay CLI. | Mandatory three-file bundle: `0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt`; each is a whole saved model and author-Hub artifact-associated only. | Torch 1.12.1, torchvision 0.13.1, CUDA 12.4, open-clip-torch 2.20.0; Python unspecified. | `BLOCKED_COMMAND_AND_CHECKPOINT_BUNDLE`, DataComp OpenCLIP asset, manual preparation/caches. |

## CLVC-Net

**Pinned source:** [`iLearn-Lab/SIGIR21-CLVC-Net@bd9b6889489806cb60676597880f91d918b279cb`](https://github.com/iLearn-Lab/SIGIR21-CLVC-Net/tree/bd9b6889489806cb60676597880f91d918b279cb).

- **Checkpoint:** `train.py` saves independent whole `local_model.pt` and `global_model.pt`. Author Drive [`CLVCNET.zip`](https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing) exposes no author-attested member/category mapping or hash. **BLOCKED_CHECKPOINT_MAPPING**.
- **Command:** README's `python test.py` cannot be used: pinned `test.py` has no parser, main, construction, or checkpoint load. **COMMAND_BLOCKED_SOURCE_ENTRYPOINT**.
- **Preparation:** FashionIQ start-kit resize produces `resized_image/{category}`; linked start-kit defaults to 256-square resize. Validation applies Resize(256), CenterCrop(224), ToTensor, ImageNet normalization. Never run this automatically.
- **External runtime asset:** both branches call local `resnet50(pretrained=True)`; source uses PyTorch `resnet50-19c8e357.pth` URL but gives no fixed local cache path. Fail closed on provenance.
- **Evaluator semantics:** `datasets.py:FashionIQ.get_test_targets()` forms endpoint union; `test.py:test()` masks reference and ranks. `fashioniq_val_split`.

## DCNet

**Pinned source:** [`ozmig77/dcnet@68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`](https://github.com/ozmig77/dcnet/tree/68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb).

- **Checkpoint/run directory:** README and `test.py` require `python test.py --resume logdir/fashioniq_dcnet/`. `parse_config.py` reads `<resume>/config.json`; `BaseTrainer._resume_checkpoint` reads `<resume>/trained_model.pth` then `load_state_dict(checkpoint['state_dict'])`. Workbench represents this directory without faking one checkpoint file. Drive folder association exists but hash/content baseline mapping remains **UNVERIFIED**.
- **Command:** **COMMAND_AUDITED**, repo cwd, argv exactly `python test.py --resume <run-directory>`.
- **Standard boundary:** `configs/ce/fashioniq_dcnet.json` selects ImageNet-pretrained ResNet-50. `fashioniq_dcnet_deep.json` selects DenseNet-169 plus `deepfashion/logdir/deepfashion_densenet/best_ckpt.pth.tar`; it is a separate optional training configuration, not standard FashionIQ baseline dependency.
- **Preparation:** `resize_img.py` writes 256×256 `resized_images/`. `process_cap.py` creates GloVe PKLs consumed by evaluation. Its unseeded `np.random.normal` OOV vectors make regeneration nondeterministic; author PKLs preferred. spaCy `en_vectors_web_lg` and NLTK `punkt` are preparation-only when PKLs exist.
- **Runtime asset:** standard ResNet-50 ImageNet weights are evaluation-time requirement with unknown local cache provenance.
- **Evaluator:** CenterCrop(224), ToTensor, ImageNet normalization; reference is masked in both branches and `log_softmax` scores fuse. `fashioniq_full_gallery_ref_excluded`.

## Combiner RN50x4 noft

**Pinned source:** [`ABaldrati/CLIP4Cir@dfed9f748a8a4d05abf613164e06d59b22dbdf13`](https://github.com/ABaldrati/CLIP4Cir/tree/dfed9f748a8a4d05abf613164e06d59b22dbdf13).

- **Checkpoint / command:** `src/validate.py` can load a Combiner key and optional CLIP key, but author Drive metadata does not map file/hash to requested RN50x4 noft label. README path placeholders cannot become a command. **COMMAND_BLOCKED_CHECKPOINT_MAPPING**.
- **External runtime asset:** OpenAI `clip.load('RN50x4')`, source cache `~/.cache/clip`, URL hash `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd`; no local asset provenance established.
- **Transform:** `TargetPad(1.25)`, bicubic resize to CLIP-loaded resolution, center crop, RGB, tensor, CLIP normalization. Full split gallery retains reference. `fashioniq_original_split`.

## CLIP4Cir RN50x4 fullft

Same pin and evaluation transform as Combiner. Native fullft loader first loads base RN50x4 then separately overlays `CLIP` state; Combiner loader separately requires `Combiner` state. Author Drive exposes no verified pair identity, names, or hashes. Fullft remains **BLOCKED_CHECKPOINT_BUNDLE** and never becomes a synthetic single-file checkpoint.

## TG-CIR

**Pinned source:** [`iLearn-Lab/MM23-TG-CIR@65fa78eaf8cabe8197fcc22ccab42207c591bdee`](https://github.com/iLearn-Lab/MM23-TG-CIR/tree/65fa78eaf8cabe8197fcc22ccab42207c591bdee).

- **Checkpoint / command:** `test.py` exposes `test(params, model, testset, category)` only. `train.py` evaluates during training and saves whole `best_model.pt`; neither gives a source-backed checkpoint replay CLI. README-linked [`TG-CIR.zip`](https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view) member mapping and hash remain **UNVERIFIED**.
- **Preparation/caches:** source requires resized category JPEGs and source-provided correction dictionaries staged under `captions/`. `fashion_iq_data.json` gates six test PKLs with no freshness check; cache presence is not validity.
- **Runtime asset:** `clip.load` uses ViT-B/16 on CUDA; cache/provenance unknown. Same returned CLIP preprocess serves training/evaluation.
- **Evaluator:** endpoint union, reference `-10e10`, macro recall. `fashioniq_val_split`.

## SPRC BLIP-2

**Pinned source:** [`chunmeifeng/SPRC@2935a5397732260d1db6fa577e5926f963e36f0f`](https://github.com/chunmeifeng/SPRC/tree/2935a5397732260d1db6fa577e5926f963e36f0f).

- **Evaluator:** entrypoint is `src/blip_validate.py`; `src/validate_blip.py` is helper metrics code. CLI model/backbone remains unresolved for `sprc_fiq.pt`, so command is structurally audited but **COMMAND_BLOCKED_MODEL_MAPPING**.
- **Checkpoint:** evaluator loads `checkpoint[blip_model.__class__.__name__]` with `strict=False`. Author [OneDrive `sprc_fiq.pt`](https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj) provides no inspected class-key/model/backbone proof. Do not let non-strict load simulate a verified replay.
- **Runtime asset / transform:** LAVIS `blip2_pretrain.yaml` initializes ViT-g BLIP-2 from author-pinned pretrain URL; local cache is unknown. Local TargetPad ratio 1.25, bicubic 224, center crop, RGB, tensor, CLIP normalization. Full split gallery retains reference. `fashioniq_original_split`.
- **Score discrepancy:** Table 1 R@10 54.92 conflicts with printed category arithmetic 54.72. Both remain documented; neither is reproduced.

## LIMN base iteration 0

**Pinned source:** [`iLearn-Lab/TPAMI24-LIMN@7d7bc9b116f594a65ac22457491edf28a88d3c3e`](https://github.com/iLearn-Lab/TPAMI24-LIMN/tree/7d7bc9b116f594a65ac22457491edf28a88d3c3e); author Hub revision [`30560ad575a56c39cc39047d1a474269338c77c0`](https://huggingface.co/iLearn-Lab/TPAMI24-LIMN/tree/30560ad575a56c39cc39047d1a474269338c77c0).

- **Bundle:** exact base-iteration-0 whole-model files are `0_dress_best_model.pt` (`96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c`), `0_shirt_best_model.pt` (`9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74`), and `0_toptee_best_model.pt` (`79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2`). All three are mandatory; source saves `torch.save(model, ...)`. Mapping is artifact-associated, not replay-validated.
- **Command:** README gives category training commands from `LIMN/`; `LIMN/test.py:test(...)` has no standalone replay CLI. **COMMAND_BLOCKED_SOURCE_ENTRYPOINT**.
- **Preparation/runtime asset:** start-kit resized images and manually staged source `correct_files/correction_dict_{category}.json`; caches lack freshness validation. `open_clip.create_model_and_transforms` requires user-supplied `laionCLIP-ViT-L-14-DataComp.XL-s13B-b90K/open_clip_pytorch_model.bin`; no cache lookup/path substitution is authorized.
- **Evaluator:** category endpoint union, reference mask, macro recall. `fashioniq_val_split`; no LIMN+, later iteration, or partial category bundle is eligible.

## Exclusion

NEUCORE remains absent. Its forward/reverse caption-order queries double query universe. It is not a fallback, checkpoint source, asset source, or comparison cohort.
