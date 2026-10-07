# Legacy FashionIQ source audit

Audit date: 2026-10-08. Static audit of immutable author source pins and author-linked artifacts only. No upstream clone, dataset, checkpoint, environment, preprocessing, patch, or GPU evaluation was created or run. Reported paper scores are sanity references, never reproductions.

Preparation contracts live in [`preparation_contracts.yaml`](../registry/preparation_contracts.yaml). They describe required inputs; they do not authorize any preprocessing command.

## CLVC-Net

- **Source / checkpoint.** [`iLearn-Lab/SIGIR21-CLVC-Net@bd9b6889489806cb60676597880f91d918b279cb`](https://github.com/iLearn-Lab/SIGIR21-CLVC-Net/tree/bd9b6889489806cb60676597880f91d918b279cb). Author Drive exists, but filename, category mapping, hash, loader pairing, command, and environment are **UNVERIFIED**.
- **Raw inputs.** FashionIQ train/validation captions, validation split JSON, and images.
- **Offline preprocessing / generated artifacts.** `resized_image/{dress,shirt,toptee}` JPEGs required. Resize generator details and provenance are **UNKNOWN**. Possible pretrained ResNet cache is **UNRESOLVED**.
- **External assets.** torchvision ResNet-50 ImageNet weights/cache is unresolved.
- **Evaluation transform.** **UNKNOWN** from audited evidence.
- **Caption/text pipeline.** `datasets.py:FashionIQ.get_test_targets()` uses `<BOS> caption_1 <AND> caption_2 <EOS>`.
- **Query construction / gallery / reference policy.** One query per annotation; first-seen reference/target endpoint union; `test.py:test()` masks source before descending ranking.
- **Scoring / metrics.** Source rank/metric internals beyond source mask are **UNKNOWN**. Assignment is `fashioniq_val_split` only.
- **Preparation determinism.** **UNKNOWN**. No automatic execution.
- **Current blockers / instrumentation.** Checkpoint mapping, native command, transform, resize provenance, runtime all blocked. Instrument only after official aggregate metric parity.

## DCNet

- **Source / checkpoint.** [`ozmig77/dcnet@68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`](https://github.com/ozmig77/dcnet/tree/68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb). Author Drive folder exists. `config.json` / `trained_model.pth` association and hash are **UNVERIFIED**.
- **Raw inputs.** Raw FashionIQ caption JSON, validation `image_splits/split.{category}.val.json`, and source JPG images.
- **Offline preprocessing.** `preprocess/resize_img.py` writes `resized_images/`: it calls `image.resize([256, 256], Image.ANTIALIAS)`, preserves image format, and prints/continues per-image failures. `preprocess/process_cap.py` writes `glove_vecs.pkl` and caption GloVe PKLs.
- **External assets.** `process_cap.py` requires spaCy `en_vectors_web_lg` and NLTK `punkt`.
- **Generated artifacts.** Evaluation requires `captions/cap.{dress,shirt,toptee}.glove.val.pkl` and `resized_images/`. `data_loader/CE_dataset.py:CE.__init__` consumes the PKLs; it does not construct these vectors from raw captions at evaluation.
- **Evaluation transform.** `data_loader/data_loaders.py:dataset_loader` validation transform is `CenterCrop(224)`, `ToTensor()`, then ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. Training `RandomCrop(224)` / `RandomHorizontalFlip` is separate and must not be used for evaluation.
- **Caption/text pipeline.** `process_cap.py` lowercases, replaces hyphens with spaces, removes periods, applies NLTK tokenization, loads spaCy GloVe vectors, and omits low-frequency OOV words. For OOV words with train-count over two it samples `np.random.normal(0, 0.3, (300,))`.
- **Preparation determinism.** **NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT**: `process_cap.py` has no seed for those NumPy OOV samples. `test.py` evaluator seeds do not recreate preprocessing state. Never auto-run it. Author-generated PKLs are preferred. Any later explicitly approved regeneration must record seed state, NumPy version, spaCy model/version, NLTK tokenizer/version, and every generated hash.
- **Query construction / gallery / reference policy.** `CE_dataset.py:CE.__init__` uses ordered `split.{category}.val.json` directly as `val_trg`. `TrainerJoint._valid_epoch` masks reference score to `-10e10` in composition and correction branches.
- **Scoring / metrics.** DCNet ranks `F.log_softmax(score_comp) + F.log_softmax(score_corr)` after the masks; source emits top-50. This is `fashioniq_full_gallery_ref_excluded` semantics, independent of execution readiness.
- **Current blockers / instrumentation.** Author PKLs, resize output, checkpoint/config mapping, legacy environment, command verification, and official reproduction are blocked. Export only existing final rankings after aggregate parity.

## Combiner RN50x4 noft and CLIP4Cir RN50x4 fullft

- **Source / checkpoint.** [`ABaldrati/CLIP4Cir@dfed9f748a8a4d05abf613164e06d59b22dbdf13`](https://github.com/ABaldrati/CLIP4Cir/tree/dfed9f748a8a4d05abf613164e06d59b22dbdf13). Author Drive folder exists; noft/fullft filenames, paired model states, hashes, and commands are separately **UNVERIFIED**.
- **Raw inputs / preprocessing.** Standard FashionIQ PNG layout; no generated dataset artifact established by audit.
- **External assets.** RN50x4 CLIP weights or exactly paired fine-tuned state are required; mapping is **BLOCKED_PROVENANCE**.
- **Evaluation transform / text / query / gallery.** `src/data_utils.py` and `src/validate.py` establish one query per annotation, full ordered validation split gallery, reference eligible. TargetPad / CLIP transform details and cache state are **UNKNOWN** here.
- **Scoring / metrics.** Source ranks full gallery without reference removal; macro category recall. Both map to `fashioniq_original_split`.
- **Preparation determinism / blockers / instrumentation.** Evaluation determinism is expected from fixed inputs but not source-verified. Do not automate checkpoint or asset selection. Instrument only after native command and model pairing prove parity.

## TG-CIR

- **Source / checkpoint.** [`iLearn-Lab/MM23-TG-CIR@65fa78eaf8cabe8197fcc22ccab42207c591bdee`](https://github.com/iLearn-Lab/MM23-TG-CIR/tree/65fa78eaf8cabe8197fcc22ccab42207c591bdee). README-linked `TG-CIR.zip` exists; FashionIQ member/hash mapping is **UNVERIFIED**.
- **Raw inputs.** FashionIQ train/validation captions and validation split JSON.
- **Offline preprocessing / generated artifacts.** `resized_image/{dress,shirt,toptee}` required; source generator, resize transform, and reproducibility are **UNKNOWN**. Correction dictionaries must be manually placed at expected `captions/` paths. `fashion_iq_data.json` and test query/target PKLs are caches; source does not validate freshness.
- **External assets.** OpenAI CLIP ViT-B/16 cache/weights required; exact provenance is unresolved.
- **Evaluation transform / caption pipeline.** Exact transform is **UNKNOWN**. Dictionary substitutions are source-specific; never reuse DCNet, LIMN, or iLearn artifacts without source evidence.
- **Query construction / gallery / reference policy.** `datasets.py:get_test_data` creates first-seen endpoint union; `test.py:test` masks source with `-10e10` before rank. `fashioniq_val_split`.
- **Scoring / metrics.** Macro recall evaluation; finer details **UNKNOWN**.
- **Preparation determinism / blockers / instrumentation.** **UNKNOWN**. Dictionary provenance/path, resized images, CLIP cache, runtime, and checkpoint mapping block execution. No auto-preprocessing.

## SPRC BLIP-2

- **Source / checkpoint.** [`chunmeifeng/SPRC@2935a5397732260d1db6fa577e5926f963e36f0f`](https://github.com/chunmeifeng/SPRC/tree/2935a5397732260d1db6fa577e5926f963e36f0f). Author OneDrive `sprc_fiq.pt` exists; checkpoint-to-BLIP-2/backbone mapping is **UNVERIFIED** and source loads non-strictly.
- **Raw inputs / generated artifacts.** Standard FashionIQ PNG validation layout. No generated artifact is established.
- **External assets.** Compatible BLIP-2/LAVIS model assets required; mapping/cache state is **BLOCKED_PROVENANCE**.
- **Evaluation transform / caption / query / gallery.** `src/data_utils.py` and `src/validate_blip.py` use full split order, one query per annotation, reference eligible. Exact transform and text internals are **UNKNOWN** here.
- **Scoring / metrics.** Macro category recall; `fashioniq_original_split`. Table 1 prints R@10 54.92 while category values average 54.72; retain discrepancy, do not replace either with a reproduction.
- **Preparation determinism / blockers / instrumentation.** **UNKNOWN**. Native command, compatible backbone, checkpoint pairing, and runtime block evaluation. No auto-setup.

## LIMN base iteration 0

- **Source / checkpoint.** [`iLearn-Lab/TPAMI24-LIMN@7d7bc9b116f594a65ac22457491edf28a88d3c3e`](https://github.com/iLearn-Lab/TPAMI24-LIMN/tree/7d7bc9b116f594a65ac22457491edf28a88d3c3e). Author Hub revision [`30560ad575a56c39cc39047d1a474269338c77c0`](https://huggingface.co/iLearn-Lab/TPAMI24-LIMN/tree/30560ad575a56c39cc39047d1a474269338c77c0) has base-iteration-0 category artifact hashes/metrics. Serialized replay remains unvalidated.
- **Raw inputs.** FashionIQ train/validation captions and validation split JSON.
- **Offline preprocessing / generated artifacts.** `resized_image/{dress,shirt,toptee}` required. Correction dictionaries from source `correct_files/` require manual placement. `train_data.json` and test query/target PKLs are cache artifacts without provenance/freshness validation.
- **External assets.** OpenCLIP ViT-L-14 DataComp weights required; **BLOCKED_EXTERNAL_ASSET**.
- **Evaluation transform / caption pipeline.** Exact transform and correction-file derivation are **UNKNOWN**.
- **Query construction / gallery / reference policy.** `LIMN/datasets.py` builds endpoint union; `LIMN/test.py:test` masks source before descending rank. `fashioniq_val_split`; do not generalize to LIMN+ or later self-training iterations.
- **Scoring / metrics.** Macro recall; replay details are **UNKNOWN**.
- **Preparation determinism / blockers / instrumentation.** **UNKNOWN**. No audited replay command, runtime, local data, or OpenCLIP asset. No automatic preparation or instrumentation.

## Exclusions

NEUCORE remains absent. Its forward/reverse caption-order queries double annotation query universe, so it is not a protocol fallback, artifact source, checkpoint source, or comparison cohort.

## Primary sources

- TG-CIR paper: [arXiv:2309.01366](https://arxiv.org/abs/2309.01366); official archive: [Google Drive](https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view).
- SPRC paper Table 1: [arXiv HTML](https://arxiv.org/html/2310.05473#S4.T1); author checkpoint link: [OneDrive](https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj).
- LIMN paper: [DOI 10.1109/TPAMI.2023.3346434](https://doi.org/10.1109/TPAMI.2023.3346434).
- CLVC-Net paper: [DOI 10.1145/3404835.3462967](https://doi.org/10.1145/3404835.3462967).
- DCNet paper PDF: [pinned source](https://github.com/ozmig77/dcnet/blob/68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb/DCNet_Kim2021.pdf).
- CLIP4Cir paper: [arXiv:2308.11485](https://arxiv.org/abs/2308.11485).
