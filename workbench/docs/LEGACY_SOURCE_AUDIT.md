# Legacy FashionIQ source audit

Audit date: 2026-10-08. Source/checkpoint evidence only. No source clone, checkpoint, dataset, environment, or GPU evaluation was created or run. This document does not report reproductions.

| Model | Immutable author source | Native evaluator evidence | Checkpoint / score evidence | Registry verdict |
| --- | --- | --- | --- | --- |
| CLVC-Net | [`iLearn-Lab/SIGIR21-CLVC-Net@bd9b6889489806cb60676597880f91d918b279cb`](https://github.com/iLearn-Lab/SIGIR21-CLVC-Net/tree/bd9b6889489806cb60676597880f91d918b279cb) | `datasets.py:FashionIQ.get_test_targets()` builds source-order reference/target union; `test.py:test()` masks source before descending rank. One query per annotation, captions as `<BOS> c1 <AND> c2 <EOS>`. | Author Drive link exists; filename/category/hash/load command and paper metrics unverified. | `fashioniq_val_split`; `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`. |
| DCNet | [`ozmig77/dcnet@68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`](https://github.com/ozmig77/dcnet/tree/68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb) | `CE_dataset.py` consumes one generated GloVe PKL row per annotation. `TrainerJoint._valid_epoch` ranks ordered `val_trg`, masks reference in both branches, then sums branch `log_softmax` values. | Author Drive folder exists; exact config/weight/hash mapping unverified. Paper validation rows are not artifact evidence. | Protocol unavailable pending independent proof that split JSON is complete validation gallery; blocked. |
| Combiner RN50x4 noft | [`ABaldrati/CLIP4Cir@dfed9f748a8a4d05abf613164e06d59b22dbdf13`](https://github.com/ABaldrati/CLIP4Cir/tree/dfed9f748a8a4d05abf613164e06d59b22dbdf13) | `src/data_utils.py` builds full split-order gallery; `src/validate.py` ranks it without reference removal; one query per annotation. | Author Drive folder exists. `noft` artifact meaning/file/pairing/hash and requested score are unverified. Table 9 RN50x4 value cannot substitute. | `fashioniq_original_split`; blocked. |
| CLIP4Cir RN50x4 fullft | Same CLIP4Cir pin | Same full-gallery/reference-eligible evaluator. | `fullft` artifact meaning/file/pairing/hash and requested score are unverified. | `fashioniq_original_split`; blocked. |
| TG-CIR | [`iLearn-Lab/MM23-TG-CIR@65fa78eaf8cabe8197fcc22ccab42207c591bdee`](https://github.com/iLearn-Lab/MM23-TG-CIR/tree/65fa78eaf8cabe8197fcc22ccab42207c591bdee) | `datasets.py:get_test_data` builds source-order endpoint union; `test.py:test` sets source score to `-10e10` before descending rank. One query per annotation. Loader needs resized images and dictionaries manually installed under `captions/`; cache freshness is not validated upstream. | Paper macro R@10/R@50: 51.32/73.09. README-linked `TG-CIR.zip` exists, but its FashionIQ member/hash mapping is unverified. | `fashioniq_val_split`; blocked. |
| SPRC BLIP-2 | [`chunmeifeng/SPRC@2935a5397732260d1db6fa577e5926f963e36f0f`](https://github.com/chunmeifeng/SPRC/tree/2935a5397732260d1db6fa577e5926f963e36f0f) | `src/data_utils.py` uses full split order; `src/validate_blip.py` does not remove reference and macro-averages categories. | Table prints R@10 54.92 while category rows average 54.72; R@50 74.97. Preserve discrepancy. Linked `sprc_fiq.pt` BLIP-2 model/backbone mapping is unverified and source loads non-strictly. | `fashioniq_original_split`; blocked. |
| LIMN base iteration 0 | [`iLearn-Lab/TPAMI24-LIMN@7d7bc9b116f594a65ac22457491edf28a88d3c3e`](https://github.com/iLearn-Lab/TPAMI24-LIMN/tree/7d7bc9b116f594a65ac22457491edf28a88d3c3e) | `LIMN/datasets.py` builds source-order endpoint union; `LIMN/test.py:test` masks source before descending rank. One query per annotation. | Author Hub revision [`30560ad575a56c39cc39047d1a474269338c77c0`](https://huggingface.co/iLearn-Lab/TPAMI24-LIMN/tree/30560ad575a56c39cc39047d1a474269338c77c0) provides `0_{dress,shirt,toptee}_best_model.pt` hashes and metric JSONs. They are base iteration 0 only, author artifact-associated, not locally replayed. | `fashioniq_val_split`; `BLOCKED_NO_AUDITED_REPLAY_COMMAND`. |

## Native preparation boundaries

- CLVC-Net: raw train/val captions, split JSON, generated `resized_image/{category}` JPEGs, and possible pretrained ResNet cache. No author checkpoint evaluation command.
- DCNet: `cap.{category}.glove.val.pkl`, `resized_images`, split JSON, matching `config.json`/`trained_model.pth`; `process_cap.py` needs spaCy `en_vectors_web_lg` and NLTK `punkt`. Never regenerate or substitute these implicitly.
- CLIP4Cir and SPRC: standard FashionIQ PNG image layout. Their different upstream preprocessing/model assets are not interchangeable.
- TG-CIR and LIMN: resized JPEG layout. Both require correction dictionaries, but only their own pinned source supplies evidence for its files; do not copy assets across methods. LIMN also requires external OpenCLIP weights.

## Exclusions

NEUCORE is intentionally not registered. Its source doubles each annotation into forward/reverse caption-order queries, creating a different query universe. It is not a protocol fallback, asset source, checkpoint source, or comparison cohort for any entry here.

## Primary sources

- TG-CIR paper: [arXiv:2309.01366](https://arxiv.org/abs/2309.01366); official archive: [Google Drive](https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view).
- SPRC paper Table 1: [arXiv HTML](https://arxiv.org/html/2310.05473#S4.T1); author checkpoint link: [OneDrive](https://1drv.ms/u/s!Aj0q22vyiZbabnUya4mnufIBtYI?e=n4ZVKj).
- LIMN paper: [DOI 10.1109/TPAMI.2023.3346434](https://doi.org/10.1109/TPAMI.2023.3346434).
- CLVC-Net paper: [DOI 10.1145/3404835.3462967](https://doi.org/10.1145/3404835.3462967).
- DCNet paper PDF: [pinned source](https://github.com/ozmig77/dcnet/blob/68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb/DCNet_Kim2021.pdf).
- CLIP4Cir paper: [arXiv:2308.11485](https://arxiv.org/abs/2308.11485).
