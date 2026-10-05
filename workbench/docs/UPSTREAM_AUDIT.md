# Upstream and checkpoint audit

Audit date: 2026-10-05. This is source/checkpoint evidence, not reproduction evidence. No model artifact was downloaded and no official evaluation was run in this checkout. `sync_upstreams.py` checks out registry-recorded commits; pins were observed GitHub `main` HEADs at audit time.

| Model | Source pin | Checkpoint evidence | Training noise | Native protocol | State |
| --- | --- | --- | --- | --- | --- |
| CSMCIR | `qzp2018/CSMCIR@774f94e2076ff17ea91703a6239d2a08f0e1a44e` | Author-linked `peng12138/CSMCIR`, `fashioniq_tuned_clip_best.pt` | 0% | original | URL recorded; evaluator command audited; not run |
| Air-Know | `iLearn-Lab/CVPR26-Air-Know@6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | Official iLearn-Lab Hub: `airknow_fiq_0.5.pt`, `airknow_fiq_0.8.pt` | 50%, 80% | original | No clean checkpoint; command not audited |
| ConeSep | `iLearn-Lab/CVPR26-ConeSep@bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | Official iLearn-Lab Hub: `ConeSep-FIQ-N0.2.pt`, `N0.5.pt`, `N0.8.pt` | 20%, 50%, 80% | original | No clean checkpoint; command not audited |
| PTHA + MTST | unavailable | No author-linked FashionIQ fine-tuned weight verified | n/a | n/a | `BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` |
| HABIT | `iLearn-Lab/AAAI26-HABIT@6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | Official iLearn-Lab Hub: `HABIT-FIQ_N0.2.pt`, `N0.5.pt`, `N0.8.pt` | 20%, 50%, 80% | original | No clean checkpoint; command not audited |
| INTENT | `iLearn-Lab/AAAI26-INTENT@2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` | Official iLearn-Lab Hub: `intent_fiq_0.2.pt`, `0.5.pt`, `0.8.pt` | 20%, 50%, 80% | original | No clean checkpoint; command not audited |
| HINT | `iLearn-Lab/ICASSP26-HINT@bec50b6c8c19111893b617979502b948d1cea5b2` | Official iLearn-Lab Hub: `fashioniq.pt` | 0% | val | URL recorded; command not audited |
| ENCODER | `iLearn-Lab/AAAI25-ENCODER@29a2a31d6a56f677bf450c3be7cdaef423fb7018` | Official README Google Drive folder; FashionIQ filename metadata known | 0% | val | direct file URL/hash unresolved; command audited; not run |
| PAIR | `iLearn-Lab/ICASSP25-PAIR@bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` | Existing audit names `pair-B1.pt`, `pair-B2.pt`; author source/mapping unverified | unknown | val | `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` |

## Protocol observations

CSMCIR uses the complete ordered FashionIQ validation split gallery and leaves reference eligible. ENCODER/PAIR default `val-split` uses first-seen validation reference/target union and removes reference before recall. HINT uses the same union/reference-removal behavior. Local protocol mapping is documented in [PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md).

## CSMCIR validation requirements

Pinned local `CSMCIR@774f94e2076ff17ea91703a6239d2a08f0e1a44e` dispatches `validate_blip_csmcir.py:main` to `blip_validate_fashioniq`. That function builds `FashionIQDataset('val', [category], 'relative'/'classic')` for `dress`, `toptee`, and `shirt`. `data_utils_csmcir.py:FashionIQDataset` reads the canonical `fashionIQ_dataset` base layout and `<FASHIONIQ_ROOT>/qwen_captions/{dress,shirt,toptee}_cot_val.json` through that link. Its `extract_index_blip_caption_features` call (`validate_blip_csmcir.py:612`; `utils_csmcir.py:97-103`) also reads source-root `COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json`.

Official validation is fixed-root/cwd-sensitive and has no ordinary dataset-root CLI. `prepare_dataset.py --model csmcir` validates the base FashionIQ layout, the three dataset-root Qwen files, the three source-root COT files, and safely creates/checks `<third-party-root>/CSMCIR/fashionIQ_dataset -> <FASHIONIQ_ROOT>`. It does not fabricate assets. The evaluator emits aggregate metrics only; it cannot create per-query schema-v2 results. Future observation-only ranking export remains separate and must follow proven official aggregate-metric parity.

## ENCODER limitations

ENCODER's audited command uses `val-split` and requires a FashionIQ checkpoint plus upstream-root `open_clip_pytorch_model.bin`. `evaluate_model.py` calls `open_clip.create_model_and_transforms('ViT-B-32', pretrained='./open_clip_pytorch_model.bin')`; checkpoint alone is insufficient. The official Google Drive folder is recorded, but a direct FashionIQ checkpoint file URL, official checkpoint hash, and local artifact remain unresolved. Do not invent/download-substitute any of these assets during no-install workbench operation.

## Provenance policy

Registry endpoints are official/author-linked only. `expected_sha256` stays `null` unless upstream publishes a hash. A locally computed hash belongs only in ignored `artifacts/checkpoints/download_manifest.json`; it does not upgrade registry provenance. Paper scores are sanity references, never local reproductions. All listed runs remain `NOT_RUN` or blocked until verified evidence changes.
