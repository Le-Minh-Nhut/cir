# Upstream and checkpoint audit

Audit date: 2026-10-05. This is source/checkpoint evidence, not reproduction evidence. No model artifact was downloaded and no official evaluation was run in this checkout. `sync_upstreams.py` checks out registry-recorded commits; pins were observed GitHub `main` HEADs at audit time.

| Model | Source pin | Checkpoint evidence | Training noise | Literature label | Exact native protocol | State |
| --- | --- | --- | --- | --- | --- | --- |
| CSMCIR | `qzp2018/CSMCIR@774f94e2076ff17ea91703a6239d2a08f0e1a44e` | Author-linked `peng12138/CSMCIR`, `fashioniq_tuned_clip_best.pt` | 0% | original | `fashioniq_original_split` | URL recorded; evaluator command audited; not run |
| Air-Know | `iLearn-Lab/CVPR26-Air-Know@6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | Official iLearn-Lab Hub: `airknow_fiq_0.5.pt`, `airknow_fiq_0.8.pt` | 50%, 80% | original | `fashioniq_full_gallery_ref_excluded` | No clean checkpoint; command not audited |
| ConeSep | `iLearn-Lab/CVPR26-ConeSep@bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | Official iLearn-Lab Hub: `ConeSep-FIQ-N0.2.pt`, `N0.5.pt`, `N0.8.pt` | 20%, 50%, 80% | original | `fashioniq_full_gallery_ref_excluded` | No clean checkpoint; command not audited |
| PTHA + MTST | unavailable | No author-linked FashionIQ fine-tuned weight verified | n/a | n/a | n/a | `BLOCKED_NO_VERIFIED_FASHIONIQ_CHECKPOINT` |
| HABIT | `iLearn-Lab/AAAI26-HABIT@6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | Official iLearn-Lab Hub: `HABIT-FIQ_N0.2.pt`, `N0.5.pt`, `N0.8.pt` | 20%, 50%, 80% | original | `fashioniq_full_gallery_ref_excluded` | No clean checkpoint; command not audited |
| INTENT | `iLearn-Lab/AAAI26-INTENT@2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` | Official iLearn-Lab Hub: `intent_fiq_0.2.pt`, `0.5.pt`, `0.8.pt` | 20%, 50%, 80% | original | `fashioniq_full_gallery_ref_excluded` | No clean checkpoint; command not audited |
| HINT | `iLearn-Lab/ICASSP26-HINT@bec50b6c8c19111893b617979502b948d1cea5b2` | Official iLearn-Lab Hub: `fashioniq.pt` | 0% | val | `fashioniq_val_split` | URL recorded; command not audited |
| ENCODER | `iLearn-Lab/AAAI25-ENCODER@29a2a31d6a56f677bf450c3be7cdaef423fb7018` | Official README Google Drive folder; FashionIQ filename metadata known | 0% | val | `fashioniq_val_split` | direct file URL/hash unresolved; command audited; not run |
| PAIR | `iLearn-Lab/ICASSP25-PAIR@bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` | Existing audit names `pair-B1.pt`, `pair-B2.pt`; author source/mapping unverified | unknown | val | `fashioniq_val_split` | `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` |

## Protocol observations

CSMCIR uses complete ordered FashionIQ validation gallery and leaves reference eligible (`fashioniq_original_split`; literature label `original`). Air-Know, ConeSep, HABIT, and INTENT use that same gallery but mask reference before recall (`fashioniq_full_gallery_ref_excluded`; literature label `original`). ENCODER/PAIR default `val-split` uses first-seen validation reference/target union and removes reference before recall. HINT uses same union/reference-removal behavior. Local protocol mapping is documented in [PROTOCOL_AUDIT.md](PROTOCOL_AUDIT.md).

## iLearn FashionIQ assets and evaluator semantics

Pinned ENCODER, HINT, PAIR, Air-Know, ConeSep, HABIT, and INTENT sources all open category-local `captions/correction_dict_{dress,shirt,toptee}.json` for FashionIQ train and validation caption construction. Their shared transformation is lowercase, ASCII `string.punctuation` to spaces, whitespace tokenization, exact dictionary lookup, and source-order `caption_1 and caption_2` composition. The dictionaries are not tracked in inspected iLearn trees.

`iLearn-Lab/SIGIR24-DQU-CIR@49f53992447a9a84b897d9b2e2f2c91dd2cc8788` tracks similarly named dictionaries at `data/FashionIQ/captions/`: dress blob `249af320785363d150d9f0f73c7bd3b3f6a2c7ec`, shirt blob `e6e4adfd3fa3dbde2f4f0c760498d821dea192af`, and toptee blob `0fcbc39a9a9a2b24ec99bd9358d100eda2798831`. These are Git blob IDs, not file SHA-256 values. No inspected iLearn source identifies DQU-CIR as their source. Workbench marks dictionary provenance **UNVERIFIED** and only blocks a native iLearn evaluation when local files are absent; it never downloads or substitutes DQU content.

Air-Know, ConeSep, HABIT, and INTENT each select `original-split` in their FashionIQ dataset constructor, enumerate full validation split order, and set reference similarity to `-10e10` before FashionIQ R@1/R@10/R@50. HINT constructs the pair-union validation gallery and removes reference before R@1/R@10/R@50. ENCODER and PAIR use the pair-union/reference-removal variant described in the protocol audit. These static evaluator observations do not make their commands, checkpoints, environments, or reproductions ready.

## Legacy source audit

| Model | Source pin | Native evaluator | Checkpoint / command / assets | Readiness |
| --- | --- | --- | --- | --- |
| CLVC-Net | `iLearn-Lab/SIGIR21-CLVC-Net@bd9b6889489806cb60676597880f91d918b279cb` | `fashioniq_val_split`: endpoint union, source mask | Separate local/global models; Drive mapping unavailable. Pinned `test.py` has no replay CLI. ResNet-50 asset unresolved. | `BLOCKED_CHECKPOINT_MAPPING` / source entrypoint |
| DCNet | `ozmig77/dcnet@68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` | `fashioniq_full_gallery_ref_excluded`: split order, dual reference mask, log-softmax fusion | Audited `--resume` directory with `config.json` + `trained_model.pth`; standard ResNet-50 asset unresolved; Drive hash/content mapping unresolved. | command audited; otherwise blocked |
| Combiner RN50x4 noft | `ABaldrati/CLIP4Cir@dfed9f748a8a4d05abf613164e06d59b22dbdf13` | `fashioniq_original_split`: full split, reference eligible | Combiner state mapping and RN50x4 base CLIP asset unresolved. | `BLOCKED_CHECKPOINT_MAPPING` |
| CLIP4Cir RN50x4 fullft | Same CLIP4Cir pin | Same full-gallery evaluator | Requires separate Combiner and CLIP states; author pairing unresolved. | `BLOCKED_CHECKPOINT_BUNDLE` |
| TG-CIR | `iLearn-Lab/MM23-TG-CIR@65fa78eaf8cabe8197fcc22ccab42207c591bdee` | `fashioniq_val_split`: endpoint union, source mask | Archive member mapping unresolved; pinned `test.py` has no replay CLI; ViT-B/16 asset unresolved. | `BLOCKED_COMMAND` |
| SPRC BLIP-2 | `chunmeifeng/SPRC@2935a5397732260d1db6fa577e5926f963e36f0f` | `fashioniq_original_split`: full split, reference eligible | `src/blip_validate.py` CLI shape audited; `sprc_fiq.pt` model/backbone/class mapping and LAVIS asset unresolved. | `BLOCKED_COMMAND_AND_CHECKPOINT_MAPPING` |
| LIMN base iteration 0 | `iLearn-Lab/TPAMI24-LIMN@7d7bc9b116f594a65ac22457491edf28a88d3c3e` | `fashioniq_val_split`: endpoint union, source mask | Exact author-hash-verified dress/shirt/toptee whole-model bundle; workbench wrapper calls pinned native evaluator; runtime unexecuted. | `BLOCKED_RUNTIME_PREREQUISITES` |

All seven command, checkpoint, preparation, environment, and runtime-asset records are independently tracked in registry and doctor. See [LEGACY_SOURCE_AUDIT.md](LEGACY_SOURCE_AUDIT.md) for source-level evidence.

NEUCORE remains out of registry scope. Its doubled forward/reverse caption queries are neither a fallback nor evidence for any model above.

## CSMCIR validation requirements

Pinned local `CSMCIR@774f94e2076ff17ea91703a6239d2a08f0e1a44e` dispatches `validate_blip_csmcir.py:main` to `blip_validate_fashioniq`. That function builds `FashionIQDataset('val', [category], 'relative'/'classic')` for `dress`, `toptee`, and `shirt`. `data_utils_csmcir.py:FashionIQDataset` reads the canonical `fashionIQ_dataset` base layout and `<FASHIONIQ_ROOT>/qwen_captions/{dress,shirt,toptee}_cot_val.json` through that link. Its `extract_index_blip_caption_features` call (`validate_blip_csmcir.py:612`; `utils_csmcir.py:97-103`) also reads source-root `COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json`.

`prepare_dataset.py --model csmcir` validates base FashionIQ and creates/checks `<third-party-root>/CSMCIR/fashionIQ_dataset -> <FASHIONIQ_ROOT>` before auxiliary acquisition. Strict `doctor.py --scope real --model csmcir` remains final runtime gate for link, Qwen, COT, checkpoint, source pin, evaluator, and dataset.

## CSMCIR auxiliary asset audit

**COT source: NOT FOUND.** Pinned author [repository tree](https://api.github.com/repos/qzp2018/CSMCIR/git/trees/774f94e2076ff17ea91703a6239d2a08f0e1a44e?recursive=1), exact-path GitHub history requests, releases/tags, README, and author-linked [Hugging Face revision tree](https://huggingface.co/api/models/peng12138/CSMCIR/tree/cf0c19bb346266c295b4b5772ebf03bd1c4f0467?recursive=true&expand=false) were checked. None contains `COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json`; exact GitHub paths return 404 and path-history requests return `[]`. No automatic COT URL is registered. Final preflight reports `BLOCKED: required COT_ours2 captions have no verified automatic acquisition source` until files are manually placed with known provenance.

**Qwen source: verified immutable revision.** Pinned upstream [README](https://raw.githubusercontent.com/qzp2018/CSMCIR/774f94e2076ff17ea91703a6239d2a08f0e1a44e/README.md) directs Qwen captions to `https://huggingface.co/peng12138/CSMCIR`. Immutable author revision `cf0c19bb346266c295b4b5772ebf03bd1c4f0467` contains `fashioniq_qwen_captions/qwen_captions/{dress,shirt,toptee}_cot_val.json`. Registry URLs use `/resolve/cf0c19bb346266c295b4b5772ebf03bd1c4f0467/`; upstream publishes no file SHA-256, so local manifest hashes remain local evidence only.

Official validation is fixed-root/cwd-sensitive and has no ordinary dataset-root CLI. The evaluator emits aggregate metrics only; it cannot create per-query schema-v2 results. Future observation-only ranking export remains separate and must follow proven official aggregate-metric parity.

## ENCODER limitations

ENCODER's audited command uses `val-split` and requires a FashionIQ checkpoint plus upstream-root `open_clip_pytorch_model.bin`. `evaluate_model.py` calls `open_clip.create_model_and_transforms('ViT-B-32', pretrained='./open_clip_pytorch_model.bin')`; checkpoint alone is insufficient. Its evaluator imports `datasets1`, but this file is absent from pinned source. Workbench blocks execution until the author-provenanced dependency is present. The official Google Drive folder is recorded, but a direct FashionIQ checkpoint file URL, official checkpoint hash, and local artifact remain unresolved. Do not invent/download-substitute any of these assets during no-install workbench operation.

## CSMCIR: the evaluator cannot import (supersedes the caption blockers as the first failure)

Pinned `CSMCIR@774f94e2076ff17ea91703a6239d2a08f0e1a44e` cannot load its own model package.
`src/validate_blip_csmcir.py:12` does `from lavis.models import load_model_and_preprocess`, and
`src/lavis/models/__init__.py` then executes **98 unguarded** `from … import …` statements at lines
44-151, every one naming a module that is absent from the repository:

```
$ python3 -c "...import re; count lines whose target .py is missing..."
98 lines, 44 .. 151
```

None of lines 44-151 is inside a `try`, so the first missing module raises `ModuleNotFoundError` and
the import of the entrypoint fails. The class CSMCIR actually needs *is* present and *is* registered
(`blip2_qformer_cir_align_prompt_csmcir.py:27`,
`@registry.register_model("blip2_cir_align_prompt_csmcir")`) and *is* imported at line 152 — but 152
runs after the 98 failures, so the registration never happens and
`load_model_and_preprocess(name="blip2_cir_align_prompt_csmcir")` cannot resolve either. Confirmed
against the public revision: the same lines are present and the module returns HTTP 404. The
one-line fix and its verification are recorded in `workbench/patches/csmcir.md`, and are deliberately
not applied automatically, because editing the model package would change which classes LAVIS
registers.

So CSMCIR is `BLOCKED_SOURCE_UNIMPORTABLE`. The Qwen and `COT_ours2` requirements below are real but
strictly downstream of this.

## LAVIS-hosted evaluators: the cache root is hard-coded to an author machine path

Every LAVIS-based checkout in the registry — CSMCIR, SPRC, and the five iLearn models — vendors a
`lavis/configs/default.yaml` whose `cache_root` is an absolute path on the author's cluster, not a
user cache:

| Model | `cache_root` in the pinned `lavis/configs/default.yaml` |
| --- | --- |
| CSMCIR | `/raid/temp/home/.cache/lavis` |
| SPRC | `/raid/temp/home/.cache/lavis` |
| HABIT | `/export/home/.cache/lavis` |
| AirKnow | `/export/home/.cache/lavis` |
| ConeSep | `/export/home/.cache/lavis` |
| INTENT | `/export/home/.cache/lavis` |
| HINT | `/export/home/.cache/lavis` |

LAVIS resolves it through `registry.get_path("cache_root")` (`src/lavis/common/utils.py:47,
datasets/builders/base_dataset_builder.py:108`), which reads the config file, **not** an environment
variable. So the location the workbench records for LAVIS weights (`~/.cache/lavis`, from the
registry's `user_cache` scope) is not where these evaluators look, and on a fresh host they would
attempt to download into a path that usually does not exist.

Two consequences, both real:

1. On a host where `/raid/temp/home` or `/export/home` does not exist, the LAVIS download fails, and
   the failure surfaces as a download error deep inside the evaluator rather than as a missing asset.
2. Where it does exist, the evaluator writes there instead of the memory not tracked by the
   workbench's provenance records.

The honest handling is the same as for the other source defects: report it rather than patch it.
Editing `lavis/configs/default.yaml` modifies the official implementation and would change where the
evaluator's weights come from, so it is left as an operator step. The fix is one line — set
`cache_root` to the path the registry records — and being a config file rather than code, it is a
far smaller deviation than the CSMCIR import defect.

CSMCIR and SPRC are already blocked for other reasons, so this only changes what an operator must do
for a model that would otherwise run. It is recorded here so that the first iLearn-family experiment
does not spend its debugging time on a download landing in the wrong directory.

## CSMCIR checkpoint hash: author-published after all

The row above records `expected_sha256: null` on the grounds that upstream publishes no hash. For the
checkpoint that is not quite right: Hugging Face stores large files through Git-LFS and publishes the
content SHA-256 as the pointer `oid`. That convention was verified empirically before being relied on
— downloading one 14,431,718-byte blob from this same repository and confirming its
`sha256sum` equals the `lfs.oid` reported by the API. The checkpoint therefore has an authoritative
hash, now recorded in the registry:

| File | Bytes | SHA-256 (author LFS oid) |
| --- | --- | --- |
| `fashioniq_tuned_clip_best.pt` | 2720615859 | `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e` |

The Qwen caption files are *not* LFS objects (their `oid` is a 40-hex Git blob id, not a SHA-256), so
their registry digests remain locally computed evidence, as the provenance policy states.

## GPU requirements: `cuda: UNKNOWN` must not read as "no GPU needed"

`cuda: UNKNOWN` in the registry means *the upstream README declares no version*. It was being read by
`runtime._requires_gpu` as *no GPU is required*, which suppressed the GPU-verification gate for
evaluators that call `.cuda()` unconditionally. Three were affected and now declare a version:

| Model | Evidence | Was | Now |
| --- | --- | --- | --- |
| ENCODER | `evaluate_model.py:64` `Encoder_model.cuda()`, no `is_available()` | UNKNOWN | 12.6 |
| CLVC-Net | `test.py:28,47` `.cuda()` per batch, no guard | UNKNOWN | 10.2 |
| TG-CIR | `test.py:31,47` `.cuda()` per batch, no guard | UNKNOWN | 11.0 |

The remaining `UNKNOWN` entries (CSMCIR, DCNet, Combiner, CLIP4Cir, SPRC) were re-checked the same
way and genuinely guard on `torch.cuda.is_available()`, so `UNKNOWN` is correct for them.

## iLearn correction dictionaries: confirmed required, confirmed unpublished

The section above already records that seven iLearn-family sources open
`captions/correction_dict_{dress,shirt,toptee}.json` and that the dictionaries are not tracked in the
inspected trees. That conclusion was re-verified at runtime: for any `FashionIQDataset`, the
FashionIQ branch of `extract_index_blip_caption_features` reads the dictionary before it touches a
single image, so the file is required by the *query* path, not only by training. The canonical
FashionIQ distribution does not contain these files and no iLearn repository publishes them
(`iLearn-Lab/ICASSP26-HINT`, `AAAI26-HABIT`, `AAAI25-ENCODER` trees contain none producing them).
They remain `EXTERNAL_AUTHOR_ASSET_BLOCKED` with **UNVERIFIED** provenance. This blocks none of them
today, because each is already `command not runnable` — none of the five iLearn evaluators
(`test_BLIP2.py`, `test.py` ×4) has an `ArgumentParser` or a `__main__`, so no adapter can invoke them
with arguments and each correctly reports the blocker instead of a false ready.

## Pinned-repo dependency method

An asset is required only when a **reachable call path from the registered entrypoint** opens it at
runtime. A filename occurring anywhere in the tree — a class never instantiated
(`FashionIQDataset_imageedit`), a commented-out branch, a training-only script, an abandoned
experiment — does not make it required. Conversely, a file the active evaluator opens but the
registry does not check is a false negative and equally release-blocking. The audit files under
`workbench/registry/assets/` and `workbench/registry/audits/` are **documentation**; only
`workbench/registry/auxiliary_assets.yaml` drives preflight, download and fingerprinting, and it
currently declares 14 assets. The false positives catalogued in those audit files (iLearn-generated
test PKLs, `datasets1`, `MHTransformer_rev`, DCNet's raw GloVe and `dict.attr.json`) therefore do not
reach production; they are recorded here so they are not re-imported later.



## Provenance policy

Registry endpoints are official/author-linked only. `expected_sha256` stays `null` unless upstream publishes a hash. A locally computed hash belongs only in ignored `artifacts/checkpoints/download_manifest.json`; it does not upgrade registry provenance. Paper scores are sanity references, never local reproductions. All listed runs remain `NOT_RUN` or blocked until verified evidence changes.
