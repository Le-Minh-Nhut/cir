# Vendored recovery — sixteen model entries, one row each

> **Superseded** by `FINAL_16_MODEL_RECOVERY_STATUS.md`, which carries the per-entry verification
> levels reached after the delivery defects were closed and the wrappers, parsers and adapters were
> finished. Kept for the vendoring provenance and the suggestion-disposition record it holds.

**Branch**: `research/cir-failure-workbench`. **Scope**: the upstream source of all sixteen
registered entries is now vendored as *tracked source* under `workbench/third_party/<source_dir>/`,
nested Git metadata is removed, the fifteen distinct upstreams are recorded with an exact commit and
a per-file digest manifest, and the model-specific fixes the source justified are applied and
recorded as `local_fixes`.

This document supersedes the recovery columns of `r8/R8_16_MODEL_STATUS.md`. That file's matrix said
"No run of any model has been executed on the target GPU in this pass, so no model is GPU-verified";
that is still true here, and the status vocabulary below keeps that distinction explicit rather than
folding a static result into an execution result.

## 1. How to read the columns

| column | meaning |
|---|---|
| `upstream_commit` | the exact commit the vendored tree was taken from, recorded in `provenance.json` |
| `nested_git_removed` | the nested `.git`/`.gitmodules` and committed bytecode cache were removed from the checkout |
| `local_fixes` | files that intentionally differ from the upstream commit, recorded in `provenance.json` |
| `checkpoint_variant` | which published checkpoint the entry evaluates; noise levels kept separate |
| `env` | whether the model's interpreter/dependency set is declared and provisioned |
| `native_eval` | whether the official evaluator can be invoked as published |
| `adapter` / `parser` | whether the workbench can build the command / read its output |
| `per_query` | whether a per-query artifact is implemented |
| `GPU` | whether the model has actually been run on a GPU **here** |

Legend for the status vocabulary: `FIXED_AND_VERIFIED` (source-level change proven by a runnable
check), `IMPLEMENTED_RUNTIME_PENDING`, `ASSET_SETUP_PENDING`, `CHECKPOINT_MAPPING_PENDING`,
`AUTHOR_INPUT_REQUIRED`, `EXPERIMENTAL_RECONSTRUCTION`, `GPU_VALIDATION_PENDING`,
`SOURCE_INSPECTION_ONLY`.

## 2. Matrix

| model_id | source_dir | upstream url | upstream_commit | nested_git_removed | local_fixes | checkpoint_variant | mapping | env | native_eval | adapter | parser | per_query | GPU | status | remaining blocker | next step |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `limn` | `LIMN` | https://github.com/iLearn-Lab/TPAMI24-LIMN | `7d7bc9b116f594a65ac22457491edf28a88d3c3e` | yes | none | `base_iter0_{dress,shirt,toptee}` (3 whole-model ckpts) | VERIFIED (Hub LFS oid) | declared | runnable (`LIMN/test.py`) | yes | yes | LIMN_QUERY_OBSERVER_V1 | **no** | `GPU_VALIDATION_PENDING` | 3 ckpts ≈5.4 GB unstaged; no torch on this host | stage ckpts, run `./pc.sh` on the 5070 Ti |
| `encoder` | `ENCODER` | https://github.com/iLearn-Lab/AAAI25-ENCODER | `29a2a31d6a56f677bf450c3be7cdaef423fb7018` | yes | `datasets1.py` (added), `evaluate_model.py` | author FashionIQ ckpt | PENDING | declared | runnable after fix | yes | no | none | **no** | `IMPLEMENTED_RUNTIME_PENDING` | LAION ViT-B/32 `open_clip_pytorch_model.bin` now registered, unstaged; ckpt unstaged | download backbone + ckpt, then `evaluate_model.py --dataset fashioniq` |
| `hint` | `HINT` | https://github.com/iLearn-Lab/ICASSP26-HINT | `bec50b6c8c19111893b617979502b948d1cea5b2` | yes | none | `fashioniq.pt` (`iLearn-Lab/HINT`) | VERIFIED (2 724 451 182 B, sha256 `bf9c72d4…`) | declared | callable, no CLI | inherited | no | none | **no** | `IMPLEMENTED_RUNTIME_PENDING` | 2.7 GB ckpt unstaged; replay module reviewed but not placed | place `workbench/replay/hint.py`, stage ckpt |
| `pair` | `PAIR` | https://github.com/iLearn-Lab/ICASSP25-PAIR | `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` | yes | `models/MHTransformer_rev.py` (reconstructed) | `pair-B1.pt` / `pair-B2.pt` | **UNRESOLVED** (no dataset/category label published) | declared | callable, no CLI | inherited | no | none | **no** | `CHECKPOINT_MAPPING_PENDING` | Hub publishes both files with no dataset/category indication | ask the author, or identify B1/B2 from the checkpoint's own config |
| `clvc_net` | `CLVCNet` | https://github.com/iLearn-Lab/SIGIR21-CLVC-Net | `bd9b6889489806cb60676597880f91d918b279cb` | yes | `resnet.py` (import compat) | `CLVCNET.zip` (Drive) | PENDING | declared | callable, no CLI | inherited | no | none | **no** | `CHECKPOINT_MAPPING_PENDING` | archive interior uninspected; needs 6 whole-model files (2 per category) | inspect the archive, then wire a replay wrapper |
| `tgcir` | `TGCIR` | https://github.com/iLearn-Lab/MM23-TG-CIR | `65fa78eaf8cabe8197fcc22ccab42207c591bdee` | yes | none | `TG-CIR.zip` (Drive) | PENDING | `clip==0.2.0` pin is WRONG — needs `openai/CLIP` | callable, no CLI | inherited | no | none | **no** | `CHECKPOINT_MAPPING_PENDING` | archive member mapping unknown; no CLI/loader in pinned `test.py` | replace the clip pin with a git revision, inspect the ZIP |
| `habit` | `HABIT` | https://github.com/iLearn-Lab/AAAI26-HABIT | `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | yes | none | `HABIT-FIQ_N0.{2,5,8}.pt` | VERIFIED (3 sha256) | declared | callable, no CLI | inherited | no | none | **no** | `ASSET_SETUP_PENDING` | no clean-trained ckpt published (20/50/80 % only); evaluate *noisy* weights on clean val, labelled | stage a ckpt, keep training vs evaluation noise separate |
| `intent` | `INTENT` | https://github.com/iLearn-Lab/AAAI26-INTENT | `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` | yes | none | `intent_fiq_0.{2,5,8}.pt` | VERIFIED (3 sha256) | declared | callable, no CLI | inherited | no | none | **no** | `ASSET_SETUP_PENDING` | same as HABIT: no clean ckpt published | stage a ckpt, label the noise condition |
| `conesep` | `ConeSep` | https://github.com/iLearn-Lab/CVPR26-ConeSep → resolves to **Lee-zixu/ConeSep** | `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | yes | none | `ConeSep-FIQ-N0.{2,5,8}.pt` | VERIFIED (3 sha256) | declared | runnable import path | inherited | no | none | **no** | `ASSET_SETUP_PENDING` | resolve redirect: recorded URL is the redirect target | stage a ckpt |
| `airknow` | `AirKnow` | https://github.com/iLearn-Lab/CVPR26-Air-Know | `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | yes | `data_utils.py`, `lavis/models/__init__.py` | `airknow_fiq_0.{5,8}.pt` | VERIFIED (2 sha256) | declared | previously unimportable, imports now | inherited | no | none | **no** | `IMPLEMENTED_RUNTIME_PENDING` | no `__main__`/checkpoint loader in `test_BLIP2.py`; image ext mismatch resolved | add the replay wrapper; decide the `model.train()` variant |
| `dcnet` | `DCNet` | https://github.com/ozmig77/dcnet | `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` | yes | none | `trained_model.pth` (Drive) | PENDING | declared | CLI exists (`test.py --resume`) | yes | no | none | **no** | `EXPERIMENTAL_RECONSTRUCTION` | `cap.*.glove.val.pkl` caches unpublished; OOV vectors are unseeded | publish the caches, or accept the seeded reconstruction as an explicit variant |
| `csmcir` | `CSMCIR` | https://github.com/qzp2018/CSMCIR | `774f94e2076ff17ea91703a6239d2a08f0e1a44e` | yes | none (approved patch is separate) | `fashioniq_tuned_clip_best.pt` (2 720 615 859 B, sha256 `c8812071…`) | VERIFIED | declared | needs the one approved patch | yes | yes | none | **no** | `IMPLEMENTED_RUNTIME_PENDING` | 2.7 GB ckpt + BLIP-2/EVA-ViT-g caches unstaged; CoT captions now have a source | stage assets; the CoT “captions unavailable” blocker is WITHDRAWN |
| `combiner_rn50x4_noft` | `CLIP4Cir` | https://github.com/ABaldrati/CLIP4Cir | `dfed9f748a8a4d05abf613164e06d59b22dbdf13` | yes | none | RN50x4 Combiner (FashionIQ noft) | PENDING | declared | CLI exists (`src/validate.py`) | yes | yes | none | **no** | `CHECKPOINT_MAPPING_PENDING` | 8.6 GB release archive (5 parts, digests published) not downloaded | reconstruct the archive, extract members, load |
| `clip4cir_rn50x4_fullft` | `CLIP4Cir` | https://github.com/ABaldrati/CLIP4Cir | `dfed9f748a8a4d05abf613164e06d59b22dbdf13` | yes | none | RN50x4 fine-tuned CLIP + Combiner (FashionIQ fullft) | PENDING | declared | CLI exists (`src/validate.py`) | yes | yes | none | **no** | `CHECKPOINT_MAPPING_PENDING` | same archive; both members must come from the same configuration | extract from the same archive, verify pairing |
| `sprc` | `SPRC` | https://github.com/chunmeifeng/SPRC | `2935a5397732260d1db6fa577e5926f963e36f0f` | yes | none | `sprc_fiq.pt` (ViT-g) / `sprc_fiq_vitl.pt` | PENDING | declared | CLI exists | inherited | yes | none | **no** | `AUTHOR_INPUT_REQUIRED` | OneDrive 403; Baidu needs interactive auth; only the ViT-L variant is publicly reachable and switching backbone is a scientific variant change | author mirror, or record the ViT-L variant explicitly |
| `ptha_mtst` | `MTST_PTHA` | https://github.com/zyinan99/MTST_PTHA | `6a02ee4763fc2260b6abfb18be78853facb91d86` | yes | `MTST_PTHA/src/data_utils.py`, `MTST_PTHA/src/lavis/models/__init__.py` | none | **none published** | declared | package now imports; no CLI | **none** | no | none | **no** | `AUTHOR_INPUT_REQUIRED` | no FashionIQ checkpoint is published (Hub has CIRR only) | author must publish FashionIQ weights |

**GPU column is `no` for every entry and that is the honest reading**: this host is an RTX 3050
Laptop (4 GiB, sm_86) with no torch installed anywhere, so nothing has been executed on a GPU. Every
non-LIMN entry is additionally `GPU_VALIDATION_PENDING` on the target 5070 Ti.

## 3. What changed in this pass

1. **Vendored source.** Fifteen distinct upstreams, sixteen entries (`CLIP4Cir` serves two), committed
   as plain files with `provenance.json` each. `vendor_upstreams.py` performs the clone → pin →
   copy → metadata-removal sequence and refuses to clobber a divergent local file silently.
2. **Integrity model.** A `git status` check is impossible against a tree with no `.git`, and was
   weaker anyway (it cannot distinguish a reviewed patch from a post-pin edit). `vendored_source.py`
   re-hashes the recorded manifest instead, and treats registry-declared runtime artifacts and
   recorded local fixes as approved.
3. **Six justified source fixes**, each recorded in `local_fixes`:
   - `ENCODER/datasets1.py` — the module `evaluate_model.py` imports does not exist upstream; the
     local `datasets.py` defines every name it uses. Both the import and an alias module were added
     (`workbench/backend/preflight.py:212` already required `datasets1.py`).
   - `ENCODER/evaluate_model.py` — `import datasets as datasets1`.
   - `CLVCNet/resnet.py` — `torchvision.models.utils.load_state_dict_from_url` moved to `torch.hub`;
     the fallback preserves the identical ResNet-50 weights URL.
   - `PAIR/models/MHTransformer_rev.py` — reconstructed. Verified line-by-line: after removing the
     ordinary-attention class from `MHTransformer.py` and renaming, exactly **two** lines differ (the
     class name and its instantiation). `MultiHeadAttentionRev.forward` uses
     `1 - attention.softmax(dim=2)`, which is *not* ordinary attention.
   - `AirKnow/lavis/models/__init__.py` — two unguarded imports of modules absent from the pinned
     revision.
   - `AirKnow/data_utils.py` — `.jpg` literal with a `.png` fallback that cannot change behaviour when
     the expected file is present.
   - `MTST_PTHA` — one unguarded import of an absent module, and `base_path` now reads
     `MTST_DATA_ROOT` with the upstream literal as fallback.
4. **Registry truth repairs.** The CSMCIR `COT_ours2` "captions unavailable" blocker is **withdrawn**:
   the author Hub publishes all three files, and the local copies are byte-identical (verified
   sha256). Four auxiliary assets were added (`encoder` backbone, `hint` checkpoint, `airknow`+`csmcir`
   BLIP-2/EVA-ViT-g caches). `ptha_mtst` gained a `source_dir`, a commit pin, and its protocol —
   `fashioniq_original_split`, because `validate_blip.py:44-51` builds labels against targets only and
   contains **no** reference mask, unlike the CIRR path at `:258-260`.

## 4. Suggestion dispositions

| suggestion | disposition | evidence |
|---|---|---|
| ENCODER `import datasets1` → `import datasets as datasets1` | **ACCEPTED (modified)** | the module genuinely does not exist; both the import and a re-export alias were added |
| ENCODER: inspect real CLI args, don't trust README | **ACCEPTED** | flags enumerated from `evaluate_model.py`; adapter's flags match the real parser |
| PAIR: `MHTransformer_rev` from `MultiHeadAttentionRev` | **ACCEPTED** | reconstruction proven exact (2-line diff); the reverse-attention maths is preserved |
| PAIR: do not alias reverse→ordinary attention | **ACCEPTED** | reconstruction uses `MultiHeadAttentionRev`, never the ordinary class |
| HINT: verify `fashioniq.pt`; build a replay wrapper | **ACCEPTED** | Hub file verified (2 724 451 182 B, sha256 `bf9c72d4…`); `workbench/replay/hint.py` is **placed and wired** (adapter + `hint_native_recall` parser + cache isolation); checkpoint unstaged |
| HINT: do not rewrite the evaluator | **ACCEPTED** | `test.test()` is called unchanged |
| CLVC-Net: `load_state_dict_from_url` → `torch.hub` | **ACCEPTED** | same weights URL retained; both imports tried |
| CLVC-Net: preserve ResNet-50 weights and preprocessing | **ACCEPTED** | URL and transform untouched |
| TG-CIR: verify the CLIP implementation | **ACCEPTED** | `clip.load(..., jit=False)` is OpenAI CLIP's signature; the PyPI `clip==0.2.0` pin is recorded as KNOWN-WRONG |
| DCNet: unseeded OOV reconstruction, labelled | **ACCEPTED** | `process_cap.py:31` and `text2vec.py:80` are unseeded; original caches unpublished; recorded as `EXPERIMENTAL_RECONSTRUCTION`, not applied |
| HABIT/INTENT/ConeSep/Air-Know: wrappers, assets, machine paths | **ACCEPTED (partial)** | Air-Know import + image-path fixes applied; wrappers block on the absence of a CLI in the pinned evaluators, which is reported not invented |
| Air-Know: do not silently switch eval mode | **ACCEPTED** | `model.train()` at `test_BLIP2.py:10` governs the FashionIQ path; left as published and documented as a variant |
| CSMCIR: “captions unavailable” | **REJECTED as stated** | the Hub publishes them; blocker withdrawn with byte-level evidence |
| CSMCIR: excessive eager imports | **ACCEPTED** | 98 unguarded imports confirmed; the existing approved patch covers it |
| Combiner/CLIP4Cir: verify part digests before reconstructing | **ACCEPTED** | all five parts and API digests recorded; bytes deliberately not downloaded (8.6 GB, no torch) |
| CLIP4Cir issues #9/#10 | **ACCEPTED** | #9 confirms 2560/5120 for RN50x4 (README's CIRR command is the typo); #10 explains the released-checkpoint delta |
| SPRC: check the Baidu mirror | **ACCEPTED** | probed: Baidu reachable but gated; OneDrive 403; only ViT-L Google Drive links are public |
| SPRC: never swap ViT-L for ViT-G silently | **ACCEPTED** | recorded as a variant change, not applied |
| PTHA+MTST: integrate the author source | **ACCEPTED** | vendored and pinned; one import guarded, data root configurable |
| PTHA+MTST: FashionIQ weights unavailable | **ACCEPTED** | Hub publishes CIRR only; reported precisely, no substitution |

## 5. Files changed

Vendored: `workbench/third_party/<15 source dirs>` — **7 375 files** in the delivered tree, plus a
`provenance.json` in each.

New:
- `workbench/backend/vendored_source.py` — the vendored-source contract (pin + digest manifest).
- `workbench/scripts/vendor_upstreams.py` — the clone → pin → copy → metadata-removal step.
- `workbench/replay/hint.py` — the missing evaluation shell for HINT.
- `workbench/tests/test_vendored_source_integrity.py` — 13 tests for the contract above.
- `workbench/docs/reviews/VENDORED_RECOVERY_16_MODEL_STATUS.md` — this file.

Shared:
- `workbench/backend/runtime.py` — routes source checks through the vendored contract; a nested
  checkout keeps precedence over a manifest beside it.
- `workbench/backend/adapters/models.py` — builds the HINT replay command.
- `workbench/replay/common.py` — pin check via the vendored contract.
- `workbench/scripts/sync_upstreams.py` — reports a vendored checkout as clean-and-pinned.
- `workbench/registry/models.yaml` — `ptha_mtst` gains a source, a pin and its protocol.
- `workbench/registry/auxiliary_assets.yaml` — COT_ours2 captions become a published asset; the
  encoder backbone, HINT checkpoint, and AirKnow/CSMCIR BLAS-2 and EVA-ViT-g caches are declared.
- `.gitignore`, `.gitattributes` — vendored source is tracked; its bytes are never normalized.
- `workbench/tests/test_download_auxiliary_assets.py`,
  `workbench/tests/test_release_blocker_source_integrity.py` — restated in terms of the mechanisms
  they exercise rather than the disproven "captions unavailable" claim.

Four commits on `research/cir-failure-workbench`:
`5e6e284` vendoring + recovery fixes · `07f7d3a` keep the bytes pinned but the diff readable ·
`8eb9c6c` correct a CRLF count · `e894dbd` nested checkout takes precedence over a manifest.

## 6. Running a recovered workflow

The pinned source arrives with the branch, so the first step is a normal checkout:

```bash
git clone git@github.com:Le-Minh-Nhut/cir.git && cd cir
git checkout research/cir-failure-workbench
grep -H upstream_commit workbench/third_party/*/provenance.json   # pin, resolved URL, local fixes
```

**LIMN** (the one workflow with a launcher, evaluator, protocol, ranking and observer already wired):

```bash
python workbench/replay/limn.py \
  --checkpoint-root <dir with 0_dress_best_model.pt, 0_shirt_best_model.pt, 0_toptee_best_model.pt> \
  --dataset-root    <canonical FashionIQ root> \
  --source-root     workbench/third_party/LIMN \
  --preflight-only          # drop this flag to evaluate, and it will run the pinned test.py:test()
# or the whole gated pipeline:  ./workbench/run.sh autopilot --config <pc_autopilot.yaml> --apply --resume
```

**HINT** (its evaluator has no CLI at all; the wrapper supplies one and calls `test.py:test()`):

```bash
python workbench/replay/hint.py \
  --checkpoint-root <dir containing fashioniq.pt> \
  --dataset-root    <canonical FashionIQ root> \
  --source-root     workbench/third_party/HINT \
  --preflight-only
# fetches its checkpoint via the registry asset:
python workbench/scripts/download_auxiliary_assets.py --model hint
```

**Any other model** — stage its assets, then call the official evaluator from its checkout:

```bash
python workbench/scripts/download_auxiliary_assets.py --all --dry-run   # then drop --dry-run
python workbench/scripts/download_checkpoints.py        --all --dry-run  # then drop --dry-run
cd workbench/third_party/<source_dir> && python <the registry's script for that model>
```

**CSMCIR** additionally needs its approved patch, which is declared in the registry and applied to a
derived copy rather than in place:

```bash
patch -p1 -d workbench/third_party/CSMCIR < workbench/patches/csmcir_models_init_compat.patch
```

Every command above uses registry-declared paths only; no laptop-specific absolute path is required.
Readiness, per-model environment pins and asset URLs all come from `workbench/registry/`.
