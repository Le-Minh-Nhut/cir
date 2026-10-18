# airknow — reproducibility reassessment

Independent adversarial, READ-ONLY audit. Nothing under `/home/heheboiz/data/cir` was written except
this file. Scratch in `/tmp/aud`. Everything is `STATIC_AUDIT` unless marked `OBSERVED`.
**No GPU inference was run and no checkpoint was loaded. Nothing here is `RUNTIME_VERIFIED`.**

---

## 1. MODEL IDENTITY

| Field | Value | Source |
|---|---|---|
| model_id | `airknow` | `workbench/registry/models.yaml:97` |
| method_name | Air-Know | `models.yaml:107` |
| paper_title | "Air-Know: Arbiter-Calibrated Knowledge-Internalizing Robust Network for Composed Image Retrieval" | `models.yaml:111` |
| paper_url | `https://arxiv.org/abs/2604.19386` | `models.yaml:112` |
| upstream_repo_url | `https://github.com/iLearn-Lab/CVPR26-Air-Know` | `models.yaml:113` |
| upstream_commit_sha | `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | `models.yaml:115` |
| source_dir | `AirKnow` | `models.yaml:117` |
| fashioniq_layout | `fashioniq_ilearn_resized` | `models.yaml:118` |
| native_protocol | `fashioniq_full_gallery_ref_excluded` | `models.yaml:121` |
| reported_r10 / r50 / mean | 55.75 / 75.72 / 65.73 | `models.yaml:123-125` |
| reproduction_status | `NOT_RUN` | `models.yaml:126` |
| strict_clean_checkpoint_available | `false` | `models.yaml:127` |

**OBSERVED** pin check:
```
$ git -C workbench/third_party/AirKnow rev-parse HEAD
6ab6fa95b746f85b08eea4cb66eb51448bce99b7
```
Matches `models.yaml:115`. Checkout history is a single commit (OBSERVED
`git rev-list --count HEAD` → `1`), so the pin is a snapshot, not a point in a history.

**SNAPSHOT 2 (current upstream), OBSERVED:**
```
$ git -C workbench/third_party/AirKnow ls-remote origin HEAD
6ab6fa95b746f85b08eea4cb66eb51448bce99b7	HEAD
```
Current upstream HEAD equals the pin. **The authors have changed nothing since the pin**: no CLI was
added, no dependency was fixed, no checkpoint was published on GitHub, and `6ab6fa95` is still the
only commit. A user has since replaced the repo (README points at `ZhihFu/Air-Know`, and GitHub
redirects `iLearn-Lab/CVPR26-Air-Know` there) but the content is unchanged.

## 2. SOURCE STATUS

**Accessible: yes. Complete: no** — three separate incompletenesses, none of them the workbench's.

1. Two class modules are absent although the package imports them (§3.1).
2. The pinned tree ships no weights, no loader, and no checkpoint reference (§5).
3. The published checkpoints require correction dictionaries and a resized-image tree that are
   published nowhere (§6).

OBSERVED hygiene note (not a defect in this model's own logic, but it affects every path
derivation): the checkout tracks **1837 files of which 1000+ are `.pyc`** and contains **no
`.gitignore`** — the author committed their `__pycache__` trees, exactly as CSMCIR did.

## 3. NATIVE EVALUATION WORKFLOW

### 3.1 Entrypoint, CLI, and what a wrapper must call

**There is no `__main__` in the evaluator.** OBSERVED:
```
$ grep -rn "__main__" *.py
cirr_sub_BLIP2.py:92: if __name__ == '__main__':
train_BLIP2.py:339: if __name__ == '__main__':
# test_BLIP2.py -> no match
```

**The native test function.** Signature, verbatim, `test_BLIP2.py:9`:

```python
def test(params, model, testset, category, txt_processors):
```

It is at module scope in `test_BLIP2.py:9`, with the FashionIQ branch at `:82-177`. So:

- **Is evaluation callable without a CLI? YES.** `test(params, model, testset, category, txt_processors)`
  is a plain function; nothing about it needs argparse. It was designed for
  loop-within-training use.
- **Only callsite:** `train_BLIP2.py:298`
  `t = test.test(args, model, trainset, category, txt_processors)`, inside `train_and_evaluate`
  (`:268-337`), reached from `train_BLIP2.py:339 __main__` → `:362`.
- **But that callsite is a training loop, not an evaluation.** `test` is invoked once per epoch after
  a training epoch (`:293-303`), and the script cannot be pointed at a checkpoint to evaluate:
  OBSERVED `grep -n "torch.load|load_state_dict" train_BLIP2.py test_BLIP2.py` → **no match**.
  The only weight handling in the repo is `torch.save(model, ...)` (`train_BLIP2.py:366`); the
  declared `--restore_file` (`:66-68`) is never read.

**Exact call chain a wrapper would have to perform** (all `file:line` verified):

```
1. sys.path must include the checkout root AND the checkout root must be cwd  (train_BLIP2.py:18-23
   use bare `import utils/datasets/test_BLIP2`; datasets.py:33-35 concatenate the path string, so
   the caller's path must end with a separator — 'path + resized_image')
2. from lavis.models import load_model_and_preprocess         train_BLIP2.py:32  (dies; see 3.1a)
3. model, vis, txt_processors = load_model_and_preprocess(
        name="Blip2QformerCir", model_type="pretrain", is_eval=False, device=...)   :196
   -> registry.get_model_class("Blip2QformerCir")            lavis/models/__init__.py:40 (never runs)
   -> cls.from_pretrained("pretrain")                        lavis/models/__init__.py:175-190
   -> lavis/configs/models/blip2/blip2_pretrain.yaml:10
      "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth"
      (loads the BASE BLIP-2 weights — not Air-Know's)
4. The Air-Know weights then have to be applied by the wrapper itself:
   torch.load(<checkpoint>, weights_only=False)   ->  the .pt IS a whole-module pickle (evidence §5)
   (the repo has NO code that does this; cirr_sub_BLIP2.py:120 is the only loader and it is the
    CIRR-submission path, not FashionIQ)
5. preprocess = targetpad_transform(1.25, 224)               train_BLIP2.py:112-119 -> data_utils.py
6. testset = datasets.FashionIQ(path=<fashioniq_root_with_trailing_slash>,
                                transform=preprocess, noise_ratio=0)     train_BLIP2.py:125-129
   -> datasets.py:183  image_splits/split.{cat}.val.json
   -> datasets.py:185  captions/cap.{cat}.val.json
   -> datasets.py:187  captions/correction_dict_{cat}.json        <-- MISSING INPUT
   -> datasets.py:155/166  resized_image/{cat}/{id}.jpg           <-- MISSING INPUT
7. For each category in ['dress','shirt','toptee']:
   out = test_BLIP2.test(args, model, testset, category, txt_processors)   test_BLIP2.py:9 / :82-177
   -> model.extract_retrieval_compose(imgs, mods)   :113 -> blip2_cir.py:463 -> returns (N,1,1,256)
   -> model.extract_retrieval_target(imgs)          :129 -> blip2_cir.py:506 -> returns (M,256,32)
   -> sims = np.matmul(...).squeeze().max(-1)       :142-144 -> (N,M)
   -> sims[i, test_targets_id.index(t['source_img_id'])] = -10e10   :151   (reference masked)
   -> nn_result = np.argsort(-sims[i,:])[:50]       :154
   -> for k in [1,10,50]: r = 100*hits/len(nn_result)  :168-174
   -> returns [('dress_r1',v),('dress_r10',v),('dress_r50',v)]
8. The wrapper then has to form the macro itself: the source emits NO aggregate.
```

`args` must be an object exposing at least `dataset`, `batch_size`, `local_rank`, `num_workers`
(read at `test_BLIP2.py:27, 108, 184, 206, 231, 290, 305`). A wrapper therefore has to synthesize an
`argparse.Namespace` or reuse `train_BLIP2`'s parser — it cannot call `test` with the bare 5 arguments
and nothing else.

### 3.1a The package cannot be imported — three independent causes, all OBSERVED

**(i) Two absent class modules, at module scope, before the class this model needs.**

```
$ grep -rn "from lavis.models.blip_models.blip_cir" lavis/models/__init__.py
33:from lavis.models.blip_models.blip_cir import BlipCir
34:from lavis.models.blip_models.blip_cir_caption import BlipCirCaption
$ ls lavis/models/blip_models/ | grep -i cir     -> no matches
$ ls lavis/models/blip_models/blip_cir.py
ls: cannot access 'lavis/models/blip_models/blip_cir.py': No such file or directory
$ git ls-files lavis/models/blip_models/ | grep -i cir
lavis/models/blip_models/.ipynb_checkpoints/blip_cir-checkpoint.py
lavis/models/blip_models/__pycache__/blip_cir.cpython-{37,38,39,312}.pyc
lavis/models/blip_models/__pycache__/blip_cir_caption.cpython-{37,38,39,312}.pyc
```

`lavis/models/blip_models/` **is** a regular package (it has `__init__.py`), so a missing
`blip_cir.py` inside it cannot fall back to a namespace package. `Blip2QformerCir` — the class this
evaluator needs — is imported at `lavis/models/__init__.py:40`
`from lavis.models.blip2_models.blip2_cir import Blip2QformerCir`, which is **after** line 33; the
package raises at line 33 and line 40 never executes.

This matches the workbench's own label exactly: `models.yaml:98` `source_blocker`
("no runnable evaluation path is registered for this model: the pinned evaluator is a callable with
no CLI or checkpoint loader, the adapter deliberately declines to build a command …"). The registry
text is **correct**; it does not say the package cannot import, and adding that would be an
improvement.

*(Correction to a hypothesis in `workbench/docs/reviews/models/airknow.md` CF-1: my earlier
enumeration of "unresolvable imports" additionally listed `lavis.common`,
`lavis.datasets.base_dataset_builder`, `lavis.models.ulip_models*` — those are **not** import-time
failures. `lavis/common/` and `lavis/models/ulip_models/` are namespace packages (no `__init__.py`,
OBSERVED), so `from lavis.common import dist_utils` (`lavis/common/logger.py:16`) resolves;
`lavis/datasets/base_dataset_builder` is imported only inside a function body
(`lavis/common/registry.py:31`, indented) and never runs on this path — the working import at
`:35` is `lavis/datasets/builders/base_dataset_builder`. Only lines 33-34 are real blockers.
My file-extension heuristic was too naive and I am withdrawing that part of the earlier finding.)*

**(ii) A hard `torchaudio` dependency in the eager import graph.** OBSERVED, live trace with a
`sys.modules` stub for `torchaudio` injected so it would not be the first failure:

```
lavis/__init__.py:15                      from lavis.datasets.builders import *
lavis/datasets/builders/__init__.py:8     -> base_dataset_builder
lavis/datasets/builders/base_dataset_builder.py:18  -> lavis.processors
lavis/processors/__init__.py:29           from lavis.processors.audio_processors import BeatsAudioProcessor
lavis/processors/audio_processors.py:10   import torchaudio.transforms as transforms
ModuleNotFoundError: No module named 'torchaudio.transforms'; 'torchaudio' is not a package
```
So even after (i) is fixed, `from lavis.models import load_model_and_preprocess`
(`train_BLIP2.py:32`) executes `lavis/__init__.py:15-18`, which pulls the whole processors package
and therefore `torchaudio`. `torchaudio` is **absent from the registry's environment declaration**
(`models.yaml:99-106` lists only `scikit-learn`, `transformers`, `salesforce-lavis`, `timm`) although
the upstream Hub card tells the user to install it
(`huggingface.co/iLearn-Lab/CVPR26-Air-Know/raw/main/README.md`, Step 1:
`pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0`). OBSERVED:
`workbench/backend/runtime.py:753-754, 825, 857-858` only maps `pytorch`/`torchvision`; there is no
`torchaudio` key. **`pip install torchaudio==2.1.0` is a normal, installable wheel — remedy is one
registry line.**

**(iii) One further reachability bug the source ships itself.** `lavis/models/ulip_models/ULIP_models.py:16`
`from lavis.models.ulip_models import losses`: `lavis/models/ulip_models/` has **no `__init__.py`**
(OBSERVED), so it is a namespace package and `lavis.models.ulip_models.losses` is a submodule, not an
attribute of the namespace package — CPython raises `ImportError` unless `losses` was imported first.
Not on the `Blip2QformerCir` path (`blip2_vicuna_xinstruct.py:33` is the only importer and
`lavis/models/__init__.py` never imports it, OBSERVED `grep xinstruct` → no match), so **P3, latent**.

## 4. DEPENDENCIES

D1-D8 were **not defined in any file I could read** (`grep` over `workbench/` for `D1–D8`,
`DEPENDENCY_CATEGORY`, and the dependency-category wording returned no definition). I therefore
report the category in words per import rather than inventing a numeric mapping. My labels, applied
consistently:

- **local-source-and-present** — a module inside the pinned tree that exists;
- **local-source-and-absent** — a module inside the pinned tree that does not exist;
- **third-party-installable** — a PyPI/conda package;
- **OS/runtime** — not a Python package at all.

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `from lavis.models.blip_models.blip_cir import BlipCir` | `lavis/models/__init__.py:33` | upstream | — | **local-source-and-absent** — NOT installable | 3.8 / 2.1.0 | author must ship the file; a guard changes which classes register | OBSERVED absent |
| `from lavis.models.blip_models.blip_cir_caption import BlipCirCaption` | `lavis/models/__init__.py:34` | upstream | — | **local-source-and-absent** | 3.8 / 2.1.0 | same | OBSERVED absent |
| `import torchaudio` + `import torchaudio.transforms as transforms` + `import torchaudio.compliance.kaldi` | `lavis/processors/audio_processors.py:9,10,13` (reached from `lavis/processors/__init__.py:29` ← `lavis/__init__.py:15`) | third-party | 2.1.0 (Hub card) | **third-party-installable** — wheel exists | 3.8 / 2.1.0 / cu121 | add `torchaudio==2.1.0` to `models.yaml` env | OBSERVED `ModuleNotFoundError` |
| `import torch` / `torchvision` | repo-wide | third-party | 2.1.0 / 0.16.0 | installable | cu121 declared | already in registry | not run |
| `from lavis.models import load_model_and_preprocess` | `train_BLIP2.py:32`; also `cirr_sub_BLIP2.py:93` | local | — | transitively blocked by the two rows above | 3.8 / 2.1.0 | none until the tree is complete | OBSERVED fail |
| `from lavis.models.Llm_online.gpt import GPT4o` | `train_BLIP2.py:33` | local (present) | — | present, but the module does `from openai import OpenAI` at `:2` | — | none needed | see next row |
| `from openai import OpenAI` | `lavis/models/Llm_online/gpt.py:2` | third-party | unpinned | **third-party-installable** | — | add `openai` to the env | not run |
| `from thop import profile` | `train_BLIP2.py:14` | third-party | — | **third-party-installable** (`thop`) — **not in the registry env** and not in the Hub card | — | add `thop` | not run; imported at module scope, never used |
| `import clip` | `train_BLIP2.py:17`, `cirr_sub_BLIP2.py:6` | third-party | — | **third-party-installable** (`git+https://github.com/openai/CLIP`) — not on PyPI as `clip`; not in the registry env | — | add the OpenAI CLIP wheel/git dep | not run; imported, never used |
| `import setproctitle` | `train_BLIP2.py:31` | third-party | — | **third-party-installable** — not in the registry env | — | add `setproctitle` | not run; used at `:29,:343` |
| `from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation` | `datasets.py:13` | third-party | 4.25.0 (registry) | installable | — | none | not run |
| `from transformers import …` (`BertConfig`, `BertModel`) | `lavis/models/blip2_models/blip2.py:32,48,55` | third-party | 4.25.0 | installable | — | none | not run; **hardcoded `/root/bert-base-uncased`** |
| `import cv2`/`numpy`/`PIL`/`sklearn` | `datasets.py:2-16`, `data_utils.py` | third-party | `scikit-learn==1.3.2` declared | installable | — | none | not run |
| CUDA itself (`.cuda()`, `.to('cuda')`, `GradScaler`) | `test_BLIP2.py:111,128,293,308`; `train_BLIP2.py:196-197` | OS/runtime | cu121+ | n/a | **CUDA required, no CPU path** | GPU host | not run |

Installs ABOVE the declared environment that the pinned source actually needs at import time:
`torchaudio`, `thop`, `clip` (OpenAI), `setproctitle`, `openai`. `thop` and `clip` are imported and
never used; `setproctitle` is used. All are **installable** — this is
`FIRST_PARTY_ENVIRONMENT_CONFIGURATION`, not a source defect.

## 5. CHECKPOINTS

Registry (`models.yaml:129-149`): two variants, both `status: NO_CLEAN_CHECKPOINT`,
`checkpoint_mapping_status: VERIFIED_METADATA`, `expected_size_bytes: null`, `expected_sha256: null`.

| variant | file | declared training noise | OBSERVED size | OBSERVED LFS oid (sha256) |
|---|---|---|---|---|
| `fiq_n05` | `airknow_fiq_0.5.pt` | 50 % | 2724056638 | `f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6` |
| `fiq_n08` | `airknow_fiq_0.8.pt` | 80 % | 2724056638 | `756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e` |

OBSERVED: both URLs return `200` with `Content-Length: 2724056638`; the Hub repo
`iLearn-Lab/CVPR26-Air-Know` (revision `7ad3a0d81fbd1b87d9ef7b9877c81950a34b6b85`) contains
**exactly four files**: `.gitattributes`, `README.md`, and the two `.pt` files above. So
`strict_clean_checkpoint_available: false` (`models.yaml:127`) and the `notes` line
(`models.yaml:128`) are **both correct**.

**What the checkpoint IS.** OBSERVED, HTTP range reads of the ZIP central directory and the pickle:
```
size 2724056638, entries 865
first member: model_epoch_1/data.pkl (1162828 bytes), then model_epoch_1/byteorder
$ pickletools.genops(data.pkl) first constants:
['training', '_parameters', 'query_tokens', 'storage', '0', 'cuda:0', 'prompt_tokens', '1', …]
$ module refs in data.pkl:
lavis.models.blip2_models.blip2_cir  lavis.models.blip2_models.Qformer
lavis.models.blip2_models.blip2      lavis.models.eva_vit
$ string counts: Blip2QformerCir 1  Proxy 1  eki 9  alpha 3  visual_encoder 1  Qformer 15
```
`_parameters` as the first key of a `state_dict`, plus `lavis.models.blip2_models.blip2_cir` as a
class global, proves this is a **whole-`nn.Module` pickle of `Blip2QformerCir`** (torch's
`state_dict` nesting is `{tensor_name: tensor}`, never under `_parameters`, and carries no class
globals). Confirmation of the archive-prefix convention, run locally on this host (OBSERVED,
torch 2.14.0+cpu): `torch.save(nn.Module, "whole_module.pt")` → ZIP member `whole_module/data.pkl`;
`torch.save(state_dict, "state_dict.pt")` → `state_dict/data.pkl`. The directory prefix is derived
from the **filename**, so it does not distinguish the two — the pickle contents do, and they do.

**Loader contract implied:** `torch.load(..., weights_only=False)` → callable module. The only loader
in the pinned repo is `cirr_sub_BLIP2.py:120` (CIRR submission path, which reads the model dir with
`os.listdir(model_dir)` and hardcodes `/root/autodl-tmp/cirr_data/CIRR/` at `:117`). **No FashionIQ
loader exists anywhere.**

**Environment trap (P2).** The checkpoint was serialized on `cuda:0` (storage device strings present
in `data.pkl`), and `blip2_cir.py:324` does `target_feats.cuda()` unconditionally, and
`test_BLIP2.py:111,128,293,308` do `.cuda()`. So the pickle is device-tagged and the evaluator is
CUDA-only; a CPU smoke test of the loader is not available without editing the source.

**Availability class:** the bytes are reachable (`URL_VERIFIED`-equivalent for both files) but neither
is the 0 %-noise artifact the paper row describes, and the registry's own `status` already says so.
The correct class for the row the registry declares is `CHECKPOINT_UNPUBLISHED` for the *clean*
variant, and `PUBLIC_CHECKPOINT_NOT_MAPPED` for the training-condition mapping (the Hub never
associates a file with a dataset, category set or noise ratio; the mapping is filename-inferred).

## 6. AUXILIARY ASSETS

| asset | purpose | required by | present in repo? | public source? | class |
|---|---|---|---|---|---|
| `captions/correction_dict_{dress,shirt,toptee}.json` | token→token rewrite applied **before** the LAVIS caption processor | `datasets.py:110,187` (val), `:110` (train init) | **no** — `git ls-files` has no `correction_dict*`; 0 present on disk | **no** — not on GitHub, not on the author Hub (`iLearn-Lab/CVPR26-Air-Know` has 4 files, OBSERVED) | `AUXILIARY_ASSET_MISSING` |
| `resized_image/{dress,shirt,toptee}/{id}.jpg` | gallery + query images | `datasets.py:33,155,166` | **no** | **no** — `fashioniq_ilearn_resized` is a per-author reshuffle with no published recipe or archive | `PUBLIC_ASSET_NOT_PREPARED` |
| `judge/judge.train.{noise_ratio}.json` + `test_*.pkl` | EKI anchor set (training) | `datasets.py:238` (`FashionIQ_V`) | no | no | training-only, not an evaluation blocker |
| BLIP-2 base weights `blip2_pretrained.pth` | encoder init | `lavis/configs/models/blip2/blip2_pretrain.yaml:10` (`https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth`) | n/a | yes (Google storage) | reachable |
| EVA-ViT-g, bert-base-uncased | encoder/tokenizer | `blip2.py:32,48,55` — **hardcoded `/root/bert-base-uncased`** | n/a | bert: yes; but the path is author-local | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` |

**Correction dictionaries: needed? PRESENT? PUBLIC?**
- Needed: **YES, for the official path.** `datasets.py:187` opens
  `captions/correction_dict_{name}.json` unconditionally, before any query is built, so
  `datasets.FashionIQ(...)` raises `FileNotFoundError` without it. Its effect is real:
  `correct_text` (`:122-127`) lowercases, replaces all `string.punctuation` with spaces, then maps
  every token through the dict. Without the dict the caption text differs, and the caption feeds
  `extract_retrieval_compose` (`blip2_cir.py:479-486`) — the query embedding, i.e. the score.
- Present: **NO.** 0 files tracked, 0 on disk.
- Public: **NO.** I searched the author GitHub repo (not tracked) and the author Hugging Face repo
  (4 files, OBSERVED). For the record, the sibling iLearn-Lab CIR Hub repos also ship **only weights**
  (OBSERVED: `CVPR26-ConeSep` → 3 `.pt`; `AAAI26-HABIT` → 6 `.pt`; `AAAI26-INTENT` → 4 `.pt`;
  `AAAI25-ENCODER` → 4 `.pt`). **No iLearn-Lab CIR Hub repo publishes a correction dict or a
  resized-image tree.** This is consistent across the family: the asset is generated by the authors
  from a private copy and never released.

`workbench/registry/auxiliary_assets.yaml` has **zero `airknow` rows** (OBSERVED: `grep -n model_id:`
→ `csmcir` ×10, `clvc_net`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc` ×2, and nothing
else). `workbench/backend/fashioniq_layout.py:25-34` does check the three `correction_dict_*.json`
paths and `resized_image/{cat}` for this layout, so preflight reports them — but they are not
asset-registered, so there is no provenance, digest or labelled-variant path for them.

## 7. FASHIONIQ PROTOCOL

Registry: `native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:121`),
`supported_protocols: [fashioniq_full_gallery_ref_excluded]` (`models.yaml:122`),
`literature_split_label: original` (`models.yaml:120`). **Quote the authoritative declaration:**
Protocol B = `fashioniq_full_gallery_ref_excluded` — full gallery, reference excluded. I am not
proposing any change to it.

- **Gallery:** `datasets.py:31` hardcodes `self.split = 'original-split'`; `get_test_data` (`:181-225`)
  reads `image_splits/split.{name}.val.json` (`:183`) and, in the `original-split` branch
  (`:218-224`), enumerates **every** index in file order. Full ordered val split. The `val-split`
  pair-union branch (`:206-217`) is unreachable because `self.split` is never anything else.
- **Reference policy:** excluded. `test_BLIP2.py:147-151` builds `test_targets_id` from the gallery and
  sets `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` **before** `:154`
  `np.argsort(-sims[i, :])[:50]`. This is a mask, not a column removal — a documented, deliberate
  divergence from the workbench helper `recall_at_k_reference_excluded`
  (`src/evaluation/fashioniq.py:35-77`), which removes the column. Same ranking on the top-k prefix.
- **Query construction:** `datasets.py:186-192` — caption pair corrected, then joined
  `"{0} and {1}"`; `test_BLIP2.py:112` then runs `txt_processors["eval"]` (`blip_caption`,
  `lavis/processors/blip_processors.py:28-68`), which lowercases, strips a **narrower** punctuation
  set than `correct_text`'s `string.punctuation`, collapses whitespace and truncates at 50 words.
- **Categories:** `['dress','shirt','toptee']` (`datasets.py:107`, `train_BLIP2.py:133`).
- **Recall computation:** `test_BLIP2.py:168-174` — `nn_result = argsort(-sims)[:50]`,
  `for k in [1,10,50]`, `r = 100*hits/len(nn_result)`, output as `('{cat}_r{k}', r)`.
  Note R@1 is computed and reported though FashionIQ convention is R@10/R@50.
- **Macro:** the source produces **none**. The trainer sums the three R@10 values as a selection
  score (`train_BLIP2.py:296-301`) and writes a flat 9-key JSON. The paper's `AVG.` and the
  registry's `reported_*` are `(mean R@10 + mean R@50)/2`.
- **Paper row identity (OBSERVED, read from the pinned image `assets/results-fiq.png`):** the
  `0%` row Air-Know reads `Dress 51.03/73.08 · Shirt 56.76/75.05 · Toptee 59.47/79.02 · Average
  55.75/75.72 · AVG. 65.73`. It matches `models.yaml:123-125` exactly. The same table's
  `50%` row is `47.90/71.09 · 53.95/73.09 · 57.71/77.94 · 53.19/74.04 · 63.61` and the `80%` row is
  `42.52/66.03 · 51.31/70.26 · 53.86/74.41 · 49.23/70.23 · 59.73`. **The registry's row is the 0 %
  row; the published checkpoints are the 50 % and 80 % rows.** (An earlier workbench note recorded
  `reported_mean: 65.74`; the current registry value `65.73` matches the paper. The older note was
  wrong, not the current file.)

**Evaluation-mode defect (P1, source-level, reported not repaired).** `test_BLIP2.py:9-11`:
```python
def test(params, model, testset, category, txt_processors):
    model.train()
    with torch.no_grad():
```
`torch.no_grad()` disables gradients, **not** dropout. The `fashion200k` branch opts out with
`model.eval()` at `:13`; the FashionIQ branch (`:82-177`) never does. Live dropout on this path:
`Proxy` with `nn.Dropout(p=0.1)` (`blip2_cir.py:58,61`, constructed at `:146-150`) and the Q-Former
`BertConfig` (`blip2.py:48-55`). This is an upstream behaviour, not a workbench bug, and it makes the
native evaluator **non-deterministic**. The workbench does not replicate it and must not; a run that
calls `model.eval()` is a VARIANT, and that must be stated in any report.

## 8. WORKBENCH INTEGRATION

- **Adapter:** `workbench/backend/adapters/models.py:54-56` — `AirKnowAdapter(OfficialScriptAdapter)`
  with only `script = "test_BLIP2.py"` and **no `command()` override**. `OfficialScriptAdapter.command`
  raises `NotImplementedError("upstream evaluation command requires audit before execution")`
  (`workbench/backend/adapters/models.py:24-25`), so `command_is_audited`
  (`workbench/scripts/evaluate_models.py:130-131`) returns `False` and
  `evaluate_models.py:364-365` returns `SKIPPED: adapter command not audited`. **The deliberate
  decline is real and correct**: there is no audited command because no probeable command exists.
- **Planner / configs:** no `airknow` selection profile exists (`workbench/configs/` has
  `pc_autopilot.pilot_csmcir.yaml`, `pc_autopilot.pilot_limn.yaml`, `all_available`); `grep -rn airknow
  workbench/configs` → no match.
- **Checkpoint prep:** registry-driven; no `airknow` directory under `workbench/artifacts/checkpoints/`.
- **Preflight:** `workbench/backend/preflight.py:40-116` would emit, via
  `fashioniq_layout.py:25-34` + `:92-136`, blocks for `resized_image/{dress,shirt,toptee}`,
  the three `correction_dict_*.json`, and the image-extension check for `.jpg`. Since
  `auxiliary_assets.yaml` has no `airknow` rows, none of those is asset-registered (no digest, no
  provenance, no labelled-variant route). Correct but under-specified — P2.
- **Metric parser:** **absent.** `workbench/backend/metrics_extraction.py` dispatches only
  `limn` (`:403`) and `csmcir` (`:404-405`) plus `combiner`/`clip4cir` (`:406-407`) and `sprc`
  (`:408-411`); `airknow` falls through to the default at `:413-…` and would return
  `AGGREGATE_PARSER_UNAVAILABLE` with `observed_metrics=None`. So **even after the source runs, no
  number can be extracted by the workbench** without a new parser. The source's own output shape —
  a Python list of `('{cat}_r{k}', float)` tuples, `test_BLIP2.py:175` — is trivially parseable, but
  nothing parses it today.
- **Per-query support:** impossible without editing the official implementation. `test` returns only
  aggregate tuples; the per-query similarity matrix is local to the function. Correct not to have it.

## 9. CONFIRMED DEFECTS

**D-1 (P0) — the pinned `lavis.models` package cannot be imported; two absent class modules sit at
module scope before the class the evaluator needs.**
Root cause: the author shipped `.pyc`/`.ipynb_checkpoints` of `blip_cir`/`blip_cir_caption` and the
`.py` files were never committed. Reproduction:
`lavis/models/__init__.py:33-34`; `ls lavis/models/blip_models/blip_cir.py` → not found;
`lavis/models/__init__.py:40` is after 33.
Fix: the author must commit the two modules, or the imports must be guarded — but guarding changes
which classes LAVIS registers, so a guarded checkout is a variant, not parity. There is no
`workbench/patches/airknow*` file (OBSERVED `ls workbench/patches/` → `csmcir.md`,
`csmcir_models_init_compat.patch`).
Regression result: NOT_RUN.

**D-2 (P0) — the declared FashionIQ inputs do not exist publicly.**
`captions/correction_dict_{cat}.json` (`datasets.py:187`) and the `resized_image/` tree
(`datasets.py:33`) are required unconditionally and are published nowhere (OBSERVED: author GitHub,
author Hub, four sibling iLearn-Lab CIR Hubs — none contains either). Reproduction: the source lines
above; the file lists.
Fix: **operator must obtain them from the authors** (the correction dicts are small JSON; the resized
tree is a per-author reshuffle whose recipe is unpublished — so a self-resized tree is a variant, and
its resampling differs from whatever the authors used). This is `AUXILIARY_ASSET_MISSING` +
`PUBLIC_ASSET_NOT_PREPARED`, and `NONDETERMINISTIC_PREPROCESSING` applies to any self-made tree.
Regression result: NOT_RUN.

**D-3 (P0) — no runnable path exists even in principle, because no FashionIQ loader exists.**
There is no `torch.load`/`load_state_dict` on the evaluation path (OBSERVED `grep` in
`train_BLIP2.py`/`test_BLIP2.py` → no match); `load_model_and_preprocess` loads **base BLIP-2 weights**
from `blip2_pretrain.yaml:10`, not Air-Know's; the only loader in the repo is the CIRR-submission path
`cirr_sub_BLIP2.py:120`, and it reads a whole-model pickle from a model directory that the FashionIQ
path never produces. A wrapper must therefore do the whole-model `torch.load` itself — feasible
(§11) but it is net-new code with no upstream analogue, so the result is an adapted run.
Regression result: NOT_RUN.

**D-4 (P1) — `test()` leaves the model in training mode.** See §7. Documented, not repaired by the
workbench. Any reproduction must decide: replay verbatim (non-deterministic) or declare VARIANT.
Regression result: NOT_RUN.

**D-5 (P1) — the 0 % paper row vs 50/80 % checkpoints.** `models.yaml:123-125` transcribe the 0 %
row exactly (OBSERVED against `assets/results-fiq.png`), while both published checkpoints are
noise-trained (`models.yaml:134,144`). Both variants are correctly labelled
`NO_CLEAN_CHECKPOINT`, so the registry does not mislead — but no run can legitimately be compared to
`reported_*` for this model. Reproduction: the image and the `models.yaml` lines.
Regression result: NOT_RUN.

## 10. REMAINING BLOCKERS

| # | class | ownership | evidence | remediation |
|---|---|---|---|---|
| B1 | `UPSTREAM_IMPORT_BUG` | upstream (iLearn-Lab) | `lavis/models/__init__.py:33-34`; both modules absent while their `.pyc` are tracked | author commits the modules, or an approved (variant) guard patch is declared as for csmcir |
| B2 | `INSTALLABLE_DEPENDENCY_MISSING` | workbench | `lavis/processors/audio_processors.py:9-13` reached from `lavis/__init__.py:15`; `torchaudio` absent from `models.yaml:99-106` | add `torchaudio==2.1.0` (plus `thop`, `setproctitle`, `clip`, `openai`) to the environment declaration |
| B3 | `AUXILIARY_ASSET_MISSING` | operator | `datasets.py:187`; 0 tracked, 0 on disk, 0 published on any iLearn-Lab Hub | obtain the three `correction_dict_*.json` from the authors; register them as auxiliary assets with digests |
| B4 | `PUBLIC_ASSET_NOT_PREPARED` | operator | `datasets.py:33,155,166`; no published resized tree or recipe | obtain the author tree; a self-made tree must be labelled VARIANT and the resampling recorded |
| B5 | `CHECKPOINT_UNPUBLISHED` | upstream | Hub has exactly two files, at 50 % and 80 %; paper row is 0 % | author publishes a 0 % checkpoint, or the row comparison is withdrawn |
| B6 | `FIRST_PARTY_ADAPTER_MISSING` | workbench | `workbench/backend/adapters/models.py:54-56` raises; `metrics_extraction.py` has no `airknow` branch | a wrapper + a 9-key parser; both are net-new |
| B7 | `SCIENTIFIC_PROTOCOL_UNVERIFIED` | upstream | `test_BLIP2.py:10` `model.train()` on the eval path | decide replay-verbatim vs VARIANT at run time |
| B8 | `GPU_RUNTIME_UNTESTED` | host | `models.yaml:126` `NOT_RUN`; no GPU observed | another host |

`EXTERNAL_BLOCKER` is deliberately not used.

## 11. IMPLEMENTATION PLAN

**Wrapper feasibility: MODERATE.** Not TRIVIAL (no CLI, no loader, missing inputs), not BLOCKED_BY_
MISSING_SOURCE in the sense of "impossible to write" — I can write the wrapper, but it cannot be
exercised here and two of its inputs do not exist publicly in any form.

**Exact call chain a wrapper would perform** (each step cited in §3.1):

```python
# 0. cwd = checkout root; sys.path = [checkout_root]; args = Namespace(dataset='fashioniq',
#    batch_size=..., local_rank=0, num_workers=..., ...)   # required by test_BLIP2.py:27,184,290
# 1. from lavis.models import load_model_and_preprocess              train_BLIP2.py:32
# 2. model, vis, txt = load_model_and_preprocess("Blip2QformerCir","pretrain",False,'cuda:0')
#       -> base BLIP-2 weights                                   train_BLIP2.py:196
# 3. model = torch.load('<airknow_fiq_0.5.pt>', weights_only=False)   # WHOLE-MODULE pickle (§5)
#       (the registry filename; 0 % unavailable)
# 4. model.cuda(); model.eval()            # .eval() is a deliberate deviation from :10 — VARIANT
# 5. preprocess = targetpad_transform(1.25, 224)                     train_BLIP2.py:112-119
# 6. testset = datasets.FashionIQ(path='<root>/', transform=preprocess, noise_ratio=0)  :125-129
# 7. for cat in ('dress','shirt','toptee'):
#        out = test_BLIP2.test(args, model, testset, cat, txt)      :298 / test_BLIP2.py:9
# 8. macro = ((r10_d+r10_s+r10_t)/3 + (r50_d+r50_s+r50_t)/3)/2      # source emits no aggregate
```

Size: roughly 60-90 lines (argparse shell + loader + three-category loop + macro + JSON print), plus
a ~30-line parser branch if per-category output is to be extracted by the workbench rather than read
by eye. Complexity **SMALL-to-MODERATE in code volume**; the cost is not the code, it is (a) the
unpublished inputs, (b) the training-mode decision, (c) the noise-condition mismatch.

**Invariants I will not touch:** `fashioniq_full_gallery_ref_excluded` (full ordered val split,
reference masked before ranking) stays as declared. No gallery substitution, no reference-policy
change, no caption-input change, no noise-condition substitution.

**Scientific risk:** HIGH if run as-is. Three distinct ways to produce a wrong-looking number without
any obvious failure — training-mode dropout, a self-made resized tree, a self-made correction dict —
and none of them is detectable from the output alone. Any run must carry all three as explicit
variant flags.

## 12. FINAL VERDICT

**BLOCKED.**

- Pin, protocol, gallery policy, reference policy, recall arithmetic and the paper-row transcription
  are **VERIFIED** against the pinned source and against the pinned result image.
- The tagline in `models.yaml:98` is **verified correct** and is independently confirmed here.
- Two blockers are **new** relative to the existing workbench documentation and are demonstrated:
  (1) `lavis.processors` → `torchaudio` (`lavis/processors/audio_processors.py:9`) sits on the eager
  import graph of `train_BLIP2.py:32` and is missing from the declared environment; (2) the published
  checkpoints are whole-`nn.Module` pickles with no FashionIQ loader anywhere in the repo.
- The `np.matmul` tensor-shape worry recorded in the previous audit is **not** a defect: OBSERVED
  `np.matmul((N,1,1,256),(M,256,32))` → `(N,M,1,32)` → `.squeeze()` → `(N,M,32)` → `.max(-1)` →
  `(N,M)`, exactly the 2-D matrix `sims[i, j]` indexing expects. The registry's own withdrawal
  (`models.yaml:98`: "that claim was WRONG and is withdrawn") is correct.
- Nothing here is runnable on this host and nothing was run.

## 13. EVIDENCE

Repo: `https://github.com/iLearn-Lab/CVPR26-Air-Know` (redirects to `ZhihFu/Air-Know`).
Pin: `6ab6fa95b746f85b08eea4cb66eb51448bce99b7`.

```
$ git -C workbench/third_party/AirKnow rev-parse HEAD
6ab6fa95b746f85b08eea4cb66eb51448bce99b7
$ git -C workbench/third_party/AirKnow ls-remote origin HEAD
6ab6fa95b746f85b08eea4cb66eb51448bce99b7	HEAD
$ git -C workbench/third_party/AirKnow rev-list --count HEAD
1

$ grep -rn "__main__" *.py            # in the checkout root
cirr_sub_BLIP2.py:92: if __name__ == '__main__':
train_BLIP2.py:339: if __name__ == '__main__':      # test_BLIP2.py: no match
$ grep -n "def test" -A2 third_party/AirKnow/test_BLIP2.py
9:def test(params, model, testset, category, txt_processors):
10:    model.train()
11:    with torch.no_grad():

$ ls lavis/models/blip_models/ | grep -i cir       -> no matches
$ git ls-files lavis/models/blip_models/ | grep -i cir
lavis/models/blip_models/.ipynb_checkpoints/blip_cir-checkpoint.py
lavis/models/blip_models/__pycache__/blip_cir.cpython-{37,38,39,312}.pyc
lavis/models/blip_models/__pycache__/blip_cir_caption.cpython-{37,38,39,312}.pyc

# live import trace with a torchaudio stub injected, so torchaudio is not the first failure
lavis/__init__.py:15
lavis/datasets/builders/__init__.py:8
lavis/datasets/builders/base_dataset_builder.py:18
lavis/processors/__init__.py:29
lavis/processors/audio_processors.py:10
ModuleNotFoundError: No module named 'torchaudio.transforms'

# checkpoint metadata and container
$ HEAD …/resolve/main/airknow_fiq_0.5.pt -> 200 Content-Length: 2724056638
$ HEAD …/resolve/main/airknow_fiq_0.8.pt -> 200 Content-Length: 2724056638
$ GET .../raw/main/airknow_fiq_0.5.pt
oid sha256:f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6  size 2724056638
$ GET .../raw/main/airknow_fiq_0.8.pt
oid sha256:756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e  size 2724056638
$ HF api models/iLearn-Lab/CVPR26-Air-Know -> sha 7ad3a0d81fbd1b87d9ef7b9877c81950a34b6b85
   .gitattributes | README.md | airknow_fiq_0.5.pt | airknow_fiq_0.8.pt      # exactly 4 files
$ range-read of the ZIP: size 2724056638, 865 entries, first member model_epoch_1/data.pkl (1162828)
$ pickletools.genops(data.pkl) first constants:
['training', '_parameters', 'query_tokens', 'storage', '0', 'cuda:0', 'prompt_tokens', '1', …]
$ class globals in data.pkl:
lavis.models.blip2_models.blip2_cir  lavis.models.blip2_models.blip2  lavis.models.eva_vit
lavis.models.blip2_models.Qformer    (+ strings Blip2QformerCir 1, Proxy 1, eki 9, alpha 3)
$ local control (this host, torch 2.14.0+cpu):
torch.save(nn.Module, 'whole_module.pt') -> members ['whole_module/data.pkl', …]
torch.save(state_dict, 'state_dict.pt')  -> members ['state_dict/data.pkl', …]
   => prefix is filename-derived; `_parameters` + class globals are what identify the payload

# sibling iLearn-Lab CIR Hubs ship only weights
iLearn-Lab/CVPR26-ConeSep   -> 3 .pt
iLearn-Lab/AAAI26-HABIT     -> 6 .pt
iLearn-Lab/AAAI26-INTENT    -> 4 .pt
iLearn-Lab/AAAI25-ENCODER   -> 4 .pt
   (no correction_dict*, no resized tree, in any of them)

# the paper row, read from the pinned image assets/results-fiq.png
0%  Air-Know: 51.03/73.08 · 56.76/75.05 · 59.47/79.02 · 55.75/75.72 · 65.73
50% Air-Know: 47.90/71.09 · 53.95/73.09 · 57.71/77.94 · 53.19/74.04 · 63.61
80% Air-Know: 42.52/66.03 · 51.31/70.26 · 53.86/74.41 · 49.23/70.23 · 59.73
```

`file:line` index: `models.yaml:97-128,129-149`;
`workbench/backend/adapters/models.py:24-25,54-56`;
`workbench/backend/metrics_extraction.py:403-411`;
`workbench/backend/fashioniq_layout.py:25-34,83-89,92-136`;
`workbench/scripts/evaluate_models.py:130-131,364-365`;
`workbench/registry/auxiliary_assets.yaml` (zero `airknow` rows);
`AirKnow/test_BLIP2.py:9-11,13,82-96,106-132,142-151,154,168-175,179-217,277-359`;
`AirKnow/train_BLIP2.py:14,17,31,32,33,66-68,84-107,112-129,130,143,193-204,268-337,296-301,339,366`;
`AirKnow/datasets.py:13,28-62,66-101,108-131,154-173,181-225,227-263,277-300`;
`AirKnow/lavis/models/__init__.py:33,34,40`;
`AirKnow/lavis/models/blip_models/` (no `blip_cir.py`);
`AirKnow/lavis/models/blip2_models/blip2_cir.py:42-63,81,93-97,99,213-300,408-463,463-503,506-527,530-553,669-695`;
`AirKnow/lavis/models/blip2_models/blip2.py:32,48,55`;
`AirKnow/lavis/__init__.py:15-18`; `AirKnow/lavis/processors/__init__.py:29`;
`AirKnow/lavis/processors/audio_processors.py:9,10,13`;
`AirKnow/lavis/processors/blip_processors.py:28-68`;
`AirKnow/lavis/configs/models/blip2/blip2_pretrain.yaml:10,21,24-36`;
`AirKnow/cirr_sub_BLIP2.py:6,92-93,117,120`;
`AirKnow/lavis/models/ulip_models/ULIP_models.py:16` (latent, P3);
`AirKnow/README.md:107,114-117,205-213,228-236`.

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 1.** No artifact, no run, no GPU.

Within static analysis: **LEVEL 4** for pin identity, protocol/gallery/reference identity, recall
arithmetic, the two absent class modules, the `torchaudio` import chain, the checkpoint metadata
(URL/size/digest), and the checkpoint payload type (`nn.Module` pickle). **LEVEL 3** for the
dependency list (resolved by reading, not by installing) and for the auxiliary-asset availability
searches (public trees only).

**UNKNOWN, stated rather than guessed:** whether the 50 %/80 % checkpoints load and execute on a GPU
host; whether their scores are 53.19/74.04 and 49.23/70.23 as the paper's rows claim; whether the
authors retain the correction dictionaries and the resized tree and would share them; whether an
earlier 0 %/20 % checkpoint was ever published and withdrawn; and whether the `resized_image` tree's
resampling can be reproduced (no recipe anywhere).
