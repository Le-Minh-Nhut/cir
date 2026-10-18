# clvc_net — reproducibility reassessment

Independent adversarial reassessment of registered model `clvc_net` (CLVC-Net).
READ-ONLY: nothing under `/home/heheboiz/data/cir` was modified except this file. All probe output
went to `/tmp` (`/tmp/clvc_probe/`, `/tmp/clvc_api_probe.py`). No GPU inference, no weight download,
no checkpoint fetch, no install. Every claim below is **STATIC_AUDIT** unless it carries an observed
command+output; this report contains **zero RUNTIME_VERIFIED** claims.

Existing workbench audits were treated as hypotheses only. Where I re-derived a claim I say so; where
I disagree I show the code.

- Workbench HEAD at audit time: `b70d092e9d2232e34886f819e30c7327b74f3a76` (`git rev-parse HEAD`).
- Pinned checkout: `workbench/third_party/CLVCNet` at `bd9b6889489806cb60676597880f91d918b279cb`.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `clvc_net` | `workbench/registry/models.yaml:538` |
| method | CLVC-Net — "Comprehensive Linguistic-Visual Composition Network for Image Retrieval" | `models.yaml:554` (title `:557`) |
| year / venue | 2021, SIGIR | `models.yaml:556` |
| paper URL | `https://doi.org/10.1145/3404835.3462967` | `models.yaml:558` |
| source URL | `https://github.com/iLearn-Lab/SIGIR21-CLVC-Net` | `models.yaml:559` |
| pinned commit | `bd9b6889489806cb60676597880f91d918b279cb` | `models.yaml:561` |
| checkpoint variant | `author_drive_unresolved` / `checkpoint-unresolved.pt` | `models.yaml:595-596` |
| reported metrics | R@10 30.7, R@50 58.41, mean 44.56 (mean is WORKBENCH_DERIVED) | `models.yaml:575-577` |
| native protocol | `fashioniq_val_split` | `models.yaml:569`; `supported_protocols: [fashioniq_val_split]` `:570` |
| reproduction_status | `BLOCKED_CHECKPOINT_MAPPING` | `models.yaml:578` |
| command_status | `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` | `models.yaml:579` |
| environment | python 3.7.6 / torch 1.6.0 / **torchvision UNKNOWN** / cuda 10.2 | `models.yaml:583,584,585,590` |

The registry's paper row is corroborated by a committed extraction, not by me re-reading the PDF:
`workbench/registry/paper_benchmarks/extracted/clvc_net_sigir21.json` carries
`reported_metrics {r10:30.7, r50:58.41, mean:null}`, `category_metrics` dress 29.85/56.47, shirt
28.75/54.76, toptee 33.50/64.00, and the arithmetic
`(29.85+28.75+33.50)/3 = 30.70`, `(56.47+54.76+64.00)/3 = 58.41` is internally consistent.
I did not fetch the PDF myself; that row is STATIC_AUDIT *of the committed extraction*, not of the
paper bytes. STATIC_AUDIT.

---

## 2. SOURCE STATUS

- `source_available: true` (`models.yaml:560`), `source_dir: CLVCNet` (`:563`).
- OBSERVED `git -C workbench/third_party/CLVCNet rev-parse HEAD` →
  `bd9b6889489806cb60676597880f91d918b279cb` = `upstream_commit_sha` (`models.yaml:561`). Pin exact.
- OBSERVED `git -C ... rev-list --all --count` → `1`; `git log --oneline` → `bd9b688 Update README.md`.
  Single-commit snapshot: no history-level claim ("this file never changed") is testable here.
- OBSERVED `git status --porcelain` → `?? __pycache__/` only (a byproduct of import probes; the
  checkout itself is unmodified). `__pycache__` is not a source edit.
- Tree (OBSERVED): `README.md`, `blocks.py`, `datasets.py`, `model.py`, `resnet.py`, `test.py`,
  `text_model.py`, `train.py`, `utils.py`. No requirements file, no resize script, no checkpoint.
- **Complete for what the evaluator needs?** Yes for code: every module the evaluation path imports
  exists. OBSERVED, under a one-line shim for the removed torchvision submodule (§4),
  `importlib.import_module` succeeds for all of `blocks, datasets, resnet, text_model, model, test`.

SNAPSHOT 2 — current upstream state, OBSERVED this pass from the pinned checkout (no fetch, no
checkout change, pin untouched):

```
$ git -C workbench/third_party/CLVCNet ls-remote origin HEAD
bd9b6889489806cb60676597880f91d918b279cb	HEAD
$ git -C workbench/third_party/CLVCNet fetch origin --dry-run     # rc=0, no output
$ git -C workbench/third_party/CLVCNet rev-parse HEAD
bd9b6889489806cb60676597880f91d918b279cb
```

So current upstream HEAD **equals the pin**: the authors have not moved, added a CLI, changed
dependencies, or published a checkpoint beyond the one Drive link. `git branch -a` shows
`remotes/origin/main` only, and the pinned commit is its tip.

---

## 3. NATIVE EVALUATION WORKFLOW

- **Entrypoint: there is none.** `test.py` contains exactly one top-level definition and no `__main__`
  guard. OBSERVED by AST parse of the pinned file:

  ```
  top-level defs: ['test']
  has __main__ guard: False
  imports: [['numpy'], ['torch'], 'tqdm', ['torch.nn.functional']]
  ```

  `test.py:6` `def test(params, local_model, global_model, testset, dataname)`. The README's
  `python test.py` (`README.md:67-71`) therefore executes nothing.
- **Function chain (the only real one):** `train.py:259`
  `t = test.test(args, local_model, global_model, dataset, args.dataset)` inside the epoch loop of
  `train_and_evaluate` (`train.py:229`). The evaluator is a *training-loop callback*, not a program.
- The FashionIQ branch is `test.py:80-149`: queries from `datasets.py:214-230`; gallery from
  `datasets.py:233-257`; per-batch `.cuda()` at `test.py:99,115`; L2 renormalisation `test.py:121-125`;
  `sims = all_queries.dot(all_imgs.T)` `:129`; reference masked `-10e10` `:134-135`;
  `np.argsort(-sims[i,:])[:50]` `:138`; percent R@1/10/50 `:142-148`
  (`'{}_r{}'.format(dataname, k)`).
- **What a wrapper must call:** construct `img_text_model.compose_local(texts, T=7.0, dropout_p=0.0)`
  and `compose_global(texts, T=4.0, dropout_p=0.0)` (`model.py:115-116`, `:187-188`; T values from
  `train.py:51-52` defaults), build a `datasets.FashionIQ(path, name=<category>, split='val',
  transform=...)`, load the whole-model pickles, and call `test.test(params, local, global, testset,
  'fashioniq')`. `test.py` also requires a `params` object exposing only `batch_size`
  (`test.py:96,112`) — `utils.Params` or argparse both satisfy it.
- **Missing components for a runnable path:**
  1. any CLI/model construction/loading code (absent from the pin);
  2. the six whole-model checkpoint members (unmapped, §5);
  3. torchvision ResNet-50 V1 weights (§6);
  4. the resized image tree for the declared layout (§6);
  5. an aggregate parser for the model (§8).

---

## 4. DEPENDENCIES — LEGACY / REMOVED RUNTIME APIS (the central question)

Local runtime used for every probe below, OBSERVED:
`/home/heheboiz/data/cir/.venv/bin/python` = CPython **3.13.12**, **torch 2.14.0+cpu**,
**torchvision 0.29.0+cpu**, Pillow **12.3.0**, numpy **2.5.2**.

### 4.1 Removed APIs — machine-checked

| # | API call | file:line | owner | what replaced it | installable? | eval-reachable? | evidence |
|---|---|---|---|---|---|---|---|
| R1 | `from torchvision.models.utils import load_state_dict_from_url` | `workbench/third_party/CLVCNet/resnet.py:3` | upstream (vendored torchvision-0.5-era file) | `torch.hub.load_state_dict_from_url` | n/a (stdlib of torch) | **YES — import-time** | OBSERVED `import resnet -> ModuleNotFoundError No module named 'torchvision.models.utils'`; OBSERVED `from torchvision.models.utils import ... FAIL: ModuleNotFoundError` in the venv |
| R2 | `resnet.resnet50(pretrained=True)` + `model_urls[arch]` dict | `model.py:72`; `resnet.py:218-223`, `resnet.py:9-19` | upstream | `weights=ResNet50_Weights.*` | n/a | **YES — eval** | OBSERVED warning text from `torchvision/models/_utils.py:207`: "The parameter 'pretrained' is deprecated since 0.13 and may be removed in the future"; OBSERVED `resnet50(pretrained=False)` builds under the shim |
| R3 | `torch.autograd.Variable(...)` | `test.py:28`, `test.py:47`; `model.py:163`, `:178`, `:234`, `:249`; `text_model.py:96`; `train.py:184`, `:187` | upstream | nothing — pass the tensor | n/a | mixed (see 4.2) | OBSERVED `Variable(tensor) -> Tensor`, `shares storage: True`, `requires_grad False` in torch 2.14.0 |
| R4 | `from torch.autograd import Variable` | `blocks.py:4` (**unused**), `train.py:13` | upstream | — | n/a | `blocks.py` import only; `train.py` training path | OBSERVED `from torch.autograd import Variable` imports OK in 2.14.0; `grep -c Variable blocks.py` → 1 (the import only) |
| R5 | `type(x[0]) is str or type(x[0]) is unicode` (Python 2 leftover) | `text_model.py:80` | upstream | `isinstance(x[0], str)` | n/a | **latent**, not taken on this path | OBSERVED `guard('hello') -> True`; `guard([1,2,3]) -> NameError name 'unicode' is not defined` |
| R6 | `img = PIL.Image.open(f)` after bare `import PIL` | `datasets.py:5, 101, 197, 452` | upstream | `from PIL import Image` | n/a | **YES — eval** (`get_img`, `datasets.py:197`) | OBSERVED fresh interpreter: `import PIL; hasattr(PIL,'Image')` → `False`; but OBSERVED `import torch` (present at `datasets.py:6`) makes it `True`, so the failure is order-dependent and does not fire under the pinned import order |
| R7 | `torch.load(checkpoint)` with modern `weights_only` default | `utils.py:136` | upstream | `weights_only=False` for whole-module pickles | n/a | **NOT reachable** (dead helper, §5) | OBSERVED `torch.load(<whole module>)` → `UnpicklingError: Weights only load failed`; `weights_only=False` → OK |
| R8 | `from torch.cuda.amp import autocast, GradScaler` | `train.py:24` | upstream | `torch.amp` | n/a | import-time of `train.py` (training entry) | OBSERVED imports OK in 2.14.0 |
| R9 | `logging.info(...)` with no `basicConfig` | `train.py:253-266` | upstream | `logging.basicConfig` | n/a | training path | `train.py` imports `logging` (`:4`) and calls `utils.set_logger`; no root handler is configured for the `logging.info` calls in `train_and_evaluate`, so those messages are emitted at root level with no handler |

**Direct answer to the task's framing:** there is **no** `torchvision.transforms.functional.*` call,
and no per-op `transforms.functional` usage anywhere in the pin. OBSERVED
`grep -n "transforms.functional|functional_tensor|ToPILImage" *.py` → **zero matches**. The
torchvision surface is only the module-level `torchvision.transforms.*` classes
(`train.py:78-140`: `Compose`, `Resize`, `RandomCrop`, `CenterCrop`, `RandomHorizontalFlip`,
`ToTensor`, `Normalize`) — all of which still exist: OBSERVED `Compose True / Resize True /
RandomCrop True / CenterCrop True / RandomHorizontalFlip True / ToTensor True / Normalize True` in
torchvision 0.29.0. There is also no `Variable`-in-`volatile`, no `torch.range`, no
`torch.storage`, no `np.float`/`np.int` in live code (`datasets.py:368,383` occurrences are inside
comments). **R1 is the only hard removal, and it is an import-time failure, not a silent one.**

### 4.2 Is each such call reachable from the evaluation path or training-only?

- **Evaluation-reachable, fatal:** R1 (`resnet.py:3`). Any construction of `img_backbone`
  (`model.py:69-76`) imports `resnet` — and `img_backbone` is instantiated once inside each
  compositor (`model.py:123` in `compose_local`, `model.py:195` in `compose_global`), so every
  pickled model for every category needs it (2 members × 3 categories = 6 constructions). The
  evaluation path cannot even import.
- **Evaluation-reachable, functionally a no-op:** R3 at `test.py:28,47` and R3 at
  `text_model.py:96`. OBSERVED `torch.autograd.Variable(t)` **shares storage** and is
  `requires_grad=False`; `test.py` wraps tensors **inside `torch.no_grad()`** (`test.py:83`), so the
  detach is invisible. `text_model.py:96` wraps an integer token tensor that never needs grad.
- **Evaluation-reachable via the loader contract:** R7 if a replay used `utils.load_checkpoint` —
  but the trainer does not use it (§5), and it expects a `checkpoint['state_dict']` shape that the
  author never produced.
- **Reachable only from the *eval* path:** R6 (`PIL.Image` at `datasets.py:197`) — it is on the
  evaluation image path but does not fire under the pinned import order (torch's own `PIL` import at
  `datasets.py:6` populates the attribute).
- **Training-only:** R3 at `model.py:163`/`:234` (`soft_x2 = F.softmax(torch.autograd.Variable(x2))`)
  and at `model.py:178`/`:249` (`labels = torch.autograd.Variable(labels).cuda()`), R5
  (`text_model.py:80`), R8, R9. `model.py:163` is a *behavioural* difference between eras: in the
  torch-1.6 codebase `Variable(x2)` was the (no-op) idiom, whereas under modern autograd
  `Variable(t, requires_grad=False)` **detaches** `x2` from the graph. OBSERVED on torch 2.14.0:
  with `requires_grad=True` inputs, `q2.grad` stays `None` through
  `F.softmax(torch.autograd.Variable(x2))` while the plain-tensor variant produces a gradient.
  **This affects training only** — `test.py` never calls `compute_loss`.
- **R3 in the *training* loss is NOT implicitly broken by the `.cuda()` calls**: `model.py:178` /
  `:249` call `.cuda()` unconditionally, so this evaluator/model pair is CUDA-hard on the training
  path; the registry records the same for `test.py:28,47` (`models.yaml:586-589`).

### 4.3 Verified numeric neutrality of the two required modernization steps

Both experiments below are STATIC_AUDIT probes on torch 2.14.0 (no model weights, synthetic tensors):

1. **`Variable` → plain tensor is bit-identical on this code base's loss math.** OBSERVED, identical
   inputs, same seed:
   `Variable labels: 4.279970169067383  plain labels: 4.279970169067383  bit-equal: True  grad equal: True`
   and for the KL path `kl: 2.3548319339752197 2.3548319339752197  bit-equal: True`.
2. **The `resnet.py:3` replacement is the same function.** The pinned `resnet.py:9-19` `model_urls`
   hard-codes exactly the URL torchvision 0.5.0 used, and torchvision 0.5.0's own
   `torchvision/models/utils.py` is a two-line re-export — OBSERVED fetched from
   `https://raw.githubusercontent.com/pytorch/vision/v0.5.0/torchvision/models/utils.py`:

   ```python
   try:
       from torch.hub import load_state_dict_from_url
   except ImportError:
       from torch.utils.model_zoo import load_url as load_state_dict_from_url
   ```

   So `from torch.hub import load_state_dict_from_url` is *literally* what the pinned era resolved to.
   OBSERVED `torch.hub.load_state_dict_from_url` exists in 2.14.0 and downloads to
   `$TORCH_HOME/hub/checkpoints` (observed default-dir logic in `torch/hub.py`: "The default value of
   `model_dir` is `<hub_dir>/checkpoints`"), and OBSERVED the exact URL is live:
   `200 102502400 https://download.pytorch.org/models/resnet50-19c8e357.pth`.
   **Do not** "modernize" this to `models.resnet50(weights=ResNet50_Weights.DEFAULT)`: OBSERVED
   `ResNet50_Weights.DEFAULT = ResNet50_Weights.IMAGENET1K_V2` whose URL is
   `.../resnet50-11ad3fa6.pth`, a *different* weight set — that would change every computed feature
   and every recall number.

### 4.4 Minimum modern-runtime adaptation

| step | exact edit | changes a computed number? |
|---|---|---|
| A | `resnet.py:3` → `from torch.hub import load_state_dict_from_url` (one line; the same symbol the 0.5.0 path re-exported) | **No** — same V1 state_dict, same URL, byte-identical weights |
| B | when a whole-model pickle is loaded: `torch.load(path, weights_only=False)` (or `map_location`) | **No** — same bytes deserialized |
| C | (optional, non-behavioural) drop the `Variable` wrappers at `test.py:28,47`, `text_model.py:96`; replace `torch.utils.data import dataloader` (`train.py:14`) with `DataLoader` | **No** — §4.3 item 1 |
| D | **must NOT be done**: replacing the `Variable` wrapper at `model.py:163`/`:234` or removing the implicit detach | **Yes, on training** — would restore a gradient that the pinned code suppressed |
| E | **must NOT be done**: switching to the modern default ResNet-50 weights enum | **Yes, everywhere** |

Steps A+B are the whole adaptation for *evaluation*. Neither is a patch to scientific logic; both
are import/loader compatibility. A and D/E must not be conflated: the same one-line import fix is
safe, the same "modernize the deprecated call" instinct applied to `pretrained=`/`Variable` is not.

The workbench has no approved-patch mechanism in use for this model: `models.yaml:538-605` declares
no `approved_source_patches`, so any such edit must be a reviewed replay wrapper, never an in-place
patch of the pinned checkout.

### 4.5 Dependency categories (D1–D8)

The task refers to a D1–D8 dependency taxonomy. **It is not defined anywhere in this repository** —
OBSERVED `grep -rn "D1" workbench/docs` matches only unrelated text, and a prior reassessment reached
the same conclusion (`workbench/docs/reviews/model_reassessment/sprc.md:169,173`). I therefore do
**not** invent D-numbers; each dependency is classified by the workbench's own vocabulary in §10:
`LEGACY_RUNTIME_INCOMPATIBILITY` (R1, R2, R5), `UNPUBLISHED_SOURCE_MODULE` (none — all modules
present), `INSTALLABLE_DEPENDENCY_MISSING` (none beyond torchvision, which the registry lists as a
package, `models.yaml:591`), `PUBLIC_ASSET_NOT_PREPARED` (§6).

---

## 5. CHECKPOINTS

**Members the pinned trainer actually writes** (STATIC_AUDIT, `train.py:262-275`): on each
best-score epoch it saves two **whole pickled `nn.Module` objects**, not state dicts:

```
train.py:274  torch.save(local_model,  os.path.join(model_dir, 'local_model.pt'))
train.py:275  torch.save(global_model, os.path.join(model_dir, 'global_model.pt'))
```

`model_dir` is `--model_dir ./<category>` (`train.py:60`; README.md:31-49), so a full benchmark
result is **6 members**: `{local,global}_model.pt` × {dress, shirt, toptee}. This matches the
registry's `category_checkpoint_structure.per_category_members: [local, global]`,
`member_count_expected: 6` (`models.yaml:546-551`).

**Nothing in the pin loads them.** OBSERVED `grep -rn "local_model.pt|global_model.pt" *.py` → only
the two `torch.save` lines. The one loader in the tree, `utils.load_checkpoint` (`utils.py:126-141`),
expects `checkpoint['state_dict']` and is never called by `train.py`/`test.py`. `--restore_file` is
declared (`train.py:62-64`) and passed to `train_and_evaluate` (`train.py:306`), whose body
(`train.py:229`) never reads it. So the loader contract is: **the wrapper must construct both
compositors and `torch.load` the whole modules itself.**

**A consequence that is easy to miss:** a whole-module pickle stores the *class's module path*.
OBSERVED with a probe (`/tmp/clvc_probe/model_x`, a package named differently from the source tree):
the pickle stream contains `cmodel_x\ncompose_local\nq\x00)\x81...` — i.e. pickle opcode `c` (GLOBAL)
followed by module and qualified name. A wrapper must therefore import the pinned files as top-level
modules named exactly `model`, `blocks`, `text_model`, `resnet` (they are already importable that way
from the checkout root — OBSERVED `import model` OK under the §4.1 shim). Putting them in a package
would make unpickling fail.

**Availability / mapping:**

- registry variant `author_drive_unresolved`, `filename: checkpoint-unresolved.pt`,
  `download_url: null`, `expected_size_bytes: null`, `expected_sha256: null`,
  `status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED`
  (`models.yaml:595-605`).
- `source_url` = `https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view`
  (`models.yaml:598`), also linked from `README.md:74` as "Pre-trained models".
- OBSERVED this pass, read-only: fetching that Drive page returns
  `<title>CLVCNET.zip - Google Drive</title>` — a **single ZIP**, no member listing, no per-file
  hash, no size exposed. A `HEAD` on the direct-download endpoint returns
  `200 0 text/html` (the Drive interstitial), so **no size or digest is obtainable** without an
  actual download, which this audit does not perform.
- OBSERVED: `checkpoint_path` would resolve to
  `$WORKBENCH_CHECKPOINT_ROOT/clvc_net/checkpoint-unresolved.pt` (`registry.py:842-863`), a
  placeholder name; `checkpoint_missing_paths` (`registry.py:365-395`) would return it, so
  `checkpoint_is_present` is False. Independently, `validate_request` (`adapters/base.py:29-36`)
  refuses first on `checkpoint_mapping_status != VERIFIED_METADATA`.
- **Mapping confidence: none.** No file→category→{local,global} mapping is established by anything in
  the pin, and the registry explicitly forbids asserting one
  (`models.yaml:551`: "do not claim six available files"). The Drive page title tells us the archive
  is called `CLVCNET.zip`; it does not tell us the interior.

**Checksum status:** none. No published size, no published digest, nothing to compare.

**Serialization safety:** the artifact is an arbitrary-code pickle by construction
(`torch.save(<nn.Module>)`). Loading it executes whatever the pickle says. Combined with an
unverified archive from a public Drive link, this is a trust boundary, and it is correctly why the
runbook forbids blind deserialization (`workbench/docs/LEGACY_REPRODUCTION_RUNBOOK.md:8`).

---

## 6. AUXILIARY ASSETS

| name | purpose | source | availability class | generation possible? |
|---|---|---|---|---|
| torchvision ResNet-50 ImageNet **V1** weights, `resnet50-19c8e357.pth` | initialisation of both `img_backbone`s (`model.py:72` → `resnet.py:220-223`); overridden by the whole-model pickle but required to *construct* the module | `https://download.pytorch.org/models/resnet50-19c8e357.pth` (`resnet.py:12`); declared in `preparation_contracts.yaml:44-56` with `expected_sha256_prefix: 19c8e357`, `required_for_evaluation: true` | `SOURCE_HASH_PREFIX_AVAILABLE`; OBSERVED URL live: `200 102502400` | No — must be downloaded (or staged at `$TORCH_HOME/hub/checkpoints/`) |
| FashionIQ raw annotations/splits/images | dataset | canonical root | present locally | n/a |
| `resized_image/{dress,shirt,toptee}` (256, `.jpg`, per-category) | the only image tree the evaluator opens (`datasets.py:194-202`) | workbench recipe `prepare_model.py`; `preparation_contracts.yaml:26-42` | **present locally**: OBSERVED `ls resized_image/dress | sed "s/.*\\.//" | sort | uniq -c` -> `3817 jpg`; `shirt` and `toptee` hold 6346 and 5373 `jpg`, equal to their full-split sizes (the tree is the whole resized split, not the pair union) | already generated |
| (not needed) `simplevocab.json` | `text_model.py:66` only when `texts_to_build_vocab is None` | — | not on the evaluation path: both compositors are built with `texts` (`model.py:119-122`, `:191-194`) and `train.py:300` supplies `trainset.get_all_texts()` (`datasets.py:204-212`) | n/a |

OBSERVED layout-present state this pass:
`data/fashionIQ_dataset/{captions,image_splits,images,resized_image}` with
`resized_image/{dress,shirt,toptee}` populated and **no `cap.*.glove.*.pkl`** (correct for this model)
and **no `resized_images/`** (that is DCNet's directory, correctly absent here). This contradicts the
older claim that the CLVC resized tree is missing — it is present as of this pass.

---

## 7. FASHIONIQ PROTOCOL

Registry: `native_protocol: fashioniq_val_split` (`models.yaml:569`), quoted verbatim:
`fashioniq_val_split` = *"FashionIQ - Val Split (Pair-Union Gallery, Reference Excluded)"*,
`literature_split_label: val`, `evaluation_noise_pct: 0` (`models.yaml:16-19`).

I re-derived the cohort from the staged canonical data rather than trusting any document. OBSERVED,
per category, walking `cap.<cat>.val.json` in file order and appending `candidate` then `target`
first-occurrence-only (exactly `datasets.py:245-249`):

```
data/fashionIQ_dataset
  dress : ann=2017 fullsplit=3817 union=2628 union_subset_full=True targets_all_in_split=True
  shirt : ann=2038 fullsplit=6346 union=3089 union_subset_full=True targets_all_in_split=True
  toptee: ann=1961 fullsplit=5373 union=2902 union_subset_full=True targets_all_in_split=True
```

(identical numbers under `data/FashionIQ`). This matches the earlier audits' 2628/3089/2902 exactly,
and it is a genuine reduction, not the full gallery. Source side: `datasets.py:242-257` with
`gallery_all=False` — the constructor default (`datasets.py:153`), never overridden by
`train.py:86-96`. Workbench side: `build_pair_union_gallery`
(`src/datasets/fashioniq.py:186-209`) uses the same reference-then-target, first-seen rule but keyed
by **id strings** rather than integer split indices.

- gallery: pair-union, built as above — **VERIFIED** as the source's own behaviour.
- reference policy: excluded. Source `test.py:134-135` sets
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` **before** `np.argsort` at
  `test.py:138` — a mask, not a removal. The workbench protocol declares
  `reference_excluded=True` (`src/evaluation/fashioniq.py:171-176` table) and removes the column.
  Procedure differs (VARIANT); the two agree for top-k membership because `-10e10` is below any
  dot-product of L2-normalised features — but that is an argument, not an equality.
- query construction: `concat_text` (`datasets.py:173-175`) →
  `"<BOS> {cap0} <AND> {cap1} <EOS>"`, OBSERVED sample
  `datasets.py:concat_text(None, ["a b","c d"]) -> <BOS> a b <AND> c d <EOS>`.
- categories: dress/shirt/toptee (`datasets.py:152-171`, `train.py:73-96`); images
  `resized_image/<name>/<id>.jpg` (`datasets.py:195`).
- Recall computation: `test.py:142-148`, `r = 100 * r / len(nn_result)`, keys
  `fashioniq_r{1,10,50}`. Truncated ranking `[:50]` (`test.py:138`) contains both k=10 and k=50.
- **Never propose changing** the gallery, the reference policy, the caption input or the noise
  condition to make the model runnable — nothing in this report does.

---

## 8. WORKBENCH INTEGRATION

| piece | state | evidence |
|---|---|---|
| adapter | `CLVCNetAdapter(model_id="clvc_net", script="test.py")`, **no `command` override** | `workbench/backend/adapters/models.py:74-76` |
| base command | raises | `adapters/models.py:24-25` `raise NotImplementedError("upstream evaluation command requires audit before execution")` |
| observed behaviour | `[RUN] clvc_net / author_drive_unresolved` then `[SKIP] SKIPPED: adapter command not audited` | OBSERVED `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B workbench/scripts/evaluate_models.py --model clvc_net --dry-run` |
| planner | would run from the checkout root, not `src/`: `source_cwd` returns `source` for every model except csmcir | `evaluate_models.py:214-215` |
| checkpoint prep | would resolve `$WORKBENCH_CHECKPOINT_ROOT/clvc_net/<filename>`; requires `checkpoint_mapping_status == VERIFIED_METADATA` | `registry.py:842-863`; `adapters/base.py:33-36` |
| preflight (layout) | requires `resized_image/`, `resized_image/<cat>`, `cap.<cat>.{train,val}.json`, `split.<cat>.val.json` | `fashioniq_layout.py:44-52`; `registry.py:679-683` |
| preflight (images) | `.jpg` in the per-category directory | `models.yaml:565`; `fashioniq_layout.py:83-89`, `:92-136` |
| preflight (dataset link) | **none declared** — `canonical_dataset_link` exists only for csmcir, combiner, clip4cir, sprc | OBSERVED `awk` over `models.yaml` → lines 37, 665, 726, 848 |
| preflight (observed) | `torchvision ResNet-50 ImageNet weights missing: .../hub/checkpoints/resnet50-19c8e357.pth` and `external runtime asset file missing: ...` | OBSERVED `model_requirements(model, resolve_config())` |
| metric parser | none — clvc_net falls to the catch-all | `metrics_extraction.py:413-422` → `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE` |
| per-query support | none for this model (no replay module; `workbench/replay/` holds `limn*.py`, `tgcir.py`, `common.py`) | OBSERVED directory listing |
| gate | a committed row exists (`clvc_net_sigir21_fiq_table1`, `reference_status: VERIFIED`, `gallery_protocol: fashioniq_val_split`, `reference_excluded: true`) | `workbench/registry/paper_benchmarks/extracted/clvc_net_sigir21.json`; `benchmark_references.references_for_model('clvc_net')` → 1 row |

Note the asymmetry worth flagging: the *gate* now has a verified reference row, but
`models.yaml:575-577` still carries `reported_mean: 44.56`, and the extraction file itself records
`reported_metrics.mean: null` + a separate `derived_metrics` entry with
`provenance: WORKBENCH_DERIVED`, `"(30.70 + 58.41) / 2"`. Both are internally consistent (the
registry field is labelled as derived at `models.yaml:571-574`); the registry must not be read as a
paper-printed mean. P3, no action needed beyond not re-deriving it differently.

---

## 9. CONFIRMED DEFECTS

**C1 — the evaluation path cannot import on any modern torchvision (P1).**
Root cause: `resnet.py:3` imports a submodule deleted after torchvision 0.13 (the module the pinned
file targets, v0.5.0, itself only re-exports `torch.hub`). Reproduction, OBSERVED:
`import resnet -> ModuleNotFoundError No module named 'torchvision.models.utils'`.
Blast radius: `img_backbone` is constructed once per compositor (`model.py:123`, `:195`), so all six
whole-model members need it.
Fix: one-line replacement (§4.4 step A) inside a reviewed replay wrapper. Regression result: after
the shim, OBSERVED all six modules import (`OK blocks / datasets / resnet / text_model / model /
test`) and OBSERVED `resnet50(pretrained=False)` builds. **No numeric change** (§4.3 item 2).

**C2 — no runnable evaluation entrypoint exists (P1).** `test.py` defines a single function and no
`__main__` (AST evidence, §3). The adapter correctly refuses. Fix: a `workbench/replay/clvc_net.py`
in the style of `workbench/replay/tgcir.py` that builds both compositors, loads the whole-module
pickles, and calls `test.test`; never an in-place patch of the pin. Regression: none possible until
the checkpoint exists.

**C3 — checkpoint path is a placeholder with an uninspected archive (P1).** See §5.

**C4 — `pretrained=`/default-weights trap (P1 if mishandled).** The obvious modernization
(`weights=ResNet50_Weights.DEFAULT`) silently swaps V1 for V2 weights
(OBSERVED `DEFAULT = ResNet50_Weights.IMAGENET1K_V2`, URL `resnet50-11ad3fa6.pth` vs the pinned
`resnet50-19c8e357.pth`). Any modernization that does not pin V1 changes every feature. Fix: pin the
URL, not the enum; assert the downloaded filename/sha prefix.

**C5 — `PIL.Image` attribute depends on import order (P2).** `datasets.py:5` `import PIL` +
`datasets.py:197` `PIL.Image.open`. OBSERVED it works only because `torch` (imported at
`datasets.py:6`) happens to populate `PIL.Image`; OBSERVED in a bare interpreter
`hasattr(PIL,'Image')` → `False`. Fix: `from PIL import Image` (a cosmetic robustness fix; no
numeric effect).

**C6 — `torch.load` default flip would break whole-model loading (P2, latent).** `utils.py:136` uses
the bare call and is dead code today; any future replay loading a whole-module pickle must pass
`weights_only=False`. OBSERVED failure otherwise.

**Addendum on disagreement with prior audits.** Two earlier documents name this model's
legacy risk as `torchvision.models.utils.load_state_dict_from_url` alone
(`final_model_audit/clvc_net.md:270,444-447`; `FINAL_16_MODEL_AUDIT_MATRIX.md:39`) and stop there.
That is **incomplete**: the same pinned line pair also carries an unfixed `pretrained=` argument
(`model.py:72` → `resnet.py:220-223`) whose obvious modernization, `weights=ResNet50_Weights.DEFAULT`,
silently selects `IMAGENET1K_V2` — a different weight file (OBSERVED url `resnet50-11ad3fa6.pth`)
than the pinned, declared and preflight-checked V1 (`resnet50-19c8e357.pth`). No prior audit in this
repository mentions that trap (OBSERVED `grep -rn "pretrained.*deprecat|ResNet50_Weights"` over the
three clvc_net audits → no match). Conversely I found **no** support for the broader framing that this
model's suite contains removed `torchvision.transforms.functional.*` calls: OBSERVED zero matches for
`transforms.functional`, `functional_tensor`, `ToPILImage` in the pin. The one import-time blocker is
the submodule; everything else on the list is present, deprecated-but-working, latent, or training-only.

**No P0 defect was found.** Every incompatibility in this model fails loudly at import or at load
time; I found no path that would silently produce a wrong recall number. The two candidate silent
hazards were checked and dismissed: the `ndarray.dot`-then-normalise order (`test.py:121-125`, then
`:129`) matches the intended double normalisation rather than dropping it, and the `[:50]` ranking
truncation (`test.py:138`) still contains both k=10 and k=50.

---

## 10. REMAINING BLOCKERS

| # | classification (workbench vocabulary) | ownership | evidence | remediation |
|---|---|---|---|---|
| B1 | `LEGACY_RUNTIME_INCOMPATIBILITY` | upstream (vendored 0.5-era file) | `resnet.py:3`; OBSERVED ModuleNotFoundError | one-line import replacement in a replay wrapper; pin the V1 URL |
| B2 | `FIRST_PARTY_ADAPTER_MISSING` | workbench | `adapters/models.py:74-76`; OBSERVED `[SKIP] adapter command not audited` | author and review a replay module; it must set cwd, transform, params and model construction itself |
| B3 | `CHECKPOINT_UNPUBLISHED` / `CHECKPOINT_URL_INACCESSIBLE` | author | `models.yaml:595-605` (`download_url: null`, no size, no sha); OBSERVED Drive page title `CLVCNET.zip` with no listing | obtain the ZIP, inventory members, hash each, record the file→category→{local,global} map |
| B4 | `PUBLIC_ASSET_NOT_PREPARED` | operator | OBSERVED preflight: ResNet-50 V1 missing at `~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth` | stage the verified V1 file at `$TORCH_HOME/hub/checkpoints/` |
| B5 | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` | operator | `models.yaml:581` `WORKBENCH_CLVC_PYTHON`; the venv here is py3.13/torch2.14 whereas the pin declares py3.7.6/torch1.6.0 | decide between a legacy env and the modern env + §4.4 wrapper; record which, since it changes nothing numerically only if A/B are the only edits |
| B6 | `GPU_RUNTIME_UNTESTED` | operator | `models.yaml:586-590`; `test.py:28,47` unguarded `.cuda()`; host GPU OBSERVED `NVIDIA GeForce RTX 3050 Laptop GPU, 8.6` (no `nvcc` on PATH) | GPU host + CUDA-enabled torch |
| B7 | `SCIENTIFIC_PROTOCOL_UNVERIFIED` (partial) | workbench | the cohort is VERIFIED from source (§7); whether the *paper* used it is not established by anything I opened | annotate that the `val` literature label is carried by the committed extraction, not by my own read of the PDF |
| B8 | `FIRST_PARTY_ADAPTER_MISSING` (metric path) | workbench | `metrics_extraction.py:413-422` | implement `parse_clvc_output` for `('<dataset>_r<k>', value)` tuples (`test.py:148`) or record manual capture |

Not a blocker: the resized image tree. OBSERVED present and complete at the declared layout
(§6).

---

## 11. IMPLEMENTATION PLAN

1. `workbench/replay/clvc_net.py` (new): `sys.modules['torchvision.models.utils']` shim **or** a
   derived checkout carrying the one-line `resnet.py:3` change, declared as an approved source patch
   with base/derived digests — prefer the shim, since it touches nothing on disk;
   import the pinned files as top-level modules `model`/`blocks`/`text_model`/`resnet` (required by
   the pickle's GLOBAL names, §5);
   build `compose_local(texts, T=7.0, dropout_p=0.0)` and `compose_global(texts, T=4.0,
   dropout_p=0.0)`, `torch.load(..., weights_only=False)`, build `datasets.FashionIQ(path,
   name=cat, split='val', transform=Compose([Resize(256), CenterCrop(224), ToTensor,
   Normalize(ImageNet)]))`, call `test.test(params, local, global, testset, 'fashioniq')`,
   run once per category, emit the three `fashioniq_r10/r50` pairs plus the workbench macro.
2. Adapter: implement `CLVCNetAdapter.command` to invoke the replay with `--category`, `--checkpoint`,
   `--dataset-root`, `--output`.
3. `metrics_extraction.py`: add a `clvc_net` parser for the `('dress_r10', v)`-style tuples.
4. Registry: `approved_source_patches` only if the shim route is rejected.

Complexity: medium (one new module + adapter + parser). Scientific risk: **low** if and only if
§4.4 A/B are the only edits; the trap is C4, not the compatibility work. Expected outputs:
three per-category R@10/R@50 pairs (`test.py:148`) plus a workbench macro, comparable to the
committed row `clvc_net_sigir21_fiq_table1` (30.70 / 58.41) under the gate's 5% tolerance.

---

## 12. FINAL VERDICT

**BLOCKED** — and, on the specific question the assignment asked, **VARIANT** would be wrong and
"runnable with a one-line fix" would also be wrong.

Precise reading:

- The **legacy-API question** for this model is answered: exactly one hard removal exists
  (`resnet.py:3`), it is eval-reachable and import-time fatal, and the replacement is the very symbol
  torchvision 0.5.0 re-exported — VERIFIED numerically neutral. Everything else on the
  "removed API" list is either still present (all `torchvision.transforms.*` classes), unofficially
  deprecated-but-working (`pretrained=`, `torch.autograd.Variable`, `torch.cuda.amp`), latent
  (`unicode`), reachable only in training (`Variable`-detach at `model.py:163`/`:234`), or already
  avoided by upstream's own dead helper (`torch.load`).
- The model is still **BLOCKED** for reasons that the compatibility fix does not touch: no
  entrypoint, a placeholder checkpoint with an uninspected `CLVCNET.zip`, no environment, and an
  absent ResNet-50 V1 file. `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`models.yaml:578`) and
  `replay_status: BLOCKED_NO_SOURCE_REPLAY` (`:539`) remain accurate.

Nothing was reproduced. Nothing was executed end to end. **0 RUNTIME_VERIFIED claims.**

---

## 13. EVIDENCE

Repo / pin:
- `upstream_repo_url: https://github.com/iLearn-Lab/SIGIR21-CLVC-Net`, `upstream_commit_sha:
  bd9b6889489806cb60676597880f91d918b279cb` (`models.yaml:559,561`).
- OBSERVED `git -C workbench/third_party/CLVCNet rev-parse HEAD` → same SHA; `ls-remote origin HEAD`
  → same SHA (SNAPSHOT 2); `rev-list --all --count` → 1.

Key `file:line` citations (all opened this pass):
- `resnet.py:3,9-19,218-223`; `model.py:69-76,72,115-116,123,163,178,187-188,195,234,249`;
  `test.py:6,28,47,80-149,96,112,121-125,129,134-138,142-148`; `text_model.py:5,66,80,96,108`;
  `datasets.py:5,6,152-175,194-202,204-212,233-257`; `train.py:13,24,51-52,60,62-64,70-140,229,259,
  262-275,306`; `utils.py:126-141`; `README.md:18-21,28,31-49,67-74`.
- Workbench: `workbench/registry/models.yaml:16-19,538-605`; `preparation_contracts.yaml:3-56`;
  `workbench/backend/adapters/models.py:24-25,74-76`; `workbench/backend/registry.py:365-395,
  679-683,842-863`; `workbench/backend/preflight.py:24-44,86-100`; `workbench/backend/
  fashioniq_layout.py:6,44-52,83-89,92-136`; `workbench/backend/metrics_extraction.py:413-422`;
  `workbench/scripts/evaluate_models.py:214-215`; `src/datasets/fashioniq.py:186-209`;
  `src/evaluation/fashioniq.py:155-176`; `paper_benchmarks/extracted/clvc_net_sigir21.json`.

Commands and observed results (abridged; identical text quoted in §2–§9):
- `.venv/bin/python -c "from torchvision.models.utils import load_state_dict_from_url"` →
  `ModuleNotFoundError`.
- AST parse of `test.py` → `defs ['test']`, `has __main__ guard: False`.
- Import probe under shim → all six modules OK.
- Pair-union re-derivation over `data/fashionIQ_dataset` → 2628/3089/2902 vs 3817/6346/5373.
- `resnet50(pretrained=False)` → builds; `ResNet50_Weights.DEFAULT` → `IMAGENET1K_V2`.
- `HEAD https://download.pytorch.org/models/resnet50-19c8e357.pth` → `200 102502400`.
- `fashioniq-layout` counts → `resized_image/dress|shirt|toptee` = 3817|6346|5373 `.jpg`.
- `evaluate_models.py --model clvc_net --dry-run` → `[SKIP] SKIPPED: adapter command not audited`.
- `model_requirements(clvc_net, resolve_config())` → two ResNet-50 blocks.

Not run, deliberately: no GPU inference, no weight download, no checkpoint download, no install, no
write outside this file.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 of 6** — (1) identity/registry and (2) static source + protocol auditing are complete and
independently evidenced, including machine-checked import/deprecation probes against torch 2.14.0 /
torchvision 0.29.0 and an independent re-derivation of the gallery cohort. There is **no** dependency
installation, **no** checkpoint acquisition, **no** evaluator execution, **no** metric extraction
from a real run and **no** comparison against the paper. §3's chain and §7's protocol are STATIC_AUDIT.
