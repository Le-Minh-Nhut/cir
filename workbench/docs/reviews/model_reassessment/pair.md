# pair — reproducibility reassessment

Method: independent adversarial re-audit of the pinned checkout at `workbench/third_party/PAIR`
(verified `git rev-parse HEAD` = `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`, matching
`workbench/registry/models.yaml` `upstream_commit_sha`). Every prior audit document was treated as
a hypothesis and re-checked against the source; disagreements are stated with the code. Written
2026-10-18. **Three facts were newly established by this audit beyond the prior documents**, and
two of them contradict it:

* The authors' Hugging Face checkpoint is a **whole-model pickle, not a state_dict**
  (`weights_only=True` raises `UnpicklingError`; the pickle imports `model_vssp.Encoder`). It was
  written by a different code lineage than the pinned tree.
* Unpickling it (with stub modules, read-only) yields a `PAIR`-shaped architecture, **not** the
  architecture of the pinned `models/model.py` (4 single-head-group transformers
  `transformer_a|b|c|d`, `num_heads=4`, 8-vector `local_weight` vs. the pinned 3 transformers,
  `num_heads` 8/8/4, 4-vector `local_weight`). The release channel is inconsistent with the
  published code beyond the already-known missing module.
* **`pair-B1.pt` has `loss_T = 10.0`, the pinned model's initialisation value**
  (`models/model.py:6` `nn.Parameter(torch.tensor([10.]))`), while the pinned trainer trains it
  (`train.py` optimiser over all parameters). The blob's own state therefore gives no evidence of
  any training that would move `loss_T`, and none of the three datasets is nameable from it. This
  is evidence *for* keeping the mapping UNVERIFIED.

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `pair` | `workbench/registry/models.yaml:477` |
| paper | PAIR: Complementarity-guided Disentanglement for Composed Image Retrieval | `models.yaml:487` |
| venue / year | ICASSP 2025 | `models.yaml:485`; `PAIR/README.md:41` |
| paper URL | https://ieeexplore.ieee.org/document/10888153 | `models.yaml:488` |
| upstream repo | https://github.com/iLearn-Lab/ICASSP25-PAIR | `models.yaml:493` |
| pinned commit | `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f` | `models.yaml:494`; `git rev-parse HEAD` → same |
| commit date / subject | 2026-04-05 15:58:53 +0800, "Update README.md" | `git log -1 --format='%H %ci %s'` |
| checkpoint variants | `pair_b1` `pair-B1.pt`, `pair_b2` `pair-B2.pt`, both `status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED` | `models.yaml:508-539` |
| native protocol | `fashioniq_val_split` | `models.yaml:500` |
| reported metrics | R@10 52.76, R@50 74.51, mean 63.64 (derived) | `models.yaml:502-504`; `paper_benchmarks/papers.yaml:1168-1202` |
| reproduction_status | `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` | `models.yaml:506` |
| command_status | `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` | `models.yaml:509` |

## 2. SOURCE STATUS

Accessible; 21 blobs upstream and 9 tracked non-asset files locally
(`git ls-tree -r HEAD --name-only`): `README.md`, `LICENSE`, `cirr_test_submission.py`, `test.py`,
`train.py`, `data/datasets.py`, `models/MHTransformer.py`, `models/model.py`, `utils/utils.py`.
The clone is **shallow** (`.git/shallow` present, `git rev-list --count HEAD` → `1`), so local
history proves nothing; history claims below were verified over the network. SNAPSHOT 2:

```
$ git -C workbench/third_party/PAIR ls-remote origin
bfb0b98cdb4c44f6b9190baaa84d783a77dd888f	HEAD
bfb0b98cdb4c44f6b9190baaa84d783a77dd888f	refs/heads/main
$ git fetch origin --dry-run   # rc=0, no output
```

Upstream is frozen at the pin: **no branch moved, no CLI added, no dependency fixed, no module
added**. GitHub's tree at the pin confirms the *whole* Python surface:

```
$ curl -s "https://api.github.com/repos/iLearn-Lab/ICASSP25-PAIR/git/trees/bfb0b98cdb4c?recursive=1" \
    | grep -o '"path": "[^"]*"' | grep -i "\.py\""
"path": "cirr_test_submission.py"
"path": "data/datasets.py"
"path": "models/MHTransformer.py"
"path": "models/model.py"
"path": "test.py"
"path": "train.py"
"path": "utils/utils.py"
```

**Defects found by this audit that the prior documents do not mention:**

**S1 — the released checkpoints are not state dicts and not the pinned architecture.** Reported in
§5; it is the single most decision-relevant new fact, because it means "map `pair-B1.pt` to
FashionIQ" is not merely missing a label but is contradicted by the bytes' own structure.

**S2 — the repository is not self-runnable even after the missing module is supplied.** Three
independent import/CLI facts, each observed:

1. `cirr_test_submission.py:12` `datasets.CIRR(path='./', transform=...)` with its own
   `datasets` import at `:6`. The dataset lives at `data/datasets.py`; there is **no**
   `datasets.py` at the repo root (`find . -name datasets.py` → `./data/datasets.py` only), and
   `cirr_test_submission.py` contains no `sys.path` manipulation. From the repo root this is a
   `ModuleNotFoundError`.
2. `train.py:17-18`: `#import model_For2 as model` / `import model as model`, and
   `model.py` lives in `models/`; `train.py` does not touch `sys.path` either. `train.py`, the
   README's documented command, similarly cannot import its own model.
3. `data/datasets.py:33` calls `save_obj(...)`, whose definition is qualified with
   `from utils.utils import save_obj, load_obj` (`data/datasets.py:16`) — `utils` resolves only from
   the repo root, while `datasets` resolves only from `data/`. `PYTHONPATH=root:root/data` is the
   only arrangement in which both work. This is an upstream packaging defect, not a workbench one,
   and it is what actually makes the "no CLI" problem unrecoverable without a source change.

## 3. NATIVE EVALUATION WORKFLOW

**There is no runnable evaluation entrypoint.** `test.py` is a library: its only top-level
statements are four imports (`test.py:1-4`) and two function definitions —
`test(params, model, testset, category)` (`test.py:6`) and `test_cirr_valset(params, model, testset)`
(`test.py:85`). The file has **no** `__main__` block, no `argparse` (the only `argparse` in the repo
is `train.py:2`), and is never called by `train.py` (`grep -n "test\." train.py` → no match).

`train.py` is the only CLI, and its `__main__` ends at `train.py:153` with
`logging.info("Starting train for {} epoch(s)"...)` — the file has **no training call and no
evaluation call** after `create_model_and_optimizer()`. `save_ckpt` is defined at `train.py:123`
and never called. So the official repository contains no code path that produces or evaluates a
FashionIQ checkpoint, and `cirr_test_submission.py` is CIRR-only and itself unimportable (§S2.1).

Chain that *would* have to run for Protocol C, and what each step needs:

```
test.test(params, model, testset, category)                        test.py:6
  -> testset.test_queries_{dress,shirt,toptee} / test_targets_*     test.py:9-15
  -> data/datasets.py::FashionIQ.get_test_data('..')                data/datasets.py:117-171
       split.{cat}.val.json / cap.{cat}.val.json / correction_dict CATEGORIES
  -> model.extract_retrieval_compose / extract_retrieval_target     test.py:33,49
  -> models/model.py::PAIR -> Backbone.__init__                     models/model.py:37-47
  -> models/MHTransformer.py::Transformer  (present)                models/model.py:45-46
  -> MHTransformer_rev.Transformer (ABSENT)                         models/model.py:13,47
  -> open_clip.create_model_and_transforms('ViT-B-32',
       pretrained=os.path.join('./', 'open_clip_pytorch_model.bin')) models/model.py:37-38
```

`params` must supply `local_rank` (`test.py:26,43`), `batch_size` (`test.py:29,45`) and `dataset`
(`test.py:145,158`) — verified directly against the source; all three exist in `train.py`'s parser
(`train.py:27,28,35`), which is the only `argparse.Namespace` in the repository, so any wrapper
must obtain `params` from `train` (importing it, or replicating the parse) and hand it to
`test.test`. The workbench records no note about this: `workbench/backend/adapters/base.py` is 43
lines and contains no comment about `params`.

**Workbench integration is correctly fail-closed but under-modelled.** `PairAdapter`
(`workbench/backend/adapters/models.py:49-51`) declares `model_id = "pair"`, `script = "test.py"`
and inherits `OfficialScriptAdapter.command`, which raises
`NotImplementedError("upstream evaluation command requires audit before execution")` (`:24-25`).
Consequently `registry.checkpoint_availability` computes
`adapter_command_ready = False` — not because the model carries an explicit `command_status`
(verified: `model_by_id('pair')` has **no** `command_status` key; `models.yaml`'s
`command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` at `:509` is the *next* row's, `clvc_net`'s) —
but because `getattr(adapter, "command", None) is not OfficialScriptAdapter.command`
(`registry.py:980-984`). That is the right gate for the wrong reason: it blocks PAIR for "no
command written" while the real reason is "the source has no entrypoint". Observed blocker set:

```
pair pair_b1 ['command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT',
              'checkpoint missing: .../artifacts/checkpoints/pair/pair-B1.pt',
              'checkpoint mapping unresolved']
pair pair_b2 [same three]
```

(The `command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` string is produced by
`workbench/backend/preflight.py:206` reading `model.get("command_status")` — a key this row does
not have — so the message attributes a source defect to a status the registry never set. P2,
cosmetic but misleading.)

## 4. DEPENDENCIES

| import | file:line | owner | required version | installable? | classification | safe remedy |
|---|---|---|---|---|---|---|
| `MHTransformer_rev` | `models/model.py:13` (used `:47`) | upstream | — never published | **no** | `UNPUBLISHED_SOURCE_MODULE` | see §9; not installable, not substitutable |
| `open_clip` | `models/model.py:9,37-38`; `train.py:14,65`; `cirr_test_submission.py:4` | upstream (PyPI-backed) | `open-clip-torch==2.24.0` (`ENCODER/requirements.txt:155`, same author family) | yes | — | pin; stages the `open_clip_pytorch_model.bin` asset |
| `torch` / `torchvision` | `models/model.py:1`; `train.py:8` | upstream | 2.0.0 / (README) cu118 | yes | — | keep |
| `transformers` | `models/model.py:7` | upstream | README-era 4.30.x | yes | — | pin |
| `einops` | `models/model.py:11` | upstream | 0.4.x | yes | — | pin |
| `tqdm`, `numpy`, `PIL` | `test.py:3`; `data/datasets.py:3,4` | upstream | — | yes | — | pin |
| `torch` (whole-model pickle at load time) | `cirr_test_submission.py:100`; `train.py:125` (the writers) | upstream | any | n/a | `UNSAFE_CHECKPOINT_SERIALIZATION` | see §5 |

The planner's view of PAIR is **incomplete by construction and this is a real defect**:
`environment_planner.model_constraints(model_by_id('pair'))` returns
`{'tqdm': 'UNKNOWN', 'numpy': 'UNKNOWN'}` with `pip_only: []`, because there is **no
`requirements.txt` in the repo** (`_requirements_path` looks for
`requirements.txt`/`requirement.txt`, `environment_planner.py:223-231`; `git ls-tree` confirms
neither exists) and the registry declares `packages: []` (`models.yaml:485`). The README instructs
`pip install -r requirements.txt` (`README.md:113`) — a file that does not exist. So the
environment contract omits `open_clip`, `einops` and `transformers`, all of which the environment
*does* see as reachable imports (`_evaluator_imports` → `{'tqdm','torch','numpy'}` + the script
chain), while the registry's answers key only on the declaration. Classification:
`PUBLIC_ASSET_NOT_PREPARED` for the missing requirements file is too strong; the accurate label is
`FIRST_PARTY_ENVIRONMENT_CONFIGURATION` (workbench must declare the real packages or the env plan
is silently under-specified).

## 5. CHECKPOINTS

| field | registry | independently observed |
|---|---|---|
| file | `pair-B1.pt` (`models.yaml:519`) | served by `iLearn-Lab/ICASSP25-PAIR` (the *correct* author repo) — see below |
| `download_url` | `https://huggingface.co/ZhihengFu/PAIR/resolve/main/pair-B1.pt` (`models.yaml:518`) | **this URL 401s.** `curl -sIL` → `HTTP/2 401`, and the Python downloader path raises `HTTPError 401` — i.e. `download_checkpoints.py` would fail `download()` at `:170-172` |
| size | `694958565` (`models.yaml:522`) | **VERIFIED** — `x-linked-size: 694958565` at `iLearn-Lab/ICASSP25-PAIR/resolve/main/pair-B1.pt`; full download 694,958,565 bytes |
| sha256 | `null` (`models.yaml:523`) | **VERIFIED** — `c4d861ecd0baa3f1427fc7a853943f5614baf87def1cdd9c4bb7583ee925fbf2` (`sha256sum`) |
| `pair_b2` size | `694872371` (`models.yaml:536`) | **VERIFIED** — `x-linked-size: 694872371`; etag `74346fd7…f03f03` |
| serialization | *(field absent for this row)* | **whole-model pickle**, not a state dict — §5.1 |
| mapping | `UNVERIFIED` (`models.yaml:526`) | unchanged; still no author statement binds either blob to a dataset |

**5.1 The "wrong URL" defect is real and reproducible.** The registry points at
`ZhihengFu/PAIR`; that namespace no longer serves anything:

```
$ curl -sIL https://huggingface.co/ZhihengFu/PAIR/resolve/main/pair-B1.pt   -> HTTP/2 401
$ python -c "urllib.request.urlopen(Request(<that url>, method='HEAD'))"     -> HTTPError 401
$ curl -s https://huggingface.co/api/models/ZhihengFu/PAIR                  -> {"error":"Invalid username or password."}
$ curl -s -o /dev/null -w '%{http_code}' https://huggingface.co/ZhihengFu   -> 404
```

The working author location is `iLearn-Lab/ICASSP25-PAIR` (created 2026-04-10, public:
`gated: false`, `private: false`, siblings exactly `.gitattributes`, `README.md`, `pair-B1.pt`,
`pair-B2.pt`), whose files serve with the same sizes the registry already records. Comparing to a
control URL (`https://huggingface.co/bert-base-uncased/resolve/main/nope.bin` → **404**) shows
401 is "repo not found/private" for anonymous callers, so this is genuinely broken, not a
transient restriction. Classification: `CHECKPOINT_URL_INACCESSIBLE` (P1) — a one-line registry fix
that unblocks nothing by itself, because the *mapping* remains unresolved.

**5.2 The released blobs are whole-model pickles of a different architecture.**
`weights_only=True` is the safe loader the workbench sometimes uses
(`workbench/replay/common.py:52-54`, which allows `weights_only=False` only after matching trusted
author bytes); here it **fails**, which is a positive result about how the bytes were produced:

```
$ python -c "torch.load('/tmp/pair-B1.pt', map_location='cpu', weights_only=True)"
torch._weights_only_unpickler.UnpicklingError: Weights only load failed. This file can still be loaded,
to do so you have two options ... Re-running torch.load with weights_only=False ...
```

The pickle's only non-torch globals (from `data.pkl`'s `GLOBAL` opcodes, 38 of them) are:

```
model_vssp Encoder            <- root object
model_vssp Backbone
model_vssp FeatureWiseAffine
MHTransformer Transformer
MHTransformer TransformerLayer
MHTransformer MultiHeadAttention
MHTransformer MlpTransformer
MHTransformer_dis Transformer
MHTransformer_dis TransformerLayer
MHTransformer_dis MultiHeadAttention
MHTransformer_dis MlpTransformer
open_clip.model CLIP / open_clip.transformer {VisionTransformer, LayerNorm, Transformer, ResidualAttentionBlock}
open_clip.tokenizer SimpleTokenizer / regex._regex compile / _codecs encode
```

So the writer did `torch.save(model, path)` from a module named `model_vssp` — which the author
never published (`models/model.py` is the only model module) — and no `MHTransformer_rev` appears
anywhere in the pickle. (The checkpoint's zip member is named
`model_epoch_1_ShuffleTrue/data.pkl`, i.e. epoch 1 of a run named `ShuffleTrue`.) Loading it
therefore needs `model_vssp`, `MHTransformer`, `MHTransformer_dis`, a matching `open_clip` and
`regex` — five external objects — which is why `torch.load` dies with
`ModuleNotFoundError: No module named 'model_vssp'` in a bare environment.

**5.3 A stub-unpickle shows the architecture is not the pinned one.** Grafting throwaway stub
modules (`/tmp/pk.py`, `/tmp/pk2.py`, scratch only) lets the pickle load and exposes the real
structure. Comparing against the pinned model's own `state_dict` (built by importing the pinned
`models/model.py` with a stubbed `open_clip` and `MHTransformer_rev := MHTransformer`,
`/tmp/pair_modelkeys.py`):

| aspect | released `pair-B1.pt` | pinned `models/model.py` |
|---|---|---|
| top-level groups | `backbone.clip` 376, `backbone.transformer` 14, `transformer_a/b/c/d` 28 each, `affine.MLP` 6, `mlp`/`mlp_img`/`mlp_text` 6 each, `weight_chanelImg`/`_t` 6, `cross_t2i` 2, `fc`/`text_fc` 2 | `backbone.clip` 1 (stub), `backbone.transformer` 12, `transformer_a` 12, `transformer_b` 24, **no `transformer_c`/`transformer_d`**, `mlp`/`mlp_img`/`mlp_text` 6, `affine.MLP` 6, `fc`/`text_fc` 2, **no `weight_chanelImg`, no `cross_t2i`** |
| transformer count | 5 groups (`transformer`, `_a`, `_b`, `_c`, `_d`) | 3 groups (`transformer`, `_a`, `_b`) |
| transformer depth | `transformer` 1 layer; `_a.._d` 2 layers each | `transformer`, `_a` 1 layer; `_b` 2 layers (`models/model.py:45-47`) |
| `num_heads` on `transformer_a`/`_b` | **4** | 8 (`models/model.py:45-46`) |
| `local_weight` | length 8 | `local_token_num + 2` = 3 with the README's FashionIQ `P=1`; length 4 by construction (`models/model.py:8`) |
| `local_weight_dis` | length 7 | `local_token_num + 1` (`models/model.py:9`) |
| extra non-parameter attributes | `backbone.ini_is/ini_ix/ini_ts/ini_tx`, `weighted` | absent (`grep -n "ini_is\|ini_ix\|ini_ts\|ini_tx\|weight_chanelImg\|cross_t2i\|transformer_c\|transformer_d"` over the pinned tree → only `class MultiHeadAttentionRev`, `models/MHTransformer.py:61`) |
| `loss_T` | **`10.0`** (untrained initialisation, `models/model.py:6` = `torch.tensor([10.])`) | — |
| `t` | 10 (float) | `--tau_` default 10 in `train.py`'s sibling family |

Consequences: (1) the released weights cannot be loaded into the pinned model by key name — 376
`backbone.clip.*` keys have no pinned counterpart in a stub environment and, more importantly, the
transformer groups do not correspond; (2) the writer's code (`model_vssp`, `MHTransformer_dis`,
four head-groups, `ini_*` buffers, `weight_chanelImg`) is **not in the pinned tree at all**, so this
is a genuine lineage divergence between the paper's code release and its weight release. Combined
with `loss_T == 10.0`, the honest reading is that the released files are **not demonstrably the
artefacts behind the reported 52.76 / 74.51**, and the registry's `UNVERIFIED` mapping status is not
merely conservative — it is on the *more* generous side. Classification:
`CHECKPOINT_MAPPING_UNVERIFIED` (P0) **plus** `UNSAFE_CHECKPOINT_SERIALIZATION` (P1) for the
pickle-load path any wrapper would have to take, **plus** the `UNPUBLISHED_SOURCE_MODULE`
implication that `model_vssp` itself is unpublished.

Availability class: **obtainable but structurally unloadable-into-the-pinned-code.** Both files are
served with verified sizes and a verified digest for B1; the loader contract
(`cirr_test_submission.py:100` `torch.load(...)`; `train.py:125` `torch.save(model_without_ddp, path)`)
is whole-model.

## 6. AUXILIARY ASSETS

| name | purpose | source | availability | generation possible? |
|---|---|---|---|---|
| `models/MHTransformer_rev.py` | imported `models/model.py:13`, instantiated `:47` | none | `UNPUBLISHED_SOURCE_MODULE` | no — see §9 |
| `open_clip_pytorch_model.bin` | `models/model.py:37-38` `pretrained=os.path.join('./', …)`; also `train.py:66` | public OpenCLIP, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` (`x-linked-size: 605219813`, `x-linked-etag: 1bd3c717…80ad`, observed) | `PUBLIC_ASSET_NOT_PREPARED` | no generation; a public download |
| `resized_image/<cat>/<id>.jpg` | `data/datasets.py:27` `image_dir = path + 'resized_image'`, read `:107` | operator | present locally (`data/fashionIQ_dataset/resized_image/{dress,shirt,toptee}`) | yes |
| `captions/correction_dict_{dress,shirt,toptee}.json` | `data/datasets.py:65-66,122-123` | not bundled upstream; present locally | present locally | derivable |
| `captions/cap.{cat}.val.json`, `image_splits/split.{cat}.val.json` | protocol inputs, `data/datasets.py:119-121` | official FashionIQ | present locally | no |
| pickle caches `test_queries_*.pkl`, `test_targets_*.pkl`, `fashion_iq_data.json` | written `data/datasets.py:44-50`, reloaded `:52-58` | generated | generated | yes |
| `pair-B1.pt` / `pair-B2.pt` | §5 | author HF | obtainable | no |

Nothing PAIR needs is registered: `auxiliary_assets_for_model('pair')` returns `[]` and
`auxiliary_model_ids()` does not contain `pair` (executed). Unlike ENCODER there is no hardcoded
preflight entry for PAIR at all (`preflight.py` mentions `pair` only inside other comments), so the
`open_clip_pytorch_model.bin` requirement is entirely invisible to the tooling for this model.

## 7. FASHIONIQ PROTOCOL

`native_protocol: fashioniq_val_split` (`models.yaml:500`) — **Protocol C**: ordered pair-union
gallery, reference excluded; `supported_protocols: [fashioniq_val_split]` (`models.yaml:502`).
Authoritative, and nothing in this audit proposes changing it.

Verified in the source: `data/datasets.py::FashionIQ.get_test_data` reads the **val** annotations
(`split.{name}.val.json` at `:119`, `cap.{name}.val.json` at `:121`), and builds the gallery under
`if self.split == 'val-split'` (`:148-158`) as a deduplicated, query-ordered union of
`source_img_id` then `target_img_id`; the `elif self.split == 'original-split'` branch (`:160`)
instead takes the full split. `test.py:65-68` masks the reference
(`sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`); categories `dress`/`shirt`/`toptee`
come from `test.py:9-15`; Recall is computed at `k in [1,10,50]` (`test.py:74-81`).

**But the protocol is currently unreachable for two independent reasons**, both of which are
protocol-neutral to fix: (a) there is no entrypoint that drives `test.test` (§3), and (b) the only
path that reads FashionIQ has no `category` argument. `train.py:64-83` `load_dataset()` calls
`datasets.FashionIQ(path=…, transform=…, split=…)` **without** `category=`, whereas the
constructor is `def __init__(self, path, transform=None, split='val-split')`
(`data/datasets.py:23`) and `get_test_data` is always invoked with an explicit name
(`data/datasets.py:44-46`). The two theta branches that would accept a category
(`train.py:65-69`) pass `category=args.dataset` — an argument the pinned constructor does not
declare, i.e. `TypeError: __init__() got an unexpected keyword argument 'category'`. (The same
argument-mismatch pattern was found for HINT in `FINAL_16_MODEL_AUDIT_MATRIX.md:36`.)

## 8. WORKBENCH INTEGRATION

* adapter: `PairAdapter` (`workbench/backend/adapters/models.py:49-51`); inherits the raising
  `command` (`:24-25`); registered in `ADAPTERS` (`:161`).
* planner: `environment_planner.model_constraints('pair')` → `{'tqdm':'UNKNOWN','numpy':'UNKNOWN'}`
  (`pip_only: []`) — under-specified (§4).
* checkpoint prep: registry URL is dead (401) (§5.1); `download_checkpoints.py` would fail on it.
* preflight: generic only — the blockers observed are `command not runnable`,
  `checkpoint missing`, `checkpoint mapping unresolved` (executed via `runtime_blockers`). The
  `command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` string is generated from a
  `command_status` key this row does not carry (§3).
* metric parser: **absent** — `extract_aggregate_metrics('pair', '', '', 0)` →
  `AGGREGATE_PARSER_UNAVAILABLE` (executed; `metrics_extraction.py:398-413`), whereas `test.py:81`
  prints `('{cat}_r{k}', r)` tuples.
* per-query support: none, and none possible from the official output format.
* environment: `verify_environment(model, probe=True)` → `UNCONFIGURED`, `WORKBENCH_PAIR_PYTHON
  not set` (configuration, out of scope).

## 9. CONFIRMED DEFECTS

**P1 — P0 — `models/model.py:13` imports `MHTransformer_rev`, which does not exist anywhere.**
Full enumeration of every import of it in the pinned tree:

```
$ grep -rn "MHTransformer" --include=*.py .
./models/model.py:12: import MHTransformer
./models/model.py:13: import MHTransformer_rev
./models/model.py:45: self.transformer   = MHTransformer.Transformer(dim_self=hidden_dim, num_heads=8, dim_ref=hidden_dim, num_layers=1)
./models/model.py:46: self.transformer_a = MHTransformer.Transformer(dim_self=hidden_dim, num_heads=8, dim_ref=hidden_dim, num_layers=1)
./models/model.py:47: self.transformer_b = MHTransformer_rev.Transformer(dim_self=hidden_dim, num_heads=4, dim_ref=hidden_dim, num_layers=2)
```

That is **all five** references: one import, one module attribute, one constructor. No file of that
name exists in the pinned tree in any form — file, package directory, or archive:

```
$ find . -path ./.git -prune -o -name "*MHTransformer*" -print
./models/MHTransformer.py
$ git ls-tree HEAD models/
100644 blob 6f30a1fbabbdc5d738dd8f1f5a4d0793d6889a88	models/MHTransformer.py
100644 blob bb8c3b14fb5fc6eef8777ee21b9b93513d167199	models/model.py
```

**Same-class-name search: the class name `Transformer` IS defined in the tree, and
`MultiHeadAttentionRev` is present in the same file — so this is *nearly* recoverable:**

```
$ grep -n "^class " models/MHTransformer.py
7:class MlpTransformer(nn.Module):
24:class MultiHeadAttention(nn.Module):
61:class MultiHeadAttentionRev(nn.Module):
98:class TransformerLayer(nn.Module):
119:class Transformer(nn.Module):
```

**But the recovery path is broken by the file itself**: `TransformerLayer.__init__` hardcodes the
non-Rev attention — `models/MHTransformer.py:115`: `self.attn = MultiHeadAttention(dim_self, dim_ref, num_heads, bias=bias, dropout=dropout)`
— and `Transformer.forward`/`Transformer.__init__` (`:119-152`) build only `TransformerLayer`s.
So `MultiHeadAttentionRev` is **dead code inside `MHTransformer.py`**: no code path in the pinned
tree can instantiate it. `Transformer` therefore always means softmax attention, and the pinned
architecture has no way to express the reverse-attention module that `model.py:47` asks for.

Reproduce (this host, real deps absent but the failure is upstream of them):

```
$ python -c "import MHTransformer_rev"
ModuleNotFoundError: No module named 'MHTransformer_rev'
$ python -c "import model"      # from PAIR/models
ModelNotFoundError: No module named 'open_clip'   # after navigating past MHTransformer_rev
```

Fix required (see below): an unpublished
`models/MHTransformer_rev.py` defining a real `TransformerLayerRev` (using `MultiHeadAttentionRev`)
and a `Transformer` that uses it. Its parameter shapes are constrained by the checkpoint — and the
checkpoint does contain `backbone.transformer_c`/`_d` that the pinned model does not (§5.3), so the
missing module is not the only gap.

**P2 — P0 — the workbench adapter has no `command()` and the source has no entrypoint.**
`workbench/backend/adapters/models.py:49-51` + `:24-25`; `test.py` has no `__main__`/argparse;
`train.py`'s `__main__` never reaches evaluation (`train.py:153`). Both facts are true and the
workbench does the right thing (fails closed), but the registry does not record why: the row has no
`command_status` key, so the emitted message invents one (§3).

**P3 — P1 — the registered checkpoint URL is dead.** `models.yaml:518,529` point at
`ZhihengFu/PAIR`, which 401s; the live author repo is `iLearn-Lab/ICASSP25-PAIR` (§5.1).

**P4 — P0 — the released checkpoints cannot be mapped to the paper's FashionIQ row and are
structurally inconsistent with the published code.** §5.2-5.3: whole-model pickle of module
`model_vssp`; 4-head-group transformer stack vs. the pinned 3; `loss_T == 10.0`; `local_weight`
length 8 vs. 4; buffer `ini_*` and modules `weight_chanelImg`/`cross_t2i` that do not exist in the
pinned tree. Do not promote `checkpoint_mapping_status` on the strength of size/hash agreement.

**P5 — P1 — `UNSAFE_CHECKPOINT_SERIALIZATION` on the only path a wrapper could take.** Both
official loaders are `torch.load(path)` with no `weights_only` argument
(`cirr_test_submission.py:100`, and the writer at `train.py:125`), and the bytes require
`weights_only=False` (§5.2). Any adapter must therefore load arbitrary pickles; the workbench's
existing guard (`replay/common.py:52-54`, `weights_only=False` only after matching trusted bytes)
is the pattern to reuse.

**P6 — P2 — no `requirements.txt`, README instructs installing one.** `README.md:113`
`pip install -r requirements.txt`; none exists (`git ls-tree`). The env plan consequently omits
`open_clip`, `einops`, `transformers` (§4).

**P7 — P2 — packaging makes the repo unimportable even after P1 is fixed.** §S2: `datasets` and
`model` are in subdirectories with no `__init__.py` (`data/`, `models/`, `utils/` contain none),
no `sys.path` manipulation anywhere, and `data/datasets.py:16,33` needs `utils` from the root while
`cirr_test_submission.py:6,12` needs `datasets` from `data/`.

**Falsified / corrected hypotheses.** (i) `FINAL_16_MODEL_AUDIT_MATRIX.md:38` says only that
`model.py` "imports a non-existent `MHTransformer_rev`"; correct, and now sharpened to "the class
name exists but the *instantiable* module does not" (§P1). (ii) `cluster_a_encoder_split.yaml:345-361`
declares the missing module as an *asset* with `expected_relative_path: models/MHTransformer_rev.py`
and `asset_role: backbone_weights`; that is mislabelled — the file contains no weights, it is model
definition code, and severity/owner differ accordingly. (iii) `models.yaml:512-513`'s note that
"Existing audit found pair-B1.pt and pair-B2.pt" is right; what no prior document says is that the
URL in the row is dead and the bytes are a different architecture. (iv) The workbench claim that
`pair` carries `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` is **false for this row** as
written (`models.yaml:509` belongs to `clvc_net`); the *effect* the label describes is nonetheless
achieved by `registry.py:980-984`.

## 10. REMAINING BLOCKERS

| # | classification | ownership | evidence | remediation |
|---|---|---|---|---|
| B1 | `UNPUBLISHED_SOURCE_MODULE` — `MHTransformer_rev` | upstream | `models/model.py:13,47`; `models/MHTransformer.py:61,115,119-152`; §9-P1 | needs the author; a workbench-side reconstruction is a **new implementation of part of the model** (see §11) and must not be presented as official parity |
| B2 | `UNPUBLISHED_SOURCE_MODULE` — `model_vssp` (and `MHTransformer_dis`) needed to unpickle the weights | upstream | §5.2 pickled globals | same as B1 |
| B3 | `CHECKPOINT_URL_INACCESSIBLE` | workbench (registry) | §5.1 401/404 | point `download_url` at `iLearn-Lab/ICASSP25-PAIR` |
| B4 | `CHECKPOINT_MAPPING_UNVERIFIED` | upstream + workbench | §5 | keep `UNVERIFIED`; require an author statement or a per-file release note |
| B5 | `UNSAFE_CHECKPOINT_SERIALIZATION` | workbench (adapter policy) | `cirr_test_submission.py:100`; §5.2 | reuse the `replay/common.py:52-54` trusted-bytes gate |
| B6 | `FIRST_PARTY_ADAPTER_MISSING` — no `command()`, no metric parser | workbench | `adapters/models.py:24-25,49-51`; `metrics_extraction.py:398-413` | see §11 |
| B7 | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` — pinned argparse lacks `category`; no `requirements.txt` | upstream + workbench | `data/datasets.py:23`; `train.py:65-69`; `README.md:113` | add the missing arg (source patch) / declare packages |
| B8 | `GPU_RUNTIME_UNTESTED` | host | `test.py:33,49` `.cuda()`; `models/model.py:74` `del`… `.cuda()`-era code | GPU host; no source change |

Not proposed, at any point: an architecture replacement (§11), a gallery change, a reference-policy
change, a caption change, or a noise-condition change. Protocol C is authoritative and the code
implements it (§7).

## 11. IMPLEMENTATION PLAN

**What would be required, precisely** (the assignment forbids proposing an architecture
replacement, so this states the requirement and not a design):

1. **The unpublished module must be supplied faithfully.** `MHTransformer_rev.Transformer` must
   expose exactly what `model.py` calls — `Transformer(dim_self, num_heads, dim_ref, num_layers)`
   with an attribute `layers` — and, given the pinned `MHTransformer.py:115` hardcodes the softmax
   attention inside `TransformerLayer`, it must additionally define a *new* layer class that uses
   `MultiHeadAttentionRev` (`MHTransformer.py:61-95`, whose only behavioural difference from
   `MultiHeadAttention` is `attention = 1 - attention.softmax(dim=2)` at `:87`). Choosing that
   layer's attention semantics is an **architectural decision the paper does not settle in the
   released code**; it cannot be derived from the pinned tree, and it cannot be validated against
   the released checkpoint because that checkpoint's `transformer_b` has `num_heads=4` where the
   pinned model uses 4 for `_b` too but with a different group count (`_c`, `_d` present) — §5.3.
2. **A loader must exist for the released bytes**, i.e. `model_vssp` and `MHTransformer_dis`
   (`torch.load`), or the checkpoint must be converted — and converting requires knowing its
   architecture, which is exactly what is unpublished.
3. **A CLI or a wrapper must drive `test.test`** with `params.local_rank`, `params.batch_size`,
   `params.dataset` (`test.py:26,29,43,45,145`), i.e. via `train`'s parser
   (`test.py:26,43` reads `params.local_rank`, `test.py:29,45` `params.batch_size`, `test.py:145,158` `params.dataset` — verified directly).
4. **Packaging must be fixed** for any of it to import (§S2): `PYTHONPATH=<repo>:<repo>/data`, or
   added `__init__.py`/path setup — a source change.
5. **`open_clip_pytorch_model.bin` must be staged** and CWD must be the checkout (the launcher
   already sets `cwd=source`, `evaluate_models.py:214-215`,429).

Complexity: high — items 1–2 are unrecoverable from published artefacts. Scientific risk: high if
any of 1–2 is guessed, because a reconstructed module changes the model, and a reconstructed
architecture changes the numbers. Expected outputs: none until B1/B2 are resolved by the authors.

## 12. FINAL VERDICT

* `BLOCKED` — `UNPUBLISHED_SOURCE_MODULE` (`MHTransformer_rev`, `model_vssp`) **+**
  `CHECKPOINT_URL_INACCESSIBLE` (registry URL 401s) **+** `CHECKPOINT_MAPPING_UNVERIFIED`
  (no dataset binding; release channel structurally inconsistent with the published code) **+**
  `GPU_RUNTIME_UNTESTED`.
* `NOT_RUN` — nothing was executed beyond static inspection, a full 694,958,565-byte checkpoint
  download + hash, and a stub-module unpickle probe. No evaluation was attempted (no GPU, no
  entrypoint, no mapped checkpoint).
* `VERIFIED` (sub-claims, STATIC/RUNTIME-mixed): pin matches the registry; `MHTransformer_rev` has
  zero occurrences in the pinned tree and zero in the upstream tree at the pin; `Transformer` and
  `MultiHeadAttentionRev` exist but the latter is unreachable; `test.py` has no entrypoint and
  `train.py` never evaluates; Protocol C is what the data code implements; `pair-B1.pt` is
  694,958,565 bytes with sha256 `c4d861ec…fbf2`; it is a whole-model pickle of `model_vssp` whose
  transformer topology differs from the pinned model; the registered download URL returns 401 while
  `iLearn-Lab/ICASSP25-PAIR` serves with the recorded sizes.
* `UNKNOWN`: which dataset (if any) `pair-B1.pt`/`pair-B2.pt` were trained on, whether the reported
  52.76/74.51 correspond to either, and any runtime behaviour of PAIR.

## 13. EVIDENCE

* repo: `https://github.com/iLearn-Lab/ICASSP25-PAIR` @ `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`
  (2026-04-05), verified by `git rev-parse HEAD` and the GitHub trees API.
* key files: `models/model.py:7,9,11,12,13,37-38,45-47`; `models/MHTransformer.py:24,61,87,98,115,119-152`;
  `test.py:1-4,6,9-15,26,29,33,43,45,49,65-81,85`; `train.py:2,14,17-18,26-62,64-83,123,125,128-153`;
  `data/datasets.py:16,23,27,33,44-50,52-58,65-66,107,117-171`;
  `cirr_test_submission.py:4,6,12,91,94,103`; `README.md:119,221-224`;
  `workbench/registry/models.yaml:477-539`; `workbench/registry/paper_benchmarks/papers.yaml:1168-1210` (row `pair_fiq_clean_val_split`);
  `workbench/backend/adapters/models.py:24-25,49-51,161`; `workbench/backend/adapters/base.py:29-32` (the checkpoint-mapping gate);
  `workbench/backend/registry.py:980-984`; `workbench/backend/preflight.py:206`;
  `workbench/backend/metrics_extraction.py:398-413`; `workbench/backend/environment_planner.py:223-231`;
  `workbench/replay/common.py:52-54`; `workbench/scripts/evaluate_models.py:214-215,429`;
  `workbench/registry/audits/cluster_a_encoder_split.yaml:340-380`.
* commands (read-only; scratch under `/tmp`): `git rev-parse HEAD`, `git ls-tree -r HEAD`, `git ls-remote origin`,
  `git fetch origin --dry-run`, GitHub trees API for `bfb0b98cdb4c`;
  `curl -sIL https://huggingface.co/ZhihengFu/PAIR/resolve/main/pair-B1.pt` → `401`;
  `curl` to `iLearn-Lab/ICASSP25-PAIR` → `x-linked-size 694958565`, then a full download and
  `sha256sum` → `c4d861ec…fbf2`; ranged curl of `pair-B2.pt` head;
  `torch.load(..., weights_only=True)` → `UnpicklingError`; `pickletools.genops` on
  `model_epoch_1_ShuffleTrue/data.pkl` for the 38 `GLOBAL` opcodes;
  stub-module unpickle probes `/tmp/pk.py`, `/tmp/pk2.py`, and the pinned-model key dump
  `/tmp/pair_modelkeys.py`; `.venv/bin/python` calls into
  `workbench.backend.{registry,preflight,runtime,environment_planner,metrics_extraction}`.
* observed results: recorded inline in §3, §5, §9.

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 3 of 6.** Level 1 = source/pin integrity (achieved). Level 2 = upstream absence of
`MHTransformer_rev` confirmed both locally and over the GitHub API at the pinned tree (achieved).
Level 3 = the checkpoint bytes were fetched and characterised: size and sha256 measured, the file
proved to be a whole-model pickle whose module graph and transformer topology were extracted and
compared against a locally built copy of the pinned model's `state_dict` (achieved; all of §5, §9-P1
and the transformer table are observed facts). Level 4 (a real evaluation producing R@10/R@50) is
NOT reached and is unreachable without the unpublished modules. Levels 5–6 out of reach.
STATIC_AUDIT except the §5/§9 checkpoint measurements and the stub-unpickle probes, which are
RUNTIME_VERIFIED on this host.
