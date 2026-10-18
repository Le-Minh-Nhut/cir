# encoder — reproducibility reassessment

Method: independent adversarial re-audit of the pinned checkout at
`workbench/third_party/ENCODER` (verified `git rev-parse HEAD` =
`29a2a31d6a56f677bf450c3be7cdaef423fb7018`, matching `workbench/registry/models.yaml`
`upstream_commit_sha`). Every workbench audit document was treated as a hypothesis and re-checked
against the source; where this reading disagrees with a document it is stated and the code is
quoted. Written 2026-10-18T07:0xZ. Two new facts not present in the prior audits were established
here: (a) the author Google Drive folder the README links is still live and its contents are
byte-identical to the git checkout, which **retires** the "missing `datasets1.py`" defect as a
*discoverability* problem and reframes it; (b) the linked checkpoint is now measured
(631,879,365 bytes, sha256 `0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173`) and
its key set is consistent with the pinned `model_try2.py`, so the prior
`expected_size_bytes: null` / `expected_sha256: null` are now falsifiable statements rather than
unknowns.

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `encoder` | `workbench/registry/models.yaml:415` |
| paper | ENCODER: Entity Mining and Modification Relation Binding for Composed Image Retrieval | `models.yaml:435` |
| venue / year | AAAI 2025 (vol 39, no 5, pp. 5101–5109) | `models.yaml:433`; `ENCODER/README.md:1-2` (Drive copy `/tmp/drive/e.md:1`) |
| paper URL | https://ojs.aaai.org/index.php/AAAI/article/view/32541 | `models.yaml:436` |
| upstream repo | https://github.com/iLearn-Lab/AAAI25-ENCODER | `models.yaml:437` |
| pinned commit | `29a2a31d6a56f677bf450c3be7cdaef423fb7018` | `models.yaml:439`; `git -C workbench/third_party/ENCODER rev-parse HEAD` → same |
| commit date / subject | 2026-05-25 18:00:29 +0800, "Update README.md" | `git log -1 --format='%H %ci %s'` |
| checkpoint variant | `checkpoint_id: fashioniq`, `filename: fashioniq.pt`, `download_url: null`, `status: URL_UNRESOLVED` | `models.yaml:457-475` |
| serialization | `state_dict` | `models.yaml:430` |
| native protocol | `fashioniq_val_split` | `models.yaml:444` |
| reported metrics | R@10 56.13, R@50 77.59, mean 66.86 (paper Text/stated; mean WORKBENCH_DERIVED) | `models.yaml:447-449`; `workbench/registry/paper_benchmarks/papers.yaml:1104-1140` (row `encoder_fiq_clean_val_split`; stated values and their verification_method quoted at `papers.yaml:1155-1166`) |
| reproduction_status | `NOT_RUN` | `models.yaml:450` |

## 2. SOURCE STATUS

Accessible and complete for the *code*: 11 tracked files, 8 Python modules
(`git -C .../ENCODER ls-files`) and the checkout is shallow (`git rev-list --count HEAD` → `1`,
`.git/shallow` present), so local history proves nothing about what upstream ever contained —
history claims in this report were therefore verified over the network (SNAPSHOT 2, below).

**The headline defect, re-derived.** `evaluate_model.py:7` is `import datasets1`; the tree ships
`datasets.py`, not `datasets1.py`:

```
$ find . -path ./.git -prune -o -name "*datasets*" -print
./__pycache__/datasets.cpython-313.pyc
./datasets.py
```

`datasets1` is used at four further call sites: `evaluate_model.py:43` (`datasets1.FashionIQ`),
`:47` (`datasets1.Shoes`), `:51` (`datasets1.CIRR`), `:55-56` (`datasets1.Fashion200k`). The
importer is therefore not fixing a stray alias: it is a whole module rename.

**Symbol comparison, as required by the assignment — this is a MISSING MODULE, not a name
mismatch.** `datasets.py` defines exactly four dataset classes and four helpers:

```
$ grep -n "^class \|^def " datasets.py
15:def save_obj(obj, path):
18:def load_obj(path):
21:def draw_text(img, point, text, drawType="custom"):
47:def draw_text_line(img, point, text_line: str, drawType="custom"):
63:class FashionIQ(torch.utils.data.Dataset):
207:class Shoes(torch.utils.data.Dataset):
330:class CIRR(torch.utils.data.Dataset):
470:class Fashion200k(torch.utils.data.Dataset):
```

Every symbol `evaluate_model.py` reaches for through the `datasets1.` namespace
(`FashionIQ`, `Shoes`, `CIRR`, `Fashion200k`) exists in `datasets.py`. Constructor signatures also
match call-for-call: `evaluate_model.py:43` passes
`(path=, transform=[...], split=)` and `datasets.py:64` is
`def __init__(self, path, transform=None, split='val-split')`; `:47` passes `(path=, transform=)`
against `datasets.py:208`; `:51` against `datasets.py:331`; `:55-56` pass
`(path=, split=, transform=)` against `datasets.py:473`. **No symbol the importer uses is absent
from `datasets.py`.** The defect is a wrong *module name* in one file, and the remedy is
correspondingly narrow (see §9). This matters because the audit pair had recorded it as a
`SOURCE_EVIDENCE_MISSING` asset to be *acquired*
(`workbench/registry/assets/cluster_b2_encoder_family.yaml:146-166`,
`acquisition_status: UNVERIFIED`, `official_download_url: null`) — there is nothing to acquire.

**SNAPSHOT 2 — current upstream.** `git -C workbench/third_party/ENCODER ls-remote origin`
→ only two refs, both equal to the pin:

```
29a2a31d6a56f677bf450c3be7cdaef423fb7018	HEAD
29a2a31d6a56f677bf450c3be7cdaef423fb7018	refs/heads/main
```

`git fetch origin --dry-run` returned rc=0 with no output; `git log --oneline origin/HEAD -5`
prints `29a2a31 Update README.md` only (shallow local clone, so this is the same commit, not a
moved HEAD). Independent confirmation that the pinned commit still exists upstream and that the
upstream tree contains the same eight Python files:

```
$ curl -s "https://api.github.com/repos/iLearn-Lab/AAAI25-ENCODER/git/trees/29a2a31d...?recursive=1" \
    | grep -o '"path": "[^"]*"' | grep -i "\.py\""
"path": "cirr_test_submission.py"
"path": "datasets.py"
"path": "evaluate_model.py"
"path": "model_try2.py"
"path": "test.py"
"path": "token_wise_matching.py"
"path": "train.py"
"path": "utils.py"
```

**No upstream fix.** Upstream is frozen at the pin, ships no `datasets1.py`, added no CLI, fixed no
dependency and moved no branch. The regression is neither repaired nor acknowledged.

**New finding: the author Drive folder the README links is live and identical to the checkout.**
`ENCODER/README.md:249` links `https://drive.google.com/drive/folders/1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3`.
Enumerating it (`embeddedfolderview`) yields exactly the same eleven names as the git tree plus a
`ckpt/` folder, and every code file downloaded from it is byte-identical (modulo trailing
whitespace) to the pinned file:

```
$ for p in a.py:cirr_test_submission.py b.py:datasets.py c.py:evaluate_model.py d.py:model_try2.py \
           f.py:test.py g.py:token_wise_matching.py h.py:train.py i.py:utils.py; do diff ...; done
a.py == cirr_test_submission.py IDENTICAL
b.py == datasets.py            IDENTICAL
c.py == evaluate_model.py      IDENTICAL
d.py == model_try2.py          IDENTICAL
f.py == test.py                IDENTICAL
g.py == token_wise_matching.py IDENTICAL
h.py == train.py               IDENTICAL
i.py == utils.py               IDENTICAL
```

The Drive `README.md` is an older revision (it still points at a different Drive folder id and at
`python3 evaluation_model.py`, i.e. a filename that does not exist anywhere — `README.md:252`
still carries the same typo). **The Drive distribution contains no `datasets1.py` either**, so the
missing module is confirmed to be an author oversight in a single filename, not a distribution
split. `ckpt/fashioniq/fashioniq.pt` *does* exist there; see §5.

## 3. NATIVE EVALUATION WORKFLOW

Entrypoint: `evaluate_model.py`. It is a module-level script — no `if __name__ == "__main__"`, no
functions around the work — with `argparse` at `evaluate_model.py:12-32`. Chain:

```
evaluate_model.py:40  open_clip.create_model_and_transforms('ViT-B-32', pretrained=os.path.join('.', 'open_clip_pytorch_model.bin'))
evaluate_model.py:43  datasets1.FashionIQ(path=args.fashioniq_path, transform=[preprocess_train, preprocess_val], split=args.fashioniq_split)
evaluate_model.py:60  model = torch.load(args.ckpt_path)
evaluate_model.py:62  Encoder_model = Encoder(hidden_dim=512, dropout=0.5, local_token_num=P, t=tau_, wc=wc, N_p=N_p, weighted=weighted)
evaluate_model.py:63  Encoder_model.load_state_dict(model, strict=False)
evaluate_model.py:64  Encoder_model = Encoder_model.cuda()
evaluate_model.py:71-75 for category in ['dress','shirt','toptee']: t = test.test(args, Encoder_model, dataset_list[0], category)
```

`test.test` (`test.py:6`) uses `testset.test_queries_{dress,shirt,toptee}` /
`testset.test_targets_{dress,shirt,toptee}` (`test.py:9-15`), extracts composed query features and
target features on GPU (`test.py:33`, `:49`, both `.cuda()`), L2-normalises (`test.py:57-60`),
masks the reference index (`test.py:68`), and prints `('{name}_r{k}', r)` for
`k in [1,10,50]` (`test.py:71-81`).

**CLI exists and the workbench adapter matches it.** `EncoderAdapter` (`workbench/backend/adapters/models.py:36-41`)
emits `python evaluate_model.py --dataset fashioniq --fashioniq_split val-split --fashioniq_path <root>/ --ckpt_path <ckpt>`.
All four flags are declared at `evaluate_model.py:14,16,17,31`. A wrapper must call exactly that
script; there is no other official entrypoint for FashionIQ (`test.py` has no `__main__`).

Missing components to run it as published: `datasets1.py` (wrong module name),
`./open_clip_pytorch_model.bin` resolved against the **process CWD** (`evaluate_model.py:39` sets
`clip_path = '.'`; only *argument* paths are absolutised by the adapter, `models.py:41`), and a
checkpoint. Also note `evaluate_model.py:64` calls `.cuda()` unconditionally with no
`torch.cuda.is_available()` guard, and `test.py:33,49` call `.cuda()` — CPU-only execution is
impossible, which is what `models.yaml:422-427` records and what the assignment's GPU gate exists
for.

## 4. DEPENDENCIES

Classification vocabulary is the taxonomy's; the D1–D8 category labels are summarised as
"dependency-type" where the definition is not restated in this repository.

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `datasets1` | `evaluate_model.py:7` (+43,47,51,55,56) | upstream | — (module does not exist in any published distribution) | n/a — nothing to install | n/a | see §9: not a dependency at all; a wrong module name. Do NOT provision a substitute module | NOT_APPLICABLE (static) |
| `open_clip` | `evaluate_model.py:6,40`; `model_try2.py:9,135,137` | upstream (PyPI-backed) | `open-clip-torch==2.24.0` | yes, PyPI | py3.9 / torch 1.13.1 / cu12.6 | pin in the model env; already encoded | STATIC: planner resolves it (below) |
| `torch` | `evaluate_model.py:1` | upstream | `1.13.1` | yes | cu12.6 declared | keep registry pin | STATIC: planner resolves it |
| `transformers` | `model_try2.py:8` (`CLIPTextModel`, `CLIPVisionModelWithProjection`, `CLIPImageProcessor`, `AutoTokenizer`, `CLIPTextModelWithProjection`) | upstream | `transformers==4.30.2` | yes | py3.9 | keep registry pin | STATIC: resolved from `ENCODER/requirements.txt` |
| `einops` | `model_try2.py:13` | upstream | `0.4.1` | yes | any | keep pin | STATIC: resolved |
| `cv2` | `datasets.py:14` (module-level, imported even though only `draw_text*` uses it) | upstream | not pinned by name in the planner | yes (opencv-python) | any | pin `opencv-python` in the env; unpinned today | STATIC: name present, version unconstrained — see P2 below |
| `torchvision` | env contract | upstream | `0.14.1` | yes | pairs with torch 1.13.1 | keep | STATIC: pairing valid |
| `PIL`, `numpy`, `tqdm` | `datasets.py:2,4`; `test.py:3` | upstream | `Pillow`, `numpy==1.21.6`, `tqdm==4.66.1` | yes | any | keep | STATIC: resolved |

The planner derives exactly this set, and it **catches the upstream typo as a bogus package**:

```
$ .venv/bin/python -c "... environment_planner.model_constraints(model_by_id('encoder')) ..."
{'python': '3.9', 'torch': '1.13.1', 'torchvision': '0.14.1', 'cuda': '12.6',
 'packages': {'einops': '0.4.1', 'numpy': '1.21.6', 'open-clip-torch': '2.24.0', 'tqdm': '4.66.1',
              'transformers': '4.30.2', 'datasets1': 'UNKNOWN'},
 'ranges': {}, 'pip_only': ['open-clip-torch', 'torch', 'torchvision'],
 'confidence': 'VERIFIED_REQUIREMENTS',
 'evidence': [... 'ENCODER/requirements.txt', 'registry environment declaration (VERIFIED_REQUIREMENTS)', ...]}
```

`'datasets1': 'UNKNOWN'` is the environment solver trying to install a package that cannot exist:
`_evaluator_imports` walks the AST from `evaluate_model.py` and adds any non-local, non-stdlib
top-level module name (`workbench/backend/environment_planner.py:274-336`), and the local-path
resolver `local_path` (`:265-271`) only accepts a real file. So `datasets1` is classified as a
third-party *distribution*. **This is a real, if secondary, defect: the environment contract
inherits the upstream typo and will never resolve.** (P1, see §9.)

`torchvision: '0.14.1'` is declared and pairs legally with torch 1.13.1
(`environment_planner._TORCHVISION_PAIRS`); nothing in the pinned code imports `torchvision`
however, so it is a harmless over-declaration.

## 5. CHECKPOINTS

| field | registry value | independently observed |
|---|---|---|
| file | `fashioniq.pt` | present in the author Drive at `1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3/ckpt/fashioniq/fashioniq.pt` (file id `10dsBr4Numm9cRazItuq6g_t9M33yOX_H`) |
| `download_url` | `null` (`models.yaml:459`) | the Drive file is servable and *is* a direct download once confirmed — see below |
| `source_url` | Drive **folder** link (`models.yaml:460`) | folder is live, `HTTP 200` |
| `status` | `URL_UNRESOLVED` | obtainability is now established; identity is still not |
| `expected_size_bytes` | `null` (`models.yaml:467`) | **631,879,365 bytes** |
| `expected_sha256` | `null` (`models.yaml:468`) | **`0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173`** |
| `checkpoint_mapping_status` | `UNVERIFIED` (`models.yaml:476`) | still UNVERIFIED — no author statement binds the blob to the paper's FashionIQ row |

Availability class: **obtainable via a two-step Drive download; not directly linkable.** The bare
`/uc?export=download&id=…` returns the 2,460-byte Google confirmation HTML; the byte-serving
endpoint returns the file with a filename and an exact length:

```
$ curl -s -L -G --data-urlencode "id=10dsBr4Numm9cRazItuq6g_t9M33yOX_H" \
       --data-urlencode "export=download" --data-urlencode "confirm=t" -r 0-0 \
       -D h.txt "https://drive.usercontent.google.com/download"
HTTP/2 206
content-type: application/octet-stream
content-disposition: attachment; filename="fashioniq.pt"
content-range: bytes 0-0/631879365
```

Full fetch then confirmed the digest, so both values in the table are measured, not inferred:

```
$ sha256sum /tmp/drive/fashioniq.pt
0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173  fashioniq.pt
```

Loader contract, measured rather than assumed. The registry says `checkpoint_serialization:
state_dict` (`models.yaml:430`) and the code says `torch.load(args.ckpt_path)` then
`load_state_dict(model, strict=False)` (`evaluate_model.py:60,63`). Loading the blob:

```
$ python -c "torch.load('/tmp/drive/fashioniq.pt', map_location='cpu', weights_only=True)"
weights_only=True OK -> pure state_dict
type <class 'collections.OrderedDict'>; n keys 417
top-level prefixes: [('backbone', 409), ('affine', 6), ('loss_T', 1), ('local_weight', 1)]
backbone.* children: backbone.clip 302, backbone.binding_decoder 22, backbone.fac_net 7,
  backbone.weight_chanelImg 6, ... backbone.multihead_attn 4, backbone.fc 2, backbone.text_fc 2,
  backbone.cross_t2i 2, backbone.cross_i2t 2
```

Consequences: (1) it is a pure tensor/`OrderedDict` file, so the workbench's trusted-bytes
`weights_only=False` path (`workbench/replay/common.py:52-54`) is not needed — `weights_only=True`
succeeds and is the safer loader; that is an argument for `UNSAFE_CHECKPOINT_SERIALIZATION` being
*falsified* here rather than inherited. (2) It embeds the **whole CLIP backbone**
(`backbone.clip.*`, 302 keys: `visual.conv1`, `visual.transformer.resblocks.*`, `text_projection`,
`positional_embedding`, `logit_scale`). That is the fact §7/§9 turns on: the separate
`./open_clip_pytorch_model.bin` requirement is a *local-directory* requirement of a
`pretrained=` kwarg, and because the checkpoint carries `backbone.clip.*` keys, the model's own
backbone weights are overwritten by the state_dict. This does not make the missing file optional —
`create_model_and_transforms` must still succeed to *construct* the module (and supplies the
tokenizer and the 768-d `fc` input) — but it means the *runtime* numerics of the backbone come from
the checkpoint, not from the separately-loaded binary. That distinction was not drawn in
`final_model_audit/encoder.md:211-213`, which treats the binary as the operative backbone.

`strict_clean_checkpoint_available: false` (`models.yaml:454`) is consistent with
`download_url: null` and with `checkpoint_selection` semantics; it does not reflect the newly
measured obtainability.

## 6. AUXILIARY ASSETS

| name | purpose | source | availability class | generation possible? |
|---|---|---|---|---|
| `datasets1.py` (declared at `cluster_b2_encoder_family.yaml:146-166`) | wrongly declared "dataset module" | — | **not an asset**; the module was never published by anyone, and its symbols all exist in `datasets.py` | n/a |
| `open_clip_pytorch_model.bin` | `pretrained=` arg of `open_clip.create_model_and_transforms('ViT-B-32', …)` at `evaluate_model.py:40` / `model_try2.py:135` | public OpenCLIP release, `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` | PUBLIC_ASSET_NOT_PREPARED (see §7/§9) | no generation; a public download |
| `resized_image/<cat>/<id>.jpg` | gallery + query pixels; `datasets.py:68` sets `image_dir = path + 'resized_image'`, read at `:148` | operator-prepared from official FashionIQ | present locally: `/home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/dress` holds 3817 `*.jpg` | yes, from official images |
| `captions/correction_dict_{dress,shirt,toptee}.json` | modification-text corrector; `datasets.py:163-164` and `:105-106` | not bundled upstream, no author URL | present locally (all three files exist under `data/fashionIQ_dataset/captions/`) — provenance of the local copies is not established by this audit | derivable from the upstream typo list, so generation is possible |
| `test_queries_*.pkl`, `test_targets_*.pkl`, `fashion_iq_data.json` | caches written on first run (`datasets.py:75-92`) and reloaded thereafter (`:94-101`) | generated | generated artifact | yes |
| `fashioniq.pt` | see §5 | author Drive | obtainable | no |

**Dependency-category note (as required).** The `open_clip_pytorch_model.bin` requirement is not a
missing *distribution*: `open-clip-torch==2.24.0` is installable and pinned. It is a missing
*weight artifact* referenced by a hardcoded relative path — i.e. the correct classification is
`PUBLIC_ASSET_NOT_PREPARED`, not `INSTALLABLE_DEPENDENCY_MISSING`. Until this audit no code in the
workbench knew of this asset at all: `auxiliary_assets_for_model('encoder')` returns `[]` and
`auxiliary_model_ids()` does not contain `encoder` (verified by executing the registry helpers), so
`workbench/scripts/download_auxiliary_assets.py --model encoder` is not even a valid invocation
(its `--model` choices come from `auxiliary_model_ids()`, `download_auxiliary_assets.py:303`). The
only machinery that knows about the file is a hardcoded preflight check.

## 7. FASHIONIQ PROTOCOL

`native_protocol: fashioniq_val_split` (`models.yaml:444`), i.e. **Protocol C**: ordered
pair-union gallery, reference excluded. `supported_protocols: [fashioniq_val_split]`
(`models.yaml:446`). This is the registry's authoritative statement and nothing in this audit
proposes changing it.

Verified in code, at `datasets.py:158-206` (`get_test_data`): the split file read is
`split.{name}.val.json` (`:159`) and the captions `cap.{name}.val.json` (`:161`) — the *val*
annotation set, not train. Gallery construction (`datasets.py:185-196`) under
`split == 'val-split'` walks the queries and appends `source_img_id` then `target_img_id` to
`test_targets_id` only if not already present — a deduplicated, query-ordered union of the
reference and target ids. The `elif self.split == 'original-split'` branch (`:197`) instead loads
every image in `split.val.json` (full gallery). The adapter passes `--fashioniq_split val-split`
(`adapters/models.py:41`), so the full-gallery branch is unreachable from the workbench — the
registered protocol is honoured by construction, not by luck.

Reference policy: excluded by masking, `test.py:66-68`
(`sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`). Categories: `dress`, `shirt`,
`toptee`, driven by the loop at `evaluate_model.py:73`. Recall: `test.py:73-81`, R@1/R@10/R@50 with
seed-1 `nn_result` (`test.py:71`); the reported row uses R@10/R@50 only.

Preprocessing: `preprocess_train, preprocess_val` come from
`open_clip.create_model_and_transforms('ViT-B-32', ...)` (`evaluate_model.py:40`), so the transform
is OpenCLIP's for that architecture — reproducible **only if the exact LAION checkpoint is used**,
which is the point of the missing-asset defect. Query image uses `transform[0]`
(`datasets.py:150`, `stage=0`); gallery uses `transform[1]` (`datasets.py:192`, `stage=1`). Note
this means the *train* transform is applied to reference/target source images and the *val*
transform only to gallery images — an upstream choice that a wrapper must preserve verbatim.

## 8. WORKBENCH INTEGRATION

* adapter: `EncoderAdapter` — `workbench/backend/adapters/models.py:36-41`; registered in
  `ADAPTERS` at `:161`.
* planner/command: correct flags, matches upstream `argparse` (`§3`).
* checkpoint prep: `encoder` has **no** auxiliary-asset record and **no** `download_url`, so
  neither `download_checkpoints.py` nor `download_auxiliary_assets.py` can produce its
  prerequisites; the Drive URL is not in the registry and the downloader fetches only
  `checkpoint["download_url"]` (`workbench/scripts/download_checkpoints.py:226,243`).
* preflight: two model-specific hardcoded checks exist and both fire on a clean checkout —
  `workbench/backend/preflight.py:211-215`:
  ```
  if model_id == "encoder" and source is not None:
      if not (source / "datasets1.py").is_file():
          blockers.append(f"ENCODER evaluator import missing: {source / 'datasets1.py'}")
      if not (source / "open_clip_pytorch_model.bin").is_file():
          blockers.append(f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}")
  ```
  The first is a *string* check for a file that must never exist; the second is the only
  provision-aware-ish signal, and it is not backed by an asset record, a URL or a digest.
* preflight paths: `preflight.py:101-102` adds both files to the inspected path list.
* metric parser: **absent.** `extract_aggregate_metrics('encoder', …)` returns
  `AGGREGATE_PARSER_UNAVAILABLE` (executed; `workbench/backend/metrics_extraction.py:398-413`),
  while `test.py:81` prints `('dress_r10', 51.5…)`-style tuples — a machine-readable shape the
  parser could consume. Per-query support: the source has none (no per-query ids emitted), so
  per-query parity is not merely unimplemented, it is unsupported by the official script.

Observed blocker set on this host (executed against the live registry, `runtime_blockers`):

```
encoder fashioniq [
 'ENCODER evaluator import missing: .../ENCODER/datasets1.py',
 'ENCODER asset missing: .../ENCODER/open_clip_pytorch_model.bin',
 'checkpoint missing: .../artifacts/checkpoints/encoder/fashioniq.pt',
 'checkpoint mapping unresolved']
```

Environment tier: `verify_environment(model, probe=True)` → `UNCONFIGURED / UNCONFIGURED`,
`WORKBENCH_ENCODER_PYTHON not set`. This is configuration, not a defect, and is correctly out of
scope for scoring.

## 9. CONFIRMED DEFECTS

**C1 — P0 — `evaluate_model.py` imports a module that has never existed in any published form.**
Root cause: upstream typo. `evaluate_model.py:7` says `datasets1`; the file is `datasets.py`
(`git ls-files`; GitHub tree at the pin; the author Drive folder). Reproduction (static): any
`import evaluate_model` or `python evaluate_model.py …` fails at line 7 with
`ModuleNotFoundError: No module named 'datasets1'` before any other statement executes. Fix (for a
maintainer to apply deliberately, following the CSMCIR precedent of
`workbench/patches/csmcir.md`): change `evaluate_model.py:7` to `import datasets as datasets1` —
a one-token alias that leaves every call site (`:43,47,51,55,56`) untouched and does not invent a
module. Regression check after the alias: `test.py` returns `('dress_r10', r)` tuples matching the
paper row (R@10 56.13 / R@50 77.59, `papers.yaml:1155-1166`) — NOT_RUN here (needs GPU). Note that
this is NOT the same fix as provisioning a `datasets1.py`: a substitute module would be a second
copy of the dataset code and a divergence risk, and the defect is recoverable by alias, which is
materially cheaper and safer.

**C2 — P0 — the backbone weights binary is required by a CWD-relative path and is neither
declared nor provisioned.** `evaluate_model.py:39` `clip_path = '.'` and `:40`
`pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin')`; identically
`model_try2.py:134-135`. The asset audit
(`cluster_b2_encoder_family.yaml:168-188`) does hold a public URL
(`https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin`)
and calls it `AVAILABLE`, but (a) nothing consumes that record — `auxiliary_assets_for_model('encoder')`
is `[]` (executed) — and (b) the record's `official_version_or_revision: laion2B-s34B-b79K` is not
what the *code* names: the code names the bare file `open_clip_pytorch_model.bin` and never names a
LAION release; only the commented-out line `model_try2.py:134` hints at
`/root/CLIP-ViT-B-32-laion2B-s34B-b79K`. Fix: register the asset in
`workbench/registry/auxiliary_assets.yaml` so the existing, tested downloader stages it at
`WORKBENCH_THIRD_PARTY_ROOT/ENCODER/open_clip_pytorch_model.bin`. Separately, ensure the child
process CWD is the checkout: this is not currently guaranteed by the adapter (which absolutises
only its argument paths, `workbench/backend/adapters/models.py:41`) but IS guaranteed by the
installer, because `workbench/scripts/evaluate_models.py:214-215` sets
`source_cwd(model, source) = source` for every model except `csmcir`, and
`workbench/backend/jobs.py:573` launches with `cwd=job.cwd`. So the asset only has to land in the
checkout; the launch directory is already correct. `workbench/envs/README.md` records only the
Python/requirements contract per model and contains no `cd` instruction (re-checked; an earlier
draft of this report cited one that does not exist). This is PUBLIC_ASSET_NOT_PREPARED, not a
missing dependency.

**C3 — P1 — the environment contract inherits the typo as a package named `datasets1`.**
`environment_planner.model_constraints` returns `{'datasets1': 'UNKNOWN'}` because
`_evaluator_imports` treats any unresolved top-level module as third-party
(`environment_planner.py:274-336`). Whatever fix is chosen for C1 must be mirrored here or the env
plan carries an unsatisfiable spec. Severity P1: the environment is not required to launch today
(the run is blocked upstream of it) but the plan is wrong now and would surface as a confusing
resolution failure.

**C4 — P2 — `opencv-python` (import `cv2`, `datasets.py:14`) is named but never version-pinned**,
unlike every other import; `datasets.py` uses it only in `draw_text`/`draw_text_line`, which the
FashionIQ path never calls. Pin it or drop the import — the FashionIQ evaluation does not need it.

**C5 — P2 — CPU-only hosts cannot run this evaluator and the plan does not say so at the adapter
level.** `evaluate_model.py:64` and `test.py:33,49` call `.cuda()` unconditionally. `models.yaml:422-427`
records the CUDA pin and explains that the earlier `UNKNOWN` let the GPU gate never fire; the
adapter itself carries no such statement. This is correctly classified `GPU_RUNTIME_UNTESTED`, not
a defect to fix by patching the source (a `map_location`/device guard would be a workbench-side
adaptation, which the project's no-silent-alteration rule forbids for official reproduction).

**C6 — P2 — the paper's README command is wrong and diverges from the pinned checkout.**
`README.md:252` says `python3 evaluation_model.py --model_dir checkpoints/ENCODER_*.pth`, but the
shipped script is `evaluate_model.py` and it reads `--ckpt_path`, not `--model_dir`
(`evaluate_model.py:31`); the `--model_dir` flag is a *training* flag. A user following the README
gets `unrecognized arguments`. The workbench adapter is right and the README is wrong; no action
beyond documenting it.

**Falsified hypotheses.** (i) `final_model_audit/encoder.md:211-213` calls the missing LAION binary
"[the] required binary" for the backbone without noting that the released `fashioniq.pt`
*contains* 302 `backbone.clip.*` keys, so it overwrites the backbone it constructs — the file is
required for construction, not for the final numerical weights. (ii) The same-family audit record
declared `encoder_missing_module_datasets1` as an asset with `expected_relative_path: datasets1.py`
and an `output_validation_contract` demanding such a file exist
(`cluster_b2_encoder_family.yaml:152-163`); no such artifact has ever been published by the authors
or anyone else — measured this round over git, the GitHub API and the author Drive — so the
declaration is wrong in kind. (iii) `models.yaml:455` says "direct file URL and SHA remain
unverified"; the SHA is now verified (§5) and the direct URL is a stable two-step Drive endpoint.

**Not a defect here (explicitly checked).** `load_state_dict(model, strict=False)`
(`evaluate_model.py:63`) does not hide anything consequential for the *fashioniq.pt* artifact when
the exact defaults the adapter passes are used. Executed probe (checkpoint vs. model built with
`hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1, wc=2, N_p=2, weighted=True`, i.e. the
`--P/--wc/--N_p` defaults the README comment `evaluate_model.py:33` prescribes for FashionIQ):

```
n checkpoint keys 417 ; n model keys 115
MISSING    2 : ['backbone.clip.p', 'backbone.binding_decoder.learnable_relation']
UNEXPECTED 304 : ['backbone.clip.positional_embedding', ...]  # my open_clip stub has no real CLIP
```

Both MISSING keys are artifacts of the probe's stub (`clip.p`) and of `BindingDecoder`'s
`.cuda()`-constructed parameter (`model_try2.py:93`), not absent published weights. **The loose
load is nevertheless a real hazard** and is reported under §10: if a *wrong* variant were ever
mapped to `encoder`, `strict=False` would silently leave whole module groups at their random
initialisation instead of raising, and nothing in the pipeline compares the checkpoint's key set to
the model's. That is why the mapping must be established by evidence (an author statement or a
digest), not by "the load did not raise".

## 10. REMAINING BLOCKERS

| # | classification | ownership | evidence | remediation |
|---|---|---|---|---|
| B1 | `FIRST_PARTY_BUG` — upstream entrypoint cannot import | upstream (patch/alias applied deliberately by the workbench) | `evaluate_model.py:7`; `datasets.py:63,207,330,470`; §9-C1 | add the one-line alias as an `approved_source_patches` entry (CSMCIR pattern, `models.yaml:73-80` + `workbench/patches/csmcir.md`) with `base_sha256` of `evaluate_model.py` and both digests recorded |
| B2 | `PUBLIC_ASSET_NOT_PREPARED` — missing LAION ViT-B-32 weights | workbench | `evaluate_model.py:39-40`; `model_try2.py:134-135`; `auxiliary_assets_for_model('encoder') == []` | register in `workbench/registry/auxiliary_assets.yaml`; the existing downloader already stages model-source assets |
| B3 | `FIRST_PARTY_BUG` (upstream, workbench-side mitigation) — the asset path is CWD-relative | upstream | `evaluate_model.py:39`; `workbench/backend/adapters/models.py:41` (only the four argument paths are absolutised) | none needed beyond B2: `workbench/scripts/evaluate_models.py:214-215` already runs the evaluator with `cwd = source`, so a properly staged checkout-local asset is found. Do not rewrite `clip_path` in the source |
| B4 | `CHECKPOINT_URL_INACCESSIBLE` **falsified**; the live classification is `CHECKPOINT_UNPUBLISHED`→no, `PUBLIC_CHECKPOINT_NOT_MAPPED` | workbench (mapping) + upstream (identity) | §5 measurements | register a Drive-capable download (folder-member id + measured size + measured sha256) **and** keep `checkpoint_mapping_status: UNVERIFIED` until an author statement binds the blob to the paper's FashionIQ row; obtainability ≠ identity |
| B5 | `SCIENTIFIC_PROTOCOL_UNVERIFIED` for the *mapping*, not the protocol | upstream | protocol itself is verified in code (§7); the file's benchmark identity is not | do not promote the mapping on size/hash alone |
| B6 | `GPU_RUNTIME_UNTESTED` | host | `evaluate_model.py:64`; `test.py:33,49` | GPU host required; unchanged |
| B7 | `FIRST_PARTY_ADAPTER_MISSING` — no metric parser | workbench | `metrics_extraction.py:398-413`; output shape `test.py:81` | add a `parse_encoder_output` for `('{cat}_r{k}', value)` tuples, macro over categories |

Deliberately *not* proposed: any change to the gallery, the reference policy, the caption input or
the noise condition. Protocol C (`fashioniq_val_split`) is authoritative
(`models.yaml:444`) and is what the code implements (§7).

## 11. IMPLEMENTATION PLAN

1. **Alias patch** (P0, ~1 line + registry metadata). Add
   `workbench/patches/encoder_datasets_import.patch` changing `evaluate_model.py:7` to
   `import datasets as datasets1`; declare it under `approved_source_patches` with
   `base_sha256`/`derived_sha256`/`reason`. Preflight's
   `f"ENCODER evaluator import missing: …/datasets1.py"` check (`preflight.py:212-213`) must then be
   replaced by the shared `approved_patch_state` gate (`preflight.py:196-203`) rather than deleted
   silently; today that check asserts the *absence* of the fix, so applying the patch would leave a
   permanent blocker string.
2. **Backbone-weights asset** (P0). Append a record to
   `workbench/registry/auxiliary_assets.yaml` (`model_id: encoder`, `destination_scope: model_source`,
   `destination_path: open_clip_pytorch_model.bin`,
   `source_type: direct_url`,
   `source_repo: https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin`).
   Capture the digest at download time; the Hub serves `x-linked-size: 605219813` and
   `x-linked-etag: "1bd3c7172de5b207ceac554f5ab5266166f3b9baccc9af5989bc801016d080ad"`
   (observed this round), which can be promoted to `expected_sha256` after one fetch-and-verify.
   Then the hardcoded preflight check at `preflight.py:214-215` can be replaced by
   `external_asset_statuses`, which already emits the same message from a declared record.
3. **Checkpoint record** (P1). Add a `downloader`-supported Drive path or an operator-staging
   contract for `fashioniq.pt`, recording `expected_size_bytes: 631879365` and
   `expected_sha256: 0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173`, with
   `download_url` left `null` if the two-step confirm cannot be automated (do not fabricate a
   one-shot URL) — the downloader must be able to verify bytes it did not itself fetch.
4. **Metric parser** (P2). `parse_encoder_output` for the `('{cat}_r{k}', r)` tuples.
5. **Env pin** (P2). Resolve `datasets1` out of the constraint set once the alias lands; pin
   `opencv-python`.

Complexity: small (1-line source patch + 2 registry records + 1 parser). Scientific risk: low —
none of it touches protocol, gallery, captions or noise. Expected outputs: preflight clean except
the checkpoint-mapping blocker; then a GPU run producing `dress/shirt/toptee` R@10/R@50 comparable
to 56.13/77.59.

## 12. FINAL VERDICT

* `BLOCKED` — `FIRST_PARTY_BUG` (unimportable official evaluator, §9-C1) **+**
  `PUBLIC_ASSET_NOT_PREPARED` (backbone weights) **+** `PUBLIC_CHECKPOINT_NOT_MAPPED` (identity of
  the Drive blob), with `GPU_RUNTIME_UNTESTED` on top.
* `NOT_RUN` — no evaluation was executed; this host has no GPU (`evaluate_model.py:64` is
  unconditional) and neither the module alias nor the backbone weights are in place.
* `VERIFIED` (sub-claims, STATIC): the pin matches the registry; `datasets.py` defines every symbol
  `datasets1.*` is used for; the adapter's CLI matches upstream `argparse`; the protocol is
  Protocol C and the full-gallery branch is unreachable; the checkpoint is a pure state_dict of
  631,879,365 bytes with sha256 `0f31c501…9173` and 417 keys including the CLIP backbone;
  the official Drive distribution is byte-identical to the checkout and contains no `datasets1.py`.
* `UNKNOWN`: whether the Drive `fashioniq.pt` is the artifact behind the paper's FashionIQ row
  (obtainability is now established, identity is not), and any Runtime claim whatsoever.

## 13. EVIDENCE

* repo: `https://github.com/iLearn-Lab/AAAI25-ENCODER` @ `29a2a31d6a56f677bf450c3be7cdaef423fb7018`
  (2026-05-25), verified by `git rev-parse HEAD` and by the GitHub trees API.
* key files: `evaluate_model.py:7,39-40,43,60,62-64,71-75`; `datasets.py:63-206` (esp. `:68,148,159-164,185-206`);
  `test.py:6,33,49,66-81`; `model_try2.py:9,13,93,134-137`; `README.md:249-253` (`:252` wrong script name);
  `workbench/registry/models.yaml:415-476`; `workbench/registry/paper_benchmarks/papers.yaml:1104-1140` (row `encoder_fiq_clean_val_split`; stated values and their verification_method quoted at `papers.yaml:1155-1166`);
  `workbench/backend/adapters/models.py:36-41`; `workbench/backend/preflight.py:101-102,211-215`;
  `workbench/backend/metrics_extraction.py:398-413`;
  `workbench/backend/environment_planner.py:239-336,368-380`;
  `workbench/registry/assets/cluster_b2_encoder_family.yaml:146-188`.
* commands (all read-only; scratch output under `/tmp`):
  `git -C .../ENCODER rev-parse HEAD`; `git ls-files`; `find . -name "*datasets*"`;
  `grep -n "^class \|^def " datasets.py`; `git ls-remote origin`; `git fetch origin --dry-run`;
  GitHub trees API for `29a2a31d…`; `curl` folder enumeration of
  `1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3` and its `ckpt/fashioniq` child;
  ranged `curl` to `drive.usercontent.google.com/download` → `content-range: bytes 0-0/631879365`;
  `sha256sum fashioniq.pt` → `0f31c501…9173`;
  `torch.load(..., weights_only=True)` key census;
  the scratch `strict=False` probe (`/tmp/shim/probe_encoder.py`);
  `.venv/bin/python` calls into `workbench.backend.{registry,preflight,runtime,environment_planner,metrics_extraction}`.
* observed results: recorded inline in §2, §5, §8, §9.

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 3 of 6.** Level 1 = source/pin integrity (achieved). Level 2 = defect reproduced by static
inspection only, over the network *and* the local checkout (achieved). Level 3 = the checkpoint
artifact was actually fetched and its bytes/key set verified, and a construction/load probe was
executed on this host (achieved — the Drive download, `weights_only=True` load and key census are
observed, the `strict=False` delta is observed). Level 4 (a real evaluation run producing
Recall@10/50) is NOT reached: it needs the alias patch, the ~605 MB LAION asset, CUDA, and the
model env — none of which were executed. Levels 5–6 are out of reach. STATIC_AUDIT for everything
except the §5 checkpoint measurements and the §9 probe, which are RUNTIME_VERIFIED on this host.
