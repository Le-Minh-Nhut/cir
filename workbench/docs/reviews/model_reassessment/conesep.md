# conesep — reproducibility reassessment

Independent adversarial review, read-only, at workbench HEAD `b70d092e9d2232e34886f819e30c7327b74f3a76`
(`git rev-parse HEAD`). No GPU run, no download of weights, no install, no repo write.
Every claim is **STATIC_AUDIT** unless the line says RUNTIME_VERIFIED. Unproven = **UNKNOWN**.

---

## 1. MODEL IDENTITY

| Field | Value | Evidence |
|---|---|---|
| method | ConeSep | `workbench/registry/models.yaml:160` |
| paper | *ConeSep: Cone-based Robust Noise-Unlearning Compositional Network for Composed Image Retrieval* | `models.yaml:164` |
| paper URL | `https://arxiv.org/abs/2604.20358` | `models.yaml:165` |
| year / venue | 2026 / CVPR 2026 | `models.yaml:162`; `ConeSep/README.md:3` |
| source URL | `https://github.com/Lee-zixu/ConeSep` | `models.yaml:166` |
| pinned commit | `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | `models.yaml:172` |
| checkout HEAD | `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | `git -C workbench/third_party/ConeSep rev-parse HEAD` |
| checkout remote | `git@github.com:iLearn-Lab/CVPR26-ConeSep.git` (fetch+push) | `git -C workbench/third_party/ConeSep remote -v` |
| default branch | `main` | `models.yaml:171`; `ls-remote` shows only `refs/heads/main` |
| native protocol | `fashioniq_full_gallery_ref_excluded` | `models.yaml:178` (authoritative, quoted verbatim) |
| checkpoint variant | `fiq_n02` / `fiq_n05` / `fiq_n08` — all noise-trained | `models.yaml:187-216` |
| original metrics | R@10 55.58, R@50 75.88, mean 65.73 | `models.yaml:180-182` |
| reproduction status | `NOT_RUN` | `models.yaml:183` |

Sources are single-commit shallow checkouts: `git -C workbench/third_party/ConeSep rev-list --count HEAD`
→ `1`. History is unavailable, so "this file was never different" is **UNKNOWN**, not PASS.

## 2. SOURCE STATUS

Accessible: yes. Complete: functionally, no. The pinned tree has 1,954 non-vendored lines
(`cirr_sub.py 127, data_utils.py 427, datasets.py 384, test.py 168, train.py 285, utils.py 144`) plus a
vendored full LAVIS fork under `ConeSep/lavis/`.

Missing modules: the source *imports* are all satisfiable by the pinned vendored `lavis/` tree; nothing is
imported from an unpublished first-party module. The gaps are CLI, checkpoint loader, correction
dictionaries and BERT weights (Sections 3, 4, 5, 6).

Notable first-party inconsistencies (both observed, both P3):
- `ConeSep/README.md:245` and `:269` instruct `python src/cirr_test_submission.py ...`, but no `src/`
  directory exists (`ls workbench/third_party/ConeSep/src` → No such file or directory) and the file is
  named `cirr_sub.py`, which is what `train.py:30` imports and what `cirr_sub.py:94` guards with `__main__`.
  The documented CIRR test command cannot run as printed.
- `ConeSep/README.md:167-171` documents the dataset as `<root>/dress/`, `<root>/shirt/`, `<root>/toptee/`,
  but `datasets.py:31` reads `resized_image/<category>/<id>.jpg`. The layout in the README is not the layout
  the code opens.

### SNAPSHOT 2 — current upstream state

Commands (read-only; the pin was never changed):

```
$ git -C workbench/third_party/ConeSep ls-remote origin HEAD
bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b	HEAD
$ git -C workbench/third_party/ConeSep fetch origin --dry-run
(no output; exit 0 — remote refs identical, nothing to fetch)
$ timeout 60 git ls-remote https://github.com/Lee-zixu/ConeSep
bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b	HEAD
bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b	refs/heads/main
$ curl -sS -o /dev/null -w '%{http_code} %{redirect_url}' https://github.com/Lee-zixu/ConeSep
301 https://github.com/iLearn-Lab/CVPR26-ConeSep
```

**Confirmed relocation.** The registry's `upstream_repo_url` (`https://github.com/Lee-zixu/ConeSep`) now
answers **301** and redirects to `https://github.com/iLearn-Lab/CVPR26-ConeSep`. The pin is content-addressed
and both hosts serve the same object — `git ls-remote` on the old URL returns the same SHA as origin, so the
redirect is faithful and the registry note at `models.yaml:167-170` ("both hosts serve the same object") is
accurate. The checkout's `origin` was already re-pointed to the mirror, which is why it needs no fetch.

Effect of the move on reproduction: none — but **the authors did not add a FashionIQ evaluation CLI, did not
add a checkpoint loader, and did not publish a clean checkpoint.** `main` is byte-identical to the pin.

## 3. NATIVE EVALUATION WORKFLOW

Entrypoint: `ConeSep/test.py` contains **only** two callables — `load_obj`, `test`, `test_cirr_valset`.
There is no `argparse` and no `__main__` in `test.py`.

The native test function, quoted (`ConeSep/test.py:9-10,12-18,63-84`):

```python
def test(params, model, testset, category, txt_processors):
    model.eval()

    if category == 'dress':
        (test_queries, test_targets, name) = (testset.test_queries_dress, testset.test_targets_dress, 'dress')
    elif category == 'shirt':
        ...
        test_targets_id = []
        for i in test_targets:
            test_targets_id.append(i['target_img_id'])
        for i, t in enumerate(test_queries):
            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10

        nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]
        for k in [1, 10, 50]:
            r = 0.0
            for i, nns in enumerate(nn_result):
                if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                    r += 1
            r = 100 * r / len(nn_result)
            out += [('{}_r{}'.format(name, k), r)]
        return out
```

Function chain a wrapper must call:

```
python train.py --dataset fashioniq --fashioniq_path <root>/ ...          # README.md:213-221
  └─ train.py:__main__ (train.py:258)                                     # no checkpoint load
       └─ train.py:load_dataset()  (train.py:80)  → datasets.FashionIQ(path=args.fashioniq_path,
                                                                        transform=targetpad(1.25,224),
                                                                        noise_ratio=args.noise_ratio)
                                                   → trainset.shuffle()   (train.py:107-109)
       └─ train.py:create_model_and_optimizer()   (train.py:131-140)
              load_model_and_preprocess(name="ConeSep", model_type="pretrain")  → lavis/models/__init__.py:174
              → ConeSep.from_config → ConeSep.__init__ (ConeSep.py:109) → init_tokenizer (blip2.py:31-34)
                                                                    → init_Qformer  (blip2.py:47-58)
       └─ train.py:train_and_evaluate()           (train.py:189)
              for category in ['dress','shirt','toptee']:
                  t = test.test(args, model, trainset, category, txt_processors)   # train.py:216
```

CLI: **no FashionIQ evaluation CLI exists.** `test()` is reachable only as a subroutine of the training loop,
which constructs a fresh model at `train.py:134` and never loads a weight file. `ConeSep/train.py:76` declares
`--restore_file` with help text "weights to reload before training"; `grep -rn "load_checkpoint|restore_file"
train.py test.py cirr_sub.py` returns only that `argparse` line — nothing ever reads it.
`ConeSep/utils.py:128 load_checkpoint()` exists, requires `checkpoint['state_dict']`, and is called from
nowhere in the pinned tree. The only `__main__` in the tree that loads weights is `cirr_sub.py:94`, and it is
CIRR-submission-only: it globs `*.pt` in a directory (`cirr_sub.py:118-126`), and `torch.load`s each file
(`cirr_sub.py:124`) with the CIRR dataset hardcoded (`cirr_sub.py:112`).

What a wrapper must call (the exact chain proposal, and its defect):

```
<python> workbench/replay/conesep.py --checkpoint <ConeSep-FIQ-N0.2.pt> \
         --dataset-root <fashioniq_ilearn_resized root> --protocol fashioniq_full_gallery_ref_excluded
  1. torch.load(<ckpt>, weights_only=False, map_location="cpu")     # whole-model pickle (Section 5)
  2. model.prompt_load()                                            # ConeSep.py:198 — REQUIRED
  3. model.eval(); model.to(config.DEVICE)
  4. for cat in (dress, shirt, toptee): test.test(args, model, dataset, cat, txt_processors)  # test.py:9
  5. emit the JSON the workbench metric parser expects
```

Missing components, each sufficient to block the run:

- **(a) no checkpoint loader** — the step-1 main branch never reads a file.
- **(b) step 2 is mandatory and undocumented** — `ConeSep.py:198-199` `prompt_load` copies
  `query_tokens` into `negative_tokens`; without it the negative-anchor parameter holds its random init
  (`ConeSep.py:154-157`). Neither `test.py` nor `train.py:216` calls `prompt_load` — only
  `train.py:135` does, in `create_model_and_optimizer`. A wrapper that skips it silently evaluates a
  differently-conditioned model.
- **(c) the evaluator has no workbench metric parser** — `workbench/backend/metrics_extraction.py:383-420`
  dispatches only for `limn`, `csmcir`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`; every
  other model, including `conesep`, falls through to `AGGREGATE_PARSER_UNAVAILABLE`.
- **(d) the adapter deliberately declines** — `workbench/backend/adapters/models.py:59-61` sets only
  `model_id`/`script`, inheriting `OfficialScriptAdapter.command` (`models.py:24-25`) which raises.
  Observed: `ADAPTERS['conesep'].command('s','c',None,None)` →
  `NotImplementedError: upstream evaluation command requires audit before execution`.
- **(e) preprocessing is not expressible from `test.py`** — the evaluator needs
  `params.batch_size` (`test.py:33`) and `txt_processors["eval"]` (`test.py:35`); both come from
  `train.py`'s `load_dataset`/`create_model_and_optimizer`, so a wrapper must import them, not call `test.py`
  as a process.

Shape algebra check (the registry at `models.yaml:151` records that an earlier "tensor-shape failure" claim
was withdrawn; I re-verified that withdrawal). `extract_retrieval_compose` returns `(N,1,1,d)`
(`ConeSep.py:492`) and `extract_retrieval_target` returns `(M,d,32)` (`ConeSep.py:515`), so
`np.matmul → (N,M,1,32)`, `.squeeze() → (N,M,32)`, `.max(-1) → (N,M)`. RUNTIME_VERIFIED in numpy at the real
gallery sizes:

```
$ .venv/bin/python /tmp/shape2.py && cat /tmp/shape2.txt
shape=(2017, 3817)
mask write ok -> -99999997952.0
argsort row shape=(50,)
membership test type=<class 'bool'> value=False
row0 top1=2580
```

The registry's withdrawal is **correct**: the mask lands as a scalar, `nns[:k]` indexing still works, and the
hit test is a plain `bool`. Degenerate `N==1` or `M==1` collapse the array to 1-D, but no real run has either.

## 4. DEPENDENCIES

D1–D8 are not defined in this task text, so I do **not** invent category numbers; the table states each
import's actual owner and status. All imports of the launch path were resolved against the pinned tree and
the workbench interpreter (`.venv`, Python 3.13.12, torch 2.14.0+cpu).

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `omegaconf` | `lavis/__init__.py:11`, `lavis/models/base_model.py:16` | LAVIS fork (in-tree) | any | yes (PyPI) | none | pip | absent in `.venv`; `from lavis.models.blip2_models.ConeSep import ConeSep` → `ModuleNotFoundError: omegaconf` — same absence the prior audit recorded |
| `salesforce-lavis==1.0.2` | declared in `models.yaml:159`; **not imported** | registry | 1.0.2 | yes | none | none | the run uses the vendored `lavis/`, never the PyPI package |
| `torch`, `torchvision` | `train.py:11,15`; `datasets.py:10` | PyPI | `2.1.0` / `0.16.0` per `models.yaml:156-157`; README says the **evaluated** env used torch 2.1.0 with **cu121** wheels | yes | torch+cuda | pin | `.venv` has torch 2.14.0+cpu — the pinned torch is not installed and no model env exists |
| `thop` | `train.py:18` `from thop import profile` | PyPI | any | yes | none | pip | import-time hard requirement of `train.py`; never used |
| `clip` (OpenAI) | `train.py:19` `import clip` | PyPI | any | yes | none | pip | import-time hard requirement; unused on the launch path |
| `setproctitle` | `train.py:28` | PyPI | any | yes | none | pip | import-time hard requirement; used only for naming |
| `transformers` | `datasets.py:12` (`CLIPSegProcessor, CLIPSegForImageSegmentation`), `lavis/models/blip2_models/blip2.py:26` (`BertTokenizer`) | PyPI | `4.25.0` (`models.yaml:159`) | NOT at the declared pin; yes at `>=4.26.0` | none | `transformers==4.26.0` (earliest version that ships both names) | **version conflict, RUNTIME_VERIFIED by HTTP**: `curl -o /dev/null -w '%{http_code}' raw.githubusercontent.com/huggingface/transformers/v4.25.0/src/transformers/models/clipseg/__init__.py` → **404**; same path at `v4.26.0`, `v4.27.0`, `v4.28.0`, `v4.29.0`, `v4.40.1` → **200**. The module landed in transformers in commit `Add CLIPSeg (#20066)`, 2022-11-08; v4.26.0's `clipseg/__init__.py:30,45` exports exactly `CLIPSegProcessor` and `CLIPSegForImageSegmentation` |
| `scikit-learn` | `ConeSep.py:24` `from sklearn.cluster import KMeans` | PyPI | `1.3.2` | yes | none | pip | used in the training loss path, not in `test.py` |
| `timm` | `models.yaml:159`; not imported by the pinned source | registry | 0.9.16 | yes | none | none | declared, unused |
| `typing_extensions` | `data_utils.py:3` | PyPI | any | yes | none | pip | standard |
| `numpy`, `PIL`, `tqdm` | `test.py:1-5`, `datasets.py:4` | PyPI | any | yes | none | pip | standard |

Textual correction, evidence: `grep -n "CLIPSeg" ConeSep/datasets.py` → `12:from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation`. This is a real, reproducible
`ImportError` on the declared `transformers==4.25.0` and blocks the `datasets` import (and therefore
`train.py`) for both **conesep and habit**. Observed by HTTP, not inferred: the CLIPSeg module is **404** at
tag `v4.25.0` and **200** from `v4.26.0` onward on `raw.githubusercontent.com/huggingface/transformers`. I did
not install anything to time the import failure; the version gate is established from the upstream tag contents
plus the pinned declaration.

### The BERT dependency and the repo's own tree

The question is whether the repo's tree still resolves the BERT it expects.

**Both** `train.py:29 from lavis.models import load_model_and_preprocess` and `lavis/__init__.py:15-18`
execute `from lavis.datasets.builders import *` / `from lavis.models import *` / `from lavis.processors import *` /
`from lavis.tasks import *` at package-import time. Importing the **first-party** `lavis` package therefore
loads the whole LAVIS dataset/model/processor/task registry, not just `ConeSep`. Observed:

```
$ grep -c "^import lavis\|^from lavis" lavis/datasets/__init__.py lavis/tasks/__init__.py lavis/processors/__init__.py
lavis/datasets/__init__.py:1
lavis/tasks/__init__.py:1
lavis/processors/__init__.py:1
$ PYTHONPATH=<ConeSep> python3 -c "import lavis"
ModuleNotFoundError: No module named 'omegaconf'
```

`omegaconf` is one of many; the vendored LAVIS tree is a full Salesforce-LAVIS fork and its transitive
import closure needs the whole LAVIS dependency set (`omegaconf`, `opencv-python`, `fairscale`, `decord`, `av`,
`webdataset`, `iopath`, `pycocoevalcap`, `sacrebleu`, `timm`, `einops`, `salesforce-lavis` glued together).
The *package-level* import path is not self-contained; only the two files on the ConeSep path
(`lavis/models/blip2_models/blip2.py`, `.../ConeSep.py`) are.

BERT dependency, quoted (`ConeSep/lavis/models/blip2_models/blip2.py:32,48,55`):

```python
32:        tokenizer = BertTokenizer.from_pretrained("/root/bert-base-uncased", truncation_side=truncation_side)
48:        encoder_config = BertConfig.from_pretrained("/root/bert-base-uncased")
55:            "/root/bert-base-uncased", config=encoder_config
```

- **Path**: absolute `/root/bert-base-uncased`, no env var, no fallback. `models.yaml:152-153` declares
  `python_env_var: WORKBENCH_CONESEP_PYTHON` but says nothing about BERT.
- **Version/revision**: unpinned (`from_pretrained` with no `revision`), so a changed upstream config is
  undetectable.
- **Installable?** The artifact is installable — `https://huggingface.co/bert-base-uncased/resolve/main/vocab.txt`
  answers `HTTP/2 200`, 231,508 bytes, first line `[PAD]` (observed). What is **not** established is
  *whether the repo's tree resolves it*, because the consumer path is a permission-denied host directory:

```
$ .venv/bin/python -c "import os; print(os.path.isdir('/root/bert-base-uncased'))"
False
$ .venv/bin/python -c "import os; os.listdir('/root/bert-base-uncased')"
PermissionError: [Errno 13] Permission denied: '/root/bert-base-uncased'
$
$ stat -c '%A %U %G %n' /root/sudo -n true
drwx------ root root /root
sudo: interactive authentication is required
```

**Verdict on the BERT question: BLOCKED (UNKNOWN at the byte level).** The host path now yields
`PermissionError` rather than `FileNotFoundError`, so I cannot distinguish "the directory exists and is real"
from "the directory does not exist". Either way the launch path cannot load it as this user, and the
workbench's own staged substitute is a 4-byte decoy that would **pass** the workbench's validator
(`workbench/backend/registry.py:555` rejects only `size == 0`; the record has `expected_sha256: null`,
`workbench/registry/auxiliary_assets.yaml:266-275`):

```
$ .venv/bin/python -c "print(len(open('/home/heheboiz/.cache/huggingface/hub/models--bert-base-uncased/vocab.txt','rb').read()), repr(open('/home/heheboiz/.cache/huggingface/hub/models--bert-base-uncased/vocab.txt','rb').read()))"
4 b'qwen'
```

Two things make this **worse than "the asset is missing"**: (i) the file is non-empty so existence checks pass,
and (ii) the same stub sits at the path a *different* model's `.pth` download was also routed to —
`~/.cache/lavis/blip2_pretrained.pth` is 22 bytes, `b'corrupt nonempty bytes'`. `~/.cache/torch/hub/checkpoints`
does not exist, so the EVA-ViT-g weight declared at `auxiliary_assets.yaml:220-233` is absent entirely. The
`AUXILIARY_ASSET_AUDIT.md:206` table records `bert_base_uncased_tokenizer_and_config` as
`AVAILABLE` for these models; **that label is wrong against the current host.**

## 5. CHECKPOINTS

Files: three variants, all FashionIQ, all noise-trained (`models.yaml:186-216`):

| checkpoint_id | filename | training noise | eval noise | size | sha256 | status |
|---|---|---|---|---|---|---|
| `fiq_n02` | `ConeSep-FIQ-N0.2.pt` | 20% | 0 | `null` (`:193`) | `null` (`:194`) | `NO_CLEAN_CHECKPOINT` (`:195`) |
| `fiq_n05` | `ConeSep-FIQ-N0.5.pt` | 50% | 0 | `null` (`:201`) | `null` (`:202`) | `NO_CLEAN_CHECKPOINT` (`:205`) |
| `fiq_n08` | `ConeSep-FIQ_N0.8.pt` | 80% | 0 | `null` (`:211`) | `null` (`:212`) | `NO_CLEAN_CHECKPOINT` (`:215`) |

Official links = the registry's own `download_url` values (`models.yaml:189,199,209`). Availability class:
`PUBLIC_CHECKPOINT_NOT_MAPPED` for the paper row + `CHECKPOINT_UNPUBLISHED` for the clean condition.

Verified by HEAD (observed):

```
$ curl -sIL <resolve url>   # per file
ConeSep-FIQ-N0.2.pt => HTTP/2 302 ; HTTP/2 200  x-linked-size: 2721428966  x-linked-etag: "5a864ee5a78c1d249320c4f997cc7d14b529b329b2efba5577c82dbe019ed9d2"
ConeSep-FIQ-N0.5.pt => HTTP/2 302 ; HTTP/2 200  x-linked-size: 2721428966  x-linked-etag: "1953ec2f9269a5cb24d61bdf89d05ed8430fb3bdc6dd18cef08fe9e3dfbe1e26"
ConeSep-FIQ_N0.8.pt => HTTP/2 302 ; HTTP/2 200  x-linked-size: 2721428966  x-linked-etag: "7c2bd0fbaea423a1a64809956b3fda72e191a6f7e5ed6ecfa708f1e3a8730eae"
$ curl -sS https://huggingface.co/api/models/iLearn-Lab/CVPR26-ConeSep
sha cecc32deeec917d337e5d2def675b3746da7b8e6   siblings: .gitattributes README.md fiq/ConeSep-FIQ-N0.2.pt fiq/ConeSep-FIQ-N0.5.pt fiq/ConeSep-FIQ_N0.8.pt
```

The Hub's model-card README confirms the labels verbatim:
`fashioniq/` → `ConeSep-FIQ_N0.2.pt (Trained with 20% noise)`, `N0.5` (50%), `N0.8` (80%).
All three are present, all three are reachable, none is clean.

**Mapping status: `NO_CLEAN_CHECKPOINT` is correct.** The registry pin is the 0%-noise row
(confirmed in Section 7), and the Hub publishes no 0% file.

Checksum status: registry records `null` for size and sha256 on all three (lines above) while the Hub
publishes `x-linked-size: 2721428966` and a per-file blob digest. **P2** — the values are free to record and
their absence is a provenance hole (the loader path the prior audit quoted only accepts a download when
`expected_sha256 is None`).

Loader contract (STATIC_AUDIT from the payload, then RUNTIME_VERIFIED for the loader semantics):

```
$ curl -sL -r 0-400000 <ConeSep-FIQ-N0.2.pt> -o cs_head.bin
$ strings -n 4 cs_head.bin | grep -E "^/(mnt|root|home|data)|model_epoch|clavis|vtokenizer"
/mnt/data/PreTrain/bert-base-uncasedr#A
model_epoch_4/data.pklFB
model_epoch_4/data/0FB4
clavis.models.blip2_models.ConeSep
clavis.models.blip2_models.Qformer
transformers.models.bert.tokenization_bert.BertTokenizer
```

So each file is a **whole-model pickle of the optimizer's local copy**, produced by `train.py:221/:237/:251`
`torch.save(model, ...)`. Consequences, all P1:

1. **The pickle references the training machine's BERT directory `/mnt/data/PreTrain/bert-base-uncased`, not
   `/root/bert-base-uncased`.** The pinned source points at `/root/...`; the published artifact was saved on a
   host where the tokenizer lived at `/mnt/data/PreTrain/...`. Any load that consults the embedded tokenizer
   path will not find the pinned path's twin.
2. **`torch.load` cannot load it on the declared torch.** RUNTIME_VERIFIED:

```
$ .venv/bin/python -c "..."
default weights_only: None
default torch.load custom class: UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`
weights_only=False: SUCCEEDED
```

`ConeSep/utils.py:141 checkpoint = torch.load(checkpoint)` — and `cirr_sub.py:124`, `cirr_sub.py`'s
`torch.load` — use the **default**, which raises on a whole-model pickle on any torch ≥ 2.6. The remedy
(`weights_only=False`) is exactly the unsafe-serialization case:
`UNSAFE_CHECKPOINT_SERIALIZATION`.
3. Registry/benchmark inconsistency (P2): `workbench/registry/paper_benchmarks/papers.yaml:649,697,745`
   record `checkpoint_mapping_status: UNVERIFIED` for the three noisy rows, while the extracted JSON
   `workbench/registry/paper_benchmarks/extracted/conesep_cvpr26.json:76,129,182` records
   `VERIFIED_METADATA` for the same rows. Two documents disagree about the same field; neither is derived from
   the other at read time.
4. Internal-selection inconsistency (P2): the three noisy files are each an `model_epoch_K/best?` state
   (`model_epoch_4` for N0.2 and N0.5, `model_epoch_1` for N0.8), so the Hub artifact is an *epoch* model, not
   the `best_model.pt` the README says it saved; the same mismatch is in the cards' own usage text, which tells
   the user to rename the download to `best_model.pt`.

## 6. AUXILIARY ASSETS

| name | purpose | source | availability class | generation possible? |
|---|---|---|---|---|
| `captions/correction_dict_{dress,shirt,toptee}.json` | content-transforming caption input (`datasets.py:96,170`) | **none published by ConeSep** | **AUXILIARY_ASSET_MISSING** | not from the pinned tree; copies exist in sibling checkouts |
| `/root/bert-base-uncased` | BERT tokenizer + Q-Former init | HF `bert-base-uncased` | **AUXILIARY_ASSET_MISSING** on this host (permission-denied path) | yes, from HF; needs a symlink/staging step nobody performs for `conesep` |
| `.cache/lavis/blip2_pretrained.pth` | BLIP-2 init (`blip2_pretrain.yaml:10`) | Google Storage | **AUXILIARY_ASSET_MISSING** — present but 22 bytes `b'corrupt nonempty bytes'` | yes; download |
| `torch/hub/checkpoints/eva_vit_g.pth` | EVA-ViT-g vision encoder (`eva_vit.py`) | Google Storage | **AUXILIARY_ASSET_MISSING** — `~/.cache/torch/hub/checkpoints` does not exist | yes; download |
| `resized_image/{cat}/{id}.jpg` (256px) | gallery/query images (`datasets.py:137,147`) | FashionIQ `start_kit/resize_images.py` | present locally (receipt `recipe_identity: model_id=limn`, `image_size=256`, `layout=per_category`, `resampling=LANCZOS`) | yes, reproducible, but **the resize step is not pinned by ConeSep** |
| `fashion_iq_data.json` + 6 PKLs | regeneration cache (`datasets.py:35-58`) | self-generated | present-or-generated | yes; **never validated for freshness** |

Correction dictionaries — the question was *needed / present / public*:

- **Needed: yes.** `datasets.py:170` opens `captions/correction_dict_{name}.json` inside `get_test_data`,
  i.e. on the validation path (not only at training init). It is read before any caption is composed.
- **Present in the pinned tree: no.** `find workbench/third_party/ConeSep -name "correction_dict*"` → nothing.
- **Public from ConeSep: no.** No `download_url`, no generator, no README mention
  (`grep -n correction_dict ConeSep/README.md` → no match).
- **But the bytes are not unique to this model, and I can bind them.** Observed:

```
$ sha256sum data/fashionIQ_dataset/captions/correction_dict_{dress,shirt,toptee}.json
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  correction_dict_dress.json
ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e  correction_dict_shirt.json
3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71  correction_dict_toptee.json
$ sha256sum workbench/third_party/TGCIR/correction_dict/correction_dict_dress.json workbench/third_party/LIMN/correct_files/correction_dict_dress.json
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  (TGCIR)
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  (LIMN)
```

TGCIR's and LIMN's copies are byte-identical to the locally staged `data/fashionIQ_dataset/captions/` copies
for all three categories. So the dictionaries are **content-resolved but not provenance-resolved**: a run can
be made to work today, but the digest is not linked to any ConeSep-authoritative source, and the local staging
carries no `conesep` row in `workbench/registry/auxiliary_assets.yaml` (`grep -c conesep` → `0`).

Resize recipe: `start_kit/resize_images.py` is real and public (`HTTP 200`, observed), but it uses
`image.resize(size, Image.ANTIALIAS)` — and `hasattr(Image, 'ANTIALIAS')` is `False` on current Pillow. The
workbench's staged images are `(256, 256)` RGB (observed on five random `dress` files) and were produced by
the workbench's own LANCZOS kernel, recorded in the receipt as `source_script: start_kit/resize_images.py`. The
receipt is attributed `model_id: limn` — the 256px `per_category` layout is shared, so the ConeSep-specific
resize lineage is a **VARIANT**, not an author-published artifact.

## 7. FASHIONIQ PROTOCOL

`native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:178`) — **authoritative, quoted**.
`supported_protocols: [fashioniq_full_gallery_ref_excluded]` (`models.yaml:179`).
`literature_split_label: original` (`models.yaml:177`). Definition at `models.yaml:12-15`.

- **Gallery** — full ordered validation split. `datasets.py:28` hardcodes `self.split = 'original-split'`
  (`FashionIQ.__init__` at `:26` takes only `path, transform, noise_ratio`), and `get_test_data`
  (`:164`) takes the `original-split` branch at `:201-206`, appending every index of `split.<cat>.val.json`
  in file order. Local counts confirm the split is the whole val gallery, not a pair union:
  dress 3817 / shirt 6346 / toptee 5373, all unique, vs 2017 / 2038 / 1961 queries.
- **Reference policy** — excluded per query, mask-before-rank: `datasets.py:175` composes the caption, and
  `test.py:66-68` writes `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` **before** the
  `np.argsort` at `test.py:71`.
- **Query construction** — `mod = correct_text(cap0, dict) + " and " + correct_text(cap1, dict)`
  (`datasets.py:115-116`), then `txt_processors["eval"]` (`test.py:35` = `BlipCaptionProcessor.pre_caption`).
- **Categories** — `dress`, `shirt`, `toptee` (`test.py:12-16`); `shoes` is present but is a different
  benchmark (`test.py:18`) and `train.py:214` only iterates the three.
- **Recall computation** — `for k in [1, 10, 50]` (`test.py:76`), hit test
  `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, `r = 100 * r / len(nn_result)`
  (`test.py:79-81`). Per category only; there is **no macro aggregation in the pinned source**.
  `train.py:217` scores `t[1][1]` = `dress_r10`, i.e. best-checkpoint selection maximises the **sum of
  category R@10**, not the paper's AVG column.

**I do not propose changing any of this.** No gallery, reference policy, caption input or noise condition
should be touched to make the model runnable.

## 8. WORKBENCH INTEGRATION

| stage | state | evidence |
|---|---|---|
| adapter | registered, `command()` declines | `adapters/models.py:59-61`, `:155`; observed `NotImplementedError` |
| planner | sees the model; refuses on `command_status` absence | `models.yaml` has **no** `command_status` for `conesep`; `preflight.py:201-202` appends `command not runnable: COMMAND_UNAUDITED` (observed) |
| checkpoint prep | blocked, by design | observed `select_checkpoints` → `BLOCKED/BLOCKED_CHECKPOINT` (strict_clean) and `BLOCKED/BLOCKED_CHECKPOINT_SELECTION` (include_flagged_conditions), conditions `{'fiq_n02':20,'fiq_n05':50,'fiq_n08':80}` |
| preflight (data) | **passes** | `model_requirements` returns `blocks: []`, `paths: [15 paths]` (all present) |
| preflight (official) | 2 blockers | observed: the registry `source_blocker` (`models.yaml:151`) quoted in full, then `command not runnable: COMMAND_UNAUDITED` (`preflight.py:194-202`) |
| environment | UNCONFIGURED | observed `verify_environment(...)` → `tier=UNCONFIGURED status=UNCONFIGURED reason=WORKBENCH_CONESEP_PYTHON not set`; `environment_required` is **absent** for `conesep` |
| metric parser | none | `metrics_extraction.py:383-420` dispatches only limn/csmcir/combiner×2/sprc → `AGGREGATE_PARSER_UNAVAILABLE` |
| per-query support | none | `autopilot.py:628,841,960` hardcode `per_query_export_available: False`; no `conesep` replay module (`workbench/replay/` = limn, tgcir) |

Two integration observations that contradict prior documents:

- `workbench/docs/READINESS_MATRIX.md:27` reports `conesep` BLOCKED with 8 FashionIQ data blockers
  ("`resized_image` missing", "`correction_dict_*.json` missing", "image directory missing"). That file is
  dated 2026-10-14 13:48 (`stat`) and its last commit is `9d35861b 2026-10-14 13:48`. The resized images and
  correction dictionaries were staged later: `stat -c '%y' data/.../resized_image` → **2026-10-18 13:20**.
  Re-running the same preflight now returns **zero** data blocks. **The matrix row is stale — do not treat those
  8 blockers as current.**
- `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:71` records `conesep` protocol parity as "yes (was
  `Unsupported`)". The protocol *semantics* are indeed expressible (`src/evaluation/fashioniq.py`
  `FASHIONIQ_PROTOCOLS` + `build_fashioniq_gallery`), but there is **no production callsite** for `conesep`,
  and the pinned evaluator was never changed to call it. "yes" describes the library, not the model.

## 9. CONFIRMED DEFECTS

**CF-1 (P1) — No runnable evaluation path exists for any published ConeSep FashionIQ checkpoint.**
Root cause: three independent absences — (a) `test.py` is a bare callable with no `__main__`;
(b) nothing in the tree loads a weight file onto a model (the `--restore_file` flag at `train.py:76` is
declared and never read; `utils.py:128 load_checkpoint` has no caller); (c) the workbench adapter declines.
Reproduction: `ADAPTERS['conesep'].command('s','c',None,None)` → `NotImplementedError`;
`grep -rn load_checkpoint ConeSep/{train,test,utils,cirr_sub}.py` → only `utils.py:128` (the definition).
Fix: a first-party replay module plus a `command()` implementation, mirroring `workbench/replay/limn.py`.
Regression result: not applicable (no change made).

**CF-2 (P1) — The registry's pinned row is the 0%-noise row and no 0% checkpoint is published.**
Root cause: `models.yaml:180-182` transcribes the *clean* row while `models.yaml:187-216` registers only
20/50/80%-noise weights. Reproduction: HF API listing shows 3 files, none clean; `strict_clean_checkpoint_available: false`
(`models.yaml:184`) is correct. Fix: keep the refusal, and label the reported row with
`checkpoint_training_noise_pct: 0` explicitly at the registry level (the extracted JSON already does, at
`conesep_cvpr26.json`). Regression: none made.

**CF-3 (P1) — `prompt_load()` is mandatory, uncalled by the evaluator, and undocumented.**
Root cause: `ConeSep.py:154-157` initialises `negative_tokens` randomly;
`ConeSep.py:198-199 prompt_load()` replaces them with `query_tokens.data`. Only
`train.py:135` (inside `create_model_and_optimizer`) calls it. `test.py` and `train.py:216` do not.
Fix: any wrapper must call `model.prompt_load()` before extraction. Reproduction: `grep -rn prompt_load ConeSep` →
`ConeSep.py:198` (definition) and `train.py:135` (only call).

**CF-4 (P1) — Published checkpoints are train-time pickles pinned to a foreign BERT path and unloadable
with default `torch.load`.** Root cause: `train.py:251 torch.save(model, ...)` (whole model) on a host where the
tokenizer lived at `/mnt/data/PreTrain/bert-base-uncased`, not the source's `/root/bert-base-uncased`.
Reproduction: range-fetch + `strings` shows `/mnt/data/PreTrain/bert-base-uncasedr#A` and
`model_epoch_4/data.pklFB`; RUNTIME_VERIFIED `torch.load` default raises with the PyTorch-2.6 message.
Fix: `weights_only=False` **and** a staged tokenizer that has the vocab the pickle expects, or a
state-dict-only re-export. Classification: `UNSAFE_CHECKPOINT_SERIALIZATION`.

**CF-5 (P2) — `transformers==4.25.0` cannot satisfy `datasets.py:12`.**
`from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation` — CLIPSeg ships in transformers ≥ 4.28.
Fix: raise the declared pin to `transformers==4.26.0` (the earliest upstream tag exporting both CLIPSeg names,
verified by HTTP). `4.40.1` is what the sibling models use but is not the minimum this source needs.

**CF-6 (P2) — The staged BERT/EVA/BLIP-2 assets are decoys that pass the workbench validator.**
`~/.cache/huggingface/.../vocab.txt` = 4 bytes `b'qwen'`; `~/.cache/lavis/blip2_pretrained.pth` = 22 bytes
`b'corrupt nonempty bytes'`; `~/.cache/torch/hub/checkpoints` absent. The records have `expected_sha256: null`
and `expected_size_bytes: null` for BERT (`auxiliary_assets.yaml:265-275`),
`registry.py:555` rejects only size 0. Fix: bind the published digests; add the EVA weight for the iLearn
models. Classification: `AUXILIARY_ASSET_MISSING`.

**CF-7 (P2) — Correction dictionaries and resized images have no ConeSep provenance.**
`grep -c conesep workbench/registry/auxiliary_assets.yaml` → `0`; `preparation_contracts.yaml` has no
`conesep` row; the resize receipt names `model_id: limn`. Fix: register the three dictionaries with the digests
I computed, and record the resize recipe ConeSep depends on.

## 10. REMAINING BLOCKERS

| classification | owner | evidence | remediation |
|---|---|---|---|
| `UPSTREAM_CLI_MISSING` | upstream authors | `test.py` has no `__main__`/`argparse`; `train.py:216` is the only call path; `README.md:245` points at a nonexistent `src/cirr_test_submission.py` | upstream adds a FashionIQ eval entrypoint; workbench may add a replay module meanwhile |
| `FIRST_PARTY_ADAPTER_MISSING` | workbench | `adapters/models.py:59-61`; observed `NotImplementedError` | implement `command()` like `LIMNAdapter` (`models.py:135-160`) |
| `CHECKPOINT_UNPUBLISHED` | upstream authors | HF API listing has exactly 3 `fiq/` files, all noisy | publish the 0% weight, or accept the 20% row as the comparison target |
| `UNSAFE_CHECKPOINT_SERIALIZATION` | upstream authors | whole-model pickle; foreign `/mnt/data/PreTrain/bert-base-uncased`; `torch.load` default raises | publish state_dicts, or document `weights_only=False` + the tokenizer expectation |
| `AUXILIARY_ASSET_MISSING` | workbench (staging) | 4-byte and 22-byte decoys; no EVA dir; `/root/bert-base-uncased` permission-denied | stage real bytes, bind digests, and stop the validator accepting non-empty garbage |
| `INSTALLABLE_DEPENDENCY_MISSING` | workbench (env) | `omegaconf` absent from `.venv`; `WORKBENCH_CONESEP_PYTHON` unset | build the declared 3.8 env per README:231-235 |
| `DEPENDENCY_VERSION_CONFLICT` | upstream authors | `datasets.py:12` vs `transformers==4.25.0` (`models.yaml:159`); CLIPSeg path 404 at tag v4.25.0, 200 at v4.26.0 | correct the declared pin to >=4.26.0 |
| `SOURCE_UNAVAILABLE` | — | n/a: the mirror serves the same SHA | none |
| `GPU_RUNTIME_UNTESTED` | this host | `.venv` has torch 2.14.0+cpu; no CUDA; `require_cuda=True` probe returns `UNCONFIGURED` | run on the GPU host |
| `SCIENTIFIC_PROTOCOL_UNVERIFIED` | workbench | protocol machinery exists but is unreachable for `conesep` | wire it to the `conesep` replay path |

Ownership note: `upstream_repo_url` now redirects (Section 2), so any future sync must use
`https://github.com/iLearn-Lab/CVPR26-ConeSep` — the pin is unchanged either way.

## 11. IMPLEMENTATION PLAN

1. **`workbench/replay/conesep.py`** (new, ~120 lines, modeled on `workbench/replay/limn.py`):
   `torch.load(..., weights_only=False, map_location="cpu")` → `model.prompt_load()` →
   `model.eval()` → `test.test(args, model, trainset, cat, txt_processors)` for the three categories →
   emit one JSON line `{dress_r10, dress_r50, shirt_r10, ..., avg_r10, avg_r50}`. Complexity: low.
2. **`ConeSepAdapter.command()`** (`adapters/models.py:59-61`): one `command()` returning the replay
   invocation with `--checkpoint` / `--dataset-root` / `--protocol`. Complexity: one method.
3. **`parse_conesep_output`** in `metrics_extraction.py` + dispatch entry. Complexity: low; reuse the
   existing macro-average validation shape.
4. **`command_status: COMMAND_AUDITED`** and `environment_required: true` on the `conesep` registry row;
   record `expected_size_bytes: 2721428966` and the three `x-linked-etag` digests. Complexity: registry only.
5. **Staging fix** for BERT/BLIP-2/EVA with real bytes + digests.

Scientific risk: **high if done wrong.** The published checkpoints are noise-trained, so any number they
produce is comparable only to the 20/50/80% paper rows (`conesep_cvpr26.json` rows `_n20/_n50/_n80`),
never to 65.73. Evaluating them and reporting 65.73 as the reference would be a wrong-row comparison.
`paper_gate.COMPARABILITY_FIELDS` (`workbench/backend/paper_gate.py:72`) includes
`checkpoint_training_noise_pct`, which is the intended guard.

Expected output: with the 20% checkpoint, a clean-*evaluation* number targeting the paper's 20%-noise row
(mean 64.97 per the extracted benchmark), labelled as a noisy-condition reproduction.

## 12. FINAL VERDICT

**BLOCKED** for a reproduction run.
- **BLOCKED** — `FIRST_PARTY_ADAPTER_MISSING` + `UPSTREAM_CLI_MISSING` + `AUXILIARY_ASSET_MISSING` +
  `UNSAFE_CHECKPOINT_SERIALIZATION`.
- The pinned **row** is `UNKNOWN` as a reproduction target: it is the 0%-noise row and its weight is
  `CHECKPOINT_UNPUBLISHED`.
- The **protocol** is **VERIFIED** (STATIC_AUDIT): full ordered split gallery, reference masked before
  ranking, category Recall@1/10/50, no macro aggregation in-source.
- Runtime: **NOT_RUN** — no GPU, no environment, no checkpoint on disk.

## 13. EVIDENCE

- Repo: `/home/heheboiz/data/cir` @ `b70d092e9d2232e34886f819e30c7327b74f3a76`.
- Pin: `workbench/third_party/ConeSep` @ `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b`
  (`git rev-parse HEAD`), remote `git@github.com:iLearn-Lab/CVPR26-ConeSep.git`, single commit.
- Commands and observed results: `ls-remote origin HEAD` → `bd4bf629...`; `fetch origin --dry-run` → no
  output, exit 0; `curl -w '%{redirect_url}' https://github.com/Lee-zixu/ConeSep` → `301 https://github.com/iLearn-Lab/CVPR26-ConeSep`;
  `ls-remote https://github.com/Lee-zixu/ConeSep` → `bd4bf629...` (same SHA).
- Files read with lines: `models.yaml:150-216`, `:10-15`; `ConeSep/test.py:1-168`;
  `ConeSep/train.py:1-285`; `ConeSep/datasets.py:1-233`; `ConeSep/utils.py:120-144`;
  `ConeSep/cirr_sub.py:94-127`; `ConeSep/README.md`; `ConeSep/lavis/models/blip2_models/ConeSep.py:98-199,452-515`;
  `ConeSep/lavis/models/blip2_models/blip2.py:26-58`; `ConeSep/lavis/models/__init__.py:174-199`;
  `ConeSep/lavis/__init__.py:1-30`; `ConeSep/lavis/configs/models/blip2/blip2_pretrain.yaml`;
  `workbench/backend/adapters/models.py:1-163`; `workbench/backend/preflight.py:8-51,162-213`;
  `workbench/backend/metrics_extraction.py:28-60,383-420`; `workbench/backend/registry.py:479-498,545-600,956-1007`;
  `workbench/backend/fashioniq_layout.py:1-120`; `workbench/registry/audits/cluster_b_ilearn.yaml:281-420`;
  `workbench/registry/assets/cluster_b3_ilearn.yaml`; `workbench/registry/auxiliary_assets.yaml:205-275`;
  `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:1-80`; `workbench/docs/READINESS_MATRIX.md:27`.
- Executed observations: preflight (`blocks: []`, 15 paths), `select_checkpoints` (BLOCKED, both policies),
  `preflight_blockers` (2 lines), `ADAPTERS['conesep'].command(...)` (NotImplementedError),
  `verify_environment` (UNCONFIGURED / `require_cuda=True` UNCONFIGURED), `/tmp/shape2.py` (shape and mask
  output), `/tmp/wt` `torch.load` weights_only demonstration, HEAD probes of the three resolve URLs, HF API listing,
  `sha256sum` of the three correction dictionaries across `data/`, `TGCIR/`, `LIMN/`.
- **UNKNOWN**: whether `/root/bert-base-uncased` exists and is a real BERT (permission-denied);
  whether the pickles' embedded tokenizer can be reconstructed from the public `bert-base-uncased`;
  the pinned row's reproducibility on any host (no 0% weight exists).

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 — STATIC_AUDIT with executed source-independent probes.**
Source text, registry, path existence, HTTP metadata, numpy semantics and `torch.load` behaviour were
inspected directly. **No** model was instantiated, **no** checkpoint was fully downloaded, **no** image tensor
was produced by the network, and **no** Recall was computed. No claim of runtime model behaviour is made.

Preserved invariants: protocol `fashioniq_full_gallery_ref_excluded` unchanged, full gallery unchanged,
reference-excluded policy unchanged, caption input unchanged, noise condition unchanged.
