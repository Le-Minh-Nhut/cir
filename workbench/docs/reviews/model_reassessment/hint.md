# hint — reproducibility reassessment

Independent adversarial audit. READ-ONLY: no file under `/home/heheboiz/data/cir` was modified; all
scratch went to `/tmp`. No GPU job, no install, no checkpoint download. Every claim is STATIC_AUDIT
unless the line says otherwise; each cites a file:line I opened in the pinned checkout or a command
with its observed output. There is no RUNTIME_VERIFIED claim in this document.

Pre-existing audits of this model were treated as HYPOTHESES. Where my reading disagrees, the code is
shown (§9 D-HINT-5, D-HINT-6) and findings are stated as REFUTED or UPHELD explicitly.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `hint` | `workbench/registry/models.yaml:368` |
| method_name | `HINT` | `models.yaml:377` |
| paper title | HINT: Composed Image Retrieval with Dual-Path Compositional Contextualized Network | `models.yaml:381`; pinned `workbench/third_party/HINT/README.md:5` |
| paper URL | https://arxiv.org/abs/2603.26341 | `models.yaml:382` |
| year | 2026 | `models.yaml:379` |
| source URL | https://github.com/iLearn-Lab/ICASSP26-HINT | `models.yaml:383` |
| pinned commit | `bec50b6c8c19111893b617979502b948d1cea5b2` | `models.yaml:385` |
| source_dir | `HINT` | `models.yaml:387` |
| checkpoint variant | one file `fashioniq.pt`, training condition **unstated** | `models.yaml:399-414` |
| original metrics | R@10 58.48, R@50 77.93, mean 68.21 | `models.yaml:393-395` |
| native protocol | `fashioniq_val_split` | `models.yaml:390` |

Pin verification (command, observed):

```
$ git -C workbench/third_party/HINT rev-parse HEAD
bec50b6c8c19111893b617979502b948d1cea5b2        # == models.yaml:385
```

arXiv reachability, observed: `curl -s https://arxiv.org/abs/2603.26341` →
`<title>[2603.26341] HINT: Composed Image Retrieval with Dual-path Compositional Contextualized Network</title>`;
`curl -s -o /dev/null -w "%{http_code}" https://arxiv.org/pdf/2603.26341v1` → `200`. VERIFIED.
(Case differs between the registry title and the arXiv page: "Dual-Path" vs "Dual-path"; cosmetic.)

---

## 2. SOURCE STATUS

Accessible and complete as a tree. `git -C workbench/third_party/HINT status --porcelain` reports only
untracked `__pycache__/` directories — no tracked modification, no workbench patch for this model.

Two things are absent, and both matter:

* `requirements.txt` does not exist, and this README does not reference one
  (`grep -c "requirements.txt" workbench/third_party/HINT/README.md` → `0`), so the env declaration in
  `models.yaml:369-376` is the only machine-readable spec.
* `src/` does not exist, yet the README's own test command points into it:
  `README.md:231` `python src/cirr_test_submission.py checkpoints/cirr_hint/`; observed
  `ls workbench/third_party/HINT/src` → `No such file or directory`. The script is at the repo root
  (`workbench/third_party/HINT/cirr_test_submission.py`).

**SNAPSHOT 2 — current upstream state.** Read-only remote inspection, pin untouched:

```
$ git -C workbench/third_party/HINT ls-remote origin
bec50b6c8c19111893b617979502b948d1cea5b2        HEAD
bec50b6c8c19111893b617979502b948d1cea5b2        refs/heads/main
$ git -C workbench/third_party/HINT log --oneline --all
bec50b6 Create LICENSE
$ git -C workbench/third_party/HINT rev-list --count HEAD
1
$ git -C workbench/third_party/HINT fetch origin --dry-run
(no output, no error — nothing to fetch)
```

Upstream HEAD **equals** the pin; the repository is still a single squashed commit titled
"Create LICENSE". No CLI was added, no dependency was fixed, no checkpoint was published since the
pin. **SNAPSHOT 2 == SNAPSHOT 1.**

Note the author "moved" only in the sense that the Hub card decouples weights from code: the model card
at `iLearn-Lab/HINT` (repo created 2026-04-10, `lastModified 2026-04-11`, observed from the Hub API
`_id`/`sha`) publishes `fashioniq.pt` and `cirr.pt`, while the pinned GitHub commit is dated
`2026-04-09 16:38:14 +0800`. The code repository was never updated to load those weights (§5).

---

## 3. NATIVE EVALUATION WORKFLOW

**Entrypoint.** None. `workbench/third_party/HINT/test.py` is 190 lines with no `__main__` and no
`argparse` (`grep -n "__main__\|ArgumentParser" test.py` → zero hits). The only parsers in the tree are
`train.py:35-62` and `cirr_test_submission.py:89`.

**Exact call chain (every line opened):**

```
train.py:35-62            argparse -> args (dataset default 'cirr', :36; model_dir default './checkpoints/HINT', :53)
train.py:240,275          __main__ -> create_model_and_optimizer() -> train_and_evaluate(...)
train.py:116-121          create_model_and_optimizer(): load_model_and_preprocess(name="HINT", model_type="pretrain",
                                                                                  is_eval=False, device=args.device)
train.py:122              model.cuda()                                  # unconditional
train.py:71,73-89         load_dataset(): transform="targetpad", input_dim=224, target_ratio=1.25 (:74-75)
                          -> datasets.FashionIQ(path=args.fashioniq_path, transform=img_transform,
                                                noise_ratio=args.noise_ratio)     # train.py:86-89  <-- BROKEN, see §9
train.py:197-199          for ci, category in enumerate(['dress','shirt','toptee']):
                              t = test.test(args, model, trainset, category, txt_processors)
train.py:201              current_score = current_score + t[1][1]         # index [1] == R@10
test.py:11-12             def test(params, model, testset, category, txt_processors): model.eval()
test.py:14-19             category -> testset.test_queries_dress / _shirt / _toptee  (+ test_targets_*)
test.py:44-45             query batch: torch.stack(imgs).float().cuda(); mods=[txt_processors["eval"](c) for c in mods]
                          f = model.extract_retrieval_compose(imgs, mods)    -> HINT.py:401
test.py:60-61             gallery batch: imgs = model.extract_retrieval_target(imgs)  -> HINT.py:436
test.py:74                sims = np.matmul(all_queries, all_imgs)   # (M,256,1,1) x (N,256,32) -> (M,N,1,32)
test.py:75-76             sims = sims.squeeze(); sims = sims.max(-1)  -> (M,N)
test.py:79-83             test_targets_id = [...]; sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
test.py:86 / :100         nn_result = [np.argsort(-sims[i,:])[:50] for i in range(M)]
test.py:101-107           for k in [1,10,50]: r = 100*r/len(nn_result); out += [('<name>_r<k>', r)]
```

**Metric returned:** per-category `{dress,shirt,toptee} × {R@1,R@10,R@50}`; no macro, no mean.

**CLI?** No. **What a wrapper must call** — §5.

**Missing components.** No checkpoint-restore reachable from evaluation: `grep -rn
"load_checkpoint\|restore_file" HINT/*.py` → `train.py:59` (a declared-but-unused CLI flag) and
`utils.py:128` (definition, no caller). `train.py:240-275` runs `--num_epochs` (default 20, `:40`)
before producing any metric.

---

## 4. DEPENDENCIES

The D1–D8 legend is **not defined anywhere in this repository** — `grep -rn "D1" --include=*.md` over
tracked docs yields only the unrelated PC-launcher defect table at
`workbench/docs/reviews/final_pc_audit/AUDIT_F_ADVERSARIAL.md:143`. I therefore classify each missing
import by behaviour against the eight canonical dependency classes and name the class explicitly
instead of guessing a number. UNKNOWN-for-numbering, stated rather than assumed.

**Decisive distinction — reachable vs training-only.** The evaluation path for FashionIQ needs only
`test.py`, `datasets.py`, `data_utils.py`, `utils.py` and the `lavis` package that
`load_model_and_preprocess` comes from. Observed:

```
$ PYTHONPATH=<HINT> .venv/bin/python -c "import datasets, test, utils, data_utils"
datasets OK   test OK   utils OK   data_utils OK
$ PYTHONPATH=<HINT> .venv/bin/python -c "from lavis.models import load_model_and_preprocess"
ModuleNotFoundError: No module named 'torchaudio'    <-- chain below
  HINT/lavis/__init__.py:15                 from lavis.datasets.builders import *
  HINT/lavis/datasets/builders/base_dataset_builder.py:18
  HINT/lavis/processors/__init__.py:29      from lavis.processors.audio_processors import BeatsAudioProcessor
  HINT/lavis/processors/audio_processors.py:9   import torchaudio
$ ... with a stub torchaudio on PYTHONPATH (second failure)
  HINT/lavis/processors/audio_processors.py:17   from lavis.models.beats.Tokenizers import ...
  HINT/lavis/models/__init__.py:15               from lavis.models.albef_models.albef_classification import ...
  HINT/lavis/models/med.py:40-43                 from transformers.modeling_utils import (..., apply_chunking_to_forward, ...)
ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'
```

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `torchaudio` | `HINT/lavis/processors/audio_processors.py:9` | vendored LAVIS; declared in README | `2.1.0` (`README.md:126`) | YES | py38-312 / 2.1.0 / cpu+cuda | install into the model env | **REACHABLE — this is the exact first failure of `from lavis.models import …`** |
| `torchaudio.transforms` | `audio_processors.py:10` | same | `2.1.0` | YES | same | same | REACHABLE (same module) |
| `torchaudio.compliance.kaldi` | `audio_processors.py:13` | same | `2.1.0` | YES | same | same | REACHABLE (same module) |
| `moviepy.editor` | `audio_processors.py:11` | vendored LAVIS | unpinned upstream | YES | pure py | install | REACHABLE (same module) |
| `torchaudio` | `HINT/lavis/datasets/datasets/audio_captioning_datasets.py:16` | vendored LAVIS | `2.1.0` | YES | same | same | REACHABLE: `lavis/datasets/builders/__init__.py:93` imports `audio_caption_builder` |
| `torchaudio` | `HINT/lavis/datasets/datasets/audio_classification_datasets.py:16` | vendored LAVIS | `2.1.0` | YES | same | same | REACHABLE via `classification_builder.py:14` |
| `pandas` | `audio_captioning_datasets.py:15` | vendored LAVIS | unpinned | YES | pure py | install | REACHABLE |
| `torchaudio.compliance.kaldi` | `HINT/lavis/models/beats/BEATs.py:14`, `beats/Tokenizers.py:14` | vendored LAVIS | `2.1.0` | YES | same | same | REACHABLE — imported by `audio_processors.py:17` on the failing chain |
| `setproctitle` | `HINT/train.py:17` and `HINT/train.py:246` | first-party trainer | unpinned | YES | pure py | install | **TRAINING-ONLY — NOT on the evaluation path.** `test.py` never imports it; `grep -rn setproctitle HINT/*.py` → `train.py:17,22,246,249` only |
| `transformers` | `HINT/lavis/models/med.py:40-43` | vendored LAVIS | `4.25.0` (`README.md:129`) | YES | pure py | pin exactly | REACHABLE and VERSION-CRITICAL — see below |
| `torch` / `torchvision` | everywhere | — | `2.1.0` / `0.16.0` | YES | — | — | REACHABLE |

The whole `lavis` package is imported through the **audio** door before any CIR code runs:
`HINT/lavis/__init__.py:15` `from lavis.datasets.builders import *` →
`HINT/lavis/datasets/builders/__init__.py:93` (`audio_caption_builder`) and
`HINT/lavis/processors/__init__.py:29` (`BeatsAudioProcessor`) →
`HINT/lavis/processors/audio_processors.py:9,17`. So `torchaudio` and `moviepy` are on the evaluation
path **by way of the package's own `__init__`**, not by way of any audio feature HINT uses.

Observed venv versions:

```
transformers 4.57.6      torch 2.14.0+cpu       timm 1.0.30
torchaudio MISSING       moviepy MISSING        setproctitle MISSING      sklearn MISSING
```

Classification:

* `torchaudio` (all five sites) → **INSTALLABLE_DEPENDENCY_MISSING**. P1: it is the *first* failure of
  the evaluator's own model-loading import.
* `moviepy`, `pandas` → **INSTALLABLE_DEPENDENCY_MISSING**. P2.
* `transformers 4.57.6` vs README `4.25.0` → **DEPENDENCY_VERSION_CONFLICT**. `med.py:40-43` needs
  `apply_chunking_to_forward` from `transformers.modeling_utils`; observed `ImportError` on 4.57.6.
  P1.
* Unconditional audio coupling in `lavis/__init__.py:15` → **UPSTREAM_IMPORT_BUG**. P1.
* `setproctitle` → **not a blocker**: `HINT/train.py:17,246`. Training-only; `test.py` never touches
  it. P3.
* `torch==2.1.0` / Python 3.8.10 vs this host's 3.13 / `torch 2.14.0+cpu` →
  **FIRST_PARTY_ENVIRONMENT_CONFIGURATION** if anyone tries to run in the workbench venv. P2.
* GPU: `test.py:43` and `:60` call `.cuda()` unconditionally, `train.py:122` `model.cuda()` —
  **LEGACY_RUNTIME_INCOMPATIBILITY** on a CPU-only host, i.e. GPU is mandatory. P1; wraps into
  `GPU_RUNTIME_UNTESTED` at §10.

Env declaration fidelity: `models.yaml:369-376` (`python 3.8.10 / torch 2.1.0 / torchvision 0.16.0 /
cuda 12.6 / open-clip-torch 2.24.0 · scikit-learn 1.3.2 · transformers 4.25.0 · salesforce-lavis 1.0.2 ·
timm 0.9.16`) vs `README.md:120-129`. `confidence: VERIFIED_README` (`models.yaml:371`) is correct.
The README installs cu121 wheels (`README.md:126`) while the registry records CUDA `12.6`
(`models.yaml:375`) — cosmetic, P3.

---

## 5. CHECKPOINTS

Registry variant (`models.yaml:399-414`): `fashioniq` / `fashioniq.pt` /
`https://huggingface.co/iLearn-Lab/HINT/resolve/main/fashioniq.pt`, `source_type: author_huggingface`,
`checkpoint_training_noise_pct: null`, `evaluation_noise_pct: 0`,
`expected_size_bytes: 2724451182`,
`expected_sha256: bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4`,
`status: URL_VERIFIED`, `checkpoint_mapping_status: VERIFIED_METADATA`.

Observed live (STATIC_AUDIT on remote metadata; **no bytes downloaded**):

```
$ curl -s "https://huggingface.co/api/models/iLearn-Lab/HINT?blobs=true"
fashioniq.pt  2724451182  bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4
cirr.pt       2724451182  fca4657d94672571fa1dcb7615a1e8fc27a0651dd5e9ebc190943bb0acb25afd
$ curl -sIL https://huggingface.co/iLearn-Lab/HINT/resolve/main/fashioniq.pt
HTTP 200
```

Size and sha256 reproduce the registry **exactly**. Availability class **URL_VERIFIED**. No local
artifact: `ls workbench/artifacts/checkpoints/hint` → `No such file or directory`.

### 5.1 The FashionIQ checkpoint loader — located, quoted, and it does not exist

This was the assignment's first question and the answer is negative. I enumerated **every** `torch.load`
and every loader-ish symbol in the tracked first-party files:

```
$ grep -rn "torch.load\|load_checkpoint\|restore_file" HINT/*.py
HINT/train.py:59:parser.add_argument('--restore_file', default=None,
HINT/utils.py:128:def load_checkpoint(checkpoint, model, optimizer=None):
HINT/utils.py:139:    checkpoint = torch.load(checkpoint)
HINT/cirr_test_submission.py:115:            model = torch.load(os.path.join(model_dir, i))
```

Quoted, in full, the only loader that actually runs:

```python
# HINT/cirr_test_submission.py:112-117
model_dir = sys.argv[1]
file_ls = os.listdir(model_dir)
for i in file_ls:
    if ".pt" in i and f'CIRR_pred_ranks_recall{i[:-3]}.json' not in file_ls:
        model = torch.load(os.path.join(model_dir, i))
        print(i[:-3] + " start")
        test_cirr_submit_result(model, save_dir=model_dir, testset=testset, batch_size=64,
                                name=i[:-3], txt_processors=txt_processors)
```

**Which file it expects and where it looks:** it takes a **directory** as `sys.argv[1]`
(`cirr_test_submission.py:112`), scans it for any name containing `.pt` (`:114`), and `torch.load`s it
with **no `map_location`** (`:115`) — so a CUDA-saved module needs a GPU. Crucially, the script it then
calls is `test_cirr_submit_result` (`:117`, defined at `:11`), and its dataset is hard-coded to CIRR:

```python
# HINT/cirr_test_submission.py:108
testset = datasets.CIRR(path="./data/cirr_data/CIRR/", transform=preprocess)
```

Consequences, each traceable:

* **No pinned FashionIQ evaluation path loads `fashioniq.pt`.** `test.py:11`
  `def test(params, model, testset, category, txt_processors)` receives an already-built model and
  never names a path; `train.py:59-60`'s `--restore_file` is declared and **never referenced**
  (single grep hit, the declaration). Therefore the *only* code that can consume a `.pt` is the
  CIRR-only submission script.
* The Hub card confirms the intended command is that same CIRR script and names **only** `cirr.pt`:
  `Step 2: Download Model Weights` → `checkpoints/cirr.pt <-- (Rename to best_model.pt …)`;
  `Step 3` → `python src/cirr_test_submission.py checkpoints/`. `fashioniq.pt` is listed as existing
  ("`fashioniq.pt` (Trained on FashionIQ)") but **no command consumes it**, and the card's stated path
  `src/…` does not exist in either the pin or the card's own repository layout.
* **Serialization form.** `train.py:204,220,233,279` all call `torch.save(model, …)`, i.e. a whole
  pickled `HINT` object — not a `state_dict`. The shipped `utils.load_checkpoint` is inconsistent with
  that: `utils.py:139-140` `checkpoint = torch.load(checkpoint)` then
  `model.load_state_dict(checkpoint['state_dict'])`, and it has no caller. So the repository contains a
  loader that cannot read the repository's own output, and a training-save form that no loader reads.
* `map_location` absence at `cirr_test_submission.py:115` combined with a weights-only default in
  modern torch means a `state_dict` file would be fine but a whole-module file raises
  `UnpicklingError` unless `weights_only=False` is passed. I demonstrated the mechanism on a throwaway
  module under `/tmp` with torch 2.14.0 (observed `_pickle.UnpicklingError: Weights only load
  failed … Unsupported global: GLOBAL obj.M`). This is a **forward-compatibility** risk for whichever
  wrapper is written; it is not evidence about how the authors ran it (their torch 2.1.0 defaulted to
  `weights_only=False`). Marked **[INFERENCE]** where it concerns the authors' run; the torch-version
  fact itself is observed.

**Mapping confidence: metadata only.** `checkpoint_mapping_status: VERIFIED_METADATA`
(`models.yaml:414`) is the registry's own honest label: filename, size and hash are verified; the file
is not mapped to any benchmark row. `checkpoint_training_noise_pct: null` (`models.yaml:407`) is
correct — the paper states no condition and the Hub card states none; the registry's own comment at
`models.yaml:405-406` records that the 0 was inherited from `train.py:56`'s default flag, which — see
§9 D-HINT-1 — reaches no implementation in the pinned tree. Whether `fashioniq.pt` is clean is
**UNKNOWN**.

---

## 6. AUXILIARY ASSETS

| asset | purpose | source | availability | generation possible? |
|---|---|---|---|---|
| `resized_image/{dress,shirt,toptee}/*.jpg` | query + gallery images | operator-prepared from canonical FashionIQ JPEGs | **PRESENT locally, complete** (observed) | yes — `fashioniq_resize_images`, `preparation_contracts.yaml:33,95` |
| `captions/correction_dict_{dress,shirt,toptee}.json` | query-text normalisation | **`LIMN/correct_files/`, `TGCIR/correction_dict/` (pinned checkouts); bytes proven identical** | **PRESENT locally** | copy |
| `captions/cap.{cat}.{train,val}.json`, `image_splits/split.{cat}.val.json` | queries + gallery order | official FashionIQ release | PRESENT locally | n/a |
| `fashion_iq_data.json` + six `test_queries_*/test_targets_*.pkl` | in-checkout cache, required on the cache-hit branch | `datasets.py:36-47` | **ABSENT** | yes, from the train captions |
| BLIP-2 pretrained weights | backbone init | `blip2_pretrain.yaml:10`, Google URL | `AVAILABLE` (`cluster_b2_encoder_family.yaml:13-34`) | download |
| EVA-ViT-g weights | visual encoder | `eva_vit.py:443` | `AVAILABLE` (`cluster_b3_ilearn.yaml` airknow sibling) | download |
| `/root/bert-base-uncased` | tokenizer + Q-Former init | hard-coded at `blip2.py:32,48,55` | declared `AVAILABLE`; **not present on this host** | download to that exact absolute path |

Observed:

```
$ ls data/fashionIQ_dataset/
captions  image_splits  images  qwen_captions  resized_image
$ for c in dress shirt toptee; do ls data/fashionIQ_dataset/resized_image/$c | wc -l; done
3817   6346   5373
$ python -c "…every id in split.{cat}.val.json must exist as resized_image/{cat}/{id}.jpg…"
dress split 3817 missing jpg 0
shirt split 6346 missing jpg 0
toptee split 5373 missing jpg 0
$ ls data/fashionIQ_dataset/captions/
cap.{dress,shirt,toptee}.{train,val,test}.json   correction_dict_{dress,shirt,toptee}.json
```

Byte provenance of the dictionaries (this is the answer to §6's "needed, present, public?"):

```
$ sha256sum workbench/third_party/LIMN/correct_files/correction_dict_{dress,shirt,toptee}.json \
            workbench/third_party/TGCIR/correction_dict/correction_dict_dress.json \
            data/fashionIQ_dataset/captions/correction_dict_{dress,shirt,toptee}.json
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda   (dress, all three copies)
ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e   (shirt, all three copies)
3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71   (toptee, all three copies)
$ cmp LIMN/correct_files/correction_dict_dress.json data/fashionIQ_dataset/captions/correction_dict_dress.json
byte-identical
```

* **Needed?** Yes, unconditionally: `datasets.py:63` (train path) and `datasets.py:137` (val path)
  both `open(captions/correction_dict_{name}.json)`, and the dictionary changes the query string
  (`datasets.py:75-84`).
* **Present?** Yes, with worker-readable content (dress 363 entries, shirt 360, toptee 391 — observed).
* **Public?** **Not from this model's own repositories** — `find workbench/third_party/HINT -name
  "correction_dict*"` → empty; `find workbench/third_party/INTENT …` → empty. They are **present
  locally** because a *sibling* pinned checkout (LIMN, TGCIR) ships them under
  `LIMN/correct_files/` and `TGCIR/correction_dict/`, and the staged dataset bytes are provably those
  same files. So the honest classification is **AUXILIARY_ASSET_MISSING** for HINT's own upstream and
  **PRESENT-BY-SIBLING-PROVENANCE** locally — *not* "unavailable". Two standing documents still say
  otherwise and are stale:
  `workbench/docs/AUXILIARY_ASSET_AUDIT.md:358-360` lists the three `intent` dictionaries as
  `OFFICIAL_ASSET_UNAVAILABLE`, and
  `workbench/registry/assets/cluster_b2_encoder_family.yaml:57-77` records `hint_correction_dict` as
  `classification: SOURCE_EVIDENCE_MISSING`, `acquisition_status: UNVERIFIED`, with
  `current_blocker: "Not bundled in pinned HINT checkout; no official download URL or generation
  script provided in repository."` The last sentence is literally true for HINT; the classification
  omits that the bytes are present in the workbench and traceable to a pinned sibling.
* `workbench/registry/preparation_contracts.yaml` has **no `hint` entry**
  (`grep -n "hint" preparation_contracts.yaml` → zero hits; only clvc_net, dcnet, clip4cir, tgcir,
  sprc, limn, pair), so `manual_ilearn_resized` (`models.yaml:380`) is a label with no contract behind
  it. Contrast PAIR's contract at `preparation_contracts.yaml:302-339`, which declares resized images
  *and* explicitly declines to declare the dictionaries because no pinned source directory provided
  them — the same reasoning does **not** apply to HINT now that LIMN/TGCIR ship the files.
* `workbench/docs/READINESS_MATRIX.md:31` (generated 2026-10-14) still reports `hint` BLOCKED on
  "resized_image … correction_dict_dress.json"; those paths were staged on 2026-10-15 17:55 (observed
  mtime). The matrix row is **stale**.
* The cache artifacts are a real gap: `datasets.py:37` checks
  `os.path.exists(os.path.join(self.path, 'fashion_iq_data.json'))`; the cache-hit branch
  (`datasets.py:49-57`) loads `fashion_iq_data.json` plus six pickles, **none of which any layout
  helper declares** (`workbench/backend/fashioniq_layout.py:25-41` has no such path) and none of which
  exist locally. The cache-miss branch is the one that runs — and `cluster_b2_encoder_family.yaml:101-125`
  does declare them (`hint_generated_test_pkls`, `SOURCE_REGENERABLE`), so the registry knows about a
  dependency the layout helper never checks. P2.

---

## 7. FASHIONIQ PROTOCOL

Registry (`models.yaml:388-392`): `literature_split_label: val`,
`native_protocol: fashioniq_val_split`, `supported_protocols: [fashioniq_val_split]`; the protocol
itself is declared at `models.yaml:16-19` as *"FashionIQ - Val Split (Pair-Union Gallery, Reference
Excluded)"*, evaluation noise 0.

**What the source actually evaluates** (`datasets.py:25-32`, `:131-166`; `test.py:74-107`):

* **Split file names, quoted exactly** — `datasets.py:30` `self.split_dir = self.path + 'image_splits'`
  and `datasets.py:133`:
  `with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:`
  → concrete files `image_splits/split.dress.val.json`, `split.shirt.val.json`,
  `split.toptee.val.json`. Captions: `datasets.py:135` `cap.{name}.val.json`; dictionaries:
  `datasets.py:137` `correction_dict_{name}.json`.
* **Gallery — pair union, built from the split's own index space:**
  `datasets.py:146` `out['source_img_id'] = images.index(candidate)`;
  `datasets.py:148` `out['target_img_id'] = images.index(target)`;
  `datasets.py:154-159` per-query, first-seen order, `source` **then** `target`;
  `datasets.py:160-165` one gallery entry per collected id.
  There is **no `self.split` attribute at all** in this file (observed: `grep -n "self.split"
  datasets.py` → no match, exit 1), because HINT's `FashionIQ.__init__(self, path, transform=None)`
  (`datasets.py:25`) has no `split` parameter and no `original-split` branch. So HINT implements
  exactly one gallery: the pair union. This is Protocol C by construction.
* **Reference policy — masked, not removed:** `test.py:82-83`
  `for i, t in enumerate(test_queries): sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`,
  executed before `test.py:86` `nn_result = [np.argsort(-sims[i,:])[:50] …]`.
* **Recall:** `test.py:101-107` `for k in [1, 10, 50]`, hit iff
  `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, denominator
  `len(nn_result)`. Per category; no macro.
* **Categories:** `train.py:198` `for ci, category in enumerate(['dress','shirt','toptee'])`.

**Does it match the declared protocol?** Yes. `fashioniq_val_split` is declared at
`src/evaluation/fashioniq.py:178-180` as `gallery="pair_union", reference_excluded=True`, and the
workbench builder `build_pair_union_gallery` (`src/datasets/fashioniq.py:186-221`) appends reference
then target in annotation order with a `seen` set — the identical rule to `datasets.py:154-159`.
**VERIFIED** for the protocol id, gallery construction, ordering, reference policy and K-set.

Quantitative separation from the full-gallery protocol (observed, from the staged data):

```
dress : pair-union gallery=2628   full-split gallery=3817   queries=2017
shirt : pair-union gallery=3089   full-split gallery=6346   queries=2038
toptee: pair-union gallery=2902   full-split gallery=5373   queries=1961
```

so the `hint` row is genuinely a different cohort from Air-Know/ConeSep/HABIT/INTENT, and the
registry's protocol assignment is not the mirror-image mistake. Sibling code confirms the split: HABIT
`datasets.py:28,205-211`, ConeSep `datasets.py:28,201-207`, AirKnow `datasets.py:31,218-224` all
default or hard-code `'original-split'` with the `enumerate(images)` full-split loop, whereas HINT has
no such branch at all. `workbench/registry/models.yaml:121` (airknow →
`fashioniq_full_gallery_ref_excluded`) is therefore **consistent with the invoked code**; a claim that
AirKnow uses the pair-union gallery is not what the pinned source shows.

**Query construction.** `datasets.py:142` `mod_str = self.concat_text(caption['captions'], correction_dict)`
→ `datasets.py:75-80` (`correct_text`: lowercase → `string.punctuation`→space → token split → exact
dictionary substitution) → `datasets.py:82-84` (`"{} and {}"` join). Then `test.py:44` passes the
string through `txt_processors["eval"]`, the LAVIS `blip_caption` processor
(`lavis/configs/models/blip2/blip2_pretrain.yaml:35-36`) — two text stages, and the workbench models
only stage 1 (`src/datasets/fashioniq.py:93-117` `normalize_fashioniq_caption`). Parity of the second
stage is **UNKNOWN** (§10 B6).

**Metric.** `test.py:101-107`. The registry's `reported_mean: 68.21` (`models.yaml:395`) is a
workbench aggregate: `(58.48 + 77.93)/2 = 68.205 → 68.21` (observed arithmetic), matching
`src/evaluation/fashioniq.py:238-255` `"mean_recall": (recall_at_10 + recall_at_50)/2`. The paper
prints no such mean for HINT. Registry per-category values 58.48 / 77.93 are the paper's own.

---

## 8. WORKBENCH INTEGRATION

| element | state | evidence |
|---|---|---|
| adapter | `HintAdapter`, `script = "test.py"`, **no `command()` override** | `workbench/backend/adapters/models.py:44-46` |
| inherited behaviour | raises | `models.py:24-25` `raise NotImplementedError("upstream evaluation command requires audit before execution")` |
| observed | executed | `HintAdapter().command(None,None,None)` → `NotImplementedError: upstream evaluation command requires audit before execution`; `'command' in HintAdapter.__dict__` → `False` |
| planner | env group `cir-4c8b682368302099`, `SOLVER_COMPATIBLE` | `workbench/docs/READINESS_MATRIX.md:31` |
| checkpoint prep | none; no local artifact | `ls workbench/artifacts/checkpoints/hint` → `No such file or directory` |
| preflight | would report the layout paths; the row on disk is stale | `workbench/backend/fashioniq_layout.py:25-36`; matrix line 31 vs staging mtime 2026-10-15 |
| metric parser | none needed — the native path emits no structured output | `test.py:107` returns a list of tuples to `logging.info` |
| per-query support | not implemented | `test.py` never writes a file |
| run path | `run_eval.py:25` calls `build_command`, so `--model hint` cannot produce a command | `workbench/scripts/run_eval.py:14-25` |

---

## 9. CONFIRMED DEFECTS

**D-HINT-1 (P1) — the README's own FashionIQ training command crashes on its dataset constructor, and
so does the CIRR branch. REPRODUCED BY EXECUTION.**
`train.py:86-89` passes `noise_ratio=args.noise_ratio` to `datasets.FashionIQ`, whose signature is
`(self, path, transform=None)` (`datasets.py:25`). `train.py:95-100` does the same for
`datasets.CIRR`, whose signature is `(self, path, transform=None, case_look=False)` (`datasets.py:170`).
Executed:

```
$ PYTHONPATH=<HINT> .venv/bin/python -c "import datasets; datasets.FashionIQ(path='/x/', transform=None, noise_ratio=0.2)"
TypeError: FashionIQ.__init__() got an unexpected keyword argument 'noise_ratio'
$ ... datasets.CIRR(path='/x/', transform=None, case_look=False, noise_ratio=0.2)
TypeError: CIRR.__init__() got an unexpected keyword argument 'noise_ratio'
```

So the README's `python train.py --dataset fashioniq …` (`README.md:201-207`) and
`--dataset cirr` (`README.md:213-219`) both die at `train.py:89` / `:99` before any data is read.
The `--noise_ratio` flag (`train.py:56`) reaches **no implementation anywhere in the pinned tree** —
there is no `shuffle()` in HINT's `datasets.py` at all (`grep -n "def shuffle" HINT/datasets.py` →
none, unlike INTENT/HABIT/ConeSep which all have one). Classification: **UPSTREAM_IMPORT_BUG**
(first-party internal contract mismatch). Severity P1 because the upstream tree ships no runnable
training-or-eval command. Fix: none from the workbench; the wrapper must bypass `load_dataset()`.
Regression: n/a (no code change).

**D-HINT-2 (P1) — no FashionIQ checkpoint loader exists for the registered checkpoint.**
Detailed in §5.1. `test.py:11` receives a model and never names a path; `train.py:59-60`
`--restore_file` is declared and unused (one grep hit); the only `torch.load` is
`cirr_test_submission.py:115` inside a CIRR-only script (`:108`). Classification:
**PUBLIC_CHECKPOINT_NOT_MAPPED**. Fix: the wrapper owns construction + load; register it as an
approved, digest-pinned first-party file (CSMCIR precedent, `models.yaml:88-95`).

**D-HINT-3 (P1) — the pinned environment cannot import the model package.** Executed traceback in §4:
`torchaudio` missing (`audio_processors.py:9`, reached from `lavis/__init__.py:15` →
`processors/__init__.py:29`), then `transformers 4.57.6` lacking `apply_chunking_to_forward`
(`med.py:40-43`). Classification: **INSTALLABLE_DEPENDENCY_MISSING** + **DEPENDENCY_VERSION_CONFLICT**
+ **UPSTREAM_IMPORT_BUG**. NOTE the assignment's framing: `torchaudio` is **not** training-only here —
it is on the import chain of `load_model_and_preprocess`, and both `train.py` and the CIRR submission
import that. `setproctitle` **is** training-only (`train.py:17,246`; never in `test.py`) and is
therefore **not** a blocker. Fix: install `torchaudio==2.1.0`, `moviepy`, and pin `transformers==4.25.0`
in `WORKBENCH_HINT_PYTHON`.

**D-HINT-4 (P2) — the shipped loader cannot read the shipped checkpoints.** `utils.py:139-140`
`checkpoint = torch.load(checkpoint)` → `model.load_state_dict(checkpoint['state_dict'])`, while
`train.py:204,220,233,279` writes whole modules via `torch.save(model, …)`. Classification:
**UPSTREAM_IMPORT_BUG**. Fix: wrapper uses `weights_only=False` whole-module load; do not repair
`utils.py`.

**D-HINT-5 (P3, REFUTES A PRIOR AUDIT'S LINE CITATIONS) — the earlier `hint.md` line numbers for the
constructor mismatch are wrong, though the finding itself is right.**
`workbench/docs/reviews/models/hint.md:112` calls `--noise_ratio`'s default "`train.py:56`", and
`:316` says `datasets.py:25` "fixes the three directories by concatenation" and that `train.py:89`
passes the rejected kwarg. Observed line numbers:
`train.py:56` is the `--noise_ratio` declaration (correct); `datasets.py:25` is
`def __init__(self, path, transform=None):` (correct as the signature line); the directory
concatenation is at `datasets.py:28-31`, not `:25`. The earlier document's substantive conclusion
(D-HINT-1) is **UPHELD**; only the offset is corrected here. Also, that document cites
`test.py:74-76` and `:86,100` — observed values are `:74-76` (matmul/squeeze/max) and `:86,:100`
(argsort) — those match. Recording the offset because every claim in this audit is line-cited.

**D-HINT-6 (P3, UPHELD AGAINST REFUTATION) — the similarity step is well-formed; no shape defect.**
`HINT/lavis/models/blip2_models/HINT.py:433` `return fusion_feats.unsqueeze(1).unsqueeze(1)` on a
`(B,256)` tensor gives `(B,1,1,256)`; `:455` `return image_features.permute(0, 2, 1)` on `(B,32,256)`
gives `(B,256,32)`. Executed with torch:

```
fusion = torch.randn(4,256); q = fusion.unsqueeze(1).unsqueeze(1)  -> torch.Size([4, 1, 1, 256])
np.matmul(q.numpy(), g.numpy())  -> (4, 5, 1, 32) -> squeeze -> (4, 5, 32) -> max(-1) -> (4, 5)
sims[0,1] = 29.401709      # test.py:82's indexing is well-formed
```

The registry's own comment for `intent` (`models.yaml:306`) records the identical retraction for the
sibling model. **UPHELD**: `test.py:74-76` produces exactly `(queries, gallery)`.

**D-HINT-7 (P2) — the registry's `notes` field overstates usability.**
`models.yaml:398` "Official iLearn-Lab Hugging Face repository exposes FashionIQ checkpoint." True
about existence; it implies a consumable path that does not exist (D-HINT-2). The Hub card itself
names only `cirr.pt` in its usage steps. Classification: **FIRST_PARTY_ENVIRONMENT_CONFIGURATION**
(registry documentation). Fix: extend the note to state that no pinned FashionIQ loader consumes
`fashioniq.pt`.

**D-HINT-8 (P2) — declaration gaps around a model whose assets are now present.**
No `hint` entry in `workbench/registry/preparation_contracts.yaml` (grep → 0 hits);
`cluster_b2_encoder_family.yaml:66,75` still classify the dictionaries `SOURCE_EVIDENCE_MISSING` /
`UNVERIFIED` though their bytes are in the workbench and hash-identical to two pinned checkouts;
`workbench/docs/READINESS_MATRIX.md:31` is stale; the cache artifacts the evaluator reads
(`datasets.py:36-47`) are declared in an asset record (`cluster_b2_encoder_family.yaml:101-125`) but
absent from `workbench/backend/fashioniq_layout.py`. Classifications:
**FIRST_PARTY_ENVIRONMENT_CONFIGURATION** (declaration) and **PUBLIC_ASSET_NOT_PREPARED** (stale
matrix). Not run-blockers.

---

## 10. REMAINING BLOCKERS

| # | classification | owner | evidence | remediation |
|---|---|---|---|---|
| B1 | **FIRST_PARTY_ADAPTER_MISSING** + **UPSTREAM_CLI_MISSING** | workbench + upstream | `models.py:24-25,44-46`; `test.py` no `__main__`; observed `NotImplementedError` | approved observation-only wrapper (§11) |
| B2 | **PUBLIC_CHECKPOINT_NOT_MAPPED** | upstream | §5.1: `test.py:11` takes a model; `train.py:59-60` unused; only `cirr_test_submission.py:115` loads, CIRR-only | wrapper owns load; register it |
| B3 | **INSTALLABLE_DEPENDENCY_MISSING** / **DEPENDENCY_VERSION_CONFLICT** / **UPSTREAM_IMPORT_BUG** | workbench env + upstream | executed tracebacks; `audio_processors.py:9-13`; `med.py:40-43`; `lavis/__init__.py:15` | build declared env (3.8.10 / torch 2.1.0 / torchaudio 2.1.0 / transformers 4.25.0 / moviepy) |
| B4 | **AUXILIARY_ASSET_MISSING** (upstream) — *no longer a live blocker locally* | upstream | dicts present, sha256 `9e4a5987…/ca29e29e…/3dfbf15f…`, byte-identical to `LIMN/correct_files/` + `TGCIR/correction_dict/`; `find HINT -name correction_dict*` → empty | none for a run; declaration gap = B5 |
| B5 | **FIRST_PARTY_ENVIRONMENT_CONFIGURATION** (declaration) | workbench | no `hint` entry in `preparation_contracts.yaml`; stale asset classification; stale matrix | add `hint_fashioniq` contract + digests |
| B6 | **SCIENTIFIC_PROTOCOL_UNVERIFIED** (one stage) | workbench | `test.py:44` applies the LAVIS `blip_caption` processor after `datasets.py:82-84`; the workbench models only stage 1 (`src/datasets/fashioniq.py:93-117`) | confirm the exact token strings the processor yields for both stages before any numeric claim |
| B7 | **GPU_RUNTIME_UNTESTED** / **ACTUALLY_READY_PENDING_GPU** | host | `test.py:43,60` `.cuda()` unconditional; `train.py:122` `model.cuda()`; `/root/bert-base-uncased` absent | run on the approved GPU host |
| B8 | **NONDETERMINISTIC_PREPROCESSING** — checked, absent | — | `datasets.py:103-122` transform applied deterministically in `get_test_img`; `test.py:12` `model.eval()` | none; recorded as clean |

No `EXTERNAL_BLOCKER` token is used anywhere in this report. Also worth recording: the earlier
`workbench/docs/reviews/FINAL_16_MODEL_AUDIT_MATRIX.md:36` uses the forbidden token and lumps
`setproctitle` (training-only, P3) with `torchaudio` (import-path, P1); that row's blocker list is
imprecise on the one distinction this assignment asks about.

---

## 11. IMPLEMENTATION PLAN

1. **Environment** (B3). Build `WORKBENCH_HINT_PYTHON` from `README.md:120-129` verbatim: Python
   3.8.10, `torch==2.1.0`, `torchvision==0.16.0`, `torchaudio==2.1.0` (cu121),
   `transformers==4.25.0`, `scikit-learn==1.3.2`, `open-clip-torch==2.24.0`, `timm==0.9.16`, plus
   `moviepy` (reachable, absent from the README list). Record `interpreter_sha256` (`models.yaml:1-5`).
   Complexity low; scientific risk none — it is the declared env. Expected: `from lavis.models import
   load_model_and_preprocess` succeeds.
2. **Prepare `/root/bert-base-uncased`** (B7). Mandatory: hard-coded at `blip2.py:32,48,55`.
   Complexity low; risk none.
3. **Wrapper** (B1, B2). One digest-pinned, registered first-party script that:
   `load_model_and_preprocess("HINT","pretrain",is_eval=True,device=...)` →
   `torch.load(<fashioniq.pt>, map_location="cpu", weights_only=False)` (whole pickled `HINT` object
   per `train.py:204,220,233,279`) → `datasets.FashionIQ(path=<root>/ , transform=targetpad(1.25,224))`
   → `test.test(args, model, testset, cat, txt_processors)` for `['dress','shirt','toptee']` → macro.
   **The wrapper must not touch gallery construction, the reference mask, the caption composition or
   the K-set** — those are the model's identity. Complexity medium; risk: any deviation changes the
   cohort. Expected: three per-category `{R@1,R@10,R@50}` sets.
4. **Declaration** (B5). Add `hint_fashioniq` to `preparation_contracts.yaml` naming
   `resized_image/{cat}` (generator `fashioniq_resize_images`) and the correction dictionaries with
   `source_directory` pointing at `LIMN/correct_files/` (or `TGCIR/correction_dict/`) plus their three
   sha256 values; update `cluster_b2_encoder_family.yaml:57-77`'s classification and regenerate
   `READINESS_MATRIX.md`.
5. **Do not** patch `datasets.py` or `train.py` to "fix" `noise_ratio` (D-HINT-1); the wrapper never
   calls them.
6. **Record** that the Hub card's Step 3 command (`python src/cirr_test_submission.py checkpoints/`)
   cannot run as printed: `src/` does not exist and no FashionIQ path consumes `fashioniq.pt`.
7. **Scientific invariants preserved.** Protocol C (`fashioniq_val_split`: pair-union gallery,
   reference excluded, 0 % evaluation noise) is the model's declared and actual protocol and nothing in
   this plan changes the gallery, the reference policy, the caption input or the noise condition.

---

## 12. FINAL VERDICT

**BLOCKED**, and protocol-correct.

* Pin exact; current upstream **equals** the pin (SNAPSHOT 1 == SNAPSHOT 2, one-commit repo).
* Protocol A/B/C: `hint` is Protocol C by construction — `datasets.py:146-165` pair-union,
  `test.py:82-83` reference excluded, and **no `self.split` attribute exists** in the checkout, so there
  is no full-gallery branch to mistake. **VERIFIED**.
* `fashioniq.pt` exists, is reachable (HTTP 200) and its size + sha256 match the registry; **no pinned
  FashionIQ loader consumes it** — the only `torch.load` in the tree belongs to a CIRR-only submission
  script. The registry's `VERIFIED_METADATA` label is honest; its `notes` field is not.
* `torchaudio` **is** a real blocker here (import path, not training-only); `setproctitle` is **not**.
* Two standing documents (asset audit, readiness matrix) are stale about assets that are present.
* Nothing was reproduced; no metric was observed. `reproduction_status: NOT_RUN` (`models.yaml:396`)
  remains correct.

---

## 13. EVIDENCE

* repo: `https://github.com/iLearn-Lab/ICASSP26-HINT` (`models.yaml:383`).
* pinned commit: `bec50b6c8c19111893b617979502b948d1cea5b2` — `git -C workbench/third_party/HINT rev-parse HEAD`.
* SNAPSHOT 2: `git -C workbench/third_party/HINT ls-remote origin` → HEAD == pin;
  `git log --oneline --all` → one commit `bec50b6 Create LICENSE`; `rev-list --count HEAD` → 1.
* key files and digests observed:
  `HINT/datasets.py` sha256 `e26955dff720b4209135d6998690db79f1793a2029fdd4ed9efd28daa80fcd60`;
  `HINT/lavis/models/blip2_models/HINT.py` sha256 `796464a0a37ea21d9efc61ddd773636e33bbb879445fe22df95adce4de6cd237`.
* protocol: `datasets.py:25,30,131-166`; `test.py:11-19,38-61,74-76,79-86,100-107`;
  `models.yaml:16-19,368-414`; workbench `src/evaluation/fashioniq.py:35-73,132-136,171-183,238-255`,
  `src/datasets/fashioniq.py:93-117,186-221`.
* loader: `HINT/cirr_test_submission.py:108,112,114,115,117`; `HINT/test.py:11`;
  `HINT/train.py:59-60,204,220,233,279`; `HINT/utils.py:128-140`.
* checkpoint: `curl -s "https://huggingface.co/api/models/iLearn-Lab/HINT?blobs=true"`;
  `curl -sIL https://huggingface.co/iLearn-Lab/HINT/resolve/main/fashioniq.pt` → `200`.
* dependencies: executed `ModuleNotFoundError: No module named 'torchaudio'` and, after stubbing,
  `ImportError: cannot import name 'apply_chunking_to_forward'`; venv `transformers 4.57.6`,
  `torch 2.14.0+cpu`; `setproctitle moviepy sklearn torchaudio` all missing.
* constructor mismatch: executed `TypeError: FashionIQ.__init__() got an unexpected keyword argument
  'noise_ratio'` and the same for `CIRR`.
* shape: executed torch/numpy shapes `(4,1,1,256) x (4,256,32) -> (4,5)`.
* adapter: executed `HintAdapter().command(None,None,None)` → `NotImplementedError:
  upstream evaluation command requires audit before execution`.
* data: `ls`/`wc -l`/`sha256sum`/`cmp` over `data/fashionIQ_dataset/`,
  `workbench/third_party/LIMN/correct_files/`, `workbench/third_party/TGCIR/correction_dict/`.
* no checkpoint bytes were downloaded; `workbench/artifacts/checkpoints/hint` does not exist.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 — static source/protocol/asset audit with executed import, constructor and shape probes; no
environment was built and no inference was run.**

* Static reading of the pinned tree: VERIFIED.
* Pin identity and SNAPSHOT-2 comparison: VERIFIED (read-only `ls-remote`).
* Protocol/gallery/reference/K-set traceability (Protocol C): VERIFIED.
* Constructor mismatch reproduced by execution: VERIFIED.
* Import-path failures reproduced by execution: VERIFIED (two stages).
* Shape sanity reproduced by execution: VERIFIED.
* Checkpoint metadata: VERIFIED as metadata; payload identity and training condition UNKNOWN.
* Second text stage (LAVIS `blip_caption` processor) parity: UNKNOWN.
* Environment build, weight load, GPU inference, metric reproduction: **NOT RUN**.
* No claim above is RUNTIME_VERIFIED.
