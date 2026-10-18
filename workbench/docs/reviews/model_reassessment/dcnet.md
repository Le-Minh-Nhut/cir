# dcnet — reproducibility reassessment

Independent adversarial reassessment of registered model `dcnet` (DCNet).
READ-ONLY: nothing under `/home/heheboiz/data/cir` was modified except this file. All probe output
went to `/tmp` (`/tmp/dcnet_author_config.json`, `/tmp/dcnet_drive_listing.html`). No GPU inference,
no weight download, **no checkpoint download**, no install. Every claim is **STATIC_AUDIT** unless it
carries an observed command+output; this report contains **zero RUNTIME_VERIFIED** claims.

Existing workbench audits are treated as hypotheses; I re-derived the ones I cite and disagree where
the code says otherwise.

- Workbench HEAD at audit time: `b70d092e9d2232e34886f819e30c7327b74f3a76`.
- Pinned checkout: `workbench/third_party/DCNet` at `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `dcnet` | `workbench/registry/models.yaml:606` |
| method | DCNet — "Dual Compositional Learning in Interactive Image Retrieval" (AAAI 2021; FashionIQ 2020 winning solution) | `models.yaml:612,615`; `third_party/DCNet/README.md:6-8` |
| year | 2021 | `models.yaml:614` |
| paper URL | `https://github.com/ozmig77/dcnet/blob/68c79d.../DCNet_Kim2021.pdf` (author-hosted camera-ready, 1,764,944 bytes in the checkout) | `models.yaml:616`; OBSERVED `ls third_party/DCNet/DCNet_Kim2021.pdf` |
| source URL | `https://github.com/ozmig77/dcnet` | `models.yaml:617` |
| pinned commit | `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` | `models.yaml:619` |
| checkpoint variant | `fashioniq_run_directory` / `filename: fashioniq_dcnet` / `artifact_type: run_directory` / `required_files: [config.json, trained_model.pth]` | `models.yaml:647-650` |
| reported metrics | `reported_mean: 40.83` only; `reported_r10`/`reported_r50` deliberately absent | `models.yaml:629-633` |
| native protocol | `fashioniq_full_gallery_ref_excluded` | `models.yaml:627`; `supported_protocols: [fashioniq_full_gallery_ref_excluded]` `:628` |
| replay | `replay_status: REPLAY_CODE_IMPLEMENTED`, `replay_type: OFFICIAL_CLI`, `COMMAND_AUDITED` | `models.yaml:607,610,635` |
| reproduction_status | `BLOCKED_CHECKPOINT_MAPPING` | `models.yaml:634` |
| environment | python 3.7.7 / torch 1.4.0 / torchvision 0.5.0 / **cuda UNKNOWN** / `[nltk==3.5, spacy==2.3.0]` | `models.yaml:639-643` |

STATIC_AUDIT.

---

## 2. SOURCE STATUS

- OBSERVED `git -C workbench/third_party/DCNet rev-parse HEAD` →
  `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` = `upstream_commit_sha` (`models.yaml:619`). Pin exact.
- OBSERVED `git rev-list --all --count` → `1`; single commit `68c79d3 Readme typo`. No history claims.
- OBSERVED `git status --porcelain` → clean (no output).
- Tree (OBSERVED): `README.md`, `requirement.txt`, `DCNet_Kim2021.pdf`, `test.py`, `train.py`,
  `parse_config.py`, `base/`, `model/` (`model,composition,fusion,loss,metric,modules,fishnet`),
  `data_loader/` (`CE_dataset.py`, `data_loaders.py`), `trainer/` (`trainer_joint.py`,
  `trainer_ref.py`), `utils/` (`util,vocab,text2vec`), `logger/`, `configs/ce/*.json`,
  `preprocess/` (`process_cap.py`, `gen_shoe_cap.py`, `resize_img.py`, `download_fashioniq.py`),
  `experiments/*.ipynb`, `.gitignore`.
- Complete? Yes for the evaluation path. Every module imported by `test.py` exists.
- **`inspect` trap for a wrapper:** `model/model.py:6` imports `torchvision.models as models`, and
  `base/base_trainer.py:63-68` renders the whole config through `pprint`; nothing on the eval path
  needs a module that is missing.

SNAPSHOT 2 — current upstream, OBSERVED this pass from the pinned checkout (pin untouched):

```
$ git -C workbench/third_party/DCNet ls-remote origin HEAD
68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb	HEAD
$ git -C workbench/third_party/DCNet fetch origin --dry-run     # rc=0, no output
$ git -C workbench/third_party/DCNet rev-parse HEAD
68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb
```

Current upstream HEAD **equals the pin**: no CLI was added, no dependency fixed, no new checkpoint
published, the repo did not move. `git branch -a` → `remotes/origin/main` only.

---

## 3. NATIVE EVALUATION WORKFLOW

- Entrypoint: `test.py` (`test.py:96-117`). CLI: `--resume` (`:98`), `--device` (`:99`),
  `--seeds` (`:100`), `--mode` (`:101`, **default `"val"`**). README:62 documents
  `python test.py --resume logdir/fashioniq_dcnet/`.
- `test.py:103` `ConfigParser(args, 'eval')`; `:114-115` asserts `--resume` non-empty; `:117` calls
  `test(test_config)`.
- Chain: `test(config)` (`test.py:19`) → `compute_dims(config)` (`utils/util.py:64-108`) →
  `config.init('data_loader', data_loader.data_loaders, ...)` (`test.py:36-42`, cwd-relative
  `data_dir`) → `config.init('arch', model.model, ...)` (`:44-51`) → `TrainerJoint(...)` (`:52-59`,
  whose `BaseTrainer.__init__` calls `_resume_checkpoint`, `base/base_trainer.py:51-54`) →
  `trainer._valid_epoch(save_textatt=True)` (`test.py:64`) → per-category loop
  `trainer/trainer_joint.py:144-252` → prints via the logger (`test.py:66-84`) or saves a score file
  (`test.py:85-90`).
- CLI has **no** `--dataset-root`/`--data_dir`: the dataset path comes only from the config (§4 of
  the assignment, answered in §5 below).
- Adapter: `DCNetAdapter.command` → `[python, <source>/test.py, "--resume", <checkpoint-dir>]`
  (`workbench/backend/adapters/models.py:79-84`), run with cwd = the checkout root
  (`evaluate_models.py:214-215`).
- Missing components: the run directory (§5), the GloVe val PKLs and `resized_images/` (§6),
  ResNet-50 V1 weights (§6), an environment (§10), and an aggregate metric parser
  (`metrics_extraction.py:413-422`).

---

## 4. DEPENDENCIES AND LEGACY RUNTIME INCOMPATIBILITY

Local runtime for the probes below, OBSERVED: CPython 3.13.12, torch **2.14.0+cpu**, torchvision
**0.29.0+cpu**, Pillow 12.3.0, numpy 2.5.2.

| # | call | file:line | eval-reachable? | status on a modern runtime | evidence |
|---|---|---|---|---|---|
| D1 | `torch.load(ckpt_path)` bare | `base/base_trainer.py:195` | **YES** | deprecated-but-working; modern default is `weights_only` unless the payload demands otherwise | OBSERVED `torch.load(<whole module>)` → `UnpicklingError: Weights only load failed`; OBSERVED a DCNet-*shaped* dict payload (`arch/epoch/state_dict/monitor_best/config/optimizer`) loads **fine** under the default → no change needed |
| D2 | `models.resnet50(pretrained=True)` | `model/model.py:283` | **YES** | still accepted, emits a `UserWarning` | OBSERVED wrapper text at `torchvision/models/_utils.py:207`; OBSERVED `resnet50(pretrained=False)` builds |
| D3 | `Image.ANTIALIAS` | `preprocess/resize_img.py:11` | preparation only | **REMOVED** in Pillow 10+ | OBSERVED Pillow 12.3.0 `hasattr(Image,'ANTIALIAS')` → `False` |
| D4 | `Variable(...)` + unconditional `.cuda()` | `model/loss.py:4,62,87,89,106-107` | **NO** (`loss.py` is imported only by `train.py:11`; `test.py` builds `TrainerJoint(loss=None)`, `test.py:54`) | `Variable` works; `.cuda()` would fail on a CPU host | OBSERVED `from torch.autograd import Variable` OK; OBSERVED `Variable(t)` returns a Tensor sharing storage |
| D5 | `inp.data.masked_fill_(mask.data.bool(), ...)` | `model/modules.py:173,175` | **YES** | still works | OBSERVED `OK .data.masked_fill_ -inf` and `OK nan mask tensor([0., 1.])` |
| D6 | `einops.rearrange` | `model/modules.py:4` | import-time | fine; declared in `requirement.txt` unbounded | OBSERVED `einops` importable |
| D7 | `pack_padded_sequence(..., enforce_sorted=False)` | `model/modules.py` (`SharedGRU`) | not the configured pooling (`sharedconv`) | fine | `configs/ce/fashioniq_dcnet.json:44` `"text_pooling_module": "sharedconv"` |
| D8 | `__pycache__` cleanup none | — | — | a stale `test_clvc_replay` bytecode file exists under `workbench/tests/__pycache__` with no source; unrelated to DCNet, noted in §9 | OBSERVED |

**DCNet is markedly cleaner than the CLVC-Net class.** There is no removed torchvision submodule
import, no `torchvision.models.utils`, no `transforms.functional`, no Python-2-ism. The only
*hard* removal found is `Image.ANTIALIAS` (D3) and it is on the **preparation** path only — i.e. it
blocks regenerating `resized_images/`, not the evaluator.

Dependency categories (D1–D8 in the *task's* sense): the task's taxonomy is not defined anywhere in
this repository — OBSERVED it matches only unrelated text in `workbench/docs`; a prior reassessment
recorded the same (`model_reassessment/sprc.md:169,173`). I therefore classify with the workbench's
own tokens in §10 and do not invent D-numbers.

---

## 5. CHECKPOINT-ROOT COUPLING — `config.json` (assignment question 1 & 4)

### The quoted config

`workbench/third_party/DCNet/configs/ce/fashioniq_dcnet.json:17-33`:

```json
"data_loader": {
    "type": "ExpertDataLoader",
    "args": {
        "dataset_name": "CE",
        "data_dir": "dataset/fashioniq",
        "categories": ["dress", "shirt", "toptee"],
        "batch_size": 32,
        "num_workers": 6,
        "max_text_words": 30,
        "max_expert_tokens": 45,
        "mode": "train",
        "use_val": false
    }
}
```

`data_dir` is a **relative path with no leading marker**; every consumer concatenates it onto the
process's current working directory (`os.path.join`/f-string concatenation, never `Path.resolve()`).

### The code that reads it

1. `parse_config.py:18-28` — the run directory *is* the config source:

```python
if args.resume:
    self.resume = Path(args.resume)
    self.cfg_fname = self.resume / 'config.json'
...
self._config = read_json(self.cfg_fname)
```

2. `parse_config.py:39-44` — evaluation mode redirects the save/log dir into `<resume>/val`
   and rewrites the config there:

```python
else:
    if args.mode == 'val':
        save_dir = Path(self.config['trainer']['save_dir'])
        self._save_dir = self.resume / "val"
        self._log_dir = self.resume / "val"
        self._config['data_loader']['args']['mode'] = 'val'
```

3. `parse_config.py:57-60` — the side effect (`write_json(self.config, self._save_dir / 'config.json')`).
4. `base/base_trainer.py:193-206` — the weights member:

```python
ckpt_path = self.config.resume / 'trained_model.pth'
checkpoint = torch.load(ckpt_path)
...
self.model.load_state_dict(checkpoint['state_dict'])
```

5. `data_loader/CE_dataset.py:22,27,34,73` — the three relative reads:
   `os.path.join(data_dir, "resized_images")`, `read_json(Path(data_dir + "/image_splits/split.<cat>.val.json"))`,
   `memcache(data_dir + "/captions/cap.<cat>.glove.<split>.pkl")`, and
   `Image.open(os.path.join(self.image_dir, candidate_fname))` with `candidate_fname = candidate_id + '.jpg'` (`:72-73`).

### What the run directory must contain

**Minimum, enforced by code:** exactly the registry's `required_files` — `<dir>/config.json`
(`parse_config.py:20`) and `<dir>/trained_model.pth` (`base/base_trainer.py:193`), the latter a
`torch.save` dict carrying at least `state_dict` (`:206`), plus `epoch`/`monitor_best` (`:196-197`).
`best_score.pt`, `checkpoint-epoch*.pth`, `score-epoch*.pt` are never opened. **VERIFIED** — the
registry's declaration (`models.yaml:650`) is exact.

**Also on disk, but not accepted as members by the workbench:** the evaluator *writes* into the run
directory on every run — `<dir>/val/` plus a rewritten `val/config.json` (`parse_config.py:41-42,57-60`)
and `<dir>/val/att_result.pt` (`trainer/trainer_joint.py:250-252`, because `test.py:64` passes
`save_textatt=True` unconditionally). So `--resume` is **not** read-only on the author directory.

### Which run directory does the evaluator actually read? (assignment question 4)

Whatever `--resume` names, and only that: `parse_config.py:19-20` → `<resume>/config.json`, a
directory, not a file; and `base/base_trainer.py:193` → `<resume>/trained_model.pth`.
The adapter passes `checkpoint_path(model_id, checkpoint)` = `$WORKBENCH_CHECKPOINT_ROOT/dcnet/fashioniq_dcnet`
(`registry.py:842-863`, `filename: fashioniq_dcnet`, `artifact_type: run_directory`). So the
evaluated directory is
`workbench/artifacts/checkpoints/dcnet/fashioniq_dcnet/` — **OBSERVED absent**:
`checkpoint missing: .../workbench/artifacts/checkpoints/dcnet/fashioniq_dcnet` in the dry-run below.

### `data_dir` resolution — the coupling that breaks first

`data_dir: "dataset/fashioniq"` is resolved against the **process cwd**, which the workbench sets to
the checkout root (`evaluate_models.py:214-215`). So the evaluator will look for
`workbench/third_party/DCNet/dataset/fashioniq/{resized_images,image_splits,captions}`, while the
workbench's preparation contract stages those artifacts under `FASHIONIQ_ROOT`
(= `data/fashionIQ_dataset`, OBSERVED via `resolve_config()`). No court maps between them:
- `registry.py:729-740` builds the `canonical_dataset_link` runtime artifact **only when the model
  declares one**, and OBSERVED only four models do — csmcir (`models.yaml:37`), combiner (`:665`),
  clip4cir (`:726`), sprc (`:848`). **dcnet declares none.**
- `parse_config.py` also honours the *author's own* `config.json` from the Drive folder (OBSERVED
  below: same relative `data_dir`), so the coupling cannot be repaired by the shipped config either.

OBSERVED, this pass, that the evaluator's shipped config and the author's released config are
**semantically identical** (so the Drive `config.json` is not a differing variant):

```
$ diff <(jq -S . /tmp/dcnet_author_config.json) <(jq -S . workbench/third_party/DCNet/configs/ce/fashioniq_dcnet.json)
IDENTICAL (semantically)
sha256 pinned  : fc27267793c6958d8526b22a9efadb24b9e68858f2a1ae2774191259ca5968fc
sha256 author  : 80b33d43f6f49c2eeb8ddf948a205f33a78a4c19cdf25975af033ad3be1f320a  (whitespace differs only)
```

This is a **new fact** this pass: the author Drive folder serves a `config.json` whose `arch.args`,
`data_loader.args` and `trainer` blocks equal the pinned `configs/ce/fashioniq_dcnet.json`
(`backbone: resnet`, `backbone_pretrain: imagenet`, `composition: tirg`, `correction: fd`,
`fusion: hadamard`, categories dress/shirt/toptee, `max_text_words: 30`). So the previously
UNKNOWN question "was the released `trained_model.pth` trained under a different config?" resolves to
**no difference between the two documents**, and the `arch`-mismatch warning at
`base/base_trainer.py:200-204` will not fire. It does **not** prove the weights were produced by that
exact run — only that the config bytes agree.

---

## 6. GloVe / OOV PREPROCESSING — DEFINITIVE ANSWER (assignment question 2)

### The quoted code

`preprocess/process_cap.py:25-36`:

```python
vocab_words, no_word = {}, []
for word, cnt in counter.items():
    if glove.has_vector(word):
        vocab_words[word] = glove.get_vector(word)
    elif cnt > 2:
        no_word.append(word)
        vocab_words[word] = np.random.normal(0, 0.3, (300,))
print(len(vocab_words), len(no_word))

# Save vocab
with open('../dataset/fashioniq/captions/glove_vecs.pkl','wb') as f:
    pickle.dump(vocab_words, f)
```

`preprocess/process_cap.py:50-61` (the consumed path):

```python
for caption in captions:
    caption = caption.replace('-', ' ').replace('.','')
    toks = nltk.tokenize.word_tokenize(caption.lower())
    for word in toks:
        ### ------ Drop UNK words ------------------
        if word in vocab_words:
            w2v.append(vocab_words[word])
w2v = np.stack(w2v)
data['wv'] = w2v
```

### The definitive answer

1. **Is the OOV embedding random? YES.** `process_cap.py:31`
   `vocab_words[word] = np.random.normal(0, 0.3, (300,))` — a fresh 300-d Gaussian draw per OOV word
   whose train-corpus count exceeds 2.
2. **Is it seeded? NO — definitively.** OBSERVED `grep -rn "seed" preprocess/` returns two lines and
   neither is a seeding call; the file has no `np.random.seed`, no `random.seed`, no
   `torch.manual_seed`. OBSERVED the only seeding in the whole repository is `test.py:105-112`
   (`np.random.seed(0)` **at line 111**) and `train.py:25-29` — both run *after* the PKLs are
   produced, and neither affects a separate `python process_cap.py` process.
3. **Is the resulting metric reproducible across machines? NO.** Two runs of `process_cap.py` on the
   same machine, and any two machines, produce different rows for the same word; `wv` for that
   annotation therefore differs; the text features differ; the composed query feature differs; the
   ranking can change; the recall numbers can change.
4. **Is the OOV path even on the evaluator's critical path? YES, unavoidably.** The PKL is the *only*
   text source: `data_loader/CE_dataset.py:33-35` loads
   `data_dir + "/captions/cap.<cat>.glove.<split>.pkl"` and `:63-64` takes `data[index]['wv']`;
   `collate_fn` pads it to `max_text_words=30`
   (`configs/ce/fashioniq_dcnet.json:29`) and the model consumes it through
   `trainer/trainer_joint.py:178-179`. Every OOV row is inside every query's text tensor.
5. **Second, independent non-determinism in the same generator:** which words are OOV at all depends
   on the spaCy model loaded at `process_cap.py:9` (`spacy.load('en_vectors_web_lg').vocab`) and on
   `nltk.tokenize.word_tokenize` (`:22,55`) — the same text can tokenise differently across NLTK
   versions, changing both the vocabulary and the `wv` rows. So a re-run differs even with a fixed
   NumPy seed unless NLTK and spaCy versions are pinned too. (The registry's contract already says
   this: `deterministic_status: NONDETERMINISTIC_WITHOUT_AUTHOR_ARTIFACT`, and the note demands
   "record seed state, NumPy version, spaCy model/version, NLTK tokenizer/version, and generated
   hashes", `preparation_contracts.yaml:136-139`.)
6. **Words with train count ≤ 2 are silently dropped, not randomised** (`process_cap.py:58-59`
   `if word in vocab_words`): they never enter `vocab_words`, so they are removed from `wv`
   entirely — a second, deterministic-but-author-specific preprocessing choice that a reimplementation
   would have to match to be comparable.

**Consequence for this workbench:** the registered artifact set cannot be regenerated into a
comparable state. The only reproducible route is the **author's own PKLs**, staged verbatim. That is
exactly the policy in force (`automation_policy: AUTHOR_ARTIFACT_PREFERRED`,
`preparation_contracts.yaml:139`), and this audit endorses it: regeneration is not a repair, it is a
different experiment.

**Important caveat on one earlier claim.** A prior audit asserted the generator "cannot even read the
workbench root". The mechanism is real (`process_cap.py:12,35,39,40` are all `../dataset/fashioniq/…`,
i.e. relative to `preprocess/`, and nothing in the workbench maps `FASHIONIQ_ROOT` there), but it is
**not the blocker that matters**, because a *staged copy* of the author PKLs at the evaluator's
`data_dir` is what the run actually needs — and that path is `<checkout>/dataset/fashioniq`, reached
through the same missing link as §5. The regeneration defect is real and separate: it means the
PKLs, once lost, cannot be recreated faithfully by anyone.

### Availability of the official `.pkl` (assignment question 3)

- The **repository** references no published `.pkl`. OBSERVED `grep -rn "\.pkl" --include='*.py'`
  finds only producers/consumers: `CE_dataset.py:31,34` (read),
  `process_cap.py:35,46` and `gen_shoe_cap.py:106,116` (write). No download URL, no README line.
- The **author Drive folder** is the only candidate. OBSERVED this pass, read-only, with a plain GET
  of `https://drive.google.com/drive/folders/1wgygqF095Di67EaHaGOXbwh3wEzk9izB`:
  `<title>FashionIQ_ckpts - Google Drive</title>` and **exactly two rows**:

  ```
  rows: [('1ntGlPYUecVJKAVg-eZhdmt0ueN-RK52e', 'doc'), ('1Hzzc7RPMwQ_09Keb3aHDWE1UxWU5o-9w', 'doc')]
  config.json         -> aria-label 'config.json Unknown Shared', 'Size: 2 KB'
  trained_model.pth   -> row 2, size not rendered in the HTML listing
  ```

  So the folder ships **`config.json` + `trained_model.pth` and nothing else**: no PKLs, no
  `resized_images`, no `attributes/dict.attr.json`.
- OBSERVED a read-only GET of the smaller member returned 1982 bytes, sha256
  `80b33d43f6f49c2eeb8ddf948a205f33a78a4c19cdf25975af033ad3be1f320a`, and it is byte-comparable to
  the pin (§5). I did **not** download `trained_model.pth` (a `HEAD` hits Drive's confirmation
  interstitial and returns `200 0 text/html`, so no size is obtainable without the download).
- Therefore: **`AUXILIARY_ASSET_MISSING`.** The three val PKLs required by
  `data_loader/CE_dataset.py:34` have **no published source anywhere I could reach**, and their
  generator is non-reproducible (§6.1-6.5). This is the *hard* blocker for DCNet, independent of the
  checkpoint.

---

## 7. FASHIONIQ PROTOCOL

Registry: `native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:627`), quoted verbatim:
*"FashionIQ - Original Full Gallery, Reference Excluded"*, `evaluation_noise_pct: 0`
(`models.yaml:12-15`).

- Gallery: the whole ordered `image_splits/split.<cat>.val.json`. `CE_dataset.py:26-28`
  (`no_target` branch for `val_trg`/`test_trg`) does `read_json(Path(data_dir + f"/image_splits/split.{category}.{split.split('_')[0]}.json"))` — i.e. the full split file, not a pair union. Loader:
  `data_loaders.py:201-209`, `batch_size=100`, `shuffle=False`, `drop_last=False`. Features are
  concatenated in loader order (`trainer_joint.py:145-160`) and ranking indexes back into
  `val_ids = self.data_loaders[category + '_trg'].dataset.data` (`trainer_joint.py:225,229`).
- Reference policy: excluded by masking, per query, before ranking —
  `trainer_joint.py:216-218`:
  `score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10` and the same for
  `score_matrix_corr`; `id2targetidx` is built in gallery order at `:147-151`.
- Query construction: the annotation PKL's `candidate`/`target`/`captions`/`wv`
  (`CE_dataset.py:57-64`); text is the precomputed 300-d GloVe sequence.
- Categories: dress/shirt/toptee (`configs/ce/fashioniq_dcnet.json:22-26`).
- Recall computation: `model/metric.py:5-21` —
  `recall(actual, predicted, k) = len({actual} ∩ set(predicted[:k])) / 1`, averaged over queries;
  `k = 10, 50` at `:18-19`. Ranking depth is hard-coded 50 (`trainer_joint.py:228,232,240`), and the
  three reporting paths are composition, correction and their fusion
  `score_comb = F.log_softmax(score_comp[j], dim=-1) + F.log_softmax(score_corr[j], dim=-1)` (`:239`).
- Aggregates: `recall_avg` / `recall_avg_corr` / `comb_avg` (`trainer_joint.py:247-249`).
- **The source's own mask is a procedure, not a removal**: the reference still occupies a gallery
  column, so the workbench's `reference_excluded` (which removes the column) is equivalent for the
  top-k hit test only while `-10e10` is below every other score. The registry's protocol id is the
  right one (it describes the source's behaviour), and the pinned code does not implement
  `fashioniq_val_split`. I found nothing in the paper (which I did not re-read this pass) that the
  code alone could be contradicted by, and I do **not** propose changing the gallery, the reference
  policy, the caption input or the noise condition.

---

## 8. WORKBENCH INTEGRATION

| piece | state | evidence |
|---|---|---|
| adapter | `DCNetAdapter.command` → `[python, <source>/test.py, "--resume", <dir>]` | `adapters/models.py:79-84` |
| cwd | the checkout root, for every model except csmcir | `evaluate_models.py:214-215` |
| planner | dry-run resolves model + variant and then blocks on requirements | OBSERVED below |
| checkpoint prep | `<root>/dcnet/fashioniq_dcnet`; directory artifact with two members | `registry.py:842-863`, `:365-395` |
| layout check | `captions/`, `image_splits/`, flat `resized_images/`, the three `cap.<cat>.glove.val.pkl`, the three val splits | `fashioniq_layout.py:55-62`, `:83-89` |
| image extension | `.jpg`, flat `resized_images/{id}.jpg` | `models.yaml:623`; `fashioniq_layout.py:88,109-135` |
| dataset link | **none declared** | OBSERVED `awk` → only lines 37/665/726/848 |
| metric parser | none → catch-all | `metrics_extraction.py:413-422` |
| per-query support | none (no dcnet replay module) | OBSERVED `workbench/replay/` listing |
| gate | committed row `dcnet_fashioniq_val_table1`, `VERIFIED`, protocol `fashioniq_full_gallery_ref_excluded`, values r10 27.78 / r50 53.89 / mean 40.83 with r10/r50 `WORKBENCH_DERIVED` and mean `PAPER_REPORTED` | `paper_benchmarks/papers.yaml:1442-1470` |

OBSERVED, this pass:

```
$ PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B workbench/scripts/evaluate_models.py --model dcnet --dry-run
[RUN] dcnet / fashioniq_run_directory
[BLOCKED] required generated artifact missing: .../data/fashionIQ_dataset/captions/cap.dress.glove.val.pkl
[BLOCKED] required generated artifact missing: .../data/fashionIQ_dataset/captions/cap.shirt.glove.val.pkl
[BLOCKED] required generated artifact missing: .../data/fashionIQ_dataset/captions/cap.toptee.glove.val.pkl
[BLOCKED] required generated artifact missing: .../data/fashionIQ_dataset/resized_images
[BLOCKED] FashionIQ image extension requirement: image directory missing: .../data/fashionIQ_dataset/resized_images
[BLOCKED] external runtime asset file missing: torchvision ResNet-50 ImageNet weights (~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth)
[BLOCKED] checkpoint missing: .../workbench/artifacts/checkpoints/dcnet/fashioniq_dcnet
[BLOCKED] checkpoint mapping unresolved
```

and via `model_requirements(dcnet, resolve_config())`: the same six requirement blocks. The
resized-image tree that *does* exist is `data/fashionIQ_dataset/resized_image/{dress,shirt,toptee}`
(OBSERVED, 3817/6346/5373 `.jpg`) — that is CLVC-Net's **per-category** layout, which DCNet never
opens (`CE_dataset.py:22,72-73` wants flat `resized_images`). Do not point DCNet at it.

---

## 9. CONFIRMED DEFECTS

**C1 — the required GloVe val PKLs are unrepairable by regeneration (P1).**
Root cause: `process_cap.py:31` draws an unseeded `np.random.normal(0, 0.3, (300,))` for OOV words,
the file seeds nothing (OBSERVED `grep -rn "seed" preprocess/`), and the author folder ships no PKLs
(OBSERVED two Drive rows only). Fix: none on the workbench side — the PKLs must be obtained from the
author (or an author-affiliated mirror) and staged with recorded digests. Regression result: until
they exist, the evaluator cannot build its text tensors at all (`CE_dataset.py:33-35` raises).
**This is not a "fix the path" problem; a regenerated PKL is a different experiment.**

**C2 — no path from the evaluator to the staged dataset (P1).** `data_dir: "dataset/fashioniq"`
(`configs/ce/fashioniq_dcnet.json:21`) resolves against the cwd, which is the checkout root
(`evaluate_models.py:214-215`), while everything is prepared under `FASHIONIQ_ROOT`
(`data/fashionIQ_dataset`). No `canonical_dataset_link` is declared for dcnet
(OBSERVED: csmcir/combiner/clip4cir/sprc only). There is no `--data_dir` CLI
(`test.py:97-101`). Fix: declare `canonical_dataset_link: dataset/fashioniq` **or** stage a
`<checkout>/dataset/fashioniq` link, and keep it inside the approved runtime-artifact mechanism
(`registry.py:729-740`, `preflight.py:86-100`).

**C3 — the flat `resized_images/` tree does not exist and its generator is unrunnable on this host
(P1).** The evaluator opens `candidate_id + '.jpg'` in a flat directory
(`CE_dataset.py:22,72-73`); the workbench recipe declares that layout
(`preparation_contracts.yaml:90-103`, `prepare_model.py` `layout: flat`) but the tree is absent in
`data/fashionIQ_dataset` (OBSERVED `ls -d resized_images` → "No such file or directory"). The
upstream generator itself hard-codes `image_dir = '../dataset/fashioniq/images/'` and
`pattern = '*.jpg'` (`resize_img.py:68-73`) and uses `Image.ANTIALIAS` (`:11`), which is **removed**
in Pillow ≥ 10 (OBSERVED `hasattr(Image,'ANTIALIAS')` → `False` on Pillow 12.3.0) — so the upstream
prep recipe is doubly unusable, and only the workbench recipe can produce the artifact. Fix: run the
workbench recipe at the configured root and record the receipt; if the canonical images are PNG-only,
ensure the recipe's source probing matches the images actually present (a prior audit reported a
`.jpg`-only probe failure — I did not re-run the resize recipe this pass, so **that specific failure
is not re-verified here**).

**C4 — `--resume` mutates the author run directory (P2).** `parse_config.py:41-42,57-60` writes
`<dir>/val/config.json`; `trainer/trainer_joint.py:250-252` writes `<dir>/val/att_result.pt`
(whenever `--mode val`, i.e. the **default**). The integrity scope is only the two declared members
(`registry.py:365-395`), so the mutation is invisible to the workbench. Fix: run from a copy, or
declare the mutation.

**C5 — no aggregate metric extraction (P2).** OBSERVED `metrics_extraction.py:413-422` hands every
model outside {limn, csmcir, combiner, clip4cir, sprc} `AGGREGATE_PARSER_UNAVAILABLE`. The output
this evaluator prints is perfectly parseable — `test.py:77,79` emit
`[<category>] r@10, r@50: ...` and `[<category>] comp corr r@10, r@50: ...`, plus `[Avg Recall]`,
`[Avg Recall corr]`, `[comb_avg]` (`:69-73`) — but only via the logging console handler
(`logger/logger_config.json` `console` → `ext://sys.stdout`), so a parser must read stdout, not a
file. Fix: implement it, or record manual capture.

**C6 — `logger.info` before `setup_logging` (P3).** `test.py:18` calls `config.get_logger('test')`
(`parse_config.py:77-80`) which only calls `logging.getLogger(name).setLevel(INFO)`; the handler
configuration happens later in `ConfigParser.__init__` → `setup_logging`
(`parse_config.py:62`, `logger/logger.py:7-24`). Since it is the same root logger, the console
handler is attached before `test.py:18`'s first use, so order is fine — noted only because a wrapper
reimplementing the config flow could easily print nothing.

**Not a defect but worth stating:** the `arch`-mismatch warning at `base/base_trainer.py:200-204`
cannot fire here, because the author's released `config.json` is semantically identical to the pin
(§5).

---

## 10. REMAINING BLOCKERS

| # | classification (workbench vocabulary) | ownership | evidence | remediation |
|---|---|---|---|---|
| B1 | `AUXILIARY_ASSET_MISSING` | author | OBSERVED Drive folder = {`config.json`, `trained_model.pth`} only; `CE_dataset.py:33-35` needs three `cap.<cat>.glove.val.pkl`; `process_cap.py:31` unseeded | obtain the author PKLs (or an author-affiliated copy) and stage with digests; **never** regenerate and call it parity |
| B2 | `NONDETERMINISTIC_PREPROCESSING` | author | `process_cap.py:31`; OBSERVED no seeding anywhere in `preprocess/` | already correctly classified (`preparation_contracts.yaml:136-139`); keep `AUTHOR_ARTIFACT_PREFERRED` |
| B3 | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` | workbench/operator | `configs/ce/fashioniq_dcnet.json:21` vs `evaluate_models.py:214-215`; no `canonical_dataset_link` for dcnet | declare the link (or stage `<checkout>/dataset/fashioniq`) |
| B4 | `PUBLIC_ASSET_NOT_PREPARED` | operator | OBSERVED `resized_images` absent at the configured root | run `prepare_model.py` for the dcnet layout (256, flat, `.jpg`) and record the receipt |
| B5 | `PUBLIC_ASSET_NOT_PREPARED` | operator | OBSERVED preflight: ResNet-50 V1 missing at `~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth` | stage the verified V1 file; **do not** substitute modern defaults (`ResNet50_Weights.DEFAULT` = `IMAGENET1K_V2`) |
| B6 | `CHECKPOINT_URL_INACCESSIBLE` (from the workbench's point of view) / `CHECKPOINT_UNPUBLISHED` (no file URL or digest) | author | `models.yaml:651-655` (`download_url: null`, size/sha null) | the folder is reachable (OBSERVED titles) but member URLs/digests are not published; obtain and hash them before claiming availability |
| B7 | `FIRST_PARTY_ENVIRONMENT_CONFIGURATION` | workbench/operator | `models.yaml:637` `WORKBENCH_DCNET_PYTHON` unset; the venv here is py3.13/torch2.14 while the pin declares py3.7.7/torch1.4.0 | the modern route is viable (§4: almost nothing breaks), so plan it explicitly rather than defaulting to a legacy env |
| B8 | `GPU_RUNTIME_UNTESTED` | operator | `models.yaml:642` `cuda: UNKNOWN`; the eval path uses `torch.device('cuda' if available else 'cpu')` (`test.py:61`) and `base_trainer._prepare_device` degrades to CPU — **but** `model/loss.py` is training-only, so evaluation is genuinely CPU-capable | verify the declared runtime on a GPU host; record that this model is *not* in the CUDA-hard class |
| B9 | `FIRST_PARTY_ADAPTER_MISSING` (metric path) | workbench | `metrics_extraction.py:413-422` | add a `dcnet` stdout parser (`test.py:69-84`) |
| B10 | `SCIENTIFIC_PROTOCOL_UNVERIFIED` (partial) | workbench | gallery/reference are VERIFIED from code (`CE_dataset.py:26-28`, `trainer_joint.py:216-218`); whether the *paper* excluded the reference is not established by anything I opened | annotate `literature_split_label` as inferred from code unless the paper text is quoted |

---

## 11. IMPLEMENTATION PLAN

1. **Stage the data link.** Declare `canonical_dataset_link: dataset/fashioniq` for `dcnet`, or
   create `<checkout>/dataset/fashioniq → $FASHIONIQ_ROOT`. Without this the run cannot find its own
   images/splits/PKLs (`CE_dataset.py:22,27,34,73`) no matter what else is fixed. No source edit.
2. **Prepare the flat resized tree** with the workbench recipe at 256 / `.jpg` / flat
   (`prepare_model.py`), under the configured root, and keep the receipt.
3. **Obtain the profile.** `trained_model.pth` + `config.json` from the author folder, hashed; the
   registry's `required_files` already matches the loader exactly (§5), so no registry change is
   needed for the shape — only for the digests.
4. **Obtain the three val PKLs.** No substitute exists. Record their sha256 and treat them as
   author artifacts (consistent with `AUTHOR_ARTIFACT_PREFERRED`).
5. **Metric parser:** add `parse_dcnet_output` for the `[cat] comp corr r@10, r@50:` /
   `[Avg Recall]` / `[comb_avg]` lines and wire it into `extract_aggregate_metrics`. Report the
   **combined** number (`comb_avg`) for the paper comparison and label `recall_avg*` as branch means.
6. Optional hygiene: run from a copy of the run directory so `--resume` cannot mutate the author
   artifact (C4).

Complexity: medium — but **not a code problem**. Four of the six steps are asset acquisition.
Scientific risk: if and only if the author PKLs are used verbatim and no config/`data_dir` semantic
change is made, low; the whole risk sits in step 4, because a regenerated PKL is a different
experiment (B2). Expected outputs: per-category `comp corr r@10, r@50` triples plus `comb_avg`
(`test.py:79,73`), comparable to the committed row `dcnet_fashioniq_val_table1`
(27.78 / 53.89 / 40.83) under the gate's 5% tolerance.

---

## 12. FINAL VERDICT

**BLOCKED.** Not a compatibility problem: DCNet's evaluation path survives a modern runtime almost
untouched (no removed torchvision submodule, no Python-2-ism; the only hard removal,
`Image.ANTIALIAS`, is in the preparation script). The blockers are missing *public assets*:

- **B1 (`AUXILIARY_ASSET_MISSING`) is decisive and, as far as I can reach, permanent.** The three
  `cap.<cat>.glove.val.pkl` files are required by `CE_dataset.py:34`, are published nowhere I could
  find, and **cannot be regenerated reproducibly**: `process_cap.py:31` randomises every OOV word's
  300-d vector with no seed anywhere in the file (OBSERVED), and tokenisation/vocabulary depend on
  the spaCy and NLTK versions as well. A regenerated PKL set is a *different experiment* with no
  way to quantify the delta (the affected words are unknowable without the author's spaCy model and
  PKLs).
- B3/B4: the run directory contract is exact and well understood (§5), but nothing links the
  evaluator's cwd-relative `data_dir` to the prepared root, and the flat `resized_images/` tree does
  not exist.
- B6: the checkpoint folder is reachable and contains exactly the two declared members (a new,
  observed fact), but no member URL/digest is published, so `checkpoint_mapping_status` correctly
  stays `UNVERIFIED` and the workbench refuses at `adapters/base.py:33-36` before the download
  question even arises.

Secondary status: **VARIANT** on the reference-exclusion *procedure* only — the source masks the
reference column and the workbench removes it; equivalent for top-k membership, not identical as an
operation, and never to be reported as byte-equal.

Nothing was reproduced. Nothing was executed end to end. **0 RUNTIME_VERIFIED claims.**

---

## 13. EVIDENCE

Repo / pin:
- `upstream_repo_url: https://github.com/ozmig77/dcnet`;
  `upstream_commit_sha: 68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb` (`models.yaml:617,619`).
- OBSERVED `git rev-parse HEAD` = that SHA; `ls-remote origin HEAD` = that SHA (SNAPSHOT 2);
  `rev-list --all --count` = 1; `status --porcelain` clean.

Key `file:line` citations (all opened this pass):
- `parse_config.py:14-28,39-44,57-60,62,73-75`; `base/base_trainer.py:51-54,63-68,188-210,199-204`;
  `test.py:19,22,36-59,61,64,66-90,96-117`; `trainer/trainer_joint.py:140-160,144-252,166-172,
  174-195,216-218,225,228,232,239-243,247-252`; `model/metric.py:5-21,18-19`;
  `model/model.py:6,17-23,262-300,322-345`; `model/modules.py:4,164-175`; `model/loss.py:4,38,62,64,87,89,
  97-107`; `data_loader/CE_dataset.py:20-40,55-73,74-110`; `data_loader/data_loaders.py:66-70,
  160-209,211-235`; `utils/util.py:64-108,115-165`; `utils/text2vec.py:71-83`;
  `configs/ce/fashioniq_dcnet.json:17-33,35-47,70-82`; `preprocess/process_cap.py:9,12-36,39-47,50-64`;
  `preprocess/resize_img.py:9-11,32-43,45-77`; `preprocess/gen_shoe_cap.py:9,96-107`; `requirement.txt:1-13`;
  `README.md:26-29,32-41,47-52,59-66`.
- Workbench: `workbench/registry/models.yaml:10-19,606-655`; `preparation_contracts.yaml:60-139`;
  `paper_benchmarks/papers.yaml:1442-1470`; `workbench/backend/adapters/models.py:79-84`;
  `workbench/backend/registry.py:365-395,729-740,842-863`; `workbench/backend/preflight.py:24-44,
  86-100`; `workbench/backend/fashioniq_layout.py:55-62,83-89,109-135`;
  `workbench/backend/metrics_extraction.py:413-422`; `workbench/scripts/evaluate_models.py:214-215`;
  `workbench/backend/operator_config.py:108-115`.

Commands and observed results:
- Dry-run (quoted in full in §8) → eight blockers.
- Drive folder GET → title `FashionIQ_ckpts`, rows `config.json` + `trained_model.pth`; folder also
  reported "Size: 2 KB" for `config.json`.
- Author `config.json` GET → 1982 bytes, sha256 `80b33d43…f320a`; semantically identical to the pin.
- `torch.load` on a DCNet-shaped payload → OK under the modern default.
- `hasattr(Image,'ANTIALIAS')` → `False` (Pillow 12.3.0).
- `grep -rn "seed" preprocess/` → no seeding call in the generator.
- `ls data/fashionIQ_dataset/` → `captions, image_splits, images, resized_image`; no `resized_images`,
  no `*.pkl`.

Not run, deliberately: no GPU inference, no checkpoint download, no weight download, no install, no
write outside this file.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 of 6** — registry/identity and static source + protocol auditing are complete and
independently evidenced, including a live read-only inspection of the author's Drive folder (which
established, for the first time in this workbench, exactly which two members it serves) and a byte
comparison of the released config against the pin. There is **no** dependency installation, **no**
checkpoint or PKL acquisition, **no** evaluator execution, **no** metric extraction from a real run
and **no** numerical comparison to `dcnet_fashioniq_val_table1`. §3's chain, §5's contract and §6's
non-determinism finding are STATIC_AUDIT.
