# intent — reproducibility reassessment

Independent adversarial audit. READ-ONLY: no file under `/home/heheboiz/data/cir` was modified; all
scratch went to `/tmp`. No GPU job, no install, no checkpoint download. Every claim below is
STATIC_AUDIT unless the line says otherwise, and each cites a file:line I opened in the pinned
checkout or a command with its observed output. There is no RUNTIME_VERIFIED claim in this document.

Every pre-existing audit of this model was treated as a HYPOTHESIS. One of them is refuted below
(§9, D-INT-1) with an executed command, and the registry itself already withdrew that claim.

---

## 1. MODEL IDENTITY

| field | value | evidence |
|---|---|---|
| model_id | `intent` | `workbench/registry/models.yaml:305` |
| method_name | `INTENT` | `models.yaml:315` |
| paper title | INTENT: Invariance and Discrimination-aware Noise Mitigation for Robust Composed Image Retrieval | `models.yaml:319`; pinned `workbench/third_party/INTENT/README.md:8` carries the same string |
| paper URL | https://ojs.aaai.org/index.php/AAAI/article/view/39181 | `models.yaml:320` |
| year | 2026 | `models.yaml:317` |
| source URL | https://github.com/iLearn-Lab/AAAI26-INTENT | `models.yaml:321` |
| pinned commit | `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` | `models.yaml:323` |
| source_dir | `INTENT` | `models.yaml:325` |
| checkpoint variant | three FashionIQ files, 20/50/80 % training noise; no clean variant | `models.yaml:337-367` |
| original metrics | R@10 55.31, R@50 75.16, mean 65.24 | `models.yaml:331-333` |
| native protocol | `fashioniq_full_gallery_ref_excluded` | `models.yaml:329` |

Pin verification (command, observed):

```
$ git -C workbench/third_party/INTENT rev-parse HEAD
2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3        # == models.yaml:323
```

Paper URL: `curl -s -o /dev/null -w "%{http_code}" https://ojs.aaai.org/index.php/AAAI/article/view/39181`
→ `000` (no response). Crossref resolves the DOI instead:
`https://api.crossref.org/works/10.1609/aaai.v40i25.39181` → `"status":"ok"`,
`publisher: Association for the Advancement of Artificial Intelligence (AAAI)`. So the paper exists;
the registry's own URL is not reachable from here. **UNKNOWN** for the OJS page, VERIFIED for the DOI.

---

## 2. SOURCE STATUS

Accessible and complete as a tree. `git -C workbench/third_party/INTENT status --porcelain` reports
only `__pycache__/*.pyc` deletions/untracked bytecode; no tracked source file is modified, and the
workbench is not patching this checkout (contrast `models.yaml:88-95` for CSMCIR).

Two modules are absent by design (upstream never committed them), and one dependency chain is broken:

* `requirements.txt` does not exist, although the README tells the user to install from it:
  `workbench/third_party/INTENT/README.md:123` `pip install -r requirements.txt`;
  `ls workbench/third_party/INTENT/requirements.txt` → `No such file or directory`.
* The pinned `lavis` package cannot be imported. Executed:

```
$ PYTHONPATH=/home/heheboiz/data/cir/workbench/third_party/INTENT \
    /home/heheboiz/data/cir/.venv/bin/python -c "from lavis.models import load_model_and_preprocess"
ModuleNotFoundError: No module named 'torchaudio'      (first failure; see §4 for the second)
```

**SNAPSHOT 2 — current upstream state.** Read-only remote inspection, no fetch into the checkout:

```
$ git -C workbench/third_party/INTENT ls-remote origin
2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3        HEAD
2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3        refs/heads/main
$ git -C workbench/third_party/INTENT log --oneline --all
2a75fc6 Update README.md
```

Upstream HEAD **equals** the pin, and the repository still has exactly one commit
(`git rev-list --count HEAD` → `1`), i.e. no CLI was added, no dependency was fixed, no checkpoint was
published since the pin. `git fetch origin --dry-run` produced no output and no error (nothing to
fetch). **SNAPSHOT 2 == SNAPSHOT 1.** The pin was not changed.

---

## 3. NATIVE EVALUATION WORKFLOW

**Entrypoint.** There is none. `workbench/third_party/INTENT/test.py` has no `__main__` and no
`argparse` (`grep -n "__main__\|argparse" test.py` → zero hits; the file is 357 lines and ends at
`:357`). The only parser in the checkout is `train_INTENT.py:41-69` (`parser = argparse.ArgumentParser()`).

**Exact call chain (all STATIC_AUDIT, every line opened):**

```
train_INTENT.py:41-69     argparse -> args (dataset default 'cirr', :43; fashioniq_path default '/root/fashion_iq_data/', :44)
train_INTENT.py:347,378   __main__ -> create_model_and_optimizer()
train_INTENT.py:162-165   create_model_and_optimizer(): load_model_and_preprocess(name="Blip2QformerCir", model_type="pretrain", is_eval=False, device=args.device)
train_INTENT.py:167       model.cuda()                              # unconditional
train_INTENT.py:76,95-100 load_dataset() -> datasets.FashionIQ(path=args.fashioniq_path, transform=targetpad(1.25,224),
                                                                  noise_ratio=args.noise_ratio, split=args.split)
train_INTENT.py:101       trainset.shuffle()                       # training-only noise injection
train_INTENT.py:283-284   for ci, category in enumerate(['dress','shirt','toptee']):
                              t = test.test(args, model, trainset, category, txt_processors)
test.py:9-10              def test(params, model, testset, category, txt_processors): model.eval()
test.py:81-86             category -> testset.test_queries_dress / _shirt / _toptee   (and their test_targets_*)
test.py:107-113           per-batch: torch.stack(imgs).float().cuda(); mods=[txt_processors["eval"](c) for c in mods]
                          f = model.extract_retrieval_compose(imgs, mods)            -> blip2_cir.py:722
test.py:124-128           gallery: imgs = model.extract_retrieval_target(imgs)          -> blip2_cir.py:772
test.py:141               sims = np.matmul(all_queries, all_imgs)     # all_queries (M,256,1,1), all_imgs (N,256,32)
test.py:142-143           sims = sims.squeeze(); sims = sims.max(-1)  -> (M,N)
test.py:146-150           test_targets_id = [...] ; sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
test.py:153               nn_result = [np.argsort(-sims[i,:])[:50] for i in range(M)]
test.py:167-174           for k in [1,10,50]: r = 100*r/len(nn_result); out += [('<name>_r<k>', r)]
```

**Metric returned:** per-category `('dress_r1'|... , value)` for k ∈ {1,10,50}. No macro, no
mean — `test.test` returns the 3-tuple list only.

**CLI?** No. **What a wrapper must call** (see §5).

**Missing components.** No checkpoint-restore code anywhere: `grep -rn "restore_file\|load_checkpoint"
train_INTENT.py test.py cirr_sub_BLIP2.py` → the only hits are `utils.py:128` (definition, no caller).
`train_INTENT.py:361` calls `train_and_evaluate(...)` from `:368-379` and never restores from
`--model_dir`; the loop runs `--num_epochs` (default 20, `:49`) before producing any metric.

---

## 4. DEPENDENCIES

D-category legend is **not defined inside this repository** — `grep -rn "D1" --include=*.md` over the
tracked docs returns only an unrelated `final_pc_audit` defect table
(`workbench/docs/reviews/final_pc_audit/AUDIT_F_ADVERSARIAL.md:143`, whose rows are `D1..Dn` defects of
the PC launcher, not dependency classes). I therefore classify against the eight canonical dependency
classes by behaviour and record the mapping explicitly rather than guessing a numbering. This is
stated as UNKNOWN-for-numbering, not as PASS.

Reachability of each import from the FashionIQ evaluation path (`train_INTENT.py:284` → `test.test` →
`blip2_cir.py` → `lavis` package import) was proved by executing the import and by walking the
chain. Observed first failure and second failure:

```
$ PYTHONPATH=<INTENT> .venv/bin/python -c "from lavis.models import load_model_and_preprocess"
  File ".../INTENT/lavis/__init__.py", line 15, in <module>          from lavis.datasets.builders import *
  File ".../INTENT/lavis/datasets/builders/base_dataset_builder.py", line 18   from lavis.processors.base_processor import BaseProcessor
  File ".../INTENT/lavis/processors/__init__.py", line 29            from lavis.processors.audio_processors import BeatsAudioProcessor
  File ".../INTENT/lavis/processors/audio_processors.py", line 9     import torchaudio
ModuleNotFoundError: No module named 'torchaudio'

$ ... with /tmp/stub on PYTHONPATH providing a fake torchaudio (second failure)
  ... INTENT/lavis/processors/audio_processors.py:17  from lavis.models.beats.Tokenizers import TokenizersConfig, Tokenizers
  ... INTENT/lavis/models/__init__.py:15              from lavis.models.albef_models.albef_classification import AlbefClassification
  ... INTENT/lavis/models/med.py:40-43                from transformers.modeling_utils import (..., apply_chunking_to_forward, ...)
ImportError: cannot import name 'apply_chunking_to_forward' from 'transformers.modeling_utils'
```

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `torchaudio` | `INTENT/lavis/processors/audio_processors.py:9` | first-party vendored LAVIS; declared in the README | `2.1.0` (`README.md:116`) | YES (PyPI/whl) | py38-312 / 2.1.0 / cpu+cuda | install into the model env | **REACHABLE** on the import chain — proven by the executed traceback above |
| `torchaudio.transforms` | `audio_processors.py:10` | same | `2.1.0` | YES | same | same | REACHABLE (same module) |
| `torchaudio.compliance.kaldi` | `audio_processors.py:13` | same | `2.1.0` | YES | same | same | REACHABLE (same module) |
| `torchaudio` | `INTENT/lavis/datasets/datasets/audio_captioning_datasets.py:16` | vendored LAVIS | `2.1.0` | YES | same | same | REACHABLE: `lavis/datasets/builders/__init__.py:93-94` imports `audio_caption_builder` at package import |
| `torchaudio` | `INTENT/lavis/datasets/datasets/audio_classification_datasets.py:16` | vendored LAVIS | `2.1.0` | YES | same | same | REACHABLE via `classification_builder.py:14` (imported by the same `__init__`) |
| `moviepy` | `INTENT/lavis/processors/audio_processors.py:11` | vendored LAVIS | unpinned upstream | YES | pure py | install, or the stub/removal path | REACHABLE (same module that already fails) |
| `pandas` | `audio_captioning_datasets.py:15` | vendored LAVIS | unpinned | YES | pure py | install | REACHABLE |
| `setproctitle` | `INTENT/train_INTENT.py:33` and `:351` | first-party trainer (not vendored) | unpinned | YES | pure py | install | **TRAINING-ONLY — NOT on the evaluation path.** `test.py` never imports it; `grep -rn setproctitle INTENT/*.py` → only `train_INTENT.py:33,37,351,354` |
| `clip` | `INTENT/train_INTENT.py:19` | third-party `clip` (OpenAI) | unpinned, absent from README | YES | pure py | install | TRAINING-ONLY; also **unused** (`grep -n "clip\." train_INTENT.py` → no hits) |
| `matplotlib` | `INTENT/lavis/models/blip2_models/blip2_cir.py:14` | vendored LAVIS (top of the model file) | unpinned | YES | pure py | install | REACHABLE — this is a module-level import of the model class the evaluator uses |
| `sklearn` (`KMeans`) | `INTENT/lavis/models/blip2_models/blip2_cir.py:30` | vendored LAVIS | `scikit-learn==1.3.2` (`README.md:119`) | YES | pure py | install | REACHABLE (same file) |
| `transformers` | `INTENT/lavis/models/med.py:40-43` | vendored LAVIS | `4.25.0` (`README.md:119`) | YES | pure py | pin exactly | REACHABLE and **VERSION-CRITICAL** — see below |
| `torch.distributed` | `blip2_cir.py:10` | torch stdlib | any | n/a | — | none | imported, never used (`grep -n "dist\." blip2_cir.py` at module scope → none) |
| `typing_extensions` | `INTENT/data_utils.py:3` | stdlib-adjacent | any | YES | pure py | none | REACHABLE only if `data_utils` is imported |

Observed versions in the workbench venv (`.venv/bin/python -c ...`):

```
transformers 4.57.6      torch 2.14.0+cpu       timm 1.0.30
torchaudio   MISSING     moviepy MISSING        sklearn MISSING
setproctitle MISSING     clip MISSING           matplotlib MISSING
```

Classification, in the task's vocabulary:

* `torchaudio`, `moviepy`, `pandas`, `matplotlib`, `scikit-learn` → **INSTALLABLE_DEPENDENCY_MISSING**
  (all reachable from the evaluation import chain; each has a public distribution for the declared
  env). Severity **P1** for `torchaudio`/`moviepy` because they sit on the *import* path of the
  evaluator's own package, and P2 for the rest.
* `transformers 4.57.6` vs the README's `4.25.0` → **DEPENDENCY_VERSION_CONFLICT**. `med.py:40-43`
  imports `apply_chunking_to_forward` from `transformers.modeling_utils`, which modern transformers no
  longer exports; observed `ImportError`. Severity **P1**.
* The pinned `lavis` package's unconditional `from lavis.datasets.builders import *`
  (`lavis/__init__.py:15`) dragging in audio/beats modules → **UPSTREAM_IMPORT_BUG** (upstream ships
  the coupling; the workbench owns no patch for it). Severity **P1**.
* `setproctitle`, `clip` → **not blockers**: training/side-script only, and `clip` is unused.
  Severity **P3**.
* `torch==2.1.0+cuda` and Python 3.9 per `README.md:112-116`: the workbench venv is 3.13 with
  `torch 2.14.0+cpu`, so a GPU run requires the declared env —
  **FIRST_PARTY_ENVIRONMENT_CONFIGURATION** if the declared env is ever used as-is for a CPU host.
  Severity P2.

Environment declaration is faithful to the README: `models.yaml:307-314`
(`python 3.9 / torch 2.1.0 / torchvision 0.16.0 / cuda 12.6 / open-clip-torch 2.24.0 ·
scikit-learn 1.3.2 · transformers 4.25.0 · salesforce-lavis 1.0.2 · timm 0.9.16`) versus
`README.md:112,116,119`. `confidence: VERIFIED_README` (`models.yaml:309`) is correct. Note
`models.yaml:313` records CUDA `12.6` while the README installs the **cu121** wheels
(`README.md:116`) — cosmetic mismatch, P3.

---

## 5. CHECKPOINTS

Registry variants (`models.yaml:337-367`), all `status: NO_CLEAN_CHECKPOINT`,
`checkpoint_mapping_status: VERIFIED_METADATA`:

| id | filename | training noise | registry size | registry sha256 |
|---|---|---|---|---|
| `fiq_n02` | `intent_fiq_0.2.pt` | 20 % | 2722478522 | `2a504cadfd6b1c2486a2c508538b0160c5653f62c7cd11f4f3d8a9534dc26afc` |
| `fiq_n05` | `intent_fiq_0.5.pt` | 50 % | 2722478522 | `2d59ae002ecbf645a52c7e93bf8afd914e3061e2304413b97a0bf15ca43e39a9` |
| `fiq_n08` | `intent_fiq_0.8.pt` | 80 % | 2722478522 | `671bb217d3fdf32e6133fcb98eef4e80ebf831b5a203493c54507ef3c23b5cd2` |

Observed live against the author Hub (STATIC_AUDIT on remote metadata, no weight downloaded):

```
$ curl -s "https://huggingface.co/api/models/iLearn-Lab/AAAI26-INTENT?blobs=true"
intent_cirr_0.2.pt  2722478522  519939a3b50835aeef2a197cbc1fd73c12f998c4aff5fe19450faeb3ef1c0670
intent_fiq_0.2.pt   2722478522  2a504cadfd6b1c2486a2c508538b0160c5653f62c7cd11f4f3d8a9534dc26afc
intent_fiq_0.5.pt   2722478522  2d59ae002ecbf645a52c7e93bf8afd914e3061e2304413b97a0bf15ca43e39a9
intent_fiq_0.8.pt   2722478522  671bb217d3fdf32e6133fcb98eef4e80ebf831b5a203493c54507ef3c23b5cd2
```

* All three registry filenames and both registry numbers (size, sha256) reproduce **exactly**. The
  registry's digests are now truthfully LFS-oid backed. Availability class: **URL_VERIFIED**, no bytes
  local (`ls workbench/artifacts/checkpoints/intent` → `No such file or directory`).
* **There is no clean 0 % FashionIQ checkpoint.** The paper's headline row is the 0 % row
  (`models.yaml:331-333` = 55.31 / 75.16 / 65.24) and no published file matches it. The Hub card's
  "Usage" step names only `best_model.pth` generically. `strict_clean_checkpoint_available: false`
  (`models.yaml:335`) is **correct**, and `workbench/tests/test_checkpoint_selection.py:131-133`
  asserts INTENT is blocked under `STRICT_CLEAN`.

**Loader contract — this is the decisive gap.** The pinned tree contains **no FashionIQ checkpoint
loader at all**:

* `test.py:9` `def test(params, model, testset, category, txt_processors)` — receives an
  already-constructed model. It never reads a path.
* `train_INTENT.py` declares no `--restore_file` and calls no loader: `grep -rn
  "restore_file\|load_checkpoint" train_INTENT.py` → **zero hits**. (`AUXILIARY_ASSET_AUDIT.md:203`
  and the earlier `intent.md` both describe a `--restore_file` claim; the earlier audit corrected it,
  and I confirm: there is none.)
* `utils.py:128` defines `load_checkpoint(checkpoint, model, optimizer=None)` and `:139`
  `checkpoint = torch.load(checkpoint)` then `:140` `model.load_state_dict(checkpoint['state_dict'])`.
  It is **never called** by any file (`grep -rn load_checkpoint INTENT/*.py` → definition only). Note
  the loader also demands a `state_dict` key, while the trainer saves the whole module:
  `train_INTENT.py:339,384` `torch.save(model, ...)` → a pickled `Blip2QformerCir` object, not a
  `state_dict`. The shipped `utils.load_checkpoint` therefore **cannot read the authors' own files**.
* The only `torch.load` in the tree is `cirr_sub_BLIP2.py:128`, inside a CIRR-only script whose model
  directory is hard-coded to `/root/autodl-tmp/BLIP/test/` (`:122`) and whose dataset is hard-coded to
  `/root/autodl-tmp/cirr_data/CIRR/` (`:120`). It calls `test_cirr_submit_result`, never `test.test`.

**Mapping confidence.** Filename→noise-percentage is metadata-only (`VERIFIED_METADATA` is the
registry's own honest label). Whether `intent_fiq_0.2.pt` really encodes 20 % shuffling, and whether
it loads into `Blip2QformerCir` (`blip2_cir.py:115-176`), is **UNKNOWN** — no download, no load.

---

## 6. AUXILIARY ASSETS

| asset | purpose | source | availability | generation possible? |
|---|---|---|---|---|
| `resized_image/{dress,shirt,toptee}/*.jpg` | query + gallery images | operator-prepared from the canonical FashionIQ JPEGs | **PRESENT locally** (observed below) | yes — `fashioniq_resize_images` in `preparation_contracts.yaml:33,95` |
| `captions/correction_dict_{dress,shirt,toptee}.json` | query-text normalisation | **`LIMN/correct_files/` and `TGCIR/correction_dict/`, both in pinned checkouts (see below)** | **PRESENT locally, bytes proven identical to the pinned LIMN files** | copy |
| `captions/cap.{cat}.{train,val}.json`, `image_splits/split.{cat}.val.json` | queries + gallery order | official FashionIQ release | PRESENT locally | n/a |
| `fashion_iq_data.json` + six `test_queries_*/test_targets_*.pkl` | in-checkout cache | generated by `datasets.py:37-51` on cache miss | **ABSENT** locally | yes, but requires the train captions |
| `captions/cap.{cat}.train.json` | needed for the cache-miss branch | official release | PRESENT locally | n/a |
| BLIP-2 pretrained weights + EVA-ViT-g | backbone init | `blip2_pretrain.yaml:10` Google URL; `eva_vit.py:443` | declared `AVAILABLE` in `cluster_b3_ilearn.yaml` | download |
| `/root/bert-base-uncased` | tokenizer + Q-Former init | `blip2.py:32,48,55` hard-coded absolute path | declared `AVAILABLE`; **not present on this host** | download to that exact path |

**This contradicts two standing documents, and the code agrees with me.** Observed:

```
$ ls data/fashionIQ_dataset/          -> captions  image_splits  images  qwen_captions  resized_image
$ ls data/fashionIQ_dataset/resized_image/
dress  shirt  toptee
$ for c in dress shirt toptee; do ls data/fashionIQ_dataset/resized_image/$c | wc -l; done
3817   6346   5373
$ python -c "…check every id in split.{cat}.val.json exists as resized_image/{cat}/{id}.jpg…"
dress split 3817 missing jpg 0
shirt split 6346 missing jpg 0
toptee split 5373 missing jpg 0
$ ls data/fashionIQ_dataset/captions/
cap.{dress,shirt,toptee}.{train,val,test}.json   correction_dict_{dress,shirt,toptee}.json
```

and the bytes are the pinned repository's own files, not an invented substitute:

```
$ sha256sum workbench/third_party/LIMN/correct_files/correction_dict_dress.json \
            workbench/third_party/TGCIR/correction_dict/correction_dict_dress.json \
            data/fashionIQ_dataset/captions/correction_dict_dress.json
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  LIMN/correct_files/…
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  TGCIR/correction_dict/…
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  data/fashionIQ_dataset/captions/…
$ cmp LIMN/correct_files/correction_dict_dress.json data/fashionIQ_dataset/captions/correction_dict_dress.json
byte-identical
(shirt ca29e29e… and toptee 3dfbf15f… likewise identical across all three locations)
```

BUT: `workbench/docs/AUXILIARY_ASSET_AUDIT.md:358-360` still lists the three `intent` dictionaries as
`OFFICIAL_ASSET_UNAVAILABLE`, and `workbench/registry/assets/cluster_b3_ilearn.yaml:1145-1160`
records them as `classification: OFFICIAL_ASSET_UNAVAILABLE`,
`acquisition_status: BLOCKED`, `current_blocker: "Upstream repository neither tracks
captions/correction_dict_dress.json nor provides a download URL or generation script."` — the first
half is true of INTENT alone, the "no download URL" half omits that the bytes are already in the
workbench from two pinned siblings. `workbench/registry/preparation_contracts.yaml` has **no `intent`
entry** (`grep -n "intent" preparation_contracts.yaml` → zero hits; only clvc_net, dcnet, clip4cir,
tgcir, sprc, limn, pair), and `README.md:123` points at a `requirements.txt` that does not exist. So
the *data* is there, handed over by a sibling pinned checkout, but the *declaration* that would make
the preparation a supervised step is absent. Consequence: the dictionaries are present and
provenance-traceable **today**, but nothing in the workbench would re-create them from a clean root,
and no digest is pinned for them. Severity P2 (declaration), not P1 (availability).

`workbench/docs/READINESS_MATRIX.md:30` (generated 2026-10-14) still reports `intent` BLOCKED on
"resized_image … correction_dict_dress.json" paths; those paths now exist (staging mtime 2026-10-15
17:55). The matrix row is **stale**, not wrong about a live blocker.

---

## 7. FASHIONIQ PROTOCOL

Registry (`models.yaml:328-330`): `literature_split_label: original`,
`native_protocol: fashioniq_full_gallery_ref_excluded`,
`supported_protocols: [fashioniq_full_gallery_ref_excluded]`, and the protocol itself is declared at
`models.yaml:12-15` as *"FashionIQ - Original Full Gallery, Reference Excluded"*, i.e.
gallery = full validation split, reference = excluded, evaluation noise = 0.

**What the source actually evaluates** (`datasets.py:26`, `:166-210`; `test.py:141-174`):

* Split files loaded, quoted exactly — `datasets.py:32` `self.split_dir = self.path + 'image_splits'`
  and `datasets.py:168`:
  `with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name, 'val')), 'r') as f:`
  → concrete files `image_splits/split.dress.val.json`, `split.shirt.val.json`,
  `split.toptee.val.json`. Captions: `datasets.py:170` `cap.{name}.val.json`;
  dictionaries: `datasets.py:172` `correction_dict_{name}.json`.
* Gallery — full split, file order, no dedup:
  `datasets.py:203-205` `elif self.split == 'original-split': for id, image_name in enumerate(images):
  test_targets_id.append(id)`, and `:207-208` builds one gallery entry per id with
  `self.get_img(name + '_' + image_name)`. `split='original-split'` is the constructor default
  (`datasets.py:26`) and the CLI default (`train_INTENT.py:61 --split default='original-split'`). The
  alternative `val-split` branch (`datasets.py:191-202`, the pair-union gallery) is **dead** on this
  path.
* Reference policy — masked, not removed: `test.py:150`
  `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, executed before
  `test.py:153` `np.argsort(-sims[i,:])[:50]`. Under `original-split`, `test_targets_id` is the
  identity map (`datasets.py:205`), so this is exactly "the query's own reference cannot be retrieved".
* Recall — `test.py:167-174`: `for k in [1, 10, 50]`, hit iff
  `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, denominator
  `len(nn_result)` = query count. Per category, no macro.
* Categories — `train_INTENT.py:283` `for ci, category in enumerate(['dress','shirt','toptee'])`.

**Does it match the declared protocol?** Yes on all four axes (full gallery / reference excluded /
`original` literature label / 0 % evaluation noise). **VERIFIED**, with one arithmetic VARIANT — see §9
D-INT-4 (mask-then-sort vs sort-then-remove) and one *typing* nuance: the workbench's
`fashioniq_full_gallery_ref_excluded` is declared at `src/evaluation/fashioniq.py:175-177`
(`gallery="full_split", reference_excluded=True`) and is the same cohort the source builds. Sibling
methods really do share it: HABIT `datasets.py:28` (`split='original-split'`), ConeSep `datasets.py:28,201`
and AirKnow `datasets.py:31,218` (hard-coded `self.split='original-split'`), all with the
`enumerate(images)` full-split loop — so AirKnow's registry line
`workbench/registry/models.yaml:121` (`fashioniq_full_gallery_ref_excluded`) is **correct** if
`datasets.py:218` is read as the `get_test_data` scope, and the claim in the assignment brief that
AirKnow uses the val-split gallery is **not** what the pinned code shows.

Quantitative cohort separation (observed, from the staged data):

```
dress : full-split gallery=3817  pair-union gallery=2628  queries=2017
shirt : full-split gallery=6346  pair-union gallery=3089  queries=2038
toptee: full-split gallery=5373  pair-union gallery=2902  queries=1961
```

So labelling INTENT `full_gallery_ref_excluded` and HINT `val_split` is a real distinction, not a
filing convention: the two cohorts differ by ~1.5–2× in gallery width, which changes every rank.

**Query construction.** `datasets.py:175-187`: `mod_str = self.concat_text(caption['captions'], correction_dict)`
(`:177`), `out['source_img_id'] = images.index(candidate)` (`:181`),
`out['target_img_id'] = images.index(target)` (`:183`); captions normalised at `datasets.py:110-119`
(lowercase → `string.punctuation`→space → token split → exact dictionary lookup → `"{} and {}"`
join).

**Metric.** `test.py:167-174`. The workbench's macro (`src/evaluation/fashioniq.py:238-255`,
`"mean_recall": (recall_at_10 + recall_at_50)/2`) is a **workbench aggregate**, and the registry's
`reported_mean: 65.24` matches the paper's own AVG cell: `(55.31+75.16)/2 = 65.235 → 65.24` (observed
arithmetic). Per-category registry values 55.31 / 75.16 reconcile with the macro of the paper's
columns.

---

## 8. WORKBENCH INTEGRATION

| element | state | evidence |
|---|---|---|
| adapter | `IntentAdapter` exists, `script = "test.py"`, **no `command()` override** | `workbench/backend/adapters/models.py:69-71` |
| inherited behaviour | raises | `models.py:24-25` `raise NotImplementedError("upstream evaluation command requires audit before execution")` |
| observed | executed | `IntentAdapter().command(None,None,None)` → `NotImplementedError: upstream evaluation command requires audit before execution`; `'command' in IntentAdapter.__dict__` → `False` |
| planner | env group `cir-40c39aa201822895`, `SOLVER_COMPATIBLE` | `workbench/docs/READINESS_MATRIX.md:30` |
| checkpoint prep | none; no local artifact | `ls workbench/artifacts/checkpoints/intent` → `No such file or directory` |
| preflight | would report the layout paths as missing | `workbench/backend/fashioniq_layout.py:25-36`; matrix row stale (§6) |
| metric parser | none needed — the native path emits no structured output | `test.py:174` returns a Python list of tuples to `logging.info` |
| per-query support | not implemented for this model | no rank export in the pinned `test.test`; `test.py` never writes a file |
| run path | `run_eval.py:25` calls `build_command`, so `--model intent` cannot produce a command | `workbench/scripts/run_eval.py:14-25` |

---

## 9. CONFIRMED DEFECTS

**D-INT-1 (P0, REFUTES A PRIOR AUDIT) — the pinned evaluator's similarity step works; it does NOT
raise a shape error.**
The earlier `workbench/docs/reviews/models/intent.md:128,217` recorded a P0
"`np.matmul` shape failure" for `(B,256,1,1) @ (B,256,32)`. That is **wrong about which operand is
last**. `blip2_cir.py:769` is `return fusion_feats.unsqueeze(1).unsqueeze(1)` on a `(B, 256)` tensor,
which appends the new axes at the END: `(B, 256, 1, 1)`. `blip2_cir.py:792` is
`return image_features.permute(0, 2, 1)` on `(B, 32, 256)` → `(B, 256, 32)`. Contracting
`(B,256,1,1)` with `(B,256,32)` treats operand 0 as `(N,256) @ (256,32)` and **broadcasts** over the
batch axis. Executed:

```
$ .venv/bin/python -c "import numpy as np; q=np.zeros((2017,256,1,1)); g=np.zeros((3817,256,32));
                       s=np.matmul(q,g); print(s.shape, s.squeeze().shape, s.squeeze().max(-1).shape)"
(2017, 3817, 1, 32) (2017, 3817, 32) (2017, 3817)
$ torch: fusion=torch.randn(4,256); q=fusion.unsqueeze(1).unsqueeze(1) -> torch.Size([4, 1, 1, 256])
        matmul -> (4,5,1,32) -> squeeze -> (4,5,32) -> max(-1) -> (4,5); sims[0,1] = 29.401709
```

So `test.py:141-143` yields exactly `(queries, gallery)` and `test.py:150`'s
`sims[i, test_targets_id.index(...)]` indexing is well-formed. The rank is `max` over the 32 Q-Former
target tokens, which is what the code comment at `test.py:141` (`#all_queries.dot(all_imgs.T)`)
describes. The registry already carries this retraction at `models.yaml:306`. Root cause of the
earlier error: reading `unsqueeze(1)` as a *prepend*. **Recommended action:** the earlier finding is
withdrawn; no patch is needed and none must be written for this reason. The task brief's own phrasing
of this claim is likewise refuted.
Reproduction: the two commands above. Regression result: n/a (no code change).

**D-INT-2 (P1) — no runnable evaluation path.** `test.py` is a callable with no CLI
(`grep -n "__main__\|argparse" test.py` → empty), the only caller is the training loop
(`train_INTENT.py:284`), and the loop has no restore path (`grep -rn "restore_file\|load_checkpoint"
train_INTENT.py` → empty). `IntentAdapter` refuses by construction (`models.py:24-25`, observed
`NotImplementedError`). Classification: **FIRST_PARTY_ADAPTER_MISSING** (workbench side) compounded by
**UPSTREAM_CLI_MISSING** (upstream side). Fix: an approved, digest-pinned observation-only wrapper that
(a) constructs `Blip2QformerCir`, (b) loads the registered checkpoint as a whole pickled module with
`torch.load(..., weights_only=False)` — the trainer saved a module, not a `state_dict`
(`train_INTENT.py:339,384`) — and (c) calls `test.test(args, model, FashionIQ(...), category, txt_processors)`
per category, then the workbench macro. That wrapper's source must be reviewed and pinned like
`approved_source_patches` (`models.yaml:88-95` CSMCIR precedent), because a new first-party file
changes what was executed.

**D-INT-3 (P1) — no clean 0 % checkpoint exists, so the paper row is not reproducible from published
weights.** `strict_clean_checkpoint_available: false` (`models.yaml:335`) is correct; observed Hub
siblings are only `intent_cirr_0.2.pt` + the three noise-trained FashionIQ files. Classification:
**CHECKPOINT_UNPUBLISHED**. Fix: none available from the workbench. Accept explicitly that a 20/50/80 %
checkpoint is a *different model* from the 0 % row, or obtain author weights. Never present a
noise-trained file as the 0 % result.

**D-INT-4 (P3) — reference-exclusion arithmetic differs while the cohort matches.** Source masks
`-10e10` then sorts (`test.py:150,153`); the workbench removes the column after sorting
(`src/evaluation/fashioniq.py:69-73`, `ranking = ranking[ranking != reference_column]`). Equivalent
for the top-k hit test whenever every other score clears the mask, which the workbench states as its
own justification (`src/evaluation/fashioniq.py:46-52`). Classification: **SCIENTIFIC_PROTOCOL_UNVERIFIED**
only in the strict sense that the two are not the same operation; the *cohort* is identical.
Classification of the difference: VARIANT, not a defect. Fix: none; never claim numeric identity of
the two implementations.

**D-INT-5 (P2) — the declared environment cannot import the model package.** Proven twice above:
`torchaudio` missing, and with it stubbed,
`transformers 4.57.6` lacking `apply_chunking_to_forward` (`med.py:40-43`). Classifications:
**INSTALLABLE_DEPENDENCY_MISSING** + **DEPENDENCY_VERSION_CONFLICT** + **UPSTREAM_IMPORT_BUG** (the
audio/beats coupling is unconditional in `lavis/__init__.py:15` →
`processors/__init__.py:29`). The declared env (Python 3.9, `transformers==4.25.0`, `torchaudio==2.1.0`)
would satisfy all three; the venv here does not. Fix: build the declared env on the GPU host.
`torchaudio`/`moviepy` are **not** optional convenience — they are on the import path.

**D-INT-6 (P2) — the shipped checkpoint loader cannot read the shipped checkpoints.**
`utils.py:128-140` does `model.load_state_dict(checkpoint['state_dict'])`; the trainer writes whole
modules (`train_INTENT.py:339,384` `torch.save(model, …)`). No caller exists anyway. Classification:
**UPSTREAM_IMPORT_BUG** (internal contract mismatch). Fix: wrapper must own the load; do not "repair"
`utils.py`.

**D-INT-7 (P2) — undefined X / undeclared K.** The README documents
`python cirr_sub_BLIP2.py --checkpoint_path … --output_file …` (`README.md:188-190`) while
`cirr_sub_BLIP2.py` has **no argument parser** — it reads `sys.argv[1]`-style state and hard-codes
`/root/autodl-tmp/BLIP/test/` (`:122`) and `/root/autodl-tmp/cirr_data/CIRR/` (`:120`). Also
`test.py:178-200` (`lasco_query`/`lasco_target`) hard-codes `/root/autodl-tmp/LaSCo/lasco_val_queries.pkl`.
Documentation-only defect for a FashionIQ run.

**D-INT-8 (P2) — the training entry point the README documents is the one that cannot be used, and its
flag story is misleading.** `README.md:181` `python train_INTENT.py` runs 20 epochs before any metric
(`train_INTENT.py:49` `--num_epochs` default=20) and has no restore flag, so the README's "Quick Start"
never evaluates the released weights. `--noise_ratio` defaults to 0.2 (`train_INTENT.py:60`), and
`AUXILIARY_ASSET_AUDIT.md:203` claims a `--restore_file` argument that does not exist in this tree
(`grep` → 0 hits; it exists in HABIT/ConeSep/AirKnow, i.e. cross-model contamination).

---

## 10. REMAINING BLOCKERS

| # | classification | owner | evidence | remediation |
|---|---|---|---|---|
| B1 | **FIRST_PARTY_ADAPTER_MISSING** + **UPSTREAM_CLI_MISSING** | workbench + upstream | `models.py:24-25,69-71`; `test.py` no `__main__`; observed `NotImplementedError` | approved observation-only wrapper (§11) |
| B2 | **CHECKPOINT_UNPUBLISHED** | upstream/author | `models.yaml:335`; Hub siblings observed; paper row is the 0 % row | author must publish, or accept a noise-trained variant explicitly |
| B3 | **INSTALLABLE_DEPENDENCY_MISSING** / **DEPENDENCY_VERSION_CONFLICT** / **UPSTREAM_IMPORT_BUG** | workbench env + upstream | executed tracebacks; `audio_processors.py:9-13`; `med.py:40-43`; `lavis/__init__.py:15` | build declared env (py3.9 / torch 2.1.0 / transformers 4.25.0 / torchaudio 2.1.0 / moviepy) |
| B4 | **UNPUBLISHED_SOURCE_MODULE** — *no longer a live blocker* | upstream | dictionaries now present: sha256 `9e4a5987…/ca29e29e…/3dfbf15f…`, byte-identical to `LIMN/correct_files/` and `TGCIR/correction_dict/` | none for the run; declaration gap remains (B5) |
| B5 | **FIRST_PARTY_ENVIRONMENT_CONFIGURATION** (declaration) | workbench | `preparation_contracts.yaml` has no `intent` entry; `AUXILIARY_ASSET_AUDIT.md:358-360` stale | add an `intent_fashioniq` contract declaring the dictionary source dir + resized-image generator, and pin digests |
| B6 | **PUBLIC_ASSET_NOT_PREPARED** — partially stale | workbench | `READINESS_MATRIX.md:30` (2026-10-14) vs staging mtime 2026-10-15 | regenerate the matrix |
| B7 | **GPU_RUNTIME_UNTESTED** / **ACTUALLY_READY_PENDING_GPU** | host | `blip2_cir.py` imports cleanly only after B3; `test.py:110,127` call `.cuda()` unconditionally; `/root/bert-base-uncased` absent | run on the approved GPU host |
| B8 | no per-query export | workbench | pinned `test.test` writes no file | observation-only export after aggregate parity (see `workbench/docs/RESULT_SCHEMA.md:54`) |

No `EXTERNAL_BLOCKER` token is used anywhere in this report.

---

## 11. IMPLEMENTATION PLAN

1. **Environment** (B3). Create `WORKBENCH_INTENT_PYTHON` from `README.md:112-119` verbatim:
   Python 3.9, `torch==2.1.0`, `torchvision==0.16.0`, `torchaudio==2.1.0` (cu121),
   `transformers==4.25.0`, `scikit-learn==1.3.2`, `open-clip-torch==2.24.0`, `timm==0.9.16`,
   plus `moviepy` and `matplotlib` (both reachable, both absent from the README list). Record
   `interpreter_sha256` per `models.yaml:1-5`. Complexity low. Scientific risk none (it is the
   declared env). Expected: `from lavis.models import load_model_and_preprocess` succeeds.
2. **Prepare `/root/bert-base-uncased`** (B7) — mandatory, hard-coded at `blip2.py:32,48,55`.
   Complexity low; risk none.
3. **Wrapper** (B1). One first-party script, digest-pinned and registered like CSMCIR's
   `approved_source_patches` (`models.yaml:88-95`), that:
   `load_model_and_preprocess("Blip2QformerCir","pretrain",is_eval=True,device=...)` →
   `torch.load(checkpoint, weights_only=False)` (whole module; verified this is what
   `train_INTENT.py:339,384` writes) → `datasets.FashionIQ(path=<root>/ , transform=targetpad(1.25,224),
   split='original-split')` → `test.test(args, model, testset, cat, txt_processors)` for
   `['dress','shirt','toptee']` → macro. **The wrapper must not touch gallery construction, the
   reference mask, the caption normalisation or the K-set.** Complexity medium; risk: any deviation
   changes the cohort. Expected: three per-category `{r1,r10,r50}` sets.
4. **Declaration** (B5). Add `intent_fashioniq` to `preparation_contracts.yaml` naming
   `resized_image/{cat}` (generator `fashioniq_resize_images`) and the correction dictionaries with
   `source_directory` pointing at the pinned LIMN/TGCIR copies + their sha256. Complexity low,
   risk none, and it converts a silent hand-placed asset into a reproducible step.
5. **Do not** patch `blip2_cir.py`. There is no shape defect to fix (D-INT-1).
6. **Do not** change the gallery, the reference policy, the caption input or the noise condition.
   Protocol A/B/C invariants are untouched by this plan.

---

## 12. FINAL VERDICT

**BLOCKED** — with the source defect claim **refuted**.

* Protocol, gallery, reference policy, K-set and preprocessing all trace to the pinned source and
  match `fashioniq_full_gallery_ref_excluded` (**VERIFIED**, one arithmetic VARIANT).
* The pin is exact and current upstream **equals** the pin (SNAPSHOT 1 == SNAPSHOT 2).
* The previously recorded P0 similarity shape failure **does not exist**; the executed `np.matmul`
  broadcasts to `(M,N)` and the ranking indexing is well-formed (D-INT-1, REFUTED).
* The native path still cannot run: no CLI, no restore path, no adapter command, no clean 0 %
  checkpoint, and the declared environment cannot import its own model package.
* Nothing was reproduced and no metric was observed. `reproduction_status: NOT_RUN`
  (`models.yaml:334`) remains correct, and no evidence in this audit upgrades it.

---

## 13. EVIDENCE

* repo: `https://github.com/iLearn-Lab/AAAI26-INTENT` (`models.yaml:321`); remote also reports
  `iLearn-Lab/INTENT` aliasing (`README.md` link forms).
* pinned commit: `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` — `git -C workbench/third_party/INTENT rev-parse HEAD`.
* SNAPSHOT 2: `git -C workbench/third_party/INTENT ls-remote origin` → HEAD == pin;
  `git log --oneline --all` → one commit `2a75fc6 Update README.md`; `git rev-list --count HEAD` → 1.
* key files and digests observed:
  `INTENT/datasets.py` sha256 `38d185c126b3b8027b3d9bb63eb8aa3d308efd4537c76e3a0b12646801d6a176`;
  `INTENT/lavis/models/blip2_models/blip2_cir.py` sha256 `f10061620d795af57dc57b92f83152e283e59a6a34d48a47b94c9b3c2eba15f1`.
* protocol: `datasets.py:26,32,166-210`; `test.py:141-143,146-153,167-174`;
  `models.yaml:305,328-330,337-367`; workbench `src/evaluation/fashioniq.py:35-73,132-136,171-183,238-255`,
  `src/datasets/fashioniq.py:52,84,93,119,186-221`.
* checkpoint: `curl -s "https://huggingface.co/api/models/iLearn-Lab/AAAI26-INTENT?blobs=true"`;
  `curl -sIL https://huggingface.co/iLearn-Lab/AAAI26-INTENT/resolve/main/intent_fiq_0.2.pt` → `200`.
* loader: `INTENT/utils.py:128-140`; `INTENT/train_INTENT.py:339,384`; `INTENT/cirr_sub_BLIP2.py:120-128`.
* dependencies: executed `ModuleNotFoundError: No module named 'torchaudio'` and, after stubbing,
  `ImportError: cannot import name 'apply_chunking_to_forward'`; venv versions `transformers 4.57.6`,
  `torch 2.14.0+cpu`.
* adapter: executed `IntentAdapter().command(None,None,None)` → `NotImplementedError:
  upstream evaluation command requires audit before execution`.
* data: `ls`/`sha256sum`/`cmp` over `data/fashionIQ_dataset/` and `workbench/third_party/LIMN/correct_files/`.
* no checkpoint bytes were downloaded; `workbench/artifacts/checkpoints/intent` does not exist.

---

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 — static source/protocol/asset audit with executed import and shape probes; no environment
was built and no inference was run.**

* Static reading of the pinned tree: VERIFIED.
* Pin identity and SNAPSHOT-2 comparison: VERIFIED (read-only `ls-remote`).
* Protocol/gallery/reference/K-set traceability: VERIFIED (one VARIANT on exclusion arithmetic).
* Two import-path failures reproduced by execution: VERIFIED.
* Shape claim (D-INT-1): REFUTED by execution.
* Checkpoint metadata: VERIFIED as metadata; payload identity UNKNOWN.
* Environment build, weight load, GPU inference, metric reproduction: **NOT RUN**.
* No claim above is RUNTIME_VERIFIED.
