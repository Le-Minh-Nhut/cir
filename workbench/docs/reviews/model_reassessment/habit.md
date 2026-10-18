# habit — reproducibility reassessment

Independent adversarial review, read-only, at workbench HEAD `b70d092e9d2232e34886f819e30c7327b74f3a76`
(`git rev-parse HEAD`). No GPU run, no download of weights, no install, no repo write.
Every claim is **STATIC_AUDIT** unless the line says RUNTIME_VERIFIED. Unproven = **UNKNOWN**.

The prior audit `workbench/docs/reviews/models/habit.md` is treated as a **hypothesis**. Where my reading
disagrees, the code is shown.

---

## 1. MODEL IDENTITY

| Field | Value | Evidence |
|---|---|---|
| method | HABIT | `workbench/registry/models.yaml:252` |
| paper | *HABIT: Chrono-Synergia Robust Progressive Learning Framework for Composed Image Retrieval* | `models.yaml:256` |
| paper URL | `https://ojs.aaai.org/index.php/AAAI/article/view/37608` | `models.yaml:257` |
| year / venue | 2026 / AAAI 2026 | `models.yaml:254`; `HABIT/README.md:25,44` |
| source URL | `https://github.com/iLearn-Lab/AAAI26-HABIT` | `models.yaml:258` |
| pinned commit | `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | `models.yaml:260` |
| checkout HEAD | `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` | `git -C workbench/third_party/HABIT rev-parse HEAD` |
| checkout remote | `git@github.com:iLearn-Lab/AAAI26-HABIT.git` | `git -C workbench/third_party/HABIT remote -v` |
| native protocol | `fashioniq_full_gallery_ref_excluded` | `models.yaml:266` (authoritative, quoted verbatim) |
| checkpoint variants | `fiq_n02` / `fiq_n05` / `fiq_n08`, all noise-trained | `models.yaml:274-304` |
| original metrics | R@10 55.38, R@50 75.20, mean 65.29 | `models.yaml:268-270` |
| reproduction status | `NOT_RUN` | `models.yaml:271` |

Shallow single-commit checkout (`git rev-list --count HEAD` → `1`), so history-based claims are **UNKNOWN**.

## 2. SOURCE STATUS

Accessible and complete for the FashionIQ path: 2,148 non-vendored lines
(`cirr_test_submission.py 128, data_utils.py 427, datasets.py 390, test.py 264, train.py 368, utils.py 144`)
plus the vendored LAVIS fork. No unpublished first-party module is imported.

First-party inconsistency observed (P3): `HABIT/README.md:243` and `:267` document
`python src/cirr_test_submission.py ...`, but there is no `src/` directory, the file is
`cirr_test_submission.py` at the repo root, and `cirr_test_submission.py:97` sets
`blip_model_name = "Blip2QformerCir"` — a name **not registered** in the pinned LAVIS
(`grep -rn "register_model(" lavis/models/blip2_models/*.py` lists `HABIT,
blip2_image_text_matching, blip2_opt, blip2, blip2_feature_extractor, blip2_t5, blip2_t5_instruct,
blip2_vicuna_instruct, blip2_vicuna_xinstruct` — no `Blip2QformerCir`). The documented CIRR test command
cannot run as printed: it would raise at `registry.get_model_class("Blip2QformerCir")`
(`lavis/models/__init__.py:193`).

### SNAPSHOT 2 — current upstream state

```
$ git -C workbench/third_party/HABIT ls-remote origin HEAD
6ddc2cd63eac1f65693f4e821c34fb3b42aee306	HEAD
$ timeout 60 git ls-remote https://github.com/iLearn-Lab/AAAI26-HABIT
6ddc2cd63eac1f65693f4e821c34fb3b42aee306	HEAD
6ddc2cd63eac1f65693f4e821c34fb3b42aee306	refs/heads/main
$ curl -sS -o /dev/null -w '%{http_code} %{redirect_url}' https://github.com/Lee-zixu/HABIT
301 https://github.com/iLearn-Lab/AAAI26-HABIT
```

No move outstanding: the registry already points at the mirror, `main` equals the pin, and the older
`Lee-zixu/HABIT` name redirects into it. (Contrast `conesep`, whose registry URL is still the redirecting
one.) **The authors added no FashionIQ CLI, no checkpoint loader, no clean checkpoint and no dependency fix
between the pin and current `main` — the two are the same commit.**

## 3. NATIVE EVALUATION WORKFLOW — the OFFICIAL FashionIQ evaluation function

Entrypoint: `HABIT/test.py` defines exactly three callables (`load_obj`, `test`, `test_cirr_valset`); no
`argparse`, no `__main__`. The FashionIQ path of the official evaluator, quoted
(`HABIT/test.py:9-10,80-87,97,134-175`):

```python
def test(params, model, testset, category, txt_processors):
    model.eval()
    if category == 'fashion200k':
        ...
    else:
        if category == 'dress':
            (test_queries, test_targets, name) = (testset.test_queries_dress, testset.test_targets_dress, 'dress')
        elif category == 'shirt':
            ...
        elif category == 'toptee':
            ...
        elif category == 'shoes':
            ...
        elif category == 'lasco':
            (test_queries, test_targets) = testset.get_val_queries(); name = "lasco"
        elif category == 'birds':
            test_queries = testset.get_test_queries(); test_targets = testset.get_test_targets(); name = 'birds'

        with torch.no_grad():
            ...
                    f = model.extract_retrieval_compose(imgs, mods)   # test.py:112
                    imgs = model.extract_retrieval_target(imgs).data.cpu().numpy()   # test.py:128
        # feature normalization   <-- COMMENTED OUT (test.py:134-140)
        # for i in range(all_queries.shape[0]):
        #     all_queries[i, :] /= np.linalg.norm(all_queries[i, :])
        sims = np.matmul(all_queries, all_imgs)        # test.py:141
        sims = sims.squeeze()                          # test.py:142
        sims = sims.max(-1)                            # test.py:143

        test_targets_id = []
        for i in test_targets:
            test_targets_id.append(i['target_img_id'])
        for i, t in enumerate(test_queries):           # test.py:150-151
            sims[i, test_targets_id.index(t['source_img_id'])] = -10e10

        nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]   # test.py:154
        out = []
        else:
            nn_result = [np.argsort(-sims[i, :])[:50] for i in range(sims.shape[0])]   # test.py:168
            for k in [1, 10, 50]:                                                     # test.py:169
                r = 0.0
                for i, nns in enumerate(nn_result):
                    if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]:
                        r += 1
                r = 100 * r / len(nn_result)                                          # test.py:174
                out += [('{}_r{}'.format(name, k), r)]
        return out
```

Function chain a wrapper must call:

```
python train.py --dataset fashioniq --fashioniq_path <root>/ ...        # HABIT/README.md:211-219
  └─ train.py:__main__ (train.py:331)                                   # no checkpoint load
       ├─ train.py:344-351  seeding (random/torch/cuda/numpy, cudnn.deterministic)  [P2, see Section 9]
       ├─ train.py:358  trainset = load_dataset()  (train.py:91)
       │        → datasets.FashionIQ(path, transform=targetpad(1.25,224), noise_ratio=args.noise_ratio)
       │             (train.py:109-112; split defaults to 'original-split' at datasets.py:28)
       │        → trainset.shuffle()                                                (train.py:112)
       ├─ train.py:362  model, optimizer, txt_processors = create_model_and_optimizer()  (train.py:176-187)
       │        load_model_and_preprocess(name="HABIT", model_type="pretrain")  → lavis/models/__init__.py:174
       │        → HABIT.from_config → HABIT.__init__ (HABIT.py:125) → init_tokenizer (blip2.py:31-34)
       └─ train.py:364  train_and_evaluate()  (train.py:253)
              for category in ['dress','shirt','toptee']:
                  t = test.test(args, model, trainset, category, txt_processors)   # train.py:283
                  current_score = current_score + t[1][1]                          # train.py:285
```

CLI: **no FashionIQ evaluation CLI exists.** `test()` is a subroutine of the training loop only. Best-checkpoint
selection is `current_score = current_score + t[1][1]` (`train.py:285`) — i.e. **summed category R@10**, not the
paper's AVG column. The pinned source contains **no macro average** at all.

Checkpoint loader: absent from the launch path. There is no `--restore_file` argument in `HABIT/train.py` at
all (`grep -n restore_file HABIT/train.py` → no match). The only loader in the tree is the CIRR submission
script, which is broken (Section 2).

Shape algebra, verified in numpy at the real gallery sizes (`extract_retrieval_compose` → `(N,1,1,d)` at
`HABIT.py:413`; `extract_retrieval_target` → `(M,d,32)` at `HABIT.py:436`):

```
$ .venv/bin/python /tmp/shape2.py && cat /tmp/shape2.txt
shape=(2017, 3817)
mask write ok -> -99999997952.0
argsort row shape=(50,)
```

So `np.matmul → (N,M,1,32)`, `.squeeze() → (N,M,32)`, `.max(-1) → (N,M)` and the `-10e10` mask lands as a
scalar. Behaviourally correct for `N,M > 1`. (Degenerate single-element batches collapse the array to 1-D; no
real run has one.) Same algebra as `conesep`; the earlier "tensor-shape failure" claim that the registry
withdrew for `conesep` (`models.yaml:151`) would be equally wrong for `habit`.

## 4. DEPENDENCIES

D1–D8 are not defined in this task text, so I do not invent category numbers. All imports of the launch path
were resolved against the pinned tree and the workbench interpreter (`.venv`, Python 3.13.12, torch 2.14.0+cpu).

| import | file:line | owner | required version | installable? | py/torch/cuda | safe remedy | test result |
|---|---|---|---|---|---|---|---|
| `transformers` (`CLIPSegProcessor, CLIPSegForImageSegmentation`) | `datasets.py:13` | PyPI | `4.25.0` (`models.yaml:251`) | **no** at the declared pin | none | `transformers==4.26.0` | **RUNTIME_VERIFIED by HTTP**: `transformers/v4.25.0/src/transformers/models/clipseg/__init__.py` → **404**; `v4.26.0` (and `v4.27.0`, `v4.28.0`, `v4.29.0`, `v4.40.1`) → **200**. CLIPSeg landed in transformers commit `Add CLIPSeg (#20066)`, 2022-11-08; `v4.26.0`'s `clipseg/__init__.py:30,45` exports exactly both names |
| `transformers` (`CLIPTextModel, CLIPVisionModelWithProjection, CLIPImageProcessor, AutoTokenizer`) | `train.py:35` | PyPI | `4.25.0` (`models.yaml:251`) | yes at 4.25.0; the CLIPSeg line forces ≥4.26.0 anyway | none | as above | import-time hard requirement of `train.py`; the four names exist at 4.25.0 |
| `transformers` (`BertTokenizer`) | `lavis/models/blip2_models/blip2.py:26` | PyPI | as declared | yes | none | pin | standard |
| `open-clip-torch==2.24.0` | declared `models.yaml:251`; **not imported** by the pinned source | registry | 2.24.0 | yes | none | none | declared, unused |
| `salesforce-lavis==1.0.2` | declared `models.yaml:251`; **not imported** | registry | 1.0.2 | yes | none | none | the run uses the vendored `lavis/`, never the PyPI package |
| `omegaconf` | `lavis/__init__.py:11`, `lavis/models/base_model.py:16` | LAVIS fork (in-tree) | any | yes | none | pip | absent from `.venv`; blocks `import lavis` |
| `torch`, `torchvision` | `train.py:11,15`; `datasets.py:10` | PyPI | `2.1.0` / `0.16.0` (`models.yaml:248-249`); README:83 uses **cu121** wheels | yes | torch+cuda | pin | `.venv` torch is 2.14.0+cpu — the declared torch is not installed and no model env exists |
| `scikit-learn` (`DBSCAN`) | `train.py:9` | PyPI | `1.3.2` (`models.yaml:251`) | yes | none | pip | used in the training loss path (`--eps_for_DBSCAN` at `train.py:85`), not in `test.py` |
| `clip` (OpenAI) | `train.py:19` | PyPI | any | yes | none | pip | import-time hard requirement; unused on the eval path |
| `setproctitle` | `train.py:33` | PyPI | any | yes | none | pip | import-time hard requirement |
| `timm` | declared `models.yaml:251`; not imported | registry | 0.9.16 | yes | none | none | declared, unused |
| `numpy`, `PIL`, `tqdm`, `typing_extensions` | `test.py:1-5`, `datasets.py:3-14`, `data_utils.py:3` | PyPI | any | yes | none | pip | standard |

BERT dependency and the repo's own tree — see Section 6: the same hardcoded `/root/bert-base-uncased`
(`HABIT/lavis/models/blip2_models/blip2.py:32,48,55`, byte-identical across ConeSep/HABIT/INTENT —
`sha256 6a051a475a2f6043cce020b65216b4fcf53916d6a1f87643668cf63bc936a466` for all three) has the same status here
as for ConeSep: **the path is permission-denied on this host, so the tree does not resolve it, and both the
resolution and the artifact are UNKNOWN at the byte level.**

## 5. CHECKPOINTS

| checkpoint_id | filename | training noise | eval noise | size | sha256 | status |
|---|---|---|---|---|---|---|
| `fiq_n02` | `HABIT-FIQ_N0.2.pt` | 20 (`:278`) | 0 (`:279`) | `2730715930` (`:281`) | `814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03` (`:282`) | `NO_CLEAN_CHECKPOINT` (`:283`) |
| `fiq_n05` | `HABIT-FIQ_N0.5.pt` | 50 (`:288`) | 0 (`:289`) | `2730715930` (`:291`) | `8a0d3bd27cf46d8334f9596404d89ebf803da0fa65236d89815bb3929b9a4216` (`:292`) | `NO_CLEAN_CHECKPOINT` (`:293`) |
| `fiq_n08` | `HABIT-FIQ_N0.8.pt` | 80 (`:298`) | 0 (`:299`) | `2730715930` (`:301`) | `763885de6907bca4f017b336ed7ec3740cb9305c5adb6a135da45c158ddb9717` (`:302`) | `NO_CLEAN_CHECKPOINT` (`:303`) |

All three `source_type: author_huggingface`, `checkpoint_mapping_status: VERIFIED_METADATA` (`:284,:294,:304`).

Independently verified by HTTP HEAD (observed), one line per file:

```
HABIT-FIQ_N0.2.pt => HTTP/2 302 ; HTTP/2 200  x-linked-size: 2730715930  x-linked-etag: "814d2d0ceb29ef910f37562e8bea8a76cf38aa546fbc2102e495df687c276b03"
HABIT-FIQ_N0.5.pt => HTTP/2 302 ; HTTP/2 200  x-linked-size: 2730715930  x-linked-etag: "8a0d3bd27cf46d8334f9596404d89ebf803da0fa65236d89815bb3929b9a4216"
HABIT-FIQ_N0.8.pt => HTTP/2 302 ; HTTP/2 200  x-linked-size: 2730715930  x-linked-etag: "763885de6907bca4f017b336ed7ec3740cb9305c5adb6a135da45c158ddb9717"
$ curl -sS https://huggingface.co/api/models/iLearn-Lab/AAAI26-HABIT
sha 8712993b071f4668ce5b2db6070e70c856bd9174
siblings: .gitattributes README.md cirr/HABIT-CIRR-N0.2.pt cirr/HABIT-CIRR-N0.5.pt cirr/HABIT-CIRR-N0.8.pt fiq/HABIT-FIQ_N0.2.pt fiq/HABIT-FIQ_N0.5.pt fiq/HABIT-FIQ_N0.8.pt
```

**Mapping status: VERIFIED, and the registry is correct in the strongest sense** — the three recorded
`expected_size_bytes` and `expected_sha256` values are **exactly** the Hub's `x-linked-size` and `x-linked-etag`
for the same three filenames. This was checked per file, not inferred. (Contrast `conesep`, where both fields are
`null`.)

Local staging: `workbench/artifacts/checkpoints/habit/` **does not exist**; the only downloaded checkpoint
family on disk is `limn` (`workbench/artifacts/checkpoints/limn/`, three files, recorded in
`workbench/artifacts/checkpoints/download_manifest.json`).

Loader contract: the payload is a **whole-model `torch.save(model, ...)` pickle of a mid-training epoch**, not
a state dict. Observed by range fetch:

```
$ curl -sL -r 0-600000 .../fiq/HABIT-FIQ_N0.2.pt -o hb.bin
$ strings -n 4 hb.bin | grep -E "^/(mnt|root|home|data)|model_epoch|clavis|BertTokenizer" | sort -u
/mnt/data/PreTrain/bert-base-uncasedr-A
model_epoch_3/data.pklFB
clavis.models.blip2_models.HABIT
clavis.models.blip2_models.Qformer
transformers.models.bert.tokenization_bert.BertTokenizer
```

Same two defects as `conesep`, both P1: the pickle references the **training host's** tokenizer directory
`/mnt/data/PreTrain/bert-base-uncased` (not the source's `/root/bert-base-uncased`), and it is a whole model, so
`torch.load`'s default fails on torch ≥ 2.6. RUNTIME_VERIFIED on the workbench interpreter:

```
$ .venv/bin/python -c "... torch.load('/tmp/wt/thing.pkl') ..."
default weights_only: None
default torch.load custom class: UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`
weights_only=False: SUCCEEDED
```

The file name is additionally misleading: it is `model_epoch_3` (N0.2 and N0.5) and `model_epoch_2` (N0.8) —
an epoch model, while the README (`HABIT/README.md:236`) says only `best_model.pt` is saved, and the Hub's own
model card tells the evaluator to `Rename the downloaded HABIT-CIRR_N0.5.pt to best_model.pt`. Classification:
`UNSAFE_CHECKPOINT_SERIALIZATION`.

### The zero-noise checkpoint

**Declared variant: does NOT exist.** The Hub listing above contains exactly six weight files, all noisy. There
is no 0% file under any name. `strict_clean_checkpoint_available: false` (`models.yaml:272`) and
`notes: Official Hub lists only FashionIQ 20%, 50%, and 80% noise-trained weights.` (`models.yaml:273`) are
**truthful**.

`workbench/registry/paper_benchmarks/papers.yaml:845-850` registers the paper's 0% row as
`habit_aaai26_table1_fiq_n00` with `checkpoint_id: null` and `checkpoint_mapping_status: NO_CHECKPOINT` — the
honest encoding. The registerable rows are the noisy ones:
`habit_aaai26_table1_fiq_n20` (`papers.yaml:903`, `checkpoint_id: fiq_n02`,
`checkpoint_mapping_status: VERIFIED`, `reference_status: VERIFIED`, `review_status: PENDING`) with
`reported_metrics: {mean: 64.38, r10: 54.48, r50: 74.28}` and category cells
`dress 49.63/71.34, shirt 55.67/73.19, toptee 58.14/78.32`; the 0% row carries
`dress 49.99/72.38, shirt 56.62/74.68, toptee 59.51/78.53, average 55.38/75.20, AVG 65.29`.

### What upstream's training loop implies about noise

Quoted (`HABIT/datasets.py:65-88`):

```python
    def shuffle(self):
        logging.info(f'Shuffle data with noise_ratio {self.noise_ratio}.')
        if self.noise_ratio == 0:
            logging.info("无噪声模式")
            return

        num_samples = len(self.fashioniq_data)
        shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
        par_p1 = int(len(shuffle_indices) * (1/3))
        par_p2 = int(len(shuffle_indices) * (2/3))
        shuffle_candidate_indices = shuffle_indices[:par_p1]
        shuffle_captions_indices = shuffle_indices[par_p1:par_p2]
        shuffle_target_indices = shuffle_indices[par_p2:]
        ...
```

and its caller (`HABIT/train.py:109-112`):

```python
        trainset = datasets.FashionIQ(
            path=args.fashioniq_path,
            transform=img_transform,
            noise_ratio=args.noise_ratio)
        trainset.shuffle()
```

with `parser.add_argument('--noise_ratio', type=float, default=0.8, ...)` (`train.py:73`). Implications:

1. **`noise_ratio = 0` is a real, first-class path in the code, not an error case**: `shuffle()` returns
   immediately (`datasets.py:67-69`), so the training triplets are untouched. A 0% model is *trainable* by this
   source. The README states the evaluated options were `0.0, 0.2, 0.5, 0.8` (`HABIT/README.md:206`), so the paper's
   0% row came from a real run — but the weights were not released.
2. **Noise is injected at *training* time only.** `shuffle()` is called from `load_dataset()` on `fashioniq`
   (`train.py:112`) and never from `test.py` (`grep -rn shuffle HABIT/test.py` → no match). `test.py:10` calls
   `model.eval()` and the body runs under `torch.no_grad()`. The registered uniform `evaluation_noise_pct: 0` is
   therefore a correct evaluation-time statement, and a noisy checkpoint is a *provenance* fact about the
   weights (`checkpoint_training_noise_pct`), not a property of the protocol. That split is modelled correctly.
3. **The corruption is a union of field-level swaps, not "noise_ratio of fully corrupted triplets".** The
   sampled index set is partitioned into thirds — one third has its `candidate` swapped, one third its
   `captions`, one third its `target` — and the swapped values are drawn from other already-corrupted rows.
   So a row can end up with both a corrupted reference and a corrupted target, and 20% flags 20% of *rows* but
   the union of affected fields is larger than a third of the flagged rows.
4. **Reproducing a specific published checkpoint from scratch is not deterministic** — the partition uses
   `random.sample` on a shared global RNG that the main block seeds *after* the dataset is constructed
   (Section 9, `P2-1`), and `random.shuffle` on the three pools re-consumes it. Evaluating an already-published
   checkpoint is unaffected.

## 6. CORRECTION DICTIONARIES AND HARDCODED LEGACY PATHS

### Correction dictionaries

Needed: **yes, twice**, and one of the two is on the evaluation path:

```
$ grep -n correction_dict HABIT/datasets.py
100:            with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f)
101:                correction_dict = json.load(f)
105:                cap = self.concat_text(triplets['captions'], correction_dict)
112:    def correct_text(self, text, correction_dict):
115:        text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])
119:    def concat_text(self, captions, correction_dict):
120:        text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
174:        with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f)
175:            correction_dict = json.load(f)
179:            mod_str = self.concat_text(caption['captions'], correction_dict)
```

`:174-179` is inside `get_test_data` (`:168`), i.e. **the validation query captions are dictionary-corrected**.
Line numbers differ from the prior audit's `:96-101`/`:112-117` only by constant offset; the callsites are the same.

Present in the pinned tree: **no** (`find workbench/third_party/HABIT -name "correction_dict*"` → nothing).
Public from HABIT: **no** — no URL, no generator, no README mention (`grep -n -i correction HABIT/README.md` →
no match).

Content-resolved but not provenance-resolved, RUNTIME_VERIFIED:

```
$ sha256sum data/fashionIQ_dataset/captions/correction_dict_{dress,shirt,toptee}.json
9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda  .../correction_dict_dress.json
ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e  .../correction_dict_shirt.json
3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71  .../correction_dict_toptee.json
$ sha256sum workbench/third_party/TGCIR/correction_dict/correction_dict_*.json workbench/third_party/LIMN/correct_files/correction_dict_*.json
(all six) 9e4a5987… / ca29e29e… / 3dfbf15f…   # byte-identical to the staged copies, per category
```

No `habit` row in `workbench/registry/auxiliary_assets.yaml` (`grep -c habit` → `0`), and no `habit` row in
`workbench/registry/preparation_contracts.yaml` (`preparation_id` values are only
`clvc_net_fashioniq, dcnet_fashioniq, clip4cir_fashioniq, tgcir_fashioniq, sprc_fashioniq, limn_fashioniq,
pair_fashioniq`).

### Every absolute path the source expects — quoted

`HABIT/train.py` (defaults, all four):

```python
46:    parser.add_argument('--fashioniq_path', default="/root/autodl-tmp/fashion_iq_data/")
48:    parser.add_argument('--cirr_path', default="/root/autodl-tmp/cirr_data/CIRR/")
77:    parser.add_argument('--model_dir', default='/root/autodl-tmp/noisyCIR',
```

`HABIT/cirr_test_submission.py`:

```python
117:    testset = datasets.CIRR(path="/root/autodl-tmp/cirr_data/CIRR/",transform=preprocess)
```

`HABIT/lavis/models/blip2_models/blip2.py` (three sites, the tokenizer/Q-Former origin):

```python
32:        tokenizer = BertTokenizer.from_pretrained("/root/bert-base-uncased", truncation_side=truncation_side)
48:        encoder_config = BertConfig.from_pretrained("/root/bert-base-uncased")
55:            "/root/bert-base-uncased", config=encoder_config
```

`HABIT/lavis/models/blip_models/blip.py` (not on this model's path):

```python
28:        tokenizer = BertTokenizer.from_pretrained("/root/bert-base-uncased")
```

And the **fifth** absolute path is inside the published weights, not the source:
`/mnt/data/PreTrain/bert-base-uncased` (Section 5). That is the one the checkpoint actually recorded — the
pinned source's `/root/bert-base-uncased` is *not* the directory the authors trained with.

`train.py:47,49,50,51` (`--shoes_path`, `--birds_path`, `--Fashion200k_path`, `--lasco_path`) default to `""`.

The dataset root is also concatenated without a separator — `datasets.py:33`
`self.image_dir = self.path + 'resized_image'` — so `--fashioniq_path` **must** end with `/`. The README
supplies one (`HABIT/README.md:214` `"/path/to/FashionIQ/"`); a path without it silently reads a sibling
directory the layout checker (which uses `Path` joins, `workbench/backend/fashioniq_layout.py:25-34`) would
not flag.

## 7. FASHIONIQ PROTOCOL

`native_protocol: fashioniq_full_gallery_ref_excluded` (`models.yaml:266`) — **authoritative, quoted**.
`supported_protocols: [fashioniq_full_gallery_ref_excluded]` (`models.yaml:267`).
`literature_split_label: original` (`models.yaml:265`). Definition at `models.yaml:12-15`.

- **Gallery** — full ordered validation split. `datasets.py:28` defaults `split='original-split'`;
  `train.py:109-112` passes no `split`, so the default is taken; `get_test_data` (`:168`) reads
  `image_splits/split.<cat>.val.json` (`:170`) into `images` and takes the `original-split` branch
  (`:205-210`), appending every index in file order. Local counts confirm the full-split gallery:
  dress 3817, shirt 6346, toptee 5373 (all unique) against 2017 / 2038 / 1961 queries.
- **Reference policy** — excluded per query, mask before ranking (`test.py:150-151`, before the `argsort` at
  `:154`).
- **Query construction** — `concat_text` (`:119-120`) with `correct_text` (`:112-115`) on both captions, joined
  by `" and "` in source order, then `txt_processors["eval"]` (`test.py:111`).
- **Categories** — `dress`, `shirt`, `toptee` (`test.py:81-87`); `shoes`, `lasco`, `birds` exist but are other
  benchmarks (`:87-95`), and `train.py:281` iterates only the three.
- **Recall computation** — `for k in [1, 10, 50]` (`test.py:169`), hit test
  `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, `r = 100 * r / len(nn_result)`
  (`:174`). Per category; **no macro aggregation anywhere in the pinned source**
  (`train.py:285` uses `t[1][1]` = `dress_r10` for model selection).

**No change to gallery, reference policy, caption input or noise condition is proposed.**

## 8. WORKBENCH INTEGRATION

| stage | state | evidence |
|---|---|---|
| adapter | registered, `command()` declines | `adapters/models.py:64-66`, `:155`; observed `NotImplementedError: upstream evaluation command requires audit before execution` |
| planner | refuses | no `command_status` on the `habit` row; `preflight.py:201-202` appends `command not runnable: COMMAND_UNAUDITED` |
| checkpoint selection | blocked by design | observed `select_checkpoints` → `BLOCKED/BLOCKED_CHECKPOINT` (strict_clean), `BLOCKED/BLOCKED_CHECKPOINT_SELECTION` (include_flagged_conditions), `conditions={'fiq_n02':20,'fiq_n05':50,'fiq_n08':80}` |
| preflight (data) | **passes** | observed `model_requirements` → `blocks: []`, 15 paths, all present |
| preflight (official) | one blocker | observed `preflight_blockers` → `command not runnable: COMMAND_UNAUDITED` |
| environment | UNCONFIGURED | observed `verify_environment(...)` → `tier=UNCONFIGURED status=UNCONFIGURED reason=WORKBENCH_HABIT_PYTHON not set`; both with and without `require_cuda=True` |
| metric parser | none | `metrics_extraction.py:383-420` dispatches only limn/csmcir/combiner×2/sprc → `AGGREGATE_PARSER_UNAVAILABLE` |
| per-query support | none | `autopilot.py:628,841,960` hardcode `per_query_export_available: False`; `workbench/replay/` contains `limn.py`, `tgcir.py` only |

**Disagreement with two documents.**

1. `workbench/docs/READINESS_MATRIX.md:29` reports `habit` BLOCKED with the same 8 FashionIQ data blockers
   ("`resized_image` missing", "`correction_dict_*.json` missing"). That is **stale**: the file is dated
   2026-10-14 13:48 (`stat`), last touched by commit `9d35861b`; the resized images and dictionaries were staged
   on **2026-10-18 13:20** (`stat -c '%y' data/fashionIQ_dataset/resized_image`). The same preflight now returns
   **zero** data blockers. Do not carry those 8 lines forward as current.
2. `workbench/docs/reviews/models/habit.md:34-63` states the Hub sha256/size values "are not recorded in the
   registry (`expected_sha256: null`, `expected_size_bytes: null`)". That was true of an earlier registry
   revision; the registry at HEAD records **both** fields for all three `habit` variants
   (`models.yaml:281,291,301` / `:282,292,302`) and they match the Hub exactly (Section 5). The prior statement
   is now **wrong**.

## 9. CONFIRMED DEFECTS

**CF-1 (P1) — No runnable evaluation path for any published HABIT checkpoint.** Root cause: `test.py` is a bare
callable (no `__main__`), the training main block never loads a weight file, `HABIT/train.py` has no
`--restore_file` at all, and the only in-tree loader (`cirr_test_submission.py:94-126`) is CIRR-only and names
an unregistered model (`:97`). Reproduction: `ADAPTERS['habit'].command('s','c',None,None)` →
`NotImplementedError`; `grep -rn "register_model" lavis/models/blip2_models/*.py` contains no
`Blip2QformerCir`.

**CF-2 (P1) — The registry's pinned row is the 0%-noise row and no 0% checkpoint exists.** Root cause:
`models.yaml:268-270` transcribes the clean row; `models.yaml:274-304` registers only noisy variants.
Reproduction: the HF API listing has six files, all noisy. The honest encodings already exist
(`strict_clean_checkpoint_available: false` at `:272`; `NO_CHECKPOINT` at `papers.yaml:849`).

**CF-3 (P1) — `transformers==4.25.0` cannot satisfy `datasets.py:13`.** `CLIPSegProcessor`,
`CLIPSegForImageSegmentation` do not exist before 4.26.0. RUNTIME_VERIFIED by HTTP on the upstream tags
(404 at `v4.25.0`; 200 at `v4.26.0`). This is the `DEPENDENCY_VERSION_CONFLICT` for `habit`, and it fires
before any dataset code runs. Fix: `transformers==4.26.0` minimum.

**CF-4 (P1) — Published checkpoints are whole-model pickles pointing at a foreign BERT directory and are
unloadable with default `torch.load`.** Reproduction: `strings` on the fetched header shows
`/mnt/data/PreTrain/bert-base-uncasedr-A` and `model_epoch_3/data.pklFB`; RUNTIME_VERIFIED `torch.load` default
raises with the PyTorch-2.6 message and `weights_only=False` is required. Classification:
`UNSAFE_CHECKPOINT_SERIALIZATION`. Note the registry records a checkpoint digest but nothing about the
serialization contract or the embedded tokenizer path — a run can verify bytes and still fail at unpickle.

**CF-5 (P3) — Exact from-scratch reproduction is not established (and is not needed to evaluate a published
checkpoint).** `train.py:344-351` seeds `random`, `torch`, `torch.cuda` and `numpy` and sets
`cudnn.deterministic = True`; `trainset = load_dataset()` follows at `train.py:358`, so the `random.sample`
partition in `datasets.py:72` **is** seeded. What remains unseeded/unrecorded is (i) `DataLoader(..., shuffle=True,
num_workers=8)` worker seeding (`train.py:263-267` with `--num_workers` default 8, `train.py:36`
`torch.set_num_threads(2)`), and (ii) the AMP/GPU reduction order on the authors' hardware. No per-noise-setting
seed is recorded anywhere in the tree, so **whether a published checkpoint reproduces bit-exactly from the
pinned code is UNKNOWN**. This does not affect evaluating an already-published weight file. Classification:
`NONDETERMINISTIC_PREPROCESSING` only if a from-scratch run is attempted.

**CF-6 (P2) — The evaluator's own caption normalization is a different punctuation rule from the dataset's, and
the second stage's `pre_caption` punctuation set is narrower.** `correct_text` (`datasets.py:112-115`) maps
**all** of `string.punctuation` to spaces and lowercases; the LAVIS eval processor then applies
`BlipCaptionProcessor.pre_caption` (`lavis/processors/blip_processors.py:49-68`), which replaces only
`[.!"()*#:;~]` and truncates to 50 words. Both stages run. Not a defect by itself, but it means a reimplementation
that applies only one stage silently changes every caption.

**CF-7 (P2) — Dataset cache has no freshness validation.** `datasets.py:39-41` gates the whole preparation on
`if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json'))`; when it exists, `:55-63` loads
`fashion_iq_data.json` plus six PKLs and never re-reads the split/caption/correction files. Freshly corrected
dictionaries would be ignored by any later run while every path-level preflight still passes.

**CF-8 (P2) — Real BYTE-level asset gaps that the workbench labels available.**
`~/.cache/huggingface/hub/models--bert-base-uncased/vocab.txt` is **4 bytes**, `b'qwen'`;
`~/.cache/lavis/blip2_pretrained.pth` is **22 bytes**, `b'corrupt nonempty bytes'`; `~/.cache/torch/hub/checkpoints`
does not exist (EVA-ViT-g absent). `workbench/backend/registry.py:555` rejects only `size == 0`, and the records
carry `expected_sha256: null` (`auxiliary_assets.yaml:265-275`), so a non-empty decoy validates. `/root/bert-base-uncased`
is permission-denied, so the source's own path cannot be checked either. Classification: `AUXILIARY_ASSET_MISSING`.

**CF-9 (P3) — README refers to a nonexistent `src/` tree and the wrong file name** (`HABIT/README.md:243,267`
vs root-level `cirr_test_submission.py`), and the CIRR script names an unregistered model (`:97`).

## 10. REMAINING BLOCKERS

| classification | owner | evidence | remediation |
|---|---|---|---|
| `UPSTREAM_CLI_MISSING` | upstream authors | `test.py` has no `__main__`; `train.py:283` is the only callsite; `README.md:243` points at a missing `src/` | upstream adds a FashionIQ entrypoint; a workbench replay module is the interim |
| `FIRST_PARTY_ADAPTER_MISSING` | workbench | `adapters/models.py:64-66`; observed `NotImplementedError` | implement `command()` like `LIMNAdapter` (`models.py:135-160`) |
| `DEPENDENCY_VERSION_CONFLICT` | upstream authors | `datasets.py:13` vs `transformers==4.25.0` (`models.yaml:251`); tag-level 404/200 evidence | raise the declared pin to >=4.26.0 |
| `CHECKPOINT_UNPUBLISHED` | upstream authors | HF API listing: six files, all noisy | publish the 0% weight, or register the 20% row as the comparison target |
| `UNSAFE_CHECKPOINT_SERIALIZATION` | upstream authors | whole-model pickle; `/mnt/data/PreTrain/bert-base-uncased`; `torch.load` default raises | publish state dicts, or document `weights_only=False` + the tokenizer expectation |
| `AUXILIARY_ASSET_MISSING` | workbench (staging) | 4-byte and 22-byte decoys; no EVA dir; `/root/bert-base-uncased` permission-denied | stage real bytes; bind digests; stop the validator accepting non-empty garbage |
| `INSTALLABLE_DEPENDENCY_MISSING` | workbench (env) | `omegaconf` absent from `.venv`; `WORKBENCH_HABIT_PYTHON` unset | build the declared 3.8 env per `README.md:82-85` |
| `NONDETERMINISTIC_PREPROCESSING` | upstream authors | `datasets.py:72` partition is seeded (`train.py:344-351` precedes `:358`); the multi-worker loader and AMP reduction order are not | only blocks a from-scratch run; record the per-noise seed and worker policy first |
| `SOURCE_UNAVAILABLE` | — | not applicable: remote serves the pin | none |
| `GPU_RUNTIME_UNTESTED` | this host | torch 2.14.0+cpu; `require_cuda=True` probe returns `UNCONFIGURED` | run on a GPU host |
| `SCIENTIFIC_PROTOCOL_UNVERIFIED` | workbench | protocol machinery exists, unreachable for `habit` | wire it to a `habit` replay path |

## 11. IMPLEMENTATION PLAN

1. **`workbench/replay/habit.py`** (new, ~120 lines, modeled on `workbench/replay/limn.py`):
   `torch.load(..., weights_only=False, map_location="cpu")` → `model.eval()` → `model.to(device)` →
   `test.test(args, model, trainset, cat, txt_processors)` for `dress/shirt/toptee` → emit one JSON line with
   the six category metrics and the two macro averages. Complexity: low. (No `prompt_load` analogue exists in
   HABIT — `HABIT.py` has no such method — so the load step is one call shorter than `conesep`'s.)
2. **`HabitAdapter.command()`** (`adapters/models.py:64-66`): one `command()` delegating to the replay module
   with `--checkpoint` / `--dataset-root` / `--protocol`. Complexity: one method.
3. **`parse_habit_output`** + dispatch entry in `metrics_extraction.py`. Complexity: low; reuse the existing
   macro-validation shape.
4. **Registry**: set `command_status: COMMAND_AUDITED`, `environment_required: true`, and add the 20%/50%/80%
   `reference_id`s' `review_status` once reviewed. Leave `expected_size_bytes`/`expected_sha256` as they are —
   they are already correct and verified (Section 5). Complexity: registry only.
5. **Staging fix**: replace the 4-byte BERT stub and the 22-byte `.pth` stub with real bytes and bind digests.

Scientific risk: **medium.** The three registered checkpoints are noise-trained, so their clean-*evaluation*
scores pair with the paper's 20/50/80% rows (`64.38 / 62.80 / 58.94` mean) and never with `65.29`. The
`reference_id` mapping in `paper_benchmarks/papers.yaml` already does this correctly for `_n20`; the guard that
would refuse a wrong-row comparison is `paper_gate.COMPARABILITY_FIELDS` (`workbench/backend/paper_gate.py:72`),
which includes `checkpoint_training_noise_pct`.

Expected output: with `fiq_n02`, an aggregate targeting the paper's 20%-noise row
(`mean 64.38, r10 54.48, r50 74.28`), labelled as a noisy-condition reproduction.

## 12. FINAL VERDICT

**BLOCKED** for a reproduction run.
- **BLOCKED** — `FIRST_PARTY_ADAPTER_MISSING` + `UPSTREAM_CLI_MISSING` + `DEPENDENCY_VERSION_CONFLICT` +
  `AUXILIARY_ASSET_MISSING` + `UNSAFE_CHECKPOINT_SERIALIZATION`.
- The registered checkpoint **mapping** is **VERIFIED** (registry size+sha256 equal the Hub's
  `x-linked-size`/`x-linked-etag` for all three files).
- The pinned **row** is `UNKNOWN` as a reproduction target: it is the 0%-noise row and its weight is
  `CHECKPOINT_UNPUBLISHED`.
- The **protocol** is **VERIFIED** (STATIC_AUDIT): full ordered split gallery, reference masked before
  ranking, category Recall@1/10/50, no macro aggregation in-source.
- Runtime: **NOT_RUN** — no GPU, no verified environment, no checkpoint on disk.

## 13. EVIDENCE

- Repo: `/home/heheboiz/data/cir` @ `b70d092e9d2232e34886f819e30c7327b74f3a76`.
- Pin: `workbench/third_party/HABIT` @ `6ddc2cd63eac1f65693f4e821c34fb3b42aee306` (`git rev-parse HEAD`),
  remote `git@github.com:iLearn-Lab/AAAI26-HABIT.git`, single commit.
- Commands: `ls-remote origin HEAD` → `6ddc2cd6…`; `ls-remote https://github.com/Lee-zixu/HABIT` → 301 to the
  mirror; per-file `curl -sIL` on the three resolve URLs (302 → 200 with `x-linked-size: 2730715930` and the
  digests quoted in Section 5); `curl -sS https://huggingface.co/api/models/iLearn-Lab/AAAI26-HABIT` (repo sha
  `8712993b…`, six siblings).
- Files read with lines: `models.yaml:243-304`, `:10-15`; `HABIT/test.py:1-264`; `HABIT/train.py:1-368`;
  `HABIT/datasets.py:1-215`; `HABIT/utils.py:120-144`; `HABIT/cirr_test_submission.py:90-128`;
  `HABIT/README.md`; `HABIT/lavis/models/blip2_models/HABIT.py:106-169,372-436`;
  `HABIT/lavis/models/blip2_models/blip2.py:26-58`; `HABIT/lavis/models/__init__.py:174-199`;
  `workbench/registry/paper_benchmarks/papers.yaml:845-960`; `workbench/registry/paper_benchmarks/extracted/habit_aaai26.json`;
  `workbench/backend/adapters/models.py:1-163`; `workbench/backend/preflight.py:8-51,162-213`;
  `workbench/backend/runtime.py:1781-1816,2153-2173`; `workbench/backend/registry.py:479-498,545-600,956-1007`;
  `workbench/backend/metrics_extraction.py:383-420`; `workbench/backend/fashioniq_layout.py:1-120`;
  `workbench/registry/auxiliary_assets.yaml:1-275`; `workbench/registry/preparation_contracts.yaml`;
  `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:1-80`; `workbench/docs/READINESS_MATRIX.md:29`;
  `workbench/registry/assets/cluster_b3_ilearn.yaml:676-720`.
- Executed: `preflight.model_requirements` (0 blocks, 15 paths), `preflight.preflight_blockers` (1 line),
  `select_checkpoints` (BLOCKED, both policies), `ADAPTERS['habit'].command(...)` (NotImplementedError),
  `verify_environment` with and without `require_cuda=True` (UNCONFIGURED), `/tmp/shape2.py` (numpy shape/mask),
  `/tmp/wt` `torch.load` weights_only demonstration, `sha256sum` of the correction dictionaries across
  `data/`, `TGCIR/`, `LIMN/`, and the transformers tag HTTP probes.
- **UNKNOWN**: whether `/root/bert-base-uncased` exists and is a real BERT (permission-denied); whether the
  pickles' embedded tokenizer can be reconstructed from the public `bert-base-uncased`; whether a published
  checkpoint reproduces bit-exactly from the pinned code (no authoring seed per noise setting is recorded);
  the pinned row's reproducibility on any host (no 0% weight exists).

## 14. ACHIEVED VERIFICATION LEVEL

**LEVEL 2 — STATIC_AUDIT with executed source-independent probes.**
Source text, registry, path existence, HTTP metadata and headers including the Hub's published digests,
numpy semantics and `torch.load` behaviour were inspected directly. **No** model was instantiated, **no**
checkpoint was fully downloaded, **no** network inference was run, **no** Recall was computed. No runtime model
behaviour is claimed.

Preserved invariants: protocol `fashioniq_full_gallery_ref_excluded` unchanged, full gallery unchanged,
reference-excluded policy unchanged, caption input unchanged, noise condition unchanged.
