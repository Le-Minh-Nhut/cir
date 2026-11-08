# FashionIQ evaluation-protocol audit — verified matrix

Audit root: `/tmp/cir-upstream-audit-20261108T155252` (16 pristine clones, detached at the registry pins).
Raw quotes: `raw/groupA_quotes.txt`, `raw/groupB_quotes.txt`, `raw/groupC_quotes.txt`.
Citation verification (line + enclosing scope, via `ast`): `raw/cite_check.json`.
Clone provenance: `CLONE_MANIFEST.tsv`. Vendored-vs-clean diffs: `raw/vendored_diffs.txt`.

**Verification class per row.** "READ" = I opened the pinned file at the pinned commit and read the
constructing/scoring code. "EXECUTED" = I additionally ran our own `src/evaluation/fashioniq.py`
helpers on the real local splits to get counts and rank semantics (§Counts). No upstream evaluator was
executed: that needs each repo's own environment plus GPU, and a cohort evaluation owns the GPU
(pid 264541, `composition_eval.py --models intent`). Nothing below is a numerical reproduction of an
upstream reported score.

Counts used throughout (EXECUTED, `data/FashionIQ`):

| | dress | shirt | toptee | total |
|---|---:|---:|---:|---:|
| full val split `image_splits/split.<cat>.val.json` | 3817 | 6346 | 5373 | **15536** |
| pair union of `cap.<cat>.val.json` candidate ∪ target | 2628 | 3089 | 2902 | **8619** |
| val queries | 2017 | 2038 | 1961 | 6016 |

No duplicate ids inside any split file (`len(set) == len`), so the "full split" gallery is exactly the
split file's length. Every split id resolves to an existing `images/<id>.png` (0 missing), which is what
makes the PTHA existence filter a no-op on this host (§Trap 7).

---

## Group A — full validation gallery, reference KEPT → `fashioniq_original_split`

### csmcir — qzp2018/CSMCIR @ `774f94e2076ff17ea91703a6239d2a08f0e1a44e`
1. **Gallery = full val split.** `src/data_utils_csmcir.py:419-420` (inside `FashionIQDataset.__init__`):
   ```python
   with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
       self.image_names.extend(json.load(f))
   ```
   Count 15536 (sum of the three `split.<cat>.val.json`). Entry that consumes it:
   `blip_validate_fashioniq` builds `FashionIQDataset('val', [dress_type], 'classic', preprocess)`
   (`src/validate_blip_csmcir.py:598`) per category and calls
   `extract_index_blip_caption_features` → `compute_fiq_val_metrics`.
2. **No reference mask.** `compute_fiq_val_metrics` = `src/validate_blip_csmcir.py:24-57`; it sorts and
   labels against targets only:
   ```python
   sorted_indices = torch.argsort(distances, dim=-1).cpu()          # :45
   sorted_index_names = np.array(index_names)[sorted_indices]
   labels = torch.tensor(sorted_index_names == np.repeat(np.array(target_names), len(index_names))...)
   ```
   The only `reference_mask` in the file is `:261-264`, inside `compute_cirr_val_metrics`. Function-level
   scan (`raw/cite_check.json` + `ast`) confirms `mask=False` for `compute_fiq_val_metrics` and
   `mask=True` only for `compute_cirr_val_metrics` / `get_results`.
3. **No silent gallery filtering.** No `exists()` filter on `image_names` in this class.
4. **Entrypoint.** `main()` `src/validate_blip_csmcir.py:501-515`, `--dataset` default `"fashionIQ"`;
   `elif args.dataset == 'fashionIQ': blip_validate_fashioniq(...)`. No split flag exists — `'val'` is
   hard-coded at `:598`. **CONFIRMED**
5. Cited line 419 = the `with open(...)` line. Accurate (the extend is 420).

### ptha_mtst — zyinan99/MTST_PTHA @ `6a02ee4763fc2260b6abfb18be78853facb91d86`
1. **Gallery = full val split.** `MTST_PTHA/src/data_utils.py:167-168`:
   ```python
   with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
       self.image_names.extend(json.load(f))
   ```
2. **No reference mask.** `MTST_PTHA/src/validate_blip.py:24-57` `compute_fiq_val_metrics` has no mask;
   mask tokens occur only at `:257-261` and `:330-334`, both in CIRR functions.
3. **Silent gallery filtering — PRESENT.** `MTST_PTHA/src/data_utils.py:169-172`:
   ```python
   for image_name in self.image_names[:]:
       file_path = base_path / 'fashionIQ_dataset' / 'images' /f"{image_name}.png"
       if not file_path.exists():
           self.image_names.remove(image_name)
   ```
   On this host 0 of 3817/6346/5373 are missing → effective gallery = 15536. It is an extension-specific
   filter (`".png"`), not a split filter. See Trap 7.
4. **Entrypoint.** `validate_blip.py`'s `main()` has **no FashionIQ branch** (`:497-509` dispatches CIRR
   only). The runnable FashionIQ path is `src/blip_validate.py:__main__` → `clip_finetune_fiq(['dress',
   'toptee', 'shirt'], ...)` at `:200`, which imports `compute_fiq_val_metrics` from `validate_blip`. So
   the metric function is the cited one, but **the cited file is not the runnable entrypoint**.
   **CONFIRMED with entrypoint correction**
5. **Cited line 164 is wrong**: it is the comment `# get the image names`. The read is `:167-168`, the
   filter `:169-172`. (User's number is the start of the block, not the read.)

### combiner_rn50x4_noft + clip4cir_rn50x4_fullft — ABaldrati/CLIP4Cir @ `dfed9f748a8a4d05abf613164e06d59b22dbdf13`
1. **Gallery = full val split.** `src/data_utils.py:146-147`:
   ```python
   with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
       self.image_names.extend(json.load(f))
   ```
2. **No reference mask.** `compute_fiq_val_metrics` = `src/validate.py:21-57`, no mask. The file's only
   mask is `:162-166` in `compute_cirr_val_metrics`. `main()` `:263-356`: the FashionIQ branch calls
   `fashioniq_val_retrieval` for shirt/dress/toptee (`:329-341`); `--dataset` is `required=True` (no
   default), so the README command chooses it.
3. **No silent gallery filtering.**
4. **Entrypoint.** `python src/validate.py --dataset fashionIQ --combining-function combiner
   --combiner-path ... --clip-model-name RN50x4 [--clip-model-path ...]` (README `:259-260`;
   `validate.py:263-300`). One checkout serves both registry variants: `Combiner` alone = noft,
   `--clip-model-path` = fullft.
5. Cited lines 146 and 21 accurate. **CONFIRMED**

### sprc — chunmeifeng/SPRC @ `2935a5397732260d1db6fa577e5926f963e36f0f`
1. **Gallery = full val split.** `src/data_utils.py:155-156`:
   ```python
   with open(base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
       self.image_names.extend(json.load(f))
   ```
2. **No reference mask.** `src/validate_blip.py:24-57`, no mask; masks at `:257-261` and `:330-334` are
   CIRR-only. Isolated full val split; no existence filter.
3. **No silent gallery filtering.**
4. **Entrypoint.** As PTHA: `validate_blip.py:main()` dispatches CIRR only (`:508-509`); the runnable
   FashionIQ path is `src/blip_validate.py:__main__` → `clip_finetune_fiq(['dress','toptee','shirt'],...)`
   at `:155`, with `FashionIQDataset('val', [dress_type], 'classic', preprocess)` at `:41`.
   **CONFIRMED with the same entrypoint caveat as PTHA**
5. Cited lines 155 and 24 accurate.

### tme — li-shuxian/TME @ `394aad79df2c104feb32a2c96132582012765062`
1. **Gallery = full val split, per category.** `src/data.py:160-179`: `self.image_names` is only built
   from the pair union when `self.split == 'train'` (`:162-171`); the `else` branch (`:174-177`) reads
   ```python
   with open(self.path / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
       self.image_type_names[dress_type] = json.load(f)
       self.image_names.extend(self.image_type_names[dress_type])
   ```
   and `set_specific_dress_type(category)` (`:181-188`) switches `current_image_names` to that category's
   list. Evaluation uses that: `precompute_evaluation.py:40,47,53` read
   `dataset.current_image_names`.
2. **Reference KEPT for FashionIQ.** `src/precompute_evaluation.py:82-86`:
   ```python
   ref_images_indices_in_index_names = []
   if category == 'CIRR': # only for cirr, delete the reference image from current query.
       for i in reference_names:
           ref_images_indices_in_index_names.append(index_names.index(i))
       distances[list(range(distances.shape[0])), ref_images_indices_in_index_names] = 10e10
   ```
   The condition is `category == 'CIRR'`; `evaluate_features` sets `category` to `'dress'|'shirt'|'toptee'`
   for FashionIQ (`precompute_evaluation.py:15-17`), so on the FashionIQ default path the mask **never**
   runs. (`10e10`, positive, into a `1 - sim` distance — masked anyway.)
3. **No silent gallery filtering.** Images come from a precomputed `features/FashionIQ/val_images.pt` +
   `val_name_to_idx.json` pair (`data.py:250-255`); membership is whatever that cache holds, i.e. the
   split lists the precompute pass saw. Locally both cache files are 1-byte / `{}` placeholders (so not a
   reusable cache) — the *identity* is verified from code, the cache contents are not.
4. **Entrypoint.** `src/precompute_test.py:138-142` → `data.get_dataset(args.dataset, preprocess, 'val')`
   → `evaluate_features`; `precompute_train.py:81` validates on the same `'val'` dataset every epoch.
   No split flag: `'val'` is hard-coded in both.
5. Cited lines 175 and 83 accurate (`:83` is the `if category == 'CIRR':` line; the write is `:86`).
   **CONFIRMED**

---

## Group B — full validation gallery, reference MASKED → `fashioniq_full_gallery_ref_excluded`

All five share one shape: a dataset class whose `self.split` is `'original-split'` (hard-coded or
defaulted), so the gallery is the whole `split.<cat>.val.json`, plus a scorer that writes `-10e10` into the
query's own reference column before `argsort`. Note the name trap: the *split name* says
`original-split`, the *scoring rule* excludes the reference. That is Group B, not Group A.

### airknow — iLearn-Lab/CVPR26-Air-Know @ `6ab6fa95b746f85b08eea4cb66eb51448bce99b7`
1. **Gallery = full val split.** `datasets.py:218-224`:
   ```python
   elif self.split == 'original-split':
       for id, image_name in enumerate(images):
           test_targets_id.append(id)
           out['target_img_id'] = id
           out['target_img_data'] = self.get_img(name + '_' + image_name)
   ```
   `self.split = 'original-split'` is **hard-coded** at `datasets.py:31` — there is no CLI path to the
   `val-split` pair-union branch at `:206`. Count 15536.
2. **Reference MASKED.** `test_BLIP2.py:151`, inside `for i, t in enumerate(test_queries)`:
   ```python
   sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
   ```
   Conditional check: `ast` places `:151` in the **else**-branch of `if category == 'fashion200k'`
   (`:12`), which is the branch actually taken for `category in {'dress','shirt','toptee'}` → the mask
   runs. Unconditional within that branch.
3. **No silent gallery filtering.** `datasets.py` has no existence filter; `get_img/get_test_img` build
   `<cat>/<id>.jpg` and open it (missing file = crash, not silent drop).
4. **Entrypoint.** Entry is the training loop: `train_BLIP2.py:20 import test_BLIP2 as test`, called at
   `:298` inside the per-epoch validation. `test_BLIP2.py` has **no `__main__`** (0 occurrences), and the
   README's only "Testing" instruction is CIRR submission. So "the evaluator" is a callable, invoked
   from `train_BLIP2.py`; our replay shell is therefore the entry point. Category set
   `['dress','shirt','toptee']` at `:297`.
5. Cited lines 218 and 151 accurate. **CONFIRMED**

### conesep — Lee-zixu/ConeSep @ `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b`
1. **Gallery = full val split.** `datasets.py:201-207` (`elif self.split == 'original-split':`, then
   `for id, image_name in enumerate(images)`); `self.split = 'original-split'` hard-coded at `:28`.
2. **Reference MASKED.** `test.py:68`:
   ```python
   for i, t in enumerate(test_queries):
       sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
   ```
   Not inside any `if` — `ast` shows the enclosing chain is only `def test` → `with torch.no_grad()`.
3. **No silent gallery filtering.**
4. **Entrypoint.** `train.py:22 import test as test`, called from the epoch loop at `:216`; `test.py` has
   no `__main__`. `--noise_ratio` is the documented knob (README `:208-219`); nothing selects a split.
5. **URL check (EXECUTED).** `git ls-remote` on both `https://github.com/Lee-zixu/ConeSep` and
   `https://github.com/iLearn-Lab/CVPR26-ConeSep` returns the same HEAD
   `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b`. Same repo — no second model to register.
6. Cited lines 201 and 68 accurate. **CONFIRMED**

### habit — iLearn-Lab/AAAI26-HABIT @ `6ddc2cd63eac1f65693f4e821c34fb3b42aee306`
1. **Gallery = full val split.** `datasets.py:205-211`; the constructor takes
   `split='original-split'` (`:28`) and `train.py:109-112` **does not pass it**, so the default is what
   runs. The `val-split` pair-union branch at `:193` is unreachable from this repo's own entrypoint.
2. **Reference MASKED.** `test.py:151`:
   ```python
   sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
   ```
   `ast`: inside the `else` of `if category == 'fashion200k'` (`:11`) → runs for FashionIQ categories.
3. **No silent gallery filtering.**
4. **Entrypoint.** `train.py:22 import test as test`; called at the validation step. `test.py` has no
   `__main__`. README `:208-219` documents `--noise_ratio` only.
5. Cited lines 205 and 151 accurate. **CONFIRMED**

### intent — iLearn-Lab/AAAI26-INTENT @ `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3`
1. **Gallery = full val split.** `datasets.py:203-209`; constructor default `split='original-split'`
   (`:26`) and `train_INTENT.py:61` declares `--split` default `'original-split'`, forwarded at `:99`.
   The pair-union branch at `:191` is reachable only if an operator overrides `--split val-split`.
2. **Reference MASKED.** `test.py:150`, same shape as HABIT; `ast` puts it in the `else` of
   `if category == 'fashion200k'` (`:11`) → runs.
   **The user's warning is confirmed at the code level**: the default split label says `original-split`
   while the scorer excludes the reference, so the model is Group B, not Group A.
3. **No silent gallery filtering.**
4. **Entrypoint.** `train_INTENT.py` (single file; no `train.py`) → `import test as test`; `--split`
   default `'original-split'` (`:61`). README documents only `--noise_ratio`.
5. Cited lines 203 and 150 accurate. **CONFIRMED**

### dcnet — ozmig77/dcnet @ `68c79d38569f01ca39c3b4ea641b3b7eb2ccb5fb`
1. **Gallery = full val split.** `data_loader/CE_dataset.py:24-27`:
   ```python
   self.no_target = (split == 'val_trg' or split == 'test_trg')
   if self.no_target:
       self.data = read_json(Path(data_dir + f"/image_splits/split.{category}.{split.split('_')[0]}.json"))
   ```
   `split='val_trg'` → `split.split('_')[0] == 'val'` → `split.<cat>.val.json`, item order preserved, and
   `trainer_joint.py:225` reads `val_ids = self.data_loaders[category + '_trg'].dataset.data` — the same
   ordered list. Count 15536. The loader is constructed with `split='val_trg'` at
   `data_loader/data_loaders.py:113-123`.
2. **Reference MASKED, both scoring branches.** `trainer/trainer_joint.py:215-218`:
   ```python
   # Delete reference == target
   for ei, elem in enumerate(batch['meta_info']):
       score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10
       score_matrix_corr[ei, id2targetidx[elem['candidate']]] = -10e10
   ```
   `compute_score` then grades both (`:230` `r10_1,r50_1`, `:234` `r10_2,r50_2`) and the log-softmax sum
   (`:239-242`) — the reference is masked in all three reported variants.
3. **No silent gallery filtering** — `val_trg` reads the split file directly as JSON, no `exists()`.
4. **Entrypoint.** `test.py` → `TrainerJoint`, `_valid_epoch` (`trainer_joint.py:133`); and the same
   `_valid_epoch` runs every epoch when `epoch > self.val_epoch` (`:122-124`). Config
   `configs/ce/fashioniq_dcnet.json` lists `categories: [dress, shirt, toptee]`.
5. Cited line 24 is the `no_target` flag, **not** the file read (that is `:27`); line 215 is the
   comment, the masks are `:217-218`. Both point inside the right block, neither is the operative line.

---

## Group C — gallery = UNION of validation-pair images, reference MASKED → `fashioniq_val_split`

### hint — iLearn-Lab/ICASSP26-HINT @ `bec50b6c8c19111893b617979502b948d1cea5b2`
1. **Gallery = pair union.** `datasets.py:131-166`, `get_test_data(name)` reads the full split
   (`:133-134`) to index, then builds targets from the pair ids:
   ```python
   test_targets_id = []
   for i in test_queries:
       if i['source_img_id'] not in test_targets_id:
           test_targets_id.append(i['source_img_id'])
       if i['target_img_id'] not in test_targets_id:
           test_targets_id.append(i['target_img_id'])
   ```
   Ordered union, reference first per pair → 2628 / 3089 / 2902 (8619 total). No `original-split`
   option exists in this class at all.
2. **Reference MASKED, unconditional.** `test.py:82-83`; `ast` chain is `def test` →
   `with torch.no_grad()`, no `if`.
3. **No silent gallery filtering.**
4. **Entrypoint.** `train.py` imports and calls `test(...)`; `test.py` has no `__main__`. No split flag.
5. Cited line 154 is `test_targets_id = []` (the block head); construction is `:154-165`. Accurate in
   intent, off by the sentinel line.

### encoder — iLearn-Lab/AAAI25-ENCODER @ `29a2a31d6a56f677bf450c3be7cdaef423fb7018`
1. **Gallery = pair union, `val-split` branch.** `datasets.py:185-196`:
   ```python
   if self.split == 'val-split':
       for i in test_queries:
           if i['source_img_id'] not in test_targets_id: ...
   ```
   Branch taken because `__init__` default is `split='val-split'` (`:64`) and both entry points default to
   it: `train.py:35` and `evaluate_model.py:15` (`--fashioniq_split', default = 'val-split'`).
2. **Reference MASKED, unconditional.** `test.py:67-68`; no enclosing `if`.
3. **No silent gallery filtering.**
4. **Entrypoint.** `evaluate_model.py` (module-scope `args`, hence the replay wrapper) or the epoch loop
   in `train.py`; split default `'val-split'` in both.
5. Cited lines 185 and 68 accurate. **CONFIRMED**

### pair — iLearn-Lab/ICASSP25-PAIR @ `bfb0b98cdb4c44f6b9190baaa84d783a77dd888f`
1. **Gallery = pair union.** `data/datasets.py:148-159`, same shape as ENCODER; `__init__` default
   `split='val-split'` (`:23`) and `train.py:29` defaults `--fashioniq_split` to `'val-split'`.
2. **Reference MASKED, unconditional.** `test.py:67-68`.
3. **No silent gallery filtering.**
4. **Entrypoint.** `test.py:6` is `test(params, model, testset, category)` — callable only, no `__main__`;
   invoked from `train.py`.
5. Cited lines 148 and 68 accurate. **CONFIRMED**

### clvc_net — iLearn-Lab/SIGIR21-CLVC-Net @ `bd9b6889489806cb60676597880f91d918b279cb`
1. **Gallery = pair union, because `gallery_all=False`.** `datasets.py:234-257`:
   ```python
   def get_test_targets(self):
       if self.gallery_all:
           ... for idx in range(len(self.images)) ...
       else:
           queries = self.get_test_queries()
           for i in queries:
               if i['source_img_id'] not in test_targets_id: ...
   ```
   Runtime value (EXECUTED by reading the only construction site): `__init__` default
   `gallery_all=False` (`:153`) and `train.py:86-96` builds `testset = datasets.FashionIQ(path=...,
   name=args.name, split='val', transform=...)` **without** `gallery_all`, so the `if` takes the `else`
   branch. No config file or CLI flag sets it; `grep gallery_all` over the repo yields only
   `datasets.py:153,163,234`.
2. **Reference MASKED, unconditional.** `test.py:134-135`; no enclosing `if`.
3. **No silent gallery filtering.**
4. **Entrypoint.** README `:70` says `python test.py`, but `test.py` has **zero** `__main__` (EXECUTED
   count) — that command does nothing. The real driver is `train.py:259`
   `test.test(args, local_model, global_model, dataset, args.dataset)`. Registry already records
   `BLOCKED_NO_SOURCE_REPLAY`.
5. **Cited line 233 is the `def get_test_targets(self):` header**, not the branch; the branch is
   `:234` (`if self.gallery_all:`) / `:242` (`else:`). Line 135 accurate.

### tgcir — iLearn-Lab/MM23-TG-CIR @ `65fa78eaf8cabe8197fcc22ccab42207c591bdee`
1. **Gallery = pair union.** `datasets.py:148-160`; no full-split alternative in this class.
2. **Reference MASKED, unconditional.** `test.py:66-67`; no enclosing `if`.
3. **No silent gallery filtering.**
4. **Entrypoint.** `train.py:23 import test`, called at `:200`.
5. Cited line 148 is the block head (`test_targets_id = []`), accurate in intent; mask line 67 accurate.
   **CONFIRMED**

### limn — iLearn-Lab/TPAMI24-LIMN @ `7d7bc9b116f594a65ac22457491edf28a88d3c3e`
1. **Gallery = pair union (LIMN/ base only).** `LIMN/datasets.py:158-170`: ordered union of
   `source_img_id`/`target_img_id` from `cap.<cat>.val.json`, indexed by `split.<cat>.val.json`.
2. **Reference MASKED.** `LIMN/test.py:80-82`:
   ```python
   if name != 'birds':
       for i, t in enumerate(test_queries):
           sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
   ```
   Condition is only `name != 'birds'`; for `dress|shirt|toptee` it runs. The `!='birds'` guard exists
   because the Birds loader is the one dataset that enables `gallery_all` (`datasets.py:315-322,419`).
3. **No silent gallery filtering.**
4. **Entrypoint.** `LIMN/train.py:17 import test`, called at `:231` with
   `datasets.FashionIQ(path=..., name=<cat>, transform=[train,val])` (`:81-89`) — `test.py` has no
   `__main__`. Checkpoints are `0_<dataset>_best_model.pt` (README `:75`).
5. Scope: this row is the **base / iteration-0** LIMN only, as the user required. `LIMN+/` in the same
   checkout has the same shape (pair union `datasets.py:159-170`, mask under `if name != 'birds'`
   `test.py:80-82`), so I found no evidence that LIMN+ differs — but the registry registers no LIMN+
   variant and I did not extend the claim to one.
6. Cited lines 158 and 82 accurate. **CONFIRMED**

---

## Reconciliation against `workbench/registry/models.yaml`

`native_protocol` / `supported_protocols` read directly from the YAML (17 entries, 16 repos).

| model_id | repo | group per user | verified group (mine) | registry native_protocol | registry supported | verdict |
|---|---|---|---|---|---|---|
| csmcir | CSMCIR | A | A | fashioniq_original_split | [fashioniq_original_split] | **MATCH** |
| ptha_mtst | MTST_PTHA | A | A | fashioniq_original_split | [fashioniq_original_split] | **MATCH** |
| combiner_rn50x4_noft | CLIP4Cir | A | A | fashioniq_original_split | [fashioniq_original_split] | **MATCH** |
| clip4cir_rn50x4_fullft | CLIP4Cir | A | A | fashioniq_original_split | [fashioniq_original_split] | **MATCH** |
| sprc | SPRC | A | A | fashioniq_original_split | [fashioniq_original_split] | **MATCH** |
| tme | TME | A | A | fashioniq_original_split | [fashioniq_original_split] | **MATCH** |
| airknow | AirKnow | B | B | fashioniq_full_gallery_ref_excluded | [fashioniq_full_gallery_ref_excluded] | **MATCH** |
| conesep | ConeSep | B | B | fashioniq_full_gallery_ref_excluded | [fashioniq_full_gallery_ref_excluded] | **MATCH** |
| habit | HABIT | B | B | fashioniq_full_gallery_ref_excluded | [fashioniq_full_gallery_ref_excluded] | **MATCH** |
| intent | INTENT | B | B | fashioniq_full_gallery_ref_excluded | [fashioniq_full_gallery_ref_excluded] | **MATCH** |
| dcnet | DCNet | B | B | fashioniq_full_gallery_ref_excluded | [fashioniq_full_gallery_ref_excluded] | **MATCH** |
| hint | HINT | C | C | fashioniq_val_split | [fashioniq_val_split] | **MATCH** |
| encoder | ENCODER | C | C | fashioniq_val_split | [fashioniq_val_split] | **MATCH** |
| pair | PAIR | C | C | fashioniq_val_split | [fashioniq_val_split] | **MATCH** |
| clvc_net | CLVCNet | C | C | fashioniq_val_split | [fashioniq_val_split] | **MATCH** |
| tgcir | TGCIR | C | C | fashioniq_val_split | [fashioniq_val_split] | **MATCH** |
| limn | LIMN | C | C | fashioniq_val_split | [fashioniq_val_split] | **MATCH** |

**0 MISMATCH, 0 UNVERIFIED protocol rows.** The user's classification and the registry agree on every
entry. This is a source-reading verdict plus the executed count/rank checks in §Counts; no upstream
evaluator was run, so it verifies *what the code computes if run*, not *what the authors reported*.

No protocol-side registry correction is proposed (`REGISTRY_CORRECTIONS.patch` is empty). Non-protocol
defects found while checking are recorded in `PROTOCOL_FINDINGS.md`.
