# ENCODER / LIMN preprocessing parity — settling the `preprocess_train` review claim

Scope: the claim that `workbench/composition/model_encoder.py` wrongly applies `preprocess_val` where
upstream ENCODER uses `preprocess_train` for the reference and `preprocess_val` for the gallery.

Verdict summary — the claim is **partly right and partly wrong**, and the part that is right is a real
defect, but for a different reason than the review gives:

| # | Question | Verdict |
|---|---|---|
| 1 | Does upstream ENCODER's val-split use a stochastic transform for query/reference images? | **YES — CONFIRMED** |
| 2 | Is the asymmetry query-vs-gallery (as the review claims)? | **NO — the asymmetry is reference-vs-its-own-positive-target.** Gallery dedup makes it moot in effect |
| 3 | Does the adapter use `preprocess_val` for everything? | **YES — CONFIRMED** (`model_encoder.py:321`, `:340`) |
| 4 | Does that change ENCODER's published MM R@10 31.02%? | **NOT on the released-gallery protocol.** The composition numbers are internally consistent; the *replay* protocol is the one that can never match upstream |
| 5 | Is the adapter's choice defensible? | **YES** — `preprocess_train` is stochastic; a non-seeded wrapper cannot reproduce an upstream run at all |
| 6 | LIMN | **Correct.** Upstream LIMN uses `transform[1]` (val) for queries, targets AND gallery; adapter matches |

---

## 1. Verbatim source: transform list construction

`workbench/third_party/ENCODER/evaluate_model.py:38-44` — the FashionIQ eval entrypoint:

```python
def load_dataset():
    clip_path = '.'#"/root/CLIP-ViT-B-32-laion2B-s34B-b79K" #
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('ViT-B-32', pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin'))
    if args.dataset == 'fashioniq':
        print('Reading fashioniq')
        fashioniq_dataset = datasets1.FashionIQ(path = args.fashioniq_path, transform = [preprocess_train, preprocess_val], split = args.fashioniq_split)
        return [fashioniq_dataset]
```

The `datasets1` import is a thin alias, **not** a second dataset implementation —
`workbench/third_party/ENCODER/datasets1.py` is the entire file:

```python
from datasets import *
```

So `datasets1.FashionIQ` **is** `datasets.py:63 FashionIQ`. Reading `datasets.py` is reading the real
evaluation path. (This matters: a reviewer who greps `datasets1.py` sees one line and may conclude the
class is elsewhere.)

Indexing: `transform[0] = preprocess_train`, `transform[1] = preprocess_val`
(`evaluate_model.py:40`, used at `:43`). Confirmed by execution, not inference — see §6.

## 2. Verbatim: `get_img` body and what `stage` selects

`workbench/third_party/ENCODER/datasets.py:147-155`:

```python
    def get_img(self, image_name, stage=0):
        img_path = os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")
        with open(img_path, 'rb') as f:
            img = PIL.Image.open(f)
            img = img.convert('RGB')

        img = self.transform[stage](img)
        return img
```

So **`stage` is exactly the transform index**: `stage=0` → `preprocess_train`,
`stage=1` → `preprocess_val`. There is no other meaning, no split-conditional rewrite.

## 3. Every `get_img` call site in the FashionIQ class, by split

The user-facing question is "which split does our adapter evaluate", so the two construction paths are
both shown.

**Val evaluation path — `get_test_data()` (`datasets.py:158-204`).** This is the only path that produces
the `test_queries_*` / `test_targets_*` objects `test.py` consumes.

| Line | Call | Stage | Role |
|---|---|---|---|
| `:176` | *(not a get_img)* `out['source_img_id'] = images.index(candidate)` | — | position lookup |
| `:177` | `out['source_img_data'] = self.get_img(name + '_' + candidate, stage=0)` | **0 = train** | reference / query image |
| `:178` | `out['target_img_data'] = self.get_img(name + '_' + target, stage=0)` | **0 = train** | the query's own positive target, stored on the *query* |
| `:195` | `out['target_img_data'] = self.get_img(name + '_' + images[i], stage=1)` | **1 = val** | gallery row (val-split) |
| `:202` | `out['target_img_data'] = self.get_img(name + '_' + image_name, stage=1)` | **1 = val** | gallery row (original-split) |

Verbatim, `datasets.py:185-203`:

```python
        if self.split == 'val-split':
            for i in test_queries:
                if i['source_img_id'] not in test_targets_id:
                    test_targets_id.append(i['source_img_id'])
                if i['target_img_id'] not in test_targets_id:
                    test_targets_id.append(i['target_img_id'])
            
            for i in test_targets_id:
                out = {}
                out['target_img_id'] = i
                out['target_img_data'] = self.get_img(name + '_' + images[i], stage=1)       
                test_targets.append(out)
        elif self.split == 'original-split':
            for id, image_name in enumerate(images):
                test_targets_id.append(id)
                out = {}
                out['target_img_id'] = id
                out['target_img_data'] = self.get_img(name + '_' + image_name, stage=1)        
                test_targets.append(out)
```

**Training path — `__getitem__` (`datasets.py:131-145`):** `stage=0` for both candidate and target,
which is correct there (it is a training batch).

**Call sites in `test.py`:** only *consumption* — `test.py:27` (`source_img_data`), `:44`
(`target_img_data` of `test_targets`, i.e. gallery rows), `:49` `extract_retrieval_target`. `test.py`
never calls `get_img` itself, so every transform decision is already baked in by `get_test_data`.

### Per-split answer to question 2

| Split | Reference/source image | Query's positive target | Gallery image |
|---|---|---|---|
| `val-split` (upstream default, `evaluate_model.py:19`) | `stage=0` **train** | `stage=0` **train** (on the query dict) | `stage=1` **val** |
| `original-split` | `stage=0` **train** | `stage=0` **train** | `stage=1` **val** |
| train (`__getitem__`) | `stage=0` train | `stage=0` train | n/a |

## 4. Which split does our adapter actually evaluate?

The registry row for `encoder` declares **no** `protocol` field (verified:
`model_by_id('encoder')['protocol'] is None`), and the replay module states its own protocol:

`workbench/replay/encoder.py:48` → `PROTOCOL = "fashioniq_val_split"`
`workbench/replay/encoder.py:310` → `datasets.FashionIQ(path=root_str, transform=transform, split="val-split")`

So the split we mirror is `val-split`, and `stage=0`/`stage=1` above are the applicable rows.

**But note which protocol the composition numbers come from.** `workbench/composition/model_encoder.py`
does **not** use the upstream dataset class at all: it encodes paths handed to it by the runner over the
released 74,381-identity gallery (`runner.py`, `gallery_paths`). Its protocol is
`circus_fashioniq_release_v1` — a *different experiment* from `fashioniq_val_split`. The
`preprocess_train`/`preprocess_val` asymmetry lives in the upstream Dataset, which the composition
adapter never instantiates.

## 5. Adapter transform construction and every application site

`workbench/composition/model_encoder.py:321` — constructed once in the worker:

```python
        _, _, preprocess_val = open_clip_module.create_model_and_transforms(
            "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
```

`workbench/composition/model_encoder.py:336-341` — applied to **every** image, both phases:

```python
        def decode(chunk: Sequence[str]) -> Any:
            images = []
            for path in chunk:
                with Image.open(path) as image:
                    images.append(preprocess_val(image.copy()))
            return torch.stack(images)
```

and it is the same `decode` for queries and targets (`:349-355` dispatches only the *forward* on
`current["phase"]`). Grep confirms there is no other transform reference in the file:

```
$ grep -n "preprocess_val\|preprocess_train\|create_model_and_transforms\|phase ==" workbench/composition/model_encoder.py
321:        _, _, preprocess_val = open_clip_module.create_model_and_transforms(
340:                    images.append(preprocess_val(image.copy()))
345:            # (`preprocess_val`) is used unchanged, and so is the forward.
```

So: **adapter = val transform for query, positive, and gallery. Upstream val-split = train for
query+positive, val for gallery.** Confirmed asymmetry, at the lines the review cites.

## 6. Executed evidence (not source reading)

Run in `workbench/envs/encoder310-adapted/bin/python`, CPU only. Raw log:
`logs/encoder_preprocess_train_vs_val.txt`, `logs/encoder_transform_divergence.txt`.

Resolved transforms (`open_clip.create_model_and_transforms('ViT-B-32', pretrained=…)`):

```
preprocess_train:
    RandomResizedCrop(size=(224, 224), scale=(0.9, 1.0), ratio=(0.75, 1.3333), interpolation=bicubic, antialias=True)
    _convert_to_rgb → ToTensor → Normalize(mean=(0.48145466,0.4578275,0.40821073), std=(0.26862954,0.26130258,0.27577711))
preprocess_val:
    Resize(size=224, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    _convert_to_rgb → ToTensor → Normalize(same)
same object? False
train has RandomResizedCrop: True
val  has RandomResizedCrop: False
```

Measured divergence on real released images (normalized-tensor space):

| image | source size | mean abs (train − val) | max abs (train − val) | train re-draw instability | same torch seed ⇒ equal |
|---|---|---:|---:|---:|---|
| B008ZAUL8I | 154×200 | 0.164 | 3.83 | 3.83 | True |
| 245600258X | 342×407 | 0.503 | 3.83 | 3.83 | True |
| B0037KIXK4 | 342×442 | 0.205 | 3.83 | 3.27 | True |

`scale=(0.9,1.0)` means the crop is a random 90–100% region of the (already resized) image. Per-image
mean absolute change ≈ 0.16–0.50 in normalized units (≈ 19–60/255 in pixel units after undoing
normalization), and repeated draws differ from each other by up to 3.83 — i.e. **the transform is
stochastic and unseeded**, with run-to-run variability of the same magnitude as the train-vs-val gap.

## 7. Is the claim's causal story right?

The review says: "an asymmetry that a wrapper must preserve verbatim … ENCODER's published MM R@10
31.02% was computed with a different reference transform than upstream uses."

Two corrections, both evidence-backed:

1. **The asymmetry is reference-vs-gallery, but with a detail the claim omits.** It is
   *the reference image* (train) vs *gallery rows* (val). `test_queries[i]['target_img_data']`
   (datasets.py:178, `stage=0`) is computed and then **never read** by `test.py`, whose query loop
   consumes only `t['source_img_data']` (`test.py:27`) and whose gallery loop consumes
   `test_targets[*]['target_img_data']` (`test.py:44`). So the effective upstream policy is exactly two
   transforms: reference → `preprocess_train`, gallery → `preprocess_val`.
   The consequence the claim misses: the gallery is a deduplicated list of
   `source_img_id ∪ target_img_id` (`datasets.py:186-190`), so the reference image is **also a gallery
   row**. The *same physical file* is therefore encoded twice — once stochastically as the query input,
   once deterministically as a gallery row. Upstream's own protocol is internally inconsistent, and a
   faithful reproduction has to reproduce a *random draw*, which is only possible by re-seeding.
2. **31.02% is not affected by this.** That number is `recall_at_10` from the composition summary
   (`summary/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json` →
   `metrics.multimodal.recall_at_10 = 31.01782442112277`) over the released 74,381-gallery protocol,
   computed with a single deterministic transform for all inputs. It is not an upstream reproduction
   and the spec says so (`CIRCUS_COMPOSITION_SPEC.md`, "Composition status" — ENCODER mapping
   `UNVERIFIED`; the summary carries
   `reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED`). The number is internally consistent
   with its own declared transform; the defect would only be "wrong" if the run claimed upstream
   parity.

What *is* genuinely wrong, and worth fixing: **the replay path and the composition path encode the same
model with two different declared preprocessing policies while the workbench treats replay as the
upstream-faithful reference.** `workbench/replay/encoder.py:291-310` faithfully mirrors upstream
`[preprocess_train, preprocess_val]`, and even documents the stochasticity:

```
workbench/replay/encoder.py:306-309
        # The query inputs are baked in HERE, through the stochastic training transform, so this
        # construction must be reproducible for the per-query export -- which rebuilds the dataset in
        # its own process -- to observe the same inputs. See `seeded_dataset_construction`.
```

and wraps construction in `seeded_dataset_construction()` (`replay/common.py:347-380`, seed
`DATASET_CONSTRUCTION_SEED = 0`). So the replay is deterministic *by seed*, the composition adapter is
deterministic *by using the val transform*, and the two do not agree on what image the model sees.
That is the real finding: **a cross-protocol preprocessing divergence, correctly disclosed nowhere.**

## 8. LIMN — is its choice correct?

Adapter: `workbench/composition/model_limn.py:527-528`

```python
        transforms[category] = open_clip.image_transform(
            visual.image_size, is_train=False, mean=visual.image_mean, std=visual.image_std)
```

`is_train=False` **for queries as well as targets** — one transform for the whole scope.

Upstream LIMN `datasets.py:39-48,96-127`: `self.transform = transform`, and there are exactly two
image entry points:

- `get_img()` (`:116-121`) → `self.transform[0]` — used **only** by `__getitem__` (`:104-105`), i.e. the
  **training** loader.
- `get_test_img()` (`:123-130`) → `self.transform[1]` — used **only** by `get_test_data()` (`:151`,
  `:153`, `:169`) for query reference (`source_img_data`), query target, **and** every gallery row.

`LIMN/train.py:78,81` confirms the same `[preprocess_train, preprocess_val]` pair, so `transform[1]` is
`preprocess_val`. Therefore upstream LIMN's evaluation uses **validation preprocessing for all three
roles**, with no asymmetry, and `model_limn.py:527` is **correct**. The `dataset_type` mapping also
matches (`model_limn.py:512 dataset_ids = {"dress": 0, "shirt": 1, "toptee": 2}` vs upstream
`datasets.py:21-24`).

One residual, now **settled**: the adapter builds `is_train=False` via
`open_clip.image_transform(visual.image_size, …, mean=visual.image_mean, std=visual.image_std)`, i.e. it
reads mean/std from the *loaded checkpoint's* visual tower, whereas upstream builds its transforms from
`open_clip.create_model_and_transforms(...)` on the pretrained binary. Those agree only if the checkpoint
preserved CLIP's normalization values. Checked: loading
`0_dress_best_model.pt` yields a state dict with **zero** `image_mean`/`image_std` keys, so
`visual.image_mean` resolves to open_clip's module default, and that default is exactly
`(0.48145466, 0.4578275, 0.40821073)` / `(0.26862954, 0.26130258, 0.27577711)`
(`open_clip/constants.py`, read from the same env). Identical numbers, so **no divergence**. `image_size`
comes from `visual.image_size`, which for ViT-L/14 is 224 — also matching.

## 9. What a faithful wrapper would have to do

If the goal is upstream parity for the `fashioniq_val_split` replay protocol, the wrapper must apply
`preprocess_train` to *both* the query reference and the query's own positive target, `preprocess_val`
to gallery rows, and must seed the draw — `seeded_dataset_construction(seed=0)` already exists and is
already used. That is exactly what `replay/encoder.py` does; nothing to fix there.

If the goal is the CIRCUS composition protocol, the wrapper has a free choice, and one deterministic
transform for all inputs is the only defensible one, because the metric must be reproducible and the
gallery must be encoded identically to the query input for the reference-eligible rank to mean
anything. The current choice is right; what is missing is that the summary does not say "this run uses
`preprocess_val` for query inputs, whereas the replay protocol uses `preprocess_train` for them."

## 10. Blast radius

- **Affected by the (real) divergence:** `encoder` only, and only in the sense that the
  `fashioniq_val_split` replay numbers and the `circus_fashioniq_release_v1` composition numbers were
  produced under different preprocessing policies. Re-running the composition protocol with
  `preprocess_train` would be *worse*, not better: it would inject unseeded randomness into the gallery
  encoding and make the run non-reproducible.
- **Not affected:** `limn` (verified correct above); the BLIP-2 family (each builds its own
  `targetpad_transform`, no `preprocess_train` in eval); the released-gallery composition metrics
  themselves, which are self-consistent.
- **Does not share code:** nothing else imports ENCODER's transforms. `model_encoder.py` is the only
  caller of `preprocess_val` in `workbench/composition/`.

## 11. Not executed (STATIC_ONLY)

- No upstream ENCODER end-to-end evaluation was run: it needs the FashionIQ resized-image tree in the
  layout upstream expects (`resized_image/<category>/<id>.jpg`), the val-split caption/split files, CUDA,
  and the GPU is busy with a cohort evaluation. The transform-level facts above *were* executed on real
  images, which is what the claim actually turns on.
- The `31.02%` figure was not re-derived from scratch — it was read from the stored summary plus its
  rank artifact. Recomputing it needs the GPU and a 74,381-image encode.
- The LIMN norm residual (§8) **was** executed (checkpoint state-dict inspection + open_clip default
  constants), so it is no longer in this list.

Also not executed: whether upstream's stochastic reference transform, aggregated over many draws,
shifts the val-split Recall by a material amount. That needs CUDA and the resized-image tree; the
transform-level measurement above establishes only that the two transforms see different pixels.
