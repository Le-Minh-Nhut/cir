# Is `COT_ours2` a historical name for the published Qwen captions?

**Verdict: `SOURCE_COMPATIBLE_EQUIVALENCE_UNKNOWN`.**

The author-published Qwen validation captions satisfy the official loaders and the official gallery
feature extraction exactly. Nothing proves the authors' own `COT_ours2/fashioniq/*_cot_val.json`
held those same bytes, and no copy of those files is reachable. An experiment that points the COT
path at the published Qwen files is therefore technically valid — and must be labelled a
source-compatible variant, never strict official reproduction.

## The two paths are not the same consumer

The pinned source reads FashionIQ captions through three paths, and only two of them execute:

| consumer | source | path read |
| --- | --- | --- |
| `FashionIQDataset.__init__` | `src/data_utils_csmcir.py:408` | `fashionIQ_dataset/qwen_captions/{cat}_cot_{split}.json` |
| `extract_index_blip_caption_features` | `src/utils_csmcir.py:100` | `COT_ours2/fashioniq/{cat}_cot_val.json` |
| `FashionIQDataset_imageedit.__init__` | `src/data_utils_csmcir.py:223,226` | `COT_ours2/bert_captions/fashioniq/cap.*`, `COT_ours2/fashioniq/*` |

`blip_validate_fashioniq` (`src/validate_blip_csmcir.py:565`) builds both datasets at lines 596/598
and calls the gallery extractor at 612, so **one official run loads both file sets**. The dataset
copy feeds `__getitem__`; the COT copy feeds the gallery caption features. Neither substitutes for
the other.

`FashionIQDataset_imageedit` is dead code — nothing in the repository instantiates it, and it also
reads `fashionIQ_dataset/captions/keywords_in_mods_*.json`, which the canonical distribution does
not ship. None of its reads can be required at evaluation time.

## What the executing consumers require

Both need the same thing: a JSON object keyed by FashionIQ image id, whose values are objects
carrying `Final_Caption` as a non-empty string. The dataset uses the keys as lookups for both
reference and target ids in relative mode (`__getitem__`, lines 433/438).

The gallery extractor's FashionIQ branch has an `except:` at `src/utils_csmcir.py:166` whose handler
re-executes the *same* expression, so a missing id raises `KeyError` rather than being skipped.
Coverage must be complete for every gallery image.

## The published data satisfies those requirements exactly

Source: Hugging Face `peng12138/CSMCIR`, revision `cf0c19bb346266c295b4b5772ebf03bd1c4f0467` — the
revision the registry already records.

| category | bytes | sha256 | records | gallery ids | covered | missing | extra |
| --- | --- | --- | --- | --- | --- | --- | --- |
| dress | 2817386 | `b7129bafc56ed7f33ea5175e565358b042b5ef9263af002758564579ebb5b6a6` | 3817 | 3817 | 3817 | 0 | 0 |
| shirt | 5015979 | `bbe6f689dfe0c5fcc897979f90877fcaa2c4c8d3dc405cb64f1507fc78452cc9` | 6346 | 6346 | 6346 | 0 | 0 |
| toptee | 3929303 | `acdacc3d8d0565e645324ddd838cb288f1bc69434635d83ece127b93dacac1cc` | 5373 | 5373 | 5373 | 0 | 0 |

All three digests match `workbench/registry/auxiliary_assets.yaml`. Coverage is exact set equality
against `image_splits/split.{cat}.val.json`.

Schema: `Core_Essence`, `Final_Caption`, `Observation_Process`, `Visual_Attributes`; `Final_Caption`
always non-empty; zero malformed records.

The registry's own declarations are consistent with this, though not identical: the `COT_ours2`
records and the `qwen_captions` records share `expected_record_count` (3817/6346/5373),
`key_kind: fashioniq_image_id`, the same `id_source`, and both require `Final_Caption`. They differ in
the full field list — the Qwen records require all four fields (`Core_Essence`, `Visual_Attributes`,
`Observation_Process`, `Final_Caption`), while the unverified COT records name only `Final_Caption`,
which is all any executing consumer reads. The Qwen family alone carries an observed revision,
source path, digest and size.

## Why equivalence is still unproven

- **No copy of the author's COT files is reachable.** Searched `data/`, `teacher/`,
  `workbench/artifacts/`, `workbench/third_party/`, plus a bounded home scan; and every git object in
  every ref (`--diff-filter=A` on `*dress_cot_val.json`, `-S COT_ours2` across all refs). Nothing is
  committed at that path. This rules out what is reachable here and in the public trees, not what
  might exist in a backup, a mirror, or a private copy.
- **No published source.** The author's HF tree was enumerated recursively at the pinned revision
  (29 entries): the two `*_qwen_captions/qwen_captions/` directories, `shoes_captions/`,
  `cirr_test_files/`, `training_log/`, and three `*_tuned_clip_best.pt`. No `COT_ours2` anywhere.
  CSMCIR is this author's only model repository; the only dataset is `PL-Search`, unrelated. The
  claim is scoped to the public trees queried, not to HF history or deleted uploads.
- **`COT_ours2` is consistent with a working-copy directory from the SPRC project.** Published
  bytecode for deleted experimental modules embeds
  `/etc/ssd6/qianzhipeng/SPRC/COT_ours2/bert_captions/fashioniq/...` and `.../cirr/...`, and the
  public SPRC repository contains no `COT_ours2` reference in any commit. The compiled string is
  consistent with, but does not prove, a directory that travelled with the code.
- **A different caption generator cannot be excluded.** The dead `FashionIQDataset_imageedit` names
  `COT_ours2/bert_captions/fashioniq/cap.{cat}.{split}.json`, and the published bytecode for the
  deleted experimental modules references BERT checkpoint paths under `COT_ours2/bert_captions/`. So
  the author's `COT_ours2` tree also held a BERT-produced caption stream alongside the Qwen-named
  files. The executing gallery extractor reads only `COT_ours2/fashioniq/{cat}_cot_val.json`, and the
  executing dataset reads only `fashionIQ_dataset/qwen_captions/`, so the BERT stream is not what
  produces the metric — but its presence means "the author used a `COT_ours2` directory" does not by
  itself settle what the `{cat}_cot_val.json` files under it contained.
- **Basename correspondence is suggestive only.** The code's `COT_ours2/cirr/
  target_image_caption_val.json` shares a basename with the published
  `cirr_qwen_captions/qwen_captions/target_image_caption_val.json`, and the code's
  `shoes_dataset/shoes_cot_val.json` with the published `shoes_captions/shoes_cot_val.json`. All
  three datasets pair a hard-coded path with a published file of the same basename, consumed through
  the same `Final_Caption` key. Strong pattern; not identity.
- **The training log is consistent, not probative.** `training_log/fashioniq_validation_metrics.csv`
  reports the paper's numbers (best epoch 9: R@10 57.07 / R@50 77.27 / mean 67.17), and the training
  script's own validation path does call `extract_index_blip_caption_features`
  (`src/blip_fine_tune_2_csmcir.py:268`), i.e. a path that reads `COT_ours2`. That is a close fit.
  But the CSV carries no path, revision or hash metadata, and many caption sets produce the same
  rounded scores — metrics do not determine bytes. It raises the likelihood that the author's
  `COT_ours2` was their caption source; it does not identify those captions.

## What the alias does and does not buy

Established by running the real source in an isolated workspace (pinned `src/` copied, canonical
dataset symlinked read-only, author files staged):

- Control, no COT path: `extract_index_blip_caption_features` raises
  `FileNotFoundError: .../COT_ours2/fashioniq/dress_cot_val.json` at `utils_csmcir.py:101`. The
  gallery path genuinely reads COT_ours2 and not the dataset's `qwen_captions`.
- With `COT_ours2/fashioniq/{cat}_cot_val.json` pointing at the published Qwen files, all three
  categories complete: 3817 / 6346 / 5373 gallery images, each producing a caption, each caption
  equal to what the dataset class loads. No `KeyError`.

The alias is technically valid. It is a **source-compatible variant** — official code, official
preprocessing, official metric, but a caption source the workbench cannot prove is the author's.

### The check is not circular

A harness that builds its expectation from the same files the alias points at can pass for the wrong
reason, so it was mutation-tested. Five plumbing mutations (alias pointing at the wrong category's
file, a truncated alias file, correct-count-but-wrong-values, a dangling link, an absent alias) were
all caught, while the healthy control and a deliberate same-file-side perturbation behaved as
predicted. The alias therefore exercises the real gallery path, not a tautology.

### Collision behaviour worth recording

The extractor always loads all three categories and merges with `dict.update`, so on a key collision
the later file wins. `shirt` and `toptee` share **121** image ids, so 121 shirt gallery images
receive the toptee caption. This is upstream behaviour, present with or without any alias, and it is
why a check against the per-category files must expect exactly 121 deltas for shirt.

## Relation to the other CSMCIR blocker

`csmcir` is blocked first by an unrelated, independently confirmed source defect
(`workbench/patches/csmcir.md`): 98 unguarded imports of modules the author deleted, sitting before
the line that registers the class this evaluator needs. The missing `COT_ours2` captions are a
**second, separate** blocker.

## The opt-in alias mode

Because the published files are source-compatible but not proven identical, the workbench does not
silently substitute them. Staging is opt-in:

```
python workbench/scripts/pipeline.py real --model csmcir \
    --download-auxiliary-assets --stage-caption-aliases
```

or, without the pipeline wrapper:

```
python workbench/scripts/download_auxiliary_assets.py --model csmcir --stage-caption-aliases
```

What it does:

- copies the already-staged published Qwen caption of the same category and split to the registry's
  `COT_ours2/fashioniq/{cat}_cot_val.json` destination — a real file, never a symlink, because the
  runtime rejects a symlink at a declared auxiliary asset and the pinned evaluator resolves the path
  from its own checkout root
- verifies the source bytes against the digest the registry records for the published file before
  writing anything, and re-hashes the copy before replacing the destination
- refuses to alias any asset whose registry provenance is `VERIFIED`
- records the result in the download manifest with `staged_as:
  SOURCE_COMPATIBLE_CAPTION_ALIAS`, the source path, its revision and its digest
- reports it in preflight as a **warning** (`staged as a source-compatible caption alias: ... — not
  official parity`) rather than as a blocker

What it does not do:

- it never labels the run official reproduction, and `strict_clean`-style checkpoint policy is
  unaffected
- it does not touch an operator's own file: only bytes the manifest records as a workbench-staged
  alias are accepted; any other file at that path still blocks the model
- it does not weaken the alias check — a record whose bytes changed after staging stops being a
  variant and blocks again

The alias is an evaluation input, so it is already inside the stage fingerprint
(`pipeline.model_eval_inputs` includes every `required_for: official_evaluation` auxiliary asset).
Changing the staged bytes therefore invalidates resume, and an unchanged alias is reused on resume
like any other input.
