# tgcir

**STATIC_AUDIT throughout; zero RUNTIME_VERIFIED claims for this model** (no GPU job attempted, no checkpoint obtained, no evaluator executed; `/home/heheboiz/.cache/clip/` does not exist — observed `ls` → `No such file or directory`).

Snapshot audited (working tree, branch `research/cir-failure-workbench`, `git rev-parse HEAD` → `eb36aa5ab0740e7dca7dd9c78515c3c49514b4e4`):

| artifact | sha256 (observed) |
|---|---|
| `workbench/registry/models.yaml` | `318fbb6720fefde5da1f2fb3f121731f48b229777fd4b16e66bf1f5e331da42c` |
| `workbench/registry/paper_benchmarks/papers.yaml` | `f56c4f1bdc0887c734355f9041dd9acbbb7079527144b11c209f75fa2c63983b` (`registry_version: extract-1b62b520c448`, `:8`) |
| `workbench/backend/paper_gate.py` | `e3099bd3564dc903e698044c0cfe1a5e5a9d8a5dcebad8e9d96304986133633f` |

The tree is being edited concurrently (the `tgcir` registry block moved from `:775` to `:780` mid-audit; one transient `.pyc`-writing `prepare_model.py` run raised an unrelated `TypeError` on `ptha_mtst` before succeeding). All line references below are post-edit and were re-read after the move.

---

## MODEL ID

`tgcir`. Registry block `workbench/registry/models.yaml:780-834` (`- model_id: tgcir` at `:780`). Adapter `TGCIRAdapter` (`workbench/backend/adapters/models.py:114-116`), registered in `ADAPTERS` (`:140`, observed `len(ADAPTERS) == 15`). Layer-2 module `workbench/replay/tgcir.py` (95 lines, `SOURCE_COVERAGE_MANIFEST.md:41`).

## PAPER TITLE

"Target-Guided Composed Image Retrieval" — Haokun Wen, Xian Zhang, Xuemeng Song, Yinwei Wei, Liqiang Nie; MM '23, DOI `10.1145/3581783.3611817` (`models.yaml:784-785`). Independently confirmed from the primary source: `read https://arxiv.org/abs/2309.01366` returns the identical title, author list, `cs.MM`, 2023-09-04, and an abstract describing "a Target-Guided Composed Image Retrieval network (TG-CIR)" with "CLIP as the backbone". PDF `https://arxiv.org/pdf/2309.01366v1` → `1658418` bytes, sha256 `57d0df854314691607fb396ca923f29137f2e12afdf35ae02f9fdf6e2ac4affc` (observed `curl`+`sha256sum`; this reproduces the digest quoted in `workbench/registry/paper_benchmarks/verification/VerifyPapersC.json:35`). PDF page 1 is `arXiv:2309.01366v1`, i.e. v1 (the pinned registry names no version, only `paper_url`). STATIC_AUDIT.

## PINNED SOURCE COMMIT

`65fa78eaf8cabe8197fcc22ccab42207c591bdee` (`models.yaml:793`). Observed:

```
$ git -C workbench/third_party/TGCIR rev-parse HEAD
65fa78eaf8cabe8197fcc22ccab42207c591bdee
$ git -C workbench/third_party/TGCIR log -1
65fa78eaf8cabe8197fcc22ccab42207c591bdee 2026-04-09 16:03:25 +0800 Update README.md
$ git -C workbench/third_party/TGCIR status --porcelain
?? __pycache__/
```

Pin matches. The checkout is a single-commit history (upstream history was squashed into one "Update README.md" commit), so the pin's date does not date the code. The only worktree dirt is `__pycache__/` (produced by an import), which `runtime._source_dirty` (`runtime.py:2040-2052`) and `replay.common.source_checkout_status` (`replay/common.py:65-75`) both exclude as bytecode; observed `runtime.source_status(TGCIR)` → `{"tracked": [], "unauthorized": [], "approved": [], "invalid": []}`.

**The workbench block is byte-identical between HEAD and the working tree** for the `tgcir` fields: `git show HEAD:workbench/registry/models.yaml`'s `:733-770` and the working file's `:780-817` both hash `111091f96b9caa53db6ed1a5b42ef34b` (observed `awk | md5sum`), and `git diff HEAD -- workbench/registry/models.yaml | grep -i tgcir` → empty. So the claims audited here are the committed ones; the concurrent edits are elsewhere. STATIC_AUDIT.

## CHECKPOINT IDENTITY

`models.yaml:823-834`: one variant, `checkpoint_id: tgcir_archive_unresolved`, `filename: tgcir-fashioniq-unresolved.pt`, `download_url: null`, `source_url: https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view`, `source_type: author_google_drive`, `checkpoint_training_noise_pct: null`, `evaluation_noise_pct: 0`, `expected_size_bytes: null`, `expected_sha256: null`, `status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED`. `checkpoint_serialization: whole_model` (`:783`).

The URL is the README's own link: `workbench/third_party/TGCIR/README.md:15` — `- **Checkpoints**: [Download Link](https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view)`. **No download was attempted (prohibited). The archive's FashionIQ member, its size and its digest are UNKNOWN.**

The pinned source does not even name the file to look for: the only save site is `train.py:271` `torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))`, and the README's checkpoint section consists of that one Drive link with no filename. The registry's own `filename: tgcir-fashioniq-unresolved.pt` is therefore a *workbench-invented* placeholder, not an upstream filename — which the `-unresolved` suffix does declare, so it is not misleading, but it must not be treated as archive-member evidence. STATIC_AUDIT.

Because the serialization is a whole model, `replay.common.load_trusted_whole_model` refuses to unpickle without an author digest (`replay/common.py:36-39`: `raise ValueError(f"no author-verified SHA-256 for whole-model checkpoint: {path}")`) — and `expected_sha256` is `null`, so this is a hard, code-enforced refusal. Observed `select_checkpoints(model_by_id('tgcir')).as_dict()`:

```
{"status": "BLOCKED", "reason_code": "BLOCKED_CHECKPOINT",
 "reason": "no checkpoint variant records a training condition, so none can be called clean under policy strict_clean…",
 "strict_clean_checkpoint_available": false, "conditions": {"tgcir_archive_unresolved": null},
 "evidence": {"available": ["tgcir_archive_unresolved (UNVERIFIED, training condition unrecorded)"]}}
```

`artifacts/checkpoints/` contains only `.gitkeep` — observed.

## EVALUATION PROTOCOL

`native_protocol: fashioniq_val_split`, `supported_protocols: [fashioniq_val_split]`, `literature_split_label: val` (`models.yaml:800-802`). Protocol definition `models.yaml:16-19`; resolver `src/evaluation/fashioniq.py:171-183` with `resolve_fashioniq_protocol` raising on an unknown name (`:190-200`), and `build_fashioniq_gallery` branching on the declared gallery (`:204-222`).

Confirmed against the pinned evaluator (see GALLERY CONTRACT): the partition is the first-seen ordered union of each validation annotation's reference then target, with the query's own reference masked out of its ranking. The paper states no gallery construction or reference policy at all — `pdftotext -layout` of the whole PDF contains **zero** occurrences of "validation" and zero of "spell" (`grep -ic` → `0`, `0`), and the only protocol sentence is §4.1.3: "For FashionIQ, we followed the evaluation metric used in the companion challenge [15], namely, R@10 and R@50 for each of the three categories" (`full.txt:494-495`; ref [15] = Guo et al. 2020). So `fashioniq_val_split`/`val`/`reference_excluded: true` are *source*-derived, not paper-derived, and `models.yaml:822` says exactly that. Honest. STATIC_AUDIT.

## DATASET CONTRACT

`preparation_id: tgcir_fashioniq` (`models.yaml:799`), contract `workbench/registry/preparation_contracts.yaml:170-219`, `fashioniq_layout: fashioniq_ilearn_resized_training` (`:796`).

Source side, `datasets.py:24-27` concatenates strings, so the root must end in `/`: `self.image_dir = self.path + 'resized_image'`, `self.split_dir = self.path + 'image_splits'`, `self.caption_dir = self.path + 'captions'`. The layout yields `captions/`, `image_splits/`, `resized_image/{dress,shirt,toptee}` and both train and val captions (`workbench/backend/fashioniq_layout.py:33-41`), which covers the fields a *fresh* constructor opens — including `cap.{c}.train.json` at `datasets.py:57`, because the cache-miss branch calls `train_init_process()` (`:29-33`). This layout is the right one for tgcir and does not have the val-only hole the sibling audit found for HINT.

Observed requirements (real config, real filesystem):

```
raw_missing: []            # all six train/val captions + three val splits present
generated_missing: [resized_image/dress, resized_image/shirt, resized_image/toptee,
                    captions/correction_dict_dress.json, …_shirt.json, …_toptee.json]
image_problems: ['image directory missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image']
aux: [] ; warnings: []
```

`ls data/fashionIQ_dataset` → `captions image_splits images qwen_captions` (no `resized_image`, no `correction_dict_*.json`). STATIC_AUDIT.

## PREPROCESSING CONTRACT

**Image path.** `resized_image/<category>/<id>.jpg` — `datasets.py:100` and `:110`. Matches `models.yaml:797` and `fashioniq_layout.py:118-125`.

**Image transform.** `train.py:88` `clip, preprocess = clip.load(args.img_encoder, device='cuda', jit=False)`; `:93` `img_transform = preprocess`; passed at `:96-98`. Backbone default `--img_encoder` = `'ViT-B/16'` (`train.py:56`), matching paper §4.1.2 (`full.txt:452`) and `papers.yaml:1525-1526`.

I confirmed the transform is *not* declared by the pinned repo, so `preprocessing_id: clip_vitb16_eval_preprocess_resize224_bicubic_centercrop224_clip_norm` (`papers.yaml:1537`) asserts an external dependency. The declared README pin `clip==0.2.0` (`README.md:19`; `models.yaml:815`) **does not resolve to OpenAI CLIP**: `curl -s https://pypi.org/pypi/clip/0.2.0/json` lists one sdist whose `PKG-INFO` reads `Name: clip / Version: 0.2.0 / Summary: A CLI clipboard manager` — the package at that name/version is `silent1mezzo/clip`, with no `clip.py` at all. Falling back to the real source, openai/CLIP `clip/clip.py:79-86` `_transform(n_px)` = `Resize(n_px, interpolation=BICUBIC) → CenterCrop(n_px) → _convert_image_to_rgb → ToTensor → Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))`, with `ViT-B/16` → `Resize(224)`. The *id string* describes the real pipeline; the *package coordinate* does not point at it.

**Text.** `datasets.py:71-80` — `correct_text` lowercases, maps `string.punctuation`→space, strips, splits, substitutes token-wise from `correction_dict_{category}.json`; `concat_text` joins `"{} and {}"`. Consumed at `:136` and tokenized at `model.py:119-120` (`clip.tokenize(txt).cuda()`). Workbench `src/datasets/fashioniq.py:93-117` is textually identical, and `compose_fashioniq_caption(policy="normalized_ordered_and")` (`:158-162`) produces the same string. **But the tgcir registry block sets neither `validation_caption_policy` nor `correction_policy`**, and the workbench has no tgcir model (`grep -rn "tgcir" src/` → no matches), so `pipeline.preparation_recipe_digest` records `null` for both (`workbench/scripts/pipeline.py:939-940`). **Text-stage parity = UNKNOWN / not on the executable path.** STATIC_AUDIT.

## AUGMENTATION CONTRACT

`augmentation_policy: null` (`papers.yaml:1538`). Whole-tree grep for `transforms\.|RandomCrop|RandomHorizontal|RandomResized|augment|crop_size` → only two hits, neither on the evaluation path:

```
datasets.py:351:  transform = transforms.Compose([transforms.Resize((256,256)), transforms.ToTensor()])
train.py:51:     parser.add_argument('--crop_size', type=int, default=224)
```

`datasets.py:351` is inside the `CIRR` class (`:295`), unreachable from `FashionIQ`; `--crop_size` is parsed and never read. **No augmentation is applied on the FashionIQ evaluation path.** Correct. STATIC_AUDIT.

## GALLERY CONTRACT

`datasets.py:125-160` (`FashionIQ.get_test_data`), cited lines re-read:

```
127  with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name,'val')),'r') as f: images = json.load(f)
134  for idx in range(len(ref_captions)):
140      out['source_img_id'] = images.index(candidate)
142      out['target_img_id'] = images.index(target)
148  test_targets_id = []
149  for i in test_queries:
150      if i['source_img_id'] not in test_targets_id: test_targets_id.append(i['source_img_id'])
152      if i['target_img_id'] not in test_targets_id: test_targets_id.append(i['target_img_id'])
158      out['target_img_data'] = self.get_test_img(name + '_' + images[i])
```

Gallery = annotation-order, first-seen union of each query's reference then target; observation ids are indices into `split.<category>.val.json`, so the dedup key is an integer, and dedup happens on the *first* occurrence (no sort anywhere). This is `fashioniq_val_split`'s `pair_union`. Re-derived from the real local annotations:

```
dress  ann 2017 split 3817 union 2628 union-not-in-split 0
shirt  ann 2038 split 6346 union 3089 union-not-in-split 0
toptee ann 1961 split 5373 union 2902 union-not-in-split 0
```

Reference exclusion is `test.py:63-70`: `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, then `nn_result = [np.argsort(-sims[i, :])[:50] …]` (`:70`) — masked **before** ranking. Workbench mechanism differs by design, not by cohort: `src/evaluation/fashioniq.py:35-78` sorts then removes the reference column, documented at `:44-52`. `PROTOCOL_AUDIT.md:58-61` and `FASHIONIQ_PROTOCOL_PARITY.md:29-36` record the same. The pinned mask is `-10e10` on a raw cosine matrix (`test.py:61`), which cannot reorder a correct top-k, so the two agree for the hit test while remaining different operations. STATIC_AUDIT.

## METRIC CONTRACT

`test.py:72-80`: `for k in [1, 10, 50]: … r = 100 * r / len(nn_result)`, denominator = query count, returning `('<category>_r<k>', value)` pairs. No macro is computed on the FashionIQ path: aggregation happens only in the training loop, `train.py:198-204`, where `current_score = current_score + t[1][1]` sums **one cell per category** (the `<category>_r10`) as a checkpoint-selection score (`:202`). The workbench macro is the equal-weight mean of category recalls (`src/evaluation/fashioniq.py:238-247`) and the reported mean is `mean(macro_r10, macro_r50)`. Correct as recorded. STATIC_AUDIT.

## SOURCE-TO-WORKBENCH PARITY

| Component | Source evidence | Workbench | Status |
|---|---|---|---|
| Cohort / gallery | `datasets.py:127-159` | `models.yaml:801`; `src/evaluation/fashioniq.py:171-222`; `src/datasets/fashioniq.py:186-207` | EQUIVALENT_WITH_EVIDENCE (sizes re-derived) |
| Reference exclusion | mask then `np.argsort` (`test.py:66-70`) | sort then delete column (`fashioniq.py:35-78`) | VARIANT (P3, documented) |
| Tie order | `np.argsort(-sims)` | `torch.argsort(..., descending=True)`, no `stable` | VARIANT (P3) |
| Recall denominator | query count (`test.py:79`) | query count (`fashioniq.py:33`) | MATCH |
| Text string | `datasets.py:71-80` | `src/datasets/fashioniq.py:93-117,158-162` | EQUIVALENT at string level; **not invoked for tgcir** |
| Text → model | `clip.tokenize` (`model.py:119-120`) | external `clip`; PyPI `clip==0.2.0` is a different project | **NOT_YET_VERIFIED** |
| Image transform | `clip.load` preprocess (`train.py:88,93-98`) | `clip_vitb16_…bicubic_centercrop224…` = real OpenAI `_transform` | recipe EQUIVALENT; coordinate unpinned |
| Evaluator command | callable only (`test.py:6`); caller `train.py:200` | adapter inherits abstract `command` | **NO PARITY POSSIBLE** |
| Metric extraction | `'{}_r{}'` pairs (`test.py:80`) | `metrics_extraction.py:328-341` | NOT IMPLEMENTED (P1) |
| Paper row | Table 1 p.7 | `papers.yaml:1517-1585` VERIFIED/CONFIRMED | MATCH |
| Registry headline | same row | `models.yaml:803-806` 51.32/73.09/null | MATCH |
| Row `checkpoint_training_noise_pct: 0` | paper silent | `papers.yaml:1528` vs variant `null` | MISMATCH (P1) |
| `strict_clean_checkpoint_available: null` | computed `false` | `models.yaml:821` | MISMATCH (P3) |

## RUNTIME STATUS

**NOT_RUN.** Backend preflight (observed, real config):

```
[B] required generated artifact missing: …/resized_image/{dress,shirt,toptee}
[B] required generated artifact missing: …/captions/correction_dict_{dress,shirt,toptee}.json
[B] FashionIQ image extension requirement: image directory missing: …/fashionIQ_dataset/resized_image
[B] external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache
[B] command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
[B] execution environment not verified: WORKBENCH_TGCIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED]
```

`runtime.runtime_blockers(...)` adds `checkpoint missing: …/tgcir/tgcir-fashioniq-unresolved.pt` and `checkpoint mapping unresolved`. Replay preflight (observed) exits 1 with 17 blockers including `author-verified checkpoint SHA-256 is unavailable; whole-model pickle loading is forbidden`.

**A prior-round finding is now fixed:** `replay/common.py:65-75` filters bytecode via `runtime.is_bytecode`/`_path_of`, so the old false `source checkout is dirty` blocker no longer fires. Observed: no dirty line, and `source_status` is empty.

GPU: `environment.cuda: '11.0'` (`models.yaml:819`) → `_requires_gpu` True (`runtime.py:805-815`). The only tgcir artifact is the mock `results/fashioniq_val_split/mock/mock_tgcir-tgcir_archive.json` (`data_kind: mock`); `gpu_capabilities` mentions tgcir 0 times. STATIC_AUDIT.

## PAPER-METRIC STATUS

`reported_r10: 51.32`, `reported_r50: 73.09`, `reported_mean: null` (`models.yaml:803-805`). **Re-derived from the real paper.** `pdftotext -layout -f 7 -l 7` gives the row (`full.txt:543`, PDF page 7):

```
TG-CIR   45.22  69.66   52.60  72.52   56.14  77.10   51.32  73.09   25.89  63.20  85.07  58.05
```

Second route agrees cell-for-cell: HTML `<td>` parse of `https://arxiv.org/html/2309.01366v1` (HTTP 200, 296661 bytes) → `['TG-CIR','45.22','69.66','52.60','72.52','56.14','77.10','51.32','73.09','25.89','63.20','85.07','58.05']`, plus Table 2 (CIRR) and Table 3 (ablation: `51.32 73.09 58.05 75.57`). Third route: `pdftotext -layout -f 8 -l 8` repeats Table 3. Arithmetic: `(45.22+52.60+56.14)/3 = 51.31999999999999 → 51.32`; `(69.66+72.52+77.10)/3 = 73.09333… → 73.09`; `(51.32+73.09)/2 = 62.205`. **No mean is printed for FashionIQ** — `grep -c "62.205" full.txt` → 0. `reported_mean: null` is honest and the derived 62.205 is correctly `WORKBENCH_DERIVED` (`papers.yaml:1546-1555`). `reference_status: VERIFIED`, `review_status: CONFIRMED`.

**Gate behaviour (direct probes, observed):**

```
select_reference('tgcir', None, 'fashioniq_val_split') → exact condition match
select_reference('tgcir', 0,    'fashioniq_val_split') → exact condition match
select_reference('tgcir', 20,   'fashioniq_val_split') → None, "no verified reference for tgcir under training noise 20%"
evaluate_gate(ref, {r10,r50,mean}, run_meta{noise=0}, all three categories) → PASS_WITHIN_5_PERCENT, 9 comparisons
evaluate_gate(ref, {r10,r50}, …)      → ALERT_METRIC_DEVIATION, ["mean: no usable reproduced value"]
evaluate_gate(ref, None, …)           → FAILED_OUTPUT_VALIDATION, ["no reproduced metrics"]
```

So the gate genuinely compares nine metrics. **But the registered path cannot supply them:** `metrics_extraction.py:328-341` dispatches only `limn` and `csmcir` and returns `AGGREGATE_PARSER_UNAVAILABLE`/`observed_metrics=None` for everything else, so `evaluate_models.py:158-175` → `evaluate_gate` → `FAILED_OUTPUT_VALIDATION` (`paper_gate.py:301-303`) even for a perfect run.

## CRITICAL FINDINGS

1. **P1 — the paper row's `checkpoint_training_noise_pct: 0` is unstated by the paper.** `papers.yaml:1528`; `grep -ic noise full.txt` → **0**. The registry variant records `null` (`models.yaml:829`). If a checkpoint ever lands, `_comparability_blockers` (`paper_gate.py:241-258`) sees `want=0`/`got=null` → `BLOCKED_CHECKPOINT_MISMATCH`. Fail-closed, but caused by an unsupported reference field. Action: set the row's field to `null` or record the inference explicitly.
2. **P1 — nothing can consume TG-CIR's output.** `metrics_extraction.py:328-341` has no tgcir parser → observed `FAILED_OUTPUT_VALIDATION` for a perfect run. Action: state the terminal state, or implement a parser for `test.py:80`'s pairs.
3. **P1 — the adapter names a bare callable as the script.** `test.py` = 162 lines, `def test` `:6`, `def test_cirr_valset` `:86`, no `argparse`/`__main__`. Only caller `train.py:200`, reached from `train.py:242 __main__` → `:267` → `:271`. Running `test.py` does nothing; `train.py` re-trains 30 epochs (`--num_epochs` default `:48`). `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`:808`) and `reproduction_status: BLOCKED_COMMAND` (`:807`) are correct.
4. **P1 — two of the three registered CLIP asset coordinates do not exist upstream.** `cluster_b2_encoder_family.yaml:432,434` records `…5806e77cd80cf8775b1045601c1438b6d00e063f3b21f8f47e222d83ff529a4e/…`; `cluster_a_encoder_split.yaml:532,534` records `…5806e77cd80de873e1bcf07008020ec517735f26879dc332db4704983d446d82/…`. Both `official_download_url`s return **404** (observed `curl -I`), and both digests differ from the real Azure hash `…5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/…` (HTTP **200**), read from openai/CLIP `clip/clip.py:37`. Mitigated: neither file is loaded (`grep -rn "registry/assets\|registry/audits" --include="*.py" workbench/` → 0 hits) and the contract records `BLOCKED_EXTERNAL_ASSET_PROVENANCE` with no URL (`preparation_contracts.yaml:206-216`). Action: correct or delete both URIs/hashes.
5. **P2 — the declared backbone package `clip==0.2.0` does not resolve to OpenAI CLIP.** PyPI `clip` 0.2.0 = `silent1mezzo/clip`, "A CLI clipboard manager"; openai/CLIP publishes **no tags** (`api.github.com/repos/openai/CLIP/tags` → `[]`; `refs/tags/0.2.tar.gz` → 404). The evaluator's whole image pipeline is that package's return value (`train.py:88,93`). Action: pin a git revision.
6. **P2 — the `resized_image/` generator is `UNKNOWN` while four sibling contracts declare a runnable recipe for the same artifact.** `preparation_contracts.yaml:187-195` (`generator: UNKNOWN`, `cwd: UNKNOWN`); `prepare_model.resize_is_complete` (`:155-158`) requires `fashioniq_resize_images` and `_run_resize` (`:465-482`) requires `parameters.image_size`/`source_script`. Observed generators: `clvc_net_/dcnet_/limn_/pair_fashioniq` → `fashioniq_resize_images` with `image_size: 256, source_script: start_kit/resize_images.py` + URL + evidence; `tgcir_fashioniq` → `UNKNOWN`. Consequence: `prepare_model.py --execute` has no runnable step (`:711-716`). Defensible (pinned README never mentions resizing; `grep -in "crop|resize|image_size|preprocess" README.md` → no match) but the size is an unspecified operator decision nothing checks.
7. **P2 — `UPSTREAM_AUDIT.md` makes two false statements about this model.** `:169-181` "iLearn correction dictionaries: confirmed required, confirmed **unpublished**" asserts the dictionaries are "absent locally, published nowhere in the pinned trees" — **false for tgcir**: `workbench/third_party/TGCIR/correction_dict/` ships all four (`README.md:34`), observed digests `dress 9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda (363)`, `shirt ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e (360)`, `toptee 3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71 (391)`, `shoes 50451054df2d2365d732413bea40983c03c6f00ea4361ec258d16bd2c6238c24 (108)`. Propagates to `OPEN_BLOCKERS.md:167-171` (its `OFFICIAL_ASSET_UNAVAILABLE` string appears in that doc alone — stale prose, not a runtime state). Per-model docs and the registry are correct (`AUXILIARY_ASSET_AUDIT.md:90`; `cluster_b2:453-464`; `PHASE2_SOURCES_B.md:131`; `auxiliary_assets.yaml` has no tgcir row). Real gap: the bundled dictionaries are **unhashed anywhere** (`cluster_b2:456` `expected_sha256: null`).
8. **P3 — two documentation claims about the registration are wrong.** (a) Both audit files say the invocation is `python train.py … --fashioniq_path <path>/ …` "(TGCIR/README.md)", but `grep -c "fashioniq_path" README.md` → **0**; `README.md:52-55` is a training command with no data path. (b) `cluster_b2:427,442` cites `model.py:33` for `Backbone.__init__`'s `clip.load` (actually `:41`) and `train.py:53` (actually `--seed`); `cluster_a:525-526,542` gives the correct `train.py:88`/`model.py:41`.
9. **P3 — `strict_clean_checkpoint_available: null` disagrees with the runtime computation.** `models.yaml:821` `null` vs observed `select_checkpoints(...)` → `False`; the frontend type reads the registry field (`frontend/types.ts:4`). Same defect class the batch-2 ENCODER audit recorded.
10. **P3 — the resize-receipt `resampling: LANCZOS` machinery is inert for tgcir** (`resize_is_complete` returns `None`), so context item 6 cannot be observed for this model.

**No P0 was found.** The gallery is order-preserving and first-seen, the mask precedes ranking, the denominator is the query count, and the reference (not the target) is masked. The failure mode for tgcir is refusal, not wrongness.

## MINOR FINDINGS

- No `validation_caption_policy`/`correction_policy` on the block → the recipe digest hashes `null` for both, so the text-parity claim is unpinned by the recipe. P3.
- Extracted-bundle rows carry no `pdf_sha256`/`source_url` field, so row evidence cannot be re-checked from the row. P3.
- `models.yaml:822`'s note omits the gallery-size claim; sizes are re-derivable and were re-derived. P3.
- `fashioniq_ilearn_resized_training` correctly demands train captions a fresh constructor reads (`datasets.py:57`) — the one place tgcir's registration is stricter than the val-only variant. NONE.
- `datasets.py:29,45-53` trusts the six PKLs with no freshness check; warned by the contract (`:219`) and blocked by `replay/tgcir.py:56-62`. NONE.
- The pinned repo's only commit is "Update README.md" (2026-04-09); the registry records no release-vs-HEAD distinction. P3.

## UNRESOLVED EVIDENCE

1. Drive archive FashionIQ member, size, sha256 — UNKNOWN (no download attempted; no pinned file names it).
2. Whole-model deserializability under the declared stack — UNKNOWN (`torch==1.7.0`, `README.md:25`).
3. Whether `~/.cache/clip/ViT-B-16.pt` would be accepted provenance — UNKNOWN (correct URL 200, two registered hashes 404).
4. The exact OpenAI CLIP revision behind `clip==0.2.0` — UNKNOWN (no tags upstream).
5. Token-ID equivalence of `concat_text` → `clip.tokenize` vs the workbench composer — UNKNOWN.
6. Resized-image size and resampling — UNKNOWN (no source, no README statement, no contract parameters).
7. Whether the bundled correction dicts match the authors' canonical ones elsewhere — UNKNOWN (hashable but unhahsed in the registry).
8. Whether the paper's FashionIQ numbers come from an `image_splits` val split — UNKNOWN (zero "validation" occurrences in the PDF).

---

## EXECUTION TRACE

**paper/README/config → implementation.** `models.yaml:785` → `read https://arxiv.org/abs/2309.01366` (matches); `curl -sL https://arxiv.org/pdf/2309.01366v1` → 1658418 B, sha256 `57d0df854314691607fb396ca923f29137f2e12afdf35ae02f9fdf6e2ac4affc`; `pdftotext -layout -f 7 -l 7` → Table 1 header + `TG-CIR` row; HTML `<td>` parse of `https://arxiv.org/html/2309.01366v1` (200) → same row; `pdftotext -layout -f 8 -l 8` → Table 3. **Verified.** `README.md:15,19,25,26,34` → `models.yaml:814-815,827` and `correction_dict/` (4 files). **Verified except the `clip==0.2.0` coordinate.**

**config → source.** `models.yaml:793` = `git rev-parse HEAD` output; `models.yaml:790` `source_dir: TGCIR` exists. **Verified.**

**source → dataset.** `datasets.py:24-27,57,127,129,131` (+`59`) → `preparation_contracts.yaml:172-205` + `fashioniq_layout.py:33-41`. Observed `raw_missing: []`, `generated_missing: 6`, `image_problems: [resized_image missing]`. **Contract verified; unsatisfied locally.**

**dataset → preprocessing.** `datasets.py:100,110` → `models.yaml:797`. `train.py:88,93,96-98` → `papers.yaml:1537`; openai/CLIP `clip.py:79-86` confirms the id. `datasets.py:71-80` → `src/datasets/fashioniq.py:93-117,158-162`, **no tgcir route invokes it**. **Verified at string/recipe level; UNKNOWN at token/pixel level.**

**preprocessing → official evaluator.** `test.py:6,20-51,31,47,61,66-70,74-80`; caller `train.py:200`. `grep` for entrypoints → only `cirr_test_submission.py:94`, `train.py:3,39`. **Verified it is a callable, not an entrypoint.**

**official evaluator → workbench invocation.** `adapters/models.py:114-116` → `:24-25` raises; `preflight.py:184-187` → not audited. Observed `command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. **BROKEN BY DESIGN.**

**workbench invocation → checkpoint.** `registry.py:631-634` → `artifacts/checkpoints/tgcir/tgcir-fashioniq-unresolved.pt`; observed missing; `select_checkpoints` BLOCKED; `replay/common.py:36-39` refuses unpickling. **BROKEN.**

**checkpoint → output.** No run; `replay/tgcir.py` exits 1 with 17 blockers. **NOT_RUN.**

**output → metric comparison.** `metrics_extraction.py:328-341` → `observed_metrics=None` → `evaluate_models.py:158-175` → observed `FAILED_OUTPUT_VALIDATION`. **BROKEN.**

---

## CHECKLIST

1. Registered model_id / source_dir consistent — VERIFIED — `models.yaml:780,790`; `ADAPTERS` len 15 — NONE — none.
2. Pinned commit matches the checkout — VERIFIED — `git rev-parse HEAD` → `65fa78eaf8cabe8197fcc22ccab42207c591bdee` — NONE — none.
3. Checkout clean at the pin — VERIFIED — `git status --porcelain` → only `?? __pycache__/`; `runtime.source_status` → all lists empty — NONE — none.
4. Entry byte-identical HEAD↔worktree — VERIFIED — both `md5 111091f96b9caa53db6ed1a5b42ef34b` — NONE — none.
5. `upstream_repo_url` resolves — VERIFIED — `models.yaml:786`; `README.md:1-9,15` — NONE — none.
6. Paper title/authors/year/arXiv id resolve — VERIFIED — `read https://arxiv.org/abs/2309.01366` — NONE — none.
7. Paper numbers re-derived from the real paper — VERIFIED — `pdftotext -layout -f 7 -l 7` row; HTML `<td>` parse agrees; `(45.22+52.60+56.14)/3=51.32`, `(69.66+72.52+77.10)/3=73.09` — NONE — none.
8. Paper page/table/row localisation — VERIFIED — `papers.yaml:1568-1570` vs `pdftotext -f 7` — NONE — none.
9. No mean is published for FashionIQ — VERIFIED — `grep -c "62.205" full.txt` → 0; `reported_mean: null`; derived marked `WORKBENCH_DERIVED` — NONE — none.
10. Registry headline equals the committed row — VERIFIED — `models.yaml:803-805` = `papers.yaml:1542-1545` — NONE — none.
11. Row states the conditions it is judged on — VERIFIED — `papers.yaml:1527,1530,1533-1536`; gate did not return `BLOCKED_REFERENCE_INCOMPLETE` — NONE — none.
12. Row's `checkpoint_training_noise_pct: 0` is paper-backed — MISMATCH — `papers.yaml:1528` vs `grep -ic noise full.txt` → 0 — P1 — set `null` or record the inference (CF-1).
13. Registry variant vs row noise field agree — MISMATCH — `models.yaml:829` `null` vs `papers.yaml:1528` `0`; observed `conditions {"tgcir_archive_unresolved": null}` — P1 — same fix.
14. Reference is selectable — VERIFIED — observed `select_reference(...,0,…)` → `exact condition match`; `(...,20,…)` → `no verified reference` — NONE — none.
15. Per-category metrics compared when both sides publish them — VERIFIED — `paper_gate.py:348-363`; observed 9 comparisons — NONE — none.
16. Paper category with no observed counterpart recorded as unchecked — VERIFIED — observed detail naming `shirt_r10, shirt_r50, toptee_r10, toptee_r50 … not compared` — NONE — none.
17. Gate can pass and fail for this model — VERIFIED — observed `PASS_WITHIN_5_PERCENT` (9 comparisons) and `ALERT_METRIC_DEVIATION` — NONE — none (refutes the prior review).
18. Gate reachable on the registered path — MISMATCH — `metrics_extraction.py:328-341` → `observed_metrics=None` → `FAILED_OUTPUT_VALIDATION` — P1 — CF-2.
19. Native protocol is the cohort the source computes — VERIFIED — `models.yaml:801` vs `datasets.py:127-159` + `test.py:66-70` — NONE — none.
20. Protocol table entry matches — VERIFIED — `src/evaluation/fashioniq.py:171-183` vs `models.yaml:16-19` — NONE — none.
21. `literature_split_label` agrees with source — VERIFIED — `models.yaml:800`; `datasets.py:127` — NONE — none.
22. `supported_protocols` includes `native_protocol` — VERIFIED — `models.yaml:801-802` — NONE — none.
23. Protocol resolution is non-defaulting — VERIFIED — `fashioniq.py:190-200` — NONE — none.
24. Gallery construction matches `fashioniq_val_split` — VERIFIED — `datasets.py:134,148-153` vs `src/datasets/fashioniq.py:186-207` — NONE — none.
25. Gallery sizes re-derived — VERIFIED — dress 2628/3817, shirt 3089/6346, toptee 2902/5373, 0 exceptions — NONE — none.
26. Reference exclusion implemented — VERIFIED — `test.py:66-70` mask before `np.argsort` — NONE — none.
27. Reference-exclusion procedure identical — VARIANT — mask-then-sort vs sort-then-remove; documented `fashioniq.py:44-52` — P3 — no numeric-identity claim.
28. Tie-breaking identical — VARIANT — `np.argsort` vs `torch.argsort` no `stable`; ties declared unpinnable (`fashioniq.py:144-147`) — P3 — recorded.
29. Recall denominator — VERIFIED — `test.py:79` vs `fashioniq.py:33` — NONE — none.
30. Recall key set — VERIFIED — `test.py:74` vs `paper_gate.py:29` — NONE — none.
31. Source aggregate equals the registry's macro claim — VARIANT (documented) — `train.py:202` sums one R@10 cell per category for selection, never prints a macro — P3 — none.
32. Evaluation-only CLI exists upstream — NOT_RUN (absent) — `test.py` 162 lines, no `__main__`/`argparse` in tree except `cirr_test_submission.py:94`, `train.py:3,39` — P1 — never patch the pin (CF-3).
33. Documented upstream command runnable — MISMATCH — `README.md:52-55` has no data path; `grep -c fashioniq_path README.md` → 0 — P3 — fix the cite.
34. Adapter builds a command — NOT_RUN — `adapters/models.py:114-116`/`:24-25`; observed `command not runnable` — P1 — keep blocked.
35. `command_status` matches the code — VERIFIED — `models.yaml:808` — NONE — none.
36. `reproduction_status` matches the strongest blocker — VERIFIED — `models.yaml:807` — NONE — none.
37. `replay_status`/`replay_type` honest — VERIFIED — `models.yaml:781,784`; `replay/tgcir.py:1-5`; observed exit 1 — NONE — none.
38. Replay preflight agrees with backend on source cleanliness — VERIFIED (fixed) — `replay/common.py:65-75`; no dirty line observed — NONE — prior finding resolved.
39. Registered checkpoint is byte-identifiable — MISMATCH — `models.yaml:823-834` all-null identity; checkpoint absent — P1 — obtain and record members/sizes/hashes.
40. Checkpoint mapping verified — UNKNOWN — `models.yaml:833-834`; `select_checkpoints` BLOCKED — P1 — keep blocked.
41. Author archive identity verified — NOT_RUN — no pinned file names it; `train.py:271` says `best_model.pt` — P1 — inventory first.
42. Registered `filename` is an upstream name — MISMATCH — `models.yaml:824` invented placeholder; audit files quote it as upstream — P3 — label it.
43. Serialization form declared — VERIFIED — `models.yaml:783`; `train.py:271` — NONE — none.
44. Whole-model guard enforced — VERIFIED — `replay/common.py:36-39` — NONE — none.
45. External CLIP asset provenance resolvable — MISMATCH — both registered URLs 404; correct URL 200 (`clip.py:37`) — P1 — CF-4.
46. Registered CLIP cache path matches the consumer — VERIFIED — `~/.cache/clip/ViT-B-16.pt` vs `clip.py:45,48` — NONE — none.
47. Contract's blocked CLIP state honest — VERIFIED — `preparation_contracts.yaml:209`; observed blocker — NONE — none.
48. Declared backbone package resolves to the consumer — MISMATCH — `clip==0.2.0` = clipboard CLI; no openai/CLIP tags — P2 — CF-5.
49. `cuda` requirement declared and gated — VERIFIED — `models.yaml:819`; `runtime.py:805-815`; `test.py:31,47` verified — NONE — none.
50. Environment confidence label justified — VERIFIED — `models.yaml:806`; `README.md:19-27` — NONE — none.
51. Dataset layout matches what the evaluator opens — VERIFIED — `fashioniq_layout.py:33-41` vs `datasets.py:25-27,57,100,110,127,129,131` — NONE — none.
52. Image extension matches the evaluator — VERIFIED — `models.yaml:797`; `datasets.py:100,110`; `fashioniq_layout.py:118-125` — NONE — none.
53. Path-separator requirement honoured — CONCERN — `datasets.py:24-27` string concatenation; undocumented — P3 — document.
54. Preparation contract registered — VERIFIED — `models.yaml:798-799`; `preparation_contracts.yaml:170`; `raw_missing: []` — NONE — none.
55. Declared generated artifacts match the loader — VERIFIED — contract `:190-202` vs `datasets.py:100,110,59,131` — NONE — none.
56. Declared resize artifact reproducible from the pinned repo — MISMATCH — contract `:194-195` `generator: UNKNOWN`; `prepare_model.py:155-158,465-482,711-716` — P2 — CF-6.
57. Sibling contracts that solve it — VERIFIED — observed four siblings with `image_size: 256` + `start_kit/resize_images.py` — NONE — informational.
58. Correction dicts source-provided and staged by a declared generator — VERIFIED — `README.md:34`; `correction_dict/` (4 files); contract `:196-205` — NONE — none.
59. Correction dict bytes verifiable — MISMATCH — no `expected_sha256` anywhere; files hashed here (`9e4a5987…`, `ca29e29e…`, `3dfbf15f…`) — P2 — record digests.
60. Corpus claim "published nowhere in the pinned trees" — MISMATCH — `UPSTREAM_AUDIT.md:169-172` vs `README.md:34` + four observed files; propagates to `OPEN_BLOCKERS.md:167-171` — P2 — CF-7.
61. Per-model correction-dict claims — VERIFIED — `AUXILIARY_ASSET_AUDIT.md:90`; `cluster_b2:453-464`; `PHASE2_SOURCES_B.md:131` — NONE — none.
62. Cache freshness hazard recorded — VERIFIED — contract `:219`; `replay/tgcir.py:56-62`; `datasets.py:29,45-53` — NONE — none.
63. Text composition string parity — VERIFIED (string level) — `datasets.py:71-80` vs `src/datasets/fashioniq.py:93-117,158-162` — NONE — none.
64. Text composition on this model's executable path — NOT_RUN — `grep -rn tgcir src/` → 0; no caption/correction policy → `pipeline.py:939-940` hashes `null` — P2 — claim unpinnable.
65. Token-ID parity with `clip.tokenize` — UNKNOWN — `model.py:119-120`; `import clip` → ModuleNotFoundError — P2 — unobservable.
66. Image transform claim is the real consumer's — VERIFIED — `papers.yaml:1537` vs `clip.py:79-86` + `Resize(224)`; `train.py:88,93` — NONE — none.
67. Common augmentation declared/needed — VERIFIED (none needed) — `papers.yaml:1538`; tree grep → `datasets.py:351` (CIRR), `train.py:51` (never read) — NONE — none.
68. Metric extraction implemented — NOT_RUN — `metrics_extraction.py:328-341`; observed `AGGREGATE_PARSER_UNAVAILABLE` — P1 — CF-2.
69. Legacy 0.05 parity routed through `compare_metric` — VERIFIED — `metrics_extraction.py:57-60`; remaining literals `:186` (LIMN), `:275` (CSMCIR) self-consistency only — NONE — none.
70. `_check_parity` cannot contradict the gate — VERIFIED — returns `PARITY_NOT_EVALUATED` when the paper's `mean` is `null` (tgcir's case) — NONE — none.
71. Registry headline agrees with the committed reference — VERIFIED — `models.yaml:803-805` = `papers.yaml:1542-1545`; `autopilot.py:620-622` — NONE — none.
72. `strict_clean_checkpoint_available` honest — MISMATCH — `models.yaml:821` `null` vs computed `False`; `frontend/types.ts:4` reads the registry field — P3 — set false.
73. Documented blockers are source-backed — VERIFIED — `PC_REPRODUCTION_HANDOFF.md:29`, `AUTOPILOT_GUIDE.md:84`, `AUTOMATION_RUNBOOK.md:166`, `LEGACY_SOURCE_AUDIT.md:11,52`, `UPSTREAM_SOURCE_AUDIT.md:58-61` — NONE — none.
74. Registry/audit asset files are not loaded by runtime — VERIFIED — `grep -rn "registry/assets\|registry/audits" --include="*.py"` → 0; `registry.py:107-139`; `UPSTREAM_AUDIT.md:185-189` — NONE — none (CF-4 mitigation).
75. Readiness classification not mislabelled — VERIFIED — observed `state: BLOCKED_IMPLEMENTATION`, `reason: command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`; first-match order `pc_readiness.py:44-60` masks dataset/asset blockers in the *reason* only — P3 — cosmetic.
76. No real result artifact exists — VERIFIED — only the mock; `gpu_capabilities` mentions tgcir 0 times — NONE — none.
77. Mock never treated as evidence — VERIFIED — `paper_gate.py:290-292` — NONE — none.
78. Reference files internally consistent for tgcir — VERIFIED — `papers.yaml:1517-1585` vs `extracted/legacy_cir_fashioniq_bundle.json`; `VerifyPapersC.json:34` nit reproduced — P3 — optional tidy.
79. Prior-round claims re-derived — VERIFIED with one refutation — prior pin/paper/gallery/metric claims confirmed; its P2 "replay uses a stricter dirty check" is stale (fixed); its "gate can never pass, fail, or compare" is refuted (observed) — NONE — update the prior review.
80. No fabricated hash/URL/line/metric — VERIFIED — the only two 404 URLs are reported as 404 — NONE — none.
81. No runtime claim without observation — VERIFIED — zero RUNTIME_VERIFIED claims — NONE — none.

---

**FINAL VERDICT: BLOCKED** — the pin (`65fa78eaf8cabe8197fcc22ccab42207c591bdee`), the paper identity, every published number (Table 1 p.7 `45.22 69.66 / 52.60 72.52 / 56.14 77.10 / 51.32 73.09`, PDF sha256 `57d0df8543…` reproduced, `51.32`/`73.09` re-derived as the macro of the three category cells), the pair-union/ref-excluded cohort (2628/3089/2902 re-derived from the real annotations), the recall formula, the no-augmentation contract and the preparation contract all check out, and the gate genuinely compares nine metrics for this model — but nothing on the registered path can execute or be judged: the pinned evaluator is a bare callable whose only caller re-trains (`test.py:6`, `train.py:200,242,267`), the adapter inherits an abstract `command` that raises (`adapters/models.py:24-25,114-116`), the Drive archive's FashionIQ member/hash/size are unstated and two of the registered CLIP weight coordinates 404, the resized-image tree is absent with a generator declared `UNKNOWN`, no output parser exists for tgcir (`metrics_extraction.py:328-341`), and the paper row the gate would compare against carries a `checkpoint_training_noise_pct: 0` the paper never states (`grep -ic noise` → 0) — STATIC_AUDIT, no run, no checkpoint, no dataset.
