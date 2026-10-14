# tgcir (TG-CIR)

MODEL ID / PAPER TITLE / PINNED SOURCE COMMIT / CHECKPOINT IDENTITY / EVALUATION PROTOCOL /
DATASET CONTRACT / PREPROCESSING CONTRACT / AUGMENTATION CONTRACT / GALLERY CONTRACT /
METRIC CONTRACT / SOURCE-TO-WORKBENCH PARITY / RUNTIME STATUS / PAPER-METRIC STATUS /
CRITICAL FINDINGS / MINOR FINDINGS / UNRESOLVED EVIDENCE / FINAL VERDICT

---

## MODEL ID

`tgcir`. Registry block `workbench/registry/models.yaml:721-772` (`- model_id: tgcir` at `:721`);
adapter `TGCIRAdapter` at `workbench/backend/adapters/models.py:112-114`.
Layer-1/2 module: `workbench/replay/tgcir.py` (95 lines, `SOURCE_COVERAGE_MANIFEST.md:41`).

## PAPER TITLE

"Target-Guided Composed Image Retrieval", Haokun Wen, Xian Zhang, Xuemeng Song, Yinwei Wei,
Liqiang Nie — MM '23, DOI `10.1145/3581783.3611817`, arXiv `2309.01366v1` (`models.yaml:730-731`;
`https://arxiv.org/abs/2309.01366` fetched: authors and abstract match the registry title).
STATIC_AUDIT.

## PINNED SOURCE COMMIT

`65fa78eaf8cabe8197fcc22ccab42207c591bdee` (`models.yaml:734`). Verified against the checkout:

```
$ git -C workbench/third_party/TGCIR rev-parse HEAD
65fa78eaf8cabe8197fcc22ccab42207c591bdee
$ git -C workbench/third_party/TGCIR log -1 --format='%H %ci %s'
65fa78eaf8cabe8197fcc22ccab42207c591bdee 2026-04-09 16:03:25 +0800 Update README.md
```

Pin matches. Note the pin's date (2026-04-09) is after publication (2023): the pinned tree is a
later README revision, not the artifact as released in 2023. Its `README.md:17-28` still lists the
2023-era dependency set. Workbench repo itself: `git rev-parse HEAD` →
`d9431fe9632b67a329771cc06567487b591afb4b`, **dirty** (`git status --porcelain` lists
`M workbench/registry/models.yaml`, `M workbench/backend/registry.py`, `?? workbench/backend/paper_gate.py`, …).
The `tgcir` block is byte-identical between HEAD and the working tree (`git show HEAD:workbench/registry/models.yaml`
parsed and compared against the working file → `IDENTICAL`), so the registry claims audited here
are the committed ones. STATIC_AUDIT.

## CHECKPOINT IDENTITY

`models.yaml:764-772`: one variant, `checkpoint_id: tgcir_archive_unresolved`,
`filename: tgcir-fashioniq-unresolved.pt`, `download_url: null`,
`source_url: https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view`,
`source_type: author_google_drive`, `expected_size_bytes: null`, `expected_sha256: null`,
`status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED`.
`checkpoint_serialization: whole_model` (`:724`). The source URL is the README's own link
(`workbench/third_party/TGCIR/README.md:15`). **The archive's FashionIQ member, its size and its
digest are UNKNOWN** — no download attempted (prohibited). Because the serialization is a whole
model, the workbench's own loader refuses to unpickle without an author digest
(`workbench/replay/common.py:34-49`: `raise ValueError(f"no author-verified SHA-256 for whole-model checkpoint: {path}")`).
STATIC_AUDIT.

## EVALUATION PROTOCOL

`native_protocol: fashioniq_val_split`, `supported_protocols: [fashioniq_val_split]`
(`models.yaml:742-743`), `literature_split_label: val` (`:741`).
Protocol definition: `workbench/registry/models.yaml:16-19` ("FashionIQ - Val Split (Pair-Union
Gallery, Reference Excluded)") and `src/evaluation/fashioniq.py:166-168`
(`gallery="pair_union", reference_excluded=True`). Resolution is explicit and non-defaulting
(`src/evaluation/fashioniq.py:178-189`).
This is confirmed by the pinned evaluator, see GALLERY CONTRACT. STATIC_AUDIT.

## DATASET CONTRACT

`preparation_id: tgcir_fashioniq` (`models.yaml:740`), contract
`workbench/registry/preparation_contracts.yaml:170-219`:

- raw inputs (`:172-185`): `captions/cap.{dress,shirt,toptee}.{train,val}.json`,
  `image_splits/split.{dress,shirt,toptee}.val.json` — all six train/val caption files and three
  val splits exist locally (`ls data/fashionIQ_dataset/captions`, `ls .../image_splits`).
- generated artifacts (`:186-205`): `resized_image/{dress,shirt,toptee}` with
  `generator: UNKNOWN`, `cwd: UNKNOWN`; and `captions/correction_dict_{dress,shirt,toptee}.json`
  with `generator: pinned_source_file`, `source_directory: correction_dict`. Both are **missing on
  this host** — see RUNTIME STATUS.
- external asset (`:206-216`): OpenAI CLIP ViT-B/16 weights/cache,
  `state: BLOCKED_EXTERNAL_ASSET_PROVENANCE`, `location_env_var: WORKBENCH_TGCIR_VITB16_WEIGHTS`
  (`workbench/config/workbench.env.example:17` declares the name; nothing sets it).
- `automation_policy: MANUAL_REQUIRED`; `deterministic_status: UNKNOWN`.

Source side, `datasets.py:24-27` shows the root must end in `/`:
`self.path = path`; `self.image_dir = self.path + 'resized_image'`; `self.split_dir = self.path + 'image_splits'`;
`self.caption_dir = self.path + 'captions'`. Dataset freed-ids: the old 2023 host is **not** the
2023 released tree; corrected dependency list is README-specific.
STATIC_AUDIT.

## PREPROCESSING CONTRACT

- Image path: `resized_image/<category>/<id>.jpg` — `datasets.py:100` and `:110`
  (`os.path.join(self.image_dir, image_name.split('_')[0], image_name.split('_')[1] + ".jpg")`).
- Image transform: the CLIP preprocess returned by `clip.load` (`train.py:88`
  `clip, preprocess = clip.load(args.img_encoder, device='cuda', jit=False)`; `:90 img_transform = preprocess`;
  passed to `datasets.FashionIQ(path=..., transform=img_transform)` at `train.py:97-98`).
  Backbone default `--img_encoder` = `'ViT-B/16'` (`train.py:56`), matching the paper
  (`"ViT-B/16 version) as the backbone"`, arXiv HTML §4.1.2).
- Text: `datasets.py:71-80` — `correct_text` (lowercase, `string.punctuation`→space, token-wise lookup
  in `correction_dict_{category}.json`), then `"{} and {}".format(two corrected captions)`
  (`:78-81`). Consumed at `:136` (`get_test_data`) and tokenized by
  `model.py:119-120` (`clip.tokenize(txt).cuda()`).
- Workbench counterpart exists and is a *different* function: `src/datasets/fashioniq.py:93-117`
  (`normalize_fashioniq_caption`) and `:119-183` (`compose_fashioniq_caption`, policy
  `ordered_and` at `:151-156`). Its docstring (`:95`) names ENCODER, not TG-CIR; the workbench-side
  composition for tgcir is not pinned by registry metadata
  (`workbench/scripts/pipeline.py:923` reads an optional `validation_caption_policy` that the
  tgcir registry block does not set). Whether the composed string matches TG-CIR's exactly is
  **UNKNOWN**, and `upstream`'s `concat_text` uses the same rule as the workbench's
  `normalized_ordered_and`, so this is likely but unproven. STATIC_AUDIT.

## AUGMENTATION CONTRACT

`grep -n "transforms|RandomCrop|RandomHorizontal|augment|crop_size|resize" *.py` in
`workbench/third_party/TGCIR` → only `datasets.py:351` (`transforms.Resize((256,256))`, CIRR class,
not reachable from FashionIQ eval) and `train.py:51` (`--crop_size` default 224, an unused arg for
evaluation). **No augmentation is applied on the FashionIQ evaluation path.** The training-side
resize is the operator-provided `resized_image/<category>` cache, whose generator the contract
declares `UNKNOWN` (`preparation_contracts.yaml:194`). STATIC_AUDIT.

## GALLERY CONTRACT

Pinned evaluator gallery, `datasets.py:125-160` (`FashionIQ.get_test_data`):

```
127: with open(os.path.join(self.split_dir, "split.{}.{}.json".format(name,'val')),'r') as f: images = json.load(f)
134: for idx in range(len(ref_captions)):      # annotation order
140:     out['source_img_id'] = images.index(candidate)
142:     out['target_img_id'] = images.index(target)
148: test_targets_id = []
149: for i in test_queries:
150:     if i['source_img_id'] not in test_targets_id: test_targets_id.append(i['source_img_id'])
152:     if i['target_img_id'] not in test_targets_id: test_targets_id.append(i['target_img_id'])
158:     out['target_img_data'] = self.get_test_img(name + '_' + images[i])
```

So the gallery is the **first-seen, order-preserving union of each annotation's reference then
target**, with ids resolved to positions in `split.<category>.val.json` — exactly
`fashioniq_val_split`'s "pair_union" (`src/evaluation/fashioniq.py:186-209`, docstring at `:187-189`), and
`models.yaml:16-19`. Reference exclusion is `test.py:63-70`:
`test_targets_id.append(i['target_img_id'])` … `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`
then `np.argsort(-sims[i, :])[:50]` — masked before ranking, matching
`workbench/docs/PROTOCOL_AUDIT.md:61` and `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:13`.
Workbench removes the column instead of masking (`src/evaluation/fashioniq.py:70-75`), a documented,
test-pinned difference in *mechanism* with the same cohort (`recall_at_k_reference_excluded`
docstring `:44-52`).
Measured on the real local annotations (`python3` count over `cap.<cat>.val.json` /
`split.<cat>.val.json`): pair-union 2628 vs split 3817 (dress), 3089 vs 6346 (shirt),
2902 vs 5373 (toptee), with 0 union ids absent from the split file — the reduced gallery is real
and well-formed here. STATIC_AUDIT.

## METRIC CONTRACT

`test.py:74-80`: `for k in [1, 10, 50]: … r = 100 * r / len(nn_result)`, denominator = query count;
the return value is a list of `('<category>_r<k>', value)` pairs. Macro aggregation happens in the
*training* loop only: `train.py:198-203`
(`for ci, category in enumerate(['dress','shirt','toptee']): t = test.test(args, model, trainset, category);
current_score = current_score + t[1][1]` — `t[1][1]` is `dress_r10`). The workbench macro is an
equal-weight arithmetic mean of per-category recalls (`src/evaluation/fashioniq.py:223-240`); this is
laid out in `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:47-48` ("`macro_average_fashioniq` takes
the equal-weight mean of category recalls, and the reported mean is the arithmetic mean of macro
R@10 and R@50"). The pinned source never prints a macro mean (it sums one cell as a selection
score). STATIC_AUDIT.

## SOURCE-TO-WORKBENCH PARITY

Gallery semantics and reference policy: equivalent (above). Command: **not equivalent and not
present.** `TGCIRAdapter` (`workbench/backend/adapters/models.py:112-114`) declares only
`model_id`/`script`, so it inherits `OfficialScriptAdapter.command` which raises
`NotImplementedError("upstream evaluation command requires audit before execution")`
(`:25-26`). `workbench/backend/preflight.py:133-145` then classifies it as unaudited:
`audited = (command is not None and command is not OfficialScriptAdapter.command and model.get("command_status") in {None,"COMMAND_AUDITED","REPLAY_COMMAND_IMPLEMENTED"})`;
fails → `command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. Baselines with real commands
(`CombinerNoftAdapter` `:97-106`, `TGCIRAdapter` has none) show the contrast.
Layer-2 `workbench/replay/tgcir.py:1-5` explicitly refuses to run:
"this module never loads checkpoint data, constructs replacement weights, initializes CLIP, or runs
inference" — it only prints blockers (`:69-79`). STATIC_AUDIT.

## RUNTIME STATUS

NOT_RUN. No inference was attempted (no GPU environment, no checkpoint, no data). Two independent
blocker producers were run read-only with `PYTHONDONTWRITEBYTECODE=1 python3 -B` and both fail closed.

Observed (workbench layer-1 preflight, `workbench/backend/preflight.py:110-156`):

```
[B] required generated artifact missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image/dress
[B] required generated artifact missing: .../resized_image/shirt
[B] required generated artifact missing: .../resized_image/toptee
[B] required generated artifact missing: .../captions/correction_dict_dress.json
[B] required generated artifact missing: .../captions/correction_dict_shirt.json
[B] required generated artifact missing: .../captions/correction_dict_toptee.json
[B] FashionIQ image extension requirement: image directory missing: .../resized_image
[B] external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache
[B] command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
[B] execution environment not verified: WORKBENCH_TGCIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED]
```

Observed (replay preflight, `workbench/replay/tgcir.py` invoked read-only):

```
$ python3 -B workbench/replay/tgcir.py --source-root workbench/third_party/TGCIR \
    --dataset-root data/fashionIQ_dataset --checkpoint-root /tmp/nonexistent-ckpt
[BLOCKED] pinned TG-CIR source unavailable: source checkout is dirty
[BLOCKED] TG-CIR resized-image directory missing: data/fashionIQ_dataset/resized_image/dress
... (3 resized dirs, 3 dictionaries, 7 native caches) ...
[BLOCKED] checkpoint mapping unresolved: https://drive.google.com/file/d/1OdZTtJqy-.../view does not verify FashionIQ archive member
[BLOCKED] author-verified checkpoint SHA-256 is unavailable; whole-model pickle loading is forbidden
[BLOCKED] OpenAI CLIP ViT-B/16 runtime asset/cache provenance and local availability are unverified
EXIT=1
```

`ls /home/heheboiz/.cache/clip/` → `No such file or directory`; `ls data/fashionIQ_dataset` →
`captions image_splits images qwen_captions` (no `resized_image`). Checkpoint dir absent. The GPU
gate is live: `cuda: '11.0'` (`models.yaml:760`) is not `UNKNOWN`, so `_requires_gpu`
(`workbench/backend/runtime.py:805-815`) returns True and inspection reports the model as pending a
GPU-host verification (`:1221-1230`). The only tgcir artifact under `workbench/artifacts/` is a mock:
`workbench/artifacts/results/fashioniq_val_split/mock/mock_tgcir-tgcir_archive.json`
(`"data_kind": "mock"`, `reported_paper_metrics` all null) — never evidence.
STATIC_AUDIT (no execution).

## PAPER-METRIC STATUS

`reported_r10: 51.32`, `reported_r50: 73.09` (`models.yaml:744-745`); `reported_mean: null` (`:746`)
with `reported_score_note: Paper reports macro R@10 and R@50; 62.205 is derivable but not presented
as a reproduced metric.` (`:747`).

Verified against the paper. arXiv HTML Table 1 ("Performance comparison on FashionIQ and Shoes with
respect to R@k"), two-row header with `colspan`, columns flattened as
`Dresses[R@10,R@50] Shirts[R@10,R@50] Tops&[R@10,R@50] Avg[R@10,R@50]` (header cells:
`Dresses colspan=2`, `Shirts colspan=2`, `Tops&Tees colspan=2`, `Avg colspan=2`). The `TG-CIR` data
row renders 12 numbers:

```
TG-CIR  45.22  69.66 | 52.60  72.52 | 56.14  77.10 | 51.32  73.09  || Shoes 25.89 63.20 85.07 58.05
```

So **51.32 / 73.09 are the FashionIQ-Average R@10 / R@50 cells** — this is the FashionIQ row, and
the FashionIQ per-category numbers (45.22/69.66/52.60/72.52/56.14/77.10) are *different* numbers
that the registry correctly does not use. **No mean is published for FashionIQ-Avg**: the paper
states only "we followed the evaluation metric used in the companion challenge (Guo et al., 2019),
namely, R@10 and R@50 for each of the three categories" (§4.1.3). Searching the fetched text for
`62.205` → no match; the only `62.x` values in the document are `62.51`
(FashionVLP's FashionIQ-Avg R@50, same table) and `62.20` (ARTEMIS's CIRR Avg R@10/R@50 mean, Table
2) — neither is TG-CIR's. `(51.32+73.09)/2 = 62.205` under the workbench's mean convention
(`macro_average_fashioniq`, `src/evaluation/fashioniq.py:239`) but that number is **derivable, not
published**, so `reported_mean: null` is honest rather than an omission. (Caveat for other
uses: Table 3's `TG-CIR` row in the ablation section repeats `51.32/73.09` as FashionIQ-Avg, i.e.
the main-table values are the ones reused in the ablation.) STATIC_AUDIT.

## CRITICAL FINDINGS

1. **P1 — the `BLOCKED_COMMAND` root cause is precise and reproducible: the pinned evaluation entry
   point is a bare callable, and the only FashionIQ caller is the training loop.**
   `workbench/third_party/TGCIR/test.py` is 162 lines and contains `def test(params, model, testset, category)`
   at `:7`, plus `test_cirr_valset` at `:82`; `grep -n "__main__|argparse|ArgumentParser|sys.argv|def main" *.py`
   matches **only** `cirr_test_submission.py:94`, `train.py:3/39/242`. In `test.py` there is no
   `argparse`, no `if __name__`, no checkpoint load: the model arrives as an argument. Its only
   FashionIQ invocation is `train.py:200` `t = test.test(args, model, trainset, category)` inside
   `train_and_evaluate` (`:172-221`), reached from `train.py:242 __main__` via
   `:262-271` (`trainset = load_dataset()` → `train_and_evaluate` → `torch.save(current_model, ...'best_model.pt')`).
   Running `test.py` directly would do nothing at all (`python test.py` defines functions and exits),
   and running `train.py` re-trains (30 epochs, `--num_epochs` default `:53`) before evaluating, which
   is not an inference replay. The one other `__main__` in the tree is CIRR-only and unusable:
   `cirr_test_submission.py:94-101` uses literal ellipses (`path=...`, `model_dir=...`) and calls
   `clip.load('ViT-B/16', device='cpu')`. **Registry `command_status:
   COMMAND_BLOCKED_SOURCE_ENTRYPOINT` (`models.yaml:749`) and `reproduction_status: BLOCKED_COMMAND`
   (`:748`) are correct**, and `workbench/docs/AUXILIARY_ASSET_AUDIT.md:97`, `LEGACY_SOURCE_AUDIT.md:11`,
   `UPSTREAM_SOURCE_AUDIT.md:58-61` agree with the code.
2. **P1 — no local route to a real run exists even if the CLI problem were solved:** resized category
   images, the three correction dictionaries at the loader's dataset path, the seven native caches,
   the checkpoint and the CLIP cache are all absent (see RUNTIME STATUS). The generated `resized_image`
   artifacts are declared `generator: UNKNOWN` (`preparation_contracts.yaml:194`), i.e. reproducing
   them is not currently a contract but an operator decision scaled to image size — and image size is
   unstated in the pinned source (`train.py:51 --crop_size 224` default is not the cache's resize size).
3. **P2 — two cleanliness implementations disagree about the same checkout, and the replay one is
   the stricter.** `workbench/replay/common.py:61-66` flags **any** `git status --porcelain` output:
   observed `[BLOCKED] pinned TG-CIR source unavailable: source checkout is dirty`. The backend's
   `runtime._source_dirty` (`workbench/backend/runtime.py:1998-2010`) filters bytecode
   (`is_bytecode`, `:1829-1835`) and declared artifacts; the same call there returns
   `(True, '65fa78eaf8cabe8197fcc22ccab42207c591bdee', None)`. The only dirt in the TGCIR checkout is
   `?? __pycache__/` (produced 2026-10-10, i.e. after the repo work). Neither implementation is
   wrong in principle, but for this model they disagree, so "source unavailable" seen from the replay
   script is a false blocker a future run would hit. Action: make the replay path use
   `runtime.source_clean_and_pinned`.
4. **P2 — `cuda: '11.0'` is a tool-derived requirement, not a published one, and the registry and
   docs should say so at the point of reading.** `README.md:17-28` lists no CUDA version and no
   Python version (hence `python: UNKNOWN`, `models.yaml:753`, `confidence: VERIFIED_README`); the
   paper lists none. `models.yaml:756-759` explains the `UNKNOWN → 11.0` change, and
   `workbench/docs/UPSTREAM_AUDIT.md:164` attributes it to `test.py:31,47` `.cuda()` per batch with no
   `is_available()` guard — verified at `test.py:31` and `:47`. The value is therefore a *lower-bound
   requirement* (any working CUDA-11+ stack), not the authors' environment; `LEGACY_SOURCE_AUDIT.md:11`
   still says "Python/CUDA unspecified", which is the accurate provenance statement.
5. **P2 — the registry's `method_name`/metrics/protocol are not the failure here; the failure is
   exactly what it declares**: `BLOCKED_COMMAND` + `COMMAND_BLOCKED_SOURCE_ENTRYPOINT` + unresolved
   checkpoint mapping + manual preparation. No P0 was found: nothing in the pinned evaluator would
   silently corrupt a metric. The gallery is order-preserving, the mask is applied before ranking,
   the recall denominator is the query count (`test.py:74-79`), and the reference is the removed
   column, not the target.

## MINOR FINDINGS

1. **P3 — workbench caption-composition parity for tgcir is not pinned.** `src/datasets/fashioniq.py:119-183`
   implements the same two-caption `" and "` rule the source uses (`datasets.py:78-81`), but no
   registry field selects the policy for tgcir (`pipeline.py:923` reads
   `validation_caption_policy`, unset in the tgcir block). Harmless while the model cannot run;
   must be fixed before a replay is trusted.
2. **P3 — `reported_score_note` and the paper-gate interact as intended.** With `reported_mean: null`,
   `compare_metric('mean', None, reproduced)` is not comparable and only R@10/R@50 gate
   (`workbench/backend/paper_gate.py:197-203`); a run could therefore pass on two of three metrics
   while the mean is the metric the workbench itself normally derives. This is the registry author's
   deliberate choice (`:747`), not an accident, but a reviewer should know it.
3. **P3 — `fashioniq_layout: fashioniq_ilearn_resized_training`** (`models.yaml:737`) is verified:
   `workbench/backend/fashioniq_layout.py:25-42` requires `captions/`, `image_splits/`,
   `resized_image/{cat}`, the three val captions, the three correction dicts, the three val splits
   **and** the three *train* captions — the last is real, because
   `datasets.py:29-33,55-66` calls `train_init_process()` (reads `cap.{cat}.train.json`,
   `:57`) whenever `fashion_iq_data.json` is absent.
4. **P3 — the local dataset satisfies the raw half of the contract** but not the generated half:
   `data/fashionIQ_dataset/captions/cap.{cat}.{train,val}.json` and
   `image_splits/split.{cat}.val.json` are present; `images/` holds 74381 `.png` files, which is a
   *different* layout from the `.jpg` per-category directory TG-CIR reads (`datasets.py:100`) —
   the registry's `fashioniq_image_extensions: [".jpg"]` (`models.yaml:738`) is correct and the
   standard-layout `images/` tree is irrelevant to this model.
5. **P3 — asset-audit line citations in the repo docs are internally inconsistent**: the same
   dictionary asset is sourced to `datasets.py:59` in one place and `datasets.py:59, 137` in
   another (`AUXILIARY_ASSET_AUDIT.md:90` vs `registry/audits/cluster_a_encoder_split.yaml:521`),
   and `resized_image` to `datasets.py:25` vs `:28` vs `:102` (`AUXILIARY_ASSET_AUDIT.md:92` vs
   `registry/assets/cluster_b2_encoder_family.yaml:470,486`). The true lines are `:25`
   (`self.image_dir`), `:59` (correction dict in `train_init_process`), `:100/:110` (image join),
   `:131` (correction dict in `get_test_data`). Cosmetic, but a citation that points at the wrong
   line is not evidence.

## UNRESOLVED EVIDENCE

1. The `TG-CIR.zip` Drive archive: member list, the FashionIQ member, its size and its SHA-256.
   `download_url: null`, `expected_size_bytes: null`, `expected_sha256: null` (`models.yaml:767-771`).
   No download attempted (prohibited). UNKNOWN.
2. Whether the published checkpoint's correction dictionaries are byte-identical to the pinned
   `correction_dict/correction_dict_{dress,shirt,toptee}.json`. The pinned tree ships them
   (`git ls-files` lists all four), but the checkpoint was trained against whatever the authors had.
   UNKNOWN.
3. OpenAI CLIP `ViT-B/16` actual local cache contents: `/home/heheboiz/.cache/clip/` does not exist.
   The `preparation_contracts.yaml:216` note ("verify its documented weight SHA-256") is a task, and
   the audit's own `expected_sha256: 5806e77c…` (registry audit yaml) refers to the public OpenAI
   release, which the protocol says may **not** carry over to the paper's run. UNKNOWN.
4. Whether the 2023-release tree (as opposed to the 2026-04-09 pinned HEAD) differs in the
   evaluation path. `git log -1` shows only a README update at the pin; the prior history is not
   examined here. UNKNOWN.
5. The canonical 55-question checklist: `/tmp/audit_assignments.md:47` and `/tmp/audit_context.md:47`
   both require "all 55 questions", but neither file, nor any file under `/home/heheboiz/data/cir`,
   contains that list (`grep -rn "CHECKLIST" workbench/docs` → no checklist template;
   `workbench/docs/reviews/models/` holds 12 model reviews and none is `tgcir.md`). The 55 probes in
   the CHECKLIST below are therefore **mine**, one per checkable fact, recorded as a substitution —
   the same gap is recorded at `workbench/docs/reviews/models/clip4cir_rn50x4_fullft.md:266`.

## FINAL VERDICT

BLOCKED. The registered protocol, gallery construction, reference exclusion, metric definition and
paper row are all verified against the pinned source and the paper, but the pinned evaluator has no
usable CLI at all (function-only `test.py`, FashionIQ evaluation reachable only from the training
loop), and the checkpoint, preparation artifacts and CLIP asset are unresolved — so nothing about
this model's reproduction can be executed, and the registry's `BLOCKED_COMMAND` is the correct status.

---

# EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation
-> checkpoint -> output -> metric comparison`, each arrow with its evidence.

1. **paper/README/config → pinned source.** Registry `upstream_repo_url: https://github.com/iLearn-Lab/MM23-TG-CIR`,
   `upstream_commit_sha: 65fa78eaf8cabe8197fcc22ccab42207c591bdee` (`models.yaml:732-734`). Verified:
   `git -C workbench/third_party/TGCIR rev-parse HEAD` → `65fa78eaf8cabe8197fcc22ccab42207c591bdee`.
   README names the same paper and ships the correction dictionaries (`README.md:14-15,32-34`).
2. **pinned source → source dataset.** `datasets.py:24-27` derives `resized_image/`, `image_splits/`,
   `captions/` from a root that must end in `/`; `:127` reads `split.<cat>.val.json`, `:129` reads
   `cap.<cat>.val.json`, `:131` reads `captions/correction_dict_<cat>.json`,
   `:57,:59` read the train captions and dictionaries; `:29-33` writes `fashion_iq_data.json` and
   `:34-43` writes six PKLs. Local check: the raw caption/split files exist; `resized_image/`,
   the staged dictionaries, the caches and the CLIP cache do not (`ls` outputs above) → **arrow
   broken here.**
3. **source dataset → preprocessing.** Image path `resized_image/<cat>/<id>.jpg` (`datasets.py:100,110`),
   transform = CLIP preprocess (`train.py:88,97-98`), text = punctuation-strip + dictionary lookup +
   `"cap1 and cap2"` (`datasets.py:71-81`, consumed `:136`), tokenizer `clip.tokenize(…).cuda()`
   (`model.py:120`). No augmentation on the eval path (grep above). **Unverified numerically:** no
   images exist to load, so no pixel-level comparison was made. STATIC_AUDIT.
4. **preprocessing → official evaluator.** `test.py:7` `def test(params, model, testset, category)`;
   `:20-51` per-batch `.cuda()` feature extraction; `:61` `sims = all_queries.dot(all_imgs.T)`;
   `:67` reference mask `-10e10`; `:70` `np.argsort(-sims[i,:])[:50]`; `:74-80` recall with the query
   count as denominator. Entry: nothing. `test.py` has no `__main__`/`argparse` (grep returns no
   match in `test.py`); the only FashionIQ call is `train.py:200` inside the training loop. **Arrow
   broken: the evaluator cannot be invoked for inference from the pinned tree.**
5. **official evaluator → workbench invocation.** `TGCIRAdapter` (`adapters/models.py:112-114`) has no
   `command()` override → inherits `OfficialScriptAdapter.command`, which raises
   (`adapters/models.py:25-26`). `preflight.py:133-145` therefore emits
   `command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. The layer-2 module
   `workbench/replay/tgcir.py:1-5,57-79` is deliberately a blocker printer, not a runner.
   **Arrow broken.**
6. **workbench invocation → checkpoint.** `checkpoint_variants[0]` → `tgcir-fashioniq-unresolved.pt`,
   URL `https://drive.google.com/file/d/1OdZTtJqy-…/view`, `checkpoint_mapping_status: UNVERIFIED`
   (`models.yaml:765-772`). `replay/tgcir.py` prints "checkpoint mapping unresolved … does not verify
   FashionIQ archive member" and "author-verified checkpoint SHA-256 is unavailable; whole-model
   pickle loading is forbidden". `replay/common.py:34-49` enforces the digest. **Arrow broken.**
7. **checkpoint → output.** No output exists. The only tgcir result file on disk is the mock
   `workbench/artifacts/results/fashioniq_val_split/mock/mock_tgcir-tgcir_archive.json`
   (`"data_kind": "mock"`, `"reported_paper_metrics": {"r10": null, "r50": null, "mean": null}`,
   `"reproduced_metrics": {"r10": 8.33, "r50": 20.83, "mean": 14.58}`), which is a fixture.
   **NOT_RUN.**
8. **output → metric comparison.** Would be `paper_gate.evaluate_gate`
   (`workbench/backend/paper_gate.py:160-212`): a mock is rejected at `:171-173`
   ("a mock or placeholder result is never evidence about reproduction"); with `reported_mean: null`
   only R@10/R@50 are comparable (`:197-203`). Reference values 51.32/73.09 are verified as the
   paper's FashionIQ-Avg cells (above). **Comparison never reached — NOT_RUN; reference values are
   VERIFIED.**

---

# CHECKLIST

Per shared context, the canonical 55-question list is not present in either assignment file or the
repo (item 55 records this); the 55 probes below are one per checkable fact, in the required
`N. <question short form> — verdict — evidence — severity — action` form. All verdicts are
STATIC_AUDIT unless the item says otherwise.

1. Registered `model_id` present and unique — VERIFIED — `workbench/registry/models.yaml:721`
   (`- model_id: tgcir`); no second occurrence in the file — NONE — none.
2. `method_name` matches the paper — VERIFIED — `models.yaml:727` "TG-CIR"; arXiv:2309.01366 title
   "Target-Guided Composed Image Retrieval" — NONE — none.
3. `paper_url` resolves to that paper — VERIFIED — `models.yaml:731` `https://arxiv.org/abs/2309.01366`;
   fetched, title and authors match — NONE — none.
4. `publication_year` correct — VERIFIED — `models.yaml:729` `2023`; arXiv listing "Submitted on 4 Sep
   2023"; README says "[ACM MM 2023]" — NONE — none.
5. `upstream_repo_url` matches the README/tree — VERIFIED — `models.yaml:732`
   `https://github.com/iLearn-Lab/MM23-TG-CIR`; `workbench/third_party/TGCIR` is the checkout with the
   paper's README — NONE — none.
6. `upstream_commit_sha` equals the checkout HEAD — VERIFIED — `git -C workbench/third_party/TGCIR
   rev-parse HEAD` → `65fa78eaf8cabe8197fcc22ccab42207c591bdee` — NONE — none.
7. `source_available`/`source_dir` correct — VERIFIED — `models.yaml:735-736`; the directory exists
   with 6 tracked `.py` files plus README — NONE — none.
8. Source pin date consistent with the release — UNKNOWN — `git log -1` shows `2026-04-09 Update
   README.md` at the pin; the tree content is a later revision than the 2023 artifact — P3 — record
   that the pin is a post-publication README commit.
9. Pinned evaluator entry point named by the registry exists — VERIFIED —
   `workbench/backend/adapters/models.py:114` `script = "test.py"`;
   `workbench/third_party/TGCIR/test.py` exists — NONE — none.
10. That entry point accepts a CLI — **FAIL** — `test.py:7` `def test(params, model, testset, category)`;
    `grep -n "__main__|argparse|ArgumentParser|sys.argv|def main" *.py` matches only
    `cirr_test_submission.py:94`, `train.py:3/39/242`; `test.py` has none — P1 — `BLOCKED_COMMAND`
    is correct; do not construct a guessed command.
11. Any other in-repo entry point evaluates FashionIQ for inference — **FAIL** — only caller is
    `train.py:200` inside the epoch loop `train.py:198-203`, reached from `train.py:242`/`:262-271`,
    which re-trains and then `torch.save`s `best_model.pt` (`train.py:271`) — P1 — none exists.
12. `cirr_test_submission.py` is a usable alternative — **FAIL** — `:94-101` uses literal `...`
    placeholders (`path=...`, `model_dir=...`), loads `CIRR.pt` and a `CIRR` dataset, `device='cpu'` —
    P2 — CIRR-only, not a FashionIQ replay.
13. `command_status` value matches the code — VERIFIED — `models.yaml:749`
    `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`; `preflight.py:135-145` produces exactly that string for
    this model (observed in preflight output) — NONE — none.
14. Adapter fails closed rather than fabricating a command — VERIFIED —
    `adapters/models.py:25-26` `raise NotImplementedError("upstream evaluation command requires
    audit before execution")`; `TGCIRAdapter` `:112-114` does not override it — NONE — none.
15. Layer-2 replay module cannot execute inference — VERIFIED — `workbench/replay/tgcir.py:3-4`
    ("never loads checkpoint data, constructs replacement weights, initializes CLIP, or runs
    inference") — NONE — none.
16. `reproduction_status` consistent with the above — VERIFIED — `models.yaml:748`
    `BLOCKED_COMMAND` — NONE — none.
17. `replay_status`/`replay_type` consistent — VERIFIED — `models.yaml:722,725` both
    `PREFLIGHT_ONLY`; matches `workbench/tests/test_replay.py:279` — NONE — none.
18. `native_protocol` buildable by the workbench evaluator — VERIFIED — `models.yaml:742`
    `fashioniq_val_split`; `src/evaluation/fashioniq.py:166-168` defines it,
    `:178-189` resolves without fallback — NONE — none.
19. `supported_protocols` contains the native protocol only — VERIFIED — `models.yaml:743`;
    adapter validates membership (`adapters/base.py:27-29`) — NONE — none.
20. `literature_split_label` matches the paper — VERIFIED — `models.yaml:741` `val`; the paper says
    it used "the standard evaluation protocol for each dataset" and FashionIQ R@10/R@50 per category
    (§4.1.3) with the challenge metric — NONE — none.
21. Gallery construction matches the source — VERIFIED — `datasets.py:148-159` first-seen
    reference-then-target union; equals `src/evaluation/fashioniq.py:186-209`
    (`build_pair_union_gallery`) — NONE — none.
22. Gallery is annotation-ordered, not split-file-ordered — VERIFIED — `datasets.py:134` iterates
    `ref_captions`, `:149-153` appends in that order; the split file is used only for id→index
    (`:140,:142`) — NONE — none.
23. Reference exclusion happens before ranking — VERIFIED — `test.py:66-70` mask `-10e10` then
    `argsort` — NONE — none.
24. Workbench reference exclusion is the same cohort — VERIFIED — `src/evaluation/fashioniq.py:66-77`
    removes the reference column; docstring `:44-52` states the equivalence and its caveat — NONE — none.
25. Gallery sizes are plausible on the real annotations — VERIFIED — measured: dress 2628 vs 3817,
    shirt 3089 vs 6346, toptee 2902 vs 5373, 0 union ids missing from the val split — NONE — none.
26. Recall denominator is the query count, not the gallery size — VERIFIED — `test.py:79`
    `r = 100 * r / len(nn_result)`; `src/evaluation/fashioniq.py:77` mean over the hit vector — NONE — none.
27. Reported k values match the source's k list — VERIFIED — `test.py:74` `for k in [1, 10, 50]`;
    registry publishes r10/r50 only — NONE — none.
28. `reported_r10` is a paper value for this method and split — VERIFIED — paper Table 1 `TG-CIR` row,
    FashionIQ-Avg `R@10` = 51.32 (header cells: `Avg` colspan=2, `R@10` rowspan=2) — NONE — none.
29. `reported_r50` likewise — VERIFIED — same row, FashionIQ-Avg `R@50` = 73.09 — NONE — none.
30. `reported_r10/r50` are not the per-category cells — VERIFIED — the TG-CIR row's per-category
    values are 45.22/69.66 (dress), 52.60/72.52 (shirt), 56.14/77.10 (toptee) — NONE — none.
31. `reported_mean: null` reflects a genuinely absent paper value — VERIFIED — §4.1.3 publishes only
    R@10/R@50 per category; text search for `62.205` → no match; the only `62.x` in the document are
    62.51 (FashionVLP, FashionIQ-Avg R@50) and 62.20 (ARTEMIS, CIRR Avg mean, Table 2) — NONE — none.
32. The derivable mean is arithmetically 62.205 — VERIFIED — `python3 -c "print((51.32+73.09)/2)"` →
    `62.205`; convention documented at `src/evaluation/fashioniq.py:239` — NONE — none.
33. `reported_score_note` states the derivation honestly — VERIFIED — `models.yaml:747`
    ("62.205 is derivable but not presented as a reproduced metric") — NONE — none.
34. `fashioniq_layout` matches the source's path handling — VERIFIED — `models.yaml:737`
    `fashioniq_ilearn_resized_training`; `workbench/backend/fashioniq_layout.py:25-42` vs
    `datasets.py:24-27` — NONE — none.
35. That layout's *training* captions really are read at eval time — VERIFIED — `datasets.py:29-33`
    and `:57-59` run `train_init_process()` whenever `fashion_iq_data.json` is absent — NONE — none.
36. Registry `fashioniq_image_extensions` matches the source's extension — VERIFIED — `models.yaml:738`
    `[".jpg"]`; `datasets.py:100,110` append `".jpg"` — NONE — none.
37. `preparation_contract`/`preparation_id` resolve — VERIFIED — `models.yaml:739-740`;
    `registry.py:187-194` (`preparation_contract_for_model`) resolves `tgcir_fashioniq` at
    `preparation_contracts.yaml:170-219` — NONE — none.
38. Raw dataset inputs are declared and present — VERIFIED — `preparation_contracts.yaml:173-185`;
    `ls data/fashionIQ_dataset/captions` and `.../image_splits` list all nine files — NONE — none.
39. Generated artifacts are declared — VERIFIED — `preparation_contracts.yaml:186-205` (resized
    dirs `generator: UNKNOWN`; dictionaries `generator: pinned_source_file`, `source_directory:
    correction_dict`) — NONE — none.
40. Generated artifacts are present locally — **FAIL** — preflight: `required generated artifact
    missing: .../resized_image/{dress,shirt,toptee}`, `.../captions/correction_dict_{dress,shirt,toptee}.json`;
    `ls data/fashionIQ_dataset/resized_image` → `No such file or directory` — P1 — operator preparation
    required (manual).
41. The dictionaries exist in the pinned tree for staging — VERIFIED — `git -C workbench/third_party/TGCIR
    ls-files` lists `correction_dict/correction_dict_{dress,shirt,shoes,toptee}.json` — NONE — none.
42. The dictionary *content* is proven to be the authors' spell-correction set — **UNKNOWN** — the files
    are in the pinned tree, but the published checkpoint's training-time dictionary is not observable —
    P2 — record as an assumption if a replay is ever run.
43. Native caches declared and checked — VERIFIED — `workbench/replay/tgcir.py:31-34` lists
    `fashion_iq_data.json` plus six PKLs; all seven reported missing — NONE — none.
44. Cache *freshness* is validated by anyone — **FAIL** (documented gap) — `datasets.py:29/46`
    branches on existence only; `replay/tgcir.py:50-55` emits "native cache freshness/content is
    unverified" only when all caches are present; `preparation_contracts.yaml:219` says the same —
    P2 — none required now; do not trust a cache file without hashing it.
45. External CLIP asset declared with provenance state — VERIFIED — `preparation_contracts.yaml:206-216`
    `state: BLOCKED_EXTERNAL_ASSET_PROVENANCE`; preflight prints "external runtime asset provenance
    unresolved: OpenAI CLIP ViT-B/16 weights/cache" — NONE — none.
46. External asset is actually present on this host — **FAIL** — `ls /home/heheboiz/.cache/clip/` →
    `No such file or directory` — P1 — operator action.
47. Backbone identity (ViT-B/16) matches the paper — VERIFIED — `train.py:56` default `'ViT-B/16'`,
    `train.py:88` `clip.load`; paper §4.1.2 "(ViT-B/16 version) as the backbone" — NONE — none.
48. Environment pin (torch/torchvision/clip/numpy/pandas) matches the README — VERIFIED —
    `models.yaml:754-761` `1.7.0 / 0.8.0 / clip==0.2.0, numpy==1.22.3, pandas==1.4.2, Pillow`;
    `README.md:19-27` lists exactly `clip==0.2.0`, `numpy==1.22.3`, `pandas==1.4.2`, `Pillow==9.0.1 /
    10.0.0`, `torch==1.7.0`, `torchvision==0.8.0` — NONE — none (registry drops matplotlib/seaborn/tqdm,
    which the evaluation path does not import — `test.py:1-5` imports numpy, torch, tqdm, F).
49. Environment pin Python version matches a README statement — **UNKNOWN** — `models.yaml:753`
    `python: UNKNOWN`; README declares none (`README.md:17-28`) — NONE — correctly recorded as UNKNOWN.
50. `cuda: '11.0'` provenance is honest — VARIANT — `models.yaml:756-760` explains the change and
    `workbench/docs/UPSTREAM_AUDIT.md:164` attributes it to unguarded `.cuda()` at `test.py:31,47`
    (verified); README/paper state no CUDA version — P2 — present it as a requirement, not as the
    authors' environment.
51. CPU-only execution is impossible and the workbench knows it — VERIFIED — `test.py:31,47,101,115`
    `.cuda()` with no `torch.cuda.is_available()` guard
    (`grep -n "cuda()" test.py` → 4 hits); `runtime._requires_gpu` (`runtime.py:805-815`) returns True
    for `11.0` — NONE — none.
52. Checkpoint fingerprint is present — **FAIL** — `models.yaml:767-771` `download_url: null`,
    `expected_size_bytes: null`, `expected_sha256: null`, `checkpoint_mapping_status: UNVERIFIED` —
    P1 — cannot be scheduled; do not unpickle (`replay/common.py:34-49`).
53. Whole-model checkpoint loading is gated on the author hash — VERIFIED —
    `replay/common.py:34-49` (`load_trusted_whole_model` refuses without `expected_sha256` and
    compares the digest before `torch.load(..., weights_only=False)`); `checkpoint_serialization:
    whole_model` at `models.yaml:724` — NONE — none.
54. Reported statuses agree across the registry, docs and code — VERIFIED — `models.yaml:748-749`
    vs `AUXILIARY_ASSET_AUDIT.md:82-97`, `LEGACY_SOURCE_AUDIT.md:11`, `UPSTREAM_AUDIT.md:37`,
    `UPSTREAM_SOURCE_AUDIT.md:58-61`, `PROTOCOL_AUDIT.md:61`, `AUTOMATION_RUNBOOK.md:166` — NONE — none.
55. A canonical 55-question checklist exists in the provided material — **FAIL** —
    `grep -rn "CHECKLIST" /tmp/*.md workbench/docs` returns none;
    `/tmp/audit_assignments.md:47` and `/tmp/audit_context.md:47` require "all 55 questions" without
    listing them; `workbench/docs/reviews/models/` has 12 reviews, none `tgcir.md` — P3 — this
    checklist is a substitution, per `workbench/docs/reviews/models/clip4cir_rn50x4_fullft.md:266`.

---

FINAL VERDICT: BLOCKED

BLOCKED — the pinned TG-CIR evaluator genuinely has no usable CLI (`test.py:7` is a bare function,
FashionIQ evaluation is only reachable from the `train.py:198-203` training loop), and the
checkpoint, resized images, staged correction dictionaries and CLIP ViT-B/16 asset are all unresolved,
while the protocol (`fashioniq_val_split` pair-union + pre-ranking reference mask), the metric
definition and the paper row (FashionIQ-Avg R@10 51.32 / R@50 73.09; no published mean) are verified
against the code and the paper.

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left as
written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 1 committed row(s):

  - `tgcir_fashioniq_val_table1` — status `VERIFIED`, 0% training noise, protocol `fashioniq_val_split`, values `{'r10': 51.32, 'r50': 73.09, 'mean': 62.205}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'WORKBENCH_DERIVED'}`

- **Readiness now:** `BLOCKED_IMPLEMENTATION` — command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT

- **Reference per registered checkpoint condition:**

  - `tgcir_archive_unresolved` (noise None): `tgcir_fashioniq_val_table1` status `VERIFIED` — exact condition match
