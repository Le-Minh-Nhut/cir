# limn

MODEL ID: `limn` (LIMN, base iteration 0; LIMN+ out of scope)

PAPER TITLE: *Self-Training Boosted Multi-Factor Matching Network for Composed Image Retrieval*, IEEE TPAMI 46(5):3665-3678, 2024 — `workbench/registry/models.yaml:852-853` (DOI `https://doi.org/10.1109/TPAMI.2023.3346434`). Independently confirmed this round: arXiv 2305.09979v2 (fetched, `pdftotext -layout`, `/tmp/audit-final/limn/paper.txt:42-55`) states LIMN is the base network and LIMN+ is "the iterative dual self-training paradigm enhanced LIMN" — two systems, not interchangeable. `workbench/third_party/LIMN/README.md:166-169` carries the same volume/number/pages/year. The paper row committed at `papers.yaml:1941-1945` correctly excludes the LIMN+ row.

PINNED SOURCE COMMIT: `7d7bc9b116f594a65ac22457491edf28a88d3c3e` (`models.yaml:856`).

```
$ git -C workbench/third_party/LIMN rev-parse HEAD
7d7bc9b116f594a65ac22457491edf28a88d3c3e
$ git -C workbench/third_party/LIMN log --oneline -1
7d7bc9b Update README.md
```

Commit pin: **MATCH**, and the **P2 dirty-checkout finding of the prior review no longer reproduces**. `common.py:51-77` now filters porcelain lines through `runtime.is_bytecode`/`_path_of` (`common.py:69-74`), so the two untracked `__pycache__/` entries no longer count:

```
$ git -C workbench/third_party/LIMN status --porcelain
?? LIMN+/__pycache__/
?? LIMN/__pycache__/
$ .venv/bin/python -c "... workbench.replay.limn.validate_source(Path('workbench/third_party/LIMN'))"
source: []
$ ... workbench.backend.runtime.source_clean_and_pinned(model_by_id('limn'), p)
(True, '7d7bc9b116f594a65ac22457491edf28a88d3c3e', None)
$ ... runtime.source_status(p)  ->  {'tracked': 0, 'unauthorized': 0, 'approved': 0, 'invalid': 0}
$ ... common.source_checkout_status(p, 'deadbeef')
(False, '7d7bc9b1...', 'source pin mismatch: expected deadbeef, found 7d7bc9b1...')
```

CHECKPOINT IDENTITY: three category-specific whole saved models. Registry metadata re-matched against the author Hub LFS metadata **fetched this round** (STATIC_AUDIT; no bytes downloaded):

| checkpoint_id | file | registry size / sha256 | Hub LFS oid (fetched) |
|---|---|---|---|
| `base_iter0_dress` | `0_dress_best_model.pt` | 1792010157 / `96a335d4…fc1c` (`models.yaml:887-888`) | `96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c`, size 1792010157 |
| `base_iter0_shirt` | `0_shirt_best_model.pt` | 1792010157 / `9260c7f7…0d74` (`:896-897`) | `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74`, size 1792010157 |
| `base_iter0_toptee` | `0_toptee_best_model.pt` | 1792010157 / `79f663a1…ce2` (`:905-906`) | `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2`, size 1792010157 |

Evidence: `curl "https://huggingface.co/api/models/iLearn-Lab/TPAMI24-LIMN/tree/30560ad575a56c39cc39047d1a474269338c77c0?recursive=true"` returned 34 entries with exactly those `lfs.oid`/`size` pairs, and the model API returned `sha: 30560ad575a56c39cc39047d1a474269338c77c0` — the revision all three download URLs pin (`models.yaml:884,893,902`). Whole pickles, not state dicts: `LIMN/train.py:263` is `torch.save(model, ...)`; the wrapper asserts `isinstance(model, native_model.LIMN)` (`workbench/replay/limn.py:129-130`). Registry declares `checkpoint_serialization: whole_model` (`:846`) and bundle `base_iter0_all_categories` with all three required (`:909-914`). `checkpoint_training_noise_pct` is `null` for all three (`:885,894,903`); `evaluation_noise_pct: 0`.

EVALUATION PROTOCOL: registry `native_protocol: fashioniq_val_split` (`:864`), defined at `models.yaml:15-19` as "Val Split (Pair-Union Gallery, Reference Excluded)". Checked against the pinned evaluator and the local dataset:

- Gallery: `LIMN/datasets.py:158-169` builds `test_targets_id` as a first-seen ordered union of each validation annotation's `source_img_id` then `target_img_id`. Endpoint subset, not the whole split file. Measured by me: dress val split 3817 / endpoints 2628 / 2017 queries; shirt 6346 / 3089 / 2038; toptee 5373 / 2902 / 1961; endpoints ⊆ val split True for all three, endpoints ⊆ train split False for all three.
- Reference exclusion: `LIMN/test.py:80-82` writes `-10e10` into the reference column, then `:85` `argsort`s. Score masking, not gallery removal.
- Recall: `LIMN/test.py:88-97`, percentage of the category's query count over the top-50 ranking (`:89` `for k in [1, 10, 50]`, `:94` `r = 100 * r / len(nn_result)`). No cross-category macro anywhere in the pinned evaluator.
- Independent corroboration for the paper's counting units: local caption triplets are 18000 train / 6016 val across the three categories, and local train+val image splits sum to 46609 + 15538 = 62147 — both matching the paper's own text (`paper.txt:532-535`: "∼46K images and ∼15K images are used for training and testing … 18K triplets for training and ∼6K for testing").

Protocol claim: **SUPPORTED**, and the paper-vs-`val` label is now **strongly consistent** (paper counts match the local val/train splits). I still did not find the paper stating which split it evaluates, so I keep a residual `UNKNOWN` on the literal sentence.

DATASET CONTRACT: `fashioniq_layout: fashioniq_ilearn_resized_training` (`:859`), `preparation_contract: manual_ilearn_resized` / `preparation_id: limn_fashioniq` (`:861-862`), contract body `workbench/registry/preparation_contracts.yaml:249-300`. Every declared path traces to a consumer: `captions/cap.{cat}.{train,val}.json` → `datasets.py:68,139`; `captions/correction_dict_{cat}.json` → `datasets.py:70,141`; `image_splits/split.{cat}.val.json` → `datasets.py:137`; `resized_image/{cat}/` → `datasets.py:44,112,121`; source-shipped `correct_files/correction_dict_{cat}.json` → README:52; resize recipe → README:51.

SAFETY (verified): `datasets.py:44` concatenates `path + 'resized_image'`, so the root must end in `/`; the wrapper supplies `path=f"{cache_root}/"` (`limn.py:139`) where `cache_root` comes from `tempfile.TemporaryDirectory` (absolute, no trailing separator) — correct.

Contract-level defect (unchanged): `preparation_contracts.yaml:284` `source_evidence: the model README documents this script and this output path` overstates — README:51 documents a **flat** resized directory, not the `per_category` layout the contract asks for (`:285`) and the evaluator reads (`datasets.py:112`). The transform itself matches (`README:51` points at `start_kit/resize_images.py`; fetched copy defaults `--image_size 256`, `/tmp/audit-final/limn/resize_images.py:59-69`, and `resize_image(image,size)` calls `image.resize(size, Image.ANTIALIAS)` at `:8-10`). **P3**, unchanged.

PREPROCESSING CONTRACT (image): wrapper rebuilds the factory semantics rather than inventing a transform — `limn.py:133-136` uses `open_clip.image_transform(visual.image_size, is_train=…, mean=visual.image_mean, std=visual.image_std)`, mirroring `open_clip.factory.create_model_and_transforms`. Native stage assignment: query and gallery both use `self.transform[1]` (`datasets.py:125`), i.e. the val transform; the evaluator never calls `get_img` (`:117`, index 0). MATCH (STATIC_AUDIT).

**New P2 finding (resampling mismatch).** The workbench states, and records in the receipt, that it resizes with LANCZOS:

```
workbench/scripts/prepare_model.py:88-91
#: The PIL resampling kernel the workbench-applied resize uses. The FashionIQ start_kit tool uses
#: the PIL default (nearest-neighbour); the workbench deliberately uses LANCZOS for the higher
#: quality the evaluators read off disk, and records the choice here so a change invalidates receipts.
RESAMPLING = "LANCZOS"
```

That comment is wrong on both halves. The pinned start_kit tool does not use the PIL default; it explicitly passes `Image.ANTIALIAS` (`resize_images.py:8-10`), which is `Resampling.LANCZOS` — so the workbench's LANCZOS choice actually *matches* the upstream tool, and the "deliberately … higher quality" framing describes a difference that does not exist. The comment also calls the PIL default "nearest-neighbour"; PIL's documented default filter for `Image.resize` is `BICUBIC` for non-palette images. So the receipt's `resampling: LANCZOS` is not the mismatch; the *comment* misstates provenance in a direction that would make a reader believe the derived images are a deliberate quality upgrade over the authors' recipe.

The digests are not affected: the kernel is not in `recipe_digest` (the digest covers the declared `parameters` only, `prepare_model.py:96-99,156`), and `resize_is_complete` never reads `resampling` (verified: `resampling` appears only at `prepare_model.py:88,140,143`). Recorded LANCZOS bytes therefore persist while the comment denies they equal the tool's output. **P2** (provenance text, not correctness of the bytes). Fix: correct the comment to "`Image.ANTIALIAS` == LANCZOS, the kernel the pinned start_kit tool passes", and fold the kernel into `recipe_digest` if a kernel change must invalidate receipts.

PREPROCESSING CONTRACT (text): `datasets.py:82-91` lowercases, maps every `string.punctuation` char to a space, strips/splits, applies the loaded dictionary by exact token lookup, joins with one space, and `concat_text` (`:89-90`) joins caption[0] + literal `" and "` + caption[1]. The wrapper never touches the caption — it constructs the native `FashionIQ` dataset (`limn.py:138-141`) — so normalization, ordering and the OpenCLIP tokenizer (`LIMN/model.py:71`) are the source's own. MATCH (STATIC_AUDIT).

AUGMENTATION CONTRACT: no training augmentation reaches an evaluated image. The only augmentation in the native path is OpenCLIP's `preprocess_train` (`RandomResizedCrop`); `datasets.py:120-127` selects index 1 (`preprocess_val`) for every validation image, and `get_img` (index 0) is never called for the FashionIQ val set. The wrapper builds both transforms for signature fidelity and passes `[preprocess_train, preprocess_val]` (`limn.py:133-140`), matching `LIMN/train.py:78,81` exactly. MATCH (STATIC_AUDIT).

GALLERY CONTRACT: gallery = first-seen ordered union of (reference, target) endpoint ids (`datasets.py:158-169`), consumed by `test.py:74-85`; one vector per endpoint id. The wrapper supplies the native dataset object, so gallery construction is native. Only wrapper-side influence is `isolated_cache_root` (`common.py:86-95`), which symlinks `captions/`, `image_splits/`, `resized_image/` into a fresh temp root so the native `{cat}_train_data.json` / `test_queries_{cat}.pkl` / `test_targets_{cat}.pkl` caches (`datasets.py:49-64`) are regenerated rather than reused. Data-integrity improvement; also a contract difference from the native code, which trusts a pre-existing cache. `preparation_contracts.yaml:300` already flags the caches as not provenance-validated.

METRIC CONTRACT: per-category R@1/R@10/R@50 from `test.py:88-97`, unchanged. The cross-category macro is workbench-authored (`common.py:98-108`), applied only to the full three-category set in canonical order (`limn.py:163`), and the parser recomputes it independently (`metrics_extraction.py:157-186`) and rejects duplicates/missing categories/out-of-range recalls (`:41-48,120-140`). Registry records no paper metric (`reported_r10/r50/mean` all `null`, `:866-868`), so the *registry* reference is structurally absent — but the committed **benchmark row** now exists (below).

SOURCE-TO-WORKBENCH PARITY: `replay_type: SOURCE_FAITHFUL_WORKBENCH_REPLAY` (`:847`) is accurate. `replay_category` (`limn.py:100-149`) imports the pinned `model`, `datasets`, `test` modules and verifies each resolved `__file__` is the pinned file (`:115-122`), hash-verifies before unpickling a whole model (`:128` → `common.py:36-48`), type-checks it (`:129`), moves it to CUDA (`:131`), builds the native dataset (`:138-141`), and calls `native_test.test(SimpleNamespace(local_rank=0, batch_size=32), model, dataset, category)` (`:142`). No part of encoder, composition, normalization, similarity or recall is reimplemented. The only wrapper-authored arithmetic is the macro.

RUNTIME STATUS: registry says `BLOCKED_RUNTIME_PREREQUISITES` (`:869`) / `command_status: REPLAY_COMMAND_IMPLEMENTED` (`:870`). Independently reproduced read-only this round with the workbench venv:

```
$ .venv/bin/python -c "... validate_source(Path('workbench/third_party/LIMN'))"   -> []        # was ['… dirty'] last round
$ ... validate_dataset(Path('data/fashionIQ_dataset'), ('dress',))
['LIMN dataset requirement missing: …/captions/correction_dict_dress.json',
 'LIMN dataset requirement missing: …/resized_image/dress']
$ ... validate_environment()
['LIMN environment version mismatch: expected torch==1.12.1, found 2.14.0+cpu',
 'LIMN environment version mismatch: expected torchvision==0.13.1, found 0.29.0+cpu',
 'LIMN environment package missing: open_clip_torch==2.20.0']
$ .venv/bin/python workbench/scripts/evaluate_models.py --model limn --dry-run
[RUN] limn / base_iter0_all_categories
[BLOCKED] required generated artifact missing: …/resized_image/{dress,shirt,toptee}
[BLOCKED] required generated artifact missing: …/captions/correction_dict_{dress,shirt,toptee}.json
[BLOCKED] FashionIQ image extension requirement: image directory missing: …/resized_image
[BLOCKED] checkpoint missing: …/artifacts/checkpoints/limn/0_dress_best_model.pt
[BLOCKED] checkpoint bundle incomplete: base_iter0_all_categories: …/0_dress_best_model.pt
$ .venv/bin/python workbench/scripts/pc_readiness.py --json   (limn model section)
state: BLOCKED_ENVIRONMENT   reason: execution environment not verified: WORKBENCH_LIMN_PYTHON=None …
$ echo $WORKBENCH_LIMN_PYTHON   # unset
$ ls workbench/artifacts/checkpoints/   # empty; no limn run directory anywhere
$ find . -name '*_aggregate.json' -not -path './.git/*'   # (none)
```

Status: **CONFIRMED BLOCKED** — three checkpoints absent, resized images and staged dictionaries absent, CUDA/dependency stack absent, `WORKBENCH_LIMN_PYTHON` unset. No evaluation has ever produced metrics. Every claim in this document is **STATIC_AUDIT**; nothing is RUNTIME_VERIFIED.

Furthermore, **the local config does not point at the dataset the rest of this audit describes**: `resolve_config()` returns `FASHIONIQ_ROOT=/home/heheboiz/data/cir/data/fashionIQ_dataset` (no `workbench/config/workbench.env`; no `FASHIONIQ_ROOT`/`CIR_DATA_ROOT` in the environment), while `workbench/configs/pc_autopilot.pilot_limn.yaml:59` sets `dataset_root: ../../data/FashionIQ` — a *symlink* to the same directory. Same bytes today, but the live default and the pilot's declared root are two different path strings; if the symlink is ever repointed the gate's `run_meta` would not notice, because it hard-codes `"dataset": "fashioniq"` (`evaluate_models.py:143`). **P3**.

PAPER-METRIC STATUS (updated). The registry still carries **no** transcribed paper row (`reported_r10/r50/mean` all `null`, `:866-868`). But the committed benchmark registry now has a verified row for exactly this experiment:

`workbench/registry/paper_benchmarks/papers.yaml:1935-1968` — `limn_fashioniq_base_iter0_table1`, `reference_status: VERIFIED`, `review_status: CONFIRMED`, `checkpoint_id: base_iter0_all_categories`, `gallery_protocol: fashioniq_val_split`, `split: val`, `reference_excluded: true`, `checkpoint_training_noise_pct: 0`, `reported_metrics {r10: 55.91, r50: 77.82, mean: null}`, `derived_metrics [mean 66.87 WORKBENCH_DERIVED]`, and per-category `dress 50.72/74.52, shirt 56.08/77.09, toptee 60.94/81.85` (`:1944-1945`). Its `verification_method` (`:1956-1963`) explicitly excludes the LIMN+ row and warns that arXiv v1 publishes a different campaign.

I re-verified the numbers from primary sources this round:

```
$ curl -sL https://arxiv.org/pdf/2305.09979v2 -o paper.pdf && pdftotext -layout paper.pdf paper.txt
paper.txt:597  LIMN  50.72 74.52  56.08 77.09  60.94 81.85  55.91 77.82  68.20 87.45  77.83
paper.txt:598  LIMN+ 52.11 75.21  57.51 77.92  62.67 82.66  57.43 78.60  68.37 88.07  78.22
paper.txt:678  (Table IV ablation, FashionIQ-Avg)  LIMN  55.91  77.82  77.83  72.19
$ curl -sL .../2305.09979v1 -o paper_v1.pdf && pdftotext -layout paper_v1.pdf paper_v1.txt
paper_v1.txt:602  LIMN  … 36.30  62.20  Avg        # v1 is the Birds-to-Words campaign — must not be cited
$ (50.72+56.08+60.94)/3 = 55.9133 -> 55.91 ;  (74.52+77.09+81.85)/3 = 77.82
```

Note for the record: the prior review's final section presented `56.084396467124634` for shirt R@10, whereas the paper prints `56.08`. Both are the same rounded cell — the registry's arithmetic check `(50.72+56.08+60.94)/3 = 55.9133` matches the paper's Avg, and the author-artifact macro recomputes to `55.9139` — but a reader should use the registry value, not the unrounded one, since the macro is defined over the rounded cells the paper prints.

**Registry/benchmark-row divergence (P2, changed this round).** The immediate-past review's REFRESH block listed the same values under the key `r50: 77.82`; the committed file today reads:

```
$ grep -n 'r50' workbench/registry/paper_benchmarks/papers.yaml | sed -n '/1935/,$p'   # (contextual)
1941:    mean: null
1942:    r10: 55.91
1943:    r50: 77.82
```

so 77.82 is correct and the earlier `77.82`-vs-`77.82` question is moot; what *did* change is that the row is now `review_status: CONFIRMED` and the gate's `mean` comparison is fed by the `derived_metrics` 66.87 rather than a printed mean. Both my independent routes agree, so the row is sound. The remaining registry-level gap is that `models.yaml:866-868` is still all-null, which means `metrics_extraction._check_parity` is permanently `PARITY_NOT_EVALUATED` for LIMN (`autopilot.py:772` reads `model.get("paper_metrics")`, which does not exist) and the only live verdict is `benchmark_gate`. That is internally consistent (`evaluate_models.py:1417-1425` labels the registry row `paper_sanity_metrics`), but two differently-named metric surfaces now disagree about whether a paper row exists for limn; the display-only `parity_status` field therefore always reads "not evaluated" on a run that the gate may grade as PASS. **P3** (labeling).

**P0 finding — the committed reference can never certify a LIMN run through the pipeline that produces it.**

`evaluate_models.py:137-140` builds `run_meta["checkpoint_training_noise_pct"]` from `plan.checkpoint_training_noise_pct`, which `guarded_plan` fills from `checkpoint.get("checkpoint_training_noise_pct")` (`:355`). For LIMN that is the **bundle anchor**: `candidates()` rewrites the selection to `checkpoint_id: base_iter0_all_categories` reusing the *dress* variant dict (`evaluate_models.py:212-216,220-226`), and all three variants record `null` (`models.yaml:885,894,903`). Selection succeeds — `select_reference` only constrains on noise when it is not `None` (`benchmark_references.py:170-173`) and returns `"exact condition match"` for limn under both `None` and `0`. Then `paper_gate._comparability_blockers` (`paper_gate.py:241-260`) sees `reference=0`, `run=None` and emits the **hard** blocker `"checkpoint_training_noise_pct missing from the run record: reference=0"`, which routes to `BLOCKED_CHECKPOINT_MISMATCH` at `:335-337` before any metric is compared. Reproduced end-to-end through the real report path, with a run whose metrics are *exactly* the paper's:

```
$ .venv/bin/python -c "... from workbench.scripts.evaluate_models import _benchmark_gate_verdict ..."
variant_provenance = {'caption_aliases': [], 'caption_alias_staged': False, 'approved_source_patch': None, 'warnings': []}
PERFECT RUN -> {'status': 'BLOCKED_CHECKPOINT_MISMATCH', 'passed': False, 'variant': False,
 'blockers': ['checkpoint_training_noise_pct missing from the run record: reference=0'],
 'detail': 'the reference and the run do not describe one experiment',
 'reference_evidence': None, 'comparisons': [], 'selection_reason': 'exact condition match'}
```

With the run record corrected to `checkpoint_training_noise_pct: 0`, the same call grades correctly (`ALERT_METRIC_DEVIATION` on injected per-category deviations, all nine comparisons evaluated). So the defect is exactly the `null`↔`0` join. Consequence: no LIMN run produced by `run.sh`, `autopilot`, or `evaluate_models.py --model limn` can receive a numerical verdict; every one is refused with a message that blames the *run* for a metadata gap in the *reference*, and a genuine 11.3% dress R@10 deviation would never be surfaced as `ALERT_METRIC_DEVIATION` because the gate halts before comparisons. The same gap is visible in the readiness report, which prints `"checkpoint_training_noise_pct": null` beside `reference_status: VERIFIED` and `reason: "exact condition match"` for all three variants — internally contradictory. Severity **P0** (silently suppresses every real deviation). Minimum fix: record the comparison on the *declared* condition, i.e. have `select_reference`/`_reference_constraint_count`/`_comparability_blockers` treat a reference row whose `checkpoint_training_noise_pct` is stated as constraining only when the run states one *and* the registry declares one for that variant — or, cleaner, have the LIMN bundle anchor carry an explicit `checkpoint_training_noise_pct: 0` with the `PUBLISHED_NOT_LABELLED` caveat the pilot config already documents, rather than `null`. Either way the current pair (null variant + 0 reference) must not coexist.

Heterogeneous with the same driver: `papers.yaml:1941` sets `checkpoint_training_noise_pct: 0` while `models.yaml:885` sets `null`; `pc_readiness` and the pilot config (`pc_autopilot.pilot_limn.yaml:34-40`) are explicit that the author publishes no training-condition label and that `strict_clean` must refuse. The reference row should not assert a condition the registry's own provenance says is unpublished — or it must be an explicitly declared workbench assumption, and then `run_meta` must carry it.

CATEGORY-GATE CHECK (this round's P1). Verified as a hypothesis against the code and by execution. `paper_gate.py:345-360` compares `dress/shirt/toptee × r10/r50` when both sides state them; a paper category with no observed counterpart lands in `category_unchecked` and is reported in `detail` on a pass (`:382-384`), never as a match. `evaluate_models.py:169` passes `extraction.category_metrics`. Direct test with the noise mismatch removed:

```
$ ... evaluate_gate(ref, obs, run_meta=<noise=0>, reproduced_category_metrics=cm)
ALERT_METRIC_DEVIATION
  r10   55.91 55.767  0.256 True        shirt_r10  56.08 60.000  6.990 False
  r50   77.82 77.820  0.000 True        dress_r10  50.72 45.000 11.278 False
  mean  66.87 66.793  0.115 True        toptee_r10 60.94 62.300  2.232 True
                                        (all *_r50 0.000 True)
```

Confirmed: a strong macro (0.26% off) can no longer conceal a broken category. But the LIMN parser makes the category route **unreachable**, because `parse_limn_output` returns `observed_metrics=None`/`category_metrics=<rows>` with status `EVALUATION_COMPLETED_METRICS_UNPARSED` whenever the declared macro disagrees with the recomputed one (`metrics_extraction.py:186-195`), and `_benchmark_gate_verdict` reads only `extraction.observed_metrics` for the aggregate (`evaluate_models.py:135`) while still forwarding `extraction.category_metrics` (`:169`). The gate's early `if not reproduced_metrics: return FAILED_OUTPUT_VALIDATION` then fires. A run whose per-category numbers are real but whose declared aggregate is stale is discarded wholesale instead of being graded per category. **P2** (defensive but lossy). Action: on a declared-vs-computed macro disagreement, keep `observed_metrics` (recomputed) and add the disagreement as a gate blocker rather than nulling the payload.

VARIANT-PROVENANCE CHECK (this round's P3). `evaluate_models.py:243-273` builds `variant_provenance` from preflight evidence only (declared caption aliases whose bytes match, plus an applied approved source patch) and the gate reads it as `run_meta["caption_alias_staged"]`/`["declared_variant"]` (`:146-147`), used at `paper_gate.py:370` to downgrade a pass to `PASS_WITHIN_5_PERCENT_VARIANT` — never to pass anything through. For LIMN the record is `{'caption_aliases': [], 'caption_alias_staged': False, 'approved_source_patch': None, 'warnings': []}` and `source_blocker` is absent, and `workbench/patches/` contains only `csmcir.md`/`csmcir_models_init_compat.patch`, so LIMN can never carry a variant label and its status is `BLOCKED_CHECKOUT`-independent. Field-level behavior is correct; the only limn-specific consequence is that the P0 above means the `variant` flag is moot until the reference join is fixed.

CRITICAL FINDINGS

1. **P0 — committed reference vs run metadata: LIMN can never be graded.** `papers.yaml:1941` records `checkpoint_training_noise_pct: 0`; `models.yaml:885,894,903` record `null`; `evaluate_models.py:140` forwards the bundle-anchor `null`; `paper_gate.py:241-260,335-337` refuses with `BLOCKED_CHECKPOINT_MISMATCH` before comparing. A run whose metrics are exactly the paper's is refused; a run 11% off on dress is refused identically. Evidence: the executed `_benchmark_gate_verdict` probe above.
2. **P2 — resampling provenance comment is false.** `prepare_model.py:88-91` claims the pinned start_kit tool uses "the PIL default (nearest-neighbour)"; the tool passes `Image.ANTIALIAS` (`/tmp/audit-final/limn/resize_images.py:8-10`), i.e. LANCZOS, so there is no quality delta and PIL's documented default filter is BICUBIC, not nearest. The receipt value (`:143`) is right; the sentence licensing it is not, and the kernel is outside `recipe_digest` (`:96-99`) so receipts cannot see a kernel change.
3. **P2 — `resize_is_complete` reproduces the old `.jpg`-only assumption.** `prepare_model.py:253-256` tries `{id}{recipe_ext}` then only `{id}.jpg`; `_run_resize` uses `_canonical_image_path` with the full declared extension list (`:365,368-383`). The local canonical root is 74381 `.png` and zero `.jpg`. A completed resize therefore cannot validate on this host: every identifier fails the receipt check and the whole set is regenerated on each resume.
4. **P2 — constructor-audit contract path cites `images/{id}.jpg`.** `workbench/registry/audits/cluster_d_csmcir_limn.yaml:269-271` declares `raw_input_dependencies: images/{id}.jpg` and `output_validation_contract: All validation split image IDs exist … as .jpg`; the canonical distribution here is 100% `.png` (verified per split: dress/shirt/toptee `{'.png': N}`). The consumer is correct; the audit document is not.
5. **P2 — no error handling around the native evaluator call.** `limn.py:137-142` calls `native_datasets.FashionIQ(...)` and `native_test.test(...)` unguarded. Because the dataset root is a `TemporaryDirectory` while the native code writes caches into it (`datasets.py:49-64`), a missing image or mid-run exception surfaces as a traceback rather than a structured `[BLOCKED]` line. Diagnosis-only.
6. **P2 — misleading `source_evidence` for the resize artifact.** `preparation_contracts.yaml:284` says the README documents "this output path"; README:51 documents a flat directory, contract asks `per_category` (`:285`). Transform and size do match. Unchanged from the prior review.
7. **P2 — a paper-published category can be left unchecked without failing the run.** `paper_gate.py:351-358` records `category_unchecked` in `detail` only. Correct per the frozen policy, but for LIMN the registry publishes all six category cells, so an observed set missing one is evidence of a truncated run rather than of an unstated paper metric; it should at least carry a blocker string, not only prose. (Static.)

MINOR FINDINGS

8. **P3 — `fashioniq_image_extensions: ['.jpg']` describes the derived tree, not `images/`.** `models.yaml:860` → `preflight.py:42-44` → `fashioniq_layout.py:83-89` maps `fashioniq_ilearn_resized_training` to `('resized_image', True)`. Scratch check with `.png` files in `resized_image/dress`: `['3817 of 3817 dress validation images lack .jpg in /tmp/limnprobe/root/resized_image/dress (first: B009PMCJLW.jpg)', …]`. Correct check, misleading field name.
9. **P3 — duplicate dataset-root identity.** Live default `data/fashionIQ_dataset` (symlink target) vs pilot config `data/FashionIQ` (symlink); gate `run_meta["dataset"]` hard-coded `"fashioniq"` (`evaluate_models.py:143`).
10. **P3 — `parity_status` and `benchmark_gate` disagree about whether a paper row exists for limn** (`models.yaml:866-868` null vs `papers.yaml:1935-1968` verified), because `autopilot.py:772` reads `model.get("paper_metrics")` which no registry entry defines. Display-only today.
11. **P3 — unguarded `.cuda()` and `torch.load` in the read path.** `limn.py:129-131` type-checks then calls `.cuda()` unconditionally; `common.py:46-48` passes `weights_only=False` only after hash equality (safe), but on a host with torch but no CUDA the failure is an uncaught `AssertionError` from the native code. The *preflight* catches it correctly (`limn.py:87-90`); the read path does not.
12. **P3 — `upstream_default_branch: null`** (`models.yaml:855`) while other entries use e.g. `main`. Cosmetic; the pin is a SHA.
13. **P3 — identical checkpoint sizes.** All three registered files are 1792010157 bytes with three distinct LFS oids (fetched). Plausible for three `torch.save(model)` pickles of one architecture; equality of size is not identity. Registry uses SHA-256; recorded only to pre-empt the inference.
14. **P3 — the README's own checkpoint pointer is stale.** `workbench/third_party/LIMN/README.md:14` links `https://huggingface.co/HaokunWen/LIMN/tree/main`, which now 307-redirects to `iLearn-Lab/TPAMI24-LIMN` (observed: `curl` returned `Temporary Redirect. Redirecting to /api/models/iLearn-Lab/TPAMI24-LIMN`). Superseded by the repo rename; no reproduction impact.
15. **P3 — author artifact rows.** `dress_0_metrics_best.json` / `shirt_0_metrics_best.json` / `toptee_0_metrics_best.json` fetched from the pinned Hub revision (etags `0f49d0e8…`, `16b29799…`, `5fd1316c…`; 208/210/213 bytes; parameter block `prefix_num 8, head 8, layer 2, norm 1.0, temperature 0.1`, matching README:81-83 and `papers.yaml:1917-1920`). Author macro recomputed by me: R@10 55.9139, R@50 77.8160, mean 66.8649 — matching the paper cells to rounding. The registry cites this as verification route (c) but records nothing in `models.yaml`. Informational; the benchmark row supersedes the prior review's "omitted author baseline" complaint.

UNRESOLVED EVIDENCE

- Whether the TPAMI paper states its FashionIQ gallery construction (course vs fine split, reference included/excluded) in words: **UNKNOWN**. I read the full v2 text (`paper.txt`) and the paper only reports dataset scale and "recall at rank k … as previous efforts [8], [11], [19], [33], [47]". No protocol sentence found. `models.yaml:863` `literature_split_label: val` remains an inference — though now a well-supported one, since the paper's own counts (∼46K train / ∼15K test images, 18K train / ∼6K test triplets, `paper.txt:532-535`) match the local train+val splits exactly (46609 images, 18000/6016 triplets).
- Whether the resized-image set at 256×256 is bit-identical to what the authors used: unresolvable by construction. The author publishes no resized images and no checksum. Marked UNKNOWN; not a blocker for the declared evaluator, which only reads whatever resized JPEGs are staged. The kernel is now at least *known to match* (ANTIALIAS == LANCZOS), contrary to the code comment.
- Whether the three checkpoint SHA-256 values are the file bytes' hashes: they are the Hub's LFS oids. No byte was downloaded; local delivery is UNVERIFIED.
- Whether `resolve_config()` will point at the same directory on the PC: the live default is a symlink and the pilot config uses a different path string. UNKNOWN until run on that host.
- Any GPU behaviour: UNKNOWN — no CUDA host, CPU-only torch 2.14.0+cpu in `.venv`.

FINAL VERDICT: **BLOCKED**

EXECUTION TRACE

`paper/README/config` → `source dataset` → `preprocessing` → `official evaluator` → `workbench invocation` → `checkpoint` → `output` → `metric comparison`.

1. **paper/README/config → declared stack.** README:20-24 states GPU 1×A100-40G, CUDA 12.4, `torch == 1.12.1`, `torchvision == 0.13.1`, `open-clip-torch == 2.20.0`. `models.yaml:871-878` records exactly `pytorch 1.12.1`, `torchvision 0.13.1`, `cuda 12.4`, `packages: [open_clip_torch==2.20.0]`, `python: UNKNOWN`, `confidence: VERIFIED_README`. **MATCH** including the honest UNKNOWN (README states no Python version). `LIMN/model.py:15,17` shows the real dependency surface — `open_clip.create_model_and_transforms('ViT-L-14', pretrained=…)` from a local path plus `open_clip.get_tokenizer` — consistent with the registered package.
2. **source dataset.** `fashioniq_ilearn_resized_training` (`models.yaml:859`) → `fashioniq_layout.py:37-41` → native reads at `datasets.py:68,70,137,139,141,44,112`. **MATCH.** Local data present for exactly these: `data/fashionIQ_dataset/{captions,image_splits,images}` + symlink `data/FashionIQ`.
3. **preprocessing.** Wrapper derives the transform from the *loaded* model's `visual` attributes (`limn.py:133-136`), mirroring `open_clip.factory`; validation images use index 1 only (`datasets.py:125`); text handled entirely natively (`datasets.py:82-91`, `model.py:71`). Resize recipe = `start_kit/resize_images.py` at 256 with ANTIALIAS/LANCZOS, emitted per category as `.jpg` (`prepare_model.py:579`, `_run_resize:445-503`). **MATCH**, with the P2 comment defect recorded above. **STATIC_AUDIT.**
4. **official evaluator.** `LIMN/test.py:23 def test(params, model, testset, category)`; the only in-tree caller is `LIMN/train.py:231`. `test.py` has no `__main__` and no standalone CLI anywhere in the pinned tree, so a harness must be supplied — a fact about the author's repository, not a workbench shortcut. `AUXILIARY_ASSET_AUDIT.md:321` states the same mapping and matches the code.
5. **workbench invocation.** `LIMNAdapter.command` (`adapters/models.py:123-137`) launches `workbench/replay/limn.py` with `--checkpoint-root/--dataset-root/--source-root`, adding `--category <cat>` unless the unit is the whole bundle. The replay: source pin (`limn.py:53-55` → `common.py:51-77`), per-category checkpoint SHA-256 + mapping status (`:32-51`), dataset (`:59-73`), environment (`:76-91`), then `importlib` of the three pinned modules with `__file__` identity enforcement (`:112-122`), hash-gated unpickle (`:128` → `common.py:36-48`), `model.cuda().eval()` (`:131`), native dataset (`:138-141`), native test (`:142`). **MATCH.**
6. **checkpoint.** Three URLs pin revision `30560ad5…` (`models.yaml:884,893,902`); fetched Hub tree returns LFS oids equal to `:888,897,906` and sizes equal to `:887,896,905`; fetched model API confirms the revision. Native save site `train.py:263` binds iteration 0 (`train.py:72` `--i` default 0) to those filenames; LIMN+ uses `augment_iteration` instead (`LIMN+/train.py:44,293`), so iteration 0 is LIMN-only. **MATCH at metadata level**; bytes UNVERIFIED locally.
7. **output.** `limn.py:143-149` builds `ReplayResult` from `metrics[f"{category}_r{n}"]`, the key format produced at `test.py:95`. Whole-run JSON at `:190-198` carries model, protocol, per-category rows and `complete_benchmark` from `macro_aggregate` (`common.py:98-108`, applied only to the full three-category set, `limn.py:163`). Output path create-only (`:199-208`). Consumer `parse_limn_output` (`metrics_extraction.py:74-206`) re-derives the macro and marks UNPARSED on duplicate/missing category, out-of-range recall, or >0.05 declared-vs-computed disagreement. **MATCH**, plus independent recomputation on the read side.
8. **metric comparison.** `_benchmark_gate_verdict` (`evaluate_models.py:124-171`) selects `limn_fashioniq_base_iter0_table1` with `"exact condition match"` and calls `evaluate_gate` — which returns `BLOCKED_CHECKPOINT_MISMATCH` because `run_meta.checkpoint_training_noise_pct` is `null` while the reference states `0` (`paper_gate.py:241-260,335-337`). **Injected probe with paper-exact metrics: `BLOCKED_CHECKPOINT_MISMATCH`, `comparisons: []`, `reference_evidence: None`.** Correcting the run record to `0` yields `ALERT_METRIC_DEVIATION` on the injected per-category deviations with all nine comparisons evaluated. The comparison path is thus *implemented and working*, but unreachable for every limn run as produced.

CHECKLIST

1. Registered model exists and is uniquely identified — **VERIFIED** — `models.yaml:843`; 16 entries, 16 unique ids (`yaml.safe_load`). — NONE — none.
2. Paper identity is a real, resolvable publication — **VERIFIED** — `models.yaml:852-853`; arXiv 2305.09979v2 fetched (`paper.txt:1-58`) carries the same title/authors/venue; README:166-169 same volume/number/pages/year. — NONE — none.
3. `upstream_repo_url` resolves to the pinned checkout — **VERIFIED** — `models.yaml:854` + `git rev-parse HEAD` + `git log -1`. — NONE — none.
4. `upstream_commit_sha` equals local checkout HEAD — **VERIFIED** — `7d7bc9b1…` at `models.yaml:856` and from `git rev-parse HEAD`. — NONE — none.
5. Checkout is clean at the pin — **VERIFIED** — `validate_source` → `[]`; `source_clean_and_pinned` → `(True, '<sha>', None)`; runtime `source_status` → all four classes 0. The prior round's P2 is fixed by `common.py:65-74` absorbing `runtime.is_bytecode`. — NONE — none.
6. Registry and pinned source agree on the source directory name — **VERIFIED** — `models.yaml:858` `source_dir: LIMN`; wrapper resolves `source_root / "LIMN"` (`limn.py:113`), matching README:32. — NONE — none.
7. Registry declares the correct number and identity of checkpoints — **VERIFIED** — three variants `models.yaml:881-908` matching fetched Hub siblings. — NONE — none.
8. Checkpoint filenames match the pinned save site — **VERIFIED** — `train.py:263`; `--i` default 0 at `train.py:72`; LIMN+ differs (`LIMN+/train.py:44,293`). — NONE — none.
9. "Three category-specific whole-model files" — **VERIFIED** — `train.py:263` `torch.save(model, …)`; `models.yaml:846,912`; type check `limn.py:129-130`. — NONE — none.
10. "Resized images required" — **VERIFIED** — `datasets.py:44,112,121`; README:51; wrapper requires the directory (`limn.py:67`). — NONE — none.
11. "Correction dictionaries required" — **VERIFIED** — consumed at `datasets.py:70,141`; shipped in the pin (`correct_files/correction_dict_{dress,shirt,toptee,shoes}.json`, counted 363/360/391/108 entries); README:52. — NONE — none.
12. "Base iteration 0 required" — **VERIFIED** — `train.py:72,263`; README:75; registry records only iteration-0 files. — NONE — none.
13. Every declared dataset/staging path has a native consumer — **VERIFIED** — table above, each row a `datasets.py`/README line. — NONE — none.
14. Native protocol is `fashioniq_val_split` — **VERIFIED** — `datasets.py:158-169`; `test.py:74-85`; `models.yaml:864`. — NONE — none.
15. Gallery is the pair union, not the full split — **VERIFIED** — `datasets.py:159-163`; measured 2628/3089/2902 endpoints vs 3817/6346/5373 val splits; endpoints ⊆ val, ⊄ train. — NONE — none.
16. Reference exclusion is a mask, not removal — **VERIFIED** — `test.py:80-85`. — NONE — none.
17. Recall denominator is the category query count — **VERIFIED** — `test.py:94`, `nn_result` one entry per `test_queries[i]`. — NONE — none.
18. Recall computed at K = 1, 10, 50 — **VERIFIED** — `test.py:89`. — NONE — none.
19. No source-level cross-category macro passed off as the paper's — **VERIFIED** — `test.py` returns per-category triples only; macro at `common.py:98-108`; parser recomputes (`metrics_extraction.py:157-186`). — NONE — none.
20. Paper's protocol label (`val` vs `original`) is verified — **UNKNOWN** — no protocol sentence in the paper text; `models.yaml:863` asserts `val`, now consistent with the paper's own counts (18000 train / 6016 val triplets vs `paper.txt:532-535`). — P3 — obtain the paper's evaluation-code statement or a cited reproduction before publishing a paper-vs-run comparison.
21. Registry declares the correct protocol family for the pinned evaluator — **VERIFIED** — `models.yaml:15-19` + `:864`; evaluator code matches. — NONE — none.
22. The generic FashionIQ builder expresses this protocol — **VERIFIED (documentation only)** — `FASHIONIQ_PROTOCOL_PARITY.md:64`; the dedicated replay intentionally does not call it. — NONE — none.
23. Workbench documents that masking ≠ removal — **VERIFIED (documentation)** — `FASHIONIQ_PROTOCOL_PARITY.md:45-48`; not exercised here. — NONE — none.
24. Declared image extension is the one the evaluator opens — **VERIFIED** — `models.yaml:860`; `datasets.py:112` `+ ".jpg"`; `fashioniq_layout.py:81-82`. — NONE — none.
25. The extension check inspects the directory the evaluator reads — **VERIFIED** — `fashioniq_layout.py:83-89`; scratch run produced `… lack .jpg in /tmp/limnprobe/root/resized_image/dress`. — P3 — rename/annotate the registry field.
26. Resized-image recipe matches a documented author tool — **VERIFIED** — README:51 → `start_kit/resize_images.py`; fetched script `--image_size` default 256 (`resize_images.py:59-69`); contract `preparation_contracts.yaml:281-283`. — NONE — none.
27. Resize output layout matches the evaluator read path — **VERIFIED** — contract `:285-286`; generator `prepare_model.py:530-531,579`; `datasets.py:112`. — NONE — none.
28. Resize recipe provenance sentence is accurate — **VARIANT** — `preparation_contracts.yaml:284` claims the README documents "this output path"; README:51 documents a flat directory. — P3 — adjust the wording.
29. Resize generator can consume the canonical image distribution — **VERIFIED** — `_canonical_image_path` (`prepare_model.py:365,368-383`) tries the derived extension then `.png/.jpg/.jpeg/.PNG/.JPG/.JPEG`; live check resolved `images/245600258X.png` for a `.jpg`-derived recipe and returned None for a nonexistent id. The prior round's P2 is fixed for the *generator*. — NONE — none.
30. Resize completion check can validate that same distribution — **VARIANT** — `resize_is_complete` (`prepare_model.py:253-256`) still tries `{id}{recipe_ext}` then only `{id}.jpg`; canonical root is 74381 `.png`, 0 `.jpg`; a completed resize can therefore never validate here and is regenerated on every resume. — **P2** — reuse `_canonical_image_path` (or the receipt's recorded `source_extensions`, written at `:552`) instead of the hard-coded `.jpg` fallback.
31. Resampling kernel claim in the code comment is accurate — **VARIANT** — `prepare_model.py:88-91` says the tool uses "the PIL default (nearest-neighbour)"; the fetched tool passes `Image.ANTIALIAS` (LANCZOS) and PIL's default filter for RGB is BICUBIC. — **P2** — correct the comment; fold the kernel into `recipe_digest` if it must invalidate receipts.
32. Correction dictionaries are staged byte-identically from the pin — **VERIFIED** — `prepare_model.py:305-340` (`_stage_source_assets`, copies from `<source>/correct_files/`, refuses a differing destination, reports absent declared assets); `preparation_contracts.yaml:294-296`. — NONE — none.
33. Correction dictionaries are the source's own — **VERIFIED** — present in the pin; loader `datasets.py:70,141`. — NONE — none.
34. Image transform derived from the model, not hard-coded — **VERIFIED** — `limn.py:133-136`. — NONE — none.
35. Eval transform uses no random crop — **VERIFIED** — `datasets.py:120-127` index 1 for every validation image; `get_img` (index 0, `:117`) is never called on this path. — NONE — none.
36. Text normalization reproduces the source — **VERIFIED** — `datasets.py:82-91`; wrapper never rewrites captions (`limn.py:138-141`). — NONE — none.
37. Caption composition order and separator match — **VERIFIED** — `datasets.py:89-90` `"{} and {}"`. — NONE — none.
38. Tokenizer is the source's own — **VERIFIED** — `model.py:71` `open_clip.get_tokenizer('ViT-L-14')`, reached through the unpickled model. — NONE — none.
39. Augmentation contract stated, non-empty only where it should be — **VERIFIED** — only `preprocess_train` exists in the native path and it never reaches an evaluated image. — NONE — none.
40. `--output` is never overwritten — **VERIFIED** — `limn.py:199-208` mode `"x"` + `FileExistsError` → `[BLOCKED]`. — NONE — none.
41. A partial-category run cannot emit a macro — **VERIFIED** — `common.py:100-103` requires `len(results)==len(CATEGORIES)` and exact canonical order; `limn.py:163`; tests `test_replay.py:34-39,91-95`. `pytest workbench/tests/test_replay.py -q` → 34 passed (assertion evidence, not runtime evidence). — NONE — none.
42. Metric parser independently recomputes the macro — **VERIFIED** — `metrics_extraction.py:157-186` rejects disagreement >0.05. — NONE — none.
43. Parser rejects out-of-range or non-finite recalls — **VERIFIED** — `metrics_extraction.py:41-48,64-72`; enforced on every category metric and on the declared aggregate (`:157-166`). — NONE — none.
44. Parser failure discards the whole payload — **VARIANT** — on a declared-vs-computed macro disagreement the parser nulls `observed_metrics` while still returning `category_metrics` (`metrics_extraction.py:186-195`), and the gate reads only `observed_metrics` (`evaluate_models.py:135`) → `FAILED_OUTPUT_VALIDATION`. Real per-category evidence is thrown away. — **P2** — keep the recomputed aggregate and emit the disagreement as a gate blocker.
45. Gate refuses to compare when no reference metric exists — **VERIFIED** — `paper_gate.py:300-325,366-368` (`METRIC_REFERENCE_UNVERIFIED`). Applies to models with no row; limn now has one. — NONE — none.
46. Registry carries a paper row — **VERIFIED NEGATIVE** — `models.yaml:866-868` all null; no number in `workbench/docs/**.md`. — P3 — note that the committed benchmark row supersedes it; a reader of `models.yaml` alone would conclude no reference exists.
47. An author-reference row for the exact registered experiment exists and is *reachable by the gate* — **VARIANT** — the row exists and is `VERIFIED`/`CONFIRMED` (`papers.yaml:1935-1968`) and `select_reference` returns it with `"exact condition match"`, but `evaluate_gate` refuses before comparing because `papers.yaml:1941` states `checkpoint_training_noise_pct: 0` while `models.yaml:885,894,903` state `null` and `evaluate_models.py:140` forwards the `null`. Probe with paper-exact metrics → `BLOCKED_CHECKPOINT_MISMATCH`, `comparisons: []`. — **P0** — reconcile the two fields; a run 11.3% off on dress is refused exactly like a perfect one.
48. Category metrics are compared and can no longer be averaged away — **VERIFIED** — `paper_gate.py:345-360`; probe with the noise field aligned produced `ALERT_METRIC_DEVIATION` with `dress_r10 11.278%`/`shirt_r10 6.990%` flagged although the macro was 0.26% off. — NONE — none.
49. A category the paper publishes but the run omits is reported, not matched — **VERIFIED** — `paper_gate.py:351-358,382-384`; `category_unchecked` reaches `detail`, never `comparisons`. — P2 (see finding 7) — record it as a blocker string, not only prose.
50. `variant_provenance` is recorded and only downgrades a pass — **VERIFIED** — `evaluate_models.py:243-273,1417-1427`; gate use at `paper_gate.py:370`; limn's record is empty and `workbench/patches/` has no limn patch. — NONE — none.
51. Autopilot rows carry `benchmark_gate`/`variant_provenance` and a deviation never demotes status — **VERIFIED** — `autopilot.py:954-962`; the previous round's JSON asserted this and the code matches. — NONE — none.
52. Declared stack matches README — **VERIFIED** — README:20-24; `models.yaml:871-878`. — NONE — none.
53. Python version claim is honest — **VERIFIED** — `models.yaml:874` `UNKNOWN`; README states none. — NONE — none.
54. CUDA requirement is real, not assumed — **VERIFIED** — `LIMN/test.py:39,57` `.cuda()`; `model.py:71` `.cuda()`; `limn.py:87-90` turns it into a blocker. — NONE — none.
55. Local environment satisfies the declared stack — **VERIFIED NEGATIVE** — `validate_environment()` → torch 2.14.0+cpu / torchvision 0.29.0+cpu / missing `open_clip_torch`; `WORKBENCH_LIMN_PYTHON` unset. — NONE — this is the BLOCKED reason.
56. Preflight fails closed on the local dataset — **VERIFIED** — `evaluate_models.py --dry-run` output quoted above. — NONE — none.
57. Replay refuses an unpinned source — **VERIFIED** — `common.py:75-76` with a wrong-commit probe returning `source pin mismatch: expected deadbeef, found 7d7bc9b1…`. — NONE — none.
58. Whole-model checkpoints are unpickled only after hash verification — **VERIFIED** — `common.py:36-48`, called at `limn.py:128`; `weights_only=False` is past the equality check. — NONE — none.
59. The replay cannot load a non-pinned module — **VERIFIED** — `limn.py:115-122`; `sys.path.insert(0, str(source_path))` at `:114`; `sys.dont_write_bytecode = True` at `:112`. — NONE — none.
60. The replay reimplements part of the official evaluator — **VERIFIED NEGATIVE** — `limn.py:142` calls native `test.test`; the only wrapper arithmetic is `common.py:98-108`. — NONE — none.
61. Constructor-audit declares the canonical image extension actually used — **VARIANT** — `workbench/registry/audits/cluster_d_csmcir_limn.yaml:269-271` records `images/{id}.jpg`; the local canonical distribution is 100% `.png` (74381 files; per-split check `{'.png': 3817/6346/5373}`). — **P2** — correct the audit document.
62. Dataset-root identity is the same in config and in the pilot config — **VARIANT** — live `resolve_config().FASHIONIQ_ROOT` = `data/fashionIQ_dataset`; `pc_autopilot.pilot_limn.yaml:59` = `data/FashionIQ` (a symlink to it); gate `run_meta["dataset"]` hard-coded (`evaluate_models.py:143`). — P3 — assert the resolved root in `run_meta` rather than a constant.
63. A reproduction result exists to audit — **NOT_RUN** — no limn run directory, no `*_aggregate.json` anywhere, `reports/` empty, only `mock` result files; `reproduction_status: BLOCKED_RUNTIME_PREREQUISITES` (`models.yaml:869`). — NONE — none.

FINAL VERDICT: **BLOCKED**

The reproduction path is source-faithful and correctly refuses to run (three category checkpoints, resized images, staged dictionaries, and a CUDA OpenCLIP stack are all absent, and `WORKBENCH_LIMN_PYTHON` is unset), and no evaluation has ever produced metrics — but the committed reference row for this exact experiment can never certify a LIMN run, because the gate compares a reference that states `checkpoint_training_noise_pct: 0` against a run record that carries `null`, so every run is refused with `BLOCKED_CHECKPOINT_MISMATCH` before a single metric is compared.

EXECUTION TRACE (commands run this round, read-only)

```
git -C workbench/third_party/LIMN rev-parse HEAD           -> 7d7bc9b116f594a65ac22457491edf28a88d3c3e
git -C workbench/third_party/LIMN status --porcelain       -> ?? LIMN+/__pycache__/ ; ?? LIMN/__pycache__/
git -C workbench/third_party/LIMN log --oneline -1         -> 7d7bc9b Update README.md
curl HF api iLearn-Lab/TPAMI24-LIMN tree @30560ad5…        -> 34 entries; 3 LFS oids == models.yaml:888/897/906
curl HF api iLearn-Lab/TPAMI24-LIMN (model)                -> sha 30560ad575a56c39cc39047d1a474269338c77c0
curl HF resolve …/dress_0|shirt_0|toptee_0_metrics_best.json -> R@10 50.7189/56.0844/60.9383 ; R@50 74.5166/77.0854/81.8460
curl arxiv.org/pdf/2305.09979v2 ; pdftotext -layout       -> paper.txt:597 LIMN 50.72/74.52 56.08/77.09 60.94/81.85 55.91/77.82
curl arxiv.org/pdf/2305.09979v1 ; pdftotext -layout       -> paper_v1.txt:602 LIMN Avg 36.30/62.20 (different campaign)
curl raw.githubusercontent…/fashion-iq/master/start_kit/resize_images.py
                                                           -> :8-10 image.resize(size, Image.ANTIALIAS) ; :67 --image_size default 256
.venv/bin/python  validate_source / validate_dataset / validate_environment / validate_checkpoint_bundle
                                                           -> [] / 2 missing / 3 mismatch / 1 missing
.venv/bin/python workbench/scripts/evaluate_models.py --model limn --dry-run
                                                           -> 6 generated artifacts + resized_image + 3 checkpoints blocked
.venv/bin/python workbench/scripts/pc_readiness.py --json  -> limn state BLOCKED_ENVIRONMENT
.venv/bin/python  select_checkpoints('limn', strict_clean)  -> BLOCKED_CHECKPOINT (no clean-trained variant recorded)
.venv/bin/python  select_checkpoints('limn', include_flagged_conditions) -> SELECTED base_iter0_all_categories
.venv/bin/python  references_for_model('limn')             -> 1 row, VERIFIED, r10 55.91 / r50 77.82 / mean 66.87 (WORKBENCH_DERIVED)
.venv/bin/python  select_reference('limn', noise=None|0)   -> same row, "exact condition match"
.venv/bin/python  evaluate_gate(perfect paper metrics)     -> BLOCKED_CHECKPOINT_MISMATCH ; blockers ['checkpoint_training_noise_pct missing…reference=0'] ; comparisons []
.venv/bin/python  evaluate_gate(noise=0, injected deviations) -> ALERT_METRIC_DEVIATION ; dress_r10 11.278% / shirt_r10 6.990% False
.venv/bin/python  _canonical_image_path(images, '245600258X', '.jpg') -> data/fashionIQ_dataset/images/245600258X.png
.venv/bin/python  image_extension_problems(.jpg, ilearn_resized_training) -> '3817 of 3817 dress validation images lack .jpg …'
.venv/bin/python  source_clean_and_pinned(limn)           -> (True, '7d7bc9b1…', None) ; source_status all classes 0
.venv/bin/python -m pytest workbench/tests/test_replay.py -q -> 34 passed (assertion evidence only)
```
