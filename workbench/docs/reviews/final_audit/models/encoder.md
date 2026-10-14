# encoder

**STATIC_AUDIT throughout.** No checkpoint downloaded, no weights loaded, no GPU job run, no evaluator executed. Evidence = pinned source text, committed registry rows, executed read-only commands (in-process registry/gate calls, `python3 -m workbench.scripts.pc_readiness`), and HTTP/PDF fetch of the real paper. Every claim labelled. Where a prior-round doc disagrees with the code, the code is quoted.

Repo: `/home/heheboiz/data/cir`, HEAD `eb36aa5ab0740e7dca7dd9c78515c3c49514b4e4`, branch `research/cir-failure-workbench` (`eb36aa5`), working tree **dirty** (`git status --porcelain` → 15 ` M` + 3 `??`). Line numbers: working tree for `models.yaml` / `papers.yaml`; all are byte-identical to HEAD for the encoder block (`git show HEAD:workbench/registry/models.yaml | grep -n "model_id: encoder"` → `402`).

---

## MODEL ID

`encoder`, `method_name: ENCODER` (`workbench/registry/models.yaml:406,422`). Registry block `:406-455`; exactly one row (`grep -c "model_id: encoder"` → 1 of 16 models).

## PAPER TITLE

`models.yaml:426-427` — "ENCODER: Entity Mining and Modification Relation Binding for Composed Image Retrieval", `https://ojs.aaai.org/index.php/AAAI/article/view/32541`.

Re-derived against the real paper (STATIC_AUDIT, network fetch succeeded on this round):
- `read https://r.jina.ai/https://ojs.aaai.org/index.php/AAAI/article/view/32541` → landing page, exact title, authors Zixu Li / Zhiwei Chen / Haokun Wen / Zhiheng Fu / Yupeng Hu / Weili Guan, DOI `10.1609/aaai.v39i5.32541`, *Proceedings of the AAAI Conference on Artificial Intelligence* 39(5), 5101–5109, published 2025-04-11. The DOI's volume/issue **`v39i5` matches the landing page's stated Vol. 39 No. 5**, and the galley text carries printed page numbers `5101`, `5103`, `5105`, `5106`, `5107` (`/tmp/audit-final/encoder/aaai_encoder.txt:40,88,261,285,304`) against the cited `5101–5109`. MATCH.
- Galley fetched: `curl -sSL -m 120 -A 'Mozilla/5.0' 'https://r.jina.ai/https://ojs.aaai.org/index.php/AAAI/article/download/32541/34696' -o /tmp/audit-final/encoder/aaai_encoder.txt` → 47320 bytes, "Number of Pages: 9".
- `models.yaml:424` `publication_year: 2025` MATCHES.

**Registry cross-check (see CRITICAL FINDINGS C1):** the *benchmark* row carries `paper_title: 'HINT: …'` and `paper_id: hint_icassp26` (`papers.yaml:1102-1103`), i.e. a different paper from `models.yaml:426`.

## PINNED SOURCE COMMIT

- Declared: `models.yaml:430` `upstream_commit_sha: 29a2a31d6a56f677bf450c3be7cdaef423fb7018`.
- Observed: `git -C workbench/third_party/ENCODER rev-parse HEAD` → `29a2a31d6a56f677bf450c3be7cdaef423fb7018`. MATCH. `git log --oneline -3` → single commit `29a2a31 Update README.md` (squashed/export copy; `git rev-list --all --count` = 1, so no branch-history claim is testable locally).
- Upstream tree at that SHA confirmed independently over the network: `read https://api.github.com/repos/iLearn-Lab/AAAI25-ENCODER/git/trees/29a2a31d6a56f677bf450c3be7cdaef423fb7018` → `truncated: false`, 12 blobs + `assets` tree, and **no `datasets1.py`**, no `open_clip_pytorch_model.bin`. The pinned checkout's `git ls-files` matches this listing exactly.
- `upstream_repo_url` `models.yaml:428` = `https://github.com/iLearn-Lab/AAAI25-ENCODER`; `upstream_default_branch: main` (`:429`). NOT re-verified over the network (no `git ls-remote` run).
- Checkout cleanliness: `git status --porcelain` in `workbench/third_party/ENCODER` → only `?? __pycache__/` (8 `.pyc`), which `runtime._source_dirty` (`workbench/backend/runtime.py:2040-2052`) excludes. Clean.

## CHECKPOINT IDENTITY

One variant (`models.yaml:444-455`): `checkpoint_id: fashioniq`, `filename: fashioniq.pt`, `download_url: null`, `source_url` = a **Google Drive folder** `https://drive.google.com/drive/folders/1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3`, `expected_size_bytes: null`, `expected_sha256: null`, `status: URL_UNRESOLVED`, `checkpoint_mapping_status: VERIFIED_METADATA`.

- README:249 links the same folder: `Our checkpoints are released at [Google Drive](...1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3)`.
- No local artifact: `checkpoint_path('encoder', variant, …)` → `/home/heheboiz/data/cir/workbench/artifacts/checkpoints/encoder/fashioniq.pt`, `exists → False` (observed). `runtime_plan_blockers` therefore appends `checkpoint missing: …/encoder/fashioniq.pt` (`workbench/backend/runtime.py:2145-2148`).
- No filename URL, no size, no digest → the file's identity is **UNKNOWN**. It is not established that any file in that folder is the FashionIQ `val-split` model. The extracted bundle says the same in its own words (`workbench/registry/paper_benchmarks/extracted/hint_icassp26.json`, encoder `checkpoint_mapping_evidence`: "no per-file URL or digest for ENCODER_FashionIQ.pth exists in the pinned tree; therefore which file inside the Drive folder is the clean FashionIQ checkpoint is NOT established") — but the *bundle* field is `UNVERIFIED` while the *published* row says `VERIFIED_METADATA` (§C4).
- `models.yaml:442` `strict_clean_checkpoint_available: true` is **contradicted** by the runtime computation of that same quantity (§C5).

## EVALUATION PROTOCOL

- Registry: `native_protocol: fashioniq_val_split` (`:435`), `literature_split_label: val` (`:436`), `supported_protocols: [fashioniq_val_split]` (`:437`); protocol table `workbench/registry/models.yaml:16-19` ("Pair-Union Gallery, Reference Excluded").
- Source default: `evaluate_model.py:15` `--fashioniq_split` default `'val-split'`; `datasets.py:64` `FashionIQ.__init__(path, transform=None, split='val-split')`.
- Adapter passes it explicitly: `EncoderAdapter.command` → `… --dataset fashioniq --fashioniq_split val-split --fashioniq_path <root>/ --ckpt_path <ckpt>` (`workbench/backend/adapters/models.py:36-41`).
- Category set `{dress,shirt,toptee}`: `evaluate_model.py:72-75`, `datasets.py:102`, `datasets.py:9-14`.
- MATCH between declared protocol and the source's default branch. STATIC_AUDIT.
- **The paper states no such protocol.** Re-derived from the galley: §4.1 "Datasets." names only "FashionIQ (Wu et al. 2021)"; §4.1 "Evaluation." says only "FashionIQ evaluation used R@10, R@50, and their category-wise averages" (`aaai_encoder.txt:247,255-257`). No gallery construction, no reference policy, no split name anywhere in the paper (`grep -n -i "split\|gallery" aaai_encoder.txt` → no protocol hits in the results sections). So the protocol is **source-derived, not paper-derived** — which the row should say.

## DATASET CONTRACT

Root is string-concatenated, so it MUST end with a separator:
`self.path + 'resized_image' | 'image_splits' | 'captions'` (`datasets.py:68-70`); the adapter appends the slash (`adapters/models.py:41`, `f"{request.dataset_root}/"`).

Per category the evaluator opens, in this order (`datasets.py:158-164`): `image_splits/split.{c}.val.json`, `captions/cap.{c}.val.json`, `captions/correction_dict_{c}.json`; images at `resized_image/{c}/{id}.jpg` (`datasets.py:148`).

A **fresh** constructor additionally opens `captions/cap.{c}.train.json` and `correction_dict_{c}.json` (`datasets.py:102-106`) and writes root-level caches `fashion_iq_data.json` + `test_queries_{c}.pkl`/`test_targets_{c}.pkl` (`datasets.py:74-88`), then trusts them blindly on later runs with no freshness check (`datasets.py:90-98`).

Registry: `fashioniq_layout: fashioniq_ilearn_resized` (`:433`), `fashioniq_image_extensions: [".jpg"]` (`:434`), `preparation_contract: manual_ilearn_resized` (`:425`). The layout's path list is `workbench/backend/fashioniq_layout.py:25-34` — it includes `resized_image/{c}`, `cap.{c}.val.json`, `correction_dict_{c}.json`, `split.{c}.val.json`, and **excludes** `cap.{c}.train.json`.

Locale (observed, read-only): `data/fashionIQ_dataset` = active `FASHIONIQ_ROOT` (`resolve_config()`); it contains `captions/`, `image_splits/`, `images/` (74381 files), `qwen_captions/`, but **no `resized_image/` and no `correction_dict_*.json`** (`find data -name "correction_dict*"` → empty; `find data -maxdepth 3 -name "resized_image*"` → empty). A second unrelated tree `data/FashionIQ` holds the canonical annotations.

## PREPROCESSING CONTRACT

- `evaluate_model.py:39-40` (numbered): `clip_path = '.'` then `open_clip.create_model_and_transforms('ViT-B-32', pretrained=os.path.join(clip_path, 'open_clip_pytorch_model.bin'))`; `model_try2.py:134-135` does the same inside `Backbone.__init__`, with a commented alternate `/root/CLIP-ViT-B-32-laion2B-s34B-b79K`.
- `evaluate_model.py:43` passes `transform=[preprocess_train, preprocess_val]` into `datasets1.FashionIQ(...)`.
- **Asymmetry is real and source-exact**: query/reference uses `stage=0` = `preprocess_train` (`datasets.py:177`), gallery uses `stage=1` = `preprocess_val` (`datasets.py:195`).
- Text: lowercase → ASCII `string.punctuation`→space → tokenize → dictionary substitution → join (`datasets.py:117-122`), composed as `cap0 and cap1` (`datasets.py:124-126`).
- Backbone version pinned: `requirements.txt:155` `open-clip-torch==2.24.0`; README:46 warns OpenCLIP version changes metrics.
- The registry pins **none** of this: `preprocessing_id: null` in the paper row (`papers.yaml:1117`), `packages: []` in `models.yaml:419`. And the binary the recipe needs is absent from the pinned tree (`evaluate_model.py:40`). See §C2/C8.

## AUGMENTATION CONTRACT

Evaluation supplies `[preprocess_train, preprocess_val]` and the **reference/query image is transformed with the training transform** (`datasets.py:177`, `stage=0`). `evaluate_model.py` never seeds any RNG — it has no `manual_seed` and no `--seed` argument (`evaluate_model.py:13-33`); only `train.py:58,229-236` seeds (default 42). Retrieval ranks are therefore stochastic run-to-run from the evaluator as packaged. Source property, not a workbench defect — but it caps what any reproduction may claim. P1.

## GALLERY CONTRACT

`val-split` branch (`datasets.py:185-196`): `test_targets_id` is the **first-seen ordered union** of each annotation's `source_img_id` then `target_img_id`; the gallery is built by `images[i]` for each id. Reference removal is a **mask, not a drop**: `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (`test.py:67-68`) *before* `nn_result = [np.argsort(-sims[i, :])[:50] …]` (`test.py:71`).

Workbench side: `src/evaluation/fashioniq.py:build_pair_union_gallery` emits reference-then-target first-seen; `src/evaluation/fashioniq_encoder.py:38-51` stable-sorts **then** drops the reference column. Same ordered set; ranking operation differs (mask vs post-sort column removal). `docs/PROTOCOL_AUDIT.md:38` states exactly this and is correct.

## METRIC CONTRACT

- Upstream emits **per-category only**: `out += [('{}_r{}'.format(name, k), r)]` for `k in [1,10,50]` (`test.py:75-81`), printed per category by `evaluate_model.py:72-75`. No macro R@10/R@50 and no mean anywhere in the source.
- Workbench aggregate is a workbench construction: `_synthetic_aggregate` (`workbench/scripts/evaluate_models.py:69-118`) defines `macro_r10`, `macro_r50`, `macro_mean` — but it only runs for models the registry marks `synthetic_source` (`:1386`), and encoder is not one (`grep -c synthetic_source workbench/registry/models.yaml` → 0).
- **The actual path is worse than "macro, not source metric": no parser exists at all.** `extract_aggregate_metrics` handles only `limn` and `csmcir` and otherwise returns `AGGREGATE_PARSER_UNAVAILABLE` / `parity_status = PARSER_UNAVAILABLE` (`workbench/backend/metrics_extraction.py:328-342`). Observed by executing it:
  ```
  $ python3 -c "from workbench.backend.metrics_extraction import extract_aggregate_metrics; r=extract_aggregate_metrics('encoder', "[('dress_r1', 24.1), ('dress_r10', 51.5), ('dress_r50', 76.95)]\n[('shirt_r1', 22.0), ('shirt_r10', 54.86), ('shirt_r50', 74.93)]\n", '', 0, paper_metrics={'r10':56.13,'r50':77.59,'mean':66.86}); print(r.extraction_status, r.observed_metrics, r.category_metrics, r.parity_status)"
  AGGREGATE_PARSER_UNAVAILABLE None None PARSER_UNAVAILABLE
  ```
  (The stdout above is ENCODER's own real format, `test.py:81`.) So `extraction.observed_metrics` is always `None` for encoder, the four category fields in the autopilot row stay `None` (`workbench/backend/autopilot.py:612-617,805-819`), and the gate can only ever return `FAILED_OUTPUT_VALIDATION` ("the evaluator produced nothing to compare") — regardless of how correct the run was.
- Registry `reported_mean: 66.86` (`:440`) = `(56.13+77.59)/2` = 66.86 and is **not** a paper column; the row labels it correctly as `WORKBENCH_DERIVED` (`papers.yaml:1125-1133`) but `models.yaml:440` has no provenance marker.

## SOURCE-TO-WORKBENCH PARITY

- **Adapter command: correct for the source's argparse** (`evaluate_model.py:13-33`: `--dataset`, `--fashioniq_split`, `--fashioniq_path`, `--ckpt_path`), and `--fashioniq_path <root>/` compensates for the string concatenation. cwd = source dir (`source_cwd`, `workbench/scripts/evaluate_models.py:185-186`; `subprocess.Popen(plan.command, cwd=plan.cwd, …)`, `:488`), which is what `pretrained='./open_clip_pytorch_model.bin'` requires.
- **But the entrypoint cannot import.** `evaluate_model.py:7` `import datasets1`; the pinned tree ships `datasets.py` only. Confirmed twice: `git ls-files` (no `datasets1.py`) and the GitHub tree API at the pinned SHA (`truncated: false`, no such path). Also `evaluate_model.py:43-56` calls `datasets1.FashionIQ/Shoes/CIRR/Fashion200k`, and `datasets.py:63,207,330,470` define exactly those five names. [INFERENCE] stale import name for the same module — unproven, and no alias may be added without author evidence.
- Preflight detects both source defects and blocks: observed
  ```
  $ python3 -m workbench.scripts.pc_readiness --model encoder
  encoder                    BLOCKED_DATASET
      FashionIQ fashioniq_ilearn_resized requirement missing: …/data/fashionIQ_dataset/resized_image
  summary: {"BLOCKED_DATASET": 1}
  ```
  and the full 10-blocker list from `preflight_blockers` (`workbench/backend/preflight.py:212-215` for the two encoder-specific entries): 7 layout paths + 1 image-extension problem + `ENCODER evaluator import missing: …/ENCODER/datasets1.py` + `ENCODER asset missing: …/ENCODER/open_clip_pytorch_model.bin`. The two encoder-specific checks are correct and well-placed.
- **The classification of that blocker list is wrong.** `classify` returns the first `BLOCKER_KINDS` match, and `("FashionIQ", BLOCKED_DATASET)` (`workbench/scripts/pc_readiness.py:52`) precedes `("asset", BLOCKED_EXTERNAL_ASSET)` (`:57`). So `ENCODER evaluator import missing` — the report's own words for a **P0 source defect** — classifies as `BLOCKED_EXTERNAL_ASSET` when it is the leading blocker (observed: `classify(['ENCODER evaluator import missing: y'])` → `('BLOCKED_EXTERNAL_ASSET', …)`), and is masked entirely behind `BLOCKED_DATASET` today. Context item 5 claimed "unverified blockers now classify as BLOCKED_EXTERNAL_ASSET"; that fix did not touch `evaluator import missing`, which is exactly the misclassification it was meant to remove. P1.
- Local `src/evaluation/fashioniq_encoder.py` is **not** ENCODER: `:120-136` calls `model.retrieve(...)`, no `def retrieve` exists anywhere in `src/` (`grep -rn "def retrieve" src/` → empty), and nothing outside `tests/unit/test_encoder_mode_isolation.py` + `workbench/tests/test_protocols.py:7,21` imports it. Must never be cited as reproduction.
- Selection-score parity: upstream sums `t[1][1]` (category R@10) across the three categories (`train.py:186-189`); local `fashioniq_encoder_selection_score` (`src/evaluation/fashioniq_encoder.py:73-78`) sums `{cat}_r10`. EQUIVALENT.
- Source `train.py` is also unimportable: `train.py:17` `import model as model`, and only `model_try2.py` exists. README:44's "All codes are transfered to the Github Repo" is false for the pinned tree.

## RUNTIME STATUS

**BLOCKED** (STATIC_AUDIT; no GPU job attempted, no run report exists). `models.yaml:441` `reproduction_status: NOT_RUN`. Four independent hard blockers: (1) evaluator unimportable (`datasets1`); (2) backbone binary absent and required at cwd; (3) checkpoint absent, no URL, no size, no hash, mapping gate refuses; (4) CUDA-mandatory (`evaluate_model.py:64` `Encoder_model = Encoder_model.cuda()` unconditional). Plus 7 missing layout paths in the local root. No encoder aggregate report exists under `workbench/artifacts/results/**` (observed: only `mock/` fixtures for airknow, habit, csmcir, sprc, pair, tgcir).

## PAPER-METRIC STATUS

**VERIFIED as a transcription; not reproduced.** Two independent routes on this round:

1. Real paper, fetched fresh: `aaai_encoder.txt:263` → `ENCODER(Ours) 51.51 76.95 54.86 74.93 62.01 80.88 56.13 77.59 26.97 65.59 86.48 59.68`, table caption at `:265` "Table 1: Performance comparison on FashionIQ and Shoes relative to R@k(%)". The publisher page (`paper_page: 5`) and the registry's excerpt (`papers.yaml:1149-1150`) match byte-for-byte on the row.
2. Pinned repo figure: `workbench/third_party/ENCODER/assets/fashion-results.png` read as an image → bottom row `ENCODER(Ours)`, FashionIQ avg cells **56.13 / 77.59**, Dresses 51.51/76.95, Shirts 54.86/74.93, Tops&Tees 62.01/80.88, Shoes 26.97/65.59/86.48/59.68. Identical to the PDF row.

Arithmetic re-derived: `mean(51.51, 54.86, 62.01) = 56.127 → 56.13`; `mean(76.95, 74.93, 80.88) = 77.587 → 77.59`; `(56.13+77.59)/2 = 66.86`. Reproduces the printed Avg cells exactly.

Backbone: paper §4.1 "Implementation Details" — "ENCODER is built upon the pre-trained CLIP (Radford et al. 2021) (ViT-B/32 version)" (`aaai_encoder.txt:249`) — MATCHES the row's `backbone: CLIP ViT-B/32` (`papers.yaml:1106`) and the code's `'ViT-B-32'` (`evaluate_model.py:40`).

Training condition: the paper **never states one**. Re-derived: `grep -n -i "noise\|condition\|clean\|perturb" aaai_encoder.txt` → hits only for "irrelevant factor perturbation" (a method concept, `:28,:58,:294`) and "Conditioned and composed image retrieval" (a citation title, `:310`). No noise levels, no clean/noisy block (contrast HABIT/AirKnow/intent, whose rows carry 20/50/80 blocks). `models.yaml:450` `checkpoint_training_noise_pct: 0` is therefore an **extraction assumption**, and the bundle's own `checkpoint_mapping_evidence` admits it: "checkpoint_training_noise_pct 0 is the paper's training condition, not a verified file property". §C6.

## CRITICAL FINDINGS

**C1 — P1 — The paper rows for `encoder`, `hint` and `pair` all cite the wrong paper.**
`encoder_fiq_clean_val_split` (`papers.yaml:1100-1103`):
```
- reference_id: encoder_fiq_clean_val_split
  model_id: encoder
  paper_id: hint_icassp26
  paper_title: 'HINT: Composed Image Retrieval with Dual-Path Compositional Contextualized Network'
```
while `models.yaml:426` says the ENCODER paper is "ENCODER: Entity Mining and Modification Relation Binding for Composed Image Retrieval" (AAAI-25, `:427`). The same two rows for `hint` and `pair` are the only other rows carrying `paper_id: hint_icassp26` (`grep -rn hint_icassp26` → `papers.yaml:1040,1102,1165`, no other file). Cause: the seed bundle `workbench/registry/paper_benchmarks/extracted/hint_icassp26.json` places ENCODER's and PAIR's Table 2 rows in an otherwise HINT paper block (`hint_icassp26.json:214-218` = `paper_id`/`paper_title` "HINT: …" for all three experiments), and `build_paper_registry.build_record` copies the paper block field-by-field (`workbench/scripts/build_paper_registry.py:88-92`). The consequence is not cosmetic: `BenchmarkReference.as_evidence()` (`workbench/backend/benchmark_references.py:82-99`) writes `paper_id` and `paper_title` into every run record's gate evidence, so a future ENCODER run would be certified against a row that names a different publication. Registry fix must start in the bundle (source of truth), then re-run `build_paper_registry.py`; also fix the bundle's `paper` block, which still lists `pdf_sha256`/`paper_url` for HINT's arXiv paper while carrying ENCODER's rows.

**C2 — P0 — The pinned evaluator is unimportable.** `evaluate_model.py:7` `import datasets1`; absent from `git ls-files`, from the GitHub tree at the pinned SHA, and from the filesystem scope searched. Every execution path dies at import before any metric exists. Correctly detected and blocked by `preflight.py:212-213`. No author-backed fix exists in the pinned tree.

**C3 — P0 — A correct run of the pinned evaluator can never pass the gate.** `extract_aggregate_metrics` has no encoder parser (`metrics_extraction.py:328-342`), so `observed_metrics` is `None` → `evaluate_gate` returns `FAILED_OUTPUT_VALIDATION`, "the evaluator produced nothing to compare" (`paper_gate.py:301-303`). Observed both ends by execution: extraction → `AGGREGATE_PARSER_UNAVAILABLE`, and the gate → `FAILED_OUTPUT_VALIDATION` for encoder with `observed=None`. Meanwhile upstream prints a fully regular, trivially parseable `[('dress_r1', …), ('dress_r10', …), ('dress_r50', …)]` per category (`test.py:81`, `evaluate_model.py:74-75`). Until a parser exists, the workbench has no numeric line of sight to ENCODER at all. P0 because it silently converts a scientific result into "no data" at the acceptance boundary.

**C4 — P1 — `checkpoint_mapping_status` says VERIFIED while the runtime refuses it.** `models.yaml:455` `VERIFIED_METADATA`, but `runtime_plan_blockers` appends "checkpoint mapping unresolved" for anything else (`workbench/backend/runtime.py:2147-2148`), so the declaration is inert; `checkpoint_selection._describe` displays it (`workbench/backend/checkpoint_selection.py:99-102`). The *source* bundle says the opposite: `hint_icassp26.json` encoder `checkpoint_mapping_status: "UNVERIFIED"`. The committed row was promoted to `VERIFIED_METADATA` even though `download_url: null` / `expected_sha256: null` / `expected_size_bytes: null` (`:447,452-453`) — nothing in the pinned README, the tree, or the paper binds any Drive file to FashionIQ. Registry validator does not catch it because its mapping/source-state consistency rule keys off `status: URL_UNRESOLVED`, which is orthogonal to mapping (`workbench/backend/registry.py:30-37,85-103`).

**C5 — P1 — `strict_clean_checkpoint_available: true` is false.** `models.yaml:442` says true; the runtime computation of that exact field returns false (observed, in-process):
```
$ python3 -c "…select_checkpoints(model_by_id('encoder')).to_dict()"
{'status': 'BLOCKED', 'reason_code': 'BLOCKED_CHECKPOINT',
 'reason': 'no obtainable checkpoint variant: fashioniq is registered without an author-linked download URL and is not present locally',
 'strict_clean_checkpoint_available': False, 'conditions': {'fashioniq': 0}}
$ registry strict_clean field: True
```
`select_checkpoints` requires `checkpoint_mapping_status == VERIFIED_METADATA` **and** an obtainable variant (`checkpoint_selection.py:126-130,83-96`). The declared value is a pre-download wish, and it is the field the frontend type reads (`workbench/frontend/types.ts:4` — `strict_clean_checkpoint_available:boolean|null`). Two representations of one fact, disagreeing, with no test pinning them together (`grep -rn "strict_clean" workbench/tests/` → only `selection.*`, never `registry.*`).

**C6 — P2 — `checkpoint_training_noise_pct: 0` is an extraction assumption the paper does not support.** `models.yaml:450` and `papers.yaml:1108`. The paper states no training condition (negative search above). The gate's *first* rule is that a 0%-noise row cannot be the reference for a noise-trained checkpoint (`paper_gate.py:13-15,241-258,324-337`), so this asserted `0` is exactly the join the LIMN fix was written to stop doing — and the same critic who fixed LIMN left ENCODER's asserted `0` in place. Both sides are self-consistent only because the registry asserts 0 on both the variant and the row. The honest value is `null` unless and until a file property proves it (mirroring `hint_icassp26.json`'s own caveat: "not a verified file property"). P2, not P1, because refusing on `noise is None` is not an option in `select_reference` — the row would simply stop matching, which is the correct outcome.

**C7 — P1 — The OpenCLIP backbone binary is required, absent, and unregistered.** `evaluate_model.py:40` and `model_try2.py:135` both need `./open_clip_pytorch_model.bin`; adapter cwd is the source root. `grep -c encoder workbench/registry/auxiliary_assets.yaml` → 0, and the encoder has no `preparation_id` (observed `preparation_contract_for_model(encoder) → None`), so the asset has no acquisition row, no hash and no destination. The cluster audit proposed a URL (`workbench/registry/audits/cluster_a_encoder_split.yaml:218` → `https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin`), which is consistent with the commented LAION path at `model_try2.py:134` but is **not** implemented anywhere. Detected correctly as a blocker (`preflight.py:214-215`), but the machine-readable contract knows nothing about it.

**C8 — P1 — Evaluation is stochastic and unseeded.** Query/reference uses `preprocess_train` (random-resized-crop) at `datasets.py:177`; `evaluate_model.py` seeds nothing. Any future number is run-to-run stochastic and must be reported as such.

**C9 — P1 — `preparation_contract: manual_ilearn_resized` resolves to no contract, and the layout omits inputs the evaluator needs.** Observed `preparation_contract_for_model(encoder) → None`; `manual_ilearn_resized` is only an enum member (`registry.py:73`) with no `preparation_id`, and `preparation_contracts.yaml` has no encoder row (`grep encoder …` → 0 hits; ids are `clvc_net_/dcnet_/clip4cir_/tgcir_/sprc_/limn_fashioniq`). Consequently preflight falls through to `fashioniq_required_paths` → layout paths (`preflight.py:27-30`), which: (a) cover only validation-time inputs, missing `cap.{c}.train.json` that a fresh constructor reads at `datasets.py:103`; (b) have no cache-isolation policy for the root-level `fashion_iq_data.json`/PKLs the evaluator writes (`datasets.py:77-88`) — LIMN and SPRC's contracts carry an explicit note against trusting or copying foreign caches (`preparation_contracts.yaml:219,300`), encoder's carries nothing; (c) leave `correction_dict_{c}.json` with **no published provenance** — required at `datasets.py:163`, not in `git ls-files`, no URL in the README. The correction dictionary is a model input, so this is a provenance hole in the caption stage.

**C10 — P1 — `cuda: '12.6'` contradicts the source.** `models.yaml:418` vs README:132 `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`, while the adjacent comment (`:413-417`) calls its value "the iLearn family's pinned one". The GPU requirement itself is real (`evaluate_model.py:64`) and the comment's reasoning for not writing UNKNOWN is sound; the version it substitutes is not attested by this repo. Also `packages: []` (`:419`) against a 300+ line `requirements.txt` pinning `open-clip-torch==2.24.0` (`:155`), `torch==1.13.1` (`:276`), `torchvision==0.14.1` (`:280`), `numpy==1.21.6` (`:149`), `transformers==4.30.2` (`:284`).

## MINOR FINDINGS

- **M1 — P2 — `pc_readiness.classify` misfiles the P0 source defect.** `("FashionIQ", BLOCKED_DATASET)` at `pc_readiness.py:52` precedes `("asset", BLOCKED_EXTERNAL_ASSET)` at `:57`, so `ENCODER evaluator import missing` classifies as `BLOCKED_EXTERNAL_ASSET` when it leads (observed) and is hidden behind `BLOCKED_DATASET` today. Context item 5's claim ("unverified blockers now classify as BLOCKED_EXTERNAL_ASSET (was wrongly BLOCKED_IMPLEMENTATION)") does not cover this string; the report's own sentence points the operator at resized images rather than at `datasets1.py`. Fix: add `("evaluator import missing", BLOCKED_IMPLEMENTATION)` above the generic matches — it is a property of the published source, which no local preparation can fix.
- **M2 — P3 — Prior-round doc is stale: required-train-input line numbers.** `workbench/docs/reviews/models/encoder.md`, DATASET CONTRACT, cites `datasets.py:103-106` for the fresh-constructor train reads; at HEAD they are `datasets.py:102-106` (`for name in ['dress','shirt','toptee']` at `:102`, `cap.{name}.train.json` at `:103`, `correction_dict_{name}.json` at `:105`). Its `datasets.py:185-196` gallery citation is exact. It also says `workbench/registry/models.yaml:382-431`; the block is now `:406-455` (models.yaml carries 24 pre-encoder lines of other models).
- **M3 — P3 — Prior-round doc misstates doc provenance.** `encoder.md`, SOURCE-TO-WORKBENCH PARITY, says "`src/evaluation/fashioniq_encoder.py:38-51` … bump-free"-style claims and asserts the file is "not wired to any protocol dispatcher (`build_fashioniq_gallery` / `evaluate_fashioniq` are never imported outside tests/`src/evaluation/fashioniq.py`)". Actually `src/evaluation/fashioniq_encoder.py` **is** imported, by `workbench/tests/test_protocols.py:7,21` and `tests/unit/test_encoder_mode_isolation.py:8` — i.e. by tests only, which is the same conclusion but the named files are wrong.
- **M4 — P3 — README command is wrong twice.** README:252-253 documents `python3 evaluation_model.py --model_dir checkpoints/ENCODER_{…}.pth`; the file is `evaluate_model.py` and its loader argument is `--ckpt_path` (`evaluate_model.py:32`), not `--model_dir`. The adapter is right; the README is wrong. (This is the upstream README, not a workbench defect.)
- **M5 — P3 — `reported_mean: 66.86` has no provenance marker in `models.yaml`.** `:440` records it beside `reported_r10/r50` (`:438-439`) with no hint that it is arithmetic, while the benchmark row labels it `WORKBENCH_DERIVED` (`papers.yaml:1126-1133`). Consumers that read `models.yaml` directly get an unlabelled derived number (e.g. `autopilot.py:620-622` → `paper_mean`, surfaced as `paper_metrics.mean` at `:842`).
- **M6 — P3 — Cluster audit over-claims branch history.** `workbench/registry/audits/cluster_a_encoder_split.yaml:206,333`: "missing from the pinned checkout **across all git branches/commits**". The local checkout has one commit (`git rev-list --all --count` → 1), so this is untestable locally; it is however corroborated for the pinned SHA by the GitHub tree API (`truncated: false`). Wording should be "the pinned tree" or cite the API result.
- **M7 — P3 — Stale-doc reference to `paper_metrics` is now correct but the note is at the wrong layer.** `autopilot.py:771-774` correctly documents that `models.yaml` has no `paper_metrics` key (verified: `grep -c paper_metrics workbench/registry/models.yaml` → 0) and now reads `reported_r10/r50/mean`. That fix is real and matches context item 3. Remaining gap: the fallback chain `({…} or report_payload.get("paper_sanity_metrics") or {})` (`:774-775`) means a fabricated run JSON can inject paper values into the live row — item 3's fix is defeated whenever `reported_*` is null (encoder: they are not null, so encoder is not affected).
- **M8 — P3 — `data/` holds two FashionIQ trees.** `resolve_config().FASHIONIQ_ROOT` = `data/fashionIQ_dataset` (no `resized_image/`, no correction dicts), while `data/FashionIQ` holds canonical `captions/`, `image_splits/`, `images/`. A human following `FASHIONIQ_ROOT` and a human following `data/FashionIQ` see different files. Not an encoder-specific defect; noted because encoder's 7 missing layout paths live in the first tree.

## UNRESOLVED EVIDENCE

- **Google Drive folder contents** (`1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3`): not listed this round (no fetch attempted; folder listing requires authenticated/quota-bearing access). Files, sizes, digests → UNKNOWN. Therefore "is `fashioniq.pt` the FashionIQ val-split state dict?" → UNKNOWN.
- **Whether `datasets1` is an unpublished author file or a rename defect** → UNKNOWN. No evidence in the pinned tree, in the GitHub tree API, or in the paper's code-availability statement (the paper names no repository at all).
- **Provenance of `captions/correction_dict_{dress,shirt,toptee}.json`** → UNKNOWN. Required at `datasets.py:163`; absent from `git ls-files`; the README's FashionIQ section (`README.md:154-186`) documents `captions/`, `image_splits/`, `dress/`, `shirt/`, `toptee/` and never a correction dictionary. It is the same asset SPRC's and DCNet's contracts flag as unprovenanced.
- **Which artifact the Drive folder's FashionIQ file was trained from, and whether the paper's 10-epoch / batch-128 / κ=0.8,γ=0.5,μ=0.5, P=4, E=3 recipe (`aaai_encoder.txt:249-253`) produced the released weights** → UNKNOWN. Note the code's own claim differs: `evaluate_model.py:35` says `{P,wc,N_P}: … Shoes&FashionIQ:{3,2,2}` — i.e. **P=3 for FashionIQ** — while the paper says "latent factor number P to 4" (`:251`). The adapter passes no `--P`, so it runs the argparse default `--P 3` (`evaluate_model.py:22`), which agrees with the code comment and disagrees with the paper. Flagged as a tension, not a defect: notable for anyone who later tunes these.
- **Any runtime metric for encoder** → UNKNOWN. No GPU run, no checkpoint, no dataset preparation, no aggregate report on disk.
- **`upstream_default_branch: main` and the repo URL** were not re-verified over the network this round.
- **Paper PDF byte digest** → not computed this round; the PDF was fetched only through the text proxy. (`hint_icassp26.json` records `pdf_sha256: fbe97798e89ef21d7f30a895a151b94ff062f708c5060ac397e494adf05f2373`, but that digest belongs to the **HINT** arXiv paper and must not be attached to ENCODER — see C1.)
- **Direct AAAI download link** `…/download/32541/34696` worked through the proxy; direct `curl` to `ojs.aaai.org` failed at the TLS layer (`HTTP/2 stream 1 was not closed cleanly: PROTOCOL_ERROR`). No direct-fetch hash.

## FINAL VERDICT

**BLOCKED** — the pinned ENCODER evaluator cannot be imported (`import datasets1` at `evaluate_model.py:7`; the module is absent from the pinned tree, from the GitHub tree at the audited SHA, and from the checkout), cannot be executed (`./open_clip_pytorch_model.bin` absent and unregistered, `fashioniq.pt` absent with no URL/size/hash and a runtime-refused checkpoint mapping, CUDA-mandatory at `evaluate_model.py:64`), and even a successful run could not be judged (no encoder parser in `metrics_extraction.py` → gate returns `FAILED_OUTPUT_VALIDATION`); the only verified numbers remain the paper's own Table 1 FashionIQ row (56.13/77.59, STATIC_AUDIT — re-derived this round from the real 9-page AAAI-25 galley and the pinned results figure), while the registry's own governance fields for this model disagree with the runtime (`checkpoint_mapping_status` VERIFIED vs `runtime.py:2147` refusal; `strict_clean_checkpoint_available: true` vs computed `false`; `checkpoint_training_noise_pct: 0` asserted by a paper that states no training condition) and its benchmark row cites the wrong paper.

---

# EXECUTION TRACE

`paper/README/config → source dataset → preprocessing → official evaluator → workbench invocation → checkpoint → output → metric comparison`

1. **paper/README/config →** README:249 (Drive folder), README:46 (OpenCLIP-version warning), README:128 (`python=3.9`), README:132 (`cu118`); registry mirrors at `models.yaml:427,428,410,418`. Paper fetched this round: `aaai_encoder.txt` (Table 1 row at `:263`, §4.1 at `:247-257`). One contradiction (cuda 12.6 vs cu118, C10) and one wrong paper identity (C1).
2. **→ source dataset**: `evaluate_model.py:17` `--fashioniq_path` (default `.`) → string-concatenated at `datasets.py:68-70` (trailing separator required) → read at `datasets.py:159-164` and images at `datasets.py:148`. Adapter supplies `f"{request.dataset_root}/"` (`adapters/models.py:41`) and `dataset_root = args.dataset_root or config.FASHIONIQ_ROOT` (`evaluate_models.py:1759`). ARROW OK as written; the data itself is absent (7 layout paths, observed).
3. **→ preprocessing**: `evaluate_model.py:39-43` `open_clip.create_model_and_transforms('ViT-T-32'…)`-equivalent call needing `./open_clip_pytorch_model.bin` → `[preprocess_train, preprocess_val]` → `stage=0` query (`datasets.py:177`) / `stage=1` gallery (`datasets.py:195`). ARROW BROKEN: binary absent, and unseeded (C7, C8).
4. **→ official evaluator**: entry is `evaluate_model.py:60-64` (`torch.load` → `load_state_dict(strict=False)` → `.cuda()`) → `:72-75` per-category `test.test` → `test.py:39,49` feature extraction → `test.py:67-68` mask → `:71` argsort → `:75-81` R@1/10/50. **Never reached**: `evaluate_model.py:7 import datasets1` raises `ModuleNotFoundError` first. Asserted from source (the module is provably absent); not run — running is unnecessary and prohibited.
5. **→ workbench invocation**: `EncoderAdapter.command` (`adapters/models.py:41`) → `python <src>/evaluate_model.py --dataset fashioniq --fashioniq_split val-split --fashioniq_path <root>/ --ckpt_path <ckpt>`; cwd = source root (`evaluate_models.py:185-186`, `:488`); gated by `blockers()` → `runtime_plan_blockers` + `preflight_blockers` (`:276-285`). ARROW OK; refused before launch (observed 10 preflight + 1 runtime blocker).
6. **→ checkpoint**: `models.yaml:444-455` → `download_url: null`, folder URL, `expected_sha256: null`, `checkpoint_mapping_status: VERIFIED_METADATA`; `runtime_plan_blockers` appends `checkpoint missing: …/encoder/fashioniq.pt` **and** `checkpoint mapping unresolved` (`runtime.py:2145-2148`). ARROW BROKEN twice over.
7. **→ output**: upstream prints `print(t)` per category (`evaluate_model.py:74-75`; shape `[('dress_r1', …), ('dress_r10', …), ('dress_r50', …)]`, `test.py:81`). No macro, no mean, no JSON. NOT REACHED.
8. **→ metric comparison**: `extract_aggregate_metrics('encoder', …)` → `AGGREGATE_PARSER_UNAVAILABLE`, `observed=None`, `parity=PARSER_UNAVAILABLE` (observed). Gate → `FAILED_OUTPUT_VALIDATION` (observed). Registry 56.13/77.59 vs paper Table 1 FashionIQ Avg 56.13/77.59 → MATCH (STATIC_AUDIT, two independent routes). `reported_mean 66.86` = arithmetic mean of the two, not a paper column. **No reproduced number exists.** Counterfactual only: with a parser supplying values, the gate does compare (observed `PASS_WITHIN_5_PERCENT` on 9 comparisons — 3 aggregate + 6 category — for `{'r10':54.0,'r50':75.0,'mean':64.5}` plus categories).

---

# CHECKLIST

1. **Registry entry exists, loads, unique** — VERIFIED — `models.yaml:406-455`; `grep -c "model_id: encoder"` → 1 of 16; `load_registry()` returns the dict, all 28 fields present — NONE — none.
2. **`upstream_commit_sha` equals checkout HEAD** — VERIFIED — `models.yaml:430` = `git -C workbench/third_party/ENCODER rev-parse HEAD` = `29a2a31d6a56f677bf450c3be7cdaef423fb7018` — NONE — none.
3. **Pinned tree matches upstream at that SHA** — VERIFIED — GitHub tree API at the pin returns `truncated:false`, 12 blobs + `assets`, identical to `git ls-files` — NONE — none.
4. **Checkout clean** — VERIFIED — `git status --porcelain` → only `?? __pycache__/`, excluded by `runtime.py:2040-2052` — NONE — none (P3 cosmetic: `.pyc` present).
5. **Checkout has testable history** — UNKNOWN — `git rev-list --all --count` → 1 (squashed export) — P3 — no branch-level claim can be made locally.
6. **`method_name`/`paper_title` match the real paper** — VERIFIED — `models.yaml:422,426` vs fetched AAAI page + galley header — NONE — none.
7. **`paper_url` resolves to that paper** — VERIFIED — landing page via proxy: exact title, authors, vol 39 no 5 pp. 5101–5109, matching galley folios `5101/5103/5105/5106/5107` — NONE — none (direct curl to ojs fails at TLS; proxy route documented).
8. **`upstream_repo_url` correct** — VARIANT — `models.yaml:428` names `iLearn-Lab/AAAI25-ENCODER`; the fetched paper names no repository at all, so this is README/tree-derived, not paper-attested — P3 — leave; note the provenance.
9. **`source_available`/`source_dir` correct** — VERIFIED — `:431-432`; `workbench/third_party/ENCODER` is a real Git root (`git rev-parse --show-toplevel` from it succeeds) — NONE — none.
10. **Registration consistent with the source's category model** — VERIFIED — `:420` `category_model_policy: shared`; `evaluate_model.py:60-63` loads **one** state dict and `:72-74` reuses it for all three categories; exactly one variant (`:444`) — NONE — none.
11. **`category_model_policy: shared` has a production consumer** — VARIANT — zero non-test readers (`grep -rn category_model_policy workbench/ | grep -v tests/` → only `replay/limn.py:191`, its own payload); pinned only by tests (`test_strict_acceptance_matrix.py:130-134`, `test_replay.py:108-111`, `test_adversarial_orchestration.py:37-40`, `test_protocols.py:139-140`) — P3 — fine for a single shared checkpoint; the field is documentation, and here the documentation is right.
12. **Shared policy means one paper row for one checkpoint** — VERIFIED — one variant `fashioniq` (`:445`), one row (`papers.yaml:1100`), `select_reference(encoder, noise=0, protocol=fashioniq_val_split)` → `exact condition match` (observed); no category-model routing code exists for encoder and none is needed — NONE — none. (Distinguish from CLVC-Net, whose `shared` claim is stale per `docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:25,58`; that finding does **not** apply to encoder.)
13. **Declared protocol matches source default** — VERIFIED — `:435`; `evaluate_model.py:15`; `datasets.py:64`; adapter passes it explicitly — NONE — none.
14. **Protocol derivable from the paper** — UNKNOWN (negative) — paper §4.1 names only "FashionIQ (Wu et al. 2021)" and "R@10, R@50, and their category-wise averages" (`aaai_encoder.txt:247,255-257`); no gallery/split statement — P3 — keep the source-derived protocol and say so in the row (the extracted bundle already does).
15. **Gallery construction matches `fashioniq_val_split`** — VERIFIED — `datasets.py:185-196` first-seen ordered source→target union; `test.py:67-68` masks each query's own reference out of the ranking — NONE — none.
16. **Reference-exclusion operation identical to workbench** — VARIANT — upstream masks the reference score to `-10e10` **before** `argsort` (`test.py:67-71`); local `src/evaluation/fashioniq_encoder.py:38-51` stable-sorts then removes the column — same cohort, not proven numerically identical — P2 — keep the distinction; never assert numeric equivalence.
17. **Category set {dress,shirt,toptee}** — VERIFIED — `evaluate_model.py:72-74`; `datasets.py:9-14,102`; protocol `papers.yaml:1134-1143` — NONE — none.
18. **Annotation/gallery files as declared** — VERIFIED — `datasets.py:159-161`; layout list `fashioniq_layout.py:25-34`; `papers.yaml:1111-1112` — NONE — none.
19. **Image layout `.jpg` under `resized_image/{category}`** — VERIFIED — `datasets.py:148` `+ ".jpg"`; `:68`; `models.yaml:433-434` — NONE — none.
20. **Correction dictionaries present with provenance** — UNKNOWN — required `datasets.py:163`; absent from `git ls-files`; no URL in README:154-186; not in the local root (`find data -name "correction_dict*"` → empty) — P1 — stage them as explicitly unverified inputs or obtain author provenance (C9c).
21. **Trailing-separator path contract honoured** — VERIFIED — `datasets.py:68-70`; `adapters/models.py:41` `f"{request.dataset_root}/"` — NONE — none.
22. **Caption normalization parity** — VERIFIED — `datasets.py:117-122` lowercase/punctuation/`correction_dict` substitution; local normalizer `src/datasets/fashioniq.py` per `docs/audits/PHASE2_SOURCES_A.md:55-56` — NONE — none.
23. **Caption composition order and ` and `** — VERIFIED — `datasets.py:124-126`; composed `cap[0] and cap[1]` — NONE — none.
24. **Tokenizer pinned** — VARIANT — `open_clip.get_tokenizer('ViT-B-32')` (`model_try2.py:137`) under `open-clip-torch==2.24.0` (`requirements.txt:155`); context length/truncation not asserted by source — P2 — do not claim tokenizer parity beyond the pin; the registry records no OpenCLIP note at all.
25. **`./open_clip_pytorch_model.bin` required at cwd** — VERIFIED — `evaluate_model.py:40`, `model_try2.py:135`; cwd = source root (`evaluate_models.py:185-186`, `:488`) — P1 — must be staged at `ENCODER/open_clip_pytorch_model.bin`.
26. **Backbone binary registered with acquisition + hash** — VARIANT — `grep -c encoder auxiliary_assets.yaml` → 0; no `preparation_id` (`preparation_contract_for_model(encoder)` → `None`); candidate URL exists only as audit prose (`registry/audits/cluster_a_encoder_split.yaml:218`) — P1 — add a real asset row with URL+size+digest (C7).
27. **Query/gallery transform asymmetry reproduced** — VERIFIED (and the *cause* of P1 item 8) — `datasets.py:177` `stage=0` vs `:195` `stage=1` — P1 — never transform both sides alike; never claim determinism.
28. **Evaluation determinism** — VARIANT — `preprocess_train` on the query path (`datasets.py:177`) plus no seeding anywhere in `evaluate_model.py` (`:13-33`, no `--seed`; only `train.py:58,229-236` seeds) — P1 — report any future number as stochastic (C8).
29. **OpenCLIP version pin recorded in the registry** — VARIANT — `requirements.txt:155`; `models.yaml:419` `packages: []`; README:46 explicitly warns versions move metrics — P2 — record the pin.
30. **`python: '3.9'` supported** — VARIANT — README:128 `conda create -n encoder_env python=3.9`; not a `requirements.txt` pin; `:409` labels it `VERIFIED_REQUIREMENTS` — P3 — relabel as README-derived.
31. **torch/torchvision pins correct** — VERIFIED — `requirements.txt:276,280` = 1.13.1 / 0.14.1 = `models.yaml:411-412` — NONE — none.
32. **CUDA declaration consistent with README** — VARIANT — `models.yaml:418` `12.6` vs README:132 `cu118`; comment `:413-417` attributes it to the family — P2 — reconcile (C10).
33. **GPU-mandatory nature captured** — VERIFIED — `evaluate_model.py:64` unconditional `.cuda()`; comment `:413-417`; CUDA-hard evaluator correctly cannot be recorded as CPU-capable — NONE — none.
34. **Evaluator importable** — VERIFIED-NEGATIVE — `evaluate_model.py:7 import datasets1`; absent from `git ls-files`, from the GitHub tree at the pin, and from the local filesystem search — P0 — obtain author-backed module; never add an unverified alias (C2).
35. **`datasets1` exists under another name** — VARIANT — `datasets.py:63,207,330,470` defines exactly the five names used at `evaluate_model.py:43-56` — P0 — [INFERENCE] rename defect; needs author confirmation (C2).
36. **`train.py` importable** — VERIFIED-NEGATIVE — `train.py:17 import model as model`; only `model_try2.py` exists — P2 — README:44 "All codes are transfered" is false for the pinned tree; affects retraining claims only.
37. **README's own eval command is accurate** — VARIANT — README:252-253 `python3 evaluation_model.py --model_dir …`; real file `evaluate_model.py`, real arg `--ckpt_path` (`evaluate_model.py:32`) — P3 — adapter is right (M4).
38. **Workbench adapter command matches source argparse** — VERIFIED — `adapters/models.py:41` vs `evaluate_model.py:13-33` — NONE — none.
39. **Adapter cwd correct for the relative weights** — VERIFIED — `evaluate_models.py:185-186,488`; `source_cwd` is source-only for csmcir — NONE — none.
40. **Preflight blocks execution** — VERIFIED — observed 10 blockers (`preflight.py:212-215` for the two encoder-specific ones) + 1 runtime blocker (`runtime.py:2145-2148`) — NONE — none.
41. **Preflight classifies the P0 source defect correctly** — MISMATCH — `classify(['ENCODER evaluator import missing: y'])` → `('BLOCKED_EXTERNAL_ASSET', …)`; and behind `BLOCKED_DATASET` today (`pc_readiness.py:52` before `:57`) — P1 — add a `BLOCKED_IMPLEMENTATION` rule above the generic matches (M1).
42. **Layout covers every input the evaluator opens** — VARIANT — `fashioniq_layout.py:25-34` omits `cap.{c}.train.json` read by a fresh constructor at `datasets.py:102-106` — P2 — use a train-inclusive check or pin the first-run caches (C9a).
43. **Fresh-run caches handled with an isolation policy** — VARIANT — `datasets.py:74-88` writes and then blindly trusts `fashion_iq_data.json` + 6 PKLs at the shared root; no policy anywhere for encoder (contrast `preparation_contracts.yaml:219,300` for SPRC/LIMN) — P2 — declare a cache root or require a clean first run (C9b).
44. **Preparation contract resolves to a real contract** — VARIANT — `preparation_contract_for_model(encoder)` → `None`; `manual_ilearn_resized` is only an allow-list enum value (`registry.py:73`); no encoder row in `preparation_contracts.yaml` — P2 — add an encoder contract or declare the layout fallback explicitly (C9).
45. **`auxiliary_assets.yaml` row for encoder** — VARIANT — zero encoder entries — P1 — register the OpenCLIP binary and the correction dictionaries.
46. **Paper row exists and is selected** — VERIFIED — `papers.yaml:1100-1164`; `select_reference('encoder', checkpoint_training_noise_pct=0, protocol_id='fashioniq_val_split')` → `exact condition match`, `reference_status: VERIFIED` (observed) — NONE — none.
47. **Paper row cites the model's own paper** — MISMATCH — `papers.yaml:1102-1103` `paper_id: hint_icassp26` / `paper_title: 'HINT: …'` vs `models.yaml:426` ENCODER paper — P1 — fix in the source bundle, then rebuild (C1).
48. **`reported_r10` value** — VERIFIED — Table 1 `ENCODER(Ours)` FashionIQ Avg R@10 = 56.13 (`aaai_encoder.txt:263`; pinned figure row) = `models.yaml:438` — NONE — none (STATIC_AUDIT).
49. **`reported_r50` value** — VERIFIED — same row, 77.59 = `models.yaml:439` — NONE — none (STATIC_AUDIT).
50. **`reported_mean` value and label** — VERIFIED value / VARIANT label — `(56.13+77.59)/2 = 66.86` = `models.yaml:440`; the paper has no mean column; the row labels it `WORKBENCH_DERIVED` (`papers.yaml:1126-1133`) but `models.yaml:440` carries no marker — P3 — label it in `models.yaml` too (M5).
51. **Category metrics match the paper** — VERIFIED — dress 51.51/76.95, shirt 54.86/74.93, toptee 62.01/80.88 (`papers.yaml:1134-1143`) vs `aaai_encoder.txt:263` and the pinned figure — NONE — none (STATIC_AUDIT).
52. **Aggregate reproduces from the categories as printed** — VERIFIED — `mean(51.51,54.86,62.01) = 56.127 → 56.13`; `mean(76.95,74.93,80.88) = 77.587 → 77.59` — NONE — none.
53. **`backbone` matches paper and code** — VERIFIED — paper §4.1 "(ViT-B/32 version)" (`aaai_encoder.txt:249`); `papers.yaml:1106`; `evaluate_model.py:40` `'ViT-B-32'` — NONE — none.
54. **`checkpoint_training_noise_pct` attested by the paper** — MISMATCH — asserted `0` (`models.yaml:450`, `papers.yaml:1108`); the paper states no training condition (`grep -n -i "noise" aaai_encoder.txt` → only method-perturbation prose and a citation title); the bundle's own evidence admits it is "the paper's training condition, not a verified file property" — P2 — set `null` (or record the assumption as assumption), and accept that the row then stops matching (C6).
55. **`checkpoint_mapping_status` honest** — MISMATCH — `models.yaml:455` `VERIFIED_METADATA` vs bundle `UNVERIFIED`; runtime treats non-VERIFIED as `checkpoint mapping unresolved` (`runtime.py:2147-2148`) and computed selection shows the checkpoint unobtainable — P1 — demote to UNVERIFIED until a file is hashed (C4).
56. **`strict_clean_checkpoint_available` honest** — MISMATCH — `models.yaml:442` `true` vs computed `false` (observed `select_checkpoints(...).strict_clean_checkpoint_available` → `False`) — P1 — derive it or set false (C5).
57. **Checkpoint direct URL resolvable** — UNKNOWN — `download_url: null` (`:447`); folder only (`:448`) — P1 — resolve outside a read-only audit.
58. **Checkpoint size/hash recorded** — NOT_RUN — `:452-453` both null — P1 — record size+SHA-256 after download.
59. **Checkpoint present locally** — NOT_RUN — `…/checkpoints/encoder/fashioniq.pt` `exists → False` (observed) — P1 — stage it.
60. **Checkpoint alone sufficient** — VERIFIED-NEGATIVE — `model_try2.py:135` and `evaluate_model.py:40` need the OpenCLIP binary before `load_state_dict` (`evaluate_model.py:63`) — P1 — document as a two-asset dependency.
61. **Native macro/mean emitted by the source** — VERIFIED-NEGATIVE — `test.py:75-81` per-category only; `evaluate_model.py:74-75` prints it — NONE — the workbench macro must never be attributed to the authors.
62. **Workbench aggregate defined and labelled** — VARIANT — `evaluate_models.py:69-118` defines `macro_r10/r50/mean`, but only for `synthetic_source` models (`:1386`); encoder is not synthetic (registry has zero `synthetic_source` keys) — P3 — if a macro is ever produced for encoder, label it as workbench-derived.
63. **Encoder metric parser exists** — VERIFIED-NEGATIVE — `extract_aggregate_metrics` handles only limn/csmcir (`metrics_extraction.py:328-342`); observed `AGGREGATE_PARSER_UNAVAILABLE` on ENCODER's own stdout shape — P0 — write the parser for `[('{cat}_r{k}', v), …]` (C3).
64. **Gate can judge an encoder run** — VERIFIED-NEGATIVE — observed `evaluate_gate(ref, None, …)` → `FAILED_OUTPUT_VALIDATION`, "the evaluator produced nothing to compare" (consequence of item 63) — P0 — same fix (C3).
65. **Gate compares correctly when metrics exist** — VERIFIED — observed `PASS_WITHIN_5_PERCENT` with 9 comparisons (r10 3.79%, r50 3.34%, mean 3.53%, plus six category metrics ≤4.88%) — NONE — none (this is the counterfactual, not an encoder result).
66. **Per-category gate comparison works for this row** — VERIFIED — the row publishes all six category metrics (`papers.yaml:1134-1143`) and the gate compares them when the run observes them (`paper_gate.py:345-363`) — NONE — none; note it can only ever fire once item 63 is fixed.
67. **`reproduction_status` honest** — VERIFIED — `models.yaml:441` `NOT_RUN`; no encoder report under `workbench/artifacts/results/**` (observed) — NONE — none.
68. **Local encoder-shaped evaluator is not mistaken for ENCODER** — VARIANT — `src/evaluation/fashioniq_encoder.py:120-136` calls `model.retrieve`; no `def retrieve` in `src/`; imported only by tests (`test_protocols.py:7,21`; `tests/unit/test_encoder_mode_isolation.py:8`) — P2 — mark dead/generic; never cite as reproduction. (Prior round names the wrong files; M3.)
69. **Selection-score parity** — VERIFIED — `train.py:186-189` sums `t[1][1]` (category R@10) over three categories; `src/evaluation/fashioniq_encoder.py:73-78` sums `{cat}_r10` — NONE — none.
70. **Hyperparameter declaration consistent between paper and code** — VARIANT — paper: P=4, E=3, τ=0.1 (`aaai_encoder.txt:251`); code comment: FashionIQ `{P,wc,N_P} = {3,2,2}` (`evaluate_model.py:35`) with argparse defaults `--P 3`, `--wc 2`, `--N_p 2` (`:22,29,30`); adapter passes none of them — P2 — record which side the run used; do not claim paper parity for tunables.
71. **Prior-round review internally consistent with the code** — VARIANT — most citations check out (`datasets.py:64,68-70,148,159,163,177,195`; `test.py:66-71,73-81`; `evaluate_model.py:7,40,60-64`, `adapters/models.py:41`); three drifted/wrong: train-read line offsets (M2), the `fashioniq_encoder.py` importer list (M3), and it labels `models.yaml:455` `VERIFIED_METADATA` without noting the runtime refuses it — P3 — refresh.
72. **`registry/audits` prose consistent with the code** — VARIANT — `cluster_a_encoder_split.yaml:187-228,331-333` correctly describes the `datasets1` and OpenCLIP blockers, but over-claims "across all git branches/commits" in a 1-commit clone (and its ``test.py:45,61``/``data_utils.py`` line cites belong to the HINT section, not ENCODER) — P3 — tighten wording (M6).
73. **`README`-derived audit doc consistent** — VARIANT — `docs/AUTOMATED_DATA_PREPARATION.md:55` and `docs/FINAL_PC_READINESS.md:153` / `docs/PC_REPRODUCTION_HANDOFF.md:35` all state the `datasets1` + OpenCLIP + unresolved-URL blockers correctly; `cluster_b2_encoder_family.yaml:143-144` repeats the import failure — P3 — none required (the docs and the code agree here).
74. **End-to-end acceptance: source → checkpoint → metric** — NOT_RUN — import fails (`evaluate_model.py:7`), backbone binary absent, checkpoint absent with mapping refused by `runtime.py:2147`, parser absent, GPU-only — P0 — unblock only after all prerequisites carry author-backed provenance.

---

**FINAL VERDICT: BLOCKED — the pinned ENCODER evaluator is unimportable (`datasets1`, absent from tree, upstream pin, and filesystem) and unrunnable (backbone binary absent and unregistered, checkpoint absent with no URL/size/hash and a runtime-refused mapping, CUDA-mandatory, unseeded), and even a completed run could not be judged (no encoder parser → gate `FAILED_OUTPUT_VALIDATION`), while the only verified numbers — the paper's Table 1 FashionIQ Avg 56.13/77.59, re-derived this round from the real 9-page AAAI-25 galley and the pinned results figure — are a transcription, not a reproduction; the registry's own governance fields for this model contradict the runtime (`checkpoint_mapping_status` VERIFIED vs `runtime.py:2147`, `strict_clean_checkpoint_available: true` vs computed `false`, an asserted 0% training noise the paper never states) and its benchmark row cites a different paper.
