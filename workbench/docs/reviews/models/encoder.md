# encoder

MODEL ID — `encoder` (method_name ENCODER), `workbench/registry/models.yaml:382-431`.

PAPER TITLE — "ENCODER: Entity Mining and Modification Relation Binding for Composed Image Retrieval", AAAI-25, `models.yaml:402-403` (`https://ojs.aaai.org/index.php/AAAI/article/view/32541`). Direct fetch of the OJS landing page and PDF failed (`read` -> "socket connection was closed unexpectedly"); no paper text was read. Metrics were verified instead from the pinned repo's own results figure. STATIC_AUDIT.

PINNED SOURCE COMMIT — registry declares `29a2a31d6a56f677bf450c3be7cdaef423fb7018` (`models.yaml:406`). `git -C workbench/third_party/ENCODER rev-parse HEAD` -> `29a2a31d6a56f677bf450c3be7cdaef423fb7018`. MATCH. STATIC_AUDIT.
Caveat: `git rev-list --all --count` = **1**; `git log --oneline --all` = single commit `29a2a31 Update README.md`. The checkout is a squashed/export copy, so no history-level claim can be tested in it.

CHECKPOINT IDENTITY — one variant `fashioniq`, filename `fashioniq.pt`, `download_url: null`, `source_url: https://drive.google.com/drive/folders/1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3` (a **folder**), `expected_sha256: null`, `expected_size_bytes: null`, `status: URL_UNRESOLVED`, `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:420-431`). README:249 links the same Drive folder. No `fashioniq.pt` exists locally (`find / -maxdepth 6 -name fashioniq.pt` -> only 1-byte test fixtures under `/tmp/...`). Identity is UNVERIFIED: no file URL, no size, no hash, no code that proves the file is the FashionIQ `val-split` benchmark model.

EVALUATION PROTOCOL — registry `native_protocol: fashioniq_val_split`, `literature_split_label: val`, `supported_protocols: [fashioniq_val_split]` (`models.yaml:411-413`). Registry protocol table (`models.yaml:357-366`) and workbench table `src/evaluation/fashioniq.py:167-168` agree: `pair_union` gallery, `reference_excluded=True`. Upstream default `--fashioniq_split val-split` (`evaluate_model.py:15`) and `FashionIQ.__init__(..., split='val-split')` (`datasets.py:64`). The adapter also passes `val-split` explicitly (`workbench/backend/adapters/models.py:41`). Protocol label matches the source's default branch. STATIC_AUDIT.

DATASET CONTRACT — root must end in `/`: `self.path + 'resized_image' | 'image_splits' | 'captions'` (`datasets.py:68-70`); the adapter appends the slash (`adapters/models.py:41`). Per category = `{dress,shirt,toptee}` (`datasets.py:102`, `evaluate_model.py:73`). Per category the evaluator opens `image_splits/split.{c}.val.json` (`datasets.py:159`), `captions/cap.{c}.val.json` (`:161`), `captions/correction_dict_{c}.json` (`:163`), and images `resized_image/{c}/{id}.jpg` (`:148`). A **fresh** constructor additionally opens `captions/cap.{c}.train.json` and `correction_dict_{c}.json` (`:103-106`) and writes root caches `fashion_iq_data.json` + `test_queries_{c}.pkl`/`test_targets_{c}.pkl` (`:77-88`), with no freshness check (`:74`). Registry layout `fashioniq_ilearn_resized` (`models.yaml:409`) is checked by `fashioniq_layout.py:25-34`: val captions, splits, correction dicts, resized dirs — **not** train captions, not the cache files. `preparation_contract: manual_ilearn_resized` (`models.yaml:401`) has **no** `preparation_id`, so `prepare_model.py` gets no contract (`registry.py:187-190` returns None). Correction dictionaries are absent from the pinned tree (not in `git ls-files`) and have no source URL -> UNVERIFIED provenance.

PREPROCESSING CONTRACT — `evaluate_model.py:40` builds `open_clip.create_model_and_transforms('ViT-B-32', pretrained=os.path.join('.', 'open_clip_pytorch_model.bin'))`. Asymmetry is real and source-exact: query/reference uses `stage=0` (`preprocess_train`, random-resized-crop) at `datasets.py:177`, gallery uses `stage=1` (`preprocess_val`) at `datasets.py:195`. Text: lowercase -> ASCII `string.punctuation`->space -> tokenize -> dictionary substitution -> join (`datasets.py:117-121`), composed as `cap0 and cap1` (`:125`). OpenCLIP version is pinned in `requirements.txt:155` (`open-clip-torch==2.24.0`).

AUGMENTATION CONTRACT — evaluation supplies `[preprocess_train, preprocess_val]` and the **reference image is transformed with the training transform**, so evaluation is stochastic/rand-crop dependent. `evaluate_model.py` never seeds RNG (no `manual_seed`; only `train.py:229-234` seeds with `--seed 42`, `train.py:58`). Retrieval ranks are therefore not reproducible run-to-run from the evaluator as packaged -> P1.

GALLERY CONTRACT — `val-split` branch (`datasets.py:185-196`) builds `test_targets_id` as first-seen union of each annotation's `source_img_id` then `target_img_id`, then the gallery as `[split_index[i] for i in test_targets_id]`. Workbench `build_pair_union_gallery` (`src/datasets/fashioniq.py:186-209`) emits exactly reference-then-target first-seen per annotation. **EQUIVALENT** (same ordered set; upstream maps to positions in `split.{c}.val.json`, local keeps the ID strings).

METRIC CONTRACT — upstream `test.py:73-81` returns per-category `('{cat}_r{1,10,50}', pct)`; ranking is `np.argsort(-sims[i,:])[:50]` (`:71`) after masking the reference score, `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (`:66-68`). Upstream emits **no** macro R@10/R@50 and **no** mean. Workbench aggregate is `macro_average_fashioniq` (`src/evaluation/fashioniq.py:223-240`: macro R@10, macro R@50, `mean_recall=(r10+r50)/2`) — a workbench aggregate, correctly not attributed to source. Registry `reported_mean: 66.86` (`models.yaml:416`) is exactly `(56.13+77.59)/2` and is **not** a paper column.

SOURCE-TO-WORKBENCH PARITY —
- gallery/reference cohort: EQUIVALENT, but operation differs — upstream masks before sort; local `fashioniq_encoder_recall_at_k` (`src/evaluation/fashioniq_encoder.py:38-51`) stable-sorts then drops the reference column. Same cohort, not proven numerically identical.
- model path: `src/evaluation/fashioniq_encoder.py:120-136` calls `model.retrieve(...)` with teacher/native feature caches; upstream calls `model.extract_retrieval_compose` / `extract_retrieval_target` (`test.py:39,49`). No `def retrieve` exists anywhere in `src/` (`grep -rn "def retrieve" src/` -> empty) and `src/encoders/` is empty. The local encoder-shaped evaluator is dead code with no executing model and **must not** be presented as ENCODER reproduction. It is also not wired to any protocol dispatcher (`build_fashioniq_gallery` / `evaluate_fashioniq` are never imported outside tests/`src/evaluation/fashioniq.py`).
- selection score: upstream `train.py:184-189` sums `t[1][1]` (R@10) across three categories; local `fashioniq_encoder_selection_score` (`src/evaluation/fashioniq_encoder.py:73-78`) sums `{cat}_r10`. EQUIVALENT.

RUNTIME STATUS — **BLOCKED**. Observed read-only preflight (`python3 /tmp/pb_ro.py`, 10 blockers, print-only, nothing executed):
`ENCODER evaluator import missing: .../ENCODER/datasets1.py`; `ENCODER asset missing: .../ENCODER/open_clip_pytorch_model.bin`; 7 missing FashionIQ `fashioniq_ilearn_resized` paths (resized_image + correction dicts) + 1 image-extension problem. Sources: `workbench/backend/preflight.py:91,149-154`. No GPU run attempted. `reproduction_status: NOT_RUN` (`models.yaml:417`).

PAPER-METRIC STATUS — `assets/fashion-results.png` (read as image): ENCODER row = Dresses 51.51/76.95, Shirts 54.86/74.93, Tops&Tees 62.01/80.88, **Avg 56.13 / 77.59**. Registry `reported_r10: 56.13`, `reported_r50: 77.59` MATCH the FashionIQ Avg row. Macro check: (51.51+54.86+62.01)/3 = 56.13; (76.95+74.93+80.88)/3 = 77.59. STATIC_AUDIT (figure read, not a run). No reproduction.

## CRITICAL FINDINGS
1. **P0 — `datasets1` does not exist.** `evaluate_model.py:7` `import datasets1`; `git ls-files` (`datasets.py`, no `datasets1.py`); `find workbench/third_party/ENCODER -name "datasets1*"` -> empty; filesystem-wide `find / -xdev -name "datasets1*"` -> empty; `git grep -n datasets1` matches only `evaluate_model.py`. The exact four classes imported through it (`FashionIQ`, `Shoes`, `CIRR`, `Fashion200k`, `evaluate_model.py:43-56`) all exist in the shipped `datasets.py:63,207,330,470`. [INFERENCE] same module, stale import name — but that is unproven, and no alias may be added without author evidence. Every execution path dies at import.
2. **P0 — `train.py` is also unimportable**: `train.py:17` `import model as model`; only `model_try2.py` exists. The "all codes are transfered to the Github Repo" claim (README:44) is false for the pinned tree.
3. **P1 — checkpoint identity unresolved** (`models.yaml:430` `URL_UNRESOLVED`): folder URL only, no hash/size, no local artifact. `expected_size_bytes`/`expected_sha256` null => `runtime_blockers` can never hash-check it (`runtime.py:2107-2110`).
4. **P1 — `open_clip_pytorch_model.bin` required at cwd and absent.** `evaluate_model.py:40` and `model_try2.py:135` both `pretrained=os.path.join('.', 'open_clip_pytorch_model.bin')`. Adapter runs with cwd = source dir (`evaluate_models.py:124-125`, `:392`), so the file must sit at `ENCODER/open_clip_pytorch_model.bin`. It is not in `git ls-files`, not in `auxiliary_assets.yaml` (`grep -c encoder auxiliary_assets.yaml` = 0) and not checked as a pip-fetchable asset. The checkpoint alone is **insufficient**, confirmed.
5. **P1 — preprocessing asymmetry + no seed.** Query uses `preprocess_train` (`datasets.py:177`) and RNG is unseeded in the evaluator => ranks non-reproducible. This is a source property, not a workbench bug, but it caps what any reproduction can claim.
6. **P1 — registry `cuda: '12.6'` contradicts README:132** (`--index-url .../cu118`, i.e. CUDA 11.8) while the comment at `models.yaml:389-393` calls it "the iLearn family's pinned one". The GPU requirement itself is real: `evaluate_model.py:64` `Encoder_model = Encoder_model.cuda()` unconditional.

## MINOR FINDINGS
- `models.yaml:385` confidence `VERIFIED_REQUIREMENTS` for `python: '3.9'`; `requirements.txt` has no Python pin — the value comes from README:128 (`conda create -n encoder_env python=3.9`). Label should be README-derived.
- `models.yaml:395` `packages: []` misdeclares the environment; `requirements.txt` pins `open-clip-torch==2.24.0` (:155), `torch==1.13.1` (:276), `torchvision==0.14.1` (:280), `transformers==4.30.2` (:284), `einops==0.4.1` (:56), `numpy==1.21.6` (:149), `tqdm==4.66.1` (:282). The planner recovers most from the manifest, but `model_constraints()` also emits a bogus pip package `{"datasets1": "UNKNOWN"}` — a missing source module is being treated as an installable distribution (observed in the printed `packages` dict). P2.
- README's documented command is wrong twice: `python3 evaluation_model.py` (file is `evaluate_model.py`) with `--model_dir` (README:252-253) — `evaluate_model.py:13-32` has **no** `--model_dir`, only `--ckpt_path`.
- The registry's OpenCLIP note is accurate but lives only as prose: README:46 says "different versions of open_clip can impact model performance ... clarified the environment dependencies (requirements.txt)"; pin is `requirements.txt:155`. The registry itself records no OpenCLIP note (`grep -i open.clip models.yaml` -> 4 hits, all other models).
- `workbench/registry/assets/cluster_b2_encoder_family.yaml:166,265` claim absence "across all git branches/commits"; the local clone has 1 commit, so that claim is untestable here — UNKNOWN, not confirmed.
- `reported_mean: 66.86` is a workbench arithmetic mean, not a paper value (figure has no mean column).

## UNRESOLVED EVIDENCE
- Google Drive folder contents / `fashioniq.pt` real name+hash+size: unreachable, no network fetch attempted. UNKNOWN.
- Whether `datasets1` is an author file published outside git, or a rename defect: UNKNOWN (single-commit clone; no remote history available locally; no network).
- Provenance of the three required `captions/correction_dict_{c}.json` blobs: UNKNOWN.
- Provenance of `open_clip_pytorch_model.bin` (which CLIP checkpoint, which open_clip build): UNKNOWN. Note `model_try2.py:134` carries a commented alternate path `/root/CLIP-ViT-B-32-laion2B-s34B-b79K`, suggesting a **LAION** ViT-B-32, not OpenAI — the registry records no such note for encoder.
- Any runtime metric: UNKNOWN (no GPU; no checkpoint; no dataset).

## FINAL VERDICT
BLOCKED — the pinned evaluator cannot be imported (`import datasets1`, absent from the pinned tree and from the whole filesystem) and cannot be executed (`open_clip_pytorch_model.bin` absent, checkpoint URL unresolved, GPU-only), so no ENCODER metric can be reproduced; the FashionIQ Avg row (56.13/77.59) is verified only as the paper figure's own number (STATIC_AUDIT).

# EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation -> checkpoint -> output -> metric comparison`

1. **paper/README/config ->**: README:249 `Our checkpoints are released at [Google Drive](...1QuBQVIybLwn-qFSORhpBsnXy6ebhtNr3)`; README:46 OpenCLIP-version warning; README:128 python 3.9; README:132 cu118 torch index. Registry mirrors: `models.yaml:403,406,424,388,394`. Evidence present, no contradiction except cuda (`models.yaml:394` 12.6 vs README:132 cu118).
2. **-> source dataset**: `evaluate_model.py:17 --fashioniq_path` (default `.`), consumed at `datasets.py:68-70` by string concat (trailing-slash needed), read at `datasets.py:159-165` (`split.*.val.json`, `cap.*.val.json`, `correction_dict_*.json`) and `:148` (`resized_image/{c}/{id}.jpg`). Adapter supplies `f"{request.dataset_root}/"` (`adapters/models.py:41`). Arrow OK; dataset blobs absent locally (preflight output above).
3. **-> preprocessing**: `evaluate_model.py:40` -> `open_clip.create_model_and_transforms('ViT-B-32', pretrained='./open_clip_pytorch_model.bin')`; transform list `[preprocess_train, preprocess_val]` (`:43`); applied `stage=0` query / `stage=1` gallery (`datasets.py:177,195`). Requires the absent `.bin`. Arrow BROKEN.
4. **-> official evaluator**: `evaluate_model.py:60-64` `torch.load` -> `load_state_dict(strict=False)` -> `.cuda()`; `:72-75` per-category `test.test(...)`; `test.py:39,49` feature extraction; `test.py:66-71` reference mask + `argsort`; `test.py:73-81` R@1/10/50. Entry never reached: `evaluate_model.py:7 import datasets1` raises `ModuleNotFoundError` first (asserted from source; not run — running is unnecessary and the module is provably absent from the tree and the filesystem).
5. **-> workbench invocation**: `EncoderAdapter.command` (`adapters/models.py:36-41`) -> `python <src>/evaluate_model.py --dataset fashioniq --fashioniq_split val-split --fashioniq_path <root>/ --ckpt_path <ckpt>`; cwd = source dir via `source_cwd` (`scripts/evaluate_models.py:124-125`) used by `Popen(plan.command, cwd=plan.cwd)` (`:392`). Command shape matches the source's argparse (`evaluate_model.py:13-32`). Arrow OK; refused before launch by `preflight_blockers` (`preflight.py:149-154`).
6. **-> checkpoint**: `models.yaml:420-431` -> `download_url: null`, folder URL, no hash/size; `runtime_blockers` would append `checkpoint missing` (`runtime.py:2105-2106`). Arrow BROKEN.
7. **-> output**: upstream prints `print(t)` per category (`evaluate_model.py:75`); per-category `{cat}_r1/r10/r50` only (`test.py:81`). No macro, no mean. NOT REACHED.
8. **-> metric comparison**: registry 56.13/77.59 vs figure Avg 56.13/77.59 (MATCH, STATIC_AUDIT); `reported_mean 66.86` = arithmetic mean of the two, not a paper column. No reproduced number exists.

# CHECKLIST

1. Registry entry exists and loads — VERIFIED — `models.yaml:382-431`; observed `load_registry()` returns the model dict with all fields — NONE — none.
2. `upstream_commit_sha` equals checkout HEAD — VERIFIED — `git -C .../ENCODER rev-parse HEAD` -> `29a2a31d6a56f677bf450c3be7cdaef423fb7018`; `models.yaml:406` — NONE — none.
3. Checkout is clean — VARIANT — `git status --porcelain` -> only `?? __pycache__/` (bytecode, ignored by `runtime.py:1825-1831`) — P3 — remove `__pycache__` before a run.
4. `datasets1` absent from pinned source — VERIFIED — `evaluate_model.py:7`; `git ls-files` lists no `datasets1.py`; `git grep datasets1` matches only `evaluate_model.py`; `find / -xdev -name "datasets1*"` -> empty — P0 — obtain author-backed module or fix import; do not alias.
5. `datasets1` present under another name — VARIANT — `datasets.py:63,207,330,470` defines exactly `FashionIQ/Shoes/CIRR/Fashion200k`, the four symbols used at `evaluate_model.py:43-56` — P0 — treat as rename [INFERENCE]; require author confirmation before any alias.
6. `open_clip_pytorch_model.bin` required — VERIFIED — `evaluate_model.py:40`, `model_try2.py:135`; `preflight.py:153-154` — P1 — stage with verified provenance; registry has no asset row.
7. Checkpoint alone sufficient — VERIFIED (insufficient) — same two lines; class construction needs backbone weights before `load_state_dict` — P1 — document as two-asset dependency.
8. Checkpoint direct URL resolvable — UNKNOWN — `models.yaml:423` `download_url: null`; only a folder URL at `:424` — P1 — resolve file URL outside read-only audit.
9. Checkpoint hash/size recorded — NOT_RUN — `models.yaml:428-429` both null — P1 — record size+SHA-256 after download.
10. Checkpoint present locally — NOT_RUN — `find / -maxdepth 6 -name fashioniq.pt` -> only 1-byte `/tmp` fixtures — P1 — stage.
11. Reported R@10=56.13 — VERIFIED — `assets/fashion-results.png` ENCODER Avg row 56.13; `models.yaml:414` — NONE — none (STATIC_AUDIT).
12. Reported R@50=77.59 — VERIFIED — same figure, Avg 77.59; `models.yaml:415` — NONE — none (STATIC_AUDIT).
13. Reported mean=66.86 — VARIANT — `(56.13+77.59)/2` = 66.86 (computed); figure has no mean column — P3 — label as workbench-derived, not paper.
14. Declared protocol matches source default — VERIFIED — `models.yaml:411`; `evaluate_model.py:15`; `datasets.py:64` — NONE — none.
15. Gallery construction matches `fashioniq_val_split` — VERIFIED — `datasets.py:185-196` vs `src/datasets/fashioniq.py:186-209` — NONE — none.
16. Reference exclusion semantics match cohort — VARIANT — upstream mask-before-sort `test.py:66-68`; local post-sort column removal `src/evaluation/fashioniq_encoder.py:38-51` — P2 — keep distinction; do not assert numeric equivalence.
17. Gallery ordering preserved as first-seen — VERIFIED — `datasets.py:186-190`; `src/datasets/fashioniq.py:198-207` — NONE — none.
18. Category set {dress,shirt,toptee} — VERIFIED — `datasets.py:102`, `evaluate_model.py:73`; `src/evaluation/fashioniq_encoder.py:16` — NONE — none.
19. Annotation source files as declared — VERIFIED — `datasets.py:159-161`; layout `fashioniq_layout.py:31,33` — NONE — none.
20. Correction dictionaries available with provenance — UNKNOWN — required `datasets.py:163`; absent from `git ls-files`; no URL — P1 — treat as OPERATOR_PROVIDED unverified.
21. Resized-image layout match — VERIFIED — `datasets.py:148` `resized_image/{c}/{id}.jpg`; `fashioniq_layout.py:29-30` — NONE — none.
22. Image extension `.jpg` — VERIFIED — `datasets.py:148` `+ ".jpg"`; `models.yaml:410` — NONE — none.
23. Trailing-slash path contract — VERIFIED — `datasets.py:68-70`; adapter `f"{request.dataset_root}/"` `adapters/models.py:41` — NONE — none.
24. Caption normalization parity — VERIFIED — `datasets.py:117-121` vs `src/datasets/fashioniq.py:93-117` — NONE — none.
25. Caption composition order/` and ` — VERIFIED — `datasets.py:125`; `src/datasets/fashioniq.py:158-161` — NONE — none.
26. Tokenizer/context behaviour pinned — VARIANT — tokenizer from `model_try2.py:137` under `open-clip-torch==2.24.0` (`requirements.txt:155`); exact context/truncation not asserted by source — P2 — do not claim tokenizer parity beyond the pin.
27. Query/gallery transform asymmetry reproduced — VERIFIED — `datasets.py:177` (`stage=0`) vs `:195` (`stage=1`) — P1 — never transform both sides alike.
28. OpenCLIP version pinned — VERIFIED — `requirements.txt:155`; registry `packages: []` (`models.yaml:395`) — P2 — record the pin in the registry.
29. README states OpenCLIP versions affect metrics — VERIFIED — `README.md:46`; pins "further clarified ... (requirements.txt)" — NONE — none.
30. `python: '3.9'` supported — VARIANT — README:128; not a requirements pin; confidence label `VERIFIED_REQUIREMENTS` (`models.yaml:385`) inaccurate — P3 — relabel evidence source.
31. torch/torchvision pins correct — VERIFIED — `requirements.txt:276,280` = 1.13.1/0.14.1 = `models.yaml:387-388` — NONE — none.
32. CUDA declaration consistent with README — VARIANT — `models.yaml:394` `12.6` vs README:132 `cu118` — P2 — reconcile; 12.6 is unverifiable for torch 1.13.1.
33. GPU-mandatory nature captured — VERIFIED — `evaluate_model.py:64` unconditional `.cuda()`; comment `models.yaml:389-390` — P1 — keep GPU gate; CPU-only host cannot run.
34. Trainer importable — VARIANT — `train.py:17 import model as model`, `model.py` absent (only `model_try2.py`) — P2 — README:44 "All codes are transfered" overstated; affects retraining claims only.
35. Evaluator CLI/README consistency — VARIANT — README:252-253 `evaluation_model.py --model_dir`; real file `evaluate_model.py`, arg `--ckpt_path` (`:13-32`) — P2 — adapter is right; README wrong.
36. Workbench adapter command — VERIFIED — `adapters/models.py:36-41` matches source argparse — NONE — none.
37. Workbench cwd correct for `.bin` and `clip_path='.'` — VERIFIED — `scripts/evaluate_models.py:124-125,392` — P1 — `.bin` must be at source root; absent.
38. Preflight blocks execution — VERIFIED — observed 10 blockers from `preflight_blockers` (`preflight.py:91,149-154`) — NONE — none.
39. Cache artifacts handled — VARIANT — `datasets.py:77-88` writes/trusts `fashion_iq_data.json` + PKLs; layout does not check them; `prepare_model.py` fallback (no `preparation_id`) — P2 — declare as non-provenance-validated caches.
40. Train-caption requirement enforced — VARIANT — needed by fresh ctor `datasets.py:103-105`; `fashioniq_ilearn_resized` omits train (`fashioniq_layout.py:25-34`) — P2 — use a train-inclusive check or pin the cache.
41. `auxiliary_assets.yaml` row for encoder — VARIANT — zero encoder entries (`grep -c encoder` = 0) — P1 — register the OpenCLIP binary and correction dicts.
42. `registry/assets/cluster_b2_encoder_family.yaml` row — VERIFIED (present, partly untestable) — `:139-166,261-265`; the "all branches/commits" wording (`:166`) is untestable in a 1-commit clone (`git rev-list --all --count` = 1) — P3 — soften to "pinned tree".
43. Workbench macro defined and labelled — VERIFIED — `src/evaluation/fashioniq.py:223-240,321` — NONE — never present as source metric.
44. Native macro/mean emitted by source — VERIFIED (absent) — `test.py:73-83` per-category only — NONE — none.
45. Selection-score parity — VERIFIED — `train.py:184-189` vs `src/evaluation/fashioniq_encoder.py:73-78` — NONE — none.
46. Local `fashioniq_encoder.py` not mistaken for ENCODER — VARIANT — calls `model.retrieve` (`:120-136`); no `def retrieve` in `src/`; not wired to any dispatcher — P2 — mark dead/generic; never cite as reproduction.
47. Tests exist for the local helper — VERIFIED (assertion only) — `tests/unit/test_fashioniq_encoder.py:17-79`, `workbench/tests/test_protocols.py:21` — NONE — these pin the local helper, not upstream parity.
48. Patches needed/registered — UNKNOWN — `workbench/patches/` holds only csmcir files — P2 — decide whether a datasets-alias patch is ever acceptable (currently correctly absent).
49. Runtime status — VERIFIED (BLOCKED) — preflight output; `models.yaml:417` `NOT_RUN` — NONE — none.
50. `reproduction_status` honest — VERIFIED — `NOT_RUN` (`models.yaml:417`) with no result artifacts — NONE — none.
51. `strict_clean_checkpoint_available: true` — UNKNOWN — `models.yaml:418`; no file, no hash, no mapping proof beyond the README folder link — P1 — set false/unknown until the file is hashed.
52. Workbench docs internally consistent with source — VARIANT — `UPSTREAM_AUDIT.md:69-71`, `UPSTREAM_DATA_PREPARATION_AUDIT.md:104-110`, `PROTOCOL_AUDIT.md:38,44`, `PHASE2_SOURCES_A.md:49-58` all match code; `cluster_b2...:166` over-claims branch history — P3 — fix wording.
53. Paper URL resolves to the paper — UNKNOWN — `models.yaml:403`; both `read` fetches of the OJS page/PDF failed (socket closed) — P3 — none required for the verdict.
54. Determinism of evaluation — VARIANT — `preprocess_train` on query (`datasets.py:177`) + no seeding in `evaluate_model.py` — P1 — report any future number as stochastic.
55. End-to-end acceptance: source -> checkpoint -> metric — NOT_RUN — import fails (`evaluate_model.py:7`), binary absent, checkpoint unresolved, GPU-only — P0 — unblock only after all four prerequisites are author-provenanced.

FINAL VERDICT: BLOCKED — the pinned ENCODER evaluator is unimportable (`datasets1` absent from the tree and the filesystem) and unrunnable (`open_clip_pytorch_model.bin` absent, checkpoint URL/hash unresolved, CUDA-mandatory), so nothing was reproduced; the only verified numbers (56.13/77.59) are the paper figure's own FashionIQ Avg row, read statically.

---

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left
as written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 1 committed row(s):

  - `encoder_fiq_clean_val_split` — `VERIFIED`, 0% training noise, protocol `fashioniq_val_split`, values `{'r10': 56.13, 'r50': 77.59, 'mean': 66.86}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'WORKBENCH_DERIVED'}`

- **Readiness now:** `BLOCKED_SOURCE` — upstream source unavailable

- **Reference per registered checkpoint condition:**

  - `fashioniq` (noise 0): `encoder_fiq_clean_val_split` `VERIFIED` — exact condition match

- **Gate:** the run's verdict comes from `workbench/backend/paper_gate.py` against the
  committed row above, reported in the run JSON under `benchmark_gate`. A row that states no
  condition is refused (`BLOCKED_REFERENCE_INCOMPLETE`); a metric the paper never printed is
  compared only if the registry derives it, and then it is labelled `WORKBENCH_DERIVED`.
