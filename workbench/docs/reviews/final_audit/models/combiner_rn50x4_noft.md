# combiner_rn50x4_noft

Independent adversarial audit of registered model `combiner_rn50x4_noft` in the CIR Failure Analysis
Workbench. READ-ONLY: nothing under `/home/heheboiz/data/cir` was written or modified; no GPU job, no
install, no checkpoint download, no evaluator execution. Scratch is under
`/tmp/audit-final/combiner_rn50x4_noft/` (paper HTML, paper text, Drive listings, table dumps). Working
tree audited at `eb36aa5ab0740e7dca7dd9c78515c3c49514b4e4` with modified/untracked files present
(observed `git status --short`: `M workbench/backend/adapters/models.py`, `M workbench/registry/models.yaml`,
`M workbench/backend/paper_gate.py`, `M workbench/backend/metrics_extraction.py`, `? workbench/docs/reviews/final_audit/`, …).
Every claim below is **STATIC_AUDIT** or **OBSERVED** (a command I ran). This document contains **zero
RUNTIME_VERIFIED** claims — no inference was run.

## MODEL ID

- Registry block: `workbench/registry/models.yaml:655-710` (`- model_id: combiner_rn50x4_noft`).
- `method_name: Combiner RN50x4 noft` (`:667`), `source_dir: CLIP4Cir` (`:676`), `source_available: true` (`:675`),
  `upstream_default_branch: master` (`:673`), `research_generation: legacy` (`:668`), `publication_year: 2023` (`:669`).
- `grep -n "model_id: combiner_rn50x4_noft" workbench/registry/models.yaml` → exactly one hit (`:655`); `ADAPTERS`
  keys it once (`workbench/backend/adapters/models.py:140`). **Unique.**
- Adapter: `CombinerNoftAdapter` (`adapters/models.py:86-94`), `script = "src/validate.py"` (`:88`), real
  `command()` override (`:90-94`).
- Sibling rows: `clip4cir_rn50x4_fullft` at `models.yaml:711-774` — distinct `model_id`, distinct adapter
  (`models.py:97-111`), distinct `reproduction_status` (`BLOCKED_CHECKPOINT_BUNDLE`, `:747` vs
  `BLOCKED_CHECKPOINT_MAPPING`, `:686`), distinct checkpoint record (`:758-774` vs `:698-710`).
- Fields read back through the real loader (OBSERVED, `workbench.backend.registry.model_by_id`):
  `{'reported_r10': None, 'reported_r50': None, 'reported_mean': None, 'reproduction_status':
  'BLOCKED_CHECKPOINT_MAPPING', 'command_status': 'COMMAND_AUDITED', 'replay_status':
  'REPLAY_CODE_IMPLEMENTED', 'strict_clean_checkpoint_available': None, 'native_protocol':
  'fashioniq_original_split', 'literature_split_label': 'original', 'fashioniq_layout':
  'fashioniq_standard', 'preparation_contract': 'manual_standard', 'preparation_id':
  'clip4cir_fashioniq', 'canonical_dataset_link': 'fashionIQ_dataset', 'fashioniq_image_extensions': ['.jpg']}`.

## PAPER TITLE

`Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features`
(`models.yaml:670`); `paper_url: https://arxiv.org/abs/2308.11485` (`:671`); `upstream_repo_url:
https://github.com/ABaldrati/CLIP4Cir` (`:672`).

- OBSERVED (network, `read https://arxiv.org/abs/2308.11485`): title, authors Alberto Baldrati / Marco
  Bertini / Tiberio Uricchio / Alberto del Bimbo, published 2023-08-22, cs.CV. Matches the pinned
  `workbench/third_party/CLIP4Cir/README.md:1-7` (`# CLIP4Cir (ACM TOMM 2023)`).
- Registry `paper_title` and `paper_url` are accurate. **This paper is a different, later paper from the
  precursor that the noft configuration actually comes from** (see PAPER-METRIC STATUS) — the registry
  cites only the TOMM one, and the registry's own benchmark row says so.

## PINNED SOURCE COMMIT

`upstream_commit_sha: dfed9f748a8a4d05abf613164e06d59b22dbdf13` (`models.yaml:674`).
OBSERVED:

```
$ git -C workbench/third_party/CLIP4Cir rev-parse HEAD
dfed9f748a8a4d05abf613164e06d59b22dbdf13
$ git -C workbench/third_party/CLIP4Cir remote -v
origin  git@github.com:ABaldrati/CLIP4Cir (fetch) [blob:none]
origin  git@github.com:ABaldrati/CLIP4Cir (push)
$ git -C workbench/third_party/CLIP4Cir status --short
(empty)
```

Executed through the workbench's own predicate (OBSERVED):

```
source_clean_and_pinned -> (True, 'dfed9f748a8a4d05abf613164e06d59b22dbdf13', None)
```

which is `runtime.py:2055-2079` (dirty → refuse; missing pin → refuse; mismatch → refuse).
`src/validate.py` present, 360 lines. **Exact match, clean tree, enforced by the runtime.**

## CHECKPOINT IDENTITY

Registry (`models.yaml:698-710`), verbatim:

```
checkpoint_id: rn50x4_noft_unresolved
filename: rn50x4-noft-unresolved.pt
artifact_type: file
download_url: null
source_url: https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing
source_type: author_google_drive
checkpoint_training_noise_pct: null
evaluation_noise_pct: 0
expected_size_bytes: null
expected_sha256: null
status: CHECKPOINT_MAPPING_UNVERIFIED
checkpoint_mapping_status: UNVERIFIED
```

- OBSERVED: `workbench/artifacts/checkpoints/` contains only `.gitkeep`; the directory
  `workbench/artifacts/checkpoints/combiner_rn50x4_noft/` does not exist. `checkpoint_path(...)` resolves to
  `workbench/artifacts/checkpoints/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt` (`registry.py:631-634`).
- OBSERVED `checkpoint_availability` (`registry.py:699`): `automatic_download_available: False`,
  `checkpoint_downloaded: False`, `official_sha256_known: False`, `mapping_verified: False`, `runnable: False`.

**The registry names a folder, never a file, and publishes no digest.** The `CHECKPOINT_MAPPING_UNVERIFIED`
label is honest for *bytes*. But the doc sentence that justifies it is false:

- `workbench/docs/AUXILIARY_ASSET_AUDIT.md:221` (and the same text at
  `workbench/registry/assets/cluster_b4_classic.yaml:60-61`, `workbench/registry/audits/cluster_c_classic.yaml:60-61`)
  claims the Drive folder "does not document which file maps to the RN50x4 noft Combiner checkpoint, nor
  does it publish a checksum".
- OBSERVED (network, unauthenticated `embeddedfolderview`) refutes the first half. Root folder
  `1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv` lists `CIRR`, `FashionIQ`, `README.md`; `FashionIQ`
  (`1fnGeZQA4XMfEERo-qDCosJXILl1UdQ_l`) lists `RN50_fullft, RN50_imageft, RN50_noft, RN50_textft,
  RN50x4_fullft, RN50x4_noft`; `RN50x4_noft` (`1OHmPOU9VYxDynoCyDfCIY8ijTsSltxuL`) contains
  **exactly one file, `fiq_comb_RN50x4_noft.pt` (last modified 11/3/22)**; `RN50x4_fullft`
  (`1BPE33_XSm33Min0OaGW2Sl9rcddZ-WF-`) contains **two files, `fiq_clip_RN50x4_fullft.pt` and
  `fiq_comb_RN50x4_fullft.pt`**.
- The folder's own README (OBSERVED, 2449 bytes, file id `1G7kZo7paiaRd3mif9_saERyhPevRCM88`) documents the
  release structure and the state keys: items "1) RN50-based Combiner trained on out-of-the-box CLIP
  features - CVPR22 Demo paper", "2) RN50x4-based Combiner trained on out-of-the-box CLIP features -
  CVPR22 Demo paper", "6) RN50x4-based trained Combiner on fine-tuned image and text encoder"; and
  `combiner.load_state_dict(combiner_state_dict['Combiner'])`,
  `clip_model.load_state_dict(clip_state_dict["CLIP"])`, `TARGET_RATIO … equal to 1.25`,
  `PROJECTION_DIM … in the RN50x4-based is equal to 2560`, `HIDDEN_DIM … 5120`. These match
  `validate.py:304-309` (`state_dict["Combiner"]`), `:283-287` (`saved_state_dict["CLIP"]`),
  `combiner.py:11-31`, and the adapter's `2560`/`5120`.

So the noft↔fullft mapping is **documented by the author and consistent with the pinned loader's
constructors**; what is genuinely missing is a byte-level identity (no digest, no size, no
`download_url`; the file id `1Hb3lp3Vk0tSpAcMoq8KG5tFsw-jymryi` is only reachable by scraping the folder
listing, which is not what the registry does, and I did not download it). The blocker is real; the stated
reason is stale prose. **P1** — correcting it matters because the same sentence is reused verbatim for
the fullft sibling row.

### Combiner path handling vs `registry.checkpoint_missing_paths` (assignment point)

The registry's own resolution is a **single file**, not a bundle:

```
combiner checkpoint_path -> /tmp/scratch_ck/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt
combiner missing -> [PosixPath('/tmp/scratch_ck/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt')]
```

(`registry.py:269-291` treats only `run_directory`/`bundle_directory`/`bundle_placeholder` as directories;
`artifact_type: file` therefore falls to the single-file branch at `:292-295`, which requires a regular
non-symlink, non-empty file and, if a digest were recorded, a matching hash — probed directly: empty →
`[path]`, absent → `[path]`, 8 bytes → `[]`, `checkpoint_is_present` → `True`).

The adapter consumes the identical path: `CombinerNoftAdapter.command` passes `str(checkpoint)` verbatim
as `--combiner-path` (`adapters/models.py:92`). OBSERVED, with the interpreter resolver stubbed because
`WORKBENCH_CLIP4CIR_PYTHON` is unset (`runtime.py:1825` → `:1475-1493` raises):

```
combiner-> /VERIFIED/PY /srcroot/src/validate.py --dataset fashionIQ --combining-function combiner \
  --combiner-path /tmp/scratch_ck3/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt \
  --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad
```

and a full plan build with the preflight stubbed (OBSERVED) gives
`artifact_type = file`, `directory_members = []`, `bundle_manifest_dgst = None`,
`checkpoint_sha256 = a677edf4…`. **The combiner entry is internally consistent: `checkpoint_missing_paths`,
`checkpoint_path`, `_hash_checkpoint` and the adapter all address the same regular file.** No combiner-side
divergence exists.

The sibling's just-landed change is correct against the registry but exposes an *unclosed* sibling defect:
with a staged bundle at `/tmp/probe_bundle/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt/{combiner_state.pt,clip_state.pt}`,
`checkpoint_missing_paths` returns `[]` (members verified under the placeholder path) and the adapter now
reads the same members (OBSERVED, `combiner_path = checkpoint / "combiner_state.pt"` at `adapters/models.py:105-106`;
the diff is `- bundle_dir = checkpoint.parent` → `+ checkpoint / ...`), but the plan builder still classifies
`artifact_type in ("run_directory", "bundle_directory")` only (`evaluate_models.py:321`), so
`directory_members`/`bundle_manifest_digest` stay empty/`None` for `bundle_placeholder`
(OBSERVED: `artifact_type = bundle_placeholder, directory_members = [], bundle_manifest_dgst = None`).
Downstream, `_plan_members` iterates `plan.directory_members or []` and `plan.checkpoint_bundle or {}`
(`evaluate_models.py:991-1014`) → `manifest_digest(None)` → `None` (`:1036-1039`); `_members_live`
(`:643-682`) sees `multi_member = True` but `members = {}` and then `recorded_manifest = None` where
`if not recorded_manifest: return True` (`:657-658`); `_hash_checkpoint` (`:612-629`) falls through to
`if checkpoint.is_file()` → False (it is a directory) → `_sha(checkpoint)` hashing the declared member
list `[]`, which `sha256_file` turns into *all* regular children. Net: the fullft bundle's member manifest
is never recorded and member-level change detection is a no-op
(`_bundle_members_live` at `:725` is **defined and never called** — `grep -n _bundle_members_live` → `:725` only).
**This is the fullft sibling's defect, not the combiner's** — recorded here because the assignment asked
me to check the combiner path against the same helper, and the answer is that the helper is *not* what
the combiner entry needs. P2 for the sibling; no action for `combiner_rn50x4_noft`.

## EVALUATION PROTOCOL

Registry: `literature_split_label: original` (`:680`), `native_protocol: fashioniq_original_split` (`:681`),
`supported_protocols: [fashioniq_original_split]` (`:682`).

Workbench protocol table (`src/evaluation/fashioniq.py:171-181`): `fashioniq_original_split` =
`gallery="full_split"`, `reference_excluded=False`, alias `fashioniq_original`; `resolve_fashioniq_protocol`
(`:190-201`) raises on an unknown name rather than defaulting to reference-eligible.

Pinned evaluator (`workbench/third_party/CLIP4Cir/src/validate.py`), read in full:

- Gallery: `fashioniq_val_retrieval` (`:113-132`) builds `FashionIQDataset('val', [dress_type], 'classic', preprocess)`
  (`:127`) plus a relative dataset (`:129`) and `extract_index_features(...)` (`:128`).
- Candidate universe = the **complete** `image_splits/split.{category}.val.json` in file order
  (`data_utils.py:144-147` `self.image_names.extend(json.load(f))`), iterated whole
  (`utils.py:30-31,38-43`, DataLoader default `shuffle=False`) — no dedup, no sort.
- **Reference stays eligible.** `compute_fiq_val_metrics` (`:21-57`): `distances = 1 - predicted_features @ index_features.T`
  (`:44`), ascending `torch.argsort` (`:45`), labels vs `target_names` only (`:49-51`), one-hit assert (`:51`).
  There is no reference mask and no `-10e10` in the FashionIQ path. Contrast inside the same file: CIRR
  *does* mask (`:162-166`) — the strongest available evidence that FashionIQ's eligibility here is
  deliberate.
- Category loop order shirt, dress, toptee (`:325-346`); macro = `statistics.mean` of the three
  (`:353-354`); digits printed with `f"{x = }"` reprs (`:344-354`).

**Verdict on the declaration: VERIFIED (STATIC_AUDIT).** `models.yaml:681-682` is faithful to the code.
This protocol is a *different cohort* from every reference-excluded family in the workbench
(`fashioniq_full_gallery_ref_excluded`, `fashioniq_val_split`), and the docs refuse cross-protocol
aggregates (`docs/FASHIONIQ_PROTOCOL_PARITY.md:9-22`, `docs/PROTOCOL_AUDIT.md`).

## DATASET CONTRACT

- `fashioniq_layout: fashioniq_standard` (`:677`), `preparation_contract: manual_standard` (`:678`),
  `preparation_id: clip4cir_fashioniq` (`:679`), `canonical_dataset_link: fashionIQ_dataset` (`:660`),
  `fashioniq_image_extensions: [".jpg"]` (`:661`).
- Evaluator root is hard-coded, no CLI override: `data_utils.py:11` `base_path = Path(__file__).absolute().parents[1].absolute()`;
  `:140` `base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json'`;
  `:146` `… / 'image_splits' / …`; `:158/:161/:170/:176` `… / 'images' / f"{name}.jpg"`. OBSERVED
  `validate.py:263-277` argparse declares only `--dataset/--combining-function/--combiner-path/--projection-dim/--hidden-dim/--clip-model-name/--clip-model-path/--target-ratio/--transform` — **no root flag.**
- Preparation contract `clip4cir_fashioniq` (`registry/preparation_contracts.yaml:140-169`): raw inputs =
  `captions/cap.{dress,shirt,toptee}.val.json`, `image_splits/split.{…}.val.json`, `images` (`:147-153`);
  `generated_artifacts: []` (`:154`); `external_assets` = OpenAI CLIP RN50x4 base weights with
  `expected_local_location: ~/.cache/clip/RN50x4.pt` and `expected_sha256: 7e526bd1…` (`:156-166`);
  `deterministic_status: DETERMINISTIC_AT_EVALUATION`, `automation_policy: MANUAL_REQUIRED` (`:167-168`).
- OBSERVED (filesystem): `data/fashionIQ_dataset/` has `captions/` (9 files incl. all three `*.val.json`),
  `image_splits/` (9 files incl. all three `*.val.json`), `images/` (74381 files), `qwen_captions/`.
  `data/FashionIQ -> fashionIQ_dataset`. Split/gallery counts OBSERVED:
  `dress gallery=3817 triplets=2017`, `shirt gallery=6346 triplets=2038`, `toptee gallery=5373 triplets=1961`
  (total 15536 images, 6016 triplets).
- OBSERVED: `workbench/third_party/CLIP4Cir/fashionIQ_dataset` is **absent** (not a symlink, not a dir).
  Staging is implemented and generalized: `prepare_dataset.py:28-67` `prepare_canonical_link`
  (refuses a differently-targeted existing link `:49-50`, refuses any other existing path `:54-56`,
  check-only/dry-run short-circuits `:58-63`, creates `destination.symlink_to(root, target_is_directory=True)`
  `:65`), driven by `models_requiring_link()` (`:70-77`) over every model declaring
  `canonical_dataset_link` (`models.yaml:33` csmcir, `:660` combiner noft, `:716` fullft, `:835` sprc).
- Adapter ignores `EvalRequest.dataset_root` (never referenced in `adapters/models.py:90-94`); the only
  path is the hard-coded link.
- **Docs still carry the pre-generalisation claim.** `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:171-173`
  ("the workbench only links that path for CSMCIR"), `docs/audits/PHASE2_SOURCES_B.md:68`, `:80`, `:93`,
  `:106` and `docs/audits/PHASE2_DATA_CONTRACT.md:113` all cite `prepare_dataset.py:prepare_csmcir` as
  "CSMCIR-only". `grep -rn "def prepare_csmcir\|prepare_csmcir" workbench --include=*.py` → **no such
  symbol exists**; the helper is generic and the combiner/fullft/sprc entries are all covered by
  `models_requiring_link()`. (`docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:198` states the generalization
  correctly, so the same file contradicts itself.) The *behaviour* described is still right — the link is
  absent for this model today — but the citation is dead and the "CSMCIR-only" framing is stale. P3.
- OBSERVED `preflight_blockers` for this model (7 blockers, verbatim first five): three
  `FashionIQ image extension requirement: {3817,6346,5373} of N {dress,shirt,toptee} validation images
  lack .jpg in /home/heheboiz/data/cir/data/fashionIQ_dataset/images`, `OpenAI CLIP RN50x4 base weights
  missing: /home/heheboiz/.cache/clip/RN50x4.pt`, `canonical dataset link missing:
  …/third_party/CLIP4Cir/fashionIQ_dataset -> …/data/fashionIQ_dataset`, plus the auxiliary-asset
  restatement and `execution environment not verified: WORKBENCH_CLIP4CIR_PYTHON=None`.

### One registry comment is factually wrong about the root it stages

`models.yaml:656-659` (and identically `:712-715`) reads: *"`src/data_utils.py:base_path` fixes the data
root to `<checkout>/fashionIQ_dataset` … The canonical FashionIQ root is staged at that path by an
explicitly authorized preparation step."* The link does point `<checkout>/fashionIQ_dataset` at the
canonical root, and `prepare_canonical_link` will create it — but the canonical root contains **0 `.jpg`**
and 74381 `.png`, while the evaluator opens `f"{name}.jpg"` (`data_utils.py:158,161,170,176`). So the
staged root does **not** satisfy the evaluator, and the registry states no conversion or alias step
(`preparation_contracts.yaml:154` `generated_artifacts: []`, `:168` `MANUAL_REQUIRED`). Unless an
operator supplies a `.jpg`-bearing tree, every `PIL.Image.open` raises. **P0.**

### The failure would be silent-then-fatal, and can change the denominator

`data_utils.py:151-185` wraps `__getitem__` in `try/except Exception as e: print(f"Exception: {e}")`
returning `None` (`:182-183`); `utils.py:81-88 collate_fn` *filters those `None`s out*
(`batch = list(filter(lambda x: x is not None, batch))`). In classic (gallery) mode a dropped row silently
shrinks the candidate universe; in relative (query) mode it silently shrinks the query set and therefore
`len(labels)` — the Recall denominator (`validate.py:54-55`). The one-hit `assert` at `validate.py:51`
is the only downstream guard and does not fire for a *missing* query. A `.png` canonical root therefore
produces a look-alike number computed over quietly different sets rather than a clean crash. **P0.**

## PREPROCESSING CONTRACT

- `validate.py:279` `clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)`;
  `:280-281` `input_dim = clip_model.visual.input_resolution`, `feature_dim = clip_model.visual.output_dim`.
- `validate.py:289-291` `if args.transform == 'targetpad': preprocess = targetpad_transform(args.target_ratio, input_dim)`;
  defaults `--target-ratio 1.25` (`:273`) and `--transform targetpad` (`:274-275`).
- `data_utils.py:82-96`: `Compose([TargetPad(target_ratio, dim), Resize(dim, interpolation=PIL.Image.BICUBIC),
  CenterCrop(dim), _convert_image_to_rgb, ToTensor(), Normalize((0.48145466,0.4578275,0.40821073),
  (0.26862954,0.26130258,0.27577711))])`. `TargetPad.__call__` (`:54-63`) pads to the target ratio only
  when `actual_ratio >= self.target_ratio` (`:57`).
- The gallery index uses the **same transform instance** (`validate.py:127` constructs the classic dataset
  with the same `preprocess`; `data_utils.py:177` applies it).
- With RN50x4 `input_dim = 288`, `feature_dim = 640`. Corroborated by the paper on its own terms:
  OBSERVED (arXiv HTML `2308.11485v1`, body text) §4.1 — *"It takes input images of 288 × 288, and the
  size of its embedding space is d = 640"*; *"the first two linear layers before the concatenation have
  input-output dimensionality equal to (d, 4d)"* → `projection_dim = 2560`, and `combiner.py:24-25`
  (`Linear(2·4d, hidden)`, `Linear(hidden, d)`) with `hidden_dim = 5120` = 8d. `combiner.py:18-31` matches
  exactly.
- **Adapter parity: exact.** `adapters/models.py:91-94` passes
  `--dataset fashionIQ --combining-function combiner --combiner-path <path> --projection-dim 2560 --hidden-dim 5120
  --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad`. It omits `--clip-model-path`, which
  is the *defining* property of noft: `validate.py:283-287` overlays `saved_state_dict["CLIP"]` only when the
  flag is set, so noft runs on the bare `clip.load('RN50x4')` weights. The sibling adapter does pass it
  (`:107-109`). Omission is correct semantics, not an omission bug.

## AUGMENTATION CONTRACT

- **None at evaluation.** `validate.py` never calls `generate_randomized_fiq_caption`; that function is
  defined at `utils.py:57-78` (four randomized compositions, `random.random()`) and its only callers are
  training scripts (`clip_fine_tune.py`, `combiner_train.py`). STATIC_AUDIT via grep over the checkout.
- Caption composition is **deterministic**: `validate.py:89-93`
  `flattened_captions = np.array(captions).T.flatten().tolist()` then
  `f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i+1].strip('.?, ')}"` for
  `i in range(0, len, 2)`, then `clip.tokenize(input_captions, context_length=77)` — no `truncate=True`,
  so an over-long composition raises rather than being silently cut.
- `clip_model = clip_model.float().eval()` (`validate.py:124`); `combiner.eval()` (`:308`); extraction under
  `torch.no_grad()` (`utils.py:40`, `validate.py:96`).

## GALLERY CONTRACT

- Candidate universe per category = the complete `split.{category}.val.json` in source order, no dedup, no
  sort (`data_utils.py:144-147`; `utils.py:30-31,38-43`; `validate.py:127-128`).
- Reference is a gallery member by construction and is additionally *used* as the query image:
  `validate.py:80` `name_to_feat = dict(zip(index_names, index_features))` and `:101-104` retrieves the
  reference features from that map. **The reference can outrank the target; no exclusion protects
  Recall@K.** That is the declared cohort, not a bug — and it is exactly why these numbers are not
  interchangeable with the reference-excluded cohorts.
- OBSERVED gallery counts (canonical root): dress 3817, shirt 6346, toptee 5373; triplets 2017/2038/1961.
  Macro is per-category equal-weight, i.e. unweighted despite near-equal category sizes.

## METRIC CONTRACT

- `validate.py:41` gallery features normalized; `:44` `distances = 1 - predicted_features @ index_features.T`;
  `:45` ascending `argsort`; `:49-51` labels vs targets with a one-hit assert; `:54-55`
  `recall_at10 = sum(labels[:, :10]) / len(labels) * 100`, `recall_at50 = labels[:, :50]`.
- Macro at `:353-354` is `statistics.mean` over `[shirt, dress, toptee]`, appended `:325-346`. The
  evaluator prints **no R@1** and no single scalar; the gate's `mean` is therefore workbench-derived, not
  paper-printed (see PAPER-METRIC STATUS).
- Workbench gate (`paper_gate.py`): `TOLERANCE_PCT = 5.0` **relative** (`:27`), per-metric
  (`METRIC_KEYS = ("r10","r50","mean")`, `:29`), per-category when both sides publish it
  (`CATEGORY_KEYS`, `:35-37`; compared at `:349-363`; a paper category with no observed counterpart lands
  in `detail` as unchecked, `:382-384`), `COMPARABILITY_FIELDS` (`:72`), `VARIANT_KEYS =
  ("caption_alias_staged","declared_variant")` (`:83`), refusals ordered before arithmetic (`:292-335`).
- `metrics_extraction.py:309-342`: dispatch is `limn` (`:328-329`) and `csmcir` (`:330-331`) only;
  everything else returns `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE`. OBSERVED for this model with
  a synthetic stdout containing the evaluator's real shape (`shirt_recallat10 = 44.41` … `average recall10`,
  `average recall50`) → `{'extraction_status': 'AGGREGATE_PARSER_UNAVAILABLE', 'parser_id': None,
  'observed_metrics': None, 'category_metrics': None, 'parity_status': 'PARSER_UNAVAILABLE', …}`.
- `_check_parity` (`metrics_extraction.py:28-60`) now routes through `paper_gate.compare_metric` with a
  5-percent **relative** rule and refuses a partial row (`p_val is None` → `PARITY_NOT_EVALUATED`) — the
  context's change 2 holds for the code path this model would reach (it just never reaches it).

## SOURCE-TO-WORKBENCH PARITY

| Component | Pinned source | Workbench | Verdict |
|---|---|---|---|
| Commit pin | `dfed9f748a8a…` | `models.yaml:674`; observed HEAD + `source_clean_and_pinned` `(True, sha, None)` | MATCH |
| Entrypoint + CLI shape | `validate.py:263-277` | `adapters/models.py:88,91-94` | MATCH (byte-for-byte flags) |
| noft semantics | `:283-287` skipped when `--clip-model-path` unset | adapter omits it | MATCH |
| Backbone | `clip.load('RN50x4')` `:279` | `auxiliary_assets.yaml:162-178` (URL + sha256 = CDN object name, `cache_root_fixed: ~/.cache/clip`) | MATCH (see below) |
| Preprocess | `targetpad_transform(1.25, 288)` `:289-291`, `data_utils.py:82-96` | adapter `--transform targetpad --target-ratio 1.25` | MATCH |
| Caption composition | `validate.py:89-93` | adapter runs the native script; no local reimplementation is on this path | MATCH |
| Gallery / protocol | full ordered split, reference eligible | `models.yaml:681-682`; `src/evaluation/fashioniq.py:171-181` | MATCH |
| Metric extraction | stdout `…_recallatNN = …`, `average recallNN = …` `:344-354` | `metrics_extraction.py:334-342` | **MISMATCH — no parser** |
| Dataset root | `<checkout>/fashionIQ_dataset`, `.jpg` | link declared, absent; canonical root is `.png`-only | **MISMATCH** |
| Checkpoint file | `RN50x4_noft/fiq_comb_RN50x4_noft.pt` (one file) | folder URL only; `artifact_type: file`; no digest/size | **MISMATCH (identity)** |

## RUNTIME STATUS

- Registry: `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`:686`), `command_status: COMMAND_AUDITED`
  (`:687`), `replay_status: REPLAY_CODE_IMPLEMENTED` (`:662`), `replay_type: OFFICIAL_CLI` (`:665`),
  `category_model_policy: shared` (`:663`), `checkpoint_serialization: state_dict` (`:664`),
  `environment_required: true` (`:666`), `strict_clean_checkpoint_available: null` (`:696`).
- Environment block `:688-695`: `WORKBENCH_CLIP4CIR_PYTHON`, `confidence: VERIFIED_README`, python 3.8,
  pytorch 1.11.0, torchvision 0.12.0, cuda UNKNOWN, packages `[pandas==1.4.2, comet-ml==3.21.0, clip]`.
  OBSERVED faithful to `README.md:113-118` (`conda create -n clip4cir -y python=3.8`,
  `pytorch=1.11.0 torchvision=0.12.0`, `pandas=1.4.2`, `comet-ml==3.21.0`,
  `pip install git+https://github.com/openai/CLIP.git`). Note `clip` is installed from git **main**,
  i.e. unpinned, and the registry records it as the unversioned string `clip`.
- OBSERVED: `workbench/envs/` contains only `README.md` and `csmcir-import.txt`; no interpreter.
  `WORKBENCH_CLIP4CIR_PYTHON` is unset in the environment. `CombinerNoftAdapter.command` calls
  `_runtime.python_executable` (`:91` → `runtime.py:1825` → `:1475-1493`), which **raises**
  `RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_CLIP4CIR_PYTHON=None`
  (OBSERVED). So the adapter is audited but *not constructible* on this host.
- OBSERVED `guarded_plan(...)` → `plan: None` with 8 blockers (the 7 above plus `checkpoint missing:
  …/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt` and `checkpoint mapping unresolved`).
- OBSERVED `pc_readiness.classify(preflight_blockers(...))` → `('BLOCKED_ENVIRONMENT', 'execution
  environment not verified: WORKBENCH_CLIP4CIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_CLIP4CIR_PYTHON not set)')`.
  This disagrees with `docs/READINESS_MATRIX.md:60` (which prints the `.jpg` dataset blocker first) and
  with `docs/reviews/PC_RELEASE_READINESS.md:46`, which records `BLOCKED_ENVIRONMENT` — the same state by a
  different route. The classification is order-dependent text matching (`pc_readiness.py:44-71`): the
  generic needle `"environment"` precedes `"FashionIQ"` in `BLOCKER_KINDS`, so a *declared-but-uninstalled
  interpreter* (external) is ranked ahead of the `.jpg` data defect (also external). Both are honest; the
  ordering is arbitrary. P3.
- `cuda: UNKNOWN` is honest but incomplete: `utils.py:15-18` selects CUDA only `if torch.cuda.is_available()`,
  so this evaluator falls back to CPU. Not GPU-mandatory.
- **No runtime artifacts.** OBSERVED `workbench/artifacts/results/fashioniq_original_split/` contains only
  `mock/` (`mock_csmcir-fiq_a.json`, `mock_sprc-sprc_fiq.json`); `find workbench/artifacts -iname "*combiner*" -o -iname "*noft*"`
  → empty. No `run_id`, no report, no per-query export. **Nothing to compare.**

## PAPER-METRIC STATUS

Registry: `reported_r10: null`, `reported_r50: null`, `reported_mean: null` (`models.yaml:683-685`).

I re-derived this from the paper rather than trusting the rows.

**OBSERVED, all 11 `<table>` elements of `https://arxiv.org/html/2308.11485v1` parsed and dumped**
(11 elements = 10 numbered tables + the loss equation; script under scratch). Table inventory:
HTML table 2 = paper **Table 1** (FashionIQ val, CF × IFT × TFT grid, RN-50 only), 3 = Table 2 (CIRR val,
same grid), 4 = Table 3 (FashionIQ Combiner ablation), 5 = Table 4 (CIRR Combiner ablation), 6 = Table 5
(FashionIQ two-stage vs end-to-end), 7 = Table 6 (CIRR), 8 = Table 7 (FashionIQ preprocess comparison),
9 = Table 8 (CIRR preprocess), **10 = Table 9 (FashionIQ val SotA)**, 11 = Table 10 (CIRR test SotA).

Machine check of backbone coverage: `RN-50x4` appears in **HTML tables 10 and 11 only** (Tables 9 and 10);
`RN-50` in all ablations. Body text: *"Throughout all the experiments, we use the RN-50 CLIP model."*

Table 9 (`paper_page` 16 in the registry, lifted HTML table 10) — the only RN-50x4 FashionIQ row:

```
['Ours', 'RN-50x4', 'Transformer', '44.41','65.26','39.46','64.55','47.48','70.98','43.78','66.93']
```

My arithmetic on those cells: macro R@10 `(44.41+39.46+47.48)/3 = 43.7833` → printed `43.78`; macro R@50
`(65.26+64.55+70.98)/3 = 66.93` → printed `66.93`; scalar mean `(43.78+66.93)/2 = 55.355 → 55.36`. This
reproduces the sibling's committed benchmark row `clip4cir_rn50x4_fullft_fiq_tab9`
(`papers.yaml:208-289`: `reported_metrics {r10: 43.78, r50: 66.93, mean: null}`, `derived_metrics mean
55.36 WORKBENCH_DERIVED`, `reference_status: VERIFIED`, `review_status: CONFIRMED`) and
`models.yaml:744-746`. **The sibling's numbers are correct as committed, including the honest separation
of a derived mean.**

**There is no RN-50x4 noft FashionIQ row in this paper.** The registry's `null`s are therefore correct
*for the cited paper*. The committed row that says so is
`clip4cir_rn50x4_noft_fiq_absent_in_paper` (`papers.yaml:290-352`), whose `metric_provenance` carries the
full inventory above and whose `verification_method` records exhaustive greps plus a complete `<table>` dump.
Its `reference_status` is `UNVERIFIED`, `review_status` `PENDING`.

**But an author-published noft number does exist, in a paper the registry does not cite.** I fetched it
myself: `https://openaccess.thecvf.com/content/CVPR2022/papers/Baldrati_Effective_Conditioned_and_Composed_Image_Retrieval_Combining_CLIP-Based_Features_CVPR_2022_paper.pdf`,
**Table 1**, Fashion-IQ validation:

```
Proposed approach   36.36  58.00  31.63  56.67  38.19  62.42  35.39  59.03
```

§3.2: *"the CLIP encoders have been kept frozen and the only trained part of the model is the Combiner
function. The target ratio in the preprocess pipeline was set to 1.25."* §5.1 states the same 15,537 /
6,017 val split. My arithmetic: macro R@10 `(36.36+31.63+38.19)/3 = 35.3933` → printed `35.39`; macro
R@50 = `59.03`; scalar mean `(35.39+59.03)/2 = 47.21`. The Drive README attributes release item 2 to
"CVPR22 Demo paper", so this row and `fiq_comb_RN50x4_noft.pt` are the same configuration.

**Consequence for this model, and it is not a registry-numbers defect:**

- `select_reference('combiner_rn50x4_noft', checkpoint_training_noise_pct=None, protocol_id='fashioniq_original_split')`
  (OBSERVED) → `(None, "combiner_rn50x4_noft has references but none is verified: ['UNVERIFIED']")`.
  The model's three committed rows are all `UNVERIFIED` (`papers.yaml` OBSERVED via loader:
  `clip4cir_rn50x4_noft_fiq_absent_in_paper`, `combiner_rn50x4_noft__clip4cir_rn50x4_fiq_gallery_protocol`,
  `combiner_rn50x4_noft__clip4cir_paper_note_backbone_inventory_and_version`).
- Therefore `evaluate_gate` (OBSERVED, called directly with `{r10: 43.78, r50: 66.93, mean: 55.36}` and
  `data_kind='experiment', real_execution=True`) → `GateVerdict(status='METRIC_REFERENCE_UNVERIFIED',
  passed=False, variant=False, comparisons=[], blockers=['no committed reference describes this
  experiment'], detail='without a verified published row there is nothing to reproduce')`.
  The noft model is **unconditionally ungradeable**, whatever it reproduces. P1.
- The `preprocessing_id` of the absent-in-paper row (`papers.yaml:313-314`) states the noft release README
  "confirms TARGET_RATIO 1.25 for all its pretrained models" — I confirmed that string in the author Drive
  README myself. Good.
- The noft row's `aggregation_method` claims the macro is unweighted (`:319-320`). The paper's own
  Table 9 asserts this only by printing Shirt/Dress/Toptee then an Average whose value equals the
  unweighted mean (recomputed above to the printed digits); its §4.2.1 prose defines Recall@K but never
  states a weighting, and a grep of the full body text for `aggregat`/`macro`/`unweighted` → **0 hits**.
  The unweighted claim is therefore an *inference from the printed arithmetic*, not a paper statement.
  It happens to be right. P3.
- The paper-note row's `dataset_version` (`:456-458`) carries the 15,537/6,017-vs-local-15,536/6,016
  off-by-one honestly. I reproduced the local side (OBSERVED counts) and the paper side (OBSERVED §4.2.1
  text `"The validation and test sets consist of 15,537 and 15,538 images, respectively, with 6,017 and
  6,119 triplets."`). Unresolved, affects every FashionIQ model, not CLIP4Cir specifically.
- `TOLERANCE_PCT` context change 1 is present: `CATEGORY_KEYS` (`paper_gate.py:35-37`) and the category
  comparison loop (`:349-363`) do exactly what the change claims — compare only when both sides state it,
  record the rest as unchecked in `detail` (`:382-384`).

## CRITICAL FINDINGS

**C-1 — P1 — the model can never be graded: every committed reference for it is `UNVERIFIED`, and the one
verified-looking target lives in an uncited paper.** OBSERVED `select_reference(...)` → `(None, "combiner_rn50x4_noft
has references but none is verified: ['UNVERIFIED']")`; OBSERVED `evaluate_gate(...)` →
`METRIC_REFERENCE_UNVERIFIED`. Meanwhile the author's own CVPR22 table (`36.36/58.00, 31.63/56.67,
38.19/62.42, avg 35.39/59.03`, macro mean 47.21) is a genuine headline row for exactly this
configuration and is *kept out of the gate by design*. Nothing in this model's row set can ever return
`PASS`. Action: either add a `VERIFIED` noft reference row carrying the CVPR22 citation (with
`paper_id` pointing at that paper, not `clip4cir_tomm2023`) plus a `checkpoint_training_noise_pct: 0` and
protocol as derived below, or record explicitly, in the row, that grading this configuration is out of
scope. Do not silently leave "unverifiable" indistinguishable from "not attempted".

**C-2 — P0 — the declared dataset link does not satisfy the evaluator (`.jpg` vs `.png`), and the
evaluator's error path would silently change the Recall denominator rather than crash.** Registry
`:660-661` declares `canonical_dataset_link: fashionIQ_dataset` and `.jpg`; `data_utils.py:158,161,170,176`
open `f"{name}.jpg"`; OBSERVED the canonical root holds `74381 png / 0 jpg` and the link is absent;
`data_utils.py:182-183` swallows the resulting `FileNotFoundError` and `utils.py:87` drops the row, so
relative-mode drops shrink `len(labels)` at `validate.py:54-55`. Action: stage a `.jpg`-bearing tree
(hardlink/alias with a recorded conversion) as part of the same authorized step, or change the declared
extension and the evaluator's source — the former is the only option that keeps the source pinned and
unmodified. Do not run against the `.png` root.

**C-3 — P1 — no metric parser, so a successful run produces no comparable number.** OBSERVED
`extract_aggregate_metrics('combiner_rn50x4_noft', <real-shaped stdout>, '', 0, {…})` →
`AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE` (`metrics_extraction.py:334-342` dispatches only
`limn` `:328` and `csmcir` `:330`). The evaluator's output is trivially parseable
(`validate.py:344-354`: six `{cat}_recallat{10,50} = …` lines plus `average recall10` / `average recall50`,
category order shirt, dress, toptee). Action: add a parser mirroring `parse_csmcir_output`, recompute the
macro from the six category values and refuse when the declared averages disagree — exactly the pattern at
`metrics_extraction.py:160-198` (whose `0.05` tolerance is *self-consistency*, per the context's change 2,
and is the right rule here too).

**C-4 — P1 — the checkpoint is not resolvable to bytes, and the documented reason is false.**
`models.yaml:698-710` gives a folder URL, `download_url: null`, no size, no digest. OBSERVED the folder
listing names exactly one file (`FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt`) and the author Drive
README documents the release structure, the dims (2560/5120) and the `'Combiner'`/`'CLIP'` keys — so
`AUXILIARY_ASSET_AUDIT.md:221` / `cluster_b4_classic.yaml:60-61` / `cluster_c_classic.yaml:60-61`
("does not document which file maps … nor does it publish a checksum") is half-wrong: the mapping *is*
documented, the checksum is not. Action: correct the prose (three files carry it), record the author
filename and the folder README as provenance, and keep `CHECKPOINT_MAPPING_UNVERIFIED` on bytes until a
digest exists. Do not transfer this finding to the fullft row as-is; the fullft pairing question is
genuinely open and needs its own sentence.

## MINOR FINDINGS

- **M-1 — P2 — `checkpoint_training_noise_pct: null` is honest but under-describes the variant.** The row
  (`models.yaml:705`) records `null`; the sibling records `0` (`:769`) with a §4.1 note. The noft
  configuration is *by construction* noise-free (frozen encoders; CVPR22 §3.2 and the Drive README both
  say so), and `papers.yaml:303` / `:363` already assert `checkpoint_training_noise_pct: 0` for the
  noft-related rows. Actions: the two representations disagree on a condition the gate uses for
  selection; align them, or state in the row why `null` is preferred. Severity P2 because it decides
  whether a future gate ever selects a reference for this checkpoint.
- **M-2 — P3 — stale "CSMCIR-only link" citations in four docs.** `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:171-173`,
  `docs/audits/PHASE2_SOURCES_B.md:68,80,93,106`, `docs/audits/PHASE2_DATA_CONTRACT.md:113` cite
  `prepare_dataset.py:prepare_csmcir`, a function that does not exist (`grep -rn "def prepare_csmcir"
  workbench --include=*.py` → nothing); the helper is `prepare_canonical_link` (`:28`), driven by
  `models_requiring_link()` (`:70-77`) over every `canonical_dataset_link` declaration. Behaviour claim
  still true today for this model; provenance is wrong. `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:198`
  already says it correctly.
- **M-3 — P3 — the registry comment claims the staged root satisfies the evaluator, and it does not.**
  `models.yaml:656-659` / `:712-715`. Same root cause as C-2, manifesting as a false comment rather than
  false code.
- **M-4 — P2 — the upstream `clip` dependency is unpinned.** `README.md:118` installs
  `git+https://github.com/openai/CLIP.git`; `models.yaml:695` records the bare string `clip`.
  `clip.load`/`clip.tokenize` decide the RN50x4 class and the caption tokenization, so the tokenizer and
  model definition are floating. Action: record an `openai/CLIP` commit sha in the environment spec.
- **M-5 — P2 — `--projection-dim 2560` / `--hidden-dim 5120` are hard-coded without a recorded
  derivation.** `adapters/models.py:93`. Correct (`d = 640`; paper §4.1 `(d,4d)`; `combiner.py:18-31`;
  CLIP `d = 640` for RN50x4 corroborated in the same repo's paper §4.1 and the Drive README's
  "in the RN50x4-based is equal to 2560 / 5120"), but a wrong value fails only at `load_state_dict`
  (`validate.py:307`). Action: one-line derivation note on the checkpoint record.
- **M-6 — P3 — the base-weights digest is a URL convention, not a verified hash of bytes.**
  `auxiliary_assets.yaml:166,177` records `expected_sha256: 7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd`,
  which is exactly the CDN object-name segment (OpenAI's own convention, and upstream CLIP self-verifies
  against it). OBSERVED `HEAD` on that URL → `HTTP/2 200`, `content-length: 421854225`, yet
  `expected_size_bytes: null` (`:174` region) even though the length is observable. Not downloaded
  (assignment forbids). Action: record the observed size; keep `provenance_status: VERIFIED` scoped to
  "URL + published digest", not "digest of bytes on disk".
- **M-7 — P3 — `cuda: UNKNOWN` undersells a known fact.** `utils.py:15-18` falls back to CPU; no
  unguarded `.cuda()` exists in this model's path. The evaluator is therefore CPU-capable (slow), and the
  registry could say so.
- **M-8 — P3 — the noft row's `aggregation_method` states an unweighted macro the paper never asserts in
  words.** `papers.yaml:319-320`. My arithmetic confirms the printed Average equals the unweighted mean
  to the printed digits (43.7833→43.78, 66.93). Worth a note that it was inferred from the arithmetic and
  from `validate.py:353-354`, not from prose.
- **M-9 — P2 (sibling, recorded because the assignment asked) — `bundle_placeholder` never gets a member
  manifest.** `evaluate_models.py:321` covers `run_directory`/`bundle_directory` only, so for
  `clip4cir_rn50x4_fullft` (OBSERVED with a staged bundle: `artifact_type='bundle_placeholder'`,
  `directory_members=[]`, `bundle_manifest_dgst=None`) the manifest stays empty, `_plan_members`
  (`:991-1014`) returns `None`, `_members_live` (`:643-682`) short-circuits on
  `if not recorded_manifest: return True` (`:657-658`), and `_bundle_members_live` (`:725-775`) is
  **defined and never called** (`grep -n _bundle_members_live` → `:725` only). Member-level change
  detection for that bundle is a no-op. The adapter half of the change is right (OBSERVED members read
  under the placeholder path itself, matching `checkpoint_missing_paths`); the plan/report half is not
  closed. `tests/test_replay.py:144-147` still asserts only the placeholder *shape*
  (`artifact_type == "bundle_placeholder"` and `combiner_state.pt in required_files`) — no test asserts a
  staged bundle produces a plan whose members are recorded.
- **M-10 — P3 — `pc_readiness` classification of this model is order-dependent.** OBSERVED
  `('BLOCKED_ENVIRONMENT', …)` because the generic needle `"environment"` (`pc_readiness.py:51`) precedes
  `"FashionIQ"` (`:52`). The dataset blocker is printed first; the environment blocker decides the state.
  Both are external; the ordering carries no meaning, but a reader comparing `docs/READINESS_MATRIX.md:60`
  with `pc_readiness` output will see different labels for the same model.
- **M-11 — P3 — `AVOID`-level:** the alteration `checked` variable in `_members_live` (`evaluate_models.py:664-675`)
  is loop-local and dead outside its branch; harmless but it reads as unfinished refactoring on the same
  path as M-9.

## UNRESOLVED EVIDENCE

1. **Checkpoint bytes.** No `fiq_comb_RN50x4_noft.pt` on this host; no digest, no size, no `download_url`
   in the registry. The Drive listing exposes the file but I did not download it (prohibited), so
   `state_dict['Combiner']` key compatibility (`validate.py:307`) and `combiner.py:18-25` dimension
   compatibility are **UNKNOWN**. Corroborated only by the author's Drive README, which is author
   attestation, not measurement.
2. **Base RN50x4 weights.** `~/.cache/clip` does not exist (OBSERVED `ls: cannot access
   '/home/heheboiz/.cache/clip': No such file or directory`). The CDN serves 421854225 bytes, but
   whether those bytes hash to `7e526bd1…` was **not** verified (would require a ~402 MiB download).
3. **Which Drive file is bound to the CVPR22 Table 1 row.** The Drive README's item 2 attribution plus
   `§3.2`'s frozen-encoder sentence make it the only candidate, but no digest ties the published row to
   the published bytes. **UNKNOWN.**
4. **The 15,537 / 6,017 vs 15,536 / 6,016 off-by-one.** Locally reproduced; paper statements reproduced;
   cause unresolved. Affects every FashionIQ model in the workbench.
5. **Runtime numbers of any kind.** No interpreter (`WORKBENCH_CLIP4CIR_PYTHON` unset; `workbench/envs/`
   has none), no checkpoint, no link, no `.jpg` tree, no GPU run. **Nothing here is RUNTIME_VERIFIED.**
6. **Whether the gate's `mean` would be comparable at all.** `reported_metrics.mean` is `null` for both
   the noft row and Table 9, with the sibling's mean present only as `WORKBENCH_DERIVED`. For a noft row
   built from CVPR22, the same derived-mean treatment would be required; I did not build the row.

---

# EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation
-> checkpoint -> output -> metric comparison`, arrow by arrow.

**1. paper/README/config → this registry entry.** VERIFIED (STATIC_AUDIT / OBSERVED). `models.yaml:667-674`
(method name, title, arXiv URL, repo URL, pin) match OBSERVED `https://arxiv.org/abs/2308.11485`
(4 authors, cs.CV, 2023-08-22) and `workbench/third_party/CLIP4Cir/README.md:1-7`. The environment block
`:688-695` is a line-for-line transcription of `README.md:113-118`. `--projection-dim 2560` /
`--hidden-dim 5120` are consistent with the paper's §4.1 (`d = 640`; `(d,4d)`; `288×288`) and with
`combiner.py:18-31`.

**2. README/config → source dataset.** VERIFIED (STATIC_AUDIT); **fails at runtime.** `README.md:141-163`
documents `project_base_path/fashionIQ_dataset/{captions,images,image_splits}` with `.jpg` examples
(`:152-155`); `data_utils.py:11,140,146,158,161,170,176` implement exactly that, anchored on the checkout.
OBSERVED: the declared link `workbench/third_party/CLIP4Cir/fashionIQ_dataset` is absent, and the canonical
root is `74381 png / 0 jpg`. The arrow is broken by C-2 and the failure is silent (C-2 second half).

**3. source dataset → preprocessing.** VERIFIED (STATIC_AUDIT). `validate.py:289-291` selects
`targetpad_transform(1.25, 288)`; `data_utils.py:54-63,82-96` give the exact pipeline; the gallery uses
the same transform instance (`validate.py:127`, `data_utils.py:177`). Nothing external beyond torchvision.

**4. preprocessing → official evaluator.** VERIFIED (STATIC_AUDIT). `validate.py:263-277` (CLI) →
`:279` `clip.load('RN50x4')` → `:280-281` (dims) → `:289-291` (transform) → `:304-309` (Combiner built,
`state_dict['Combiner']` loaded, `.eval()`, `combining_function = combiner.combine_features`) →
`:325-354` (three categories, per-category R@10/R@50, macro). `--clip-model-path` unset → the `"CLIP"`
overlay at `:283-287` is skipped, which *is* the noft configuration.

**5. official evaluator → workbench invocation.** PARTIAL. `adapters/models.py:86-94` produces a command
that is the README's validation invocation minus the flag this variant must omit, plus explicit values for
every default; OBSERVED the exact argv (interpreter stubbed). `command_status: COMMAND_AUDITED` (`:687`)
and `replay_status: REPLAY_CODE_IMPLEMENTED` (`:662`) are accurate *as audited*. But
`_runtime.python_executable` (`runtime.py:1825` → `:1475-1493`) **raises** because
`WORKBENCH_CLIP4CIR_PYTHON` is unset (OBSERVED), so the command is not constructible on this host.
⇒ BLOCKED at this arrow, independently of data and checkpoint.

**6. workbench invocation → checkpoint.** BLOCKED. `checkpoint_path(...)` =
`workbench/artifacts/checkpoints/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt` (`registry.py:631-634`);
OBSERVED that directory does not exist; `download_url: null` (`:702`) means no acquisition path in the
registry; the author filename (`fiq_comb_RN50x4_noft.pt`, OBSERVED) differs from the placeholder name, and
the registry's own justification for that state is factually wrong (C-4). The *path handling* here is
nonetheless coherent: `artifact_type: file` → `registry.py:292-295` single-file branch → the adapter passes
that identical path as `--combiner-path` (OBSERVED). No divergence exists on the combiner side.

**7. checkpoint → output.** NOT_RUN. No evaluator process; no stdout; no `run_id`; no canonical report; no
per-query export. `workbench/artifacts/results/fashioniq_original_split/` holds only two mock files, neither
for this model (OBSERVED).

**8. output → metric comparison.** NOT_RUN, and blocked by two independent code gaps even if output existed.
(a) No parser: OBSERVED `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE` for this `model_id`
(`metrics_extraction.py:334-342`). (b) No verified reference: OBSERVED `select_reference(...)` → `(None,
"… none is verified: ['UNVERIFIED']")` and `evaluate_gate(...)` → `METRIC_REFERENCE_UNVERIFIED`. The
paper-side target for the *cited* paper is correctly absent (OBSERVED: 11 tables dumped; only HTML tables
10/11 contain RN-50x4). A candidate external row exists and I re-derived it myself: CVPR22 demo Table 1
`36.36 / 58.00 / 31.63 / 56.67 / 38.19 / 62.42 / 35.39 / 59.03` → macro 35.39 / 59.03, mean 47.21.

---

# CHECKLIST

1. **Registered `model_id` present and unique** — VERIFIED — `models.yaml:655`; `grep -n "model_id: combiner_rn50x4_noft"` → one row; `ADAPTERS` keyed once (`adapters/models.py:140`) — NONE — none.
2. **`method_name`/`paper_title` recorded and accurate** — VERIFIED — `models.yaml:667`, `:670`; matches OBSERVED arXiv 2308.11485 and `README.md:1-7` — NONE — none.
3. **`paper_url` resolves and identifies the same work** — VERIFIED — `models.yaml:671`; OBSERVED 4 authors, cs.CV, 2023-08-22, matching title — NONE — none.
4. **`upstream_repo_url` recorded** — VERIFIED — `models.yaml:672`; OBSERVED `git remote -v` → `git@github.com:ABaldrati/CLIP4Cir` — NONE — none.
5. **Pinned commit equals checkout HEAD** — VERIFIED — `models.yaml:674` == `git rev-parse HEAD` → `dfed9f748a8a…`; `source_clean_and_pinned` → `(True, sha, None)` — NONE — none.
6. **Checkout clean** — VERIFIED — `git status --short` empty; `runtime.py:2064-2071` accepts — NONE — none.
7. **`upstream_default_branch` correct** — VERIFIED — `models.yaml:673` `master`; OBSERVED `git remote -v` origin URL — NONE — none.
8. **Pin equals current upstream `master`** — UNKNOWN — no fetch performed (read-only; single-commit detached checkout) — P3 — re-verify on a writable host.
9. **`source_available` / `source_dir` correct** — VERIFIED — `models.yaml:675-676`; tree present with `src/validate.py` (360 lines) — NONE — none.
10. **Entrypoint exists and matches adapter `script`** — VERIFIED — `adapters/models.py:88` `"src/validate.py"`; file present — NONE — none.
11. **`native_protocol` matches the evaluator's gallery** — VERIFIED — `models.yaml:681` vs full ordered split `data_utils.py:144-147`, `utils.py:38-43`; table `src/evaluation/fashioniq.py:172-174` — NONE — none.
12. **`supported_protocols` is exactly the native one** — VERIFIED — `models.yaml:682`; no second branch in `validate.py` for FashionIQ — NONE — none.
13. **`literature_split_label: original` consistent** — VERIFIED — `models.yaml:680`; protocol table `fashioniq.py:172-174`; paper §4.2.1 "All results are on the validation set" — NONE — none.
14. **Reference stays eligible** — VERIFIED — `validate.py:44-55` has no mask; contrast CIRR `:162-166` — NONE — none.
15. **No reference-exclusion mechanism in the FashionIQ path** — VERIFIED — no `-10e10`/mask in `validate.py` except the CIRR branch — NONE — none.
16. **Reference can outrank the target** — VERIFIED — reference is a gallery member (`:127-128`) and is itself the query source (`:80`, `:101-104`) — P3 — none; that *is* the declared cohort.
17. **Sort direction / distance** — VERIFIED — `:44` `1 - pred @ gallery.T`, `:45` ascending `argsort` — NONE — none.
18. **Tie-breaking pinned** — UNKNOWN — `torch.argsort` default unstable; docs decline to pin ties — P3 — accept; measure-zero.
19. **Recall@K arithmetic** — VERIFIED — `:54-55` `sum(labels[:, :K]) / len(labels) * 100` — NONE — none.
20. **Denominator is the query count** — VERIFIED (static) — `len(labels)` = relative-val triplets — P0 — **CONDITIONAL**: `utils.py:87` drops `None` rows, so the denominator can silently shrink (C-2).
21. **One-target-per-query invariant** — VERIFIED — `validate.py:51` assert; OBSERVED local splits have no duplicate targets — NONE — none.
22. **Macro definition matches source** — VERIFIED — `:353-354` `statistics.mean` over `[shirt,dress,toptee]` — NONE — none.
23. **K values are 10 and 50** — VERIFIED — `:54-55` — NONE — none.
24. **`reported_r10` matches the cited paper** — VERIFIED as `null` — `models.yaml:683`; OBSERVED 11 tables dumped, RN-50x4 only in tables 10/11, Table 9's single RN-50x4 row is fully fine-tuned — NONE — none.
25. **`reported_r50` matches the cited paper** — VERIFIED as `null` — same — NONE — none.
26. **`reported_mean` matches the cited paper** — VERIFIED as `null` — same — NONE — none.
27. **An author-published noft row exists elsewhere** — VARIANT — OBSERVED CVPR22 demo Table 1 `Proposed approach` 36.36/58.00, 31.63/56.67, 38.19/62.42, avg 35.39/59.03; §3.2 frozen encoders, target ratio 1.25; Drive README item 2 — P1 — record as a candidate reference **or** state explicitly that grading this configuration is out of scope (C-1).
28. **Macro/mean recomputed from the CVPR22 row** — VERIFIED — my arithmetic: macro R@10 `35.3933→35.39`, macro R@50 `59.03`, mean `(35.39+59.03)/2 = 47.21` — NONE — none.
29. **Macro/mean recomputed from Table 9** — VERIFIED — `43.7833→43.78`, `66.93`, mean `55.355→55.36`; matches `papers.yaml:242-254` and `models.yaml:744-746` — NONE — none.
30. **Checkpoint noise provenance declared** — VARIANT — `models.yaml:705` `null` vs the sibling's `0` (`:769`) and the rows' `0` (`papers.yaml:303`, `:363`); the variant is noise-free by construction — P2 — align the representations (M-1).
31. **Checkpoint acquisition is resolvable** — BLOCKED — `download_url: null`, folder URL only, no digest/size (`models.yaml:702-708`); `checkpoint_availability` → `automatic_download_available: False`, `runnable: False` — P1 — record the file name + digest, or keep blocked and say so (C-4).
32. **Docs' justification for the unresolved mapping is accurate** — VARIANT (half refuted) — `AUXILIARY_ASSET_AUDIT.md:221` + `registry/assets/cluster_b4_classic.yaml:60-61` + `registry/audits/cluster_c_classic.yaml:60-61`; OBSERVED the folder names exactly one noft file and the Drive README documents the mapping — P1 — correct the prose in all three files (C-4).
33. **noft vs fullft mapping is verified, not assumed** — VERIFIED — OBSERVED Drive tree (`RN50x4_noft/` = one `fiq_comb_*`; `RN50x4_fullft/` = `fiq_clip_*` + `fiq_comb_*`); Drive README items 2/6; `validate.py:283-287` (CLIP overlay only via `--clip-model-path`) — NONE — positive finding.
34. **Checkpoint loadable by pinned code** — UNKNOWN — `validate.py:306-307` requires `state_dict["Combiner"]`; key corroborated by the Drive README and `utils.py:117-131` (`save_model` uses `__class__.__name__`); file never obtained — P1 — needs a run.
35. **Combiner dims match the checkpoint's** — UNKNOWN (static-consistent) — `combiner.py:18-31` (`640 → 2560 → 5120 → 640`); adapter passes 2560/5120 (`adapters/models.py:93`) — P1 — needs the real state dict.
36. **Base RN50x4 cache requirement verified** — VERIFIED — `validate.py:279` `clip.load(...)`; `auxiliary_assets.yaml:169-175` `destination_scope: user_cache`, `cache_root_fixed: ~/.cache/clip`, `destination_path: RN50x4.pt` — NONE — none.
37. **Base weights present locally** — BLOCKED — OBSERVED `ls: cannot access '/home/heheboiz/.cache/clip': No such file or directory`; preflight reports `OpenAI CLIP RN50x4 base weights missing` — P0 — stage with digest verification before any run.
38. **Base weight URL and digest recorded** — VARIANT — `auxiliary_assets.yaml:166,177` (URL + `7e526bd1…` = the CDN object name); OBSERVED `HEAD` → `HTTP/2 200`, `content-length: 421854225`; `expected_size_bytes` null — P2 — record the observed size (M-6).
39. **Base weight digest verified against bytes** — UNKNOWN — not downloaded (prohibited) — P1 — verify on the PC.
40. **Dataset root resolution** — VERIFIED — `data_utils.py:11` = checkout root; `:140,146,158,161,170,176` append `fashionIQ_dataset/…` — NONE — none.
41. **No dataset-root CLI exists** — VERIFIED — `validate.py:263-277` has no root flag — NONE — none.
42. **Adapter ignores `EvalRequest.dataset_root`** — VERIFIED — `adapters/models.py:90-94` never references it — P1 — stage the link explicitly; do not add a hidden link (C-2).
43. **Canonical dataset link present** — BLOCKED — OBSERVED `workbench/third_party/CLIP4Cir/fashionIQ_dataset` absent; preflight `canonical dataset link missing: …` — P1 — operator-run `prepare_dataset.py --model combiner_rn50x4_noft` (after C-2 is resolved).
44. **Link staging implemented and safe** — VERIFIED — `prepare_dataset.py:28-67`: refuses a differently-targeted link (`:49-50`), refuses any other existing path (`:54-56`), check-only/dry-run paths (`:58-63`), `symlink_to(..., target_is_directory=True)` (`:65`) — NONE — none.
45. **Link staging covers this model, contrary to four docs** — VERIFIED — `models_requiring_link()` (`:70-77`) reads `canonical_dataset_link`; `models.yaml:660` declares it; `prepare_csmcir` does not exist (`grep`) — P3 — fix the citations (M-2).
46. **Image extension contract matches the evaluator** — VERIFIED — `models.yaml:661` `['.jpg']`; `data_utils.py:158,161,170,176` open `.jpg` — NONE — none.
47. **Images present with the declared extension** — BLOCKED — OBSERVED `images/`: `74381 png`, `ls images | grep -c '\.jpg$'` → `0`; per-category validation ids 3817/6346/5373 all exist as `.png`, none as `.jpg`; preflight agrees — **P0** — stage a `.jpg`-extension tree or change the declaration; do not run against `.png` (C-2).
48. **Silent-skip hazard on a missing image** — VERIFIED (defect present) — `data_utils.py:182-183` `except Exception … print(…)` returning `None`; `utils.py:81-88 collate_fn` filters `None`; `validate.py:51`'s assert is the only guard and does not cover a dropped query — **P0** — same action as #47; a dropped query changes `len(labels)` (C-2).
49. **Preprocessing transform matches source exactly** — VERIFIED — `validate.py:289-291`; `data_utils.py:82-96` (bicubic, center crop, CLIP mean/std), `:54-63` (TargetPad) — NONE — none.
50. **Target ratio 1.25 and input dim 288** — VERIFIED — `:273` default + adapter `1.25`; paper §4.1 "288 × 288 … d = 640"; author Drive README "TARGET_RATIO … 1.25" — NONE — none.
51. **No augmentation at evaluation** — VERIFIED — `generate_randomized_fiq_caption` defined `utils.py:57-78`, only training callers; `validate.py` never calls it — NONE — none.
52. **Validation caption composition deterministic** — VERIFIED — `validate.py:89-93` (`strip('.?, ').capitalize()`, `" and "`, `np.array(captions).T.flatten()`) — NONE — none.
53. **Tokenization context/truncation policy** — VERIFIED — `validate.py:93` `context_length=77`, no `truncate=True` (loud failure, not silent truncation) — NONE — none.
54. **Adapter command matches the audited invocation** — VERIFIED — OBSERVED argv from `CombinerNoftAdapter.command` vs `README.md:258-269` and `validate.py:263-277` — NONE — none.
55. **Adapter correctly omits `--clip-model-path` for noft** — VERIFIED — `:90-94` has no such flag; `:283-287` therefore skips the `"CLIP"` overlay; the sibling passes it (`:107-109`) — NONE — none.
56. **Combiner checkpoint path handling vs `checkpoint_missing_paths`** — VERIFIED — OBSERVED `checkpoint_missing_paths` (single-file branch `registry.py:292-295`) and the adapter address the same path; probed empty/absent/non-empty behaviour — NONE — none (the sibling's `bundle_placeholder` gap is M-9, not this model's).
57. **Sibling `bundle_placeholder` manifest is recorded** — VARIANT (defect) — OBSERVED `artifact_type='bundle_placeholder'`, `directory_members=[]`, `bundle_manifest_dgst=None` (`evaluate_models.py:321` covers `run_directory`/`bundle_directory` only); `_plan_members` `:991-1014`; `_members_live` `:657-658` returns True; `_bundle_members_live` `:725` never called — P2 — include `bundle_placeholder` in the manifest branch and call the verifier (M-9).
58. **Environment spec matches the pinned README** — VERIFIED — `models.yaml:688-695` vs `README.md:113-118` — NONE — none.
59. **`clip` dependency pinned** — BLOCKED — `README.md:118` installs CLIP from git main; `models.yaml:695` records the bare string — P2 — record an `openai/CLIP` commit sha (M-4).
60. **Model interpreter verified / command constructible** — BLOCKED — OBSERVED `workbench/envs/` = `README.md` + `csmcir-import.txt`; `WORKBENCH_CLIP4CIR_PYTHON` unset; OBSERVED `RuntimeError: execution environment not verified …` — P1 — provision and verify the env before any run.
61. **CUDA required?** — VERIFIED (no) — `utils.py:15-18` falls back to CPU; no unguarded `.cuda()` in this path — P3 — record `cuda` as not required (M-7).
62. **Registry status fields accurate** — VERIFIED — `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`:686`), `command_status: COMMAND_AUDITED` (`:687`), `strict_clean_checkpoint_available: null` (`:696`), `checkpoint_mapping_status: UNVERIFIED` (`:710`) all consistent — NONE — none.
63. **Registry comment about the staged root is true** — VARIANT (false as written) — `models.yaml:656-659`/`:712-715` claims the root is staged at `<checkout>/fashionIQ_dataset` and satisfies the evaluator; OBSERVED link absent and root `.png`-only — P2 — correct the comment alongside C-2 (M-3).
64. **Metric parser exists for this model** — BLOCKED — OBSERVED `AGGREGATE_PARSER_UNAVAILABLE`; dispatch `metrics_extraction.py:328-331` + fallback `:334-342` — P1 — add a parser for `validate.py:344-354` with macro recomputation (C-3).
65. **Parity check reachable once a parser exists** — VARIANT — `_check_parity` (`metrics_extraction.py:28-60`) routes through `paper_gate.compare_metric` with 5% **relative** and refuses a partial row; for this model `paper_metrics` is all-`null` (`models.yaml:683-685`) → `PARITY_NOT_EVALUATED` — P1 — supply a verified noft reference or accept "not evaluated" explicitly (C-1).
66. **Gate wired through the report** — VERIFIED — `evaluate_models.py:1398` `_benchmark_gate_verdict(plan, extraction, code)` → `:1425 "benchmark_gate"`; `autopilot.py:847-848`, `:964-965` carry `benchmark_gate` + `variant_provenance`; run_meta carries `caption_alias_staged`/`declared_variant` (`:146-147`); a deviation does not demote status (`paper_gate.py:373-375` returns `ALERT_METRIC_DEVIATION` with `passed=False`, no status mutation) — NONE — none.
67. **Committed reference for this model is gradeable** — BLOCKED — OBSERVED `select_reference(...)` → `(None, "…none is verified: ['UNVERIFIED']")` and `evaluate_gate(...)` → `METRIC_REFERENCE_UNVERIFIED` — P1 — C-1.
68. **Per-category comparison added and checked** — VERIFIED — `CATEGORY_KEYS` (`paper_gate.py:35-37`), loop `:349-363`, unchecked categories into `detail` `:382-384` — NONE — none.
69. **Output artifact exists** — NOT_RUN — no process, no `run_id`, no report, no per-query export; OBSERVED `workbench/artifacts/results/fashioniq_original_split/` contains only `mock/` — P1 — nothing to compare.
70. **Cross-protocol comparability guarded** — VERIFIED — `docs/FASHIONIQ_PROTOCOL_PARITY.md:9-22`; `resolve_fashioniq_protocol` (`src/evaluation/fashioniq.py:190-201`) raises rather than defaulting — NONE — none.
71. **Paper-level protocol defect surfaced rather than papered over** — VARIANT — CLIP4Cir's FashionIQ path is reference-eligible while DCNet (the baseline §4.2.1 says it follows) masks (`third_party/DCNet/trainer/trainer_joint.py`, per the sibling audit), and Table 9 juxtaposes them; the paper never states a gallery policy (my grep of the body text for `aggregat`/`macro`/`unweighted` → 0 hits; the committed protocol row's own `source_excerpt_or_description` records the silence) — P1 — belongs in the paper-level record, not as a workbench flip; do not compare `fashioniq_original_split` numbers against reference-excluded cohorts.
72. **Local validation split is the canonical one** — VARIANT — OBSERVED 3817/6346/5373 gallery and 2017/2038/1961 triplets = 15536/6016 vs the paper's printed 15537/6017 (§4.2.1 reproduced verbatim) — P3 — dataset-provenance question across all FashionIQ models; unresolved, reported UNKNOWN not PASS.
73. **Gallery duplicate-free and order-preserving** — VERIFIED — OBSERVED split files parsed; `data_utils.py:144-147` preserves file order; DataLoader `shuffle` default False (`utils.py:30-31`) — NONE — none.
74. **No dead/obsolete combiner shim** — VERIFIED — `adapters/models.py:86-94` contains only the noft command; no alias or re-export could silently route fullft→noft; `grep checkpoint.parent` → only the sibling comment and an unrelated CLI arg — NONE — none.
75. **Sibling entry not conflated** — VERIFIED — distinct `model_id` (`:711`), adapter (`:97-111`), status (`BLOCKED_CHECKPOINT_BUNDLE` `:747` vs `BLOCKED_CHECKPOINT_MAPPING` `:686`), checkpoint record (`:758-774` vs `:698-710`) — NONE — none.
76. **Verdict not upgraded past the evidence** — VERIFIED — every arrow either fails or is STATIC_AUDIT/OBSERVED read-only; no interpreter, no checkpoint, no link, no `.jpg` tree, no GPU run; `VERIFIED`/`VARIANT` for reproduction would be unsupported — NONE — none.

---

**FINAL VERDICT: BLOCKED**

Every static fact for `combiner_rn50x4_noft` checks out against the pinned source — the pin equals the
clean checkout HEAD (enforced by `source_clean_and_pinned`), `fashioniq_original_split` with an eligible
reference is confirmed inside `validate.py` against its own CIRR contrast, the TargetPad/CLIP pipeline and
the 2560/5120 Combiner dims match `data_utils.py:82-96` and `combiner.py:18-31` plus the paper's §4.1, the
caption composition is deterministic and un-augmented, and the adapter's command is the README's invocation
minus exactly the flag that defines noft; the checkpoint path handling is coherent with
`registry.checkpoint_missing_paths` (single non-empty regular file, the same path the adapter passes). What
is missing is everything needed to *produce* a number: the canonical root is `74381 png / 0 jpg` where the
evaluator opens `.jpg` and the drop is silent (`data_utils.py:182-183` + `utils.py:87`, C-2, P0), the
declared `<checkout>/fashionIQ_dataset` link is absent, the RN50x4 cache is absent, there is no verified
model interpreter, no parser (`AGGREGATE_PARSER_UNAVAILABLE`, C-3), no checkpoint bytes with any digest
(C-4), and no committed *verified* reference — the model is ungradeable even if it ran perfectly (C-1).
The noft-vs-fullft mapping is documented by the author and corroborated by the pinned loader, but nothing
in the checkpoint's bytes was observed, so the honest state is BLOCKED, not VERIFIED or VARIANT.

**Checks the main agent should run after any fix lands:** `python3 -m pytest workbench/tests -k
"combiner or clip4cir or replay or preflight"`; `python3 workbench/scripts/pc_readiness.py` (expect
`combiner_rn50x4_noft` to stop classifying as `BLOCKED_ENVIRONMENT` once an interpreter exists);
`python3 workbench/scripts/build_paper_registry.py` if a noft reference row is added (C-1) — and confirm
the regenerated `registry_version` rather than editing `papers.yaml` in place; a targeted assertion that
`guarded_plan` records `directory_members`/`bundle_manifest_digest` for `artifact_type:
bundle_placeholder` (M-9).
