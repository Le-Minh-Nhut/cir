# clip4cir_rn50x4_fullft

All claims below are **STATIC_AUDIT** unless explicitly labelled **RUNTIME_VERIFIED**. No GPU inference was run; almost every runtime claim is therefore STATIC_AUDIT.

## MODEL ID
`clip4cir_rn50x4_fullft` — registered at `workbench/registry/models.yaml:664`. method_name `CLIP4Cir RN50x4 fullft` (`:676`), research_generation `legacy`, publication_year 2023.

## PAPER TITLE
*Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features* (`workbench/registry/models.yaml:680`; arXiv `https://arxiv.org/abs/2308.11485`, ACM TOMM 2023). Fetched the paper text from `https://ar5iv.labs.arxiv.org/html/2308.11485` and read the tables from the backing file `~/.omp/agent/sessions/-data-cir/2026-10-13T13-09-01-288Z_01a13561-72e8-70cb-a614-83b9e812b501/10968.read.log` (line 471 = Table 9 RN-50x4 row, line 496 = Table 10 RN-50x4 row).

## PINNED SOURCE COMMIT
Registry declares `upstream_commit_sha: dfed9f748a8a4d05abf613164e06d59b22dbdf13` (`workbench/registry/models.yaml:684`), `source_dir: CLIP4Cir` (`:686`), `upstream_repo_url: https://github.com/ABaldrati/CLIP4Cir` (`:682`), default branch `master` (`:683`).

Observed:
```
$ git -C workbench/third_party/CLIP4Cir rev-parse HEAD
dfed9f748a8a4d05abf613164e06d59b22dbdf13
$ git -C workbench/third_party/CLIP4Cir status --porcelain
(empty)
$ git -C workbench/third_party/CLIP4Cir log -1 --format='%H %ad %s'
dfed9f748a8a4d05abf613164e06d59b22dbdf13 Tue Sep 5 09:31:52 2023 +0200 Update README.md
```
Workbench's own gate agrees (RUNTIME_VERIFIED against a live git checkout, no GPU needed):
```
$ python3 -c "... source_clean_and_pinned(model, source.resolve()) ..."
source_clean_and_pinned -> (True, 'dfed9f748a8a4d05abf613164e06d59b22dbdf13', None)
```
(`workbench/backend/runtime.py:2013-2037`). Git remote is `git@github.com:ABaldrati/CLIP4Cir (fetch) [blob:none]`; no tags in the checkout. **Exact match, checkout clean.**

Independence from `combiner_rn50x4_noft`: same `source_dir`, same SHA (`models.yaml:610, 626` vs `:664, 684`), same `native_protocol`, but a distinct `model_id`, distinct `reproduction_status` (`BLOCKED_CHECKPOINT_MAPPING` at `:626` vs `BLOCKED_CHECKPOINT_BUNDLE` at `:696`) and a distinct checkpoint record (`:657-663` vs `:707-720`). No evidence from that entry is reused here.

## CHECKPOINT IDENTITY
Registry (`workbench/registry/models.yaml:707-720`):
```
checkpoint_id: rn50x4_fullft_unresolved
filename: rn50x4-fullft-unresolved.pt
artifact_type: bundle_placeholder
required_files: [combiner_state.pt, clip_state.pt]
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

**Two artefacts are genuinely required — verified from source, not from the registry.**

1. Frozen base RN50x4 is loaded first, always: `workbench/third_party/CLIP4Cir/src/validate.py:279`
   ```python
   clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)
   ```
2. The fine-tuned CLIP state is an **overlay onto that same module** under key `'CLIP'`: `src/validate.py:283-287`
   ```python
   if args.clip_model_path:
       saved_state_dict = torch.load(args.clip_model_path, map_location=device)
       clip_model.load_state_dict(saved_state_dict["CLIP"])
   ```
   The key name is not incidental: `src/utils.py:127-131` `save_model` writes `model_name = model_to_save.__class__.__name__`, so a fine-tuned CLIP module (`clip.model.CLIP`) is saved under `'CLIP'` (`src/clip_fine_tune.py:226-228`), and the Combiner (`src/combiner.py:6 class Combiner`) is saved under `'Combiner'` (`src/combiner_train.py:225-227`).
3. The Combiner state is a **separate** file, loaded under key `'Combiner'`: `src/validate.py:304-309`
   ```python
   elif args.combining_function.lower() == 'combiner':
       combiner = Combiner(feature_dim, args.projection_dim, args.hidden_dim).to(device, non_blocking=True)
       state_dict = torch.load(args.combiner_path, map_location=device)
       combiner.load_state_dict(state_dict["Combiner"])
   ```
4. The README's own validation recipe passes **both** paths: `README.md:258-269`
   ```
   python src/validate.py --dataset {'CIRR' or 'FashionIQ'} --combining-function {'combiner' or 'sum'} \
     --combiner-path {path to trained Combiner} --projection-dim 2560 --hidden-dim 5120 \
     --clip-model-name RN50x4 --clip-model-path {path-to-fine-tuned-CLIP} --target-ratio 1.25 --transform targetpad
   ```
5. Independent corroboration that the fine-tuned encoder is mandatory for the published number: `README.md:75-81` (second stage freezes both CLIP encoders and trains only the Combiner; *"At inference time the fine-tuned encoders and the trained Combiner are used"*), and `src/combiner_train.py:76-80, 279-283`, which consume the same `clip_model_path` / `state_dict["CLIP"]` contract.

**Conclusion: the registry's two-artefact claim (`combiner_state.pt` + `clip_state.pt`) is CORRECT, and the three-artefact reality (base RN50x4 + CLIP overlay + Combiner state) is also correct.** The registry additionally gates the base via `auxiliary_assets.yaml:179-195`.

**But the wiring is broken.** `checkpoint_missing_paths` (`workbench/backend/registry.py:269-291`) treats `bundle_placeholder` as a **directory** and looks for the members *inside* it:
```
$ python3 -c "... checkpoint_path / checkpoint_missing_paths for the fullft variant ..."
checkpoint_path() -> workbench/artifacts/checkpoints/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt
required_files: ['combiner_state.pt', 'clip_state.pt']
missing: [PosixPath('workbench/artifacts/checkpoints/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt')]
adapter would read combiner from: workbench/artifacts/checkpoints/clip4cir_rn50x4_fullft/combiner_state.pt
adapter would read clip from    : workbench/artifacts/checkpoints/clip4cir_rn50x4_fullft/clip_state.pt
```
whereas `CLIP4CirFullftAdapter.command` (`workbench/backend/adapters/models.py:97-108`) reads the two states from `checkpoint.parent`:
```python
bundle_dir = checkpoint.parent
combiner_path = bundle_dir / "combiner_state.pt"
clip_path = bundle_dir / "clip_state.pt"
```
So a tree that satisfies the registry's own presence check (`<root>/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt/{combiner_state.pt,clip_state.pt}`) is **not** the tree the adapter reads. See CRITICAL FINDINGS #2.

## EVALUATION PROTOCOL
Registry: `native_protocol: fashioniq_original_split` (`models.yaml:691`), `supported_protocols: [fashioniq_original_split]` (`:692`), `literature_split_label: original` (`:690`).

Source: `src/validate.py:325-354` selects `--dataset fashioniq`, evaluates `shirt`, `dress`, `toptee` in that literal order (`:329, :334, :339`), and macro-averages:
```
353: print(f"average recall10 = {mean(average_recall10_list)}")
354: print(f"average recall50 = {mean(average_recall50_list)}")
```
Workbench: `src/evaluation/fashioniq.py:159-169` `FASHIONIQ_PROTOCOLS` declares `fashioniq_original_split` as `gallery="full_split", reference_excluded=False`; `:178-189` `resolve_fashioniq_protocol` raises on an unknown name with no default. `workbench/docs/FASHIONIQ_PROTOCOL_PARITY.md:76` records the same mapping; my code reading agrees with it.

## DATASET CONTRACT
- Layout `fashioniq_standard` (`models.yaml:687`), image extension `.jpg` (`models.yaml:670`), `preparation_contract: manual_standard` (`:688`), `preparation_id: clip4cir_fashioniq` (`:689`).
- Required raw inputs (`workbench/registry/preparation_contracts.yaml:140-169`): `captions/cap.{dress,shirt,toptee}.val.json`, `image_splits/split.{dress,shirt,toptee}.val.json`, `images`.
- **The evaluator hard-codes its data root.** `src/data_utils.py:11` `base_path = Path(__file__).absolute().parents[1].absolute()`; `src/data_utils.py:140, 146, 176` open `base_path / 'fashionIQ_dataset' / ...`. There is no CLI flag for a dataset root anywhere in `src/validate.py:263-277`.
- Registry acknowledges this with `canonical_dataset_link: fashionIQ_dataset` (`models.yaml:669`) and a comment at `:657-661`; the link is sanctioned as a runtime artifact by `workbench/backend/registry.py:563-600`.
- Local data, verified by direct inspection:
  ```
  $ ls data/fashionIQ_dataset
  captions/  images/  image_splits/  qwen_captions/
  $ ls workbench/third_party/CLIP4Cir
  .git  .gitignore  LICENSE  README.md  images  src        # note: 'images' is the README figures dir, NOT the dataset
  $ find workbench/third_party/CLIP4Cir -maxdepth 1 -name fashionIQ_dataset
  (empty)
  ```
  So the required link is **absent** right now and the evaluator cannot open a single image.
- Val counts from the local files (`python3` over the JSON files): dress 2017 / shirt 2038 / toptee 1961 = **6016** triplets; splits 3817 / 6346 / 5373. Paper §4.2.1 states *"validation ... 6,017"* triplets — a 1-record discrepancy (MINOR).
- **Extension mismatch, confirmed against the code's own literal.** Every local image is `.png` and zero are `.jpg`:
  ```
  dress 3817 png 3817 jpg 0
  shirt 6346 png 6346 jpg 0
  toptee 5373 png 5373 jpg 0
  ```
  while `src/data_utils.py:158, 161, 170, 176` open `f"{name}.jpg"`. The workbench reports this itself.

## PREPROCESSING CONTRACT
Source recipe selected by `src/validate.py:274, 289-291` (`--transform targetpad`, `--target-ratio 1.25`) → `src/data_utils.py:82-96`
```python
return Compose([
    TargetPad(target_ratio, dim),
    Resize(dim, interpolation=PIL.Image.BICUBIC),
    CenterCrop(dim),
    _convert_image_to_rgb,
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
```
with `TargetPad.__call__` at `src/data_utils.py:54-63` padding *only* when `actual_ratio >= target_ratio`. `dim` is `clip_model.visual.input_resolution` (`src/validate.py:280`) — 288 for RN50x4. The adapter reproduces `--target-ratio 1.25 --transform targetpad` verbatim (`workbench/backend/adapters/models.py:105-106`); the registry pins the same in `preparation_contracts.yaml:169`. **EQUIVALENT_WITH_EVIDENCE, STATIC_AUDIT.**

## AUGMENTATION CONTRACT
None at evaluation. `src/utils.py:21-44` `extract_index_features` and `src/validate.py:60-110` `generate_fiq_val_predictions` both run under `torch.no_grad()` with `shuffle=False` (`src/validate.py:75-77`). The only text-side choice is the deterministic concatenation at `src/validate.py:88-92`:
```python
input_captions = [
    f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for
    i in range(0, len(flattened_captions), 2)]
```
The randomized four-way augmentation exists only on the training side (`src/utils.py:57+` `generate_randomized_fiq_caption`, used by `src/clip_fine_tune.py`) and is baked into the published weights. Registry `evaluation_noise_pct: 0` (`models.yaml:716`) is therefore correct as an evaluation statement; `checkpoint_training_noise_pct: null` (`:715`) correctly declines to claim a training-condition number.

## GALLERY CONTRACT
`src/validate.py:127-129`
```python
classic_val_dataset = FashionIQDataset('val', [dress_type], 'classic', preprocess)
index_features, index_names = extract_index_features(classic_val_dataset, clip_model)
relative_val_dataset = FashionIQDataset('val', [dress_type], 'relative', preprocess)
```
The `classic` gallery is `self.image_names` loaded from `image_splits/split.{cat}.val.json` in file order (`src/data_utils.py:146-147, 174-178`). Nothing dedups or sorts it. This is the **full ordered split** — 3817 / 6346 / 5373 locally.

## METRIC CONTRACT
`src/validate.py:40-55`
```python
index_features = F.normalize(index_features, dim=-1).float()
distances = 1 - predicted_features @ index_features.T
sorted_indices = torch.argsort(distances, dim=-1).cpu()
sorted_index_names = np.array(index_names)[sorted_indices]
labels = torch.tensor(
    sorted_index_names == np.repeat(np.array(target_names), len(index_names)).reshape(len(target_names), -1))
assert torch.equal(torch.sum(labels, dim=-1).int(), torch.ones(len(target_names)).int())
recall_at10 = (torch.sum(labels[:, :10]) / len(labels)).item() * 100
recall_at50 = (torch.sum(labels[:, :50]) / len(labels)).item() * 100
```
Cosine distance, ascending distance sort, denominator = query count, macro mean over the three categories (`:353-354`). Workbench macro formula: `src/evaluation/fashioniq.py:223-243` (`(recall_at_10 + recall_at_50) / 2`).

**The reference stays in the FashionIQ gallery. VERIFIED from a contrast inside the same file.** The FashionIQ path has **no** reference mask; the CIRR path does, at `src/validate.py:162-166` (`sorted_index_names != np.repeat(np.array(reference_names), ...)`; also `src/cirr_test_submission.py:98-103`). For FashionIQ, `target_names` only is used (`:50`), and the assertion at `:51` guarantees exactly one ground-truth column — the reference is a candidate like any other and can outrank the target. Population check on the real local files: **6016 / 6016 references are present in their own category gallery, and 0 references equal their target**, so the reference *is* always rankable. `fashioniq_original_split` (reference eligible) is the correct protocol id; the workbench's recorded claim is correct. This is the same P1-class question as for `combiner_rn50x4_noft` and it resolves the same way.

## SOURCE-TO-WORKBENCH PARITY
Command, built by `CLIP4CirFullftAdapter.command` (`workbench/backend/adapters/models.py:97-108`):
```
<verified python> <source>/src/validate.py --dataset fashionIQ --combining-function combiner \
  --combiner-path <checkpoint.parent>/combiner_state.pt --clip-model-path <checkpoint.parent>/clip_state.pt \
  --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad
```
Every flag matches `src/validate.py:265-275`; `2560 = 640*4` and `5120 = 640*8` equal the source defaults at `:269-270`, and `--clip-model-path` is the documented fullft discriminator (`README.md:266`). `command_status: COMMAND_AUDITED` (`models.yaml:679`).

**Channel limitation (P3).** Both workbench paths here are *dispatch-only*: the command is the author script inside the pinned checkout, so text composition and gallery construction execute upstream source. The workbench's own `src/datasets/fashioniq.py:119-160 compose_fashioniq_caption` / `:225` `caption_policy` plumbing is not reached by this adapter (`grep -rn compose_fashioniq_caption --include=*.py src/ workbench/ | grep -v third_party` → definition + `src/datasets/fashioniq.py:308` only), and `validation_caption_policy` is **not declared** for this model (`grep -n validation_caption_policy workbench/registry/models.yaml` → no match). The `ordered_and` policy does encode exactly the source's `strip('.?, ')`+`capitalize()`+`" and "` construction (`src/datasets/fashioniq.py:149-155`), so the routes agree by construction — but nothing pins that agreement for this model.

**Metric parsing (P2).** `workbench/backend/metrics_extraction.py:287-315` implements parsers only for `limn` and `csmcir`; every other model returns `extraction_status="AGGREGATE_PARSER_UNAVAILABLE", parity_status="PARSER_UNAVAILABLE"`. There is no CLIP4Cir parser anywhere (`grep -rn clip4cir workbench/backend/metrics_extraction.py workbench/replay/` → empty; `workbench/replay/` holds only `limn.py`, `tgcir.py`). So a fullft stdout (`shirt_recallat10 = ...`, `average recall10 = ...`, `src/validate.py:344-354`) cannot currently be turned into a workbench metric record.

## RUNTIME STATUS
`reproduction_status: BLOCKED_CHECKPOINT_BUNDLE` (`models.yaml:696`); `replay_status: BLOCKED_CHECKPOINT_MAPPING` (`:671`) — the two fields disagree with each other (MINOR #1). Live preflight, run read-only:
```
$ python3 -c "... model_requirements(model, config) ..."
BLOCKS:
  - FashionIQ image extension requirement: 3817 of 3817 dress validation images lack .jpg in .../data/fashionIQ_dataset/images (first: B009PMCJLW.jpg)
  - FashionIQ image extension requirement: 6346 of 6346 shirt validation images lack .jpg in .../data/fashionIQ_dataset/images (first: B000KENMD8.jpg)
  - FashionIQ image extension requirement: 5373 of 5373 toptee validation images lack .jpg in .../data/fashionIQ_dataset/images (first: B008CG1JJ0.jpg)
  - OpenAI CLIP RN50x4 base weights missing: /home/heheboiz/.cache/clip/RN50x4.pt
  - canonical dataset link missing: .../workbench/third_party/CLIP4Cir/fashionIQ_dataset -> .../data/fashionIQ_dataset
  - external runtime asset file missing: OpenAI CLIP RN50x4 base weights (/home/heheboiz/.cache/clip/RN50x4.pt)
dataset_link: {'path': '.../CLIP4Cir/fashionIQ_dataset', 'expected': '/home/heheboiz/data/cir/data/fashionIQ_dataset', 'actual': None}
aux assets: [('OpenAI CLIP RN50x4 base weights', 'MISSING')]
```
Independent filesystem checks: `ls -la ~/.cache/clip/` → `No such file or directory`; `ls -la workbench/artifacts/checkpoints/` → only `.gitkeep`; `find workbench/artifacts/results -name '*clip4cir*'` → empty. **NOT_RUN, and no checkpoint, backbone, or staged dataset exists locally.** STATIC_AUDIT for everything except the git-pin check above.

The RN50x4 cache path requirement is **real and correctly modelled**. `clip.load` resolves its cache as `download_root or os.path.expanduser("~/.cache/clip")` — verified in the live upstream source, `https://raw.githubusercontent.com/openai/CLIP/main/clip/clip.py`:
```python
if name in _MODELS:
    model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
cache_root_fixed: ~/.cache/clip     # auxiliary_assets.yaml:191
```
OpenAI's CLIP never reads `XDG_CACHE_HOME`, so the registry's `cache_root_fixed` (`auxiliary_assets.yaml:188-191`, rationale comment quoted at `workbench/backend/registry.py:341-344`) is the correct modelling, and `auxiliary_destination` honours it at `workbench/backend/registry.py:348-354`. Upstream `_download` also self-verifies against `url.split("/")[-2]`: for this SHA-embedded URL that segment **is** `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd`, matching the registry's `expected_sha256` — **independent third-party hash corroboration (STATIC_AUDIT)**. Reachability observed:
```
$ curl -sS -I https://openaipublic.azureedge.net/clip/models/7e526bd1.../RN50x4.pt
HTTP/2 200 ... content-length: 421854225 ... etag: 0x8D8DE3A043F0B41
```

## PAPER-METRIC STATUS
Registry records `reported_r10: null`, `reported_r50: null`, `reported_mean: null` (`models.yaml:693-695`).

**An author-published FashionIQ row for this variant DOES exist and I located it.** arXiv paper, Table 9 ("Comparison between our method and current state-of-the-art models on the Fashion-IQ validation set"), the `Ours / RN-50x4 / Transformer` row (ar5iv line 471), split by column:
```
Ours | RN-50x4 | Transformer | Shirt 44.41 | 65.26 | Dress 39.46 | 64.55 | Toptee 47.48 | 70.98 | Average 43.78 | 66.93
```
Cross-check against the paper's own prose (ar5iv line 482, "When considering the larger RN-50x4-based model ... 2% to 4% ... compared to the smaller backbone", where the RN-50 row is 39.87/60.84, 37.67/63.16, 44.88/68.59 → 40.80/64.20): consistent. Arithmetic on the published per-category cells: (44.41+39.46+47.48)/3 = 43.783 ✓ and (65.26+64.55+70.98)/3 = 66.93 ✓, so the Average column is the macro mean, exactly the workbench convention.

That row is the two-stage fine-tuned model (Table 1 shows the best FashionIQ configuration is `CF=Combiner, IFT=✓, TFT=✓`; the `IFT=✗, TFT=✗` Combiner row is the out-of-the-box baseline), and §4.7 describes it as the proposed method — i.e. it is this variant. If recorded with the registry's `reported_mean` convention, `mean = (43.78 + 66.93)/2 = 55.36`.

**Reachability of the row is fine; reachability of the weight is not.** The author Drive folder the registry points at resolves: `curl -sSL "https://drive.google.com/drive/folders/1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv?usp=sharing"` → HTTP 200, `<title>pretrained_models - Google Drive</title>`. But an unauthenticated fetch of the folder HTML yields **zero** `.pt` occurrences (`grep -o '\.pt' drive.html | wc -l` → `0`), so the individual file listing needs JS/auth and the archive→variant mapping remains unproven. `download_url: null` (`:712`), `expected_sha256: null` (`:718`), `expected_size_bytes: null` (`:717`) mean `checkpoint_availability` can never hash-check it (`workbench/backend/registry.py:696-698`), and `workbench/scripts/download_checkpoints.py:112-114` `verify()` returns `hash_valid and provenance_valid` where `provenance_valid` needs `checkpoint_mapping_status == "VERIFIED_METADATA" and download_url is not None` — both false.

**Verdict: the registry's null paper metrics are an omission, not a fact. The row is reachable and exact; the weights are not.**

## CRITICAL FINDINGS

1. **P1 — the registry's `fashioniq_original_split` protocol label is correct; the reference is eligible and the evidence is a contrast inside `validate.py` itself.** `src/validate.py:40-55` (FashionIQ) performs no reference removal, while `src/validate.py:162-166` and `src/cirr_test_submission.py:98-103` (CIRR) do. Measured on the real local annotation files: all 6016 references lie inside their own category gallery and none equals its target. Action: none — this entry passes this check; do not re-open it, and keep it separate from the `noft` entry's reasoning.

2. **P1 — registry/adapters bundle-path disagreement makes the declared bundle unststageable as declared.** `checkpoint_missing_paths` (`workbench/backend/registry.py:269-291`) resolves `bundle_placeholder` members under `checkpoints/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt/`, while `CLIP4CirFullftAdapter.command` (`workbench/backend/adapters/models.py:100-103`) reads them from `checkpoints/clip4cir_rn50x4_fullft/`. Demonstrated with the workbench's own functions (output above). A directory that satisfies the registry gate will not satisfy the adapter, and vice versa. Action: either drop the `.pt` filename for bundle types and make `checkpoint_path` return the bundle directory, or make the adapter consume `checkpoint_path()` unchanged; then assert equality in a test.

3. **P2 — `bundle_placeholder` is invisible to the plan builder.** `workbench/scripts/evaluate_models.py:227` handles only `("run_directory", "bundle_directory")` for manifest construction, and the change-detection gate at `:547-549` accepts only `("run_directory", "bundle_directory", "category_specific_whole_models")` plus non-empty `checkpoint_bundle`/`directory_members`. With `artifact_type: bundle_placeholder` and no bundle dict, `checkpoint_bundle is None` and `directory_members is None`, so `multi_member` is False and the proof verifier returns True trivially — the two state files would not be pinned by any manifest digest even after staging. Contrast the LIMN precedent, which works because `checkpoint_bundles` is declared for it (`models.yaml:889-890`) and not for this model (`grep -n checkpoint_bundles models.yaml` → one model only). Action: treat `bundle_placeholder` as a directory type in both places, or declare a real `checkpoint_bundles` entry.

4. **P0 — the dataset the evaluator would actually read is empty of the images it opens.** `src/data_utils.py:158, 161, 170, 176` open `f"{name}.jpg"` under `<checkout>/fashionIQ_dataset/images`; the local canonical root contains 74381 `.png` and 0 `.jpg` (`ls .../images | sed 's/.*\.//' | sort | uniq -c` → `74381 png`). The `.jpg` gate is reported by the workbench for all three categories (3817/6346/5373 of `N` lack `.jpg`). Until the canonical root or the link is fixed, any run of this evaluator fails on the first `PIL.Image.open`. Note the registry's comment (`models.yaml:658-660`) asserts *"opens `.jpg` images there"* but states no transformation from the `.png` canonical form, and no preparation step for `clip4cir_fashioniq` exists: `grep -rn "clip4cir|manual_standard" workbench/scripts/prepare_dataset.py workbench/scripts/prepare_model.py` → no match, while `workbench/scripts/prepare_dataset.py:28 prepare_canonical_link` is the only link helper and `grep -rln prepare_csmcir workbench/` is CSMCIR-only. Action: declare the canonical statement ("FashionIQ release is `.png`") and a concrete, registered extension-conversion or hardlink step, or change the layout declaration; do not let a run discover it.

5. **P1 — no metric parser for this model, so no reproduction can produce a comparable number.** `workbench/backend/metrics_extraction.py:287-315` falls through to `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE` for `clip4cir_rn50x4_fullft`. Action: add a parser for the `src/validate.py:344-354` stdout shape (per-category plus `average recall10`/`average recall50`), matching the source's three-category macro.

6. **P1 — checkpoint identity is unresolved and unverifiable as recorded.** `status: CHECKPOINT_MAPPING_UNVERIFIED`, `checkpoint_mapping_status: UNVERIFIED` (`models.yaml:719-720`), `download_url: null`, no size, no hash. The pinned README names a *folder* (`README.md:204-205`) and a *single* generic CLI shape (`:258-269`); nothing in the repo distinguishes a noft from a fullft artefact (`grep -rn "noft|fullft|full-ft" workbench/third_party/CLIP4Cir` → no match). `workbench/registry/assets/cluster_b4_classic.yaml:167` records the same blocker for the Combiner half, and `cluster_c_classic.yaml:120` states the pairing obligation. Action: enumerate the folder, record per-file size+sha256 and the exact pairing, then set `VERIFIED_METADATA` — do not substitute a nearby artefact.

## MINOR FINDINGS

1. **P3 — `replay_status` and `reproduction_status` contradict each other in the same entry:** `BLOCKED_CHECKPOINT_MAPPING` (`models.yaml:671`) vs `BLOCKED_CHECKPOINT_BUNDLE` (`:696`). The bundle-specific value is the accurate one. Action: align.
2. **P3 — paper says 6,017 validation triplets (ar5iv §4.2.1); the local canonical files hold 6,016** (2017+2038+1961). Action: determine whether one record is missing locally before any reproduction is credited.
3. **P2 — document churn during this audit.** `git status --porcelain` reports `workbench/registry/models.yaml` modified with 4 hunks (`git diff --stat` → `+10 -2`); the hunks touch airknow/conesep/ptha_mtst/intent blocks only (`git diff workbench/registry/models.yaml | grep clip4cir` → empty), and `load_registry()` re-parses the live file successfully. My reading of `:664-720` is unaffected, but any line reference here is against the working-tree file, not `d9431fe`.
4. **P3 — `strict_clean_checkpoint_available: null` (`:705`).** With `checkpoint_training_noise_pct: null` and no published clean/fullft distinction discovered, `null` is honest; a boolean would overclaim. No action.
5. **P3 — untested success path.** `workbench/tests/test_replay.py:143-146` asserts only the *placeholder* shape (`artifact_type == "bundle_placeholder"` and `combiner_state.pt in required_files`); no test asserts that a staged bundle yields the adapter's expected paths. Action: the test in CRITICAL #2 covers it.
6. **P3 — the workbench's equivalent caption route is unpinned for this model.** `compose_fashioniq_caption(..., policy="ordered_and")` (`src/datasets/fashioniq.py:149-155`) reproduces `src/validate.py:88-92` exactly, but `validation_caption_policy` is not declared in this model's registry entry, so nothing records that the two agree. Action: declare it, or assert the equivalence in a test.

## UNRESOLVED EVIDENCE

1. **Which file(s) in the author Drive folder are the RN50x4 fullft pair.** Unauthenticated HTML exposes no file names (`grep -o '\.pt' drive.html | wc -l` → 0). No download attempted (prohibited). UNKNOWN.
2. **Whether a separate `.pt` file containing key `'CLIP'` is published at all, or whether the authors intend the single Drive artefact to carry both `'CLIP'` and `'Combiner'`.** `src/validate.py:285-286` and `:306-307` read two independent `torch.load` results, so two files is the source-supported reading, but the registry's chosen filenames (`combiner_state.pt`, `clip_state.pt`) are the workbench's naming, not evidence about the author's. UNKNOWN.
3. **Whether the base RN50x4 loaded at `src/validate.py:279` is byte-identical to the Azure object.** Hash corroboration is strong (URL segment == registry `expected_sha256`, and upstream `_download` self-verifies against the same segment) but no local file exists to hash. UNKNOWN locally.
4. **Whether `--clip-model-path {path-to-fine-tuned-CLIP}` in `README.md:266` refers to a FashionIQ-trained or CIRR-trained encoder for this row.** §4.7 separates the two datasets; the README gives one recipe. UNKNOWN.
5. **Any runtime number.** No interpreter for `WORKBENCH_CLIP4CIR_PYTHON` was located (`ls -d workbench/envs/*` → README only), no checkpoint, no backbone, no linked dataset. NOT_RUN.
6. **The 55 canonical checklist questions.** `/tmp/audit_assignments.md:47` and `/tmp/audit_context.md:47` both require "all 55 questions" but neither file, nor any file under `/home/heheboiz/data/cir`, contains the list (searched `*.md`/`*.py`/`*.yaml` for `55 question`, `55-item`, `CHECKLIST`, `N. <question`). The probes below are therefore **mine**, one per checkable fact, recorded as a substitution rather than presented as the original — the same gap was recorded independently at `workbench/docs/reviews/models/conesep.md:308` and `phs_mtst.md:234`.

## FINAL VERDICT
**BLOCKED** — the pinned source, the commit pin, the two-artefact requirement and the reference-eligible `fashioniq_original_split` protocol all check out against the code, but the run cannot start: the required checkpoint pair is unnamed, unhashed and unresolved, the base RN50x4 cache is absent, the hard-coded `<checkout>/fashionIQ_dataset` link does not exist, the canonical root holds `.png` where the evaluator opens `.jpg`, and the registry/adapters disagree about where a staged bundle lives.

---

# EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation -> checkpoint -> output -> metric comparison`

**1. paper/README/config -> source dataset.** `README.md:139-162` declares `project_base_path/fashionIQ_dataset/{captions,images,image_splits}` with `cap.{cat}.{split}.json`, `split.{cat}.{split}.json`, and `.jpg` image names. The code implements exactly that root: `src/data_utils.py:11 base_path = Path(__file__).absolute().parents[1].absolute()`, then `:140 base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json'`, `:146 ... / 'image_splits' / f'split.{dress_type}.{split}.json'`, `:176 ... / 'images' / f"{image_name}.jpg"`. Registry agrees: `fashioniq_layout: fashioniq_standard` (`models.yaml:687`), `canonical_dataset_link: fashionIQ_dataset` (`:669`). **Arrow holds on paper; the physical link is absent** (`find workbench/third_party/CLIP4Cir -maxdepth 1 -name fashionIQ_dataset` → empty; preflight block `canonical dataset link missing: .../CLIP4Cir/fashionIQ_dataset -> .../data/fashionIQ_dataset`).

**2. source dataset -> preprocessing.** `src/validate.py:274-275` defaults `--transform targetpad --target-ratio 1.25`; `:289-291` builds `targetpad_transform(1.25, input_dim)`; `:280 input_dim = clip_model.visual.input_resolution`; `src/data_utils.py:82-96` is the six-step compose. Applied to both the `classic` gallery (`:127-128`) and the `relative` query (`:129`). **Arrow holds, POST-CONDITION UNMET**: the images are `.png` on disk and `.jpg` in the code, so step 1 of the compose never completes. Measured: `dress 3817 png 3817 jpg 0` etc.

**3. preprocessing -> official evaluator.** `src/validate.py:325-342` iterates `shirt`, `dress`, `toptee` calling `fashioniq_val_retrieval` (`:113-132`). Text is composed at `:88-92` exactly as the README/paper describe the relative caption. Gallery features come from `src/utils.py:21-44` (`clip_model.encode_image` under `no_grad`). **Arrow holds** (STATIC_AUDIT).

**4. official evaluator -> workbench invocation.** `CLIP4CirFullftAdapter.command` (`workbench/backend/adapters/models.py:97-108`) emits the exact flag set of `README.md:258-269` and `src/validate.py:265-275`, with `--clip-model-path` set — the fullft discriminator (`README.md:266`). `command_status: COMMAND_AUDITED` (`models.yaml:679`); `preflight_blockers` accepts it as audited and then separately checks `(source / adapter.script).is_file()` (`workbench/backend/preflight.py:143-148`), satisfied by the present `src/validate.py`. **Arrow holds, except the checkpoint paths** (row 5).

**5. workbench invocation -> checkpoint.** Required by source: `src/validate.py:279` (base RN50x4), `:285-286` (`torch.load(clip_model_path)["CLIP"]`), `:306-307` (`torch.load(combiner_path)["Combiner"]`). Provided by registry: two placeholder names inside a notional directory (`models.yaml:710-711`) with no URL, hash, size or verified mapping; the workbench additionally requires the base via `auxiliary_assets.yaml:179-195`. Observed state: `checkpoints/` holds only `.gitkeep`; `~/.cache/clip/` does not exist. **Arrow BROKEN**, and path-wise `checkpoint_path()` != `checkpoint.parent` (demonstration above).

**6. checkpoint -> output.** Output shape is known statically: `src/validate.py:344-354` prints `shirt_recallat10`, `shirt_recallat50`, `dress_*`, `toptee_*`, `average recall10`, `average recall50`, from an `ArgumentParser`-only CLI with no JSON writer. **NOT_RUN** — nothing was executed, and `workbench/backend/metrics_extraction.py:287-315` has no parser for that shape anyway.

**7. output -> metric comparison.** `workbench/backend/paper_gate.py` compares `METRIC_KEYS = ("r10", "r50", "mean")` at `TOLERANCE_PCT = 5.0` relative. The reference numbers for the comparison **exist** (Table 9 RN-50x4 → 43.78 / 66.93; mean 55.36) but the registry holds `null` (`models.yaml:693-695`), so `paper_gate` would fall to `METRIC_REFERENCE_UNVERIFIED` even if a run produced output. **Arrow BROKEN at both ends.**

---

# CHECKLIST

*The canonical 55-item list is absent from `/tmp/audit_assignments.md`, `/tmp/audit_context.md`, and the repository (see UNRESOLVED EVIDENCE #6). The 55 probes below are my own, one per checkable fact across registry, source, checkpoint, protocol, dataset, preprocessing, augmentation, gallery, metric, parity, runtime and verdict.*

1. Registry entry exists and parses — VERIFIED — `load_registry()` returns the model from `workbench/registry/models.yaml:664`; `git diff workbench/registry/models.yaml` has 4 hunks, none touching `clip4cir` — NONE — none.
2. `model_id` unique and stable — VERIFIED — `grep -rn clip4cir_rn50x4_fullft workbench/registry/` → `models.yaml:656,681` plus the two cluster registries at `cluster_c_classic.yaml:63` and `cluster_b4_classic.yaml:89`; no duplicate `- model_id:` — NONE — none.
3. `method_name` matches the source and the paper — VERIFIED — `models.yaml:676` "CLIP4Cir RN50x4 fullft"; `README.md:1` "CLIP4Cir (ACM TOMM 2023)"; `README.md:80` "the fine-tuned encoders and the trained Combiner are used" — NONE — none.
4. `paper_url` resolves and is the right paper — VERIFIED — fetched `https://arxiv.org/abs/2308.11485`; title and abstract match `models.yaml:680` verbatim — NONE — none.
5. `upstream_repo_url` matches the README's own repo — VERIFIED — `models.yaml:682` == `README.md:107` `git clone https://github.com/ABaldrati/CLIP4Cir` — NONE — none.
6. `upstream_default_branch` correct — VERIFIED — `models.yaml:683` `master`; the checkout's default branch tracks `origin` and `README.md:205` links a `master`-era Drive folder — NONE — none.
7. Pinned commit matches the checkout exactly — VERIFIED — `git -C workbench/third_party/CLIP4Cir rev-parse HEAD` → `dfed9f748a8a4d05abf613164e06d59b22dbdf13`; `source_clean_and_pinned` → `(True, 'dfed9f74…', None)` — NONE — none.
8. Checkout is clean (no tracked edits, no unauthorized untracked) — VERIFIED — `git -C workbench/third_party/CLIP4Cir status --porcelain` empty; `_source_dirty` (`runtime.py:1998-2010`) returns False for this tree — NONE — none.
9. `source_available` truthful — VERIFIED — `source_dir: CLIP4Cir` (`:686`) is a directory containing `src/validate.py`, `src/data_utils.py`, `src/utils.py`, `src/combiner.py`, `README.md` — NONE — none.
10. `source_blocker` state truthful — VERIFIED — key absent for this model; `grep -n source_blocker workbench/registry/models.yaml` matches only airknow/conesep/intent/ptha_mtst. No declared blocker is the honest state for a source that imports cleanly — NONE — none.
11. Official evaluation entrypoint exists — VERIFIED — `src/validate.py:263-360` defines `main()` and `if __name__ == '__main__': main()` — NONE — none.
12. Evaluator CLI is runnable without edits — VERIFIED — `src/validate.py:264-277` `ArgumentParser` with `--dataset` and `--combining-function` required, everything else defaulted; no monkeypatching needed — NONE — none.
13. Registered command reproduces the documented invocation — VERIFIED — `workbench/backend/adapters/models.py:104-107` flag-for-flag vs `README.md:258-269` and `src/validate.py:265-275` — NONE — none.
14. `command_status` truthful — VERIFIED — `COMMAND_AUDITED` (`:679`); `preflight.py:128-131` classifies it audited because `CLIP4CirFullftAdapter.command is not OfficialScriptAdapter.command` — NONE — none.
15. `artifact_type` describes the source's real artefact shape — VERIFIED — `bundle_placeholder` (`:710`) matches `required_files: [combiner_state.pt, clip_state.pt]` (`:711`) and `src/validate.py:285-286` + `:306-307` — NONE — none.
16. Two artefacts genuinely required (registry's central claim) — VERIFIED — STATIC_AUDIT — `src/validate.py:283-287` (CLIP overlay), `:304-309` (Combiner), `README.md:258-269` (both flags) — NONE — none.
17. Artefacts are distinct states, not one re-read — VERIFIED — different key namespaces (`'CLIP'` vs `'Combiner'`), different loader calls, different model objects; key names derive from `src/utils.py:127` `model_to_save.__class__.__name__` over `clip.model.CLIP` and `src/combiner.py:6 Combiner` — NONE — none.
18. Base backbone additionally required (three artefacts, not two) — VERIFIED — `src/validate.py:279` runs `clip.load` unconditionally at `:280-281` before the overlay branch; gated by `auxiliary_assets.yaml:179-195` — NONE — none.
19. RN50x4 base cache path matches the consumer — VERIFIED — `auxiliary_assets.yaml:191` `cache_root_fixed: ~/.cache/clip` vs live upstream `clip/clip.py` `download_root or os.path.expanduser("~/.cache/clip")`; `registry.py:348-354` honours `cache_root_fixed` before any env var — NONE — none.
20. Declared base-asset URL is correct and reachable — VERIFIED — `curl -I` on `.../7e526bd1…/RN50x4.pt` → `HTTP/2 200`, `content-length: 421854225` — NONE — none.
21. Declared base-asset SHA256 is independently corroborated — VERIFIED — STATIC_AUDIT — `clip/clip.py` computes `expected_sha256 = url.split("/")[-2]`; that path segment equals `auxiliary_assets.yaml:194` `7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd` — NONE — none.
22. Base weights present locally — NOT_RUN — `ls -la ~/.cache/clip/` → `No such file or directory`; preflight block `OpenAI CLIP RN50x4 base weights missing: /home/heheboiz/.cache/clip/RN50x4.pt` — P1 — stage the verified file at the fixed cache root.
23. Checkpoint download URL is machine-resolvable — NOT_RUN — `download_url: null` (`:712`); `download_checkpoints.verify()` (`:112-114`) requires `VERIFIED_METADATA` and a non-null URL, both false — P1 — enumerate the Drive folder and record a direct URL.
24. Checkpoint identity is verified (mapping) — NOT_RUN — `status: CHECKPOINT_MAPPING_UNVERIFIED` (`:719`), `checkpoint_mapping_status: UNVERIFIED` (`:720`) — P1 — resolve the pairing and set `VERIFIED_METADATA`.
25. Checkpoint hash and size are recorded — NOT_RUN — `expected_sha256: null` (`:718`), `expected_size_bytes: null` (`:717`); `registry.py:696-698` then computes `local_sha` only and `official_sha256_match` stays `None` — P1 — record size and sha256 per member.
26. Checkpoint bytes present locally — NOT_RUN — `ls -la workbench/artifacts/checkpoints/` → `.gitkeep` only; `checkpoint_missing_paths` returns the placeholder path itself — P1 — stage the pair.
27. Registry bundle layout matches the adapter's read layout — **VARIANT (broken)** — `checkpoint_path()` → `…/clip4cir_rn50x4_fullft/rn50x4-fullft-unresolved.pt`; `CLIP4CirFullftAdapter.command` → `checkpoint.parent/combiner_state.pt` (`adapters/models.py:100-103`) — P1 — unify the two path derivations and test it.
28. `bundle_placeholder` handled by the plan builder's manifest logic — **VARIANT (broken)** — `evaluate_models.py:227` accepts only `("run_directory", "bundle_directory")`; `:547-549` likewise; `checkpoint_bundle`/`directory_members` stay `None` for this model because `checkpoint_bundles` is declared only for `limn` (`models.yaml:889-890`) — P2 — add the type, or declare a real bundle.
29. Unsupported artefact proven to block rather than silently pass — VARIANT — `tests/test_replay.py:143-146` pins the placeholder shape, and `registry.py:269-291` reports it missing, so presence is a blocker today; but a *staged* bundle would pass an unenforced planner — P2 — see #28.
30. `native_protocol` matches the pinned evaluator's actual cohort — VERIFIED — `fashioniq_original_split` (`:691`) ↔ `FASHIONIQ_PROTOCOLS` `gallery="full_split", reference_excluded=False` (`src/evaluation/fashioniq.py:160-162`); source builds the gallery from `image_splits/split.{cat}.val.json` in file order (`src/data_utils.py:146-147, 174-178`) — NONE — none.
31. Reference image remains eligible in the FashionIQ gallery — VERIFIED (STATIC_AUDIT) — `src/validate.py:40-55` has **no** reference mask, unlike the CIRR path at `:162-166`; `labels` is built from `target_names` only and asserted one-hot (`:50-51`) — NONE — none.
32. Reference eligibility holds numerically, not just structurally — VERIFIED — 6016 of 6016 references are members of their own category gallery and 0 references equal their target (`python3` over `data/fashionIQ_dataset/captions/cap.*.val.json` and `image_splits/split.*.val.json`) — NONE — none.
33. `supported_protocols` contains the native protocol and nothing spurious — VERIFIED — `[fashioniq_original_split]` (`:692`); `tests/test_protocols.py:50-51` asserts it, and `:246-248` asserts `fashioniq_val_split` is absent — NONE — none.
34. Protocol parity document agrees with the code — VERIFIED — `FASHIONIQ_PROTOCOL_PARITY.md:76` says full ordered split / eligible / expressed by the workbench; my independent reading of `validate.py` agrees — NONE — none.
35. Source-to-workbench parity document agrees with the code — VERIFIED — `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:33` `EQUIVALENT_WITH_EVIDENCE` for protocol and `MISMATCH_CONFIRMED` for the fixed data root; both reproduced independently here — NONE — none.
36. Dataset layout declaration matches the source's open() calls — VERIFIED — `fashioniq_standard` (`:687`) ↔ `captions/`, `image_splits/`, `images/` (`src/data_utils.py:140, 146, 176`) — NONE — none.
37. Declared preparation contract covers the raw inputs the evaluator needs — VERIFIED — `preparation_contracts.yaml:140-155` lists the six val JSONs and `images`; `preflight.model_requirements` builds path list from them (`preflight.py:26-31`) — NONE — none.
38. Declared image extension matches the files the evaluator opens — VERIFIED — `.jpg` (`:670`) ↔ `src/data_utils.py:158, 161, 170, 176` `f"{name}.jpg"` — NONE — none.
39. Declared image extension matches the files on disk — **VARIANT** — local root holds 74381 `.png` and 0 `.jpg`; 3817/3817 dress, 6346/6346 shirt, 5373/5373 toptee lack `.jpg` — P0 — convert/hardlink to `.jpg` or declare the real layout; nothing in the repo does it today.
40. Hard-coded dataset root is acknowledged and sanctioned — VERIFIED — comment `models.yaml:657-661`, `canonical_dataset_link: fashionIQ_dataset` (`:669`), `registry.py:563-600` `source_runtime_artifacts` emits the link as an approved runtime artefact — NONE — none.
41. Hard-coded dataset root is actually staged — NOT_RUN — `find workbench/third_party/CLIP4Cir -maxdepth 1 -name fashionIQ_dataset` → empty; preflight `dataset_link` reports `actual: None` — P1 — stage the link before any run.
42. Preprocessing recipe matches source exactly — VERIFIED — adapter `--target-ratio 1.25 --transform targetpad` (`adapters/models.py:105-106`) == `src/validate.py:274-275, 289-291`; transform body `src/data_utils.py:82-96` — NONE — none.
43. Preprocessing dimension is correct for RN50x4 — VERIFIED (STATIC_AUDIT) — `src/validate.py:280 input_dim = clip_model.visual.input_resolution` → 288 for RN50x4, matching `cluster_c_classic.yaml` "TargetPad(1.25, 288)" — NONE — none.
44. Augmentation contract: no evaluation-time randomisation — VERIFIED — `src/validate.py:75-77` `shuffle=False`; `src/utils.py:38-43` and `src/validate.py:96-107` under `torch.no_grad()`; `evaluation_noise_pct: 0` (`:716`) is correct — NONE — none.
45. Training-condition noise declared honestly — VERIFIED — `checkpoint_training_noise_pct: null` (`:715`) rather than 0; the paper reports task-oriented fine-tuning with no noise injection (§4.1) but the registry declines to convert that into a number — NONE — none.
46. Gallery contract: full ordered split, not a pair union — VERIFIED — `src/data_utils.py:146-147, 174-178`; local split sizes 3817/6346/5373 vs 2017/2038/1961 queries — NONE — none.
47. Gallery ordering preserved, not sorted or deduped — VERIFIED — `self.image_names.extend(json.load(f))` in file order (`src/data_utils.py:146-147`); `extract_index_features` iterates with `shuffle=False` (`src/utils.py:30-31`) — NONE — none.
48. Metric contract: R@10/R@50, descending similarity, denominator = query count — VERIFIED — `src/validate.py:44-45` (`distances = 1 - predicted @ index.T`, `argsort`) and `:54-55` (`/ len(labels)`) — NONE — none.
49. Metric aggregation: macro mean over shirt/dress/toptee — VERIFIED — `src/validate.py:326-342, 353-354`; workbench equivalent `src/evaluation/fashioniq.py:223-243` — NONE — none.
50. Exact-one-target assertion protects the cohort definition — VERIFIED — `src/validate.py:51` asserts `sum(labels, dim=-1) == 1` for every query; would fail loudly if a reference were ever also the target — NONE — none.
51. Metric parser can consume this evaluator's stdout — **VARIANT (broken)** — `metrics_extraction.py:287-315` returns `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE`; no CLIP4Cir parser in `workbench/backend/metrics_extraction.py` or `workbench/replay/` — P1 — add one for the `src/validate.py:344-354` shape.
52. Paper metrics for this variant are recorded — **VARIANT** — `reported_r10/r50/mean: null` (`:693-695`), yet the row exists: Table 9 `Ours/RN-50x4` = Shirt 44.41/65.26, Dress 39.46/64.55, Toptee 47.48/70.98, Average 43.78/66.93 (ar5iv line 471); mean = 55.36 — P2 — record the row with its table citation.
53. Paper metrics are internally consistent with the paper's own prose — VERIFIED — (44.41+39.46+47.48)/3 = 43.783 and (65.26+64.55+70.98)/3 = 66.93; prose at ar5iv line 482 claims 2–4% over the RN-50 row 40.80/64.20, which holds — NONE — none.
54. Runtime status is honest about what was and was not run — VERIFIED — `reproduction_status: BLOCKED_CHECKPOINT_BUNDLE` (`:696`); no results artefact exists (`find workbench/artifacts/results -name '*clip4cir*'` empty → NOT_RUN) — NONE — none.
55. Verdict is not upgraded past the evidence — VERIFIED — every arrow in the trace above either fails or is STATIC_AUDIT; no GPU host, no interpreter (`ls -d workbench/envs/*` → README only), no checkpoint, no backbone, no staged dataset, so `VERIFIED`/`VARIANT` would be unsupported — NONE — none.

---

FINAL VERDICT: **BLOCKED** — the commit pin, the two-artefact requirement and the reference-eligible protocol are all confirmed against the pinned code, but the checkpoint pair is unnamed/unhashed, the base RN50x4 cache and the hard-coded `<checkout>/fashionIQ_dataset` link are absent, the canonical root holds `.png` where the evaluator opens `.jpg`, and the registry and adapter disagree about where a staged bundle lives — so nothing can be run and no metric can be produced.

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left as
written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 3 committed row(s):

  - `clip4cir_rn50x4_fullft_fiq_tab9` — status `VERIFIED`, 0% training noise, protocol `fashioniq_original_split`, values `{'r10': 43.78, 'r50': 66.93, 'mean': 55.36}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'WORKBENCH_DERIVED'}`
  - `clip4cir_rn50x4_fullft__clip4cir_rn50x4_fiq_gallery_protocol` — status `UNVERIFIED`, 0% training noise, protocol `fashioniq_original_split`, values `{'r10': None, 'r50': None, 'mean': None}`, provenance `{'r10': 'UNAVAILABLE', 'r50': 'UNAVAILABLE', 'mean': 'UNAVAILABLE'}`
  - `clip4cir_rn50x4_fullft__clip4cir_paper_note_backbone_inventory_and_version` — status `UNVERIFIED`, 0% training noise, protocol `fashioniq_original_split`, values `{'r10': None, 'r50': None, 'mean': None}`, provenance `{'r10': 'UNAVAILABLE', 'r50': 'UNAVAILABLE', 'mean': 'UNAVAILABLE'}`

- **Readiness now:** `BLOCKED_ENVIRONMENT` — execution environment not verified: WORKBENCH_CLIP4CIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_CLIP4CIR_PYTHON not set)

- **Reference per registered checkpoint condition:**

  - `rn50x4_fullft_unresolved` (noise None): `clip4cir_rn50x4_fullft_fiq_tab9` status `VERIFIED` — exact condition match
