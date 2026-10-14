# combiner_rn50x4_noft

Independent adversarial, READ-ONLY audit. Nothing under `/home/heheboiz/data/cir` was written or modified; scratch only in `/tmp`. Every claim is `STATIC_AUDIT` unless explicitly marked `OBSERVED` (a command I ran and its output, often a live network read) or `RUNTIME_VERIFIED`. **No GPU inference, no evaluator execution, no checkpoint load happened.** Scratch files created only in `/tmp` (`/tmp/drive*.html`, `/tmp/dz_*.html`, `/tmp/drv_readme.bin`, `/tmp/clip4cir_paper.*`, `/tmp/cvpr22.*`, `/tmp/clip2103.*`).

---

## MODEL ID

`combiner_rn50x4_noft` / method_name `Combiner RN50x4 noft` (`workbench/registry/models.yaml:600`, `:612`).
- Adapter: `workbench/backend/adapters/models.py:86-94` — `CombinerNoftAdapter`, `script = "src/validate.py"`, with a real `command()` override.
- `source_dir: CLIP4Cir`, `source_available: true` (`models.yaml:620-621`), `research_generation: legacy`, `publication_year: 2023` (`:613-614`).
- Independence from its sibling: `clip4cir_rn50x4_fullft` is a **distinct** `model_id` (`models.yaml:656`) with a distinct adapter (`adapters/models.py:97-109`), a distinct `reproduction_status` (`BLOCKED_CHECKPOINT_BUNDLE` at `:696` vs `BLOCKED_CHECKPOINT_MAPPING` at `:631`) and a distinct checkpoint record (`:643-655` vs `:700-712`). No evidence from that entry is reused here; every claim below is re-derived from the pinned tree, the paper, or my own network reads.

## PAPER TITLE

*Composed Image Retrieval using Contrastive Learning and Task-oriented CLIP-based Features* (`models.yaml:615`; `paper_url: https://arxiv.org/abs/2308.11485` at `:616`; `upstream_repo_url: https://github.com/ABaldrati/CLIP4Cir` at `:617`).
- OBSERVED (network): `https://arxiv.org/abs/2308.11485` → title, authors Alberto Baldrati / Marco Bertini / Tiberio Uricchio / Alberto del Bimbo, published 2023-08-22, categories cs.CV, abstract identical to the pinned `README.md:7` title line. ACM TOMM 2023 (pinned README header `README.md:1`: `# CLIP4Cir (ACM TOMM 2023)`).
- **Registry `paper_title` is accurate.**

## PINNED SOURCE COMMIT

Registry `upstream_commit_sha: dfed9f748a8a4d05abf613164e06d59b22dbdf13` (`models.yaml:619`), `upstream_default_branch: master` (`:618`).

OBSERVED:
```
$ git -C workbench/third_party/CLIP4Cir rev-parse HEAD
dfed9f748a8a4d05abf613164e06d59b22dbdf13
$ git -C workbench/third_party/CLIP4Cir log -1 --format='%H %ci %s'
dfed9f748a8a4d05abf613164e06d59b22dbdf13 2023-09-05 09:31:52 +0200 Update README.md
$ git -C workbench/third_party/CLIP4Cir status --porcelain
(empty)
$ git -C workbench/third_party/CLIP4Cir branch -a
* (HEAD detached at dfed9f7)
  master
  remotes/origin/HEAD -> origin/master
  remotes/origin/master
```
**Exact match; checkout clean; the pin is a detached commit on `master`.** `workbench/backend/runtime.py:2013-2037` (`source_clean_and_pinned`) enforces exactly this (dirty → refuse; missing or mismatched pin → refuse).

## CHECKPOINT IDENTITY

Registry (`models.yaml:643-655`):
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
The registry names a **folder**, never a file, and publishes no digest. OBSERVED: `workbench/artifacts/checkpoints/` contains only `.gitkeep`; `workbench/artifacts/checkpoints/combiner_rn50x4_noft` does not exist.

**The registry's `CHECKPOINT_MAPPING_UNVERIFIED` label is honest — but its prose justification is factually stale, and the mapping is in fact resolvable from the author's own metadata.**

`workbench/docs/AUXILIARY_ASSET_AUDIT.md:221` (and `workbench/registry/assets/cluster_b4_classic.yaml:79`, `workbench/registry/audits/cluster_c_classic.yaml:61`) state:

> "Author Google Drive folder … does not document which file maps to the RN50x4 noft Combiner checkpoint, nor does it publish a checksum."

OBSERVED (network, unauthenticated):
```
$ curl -sL -A "Mozilla/5.0" "https://drive.google.com/embeddedfolderview?id=1ny2hhzP8HZBnXhjvDEni8P8G4_inCNTv#list"
title: pretrained_models - Google Drive
entries: CIRR -> 1_YmbdKdf5dmCoQ8HG6jCKkCwZVypSSoo
         FashionIQ -> 1fnGeZQA4XMfEERo-qDCosJXILl1UdQ_l
         README.md -> 1G7kZo7paiaRd3mif9_saERyhPevRCM88
$ curl -sL -A "Mozilla/5.0" "https://drive.google.com/embeddedfolderview?id=1fnGeZQA4XMfEERo-qDCosJXILl1UdQ_l#list"
RN50_fullft, RN50_imageft, RN50_noft, RN50_textft, RN50x4_fullft, RN50x4_noft
   RN50x4_noft -> 1OHmPOU9VYxDynoCyDfCIY8ijTsSltxuL
$ curl -sL -A "Mozilla/5.0" "https://drive.google.com/embeddedfolderview?id=1OHmPOU9VYxDynoCyDfCIY8ijTsSltxuL#list"
fiq_comb_RN50x4_noft.pt  (last-modified 11/3/22)
$ curl -sL -A "Mozilla/5.0" "https://drive.google.com/embeddedfolderview?id=1HaDyljcnP2S3sfCvOxAFSWHsveYJbrnu#list"
cirr_comb_RN50x4_noft.pt
```
The Drive folder **is** structured per dataset (`CIRR/`, `FashionIQ/`) and per variant, and the `FashionIQ/RN50x4_noft/` folder contains **exactly one** file, `fiq_comb_RN50x4_noft.pt`. The folder's own `README.md` was downloaded to `/tmp/drv_readme.bin` (2449 bytes) and says:

> "in the RN50x4-based is equal to 2560 … in the RN50x4-based is equal to 5120"
> "2) RN50x4-based Combiner trained on out-of-the-box CLIP features - CVPR22 Demo paper"
> "6) RN50x4-based trained Combiner on fine-tuned image and text encoder … (Best RN50x4-based model)"
> `combiner.load_state_dict(combiner_state_dict['Combiner'])`, `clip_model.load_state_dict(clip_state_dict["CLIP"])`

So the release structure **is documented by the author**: the noft variant is `fiq_comb_RN50x4_noft.pt` (Combiner only), and the provenance is the CVPR22 demo paper (OBSERVED: `https://openaccess.thecvf.com/content/CVPR2022/papers/Baldrati_Effective_Conditioned_and_Composed_Image_Retrieval_Combining_CLIP-Based_Features_CVPR_2022_paper.pdf`), whose §3.2 states "the CLIP encoders have been kept frozen and the only trained part of the model is the Combiner function" — i.e. that is the no-fine-tuning variant.

What is genuinely missing: no size, no SHA-256, no `download_url`, and no direct-file id in the registry (the file id `1Hb3lp3Vk0tSpAcMoq8KG5tFsw-jymryi` is only reachable by scraping the folder listing, which is not what the registry does). **The checkpoint's bytes remain unverified and no artifact-level identity is established.**

**P1.** The doc claim in `AUXILIARY_ASSET_AUDIT.md:221` is refuted by the folder listing I read live; the underlying blocker (no digest, no URL) is real, so the verdict `BLOCKED` is unchanged and the registry status is correct. Correcting the claim matters because it would otherwise license the fullft entry to reuse this finding.

## EVALUATION PROTOCOL

Registry: `literature_split_label: original` (`models.yaml:625`), `native_protocol: fashioniq_original_split` (`:626`), `supported_protocols: [fashioniq_original_split]` (`:627`).

Workbench protocol table (`src/evaluation/fashioniq.py:159-166`): `fashioniq_original_split` = `gallery="full_split"`, `reference_excluded=False`, alias `fashioniq_original`.

Native evaluator, pinned and read in full (`workbench/third_party/CLIP4Cir/src/validate.py`, 360 lines):
- Gallery: `fashioniq_val_retrieval` (`:113`) builds `FashionIQDataset('val', [dress_type], 'classic', preprocess)` (`:127`) and `FashionIQDataset('val', [dress_type], 'relative', preprocess)` (`:129`), then `extract_index_features(classic_val_dataset, clip_model)` (`:128`).
- The gallery IS `image_splits/split.{category}.val.json` in file order: `data_utils.py:144-147` (`self.image_names.extend(json.load(f))`), iterated whole by the classic-mode DataLoader (`utils.py:30-31`, `utils.py:38-43`, default `shuffle=False`).
- **The reference stays eligible.** `compute_fiq_val_metrics` (`:21-57`) computes `distances = 1 - predicted_features @ index_features.T` (`:44`), `torch.argsort(distances, dim=-1)` ascending (`:45`), labels against `target_names` only (`:49-51`), and asserts exactly one hit per query (`:51`). There is **no** reference mask, no reference removal, no `-10e10` anywhere in the FashionIQ path. Contrast within the same file: the CIRR path DOES remove the reference (`:162-166`, `reference_mask = torch.tensor(sorted_index_names != repeat(reference_names, …))`). That internal contrast is the strongest available evidence that FashionIQ's reference-eligibility here is a deliberate property of this evaluator and not an oversight.
- Category order in `main`: shirt, dress, toptee (`:329`, `:334`, `:339`); macro = `statistics.mean` of the three per-category recalls (`:353-354`).

**Verdict on the registry's protocol declaration: VERIFIED (STATIC_AUDIT).** Gallery scope, ordering, reference policy, category set and metric depths all trace to source lines. No runtime has been observed.

### Is CLIP4Cir's FashionIQ protocol the same cohort the *FashionIQ paper itself* reports?

This is the key protocol question for both CLIP4Cir entries, so I checked the dataset's own paper rather than only this repo's code.
- OBSERVED (network): Fashion IQ paper, `https://arxiv.org/html/1905.12794v3`, §4.2 table caption: "Dialog-based Image Retrieval. We report the performance on ranking percentile (P) and recall at N (R@N) at the 1st, 3rd and 5th dialog turns" — the original benchmark reports dialog-turn R@N, not the single-shot R@10/R@50 the CIR literature adopted.
- OBSERVED: the CLIP4Cir paper (arXiv 2308.11485, §4.2.1) says "We follow the standard experimental setting as in (Lee et al. 2021; Kim et al. 2021). We employ the average recall at rank K (Recall@K)… We set the batch size to 4096… All results are on the validation set". Its Table 9 `Ours RN-50x4` row (FashionIQ **validation**) is 44.41 / 65.26 / 39.46 / 64.55 / 47.48 / 70.98 / average 43.78 / 66.93.
- OBSERVED: the comparison baseline named by that paper, DCNet (`Kim et al. 2021`), in its own repo masks the reference: `workbench/third_party/DCNet/trainer/trainer_joint.py:217-218` sets `score_matrix_comp[ei, id2targetidx[elem['candidate']]] = -10e10` and the same for `score_matrix_corr`. Its paper row in CLIP4Cir's Table 9 is 23.95 / 47.30 / 28.95 / 56.07 / 30.44 / 58.29 / 27.78 / 53.89.
- OBSERVED (network): CVPR22 demo paper (`Effective Conditioned and Composed Image Retrieval…`), Table 1, `Proposed approach` = **36.36 / 58.00 / 31.63 / 56.67 / 38.19 / 62.42 / 35.39 / 59.03**, in a context of the same baseline family (CoSMo 24.90/49.18…, DCNet 23.95/47.30…, SAC w/BERT 28.02/51.86…). Its §3.2 is explicit that CLIP is frozen (noft). Its §4.2 says "To remain consistent with standard evaluation protocol of FashionIQ we consider the dataset subdivided in three categories… when retrieval is carried out, only the images of the same category are taken into consideration" — it describes category scoping only, and says nothing about removing the reference.

**This is a P1 scientific-comparability finding, but NOT the one the workbench docs assume, and it cuts the other way from a plain "workbench is wrong" claim.** The genuinely inconsistent pair is:
- CLIP4Cir's `validate.py` keeps the reference eligible; DCNet (the baseline whose setting CLIP4Cir claims to follow) masks it. The two numbers therefore live in different cohorts, and CLIP4Cir's Table 9 juxtaposition of its own reference-eligible rows against DCNet's reference-masked row is an apples-to-oranges comparison in the *paper*, not merely in the workbench.
- The workbench does **not** reproduce the paper's cross-method comparison: it declares one protocol per model and refuses cross-protocol aggregates (`docs/FASHIONIQ_PROTOCOL_PARITY.md:20-24`, `docs/PROTOCOL_AUDIT.md:64-66`). So the workbench's declaration is faithful to CLIP4Cir's *code*, and the paper's protocol is a paper-level defect that the workbench must NOT paper over by flipping `combiner_rn50x4_noft` to reference-excluded.
- Consequence for this model: any CLIP4Cir-vs-DCNet-vs-iLearn comparison inside a single workbench cohort is invalid, and `fashioniq_original_split` numbers for this model are not interchangeable with the reference-excluded cohorts used by Air-Know/ConeSep/HABIT/INTENT/DCNet. This is exactly what `models.yaml:626-627` and the docs say.

## DATASET CONTRACT

- `fashioniq_layout: fashioniq_standard` (`models.yaml:622`), `preparation_contract: manual_standard` (`:623`), `preparation_id: clip4cir_fashioniq` (`:624`), `canonical_dataset_link: fashionIQ_dataset` (`:605`), `fashioniq_image_extensions: [".jpg"]` (`:606`).
- Evaluator's data root is hard-coded, no CLI override: `data_utils.py:11` `base_path = Path(__file__).absolute().parents[1].absolute()`; `data_utils.py:140` `base_path / 'fashionIQ_dataset' / 'captions' / f'cap.{dress_type}.{split}.json'`; `:146` `… / 'image_splits' / f'split.{dress_type}.{split}.json'`; `:176` (classic gallery image) and `:158/:161/:170` (relative/train/test images) `… / 'images' / f"{name}.jpg"`. OBSERVED: `validate.py:263-277` argparse declares only `--dataset/--combining-function/--combiner-path/--projection-dim/--hidden-dim/--clip-model-name/--clip-model-path/--target-ratio/--transform` — **no dataset-root flag exists.**
- Preparation contract `clip4cir_fashioniq` (`registry/preparation_contracts.yaml:140-169`) declares `generated_artifacts: []`, raw inputs `captions/cap.{dress,shirt,toptee}.val.json`, `image_splits/split.{dress,shirt,toptee}.val.json`, `images` (`:146-153`), and policy `MANUAL_REQUIRED` / `DETERMINISTIC_AT_EVALUATION` (`:167-168`).
- OBSERVED (filesystem): `data/fashionIQ_dataset/` holds `captions/` (9 files), `image_splits/` (9), `images/` (74381 files), `qwen_captions/` (3). `data/FashionIQ -> fashionIQ_dataset`.
- OBSERVED (filesystem): `workbench/third_party/CLIP4Cir/fashionIQ_dataset` is **absent** (not a symlink, not a directory). The link staging is implemented and generalized: `workbench/scripts/prepare_dataset.py:28-68` (`prepare_canonical_link`, gated on canonical-root verified + source present + destination unoccupied + authorization), driven by `models_requiring_link()` (`:70-77`) over every model declaring `canonical_dataset_link` (csmcir `:33`, combiner noft `:605`, fullft `:669`, sprc `:781`). The final line is `destination.symlink_to(root, target_is_directory=True)` (`:65`).

## PREPROCESSING CONTRACT

- `validate.py:279` `clip_model, clip_preprocess = clip.load(args.clip_model_name, device=device, jit=False)`; `:280-281` `input_dim = clip_model.visual.input_resolution`, `feature_dim = clip_model.visual.output_dim`.
- `validate.py:289-291` `if args.transform == 'targetpad': preprocess = targetpad_transform(args.target_ratio, input_dim)`. `--target-ratio` default `1.25` (`:273`); `--transform` default `targetpad` (`:274-275`).
- `data_utils.py:82-96` `targetpad_transform(target_ratio, dim)` = `TargetPad(target_ratio, dim)` → `Resize(dim, interpolation=PIL.Image.BICUBIC)` → `CenterCrop(dim)` → `_convert_image_to_rgb` → `ToTensor()` → `Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))`. `TargetPad.__call__` (`:54-63`) pads to the target ratio only when `actual_ratio >= target_ratio` (`:57`).
- Gallery index images use the **same** transform instance (`utils.py:38-39`, `images = images.to(device)`; the transform is applied in `data_utils.py:177`).
- With `RN50x4`, `input_dim = 288` and `feature_dim = 640`. This is verified externally, not assumed: OBSERVED (network) CLIP paper `https://arxiv.org/pdf/2103.00020`, Table 19 `CLIP-ResNet hyperparameters` — RN50x4: embedding dimension **640**, input resolution **288**, width **2560**, text width 640, heads 10. This independently corroborates `--projection-dim 2560` / `--hidden-dim 5120` (= 8·d) in the adapter and the paper's §4.1 "the first two linear layers … have input-output dimensionality equal to (d, 4d) … (4d, 8d) … (8d, 1) … (8d, d)".
- **Adapter parity: exact.** `adapters/models.py:91-94` passes `--dataset fashionIQ --combining-function combiner --combiner-path <path> --projection-dim 2560 --hidden-dim 5120 --clip-model-name RN50x4 --target-ratio 1.25 --transform targetpad`. It **omits `--clip-model-path`**, which is the defining property of the noft variant (compare `:101-109` for fullft, which passes it). That omission is the correct noft semantics; it is not a shortcut.

## AUGMENTATION CONTRACT

- **None at evaluation.** `validate.py` never calls `generate_randomized_fiq_caption`; OBSERVED `grep -rn "generate_randomized_fiq_caption" src/*.py` → `utils.py:57` (definition), `clip_fine_tune.py:20,148`, `combiner_train.py:20,140`. Training-only.
- The validation caption composition is **deterministic**: `validate.py:89-92`
  ```python
  flattened_captions: list = np.array(captions).T.flatten().tolist()
  input_captions = [f"{flattened_captions[i].strip('.?, ').capitalize()} and {flattened_captions[i + 1].strip('.?, ')}" for i in range(0, len(flattened_captions), 2)]
  ```
  then `clip.tokenize(input_captions, context_length=77)` (`:93`) — **no `truncate=True`**, so an over-77-token composition raises in `clip.tokenize` rather than being silently truncated.
- `clip_model = clip_model.float().eval()` in `fashioniq_val_retrieval:124`; the Combiner is `.eval()` at `:308`; extraction is `torch.no_grad()` (`utils.py:40`, `validate.py:96`).
- Tokenizer asset: `clip.tokenize` → `clip/clip.py` imports `SimpleTokenizer`, whose `default_bpe()` resolves the **in-repo** `bpe_simple_vocab_16e6.txt.gz` (OBSERVED: `curl -sIL …/clip/bpe_simple_vocab_16e6.txt.gz` → HTTP 200, `content-length: 1356917`). No second external tokenizer download is needed. Not registered as an asset, correctly, since it is vendored in the installed package.

## GALLERY CONTRACT

- Candidate universe per category = the **complete** `split.{category}.val.json`, source-file order, no dedup, no sorting by the evaluator (`data_utils.py:144-147`; `utils.py:30-31,38-43`; `validate.py:127-128`).
- OBSERVED (local files, my own computation): `dress` 3817 gallery ids / 2017 val triplets, `shirt` 6346 / 2038, `toptee` 5373 / 1961, total 6016 triplets. Duplicate gallery ids: 0 in all three categories. Targets absent from their own gallery: 0 in all three. Duplicate target ids across queries: 0 in all three. Every target appears exactly once (this is why `validate.py:51`'s one-hit assert would hold).
- The reference is a gallery member by construction (the `compute_fiq_val_metrics` path does not touch `index_names`) and is additionally *used* as the query image: `validate.py:80` `name_to_feat = dict(zip(index_names, index_features))` and `:101-104` retrieves `reference_image_features` from that same map.
- **Consequence, stated explicitly: the reference can outrank the target, and there is no exclusion protecting Recall@K.** That is the declared cohort, not a bug — but it is exactly why this protocol cannot be compared against the reference-excluded cohorts.

## METRIC CONTRACT

- `validate.py:41` gallery features normalized; `:44` `distances = 1 - predicted_features @ index_features.T`; `:45` ascending `torch.argsort`; `:46` `sorted_index_names = np.array(index_names)[sorted_indices]`; `:49-51` labels = (sorted name == target), with the `sum(labels, dim=-1) == 1` assertion; `:54-55` `recall_at10 = sum(labels[:, :10]) / len(labels) * 100`, `recall_at50 = labels[:, :50]`.
- Macro: `:353-354` `average recall10 = mean(average_recall10_list)`, `average recall50 = mean(average_recall50_list)` over `[shirt, dress, toptee]` (append order `:331-342`). Equal-weight macro of categories; **not** a query-weighted figure.
- The registry records `reported_r10/r50/mean: null` (`models.yaml:628-630`).
- Parsimony note: this evaluator prints only per-category and average R@10/R@50 — **no R@1**, unlike the iLearn evaluators. The workbench's own cohort model (`docs/FASHIONIQ_PROTOCOL_PARITY.md:20-24`, `docs/PROTOCOL_AUDIT.md:64-66`) treats R@10/R@50/macro as the comparison surface, which matches this output exactly. Fine.

## SOURCE-TO-WORKBENCH PARITY

| Component | Pinned source | Workbench | Verdict |
|---|---|---|---|
| Commit pin | `dfed9f748a8a…` | `models.yaml:619` | MATCH |
| Entrypoint + CLI shape | `validate.py:263-277` | `adapters/models.py:86-94` | MATCH (exact flags) |
| noft semantics | no `--clip-model-path` overlay; `:283-287` skipped when unset | adapter omits the flag | MATCH |
| Backbone | `clip.load('RN50x4')` `:279`; Azure CDN hash-name | `auxiliary_assets.yaml:162-178` | MATCH (see CF-1) |
| Preprocess | `targetpad_transform(1.25, 288)` | adapter `--transform targetpad --target-ratio 1.25` | MATCH |
| Gallery/protocol | full ordered split, reference eligible | `models.yaml:626-627`; `src/evaluation/fashioniq.py:159-166` | MATCH |
| Caption composition | `validate.py:89-93` | adapter runs the native script; local `src/datasets/fashioniq.py` default is NOT byte-identical but is unreached | MATCH for the executed path |
| Dataset root | `<checkout>/fashionIQ_dataset`, `.jpg` | `canonical_dataset_link` declared; link absent; canonical root is `.png` | MISMATCH (CF-4) |
| Metric extraction | stdout `shirt_recallat10 = …` / `average recall10 = …` `:344-354` | no parser (`metrics_extraction.py:306-320`) | MISMATCH (CF-5) |
| Checkpoint file | Drive folder; one `.pt` per variant | folder URL only; no digest, no URL | MISMATCH (CF-1) |

## RUNTIME STATUS

- `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`models.yaml:631`); `command_status: COMMAND_AUDITED` (`:632`); `replay_status: REPLAY_CODE_IMPLEMENTED` (`:607`); `replay_type: OFFICIAL_CLI` (`:610`); `category_model_policy: shared` (`:608`); `checkpoint_serialization: state_dict` (`:609`); `environment_required: true` (`:611`); `strict_clean_checkpoint_available: null` (`:641`).
- Environment (`:633-640`): `WORKBENCH_CLIP4CIR_PYTHON`, `confidence: VERIFIED_README`, python 3.8, pytorch 1.11.0, torchvision 0.12.0, cuda UNKNOWN, packages `[pandas==1.4.2, comet-ml==3.21.0, clip]`. **OBSERVED: this is a faithful transcription of the pinned README** — `README.md:113` `conda create -n clip4cir -y python=3.8`, `:115` `conda install -y -c pytorch pytorch=1.11.0 torchvision=0.12.0`, `:116` `conda install -y -c anaconda pandas=1.4.2`, `:117` `pip install comet-ml==3.21.0`, `:118` `pip install git+https://github.com/openai/CLIP.git`. Note `clip` is installed from git **main**, i.e. unpinned — a real reproducibility weakness in the upstream instructions (M-3).
- OBSERVED: `workbench/envs/` contains only `README.md` and `csmcir-import.txt`; no interpreter exists. `WORKBENCH_CLIP4CIR_PYTHON` is unset; `runtime.require_verified_model_interpreter` (`runtime.py:1475-1493`) raises rather than emitting a bare interpreter when the environment is not verified, so `CombinerNoftAdapter.build_command` cannot even produce a command today.
- `cuda: UNKNOWN` is honest but incomplete: `utils.py:15-18` selects cuda only `if torch.cuda.is_available()`, so the evaluator **does** fall back to CPU. It is therefore slow but not GPU-mandatory, contrary to the README's "heavily recommended".

## PAPER-METRIC STATUS

Registry: `reported_r10: null`, `reported_r50: null`, `reported_mean: null` (`models.yaml:628-630`).

**Verified accurate for the noft variant: the CLIP4Cir TOMM paper contains no out-of-the-box/FashionIQ table.** I read all ten tables. OBSERVED (network, full text at `https://ar5iv.labs.arxiv.org/html/2308.11485`): Tables 1 and 2 vary the combining function and fine-tuning modality using **RN-50 only** ("Throughout all the experiments, we use the RN-50 CLIP model", §4.3) — the Table 1 `CF=Combiner, IFT=✗, TFT=✗` row is 31.85 / 52.50, 27.22 / 50.62, 33.81 / 57.57, average 30.96 / 53.56, on RN-50, not RN-50x4. Tables 3–8 are RN-50 ablations. Table 9 is the FashionIQ SOTA comparison and its only `Ours RN-50x4` row is 44.41 / 65.26 / 39.46 / 64.55 / 47.48 / 70.98 / average 43.78 / 66.93, described in §4.7's framing as the full two-stage method (fine-tuned encoders + Combiner). Table 10 is CIRR test. **No RN-50x4 noft FashionIQ row exists in the paper.**

**The correct source for a noft number is a different paper, which the registry does not cite.** OBSERVED (network): CVPR 2022 demo paper *Effective conditioned and composed image retrieval combining CLIP-based features* (Baldrati, Bertini, Uricchio, Del Bimbo), Table 1 `Proposed approach` = **36.36 / 58.00 / 31.63 / 56.67 / 38.19 / 62.42 / 35.39 / 59.03**, Fashion-IQ validation; its §3.2 confirms frozen CLIP encoders (noft) with TargetPad target ratio 1.25, and §5.1 confirms the val split. That is the RN50x4 noft Combiner variant, i.e. this model.

Arithmetic I executed on the published cells (not read from any doc): shirt (36.36+31.63)/2 = 33.995, dress (58.00+56.67)/2 = 57.335, toptee (38.19+62.42)/2 = 50.305; mean of the three = 47.212 → **47.21**. Under the registry's own `reported_mean` convention (`docs/PROTOCOL_AUDIT.md`: mean is the arithmetic mean of macro R@10 and R@50) the same table gives (35.39+59.03)/2 = **47.21**. Both conventions agree on 47.21 here, which is a useful consistency check.

So: the registry's nulls are **defensible for the cited paper** and **incomplete for the model**, because an author-published noft FashionIQ number does exist — in a paper the registry does not cite. Recording it is a judgement call the workbench deliberately left to `null` (no fabricated numbers), but the audit should say plainly that a candidate row exists.

## CRITICAL FINDINGS

**CF-1 — P1 — the checkpoint is not resolvable to bytes, and the docs' stated reason is wrong.** `models.yaml:643-655` names a Drive *folder* with `download_url: null`, `expected_size_bytes: null`, `expected_sha256: null`. `AUXILIARY_ASSET_AUDIT.md:221` / `cluster_b4_classic.yaml:79` / `cluster_c_classic.yaml:61` assert the folder "does not document which file maps to the RN50x4 noft Combiner checkpoint, nor does it publish a checksum". My OBSERVED folder read refutes the first half: `FashionIQ/RN50x4_noft/` contains exactly one file, `fiq_comb_RN50x4_noft.pt`, and the folder's own `README.md` states the release structure and the `Combiner`/`CLIP` state keys. The second half stands: no digest is published. Also note the Drive folder has **two** distinct parent folders (`ptx`-style duplicates, ids `1fnGeZQA4XMfEERo-qDCosJXILl1UdQ_l` and `1_YmbdKdf5dmCoQ8HG6jCKkCwZVypSSoo`, each containing `FashionIQ/` and `CIRR/` with separate RN50x4_noft files: `1OHmPOU9…/fiq_comb_RN50x4_noft.pt` vs `1HaDylj…/cirr_comb_RN50x4_noft.pt`), so a resolver must also pick the right parent — which is resolvable, since the names are unambiguous. Action: replace the prose with the actual listing evidence, keep the model blocked on the missing digest/URL, and record the per-variant filename.

**CF-2 — P1 — the noft/fullft mapping is not "assumed"; it is documented by the author and corroborated by the pinned code, but the *FA* technique the fullft selection logic is worth checking separately.** Evidence establishing noft vs fullft: (a) Drive folder `README.md` lists release item 2 as "RN50x4-based Combiner trained on out-of-the-box CLIP features" and item 6 as "RN50x4-based trained Combiner on fine-tuned image and text encoder"; (b) the Drive tree names them `RN50x4_noft/` and `RN50x4_fullft/`, and the noft folder holds only a Combiner (`fiq_comb_*`) while the fullft folder holds **two** files (`fiq_clip_RN50x4_fullft.pt`, `fiq_comb_RN50x4_fullft.pt`); (c) the pinned validator requires a second artefact only via `--clip-model-path` (`validate.py:283-287`), which the fullft adapter passes and the noft adapter does not; (d) the CVPR22 demo paper (§3.2) confirms frozen encoders for the co-released model. **Verdict: the noft/fullft mapping is genuinely VERIFIED (STATIC_AUDIT) from author metadata + source + a second paper — not merely assumed. The registry/docs understate this.** The residual unknown is byte identity (no digest), not which artefact is which.

**CF-3 — P0 — the dataset the evaluator would actually read does not contain the images it opens, and the failure would be silent-but-fatal.** `data_utils.py:158,161,170,176` open `f"{name}.jpg"`. OBSERVED: the canonical root's `images/` holds **74381 `.png` and 0 `.jpg`** (`ls | sed 's/.*\.//' | sort | uniq -c` → `74381 png`); per category, 3817/3817 dress, 6346/6346 shirt, 5373/5373 toptee validation ids exist as `.png` and **0** as `.jpg`. `models.yaml:602-603` asserts the canonical root "opens `.jpg` images there" and says the root "is staged at that path by an explicitly authorized preparation step" — but no step converts or aliases extensions: `preparation_contracts.yaml:154` declares `generated_artifacts: []` and `automation_policy: MANUAL_REQUIRED` (`:168`). The workbench's own gate reports it (`READINESS_MATRIX.md:60-61`: "FashionIQ image extension requirement: 3817 of 3817 dress validation images lack .jpg…"). Additionally `data_utils.py:151-184` wraps `__getitem__` in `try/except Exception as e: print(f"Exception: {e}")` and returns **None**, and `utils.py:81-88 collate_fn` silently **drops** None samples. So a missing image does not raise: it silently shrinks the gallery (and the query set) and then `validate.py:51`'s one-hit assert fires with a confusing message, or worse, the recall denominator silently changes if the dropped reference is a query. Action: either stage a `.jpg`-extension tree (hardlink/symlink is sufficient and byte-preserving) or correct the registry declaration; do not run this evaluator against a `.png` canonical root.

**CF-4 — P1 — the declared dataset link is absent, and the workbench request's `dataset_root` never reaches the evaluator.** OBSERVED: `workbench/third_party/CLIP4Cir/fashionIQ_dataset` does not exist. `CombinerNoftAdapter.command` (`adapters/models.py:90-94`) ignores `request.dataset_root` entirely; the only path is the hard-coded `base_path` link. `prepare_dataset.py:28-68` can create it, but it is an operator-authorized step and has not been performed. So an operator holding a perfectly good canonical root still cannot run this evaluator. Action: keep the link staging as an explicit authorized step (as the registry comment `models.yaml:601-604` intends), and record the `.png`→`.jpg` decision in the same step.

**CF-5 — P1 — no metric parser, so even a successful run yields no comparable number.** `metrics_extraction.py:292-320` dispatches only `limn` and `csmcir`; everything else returns `extraction_status="AGGREGATE_PARSER_UNAVAILABLE"`, `parity_status="PARSER_UNAVAILABLE"`. The evaluator's output shape is trivially parseable (`validate.py:344-354`: `shirt_recallat10 = …` … `average recall10 = …`, `average recall50 = …`). Action: add a parser for the six category numbers + two averages + macro recomputation, mirroring `parse_csmcir_output` (`:200-281`).

## MINOR FINDINGS

- **M-1 — P2 — `artifact_type: file` is schema-correct but `bundle_placeholder` is mishandled.** The noft entry correctly uses `artifact_type: file` (`models.yaml:646`), which `checkpoint_missing_paths` (`registry.py:269-291`) treats as a single file, consistent with `checkpoint_path` (`:618-621`). No bug for this entry. (The fullft sibling's `bundle_placeholder` is the one that diverges, `evaluate_models.py:227` handling only `run_directory`/`bundle_directory` — out of scope here, noted so the two entries are not conflated.)
- **M-2 — P2 — `fashioniq_image_extensions: ['.jpg']` is correct for the evaluator but the *comment* on the layout map is wrong for `fashioniq_standard`.** `fashioniq_layout.py:81` says "`images/{id}.png` for the standard layout" while `:84` maps `fashioniq_standard` to `("images", False)` and the registry declares `.jpg`. A reader trusting the comment at `:81` would think `.png` is the standard answer (true only for CSMCIR, `models.yaml:46-50`). Action: fix the comment; behaviour is correct.
- **M-3 — P2 — the upstream environment is not pinned for the tokenizer/CLIP package.** `README.md:118` installs `clip` from git main (unpinned), and the registry records it as the unversioned string `clip` (`models.yaml:640`). `clip.load`/`clip.tokenize` semantics are exactly what decides the caption tokenization and the RN50x4 class definition, so an unpinned `clip` is a genuine reproducibility hole. Action: record a commit sha for `openai/CLIP` in the environment spec.
- **M-4 — P2 — `--projection-dim`/`--hidden-dim` are passed without a derivation recorded anywhere.** The adapter hard-codes 2560/5120 (`adapters/models.py:93`). They are correct (`d = 640` → 4d = 2560, 8d = 5120; corroborated by CLIP paper Table 19 and CLIP4Cir paper §4.1), but the registry never states *why*, and a wrong `--projection-dim` would fail only at `load_state_dict` time. Action: add a one-line derivation note.
- **M-5 — P3 — the Drive folder's own README is the author's provenance document and the workbench does not reference it.** It is downloadable (OBSERVED, 2449 bytes) and states the key names and dims. Action: cite it in the checkpoint record; it is the closest thing to author attestation available without a digest.
- **M-6 — P3 — no R@1 is produced by this evaluator, so any workbench R@1 aggregation for this model is impossible.** Not a defect; recorded so nobody adds it. Also note the evaluator's per-category macro is unweighted while the three categories have 2017/2038/1961 queries (OBSERVED) — near-equal, so the distinction is small but real.
- **M-7 — P3 — `utils.py:15-18` device selection is CPU-capable, contradicting the README's GPU requirement.** `device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")`. The registry's `cuda: UNKNOWN` should be recorded as "not required" since no unconditional `.cuda()` exists in the FashionIQ path (OBSERVED `grep`: none in `src/validate.py`, `src/data_utils.py`, `src/utils.py`, `src/combiner.py`).

## UNRESOLVED EVIDENCE

1. **Checkpoint bytes.** No `fiq_comb_RN50x4_noft.pt` on this host; no digest, no size, no direct URL in the registry; Drive serves the file (`content-disposition: attachment; filename="fiq_comb_RN50x4_noft.pt"`, HTTP 200) but no `content-length` was exposed to HEAD, so even the size is UNKNOWN. The `'Combiner'` key is documented by the author and required by `validate.py:306-307`, but I did not load the file, so key/dimension compatibility is UNKNOWN.
2. **Base RN50x4 weights.** `~/.cache/clip` does not exist on this host (OBSERVED: `ls: cannot access '/home/heheboiz/.cache/clip'`). The registry's expected SHA-256 equals the CDN's own object name (`auxiliary_assets.yaml:177`), and the URL resolves (OBSERVED: HTTP 200, `content-length: 421854225` = 402.3 MiB — *not* 448 MiB as I initially inferred; I report only the observed value). Whether a served byte stream hashes to that name was **not** verified — that would require downloading ~400 MiB, which the assignment forbids. STATIC_AUDIT only.
3. **`expected_size_bytes` for the base weights is `null`** even though the CDN publishes a length; likewise for the Combiner artifact. Not fillable without downloading.
4. **Whether the workbench's `fashionIQ_dataset` link target matches the annotation files the paper used.** I confirmed the local val splits are the canonical ones by count (3817/6346/5373 images; 2017/2038/1961 triplets; 6016 total). The paper says 15,537 val images / 6,017 triplets for FashionIQ (arXiv 2308.11485 §4.2.1; CVPR22 §5.1 identical). Local: 3817+6346+5373 = 15536 images and 2017+2038+1961 = 6016 triplets — **one image and one triplet fewer than the paper's 15,537/6,017**. I cannot resolve that off-by-one (FashionIQ's own paper Table 1 also reports Tops&Tees val = 5,374, i.e. also one more than the local 5,373). This is a pre-existing dataset-download question affecting every FashionIQ model in the workbench, not a CLIP4Cir-specific defect, and I did not chase it. Reported as UNKNOWN, not PASS.
5. **Tie behaviour.** `torch.argsort` default is unstable; the workbench explicitly declines to pin ties (`docs/FASHIONIQ_PROTOCOL_PARITY.md`). Measure-zero, UNKNOWN, not claimed.
6. **Whether the CVPR22 Table 1 `36.36/58.00/31.63/56.67/38.19/62.42/35.39/59.03` row is exactly `fiq_comb_RN50x4_noft.pt`.** Corroborated by the Drive README's own attribution of item 2 to the "CVPR22 Demo paper" and by §3.2's frozen-encoder statement, but no per-file hash ties the published row to the published bytes. UNKNOWN.
7. **No runtime observation of any kind.** No interpreter, no checkpoint, no link, no GPU. Nothing in this document is RUNTIME_VERIFIED.

## FINAL VERDICT

**BLOCKED.**

---

# EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation -> checkpoint -> output -> metric comparison`, arrow by arrow.

**1. paper/README/config → this registry entry.** VERIFIED (STATIC_AUDIT). `models.yaml:615-619` (title, arXiv URL, repo URL, pin) match `https://arxiv.org/abs/2308.11485` (OBSERVED network) and `workbench/third_party/CLIP4Cir/README.md:1-7` (OBSERVED local). The environment block `:633-640` is a line-for-line transcription of `README.md:113-118` (OBSERVED). `--projection-dim 2560 / --hidden-dim 5120` in the adapter are consistent with CLIP `d=640` (CLIP paper Table 19, OBSERVED network: RN50x4 → dim 640, res 288, width 2560) and with CLIP4Cir paper §4.1 ((d,4d) then (4d,8d)).

**2. README/config → source dataset.** VERIFIED (STATIC_AUDIT). `README.md:141-161` documents `project_base_path/fashionIQ_dataset/{captions,images,image_splits}` with `.jpg` examples (`:153-155`); `data_utils.py:11,140,146,176` implement exactly that, anchored on the checkout. **Arrow fails at runtime:** the declared link `workbench/third_party/CLIP4Cir/fashionIQ_dataset` is absent (OBSERVED), and the canonical root is `.png`-only (OBSERVED, 74381 png / 0 jpg). ⇒ BLOCKED at this arrow.

**3. source dataset → preprocessing.** VERIFIED (STATIC_AUDIT). `validate.py:289-291` selects `targetpad_transform(1.25, 288)`; `data_utils.py:54-63,82-96` give the exact pipeline; the gallery DataLoader uses the same instance (`utils.py:38-43`). Nothing external is needed beyond torchvision.

**4. preprocessing → official evaluator.** VERIFIED (STATIC_AUDIT). `validate.py:263-277` (CLI) → `:279` (`clip.load('RN50x4')`) → `:280-281` (dims) → `:304-309` (Combiner built, `state_dict['Combiner']` loaded, `.eval()`, `combining_function = combiner.combine_features`) → `:325-354` (three categories, per-category R@10/R@50, macro). The `'Combiner'` key is corroborated by the author's own Drive README (OBSERVED) and by `utils.py:117-131` `save_model`, which saves under `model_to_save.__class__.__name__` (i.e. `'Combiner'`).

**5. official evaluator → workbench invocation.** PARTIAL, STATIC_AUDIT only. `adapters/models.py:86-94` produces a command that is byte-for-byte the README's validation invocation minus the flags this variant must omit, plus explicit values for every default. `command_status: COMMAND_AUDITED` (`models.yaml:632`) and `replay_status: REPLAY_CODE_IMPLEMENTED` (`:607`) are accurate as *audited*, but `CombinerNoftAdapter.build_command` calls `_runtime.python_executable(self.model)` (`:91`), which delegates to `require_verified_model_interpreter` (`runtime.py:1475-1493`) and **raises** because `WORKBENCH_CLIP4CIR_PYTHON` is unset. So the command is audited but not constructible here. ⇒ BLOCKED at this arrow (no interpreter), independently of the data and checkpoint blockers.

**6. workbench invocation → checkpoint.** BLOCKED. `checkpoint_path('combiner_rn50x4_noft', variant, root)` = `workbench/artifacts/checkpoints/combiner_rn50x4_noft/rn50x4-noft-unresolved.pt` (`registry.py:618-621`); OBSERVED: that directory does not exist. `download_url: null` means no acquisition path exists in the registry; the author file name (`fiq_comb_RN50x4_noft.pt`, OBSERVED) differs from the registry placeholder name, so even a staged file would need the operator to know the mapping.

**7. checkpoint → output.** NOT_RUN. No evaluator process was started; no stdout exists. There is no `run_id`, no canonical result JSON, no per-query export. The only artifacts in `workbench/artifacts/` are `environment_assignment.json` and `environment_plan.json`.

**8. output → metric comparison.** NOT_RUN, and blocked by a code gap even if output existed. `metrics_extraction.py:311-320` returns `AGGREGATE_PARSER_UNAVAILABLE`/`PARSER_UNAVAILABLE` for this `model_id` (it dispatches only `limn` at `:306` and `csmcir` at `:308`). The paper-side target is also absent from the registry (`reported_*: null`, `models.yaml:628-630`) — correctly so for the cited paper, which publishes no RN-50x4 noft FashionIQ row (OBSERVED: full text read; Tables 1–2 are RN-50-only, Table 9's sole RN-50x4 row is the fine-tuned model). A candidate external row does exist: CVPR22 demo Table 1 `36.36 / 58.00 / 31.63 / 56.67 / 38.19 / 62.42 / 35.39 / 59.03`, macro 47.21 (my arithmetic on those cells).

---

# CHECKLIST

Each item: `N. question — verdict — evidence — severity — action`.

1. **Registered `model_id` present and unique** — VERIFIED — `models.yaml:600`; `grep -n "model_id: combiner_rn50x4_noft"` → exactly one model row; `ADAPTERS` keyed once (`adapters/models.py:138`) — NONE — none.
2. **`method_name` and `paper_title` recorded and accurate** — VERIFIED — `models.yaml:612`, `:615`; title matches arXiv 2308.11485 and `README.md:7` — NONE — none.
3. **`paper_url` resolves and identifies the same work** — VERIFIED — `models.yaml:616`; OBSERVED arXiv API/HTML: 4 authors, cs.CV, 2023-08-22, matching title — NONE — none.
4. **`upstream_repo_url` recorded** — VERIFIED — `models.yaml:617`; OBSERVED `git remote -v` → `git@github.com:ABaldrati/CLIP4Cir` — NONE — none.
5. **Pinned commit equals checkout HEAD** — VERIFIED — `models.yaml:619` == `git rev-parse HEAD` → `dfed9f748a8a4d05abf613164e06d59b22dbdf13` — NONE — none.
6. **Checkout clean** — VERIFIED — `git status --porcelain` empty; `runtime.py:2013-2037` accepts — NONE — none.
7. **`upstream_default_branch` correct** — VERIFIED — `models.yaml:618` `master`; OBSERVED `branch -a` → `remotes/origin/HEAD -> origin/master` — NONE — none.
8. **Pin equals current upstream `master`** — UNKNOWN — no fetch performed (read-only, no network write); single-commit detached checkout — P3 — re-verify on a writable host.
9. **`source_available` / `source_dir` correct** — VERIFIED — `models.yaml:620-621`; tree present with `src/validate.py` — NONE — none.
10. **Entrypoint exists and matches the adapter `script`** — VERIFIED — `adapters/models.py:88` `script="src/validate.py"`; file present, 360 lines — NONE — none.
11. **`native_protocol` matches the evaluator's gallery** — VERIFIED — `models.yaml:626` == full ordered split, `data_utils.py:144-147`; table `src/evaluation/fashioniq.py:160-162` — NONE — none.
12. **`supported_protocols` is exactly the native one** — VERIFIED — `models.yaml:627`; no second branch exists in `validate.py` for FashionIQ — NONE — none.
13. **`literature_split_label: original` consistent** — VERIFIED — `models.yaml:625`; protocol table `:10`; paper §4.2.1 "All results are on the validation set" with the dataset's original val split — NONE — none.
14. **Reference stays eligible in the evaluator** — VERIFIED — `validate.py:44-55`: no mask, no removal; contrast with CIRR at `:162-166` — NONE — none.
15. **Reference-exclusion mechanism recorded as none (not mask, not removal)** — VERIFIED — `grep -n "10e10\|mask" src/validate.py` → no FashionIQ-path hit; only the CIRR `reference_mask` at `:163-166` — NONE — none.
16. **Reference can outrank the target (probability>0)** — VERIFIED — reference is a gallery member (`:127-128`, `:80`, `:101-104`) and no exclusion applies — P3 — none; this *is* the declared cohort.
17. **Sort direction / distance** — VERIFIED — `:44` `1 - pred @ gallery.T`, `:45` ascending `argsort` — NONE — none.
18. **Tie-breaking pinned** — UNKNOWN — `torch.argsort` default unstable; workbench declines to pin — P3 — accept; measure-zero.
19. **Recall@K arithmetic (`100·hits/N`)** — VERIFIED — `:54-55` — NONE — none.
20. **Denominator is the query count** — VERIFIED — `len(labels)` = number of relative-val triplets (`:187`), not gallery size — NONE — none.
21. **One-target-per-query invariant** — VERIFIED (static) — `:51` assert; OBSERVED 0 duplicate targets and 0 targets absent from their gallery across all three categories — NONE — none.
22. **Macro definition matches source** — VERIFIED — `:353-354` `statistics.mean` over three per-category recalls, order `[shirt, dress, toptee]` — NONE — none.
23. **K values are 10 and 50** — VERIFIED — `:54-55` — NONE — none.
24. **`reported_r10` matches the cited paper** — VERIFIED as `null` — `models.yaml:628`; OBSERVED full read of arXiv 2308.11485: no RN-50x4 noft FashionIQ row exists — NONE — none.
25. **`reported_r50` matches the cited paper** — VERIFIED as `null` — same evidence — NONE — none.
26. **`reported_mean` matches the cited paper** — VERIFIED as `null` — same evidence — NONE — none.
27. **An author-published noft row exists elsewhere** — VARIANT — CVPR22 demo paper Table 1 `Proposed approach` 36.36/58.00/31.63/56.67/38.19/62.42/35.39/59.03 (OBSERVED network); Drive README attributes item 2 to that paper and §3.2 confirms frozen CLIP — P2 — record as a candidate sanity reference with the different citation, or state explicitly that the registry deliberately records nulls.
28. **Macro recomputed from the CVPR22 row** — VERIFIED — my arithmetic: shirt 33.995, dress 57.335, toptee 50.305, mean 47.212; `(35.39+59.03)/2 = 47.21` — NONE — none.
29. **Checkpoint variants declared with noise provenance** — VARIANT — `checkpoint_training_noise_pct: null`, `evaluation_noise_pct: 0` (`models.yaml:650-651`). The variant is by construction clean (frozen, out-of-the-box CLIP); `null` training noise is honest but the model has no injected noise — P3 — set `checkpoint_training_noise_pct: 0` with a note.
30. **Checkpoint acquisition is resolvable** — BLOCKED — `download_url: null` (`:647`); only a folder URL (`:648`); no digest/size (`:652-653`) — P1 — record per-variant filename + digest or keep blocked and say so.
31. **Docs' justification for the unresolved mapping is accurate** — VARIANT (refuted in part) — `AUXILIARY_ASSET_AUDIT.md:221` claims the folder "does not document which file maps to the RN50x4 noft Combiner checkpoint"; OBSERVED listing shows `FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` as the sole file and the folder README documents the keys — P1 — correct the prose; keep the model blocked on bytes.
32. **noft vs fullft mapping is verified, not assumed** — VERIFIED — OBSERVED Drive tree (`RN50x4_noft/` = one Combiner file; `RN50x4_fullft/` = `fiq_clip_…` + `fiq_comb_…`); Drive README items 2 and 6; `validate.py:283-287` (CLIP overlay only when `--clip-model-path`); CVPR22 §3.2 — NONE (positive finding) — document it as verified.
33. **Checkpoint loadable by pinned code** — UNKNOWN — `validate.py:306-307` requires `state_dict['Combiner']`; key corroborated by the Drive README and by `utils.py:127-131`; file never obtained or loaded — P1 — needs a run.
34. **Combiner architecture/dims match the checkpoint's** — UNKNOWN (static-consistent) — `combiner.py:11-31` (`clip_feature_dim=640, projection_dim=2560, hidden_dim=5120` → `output_layer: 5120→640`); adapter passes 2560/5120 — P1 — needs the real state dict.
35. **Base RN50x4 cache requirement verified** — VERIFIED — `validate.py:279` `clip.load(args.clip_model_name, …)`; `openai/CLIP` `clip/clip.py:120` `download_root or os.path.expanduser("~/.cache/clip")` (OBSERVED fetch of `raw.githubusercontent.com/openai/CLIP/main/clip/clip.py`) — NONE — none.
36. **Registry records the cache path the consumer actually reads** — VERIFIED — `auxiliary_assets.yaml:169-175`: `destination_scope: user_cache`, `cache_root_fixed: ~/.cache/clip`, with the correct explanation — NONE — none (this matches `commit 521687b fix: stage OpenAI CLIP weights in the cache CLIP actually reads`).
37. **Base weights present locally** — BLOCKED — OBSERVED `ls: cannot access '/home/heheboiz/.cache/clip': No such file or directory`; `READINESS_MATRIX.md:60` reports the same — P0 — stage with digest verification before any run.
38. **Base weight URL and expected digest recorded** — VARIANT — `auxiliary_assets.yaml:166,177` record the URL and `expected_sha256 7e526bd1…` (which equals the CDN object name, by OpenAI's convention); OBSERVED the URL returns HTTP 200, `content-length: 421854225`; `expected_size_bytes` is `null` — P2 — record the observed size; do not treat the URL-embedded name as a verified digest of downloaded bytes without hashing.
39. **Base weight digest actually verified against bytes** — UNKNOWN — not downloaded (assignment forbids ~400 MiB downloads) — P1 — verify on the PC.
40. **Dataset root resolution** — VERIFIED — `data_utils.py:11` `base_path` = checkout root; `:140,146,158,161,170,176` append `fashionIQ_dataset/…` — NONE — none.
41. **No dataset-root CLI exists** — VERIFIED — `validate.py:263-277` argparse list has no root flag — NONE — none.
42. **Workbench ignores `EvalRequest.dataset_root` for this model** — VERIFIED — `adapters/models.py:90-94` never references `request.dataset_root` — P1 — stage the link explicitly; do not add a hidden link (CF-4).
43. **Canonical dataset link present** — BLOCKED — OBSERVED `workbench/third_party/CLIP4Cir/fashionIQ_dataset` absent (not a symlink, not a dir); `READINESS_MATRIX.md:60` — P1 — operator-run `prepare_dataset.py --model combiner_rn50x4_noft`.
44. **Link staging is implemented and safe** — VERIFIED — `prepare_dataset.py:28-68`: refuses if destination exists and is not the expected link (`:58-61`), refuses if it resolves elsewhere (`:53-56`), creates only under authorization (`:65`); driven by `models_requiring_link()` over `canonical_dataset_link` declarations — NONE — none.
45. **Image extension contract matches the evaluator** — VERIFIED — `models.yaml:606` `['.jpg']`; source opens `.jpg` at `data_utils.py:158,161,170,176` — NONE — none.
46. **Images present with the declared extension** — BLOCKED — OBSERVED `images/`: 74381 `.png`, 0 `.jpg`; per-category validation ids: 3817/3817 dress, 6346/6346 shirt, 5373/5373 toptee exist as `.png`, 0 as `.jpg`; gate agrees (`READINESS_MATRIX.md:60-61`) — **P0** — stage a `.jpg`-extension tree (hardlink) or change the declaration; do not run against `.png`.
47. **Silent-skip hazard on a missing image** — VERIFIED (defect present) — `data_utils.py:151-184` `except Exception as e: print(...)` returning `None`; `utils.py:81-88 collate_fn` drops `None`; `validate.py:51`'s assert is the only downstream guard — **P0** — same action as #46; additionally, a dropped *query* row would change the denominator — P0.
48. **Preprocessing transform matches source exactly** — VERIFIED — `validate.py:289-291`; `data_utils.py:54-63,82-96` (bicubic, center crop, CLIP mean/std) — NONE — none.
49. **Target ratio 1.25 and input dim 288** — VERIFIED — `:273` default, adapter passes `1.25`; `input_dim = clip_model.visual.input_resolution` = 288 for RN50x4 (CLIP paper Table 19, OBSERVED) — NONE — none.
50. **No augmentation at evaluation** — VERIFIED — OBSERVED `grep -rn generate_randomized_fiq_caption src/*.py` → definition + training callers only; `validate.py` never calls it — NONE — none.
51. **Validation caption composition is deterministic** — VERIFIED — `validate.py:89-93` (`strip('.?, ').capitalize()` then `' and '` then stripped second caption; `np.array(captions).T.flatten()` preserves source annotation order) — NONE — none.
52. **Tokenization context length and truncation policy match** — VERIFIED — `validate.py:93` `context_length=77`, no `truncate=True` (loud failure rather than silent truncation) — NONE — none.
53. **Adapter command matches the audited invocation** — VERIFIED — `adapters/models.py:91-94` vs `README.md:258-269` — NONE — none.
54. **Adapter correctly omits `--clip-model-path` for noft** — VERIFIED — `:90-94` has no such flag; `validate.py:283-287` therefore does not overlay `'CLIP'`; fullft's adapter at `:101-109` does pass it — NONE — none.
55. **Environment spec matches the pinned README** — VERIFIED — `models.yaml:633-640` vs `README.md:113-118` — NONE — none.
56. **Model interpreter verified / command constructible** — BLOCKED — OBSERVED `workbench/envs/` has only `README.md` + `csmcir-import.txt`; `WORKBENCH_CLIP4CIR_PYTHON` unset; `runtime.py:1475-1493` raises; `python_executable` (`:1818-1825`) never fabricates a path — P1 — provision and verify the env before any run.
57. **CUDA required?** — VERIFIED (no) — `utils.py:15-18` falls back to CPU; no unguarded `.cuda()` in this model's path — P3 — record `cuda` as not required rather than UNKNOWN.
58. **Registry status fields accurate** — VERIFIED — `reproduction_status: BLOCKED_CHECKPOINT_MAPPING` (`:631`), `command_status: COMMAND_AUDITED` (`:632`), `strict_clean_checkpoint_available: null` (`:641`), `checkpoint_mapping_status: UNVERIFIED` (`:655`) all consistent with what I found — NONE — none.
59. **Metric parser exists for this model** — BLOCKED — `metrics_extraction.py:306-320` dispatches only `limn`/`csmcir`; everything else `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE` — P1 — add a parser for `validate.py:344-354` with macro recomputation.
60. **Parity check reachable once a parser exists** — VARIANT — `_check_parity` (`metrics_extraction.py:27-43`) compares `r10/r50/mean` against `paper_metrics`, which for this model would be all-`null` (`models.yaml:628-630`) → `PARITY_NOT_EVALUATED` — P2 — supply the CVPR22 row as the sanity reference if one is wanted, or accept `not evaluated`.
61. **Output artifact exists** — NOT_RUN — no evaluator process, no canonical result JSON, no per-query export; `workbench/artifacts/` holds only `environment_assignment.json` and `environment_plan.json` — P1 — nothing to compare.
62. **`run_id`-based artifact identity honoured** — VERIFIED (as policy) — workbench README operating rules ("Treat `run_id`, not `model_id`, as artifact identity"); no run exists here — NONE — none.
63. **Cross-protocol comparability guarded** — VERIFIED — `docs/FASHIONIQ_PROTOCOL_PARITY.md:20-24`, `docs/PROTOCOL_AUDIT.md:64-66` refuse cross-protocol aggregates; `resolve_fashioniq_protocol` (`src/evaluation/fashioniq.py:178-190`) raises on unknown names rather than defaulting to reference-eligible — NONE — none.
64. **Paper-vs-paper protocol consistency of the numbers this model would be compared to** — VARIANT — CLIP4Cir's FashionIQ path is reference-eligible while DCNet (`trainer_joint.py:217-218`, the baseline whose setting CLIP4Cir §4.2.1 claims to follow) masks the reference; CLIP4Cir Table 9 juxtaposes them — **P1** — do not compare `fashioniq_original_split` numbers against `fashioniq_full_gallery_ref_excluded` numbers; the workbench already enforces this, and the finding belongs in the paper-level record.
65. **Local validation split is the canonical one** — VARIANT — OBSERVED 3817/6346/5373 gallery ids and 2017/2038/1961 = 6016 triplets, vs the paper's 15,537 images / 6,017 triplets (off by one image and one triplet; FashionIQ's own paper Table 1 also lists Tops&Tees val = 5,374 vs the local 5,373) — P3 — a dataset-provenance question affecting every FashionIQ model in the workbench; unresolved here, reported as UNKNOWN, not PASS.
66. **Gallery duplicate-free and order-preserving** — VERIFIED — OBSERVED 0 duplicate ids per category; `data_utils.py:144-147` preserves file order; `utils.py` loader `shuffle` default False — NONE — none.
67. **No dead/obsolete checkpoint shim for this model** — VERIFIED — `adapters/models.py:86-94` contains only the noft command; no alias/re-export path exists that could silently route fullft→noft — NONE — none.
68. **Sibling entry not conflated** — VERIFIED — distinct `model_id` (`:656`), distinct adapter (`:97-109`), distinct status (`BLOCKED_CHECKPOINT_BUNDLE` `:696`), distinct checkpoint record (`:700-712`) — NONE — none.
69. **Verdict not upgraded past the evidence** — VERIFIED — every arrow in the trace either fails or is STATIC_AUDIT/OBSERVED-read-only; no GPU host, no interpreter, no checkpoint, no link, no `.jpg` tree — `VERIFIED`/`VARIANT` would be unsupported — NONE — none.

---

FINAL VERDICT: **BLOCKED** — the registry's static facts for `combiner_rn50x4_noft` are correct where they can be checked (pin equal to the checkout and clean, `fashioniq_original_split` with an eligible reference confirmed inside `validate.py` against its own CIRR contrast, no augmentation at evaluation, the exact TargetPad/CLIP pipeline, and the no-`--clip-model-path` noft CLI shape), and the noft-vs-fullft mapping is genuinely verified from the author's own release metadata rather than assumed; but nothing can run: the canonical root holds 74381 `.png` where the evaluator opens `.jpg` (with a silent `None`-skip path that would corrupt the ranking rather than raise), the hard-coded `<checkout>/fashionIQ_dataset` link is absent, the RN50x4 cache and the model interpreter do not exist on this host, the checkpoint is a folder URL with no file name, size or digest, and no metric parser exists for the evaluator's stdout — so no number can be produced, and the companion P1 finding is that CLIP4Cir's reference-eligible FashionIQ numbers are not cohort-comparable with the reference-excluded baselines they are tabulated against in their own paper.

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left as
written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 3 committed row(s):

  - `clip4cir_rn50x4_noft_fiq_absent_in_paper` — status `UNVERIFIED`, 0% training noise, protocol `fashioniq_original_split`, values `{'r10': None, 'r50': None, 'mean': None}`, provenance `{'r10': 'UNAVAILABLE', 'r50': 'UNAVAILABLE', 'mean': 'UNAVAILABLE'}`
  - `combiner_rn50x4_noft__clip4cir_rn50x4_fiq_gallery_protocol` — status `UNVERIFIED`, 0% training noise, protocol `fashioniq_original_split`, values `{'r10': None, 'r50': None, 'mean': None}`, provenance `{'r10': 'UNAVAILABLE', 'r50': 'UNAVAILABLE', 'mean': 'UNAVAILABLE'}`
  - `combiner_rn50x4_noft__clip4cir_paper_note_backbone_inventory_and_version` — status `UNVERIFIED`, 0% training noise, protocol `fashioniq_original_split`, values `{'r10': None, 'r50': None, 'mean': None}`, provenance `{'r10': 'UNAVAILABLE', 'r50': 'UNAVAILABLE', 'mean': 'UNAVAILABLE'}`

- **Readiness now:** `BLOCKED_ENVIRONMENT` — execution environment not verified: WORKBENCH_CLIP4CIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_CLIP4CIR_PYTHON not set)

- **Reference per registered checkpoint condition:**

  - `rn50x4_noft_unresolved` (noise None): `None` status `None` — combiner_rn50x4_noft has references but none is verified: ['UNVERIFIED']
