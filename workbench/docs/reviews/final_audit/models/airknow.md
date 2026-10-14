# airknow

**Round:** final audit, independent re-derivation. Every claim in `workbench/docs/reviews/models/airknow.md` (prior round) was treated as a hypothesis and re-checked against the pinned tree; the four committed benchmark rows were re-derived from the paper. All verdicts below are **STATIC_AUDIT** unless a command's observed output is quoted. No GPU run was performed or is possible here.

## MODEL ID

`airknow` — registry block `workbench/registry/models.yaml:93-145`. Unique in the registry (`airknow` count 1 of 16 model ids; executed parse of `models.yaml`). `method_name: Air-Know` (`:103`).

Prior-round claim "registry block `models.yaml:81-132`" is **stale**: the block now starts at `:93` and ends at `:145` (`source_blocker` at `:94`, `checkpoint_variants` at `:125`). Every line citation in the prior document is offset by −12 and the variant rows moved. Not a substantive defect; a documentation-accuracy defect (P3).

## PAPER TITLE

`Air-Know: Arbiter-Calibrated Knowledge-Internalizing Robust Network for Composed Image Retrieval` (`models.yaml:107`).
- `read https://arxiv.org/abs/2604.19386` → HTTP 200 via the arXiv API, title identical; authors **Zhiheng Fu, Yupeng Hu, Qianyun Yang, Shiqi Zhang, Zhiwei Chen, Zixu Li**; published 2026-04-21; category cs.CV.
- `read https://arxiv.org/html/2604.19386v1` → full HTML; affiliation stack "Shandong University", author link `<https://github.com/ZhihFu/Air-Know/>`.
- `workbench/third_party/AirKnow/README.md:5` (title `[CVPR 2026] Air-Know: …`), `:7-29` (same six authors, same affiliation). **MATCH.**

## PINNED SOURCE COMMIT

| Claim | Observed | Verdict |
|---|---|---|
| registry SHA == checkout HEAD | `git -C workbench/third_party/AirKnow rev-parse HEAD` → `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` == `models.yaml:111` | **VERIFIED** |
| remote == `upstream_repo_url` | `git config --get remote.origin.url` → `https://github.com/iLearn-Lab/CVPR26-Air-Know.git` == `:109` | **VERIFIED** |
| `upstream_default_branch: main` (`:110`) | `git branch -a --contains HEAD` → `* main`, `remotes/origin/HEAD -> origin/main` | **VERIFIED** |
| pin is current upstream `main` | `curl -s "https://api.github.com/repos/iLearn-Lab/CVPR26-Air-Know/commits?sha=main&per_page=3"` → newest commit `6ab6fa95b746… 2026-09-28T06:40:56Z` = the pin | **VERIFIED** (prior document called this UNKNOWN because no fetch was permitted; a read-only HTTP GET is permitted and settles it) |
| no modified tracked file | `git status --porcelain` → 846 entries; `grep -vc __pycache__` → **0**; `runtime.py:1829-1837` classifies `__pycache__/*.pyc` as non-source | **CLEAN** (bytecode-only) |
| repo identity | `curl -sI https://api.github.com/repos/ZhihFu/Air-Know` → `301 location: https://api.github.com/repositories/1185360679`; that id resolves to `full_name iLearn-Lab/CVPR26-Air-Know, default_branch main`; `curl -sIL https://github.com/ZhihFu/Air-Know` → effective URL `https://github.com/iLearn-Lab/CVPR26-Air-Know`, HTTP 200 | **VERIFIED** — the paper's `ZhihFu/Air-Know` and the registry's `iLearn-Lab/CVPR26-Air-Know` are one repository. The extracted payload's own note (`paper_benchmarks/extracted/airknow.json`, `paper.research_note`: "I could not confirm that") is **too pessimistic**: a GitHub repo *rename* answers `301` from the API and `404` after it is garbage-collected, and `ZhihFu/Air-Know` answers `301`, not `404`. |
| single-commit checkout | `git rev-list --count HEAD` → `1`; upstream `main` history is README-only commits | **VERIFIED** (upstream publishes code + README; no second commit to compare against) |

## CHECKPOINT IDENTITY

STATIC_AUDIT + HTTP metadata. No weight downloaded, opened, hashed or loaded; `workbench/artifacts/checkpoints/` contains only `.gitkeep`.

| checkpoint_id | filename | training noise | eval noise | size | sha256 | status | mapping |
|---|---|---|---|---|---|---|---|
| `fiq_n05` | `airknow_fiq_0.5.pt` (`:127`) | 50 (`:130`) | 0 (`:131`) | `null` (`:132`) | `null` (`:133`) | `NO_CLEAN_CHECKPOINT` (`:134`) | `VERIFIED_METADATA` (`:135`) |
| `fiq_n08` | `airknow_fiq_0.8.pt` (`:137`) | 80 (`:140`) | 0 (`:141`) | `null` (`:142`) | `null` (`:143`) | `NO_CLEAN_CHECKPOINT` (`:144`) | `VERIFIED_METADATA` (`:145`) |

Independent Hub check (executed `curl`):
```
GET https://huggingface.co/api/models/iLearn-Lab/CVPR26-Air-Know/tree/main?recursive=true
  .gitattributes      1519 B
  README.md           5285 B
  airknow_fiq_0.5.pt  2724056638 B  lfs.oid f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6
  airknow_fiq_0.8.pt  2724056638 B  lfs.oid 756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e
```
Exactly two weight files, named `0.5` and `0.8`; no `0.0`, no `0.2`. `strict_clean_checkpoint_available: false` (`:123`) is **correct**. The registry note (`:124`, "Official Hub lists only FashionIQ 50% and 80% noise-trained weights") asserts *FashionIQ* — the Hub associates no file with a dataset, a category set or a noise level; only the filename does (UNRESOLVED #1).

Executed `select_checkpoints`:
```
strict_clean                -> BLOCKED / BLOCKED_CHECKPOINT
  reason: no checkpoint is recorded as clean-trained under policy strict_clean: fiq_n05=50%, fiq_n08=80%.
include_flagged_conditions  -> BLOCKED / BLOCKED_CHECKPOINT_SELECTION
  reason: more than one source-backed variant is registered and none is declared a bundle: fiq_n05, fiq_n08.
```
`checkpoint_availability(airknow, fiq_n05/fiq_n08)` → `checkpoint_downloaded: False`, `source_metadata: True`, `automatic_download_available: True`, `adapter_command_ready: False` (`registry.py:714-719`: `command_status` absent → falls through to "adapter does not override `command`", which is true, `adapters/models.py:54-56`). Both variants are `NO_CLEAN_CHECKPOINT` although `checkpoint_mapping_status: VERIFIED_METADATA`; `registry.py:84` separately declares `evaluation_noise_pct != 0` an error, so the eval condition is pinned to 0 by construction.

## EVALUATION PROTOCOL

`native_protocol: fashioniq_full_gallery_ref_excluded` (`:117`), `supported_protocols: [same]` (`:118`), `literature_split_label: original` (`:116`).

1. **The evaluator is a bare callable.** `test_BLIP2.py:9` `def test(params, model, testset, category, txt_processors)`; categories dispatched at `:81-96`. `grep -n "__main__\|argparse\|ArgumentParser" test_BLIP2.py` → **no match** (rc=1). Sole callsite `train_BLIP2.py:298` inside `train_and_evaluate` (`:268-337`), reached from `train_BLIP2.py:340 __main__` → `:362`. **No standalone entrypoint.**
2. **Gallery = full ordered validation split.** `datasets.py:31` `self.split = 'original-split'` (hardcoded; `__init__` `:29` takes only `path, transform, noise_ratio`). `get_test_data` (`:181-224`) reads `image_splits/split.{name}.val.json` (`:183`), and at `:218-224` (`elif self.split == 'original-split':`) enumerates **every** index in file order. The `val-split` pair-union branch (`:206-217`) is unreachable dead code.
3. **Reference masked before ranking.** `test_BLIP2.py:147-151` builds `test_targets_id` from `target_img_id` and writes `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`; the sort is `:154 np.argsort(-sims[i, :])[:50]`. **Mask, not column removal.** `docs/FASHIONIQ_PROTOCOL_PARITY.md:70` records the same. The workbench deliberately removes the column (`src/evaluation/fashioniq.py:35-77`, docstring `:42-53`).
4. **Recall mathematics.** `:168` `argsort(-sims[i,:])[:50]`; `:169` `for k in [1, 10, 50]`; `:172` hit test; `:174` `r = 100 * r / len(nn_result)`, denominator = query count. **CONFIRMED** (R@1 is also computed and returned, `:175`).
5. **Sort/ties.** `-sims` descending; numpy's default unstable sort, unpinned by source and declined by the workbench (`FASHIONIQ_PROTOCOL_PARITY.md:47-48`). Measure-zero for `F.normalize` features (`blip2_cir.py:499-500`, `:525`). **UNKNOWN-by-design, not a defect.**
6. **The withdrawn tensor-shape claim is genuinely wrong; re-derived independently.** `:142-144` `np.matmul(all_queries, all_imgs)` → `.squeeze()` → `.max(-1)`. `blip2_cir.py:499-503` returns `fusion_feats.unsqueeze(1).unsqueeze(1)` with `fusion_feats` of dim `embed_dim=256` (`:109,135-136`) → `(N,1,1,256)`; `:506,525-526` returns `image_features.permute(0,2,1)` → `(M,256,32)` (`num_query_token=32`, `blip2_config:21`). Executed numpy simulation: `matmul` → `(5,7,1,32)`, `squeeze` → `(5,7,32)`, `max(-1)` → `(5,7)`; `sims[2,4] = -10e10` writes through, `argsort(-s[0,:])[:3]` works. The registry's withdrawal note (`models.yaml:94`, last sentence) is **CORRECT**.

## DATASET CONTRACT

`fashioniq_layout: fashioniq_ilearn_resized` (`:114`), `fashioniq_image_extensions: [".jpg"]` (`:115`), `preparation_contract: manual_ilearn_resized` (`:106`). Source root is string-concatenated (`datasets.py:33-35`), so the caller's `--fashioniq_path` MUST end with a separator — unrecorded in the registry.

| Input | Source callsite | Layout checker |
|---|---|---|
| `image_splits/split.{cat}.val.json` | `datasets.py:183` | `fashioniq_layout.py:33` |
| `captions/cap.{cat}.val.json` | `datasets.py:185` | `fashioniq_layout.py:31` |
| `captions/correction_dict_{cat}.json` | `datasets.py:110` (train), `:187` (val) | `fashioniq_layout.py:32` |
| `captions/cap.{cat}.train.json` | `datasets.py:108` (cache-miss branch) | only in `ilearn_resized_training_paths` (`:37-41`) |
| `resized_image/{cat}/{id}.jpg` | `datasets.py:155` (`get_img`), `:166` (`get_test_img`) | `fashioniq_layout.py:29-30`, `:85`, `:92-136` |
| `fashion_iq_data.json` + 6 PKLs | `datasets.py:38-62` | **not checked** |
| `judge/judge.train.{noise_ratio}.json` | `datasets.py:238` | **not checked** (training only) |

**Local readiness (executed).** `preflight_blockers(airknow, resolve_config())` → **10 blockers**: `source cannot run: …` (from `models.yaml:94` via `preflight.py:198-204`), 8 dataset/extension blockers, and `command not runnable: COMMAND_UNAUDITED`. `ls data/fashionIQ_dataset` → `captions image_splits images qwen_captions`; `images/` holds 74 381 PNGs; `captions/` holds 9 `cap.*.json` and **0 `correction_dict_*`**; no `resized_image/`, no `fashion_iq_data.json`, no `*.pkl`, no `judge/`. `docs/READINESS_MATRIX.md:26` records the same. **BLOCKED (dataset + source).**

Counts read from disk: dress 3817 gallery / 2017 queries; shirt 6346 / 2038; toptee 5373 / 1961; gallery files duplicate-free (`len(d)==len(set(d))`); `set(refs) ⊆ gallery` and `set(targets) ⊆ gallery` → `True` ×3; `cap.*.val.json` records have keys `{candidate, captions, target}`. Not cross-checked against the official FashionIQ release (UNRESOLVED #4).

**Gap A (P2) — unchecked, unversioned cache.** `datasets.py:38` gates on `os.path.exists('fashion_iq_data.json')`; on the hit path (`:55-62`) the JSON and six PKLs are loaded and the split/caption/correction files are never re-read. A stale cache is trusted (`docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:51-53`).
**Gap B (P2) — resized-image staging recipe missing from the registry.** `workbench/registry/preparation_contracts.yaml` has **0** `airknow` rows (grep count 0), so no `fashioniq_resize_images` or `pinned_source_file` step reaches preflight for this model — yet `workbench/docs/AUTOMATED_DATA_PREPARATION.md:18-25` states a four-step contract including "resized JPEGs, derived from the canonical images". The pinned AirKnow tree ships **no** `correct_files/` and **no** resize script (only `workbench/third_party/LIMN/correct_files` exists under any checkout), so the doc's evidence phrase "the author ships `correct_files/correction_dict_<cat>.json`" is **false for airknow** — `grep -n -i "resize\|start_kit\|correct" AirKnow/README.md` → only `--batch_size 256` lines. Registry-directory row is honest (`PC_TRANSFER_CHECKLIST.md:56`: "not shipped and not published"); the automated doc generalises across models and is WRONG for this one (P2).
**Gap C (P2) — the EKI anchor dataset is absent everywhere.** `judge/judge.train.{noise_ratio}.json` (`:238`), consumed by `FashionIQ_V` (`:227-263`) and instantiated by `load_EKI_dataset` (`train_BLIP2.py:84-107`, `:90-94`), exists nowhere: not tracked, not in `preparation_contracts.yaml` (0 rows), not in `fashioniq_layout.py` (0 mentions), not local. Training cannot be reproduced; evaluation is unaffected.

## PREPROCESSING CONTRACT

Registry matrix says caption stage `EQUIVALENT_WITH_EVIDENCE (pre-LAVIS string stage only)`, image stage `EQUIVALENT_WITH_EVIDENCE` (`SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:27`).

**Stage 1** (`datasets.py:122-131`): `correct_text` lowercases → translates **all** `string.punctuation` to spaces → `.strip().split()` → exact per-token `correction_dict` lookup → rejoin; `concat_text` joins `cap[0] + " and " + cap[1]` in source order. Applied at `:192` (val) and `:115` (train).
**Stage 2** (`test_BLIP2.py:112` `txt_processors["eval"](caption)`): `lavis/processors/blip_processors.py:28-37` class `BlipCaptionProcessor`, `pre_caption` `:49-68` lowercases, replaces `[.!\"()*#:;~]` (a **narrower** set than `string.punctuation`), collapses whitespace, truncates at `max_words=50` (`:30,45`). Selected by `lavis/configs/models/blip2/blip2_pretrain.yaml:32-36` (`name: blip_caption`, eval).
**Stage 3:** `blip2_cir.py:480-486` BERT tokenizer `padding="max_length", truncation=True, max_length=self.max_txt_len`, `max_txt_len=32` (`:110,140`). Tokenizer literal `/root/bert-base-uncased` at `lavis/models/blip2_models/blip2.py:32` (also `:48`, `:55`) — **no registry record, no fallback**.
**Image:** `train_BLIP2.py:111-119` `transform = "targetpad"`, `input_dim=224`, `target_ratio=1.25` → `data_utils.py:92-106` (`TargetPad(1.25,224)` → `Resize(224, BICUBIC)` → `CenterCrop(224)` → RGB → ToTensor → CLIP mean/std). `TargetPad.__call__` (`:64-73`) pads only when `actual_ratio >= 1.25`. Eval and query paths share one transform instance (`datasets.py:154-173`); there is **no eval-specific transform** in the source.

## AUGMENTATION CONTRACT

`FashionIQ.shuffle()` (`datasets.py:66-101`) is the only augmentation hook: `:68` early-returns for `noise_ratio == 0`; `:73` samples `int(noise_ratio*N)` indices; `:74-78` splits them into candidate/caption/target thirds; `:87,:89-90` flip `labels`/`qlabels`; `:92-97` shuffle and write back. Matches the paper §3.1 ("simulate noisy correspondence by randomly shuffling the elements of a certain proportion of samples within the training set").

Callsites: `grep -rn "\.shuffle()"` → `train_BLIP2.py:130` (FashionIQ), `:143` (CIRR). `grep -c "noise_ratio" test_BLIP2.py` → **0**. **No evaluator calls it; training noise is never evaluation noise.** Corruption on the eval path: none found — `get_test_data` reads raw val annotations, and the cache substitutes the identical records the same function wrote. `FashionIQ_V` (the label-bearing sibling) is instantiated only by `load_EKI_dataset` (`train_BLIP2.py:84-107`) and consumed only by `v_trainloader` (`:275-279`, `:291`); `test.test` receives `trainset` (`:298`). **CONFIRMED.**

## GALLERY CONTRACT

Full ordered `split.{category}.val.json`, reference masked per query before ranking (`datasets.py:31`, `:218-224`; `test_BLIP2.py:147-151`, `:154`). Workbench expression: `FASHIONIQ_PROTOCOLS['fashioniq_full_gallery_ref_excluded']` = `gallery='full_split', reference_excluded=True` (`src/evaluation/fashioniq.py:175-177`); `build_fashioniq_gallery` (`:204-206`) → `build_original_gallery` (`:121`); `recall_at_k_reference_excluded` (`:35-77`) removes the column, documented as a divergence from the mask. **Parity is definitional, not demonstrated on Air-Know scores** (UNRESOLVED #3).

No `airknow` module exists under `workbench/replay/` (`ls` → `common.py limn.py tgcir.py`); the full-gallery helpers are reached only from `tests/unit/test_fashioniq_protocols.py` and `workbench/tests/test_protocols.py`.

## METRIC CONTRACT

**Source output shape.** `test()` returns `[(name, value)]` per category (`:175`), no JSON and no aggregate. The trainer sums the three R@10 values as its selection score (`train_BLIP2.py:300` `current_score = current_score + t[1][1]` — a **sum**, not a mean) and persists a flat map of all nine keys (`:315` `utils.save_dict_to_json(test_metrics, "metrics_best.json")`); `:303`,`:319`,`:333` write `model_epoch_{epoch}.pt`, `:366` writes `best_model.pt`.
**Workbench parser: none.** `metrics_extraction.py:309` `extract_aggregate_metrics` dispatches only `limn` (`:328`) and `csmcir` (`:330`); `airknow` falls through to `extraction_status="AGGREGATE_PARSER_UNAVAILABLE"` (`:335`), `parity_status="PARSER_UNAVAILABLE"` (`:340`), `observed_metrics=None`. Asserted at `workbench/tests/test_adversarial_metrics.py:98-101` (no fabrication) and `:104-108` (paper value never substituted for observed). **No `airknow` number can be extracted even if the evaluator ran** (P1).
**Tolerance now single-sourced.** `_check_parity` (`metrics_extraction.py:28-61`) still defaults `tolerance: float = 0.05` (`:31`) but the value is **dead**: `:57` routes every comparison through `paper_gate.compare_metric` and `:58` tests `comparison.within_tolerance`. `paper_gate.TOLERANCE_PCT = 5.0` (`paper_gate.py:27`). No caller passes `tolerance`. The prior document's MF-5 ("two tolerance regimes") is **superseded**; what remains is one dead defaulted parameter (P3, cosmetic).

## SOURCE-TO-WORKBENCH PARITY

| Component | Registry/doc claim | My check | Verdict |
|---|---|---|---|
| Protocol | `fashioniq_full_gallery_ref_excluded` | `datasets.py:31,218-224`; `test_BLIP2.py:147-151,154` | **EQUIVALENT** (mask vs removal) |
| Caption stage | `EQUIVALENT (pre-LAVIS string stage only)` | `datasets.py:122-131`; `blip_processors.py:49-68` | **EQUIVALENT for stage 1**; stage 2 is LAVIS-internal and never exercised |
| Image stage | `EQUIVALENT_WITH_EVIDENCE` | `data_utils.py:92-106` | **EQUIVALENT (static)** |
| Preparation contract | `EXTERNAL_ASSET_REQUIRED` | no `airknow` rows in either registry; 3 of 4 items absent locally | **CONFIRMED GAP** |
| Evaluator command | `NOT IMPLEMENTED` | `adapters/models.py:54-56`; executed → `NotImplementedError` | **CONFIRMED** |
| Overall | `INCOMPLETE_IMPLEMENTATION` | — | **AGREE** |
| Registry asset audit | `cluster_b3_ilearn.yaml:4-40` | two of three recorded `sha256` values are **stale/wrong**: `AirKnow/data_utils.py` actual `c6a90c6d7a57bbcc9604e0fb2b24875289909a6d8b727dfcb11597660e1843ca` vs recorded `c6a90c6d7a57a8bf6ff00874ff0b497042a98f121d5c21f7dbbe03ec417430d8`; `AirKnow/utils.py` actual `694a5081a6eb81ef29607d50d9a8c78335a59e43968b5e9e58ff84421833a85f` vs recorded `694a5081a6eb6408d66e746a5d42188ff6e6cf1e4a5bf4d289aa841ee135cf50` (matching size/line counts) | **WRONG (P3)** — no runtime consumer hashes these, but the evidence field is not evidence |
| Same file, `fork_diff_evidence` line refs | "blip2.py lines 102-103" / "eva_vit.py line 443" | actual: `blip2.py:97-99`; `eva_vit.py:443` correct | **1 of 2 off by 4 lines (P3)** |

## RUNTIME STATUS

- **Environment (registry):** `WORKBENCH_AIRKNOW_PYTHON`, `VERIFIED_README`, python `3.8.10`, torch `2.1.0`, torchvision `0.16.0`, cuda `12.6`, packages `scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16` (`models.yaml:95-102`). Upstream: `README.md:107` (Python 3.8.10 / CUDA 12.6), `:114` (torch 2.1.0 / tv 0.16.0 cu121), `:117` (the four pins verbatim). `README.md:110` says `conda create -n conesep` — a sibling's env name copy-pasted into this README (cosmetic upstream defect).
- **Not installed.** `workbench/artifacts/environment_plan.json`: `interpreter_mapping["airknow"] → None`; the containing group has `conda_env_path .../artifacts/environments/cir-4c8b682368302099`, `members ['airknow','conesep','habit','hint']`, `installation_status: NOT_REQUIRED`, solver `conda PARTIAL`. `ls workbench/artifacts/environments` → **No such file or directory**. `READINESS_MATRIX.md:14-19` → `ENVIRONMENT_INSTALLED: 0`, `RUNTIME_VERIFIED: 0`, `SCIENTIFICALLY_REPRODUCED: 0`.
- **Adapter:** `AirKnowAdapter` (`adapters/models.py:54-56`) names `script = "test_BLIP2.py"` only. Executed: `cls.command(...)` → `NotImplementedError: upstream evaluation command requires audit before execution` (`adapters/models.py:25`); `cls.command is OfficialScriptAdapter.command` → `True`. `command_status` absent from the registry block, so `checkpoint_availability` reports `adapter_command_ready: False`.
- **Readiness state:** executed `preflight_blockers` → 10 blockers; `pc_readiness.classify` → **`BLOCKED_SOURCE`**, first blocker `source cannot run: …` (ordering at `pc_readiness.py:44-71`). `READINESS_MATRIX.md:26` shows `BLOCKED` with the dataset list.
- **Result records:** none. Only `workbench/artifacts/results/fashioniq_full_gallery_ref_excluded/mock/mock_airknow-fiq_n05.json` with `run.data_kind: "mock"`, `checkpoint_sha256: null`, `command_digest: null`, `model_id: "mock_airknow"`, synthetic `reproduced_metrics {r10: 12.5, r50: 20.833, mean: 16.667}`. `paper_gate.evaluate_gate` returns `NOT_RUN` for `data_kind != "experiment"` (executed below). Not evidence.
- **GPU inference: not attempted, none available. Every claim here is STATIC_AUDIT.**

## PAPER-METRIC STATUS

The paper prints a category-level FashionIQ table (Table 1, page 6) with a Noise-block column: rows `0%`, `20%`, `50%`, `80%`. Air-Know's row in each block, read from `https://arxiv.org/html/2604.19386v1`:

| noise block | Dress R@10/R@50 | Shirt | Toptee | Avg R@10 | Avg R@50 | AVG. |
|---|---|---|---|---|---|---|
| 0% | 51.03/73.08 | 56.76/75.05 | 59.47/79.02 | **55.75** | **75.72** | **65.73** |
| 20% | 50.22/73.48 | 56.38/74.73 | 58.95/78.94 | 55.18 | 75.71 | 65.45 |
| 50% | 47.90/71.09 | 53.95/73.09 | 57.71/77.94 | 53.19 | 74.04 | 63.61 |
| 80% | 42.52/66.03 | 51.31/70.26 | 53.86/74.41 | 49.23 | 70.23 | 59.73 |

Observed HTML lines 194 / 199 / 204 / 209 (rowspans: 0% row 189, 20% row 195, 50% row 200, 80% row 205). Arithmetic re-executed from the \*category cells\*, round-half-up:
```
0%  macro10=55.7533 -> 55.75 = printed ; macro50=75.7167 -> 75.72 = printed ; mean=65.735 -> 65.73 = printed
20% macro10=55.1833 -> 55.18 = printed ; macro50=75.7167 -> 75.72, but the paper PRINTS 75.71 in the Average R@50 cell
50% macro10=53.1867 -> 53.19 = printed ; macro50=74.0400 = printed ; mean=63.6133 -> 63.61 = printed
80% macro10=49.2300 = printed ; macro50=70.2333 -> 70.23 = printed ; mean=59.7317 -> 59.73 = printed
```

### Registry rows re-checked (4 rows, `papers.yaml`)

| reference_id | line range | reported | categories | condition | checkpoint mapping | status | agreement |
|---|---|---|---|---|---|---|---|
| `airknow_fiq_0pct` | `:10-68` | `{r10 55.75, r50 75.72, mean 65.73}` (`:36-38`) | 51.03/73.08, 56.76/75.05, 59.47/79.02 (`:41-49`) | 0%, `checkpoint_id: null` (`:19-20`) | `NO_CHECKPOINT` | `VERIFIED`/`CONFIRMED` | **digit-for-digit** |
| `airknow_fiq_20pct` | `:69-115` | `{55.18, 75.71, 65.45}` (`:91-93`) | 50.22/73.48, 56.38/74.73, 58.95/78.94 (`:96-104`) | 20%, `null` (`:77-78`) | `NO_CHECKPOINT` | `VERIFIED`/`PARTIAL` | **digit-for-digit** |
| `airknow_fiq_50pct` | `:116-161` | `{53.19, 74.04, 63.61}` (`:138-140`) | 47.90/71.09, 53.95/73.09, 57.71/77.94 (`:143-151`) | 50%, `fiq_n05` (`:124-125`) | `VERIFIED` | `VERIFIED`/`CONFIRMED` | **digit-for-digit** |
| `airknow_fiq_80pct` | `:162-207` | `{49.23, 70.23, 59.73}` (`:184-186`) | 42.52/66.03, 51.31/70.26, 53.86/74.41 (`:189-197`) | 80%, `fiq_n08` (`:170-171`) | `VERIFIED` | `VERIFIED`/`CONFIRMED` | **digit-for-digit** |

Cross-checks: `paper_benchmarks/extracted/airknow.json` and `verification/VerifyPapersA.json` carry the same values and the same verdicts (`0pct CONFIRMED`, `20pct PARTIAL`, `50pct CONFIRMED`, `80pct CONFIRMED`); the built `papers.yaml` drops `derived_metrics` (`derived= []` for all four rows), so the extracted payload's suspicious derived `macro_r50_from_categories = 75.72` for the 20% row — which the paper's own Average cell contradicts (75.71) — **never reaches runtime**. `registry_version: extract-251fa9c32998` (`papers.yaml:8`) is consistent with the runtime expectation at `papers.yaml:1-5`.

**Registry headline metrics.** `models.yaml:119-121` now reads `reported_r10: 55.75`, `reported_r50: 75.72`, **`reported_mean: 65.73`**. The prior document's MF-1 ("`reported_mean: 65.74` … off by 0.01 pp") is **superseded and now false**: the value was corrected to the paper's `65.73` after that audit. Re-verified: the paper and its own `assets/results-fiq.png` print `65.73`. `reported_r10`/`reported_r50` transcribe the **0%-noise** block exactly.

**Comparability (executed, `paper_gate`).** Reference selected by condition, not by metric (`benchmark_references.py:145-190`):
```
select_reference("airknow", checkpoint_training_noise_pct=50, protocol=…) -> airknow_fiq_50pct | exact condition match
select_reference("airknow", checkpoint_training_noise_pct=80, protocol=…) -> airknow_fiq_80pct | exact condition match
select_reference("airknow", checkpoint_training_noise_pct=0,  protocol=…) -> airknow_fiq_0pct  | exact condition match
evaluate_gate(80pct row, observed = row values, meta noise 80, data_kind="experiment")
  -> PASS_WITHIN_5_PERCENT, detail: six category metrics published by the paper but not observed -> not compared
evaluate_gate(80pct row, observed r10=70.0, meta noise 80)            -> ALERT_METRIC_DEVIATION
evaluate_gate(80pct row, observed = row values, meta noise 0, ckpt None) -> BLOCKED_CHECKPOINT_MISMATCH
evaluate_gate(80pct row, observed r50 only impact, cats {dress:{r10,r50}}) -> dress compared; shirt/toptee reported unchecked
evaluate_gate(50pct row, … meta declared_variant/caption_alias_staged True) -> PASS_WITHIN_5_PERCENT_VARIANT (variant=True)
evaluate_gate(50pct row, …, data_kind="mock")                          -> NOT_RUN
```
So the prior document's **CF-3 is now partly superseded**: the gate is genuinely wired end-to-end (reference selection by condition in `evaluate_models.py:150-154`, `run_meta` constructed at `:137-148`, per-category comparison at `paper_gate.py` (`CATEGORY_KEYS` `:35-38`), `benchmark_gate` and `variant_provenance` carried into the report at `evaluate_models.py:1427` and the autopilot row at `autopilot.py:844-845`). A 50%-checkpoint run can no longer be judged against the 0% row. **What remains open (P2):** the registry entry still exposes the 0%-noise headline (`reported_r10/r50/mean`) with **no** condition field and no link to a reference row (`grep -rn "paper_row_noise_pct" workbench/` → 0 hits outside the two review docs), and `autopilot.py:620-622` surfaces those three floats on **every** row including `fiq_n05`/`fiq_n08` rows, with the measured difference computed against them at `:814-828`. The gate cannot be fooled; a *reader* of a row can.

**No reproduction attempted** — `reproduction_status: NOT_RUN` (`:122`). **PARITY NOT EVALUATED.**

## CRITICAL FINDINGS

**CF-1 (P1, blocker) — the pinned `lavis.models` package cannot be imported, and this defect is AirKnow's alone.**
`workbench/third_party/AirKnow/lavis/models/__init__.py:33-34`
```python
33  from lavis.models.blip_models.blip_cir import BlipCir
34  from lavis.models.blip_models.blip_cir_caption import BlipCirCaption
```
Both leaf modules are **absent** from the pinned tree at every level:
```
$ ls lavis/models/blip_models/ | grep -i cir      -> (no match)
$ git ls-files lavis/models/blip_models/ | grep -i cir
  lavis/models/blip_models/.ipynb_checkpoints/blip_cir-checkpoint.py
  lavis/models/blip_models/__pycache__/blip_cir.cpython-{37,38,39,312}.pyc
  lavis/models/blip_models/__pycache__/blip_cir_caption.cpython-{37,38,39,312}.pyc
```
**Independent confirmation the defect exists upstream, not just in this checkout** (read-only GitHub API, pinned ref):
```
GET https://api.github.com/repos/iLearn-Lab/CVPR26-Air-Know/contents/lavis/models/blip_models?ref=6ab6fa95b746… 
  -> .ipynb_checkpoints, __init__.py, __pycache__, blip.py, blip_caption.py, blip_classification.py,
     blip_feature_extractor.py, blip_image_text_matching.py, blip_nlvr.py, blip_outputs.py,
     blip_pretrain.py, blip_retrieval.py, blip_vqa.py, nlvr_encoder.py
```
No `blip_cir*`. Bytecode cannot substitute: `.pyc` is only consulted inside a **package's own** `__pycache__`, never for a leaf module, and CPython resolves the parent package before the leaf (observed live):
```
$ mkdir -p synth/pkg; echo x=1 > synth/pkg/__init__.py; echo y=2 > synth/pkg/leaf.py
$ python3 -c "import sys;sys.path.insert(0,'synth');import pkg.leaf;print(pkg.leaf.y)"  -> 2      (source present)
$ rm synth/pkg/leaf.py
$ python3 -c "import sys;sys.path.insert(0,'synth');import pkg.leaf"
  ModuleNotFoundError: No module named 'pkg.leaf'
```
Package import attempt against the pinned tree (the only interpreter available, 3.10/3.14; the missing dep fires first and is **not** the blocker): `python3.10 -c "import lavis.models"` → `ModuleNotFoundError: No module named 'omegaconf'` raised from `lavis/__init__.py:11`, i.e. before the AirKnow-specific lines. Static resolution of the two imports:
```
resolve lavis.models.blip_models.blip_cir        -> package=False source/bytecode_at_leaf=False
resolve lavis.models.blip_models.blip_cir_caption-> package=False source/bytecode_at_leaf=False
```
So at `:33` the loader raises `ModuleNotFoundError`; `:40 from lavis.models.blip2_models.blip2_cir import Blip2QformerCir` — the class this evaluator needs (`Blip2QformerCir` is registered inside `blip2_cir.py:80-81`, and that file **is** tracked) — never executes.
**Contrast (executed):** `ConeSep`, `HABIT`, `INTENT` carry the **same two import lines** at the same numbers *and* each ship the file (`ls ConeSep|HABIT|INTENT/lavis/models/blip_models/blip_cir.py` → all exist). The missing files are AirKnow's alone.
Call chain: `train_BLIP2.py:28 from lavis.models import load_model_and_preprocess` executes at module import, so `AirKnowAdapter.script = "test_BLIP2.py"` (`adapters/models.py:56`) is unreachable as written. P1 rather than P0: it yields **no number**, not a wrong one. **No `workbench/patches/airknow*` exists** (`ls workbench/patches/` → `csmcir.md`, `csmcir_models_init_compat.patch`), and `models.yaml:94` correctly declares this as `source_blocker` (so the preflight's `source cannot run:` blocker is honest), but the *stated reason* in that string — "the pinned evaluator is a callable with no CLI or checkpoint loader, the adapter deliberately declines to build a command, and the required correction dictionaries and resized images have no published source" — **does not name CF-1**, the only hard blocker. A reader of the registry sees the softer reasons and not the unusable package (P2).
*Action:* commit/vendor the two modules (or guard the two imports) and re-derive whether `Blip2QformerCir` loads; then extend `models.yaml:94` to name the import failure.

**CF-2 (P1) — the FashionIQ evaluator runs with the model in training mode.**
`test_BLIP2.py:9-11`: `def test(...)` → `model.train()` (`:10`) → `with torch.no_grad():` (`:11`). The `fashion200k` branch opts out (`:13 model.eval()`); the FashionIQ branch (`:81-175`) never does. `no_grad()` disables gradients, not dropout. Live dropout: `blip2_cir.py:42-63` class `Proxy` with `nn.Dropout(p=0.1)` at `:58,:61`, built at `:146-149` with `dropout_rate=0.1`; the Q-Former's `BertConfig` (`blip2.py:48`, `:55`) inherits BERT-base dropout defaults. **Every sibling calls `model.eval()`**: `ConeSep/test.py:10`, `HABIT/test.py:10`, `INTENT/test.py:10` (executed). Independently recorded at `registry/assets/cluster_b3_ilearn.yaml` (`fork_diff_evidence`, first bullet), `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:144-148`. Effect: non-deterministic, unlabelled protocol divergence. The registry block records it **nowhere**.
*Action:* replay verbatim (and say so), or declare a VARIANT. The gate's `declared_variant` flag (`paper_gate.py:83`) exists for exactly this and is never set for airknow.

**CF-3 (P1) — no runnable end-to-end path, in three independent places.**
(a) package import dies at `lavis/models/__init__.py:33` (CF-1).
(b) `test_BLIP2.py` has no `__main__`/argparse (grep → no match), and the only script that calls it has no weight-loading path: `--restore_file` is declared at `train_BLIP2.py:67` and **never read** (`grep -n restore` → the declaration only); `grep -rn "torch.load|load_state_dict|load_checkpoint"` limited to `train_BLIP2.py`/`test_BLIP2.py` → no match; the trainer only writes (`:303`,`:319`,`:333`,`:366`). The two generic loaders that do exist are never reached by the evaluator path — `utils.py:128-143 load_checkpoint` has **0** callsites, `lavis/models/__init__.py:112 model.load_checkpoint(checkpoint)` is only invoked when `load_model(..., checkpoint=…)` is passed one, and neither `train_BLIP2.py:196` nor `eval` passes one. The only loader in the tree is the **CIRR** path `cirr_sub_BLIP2.py:125 torch.load(..., weights_only=False)`, whose input is a directory of `*.pt` files produced by this repo's own trainer and whose dataset is hardcoded (`:117` `/root/autodl-tmp/cirr_data/CIRR/`). **There is no loader for `airknow_fiq_*.pt`.**
(c) `AirKnowAdapter.command` raises (executed).
*Action:* write and audit a standalone FashionIQ entrypoint that loads the published checkpoint (`Blip2QformerCir` construction + a checkpoint-compat path), then register `command_status`.

## MINOR FINDINGS

**MF-1 (P2)** `models.yaml:94` `source_blocker` names three reasons but not the import failure (CF-1), which is the only one that cannot be worked around. Recommend folding CF-1 into the string.
**MF-2 (P2)** `reported_r10/r50/mean` (`:119-121`) are the 0%-noise row with no condition field; `autopilot.py:620-622` prints them into every row and `:814-828` computes `measured_difference` against them, with no per-condition reference row attached. The gate is immune; the row reader is not. Add `paper_row_noise_pct` (or always attach `benchmark_gate.selection_reason`).
**MF-3 (P2)** Hardcoded `/root/bert-base-uncased` (`blip2.py:32,48,55`), no registry record, no fallback; `BertConfig.from_pretrained` (`:48`) pins no revision.
**MF-4 (P2)** `preparation_contracts.yaml` has 0 `airknow` rows, so the resized-image staging recipe never reaches preflight, while `AUTOMATED_DATA_PREPARATION.md:18-25` claims the author ships correction dictionaries and a documented `start_kit/resize_images.py`; neither exists in the AirKnow checkout. The doc generalises across the iLearn family and is factually wrong for this model.
**MF-5 (P2)** `docs/UPSTREAM_AUDIT.md:25` records a DQU-CIR provenance **hypothesis** (Git blob IDs for similarly named dictionaries) and correctly labels it "UNVERIFIED". The correction dictionaries remain identity-less; the only registration is a blocker string (`PC_TRANSFER_CHECKLIST.md:56`).
**MF-6 (P2)** EKI anchor dataset `judge/judge.train.{noise_ratio}.json` (`datasets.py:238`) exists nowhere: not tracked, not registered, not local. Training only (40% of it is used per run: `train_BLIP2.py:102-103`, `train_ratio = args.EKI_batch/100`, default `EKI_batch=25` at `:73`); the paper names it as `D_anchor` §3.2. Training cannot be reproduced.
**MF-7 (P3)** Two READMEs document `python src/cirr_test_submission.py …` (`README.md:233`); `ls src/` → **No such file or directory**; `git ls-files | grep -c "^src/"` → 0. The file is `cirr_sub_BLIP2.py` at the repo root, with `__main__` at `:92`.
**MF-8 (P3)** `lavis/models/__init__.py` `__all__` (`:60-90`) still exports `BlipCir`/`BlipCirCaption` although the modules are absent — the dead export is the symmetry of the dead import.
**MF-9 (P3)** `metrics_extraction._check_parity` keeps a dead `tolerance: float = 0.05` parameter (`:31`) whose value is ignored (`:57-58` route through `paper_gate.compare_metric`). Remove it or wire it.
**MF-10 (P3)** `registry/assets/cluster_b3_ilearn.yaml` records two stale sha256 values for `AirKnow/data_utils.py` and `AirKnow/utils.py` and a 4-line-off `blip2.py` reference; the richer 9-asset audit never reaches preflight anyway (`registry.py:334-335` reads only `auxiliary_assets.yaml`, which has **0** `airknow` rows of 14 total).
**MF-11 (P3)** Every line citation in `workbench/docs/reviews/models/airknow.md` is offset by −12 against the current registry (`models.yaml:93-145`), and the `REFRESH` block at the end of that file contradicts its own body (correct `65.73` there, `65.74` earlier).
**MF-12 (P3)** `workbench/backend/paper_gate.py` carries a **+38/−1** uncommitted change introducing `CATEGORY_KEYS`/`deviations` (`git status --porcelain` → ` M workbench/backend/paper_gate.py`; `git diff --stat` → `38 +1`). All my line references are against the working tree, as the prior reviewer warned for its own file.

## UNRESOLVED EVIDENCE

1. **Checkpoint condition and payload identity.** No weight downloaded or opened. `checkpoint_training_noise_pct: 50/80` is filename + registry metadata; the Hub exposes size and an LFS `oid` (observed) and **states no training condition**. Whether these are FashionIQ checkpoints at all is **UNKNOWN**.
2. **Checkpoint loadability.** Whether `torch.load("airknow_fiq_0.5.pt")` yields a `Blip2QformerCir`-compatible object is **UNKNOWN**; no loader exists and the payload was never touched. I do **not** transfer any sibling Hub's whole-model-pickle finding here.
3. **Real-output equivalence of mask vs column removal.** Argued from code and pinned by tests; never demonstrated on Air-Know scores. **UNKNOWN.**
4. **Whether the on-disk FashionIQ counts equal the official release.** Local files are internally consistent; not verified upstream. **UNKNOWN.**
5. **Correction-dictionary contents and origin.** No provenance beyond a hypothesis. **UNKNOWN.**
6. **The `resized_image/` staging recipe.** No script in any readable file; the workbench's own doc describes one that does not exist. **UNKNOWN.**
7. **Whether upstream ever published 0 %/20 % weights and removed them.** Current file set and full main history are observed; their history is not. The registry's *negative* claim is correct for the observable repo.
8. **The 20%-block Average R@50 discrepancy in the paper** (cells average to 75.72; cell prints 75.71). The committed row transcribes the printed `75.71` and drops `derived_metrics`, so it is not a workbench defect; the paper's rounding is inconsistent across its four blocks (0 % half-up, 50 % truncation, 80 % half-up). **UNKNOWN** which convention the authors used.
9. **Canonical checklist text.** `/tmp/audit_assignments.md:47` and `/tmp/audit-final/context.md` specify "all 55 questions" and the required line format, but the 55 questions themselves are in no readable file. The probes below are **mine**, one per artefact the assignment enumerates; the substitution is explicit.

## EXECUTION TRACE

```
paper/README/config -> source dataset -> preprocessing -> official evaluator
  -> workbench invocation -> checkpoint -> output -> metric comparison
```

**1. paper/README/config → pinned, but the pin is a dead file set.**
- Paper: `read https://arxiv.org/abs/2604.19386` → HTTP 200 (title, six authors, cs.CV, 2026-04-21); `read https://arxiv.org/html/2604.19386v1` → full HTML; Table 1 Air-Know cells at HTML lines 194/199/204/209 naming methods in the 0%/20%/50%/80% blocks.
- README: `:101` clone URL (`ZhihFu/Air-Know`, which `301`s to `iLearn-Lab/CVPR26-Air-Know`), `:107` py/CUDA, `:110` (`-n conesep`), `:114-117` pins, `:200` noise ratios, `:205-213` train command, `:233` a non-existent `src/` path.
- Config: `lavis/configs/models/blip2/blip2_pretrain.yaml:10` (LAVIS BLIP-2 weights URL), `:21` (32 queries), `:32-36` (blip_caption processors); `blip2_cir.py:107-110` (`num_query_token=32`, `embed_dim=256`, `max_txt_len=32`). **Not registered for `airknow`** (0 rows in both auxiliary registries).
- ⇒ paper ✓, README ✓, config ✓, **registry pin of config/weights/tokenizer ✗ MISSING.**

**2. source dataset → present, wrong layout.** Required `resized_image/{cat}/*.jpg` (`datasets.py:33-35,154,165`) + val splits/captions/correction dicts (`:183,185,187`). Present: `captions image_splits images qwen_captions`; no `resized_image/`, 0 `correction_dict_*`, no cache files, no `judge/`. Executed preflight → **10 blockers**. ⇒ **BLOCKED.**

**3. preprocessing → confirmed in source, never exercised.** `train_BLIP2.py:111-119` → `data_utils.py:92-106`; `datasets.py:122-131` → `test_BLIP2.py:112` → `blip_processors.py:49-68` → `blip2_cir.py:480-486` → `blip2.py:32`. ⇒ **STATIC_AUDIT VERIFIED**; the two text stages are not interchangeable.

**4. official evaluator → exists as a function; not reachable.** `test_BLIP2.py:9-175`; only callsite `train_BLIP2.py:298`; no `__main__`; `:10` leaves `model.train()` on (CF-2); the module-level import chain dies at `lavis/models/__init__.py:33` (CF-1, confirmed against the upstream ref). ⇒ **NOT RUNNABLE.**

**5. workbench invocation → does not exist.** `adapters/models.py:54-56`; executed → `NotImplementedError`; `cls.command is OfficialScriptAdapter.command` → `True`. No `airknow` replay module. Full-gallery helpers reached from tests only. `registry.py:714-719` → `adapter_command_ready: False`. ⇒ **BLOCKED.**

**6. checkpoint → identity partial; bytes untouched.** `models.yaml:126-145`; Hub listing (2 files, sizes, LFS oids observed); `checkpoints/airknow/` absent. Condition is metadata-only (UNRESOLVED #1); no loader exists (CF-3b). ⇒ **PARTIAL; bytes/loadability/condition UNKNOWN.**

**7. output → none.** `reproduction_status: NOT_RUN` (`:122`); only `mock/mock_airknow-fiq_n05.json` (`data_kind: "mock"`, null digest fields). `metrics_extraction.py:328-340` would return `AGGREGATE_PARSER_UNAVAILABLE`/`PARSER_UNAVAILABLE` regardless. ⇒ **NOT_RUN.**

**8. metric comparison → transcription verified; reproduction impossible.** 4 committed rows re-derived from the paper cell-for-cell; `select_reference` + `evaluate_gate` executed for match / deviation / condition-mismatch / partial-category / variant / mock cases with the observed statuses above; `models.yaml:119-121` matches the 0%-noise row including the corrected mean. ⇒ **transcription VERIFIED (4/4 rows), gate behaviour VERIFIED (executed), PARITY NOT EVALUATED.**

## CHECKLIST

*Provenance: the canonical 55-question list is absent from every readable file (UNRESOLVED #9). The probes below are mine, in the required format.*

1. **Registered `model_id` present and unique** — VERIFIED — `models.yaml:93`; executed parse: `airknow` count 1 of 16 — NONE — none.
2. **`method_name`/`paper_title` match the upstream README** — VERIFIED — `models.yaml:103,107` vs `README.md:5` — NONE — none.
3. **`paper_url` resolves and identifies the same work** — VERIFIED — arXiv API HTTP 200; title + six authors match — NONE — none.
4. **Registry `upstream_commit_sha` equals checkout HEAD** — VERIFIED — `models.yaml:111` == `git -C workbench/third_party/AirKnow rev-parse HEAD` — NONE — none.
5. **Checkout remote equals `upstream_repo_url`** — VERIFIED — `git config --get remote.origin.url` — NONE — none.
6. **`upstream_default_branch` matches** — VERIFIED — `git branch -a --contains HEAD` → `* main` — NONE — none.
7. **Pin equals current upstream `main`** — VERIFIED — GitHub API `commits?sha=main` newest == `6ab6fa95b746…` — P3 — record the check in the registry so nobody re-opens it.
8. **`ZhihFu/Air-Know` == `iLearn-Lab/CVPR26-Air-Know`** — VERIFIED — `curl -sI https://api.github.com/repos/ZhihFu/Air-Know` → `301 → repositories/1185360679` == `full_name iLearn-Lab/CVPR26-Air-Know`; `curl -sIL https://github.com/ZhihFu/Air-Know` → effective URL the iLearn-Lab repo, 200 — P3 — correct the extracted payload's `research_note`, which calls this unconfirmed.
9. **No modified tracked file** — VERIFIED — `git status --porcelain`; non-bytecode count 0; `runtime.py:1829-1837` — P3 — keep `.pyc` out of the pin record.
10. **`source_available`/`source_dir` correct** — VERIFIED — `models.yaml:112-113`; tree present — NONE — none.
11. **`native_protocol` matches the source gallery** — VERIFIED — `datasets.py:31,218-224`; `FASHIONIQ_PROTOCOL_PARITY.md:70` — NONE — none.
12. **`supported_protocols` is exactly the native one** — VERIFIED — `models.yaml:118`; the `val-split` branch (`datasets.py:206-217`) is unreachable (`self.split` hardcoded `:31`) — P3 — note it as dead code.
13. **`literature_split_label: original` consistent** — VERIFIED — `:116`; Table 1 is the FashionIQ **validation** set, no gallery protocol stated by the paper — NONE — none.
14. **Reference excluded before ranking** — VERIFIED — `test_BLIP2.py:151` mask precedes `:154` argsort — NONE — none.
15. **Mask value and mechanism recorded (mask, not removal)** — VERIFIED — `-10e10` at `:151`; workbench removes the column (`src/evaluation/fashioniq.py:35-77`) — P3 — keep the distinction visible.
16. **Sort descending by similarity** — VERIFIED — `:154,168` `np.argsort(-sims[i, :])` — NONE — none.
17. **Tie-breaking policy pinned** — UNKNOWN — numpy default unstable; workbench declines to pin (`FASHIONIQ_PROTOCOL_PARITY.md:47-48`) — P3 — accept; measure-zero for L2-normalized features.
18. **Recall@K mathematics correct (`100*hits/N`)** — VERIFIED — `:169-174` — NONE — none.
19. **Denominator is the query count** — VERIFIED — `:174` `len(nn_result)` == `len(test_queries)` — NONE — none.
20. **R@1 additionally computed** — VERIFIED — `:169` `for k in [1, 10, 50]`; `:175` returns it — P3 — harmless; document.
21. **Macro/mean definition matches the paper** — VERIFIED — executed per-block arithmetic; `src/evaluation/fashioniq.py:238` — NONE — none.
22. **`reported_r10` matches the paper** — VERIFIED — `models.yaml:119` == arXiv HTML line 194 == `extracted/airknow.json` == `papers.yaml:37` → `55.75` — NONE — none.
23. **`reported_r50` matches the paper** — VERIFIED — `:120` == both sources → `75.72` — NONE — none.
24. **`reported_mean` matches the paper** — VERIFIED — `:121` `65.73` == paper `65.73` == `papers.yaml:36` (the prior round's `65.74` defect is fixed) — NONE — none.
25. **Category cells consistent with the average rows** — VERIFIED ×4 — executed `55.7533→55.75`, `75.7167→75.72`, `65.735→65.73`; `55.1833→55.18`, `65.45`; `53.1867→53.19`, `74.04`, `63.6133→63.61`; `49.23`, `70.2333→70.23`, `59.73` — NONE — none. Note the paper's own 20% Average R@50 cell prints `75.71` while its cells average to `75.72`; the row transcribes the printed value.
26. **All four paper rows committed, one per noise condition** — VERIFIED — `papers.yaml:10-68,69-115,116-161,162-207`; `checkpoint_training_noise_pct` 0/20/50/80 — NONE — none.
27. **Committed rows' condition fields present and usable by selection** — VERIFIED — executed `select_reference("airknow", noise 0/50/80, protocol)` → `airknow_fiq_0pct/50pct/80pct`, `exact condition match` — NONE — none.
28. **Rows' `gallery_protocol` matches the model's native protocol** — VERIFIED — `papers.yaml:27,105,135,181` `fashioniq_full_gallery_ref_excluded` == `models.yaml:117` — NONE — none.
29. **Row statuses honest (`VERIFIED` only where re-derived)** — VERIFIED — 4/4 `reference_status: VERIFIED` and I re-derived cell-for-cell from the paper; `review_status` `CONFIRMED/PARTIAL/CONFIRMED/CONFIRMED` matches `VerifyPapersA.json` — NONE — none.
30. **Reference selection is by condition, not by metric** — VERIFIED — `benchmark_references.py:145-190`; executed (0-noise run vs 80% row → `BLOCKED_CHECKPOINT_MISMATCH`) — NONE — none.
31. **`paper_meta`/`run_meta` constructed in production** — VERIFIED (supersedes prior CF-3) — `evaluate_models.py:137-148` builds `run_meta` incl. `checkpoint_training_noise_pct` and the variant flags; `:150-154` selects the reference — NONE — none.
32. **Reported row's noise condition labelled in the registry entry** — VARIANT — no `paper_row_noise_pct` field anywhere (`grep -rn` → 0); `models.yaml:119-121` is the 0% row with no marker while variants say `evaluation_noise_pct: 0` (`:131,141`) — **P2** — add the field (MF-2).
33. **A run's headline display cannot silently target the wrong row** — VARIANT — `autopilot.py:620-622` copies `reported_*` onto every row; `:814-828` computes `measured_difference` against them; the gate's own verdict is carried separately at `:844-845` — **P2** — attach `benchmark_gate.selection_reason` beside the headline, or null the headline when the row's condition differs.
34. **Gate cannot be fooled by close numbers across conditions** — VERIFIED — executed: control `PASS_WITHIN_5_PERCENT`, deviation `ALERT_METRIC_DEVIATION`, condition mismatch `BLOCKED_CHECKPOINT_MISMATCH`, mock `NOT_RUN` — NONE — none.
35. **Per-category paper metrics compared, unchecked ones reported** — VERIFIED — executed: full observation compares six category keys; partial observation reports `dress` compared and `shirt/toptee` "published by the paper but the run observed none, so they were not compared" — NONE — none.
36. **Variant runs are labelled, not waved through** — VERIFIED — executed `declared_variant=True` → `PASS_WITHIN_5_PERCENT_VARIANT` — NONE — none.
37. **Gallery equals the full val split in file order** — VERIFIED — `datasets.py:218-224` — NONE — none.
38. **Gallery sizes match the real splits** — VERIFIED (local disk) — 3817/6346/5373, duplicate-free — P3 — not cross-checked upstream (UNRESOLVED #4).
39. **Every reference/target inside its gallery** — VERIFIED — executed `set ⊆ gallery` → `True` ×3, plus 0 duplicate query triplets — NONE — none.
40. **Layout id `fashioniq_ilearn_resized` matches source paths** — VERIFIED — `fashioniq_layout.py:25-34` vs `datasets.py:33-35,183,185,187` — NONE — none.
41. **Image extension `.jpg` matches source** — VERIFIED — `datasets.py:155,166`; `fashioniq_layout.py:85` — NONE — none.
42. **Layout checker covers the eval-required files** — VERIFIED — `fashioniq_layout.py:25-34`, reached via `paths_for_layout` (`:65-76`) — NONE — none.
43. **Layout checker covers *all* files the evaluator opens** — VARIANT — cache JSON + 6 PKLs (`datasets.py:38-62`) unchecked; train captions only in `ilearn_resized_training_paths` (`:37-41`); `judge/` unchecked — P2 — cover the cache branch or invalidate on mismatch.
44. **Cache staleness guarded** — NOT_RUN (absent) — `datasets.py:38` gates on existence; `:55-62` never re-reads sources — P2 — require a provenance stamp in `fashion_iq_data.json`.
45. **Correction dictionaries present and provenance-bearing** — VARIANT — read at `datasets.py:187`/`:110`; 0 `airknow` rows in `auxiliary_assets.yaml` (14 assets, 0 airknow); 0 present locally; only hint is a labelled hypothesis (`UPSTREAM_AUDIT.md:25`) — P2 — register with a digest.
46. **Source root separator contract recorded** — VARIANT — `datasets.py:33-35` concatenates `path + 'resized_image'`; the registry has no such field — P2 — state it in `preparation_contract`/notes.
47. **Caption normalization matches source** — VERIFIED — `datasets.py:122-127` — NONE — none.
48. **Composition order and `" and "` joiner** — VERIFIED — `datasets.py:129-131` — NONE — none.
49. **LAVIS eval processor applied to the composed string** — VERIFIED — `test_BLIP2.py:112`; `blip_processors.py:49-68`; `blip2_pretrain.yaml:32-36` — NONE — none.
50. **Tokenizer path pinned in the registry** — VARIANT — `/root/bert-base-uncased` (`blip2.py:32,48,55`); no registry record or fallback — P2 — record as an asset or use an env var.
51. **`max_txt_len` recorded** — VARIANT — source default 32 (`blip2_cir.py:110,140`); config and registry silent — P3 — note in the prep contract.
52. **Image transform recipe matches source** — VERIFIED — `train_BLIP2.py:111-119`; `data_utils.py:92-106` — NONE — none.
53. **Eval and query images share one transform instance** — VERIFIED — `datasets.py:154-173`; no eval-specific transform exists (the registry vocabulary has no field for it) — P3 — record the omission.
54. **Staged resized images have a pinned generation recipe** — VARIANT — no script in the pinned tree; 0 `airknow` rows in `preparation_contracts.yaml`; `AUTOMATED_DATA_PREPARATION.md:18-25` claims one that does not exist — P2 — pin a resizer + quality, or hash the staged tree.
55. **Noise injection touches only training data** — VERIFIED — `datasets.py:66-101`; called only at `train_BLIP2.py:130,143`; `grep -c noise_ratio test_BLIP2.py` → 0 — NONE — none.
56. **Label-bearing sibling class is training-only** — VERIFIED — `FashionIQ_V` (`datasets.py:227-263`) instantiated only by `load_EKI_dataset` (`train_BLIP2.py:84-107`), consumed only by `v_trainloader` (`:275-279,291`); `test.test` receives `trainset` (`:298`) — NONE — none.
57. **EKI anchor dataset obtainable** — VERIFIED-NEGATIVE — `datasets.py:238`; not tracked, not registered, not local — P2 — publish or document as non-reproducible training input (MF-6).
58. **Evaluator runs in eval mode / no grad** — VERIFIED-NEGATIVE — `test_BLIP2.py:10` `model.train()`; only the `fashion200k` branch calls `eval()` (`:13`); siblings call `eval()` — **P1** — replay verbatim or declare a VARIANT (CF-2).
59. **Checkpoint variants declared with noise provenance** — VERIFIED — `models.yaml:126-145`; Hub file set matches exactly — NONE — none.
60. **Checkpoint URLs resolve** — VERIFIED (STATIC) — Hub tree/API listing observed with sizes and LFS oids — NONE — none.
61. **Checkpoint sizes recorded** — VARIANT — `:132,:142` `null` though the Hub publishes `2724056638` — P2 — record.
62. **Checkpoint sha256 recorded** — VARIANT — `:133,:143` `null` though the Hub exposes `f4d25ed9…`/`756fc0bd…` — P2 — record.
63. **Checkpoint file set matches the author Hub** — VERIFIED — `{.gitattributes, README.md, airknow_fiq_0.5.pt, airknow_fiq_0.8.pt}` — NONE — none.
64. **A clean (0 %) checkpoint is published** — VERIFIED-NEGATIVE — Hub holds exactly `0.5` and `0.8`; card names `0.0/0.2/0.5/0.8` as *trainable* ratios; repo README has no checkpoint section — NONE (registry correct) — none.
65. **`strict_clean_checkpoint_available` accurate** — VERIFIED — `:123`; executed `select_checkpoints` → `BLOCKED`, `False` — NONE — none.
66. **Checkpoint selection reaches a runnable variant** — VERIFIED-NEGATIVE — executed: both policies `BLOCKED` (`BLOCKED_CHECKPOINT`, `BLOCKED_CHECKPOINT_SELECTION`) — P2 — declare a bundle or a policy default, else the model can never be selected.
67. **Checkpoint loadable by pinned code** — UNKNOWN — no loader (`--restore_file` declared `train_BLIP2.py:67`, never read; no `torch.load` in `train_BLIP2.py`/`test_BLIP2.py`; `utils.py:128` uncalled; `cirr_sub_BLIP2.py:125` is the CIRR path); payload untouched — **P1** — needs a run (CF-3b).
68. **Standalone official evaluator exists** — VERIFIED-NEGATIVE — no `__main__`/argparse in `test_BLIP2.py`; only `train_BLIP2.py:298` — **P1** — write and audit an entrypoint.
69. **Pinned `lavis.models` package importable** — VERIFIED-NEGATIVE — `lavis/models/__init__.py:33-34` import two modules AirKnow alone never shipped; confirmed against the upstream ref via GitHub contents API; siblings ship the files; bytecode cannot substitute (live demonstration) — **P0/P1** — commit or guard the modules (CF-1). Rated P1 by the prior document because it produces no number; I keep P1 for the same reason but note that a package that cannot import makes the *registered evaluator script* definitionally dead.
70. **Registered `source_blocker` names the real blocker** — VARIANT — `models.yaml:94` names CLI/loader/adapter/asset reasons but not CF-1 — P2 — extend it.
71. **Adapter produces a command / evaluator reaches the protocol** — BLOCKED — `adapters/models.py:54-56`; executed `NotImplementedError`; `registry.py:714-719` → `adapter_command_ready: False`; preflight appends `command not runnable: COMMAND_UNAUDITED` — **P1** — no `airknow` number can be produced today.
72. **Output exists and parity compared** — BLOCKED — `reproduction_status: NOT_RUN` (`:122`); only a `data_kind: "mock"` fixture; `metrics_extraction.py:328-340` → `AGGREGATE_PARSER_UNAVAILABLE`/`PARSER_UNAVAILABLE`; `src/evaluation/fashioniq.py` helpers reached from tests only — **P1** — write the parser when a run exists.
73. **Readiness state matches the evidence** — VERIFIED — executed `preflight_blockers` → 10 blockers; `pc_readiness.classify` → `BLOCKED_SOURCE`; `READINESS_MATRIX.md:26` `BLOCKED` — NONE — none.
74. **Environment declared from the README and not claimed installed** — VERIFIED — `models.yaml:95-102` vs `README.md:107,114,117`; `environment_plan.json` `interpreter_mapping.airknow: null`, group `installation_status: NOT_REQUIRED`, env dir absent; `READINESS_MATRIX.md:14-19` all zero — NONE — none.
75. **Prior audit's claims re-checked, not paraphrased** — VERIFIED — every one of the 74 probes above was re-run against the pinned tree/working-tree files; three prior claims are **superseded**: registry block is `:93-145` not `:81-132`; `reported_mean` is now `65.73` (MF-1 resolved); production *does* construct `paper_meta`/`run_meta` (CF-3 resolved for the gate, open only for the display path). `MF-5` (two tolerance regimes) is reduced to a dead parameter. — P3 — mark the old document superseded rather than leaving contradictory text.

## FINAL VERDICT

**BLOCKED.** Every static registry claim I can check is now correct — SHA == checkout HEAD and == current upstream `main`, `ZhihFu/Air-Know` `301`s to the registered repo, protocol == `datasets.py:31` + `:218-224` with the reference masked to `-10e10` before `argsort` at `test_BLIP2.py:151/154`, `strict_clean_checkpoint_available: false` is right (Hub publishes exactly two weights at 50 % and 80 %), no evaluator calls `shuffle()` so evaluation noise is genuinely 0, all four committed paper rows transcribe Table 1 cell-for-cell including the corrected `65.73`, and the gate is genuinely wired by condition (executed: match `PASS`, deviation `ALERT`, 0-vs-80 `BLOCKED_CHECKPOINT_MISMATCH`, mock `NOT_RUN`, variant `…_VARIANT`) — but nothing on this host can produce a number: the pinned `lavis.models` package **cannot be imported** (`lavis/models/__init__.py:33-34` import two modules AirKnow alone never shipped, confirmed absent at the pinned upstream ref, while ConeSep/HABIT/INTENT ship the same imports *and* the files), the registered script has no `__main__`, no loader for `airknow_fiq_*.pt` exists anywhere in the tree, the adapter deliberately raises, the declared dataset layout is absent locally, and the paper row a comparison would target is the 0 %-noise row for which the author publishes no checkpoint.

**Superseded from the prior round (explicit):** registry line range (`:93-145`); `reported_mean` now `65.73`; `paper_meta`/`run_meta` **are** constructed in production (`evaluate_models.py:137-154`); the "two tolerance regimes" is now one dead parameter; the pin **is** current `main`; and the `ZhihFu`↔`iLearn-Lab` identity question the extracted payload left open is answered by the `301`.
