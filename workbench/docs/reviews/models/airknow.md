# airknow

**MODEL ID** — `airknow` (registry block `workbench/registry/models.yaml:81-132`).

**PAPER TITLE** — `Air-Know: Arbiter-Calibrated Knowledge-Internalizing Robust Network for Composed Image Retrieval` (`models.yaml:94`). `read https://arxiv.org/abs/2604.19386` → HTTP 200 (arXiv API); `read https://arxiv.org/html/2604.19386v1` → full HTML. Title, six authors (Fu, Hu, Yang, Zhang, Chen, Li; Shandong University) and the bibtex block match `workbench/third_party/AirKnow/README.md:5,8-25,352-360`.

**PINNED SOURCE COMMIT** — registry `upstream_commit_sha: 6ab6fa95b746f85b08eea4cb66eb51448bce99b7` (`models.yaml:98`).

| Claim | Evidence (observed) | Verdict |
|---|---|---|
| Registry SHA == checkout HEAD | `git -C workbench/third_party/AirKnow rev-parse HEAD` → `6ab6fa95b746f85b08eea4cb66eb51448bce99b7` | **MATCH** |
| Remote == `upstream_repo_url` | `git config --get remote.origin.url` → `https://github.com/iLearn-Lab/CVPR26-Air-Know.git`; `read https://api.github.com/repos/ZhihFu/Air-Know` → `full_name: iLearn-Lab/CVPR26-Air-Know` (README link and registry URL are one repo) | **MATCH** |
| `upstream_default_branch: main` (`:97`) | `git branch -a --contains HEAD` → `* main`, `remotes/origin/HEAD -> origin/main` | **MATCH** |
| No modified tracked file | `git status --porcelain` → 846 entries, all ` D …__pycache__…pyc`; `grep -vc __pycache__` → `0`. `runtime.py:1829-1837` treats these as non-source | **CLEAN** (bytecode-only) |
| History depth | `git rev-list --count HEAD` → `1`; `git log -1` → `6ab6fa95… 2026-09-28 14:40:56 +0800 Update README.md` | single-commit checkout |

**VERIFIED (pin).** Strongest claim in the audit.

---

## CHECKPOINT IDENTITY

STATIC_AUDIT + HTTP metadata. **No checkpoint downloaded, opened, hashed or loaded.**

| checkpoint_id | filename | training noise | eval noise | size | sha256 | status | mapping |
|---|---|---|---|---|---|---|---|
| `fiq_n05` | `airknow_fiq_0.5.pt` (`:114`) | 50 (`:117`) | 0 (`:118`) | `null` (`:119`) | `null` (`:120`) | `NO_CLEAN_CHECKPOINT` (`:121`) | `VERIFIED_METADATA` (`:122`) |
| `fiq_n08` | `airknow_fiq_0.8.pt` (`:124`) | 80 (`:127`) | 0 (`:128`) | `null` (`:129`) | `null` (`:130`) | `NO_CLEAN_CHECKPOINT` (`:131`) | `VERIFIED_METADATA` (`:132`) |

**Independent check of the published condition set** (urllib against the Hub API, output observed):
```
id iLearn-Lab/CVPR26-Air-Know  →  .gitattributes, README.md, airknow_fiq_0.5.pt, airknow_fiq_0.8.pt
airknow_fiq_0.5.pt 2724056638 f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6
airknow_fiq_0.8.pt 2724056638 756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e
HEAD …/resolve/main/airknow_fiq_0.5.pt  →  200, Content-Length 2724056638
```
The Hub contains **exactly two weight files**, named `0.5` and `0.8`. The author Hub card (`read …/raw/main/README.md`) says only *"trained and evaluated on standard CIR datasets under various simulated noise ratios (e.g., 0.0, 0.2, 0.5, 0.8)"* — **no 0.0 file, no 0.2 file, no checkpoint list**. The upstream repo publishes **no checkpoint list at all**: `grep -n -i "hugging|checkpoint|\.pt" third_party/AirKnow/README.md` → only `:117` (pip), `:200` (ratios), `:208,221` (`--model_dir ./checkpoints/fashioniq_noise0.8`), `:233` (a `src/cirr_test_submission.py` path that does not exist — MF-4).

**Clean-checkpoint question.** `strict_clean_checkpoint_available: false` (`models.yaml:110`) is **CORRECT** (Hub listing above). `notes: Official Hub lists only FashionIQ 50% and 80% noise-trained weights` (`:111`) is accurate for the file set, but *FashionIQ* is a registry/Hub-filename assertion — the Hub never associates a file with a dataset, category set or noise level (UNRESOLVED #1). Executed `select_checkpoints`: `strict_clean` → `BLOCKED BLOCKED_CHECKPOINT`, reason `no checkpoint is recorded as clean-trained under policy strict_clean: fiq_n05=50%, fiq_n08=80%.`; `include_flagged_conditions` → `BLOCKED BLOCKED_CHECKPOINT_SELECTION`, reason `more than one source-backed variant is registered and none is declared a bundle: fiq_n05, fiq_n08.` — this model has **no `checkpoint_bundles`**, so even the permissive policy cannot resolve one (`checkpoint_selection.py:126-130,181-199`).
**Consequence (P1):** the pinned paper row (55.75/75.72/65.74) is the **0 %-noise row**; the obtainable checkpoints are 50 % (53.19/74.04/63.61) and 80 % (49.23/70.23/59.73). Any comparison to 55.75 would be a wrong-row comparison. `paper_gate.py:40-47` lists `checkpoint_training_noise_pct` in `COMPARABILITY_FIELDS` and executed does refuse a mismatch (`BLOCKED_CHECKPOINT_MISMATCH`), control passes (`PASS_WITHIN_5_PERCENT`) — but **no production caller constructs `paper_meta`/`run_meta`** (`grep -rn` finds them only in `paper_gate.py` and `tests/test_paper_gate.py`), so the row's condition is unlabelled and the guard can never fire (CF-3).

---

## EVALUATION PROTOCOL

Registry: `native_protocol: fashioniq_full_gallery_ref_excluded` (`:104`), `supported_protocols: [same]` (`:105`), `literature_split_label: original` (`:103`); protocol table `:12-15`.

1. **Exact evaluator.** `test_BLIP2.py:9` `def test(params, model, testset, category, txt_processors)`; categories at `:82-96`. `grep -n "__main__|argparse|ArgumentParser" test_BLIP2.py` → **no match**. Only callsite: `train_BLIP2.py:298` inside `train_and_evaluate()` (`:268-337`), reached from `__main__` `:362`. **No standalone entrypoint.**
2. **Gallery construction.** `datasets.py:31` `self.split = 'original-split'` (hardcoded; `__init__` at `:29` takes only `path, transform, noise_ratio`). `get_test_data()` (`:181-225`) reads `image_splits/split.{name}.val.json` (`:183`), then `elif self.split == 'original-split':` (`:218`) enumerates **every** index in file order (`for id, image_name in enumerate(images)`) → **full ordered validation split**. The `val-split` pair-union branch (`:206-217`) is dead code. **CONFIRMED.**
3. **Reference excluded before ranking — exact lines.** `test_BLIP2.py:147-151`
   ```python
   test_targets_id = []
   for i in test_targets:
       test_targets_id.append(i['target_img_id'])
   for i, t in enumerate(test_queries):
       sims[i, test_targets_id.index(t['source_img_id'])] = -10e10
   ```
   then `:154` `nn_result = [np.argsort(-sims[i, :])[:50] …]`. Reference **masked to `-10e10` BEFORE the sort**, not removed from the gallery. `docs/FASHIONIQ_PROTOCOL_PARITY.md:70` agrees; the workbench deliberately uses column removal (`src/evaluation/fashioniq.py:35-77`, docstring `:42-53`). **CONFIRMED.**
4. **Recall@K maths.** `:168` `np.argsort(-sims[i, :])[:50]`; `:169` `for k in [1, 10, 50]`; `:172` hit test; `:174` `r = 100 * r / len(nn_result)`. Denominator = query count. **CONFIRMED.**
5. **Sort direction / ties.** `-sims` → descending; numpy default unstable sort, unpinned by source and by the workbench (`docs/FASHIONIQ_PROTOCOL_PARITY.md:47-48`). Measure-zero for `F.normalize` features (`blip2_cir.py:499-500`). **UNKNOWN-by-design, not a defect.**

**Similarity-tensor check (P3).** `test_BLIP2.py:142-144` `np.matmul(...)` → `.squeeze()` → `.max(-1)`. `extract_retrieval_compose` returns `fusion_feats.unsqueeze(1).unsqueeze(1)` (`blip2_cir.py:503`) with `fusion_feats` dim `embed_dim=256` (`:109,136`) → `(N,1,1,256)`; `extract_retrieval_target` returns `permute(0,2,1)` (`:526`) → `(M,256,32)` (`num_query_token=32`, `blip2_pretrain.yaml:21`). Executed numpy simulation with those dims: `matmul` `(N,M,1,32)` → `.squeeze()` `(N,M,32)` → `.max(-1)` **`(N,M)` 2-D**; `sims[2,4]=-10e10` writes through and leaves other rows unchanged; `argsort(-s[0,:])[:3]` → `(3,)`. Mask lands on a real element; downstream line indices (`:154,168`) are therefore correct at inspection **but were timed at `:151/154`** — the elided earlier note about `:150` was a tool line-numbering artefact, resolved by `awk` printing (`:147-151`). The inline comment `#all_queries.dot(all_imgs.T)` is stale. Feature normalisation (`:134-138`) is **commented out** (siblings keep it active); features still arrive unit-normalised from inside the model (`blip2_cir.py:499,525`). Recorded as a real fork divergence at `workbench/registry/assets/cluster_b3_ilearn.yaml:28,32-34`.

---

## DATASET CONTRACT

Registry: `fashioniq_layout: fashioniq_ilearn_resized` (`:101`), `fashioniq_image_extensions: [".jpg"]` (`:102`), `preparation_contract: manual_ilearn_resized` (`:93`). Source root is string-concatenated, so the caller's path **must end with a separator** (`datasets.py:33-35`).

| Input | Source callsite | Layout checker |
|---|---|---|
| `image_splits/split.{cat}.val.json` | `datasets.py:183` | `fashioniq_layout.py:33` |
| `captions/cap.{cat}.val.json` | `datasets.py:185` | `fashioniq_layout.py:31` |
| `captions/correction_dict_{cat}.json` | `datasets.py:187` | `fashioniq_layout.py:32` |
| `captions/cap.{cat}.train.json` | `datasets.py:108` (cache miss) | only in `ilearn_resized_training_paths` (`:37-41`) |
| `resized_image/{cat}/{id}.jpg` | `datasets.py:155,166` | `fashioniq_layout.py:29-30` + `:92-136` |
| `fashion_iq_data.json`, 6 PKLs | `datasets.py:38-62` | **not checked** |

`_LAYOUT_IMAGE_DIRECTORY["fashioniq_ilearn_resized"] = ("resized_image", True)` (`fashioniq_layout.py:85`) matches the source path. **CONFIRMED.**

**Local readiness (executed).** `model_requirements(airknow, resolve_config())` → **8 blocks**: missing `resized_image`, `resized_image/{dress,shirt,toptee}`, `captions/correction_dict_{dress,shirt,toptee}.json`, and `image extension requirement: image directory missing: …/resized_image`. `ls …/fashionIQ_dataset` → `captions image_splits images qwen_captions` (74 381 PNGs in `images`). `captions/` holds 9 `cap.*.json` but **0 `correction_dict_*`**; no `fashion_iq_data.json`, no `*.pkl`, no `judge/`. `docs/READINESS_MATRIX.md:26` records the same. **BLOCKED.**
Counts read from disk: dress 3817 gallery / 2017 queries; shirt 6346 / 2038; toptee 5373 / 1961; all duplicate-free; `set(refs) ⊆ gallery` and `set(targets) ⊆ gallery` → `True` for all three. Not cross-checked against the official release (UNRESOLVED #6).

**Gap 1 (P2) — unchecked, unversioned cache.** `datasets.py:38` `if not os.path.exists(…'fashion_iq_data.json')`; on the hit path (`:55-62`) the JSON and six PKLs are loaded and the split/caption/correction files are **never re-read**. Stale cache is trusted (`docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:51-53`).
**Gap 2 (P2) — correction dictionaries have no provenance.** Opened at `:187`/`:110`; untracked by the repo; `auxiliary_assets.yaml` has 0 `airknow` rows; 0 present locally. The only provenance hint anywhere names an unrelated sibling repo by Git blob id (`docs/UPSTREAM_AUDIT.md:25`) — a hypothesis, not an identity.

---

## PREPROCESSING CONTRACT

Registry `SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:27`: caption stage `EQUIVALENT_WITH_EVIDENCE (pre-LAVIS string stage only)`, image stage `EQUIVALENT_WITH_EVIDENCE`.

**Stage 1** (`datasets.py:122-131`): `correct_text` lowercases → translates **all** `string.punctuation` to spaces → `.strip().split()` → exact per-token `correction_dict` lookup → rejoin; `concat_text` joins `cap[0] + " and " + cap[1]` in source order. Applied at `:192` (val) and `:116` (train init). **CONFIRMED.**
**Stage 2** (`test_BLIP2.py:112` `txt_processors["eval"](caption)`): `blip_processors.py:49-68` lowercases, replaces `[.!\"()*#:;~]` (a **narrower** set than `string.punctuation`), collapses whitespace, truncates at `max_words=50` (`:30,45`). Selected by `blip2_pretrain.yaml:32-36` (`name: blip_caption`). The two stages are **not interchangeable** (`docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:56-69`). **CONFIRMED.**
**Stage 3:** `blip2_cir.py:480-486` BERT tokenizer, `padding="max_length", truncation=True, max_length=self.max_txt_len` with `max_txt_len=32` (`:110,140`). Tokenizer literal `/root/bert-base-uncased` at `blip2.py:32` (also `:48`, `:55`) — **no registry record, no fallback** (MF-2).
**Image:** `train_BLIP2.py:111-119` `targetpad_transform(1.25, 224)` → `data_utils.py:92-106` (TargetPad → `Resize(224, BICUBIC)` → `CenterCrop(224)` → RGB → ToTensor → CLIP mean/std). `TargetPad.__call__` pads only when `actual_ratio >= 1.25` (`:64-73`). **CONFIRMED.**
**Resize staging recipe: UNKNOWN** — no script in the tree; 0 `airknow` rows in `preparation_contracts.yaml` (UNRESOLVED #7).

---

## AUGMENTATION CONTRACT

`FashionIQ.shuffle()` (`datasets.py:66-101`) is the only augmentation hook:
```
66  def shuffle(self):
68      if self.noise_ratio == 0:  logging.info("无噪声模式"); return
73      shuffle_indices = random.sample(range(num_samples), int(self.noise_ratio * num_samples))
74-78  split into candidate / captions / target thirds
87      for i in shuffle_indices: self.labels[i] = 0; self.qlabels[i] = 0
89-90  flip qlabels on a further random 20%
92-97  random.shuffle(noise_candidate|captions|target); write back
```
Matches the paper (§3.1): *"we simulate noisy correspondence by randomly shuffling the elements of a certain proportion of samples within the training set"*.
**Callsites: two, both training.** `grep -rn "\.shuffle()"` → `train_BLIP2.py:130` (FashionIQ), `:143` (CIRR). `test_BLIP2.py` has **no** `noise_ratio` reference (grep → 0). **No evaluator calls it; training noise is never evaluation noise** — the project's `evaluation_noise_pct = 0` requirement holds on the eval path, independently of the registry assertion (`:118,128`) and of `registry.py:84`.
**Corruption on the eval path: none found.** `get_test_data` reads the raw val annotations; the cache substitutes the identical records the same function wrote.
**Label-bearing sibling class is training-only.** `FashionIQ_V` (`datasets.py:227-263`) reads `os.path.join(self.path+'judge', f'judge.train.{self.noise_ratio}.json')` (`:238`) and exposes `labels`/`Qlabels`. Instantiated only by `load_EKI_dataset()` (`train_BLIP2.py:84-107`, `:90-94`), consumed only by `v_trainloader` (`:275-279,291`) — **never passed to `test.test`**, which receives `trainset` (`:298`). **CONFIRMED**, but the `judge/` dataset it needs is missing everywhere (MF-3).

---

## GALLERY CONTRACT

Full ordered `split.{category}.val.json`, reference masked per query before ranking — evidence quoted above (`datasets.py:31,218-224`; `test_BLIP2.py:147-151`). Local counts 3817/6346/5373, duplicate-free, all refs and targets inside. Workbench expression: `FASHIONIQ_PROTOCOLS['fashioniq_full_gallery_ref_excluded']` = `gallery='full_split', reference_excluded=True` (`src/evaluation/fashioniq.py:159-169`); `build_fashioniq_gallery` (`:198-206`) → `build_original_gallery` (`:109-117`); `recall_at_k_reference_excluded` (`:35-77`) removes the column (documented divergence from the mask). Callsites outside that module are tests only (`tests/unit/test_fashioniq_protocols.py`; `workbench/tests/test_protocols.py:12`).

---

## METRIC CONTRACT

**Source output shape.** `test()` returns `[(name, value)]` per category: `:175` `out += [('{}_r{}'.format(name, k), r)]` for `k in {1,10,50}`. Not JSON, no aggregate. The trainer sums the three R@10 values (`train_BLIP2.py:296-301` `current_score = current_score + t[1][1]`) — the selection score is a **sum**, not a mean — and persists a flat 9-key map via `utils.save_dict_to_json` (`:315`). The paper's `AVG.` is the mean of the three R@10 and the three R@50, then `(r10+r50)/2`; the workbench helper implements exactly that (`src/evaluation/fashioniq.py:223-240`).
**Workbench parser: none.** `metrics_extraction.py:287-320` dispatches only `limn`/`csmcir`; `airknow` falls through to `extraction_status="AGGREGATE_PARSER_UNAVAILABLE"`, `parity_status="PARSER_UNAVAILABLE"`, `observed_metrics=None` (`:311-320`) — asserted at `tests/test_adversarial_metrics.py:98-101` and `:104-108` (paper metric never substituted for observed). **No `airknow` number can be extracted even if the evaluator ran.** (P1.)
**Two tolerance regimes:** absolute `0.05` in `metrics_extraction._check_parity:28` vs relative `5.0` in `paper_gate.TOLERANCE_PCT:11` (MF-5).

---

## PAPER-METRIC STATUS

| source | Dress R@10/R@50 | Shirt | Toptee | Avg R@10 | Avg R@50 | AVG. |
|---|---|---|---|---|---|---|
| arXiv HTML Table 1 (`0%` block) | 51.03/73.08 | 56.76/75.05 | 59.47/79.02 | **55.75** | **75.72** | **65.73** |
| `assets/results-fiq.png` (`0%` row, read as image) | 51.03/73.08 | 56.76/75.05 | 59.47/79.02 | 55.75 | 75.72 | 65.73 |
| registry `models.yaml:106-108` | — | — | — | 55.75 | 75.72 | **65.74** |

Executed: `(51.03+56.76+59.47)/3 = 55.7533`; `(73.08+75.05+79.02)/3 = 75.7167`; `(55.7533+75.7167)/2 = 65.735`; `round(65.735,2) = 65.73`. **`reported_r10`/`reported_r50` VERIFIED; `reported_mean` is off by 0.01 pp** (registry `65.74` vs both upstream sources `65.73`; relative 0.0152 %) (MF-1, P2).
Other rows isolate the comparability problem: 20 % → 55.18/75.71/65.45; 50 % → 53.19/74.04/63.61; 80 % → 49.23/70.23/59.73.
**No reproduction attempted** — `reproduction_status: NOT_RUN` (`:109`). The only result artefact is `workbench/artifacts/results/fashioniq_full_gallery_ref_excluded/mock/mock_airknow-fiq_n05.json` (`data_kind: "mock"`, `checkpoint_sha256: null`, `command_digest: null`, synthetic metrics 12.5/20.83/16.67) — not evidence (`paper_gate.py:171-173` returns `NOT_RUN` for `data_kind != "experiment"`). **PARITY NOT EVALUATED.**

---

## SOURCE-TO-WORKBENCH PARITY

| Component | Registry/doc claim | My check | Verdict |
|---|---|---|---|
| Protocol | `fashioniq_full_gallery_ref_excluded` | `datasets.py:31,218-224`; `test_BLIP2.py:147-151,154` | **EQUIVALENT** (mask, not removal) |
| Caption stage | `EQUIVALENT (pre-LAVIS string stage only)` | `datasets.py:122-131`; two-stage composition vs `blip_processors.py:49-68` | **EQUIVALENT for stage 1; stage 2 LAVIS-internal, unexercised** |
| Image stage | `EQUIVALENT_WITH_EVIDENCE` | `data_utils.py:92-106` | **EQUIVALENT (static)** |
| Preparation contract | `EXTERNAL_ASSET_REQUIRED` | all four requirements absent from the registry; three absent locally | **CONFIRMED GAP** |
| Evaluator command | `NOT IMPLEMENTED` | `adapters/models.py:54-56`; executed → `NotImplementedError` | **CONFIRMED** |
| Overall | `INCOMPLETE_IMPLEMENTATION` | — | **AGREE** |

One nuance the matrix flattens: gallery images go through `get_img` (`datasets.py:154-163`) and queries through `get_test_img` (`:165-173`), but both receive the same `targetpad_transform` instance, so the evaluator has **no eval-specific transform**; the layout vocabulary cannot express that (not a mismatch — a missing concept).

---

## RUNTIME STATUS

- **Environment (registry):** `WORKBENCH_AIRKNOW_PYTHON`, `VERIFIED_README`, python `3.8.10`, torch `2.1.0`, torchvision `0.16.0`, cuda `12.6`, packages `scikit-learn==1.3.2, transformers==4.25.0, salesforce-lavis==1.0.2, timm==0.9.16` (`models.yaml:82-89`). Upstream: `README.md:107` (Python 3.8.10 / CUDA 12.6), `:114` (torch 2.1.0 / tv 0.16.0 cu121), `:117` (the four pins, verbatim). Note `README.md:110` `conda create -n conesep` — a sibling's env name copy-pasted.
- **Not installed.** `environment_plan.json` maps `airknow` → `cir-4c8b682368302099` (`interpreter_mapping: {"airknow": null}`, solver `PARTIAL`), group `installation_status: NOT_REQUIRED`, `conda_env_path …/artifacts/environments/cir-4c8b682368302099`; `ls …/artifacts/environments` → **No such file or directory**. `READINESS_MATRIX.md` counts `ENVIRONMENT_INSTALLED: 0`, `RUNTIME_VERIFIED: 0`, `SCIENTIFICALLY_REPRODUCED: 0`.
- **Adapter:** `AirKnowAdapter` (`adapters/models.py:54-56`) names `script = "test_BLIP2.py"` only. Executed → `NotImplementedError: upstream evaluation command requires audit before execution` (`:24-25`). `command_status` absent from the registry block (count 0), so `checkpoint_availability` (`registry.py:701-706`) correctly reports `adapter_command_ready: False`.
- **Result records:** none (mock only). **GPU inference: not attempted, none available. Every claim here is STATIC_AUDIT.**

---

## CRITICAL FINDINGS

**CF-1 (P1, blocker) — the pinned `lavis.models` package cannot be imported, so the registered entrypoint cannot start; and this defect is unique to AirKnow.**
`workbench/third_party/AirKnow/lavis/models/__init__.py:33-34`
```python
33  from lavis.models.blip_models.blip_cir import BlipCir
34  from lavis.models.blip_models.blip_cir_caption import BlipCirCaption
```
Both modules are **untracked**:
```
$ git ls-files | grep -E "blip_cir|Llm_online"
lavis/configs/models/blip_cir.yaml
lavis/configs/models/blip_cir_caption.yaml
lavis/configs/models/blip_cir_large.yaml
lavis/models/blip_models/.ipynb_checkpoints/blip_cir-checkpoint.py
lavis/models/blip_models/__pycache__/blip_cir.cpython-{37,38,39,312}.pyc
lavis/models/blip_models/__pycache__/blip_cir_caption.cpython-{37,38,39,312}.pyc
$ ls lavis/models/blip_models/blip_cir.py …
ls: cannot access … No such file or directory
$ python3 - <<EOF   # resolve every module imported by lavis/models/__init__.py
distinct modules imported: 34
unresolvable: ['lavis.models.blip_models.blip_cir', 'lavis.models.blip_models.blip_cir_caption', 'omegaconf']
```
The `.py` sources exist only as `.pyc` under `__pycache__/` and as a Jupyter checkpoint. Python resolves packages before leaf modules, so `from lavis.models.blip2_models.blip2_cir import Blip2QformerCir` at `:40` — the class this evaluator needs — **never executes**; the package import aborts at `:33`. `omegaconf` is a declared LAVIS dependency, listed for completeness, not the blocker.
**Contrast evidence (background check, observed):** `ConeSep`, `HABIT` and `INTENT` carry the **identical** lines `33:from lavis.models.blip_models.blip_cir import BlipCir` / `34:from lavis.models.blip_models.blip_cir_caption import BlipCirCaption`, but each of them **does ship** the file (`ConeSep/lavis/models/blip_models/blip_cir.py`, `HABIT/…`, `INTENT/…` all resolve). The import lines are lineage; the missing files are **AirKnow's alone**. This is a genuine model-specific defect, not a shared-family quirk.
Call chain: `train_BLIP2.py:32` `from lavis.models import load_model_and_preprocess` executes at module import, before `:298` `t = test.test(...)` is ever reached; `AirKnowAdapter.script = "test_BLIP2.py"` is therefore unreachable as written. P1 rather than P0 because it produces no *wrong number*, it produces **no number**. The AirKnow analogue of the CSMCIR blocker in `workbench/patches/csmcir.md`; **no `workbench/patches/airknow*` exists** (`ls workbench/patches/` → `csmcir.md`, `csmcir_models_init_compat.patch`).
*Action:* commit/vendor the two modules or guard the two imports, then re-derive whether `Blip2QformerCir` loads.

**CF-2 (P1) — the FashionIQ evaluator runs with the model in training mode.**
`test_BLIP2.py:9-11`: `def test(...)` → `model.train()` → `with torch.no_grad():`. The `fashion200k` branch opts out (`:13` `model.eval()`); the FashionIQ branch (`:82-175`) never does. `no_grad()` disables gradients, not dropout. Live dropout: `blip2_cir.py:42-63` `Proxy` (`nn.Dropout(p=0.1)` at `:58,61`; built `:146-150` with `dropout_rate=0.1`) and the Q-Former's `BertConfig` (`blip2.py:48-55`), which inherits BERT-base dropout defaults. **Every sibling calls `model.eval()`**: `ConeSep/test.py:10`, `HABIT/test.py:10`, `INTENT/test.py:10` (executed grep). Independently recorded: `registry/assets/cluster_b3_ilearn.yaml:27`, `registry/audits/cluster_b_ilearn.yaml:11`, `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md:146-148`, `docs/AUXILIARY_ASSET_AUDIT.md:124`. The published numbers came from this same code, so verbatim replay reproduces the authors' protocol *including* its stochasticity; a re-run cannot be bit-reproducible and its variance is unquantified. Forcing `eval()` would define a **different experiment** and must not be silently substituted.
*Action:* replay verbatim (and say so), or declare a VARIANT. P1: documented, unremediated protocol divergence with non-deterministic effect. The registry block records it nowhere.

**CF-3 (P1) — 0 %-noise paper row vs 50/80 % checkpoints, with no `paper_meta` to trigger the guard.** Evidence in CHECKPOINT IDENTITY and PAPER-METRIC STATUS. `paper_gate.py:40-47` catches exactly this, executed: control `PASS_WITHIN_5_PERCENT`; mismatch `BLOCKED_CHECKPOINT_MISMATCH`. But `grep -rn "paper_meta|run_meta" workbench/` returns matches only in `paper_gate.py` and `tests/test_paper_gate.py`; production carries only the three floats (`evaluate_models.py:215`, `autopilot.py:772,839`). No `paper_row_noise_pct` field exists (`grep -c` → 0). *Action:* add it and thread it into the gate; until then no comparison to 55.75/75.72/65.74 may be reported as a reproduction.

**CF-4 (P1) — no runnable end-to-end path, in three independent places.** (a) package import dies at `lavis/models/__init__.py:33` (CF-1). (b) `test_BLIP2.py` has no `__main__`/CLI (grep → no match) and the only script that calls it has no weight-loading path: `--restore_file` is declared at `train_BLIP2.py:67` and **never read** (`grep -n restore` → the declaration only); `grep -n "torch.load|load_state_dict" train_BLIP2.py test_BLIP2.py` → **no match**; the only write is `torch.save(model, …'best_model.pt')` (`:366`). **There is no loader for `airknow_fiq_*.pt`** (same conclusion as `docs/AUXILIARY_ASSET_AUDIT.md:122`). (c) `AirKnowAdapter.command` raises (executed).

---

## MINOR FINDINGS

**MF-1 (P2)** `reported_mean: 65.74` (`:108`) vs the paper's `65.73`; executed `round(65.735,2)=65.73`. Correct it.
**MF-2 (P2)** Hardcoded `/root/bert-base-uncased` (`blip2.py:32,48,55`), no registry record, no fallback — unportable as written.
**MF-3 (P2)** The EKI anchor dataset `judge/judge.train.{noise_ratio}.json` (`datasets.py:238`, via `FashionIQ_V`; consumed by `load_EKI_dataset`, `train_BLIP2.py:84-107,275-279`) exists nowhere: not tracked (`git ls-files | grep judge` → no match), not in `preparation_contracts.yaml` (0 rows), not in `fashioniq_layout.py` (0 mentions), not local. The paper names it (`D_anchor`, §3.2). Only 40 % is used per run (`train_BLIP2.py:102` `train_ratio = args.EKI_batch/100`, default `EKI_batch=25`, `:73`). Training cannot be reproduced; evaluation unaffected.
**MF-4 (P3)** Both READMEs document `python src/cirr_test_submission.py …` (`README.md:233`); executed `ls src/` → **No such file or directory**. The file is `cirr_sub_BLIP2.py` at the root, with `__main__` hardcoding `/root/autodl-tmp/cirr_data/CIRR/` (`:117`). `cluster_b3_ilearn.yaml:8` correctly cites the *training* block `README.md:205-213` instead.
**MF-5 (P3)** Two tolerance regimes (absolute 0.05 vs relative 5 %).
**MF-6 (P3)** Zero `airknow` rows in `auxiliary_assets.yaml` and `preparation_contracts.yaml`; the LAVIS BLIP-2 pretrain weights (`blip2_pretrain.yaml:10`), EVA-ViT-g URL (`eva_vit.py:443`), the BERT directory and the three dictionaries are unnamed prerequisites. A richer 9-asset audit exists at `workbench/registry/assets/cluster_b3_ilearn.yaml:56-120+` but `auxiliary_assets_for_model` reads only `auxiliary_assets.yaml`, so it never reaches preflight.
**MF-7 (P3)** Single-commit checkout; whether the pin is current `main` is UNKNOWN (no fetch permitted).
**MF-8 (P3)** `BertConfig.from_pretrained("/root/bert-base-uncased")` (`blip2.py:48`) pins no revision — a changed upstream config is undetectable.
**MF-9 (P3)** `prepare_dataset.py:17-18` `required_layout()` returns `standard_paths(root)` unconditionally, so the dataset-preparation entry point validates the wrong layout for every `fashioniq_ilearn_resized` model; strict runtime preflight uses `paths_for_layout` instead (`registry.py:679-683`) — hence the correct 8-block list above while `prepare_dataset` measures a different tree (`docs/audits/PHASE2_DATA_CONTRACT.md:15`).

---

## UNRESOLVED EVIDENCE

1. **Checkpoint condition and payload identity.** No weight downloaded or opened. `checkpoint_training_noise_pct: 50/80` is filename + registry metadata; the Hub exposes only size and a SHA-256 `oid` (`f4d25ed9…`, `756fc0bd…` — observed, not invented) and **states no training condition**. Whether these are FashionIQ checkpoints at all is **UNKNOWN**.
2. **Checkpoint loadability.** Whether `torch.load("airknow_fiq_0.5.pt")` yields a `Blip2QformerCir`-compatible object is **UNKNOWN**; no loader exists in the source and the payload was never touched. I do **not** transfer the sibling ConeSep Hub's whole-model-pickle finding to this Hub. **UNKNOWN.**
3. **Real-output equivalence of mask vs column removal.** Argued from the code and pinned by tests, never demonstrated on AirKnow scores. **UNKNOWN.**
4. **Origin of `65.74`.** Both upstream sources print `65.73`; the registry's `65.74` is unexplained. **UNKNOWN.**
5. **Correction-dictionary contents vs the paper's.** No provenance. **UNKNOWN.**
6. **Whether the on-disk FashionIQ counts equal the official release.** Not verified upstream. **UNKNOWN.**
7. **The `resized_image/` staging recipe.** No script anywhere. **UNKNOWN.**
8. **Whether upstream ever published a 0 %/20 % checkpoint and removed it.** The current file set is observed; its history is not. **UNKNOWN.** (The registry's *negative* claim is correct for the observable repo.)
9. **The 55-item checklist's canonical question text.** `/tmp/audit_assignments.md:47` and `/tmp/audit_context.md:47` specify "all 55 questions" and the format, but the list itself is in **no file I can read** — `/tmp/audit_assignments.md` is 449 lines of registry facts and assignment list, and nothing under `/home/heheboiz/data/cir` matches `55 question|55-item|checklist` beyond the format instructions. The 55 probes below are **mine**, one per fact, covering every artefact the assignment enumerates. Recorded so the substitution is explicit.

---

## FINAL VERDICT

**BLOCKED.** Every static registry claim I can check against the pinned tree is **correct** — SHA matches the checkout, `fashioniq_full_gallery_ref_excluded` is exactly `datasets.py:31,218-224` + `test_BLIP2.py:147-151` (full ordered val split, reference masked to `-10e10` before `argsort`), `strict_clean_checkpoint_available: false` is right (Hub publishes exactly two weights at 50 % and 80 %), no evaluator calls `shuffle()` so evaluation noise is genuinely 0, and `reported_r10/r50` transcribe the paper's 0 %-noise row exactly. But nothing here can produce a number: the pinned `lavis.models` package **cannot be imported** (`lavis/models/__init__.py:33-34` import two modules AirKnow alone never shipped — the siblings ship the same imports *and* the files), the registered script has no `__main__`, no checkpoint loader exists, the adapter raises `NotImplementedError` (executed), the registered `resized_image`/correction-dictionary layout is absent locally (8 blocks, executed), the metric extractor has no `airknow` parser, and the paper row a comparison would target is the 0 %-noise row for which no checkpoint exists while the gate's noise guard is fed no `paper_meta`. No runtime claim was observed or is made.

---

## EXECUTION TRACE

```
paper/README/config -> source dataset -> preprocessing -> official evaluator
  -> workbench invocation -> checkpoint -> output -> metric comparison
```

**1. paper/README/config → partly pinned; the pin is a dead file set.**
- Paper: `read https://arxiv.org/abs/2604.19386` HTTP 200; `read https://arxiv.org/html/2604.19386v1` full HTML; Table 1 read with cells. Title/authors match `README.md:5,8-25`.
- README: `:107` py/CUDA, `:110` (`-n conesep`), `:114-117` pins, `:200` ratios, `:204-213` train cmd, `:228-236` a non-existent `src/` path (MF-4).
- Config: `blip2_pretrain.yaml:10` (LAVIS weights), `:21` (32 queries), `:24-36` (processors); `blip2_cir.py:109,110,135-136` (`embed_dim=256`, `max_txt_len=32`). **Not registered for `airknow`** (0 rows).
- ⇒ paper ✓, README ✓, config ✓; **registry pin of config/weights/tokenizer ✗ MISSING.**

**2. source dataset → present, wrong layout.** Required `resized_image/{cat}/*.jpg` + val splits/captions/correction dicts (`datasets.py:33-35,183,185,187`). Present: `captions image_splits images qwen_captions`; no `resized_image/`, no `correction_dict_*` (0), no cache files, no `judge/`. Executed preflight → **8 blocks**. ⇒ **BLOCKED.**

**3. preprocessing → confirmed in source, never exercised.** `train_BLIP2.py:111-119` → `data_utils.py:92-106`; `datasets.py:122-131` → `test_BLIP2.py:112` → `blip_processors.py:49-68` → `blip2_cir.py:480-486` → `blip2.py:32`. ⇒ **STATIC_AUDIT VERIFIED**; two-stage composition non-interchangeable.

**4. official evaluator → exists as a function; not reachable.** `test_BLIP2.py:9-175`; only `train_BLIP2.py:298`; no `__main__`; `:10` leaves `model.train()` on (CF-2); package dies at `lavis/models/__init__.py:33` (CF-1). ⇒ **NOT RUNNABLE.**

**5. workbench invocation → does not exist.** `adapters/models.py:54-56`; executed → `NotImplementedError`. No `airknow` module in `workbench/replay/`. Full-gallery helpers reached from tests only. `registry.py:701-706` → `adapter_command_ready: False`. ⇒ **BLOCKED.**

**6. checkpoint → identity partial; bytes untouched.** `models.yaml:112-132`; Hub listing (2 files); per-file SHA-256; `HEAD` 200 / 2724056638. `artifacts/checkpoints/airknow/` absent. Condition is metadata-only (UNRESOLVED #1); no loader exists (CF-4b). ⇒ **PARTIAL; bytes/loadability/condition UNKNOWN.**

**7. output → none.** `reproduction_status: NOT_RUN` (`:109`); only `mock/mock_airknow-fiq_n05.json` (`data_kind: "mock"`, null hashes/digests). `metrics_extraction.py:311-320` would return `AGGREGATE_PARSER_UNAVAILABLE` anyway. ⇒ **NOT_RUN.**

**8. metric comparison → transcription verified; reproduction impossible.** registry `:106-108` vs arXiv Table 1 vs `assets/results-fiq.png` → 55.75/75.72 identical; mean 65.74 vs 65.73 (MF-1). Executed arithmetic `55.7533 / 75.7167 / 65.735`. Executed gate: control pass; noise mismatch → `BLOCKED_CHECKPOINT_MISMATCH`; no production caller supplies the metadata (CF-3). ⇒ **transcription VERIFIED (r10, r50), one error (mean), PARITY_NOT_EVALUATED.**

---

## CHECKLIST

*Provenance: the canonical 55-question list is absent from every readable file (UNRESOLVED #9). The probes below are mine, in the required format.*

1. **Registered `model_id` present and unique** — VERIFIED — `models.yaml:81`; `tests/test_protocols.py:30-33` — NONE — none.
2. **`method_name`/`paper_title` match the upstream README** — VERIFIED — `models.yaml:90,94` vs `README.md:5` — NONE — none.
3. **`paper_url` resolves and identifies the same work** — VERIFIED — arXiv API HTTP 200; title + 6 authors match — NONE — none.
4. **Registry `upstream_commit_sha` equals checkout HEAD** — VERIFIED — `models.yaml:98` == `git -C workbench/third_party/AirKnow rev-parse HEAD` — NONE — none.
5. **Checkout remote equals `upstream_repo_url`** — VERIFIED — `git config --get remote.origin.url`; GitHub API `full_name: iLearn-Lab/CVPR26-Air-Know` — NONE — none.
6. **`upstream_default_branch` matches** — VERIFIED — `git branch -a --contains HEAD` → `* main` — NONE — none.
7. **No modified tracked file** — VERIFIED — `git status --porcelain`; non-bytecode count 0; `runtime.py:1829-1837` — P3 — keep `.pyc` out of the pin record.
8. **Pin equals current upstream `main`** — UNKNOWN — single-commit checkout, no fetch permitted — P3 — re-verify on a writable host.
9. **`source_available`/`source_dir` correct** — VERIFIED — `models.yaml:99-100`; tree present — NONE — none.
10. **`native_protocol` matches the source gallery** — VERIFIED — `datasets.py:31,218-224`; `docs/FASHIONIQ_PROTOCOL_PARITY.md:70` — NONE — none.
11. **`supported_protocols` is exactly the native one** — VERIFIED — `models.yaml:105`; the `val-split` branch (`datasets.py:206-217`) is unreachable (`self.split` hardcoded `:31`) — P3 — note it as dead code.
12. **`literature_split_label: original` consistent** — VERIFIED — `:103`; protocol table `:12-15`; paper §4.1 reports per-category FashionIQ R@10/R@50 — NONE — none.
13. **Reference excluded before ranking** — VERIFIED — `test_BLIP2.py:151` mask precedes `:154` argsort — NONE — none.
14. **Mask value and mechanism recorded (mask, not removal)** — VERIFIED — `-10e10` at `:151`; `docs/FASHIONIQ_PROTOCOL_PARITY.md:70`; workbench removes the column (`src/evaluation/fashioniq.py:42-53`) — P3 — keep the distinction visible.
15. **Sort descending by similarity** — VERIFIED — `:154,168` `np.argsort(-sims[i, :])` — NONE — none.
16. **Tie-breaking policy pinned** — UNKNOWN — numpy default unstable; workbench declines to pin (`FASHIONIQ_PROTOCOL_PARITY.md:47-48`) — P3 — accept; measure-zero.
17. **Recall@K maths correct (`100*hits/N`)** — VERIFIED — `:169-174` — NONE — none.
18. **Denominator is the query count** — VERIFIED — `:174` `len(nn_result)` == `len(test_queries)` — NONE — none.
19. **R@1 additionally computed** — VERIFIED — `:169` `for k in [1, 10, 50]` — P3 — harmless; document.
20. **Macro/mean definition matches the paper** — VERIFIED — executed `65.735`; `src/evaluation/fashioniq.py:223-240` — NONE — none.
21. **`reported_r10` matches the paper** — VERIFIED — `:106` == arXiv Table 1 == `assets/results-fiq.png` → `55.75` — NONE — none.
22. **`reported_r50` matches the paper** — VERIFIED — `:107` == both sources → `75.72` — NONE — none.
23. **`reported_mean` matches the paper** — VARIANT — `:108` `65.74` vs both sources `65.73`; executed `round(65.735,2)=65.73` — P2 — correct to `65.73`.
24. **`reported_mean` == `(r10+r50)/2`** — VERIFIED — executed `(55.75+75.72)/2 = 65.735`; paper prints `65.73`; registry rounds the other way (#23) — P3 — pick one convention.
25. **Category cells consistent with the average row** — VERIFIED — executed `55.7533 → 55.75`; `75.7167 → 75.72` — NONE — none.
26. **Reported row's noise condition labelled** — VERIFIED-NEGATIVE — no `paper_row_noise_pct` field (`grep -c` → 0); the row is the `0%` block; variants say `evaluation_noise_pct: 0` (`:118,128`) — **P1** — add the field (CF-3).
27. **Gallery equals the full val split file order** — VERIFIED — `datasets.py:218-224` — NONE — none.
28. **Gallery sizes match the real splits** — VERIFIED (local disk) — 3817/6346/5373, duplicate-free — P3 — not cross-checked upstream (UNRESOLVED #6).
29. **Every reference/target inside its gallery** — VERIFIED — executed `set ⊆ gallery` → `True` ×3 — NONE — none.
30. **Layout id `fashioniq_ilearn_resized` matches source paths** — VERIFIED — `fashioniq_layout.py:25-34` vs `datasets.py:33-35,183,185,187` — NONE — none.
31. **Image extension `.jpg` matches source** — VERIFIED — `datasets.py:155,166`; `fashioniq_layout.py:85` — NONE — none.
32. **Layout checker covers the eval-required files** — VERIFIED — `fashioniq_layout.py:29-33` — NONE — none.
33. **Layout checker covers *all* files the evaluator opens** — VARIANT — cache JSON + 6 PKLs (`datasets.py:38-62`) unchecked; train captions only in `ilearn_resized_training_paths` (`:37-41`) — P2 — cover the cache branch or invalidate on mismatch.
34. **Cache staleness guarded** — NOT_RUN (absent) — `datasets.py:38` gates on existence; `:55-62` never re-reads sources; `UPSTREAM_DATA_PREPARATION_AUDIT.md:51-53` — P2 — require a provenance stamp in `fashion_iq_data.json`.
35. **Correction dictionaries present and compared** — VARIANT — `datasets.py:187`; 0 `airknow` rows in `auxiliary_assets.yaml`; 0 present locally — P2 — register with a digest.
36. **Caption normalization matches source** — VERIFIED — `datasets.py:122-127` — NONE — none.
37. **Composition order and `" and "` joiner** — VERIFIED — `datasets.py:129-131` — NONE — none.
38. **LAVIS eval processor applied to the composed string** — VERIFIED — `test_BLIP2.py:112`; `blip_processors.py:49-68`; `blip2_pretrain.yaml:32-36` — NONE — none.
39. **Tokenizer path pinned in the registry** — VARIANT — `/root/bert-base-uncased` (`blip2.py:32,48,55`); no registry record or fallback — P2 — record as an asset or use an env var.
40. **`max_txt_len` recorded** — VARIANT — source default 32 (`blip2_cir.py:110,140`); config and registry silent — P3 — note in the prep contract.
41. **Image transform recipe matches source** — VERIFIED — `train_BLIP2.py:111-119`; `data_utils.py:92-106` — NONE — none.
42. **Staged resized images have a pinned generation recipe** — VARIANT — no script; 0 `airknow` rows in both registries — P2 — pin a resizer + JPEG quality, or hash the staged tree.
43. **Noise injection touches only training data** — VERIFIED — `datasets.py:66-101`; called only at `train_BLIP2.py:130,143`; 0 `noise_ratio` refs in `test_BLIP2.py` — NONE — none.
44. **Evaluator runs in eval mode / no grad** — VERIFIED-NEGATIVE — `test_BLIP2.py:10` `model.train()`; only the `fashion200k` branch calls `eval()` (`:13`); siblings call `eval()` (`ConeSep/test.py:10`, `HABIT/test.py:10`, `INTENT/test.py:10`) — **P1** — replay verbatim or declare a VARIANT (CF-2).
45. **Checkpoint variants declared with noise provenance** — VERIFIED — `:112-132`; Hub listing matches exactly — NONE — none.
46. **Checkpoint URLs resolve** — VERIFIED (STATIC) — `HEAD …/airknow_fiq_0.5.pt` → 200, `Content-Length: 2724056638` — NONE — none.
47. **Checkpoint sizes recorded** — VARIANT — `:119,129` `null` though the Hub publishes `2724056638` — P2 — record.
48. **Checkpoint sha256 recorded** — VARIANT — `:120,130` `null` though the Hub API exposes `f4d25ed9…`/`756fc0bd…` — P2 — record.
49. **Checkpoint file set matches the author Hub** — VERIFIED — Hub listing = `{.gitattributes, README.md, airknow_fiq_0.5.pt, airknow_fiq_0.8.pt}` — NONE — none.
50. **A clean (0 %) checkpoint is published** — VERIFIED-NEGATIVE — Hub holds exactly `0.5` and `0.8`; card publishes no 0.0 file; repo README has no checkpoint section — NONE (registry correct) — none.
51. **`strict_clean_checkpoint_available` accurate** — VERIFIED — `:110`; executed `select_checkpoints` → `BLOCKED`, `False` — NONE — none.
52. **Checkpoint loadable by pinned code** — UNKNOWN — no loader (`--restore_file` declared `train_BLIP2.py:67`, never read; `grep torch.load/load_state_dict` → no match; only `torch.save(…'best_model.pt')` `:366`); payload untouched — **P1** — CF-4b; needs a run.
53. **Standalone official evaluator exists** — VERIFIED-NEGATIVE — no `__main__`/`argparse` in `test_BLIP2.py` (grep → no match); only `train_BLIP2.py:298` — **P1** — write and audit an entrypoint.
54. **Pinned `lavis.models` package importable** — VERIFIED-NEGATIVE — `lavis/models/__init__.py:33-34` import `blip_cir`/`blip_cir_caption`, **untracked** (executed `git ls-files` + module scan: 2 unresolvable in-repo modules); siblings carry the same imports **and ship the files** — **P1** — commit/guard the modules (CF-1); this alone blocks the model.
55. **Adapter produces a command / evaluator reaches the protocol / output exists and parity compared** — BLOCKED — `adapters/models.py:54-56`, executed `NotImplementedError`; `metrics_extraction.py:311-320` → `AGGREGATE_PARSER_UNAVAILABLE` (asserted `tests/test_adversarial_metrics.py:99-101`); full-gallery helpers reached from tests only; `reproduction_status: NOT_RUN` (`:109`) with only a `data_kind: "mock"` fixture; the comparability gate's `paper_meta` is never constructed — **P1** — no `airknow` number can be produced today.

---

**FINAL VERDICT: BLOCKED** — the registry's static facts for `airknow` check out against the pinned tree (pin, protocol, full split with the reference masked before `argsort`, `strict_clean=false`, two exact paper-metric transcriptions, and a genuinely clean evaluation path), but no runnable path to any number exists: AirKnow alone is missing the two modules its `lavis/models/__init__.py:33-34` imports (the siblings ship the same imports and the files), the registered script has no `__main__`, no checkpoint loader exists, the adapter raises `NotImplementedError`, the registered dataset layout is absent locally, and the paper row a comparison would target is the 0 %-noise row for which the author publishes no checkpoint.",

---

---

## REFRESH (this engagement)

Appended after the work that landed on top of this audit. Earlier findings above are left
as written; where one has been superseded, the superseding fact is stated here.

- **Static benchmark reference.** 4 committed row(s):

  - `airknow_fiq_0pct` — `VERIFIED`, 0% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 55.75, 'r50': 75.72, 'mean': 65.73}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `airknow_fiq_20pct` — `VERIFIED`, 20% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 55.18, 'r50': 75.71, 'mean': 65.45}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `airknow_fiq_50pct` — `VERIFIED`, 50% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 53.19, 'r50': 74.04, 'mean': 63.61}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`
  - `airknow_fiq_80pct` — `VERIFIED`, 80% training noise, protocol `fashioniq_full_gallery_ref_excluded`, values `{'r10': 49.23, 'r50': 70.23, 'mean': 59.73}`, provenance `{'r10': 'PAPER_REPORTED', 'r50': 'PAPER_REPORTED', 'mean': 'PAPER_REPORTED'}`

- **Readiness now:** `BLOCKED_SOURCE` — source cannot run: no runnable evaluation path is registered for this model: the pinned evaluator is a callable with no CLI or checkpoint loader, the adapter deliberately declines to build a command, and the required correction dictionaries and resized images have no published source; the paper row is the 0%-noise row and no 0% checkpoint is published (see workbench/docs/reviews/models for the model's audit). An earlier note in this registry claimed a tensor-shape failure in the similarity computation; that claim was WRONG and is withdrawn - np.matmul broadcasts and the ranking matrix has the shape the indexing expects

- **Reference per registered checkpoint condition:**

  - `fiq_n05` (noise 50): `airknow_fiq_50pct` `VERIFIED` — exact condition match
  - `fiq_n08` (noise 80): `airknow_fiq_80pct` `VERIFIED` — exact condition match

- **Gate:** the run's verdict comes from `workbench/backend/paper_gate.py` against the
  committed row above, reported in the run JSON under `benchmark_gate`. A row that states no
  condition is refused (`BLOCKED_REFERENCE_INCOMPLETE`); a metric the paper never printed is
  compared only if the registry derives it, and then it is labelled `WORKBENCH_DERIVED`.
