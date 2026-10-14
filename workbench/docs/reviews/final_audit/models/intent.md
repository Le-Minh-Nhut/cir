# intent

## MODEL ID
`intent` (INTENT) — `workbench/registry/models.yaml:301-345`. Independently re-parsed the block from a copy at `/tmp/audit-final/intent/models.yaml` (`yaml.safe_load`, output below), so the citations do not rest on the YAML reader.

## PAPER TITLE
`INTENT: Invariance and Discrimination-aware Noise Mitigation for Robust Composed Image Retrieval` — `workbench/registry/models.yaml:315`; identical string at `workbench/third_party/INTENT/README.md:1`. STATIC_AUDIT.

**Paper actually retrieved (this is the change from the prior round, which could only read `figs/fiq.png` and a failed OJS URL).** The arXiv v1 exists and is reachable:

```
$ curl -sL -o intent.pdf https://arxiv.org/pdf/2604.18051v1
$ ls -l intent.pdf && sha256sum intent.pdf
-rw-rw-r-- 1 heheboiz heheboiz 9279156 Oct 15 02:27 intent.pdf
06e606654a1c0c801b91baf16d934d1da1e3246016cead3d89dab51bb28daa69  intent.pdf
$ pdftotext -layout -f 6 -l 6 intent.pdf - | grep -A1 "INTENT (Ours)"
              INTENT (Ours)               50.32   72.10           56.32    74.93   59.28    78.45     55.31     75.16     65.24
              INTENT (Ours)               49.32   71.43           55.32    73.57   58.01    78.46     54.22     74.49     64.36
              INTENT (Ours)               47.99   71.24           52.78    72.48     56.79    76.23     52.52     73.32     62.92
              INTENT (Ours)               42.07   65.58           50.38    69.41     53.09    73.91     48.51     69.63     59.07
```

`read https://arxiv.org/html/2604.18051v1` returns the full manuscript (Table 1 = CIRR, Table 2 = FashionIQ). The PDF byte count and sha256 above reproduce the digest the extracted bundle recorded (`papers.yaml:1281` / `extracted/intent_aaai26.json` `paper.pdf_sha256`) — I recomputed it, did not copy it. The registry's `paper_url` is the OJS landing page (`models.yaml:316`); OJS is not needed and was not retried. The author-rendered `workbench/third_party/INTENT/figs/fiq.png` I read independently matches the arXiv Table 2 digit-for-digit, including the bolded `50.32 / 72.10` in the 0% Dress column. VERIFIED (paper identity + numbers).

## PINNED SOURCE COMMIT
```
$ git -C workbench/third_party/INTENT rev-parse HEAD
2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3
$ git -C workbench/third_party/INTENT log -1 --format='%H %ci %s'
2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3 2026-05-25 18:15:51 +0800 Update README.md
$ git -C workbench/third_party/INTENT status --porcelain | grep -v pycache   ->  (empty)
```
Pin matches checkout (`models.yaml:319` == rev-parse), `source_dir: INTENT` exists (`models.yaml:321`, `source_available: true` at `:320`), and **no source file in the pinned tree was modified by the workbench** — the only untracked paths are `__pycache__/*.pyc`. Upstream identity: `curl -sL https://api.github.com/repos/iLearn-Lab/AAAI26-INTENT` → `full_name iLearn-Lab/AAAI26-INTENT`, `default_branch main`, `fork False`, `pushed_at 2026-05-25T10:15:52Z`; `curl -sL -o /dev/null -w '%{url_effective}' https://github.com/ZivChen-Ty/INTENT` → `https://github.com/iLearn-Lab/AAAI26-INTENT`. One repo, two names. VERIFIED. Whether GitHub's current HEAD still equals the pin is UNKNOWN (no clone/fetch performed).

## CHECKPOINT IDENTITY
`yaml.safe_load` of the registry block:

```
strict_clean False status NOT_RUN
fiq_n02 intent_fiq_0.2.pt 20 0 NO_CLEAN_CHECKPOINT VERIFIED_METADATA None None
fiq_n05 intent_fiq_0.5.pt 50 0 NO_CLEAN_CHECKPOINT VERIFIED_METADATA None None
fiq_n08 intent_fiq_0.8.pt 80 0 NO_CLEAN_CHECKPOINT VERIFIED_METADATA None None
```

Live Hub metadata (STATIC_AUDIT on remote metadata, no download as this box is CPU-only; this is the only kind of runtime evidence available for these models):

```
$ curl -sL 'https://huggingface.co/api/models/iLearn-Lab/AAAI26-INTENT?blobs=true' | python3 -c '...'
sha 16ee2d31f05576bbebbad7da0530b412f5e7a1d8 usedStorage 10889914088
.gitattributes 1519 None
README.md 5102 None
intent_cirr_0.2.pt 2722478522 519939a3b50835aeef2a197cbc1fd73c12f998c4aff5fe19450faeb3ef1c0670
intent_fiq_0.2.pt 2722478522 2a504cadfd6b1c2486a2c508538b0160c5653f62c7cd11f4f3d8a9534dc26afc
intent_fiq_0.5.pt 2722478522 2d59ae002ecbf645a52c7e93bf8afd914e3061e2304413b97a0bf15ca43e39a9
intent_fiq_0.8.pt 2722478522 671bb217d3fdf32e6133fcb98eef4e80ebf831b5a203493c54507ef3c23b5cd2
```

- The three registry filenames exist verbatim (`models.yaml:335,346,357`). VERIFIED.
- **New this round: every file publishes an `lfs.sha256`**, and all four are 2 722 478 522 B. The registry records `expected_size_bytes: null` and `expected_sha256: null` for all three (`models.yaml:340-341,351-352,362-363`). This is now a concrete, actionable gap, not just an absence: a size change to any of these files (HF re-upload, sharded re-pack) would be invisible to the workbench. P2.
- **No 0%-noise FashionIQ checkpoint exists.** The Hub README (`curl -sL https://huggingface.co/iLearn-Lab/AAAI26-INTENT/raw/main/README.md`) says only "Download the checkpoint files (e.g., `best_model.pth`)" and "trained and evaluated on standard CIR datasets under various noise ratios" — it never names a clean file. So `strict_clean_checkpoint_available: false` (`models.yaml:331`) is correct. VERIFIED.
- `checkpoint_mapping_status: VERIFIED_METADATA` (`models.yaml:343,354,365`) is honest as far as it goes: the filename↔noise mapping rests on filenames only. **Whether `intent_fiq_0.2.pt` actually encodes 20% shuffling is UNKNOWN** — nothing was downloaded, and the Hub README does not publish a per-file noise table.

## EVALUATION PROTOCOL
Registry: `native_protocol: fashioniq_full_gallery_ref_excluded`, `supported_protocols: [fashioniq_full_gallery_ref_excluded]`, `literature_split_label: original` (`models.yaml:324-326`).

Traced STATIC_AUDIT through the pinned source:
- Only caller: `train_INTENT.py:284` `t = test.test(args, model, trainset, category, txt_processors)`, looped over `['dress','shirt','toptee']` at `train_INTENT.py:283`, inside `train_and_evaluate` invoked at `train_INTENT.py:380` from `__main__` (`:347`).
- Split default: `datasets.py:26` `def __init__(self, path, category=None, transform=None, noise_ratio=0, split='original-split')`; CLI default `train_INTENT.py:59`.
- Query set: `test.py:82-86` picks `testset.test_queries_{dress,shirt,toptee}`; built at `datasets.py:174-187` from `cap.{cat}.val.json` (`datasets.py:170`) with `source_img_id = images.index(candidate)` (`:181`) and `target_img_id = images.index(target)` (`:183`).
- Gallery: `datasets.py:203-209` (`elif self.split == 'original-split'`) enumerates every id in `image_splits/split.{cat}.val.json` in file order. The `val-split` branch (`datasets.py:191-202`) is dead under the default.
- Reference exclusion: `test.py:150` `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10`, before `np.argsort(-sims[i, :])[:50]` at `test.py:153`.
- Recall: `test.py:167-174`, `for k in [1, 10, 50]`, hit iff `test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]`, denominator = query count.

Workbench side: `src/evaluation/fashioniq.py:175-177` declares `fashioniq_full_gallery_ref_excluded` with `gallery="full_split"`, `reference_excluded=True`; `build_fashioniq_gallery` (`:204-219`) returns `build_original_gallery` (`:121-129`) = the ordered split file; `recall_at_k_reference_excluded` (`:35-77`) removes the reference column and searches the target; `macro_average_fashioniq` (`:238-255`) macro-averages per category and sets `mean_recall = (recall_at_10 + recall_at_50)/2`.

**Verdict: VERIFIED** for protocol id, gallery scope, ordering, reference policy and K-set — every element traces to a line in the pinned evaluator. **VARIANT** only on the exclusion *mechanism*: source masks to `-10e10` then sorts; workbench removes the column. Same cohort, different arithmetic, and the workbench docstring already concedes the dependence on the score bound (`src/evaluation/fashioniq.py:47-52`).

## DATASET CONTRACT
- `fashioniq_layout: fashioniq_ilearn_resized`, `fashioniq_image_extensions: [".jpg"]` (`models.yaml:322-323`). Required path list: `workbench/backend/fashioniq_layout.py:24-36` — `captions/`, `image_splits/`, `resized_image/`, `resized_image/{dress,shirt,toptee}`, `cap.{cat}.val.json`, `correction_dict_{cat}.json`, `split.{cat}.val.json`. Image-directory map: `fashioniq_layout.py:85-86` `"fashioniq_ilearn_resized": ("resized_image", True)`.
- Source opens exactly that: `datasets.py:31-33` (`resized_image`, `image_splits`, `captions`), `datasets.py:139`/`:150` `os.path.join(self.image_dir, name.split('_')[0], name.split('_')[1] + ".jpg")`.
- `preparation_contract: manual_ilearn_resized` (`models.yaml:314`) is a **layout name, not a registered contract**: `workbench/registry/preparation_contracts.yaml` contains only `clvc_net_fashioniq, dcnet_fashioniq, clip4cir_fashioniq, tgcir_fashioniq, sprc_fashioniq, limn_fashioniq` (`grep -n 'preparation_id:'`), and `grep -rn intent workbench/registry/preparation_contracts.yaml workbench/registry/auxiliary_assets.yaml` → **exit 1, zero matches in both**. `workbench/backend/registry.py:73` permits the literal, so the declaration validates but is inert.
- Local data root `data/fashionIQ_dataset/`: `captions/` (only `cap.*.{test,train,val}.json`), `image_splits/`, `images/` (74381 files, **extension `png`**), `qwen_captions/`. **`resized_image/` absent. `ls data/fashionIQ_dataset/captions/correction_dict_*` → no such file.**
- OBSERVED split statistics (read the local JSON directly): dress 3817 gallery / 2017 queries; shirt 6346 / 2038; toptee 5373 / 1961; every split duplicate-free. Matches `READINESS_MATRIX.md:30`'s missing-path list and the workbench's own uniqueness asserts (`src/evaluation/fashioniq.py:128,134`).
- **The upstream README documents a directory layout that INTENT's own code cannot read.** `README.md`'s FashionIQ tree has `captions/`, `image_splits/`, and flat `dress/ shirt/ toptee/`; `grep -n 'resized\|correction_dict' workbench/third_party/INTENT/README.md` → **no matches**. The pinned `datasets.py:139,150` hard-codes `resized_image/<category>/<id>.jpg`, and `datasets.py:98,172` hard-codes `correction_dict_{cat}.json`. So the README's data-prep section is insufficient to run the shipped code, and `PC_TRANSFER_CHECKLIST.md:56` is right that the dicts are "not shipped and not published".

**Verdict: BLOCKED** — P1 for the missing dictionaries (the caption string is a model input, and `concat_text` composes it verbatim into the query), P1 for `resized_image/`.

## PREPROCESSING CONTRACT
- `train_INTENT.py:79-81` forces `transform = "targetpad"`, `input_dim = 224`, `target_ratio = 1.25` → `:87` `targetpad_transform(1.25, 224)`; pipeline `data_utils.py:92-105` = `TargetPad(1.25,224)` → `Resize(224, BICUBIC)` → `ToTensor` → CLIP `Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))`.
- Eval applies only this transform: `datasets.py:149-157` `get_test_img` opens `PIL.Image`, `convert('RGB')`, `self.transform(img)` — no random op. (The `original-split` gallery uses `get_img`, `datasets.py:138-147`, which applies the transform under `if self.transform`; the transform is always set by `load_dataset`, so no behavioural difference at eval.)
- Caption composition at eval: `datasets.py:170-177` reads `cap.{cat}.val.json` then `correction_dict_{cat}.json`, then `mod_str = self.concat_text(caption['captions'], correction_dict)`; `correct_text` (`:110-115`) lowercases, maps `string.punctuation`→space, tokenizes, exact dictionary substitution; `concat_text` (`:117-119`) joins with `" and "`.

**Verdict: VERIFIED** for the deterministic TargetPad@1.25/224 evaluation transform. The caption normalization is VERIFIED as code, but depends on the unpinned dictionaries (above).

## AUGMENTATION CONTRACT
- Registry declares no augmentation policy; `evaluation_noise_pct: 0` for all three variants (`models.yaml:339,350,361`).
- Evaluation-side augmentation: none. `test.test` calls `model.eval()` (`test.py:10`); both eval feature extractors are `@torch.no_grad()` (`lavis/models/blip2_models/blip2_cir.py:721,771`). `FashionIQ.shuffle()` (`datasets.py:63`) is the only corruptor and is invoked only at `train_INTENT.py:101`, immediately after construction. Inside it, `noise_ratio == 0` returns before touching data (`datasets.py:65-67`).
- Training-only machinery never reached at eval: `colorful_spectrum_mix` (`blip2_cir.py:57`, called from `forward`), `text_proj_cau` (`blip2_cir.py:171`, used at `:523,526` inside training losses).

**Verdict: VERIFIED** (training-only).

## GALLERY CONTRACT
- Full split, file order, no sort, no dedup beyond the file's own uniqueness: `datasets.py:204-205` `for id, image_name in enumerate(images): test_targets_id.append(id)`.
- Reference excluded per query at ranking time, not by shrinking the gallery: `test.py:149-150`.
- Workbench equivalent: `src/evaluation/fashioniq.py:175-177` + `:204-219` + `:35-77`.

**Verdict: VERIFIED.** Under `original-split`, `test_targets_id` is `range(len(split))`, so `test_targets_id.index(...)` at `test.py:150` and `:171` is the identity map and the mask lands in-bounds.

## METRIC CONTRACT
- Source emits `('{name}_r{k}', r)` for k in `[1,10,50]` (`test.py:168-174`) — per-category R@1/R@10/R@50, each `100 × hits/query_count`.
- Checkpoint-selection metric is a **sum** of per-category R@10 (`train_INTENT.py:286`), index `[1]` being R@10 given the `1,10,50` order (`test.py:168`). The trainer never computes a macro.
- Workbench emits macro R@10/R@50 plus `mean_recall` (`src/evaluation/fashioniq.py:238-255`). The registry does not claim the source prints a macro, and the paper's "Average" column is a published macro — so the aggregate is comparable to the paper, not to a source-printed metric.

**Verdict: VERIFIED** for source per-category metrics; the workbench macro is an aggregate, correctly labelled as such.

## SOURCE-TO-WORKBENCH PARITY
| Element | Pinned source | Workbench | Verdict |
|---|---|---|---|
| Protocol id | `datasets.py:203` + `test.py:150` | `fashioniq_full_gallery_ref_excluded` (`models.yaml:325`; `src/evaluation/fashioniq.py:175`) | VERIFIED |
| Gallery | ordered `split.{cat}.val.json` (`datasets.py:204-205`) | `build_original_gallery` (`src/evaluation/fashioniq.py:121-129`) | VERIFIED |
| Reference rule | mask `-10e10` then argsort (`test.py:150,153`) | remove column then argsort (`src/evaluation/fashioniq.py:70-75`) | VARIANT (equivalent cohort) |
| K set | `[1,10,50]` (`test.py:168`) | `recall_at_10`, `recall_at_50` (`src/evaluation/fashioniq.py:112-119`) | VERIFIED |
| Preprocessing | targetpad 1.25/224 (`train_INTENT.py:87`) | declared `fashioniq_ilearn_resized` (.jpg, per-category) | VERIFIED |
| Score arithmetic | `matmul → squeeze → max(-1)` (`test.py:141-143`) | workbench evaluators take a precomputed 2-D score matrix | **VERIFIED equivalent** — see CRITICAL FINDINGS for the executed check |
| Evaluation command | `test.test` is a callable; `grep -c '__main__\|argparse' test.py` → `0` | `IntentAdapter` (`workbench/backend/adapters/models.py:69-71`) inherits `OfficialScriptAdapter.command`, which raises `NotImplementedError` (`models.py:24-25`) | NOT_RUN |
| Executable model implementation | full BLIP-2 Q-Former INTENT model | none: `ls src/models/` → `iag_srme`, `taper_mag` only; no workbench INTENT model | BLOCKED |

`workbench/scripts/run_eval.py:25` would raise for `--model intent` (`ADAPTERS[args.model]().build_command(request)` → inherited stub).

## RUNTIME STATUS
**NOT_RUN / BLOCKED.** No runtime claim is made anywhere in this document (no GPU/MPS available; nothing downloaded). Reasons, each traceable:
1. `train_INTENT.py` trains `--num_epochs 20` before evaluating (`:49`, `:380`), has **no** checkpoint-restore path, and `utils.load_checkpoint` (`utils.py:128-143`) has zero callers in the root scripts (`grep -rn 'load_checkpoint' train_INTENT.py test.py cirr_sub_BLIP2.py` → no match). The only `torch.load` outside `lavis/` is `cirr_sub_BLIP2.py:128`.
2. `cirr_sub_BLIP2.py` is CIRR-only: `datasets.CIRR(path="/root/autodl-tmp/cirr_data/CIRR/")` (`:122`) and a `torch.load` sweep over a hardcoded `/root/autodl-tmp/BLIP/test/` directory (`:124-129`). It is not a FashionIQ evaluator. The README's documented invocation (`README.md:188-190`, `--checkpoint_path … --output_file …`) **passes arguments the script has no parser for** (`grep -n argparse cirr_sub_BLIP2.py` → no match), even though the same invocation is repeated in the Hub README's Step 3.
3. Data assets missing locally: `resized_image/` and `correction_dict_*.json` (DATASET CONTRACT).
4. `IntentAdapter` has no `command()` override, so no official command can be constructed (P1).
5. `workbench/backend/metrics_extraction.py:309-341` has parsers only for `limn` and `csmcir`; `intent` falls through to `AGGREGATE_PARSER_UNAVAILABLE` / `PARSER_UNAVAILABLE`, so even a successful run's stdout would not be turned into metrics or a parity status.
6. `workbench/artifacts/checkpoints/` contains no INTENT checkpoint (`find … -iname '*intent*'` → nothing); `workbench/artifacts/reports/` has no intent run.

## PAPER-METRIC STATUS
**Re-derived from the real paper this round** (arXiv 2604.18051v1 HTML + PDF page 6 + author figure `figs/fiq.png`), not from any workbench doc. Table 2, `INTENT (Ours)` rows:

| block | Dress R@10/R@50 | Shirt | Toptee | Avg R@10 | Avg R@50 | AVG. |
|---|---|---|---|---|---|---|
| 0% | 50.32 / 72.10 | 56.32 / 74.93 | 59.28 / 78.45 | 55.31 | 75.16 | 65.24 |
| 20% | 49.32 / 71.43 | 55.32 / 73.57 | 58.01 / 78.46 | 54.22 | 74.49 | 64.36 |
| 50% | 47.99 / 71.24 | 52.78 / 72.48 | 56.79 / 76.23 | 52.52 | 73.32 | 62.92 |
| 80% | 42.07 / 65.58 | 50.38 / 69.41 | 53.09 / 73.91 | 48.51 | 69.63 | 59.07 |

Registry `reported_r10/r50/mean = 55.31 / 75.16 / 65.24` (`models.yaml:327-329`) and all four rows in `papers.yaml:1232,1289,1340,1389` (each `reference_status: VERIFIED`, independent `yaml.safe_load` confirms) match the paper digit-for-digit.

Arithmetic re-derived (Decimal ROUND_HALF_UP, `/tmp/audit-final/intent`):
- macro R@10 = (50.32+56.32+59.28)/3 = 55.3066… → **55.31** = printed; macro R@50 = (72.10+74.93+78.45)/3 = **75.16** = printed. The Average column *is* the macro of its own category columns.
- (55.31+75.16)/2 = 65.235 → **65.24** = printed AVG. But the mean of the **unrounded** macros is 65.2333… → 65.23. The printed mean uses the printed Average cells.
- Same check on 20%: macro R@10 54.22, macro R@50 74.49, printed AVG 64.36; unrounded mean of macros = **64.35**. So the 0.01 discrepancy is not a one-off in the 0% row. (50% and 80% happen to agree either way: 62.92 and 59.07.)
- The workbench computes `mean_recall` from unrounded inputs (`src/evaluation/fashioniq.py:254`), i.e. it would produce 65.23/64.35 where the paper printed 65.24/64.36. Rounding convention, P3.
- The paper's Section 4.2 corroborates the 20% row: `σ=20%, INTENT surpasses TME by 0.39% on Avg`; 64.36 − 63.97 = 0.39. ✔
- R@1 is computed by the source (`test.py:168`) but Table 2 prints no R@1 column and the registry has none. NONE.

**No local metric was produced.** The comparison is paper-vs-registry only.

## CRITICAL FINDINGS
1. **The prior round's P0 is refuted, and I re-derived the refutation from the real shapes.** `extract_retrieval_compose` returns `fusion_feats.unsqueeze(1).unsqueeze(1)` (`blip2_cir.py:769`) where `fusion_feats = F.normalize(self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1)` (`:765-766`) and `text_proj: Linear(hidden, 256)` (`:170`, `embed_dim=256` at `:146`) → `(B, 1, 1, 256)`. `extract_retrieval_target` returns `image_features.permute(0, 2, 1)` (`:792`) where `image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)` over the Q-Former output `(B, num_query_token=32, hidden)` (`:142`, `:790-791`) → `(G, 256, 32)`. Executed numpy (2.3.5):
   ```
   q (2017, 1, 1, 256) g (3817, 256, 32)
   matmul -> (2017, 3817, 1, 32)
   squeeze -> (2017, 3817, 32)
   max(-1) -> (2017, 3817)
   2-D mask write ok: -1e+11
   argsort row -> [2549 2535 2536]
   ```
   And the broadcast is not merely shape-compatible, it computes the intended quantity: with small tensors, `matmul→squeeze→max(-1)` at `[i,j]` equals `max_t <q_i, g_j[:,t]>` for every pair I tried. So there is **no defect here**, and the registry's withdrawal note (`models.yaml:302`, last sentence) is **CORRECT**. Severity **NONE** for the code. But `workbench/docs/reviews/models/intent.md` still carries the refuted claim as its CRITICAL FINDING 1 and CHECKLIST 34 at **P0** (lines 128-131, 217), with no supersession marker preceding it; only the appended REFRESH block contradicts it. That is a live P2 documentation defect: the file reads as an open P0 against a model whose registry says the opposite.
2. **No clean FashionIQ checkpoint exists and the paper's headline row has no published weights.** Hub siblings are exactly `intent_cirr_0.2.pt` + `intent_fiq_0.2/0.5/0.8.pt` (observed above), and the Hub README names no clean file. `evaluation_noise_pct` must be 0 by construction (`workbench/backend/registry.py:84-85` raises otherwise, and all three variants declare 0 at `models.yaml:339,350,361`), while `strict_clean` selection requires `checkpoint_training_noise_pct == 0` (`workbench/backend/checkpoint_selection.py:126-129`) — so the strict-clean cohort for `intent` is empty and `select_checkpoints` refuses. `workbench/tests/test_checkpoint_selection.py:130` pins that. **But the non-strict policy is not empty**: `include_flagged_conditions` plus an explicit `requested=["fiq_n02"]` is accepted whenever the variant's mapping is `VERIFIED_METADATA` and it is obtainable (`checkpoint_selection.py:138-149`, all three variants have `download_url`), because `_condition(20) == 0` is False but the explicit-request branch never re-tests the condition. On that path `run_meta` carries `checkpoint_training_noise_pct: 20` (`workbench/scripts/evaluate_models.py:137-140,152`), which the gate compares against the reference's stated condition (`paper_gate.py:72-79`, `:211-258`) — so the 20%/50%/80% rows are usable references for the corresponding noise-trained socket, the 0% row is not. **P1**: the paper's headline numbers can never be reproduced from published weights, by construction.
3. **`preparation_contract: manual_ilearn_resized` names no contract, so the declaration is inert, and the two assets it implies are absent and unpinned.** `correction_dict_{dress,shirt,toptee}.json` and `resized_image/{cat}/*.jpg` are not generated, not downloaded, not hash-verified, and not present locally; `auxiliary_assets.yaml` has zero `intent` entries; `AUXILIARY_ASSET_AUDIT.md:358-360` classifies all three dicts `OFFICIAL_ASSET_UNAVAILABLE`; `PC_TRANSFER_CHECKLIST.md:56` agrees. **P1** — the caption string is a model input (`datasets.py:177`).
4. **The declared evaluation command does not exist.** `IntentAdapter` (`workbench/backend/adapters/models.py:69-71`) has no `command()` override → inherited `NotImplementedError` (`models.py:24-25`). The registry records `source_blocker` (`models.yaml:302`) and `reproduction_status: NOT_RUN` (`:330`), so the fact is recorded; this is a deliberate, documented refusal, consistent across `AUTOMATION_RUNBOOK.md:157`, `AUTOPILOT_GUIDE.md:74,88`, `PC_RELEASE_READINESS.md:40`, `FINAL_REPRODUCIBILITY_AUDIT.md:28`. **P1** as an external blocker, not a workbench defect to patch.
5. **The released training entry point's defaults contradict the paper's stated recipe.** Paper §4.1: "trained INTENT for **10 epochs** … initial learning rate of **4e-5** … batch size is set to **128**" and "μ=**0.2**, α=**0.6**". Pinned code: `--num_epochs 20` (`train_INTENT.py:49`), `--lr 2e-5` (`:54`), `--mu 0.5` (`:56`), `--alpha 1.0` (`:57`). Q=32 and D=256 do match (`blip2_cir.py:142`, `:146`; paper "we set Q … to 32 … D is set to 256"). So any train-then-evaluate route, even if a checkpoint loader existed, would not reproduce the paper's numbers without operator-supplied overrides. **P2.**
6. **The noise machinery cannot leak into evaluation inputs** — `shuffle()` is called only at `train_INTENT.py:101`; `test.test` reads only `test_queries_*`/`test_targets_*` built from `cap.{cat}.val.json` (`datasets.py:170`); `noise_ratio == 0` returns at `datasets.py:65-67`. **This holds, P0 NONE.** One residual caveat at P3: `datasets.py:37-61` caches query/target lists to `test_queries_*.pkl` and reuses them whenever `fashion_iq_data.json` exists, with no freshness or provenance check — a stale cache from a differently-prepared dataset root would be silently used.

## MINOR FINDINGS
1. **`intent` has no aggregate-output parser.** `workbench/backend/metrics_extraction.py:309-341` dispatches only `limn`/`csmcir`; everything else returns `AGGREGATE_PARSER_UNAVAILABLE` (`:329-341`). Combined with `paper_gate.py`'s category comparison (`:348-368`), a future INTENT run would be judged on whatever aggregate the operator hand-supplied, and its 6 published category cells would be recorded as unchecked. P2 for whoever runs first.
2. **Registry `expected_size_bytes`/`expected_sha256` are `null` while the Hub now publishes both** (`models.yaml:340-341,351-352,362-363`; observed sizes and `lfs.sha256` above). P2 — populate them; the digests were observable in one request.
3. **Registry `notes` understates the checkpoint situation** (`models.yaml:332`): the Hub also ships `intent_cirr_0.2.pt`, which the three `checkpoint_variants` do not record. P3, cosmetic.
4. **`README.md:123` tells the user to `pip install -r requirements.txt`, and no such file exists** (`ls workbench/third_party/INTENT/requirements.txt` → No such file; `git ls-files | grep -i require` → empty). The same broken instruction is echoed in the Hub README's Step 1. The env declaration (`models.yaml:305-310`, five pinned packages) is the only usable source and matches `README.md:112,116,119`. P2 for anyone provisioning from the README alone.
5. **`AUXILIARY_ASSET_AUDIT.md:203` claims `train_INTENT.py` "defines argument `--restore_file`"** — `grep -rn restore_file workbench/third_party/INTENT/ --include=*.py` → **no matches** (the flag exists in HABIT/train.py and ConeSep/train.py, per the same file's lines 149,176; this looks like cross-model contamination). The substantive claim (no checkpoint-restore routine) is correct. P3, doc hygiene.
6. **`registry.environment.cuda: '12.6'`** (`models.yaml:309`) vs the README's CUDA-12.1 wheels (`README.md:116`). P3.
7. **`literature_split_label: original`** is correct against the source default, but the pinned `datasets.py:191-202` also ships a `val-split` branch the registry does not record; `FASHIONIQ_PROTOCOL_PARITY.md:73,84-87` does. P3.
8. **The paper itself never states the gallery construction or reference policy.** I searched the arXiv v1 HTML for `gallery` and for `fully-supervised`/`fully supervised`: `grep -o 'gallery[^<]*' intent.html` → no matches, and **the string "fully-supervised" is absent from arXiv v1** (`grep -o 'fully-supervised' intent.html | wc -l` → 0). The "0% noise setting is equivalent to the traditional fully-supervised CIR paradigm" sentence exists only in `README.md:70-71` and the Hub README. So the registry's `gallery_protocol` footnote (`papers.yaml:1238`: "the paper prints no gallery size or revision") is right, and the fully-supervised reading of the 0% block rests on the author READMEs, not on the paper. P3 — worth one sentence in the row.
9. **`train_INTENT.py:19` `import clip`** is unused (`grep -n 'clip\.' train_INTENT.py` → nothing) and undocumented in the README's dependency list. P3, import-time failure mode only if the trainer is ever invoked.

## UNRESOLVED EVIDENCE
- **No run of any kind.** No checkpoint downloaded, no weights loaded, no FashionIQ inference, no GPU. Every metric statement above is from the paper, from arithmetic on paper numbers, or from numpy shape semantics executed in `/tmp`.
- **Checkpoint payload identity**: whether `intent_fiq_0.2/0.5/0.8.pt` encode the stated training noise levels, and whether they load into the pinned `Blip2QformerCir` through a loader that does not exist, is UNKNOWN. The Hub `lfs.sha256` values are now observed but nothing local to compare them against.
- **Whether the paper's numbers are reachable from the released code at all**: the training defaults contradict the paper (finding 5) and no restore path exists, so even a full GPU environment cannot replay the published recipe from this pin. UNKNOWN whether an unshipped internal revision differs.
- **GitHub remote HEAD vs pin**: not compared; no clone/fetch performed.
- **OJS landing page**: not fetched (arXiv PDF/HTML is the better source and matches the author figure); `models.yaml:316`'s OJS URL is therefore only corroborated by the arXiv title and the author README badge.
- **Hub README authored digest**: the HF README content I quoted is from a live fetch, but I did not verify the 5102-byte `README.md` blob digest against its published hash.

## FINAL VERDICT
**BLOCKED** — the registry's protocol, gallery, reference rule, preprocessing and metric definitions trace faithfully to the pinned `INTENT` source, the paper's Table 2 is now re-derived from the real arXiv manuscript (four rows, digit-for-digit) and the withdrawn tensor-shape claim is confirmed WRONG by an executed shape check, but no runnable evaluation path exists (callable-only evaluator, no restore code, adapter deliberately refuses a command), the two required input assets have no published source, only noise-trained weights exist for a 0%-noise headline row, and the released training defaults contradict the paper's stated recipe — so nothing can be reproduced.

---

# EXECUTION TRACE

`paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation -> checkpoint -> output -> metric comparison`

1. **paper/README/config ->** OJS not fetcehd; the real paper is `arxiv.org/pdf/2604.18051v1`, 9 279 156 B, sha256 `06e606…daa69` (recomputed by me), `pdftotext -layout -f 6` reproduces all four `INTENT (Ours)` rows, arXiv HTML `:1-560` reproduces them independently, and `figs/fiq.png` (1914x1160, read as an image) shows the identical table. DOI/title also in `extracted/intent_aaai26.json`. Environment: `README.md:112,116,119` (`python 3.9`, `torch==2.1.0`, `open-clip-torch==2.24.0 scikit-learn==1.3.2 transformers==4.25.0 salesforce-lavis==1.0.2 timm==0.9.16`) == `models.yaml:305-310`. `README.md:123` points at a non-existent `requirements.txt`. **Evidence: LIVE fetch (curl + read) + grep.**
2. **-> source dataset:** `datasets.FashionIQ` (`datasets.py:26`) with `path = args.fashioniq_path` (`train_INTENT.py:95-100`), reading `cap.{cat}.{train,val}.json` (`datasets.py:96,170`), `split.{cat}.val.json` (`:168`), `correction_dict_{cat}.json` (`:98,172`), `resized_image/{cat}/{id}.jpg` (`:139,150`). Local root has `captions/`, `image_splits/`, `images/` (74381 `.png` files) but **no** `resized_image/` and **no** correction dicts (observed `ls`); `READINESS_MATRIX.md:30` lists the same missing paths. **Evidence: `ls` + OBSERVED JSON stats (3817/6346/5373 galleries, 2017/2038/1961 queries).**
3. **-> preprocessing:** `train_INTENT.py:79-81` fixed `targetpad` / `input_dim 224` / `target_ratio 1.25` -> `:87` -> `data_utils.py:92-105`; eval applies it once in `get_test_img` (`datasets.py:149-157`). Captions normalized at `datasets.py:110-119`. **Evidence: source lines.**
4. **-> official evaluator:** `train_INTENT.py:283-284` -> `test.test` (`test.py:9`); gallery `datasets.py:203-209`; mask `test.py:150`; rank `test.py:153`; recall `test.py:167-174`. Metric delivery is arithmetically sound: `matmul → squeeze → max(-1)` on `(B,1,1,256) x (G,256,32)` yields `(B,G)` and equals the intended max-over-target-tokens similarity (executed). **Evidence: code lines + one executed numpy check.**
5. **-> workbench invocation:** `IntentAdapter` (`adapters/models.py:69-71`) has no `command()`; inherited stub raises `NotImplementedError` (`adapters/models.py:24-25`); `run_eval.py:25` would raise; no workbench INTENT model exists (`src/models/` -> `iag_srme`, `taper_mag`); no intent parser in `metrics_extraction.py:309-341`. **Evidence: file listing + source lines.**
6. **-> checkpoint:** registry URLs (`models.yaml:336,347,358`) exist on the Hub with sizes 2 722 478 522 B and published `lfs.sha256` (observed via `?blobs=true`); registry `expected_size_bytes`/`expected_sha256` are `null`; **nothing downloaded**; no local artifact (`find workbench/artifacts/checkpoints -iname '*intent*'` -> empty). **Evidence: LIVE HTTP metadata.**
7. **-> output:** none. No run, no result JSON, no ranking, no per-query export, no report. **Evidence: no `intent` artifact anywhere under `workbench/artifacts/`.**
8. **-> metric comparison:** paper Table 2 (real arXiv PDF, page 6) `R@10 55.31`, `R@50 75.16`, `AVG 65.24` for the 0% block; registry `models.yaml:327-329` reproduces them verbatim; `papers.yaml:1232,1289,1340,1389` carry all four rows, each `VERIFIED`, cross-checked by independent `yaml.safe_load` and against `extracted/intent_aaai26.json`. Arithmetic re-derived for every block; the printed AVG. equals the mean of the two *printed* Average cells, not of the unrounded macros (0%: 65.235 vs 65.2333; 20%: 64.36 vs 64.35). **No local metric was produced**, so the comparison is paper-vs-registry only.

---

# CHECKLIST

Format: `N. short form — verdict — evidence — severity — action`.

1. Paper title matches registry — VERIFIED — `models.yaml:315` == `INTENT/README.md:1` — NONE — none.
2. Paper retrievable — VERIFIED — `curl -sL https://arxiv.org/pdf/2604.18051v1` -> 9279156 B, sha256 `06e606…daa69`; `pdftotext -layout -f 6 -l 6` reproduces all four rows; `read https://arxiv.org/html/2604.18051v1` returns the manuscript — NONE — none (registry `paper_url` is OJS; consider recording the arXiv URL as well).
3. OJS `paper_url` — UNKNOWN — not fetched this round; the arXiv manuscript is the better source and matches the author figure — P3 — none.
4. Upstream repo identity — VERIFIED — GitHub API `iLearn-Lab/AAAI26-INTENT`, `default_branch main`, `fork False`; `curl -L` of `github.com/ZivChen-Ty/INTENT` -> `iLearn-Lab/AAAI26-INTENT` — NONE — none.
5. Pinned sha matches registry — VERIFIED — `models.yaml:319` == `git -C workbench/third_party/INTENT rev-parse HEAD` == `2a75fc6f57d1a036bf3167a7fe1d2cee072f51f3` — NONE — none.
6. Tracked tree clean — VERIFIED — `git status --porcelain | grep -v pycache` -> empty — NONE — none.
7. `source_dir` present and consistent — VERIFIED — `models.yaml:320-321`; directory holds `test.py`, `datasets.py`, `train_INTENT.py`, `lavis/` — NONE — none.
8. Checkpoint filenames match Hub — VERIFIED — API siblings `intent_fiq_0.2.pt`, `intent_fiq_0.5.pt`, `intent_fiq_0.8.pt` == `models.yaml:335,346,357` — NONE — none.
9. Checkpoint URLs resolve — VERIFIED — HF API `?blobs=true` lists all three with `size 2722478522` — NONE — none.
10. Published checkpoint digests — VERIFIED remote / UNKNOWN local — `lfs.sha256` `2a504cad…`, `2d59ae00…`, `671bb217…`, `519939a3…` observed; registry `null` — P2 — populate `expected_sha256`/`expected_size_bytes` (`models.yaml:340-341,351-352,362-363`).
11. Noise coverage 20/50/80 — VERIFIED — Hub siblings + `models.yaml:338,349,360` — NONE — none.
12. Clean 0% FashionIQ checkpoint — VERIFIED absent — Hub has only `intent_cirr_0.2.pt` + 3 FIQ noise files; Hub README names no clean file — P1 — the 0% row is not reproducible from published weights; say so in the row.
13. `strict_clean_checkpoint_available` — VERIFIED (correct value) — `models.yaml:331` false; `checkpoint_selection.py:126-130`; `tests/test_checkpoint_selection.py:130` — NONE — none.
14. Non-strict selection reachable — VERIFIED — `checkpoint_selection.py:138-149` accepts an explicit `requested` id when the mapping is `VERIFIED_METADATA` and obtainable, without re-testing the noise condition; `evaluate_models.py:137-140,152` records the condition into `run_meta` — NONE — none (informational; the gate then refuses a mismatched condition).
15. `checkpoint_mapping_status` — VERIFIED (metadata-level); payload identity UNKNOWN — `models.yaml:343,354,365`; Hub README publishes no per-file noise table — P2 — mark payload identity explicitly unverified.
16. Registry `notes` completeness — VARIANT — `models.yaml:332` omits `intent_cirr_0.2.pt` — P3 — extend the note.
17. `native_protocol` id — VERIFIED — `models.yaml:325` == `src/evaluation/fashioniq.py:175-177` — NONE — none.
18. Source default split matches protocol — VERIFIED — `datasets.py:26` and `train_INTENT.py:59` default `'original-split'`; branch `datasets.py:203` — NONE — none.
19. Source's second gallery branch — VERIFIED — `datasets.py:191-202` `val-split` (dead under default) — P3 — registry could note it (`FASHIONIQ_PROTOCOL_PARITY.md:73` does).
20. Gallery = full validation split — VERIFIED — `datasets.py:204-205` — NONE — none.
21. Gallery ordering = file order — VERIFIED — `enumerate(images)` over `split.{cat}.val.json`; workbench `build_original_gallery` (`src/evaluation/fashioniq.py:121-129`) — NONE — none.
22. Gallery uniqueness — VERIFIED — OBSERVED 3817/6346/5373 unique; `src/evaluation/fashioniq.py:128,134` assert it — NONE — none.
23. Query universe from `cap.{cat}.val.json` — VERIFIED — `datasets.py:170-187` — NONE — none.
24. Query->gallery mapping — VERIFIED — `datasets.py:181,183` `images.index(...)`; `test_targets_id == range(len(split))` under `original-split` — NONE — none.
25. Reference exclusion present — VERIFIED — `test.py:150` — NONE — none.
26. Exclusion mechanism equivalence — VARIANT — mask `-10e10` (`test.py:150`) vs column removal (`src/evaluation/fashioniq.py:70-75`); docstring `:47-52` concedes the bound dependence — P3 — none (equivalent cohort).
27. Ranking shape and indexing — VERIFIED — executed `matmul→squeeze→max(-1)` on the real shapes -> `(2017,3817)`; mask writes through; `argsort` works — P0 NONE — prior round's P0 was wrong; correct `workbench/docs/reviews/models/intent.md:128-131,217` — P2 doc fix.
28. Recall@K = target in top-k — VERIFIED — `test.py:170-173` — NONE — none.
29. K values {1,10,50} — VERIFIED — `test.py:168` — NONE — none.
30. Sorting tie policy — VARIANT — `np.argsort(-sims[i,:])[:50]` (`test.py:153`) not guaranteed stable; workbench documents tie ambiguity (`src/evaluation/fashioniq.py:144-146`) — P3 — none.
31. Metric denominator — VERIFIED — `r = 100*r/len(nn_result)` (`test.py:173`) = query count — NONE — none.
32. Mean identity — VARIANT (rounding) — paper `(55.31+75.16)/2 = 65.235 -> 65.24` and `(54.22+74.49)/2 -> 64.36`; unrounded macros give 65.23 / 64.35; workbench uses unrounded inputs (`src/evaluation/fashioniq.py:254`) — P3 — record the rounding convention.
33. R@10 = 55.31 — VERIFIED vs paper — arXiv PDF p.6 + HTML Table 2 + `figs/fiq.png`; `models.yaml:327`; macro of 50.32/56.32/59.28 = 55.3067 — NONE — none.
34. R@50 = 75.16 — VERIFIED vs paper — as above; macro of 72.10/74.93/78.45 = 75.16 — NONE — none.
35. mean = 65.24 — VERIFIED vs paper — as above — NONE — none.
36. Per-category paper values, all four blocks — VERIFIED — arXiv PDF p.6 (`pdftotext -layout`) and HTML; `papers.yaml` rows n20/n50/n80 — NONE — none.
37. R@1 published — UNKNOWN — source computes it (`test.py:168`) but Table 2 has no R@1 column — P3 — none.
38. Dataset layout matches source paths — VERIFIED — `fashioniq_layout.py:24-36` vs `datasets.py:31-33,139,150` — NONE — none.
39. Image extensions `.jpg` — VERIFIED — `models.yaml:323`; `datasets.py:139,150` hard-codes `.jpg`; `fashioniq_layout.py:85-86` — NONE — none.
40. Local source images are `.png`, derived tree absent — VERIFIED — `ls data/fashionIQ_dataset/images | sed 's/.*\.//' | sort -u` -> `png`; `resized_image/` absent — P3 — the `prepare_model.py` resize path (`:150-215`) is the supported route; `source_extensions` now recorded (`:259,622`) — none further.
41. README data-prep is insufficient for the code — VERIFIED — `README.md` FashionIQ tree lists `dress/ shirt/ toptee/`; `datasets.py:31` requires `resized_image`; `grep -n 'resized\|correction_dict' README.md` -> no matches — P2 — the registry's `fashioniq_layout` is the authoritative statement; note the README gap.
42. Correction dictionaries present locally — BLOCKED — `ls data/fashionIQ_dataset/captions` -> only `cap.*.json`; required at `datasets.py:98,172` — P1 — obtain author-pinned files or do not evaluate.
43. Correction dictionary provenance — BLOCKED — `AUXILIARY_ASSET_AUDIT.md:358-360` `OFFICIAL_ASSET_UNAVAILABLE`; `auxiliary_assets.yaml` has 0 `intent` entries (`grep -rn intent …` -> exit 1) — P1 — record as an unverified blocker.
44. Local split sizes/uniqueness — VERIFIED — OBSERVED dress 3817/2017, shirt 6346/2038, toptee 5373/1961, all unique — NONE — none.
45. Preprocessing transform — VERIFIED — `train_INTENT.py:79-81,87`; `data_utils.py:92-105` — NONE — none.
46. Eval-time transform deterministic — VERIFIED — `datasets.py:149-157`; `model.eval()` (`test.py:10`); `@torch.no_grad()` (`blip2_cir.py:721,771`) — NONE — none.
47. Training-only intervention machinery — VERIFIED — `colorful_spectrum_mix` (`blip2_cir.py:57`), `text_proj_cau` (`:171,523,526`) reached only via training `forward` — NONE — none.
48. Noise injection scope — VERIFIED (training only) — `shuffle()` (`datasets.py:63`) called only at `train_INTENT.py:101`; no call from `test.test` — NONE — none.
49. `evaluation_noise_pct: 0` enforced — VERIFIED — `models.yaml:339,350,361`; `workbench/backend/registry.py:84-85` raises otherwise — NONE — none.
50. Noise can leak into eval inputs — VERIFIED not to — `datasets.py:65-67` returns at ratio 0; val queries from `cap.*.val.json` (`:170`) are never shuffled — P3 — the `test_queries_*.pkl` cache (`datasets.py:37-61`) is unvalidated for freshness; record a freshness contract if a cache is ever shipped.
51. BLIP-2 base weights and hardcoded paths — BLOCKED — `/root/bert-base-uncased` in `lavis/models/blip2_models/blip2.py:32,48,55`; EVA-ViT-g from a Google URL (`lavis/models/eva_vit.py:442`); backbone config `lavis/configs/models/blip2/blip2_pretrain.yaml:10`; no `intent` entries in `auxiliary_assets.yaml` — P1 — declare these per model or state they are operator-supplied.
52. Environment declaration — VERIFIED — `models.yaml:305-310` == `README.md:112,116,119`; `cuda '12.6'` vs README's 12.1 wheels — P3 — align or mark approximate.
53. Evaluator invocation / adapter / runnable status — BLOCKED — `IntentAdapter` (`adapters/models.py:69-71`) inherits the `NotImplementedError` stub (`:24-25`); `grep -c '__main__\|argparse' test.py` -> `0`; no restore code in `train_INTENT.py`; `utils.load_checkpoint` (`utils.py:128-143`) uncalled — P1 — no executable command exists; keep `reproduction_status: NOT_RUN` and the `source_blocker` (`models.yaml:302,330`).
54. README CIRR invocation works — BLOCKED — `README.md:188-190` passes `--checkpoint_path`/`--output_file`; `grep -n argparse cirr_sub_BLIP2.py` -> no matches; the script sweeps a hardcoded `/root/autodl-tmp/BLIP/test/` (`:124-129`) — P2 — doc defect reproduced in both the GitHub and Hub READMEs.
55. Training defaults match the paper — BLOCKED — paper: 10 epochs, lr 4e-5, μ=0.2, α=0.6; `train_INTENT.py:49,54,56,57`: 20 epochs, lr 2e-5, μ=0.5, α=1.0 — P2 — record the divergence in the audit; a replay needs explicit overrides.
56. No workbench INTENT model implementation — BLOCKED — `ls src/models/` -> `iag_srme`, `taper_mag`; nothing implements `Blip2QformerCir` — P1 — no workbench-side evaluation path exists either.
57. Aggregate metric parser — BLOCKED — `metrics_extraction.py:309-341` dispatches only `limn`/`csmcir`; `intent` -> `AGGREGATE_PARSER_UNAVAILABLE` — P2 — add a parser only alongside a runnable command, not before.
58. Gate would compare categories — VERIFIED (mechanism) — `paper_gate.py:348-368` compares a category only when both sides state it, else records it unchecked; INTENT rows publish all 6 cells — NONE — none.
59. Benchmark rows present and VERIFIED — VERIFIED — `papers.yaml:1232,1289,1340,1389`, each `reference_status: VERIFIED`, independent `yaml.safe_load` confirms values and noise conditions — NONE — none.
60. `papers.yaml` rebuilt from the bundle — VERIFIED — `build_paper_registry.py:23,161,190` reads `paper_benchmarks/extracted/*.json`; `extracted/intent_aaai26.json` carries the four rows with matching values and `review_status: PENDING` — NONE — none (rows are mechanically derived, as documented).
61. Prior audit's withdrawn claim — VERIFIED (prior claim WRONG) — executed numpy: `(2017,1,1,256)x(3817,256,32) -> (2017,3817,1,32) -> squeeze (2017,3817,32) -> max(-1) (2017,3817)`; equals max-over-tokens — P2 — rewrite `workbench/docs/reviews/models/intent.md:128-131` and checklist item 34 so the refuted P0 is not left standing.
62. Registry `source_blocker` accuracy — VERIFIED — `models.yaml:302`: callable-only evaluator (true, `test.py` has no `__main__`), adapter declines (true, `adapters/models.py:24-25`), no published source for dicts/images (true), 0% row has no weights (true) — NONE — none.
63. Registry readiness category — VERIFIED — `pc_readiness.py:36-71`: "source cannot run" maps to `BLOCKED_SOURCE`, matching the audit's readiness label — NONE — none.

FINAL VERDICT: BLOCKED
