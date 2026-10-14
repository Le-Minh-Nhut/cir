# conesep

**STATIC_AUDIT throughout.** No checkpoint was downloaded, no weights were loaded, no GPU job was run, no evaluator was executed. Everything below is derived from pinned source text, committed registry rows, executed read-only inspection commands, and HTTP/image inspection. Every claim is labelled. Where a prior-round document disagrees with the code, the code is quoted.

Repo under audit: `/home/heheboiz/data/cir`, HEAD `eb36aa5ab0740e7dca7dd9c78515c3c49514b4e4`, branch `research/cir-failure-workbench`, **working tree dirty** (`git status --porcelain` -> 12 ` M` files + 3 `??`).

---

## MODEL ID

`conesep` (`method_name: ConeSep`).

- Registry entry: `workbench/registry/models.yaml:146-208` (`- model_id: conesep` at `:146`).
  ```
  $ git show HEAD:workbench/registry/models.yaml | grep -n "model_id: conesep"
  146:  - model_id: conesep
  ```
  The prior-round review cites `models.yaml:133-194` for this block; at HEAD the block is **146-208**. The 13-line offset is real and every line number in that document is off by it. (The cited base `d9431fe9632b67a329771cc06567487b591afb4b` exists and does have the block at `:133` -- `git show d9431fe`:workbench/registry/models.yaml | grep -n "model_id: conesep"` -> `133`. So the review describes an older revision, not HEAD. The content is otherwise identical; I re-derived every line below from HEAD.)
- `source_dir: ConeSep` (`:166`), `source_available: true` (`:165`), `paper_url: https://arxiv.org/abs/2604.20358` (`:161`).

## PAPER TITLE

`ConeSep: Cone-based Robust Noise-Unlearning Compositional Network for Composed Image Retrieval` (`models.yaml:160`).

- `read https://arxiv.org/abs/2604.20358` -> HTTP 200, title identical, authors Zixu Li, Yupeng Hu, Zhiwei Chen, Mingyu Zhang, Zhiheng Fu, Liqiang Nie, published 2026-04-22. Same author list as the pinned README (`workbench/third_party/ConeSep/README.md:8-13`).
- Pinned README title: `third_party/ConeSep/README.md:5` `(CVPR 2026) ConeSep: Cone-based Robust Noise-Unlearning ...`.
- **P1 (new): wrong upstream repository URL.** `upstream_repo_url: https://github.com/iLearn-Lab/CVPR26-ConeSep` (`:162`). The paper's own HTML prints `<https://github.com/Lee-zixu/ConeSep/>` (extracted from `https://arxiv.org/html/2604.20358v1`), the author-hosted Hub model card links `https://github.com/Lee-zixu/ConeSep`, and the pinned README issues badge (`README.md:44`) points to the same. The pinned *checkout's* origin is `iLearn-Lab/CVPR26-ConeSep.git` (so the registry string does match the checkout), but a reader following the field lands on a different repo than the paper names. Labelling problem, not a data-integrity problem.

## PINNED SOURCE COMMIT

| Claim | Evidence | Verdict |
|---|---|---|
| Registry pin `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` (`:164`) equals checkout HEAD | `git -C workbench/third_party/ConeSep rev-parse HEAD` -> `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` | MATCH |
| Remote equals `upstream_repo_url` (`:162`) | `git config --get remote.origin.url` -> `https://github.com/iLearn-Lab/CVPR26-ConeSep.git` | MATCH (as a checkout fact) |
| `upstream_default_branch: main` (`:163`) | `git branch -a --contains HEAD` -> `* main`, `remotes/origin/HEAD -> origin/main` | MATCH |
| No modified tracked file | `git -C ... status --porcelain \| grep -vc __pycache__` -> `0` (77 entries, all `?? __pycache__/`) | CLEAN |
| History depth | `rev-list --count HEAD` -> `1`; `git log -1` -> `bd4bf629... 2026-05-25 17:59:18 +0800 Update README.md` | single-commit checkout; no upstream history to diff against |
| Sandbox broken | `git cat-file -t d9431fe...` -> `commit` (prior-round review's base commit is present) | the repo *is* a git repo; the prior round's "not a git repository" observation no longer reproduces |

**Verdict: VERIFIED (pin).**

## CHECKPOINT IDENTITY

STATIC_AUDIT (registry + HTTP HEAD + Hub listing). No payload inspected.

Registry declares three variants (`models.yaml:178-208`):

| checkpoint_id | filename (`:line`) | training noise | eval noise | size | sha256 | status | mapping |
|---|---|---|---|---|---|---|---|
| `fiq_n02` | `ConeSep-FIQ-N0.2.pt` (`:180`) | 20 (`:183`) | 0 (`:184`) | `null` (`:185`) | `null` (`:186`) | `NO_CLEAN_CHECKPOINT` (`:187`) | `VERIFIED_METADATA` (`:188`) |
| `fiq_n05` | `ConeSep-FIQ-N0.5.pt` (`:190`) | 50 (`:193`) | 0 (`:194`) | `null` (`:195`) | `null` (`:196`) | `NO_CLEAN_CHECKPOINT` (`:197`) | `VERIFIED_METADATA` (`:198`) |
| `fiq_n08` | `ConeSep-FIQ_N0.8.pt` (`:200`) | 80 (`:203`) | 0 (`:204`) | `null` (`:205`) | `null` (`:206`) | `NO_CLEAN_CHECKPOINT` (`:207`) | `VERIFIED_METADATA` (`:208`) |

Independent HTTP check (executed, `curl -sI -L`):

```
== ConeSep-FIQ-N0.2.pt   x-linked-size: 2721428966  x-linked-etag: "5a864ee5a78c1d249320c4f997cc7d14b529b329b2efba5577c82dbe019ed9d2"  -> HTTP/2 200, content-length 2721428966
== ConeSep-FIQ-N0.5.pt   x-linked-size: 2721428966  x-linked-etag: "1953ec2f9269a5cb24d61bdf89d05ed8430fb3bdc6dd18cef08fe9e3dfbe1e26"  -> HTTP/2 200, content-length 2721428966
== ConeSep-FIQ_N0.8.pt   x-linked-size: 2721428966  x-linked-etag: "7c2bd0fbaea423a1a64809956b3fda72e191a6f7e5ed6ecfa708f1e3a8730eae"  -> HTTP/2 200, content-length 2721428966
```

- The three registry URLs (`:181`, `:191`, `:201`) are the canonical resolve URLs for exactly those files. The inconsistent `N0.8` underscore spelling is faithfully copied.
- Hub tree `https://huggingface.co/iLearn-Lab/CVPR26-ConeSep/tree/main/fiq` lists **exactly** three files, one upload commit `36139d8`, folder size 8.16 GB, all detected as `pickle` with `lavis.models.blip2_models.ConeSep.ConeSep` in the import list -> whole-model pickles, consistent with `train.py:251` `torch.save(model, ...)`.
- The Hub **model card** (`/raw/main/README.md`) documents the naming as `fashioniq/ConeSep-FIQ_N0.2.pt` and shows the tree with a `cirr/` folder. The Hub's actual tree has **no `cirr/` directory** and the files live in **`fiq/`**. The card is copy-pasted and self-inconsistent with its own repo; the paper's Table 2 does report four CIRR `ConeSep (Ours)` rows (extracted from the HTML). So the *published file set* is 3 FashionIQ blobs with no CIRR weights, and the registry's `notes` sentence (`:177`) is correct about that.

**Gaps (P2):** `expected_size_bytes` / `expected_sha256` are `null` for all three variants even though the Hub publishes `2721428966` and a per-file SHA-256. `workbench/scripts/download_checkpoints.py:110-114` prints `LOCAL SHA256 COMPUTED - official SHA256 unavailable` and sets `hash_valid = True` when `expected_sha256 is None`; `:117` accepts `checkpoint_mapping_status == VERIFIED_METADATA` + a `download_url` as `provenance_valid`. Any 2.72 GB blob from that URL therefore passes the gate with no content binding.

**Clean-checkpoint question:** the author Hub publishes no 0 % FashionIQ weight (`fiq/` = 20/50/80 only), and the pinned README has no checkpoint section (only `best_model.pt` output naming, `README.md:238`, and the CIRR submission script, `:245`). `strict_clean_checkpoint_available: false` (`:176`) is **correct**.

## EVALUATION PROTOCOL

`native_protocol: fashioniq_full_gallery_ref_excluded` (`:170`), `supported_protocols: [fashioniq_full_gallery_ref_excluded]` (`:171`), `literature_split_label: original` (`:169`); protocol table at `models.yaml:12-15`.

Checked against the pinned source -- all five sub-claims hold:

1. **Evaluator.** `third_party/ConeSep/test.py:9` `def test(params, model, testset, category, txt_processors)`; `dress/shirt/toptee/shoes` at `:12-19`. `grep -n "__main__\|argparse\|ArgumentParser" test.py` -> **no match** (exit 1). Only callsite: `train.py:216` `t = test.test(args, model, trainset, category, txt_processors)` inside `train_and_evaluate()`.
2. **Gallery.** `datasets.py:28` `self.split = 'original-split'`; `__init__` signature `(self, path, transform=None, noise_ratio=0)` (`:26`) has no split switch. `datasets.py:201-206` `elif self.split == 'original-split': for id, image_name in enumerate(images): ...` -> full ordered val split. The `val-split` branch (`:189-200`) is dead for this model.
3. **Reference exclusion.** `test.py:64-68` builds `test_targets_id` then `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` -- **mask before sort**, at `:68`, ahead of `argsort` at `:71`/`:75`. Registry claim confirmed; `docs/FASHIONIQ_PROTOCOL_PARITY.md:71` agrees ("masked before ranking").
4. **Recall maths.** `test.py:71,75` `np.argsort(-sims[i, :])[:50]`; `:76` `for k in [1, 10, 50]`; `:79` `if test_targets_id.index(test_queries[i]['target_img_id']) in nns[:k]: r += 1`; `:81` `r = 100 * r / len(nn_result)`. Hit fraction x 100, denominator = query count.
   **I now believe the prior-round shape argument is WRONG, and say so.** Executed in `/tmp`:
   ```
   matmul((5,1,1,256),(7,256,32)) -> (5, 7, 1, 32)
   .squeeze()                     -> (5, 7, 32)      # the 1-dim is squeezed
   .max(-1)                       -> (5, 7)         # 2-D, exactly what test.py indexes
   ```
   `extract_retrieval_compose` returns `(N,1,1,d)` (`ConeSep.py:492`) and `extract_retrieval_target` returns `(M,d,32)` (`ConeSep.py:515`). The prior round asserted `.squeeze()` leaves `(N,M,1,32)` and hence a "4-D `sims`"; it does not. The registry's own `source_blocker` text (`:147`) contains the same withdrawal ("An earlier note in this registry claimed a tensor-shape failure in the similarity computation; that claim was WRONG and is withdrawn"). The stale comment `#all_queries.dot(all_imgs.T)` at `test.py:59` is real (P3).
5. **Tie-breaking.** Not specified by the source or the workbench; numpy's default quicksort on `-sims`. UNKNOWN-by-design, measure-zero for `F.normalize`d float features (`ConeSep.py:488-490`, `:514`).

## DATASET CONTRACT

`fashioniq_layout: fashioniq_ilearn_resized` (`:167`), `fashioniq_image_extensions: [".jpg"]` (`:168`), `preparation_contract: manual_ilearn_resized` (`:159`).

Source paths (trailing separator required -- `__init__` concatenates): `datasets.py:31-33` `self.image_dir = self.path + 'resized_image'`, `split_dir`, `caption_dir`.

| Input | Source callsite | Layout checker |
|---|---|---|
| `image_splits/split.{cat}.val.json` | `datasets.py:166` | `fashioniq_layout.py:33` |
| `captions/cap.{cat}.val.json` | `datasets.py:168` | `fashioniq_layout.py:31` |
| `captions/correction_dict_{cat}.json` | `datasets.py:170` | `fashioniq_layout.py:32` |
| `captions/cap.{cat}.train.json` | `datasets.py:94` (cache miss only) | in `ilearn_resized_training_paths` (`fashioniq_layout.py:37-41`), **absent** from `ilearn_resized_paths` |
| `resized_image/{cat}/{id}.jpg` | `datasets.py:137,147` | `fashioniq_layout.py:29-30` + extension check `:92-...`; `_LAYOUT_IMAGE_DIRECTORY["fashioniq_ilearn_resized"] = ("resized_image", True)` (`:85`) |
| `fashion_iq_data.json`, `test_{queries,targets}_{cat}.pkl` | `datasets.py:35,38,44-58` | **not checked** |

**Gap 1 (P2) -- unvalidated cache.** `datasets.py:35` gates all preparation on `if not os.path.exists(os.path.join(self.path, 'fashion_iq_data.json'))`; the `else` at `:51-59` loads the JSON + six PKLs and never re-reads the split/caption/correction files. Layout preflight passes against edited JSONs while the evaluator silently uses stale pickles.

**Gap 2 (P2) -- the correction dictionaries are a content-transforming, unprovenanced input.** `datasets.py:96` (train) and `:170` (val) open `captions/correction_dict_{cat}.json`. `grep -c conesep workbench/registry/auxiliary_assets.yaml` -> `0`; `preparation_contracts.yaml` -> `0`. `workbench/docs/AUXILIARY_ASSET_AUDIT.md:352-354` records all three as `OFFICIAL_ASSET_UNAVAILABLE`. New executed observation: the **same** dictionary files are vendored in two other pinned checkouts and are byte-identical:
```
LIMN/correct_files/correction_dict_dress.json  sha256 9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda
TGCIR/correction_dict/correction_dict_dress.json sha256 9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda   (equal, 363 keys)
shirt   ca29e29e...  (=)  toptee 3dfbf15f...  (=)
```
So an identity **is** discoverable -- it is just not in the registry. (Whether they are the ones behind Table 1 remains UNKNOWN.)

**Local readiness (executed, read-only):**
```
$ PYTHONPATH=. python3 -c "...preflight.model_requirements(conesep...)['blocks']"
n_blocks 8
 - FashionIQ fashioniq_ilearn_resized requirement missing: /home/heheboiz/data/cir/data/fashionIQ_dataset/resized_image
 - ... /resized_image/dress ; /shirt ; /toptee
 - ... /captions/correction_dict_dress.json ; _shirt.json ; _toptee.json
 - FashionIQ image extension requirement: image directory missing: .../resized_image
$ ls data/fashionIQ_dataset
captions  image_splits  images  qwen_captions      # not the iLearn resized layout
```
`docs/READINESS_MATRIX.md:27` records the same blocker (generated at an earlier revision: `pc_readiness.repository_section()` currently reports `dirty_files: 15`).

## PREPROCESSING CONTRACT

Two text stages and one image stage; none of them is registry-pinned.

**Stage 1 -- dictionary-corrected caption composition** (`datasets.py:108-117`):
```python
def correct_text(self, text, correction_dict):
    trans=str.maketrans({key: ' ' for key in string.punctuation})
    tokens = str(text).lower().translate(trans).strip().split()
    text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])
def concat_text(self, captions, correction_dict):
    text = "{} and {}".format(self.correct_text(captions[0], ...), self.correct_text(captions[1], ...))
```
Applied at `datasets.py:175`. lowercase -> all ASCII punctuation -> spaces -> tokenize -> exact token substitution -> `cap1 + " and " + cap2`.

**Stage 2 -- LAVIS eval processor**, `test.py:35` `mods = [txt_processors["eval"](caption) for caption in mods]` -> `BlipCaptionProcessor` (`lavis/processors/blip_processors.py:28-47`, `@registry.register_processor("blip_caption")`), `pre_caption` (`:49-66`): lowercase + replace **only** `[.!\"()*#:;~]`, collapse whitespace, `rstrip("\n")`, truncate to `max_words=50` (`:30`, `:45`). Narrower than stage 1, so stage 2 is not a no-op on the composed string.

**Tokenizer:** `ConeSep.py:133` `self.tokenizer = self.init_tokenizer()` -> `blip2.py:31-34`, hardcoding `BertTokenizer.from_pretrained("/root/bert-base-uncased", ...)`; also `blip2.py:48` (Q-Former config) and `:55` (weights). Tokenization at `ConeSep.py:219-221` `padding="max_length", truncation=True, max_length=self.max_txt_len`, default `max_txt_len=32` (`ConeSep.py:128`). `auxiliary_assets.yaml:249-260` registers a `bert-base-uncased tokenizer` asset but **only for `model_id: csmcir`** -- nothing for `conesep`. P2.

**Images:** `train.py:91-99` hardcodes `transform="targetpad"`, `input_dim=224`, `target_ratio=1.25` -> `targetpad_transform(1.25,224)`, passed to both train and test datasets (`train.py:108-115`); val images go through the same transform via `get_test_img` (`datasets.py:146-154`). `data_utils.py:92-106` = `TargetPad(1.25,224) -> Resize(224, BICUBIC) -> CenterCrop(224) -> _convert_image_to_rgb -> ToTensor -> Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))`; `TargetPad.__call__` (`data_utils.py:64-73`) pads only when `actual_ratio >= target_ratio`.

**Registry pins none of this.** `preparation_contract: manual_ilearn_resized` is validated only as an enum (`workbench/backend/registry.py:73`). Staged `resized_image/{cat}/*.jpg` have no pinned producer anywhere in the tree. P2.

## AUGMENTATION CONTRACT

No image augmentation. The only stochastic element is training-time noise: `FashionIQ.shuffle()` (`datasets.py:61-90`) -- `random.sample(range(num_samples), int(self.noise_ratio * num_samples))`, then thirds for candidate / captions / target (`:69-73`), `random.shuffle` per partition (`:79-81`), unseeded. Called only from `load_dataset()` (`train.py:109` fashioniq, `:117` cirr). `grep -rn "shuffle" third_party/ConeSep/test.py` -> no match. `test.py:10` `model.eval()`, body under `torch.no_grad()` (`:21`).

Therefore the registry's uniform `evaluation_noise_pct: 0` (`:184,194,204`) is correct as an evaluation-time statement, and `checkpoint_training_noise_pct` (20/50/80) is the right place for the provenance fact. The partition is thirds over a `noise_ratio` subset (effective corruption is a union, not `noise_ratio` fully-corrupted triplets), and the RNG is unseeded, so bit-reproducing a published checkpoint from scratch is not deterministic from the pinned code (P3; irrelevant to evaluating a published checkpoint).

## GALLERY CONTRACT

- Candidate universe = every image of `split.{cat}.val.json`, in file order (`datasets.py:201-206`).
- Reference-only exclusion per query by the `-10e10` mask (`test.py:68`), applied before the sort (`test.py:71`).
- Workbench expression: `src/evaluation/fashioniq.py:159-169` `FASHIONIQ_PROTOCOLS["fashioniq_full_gallery_ref_excluded"]` = `gallery="full_split"`, `reference_excluded=True`; `:204-216` `build_fashioniq_gallery` routes `full_split` to `build_original_gallery`; `:35-78` `recall_at_k_reference_excluded` **removes** the reference column (`:72` `ranking = ranking[ranking != reference_column]`) rather than masking -- the docstring at `:45-53` states the equivalence condition explicitly. `:238-256` `macro_average_fashioniq` = equal-weight mean of `recall_at_10`/`recall_at_50` plus `mean_recall`.
- **Reachability (re-verified this round):** `grep -rn "evaluate_fashioniq_recall|build_fashioniq_gallery|macro_average_fashioniq|evaluate_fashioniq_category" --include=*.py src/ workbench/ tests/` -> production hits only inside `src/evaluation/fashioniq.py` itself; all others are `tests/unit/test_fashioniq_protocols.py`, `tests/unit/test_fashioniq_dispatch.py`, `tests/unit/test_encoder_mode_isolation.py`. `grep -rn conesep workbench/replay/` -> empty (`workbench/replay/` = `common.py`, `limn.py`, `tgcir.py`). No registered `conesep` run can reach the protocol machinery. P1.

## METRIC CONTRACT

**Source.** `test.py:76-82` computes `{cat}_r1`, `{cat}_r10`, `{cat}_r50`. There is **no macro aggregation in the pinned source**; `train.py:216-218` selects checkpoints on `t[1][1]` = `dress_r10`, i.e. the sum of category R@10:
```python
t = test.test(args, model, trainset, category, txt_processors)
current_score = current_score + t[1][1]
```
A run of the pinned evaluator emits category numbers; it never emits the paper's average row. (Matches `docs/UPSTREAM_DATA_PREPARATION_AUDIT.md` finding 4: "No iLearn method publishes a category macro".)

**Paper.** Table 1 grid parse (`https://arxiv.org/html/2604.20358v1`, executed) returns exactly four `ConeSep (Ours)` FashionIQ rows in the `0% / 20% / 50% / 80%` blocks, header `Noise Methods Dress Shirt Toptee Average R@10 R@50 R@10 R@50 R@10 R@50 R@10 R@50 AVG.`, and the metric sentence: "For FashionIQ, we follow its standard metrics [90], reporting R@10 and R@50 for each of the three categories (Dresses, Shirts, Tops&Tees)".

**Workbench.** `src/evaluation/fashioniq.py:238-256` macro; `metrics_extraction.py:152` `computed_mean = (computed_r10_macro + computed_r50_macro) / 2.0`; `paper_gate.py:27` `TOLERANCE_PCT = 5.0` (**relative**); `metrics_extraction._check_parity` now routes through `paper_gate.compare_metric` (`:53-56`), so the two rules agree.

## SOURCE-TO-WORKBENCH PARITY

`docs/SOURCE_TO_WORKBENCH_PARITY_MATRIX.md:28`: protocol `EQUIVALENT_WITH_EVIDENCE`, preparation `EXTERNAL_ASSET_REQUIRED (same LAVIS/BERT/dictionary set)`, caption `EQUIVALENT_WITH_EVIDENCE`, image `EQUIVALENT_WITH_EVIDENCE`, evaluator command `NOT IMPLEMENTED`, overall `INCOMPLETE_IMPLEMENTATION`. The matrix header (`:11-16`) says "No model is EXACT_MATCH" and defines `EQUIVALENT_WITH_EVIDENCE` as behaviour agreement at a boundary, never score reproduction.

| Component | Verdict | Evidence |
|---|---|---|
| Protocol semantics | SUPPORTED | `datasets.py:28,201-206`; `test.py:64-82`; `src/evaluation/fashioniq.py:159-169,204-216` still test-only |
| Caption stage | SUPPORTED (partial) | source stages confirmed (`datasets.py:108-117`, `test.py:35`, `blip_processors.py:49-66`); cross-method *string* equality for a real caption not demonstrated, and dictionary provenance is external |
| Image stage | SUPPORTED (partial) | `targetpad_transform(1.25,224)` confirmed (`train.py:91-99`, `data_utils.py:92-106`); the staged JPEGs have no pinned recipe |
| Preparation contract | EXTERNAL_ASSET_REQUIRED -- confirmed | `grep -c conesep` -> 0 in both `auxiliary_assets.yaml` and `preparation_contracts.yaml`; `/root/bert-base-uncased`, LAVIS `blip2_pretrained.pth` (`lavis/configs/models/blip2/blip2_pretrain.yaml:10`), dictionaries, cache PKLs all unregistered |
| Evaluator command | NOT IMPLEMENTED -- confirmed by execution | `workbench/backend/adapters/models.py:59-61`; `ADAPTERS['conesep'].command('s','c',None,None)` -> `NotImplementedError: upstream evaluation command requires audit before execution` |
| Overall | INCOMPLETE_IMPLEMENTATION -- confirmed | no `test.py` CLI, no checkpoint loader, unaudited command |

## RUNTIME STATUS

- `reproduction_status: NOT_RUN` (`:175`). Consistent with everything observed.
- `WORKBENCH_CONESEP_PYTHON` -> `UNSET`.
- `workbench/artifacts/checkpoints/` -> `.gitkeep` only (zero checkpoints on disk).
- `select_checkpoints` (executed):
  ```
  strict_clean                 BLOCKED  no checkpoint is recorded as clean-trained under policy strict_clean: fiq_n02=20%, fiq_n05=50%, fiq_n08=80%. ...
  include_flagged_conditions   BLOCKED  more than one source-backed variant is registered and none is declared a bundle: fiq_n02, fiq_n05, fiq_n08. ...
  ```
- `preflight_blockers` (executed) -> 10 blockers, classified `BLOCKED_SOURCE`:
  ```
  0  source cannot run: no runnable evaluation path is registered for this model: ... (models.yaml:147)
  1-7 FashionIQ fashioniq_ilearn_resized requirement missing: ... (resized_image{,/dress,/shirt,/toptee}, correction_dict_{dress,shirt,toptee}.json)
  8  FashionIQ image extension requirement: image directory missing: .../resized_image
  9  command not runnable: COMMAND_UNAUDITED
  ```
  The declared `source_blocker` is surfaced first because `preflight.py:198-204` inserts it at position 0.
- `pc_readiness.model_section` (executed) -> `state: BLOCKED_SOURCE`; `auxiliary_assets: []`; per-variant references all `reference_id: conesep_fiq_n2x/n5x/n8x`, `reference_status: VERIFIED`, `reason: "exact condition match"` (see CF-1).
- `evaluate_models.candidates` (executed) enumerates `fiq_n02 (20)`, `fiq_n05 (50)`, `fiq_n08 (80)`; `guarded_plan` returns `None, ["SKIPPED: adapter command not audited"]` (`evaluate_models.py:290-291`).
- Import probe of the pinned model class: `lavis.models.blip2_models.ConeSep` pulls `lavis`'s dependency set (omegaconf et al.) which is absent in this shell -- a dependency-absence fact, not a source defect. (Prior round reported `ModuleNotFoundError: No module named 'omegaconf'`; I did not re-run it, and I do not claim a runtime result either way.)
- No GPU evaluation was attempted; none is possible here.

## PAPER-METRIC STATUS

**(a) Registry rows vs the paper and vs the author image.** Four committed rows (`workbench/registry/paper_benchmarks/papers.yaml:588-779`). Re-derived arithmetic (executed):

| reference_id | reported r10/r50/mean | category cells | avg-R@10 recomputed | avg-R@50 recomputed | (r10+r50)/2 |
|---|---|---|---|---|---|
| `conesep_fiq_n00` | 55.58 / 75.88 / 65.73 | 50.96/73.02, 56.98/75.22, 58.80/79.40 | 55.580000 | 75.880000 | 65.73 |
| `conesep_fiq_n20` | 54.93 / 75.01 / 64.97 | 50.23/72.19, 56.28/74.45, 58.29/78.38 | 54.933333 | 75.006667 | 64.97 |
| `conesep_fiq_n50` | 53.39 / 74.15 / 63.77 | 48.39/71.34, 54.56/73.25, 57.22/77.87 | 53.390000 | 74.153333 | 63.77 |
| `conesep_fiq_n80` | 49.25 / 70.30 / 59.77 | 43.33/66.01, 51.23/70.35, 53.19/74.55 | 49.250000 | 70.303333 | 59.77 |

All four agree digit-for-digit with the paper HTML grid and with `third_party/ConeSep/assets/results-fiq.png` (read as an image this round: the `0%` and `20/50/80%` `ConeSep (Ours)` rows match). `derived_metrics: []` on all four is **correct** -- 65.73 is printed as AVG. and is not workbench-derived.

**(b) The row-to-checkpoint ids do not resolve (P1, new).** `conesep_fiq_n20/50/80` carry `checkpoint_id: fiq_n20` / `fiq_n50` / `fiq_n80` (`papers.yaml:643,691,739`), but `models.yaml` registers those conditions as `fiq_n02` / `fiq_n05` / `fiq_n08` (`:179,189,199`). Executed cross-check of **every** paper row against `models.yaml` variants+bundles:
```
dangling checkpoint_id in paper rows (not a models.yaml variant nor bundle):
  conesep_fiq_n20 conesep fiq_n20 UNVERIFIED VERIFIED
  conesep_fiq_n50 conesep fiq_n50 UNVERIFIED VERIFIED
  conesep_fiq_n80 conesep fiq_n80 UNVERIFIED VERIFIED
```
`conesep` is the **only** model in the registry with this defect (`airknow_fiq_50pct` -> `fiq_n05`, `habit_aaai26_table1_fiq_n50` -> `fiq_n05` all resolve). Consequence, executed through the production helper:
```
$ ... _benchmark_gate_verdict(Namespace(model_id='conesep', protocol=..., checkpoint_id='fiq_n02', checkpoint_training_noise_pct=20, ...), E, 0)
20 fiq_n02 -> BLOCKED_CHECKPOINT_MISMATCH | blockers ["checkpoint_id mismatch: reference='fiq_n20' run='fiq_n02'"]
50 fiq_n05 -> BLOCKED_CHECKPOINT_MISMATCH | blockers ["checkpoint_id mismatch: reference='fiq_n50' run='fiq_n05'"]
80 fiq_n08 -> BLOCKED_CHECKPOINT_MISMATCH | blockers ["checkpoint_id mismatch: reference='fiq_n80' run='fiq_n08'"]
```
`benchmark_references.select_reference` nevertheless returns `"exact condition match"` and `paper_gate` records `reference_status: VERIFIED`; only `paper_gate.COMPARABILITY_FIELDS`'s `checkpoint_id` comparison catches it (`paper_gate.py:72-79`, `:245-256`). Correct `checkpoint_training_noise_pct`, correct protocol, correct dataset/split -- and the run would still be refused as a checkpoint mismatch. P1 (blocks every registered variant from reaching a metric comparison, for an id-naming reason rather than a scientific one).

**(c) The old absolute-parity field remains a contradictory display annotation (P1, new).** `metrics_extraction.py:53-56` now compares through `paper_gate.compare_metric` (relative 5 %), but the *reference it is handed* is still the registry's **0 %-noise headline** pair, built unconditionally from the model row:
```python
paper_metrics = {key.removeprefix("reported_"): model.get(key) for key in ("reported_r10","reported_r50","reported_mean")}   # evaluate_models.py:309
```
Executed `_check_parity(observed, {'r10':55.58,'r50':75.88,'mean':65.73})`:
```
54.93 -> PARITY_VERIFIED      # the 20%-noise row
53.39 -> PARITY_VERIFIED      # the 50%-noise row
49.25 -> PARITY_MISMATCH      # the 80%-noise row
```
So a 20 %- or 50 %-noise-trained checkpoint would report `parity_status: PARITY_VERIFIED` against the *clean* row while `benchmark_gate` says `BLOCKED_CHECKPOINT_MISMATCH`. Two contradictory statuses remain in one report (`evaluate_models.py:1414` `benchmark_gate`, `:1420` `parity_status`), and `parity_status` is still surfaced by `/api/benchmark-runs` (`main.py:115`) and typed in `frontend/gate.ts:8`. P1.

## CRITICAL FINDINGS

**CF-1 (P1) -- no runnable evaluation path; three independent blockers.**
1. No standalone evaluator. `test.py` has no `__main__`/`ArgumentParser` (grep, exit 1); `test()` is reachable only from `train.py:216` inside `train_and_evaluate()`, which builds a **fresh** model through `create_model_and_optimizer()` (`train.py:133-140`) and never loads a checkpoint.
2. No checkpoint loader. `train.py:76-78` declares `--restore_file`; nothing reads it (`grep -rn "restore_file" . --include=*.py` -> the `argparse` line only). `utils.py:128-141 load_checkpoint()` requires `checkpoint['state_dict']` and is called from nowhere in `train.py`/`test.py`. The only `torch.load` of a model in the tree is `cirr_sub.py:124`, CIRR-submission-only. Since the Hub files are whole-model pickles (import list includes `ConeSep.ConeSep`), not `state_dict` dicts, `utils.load_checkpoint` would not load them anyway.
3. Adapter declines, and preflight agrees. `workbench/backend/adapters/models.py:59-61`; executed `NotImplementedError`; `preflight.py:184-195` marks `command not runnable: COMMAND_UNAUDITED` because `command is OfficialScriptAdapter.command`. `workbench/patches/` holds only CSMCIR artefacts.

**CF-2 (P1) -- the registry's headline row is the 0 %-noise row, and no 0 % checkpoint exists.** See CHECKPOINT IDENTITY / PAPER-METRIC STATUS. `strict_clean_checkpoint_available: false` (`:176`) is correct, and the gate's `checkpoint_training_noise_pct` field would refuse the cross-condition comparison -- but the row itself carries no "0 % training noise" label, and every variant says `evaluation_noise_pct: 0`. A reader can pair those and conclude the noise-trained weights target 65.73; their real targets are 64.97 / 63.77 / 59.77.

**CF-3 (P2) -- the registered paper rows' `checkpoint_id` values are dangling.** Detailed in PAPER-METRIC STATUS (b). Currently masked into `BLOCKED_CHECKPOINT_MISMATCH`; if anyone "fixes" it by suppressing the `checkpoint_id` comparability field, the gate would start passing runs on condition alone while the registry rows still name ids that do not exist.

**CF-4 (P2) -- the workbench's FashionIQ protocol machinery is unreachable for `conesep`.** `src/evaluation/fashioniq.py` implements full-gallery + reference-excluded + macro, and is exercised only by `tests/unit/*`. No production adapter, replay module, or pipeline stage calls it; `workbench/replay/` has no `conesep`.

**CF-5 (P2) -- dataset contract has an unchecked cache and unprovenanced dictionaries.** `datasets.py:35-59` (staleness) and `datasets.py:96,170` + zero registry rows. Both mutate captions/gallery without failing any preflight check. The dictionaries exist, byte-identical, in `LIMN/correct_files/` and `TGCIR/correction_dict/` -- an identity the registry could record today.

## MINOR FINDINGS

- **MF-1 (P2) -- no size/sha binding on the three checkpoints.** `:185-186,195-196,205-206` are `null`; the Hub publishes `2721428966` and a SHA-256 per file (recorded above); `download_checkpoints.py:110-117` treats an absent expected hash as pass.
- **MF-2 (P2) -- hardcoded `/root/bert-base-uncased`** at `blip2.py:32,48,55`; the BERT asset row exists but only for `csmcir` (`auxiliary_assets.yaml:249-260`).
- **MF-3 (P2) -- no preprocessing or auxiliary assets registered for `conesep`.** `grep -c conesep` -> 0 in both registries. `auxiliary_assets_for_model` reads only `auxiliary_assets.yaml`, so `workbench/docs/AUXILIARY_ASSET_AUDIT.md:128,352-354` (the richer per-asset audit, which *does* name the dictionaries and LAVIS weights) never reaches preflight.
- **MF-4 (P2) -- `upstream_repo_url` names a repo the paper does not.** See PAPER TITLE. Registry `:162` = `iLearn-Lab/CVPR26-ConeSep`; paper + Hub card + pinned README badge = `Lee-zixu/ConeSep`.
- **MF-5 (P2) -- `docs/PC_RELEASE_READINESS.md:39`, `docs/reviews/FINAL_REPRODUCIBILITY_AUDIT.md:25`, `docs/READINESS_MATRIX.md:27`** classify `conesep` as `BLOCKED_SOURCE`/`BLOCKED`. All three placements of `airknow, conesep` on one line are correct; `AIRKNOW`'s own `upstream_repo_url` audit is a separate model's problem. No contradiction found.
- **MF-6 (P3) -- dead `val-split` branch** (`datasets.py:189-200`), unreachable because `self.split` is hardcoded at `:28` and no constructor argument overrides it.
- **MF-7 (P3) -- stale comment** `test.py:59` `#all_queries.dot(all_imgs.T)`.
- **MF-8 (P3) -- prior-round line numbers are stale.** `workbench/docs/reviews/models/conesep.md` cites `models.yaml:133-194` and `d9431fe...`; the block is at `:146-208` at HEAD. Findings survive, citations do not.
- **MF-9 (P3) -- single-commit checkout** (`rev-list --count HEAD` -> 1). The pin is faithful to the registry string; whether it is current upstream `main` cannot be established without a fetch (forbidden here). UNKNOWN, not PASS.
- **MF-10 (P3) -- trailing-slash footgun.** `datasets.py:31-33` concatenates `self.path + 'resized_image'`; a root without a trailing separator silently reads a sibling path. `fashioniq_layout.py:25-34` uses `Path` joins and cannot detect it.

## UNRESOLVED EVIDENCE

1. **Checkpoint payload identity.** No weight file downloaded or opened. 20/50/80 are filename + Hub-card + registry assertions, not verified properties of the bytes. **UNKNOWN.**
2. **Checkpoint loadability.** Whether `torch.load("ConeSep-FIQ-N*.pt")` yields an object whose architecture matches `lavis/models/blip2_models/ConeSep.py:ConeSep` is **UNKNOWN**. The Hub import lists indicate whole-model pickles requiring an importable `lavis` at unpickle time.
3. **Whether mask-before-sort and column-removal produce identical per-query rankings on real ConeSep scores.** Argued (`src/evaluation/fashioniq.py:45-53`), not demonstrated on real outputs. **UNKNOWN.**
4. **Whether the pin equals current upstream `main`.** No fetch. **UNKNOWN.**
5. **Whether the LIMN/TGCIR correction dictionaries are the ones behind Table 1.** Same bytes exist in two pinned trees; the link to the paper run is unprovenanced. **UNKNOWN.**
6. **Whether the `resized_image` staging recipe matches what produced the published numbers.** No producer script anywhere. **UNKNOWN.**
7. **Whether a `cirr/` checkpoint set exists anywhere** (Hub card advertises it; the Hub tree has no such folder; the paper prints four CIRR ConeSep rows). **UNKNOWN.**
8. **The canonical checklist question list.** Not present in `/tmp/audit-final/context.md` nor anywhere under `/home/heheboiz/data/cir`. The items below are mine, covering every artefact this audit touched.

## FINAL VERDICT

**BLOCKED.** Every static claim I could check about `conesep` is correct -- the pin matches the checkout, `fashioniq_full_gallery_ref_excluded` is exactly what `datasets.py:28,201-206` + `test.py:64-82` implement, `strict_clean_checkpoint_available: false` is right because only 20/50/80 % weights are published, and all four committed rows reproduce Table 1 digit-for-digit and are internally consistent with `mean=(r10+r50)/2`. But the model cannot be reproduced: the pinned tree has no standalone evaluator and no checkpoint loader, the workbench adapter raises `NotImplementedError`, the registered dataset layout and correction dictionaries are absent locally, the registered row is the 0 %-noise row with no 0 % weight, and the two new defects (`checkpoint_id` namespace mismatch, `parity_status` judged against the wrong row) mean even a hypothetical run would be refused by the gate for the wrong reason while a stale field printed `PARITY_VERIFIED`.

---

## EXECUTION TRACE

```
paper/README/config -> source dataset -> preprocessing -> official evaluator -> workbench invocation
  -> checkpoint -> output -> metric comparison
```

**1. paper/README/config -> downstream is not pinned by config.**
- Paper: `https://arxiv.org/abs/2604.20358` (HTTP 200, title/authors match `README.md:8-13`); Table 1 grid parsed from `https://arxiv.org/html/2604.20358v1`; synthetic statement and metric sentence extracted.
- README: `third_party/ConeSep/README.md:5` (title), `:78` (0 % = fully-supervised), `:83-85` (Table 1 + `assets/results-fiq.png`), `:208` (`0.0/0.2/0.5/0.8`), `:213-230` (train commands), `:238` (`best_model.pt`/`metrics_best.json`), `:245` (CIRR submission), `:269` (`cirr_test_submission.py`).
- Config: `lavis/configs/models/blip2/blip2_pretrain.yaml:10` -> `blip2_pretrained.pth`; `image_size: 224`, `num_query_token: 32`.
- => **paper OK, README OK, config OK; registry pin of config/backbone weights MISSING** (`grep -c conesep auxiliary_assets.yaml` -> 0).

**2. source dataset -> present, wrong layout.**
- Required: `resized_image/{dress,shirt,toptee}/*.jpg`, `image_splits/split.{cat}.val.json`, `captions/cap.{cat}.val.json`, `captions/correction_dict_{cat}.json` (`datasets.py:31-33,137,147,166,168,170`).
- Present: `data/fashionIQ_dataset/{captions,image_splits,images,qwen_captions}`; no `resized_image/`, no `correction_dict_*`.
- => **BLOCKED** (8 layout blockers, executed).

**3. preprocessing -> confirmed in source, never exercised.**
- `datasets.py:108-117`; `test.py:35`; `blip_processors.py:49-66`; `blip2.py:32,48,55`; `ConeSep.py:219-221,128`; `train.py:91-99`; `data_utils.py:92-106`.
- => **STATIC_AUDIT VERIFIED**; no image was ever transformed.

**4. official evaluator -> exists as a function, not reachable standalone.**
- `test.py:9-84`; only callsite `train.py:216`; no `__main__`; no checkpoint-loading path (`grep restore_file/load_checkpoint`).
- => **NOT RUNNABLE as an evaluator** (CF-1).

**5. workbench invocation -> does not exist.**
- `adapters/models.py:59-61` -> inherited `command()`; executed `NotImplementedError: upstream evaluation command requires audit before execution`.
- `preflight.py:184-195` -> `command not runnable: COMMAND_UNAUDITED`; `guarded_plan` -> `SKIPPED: adapter command not audited` (`evaluate_models.py:290-291`).
- `src/evaluation/fashioniq.py` full-gallery machinery: test-only callsites.
- => **BLOCKED.**

**6. checkpoint -> URL/identity verified; bytes untouched.**
- `models.yaml:178-208`; Hub `fiq/` listing (exactly three files, commit `36139d8`, 8.16 GB); three `curl -sI -L` -> `302 -> 200`, `x-linked-size: 2721428966`, three distinct `x-linked-etag` values. No download; `artifacts/checkpoints/` holds only `.gitkeep`.
- => **PARTIAL -- identity yes; bytes/loadability/training condition UNKNOWN.**

**7. output -> none.**
- `reproduction_status: NOT_RUN` (`:175`); no result record; `grep -rl conesep workbench/results/` -> empty.
- => **NOT_RUN.**

**8. metric comparison -> against the paper's *transcription*, never against a run.**
- Registry `:172-174` vs paper HTML vs `assets/results-fiq.png` -> identical for the 0 % row; the other three rows verified against the paper too.
- Executed arithmetic: `(50.96+56.98+58.80)/3 = 55.580000`; `(73.02+75.22+79.40)/3 = 75.880000`; `(55.58+75.88)/2 = 65.73`.
- => **transcription VERIFIED; reproduction NOT_RUN**, and (CF-3) the run-to-row binding is currently dangling.

---

## CHECKLIST

1. **Registered `model_id` present and unique** -- VERIFIED -- `models.yaml:146`; `git show HEAD:... | grep -n "model_id: conesep"` -> 146, one hit -- NONE -- none.
2. **`method_name`/`paper_title` match the upstream README title** -- VERIFIED -- `models.yaml:156,160` vs `third_party/ConeSep/README.md:5` -- NONE -- none.
3. **`paper_url` resolves and identifies the same work** -- VERIFIED -- `read https://arxiv.org/abs/2604.20358` HTTP 200, same title/authors -- NONE -- none.
4. **`upstream_repo_url` matches the paper's stated repo** -- VARIANT -- paper HTML `<https://github.com/Lee-zixu/ConeSep/>`; Hub card + `README.md:44` badge same; registry `:162` = `iLearn-Lab/CVPR26-ConeSep` (which does match `git config --get remote.origin.url`) -- P2 -- add the paper's URL as a `paper_repo_url`/note; keep the checkout URL.
5. **Registry `upstream_commit_sha` equals the checkout HEAD** -- VERIFIED -- `models.yaml:164` = `git -C workbench/third_party/ConeSep rev-parse HEAD` = `bd4bf629ebe68045c1a54dcc61e4a3ea65f8759b` -- NONE -- none.
6. **`upstream_default_branch` matches** -- VERIFIED -- `git branch -a --contains HEAD` -> `* main`, `remotes/origin/HEAD -> origin/main` -- NONE -- none.
7. **Checkout has no modified tracked files** -- VERIFIED -- `git status --porcelain | grep -vc __pycache__` -> `0` -- P3 -- none.
8. **Pin equals current upstream `main`** -- UNKNOWN -- single-commit checkout (`rev-list --count HEAD` -> 1); no fetch permitted -- P3 -- re-verify at run time.
9. **`source_available`/`source_dir` correct** -- VERIFIED -- `:165-166`; tree present -- NONE -- none.
10. **Declared `native_protocol` matches the source evaluator's gallery** -- VERIFIED -- `datasets.py:28,201-206`; `:170`; `docs/FASHIONIQ_PROTOCOL_PARITY.md:71` -- NONE -- none.
11. **`supported_protocols` is exactly the native one** -- VERIFIED -- `:171`; `val-split` branch `datasets.py:189-200` unreachable -- NONE -- none.
12. **`literature_split_label: original` consistent** -- VERIFIED -- `:169`; protocol table `:12-15` -- NONE -- none.
13. **Reference exclusion precedes ranking** -- VERIFIED -- `test.py:68` before `:71/:75` -- NONE -- none.
14. **Mask-vs-removal distinction recorded** -- VERIFIED -- `test.py:68` `-10e10`; `docs/FASHIONIQ_PROTOCOL_PARITY.md:71`; `src/evaluation/fashioniq.py:45-53,72` -- P3 -- keep visible.
15. **Recall@K maths (`100*hits/N`)** -- VERIFIED -- `test.py:76-81` -- NONE -- none.
16. **Denominator is the query count** -- VERIFIED -- `test.py:81` `len(nn_result)` -- NONE -- none.
17. **Sort is descending** -- VERIFIED -- `test.py:71,75` `np.argsort(-sims[i, :])` -- NONE -- none.
18. **Similarity matrix shape supports the indexing (prior-round claim)** -- VERIFIED-NEGATIVE for the prior claim -- executed `matmul((5,1,1,256),(7,256,32)) -> (5,7,1,32)`; `.squeeze() -> (5,7,32)`; `.max(-1) -> (5,7)`; `test.py:59-61` indexes a 2-D array, exactly as written -- P3 -- prior round's "4-D sims" is wrong; correct it in the review document.
19. **Tie-breaking pinned** -- UNKNOWN -- numpy default quicksort; `docs/FASHIONIQ_PROTOCOL_PARITY.md:35-36` -- P3 -- accept; measure-zero for `F.normalize`d floats.
20. **Macro/mean definition matches the paper (`(avg R@10 + avg R@50)/2`)** -- VERIFIED -- executed `65.73` both ways; paper prints `AVG.` -- NONE -- none.
21. **`reported_r10/r50/mean` match the paper** -- VERIFIED -- `:172-174` = `55.58/75.88/65.73` = paper HTML = `assets/results-fiq.png` -- NONE -- none.
22. **Category cells consistent with the average row** -- VERIFIED -- executed `50.96/56.98/58.80 -> 55.580000`, `73.02/75.22/79.40 -> 75.880000` -- NONE -- none.
23. **All four benchmark rows match the paper** -- VERIFIED -- `papers.yaml:588-779`; HTML grid gives the same four rows; recomputed averages agree; `derived_metrics: []` correct -- NONE -- none.
24. **Reported row's noise condition is labelled on the model entry** -- VARIANT -- `:172-174` carry no `0%` label; `:177` `notes` names only the published conditions; the four rows are `reference_status: VERIFIED` with no comparability field for `checkpoint_training_noise_pct` on the *model* side -- P1 -- add `paper_row_training_noise_pct: 0` on the entry, or cite `conesep_fiq_n00` explicitly.
25. **Paper row `checkpoint_id` resolves to a registered variant** -- BLOCKED -- executed dangling-id scan -> `conesep_fiq_n20/fiq_n20`, `n50/fiq_n50`, `n80/fiq_n80`; the only model in the registry with this defect; gate refuses with `checkpoint_id mismatch` -- P1 -- rename the rows' ids to `fiq_n02/05/08` (or register aliases) and add a consistency test over `papers.yaml` x `models.yaml`.
26. **Gate uses the correct reference for each checkpoint condition** -- VARIANT -- `select_reference` returns the right row (independent probe: `exact condition match` for 20/50/80), but `paper_gate.COMPARABILITY_FIELDS`'s `checkpoint_id` makes every conesep run `BLOCKED_CHECKPOINT_MISMATCH` -- P1 -- same fix as 25.
27. **`parity_status` cannot contradict the gate** -- BLOCKED -- executed `_check_parity({'r10':54.93,...}, 0%-row) -> PARITY_VERIFIED` and `{'r10':53.39,...} -> PARITY_VERIFIED`; `paper_metrics` built unconditionally from the headline row (`evaluate_models.py:309`); surfaced at `main.py:115`, `frontend/gate.ts:8` -- P1 -- build `parity` from the *selected* reference, or delete the field.
28. **Protocol machinery actually reaches `conesep`** -- BLOCKED -- all production callsites of the full-gallery helpers are inside `src/evaluation/fashioniq.py`; every external call is in `tests/unit/`; `workbench/replay/` has no conesep -- P1 -- wire an audited adapter/replay before any conesep claim.
29. **Gallery equals the full val split in file order** -- VERIFIED -- `datasets.py:203` `for id, image_name in enumerate(images)` -- NONE -- none.
30. **Layout id `fashioniq_ilearn_resized` matches source paths** -- VERIFIED -- `fashioniq_layout.py:25-34` vs `datasets.py:31-33,166,168,170` -- NONE -- none.
31. **Image extension `.jpg` matches source** -- VERIFIED -- `datasets.py:137,147`; `fashioniq_layout.py:85` -- NONE -- none.
32. **Local dataset layout satisfies the registered contract** -- BLOCKED -- executed 8 blockers; `ls data/fashionIQ_dataset` -> `captions image_splits images qwen_captions` -- P1 -- stage the resized tree + dictionaries, or re-register the true layout.
33. **Layout checker covers every file the evaluator opens** -- VARIANT -- cache JSON + six PKLs (`datasets.py:35,52-58`) unchecked; train captions only in the `_training` variant (`fashioniq_layout.py:37-41`) -- P2 -- add the cache branch or invalidate on mismatch.
34. **Cache staleness guarded** -- NOT_RUN (absent) -- `datasets.py:35` short-circuits `:41-49` when the JSON exists; `:52-59` loads PKLs without re-reading sources -- P2 -- require a provenance stamp.
35. **Correction dictionaries present and compared** -- VARIANT -- `datasets.py:96,170`; `grep -c conesep auxiliary_assets.yaml` -> 0; executed: identical bytes also live at `LIMN/correct_files/` and `TGCIR/correction_dict/` (`9e4a5987...`, `ca29e29e...`, `3dfbf15f...`) -- P2 -- register the three dictionaries with those digests.
36. **Caption normalization matches source (lowercase, all punctuation, token map)** -- VERIFIED -- `datasets.py:108-113` -- NONE -- none.
37. **Caption composition order and `" and "` joiner** -- VERIFIED -- `datasets.py:116` -- NONE -- none.
38. **Second caption stage (LAVIS eval) is correct and not a no-op** -- VERIFIED -- `test.py:35`; `blip_processors.py:49-66` (regex `[.!\"()*#:;~]`, `max_words=50`) -- NONE -- none.
39. **Tokenizer path pinned in the registry** -- VARIANT -- source hardcodes `/root/bert-base-uncased` (`blip2.py:32,48,55`); the BERT row exists only for `csmcir` (`auxiliary_assets.yaml:249`) -- P2 -- register it for conesep or patch to an env var.
40. **`max_txt_len` recorded** -- VERIFIED (source) / UNKNOWN (registry) -- `ConeSep.py:128` default 32, used at `:219-221` -- P3 -- note it in the prep contract.
41. **Image transform recipe matches source** -- VERIFIED -- `train.py:91-99`; `data_utils.py:92-106` (TargetPad 1.25 -> Resize 224 BICUBIC -> CenterCrop -> CLIP Normalize) -- NONE -- none.
42. **Staged resized images have a pinned generation recipe** -- VARIANT -- no producer script in the tree; zero registry rows -- P2 -- pin a resizer + JPEG quality, or hash the staged tree.
43. **Training noise touches only training data** -- VERIFIED -- `datasets.py:61-90`; called at `train.py:109,117` only; `grep shuffle test.py` empty -- NONE -- none.
44. **Evaluator runs in eval mode / no grad** -- VERIFIED -- `test.py:10`, `:21` -- NONE -- none.
45. **Checkpoint variants declared with noise provenance** -- VERIFIED -- `:178-208` -- NONE -- none.
46. **Checkpoint URLs resolve** -- VERIFIED (STATIC) -- three `curl -sI -L` -> `302` then `200` -- NONE -- none.
47. **Checkpoint sizes recorded** -- VARIANT -- all `null` (`:185,195,205`) though the Hub reports `2721428966` -- P2 -- record them.
48. **Checkpoint sha256 recorded** -- VARIANT -- all `null` (`:186,196,206`) though the Hub exposes per-file `x-linked-etag` -- P2 -- record them.
49. **Checkpoint file set matches the author Hub** -- VERIFIED -- `fiq/` listing = exactly the three registered filenames -- NONE -- none.
50. **A clean (0 %) checkpoint is published** -- VERIFIED-NEGATIVE -- Hub tree = 3 files, all 20/50/80; README has no checkpoint section -- NONE (registry correct) -- none.
51. **`strict_clean_checkpoint_available` is accurate** -- VERIFIED -- `:176` false; executed `select_checkpoints` -> BLOCKED under both policies -- NONE -- none.
52. **Checkpoint loadable by pinned code** -- UNKNOWN -- no loader (`restore_file` unread, `utils.load_checkpoint` uncalled, files are whole-model pickles per the Hub import list) -- P1 -- see CF-1.
53. **Standalone official evaluator exists** -- VERIFIED-NEGATIVE -- `grep -n "__main__\|argparse\|ArgumentParser" test.py` -> no match; only `train.py:216` -- P1 -- write and audit an eval entrypoint.
54. **Workbench adapter produces a command** -- VERIFIED-NEGATIVE -- `adapters/models.py:59-61`; executed `NotImplementedError: upstream evaluation command requires audit before execution` -- P1 -- implement `ConeSepAdapter.command` after 53/52.
55. **Preflight classification reflects the declared source defect** -- VERIFIED -- executed `preflight_blockers` -> 10 blockers, `classify` -> `('BLOCKED_SOURCE', "source cannot run: no runnable evaluation path ...")`; `pc_readiness.model_section` -> `state: BLOCKED_SOURCE`, `auxiliary_assets: []` -- NONE -- none.
56. **Run output exists and parity was compared** -- BLOCKED -- `reproduction_status: NOT_RUN` (`:175`); no result record; `guarded_plan` -> `SKIPPED: adapter command not audited` -- P1 -- no conesep number can be produced today.

---

**FINAL VERDICT: BLOCKED** -- the registry's static facts for `conesep` hold against the pinned tree and the paper (pin, protocol, `strict_clean=false`, all four metric rows digit-exact), but there is no runnable path to any number (no standalone evaluator, no checkpoint loader, unaudited/declining adapter, absent registered dataset layout and dictionaries), the pinned row is the 0 %-noise row for which no weight is published, and two committed defects -- dangling `checkpoint_id`s in the three non-clean paper rows and a `parity_status` still judged against the 0 % headline -- mean a future run would be refused as a checkpoint mismatch while a stale field prints `PARITY_VERIFIED`.
