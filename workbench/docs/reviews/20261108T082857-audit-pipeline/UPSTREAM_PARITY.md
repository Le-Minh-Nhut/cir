# UPSTREAM_PARITY — `habit` (BLIP-2) and `encoder` (open_clip)

The rule this file obeys: comparing the adapter to another invocation of the same adapter proves
nothing. Each row is either **executed** on a CPU-feasible slice, or marked `STATIC_ONLY` with the
reason. Where upstream code could not be run, the row says so and does **not** claim parity.

Executed in this audit:
- ENCODER's resolved `preprocess_train` / `preprocess_val` chains, and the measured pixel divergence
  between them on real released images (`logs/encoder_preprocess_train_vs_val.txt`,
  `logs/encoder_transform_divergence.txt`).
- ENCODER's empty-text tokenization through `open_clip.get_tokenizer('ViT-B-32')`.
- HABIT-family caption processing through the **pinned LAVIS processor** (`BlipCaptionProcessor`),
  obtained via `replay/lavis_processors.text_processors` from the pinned BLIP-2 config — i.e. the same
  object the adapter uses, not a reimplementation.
- HABIT-family BERT tokenization of the processed caption and of `""`.
- LIMN's checkpoint normalization buffers + open_clip defaults.
- The released gallery's duplicate structure (the tie regime both models are scored under).

Not executed: any GPU forward, any upstream `test.py` run, any full-gallery encode. See `NOT_RUN`.

---

## 1. HABIT — upstream `AAAI26-HABIT` vs `workbench/composition/model_habit.py`

Upstream files: `third_party/HABIT/{test.py,datasets.py,data_utils.py}`, model
`lavis/models/blip2_models/HABIT.py`.

| # | Axis | Upstream behaviour | Workbench adapter | Identical? | Verdict |
|---|---|---|---|---|---|
| 1 | Image transform construction | `data_utils.py:92-105` `targetpad_transform(1.25, 224)`, built in the eval wrapper (`replay/habit.py:163`) | same function, same args (`composition/model_habit.py:340`) | **YES** — same function object from the same checkout | identical |
| 2 | Transform chain | TargetPad(1.25) → Resize(224,BICUBIC) → CenterCrop(224) → rgb → ToTensor → CLIP Normalize (`data_utils.py:99-104`) | same list | YES | identical |
| 3 | Aspect handling | pad to ratio 1.25, no stretch (`data_utils.py:64-76`) | same | YES | identical |
| 4 | Query vs gallery transform | one `transform` for the dataset; both `source_img_data` and `target_img_data` go through it (`datasets.py:139,141` and `:195,202`) | one `transform`, one `decode` for both phases | YES | identical |
| 5 | Eval-time augmentation | none (TargetPad/Resize/CenterCrop are deterministic) | none | YES | identical |
| 6 | RGB / channels | `img.convert('RGB')` (`datasets.py` `get_img`) | `image.convert("RGB")` in the worker | YES | identical |
| 7 | Text processor identity | `txt_processors["eval"]` from `configs/models/blip2/blip2_pretrain.yaml` → `blip_caption` → `BlipCaptionProcessor(prompt="", max_words=50)` (`blip_processors.py:29-73`) | `processors[0]["eval"]` from the same config (`replay/lavis_processors.py:199-218`) | **YES — executed**: lowercases, strips `.!\"()*#:;~`, collapses whitespace, `max_words=50` | identical |
| 8 | Caption string / separators / prefixes | adapter feeds `query.query_txt` verbatim; upstream feeds `concat_text(captions[0], captions[1])` joined with `" and "` (`datasets.py:120`) | released `query_txt` used unchanged; never re-split on "and" | **NO — intentional adaptation** | The released CIRCUS text *is* the already-composed caption. Re-splitting it to re-join it would invent structure the release does not assert. Justified by `CIRCUS_COMPOSITION_SPEC.md` "Query text is passed as supplied; it is never re-split on `and`". |
| 9 | Tokenizer | `Blip2Base.init_tokenizer` → BERT from `/root/bert-base-uncased`, `[DEC]` added as bos (`blip2.py:31-34`) | in-model, no adapter tokenizer | YES (same model code) | identical — but see row 10 |
| 10 | `bert-base-uncased` asset availability | upstream reads a machine-local path | the local path is **absent/unreadable here**; a cached HF snapshot exists instead | **NO — environment-level** | `UNRESOLVED`: which vocab the model actually resolves at run time was not executed. Absent the asset the model load would fail, and a HABIT run did complete, so some resolution works; *which* one is not established. |
| 11 | Max text length / padding / truncation | `max_length=32, padding="max_length", truncation=True` (`HABIT.py:392-398`) | same, inside the model | YES | identical |
| 12 | Padding excluded from pooling | attention mask covers Q-Former queries + real text tokens (`HABIT.py:399-407`); pooling is the fixed slot `[..., 32, :]` | same | YES | identical |
| 13 | Empty text token pattern | not exercised upstream by the release path | executed: `""` → `[CLS][SEP]` + 30 pads (2 non-pad ids); attention mask length 2 | n/a | **intentional adaptation** — spec allows ordinary special tokens; verified to be exactly that |
| 14 | Feature width / pooling | 256-d, `text_proj(fusion[...,32,:])` then L2 (`HABIT.py:408-413`) | 256-d | YES | identical |
| 15 | Target representation | `(G, 32, 256)` from `vision_proj` over Q-Former tokens, `permute(0,2,1)` (`HABIT.py:416-436`) | same layout, transposed in the worker | YES | identical |
| 16 | Similarity | `np.matmul(all_queries, all_imgs)` then `.max(-1)` over the 32 vector pairs (`test.py:141-143`) | `np.einsum("nd,gqd->ngq",…).max(-1)` (`model_habit.py:246-258`) | **YES semantically** | identical reduction; expression rewritten, not re-derived |
| 17 | Query normalization before dot | commented out in the release branch (`test.py:133-137` are comments) — relies on in-model L2 | relies on in-model L2 | YES | identical |
| 18 | Gallery | per-category FashionIQ split list, deduplicated `source ∪ target` (`datasets.py` `get_test_data`) | full released ordered 74,381 list | **NO — intentional adaptation** | `CIRCUS_COMPOSITION_SPEC.md` "Protocol and gallery". Different experiment by design. |
| 19 | Reference eligibility | **masked**: `sims[i, targets.index(source)] = -10e10` (`test.py:151`) | **eligible**, measured rank 1 | **NO — intentional adaptation** | spec: "The reference is eligible in ranking … The native `-10e10` reference mask … is deliberately not inherited." |
| 20 | Rank indexing | `np.argsort(-sims)[:k]` then positional membership test (`test.py:154,78`) — effectively 1-based membership | `1 + #greater + #earlier-equal` | YES for the reported rank | identical outcome, different implementation |
| 21 | Metric definition | `recall_top{k}_correct_composition` = percent over queries, k∈{1,10,50} (`test.py:70-76`); printed **per category**, no source-level macro (`evaluate_model.py:72-75`) | Recall percent + nDCG/MRR fractions + gap (`metrics.py:73-76`), pooled over the cohort | **NO — intentional** | spec defines nDCG/gap as primary; also note upstream reports three category numbers whereas the composition protocol reports one pool |
| 27 | Gallery size (upstream val-split) | `split.<cat>.val.json` lengths measured locally: **3817 / 6346 / 5373** for dress/shirt/toptee | 74,381 released identities | **NO — intentional** | quantified: upstream's per-category gallery is 5-20x smaller, so upstream Recall is not comparable to a 74k full-gallery number in either direction |
| 28 | `TargetPad` no-op branch | if `max(w,h)/min(w,h) < 1.25` the image is **not padded at all** (`data_utils.py:64-73`) | same | YES | identical — worth stating because "TargetPad" is not a constant operation |
| 22 | Checkpoint key matching | whole pickled module | trusted whole-model load + expected sha256 | YES | identical |
| 23 | `.eval()` / no_grad | `model.eval()` (`test.py:10`), forward decorated `@torch.no_grad()` | `.eval()`, `torch.no_grad()` | YES | identical |
| 24 | dtype / autocast | `maybe_autocast()` inside the model (`blip2.py:36`) | inherits model code | YES | identical |
| 25 | Batch size effect on results | upstream batches at 64 | 32 (or `WORKBENCH_COMPOSITION_BATCH_SIZE`) | NO, immaterial | executed: batch-1 vs batch-32 score delta ≤ 6.6e-7, **0 rank changes** (`logs/differential_stages.json`, `logs/rank_batch200.txt`) |
| 26 | Final incomplete batch | `len(imgs) >= batch_size or t is test_queries[-1]` (`test.py:107`) | `range(0,n,b)` remainder | YES | identical semantics; adapter asserts completeness |

### HABIT summary

Faithful on every preprocessing, tokenization, pooling and similarity axis (rows 1-7, 11-17, 21-26).
Divergent on exactly four things, all of them **intentional and documented in the spec**: caption
re-joining (8), gallery composition (18), reference masking (19), and metric set (21). One genuine
`UNRESOLVED`: which `bert-base-uncased` vocab is actually resolved (10).

---

## 2. ENCODER — upstream `ENCODER` vs `workbench/composition/model_encoder.py`

Settled transform analysis lives in `ENCODER_LIMN_PREPROCESSING_VERDICT.md`; cited, not re-derived.

Upstream files: `third_party/ENCODER/{evaluate_model.py,datasets.py,test.py,model_try2.py}`.

| # | Axis | Upstream behaviour | Workbench adapter | Identical? | Verdict |
|---|---|---|---|---|---|
| 1 | Transform construction | `create_model_and_transforms('ViT-B-32', pretrained=open_clip_pytorch_model.bin)` → `[preprocess_train, preprocess_val]` (`evaluate_model.py:40,43`) | same call, third return only (`model_encoder.py:321`) | construction YES; selection **NO** | see rows 2-3 |
| 2 | Reference transform | `preprocess_train` = RandomResizedCrop(224, scale=(0.9,1.0), bicubic) — **stochastic, unseeded** (`datasets.py:177`) | `preprocess_val` = Resize+CenterCrop — deterministic | **NO** | **INTENTIONAL ADAPTATION.** Executed: mean abs pixel diff 0.16/0.50/0.21, unseeded draw spread up to 3.83 normalized units. Upstream is not reproducible; deterministic is the only reproducible choice for a released-gallery protocol. `ENCODER_LIMN_PREPROCESSING_VERDICT.md` §6-7. |
| 3 | Gallery transform | `preprocess_val` (`datasets.py:195,202`) | `preprocess_val` | **YES** | identical |
| 4 | Query target transform | `stage=0` computed at `datasets.py:178` but **never read** by `test.py` (only `source_img_data` at `:27`) | n/a | n/a | dead upstream code; no parity obligation |
| 5 | Eval-time augmentation | **yes, effectively** — RandomResizedCrop on every reference | none | **NO** | same row as 2 |
| 6 | Aspect handling | val: shortest-edge Resize + CenterCrop | same | YES | identical |
| 7 | Normalization mean/std | CLIP constants via open_clip | same object | YES | identical |
| 8 | Tokenizer | `open_clip.get_tokenizer('ViT-B-32')` (`model_try2.py:137`), 77-length, **no attention mask** | same (in-model) | YES | identical |
| 9 | Padding handling in pooling | `mean(fuse_local, dim=1)` over **6** fused tokens `cat([global(1), action(N_p=2), selected(local_token_num=3)])` (`model_try2.py:412,428,490`); the 77-token text context is never pooled directly, but pads influence token selection via `text_local_fea` and `ScoreNet(sparse_ratio=0.6)` keeping 47 of 77 (`:42-50,77-89`) | same code | YES | identical — consequence: `image_only` is a pad-driven `[SOT][EOT]` query, not "no text" |
| 10 | Empty text tokens | not exercised | executed: `''` → `[49406,49407,0×75]` | n/a | **intentional adaptation**, spec-compliant |
| 11 | Caption string | upstream joins two captions with `" and "` from `cap.<cat>.val.json` | released `query_txt`, unchanged | **NO — intentional** | spec: never re-split |
| 12 | Feature width / pooling | 512-d, `F.normalize(mean(fuse_local,1))` (`model_try2.py:487-497`) | same | YES | identical |
| 13 | Similarity | `all_queries.dot(all_imgs.T)` after per-row L2 (`test.py:55-62`) | L2 both sides + dot, chunked 8192 (`model_encoder.py:221-238`) | YES | identical; chunking changes nothing (executed: 0 rank diffs) |
| 14 | **Reference mask** | **`sims[i, targets.index(source)] = -10e10` (`test.py:68`)** | **not applied** | **NO — intentional** | spec requires reference-eligible ranking; measured probe returns rank 1 |
| 15 | Gallery | per-category deduplicated val-split list | full released 74,381 ordered list | **NO — intentional** | different protocol by design |
| 16 | Rank computation | `argsort(-sims)[:50]`, then `test_targets_id.index(target) in nns[:k]` — 0-based positions, membership test | 1-based exact rank over the whole gallery | **NO — intentional** | spec defines exact full-gallery rank `r` (1-based) and derives hit/nDCG/MRR from it. Upstream truncates at 50 and reports membership only. |
| 17 | Checkpoint load | `load_state_dict(model, strict=False)` (`evaluate_model.py:75`) | `load_state_dict(state, strict=False)` + `_assert_checkpoint_loaded` (`model_encoder.py:315-318`) | YES, adapter adds an assertion | identical + more strict |
| 18 | Base weights source | `open_clip_pytorch_model.bin` next to the source | same file under the vendored source root | YES | identical |
| 19 | `.eval()` / device | `Encoder_model.cuda()` then `test()` calls `model.eval()` (`test.py:7`) | `.cuda().eval()` | YES | identical |
| 20 | Determinism of the forward | not pinned | not pinned in composition; replay pins with `enable_deterministic_inference` (`replay/encoder.py:288`) | NO | `SUSPECTED`: composition ranks are reproducible in the *controller* stage (executed A4: bit-identical) but per-run GPU determinism is not claimed. |
| 21 | Batch effect | 64 | 32 | NO, immaterial | executed: ≤6.6e-7 score delta, 0 rank flips |
| 22 | Metric units | percent Recall only | Recall percent + nDCG/gap fractions | **NO — intentional** | spec's unit discipline |

### ENCODER summary

Preprocessing differs in exactly one defended way (rows 2/5: deterministic vs stochastic reference
transform) and that difference **cannot change the composition metrics** because the composition
protocol uses the deterministic choice consistently for query and gallery — internally coherent, and
already labelled as a non-reproduction (`reproduction_status:
METRICS_MEASURED_MAPPING_UNVERIFIED`). Ranking semantics differ intentionally (rows 14, 16) because
the spec defines an exact full-gallery rank with the reference eligible.

The one thing a reader cannot currently recover from an ENCODER summary is *which* transform served
which role. That is a provenance/note gap (`FINDINGS.md` F2 residual), not a behavioural defect, and is
why no patch is proposed for it.

---

## 3. LIMN (third model, for completeness)

| Axis | Upstream (`LIMN/datasets.py`) | Adapter (`model_limn.py`) | Verdict |
|---|---|---|---|
| Eval transform | `get_test_img` → `transform[1]` (val) for reference, target, gallery (`:151,153,169`) | `image_transform(..., is_train=False, ...)` for all roles (`:527`) | **identical** |
| Training transform leakage | `transform[0]` only via `__getitem__` (`:104-105`), never in eval | not used | identical |
| Normalization constants | open_clip defaults | resolved from the tower; checkpoint has no mean/std buffers, so = defaults (executed) | **identical** |
| `dataset_type` id | `{dress:0, shirt:1, toptee:2}` (`:21-24`) | same literal (`:512`) | identical |
| Reference mask | not masked in the val path | not masked | identical |
| Rank | own argsort | 1-based, earlier-tie-wins (`:400-416`) | intentional (spec rank definition) |

---

## 4. What could not be executed, and therefore is NOT claimed

| Claim that would need execution | Why not | Marking |
|---|---|---|
| "Upstream HABIT reproduces its published R@10 on the val split" | needs the `resized_image/<category>/` tree + CUDA; GPU busy all audit | `STATIC_ONLY` |
| "Upstream ENCODER reproduces its published R@10" | same | `STATIC_ONLY` |
| "HABIT's BERT vocab is the pinned `/root/bert-base-uncased`" | path not readable here; a HF snapshot exists | `UNRESOLVED` |
| "The two protocols agree anywhere" | they are different protocols (gallery, mask, rank semantics) | not claimed |
| "Adapter == adapter" | excluded by the rule at the top of this file | n/a |

Every "identical" row above is either (a) the same function/object from the same vendored checkout, or
(b) executed on a CPU slice. No row asserts parity from source reading alone.
