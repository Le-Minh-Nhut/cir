# PIPELINE_TRACE — static trace per model family

Every claim below carries a `file:line`. Where a line was read directly in this audit it is unmarked.
Where a fact comes from a grep sweep rather than full-file reading, it is marked `[GREP]`. Nothing in
this file was executed on GPU (see `NOT_RUN`); all executed evidence lives in `FINDINGS.md` and
`DIFFERENTIAL_TESTS.md`.

Shared contract, true for all nine adapters unless a row says otherwise:

- Gallery = the released ordered list of **74,381** identities, order preserved, no sort/dedupe
  (`data.py:214`).
- Reference **is** eligible in the gallery; upstream's `-10e10` mask is deliberately not inherited
  (`CIRCUS_COMPOSITION_SPEC.md`, "Protocol and gallery"; measured per adapter via
  `adapters.py:424-470`).
- Query text is passed through unchanged, never re-split on "and"
  (`inputs.py:20-28`; no split anywhere in `workbench/composition/`).
- Conditions come from `inputs.adapt_query_inputs` (`inputs.py:20-28`): `multimodal` → raw image +
  text; `text_only_black_image` → `black_image_like` raw PIL replacement + text;
  `image_only_empty_text` → raw image + `""`.
- Black replacement happens on the **raw PIL image before the transform**, staged to a PNG which the
  adapter then opens (`runner.py::_stage_query_path`). Executed proof: constant normalized tensor
  (−1.675 mean), not zeros — see `FINDINGS.md` D4.
- Scoring is controller-side NumPy over cached float32 feature matrices; `rank` is 1-based and ties
  break by earlier gallery position (`adapters.py:352-368`).

---

## 1. BLIP-2 family

### 1.1 `habit`

**Images.** `targetpad_transform(1.25, 224)` built once per worker
(`composition/model_habit.py:340` `[GREP]`, upstream `third_party/HABIT/data_utils.py:92-105`).
Chain: `TargetPad(1.25)` → `Resize(224, BICUBIC)` → `CenterCrop(224)` → `_convert_image_to_rgb` →
`ToTensor()` → `Normalize((0.48145466,0.4578275,0.40821073),(0.26862954,0.26130258,0.27577711))`.
`TargetPad` pads to ratio 1.25 only when the image is more elongated than that, else passes it through
(`data_utils.py:64-76`) — aspect ratio preserved by **pad**, never stretch. Decode is PIL with
`convert("RGB")` (`model_habit.py:~365`), so channel order is RGB, 3 channels, uint8 → float via
`ToTensor`. **Query and gallery use the same transform** (one `transform` object, one `decode`
closure used by both phases — `model_habit.py:~358-375`). No augmentation in eval: `TargetPad` and
`Resize`/`CenterCrop` are deterministic, so no train-time augmentation leaks. Batch decode is
parallelised with `prefetch_rows` but `map` preserves order, so rows stay aligned with gallery order
(`adapters.py:120-160`).

**Text.** `text_processor = processors[0]["eval"]` from the pinned LAVIS config
`configs/models/blip2/blip2_pretrain.yaml` → `blip_caption` → `BlipCaptionProcessor(prompt="",
max_words=50)` (`third_party/HABIT/lavis/processors/blip_processors.py:29-73`). Applied per caption in
the worker (`model_habit.py:~370`). Executed: it lowercases, strips `.!\"()*#:;~` to spaces, collapses
runs of whitespace, strips, and truncates to 50 **words**. Output is a *string*, not ids — the ids are
produced inside the model's own `self.tokenizer` (`HABIT.py:392-398`) with
`padding="max_length", truncation=True, max_length=32`. BERT tokenizer with `[DEC]` added as bos
(`blip2.py:31-34`); effective special tokens are `[CLS] … [SEP]` + pad. So the adapter never sees or
constructs tokens; it hands over the processed string.

**Pooling / label leakage.** Pooling is a **fixed slot**, not a mask-weighted mean:
`fusion_output.last_hidden_state[:, 32, :]` (`HABIT.py:410-412`) — index 32 = the first text token
position after the 32 Q-Former query tokens. Padded text tokens are excluded from **attention** by
`attention_mask = cat([query_atts, text_tokens.attention_mask])` (`HABIT.py:399-407`), so padding
cannot contribute. No label information reaches the query: the query is (reference image, text) only.

**Model / scoring.** Whole pickled model loaded with a trusted loader, `.to(device).eval()`
(`model_habit.py:~333`). `torch.no_grad()` around the forwards. Forward args are
`(pixels, texts)` for queries, `(pixels)` for targets. Query emits one L2-normalized 256-d vector
(`HABIT.py:408-413`); target emits `(G, 32, 256)` permuted to that layout (`HABIT.py:416-436`,
`model_habit.py` target path transposes to `(0,2,1)`). Scoring is
`np.einsum("nd,gqd->ngq", queries, targets).max(axis=-1)` (`model_habit.py:246-258`) — a **max over the
32 matched/mismatched token pairs**, i.e. upstream's own dual-vector matching, then max over targets.
No temperature at inference. Rank is 1-based over the full gallery.

**Compatibility.** Checkpoint keys: the worker either loads a whole module or calls
`load_state_dict(strict=False)` and then asserts through `_assert_checkpoint_loaded`
(`model_encoder.py:315-318` shows the same pattern; HABIT uses `load_trusted_whole_model` plus the
checkpoint's expected sha256). `strict=False` is a real risk surface — see `FINDINGS.md` §D1(c).

### 1.2 `intent`, `hint`

Structurally identical to `habit` on every axis measured:

- `transform = data_utils.targetpad_transform(1.25, 224)` — `model_intent.py:325`, `model_hint.py:329`
  `[GREP]`.
- `network.to(device).eval()` — `model_intent.py:324`, `model_hint.py:328` `[GREP]`.
- score `np.einsum("nd,gqd->ngq", …).max(axis=-1)` — `model_intent.py:248`, `model_hint.py:250`
  `[GREP]`, i.e. the same `(G, Q=32, 256)` target layout and the same 256-d query width.

Their text processors come from the same LAVIS mechanism
(`replay/lavis_processors.py:199-218` + each `replay/<model>.py` `PROCESSOR_CONFIGS`), so the
`blip_caption` normalization applies identically. Verified internals (agent trace, line-cited):
ids are built by `blip2_cir.py:744-751` with `padding="max_length", truncation=True,
max_length=32`; the **attention mask is used**
(`attention_mask = cat([query_atts, text_tokens.attention_mask])`, `blip2_cir.py:753-763`); pooling
is the **fixed index 32** (`:767-769`), i.e. the first text token after the 32 query embeds, so pads
are excluded from both attention and pooling. Checkpoints are whole pickles with a mandatory SHA-256
(`model_intent.py:316-317`; `common.py:45-56` refuses a missing digest). `intent` and `hint` differ
from `habit` in model code and checkpoint only, not in the preprocessing or metric contract.

### 1.3 `tme`

- Transform: `targetpad_transform(target_ratio=1.25, dim=224)` imported from the **TME** checkout's own
  `utility` module (`model_tme.py:353,368` `[GREP]`) — same geometry and normalization as the rest of
  the family, but a *different source file*, so it is a separate copy of the same math.
- Checkpoint load: `load_state_dict(state, strict=False)` (`model_tme.py:362` `[GREP]`) — note this is
  the pattern where the base model comes from lavis/HF cache and only the fine-tuned delta is staged;
  `strict=False` can silently accept a partially-matching checkpoint.
- `network.eval()` (`model_tme.py:367` `[GREP]`).
- Text: the worker builds ids itself with `max_length=network.max_txt_len, return_tensors="pt"`
  (`model_tme.py:406` `[GREP]`) — so unlike HABIT the adapter owns the tokenizer call.
- Score: `np.einsum("nd,gqd->ngq", …).max(axis=-1)` (`model_tme.py:252` `[GREP]`), same layout.
- Registry note: TME's checkpoint is `fashioniq_0` (training noise 0%), and no 20% variant exists —
  a cross-model non-comparability the spec already labels.

### 1.4 `conesep`, `csmcir`, `ptha_mtst`

These three differ from the `habit`/`intent`/`hint`/`tme` group in the **target layout**, and that is
the material difference:

| model | score expression | target layout | load |
|---|---|---|---|
| `conesep` | `np.einsum("nd,mdk->nmk", query, target, optimize=True).max(axis=-1)` — `model_conesep.py:79` `[GREP]` | `(M, d, k)` | whole model, `.cuda().eval()` `model_conesep.py:296` `[GREP]` |
| `csmcir` | same `"nd,mkd->nmk"` form after explicit L2 of the target matrix (`vectors / norm`) — `model_csmcir.py:115-116` `[GREP]` | `(M, k, d)` | `load_state_dict(strict=False)` `model_csmcir.py:217` `[GREP]` |
| `ptha_mtst` | same `"nd,mkd->nmk"` form — `model_ptha_mtst.py:86` `[GREP]` | `(M, k, d)` | `load_state_dict(payload[key], strict=True)` `model_ptha_mtst.py:315` `[GREP]` |

All three resample with `targetpad_transform(1.25, 224)` from their own `data_utils` variants
(`model_conesep.py:297`, `model_csmcir.py:255,266`, `model_ptha_mtst.py:318-319` `[GREP]`).

Note the axis-order disagreement inside one family: `conesep` indexes `mdk` while `csmcir`/`ptha_mtst`
use `mkd`, and `ptha_mtst` declares width `32 * 256` (`model_ptha_mtst.py:82`). All three then reduce
with `.max(axis=-1)`. Each is internally consistent with its own encode path, but a target matrix
produced in one convention and scored in another would silently produce a valid-shaped, wrong score.
`ptha_mtst` is the only one of the nine using `strict=True`, which is the safer choice.

`csmcir` cannot be a real result: per the spec its gallery needs per-image `COT_ours2` captions that
the released CIRCUS inputs do not carry, and the spec explicitly forbids substituting the local
`qwen_captions`. Any stored `COMPLETED` row for `csmcir` should be treated as unverified.

---

## 2. open_clip family

### 2.1 `encoder`

**Images.** `open_clip.create_model_and_transforms("ViT-B-32", pretrained=<staged
open_clip_pytorch_model.bin>)` and the adapter takes the **third** return value only:
`model_encoder.py:321-322`. Resolved chain (executed, `logs/encoder_preprocess_train_vs_val.txt`):
`Resize(224, BICUBIC, antialias=True)` → `CenterCrop(224)` → `_convert_to_rgb` → `ToTensor()` →
`Normalize` (CLIP constants). Aspect ratio preserved by shortest-edge resize then centre crop.
No resize-to-square stretch. One transform serves **queries, positives and gallery** — the same
`decode` closure is used for both phases (`model_encoder.py:336-341`, phase dispatch at `:349-355`).
No eval-time augmentation.

This is a **deliberate divergence** from upstream, which uses `preprocess_train`
(RandomResizedCrop, stochastic) for the reference image and `preprocess_val` for gallery rows
(`third_party/ENCODER/datasets.py:177` vs `:195`; `evaluate_model.py:38-44`). Settled in
`ENCODER_LIMN_PREPROCESSING_VERDICT.md`: intentional adaptation, no metric consequence. See
`UPSTREAM_PARITY.md`.

**Text.** `model.extract_retrieval_compose(batch, texts)` → `compose_feature` →
`self.tokenizer(txt)` where `self.tokenizer = open_clip.get_tokenizer('ViT-B-32')`
(`third_party/ENCODER/model_try2.py:137,385`). That tokenizer produces a **fixed 77-length padded id
matrix**; the tower applies only CLIP's causal `attn_mask` (`:343`), never a padding mask.
Executed: `''` → `[49406, 49407, 0×75]`.

Pooling is `torch.mean(fuse_local, dim=1)` (`model_try2.py:490`), and `fuse_local`'s token axis is
`cat([global(1), action(N_p=2), selected(local_token_num=3)])` = **6 tokens**, not 77
(`model_try2.py:412,428`; adapter ctor `local_token_num=3`, `N_p=2` at `model_encoder.py:309`).
Verified by shape propagation through `FeatureWiseAffine` (`:435-454`, `chanel = x.shape[1]*2` = 12 →
gamma/beta reshape `(B,6,512)`).

Pads still reach the query **indirectly**: the 77-token context forms `text_local_fea` (`:386-388`),
which scores token selection — `ScoreNet(sparse_ratio=0.6)` keeps `ceil(77*0.6)=47` of 77 (`:42-50`)
and `FactorNet` then softmaxes those 47 down to 3 (`:77-89`). For empty text all 75 pads share one
embedding id, so pads plausibly fill the 47 slots (`INFERENCE`, `STATIC_ONLY` — needs a forward).
Net: `image_only_empty_text` for ENCODER is a pad-driven `[SOT][EOT]` query, **not** "no text", and
not comparable to the masked/pad-excluded conditions of the other families. No label leakage.

**Model / scoring.** `load_state_dict(state, strict=False)` then `_assert_checkpoint_loaded`
(`model_encoder.py:315-319`), `.cuda().eval()`. `torch.no_grad()` per batch. Query output is one
512-d vector, `F.normalize(mean(fuse_local, dim=1))`; target is `F.normalize(mean(tag_local, dim=1))`
(`model_try2.py:487-497`). Adapter scoring re-normalizes both sides and dots, chunked over 8192
gallery rows (`model_encoder.py:221-238`). No reference mask — upstream's `-10e10`
(`test.py:68`) is deliberately not inherited. Rank 1-based.

### 2.2 `limn`

**Images.** Per-category transform built from the **loaded checkpoint's** visual tower:
`open_clip.image_transform(visual.image_size, is_train=False, mean=visual.image_mean,
std=visual.image_std)` (`model_limn.py:527-528`). Executed: the checkpoint carries no
`image_mean`/`image_std` buffers, so these resolve to open_clip defaults
`(0.48145466,…)`/`(0.26862954,…)` — identical to upstream, and `image_size` = 224. One transform per
scope, used for queries **and** targets. Correct: upstream LIMN's eval path is `get_test_img` →
`transform[1]` (validation) for reference, target and every gallery row
(`third_party/LIMN/LIMN/datasets.py:123-130,151,153,169`); `transform[0]` is training-only via
`__getitem__` (`:104-105`).

**Text.** Inside `model.compose_query(batch, texts, dataset_type)`
(`third_party/LIMN/LIMN/model.py:149-195`); the checkpoint's own tokenizer is used, no adapter-side
text preprocessing. `dataset_type` is supplied explicitly: `{"dress":0,"shirt":1,"toptee":2}`
(`model_limn.py:512`), matching upstream `dataset_name_dict` (`datasets.py:21-24`).

**Scoping.** `category_scopes = ("dress","shirt","toptee")`, one whole-model checkpoint per scope
(`model_limn.py:134`). Scoring needs a `Mapping` of scope → full-gallery matrix and raises
`ExactRankUnavailable` rather than guessing when it is not given one
(`model_limn.py:370-375`). The 119 ambiguous-category queries are excluded by
`_scope_for_query` and land in `scope_errors` — coverage loss, not a retrieval miss, as the spec
requires. Observed: the stored LIMN run reports `cohort 5884` of 6003 (98.0% coverage).

**Pooling / scoring.** Query and gallery features are L2-normalized 768-d; score is a normalized dot
product chunked at 8192 (`model_limn.py:381-398`); rank re-implements the same tie rule as
`_ranks_from_scores` (`model_limn.py:400-416`). `rank_and_topk` is NOT overridden for LIMN, so
`CompositionAdapter`'s generic implementation applies — which calls `self.score` with the mapping, so
it works.

---

## 3. Cross-model differences (the high-value table)

| Axis | `habit` | `intent` | `hint` | `tme` | `conesep` | `csmcir` | `ptha_mtst` | `encoder` | `limn` |
|---|---|---|---|---|---|---|---|---|---|
| resize geometry | TargetPad 1.25 → 224 | same | same | same (`utility`) | same | same | same | shortest-edge + CenterCrop | shortest-edge + CenterCrop |
| interpolation | BICUBIC | BICUBIC | BICUBIC | BICUBIC | BICUBIC | BICUBIC | BICUBIC | BICUBIC (antialias) | BICUBIC (antialias) |
| aspect handling | pad | pad | pad | pad | pad | pad | pad | crop | crop |
| normalize mean/std | CLIP | CLIP | CLIP | CLIP | CLIP | CLIP | CLIP | CLIP | CLIP (from tower) |
| query vs gallery transform | same | same | same | same | same | same | same | same | same |
| eval augmentation | none | none | none | none | none | none | none | none | none |
| caption normalization | `blip_caption` (lower, strip punct, 50 words) | same | same | adapter builds ids | own | own | own | open_clip BPE | checkpoint tokenizer |
| tokenizer | BERT, in-model | BERT, in-model | BERT, in-model | in adapter | in model | in model | in model | open_clip (77, no mask) | in model |
| max text len | 32 | 32 | 32 | `network.max_txt_len` | own | own | own | 77 | own |
| attention mask used for pooling | masked attention, fixed slot 32 | masked, slot 32 | masked, slot 32 | own | own | own | own | **no pad mask; mean over 6 fused tokens** (pads bias selection) | own |
| feature width | 256 | 256 | 256 | 256 | 8192 | 8192 | 32×256 | 512 | 768 |
| target layout | `(G,32,256)` | `(G,32,256)` | `(G,32,256)` | `(G,32,256)` | `(M,d,k)` `mdk` | `(M,k,d)` `mkd` | `(M,k,d)` `mkd` | `(G,512)` | `(G,768)` |
| reduction | max over 32 | max over 32 | max over 32 | max over 32 | max over last | max over last | max over last | dot | dot |
| L2 before dot | in-model | in-model | in-model | in-model | identity | explicit | identity | both sides | both sides |
| checkpoint load | whole/trusted | trusted | trusted | `strict=False` | trusted | `strict=False` | **`strict=True`** | `strict=False` + assert | per-scope trusted |
| reference masked | no | no | no | no | no | no | no | no | no |
| rank indexing | 1-based | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |

Three real disagreements worth flagging:

1. **`encoder` is the only model whose pooling ignores padding** (no attention mask exists to ignore
   it with), so its `image_only_empty_text` condition is a mean over `[SOT][EOT]` + 75 pads. Every
   other model either masks padding or pools a fixed non-pad slot. Cross-model comparisons of the
   image-only condition are therefore not like-for-like, and the spec's "ordinary tokenizer special
   tokens may remain" does not cover the *pad-mean* part of this.
2. **`conesep` vs `csmcir`/`ptha_mtst` disagree on target axis order** (`mdk` vs `mkd`) while all three
   end with `.max(axis=-1)`. Each is self-consistent, but this is the kind of asymmetry that turns a
   transposed tensor into a plausible-looking score.
3. **`strict=False` in five of nine adapters** (`tme`, `csmcir`, `encoder`, plus HABIT/INTENT/HINT
   whole-module paths that bypass key matching entirely). Only `ptha_mtst` fails loudly on a key
   mismatch.

---

## 4. Per-model detail that was later verified (was UNRESOLVED)

The three reading agents returned after the first draft. The following replaced inference with
observation:

**`intent`** — `targetpad_transform(1.25,224)` at `model_intent.py:325`, defined
`third_party/INTENT/data_utils.py:99-106`. One transform, one `decode()` closure for both phases
(`:343-352`). Caption processor from `blip2_pretrain.yaml` → `BlipCaptionProcessor(prompt="",
max_words=50)`. Tokenizer inside the model: `blip2_cir.py:744-751`,
`padding="max_length", truncation=True, max_length=32`. **Attention mask IS used**:
`attention_mask = cat([query_atts, text_tokens.attention_mask])` → Q-Former BERT
(`blip2_cir.py:753-763`). Pooling is the **fixed index 32** (`:767-769`) — after the 32 query embeds
that is the first text token (`[CLS]`); pad positions are masked out of attention. Checkpoint is a
whole pickled module with a mandatory SHA-256 (`model_intent.py:316-317`;
`common.py:45-56` refuses a missing digest and verifies bytes before `weights_only=False`).

**`hint`, `tme`, `conesep`, `csmcir`, `ptha_mtst`** — same shared substrate: the same
`BlipCaptionProcessor` class from each checkout's own copy (`HINT/.../blip_processors.py:28-68`,
byte-equivalent), the same `BertTokenizer` + `[DEC]` registration (`HINT/.../blip2.py:31-34`;
TME/MTST identical), and `max_txt_len=32` default in every class with no shipped config overriding
it. Note on `[DEC]`: it is *registered* as bos but a single-sequence encode emits only
`[CLS] …words… [SEP]`, so the effective special-token pattern is the ordinary BERT pair. That
confirms the empty-text finding in `FINDINGS.md` D3 for this whole family, not just `habit`.

**`tme`** — transform from its own `utility.targetpad_transform` (`model_tme.py:353,368`);
`load_state_dict(strict=False)` (`:362`).

**`conesep`/`csmcir`/`ptha_mtst`** — target layouts `(M,d,k)` vs `(M,k,d)` as in §1.4; only
`ptha_mtst` uses `strict=True` (`model_ptha_mtst.py:315`).
