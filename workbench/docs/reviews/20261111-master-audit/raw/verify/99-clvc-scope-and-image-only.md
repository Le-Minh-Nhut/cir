# 99 — CLVC-Net: the 119 unscoped queries, and a named image-encoder-only path

Branch `research/cir-failure-workbench`, 2026-11-12, owner of this file plus
`workbench/composition/model_clvc_net.py` and a new test file
`workbench/tests/test_clvc_scope_and_image_only.py`. Not touched: `inputs.py` CONDITION list,
`metrics.py` `_CONDITIONS`, `runner.py`, `backend/composition.py`, registry YAML, `model_limn.py`
(zero lines changed — the resolver is REUSED, not edited), `workbench.env`.

No GPU job. No composition run — not even a smoke (a smoke limits queries, not the 74 381 × 3
scopes). The live units on 8000/5173 were never touched; `/tmp/cir_gpu.lock` was never read or
deleted. Interpreter `workbench/envs/ctl312h/bin/python`, every test run with
`-p no:cacheprovider --basetemp=/tmp/<unique>`.

The two checkpoint-based tests run the REAL CLVC-Net pickles on CPU (`CUDA_VISIBLE_DEVICES=""`),
digest-verified through `load_trusted_whole_model`, ~1.3 GB peak RSS, ~4 s per test. They skip when
the bundle/annotations are absent, so a fresh clone still runs the logic layer.

---

## 1. Is the category contract the same for CLVC-Net and LIMN?

**Yes — and the reuse is the point, not a convenience.** Measured on the code, not inferred:

| property | LIMN (`model_limn.py`) | CLVC-Net (`model_clvc_net.py`) |
|---|---|---|
| scopes | `("dress","shirt","toptee")` | `("dress","shirt","toptee")` |
| a query is scored in | exactly one scope | exactly one scope (`_query_categories` refuses `None`) |
| gallery | one whole-release gallery per scope | one per-category gallery matrix per scope |
| category source | release loader, then the query's own FashionIQ caption annotation | same |
| refusal when the release does not determine one | unscored | unscored (`adapter_query_category_missing`) |

The two differ in *how* a scoped query is embedded (LIMN: one whole model per scope; CLVC: a
local/global pair per scope). That is the SCORING layer. The ROUTING contract — which queries get a
category at all, and where the category comes from — is identical, so CLVC-Net now calls
`model_limn.resolve_query_scope` / `caption_scope_index` / `normalize_query_text` /
`query_reference_stem` verbatim. **`model_limn.py` was not modified** (zero diff): no extraction was
needed because the four functions were already module-level and side-effect free. `model_limn.py`'s
own tests are green (§5), which is the regression check for the reuse.

A second implementation would have been free to drift, and drift here means the two models score
different query sets — the exact cross-model comparability the audit report §4.3 protects.

## 2. Dry run over all 6 003 queries

Written to `/tmp/clvc_scope_dryrun.json` (outside the repo, 6 003 rows). Per row: `qid`,
`loader_category`, `reference_stem`, `normalized_text`, `holding_categories`, the annotation file
that would supply a category, that file's digest, the resolved `scope` and the `reason`. Trap: the
path was passed as `workbench/composition/data.py:raw` once and `sed` reported
`No such file or directory`.

```
loader_scoped                        5884
resolver_agrees_with_loader          5884
resolver_disagrees_with_loader          0     <- a disagreement would be a bug, not a refinement
recovered_from_annotation             117     (shirt 46, toptee 71)
still_unscoped                          2
```

Annotation provenance (all three files usable, none off-pin; the index's `incomplete` is `false`):

| file | rows | sha256 |
|---|---|---|
| `captions/cap.dress.val.json` | 2017 | `5e5117d45695df9c3ca91fac3e0bc49e6422ae9c8b3793f0ac9b3ab83adb7de9` |
| `captions/cap.shirt.val.json` | 2038 | `7b7ca3797b85dddfd83e74cdb4454cb63e3e7c09acef2590155efbacea6d7feb` |
| `captions/cap.toptee.val.json` | 1961 | `b4e09f428e6c255cc4b4ec46e77908305ab83075d82840c960be81af15cdc5eb` |

Reason counts over 6 003: `loader` 5 884, `annotation:toptee` 71, `annotation:shirt` 46,
`unresolved:annotation_names_two_categories` 2.

Through the RUNNER's own seam (`runner._scope_for_query`, which is what a run calls):

```
runner seam routed: 6001   unscoped: 2   scope errors raised: 2
per scope: {'shirt': 2035, 'dress': 2010, 'toptee': 1956}
scope digest (with resolver)  3a1c5a6763154870536f3586491d5bea9d87b73a9836363dc298d2accdd34d87
scope digest (loader only)    8a5dd517595eb4883d8b3174f410f87980fbc612418231e2be4e0e3847cd1c19
```

Apart (the two digests differ) is what makes a re-run mint a new run id instead of silently reusing
`eab10106`'s all-null ranks for those 117 queries (`runner._scope_resolution_identity`, whose own
docstring names that trap).

## 3. The real count: 117 and 2 — the lead's hypothesis, falsified in part and confirmed in part

The hypothesis was "117 recoverable and 2 genuinely ambiguous". **Measured: 117 and 2.** Treated as
a hypothesis, not a target — the numbers were read off the resolver before this file was written,
and they agree with the counts `test_limn_scope_routing.py` already pinned for LIMN.

The two that stay unresolved, with the reason:

| qid | reference stem | joined caption text | holders | reason |
|---|---|---|---|---|
| `7:3038` | `B004TMGDME` | `is darker and has a different graphic and is darker` | shirt, toptee | `unresolved:annotation_names_two_categories` |
| `7:3790` | `B0079K0YBO` | `is grey and has a less whimsical graphic and is lighter` | shirt, toptee | `unresolved:annotation_names_two_categories` |

Both are the same caption PAIR filed under two category files with the same reference identity, so
the release names two categories for one `(reference identity, joined caption text)`. Per the
assignment rule they stay unscored. No category was taken from the target image, from a retrieval
score, or by combining scores across the three checkpoints — the resolver's only inputs are the
loader's assignment and the caption files.

Stem-only routing would have been WORSE, and this is measured rather than argued: of the 119, 93
references sit in exactly one split, **26 sit in both** — the stem alone would have silently picked
a category for 26 queries the annotation resolves by text (and for `7:3038`/`7:3790` the stem cannot
decide at all).

## 4. Modality leakage: verdict and its scope

**Verdict (a): a dataset label available independently of pixels — with one condition that must
state it out loud.**

* The routing input is the reference image's **stem** (a FashionIQ product id) resolved against the
  authors' `cap.*.val.json` / `split.*.val.json`. It is release metadata: the same value is
  obtainable from the annotation file with no image bytes at all. It is not derived from the
  reference's pixels, and it is not the target image, not a score, not a checkpoint comparison.
* Consequence that is easy to get wrong: under `text_only_black_image` the reference PIXELS are
  replaced by a black image, yet routing still consults the row's category. So that condition sees
  one piece of reference-derived metadata. Same for `clvc_image_encoder_only`, where the reference is
  the only input and the label is legitimate.
* Declared in code as `IMAGE_ENCODER_ONLY_LEAKS_SCOPE` (a string a test asserts on), and asserted
  structurally: `Adapter.query_scope` takes `(self, query, candidate_names)` — no condition, no
  gallery, no score — so the label cannot start depending on them without changing that signature.
* **The honest alternative, described and NOT shipped:** a run identity that declares the exemption
  explicitly — e.g. `protocol_id` keeping `circus_fashioniq_release_v1` but a summary field
  `reference_metadata_seen: {"text_only_black_image": ["category_label"]}` in the run identity, so
  the condition's claim ("the reference is removed") is narrowed in the artifact rather than
  contradicted by an unstated property. That needs `runner.py` (identity) and `backend/composition.py`
  (display) — outside my ownership. Patch spec in §7.
* What must not be claimed anywhere: "text_only_black_image sees no reference information". Verdict
  (a) permits the label; it does not make the claim true.

## 5. Tests, and the numbers behind them

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_clvc_scope_and_image_only.py \
  workbench/tests/test_clvc_net_composition_parity.py \
  workbench/tests/test_limn_scope_routing.py \
  workbench/tests/test_limn_annotation_provenance_r05.py \
  workbench/tests/test_limn_scope_runner.py \
  workbench/tests/test_circus_adapters_rest.py \
  workbench/tests/test_composition_adapters_missing.py \
  workbench/tests/test_circus_inputs.py \
  -p no:cacheprovider --basetemp=/tmp/ptA -q
149 passed, 2 warnings in 13.62s
```

The two warnings are pre-existing Pillow `getdata` deprecations in `test_circus_inputs.py`.

New file `workbench/tests/test_clvc_scope_and_image_only.py`, 15 tests:

| test | what it pins |
|---|---|
| `test_a_unique_annotation_row_resolves_to_that_category` | unique row → that category |
| `test_two_categories_for_one_caption_pair_stay_unscored` | both holders → `None`, holders reported |
| `test_a_missing_annotation_file_refuses_to_route` | `incomplete` honoured: absent file ≠ absent row |
| `test_a_corrupt_annotation_file_refuses_to_route` | unreadable file → `None` |
| `test_an_off_pin_annotation_file_refuses_to_route` | digest/row-count mismatch → `None` |
| `test_routing_does_not_depend_on_the_condition_or_the_gallery` | the leakage declaration is asserted, and the signature cannot silently grow a condition |
| `test_the_resolver_agrees_with_the_loader_on_every_scoped_query` | 5 884 / 5 884, 0 disagreements |
| `test_the_119_split_into_117_recovered_and_2_genuinely_ambiguous` | 5 884 + 117 + 2 = 6 003; the two qids named |
| `test_the_text_tower_is_never_called_on_the_image_only_path` | `compose_img_text` patched to RAISE on both halves |
| `test_the_edit_caption_is_inert_on_the_image_only_path` | two captions → identical tensors; control shows the same seam DOES carry text under `queries` |
| `test_the_image_only_row_equals_the_gallery_row_within_the_stated_tolerance` | measured, §6 |
| `test_the_image_only_row_is_scored_by_the_unmodified_scorer` | reference eligible (rank 1), unrelated row 3 |
| `test_scoring_rank_and_tie_break_are_the_multimodal_paths_own` | `rank`/`rank_and_topk` are the base implementations |
| `test_the_image_only_path_transports_no_text_at_all` | texts are REFUSED, not ignored |
| `test_image_only_empty_text_stays_declared_unsupported` | old condition untouched |

### Reproduction, on the pre-fix bytes

Same three facts, executed against `git show HEAD:workbench/composition/model_clvc_net.py`:

```
PREFIX has query_scope: False
PREFIX still_unscoped: 119 of 119
PREFIX has encode_image_only_queries: False
PREFIX module has IMAGE_ENCODER_ONLY: False
PREFIX worker op branches: False
```

After: the same three probes read `still_unscoped: 2` (the runner seam, §2), `encode_image_only_queries`
present, `IMAGE_ENCODER_ONLY` declared, and the worker has an `image_only` operation branch. So the
scope tests fail before the change (119 unscoped) and pass after (2), and the image-only tests cannot
even import the symbol before it.

## 6. The image-only path

### 6.1 What upstream actually provides

The paper (`https://xuemengsong.github.io/SIGIR_CTI-IR.pdf`, fetched, HTTP 200, 2 859 422 B,
`pdftotext -layout` → 117 911 chars) says, §4.2 "On Model Comparison (RQ1)":

> "**Image_Only** simply takes the reference image representation as the composed query
> representation."

and Table 1 reports it as a BASELINE: FashionIQ avg R@10 **5.15**, R@50 **13.58** (Dress 4.96/13.34,
Shirt 5.40/14.13, Tops&Tees 5.10/13.26).

CIRCUS §2.2 (`https://arxiv.org/html/2605.14787v2`, fetched) defines the image-only condition:

> "**Image-only**: the reference image only; the editing text is removed."

**The pinned source has no image-only CODE PATH.** `test.py` has exactly one query expression
(`:100`, `compose_img_text`) and one target expression (`:116`, `extract_img_feature`); the string
"image only"/"Image_Only"/"img_only" does not appear in any `.py` or `.md` under
`workbench/third_party/CLVCNet/`. So the paper's `Image_Only` baseline is a reported ablation with
no shipped entry point. What IS shipped and reusable is the target-row expression — the half of the
pipeline the ablation needs.

Note the numbers of the two sources are not comparable to a CIRCUS Full: the paper's 5.15 is
FashionIQ native (per-category split, filtered gallery); CIRCUS ranks over the whole 74 381-image
release gallery.

### 6.2 What was implemented

`clvc_image_encoder_only`, a NAMED condition, never a redefinition of `image_only_empty_text`:

* `Adapter.encode_image_only_queries(paths, categories)` → worker operation `"image_only"`.
* The row comes from `composed_rows(...)`, which for `targets` AND `image_only` calls ONE builder
  (`image_row_builder`): `cat(F.normalize(local.extract_img_feature(imgs)[1]),
  F.normalize(global.extract_img_feature(imgs)[1]))`, i.e. `test.py:116` verbatim — same transform
  (`Resize(256)/CenterCrop(224)/ToTensor/Normalize(ImageNet)`), same half order, same normalisation.
* Scoring is untouched: `Adapter.score` (delegating to the base cosine) and
  `Adapter.rank`/`rank_and_topk` are the multimodal path's own objects. Reference eligibility,
  gallery membership, positive ids and the tie-break are therefore the same code, not a copy.
* `image_only_empty_text` stays `ABLATION_UNSUPPORTED` with `EMPTY_TEXT_REASON`, and a test asserts
  the new name does not inherit or weaken that reason.
* A caller that hands the image-only path any text gets a `ValueError` (`transports no text`), so the
  condition cannot silently start using a caption.

### 6.3 Measurements on real weights (CPU, `dress` pair)

Tolerance stated BEFORE measuring: **`IMAGE_ONLY_TOLERANCE = 1e-6`** absolute on the 2 048-d row —
not bit identity, because the two rows come from different batch shapes (one image vs a batch).

| claim | measurement |
|---|---|
| image-only row vs the SAME image's gallery row | **6.705522537231445e-08** (≤ 1e-6) |
| batch-1 vs batch-3 gallery row (the source of that delta) | **6.146728992462158e-08** |
| repeat of the identical batch-1 call | **0.0** |
| two different captions, image-only | **0.0** — identical tensors |
| row norm (two unit halves) | **2.0000** (√2) |
| text tower on the image-only path | `compose_img_text` patched to raise on BOTH halves: still returns `(2, 2048)` |
| reference eligible | image-only row ranks its own gallery row **1**; an unrelated row ranks **3** |

The caption-inertness control matters: on the stub seam, `compose_img_text` receives the caption and
two different captions produce different rows, so "inert" is a property of the operation and not of a
function that drops its arguments. The REAL text tower cannot be used for that control on CPU: it
hardwires `.cuda()` (`text_model.py:108`, `first_hidden[0].cuda()`), which is the same structural
fact `EMPTY_TEXT_REASON` rests on.

### 6.4 Run identity a Full run would use

Under the new profile the run id changes for four independent reasons, all already in the identity
dict (`runner.py:1816-1900`) — no runner edit needed for them:

1. `adapter_module_source_hash` — `model_clvc_net.py` bytes changed.
2. `adapter_source_hashes` — `encode_queries`/`encode_targets`/`score`/`rank`/`mode_supported`
   unchanged in body, but `mode_support` now answers `True` for the new condition.
3. `mode_support` — a new condition entry.
4. `query_scope_resolution` — `3a1c5a67…` instead of `8a5dd517…`.

What a Full run would then record, and what is NOT yet true:

* With `--conditions multimodal,text_only_black_image`: 6 001 scored / 2 unscored per condition
  (coverage `0.99967`), instead of 5 884. **Not measured on GPU** — that is the requested run.
* With the image-only condition included: impossible without §7's plumbing, because `CONDITIONS` in
  `inputs.py` / `metrics._CONDITIONS` / `backend.composition.CONDITIONS` would reject the name
  before any model code ran.
* `composition_gap_ndcg` for CLVC becomes computable **only** when all three conditions carry ranks;
  with the new condition named but not plumbed, the gap stays `null` with the existing reason.

## 7. Patch spec outside my ownership (phase B)

**7.1 `workbench/composition/inputs.py`** — symbol `CONDITIONS`:

```
before: CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")
after:  CONDITIONS = CONDITION_NAMES  # where CONDITION_NAMES adds "clvc_image_encoder_only"
```
plus `CONDITION_LABELS["clvc_image_encoder_only"] = "Image only (image encoder)"` and an
`adapt_query_inputs` branch: for the new condition return `(image, "")` **and** the runner must call
`encode_image_only_queries` instead of `encode_queries`. Simplest correct shape: a
`CONDITION_ENCODERS = {condition: "encode_queries" | "encode_image_only_queries"}` map that
`runner._encode_queries` consults via `getattr(adapter, name, None)`, raising `Refused` when the
adapter has no such method — that keeps every other adapter's behaviour byte-identical (no
condition they declare maps to the new method).

**7.2 `workbench/composition/metrics.py`** — `_CONDITIONS` (line 11):
add the new name. **Do not** replace `image_only_empty_text`: `metrics.compose` is positional
(`_CONDITIONS[0..2]` are mm/text/image) and the gap formula compares mm against
`max(text, image)`. A 4-tuple breaks both. Correct change: keep `_CONDITIONS` as the three
protocol conditions and add a separate `EXTRA_CONDITIONS = ("clvc_image_encoder_only",)` that
`compose` computes metrics for but excludes from the gap. Alternative, cheaper and what I would
choose: **do not** add it to `metrics` at all in phase B; treat `clvc_image_encoder_only` as an
adapter-declared ablation that `metrics` ignores, and publish its ranks/Top-K only.

**7.3 `workbench/backend/composition.py`** — `CONDITIONS` (line 18) and the two fixture literals at
`:1497-1498` (`FIXTURE_DEVELOPMENT_ONLY`). Add the new name to the display tuple and give the
fixture builder a rank series, or the run-detail page will show the condition with no row. Also
`:1049-1050` reads `ranks.get("image_only_empty_text")` positionally — keep that key.

**7.4 `runner.py`** — only if 7.1's encoder map is adopted: `_encode_queries` currently calls
`adapter.encode_queries(paths, texts, categories)` unconditionally (`:794`, `:816`). Change to the
map-resolved method, passing `texts` only for `encode_queries`. Also `_stage_query_path` must return
the ORIGINAL path for the new condition (like `image_only_empty_text` does), which it already does by
falling through the `text_only_black_image` branch.

**7.5 registry YAML** — nothing required; the condition is adapter-declared. If the frontend needs a
label, `CONDITION_LABELS` in 7.1 is the single source.

**7.6 Leakage declaration (optional, recommended)** — in `runner.py`'s identity dict:
`"reference_metadata_seen": {"text_only_black_image": ["category_label"],
"clvc_image_encoder_only": ["category_label"]}`. This mints a new run id for every existing run
(none of them declared it), which is the price of the narrower claim; it was NOT shipped here because
it changes run identity for models that have nothing to do with CLVC.

## Còn chặn

1. **No Full run under the new scope resolver.** The 117 recovered queries have never been encoded or
   scored. Requested (GPU, needs the lock and the operator's go-ahead):
   `workbench/scripts/gpu_job.sh workbench/envs/clvc39-la/bin/python workbench/scripts/composition_eval.py --config /home/heheboiz/cir_rerun_20261109.yaml --models clvc_net --subsets full --topk-depth 100`.
   Expected effect, stated so it can be checked rather than believed: `unscored_count` 119 → 2 per
   active condition, coverage `0.98018` → `0.99967`, `missing_query_count` 238 → 4, and the metrics
   MOVE (they are recomputed over a 117-query larger cohort, so a metric that stays bit-identical
   would mean the queries were not scored).
2. **`clvc_image_encoder_only` cannot run until §7 lands.** It is declared supportable and its row
   expression is measured on CPU; it has no path through `runner.py`'s condition loop, `metrics`, or
   the backend display. No GPU number exists for it, and none is claimed.
3. **The two queries `7:3038` and `7:3790` are still unscored, by declared policy.** The release
   names two categories for them. Resolving them would need a rule that is not in the release (a
   target-image lookup, a score comparison, or a curation decision), and any of those breaks the
   "assign only from exactly one named category" rule this workstream is bound to.
4. **The GPU-side claim is unverified.** Every measurement in §6.3 was taken on CPU with
   `torch 2.15.0+cu132` in `ctl312h`. The worker path uses `.cuda()` and `torch 2.7.1+cu128` in
   `clvc39-la`; the expressions are identical, but "identical expression" is a code fact, not a
   measured GPU parity.
5. **`TextLSTMModel.forward`'s `.cuda()` hardwiring is still the reason the multimodal caption
   control could not run on real weights** (`text_model.py:108`). It is the same mechanism as
   `EMPTY_TEXT_REASON`; not fixed here, out of scope, and fixing it would change the model's bytes.
