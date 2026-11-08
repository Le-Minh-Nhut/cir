# FINDINGS — input→output pipeline audit

Evidence dir: `workbench/docs/reviews/20261108T082857-audit-pipeline/`.
Branch `research/cir-failure-workbench`. Nothing committed, nothing pushed, no source file edited.

Scope: BLIP-2 family (`habit`, `intent`, `hint`, `tme`, `conesep`, `csmcir`, `ptha_mtst`) and
open_clip family (`encoder`, `limn`). GPU was busy with a cohort evaluation
(`pgrep composition_eval` → pid 264541, `nvidia-smi` → 3654 MiB) for the whole audit, so every GPU
forward is `NOT_RUN`; all numbers below come from CPU execution on real data or from source.

Status vocabulary: `CONFIRMED_DEFECT`, `INTENTIONAL_ADAPTATION`, `SUSPECTED`, `UNRESOLVED`.

---

# 1. Highest-risk findings

## F1 — `CONFIRMED_DEFECT` (severity: HIGH, disclosure-grade) — the `--seed` flag is inert and the run identity binds an RNG that is never applied

`workbench/scripts/composition_eval.py:53`:

```python
parser.add_argument("--seed", type=int, default=0, help="seed recorded for deterministic summaries")
```

`seed` is threaded into `run_composition` and hashed into `identity` (`runner.py:944`, `"seed": seed`),
so it mints a distinct `run_id`. But **nothing in the composition path applies it.** Verified:

```
$ grep -n "manual_seed\|np.random.seed\|random.seed\|deterministic" \
    workbench/composition/runner.py workbench/composition/adapters.py workbench/scripts/composition_eval.py
workbench/composition/runner.py:244:  #: Per-query failure reasons ... deterministically.   (comment only)
workbench/scripts/composition_eval.py:53:  (the flag itself)
```

Zero seeding calls. Consequences:

- Changing `--seed` produces a *new* `run_id` and a *byte-identical* result. The identity claims a
  controlled variable that is not controlled. That is the worst kind of provenance: it looks like a
  reproducibility parameter and is decorative.
- The reverse also holds: **the run id does not bind the RNG state that actually produced the
  features.** For the BLIP-2 family, which has no stochastic eval transform, this is harmless today.
  For anything that does draw randomness at encode time it is a silent-staleness hole. See F2 for the
  one worker where randomness is real.

Affected: all nine adapters (the field is shared). Fix: either seed properly at worker launch or drop
the field from `identity` and rename the CLI flag to `--seed-recorded`. Patch:
`patches/0001-seed-is-inert.patch`.

## F2 — `INTENTIONAL_ADAPTATION` (verified, reported not patched) — ENCODER composition deliberately uses the deterministic transform where upstream evaluation is stochastic

Settled on disk in `ENCODER_LIMN_PREPROCESSING_VERDICT.md` (source lines + CPU execution). **No code
change is warranted or proposed.** Summary of the settled rows:

| | reference image | gallery images |
|---|---|---|
| upstream `fashioniq_val_split` (`datasets.py:177` vs `:195`) | `preprocess_train` (RandomResizedCrop, **stochastic**) | `preprocess_val` |
| workbench replay `replay/encoder.py:291-310` | `preprocess_train`, seeded via `seeded_dataset_construction(0)` | `preprocess_val` |
| workbench composition `model_encoder.py:321,340` | `preprocess_val` | `preprocess_val` |

Measured on real released images: mean |train − val| = 0.16 / 0.50 / 0.21 in normalized-tensor units
(max 3.83), and repeated unseeded draws differ by up to 3.83 — the upstream transform is genuinely
stochastic.

Why the adapter's choice is the correct one and must not change: upstream's reference path is
unseeded and non-reproducible, so no wrapper can reproduce an upstream run from it at all; and this
protocol ranks over the released 74,381-image gallery where the reference image **is also a gallery
row**, so a stochastic reference encoding would encode the same physical image two different ways and
break the reference-eligible rank.

Consequence for the published numbers: **none.** `recall_at_10 = 31.02%` is a
`circus_fashioniq_release_v1` number, produced with one deterministic transform for all inputs, and is
internally consistent with its own declared policy. It is not an upstream reproduction and the summary
already carries `reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED`. No recomputation needed.

Residual, and the only thing worth recording: the composition summary does not *state* that it uses
`preprocess_val` for query inputs where the replay protocol uses `preprocess_train`. That is a
report-note gap, carried in `UPSTREAM_PARITY.md` "intentional adaptations", not a defect.

## F3 — `CONFIRMED_DEFECT` (severity: MEDIUM) — a resumed run's gallery feature cache is accepted without any shape check

`runner.py:407` versus `runner.py:456`:

```python
# _encode_gallery
cached = _load_memmap(destination, fingerprint)              # no shape argument
# _encode_queries
cached = _load_memmap(destination, fingerprint, (len(queries), dimension))   # shape checked
```

`_load_memmap` only enforces shape when given one (`runner.py:333-343`). Executed demonstration
(`/tmp/stalecache.py` → `logs/stale_cache_demo.txt`): a 100-row gallery cache written under a valid
fingerprint is reused by the gallery path with no complaint, while the same file fails a query-style
`(200, 4)` check, and a different fingerprint is correctly refused.

What this does and does not mean. It is **not** silent staleness across model/checkpoint/preprocessing
changes — those change `fingerprint`, and the fingerprint mismatch is refused. It *is* a hole for a
truncated or partially-overwritten `.npy` that still has its sidecar `.json` (e.g. an interrupted
write, a filesystem short write, or a manually copied cache). The metadata records `shape`, so the
fix is one line: pass `metadata["shape"]`. Patch: `patches/0003-gallery-cache-shape-check.patch`.

---

# 2. The seven explicit high-risk items (task section D)

## D1 — Does the feature cache validate everything that affects the cached representation?

**Verdict: `CONFIRMED_DEFECT` (partial).** The run identity is unusually thorough — it binds gallery
order, per-file content digests of every resolved image, query text/reference/positives/category/
subsets, the checkpoint content hash, model provenance (upstream commit, approved patches, vendored
source digest), the adapter module hash, five adapter method source hashes, and the `adapters.py` and
`inputs.py` module hashes (`runner.py:910-948`).

Inputs that can change a feature or a rank and are **absent** from `identity`/`fingerprint` (and
therefore from every cache sidecar):

| # | Missing input | Why it matters | Verified |
|---|---|---|---|
| a | Controller numerics: `numpy`/BLAS version, `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | controller-side `score()` and `_ranks_from_scores` are numpy (`adapters.py:325-368`); BLAS threading changes reduction order | read |
| b | `WORKBENCH_DECODE_WORKERS` and the torch thread pin (`limit_torch_threads`) | changes the decode pool and intra-op threads in every worker | read (`adapters.py:102-164`) |
| c | Pretrained base weights outside the staged checkpoint (lavis/HF cache for `tme`/`csmcir`; ENCODER's `open_clip_pytorch_model.bin`) | `strict=False` loads can silently take base weights from a cache | read |
| d | Adapter-imported replay modules (`workbench/replay/{tme,limn,common}.py`) | code that shapes the forward is unhashed | read |
| e | `WORKBENCH_RUNTIME_ADAPTATION` | recorded in the summary but not in `identity`; selects dependency sets and autocast behaviour | read |
| f | GPU identity beyond `get_device_name(0)` | no driver/CUDA/cuDNN version, no device index; same-named GPUs collide | read (`runner.py:557-564`) |
| g | `topk.py` source (tie policy) | a tie-rule change reuses an old export under the same run/attempt key | read |
| h | YAML annotations feeding category scope (`image_splits/*.val.json`, `captions/cap.*.test.json`) | their influence reaches identity only via the resulting `category`; a scope-fix is invisible unless the assignment itself changes | read |
| i | `seed` *is* in identity but is never applied | opposite failure: binds a variable that does nothing (F1) | executed |

The `preprocessing` slice is a **fixed seven-key** projection of the registry
(`runner.py:918-920`: `fashioniq_layout`, `fashioniq_image_extensions`, `transform`, `image_size`,
`text_processor`, `candidate_normalization`, `scoring`). For `encoder` and `limn` all seven are
`null` — verified from the registry — so for the open_clip family the preprocessing identity is
carried only by module hashes, not by any declared transform. That is adequate in practice (the
transform is in the hashed module) but it means the summary cannot tell a reader *what* transform ran.

## D2 — Are padded tokens excluded from pooling?

**Verdict: `CONFIRMED_DEFECT` for the BLIP-2/HABIT pooling position; `NOT_APPLICABLE` elsewhere.**

- HABIT (`third_party/HABIT/.../HABIT.py:372-413`) fuses with
  `padding="max_length", truncation=True, max_length=32` and then pools **a fixed slot**:
  `fusion_output.last_hidden_state[:, 32, :]`. The 32 is a Q-Former *query-token* index, not a text
  position, so it is position-invariant to padding — correct. Text padding is masked for attention
  (`attention_mask = cat([query_atts, text_tokens.attention_mask])`), so padded text tokens do not
  contribute. **Empty text does not break this** (see D3).
- ENCODER (`model_try2.py:380-432`): `self.tokenizer(txt)` is `open_clip.get_tokenizer('ViT-B-32')`,
  which yields a fixed 77-length padded id matrix with **no attention mask** passed to the text tower
  (`model_try2.py:385` `txt_token = self.tokenizer(txt).cuda()`; the tower applies only CLIP's causal
  `attn_mask`, `:343`). Executed: `''` → `[49406, 49407, 0×75]`.

  **Correction (a first draft of this file overstated this).** Pooling is **not** a mean over 77
  positions. `extract_retrieval_compose` means over `fuse_local`, whose token axis is
  `img_select_tokens_con` / `cap_select_tokens_con` — `cat([global(1), action(N_p=2),
  selected(local_token_num=3)])` = **6 tokens** (`model_try2.py:412,428`, adapter ctor
  `local_token_num=3` at `model_encoder.py:309`). Verified by shape propagation through
  `FeatureWiseAffine` (`:435-454`: `chanel = x.shape[1]*2` = 12, gamma/beta reshape to `(B,6,512)`).

  The pads still matter, indirectly and materially: the 77-token context feeds `text_local_fea`
  (`:386-388`), which drives the attention scores used to select which tokens survive —
  `ScoreNet(sparse_ratio=0.6)` keeps `ceil(77*0.6)=47` of 77 (`:42-50`), then `FactorNet` softmaxes
  over those 47 to emit 3 tokens (`:77-89`). For empty text all 75 pad positions share one embedding
  id (positional embeddings still differ), so pads are strong candidates to fill the 47 selected
  slots. `INFERENCE`/`STATIC_ONLY`: not measured, because it needs a model forward. What is measured
  is the tokenization above. Conclusion stands unchanged in direction and weaker in degree: ENCODER's
  `image_only_empty_text` is a specific `[SOT][EOT]`+pad-driven query, **not** "no text", and it is
  not comparable to a masked/pad-excluded condition in the other families. Not a workbench defect.
- LIMN: text goes through the checkpoint's own tokenizer inside `compose_query`; no adapter-side
  masking to audit, and the spec's "ordinary tokenizer special tokens may remain" covers it.

## D3 — `image_only_empty_text`: does an empty string produce the sequence the upstream ablation intends?

**Verdict: `INTENTIONAL_ADAPTATION`, verified by execution.**

The chain is: `inputs.adapt_query_inputs` returns `""` (`inputs.py:26`) → the BLIP-2 path lowercases/
punctuation-strips it to `""` via `BlipCaptionProcessor.pre_caption` → BERT tokenizes to
`[CLS] [SEP]` + pads.

Executed (`logs/empty_text_tokens.txt`, `logs/encoder_empty_text_tokens.txt`):

```
BlipCaptionProcessor: 'IS LONG AND HAS A FLORAL PRINT' -> 'is long and has a floral print'
                      ''  -> ''
                      '   ' -> ''
BERT (max_length=32, padding=max_length): '' -> ids [101, 102, 0,0,...], non-pad count 2, decoded '[CLS] [SEP]'
open_clip ViT-B-32 tokenizer:            '' -> [49406, 49407, 0×75], pad_count 75
```

So `""` yields the ordinary `[CLS][SEP]` pair — **exactly** "remove edit text before tokenization;
ordinary tokenizer special tokens may remain" (`CIRCUS_COMPOSITION_SPEC.md`, Conditions table). It
does **not** produce a padding-only sequence and does **not** produce a different special-token
pattern. The spec is satisfied.

Caveat worth recording: `condition_scaled__` — there is no whitespace-preserving path, so
`"   "` and `""` are indistinguishable by design (the processor strips). If a future ablation wants
"empty but not absent" text it cannot express it through this processor.

## D4 — `text_only_black_image`: is the black image applied to the RAW image before the transform?

**Verdict: `INTENTIONAL_ADAPTATION`, correct, verified by execution.**

`runner.py::_stage_query_path` opens the file, `copy()`s the raw PIL image, calls
`adapt_query_inputs`, and for `text_only_black_image` **saves the adapted PIL image to disk** and
returns that path; the adapter worker then opens and transforms it:

```python
def _stage_query_path(path, query, condition, work_root):
    with Image.open(path) as source:
        image = source.copy()
    adapted, text = adapt_query_inputs(image, query.query_txt, condition)
    if condition == "text_only_black_image":
        staged = work_root / "queries" / condition / f"{_component(query.qid)}.png"
        ...
        adapted.save(staged, format="PNG")
        return staged, text
    return path, text
```

`inputs.black_image_like` returns `Image.new(image.mode, image.size, 0)`, i.e. **same size, same
mode, raw zeros, before any transform**. Executed: a 342×444 black RGB image through ENCODER's
`preprocess_val` yields a constant tensor with unique values
`[-1.7923, -1.7521, -1.4802]` (mean −1.6749, std 0.1386) — i.e. the normalized-black color, **not**
zeros. A zero-after-normalization or zeroed-feature implementation would show mean 0. So the
implementation matches the spec's `_black_image_same_size` requirement and is distinguishable by
measurement from the wrong intervention.

Also checked: 4000 random released images are `{'RGB': 3998, 'L': 2}`; for the 2 non-RGB files,
`black_image_like` + `convert("RGB")` is still exactly `(0,0,0)`, so no mode bug.

## D5 — Is the reference in the gallery, and does any adapter mask it?

**Verdict: `INTENTIONAL_ADAPTATION`, verified (and the `!=` check is real).**

- Gallery is the released ordered list of 74,381 identities; the reference is a member of it (the
  spec requires reference-eligible ranking, and deliberately does not inherit upstream's `-10e10`
  mask).
- `reference_eligible` is a **class literal `True`** in all nine adapters, but the runner does not
  trust it: it calls `adapter.measured_reference_eligibility()`, which runs the adapter's own
  `score`+`rank` on a synthetic gallery where the reference is the exact best match and reports the
  measured rank (`adapters.py:424-470`, called at `runner.py:884-890`).
- Executed evidence from the real HABIT run's identity: `reference_eligibility = {"measured": true,
  "reference_eligible": true, "reference_rank": 1, ...}`.
- Upstream ENCODER/HABIT **do** mask the reference (`-10e10` at `test.py:64`/`:68` for ENCODER,
  `test.py:151` for HABIT); the spec explicitly declines to inherit it, so this divergence is
  intentional and documented.
- No adapter masks the reference row. LIMN declares `reference_eligible = True` but its
  `score()` requires a `Mapping` of scope matrices, so the probe reports `unmeasurable` rather than a
  false verdict — honest, and the real LIMN summaries show `reference_eligibility: None`.

## D6 — Tie policy: are exact ties systematic?

**Verdict: `CONFIRMED_DEFECT` in the *data*, not in the code. Tie handling is correct; the gallery has genuine duplicates.**

`_ranks_from_scores` counts `1 + #greater + #earlier-equal` (`adapters.py:352-368`). Measured on the
real LIMN dress gallery matrix (`logs/duplicate_embeddings.json`, `logs/dup_groups_identity.txt`,
`logs/dup_pixel_check.txt`):

- **851 duplicate groups covering 2,386 rows; 1,535 excess rows** (out of 74,381). Largest group: 54.
- The duplicates are **byte-identical source files under different product IDs** — verified by
  `sha256` of the `.png` files (`logs/dup_pixel_check.txt`: `file_sha_equal True` for 8/8 groups
  checked; pixel correlation 1.0, mean-abs-diff 0.0). This is a dataset property, not a resolver bug:
  an independent resolution check shows 74,381 distinct identities → 74,381 distinct files, 0
  collisions.
- Rank stability under those ties was tested directly
  (`logs/tie_rank_stability.json`): 25 duplicate groups used as queries, positive placed at the
  *last* duplicate so an exact tie must be broken by position. **0 rank flips**, and the batch-1 vs
  batch-32 score delta of 5.4e-7 did not move any rank. A 200-query random sample also gave **0 rank
  diffs** (max |Δscore| = 2.4e-7).

So: the tie rule is correct and stable, but any query whose positive is one of a 54-strong
byte-identical duplicate cluster is, in effect, scored against a repeated gallery item. That
**inflates** recall for those queries and is a property of the released data the spec pins. It is not
fixable without changing the pinned gallery — it should be *reported* as a caveat, and it is
currently not mentioned anywhere in the summaries.

## D7 — Final-batch handling in `_encode_*`: can the last batch be dropped, duplicated, or misaligned?

**Verdict: `INTENTIONAL_ADAPTATION` (correct), verified by reading and by the A1/A2 measurements.**

- `_encode_gallery`: `range(0, len, batch_size)`; remainder written at `mapped[start:start+len(paths)]`;
  completeness asserted by `position != len(image_paths)` (`runner.py:415-434`). No empty batch, no
  drop, no duplicate.
- `_encode_queries`: matrix pre-filled with **NaN** (`features[:] = np.nan`, `:473`), so **every query
  keeps its row**; failures stay NaN and are later flagged `"query features unavailable"`
  (`:1121-1126`). Writes are `features[surviving] = encoded` with an exact-shape guard (`:496-501`);
  the single-item fallback writes `features[index]` per index (`:513-520`). Misalignment is
  structurally impossible because the write index is the original row index, not a running counter.
- Rank loop: `len(exact_ranks) != len(valid_indices)` is a hard error (`:1157-1158`), so an adapter
  returning the wrong count cannot silently shift ids.
- Memmap row ⇔ query identity is the position in the `queries` tuple, **not** `Query.index`
  (`:1115-1121`), and both encode and rank use the same list — consistent.

The one residual is F3 (no shape check on gallery cache reuse), which is a cache-validity issue, not
a batching issue.

---

# 3. Differential axes (task section C)

Full data: `logs/differential_stages.json`, `logs/tie_rank_stability.json`. Tolerances were declared
in the script **before** the run and were not relaxed:

```
repeat determinism     : atol 0.0,   rtol 0.0     (same call, same process ⇒ bit-equal)
score chunked (1 vs 32): atol 1e-6,  rtol 0.0
discrete outputs       : EXACT equality (ranks, positions, candidate order, ids)
```

| Axis | Result | First divergent stage |
|---|---|---|
| A1 batch 1 vs 32 vs all (score stage) | batch-32 **bit-identical** to all-at-once; batch-1 differs by **max 6.56e-7** (< declared 1e-6) | none — divergence is inside the declared float tolerance at the **score** stage only |
| A1b rank effect of that float noise (200 real queries) | **0 rank differences**, max |Δscore| 2.4e-7 | none reaches the **rank** stage |
| A2 `_ranks_from_scores` vs brute-force stable argsort | 8/8 identical | none |
| A3 exact-duplicate tie rank (25 real duplicate groups, positive at last duplicate) | **0 flips**; rank identical under batch 1 and 32 | none |
| A4 repeated run determinism | **bit-identical** (`exact_equal: true`, max 0.0) | none |
| A5 metric aggregation vs dict ordering (real 6003-query HABIT ranks) | gap identical to the last bit (`0.388704727214254` both ways), cohort 6003 both ways | none |
| cold vs warm cache, cache enabled vs disabled, final incomplete batch on real data | `NOT_RUN` — see §5 | — |

Note on A5: `compose()` builds `qids` as a `set` (`metrics.py:104`) and iterates it to form the common
cohort (`:126`), so float summation order is hash-seed dependent in principle. On this data the
reversal made **no** difference to the last bit. The theoretical exposure is real but unmeasurable
here, so it stays `SUSPECTED` (see §4) rather than confirmed.

---

# 3b. Additional findings established by the three reading agents

These arrived after the first draft and are evidence-backed at `file:line`; none changes a verdict
above, two add real exposure.

## F4 — `CONFIRMED_DEFECT` (severity: MEDIUM) — ENCODER's adapter hard-codes the checkpoint mapping
status instead of reading the registry

`model_encoder.py:302-303` constructs the checkpoint record it passes to the loader with
`"checkpoint_mapping_status": "VERIFIED_METADATA"` as a **literal**. That literal is one of the four
gates in `_unused_checkpoint_keys` (`replay/encoder.py:73-102`), which decides whether the two
provably-unread `binding_decoder.fc.0.*` tensors may be skipped silently. So a registry downgrade of
that status would **not** tighten the adapter. The other three gates (registry digest `0f31c5…`,
clean source pin, the module not declaring those names) still apply, so the exposure is bounded: two
tensors shown not to be read (`R16_ENCODER_KEY_COMPAT.md` §3, `max_abs_diff 0.0` under perturbation).
Fix: read the status from the registry record rather than asserting it.

Related inconsistency worth recording: `registry/models.yaml` says
`checkpoint_mapping_status: VERIFIED_METADATA` while the same block says
`reproduction_status: METRICS_MEASURED_MAPPING_UNVERIFIED` and "direct file URL and SHA remain
unverified". Both labels are honest about different things; a reader who takes only the first will
over-trust the mapping.

## F5 — `SUSPECTED` (severity: LOW-MEDIUM) — ENCODER's auxiliary CLIP weights are presence-checked, never hash-checked, on the adapter path

`model_encoder.py:105-107` (`status()`) and the preflight path (`replay/encoder.py:137-145`) both only
check that `open_clip_pytorch_model.bin` exists. Its digest is pinned in
`registry/auxiliary_assets.yaml:296-297` (`1bd3c717…`, 605,219,813 B), and enforcement lives in the
runtime artifact validator (`backend/runtime.py:2916-2933`) — **not** in the adapter. A locally
replaced CLIP binary of the right shape would pass `status()` and load. Since that binary supplies the
frozen image/text towers, a swap silently changes every feature while `checkpoint_identity` (the
staged checkpoint hash) stays constant. This is exactly the class of input `FINDINGS.md` D1 lists as
missing from run identity — here it is missing from the *status gate* as well.

## F6 — `CONFIRMED_DEFECT` (severity: LOW, documentation) — a comment justifies tie-order reproduction by a false fact

`replay/limn_observer.py:162-169` states its tie order is reproduced because "argsort is stable in
NumPy". NumPy's default `argsort` is `'quicksort'` (introsort), explicitly **not** stable. The code
still reproduces the native ordering — it issues the *same call* on the *same inputs*
(`limn_observer.py:170`) — so the behaviour is correct and the **reason** is wrong. Risk: a future
change of `kind=`, NumPy version, or input size could break the reproduction while the comment
reassures. Fix: state that the same call is reissued, or pass `kind="stable"` on both sides.

## F7 — `INTENTIONAL_ADAPTATION` (quantified) — upstream's per-category galleries are 5-20× smaller

Measured lengths of `split.<cat>.val.json`: **3817 / 6346 / 5373** (dress/shirt/toptee) versus the
released 74,381. Upstream HABIT/ENCODER also report **three category numbers**, never a pool, and
mask the reference. So an upstream Recall and a composition Recall differ in gallery size, reference
policy, cohort pooling and metric set simultaneously — they are not comparable in either direction,
and no composition number should be read as reproducing them.

## F8 — `UNRESOLVED` — which `bert-base-uncased` vocab the BLIP-2 family actually resolves

Upstream hard-codes `/root/bert-base-uncased` (`blip2.py:31-34`). That path is absent/unreadable here
(`ls` → Permission denied; `exists False`), while a cached HF snapshot
(`models--bert-base-uncased/snapshots/86b5e093…`) is present. The `[DEC]` bos token is *registered*
but a single-sequence encode emits only `[CLS] … [SEP]`, so the special-token pattern is unaffected;
a different *vocab* would still change ids and therefore embeddings. Which resolution the model load
takes was not executed.

# 4. Suspected / unresolved

| ID | Status | Item |
|---|---|---|
| S1 | `SUSPECTED` | `compose()` sums over a `set`-derived cohort (`metrics.py:104,126`), so `gap_*`/`mrr_gap` could differ in the last bits across processes with different `PYTHONHASHSEED`. Not observed on the real 6003-query table (A5 exact). Fix if ever reproduced: sort the cohort. |
| S2 | `SUSPECTED` | `encoder` and `limn` registry rows leave all seven bound preprocessing keys `null` (`runner.py:918-920`), so the preprocessing identity rests entirely on module hashes. Adequate, but a summary reader cannot recover the transform from the summary. |
| S3 | `UNRESOLVED` | Whether upstream's stochastic reference transform materially shifts the **val-split Recall** (as opposed to producing different pixels, which is proven). Needs CUDA + the resized-image tree. |
| S4 | `RESOLVED` (was UNRESOLVED) | Cross-family consistency is now traced: all seven BLIP-2 adapters share `targetpad_transform(1.25,224)`, the same `BlipCaptionProcessor`, the same BERT tokenizer and `max_txt_len=32`, and all except `ptha_mtst` load with `strict=False`. See `PIPELINE_TRACE.md` §4. They still disagree on target axis order (`mdk` vs `mkd`) and feature width (256 vs 8192). |
| S5 | `SUSPECTED` | `csmcir` and `ptha_mtst` declare all three conditions supported but cannot run without inputs the released data does not carry (per spec: `cs mcir` needs per-image `COT_ours2` captions). Their `COMPLETED` status in stored artifacts should be re-checked against that claim; a "completed" row for an adapter that cannot reproduce the gallery would be a false result. |

---

# 5. What could NOT be executed, and why

| Item | Why | Marking |
|---|---|---|
| Any GPU forward (BLIP-2/open_clip, all conditions) | GPU busy for the whole audit: `pgrep -af composition_eval` → pid 264541; `nvidia-smi --query-compute-apps` → 3654 MiB in use. Per instructions, no second evaluator. | `NOT_RUN` |
| End-to-end cold vs warm cache, cache-disabled, and a real query count not divisible by 32 through `_encode_queries` | needs a worker/model launch (GPU) or an adapter; the CPU-only `CompositionAdapter` path has no cache to warm at the gallery scale | `NOT_RUN` |
| Upstream ENCODER / HABIT / LIMN end-to-end runs on the val-split | needs the `resized_image/<category>/<id>.jpg` tree and CUDA | `STATIC_ONLY` |
| Re-deriving the published `31.02%` / `0.3887` from scratch | needs a 74,381-image encode on GPU | `NOT_RUN` (read from stored artifacts instead) |
| Whether the EM/`csmcir` path can be executed at all | missing required inputs by the spec's own account | `STATIC_ONLY` |

---

# 6. Verified as justified (intentional adaptations)

| Item | Evidence |
|---|---|
| Reference is eligible in ranking; upstream's `-10e10` mask deliberately not inherited | `CIRCUS_COMPOSITION_SPEC.md` "Protocol and gallery"; measured probe returns rank 1 |
| Black image applied to the **raw** PIL image before the transform, same size/mode | `inputs.black_image_like`, `runner._stage_query_path`; executed: constant normalized tensor, not zeros |
| Empty text yields ordinary `[CLS][SEP]` special tokens only | executed for BERT (2 non-pad ids) and open_clip (SOT/EOT + pads) |
| `image_only`/`text_only` named by removed input, not by upstream file name | matches spec's explicit warning about misleading upstream filenames |
| Query text passed unchanged, never re-split on "and" | `inputs.adapt_query_inputs` returns `query.query_txt` verbatim; no splitting anywhere in the adapter path |
| nDCG is a fraction, Recall is percent | `metrics.py:73-76` encodes exactly this; gap is a fraction (`:148`) |
| Gap computed after averaging on a common cohort | `metrics.py:126-148` builds `common` first, then averages, then ratios |
| Unscored queries reduce coverage and never enter a mean | `metrics.py:59,72` |
| LIMN uses the validation transform for all three roles | matches upstream `datasets.py` `get_test_img` → `transform[1]` |
| ENCODER composition uses one deterministic transform for all inputs | correct for this protocol; the *undisclosed difference from replay* is F2 |

---

# 6b. Corrections made during this audit

Recorded because a claim that was published and then retracted is evidence about how much the rest
should be trusted.

| What I first wrote | What the evidence showed | Where corrected |
|---|---|---|
| "ENCODER's query representation is a mean over **77** fused positions **including 75 pads**", used to argue the image-only condition is pad-dominated | WRONG. The pooled axis is `cat([global(1), action(2), selected(3)])` = **6** tokens (`model_try2.py:412,428`); pads never enter that mean directly. Pads still influence *which* 3 tokens are selected (ScoreNet keeps 47 of 77, then FactorNet softmaxes to 3), so the substantive conclusion survives but the mechanism and magnitude were overstated. | `FINDINGS.md` D2, `PIPELINE_TRACE.md` §2.1, `UPSTREAM_PARITY.md` row 9 |
| Upstream HABIT "reports a single pooled Recall" | WRONG. It prints one number **per category** (`evaluate_model.py:72-75`) over a per-category gallery measured at 3817 / 6346 / 5373 — not the 74k pooled cohort. | `UPSTREAM_PARITY.md` rows 21, 27 |
| `intent`/`hint`/`tme` token+pooling internals `UNRESOLVED` | Now traced: same LAVIS substrate, attention mask **is** used, pooling is the fixed index 32 (`blip2_cir.py:753-769`). | `PIPELINE_TRACE.md` §4, `FINDINGS.md` S4 |

The pattern: three of my published claims were wrong in a way that made the pipeline look *worse* or
*less understood* than it is. Two were caught by delegating the reading; one by a shape-propagation
check. Everything in §1-§3 above was either executed or line-cited; where a claim rests on shape or
control-flow inference it now says so.

# 7. Patches (proposed, NOT applied)

| Patch | Addresses | Files touched |
|---|---|---|
| `patches/0001-seed-is-inert.patch` | F1 | `workbench/composition/runner.py`, `workbench/scripts/composition_eval.py` |
| `patches/0003-gallery-cache-shape-check.patch` | F3 | `workbench/composition/runner.py` |

No patch for F2 — that item is an intentional adaptation and is reported, not changed.
Both patches were verified with `git apply --check` and applied in sequence to a scratch copy, then
`ast.parse`d clean. Sequence 0001 then 0003: both touch `runner.py`.

# 8. Raw evidence index

```
logs/duplicate_embeddings.json          duplicate-group counts per scope
logs/dup_groups_identity.txt            group → gallery positions → identities
logs/dup_pixel_check.txt                source-file sha256 equality for duplicate groups
logs/differential_stages.json           axes A1..A5 with declared tolerances
logs/tie_rank_stability.json            25 duplicate groups, rank under batch 1 vs 32
logs/encoder_preprocess_train_vs_val.txt  resolved open_clip transforms
logs/encoder_transform_divergence.txt   measured train-vs-val pixel divergence
logs/empty_text_tokens.txt              BERT/BlipCaptionProcessor empty-text tokenization
logs/encoder_empty_text_tokens.txt      open_clip tokenizer on empty text
logs/stale_cache_demo.txt               gallery cache accepted without shape check
logs/limn_checkpoint_norm_probe.txt     no image_mean/std buffers in the checkpoint
logs/open_clip_default_norm.txt         open_clip default mean/std values
scripts/                                every script used, runnable
```
