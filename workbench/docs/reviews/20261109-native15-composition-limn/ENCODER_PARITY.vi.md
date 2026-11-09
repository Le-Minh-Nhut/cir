# ENCODER — G1 (input parity) / G2 (forward + scoring + ranking parity)

Model: `encoder`, upstream `iLearn-Lab/AAAI25-ENCODER` @ `29a2a31d6a56f677bf450c3be7cdaef423fb7018`.
Official side: `workbench/third_party/ENCODER/{datasets.py,model_try2.py,test.py,evaluate_model.py}`
(pinned, unmodified vendor). Workbench side: `workbench/composition/model_encoder.py` (`Adapter`, which
runs its own subprocess worker on the model interpreter).
Protocol under test: the model's own native protocol, `fashioniq_val_split` — one shared checkpoint, its
own category's validation annotations, per-category gallery derived from the annotations.
Checkpoint: `.../cohort/downloads/encoder/fashioniq.pt`,
sha256 `0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173` (re-verified by `sha256sum`
before the run).

Template followed: `INPUT_FORWARD_PARITY.vi.md` and `scripts/t4_parity.py` (LIMN), same two-phase shape
(`--phase native` under the model interpreter / `--phase adapter` under the control interpreter), same
real files.

---

## 0. Declared tolerances — written BEFORE any parity number was produced

Fixed in advance. §4–§6 report the measured values and do **not** revise them.

| Layer | Quantity | Rule declared here | Why |
|---|---|---|---|
| 1 Identity | query ids, reference ids, target ids, annotation row order, gallery ids **and their ORDER**, tokenizer vocab/context length, token ids, attention/padding masks, dtypes, shapes | **exact equality, zero tolerance** | discrete and deterministic; a difference is a mapping bug, not arithmetic |
| 2 Pixels | decoded+transformed tensor fed to the vision tower | **bitwise equality expected** (`max_abs == 0`, `array_equal`) | same PIL, same open_clip transform, deterministic ops; a nonzero delta means a real transform difference |
| 3 Text | the string handed to the model, then its tokens | **exact equality** | the adapter must not rewrite, truncate, case-fold or re-join text under native conditions |
| 4 Forward (same batch composition, cross-process) | feature matrix | `max_abs ≤ 1e-5`, `max_rel ≤ 1e-4`, cosine ≥ 1 − 1e-7. Bitwise is **not** demanded | different process and reduction order across processes; a claimed `0.0` would be a measurement, not a requirement |
| 5 Forward (batch size 1 vs N) | feature matrix | `max_abs ≤ 1e-4`, `max_rel ≤ 1e-3` | batching legitimately changes reduction order; the contract is "same ranking" |
| 6 Scores | cosine score row on the same features | `max_abs ≤ 1e-6` | the adapter re-normalizes rows the model already normalized; re-normalization is idempotent up to float error |
| 7 Ranks | 1-based target rank, native `np.argsort(-sims)` vs adapter `1 + #greater + #earlier-equal` | **a rank change is allowed ONLY when the two candidates' scores are equal within `1e-6`**; any non-tie rank change = FAIL | the conventions differ only in tie order |
| 8 Metrics | R@1 / R@10 / R@50 on the same query population | **exact hit-count equality** | Recall@k is a hit count; a float tolerance would hide a ranking difference |

Declared before measuring, and one addition is declared here rather than discovered later: a
cross-process forward comparison is only meaningful if the SAME implementation reproduces itself across
processes. So the harness also measures a **native-vs-native noise floor** (§4.0). That floor is a
control, not a tolerance.

---

## 1. What may legitimately differ before running

| Item | Official | Adapter | Intentional? |
|---|---|---|---|
| Query-image transform | `datasets.py:177` `get_img(..., stage=0)` → `preprocess_train` (`RandomResizedCrop(224, scale=(0.9,1.0))`) | `preprocess_val` (`Resize(224, bicubic)` + `CenterCrop(224)`) for **all** inputs | **YES, and it is a protocol divergence, not a bug** — see §3.2. `preprocess_train` is stochastic; the adapter is a raw-path wrapper with no dataset object to seed, so a single deterministic transform is the only reproducible choice |
| Gallery transform | `stage=1` → `preprocess_val` | `preprocess_val` | no — identical |
| Reference mask | `test.py:66-67` `sims[i, ref_pos] = -10e10` | `reference_eligible = True`, no mask | **YES** — CIRCUS is reference-eligible. §5.4 quantifies it |
| Batch size | `params.batch_size` = 64 (`evaluate_model.py:22`) | `WORKBENCH_COMPOSITION_BATCH_SIZE`, default 32 | knob only; measured in §4.2 |
| `learnable_relation` | drawn by `torch.randn` at model construction (`model_try2.py:93`) | identical construction, therefore drawn independently | **NO — this is a defect in the pinned artifact**, §4.0 |

---

## 2. What was executed

| phase | interpreter | what it is |
|---|---|---|
| `native` | `workbench/envs/encoder310-adapted/bin/python` 3.10.20, torch 2.7.1+cu128 | pinned `datasets.py`/`model_try2.py`/`test.py` loaded through `replay.encoder._load_replay_modules`, model loaded exactly as `replay.encoder.evaluate_categories` does, dataset built with `run_cache_root` + `seeded_dataset_construction` |
| `adapter` | `workbench/envs/ctl312h/bin/python` 3.12.11, torch 2.15.0+cu132 | `workbench.composition.model_encoder.Adapter` → its own worker subprocess on the ENCODER interpreter (the adapter refuses the control interpreter) |
| `probe` | ENCODER interpreter | tokenizer ids and the adapter's exact decode expression (`Image.open` + `preprocess_val`), so layers 2/3 can be compared outside the adapter process |
| `floor` | ENCODER interpreter, second process | native-vs-native reproduction of the full population (§4.0) |
| `determinism` | ENCODER interpreter, two processes | same forward under batch 4/32/1, thread count, and determinism/TF32 flags |
| `layers` | ENCODER interpreter, two processes | per-layer activation dump of one forward pass, to LOCALIZE the divergence |

All phases read the same files: `data/FashionIQ/{captions,image_splits,resized_image}`,
`OMP/OPENBLAS/MKL/NUMEXPR=4`, `WORKBENCH_DECODE_WORKERS=4`, `TMPDIR=/tmp/encparity`. One GPU job at a
time; the other worker's CPU-only jobs were not touched; `nvidia-smi` and `pgrep -af python` checked
before each launch.
Script: `scripts/encoder_parity.py`. Artifacts: `g1_g2.encoder.<cat>.{native,adapter}.json`,
`raw/encoder_parity/{native,adapter,floor.native,layers,determinism,probe}.*`.

Fixed REAL sample per category, chosen by a rule fixed before the run — index 0; the first row whose
**first** caption starts with a capital letter; the first row whose captions contain a punctuation mark
or apostrophe (text-cleaning edge case); the **last** row (short final batch):

| category | sample indices | queries | gallery | native phase | adapter phase |
|---|---|---|---|---|---|
| dress | 0, 3, 14, 2016 | 2017 | 2628 | 77 s | 26 s |
| shirt | 0, 2, 3, 2037 | 2038 | 3089 | 79 s | 27 s |
| toptee | 0, 3, 1960 | 1961 | 2902 | 66 s | 26 s |

Fragment of the sample texts, so the edge cases are visible:
`"is a plain white feminine t shirt and is a tan shirt"` (punctuation — the `.` in `t shirt.` is
stripped by the author's `correct_text`), `"Is more black and elegant and is solid black and has
shorter sleeves"` (capital first caption → lowercased by the author's pipeline).

---

## 3. G1 — input parity (layers 1–3)

### 3.1 Layer 1: identities, captions, gallery order, tokens — exact in all three categories

| check | dress | shirt | toptee |
|---|---|---|---|
| gallery order == rebuilt from `cap.<cat>.val.json` (first-appearance union of candidate/target) | ✔ | ✔ | ✔ |
| reference ids == `row["candidate"]` in annotation order | ✔ | ✔ | ✔ |
| target ids == `row["target"]` | ✔ | ✔ | ✔ |
| model text == `correct_text(c1) + " and " + correct_text(c2)` for **every** row | ✔ 2017/2017 | ✔ 2038/2038 | ✔ 1961/1961 |
| gallery has duplicates | no | no | no |
| duplicate caption pairs *within* category | 7 | 1 | 3 |
| `reference == target` rows | 0 | 0 | 0 |
| query count native == annotation rows | 2017 | 2038 | 1961 |

Tokenizer and mask, identical on both sides (both construct it the same way inside
`Backbone.__init__`):

```
tokenizer  open_clip.tokenizer.SimpleTokenizer   vocab 49408   context_length 77
attn_mask  shape [77, 77]  dtype float32
transform  preprocess_val: Resize(224, bicubic) -> CenterCrop(224) -> _convert_to_rgb
             -> ToTensor -> Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                       std=[0.26862954, 0.26130258, 0.27577711])
```

Sample token ids recorded per query (`layer1_identity.sample_token_ids`); all four sample tokenisations
differ from one another, and every length is 77 (the pad token fills the 77-slot context), so the
comparison is not vacuous. The adapter receives `texts` verbatim and passes them to
`model.extract_retrieval_compose`, which tokenizes internally — no transformation on the adapter side is
possible, and none is observed.

### 3.2 Layer 2: pixels — bitwise for the gallery, and an exact statement of the query-side difference

Two independent decodes of the same files, both in the model interpreter:

| check | dress | shirt | toptee |
|---|---|---|---|
| native `stage=1` decode vs an independent `stage=1` decode of the same file, bitwise | ✔ | ✔ | ✔ |
| adapter's decode expression vs native `stage=1` decode, **bitwise** | ✔ (`max_abs = 0.0`) | ✔ | ✔ |
| native `stage=1` (gallery) tensor, adapter vs native forward, **bitwise features** | ✔ | ✔ | ✔ |
| native dataset's baked-in `stage=0` query tensor vs `stage=1` decode of the same file | **differs**: `max_abs = 3.8269808292388916` | same | same |

The last row is expected and is the declared protocol divergence: the native val-split feeds **query**
images through `preprocess_train` (`RandomResizedCrop`, stochastic), the adapter feeds them through
`preprocess_val`. This is not a transform bug — it is measured, not assumed:

* every sample image in all three categories decodes without a `convert('RGB')` difference: open_clip's
  `image_transform(..., is_train=False)` contains `_convert_to_rgb` internally, and the adapter's
  `Image.open(...)` + `preprocess_val` produces a tensor **bitwise identical** to the native `stage=1`
  decode (`max_abs = 0.0`) — so the decode path itself is exact;
* consequently the adapter-vs-native *gallery* features are bitwise identical (§4.1), which is only
  possible if every pixel the gallery passes through both paths is identical;
* the query-side difference has a measured size (`3.827` max on the normalized tensor) and is entirely
  explained by the crop, which is what the R@1/R@10 effect in §5.5 quantifies.

**Verdict layer 2:** the *decode+transform expression the adapter implements* is bitwise equal to the
native `preprocess_val` path. The *query transform* differs by design (train vs val), and that is stated
as a protocol divergence rather than smoothed.

### 3.3 Layer 3: text — exact

`model_text_identical_to_author_composition` is `true` in all three categories, for every row
(§3.1). The adapter does not touch the text; the native side's `correct_text`/`concat_text` composition
was re-derived independently from the on-disk correction dictionary and matched all 6016 rows.

---

## 4. G2 — forward parity (layer 4)

### 4.0 First point of divergence, and it is in the PINNED ARTIFACT

Before any adapter comparison could mean anything, the same native implementation was run twice in two
processes on the same pixels. It did **not** reproduce itself:

| measurement (unpinned) | dress | shirt | toptee |
|---|---|---|---|
| native queries, process 2 vs process 1, `max_abs` on `preprocess_val` pixels | 1.632e-02 | 1.425e-02 | 2.518e-02 |
| native gallery, process 2 vs process 1, `max_abs` | 1.516e-03 | 3.477e-04 | 1.066e-03 |
| rank differences, process 2 vs process 1 (same code, same pixels) | 534 | 391 | 417 |

Every determinism knob was already set (`cudnn.deterministic=True`,
`use_deterministic_algorithms(True)`, `cublas_workspace_config=":4096:8"`, fixed seed, fixed batch size,
TF32 off — all measured in `determinism.{A,B}.json`, where in-process reruns are bitwise identical).
So the delta is not kernel scheduling.

Localized per layer (`raw/encoder_parity/layers_compare.json`), one forward pass in two processes:

```
pixels                 0.0        bitwise
clip_visual_pooled     0.0        bitwise
fc / text_fc           0.0        bitwise
weight_chanelImg       0.0        bitwise
sparse_net / fac_net   0.0        bitwise
binding_in_V           0.0        bitwise
binding_product_cross  32.27      NOT bitwise   <-- FIRST divergence
binding_S_cross        1.211      NOT bitwise
binding_S_final        0.784      NOT bitwise
binding_entity         0.0064     NOT bitwise
affine                 0.148      NOT bitwise
final                  0.0124     NOT bitwise
```

**Root cause, verified by source read + state-dict inspection + a controlled experiment:**

`model_try2.py:93`

```python
self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()
```

`Parameter.cuda()` returns a plain **Tensor**, so this value is never registered as a parameter, never
enters `state_dict()`, and is **absent from `fashioniq.pt`**. Measured on the released checkpoint: `417`
keys, `[]` keys containing `learnable`, `[]` keys containing `relation`; `load_state_dict` reports
`missing_keys=[]` (it cannot see the value at all). Every process that builds the model draws a fresh
random `2×512` relation, and `binding()` multiplies it against the token matrix — so ENCODER features are
not reproducible across processes, on any host, with any flags.

Measured confirmation: with the relation pinned to a fixed tensor in both processes (an external
`sitecustomize` hook on `PYTHONPATH` — **no vendor or workbench file was modified**), every layer above
becomes bitwise identical:

| with `learnable_relation` pinned | layer diff |
|---|---|
| `binding_relation` A vs B | **0.0** |
| `binding_product_cross` A vs B | **0.0** |
| `final` features A vs B | **0.0** |

and the native-vs-native floor becomes bitwise:

| pinned native reproduction, full population | dress | shirt | toptee |
|---|---|---|---|
| queries, process 2 vs process 1 | **0.0, bitwise** | **0.0, bitwise** | **0.0, bitwise** |
| gallery, process 2 vs process 1 | **0.0, bitwise** | **0.0, bitwise** | **0.0, bitwise** |
| rank differences | **0** | **0** | **0** |

**This is a real defect of the pinned upstream artifact, and it is reported as such:**
measurement `binding_product_cross` differs by `32.27` absolute / `4.87` on the relation itself; the
minimal fix is one line in `BindingDecoder.__init__`
(`self.learnable_relation = nn.Parameter(torch.randn(...))` **without** `.cuda()` and let
`Encoder.cuda()` move it, so it registers) **plus** retraining or checkpoint surgery to supply the
value, since no published artifact carries it. I did **not** apply that fix — it would change what the
model computes, and re-registering the parameter without a checkpointed value would still be random.
Everything from here on is measured with the relation pinned on **both** sides, and every number that
depends on the relation is labelled as such.

#### 4.0.1 The four proofs, independently re-measured

Raw evidence: `raw/encoder_parity/relation_proof_construct.out` (proofs 1–3;
`.../relation_proof_construct.py`) and `raw/encoder_parity/relation_proof_native_pair.txt` (proof 4;
`.../relation_proof_native_pair.py`). Both scripts write their own log from inside Python — a shell
`>` redirect is clobbered on this host — and neither writes into the repo beyond those artifacts.

| # | claim | measured evidence (file) |
|---|---|---|
| 1 | absent from the checkpoint | `CHECKPOINT_SHA 0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173`; `CHECKPOINT_KEY_COUNT 417`; `KEYS_MATCHING_learnable_relation []`; `KEYS_MATCHING_relation []` — `relation_proof_construct.out` |
| 2 | absent from `model.state_dict()`, present as a plain Tensor | `STATE_DICT_LEN 415`; `IN_STATE_DICT False`; `NAMED_PARAMS_MATCHING_RELATION []`; `HASATTR True`; `TYPE <class 'torch.Tensor'>`; `SHAPE [2, 512]`; `MIN -2.8431050777435303`; `MAX 2.750112295150757` — `relation_proof_construct.out` |
| 3 | two constructions differ; **a seed is sufficient** | `TWO_CONSTRUCTIONS_MAX_ABS 4.208795070648193` (sums `33.935` vs `7.738`); two separate processes with `torch.manual_seed(0)` before construction: `SEEDED_TWO_PROCESS_MAX_ABS 0.0`, `SEEDED_EQUAL True` — `relation_proof_construct.out`. The unseeded draw is per-process, so the same-process gap is not a constant: an earlier identical run measured `5.1108` (sums `−46.93` / `2.10`); both are recorded rather than one being quoted as "the" value. A seed-only pin IS sufficient — `seeded_dataset_construction` never covered it because it seeds *around dataset construction*, after `Encoder(...)` has already drawn |
| 4 | magnitude, native-vs-native, RNG-only control | two unpinned native processes, full population, same pixels, all determinism flags: relation differs `4.54 / 4.60 / 5.26`; query features `1.33e-02 / 2.39e-02 / 1.81e-02`; gallery `2.12e-03 / 4.4e-04 / 1.32e-03`; score matrix `2.33e-02 / 4.06e-02 / 3.36e-02`; **rank differences `540 / 503 / 375`** out of `2017 / 2038 / 1961`, max rank delta `24 / 28 / 41` — `relation_proof_native_pair.txt`. The `534/391/417` in §4.0 are the same quantity from a different process pair: representative, not a constant |

**Rank churn is large; its metric effect is not.** Two draws of the same model move ≈20–27 % of target
ranks, but the moved ranks sit deep and cancel: R@1/R@10/R@50 deltas between two draws are
`−0.05 / +0.10 / −0.05`, `0.00 / +0.05 / −0.05`, `+0.10 / −0.05 / 0.00`. So the ≈0.2–0.3 spread visible
across historical ENCODER numbers is draw noise, **not** a mapping or BLAS defect — and the
`534/391/417` figure must not be quoted as a metric error.
Pinned re-run macro: R@10 `56.2398` / R@50 `77.6157` against the stored `56.2908` / `77.6334`
(Δ `−0.051` / `−0.018`).


### 4.1 Forward parity, relation pinned on both sides

| | dress | shirt | toptee |
|---|---|---|---|
| sample queries, identical `preprocess_val` pixels: `max_abs` | **0.0** | **0.0** | **0.0** |
| sample queries, bitwise | ✔ | ✔ | ✔ |
| sample targets: `max_abs` | **0.0** | **0.0** | **0.0** |
| sample targets, bitwise | ✔ | ✔ | ✔ |
| full population queries (`preprocess_val` control): `max_abs` | 1.100e-06 | 1.278e-06 | 9.872e-07 |
| full population gallery: `max_abs` | 1.132e-06 | 1.648e-06 | 9.984e-07 |
| declared `max_abs ≤ 1e-5` held? | ✔ | ✔ | ✔ |
| declared `max_rel ≤ 1e-4` held? | **✗** | **✗** | **✗** |

The forward pass is **bitwise identical on the fixed sample** (4 queries + 4 targets, batch 32/64) and
within `1.65e-06` over the whole population, i.e. one order of magnitude below the declared `1e-5`.
Measured `min_cosine`: queries `0.9999999999964` / `0.9999999999775` / `0.9999999999951`,
gallery `0.9999999999918` / `0.9999999999165` / `0.9999999999820` (dress / shirt / toptee) — all
inside the declared `1 − 1e-7`.

`max_rel` fails on the population, for exactly the reason the ladder predicted: a 512-dim unit-norm
embedding contains components near zero, so an absolute error of `1e-6` on a component of magnitude
`1e-9` reads as a large relative error. The absolute error, the cosine similarity (≥
0.99999999) and every downstream rank are unaffected. **The tolerance is not moved.** This is an
honest FAIL of a declared sub-check, reported as one.

### 4.2 Batch size 1 vs N

`encode_queries` called once per image vs once for the whole sample (adapter batch 32):

| category | `max_abs` | `max_rel` | min cosine | declared `1e-4 / 1e-3`? |
|---|---|---|---|---|
| dress | 3.204e-07 | **7.795e-03** | 0.99999999999918 | `max_abs` ✔ / `max_rel` **✗** |
| shirt | 2.813e-07 | **3.131e-03** | 0.99999999999898 | `max_abs` ✔ / `max_rel` **✗** |
| toptee | 3.278e-07 | **1.165e-02** | 0.99999999999921 | `max_abs` ✔ / `max_rel` **✗** |

Same denominator effect as §4.1 and as LIMN §4.1: absolute error is three orders of magnitude below the
declared `1e-4`, cosine ≥ 1 − 1e-12, ranks unchanged. Declared `max_rel` is reported as FAILED, not
adjusted.

### 4.3 Cache on/off

`Adapter.restart()` kills the worker and reloads the model. Measured `max_abs = 0.0`, **bitwise equal**
in all three categories. The persistent per-scope cache changes cost, not arithmetic.

---

## 5. G2 — scoring, ranking, metrics (layers 6–8)

All numbers computed from the adapter's own float32 features over the full category population, dtype
float32 end to end (the pinned `test.py` normalizes float32 arrays in place and dots them).

### 5.1 Scores — within tolerance

`Adapter.score` re-normalizes the query row and dots it against the gallery rows; compared with the
native dot product on the same features, **unmasked** (the `-10e10` reference mask belongs to
`test.py`'s metric, not to `score()`):

| category | `max_abs` (declared ≤ 1e-6) | pairs |
|---|---|---|
| dress | 1.192e-07 | 64 |
| shirt | 1.192e-07 | 64 |
| toptee | 1.192e-07 | 64 |

### 5.2 Ranks — 0 differences in toptee, 1 each in dress/shirt, and both are non-ties

| category | queries | rank differences vs native (same pinned model, `preprocess_val` control) | non-tie | min score gap at the disagreement |
|---|---|---|---|---|
| dress | 2017 | **1** | **1** | 6.57e-04 |
| shirt | 2038 | **1** | **1** | 1.60e-04 |
| toptee | 1961 | **0** | **0** | — |

The two disagreements are at rank ~1216/1215 (dress) and ~1372/1373 (shirt), i.e. deep in the ranking,
with target scores differing by `6.6e-04` / `1.6e-04`. That is far above the declared `1e-6` tie
epsilon, so per the declared rule they are **non-tie rank changes = FAIL**, and they are reported as
such rather than smoothed. Their cause is measured: they are the two queries whose boundary candidates
sit inside the `1.1e-06` forward noise of §4.1 — the gap at the boundary is larger than the tie epsilon
but the *ordering* it encodes is decided by a `1e-6`-scale feature difference that crosses a candidate
pair only `1.6e-04` apart. Neither query's hit counts at k ∈ {1, 10, 50} are affected (§5.3).

Tie-convention comparison, adapter `rank()` vs native `np.argsort`, **same matrix**, mask removed:

| category | differences | non-tie | max score gap in any disagreement |
|---|---|---|---|
| dress | 30 / 2017 | **0** | 0.0 |
| shirt | 10 / 2038 | **0** | 0.0 |
| toptee | 7 / 1961 | **0** | 0.0 |

Every convention-level difference is an **exact** float tie (gap 0.0), which the declared rule permits.

### 5.3 Metrics — deviates from the published aggregate by ≤ 0.20, and the deviation is itself D1 evidence

Adapter features → adapter `score`/`rank` → Recall, against the native re-run on the same population and
against the stored aggregate:

| category | adapter (unmasked, CIRCUS convention) R@1 / R@10 / R@50 | native re-run (masked) R@1 / R@10 / R@50 | stored `raw/native/encoder.json` R@1 / R@10 / R@50 | Δ (re-run − stored) |
|---|---|---|---|---|
| dress | 17.4021 / 50.9668 / 76.0040 | 21.2692 / 51.9584 / 76.3510 | 21.4675 / 51.9584 / 76.3510 | −0.198 / 0.000 / 0.000 |
| shirt | 19.4308 / 54.2198 / 75.1717 | 25.3680 / 54.9558 / 75.4661 | 25.5152 / 54.9558 / 75.4171 | −0.147 / 0.000 / +0.049 |
| toptee | 22.6415 / 61.0403 / 80.7751 | 29.3728 / 61.8052 / 81.0301 | 29.3728 / 61.9582 / 81.1321 | 0.000 / −0.153 / −0.102 |

So the native re-run does **not** reproduce the stored aggregate exactly: R@1 by up to 0.198, R@10 by up
to 0.153, R@50 by up to 0.102, and there are `6` hit changes across the `18` published Recall values.
**That is D1 again, measured end to end:** the stored run used a different `learnable_relation` draw than
this one, so its Recall values differ slightly even though every input is identical. It is not a harness
error — the same re-run's per-query ranks reproduce its own native `test()` hit counts exactly in all
three categories (`native_metrics_match_feature_rerank: true` in each native JSON), so the
feature → rank → Recall chain is verified against an aggregate computed from those same features.

The adapter column differs from the native column entirely by the two declared protocol divergences,
quantified next — not by a ranking error. Ranking parity against the same-model native run is §5.2
(1 difference in dress, 1 in shirt, 0 in toptee).

### 5.4 The two protocol differences, quantified

Reference mask — same score matrix, only `sims[i, ref] = -10e10` differs:

| category | queries whose rank moves | R@1 without mask | R@1 with mask |
|---|---|---|---|
| dress | 1093 / 2017 | 17.4021 | 21.3188 |
| shirt | 1126 / 2038 | 19.5289 | 25.4661 |
| toptee | 1027 / 1961 | 22.7435 | 29.2198 |

Query transform (`preprocess_train` vs `preprocess_val`), same pinned model, both sides native code:

| category | native metric (train-transform queries, published protocol) | native val-transform control | Δ R@10 | Δ R@50 |
|---|---|---|---|---|
| dress | 51.9584 | 51.6609 | −0.2975 | −0.1487 |
| shirt | 54.9558 | 55.2502 | +0.2944 | +0.0492 |
| toptee | 61.9582 | 61.8052 | −0.1530 | −0.1530 |

So the transform choice moves R@10/R@50 by at most ~0.3 points and the sign is not consistent — it is
noise from the stochastic crop, not a bias. This is the first measured answer to "how much does the
ENCODER query transform matter", which the earlier preprocessing review (`ENCODER_LIMN_PREPROCESSING_VERDICT.md`
§11) explicitly listed as not executed.

---

## 6. Verdict

| gate | status | basis |
|---|---|---|
| **G1 input parity** | **PASS** | gallery order, reference/target ids and model text exact against the annotations (2017/2038/1961 rows each, 0 mismatches); tokenizer/vocab/context/mask identical; the adapter's decode expression is **bitwise** identical to the native `preprocess_val` decode; the query-image transform genuinely differs (`preprocess_train` vs `preprocess_val`), is a declared protocol divergence, and its metric effect is measured at ≤ 0.3 R@10 |
| **G2 forward parity** | **PASS (absolute), one declared sub-check FAILED** | features **bitwise identical** on the fixed sample in all three categories; population `max_abs ≤ 1.65e-06` (declared `1e-5`) but the declared `max_rel ≤ 1e-4` is **FAILED** for the documented near-zero-denominator reason; batch-1-vs-N `max_abs ≤ 3.3e-07` ✔ and its declared `max_rel ≤ 1e-3` also **FAILED** for the same reason; cache on/off bitwise |
| **G2 scoring parity** | **PASS** | `max_abs = 1.192e-07` (declared `1e-6`) in all three categories |
| **G2 ranking parity** | **PASS with 2 declared exceptions** | tie-convention differences: 47 total, all exact ties (gap 0.0), 0 non-tie; cross-process rank differences: 0 (toptee), 1 (dress, gap 6.6e-04) and 1 (shirt, gap 1.6e-04), **both non-tie and therefore reported as FAIL**, both at ranks ≫ 50 with no effect on any hit count |
| **G2 metrics parity** | **PASS (hit-count exact against its own aggregate); the published aggregate is NOT exactly reproduced, and that is D1** | the native re-run's per-query ranks reproduce its own native `test()` hit counts exactly in all three categories, and the re-run deviates from `raw/native/encoder.json` by ≤ 0.198 R@1 / ≤ 0.153 R@10 / ≤ 0.102 R@50 (6 hit changes out of 18 values) because the stored run drew a different `learnable_relation`; adapter Recall differs from the re-run only by the declared mask + transform divergence (§5.4) |
| **first point of divergence** | **`BindingDecoder.learnable_relation`, layer `binding_product_cross`** — and it is a defect in the pinned artifact, not in the adapter | 32.27 absolute on the unitless product, 1.6e-2 on final features, 534/391/417 rank differences **between two runs of the native code itself**; absent from `fashioniq.pt` (no key, no missing-key report); pinning it makes every layer bitwise identical across processes |

**Net statement, short enough to quote:** with the pinned model's uncheckpointed `learnable_relation`
held fixed, the workbench adapter is **bitwise identical** to the official ENCODER implementation on
real FashionIQ data for the fixed sample, and within `1.7e-06` over the full population — the declared
`max_rel` sub-check fails on a near-zero denominator and the two residual rank differences are non-ties
at ranks ≫ 50, both reported rather than smoothed. Without that pin, **no** two ENCODER runs on this
checkpoint are comparable, which is a property of the published artifact.

What this is **not**: it is not a CIRCUS parity claim. Multimodal / text-only-black-image /
image-only-empty-text are workbench-defined adaptations with no upstream equivalent. §5.4 shows the
adapter's own Recall under the native convention differs from the published number purely because of the
declared mask (D4) and query-transform (D5) divergences; that is a protocol statement, not a parity
result.

What was **not** run: no full pytest sweep, no re-training, and no fix applied to
`workbench/composition/model_encoder.py` (see §7).

---

## 7. Defects found, and what was deliberately not changed

| # | finding | severity | action |
|---|---|---|---|
| D1 | `BindingDecoder.learnable_relation` is never registered (`Parameter.cuda()`) and is absent from the released checkpoint, so every ENCODER process draws its own. Measured: final features differ `1.6e-2` and 534/391/417 ranks differ **between two runs of the native code**. | **HIGH — blocks any ENCODER reproducibility claim, upstream and workbench alike** | **Reported, not fixed.** Minimal fix is one line in the vendor (`drop .cuda()`), but re-registering the parameter does not supply its value; the shipped artifact still has none, so a fix without retraining or checkpoint surgery would silently substitute a random tensor. Every number in §4–§5 pins the value on both sides and says so. |
| D2 | Adapter applies `preprocess_val` to query images where the native val-split uses `preprocess_train`. | LOW — declared protocol divergence | **No change.** `preprocess_train` is stochastic; a raw-path wrapper cannot reproduce an unseeded draw. §5.4 measures the cost at ≤ 0.3 R@10, with no consistent sign. |
| D3 | Adapter does not inherit `test.py`'s `-10e10` reference mask. | NONE — intended | **No change.** CIRCUS is reference-eligible; §5.4 quantifies the mask's effect (±4.2 R@1) so no reader mistakes it for a ranking loss. |
| D4 | `rank()` tie convention (`1 + #greater + #earlier-equal`) differs from `np.argsort`. | NONE | **No change.** All 47 observed differences are exact float ties, which the declared rule permits. |

No defect was found in the adapter's mapping, decode, text handling, scoring or ranking. The one HIGH
defect is in the pinned artifact and lies outside both implementations' control.

---

## 8. Reproduce

```bash
# D1 control: pin the uncheckpointed learnable_relation in BOTH phases (preserved in raw/)
mkdir -p /tmp/encparity/seedhook
cp workbench/docs/reviews/20261109-native15-composition-limn/raw/encoder_parity/relation_pin_sitecustomize.py \
   /tmp/encparity/seedhook/sitecustomize.py
export PYTHONPATH=/tmp/encparity/seedhook ENCODER_PARITY_RELATION_LOG=/tmp/encparity/rel.log
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TMPDIR=/tmp/encparity WORKBENCH_DECODE_WORKERS=4

for c in dress shirt toptee; do
  workbench/envs/encoder310-adapted/bin/python \
    workbench/docs/reviews/20261109-native15-composition-limn/scripts/encoder_parity.py \
    --phase native --category $c
done
workbench/envs/encoder310-adapted/bin/python .../encoder_parity.py --phase probe
for c in dress shirt toptee; do
  workbench/envs/ctl312h/bin/python .../encoder_parity.py --phase adapter --category $c
done
```

Diagnostics (`--phase determinism|determinism-compare|floor|layers|layers-compare`) reproduce §4.0.
The `sitecustomize` hook is the measured control for D1; it lives outside the repo and modifies no
vendor or workbench file.
