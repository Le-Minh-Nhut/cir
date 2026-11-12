# Third-model parity: HINT, and the shared capture/compare interface

Date: 2026-11-12. Model: `hint` (`checkpoint_id: fashioniq`, sha256
`bf9c72d42206b667ed586758c15c648970df821857e4067f26155a114a6517d4`, 2 724 451 182 B — recomputed
here, not read from the registry).
Producer: `scripts/third_model_parity.py`. Intermediates: `raw/third_model/`.

**State: HINT is now a THIRD model with PAIRED G1/G2 evidence, beyond ENCODER and LIMN.** All six
artifacts exist, written by real capture on 2026-11-12 (native 170 s, adapter 3× ~105 s, under
`gpu_job.sh`'s flock). `GET /api/runs/parity-gate` discovers and judges it on its own files.

Verdict: `PARITY_MANDATORY_SUBCHECK_FAILED` with **2** failures, both `batch_1_vs_n.max_rel`
(shirt `5.05e-03`, toptee `4.21e-02`) against the unchanged declared `batch_max_rel 1e-3`. Every other
mandatory dimension is measured and holds — including **bitwise-identical** sample forwards and
**zero real rank differences**. The two failures are genuine, reported, and not repaired: §6a explains
what they are and why the tolerance was not moved after seeing them.

## 1. HINT vs LIMN: construction differences that break a copy of `t4_parity.py`

A third clone of `t4_parity.py` would have been wrong in five places. Every row is read off the pinned
source, and the LIMN column is what `t4_parity.py` assumes.

| # | aspect | LIMN (`t4_parity.py` assumption) | HINT (measured) | why a copy is wrong |
|---|---|---|---|---|
| 1 | **gallery scope** | per category: `datasets.py` builds `test_queries_<name>`/`test_targets_<name>` for the one category it was constructed with (`LIMN/datasets.py:136-169`, one `self.name` per instance) | **category-agnostic**: one checkpoint, and `datasets.py:39-41` builds all three categories' test sets at construction; `model_hint.py:229-235` **raises** on a non-`None` target category (`encode_targets`, `:235`) | the adapter call in `t4_parity.py:313-318` (`[category] * n`) is refused by HINT. The `categories` argument must be `[None] * n`, and `score()` is a single einsum over `(G,Q,D)`, not a per-scope `Mapping` |
| 2 | **target feature shape** | `(N, 768)` — one vector per image (`model_limn.py`; `t4_parity.py` asserts nothing) | `(N, 32, 256)` — BLIP-2 Q-former token layout. Native `HINT.py:436-455` returns `(N,256,32)`; the adapter **transposes** to `(N,32,256)` (`model_hint.py:366`) | comparing without the transpose is a shape error at best; the adapter also hard-asserts its own shape (`model_hint.py:198-211`, `expected` at `:202-206`), so a wrong expectation fails loudly |
| 3 | **score / rank API** | `score`/`rank` take a `Mapping` of per-scope galleries (`adapters.py:368-406`, used at `t4_parity.py:398-406`) | `score` takes a bare **`(G,Q,D)` matrix** and does `einsum("nd,gqd->ngq").max(-1)` (`model_hint.py:238-250`); `rank` inherits the base 2-D path (`adapters.py:435-443`) | passing `{category: gallery}` to HINT's `score` raises; `rank` must be given the 2-D gallery and `[[position]]` positives |
| 4 | **query encoding** | `compose_query(batch, texts, kinds)` — a **category id tensor** conditions the forward (`t4_parity.py:246-249`, `compose_query(batch, texts, kinds)`) | `extract_retrieval_compose(img, mod)` — **no category id at all**; text is tokenised INSIDE the model with `max_txt_len=32` truncation (`HINT.py:401-433`) | there is no comparable per-token id list to record. `t4_parity.py:166-171` records `sample_tokens`; for HINT the comparable quantity is the query **feature**, and a token-list field would have been a fabricated column |
| 5 | **preprocessing path** | adapter/native differ by transform (`preprocess_train` vs `preprocess_val`), which `t4_parity.py` records as `native_tensor` and the gate deliberately excludes | both sides take the **same val** transform, `targetpad_transform(1.25, 224)`: native at `replay/hint.py:184` (the pinned `test.py` never builds one; `replay/hint.py:184` does), adapter at `model_hint.py:329` | a "train-vs-val" block would measure a divergence that does not exist for HINT |

What is the SAME, and was therefore reused rather than rewritten:

* gallery construction — ordered union of each caption row's `candidate` then `target`
  (`datasets.py:131-169` (`get_test_data`)), byte-identical rule to LIMN's (`LIMN/datasets.py:136-169`), including the
  `images.index(...)` positions.
* the reference mask — `sims[i, test_targets_id.index(t['source_img_id'])] = -10e10` (`test.py:82-83`; LIMN `test.py:81-82`).
* the ranking convention — `np.argsort(-sims[i, :])[:50]`, 1-based index of the target (`test.py:86`; LIMN `test.py:85`).
* text composition — `correct_text` (`datasets.py:75-80`) + concat (`datasets.py:82-84`), imported from
  `t4_parity.py`, not re-typed.
* the protocol — `fashioniq_val_split`, same query counts.
* the tolerance table — imported from `t4_parity.py` (§5). No tolerance was moved or restated.

### `relation_state`: NOT applicable, and here is the measurement

`workbench/third_party/HINT/lavis/models/blip2_models/HINT.py` — inside `__init__` (`:243-303`) the
only stochastic draw is `self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)`
at `:298`; `self.temp` (`:293`) is `0.07 * torch.ones([])`, deterministic. No `torch.randn` in
`__init__` (grep: 0 hits inside the constructor).

Two things follow, and both are recorded rather than argued:

1. `prompt_tokens` is **unused by the evaluated path**: grep over the whole pinned tree finds it only
   at its declaration (`:295`) and its init (`:298`). `extract_retrieval_compose` uses
   `self.query_tokens` (`:410`), which comes from `init_Qformer` and **is** in `state_dict`, so the
   checkpoint restores it.
2. It is also not a `learnable_relation`-class defect: ENCODER's tensor is *consumed* by the forward
   and absent from `state_dict`; HINT's draw is consumed by nothing.

So the artifacts carry **no `relation_state` key on either side**. The gate's rule
(`parity_gate.py:1030-1088`) makes this a NON-mandatory check reported `ok=False` (dimension
`relation_parity` False), it does **not** make it a mandatory failure — that only happens for
`RELATION_DRAWING_MODELS = ("encoder",)` (`:266-267`) or when either artifact carries the key at all.
**Do not add the key**: writing a digest for a tensor the forward never reads would put a measured-looking
value beside a check that must stay non-mandatory. `hint` is deliberately absent from
`RELATION_DRAWING_MODELS`, and that file is out of this lane's scope, so the choice is recorded here
rather than encoded there.

## 2. The shared interface — what the NEXT model reuses

Everything below lives in `scripts/third_model_parity.py` and is **model-independent**. A new model
supplies §3 only.

```python
@dataclass(frozen=True)
class ModelScope:                        # data, no behaviour
    model_id: str
    adapter_module: str
    replay_module: str
    checkpoint_id: str
    categories: tuple[str, ...]
    gallery_scope: str                   # "per-category" | "category-agnostic"
    target_rank: int                     # 2 = one vector/image, 3 = BLIP-2 token layout
    target_category: str | None          # None for category-agnostic models
    extra_tolerances: dict | None
    def declared_tolerances(self) -> dict

class CaptureHook:                       # the two hooks + the scope function
    scope: ModelScope
    def gallery(self, loaded, category) -> dict           # SCOPE FUNCTION
    def native_features(self, loaded, paths, texts) -> np.ndarray      # CAPTURE HOOK, query
    def native_target_features(self, loaded, paths) -> np.ndarray      # CAPTURE HOOK, target

# ---- shared, imported by the next model, never re-implemented ----
def fixed_sample(caption_rows, gallery_ids, *, query_count=5, gallery_count=20) -> dict
def digest_ids(values: Sequence[str]) -> str
def compare(actual, reference) -> dict            # max_abs / max_rel / min_cosine / bitwise_equal
def toleranced(block, *, abs_key, rel_key, declared) -> dict
def native_control(category) -> dict              # stored native ranks, re-verified
def decode_equality(paths) -> dict
def identity_phase(hook, category, config) -> dict
def native_phase(hook, category, config) -> dict
def adapter_phase(hook, category, config) -> dict
def scoring_ranking(hook, category, inputs, config, declared) -> dict
def update_native_metrics_flag(hook, category, ranking) -> None
def emit_native(hook, category, identity, config) -> Path
def self_check() -> int                            # CPU, no weights, no worker
```

Deliberately NOT abstracted, because it is model-specific and one model's answer is not a template:
checkpoint loading (HINT unpickles a whole module via `load_trusted_whole_model`), the transform call,
and the text processor.

`HintCapture` is instance 1. A second model is: subclass `CaptureHook`, fill `scope`, implement the
three methods, and call the shared phases. Note the honest limit of this claim — the interface has
exactly **one** implementation, so "the next model reuses it" is a property of its shape, not
something exercised by a second model yet.

## 3. The sample, and why each rule exists

Five query slots, spent in priority order (`fixed_sample`), with the losers recorded:

| slot | rule | property it tests |
|---|---|---|
| 1-2 | `multi_positive_first` / `multi_positive_second` — two rows sharing one `candidate` | one reference, **two** targets: what a per-query max-over-targets metric has to get right |
| 3 | `last_row` | the final **partial** batch (`1961 = 32*61 + 9`) |
| 4-5 | `capitalized_first_caption`, `short_second_caption` | text normalisation; a short half-caption |

Any rule that finds no free slot is listed in `selection_rules_skipped_for_slot_budget` — on all three
categories that is `["first_row", "middle_row"]`, because the two property-bearing rules and the tail
claim five slots on their own. Measured sample, all three categories:

| category | query indices | two rows share a reference | gallery images | image batch partial |
|---|---|---|---|---|
| dress | 0, 14, 18, 816, 2016 | yes | 20 | yes (20 % 32 ≠ 0) |
| shirt | 1, 2, 8, 798, 2037 | yes | 20 | yes |
| toptee | 0, 3, 15, 637, 1960 | yes | 20 | yes |

The gallery sample takes both positives of every sampled query first (a reference missing from the
gallery would make the rank meaningless), then the category's last gallery entry, then gallery order.

## 4. Gates registration: the capture sources, and why NOT re-running native

Decision: **the native side reuses the stored measurement; the adapter side needs a fresh capture.**
The stored files were inspected, and none of them is a native-vs-adapter capture:

| file | what it is | reusable as |
|---|---|---|
| `checkpoint_selection/20261106T233409+0700/hint/measurements/hint_native_verify.json` | 2017/2038/1961 **per-query native ranks**, `reference_masked: true`, checkpoint sha re-verified, recalls recomputed from the ranks | **the native rank control** for `scoring_ranking` |
| `hint_native_b16.json` | same, batch 16; ranks byte-identical to the batch-64 file | cross-check only |
| `hint_parity_top50.json` | pinned `test.py` vs a hand transcription — **both native**; top-50 set/order equality, `score_tie_pairs_in_row` | evidence the native path agrees with itself |
| `hint_doc_command.json` | native recalls only (R@10 59.803…, R@50 80.130…) | the native aggregate |
| (`raw/`) | **no HINT file at all** in the review's `raw/native/` | — |

None carries an adapter feature, a `layer2_4_forward` block, or a forward delta. Reshaping them into
`g1_g2.hint.*.adapter.json` would have meant **writing the adapter side by hand**, which the brief
forbids and which would be a fabricated pair. So: native ranks reused (they are a real measurement of
the same bytes and population — `native_control()` re-verifies the checkpoint digest and that the
stored ranks still reproduce the recorded macro before using them), adapter side left for the capture.

## 5. Tolerances: which table, and that nothing moved

Imported from `t4_parity.py:35-42` (`from t4_parity import TOL`), which is the table both the LIMN
pair and the ENCODER pair declare (`PREMISES.md:213`, `g1_g2.encoder.dress.native.json`):

```
forward_max_abs 1e-05   forward_max_rel 1e-04   batch_max_abs 1e-04
batch_max_rel   1e-03   score_max_abs   1e-06   tie_epsilon   1e-06
```

No model-specific override exists for HINT (`extra_tolerances: None`), so none was invented. Both
sides of every HINT pair will declare this same table through `ModelScope.declared_tolerances()`;
`parity_gate._tolerances` refuses a pair whose two sides disagree, so one definition is also the
mechanism that keeps them equal.

## 6. Measured values, per comparator

All six artifacts come from one real capture (native then adapter, both under `gpu_job.sh`'s flock:
`[gpu_job] lock held, GPU empty, launching:` — no MAX_WAIT or refusal line). The native interpreter was
NOT the controller: the checkpoint unpickles a `BertTokenizer` with `_added_tokens_encoder`, which only
the registry-declared `transformers==4.25.0` build still has; under the controller's 4.36.2 the native
forward dies with `AttributeError: 'BertTokenizer' object has no attribute '_added_tokens_encoder'`.
So native ran under `ilearn39-adapted` and the adapter phase under `ctl312h` (the adapter refuses to
launch a worker from its own environment). Both directions are now enforced by a guard in `main()`.

### 6a. Forward and pixel parity: bitwise identical

| category | `sample_forward.query` max_abs | `.target` max_abs | `min_cosine` | `layer2_pixels` max_abs |
|---|---|---|---|---|
| dress | **0.0** (bitwise) | **0.0** (bitwise) | 1.0 | **0.0** (bitwise) |
| shirt | **0.0** (bitwise) | **0.0** (bitwise) | 1.0 | **0.0** (bitwise) |
| toptee | **0.0** (bitwise) | **0.0** (bitwise) | 1.0 | **0.0** (bitwise) |

The adapter's query and gallery features are the native evaluator's, byte for byte, on all three
categories; the pixel decode is likewise identical. `forward_parity` holds in every category.

`cache_off_restart` max_abs `0.0` in all three: a restarted worker reproduces its own features exactly.

### 6b. Rank/tie parity: zero real differences

| category | `rank_differences` | of which the control's `0` sentinel | **on a real native rank** | `unexplained_rank_differences` |
|---|---|---|---|---|
| dress | 492 | 492 | **0** | 0 |
| shirt | 382 | 382 | **0** | 0 |
| toptee | 323 | 323 | **0** | 0 |

`hint_native_verify.json` stores `0` for "outside the top 50 / reference masked" (`rank_counts
.below_top50_or_masked` = 492 = the count of zeros in dress). The adapter-side rank at those same rows
is a true rank > 50 — the same outcome recorded two ways. Counting them as disagreement is what made a
first run report 492/382/323 "differences" and `max_abs_rank_delta` in the thousands; separating the
sentinel shows **the two sides agree on every query that has a real native rank**.

### 6c. Metrics parity: independently recomputed on both sides

| category | native hits (r1/r10/r50) | adapter hits, native protocol | equal |
|---|---|---|---|
| dress | 432 / 1034 / 1525 | 432 / 1034 / 1525 | **true** |
| shirt | 663 / 1272 / 1656 | 663 / 1272 / 1656 | **true** |
| toptee | 628 / 1289 / 1638 | 628 / 1289 / 1638 | **true** |

`native_metrics_match_feature_rerank: true` on all three, and it is earned: the native vector is the
stored per-query ranks' own counts (which reproduce the recorded macro — `21.4179/51.2643/75.6073` etc.
— to 1e-9), and the adapter vector is computed here from adapter features ranked under the native
protocol. Neither was copied from the other.

### 6d. Score parity

`score_max_abs_delta_vs_dot` = `2.09e-07` (dress) / `1.79e-07` (shirt, toptee) against declared
`score_max_abs 1e-06` — inside. These are the float32 floor of the same dot product.

### 6e. Batch parity: the two failures, and the tolerance that did not move

| category | `batch_1_vs_n.max_abs` | `batch_1_vs_n.max_rel` | declared `batch_max_rel` | verdict |
|---|---|---|---|---|
| dress | 1.341e-07 | 1.526e-04 | 1e-03 | pass |
| shirt | 1.527e-07 | **5.050e-03** | 1e-03 | **FAIL** |
| toptee | 1.527e-07 | **4.214e-02** | 1e-03 | **FAIL** |

`min_cosine` is `0.9999999999999` and `batch_max_abs` is `1.5e-07` on both failing categories: the
batch-1-vs-N difference is ~1e-7 in absolute terms and the RELATIVE figure is large only because
`max_rel` divides by the smallest |reference| component in the row. This is the same denominator
artefact ENCODER's nine `max_rel` failures were diagnosed as (`ENCODER_RELATION.md` §11), on a smaller
set. **It is reported as a failure because it is one under the declared table**, and the table was not
touched after the numbers were read — that would be moving the measurement to fit the result.
`batch_max_rel 1e-3` and `forward_max_rel 1e-4` are exactly the values in `t4_parity.py:35-42`.

Absent by construction, and honestly so: no `relation_state` (§1), and no `layer2_pixels
.adapter_decode_vs_native_train_query` (HINT has no train-vs-val split; §1 row 5).

### 6f. Defects this turn caught before they reached the artifacts

1. **`query_order_sha256` sample-vs-population** — native digested the 5-row sample, the adapter the
   full population, which would have been a mandatory `input.identity_agreement` failure in all three
   categories. Fixed at the source; `--self-check` now asserts agreement over
   `parity_gate.IDENTITY_FIELDS` and refuses a one-sided write via `assert_identity_agreement`, which
   runs BEFORE the adapter artifact is written.
2. **3-D gallery normalised on the wrong axis** — `np.linalg.norm(gallery, axis=1)` normalised across
   the 32 Q-former tokens instead of the 256-d feature axis, giving a score delta of **1.17** against a
   `1e-6` tolerance and dropping r1 from 432 hits to 373. Fixed to `axis=-1`; `--self-check` now proves
   the two normalisations are distinguishable and that only the feature-axis one leaves unit vectors.
3. **Rank-`0` sentinel counted as a hit** — `rank <= k` treated the control's "missed" marker as rank
   zero, inflating native r1 from 432 to 924 and reporting 1080 "unexplained" rank differences. Fixed;
   the control now certifies `zero_rank_count == rank_counts.below_top50_or_masked`.

All three were caught by checking emitted numbers against the stored native measurement on CPU, not by
a passing gate.

## 7. Gate state, read-only

```
$ python -c "... _discover_models(evidence_root())"
['encoder', 'hint', 'limn']

$ python -c "... evaluate_model('hint')"
PARITY_MANDATORY_SUBCHECK_FAILED {'dress': [], 'shirt': [], 'toptee': []} 2 [
  ('shirt.batch_1_vs_n.max_rel',  False, 0.005049934976857354),
  ('toptee.batch_1_vs_n.max_rel', False, 0.04213516386699195)]

$ python -c "... evaluate_all()"
passed_models []
standard ['parity-absolute-relative/v2', 'parity-relative/v1']
encoder PARITY_MANDATORY_SUBCHECK_FAILED   9 failures
hint    PARITY_MANDATORY_SUBCHECK_FAILED   2 failures
limn    PARITY_MANDATORY_SUBCHECK_FAILED   3 failures
```

`missing_by_category` is empty for every category: nothing mandatory is unmeasured. `passed_models` is
empty for ALL THREE models — HINT is not an outlier, and no model reaches `PASS`.

**The acceptance criterion is met and stated precisely**: HINT is discovered from its own filenames and
receives a verdict COMPUTED FROM ITS OWN PAIRED ARTIFACTS — not `NOT_RUN`, not `EVIDENCE_MISSING`, and
not an asserted pass. It is the third model with paired G1/G2 evidence after ENCODER and LIMN. What
blocks a pass for HINT is exactly two `batch_1_vs_n.max_rel` subchecks, both listed above with their
measured values; §6e states why they are small in absolute terms and why they are nonetheless reported
as failures.

## 8. GPU_READY — spent

`GPU_READY` was `yes`; the green light was used and the capture is DONE. Nothing is queued.

Both phases were launched detached under `workbench/scripts/gpu_job.sh` with `MAX_WAIT=3600`, each
acquiring the flock normally (`[gpu_job] lock held, GPU empty, launching:`):

```bash
# native -- MUST be the model interpreter (transformers==4.25.0)
env OMP_NUM_THREADS=2 ... workbench/scripts/gpu_job.sh \
  workbench/envs/ilearn39-adapted/bin/python .../third_model_parity.py --phase native

# adapter -- MUST be the controller (the adapter refuses to launch a worker from its own environment)
env OMP_NUM_THREADS=2 ... WORKBENCH_COMPOSITION_BATCH_SIZE=32 workbench/scripts/gpu_job.sh \
  workbench/envs/ctl312h/bin/python .../third_model_parity.py --phase adapter
```

The model interpreter is still resolved from `models.yaml`'s `python_env_var` by
`runtime.resolve_model_interpreter` INSIDE `Adapter.__init__`; it is never passed as an argument to the
adapter. Timing: native 61.5 / 55.3 / 53.6 s per category; adapter 100.0 / 111.8 / 103.3 s. Logs:
`raw/third_model/{native,adapter3}.log`.

To reproduce from scratch, delete `raw/third_model/{sample_,population_}*` and the six artifacts, then
run the two commands in that order. Re-running only `--phase adapter` is safe and is what the last two
fixes used.

## 9. Files

Owned and touched:

* `scripts/third_model_parity.py` — producer (new).
* `raw/third_model/{identity,checkpoint_identity,population_*}.json|npz` — intermediates (new).
* `g1_g2.hint.{dress,shirt,toptee}.native.json` — the three NATIVE artifacts (new, measured).
* `g1_g2.hint.{dress,shirt,toptee}.adapter.json` — the three ADAPTER artifacts (new, measured).

Nothing else was modified: no `parity_gate.py`, no `composition.py`, no `models.yaml`, no
`workbench.env`, no ENCODER/LIMN artifact, no tolerance changed, no commit, no staging. The GPU was
used (that was the green light); `gpu_job.sh` was never bypassed and the lock was never deleted.
