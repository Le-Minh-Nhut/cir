# ENCODER — the unseeded relation, and what wiring it proved

Date: 2026-11-12. Model: `encoder` (`checkpoint_id: fashioniq`, sha256 `0f31c501…9173`).
Scope: `g1_g2.encoder.{dress,shirt,toptee}.{native,adapter}.json` and
`scripts/encoder_parity.py`. This document records a re-measurement; it does not adjudicate any
doc-level claim (see §8 for the queue handed to the doc worker).

## 1. Defect

Both PRODUCTION paths already share one relation pin:

| path | seeds | records |
|---|---|---|
| adapter (`workbench/composition/model_encoder.py`) | `:500` `torch.manual_seed(relation_seed)`, before `Encoder(...)` at `:504` | `:231-239` `_relation_record` |
| native replay (`workbench/replay/encoder.py`) | `:324` `torch.manual_seed(RELATION_SEED)`, before `Encoder(...)` | `:253-270` `_relation_record` |

with the constants at `model_encoder.py:59` (`RELATION_SEED = 20261111`) and `:65`
(`RELATION_EXPECTED_DIGEST`), imported by the replay module at `:36`.

The G1/G2 **producer** was not wired. `scripts/encoder_parity.py` constructed `Encoder` unseeded
(`:388` in the pre-fix file) and
`grep -c "manual_seed\|RELATION_SEED\|relation_state"` on it returned **0**. Consequently all six
artifacts carried no `relation_state`, and the gate reported:

```
relation.<cat>.digest measured='native=not_recorded, adapter=not_recorded'
```

as a MANDATORY failure for `encoder` (`parity_gate.py:267` `RELATION_DRAWING_MODELS`,
`_relation_checks` ~`:779`).

## 2. Root cause, one line

`BindingDecoder.learnable_relation` is `nn.Parameter(torch.randn(N_p, dim)).cuda()`
(`workbench/third_party/ENCODER/model_try2.py:93`) — constructed, absent from `state_dict()` and
from the released 417-key checkpoint (`raw/encoder_parity/relation_proof_construct.out`:
`IN_STATE_DICT False`, `SHAPE [2, 512]`). Nothing pinned the draw, so the tensor is a function of
whatever the global RNG held at construction: **unpinned, and therefore not reproducible across
processes or runs.**

What the pre-fix evidence does and does not show, stated precisely because the first draft of this
section overclaimed it. The two sides of a *single* artifact pair were **not** internally
inconsistent: measured on the HEAD arrays, cross-side feature distance was
`1.1324882507324219e-06` (dress) / `1.648440957069397e-06` (shirt), which reproduces the HEAD
artifacts' own `population.features_vs_native_val_control.max_abs` and `gallery_vs_native.max_abs` to
the digit. Both processes consumed the same construction graph under the same default RNG state, so
they drew the same bytes by circumstance. The era shift between runs moves the same features by
`0.013255782425403595` / `0.01691335439682007` — roughly **1e4× larger** — which is the actual
defect: the value differed from run to run by ~1e-2 in feature space.

So the defect is **unpinned and unreproducible across runs**, not "the two sides used different
relations". Nothing *guaranteed* the agreement within a pair; it was an accident of two identical
processes, and any change in construction order, library version or RNG consumption upstream would
have broken it silently, with `forward_parity` unable to see it because it only reads layer outputs.
Source: `raw/stable_ids/README.md` (stable-ID lane).

## 3. Fix

`workbench/docs/reviews/20261109-native15-composition-limn/scripts/encoder_parity.py`,
39 insertions / 3 deletions:

* **`:137` area, `native_phase`** — `from workbench.composition.model_encoder import RELATION_SEED`
  (imported, never copied) and `torch_mod.manual_seed(RELATION_SEED)` immediately before
  `model_try2.Encoder(...)`, inside the existing `os.chdir(source_root)` block: the same placement
  as both production paths.
* **`:158-161`** — the record comes from the production recorder, `replay._relation_record(model,
  None)`. No third digest formula: `sha256` over the float32 bytes of
  `backbone.binding_decoder.learnable_relation`, same vocabulary (`matches_pinned_digest`).
  `destination=None` because the record travels in the artifact rather than an `.npy`.
* **`:749-755` area, `adapter_phase`** — `adapter.prepare_identity()`, the production entry point
  the runner itself calls, **inside** the `try` so a failed handshake still closes the worker.
* **`main()`** — the record is lifted from the body to the document root, where
  `parity_gate._relation_checks` reads it: `("M-01: read from the WHOLE document, not the `body`")`.
* **`RAW`** now honours `ENCODER_PARITY_RAW`, so a re-measurement writes its intermediates to
  `/tmp` instead of over the tracked `raw/encoder_parity/*.npz` it is meant to be compared against.

Nothing else moved: no tolerance changed, no check deleted, no FAIL relabelled.

## 4. Gate, before and after

### Before (16 failures)

```
PARITY_MANDATORY_SUBCHECK_FAILED
relation_parity | relation.dress.digest |ok= False | native=not_recorded, adapter=not_recorded
forward_parity | dress.population.features_vs_native_val_control.max_rel |ok= False | 3.3548386852382883
forward_parity | dress.population.gallery_vs_native.max_rel |ok= False | 1.3636362821713672
batch_parity   | dress.batch_1_vs_n.max_rel |ok= False | 0.0077950176161260265
scoring_parity | dress.ranks.non_tie_changes |ok= False | 1.0
scoring_parity | dress.ranks.min_gap_vs_tie_epsilon |ok= False | 0.0006571710109710693
relation_parity | relation.shirt.digest |ok= False | native=not_recorded, adapter=not_recorded
forward_parity | shirt.population.features_vs_native_val_control.max_rel |ok= False | 0.7519744456046599
forward_parity | shirt.population.gallery_vs_native.max_rel |ok= False | 7.7499972070904795
batch_parity   | shirt.batch_1_vs_n.max_rel |ok= False | 0.0031308995258445375
scoring_parity | shirt.ranks.non_tie_changes |ok= False | 1.0
scoring_parity | shirt.ranks.min_gap_vs_tie_epsilon |ok= False | 0.00016041100025177002
relation_parity | relation.toptee.digest |ok= False | native=not_recorded, adapter=not_recorded
forward_parity | toptee.population.features_vs_native_val_control.max_rel |ok= False | 7.850000984209719
forward_parity | toptee.population.gallery_vs_native.max_rel |ok= False | 3.897590520814516
batch_parity   | toptee.batch_1_vs_n.max_rel |ok= False | 0.011646653843520266
```

### After (9 failures, 99 checks)

```
STATUS PARITY_MANDATORY_SUBCHECK_FAILED
FAILURES 9
CHECKS 99
  FAIL dress.population.features_vs_native_val_control.max_rel |mandatory= True | 0.7499999820540181
  FAIL dress.population.gallery_vs_native.max_rel |mandatory= True | 10.461538814286977
  FAIL dress.batch_1_vs_n.max_rel |mandatory= True | 0.003176237364982574
  FAIL shirt.population.features_vs_native_val_control.max_rel |mandatory= True | 4.236841887544606
  FAIL shirt.population.gallery_vs_native.max_rel |mandatory= True | 0.5312498017651329
  FAIL shirt.batch_1_vs_n.max_rel |mandatory= True | 0.0027347281757461243
  FAIL toptee.population.features_vs_native_val_control.max_rel |mandatory= True | 6.2777780084400545
  FAIL toptee.population.gallery_vs_native.max_rel |mandatory= True | 3.133333651667554
  FAIL toptee.batch_1_vs_n.max_rel |mandatory= True | 0.0015533629650869047
--- relation ---
   relation.dress.digest ok= True | native=20db0c27c979e9a0 adapter=20db0c27c979e9a0
   relation.shirt.digest ok= True | native=20db0c27c979e9a0 adapter=20db0c27c979e9a0
   relation.toptee.digest ok= True | native=20db0c27c979e9a0 adapter=20db0c27c979e9a0
```

`16 → 9`, not `16 → 13`. Seven checks flipped, not three:

| check | before | after |
|---|---|---|
| `relation.<cat>.digest` ×3 | `not_recorded` | `ok=True`, both sides `20db0c27…` |
| `dress.ranks.non_tie_changes` | `1.0` | `0.0` |
| `dress.ranks.min_gap_vs_tie_epsilon` | `6.5717e-4` FAIL | `0.0` ok |
| `shirt.ranks.non_tie_changes` | `1.0` | `0.0` |
| `shirt.ranks.min_gap_vs_tie_epsilon` | `1.6041e-4` FAIL | `0.0` ok |

The four rank checks flipped for a **measured** reason, and it is not the one this section originally
gave. They did **not** evidence two sides using different relations — the pre-era pair shared one
relation (cross-side feature distance `1.1324882507324219e-06` dress / `1.648440957069397e-06` shirt,
reproducing the HEAD artifacts' own `max_abs` to the digit). What happened is that both historical
disagreements were a **knife edge inside one shared-relation state**:

| | dress `7:1565` | shirt `7:3362` |
|---|---|---|
| reference / target | `B004DTTRA8` / `B0048OYGF4` | `B00AMD5I7U` / `B005LIXPPI` |
| pre ranks (native, adapter) | 1215, 1216 | 1373, 1372 |
| post ranks (native, adapter) | **1222, 1222** | **1372, 1372** |
| pre deciding margin | `1.043081283569336e-07` | `1.1920928955078125e-07` |
| pre cross-side row `max_abs` (that state's own float32 noise) | `8.344650268554688e-07` | `7.7858567237854e-07` |
| **pre margin ÷ that noise** | **0.125** | **0.1531** |
| era shift (movement ÷ pre margin) | 34169.29 | 8192.13 |

The deciding margin was **an eighth to a sixth of the cross-side float32 noise of the very state it
lived in**. Reseeding moves the same features by `0.013255782425403595` / `0.01691335439682007` —
3.4e4× / 8.2e3× that margin — which destroys the edge and sends both queries to `1222` / `1372` on
**both** sides. The checks now pass because the scores moved away from a sub-noise boundary, not
because a discrepancy was reconciled.

This is **not** a tie, and must not be described as one: there are zero exact target ties in any
stored era, and `1 + #greater + #earlier-equal` equals the argsort position on all four rows. A
tie-policy explanation would need a relation state in which the target ties a gallery row exactly,
which no stored era exhibits. Source: `raw/stable_ids/README.md`.

Neither case moves a hit count — both ranks sit far outside every cutoff (`~1222`, `~1372` against
k ∈ {1, 10, 50}) — which is why `min_gap_abs: 0.0` with `non_tie_count: 0` is safe here, and why the
historical 1-rank difference never affected a reported metric.

**The two historical rank disagreements are resolved for the parity question**: `dress` index
**1569** (adapter 1216 vs native 1215) and `shirt` index **1352** (1372 vs 1373) both report
`count: 0, head: []` in the regenerated artifacts, as does `toptee`. Their *doc-level* wording is
still being adjudicated, and the stable-ID lane's own caveat applies: that lane's auxiliary counters
(`adapter_vs_argsort_unmasked.count` 28 dress / 7 shirt, `native_mask_only_effect.count` 1093 dress,
shirt `adapter_style_recall.r1` 19.479882237487733 vs 19.528949950932 recomputed) do **not** reproduce
and must not be read as reproduced. The two decisions this review rests on —
`rank_differences_vs_native_val_control = 0` and the per-query ranks — do reproduce exactly from the
stored arrays.

## 5. The six `relation_state` blocks

`seed = 20261111` on all six; `expected_sha256 =
20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` on all six;
`matches_pinned_digest = true` on all six.

| category | side | `sha256` | agrees |
|---|---|---|---|
| dress | native | `20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` | — |
| dress | adapter | `20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` | ✓ |
| shirt | native | `20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` | — |
| shirt | adapter | `20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` | ✓ |
| toptee | native | `20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` | — |
| toptee | adapter | `20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92` | ✓ |

**Independent cross-confirmation.** The seeded composition run
`workbench/artifacts/composition/summary/encoder_fashioniq_9b806c22221e944ff10e514e/full.json`
records `relation_state.sha256 = 20db0c27…`, `matches_pinned_digest: true`, `seed: 20261111` — the
same bytes, in a run that never touched this harness. So the pin reproduces across three independent
processes (parity native, adapter worker, composition runner).

The pinned constant is **not** a first-draw digest: `Backbone.__init__` constructs the CLIP tower and
many `nn.Linear`/`nn.Sequential` before `BindingDecoder` draws, so a bare
`torch.manual_seed(20261111); torch.randn(2, 512)` yields `768805d7…`, not the pin. The pin is the
digest of the draw at its actual position in the construction order. It is not "fixed" to match any
measurement here; all three processes agree on it as-is.

## 6. Negative test

Copies of the six REGENERATED artifacts under `/tmp/enc_rel_neg2`, judged with
`evaluate_model('encoder', root=…)`:

| step | status | `len(failures)` | relation check |
|---|---|---|---|
| baseline copy | `PARITY_MANDATORY_SUBCHECK_FAILED` | **9** | all three ok |
| `shirt.adapter.relation_state.sha256 := "de"*32` | `PARITY_MANDATORY_SUBCHECK_FAILED` | **10** | `relation.shirt.digest ok=False mandatory=True \| native=20db0c27c979e9a0 adapter=dededededededede` |

The check bites: equal digests pass, one perturbed digest adds exactly one cited failure. Nothing in
the model or the artifacts was touched to produce this — only a JSON field in a `/tmp` copy.

## 7. Key-set diff

For all six: `ROOT_removed []  ROOT_added ['relation_state']  BODY_removed []  BODY_added []`.

Old root keys — `category`, `phase`, `model_id`, `protocol`, `tolerances`, `native`|`adapter` — are a
strict subset of the new ones; exactly one key added; **no** body key added or removed on any side.
This matters because `_identity_cross_checks` fails on a field carried by one side only; with no body
change and `relation_state ∉ IDENTITY_FIELDS` (`model_id`, `checkpoint_id`, `checkpoint_identity`,
`protocol_id`, `protocol`, `query_order_sha256`, `gallery_order_sha256`), the post-regeneration
failure list contains **zero** `input.identity_agreement.*` names.

## 8. The native-metric shift: the old numbers were unseeded

Regenerating the native side moved its Recall. That is not noise — it is the defect:

| category | metric | old (HEAD) | new (seeded) | Δ |
|---|---|---|---|---|
| dress | r1 | 21.269211700545366 | 21.120475954387704 | −0.1487 |
| dress | r10 | 51.95835399107585 | 51.71046108081309 | −0.2479 |
| dress | r50 | 76.35101636093208 | 76.30143777887952 | −0.0496 |
| shirt | r1 | 25.36800785083415 | 25.318940137389596 | −0.0491 |
| shirt | r10 | 54.9558390578999 | 54.90677134445535 | −0.0491 |
| shirt | r50 | 75.4661432777233 | 75.3189401373896 | −0.1472 |
| toptee | r1 | 29.3727689954105 | 29.117797042325343 | −0.2550 |
| toptee | r10 | 61.805201427842938 | 62.0601733809281 | +0.2550 |
| toptee | r50 | 81.030086690464 | 80.97909229984701 | −0.0510 |

Macro over the three categories, averaged from the nine values above: R@1 `25.3367 → 25.1857`,
R@10 `56.2398 → 56.2258`, R@50 `77.6157 → 77.5332`.

**The old native-side numbers were drawn from an unseeded RNG and are therefore not reproducible
from the code that produced them.** The numbers in this section are now the reproducible ones: same
code, same seed, any process, identical bytes (§5).

## 9. TASK A — consistency against the stored native run

> **The subsection below is SUPERSEDED.** It is a single-reference comparison: it compared the
> seeded parity native side against `raw/native/encoder.json` only. There are **three** native
> numbers for this model, not two, and neither stored artifact records a seed — so no readback of
> this artifact can be called "reproduced". The corrected three-way analysis is §9b. The original is
> kept verbatim so the correction is visible rather than silent.

### 9a. Single-reference comparison (SUPERSEDED — do not cite)

Located artifact: `workbench/docs/reviews/20261109-native15-composition-limn/raw/native/encoder.json`
(`recorded_at 2026-11-09T13:23:23Z`, `EXITCODE=0`, `WALL_SECONDS=131`). No per-category copy of the
same run exists elsewhere; `workbench/artifacts/composition/summary/encoder_fashioniq_*` are
composition runs on a 6003-query scope, a different protocol, so they are not comparable here.

Query sets match the parity harness exactly — 2017 / 2038 / 1961 — and both sides use the same
gallery (`g1_g2.encoder.<cat>.native.json`: `population {queries: 2017|2038|1961, gallery:
2628|3089|2902}`).

| category | metric | stored native run | old parity native | NEW parity native | old == run |
|---|---|---|---|---|---|
| dress | r1 | 21.4675260287556 | 21.2692117005454 | 21.1204759543877 | no |
| dress | r10 | 51.9583539910759 | 51.9583539910759 | 51.7104610808131 | **yes** |
| dress | r50 | 76.3510163609321 | 76.3510163609321 | 76.3014377788795 | **yes** |
| shirt | r1 | 25.5152109911678 | 25.3680078508342 | 25.3189401373896 | no |
| shirt | r10 | 54.9558390578999 | 54.9558390578999 | 54.9067713444554 | **yes** |
| shirt | r50 | 75.4170755642787 | 75.4661432777233 | 75.3189401373896 | no |
| toptee | r1 | 29.3727689954105 | 29.3727689954105 | 29.1177970423253 | **yes** |
| toptee | r10 | 61.958184599694 | 61.8052014278429 | 62.0601733809281 | no |
| toptee | r50 | 81.1320754716981 | 81.030086690464 | 80.979092299847 | no |

Deltas, new − run: dress `−0.3471 / −0.2479 / −0.0496`; shirt `−0.1963 / −0.0491 / −0.0981`;
toptee `−0.2550 / +0.1020 / −0.1530`. Macro `R@10 56.2908 → 56.2258` (−0.0650), `R@50 77.6334 →
77.5332` (−0.1002).

**Outcome: (c), with (b) as the magnitude.** The stored native run is itself unseeded. Two facts
establish it rather than infer it:

1. **Chronology.** The run was recorded `2026-11-09T13:23Z`. `RELATION_SEED` landed on
   `2026-11-11 03:16 +0700` (commit `be14393`). The run predates the constant that would have made it
   reproducible, and `encoder.stderr.log` contains no relation record.
2. **Exact equality with the old, unseeded parity native side** — `dress` r10/r50, `shirt` r10 and
   `toptee` r1 match the run to the last bit (`51.95835399107585 == 51.95835399107585`, etc.). Two
   unseeded draws that happen to land on identical values in several cells is the signature of the
   same *kind* of run, not of a shared seed.

The mismatch is not a query-subset, gallery or transform difference: those are byte-identical (same
2017/2038/1961, same gallery, `native_metrics_match_feature_rerank: true` in all three). It is the
relation draw.

**Consequence for the published aggregate.** The stored macro **R@10 56.2908 / R@50 77.6334 must be
marked NOT-REPRODUCIBLE until rerun on the seeded path.** The reproducible value from the seeded
parity native side is **R@10 56.2258 / R@50 77.5332**. They differ by 0.065 / 0.100 points — small,
but the point is not size: the stored figure cannot be regenerated from the repository as it stands,
whereas the new one can, in any process. No native evaluation was rerun for this comparison; it is a
comparison of stored numbers.

### 9b. Corrected three-way comparison (use this)

**Threshold, declared before the numbers were read: 1e-3 absolute, on recall percentage points.**
An artifact is called `REPRODUCED_BY_SEEDED_PATH` only if all nine of its cells (3 categories ×
r1/r10/r50) sit within 1e-3 pp of the seeded parity native side; `NOT_REPRODUCED` if it provably
diverges beyond that; `UNDETERMINED` if the question cannot be decided from what is stored.

Three native aggregates exist for `encoder`/`fashioniq`:

| id | file | `recorded_at` | macro R@10 | macro R@50 |
|---|---|---|---|---|
| **seeded** | `g1_g2.encoder.<cat>.native.json` (this re-measurement) | 2026-11-12 | 56.225801935399 | 77.533156738705 |
| **registry** | `artifacts/checkpoint_selection/20261106T002617+0700/cohort/measurements/encoder_fashioniq.json` | 2026-11-06T14:39:45Z | 56.223101675386 | 77.667385392714 |
| **review** | `docs/reviews/20261109-native15-composition-limn/raw/native/encoder.json` | 2026-11-09T13:23:23Z | 56.290792549557 | 77.633389132303 |

#### Identity fields, side by side

| field | registry | review |
|---|---|---|
| `model_id` | `encoder` | `encoder` |
| `checkpoint_id` | `fashioniq` | `fashioniq` |
| `protocol` | `fashioniq_val_split` | `fashioniq_val_split` |
| `checkpoint` | `…/cohort/downloads/encoder/fashioniq.pt` | same path |
| `checkpoint_sha256` | `0f31c501…9173` | `0f31c501…9173` |
| `source_commit` | `29a2a31d6a56f677bf450c3be7cdaef423fb7018` | same |
| `relation_state` | **absent** | **absent** |
| `seed` / `relation_seed` | **absent** | **absent** |
| `batch_size` | **absent** | **absent** |
| `query_count` / `gallery_count` | **absent** (not stored) | **absent** (not stored) |
| `per_query_published` | `false` | `false` |
| top-level keys | `model_id, protocol, results, aggregate, recorded_at` | identical set |

Both record the same checkpoint bytes and the same source commit, and **neither states the relation
it was computed with**. Neither states a batch size. `encoder.stderr.log` for the review run contains
only a timm `FutureWarning` — no seed, no batch size, no relation line. Query and gallery counts are
**not recorded in either file**; the hit-count table below reads them on the parity harness's
2017/2038/1961 (gallery 2628/3089/2902) because every stored value lands on an exact integer count at
those denominators, which is consistent with them but does not prove them.

#### Per-category, full precision (Δ in percentage points)

| cat | metric | seeded | registry | review | Δ seeded−reg | Δ seeded−rev | Δ reg−rev |
|---|---|---|---|---|---|---|---|
| dress | r1 | 21.120475954388 | 21.219633118493 | 21.467526028756 | −0.09916 | −0.34705 | −0.24789 |
| dress | r10 | 51.710461080813 | 51.859196826971 | 51.958353991076 | −0.14874 | −0.24789 | −0.09916 |
| dress | r50 | 76.301437778880 | 76.351016360932 | 76.351016360932 | −0.04958 | −0.04958 | **0.00000** |
| shirt | r1 | 25.318940137390 | 25.613346418057 | 25.515210991168 | −0.29441 | −0.19627 | +0.09814 |
| shirt | r10 | 54.906771344455 | 55.004906771344 | 54.955839057900 | −0.09814 | −0.04907 | +0.04907 |
| shirt | r50 | 75.318940137390 | 75.417075564279 | 75.417075564279 | −0.09814 | −0.09814 | **0.00000** |
| toptee | r1 | 29.117797042325 | 29.474757776645 | 29.372768995411 | −0.35696 | −0.25497 | +0.10199 |
| toptee | r10 | 62.060173380928 | 61.805201427843 | 61.958184599694 | +0.25497 | +0.10199 | −0.15298 |
| toptee | r50 | 80.979092299847 | 81.234064252932 | 81.132075471698 | −0.25497 | −0.15298 | +0.10199 |

Hit counts (exact, since every stored value is an integer count on its category's query set):

| cat | metric | seeded | registry | review |
|---|---|---|---|---|
| dress | r1 / r10 / r50 | 426 / 1043 / 1539 | 428 / 1046 / 1540 | 433 / 1048 / 1540 |
| shirt | r1 / r10 / r50 | 516 / 1119 / 1535 | 522 / 1121 / 1537 | 520 / 1120 / 1537 |
| toptee | r1 / r10 / r50 | 571 / 1217 / 1588 | 578 / 1212 / 1593 | 576 / 1215 / 1591 |

**No pair agrees within 1e-3 pp on any cell.** The smallest observed separation anywhere in the
three-way table is `dress` r50 seeded−registry at 0.04958 pp — 49.6× the threshold — and that number
is one R@50 hit on 2017 queries (the quantisation step, `100/2017 = 0.04958`). Every disagreement in
this table is an integer number of hits (1–7 hits per cell).

#### Agreement at coarse precision — stated, and weaker than it looks

`56.2258` (seeded) vs `56.2231` (registry) vs `56.2908` (review) share a **one-decimal** prefix
`56.2`; at two decimals seeded `56.23` and registry `56.22` already part, and the review artifact's
`56.29` never agrees. So the honest statement is: *the seeded path and the registry measurement agree
on macro R@10 to one decimal; they do not agree at full precision, and their per-category cells
differ by up to 0.357 pp.* On R@50 the two are 0.134 pp apart, which is ~3.6× further.

**Why the macro looks tighter than its own categories.** The nine `seeded − registry` cells cancel:
their sum is `+0.00809` pp, i.e. a mean of `+0.00270` pp. The per-category deltas are up to 0.357 pp
in magnitude, but they enter with mixed signs across categories and cutoffs, so the macro agrees far
better than any individual cell. A macro-level tolerance alone would therefore have called this
"reproduced" on R@10 — which is exactly the failure mode this three-way table exists to prevent.
The macro must not be read without the per-category table.

#### Verdicts against the pre-declared 1e-3 pp threshold

| stored artifact | verdict | basis |
|---|---|---|
| `cohort/measurements/encoder_fashioniq.json` (registry) | **NOT_REPRODUCED** | all 9 cells exceed 1e-3 pp (the smallest, `dress` r50 at 0.04958 pp, is still 49× over). Max cell 0.35696 pp. Macro R@10 agrees only to one decimal. |
| `raw/native/encoder.json` (review) | **NOT_REPRODUCED** | all 9 cells exceed 1e-3 pp; max cell 0.34705 pp. Macro R@10 0.065 pp and R@50 0.100 pp away. |
| neither: the *seeded* path | reproducible by construction | carries `relation_state.sha256 = 20db0c27…`, `seed 20261111`, and reproduces the same bytes in three independent processes (§5). |

**Neither stored artifact is `UNDETERMINED`** — both diverge far beyond the threshold, so the
question is settled, not unanswerable. But note the shape of the uncertainty: neither artifact
records the relation it used, so **neither can be repaired to a single value either**. The correct
reading is not "the seeded path falsifies the stored numbers"; it is "the stored numbers record two
and possibly more draws of a tensor nobody wrote down, so no single number was ever one measurement."

#### Third finding — two "native ENCODER" aggregates exist, and the registry does not know about the second

The registry and the review artifacts **disagree with each other** on every cell (Δ reg−rev up to
0.248 pp on dress r1; macro R@10 0.0677 pp, R@50 0.0340 pp). Same checkpoint sha256, same source
commit, same protocol, different `recorded_at` (2026-11-06 vs 2026-11-09). That is the
`TWO_NATIVE_MACROS` class the disposition matrix flags for `clip4cir_rn50x4_fullft` (`MODEL_MATRIX.md`
row 337, blocker 33) — **not currently flagged for `encoder`**.

Evidence that the matrix does not carry the second encoder macro: `MODEL_MATRIX.md:121` cites only
`cohort/measurements/encoder_fashioniq.json` (R@10 56.223101675386054 / R@50 77.66738539271432), and
grep over `MODEL_MATRIX.md`, `MODEL_MATRIX.csv`, `BASELINE.md` and `PREMISES.md` for
`raw/native/encoder.json`, `56.2908` and `77.6333` returns nothing. So the matrix is internally
consistent but records one of two stored aggregates.

**Which a reader should cite.** The **registry** measurement
(`cohort/measurements/encoder_fashioniq.json`), for three reasons: it is the file `MODEL_MATRIX.md`
already names as ENCODER's counted result; it is the earlier of the two on the same checkpoint; and
`20261110-e2e-repair/FINAL_REPORT.vi.md:211` independently notes the web surface serves
`encoder_fashioniq.json` while the review artifact is the "BC-cũ §3.6" quotation. Citing the review
artifact instead silently switches the number a reader compares against the published table.

Neither should be cited as *the reproducible* native ENCODER result. The only figure that is
reproducible from this repository today is **R@10 56.2258 / R@50 77.5332** from the seeded parity
native side — and even that is a parity-harness population, not a re-run of the evaluation entry
point. No native evaluation was rerun for any of this; it is a comparison of stored numbers.

## 10. TASK B — documents quoting the replaced values

Every hit of the old native-side values, with the classification the doc worker needs. **Live claim =
must be corrected; historical record = leave.** No doc was edited here.

| # | file:line | value(s) | class |
|---|---|---|---|
| 1 | `20261109-native15-composition-limn/ENCODER_PARITY.vi.md:353` | `21.2692 / 51.9584 / 76.3510` (native re-run column), `−0.198/0.000/0.000` vs stored | **LIVE — must be corrected.** This is the §5.3 metrics table claiming the re-run reproduces; its "D1 again, measured end to end" narrative was written when the parity native side was unseeded and now describes a measurement that no longer exists. |
| 2 | `ENCODER_PARITY.vi.md:354` | `25.3680 / 54.9558 / 75.4661` | **LIVE — must be corrected.** Same table. |
| 3 | `ENCODER_PARITY.vi.md:355` | `29.3728 / 61.8052 / 81.0301` | **LIVE — must be corrected.** Same table. |
| 4 | `ENCODER_PARITY.vi.md:383-385` | `51.9584 / 54.9558 / 61.9582` and val-control deltas | **LIVE — must be corrected.** §5.4 transform table's native column is the unseeded native metric; the val-control column is a different quantity and stays valid, but the Δ column is now wrong. |
| 5 | `20261110-e2e-repair/FINAL_REPORT.vi.md:104-105` | dress index 1569 (1216 vs 1215, `6.5717e-4`), shirt 1352 (1372 vs 1373, `1.6041e-4`) | **LIVE — must be corrected.** These two disagreements are now gone (`count: 0` in both regenerated artifacts). The record is of a phenomenon that the pin removed. |
| 6 | `20261110-e2e-repair/PARITY.vi.md:228-240` | four `within_tolerance:false` blocks, incl. `query_native_tensors` max_abs `0.021249 / 0.041421 / 0.023708` and the "2 non-tie / 0 qids" counts | **LIVE — must be corrected** for the rank counts (2 → 0). The `max_abs` values and the GENUINE_FAIL/DENOMINATOR_ARTEFACT taxonomy remain valid — see §11, they did not move. |
| 7 | `20261110-e2e-repair/GUARDS_FIXES.vi.md:20` | "`encoder` có **13** subcheck bắt buộc FAIL … 2 non-tie rank change" | **LIVE — must be corrected** (13 → 9, 2 non-tie → 0). |
| 8 | `GUARDS_FIXES.vi.md:368-369` | `non_tie_count` dress **1** (gap `6.572e-04`), shirt **1** (`1.604e-04`) | **LIVE — must be corrected** (both 0). |
| 9 | `GUARDS_FIXES.vi.md:429` | `scoring_parity … ❌ (2 non-tie change)` | **LIVE — must be corrected** (now passes). |
| 10 | `GUARDS_FIXES.vi.md:497` | "13 failure của `encoder`: … 4 `ranks.*` (2 `non_tie_changes` + …)" | **LIVE — must be corrected** (9 failures, 0 rank failures). |
| 11 | `20261109-native15-composition-limn/EVIDENCE_INDEX.md:515` | "G2 PASS with **two honestly-declared failures** … 2 non-tie rank" | **LIVE — must be corrected** (0 non-tie rank failures now). |
| 12 | `20261109-native15-composition-limn/FINAL_REPORT.vi.md:522` | "**0** thay đổi hit count — nhưng **2 rank change non-tie**" | **LIVE — must be corrected** (0 rank change non-tie). |
| 13 | `20261109-native15-composition-limn/INPUT_FORWARD_PARITY.vi.md:206-240` | rank-difference tables, "non-tie disagreements" | **LIVE — must be corrected** if the tables are populated from the old artifacts (they are the same measurement). |
| 14 | `20261111-master-audit/MODELS_REST_VERIFICATION.vi.md:330` | "R@1 có mặt (`dress` 21.2692117…)" | **LIVE — must be corrected**, or explicitly marked as the unseeded figure. |
| 15 | `20261110-e2e-repair/FINAL_REPORT.vi.md:211` | `56.2908 / 77.6334 / 51.9584 / 54.9558 / 61.9582` quoted from `raw/native/encoder.json` | **Ambiguous → doc worker decides.** The quote of the stored file is accurate; what is live is any implication that 56.2908 is the model's reproducible measure. Mark not-reproducible per §9. |
| 16 | `20261109-native15-composition-limn/raw/native/encoder.json:12,28` | `76.35101636093208`, `54.9558390578999` | **Historical record — leave.** Stored run artifact; §9 documents it as unseeded. |
| 17 | `raw/encoder_parity/relation_proof_native_pair.txt:2,6` | deltas `−0.0496/0.0992/−0.0496`, `0.102/−0.051/0.0` | **Historical record — leave.** This is the measurement that *established* the two independent draws (proof of the defect). |
| 18 | `raw/encoder_parity/{native,adapter}.{dress,shirt,toptee}.log` | old values throughout | **Historical record — leave.** Logs of the pre-fix run, by definition. |
| 19 | `raw/native/habit.json:15,22` | `21.269211700545366` | **False positive.** HABIT, different checkpoint (`814d2d0c…`), different protocol (`fashioniq_full_gallery_ref_excluded`); numeric coincidence in r1 only. |

## 11. The remaining nine failures

All nine are `*.max_rel` on the two sides' features, and `max_rel` is denominator-unstable at
near-zero reference elements: it divides by the smallest |reference| in the matrix, and a component
with |ref| ≤ 1e-6 turns float32 noise into a large ratio.

| check | `max_abs` | `max_rel` | `max_rel` denominator ≈ |
|---|---|---|---|
| `dress.population.features_vs_native_val_control.max_rel` | 1.125e-06 | 7.500e-01 | 1.5e-06 |
| `dress.population.gallery_vs_native.max_rel` | 1.140e-06 | 1.046e+01 | 1.1e-07 |
| `dress.batch_1_vs_n.max_rel` | 3.353e-07 | 3.176e-03 | 1.1e-04 |
| `shirt.population.features_vs_native_val_control.max_rel` | 1.296e-06 | 4.237e+00 | 3.1e-07 |
| `shirt.population.gallery_vs_native.max_rel` | 1.650e-06 | 5.312e-01 | 3.1e-06 |
| `shirt.batch_1_vs_n.max_rel` | 2.831e-07 | 2.735e-03 | 1.0e-04 |
| `toptee.population.features_vs_native_val_control.max_rel` | 1.002e-06 | 6.278e+00 | 1.6e-07 |
| `toptee.population.gallery_vs_native.max_rel` | 1.013e-06 | 3.133e+00 | 3.2e-07 |
| `toptee.batch_1_vs_n.max_rel` | 3.204e-07 | 1.553e-03 | 2.1e-04 |

`max_abs` is flat against the pre-fix run (dress val-control `1.100e-06 → 1.125e-06`, gallery
`1.132e-06 → 1.140e-06`, batch `3.204e-07 → 3.353e-07`; shirt and toptee likewise within ~2%). The
pin therefore did not degrade the forward agreement — it left it where it was, and the same nine
denominator artefacts remain. **They were neither moved nor deleted**: `forward_max_rel` is still
`1e-4`, `batch_max_rel` still `1e-3`, and the gate still fails on all nine.

### 11b. Three recall vectors in the G2 block, and the one a reader will misread

`adapter.scoring_ranking` carries **three** native-convention recall vectors per category, and two of
them are byte-equal. Verified per category, not assumed from one:

| category | vector | r1 | r10 | r50 |
|---|---|---|---|---|
| dress | `native_convention_recall_on_adapter_features` | 21.269211700545366 | 51.71046108081309 | 76.25185919682697 |
| dress | `native_convention_recall_val_control` | 21.269211700545366 | 51.71046108081309 | 76.25185919682697 |
| dress | `native_convention_recall_native_primary` | 21.120475954387704 | 51.71046108081309 | 76.301437778879517 |
| shirt | `native_convention_recall_on_adapter_features` | 25.564278704612367 | 55.201177625122668 | 75.368007850834147 |
| shirt | `native_convention_recall_val_control` | 25.564278704612367 | 55.201177625122668 | 75.368007850834147 |
| shirt | `native_convention_recall_native_primary` | 25.318940137389596 | 54.90677134445535 | 75.318940137389603 |
| toptee | `native_convention_recall_on_adapter_features` | 29.01580826109128 | 61.601223865374806 | 80.877103518612955 |
| toptee | `native_convention_recall_val_control` | 29.01580826109128 | 61.601223865374806 | 80.877103518612955 |
| toptee | `native_convention_recall_native_primary` | 29.117797042325343 | 62.060173380928099 | 80.979092299847011 |

Equality verdict per category — **the same pattern in all three**, so it generalises:

| category | `on_adapter_features == val_control` | `== native_primary` | `native_mask_only_effect.count` |
|---|---|---|---|
| dress | **yes**, all three cutoffs | no | 1093 |
| shirt | **yes**, all three cutoffs | no | 1132 |
| toptee | **yes**, all three cutoffs | no | 1024 |

`on_adapter_features` and `val_control` are identical because they are the *same quantity*: the
adapter's features scored under the native ranking convention against the val-transform control
population. `on_adapter_features` additionally equals the native artifact's own
`val_transform_control_rank_recall` on all 9 cells. `native_primary` equals the native artifact's
`native_metrics_recheck` on all 9 cells — it is the native pipeline's own primary-population metric.

**The difference between them is the query transform, not the reference mask.** `native_primary` uses
`preprocess_train` (the stochastic `RandomResizedCrop`) for queries; `val_control` uses
`preprocess_val`. Both are reference-masked. The mask-only block separates a different pair: its
`recall_with_mask` reproduces `val_control`/`on_adapter_features` exactly, and its
`recall_without_mask` is the unmasked figure — so `native_mask_only_effect.count`
(1093 / 1132 / 1024) is the count of queries whose rank moves when the `-10e10` reference mask is
applied **within the val-transform control population**, not the cause of the
`native_primary` − `on_adapter_features` delta. That delta is
`−0.1487 / 0.0 / +0.0496` (dress), `−0.2453 / −0.2944 / −0.0491` (shirt),
`+0.1019 / +0.4589 / +0.1019` (toptee).

**The sentence a reader must not misread.** The R@1 that looks *lower* — `~25.32` for shirt against
`~25.56`, and `~21.12` vs `~21.27` for dress — is the same model measured on `preprocess_train`
**query tensors**, i.e. the native pipeline's primary protocol; it is **not** an adapter deficit, and
the adapter is not the weaker side. Both numbers are computed from identical features under the
identical reference mask. Likewise R@50: `75.3189` vs `75.3680` (shirt) is the training-transform
crop, not an adapter loss. The sign is not consistent across categories — `toptee` runs the other way
(`29.1178` primary vs `29.0158` control) — which is the signature of a stochastic crop rather than a
systematic bias.

**No cross-side metric divergence is claimed here.** A reviewer previously suspected one and
retracted it; this subsection does not revive it. The two equal vectors are the same measurement
recorded twice under two names, the third is the other protocol arm, and all three agree with the
native artifact's own corresponding fields on all 9 cells each. Recorded once in
`20261109-native15-composition-limn/ENCODER_PARITY.vi.md:383-385` as Δ R@10 `−0.2975 / +0.2944 /
−0.1530`; see §10 row 4 for that doc's status.

## 12. Run conditions

GPU, necessarily. `BindingDecoder.__init__` calls `.cuda()`, so construction itself cannot run on
CPU — `CUDA_VISIBLE_DEVICES=""` gives `cuda_available False` then
`CONSTRUCTION_FAILED RuntimeError No CUDA GPUs are available`. Launched detached through
`workbench/scripts/gpu_job.sh` (`MAX_WAIT=3600`); the log's first line,
`[gpu_job] lock held, GPU empty, launching`, is the flock being acquired normally (no `MAX_WAIT` /
lock-refused line). Timeline: native dress 20:06:42 → adapter toptee 20:11:25, done 20:11:50.
Intermediates went to `/tmp/encparity_raw` via `ENCODER_PARITY_RAW`; no tracked
`raw/encoder_parity/*.npz` was written (`git status --porcelain` and `git diff --name-only` scoped to
`raw/` both return zero lines).
