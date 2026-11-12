# Paired-parity coverage — measured state

Lane: paired-parity coverage, 2026-11-13. Every number below came from a file written by a command in
this directory (or from the artifact it names), not from `MODEL_MATRIX.md`.

## 1. What was asked, and what this lane actually moved

The acceptance requirement: *a real paired probe for every executable, in-scope configuration and
required category; fixed representative samples unless the contract requires more; a reproducible
blocker for anything whose upstream assets or implementation are genuinely unavailable.*

Delivered in this lane:

| | before | after | evidence |
|---|---|---|---|
| registry models | 17 | 17 | `executability.json:registry_count` |
| models with a **complete** native+adapter pair over all three categories | **3** (`encoder`, `hint`, `limn`) | **4** (`+ habit`) | `pairs_on_disk.json` |
| models with native-side G1 identity evidence only | 0 | 3 (`conesep`, `intent`, `habit` had none; `habit` completed) | `g1_g2.{conesep,intent}.*.native.json` |
| models with a reproducible blocker record | 0 | 7 (`csmcir`, `airknow`, `pair`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`) | `blockers.txt`, `blockers.json` |

**The honest headline: one new complete pair, three new native halves, seven rerunnable blockers.**
Not fourteen new pairs. The lane spent its GPU budget discovering that the shared producer could not
express these models without being extended, and extending it correctly cost more than the probes.

`habits`'s pair is the only one that reached a gate verdict this lane; `conesep` and `intent` carry
G1 identity evidence and **no adapter half**, so the gate reports them `PARITY_EVIDENCE_MISSING` —
incomplete, never a pass.

## 2. Executability, measured (`executability.json`)

Each column names the production function it came from. `EXECUTABLE` requires all four: the declared
`python_env_var` resolves through the symlink to an existing file, `Adapter.status().ready` is true,
at least one declared checkpoint is present on disk, and an adapter module exists.

| model | `python_env_var` | resolves? | checkpoint (path, size) | adapter module | protocol | scope | verdict |
|---|---|---|---|---|---|---|---|
| csmcir | WORKBENCH_CSMCIR_PYTHON | yes → `artifacts/environments/cir-005ef4e1b02999a5/bin/python` | `cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt` **2 720 612 859 B** | exists | fashioniq_original_split | — | `BLOCKED(ADAPTER_NOT_READY)` |
| airknow | WORKBENCH_AIRKNOW_PYTHON | **NO** — unset | `checkpoints/airknow/airknow_fiq_0.5.pt` 2 724 056 638 B; `…0.8.pt` same | exists | fashioniq_full_gallery_ref_excluded | — | `BLOCKED(ENV_VAR_UNSET)` |
| conesep | WORKBENCH_CONESEP_PYTHON | yes → `ilearn39-adapted` | `conesep/ConeSep-FIQ-N0.2.pt` **2 721 428 966 B** (N0.5/N0.8 absent) | exists | fashioniq_full_gallery_ref_excluded | `(None,)` | **EXECUTABLE** |
| ptha_mtst | WORKBENCH_PTHA_MTST_PYTHON | yes → `cir-005ef4e1b02999a5` | `ptha_mtst/tuned_mtst_cirr.pt` **2 720 352 973 B** | exists | fashioniq_original_split | `(None,)` | **EXECUTABLE** |
| habit | WORKBENCH_HABIT_PYTHON | yes → `ilearn39-adapted` | `habit/HABIT-FIQ_N0.2.pt` **2 730 715 930 B** | exists | fashioniq_full_gallery_ref_excluded | `(None,)` | **EXECUTABLE** |
| intent | WORKBENCH_INTENT_PYTHON | yes → `ilearn39-adapted` | `intent/intent_fiq_0.2.pt` **2 722 478 522 B** | exists | fashioniq_full_gallery_ref_excluded | `(None,)` | **EXECUTABLE** |
| hint | WORKBENCH_HINT_PYTHON | yes → `ilearn39-adapted` | `hint/fashioniq.pt` **2 724 451 182 B** | exists | fashioniq_val_split | `(None,)` | **EXECUTABLE** |
| encoder | WORKBENCH_ENCODER_PYTHON | yes → `encoder310-adapted` | `encoder/fashioniq.pt` **631 879 365 B** | exists | fashioniq_val_split | per-category | **EXECUTABLE** |
| pair | WORKBENCH_PAIR_PYTHON | **NO** — unset | `pair/pair-B1.pt` 694 958 565 B, `pair-B2.pt` 694 872 371 B (mapping UNVERIFIED) | exists | fashioniq_val_split | — | `BLOCKED(ENV_VAR_UNSET)` |
| clvc_net | WORKBENCH_CLVC_PYTHON | yes → `clvc39-la` | `checkpoints/clvc/` dir **1 185 968 194 B** | exists | fashioniq_val_split | `(dress,shirt,toptee)` | **EXECUTABLE** |
| dcnet | WORKBENCH_DCNET_PYTHON | **NO** — unset | `checkpoints/dcnet/` dir **2 882 704 059 B** | exists | fashioniq_full_gallery_ref_excluded | — | `BLOCKED(ENV_VAR_UNSET)` |
| combiner_rn50x4_noft | WORKBENCH_CLIP4CIR_PYTHON | **NO** — unset | `…/RN50x4_noft/fiq_comb_RN50x4_noft.pt` **236 017 921 B** | exists | fashioniq_original_split | — | `BLOCKED(ENV_VAR_UNSET)` |
| clip4cir_rn50x4_fullft | WORKBENCH_CLIP4CIR_PYTHON | **NO** — unset | `…/RN50x4_fullft/` dir **949 892 616 B** | exists | fashioniq_original_split | — | `BLOCKED(ENV_VAR_UNSET)` |
| tgcir | WORKBENCH_TGCIR_PYTHON | yes → `tgcir39-la` | `downloads/tgcir/x/checkpoints/Fashioniq.pt` **607 928 637 B** | exists | fashioniq_val_split | `(None,)` | **EXECUTABLE** |
| sprc | WORKBENCH_SPRC_PYTHON | **NO** — unset | `sprc_fiq_vitl.pt` 1 459 756 541 B (paper's `sprc_fiq.pt` absent) | **none declared** | fashioniq_original_split | — | `NO_ADAPTER` |
| limn | WORKBENCH_LIMN_PYTHON | yes → `limn310-adapted` | three `0_<cat>_best_model.pt`, 1 792 010 157 B each | exists | fashioniq_val_split | per-category | **EXECUTABLE** |
| tme | WORKBENCH_TME_PYTHON | yes → `tme-adapted` | `tme/fashioniq_0_best_model.pth` **2 728 382 104 B** | exists | fashioniq_original_split | `(None,)` | **EXECUTABLE** |

Totals: **10 EXECUTABLE · 5 BLOCKED(ENV_VAR_UNSET) · 1 BLOCKED(ADAPTER_NOT_READY) · 1 NO_ADAPTER**
(= 17).

## 3. Pair status per model

| model | pair status | gate status | failures | failure classes |
|---|---|---|---|---|
| encoder | `PAIRED` (pre-existing) | `PARITY_MANDATORY_SUBCHECK_FAILED` | 9 | measured_disagreement |
| hint | `PAIRED` (pre-existing) | `PARITY_MANDATORY_SUBCHECK_FAILED` | 2 | measured_disagreement |
| limn | `PAIRED` (pre-existing) | `PARITY_MANDATORY_SUBCHECK_FAILED` | 3 | measured_disagreement |
| **habit** | **`NEW_PAIR`** | `PARITY_MANDATORY_SUBCHECK_FAILED` | **5** | measured_disagreement (2), absent_evidence_refusal (3) |
| conesep | `G1_ONLY` | `PARITY_EVIDENCE_MISSING` | 0 | — |
| intent | `G1_ONLY` | `PARITY_EVIDENCE_MISSING` | 0 | — |
| csmcir | `BLOCKED(ADAPTER_NOT_READY)` | `NOT_RUN` | — | — |
| airknow | `BLOCKED(ENV_VAR_UNSET)` | `NOT_RUN` | — | — |
| dcnet | `BLOCKED(ENV_VAR_UNSET)` | `NOT_RUN` | — | — |
| combiner_rn50x4_noft | `BLOCKED(ENV_VAR_UNSET)` | `NOT_RUN` | — | — |
| clip4cir_rn50x4_fullft | `BLOCKED(ENV_VAR_UNSET)` | `NOT_RUN` | — | — |
| sprc | `NO_ADAPTER` | `NOT_RUN` | — | — |
| pair | `DEFERRED_FROM_WEB` | `NOT_RUN` | — | — |
| ptha_mtst | `UNDONE(TRANSFER_CHECKPOINT)` | `NOT_RUN` | — | — |
| clvc_net | `UNDONE(NEEDS_TWO_MODEL_HOOK)` | `NOT_RUN` | — | — |
| tgcir | `UNDONE(NEEDS_CLIP_PREPROCESS_HOOK)` | `NOT_RUN` | — | — |
| tme | `UNDONE(NO_SINGLE_CATEGORY_API)` | `NOT_RUN` | — | — |

Those four are not "unknown". Each has a measured contract in `unprobed_contracts.json`, quoted from
its own pinned code, and the seam a future producer must cross:

* **`clvc_net` — needs a distinct hook, not a parameter.** The evaluator drives TWO models and
  concatenates: `third_party/CLVCNet/test.py:48,116` do
  `torch.cat([F.normalize(local_model.extract_img_feature(imgs)[1]), F.normalize(global_model.…[1])], dim=1)`
  → one 2048-d row from two halves. The shared contract is `native_features(paths, texts)` on ONE
  network. Its loader also returns a **3-tuple with no `data_utils`**
  (`replay/clvc_net.py:125` → `(torch, datasets, test_module)`), and its dataset signature has no
  `noise_ratio` (`third_party/CLVCNet/datasets.py:14`).
* **`tme` — no single-category API, so the artifact shape every other pair uses is unexpressible.**
  `replay/tme.py:352-356` raises `TME native evaluator only supports the full FashionIQ category set`
  for any subset, and reads the dataset through a module global
  (`tme_data.FashioniqPath`, `:365`) rather than a handle.
* **`tgcir` — needs `clip.load('ViT-B/16')` preprocess, not `targetpad_transform(1.25, 224)`.**
  Loader `replay/tgcir.py:174`, returning at `:193` as a 3-element tuple whose middle is itself a
  3-tuple; the pinned source is CUDA-hard with no device guard.
* **`ptha_mtst` — a scope decision, not a producer gap.** `cirr_transfer` is a CIRR→FashionIQ
  transfer checkpoint (`reproduction_status: SELECTED_TRANSFER_ONLY`); a pair would measure a
  configuration the campaign does not treat as a FashionIQ reproduction. That call belongs to the
  acceptance owner.

### HABIT's five failures, verbatim from `gate.json`

| check | class | declared | measured |
|---|---|---|---|
| `dress.ranks.evidence_present` | absent_evidence_refusal | a rank comparison beside the score comparison | no rank comparison recorded |
| `shirt.batch_1_vs_n.max_rel` | measured_disagreement | 1e-3 | 0.0050964738236841 |
| `shirt.ranks.evidence_present` | absent_evidence_refusal | a rank comparison beside the score comparison | no rank comparison recorded |
| `toptee.batch_1_vs_n.max_rel` | measured_disagreement | 1e-3 | 0.006168472315303178 |
| `toptee.ranks.evidence_present` | absent_evidence_refusal | a rank comparison beside the score comparison | no rank comparison recorded |

`missing`: `["metrics_parity"]` in **all three** categories. Reason, and the lane MUST NOT paper over
it: HABIT has no stored native per-query-rank measurement, so the adapter's ranks cannot be compared
against a native *measurement*. The producer therefore records the convention checks it can make and
declares `native_rank_control_absent` rather than substituting the adapter's own output for a control.
`rating` is `INCOMPLETE` on that dimension — the honest answer, not a "clean" one.

Both `batch_1_vs_n.max_rel` failures are the **same vanishing-denominator class** as encoder's 9 and
limn's 3 (LEDGER L10): `max_abs` is 1.6e-07 against a declared 1e-4 atol. **No tolerance was moved.**

Measured `batch_1_vs_n.max_rel` per category after the final rewrite: dress **0.00016192595832636484**
(inside 1e-3, so no failure), shirt **0.0050964738236841**, toptee **0.006168472315303178**. Sample
forward `max_abs` is **0.0** in all three and `cache_off_restart.bitwise_equal` is **true** in all
three, so the adapter's own worker is reproducible and the failure is confined to batch-size reduction
order.

**The adapter artifacts were written three times.** This lane relaunched the HABIT adapter batch twice
while an earlier invocation was still queued behind the GPU lock, so two redundant jobs ran after the
first one succeeded (artifact mtimes 04:27/04:30/04:32, then 04:35/04:38; `pgrep` confirms no job
remains). `gate.json` was regenerated against the final bytes at the end of that sequence and the
verdict is **unchanged across all three writes** — same five failures, same measured values to every
digit printed above. That reproducibility is itself evidence about the measurement: the adapter's
worker returns the same features on every run (`cache_off_restart.bitwise_equal: true`).

The duplicate runs were waste, not correctness: each cost ~130-180 s of exclusive GPU. Recorded here
rather than quietly cleaned up, because a lane that double-books the GPU lock has done exactly what the
one-job-at-a-time rule exists to prevent, even though the lock itself was respected.

## 4. New artifacts

All under `workbench/docs/reviews/20261109-native15-composition-limn/`.

| artifact | interpreter | phase | sample | result |
|---|---|---|---|---|
| `g1_g2.habit.{dress,shirt,toptee}.native.json` | `ilearn39-adapted` | identity+native | 5 queries / 20 gallery images, 3 categories | G1 all-true; native forward measured |
| `g1_g2.habit.{dress,shirt,toptee}.adapter.json` | `ctl312h` (controller) | adapter | same fixed sample, **full** population encode | see §3 |
| `g1_g2.intent.{dress,shirt,toptee}.native.json` | `ilearn39-adapted` | identity+native | 5 / 20 | G1 all-true |
| `g1_g2.conesep.{dress,shirt,toptee}.native.json` | `ilearn39-adapted` | identity+native | 5 / 20 | G1 all-true |

Sample size and why it is enough for what is claimed: 5 query slots spent on the properties the sample
exists for (two rows sharing one reference — a per-query max-over-targets metric must get that right;
the last row, which is the only one a partial final batch reaches; a capitalised first caption; a
second caption under 12 characters), plus 20 gallery images chosen so the image batch is itself
partial. It is enough to prove the *forward and the convention*, and it is **not** enough to claim a
population verdict: `population_scope` is recorded in every artifact and only `habit` was encoded over
the whole category.

G1 identity, identical on all nine native artifacts and re-derived, not copied:
`query_count_native` 2017/2038/1961 = `query_count_annotations`; `gallery_count_native` 3817/6346/5373 =
`gallery_count_rebuilt_from_annotations`; `gallery_order_identical`, `reference_ids_identical`,
`target_ids_identical`, `model_text_identical_to_author_composition` all **true**.

### GPU job lock lines (`verbatim`)

```
[gpu_job] lock held, GPU empty, launching: workbench/envs/ilearn39-adapted/bin/python -u workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py --model habit --phase native --category dress
GPU-9bc596d9-19e9-6f43-dc2c-a80da924263c, 758 MiB, 16303 MiB
```

Native phases: dress 44.6 s, shirt 49.6 s, toptee 48.3 s (habit); 47.8/52.3/41.9 s (intent);
41.2/42.1/36.1 s (conesep). Adapter phases: dress 146.7 s, shirt 180.5 s, toptee 159.0 s. Every job ran
through `gpu_job.sh`; the lock was never bypassed, deleted, or its holder killed. Two of this lane's
own jobs were refused with `[gpu_job] lock held; retrying in 30s` while a sibling held it, and waited.

## 5. Blocker records (`blockers.txt`, `blockers.json`)

Every blocker is a command plus its verbatim output. Rerunnable by copy-paste.

### (a) One removed conda env, five declared interpreters — `INTERPRETER_BROKEN_SYMLINK`

```
$ readlink -f workbench/envs/airknow-adapted/bin/python
exit=1
```
The five declared symlinks all point at the same absent target:
`airknow-adapted`, `pair-adapted`, `sprc-adapted`, `clip4cir-adapted`, `limn-adapted` →
`/home/heheboiz/miniconda3/envs/teacher-sprc/bin/python`.
Asset missing: that interpreter. Not repaired here — the operator's environment is not this lane's.

Note `limn-adapted` is dangling while LIMN **is** executable: the registry resolves LIMN to
`limn310-adapted`, a different, working env. The dangling one is a leftover alias, which is why the
verdict for LIMN is `EXECUTABLE` and for `airknow` is not.

### (b) Unset interpreter variables — `PYTHON_ENV_VAR_UNSET`

```
$ grep -n -E 'WORKBENCH_(AIRKNOW|PAIR|DCNET|CLIP4CIR|SPRC)_PYTHON' workbench/config/workbench.env
exit=1
```
`exit=1` and no output: the variables are absent from the operator's env file. `workbench.env` was
**not** modified. Models affected: `airknow`, `pair`, `dcnet`, `combiner_rn50x4_noft`,
`clip4cir_rn50x4_fullft`, `sprc`.

### (c) CSMCIR — `BLOCKED(ADAPTER_NOT_READY)`, and it is a science blocker, not an env one

```
$ workbench/envs/ctl312h/bin/python raw/parity-coverage/probe_csmcir.py
```
Preflight: `[]`. Interpreter resolves, checkpoint present (2 720 612 859 B), mapping verified. The
adapter refuses, verbatim from `Adapter.status()`:

> CSMCIR cannot run through the composition adapter: the adapter's score() is a 256-d text embedding
> expanded x32 into an 8192-d dot product, whereas the native evaluator's inference() uses
> attention-weighted token importance and a max over 32 query tokens. Those are different quantities,
> so adapter scores would be a substitution, not the measured contract. […] local captions cover only
> the FashionIQ validation split (15,415 of 74,381 CIRCUS gallery images, or 20.72%), leaving 58,966
> images uncaptioned.

This is the "adapter cannot record the relation" case: no paired probe is possible because the
adapter's scoring is a **different quantity** from the native evaluator's. Reported as missing, not
clean.

### (d) SPRC — `NO_ADAPTER`

`workbench/registry/models.yaml` declares no `composition_adapter` for SPRC, and
`workbench/composition/` has no `model_sprc.py`. The class does not exist. Additionally the paper's
ViT-g checkpoint (`sprc_fiq.pt`) is absent home-wide; only `sprc_fiq_vitl.pt` (1 459 756 541 B) is
present.

### (e) PAIR — `DEFERRED_FROM_WEB`, not a new failure

```
$ workbench/envs/ctl312h/bin/python -c "... web_deferred_models(load_registry())"
{"pair": "Chưa có mapping checkpoint chính thức: pickle được tạo bởi các class mà repo đã pin không có (BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED), …"}
```
Kept as `DEFERRED_FROM_WEB` with its existing blocker codes
(`BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` + `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`). **PAIR was not run.**

## 6. What this lane changed in the shared producer, and why that was necessary

`workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py` was extended with
**parameters, not a fork**. HINT remains its first instance and its behaviour is unchanged
(`--self-check` green before and after; HINT's artifacts were not rewritten).

Added: `ModelScope.native_control_path`, `.population_scope`, `.processor_config`, `.protocol_id`;
`CaptureHook.dataset_kwargs()`; a generic `LavisFashionIQCapture` hook; `load_replay_modules()`
(signature-aware, because HINT's loader takes 2 arguments and the LAVIS-family ones take 1);
`_scopes()` + `--model`/`--list-models`; and per-model namespacing of every `raw/third_model` scratch
path.

Three defects were found by running it, and each is recorded rather than hidden:

1. **`native_phase` OOM-killed** at 27.5 GiB peak. Cause: it constructed `FashionIQ` with
   `targetpad_transform(1.25, 224)`, which decodes and transforms every query and gallery image of all
   three categories, while the hooks open their own images from paths. Fixed by passing an identity
   transform (HABIT's `datasets.py:157` calls `self.transform(img)` unconditionally, so `None` raises
   `TypeError`). The prior, committed HABIT native run recorded the same **27.54 GiB** peak.
2. **`decode_equality` defaulted to HINT's checkout**, so a HABIT run died with
   `ImportError: this process already prepared the lavis context for …HABIT/lavis`. Now takes the hook.
3. **`update_native_metrics_flag` crashed** when there was no native rank control. It now *withholds*
   the flag and records `native_rank_control_absent`, because a bare `False` would read as a measured
   disagreement between count vectors that do not exist.

## 7. Tests

```
$ workbench/envs/ctl312h/bin/python -m pytest workbench/tests -k parity -q
148 passed, 3179 deselected, 6 warnings in 41.33s
```

Zero failures, measured **after** HABIT's six new artifacts landed, so the gate's discovery path
(`_discover_models` picks HABIT up from its own filenames) is exercised by the existing suite without
an edit. No test file under `workbench/tests` was touched by this lane, and no test was deleted or
relabelled.

## 8. Files touched, and nothing else

Written: this directory (`raw/parity-coverage/*`), and new
`workbench/docs/reviews/20261109-native15-composition-limn/g1_g2.{habit,intent,conesep}.*.{native,adapter}.json`
plus their scratch under `raw/third_model/`.

Modified: `workbench/docs/reviews/20261112-parity-repair/scripts/third_model_parity.py` only.

Not touched: `workbench/backend/*`, `workbench/frontend/*`, `workbench/composition/*`,
`workbench/scripts/*`, `workbench/registry/models.yaml`, `workbench/config/workbench.env`,
`workbench/tests/*`, the pre-existing `g1_g2.{encoder,hint,limn}.*` artifacts, `CLEANUP_REPORT.md.save`.
Nothing committed; nothing staged.

## 9. What is still missing, and why it is not blank

Four `EXECUTABLE` models have no pair: `ptha_mtst`, `clvc_net`, `tgcir`, `tme`. Each has a measured
contract in `unprobed_contracts.json`, quoted from its own pinned code:

| model | blocker on a pair | seam a future producer must cross |
|---|---|---|
| `clvc_net` | the evaluator drives TWO models and concatenates their features (`third_party/CLVCNet/test.py:48,116`) | a two-model capture hook — not another `ModelScope` field |
| `tme` | `replay/tme.py:352-356` raises on any category subset: "no single-category API" | a whole-three-category artifact shape, which no existing pair uses |
| `tgcir` | loader returns a 3-element tuple (`replay/tgcir.py:174`, return `:193`) and needs `clip.load('ViT-B/16')` preprocess, not `targetpad_transform(1.25, 224)` | a CLIP-preprocess-aware hook plus a device guard |
| `ptha_mtst` | `cirr_transfer` is a CIRR→FashionIQ transfer checkpoint (`SELECTED_TRANSFER_ONLY`) | a scope decision by the acceptance owner, not a producer change |

The lane's own honest accounting: it produced **one** new complete pair (`habit`) and **three** new
G1-native halves (`habit`, `intent`, `conesep`), not the fourteen the coverage gap nominally asked for.
The GPU budget went to making the shared producer able to express a second model family at all, and
three defects in it were found and fixed along the way (§6). Anything beyond `habit` needs the hook
shapes listed above, which are design work rather than more probe time.

---

# Follow-up round (same lane, 2026-11-13)

## F1 — HABIT's `ranks.evidence_present` gap CLOSED (5 failures → 2)

The three `absent_evidence_refusal` failures were a producer gap, not a model property: the adapter
holds its own ranks, so ordering was measurable without a native control.

**What was implemented.** `scoring_ranking`'s control-less branch now records the comparison
`parity_gate._rank_checks` reads — adapter production `rank()` vs `numpy argsort(-sims)` applied to the
SAME adapter features — with each difference classified against `tie_epsilon`. Keys:
`unexplained_rank_differences`, `rank_differences`, `first_unexplained_index`, `rank_comparison`.

**A mistake I made and corrected before recording the number.** My first version compared
`adapter_style` (which does NOT apply the reference mask, `model_habit.py reference_eligible = True`)
against the MASKED native convention. That produced **663 "unexplained differences" on dress** — the
mask, not an ordering defect. HINT's own artifact proves which convention the adapter reproduces
(`adapter_vs_argsort_unmasked_difference_count: 26`, not 0). Corrected to the unmasked basis on both
sides before any verdict was recorded.

| category | rank differences | unexplained | verdict |
|---|---|---|---|
| dress | 28 | **0** | measured, clean |
| shirt | 13 | **0** | measured, clean |
| toptee | 13 | **0** | measured, clean |

Regenerated: `g1_g2.habit.{dress,shirt,toptee}.adapter.json` (one `gpu_job.sh` launch per phase, one
log `/tmp/habit_adapters_all.log`; no duplicate launches this round).

**Gate before → after:** HABIT `PARITY_MANDATORY_SUBCHECK_FAILED` with **5** failures
(2 `measured_disagreement` + 3 `absent_evidence_refusal`) → **2** failures, both
`measured_disagreement`, `missing: ["metrics_parity"]` unchanged:

| check | declared | measured |
|---|---|---|
| `shirt.batch_1_vs_n.max_rel` | 1e-3 | 0.0050964738236841 |
| `toptee.batch_1_vs_n.max_rel` | 1e-3 | 0.006168472315303178 |

The three checks became **measured and ok**. They did not become passing-by-declaration: the value is a
derived count, and a non-zero count would have failed.

## F2 — `tgcir` PAIRED (new complete pair)

**The hook contract added (quoted from the producer).** `ClippedModelCapture(LavisFashionIQCapture)`,
declared entirely as scope data plus two overrides — no branch inside the shared phases:

```
transform_source: "clip"   # clip.load('ViT-B/16', jit=False).preprocess (train.py:87,90)
                           # NOT data_utils.targetpad_transform(1.25, 224)
native_device:    "cuda"
restart_supported: False   # model_tgcir.py:344 _check_worker_pid raises adapter_worker_reloaded
population_scope: "sample"
```

and, on the hook: `replay_modules()` (overridden to flatten `_load_tgcir`'s nested 3-tuple),
`dataset_kwargs()` (no `noise_ratio`), `gallery()` (union of source+target ids in first-appearance
order — 2628 for dress, not 3817), `load_native()` (`torch.load(weights_only=False)` + the
`batch_first` pickle patch), and CLIP-preprocess forwards.

Five real defects were exposed by running it, each fixed in the producer rather than in a copy:
`_load_replay_modules` does not exist for tgcir; `_load_tgcir`'s index 1 is a tuple; the dataset takes
no `noise_ratio`; the checkpoint lives at `downloads/tgcir/x/checkpoints/` so `checkpoint_identity`
and `preflight` must use the REGISTRY's resolved path (also fixed for every model); and the gallery is
the union, not the full split list (`gallery_order_identical: false` on the first identity run was a
true statement about my hook). A sixth: the scoring path hardcoded HINT's 3-D gallery and raised
`ValueError: einstein sum subscripts…`, now rank-aware (2-D and 3-D).

**Gate verdict:** `PARITY_MANDATORY_SUBCHECK_FAILED`, **3** failures, all `measured_disagreement`,
`missing: ["metrics_parity"]`:

| check | declared | measured |
|---|---|---|
| `dress.batch_1_vs_n.max_rel` | 1e-3 | 0.24029609477008945 |
| `shirt.batch_1_vs_n.max_rel` | 1e-3 | 0.07048293904534349 |
| `toptee.batch_1_vs_n.max_rel` | 1e-3 | 0.029507544996198128 |

`batch_parity` is measured; **`metrics_parity` is missing**, because tgcir has no stored native
per-query-rank control, and `population_scope: "sample"` means the component scope is a PREVIEW.
**This pair is INCOMPLETE, not complete coverage** — the honest reading of "chunked".

Lock lines (each phase its own `gpu_job.sh` launch):

```
[gpu_job] lock held, GPU empty, launching: workbench/envs/tgcir39-la/bin/python -u … --model tgcir --phase identity --category dress
[gpu_job] lock held, GPU empty, launching: workbench/envs/tgcir39-la/bin/python -u … --model tgcir --phase native --category dress
[gpu_job] lock held, GPU empty, launching: workbench/envs/ctl312h/bin/python -u … --model tgcir --phase adapter --category dress
```

## F3 — `clvc_net` NOT attempted

Left `BLOCKED(NEEDS_A_DISTINCT_HOOK)`. The reason is the one `unprobed_contracts.json` records: the
evaluator drives TWO networks and concatenates their per-half-normalised features
(`third_party/CLVCNet/test.py:48,116`), a shape the shared contract does not express as a parameter.
Generalising it would mean a new native-capability abstraction on the same footing as `transform_source`
and `restart_supported`; that is design work, and a bespoke branch inside `third_model_parity.py` was
explicitly the wrong outcome. `tme` stays `BLOCKED(NO_SINGLE_CATEGORY_AND_NO_PER_SAMPLE_API)`.
`ptha_mtst` recorded as `SCOPE_DECISION` (no pair: `SELECTED_TRANSFER_ONLY`); not run.

## F4 — counts after this round

| | before lane | after F1 | after F2 |
|---|---|---|---|
| complete pairs (native+adapter, all 3 categories) | 3 | 4 | **5** |
| models with any measured pair | 3 | 4 | **5** |
| HABIT failures | — | 5 | **2** |
| tgcir | no artifact | no artifact | **3 failures, INCOMPLETE** |

Complete pairs: `encoder`, `habit`, `hint`, `limn`, `tgcir` (`pairs_on_disk.json`).
