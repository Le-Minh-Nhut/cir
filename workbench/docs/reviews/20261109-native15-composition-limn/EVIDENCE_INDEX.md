# EVIDENCE INDEX — native 15 + CIRCUS composition 8

Session: 2026-11-09. Branch `research/cir-failure-workbench`.
Read-only on production source. No GPU evaluation run by this workstream.
Control interpreter: `workbench/envs/ctl312h/bin/python` (CPython 3.12.11).

Files produced here: `NATIVE_EVAL_MATRIX.csv`, `COMPOSITION_MATRIX.csv`, `GATES.csv`,
`EVIDENCE_INDEX.md`, and helper scripts/artifacts under `scripts/`.

---

## 1. Count reconciliation (established from source, not copied)

| Quantity | Value | How established |
|---|---|---|
| Models declared in registry | **17** | `yaml.safe_load(workbench/registry/models.yaml)['models']` → `len == 17` (`scripts/dump_registry.py`, line `COUNT 17`) |
| Active native targets | **15** | 17 minus the 2 user-deferred (`csmcir`, `sprc`) |
| Deferred | **2** | `csmcir`, `sprc` — kept visible in the matrix with `user-deferred` reason, not deleted |
| Have a `composition_adapter` | **16 of 17** | `[m for m in models if m.get('composition_adapter')]` → 16. The **one missing is `sprc`** (verified, not assumed). So "16 of 17" is correct. |
| In the selected composition eight | **8** | `encoder, conesep, habit, intent, hint, limn, ptha_mtst, tme` — each has a `full.json` whose `run_id` is a 24-hex id from the 2026-11-09 clean rerun |

### Composition summary directory accounting (22 dirs on disk)
`workbench/artifacts/composition/summary/` holds **22** `full.json`-bearing dirs, not 8:
- **8** are the selected 2026-11-09 clean-rerun runs (`COMPOSITION_MATRIX.csv`).
- **11** are older/alternate real runs of the same 8 models (e.g. `encoder_fashioniq_6a528e87…`,
  `limn_…_10173dc4…`, `ptha_mtst_…_315dd590…`). They are superseded by the selected run and are
  not dropped — they remain on disk and are named here.
- **2** are `SYNTHETIC_*` fixtures (`is_synthetic=true`, `reproduction_status=FIXTURE_DEVELOPMENT_ONLY`).
- **1** is `released_audit_task7_mm_any` (a release audit, `model_id=released_audit`).

So 8 + 11 + 2 + 1 = 22. The task's "8 relevant dirs" refers to the selected set only; the other 14
are not silently deleted, they are classified above.

### One discrepancy to state plainly
The task brief said "16 of 17 have a composition adapter." That is **true** — but the odd one out is
`sprc`, not any of the 15 active targets. All 15 active native targets have a composition adapter;
`sprc` (deferred) is the only model without one. `csmcir` (deferred) **does** have one.

---

## 1a. Expected query count — MEASURED, not assumed

Command actually run:
```
$ workbench/envs/ctl312h/bin/python -c "import json; [print(c, len(json.load(open(f'data/FashionIQ/captions/cap.{c}.val.json')))) for c in ('dress','shirt','toptee')]"
dress 2017
shirt 2038
toptee 1961
SUM 6016
```
Measured per-category val annotation counts: **dress 2017, shirt 2038, toptee 1961 → 6016**.
(Distinct from the CIRCUS release cohort `EXPECTED_QUERY_COUNTS["full"]=6003` in
`workbench/composition/data.py:16`; 6016 vs 6003 is a cohort difference, not a defect.)

### Per-model expected count — sourced from each evaluator's own manifest read
Every native evaluator below was inspected and **all read the same `cap.{cat}.val.json` manifests**
(each iterates every annotation row as one query; none deduplicates queries). DCNet reads the
`glove` pkl derived from those same annotations — measured `cap.*.glove.val.pkl` = 2017/2038/1961 =
6016. So **no row needed `unknown`**; each carries its own citation:

| model | manifest read (file:line) | count |
|---|---|---|
| airknow | `AirKnow/datasets.py:185` cap.{name}.val.json | 6016 |
| conesep | `ConeSep/datasets.py:168` | 6016 |
| ptha_mtst | `MTST_PTHA/.../data_utils.py:158` | 6016 |
| habit | `HABIT/datasets.py:172` | 6016 |
| intent | `INTENT/datasets.py:170` | 6016 |
| hint | `HINT/datasets.py:135` | 6016 |
| encoder | `ENCODER/datasets.py:161` | 6016 |
| pair | `PAIR/data/datasets.py:120` | 6016 |
| clvc_net | `CLVCNet/datasets.py:168` (split=val) | 6016 |
| dcnet | `DCNet/data_loader/CE_dataset.py:31` cap.{cat}.glove.val.pkl | 6016 (measured pkl) |
| combiner_rn50x4_noft | `CLIP4Cir/src/data_utils.py:140` | 6016 |
| clip4cir_rn50x4_fullft | `CLIP4Cir/src/data_utils.py:140` | 6016 |
| tgcir | `TGCIR/datasets.py:129` | 6016 |
| limn | `LIMN/LIMN/datasets.py:139` | 6016 |
| tme | `TME/src/data.py:158` | 6016 |

Note: `fashioniq_val_split` models still count **queries** from `cap.*.val.json` (6016); they differ
only in **gallery** construction (pair-union vs full split), not query count. `fashioniq_original_split`
and `full_gallery_ref_excluded` also use the full annotation set as queries. Hence the 6016 applies
uniformly — and it is now traced to each evaluator, not hard-coded.

## 2. Native execution vs verification — the distinction enforced

### 2.0 UPDATE 2026-11-09 (worker `native-exec` + `limn-parity`) — 9 of 15 active now executed

Re-read this section: the counts below supersede the earlier "only combiner executed" state.

**Completed native runs (verified from artifacts this session):**

| model | protocol | macro R@10 | macro R@50 | wall | peak RSS | artifact |
|---|---|---|---|---|---|---|
| encoder | fashioniq_val_split | 56.2908 | 77.6334 | 131 s | 18.63 GiB | `raw/native/encoder.json` |
| hint | fashioniq_val_split | 59.8034 | 80.1308 | 271 s | 20.51 GiB | `raw/native/hint.json` |
| habit | fashioniq_full_gallery_ref_excluded | 54.8733 | 74.5034 | 421 s | 27.54 GiB | `raw/native/habit.json` |
| intent | fashioniq_full_gallery_ref_excluded | 54.8662 | 75.4319 | 440 s | 26.11 GiB | `raw/native/intent.json` |
| conesep | fashioniq_full_gallery_ref_excluded | 54.7613 | 75.2151 | 441 s | 27.05 GiB | `raw/native/conesep.json` |
| tme | fashioniq_original_split | 54.9229 | 74.8384 | 120 s | 25.54 GiB | `raw/native/tme.json` |
| ptha_mtst | fashioniq_original_split | **37.8978** | **59.8364** | 240 s | 25.79 GiB | `raw/native/ptha_mtst.result.json` |
| combiner_rn50x4_noft | fashioniq_original_split | 35.1078 | 58.4756 | — | — | `workbench/artifacts/reports/…020a70b0_aggregate.json` |
| limn | fashioniq_val_split | 55.7919 | 77.6378 | — | — | `limn_native.json` |

All 9 were re-read: each of the 7 new files shows 3 categories with `complete=true` and
`parity_mismatches=[]`; `ptha_mtst.result.json` shows `strict_load=true` with an empty
`load_report`.

**`ptha_mtst` is a TRANSFER, not a FashionIQ reproduction.** Its file is labelled
`"CIRR checkpoint -> FashionIQ | cross-dataset transfer | exploratory"`, `train_dataset: CIRR`.
`paper_fashioniq_context_only` reports paper R@10 54.99 vs observed 37.90 = **−17.09** gap. It must
never be read as a FashionIQ paper row.

**`native_verified` is `Y` for LIMN only.** The other 8 completed runs have G1/G2 `NOT_RUN`; a
completed run is not a verified reproduction.

**Counting:** 7 (`native-exec`) + `combiner_rn50x4_noft` + `limn` = **9 of 15 active**. `csmcir`
also has a historical completed run but is **deferred**, so it is not counted in the active 15.
(The brief's "7 + combiner = 8" omits LIMN, which also has a genuine native run.) Still without a
completed native run: airknow, pair, clvc_net, dcnet (ran, FAILED), clip4cir_rn50x4_fullft, tgcir
= **6**. Hence `Native completeness = FAIL` (gate is "all 15", not "most").

**LIMN G2 nuance — do not flatten to "G2 PASS".** G1 PASS (inputs exact). G2 PASS on **absolute**
error (sample features bitwise 0.0; score `max_abs` 5.4e-07) and on **ranking** (0 rank
differences, 47 exact ties). But the **batch-size 1-vs-16 relative sub-check FAILS its own
pre-declared `max_rel ≤ 1e-3`** (measured 1.14e-2) for a documented `|reference|`-denominator
reason, and the worker explicitly **did not move the tolerance**. Recorded as
`g2_status = PASS-with-caveat`.

**LIMN dual-scope gate = PASS.** `limn_dual_scope.json` holds 2 query IDs (7:3038, 7:3790) × 2
scopes (shirt, toptee) × 3 conditions = **12 records**, each with a `top100_export` JSONL.
Verified all 12 JSONLs: 100 ordered candidates (rank 1..100), `depth=100`, target rank present
(e.g. 7:3038 shirt/mm rank 3, toptee/mm 43; 7:3790 shirt/mm 65, toptee/mm 43). The canonical metric
stays 6001/6003 — both queries remain unscored there, no diagnostic number folded in.

---

## 2.1 Original framing (still true for the remaining 6)

- `native_executed = Y` **only** when a model's own upstream evaluator ran to `EXECUTION_COMPLETED`
  and produced an aggregate report on disk.
- `native_verified = Y` **only** when G1 (input parity) and G2 (forward/scoring parity) are both
  established. As of the 2026-11-09 update, **only LIMN** reaches this (`g2_status =
  PASS-with-caveat`); every other row is `N`.

Genuine completed native run (first one, before the 2026-11-09 batch): **`combiner_rn50x4_noft`**
(`run_id …020a70b0`), R@10 35.1078 / R@50 58.4756, 3 category metrics present, `return_code 0`,
`completion_status COMPLETED`, `verdict PROOF_VALIDATED`. `native_executed=Y, native_verified=N` —
a run exists, but no published reference row describes the experiment
(`benchmark_gate METRIC_REFERENCE_UNVERIFIED`), so parity cannot be asserted.

`csmcir` also has a historical `EXECUTION_COMPLETED` run (`…f03887d2`, R@10 57.0507 / R@50 77.2986)
but is **user-deferred**, not part of the active 15.

`dcnet` executed but `EXECUTION_FAILED` (see §3) — not a completed run.

---

## 3. Stale-blocker revalidation — commands and real output

All commands below were run in this session. Where output garbled in the shell, it was redirected to
a file and read back.

### 3.1 `dcnet` — `weights_only` is a torch-signature fact, not a corrupt checkpoint
```
$ grep -n "weights_only" workbench/third_party/DCNet/base/base_trainer.py
196:        # torch >= 2.6 refuses under the new default `weights_only=True`. The checkpoint is
199:        checkpoint = torch.load(ckpt_path, weights_only=False)

$ workbench/artifacts/environments/dcnet-377/bin/python -c "import torch,sys,inspect; print(sys.version.split()[0], torch.__version__, 'weights_only' in inspect.signature(torch.load).parameters)"
3.7.7 1.4.0+cpu False
```
The aggregate's real traceback (`stderr.log` of run `…f969aef3`):
```
File ".../base/base_trainer.py", line 199, in _resume_checkpoint
    checkpoint = torch.load(ckpt_path, weights_only=False)
TypeError: 'weights_only' is an invalid keyword argument for load()
```
**Current truth:** the approved patch `workbench/patches/dcnet_checkpoint_weights_only_optout.patch`
adds `weights_only=False` to support torch ≥ 2.6, but the pinned env is **torch 1.4.0**, whose
`torch.load` has **no** `weights_only` parameter (`False`). So the patch breaks the pinned env. This
is a registry/patch two-way contradiction, **not** a corrupt checkpoint. Checkpoint digests verify
clean (`dcnet_dir a7f9ce0e…` = aggregate `checkpoint_sha256`).

### 3.2 `clvc_net` — real `.cuda()` count on the scoring path
```
$ grep -n "\.cuda()" workbench/third_party/CLVCNet/test.py workbench/third_party/CLVCNet/model.py workbench/third_party/CLVCNet/text_model.py
test.py:28   imgs = torch.autograd.Variable(imgs).cuda()
test.py:47   imgs = torch.autograd.Variable(imgs).cuda()
test.py:99   imgs = torch.stack(imgs).float().cuda()
test.py:115  imgs = torch.stack(imgs).float().cuda()
text_model.py:96    itexts = torch.autograd.Variable(itexts).cuda()
text_model.py:108   first_hidden = (first_hidden[0].cuda(), first_hidden[1].cuda())
model.py:178 labels = torch.autograd.Variable(labels).cuda()   (loss path, not scoring)
model.py:249 labels = torch.autograd.Variable(labels).cuda()   (loss path, not scoring)
```
Whole repo: **14** occurrences across 4 files (`test.py` 4, `model.py` 2, `text_model.py` 2,
`train.py` 6). On the **scoring path** (`test.py::test` → `compose_img_text` →
`TextLSTMModel.forward_encoded_texts`/`forward_lstm_`): **6 unguarded `.cuda()`**
(test.py:28,47,99,115 and text_model.py:96,108). The two in `model.py` and six in `train.py` are
training/loss paths, not the scoring path. There is **no interpreter/env** for `clvc_net` (registry
declares py3.7.6/torch1.6.0, but no venv exists).

### 3.3 `combiner_rn50x4_noft` / `clip4cir_rn50x4_fullft` — interpreter exists; validate.py output shape
```
$ ls -la workbench/artifacts/environments/clip4cir-38/bin/python
... -> .../cpython-3.8-linux-x86_64-gnu/bin/python3.8
$ workbench/artifacts/environments/clip4cir-38/bin/python -c "import sys,torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
3.8.20 1.11.0+cpu False
```
**`validate.py` prints aggregate only**, per category — not per-query. The FashionIQ branch prints
`shirt/dress/toptee` R@10/R@50 and `average recall10/recall50` (`validate.py:344-354`); the CIRR
branch prints group/recall scalars (`:317-323`). No per-query rows are printed. This is confirmed by
the real combiner stdout: 8 metric lines + average, no per-query listing.

### 3.4 `tgcir` — pinned `test.py` has no `__main__`/argparse/loader
```
$ grep -c "__main__"        workbench/third_party/TGCIR/test.py  → 0
$ grep -c "argparse"        workbench/third_party/TGCIR/test.py  → 0
$ grep -c "load_state_dict" workbench/third_party/TGCIR/test.py  → 0
$ grep -c "torch.load"      workbench/third_party/TGCIR/test.py  → 0
```
`test.py` is a bare `def test(params, model, testset, category)` with unguarded `.cuda()` in the
scoring loop. `workbench/replay/tgcir.py` supplies the shell. No interpreter/env exists (`python:
UNKNOWN` in registry; no `WORKBENCH_TGCIR_PYTHON` in `workbench.env`). Checkpoint member
`downloads/tgcir/x/checkpoints/Fashioniq.pt` exists, sha `c37768f3…` (matches registry).

### 3.5 `airknow` — usable interpreter? torch imports; forward pass blocked
```
$ workbench/envs/airknow-adapted/bin/python -c "import sys,torch; print(sys.version.split()[0], torch.__version__, torch.cuda.is_available())"
3.11.15 2.7.1+cu128 True
```
Registry declares **Python 3.8.10**; the adapted env is **3.11.15** (version mismatch). torch,
transformers, timm, fairscale, peft, diffusers, iopath, omegaconf, sklearn all import
(`scripts/probe_airknow.out`). But the actual evaluator graph fails:
```
$ cd workbench/third_party/AirKnow && python -c "sys.path+['.','lavis']; import lavis.models"
lavis.models FAIL ModuleNotFoundError("No module named 'moviepy'")
```
(`scripts/probe_airknow2.out`; `datasets` and `test_BLIP2` do import). So a **forward pass is not
possible** as provisioned — `lavis.models` cannot import (missing `moviepy`, a transitive dep not in
the registry contract). No native aggregate artifact exists.

### 3.6 `pair` — key/shape claim re-derived (CORRECTION: earlier log was a clobbered redirect)

**Correction notice.** An earlier version of `scripts/pair_rederive.out` was a clobbered shell
redirect — it contained git output (`clean — nothing to commit`), not script stdout, and
`scripts/pair_numbered.txt` was a grep result about a `FutureWarning`. Both were invalid evidence and
have been replaced/removed. The run was repeated with logging written **from inside Python**
(a `Tee` on `sys.stdout`/`sys.stderr`), and the real log is now `scripts/pair_rederive.out`
(21 lines of actual key-count output, first lines below).

Real log, first 15 lines (`scripts/pair_rederive.out`):
```
model_vssp files in pinned repo: ABSENT
MHTransformer_dis files in pinned repo: ABSENT
.../transformers/utils/generic.py:441: FutureWarning: ... (benign import warning)
  _torch_pytree._register_pytree_node(
.../transformers/utils/generic.py:309: FutureWarning: ... (benign import warning)
  _torch_pytree._register_pytree_node(
ROOT_CLASS: model.PAIR
STATE_DICT_ENTRIES: 455
UNIQUE_TENSOR_OBJECTS_IN_STATE_DICT: 455
NAMED_PARAMETERS: 455 NAMED_BUFFERS: 1 SUM: 456
PREFIX_COUNTS: {'loss_T': 1, 'local_weight': 1, 'local_weight_dis': 1, 'backbone': 446, 'affine': 6} SUM: 455
PREFIX_COUNTS_TOP2: {'backbone': 446, 'affine': 6}
FRESH_MODEL_CLASS: model_public.PAIR
FRESH_MODEL_KEY_COUNT: 381
```

**455 vs 456 reconciled.** Both numbers are correct measurements of different things:
- `state_dict()` returns **455** entries (the identity used for key-diffing).
- `named_parameters()` = 455 and `named_buffers()` = 1, summing to **456** registered tensors.
- The 1 buffer is `backbone.clip.attn_mask`, shape (77,77), **non-persistent**
  (`_non_persistent_buffers_set`), so it is excluded from `state_dict()`. `scripts/pair_probe.out`:
  `BUFFER backbone.clip.attn_mask ... in_state_dict False` / `non_persistent True`.
- Hence `native-blockers`' 456 is the parameter+buffer registration count; my 455 is the
  `state_dict` entry count. Neither is a defect. **The prefix breakdown 446+6+1+1+1 = 455** matches
  `state_dict`; adding the buffer gives 456.

**Raw pickle globals (the actual blocker).** `scripts/pair_pickle_globals.out` decodes the pickle's
`GLOBAL` opcodes directly (no alias loader): the released checkpoint names
`model_vssp.Encoder`, `model_vssp.Backbone`, `model_vssp.FeatureWiseAffine`, `MHTransformer_dis.Transformer`,
`MHTransformer_dis.TransformerLayer`, `MHTransformer_dis.MultiHeadAttention`,
`MHTransformer_dis.MlpTransformer` — **none of which exist in the pinned repo**
(`find workbench/third_party/PAIR -iname "model_vssp*"` and `-iname "MHTransformer_dis*"` → nothing).
The pickle can therefore only be opened with the exploratory alias loader; the "public model" it is
diffed against is a **hypothesis**, not the author's class.

**Diff against the public class (real log):**
```
MISSING_IN_CKPT 0
UNEXPECTED_IN_CKPT 74
SHAPE_MISMATCH 70
PARAMS_TOTAL 172814185
PARAMS_BYPASSED 8986448
PCT_BYPASSED_PLUS_UNCLASS 5.2001
```
So 455/381/74/70/5.20% **are reproducible** — from the exploratory-mapping restore. The caveat that
must travel with them: they describe the pickle vs a *hypothesised* public class, because
`model_vssp` is absent. They do **not** prove the author's model geometry; they quantify the gap to
the nearest public class. Mapping remains `UNVERIFIED`.

---

## 4. Checkpoint digests (computed on disk this session)

Computed with the workbench's own canonical hasher
`workbench/backend/registry.py::sha256_file` (single files hash bytes; directory artifacts hash the
sorted `(relative-path, bytes)` member set with a `directory\0` prefix and length framing). Output:
`scripts/checkpoint_hashes_canonical.json` / `.log`. A second, independent hasher
(`scripts/hash_ckpts.py`) produced matching single-file digests.

| key | sha256 | note |
|---|---|---|
| csmcir | c8812071…fa49e | = registry expected |
| airknow_fiq_n05 | f4d25ed9…f79bc6 | = registry expected |
| airknow_fiq_n08 | 756fc0bd…79c54e | = registry expected |
| conesep_fiq_n02 | 5a864ee5…19ed9d2 | = registry expected |
| ptha_mtst_cirr_transfer | 1c4dab86…b7588f | = registry expected |
| habit_fiq_n02 | 814d2d0c…76b03 | = registry expected |
| intent_fiq_n02 | 2a504cad…c26afc | = registry expected |
| hint_fashioniq | bf9c72d4…6517d4 | = registry expected |
| encoder_fashioniq | 0f31c501…da9173 | = registry expected |
| pair_b1 | c4d861ec…25fbf2 | = registry expected |
| pair_b2 | 74346fd7…aeb3f03 | = registry expected |
| clvc_bundle (6 files) | b504245d…0ee6b2 | directory digest over 6 members |
| dcnet_dir | a7f9ce0e…ecaaaa4 | = aggregate `checkpoint_sha256`; directory digest |
| combiner_noft | b31882fe…5972955 | = registry expected + aggregate |
| clip4cir_fullft_dir | 9d13b110…dcae8b4 | directory digest over 2 members |
| tgcir | c37768f3…02e966 | = registry expected |
| limn_bundle (3 files) | 4c6d1f56…e23574 | directory digest over 3 members |
| tme_fashioniq_0 | fea7fc08…103968a | registry expected_sha256 is null (Kaggle, no published hash) |
| sprc_vitl | 19386104…dbea08 | = registry expected |

Note: `dcnet_dir` and `clip4cir_fullft_dir` digests are **directory** digests (per
`sha256_file(path, required_files=…)`), which differ from the simple concat digests in
`scripts/checkpoint_hashes.json`; the canonical file is authoritative for pipeline identity.

---

## 5. Interpreter inventory (probed, `scripts/env_probe.json`)

| model | interpreter | python | torch | cuda |
|---|---|---|---|---|
| airknow | envs/airknow-adapted | 3.11.15 | 2.7.1+cu128 | True |
| conesep/habit/intent/hint | envs/ilearn39-adapted | 3.9.25 | 2.7.1+cu128 | True |
| ptha_mtst | artifacts/environments/cir-005ef4e1… | 3.10.22 | 2.7.1+cu128 | True |
| encoder | envs/encoder310-adapted | 3.10.20 | 2.7.1+cu128 | True |
| pair | envs/pair-adapted | 3.11.15 | 2.7.1+cu128 | True |
| clvc_net | **none** | — | — | — |
| dcnet | artifacts/environments/dcnet-377 | 3.7.7 | 1.4.0+cpu | False |
| combiner/fullft | artifacts/environments/clip4cir-38 | 3.8.20 | 1.11.0+cpu | False |
| tgcir | **none** | — | — | — |
| limn | envs/limn310-adapted | 3.10.20 | 2.7.1+cu128 | True |
| tme | envs/tme-adapted | 3.10.22 | 2.7.1+cu128 | True |
| csmcir | artifacts/environments/cir-005ef4e1… | 3.10.22 | 2.7.1+cu128 | True |
| sprc | artifacts/environments/sprc-39 | 3.9.25 | 2.0.1+cpu | False |

Environments on disk under `workbench/artifacts/environments/`: `clip4cir-38`, `dcnet-377`,
`sprc-39`, `cir-005ef4e1b02999a5` (all present). `clvc_net` and `tgcir` have **no** interpreter.

---

## 6. What is genuinely blocked, and by what

- **dcnet** — blocked by torch-version vs patch contradiction (torch 1.4.0 has no `weights_only`).
- **clvc_net** — blocked by no interpreter/env + unguarded `.cuda()` (6 on scoring path) requiring CUDA.
- **tgcir** — blocked by no interpreter/env + `test.py` has no CLI/loader; replay wrapper exists but
  no env to run it.
- **airknow** — blocked by `lavis.models` failing to import (`moviepy` missing) and py3.11 vs 3.8.10.
- **pair** — blocked by checkpoint mapping UNVERIFIED: the pickle names `model_vssp.*`/`MHTransformer_dis.*`
  which are absent from the pinned repo, so it can only be opened by an exploratory alias loader.
  Against the nearest public class: 455 state_dict entries (456 incl. 1 non-persistent buffer) vs 381
  keys, 74 unexpected, 70 shape mismatch, 5.2001% params bypassed (reproduced, `scripts/pair_rederive.out`).
- **clip4cir_rn50x4_fullft** — blocked (per prior audit, not re-run) by adapter hard-coding
  `combiner_state.pt`/`clip_state.pt` while disk names are `fiq_comb_*`/`fiq_clip_*`; registry
  `BLOCKED_CHECKPOINT_BUNDLE`.
- **csmcir, sprc** — user-deferred (kept visible).

---

## 6a. Gate strictness — what the PASS rows actually read

- **Composition correctness = PASS** was re-audited field-by-field over the 8 selected `full.json`
  files (not "files exist"). Literal values read: `execution_status` COMPLETED ×7 / PARTIAL ×1
  (limn); `failed_expectations` = `[]` ×8; `hash_verification` = `matched` ×8; `topk_export.status`
  TOPK_EXPORT_COMPLETE ×7 / TOPK_EXPORT_COMPLETE_PARTIAL ×1 (limn, `partial=true`,
  `missing_query_count=6`); `condition_status` COMPLETED ×7 / PARTIAL ×1 (limn: 2 unscored per
  condition). PASS covers **metric completeness + clean export**, not native parity.
- **LIMN dual-scope = NOT_RUN.** The resolver working (13 tests pass) and a 6001/6003 summary do not
  satisfy the gate. The gate needs 2 query IDs × 2 scopes × 3 conditions = 12 condition rankings with
  ordered Top-100 and target ranks; those artifacts do not exist. `limn-parity` owns this.
- **Data/cache integrity = PASS**, with the declaration count stated: of the 17 chosen variants,
  **13 declare `expected_sha256`** (all 13 match the on-disk canonical digest) and **4 declare none**
  — `clvc_net` (bundle, no per-member top-level hash), `dcnet` (run_directory), `clip4cir_rn50x4_fullft`
  (bundle), `tme` (Kaggle, no published hash). For those 4 only the on-disk digest is known.

## 6b. CORRECTION — `COMPOSITION_MATRIX.csv` `blas_pin` column (2 defects fixed)

Two cells were wrong as first written; both are corrected in place and recorded here rather than
silently overwritten.

**Defect 1 — `encoder` misattribution.** The original cell said
`BLAS-sensitive: 12727/18009 ranks changed vs unpinned`. That is wrong. Primary evidence:
`workbench/docs/reviews/20261108T-throughput/d3b_encoder_aggregate_scan.txt` is a controlled A/B
(same vectors, full 6003 queries, real encode, only BLAS threads 1 vs 10 changed). All three
conditions report `"changed": 1, "max_drank": 1, "r10_delta": 0.0, "r50_delta": 0.0`,
`ndcg_delta 2.18e-05` — BLAS thread count moves **one** rank, not 12,727.

**Second correction (same cell, later turn).** An earlier revision of this note said the cause was
*only* the interpreter change. That was **incomplete**: the adapter source **also** changed, and that
is the primary cause. Read from the rank payload `runs/<id>/ranks/encoder_fashioniq.json` →
`.identity.*` (the summary `full.json` has these as `null`):

| field | old `6a528e87…` | new `5dee7425…` |
|---|---|---|
| `adapter_module_source_hash` | `1a8b0dd25827ca0ba90f504d81f2fa190cede1ee3fb3bfe335e70ab359005823` | `bf60d855be3d72dd2ff0abc6d8b9dc12dbb5fad4a50aa87aaa774454f65a76ab` |
| `adapter_source_hashes.rank` | `5f828b13fef94a3be4e63806cf383d4f8d67a92561f136005bd6cf50315eb978` | `f0914b12834acd0bf7542913384b9b08bd9a742af91816056f1bd73022cf3d4a` |
| `adapter_source_hashes.score` | `0ab4fe1dd028b3a803e79617d6bf24f0d8de410553219a87e98125833847d2cd` | `0ab4fe1d…` (**identical**) |

So the **rank method** changed while **score** did not. A changed rank function can move metrics in
opposite directions — which is exactly what happened: **R@10 went UP 31.0178 → 31.0345 while R@50
went DOWN 49.5752 → 49.4753**. A thread-count change cannot do that.

The interpreter also differs:
- old `encoder_fashioniq_6a528e87…/full.json`: `envs/pair-adapted/bin/python`,
  `Python 3.11.15; torch 2.7.1+cu128`, `verification_status: PYTHON_VERSION_MISMATCH`, `verified: false`
- new `encoder_fashioniq_5dee7425…/full.json`: `envs/encoder310-adapted/bin/python`,
  `Python 3.10.20; torch 2.7.1+cu128`, `verified: true`

Conclusion: `6a528e87` and `5dee7425` are **not the same computation** (changed rank method +
changed interpreter) and must not be compared as a BLAS-sensitivity measurement. The cell now names
both causes, hash-first.

> **Tooling trap — nested `?q=` selectors return false nulls.** On these artifact files a nested-object
> `?q=` selector (e.g. asking for `.identity.adapter_module_source_hash` in one query) returned `null`,
> which led to the wrong "no source change" conclusion above. The values are present; they must be read
> **one scalar at a time** (load the JSON, then index `identity` then the key). `full.json` genuinely
> omits both fields — a gap in the summary, not evidence of no change. Anyone re-checking must read the
> rank payload with scalar access, not a nested selector.
>
> **Rule:** nested-object `?q=` selectors on these artifact JSONs return false nulls — read scalars
> individually; and `full.json` does not carry the adapter source hashes while
> `runs/<id>/ranks/<stem>.json` does.

**Defect 2 — `limn` misgrouped as "plain einsum".** `workbench/composition/model_limn.py:400` uses
`query @ np.asarray(gallery[start:stop]).T` — a GEMM path, batched in 8192-row blocks.
`BLAS-POLICY.vi.md` §2b lists limn in the GEMM group. Cell now says GEMM path.

**Group audit of all 8 rows** against `BLAS-POLICY.vi.md` §2b and the actual `score()` bodies
(read this session):

| model | score() implementation (file:line) | group | cell now says |
|---|---|---|---|
| encoder | `model_encoder.py:237` `queries @ gallery.T` | GEMM | GEMM path ✔ |
| conesep | `model_conesep.py:79` `einsum(..., optimize=True)` | GEMM | GEMM path ✔ |
| ptha_mtst | `model_ptha_mtst.py:87` `einsum(..., optimize=True)` | GEMM | GEMM path ✔ |
| limn | `model_limn.py:400` `query @ gallery.T` | GEMM | GEMM path ✔ |
| habit | `model_habit.py:258` `einsum("nd,gqd->ngq")` no optimize | naive-einsum | naive-einsum path ✔ |
| intent | `model_intent.py:248` `einsum("nd,gqd->ngq")` no optimize | naive-einsum | naive-einsum path ✔ |
| hint | `model_hint.py:250` `einsum("nd,gqd->ngq")` no optimize | naive-einsum | naive-einsum path ✔ |
| tme | `model_tme.py:252` `einsum("nd,gqd->ngq")` no optimize | naive-einsum | naive-einsum path ✔ |

`csmcir` is also GEMM-group per §2b but is deferred and not in the selected eight.

**`conesep` 17/18009 is genuine, not an env swap.** Both conesep runs used the same interpreter
(`ilearn39-adapted` 3.9.25, `verified: true`), so its 17-rank movement is a real 1e-7 GEMM near-tie —
a different situation from encoder, and the cell now says so.

**Repo-wide sweep for the same wrong claim (2026-11-09).** Grepped `12727`, `12,727`, `18009`,
`18,009`, `BLAS-sensiti*`, `ranks moved` across `workbench/docs/**`, `workbench/docs/*.md`, and
`workbench/registry/models.yaml`. The registry `encoder` entry carries **no** 12727 note (its note is
only about the unverified Drive URL). Hits outside this review dir were all in
`workbench/docs/reviews/20261109-clean-rerun/`. Each was classified and, where it attributed the
old→new encoder delta to BLAS, corrected in place with a dated CORRECTION line:

| file | status | action |
|---|---|---|
| `clean-rerun/RUN_LOG.csv` (encoder notes) | WRONG ("BLAS-sensitive as measured: 12727…") | rewritten: A/B moves 1 rank; cause is changed rank method (hash) + interpreter |
| `clean-rerun/BLAS_IDENTITY_FINDINGS.md:115` | WRONG ("Genuinely BLAS-sensitive") | CORRECTION paragraph added |
| `clean-rerun/GPU_LANE_REPORT.md:70` | WRONG (material movement implied BLAS) | CORRECTION bullet added |
| `clean-rerun/INPUT_PIPELINE_AUDIT.vi.md:209` | WRONG ("BLAS-sensitive (encoder 12727)") | CORRECTION line added |
| `clean-rerun/RERUN_DECISIONS.csv:16` | WRONG ("rerun under pinned BLAS; 12727…") | blocker rewritten with CORRECTION |
| `clean-rerun/MODEL_MATRIX.csv:16` | WRONG (same phrase) | blocker rewritten with CORRECTION |
| `clean-rerun/EVIDENCE_INDEX.md:20,51` | NEUTRAL (counts only) | left unchanged |
| `clean-rerun/BLAS_IDENTITY_FINDINGS.md:47` | NEUTRAL (per-file total table) | left unchanged |
| `20261108T153045-audit-equivalence/query_manifest.json` | NOT RELATED (12727 is an image index) | left unchanged |
| `20261109-native15-composition-limn/COMPOSITION_VERIFICATION.vi.md` | defect report **quoting** the old wording (other worker's file) | left unchanged — it is the record of the defect, not a claim |

**Cause of the 12,727 delta — final statement (supersedes the earlier "interpreter only" wording in
the six rows above).** It is **not BLAS**. The primary cause is a changed adapter **rank** method
(`adapter_module_source_hash` and `adapter_source_hashes.rank` differ; `score` hash identical), with
the interpreter change as a co-factor. Corroborated by R@10 rising while R@50 fell — impossible under
a thread-count change. See Defect 1 above for the hash table.

No metric, run id, or gate status was changed by this sweep — attribution/provenance only. The
`conesep`/`ptha_mtst` "under pinned BLAS" cells describe a *genuine* same-interpreter near-tie and
are correct as written.

## 7. Not determinable / limitations

- G1/G2 parity for **every** model: no differential against a pinned reference was run; written
  `NOT_RUN`, never PASS.
- Native run for the 13 active models other than combiner: no aggregate artifact exists; this
  workstream did not run GPU evaluations (by rule).
- `clip4cir_rn50x4_fullft` adapter name mismatch was not re-executed; recorded as prior-audit claim.
- TME `expected_sha256` is null in registry (Kaggle download), so only the on-disk digest is known,
  not an author-published value.

## 8. Added with the encoder-parity and extra-native workstreams (2026-11-09)

- `NATIVE_RUNS_EXTRA.vi.md` — the two additional native evaluations and the encoder G1/G2 work:
  what ran, under which interpreter, and what it does and does not establish.
- `ENCODER_PARITY.vi.md` — the encoder native-vs-adapter input/forward differential. G1 PASS;
  G2 PASS with **two honestly-declared failures**, neither with a moved tolerance: 2 non-tie rank
  changes at ranks far below 50 (no R@10/R@50 effect), and the batch 1-vs-N `max_rel` sub-check at
  `1.165e-2` against its declared `1e-3` (`max_abs` within tolerance).
  **Superseded in the rank half only:** the regenerated artifacts report `non_tie_count: 0` /
  `min_gap_abs: 0.0` in all three categories, so the non-tie rank failures are **0**, not 2 — see
  `20261112-parity-repair/ENCODER_RELATION.md` §4. The batch `max_rel` failure stands; the encoder
  gate is now **9** mandatory failures, not 13.
- `g1_g2.encoder.{dress,shirt,toptee}.{native,adapter}.json` — the per-category evidence for that
  differential, alongside the existing LIMN `g1_g2.*` files.
- `raw/encoder_parity/` — `adapter.*.npz`, `relation_proof_*` and the supporting captures. The
  `relation_proof_*` files exist because of the pinned upstream defect below.
- `raw/native/dcnet.json` — dcnet's completed native run (after its checkpoint-load blocker was
  fixed). **UNITS: fractions** — macro R@10 `0.27930`, R@50 `0.53761`. `R@1` is null because upstream
  computes only R@10/R@50.
- `raw/native/clip4cir_fullft.json` — `clip4cir_rn50x4_fullft`'s first completed native run.
  **UNITS: percent** — macro R@10 `43.8017`, R@50 `66.9836`. `R@1` null for the same reason.

### 8.1 Upstream defect: `learnable_relation` is drawn, not loaded

In the pinned ENCODER upstream, `learnable_relation` is created by
`nn.Parameter(torch.randn(N_p, dim, ...))` at model construction (`model_try2.py:93`) and is not
restored from the checkpoint. Two constructions therefore draw *different* tensors, so a native-vs-
adapter comparison can only be exact once that parameter is pinned on both sides. `raw/encoder_parity/
relation_proof_*` record that pinning. This is a **defect in the pinned author code**, not in the
adapter, and it is the reason the encoder G2 comparison reports any difference at all.

### 8.2 Do not compare dcnet's numbers to clip4cir's

`dcnet` reports R@10/R@50 as fractions (`0.27930`) and `clip4cir_rn50x4_fullft` as percentages
(`43.8017`). They are the same kind of quantity in different units. Additionally, dcnet's
`comb_avg 0.40845` in `info.log` is a mean over all six R@10/R@50 cells, **not** a macro over the
three categories — never cite it as a macro. The macro is the 3-category mean of the comb column,
`0.27930` / `0.53761`.

> The false-null selector warning at §6 stands for these new files too: read scalars with a top-level
> `?q=` selector, not a nested-object one.
