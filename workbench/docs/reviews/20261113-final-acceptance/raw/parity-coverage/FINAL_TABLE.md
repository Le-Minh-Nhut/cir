# FINAL_TABLE — paired-parity coverage, final state (2026-11-13)

Read-only record. Every cell is derived from a file on disk or from `evaluate_all()` run in this
round; nothing is quoted from `MODEL_MATRIX.md` prose. Sources named per column.

Sources:
- **executability / interpreter / checkpoint / adapter** — `raw/parity-coverage/executability.json`
  (produced by `enumerate_executability.py`, which calls `resolve_model_interpreter`,
  `checkpoint_availability`, `preflight_blockers` and `Adapter.status()`).
- **gate** — `raw/parity-coverage/gate.json` (`parity_gate.evaluate_all()` in-process; never the
  stale `:8000` service).
- **pair** — `raw/parity-coverage/pairs_on_disk.json` (filename census, not a matrix).
- **native eval** — `workbench/docs/reviews/20261109-native15-composition-limn/raw/native/*.json`.
- **CIRCUS eval** — `workbench/artifacts/composition/**/full.json`, counted per `model_id`.

## The table

| model_id | interpreter | checkpoint | adapter | pair(state) | gate(status / failures / classes) | native eval | CIRCUS eval | disposition | evidence path |
|---|---|---|---|---|---|---|---|---|---|
| csmcir | `artifacts/environments/cir-005ef4e1b02999a5/bin/python` ✔ | `fashioniq` 2 720 612 859 B, mapped ✔ | `model_csmcir` ✔ exists, **not ready** | — | `NOT_RUN` (no artifact) | ✔ aggregate | 0 | **BLOCKED(ADAPTER_NOT_READY)** — adapter `score()` is a different quantity from the native evaluator (256-d text embedding ×32 vs attention-weighted token importance) | `executability.json`; `blockers.txt` probe_csmcir |
| airknow | **unset** (`WORKBENCH_AIRKNOW_PYTHON`); symlink → absent `teacher-sprc` | `fiq_n05` 2 724 056 638 B, `fiq_n08` same, both mapped | `model_airknow` ✔ | — | `NOT_RUN` | — | 0 | **BLOCKED(INTERPRETER_BROKEN_SYMLINK + ENV_VAR_UNSET)** | `blockers.txt` §a, §b; `probe_airknow.py` |
| conesep | `ilearn39-adapted` ✔ | `fiq_n02` 2 721 428 966 B mapped; `fiq_n05`/`fiq_n08` **absent** | `model_conesep` ✔ ready | **G1_ONLY** (native all 3 cats) | `PARITY_EVIDENCE_MISSING` / 0 / — | ✔ aggregate | 2 | **PAIRED? no — G1 only.** Native half measured; no adapter half | `g1_g2.conesep.*.native.json` |
| ptha_mtst | `artifacts/environments/cir-005ef4e1b02999a5/bin/python` ✔ | `cirr_transfer` 2 720 352 973 B mapped | `model_ptha_mtst` ✔ ready | — | `NOT_RUN` | ✔ aggregate | 2 | **SCOPE_DECISION** — no pair: `SELECTED_TRANSFER_ONLY`, so a pair would advertise a configuration the campaign does not treat as a FashionIQ reproduction. **Not run** | `unprobed_contracts.json` |
| habit | `ilearn39-adapted` ✔ | `fiq_n02` 2 730 715 930 B mapped | `model_habit` ✔ ready | **PAIRED** (3 cats) | `PARITY_MANDATORY_SUBCHECK_FAILED` / **2** / measured_disagreement | ✔ aggregate | 3 | **PAIRED** — 2 relative-term failures, `metrics_parity` missing | `g1_g2.habit.{dress,shirt,toptee}.{native,adapter}.json` |
| intent | `ilearn39-adapted` ✔ | `fiq_n02` 2 722 478 522 B mapped | `model_intent` ✔ ready | **G1_ONLY** | `PARITY_EVIDENCE_MISSING` / 0 / — | ✔ aggregate | 2 | **PAIRED? no — G1 only** | `g1_g2.intent.*.native.json` |
| hint | `ilearn39-adapted` ✔ | `fashioniq` 2 724 451 182 B mapped | `model_hint` ✔ ready | **PAIRED** (3 cats) | `PARITY_MANDATORY_SUBCHECK_FAILED` / **2** / measured_disagreement | ✔ aggregate | 2 | **PAIRED** — pre-existing | `g1_g2.hint.*.{native,adapter}.json` |
| encoder | `encoder310-adapted` ✔ | `fashioniq` 631 879 365 B mapped | `model_encoder` ✔ ready | **PAIRED** (3 cats) | `PARITY_MANDATORY_SUBCHECK_FAILED` / **9** / measured_disagreement | ✔ aggregate | 6 | **PAIRED** — pre-existing | `g1_g2.encoder.*.{native,adapter}.json` |
| pair | **unset** (`WORKBENCH_PAIR_PYTHON`); symlink dangling | `pair_b1` 694 958 565 B, `pair_b2` 694 872 371 B, **mapping UNVERIFIED** | `model_pair` ✔ | — | `NOT_RUN` (deferred) | — (exploratory rows only) | 0 | **DEFERRED_FROM_WEB** — `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` + `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. **Not run** | `blockers.txt` §e; `blockers.json` |
| clvc_net | `clvc39-la` ✔ | `clvc/` dir **1 185 968 194 B**, mapped | `model_clvc_net` ✔ ready | — | `NOT_RUN` | ✔ (per-category ranks in `CLVC_NATIVE.md`) | 6 | **BLOCKED(NEEDS_A_DISTINCT_HOOK)** — evaluator drives TWO networks and concatenates (`CLVCNet/test.py:48,116`) | `unprobed_contracts.json` |
| dcnet | **unset** (`WORKBENCH_DCNET_PYTHON`) | `dcnet/` dir **2 882 704 059 B**, mapped | `model_dcnet` ✔ | — | `NOT_RUN` | ✔ aggregate | 0 | **BLOCKED(ENV_VAR_UNSET)** — plus declared torch 1.4.0 cannot run the approved `weights_only` opt-out | `blockers.txt` §b; `probe_dcnet.py` |
| combiner_rn50x4_noft | **unset** (`WORKBENCH_CLIP4CIR_PYTHON`) | `…/RN50x4_noft/fiq_comb_RN50x4_noft.pt` 236 017 921 B mapped | `model_combiner_rn50x4_noft` ✔ | — | `NOT_RUN` | — | 0 | **BLOCKED(ENV_VAR_UNSET)** + image-extension + dataset-link blockers | `blockers.txt` §b |
| clip4cir_rn50x4_fullft | **unset** (`WORKBENCH_CLIP4CIR_PYTHON`) | `…/RN50x4_fullft/` dir 949 892 616 B mapped | `model_clip4cir_rn50x4_fullft` ✔ | — | `NOT_RUN` | ✔ aggregate | 0 | **BLOCKED(ENV_VAR_UNSET)** + same data blockers | `blockers.txt` §b |
| tgcir | `tgcir39-la` ✔ | `downloads/tgcir/x/checkpoints/Fashioniq.pt` 607 928 637 B mapped | `model_tgcir` ✔ ready | **PAIRED** (3 cats) | `PARITY_MANDATORY_SUBCHECK_FAILED` / **3** / measured_disagreement | ✔ sample-scope | 5 | **PAIRED (INCOMPLETE)** — `population_scope='sample'`, `metrics_parity` missing. See `TGCIR_BATCH.md` | `g1_g2.tgcir.*.{native,adapter}.json` |
| sprc | **unset**; symlink dangling | `sprc_fiq_vitl.pt` 1 459 756 541 B mapped; paper's `sprc_fiq.pt` **absent** | **none declared** | — | `NOT_RUN` | — | 0 | **NO_ADAPTER** — no `model_sprc.py` exists; class does not exist | `executability.json` |
| limn | `limn310-adapted` ✔ | three `0_<cat>_best_model.pt`, 1 792 010 157 B each, mapped | `model_limn` ✔ ready | **PAIRED** (3 cats) | `PARITY_MANDATORY_SUBCHECK_FAILED` / **3** / measured_disagreement | ✔ aggregate | 3 | **PAIRED** — pre-existing | `g1_g2.<cat>.{native,adapter}.json` |
| tme | `tme-adapted` ✔ | `fashioniq_0` 2 728 382 104 B mapped; `cirr_0` absent | `model_tme` ✔ ready | — | `NOT_RUN` | ✔ aggregate | 2 | **BLOCKED(NO_SINGLE_CATEGORY_AND_NO_PER_SAMPLE_API)** — `replay/tme.py:352-356` refuses a category subset | `unprobed_contracts.json` |

## Counts — each independent, each from the named source

| count | value | how it was derived |
|---|---|---|
| registry models | **17** | `yaml.safe_load(models.yaml)['models']` |
| have a **native eval** (aggregate artifact in `raw/native/`) | **9** models | `raw/native/*.json` stems, fixtures excluded: `clip4cir_fullft`, `conesep`, `dcnet`, `encoder`, `habit`, `hint`, `intent`, `ptha_mtst`, `tme` |
| have a **measured pair** (native+adapter, all 3 categories) | **5** | `pairs_on_disk.json:complete_pair_count` → `encoder`, `habit`, `hint`, `limn`, `tgcir` |
| **pass** the gate | **0** | `gate.json:passed_models` → `[]` |
| gate judged at all | **7** | `gate.json:discovered_models` → `encoder, habit, hint, limn, tgcir` + `conesep`, `intent` at `EVIDENCE_MISSING` |
| gate failures, total | **19** | 9 encoder + 2 habit + 2 hint + 3 limn + 3 tgcir |
| ...of which `measured_disagreement` | **19** | all 19: encoder 9, habit 2, hint 2, limn 3, tgcir 3. HABIT's three rank checks were closed in round F1 (5 failures → 2) |
| ...of which `absent_evidence_refusal` | **0** | HABIT's three were closed in F1 |
| ...of which `derived_invariant` | **0** | — |
| `metrics_parity` missing | **4 models** | `encoder`, `habit`, `tgcir` (+ `conesep`/`intent` unjudged) |
| **EXECUTABLE** | **10** | `executability.json:verdicts` |
| **BLOCKED** (measured, rerunnable) | **6** | 5 × `ENV_VAR_UNSET` + 1 × `ADAPTER_NOT_READY` |
| **NO_ADAPTER** | **1** | `sprc` |
| **DEFERRED_FROM_WEB** | **1** | `pair` |
| **SCOPE_DECISION** | **1** | `ptha_mtst` |
| executable but **not probed** (specified blocker) | **2** | `clvc_net` (`NEEDS_A_DISTINCT_HOOK`), `tme` (`NO_SINGLE_CATEGORY_API`) |
| CIRCUS composition runs (real, fixtures excluded) | **34** | `workbench/artifacts/composition/**/full.json` = 36, minus exactly 2 synthetic runs whose `model_id` starts `fixture_` (`fixture_encoder`, `fixture_habit`). `released_audit` is a REAL run with a non-model label and is counted |

**Blocked vs specified.** Blocked (an asset or implementation is genuinely missing, with a command and
verbatim error that reproduces): `csmcir`, `airknow`, `dcnet`, `combiner_rn50x4_noft`,
`clip4cir_rn50x4_fullft`, `sprc`, `pair` = **7**. Specified-but-not-built (code exists, hook does not):
`clvc_net`, `tme` = **2**. Decision, not a gap: `ptha_mtst` = **1**. G1-only, adapter half absent:
`conesep`, `intent` = **2**.

## Pair status, explicit

- **PAIRED** (5): `encoder`, `habit`, `hint`, `limn`, `tgcir`.
  Of these, `tgcir` is **PAIRED but INCOMPLETE**: sample-scope preview, `metrics_parity` missing.
- **G1_ONLY** (2): `conesep`, `intent`.
- **BLOCKED** (6 + `pair` deferred): `csmcir`, `airknow`, `dcnet`, `combiner_rn50x4_noft`,
  `clip4cir_rn50x4_fullft`, `sprc`, `pair`.
- **SCOPE_DECISION** (1): `ptha_mtst`.
- **NOT PROBED, specified** (2): `clvc_net`, `tme`.

## What "PASS" would require

No model passes. The four judged pairs with failures fail on `max_rel` only — every one has `max_abs`
inside its atol and `min_cosine ≥ 0.9999999`. That is the recorded, unresolved decision in LEDGER L10,
not a defect this lane may fix by moving a tolerance.

## Independent corroboration of this table

A second, independent read of the tree — a fresh process re-resolving every interpreter and re-running
`evaluate_all()`, with no reference to this document — reproduced the columns that matter:

| column | this table | independent read | agree |
|---|---|---|---|
| interpreters that resolve | 10 of 17 | 10 of 17 (`interp_ok: false` for exactly `airknow`, `pair`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `sprc`) | ✔ |
| gate judgeable | `conesep`, `intent` at `EVIDENCE_MISSING`; `habit`, `hint`, `encoder`, `tgcir`, `limn` judged | identical set and statuses | ✔ |
| per-model failure counts | encoder 9, habit 2, hint 2, limn 3, tgcir 3 | encoder 9, habit 2, hint 2, limn 3, tgcir 3 | ✔ |

The second read stopped after 16 models (`tme` was cut off when the job was cancelled), so its `tme`
row is absent rather than different; `tme`'s row here comes from `executability.json`, which covers all
17. Nothing in the independent read contradicted a published cell.
