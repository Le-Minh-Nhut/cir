# CSMCIR run — 2026-11-07

Evidence for the first measured CSMCIR result, and for closing D-8.

The previous report said CSMCIR was blocked on `COT_ours2/bert_captions/` and
`captions/keywords_in_mods_*.json`. That was wrong: both are read only by `FashionIQDataset_imageedit`,
which nothing instantiates. `call_path_probe.txt` is the measurement that settles it — 12 files
opened by the real constructors, zero of them either of those paths.

## What is here

| file | what it proves |
|---|---|
| `csmcir/call_path_probe.txt` | which files the evaluation path actually opens, with the constructors that opened them |
| `csmcir/asset_verification.txt` | three caption JSONs vs published sha256, record counts, gallery/reference/target coverage, and the 121-ID shirt/toptee overlap |
| `csmcir/measurement.json` | per-category R@10/R@50, macro, checkpoint and input digests, operational parameters, and both exit-0 runs |
| `csmcir/runs/` | **the two exit-0 runs**, each with its own `command.json`, `invocation_outcome.json` and `aggregate_report.json`; table in `runs/README.md` |
| `csmcir/full_eval_pipeline.txt` | run `f03887d2` end to end: header naming the real on-disk run dir, then `stderr.log` and `stdout.log` of that run verbatim, each labelled with source path, size and sha256 |
| `csmcir/pipeline_command.json` | `command.json` of run `f03887d2`, byte-identical (sha256 `5673efbf…`) |
| `csmcir/recipe_check.txt` | the non-GPU part of the 3.23 recipe executed, with actual output |
| `csmcir/digest_verification.txt` | base/derived/patch/`.pyc`/checkpoint digests measured on this machine |
| `csmcir/scripts/` | the four HISTORICAL operational steps of the first (route-B) measurement; all three post-builder helpers now exit 1 on a builder-produced tree by design |
| `glove_recipe/determinism.json` | two seeded runs producing identical hashes, all ten matching the artifact DCNet was measured against |
| `csmcir/full_eval.txt` | route-B manual invocation on the `csmcir_derived` checkout; no pipeline run dir, superseded for citation by `f03887d2` |
| `csmcir/fresh_tree_eval.txt` | same route run end to end in `/tmp/derive_fresh` (tree still on disk, `validate_blip_csmcir.py` = `9af8d120…`); same macro, superseded for citation by `f03887d2` |
| `tests/SUMMARY.json` | suite result with a baseline worktree for comparison, and each remaining failure's cause |

## Authoritative Archive vs Working Copy

- `workbench/docs/reviews/checkpoint_selection/20261107T021500+0700/` is the **authoritative archived record** and citation-grade evidence bundle, indexed by `EVIDENCE_INDEX.json`. Its scripts (such as hardened `csmcir/scripts/build_csmcir_derived.py` with `--dataset-root` and `--expect-pyc-sha256`) are the authoritative versions.
- `workbench/artifacts/checkpoint_selection/20261107T021500+0700/` was the **session working copy** generated during execution; it is not the canonical evidence archive.


## The bundle used to glue two runs together

The earlier `full_eval_pipeline.txt` carried a six-line `[RUN]` header from `evaluate_models.py` and then
a log body from a different invocation. The header's run id did not exist on disk and never had:
`evaluate_models.py` prints `[RUN] log directory:` using an id from `make_run_id()`
(`evaluate_models.py:551`, which calls `uuid.uuid4()`), and then `execute()` mints a **different** id
(`evaluate_models.py:2090`). The printed path is a prediction, not the run.

The file is rebuilt from run `f03887d2` only: its header names that run's real directory, and the body is
`stderr.log` then `stdout.log` of that run, each pasted verbatim between labelled BEGIN/END markers with
its byte count and sha256. Nothing is stitched across runs.

## Two exit-0 runs, identical to 1e-13

| | run A | run B |
|---|---|---|
| run id | `20261107T054343058424Z_csmcir_fashioniq_f564570e` | `20261107T054933277898Z_csmcir_fashioniq_f03887d2` |
| `return_code` | 0 | 0 |
| macro R@10 / R@50 | 57.05073873202006 / 77.29859352111816 | 57.05073873202006 / 77.29859352111816 |
| `evaluator_environment_knobs` | **`null` — not recorded** | all four knobs |
| full provenance | **no** | **yes** |

Only run B has full provenance. Run A is an independent replication of the score, not evidence that the
operational knobs were set. Details and byte copies: `csmcir/runs/`.

## Reproduction was rebuilt from scratch (route B, historical)

The first pass left three ways for the documented commands to fail that the staged tree hid: the
checkpoint path was one directory level off, the builder copied `src` without the `fashionIQ_dataset`
and `COT_ours2` links the evaluator opens, and the call-path probe resolved its checkout relative to
`__file__`, which is not a repository root inside an evidence bundle. All three are fixed, and
`fresh_tree_eval.txt` is the result of running the corrected sequence end to end in `/tmp/derive_fresh`:
macro 57.05073873202006 / 77.29859352111816, identical to the digit.

The builder now also stops when an approved patch does not produce the declared `derived_sha256`, and
checks the author `.pyc` hash before using it. Printing a mismatch and continuing left a half-repaired
tree that later read as an approved variant.

**The standard pipeline no longer needs that route.** `prepare_model --execute` applies the declared
patches to the pinned checkout and `download_auxiliary_assets` stages the bytecode; the three post-builder
helpers (`apply_operational_save_memory.py`, `thread_save_memory.py`, `set_query_batch.py`) now stop with
`found 0` because the patch already made those two knobs env-overridable. They are kept for provenance of
the first measurement, and are labelled HISTORICAL wherever they appear. Current instruction:
CHECKPOINT_SELECTION_REPORT.md 3.23.5b.

## Result

| category | R@10 | R@50 |
|---|---|---|
| dress | 52.20624804496765 | 74.86366033554077 |
| shirt | 57.75269865989685 | 76.1040210723877 |
| toptee | 61.19326949119568 | 80.92809915542603 |
| **macro** | **57.05073873202006** | **77.29859352111816** |

Both runs, from their own `aggregate_report.json`. Paper says 57.07 / 77.27.

## What this result is not

It is a **public-caption path-mapped research run**, not an official reproduction. The caption bytes
match what the author published at the pinned revision (`downloaded-file provenance: VERIFIED`), but
nothing establishes that these are the captions used to produce the paper's numbers
(`historical evaluation-input equivalence: NOT PROVEN`). Two copies hashing the same does not settle
that question, and a score close to the paper's does not either.

## The checkpoint needed a class the repository does not contain

`fashioniq_tuned_clip_best.pt` is stored under the key
`Blip2QformerCirAlignPromptTargetImageCaptionshiyan92`, not the CSMCIR class name. That module has no
`.py` at the pinned revision — only an author-published `.pyc` for CPython 3.10. Its state_dict is 860
tensors with zero missing and zero shape mismatches against the checkpoint, so it is the class the
weights belong to. The run uses it.

One tensor, `token_importance` (32 trained values), is in the checkpoint but in neither class's
`state_dict`. Checked on live instances, not inferred: the attribute is a plain `Tensor`, not an
`nn.Parameter`, it appears in neither `named_parameters()` nor `named_buffers()`, and its runtime value
is all ones. The author writes `nn.Parameter(...).unsqueeze(0).unsqueeze(-1)`, and `unsqueeze` returns a
plain Tensor, so the parameter is never registered.

The consequence is worth stating plainly: **the 32 trained values are never used.** Both classes run
with the ones vector regardless of what the checkpoint holds, so dropping that key cannot bias the
measurement. Re-assigning it as a Parameter would change forward behaviour relative to the author, which
is why it was not done.

The class the checkpoint belongs to has the same evaluator-facing surface: `inference`, `forward` and
`extract_target_caption_features` are each defined on both classes, and both share the
`Blip2Base → BaseModel → Module` ancestry. That is a matching surface and matching state_dict keys, not a
proof that the two forwards are byte-identical; the macro landing on the paper's numbers is a numerical
observation, and is recorded as one.

## Verification

```
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_approved_source_patch.py workbench/tests/test_vendored_approved_patch.py \
  workbench/tests/test_component_boundaries.py workbench/tests/test_csmcir_source_blocker.py \
  workbench/tests/test_vendored_source_integrity.py workbench/tests/test_verified_declarations.py \
  workbench/tests/test_release_gate_scenarios.py workbench/tests/test_pc_readiness.py -q
# 107 passed
```
