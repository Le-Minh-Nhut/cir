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
| `csmcir/measurement.json` | per-category R@10/R@50, macro, checkpoint and input digests, operational parameters |
| `csmcir/full_eval.txt` | the run itself: exit 0, `Missing keys []`, the printed metric JSON |
| `csmcir/scripts/` | the four operational steps, each written to explain why it exists |
| `glove_recipe/determinism.json` | two seeded runs producing identical hashes, all ten matching the artifact DCNet was measured against |

## Result

| category | R@10 | R@50 |
|---|---|---|
| dress | 52.2062 | 74.8637 |
| shirt | 57.7527 | 76.1040 |
| toptee | 61.1933 | 80.9281 |
| **macro** | **57.0507** | **77.2986** |

Paper says 57.07 / 77.27.

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
