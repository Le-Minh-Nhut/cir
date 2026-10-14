# First GPU experiment

One page. Read it, then run one command.

## What to run

```bash
./workbench/run.sh autopilot \
    --config workbench/configs/pc_autopilot.pilot_limn.yaml \
    --apply \
    --resume
```

Copy that config outside the repository first and make its six `paths` entries absolute, plus the
dataset root. The config's own header states the requirement; every non-absolute entry is resolved
relative to the config file's directory, which is wrong for a copy.

`--resume` is safe on a first invocation. Without `--apply` the same command only plans.

## Why LIMN and not CSMCIR

Decided by tracing the pinned sources, not from an earlier report.

**LIMN can run.** Three whole-model checkpoints with sizes and sha256 values verified against the
author repository's own Hugging Face LFS pointer metadata; resized images generated from the canonical
images by the documented FashionIQ resize recipe; correction dictionaries staged from the pinned
checkout rather than re-derived; and a reviewable native replay (`workbench/replay/limn.py`) instead of
an invented command for a script that has none.

| Checkpoint | Bytes | SHA-256 |
| --- | --- | --- |
| `0_dress_best_model.pt` | 1792010157 | `96a335d4eb8b1e77d35575aab9177637208e2eb6c59e12f833f67e5cb0e7fc1c` |
| `0_shirt_best_model.pt` | 1792010157 | `9260c7f77d3017d8cc10e6652cc8cfe275dbc9704366fa2bf581a24808930d74` |
| `0_toptee_best_model.pt` | 1792010157 | `79f663a14d89d455fdc3612b0b36d011c06d7957d1475a654755e1053a6cece2` |

**CSMCIR cannot run from its pinned source at all.** `src/validate_blip_csmcir.py` imports
`lavis.models`, whose `__init__` executes 98 unguarded `from … import …` statements (lines 44–151) for
modules the author never committed, *before* the line that registers the class CSMCIR needs (line 152).
The package raises `ModuleNotFoundError` during import, and because the failing lines precede the
registration, the model name `blip2_cir_align_prompt_csmcir` would never resolve even if the import
were bypassed. Reproduced against the public pinned revision. On top of that, CSMCIR's official
FashionIQ path needs `COT_ours2/fashioniq/{category}_cot_val.json`, which genuinely drives gallery
features and which the author never published. The one-line fix and its verification are in
`workbench/patches/csmcir.md`; it is not applied automatically, because editing the model package
would change which classes LAVIS registers.

## What LIMN needs from you

- The FashionIQ dataset, laid out as the tool's own `README` documents: `captions/`,
  `image_splits/`, `images/`. The canonical distribution ships `.png`; LIMN's evaluator reads
  `resized_image/<category>/<id>.jpg`, which the workbench generates.
- A CUDA device. `LIMN/test.py:39` calls `.cuda()` unconditionally, so the run reports `BLOCKED_GPU`
  rather than silently falling back to CPU.
- Permission for the workbench to create one conda environment, download ~5.0 GiB of checkpoints and
  write derived images. All five grants are already set in the pilot config.
- The ledger signing key, created once and reused for every run. The config header has the exact
  `python -c` line. Reusing it matters: a new key each shell makes every earlier run unverifiable.

## Two facts to state when you report numbers

1. **The training condition is not published.** LIMN's author publishes no training-condition label
   for these weights, so the registry records the condition as unpublished. The pilot therefore needs
   `checkpoint_policy: include_flagged_conditions`, and any Recall reported from it must be labelled
   *"training condition not published by the author"* — this is **not** a verified
   0%-training-noise checkpoint, and the strict-clean comparison must not include it.
2. **Synthetic tests prove wiring, not numbers.** Everything verified so far ran with synthetic
   fixtures. No checkpoint has been loaded, no official evaluator has run against real FashionIQ, and
   no Recall has been reproduced. Those happen on your GPU box, and only then.

## What a run produces

Under `reports/`: an autopilot summary, a blockers file naming every model that could not run with its
exact reason, per-model run directories, and completion proofs. Per-query results, when generated,
land under `results/` and are indexed for the explorer UI. `--report` reads the latest summary;
`--status` reads current state.

## The rest of the finger

`pc_autopilot.all_available.yaml` runs every eligible model with flagged conditions explicit. Most
finish `BLOCKED` with a specific reason — no source, no checkpoint, an unverified mapping, an
unpublished author asset, or an evaluator with no command-line interface. That is the workbench
working: read `blockers.json` rather than treating a blocked model as a failure.

The five iLearn models (Air-Know, ConeSep, HABIT, INTENT, HINT) and PAIR have pinned evaluators with
neither an `ArgumentParser` nor a `__main__`, so nothing can pass them arguments; each reports
`command not runnable`. They would additionally need `captions/correction_dict_{dress,shirt,toptee}.json`,
which the canonical distribution does not ship and no iLearn repository publishes.

## One edit to make before running any LAVIS-based model

Every LAVIS checkout in the registry vendors a `lavis/configs/default.yaml` whose `cache_root` is an
absolute path on the author's cluster — `/raid/temp/home/.cache/lavis` for CSMCIR and SPRC,
`/export/home/.cache/lavis` for Air-Know, ConeSep, HABIT, INTENT and HINT. LAVIS reads that value
from the config file, not from the environment, so the weights the workbench stages under
`~/.cache/lavis` are not where the evaluator looks, and on a host without those directories the
download fails from inside the evaluator.

Before running an iLearn-family or SPRC model, set `cache_root` in that model's
`lavis/configs/default.yaml` to the path the registry records for its weights. It is a config value,
not code, and it is the difference between a download landing in the right place and a confusing
failure at run time. `workbench/docs/UPSTREAM_AUDIT.md` records the exact per-model values.

