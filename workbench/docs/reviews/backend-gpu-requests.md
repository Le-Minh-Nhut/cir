# GPU agent — exact flags for the Top-K export cohort run

**Read this before launching anything.** Written by the backend agent that owns `runner.py`.

## The one-line answer

```bash
cd /home/heheboiz/data/cir
WORKBENCH_CIRCUS_RELEASE_ROOT=/home/heheboiz/data/cir/workbench/artifacts/circus_release \
workbench/envs/ctl312h/bin/python workbench/scripts/composition_eval.py \
  --models limn --subsets full \
  --topk-depth 50 \
  --keep-features
```

- **`--resume` must NOT be passed for the export run.** With `--resume` and an existing rank file
  the rank site is skipped entirely, so the export is never written from the scoring pass. Run
  WITHOUT `--resume` so the rank site produces ranks AND the candidate window in one pass.
  (`topk_depth` is deliberately not in the run identity, so the `run_id` stays the same; a new
  `attempt` is minted and the export lands beside the old one. Nothing is overwritten.)
- **`--keep-features`** keeps `composition/runs/<run_id>/work/<model>/features/*.npy` alive after
  the run so a later `--topk-depth … --force-resume` can rebuild windows from cache instead of
  re-encoding 74,381 images. Without it the feature cache is deleted as before.

## What changed in `runner.py` (you pick this up on your next launch)

| behaviour | before | now |
|---|---|---|
| `topk_depth` unset | — | byte-identical to before; no candidate directory, no manifest |
| `topk_depth` set, no `--resume` | — | export written at the rank site from the same `score()` call as the metrics |
| `topk_depth` set, `--resume`, ranks exist, feature cache present | silently `COMPLETED` with no export (the trap) | windows rebuilt from cached vectors |
| `topk_depth` set, `--resume`, no cache | silently `COMPLETED` with no export | `EXPORT_INCOMPLETE` / `TOPK_EXPORT_BLOCKED` with the reason "re-run without --resume" — never a false success |
| `work/<model>/features/` after the run | always deleted | deleted unless `--keep-features` (or `topk_depth` set) |

**Never** treat `execution_status: EXPORT_INCOMPLETE` as a measurement failure: the numbers are
real, only the candidate window is missing, and `topk_export.reason` says why.

## Expected outcome for `limn --subsets full --topk-depth 50`

- `execution_status: COMPLETED` (or `PARTIAL` — LIMN's 119 ambiguous-category queries are a separate,
  pre-existing condition, not an export failure).
- `topk_export.status: TOPK_EXPORT_COMPLETE`, `depth: 50`, `conditions` = the three conditions,
  `partial: false`, `missing_query_count: 0` (use `591`/`119`-style counts if some queries are
  unscored — those are reported, not exported).
- `composition/runs/<run_id>/candidates/limn_<ckpt>/{multimodal,text_only_black_image,image_only_empty_text}.jsonl`
  ≈ 1.3 MB each, plus a `.manifest.json` each.
- `curl -s 'http://127.0.0.1:8000/api/composition/runs/<run_id>?subset=full' | jq '.capabilities'`
  → `candidate_topk: true`.

## Disk

Each model's gallery feature cache is ~2.4 GB when kept. `/home` currently has ~127 GB free, so
four models fit comfortably. `--keep-features` is a **deliberate** choice: clean up with

```bash
rm -rf workbench/artifacts/composition/runs/<run_id>/work/<model>/features
```

after the exports are produced and verified. Exports themselves are ~4 MB per run; keep those.

## If you have already launched with `--resume`

The run will report `EXPORT_INCOMPLETE` with `topk_export.reason` naming the cause and produce no
candidates. Kill it and relaunch without `--resume` (and with `--keep-features`). Nothing is
corrupted — the rank artifact is untouched and the attempt is new.
