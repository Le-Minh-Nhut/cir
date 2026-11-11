# R15 correction sidecar — Air-Know committed run artifacts

The four committed artifacts `airknow_run3cat_gpu_run{1..4}.json` are **not rewritten**: a published
artifact is evidence, and editing it to fix a claim destroys the evidence. This file records what is
wrong in their metadata and what a re-emitted artifact carries instead.

## What those artifacts claim

Each carries, in `determinism`:

```json
{"repeat_count": <1..4>, "run_to_run_delta_macro_r10": 0.4744,
 "measured_two_run_macro_r10": [52.9241, 52.4497],
 "four_run_macro_r10_mean": 52.5468, "four_run_macro_r10_sd": 0.2673,
 "four_run_macro_r10_spread": 0.6253}
```

Recomputed from the four artifacts' own `aggregate.macro_r10`
(52.924086629607395, 52.44973964180556, 52.29879994438645, 52.51441404945451):

- mean 52.54676006631348, sample sd 0.26728186433775697, spread 0.6252866852209422 — the values above
  are correct **as a summary of the four runs**.
- `run_to_run_delta_macro_r10: 0.4744` is NOT a property of the artifact carrying it. It is the
  absolute macro_r10 gap between draw 1 and draw 2 of the campaign, stamped onto every run, including
  run 4, which was never compared to anything. It is a per-run field describing a group statistic.

## What a re-emitted artifact carries (this change)

- no `run_to_run_delta_macro_r10` field at all;
- `determinism.historical_campaign_delta_macro_r10 = {"value": 0.4744, "applies_to_this_run": false,
  "provenance": ...}` — the same number, labelled as not applicable to the run;
- `determinism.variance_group`, computed at emit time from the artifacts in the output directory:
  `variance_group_key`, `identity`, `repeat_index`, `repeat_count`, `metric`, `n`, `mean`, `sd`,
  `spread`, `insufficient_draws`, `value_of_this_run`, `group_members`, `group_excluded`.
- `runtime.batch_size` and `runtime.transform`, so a later run can prove it measured the same
  population before pooling.

Because these four artifacts predate `runtime.batch_size` / `runtime.transform`, a new run's group
builder **excludes** them (reason: `identity not recorded`) instead of pooling them. Re-emitting the
group over them requires those fields; that is a re-run, not a rewrite.

Mixed-mode pooling (native train-mode with live dropout vs `--eval-mode`) raises rather than
reporting one sd over two experiments.
