# Checkpoint selection — review bundle (session 20261106T233409+0700)

Reviewable evidence for the CIR workbench checkpoint-selection run. **No model weights, images,
environments or dense gallery tensors are included.** Everything here is text: scripts, JSON
measurements, and trimmed stdout.

Start with `EVIDENCE_INDEX.json` — it maps every file to its source path, size and SHA-256, and marks
whether it was trimmed.

## What a reviewer can check without this PC

* **HINT** — `hint/measurements/hint_native_verify.json` holds per-query ranks; recompute
  `R@1/R@10/R@50` from them and compare against the stored recalls. `hint/measurements/hint_parity_top50.json`
  records, per category, how many queries had equal top-50 **index sets** and identical **ordering**.
* **PTHA** — `ptha/measurements/ptha_transfer_run1.json` has per-category metrics plus the contract report
  (before/after shapes, reused parameters, new parameters, boundary equivalence).
* **DCNet** — `dcnet/measurements/dcnet_patched_cli.json` compares the patched run against the previous
  shim-based run, per category, with absolute differences.
* **PAIR** — `pair/measurements/summary.json` keeps all four B1/B2 × H_normal/H_rev measurements;
  `pair/manifests/pair_extra_parameter_evidence.json` holds the measured geometry and the exhaustive
  search result for the extra parameter names.

## What only this PC can reproduce

Actual GPU evaluation. Those runs need the checkpoints, the dataset and CUDA; the numbers above are the
artifacts of runs already performed, not something a text-only reviewer can regenerate. Commands with
correct cwd/imports are in `CHECKPOINT_SELECTION_REPORT.md` section 5.

## Conventions used in this bundle

* Historical artifacts from the previous session are copied **unchanged** and named `*_previous_session*`
  or `ptha_previous_blocker.json`. They are not relabelled as this session's output.
* Long stdout is cut to the last 200 lines and marked `"trimmed": true` in the index. Metric lines are
  always inside that window; progress bars are what gets dropped.
* No secrets, tokens or signed URLs are present. No redaction was necessary; nothing was altered.

## Known limits, stated plainly

* HINT's training noise is unknown, so the +1.32/+2.20 pp gap to the paper row is unexplained, not
  "better than paper".
* PTHA is a CIRR→FashionIQ transfer, not a FashionIQ reproduction.
* PAIR is exploratory: the public checkout does not define a consumer for every restored module, so
  equivalence with the original implementation remains unresolved.
* TME's measured feature dtype is float32 on the runtime used here; that does not prove the historical
  float16 contract wrong for every runtime.
