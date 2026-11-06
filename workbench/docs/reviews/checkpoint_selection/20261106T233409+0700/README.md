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
* **PTHA** — `ptha/measurements/ptha_transfer_run1.json` and `ptha_transfer_run2.json` have per-category
  metrics plus the contract report (which class was loaded, which contract was dispatched, before/after
  shapes, reused parameters, new parameters). Run 2 was taken **after** the target-feature dispatch was
  rewritten to key off the loaded class, and is bit-identical to run 1 on all three categories — that
  equality is the evidence that the refactor changed which branch projects, not the CirCat arithmetic.
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
* Trimming is recorded **per entry** as `provenance.trimmed`, never assumed. The nine `*/logs/*.txt`
  files recovered this session are **verbatim**, including ANSI progress bars, `trimmed: false`.
  `ptha/measurements/ptha_transfer_run2.json` is the one trimmed artifact: the native prints precede the
  JSON on stdout, so only the JSON object is stored there, with the full stream kept beside it at
  `ptha/logs/ptha_transfer_run2_stdout.txt`.
* These logs are `.txt` rather than `.log` because `.gitignore:116` ignores `*.log` globally. That rule
  is exactly why an earlier revision shipped an index declaring nine files that never reached the
  repository; the bytes are otherwise unchanged.
* No secrets, tokens or signed URLs are present. No redaction was necessary; nothing was altered.

## Known limits, stated plainly

* HINT's training noise is unknown, so the +1.32/+2.20 pp gap to the paper row is unexplained, not
  "better than paper".
* PTHA is a CIRR→FashionIQ transfer, not a FashionIQ reproduction.
* PAIR is exploratory: the public checkout does not define a consumer for every restored module, so
  equivalence with the original implementation remains unresolved.
* TME's measured feature dtype is float32 on the runtime used here; that does not prove the historical
  float16 contract wrong for every runtime. The historical value therefore remains the default, and the
  adapted runtime is declared separately in `preparation_contracts.yaml` under `runtime_overrides`.
* PTHA's TGIR branch is covered by behavioural fixtures only — no TGIR checkpoint exists on this machine,
  so that path is not checkpoint- or GPU-verified.
