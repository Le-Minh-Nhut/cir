# Proposed registry diff — CSMCIR

**Not applied.** The prompt keeps registry policy decisions with the operator; this file is the exact
change for review, with the evidence behind each field.

Two facts drive it: the entry has now been measured (macro 57.0507 / 77.2986 on
`fashioniq_original_split`), and that measurement is a **public-caption path-mapped research run**, not
an official reproduction. The registry has to say both things.

```diff
--- a/workbench/registry/models.yaml
+++ b/workbench/registry/models.yaml
@@ -84,7 +84,26 @@
-    reproduction_status: NOT_RUN
+    # Measured 2026-11-07. The number is real; its EVIDENTIAL CLASS is not `official reproduction`.
+    # `reproduced` would claim the authors' own evaluation inputs, and the caption provenance holds
+    # only one half of that: the bytes match what the author published at the pinned revision
+    # (`downloaded-file provenance: VERIFIED`), but no author statement establishes these are the
+    # files behind the paper's numbers (`historical evaluation-input equivalence: NOT PROVEN`).
+    reproduction_status: RESEARCH_RUN_PUBLIC_CAPTION_PATH_MAPPED
+    measured_r10: 57.0507
+    measured_r50: 77.2986
+    measured_mean: 67.1747
+    measurement_protocol: fashioniq_original_split
+    measurement_evidence: workbench/docs/reviews/checkpoint_selection/20261107T021500+0700/csmcir/measurement.json
+    # The evaluator cannot be started from the pinned revision as published (98 unguarded imports of
+    # absent modules sit before the registration). The two approved patches fix that, and both are
+    # already declared above. What is ALSO needed for this checkpoint -- and is deliberately NOT
+    # declared as a third approved patch, because it is not a source repair -- is the author-published
+    # .pyc for `blip2_qformer_cir_align_prompt_target_image_caption_shiyan92`: the checkpoint was
+    # saved by that class, which ships with no .py at this revision. Declaring it belongs to whoever
+    # decides how sourceless bytecode is admitted into the vendor tree.
+    checkpoint_class: Blip2QformerCirAlignPromptTargetImageCaptionshiyan92
+    checkpoint_class_provenance: author_published_pyc_cpython310
     strict_clean_checkpoint_available: true
```

## Why each field, and what is deliberately NOT proposed

| field | reason |
|---|---|
| `reproduction_status` | a new value rather than `reproduced`: the run is real, the official-input claim is not established |
| `measured_*` | the numbers, matching the report's 3.18.5 table |
| `measurement_protocol` | `fashioniq_original_split` — the entry's native protocol, unchanged |
| `measurement_evidence` | a path inside the pushed bundle, so the claim is checkable after clone |
| `checkpoint_class` | names the class the weights belong to, which is not the registry's default model name |
| `checkpoint_class_provenance` | records that the class exists only as author bytecode |

**Not proposed:**

- No change to `reported_r10`/`reported_r50` — those are the paper's numbers and stay separated from
  measured ones.
- No promotion of `checkpoint_variants[0].status` beyond `URL_VERIFIED` /
  `checkpoint_mapping_status: VERIFIED_METADATA`. The checkpoint's identity is author-published and
  hash-verified; the *class* it belongs to is inferred from a state_dict match (860 tensors, zero
  missing, zero shape mismatch) plus the author's `.pyc`. That inference is strong but it is not an
  author statement, and the two fields are about the file, not the architecture.
- No third `approved_source_patches` entry for the `save_memory`/batch-size operational changes. Those
  are runtime parameters, not source repairs, and the vocabulary for them is
  `operational_parameters` in the measurement JSON — where they are recorded.

## Verification the diff would need before merge

```bash
cd /home/heheboiz/data/cir
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_verified_declarations.py workbench/tests/test_release_gate_scenarios.py \
  workbench/tests/test_component_boundaries.py -q
workbench/envs/ctl312h/bin/python workbench/scripts/print_roots.py --json
```

`reproduction_status` is NOT an enum: the only consumer compares it to the literal `"VERIFIED"`
(`registry.py:2011`), so a new label does not break parsing. It does mean `runtime_verified` stays
`False` for this model, which is the correct reading — the run is measured, not verified reproduction.
