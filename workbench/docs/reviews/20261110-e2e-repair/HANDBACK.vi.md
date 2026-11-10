# Handback — proposed change I do not own

Owner of `workbench/tests/fixtures/**` and `workbench/backend/main.py`, `workbench/frontend/**`.

## 1. Stale protocol label in the PC-launcher fixture (blocks consistency, not code)

`workbench/tests/fixtures/pc_launcher/models.json:1723-1725` declares TME's protocol as the alias:

```json
    "literature_split_label": "val",
    "native_protocol": "fashioniq_val_split_ref_included",
    "supported_protocols": [
      "fashioniq_val_split_ref_included"
    ],
```

The registry says otherwise — `workbench/registry/models.yaml:1669 →
`native_protocol: fashioniq_original_split`, `literature_split_label: original`, and its own comment
above that line records why the alias was retracted. The alias still *resolves* in the evaluator
(`src/evaluation/fashioniq.py:186`) but is refused for persisted artifacts
(`workbench/backend/schemas/results.py:12-16,136`).

**Proposed edit (exact):**

```diff
-    "literature_split_label": "val",
-    "native_protocol": "fashioniq_val_split_ref_included",
+    "literature_split_label": "original",
+    "native_protocol": "fashioniq_original_split",
     "supported_protocols": [
-      "fashioniq_val_split_ref_included"
+      "fashioniq_original_split"
     ],
```

(Each block is one line in the file; the `-`/`+` pairs above are the same line, so the net change is
`"val"` → `"original"` on line 1722 and the three `..._ref_included` occurrences → the registry id.)

Rationale, measured: `literature_split_label` must equal the registry's value for the protocol
(`registry.py:71-73`), so with `val` + `fashioniq_original_split` the loader refuses the fixture.
TME's FashionIQ cohort is the full val split with the reference eligible
(`third_party/TME/src/precompute_evaluation.py:83` guards the mask with `if category == 'CIRR'`), which
is `fashioniq_original_split`, and the native run recorded in
`raw/native/tme.json` already used that label.

Nothing else in the fixture needs to change; `reported_r10/r50/mean` are unaffected.

## 2. No API/label change requested

I did not need any change in `workbench/backend/main.py` or `workbench/frontend/**` for Phase A or B.
`release_protocol_ids()` already admits `circus_fashioniq_release_v1` from the release on disk
(`workbench/backend/composition.py:1020-1030`), so the CIRCUS cohort is annotatable without a literal
being added anywhere.
