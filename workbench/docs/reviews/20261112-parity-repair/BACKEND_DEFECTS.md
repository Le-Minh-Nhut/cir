# Backend defects — evaluation record (2026-11-12)

Scope: four reported defects against `workbench/backend/composition.py`. Two were reproduced and
fixed inside files this pass owns; two are reproduced and handed over as patch specs because the
code they need is not owned here. Nothing here advances `TESTED_SOURCE_SHA`.

## Verdicts

| # | defect | verdict | changed / handed over |
|---|---|---|---|
| 1 | `_recorded_source_matches_disk` compared the wrong hash, so `scoring_impl` was `unknown` for every run | **REPRODUCED** — fixed | `workbench/backend/composition.py:689-729` |
| 2 | `served_conditions` advertised a condition the run never measured | **REPRODUCED at a different site** — fixed | `workbench/backend/composition.py:23-38`, `:40-78`, `:1593-1608` |
| 3 | `gallery_identity_sha256` has two formulas under one key | **REPRODUCED** — spec only | `20261112-parity-repair/GALLERY_DIGEST_PATCH_SPEC.md` |
| 4 | `adapter_query_scope_resolution` counts "does not route" as "could not scope" | **REPRODUCED** — spec only | `20261112-parity-repair/SCOPE_RESOLUTION_PATCH_SPEC.md` |

## Defect 1 — the disk guard hashed one module and compared another

Claim: `scoring_module_source_hash` is the hash of the SHARED `workbench/composition/adapters.py`,
but the guard compared it against `_module_source_hash(capability(model_id))`, the model's own
`model_<id>.py`. Two different files, so the comparison was false for every artifact ever recorded.

Reproduction (one command). Save the script below as `/tmp/delta.py`, then:

```
PYTHONPATH=$PWD workbench/envs/ctl312h/bin/python /tmp/delta.py
```

It drives the real handler (`workbench.backend.main.composition_run`, i.e. what
`GET /api/composition/runs/{id}` calls) twice — once against the code on disk, once with the
pre-fix helper bodies monkeypatched back in.

```python
"""Reproduction probe for BACKEND_DEFECTS.md — save as /tmp/delta.py."""
import json
import sys
import tempfile
from pathlib import Path

import workbench.composition.adapters  # noqa: F401  (populates sys.modules)
import workbench.composition.model_clvc_net  # noqa: F401
from workbench.backend import composition as C
from workbench.backend import main as M
from workbench.composition.adapters import capability
from workbench.composition.runner import _module_source_hash

RUN = "da0a2162be0e159a1a456cec"


def pre_fix_guard(summary):
    """HEAD's body: hashes the SHARED adapters module, compares to the ADAPTER module's hash."""
    recorded = summary.get("scoring_module_source_hash")
    model_id = summary.get("model_id")
    if not recorded or not model_id:
        return False
    module_name = capability(model_id)
    if not module_name:
        return False
    module = sys.modules.get(module_name) or __import__(module_name, fromlist=["Adapter"])
    return _module_source_hash(module) == recorded


def pre_fix_displayed_conditions(summary):
    """HEAD's body: the protocol triple, plus any ABLATION_CONDITIONS it scored."""
    from workbench.composition.inputs import ABLATION_CONDITIONS

    scored = set()
    if isinstance(summary, dict):
        for field in ("condition_status", "conditions"):
            value = summary.get(field)
            if isinstance(value, (dict, list, tuple)):
                scored.update(str(n) for n in value)
        metrics = summary.get("metrics")
        if isinstance(metrics, dict):
            for role in ("multimodal", "text_only", "image_only"):
                bucket = metrics.get(role)
                if isinstance(bucket, dict) and isinstance(bucket.get("condition"), str):
                    scored.add(bucket["condition"])
    extra = tuple(n for n in ABLATION_CONDITIONS if n in scored and n not in C.CONDITIONS)
    return (*C.CONDITIONS, *extra)


def clear():
    C._SUMMARY_FILE_CACHE.clear()
    C._RANKS_CACHE.clear()


def served_impl():
    clear()
    return M.composition_run(run_id=RUN, subset="full")["scoring_impl"]


def served_display(run_id=RUN):
    clear()
    return [item["id"] for item in
            M.composition_run(run_id=run_id, subset="full")["conditions_display"]]


print("== A. scoring_impl, GET /api/composition/runs/%s ==" % RUN)
real_guard = C._recorded_source_matches_disk
post = served_impl()
C._recorded_source_matches_disk = pre_fix_guard
pre = served_impl()
C._recorded_source_matches_disk = real_guard
print("PRE-FIX :", json.dumps({"impl": pre["impl"], "source": pre["source"]}, sort_keys=True))
print("POST-FIX:", json.dumps({"impl": post["impl"], "source": post["source"]}, sort_keys=True))

print()
print("== B. the guard itself, summary carrying TODAY's two on-disk hashes ==")
live = {"model_id": "clvc_net",
        "scoring_module_source_hash": _module_source_hash(
            sys.modules["workbench.composition.adapters"]),
        "adapter_module_source_hash": _module_source_hash(sys.modules[capability("clvc_net")])}
print("POST-FIX guard:", C._recorded_source_matches_disk(live))
print("PRE-FIX  guard:", pre_fix_guard(live))
print("POST-FIX guard, adapter hash perturbed:",
      C._recorded_source_matches_disk(dict(live, adapter_module_source_hash="0" * 64)))

print()
print("== C. conditions_display ==")
real_displayed = C.displayed_conditions
post_display = served_display()
C.displayed_conditions = pre_fix_displayed_conditions
pre_display = served_display()
C.displayed_conditions = real_displayed
print("run %s PRE-FIX :" % RUN, pre_display)
print("run %s POST-FIX:" % RUN, post_display)
print("legacy dict (no `conditions`) PRE-FIX :", pre_fix_displayed_conditions({"metrics": {}}))
print("legacy dict (no `conditions`) POST-FIX:", C.displayed_conditions({"metrics": {}}))

print()
print("== D. defect 2's surviving case: AGGREGATE-ONLY 3-condition run (no rank table) ==")
tmp = Path(tempfile.mkdtemp())
variant = ["multimodal", "text_only_black_image", "clvc_image_encoder_only"]
payload = {"schema_version": 1, "model_id": "clvc_net", "checkpoint_id": "ckpt",
           "run_id": "agg1", "subset": "full",
           "protocol_id": "circus_fashioniq_release_v1", "conditions": variant,
           "query_count": 2, "gallery_count": 4, "execution_status": "COMPLETED",
           "metrics": {"coverage": 1.0},
           "condition_status": {name: {"status": "COMPLETED"} for name in variant}}
path = tmp / "composition" / "summary" / "clvc_ckpt_agg1" / "full.json"
path.parent.mkdir(parents=True)
path.write_text(json.dumps(payload))
real_config = M.resolve_config
M.resolve_config = lambda: type("Cfg", (), {"WORKBENCH_ARTIFACTS_ROOT": tmp,
                                            "FASHIONIQ_ROOT": str(tmp / "fiq")})()
clear()
post_agg = [item["id"] for item in
            M.composition_run(run_id="agg1", subset="full")["conditions_display"]]
C.displayed_conditions = pre_fix_displayed_conditions
clear()
pre_agg = [item["id"] for item in
           M.composition_run(run_id="agg1", subset="full")["conditions_display"]]
M.resolve_config = real_config
print("PRE-FIX :", pre_agg)
print("POST-FIX:", post_agg)
```

Measured output (rc=0, wall 1.16 s):

```
== A. scoring_impl, GET /api/composition/runs/da0a2162be0e159a1a456cec ==
PRE-FIX : {"impl": "unknown", "source": "unavailable"}
POST-FIX: {"impl": "unknown", "source": "unavailable"}

== B. the guard itself, summary carrying TODAY's two on-disk hashes ==
POST-FIX guard: True
PRE-FIX  guard: False
POST-FIX guard, adapter hash perturbed: False

== C. conditions_display ==
run da0a2162be0e159a1a456cec PRE-FIX : ['multimodal', 'text_only_black_image', 'clvc_image_encoder_only']
run da0a2162be0e159a1a456cec POST-FIX: ['multimodal', 'text_only_black_image', 'clvc_image_encoder_only']
legacy dict (no `conditions`) PRE-FIX : ('multimodal', 'text_only_black_image', 'image_only_empty_text')
legacy dict (no `conditions`) POST-FIX: ('multimodal', 'text_only_black_image', 'image_only_empty_text')

== D. defect 2's surviving case: AGGREGATE-ONLY 3-condition run (no rank table) ==
PRE-FIX : ['multimodal', 'text_only_black_image', 'image_only_empty_text', 'clvc_image_encoder_only']
POST-FIX: ['multimodal', 'text_only_black_image', 'clvc_image_encoder_only']
```

Section B is the proof the guard works; section A is the honest answer for a stale run; section D
is the case defect 2 actually still produced on HEAD.

The served value for that specific run is `unknown` before AND after, and that is the honest
answer: its recorded `adapter_module_source_hash 3b8e1bb5…` is `git show
1e37f02:workbench/composition/model_clvc_net.py | sha256sum`, i.e. the file as it stood when the
run was recorded. `a2c2ad7` changed `model_clvc_net.py` afterwards (today `7f150b6a…`), so that run
is genuinely historical. What was broken is that the guard could not tell the difference: it would
also have answered `False` for a run whose bytes are identical to today's, which is what the
synthetic summary in the probe shows (`True` post-fix, `False` pre-fix). Its
`scoring_module_source_hash 7ec1e34d…` does equal today's `adapters.py`, so the two-hash split is
visible on real data.

Change — `workbench/backend/composition.py:689-729`:

* reads BOTH `scoring_module_source_hash` and `adapter_module_source_hash`;
* absent **or** `"unavailable"` on either → `False` (never `live`);
* both must match: scoring hash ↔ `_module_source_hash(workbench.composition.adapters)`, adapter
  hash ↔ `_module_source_hash(capability(model_id))`.

Tests, driven through the disk guard (`get_composition_run_detail`, not
`scoring_implementation(..., live=True/False)` directly):

* `workbench/tests/test_review_defects_evaluator.py::test_a_run_whose_two_module_hashes_match_disk_serves_the_derived_impl`
* `workbench/tests/test_review_defects_evaluator.py::test_one_perturbed_hash_keeps_the_impl_unknown`
* `workbench/tests/test_review_defects_evaluator.py::test_an_absent_hash_never_reads_as_live`

Pre-fix behaviour of the first: `{'impl': 'unknown', 'basis': 'historical run records no
scoring_impl and no hash-to-implementation table exists…', 'adapter_hash_known': True,
'scoring_hash_known': True}` — i.e. both hashes PRESENT and still `unknown`, which is the defect.

## Defect 2 — the unmeasured condition was advertised

The claim pointed at `served_conditions` (then line 1551). That site was **already narrowed** by
`07a3d10`, which made it list only conditions the rank artifact carries a table for. Probing HEAD
with the real run confirms it: `['multimodal', 'text_only_black_image', 'clvc_image_encoder_only']`,
no fourth entry, and the run's own `conditions`/`condition_status` keys are exactly those three.

The surviving defect is one function earlier: `displayed_conditions` (line 26 at HEAD)
unconditionally prepended the module-level `CONDITIONS`. For a run WITH a rank table the
table-filter masked it; for an **aggregate-only** run (`ranks_data is None`, the fallback branch)
the protocol triple came back and `image_only_empty_text` was advertised even though the run
measured three other conditions.

Reproduction, same command as above, section D (aggregate-only 3-condition summary):

```
PRE-FIX : ['multimodal', 'text_only_black_image', 'image_only_empty_text', 'clvc_image_encoder_only']
POST-FIX: ['multimodal', 'text_only_black_image', 'clvc_image_encoder_only']
```

And the real run, which the table filter already protected (both sides identical):

```
run da0a2162be0e159a1a456cec PRE-FIX : ['multimodal', 'text_only_black_image', 'clvc_image_encoder_only']
run da0a2162be0e159a1a456cec POST-FIX: ['multimodal', 'text_only_black_image', 'clvc_image_encoder_only']
```

Legacy summary with no `conditions` field (must NOT change):

```
PRE-FIX : ('multimodal', 'text_only_black_image', 'image_only_empty_text')
POST-FIX: ('multimodal', 'text_only_black_image', 'image_only_empty_text')
```

Changes:

* `workbench/backend/composition.py:23-38` — new `run_conditions(summary)`: the run's own
  `conditions`, protocol names first then declared order. `None` when the summary predates the
  field, distinct from an empty declaration.
* `workbench/backend/composition.py:40-78` — `displayed_conditions` uses `run_conditions` as the
  authority; `CONDITIONS` is now only the pre-field fallback. A condition recorded in
  `condition_status`/`metrics` under a name the declaration omits is still added, so a real
  measurement can never be hidden.
* `workbench/backend/composition.py:1593-1608` — `served_conditions` is built from the declared
  conditions plus any condition the rank artifact holds a real rank for. A declared but refused
  condition (`image_only_empty_text` → `ABLATION_UNSUPPORTED`) keeps its entry, so the
  no-regression requirement holds; an all-null table for a condition nobody declared is not
  offered.

Tests (`workbench/tests/test_composition_api_contracts.py`):

* `test_a_three_condition_variant_serves_only_those_three` — asserts the served list for a
  3-condition variant AND the aggregate-only `conditions_display` case; fails on HEAD.
* `test_a_declared_but_unsupported_condition_is_still_advertised` — `ABLATION_UNSUPPORTED` keeps
  its selector entry and its status.
* `test_a_legacy_summary_without_the_field_still_serves_the_protocol_triple` — pre-field summary
  still serves the protocol triple.

## Defect 3 — two `gallery_identity_sha256` formulas (FIXED 2026-11-13)

At the time of this record, computed on one loaded release: plan/blocked branch
`0bfd43b9cc0162885f06c60cfc751602ee9f3bed2fb4d206ffcebe3f7b443462`, runner
`1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`, `gallery_count 74381` for
both. **Fixed the same way the spec proposed:** `_plan` and the blocked-report branch in
`workbench/scripts/composition_eval.py` now call `runner._gallery_fingerprint`, so the plan, the
blocked report and a completed run publish ONE digest
(`1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4` on this release, measured
2026-11-13). The fingerprint payload was NOT changed, so artifacts recorded before the fix still
re-verify on read. `0bfd43b9…` is now a historical value only. Acceptance test:
`workbench/tests/test_gallery_identity_one_formula.py`. Change request (kept as the record):
`workbench/docs/reviews/20261112-parity-repair/GALLERY_DIGEST_PATCH_SPEC.md`.

## Defect 4 — routing counted as failure (FIXED 2026-11-13; artifact left as history)

Artifact `7b1fa85007bffdab931f7f19`: `adapter_routed 0`, `still_unscoped 6003`,
`unscoped_qids` 6003 entries, `category_scopes [null]`, `resolver null`, and every
`condition_status.*.unscored_count` `0`. Cause is `runner.py:561-562` (`if None in scopes: return
None, None`) plus `runner.py:2534` counting any `None` scope as unscoped. The honest fix is in the
runner, not here — `backend/composition.py` only serves the recorded dict, and rewriting it would
diverge from `derive_routing_cohort_gap.py`, which replays against the artifact. Change request,
including how to keep `test_derived_subset_status.py:131-145` green, in
`workbench/docs/reviews/20261112-parity-repair/SCOPE_RESOLUTION_PATCH_SPEC.md`.

**Implemented in the runner** (`runner._scope_resolution_record`): unresolved is derived from the
returned REASON, `category_agnostic` and `global_scoped` are recorded, `adapter_routed` keeps its
narrow meaning, and `still_unscoped == sum(condition_status[*].unscored_count)` holds again. The
`7b1fa850…` summary is immutable and keeps its old block; no document may quote "6003 unscoped" as a
routing failure. The `test_derived_subset_status.py` invariant stays green because LIMN's two
unscoped qids carry real reasons.

## Test counts measured

| file | result |
|---|---|
| `workbench/tests/test_composition_api_contracts.py` | 10 passed |
| `workbench/tests/test_review_defects_evaluator.py` | 28 passed |
| `workbench/tests/test_api_outcome_triple.py` | 9 passed |
| `workbench/tests/test_derived_subset_status.py` | 5 passed |
| 14 composition-related files together | 143 passed, 0 failed |
| `workbench/tests/test_handoff_commands.py` | 1 failed, 7 passed |

The one failure is `test_no_release_document_points_at_a_superseded_source_sha`, which asserts that
`TESTED_SOURCE_SHA` contains every production byte on disk. The pin advance is pending a stationary
full-suite sweep and is owned by the lead; it is deliberately NOT made here, and with
`workbench/backend/composition.py` reverted that file passes 8/8, which is the proof the failure is
the pin and not these changes.
