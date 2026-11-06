# Test status on this machine

## Environment limit (pre-existing, not caused by this session)

The repository contains **PEP 701 f-string syntax**, which is only legal from **Python 3.12**.
Example, `workbench/backend/environment_planner.py:669`:

```python
f"{', '.join(sorted(_distribution_names(module, set())
                    or _package_name(name)
                    for name in ...))}"
```

Measured:

| interpreter | `ast.parse(environment_planner.py)` |
|---|---|
| Python 3.11.15 (all conda envs here) | `SyntaxError: unterminated string literal (line 669)` |
| Python 3.12.3 (system `python3`) | **OK** |

`git show HEAD:workbench/backend/environment_planner.py` fails identically under 3.11, i.e. the file
was already like this at the commit this session started from. **No 3.12 interpreter with `pip` exists
on this machine**, so a full-suite run could not be completed here.

Exactly **8** test modules fail to *collect*, all for that reason:

`test_composition_environment_binding`, `test_environment_planner`, `test_environment_provisioning_a32`,
`test_r11_remediations`, `test_r12_remediations`, `test_r13_remediations`,
`test_r25_selection_provisioning`, `test_release_gate_scenarios`.

Separately, **12 individual tests inside `test_runtime_adaptation.py`** fail — that module imports fine,
but those 12 bodies import `workbench.backend.environment_planner:model_constraints`, which is one of the
3.12-only modules above.

## What was actually run here (Python 3.11 venv, `workbench/envs/test-adapted`)

### Update 2026-11-07 (PTHA dispatch + TME runtime dtype)

Combined run, one command, exit 0:

```
test_ptha_target_feature_contract.py   11 passed   (rewritten this session)
test_ptha_mtst_preflight.py             passed
test_r19_ptha_wrapper.py                passed
test_tme_feature_dtype_contract.py     14 passed   (new this session)
test_tme_preparation.py                14 passed
test_r21_tme_lifecycle.py               included above
test_approved_source_patch.py          \
test_vendored_approved_patch.py        \
test_vendored_source_integrity.py      \
test_verified_declarations.py          \
test_dcnet_replay.py                    60 passed (integrity gates)
----------------------------------------------------------------
TOTAL                                   110 passed
```

The PTHA suite was **rewritten** this session, not merely extended: the old version counted occurrences
of the string `_target_feature_contract` in a source file and executed no contract at all. The new one
calls the production dispatcher and counts how many times `vision_proj` is actually applied
end-to-end, for both supported classes.

`test_tme_feature_dtype_contract.py` is new and exercises the production resolver
(`registry.generated_artifact_parameters`) and the production validator (`prepare_model._tme_validate`)
under both the default and the adapted profile.

### Real-model verification (not a unit test)

PTHA — full 3-category transfer re-run on GPU after the dispatch fix:

```
macro R@10 37.897844   R@50 59.836390
per-category delta vs the previous run: +0.000000 / +0.000000 / +0.000000
```

The refactor is **bit-identical** on the measured CirCat path, as expected: that branch projected once
before and once after. What changed is the TGIR branch, which is no longer projected twice.

TME — real checkpoint, 24 real gallery images, batches `10 / 10 / 4` (partial final batch exercised):
`vit_encode` returns `float32 [24,257,1408]`; the production validator accepts those bytes under the
adapted `float32` contract, rejects the identical bytes under the historical `float16` contract, and
still rejects `float64`. See `tme/measurements/tme_real_cache_smoke.json`.

### Not verified

* **TGIR has no checkpoint here.** That branch is covered by behavioural fixtures only — no author
  checkpoint, no GPU run. The fixtures prove the control flow, not that a real TGIR checkpoint works.
* **TME full-cohort ranking was not re-run** this session. The 22.49 GB cache is deleted and disk has
  ~25 G free; the change touches only the dtype gate and size bound, not numerics. The earlier
  full-cohort result stands and records `feature_dtype = torch.float32`.
* **No green CI.** See the Python 3.12 limit above.

### Re-checked 2026-11-07, after this session's shared-file edits

`pytest workbench/tests --collect-only -q` in `workbench/envs/test-adapted`:

```
2044 tests collected, 8 errors during collection
```

The 8 errors are exactly the 8 modules listed at the top of this file, all for the PEP 701
Python 3.12 reason. **No new module became uncollectable** from the `registry.py` /
`prepare_model.py` / `tme.py` signature changes — that was the specific risk being checked.

The 60 integrity tests matter most for the DCNet change: editing a vendored file
(`base/base_trainer.py`) would normally trip the source-integrity gate, and these passing means the
`approved_source_patches` declaration suppresses it exactly as intended, the same way CSMCIR's does.

`workbench/tests/test_runtime_adaptation.py` had **two assertions that encoded the old LIMN contract**
(`open_clip_torch >=2.24,<2.27`). They were updated to the newly declared `==2.20.0`, which is the
intended behaviour change recorded in the report (§3.15), not a test being bent to pass.

## What still needs CI

The full suite must be run under Python 3.12 (the version the repo targets) before release. This session
verified the changed modules parse under 3.11, pass their targeted tests, and were exercised by real GPU
runs; it did **not** establish green CI.
