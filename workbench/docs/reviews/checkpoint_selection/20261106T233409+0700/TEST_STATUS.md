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

8 test modules fail to *collect* for that reason alone; 4 more fail only through that import:
`test_composition_environment_binding`, `test_environment_planner`, `test_environment_provisioning_a32`,
`test_r11_remediations`, `test_r12_remediations`, `test_r13_remediations`,
`test_r25_selection_provisioning`, `test_release_gate_scenarios`.

## What was actually run here (Python 3.11 venv, `workbench/envs/test-adapted`)

```
workbench/tests/test_ptha_target_feature_contract.py   5 passed
workbench/tests/test_ptha_mtst_preflight.py            passed
workbench/tests/test_r19_ptha_wrapper.py               passed
workbench/tests/test_runtime_adaptation.py             6 passed (12 collection/import failures from the 3.11 limit above)
```

`workbench/tests/test_runtime_adaptation.py` had **two assertions that encoded the old LIMN contract**
(`open_clip_torch >=2.24,<2.27`). They were updated to the newly declared `==2.20.0`, which is the
intended behaviour change recorded in the report (§3.15), not a test being bent to pass.

## What still needs CI

The full suite must be run under Python 3.12 (the version the repo targets) before release. This session
verified the changed modules parse under 3.11, pass their targeted tests, and were exercised by real GPU
runs; it did **not** establish green CI.
