# R12 — Independent deep recheck: dispositions and evidence

Baseline at the start of this round: `ac59f52` (== the R12 reviewed revision), clean tree, remote in
sync. Every finding was reproduced against the production entry points before repair.

## 1. Dispositions

| ID | Disposition | Change | Evidence |
|---|---|---|---|
| **R12-01** | `FIXED` | `unreachable_modules` no longer lists `sklearn`/`matplotlib`/`mpl_toolkits`; `habit`, `intent`, `conesep`, `airknow` declare `scikit-learn==1.3.2`; `intent` declares `matplotlib`; the false test invariant replaced by a native-class-import check | `_conda_specs_for_group` for the merged LAVIS group and for INTENT both contain `scikit-learn=1.3.2` and `matplotlib`; `test_the_plan_provides_every_module_the_native_class_imports` asserts the class file's own module-scope imports |
| **R12-02** | `FIXED` | Direct references carried verbatim through parse → merge → grouping → install spec → verification; registry declarations no longer dropped when a checkout ships `requirements.txt`; scanner uses a declared source instead of inventing a name; identity verified from `direct_url.json` | `_pip_specs_for_group` emits the pinned URL for BOTH tgcir and sprc; the reference was installed for real and `clip.load`/`clip.tokenize`/`clip.available_models` are present with the commit recorded in metadata |
| **R12-03** | `FIXED` | Runbook uses the real wrapper with its three required flags; `workbench.cli` removed; the spec is no longer exempt from the command audit | `test_the_pc_runbook_names_only_entrypoints_that_exist` and the module/script/placeholder audit both pass |
| **§6.1 lifecycle boundary** | `FIXED` | The test records the boundary state at BOTH points and asserts the load resolved the recorded path; it no longer claims an uncached class its snapshot could not support | All five wrappers pass under the declared interpreter; the report states which case each run was |
| **§6.2 interpreter identity** | `ALREADY_FIXED` (as a refusal) | The verifier's `BLOCKED PYTHON_VERSION_MISMATCH` on the 3.8.20 fixture is correct, not a defect to loosen | The fixture is labelled compatibility evidence, not a verified profile |
| **§6.3 decord** | `FIXED` (narrowed) | `pip check`'s complaint is accepted ONLY for `decord==0.6.0` with that exact message, and only when the module actually imports | `WHEEL` records `Tag: cp36-cp36m-manylinux2010_x86_64` while the file is `py3-none-manylinux2010`; `import decord` succeeds. Any other line still blocks |
| **§6.4 count tolerance** | `FIXED` | The tolerance is gone; passes + failures + skips must equal the collection | `1771 = 1758 + 2 + 11` for the tested revision |

## 2. Findings from the independent audit of this round's own change

The audit of `b98a25d` found four defects in the reference handling added by that commit. All fixed
at `eafb25c`, each with a regression: both conflict orders in `_constraints`, the solver-spec builder,
the pin scan, and URL+revision identity.

## 3. Subagent policy

| Wave | Agent alias | Actual model | Permitted? |
|---|---|---|---|
| R12 review (3 tasks) | `gemini-audit` | `9router/ag/gemini-3.8-flash-high:high` | yes — priority 1 |
| R11 (recorded) | `gemini-audit` | priority 1 | yes |
| R11 (recorded) | `qwen-audit`, `glm-audit` | `wdb/Qwen/Qwen3.8-27B:high`, `wdb/zai-org/GLM-5.3-Flash:high` | **NO — violation, recorded** |

Priority 2 (`cx/gpt-6-luna`) and priority 3 (`wdb/deepseek-ai/DeepSeek-V4.1-Flash`) are absent from
`models.yml`, so priority 1 is the only permitted model available; it was used for every R12 review.

## 4. Test accounting

| Run | Result | Scope |
|---|---|---|
| `pytest workbench/tests -q` on `eafb25c` | **1758 passed, 2 failed, 11 skipped**; 1771 collected | full suite; the 2 failures are the release-metadata guards that this documentation commit resolves |
| Fresh clone acceptance | see the final message | subset against the pushed revision |
| Environment audit (`plan_environments`, injected solver) | groups carry `scikit-learn=1.3.2`, `matplotlib`, and the pinned CLIP URL | planning |
| Real reference install | `clip.load`/`tokenize`/`available_models` present; commit recorded | CPython 3.8.20, torch 2.1.0+cpu |

## 5. Not verified here

Native Conda/CUDA environments, author checkpoints, GPU execution, FashionIQ full validation, paper
parity. `conda` IS present on this host (a real solve exceeds the planning timeout, so plans are built
with an injected solver); `clang` is absent, which is the prerequisite the reviewer's two E2E
failures named.
