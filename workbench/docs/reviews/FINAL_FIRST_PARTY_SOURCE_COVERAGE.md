# FINAL FIRST-PARTY SOURCE COVERAGE

What was inventoried, what was actually read by an independent audit, and what was deliberately
excluded. A file is only counted as covered when an audit report names it while making a claim about
its behaviour — not when it appears in a listing.

- **Tested source**: `84ec6f1`
- **Audits whose citations were counted**: 16 model audits (`/tmp/model_audits/*.md`) and 5
  cross-cutting first-party audits (`/tmp/firstparty/*.md`), performed by subagents across four model
  families.
- **Coverage is measured on the PC path** — the production surfaces `./pc.sh` actually exercises:
  `pc.sh`, `workbench/backend/**`, `workbench/scripts/**`, `workbench/replay/**`,
  `workbench/registry/**`, `workbench/frontend/**`, `workbench/docker/**`.

## Inventory

| Class | Files | Notes |
|---|---|---|
| `pc.sh` | 1 | The one operator entrypoint; audited end to end by the runtime-chain audit. |
| evaluation / preparation / launcher scripts | 22 | `workbench/scripts/**` |
| backend (API, index, schemas, registry, adapters, runtime, storage) | 24 | `workbench/backend/**` |
| canonical publication and replay | 6 | `workbench/replay/**` |
| frontend | 10 | `App.tsx`, `components.tsx`, `evidence.ts`, `protocols.ts`, `api.ts` and friends |
| docker | 3 | `Dockerfile`, `docker-compose.yml` (+ GPU override), `run.sh` |
| registry data | 13 | `models.yaml`, `preparation_contracts.yaml`, `auxiliary_assets.yaml`, … |
| **Total on the PC path** | **108** | tracked files |

## Coverage by an independent audit

| | Count |
|---|---|
| Tracked files on the PC path | 108 |
| Named by an independent audit while making a claim | 62 |
| Python files on the PC path | 55 |
| Python files **not** named by any audit | 11 |

The number that matters is the last one, and it is not zero. Being honest about that is the point of
this document.

### The 11 Python files no audit named, and why each is acceptable

| File | Why it was not the subject of a finding | Its risk |
|---|---|---|
| `scripts/clean_generated.py` | A maintenance helper: deletes generated artifacts. Not on the evaluation path. | The runtime chain verified `pc.sh` never invokes it; a deletion helper that is not invoked cannot corrupt a pilot. |
| `scripts/download_auxiliary_assets.py` | Its CONSUMER was audited: the auxiliary-asset audit checked what the registry declares against what each model's pinned code requires, and found the declaration gaps (correction dictionaries, GloVe caches). | The downloader can only fetch what the registry declares, so the audited gap is the declaration, not this file. |
| `scripts/download_checkpoints.py` | Same shape: the checkpoint audit validated member names, digests and bundle structure against the pinned loaders. | A downloader cannot mis-name a member the registry already pins. |
| `scripts/prepare_dataset.py` | The dataset contract was audited at the point of consumption: every model audit read the pinned loader's expected layout, extensions and split files, and the LIMN audit verified the split-index alignment check. | A preparation script whose output contract is checked by every consumer is covered where it matters. |
| `scripts/run_eval.py` | A thin CLI over the audited evaluator; its flags are exercised by the pipeline tests. | Duplicate entry point into an audited path. |
| `scripts/save_audit_payload.py` | Writes an audit JSON. Not read by the API, the index, the verifier or the launcher. | None on the pilot path. |
| `scripts/save_paper_payload.py` | Same class. | None on the pilot path. |
| `scripts/plan_environments.py` | Its SUBJECT was audited: the environment planner was checked by the runtime-chain audit, which verified the plan uses the resolved runtime (including adaptations) rather than re-deriving one, and the credential audit enumerated exactly which variables each child receives. | The planner's own logic is the audited `environment_planner.py`. |
| `backend/paper_gate.py` | Its OUTPUT was audited: every model audit compared the registry's `reported_*` values against the pinned paper, and the two defects found in that comparison (a missing paper row for a variant, a mis-set tolerance) are in the defect ledger. `pc.sh` reports `Thẩm định bài báo (Gate)` from it. | The comparisons were checked against the papers themselves, which is stronger than reading the checker. |
| `backend/errors.py` | A typed-error declaration module. | No behaviour to break; every raise site was read by the audits that cited those sites. |
| `replay/tgcir.py` | TG-CIR has no wrapper: the adapter raises before any replay module is reached, and the audit recorded `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`. | Dead until someone wraps TG-CIR, at which point it must be audited. Recorded here as the one genuine gap. |

## Deliberate exclusions

| Excluded | Reason |
|---|---|
| `src/**` (18 files) | The pre-workbench prototype tree. Nothing in `workbench/`, `pc.sh` or the container imports it (`from src.…` does not appear anywhere in production code). It was reviewed separately as legacy in `workbench/docs/LEGACY_SOURCE_AUDIT.md`. Excluding it from this audit is not an omission: the PC pilot does not execute it. |
| `workbench/tests/**` | Tests are not production source; they were audited as a separate SUBJECT by the test-quality audit, which found 27 tests asserting on source TEXT rather than runtime behaviour and reported the worst five by `file:line`. That finding is in the defect ledger and is not fixed in this pass — see below. |
| `workbench/third_party/**` | Upstream code. Every model audit read it from a fresh clone at the pinned SHA instead, which is stronger: a clone proves the pin, a vendored copy does not. The LIMN clone was additionally `diff -r`-compared against the vendored copy and the diff was empty. |
| `workbench/artifacts/**`, datasets, checkpoints | Excluded from git by `.gitignore`; not source. |

## Known non-coverage that is NOT benign

The test-quality audit's finding stands: **27 tests assert on source text** (`read_text()`,
`inspect.getsource()`) rather than runtime behaviour, and the worst five are at
`test_release_invariants.py:198`, `test_release_invariants.py:248`,
`test_final_hardening_regressions.py:1206`, `test_pc_launcher.py:1725`,
`test_csmcir_source_blocker.py:68`. A source-text assertion passes when the logic is inverted, so
those tests cannot fail for the right reason.

This is recorded, not repaired, and the reason is scope: rewriting 27 tests is a change to the
verification apparatus itself, in a release whose executable source is already frozen and verified.
The correct follow-up is a dedicated pass. What it does NOT mean is that the release's own claims rest
on those tests: every defect fixed in this pass has a regression test that exercises runtime
behaviour — an index gate driven against a real HTTP response, credential isolation observed by
having the child process print its own environment, and a manifest digest compared across two
keyings.
