# SWEEP — serial full suite at `2a832bc` (2026-11-09)

One serial run, no concurrency, on the pushed branch tip with a clean worktree. Raw log:
`full_suite_2a832bc.txt` (48,329 B, header carries revision / command / interpreter / start / end /
duration / exit code / summary).

```
revision    2a832bc0efe482d1536e466232a3e2803929d1d8
command     workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q \
            -p no:cacheprovider -rfE --tb=no
interpreter CPython 3.12 (workbench/envs/ctl312h)
start       2026-11-09T08:04:08+07:00
end         2026-11-09T09:02:05+07:00
duration    3415.28 s (0:56:55)
exit code   1
summary     21 failed, 2510 passed, 47 skipped, 290 warnings
collected   2578 = 2510 + 21 + 47   (exact, nothing unaccounted)
```

## New vs known

Baseline: `20261109T012857+0700-sweep-final/full_suite_80842661.txt` — 18 failed, 2480 passed,
47 skipped = 2545 collected.

| | count |
|---|---|
| retained from the 18 | **18** |
| dropped | **0** |
| **new** | **3** |
| total now | 21 |

The 18 are byte-identical id-for-id to the previous sweep — nothing regressed silently and nothing
was accidentally fixed. The delta is entirely three additions, and **all three are consequences of
commits made after `80842661`, not of the rerun work**:

### 1. `test_audit_fix_encoder_helpers_imported.py::test_every_adapter_is_discovered`

```
AssertionError: assert 10 == 9
```

The test hard-codes the adapter population (`:123 assert len(_adapter_modules()) == 9`).
`20a406a` added the Air-Know adapter, taking the registry from 9 `composition_adapter:` entries to
10. The test's docstring still reads "nine adapters exist today".

**Attribution:** caused by `20a406a` (Air-Know adapter). The test is a **stale count guard**, not a
production defect — the extra adapter is intended. Fix is to update the expected population, ideally
by deriving it from the registry rather than hard-coding 9.

### 2. `test_circus_env_gate_omp.py::test_every_composition_adapter_is_discovered`

```
AssertionError: assert 10 == 9
```

Same cause, same commit: `:39 assert len(model_ids) == 9` and `:40 assert len(ADAPTER_MODULES) == 9`.

**Attribution:** `20a406a`, same stale count.

### 3. `test_frontend_catalog.py::test_the_inspector_renders_null_scores_as_a_dash_and_states_the_window`

```
AssertionError: the imported provenance line must be visible
```

The test asserts (`:357`) that `QueryInspector.tsx` contains the literal `"NGUỒN"`. At `80842661`
the component rendered `<Badge tone="warn">NGUỒN: {importedSource}</Badge>`. `87ff419` (Light
workspace frontend) rewrote that line as `Nguồn ứng viên: <b>{IMPORTED_LABEL[...]}</b>.` — sentence
case, and a more complete sentence. The *fact* the test guards (imported candidates are labelled with
their provenance, never presented as this model's output) is **still true**: `IMPORTED_LABEL` and
`released_audit_retriever` are both still present at `:13-14` and `:205`, which the earlier
assertions in the same test confirm.

**Attribution:** `87ff419` changed the label text; `workbench/tests/test_frontend_catalog.py` was
never updated after it (`git log 87ff419..HEAD -- workbench/tests/test_frontend_catalog.py` is empty).
So the test is pinned to a **string that the redesign intentionally changed**, while the behavioural
contract it names still holds.

## Verdict

**No new production defect.** All three additions are test-vs-code drift, each traceable to a named
commit that postdates the previous sweep's revision:

| new failure | introduced by | class |
|---|---|---|
| adapter count 10 != 9 (×2) | `20a406a` Add the Air-Know composition adapter | stale hard-coded population |
| `NGUỒN` label missing | `87ff419` Light workspace frontend | test pinned to replaced copy |

Both are the kind of failure the gate exists to surface, and neither may be silenced by editing the
assertion away without deciding which side is right. For the adapter counts the code is right and the
count should be derived, not literals. For the label, the component's new sentence is better copy and
the test should assert the *contract* (an imported run shows its source, via `IMPORTED_LABEL`) rather
than one capitalisation.

Not fixed here: this sweep ran on a frozen tree, and each of these needs its own failing-first test
plus a decision about whether the assertion or the code moves. They are recorded, not papered over.

## What this sweep does NOT establish

- **G8 remains NOT_RUN.** A single serial run gives no baseline: no repetition, no variance, and the
  host ran other lanes' work for part of the window.
- **G1/G2 remain NOT_RUN.** No native-vs-adapter differential, no forward parity.
- The 18 retained failures include `test_handoff_commands.py::test_no_release_document_points_at_a_
  superseded_source_sha`, which fails by construction until `TESTED_SOURCE_SHA` is advanced — which
  is the next step, not a defect.
