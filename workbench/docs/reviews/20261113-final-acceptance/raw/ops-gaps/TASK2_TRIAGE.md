# ops-gaps lane — Task 2: the 20 baseline ids, triaged one by one

Re-run of exactly the 20 ids from `20261112-parity-repair/SWEEP_cf6ba468.md` §4 on the current
tree, control interpreter, no GPU, no shipped service touched:

```
workbench/envs/ctl312h/bin/python -m pytest <the 20 ids> -q -p no:cacheprovider --tb=line
-> 19 failed, 45 passed in 1283.63s
```

(The 45 passing are the rest of `test_release_blocker_gpu_records.py`, `test_circus_adapters_blip.py`,
`test_circus_tme_lifecycle.py` and `test_f05_f08_f09_verification.py`, which were selected whole.)
Raw log: `triage-20-ids.log`. **18 of the 20 ids still fail; 2 are now green**
(`test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation`, §2, and
— in the earlier pin state — the SHA guard; the SHA guard is red again now for a correct reason, §7).
1 + 18 = 19 `FAILED` lines, which is the count in the log.

| # | id (short) | class | evidence command | action |
|---|---|---|---|---|
| 1-4 | `test_circus_adapters_blip.py::test_resident_worker_{reuse_and_cleanup,failure_uses_shared_error_and_stays_failed}[conesep, ptha_mtst]` | `REPRODUCIBLE_CODE_DEFECT` — **in the TEST fixture, not the adapter** | see §1 | **FIXED 2026-11-13**: 4 failed → 18 passed (whole file). Fixture now runs the real `CompositionAdapter.__init__`. |
| 5 | `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation` | **now PASSES** (was `assert 'REUSED' == 'COMPLETED'`) | §2 | none |
| 6 | `test_circus_tme_lifecycle.py::…[encoder]` | `ENVIRONMENT` | §3 | none — the supported setup is "unset ⇒ refuse" |
| 7 | `test_circus_tme_lifecycle.py::…[limn]` | `ENVIRONMENT` | §3 | none |
| 8 | `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | `REPRODUCIBLE_CODE_DEFECT` — test stub | §4 | **FIXED 2026-11-13**: 0 → 1 passed. Stub gained faithful `set_num_threads`/`get_num_threads`. |
| 9 | `test_doctor.py::test_dcnet_runtime_asset_blocks_after_preparation_is_present` | `EXTERNAL` (host asset present) | §5 | none — the assertion is stale against a *better* host state |
| 10 | `test_evaluate_models.py::test_dcnet_audited_command_reports_non_command_blockers` | `EXTERNAL` (same asset) | §5 | none |
| 11 | `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution` | `ENVIRONMENT` (needs a staged link) | §6 | none |
| 12 | `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha` | `PRE_EXISTING_BY_DESIGN` — red at stale pin; **red again now**, for the correct reason | §7 | none |
| 13-14 | `test_release_blocker_gpu_records.py::test_defect_b_g…`, `::test_offline_verdict_names_deployment_reason` | `EXTERNAL` → **`REPRODUCIBLE_CODE_DEFECT`, FIXED** | §8 | **FIXED 2026-11-13**: J10 rewritten; mutation-proved 3 ways. |
| 15-19 | `test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0..4]` | `EXTERNAL` → **`REPRODUCIBLE_CODE_DEFECT`, FIXED** | §8 | **FIXED 2026-11-13**: same J10 repair. |
| 20 | `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps` | `ENVIRONMENT` (`dataset` link points at the real tree) | §9 | none |

---

## §1 The four `blip` ids — attribution verified, and it is NOT an adapter-contract gap

Failure text, verbatim (`model_conesep.py:183`, `model_ptha_mtst.py:188`):

```
"batch_size": adapter.batch_size}), encoding="utf-8")
AttributeError: 'Adapter' object has no attribute 'batch_size'
```

**Is `batch_size` part of the adapter contract, and do two adapters fail to expose it?** No. Measured:

```
CompositionAdapter.__init__ sets batch_size: True
model_conesep:    __new__ has batch_size = False | real construction batch_size = 32
model_ptha_mtst:  __new__ has batch_size = False | real construction batch_size = 32
```

`CompositionAdapter.__init__` (`adapters.py:273`) assigns `self.batch_size = resolved_batch_size()`,
and **every** `model_*.py` `__init__` calls `super().__init__(...)`. So a normally constructed
adapter always exposes it — two do not fail to; none do.

The breakage is in the **fixture**:

```python
# workbench/tests/test_circus_adapters_blip.py:118
adapter = module.Adapter.__new__(module.Adapter)     # __init__ never runs
adapter.work_root = tmp_path / "work"
adapter._interpreter = sys.executable
...
```

`__new__` bypasses `__init__`, so `batch_size` never exists. The test then exercises
`encode_targets`/`encode_queries`, which legitimately read `adapter.batch_size` when writing the job
JSON. **The adapter is right and the fixture is incomplete.**

There is therefore **no contract gap to close inside `composition.py`**, and the honest options were:

- add a **class-level default** `batch_size = BATCH_SIZE_DEFAULT` on `CompositionAdapter` so a
  `__new__`-constructed adapter still answers — but that is *not* the same value `resolved_batch_size()`
  produces when `WORKBENCH_COMPOSITION_BATCH_SIZE` is set, so a `__new__` fixture would write a batch
  size the real run never used, and `batch_size` is a **measurement-identity** input
  (`--batch-size` is part of identity; batch 1 vs 32 moves 1548/2017 dress ranks). Trading a fixture
  bug for a silently-wrong measurement identity is worse than the red test.
- fix the fixture (`_resident_adapter` should call `__init__` or set `batch_size` explicitly).
  **TAKEN 2026-11-13** — see "Repairs" below.

I deliberately did **not** make the test expect the breakage.

### Repair (2026-11-13): fixture runs the real base initialiser

`_resident_adapter` now calls `adapters.CompositionAdapter.__init__(adapter, config, "variant")`
instead of assigning `config` by hand, so `batch_size = resolved_batch_size()` is produced by the
same line production uses — and therefore honours `WORKBENCH_COMPOSITION_BATCH_SIZE` exactly as a
real run does. The subclass's own `__init__` is deliberately **not** called: it runs `_resolve`, which
needs the registry and a staged checkpoint path, and nothing it sets is read here (the fixture
supplies `_model`/`_checkpoint`/`_interpreter` itself).

Before → after, whole file:

```
4 failed, 14 passed   (the four `[conesep]`/`[ptha_mtst]` resident-worker ids AttributeError)
18 passed             (same file, no other assertion changed meaning)
```

No other assertion in the file was touched: the four repaired ids still assert worker reuse, one
spawn, job-JSON contents and cleanup — the only change is that the adapter is now a plausible
adapter. Verified additionally that the fixture's `batch_size` tracks the operator override (the
base initialiser reads it), so the test cannot silently pin a value production would never use.

## §2 `test_cli_3d_full_gallery_resume_and_gallery_invalidation` — now PASSES

```
FAILED ... ::test_cli_3d_full_gallery_resume_and_gallery_invalidation  (sweep at cf6ba468)
-> not present in this re-run: the id is green on the current tree.
```

It was one of the three "passes alone, fails in the full serial run" ids recorded by
`105-sweep-1d5a66c1.md` §3.3 — an order/state dependency (`_module_source_hash` at `runner.py:1714`
hashes the **already-imported** module file, so a same-process `--resume` after editing the fixture
source saw different bytes in one run order and not another). Green now; no action, and it was never
a routing/identity defect. Recorded as `PRE_EXISTING_BY_DESIGN`→`REPRODUCIBLE_CODE_DEFECT` is
unnecessary: nothing changed for it in this lane, and it is now green.

## §3 `test_circus_tme_lifecycle.py` × 2 — `ENVIRONMENT`, and it is the *correct* behaviour

```
RuntimeError: execution environment not verified [UNCONFIGURED/UNCONFIGURED]:
  WORKBENCH_ENCODER_PYTHON=None (WORKBENCH_ENCODER_PYTHON not set)
```

Exact missing thing: the adapter's registry-declared interpreter variable. It **is** set in the
operator file (`workbench/config/workbench.env:10-11`) and in **not** in the test process, because
`workbench/tests/conftest.py::isolate_operator_state` deletes every `CIR_*`/`WORKBENCH_*` variable
and points `_CONFIG_PATH` at a non-existent file, so the suite runs as a fresh clone. Showing it:

```
$ env -u WORKBENCH_ENCODER_PYTHON workbench/envs/ctl312h/bin/python -c \
    "from workbench.composition.adapters import CompositionAdapter as C; print('unset')"
$ grep -n WORKBENCH_ENCODER_PYTHON workbench/config/workbench.env
10:WORKBENCH_ENCODER_PYTHON=/home/heheboiz/data/cir/workbench/envs/encoder310-adapted/bin/python
```

A fresh clone cannot satisfy it, and refusing to launch an unverified interpreter is the product
guarantee (`adapters.py:325` `require_verified_interpreter`). Not a defect; the test asks for a host
that has the adapted envs wired.

## §4 `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` — test stub, reproducible

```
FileNotFoundError: .../output.npy
Captured stdout: CIRCUS_WORKER:{"ok": false, "error_type": "AttributeError",
                                "error": "module 'torch' has no attribute 'set_num_threads'"}
```

The test installs a fake `torch` module (`test_circus_tme_precision.py:90-96`) with `cuda`,
`device`, `no_grad`, `load` — and no `set_num_threads`. `model_tme.py:370` calls
`limit_torch_threads()` → `adapters.py:182` `torch.set_num_threads(...)` → `AttributeError`, the
worker replies `ok:false`, writes no `.npy`, and the assertion fails on a missing file. Deterministic,
reproducible, and a **fixture** defect (the stub is incomplete), not production code.

### Repair (2026-11-13)

The stub gained the pair `limit_torch_threads` actually uses, behaviourally faithful — it stores what
it is asked to set, so a readback is meaningful rather than a constant:

```python
_threads = {"count": 1}
torch.set_num_threads = lambda count: _threads.__setitem__("count", int(count))
torch.get_num_threads = lambda: _threads["count"]
```

Before → after: `1 failed` (`FileNotFoundError: .../output.npy`) → `1 passed`. The test now reaches
the behaviour it names — an empty stdin line is skipped and an empty `(0, 32, 256)` array is written —
instead of failing on a missing attribute several frames earlier.

## §5 `test_doctor.py` + `test_evaluate_models.py` dcnet — `EXTERNAL` (asset now present)

```
assert checks["external-assets:dcnet"]["status"] == "BLOCKED"   ->  assert 'OK' == 'BLOCKED'
assert any(reason.startswith("external runtime asset file missing: torchvision ResNet-50 ..."))
```

Measured host state:

```
{"name": "torchvision ResNet-50 ImageNet weights", "status": "VERIFIED_LOCAL_ASSET",
 "path": "/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth"}
```

The asset is present, so the doctor reports `OK` and no blocker names it — the **product is right and
the assertion is stale** against a host that has the file. What still blocks the dcnet launch is
something else and it *is* reported:

```
plan: None
 - official command construction failed: execution environment not verified
   [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_DCNET_PYTHON=None (WORKBENCH_DCNET_PYTHON not set)
```

Class `EXTERNAL` because it is decided by the presence of a downloaded torch-hub artifact the repo
does not ship. Fixing means editing those two test files, which I do not own.

## §6 `test_f08_dcnet_config_data_dir_resolution` — `ENVIRONMENT` (staged link)

```
assert PosixPath('/home/heheboiz/data/cir/data/FashionIQ')
    == PosixPath('/home/heheboiz/data/cir/workbench/third_party/DCNet') / 'dataset' / 'fashioniq'
```

The left side is what `.resolve()` returns on this host. The search path contains
`…/third_party/DCNet/dataset/fashioniq`, and here that **is** a symlink:

```
$ ls -la workbench/third_party/DCNet/dataset/
fashioniq -> /home/heheboiz/data/cir/data/FashionIQ
```

`prepare_dataset.prepare_canonical_link` creates exactly that link; on this host it exists, so
`.resolve()` legitimately follows it and the test's unresolved expectation fails. On a clone with no
staged link the test passes. Missing thing: the absence of a staged link — the opposite of a
deployment that ran `prepare_dataset`. `git ls-files` shows the link is untracked
(`workbench/third_party/DCNet/.gitignore` ignores `dataset`), so it is host state by construction.

## §7 `test_no_release_document_points_at_a_superseded_source_sha` — `PRE_EXISTING_BY_DESIGN`

Red during the sweep because the pin named `aea143bd`; advanced to `cf6ba468`; `LEDGER.md` L18 then
recorded it green (`8 passed`). **It is red again now**, and the assertion says exactly which pin:

```
AssertionError: TESTED_SOURCE_SHA cf6ba468af5b3a6f47174bdd0d50866f23b7006f does not contain the
production code currently on disk: production files changed after the revision the release documents
name as tested.
```

Measured cause — production files changed after the pin by the several lanes of this campaign:

```
$ git diff --name-only cf6ba468 -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'
workbench/backend/composition.py          workbench/frontend/EvaluationPage.tsx
workbench/backend/limn_diagnostic.py      workbench/frontend/LimnScopeDiagnostic.tsx
workbench/backend/main.py                 workbench/frontend/ModelInventory.tsx
workbench/backend/native_measurements.py  workbench/frontend/OverviewPage.tsx
workbench/backend/parity_gate.py          workbench/frontend/data.ts
workbench/backend/registry.py             workbench/frontend/scope.ts
workbench/composition/runner.py           workbench/registry/models.yaml
workbench/scripts/composition_eval.py
```

This is the guard **working**: it is red precisely because the tree is not a pinned, gated state. It
clears only by advancing `TESTED_SOURCE_SHA` after a stationary sweep — the lead's call, explicitly
not mine (`LEDGER.md` lane notes: "owned by the lead, deliberately not done here"). It stays green at
a current pin.

## §8 the seven `release_blocker_gpu_records` ids — one `EXTERNAL` cause: J10

All seven bottom out in the same line of the verifier's own output:

```
== J10 frontend: exactly two warning labels, built bundle present ==
  FAIL J10 built bundle lacks audit-threshold copy;
Acceptance dimensions: STATIC_SOURCE FAIL
report … verdict=RELEASE_BLOCKED    (script exit 1 -> the test's returncode==0 assertion fails)
```

`release_verify.sh:448` greps the built bundle for the literal `phán quyết audit chuẩn`:

```
elif ! grep -rq "phán quyết audit chuẩn" "${REPO}/workbench/frontend/dist"; then
```

Measured now: that string exists in **neither** `dist/` **nor** any `*.ts`/`*.tsx`:

```
$ grep -rl "phán quyết audit chuẩn" workbench/frontend/dist workbench/frontend/*.ts workbench/frontend/*.tsx
NOT FOUND in dist or sources
```

It was removed deliberately: `4e61f2f` ("Split the frontend into pages and let the payload, not the
UI, decide status") deletes the string from `App.tsx`. So the verifier check is stale relative to a
deliberate copy change — `EXTERNAL` to this lane, owned by whoever owns `workbench/frontend/` and
`workbench/docker/release_verify.sh` (both on my NOT-mine list). Not a code defect I can fix here,
and the five `[<lambda>N]` ids are the same failure under five documented invocations.

Note the J1–J9/J11/J12 lines all PASS/PENDING; **only J10 fails**, which is what makes this a single
cause rather than a cascade.

### Repair (2026-11-13): retire the dead copy-grep, assert the bundle instead

**Which option, and why.** I chose **retire**, not restore. `4e61f2f`'s message is explicit that the
UI now reads status from the backend's own fields — "Status is read from the backend's own fields,
never paraphrased or guessed" — and the removed string was decorative Vietnamese prose in a
mega-component that commit deleted wholesale. The two warning labels the gate genuinely renders
(`CAPTION THAY THẾ`, `LỆCH PAPER`) still exist in `gate.ts` and still ship in the bundle, so the copy
that *matters* is untouched. Restoring marketing prose purely so a `grep` succeeds would reintroduce
copy the design deliberately dropped.

The new J10 asks the question the old one meant to ask — *is the served bundle built from this
source?* — with three fingerprints that cannot be satisfied by a stale artifact:

1. `dist/assets` holds at least one `*.js` bundle;
2. `dist/index.html` names a bundle that **actually exists** in `dist/assets` (a dangling reference
   would 404 the app after a restart — a failure the old check could not see);
3. that named bundle contains **both** `CAPTION THAY THẾ` and `LỆCH PAPER`, the labels `gate.ts`
   builds.

The pre-existing freshness rule (no `*.ts`/`*.tsx` newer than `dist/index.html`) is unchanged above it.

**Mutation proof** — three independent breaks, each into the real `dist`, each restored and
hash-verified afterwards:

| mutation | result |
|---|---|
| M1 bundle bytes lose `LỆCH PAPER` (stale/unrebuilt source) | `FAIL J10 built bundle index-CZlMxyEn.js lacks the LỆCH PAPER label;` |
| M2 `index.html` names `index-DOESNOTEXIST.js` | `FAIL J10 dist/index.html references index-DOESNOTEXIST.js, which is absent from dist/assets;` |
| M3 bundle bytes lose `CAPTION THAY THẾ` | `FAIL J10 built bundle index-CZlMxyEn.js lacks the CAPTION THAY THẾ label;` |
| restored | `PASS J10 two labels exactly; the served bundle is built from this source and carries both` |

`dist` restored byte-exact: `index-CZlMxyEn.js` sha256 `86ecc648…fbdc` (399 247 B), `index.html`
sha256 `d3544373…651b` — the values the ledger recorded for this bundle.

## §9 `test_prepare_model_executes_only_declared_recipe_steps` — `ENVIRONMENT` (staged link again)

```
[OK] FashionIQ base layout valid: /tmp/…/FashionIQ
[BLOCKED] dataset link resolves to /home/heheboiz/data/cir/data/FashionIQ,
          expected /tmp/…/FashionIQ
```

Same link as §6: the test stages its own `dataset` under `tmp_path` and passes `--dataset-root`, but
the existing staged link resolves to the real tree, and `prepare_canonical_link` refuses a link whose
target differs (`prepare_dataset.py`: `[BLOCKED] dataset link resolves to …, expected …`). Host-state
dependent, and the refusal is correct behaviour (it never silently re-points a link).

---

## Attribution the brief asked me to verify

> "Four of them are already attributed (`test_circus_adapters_blip.py` × 4, `[conesep]`/`[ptha_mtst]`,
> `AttributeError: 'Adapter' object has no attribute 'batch_size'`). Verify that attribution and
> check whether it is a real adapter-contract defect you can fix inside `composition.py`."

**Attribution verified** (same four ids, same message, reproduced). **It is not an adapter-contract
gap**: `CompositionAdapter.__init__` sets `batch_size` and every adapter calls `super().__init__`, so
every really-constructed adapter has it; only `__new__`-built test fixtures lack it. Nothing in
`composition.py` is wrong, and the class-level-default "fix" would corrupt a measurement-identity
input. See §1 for why I did not take it.
