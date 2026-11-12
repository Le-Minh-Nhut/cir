# Repairs pass — 2026-11-13 (second pass, ownership extension)

Three defects previously classified "reported, not patched" were repaired under the ownership
extension. No test was weakened, no tolerance moved, no measured float pinned, no blanket skip.

| # | item | before | after |
|---|---|---|---|
| 1 | `test_circus_adapters_blip.py` fixture | 4 failed / 14 passed | **18 passed** |
| 2 | `test_circus_tme_precision.py` torch stub | 1 failed | **1 passed** |
| 3 | J10 in `release_verify.sh` | 7 ids failed | **J10 PASS**, mutation-proved |
| 4 | frontend unit template | stop burnt full 10 s | **51 ms stop, 5 s budget** |

## 1. The four `blip` ids — fixture, not adapter contract

Defect: `_resident_adapter` built the adapter with `__new__`, so `batch_size` (assigned in
`CompositionAdapter.__init__`, `adapters.py:273`) never existed; every `encode_*` raised
`AttributeError` before reaching the resident-worker behaviour the tests are about.

Fix: the fixture now calls the **real base initialiser** —

```python
adapters.CompositionAdapter.__init__(
    adapter, SimpleNamespace(WORKBENCH_THIRD_PARTY_ROOT=tmp_path), "variant")
```

— rather than assigning `config` by hand. That keeps `batch_size` produced by the same line production
uses, so it honours `WORKBENCH_COMPOSITION_BATCH_SIZE` exactly as a real run does. The subclass's own
`__init__` is deliberately not called: it runs `_resolve`, which needs the registry and a staged
checkpoint, and nothing it sets is read here.

I did **not** add a class-level `batch_size` default. That was the tempting one-liner and it is wrong:
it would report `BATCH_SIZE_DEFAULT` even when the operator override is set, so a `__new__` fixture
would write a batch size no real run used — and `--batch-size` is a measurement-identity input
(batch 1 vs 32 moves 1548/2017 dress ranks).

```
before: 4 failed, 14 passed   (the `[conesep]`/`[ptha_mtst]` × reuse_and_cleanup / failure_uses_shared_error ids)
after:  18 passed             (same file; no other assertion changed meaning)
```

## 2. `test_worker_skips_empty_stdin_lines` — stub completeness

Defect: the fake `torch` module lacked `set_num_threads`, so `model_tme.py:370`
`limit_torch_threads()` raised, the worker replied `ok:false`, no `.npy` was written, and the test
failed on `FileNotFoundError` instead of on the empty-stdin behaviour it names.

Fix — behaviourally faithful, stores what it is asked to set:

```python
_threads = {"count": 1}
torch.set_num_threads = lambda count: _threads.__setitem__("count", int(count))
torch.get_num_threads = lambda: _threads["count"]
```

```
before: 1 failed  FileNotFoundError: .../output.npy
        CIRCUS_WORKER:{"ok": false, "error_type": "AttributeError",
                       "error": "module 'torch' has no attribute 'set_num_threads'"}
after:  1 passed
```

## 3. J10 — retired the dead copy-grep

Defect: `release_verify.sh` grepped the built bundle for `phán quyết audit chuẩn`, Vietnamese prose
that `4e61f2f` **deliberately deleted** from `App.tsx` when it made status payload-driven. The check
asserted the presence of copy whose removal was the point, so it could never pass — and seven
`test_release_blocker_gpu_records.py` ids were red for that reason alone.

**Chosen option: retire, not restore.** `4e61f2f`'s message states the design rule explicitly ("Status
is read from the backend's own fields, never paraphrased or guessed") and the removed string was
decorative prose in a mega-component that commit deleted wholesale. The two labels the gate genuinely
renders still exist in `gate.ts` and still ship. Restoring marketing copy so a `grep` succeeds would
reintroduce what the design dropped.

The new check asks the question the old one meant: *is the served bundle built from this source?*

1. `dist/assets` holds at least one `*.js` bundle;
2. `dist/index.html` names a bundle that **exists** in `dist/assets` (a dangling reference 404s the app
   after a restart — a failure the old check could not see);
3. that named bundle contains **both** `CAPTION THAY THẾ` and `LỆCH PAPER`, the labels `gate.ts` builds.

The freshness rule above it (no `*.ts`/`*.tsx` newer than `dist/index.html`) is unchanged.

**Mutation proof** (each break applied to the real `dist`, then restored and hash-verified):

```
M1 bundle loses "LỆCH PAPER"        -> FAIL J10 built bundle index-CZlMxyEn.js lacks the LỆCH PAPER label;
M2 index.html names a missing file  -> FAIL J10 dist/index.html references index-DOESNOTEXIST.js, which is absent from dist/assets;
M3 bundle loses "CAPTION THAY THẾ"  -> FAIL J10 built bundle index-CZlMxyEn.js lacks the CAPTION THAY THẾ label;
restored                            -> PASS J10 two labels exactly; the served bundle is built from this source and carries both
```

`dist` restored byte-exact: `index-CZlMxyEn.js` sha256 `86ecc6486baa24291a4d31e89531a51e58d41e3ae5e0757ff5ebd05bd81afbdc`
(399 247 B), `index.html` sha256 `d354437368880a745f1a33b01de8bd85c74915fcf549b096dc471fe37d97651b`.

## 4. Frontend `TimeoutStopSec` — zero margin

Defect: `Type=simple` makes npm MainPID but the listener is `node`, a grandchild. `KillMode=mixed`
signals MainPID only, so npm died in 22 ms while `node` held :5173 for **32.5 s**, until the cgroup
SIGKILL at `TimeoutStopSec=10`. Every normal stop burnt the full budget and a same-port restart inside
that window failed.

Fix in `workbench/systemd/cir-workbench-frontend.service.in`: `KillMode=control-group`,
`TimeoutStopSec=5`. The stale comment above `ExecStart` claiming `mixed` was measured-correct was
corrected in place rather than left to contradict the setting.

| KillMode | npm exit | node exit | port free | restart same port |
|---|---|---|---|---|
| `mixed` (old, 10 s) | 22 ms | **32 506 ms** | 66 266 ms | only after the cgroup SIGKILL |
| `control-group` (new, 5 s) | 22 ms | **24 ms** | **51 ms** | **immediately** |

New budget `TimeoutStopSec=5` against a measured **51 ms** stop = **~98x margin** (was 0x: the full
10 s was consumed by design). The backend keeps `mixed`/30 s on purpose — a direct group signal to a
GPU job's child would skip the queue's truthful "failed" bookkeeping.

Locked by `workbench/tests/test_frontend_stop_contract.py` (4 tests, new): asserts the frontend's
KillMode and timeout, and — as the **negative control** — that the backend still uses `mixed`, so a
future "simplification" that homogenises both units fails a test instead of breaking one silently.
Mutation-proved by reverting the template: `3 failed, 1 passed` (the backend control correctly stays
green), restored to 4 passed.

## Test runs

| command | result |
|---|---|
| `pytest test_circus_adapters_blip.py test_circus_tme_precision.py test_frontend_stop_contract.py test_systemd_user_services.py -q` | **35 passed** |
| `pytest test_frontend_stop_contract.py -q` | **4 passed** (3 failed under the mutation) |
| `pytest test_release_blocker_gpu_records.py::test_defect_b_g… ::test_offline_verdict… -q` | **2 passed** in 276.41 s |
| `pytest workbench/tests -k "release_verify or circus_adapters or circus_tme" -q` | **2 failed, 43 passed, 3282 deselected** in 1179.30 s |

The 2 remaining failures in that selection are the `test_circus_tme_lifecycle.py`
`[encoder]`/`[limn]` ids, which are `ENVIRONMENT` by construction: `workbench/tests/conftest.py`
deletes every `CIR_*`/`WORKBENCH_*` variable so the suite behaves like a fresh clone, and the
adapters refuse to launch an unverified interpreter by design. Not fixable without wiring adapted
envs into the test process or weakening the refusal — neither is acceptable.

Every id this pass set out to repair is green:

| id group | before | after |
|---|---|---|
| `test_circus_adapters_blip.py` ×4 | 4 failed | **passed** |
| `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines` | failed | **passed** |
| `test_release_blocker_gpu_records.py` ×7 | 7 failed | **passed** |
