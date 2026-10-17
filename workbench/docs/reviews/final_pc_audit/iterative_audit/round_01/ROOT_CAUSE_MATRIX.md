# Round 01 — adversarial audit results (source `a2861b2`, consolidated into `04ae2ad`)

Eight independent read-only audit slices ran against the frozen tree `a2861b2`, on four different
models (`gemini-scout`, `v41audit`, `v41review`, `gemini`). Every finding below was **reproduced by
the lead before it was accepted**, and every rejected finding is recorded with the reason.

Frozen tree: `a2861b2`. Consolidated fix commit: `04ae2ad`. The two wave-A fixes (A/B/C/D) were
audited *as fixes* — the slices were told to attack them, and two of them fell.

## Confirmed and fixed

| # | Severity | Slice | Finding | Reproduction | Fix |
|---|---|---|---|---|---|
| R01-1 | P0 | Backend | The wave-A receipt token used member `size:mtime`, both writer-forgeable. A same-length in-place rewrite + `os.utime` moved neither, and `certified=true` was still served over a tampered weight | Tamper one member, restore mtime, leave report/ledger/proof identical → `certified: True` | Token is each member's **content digest**, memoized under `(ctime, mtime, size, inode)`; `ctime` cannot be set backwards by userspace |
| R01-2 | P0 | Backend | The token resolved its root from `checkpoint_variants[0]`, so a bundle looked inside a non-existent path and every single-file checkpoint hashed to the literal `unresolved` — those keys never moved | `_checkpoint_stamp('csmcir','fashioniq')` → `unresolved`; bundle → `missing:0_dress…` | Resolution goes through `checkpoint_path`; verified for bundle / single-file / run-directory / nested-placeholder artifacts |
| R01-3 | P0 | Backend | A corrupted HMAC ledger made every endpoint answer `200 []`, indistinguishable from an empty workspace (I02/I17) | Write `{"corrupted": true}` over the ledger → `benchmark_runs() == []` | Raises `ledger_corrupt`; a **missing** ledger stays an ordinary empty answer |
| R01-4 | P0 | Backend | The token hashed through this module's `sha256_file`, which the HTTP-layer tests patch to a constant. A stub for every input is not a change token | `_checkpoint_stamp` returned `0_dress…:0000…` after an edit | Resolves `workbench.backend.registry.sha256_file` — the hasher the verifier actually uses |
| R01-5 | P0 | Tests/Verifier | A `RELEASE_BLOCKED` verdict exited **0**: the bash `fails` counter only walks J1–J12 and `TARGET_BINDING` has no J-check (I20) | `WORKBENCH_RELEASE_TARGET_MODEL=nonexistent … ; echo $?` → `0` | The report block owns the exit code via a distinct `sys.exit(3)` |
| R01-6 | P0 | Models | `checkpoint_by_id` stored `bundle_members` as raw **ids** while five consumers indexed it as **records**; `completion_validator` swallowed the `TypeError` in a bare `except` and fell back to `Path('.')` — a bundle's proof validated against the working directory | `checkpoint_availability(model, bundle_record)` → `TypeError: string indices must be integers`; `_resolve_checkpoint()` → `(Path('.'), None)` | The derived record carries three consistent views (member records, filenames, ids) and a directory `artifact_type` |
| R01-7 | P0 | Backend | `/api/jobs` rebuilt the bundle record by hand and dropped every member fact, so a **complete** LIMN bundle was refused as missing on the real path | `guarded_plan` on the production fold → `['checkpoint missing: …/checkpoints/limn']` | The queue routes through `checkpoint_by_id`, the same resolver the evaluator uses |
| R01-8 | P1 | Deploy | The verifier **deleted** unreadable capability artifacts from the operator's live artifacts root | `aa.json`/`bb.json` present before the scan, absent after | Reported, never deleted — a lingering broken artifact is a fact about the deployment |
| R01-9 | P1 | Deploy | `TARGET_BINDING` only checked the model, so a fictional checkpoint or protocol PASSED (I06/I07) | `WORKBENCH_RELEASE_TARGET_CHECKPOINT=fake_ckpt` → `TARGET_BINDING: PASS` | All three identities are resolved against the registry |
| R01-10 | P1 | Deploy | The CLI target options were exported under short names while the block read `WORKBENCH_RELEASE_TARGET_*`, so `--target-model` was silently dropped | `--target-model csmcir` → report still said `limn` | One name per setting, read directly in the block |
| R01-11 | P1 | Runtime | The environment cache key omitted host, driver and `CUDA_VISIBLE_DEVICES`, so a cached `GPU_RUNTIME_READY` report authorized a launch the capability record itself refuses (I08/I13) | Change `CUDA_VISIBLE_DEVICES` → the cache key did not move | Host identity is in the key, and the record's own `host_identity` is compared on reuse |
| R01-12 | P1 | Runtime | The interpreter contributed only `size:mtime` to the environment fingerprint, so a same-length swap with mtime restored left it unchanged | Swap 100 bytes, restore `utime`, recompute → fingerprint identical | The binary's content digest is part of the fingerprint; the advisory memo key gained `ctime` |
| R01-13 | P2 | Deploy | `run.sh` parsed `.env` differently from `pc.sh` and compose: `"8000" ` (quotes and space retained) and a final line without a newline silently dropped | Same `.env` through both parsers → `["8000" ]`/`5173` vs `8000`/`9999` | Comment/quote stripping matches compose; the read loop takes a trailing line without a newline |
| R01-14 | P2 | Deploy | `pc.sh`'s `run_id` fallback took the newest report on disk regardless of identity, so a certificate about one artifact could be printed for another (I05/I06/I07) | Two reports on disk, target absent → the other model's `run_id` selected | The fallback filters on model + checkpoint + protocol |
| R01-15 | P2 | Frontend | `warningLabels(undefined)` returned `[]`, so a `/api/runs` row with no gate record rendered with no warning at all | `warningLabels(undefined, 5)` → `[]` | Fails closed to `uncorroborated` |
| R01-16 | P2 | Frontend | The backend's `refused` state fell through to the generic `uncertified` label | `refused` and `uncorroborated` produced the same label | `refused` is its own `WarningLabel` |
| R01-17 | P2 | Frontend | `GateDetail` printed `(not certified research evidence)` immediately above a green `certified` badge whenever paper parity deviated | Deviation + certified → both strings on one screen | Paper deviation is no longer conflated with certification |
| R01-18 | P2 | Frontend | An empty bundle `members: []` defaulted `ready` to `true`, so the UI offered "Queue official bundle (COMPLETE multi-category)" with zero members (I04) | `mems.length > 0 ? … : true` → `READY` | `ready = mems.length > 0 && mems.every(runnable)` |
| R01-19 | P2 | Frontend | An aggregate-only set with **no synthetic** member was announced as `REAL EVALUATION` (I03/I14/I16) | Synthetic/refused aggregate only → `AGGREGATE-ONLY REAL EVALUATION` | `REAL EVALUATION` is only claimed for certified evidence; synthetic sets get the development warning |
| R01-20 | P2 | Frontend | An uncertified aggregate alongside per-query runs was silently suppressed (banner `""` while the counters showed it) | 1 certified + 1 uncertified + 1 experiment run → `""` | The uncertified aggregate is announced |
| R01-21 | P2 | Models | `pc.sh`'s member reasons concatenated a possibly-null `checkpoint_id`, so a malformed response raised `TypeError` inside selection instead of naming the member | `{'checkpoint_downloaded': False}` → `TypeError` | Coerced, and an unnamed member is reported as `<unnamed member>` |
| R01-22 | P2 | Tests | `workbench/tests/fixtures/pc_launcher/models.json` invented an `available` field the real producer never emits **and omitted nine real ones** — which is why the broken launcher passed its own tests for a whole release | Key-set diff against a live `checkpoint_availability` call → 9 missing, 1 extra | Fixture regenerated from the producer; a guard compares key sets and forbids `"available"` |

## Findings reproduced and REJECTED

| Claim | Slice | Why rejected |
|---|---|---|
| `/api/models` populates bundles under `entry["bundles"]`, so `pc.sh` reads an empty member list | Models | **Refuted.** `main.py:167-175` writes `entry["checkpoint_bundles"]` with `members`. The auditor's own harness built the payload under a different key. Verified by calling the real route. |
| The LIMN gate can never certify a LIMN run: the reference's `checkpoint_id` cannot match the bundle spelling, and per-category spellings are `BLOCKED_CHECKPOINT_MISMATCH` | Science | **Refuted for the real path.** With `run_meta` built as `evaluate_models.py:148-170` builds it, `limn/base_iter0_all_categories` matches the reference and reaches `ALERT_METRIC_DEVIATION` with no checkpoint blocker. The auditor's hand-built `run_meta` used `dataset: "FashionIQ"` where production writes `"fashioniq"`. |
| `MACRO_TOLERANCE_PP` is a slack over an exact check, and the docstring's "four orders of magnitude" is wrong by 1.5 orders | Science | **Partly accepted, as a documentation defect only.** The arithmetic criticism is correct (the smallest committed metric is 30.70, so the gate's smallest actionable deviation is ~1.54 pp ≈ 31× the bound, not 10⁴×), and the docstring did describe slack while justifying it as if exactness were needed. The bound itself is correct and unchanged (it is the parser's own bound); the rationale now states what it is and what ceiling it leaves. |
| The fixture omits `checkpoint_bundles` for `csmcir` | Tests | **Confirmed as drift, not a defect**: the real API only emits the key when a model declares a bundle, and `csmcir` declares none. The regenerated fixture omits it, and the new guard asserts the key is present exactly when the registry declares a bundle. |

## Method note

The audit's value came from attacking the fixes rather than confirming them: R01-1 and R01-4 were
both failures of the wave-A fix itself, and neither would have been found by re-running the wave-A
tests, which passed.
