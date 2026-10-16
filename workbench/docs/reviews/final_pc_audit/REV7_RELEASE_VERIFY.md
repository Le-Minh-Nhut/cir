# REV7 — Release verifier: can `release_verify.sh` certify a GPU it did not authoritatively load?

Adversarial, read-only review of the release-fix wave at baseline `b45c068`.
Scope: `workbench/docker/release_verify.sh`, `workbench/tests/test_release_blocker_gpu_records.py`,
`workbench/tests/test_final_hardening_regressions.py`.
All probes are throwaway scripts under `/tmp/rev7/`; every verifier run used an isolated
`WORKBENCH_ARTIFACTS_ROOT`. No file in the repository was modified: `git status --short` after the
session is identical to the pre-session list, and `workbench/artifacts/gpu_capabilities` still held
its 1944 records (count before == after).

**Verdict: SOUND_WITH_RESERVATIONS.** Every attack the assignment named fails closed, with the
reason named in the report *and* on stderr: a hand-written record, a record signed under another
deployment's key, a renamed record whose filename key disagrees, a record for another host, a
record with another dependency contract, a synthetic/override record and a 0-byte leftover are all
rejected. J1 correctly separates a clean first boot from existing ledger-corroborated results, and
refuses mock/uncorroborated output in all directions I probed. The honest cases and the
false-failure direction hold at the script level. One substantive gap: the record's
`interpreter_fingerprint` — named in the new header comment as part of the environment binding — is
checked only for presence, so a record whose fingerprint belongs to a *different* environment is
reported `VERIFIED`/`GPU_RUNTIME=PASS` (probe D6). Two smaller findings concern lost failure detail
and a documented dev state that now blocks the release.

Reproduction environment: `.venv/bin/python` 3.13 (the only interpreter here with
`pytest`/`yaml`/`fastapi`). The verified report block is lifted verbatim from the script, using the
same extraction the reviewed tests use (`test_release_blocker_gpu_records.py:55-71`).

---

## 1. What executes, and what it certifies

`workbench/docker/release_verify.sh:471-475` exports `J1..J12`/`D1..D12`; the report block
(`:500-686`) folds the GPU scan into `dimensions.GPU_RUNTIME`. The scan (`:606-655`) globs
`gpu_capability_directory(config)/*.json`, requires `record.capability_key == basename`, then runs
`_unverified_reason` (`:553-604`), whose last act is
`_runtime.verify_environment(model, probe=True, authoritative=True)` (`:596-601`). `real_gpu` is
true only when that returns `TIER_GPU_READY` **and** the runtime attached a record whose
`capability_key` equals the file's key with `capability_verified is True` (`:601-604`).

Baseline state of the reviewed suites:

```
$ .venv/bin/python -m pytest workbench/tests/test_release_blocker_gpu_records.py -q
9 passed in 4.04s
$ .venv/bin/python -m pytest workbench/tests/test_final_hardening_regressions.py -q -k "gpu or capability or scanner"
4 passed in 0.40s
$ bash -n workbench/docker/release_verify.sh ; echo rc=$?
rc=0
```

Mutation probes — each a one-hunk edit of the shipped script, run through the reviewed suites with
`RELEASE_VERIFY_SCRIPT` pointing at the mutant (12 selected tests = 9 GPU + 3 hardening):

| Mutation | Reviewed suites | Meaning |
|---|---|---|
| baseline | `12 passed in 4.32s` | — |
| drop `_gpu_capability_signature_valid` (`:571`) | `2 failed, 3 passed` | the HMAC check is load-bearing |
| drop the unreadable-file naming/removal (`:640-647`) | `1 failed, 11 passed` | removal is load-bearing |
| neuter `record.get("capability_key") != _expected_key(model)` (`:591`) | `12 passed` | §5.4 is untested |
| erase the J1 failure detail (`:118` → `fail J1 ""`) | `4 passed` | §5.1 is untested |

---

## 2. Probes

### 2.1 Forged / foreign / misnamed / stale records (all fail closed — correct)

Each row is a separate isolated root; `*` marks the reason string in
`gpu_capability_scan.unverified`.

| Probe | Construction | `status` | `real_gpu` | `GPU_RUNTIME` | Named reason |
|---|---|---|---|---|---|
| A | hand-written record, filename == `capability_key`, genuine field shape, no HMAC | `UNVERIFIED` | `false` | `PENDING` | `*` *the record is not signed with this deployment's operator key* |
| B | genuine record bytes re-signed with a different operator key | `UNVERIFIED` | `false` | `PENDING` | `*` *not signed with this deployment's operator key* |
| C | genuine record renamed so the filename key disagrees | `WRONG_KEY` | `false` | `PENDING` | `wrong_key: ["deadbeef"]` |
| D3 | genuine record, `host_identity.node` relabelled, re-signed | `UNVERIFIED` | `false` | `PENDING` | `*` *the record was probed on a different host, driver or device selection* |
| D4 | genuine record, `dependency_fingerprint` replaced, re-signed | `UNVERIFIED` | `false` | `PENDING` | `*` *the record's dependency fingerprint is not this environment's contract* |
| D5 | genuine record **plus** a legacy `gpu_capability.json` beside it | `WRONG_KEY` | `true` | `PASS` | stray listed in `wrong_key`; the genuine record still certifies |
| E | 0-byte `leftover.json` beside a genuine record | `UNREADABLE` | `true` | `PASS` | `unreadable: ["leftover.json"]`, file removed, record kept |
| F | genuine record alone | `VERIFIED` | `true` | `PASS` | — |
| H | empty capabilities directory | `UNVERIFIED` | `false` | `PENDING` | `unverified: []` (nothing found ≠ rejected) |
| K | record written by the runtime with `probe_source: override` | `UNVERIFIED` | `false` | `PENDING` | `*` *probe_source=override is a synthetic test probe, not a real probe* |
| — | capabilities directory absent entirely | `UNVERIFIED` | `false` | `PENDING` | — |
| — | operator key unset: writer produced an *unsigned* record | `UNVERIFIED` | `false` | `PENDING` | `*` *not signed with this deployment's operator key* |

Probe A through the **whole script**, isolated root, `--offline`:

```
$ bash workbench/docker/release_verify.sh --offline --report $S/report.json ; echo RC=$?
[WARN] capability record(s) that are not verified evidence: handwritten.json: the record is not signed with this deployment's operator key
report written: $S/report.json verdict=GPU_VALIDATION_PENDING
RELEASE VERIFIER: no unmet criteria (5 pending)
RC=0
$ python3 -c 'import json;r=json.load(open("'$S'/report.json"));print(r["verdict"], r["gpu_capability_scan"])'
GPU_VALIDATION_PENDING {'status': 'UNVERIFIED', 'unreadable': [], 'wrong_key': [], 'unverified': ["handwritten.json: the record is not signed with this deployment's operator key"], 'unverified_total': 1, 'real_gpu': False}
```

The environment-only test switch does not reopen it: with `WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE=1`
an `override` record still reports `probe_source=override is a synthetic test probe`, while a
genuine `interpreter` record verifies. So the baseline defect F.md §Defect 2 is fixed twice over —
by the per-record rejection *and* by `status`/`unverified` now being emitted unconditionally
(`:655-661`), where the baseline only emitted the block when it degraded.

### 2.2 Honest cases, volume, and the false-failure direction

Both directions terminate and report the true condition:

```
stale history : 3000 superseded signed records + 1 genuine  -> 0.4 s, status=VERIFIED real_gpu=True
                unverified_total=3000, sample = 10 entries + "... and 2990 more", stderr 1378 B
real deployment: 1944 records copied to an isolated root   -> 0.18 s, 4.5 KB report, status=UNVERIFIED
no records    : status=UNVERIFIED real_gpu=False, exit 0   (no false failure)
```

The bounded sample plus the full count is what keeps volume from burying the finding, and the
"per-model key computed once" comment (`:588-590`) is real: the healthy scan touched 1944 files in
0.18 s rather than hashing 1944 environments. Two residual notes, neither a defect:
`unverified_total` is counted after the early `WRONG_KEY`/HMAC exits, so the count is of records
that reached `_unverified_reason`; and the `[WARN]` rule on stderr is truncated by the same
`_sample` (1378 B for 3000 offenders) — the JSON carries `unverified_total`.

**False failure found — a failed check's reason is never stored in the report.** The inner python
blocks are captured with `j1="$(…)"` (`:66`, and the same shape at `:349` J9, `:425` J12). Bash
captures **stdout only**, and every one of these blocks reports failure by raising, so the
diagnosis goes to stderr (inherited by the console) and `"$j1"` is empty:

```
$ bash -c 'j="$(python3 -c "raise SystemExit(1)")"; printf "captured=[%s] len=%d\n" "$j" "${#j}"'
Traceback …            # printed by the child, not captured
captured=[] len=0
```

Observed on the live stack (§2.4): the console showed the J1 `AssertionError` text, while

```
$ python3 -c 'import json;r=json.load(open("real_report.json"));print(repr(r["checks"]["J1"]))'
{'status': 'FAIL', 'detail': ''}
$ grep -c "presented as real evidence" real_report.json real_out.txt
real_report.json:0
real_out.txt:1
```

So for J1, J9, J12 (and J5/J6/J7 on the same pattern) the machine-readable report loses the reason
on the failure path; only `gpu_capability_scan` is unconditionally complete. Criterion (f), §5.1.

### 2.3 J1: clean first boot vs existing valid corroborated results

The reviewed tests genuinely execute the shipped block (`:55-71` extract `say "== J1 clean startup`
through `EOF`), and I re-extracted it independently to rule out a paraphrase:

```
clean-first-boot: rc=0  16 models; boot state=clean first boot: 0 run(s) visible, every one ledger-corroborated real evidence
one-mock-run:     rc=1  AssertionError: runs presented as real evidence without corroboration:
                        mock_pair-pair_b1 [mock] labelled data_kind=mock: a development fixture, not research evidence
```

End-to-end with the real script and an isolated artifacts root: with `[]` on `/api/runs`, J1 is
`PASS … boot state=clean first boot`; with one run whose report is genuinely ledger-authenticated
(`return_code=0`, `INVOCATION_COMPLETED`), J1 is `PASS … boot state=existing valid results: 1
run(s) visible`. Using entries minted by `evaluate_models.append_evidence_entry` under an isolated
key, the corroboration direction is enforced in every direction:

| Visible run's ledger/report state | J1 |
|---|---|
| `return_code=0` + `INVOCATION_COMPLETED` | rc=0, *existing valid results* |
| `return_code=1` (child failed) | rc=1, uncorroborated |
| `return_code=0` + `EXECUTION_FAILED` | rc=1, uncorroborated |
| no report at all | rc=1, *no evidence-ledger-corroborated aggregate report for this run* |
| report bytes copied to a second file claiming another `run_id` | rc=1 (the copy inherits no witness) |

Mock output is refused because `data_kind` is part of the **authoritative run identity**
(`backend/main.py:143-183`), which is the right place to decide it — not a defect in itself.

### 2.4 The real deployment, and the false failure this review must record

Running the shipped verifier unchanged against the live stack on `127.0.0.1:8000`:

```
$ bash workbench/docker/release_verify.sh --base http://127.0.0.1:8000 --report /tmp/rev7/real_report.json
  FAIL J1
J1   FAIL
RELEASE VERIFIER: FAILED (1 unmet, 0 pending)      # RC=1, verdict RELEASE_BLOCKED
AssertionError: runs presented as real evidence without corroboration: mock_airknow-fiq_n05 [mock]
…; mock_csmcir-fiq_a [mock] …; mock_habit-fiq_n05 [mock] …; mock_pair-pair_b1 [mock] …;
mock_sprc-sprc_fiq [mock] …; mock_tgcir-tgcir_archive [mock] …
```

Those six runs are the fixtures the runbook classifies as *"Never research evidence"*, and the
mock workflow is a documented operator flow
(`workbench/docs/AUTOMATION_RUNBOOK.md:174-176`, `workbench/scripts/clean_generated.py`,
`workbench/scripts/pipeline.py:2318-2326`). J1 previously accepted `data_kind == "mock"` and now
fails on it, so the release verifier reports `RELEASE_BLOCKED` on a documented dev state until
`clean_generated.py` runs. This is the false-failure direction the assignment asked about, and it
is unrecoverable from the report alone because the detail is empty (§2.2). §5.2.

### 2.5 Retracted claim (recorded for honesty)

Mid-way through this review I twice observed `PENDING J1 boot state UNOBSERVED: no live stack`
against a stub that `curl` reported healthy, and wrote it up as a config-cache defect. **That write-up
was wrong and is retracted.** `load_registry()` re-reads the file on every call
(`workbench/backend/registry.py:20-26`) and never caches, so the mechanism I described does not
exist in this code. Repeating the exact sequence (stub started once, then two full verifier runs)
gave `PASS J1 … clean first boot` both times with `curl=200` before and after each. The two
observations were harness artifacts: in the first the stub process had died (`curl … 000`), in the
second the stub was started by the same shell that then exited. No finding.

---

## 3. The one field the scanner never compares (D6)

`_unverified_reason` accepts `record["interpreter_fingerprint"]` on presence alone:

```
$ sed -n '583,585p' workbench/docker/release_verify.sh
    if not record.get("interpreter_fingerprint"):
        return "the record carries no interpreter fingerprint"
```

`verify_environment` then rejects a mismatching fingerprint only through
`_cached_gpu_capability`, and only against a non-`None` freshly computed value
(`backend/runtime.py:1216-1218`) — but the fingerprint a `verify_environment` report carries is
filled in by `_remember` (`runtime.py:1659-1676`), which on this path stores `None` because
`report.content_verified` is false, and `_reuse` bails out on that same `None`
(`runtime.py:1634-1636`). So the comparison is skipped and the record is loaded. Probe D6:

```
$ python3 /tmp/rev7/probe_d6b.py
distinct fingerprints: True
tampered-valid-signature True
--- runtime read-back ---
runtime tier GPU_RUNTIME_READY status GPU_RUNTIME_READY launch_ready True
runtime attached record fingerprint 827485b90ee592f6a5b8f574adf8411f4c15f710b4e04ce1b8773674c9a26c6b
--- release verifier ---
status VERIFIED real_gpu True dim PASS verdict ENGINEERING_INCOMPLETE
reasons []
```

The record is a real probe's, re-signed honestly under **this** deployment's key (so the HMAC is
genuine and every other binding is satisfied), with `interpreter_fingerprint` replaced by another
virtualenv's *real* fingerprint — printed above as distinct from this environment's. Both the
runtime and the release verifier certify it. `capability_key` does not catch it, because
`_gpu_capability_key` (`runtime.py:1027-1046`) derives the key's fingerprint from the *live*
environment, never from the record. The check the header comment (`:513-518`) advertises is, on
this path, a presence check. §5.4.

**How a mutated record is reachable.** Not only by editing a file: the writer signs whatever dict
it is handed. `record_gpu_capability` (`runtime.py:960-997`) computes
`signature = _gpu_capability_signature(payload)` (`:982`) for any payload, and the only gate on
`verify_gpu_runtime` is `if not authorized: raise` (`runtime.py:1893-1902`) — a function-local flag,
not a trust boundary. Probe D8, in an isolated root with a real venv that has **no torch**:

```
$ python3 /tmp/rev7/probe_selfauthored.py
authored /tmp/rev7-self-…/gpu_capabilities/a484936657a0d6059a402e227f4e208e024a2d868e192c87b11c0984902b2169.json hmac True
torch present in that interpreter: False
scan -> VERIFIED real_gpu True dim GPU_RUNTIME = PASS
runtime -> runtime tier GPU_RUNTIME_READY launch_ready True
```

No probe ran and no CUDA device was involved: the record was written through the same writer the
reviewed test's positive control uses (`test_release_blocker_gpu_records.py:141-158`). So `VERIFIED`
currently means *"a well-formed record signed under this deployment's key, whose key names this
model + interpreter + contract + host, exists"* — strictly stronger than the baseline it replaced,
but not "a probe ran". §5.5.

---

## 4. Test-quality findings on the two reviewed suites

* **Not vacuous.** Dropping the HMAC check fails 2 of the 5 GPU tests; dropping the
  unreadable-file removal fails `test_an_unreadable_gpu_capability_artifact_is_reported_not_swallowed`.
  The suites do their job on the paths they name.
* **No test covers `interpreter_fingerprint` (D6).**
  `test_a_record_for_another_host_is_not_evidence` (`:237-255`) relabels the *host*; nothing
  exercises the fingerprint. Mutation check: neutering the capability-key comparison
  (`release_verify.sh:591`) leaves all 12 reviewed tests green.
* **Documented false negative.**
  `test_final_hardening_regressions.py:637-648` writes a fixture whose `capability_key` is the
  constant `"interpreter_probe"` and then asserts `real_gpu`/`status` as if it were the one record
  the scan accepted. No environment can ever have that key (it is derived), so the test pins the
  *unreadable* branch, not the mechanism its docstring describes. Replicating the fixture exactly
  (same root, same key env, same payload, plus `leftover.json`) shows what the scan really says:

  ```
  status UNREADABLE real_gpu False dim PENDING
  unreadable ['leftover.json']
  unverified ["614b825e…cfb.json: the record names an unknown model 'fixture_model'"]
  ```

* **Detail loss is untested** (§5.1): erasing the J1 failure detail (`:118`) leaves all four J1
  tests green, because they assert on captured stdout/stderr, never on `checks.J1.detail`.

---

## 5. Findings

### 5.1 P2 — CONFIRMED_BUG: a failed check's reason is not persisted in the report
`workbench/docker/release_verify.sh:66,349,425` (stdout-only capture) + `:42-44` (`DETAILS=$2`) +
`:471-486` (report assembly). Observed: live J1 failure → `checks.J1 == {"status": "FAIL", "detail": ""}`
while the reason existed only on the console. Criterion (f): the reason for a rejection must remain
inspectable — it does not, in the machine-readable artefact a release decision is recorded from.

### 5.2 P2 — CONFIRMED_BUG (false failure): a documented dev state now blocks the release
`workbench/docker/release_verify.sh:93-95`. Observed: the deployment's six `mock_*` fixtures make J1
`FAIL` and the verdict `RELEASE_BLOCKED`, although `AUTOMATION_RUNBOOK.md:174-176` defines them as a
supported non-evidence state and `clean_generated.py` exists to remove them. Either the mock
workflow is retired (then the cleanup step belongs in the release procedure and the runbook), or J1
must distinguish "mock present, not evidence" from "non-mock without corroboration".

### 5.3 P3 — Resolved by refutation
The cross-process config-cache defect I initially wrote here does not exist: `load_registry()`
re-reads per call (`registry.py:20-26`) and the repeated run was green both times. Recorded in §2.5
so the false lead is not re-investigated.

### 5.4 P1 — CONFIRMED_BUG: the environment binding named in the header comment is a presence check
`release_verify.sh:581-585` + `backend/runtime.py:1216-1218,1634-1636,1659-1676`. Evidence: probe
D6 above — a signed record whose `interpreter_fingerprint` belongs to a different environment is
reported `status=VERIFIED`, `real_gpu=True`, `GPU_RUNTIME=PASS`, and the runtime independently
reports `GPU_RUNTIME_READY`/`launch_ready=True`. Untested (§4) and mutation-invisible.

### 5.5 P2 — UNVERIFIED_RISK: `verified` does not require that a probe ran
`backend/runtime.py:960-997` (writer signs any dict) and `:1893-1913` (`verify_gpu_runtime`'s only
gate is a function-local `authorized` flag, and it early-returns from the persisted cache).
Evidence: probe D8 — a self-authored, runtime-signed record with no torch and no CUDA device reaches
`VERIFIED`/`PASS` here and `GPU_RUNTIME_READY`/`launch_ready=True` for the launch gate. Whether the
intended boundary is "artifacts written by the operator's own pipeline are trusted" cannot be
settled from this file alone, so this is classified as a risk against the verifier's claim rather
than a bug in the verifier.

---

## 6. Verdict

**SOUND_WITH_RESERVATIONS.** The wave fixes the two defects it was written for and I could not make
it certify a GPU by tampering with any field it *reads*: the HMAC, host, adaptation, dependency,
filename/key and synthetic-source bindings are all enforced and all named on rejection (probes
A–K), the four new test suites are load-bearing where they name a mechanism, and the honest and
adverse volume cases terminate with the true condition. Residual reservations, in priority order:

1. `interpreter_fingerprint` is never compared to the live environment (5.4) — a record for another
   environment is certified, and the reviewed suites cannot see it.
2. `verified` means a record exists, not that a probe ran (5.5) — the writer signs anything.
3. A failed check's reason is dropped from the report (5.1).
4. The six documented mock fixtures now block the release (5.2).
