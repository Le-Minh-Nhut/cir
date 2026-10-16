# Final PC Audit — Independent Review 5: `verify_run.py` / `pc.sh` Verification Chain

- **Target (reviewed only)**: `workbench/scripts/verify_run.py` (new), `pc.sh`, `workbench/tests/test_release_blocker_adversarial_final.py`, `workbench/tests/test_pc_launcher.py`
- **Baseline**: `b45c068`; branch `research/cir-failure-workbench`; fixes are uncommitted in the working tree
- **Method**: every claim below is a probe that was executed. Synthetic runs were built with the *repository's own* ledger writer/proof validator (`workbench/scripts/evaluate_models.py:append_evidence_entry`, `validate_run_manifest`) so that a "honest" fixture is honest by the system's own definition, never by hand-forged JSON.
- **Probe harness** (throwaway, `/tmp` only): `/tmp/rev5/gen.py` builds a run; `/tmp/rev5/drive.py` runs the probe matrix; `/tmp/rev5/drive_guards.py`; `/tmp/rev5/probe_jobs.py`; `/tmp/rev5/probe_noapi.py`; `/tmp/rev5/probe_j1_scan.py`.
- **Suite status (run, not re-audited)**: `test_release_blocker_invocation_proof.py` 14 passed; `test_release_blocker_adversarial_final.py` 12 passed; `test_pc_launcher.py` 25 passed.

---

## 1. Overall Verdict

**VERDICT: SOUND_WITH_RESERVATIONS**

The core chain is honest. `pc.sh` no longer fabricates a report and no longer prints a success
claim without a passing `verify_run.py`; the verifier's four corroborating ingredients (report
identity, HMAC ledger witness with `return_code==0`/`INVOCATION_COMPLETED`, completion proof,
LIMN category finiteness) all refused every forgery I could build inside the run tree, including
a tamper performed *after* the ledger entry was written. Two independent integrity paths remain
intact, and I could not make either lie about a run whose bytes it did not author.

Three reservations remain, one of which is a **false-pass path that the wave's own tests do not
cover** (a *leftover* verdict file, P1), one is a **false-failure that makes real LIMN runs
unverifiable from the host** (P1/P2), and one is a **container-reuse safety check that fails
open when the API does not answer** (P2).

---

## 2. Attack surface: attempt to make the verifier lie

All probes use a fully consistent synthetic run produced by the repository's own writer unless
stated otherwise. `verify_run.py` was invoked exactly as `pc.sh` invokes it:

```
CIR_REPO_ROOT=<root> WORKBENCH_LEDGER_KEY=<key> \
  python3 workbench/scripts/verify_run.py --run-id <id> --model limn \
    --checkpoint base_iter0_all_categories --protocol fashioniq_val_split --json
```

| # | Probe | Command | Observed |
|---|-------|---------|----------|
| v01 | Honest synthetic run (positive control) | `python /tmp/rev5/gen.py /tmp/rev5/p1 run_r5_honest` then verifier | `rc=0`, `{"verified": true, "reasons": []}` |
| v02 | Proof belongs to a *different* run id (old proof replayed under a new run id) | `verify_run.py --run-id run_v02` with `proof.run_id="run_v02_old"` | `rc=1`, `completion proof rejected: declared output belongs to run 'run_v02_old', not the current run 'run_v02'` |
| v03 | Replay an old run's ledger entry for a new run id | ledger entry keyed `run_v03_old`, verifier asked for `run_v03_new` | `rc=1`, `the evidence ledger holds no entry for run run_v03_new` |
| v04 | Report tampered **after** its ledger entry exists | overwrote `observed_metrics` in the report file | `rc=1`, `the evidence ledger's report digest does not match the report file; completion proof rejected: declared output report digest does not match its report` |
| v05 | Forged ledger entry (digest recomputed, HMAC not) | hand-edited `entry_digest` in `evidence_ledger.jsonl` | `rc=1`, `the evidence ledger is missing or its HMAC chain does not verify` |
| v06 | NaN / out-of-range Recall | `dress.r10=nan`, `shirt.r10=-5.0`, `toptee.r50=105.0` | `rc=1`, `category dress r10 is not a finite Recall in [0, 100]: nan` (and the other two) |
| v07 | Complete-looking run missing one category | `category_metrics` without `toptee` | `rc=1`, `category toptee is missing from the report` |
| v08 | Recall **exactly at the bounds** 0 and 100 (controls, `int` and `float`) | `dress.r10=0.0/r50=0`, `shirt=100.0/100`, `toptee=0/100.0` | `rc=0`, `verified: true` — **bounds are inclusive, no off-by-one** |
| v09 | Proof/report carry a *different* environment, report not touched | `proof.environment_digest="other_env_digest"` | `rc=0`, `verified: true` |
| v10 | Report identity forged (other run/checkpoint), ledger absent for it | report `run_id="run_other"`, `checkpoint_id="base_iter0_dress"` | `rc=1`, `report run_id is 'run_other', expected 'run_v10'` |
| v11 | Completion proof deleted, report+ledger intact | removed `*_completion.json` | `rc=1`, `completion proof rejected: declared output missing: …_completion.json` |
| v12 | Category present but `null` | `shirt: {"r10": null, "r50": null}` | `rc=1`, `category shirt r10 is not a finite Recall in [0, 100]: None` |

**Conclusion.** No probe satisfied the verifier without a real, ledger-witnessed, proof-backed
run. (c) *a way to make `verify_run.py` exit 0 on a forged/tampered run* — **not found** for
tampering inside the run tree. The reason is structural, not incidental: `verify_run.py` calls
the evaluator's own `validate_run_manifest`, which cross-checks the proof against the report
*digest*, the report against the ledger *entry digest*, the run directory name against the run
id (`evaluate_models.py:943-968`), `command.json` against the proof, and the `stdout/stderr`
digest against `evidence_digest`.

### 2.1 v04 is the important one, and it is correct

The wave's own test `test_adv03` forges a ledger entry *whose HMAC is not valid*, which is the
easy half. My v04 forges nothing: it lets the repository write the ledger entry, then rewrites
the report. The digest witness catches it, and the proof's `require_checksum` path catches it
independently. Two checks, either of which suffices.

---

## 3. `pc.sh`: can it still print a success claim without a passing verification?

`validate_and_summarize()` (`pc.sh:1037-1178`) now:

1. resolves `verify_run.py` (repo root, then `SCRIPT_DIR`) and **fails closed if absent**
   (`pc.sh:1060-1063`);
2. runs it with `--json`, captures `rc` (`pc.sh:1068-1069`);
3. parses the verdict in an embedded Python block that `sys.exit(...)`s unless
   `rc == 0` **and** `verified is truthy` (`pc.sh:1071-1108`), so the process aborts before any
   success line is printed;
4. prints `CORROBORATED` only after that gate (`pc.sh:1131-1137`).

The banner text is printed only after the gate, and the gate is *stricter* than the printed
claim needs: `rc != 0` alone aborts even if the JSON says `verified: true`.

| # | Probe | Command | Observed |
|---|-------|---------|----------|
| b1 | `verify_run.py` **missing** | delete the script, call `validate_and_summarize` | `rc=1`, `Giai đoạn thất bại: VERIFICATION`, `Không tìm thấy công cụ xác thực độc lập verify_run.py tại …`; `CORROBORATED` not printed |
| b2 | Payload **malformed** (verifier exits 0 printing `{not json at all`) | stub verifier | `rc=1`, aborts inside the parser; no `CORROBORATED` |
| b3 | Payload valid JSON but no `verified` key | stub prints `{"reasons": []}`, exit 0 | `rc=1`, `verify_run.py trả về mã lỗi 0`; no `CORROBORATED` |
| b4 | Verifier exits **1** while claiming `verified: true` | stub | `rc=1`, `verify_run.py trả về mã lỗi 1`; no `CORROBORATED` |
| b5 | Verifier exits 0 claiming `verified: true` for a nonexistent report | stub | `rc=0`, `CORROBORATED` printed — **the claim is exactly as trustworthy as the verifier; see §7** |
| b6 | Real verifier, sandbox with **no** report/ledger/proof | repository's own `verify_run.py` | `rc=1`, `aggregate report not found: …`; no `CORROBORATED` |
| c1 | **Full `pc.sh` run**, run id taken from `/api/runs`+`/api/benchmark-runs` (a run the ledger does not know), all evidence present and consistent except the ledger | `run_pc(sandbox, [], env)` with the corroborated fixture | `rc=1`, `Xác thực thất bại: … the evidence ledger's report digest does not match the report file; completion proof rejected: declared output report digest does not match its report` |
| c2 | **Full `pc.sh` run**, API serves an empty `/api/runs` and an unverified last `/api/benchmark-runs` entry; `validate_and_summarize` called with that run id | see `probe_noapi`/`drive_vsum` | `rc=1`, `the evidence ledger holds no entry for run run_limn_forged_02; completion proof rejected: declared output missing: …` |

**Conclusion.** (b) *a path where `pc.sh` still prints a success claim without corroboration* —
**not found** for real runs. The `CORROBORATED` string is reachable only through the embedded
verdict parser, and the parser refuses a missing script, a malformed payload, a missing
`verified` key, and a non-zero verifier exit code. Note the resolution order in §7: b5 shows the
claim is only as strong as `$REPO_ROOT/workbench/scripts/verify_run.py`.

### 3.1 One path where the API returns nothing (informational, not a false pass)

`execute_evaluation()` still has the pre-existing fallback (`pc.sh:639-648`, unchanged from the
baseline) that, when the job carries no run identity, adopts **the last** `/api/benchmark-runs`
entry as `FINAL_RUN_ID`. That value is now *verified* before use, and my c2 probe confirmed a
fabricated last entry is rejected with the reasons above. So the fallback can no longer select a
success; the residual risk is only that a *legitimate* last entry is adopted as the reported run
id (a mislabelling, never a false pass). Not a finding against this wave.

---

## 4. `jobs.py` verdict chain (transitive to `verify_run.py`'s claims)

`settled_status()` (`workbench/backend/jobs.py:126-152`) publishes `completed` only for
`PROOF_VALIDATED`, and settles `failed` for a dispatch that recorded no verdict. I drove the real
`SingleGpuQueue` with a **leftover verdict file** and a child that exits non-zero:

| # | Probe | Command | Observed |
|---|-------|---------|----------|
| j1 | `logs/<model>__<ckpt>__<run_id>/invocation_outcome.json` pre-seeded (or left over from the same directory reused by an operator retry) with `verdict="PROOF_VALIDATED"`, `completed: true`; child exits 3; no report, no proof, no ledger entry | `/tmp/rev5/probe_jobs.py` | `child exit: 3`, **`job status: completed`**, `job error: None`, `proof on disk: none` |
| j2 | Same, but the child *also* prints output | `/tmp/rev5/probe_proof_dropped.py` | `child exit: 3`, **`job status: completed`** |

**FINDING 1 (P1) — the outcome file is trusted on `run_id` equality alone; it is not bound to
this dispatch and not bound to any proof.**

- **Evidence**: `workbench/backend/jobs.py:107-123` reads
  `Path(run_directory)/"invocation_outcome.json"` and accepts it when
  `payload.get("run_id") == provenance["run_id"]`. Nothing else is checked — not `capability`,
  not a dispatch nonce, not mtime, and (unlike every other artifact in this chain) not the
  proof/ledger, so the queue will publish `completed` for a dispatch that never produced a
  report, a proof, or a ledger entry.
- **Why the directory can be pre-existing**: `jobs.py:183-189` creates *only* `queue_jobs/<id>/`;
  the run directory belongs to the operator. `evaluate_models.invoke_evaluator()` claims it with
  `mkdir(exist_ok=False)` (`evaluate_models.py:523-528`) and **aborts the run** if it already
  exists — so a retry of the same run id after an interrupted first attempt fails before it can
  rewrite the stale verdict. A verdict file left by any earlier attempt in that directory is
  therefore read as this dispatch's verdict, whatever the child actually did.
- **Severity rationale**: this is exactly the defect class the wave set out to close ("a job can
  reach `completed` without a validated proof"). It is *not* reachable through `pc.sh` (which
  re-verifies the run before claiming success) and it needs write access to the artifacts tree —
  the same pre-condition as writing the report itself — but the queue's own test suite covers
  only the missing-verdict direction, so the guarantee `completed ⇒ validated proof` does not
  hold as stated for the job status.
- **Smallest fix direction (not implemented — read-only review)**: have the queue confirm the
  verdict against the artifacts rather than the file alone — e.g. require the operator to stamp
  the dispatch identity into the file and compare it, and/or run the same
  `em.completion_validator(...)` the verifier uses. Either is a check against evidence that
  exists; the current check is against a file the dispatcher did not author.

---

## 5. Was a legitimate run wrongly rejected? (false failures)

### 5.1 Different interpreter / environment

| # | Probe | Command | Observed |
|---|-------|---------|----------|
| e1 | Run recorded when `WORKBENCH_LIMN_PYTHON=/usr/bin/python3`; verification with a *different* configured interpreter | `WORKBENCH_LIMN_PYTHON=/usr/bin/python3 verify_run.py …` | `rc=0`, `verified: true` |
| e2 | Verification with `WORKBENCH_LIMN_PYTHON=/bin/sh` (not even an interpreter) | same | `rc=0`, `verified: true` |
| e3 | Proof's `environment_digest` replaced with another environment's digest; report untouched | v09 above | `rc=0`, `verified: true` |

**FINDING 2 (P2) — the environment identity is an integrity check, not a comparability check.**

The task asks whether a run "whose proof exists but for a different interpreter/environment" is
rejected. It is **not**. `verify_run.py` passes `require_environment_identity=True`, but that flag
(`evaluate_models.py:938-1054`) only asserts the *proof file carries* `environment_digest`,
`evidence_digest` and `run_directory`, that the run directory name matches the run id, and that
`command.json` + `stdout/stderr` agree with them. It never recomputes the environment and never
compares the digest to `resolve_config()` or to a live interpreter probe. `verify_run.py`'s own
`identity` block therefore reports the report's `git_commit` and `runtime_adaptation` only.

Consequence in the honest direction: `environment_digest` is currently *nominal evidence* — every
probe above verified successfully under an environment that never produced the run, because the
records are self-consistent. The run's own report does carry real environment data
(`environment`, `model_interpreter`, `environment_digest`, `evaluate_models.py:1557-1562`), and a
consumer that needs "which interpreter produced this" can read that from `report_path`; the
verifier's `identity` block simply does not assert it.

### 5.2 Container/host checkpoint-path aliasing — **the false failure I did find**

| # | Probe | Command | Observed |
|---|-------|---------|----------|
| p1 | Proof+report record the bundle manifest the way the **container** computes it (member `path` under `/checkpoints/limn/…`), same bytes as the host copy | `/tmp/rev5/probe_mount_alias2.py` | `rc=1`, `completion proof rejected: checkpoint members no longer match the recorded artifact manifest` |
| p2 | Digest comparison of the same bytes under container vs host declared paths | inline probe | `container-recorded: f0baf5aa…`, `host-recomputed: 45394b8b…`; equal: **False** |

**FINDING 3 (P1) — `bundle_manifest_digest` fingerprints the *mount path string*, so a
docker-produced LIMN bundle run cannot be verified from the host.**

- `build_artifact_manifest()` (`evaluate_models.py:1040-1057`) stores `"path": str(member_path)`
  and `manifest_digest()` (`evaluate_models.py:1089-1091`) hashes that dict including `path`.
  `validate_run_manifest._members_live()` (`evaluate_models.py:729-763`) recomputes the manifest
  from `plan.checkpoint_bundle` / `plan.directory_members` — which
  `completion_validator._resolve_bundle()` (`evaluate_models.py:1742-1767`) builds from **this
  host's** `config.WORKBENCH_CHECKPOINT_ROOT`.
- The container's `WORKBENCH_CHECKPOINT_ROOT=/checkpoints` (`workbench/docker/.env.example:16`)
  vs the host's `workbench/artifacts/checkpoints` are the same bytes on two different paths
  (`docker-compose.yml:54`), and the digest covers the path. `completion_validator` never
  rewrites `path` to a canonical relative form before comparing, so the recorded digest can never
  be reproduced host-side.
- **This is the same class of defect the `p2`-contract test guards against**: a report is real
  evidence only when corroborated, and J1 rejects an uncorroborated run as *not real* — so a
  genuinely docker-produced LIMN run is rejected twice (by J1's scan, and by `verify_run.py` when
  the host re-checks it). The wave's positive control
  (`test_pc_launcher._create_corroborated_run`, `test_pc_launcher.py:636-780`, and
  `test_release_blocker_adversarial_final.py:test_adv12`) builds the manifest with the host's own
  builder, i.e. against the host paths, so the false failure is structurally invisible to the
  suite.

### 5.3 Boundary values 0 and 100

Probe v08 (both `int` and `float`, on three categories) verified `rc=0`. `_is_recall`
(`verify_run.py:38-44`) rejects `bool` explicitly and accepts `0.0 <= v <= 100.0`; no off-by-one,
no type coercion hazard. **NOT_A_BUG.**

### 5.4 LIMN bundle run (positive control)

Probe v01 is a LIMN bundle run whose proof was validated by the repository's own
`validate_run_manifest` through `completion_validator` with `require_checksum` and
`require_environment_identity` — verified: `rc=0`. Probe v08 is the same run with boundary
values. **PASS.**

---

## 6. Evidence loss and inspectability

| # | Probe | Observed |
|---|-------|----------|
| x1 | Job settles `failed` on a rejected/silent proof | `job.return_code` keeps the **native** child code (`jobs.py:530`), `job.error` carries the operator's recorded `reason` (`evaluate_models.py:1690` → `jobs.py:135-151`), and the reason is appended to `stderr_tail` (`jobs.py:535`). Test `test_release_blocker_invocation_proof.py::test_a_nonzero_exit_records_the_native_code_as_evidence` asserts `return_code == 7`. **Preserved.** |
| x2 | Verifier refusal reason | printed to **stderr** as `[FAIL] <reason>` (`verify_run.py:126-128`) and returned as a `reasons[]` array; `pc.sh` re-emits the joined reasons in its failure banner (`pc.sh:1097-1105`). **Preserved.** |
| x3 | Artifact path on success | `report_path` and, on failure, the exact path *inside* the reason (e.g. `declared output missing: /…/_completion.json`). **Preserved.** |
| x4 | Verdict artifact | written twice — beside the report at
`reports/<model>_<ckpt>_<run>_invocation_outcome.json` and in the run directory
(`evaluate_models.py:1407-1408`). **Preserved.** |

No evidence loss found.

---

## 7. Residual risk: same-tree verifier resolution (INFO, pre-existing shape)

`pc.sh` resolves the verifier as `$REPO_ROOT/workbench/scripts/verify_run.py` then
`$SCRIPT_DIR/workbench/scripts/verify_run.py` (`pc.sh:1057-1063`), i.e. inside the tree it is
verifying. Probe b5 shows that a substituted verifier claiming `verified: true` produces
`rc=0` + `CORROBORATED`. This is inherent to "run a script from the checkout": an attacker with
write access to the checkout can substitute the script, the same way they can substitute
`python3`. It is **not new**, and it is not the run-tree forgery the wave is about (an attacker
confined to `workbench/artifacts/` cannot reach it). Recorded as INFO for the release decision,
not as a defect of this wave.

---

## 8. Findings summary

| ID | Sev | File:line | Class | Statement |
|----|-----|-----------|-------|-----------|
| F1 | **P1** | `workbench/backend/jobs.py:107-123`, `:183-189`; `workbench/scripts/evaluate_models.py:523-528` | CONFIRMED_BUG | `invocation_verdict()` accepts a pre-existing `invocation_outcome.json` on `run_id` equality alone. A leftover verdict file (`PROOF_VALIDATED`) makes a dispatch whose child exits non-zero and which produced no report/proof/ledger entry settle **`completed`** (probes j1/j2: `child exit: 3`, `job status: completed`). The queue's guarantee "`completed` only on `PROOF_VALIDATED`" therefore does not mean "only on a validated proof". Not reachable via `pc.sh`, which re-verifies. |
| F2 | **P1** | `workbench/scripts/evaluate_models.py:1040-1057`, `:1089-1091`, `:729-763`, `:1742-1767` | CONFIRMED_BUG | `bundle_manifest_digest` includes each member's absolute `path`. A docker-produced run records `/checkpoints/limn/…`, the host recomputes `…/workbench/artifacts/checkpoints/limn/…`, digests differ, and a legitimate LIMN bundle run is rejected (`checkpoint members no longer match the recorded artifact manifest`, probe p1). The positive-control fixtures cannot see it because they build the manifest host-side. |
| F3 | **P2** | `workbench/scripts/verify_run.py:104-108`; `workbench/scripts/evaluate_models.py:938-1054` | UNVERIFIED_RISK (behaviour confirmed) | `require_environment_identity` is self-consistency only: no recomputation, no comparison with `resolve_config()` or a live interpreter. Probes e1/e2/e3 verify a run under `/usr/bin/python3`, under `/bin/sh`, and after rewriting `environment_digest`, all `rc=0`. A consumer that needs "was this produced by the recorded interpreter" must read the report's own `environment`/`model_interpreter`, not the verdict. |
| F4 | **P2** | `pc.sh:626-641` | CONFIRMED_BUG | The active-job check for container reuse swallows any API failure into `0 none` (probe: non-list payload and empty body both yield `0 none`). With the container's task list unreadable, a mismatched-configuration container is restarted instead of refused — the check that exists to protect an in-flight job fails **open**. |
| F5 | P3 | `pc.sh:1131-1137` | NOT_A_BUG / INFO | All displayed values (`git_commit`, `runtime_adaptation`, `gate`, metrics) are read from the *report bytes* named by the verified `report_path`; the report is digest-witnessed by the ledger and re-hashed by the proof. Frame "Conversation/Environment Identifier" is not among them, so a report departing from the run's recorded environment is not misreported by `pc.sh` — but nothing in `pc.sh` cross-checks the report's `model_interpreter` against the container either (same root cause as F3). |
| F6 | INFO | `pc.sh:1057-1063` | UNVERIFIED_RISK (behaviour confirmed, pre-existing) | The verifier is resolved from the tree under verification; a substituted verifier yields `rc=0` + `CORROBORATED` (probe b5). Inherent to running a checked-in script; not reachable from the artifacts tree. |

**Explicitly NOT_A_BUG (probed):** v04 post-ledger report tampering; v06/v12 NaN/`None`/out-of-range
Recall; v07 missing category (a valid LIMN bundle run with a missing category is structurally
impossible, so the check's value is against a truncated *report*, which is exactly what it
catches); v08 bounds 0/100 inclusive; c1/c2 `pc.sh` never printing `CORROBORATED` on a
non-verifying run; b1–b4, b6 the missing/malformed/lying-verifier paths.

---

## 9. Recommendation

- **F1** is the one that should block a release claim of the form "a job is `completed` only
  with a validated proof". The check is cheap: bind the verdict to the dispatch (or re-validate
  the proof in `invocation_verdict`), and make `evaluate_models.invoke_evaluator` refuse to
  start when the run directory already holds a verdict it did not write (today it refuses the
  directory, leaving the stale verdict in place).
- **F2** should be confirmed against a real docker run before the release: if the container
  records `/checkpoints`-rooted manifests, every docker-produced LIMN run is currently
  unverifiable host-side (and J1 will call it "not real evidence"). Canonicalising the manifest
  to relative member names (or comparing only `filename`+`sha256`) fixes it at the builder.
- **F3/F5** are documentation-grade: either assert the environment in `verify_run.py`, or state
  in the report docstring that `environment_digest` is an in-run integrity witness, not a
  comparability claim.
- **F4** is a two-line change: distinguish "API answered: no active jobs" from "API did not
  answer", and fail the reuse path in the second case.
