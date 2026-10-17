# AUDIT_A — slice A: `pc.sh`, `workbench/docker/`, `workbench/config/`, `.dockerignore`, `serve_workbench.py`, `operator_config.py`

Iterative audit round 01. AUDIT ROUND 01 / slice A. Branch `research/cir-failure-workbench`, HEAD `92ecbd94052bc7b4829abd4314af8b63d7d6bea3` + an UNCOMMITTED fix wave.

READ-ONLY: the only file written under `/home/heheboiz/data/cir` is this report. Scratch is under `/tmp`.
No docker build, no download, no whole pytest suite. Targeted runs: `test_pc_launcher.py` (37 passed, 79s),
`test_concurrency_ownership.py` (11 passed), `test_http_api_layer.py` + `test_release_blocker_gpu_records.py` (45 passed),
`pc.sh --doctor` (rc=0), `release_verify.sh` (offline + live), one mutated copy of the verifier in `/tmp`.

## 0. IMPORTANT — `pc.sh` and `workbench/backend/main.py` CHANGED UNDER THIS AUDIT

Snapshot taken at the start, re-verified before writing:

| file | mtime | md5 |
|---|---|---|
| `pc.sh` | 2026-10-16 23:31:51 | `6b8ce8dd001e7d509afe633969405e55` |
| `workbench/docker/release_verify.sh` | 2026-10-16 23:11:05 | `2e0eeef402d285a1d93414929ac253fe` |
| `workbench/backend/main.py` | **2026-10-17 00:04:55** | `74f1495ce14a83dc1065ed6ad016470a` |
| `workbench/backend/runtime.py` | 2026-10-16 23:05:19 | `b5563822f727460bfe54b0a1b73832c9` |
| `workbench/backend/jobs.py` | 2026-10-16 23:02:19 | `37e030e709c9109048328f38238c7725` |

`workbench/backend/main.py` was rewritten (792 → 824 lines) **while this audit ran**: `_development_receipt_refusal()`
(observed live, and still present in the 23:01 `git diff`) was replaced by `_development_process()` /
`_development_opt_in()` / `_certification_mode()` / `_no_proof_refusal()`. Verdicts below state which revision each
finding was checked against. **`pc.sh` was read in place at 23:31 and was NOT re-read afterwards — so was
`workbench/docker/release_verify.sh`. Every `pc.sh`/`release_verify.sh` finding is against the md5 above; re-confirm
line numbers before acting.**

Coverage target `SOURCE_MANIFEST.md` names 100 files; this slice owns 6 of them (`pc.sh`, `workbench/config/workbench.env.example`,
`workbench/docker/{.env.example,Dockerfile,acceptance.sh,docker-compose.gpu.yml,docker-compose.yml,mutation_checks.sh,release_verify.sh,run.sh}`)
plus the slice-scoped but unlisted `.dockerignore`, `workbench/scripts/serve_workbench.py`, `workbench/backend/operator_config.py`.

---

## A1 (CONFIRMED, P0) — `pc.sh` fail-closed messages lose the reason they exist to carry

**Evidence.** `pc.sh:845-866` appends the API-unreadable reason to `mismatch_reasons`; `pc.sh:868-874` then declares
`local mismatch_reasons=()` a SECOND time in the same function scope and re-copies from `inspect_mismatch_reasons`:

```
871:            local mismatch_reasons=()
872:            if (( ${#inspect_mismatch_reasons[@]} > 0 )); then
873:                mismatch_reasons=("${inspect_mismatch_reasons[@]}")
874:            fi
```

`local` re-initialises, so the entry pushed at line 865 is destroyed.

**Reproduction** (`/tmp/arr_repro2.sh`, exact replication of 844-874 + 879-892):

```
config_satisfies=0 mounts_known=1
mismatch_reasons=1 : Mount mismatch A
FAIL_STAGE message would be: Container ... (Mount mismatch A)
```
The `jobs_known == 0` branch at `pc.sh:884-886` interpolates `reasons_str`, so its stated reason
"…cấu hình lại container lúc này có thể phá một thí nghiệm đang chạy. Chi tiết: ${reasons_str}" omits the
*only* explanation of the first clause ("Không đọc được danh sách tác vụ …"). `pc.sh:891-893`
(`mounts_known == 0`) has the same shape.

**Root cause.** Duplicate `local` declaration of the same name in one function (bash re-initialises both times);
verified: `f(){ local x=1; local x=2; echo $x; local a=(); local a=(); echo ${#a[@]}; }` → `2` / `0`.

**Expected vs actual.** Expected: the operator sees the API-unreadable reason in the CONTAINER_REUSE failure.
Actual: it is dropped; the surviving text reads as a *configuration* problem.

**Minimal corrective action.** Delete line 871 (`local mismatch_reasons=()`); the `+=` at 865 already created the
array. `pc.sh` line 869-870 declare the other scalars fresh and are correct.

**Not caught by tests — and the tests depend on this bug.** `test_pc_launcher.py:1128,1236`
(`MOCK_JOB_STATUS`/`MOCK_API_DOWN`) match only words that survive the wipe. Two tests that *should* cover the
position-fixed wiring pass only because of it: `test_pc_launcher.py:1143-1147` and `:1001` set
`MOCK_CONTAINER_ADAPTATION = "none"` — a value `pc.sh:514-573` cannot produce (it exports `""` or `sm_120`), so the
mismatch is spurious and never reaches the second `inspect_config_satisfies` assignment. **If A1 is fixed without
correcting these two tests they will fail**, and the failure will look like a regression in the correction path.

## A2 (CONFIRMED, P0) — `release_verify.sh` cannot pass J1 in any repo that has `.venv/`; the released `--offline` result is not reproducible here

**Evidence.** `release_verify.sh:31-32`:

```
31: PY="$(command -v python3 || true)"
32: [[ -x "${REPO}/.venv/bin/python" ]] && PY="${REPO}/.venv/bin/python"
```
`release_verify.sh:486-489`:
```
486: "${PY}" - "$REPORT" "$@" <<'EOF'
487: import json, sys
488: report_path, = sys.argv[1:2]
489: EOF
```
In `.venv` there is no `yaml`. Observed (report 23:07, before `main.py` was rewritten):

```
$ bash workbench/docker/release_verify.sh --report /tmp/rv_live.json
  FAIL J1 Traceback (most recent call last):
  File "<stdin>", line 25, in <module>
  File "…/workbench/backend/main.py", line 15, in <module>
    from fastapi import FastAPI, HTTPException, Request
ModuleNotFoundError: No module named 'fastapi'
J12  PASS …
RELEASE VERIFIER: FAILED (1 unmet, 1 pending)
```
Cause of the missing `yaml`: `release_verify.sh:86` does `import yaml` for the registry, and `PY` is the venv
interpreter. With `--offline` the same block still runs (J1 is entered, then `import yaml` fails).

**Independence — this is NOT the mutation I injected.** In `/tmp/rvmut2/repo` (complete repo copy, only
`raise RuntimeError("simulated internal verifier failure")` injected at the top of the report block) the run failed
at J1 first; the injected mutation never executed. **When J1 is made to pass, the injected failure is silently
swallowed and the script still exits 0** — see A9. So the true severity here is: *the verifier cannot reach its own
report block on this deployment shape, and when it does reach it a crash there is invisible.*

**Expected vs actual.** Expected (this release's own documented result, `docs/reviews/FINAL_PC_RELEASE_REPORT.vi.md:68`):
"`release_verify.sh --report /tmp/rv_report.json` … 12 PASS, verdict=RELEASE_READY, exit 0". Actual on this host:
J1 FAIL, exit 1, `RELEASE_BLOCKED` in the JSON.

**Minimal corrective action.** Delete lines 486-489 (dead: no consumer reads `report_path`; the live block at 497
re-derives it from `sys.argv[1]`). If the `.venv` pin is to stay, pin the missing dependency too — but the lazy fix
is the deletion.

## A3 (CONFIRMED, P1) — `report_path, = sys.argv[1:2]` also breaks on the documented launcher invocation

**Evidence.** `--base` is an accepted flag (`release_verify.sh:25`), so `"$@"` is non-empty and `sys.argv[1:2]` holds
two elements. Reproduced:

```
$ python3 -c 'import sys
try:
    report_path, = sys.argv[1:2]
    print("argv ok")
except ValueError as e:
    print("HEREDOC VALUEERROR:", e)' --base http://127.0.0.1:8000
HEREDOC VALUEERROR: not enough values to unpack (expected 1, got 0)
```
(0 elements shown because the extra positional is consumed by the slice; with `"$@"` appended the slice is
`["--base","http://…"]`, 2 elements, same `ValueError`.) Dead code with two failure modes; no verdict.

**Action.** Same as A2 — delete the block.

## A4 (CONFIRMED, P1) — `pc.sh` mandates an in-container observation for GPU and adaptation; on this fleet neither can ever succeed

This is the host↔container boundary defect the slice was asked to attack, and it is live *right now*.

**A4a — the running container has no GPU request and no toolkit runtime.**
```
$ docker inspect --format '{{json .HostConfig.DeviceRequests}}' cir-workbench
null
$ bash pc.sh --doctor
[OK] NVIDIA Driver & GPU    : Driver 580.178.04 | Thiết bị: NVIDIA GeForce RTX 3050 Laptop GPU (Compute 8.6)
[PENDING] NVIDIA Container Runtime: Chưa phát hiện runtime 'nvidia' trong Docker
```
`pc.sh:612-630` therefore sets `inspect_config_satisfies=0` **unconditionally** for any run (GPU is enabled by
default; `--no-gpu` is the only escape). `inspect_container_config` is now the single source of truth for reuse
(868) *and* for the post-restart recheck (745, 901, 912), so `run.sh up` is requested, the container is recreated
with the same `.env`, and `require_corrected_configuration` fails closed — with the remedy text
"Sửa các khóa host path … HOST_FASHIONIQ_ROOT / HOST_WORKBENCH_CHECKPOINT_ROOT / HOST_WORKBENCH_ARTIFACTS_ROOT"
(`pc.sh:753`), which is **the wrong remedy for a GPU-missing container**. Correct behaviour (fail closed, no silent
CPU run), wrong diagnosis.

**A4b — the `nvidia` runtime check needs a new login session.** `pc.sh:405-446` installs
`nvidia-container-toolkit` and then re-tests `docker info --format '{{json .Runtimes}}' | grep -q '"nvidia"'` in the
same shell. With no `nvidia` runtime registered a patient run is therefore impossible.

**A4c — a 5070 Ti (sm_120) host with `WORKBENCH_ENABLE_GPU=1` is refused at the adaptation layer before anything
runs.** `pc.sh:505-541` exports `WORKBENCH_RUNTIME_ADAPTATION=sm_120` (models.yaml:946 `adapted_runtime.target_compute_capability: 'sm_120'`).
The container it is pointed at carries `WORKBENCH_RUNTIME_ADAPTATION=""`:
```
$ docker exec cir-workbench printenv WORKBENCH_RUNTIME_ADAPTATION
            (empty, rc=0)
$ bash /tmp/adapt.sh                      # exact replication of pc.sh:634-642
MISMATCH: container='' required='sm_120'
```
`pc.sh:632-643` therefore refuses, and `run.sh up` recreates from the same `.env` whose
`WORKBENCH_RUNTIME_ADAPTATION` line is still commented (`workbench/docker/.env:39`), then
`require_corrected_configuration` fails closed again — again naming host-path keys, not
`WORKBENCH_RUNTIME_ADAPTATION`. Two consecutive `run.sh up` calls to reach the same refusal is also the
"single-startup invariant" being violated in spirit: `test_pc_launcher.py:1186,1223` pass only because
`MOCK_CONTAINER_ADAPTATION` is set to `sm_120`, which no real `.env` in this tree produces.

**Expected vs actual.** Expected: on the target PC, `./pc.sh` prepares and evaluates LIMN on sm_120. Actual: any
healthy 5070 Ti host on first boot with default `.env` fails at CONTAINER_RECONFIGURE before preparation.

**Minimal corrective action.** Split `inspect_container_config`'s reasons into *correctable by `run.sh up`* (mounts,
adaptation env) and *host-level prerequisites* (no `nvidia` runtime, no `DeviceRequests`), and give
`require_corrected_configuration` a remedy that names the actual failing layer. And make `configure_environment`
always write `WORKBENCH_RUNTIME_ADAPTATION` / `WORKBENCH_ENABLE_GPU` into `.env` (it does so only under
`created_new_env == 1`, `pc.sh:566-579`), so "restart to correct" is not a no-op for those two keys.

## A5 (CONFIRMED, P1) — the ledger/capability key resolves differently in the host and in the container (false FAIL, both directions)

**A5a — the two key files disagree on this host, and the ledger cannot be read at all.**
```
$ XDG_CONFIG_HOME=[unset]; HOME=/home/heheboiz
$ sha256sum .workbench_ledger_key ~/.config/cir/workbench_ledger_key
4b081d9ab5f6c37524a7fafef3302335146d7d2ebc29727399bbf4686e220b3e  .workbench_ledger_key
763eb1f9b5f8c888c39d102723c66e5eda544c0d1b0a5fba07202f18a0946557  /home/heheboiz/.config/cir/workbench_ledger_key
$ python3 -c "...em.ledger_hmac_key(resolve_config())..."
ledger_hmac_key RAISED: LedgerKeyError two different ledger keys exist (…/.workbench_ledger_key and
  /home/heheboiz/.config/cir/workbench_ledger_key); keep one and remove the other, or set WORKBENCH_LEDGER_KEY
read_evidence_ledger: None
```
`evaluate_models.py:1145-1186` explicitly raises on the two-key state; `read_evidence_ledger` converts that to
`None` (`evaluate_models.py:1346-1347`), and `verify_run.verify:107-110` turns `None` into
"the evidence ledger is missing or its HMAC chain does not verify". **Any host-side verifier run therefore reports
every finished run as uncorroborated.** (The artifact itself is not corrupt: `workbench/artifacts/evidence_ledger.jsonl`
is a 1-entry `r1` fixture written 2026-10-16 02:33. The key files were created by test glue — both are untracked/ignored.)

**A5b — the container and the host compute different capability keys for the *same* environment.**
`runtime._gpu_capability_key:1032` takes the interpreter from `os.environ[python_env_var]` **only**, while
`verify_environment:1255-1277` adopts the persisted `environment_assignment.json` when the env var is unset. The
assignment exists and points at a real in-container interpreter:
```
limn -> RUNTIME_VERIFIED /artifacts/environments/cir-1bbd9872cf166f0d/bin/python
$ docker exec cir-workbench sh -c 'cd /repo && WORKBENCH_LIMN_PYTHON=<that> WORKBENCH_RUNTIME_ADAPTATION=sm_120 python3 -c "...runtime._gpu_capability_key(limn)"'
interp exists: True
capability key (container, sm_120): 417924a5d4a57f2e674d92fd99425740be6c26e4cd8ae8a845bccf68c877dc68
$ (host, same env)                                       host capability key (sm_120): 678c363d603f8e0a534f24833e1ade8feb732c72fc2fc922139e039b4fd8b299
$ (host, env var unset — the state on this box)          key WITHOUT env var: 28c012ba6818ea0cb3d5caa67ba3822429b1d1bd87f273293843f8100765acd6
$ (host, env var set to the same interpreter)            key WITH    env var: 4e89018ca18ceb805064a8702c170826194dddbf7b9b0bce0b62273607896b0f
```
Four distinct keys. `release_verify.sh:568-573` `_expected_key(model) = _runtime._gpu_capability_key(model)` is the
host-process key; `_unverified_reason` (`release_verify.sh:575`) also compares `record.host_identity != _runtime._host_identity()`, and
`_host_identity:1009` includes `os.uname().nodename` (the container's hostname vs the host's). **A record produced
anywhere other than the same process configuration is never accepted, and `GPU_RUNTIME` is permanently PENDING.**
`release_verify.sh` additionally sources no env file (only `CIR_REPO_ROOT` in J12) and only discovers `PY` from
`${REPO}/.venv/bin/python` — a `.venv` that does not exist on a clean target PC.

The record half of fix D is nonetheless **verified fixed** (I ran `_remember` directly against the current source):
```
$ WORKBENCH_NO_ENV_CACHE=1 python3 -c "...runtime._remember(model, False, report)..."
before  contract_digest = None
after   contract_digest = d92a85eef4925f45b15c60b59dc01ca6e0b5376f5b44de9c70f08a72b0e2e135
with cache enabled        = d92a85eef4925f45b15c60b59dc01ca6e0b5376f5b44de9c70f08a72b0e2e135
```
So `dependency_fingerprint: null` under `WORKBENCH_NO_ENV_CACHE=1` (runtime.py:2056 reading the ungated
`report.contract_digest`) is closed. **But the fix is narrow: `report.contract_digest` is still `None` on the
interpreter-less returns `runtime.py:1253-1255` and `:1279-1280`, which do not call `_remember`.** And any
`_remember` reached with `report.interpreter is None` returns before line 1690 only in the *fingerprint* branch;
line 1690 runs first, so covered. Residual risk is the `TIER_UNCONFIGURED` early returns only.

**Action.** One resolution function for "this model's interpreter" (env var, else assignment) shared by
`_gpu_capability_key`, `_cached_gpu_capability`, `_revalidate` and `release_verify`; or state in
`PC_TRANSFER_CHECKLIST.md` that the record must be produced and read by the same process configuration, and that a
second ledger key file is a hard error.

## A6 (CONFIRMED, P1, fixed mid-audit — verify) — the new API certification path refused legitimate corroborated runs

**Checked against the 23:01 revision** (the one in `git diff` when this audit started).
`main.py:348-350` (that revision) called `_development_receipt_refusal()` **before** looking at the proof, and that
function returned a refusal string unless `PYTEST_CURRENT_TEST`, `WORKBENCH_DEVELOPMENT_MODE`, or
`WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1` was set. Consequence: in a production container none of the three is
set, so `_certify_report` returned
`{"certified": false, "evidence_reasons": ["this workbench process is not the evaluator deployment …"]}`
**even when a valid completion proof existed** — `_verifier_verdict` was never called. That is a false FAIL against
`pc.sh:1116-1121`, which accepts a reuse candidate only when
`is_research_evidence` is true (`FINAL_RUN_ID` path, `pc.sh:1128-1131`), and against
`pc.sh:1263-1300 validate_and_summarize`, which hard-fails on `verify_rc != 0 or not verified`. Every
already-finished legitimate run would be re-evaluated.

**Re-checked against the CURRENT revision (00:04:55, md5 `74f1495c…`): the hole is closed.**
`main.py:225-244 _certification_mode(proof_exists)` returns `"deployment"` whenever a proof exists, and
`_development_process()` at 208-216 is now opt-in state, not the absence of a flag. Directly exercised:
```
== case 1: NO proof, production process ==
 mode: refused | receipt: certified=False, certification_mode=refused
== case 2: proof EXISTS ==
 mode: deployment | verifier called: 1
 receipt: {'is_research_evidence': True, 'certified': True, 'certification_mode': 'deployment'}
```
A development opt-in can no longer pre-empt the authoritative path for a report that has a proof.

## A7 (CONFIRMED, P1) — a report with no proof is refused even when it is the deployment's own artefact

Same revision. `_certification_mode` requires `proof_exists` for `"deployment"`; otherwise a production process gets
`"refused"` (`main.py:395-397`) with
"this report has no completion proof, so the authoritative verifier cannot validate its environment identity here".
Operationally: an aggregate report survives but its completion proof is deleted / pruned / lost, or a run was
recorded through a path that writes no proof → the API serves it with `certified=false`, so `pc.sh`'s dedup loop
(`pc.sh:1113-1122`) refuses to reuse it and re-runs hours of GPU work. The refusal is *correct* under "only
proof-validated certification", but nothing in this slice documents that the proof is an irreplaceable half of the
evidence pair, and `pc.sh` never checks for it before dispatching. **P2 as a bug, P1 as a diagnosis gap**: the
operator is told "no proof" only in the API JSON, never in the launcher's Vietnamese failure text.

**Action.** Either name the missing proof in `pc.sh`'s REGISTRY/VERIFICATION messages, or have `pc.sh` treat
`certification_mode == "refused"` with `execution_authenticated == true` as "do not reuse, do not claim failure" and
say so.

## A8 (CONFIRMED, P1) — `release_verify.sh` exits 0 with no report when its own Python block raises

**Evidence.** `release_verify.sh:15` is `set -uo pipefail` — **no `set -e`** — and the trailing report block's exit
status is never tested (`:497` … `:808`, then the `fails` test at `:810`). The last live case of A2/A6 proved it
end-to-end in `/tmp/rvmut2/repo` (only `raise RuntimeError("simulated internal verifier failure")` injected at the
top of that block, otherwise the unmodified script against the live stack with `.venv` present):

```
$ bash workbench/docker/release_verify.sh --offline --report /tmp/rvmut2/out5.json
… J12  PASS …
RELEASE VERIFIER: no unmet criteria (5 pending)
rc=0
report exists: NO
```
The script prints success, writes **no JSON report**, and exits 0. A crash anywhere in the 300-line scan (the
capability scan, the evidence scan, the dimension roll-up, the report write) is invisible. This is *not* the
mutation's own failure surfacing: on the same tree with a *failing* J1 the exit was 1 and the traceback was
attributed to J1 (`J1   FAIL     Traceback (most recent call last):`), i.e. crashes *inside* `$(...)` are reported
and crashes in this block are not.

**Action.** `|| { say "verifier report block failed"; exit 2; }` on the `:497` invocation, or add `set -e` with the
existing intentional non-zero paths reviewed (J8 returns 3 deliberately, `_unverified_reason` returns values, the
J11 `docker` calls are already guarded).

## A9 (CONFIRMED, P2) — the new `DEPLOYMENT`/`CONFIGURATION`/`OPERATIONS` dimensions are satisfiable; the NEW false FAIL is `J1`, not the dimension set

The dimension gating itself is **sound** and the "permanently PENDING" hypothesis is **FALSE_POSITIVE**:

```
$ bash workbench/docker/release_verify.sh --report /tmp/rv_live.json      # live stack, 27s
J2   PASS     …=>/artifacts …=>/checkpoints …=>/data/FashionIQ/images
J3   PASS     derived tree writable; images/ and image_splits/ reject writes (read-only)
J11  PASS     artifacts persisted across container restart (disposable container, isolated /tmp/…)
  STATIC_SOURCE PASS  DEPLOYMENT PASS  CONFIGURATION PASS  MODEL_ENVIRONMENT PASS  OPERATIONS PENDING
report written: /tmp/rv_live.json verdict=GPU_VALIDATION_PENDING
```
`DEPLOYMENT` and `CONFIGURATION` reach PASS on this healthy host. `OPERATIONS` (J8) is PENDING only because
`/api/jobs` is `[]` — the queue is empty, and J8 honestly refuses to certify a lifecycle it never observed
(`release_verify.sh:352-365`, `NO_JOBS_OBSERVED` → `pending`). **DEFECT: on a PC that has completed a run but has
no in-flight job, `OPERATIONS` PENDING plus `GPU_RUNTIME`/`MODEL_*` PASS yields `ENGINEERING_INCOMPLETE`, i.e. a
worse verdict than before the dimension split, for a state that is fully healthy.** (Verified the roll-up logic in
isolation: `OPERATIONS PENDING  statuses=['PENDING']`.) The `--offline` report's `DEPLOYMENT: PENDING` is likewise
from J11's `pending J11 "offline mode"`, not from a defect.

The real new false FAIL is **J1**, and it is not a dimension problem: it is A2. It also has a **second, independent**
false-FAIL cause: J1 boots `import yaml` and `from workbench.backend.main import _aggregate_reports` inside the
*script's* interpreter (`.venv` when present), so a verifier run in a repo whose `.venv` lacks the backend's
dependencies classifies every run as `misrepresented` and fails the whole dimension. The verdict roll-up
(`release_verify.sh:790-799`) is correct: empty-check dimensions get PENDING via `elif all(s in ("PASS","PENDING") …)`
on an empty list, which is intentional and matches the comment at `:512-515`.

**Action.** (a) fix A2; (b) give J8 a deterministic offline component (e.g. drive one `queued→cancelled` job through
the lifecycle in a disposable container, the way J11 is done) or explicitly document `OPERATIONS` as
"requires a pilot run" in the report's verdict text — otherwise `ENGINEERING_INCOMPLETE` will be read as a real
failure on the target PC.

## A10 — Checklist answers (the slice's named questions)

| Question | Verdict | Evidence | Severity |
|---|---|---|---|
| Mount provenance | **VERIFIED** (design), with an asymmetry | 7 bind mounts, `images/`+`image_splits/` RO: `docker inspect --format '{{json .Mounts}}' cir-workbench`; both RO (`RW:false`) and `docker exec -u 0 cir-workbench touch /data/FashionIQ/images/.probe` → `Read-only file system`. `/data/FashionIQ/captions` is **writable** (observed `CAPTIONS_WRITABLE`), so canonical inputs are protected by their own submount, not by being outside a writable tree | NONE / P3 (asymmetry undocumented) |
| NEW post-restart recheck | **VARIANT** — the recheck is wired (745, 901, 912) and fails closed; but it re-runs a check that can never pass for GPU/adaptation (A4), and its remedy text names the wrong keys. The `run.sh up` idempotence argument in `pc.sh:666-673` is stated but not enforced | A4; `pc.sh:742-755` | P1 |
| Single-startup invariant | **FAIL** for the A4 path (two `run.sh up` calls before the same refusal); **VERIFIED** for the `container_running == 0` fresh-start path (`pc.sh:917-920`) | live `pc.sh --doctor` + `/tmp/adapt.sh` | P1 |
| Ledger/capability key resolution + durability | **VARIANT** — resolution order and the two-key hard error are implemented and correct (`evaluate_models.py:1145-1193`); durability across recreation relies on `XDG_CONFIG_HOME` being declared in the IMAGE, and it is (`Dockerfile:56-71`); but the host and container disagree about which key/interpreter/hostname they mean (A5) | A5a/A5b | P1 |
| `docker exec` environment propagation | **VERIFIED** for the verifier (`pc.sh:1013-1017` re-declares the container's own `Config.Env`), **VARIANT** for its comment: it claims "The ledger key resolves from XDG_CONFIG_HOME" — the *container* resolves it from `.env`, and on today's container `XDG_CONFIG_HOME` is absent (`printenv` rc=1) because the `Dockerfile:56-71` change post-dates the running image | `pc.sh:1006-1017`; container `printenv` rc=1 | P2 (stale image) |
| Readiness polling | **VERIFIED** — cheap probe `/api/jobs`, bounded `min(TIMEOUT,180)`, both services checked, fails closed with the codes (`pc.sh:936-951`) | live run reached `[OK] Docker stack đã sẵn sàng` | NONE |
| Fail-closed on API/docker unreachable | **VERIFIED** — `running_jobs_known == 0` and `jobs_known == 0` both `fail_stage` rather than restart (`pc.sh:830-835`, `884-886`); docker missing returns 1 (`pc.sh:333-336`) | A1 is about the *message*, not the decision | NONE |
| GPU override selection | **CONFIRMED BROKEN at the boundary** | A4 | P1 |
| Dataset read-only protection | **VERIFIED** (J3 live PASS; `touch` on `images/` → `Read-only file system`) | above | NONE |
| Disk-full | **CONFIRMED, P1** — the launcher's floor is 25 GB beyond which it *fails* (`pc.sh:285-302`), but this host has **5 GB free while `./pc.sh` is the documented entrypoint**: `df -PB 1G .` → `5`; `./pc.sh --doctor` → `[FAIL] Dung lượng đĩa trống : 5 GB (thấp hơn mức tối thiểu 25 GB)` with **rc=0** (doctor mode returns 0 before the `has_errors` test at `pc.sh:508-512`, by design). Docker's own layer store is on the same 98%-full filesystem. No component handles ENOSPC (`grep ENOSPC` in `workbench/backend` → nothing); `_persist` swallows `OSError` (`jobs.py:454-462`) so a full disk silently loses job state | `pc.sh:285-301`; `df`; live doctor | P1 |
| uid/gid | **CONFIRMED, P2** — defaults are safe (`docker-compose.yml:22` `1000:1000`, `pc.sh:582-587` exports `id -u/-g`), but `workbench/docker/.env` carries no `WORKBENCH_UID`/`WORKBENCH_GID` and compose is invoked by `run.sh` with only `--env-file .env` (`run.sh:79`), so the `pc.sh` exports are **not** what compose resolves. Harmless today (both 1000), a real ownership defect the moment `pc.sh`'s uid differs from the file default | `.env` (no such keys); `docker inspect --format '{{.Config.User}}'` → `1000:1000` | P2 |
| Active job destroyed by a restart | **VERIFIED / no false PASS found** — `pc.sh` asks the active-job question before any restart in both branches (830-835 for the HTTP≠200 branch via the persisted `queue_jobs/*/queue_state.json`; 840-866 + 884-890 for the HTTP=200 branch); `jobs.py:286-298` settles queued/preparing as `cancelled` under the same lock `shutdown` sets `_stopping` under, and `jobs.py:536-541` re-reads both under the lock before `Popen`, so a legitimate queued job cannot be left unrun by a shutdown that happens *after* dispatch — it is refused *before*, which is the intended semantics. I found **no** dispatch predicate that lets an already-`cancelled` or post-shutdown job spawn, and no path that turns a legitimate queued job into a permanent non-run except the shutdown itself | tests re-run: `test_concurrency_ownership.py` 11 passed | NONE |

## A11 — Smaller findings (no false-PASS found in this slice)

- **P2 `release_verify.sh` `BASE` is hardcoded** `http://127.0.0.1:8000` (`:22`); `WORKBENCH_BACKEND_PORT` is never read. On a host with a custom port every live journey is a false FAIL unless `--base` is passed by hand — and passing `--base` triggers A3.
- **P2 `.dockerignore:16`** excludes `workbench/frontend/node_modules` from the build context; the `Dockerfile:19` `COPY … package-lock.json` is fine, but `Dockerfile:9` `npm ci` is cached on `package.json`+`package-lock.json`, so a stale context plus a bumped dependency without a lockfile change is not detected. Low; noted only because it is in scope.
- **P3 stale container/image**: the running `cir-workbench` (image built 2026-10-15 13:06) predates `.dockerignore` (2026-10-16 00:11), `serve_workbench.py` (05:18), `docker-compose.yml` (20:36) and `release_verify.sh`. **No live observation in this report may be read as evidence about the current source of those files.** J1 live was additionally misattributed by this audit until the `.venv` hypothesis was tested; all live API/J3/J11 results were cross-checked against the current source before being cited.
- **P3 `pc.sh:404-446`** offers to install `nvidia-container-toolkit` interactively (`read -r -p` + `sudo apt-get`) from inside a "preflight" whose `--doctor` mode claims read-only. `--doctor` cannot reach it (`is_doctor` guards), so this is honest, but the non-doctor path mutates the host before the operator has been shown the whole report.
- **P3 `run.sh:114`** returns 3 from `cmd_doctor` when GPU is disabled; `cmd_up` calls `cmd_doctor || true` (`run.sh:88`) so the exit code is discarded — intentional, but it means `run.sh up` never reports a PENDING GPU.
- **NOT CONFIRMED / FALSE_POSITIVE**: "the new `jobs.py` dispatch predicate stops a legitimate queued job from running" — no such path found (11/11 ownership tests pass; the `_stopping` read is inside the same `with self._condition` as the `Popen`, `jobs.py:531-541`).
- **NOT CONFIRMED**: "`release_verify.sh`'s new dimension set makes a satisfiable dimension permanently PENDING" — refuted, see A9.
- **NOT CONFIRMED**: leading `..`/`.` in a mount `Source` breaking the `realpath` compare in `pc.sh:653-698` — compose/Docker normalise bind sources before `docker inspect` reports them (observed: `../../data/FashionIQ` in `.env` → `/home/heheboiz/data/cir/data/FashionIQ` in `.Mounts`), so both sides are already absolute.
- **UNKNOWN (evidence not available)**: whether `nvidia-container-runtime` + `DeviceRequests` make `inspect_container_config` pass on a real sm_120 host — no 5070 Ti and no `nvidia` runtime exist here, so A4a/A4c are reasoned from static control flow plus the observed container state, not reproduced.
- **UNKNOWN**: durable key creation under `/artifacts/config/cir` in a properly rebuilt container (that path does not exist yet, because the running image predates `Dockerfile:56-71`); untested by execution.

## A12 — Files NOT inspected (honest scope)

Audited: `pc.sh` (all 1437 lines), `workbench/docker/{Dockerfile,docker-compose.yml,docker-compose.gpu.yml,run.sh,release_verify.sh,.env,.env.example}`, `.dockerignore`, `workbench/config/workbench.env.example`, `workbench/scripts/serve_workbench.py`, `workbench/backend/operator_config.py`, plus the read-only cross-reads cited above (`runtime.py`, `jobs.py`, `main.py`, `evaluate_models.py`, `verify_run.py`).

**Read but not line-by-line audited**: `workbench/docker/acceptance.sh` (312 lines, skimmed headers/groups A–I),
`workbench/docker/mutation_checks.sh` (228 lines, skimmed). **Not inspected at all**: every other file in `SOURCE_MANIFEST.md`
(89 of 100, owned by the other slices), `workbench/docs/**` beyond the citations above, `workbench/tests/test_release_blocker_gpu_records.py`
beyond its dimension assertions, `workbench/frontend/*` beyond the two-line `App.tsx`/`types.ts` diff, and
`src/**`.

## A13 — Execution trace

1. `git status --porcelain`, mtimes, md5 — established the dirty tree and the 100-file coverage target.
2. Read `pc.sh` 1-1437 in ranges; found A1 by reading the two `local mismatch_reasons` declarations, then reproduced it in `/tmp/arr_repro2.sh` and proved the bash semantics in `/tmp/dup_local.sh`.
3. Read `release_verify.sh` end-to-end; ran `--offline` (rc=0, `GPU_VALIDATION_PENDING`) and live (rc=0, J1 FAIL, J3/J11 PASS).
4. Stood up read-only docker probes: `inspect` (mounts/env/DeviceRequests/user/restart policy), `exec` (uid, key paths, `printenv`, `touch` RO tests). No container state mutated beyond one `mktemp -d`-backed disposable container created by J11 itself, which the script removes.
5. Reproduced the capability-key divergence across host/container and the adaptation mismatch (`/tmp/adapt.sh`).
6. Ran 3 targeted pytest files (93 tests, all pass) — the first surface where the shipped test expectations depend on A1.
7. Built `/tmp/rvmut2/repo` (copy of `workbench/docker` + symlinked `workbench/*`, `pyproject.toml`) and injected one `raise` into the verifier's report block; demonstrated A8 (exit 0, no report) and, by the failure order, that A2's J1 failure pre-empts the injected failure.
8. Re-read `workbench/backend/main.py` before writing and discovered it had been rewritten at 00:04:55; re-tested A6/A7 against both revisions.
9. Wrote this report.

## A14 — Can this fail on the real first run on an RTX 5070 Ti PC?

**Yes, in at least four independent ways, and for a fresh PC most of them fire on the first invocation.**

1. **Before any evaluation, `./pc.sh` stops at CONTAINER_RECONFIGURE (A4).** With the shipped `.env`, `WORKBENCH_RUNTIME_ADAPTATION` is commented out and `WORKBENCH_ENABLE_GPU` is commented out; `pc.sh` exports `sm_120` and `ENABLE_GPU=1`, inspects a container that has neither, refuses, calls `run.sh up` (which reads neither value from the environment), re-inspects, refuses again, and prints a remedy naming `HOST_FASHIONIQ_ROOT`. On a 5070 Ti the GPU half is *also* unsatisfiable until `nvidia-container-toolkit` is installed **and the operator opens a new session** (`pc.sh:405-446` re-tests `docker info` in the same shell that just installed the toolkit). Nothing in the reported reason tells the operator that the toolkit is the blocker: the first reason printed is the adaptation mismatch, the second is the GPU one, and the remedy names neither.
2. **The record/verification half cannot corroborate itself (A5).** Even after the stack is right, `release_verify.sh` computes the capability key, the interpreter (from `os.environ[python_env_var]`, which unset means "no interpreter") and the host identity (container hostname) in the *host* process, and it adopts `${REPO}/.venv/bin/python` — a venv a clean target PC does not have. `GPU_RUNTIME` cannot go PASS from the documented invocation. `./pc.sh --verify-only` does run the probe in-container, so the *record* can be produced there; reading it back from the host cannot work.
3. **`release_verify.sh` will not pass at all on a repo with a `.venv` (A2/A3), and will silently exit 0 without writing a report if its own scan raises (A8).** Both were reproduced on this host.
4. **Disk.** The five modalities above all assume the box has ≥25 GB. This one has 5 GB and `--doctor` says the right thing while exiting 0.

**Where the design is genuinely sound, and can be relied on:** dataset read-only protection (`images/`, `image_splits/` are separate RO binds and J3 proves it live), the never-restart-over-an-active-job decision (both API branches ask the persisted queue and refuse rather than tear down a run), the `docker exec` environment re-declaration for the verifier, readiness polling, and the `jobs.py` shutdown/dispatch handshake (queued/preparing jobs are settled `cancelled` under the same lock that `_stopping` is set under, and the guard is re-read under that lock immediately before `Popen`). **The failure mode of this slice is not fabricating success — it is refusing a healthy host and saying the wrong reason.**
