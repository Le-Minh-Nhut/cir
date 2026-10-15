# Final PC audit — slice A: Docker deployment + `./pc.sh` orchestration

Auditor: independent adversarial pass. HEAD `0b0f3a6` (+ untracked `pc.sh`, read in place; it changed
mid-audit — see the note below — final revision `572cca17…dad196d4`, 1083 lines).
READ-ONLY: no repo file was written except this report. No docker build, no long-lived server,
no download, no full pytest. Two throwaway containers were started from the **existing local
image** `cir-workbench:fix2` (built 2026-10-16 02:39, i.e. before the `pc.sh`/registry work) and
removed afterwards; nothing was written under the repo.

Every claim below is **STATIC_AUDIT** unless the line says **OBSERVED**, which means a command
was run and its output is quoted. `pc.sh` is untracked and was **edited by another agent while this
audit ran**: findings are anchored to the revision `sha256=572cca17…dad196d4`, 1083 lines,
mtime 2026-10-16 04:58:02 +0700, and re-verified after the change. Where the new revision fixed
something, the finding says so explicitly.

---

## A) Files inspected

`workbench/docker/{Dockerfile, docker-compose.yml, docker-compose.gpu.yml, .env.example, .env, run.sh}`,
`.dockerignore`, `workbench/scripts/serve_workbench.py`, the untracked root `pc.sh`, `workbench/run.sh`,
`workbench/backend/{runtime.py, operator_config.py, main.py, registry.py, jobs.py}`,
`workbench/scripts/{pipeline.py, evaluate_models.py, gpu_check.py, download_checkpoints.py}`,
`workbench/frontend/{package.json, vite.config.mts}`, `workbench/tests/test_pc_launcher.py` + its
`fixtures/pc_launcher/bin/*` shims, and (as hypotheses only, never trusted) the existing
`final_pc_audit/{D,F}.md`, `final_hardening/*` and `docs/PC_*.md`.

---

## B) Execution paths traced

1. `./pc.sh`: `run_preflight` (`:230-508`) → `configure_environment` (`:509-590`) →
   `start_docker_and_wait_readiness` (`:591-650`, `run.sh up` `:614`, poll `:618-640`) →
   `run_model_preparation` (`:652-668`, `run.sh prepare` `:656`) → `verify_gpu_in_container` (`:670-690`,
   `probe_gpu_capability` `:678`) → `execute_evaluation` (`:692-883`, `POST /api/jobs` `:799`) →
   `validate_and_summarize` (`:885-1025`, index rebuild `:892`) → `main` (`:1027-1083`).
2. Container: `serve_workbench.py:50` spawns `python -m uvicorn workbench.backend.main:app` (cwd `/repo`);
   `:52` spawns `npm run preview -- --host 0.0.0.0 --port 5173 --strictPort`; `:150` injects
   `WORKBENCH_BACKEND_URL=http://127.0.0.1:8000` for the frontend only, and `vite.config.mts:7` proxies
   `/api` there; `:170-176` exits with the first dead child's status — **OBSERVED**: the existing image
   started both halves and each bound its port.
3. Launch gate: `POST /api/jobs` (`main.py:420`) → `guarded_plan` → `runtime.require_launch_ready`
   (`runtime.py:1646-1660`) → `verify_environment(probe=True)` → `_cached_gpu_capability`
   (`runtime.py:1130-1181`); the tier must be in `LAUNCH_READY_TIERS` (`runtime.py:1417`).
4. Capability persistence: `verify_gpu_runtime` (`runtime.py:1819-1966`) → `record_gpu_capability`
   (`runtime.py:958-992`) → HMAC signature (`runtime.py:979-981`) requires `_gpu_cache_key()`
   (`runtime.py:1063-1107`).

---

## C) Confirmed defects

### A-1 — P0 — The GPU capability record cannot be signed inside the container, so no GPU evaluation can ever launch

**Verdict:** CONFIRMED (STATIC_AUDIT + OBSERVED in-container reproduction).
**Location:** `workbench/docker/Dockerfile:57-59,61-64`; `workbench/docker/docker-compose.yml:22,40`;
`workbench/backend/runtime.py:1080-1083,1088,1106-1107,979-981,1139`,`:1170`,`:1646-1660`;
`workbench/backend/main.py:420-422`.
**Root cause:** `_gpu_cache_key()` resolves the operator key from
`Path(getattr(resolve_config(), "CIR_REPO_ROOT")) / ".workbench_ledger_key"` (`runtime.py:1080-1083`)
and, when the file is absent, creates it under that parent (`:1086-1102`). In the container
`CIR_REPO_ROOT=/repo` (`Dockerfile:64`, `workbench/docker/.env.example:12`) and the image chowns only
`/repo/workbench/frontend/node_modules` and `/artifacts` (`Dockerfile:59`), leaving `/repo`
root-owned, while compose runs the process as uid 1000 (`docker-compose.yml:22`, `Dockerfile:57`).
The repository's own `.workbench_ledger_key` (present on the authoring host, `ls -l` → mode 600)
is **not** copied by the image: `Dockerfile:36-38` COPYs only `pyproject.toml README.md src/
workbench/`, and the build context is `/repo`. Therefore:

* no key file exists in the container,
* `mkdir /repo/.workbench_ledger_key`-parent fails with `OSError`,
* `except OSError: return b""` (`runtime.py:1106-1107`),
* `_gpu_capability_signature` → `None`, and `record_gpu_capability` writes the record **without** a
  `signature` field because of the `if signature:` guard (`runtime.py:979-981`),
* `_cached_gpu_capability` requires `_gpu_capability_signature_valid(payload)`
  (`runtime.py:1139`, `:1170`) — an unsigned record is discarded,
* `require_launch_ready` (`runtime.py:1646-1660`) then raises `GpuVerificationError`, and
  `POST /api/jobs` answers `409 gpu_verification_required` (`main.py:420-422`).

**Reproduction (OBSERVED, no repo mutation, container removed with `--rm`):**

```
$ docker run --rm --user 1000:1000 --entrypoint python -w /repo cir-workbench:fix2 -c \
  "import sys;sys.path.insert(0,'/repo');from workbench.backend import runtime;print(repr(runtime._gpu_cache_key()))"
b''
$ docker run --rm --user 1000:1000 --entrypoint python -w /repo cir-workbench:fix2 -c \
  "import sys;sys.path.insert(0,'/repo');from workbench.backend import runtime;print(runtime._gpu_capability_signature({'capability_verified': True}))"
None
# with a throwaway artifacts root, the record is written but unsigned, and is ignored on reload:
record signature: None
record file exists: True /tmp/art/gpu_capabilities/8c8a…json
reload honored: None
$ docker run --rm --user 1000:1000 --entrypoint sh cir-workbench:fix2 -c 'touch /repo/x'
touch: cannot touch '/repo/x': Permission denied
```
Host-side contrast (why this never showed up on the authoring laptop): the repo root **does** hold
`.workbench_ledger_key` (`-rw------- heheboiz`, gitignored at `.gitignore:248`), and the existing
records under `workbench/artifacts/gpu_capabilities/` do carry `"signature"`.
**Expected:** `pc.sh` step 5 records a signed capability and step 7 evaluates.
**Actual:** step 5 prints success (`probe_gpu_capability` only checks the *payload*: `pipeline.py:1400-1404`;
step 7 is refused by both the API (`main.py:420-422`) and the evaluator (`evaluate_models.py:1862-1869`).
`pc.sh:678-686` converts that into a hard failure with no relation to the real cause.
**Impact:** the headline workflow ("one command reproduces LIMN on the RTX 5070 Ti") cannot produce a
single evaluated run in the shipped container. Whole subsystem inert.
**Minimal fix (one place, all callers):** make the container able to resolve *the same* key the
ledger uses. Either
(a) set `WORKBENCH_LEDGER_KEY` for the service (compose `environment:` from a host `.env` value, or a
docker secret), or
(b) point capability signing at the same private path the ledger uses
(`evaluate_models.py:1129-1130`: `$XDG_CONFIG_HOME/cir/workbench_ledger_key`, i.e. `/artifacts/config/cir/…`
in the container) and create it there, so `runtime.py` and `evaluate_models.py` share ONE resolution
order. (b) is the smaller diff conceptually but touches two files; (a) is one line in `docker-compose.yml`
plus one `.env.example` line. Do **not** chown `/repo` to uid 1000: repo-root writability is exactly what
`evaluate_models.py:1118-1122` moved away from.

### A-2 — FIXED in the new `pc.sh` revision (was P1) — GPU-verification failure used to kill the script silently

**Verdict:** FALSE_POSITIVE **for revision `572cca17…`**; it was CONFIRMED for the previous revision
(captured earlier in this audit as `pc.sh:659`, `sha256=98deabcf…`, 1051 lines) where the command
substitution had no `||` guard.
**Location:** `pc.sh:678` now reads
`probe_out="$(docker exec cir-workbench python -c … )" || probe_rc=$?`, so the dead handler at
`pc.sh:680-686` is reachable and the diagnostic file is written.
**Root cause (of the pre-edit form):** the command substitution was a simple command with no `||`, so
under `set -euo pipefail` (`pc.sh:5`) a non-zero probe exit terminated the script and the following
`if (( probe_rc != 0 ))` block was unreachable.
Evidence that the construct is dangerous when unguarded (**OBSERVED**):
```
$ bash -c 'set -e; x="$(false)"; echo after'; echo rc=$?
rc=1              # "after" is never printed
```
**Retained as a regression requirement:** `pc.sh:678` must keep the `|| probe_rc=$?` clause; without it
the A-1 failure mode degrades from "actionable summary" to "silent exit 1".
**Minimal fix:** none (already applied); add the guard to §G-3 so a future edit cannot drop it.

### A-3 — P0 — `/api/models` cannot answer inside `pc.sh`'s 5 s poll budget once LIMN's checkpoints are on disk, so readiness never passes

**Verdict:** CONFIRMED (STATIC_AUDIT + OBSERVED timings on this host).
**Location:** `pc.sh:605` and `pc.sh:630` (`curl --max-time 5 … /api/models`); cost source
`workbench/backend/main.py:103-121` and `:246` → `registry.checkpoint_availability`
(`registry.py:771-803`) → `registry.sha256_file` (`registry.py:641`), plus
`runtime.runtime_blockers` (`runtime.py:2306-2328`, hash at `:2316-2318`) called from inside
`checkpoint_availability` via `runtime_blockers`.
**Root cause:** for each of the 16 registered models `models()` builds `checkpoint_availability` for
every variant (25 variants) **and then again** for every bundle member of every bundle; LIMN has a
3-member bundle over the same 3 files. Each `checkpoint_availability` hashes the on-disk checkpoint
once for the availability digest (`registry.py:781`) and a **second** time inside
`runtime_blockers` (`runtime.py:2322-2324`), and that second hash is not cached by anything. LIMN's
three checkpoints are 1.79 GB each (5.37 GB total; `du -sh workbench/artifacts/checkpoints` → 5.1 GB).
**Observed:**
```
$ .venv/bin/python -c "from workbench.backend import main; import time; t=time.time(); main.models(); print(time.time()-t)"
45.72s / 45.57s / 46.54s   # three consecutive real /api/models bodies, models=16
availability call1: 5.76s  call2: 5.73s      # the cache dict does not help a full 1.79 GB read
runtime_blockers call1: 4.56s  call2: 4.36s  # unbounded, re-hashes the same file
full 25-variant availability sweep: 25.03s / 24.79s / 25.75s (three consecutive passes)
```
The endpoint is a sync `def`, so it occupies a threadpool worker for the whole 45 s; the caches that
exist are process-local module dicts, so a **restarted** container pays the full cost again on its
first request.
**Expected:** `curl --max-time 5` gets HTTP 200 within the readiness window (`pc.sh:618-634`, ≤180 s).
**Actual:** every poll is aborted at 5 s with `http_code 000`; even the cheap pre-check
(`pc.sh:605`) cannot succeed. `pc.sh` therefore fails with
`DOCKER_READINESS … Backend HTTP 000` after its 180 s deadline although the stack is healthy, and
`--timeout` cannot help (`pc.sh:618-623` caps the readiness budget at 180 s).
**Impact:** deterministic failure of `./pc.sh` whenever pre-existing checkpoints are present —
i.e. on the second invocation (normal research loop) and on the documented transfer path that
advises copying `paths.checkpoints` (`workbench/docs/PC_TRANSFER_CHECKLIST.md`). Combined with §A-1
this means the shipped one-command path has no reachable success state except a from-scratch first
run that happens to have no checkpoints at readiness time.
**Minimal fix:** poll a cheap liveness endpoint (e.g. `GET /api/health`, or `GET /api/jobs`, which is
`JOBS.list()` only) instead of `/api/models`, and give the readiness phase its own budget
(`--ready-timeout`, default 600 s). Independently: make `runtime_blockers` use the same
size+mtime digest cache the availability view already uses, and build each bundle member's
`checkpoint_availability` **once** and reuse it in `models()` — that is two small edits and removes
the 45 s body entirely.

### A-4 — P1 — No documented way to set `WORKBENCH_ENABLE_GPU` actually reaches the stack, and `.env` is not read by compose

**Verdict:** CONFIRMED (STATIC_AUDIT + OBSERVED).
**Location:** `workbench/docker/run.sh:33-66` (`env_value`, `compose()`), `.env.example:41-43`;
`pc.sh:544-548` (was `:536-547` before the concurrent edit); `docker-compose.yml:28-40`.
**Root cause:** `run.sh` resolves ports by parsing `.env` *itself* (`env_value`, `:33-54`) — it never
exports what it reads — and `compose()` explicitly passes `--env-file "${HERE}/.env"` (`run.sh:66`).
An explicit `--env-file` **disables** Compose's implicit `.env` loading, and no service declares
`env_file` for `WORKBENCH_ENABLE_GPU` (`docker-compose.yml:26-27` uses `env_file: - .env` but the
resolved container env does not contain the key). **OBSERVED**:
```
$ docker compose -f workbench/docker/docker-compose.yml --env-file workbench/docker/.env config
  environment:
    WORKBENCH_BACKEND_PORT: "8000"
    … (no WORKBENCH_ENABLE_GPU)
```
So a `WORKBENCH_ENABLE_GPU=1` line in `.env` selects `docker-compose.gpu.yml` at launch time
(`run.sh:57-61`) but is invisible to the container, and the container's own GPU-gating/attribution
code sees nothing. `pc.sh` exports it in the shell and now also writes it into `.env` when it creates that file
(`pc.sh:558-564`), which the container still never sees, so pc.sh, `run.sh` and the container can
disagree about what "GPU enabled" means.
The `pilot` path additionally refuses the flag (`run.sh:7-12`, `cmd_pilot` uses *no* `--verify-gpu`),
so the documented claim in `docker-compose.yml:5-6` that `.env` is the switch is **FALSE** for the
combination "stack started once CPU-only, then enabled".
**Impact:** confusing half-state (GPU override active at launch, absent in-env), and the documented
one-liner `WORKBENCH_ENABLE_GPU=1 ./run.sh up` is the only form that works, which conflicts with
`.env.example` telling the operator to put the flag in `.env`.
**Minimal fix:** add `WORKBENCH_ENABLE_GPU: "${WORKBENCH_ENABLE_GPU:-0}"` (and, if desired, the
adaptation value) to `docker-compose.yml:28-40`, or have `run.sh` export the resolved value and drop
`--env-file` in favour of an implicit `.env` next to the compose file. One line either way.

### A-5 — P1 — Prebuilt image hardcodes uid 1000 while `pc.sh` injects the invoking uid, so the documented "clear PC" path fails for any operator with a different uid

**Verdict:** CONFIRMED (STATIC_AUDIT + OBSERVED).
**Location:** `Dockerfile:57,59` (`useradd --uid 1000`, `chown -R 1000:1000 …/node_modules /artifacts`),
`docker-compose.yml:22`, `pc.sh:579-584` (`export WORKBENCH_UID="$(id -u)"`).
**Observation:** the image's baked ownership:
```
$ docker run --rm --entrypoint sh cir-workbench:fix2 -c 'ls -ld /artifacts /artifacts/mamba'
drwxr-xr-x 3 appuser appuser /artifacts
drwxr-xr-x 2 appuser appuser /artifacts/mamba
```
With `WORKBENCH_UID` ≠ 1000 the appuser-owned `/artifacts/mamba` (the `MAMBA_ROOT_PREFIX`,
`Dockerfile:62`) and `XDG_CONFIG_HOME=/artifacts/config` (`docker-compose.yml:40`) are visible but not
writable; the first environment provision fails. The documented remedy (`Dockerfile:51-56`: "an
operator with a different uid rebuilds or overrides both values together") directly contradicts the
one-command promise, and the uid flag appears nowhere in `workbench/docs/PC_DOCKER_QUICKSTART.md`
or `PC_TRANSFER_CHECKLIST.md`.
**Impact:** P1 on any target whose first user is not uid 1000 (common on shared/lab machines); the
failure surfaces ~20 minutes into `prepare`, as a conda/micromamba error.
**Minimal fix:** `chown -R 0:0` nothing; instead make the runtime paths group/other-writable where the
image owns them (`chmod 0777 /artifacts /artifacts/mamba` is the lazy version), or drop `pc.sh`'s
uid injection and keep compose's `WORKBENCH_UID:-1000`, documenting "set WORKBENCH_UID only together
with a rebuild". Pick one; today the two mechanisms disagree.

### A-6 — P2 — `pc.sh` disk gate (>10 GB) contradicts the documented requirement (>50 GB) and no component handles disk-full

**Verdict:** CONFIRMED.
**Location:** `pc.sh:282-297` (was `:258-272`); `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md:30`
("Tối thiểu 50 GB");
`grep -rn "statvfs|disk_usage|ENOSPC" workbench/backend workbench/scripts` → only a vendored
`huggingface_hub` copy, **no first-party check**.
**OBSERVED** on this host: `df -P` col-4 `11` GB → `pc.sh` prints `[OK] … 11 GB` while checkpoints
alone are 5.1 GB (`du -sh workbench/artifacts/checkpoints`) and the LIMN bundle is 3 × 1.79 GB.
**Impact:** a 10–20 GB host passes preflight, then hits `ENOSPC` mid-download/mid-unpack; the
pipeline reports it as a generic stage failure with no disk-specific guidance.
**Minimal fix:** raise the threshold to the documented figure and reuse it for a warning line about
the checkpoint bundle size; add an explicit `ENOSPC` branch in the checkpoint/dataset stages.

### A-7 — P2 — `pc.sh` believes Docker has the NVIDIA runtime when any `nvidia` word appears in `docker info`

**Verdict:** CONFIRMED (OBSERVED false positive).
**Location:** `pc.sh:402` (was `:382`).
**Reproduction:**
```
$ PATH=/tmp/shimprobe:$PATH bash -c 'if docker info --format ... | grep -q "\"nvidia\"" || docker info | grep -iq nvidia; then echo PRESENT; fi'
pc.sh branch => reports RUNTIME PRESENT (false OK)
# shim's docker info prints: "Name: nvidia-pc", "Runtimes: runc"
```
(Docker's own `docker info` output is unpiped and unfiltered here, so any host name, storage driver
string or future plugin mentioning "nvidia" flips the check.)
**Impact:** on a host whose container hostname contains "nvidia" but which has no
`nvidia-container-runtime`, `pc.sh` skips the toolkit warning and proceeds to the failed
`docker compose` device reservation, with no actionable message. The real first-run symptom
(**OBSERVED**) is `Error response from daemon: could not select device driver "nvidia" with
capabilities: [[gpu]]`.
**Minimal fix:** drop the `|| docker info | grep -iq nvidia` fallback; the precise
`{{json .Runtimes}} | grep -q '"nvidia"'` form is already first in the `||` chain.

### A-8 — P2 — `run.sh up` swallows every doctor failure, including "the API never came up"

**Verdict:** CONFIRMED.
**Location:** `run.sh:86-91` (`cmd_up` ends with `cmd_doctor || true`).
**Root cause:** `cmd_doctor` returns 1 (API unreachable / probe failure) or 3 (GPU disabled) and every
return value is discarded, so `run.sh up` exits 0 and prints the two URLs regardless.
**Impact:** scripted/one-command flows (`pc.sh:614`) cannot distinguish "up and serving" from
"container started but the backend died". `pc.sh` re-checks readiness itself, so the impact is
contained to direct `run.sh up` users and any CI wrapper — P2.
**Minimal fix:** keep `|| true` for the GPU-disabled branch only, but propagate a non-zero doctor
into a non-zero `up` (e.g. capture the code and `return` it after printing, or add `--strict`).

### A-9 — P2 — vendored micromamba fetched from `latest` without a checksum

**Verdict:** CONFIRMED.
**Location:** `Dockerfile:30-34` (unchanged).
**Root cause:** `curl -fsSL "https://micro.mamba.pm/api/micromamba/linux-64/latest" | tar …` pins
nothing: no version, no digest, no TLS-pinned artifact. Two builds months apart can differ, and the
alias `/usr/local/bin/micromamba` (`:34`) hides the provenance.
**Impact:** image reproducibility; a compromised/changed endpoint silently alters the solver used to
provision every model environment.
**Minimal fix:** pin a versioned URL plus `sha256sum -c` (one extra `RUN` line), or install
micromamba from a pinned conda-forge package. Same for the base images if byte-reproducibility is
claimed anywhere.

### A-10 — P3 — `pc.sh` fallback declares uid = caller but GID = 1000

**Verdict:** CONFIRMED.
**Location:** `pc.sh:582-584` (was `:560-566`).
**Root cause:** when `id` is unavailable the code sets `WORKBENCH_UID="${UID}"` and
`WORKBENCH_GID="1000"` — an arbitrary pair that need not correspond to any real group; combined with
A-5 that is a write-permission lottery.
**Minimal fix:** fail loudly ("cannot determine uid/gid") rather than guess.

### A-11 — P3 — `probe_gpu_capability` prints `None` as the recorded capability key

**Verdict:** CONFIRMED (STATIC_AUDIT).
**Location:** `workbench/scripts/pipeline.py:1400-1404` reads `capability.get("capability_key")` from
the return of `verify_gpu_runtime`, but that function returns the pre-record `payload`
(`runtime.py:1966`); `capability_key` is added by `record_gpu_capability` (`runtime.py:966-969`)
to a *different* dict. Same pattern in `evaluate_models.py:1798-1803`.
**Impact:** cosmetic but operator-facing: `pc.sh` prints
`[OK] GPU capability verified and recorded: None` on success, which reads like a bug and weakens the
evidence line. **Fix:** return the recorded payload (or merge the key into the returned dict).

---

## D) Probable defects needing reproduction (no GPU host available here)

| ID | Sev | Finding | Evidence | What is needed |
|---|---|---|---|---|
| A-P1 | P1 | `pc.sh` `WORKBENCH_ENABLE_GPU=1` + `--no-gpu` interplay: `--no-gpu` sets `ENABLE_GPU=0`, but preflight already skipped the toolkit check (`pc.sh:362`, `:399`), so a CPU-only run is possible while `WORKBENCH_RUNTIME_ADAPTATION=sm_120` is still exported (`pc.sh:552-557`) | `pc.sh:362,399,544-557` | one run of `./pc.sh --no-gpu` against a stack that is deliberately CPU-only; observe whether the adaptation value reaches provisioning |
| A-P2 | P2 | GPU override + `restart: unless-stopped` (`docker-compose.yml:58`): a container that fails to start because the toolkit was removed stays in a restart loop reporting `restarting`, and `pc.sh`'s readiness poll only ever sees HTTP failures | `docker-compose.yml:58`, `pc.sh:618-634` | induce `could not select device driver "nvidia"` with `restart: unless-stopped` and watch `docker compose ps` / `run.sh status` |
| A-P3 | P2 | `.dockerignore:31-33` ignores `workbench/artifacts` but the build still runs on a 173 GB root with 11 GB free; buildkit's context transfer of `workbench/frontend/` (106 MB incl. `node_modules`) and `workbench/tests` (6.2 MB) is not free | `.dockerignore`, `du -sh` | time `docker compose build` on a full disk and check for `no space left on device` in buildkit output |
| A-P4 | P2 | `docker-compose.yml:37-40` rationale ("non-root uid with no writable home directory") is stale: `/home/appuser` exists and is uid-1000 owned (`Dockerfile:57,61`), so the `XDG_CONFIG_HOME` redirection is defensive | `Dockerfile:57,61` | rerun `_gpu_cache_key()` with `XDG_CONFIG_HOME` unset |
| A-P5 | P3 | `serve_workbench.py:170-176` exits with the first dead child, so a `vite preview` crash tears down the backend | `serve_workbench.py:52,170-176` | occupy 5173 on the host and observe |

No fabricated runtime results: nothing above is GPU-verified (no toolkit on this host —
`ls /usr/bin/nvidia-ctk` → absent, `docker info .Runtimes` → `runc` only).

---

## E) Test coverage gaps

1. **No test builds or starts the image**: `grep -rn "docker-compose\|Dockerfile" workbench/tests/*.py`
   returns nothing, so every `Dockerfile`/compose claim in this slice is unverified by the suite.
   A-1, A-4 and A-5 would all be caught by one test that runs the built image once as uid 1000 and
   asserts `_gpu_cache_key()` is non-empty.
2. `test_pc_launcher.py` (471 L, untracked) drives `pc.sh` through PATH shims
   (`tests/fixtures/pc_launcher/bin/{docker,curl,df,git,nvidia-smi}`); its docker shim emits
   `Runtimes: runc nvidia` as a text line (`fixtures/…/bin/docker:69-79`) — it *matches the
   implementation* instead of the daemon's JSON, which is why A-7 is invisible to it.
3. `test_final_hardening_regressions.py:150-155` asserts one property of `run.sh` (no f-string
   `${…!r}`), not its behavior; nothing covers the doctor probe or `pc.sh`'s `set -e` handling.
4. `test_release_blocker_cuda_readiness.py::test_the_fallback_key_is_created_so_a_capability_can_be_signed`
   asserts the fallback key is created **with a writable `CIR_REPO_ROOT`** (`monkeypatch.setattr(…,
   CIR_REPO_ROOT=tmp_path)`) — exactly the environment the container does not have. The test encodes
   the assumption that breaks A-1.
5. No test covers disk-full, a missing `env_file`, or `WORKBENCH_UID≠1000`.

## F) Recommended minimal fixes (ordered)

1. **A-1 (one compose line + one doc line).** Add `WORKBENCH_LEDGER_KEY: "${WORKBENCH_LEDGER_KEY:-}"`
   to `docker-compose.yml` `environment:`, generate it in `pc.sh` when absent (`umask 077; head -c 32
   /dev/urandom | od -An -tx1 | tr -d ' \n'` into `workbench/docker/.env`), and document it in
   `.env.example` beside `WORKBENCH_UID`. Alternative (two files): teach `runtime._gpu_cache_key` the
   same `$XDG_CONFIG_HOME/cir/workbench_ledger_key` path `evaluate_models.ledger_hmac_key` already uses.
2. **A-3 (small).** Give readiness its own budget (fixed 600 s or `--ready-timeout`) and poll a cheap
   liveness endpoint (`/api/jobs`) instead of `/api/models`; separately, make `runtime_blockers` reuse
   the availability digest and build each bundle member once.
3. **A-4 (one line).** Pass `WORKBENCH_ENABLE_GPU` through `docker-compose.yml`'s `environment:` block.
4. **A-2 (already applied).** Keep the `|| probe_rc=$?` guard at `pc.sh:678`; protect it with §G-3.
5. **A-5 (one line).** `chmod 0777 /artifacts /artifacts/mamba` in the image, or stop injecting
   `WORKBENCH_UID` from `pc.sh`; do both only if rebuild-per-uid is genuinely intended.
6. **A-7 (delete a clause).** Remove `|| docker info 2>/dev/null | grep -iq nvidia`.
7. **A-6 (two lines).** Align the threshold with the documentation and add an `ENOSPC` branch.
8. **A-8..A-11.** Propagate doctor's exit code from `up`; pin micromamba with a checksum; fail rather
   than guess uid/gid; return the recorded payload from `verify_gpu_runtime`.

No rewrite is warranted: the compose contract, read-only dataset submounts, fail-closed GPU selection
and the capability-probe evidence standard are all sound. The blocker is a key-resolution
inconsistency plus a readiness probe aimed at the wrong endpoint.

---

## G) Exact regression tests required

1. `test_container_can_sign_a_gpu_capability(tmp_path)`: run the built image as
   `--user 1000:1000` with the compose env, call `runtime._gpu_cache_key()`, assert non-empty, then
   `record_gpu_capability` + `_cached_gpu_capability` round-trip returns the record. Fails today
   (§A-1). Mark `@pytest.mark.docker`.
2. `test_compose_propagates_gpu_switch()`: `docker compose config` must show
   `WORKBENCH_ENABLE_GPU` in `services.workbench.environment`. Fails today (§A-4).
3. `test_pc_sh_reports_probe_failure_with_summary()`: shimmed `docker exec` returning 1 must produce
   the LỖI THỰC THI block plus a `gpu_diagnostic_*.txt`, exit 1. **Passes** at `pc.sh:678`; it exists to
   stop the `|| probe_rc=$?` guard from being removed again (§A-2).
4. `test_pc_sh_readiness_budget_is_independent_of_job_timeout()`: assert the readiness deadline for
   `--timeout 30` is not 30 s; assert the phase uses its own constant. Fails today (§A-3).
5. `test_doctor_detects_missing_nvidia_runtime_from_json()`: docker shim emitting
   `{{json .Runtimes}}` = `{"runc":…}` and a hostname containing `nvidia` must print the PENDING
   branch. Fails today (§A-7).
6. `test_run_sh_up_propagates_failed_doctor()`: fake doctor returning 1 → `run.sh up` exits non-zero.
7. `test_dockerfile_pins_micromamba():` parse `Dockerfile` and assert no `latest` URL without an
   adjacent checksum (§A-9).

---

## GPU probe semantics — what `run.sh cmd_doctor` proves and what it does not

`run.sh:118-163` executes, **inside the orchestrator container**, with the image's own Python:

* `nvidia-smi --query-gpu=name,compute_cap` (proves the toolkit + driver inject a device node and a
  working `nvidia-smi`),
* `import torch; torch.cuda.is_available()` and one real kernel: `x = torch.randn(8,8,"cuda"); (x@x).sum().isfinite()` (`run.sh:130-137`),
* `torch.cuda.get_device_capability(0)` / `get_device_name(0)` as printed strings.

The interpreter is the image's: **OBSERVED**
`docker run --rm --entrypoint python cir-workbench:fix2 -c "import torch;print(torch.__version__, torch.version.cuda)"`
→ `2.14.0+cu130 13.0`, arch list `['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`. It is *not*
the isolated model environment (`registry models.yaml` `python_env_var: WORKBENCH_LIMN_PYTHON`,
env-prefix `…/environments/<id>/bin/python`, `environment_planner`/`evaluate_models` env allowlists),
and `python` on PATH cannot reach a conda prefix.

**Proves:** the host has a working NVIDIA container toolkit configured for Docker; the *orchestrator*
image's torch can allocate and execute a CUDA kernel on this device; the device advertises the
expected compute capability.
**Does NOT prove:**
1. that the model environment's own torch can see CUDA (different interpreter, different wheel; the
   historical LIMN pin is torch 1.12.1 and the adapted profile is cu128 `torch>=2.7,<3`);
2. that the model environment's torch **ships sm_120 kernels** — the string check
   (`compute_capability`) is not an arch check; the real compiled-core/arch verification lives in
   `runtime.verify_gpu_runtime` (`runtime.py:1891-1934`: `torch_has_compiled_core`,
   `core_loadable` via a ctypes `PyInit__C` load in an isolated child, `torch_file_digest`, and
   `adaptation_mismatch` when `WORKBENCH_RUNTIME_ADAPTATION=sm_120` is set but the arch list or the
   probed device disagrees);
3. that the official evaluator can load LIMN's weights, run its forward pass, or fit in 16 GB;
4. anything about determinism or metric parity.
So `run.sh doctor` is a **passthrough** check only. The author's own comment in `gpu_check.py:3-6`
("runs the same evidence standard the workbench uses for models") overstates it: `gpu_check.py` also
probes the orchestrator interpreter. The model-scoped proof is `probe_gpu_capability`
(`pipeline.py:1376-1407`) — which is the step blocked by §A-1.

---

## Checklist verdicts (assignment items)

| # | Item | Verdict | Sev | Evidence / action |
|---|---|---|---|---|
| 1 | Docker build correctness | VARIANT | P2 | stages/pip pins coherent (`Dockerfile:7-82`), image starts and serves (**OBSERVED**); `latest` micromamba (A-9) | pin micromamba |
| 2 | GPU passthrough / toolkit | VARIANT | P1 | `docker-compose.gpu.yml:8-14` correct (Compose v2 honours `deploy.resources`; **OBSERVED** daemon error proves it); detection false-positive `pc.sh:402` (A-7) | fix the grep |
| 3 | compose override precedence | VERIFIED | NONE | `run.sh:57-61`; merged `docker compose config` shows the device reservation (OBSERVED) | none |
| 4 | host-vs-container paths | VERIFIED | NONE | internal paths consistent (`Dockerfile:61-76`, `.env.example:12-22`); `HOST_*` resolve from `workbench/docker` (**OBSERVED** from `/tmp`) | none |
| 5 | bind mounts | VERIFIED | NONE | 7 mounts; `:ro` submounts kernel-enforced (**OBSERVED** isolated probe) | none |
| 6 | file ownership / UID / GID | VARIANT | P1 | A-5, A-10; on disk `checkpoints/limn` + `environments/cir-*` are root-owned, `results/` uid 1000 | one uid mechanism |
| 7 | disk permissions | VERIFIED | NONE | uid 1000 writes `/artifacts`, cannot write `/repo` (**OBSERVED**); intentional | none |
| 8 | cache directories | VARIANT | P2 | `MAMBA_ROOT_PREFIX=/artifacts/mamba` (`Dockerfile:62`) and `XDG_CONFIG_HOME=/artifacts/config` persist; `HOME=/home/appuser` is image-local, so `$HOME/.cache` (`registry.py:347-356`) dies on rebuild | point `TORCH_HOME`/`HF_HOME` at `/artifacts` |
| 9 | startup / restart | VERIFIED | NONE | `restart: unless-stopped` (compose:58); `serve_workbench` exits on child death (OBSERVED); phantom `running` jobs recovered (`main.py:38-61`) | none |
| 10 | disk-full handling | UNKNOWN | P2 | no first-party `statvfs`/`ENOSPC` anywhere; gate is 10 GB vs 50 GB documented (A-6) | add check + branch |
| 11 | dataset read-only protection | VERIFIED | NONE | compose:49-50, kernel-enforced (**OBSERVED**) | none |
| 12 | persistent storage | VERIFIED | NONE | index/runs/receipts/ledger under `/artifacts`; checkpoints and results separate mounts | none |
| 13 | docker security | VARIANT | P3 | non-root uid, pinned bases, no `privileged`/socket (**OBSERVED** `id` → 1000); no `cap_drop`, repo tree baked root-owned | optional `cap_drop: [ALL]` |
| 14 | image reproducibility | VARIANT | P2 | base images + pip pins exact (`Dockerfile:7,18,46-48`); `latest` micromamba (A-9), `curl \| tar` | pin micromamba |
| 15 | fresh install | BLOCKED | P0 | A-1 makes a clean Ubuntu+RTX5070Ti host fail at the evaluation step; A-3 stops `pc.sh` earlier once checkpoints exist; A-5 for uid≠1000 | fix A-1 **and** A-3 |
| 16 | Ubuntu compat **without** version hardcoding | VERIFIED | NONE | no `ubuntu`/`jammy`/`noble`/`22.04`/`24.04` literal in scope (only the Debian tag `python:3.13-slim-bookworm`); distro read via `/etc/os-release` (`pc.sh:244-247`) | keep prose in docs |
| 17 | `pc.sh` reusing `run.sh` | VERIFIED (caveats) | P1 | `pc.sh:19`; `run.sh up` (`:614`), `run.sh prepare` (`:656`); no evaluation-engine copy — only `/api` polling glue (`:697-885`) and `env_value`/`resolve_path` (`pc.sh:65,133` vs `run.sh:33-54`) | commit `pc.sh` 100755 |
| 18 | GPU probe semantics | VERIFIED | NONE | §above: passthrough proof only | none |

---

## Execution trace

1. `git log/status`, `ls` root → HEAD `0b0f3a6`; untracked `pc.sh` and
   `workbench/docs/PC_ONE_COMMAND_GUIDE.vi.md`; modified `workbench/backend/registry.py`
   (uncommitted `/api/models` availability-digest cache) and `test_final_hardening_regressions.py`.
2. Read every A-slice file in full, plus `workbench/run.sh`, `package.json`, `vite.config.mts`.
3. `bash -n run.sh` / `bash -n pc.sh` → syntax OK.
4. Traced the launch gate through `main.py`, `runtime.py`, `pipeline.py`, `evaluate_models.py`,
   `jobs.py`, `registry.py`.
5. Rendered the resolved stack (`docker compose -f … -f docker-compose.gpu.yml --env-file .env config`)
   from `workbench/docker` **and** from `/tmp`: identical relative-`HOST_*` resolution, absolute
   `source:` paths.
6. Isolated `/tmp` probes: read-only submount semantics (parent writable, child `Read-only file
   system`); missing toolkit (`could not select device driver "nvidia"`); `compose run --quiet` does
   **not** suppress container stdout (run.sh's probe parsing is sound).
7. In-container probes with the existing image `cir-workbench:fix2` (`--rm`): torch build/arch list,
   `id`/`ls -ld`, uid-1000 write permissions, `_gpu_cache_key()` → `b''`,
   `_gpu_capability_signature` → `None`, unsigned record written then ignored on reload, contained 80 s
   start of both halves.
8. Cost measurements: `main.models()` (real endpoint body) 45.72/45.57/46.54 s over three calls;
   `checkpoint_availability` 5.76/5.73 s; `runtime_blockers` 4.56/4.36 s (unbounded, same file
   re-hashed); 25-variant sweep 25.03/24.79/25.75 s.
9. `df`/`du` for the disk gate; `docker info` grep probes; shimmed-`docker` demo of the `nvidia`
   false positive.
10. Re-anchoring pass after `pc.sh` changed under the audit (`98deabcf…` → `572cca17…`): every `pc.sh`
    citation above re-read from the current file.
11. Nothing was written outside this report and `/tmp`; both throwaway containers removed.

## Final verdict

Slice verdict: **BLOCKED** (two independent P0s) — the Docker slice is structurally sound: the compose
contract, read-only dataset protection, host/container path separation, persistent storage and the
fail-closed GPU selection are all correct and were reproduced. But (i) the capability-signing key
cannot be resolved inside the shipped container, so a GPU capability can never be recorded as
authenticated and no GPU evaluation can launch (§A-1), and (ii) `pc.sh`'s readiness probe asks for a
45 s endpoint with a 5 s timeout, so the one-command path cannot get past readiness once checkpoints
are on disk (§A-3). Items §A-4..§A-11 are P1–P3 and independently fixable; §A-2 was fixed by the
concurrent `pc.sh` edit and is kept only as a regression requirement.

**Can this implementation fail during the real first run on an RTX 5070 Ti PC? YES** — on a clean
Ubuntu host + RTX 5070 Ti either the readiness phase times out (45 s `/api/models` body vs the 5 s
`curl --max-time` at `pc.sh:605,630`, P0 §A-3), or, with no checkpoints present so readiness succeeds,
the run reaches GPU verification, records an **unsigned** capability
(`runtime.py:979-981` with `_gpu_cache_key() == b''`, P0 §A-1), and is refused by the launch gate
(`runtime.py:1646-1660` → API `409 gpu_verification_required` at `main.py:420-422`).
