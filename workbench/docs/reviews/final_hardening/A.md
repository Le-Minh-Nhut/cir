# Slice A — Docker / Deployment audit

Scope: `workbench/docker/` (run.sh, Dockerfile, docker-compose.yml, docker-compose.gpu.yml,
.env.example, release_verify.sh, mutation_checks.sh, acceptance.sh), `workbench/scripts/serve_workbench.py`,
`workbench/backend/operator_config.py`.
Reviewer: read-only audit of the WORKTREE (see *Baseline*). No file under the repo was modified
except this report. GPU inference was not run.

## Baseline / method

* HEAD = `e9cafaf`; the worktree ALSO carries the main agent's in-flight fixes
  (`git status --porcelain` → `M workbench/docker/run.sh`, `M workbench/docker/release_verify.sh`,
  `M workbench/backend/{environment_planner,runtime}.py`, `M workbench/scripts/evaluate_models.py`,
  `M workbench/registry/models.yaml`; `?? workbench/docs/reviews/final_hardening/`).
  Line numbers below are the **worktree** file, and every finding that the main agent has already
  fixed is called out explicitly; where a claim needs the pinned revision I use `git show HEAD:<file>`.
* Live docker was used read-only: `docker version` = 29.9.0, `docker compose version` v5.6.0,
  `docker compose config`, `docker inspect`, `docker exec`, and one `docker restart` of the RUNNING
  container (see A-05; restart is the J11 subject and is non-destructive: no data removed).
* Host GPU = `NVIDIA GeForce RTX 3050 Laptop GPU, 8.6`; `docker info .Runtimes` = only
  `[io.containerd.runc.v2, runc]` → no nvidia runtime. So GPU work here is STATIC_AUDIT + daemon
  error observation, never a claimed runtime pass.
* Everything below is STATIC_AUDIT unless a command+output is quoted.

## A-01 — `${2!r}` is not bash; malformed .env port aborts with a substitution error
* Severity: P1
* file:function: `workbench/docker/run.sh` → `validate_env` (run.sh:51-57, the message on line 53) and
  the ENABLE_GPU guard (run.sh:72).
* Root cause: Python f-string syntax (`{value!r}`) written into bash. Under `set -e` the failed
  substitution aborts the script with the interpreter's own diagnostic before the intended message
  is printed.
* Repro (real script, scratch copy so the repo is untouched):
  `cd /tmp/rt2 && printf 'WORKBENCH_BACKEND_PORT=80a0\n' > .env && bash run.sh doctor`
* Observed:
  `run.sh: line 53: invalid ${1}=${2!r}: must be an integer port: bad substitution` and the process
  exits **1**. Same abort for an inline comment, which docker compose accepts:
  `WORKBENCH_BACKEND_PORT=8000   # host port` → same bad-substitution abort, while
  `docker compose --env-file <same file> config` publishes `"8000"`.
* Expected: `invalid WORKBENCH_BACKEND_PORT=80a0: must be an integer port` and `exit 2` (the documented
  behaviour in the same file's contract comment, run.sh:31-37).
* Impact: the "a mistyped port must not route requests to a stack the operator did not configure"
  guarantee degrades into an opaque bash error; an operator following `.env.example` (which contains
  commented values and prose) can be blocked by a trailing comment.
* Minimal fix: `say "invalid ${1}=$2: must be an integer port"` (drop `!r`), same on line 72.
* Regression test: `bash run.sh doctor` with `WORKBENCH_BACKEND_PORT=80a0` must exit 2 and the message
  must name `WORKBENCH_BACKEND_PORT`.
* Classification: CONFIRMED

## A-02 — the operator's real `workbench/docker/.env` is baked into the image
* Severity: P1 (credential/host-config leak + stale-config divergence)
* file:function: `.dockerignore:23` (`*.env`) vs `Dockerfile:38` (`COPY workbench/ ./workbench/`).
* Root cause: docker's rule "a pattern with no slash matches basenames *without recursing* when
  negations exist" — and line 24/25 do add negations. The only negation is the `.env.example`, so
  `workbench/docker/.env` (no slash anywhere in the pattern) is never excluded. The author's intent is
  in the neighbouring comment (`.dockerignore:21` header, whose block continues at `.dockerignore:22`).
* Repro (image as built):
  `docker run --rm --entrypoint sh docker-workbench -c 'md5sum /repo/workbench/docker/.env; stat -c %s /repo/workbench/docker/.env'`
* Observed: `21e9c24b056b4e8f70ec97b18d8e911f  /repo/workbench/docker/.env`, size 1775 — byte-identical to
  the host `workbench/docker/.env` (`md5sum` of the host file is the same). Independent minimal repro:
  a scratch build with the repo `.dockerignore` + `COPY . /w/` reports
  `#7 /w/workbench/docker/.env` from `find /w -name '*.env*'`.
* Expected: the container image carries `.env.example` only; every host-only path/credential lives at
  run time.
* Impact: any push/publish of `docker-workbench:latest` ships the operator's environment file; and
  because `run.sh`/compose pass `--env-file` from the host, a stale in-image copy silently disagrees
  with the file a later rebuild mounts. It also contradicts the file's own header.
* Minimal fix: add `workbench/docker/.env` (or `**/.env`) to `.dockerignore`. Verified sufficient in a
  scratch build: with `**/.env` the same `find` no longer lists `workbench/docker/.env`.
* Regression test: after `docker build`, `docker run --rm --entrypoint sh <img> -c 'test ! -e /repo/workbench/docker/.env'`.
* Classification: CONFIRMED

## A-03 — container runs as root against host bind mounts (root-owned artifacts on the host)
* Severity: P2
* file:function: `workbench/docker/Dockerfile` (no `USER` instruction; grep count 0) +
  `docker-compose.yml:31-47` (all bind mounts).
* Root cause: the image never drops privileges; the operator's own repo tree is mounted in, and every
  write from inside the container lands on the host as uid 0.
* Repro:
  `docker exec cir-workbench id` and a host ownership histogram:
  `python3 -c "os.walk('workbench/artifacts') … lstat().st_uid"`
* Observed: container `uid=0(root) gid=0(root)`; `workbench/artifacts` histogram `{0: 25589, 1000: 1623}`
  (e.g. `workbench/artifacts/environment_assignment.json`,
  `workbench/artifacts/checkpoints/limn/0_dress_best_model.pt` owned by root);
  `data/FashionIQ/resized_image` histogram `{0: 15540}`; the repo itself is owned by uid 1000
  (`id -u` = 1000, `id -un` = heheboiz).
* Expected: either an explicit, documented "the workbench takes ownership of the mounted trees", or a
  non-root container mapped to the operator uid.
* Impact: a host-side `rm`/`git clean`/editor write or a later non-root run hits EPERM on files the
  user cannot delete; `docker compose down` cannot repair it; deleting a checkpoint forces a
  multi-GB re-download from an unwritable directory.
* Minimal fix: document it in `PC_DOCKER_QUICKSTART.md`, or add
  `user: "${WORKBENCH_UID:-0}:${WORKBENCH_GID:-0}"` to the service and set it from `.env`.
* Regression test: `docker exec <container> test "$(id -u)" = "$(stat -c %u <artifacts mount>)"`.
* Classification: CONFIRMED

## A-04 — `/api/models` cannot be a probe, yet `doctor` treats it as one
* Severity: P2
* file:function: `workbench/docker/run.sh` → `cmd_doctor` endpoint loop (run.sh:102-105);
  `workbench/backend/main.py:70` `models()` → `workbench/backend/registry.py:703`
  `checkpoint_availability` → `registry.py:713` `sha256_file(path, …)`.
* Root cause: `/api/models` hashes every present checkpoint file (multi-GB) for every model on every
  request; doctor probes it with curl's default 8 s timeout (`curl -s -o /dev/null -w '%{http_code}'`
  with no `--max-time`).
* Repro:
  `cd workbench/docker && bash run.sh doctor` and a timed request:
  `curl -s -o /dev/null -w '%{http_code} %{time_total}' http://127.0.0.1:8000/api/models`
* Observed: doctor printed `  /api/models: 000` while `runs|jobs|benchmark-runs|cohorts` printed `200`
  on the same healthy stack; the timed call returned `200` after **71.06 s**; `len(models) == 16`.
  The verifier shares the assumption (release_verify.sh:46-50 uses `/api/jobs` for liveness, which is
  the correct cheap probe).
* Expected: a health check reports the stack's state, not a timeout artefact of an expensive endpoint.
* Impact: `./run.sh up` ends with a doctor that reports the backend as unreachable (`000`) on a good
  install — the operator's first signal is a false negative; `pilot` additionally runs this call **twice**
  (run.sh:235 and run.sh:240) before it can queue anything.
* Minimal fix: probe liveness on `/api/jobs` (and keep `/api/models` as a separate, long-timeout
  informational check), or cache the availability result.
* Regression test: doctor against a populated artifacts tree must print `200` for `/api/models`
  within the probe budget.
* Classification: CONFIRMED

## A-05 — the job queue is process memory only; a restart silently erases jobs
* Severity: P2 (P1 boundary: a killed running job leaves orphan artifacts and no record)
* file:function: `workbench/backend/jobs.py:89-99` (`self._jobs: list[Job] = []`),
  `workbench/backend/main.py:34` (`JOBS = SingleGpuQueue()`), `main.py:351`
  (`queue_jobs/<uuid4().hex>` — write-only, nothing reads it back: `grep -rn queue_jobs` finds only
  producers and tests).
* Root cause: no persistence and no startup reconciliation for queue state; `restart: unless-stopped`
  (docker-compose.yml:48) plus J11's persistence claim cover *files*, not the queue.
* Repro: `docker restart cir-workbench` (10.4 s), then `curl -s http://127.0.0.1:8000/api/jobs`.
* Observed: service back on 8000/5173 with `200`; `/api/jobs` = `[]`;
  `workbench/artifacts/queue_jobs` does not exist. A job that was `queued`/`running` at restart time is
  simply gone — not `failed`, not `cancelled`, not re-queued.
* Expected: an explicit contract. Either the queue is durable (reconcile `queue_jobs/<job_id>/` at
  startup and mark interrupted jobs terminal), or the docs state that `run.sh up`/restart drops queued
  work.
* Impact: an interruption between queued and completed produces a run directory with partial evaluator
  output and no queue record; `/api/jobs` (and therefore `run.sh status`, J8, and the UI) report a
  clean queue. Recovery has to be inferred from the filesystem.
* Minimal fix: write the `Job` dataclass into `queue_jobs/<job_id>/job.json` at enqueue/finalize and
  load+reconcile at startup (marking non-terminal jobs `failed` with a named reason).
* Regression test: enqueue → restart → assert the job is present and terminal (or re-queued), never absent.
* Classification: CONFIRMED

## A-06 — launcher docs promise `--verify-gpu` for pilot; the pilot path never passes it
* Severity: P3 (documentation/contract mismatch, no functional break today)
* file:function: `run.sh:9` (header), `run.sh:243` (`cmd_pilot` → `cmd_prepare`), `run.sh:247`
  (POST `/api/jobs`), `workbench/backend/main.py:342-348` (queue argv), `scripts/pipeline.py:1806,1841`
  (only `args.verify_gpu` adds the flags), `pipeline.py:1882` (`setup_stages`).
* Root cause: `setup` does include the `gpu-verification` stage (pipeline.py:1790-1797, selected by
  `setup_stages`), but the `--verify-gpu` argument that relaxes preflight/attaches the flag to the
  evaluation (`pipeline.py:1806`, `1841`) is never supplied by run.sh, and the API-built argv never
  contains `--verify-gpu` either.
* Repro: `grep -n "verify-gpu" workbench/docker/run.sh` → only the comment on line 9.
* Observed vs expected: comment claims "the same guarded pipeline, with `--verify-gpu`"; actual argv is
  `pipeline.py setup … --allow-gpu-eval --resume` (run.sh:216-221) and
  `evaluate_models.py --model … --top-k … --run-id …` (main.py:342-348). PC_DOCKER_QUICKSTART §5 makes
  the same claim.
* Impact: an operator reading the header believes GPU re-verification is forced per pilot run; the
  actual gate is `runtime.require_launch_ready` at queue admission (main.py:333-339), which is
  fail-closed — so the outcome is safe, the *wording* is wrong.
* Minimal fix: reword run.sh:9 / quickstart §5, or append `--verify-gpu` where intended.
* Regression test: assert run.sh's prepare/pilot argv contains `--verify-gpu` iff the header claims it.
* Classification: CONFIRMED

## A-07 — release_verify J5/J7/J8 remain weak (P1-E facets, Docker side)
* Severity: P1 (evidence-strength, not runtime)
* file:function: `workbench/docker/release_verify.sh` → J5 line 197, J7 lines 256-274, J8 lines 279-293.
* Root cause: J5 asserts on `inspect.getsource(runtime)` substrings (`"probe_source" in src and
  "GPU_PROBE_OVERRIDE_ENV" in src`) — a string-source check that survives the behaviour being deleted
  as long as the identifier appears anywhere in the module. J7 exercises only `evaluate_gate`'s refusal
  of `data_kind="mock"` and `evaluation_completed=False`; it never builds a partial (single-category)
  aggregate through the extraction path the comment on its own line 255 describes. J8 asserts only
  `{statuses} - allowed == ∅`, which an empty list satisfies.
* Repro: `curl -s http://127.0.0.1:8000/api/jobs` → `[]`, i.e. the live stack J8 saw would have been
  PASSed with a vacuous assertion (the verifier's own report text is
  `"{len(jobs)} jobs, all statuses within the backend enum"`).
* Observed vs expected: the real behaviour is covered — but by the unit suite, not by the verifier:
  `workbench/tests/test_adversarial_metrics.py:31-52` (fabricated aggregate / missing category /
  duplicate category), `workbench/tests/test_paper_gate.py` (per-category gap), and
  `workbench/tests/test_mutation_effectiveness.py:237`. The release gate itself stays string/empty-vacuous.
* Impact: the artifact an operator reads as "release verified" can be green with the GPU gate or the
  aggregate contract regressed.
* Minimal fix: J5 → drive `runtime.verify_environment`/`require_launch_ready` on a CUDA-declaring model
  with an absent probe and require BLOCKED; J7 → call `metrics_extraction.parse_limn_output` with two
  categories and require no aggregate; J8 → require ≥1 job, else report PENDING ("no jobs to
  lifecycle-check, backend enum unexercised").
* Regression test: mutate `probe_source` handling out of `runtime` and require J5 to FAIL.
* Classification: CONFIRMED (for J5/J8 vacuity) / PROBABLE (for J7's coverage gap — the gap is proven,
  whether it counts as "weak" depends on the release spec).

## A-08 — acceptance.sh groups B and C cannot fail for the reason they claim
* Severity: P2
* file:function: `workbench/docker/acceptance.sh` → group B (lines 58-73, regex on line 64), group C
  (lines 110-119).
* Root cause: B's pattern requires a double-quoted literal (`` api<...>("/api/<endpoint>" ``) and
  `Promise.all[` on the same line; `App.tsx:29` uses a template literal
  (`` api<Analysis[]>(`/api/analysis?...`) ``) and the endpoints B names are not the ones bundled. C is
  advertised (line 11) as "served backend reports its own config roots via /api endpoints" but asserts
  only `isinstance(models, list) and models`.
* Repro (directon the real regex, with a realistic regression file):
  `for endpoint in models cohorts jobs runs; do grep -q "Promise.all\[[^]]*api<[^>]*>(\"/api/${endpoint}\"" /tmp/b.tsx; done`
  where `/tmp/b.tsx` contains `Promise.all([api<Model[]>("/api/models"), api<Job[]>("/api/jobs")])`.
* Observed: `missed models | missed cohorts | missed jobs | missed runs` (0 of 4). Group C on the live
  stack printed `PASS served backend reports 16 registered models from its own config` — a statement
  about count only, and it would print the same for a stack serving a different artifacts root.
* Expected: B blocks the reintroduction of a bundled startup load; C proves the served roots equal the
  declared ones.
* Impact: "PASS each startup endpoint loads independently" and "PASS … from its own config" are not
  evidence of what they assert.
* Minimal fix: B — match fetch calls with a real matcher (or count `Promise.all` containing ≥2 startup
  endpoints) and cover the mapping endpoints; C — read the roots from an endpoint that exposes them and
  compare with `operator_config.resolve_config()` in-process.
* Regression test: the synthetic `/tmp/b.tsx` above must make B fail.
* Classification: CONFIRMED

## A-09 — mutation_checks M3 never mutates, uses the healthy observation as its oracle, and its
"startup restore" does not exist
* Severity: P2
* file:function: `workbench/docker/mutation_checks.sh` → header line 5, `mutate()` line 36
  (`.venv/bin/python`, relative), M3 lines 84-104 (oracle on line 98), backup dir lines 28-31.
* Root cause: three separate defects. (1) M3 copies no file and runs the *unmutated* `compare_metric`;
  it greps its own output for `"FIXED"` (printed on line 93 whenever the code is alive), so the check
  cannot detect the regression it names. (2) `mutate()` resolves `.venv/bin/python` relative to the
  caller's cwd, while `$REPO` is computed — `rc=127` outside the repo root (reproduced with a scratch
  `mutate()` copy: `bash: line 1: .venv/bin/python: No such file or directory`). (3) The comment on
  line 28 claims "the startup restore below repairs a tree left mid-mutation", but line 30 does
  `rm -rf "$BACKUP_DIR"` at startup and no code path reads the backups back:
  `grep -rn mutation_backups workbench/` → only `mutation_checks.sh:29`.
* Repro: read lines 84-104; run `grep -rn "mutation_backups" workbench/`.
* Observed vs expected: expected per header line 5 — every check mutates the code to the broken
  behaviour and *fails* under it. Actual for M3 — pass iff the module imports and the contract holds;
  for a crashed run — mutated files stay mutated, and the next run deletes the only pristine copies
  (`rm -rf` before `mkdir`).
* Impact: a mutation suite whose M3/M4/M5 legs rest on greps (`driver: nvidia` at line 129,
  `--env-file /dev/null` at line 107) or on the live stack's current state reports
  `MUTATION CHECKS PASSED` without proving detection; and a SIGKILL mid-run leaves the repository with a
  mutated `run.sh` and no recovery path.
* Minimal fix: M3 — copy `paper_gate.py`, apply the old-behaviour edit, and require the assertion to
  change verdict; `mutate()` — use `"${REPO}/.venv/bin/python"`; delete line 28's claim or implement a
  restore-before-rm step.
* Regression test: run the suite with `${REPO}/workbench/artifacts/.mutation_backups` pre-populated by a
  killed run and assert `run.sh` is restored.
* Classification: CONFIRMED

## A-10 — first boot is honest (positive evidence, no defect)
* Severity: NONE
* file:function: `workbench/backend/storage.py:13-19`, `workbench/backend/main.py:58-63`.
* Evidence: `WORKBENCH_ARTIFACTS_ROOT=/tmp/fb-artifacts .venv/bin/python -c "list_runs()"` →
  `WorkbenchError index_unavailable | DuckDB index is unavailable. Rebuild it from canonical result JSON.`
  And with the live stack, `workbench/artifacts/{annotations,cohorts}.json` do not exist, yet
  `/api/jobs`, `/api/benchmark-runs`, `/api/cohorts` all returned `200` (`read_json` returns `[]`).
  The docker-compose/root mount layout is consistent: `/repo` = image build context only, `/data/FashionIQ`
  + `/third_party` + `/checkpoints` + `/artifacts` + `/results` are the operator trees
  (`docker compose config` binds, verified).
* Classification: CONFIRMED (as evidence supporting PC_DOCKER_QUICKSTART §2, not as a finding)

## A-11 — run.sh and docker compose parse `.env` differently; recovery env vars have no docker path
* Severity: P3
* file:function: `run.sh:32-50` (`env_value`) vs compose's dotenv parser;
  `workbench/config/workbench.env.example:14,26` vs `workbench/docker/.env.example`.
* Root cause: `env_value` matches `"${line%%=*}" == "$key"` literally after stripping from the first `#`;
  compose accepts `export KEY=VALUE` and quotes.
* Repro: a file containing `export WORKBENCH_BACKEND_PORT=9000` → `docker compose config` publishes
  `"9000"` while an `env_value`-equivalent loop prints `no-match key=[export WORKBENCH_BACKEND_PORT]`
  (the same body as run.sh:32-50, reproduced in `/tmp`).
* Also observed: `.env`-driven values that compose forwards correctly DO reach the container —
  `WORKBENCH_RUNTIME_ADAPTATION=sm_120` appears in `docker compose config` as
  `WORKBENCH_RUNTIME_ADAPTATION: sm_120` (docker-compose.yml:30) — but the per-model weight/interpreter
  overrides documented in `workbench/config/workbench.env.example` (`WORKBENCH_CLVC_RESNET50_WEIGHTS`,
  `WORKBENCH_LIMN_PYTHON`, …) are listed neither in `.env.example` nor in compose, so the docker path
  cannot set them. `operator_config._KEYS` accepts them (operator_config.py:12-41) if they are merely
  present in the process environment.
* Expected: one env grammar for the documented workflow.
* Impact: a quoted/`export`ed `.env` value silently falls back to the default in run.sh's own
  BASE/port reporting while compose uses the file value — the two halves of the same command disagree.
* Minimal fix: strip a leading `export ` (and quotes) in `env_value`, or read values with
  `docker compose config --format json` so there is exactly one parser.
* Regression test: `export`-prefixed and quoted `.env` values produce identical run.sh and compose views.
* Classification: CONFIRMED

## A-12 — `run.sh prepare` ignores a `.env` WORKBENCH_DATASET_ROOT (P1-D facet)
* Severity: P3
* file:function: `run.sh:218` (`--dataset-root "${WORKBENCH_DATASET_ROOT:-/data/FashionIQ}"`).
* Root cause: the one remaining shell-only read inside a command that already resolved its other
  settings from `.env`; compose does not forward a substitutable `WORKBENCH_DATASET_ROOT` either
  (docker-compose.yml:23-30 sets only port/URL/adaptation), and the running container's env confirms it:
  `docker inspect cir-workbench --format '{{range .Config.Env}}…' | grep WORKBENCH` lists no
  `WORKBENCH_DATASET_ROOT`.
* Repro: `grep -n WORKBENCH_DATASET_ROOT workbench/docker/run.sh` → line 218 only, `${…:-default}` form.
* Expected: the precedence contract stated in the file's own comment (run.sh:31-34), i.e. `.env` is
  honoured for every setting it documents (`.env.example:30` documents it).
* Impact: today the default is correct so nothing breaks; a `.env` override silently prepares a
  different dataset root than the operator configured, and the job later fails preflight for a reason
  the operator cannot see in the command line.
* Minimal fix: `--dataset-root "$(env_value WORKBENCH_DATASET_ROOT /data/FashionIQ)"`.
* Regression test: `.env` with `WORKBENCH_DATASET_ROOT=/other` must appear verbatim in the prepare argv.
* Classification: CONFIRMED

## A-13 — container selection by `--filter name=workbench` is unscoped
* Severity: P2
* file:function: `run.sh:114` (`docker ps -q --filter name=workbench | head -1`),
  and the same pattern in `release_verify.sh:91,94,114,117` (`docker ps -q --filter name=workbench`).
* Root cause: `name=` is a substring match on the container name, not a project-scoped lookup; the real
  container carries compose labels that identify it exactly
  (`com.docker.compose.project=docker`, `com.docker.compose.project.working_dir=/home/heheboiz/data/cir/workbench/docker`,
  `com.docker.compose.project.config_files=…/docker-compose.yml`), and `container_name: cir-workbench`
  (docker-compose.yml:17) is deterministic.
* Repro: `docker inspect $(docker ps -q --filter name=workbench|head -1) --format '{{index .Config.Labels "com.docker.compose.project"}} …'`.
* Observed: exactly one match today, so the failure is latent — but `head -1` picks arbitrarily among
  matches, and any second project containing "workbench" would redirect doctor's GPU probe (and, at
  HEAD, J11's restart) at a foreign stack.
* Minimal fix: `docker ps -q --filter label=com.docker.compose.project.working_dir=<HERE>` or
  `--filter name=^cir-workbench$`.
* Regression test: create a decoy container named `workbench-decoy` and assert doctor still targets
  `cir-workbench`.
* Classification: CONFIRMED (latent)

## A-14 — GPU override and its fail-closed path behave as documented (verified)
* Severity: NONE
* Evidence: `docker compose -f docker-compose.yml -f docker-compose.gpu.yml --env-file .env run --rm -T workbench python -c "print('GPU-CONTAINER-OK')"`
  → `Error response from daemon: could not select device driver "nvidia" with capabilities: [[gpu]]`
  (no nvidia runtime on this host: `docker info --format '{{json .Runtimes}}'` = `io.containerd.runc.v2`, `runc`).
  `WORKBENCH_ENABLE_GPU` unset in `.env` → the real `bash run.sh doctor` printed the PENDING line and did
  not claim a capability. `--quiet` does **not** swallow stdout: `docker compose run --rm -T --quiet
  workbench python -c "print('STDOUT-VISIBLE')"` printed `STDOUT-VISIBLE`, so doctor's
  parse-the-last-line design (run.sh:139-181) is sound, and the daemon error degrades into
  "gpu probe produced no parseable result" + exit 3. Note `run.sh:139` still uses the base+override
  resolved set, so P1-C (base-compose-only GPU probe at HEAD:67) is fixed in the worktree.
* Also verified: a `WORKBENCH_FRONTEND_PORT=15173` `.env` publishes host 15173 while the container
  keeps 5173 (`docker compose config`), matching the comment at docker-compose.yml:24-28.

## A-15 — RO canonical mounts are complete for images/ and image_splits/ (verified)
* Severity: NONE
* Evidence (in-container, read-only):
  `docker exec cir-workbench sh -c 'grep FashionIQ /proc/self/mountinfo'` →
  `/data/FashionIQ/image_splits ro`, `/data/FashionIQ/images ro`;
  in-container `os.statvfs` → ST_RDONLY for both, rw for `/data/FashionIQ` and `captions/`.
  A rename of the mountpoint is refused by the kernel from the same host:
  `mv /data/FQ/images /data/FQ/images_bak` → `Resource busy`.
* Residual (policy, not kernel): `captions/`, `resized_image/`, `qwen_captions/` are writable by design
  because LIMN/TGCIR derive `correction_dict_*` and resized images there
  (registry `preparation_contract: manual_ilearn_resized`; `workbench/replay/limn.py:66,68`), and
  `docker-compose.yml:35-38` says exactly that. The resume guarantee for those derived inputs comes from
  the stage fingerprint (`scripts/pipeline.py:1000-1025` via `fashioniq_required_paths`,
  tree-walk hashing at `pipeline.py:203-277`), not from the mount table — worth stating in the docs so
  "canonical inputs are immutable" is not read as covering every file under the dataset root.
* Classification: CONFIRMED (as evidence)

## Execution trace (read-only; no repo state mutated)

```
git log --oneline -3                       → e9cafaf, 3310b5b, 767ae3d
git status --porcelain                     → M run.sh, M release_verify.sh, M evaluate_models.py,
                                             M environment_planner.py, M runtime.py, M models.yaml,
                                             ?? docs/reviews/final_hardening/
docker version                             → 29.9.0 ; docker compose version → v5.6.0
cd workbench/docker && docker compose -f docker-compose.yml -f docker-compose.gpu.yml \
    --env-file .env config                 → binds to /home/heheboiz/data/cir/{data/FashionIQ,trees…};
                                             images+image_splits read_only: true
docker ps                                  → cir-workbench Up (docker-workbench)
docker exec cir-workbench id               → uid=0(root)
docker exec … /proc/self/mountinfo         → image_splits ro, images ro
docker exec … python -c 'os.statvfs(...)'  → ST_RDONLY images/, image_splits/
docker run --rm -v /tmp/rotest/... alpine  → rename mountpoint: Resource busy
docker run --rm … md5sum /repo/workbench/docker/.env
                                           → 21e9c24b056b4e8f70ec97b18d8e911f == host file
scratch builds (/tmp/ignoretest, /tmp/ig3) → *.env does not exclude workbench/docker/.env
docker compose … run --rm -T --quiet …     → --quiet keeps stdout (STDOUT-VISIBLE)
docker compose -f … -f ….gpu.yml run …     → could not select device driver "nvidia"
WORKBENCH_ENABLE_GPU unset + bash run.sh doctor
                                           → GPU validation: PENDING (no fabricated capability)
bash run.sh doctor                         → /api/models: 000 ; runs|jobs|benchmark-runs|cohorts: 200
curl -w '%{http_code} %{time_total}' /api/models
                                           → 200 71.06 ; len(models)=16
curl /api/jobs                             → []
docker restart cir-workbench (10.4 s) then curl
                                           → 200/200, /api/jobs = [], queue_jobs absent
/tmp/rt2 repro: .env WORKBENCH_BACKEND_PORT=80a0 + bash run.sh doctor
                                           → “bad substitution”, rc=1   ; inline comment ⇒ same
docker inspect $(docker ps -q --filter name=workbench|head -1) --format labels
                                           → com.docker.compose.project=docker, working_dir=…/workbench/docker
env -i .venv/bin/python -c resolve_config() → defaults: FASHIONIQ_ROOT=<repo>/data/fashionIQ_dataset (data/FashionIQ symlink)
env -i FASHIONIQ_ROOT=/x WORKBENCH_ARTIFACTS_ROOT=rel/art … resolve_config()
                                           → /x ; <repo>/rel/art ; <repo>/rel/art/checkpoints
WORKBENCH_ARTIFACTS_ROOT=/tmp/fb-artifacts … list_runs()
                                           → WorkbenchError index_unavailable
ownership histogram (python os.walk)       → artifacts {0:25589, 1000:1623}; resized_image {0:15540}
```

## Final verdict

**BLOCKED** — the deployment slice is not releasable as it stands, for two independently
repo-verified reasons, both outside the known bug list:

1. **A-02 (P1, CONFIRMED):** the built image contains the operator's real `workbench/docker/.env`
   (byte-identical `md5` to the host file) because `.dockerignore:23`'s `*.env` cannot reach
   `workbench/docker/.env`. Shipping that image leaks host paths/credentials and contradicts the
   ignore file's own stated intent.
2. **A-01 (P1, CONFIRMED):** every malformed-or-commented port/GPU value in `.env` aborts `run.sh` with
   a bash *bad substitution* (rc 1) instead of the documented validated error (rc 2) — the exact guard
   the main agent just added for `.env` precedence is the part that fails.

Everything else is P2/P3 and listed above; the GPU fail-closed contract (A-14), the read-only canonical
mounts (A-15) and the first-boot `index_unavailable` honesty (A-10) were verified as correct, and the
two known P1s I was told not to re-investigate are indeed fixed in the worktree
(run.sh:32-79 now resolves `.env` with documented precedence; run.sh:114-140 now probes the resolved
set, and at HEAD the same probe used base-compose-only — `git show HEAD:workbench/docker/run.sh:67`).
Known P1-F is also already fixed in the worktree: J11 now uses a disposable `j11-persist-$$-$RANDOM`
container with a `mktemp -d` artifacts dir (release_verify.sh:338-374), so only the HEAD revision
restarted the production container (`git show HEAD:workbench/docker/release_verify.sh:355`).
