# AUDIT_B — reproduction-path audit (slice B)

Model under audit: **limn** (`workbench/registry/models.yaml:904-989`) — the only registered model
that declares `adapted_runtime` (`target_compute_capability: sm_120`, `pip_index_url:
https://download.pytorch.org/whl/cu128`, `models.yaml:945-953`).

Tree state: working tree of `research/cir-failure-workbench`, HEAD `92ecbd94` + 13 uncommitted
files (`git diff --name-only`). Uncommitted deltas read: `pc.sh`, `backend/jobs.py`,
`backend/main.py`, `backend/runtime.py`, `docker/release_verify.sh`, `frontend/App.tsx`,
`frontend/types.ts`, 4 test files + 2 fixtures. Deltas NOT yet reviewed by me as theirs: none
target slice B except `runtime.py`, `main.py`, `release_verify.sh`, `pc.sh`, `jobs.py` (all read).

Tags: `[STATIC]` source read · `[EXEC]` command observed on this host · `[INFERENCE]` stated
deduction. GPUs were not used; no runtime GPU claim is made anywhere in this report.

---

## 1. Runtime adaptation propagation

**Q1.1 — Is there one adaptation source?** `VERIFIED`. `runtime.runtime_adaptation()`
(`runtime.py:778-803`) reads `WORKBENCH_RUNTIME_ADAPTATION` and requires it to equal
`environment.adapted_runtime.target_compute_capability` from the *model dict it is handed*
(`runtime.py:793-802`). Every hop that decides the verified dependency set routes through
`runtime.resolved_runtime_environment()` or `runtime.runtime_adaptation(model)`:
`environment_planner.py:377-378` (planner), `environment_planner.py:806` (installer),
`evaluate_models.py:291-296` (provenance), `replay/limn.py:91-95`, and
`runtime.py:1489-1493` (`_environment_contract_digest`). `[STATIC]`

**Q1.2 — Does one hop read a different source?** `CONFIRMED`, P3. `pc.sh:552-559` derives
`target_adaptation` from the registry **file** (`models.yaml`) and halts with `sm_120`, not from
the registry the process would load (`WORKBENCH_REGISTRY_PATH`, `registry.py:20-23`); and
`pc.sh:546-550` hard-codes `if model == 'limn': print('sm_120')` as a fallback when the YAML parse
fails. So an operator registry that declares no adaptation still gets a container started with
`WORKBENCH_RUNTIME_ADAPTATION=sm_120` (`pc.sh:561`, `docker-compose.yml:35`), while the runtime
decides from the loaded model. Repro (no container needed): `grep -n "WORKBENCH_REGISTRY_PATH"
pc.sh` → no match. Recommended action: have `pc.sh` ask the container for the model's declared
target (`docker exec … python -c '… runtime.runtime_adaptation(model)…'`) instead of parsing the
YAML on the host. Impact today: benign — verified below.

**Q1.3 — Does the selection reach the evaluator?** `VERIFIED`. `WORKBENCH_RUNTIME_ADAPTATION` is
allow-listed into the evaluator child (`evaluate_models.py:441-456`, name at `:454`), is on the
container's `environment:` (`docker-compose.yml:35`), and pc.sh's config check compares the
container's value with the host's expectation (`pc.sh:633-641`), failing the run on mismatch when
`--reuse` forces a restart (`pc.sh:868-912`). `[STATIC]`

**Q1.4 — Consistency of the adaptation across hops on this host.** `VARIANT` (P3, no runtime
impact observed). `docker inspect cir-workbench` declares `WORKBENCH_RUNTIME_ADAPTATION=` (empty)
and no `WORKBENCH_LIMN_PYTHON`; container-computed `_gpu_capability_key` for all 5 CUDA-declaring
models + limn reproduced the host values only for limn
(`records matching a container-computed key: {'1d4407b5…45ca': ['limn']}`). That limn record was
written inside the container (`interpreter=/artifacts/environments/cir-1bbd9872cf166f0d/bin/python`)
with `runtime_adaptation: null` and `capability_verified: false`, i.e. the real host has not
probed sm_120 yet. `[EXEC]`

---

## 2. Interpreter assignment and fingerprinting

**Q2.1 — Adopted interpreter, and divergence from the key's interpreter.** `CONFIRMED`, **P1**.
`verify_environment` (`runtime.py:1246-1280`) resolves the interpreter as
`os.environ[python_env_var]`, and only when that is unset adopts the persisted assignment
(`runtime.py:1256-1275`). `_gpu_capability_key` (`runtime.py:1027-1046`) computes
`configured = os.environ.get(env.get("python_env_var") or "", "")` — **env var only**
(`runtime.py:1031`) — so the key falls back to `f390aea0…0ff1` (no interpreter, no fingerprint)
whenever the variable is unset, even though the reader is using a real interpreter.

Reproduction `[EXEC]` (throwaway sandbox `/tmp/audit_b/envvar2`, registry `WORKBENCH_REGISTRY_PATH`,
persisted `environment_assignment.json` = `RUNTIME_VERIFIED` under the artifacts root):

```
WRITE (var set) : key=6c93620407db03ea4bdaf1ccaad65008aae56dfeaa0979123692e109501ce257
READ  (var set) : cached=true  tier=GPU_RUNTIME_READY   key=6c936204…  (same)
READ  (var unset): cached=false tier=GPU_VERIFICATION_PENDING
                   interpreter=/tmp/.../environments/probe/bin/python
                   key=f390aea07d335e29e76c5c533d55ff4e8c246261f0a0bd20a1d9579f515e0ff1
```

Consequence: `record_gpu_capability` names the file after `_gpu_capability_key(model)`
(`runtime.py:968`, `:999`) and `_cached_gpu_capability(authoritative=True)` looks it up by the
same function (`runtime.py:1210-1213`); `release_verify.sh:570-612` compares
`record.capability_key == _runtime._gpu_capability_key(model)` and finally asks
`verify_environment(authoritative=True)`. With the env var unset in the *verifying* process —
exactly what `docker exec cir-workbench python3 workbench/scripts/verify_run.py` does
(`pc.sh:1010-1016`, where `docker inspect` showed **no** `WORKBENCH_*_PYTHON`), and what
`release_verify.sh` does in its own interpreter — the host's real capability record can never be
found, so `real_gpu` stays `False` and `GPU_RUNTIME` stays `PENDING` on the target hardware.

Measured on the shipped artifacts: 402 records carry `probe_source: interpreter`; 0 records
currently match any container-computed key other than limn's `capability_verified: false` probe
(`[EXEC]`, container query above). Minimal corrective action: resolve the interpreter through the
same path `verify_environment` uses (env var, then the persisted assignment) inside
`_gpu_capability_key`, or persist the key on the assignment/record and read it back.

**Q2.2 — Fingerprint binding of a record to its environment.** `VERIFIED` for the value that
*is* present. `_record_environment_fingerprint` (`runtime.py:1147-1163`) uses the same prefix
derivation as the key (`Path(interpreter).parent.parent`, unresolved), and the authoritative
reader recomputes it (`runtime.py:1239-1244`). `[STATIC]`

**Q2.3 — `report.environment_fingerprint is None` on the advisory path.** `EXPECTED_BEHAVIOR`
(design), but see Q2.4 for the consequence. `_remember(..., report)` zeroes
`environment_fingerprint` when `content_verified` is false (`runtime.py:1691-1703`); the
authoritative path sets it. `[STATIC]`

**Q2.4 — Advisory read does not re-check the environment the record describes.** `CONFIRMED`,
P2. `_cached_gpu_capability(authoritative=False)` (`runtime.py:1180-1208`) compares only
`model_id`, interpreter **string**, `dependency_fingerprint`, `host_identity`, signature and
`probe_source` — it never calls `_record_environment_fingerprint`. Rather than infer, I reloaded
the module and spied on the symbol:

```
[EXEC] reported (advisory=True, env drifted): reached_comparison: false, spy_calls: 0,
       advisory_tier=GPU_RUNTIME_READY, gpu_capability=true
```

A record probed under one interpreter therefore can certify another *in the same process* (the
`_GPU_CAPABILITY` in-memory pass at `runtime.py:1181-1188` matches on the interpreter **string**,
so even a distinct interpreter path can slip through when the record's `interpreter` field was
overwritten). Minimal corrective action: drop the `authoritative` split at the comparison
(`runtime.py:1240`) — compare the fingerprint on both paths, or at minimum require
`cached["interpreter"] == report.interpreter`.

---

## 3. GPU capability records

**Q3.1 — Signing and key derivation.** `VERIFIED`. `_gpu_capability_signature` HMAC-SHA256 over
the sorted record (`runtime.py:1050-1062`); key resolution order `WORKBENCH_LEDGER_KEY` →
`<CIR_REPO_ROOT>/.workbench_ledger_key` → `$XDG_CONFIG_HOME/cir/workbench_ledger_key`
(`runtime.py:1065-1125`); failures return `b""` so nothing is trusted. Observed: `signature_valid:
true` for every record written in the sandbox. `[EXEC]` The legacy path cannot shadow the
durable one: `/repo` in the image is root-owned (`Dockerfile:36-41`, root-owned; the only `chown` is `:64` on `node_modules`), the
compose runs `${WORKBENCH_UID:-1000}` (`docker-compose.yml:26`) and never mounts a file at
`/repo/.workbench_ledger_key`, and the `os.access(repo_root, W_OK)` branch is only reachable when
`XDG_CONFIG_HOME` is unusable (`runtime.py:1090-1097`); `XDG_CONFIG_HOME=/artifacts/config` is
declared in the image `ENV` (`Dockerfile:76`) and the compose env (`docker-compose.yml:40`).
`[STATIC]`

**Q3.2 — Host identity / expiry.** `VERIFIED` as implemented. `_host_identity` records node,
machine, `CUDA_VISIBLE_DEVICES` and the NVIDIA driver string (`runtime.py:1000-1025`) and is part
of the key hash (`runtime.py:1041-1045`); the record states `expires_on`
(`runtime.py:2063-2065`). No TTL is enforced (an unchanged host reuses a record indefinitely):
`EXPECTED_BEHAVIOR`, documented on the record itself as "revalidation".

**Q3.3 — The JUST-FIXED contract-digest write path.** `VERIFIED` **as fixed, both directions.**

* Source: `_remember` now sets `report.contract_digest = _environment_contract_digest(model)`
  *before* the cache gate (`runtime.py:1690`); HEAD `92ecbd94` sets it after
  (`git show HEAD:workbench/backend/runtime.py`, inside the `_cache_enabled()` early return).
* `WORKBENCH_NO_ENV_CACHE` **unset**: writer `contract_digest =
  6f93681f…92a5`, `env_fp = 2ddf791f…f8519`, `written_fp == env_fp`, fresh reader
  `tier=GPU_RUNTIME_READY`. `[EXEC]`
* `WORKBENCH_NO_ENV_CACHE=1`: writer `env_fp = null` but `contract_digest = 6f93681f…92a5`
  (identical), fresh reader `tier=GPU_RUNTIME_READY`, in the *same* mode and in the *other* mode.
  `[EXEC]` i.e. a record written with cache disabled is accepted with it unset and vice versa.
* Foreign interpreter still refused: two venvs A and B; record probed by B, reader resolving A →
  `GPU_VERIFICATION_PENDING / gpu_capability=false`; reader resolving B →
  `GPU_RUNTIME_READY / gpu_capability=true`. Also on full byte drift (new dist-info member added):
  authoritative reader refused (`GPU_VERIFICATION_PENDING`) while the advisory reader still
  accepted (`GPU_RUNTIME_READY`) — the Q2.4 finding. `[EXEC]`

**Q3.4 — Synthetic (`probe_source: override`) refusal in production.** Refusal held in every path
I exercised. `[EXEC]`

| path | switch | result |
|---|---|---|
| `verify_environment(advisory=True)` | unset | `GPU_VERIFICATION_PENDING`, `gpu_capability=false` |
| `verify_environment(authoritative=True)` | unset | `GPU_VERIFICATION_PENDING`, blockers non-empty |
| `require_launch_ready` | unset | `REFUSED: GpuVerificationError` |
| disk record written by an `override` probe, fresh process | unset | refused (never reached the comparison; see Q2.4) |

**Q3.5 — Same-process escape via the in-memory map.** `CONFIRMED`, P3.
`_GPU_CAPABILITY` is not filtered by `probe_source` on the advisory pass
(`runtime.py:1181-1188`) whereas the disk pass is (`runtime.py:1190-1195`), so in a process that
wrote the record while `WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE=1` *and* unset the switch later, a
synthetic record yields `GPU_RUNTIME_READY` and `require_launch_ready` **ALLOWS** the launch:

```
[EXEC] switch=1  -> tier=GPU_RUNTIME_READY, gpu_capability=true, launch_gate=ALLOWED
```

In production the switch would have to be unset between the write and the read inside one process
(single uvicorn worker, `serve_workbench.py:48-50`, no `--workers`), so this is a latent
in-process hole, not an open production bypass. Minimal corrective action: apply the same
`probe_source` filter to the in-memory pass as to the disk pass, or clear `_GPU_CAPABILITY` when
`GPU_PROBE_OVERRIDE_ENV` is unset.

**Q3.6 — Record shape hygiene.** `CONFIRMED`, P3. 2100 records in
`workbench/artifacts/gpu_capabilities/`: `probe_source` → `{override: 1562, interpreter: 402,
forged: 136}`; 136 records (all `forged`) carry neither `interpreter_fingerprint` nor
`dependency_fingerprint`. They are refused by `release_verify.sh:588-590` and by the
missing-fingerprint branch only insofar as that branch is unconditional — it is not
(`runtime.py:1240`), so such a record would be accepted by `_cached_gpu_capability` if it were
signed and its interpreter string matched. `[EXEC]` (directory census) + `[STATIC]`.

---

## 4. Environment cache identity

`VERIFIED` (no defect found). `_cache_key` = `model_id | python_env_var | configured |
contract_digest | cuda` (`runtime.py:1635-1638`); `_environment_contract_digest` covers python,
pytorch, torchvision, cuda, packages, required_imports, `interpreter_sha256`,
`environment_required` and the resolved adaptation (`runtime.py:1484-1503`), so a registry edit or
an adaptation switch invalidates it. `_reuse` re-verifies the fingerprint and the pinned
`interpreter_sha256` before reuse (`runtime.py:1665-1681`). `[STATIC]`

---

## 5. Parent/child environment consistency

**Q5.1 — Adaptation.** Consistent: `patch.dict(os.environ, child_environment)` in the autopilot
(`autopilot.py:477-502`) does not remove `WORKBENCH_RUNTIME_ADAPTATION`; the queue copies
`os.environ` and removes only the secret prefixes (`jobs.py:36-77`, allowlist
`WORKBENCH_LEDGER_KEY`); the evaluator allow-list contains the adaptation name
(`evaluate_models.py:454`). `[STATIC]`

**Q5.2 — Interpreter.** `autopilot.py:458-490` publishes `python_env_var → interpreter` into the
child environment only for `RUNTIME_VERIFIED` assignments; `verify_environment` falls back to the
persisted assignment when the variable is missing (`runtime.py:1256-1275`). This is the *only*
reason the GPU probe is reachable at all from `pc.sh`'s `docker exec` — and it is also what makes
Q2.1 visible (the fallback resolves an interpreter that `_gpu_capability_key` does not).
`[STATIC]`

---

## 6. Failed-install recovery

`VERIFIED` (no defect found). Every failure path removes the prefix (`environment_planner.py:1470,
1484, 1493, 1504…`) and `_clear_unverified_group_interpreter` (`:1741-1750`) plus
`plan["interpreter_mapping"] = {… verified only …}` (`:1789-1794`) prevent a dead path from being
published; an occupied prefix is never overwritten and never deleted (`:1416-1449`); a staging
prefix is refused as an environment (`:92-95`, `:1347-1350`). The `interpreter_sha256` mismatch is
enforced in three read paths (`runtime.py:1675-1681`, `:1675-1681` via `_reuse`, and
`release_verify.sh` J12). `[STATIC]`

---

## 7. The GPU probe (`gpu_check.py`, `pipeline.probe_gpu_capability`)

`EXPECTED_BEHAVIOR`, P3 note. `gpu_check.py:1-92` is an honest container-level report: it imports
the **container's** torch (`:44`), reports `PENDING` with a reason when nothing is visible
(`:56-60`) and returns 0 for `VERIFIED`/`PENDING` (`:88`). It is not the model capability path and
does not write a record — deliberately, so it cannot certify a model. The model path is
`pipeline.probe_gpu_capability` (`pipeline.py:1376-1410`) → `verify_gpu_runtime(authorized=True,
force=False)` (`pipeline.py:1400`) → capability gate (`runtime.py:2017-2032`) → arch/device check
against the adapted target (`runtime.py:2033-2060`). Note for the reviewer: `require_cuda=True,
force=False` means this stage **reuses** an existing capability instead of "Gather(verify) on the
GPU host" as its prose claims; `force=True` is only used by `evaluate_models.py:1920,1980`.
`[STATIC]`

---

## 8. Attack on the recent fixes (false PASS and false FAIL)

**F1 — `jobs.py` dispatch predicate still lets a legitimate queued job run.** `VERIFIED`.
`_next_job` returns `None` once `_stopping` is set and marks the job `preparing` under the same
lock (`jobs.py:345-355`); the spawn decision re-reads both under that lock
(`jobs.py:533-542`: `if job.status == "cancelled" or self._stopping: continue`), and
`shutdown()` sets `_stopping` and settles queued/preparing jobs in one `with self._condition:`
block (`jobs.py:284-301`). `_persist` is documented and used as lock-held (`jobs.py:443-448`), so
the added writes cannot interleave. `[EXEC]`
`.venv/bin/python -m pytest workbench/tests/test_concurrency_ownership.py -q` → **11 passed in
4.54s** (includes `test_shutdown_settles_a_queued_job_it_never_started`,
`test_shutdown_during_a_blocking_launch_guard_cannot_reach_popen`). No false-FAIL found. Cross-check
of the same file done in the parallel slice, not repeated here.

**F2 — API certification path: can it refuse a legitimate corroborated run?** `PROBABLE`, P2 —
**only reachable inside a dev/test process.**
`_development_receipt_refusal` (`main.py:214-236`) returns `""` (certificates allowed, proof not
validated) when `PYTEST_CURRENT_TEST` or `WORKBENCH_DEVELOPMENT_MODE` is *set to any non-empty
value*, or when `WORKBENCH_ALLOW_DEVELOPMENT_CERTIFICATES=1`. `_certify_report` consults it
**before** the proof branch (`main.py:347-350`), so a legitimate run whose completion proof exists
and validates is certified **ledger-level only, with `is_research_evidence` never evaluated**
(`main.py:304-314`) — a false PASS of the weaker question while the stronger one is skipped. In a
deployment the flag is absent, so `_verifier_verdict` (`main.py:279-301`) answers normally: no
false FAIL found. `[STATIC]`

**F3 — `release_verify.sh` dimension set: permanently PENDING dimension?** `CONFIRMED`, **P1**
(a false BLOCK, not a false PASS). Dimensions `MODEL_INFERENCE` and `MODEL_EVALUATION` have
`"checks": []` (`release_verify.sh:523-524`); with the new exhaustive guard
(`:531-537`, `elif statuses and all(...)`) an empty check list can only yield `NOT_APPLICABLE` or
`PENDING`, never `PASS`. The downstream `if _accepted_evidence:` branch *does* write
`MODEL_INFERENCE = PASS` (`:747-748`) **and** `SCIENTIFIC_OUTPUT = PASS` (`:751-752`) — i.e. it
overrides J7's own verdict, which is the real DESIGN_FALSE_PASS: J7's `PENDING` (no corroborated
run) becomes `PASS` for any run that only satisfies the weaker ledger/proof/admission test.

On the other hand the pending branch is correct in direction: with evidence absent,
`MODEL_INFERENCE` gets `PENDING` with the honest text "synthetic/self-test evaluator is not real
model execution" (`:765-767`) — my reading **disagrees** with any claim that this branch upgrades a
synthetic dimension to `PASS`. The shipped report confirms the empty-checks case cannot reach
`PASS`: `workbench/docker/release_verify_report.json` → `MODEL_INFERENCE PENDING`,
`MODEL_EVALUATION PENDING`, `GPU_RUNTIME PENDING`, `summary {pass:7, fail:0, pending:5}`, verdict
`GPU_VALIDATION_PENDING` reached purely through `mandatory_pending` (`:790-797`). `[EXEC]`

**F4 — J5's source-grep.** `CONFIRMED`, P3. `release_verify.sh:254` still asserts
`"probe_source" in src and "GPU_PROBE_OVERRIDE_ENV" in src` over `inspect.getsource(runtime)`, so
deleting the *behaviour* leaves J5 green. Mitigated (not fixed) by the real read-back at `:590-626`
(exact now — `:586-589` special-cases `override` explicitly and only a test switch sends the
`ALLOW_SYNTHETIC_GPU_PROBE=1` branch to `:584`, whose status mapping tests the switch alone, so a
synthetic record cannot be mapped to VERIFIED). `[STATIC]`

**F5 — `pc.sh` restart recheck.** `VERIFIED`. `require_corrected_configuration` re-runs the same
inspection after `run.sh up` and fails closed on residue (`pc.sh:742-753`, called at `:901` and
`:912`). Two P3 gaps: the remediation text names only the three host-path keys (`pc.sh:752`) while
the mismatch may be the adaptation (`:637-640`) or GPU visibility (`:626-630`); and the adaptation
is written into `.env` only when `.env` was freshly created (`pc.sh:568-583`,
`created_new_env` from `:520-526`), so a relaunch of an existing checkout can leave the container
with an empty adaptation and fail the recheck spuriously. `[STATIC]`

**F6 — frontend checkpoint bundles.** `CONFIRMED`, P2. The backend attaches the full receipt
(`main.py:452-467`, projection `main.py:484-516`) and the frontend types do not declare it and
never read it — `grep -rn "verification" workbench/frontend/App.tsx workbench/frontend/gate.ts
workbench/frontend/types.ts …` → **no matches** (only `lib.dom.d.ts` under `node_modules`), while
App.tsx renders the run card with warnings computed from `benchmark_gate` alone. A run the backend
marks `certified:false` is therefore presented as a normal run; the only honest marker left is the
`variant_provenance.caption_alias_staged` amber label (`frontend/gate.ts:44-52`). Corrective
action: surface `verification.certified` as a label (the projection already carries it).

---

## Findings table

| # | Verdict | Sev | Location | One-line |
|---|---|---|---|---|
| B-1 | CONFIRMED | P1 | `runtime.py:1031` (+`:1027-1046`, `release_verify.sh:570-612`, `pc.sh:1010-1016`) | `_gpu_capability_key` derives the key from the env var only, so a real capability record written under `WORKBENCH_<MODEL>_PYTHON` cannot be re-read by a process without it (docker exec verifier) → `GPU_RUNTIME` pinned PENDING |
| B-2 | CONFIRMED | P2 | `runtime.py:1180-1208` vs `:1239-1244` | advisory reader never recomputes the environment fingerprint (spy: 0 calls); same-process record reuse must be excluded to reject a stale synthetic record |
| B-3 | CONFIRMED | P1 | `release_verify.sh:531-537`, `:751-752` | an empty `checks: []` dimension can never PASS; and `_accepted_evidence` upgrades J7/`SCIENTIFIC_OUTPUT` to PASS, overriding the check's own PENDING |
| B-4 | CONFIRMED | P3 | `runtime.py:1181-1188` | in-memory capability map is not filtered by `probe_source`; synthetic record ACCEPTS launch if the switch is unset after the write |
| B-5 | CONFIRMED | P3 | `release_verify.sh:254` | J5 still greps source text; behaviour could be deleted undetected |
| B-6 | CONFIRMED | P3 | `pc.sh:546-559` | adaptation derived from the registry file (with a hard-coded limn fallback), not from the registry the runtime loads |
| B-7 | CONFIRMED | P3 | `main.py:229-231`, `:347-350` | `PYTEST_CURRENT_TEST`/`WORKBENCH_DEVELOPMENT_MODE` set to any non-empty value skips proof validation and can certify a run the verifier would refuse |
| B-8 | CONFIRMED | P2 | `main.py:452-467`, `frontend/types.ts`, `App.tsx` | receipt produced but never consumed; uncertified runs render as ordinary runs |
| B-9 | CONFIRMED | P3 | `pc.sh:752`, `:568-583` | remediation text omits the adaptation/GPU causes; adaptation written to `.env` only on first creation |
| B-10 | FALSE_PASS? no → EXPECTED_BEHAVIOR | P3 | `runtime.py:1000-1025` | no TTL on a capability; host/driver/`CUDA_VISIBLE_DEVICES` changes do invalidate it |
| B-11 | CONFIRMED | P3 | `artifacts/gpu_capabilities/*.json` (2100 records) | 136 unsigned/forged records lack both fingerprints; they pass the falsy-guard at `runtime.py:1240` |
| B-12 | KNOWN_LIMITATION | P3 | `pipeline.py:1376-1410` | the `gpu-verification` stage runs with `force=False`, so it reuses a cached record instead of re-probing |

No `EXTERNAL_DEPENDENCY_PENDING` finding: every claim above is source- or command-backed; the one
external input I could not exercise is real CUDA hardware.

---

## Files inspected / not inspected

Inspected (read in full or in the cited ranges): `workbench/backend/runtime.py`,
`backend/environment_planner.py`, `backend/jobs.py`, `backend/main.py`, `backend/registry.py`,
`backend/operator_config.py` (read only; not in manifest), `backend/preflight.py` (read only;
outside slice), `backend/autopilot.py` (read only; outside slice), `scripts/evaluate_models.py`
(read only; outside slice), `scripts/verify_run.py` (read only; outside slice),
`scripts/pipeline.py` (cited ranges), `scripts/gpu_check.py` (full),
`scripts/manage_environment.py` (full), `scripts/pc_readiness.py` (cited ranges),
`scripts/doctor.py` (cited ranges), `registry/models.yaml` (header + per-model blocks),
`replay/limn.py` (cited range), `docker/release_verify.sh`, `docker/Dockerfile`,
`docker/docker-compose.yml`, `docker/.env.example`, `frontend/App.tsx`, `frontend/types.ts`,
`frontend/gate.ts`, `pc.sh` (cited ranges), `workbench/docker/release_verify_report.json`.

NOT inspected: `src/**`, `backend/benchmark_references.py`, `backend/analysis.py`,
`backend/storage.py`, `backend/index.py`, `backend/metrics_extraction.py`, `backend/paper_gate.py`
(only its `data_kind` vocabulary was grepped), `backend/checkpoint_selection.py`,
`backend/fashioniq_layout.py`, `backend/adapters/*`, `registry/assets/*`, `registry/audits/*`,
`registry/auxiliary_assets.yaml`, `registry/preparation_contracts.yaml`,
`registry/paper_benchmarks/papers.yaml`, `docker/acceptance.sh`, `docker/mutation_checks.sh`,
`docker/run.sh`, `scripts/prepare_*`, `scripts/download_*`, `scripts/readiness_matrix.py`,
`scripts/build_paper_registry.py`, `pc.sh` outside the cited ranges, all test files except as
noted.

## Verdict

- Runtime adaptation propagation and the advisory/authoritative split: **VARIANT** (one source is
  read by all hops; `pc.sh` derives it differently and the key ignores the persisted assignment).
- GPU capability contract, both directions, including the just-landed `_remember` fix:
  **VERIFIED** — write-with-cache / write-without-cache cross-accepted, foreign interpreter
  refused, synthetic record refused on every production path. The one qualification is B-2/B-4:
  the *same-process* advisory reader is short of a check the authoritative reader performs.
- `release_verify.sh` dimension set: **VARIANT** — no satisfiable dimension is permanently
  unusable, but `SCIENTIFIC_OUTPUT`/`MODEL_INFERENCE` can be upgraded to `PASS` by a weaker test
  than their own check (B-3), and J5 is still a text assertion (B-5).
- Whole slice: **BLOCKED**, on B-1 alone: with no `WORKBENCH_<MODEL>_PYTHON` in the verifying
  process, a genuine sm_120 capability record cannot be loaded by key.

## Can this fail on the real first run on an RTX 5070 Ti PC?

Yes, and the most likely failure is not the GPU. With GPU_ENABLE, correct drivers and a successful
`--verify-gpu-only`, the record lands under `/artifacts/gpu_capabilities/<key>.json` with the
runtime adaptation unset (`runtime_adaptation: null` in the only limn record the container can
name), and the later `release_verify.sh` / `docker exec` verification process — where
`docker inspect` shows no `WORKBENCH_LIMN_PYTHON` — computes a *different* capability key
(`f390aea0…`, no interpreter) and reports `GPU_RUNTIME PENDING` with
"real_gpu: false" (report reproduction: `gpu_capability_scan.status=UNVERIFIED`,
`unverified_total=2096`, `accepted_evidence_runs=[]`). Q3.4's synthetic refusal is not the cause;
the cause is B-1. Second most likely: `pc.sh` restart/recheck (B-9), which writes the adaptation
into `.env` only when `.env` was created on that same invocation. On the GPU path itself nothing
I read can fake a pass (`probe_source: override` never certifies; `capability_verified: false` is
recorded honestly), so the more probable risk on this host is a **false FAIL of the release
verifier**, not a false PASS.
