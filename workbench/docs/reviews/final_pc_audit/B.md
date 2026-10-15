# Final PC Audit — Slice B: Runtime Adaptation & Environment Provisioning

Scope: `workbench/backend/runtime.py`, `workbench/backend/environment_planner.py`,
`workbench/scripts/manage_environment.py`, `workbench/scripts/pc_readiness.py`,
`workbench/backend/operator_config.py`, `workbench/registry/models.yaml`.
Base revision for this audit: **HEAD `0b0f3a6`**. A concurrent agent was editing the working tree
during this audit (`pc.sh`, `docs/PC_ONE_COMMAND_GUIDE.vi.md`, `workbench/backend/registry.py`,
`workbench/tests/*`); those files are **UNTRACKED/MODIFIED — not part of `0b0f3a6`**. Every `pc.sh`
claim below states the digest I read: `sha256 pc.sh = 572cca17a4f4aff10dc493e2538fd094f8d1fd252a33010f1c270449dad196d4` (1083 lines).
No GPU inference was run; every GPU claim is STATIC_AUDIT.

## A) Files inspected

Pinned target slice: `workbench/backend/runtime.py` (2323 L), `workbench/backend/environment_planner.py` (1912 L),
`workbench/scripts/manage_environment.py` (427 L), `workbench/scripts/pc_readiness.py` (429 L),
`workbench/backend/operator_config.py` (129 L), `workbench/registry/models.yaml` (989 L).
Consumed-by: `workbench/scripts/evaluate_models.py`, `workbench/scripts/pipeline.py`,
`workbench/backend/autopilot.py`, `workbench/backend/jobs.py`, `workbench/backend/preflight.py`,
`workbench/backend/errors.py`, `workbench/replay/limn.py`, `workbench/replay/common.py`,
`workbench/scripts/readiness_matrix.py`, `workbench/docker/{Dockerfile,docker-compose.yml,docker-compose.gpu.yml,run.sh}`,
`pc.sh` (uncommitted), `.dockerignore`, `workbench/tests/test_runtime_adaptation.py`,
`workbench/tests/test_final_hardening_regressions.py`, `workbench/tests/test_pc_launcher.py`.
Installed-package evidence: `.venv/lib/python3.13/site-packages/torch/cuda/__init__.py` (`_extract_arch_version`, `_check_cubins`).

## B) Execution paths traced

### B.1 `WORKBENCH_RUNTIME_ADAPTATION=sm_120` through every hop — one contract or several?

| Hop | Code | Reads |
|---|---|---|
| resolution (single source) | `runtime.py:776-799` `runtime_adaptation(model)` | `os.environ[ADAPTATION_ENV]` (`runtime.py:774`) vs registry `environment.adapted_runtime.target_compute_capability`; raises `UNKNOWN_RUNTIME_ADAPTATION` on mismatch |
| resolved contract | `runtime.py:803-835` `resolved_runtime_environment()` | `runtime_adaptation(model)` (**strict**, `undeclared_is_error=True`) |
| planner/solver | `environment_planner.py:368-396` `model_constraints()` | `resolved_runtime_environment()` + `runtime_adaptation()` (strict), with a documented fallback to historical pins for undeclared models (`:378-395`) |
| installer | `environment_planner.py:1500-1509` | `group["constraints"]["pip_index_url"]` seeded from `spec["pip_index_url"]` (`:510-513`) via `_constraints` (`:571-579`) |
| interpreter | same spec objects (`:1049-1088`) | no adaptation read |
| verifier | `runtime.py:1179`, `:1274` `runtime_adaptation(model, undeclared_is_error=False)`; `:1329` `report.runtime_adaptation` | per-model registry block, tolerant |
| planner→verifier bridge | `environment_planner.py:797-808` `_runtime_model()` | `resolved_runtime_environment()` (**strict**) ← **HOP THAT DISAGREES**, see C2 |
| GPU probe gate | `runtime.py:1910` `report.runtime_adaptation` then `:1912-1923` arch/device comparison | verifier report ← **BROKEN COMPARISON**, see C1 |
| evaluator child | `evaluate_models.py:454` allowlist entry → `:488-492` | `os.environ` |
| provenance | `evaluate_models.py:291-309`, `:1501-1503` | `runtime_adaptation(model)` (strict, wrapped in `except Exception: adaptation = None`) |
| readiness/CPU report | `pc_readiness.py:157`, `:303-306` | registry (raw), **not** the resolved contract |

Result: the *resolved* contract (historical pins replaced by `adapted_runtime`, plus `pip_index_url`,
plus `_adapted_runtime` provenance field) is genuinely threaded planner→solver→installer→verifier→probe→provenance,
and the historical and adapted contracts get **different environment ids** (verified:
`cir-1bbd9872cf166f0d` torch `1.12.1` vs `cir-5f868bc31646189b` torch `<3,>=2.7`).
Two hops violate it: `_runtime_model()` (C2) and the GPU-probe arch comparison (C1).

### B.2 No-adaptation model under the variable

```bash
$ WORKBENCH_RUNTIME_ADAPTATION=sm_120 .venv/bin/python -c "<runtime_adaptation for every registry model>"
csmcir -> REFUSED ('UNKNOWN_RUNTIME_ADAPTATION', "unknown WORKBENCH_RUNTIME_ADAPTATION='sm_120' for model 'csmcir': declared adaptation is none")
airknow, conesep, ptha_mtst, habit, intent, hint, encoder, pair, clvc_net, dcnet,
combiner_rn50x4_noft, clip4cir_rn50x4_fullft, tgcir, sprc -> REFUSED (same)
limn   -> {'target_compute_capability': 'sm_120', ...}
```
`runtime_adaptation(..., undeclared_is_error=False)` returns `None` for those models, so
`verify_environment` (`runtime.py:1274`) and `_environment_contract_digest` (`runtime.py:1430`)
do **not** force LIMN's profile onto them and do not crash — **that part is correct**.
`plan_environments` (`environment_planner.py:861`) also survives and keeps its per-model contracts
(verified: 10 groups, `csmcir` torch `==2.0.1`, only `limn` carries `pip_index_url`).
But the tolerant path is not honoured everywhere — the resulting failure is C2.

### B.3 Is CPU-only orchestrator + GPU-only isolated model environment the real design?

**Yes, and it is stated in the image contract**: `Dockerfile:38-40` "torch/torchvision (CPU wheels
suffice for orchestration; model environments are separate)"; observed `.venv torch 2.14.0+cpu`;
`runtime.py:1616/1647/1836` verify the *model's own* interpreter, never the orchestrator's;
`pc_readiness.py:254-257` explicitly refuses to let host torch reclassify a model.
Nothing in this slice breaks the split: pip/conda subprocesses receive
`_package_manager_env()` (`environment_planner.py:42-77`), probes receive `_probe_child_env()`
(`runtime.py:64-86`), both deny-prefix credential families, and the untrusted evaluator child
gets `_evaluator_environment()` (`evaluate_models.py:469-497`). The one break is not the split
itself but the extra-index policy (D4) and the capability key's host-side location (D5).

## C) Confirmed defects

### C1 — P0 — GPU-probe adaptation check can never pass: `sm_120` compared against `sm_`-prefixed arch names
- **Location**: `workbench/backend/runtime.py:1912-1918`
  ```python
  target = adaptation_target.replace("sm_", "").replace("_blackwell", "")   # "120"
  arch_list = [str(a).lower() for a in result.get("torch_arch_list") or []]
  if target not in arch_list:      # "120" in ["sm_70", ..., "sm_120"] -> False
      capability = False
  ```
- **Root cause**: `target` is a bare compute-capability number while `torch.cuda.get_arch_list()`
  returns prefixed names. Verified in the in-tree torch: `torch/cuda/__init__.py` `get_arch_list()`
  is `torch._C._cuda_getArchFlags().split()`, and torch's own consumer filters
  `supported_sm = [_extract_arch_version(arch) for arch in arch_list if "sm_" in arch]`, where
  `_extract_arch_version` does `arch_string.split("_", maxsplit=2)[1]` — i.e. elements are shaped
  `sm_<code>`. The comparison is unsatisfiable unless the list literally contained a bare `"120"`.
- **Reproduction** (static, no GPU needed):
  ```bash
  $ .venv/bin/python -c "a=['sm_70','sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']; print('120' in a)"
  False
  ```
- **Expected vs actual**: for an sm_120 adaptation on an RTX 5070 Ti the probe should accept
  `sm_120` in the arch list and `compute_capability 12.0`; actual → `capability = False`,
  `result["adaptation_mismatch"]` set, and since the same branch feeds
  `payload["capability_verified"]`, `record_gpu_capability` persists
  `capability_verified: false` (`runtime.py:963-985`) **plus** `verify_gpu_runtime` raises
  `GpuVerificationError` (`runtime.py:1940-1947`) because `require_cuda=True`.
- **Impact**: `gpu-verification`, `--verify-gpu-only`, the API launch gate
  (`main.py:78,411` → `require_launch_ready` → `runtime.py:1647-1657` → `LAUNCH_READY_TIERS`)
  and `evaluate_models.py:1866` all refuse. A correct cu128 torch on a real 5070 Ti is recorded
  as *no capability*. The first real run cannot reach an evaluation.
- **Note**: the sibling comparison `device_cap != target` (`:1920-1923`) is correct — it uses
  `"12.0".replace(".","") == "120"`.
- **Minimal fix** (mirror torch's own parser instead of a string replace):
  ```python
  def _arch_codes(values):
      return {re.sub(r"^(?:sm_|compute_)", "", v.lower()).removesuffix("a").removesuffix("f")
              for v in values}
  if target not in _arch_codes(result.get("torch_arch_list") or []):
  ```

### C2 — P1 — `_runtime_model()` re-resolves strictly: every model without an adaptation crashes the installer verifier
- **Location**: `workbench/backend/environment_planner.py:797-808` (`_runtime_model` → `runtime.resolved_runtime_environment`, strict at `runtime.py:813`), called from `:1267` (`_verify_group_interpreter`) and `:748/831` (`_group_status`).
- **Root cause**: `model_constraints()` deliberately falls back to historical pins for a model that
  declares no `adapted_runtime` (`environment_planner.py:378-395`, comment: "a process-global adaptation
  selection must not break unrelated models"), but the function that hands the model to the verifier
  calls the **strict** resolver and re-raises. The two disagree by construction. `_verify_group_interpreter`
  does not wrap the call, so the `WorkbenchError` escapes `_install_at_prefix` → `_provision_group`
  → `_provision_all_groups` → `provision_environment_plan`.
- **Reproduction** (stubbed `subprocess` + `inspect_environment_distributions` + `_evaluator_imports`
  so only this hop is exercised; `/tmp/hop_final.py`, run from the repo root with `.venv/bin/python`):
  ```
  --- WORKBENCH_RUNTIME_ADAPTATION unset ---
  csmcir: ok=False reason=csmcir: runtime.verify_environment failed: UNSAFE_INTERPRETER (cannot use system Python)
  limn:   ok=False reason=limn: runtime.verify_environment failed: UNSAFE_INTERPRETER (cannot use system Python)
  --- WORKBENCH_RUNTIME_ADAPTATION=sm_120 ---
  csmcir: RAISED WorkbenchError: ('UNKNOWN_RUNTIME_ADAPTATION', "unknown WORKBENCH_RUNTIME_ADAPTATION='sm_120'
          for model 'csmcir': declared adaptation is none", {'model_id': 'csmcir', 'requested': 'sm_120', 'declared': None})
  limn:   ok=False reason=limn: runtime.verify_environment failed: UNSAFE_INTERPRETER (cannot use system Python)
  ```
  (the `no var` rows prove the two branches differ only by the variable; the `ok=False` reasons are
  artefacts of pointing at a non-model interpreter and are irrelevant here).
- **Expected vs actual**: expected `ok=False, reason="<mid>: runtime.verify_environment failed: …"`
  — a refusal the installer records as `FAILED` (`environment_planner.py:1533-1540`). Actual: a
  `WorkbenchError` that the planner and installer do not catch. `WorkbenchError` is a `@dataclass`
  over `Exception` (`errors.py:7-8`), *not* a `RuntimeError`, so the guards written for it in the
  callers do not apply either.
- **Impact / blast radius**:
  1. Any provisioning/verification path that must verify a group containing a **non-adapted** model
     while the variable is set: `manage_environment.py:216/267/282` → `provision_environment_plan`,
     `plan_environments.py --allow-env-install`, `autopilot.py:428` (where the broad `except` at
     `:439` degrades it to a whole-run `assignment = {"error": …}`), `readiness_matrix.py:29`.
  2. `_install_at_prefix:1533` runs **after** `conda create` + `pip install`; the raise aborts the
     loop before the `FAILED`/cleanup bookkeeping at `:1535-1540`, orphaning a fully populated prefix.
  3. Escalation to the RTX-5070-Ti first run: `pc.sh` **writes** `WORKBENCH_RUNTIME_ADAPTATION=sm_120`
     into `workbench/docker/.env` for `--model limn` (`pc.sh:549`), and
     `docker-compose.yml:35` injects that into the container environment. Every later container
     process then sees `sm_120`, and any `prepare csmcir` — including through
     `manage_environment.py`'s shared-assignment branch, whose group is not the caller's model
     (`manage_environment.py:200-216` passes all group members) — dies as in (1).
  LIMN itself is **not** affected (it declares `sm_120`, `registry/models.yaml:945`). The stated
  requirement "a model declaring NO adapted_runtime … does NOT crash under that variable" is
  therefore **not met** at this hop.
- **Minimal fix** (two lines, and it makes every `_runtime_model` caller safe):
  ```python
  def resolved_runtime_environment(model, *, undeclared_is_error=True):
      adapted = runtime_adaptation(model, undeclared_is_error=undeclared_is_error)
  # environment_planner._runtime_model:
  env = dict(runtime.resolved_runtime_environment(model, undeclared_is_error=False))
  ```
  A model that declares a *mismatched* adaptation still raises (`runtime.py:793-798` keeps
  `target` non-empty), so this does not weaken the registry contract.

### C3 — P1 — Historical pin `torch==1.12.1` cannot be installed from the configured pip index
- **Location**: `environment_planner.py:1500-1509` (pip argv), `workbench/registry/models.yaml:938` (`pytorch: '1.12.1'`, `cuda: '12.4'`), `third_party/LIMN/requirements.txt` (`torch==1.12.1`, `torchvision==0.13.1`).
- **Root cause**: for the historical contract no `pip_index_url` exists (`registry/models.yaml:950` puts it only inside `adapted_runtime`), so pip resolves against PyPI's default index. Derived argv for LIMN's historical group (computed from `_pip_specs_for_group`, `environment_planner.py:1114-1144`):
  `python -m pip install torch==1.12.1 torchvision==0.13.1 clip …` — no `--extra-index-url`.
- **Evidence (STATIC_AUDIT, no download performed)**: PyPI never hosted CUDA wheels for the
  1.x line; historically, for torch 1.12/cu113, `pip install torch==1.12.1+cu113` answers
  *"No matching distribution found"* and must be fetched from `download.pytorch.org/whl/cu113`
  (PyTorch prior-versions/install-matrix docs). The registry's own `cuda: '12.4'` for that pin is
  internally inconsistent — torch 1.12 has no cu124 build.
- **Expected vs actual**: expected a working environment in which the 3 LIMN checkpoints can be
  loaded (`LIMN_CHECKPOINTS` binds `0_dress|shirt|toptee_best_model.pt`, `replay/limn.py:24-31`).
  Actual: pip exits non-zero → `_install_at_prefix:1520-1526` records
  `pip install exited N: …` and `state=FAILED`.
- **Impact**: without `WORKBENCH_RUNTIME_ADAPTATION`, LIMN is uninstallable on the target PC; the
  historical reproduction path exists only as documentation. `pc_readiness.py --model limn`
  currently reports only `BLOCKED_ENVIRONMENT … WORKBENCH_LIMN_PYTHON not set` (observed run) and
  cannot surface this class of failure.
- **Classification**: **PROBABLE** — the behaviour difference is not reproducible here (no
  network install was performed, by instruction). See D3 for the exact command to settle it.

### C4 — P1 — GPU capability is written but can never be re-read in the deployed container
- **Location (writer/reader)**: `runtime.py:1057-1101` `_gpu_cache_key()` — `WORKBENCH_LEDGER_KEY`, else
  `Path(resolve_config().CIR_REPO_ROOT) / ".workbench_ledger_key"`, created on first use.
  Gate: `_cached_gpu_capability` `runtime.py:1131-1136` + `_gpu_capability_signature_valid` `:1104-1110`.
  Bridge: `schema_layout=runtime.verify_gpu_runtime` → `record_gpu_capability(model, payload, config)`
  with `config` = `resolve_config()` (`pipeline.py:1399-1401`, `evaluate_models.py:1851-1853`, `main`).
- **Root cause**: in the image `CIR_REPO_ROOT=/repo` (`Dockerfile:64`) and `/repo` is owned by root
  while the container runs as uid 1000 (`docker-compose.yml:22`, `Dockerfile:57-59` chowns only
  `node_modules` and `/artifacts`), and `.dockerignore` excludes `**/.env` so no key file ships.
  `.env.example` and `.env` are bind-mounted? no — only `env_file:` (compose reads it, it is not exported
  as `WORKBENCH_LEDGER_KEY` unless the line exists; `workbench/docker/.env.example` contains **no**
  `WORKBENCH_LEDGER_KEY`). Therefore `WORKBENCH_LEDGER_KEY` is unset in the container.
- **Reproduction** (throwaway roots in `/tmp`; repo untouched):
  ```bash
  $ CIR_REPO_ROOT=/tmp/ro-repo WORKBENCH_ARTIFACTS_ROOT=/tmp/ro-artifacts .venv/bin/python -c \
    "from workbench.backend import runtime; print(repr(runtime._gpu_cache_key()))"
  b''
  $ WORKBENCH_LEDGER_KEY=anything python -c "…"   # -> b'anything' (env override works)
  ```
  `_gpu_cache_key` then ignores the `OSError` → `b''` (`runtime.py:1100-1101`) →
  `_gpu_capability_signature` returns `None` → the record is stored **unsigned** (`:973-976`) →
  `_cached_gpu_capability` rejects it (`:1133-1136`, `:1124-1150`) → `verify_environment` never
  reaches `TIER_GPU_READY` (`:1364-1370`), so `require_launch_ready` raises forever.
- **Expected vs actual**: the module docstring promises "a later preflight or doctor run must see it"
  (`runtime.py:1127-1129`); the actual container cannot authenticate any record it writes.
- **Impact**: even after C1 is fixed, every launch re-probes (assuming a writable repo root, which
  it is not — the write itself raises `PermissionError` from `key_path.parent.mkdir` at `:1082` on
  the read-only-in-practice `/repo`), and a *successful* probe still leaves the gate closed.
  24/7 `restart: unless-stopped` (`docker-compose.yml:52`) does not help: the bytes are lost.
- **Note**: the same failure mode is guarded on the *ledger* side — `evaluate_models.py:1128-1140`
  refused to keep the key at the repo root and moved it to `XDG_CONFIG_HOME/cir/workbench_ledger_key`
  (compose pins `XDG_CONFIG_HOME=/artifacts/config`, `docker-compose.yml:39`).
  `_gpu_cache_key` was **not** migrated; only the ledger key reader is. The `except OSError: return b""`
  makes this silent — no diagnostic reaches the operator.
- **Minimal fix**: resolve the capability key from the same private path the ledger uses
  (`XDG_CONFIG_HOME/cir/workbench_ledger_key`) before the repo-root fallback, and have `pc.sh`
  create that file (32 random bytes, mode 600) under the artifacts mount if absent. Do not move to
  "unsigned cache": an unauthenticated capability is exactly what the signature exists to prevent.

### C5 — P2 — `--extra-index-url` lets a substitute/deprecated PyTorch index shadow the declared CUDA index
- **Location**: `environment_planner.py:1502-1509`.
- **Root cause**: the adapted profile needs `https://download.pytorch.org/whl/cu128` for
  `torch`/`torchvision`, but the same pip line carries ordinary PyPI names, so the code uses
  `--extra-index-url` (deliberate, `:1503-1506`). pip has no index priority: with extra indexes it
  is documented to pick the *highest version*, and the same-host
  `download.pytorch.org/whl/cu129` currently serves a `torch` **substitute** distribution; a CPU
  wheel with a larger version can likewise win.
- **Impact**: the installed torch may be a build without cu128/sm_120 kernels *while satisfying*
  `torch>=2.7,<3`, so every version check in `_verify_group_interpreter:1181-1197` and
  `verify_environment:1290-1325` passes. Only C1's arch gate would catch it — and C1 is broken,
  which is what makes this a real hole rather than a theoretical one.
- **Classification**: **PROBABLE** (pip's resolver behaviour is documented; the actual resolution on
  the target index is not reproducible offline).
- **Minimal fix**: split the pip line — install `torch`/`torchvision` with
  `--index-url <declared index>` first, then the PyPI-only names with the default index; or at
  minimum assert the installed torch's `version.cuda` equals the adapted `cuda` value before
  marking the group `RUNTIME_VERIFIED`.

### C6 — P2 — CPU readiness report is adaptation-blind
- **Location**: `pc_readiness.py:150-159` (`"pytorch": raw registry value`, `"adapted_runtime": raw
  block`), `:303-306` (`intended_gpu.compute_capability: "sm_120"`); contrast
  `readiness_matrix.py:45,127` which reports `group["runtime_adaptation"]` and
  `environment_adapted` correctly, and `runtime.py:1972-1982 environment_blockers()` which consumes it.
- **Root cause**: the report reads registry fields directly instead of
  `runtime.resolved_runtime_environment(model)`.
- **Reproduction** (observed, repo root):
  ```bash
  $ WORKBENCH_RUNTIME_ADAPTATION=sm_120 timeout 120 .venv/bin/python workbench/scripts/pc_readiness.py --model limn --model csmcir
    limn                       BLOCKED_ENVIRONMENT
        execution environment not verified: WORKBENCH_LIMN_PYTHON=None [UNCONFIGURED/UNCONFIGURED] …
  $ WORKBENCH_RUNTIME_ADAPTATION=sm_120 .venv/bin/python -c "<environment_section per-model>"
    limn -> pytorch=1.12.1  cuda=12.4  adapted_runtime={…sm_120…}   # contract actually in force is torch>=2.7,<3 / cu128
  ```
- **Expected vs actual**: the report should show the contract the run will verify against
  (`torch >=2.7,<3`, `cuda 12.8`, cu128 index) and mark the model adapted; it shows the historical
  pins and a raw block. The `pc.sh --doctor` surface (`pc.sh:206-235`) inherits this.
- **Impact**: the operator's PC-side readiness view cannot distinguish "no adapter" from
  "adapter declared but not selected", and shows a CUDA string that is not the one installed.
- **Minimal fix**: derive `pytorch`/`cuda`/`adapted` from `resolved_runtime_environment()` and add
  `"adapted_runtime_selected": bool(runtime.runtime_adaptation(model, undeclared_is_error=False))`.

### C7 — P3 — `ENVIRONMENT_REQUIREMENTS_UNVERIFIED` tier has no consumer
- **Location**: declared `runtime.py:32`; set at `:1348`, `:1370-1373`, `:1376-1381`, `:1395-1402`;
  consumers `READY_TIERS` (`:1409`), `LAUNCH_READY_TIERS` (`:1411`), `environment_blockers`
  (`:1972-1982`), `pc_readiness.py:259-262`, `manage_environment.py:68`.
- **Root cause / impact**: the status string `"REQUIREMENTS_UNVERIFIED"` is a member of
  `_BLOCKED_STATUSES` (`:1403-1406`) so the *status* string is consumed, but the *tier* constant has
  no gating use. `environment_blockers` returns the generic
  `execution environment not verified: …` message and discards the tier, and no code branches on it.
  Dead weight plus one inconsistent status string: `:1348/:1376/:1400` use the
  `"REQUIREMENTS_UNVERIFIED"` literal while `:1370` uses the tier constant for the same state.
- **Minimal fix**: use `TIER_REQUIREMENTS_UNVERIFIED` for all three sites, or delete the constant.

### C8 — P3 — Silent fallback in `model_constraints()` can verify against the wrong contract
- **Location**: `environment_planner.py:384-395`, in particular `except Exception: env = dict(model.get("environment") or {}); adaptation = None` (`:391-395`).
- **Root cause**: only `WorkbenchError` is handled precisely; any other resolution failure is treated
  as "historical contract". A bug in `resolved_runtime_environment` would therefore make the planner
  plan, install and verify the historical pins *while the operator asked for the adaptation*, with a
  provenance field (`_adapted_runtime`) absent — i.e. a silent downgrade.
- **Classification**: **CONFIRMED** (code fact); no triggering input found, so severity P3.
- **Minimal fix**: re-raise non-`WorkbenchError` exceptions, or record a
  `runtime adaptation resolution failed: …` blocker in `spec["evidence"]`/`unresolved`.

## D) Probable defects needing reproduction

### D1 — `sm_120` reaching the container through `docker compose exec`
**PROBABLE (would make C2 fire on `prepare csmcir`).** docker/compose's `exec` reference documents
only `-e/--env` as the way to set variables on the exec'ed process and states no host-inheritance;
the general Compose rule is that shell variables reach containers only via an `environment:` mapping
or `--env-file`. `workbench/docker/run.sh:203` uses `compose exec -T workbench python …` with no
`-e`, so the variable reaches the pipeline **only** via the container's own environment, which
`docker-compose.yml:35` (`WORKBENCH_RUNTIME_ADAPTATION: "${WORKBENCH_RUNTIME_ADAPTATION:-}"`) or
`pc.sh:549` writing it into `.env` does supply. `pc.sh` even depends on the negative (its own
export of the variable is deliberately suppressed for a non-adapted model, `pc.sh:555`, which is only
safe under the no-inheritance reading).
**Settle it on the target host**: `docker compose exec -T workbench env | grep RUNTIME_ADAPTATION`
after `WORKBENCH_RUNTIME_ADAPTATION=sm_120 ./run.sh up` versus after setting it only in the host shell.
If it *is* inherited, pc.sh:445's `sed -i` write-back of `HOST_FASHIONIQ_ROOT` and the absence of a
negative assertion make every preparation path inherit the C2 crash.

### D2 — LIMN (torch ≥2.7) replay correctness under the adapted contract
**UNKNOWN.** The adapted package map keeps the historical train/test pins from
`third_party/LIMN/requirements.txt` (`model_constraints` output: `numpy 1.21.5`, `packaging 23.1`,
`pillow 9.4.0`, `tqdm 4.65.0`, `pyyaml UNKNOWN`) while `replay/limn.py:160-163` unpickles a whole
model with `weights_only=False` (`load_trusted_whole_model`, `replay/common.py:36-48`) and calls
`open_clip.image_transform`, not the pinned `open_clip==2.20.0`. Nothing in this slice tests the
adapted stack's imports; `execution_plan`/provenance records only `runtime_adaptation`
(`evaluate_models.py:1501-1503`), not the resolved constraint set.
**Settle it**: run `evaluate_models.py --model limn --verify-gpu` on the target and confirm the
checkpoint deserializes; then assert `open_clip_torch`/`timm`/`numpy` versions in the report against
the adapted map.

### D3 — Historical LIMN pin installability (ties to C3)
**Reproduce offline-safely with a URL probe only, no install**:
`pip index versions torch --index-url https://pypi.org/simple` vs.
`curl -sI https://download.pytorch.org/whl/cu113/torch/ | grep -c 1.12.1`. Neither was performed here.

### D4 — Is `timm==0.9.12`/`open_clip_torch>=2.24,<2.27` internally consistent with the `pip_index_url` extra-index rule?
**PROBABLE.** `open-clip-torch` pivoted to a `timm` dependency range that may exclude/require newer
`timm` than the historical pin retained in the adapted map; combined with C5's extra-index rule this
can silently resolve a `timm` version the adapted LIMN code was never tested with. Settle on the
target with `pip install --dry-run` (read-only) and compare against the adapted map.

## E) Test coverage gaps

| Requirement | Existing test | Gap |
|---|---|---|
| adaptation changes the plan | `test_final_hardening_regressions.py:22-45`, `test_runtime_adaptation.py` | only LIMN fixtures; no test asserts a **non-adapted** model survives with `sm_120` set |
| verifier consumes the resolved contract | `test_final_hardening_regressions.py:48-66` calls `_runtime_model(limn_model, spec)` | LIMN only → C2 is untouched by the suite; add the same call for `csmcir`/`airknow` under `sm_120` |
| installer path | none | `_verify_group_interpreter`/`_install_at_prefix`/`_provision_group` are only exercised through e2e mocks; C2's raise is invisible |
| GPU adaptation gate | none | `grep -r "arch_list\|adaptation_mismatch" workbench/tests/*.py` → **0 hits**; `test_release_blocker_cuda_readiness.py` hardcodes `compute_capability: "8.9"` (sm_89). C1 has no test at all |
| capability persistence across processes | `test_release_blocker_cuda_readiness.py:761-775` (repo-root key) | no test for `CIR_REPO_ROOT` unwritable / key unset → `b""` → unsigned record rejected (C4) |
| originality/cache identity | `test_runtime_adaptation.py:44-52` | digest inequality only; no assertion that environment **ids** differ (verified manually: `cir-…0d` vs `cir-…89b`) |
| pc_readiness contract display | `test_pc_readiness.py` | no adaptation-aware expectation (C6) |

## F) Recommended minimal fixes (in order)

1. **C1 (P0)** — replace the arch-list membership test with prefix-aware parsing (`_arch_codes` above);
   keep the `device_cap` compare. Add the missing regression test (G1).
2. **C2 (P1)** — thread `undeclared_is_error=False` through `resolved_runtime_environment` and use it
   in `_runtime_model`; do not weaken the mismatched-declaration refusal. Add G2.
3. **C4 (P1)** — resolve the capability HMAC key from `XDG_CONFIG_HOME/cir/workbench_ledger_key`
   (mirroring `evaluate_models.py:1128-1140`) before the repo-root fallback; have `pc.sh` create that
   file (32 random bytes, 0600) under `WORKBENCH_ARTIFACTS_ROOT/config/cir/` on first run, and set
   `WORKBENCH_LEDGER_KEY` in `docker-compose.yml` from `.env` so a pod restart keeps the same key.
   Log loudly instead of returning `b""` silently.
4. **C5/D4 (P2)** — split the pip invocation so `torch`/`torchvision` come from
   `--index-url <declared>`, and assert the installed `torch.version.cuda` equals the adapted `cuda`.
5. **C3 (P1, PROBABLE)** — add the PyTorch wheel index for the *historical* pin (or mark LIMN's
   historical contract `UNINSTALLABLE_ON_PIP`), so the readiness report stops promising a historical
   reproduction it cannot provision.
6. **C6/C7/C8 (P2/P3)** — make `pc_readiness` report the resolved contract; use the
   `TIER_REQUIREMENTS_UNVERIFIED` constant consistently or delete it; stop swallowing unknown
   resolution errors in `model_constraints`.
7. **D1** — add one negative assertion in `pc.sh` after `configure_environment`: a non-adapted model
   must produce `WORKBENCH_RUNTIME_ADAPTATION` empty in `docker compose config`
   (`workbench/tests/test_pc_launcher.py` already has the shape for its positive assertion at `:232-252`).

## G) Exact regression tests required

- **G1 (C1, P0)** — `runtime.verify_gpu_runtime` with `set_gpu_probe_for_tests` returning
  `{"torch_arch_list": ["sm_80", "sm_86", "sm_90", "sm_100", "sm_120"], "compute_capability": "12.0",
  "cuda_available": True, "allocated_tensor": True, "computed": True, "torch_has_compiled_core": True, …}`
  and a model whose adaptation resolves to `sm_120` → must **not** raise and must persist
  `capability_verified: True`; then run it again with `torch_arch_list: ["sm_80", …]` → must set
  `adaptation_mismatch` and raise. Also assert the `_blackwell`-suffixed target spelling.
- **G2 (C2, P1)** — `monkeypatch.setenv("WORKBENCH_RUNTIME_ADAPTATION", "sm_120")` and call
  `environment_planner._runtime_model(csmcir_model, spec)` (and the same for every registry model
  with no `adapted_runtime`) → must return the historical environment, must **not** raise;
  and `_runtime_model(limn_model, spec)` in the same process → must return the adapted constraints.
- **G3 (C2 + D1, P1)** — `provision_environment_plan` with two planned groups, one adapted (`limn`)
  and one not (`csmcir`), the variable set, `allowed=True, network_allowed=True`, conda/pip stubbed:
  assert an `Exception` is **not** raised and `csmcir`'s group records a truthful
  `state`/`installation_status` rather than an escaped error. Plus a `pc.sh` test asserting
  `docker compose config` shows `WORKBENCH_RUNTIME_ADAPTATION:` empty for `--model csmcir`.
- **G4 (C4, P1)** — with `CIR_REPO_ROOT` pointing at a non-writable directory and
  `XDG_CONFIG_HOME` pointing at a writable one, `record_gpu_capability` followed by a fresh-process
  `_cached_gpu_capability` must return a `capability_verified` record; and with **no** key available
  at all it must fail closed (no unsigned record accepted) and surface a reason.
- **G5 (C5, P2)** — assert the pip argv built by `_install_at_prefix` for an adapted group puts
  `torch`/`torchvision` behind `--index-url <pip_index_url>` and the plain-PyPI names on the default
  index, i.e. the substitute-index hole is closed.
- **G6 (C3, P1)** — with the historical contract, `_pip_specs_for_group` must either emit the wheel
  index needed for `torch==1.12.1` or `pc_readiness` must classify LIMN as
  `BLOCKED_EXTERNAL_ASSET`/`BLOCKED_ENVIRONMENT` with a reason naming the missing index, never
  `READY_*`.
- **G7 (identity)** — plan with and without `WORKBENCH_RUNTIME_ADAPTATION=sm_120` and assert the two
  LIMN `environment_id`s differ and `group["runtime_adaptation"]` is `"sm_120"` only in the adapted
  plan (extends `test_runtime_adaptation.py` beyond digest inequality).

## H) Execution trace (commands run, all read-only or `/tmp`-scoped)

1. `git log --oneline -3` → HEAD `0b0f3a6`; `git status --porcelain` → slice files clean at audit start.
2. `grep -rn "WORKBENCH_RUNTIME_ADAPTATION|adapted_runtime"` across `workbench/**` → hop map in B.1.
3. `WORKBENCH_RUNTIME_ADAPTATION=sm_120 .venv/bin/python -c "<runtime_adaptation per model>"` → 15 refusals + LIMN adapted.
4. `… plan_environments(models, solver=lambda c: (True,'ok'))` → 10 groups; only LIMN carries `pip_index_url`; ids `cir-1bbd9872cf1660d` (historical) vs `cir-5f868bc31646189b` (adapted).
5. `timeout 120 … pc_readiness.py --model limn --model csmcir` → `BLOCKED_ENVIRONMENT` / `BLOCKED_SOURCE`; C6's adaptation-blind `environment` block confirmed separately.
6. `/tmp/hop_final.py` (stubbed `subprocess`/`inspect_environment_distributions`/`_evaluator_imports`) → C2 crash reproduced with and without the variable.
7. `CIR_REPO_ROOT=/tmp/ro-repo WORKBENCH_ARTIFACTS_ROOT=/tmp/ro-artifacts … _gpu_cache_key()` → `b''` (C4).
8. `python -c "['sm_70',…,'sm_120'].__contains__('120')"` → `False` + in-tree torch `_extract_arch_version`/`_check_cubins` → C1.
9. `timeout 300 readiness_matrix.py --json` under `sm_120` → did not finish in 300 s (it also blocks on the C2 path via `plan_environments`); no result produced, so **NOT_RUN** for that report.
10. `timeout 180 plan_environments.py --output /tmp/ep.json --list` under `sm_120` → exit 0 (writes to `/tmp`, non-provisioning), no crash: the pure planning path is safe.
11. Scratch only: `/tmp/auditvenv` (a `pip install` of `torch>=2.7,<3` from `--index-url https://download.pytorch.org/whl/cu128` **failed**: `OSError [Errno 122] Disk quota exceeded` at 1.8 GB) — so no empirical arch-list capture was obtained from a modern wheel. That experiment and the venv were **deleted** afterwards. Repository left untouched except the two lock files named in C-note below, which I created and removed.

**Repo cleanliness**: the only files I created under the repo are this report and two
`workbench/artifacts/environments/.provision-*.lock` files from a provisioning probe; I deleted my
own lock file and verified `git status` is unchanged relative to the concurrent agent's edits.

## Execution/verdict summary

Every checklist question, verdicts:

| Question | Verdict | Evidence | Severity |
|---|---|---|---|
| Adaptation resolution single-source | VERIFIED | `runtime.py:776-799`, `:803-835` | NONE |
| Historical vs adapted deps in installer | VERIFIED (`pip_index_url` reaches argv `:1507-1509`); two contracts get distinct env ids | §B.1 | NONE |
| Interpreter assignment | VERIFIED (`_runtime_model` C2 aside) | `environment_planner.py:1049-1088,1267` | NONE |
| CUDA wheel compatibility | VARIANT — index declared but implemented as `--extra-index-url` | `:1502-1509` | P2 (C5) |
| torch/torchvision pairing | VERIFIED — pairs table extended for 2.7–2.9 | `:168-190`, `:599-603` | NONE |
| GPU architecture support gate | **BLOCKED** — arch comparison unsatisfiable | `runtime.py:1912-1918` | **P0 (C1)** |
| CUDA tensor execution | NOT_RUN (no GPU; probe only reachable past C1) | `runtime.py:1730-1758` | P0 |
| Model environment isolation | VERIFIED | `runtime.py:200-245`, `:64-86`; `environment_planner.py:42-77` | NONE |
| micromamba/conda provisioning | VARIANT — in-image micromamba present; `allow_network` gate precedes solver check | `Dockerfile:27-30`; `environment_planner.py:1383,1391-1396` | NONE/P3 |
| Cache identity & reuse | VERIFIED — adaptation in contract digest, distinct ids | `runtime.py:1414-1432`; §B.1 | NONE |
| Failed-install recovery | VARIANT — split/retry OK, but C2 leaves an orphaned prefix | `:1659-1701`, `:1533-1540` | P1 |
| GPU capability records | **BLOCKED** — can be written but not authenticated in the image | `runtime.py:1057-1101,1131-1136` | P1 (C4) |
| Runtime verification | VERIFIED for interpreter/deps; pending for CUDA by design | `runtime.py:1179-1407` | NONE |
| GPU verification | **BLOCKED** (C1) | §C1 | P0 |
| Parent/child environment consistency | VARIANT — ONE hop disagrees (`_runtime_model` strict) | `environment_planner.py:806` | P1 (C2) |
| Secrets/credential isolation | VARIANT — deny-lists sound (`runtime.py:64`, `environment_planner.py:63`, `evaluate_models.py:459`, `jobs.py:47`), but the capability key materialises in the container's repo root instead of the private config home | §C4 | P1 |
| CPU orchestrator + GPU-only model env | VERIFIED design; not broken by this slice (broken by C4's key location, not by the split) | `Dockerfile:38-40`; §B.3 | NONE |
| No-adaptation model under `sm_120` does NOT crash | **BLOCKED** — holds in `verify_environment`/`_environment_contract_digest`, fails in `_runtime_model` | §B.2 + §C2 | P1 |
| `./pc.sh` first-run feasibility | VARIANT — orchestrates the real path correctly (preflight→config→up→prepare→probe→evaluate→verify) but every GPU path terminates at C1, and C4 makes the probe's record unusable | `pc.sh:523-611,670-703,1033` | P0 |

**FINAL VERDICT: BLOCKED.**

Can this implementation fail during the real first run on an RTX 5070 Ti PC? **Yes — with certainty for any run that reaches GPU verification: `runtime.py:1912` compares the bare string `"120"` against `sm_`-prefixed `torch.cuda.get_arch_list()` entries, so a correct cu128 torch on an sm_120 device is recorded as having no GPU capability and the launch gate stays closed.**
