# 89 — Adapted runtime declared for `clvc_net` + `tgcir`, and the GPU capability recorded

Branch `research/cir-failure-workbench`, 2026-11-12. Owned files, and nothing else:
`workbench/registry/models.yaml` (two `environment.adapted_runtime` blocks only),
`workbench/config/workbench.env` (two appended lines), `workbench/tests/test_composition_adapters_missing.py`,
this file. Not touched: the two composition modules and their worker modules, their parity tests,
`workbench/frontend/`, `workbench/systemd/`, `scripts/serve_workbench.py`, `backend/jobs.py`.

Every measurement below was run with `workbench/envs/ctl312h/bin/python` (it has `packaging`;
85 §10 records that an interpreter without it makes `resolve_config()` raise). The GPU probe ran
under `workbench/scripts/gpu_job.sh` with `MAX_WAIT=3600`, one job at a time.

---

## 1. Two appended interpreter pins (`workbench/config/workbench.env`)

The file is gitignored operator state (`.gitignore:213`), so this is not in the commit. Two lines
appended after the existing `WORKBENCH_RUNTIME_ADAPTATION=sm_120`, nothing reordered, nothing
rewritten:

```
WORKBENCH_CLVC_PYTHON=/home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python
WORKBENCH_TGCIR_PYTHON=/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python
```

Both name environments that already existed and were measured in 85 §4 / 87 §7 / 88 §7
(`clvc39-la`, `tgcir39-la`, each `torch 2.7.1+cu128` / `torchvision 0.22.1+cu128` / Python 3.9.25).
Removing the two lines restores the previous behaviour exactly: `setting()` returns `None` for both,
`resolve_model_interpreter()` falls back to the provisioning assignment (none exists for these two,
so it returns `None`), and every adapter refuses at `no interpreter environment resolved`.

Verification, verbatim:

```
CLVC setting: /home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python
TGCIR setting: /home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python
clvc resolve: /home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python
tgcir resolve: /home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python
```

Both names are already in `operator_config._KEYS` (`WORKBENCH_CLVC_PYTHON`,
`WORKBENCH_TGCIR_PYTHON`), so the file parser accepts them without any code change.

## 2. The two registry blocks, verbatim from `models.yaml`

`clvc_net` — under `environment:`, after `packages: [torchvision]`, `models.yaml:937-948`:

```yaml
      # Measured minimal block (`raw/verify/85-composition-dossier.md` §11.5, re-probed in
      # `raw/verify/89-adapted-runtime-and-gpu-capability.md`): these five keys take the advisory
      # tier from BLOCKED/PYTHON_VERSION_MISMATCH to GPU_VERIFICATION_PENDING. `python: '3.9'` is
      # the load-bearing one — without it the historical 3.7.6 pin survives and blocks. No
      # `packages`: adding `numpy<2` changed nothing in the measured tier, and an unneeded
      # constraint is a future false blocker.
      adapted_runtime:
        target_compute_capability: sm_120
        python: '3.9'
        pytorch: '==2.7.1'
        torchvision: '==0.22.1'
        cuda: '12.8'
```

`tgcir` — under `environment:`, after the `packages:` list, `models.yaml:1314-1326`:

```yaml
      # Measured, `raw/verify/89-adapted-runtime-and-gpu-capability.md`: the same five keys take the
      # advisory tier from BLOCKED/DEPENDENCY_VERSION_MISMATCH (torch ==1.7.0) to
      # GPU_VERIFICATION_PENDING. No `packages` list here on purpose, twice over: (1) it is not
      # needed for the tier — measured identical with and without it; (2) an adapted `packages`
      # list REPLACES the historical one in `resolved_runtime_environment`, so a partial list would
      # drop `numpy`/`pandas`/`Pillow` (and the `clip` git source) from the installer's contract.
      # Omitting it keeps the historical list and overrides only the interpreter/pin quartet.
      adapted_runtime:
        target_compute_capability: sm_120
        python: '3.9'
        pytorch: '==2.7.1'
        torchvision: '==0.22.1'
        cuda: '12.8'
```

Shape, indentation (6 spaces for `adapted_runtime:`, a child of `environment:` at 4) and key order
are the `csmcir` / `limn` shape from 85 §11.1. `WORKBENCH_RUNTIME_ADAPTATION=sm_120` is already set
in `workbench.env:20`, so `runtime_adaptation()` returns the block for both rows and
`report.runtime_adaptation == "sm_120"` (measured, §3).

### 2.1 The TG-CIR package list: measured, not needed, and actively harmful to keep

The TG-CIR worker reported that its block needed `packages: [clip==1.0, numpy==1.22.3, pandas==1.4.2,
Pillow]` + `pip_index_url` "because the historical contract declares them and omitting them leaves
the tier blocked". **That is not what the verifier does.** Measured, deep-copied model dict,
`verify_environment(probe=True, advisory=True)`:

| `tgcir` block | tier | status |
|---|---|---|
| five keys only (no `packages`, no `pip_index_url`) | `GPU_VERIFICATION_PENDING` | `GPU_VERIFICATION_PENDING` |
| + `packages: [clip==1.0, numpy==1.22.3, pandas==1.4.2, Pillow]` + `pip_index_url` | `GPU_VERIFICATION_PENDING` | `GPU_VERIFICATION_PENDING` |
| + those packages, **no** `pip_index_url` | `GPU_VERIFICATION_PENDING` | `GPU_VERIFICATION_PENDING` |
| + `packages: ['clip @ git+https://github.com/openai/CLIP.git@a1d0717…']` | `BLOCKED` | `DEPENDENCY_MISSING` |
| + `pillow==11.3.0` (lowercase, pinned) | `GPU_VERIFICATION_PENDING` | `GPU_VERIFICATION_PENDING` |
| + `Pillow==11.3.0` (capital, pinned) | `GPU_VERIFICATION_PENDING` | `GPU_VERIFICATION_PENDING` |

With the packages list **and** without it the tier is identical, so the list is omitted — an
unnecessary constraint is a future false blocker on a differently-provisioned host.

Two further reasons the list would be worse than useless here, both measured:

* `resolved_runtime_environment` MERGES the adapted list into the historical one, replacing only the
  distributions it re-declares (the `_package_key` filter at `runtime.py:1073-1077`). Listing
  `clip==1.0` therefore makes the installer's contract carry **both** the git source and the version
  pin:
  `["clip @ git+…@a1d0717…", "clip==1.0", "numpy==1.22.3", "pandas==1.4.2", "Pillow"]`.
  Omitting the list keeps exactly the historical four entries.
* `verify_environment` checks ONLY `_adapted_packages(...)` when an adaptation applies
  (`runtime.py:1687-1688`) — the historical `packages` list is not consulted at all. Measured proof:
  replacing `tgcir`'s historical `packages` with a deliberately bogus
  `definitely-not-installed-anywhere==9.9` still reports `GPU_VERIFICATION_PENDING`, and
  `_adapted_packages` returned only `{'torch': '==2.7.1', 'torchvision': '==0.22.1'}`.
  So the worker's premise (the historical pins must be re-declared in the block) is false, and its
  consequence (they must be) is a stricter contract than the four pins require.

The clip trap reproduces: a VCS declaration in an adapted `packages` list is split at the first
operator character, so `clip @ git+…` is looked up under the literal distribution name
`clip @ git+https:…` and reports `DEPENDENCY_MISSING` (measured above). The Pillow trap does
**not** reproduce through this verifier: `Pillow`, `Pillow==11.3.0` and `pillow==11.3.0` all reach
`GPU_VERIFICATION_PENDING` here (`_package_key` and `normalize_distribution_name` both fold case),
so 88's `Pillow==11.3.0: null` reading came from its own harness rather than from
`verify_environment`. I cannot observe what that harness did, so the disagreement is recorded rather
than adjudicated. Either way the list is not needed for the tier.

For `clvc_net`, the CLVC worker's `packages: [numpy<2]` was measured too: identical tier with and
without (`GPU_VERIFICATION_PENDING`), so it is omitted. The worker's real need — the per-category
loader constructing under numpy 1.26.4 — is a property of the environment it already has, not of the
declared contract, and 87 §7 notes `numpy<2` there is a guard rather than an ABI-critical
requirement. Declaring a guard the verifier does not consume would be decoration; the environment
inventory already reports `numpy 1.26.4`.

## 3. Before / after `verify_environment(probe=True, advisory=True)`

Real registry rows, no in-memory mutation, operator `workbench.env` active. "Before" = the two
interpreter pins added, `adapted_runtime` absent.

**Before**

| model | `runtime_adaptation` | tier | status | reason |
|---|---|---|---|---|
| `clvc_net` | `None` | `BLOCKED` | `PYTHON_VERSION_MISMATCH` | expected Python 3.7.6, found 3.9.25 |
| `tgcir` | `None` | `BLOCKED` | `DEPENDENCY_VERSION_MISMATCH` | torch: expected ==1.7.0, found 2.7.1+cu128 |

Both already reported `interpreter=/…/envs/{clvc39-la,tgcir39-la}/bin/python` and
`python_version=3.9.25` before the change — the interpreter pins and the registry pins are two
independent facts, which is why the tier did not move until the block was declared.

**After**

| model | `runtime_adaptation` | tier | status | reason |
|---|---|---|---|---|
| `clvc_net` | `sm_120` | `GPU_VERIFICATION_PENDING` | `GPU_VERIFICATION_PENDING` | CUDA capability must be verified on the GPU host with `verify_gpu_runtime()` before this model can be launched |
| `tgcir` | `sm_120` | `GPU_VERIFICATION_PENDING` | `GPU_VERIFICATION_PENDING` | (same) |

Both are `READY_TIERS` members (`runtime.py:1867`), i.e. the interpreter gate passes; only
`verify_gpu_runtime()` can promote them to `GPU_RUNTIME_READY`. Interpreter and Python version
unchanged (`3.9.25`), `runtime_adaptation: sm_120` recorded on the report.

Resolved contract each row now verifies and installs against (`resolved_runtime_environment`):

```
clvc_net: {"python": "3.9", "pytorch": "==2.7.1", "torchvision": "==0.22.1", "cuda": "12.8",
           "packages": ["torchvision"]}
tgcir   : {"python": "3.9", "pytorch": "==2.7.1", "torchvision": "==0.22.1", "cuda": "12.8",
           "packages": ["clip @ git+https://github.com/openai/CLIP.git@a1d0717…",
                        "numpy==1.22.3", "pandas==1.4.2", "Pillow"]}
```

Installed versions the probe read from disk for both: `torch 2.7.1+cu128`, `torchvision 0.22.1+cu128`
(`==2.7.1` / `==0.22.1` match a local-version build; `SpecifierSet('==2.7.1').contains(
Version('2.7.1+cu128'))` is `True`). `clvc_net`: `numpy 1.26.4`, `pillow 11.3.0`. `tgcir`:
`clip 1.0`, `numpy 1.22.3`, `pandas 1.4.2`, `pillow 11.3.0`.

## 4. GPU capability — recorded, not refused

Run under `workbench/scripts/gpu_job.sh` (`MAX_WAIT=3600`), one job, no other compute app:

```
nvidia-smi before   NVIDIA GeForce RTX 5070 Ti, 1119 MiB, 16303 MiB
                    compute apps: (none)
gpu_job launch      GPU-9bc596d9-19e9-6f43-dc2c-a80da924263c, 1130 MiB, 16303 MiB
nvidia-smi after    NVIDIA GeForce RTX 5070 Ti,  993 MiB, 16303 MiB
                    compute apps: pid,process_name,used_memory — header only, nothing running
```

`verify_gpu_runtime(model, config, authorized=True, require_cuda=True)` returned for **both**
models, payload keys (26):
`allocated_tensor, capability_verified, compute_capability, computed, cuda_available,
dependency_fingerprint, device_count, device_name, error, expires_on, host_identity, interpreter,
interpreter_fingerprint, probe, probe_source, probed_at, revalidation, runtime_adaptation,
torch_arch_list, torch_core_loadable, torch_core_origin, torch_cuda_version, torch_file_digest,
torch_has_compiled_core, torch_module_file, torch_version`.

| field | `clvc_net` | `tgcir` |
|---|---|---|
| `capability_verified` | `true` | `true` |
| `cuda_available` | `true` | `true` |
| `device_count` | `1` | `1` |
| `device_name` | `NVIDIA GeForce RTX 5070 Ti` | `NVIDIA GeForce RTX 5070 Ti` |
| `compute_capability` | `12.0` | `12.0` |
| `allocated_tensor` | `true` | `true` |
| `torch_version` | `2.7.1+cu128` | `2.7.1+cu128` |
| `torch_cuda_version` | `12.8` | `12.8` |
| `torch_arch_list` | `sm_75, sm_80, sm_86, sm_90, sm_100, sm_120, compute_120` | (same) |
| `torch_core_loadable` / `torch_has_compiled_core` | `true` / `true` | `true` / `true` |
| `runtime_adaptation` | `sm_120` | `sm_120` |
| `probe` / `probe_source` | `gpu_capability` / `interpreter` | `gpu_capability` / `interpreter` |
| `interpreter` | `…/envs/clvc39-la/bin/python` | `…/envs/tgcir39-la/bin/python` |
| `interpreter_fingerprint` | `6b10cb3f66cb…` | `2f6b22911df6…` |
| `dependency_fingerprint` | `b51b8f44bf4c…` | `b9889e8e4e9d…` |
| `torch_file_digest` | `9801e75447c7…` | `9801e75447c7…` |
| `error` | `null` | `null` |
| `probed_at` | `2026-11-11T17:16:10.409204+00:00` | `2026-11-11T17:17:31.033949+00:00` |

`expires_on`: interpreter change, dependency change, host change, driver change,
`CUDA_VISIBLE_DEVICES` change. `host_identity`: driver 580.173.02, x86_64, node `heheboiz-nhut-pc`,
`CUDA_VISIBLE_DEVICES` / `NVIDIA_VISIBLE_DEVICES` unset. Both interpreters share the same
`torch_file_digest` (same cu128 wheel) but differ in dependency fingerprint, which is why the record
is per-interpreter rather than per-model.

Tier after the probe, and the launch gate itself:

| model | `verify_environment(probe=True)` tier | in `LAUNCH_READY_TIERS` | `require_launch_ready(require_cuda=True)` |
|---|---|---|---|
| `clvc_net` | `GPU_RUNTIME_READY` / `GPU_RUNTIME_READY` | `True` | returns `…/envs/clvc39-la/bin/python` |
| `tgcir` | `GPU_RUNTIME_READY` / `GPU_RUNTIME_READY` | `True` | returns `…/envs/tgcir39-la/bin/python` |

So `GPU_VERIFICATION_PENDING` → `GPU_RUNTIME_READY` for both. No refusal to report: the probe
succeeded, nothing was forced, and the sizes/arch lists above are the probe's own recorded
observation. The record lives under `workbench/artifacts` (the workspace `resolve_config()` returns),
which is the same workspace the gate reads; a record written to a different workspace would have
been missed (`tests/test_gpu_capability_workspace.py` is that regression).

## 5. `command_status` — left as `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, and why it is still true

Consumers (`grep -rn "command_status" workbench/ --include=*.py --include=*.ts --include=*.tsx` — no
TypeScript consumer exists, so nothing in `frontend/` reads it):

| consumer | what it does with the value |
|---|---|
| `backend/preflight.py:271,290` | `audited` requires the adapter to override `command` **and** `command_status ∈ {None, COMMAND_AUDITED, REPLAY_COMMAND_IMPLEMENTED}`; otherwise appends `command not runnable: <value>` to the preflight blocker list |
| `backend/registry.py:2025-2027` | `adapter_command_ready` in the per-model readiness record; `None` falls back to "adapter overrides `command`" |
| `scripts/doctor.py:164-168` | the `command:<model>` check — `OK` only for the same two values, prints the status as the detail |
| `composition/model_tgcir.py:9` (docstring) | states that the value is exact and that the adapter's evaluation path is **workbench-added** |
| `backend/autopilot.py:608`, `backend/main.py:36`, `scripts/{evaluate_models,pc_readiness,prepare_model}.py` | call `preflight_blockers` / `doctor`; they consume the blocker, not the raw field |

**Decision: unchanged.** The value is a statement about the *pinned upstream entrypoint*, and it is
still true: `third_party/TGCIR/test.py` is 162 lines, has zero occurrences of `argparse`,
`__main__` or `sys.argv`, and its only function is `test(params, model, testset, category)`
(`test.py:6`). `third_party/CLVCNet/test.py` is the same shape (148 lines, zero CLI hits). What the
workbench added is `replay/tgcir.py::main` — an `ArgumentParser` with `--source-root`,
`--dataset-root`, `--checkpoint-root`, `--category`, `--batch-size`, `--device`, `--output`,
`--preflight-only`, whose payload names its own provenance as `"evaluation_path":
"WORKBENCH_ADDED_EVAL_PATH"`. That is a replay path, not the source entrypoint, and it is exactly how
`limn` is already recorded (`REPLAY_COMMAND_IMPLEMENTED`, `replay_type:
SOURCE_FAITHFUL_WORKBENCH_REPLAY`) — so the vocabulary for "the workbench supplied the entrypoint"
already exists; it is not claimed here, because `tgcir`'s `replay_status` is still `PREFLIGHT_ONLY`
and its preflight measurement currently refuses on `external runtime asset provenance unresolved:
OpenAI CLIP ViT-B/16 weights/cache`. Raising `command_status` while `replay_status` says
`PREFLIGHT_ONLY` would make the two fields disagree. Measured blockers for the two rows right now:

```
tgcir    command_status= COMMAND_BLOCKED_SOURCE_ENTRYPOINT | replay_status= PREFLIGHT_ONLY
   blocker: external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache
   blocker: command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
clvc_net command_status= COMMAND_BLOCKED_SOURCE_ENTRYPOINT | replay_status= BLOCKED_NO_SOURCE_REPLAY
   blocker: torchvision ResNet-50 ImageNet weights missing: ~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth
   blocker: external runtime asset file missing: torchvision ResNet-50 ImageNet weights (…)
   blocker: command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
```

One line, as requested: *the pinned upstream `test.py` still has no CLI, and the workbench's replay
path is not the upstream entrypoint, so the value remains a true statement about upstream.*

## 6. Test snapshot — what actually happens, and the honest minimal change

Before editing, `test_status_refuses_rather_than_claiming_readiness` passed for both ids: under the
suite's `isolate_operator_state` fixture, `status()` sees no operator file and no staged checkpoint,
so it refuses for a *provisioning* reason, not for a science reason:

```
clvc_net ready= False | reason= checkpoint bundle incomplete at …/checkpoints/../../checkpoints/clvc;
                              no interpreter environment resolved for WORKBENCH_CLVC_PYTHON
tgcir    ready= False | reason= checkpoint not staged at …/Fashioniq.pt; no interpreter environment
                              resolved for WORKBENCH_TGCIR_PYTHON; environment gate would refuse the
                              run [UNCONFIGURED/UNCONFIGURED]: WORKBENCH_TGCIR_PYTHON not set
```

Against the operator's real state — pins in `workbench.env`, staged checkpoints at
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads`, adaptation
selected — both are now **ready**:

```
clvc_net ready= True | reason= None | per_mode: all three conditions (True, None)
tgcir    ready= True | reason= None | per_mode: all three conditions (True, None)
```

and with `WORKBENCH_RUNTIME_ADAPTATION` removed, both go back to `False` for the old reasons
(`PYTHON_VERSION_MISMATCH` / `DEPENDENCY_VERSION_MISMATCH`) — so the readiness is caused by the block
this task declares, not by the checkpoint staging alone.

The edit partitions the parametrization instead of deleting or weakening anything:

* `test_status_refuses_rather_than_claiming_readiness` now runs over
  `set(NEW_ADAPTERS) - READY_WITH_EVIDENCE` = the five that are still not runnable
  (`airknow`, `clip4cir_rn50x4_fullft`, `combiner_rn50x4_noft`, `dcnet`, `pair`), keeping
  `assert status.ready is False` verbatim.
* `READY_WITH_EVIDENCE = {"clvc_net", "tgcir"}` names the evidence (this file; 87 §5 / 88 §3 for
  parity) and the measured tier (`GPU_RUNTIME_READY`).
* `test_status_reports_the_measured_readiness` asserts `status.ready is True` for those two, reading
  the operator's `workbench.env` explicitly (the fixture hides operator state by design) and
  **skipping** — not passing, not weakening — when the provisioning is absent, because a fresh clone
  has no staged checkpoint and no pins and there is then nothing honest to assert. On this host both
  parametrizations PASS, not skip:

```
…::test_status_reports_the_measured_readiness[clvc_net] PASSED
…::test_status_reports_the_measured_readiness[tgcir]    PASSED
2 passed in 13.96s
```

No other test in that file was touched; every other assertion is byte-identical.

Pytest, named files only (`test_circus_env_gate.py` does not exist under that name; the file is
`test_circus_env_gate_omp.py`):

```
workbench/envs/ctl312h/bin/python -m pytest \
  workbench/tests/test_composition_adapters_missing.py \
  workbench/tests/test_circus_env_gate_omp.py \
  workbench/tests/test_blocked_models_verification.py -q
82 passed, 60 warnings in 22.41s
```

`test_blocked_models_verification.py` needed **no** edit: it asserts staged-digest equality, the
strict-load repair loop and the PAIR byte-level absence — none of it pins a refusal string, so
nothing in it became factually wrong. The 60 warnings are pre-existing `SyntaxWarning: invalid
escape sequence` notices from that file's own literals, present before this change.

## 7. Exact `git diff` of the registry rows

```
$ git diff workbench/registry/models.yaml
@@ -934,6 +934,18 @@ models:
+      # Measured minimal block (`raw/verify/85-composition-dossier.md` §11.5, re-probed in
+      # `raw/verify/89-adapted-runtime-and-gpu-capability.md`): these five keys take the advisory
+      # tier from BLOCKED/PYTHON_VERSION_MISMATCH to GPU_VERIFICATION_PENDING. `python: '3.9'` is
+      # the load-bearing one — without it the historical 3.7.6 pin survives and blocks. No
+      # `packages`: adding `numpy<2` changed nothing in the measured tier, and an unneeded
+      # constraint is a future false blocker.
+      adapted_runtime:
+        target_compute_capability: sm_120
+        python: '3.9'
+        pytorch: '==2.7.1'
+        torchvision: '==0.22.1'
+        cuda: '12.8'
@@ -1299,6 +1311,19 @@ models:
+      # Measured, `raw/verify/89-adapted-runtime-and-gpu-capability.md`: the same five keys take the
+      # advisory tier from BLOCKED/DEPENDENCY_VERSION_MISMATCH (torch ==1.7.0) to
+      # GPU_VERIFICATION_PENDING. No `packages` list here on purpose, twice over: (1) it is not
+      # needed for the tier — measured identical with and without it; (2) an adapted `packages`
+      # list REPLACES the historical one in `resolved_runtime_environment`, so a partial list would
+      # drop `numpy`/`pandas`/`Pillow` (and the `clip` git source) from the installer's contract.
+      # Omitting it keeps the historical list and overrides only the interpreter/pin quartet.
+      adapted_runtime:
+        target_compute_capability: sm_120
+        python: '3.9'
+        pytorch: '==2.7.1'
+        torchvision: '==0.22.1'
+        cuda: '12.8'
 +25 -0
```

No `command_status`, `reproduction_status`, `replay_status` or any other key on either row was
changed.

## Còn chặn

1. **No metric was produced by this task.** Not one score, rank or recall number. What this makes
   possible is a *launchable* run: interpreter verified, pins resolved, GPU capability recorded.
   `clvc_net`'s and `tgcir`'s composition numbers remain unmeasured — 87 and 88 measured parity at
   5 queries / 20 images per category, and the native campaign numbers
   (`raw/tgcir_run3cat_gpu.json`: macro R@10 49.7208 / R@50 71.1023 / mean 60.4115) are untouched.
2. **`command_status` stays `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`** on both rows (§5). Upstream still
   has no CLI; the workbench replay path is not it. Promotion to `REPLAY_COMMAND_IMPLEMENTED` would
   require `replay_status` to move first, which is not this task's file.
3. **`tgcir`'s preflight still refuses** on `external runtime asset provenance unresolved: OpenAI
   CLIP ViT-B/16 weights/cache`, and `clvc_net`'s on the missing `resnet50-19c8e357.pth`. Staging those
   assets is a provisioning step outside this task; the GPU record makes the *gate* pass, not the
   preflight.
4. **GPU capability is a per-host record with an expiry** (§4 `expires_on`). Changing the driver,
   the interpreter, the dependency set or `CUDA_VISIBLE_DEVICES` invalidates it and the tier drops
   back to `GPU_VERIFICATION_PENDING` until `verify_gpu_runtime()` is run again. Nothing here
   re-probes automatically.
5. **The registry pins and the `workbench.env` pins are independent.** Removing either the two
   `adapted_runtime` blocks or the two `WORKBENCH_*_PYTHON` lines returns both models to
   `BLOCKED` — measured in both directions — so the launchable state depends on gitignored operator
   state that a fresh clone does not have. The test in §6 says that out loud instead of hiding it.
6. **`image_only_empty_text` for `clvc_net` is still refused at the adapter level** (87 §5c): the
   readiness claim covers interpreter/GPU/checkpoint, not that condition being representable.
7. **`numpy<2` is not declared** for `clvc_net` (§2.1). If a future cu128 image ships numpy 2.x the
   per-category loader may fail at construction; the verifier would not catch it because it never
   reads the historical `packages` list under an adaptation. That is a real, named ceiling of the
   minimal block, not an oversight.
