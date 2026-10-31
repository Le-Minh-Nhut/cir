# A37 PC HANDOFF — copy-pasteable sequence

One-line note: **no sm_120 evidence exists yet.** The target GPU has never been present in this
repository's history — zero capability records report `compute_capability 12.0` — so every step below
that touches the target hardware is UNEXECUTED here and the handoff remains `TARGET_GPU_PENDING`.

Run everything from the repository root unless a step says otherwise.

---

## Six operator paths

Set these in a private `~/cir_pc.yaml` (copy of
`workbench/configs/pc_autopilot.pc_cohort.yaml`):

| YAML `paths.*` | standalone setting |
|---|---|
| `paths.registry` | `WORKBENCH_REGISTRY_PATH` (process environment only — rejected by `workbench.env`) |
| `paths.dataset_root` | `FASHIONIQ_ROOT` |
| `paths.artifacts` | `WORKBENCH_ARTIFACTS_ROOT` |
| `paths.checkpoints` | `WORKBENCH_CHECKPOINT_ROOT` |
| `paths.results` | `WORKBENCH_RESULTS_ROOT` |
| `paths.third_party` | `WORKBENCH_THIRD_PARTY_ROOT` |

---

## 1. Environment setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r workbench/backend/requirements.txt .
```

Keep per-model environments SEPARATE; never put a model environment on PATH.

```bash
# Private cohort profile OUTSIDE the repository.
cp workbench/configs/pc_autopilot.pc_cohort.yaml ~/cir_pc.yaml
$EDITOR ~/cir_pc.yaml          # all six absolute paths, finite budgets, explicit permissions

# Standalone roots mirror.
cp workbench/config/workbench.env.example workbench/config/workbench.env
$EDITOR workbench/config/workbench.env    # set CIR_REPO_ROOT and the five mapped roots

# Blackwell target: select the declared adaptation before planning or applying.
export WORKBENCH_RUNTIME_ADAPTATION=sm_120

# Root proof — refuse to proceed on any disagreement.
.venv/bin/python workbench/scripts/print_roots.py --config ~/cir_pc.yaml
```

## 2. Dataset and checkpoint preparation

```bash
.venv/bin/python workbench/scripts/doctor.py --scope workbench --json
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --plan
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --prepare-data
```

`--prepare-data` verifies the released files against their recorded hashes and inventories missing
gallery/reference/positive images. It never downloads a missing asset; a missing member is reported
by name and has to be staged first.

Environment provisioning split for the nine active models: **csmcir, conesep, habit, intent, hint and
tme are operator-provisioned**; only **limn, encoder and ptha_mtst** carry `environment_required` and
therefore take `--create --allow-env-install`. For ENCODER:

```bash
.venv/bin/python workbench/scripts/manage_environment.py --model encoder --create --allow-env-install --dry-run
.venv/bin/python workbench/scripts/manage_environment.py --model encoder --create --allow-env-install --allow-network
```

## 3. ENCODER CUDA probe

```bash
./pc.sh --verify-only
```

Then prove the actual runtime, resolving the interpreter from the assignment rather than from a shell
variable a child process cannot export back:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import subprocess

from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import model_by_id
from workbench.backend.runtime import resolve_model_interpreter

config = resolve_config()
interpreter = resolve_model_interpreter(model_by_id("encoder"), config)
if not interpreter or not Path(interpreter).is_file():
    raise SystemExit("ENCODER interpreter is unresolved; inspect roots/assignment/status first")
print("Resolved ENCODER interpreter:", interpreter, flush=True)

code = """import torch, torchvision
print('versions:', torch.__version__, torchvision.__version__, torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
a = torch.randn(512, 512, device='cuda')
b = a @ a
torch.cuda.synchronize()
assert b.isfinite().all().item()
print('CUDA allocation/matmul/synchronization OK')
"""
subprocess.run([interpreter, "-c", code], check=True)
PY
```

The gate that decides this is `workbench/backend/runtime.py:2470-2493`: it compares the declared
`target_compute_capability` against `torch.cuda.get_arch_list()` and, when a device is probed, against
the device's own capability.

## 4. ENCODER smoke

```bash
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder --smoke
```

Name the model explicitly. `--models selected` reads `selected_models` from `~/cir_pc.yaml`, and the
checked-in template ships `all_registered`, which resolves all 17 registry entries including the eight
deferred ones — the run then ends with a named blocker per deferred model instead of the pilot's
results. Replace `selected_models: all_registered` with a deliberate verified list before using
`--models selected`.

Smoke mode limits queries (default 1) and preserves the full released gallery. Schedule one GPU-heavy
model at a time; the runner serializes model runs.

## 5. Resume verification

```bash
# Re-run the unchanged command and confirm completed stages are reused.
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models encoder --smoke
```

Expect `[SKIP] resume: already completed with matching fingerprint and verified outputs`
(`workbench/scripts/pipeline.py:2309`). Two caveats to record:

- Ranks and stage records written under the old stat-based dataset identity are intentionally
  invalidated by the A37 fix; a first post-fix run recomputes them rather than reusing them.
- The fingerprint is taken from source **bytes**, so any dataset change that alters image bytes but not
  file size or mtime now produces a new `run_id` (this is the fixed defect, not a regression).

## 6. Full run

```bash
./workbench/run.sh composition-eval --config ~/cir_pc.yaml --models selected --subsets full,sf,validated --resume
```

Keep this on ONE line. A copy that wraps before `--resume` without a trailing backslash drops the
resume flag and then tries `--resume` as the next command.

Per completed run, read the model row in `<paths.artifacts>/reports/autopilot_results.csv`, take its
exact `run_id`/`model_id`/`checkpoint_id`/`protocol_id`, and verify:

```bash
.venv/bin/python workbench/scripts/verify_run.py \
    --run-id <RUN_ID> --model <MODEL_ID> --checkpoint <CHECKPOINT_ID> --protocol <PROTOCOL>
```

---

## Standing warning

`WORKBENCH_ALLOW_SYNTHETIC_GPU_PROBE` (`workbench/backend/runtime.py:1326`) is a test-only escape
hatch. Records written under it carry `probe_source: 'override'` and every reader refuses them unless
the variable is set again. A synthetic record MUST NEVER be presented as sm_120 proof — the gate to a
real `TARGET_GPU_VERIFIED` verdict is the interpreter-probed record from step 3 on the real device.
