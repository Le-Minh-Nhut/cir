# PC transfer checklist

What to copy, what to rebuild, and what only exists on the GPU host.

For the first experiment itself, see [FIRST_GPU_EXPERIMENT.md](FIRST_GPU_EXPERIMENT.md): the one
command to run, why the pilot is LIMN rather than CSMCIR, and the two facts to state when reporting
its numbers.

## Copy these (portable)

| Item | Why it is portable |
|---|---|
| the repository at the frozen commit | all manifests, registry and recipes are relative-path based |
| `workbench/configs/*.yaml` (your edited copy) | paths are per-machine by design; edit the `paths` block |
| your canonical FashionIQ dataset | it is the operator's input and is read, never written |
| `workbench/registry/` (including `audits/`) | the model, checkpoint, auxiliary-asset and preparation registries |
| previously acquired upstream checkouts under `workbench/third_party/` | they are git checkouts at pinned SHAs; `sync_upstreams.py --verify-only` confirms each pin |
| downloaded checkpoints under `paths.checkpoints` | verified by size and SHA-256 on arrival, so a copy is checkable |
| downloaded auxiliary assets under the FashionIQ root's `qwen_captions/` | same: verifying the published hash proves the copy |
| `workbench/artifacts/reports/` from any completed run | evidence, not input; keep it for comparison |

## Rebuild these on the PC (do NOT copy)

| Item | Why |
|---|---|
| conda environments under `paths.artifacts/environments/` | a conda prefix embeds absolute paths in its interpreter, every console script shebang and its `conda-meta` records. Copying the directory does not relocate it, and the workbench now refuses an environment whose console scripts point elsewhere. |
| anything under `workbench/artifacts/logs/` | per-run scratch |
| the DuckDB index | rebuilt from the results by the `validation-index` stage |
| GPU capability records under `workbench/artifacts/gpu_capabilities/` | they are bound to the host, device and interpreter that produced them; a record from another machine is not evidence about this one |

The portable form of an environment is its *specification*: the group assignment
(`<artifacts>/environment_assignment.json`), the resolved constraints in
(`<artifacts>/environment_plan.json`) and the pinned source revision. On the PC, Autopilot rebuilds
each group directly at its own final prefix under a process-level advisory lock (`.provision-<environment_id>.lock`).

## PC-only checks the laptop cannot perform

- real CUDA availability on the target driver;
- a real `torch` CUDA runtime inside the provisioned environment;
- a real checkpoint load and a real official evaluation against the canonical dataset;
- Recall values, which no laptop run produces.

The laptop proves the *workflow*: stages, ordering, authorization, validation, resume, reporting.
It never claims numeric parity.

## External author resources (manual action, no automation possible)

These have no verified acquisition source, so they are reported as blockers rather than fetched.
Each is named with the exact path the evaluator reads.

| Model | Resource | Exact path |
|---|---|---|
| `csmcir` | author CoT captions, no published URL or checksum in the repository or the author Hugging Face repo | `<CSMCIR checkout>/COT_ours2/fashioniq/{dress,shirt,toptee}_cot_val.json` |
| `encoder` | `datasets1.py`, absent from the pinned checkout | `<ENCODER checkout>/datasets1.py` |
| `pair` | `MHTransformer_rev.py`, absent from the pinned checkout | `<PAIR checkout>/MHTransformer_rev.py` |
| `habit`, `intent`, `conesep`, `airknow`, `hint` | correction dictionaries, not shipped and not published | `<dataset>/captions/correction_dict_{dress,shirt,toptee}.json` |
| `dcnet` | author GloVe validation PKLs (regeneration is nondeterministic) and the checkpoint run directory | `<dataset>/captions/cap.<cat>.glove.val.pkl`, author Drive `logdir/fashioniq_dcnet` |
| `clvc_net`, `sprc`, `tgcir`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` | author checkpoint archives with no per-file mapping or checksum | author Drive folders |

Full per-model detail, with the source line that reads each path, is in
[AUXILIARY_ASSET_AUDIT.md](AUXILIARY_ASSET_AUDIT.md).

## First run on the PC

Start with the read-only readiness check. It writes nothing, downloads nothing and runs no evaluator;
it tells you which models can proceed and what, precisely, stops the rest:

```bash
python workbench/scripts/pc_readiness.py                 # human-readable
python workbench/scripts/pc_readiness.py --json          # machine-readable
python workbench/scripts/pc_readiness.py --model limn    # one model
python workbench/scripts/pc_readiness.py --config PATH   # use an autopilot profile's paths
```

Each model is classified as one of `READY_FOR_FULL_EVALUATION`, `READY_FOR_GPU_SMOKE`,
`READY_WITH_DECLARED_VARIANT`, `BLOCKED_EXTERNAL_ASSET`, `BLOCKED_DATASET`, `BLOCKED_SOURCE`,
`BLOCKED_IMPLEMENTATION`, `BLOCKED_ENVIRONMENT` or `PC_GPU_VALIDATION_PENDING`. The last one means
every reported prerequisite is satisfied but the model's own interpreter has not recorded a verified
GPU capability (a model that declares a CUDA requirement stays pending until `verify_gpu_runtime()`
has run for it on the GPU host); the host's `nvidia-smi`/torch answer is reported separately and
never substituted for the model's own verification. Each registered checkpoint also carries its own
blocker list (presence, hash, bundle, mapping), so "check individual checkpoint" stays possible.

```bash
git clone <repo> && cd cir
git checkout <the frozen commit>

umask 077 && mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/cir"
python -c "import os,pathlib,secrets; (pathlib.Path(os.environ.get('XDG_CONFIG_HOME') or pathlib.Path.home()/'.config')/'cir'/'workbench_ledger_key').write_text(secrets.token_hex(32))"
export WORKBENCH_LEDGER_KEY="$(cat "${XDG_CONFIG_HOME:-$HOME/.config}/cir/workbench_ledger_key")"

cp workbench/configs/pc_autopilot.all_available.yaml /path/outside/repo/pc.yaml
$EDITOR /path/outside/repo/pc.yaml
```

Every entry under `paths` that is not absolute is resolved **relative to the config file's own
directory**, so a config copied out of the repository must use absolute paths for all six keys:
`registry`, `dataset_root`, `artifacts`, `checkpoints`, `results`, `third_party`. Point them at this
machine's real locations. `load_config` refuses a config whose `registry` does not exist rather than
guessing, so a wrong relative path fails immediately and names the key.

Then autopilot acquires the pinned sources, plans and provisions the environments (created directly at their
final prefix under a lock), runs the preparation stage for any model whose contract declares generable artifacts,
obtains the obtainable assets, verifies checkpoints and the GPU, and runs every eligible evaluator. Rerun the same
command with `--resume` after any interruption:

```bash
./workbench/run.sh autopilot --config /path/outside/repo/pc.yaml --apply --resume
```

### Permissions decide how much autopilot may do

Start from `pc_autopilot.pilot_limn.yaml` for a focused single-model first run, or
`pc_autopilot.all_available.yaml` to run every eligible model with flagged conditions explicit.

`pc_autopilot.pilot_limn.yaml` is the first-GPU-pilot profile: LIMN base iteration 0, with
`allow_env_install: true` so the pilot can cold-start and provision its conda environment directly on
the PC, and `allow_network`, `allow_large_downloads`, `allow_preparation`, `allow_gpu_eval` and
`allow_index_write` all enabled. It sets `checkpoint_policy: include_flagged_conditions` because the
author publishes no training-condition label for these weights -- so report the run as "training
condition not published by the author", not as a clean checkpoint.

`pc_autopilot.pilot_csmcir.yaml` is the CSMCIR variant profile. CSMCIR cannot run from its pinned
source as published: the module registry it imports executes 98 unguarded imports for files the author
never committed, before the line that registers the class it needs, so the entrypoint fails during
import. The registry therefore declares **one approved patch**
(`workbench/patches/csmcir_models_init_compat.patch`) with its base and derived file digests, and
preflight accepts a checkout only while it holds exactly that delta:

```bash
# Build the derived checkout the overlay describes. The pinned tree stays untouched.
git -C workbench/third_party/CSMCIR checkout --detach 774f94e2076ff17ea91703a6239d2a08f0e1a44e
cp -a workbench/third_party/CSMCIR /path/to/variant-root/CSMCIR
patch -p1 --directory /path/to/variant-root/CSMCIR -i workbench/patches/csmcir_models_init_compat.patch
```

Point the config's `third_party` at the **parent directory** holding that corrected checkout, not at
the checkout itself: the registry declares `source_dir: CSMCIR`, and every resolver computes
`<third_party>/<source_dir>`. Pointing `third_party` directly at the checkout makes the workbench
look for `<checkout>/CSMCIR`, which does not exist, and the run blocks on a missing source. The run
is then a **declared compatibility variant**, never official parity: it executed code the authors
did not publish, and `pc_readiness.py` reports it as `READY_WITH_DECLARED_VARIANT` for that reason.
Any other edit inside the checkout still blocks, and an unpatched checkout reports the original
source defect.

CSMCIR also needs `COT_ours2/fashioniq` captions that the author never published. The published Qwen
captions are source-compatible (same names, exact gallery-id coverage, the same `Final_Caption` field
every executing consumer reads), so staging them at that path is available as an opt-in:

```bash
python workbench/scripts/pipeline.py real --model csmcir \
    --download-auxiliary-assets --stage-caption-aliases
```

The alias is recorded in the download manifest and reported in preflight as a warning, so the run is
labelled a variant. Byte identity with the authors' own COT files remains unknown; see
`workbench/docs/COT_OURS2_AUDIT.md`.

| Permission | What it authorizes |
|---|---|
| `allow_env_install` | creating the conda environments |
| `allow_network` | reaching an index or a download source |
| `allow_large_downloads` | fetching checkpoints |
| `allow_preparation` | writing dataset-derived artifacts (resized images, staged dictionaries, dataset links) |
| `allow_gpu_eval` | running the GPU capability probe |
| `allow_index_write` | rebuilding the results index |

Every stage that needs a permission it was not granted reports `BLOCKED_AUTHORIZATION_REQUIRED` and
names the missing capability, so a partial run tells you exactly what to add.

## Reading the outcome

Every evaluation report now carries a `benchmark_gate` block, and the reference it was judged against
under `benchmark_reference`. The verdict is computed by `workbench/backend/paper_gate.py` from the
committed rows in `workbench/registry/paper_benchmarks/papers.yaml`; no model is consulted at runtime.

```bash
# The verdict for one run:
python -c "import json,sys; r=json.load(open(sys.argv[1])); print(json.dumps(r['benchmark_gate'], indent=2))" \
    <artifacts>/results/<protocol>/<model>/<checkpoint>.json
```

| Verdict | Meaning |
|---|---|
| `PASS_WITHIN_5_PERCENT` | every comparable metric is within 5 percent relative error of the published row, and the checkpoint-to-paper mapping was verified (`VERIFIED_METADATA`) — a certified pass |
| `PASS_WITHIN_5_PERCENT_VARIANT` | the same, but a declared stand-in was used (staged caption alias, approved source patch) |
| `PASS_METRICS_ONLY_MAPPING_UNVERIFIED` | the numbers are within tolerance, but the checkpoint-to-paper mapping was never verified, or the run compared none of the paper's categories — **not a certified pass** (`passed: false`); the numbers remain inspectable |
| `PASS_WITH_UNCHECKED_CATEGORIES` | the run left paper-published categories unobserved, so they were never compared — **not a certified pass** (`passed: false`); the untouched categories are named in `detail` |
| `ALERT_METRIC_DEVIATION` | at least one comparable metric exceeds 5 percent; the block names which and by how much |
| `BLOCKED_PROTOCOL_MISMATCH` / `BLOCKED_CHECKPOINT_MISMATCH` | the reference and the run do not describe one experiment |
| `METRIC_REFERENCE_UNVERIFIED` | no committed row describes this run, or the row is not verified |
| `NOT_RUN` / `FAILED_EVALUATION` / `FAILED_OUTPUT_VALIDATION` | nothing was compared, and why |

A pass is never inferred from the registry's transcribed `reported_*` triplet: that field remains in
the report as `paper_sanity_metrics`, a sanity reference only.

| File | What it tells you |
|---|---|
| `<artifacts>/reports/autopilot_summary.md` | per-model status, blockers, and the category Recall table |
| `<artifacts>/reports/blockers.json` | machine-readable blocker detail, with the reason for each |
| `<artifacts>/reports/run_manifest.json` | config digest, outcome, pipeline exit code |
| `<artifacts>/environment_assignment.json` | the interpreter each model will execute with |
| `<artifacts>/autopilot_pipeline_state.json` | stage state, including why a stage was reused |

Exit `0` is full success, `2` is partial (some models completed, others blocked or failed), `1` is a
workflow failure. A blocked model is not an engineering failure: it is a missing external resource,
named exactly.
