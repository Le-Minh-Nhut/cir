# PC transfer checklist

What to copy, what to rebuild, and what only exists on the GPU host.

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
each group at its own prefix from those.

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

```bash
git clone <repo> && cd cir
git checkout <the frozen commit>

umask 077 && mkdir -p ~/.config/cir
python -c "import secrets,pathlib;pathlib.Path.home().joinpath('.config/cir/workbench_ledger_key').write_text(secrets.token_hex(32))"
export WORKBENCH_LEDGER_KEY="$(cat ~/.config/cir/workbench_ledger_key)"

cp workbench/configs/pc_autopilot.pilot_csmcir.yaml /path/outside/repo/pc.yaml
$EDITOR /path/outside/repo/pc.yaml     # set paths.dataset_root and the artifact roots

./workbench/run.sh autopilot --config /path/outside/repo/pc.yaml --apply --resume
```

Autopilot then acquires the pinned sources, plans and provision the environments, prepares the
dataset-derived artifacts, obtains the obtainable assets, verifies checkpoints and the GPU, and runs
every eligible evaluator. Rerun the same command with `--resume` after any interruption.

## Reading the outcome

| File | What it tells you |
|---|---|
| `<artifacts>/reports/autopilot_summary.md` | per-model status, blockers, and the category Recall table |
| `<artifacts>/reports/blockers.json` | machine-readable blocker detail, with the reason for each |
| `<artifacts>/reports/run_manifest.json` | config digest, outcome, pipeline exit code |
| `<artifacts>/environment_assignment.json` | the interpreter each model will execute with |
| `<artifacts>/pipeline_state.json` | stage state, including why a stage was reused |

Exit `0` is full success, `2` is partial (some models completed, others blocked or failed), `1` is a
workflow failure. A blocked model is not an engineering failure: it is a missing external resource,
named exactly.
