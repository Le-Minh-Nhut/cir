# Final PC readiness -- Release B (GPU host)

Everything below is the software state at the commit named here. Real checkpoint inference, real
FashionIQ evaluation, numerical parity with the papers and real CUDA capability belong to the GPU
host: none of them has been executed on the laptop, and nothing in this document claims otherwise.

## 1. Commit and command

- Branch: `research/cir-failure-workbench`
- Base commit for this document: `5be9fdc8c35d11c5d8fc8d7563a82808a812fef4`
- `main` remains `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333`

One command, after the configuration file exists:

```bash
./workbench/run.sh autopilot \
    --config workbench/configs/pc_autopilot.yaml \
    --apply \
    --resume
```

`--resume` is safe on a first invocation. Use without `--apply` to plan only, `--dry-run` to plan
without side effects, `--status` / `--report` to read saved state. Every other command in this
document is a diagnostic.

## 2. Host prerequisites

| Requirement | Notes |
|---|---|
| Python | the interpreter that runs the workbench (the one this document was produced with works) |
| Package manager | `conda`, `mamba` or `micromamba` on `PATH`. The planner real-solves with it and provisioners use it to create each environment group. |
| CUDA | a driver the target PyTorch build supports. Capability is verified per environment; `CUDA_VISIBLE_DEVICES` selects devices and must stay fixed between verification and evaluation. |
| Disk | checkpoints dominate: see the budget section. Artifacts, checkpoints, results and upstream clones each live under their own configured root. |
| Network | needed only for acquisition. A verified local source/checkpoint/auxiliary asset is reused with no network permission. |

Before the first run, create the ledger signing key **once** and keep it:

```bash
umask 077 && mkdir -p ~/.config/cir
python -c "import secrets,pathlib;pathlib.Path.home().joinpath('.config/cir/workbench_ledger_key').write_text(secrets.token_hex(32))"
```

Then export the *same* key in every shell that runs the workbench:

```bash
export WORKBENCH_LEDGER_KEY="$(cat ~/.config/cir/workbench_ledger_key)"
```

Keep it outside the repo, the config, and the artifacts. It signs completion evidence, and the
evidence chain is only as trustworthy as that key: **do not regenerate it per shell**, because a new
key makes every earlier run's evidence unverifiable. Evaluator and package-manager children do not
receive it. Leaving the variable unset is also safe -- a `0600` key file is created under the
private config directory (`~/.config/cir/`) on first use and reused thereafter -- but a secret store is preferable on a shared
machine. An empty key, or a missing key while a ledger already exists, fails closed.

## 3. Configuration

Copy `workbench/configs/pc_autopilot.example.yaml` and edit. Required keys: `schema_version`,
`selected_models`, `paths`, `budgets`, `permissions`. Optional: `checkpoint_policy` (default
`strict_clean`) and `checkpoint_overrides`.

`paths.dataset_root` is the FashionIQ root itself (the directory containing `captions/`,
`image_splits/`, `images/`), not `images/`. `paths.artifacts` owns run state, reports, the evidence
ledger, completion proofs, GPU capability records and the derived index; keep it on persistent
storage. `paths.third_party` is where pinned upstream checkouts live -- the planner reads each
model's requirements from there, so it must be the same root the sync stage populates.

Two ready-made profiles are provided:

- `workbench/configs/pc_autopilot.pilot_csmcir.yaml` -- one model, the shortest verified path.
- `workbench/configs/pc_autopilot.all_available.yaml` -- every eligible model, with the flagged
  checkpoint policy so the noise-trained cohort is explicit rather than silently excluded.

## 4. Permissions

| Stage | Capabilities |
|---|---|
| source sync (acquisition) | `allow_network`, `allow_preparation` |
| source sync (verified checkout present) | none |
| checkpoint acquisition | `allow_network`, `allow_large_downloads` |
| checkpoint verification (file present) | none |
| auxiliary asset acquisition | `allow_network`, `allow_large_downloads` |
| environment creation | `allow_env_install`, `allow_network` |
| preparation (dataset link creation) | `allow_preparation` |
| preparation (link already valid) | none |
| evaluation | `allow_gpu_eval`, `allow_preparation` |
| index rebuild | `allow_index_write` |

A stage whose declared capability is not granted is refused as `BLOCKED_AUTHORIZATION_REQUIRED`. A
missing permission is never treated as permission to download.

## 5. Budgets

`max_download_bytes` must cover the selected checkpoint variants plus required auxiliary downloads,
with margin. Unknown required sizes block acquisition rather than being assumed.

The three declared LIMN category checkpoints total 5,376,030,471 bytes (~5.01 GiB); the example
budget is 10 GiB. `max_models` must cover the selection. `top_k` sets retrieval depth.

## 6. Environment groups

The planner groups models by their source-backed constraints and provisions **one environment per
group** under the artifacts root, recording four distinct states:

| State | Meaning |
|---|---|
| `CONSTRAINTS_COMPATIBLE` | the declared constraint sets do not conflict |
| `SOLVER_COMPATIBLE` | a real solve succeeded |
| `INSTALLED` | the packages are present |
| `RUNTIME_VERIFIED` | versions (exact and range) and every member's evaluator imports were verified in that interpreter |

Only `RUNTIME_VERIFIED` permits evaluation. Installation is conda-resolvable packages first, then
pip-only packages, then `pip check` and the version/import checks. A group that fails is split, only
the affected members, within a bounded number of attempts; valid groups are preserved and an
existing environment is never deleted -- a replacement is built in a staging prefix and committed
atomically only after it verifies. An environment the workbench did not create is never removed.

The authoritative assignment is `<paths.artifacts>/environment_assignment.json`. Inspect it, plus
`environment_plan.json`, before a long run. An explicitly exported `WORKBENCH_<MODEL>_PYTHON` takes
precedence for that model.

## 7. Checkpoints

`checkpoint_policy: strict_clean` (default) selects only variants whose recorded training condition
is exactly `0`. A variant whose condition is unrecorded is **not** assumed clean. Models that
publish only noise-trained weights are therefore BLOCKED under this policy.

`checkpoint_policy: include_flagged_conditions` selects them as an exploratory secondary cohort,
each keeping the condition the registry records. They are not part of a strict clean comparison.

`checkpoint_overrides` maps a model id to checkpoint ids; it is an operator-resolved mapping, wins
over the policy, and is badged in the report as an override rather than as clean.

A model with several registered variants is never evaluated under all of them: the selection is
resolved once and passed to the downloader, the checkpoint stage and the evaluator.

## 8. Missing assets (external)

These are not software defects; the models that need them stay BLOCKED.

| Model | Missing |
|---|---|
| `csmcir` | `COT_ours2/fashioniq` captions have no verified author source. Standard FashionIQ plus author Qwen captions are separate and the Qwen assets are downloadable from the registry. |
| `limn` | the three category checkpoints (URLs and hashes recorded, ~5.01 GiB), resized images (`fashioniq_ilearn_resized_training`), and the category correction dictionaries. |
| `ptha_mtst` | no source and no checkpoint variant; blocked. |
| `pair` | `pair_b1`/`pair_b2` mapping unverified. |
| `encoder` | direct checkpoint URL/hash unresolved; pinned evaluator imports `datasets1.py` which is absent; OpenCLIP binary asset required. |
| `dcnet`, `clvc_net`, `sprc`, `tgcir`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` | checkpoint mapping and/or backbone asset unresolved. |
| Air-Know / ConeSep / HABIT / INTENT | registered weights are noise-trained; no clean checkpoint recorded. |

`workbench/scripts/readiness_matrix.py` regenerates the per-model view of exactly this, with each
blocker spelled out.

## 9. First GPU pilot

`workbench/configs/pc_autopilot.pilot_csmcir.yaml` -- CSMCIR under `strict_clean`: the only model with
a recorded clean checkpoint, a pinned command and a resolved dataset path. Its one open prerequisite
is the `COT_ours2/fashioniq` captions; if those are absent the model reports `BLOCKED_CHECKPOINT`
or the corresponding asset blocker rather than running.

LIMN is the next candidate and needs `include_flagged_conditions` (its checkpoints record no
training condition) plus the three files, the resized images and the dictionaries.

```bash
./workbench/run.sh autopilot --config workbench/configs/pc_autopilot.pilot_csmcir.yaml --apply --resume
```

## 10. What a run produces

Under `paths.artifacts`:

| Path | Contents |
|---|---|
| `autopilot_execution_plan.json` | selection, checkpoint policy, environment grouping, pre-execution blockers |
| `environment_plan.json`, `environment_assignment.json` | groups, constraints, evidence, interpreter assignment |
| `autopilot_pipeline_state.json` | stage state used by resume |
| `reports/autopilot_summary.md`, `.json` | per-model status, blockers, and the category Recall table |
| `reports/autopilot_results.csv` | the same rows as CSV |
| `reports/blockers.json` | machine-readable blocker detail |
| `reports/run_manifest.json` | config digest, execution flag, outcome, pipeline exit code |

The summary carries, per completed model: run id, Dress/Shirt/Toptee Recall@10 and @50, macro, the
native protocol, the checkpoint id and provenance, the environment group, evaluation duration, the
completion-proof and aggregate-report references, the paper-reported values, and the measured
difference. A missing value is `null` with a reason, never zero. Aggregate-only results are marked
and are never presented as per-query evidence.

Progress: watch the stage lines on stdout; a run is safe to leave and inspect later. Exit code 0 with
`FULL_SUCCESS`, 2 with `PARTIAL` (some models completed, others blocked or failed), 1 with
`WORKFLOW_FAILURE`.

## 11. Resume and recovery

Rerun the same command with `--resume`. Verified stages are reused; an input that changed reruns its
dependents only. Preserve the selected models, the artifact roots, the GPU visibility and the ledger
key across retries. Do not delete the ledger, proofs, reports or checkpoints to force a retry, and do
not edit local data to satisfy a gate -- a blocker that comes from an unresolved upstream fact is
resolved with authoritative evidence or left blocked.

A failed model does not affect unrelated models; a failed global index does not change a successful
model's status. If a stage reports `BLOCKED_AUTHORIZATION_REQUIRED`, the config is missing the
capability the stage declared.

## 12. What is NOT verified here

- No real checkpoint has been loaded; no official evaluator has run against real FashionIQ.
- No Recall value in any report is a reproduction of a published number.
- No real CUDA capability has been recorded on this host.
- Per-query ranking export is not implemented; evaluation is aggregate only.
- The reference for `strict_clean` is evaluation noise 0% -- a checkpoint's *training* condition is
  reported separately and is never conflated with it.
