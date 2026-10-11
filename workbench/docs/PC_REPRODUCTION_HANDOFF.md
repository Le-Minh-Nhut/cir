# PC Reproduction Handoff (Release B)

Instructions for moving the verified laptop engineering to a GPU PC for real
reproduction. **Do not** run these until a GPU host and authoritative assets are ready.

## 0. Source checkout status

All 14 pinned upstream repositories are cloned (source-only, no LFS) under
`workbench/third_party/` and pin-verified. See `UPSTREAM_SOURCE_AUDIT.md` for the
per-repo commit, size, and evaluator-file evidence. Re-verify at any time:

```bash
python workbench/scripts/sync_upstreams.py --all --list
python workbench/scripts/sync_upstreams.py --model MODEL_ID --verify-only
```

Cloned source is not checkpoint availability. The PC host still needs the
checkpoints and auxiliary assets listed in the table below.

## 0b. Evidence ledger key (set this before any real run)

```bash
export WORKBENCH_LEDGER_KEY="$(head -c 32 /dev/urandom | base64)"   # from your secret store
```

Every completion proof is corroborated by an append-only, HMAC-authenticated ledger at
`workbench/artifacts/evidence_ledger.jsonl`. The key is kept outside the artifact tree and is
never passed to the evaluator. Losing the key makes existing entries unverifiable; the
workbench never silently rotates or regenerates it. The honest trust model (and what the
ledger does *not* protect against) is documented in `AUTOMATION_RUNBOOK.md`.

## 1. Required environment variables

```bash
export CIR_REPO_ROOT=/path/to/cir
export FASHIONIQ_ROOT=/data/FashionIQ          # captions/, image_splits/, images/
export WORKBENCH_CHECKPOINT_ROOT=$CIR_REPO_ROOT/workbench/artifacts/checkpoints
export WORKBENCH_RESULTS_ROOT=$CIR_REPO_ROOT/workbench/artifacts/results
export WORKBENCH_THIRD_PARTY_ROOT=$CIR_REPO_ROOT/workbench/third_party
```

## 1b. Model interpreter selection (pre-flight for every legacy model)

Each model with `environment_required: true` needs its own verified interpreter.
The workbench will refuse to construct a real evaluation command until the runtime
is *verified*, not merely present:

```bash
export WORKBENCH_LIMN_PYTHON=/opt/envs/limn/bin/python   # keep the invocation path as-is
python workbench/scripts/manage_environment.py --model limn --json
```

- Do **not** pre-resolve venv paths (`readlink -f`): a venv's `bin/python` commonly
  symlinks to a shared binary, and resolving it destroys the environment context.
- Readiness requires: isolated environment, Python version agreement, and a probed
  dependency set matching the source-declared constraints (`packaging` semantics).
- A GPU-only check that cannot run on a CPU host reports
  `GPU_VERIFICATION_DEFERRED`; that is not a success and will not launch a real run.
- CSMCIR additionally requires `fashionIQ_dataset` to be a symlink to the canonical
  `FASHIONIQ_ROOT`; the link is an approved runtime artifact and does not dirty the
  pinned checkout, but a wrong or dangling link is rejected before execution.

## 1d. GPU verification procedure (GPU host only)

A CUDA-declaring model (LIMN declares `cuda: '12.4'`) is `GPU_VERIFICATION_PENDING` until a
real probe proves PyTorch can use a device. Run it once per environment:

```bash
export WORKBENCH_LIMN_PYTHON=/opt/envs/limn/bin/python
python workbench/scripts/evaluate_models.py --model limn --verify-gpu-only
```

`--verify-gpu-only` is the hardware preflight: it inspects the environment, runs the
explicitly authorized probe, records the capability, prints the structured result, and exits.
It never loads a checkpoint, never starts an evaluator, never computes Recall, never writes a
completion proof, and never moves `latest_successful`. It verifies the **model environment**,
not the host: device count, device name, compute capability, a one-element tensor allocation,
and a trivial computation. A CUDA toolkit or a driver version is not proof that PyTorch can use
CUDA — only this probe is.

Records are per-capability, under `workbench/artifacts/gpu_capabilities/<capability_key>.json`,
so verifying a second model does not overwrite the first. The key includes the interpreter and
dependency fingerprints, so rebuilding an environment invalidates that model's record and it
must be re-run.

`--verify-gpu` (without `-only`) means the same thing but *then continues into evaluation* — it
establishes or refreshes the capability and proceeds if the launch gate is satisfied.

Then the orchestrated evaluation proceeds:

```bash
python workbench/scripts/pipeline.py reproduce --model limn --apply --allow-gpu-eval --resume
```

Note there is no `--verify-gpu` here. A CUDA-declaring model launches when a valid recorded
capability exists for this exact environment; the launch gate is the authority, and a missing
or stale capability refuses the launch naming what to run. `--verify-gpu` remains available on
the pipeline when the operator wants the probe re-run inside that workflow.

## 1c. PC-only verification still required

Not verifiable on the laptop, and therefore still owed on the GPU host:

- real checkpoint loading and official evaluator execution;
- official Recall reproduction against paper-reported values;
- per-query instrumentation (`per_query_export_available` is `false` today);
- **the real GPU capability probe** for CUDA-bound models (the implementation and its failure
  handling already exist and are tested; only a real device result is missing).

## 2. Acquisition and setup (explicit authorization required)

```bash
# Plan first
python workbench/scripts/pipeline.py setup --model MODEL_ID
# Execute with authorization
python workbench/scripts/pipeline.py setup --model MODEL_ID --apply \
  --allow-network --allow-large-downloads --allow-env-install --allow-preparation --resume
```

## 3. Preflight (read-only)

```bash
python workbench/scripts/doctor.py --scope real --model MODEL_ID --json
python workbench/scripts/manage_environment.py --model MODEL_ID
python workbench/scripts/prepare_model.py --model MODEL_ID
python workbench/scripts/download_checkpoints.py --model MODEL_ID --verify-only
```

## 4. Real evaluation

```bash
python workbench/scripts/pipeline.py reproduce --model MODEL_ID --apply --allow-gpu-eval --resume
```

Aggregate report: `workbench/artifacts/reports/<model>_<checkpoint>_<run_id>_aggregate.json`.
Completion proof: `workbench/artifacts/reports/<model>_<checkpoint>_<run_id>_completion.json`.
Logs: `workbench/artifacts/logs/<run_id>/`.

A stage is COMPLETE only when the current invocation produced a valid completion
proof (run id, model, checkpoint, protocol, source commit, checkpoint digest, report
digest, and invocation evidence all agree). `latest_attempt` records every run;
`latest_successful` advances only after a proven success. If a run exits 0 without
producing output, the stage fails and the previous success is preserved.

## 5. Verify aggregate metrics and analysis

```bash
python workbench/scripts/pipeline.py analyze
python workbench/scripts/validate_results.py --root "$WORKBENCH_RESULTS_ROOT" --strict-real
python workbench/scripts/rebuild_index.py --results-root "$WORKBENCH_RESULTS_ROOT"
```

## 6. Per-model handoff

| Model | Protocol | Checkpoint structure | Command | Env | Preparation | External blocker | PC next action |
|---|---|---|---|---|---|---|---|
| csmcir | fashioniq_original_split | single file | audited | python 3.9 / torch 2.0.1 | link + Qwen (auto) + COT (manual) | COT_ours2 captions lack author URL | Place COT captions, then reproduce |
| airknow | fashioniq_full_gallery_ref_excluded | single file | unaudited | 3.8/2.1.0 | manual | only 50/80% noise weights | Audit command or keep blocked |
| conesep | same | single file | unaudited | 3.8/2.1.0 | manual | only 20/50/80% weights | Audit command |
| habit | same | single file | unaudited | 3.8/2.1.0 | manual | only 20/50/80% weights | Audit command |
| intent | same | single file | unaudited | 3.8/2.1.0 | manual | only 20/50/80% weights | Audit command |
| hint | fashioniq_val_split | single file | unaudited | 3.8.10/CUDA 12.6 | manual | CLI unaudited | Audit CLI |
| encoder | fashioniq_val_split | shared single | audited | 3.9 | manual | Drive URL + missing datasets1.py | Resolve checkpoint, restore module |
| pair | fashioniq_val_split | unresolved variants | unaudited | 3.8.10/2.0.0 | manual | B1/B2 mapping unverified | Resolve mapping |
| ptha_mtst | — | none | — | — | — | no FashionIQ checkpoint | Keep blocked |
| clvc_net | fashioniq_val_split | local+global | blocked entrypoint | 3.7.6/1.6.0 | manual | no standalone CLI; ResNet-50 | Provide replay entrypoint |
| dcnet | fashioniq_full_gallery_ref_excluded | run directory | audited | 3.7.7/1.4.0 | manual PKLs | Drive hash + ResNet-50 | Verify run dir hash |
| combiner_rn50x4_noft | fashioniq_original_split | single state | audited | 3.8/1.11.0 | manual | mapping + RN50x4 base | Resolve mapping |
| clip4cir_rn50x4_fullft | fashioniq_original_split | paired bundle | audited | 3.8/1.11.0 | manual | pairing + RN50x4 base | Resolve pairing |
| tgcir | fashioniq_val_split | whole model | blocked entrypoint | 1.7.0 | manual | no CLI; ViT-B/16 | Provide replay entrypoint |
| sprc | fashioniq_original_split | state dict | blocked mapping | 3.9/2.0.1 | manual | backbone mapping | Verify mapping |
| limn | fashioniq_val_split | 3 category models | replay implemented | CUDA 12.4/1.12.1/open_clip 2.20.0 | manual images + dicts | 3 checkpoint download | Download 3 checkpoints, preflight, run |

## 7. Resume and recovery

```bash
python workbench/scripts/pipeline.py reproduce --model MODEL_ID --apply --allow-gpu-eval --resume
python workbench/scripts/pipeline.py reproduce --model MODEL_ID --apply --allow-gpu-eval --force-stage evaluation
```

State: `workbench/artifacts/pipeline_state.json` (atomic writes).

## 8. Engineering status at the Phase I/II handoff

The laptop side of the chain is complete and verified. Before starting any GPU work, know which
guarantees hold and which are still open.

**What runs immediately on the PC, no checkpoint needed:**

```bash
python workbench/scripts/doctor.py --all                      # registry, sources, layouts, assets
python workbench/scripts/prepare_dataset.py --dataset-root <FashionIQ>   # refuses a bad layout
python workbench/scripts/evaluate_models.py --model limn --verify-gpu-only   # records a capability
python -m pytest tests/unit workbench/tests -q                # 565 tests
```

**What the preflight now catches before a GPU run**: a missing dataset path, a missing declared
auxiliary asset, an image whose extension is not the one the evaluator hard-codes (CSMCIR opens
`.png` while the official release ships `.jpg`), an unverified GPU capability, a stale capability
after an environment change, an unclean or unpinned source checkout, and a run id that already
exists.

**What still requires the GPU host**: every checkpoint download and verification, every
environment build, every native evaluator run, and every Recall number. No score in any document
in this repository is a reproduction.

**What remains scientifically open, per model**, with the audit that records it:

| Model | Open item | Where |
|---|---|---|
| airknow, conesep, habit, intent, hint, pair, clvc_net, tgcir, sprc, dcnet | pinned evaluator is a callable with no source-supported CLI; reaching a runnable command is itself unresolved | `audits/PHASE2_SOURCES_{A,B,C,D}.md` |
| csmcir | LAVIS base weight and COT_ours2 provenance; torchvision version | `PHASE2_SOURCES_A.md` |
| encoder | base OpenCLIP binary provenance; `datasets1.py` presence | `PHASE2_SOURCES_A.md` |
| pair | which of `pair-B1.pt` / `pair-B2.pt` is the FashionIQ benchmark model; nothing in the source settles it | `PHASE2_SOURCES_A.md` |
| clvc_net | one registered pair cannot cover three per-category whole models | `PHASE2_SOURCES_B.md` |
| combiner_rn50x4_noft, clip4cir_rn50x4_fullft | the validator hardcodes `<checkout>/fashionIQ_dataset`, so a canonical root alone is insufficient | `PHASE2_SOURCES_B.md` |
| sprc | native inference ignores its reference argument, so local scores would be text-only | `PHASE2_SOURCES_C.md` |
| dcnet | author-generated GloVe PKLs must not be regenerated and called equivalent | `PHASE2_SOURCES_B.md` |

**How to establish Recall parity**: for one model, run the native evaluator on the pinned source
with the author checkpoint, capture per-category and per-query rankings, and compare against the
paper-reported values as sanity references only. The completion proof records the command,
environment, source commit and checkpoint digest for exactly that run; a number without a proof is
not evidence.

**Two invariants worth re-checking on the GPU host**: a queued job and a CLI run must produce the
same ledger entry shape (queue logs live under `artifacts/queue_jobs/`, run artifacts under
`artifacts/logs/`), and cancelling a job must leave no evaluator process behind.
