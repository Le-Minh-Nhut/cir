# PC Reproduction Handoff (Release B)

Instructions for moving the verified laptop engineering to a GPU PC for real
reproduction. **Do not** run these until a GPU host and authoritative assets are ready.

## 1. Required environment variables

```bash
export CIR_REPO_ROOT=/path/to/cir
export FASHIONIQ_ROOT=/data/FashionIQ          # captions/, image_splits/, images/
export WORKBENCH_CHECKPOINT_ROOT=$CIR_REPO_ROOT/workbench/artifacts/checkpoints
export WORKBENCH_RESULTS_ROOT=$CIR_REPO_ROOT/workbench/artifacts/results
export WORKBENCH_THIRD_PARTY_ROOT=$CIR_REPO_ROOT/workbench/third_party
```

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

Aggregate report: `workbench/artifacts/reports/<model>_<checkpoint>_aggregate.json`.
Logs: `workbench/artifacts/logs/<utc>_<model>_<checkpoint>/`.

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
