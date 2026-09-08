# IAG-SRME V2

Clean research implementation of the recurrent IAG-SRME model for composed image
retrieval. The architecture follows the
[V2 canonical specification](doc/CIR_IAG_SRME_UNIFIED_CANONICAL_ARCHITECTURE_AND_TRAINING_SPEC_V2_2026-09-06.md).

The current implementation is Track B (FG-CLIP v1 Base) with two orthogonal ablation
axes:

- `R0-QG` / `learned_qg`: canonical learned recurrent global query;
- `R0-NCLS` / `native_cls`: immutable image-specific penultimate CLS and current
  patches feed the exact CLS row of the native final vision block; the full-token call
  remains a parity oracle.

FG-CLIP fine-tuning is independently either `full` (vision and text trainable) or
`text_only` (vision and visual projection frozen; text encoder trainable). The text
adapter and all IAG-SRME modules remain trainable in both policies. In QG runs, `q_G`
also remains trainable because it is a task-specific parameter.

Both modes keep the following architecture fixed:

- persistent state is the penultimate patch representation, without CLS;
- proposals are regenerated from current visual state and instruction tokens;
- grounding uses native dense cosine evidence with separate softmax read and sigmoid write maps;
- actions use entity-conditioned independent gates;
- all candidates are masked local residual previews from one parent state;
- one shared ScoreNet predicts absolute marginal utility against KEEP/STOP = 0;
- rollout uses hard argmax/gather, with no Gumbel, straight-through estimator, or state mixture;
- training targets appear only in the external retrieval teacher and terminal loss;
- instruction concepts supervise the pooled proposal set at `t=0` only;
- relation binding uses a prototype bank separate from proposal queries;
- Functional DPP operates on retrieval consequences `delta_q` and detached executed history;
- correspondence is disabled.

The complete architecture is in
[`src/models/iag_srme/model.py`](src/models/iag_srme/model.py). FG-CLIP-specific
state/readout code and teacher retrieval helpers are the only extracted utilities.
The current implementation/verification snapshot is recorded separately in the
[V2 R0 experiment status](doc/CIR_IAG_SRME_V2_R0_EXPERIMENT_IMPLEMENTATION_STATUS_2026-09-06.md).

## Setup

```bash
python -m pip install -e '.[dev]'
```

The default config pins `qihoo360/fg-clip-base` at revision
`454d76372c2cf5eb48fa0d871fd0534481484d97`.

## Test

```bash
pytest -q
ruff check src tests
```

## FashionIQ smoke update

Expected data layout:

```text
data/fashionIQ_dataset/
├── captions/
├── image_splits/
└── images/
```

Run one small target-firewalled train update:

```bash
python src/canary_train_iag_srme.py \
  --dataset-root data/fashionIQ_dataset \
  --steps 1 --batch-size 2 --precision fp32
```

Add `--global-readout-mode native_cls` for the R0-NCLS canary.
Add `--finetune-policy text_only` for a frozen-vision canary.
On CUDA, the canary JSON also reports peak allocated and reserved memory in GiB.

```bash
python src/canary_train_iag_srme.py \
  --global-readout-mode native_cls \
  --finetune-policy text_only \
  --dataset-root data/fashionIQ_dataset \
  --steps 1 --batch-size 16 --precision fp16
```

## Train

```bash
# R0-QG-FULL
python src/train.py backbone=fgclip_base_full_qg

# R0-NCLS-FULL
python src/train.py backbone=fgclip_base_full_native_cls

# R0-QG-TEXT
python src/train.py backbone=fgclip_base_text_qg

# R0-NCLS-TEXT
python src/train.py backbone=fgclip_base_text_native_cls
```

The FULL/TEXT pair for a fixed readout differs only in `train_vision`, the explicit
`finetune_policy`, and experiment identity metadata. Optimization settings are shared.

The default objective config enables all requested A6 auxiliaries. Each term can be
ablated independently with Hydra overrides such as
`objective.concept_enabled=false` or `objective.dpp_enabled=false`.

Dataset loading, FashionIQ evaluation, AMP policy, checkpointing, optimizer ownership
checks, and stable `CIRSample.target_id` handling are retained from the clean-rewrite
branch.

## Matched OLD/STRONG diagnostics

Diagnostics require a persistent manifest so both checkpoints see the same training
samples, captions, ordering, target/teacher pool, and seed. The TRAIN-160 manifest
below is tracked in the repository; every command validates and reuses it.

```bash
# OLD: geometry and per-slot/slot-centered spectra
python src/diagnose_latent_geometry.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text/best.pt \
  +diagnostic_manifest=doc/diagnostics/2026-09-08_v2-r0_old-vs-strong/shared_train_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  hydra.run.dir=outputs/diagnostics/v2-r0/old_geometry

# STRONG: exact replay of the same geometry cohort
python src/diagnose_latent_geometry.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  +diagnostic_manifest=doc/diagnostics/2026-09-08_v2-r0_old-vs-strong/shared_train_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  hydra.run.dir=outputs/diagnostics/v2-r0/strong_geometry

# OLD selector/utility/STOP/caption sensitivity + official FashionIQ validation
python src/diagnose_candidate_selector.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text/best.pt \
  +diagnostic_manifest=doc/diagnostics/2026-09-08_v2-r0_old-vs-strong/shared_train_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  +diagnostic_official_eval=true \
  hydra.run.dir=outputs/diagnostics/v2-r0/old_selector

# STRONG selector replay
python src/diagnose_candidate_selector.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  +diagnostic_manifest=doc/diagnostics/2026-09-08_v2-r0_old-vs-strong/shared_train_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  +diagnostic_official_eval=true \
  hydra.run.dir=outputs/diagnostics/v2-r0/strong_selector

# Compare recurrent geometry only on IDs live in both checkpoints at each timestep
python src/compare_matched_diagnostics.py \
  --old-features outputs/diagnostics/v2-r0/old_geometry/candidate_features.pt \
  --strong-features outputs/diagnostics/v2-r0/strong_geometry/candidate_features.pt \
  --output-dir outputs/diagnostics/v2-r0/old_vs_strong_matched
```

Persistent manifests are strict: `diagnostic_batches * diagnostic_batch_size` must
match the request that created the manifest, and every stored row is processed in
the original teacher-batch grouping. Geometry runs emit
`latent_geometry_report.{json,md}` plus `candidate_features.pt`. Selector runs emit
`candidate_selector_diagnostic.{json,md}`. Training now appends update and validation
records to `metrics.jsonl` and persists `resolved_config.yaml` plus
`run_metadata.json`; checkpoints store the true optimizer/batch step counters and AMP
scaler state.

The comparison command emits `matched_geometry_comparison.{json,md}`. Its timestep
tables separate each checkpoint's native survivor cohort from the OLD/STRONG live-ID
intersection, so t1/t2 rank changes are not confounded by different STOP survivors.

Start with `headline_candidate_geometry` in the geometry JSON and `headline_metrics`
in the selector JSON. They expose per-slot and slot-centered PR, between-slot variance,
caption utility advantage, per-slot teacher utility/occupancy, selector regret,
harmful-execution rate, STOP precision/recall, and optional official FashionIQ recall.
Candidate-slot collapse should be read from `selected_fraction_given_execute`; oracle
slot concentration should be read from `oracle_best_fraction_given_oracle_execute`.

## ScoreNet gain-only rescue

This is a two-phase scorer-only experiment. Phase 1 freezes the STRONG checkpoint and
stores detached raw inputs to every trainable part of `ScoreNet`; Phase 2 rebuilds
features and optimizes every `score_net.*` parameter with only the canonical absolute
gain Huber loss. The cache is sharded because `delta [B,K,N,D]` is intentionally kept
raw. No target embedding or target-derived feature is a ScoreNet input.

```bash
# Phase 1: deterministic fixed TRAIN rollout and raw scorer cache
python src/refit_score_net.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +scorer_refit=gain_only \
  scorer_refit.mode=collect \
  scorer_refit.source_checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  scorer_refit.cache_dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/cache \
  hydra.run.dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/collect

# Phase 2: ScoreNet-only AdamW, absolute gain only
python src/refit_score_net.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +scorer_refit=gain_only \
  scorer_refit.mode=refit \
  scorer_refit.source_checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  scorer_refit.cache_dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/cache \
  scorer_refit.output_checkpoint=outputs/r0_ncls_text_strong_aux_score_gain_refit/score_gain_refit.pt \
  hydra.run.dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/refit
```

Refit rejects cache/config mismatches in source checkpoint SHA256, K, T, STOP enable,
`epsilon_stop`, retrieval temperature, and `score_dropout`. Its checkpoint is directly
loadable for inference/evaluation, but intentionally has `optimizer: null` and
`scaler: null`: full-model continuation is a warm start with a newly constructed full
optimizer, not an exact resume. The scorer-only AdamW state is stored separately as
`score_refit_optimizer`.

The raw full-TRAIN cache is intentionally large. With 18,000 FashionIQ TRAIN rows,
K=4, T=3, 196 patches and D=768, `delta` alone has an upper bound of about 60.6 GiB
in fp16 (121.1 GiB in fp32), before filesystem serialization overhead. STOP and
invalid teacher rows reduce this in practice. The current cache stays raw and sharded;
changing it to `local_mean [B,K,D]` would be a separate storage-only optimization.

Cached calibration is only an offline fit diagnostic: its states were visited by the
original STRONG policy. Better cached Pearson/MAE does not establish a successful live
policy rescue.

### TRAIN-160 live diagnostic (debugging only)

This cohort overlaps the scorer training distribution. It is useful for fit debugging
and trajectory comparison, but it is not held-out evidence.

```bash
# Original STRONG
python src/diagnose_candidate_selector.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  +diagnostic_manifest=doc/diagnostics/2026-09-08_v2-r0_old-vs-strong/shared_train_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  +diagnostic_official_eval=false \
  hydra.run.dir=outputs/diagnostics/v2-r0/strong_selector_train160

# Gain-only refit
python src/diagnose_candidate_selector.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text_strong_aux_score_gain_refit/score_gain_refit.pt \
  +diagnostic_manifest=doc/diagnostics/2026-09-08_v2-r0_old-vs-strong/shared_train_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  +diagnostic_official_eval=false \
  hydra.run.dir=outputs/diagnostics/v2-r0/score_gain_refit_selector_train160
```

### VAL-160 held-out selector diagnostic

Run STRONG first. It deterministically creates `shared_val_160.json`; the refit command
then strictly replays the same IDs, order, captions and teacher-batch grouping.

```bash
# Original STRONG creates the persistent VAL manifest
python src/diagnose_candidate_selector.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  +diagnostic_split=val \
  +diagnostic_manifest=outputs/diagnostics/v2-r0/shared_val_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  +diagnostic_official_eval=false \
  hydra.run.dir=outputs/diagnostics/v2-r0/strong_selector_val160

# Gain-only refit reuses the exact VAL cohort
python src/diagnose_candidate_selector.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text_strong_aux_score_gain_refit/score_gain_refit.pt \
  +diagnostic_split=val \
  +diagnostic_manifest=outputs/diagnostics/v2-r0/shared_val_160.json \
  +diagnostic_batches=20 +diagnostic_batch_size=8 \
  +diagnostic_official_eval=false \
  hydra.run.dir=outputs/diagnostics/v2-r0/score_gain_refit_selector_val160
```

### Full official FashionIQ validation

```bash
python src/evaluate.py \
  backbone=fgclip_base_text_native_cls \
  dataset.root=data/fashionIQ_dataset \
  +checkpoint=outputs/r0_ncls_text_strong_aux_score_gain_refit/score_gain_refit.pt \
  hydra.run.dir=outputs/r0_ncls_text_strong_aux_score_gain_refit/eval
```

Interpret results in this order: official Mean Recall; live selected teacher utility;
live oracle regret; harmful execution fraction; STOP precision/recall/F1;
selected/oracle agreement; score/teacher calibration; slot occupancy. VAL-160 is the
held-out scorer-generalization diagnostic and should include utility/regret, harmful
execution, STOP metrics, Pearson/bias/MAE/RMSE/sign agreement, agreement, slot usage
and execute rate by timestep. Official FashionIQ VAL remains the primary benchmark
through R@10, R@50 and Mean Recall.

If cached calibration improves but VAL live regret and official Recall do not, the
conclusion is that gain calibration is insufficient: ScoreNet is secondary and
upstream candidate quality remains the dominant bottleneck. `L_safe` is the next
separate research step; it is not part of this scorer-refit experiment.
