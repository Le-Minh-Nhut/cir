# IAG-SRME Implementation Guide

## Architecture-to-code map

| Research block | Implementation |
|---|---|
| FG-CLIP patch/global/text extraction and regime contract | `src/models/iag_srme/backbone.py` |
| stable text-only WHAT intents | `TextIntentEncoder` in `intent.py` |
| stable anchor-grounded sparse WHERE | `AnchorGrounder` in `grounding.py` |
| original/current/change reads | `GroundedStateReader` in `grounded_reader.py` |
| state-conditioned HOW | `GroundedEditContext` in `context.py` |
| shared, bounded, exactly support-gated edit | `SharedTokenEditor` in `editor.py` |
| accumulated-token-change retrieval bottleneck | `TokenStateReadout` in `readout.py` |
| target-free VALUE field | `ConsequenceScorer` in `scorer.py` |
| hard candidate/fixed-zero absorbing STOP | `HardStopSelector` plus recurrence in `model.py` |
| typed trajectory/firewall surface | `outputs.py` |

`IAGSRMECore.forward` computes `E`, `P`, and `A` once. It then recomputes grounded current
evidence, context, edit, candidate state/query, effect, and score at every step. Candidate states
are constructed by the literal expression `current_state[:, None] + delta_z`; only selection
commits a consequence. The public model forward signature contains reference pixels and text
tensors only.

## Directory structure

```text
src/models/iag_srme/    architecture modules
src/losses/             six independent objectives and objective composer
src/data/               raw-image FG-CLIP collation
src/datasets/           generic CIR/FashionIQ annotations
src/evaluation/         current-checkpoint FashionIQ retrieval protocol
src/diagnostics/        structural/functional metrics and control registry
src/training/           generic end-to-end optimizer/checkpoint loop
tests/                  deterministic scientific invariants
conf/                   explicit backbone/model/objective/experiment/protocol configs
```

## Environment

Python 3.11+ and PyTorch are required. The official FG-CLIP checkpoints use Hugging Face remote
model code. Entmax is pinned because sparse exact-zero support is an architectural contract.

```bash
python -m pip install -e '.[dev]'
```

The configs select FG-CLIP v1 at immutable Hugging Face revisions:

- `qihoo360/fg-clip-base@454d76372c2cf5eb48fa0d871fd0534481484d97`
- `qihoo360/fg-clip-large@5a8f0f23b5a06dc92310e907599b2a0c2d58fe6f`

The same revision is passed to model, tokenizer, and image-processor loading and is stored in
checkpoint metadata. Evaluation rejects a checkpoint/config revision mismatch. They do not
substitute FG-CLIP2.

## FashionIQ layout

Set `CIR_DATA_ROOT` to the parent of `FashionIQ`:

```text
$CIR_DATA_ROOT/FashionIQ/
  captions/cap.{dress,shirt,toptee}.{train,val}.json
  image_splits/split.{dress,shirt,toptee}.val.json
  images/<image-id>.{png,jpg,jpeg}
```

The dataset returns stable sample/reference/target IDs, modification text, and category. The
collator resolves raw images and applies the processor belonging to the configured checkpoint.

## Training

FG-CLIP Base, full image and contextual text fine-tuning from update 1:

```bash
python src/train.py backbone=fgclip_base_full experiment=iag_srme_base_full objective=core
```

FG-CLIP Large, vision and pretrained visual projection frozen throughout, contextual text
encoder plus IAG token-projection fine-tuning:

```bash
python src/train.py backbone=fgclip_large_text_ft experiment=iag_srme_large_text_ft objective=core
```

Both commands construct the full three-step graph, four previews, hard-forward selector, STOP,
terminal objective, and marginal objective before the first optimizer update. There is no
component, horizon, selection, freezing, or objective curriculum.

Both canonical experiments use `train_caption_policy=ordered_and`, so the model receives both
FashionIQ captions. Incomplete four-way caption augmentation remains an explicit ablation:

```bash
python src/train.py backbone=fgclip_base_full \
  experiment=iag_srme_base_randomized_caption_ablation objective=core
```

Device placement is owned by `src/train.py`: model and objective move to the final device before
AdamW is created. `fit()` validates device and exact parameter-object identity but never migrates
modules. Precision is explicit: `runtime.precision=fp32` disables autocast, `fp16` uses float16
autocast plus CUDA GradScaler, and `bf16` uses bfloat16 autocast without fp16 scaling.

### Text trainability contract

Canonical core training follows the contextual-token path required by the master specification:

```text
FG-CLIP text_model.last_hidden_state
→ trainable IAG hidden-size-to-d projection
→ X
→ intents / grounding / recurrent execution
```

Accordingly, `FG-CLIP text_model` and the IAG token projection are trainable and receive core
gradients. FG-CLIP's pretrained retrieval-space `model.text_projection` is frozen: its only
current use is to provide `text_semantic_global` for the detached auxiliary factor anchor. No
gradient is fabricated for it, and it is not inserted into the final query as a shortcut.

## Validation

Use the same backbone/experiment pair as the checkpoint:

```bash
python src/evaluate.py backbone=fgclip_base_full experiment=iag_srme_base_full \
  protocol=fashioniq_original checkpoint=/path/to/best_original.pt

python src/evaluate.py backbone=fgclip_large_text_ft experiment=iag_srme_large_text_ft \
  protocol=fashioniq_val checkpoint=/path/to/best_val.pt
```

Validation regenerates gallery embeddings with the loaded current checkpoint. This is mandatory
for trainable vision and deliberately remains the default for frozen vision.
Training evaluates both FashionIQ protocols independently after each epoch and writes
`best_original.pt`, `best_val.pt`, and `last.pt`. Either selected checkpoint can be evaluated
under either protocol because protocol choice is not part of backbone identity validation.

## Loss configurations

`objective=core` enables only `L_terminal + 0.5 L_marginal`.

Optional groups are `comp`, `bind`, `factor`, `unique`, and `six_loss_experimental`. Enable the
matching model branches explicitly, for example:

```bash
python src/train.py objective=comp model.enable_claim_head=true
python src/train.py objective=bind model.enable_claim_head=true
python src/train.py objective=factor model.enable_factor_head=true
```

`L_unique` accepts `active_weights[B,K]`. The objective guard rejects it without externally
justified activity weights because four proposal identities are not four guaranteed true edits.
There is intentionally no semantic NULL implementation, and STOP is never reused as one. An
all-active experiment requires explicitly disabling that guard and must be reported as such.
The provided training pipeline does not manufacture `active_weights`; therefore `objective=unique`
and `objective=six_loss_experimental` are research-interface configs, not production-ready runs.

For sample `i`, the factor relational target is
`u_i = normalize(reference_global_i + text_semantic_global_i)`. Both operands use the pinned
FG-CLIP checkpoint's trained retrieval projections. The composition is parameter-free—there is
no random detached MLP. `relational_geometry` detaches `u_i`, so `L_factor` trains the factor
fuser but cannot co-adapt or move its semantic target. The auxiliary anchor is returned only for
factor/unique losses and is never consumed by mutable state, context, editor, scorer, selector,
or retrieval readout.

## Cache legality

- Base full fine-tuning: persistent reference, target, and gallery feature caches are illegal.
- Large frozen vision: a future cache may be legal only with an exact checkpoint, preprocessing,
  image-ID, and projection manifest.
- This rewrite intentionally implements no image-feature cache. It always takes the safe live
  path and has no compatibility path for historical feature arrays.

## Target firewall

Target pixels enter only `model.encode_global_images` after the target-free forward has constructed all
intents, supports, contexts, deltas, candidate states/queries, scores, actions, and final state.
`L_terminal` may update the target encoder normally. `L_marginal` detaches the target bank and the
computed retrieval gains before score matching. Target IDs are used only to construct the
multi-positive loss mask and evaluation labels.

## Tests, smoke, and diagnostics

```bash
pytest -q
python src/smoke_iag_srme.py
python src/smoke_iag_srme.py --diagnostics
python src/smoke_fgclip_integration.py --max-steps 3
```

The smoke path uses FG-CLIP-compatible tensors when a checkpoint is unavailable and executes
intent, entmax grounding, recurrence, four previews, candidate readout, score/STOP, terminal and
marginal losses, and backward.

The model exposes the controls `full`, `zero_edit`, `single_candidate`, `repeat_candidate_1` ...
`repeat_candidate_4`, `repeat_best`, `clone_candidate_1`, `mean_candidate`, `random_candidate`,
and `frozen_t0_order`. `summarize_trajectory` reports intent cosine, grounding support/entropy/
overlap, functional effect cosine/effective rank, selected identities, STOP rate, score evolution,
and claim mass when enabled. Realized target-evaluated marginal utility is available from
`losses.marginal.detached_marginal_utilities` and remains outside the forward graph.

Reference encoding performs one official vision-model pass and derives both penultimate-layer
dense tokens and the pooled global embedding with FG-CLIP's own post-layernorm/projection logic.
Target and gallery encoding call only the global-image API and never construct dense tokens.

`smoke_fgclip_integration.py` numerically compares the one-pass reconstruction against the
official dense/global helpers and compares vision, visual-projection, and anchor-projection
gradients. It also instruments the real checkpoint: one training update must invoke two batched
vision forwards total (reference + target), while one gallery batch invokes one global-only
vision forward and never invokes the anchor projection.

Training uses a hard-forward straight-through Gumbel estimator from update 1. When enabled, one
temperature-scaled Gumbel-perturbed distribution supplies both the hard argmax and the soft
backward surrogate; evaluation is deterministic argmax. For samples already stopped, the action
is detached hard STOP, so later unrolled steps contribute no selector/candidate gradient.

### Mixed-precision islands

AMP remains active for the backbone and ordinary IAG-SRME modules. Only sensitive arithmetic is
promoted locally to FP32:

- entmax-1.5 threshold/root arithmetic for grounding and marginal targets;
- Fenchel-Young value and its analytic `prediction - target` score gradient;
- retrieval normalization, similarity, and `logsumexp`;
- bounded-vector norm and L2 normalization;
- complementary-claim log/JS operations;
- binding similarity/cross-entropy;
- factor/unique normalization, relational logits, log-PoE, and KL.

LayerNorm remains under PyTorch autocast's numerically safe operator policy. These islands do not
change the objective or disable mixed precision globally.

## GPU FashionIQ canary

Before a full run, execute the canonical Base/full, `K=4`, `Tmax=3`, `d=256`, core-only fp16
canary:

```bash
python src/canary_train_iag_srme.py \
  --dataset-root /absolute/path/to/FashionIQ \
  --steps 100 \
  --precision fp16
```

It requires CUDA and real FashionIQ images. Every logged interval reports losses, gradient norms,
parameter deltas, STOP/candidate distributions, sparse-grounding statistics, effect norm/rank,
dynamic-state changes, peak VRAM, and real module call counts. It aborts on non-finite or zero
expected gradients, exploding gradients, OOM, wrong reference/target call counts, broken STOP
identity, or sustained all-STOP/never-STOP/candidate-monopoly/identical-effect collapse.

## Unresolved research contract

The code does not claim semantic factorization is solved. Variable true edit count, factor
activity, normalized-claim behavior for inactive identities, secret-sharing, arbitrary relational
coding, and representation-versus-functional specialization require controlled experiments and
intervention evidence.
