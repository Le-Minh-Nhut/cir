# IAG-SRME R1c1: Dynamic Current-State Regrounding

## Status and lineage

Implementation branch:

```text
exp/e2e-iag-srme-r1b-visual-null-confidence-gate
  @ 88068c9328964b3a6d925955a5f2b1eeb3259d2e
        ↓
exp/e2e-iag-srme-r1c1-dynamic-reground
```

`88068c9` is a diagnostic/report-only commit after the audited R1b implementation at
`d69d0c7`. It does not add a scientific mechanism. R1c1 retains R1a's successful
`query_cap=1000.0` condition and explicitly disables R1b's applicability gate.

Implementation is complete. A trained R1c1 scientific result is not available yet.

## Hypothesis and causal delta

R1c1 tests one question:

> Does reuse of the initial visual grounding prevent fixed text actions from responding to the
> edited recurrent state?

The only scientific delta from the R1a mechanism is:

```text
Ground(I_k, A) once
        ↓
Ground(I_k, Z_t) before every recurrent decision
```

R1c1 is therefore:

```text
fixed WHAT + dynamic current-state WHERE
```

It is not R1b plus dynamic grounding. Canonical R1c1 uses:

```yaml
query_cap: 1000.0
enable_dynamic_regrounding: true
enable_dynamic_applicability: false
```

No Visual NULL or confidence scalar multiplies the edit.

## Code-path audit before intervention

The parent implementation computed:

```python
intents = intent_encoder(text_tokens, content_mask)
supports = grounder(intents, anchor)
for t in range(Tmax):
    reader(supports, anchor, current_state)
    editor(..., supports, ...)
    scorer(..., supports)
```

Consequently, reader, editor, scorer, and every REPEAT control reused the identical support map.
The intent encoder was already text-only and computed once. Candidate previews already obeyed the
same-parent invariant. Entmax already used an FP32 threshold/root island over visual token axis N.

## Mathematical implementation

Text-only intents remain fixed:

```math
I_k = Intent(q_k,T),\qquad I\in\mathbb{R}^{B\times K\times d}.
```

At every recurrent parent state, the same shared grounder computes:

```math
Q_k=W_QI_k,
```

```math
K_{t,n}=W_KZ_{t,n},
```

```math
\ell_{t,k,n}=\frac{Q_k^\top K_{t,n}}{\sqrt{d_g}},
```

```math
\pi_{t,k}=Entmax_{1.5}(\ell_{t,k,:}),
\qquad \sum_n\pi_{t,k,n}=1.
```

Entmax is unchanged and remains sparse. It is not replaced by Softmax and receives no new
temperature or diversity objective.

The immutable anchor remains `A=Z_0`. Dynamic support reads three distinct quantities:

```math
O_{t,k}=\sum_n\pi_{t,k,n}W_OA_n,
```

```math
C_{t,k}=\sum_n\pi_{t,k,n}W_CZ_{t,n},
```

```math
D_{t,k}=\sum_n\pi_{t,k,n}W_D(Z_{t,n}-A_n).
```

Existing context fusion and editor equations are unchanged. With R1b applicability disabled:

```math
\Delta Z_{t,k}=base\_delta_{t,k}.
```

Every candidate remains a same-parent preview:

```math
\widehat Z_{t+1}^{(k)}=Z_t+\Delta Z_{t,k}.
```

Target tensors remain absent from intent, grounding, evidence reading, context, editor, readout,
scorer, selector, STOP, and state transition.

## Tensor contract

Canonical shapes are:

```text
anchor A                    [B,196,256]
current state Z_t           [B,196,256]
fixed intents I             [B,4,256]
current grounding logits    [B,4,196]
current supports pi_t       [B,4,196]
temporal support trace      [B,3,4,196]
grounded evidence           [B,4,256]
contexts                    [B,4,256]
DeltaZ                      [B,4,196,256]
candidate states            [B,4,196,256]
candidate queries           [B,4,512]
scores                      [B,4]
```

`RecurrentStepOutput.raw_spatial_supports` is the actual `Ground(I,Z_t)` result before any
diagnostic control transform. `effective_spatial_supports` records the support presented to the
controlled scorer path. The backward-compatible `spatial_supports` field aliases that effective
value, so CLONE/MEAN consumers keep their historical semantics.
`IAGSRMEOutput.temporal_supports` always stacks raw supports as `[B,T,K,N]`.
`output.supports` remains a backward-compatible alias with explicit t0/initial semantics, and
`output.initial_supports` names that meaning directly.

## Control semantics

All controls reground before applying their intervention:

- `SINGLE-k`: construct current t0 consequences, execute k once, then STOP.
- `REPEAT-k`: force identity k at every live timestep while recomputing `pi[t,k]` from each updated
  parent state. It never freezes `pi[0,k]`.
- `CLONE`: compute current-timestep consequences, then clone candidate 0's consequence and support
  consistently across candidate identities. Raw `Ground(I,Z_t)` remains available separately.
- `MEAN`: compute current-timestep consequences, then average effects/support-related scorer inputs.
- `FULL`: unchanged hard candidate-or-STOP execution.

No control generates candidate k+1 from candidate k's preview. All previews use the current
selected-path parent.

## Temporal WHERE diagnostics

Diagnostics retain live-parent lineage and report separately at t0, t1, and t2:

- support mass, sparsity fraction, entropy, and effective support size;
- between-candidate support cosine and probability overlap matrices;
- same-candidate temporal cosine and probability overlap;
- fixed `TopM=10` Jaccard;
- entropy and effective-size change;
- support L1/L2 displacement;
- argmax-token movement fraction;
- pairwise cosine of candidate displacement vectors
  `d[t,k]=pi[t+1,k]-pi[t,k]`.

Transition statistics are conditioned on the preceding decision:

```text
same candidate k executed
other candidate executed
STOP
```

The STOP group is explicitly an absorbing unchanged-state recomputation, not a live edit. The
unconditional transition population uses samples live before the later timestep.

These measurements distinguish candidate-specific response from all-candidate co-motion. Motion
alone is not evidence of correct semantics.

## Self-induced grounding drift risk

R1c1 creates a feedback path:

```text
editor changes Z_t
→ projected grounding keys change
→ support moves
```

Possible negative outcomes include support jitter, Entmax discontinuities, moving cloned maps,
shared displacement, or state-key drift unrelated to residual applicability. A lower temporal
support cosine is not a success criterion by itself. It must align with functional effects,
selected target-relative utility, repeat controls, and retrieval.

## Preserved mechanisms

R1c1 preserves:

```text
K=4, Tmax=3, width=256, retrieval_dim=512
lambda_z=0.1, query_cap=1000.0
FG-CLIP Base full fine-tuning
FashionIQ original protocol
ordered_and captions
terminal + 0.5 marginal objective
shared context/editor/readout/scorer/selector
hard forward candidate-or-STOP
absorbing STOP
same-parent previews
target firewall
mixed precision and optimizer schedule
```

It adds no dynamic WHAT, semantic residual, teacher, DPP, regularizer, new STOP loss, planning,
candidate-specific grounder, or applicability gate.

## Replay and provenance

R1c1 checkpoints serialize the full `IAGSRMEConfig` including
`enable_dynamic_regrounding=true`, and use:

```text
architecture_generation = r1c1_dynamic_current_state_reground_v1
```

Diagnostic replay requires a self-describing checkpoint, `query_cap=1000`, Entmax grounding,
dynamic regrounding enabled, and applicability disabled. Legacy R0/R1a and R1b configs default a
missing dynamic-regrounding field to false. An R1b checkpoint is never silently replayed as R1c1.

## Trusted R1a comparison

The report embeds these fixed historical reference values:

```text
FULL MR                              38.764146
Deltaq t0/t1/t2                     0.336634 / 0.272417 / 0.197111
retention t1/t0, t2/t0, t2/t1      0.809238 / 0.585534 / 0.723562
selected gain t0/t1/t2              +0.07424 / +0.02488 / -0.00261
support cosine / overlap            0.999842 / 0.995108
repeated-candidate trajectories      0.957447
mean executed edits                  2.86586
```

R1b remains a negative-result reference: its dynamic applicability stayed nearly constant and was
uncorrelated with offline utility. R1b's gate is not active in R1c1.

## Verification status

Synthetic tests verify intent/grounder call counts, t0 R1a parity, deterministic support motion,
immutable anchor, same-parent previews, no applicability, target firewall, temporal trace shape,
Entmax AMP behavior, control semantics, diagnostic conditioning, and checkpoint replay.

The real pinned FG-CLIP CPU smoke verifies one intent call, three grounder calls, finite losses and
gradients, support normalization, t0 anchor-grounding parity, and nonzero state-conditioned support
change. A CUDA FashionIQ canary and full 20-epoch training have not been run in this implementation
environment.

## GPU canary

Run before full training:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/canary_train_iag_srme.py \
  --r1c1 \
  --dataset-root data/FashionIQ \
  --steps 100 \
  --precision fp16
```

The canary requires finite loss/gradients, at least 20 successful optimizer steps, `intent=1`,
`grounder=3`, `applicability=0`, nonzero cumulative grounder gradient, and grounder parameter
movement. It reports per-timestep support statistics, temporal motion, DeltaZ/Deltaq, STOP, and
candidate usage.

## Full training

Only after the GPU canary passes:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r1c1_dynamic_reground \
  experiment=iag_srme_r1c1_dynamic_reground \
  protocol=fashioniq_original
```

## Checkpoint diagnostics

```bash
RUN=<R1C1_RUN_PATH>

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/best.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1c1_dynamic_reground_best.json
```

## Scientific decision contract

Positive evidence requires aligned state-sensitive support, nontrivial candidate-specific rather
than shared displacement, healthy R1a Deltaq retention, improved late selected utility, reduced
repeat equivalence/behavior, and competitive retrieval.

R1c1 is negative or insufficient if grounding remains static, all supports co-move, moving clones
remain functionally equivalent, support jitters while retrieval degrades, R1a Deltaq retention is
destroyed, or late utility/repeat behavior does not improve.

No rescue mechanism will be added on this branch. Dynamic WHAT/R1c2 is explicitly out of scope.

## Post-implementation audit correction

The CUDA canary separates implementation readiness from scientific behavior:

- mechanical failures still abort on non-finite arithmetic, unusable AMP, insufficient successful
  optimizer steps, incorrect intent/grounder/applicability call counts, broken same-parent/support
  invariants, or absent cumulative grounder gradient/parameter movement;
- `never_stop`, candidate monopoly, clone-like effects, high support similarity, weak support
  motion, and related collapse observations are warning-only scientific outcomes;
- near-total STOP at t0 is an operational R1c1 failure only because it prevents the canary from
  exercising `Z0 -> Z1 -> Ground(I,Z1)`, not because STOP is scientifically undesirable.

Failure observations now distinguish `high_support_similarity_t0`, `t1`, and `t2` using raw
per-timestep supports with live-parent lineage. The retained legacy observation is named
`high_support_similarity_t0_legacy` because `output.supports` is t0 only. High t0 similarity is
expected from exact R1a parity and cannot by itself reject dynamic WHERE.

The target-firewall audit checks that `IAGSRMECore.forward` has no target argument, snapshots the
completed target-free rollout, and then evaluates offline selected-path metrics under an original
and permuted target bank. Only offline metrics may change; intents, raw temporal supports,
candidate states/queries, selections, and final state/query must remain unchanged.

Checkpoint provenance uses the precise term **fully self-describing model configuration**. It
means architecture/config replay is exact; it does not claim that the checkpoint alone records
every seed, Git SHA, protocol, optimizer setting, caption policy, or experiment hyperparameter.
