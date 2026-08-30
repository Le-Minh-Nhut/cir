# IAG-SRME R1b: Visual NULL and confidence-preserving grounding

## 1. Hypothesis

R1b tests one hypothesis only: forced unit-mass grounding over real image tokens can force an
inapplicable proposal to execute. A learnable Visual NULL coordinate should allow low real-token
mass, and that mass must reduce the executable edit rather than being normalized away.

## 2. Git lineage

```text
exp/e2e-iag-srme-clean-rewrite
  -> exp/e2e-iag-srme-r0-diagnostic-audit
  -> exp/e2e-iag-srme-r1b-visual-null-confidence-gate
```

R1a has no Git branch. It was R0 code run with `model.query_cap=1000.0`. R1b carries this already
validated baseline condition forward; changing the cap is not the R1b intervention.

## 3. R1a baseline

The trusted R1a checkpoint is
`outputs/2026-08-30/21-34-29/best_r1a_self_describing.pt`, epoch 3, FashionIQ-original Mean
Recall 38.764146. Mean Delta-q norms were 0.336634, 0.272417, and 0.197111 at t0/t1/t2. Mean
executed edits were 2.86586, and selected target-relative gain at t2 was approximately -0.00261.
R1b must be compared to this cap-free R1a condition, not R0's cap 0.5 condition.

## 4. Exact intervention

The only scientific change is one learnable Visual NULL logit added to the static anchor grounder,
plus preservation of `1-p_null` as execution confidence. Intent generation, immutable-anchor
grounding, recurrence, K=4, Tmax=3, context, readout, scorer, selector, STOP, losses, backbone,
data, and optimizer are unchanged.

## 5. Pre-implementation code audit

### WHAT

1. Candidate queries are `TextIntentEncoder.query_bank`, shape `[K,d]`.
2. `MultiheadAttention` reads every valid contextual text token `[B,L,d]` independently for each
   query.
3. Residual attention, FFN residual, and LayerNorm produce stable semantic intents.
4. Intent shape is `[B,K,d]`, canonically `[B,4,256]`.

### WHERE

5. `AnchorGrounder` computes projected intent/anchor dot products.
6. Legacy visual logits are `[B,K,N]`.
7. Normalization is over the final position axis, independently per candidate.
8. The operator is Entmax-1.5 in an FP32 AMP island.
9. Before R1b it normalized over only N visual tokens.
10. Each legacy support summed to one.
11. Legacy support shape was `[B,K,N]`.

### Downstream support path

12. `GroundedStateReader` pools projected A, Zt, and Zt-A with support.
13. These pooled values enter `GroundedEditContext`.
14. `SharedTokenEditor` uses support as the token-local gate.
15. The legacy editor divided support by its per-candidate maximum. `ConsequenceScorer` performed
    the same max scaling for pooled-delta features. No other runtime support normalization exists.
16. Repository search found no L1 normalization, softmax, Entmax, or mask renormalization after
    grounding besides those two max scalings.

### EDIT and recurrence

17. Legacy execution was `DeltaZ=lambda_z*(P/max(P))*tanh(direction)`.
18. `lambda_z` is applied in `SharedTokenEditor` after direction construction.
19. Because of max scaling, legacy total support magnitude did not control edit magnitude.
20. Support controlled spatial shape/locality only.
21. Grounding is computed exactly once before recurrence and is static.
22. Every candidate state is asserted equal to `Zt + DeltaZ_t,k`, so previews share one parent.

### Loss, diagnostics, and checkpoint

23. Core terminal/marginal losses do not directly consume support. Optional factorization consumes
    anchor-grounded evidence indirectly; conditional spatial reading preserves its former scale.
24. No loss directly assumes support sums to one.
25. R0 entropy/effective-size/overlap diagnostics assumed unit mass and required repair for R1b.
26. Legacy checkpoints stored backbone identity and precision but not complete model configuration.
27. R1b checkpoints now serialize the complete `IAGSRMEConfig`. The diagnostic loader infers
    state-dict-visible legacy values, preserves legacy architecture, and requires R1b checkpoints
    to be self-describing.

## 6. Mathematical formulation

```math
\ell^{vis}_{k,n}=\frac{(W_Q e_k)^T(W_K A_n)}{\sqrt{d_g}},
```

```math
\ell^{null}_k=\frac{(W_Qe_k)^Tk_{null}}{\sqrt{d_g}}+b_{null},
```

```math
P^{full}_k=\operatorname{Entmax}_{1.5}
([\ell^{vis}_{k,1:N},\ell^{null}_k]),
```

```math
p_k^{null}=P^{full}_{k,N+1},\qquad
c_k=1-p_k^{null},\qquad
P^{vis}_{k,n}=P^{full}_{k,n}.
```

The implementation asserts

```math
\sum_nP^{vis}_{k,n}=c_k=1-p_k^{null}.
```

For content reading only,

```math
\widetilde P_{k,n}=\frac{P^{vis}_{k,n}}{c_k+\epsilon}.
```

The legacy spatial gate remains

```math
S_{k,n}=\frac{\widetilde P_{k,n}}
{\max_j\widetilde P_{k,j}+\epsilon}.
```

Execution is now

```math
\Delta Z_{t,k,n}
=\lambda_z\,c_k\,S_{k,n}\,
\tanh(u_{t,k,n}).
```

Confidence appears exactly once.

## 7. Tensor contracts

```text
visual logits             [B,K,N]
NULL logits               [B,K]
augmented logits          [B,K,N+1]
full Entmax probability   [B,K,N+1]
real visual support       [B,K,N]
p_null                    [B,K]
visual confidence         [B,K]
conditional support       [B,K,N]
DeltaZ                    [B,K,N,d]
```

## 8. NULL parameterization and initialization

One shared learnable key `visual_null_key[d_g]` and scalar bias produce candidate-conditioned NULL
compatibility from the existing projected intent. They use no target, labels, state, timestep, or
future information. The key starts at zero and the bias defaults to zero. Projected-dot-product
visual logits are centered around a comparable zero scale, so NULL starts available without being
strongly preferred. No candidate-specific NULL network is introduced.

## 9. Confidence propagation and no-renormalization audit

`P_visual` is retained in `IAGSRMEOutput.supports`; it is never divided back to unit mass in the
execution path. The reader receives `P_tilde` so grounded semantic content does not shrink and
confound the experiment. The editor's pre-existing max normalization operates only on `P_tilde`,
then multiplies the separately preserved `c` once. Thus max scaling cannot erase confidence.
The scorer receives the conditional shape and the already confidence-scaled DeltaZ/Delta-q
consequences. Diagnostic shape normalization is offline-only.

## 10. Smoke and invariant tests

Tests cover NULL-dominant and real-token-dominant logits, identical spatial shape at different
confidence, monotonic edit shrinkage, exact mass conservation, nonzero NULL gradients,
same-parent previews, one grounding call per rollout, the R1a cap condition, checkpoint replay,
legacy behavior, conditional diagnostic validity, and target-firewall-safe utility analysis.

Synthetic command:

```bash
PYTHONPATH=src python src/smoke_iag_srme.py --r1b --diagnostics
```

The real pinned FG-CLIP CPU smoke (`--max-steps 3 --r1b`) completed one optimizer step with
finite terminal/marginal/total losses `0.409738/0.001266/0.410371`. NULL key/bias gradient norms
were `1.596e-6` and `3.042e-6`; their parameter deltas were nonzero. Real support mass error was
`4.172e-7`, grounder calls were exactly one, and all three recurrent steps had the expected
`[2,4,196,256]` DeltaZ and `[2,4,512]` candidate-query shapes. CUDA was unavailable in the
implementation environment, so this CPU smoke does not replace the required FP16 GPU canary.

## 11. Checkpoint reproducibility and replay guard

Every new checkpoint stores all fields of `IAGSRMEConfig`, including query cap, NULL enablement,
NULL initialization, and normalization family. Standalone evaluation rejects a stored/configured
model-config mismatch. R1b diagnostics reject a NULL state dict without self-describing config and
require cap 1000, Visual NULL enabled, and replayed FULL Mean Recall within `1e-4` of the saved
metric. Legacy R0/R1a checkpoints instantiate the legacy no-NULL architecture.

## 12. Training and diagnostic commands

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 python src/train.py \
  dataset.root=data/fashionIQ_dataset \
  model=iag_srme_r1b_visual_null \
  experiment=iag_srme_r1b_visual_null \
  protocol=fashioniq_original
```

```bash
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint outputs/<R1B_RUN>/best.pt \
  --dataset-root data/fashionIQ_dataset \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1b_visual_null_best_original.json
```

CUDA canary:

```bash
python src/canary_train_iag_srme.py \
  --dataset-root data/fashionIQ_dataset \
  --steps 100 \
  --precision fp16
```

## 13. Diagnostic contract

R1b preserves every R0 retrieval, WHAT, WHERE, functional, policy, STOP, lineage, and selected-path
metric. It adds global/per-candidate NULL distributions, thresholds, visual confidence, selected
versus non-selected NULL, selection-conditioned timestep summaries, NULL-vs-DeltaZ/Delta-q bins,
NULL-vs-offline target utility, and confidence/STOP associations.

Real support mass and conditional spatial entropy/effective size/cosine/overlap are separate.
Zero-mass candidates have no conditional spatial shape and are excluded rather than reported as
entropy zero or effective size one.

## 14. Identifiability red-team

- NULL always zero: reported as `null_effectively_ignored`.
- NULL always one: reported as `null_globally_dominant` and compared with reference behavior.
- downstream confidence erasure: ruled out structurally and tested by monotonic DeltaZ shrinkage.
- selector avoidance: selected/non-selected NULL and selection probabilities are reported.
- candidate identity shortcut: per-candidate NULL distributions and their range are reported.
- query difficulty shortcut: NULL/utility association is offline and reported without entering
  forward computation.

## 15. Training curve, best checkpoint, and results

Full FashionIQ training has not been executed in this implementation environment. Populate after
the local CUDA run:

```text
training curve: pending
best epoch: pending
best Mean Recall: pending
best checkpoint: pending
replayed metric error: pending
```

Required result table remains pending: FULL/REFERENCE/SINGLE/REPEAT/MEAN, same-parent retrieval,
NULL statistics, NULL/effect bins, NULL/utility bins, STOP/depth, DeltaZ, Delta-q, retention, and
candidate redundancy.

## 16. Scientific decision

Current verdict: **INCONCLUSIVE**. Mechanical implementation, gradient, locality, mass, replay,
and target-firewall checks pass, but no full trained R1b checkpoint exists here. Therefore this
branch cannot yet answer whether Visual NULL reduces unnecessary late editing.

R1b is not designed to solve candidate specialization collapse. Any change in support/Delta-q
cosine is secondary and must not be conflated with the applicability hypothesis.

## 17. Next experiment

If R1b produces aligned causal evidence, the next separate experiment may be R1c-1 with fixed
WHAT and dynamic `Ground(I_k,Z_t)`. It is explicitly not implemented on this branch. If R1b fails,
report whether NULL was ignored, globally dominant, confidence-unrelated, or insufficiently
state-dependent; do not add losses to rescue this experiment.
