# IAG-SRME R1c2: Dynamic Current-State WHAT Reproposal

Status: **IMPLEMENTATION READY — SCIENTIFIC RESULT PENDING**

## 1. Lineage and causal question

R1c2 branches from the frozen R1c1 implementation. R1c1 was mechanically correct but
scientifically negative: BEST FashionIQ-original Mean Recall was `38.754133`, essentially equal to
R1a `38.764146`. Its support cosine remained `0.999766 / 0.999755 / 0.999740`, showing dynamic
support motion without candidate-specific WHERE. This is the observed **moving clones** failure.

R1c2 tests one hypothesis only:

> Fixed, highly correlated WHAT intents prevent current-state dynamic WHERE from becoming
> candidate-specific. Can each persistent candidate reconsider the modification text from the
> actual edited state and re-propose its remaining semantic action?

The causal delta is:

```text
R1c2 - R1c1 = shared current-state dynamic WHAT residual at t1/t2
```

No loss, optimizer, STOP, editor, scorer, readout, backbone, or dataset change is included.

## 2. Exact mechanism

The text-only intent encoder runs once:

\[
B_k = IntentEncoder(q_k,T).
\]

`B_k` is immutable. At `t=0` the new module is bypassed exactly:

\[
I_{0,k}=B_k.
\]

For `t>0`, the same shared module is used for every candidate and timestep:

\[
E^{state}_{t,k}=CrossAttn(B_k,Z_t),
\]

\[
E^{change}_{t,k}=CrossAttn(B_k,Z_t-A),
\]

\[
Q^{text}_{t,k}=B_k+W_s[E^{state}_{t,k};E^{change}_{t,k}],
\]

\[
E^{text}_{t,k}=CrossAttn(Q^{text}_{t,k},T,M_T),
\]

\[
H_{t,k}=SiLU\left(W_h LN[B_k;E^{state}_{t,k};E^{change}_{t,k};E^{text}_{t,k}]\right),
\]

\[
\Delta I_{t,k}=W_{out}H_{t,k},
\qquad
I_{t,k}=B_k+\Delta I_{t,k}.
\]

Current-state dynamic WHERE remains the unchanged R1c1 Entmax path:

\[
\ell_{t,k,n}=\frac{(W_QI_{t,k})^T(W_KZ_{t,n})}{\sqrt d},
\qquad
\pi_{t,k}=Entmax_{1.5}(\ell_{t,k,:}).
\]

The existing reader, context, editor, readout, scorer, hard candidate/STOP selector, and state
transition consume `I_t`, `pi_t`, and the same selected-path parent as before.

## 3. Zero-init identity and gradient contract

`W_out.weight` and `W_out.bias` are initialized to exact zero. Consequently:

\[
\Delta I_{t,k}=0,
\qquad
I_{t,k}=B_k
\]

at initialization, including t1/t2 after state changes. No post-addition normalization modifies
the base intent, so zero residual gives exact identity.

On the first update, `W_out` can receive gradient while upstream evidence/text-read layers may have
zero gradient because `dL/dH = W_out^T dL/dDeltaI`. Once `W_out` moves, gradients can reach those
upstream layers. Canary readiness therefore checks cumulative output and upstream gradient and
parameter movement rather than incorrectly requiring first-step upstream gradient.

## 4. Tensor contract

```text
base intents B                  [B,4,256]
current state Z_t               [B,196,256]
accumulated change Z_t-A        [B,196,256]
state evidence                  [B,4,256]
change evidence                 [B,4,256]
text evidence                   [B,4,256]
intent residual                 [B,4,256]
current intents I_t             [B,4,256]
temporal intents                [B,3,4,256]
temporal supports               [B,3,4,196]
candidate token states          [B,4,196,256]
candidate retrieval queries     [B,4,512]
```

`output.intents` and `output.initial_intents` are the immutable base/t0 intents.
`output.temporal_intents` stacks raw WHAT, before any diagnostic control. Per-step
`current_intents` and `intent_residual` expose the same raw values. Existing raw/effective support
semantics remain unchanged; temporal supports always represent raw `Ground(I_t,Z_t)`.

## 5. Scientific invariants

- t0 R1c2 WHAT equals R1c1 WHAT exactly.
- Because `Z0=A`, t0 R1c2 WHERE equals R1c1 WHERE exactly.
- base intent encoder calls once; reproposal calls twice; grounder calls three times for `Tmax=3`.
- all candidates at a timestep inspect the same actual selected-path `Z_t`.
- all candidate previews equal `Z_t + DeltaZ_t,k`; no preview consumes a peer preview.
- candidate identity k persists through recurrence.
- query cap remains `1000`, applicability/Visual NULL remain disabled, Entmax15 remains enabled.
- model and reproposal forward signatures contain no target input.

## 6. Control semantics

- `FULL`: dynamic WHAT plus dynamic WHERE and normal hard inference policy.
- `frozen_t0_what`: holds `I_t,k=B_k` while still recomputing `Ground(B_k,Z_t)` from the actual
  current state. This is the trained-checkpoint R1c1-like inference control.
- `REPEAT-k`: forces persistent identity k, but still re-proposes its WHAT and regrounds it at each
  later live timestep.
- `SINGLE-k`: executes t0 candidate k once, then STOP; only base WHAT is executed.
- `MEAN` and `CLONE`: transform current consequences after raw dynamic WHAT/WHERE is constructed;
  raw temporal intent/support traces remain available separately.
- `REFERENCE_ONLY` and offline candidate oracle retain their existing evaluation semantics.

No target is used by any control forward.

## 7. Diagnostics

Temporal WHAT diagnostics are live-lineage conditioned and report:

- per-timestep intent cosine matrix, off-diagonal cosine, candidate norms, and residual norms;
- same-candidate temporal intent cosine and L1/L2 movement;
- candidate displacement matrix for `I[t+1,k]-I[t,k]`, detecting moving WHAT clones;
- movement conditioned on same candidate executed, other candidate executed, or STOP;
- a concise WHAT -> WHERE -> DeltaZ -> Deltaq chain per timestep.

All R1c1 temporal WHERE, functional, selection, retrieval-control, same-parent oracle, and offline
target-relative diagnostics remain. The report includes fixed R1a and R1c1 historical comparisons;
those constants never enter training or checkpoint selection.

## 8. Target firewall

Targets are absent from base intent construction, state/change reads, token-level text re-reading,
reproposal, grounding, context, editing, candidate construction, scoring, STOP, and state commit.
Targets are consumed only after the complete target-free rollout for retrieval metrics, oracle
analysis, and target-relative diagnostic utility. Tests snapshot the rollout and verify that a
permuted offline target bank can change only diagnostic values.

## 9. PASS and failure patterns

Positive evidence requires aligned results: state-sensitive and candidate-specific WHAT movement;
downstream candidate-specific WHERE/effects; healthy Deltaq magnitude; FULL benefit over
`frozen_t0_what` and trivial controls; and competitive retrieval.

Explicit negative outcomes are retained without rescue:

- **Static WHAT:** reproposal is ignored.
- **Moving WHAT clones:** all intent displacement vectors co-move.
- **WHAT diverges, WHERE clones:** grounder is insensitive to semantic differences.
- **WHAT/WHERE diverge, effects clone:** bottleneck is downstream.
- **Destructive jitter:** similarities decrease while retrieval/utility degrade.
- **STOP/scorer confound:** representation improves but policy behavior fails.

R2, DPP, VISReg, teachers, planning, and STOP repair are outside this branch.

## 10. Verification status

Synthetic tests cover exact t0 parity, zero-init identity, state/text dependence, WHAT-to-WHERE
causality, selected-lineage inputs, `frozen_t0_what`, REPEAT, raw control traces, target firewall,
two-stage gradient activation, config/replay safety, and temporal diagnostics.

The deterministic CPU smoke verifies one intent call, two reproposal calls, three grounder calls,
zero applicability calls, exact zero-init/t0 parity, support normalization, same-parent previews,
finite loss/gradients, and reproposal/grounder parameter movement.

CUDA canary and full FashionIQ training remain pending.

## 11. Commands

CUDA canary:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/canary_train_iag_srme.py \
  --r1c2 \
  --dataset-root data/FashionIQ \
  --steps 100 \
  --precision fp16
```

Full training only after canary PASS:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r1c2_dynamic_reproposal \
  experiment=iag_srme_r1c2_dynamic_reproposal \
  protocol=fashioniq_original
```

BEST diagnostic:

```bash
RUN=<R1C2_RUN_PATH>
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/best.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1c2_dynamic_reproposal_best.json
```

Run the same command with `last.pt` and a distinct output filename to audit BEST-to-LATE drift.
