# R2 Semantic Residual / Claim Firewall — Corrected v2 Contract

Status: **IMPLEMENTATION READY — SCIENTIFIC RESULT PENDING**

## Hypothesis and lineage

R1a established that recurrent depth is useful once `query_cap=1000`. R1b did not
learn meaningful scalar abstention. R1c1 produced moving-WHERE clones. R1c2's
unrestricted full-text reproposal produced moving-WHAT clones and was harmful relative
to its same-checkpoint frozen-WHAT control. R2 tests one new structural hypothesis:
future proposals need an explicit record of semantic evidence already claimed by the
selected action.

The implementation parent is the clean pre-report R1c2 code lineage, configured as the
R1c1/R1a computation family: dynamic current-state Entmax WHERE remains enabled;
R1c2 reproposal and R1b applicability are disabled. No R1c2 full-text reread remains in
the executable path.

## Mathematical contract

For valid content-token mask `M`, initialize

\[
\rho_{0,m}=M_m.
\]

The shared claim module sees candidate query `q_k`, remaining token evidence, and a
target-free current-state summary:

\[
h_{t,k,m}=LN(W_q q_k + W_T(\rho_{t,m}T_m)+W_Z\,mean_n Z_{t,n}),
\]

\[
\alpha_{t,k,m}=M_m\,\sigma(w_\alpha^T GELU(h_{t,k,m})+b_\alpha).
\]

The original v1 contract incorrectly reused `alpha` as consumption strength. With
`alpha=0.99`, this gave `rho: 1 -> 0.01 -> 0.0001 -> 0.000001`: full-text read parity
and conservative residual consumption were mathematically incompatible. R2-v2
separates read allocation from predicted satisfaction. A second minimal shared
projection of the same hidden state produces

\[
\gamma_{t,k,m}=M_m\,\sigma(w_\gamma^T GELU(h_{t,k,m})+b_\gamma).
\]

Here `alpha` means how strongly an action reads/claims evidence; `gamma` means how much
of claimed evidence it predicts has been satisfied after execution. No target or new
loss supervises either head.

Claims are bounded independently; they are not softmax-normalized over tokens. Padding
claims and consumption are exactly zero. Let

\[
w_{t,k,m}=\alpha_{t,k,m}\rho_{t,m},
\]

\[
d_{t,k}=\frac{\sum_mw_{t,k,m}T_m}{\sum_mw_{t,k,m}+\epsilon},
\qquad
s_{t,k}=\frac{\sum_mw_{t,k,m}}{\sum_mM_m+\epsilon},
\]

and magnitude-aware executable content is

\[
c_{t,k}=s_{t,k}d_{t,k}.
\]

Thus uniform residual scaling preserves semantic direction while scaling semantic mass
and executable content. In particular, `rho -> 0` implies `c -> 0`; v1's normalized-only
pooling did not satisfy this invariant.

The existing shared intent encoder receives only tokens weighted by
`alpha*rho`. The candidate retrieval readout receives `c_t,k`, never the unrestricted
full-text global embedding. Current-state dynamic grounding is unchanged:

\[
\pi_{t,k}=Entmax_{1.5}(Ground(I_{t,k},Z_t)).
\]

All K previews share the same `(Z_t,rho_t)`. Candidate residual previews are

\[
\widehat\rho^{(k)}_{t+1}=\rho_t(1-\alpha_{t,k}\gamma_{t,k}).
\]

Only after selection is the semantic state committed:

\[
\rho_{t+1}=\rho_t(1-\alpha_{t,a_t}\gamma_{t,a_t})
\]

for a selected edit. STOP keeps `rho` and `Z` unchanged. The selected visual state still
satisfies

\[
\widehat Z^{(k)}_{t+1}=Z_t+\Delta Z_{t,k}.
\]

Claim/consumption sigmoid, magnitude-aware pooling, residual preview, and selected update run in FP32 under
AMP. The residual is bounded and monotone by construction.

The replay identity is `r2_semantic_residual_claim_firewall_v2`. Checkpoints from the
invalid v1 coupling lack the consumption projection and must fail loudly rather than be
reconstructed as v2.

## Initialization and t0 parity

The claim projection weight is zero initialized and its bias is
`logit(0.99)`. Thus every valid token begins with claim 0.99, giving a candidate-weighted
token sequence close to the parent full-text sequence without random candidate-specific
shock. The consumption projection is independently zero initialized with bias
`logit(0.05)`. Initial effective consumption is therefore `0.99*0.05=0.0495`, giving
`rho1/rho0=0.9505` rather than 0.01. Both output heads receive gradient immediately;
their zero weights initially block shared hidden/projection gradients, which emerge after
the output heads move. This is close, not exact, t0 parent parity.

## Full-text highway audit

- `TextIntentEncoder`: in R2 it receives candidate-specific `alpha*rho` token weights.
- R1c2 `DynamicIntentReproposal`: disabled; its unrestricted token reread cannot execute.
- candidate `TokenStateReadout`: receives per-candidate claimed semantic content.
- context, editor, scorer, selector, and STOP: have no direct text-token/global input.
- root readout: uses the existing full `text_global` only at `Z0=A`; its displacement is
  exactly zero, so this path preserves initialization bookkeeping without contributing a
  semantic retrieval update.
- `no_claim_firewall`: explicitly labeled diagnostic-only parent-path intervention.
- targets: enter only objective/evaluation/offline diagnostics after rollout exists.

## Controls

- `FULL`: claim, selected consumption, residual-conditioned proposals, dynamic WHERE.
- `frozen_residual`: claims are computed but `rho_t=rho_0`; isolates consumption.
- `no_claim_firewall`: same checkpoint, diagnostic-only full-semantic executable bypass.
- `residual_shuffle`: cross-sample rho permutation after t0, target-free.
- `claim_swap`: cyclic candidate claim swap before executable pooling; raw claims remain
  available.
- `SINGLE-k`, `REPEAT-k`, `MEAN`, `REFERENCE`, and same-parent retrieval retain their
  established semantics. REPEAT recomputes claims and WHERE from the actual parent state.

## Diagnostics and causal interpretation

The JSON report adds `semantic_residual_diagnostics` with `rho0..rho3` distributions,
residual L1 mass, fractions near zero/one, claim mass/entropy/effective size,
semantic mass/content norm, and separate claim, gamma, and alpha-gamma cosine, selected
consumed mass/token count, and no/near-total-consumption fractions. All timestep metrics
use live-parent lineage; consumption includes selected non-STOP actions only.

The decisive same-checkpoint comparison is `FULL` versus `frozen_residual`. Claim motion
alone is insufficient: claim swap/removal/shuffle must affect functional effects and
retrieval. Offline target utility is computed only after target-free traces exist.

Warnings to audit include no consumption, consume-all-first, cloned claims, global
consumption, residual ignored, token-position shortcuts, and representational claims
whose DeltaZ/Deltaq remain cloned. These are scientific observations and do not abort a
mechanically valid canary.

## Test and canary contract

Focused tests cover residual initialization/padding, monotonicity, selected-only commit,
same-parent rho, frozen residual, claim firewall, claim swap, t0 parent proximity,
target-free signatures, FP32 AMP arithmetic, checkpoint replay, JSON diagnostics, and
the expected two-stage zero-initialized gradient path. The synthetic smoke reports
shapes, t0 errors, rho trajectory, gradient emergence, parameter movement, finite loss,
same-parent equality, and target firewall.

CUDA canary:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/canary_train_iag_srme.py \
  --r2 \
  --dataset-root data/FashionIQ \
  --steps 100 \
  --precision fp16
```

The canary hard-fails numerical/lineage/call-count/gradient/movement errors. Claim clone,
global consumption, consume-all, residual-unused, candidate monopoly, and never-STOP are
warning-only scientific outcomes.

## Training and diagnosis

Run training only after CUDA canary PASS:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r2_semantic_residual \
  experiment=iag_srme_r2_semantic_residual \
  protocol=fashioniq_original
```

```bash
RUN=<r2-output-directory>
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/best.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r2_semantic_residual_best.json
```

Repeat for `last.pt` as `reports/r2_semantic_residual_last.json`.

## PASS and kill criteria

A PASS requires aligned evidence: selective consumption, later proposals responding to
rho, claim interventions changing executable effects, FULL exceeding frozen residual,
weaker common-mode semantics/effects, healthy recurrent Deltaq, and competitive
retrieval. R2 is negative if rho is unused or globally exhausted, claims remain clones,
rho does not affect later WHAT, functional effects remain clones, frozen residual is at
least as good as FULL, trivial controls remain equivalent, or R1a's recurrent effect
survival is destroyed. A negative run must be frozen; no DPP, diversity loss, teacher,
or STOP rescue belongs in this branch.
