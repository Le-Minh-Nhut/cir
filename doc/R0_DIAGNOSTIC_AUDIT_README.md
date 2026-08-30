# R0 IAG-SRME diagnostic audit

R0 is a measurement-only audit of existing IAG-SRME checkpoints. It does not alter the model,
training objective, recurrence, selector, support computation, or FashionIQ protocol. Its purpose
is to localize where candidate distinction disappears along:

```text
TEXT → INTENT/WHAT → GROUNDING/WHERE → CONTEXT → ΔZ → Δq
     → SCORE/SELECTION → SELECTED ROLLOUT
```

The JSON report includes a machine-readable `diagnostic_definitions` section. Each definition
states its equation, population, input shape, reduction axes, and interpretation limitation.

## Tensor and lineage contract

Canonical dimensions are:

```text
B = validation queries in the current batch/population
K = 4 candidate identities
T = 3 recurrent decisions
N = visual patch tokens
d = 256 token/state width
D = retrieval dimension

I                 [B,K,d]
P                 [B,K,N]
C_t               [B_live(t),K,d]
ΔZ_t              [B_live(t),K,N,d]
q_t               [B_live(t),D]
qhat_t+1           [B_live(t),K,D]
Δq_t               [B_live(t),K,D]
```

`B_live(t)` is not the full validation size at late timesteps. It contains only trajectories live
before the decision at timestep `t`. Every timestep-dependent JSON object records its raw
`live_parent_count` or `selected_non_stop_transition_count` and a `metric_population` string.

## WHAT

For every query, R0 computes the full matrix

```math
S^I_{ij} = \cos(I_i,I_j).
```

The candidate axis is never pooled before comparison. The report contains the mean K×K matrix,
off-diagonal mean/min/max, and per-candidate intent norm. These values measure representation
similarity; they do not prove that an intent is semantically correct.

## WHERE

Each static anchor support is a probability vector `P_k ∈ R^N`. R0 reports:

```math
H(P_k) = -\sum_n P_{k,n}\log P_{k,n},
```

```math
N_{eff}(P_k) = \exp(H(P_k)),
```

the exact-positive support fraction, K×K support cosine, and probability overlap

```math
O_{ij}=\sum_n \min(P_{i,n},P_{j,n}).
```

The heuristic dominant tokenwise grounding mass share is

```math
\frac{\sum_n\max_k P_{k,n}}{\sum_{k,n}P_{k,n}}.
```

It is a concentration statistic only. It is not semantic ownership. Current supports are computed
once by the architecture, so the report states `support_static_by_current_architecture=true` and
does not present static support as an empirically measured recurrence-stability result.

## CONTEXT, true ΔZ, and true Δq

Context cosine is reported separately at `t0`, `t1`, and `t2` over live parents. Any matrix pooled
across timesteps is explicitly marked as a secondary aggregate.

For every step, R0 verifies exactly:

```math
\hat Z^{(k)}_{t+1}=Z_t+\Delta Z_{t,k}.
```

For ΔZ cosine and effective rank, only spatial and channel axes are flattened:

```text
[B_live,K,N,d] → [B_live,K,N*d].
```

The candidate axis remains intact.

R0 also verifies the actual functional effect definition:

```math
\Delta q_{t,k}=\hat q^{(k)}_{t+1}-q_t,
```

where all `K` candidate queries use the same parent `q_t/Z_t`. It reports candidate-wise norms,
mean and median norm by timestep, full K×K cosine matrices, and the uncentered effective rank

```math
r_{eff}=\exp\left(-\sum_j \pi_j\log\pi_j\right),
\qquad
\pi_j=\frac{\sigma_j}{\sum_l\sigma_l},
```

using singular values of the candidate-effect matrix.

Late-step effect retention is descriptive:

```math
R_{1/0}=\frac{E\|\Delta q_1\|}{E\|\Delta q_0\|+10^{-8}},
\qquad
R_{2/0}=\frac{E\|\Delta q_2\|}{E\|\Delta q_0\|+10^{-8}},
\qquad
R_{2/1}=\frac{E\|\Delta q_2\|}{E\|\Delta q_1\|+10^{-8}}.
```

Each expectation uses that timestep's live population, so these ratios are not causal estimates.
No universal collapse threshold is assigned.

## Same-parent counterfactuals versus selected path

`same_parent_counterfactual_diagnostics` evaluates all four previews from a common live parent:

```text
Z_t → {Zhat_t+1,0, Zhat_t+1,1, Zhat_t+1,2, Zhat_t+1,3}.
```

For each timestep it reports per-candidate FashionIQ retrieval, an offline best-candidate oracle,
the mean-candidate query control, and the exact live-parent count. Target labels rank consequences
offline only; they never select or construct a candidate.

`selected_path_marginal_diagnostics` instead observes only the action actually executed:

```math
\Delta s_t = \cos(q_{t+1},y)-\cos(q_t,y).
```

It excludes STOP and reports the full distribution by timestep. This says whether the executed
transition moved toward or away from the target representation; it cannot explain why the
target-free scorer selected that action.

Dynamic recurrence differences (`g_t`, `d_t`, context, ΔZ, candidate query, score) use only parents
that executed a non-STOP edit at the earlier step. Element counts and live executed-parent counts
are both reported.

## Selection and STOP

Two STOP populations are deliberately separate:

```math
occupancy_t = \frac{\#\{action_t=STOP\}}{\#\{all\ trajectories\}},
```

which includes already absorbed trajectories, and

```math
hazard_t = \frac{\#\{new\ STOP\ at\ t\}}{\#\{live\ parents\ before\ t\}}.
```

The report also provides raw live counts, new-stop counts, occupancy counts, candidate counts,
candidate+STOP distribution among live decisions, executed edit count, and repeated-candidate
trajectory count. STOP is never interpreted as semantic NULL or factor inactivity.

## Retrieval controls

All controls use the unchanged FashionIQ reference filtering rule.

- `FULL`: normal selected recurrent rollout.
- `REFERENCE_ONLY`: pretrained reference global representation.
- `SINGLE_k`: execute candidate `k` once from the root, then conceptually STOP.
- `REPEAT_k`: execute candidate `k` through the real recurrence for all three steps; every edit
  reads the updated state rather than multiplying a root effect.
- `MEAN_CANDIDATE`: existing matched-compute mean-candidate control.

Every control reports R@10, R@50, and Mean Recall. Ratios include best SINGLE/FULL, best
REPEAT/FULL, MEAN/FULL, and REFERENCE_ONLY/FULL. A ratio is functional evidence, not a causal
explanation of architecture behavior.

## Target firewall

Reference pixels and modification text produce the complete model rollout before R0 accesses any
target ID or target gallery feature. Targets are consumed only by offline retrieval and selected
transition evaluation. They never enter intent, grounding, context, editor, readout, scorer,
selector, candidate construction, or recurrent state.

## Run R0

```bash
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint outputs/<RUN>/best_original.pt \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r0_best_original_fashioniq_original.json
```

The CLI also accepts `--protocol fashioniq_val` without retraining or changing the checkpoint.

## Reading the JSON

Primary sections are:

```text
intent_diagnostics                         WHAT
grounding_diagnostics                      WHERE
functional_diagnostics.per_timestep        context, ΔZ, Δq
dynamic_diagnostics                        selected-lineage recurrence changes
selection_diagnostics                      actions and STOP
control_retrieval_metrics                  FULL/reference/SINGLE/REPEAT/MEAN
same_parent_counterfactual_diagnostics     all K previews from one parent
selected_path_marginal_diagnostics         actual executed transition
specialization_matrices                    secondary aggregate matrices
failure_flags                              thresholded observations with raw evidence
diagnostic_definitions                     machine-readable metric contracts
```

Failure flags contain their exact condition, threshold, supporting raw values, and interpretation
limit. They are observations, not causal diagnoses.

## What R0 can and cannot establish

R0 can establish numerical facts such as high candidate similarity at a particular stage,
late-step effect attenuation within the correct live population, a selected transition's offline
target-relative direction, and whether a functional control outperforms FULL.

R0 cannot establish semantic correctness, semantic ownership, the true number of edits, a causal
reason for collapse, or which architectural change should be made next. Those require controlled
follow-up experiments outside this branch.
