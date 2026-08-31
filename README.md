# CIR IAG-SRME R1c1 — Dynamic Current-State Re-Grounding Diagnostic README

**Date:** 2026-08-31  
**Branch:** `exp/e2e-iag-srme-r1c1-dynamic-reground`  
**Architecture generation:** `r1c1_dynamic_current_state_reground_v1`  
**Dataset / protocol:** FashionIQ original  
**Backbone:** `qihoo360/fg-clip-base`  
**Backbone revision:** `454d76372c2cf5eb48fa0d871fd0534481484d97`  
**Training precision:** FP16  
**K:** 4 candidates  
**Tmax:** 3 recurrent decisions  
**query_cap:** 1000.0  
**Grounding normalization:** Entmax-1.5  
**Dynamic applicability:** OFF  
**Visual NULL:** OFF  

---

# 1. Executive conclusion

R1c1 was designed as a clean causal test of the hypothesis:

> The main remaining collapse after R1a is caused by reusing a static visual WHERE across recurrent steps. If each fixed text-derived action is re-grounded on the current edited visual state `Z_t`, candidate specialization and recurrent utility should improve.

The intervention was deliberately narrow:

```text
R1a
+
fixed WHAT
+
dynamic current-state WHERE
```

with:

\[
I_k = \operatorname{Intent}(q_k,T)
\]

computed once per rollout, and:

\[
\pi_{t,k}
=
\operatorname{Ground}(I_k,Z_t)
\]

recomputed at every timestep.

No dynamic re-proposal, diversity regularizer, teacher, RDMReg, VISReg, DPP, semantic residual, RL, or new STOP objective was added.

The implementation is mechanically valid. The GPU canary passed:

```text
attempted steps:                         100
successful optimizer steps:              96
AMP-overflow skipped steps:               4
grounder nonzero-gradient fraction:     1.0
grounder parameter movement:            yes
dynamic regrounding:                    true
dynamic applicability:                  false
mechanical_status:                      PASS
```

However, the scientific result is negative.

The best checkpoint reached:

```text
epoch:        4
Mean Recall: 38.754133
R@10:        27.583865
R@50:        49.924401
```

which is effectively identical to R1a:

```text
R1a best Mean Recall: 38.764146
R1c1 best Mean Recall: 38.754133
difference:             -0.010013
```

Therefore dynamic current-state re-grounding alone produced no meaningful retrieval gain.

More importantly, the intended candidate-specific WHERE specialization did not emerge.

At the best checkpoint:

```text
between-candidate support cosine:
t0 = 0.999766
t1 = 0.999755
t2 = 0.999740
```

The four candidate supports remain almost identical at every timestep.

At the late checkpoint, the model does learn more state-dependent support movement, but the movement is common-mode rather than candidate-specific. The supports move together, while candidate effects become even more parallel.

The correct diagnosis is therefore:

\[
\boxed{
\text{R1c1 converts static clones into moving clones rather than solving candidate collapse.}
}
\]

A second independent failure appears during late training:

\[
\boxed{
\text{the scorer / STOP policy becomes prematurely conservative.}
}
\]

At epoch 13:

```text
FULL Mean Recall:        27.651150
best REPEAT Mean Recall: 32.377768
gap:                     +4.726618
```

Forced continuation with one repeated candidate substantially outperforms the learned FULL policy. Thus the late checkpoint is not simply failing because recurrent computation is useless; it is also failing because the learned policy stops too early.

R1c1 should therefore be frozen as a **mechanically valid but scientifically negative causal experiment**.

The next clean experiment should be R1c2: dynamic current-state re-proposal / dynamic WHAT, not a rescue stack on this branch.

---

# 2. Why R1c1 existed

R0 showed that the token-space recurrent editor remained active while retrieval-space effects collapsed rapidly:

```text
R0 ΔZ:
~2.262 -> ~2.264 -> ~2.301

R0 Δq:
~0.3665 -> ~0.0825 -> ~0.0190
```

This indicated that recurrent token edits were not necessarily dead, but their retrieval-space consequences were being strongly attenuated.

R1a causally isolated the global query cap:

```yaml
query_cap:
  0.5 -> 1000.0
```

and restored healthy late-step retrieval effects.

R1a best:

```text
Mean Recall = 38.764146

Δq norm:
t0 = 0.3366
t1 = 0.2724
t2 = 0.1971

retention:
t1/t0 = 80.9%
t2/t0 = 58.6%
t2/t1 = 72.4%
```

R1a also demonstrated that recurrent depth itself can be useful:

```text
same-parent mean-candidate MR:
t0 ≈ 24.60
t1 ≈ 34.30
t2 ≈ 39.03
```

However, candidate decomposition remained collapsed:

```text
intent cosine   ≈ 0.950
support cosine  ≈ 0.99984
support overlap ≈ 0.9951
ΔZ cosine       ≈ 0.982
Δq cosine       ≈ 0.98
```

The four action candidates were still highly redundant.

R1b tested a different hypothesis:

> Keep WHERE static, but learn a dynamic scalar WHETHER / applicability gate.

R1b was mechanically healthy but scientifically negative:

```text
best MR ≈ 39.012
confidence ≈ 0.982 nearly everywhere
p_null ≈ 0.018 nearly everywhere
late selected utility became worse than R1a
```

Thus the next clean structural question was:

> Is the problem that the same static support is reused even after the visual state changes?

R1c1 was created to answer only that question.

---

# 3. R1c1 scientific contract

R1c1 preserves R1a everywhere except the grounding source.

## 3.1 Fixed WHAT

The text-derived candidate intent is computed once:

\[
I_k
=
\operatorname{IntentEncoder}(q_k,T)
\]

for:

\[
k\in\{1,2,3,4\}.
\]

The candidate identity and action proposal are fixed throughout the rollout.

R1c1 does **not** ask:

> What action is still missing now?

It only asks:

> Given the same original action identity, where should this action look in the current state now?

## 3.2 Dynamic WHERE

For each recurrent timestep:

\[
Z_t \in \mathbb{R}^{N\times d}
\]

is the current visual token state.

The grounder recomputes:

\[
\ell_{t,k,n}
=
\frac{
(W_Q I_k)^\top
(W_K Z_{t,n})
}{
\sqrt{d_g}
}
\]

and:

\[
\pi_{t,k}
=
\operatorname{Entmax}_{1.5}(\ell_{t,k,:}).
\]

Hence:

```text
R1a:
π_k = Ground(I_k, A)
and reuse π_k for all t

R1c1:
π_t,k = Ground(I_k, Z_t)
and recompute every t
```

The immutable anchor remains:

\[
A = Z_0.
\]

Only the WHERE input changes from `A` to the current recurrent state `Z_t`.

## 3.3 Grounded evidence

Using dynamic support:

\[
O_{t,k}
=
\sum_n
\pi_{t,k,n}
W_O A_n
\]

\[
C_{t,k}
=
\sum_n
\pi_{t,k,n}
W_C Z_{t,n}
\]

\[
D_{t,k}
=
\sum_n
\pi_{t,k,n}
W_D(Z_{t,n}-A_n).
\]

These are fused with the fixed candidate intent and sent through the same context/editor/scorer pathway as R1a.

## 3.4 Same-parent counterfactual contract

All candidate consequences at timestep `t` branch from the same current parent:

\[
\widehat Z_{t+1}^{(k)}
=
Z_t+\Delta Z_{t,k}.
\]

This invariant was preserved exactly.

Therefore candidate comparisons remain valid same-parent counterfactuals.

---

# 4. What R1c1 explicitly did NOT change

The branch did not add or modify:

```text
dynamic WHAT
dynamic action re-proposal
R1b applicability
Visual NULL
semantic residual
DPP / FuncDPP
RDMReg
VISReg
variance-floor loss
orthogonality loss
teacher grounding
LLM / MLLM supervision
RL / DQN
new STOP loss
new selector
new scorer
new editor
candidate-specific editor
candidate-specific grounder
query-cap tuning
timestep decay
target-conditioned execution
```

This is important because the R1c1 result can be interpreted causally:

\[
\text{R1c1} - \text{R1a}
\approx
\text{effect of current-state dynamic WHERE}.
\]

---

# 5. Implementation and diagnostic audit status

Before training, the R1c1 branch was audited for causal contamination.

The canonical config is:

```yaml
query_cap: 1000.0
enable_dynamic_regrounding: true
enable_dynamic_applicability: false
enable_visual_null: false
grounding_normalization: entmax15
```

The forward path has:

```text
IntentEncoder calls: 1 per rollout
Grounder calls:      Tmax = 3 per rollout
Applicability calls: 0
```

The output trace separates:

```text
raw_spatial_supports
effective_spatial_supports
spatial_supports
temporal_supports
```

with:

```text
raw_spatial_supports
=
actual Ground(I, Z_t)

effective_spatial_supports
=
support consumed by diagnostic-control scorer semantics

temporal_supports
=
stack of raw Ground(I, Z_t)
```

This prevents CLONE/MEAN diagnostic controls from contaminating temporal grounding analysis.

The target firewall was also strengthened:

```text
IAGSRMECore.forward has no target input.
```

Target-relative utilities are computed only after the target-free rollout already exists.

Checkpoint replay guard confirms:

```text
architecture_generation = r1c1_dynamic_current_state_reground_v1
query_cap = 1000
dynamic_regrounding = true
dynamic_applicability = false
grounding = entmax15
saved metric == replayed metric
```

Both BEST and LATE diagnostic replays have zero metric discrepancy.

---

# 6. GPU canary

The R1c1 CUDA canary ran:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/canary_train_iag_srme.py \
  --r1c1 \
  --dataset-root data/FashionIQ \
  --steps 100 \
  --precision fp16
```

Final mechanical summary:

```text
attempted steps:                 100
successful optimizer steps:       96
skipped AMP-overflow steps:         4
first successful step:              3
final GradScaler scale:          4096
finite:                          true
mechanical_status:               PASS

grounding_nonzero_gradient_fraction:
1.0

grounding_projection parameter Δmax:
0.0004545
```

No R1b applicability pathway was active.

No candidate monopoly, never-STOP condition, or identical-candidate warning triggered at canary level.

The canary therefore established:

\[
\boxed{
\text{dynamic current-state grounding is mechanically trainable and receives gradient.}
}
\]

However, even the canary already showed very small temporal support motion:

```text
same-candidate temporal cosine ≈ 0.9999+
support L1 change ≈ 0.005 - 0.018
```

This was an early warning that dynamic recomputation did not automatically imply meaningful re-grounding.

---

# 7. Training trajectory

Observed training:

| Epoch | Train total loss | Mean Recall | Best so far |
|---:|---:|---:|---:|
| 1 | 1.4152 | 33.007 | 33.007 |
| 2 | 0.6584 | 37.509 | 37.509 |
| 3 | 0.3884 | 38.561 | 38.561 |
| 4 | 0.2392 | **38.754** | **38.754** |
| 5 | 0.1716 | 36.042 | 38.754 |
| 6 | 0.1346 | 36.001 | 38.754 |
| 7 | 0.1094 | 31.921 | 38.754 |
| 8 | 0.1035 | 31.431 | 38.754 |
| 9 | 0.0870 | 30.874 | 38.754 |
| 10 | 0.0738 | 32.922 | 38.754 |
| 11 | 0.0715 | 31.157 | 38.754 |
| 12 | 0.0635 | 31.280 | 38.754 |
| 13 | 0.0606 | **27.651** | 38.754 |

The key pattern is:

\[
L_{\text{train}}\downarrow
\qquad
\text{MR}_{val}\downarrow
\]

after epoch 4.

This is not a simple optimizer failure.

The model is successfully optimizing the training objective while validation retrieval becomes much worse.

---

# 8. Trusted checkpoints

Run:

```text
outputs/2026-08-31/13-24-24/
```

Best checkpoint:

```text
best.pt
epoch = 4
Mean Recall = 38.75413338343302
```

Late checkpoint:

```text
last.pt
epoch = 13
Mean Recall = 27.65115002791087
```

Both are trusted R1c1 replay checkpoints.

---

# 9. Retrieval — R1a vs R1c1 BEST

R1a:

```text
FULL MR = 38.764146
```

R1c1 BEST:

```text
FULL MR = 38.754133
```

Difference:

\[
38.754133 - 38.764146
=
-0.010013.
\]

This is effectively no gain.

Therefore:

\[
\boxed{
\text{dynamic current-state WHERE is not sufficient to improve best retrieval over R1a.}
}
\]

---

# 10. BEST checkpoint retrieval controls

At epoch 4:

```text
FULL             38.7541
MEAN             38.7631
best REPEAT      38.8971
best SINGLE      24.7414
REFERENCE_ONLY   14.4076
```

Ratios:

```text
MEAN / FULL         = 1.00023
best REPEAT / FULL  = 1.00369
```

Therefore the learned candidate-selection policy does not meaningfully outperform:

```text
mean all candidate effects
```

or:

```text
repeat one fixed candidate
```

This is still the same functional redundancy signature seen in R1a.

---

# 11. BEST same-parent depth remains useful

At epoch 4, mean candidate retrieval from the same current parent improves strongly with recurrent depth:

```text
t0 mean-candidate MR = 24.6512
t1 mean-candidate MR = 33.8352
t2 mean-candidate MR = 38.8594
```

Offline best-candidate oracle:

```text
t0 = 25.0997
t1 = 34.3598
t2 = 39.3777
```

This is important.

It again shows:

\[
\boxed{
\text{multi-step computation itself can still be useful.}
}
\]

The problem is not simply:

```text
"more than one step is wrong."
```

Instead, the four candidate identities remain too redundant and the policy does not exploit meaningful specialization.

---

# 12. BEST dynamic WHERE is still candidate-cloned

The central R1c1 question is whether current-state re-grounding causes candidate supports to separate.

It does not.

At epoch 4:

```text
between-candidate support cosine:
t0 = 0.9997662
t1 = 0.9997550
t2 = 0.9997398
```

For comparison, R1a static support cosine was approximately:

```text
0.999842
```

Hence:

\[
\pi_{t,0}
\approx
\pi_{t,1}
\approx
\pi_{t,2}
\approx
\pi_{t,3}
\]

for all `t`.

R1c1 reduces support similarity only trivially.

This rejects the strong hypothesis:

> Reusing one static support map across time is the primary reason all four candidates look at the same place.

Even when the support is recomputed from the current state, all four candidates still look almost identically.

---

# 13. BEST WHAT is still highly correlated

At epoch 4:

```text
mean pairwise intent cosine = 0.954016
```

Thus fixed WHAT remains strongly correlated:

\[
I_1\approx I_2\approx I_3\approx I_4.
\]

This matters because R1c1 only changes the state being queried.

If the four queries entering the grounder are already highly similar, then:

\[
Ground(I_1,Z_t),
\dots,
Ground(I_4,Z_t)
\]

have little structural reason to become meaningfully different.

This is a major clue for the next experiment.

---

# 14. BEST functional candidates remain clones

At epoch 4:

## Δq

```text
pairwise Δq cosine:
t0 = 0.982771
t1 = 0.980743
t2 = 0.975395
```

Functional effective rank:

```text
t0 = 1.8296 / 4
t1 = 1.8720 / 4
t2 = 1.9708 / 4
```

Mean Δq norm:

```text
t0 = 0.33027
t1 = 0.26697
t2 = 0.19414
```

Retention:

```text
t1/t0 = 0.80834
t2/t0 = 0.58783
t2/t1 = 0.72720
```

The good R1a retrieval-effect survival is preserved.

Therefore R1c1 does **not** recreate the original R0 `Δq` attenuation failure.

But specialization remains weak.

## ΔZ

At epoch 4:

```text
pairwise ΔZ cosine:
t0 ≈ 0.98076
t1 ≈ 0.98067
t2 ≈ 0.98068
```

Mean ΔZ norm remains healthy:

```text
t0 ≈ 2.1893
t1 ≈ 2.1733
t2 ≈ 2.1626
```

Thus the candidate edits are alive but highly parallel.

The structural chain is:

```text
correlated fixed WHAT
        ↓
near-identical dynamic WHERE
        ↓
near-identical contexts
        ↓
parallel ΔZ
        ↓
parallel Δq
```

---

# 15. BEST scientific verdict

The best checkpoint gives a clean negative result:

```text
dynamic WHERE exists
but candidate-specific WHERE does not emerge
```

and:

```text
best MR ≈ R1a
MEAN ≈ FULL
REPEAT ≈ FULL
```

Therefore R1c1 does not solve the candidate-collapse problem.

At this stage the correct verdict is already:

\[
\boxed{
\text{R1c1 is not a successful retrieval intervention.}
}
\]

The late checkpoint then explains how the failure evolves under continued optimization.

---

# 16. BEST → LATE retrieval collapse

BEST:

```text
epoch 4
FULL = 38.7541
```

LATE:

```text
epoch 13
FULL = 27.6512
```

Change:

\[
-11.1030\ \text{Mean Recall points}.
\]

The training loss simultaneously decreases from:

```text
0.2392 -> 0.0606
```

Therefore the model is increasingly fitting a training solution that generalizes poorly to validation retrieval.

---

# 17. LATE support becomes sharper, not more candidate-specific

At epoch 4:

```text
support effective size ≈ 28.03 / 196
support entropy        ≈ 3.302
support fraction       ≈ 14.82%
```

At epoch 13:

```text
support effective size ≈ 18.68 / 196
support entropy        ≈ 2.909
support fraction       ≈ 10.02%
```

So the support becomes more concentrated.

But between-candidate similarity does not improve.

LATE:

```text
support cosine:
t0 = 0.9998269
t1 = 0.9998270
t2 = 0.9998261
```

Thus:

\[
\boxed{
\text{the model learns sharper supports, but the four candidates still share almost the same support.}
}
\]

This is not useful specialization.

---

# 18. LATE dynamic WHERE becomes "moving clones"

Late training increases the amount by which support changes across recurrent state updates.

The important observation is not merely that:

\[
\pi_{t+1,k}\neq \pi_{t,k}.
\]

The important question is whether different candidates move differently.

Define:

\[
\Delta\pi_{t,k}
=
\pi_{t+1,k}-\pi_{t,k}.
\]

If candidate-specific adaptive grounding were emerging, then:

\[
\Delta\pi_{t,1},
\Delta\pi_{t,2},
\Delta\pi_{t,3},
\Delta\pi_{t,4}
\]

should not all point in the same direction.

Instead the late checkpoint shows approximately common-mode movement.

This gives the core R1c1 failure mode:

```text
static clone:
π_0 ≈ π_1 ≈ π_2 ≈ π_3

R1c1 late:
π_t changes with state
BUT
all candidate π_t move together
```

Hence:

\[
\boxed{\text{moving clones}}
\]

rather than:

\[
\boxed{\text{candidate-specific re-grounding}}.
\]

Dynamic support movement by itself is therefore not evidence of successful adaptive grounding.

---

# 19. Functional collapse gets worse at LATE

BEST Δq cosine:

```text
t0 = 0.98277
t1 = 0.98074
t2 = 0.97539
```

LATE:

```text
t0 = 0.99230
t1 = 0.99146
t2 = 0.98994
```

The candidate retrieval-space effects become **more parallel**, not less.

BEST functional rank:

```text
t0 = 1.8296
t1 = 1.8720
t2 = 1.9708
```

LATE:

```text
t0 = 1.6053
t1 = 1.6345
t2 = 1.6838
```

Thus:

\[
\boxed{
\text{continued optimization reduces functional candidate rank.}
}
\]

ΔZ shows the same trend.

LATE ΔZ cosine is approximately:

```text
~0.9905 - 0.9908
```

versus BEST:

```text
~0.9807
```

Thus the four candidate editors are increasingly producing essentially the same edit direction.

---

# 20. LATE mean candidate depth is still useful

Despite the overall validation collapse, same-parent candidate retrieval still improves strongly with depth.

LATE:

```text
mean candidate MR:
t0 = 20.9694
t1 = 27.7120
t2 = 32.3871
```

Oracle:

```text
t0 = 21.2359
t1 = 27.9046
t2 = 32.5661
```

This remains a monotonic recurrent improvement:

\[
20.97
\rightarrow
27.71
\rightarrow
32.39.
\]

Therefore the late checkpoint does **not** falsify the idea that recurrent depth can contain useful computation.

Instead, the learned FULL policy fails to exploit it.

---

# 21. LATE policy / STOP failure

The strongest evidence is the forced REPEAT control.

LATE:

```text
FULL        = 27.6512
REPEAT-0    = 32.3637
REPEAT-1    = 32.3300
REPEAT-2    = 32.2876
REPEAT-3    = 32.3778
```

Best REPEAT:

```text
32.3778
```

Gap:

\[
32.3778 - 27.6512
=
4.7266.
\]

This is a major failure.

The model's own action-selection / STOP policy is discarding substantial useful recurrent computation.

A trivial forced policy:

```text
pick one candidate identity
repeat it through recurrence
```

beats the learned FULL policy by almost five Mean Recall points.

Therefore:

\[
\boxed{
\text{the late scorer / STOP policy is miscalibrated relative to actual retrieval utility.}
}
\]

---

# 22. Why this is not simply "late edit over-edit"

At the R1a best checkpoint the third selected edit had a slightly negative average target-relative effect:

```text
t2 selected target-relative gain ≈ -0.00261
```

R1c1 BEST is essentially the same regime.

However, by the late R1c1 checkpoint, the strongest control evidence shows that additional recurrent computation remains useful while FULL dramatically underperforms forced continuation.

Therefore the epoch-13 collapse should not be summarized as:

```text
"the model edits too much."
```

A better description is:

```text
the model becomes increasingly conservative in its learned selection/STOP policy,
even though additional recurrent computation still has retrieval value.
```

This distinction matters for future STOP/planning experiments.

---

# 23. LATE candidate identity is still not meaningful

At epoch 13:

```text
candidate 0 t2 MR ≈ 32.3710
candidate 1 t2 MR ≈ 32.4274
candidate 2 t2 MR ≈ 32.3564
candidate 3 t2 MR ≈ 32.4187
```

The candidates are nearly interchangeable.

The oracle only marginally exceeds each individual candidate.

This is consistent with:

```text
support cosine ≈ 0.9998
ΔZ cosine ≈ 0.991
Δq cosine ≈ 0.990
```

Therefore the four identities still do not represent meaningfully different semantic actions.

---

# 24. R1c1 causal hypothesis verdict

Primary hypothesis:

> Static reuse of visual grounding is a major causal bottleneck behind candidate collapse.

Result:

\[
\boxed{
\text{REJECTED / strongly weakened under the current architecture and objective.}
}
\]

Evidence:

1. Dynamic current-state grounding is mechanically real.
2. It receives gradient on every successful canary update.
3. It is recomputed from live `Z_t`.
4. BEST retrieval does not improve over R1a.
5. Candidate support cosine remains approximately 0.9997-0.9998 at all timesteps.
6. Candidate ΔZ remains highly parallel.
7. Candidate Δq remains highly parallel.
8. MEAN remains equivalent to FULL.
9. REPEAT remains equivalent to FULL at BEST.
10. Continued training increases temporal support motion but mostly as common-mode co-motion.
11. Continued training lowers functional candidate rank.

Hence dynamic WHERE does not create meaningful four-way decomposition.

---

# 25. Failure mode classification

R1c1 should be classified as:

```text
MECHANICAL:
PASS

CAUSAL ISOLATION:
PASS

BEST RETRIEVAL:
NO IMPROVEMENT

DYNAMIC WHERE:
REAL BUT WEAKLY DIFFERENTIATED

CANDIDATE SPECIALIZATION:
FAIL

TEMPORAL GROUNDING:
STATE-SENSITIVE BUT COMMON-MODE

FUNCTIONAL SPECIALIZATION:
FAIL / WORSE LATE

LATE POLICY:
PREMATURE STOP / SCORER MISCALIBRATION

SCIENTIFIC VERDICT:
NEGATIVE
```

---

# 26. Updated collapse chain

The evidence now supports the following chain:

```text
highly correlated fixed WHAT
        ↓
Ground(I_k, Z_t)
        ↓
dynamic WHERE is recomputed
        ↓
BUT all four WHEREs remain nearly identical
        ↓
state change causes common-mode support movement
        ↓
contexts remain highly similar
        ↓
ΔZ remains highly parallel
        ↓
Δq remains highly parallel
        ↓
candidate identities remain interchangeable
        ↓
MEAN / REPEAT remain competitive
        ↓
continued training sharpens the common solution
        ↓
functional rank decreases
        ↓
scorer / STOP becomes increasingly conservative
        ↓
FULL stops before useful recurrence is exhausted
```

Short version:

\[
\boxed{
\text{correlated WHAT}
\rightarrow
\text{moving-clone WHERE}
\rightarrow
\text{clone effects}
\rightarrow
\text{late STOP miscalibration}.
}
\]

---

# 27. What R1c1 falsifies

R1c1 provides evidence against the following claim:

> Simply making visual grounding state-dependent is enough to make the four actions specialize.

It is not.

The model can satisfy:

\[
\pi_{t,k}=Ground(I_k,Z_t)
\]

while still learning:

\[
\pi_{t,1}
\approx
\pi_{t,2}
\approx
\pi_{t,3}
\approx
\pi_{t,4}.
\]

Furthermore, all four supports can move over time while preserving this equality approximately.

Thus:

\[
\boxed{
\text{dynamic does not imply diverse}
}
\]

and:

\[
\boxed{
\text{state-sensitive does not imply action-specific}.
}
\]

---

# 28. What R1c1 does NOT falsify

R1c1 does not establish that:

```text
multi-step CIR is wrong
```

because same-parent retrieval improves strongly with depth.

It also does not establish that:

```text
visual grounding is useless
```

because the test only concerns the current shared grounding mechanism under highly correlated fixed WHAT.

It does not establish that:

```text
dynamic action generation cannot work
```

because WHAT was explicitly frozen across timesteps.

It does not establish that:

```text
teacher grounding is necessary
```

because no teacher-free dynamic WHAT test has been completed yet.

It does not establish that:

```text
DPP / VISReg / diversity loss is required
```

because those mechanisms were intentionally excluded from this causal step.

---

# 29. Why R1c2 is now justified

R1c1 preserves:

\[
I_k
\]

for the entire rollout.

Yet the diagnostic repeatedly shows:

```text
pairwise intent cosine ≈ 0.95+
```

If the four WHAT vectors are already highly correlated, then recomputing WHERE on the current state cannot by itself create strong semantic specialization.

The next causal question becomes:

> After applying an edit, should the action proposal itself be recomputed from what remains unsatisfied in the current state?

That is R1c2.

Conceptually:

```text
R1c1:
fixed WHAT
+
dynamic WHERE

R1c2:
dynamic current-state WHAT
+
corresponding dynamic WHERE
```

Instead of:

\[
I_k = Intent(T)
\]

once, R1c2 should investigate something of the form:

\[
I_{t,k}
=
Proposal(
q_k,
T,
Z_t,
\text{state/change context}
)
\]

with strict target-free execution.

The intended question is:

> What edit is still missing NOW?

rather than:

> Where should the same original action look NOW?

---

# 30. R1c2 must remain a clean causal experiment

Do not immediately add:

```text
dynamic WHAT
+
semantic residual
+
DPP
+
VISReg
+
teacher
+
new STOP loss
```

That would destroy causal attribution.

Recommended sequence:

```text
R1c2:
dynamic current-state re-proposal only

then evaluate:
- intent cosine by timestep
- support cosine by timestep
- support displacement alignment
- ΔZ cosine / rank
- Δq cosine / rank
- same-parent utility
- FULL vs REPEAT vs MEAN
- selected target-relative gain
- STOP behavior
```

Only if R1c2 fails should more explicit specialization constraints be promoted.

---

# 31. STOP / scorer issue should be tracked separately

R1c1 reveals an independent late-training pathology:

```text
FULL late ≪ forced REPEAT late
```

This suggests:

\[
\text{learned score ordering}
\neq
\text{actual marginal retrieval utility ordering}.
\]

This should be logged as a separate future causal target.

Possible label:

```text
B5 — consequence-score / STOP calibration
```

However it should **not** be repaired on the frozen R1c1 branch.

A STOP fix added now would confound the primary R1c1 conclusion.

Recommended action:

```text
record it
freeze R1c1
test R1c2
return to STOP/planning only in its own controlled experiment
```

unless R1c2 specifically requires a policy-calibration control to remain interpretable.

---

# 32. Important caution about late collapse

R1c1 is not the first branch to show late validation degradation.

Earlier lineage already showed:

```text
training objective can continue improving
while validation retrieval degrades.
```

Therefore the correct statement is not:

> Dynamic WHERE alone caused all late collapse.

The correct statement is:

> Dynamic WHERE failed to prevent the existing late-collapse tendency, did not improve the best checkpoint, did not create candidate specialization, and at late training coexisted with even stronger functional cloning plus severe STOP / scorer under-utilization of recurrent depth.

This distinction should be preserved in future reports.

---

# 33. Matched comparison table

| Metric | R1a best | R1c1 best e4 | R1c1 late e13 |
|---|---:|---:|---:|
| FULL Mean Recall | 38.7641 | 38.7541 | 27.6512 |
| Best REPEAT | 39.2232 | 38.8971 | 32.3778 |
| MEAN | 38.8160 | 38.7631 | 27.5010 |
| Reference only | 14.5216 | 14.4076 | 14.0555 |
| t0 mean Δq norm | 0.3366 | 0.3303 | 0.2694 |
| t1 mean Δq norm | 0.2724 | 0.2670 | 0.2196 |
| t2 mean Δq norm | 0.1971 | 0.1941 | 0.1691 |
| t1/t0 Δq retention | 0.809 | 0.808 | 0.815 |
| t2/t0 Δq retention | 0.586 | 0.588 | 0.628 |
| support cosine t0 | 0.999842 | 0.999766 | 0.999827 |
| support cosine t1 | static same map | 0.999755 | 0.999827 |
| support cosine t2 | static same map | 0.999740 | 0.999826 |
| Δq cosine t0 | ~0.983 | 0.98277 | 0.99230 |
| Δq cosine t1 | ~0.982 | 0.98074 | 0.99146 |
| Δq cosine t2 | ~0.977 | 0.97539 | 0.98994 |
| functional rank t0 | ~1.82 | 1.8296 | 1.6053 |
| functional rank t1 | ~1.86 | 1.8720 | 1.6345 |
| functional rank t2 | ~1.95 | 1.9708 | 1.6838 |
| same-parent mean MR t0 | 24.60 | 24.65 | 20.97 |
| same-parent mean MR t1 | 34.30 | 33.84 | 27.71 |
| same-parent mean MR t2 | 39.03 | 38.86 | 32.39 |

Key interpretation:

```text
R1c1 best:
almost matched R1a, but did not improve specialization.

R1c1 late:
support remains cloned,
functional effects become even more parallel,
policy underuses still-useful recurrent depth.
```

---

# 34. Reproduction commands

## Training

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r1c1_dynamic_reground \
  experiment=iag_srme_r1c1_dynamic_reground \
  protocol=fashioniq_original
```

## BEST diagnostic

```bash
RUN=outputs/2026-08-31/13-24-24

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/best.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1c1_dynamic_reground_best.json
```

Expected:

```text
checkpoint epoch = 4
FULL MR = 38.75413338343302
trusted_r1c1_replay = true
```

## LATE diagnostic

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/last.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1c1_dynamic_reground_late.json
```

Expected:

```text
checkpoint epoch = 13
FULL MR = 27.65115002791087
trusted_r1c1_replay = true
```

---

# 35. Experiment ledger update

After this experiment:

```text
R0   diagnostic audit                         DONE
R1a  remove global query cap                  PASS
R1b  dynamic applicability / visual NULL      NEGATIVE
R1c1 dynamic current-state re-grounding       NEGATIVE
R1c2 dynamic current-state re-proposal        NEXT
R2   semantic residual                        PENDING
R3   quality-gated functional DPP             CONDITIONAL
R4   target-privileged grounding teacher      CONDITIONAL
R5   planning / STOP refinement               CONDITIONAL
```

---

# 36. Final scientific statement

The R1c1 experiment supports the following statement:

> **Recomputing Entmax visual grounding from the current recurrent token state at every timestep is mechanically valid but is not sufficient to resolve IAG-SRME candidate collapse on FashionIQ. The best R1c1 checkpoint reaches 38.754 Mean Recall, effectively identical to the 38.764 R1a baseline, while candidate visual supports remain nearly identical at all timesteps (between-candidate cosine approximately 0.9997–0.9998). Retrieval-space candidate effects also remain highly parallel. With further training, grounding becomes more state-sensitive but does not become meaningfully candidate-specific; the support maps move largely in common mode, while ΔZ/Δq similarity increases and effective candidate rank decreases. The late checkpoint additionally exposes a separate scorer/STOP calibration failure: FULL retrieval falls to 27.651 while forced REPEAT reaches 32.378, indicating that the learned policy stops before useful recurrent computation is exhausted. Therefore static support reuse is not the primary bottleneck. The next clean causal test should regenerate WHAT/action proposals from the current state rather than only re-grounding fixed actions.**

Compactly:

\[
\boxed{
\text{fixed correlated WHAT}
\rightarrow
\text{dynamic but moving-clone WHERE}
\rightarrow
\text{clone effects}
\rightarrow
\text{late STOP miscalibration}
}
\]

and:

\[
\boxed{
\text{R1c1 = mechanically PASS, scientifically NEGATIVE.}
}
\]

---

# 37. One-screen handoff

```text
BRANCH
exp/e2e-iag-srme-r1c1-dynamic-reground

MECHANISM
fixed WHAT
+
Ground(I_k, Z_t) every timestep

CANARY
96 / 100 successful optimizer steps
grounder gradient fraction = 1.0
mechanical PASS

BEST
epoch 4
FULL MR = 38.7541

R1a
FULL MR = 38.7641

=> no retrieval gain

BEST WHERE
support cosine:
t0 .999766
t1 .999755
t2 .999740

=> dynamic WHERE still candidate-cloned

BEST Δq
norm:
.3303 -> .2670 -> .1941

cosine:
.9828 -> .9807 -> .9754

rank:
1.83 -> 1.87 -> 1.97

=> recurrence survives
but candidates remain parallel

LATE
epoch 13
FULL MR = 27.6512

support cosine:
t0 .999827
t1 .999827
t2 .999826

Δq cosine:
.9923 -> .9915 -> .9899

rank:
1.61 -> 1.63 -> 1.68

=> functional cloning gets worse

LATE CONTROL
best REPEAT = 32.3778
FULL        = 27.6512

=> policy / STOP prematurely discards useful recurrent depth

MAIN FAILURE
dynamic WHERE did not create candidate-specific re-grounding
it created moving clones

VERDICT
R1c1 scientific NEGATIVE

NEXT
R1c2 dynamic current-state re-proposal / dynamic WHAT

DO NOT STACK YET
DPP
VISReg
RDMReg
teacher
semantic residual
new STOP loss
RL
```

---

# 38. Freeze decision

This branch should now be treated as a frozen causal record.

Do not hyperparameter-fish R1c1 into success.

Do not silently add a diversity loss and continue calling the result R1c1.

Do not modify STOP and reinterpret the resulting branch as evidence for dynamic WHERE.

Create a new branch for the next causal question.

Recommended next branch concept:

```text
exp/e2e-iag-srme-r1c2-dynamic-reproposal
```

with a separately specified implementation contract.

The value of R1c1 is not that it improved retrieval.

Its value is that it eliminated one plausible explanation:

\[
\boxed{
\text{static current-state blindness of WHERE alone is not the dominant remaining cause of candidate collapse.}
}
\]

That narrows the research space substantially.