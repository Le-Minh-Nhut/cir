# CIR V2 R0 — Q1 `L_safe` Diagnosis: Partial Safety Recovery, Lower Oracle Ceiling, and C2 Winner-Take-Most Shift

**Date:** 2026-09-09  
**Branch:** `exp/e2e-iag-srme-v2-r0`  
**Q1 implementation git SHA:** `4f4aa98c8a1ca3b0ad74c96073fbca52441c71f3`  
**Dataset:** FashionIQ  
**Backbone:** `fgclip_base_text_native_cls`  
**Finetune policy:** text-only (`train_vision=false`, `train_text=true`, `train_text_projection=false`)  
**Experiment identity:** `R0-NCLS-TEXT`  
**Primary full-VAL diagnostic cohort:** 6,016 samples  
**Primary timestep:** `t=0`  
**Gradient diagnostic:** 64 batches × 8 = 512 samples  
**Purpose:** determine whether direct all-candidate no-harm supervision improves candidate quality without destroying useful specialization, and whether it resolves the STRONG winner-take-most task-gradient failure.

---

## 1. Executive conclusion

Q1 adds the direct candidate safety objective

\[
L_{\mathrm{safe}}
=
\mathbb E_{i,t,k}
\max\left(
0,\;
\ell_{\mathrm{ret}}(\hat q_{i,t}^{(k)},Y_i)
-
\operatorname{sg}\left[\ell_{\mathrm{ret}}(q_{i,t},Y_i)\right]
\right)
\]

with:

```text
lambda_safe = 0.1
```

while keeping the STRONG objective otherwise unchanged.

The intervention is **not a complete failure**. It improves official validation Mean Recall from the STRONG checkpoint and substantially repairs mean candidate utility for several slots. However, it does **not** achieve the primary goal of producing a stronger candidate set while preserving useful specialization.

The strongest current conclusion is:

> **Q1 partially improves candidate safety and official Recall, but it lowers the all-K oracle ceiling and does not solve winner-take-most task learning. The dominant task-gradient branch simply shifts from C3 under STRONG to C2 under Q1.**

The compact failure mechanism after Q1 is:

```text
STRONG
C3 dominant task-learning branch
+
many harmful candidates
        |
        | add L_safe = 0.1
        v
candidate harm partially reduced
+
C0/C2 mean utility improves
        |
        +--> C2 becomes strongest/highest-impact branch
        |
        +--> C1/C3 edits shrink substantially
        |
        v
hard terminal task gradient re-concentrates on C2
        |
        v
winner-take-most persists, but winner changes C3 -> C2
```

Therefore Q1 should be recorded as:

> **PARTIAL POSITIVE / PRIMARY HYPOTHESIS FAILED**

It should **not** be followed by simply increasing `lambda_safe`.

---

## 2. Q1 training configuration

Q1 used:

```text
terminal_weight = 1.0
lambda_pair     = 0.5
lambda_gain     = 0.5
lambda_safe     = 0.1
lambda_c        = 0.6
lambda_bind     = 1.0
lambda_rel      = 0.6
lambda_dpp      = 0.6
```

Other relevant settings:

```text
epochs       = 20
batch_size   = 32
lr           = 1e-5
weight_decay = 0.01
seed         = 42
K            = 4
T            = 3
STOP         = enabled
epsilon_stop = 0
precision    = fp16
```

Checkpoint:

```text
outputs/r0_ncls_text_strong_aux_safe_q1/best.pt
epoch:       19
Mean Recall: 29.537111769119896
```

The final epoch was slightly worse:

```text
epoch 20 Mean Recall = 29.054
```

so all Q1 checkpoint analysis should use `best.pt`, not `last.pt`.

---

## 3. Benchmark comparison

Historical checkpoints:

```text
OLD:
Mean Recall = 30.824426313241325

STRONG:
Mean Recall = 28.244677186012268

Q1 L_safe=0.1:
Mean Recall = 29.537111769119896
```

Therefore:

```text
Q1 - STRONG = +1.292434583107628 Mean Recall
Q1 - OLD    = -1.287314544121429 Mean Recall
```

This means Q1 recovers roughly half of the STRONG benchmark degradation, but does not recover the OLD benchmark.

The benchmark result alone would make Q1 look moderately successful. The full-VAL candidate diagnostics show why that interpretation would be incomplete.

---

# 4. Full-VAL candidate quality after Q1

The full-VAL diagnostic used the same persistent 6,016-sample manifest as the OLD/STRONG diagnosis.

Q1 per-slot utility at `t=0`:

| Slot | Mean utility | Positive fraction | Harmful fraction | Oracle occupancy | Shapley contribution |
|---|---:|---:|---:|---:|---:|
| C0 | +0.087088 | 50.332% | 49.668% | 27.111% | 0.118739 |
| C1 | +0.020862 | 38.963% | 61.021% | 12.018% | 0.027576 |
| C2 | +0.004968 | 45.479% | 54.521% | 33.843% | 0.206473 |
| C3 | +0.003417 | 38.597% | 61.403% | 15.691% | 0.055194 |

Compared with STRONG:

| Slot | STRONG mean utility | Q1 mean utility | STRONG harmful | Q1 harmful |
|---|---:|---:|---:|---:|
| C0 | -0.077250 | **+0.087088** | 67.071% | **49.668%** |
| C1 | -0.102786 | **+0.020862** | 58.511% | 61.021% |
| C2 | -0.009679 | **+0.004968** | 67.570% | **54.521%** |
| C3 | -0.384116 | **+0.003417** | 64.412% | 61.403% |

This is real evidence that `L_safe` changes upstream candidate quality.

In particular:

- C0 improves strongly.
- C2 becomes less harmful on validation than under STRONG.
- C3's mean utility moves from severely negative to approximately neutral.
- All four mean utilities become non-negative.

However, C1 and C3 remain harmful on a majority of full-VAL rows.

---

# 5. Q1 lowers the all-K candidate-set ceiling

The most important negative result is the all-K oracle value.

```text
OLD all-K oracle    = 0.491764
STRONG all-K oracle = 0.453040
Q1 all-K oracle     = 0.407982
```

Thus:

```text
Q1 - STRONG oracle = -0.045058
relative drop      ≈ -9.95%
```

Q1 therefore improves official live-policy Mean Recall while making the best available candidate set at `t=0` worse under the canonical teacher utility.

This is a key distinction:

> **Q1 improves safety/average behavior enough to help the live system, but it removes part of the high-value tail that made the STRONG candidate slate more powerful under oracle selection.**

The Q1 single-slot oracle values are:

```text
C0 = 0.172348
C1 = 0.050777
C2 = 0.258070
C3 = 0.091043
```

The best fixed slot is now C2.

---

# 6. Functional credit shifts from C3 to C2

STRONG normalized Shapley shares:

```text
C0  9.961%
C1 15.803%
C2  7.880%
C3 66.356%
```

Q1 normalized Shapley shares:

```text
C0 29.104%
C1  6.759%
C2 50.608%
C3 13.529%
```

Thus the functional monopoly does not disappear. It changes identity:

```text
STRONG dominant branch: C3
Q1 dominant branch:     C2
```

Q1 effective functional K is:

```text
K_eff = 3.179203
95% CI ≈ 3.1276 .. 3.2245
```

which is higher than STRONG:

```text
STRONG K_eff ≈ 2.701
```

This should not be interpreted as automatically better candidate quality. Q1 spreads functional credit more broadly than STRONG, but the absolute all-K oracle ceiling is lower.

Therefore:

> **higher effective K and stronger differentiation do not compensate for the loss of absolute candidate-set utility.**

---

# 7. Live policy also shifts toward C2

Q1 live-policy execution at `t=0`:

```text
execute fraction = 99.119%
STOP fraction    =  0.881%
```

Selected fraction conditional on execute:

```text
C0 = 30.790%
C1 = 14.338%
C2 = 46.101%
C3 =  8.771%
```

The selector is therefore aligned with the new C2-heavy functional regime.

This is not a four-way equal-usage objective, and equal usage is not required. The issue is that the same C2 branch also becomes overwhelmingly dominant in persistent terminal task-gradient allocation.

---

# 8. Training-time `L_safe` behavior: not global no-op collapse

At the best checkpoint epoch 19, the training logs reported:

| Slot | Harmful fraction | Positive fraction | Mean `delta_q` norm |
|---|---:|---:|---:|
| C0 | 49.185% | 50.802% | 0.191208 |
| C1 | 68.867% | 31.129% | 0.067206 |
| C2 | 24.560% | 75.440% | 0.633693 |
| C3 | 61.552% | 38.431% | 0.123071 |

This rules out a simple global no-op collapse.

C2 remains highly active:

```text
mean_delta_q_norm C2 ≈ 0.634
```

while C1 and C3 become much more conservative:

```text
C1 ≈ 0.067
C3 ≈ 0.123
```

A useful qualitative description is:

```text
C2: large edits + strong TRAIN safety
C0: medium edits + approximately balanced safety
C1: small edits + still frequently harmful
C3: small edits + still frequently harmful
```

Therefore Q1 creates an asymmetric response to the no-harm objective rather than shrinking every slot uniformly.

---

# 9. Large TRAIN-to-VAL safety gap for C2

C2 is especially important.

At epoch 19 training-time logging:

```text
C2 harmful fraction ≈ 24.56%
```

On full FashionIQ validation:

```text
C2 harmful fraction ≈ 54.52%
```

This is a very large generalization gap.

The current `L_safe` teacher is an in-batch, false-negative-safe retrieval judgment. The evidence therefore suggests:

> **C2 learns the TRAIN safety objective strongly, but that safety improvement does not transfer proportionally to the full validation distribution.**

This is another reason not to simply increase `lambda_safe`.

Increasing `lambda_safe` could strengthen optimization against the same training teacher while further suppressing risky/high-value edits, without repairing validation utility.

---

# 10. N=512 gradient diagnostic: winner-take-most persists

The Q1 N=512 diagnostic used:

```text
gradient_batches    = 64
gradient_batch_size = 8
gradient_samples    = 512
```

with the correct reconstructed objective:

```text
lambda_safe = 0.1
```

## 10.1 Terminal gradient at live `t=0` proposal tensors

Q1 terminal-gradient energy:

| Slot | Mean grad norm | Nonzero fraction | Energy share |
|---|---:|---:|---:|
| C0 | 0.005296 | 31.25% | 36.35% |
| C1 | 0.002067 | 13.09% | 19.29% |
| C2 | 0.007128 | 48.05% | 38.81% |
| C3 | 0.000924 | 6.45% | 5.55% |

At the current live proposal tensor, C0 and C2 both receive substantial terminal-task exposure.

This is less skewed than the persistent query-row result below.

## 10.2 Terminal gradient at learned proposal-query rows

The parameter-level terminal-gradient energy split is:

| Slot | STRONG | Q1 |
|---|---:|---:|
| C0 | 7.50% | 4.59% |
| C1 | 2.73% | 2.85% |
| C2 | 4.73% | **90.31%** |
| C3 | **85.05%** | 2.25% |

Q1 query-row statistics:

| Slot | Mean grad norm | Nonzero fraction | Energy share |
|---|---:|---:|---:|
| C0 | 0.167984 | 95.31% | 4.59% |
| C1 | 0.085408 | 65.63% | 2.85% |
| C2 | 0.873763 | 100.00% | **90.31%** |
| C3 | 0.068807 | 42.19% | 2.25% |

This is the decisive Q1 optimization result.

Under STRONG:

```text
C3 owned ≈85% of persistent terminal query-row gradient energy.
```

Under Q1:

```text
C2 owns ≈90% of persistent terminal query-row gradient energy.
```

Therefore:

> **`L_safe` does not resolve winner-take-most terminal task learning. It changes which specialist wins the reinforcement loop.**

This strongly argues against interpreting Q1 as a solution to slot starvation.

---

# 11. What Q1 does and does not establish

## Supported

The evidence supports all of the following:

1. `L_safe=0.1` materially changes candidate learning.
2. Q1 improves official Mean Recall versus STRONG by about +1.29 points.
3. Q1 improves mean utility and/or harmful fraction for important slots, especially C0 and C2.
4. Q1 is not a global no-op collapse.
5. Q1 lowers the full-set oracle candidate ceiling.
6. Q1 shifts functional and live-policy dominance from C3 to C2.
7. Q1 leaves persistent terminal task-gradient allocation extremely concentrated.
8. Winner-take-most remains present at the learned proposal-query level.
9. Increasing `lambda_safe` is not the next justified intervention.

## Not established

The evidence does **not** establish:

1. that `L_safe` is intrinsically harmful for every weighting or teacher design;
2. that C2 was historically gradient-starved throughout training;
3. that DPP directly opposes the retrieval task at every stage;
4. that equal 25% slot usage would improve the model;
5. that reducing candidate magnitude alone explains all Q1 behavior;
6. that the recurrent architecture itself is invalid.

The gradient diagnostic remains a static-checkpoint diagnostic. It measures current exposure, not complete temporal causality.

---

# 12. Important diagnostic limitation: `L_safe` gradient is not separately attributed

The current gradient diagnostic attributes:

```text
terminal
concept
bind
dpp
pair
gain
```

but does not separately report a `safe` gradient component.

Therefore the current N=512 result supports the statement:

> **terminal task-gradient winner-take-most shifts from C3 to C2**

but should not be misreported as a direct measurement of how `L_safe` gradient itself is distributed across slots.

If a future mechanistic audit needs that question, `safe` can be added as a separate read-only attribution component. It is not required before the next intervention.

---

# 13. Q1 verdict

Q1's intended hypothesis was approximately:

> Direct no-harm supervision on all candidates will improve specialist candidate quality without destroying useful specialization, thereby raising the candidate-set ceiling and reducing winner-take-most failure.

Observed result:

```text
Official Mean Recall                improved
Mean candidate utility              improved
Some harmful fractions              improved
Global no-op collapse               no
Effective functional K              increased
All-K oracle ceiling                worsened
Persistent task-gradient monopoly   persisted
Dominant branch                     C3 -> C2
```

Therefore:

> **Q1 = PARTIAL POSITIVE / PRIMARY HYPOTHESIS FAILED**

The intervention is scientifically useful because it identifies a more precise limitation of pure no-harm supervision:

> **A quality floor is not equivalent to positive task learning for every useful candidate.**

`L_safe` says:

```text
"do not become worse than the parent"
```

but does not ensure:

```text
"become strongly useful for retrieval"
```

This distinction is now empirically visible.

---

# 14. Next experiment: Q2 candidate CE bootstrap

The next isolated intervention should target positive candidate task learning directly.

At a one-step screening regime:

\[
w_k
=
\operatorname{sg}
\left[
\operatorname{softmax}_k(-\ell_k/T_a)
\right]
\]

and:

\[
L_{\mathrm{cand}}
=
\mathbb E_i
\sum_k
w_k\ell_k.
\]

The purpose is to give multiple candidate branches direct retrieval supervision instead of only penalizing regressions.

## First Q2 test should be isolated

Start with a matched `T=1` experiment:

```text
CONTROL:
STRONG
T = 1
lambda_safe = 0
lambda_cand = 0

TREATMENT:
STRONG
T = 1
lambda_safe = 0
candidate CE bootstrap enabled
lambda_cand > 0
```

Do **not** simultaneously:

- increase `lambda_safe`;
- redesign DPP;
- add useful-subset DPP;
- add ST/Gumbel;
- change K;
- tune ScoreNet;
- force equal slot usage.

The first Q2 decision should ask:

1. Does all-K oracle increase?
2. Does positive candidate utility improve?
3. Does persistent terminal-gradient allocation become less winner-take-most?
4. Do candidate edit norms remain non-degenerate?
5. Does official Mean Recall improve or at least preserve the best Q1/OLD trade-off?

Only after direct candidate-quality learning is stable should the DPP formulation be revisited.

---

# 15. Research status after Q1

```text
OLD
high quality
high redundancy
Mean Recall ≈ 30.824
        |
        v
STRONG
real specialization
poor candidate safety
C3 terminal-gradient monopoly
Mean Recall ≈ 28.245
        |
        v
Q1 L_safe=0.1
partial safety recovery
Mean Recall ≈ 29.537
all-K oracle decreases
C2 becomes dominant
C2 terminal-gradient monopoly ≈ 90%
        |
        v
NEXT
Q2 direct candidate CE bootstrap
screen at T=1
```

The current research target is no longer simply "make the candidates different."

It is:

> **preserve useful specialization while distributing positive retrieval-task learning across multiple candidate specialists strongly enough that diversity does not come at the expense of candidate-set quality.**