# CIR / TAPER A3.3 — VALUE SOURCE × SLOT EFFECT 2×2 ABLATION DIAGNOSTIC CHECKPOINT

**Date:** 2026-08-27  
**Repository:** `Le-Minh-Nhut/cir`  
**Branch:** `exp/e2e-a3.3-value-source-slot-effect-ablation`  
**Experiment:** clean 2×2 factorial ablation over VALUE source and direct `slot_effects` injection  
**Assignment mode held fixed:** `soft_shared`  
**Validation samples:** 6016  
**P0 audit:** exact `2^K` coalition intervention with 4 Edit Slots

---

# 0. PURPOSE

A3.1 → A3.2 changed two major information pathways at the same time:

1. VALUE source changed from contextual/post-Q-Former to raw/pre-Q-Former;
2. direct teacher counterfactual `slot_effects = q_full - q_minus` was removed from the Edit-Slot latent.

Therefore the previous Recall drop could not be causally attributed.

A3.3 exists to answer one clean question:

> **Was A3.2 weak because raw VALUE itself is weak, because removing `slot_effects` removed a powerful information highway, or because both changes interact?**

The experiment deliberately does **not** introduce:

- hard-exclusive VALUE;
- Entmax / Sinkhorn / OT;
- balance loss;
- functional error ownership;
- residual pursuit;
- new multi-error loss;
- local microencoder.

Only two factors vary.

---

# 1. THE 2×2 FACTORIAL DESIGN

| Run | VALUE source | `slot_effect_in_value` | Interpretation |
|---|---|---:|---|
| **A** | contextual/post-Q-Former | ON | A3.1-like information path |
| **B** | raw/pre-Q-Former | ON | isolate VALUE-source change while preserving teacher effect |
| **C** | contextual/post-Q-Former | OFF | isolate removal of direct teacher effect |
| **D** | raw/pre-Q-Former | OFF | A3.2-soft information setting |

All four runs use:

```yaml
slot_value_assignment: soft_shared
num_slots: 4
seed: 42
num_epochs: 10
retrieval objective: unchanged
QASA: unchanged
Executor: unchanged
```

The purpose is causal attribution, not architecture optimization.

---

# 2. CHECKPOINTS

```text
A contextual + effect ON
outputs/2026-08-26/23-40-11/best.pt

B raw + effect ON
outputs/2026-08-27/00-08-41/best.pt

C contextual + effect OFF
outputs/2026-08-27/00-35-54/best.pt

D raw + effect OFF
outputs/2026-08-27/00-59-21/best.pt
```

P0 reports:

```text
reports/a3_3_2x2/A_contextual_effect_on_p0_full.json
reports/a3_3_2x2/B_raw_effect_on_p0_full.json
reports/a3_3_2x2/C_contextual_effect_off_p0_full.json
reports/a3_3_2x2/D_raw_effect_off_p0_full.json
```

---

# 3. TRAINING RESULT — FIRST-ORDER PERFORMANCE VIEW

Best training Mean Recall:

| Run | Best Mean Recall |
|---|---:|
| **A** contextual + effect ON | **58.5844** |
| **B** raw + effect ON | **56.7696** |
| **C** contextual + effect OFF | **58.7502** |
| **D** raw + effect OFF | **41.4585** |

At first glance:

```text
A ≈ C
B moderately below A
D catastrophically below all others
```

This already falsifies the simple explanations:

```text
"slot_effects alone are the main source of A3.1 performance"
```

and

```text
"raw VALUE alone necessarily destroys performance"
```

The actual story is an interaction.

---

# 4. CLEAN CAUSAL CONTRASTS

Let endpoint Mean Recall be `R`.

Observed:

```text
R_A = 58.5844
R_B = 56.7696
R_C = 58.7502
R_D = 41.4585
```

## 4.1 Effect of changing VALUE when `slot_effects = ON`

\[
A - B = 58.5844 - 56.7696 = 1.8148
\]

So with teacher counterfactual effects available:

```text
contextual VALUE -> raw VALUE
costs only ~1.81 Mean Recall
```

This is a surprisingly small drop.

Therefore:

> **Raw VALUE is not intrinsically incapable of supporting strong retrieval when the global teacher-effect pathway remains available.**

---

## 4.2 Effect of removing `slot_effects` when VALUE is contextual

\[
C - A = 58.7502 - 58.5844 = +0.1658
\]

Removing `slot_effects` under contextual VALUE causes essentially no loss.

In this run it is even slightly better, but the difference is too small to interpret as a true improvement from one seed.

Therefore:

> **Contextual/post-Q-Former VALUE already carries almost everything needed for endpoint retrieval; the explicit teacher effect is largely redundant when contextual VALUE exists.**

---

## 4.3 Effect of removing `slot_effects` when VALUE is raw

\[
B - D = 56.7696 - 41.4585 = 15.3111
\]

This is the largest contrast in the entire experiment.

When raw VALUE is used:

```text
slot_effects ON  -> 56.77
slot_effects OFF -> 41.46
```

Therefore:

> **Raw VALUE loses a major semantic/global information source, and `slot_effects` almost completely compensates for that loss.**

This is the strongest causal result of A3.3.

---

# 5. FACTOR INTERACTION

If the two factors were independent, we would expect approximately:

```text
effect(VALUE source) + effect(slot_effect removal)
```

But observed behavior is strongly non-additive.

Difference-in-differences:

\[
(A-B) - (C-D)
\]

with:

\[
A-B = 1.8148
\]

and:

\[
C-D = 17.2917
\]

so:

\[
\Delta_{\text{interaction}} \approx -15.4769
\]

Equivalent interpretation:

> `slot_effects` matter very little with contextual VALUE but matter enormously with raw VALUE.

Thus the two information highways are highly substitutable:

```text
contextual VALUE
        OR
teacher counterfactual slot_effects
```

Either one can largely support strong retrieval.

Removing both at once exposes the weak D regime.

---

# 6. P0 MASTER TABLE

| Metric | A contextual + effect ON | B raw + effect ON | C contextual + effect OFF | D raw + effect OFF |
|---|---:|---:|---:|---:|
| Mean Recall, QASA | **58.58** | 56.78 | **58.75** | 41.47 |
| all-slots Mean Recall | 58.88 | 43.43 | 58.81 | 34.99 |
| Hard partition mean K | 1.175 | **1.000** | 2.079 | **1.000** |
| Dominant hard token share | 0.976 | **1.000** | 0.714 | **1.000** |
| Gradient error-mode rank | 2.837 | 2.826 | 2.839 | 2.637 |
| Functional Phi effective rank | 1.018 | 1.160 | **0.996** | 1.168 |
| Functionally useful slots | 3.903 | 3.905 | 3.908 | 3.898 |
| QASA functional precision | 0.974 | 0.916 | **0.977** | 0.930 |
| QASA functional recall | 0.497 | 0.232 | **0.521** | 0.238 |
| best SINGLE/FULL | 0.790 | **1.042** | 0.777 | **1.034** |
| best REPEAT/FULL | 1.037 | **1.448** | 1.020 | **1.754** |
| MEANxK/FULL | 1.000 | 1.038 | 1.000 | 0.972 |
| K95 | 2.514 | 1.891 | 2.559 | 1.837 |
| K99 | 2.959 | 2.069 | 3.011 | 2.038 |

---

# 7. TASK GEOMETRY IS CONSISTENTLY MULTI-MODE

Across all four runs:

```text
A gradient rank = 2.837
B gradient rank = 2.826
C gradient rank = 2.839
D gradient rank = 2.637
```

This is important because the retrieval problem presents roughly `2.6–2.8` effective error directions regardless of which architecture cell is used.

Therefore the collapse cannot be dismissed as:

```text
"the benchmark naturally only needs one edit direction"
```

The external task geometry stays multi-mode.

---

# 8. FUNCTIONAL SLOT GEOMETRY REMAINS APPROXIMATELY RANK-1 IN ALL FOUR RUNS

Functional Phi effective rank:

```text
A = 1.018
B = 1.160
C = 0.996
D = 1.168
```

Despite very different:

- VALUE information;
- ownership geometry;
- endpoint Recall;
- presence/absence of teacher effects;

all four models converge to nearly rank-1 functional behavior.

This is one of the strongest results of the experiment.

\[
\text{task rank} \approx 2.6-2.8
\]

while

\[
\text{slot functional rank} \approx 1.0-1.17.
\]

Therefore:

> **Neither contextual VALUE nor raw VALUE nor `slot_effects` creates true functional slot decomposition.**

They mostly change how much information the collapsed solution can carry.

---

# 9. RUN A — CONTEXTUAL VALUE + SLOT EFFECT ON

A is effectively the A3.1-like information-rich setting.

Results:

```text
Mean Recall                    = 58.58
hard K                         = 1.175
dominant share                 = 0.976
gradient rank                  = 2.837
Phi rank                       = 1.018
SINGLE/FULL                    = 0.790
REPEAT/FULL                    = 1.037
MEANxK/FULL                    = 1.000
K95/K99                        = 2.514 / 2.959
```

## Diagnosis

A has excellent endpoint retrieval but almost complete functional degeneracy.

The representation gives each slot access to highly contextual/global information and also provides direct teacher counterfactual effects.

The model therefore solves the task strongly without learning distinct functional edit factors.

`MEANxK/FULL = 1.000` is especially revealing:

```text
destroy slot identity
average their content
repeat the mean
```

and the system still recovers effectively all forced-full gain.

A is thus:

> **high performance, low identifiability, low functional specialization.**

---

# 10. RUN C — CONTEXTUAL VALUE + SLOT EFFECT OFF

C is the cleanest test of whether `slot_effects` caused A3.1's strong performance.

Results:

```text
Mean Recall                    = 58.75
hard K                         = 2.079
dominant share                 = 0.714
gradient rank                  = 2.839
Phi rank                       = 0.996
SINGLE/FULL                    = 0.777
REPEAT/FULL                    = 1.020
MEANxK/FULL                    = 1.000
K95/K99                        = 2.559 / 3.011
```

## Main conclusion

Removing the explicit teacher effect:

```text
A -> C
```

does **not** hurt endpoint Recall.

Therefore the prior belief:

```text
A3.1 may be strong mainly because q_full - q_minus is injected directly
```

is falsified as a primary explanation.

More importantly, C has a much healthier hard ownership diagnostic:

```text
K:        1.175 -> 2.079
dominant: 0.976 -> 0.714
```

but functional Phi rank gets no better:

```text
1.018 -> 0.996
```

This is an especially clean demonstration that:

\[
\boxed{\text{token ownership diversity} \neq \text{functional specialization}}
\]

C can distribute token winners across ~2 slots while all slots still act in nearly one functional direction.

---

# 11. RUN B — RAW VALUE + SLOT EFFECT ON

B is the most informative shortcut cell.

Results:

```text
Mean Recall                    = 56.78
all-slots Mean Recall          = 43.43
hard K                         = 1.000
dominant share                 = 1.000
gradient rank                  = 2.826
Phi rank                       = 1.160
SINGLE/FULL                    = 1.042
REPEAT/FULL                    = 1.448
MEANxK/FULL                    = 1.038
K95/K99                        = 1.891 / 2.069
QASA functional recall         = 0.232
```

## 11.1 Absolute routing monopoly

Every valid token has the same argmax winner:

```text
hard K = 1.000
dominant = 1.000
```

So raw VALUE + teacher effect does not encourage decomposition.

The routing has fully collapsed.

---

## 11.2 Strong retrieval survives anyway

Despite absolute token monopoly:

```text
Mean Recall = 56.78
```

which is only ~1.8 below A.

Therefore high retrieval performance clearly does **not** require healthy token specialization.

---

## 11.3 Teacher effect acts as a bypass/rescue pathway

The clean comparison:

```text
B = raw + effect ON  = 56.78
D = raw + effect OFF = 41.47
```

shows that when raw VALUE loses contextual/global semantics, `slot_effects` supplies enough information to restore most endpoint quality.

Thus `slot_effects` behaves as a powerful compensating information channel.

This is not automatically a bug: the effect vector contains legitimate teacher counterfactual information.

But for the research goal of latent edit-factor decomposition, it is a shortcut because it allows strong retrieval without forcing Edit Slots to extract and specialize their own information.

---

## 11.4 QASA versus all-slots gap is huge

B:

```text
qasa_full Mean      = 56.78
all_slots_full Mean = 43.43
```

Difference:

\[
+13.35
\]

This means blindly forcing all slots is much worse than QASA selecting a subset.

Combined with:

```text
SINGLE/FULL = 1.042
```

the model is behaving like:

```text
one useful/global slot
+
harmful or redundant additional slots
```

rather than a complementary coalition.

---

# 12. RUN D — RAW VALUE + SLOT EFFECT OFF

D is the information-restricted cell.

Results:

```text
Mean Recall                    = 41.47
all-slots Mean Recall          = 34.99
hard K                         = 1.000
dominant share                 = 1.000
gradient rank                  = 2.637
Phi rank                       = 1.168
SINGLE/FULL                    = 1.034
REPEAT/FULL                    = 1.754
MEANxK/FULL                    = 0.972
K95/K99                        = 1.837 / 2.038
```

## Diagnosis

D removes both powerful global information highways:

```text
contextual post-Q-Former VALUE
AND
teacher counterfactual slot_effects
```

The resulting raw weighted word-embedding representation is much weaker.

But importantly, it **still does not specialize**.

Instead the model collapses even harder:

```text
hard K = 1
dominant = 1
SINGLE/FULL > 1
REPEAT/FULL = 1.754
Phi rank ≈ 1.17
```

Therefore A3.2's failure was not:

```text
"the model almost specialized but representation was simply too weak"
```

It is:

```text
representation became weak
AND
functional decomposition still never emerged
```

This distinguishes two separate problems:

1. **representation sufficiency**
2. **functional credit specialization**

Both must be solved.

---

# 13. THE MOST IMPORTANT RESULT: TWO SUBSTITUTE GLOBAL INFORMATION HIGHWAYS

A3.3 reveals a previously hidden structure.

## Highway 1 — contextual VALUE

```text
post-Q-Former text tokens
```

These vectors are already globally/contextually enriched.

C proves this highway alone is sufficient:

```text
contextual + effect OFF -> 58.75
```

---

## Highway 2 — teacher counterfactual effect

```text
slot_effects = q_full - q_minus
```

B proves this highway can compensate when VALUE becomes raw:

```text
raw + effect ON -> 56.78
```

---

## Remove both

D:

```text
raw + effect OFF -> 41.47
```

Therefore the old A3.1 representation was overdetermined:

```text
contextual information path
+
teacher counterfactual information path
```

Either can support a strong collapsed solution.

This explains why removing only one does not force specialization.

---

# 14. A3.2 REINTERPRETED

Previous A3.2 changed:

```text
contextual -> raw VALUE
slot_effect ON -> OFF
```

simultaneously.

The 2×2 now proves that most of its Recall collapse was caused by the **combination**, not by either isolated change.

Quantitatively:

```text
A -> B : -1.81
A -> C : +0.17
A -> D : -17.13
```

So the A3.2 degradation should no longer be described as:

> "raw VALUE destroyed performance."

The correct statement is:

> **Raw VALUE becomes too weak when the teacher counterfactual information highway is also removed.**

---

# 15. REPRESENTATION POWER AND SPECIALIZATION ARE ORTHOGONAL

The four runs occupy different points in two dimensions.

| Run | Representation / information power | Functional specialization |
|---|---|---|
| A | high | poor |
| B | high due to effect bypass | poor |
| C | high due to contextual VALUE | poor |
| D | low | poor |

This is the central scientific lesson.

Improving representation power can restore Recall:

```text
D -> B
D -> C
```

but does not make Phi rank approach task rank.

Restricting representation power can remove shortcuts:

```text
A -> D
```

but does not automatically create specialization either.

Thus:

\[
\boxed{\text{good information} \neq \text{specialization pressure}}
\]

and

\[
\boxed{\text{removing information shortcuts} \neq \text{creating specialization}}
\]

---

# 16. WHY C IS ESPECIALLY IMPORTANT

C has:

```text
hard K = 2.079
dominant = 0.714
```

which superficially looks much healthier than A/B/D.

Yet:

```text
Phi rank = 0.996
MEANxK/FULL = 1.000
REPEAT/FULL = 1.020
```

This is almost a controlled counterexample to any claim that better ownership metrics prove specialization.

The tokens are split more broadly.

The function is not.

Therefore future experiments must never use only:

- slot mass;
- active slot count;
- winner entropy;
- mask overlap;
- QASA K;

as evidence for successful decomposition.

They are routing-health metrics only.

---

# 17. WHY "FUNCTIONALLY USEFUL SLOTS ≈ 4" IS MISLEADING

All four runs report:

```text
~3.9 functionally useful slots
```

while Phi rank remains ~1.

This means each slot can have some positive marginal effect while those effects are highly redundant/collinear.

Therefore:

```text
positive usefulness != unique responsibility
```

A correct specialization claim requires:

- multiple independent functional directions;
- low clone recovery;
- coalition complementarity;
- distinct error-mode ownership.

---

# 18. REPEAT ADVERSARY RESULTS

```text
A REPEAT/FULL = 1.037
B REPEAT/FULL = 1.448
C REPEAT/FULL = 1.020
D REPEAT/FULL = 1.754
```

Interpretation:

- A/C: a repeated single slot can approximately match full coalition;
- B/D: a repeated single slot massively outperforms the intended coalition.

This remains strong evidence for an executor-ticket / repeated-compute shortcut.

Especially in D:

```text
REPEAT/FULL = 1.754
```

one slot repeated through all executor opportunities recovers ~175% of full coalition gain.

Therefore functional collapse is not only an information problem.

The recurrent execution structure still allows one useful direction to be amplified repeatedly.

---

# 19. SINGLE-SLOT RESULTS

```text
A SINGLE/FULL = 0.790
B SINGLE/FULL = 1.042
C SINGLE/FULL = 0.777
D SINGLE/FULL = 1.034
```

A/C information-rich contextual runs still benefit somewhat from coalition participation.

But B/D raw runs have:

```text
best one slot >= full coalition
```

This is direct evidence of a giant functional owner.

For B/D, extra slots are not complementary; they can be neutral or harmful.

---

# 20. MEAN-SLOT ADVERSARY

```text
A = 1.000
B = 1.038
C = 1.000
D = 0.972
```

All four are close to 1.

Therefore even when ownership geometry changes, replacing slot identity with the mean remains almost sufficient.

This is extremely strong evidence that learned slot identities are not functionally essential.

The representation may contain different token mixtures, but the downstream system is insensitive to which slot means what.

---

# 21. QASA IS NOT CREATING DECOMPOSITION

QASA functional recall:

```text
A = 0.497
B = 0.232
C = 0.521
D = 0.238
```

Precision remains high:

```text
~0.92-0.98
```

Interpretation:

- when QASA selects a slot, it often selects a useful one;
- but many useful/redundant slots are not required;
- especially in raw regimes B/D, the deployed system behaves strongly around a very small selected functional core.

This is consistent with QASA acting as a useful selector, not as a force that creates distinct functional factors.

---

# 22. HYPOTHESIS STATUS AFTER A3.3

## H1 — `slot_effects` are the main source of A3.1 performance

**Status: REJECTED AS A GENERAL CLAIM**

Evidence:

```text
A = 58.58
C = 58.75
```

Contextual VALUE without `slot_effects` preserves performance.

---

## H2 — contextual VALUE is the main source of performance

**Status: PARTIALLY REJECTED**

Evidence:

```text
B = 56.78
```

Raw VALUE can still achieve strong performance if teacher effects are present.

Contextual VALUE is one strong information highway, not the only one.

---

## H3 — raw VALUE is intrinsically too weak

**Status: CONTEXT-DEPENDENT**

Raw alone:

```text
D = 41.47
```

Raw plus teacher effect:

```text
B = 56.78
```

Therefore raw VALUE lacks sufficient information on its own in this architecture, but the failure is not intrinsic to routing or optimization alone.

---

## H4 — removing the teacher effect should force slots to specialize

**Status: REJECTED**

C:

```text
Phi rank = 0.996
MEANxK/FULL = 1.000
```

No functional specialization emerges.

---

## H5 — removing contextual VALUE should force slots to specialize

**Status: REJECTED**

B/D:

```text
hard K = 1
Phi ~1.16
SINGLE >=1
```

It instead causes giant-slot monopoly.

---

## H6 — better token partitioning implies better functional specialization

**Status: STRONGLY REJECTED**

C is the clean counterexample:

```text
hard K = 2.079
dominant = 0.714
Phi rank = 0.996
```

---

## H7 — task itself is essentially one-dimensional

**Status: REJECTED BY CURRENT P0**

Gradient rank remains:

```text
2.637–2.839
```

in every cell.

---

## H8 — endpoint retrieval objective lacks direct specialization pressure

**Status: STRONGLY SUPPORTED**

Every representational regime converges to functional rank ~1 despite task rank ~2.7.

---

# 23. WHAT THIS EXPERIMENT FALSIFIES

A3.3 falsifies the following broad strategy:

> Keep removing global information paths until specialization automatically appears.

Why?

Because:

- A has many shortcuts and collapses;
- C removes one shortcut and collapses;
- B removes the other shortcut but retains teacher effects and collapses;
- D removes both and still collapses.

The only thing that changes reliably is endpoint power, not functional factorization.

Therefore information isolation is necessary for some scientific claims, but it is not a sufficient learning mechanism.

---

# 24. WHAT THE NEXT ARCHITECTURE MUST SOLVE

Two problems are now cleanly separated.

## Problem A — representation sufficiency

D shows raw pre-Q-Former weighted embeddings are too weak on their own.

Possible later fix:

```text
restricted local microencoder
```

that preserves:

- local phrase composition;
- order;
- negation;
- binding;

without reopening whole-caption/global leakage.

But this is only a representation repair.

---

## Problem B — functional specialization pressure

All A/B/C/D show Phi rank near 1.

This requires a learning mechanism that says:

```text
slot 0 already solved error subspace E0
slot 1 should receive reward for residual E1
clone of slot 0 should receive little/no marginal reward
```

Candidate theory mechanisms already identified:

- multi-error gradient matrix `G`;
- per-negative functional signature `Phi`;
- functional ownership assignment;
- block residual pursuit;
- pair lookahead;
- direct clone/repeat penalty or acceptance test;
- matched-compute executor control.

This is now the higher-priority scientific problem.

---

# 25. SHOULD THE NEXT STEP BE A LOCAL MICROENCODER?

Not immediately as the sole experiment.

A3.3 tells us:

```text
raw representation is weak
```

so a microencoder is justified eventually.

But:

```text
contextual representation is strong
and still Phi ≈ 1
```

Therefore a better encoder by itself will almost certainly restore performance without solving the central decomposition failure.

A microencoder should be treated as:

```text
representation capacity repair
```

not:

```text
specialization mechanism
```

The two should be audited separately.

---

# 26. SHOULD FUNCTIONAL ERROR OWNERSHIP BE IMPLEMENTED NOW?

Current evidence supports moving to it.

Why:

1. task error rank is consistently >2.6;
2. functional Phi rank stays ~1 under every information cell;
3. token ownership diversity can improve without functional rank improving;
4. giant-slot and clone/repeat attacks remain strong;
5. changing information source no longer appears capable of solving the core collapse.

Therefore the next high-value experiment should target:

```text
FUNCTIONAL ERROR OWNERSHIP / BLOCK MULTI-ERROR PURSUIT
```

while keeping representation choices explicit and controlled.

A reasonable staged strategy:

```text
Stage F0:
use a strong representation setting as a controlled baseline
(e.g. contextual VALUE, slot_effect OFF)

Stage F1:
add functional multi-error ownership only

Stage F2:
if functional specialization appears,
then replace contextual VALUE with a restricted local microencoder
and test whether specialization survives without global leakage
```

This prevents confounding representation weakness with failure of the new learning mechanism.

---

# 27. RECOMMENDED BASELINE FOR THE NEXT FUNCTIONAL EXPERIMENT

Among A-D, **C** is scientifically attractive as the next controlled baseline:

```text
contextual VALUE
slot_effect OFF
soft_shared
Mean Recall = 58.75
```

Reasons:

1. strong endpoint performance;
2. removes direct teacher counterfactual effect bypass;
3. has healthier token ownership than A/B/D;
4. still has unmistakable functional collapse:
   - Phi rank ≈ 1;
   - MEANxK/FULL = 1;
   - REPEAT/FULL ≈ 1.

Thus C provides a strong, non-catastrophically-weak substrate on which to test whether functional error ownership actually changes Phi geometry.

Important caveat:

```text
contextual VALUE is still globally enriched
```

so C is not a final clean decomposition architecture.

It is a good **mechanism-development baseline**.

After functional pressure works, information isolation must be revisited.

---

# 28. ACCEPTANCE CRITERIA FOR THE NEXT FUNCTIONAL MECHANISM

Do not call the next experiment successful merely because:

```text
hard K increases
slot entropy increases
mask overlap decreases
active slot count increases
```

Required P0 movement should include:

```text
functional Phi rank materially > 1
and move toward gradient task rank

SINGLE/FULL decreases on genuinely multi-error samples

REPEAT/FULL clearly < 1

MEANxK/FULL clearly < 1 where slot identity should matter

K95/K99 increase when multiple slots are genuinely necessary

QASA functional recall improves without merely selecting redundant slots

endpoint Recall remains usable
```

A reasonable qualitative target is:

```text
task rank ≈ 2.7
Phi rank should stop living around 1.0–1.2
```

Exact numerical thresholds should not be fixed before observing variance across seeds.

---

# 29. RED-TEAM CHECKS FOR THE NEXT EXPERIMENT

Any proposed specialization mechanism should be attacked with:

## Clone attack

```text
copy best slot into all positions
```

If Recall is preserved or improves, identity is still unnecessary.

## Mean attack

```text
mean slots -> repeat across positions
```

If performance stays ~FULL, representation remains clone-like.

## Single-slot attack

If one slot recovers full coalition on multi-error queries, giant functional ownership remains.

## Matched-compute attack

Compare:

```text
K distinct slots for T steps
vs
1 slot repeated for same T steps
```

to separate specialization from extra compute.

## Exact coalition audit

With K=4, enumerate all 16 coalitions.

No proxy metric should replace this.

## Error-mode audit

Measure task gradient rank and compare with learned Phi rank.

The mechanism only succeeds if learned functional dimensionality tracks real task dimensionality more closely.

---

# 30. FINAL DIAGNOSIS

The A3.3 2×2 ablation resolves the causal ambiguity left by A3.1 → A3.2.

The observed system contains **two largely substitutable high-information pathways**:

```text
1. contextual/post-Q-Former VALUE
2. teacher counterfactual slot_effects
```

Either is sufficient to support strong retrieval:

```text
A = 58.58
B = 56.78
C = 58.75
```

Removing both produces the weak D regime:

```text
D = 41.47
```

However this performance story is almost orthogonal to the specialization story.

Across every cell:

```text
task gradient rank ≈ 2.6–2.8
functional Phi rank ≈ 1.0–1.17
```

and clone/single/repeat adversaries remain strong.

Therefore the central failure is no longer best described as:

```text
"the slots see too much global information"
```

or:

```text
"raw VALUE is too weak"
```

The deeper diagnosis is:

> **The model has no learning pressure that assigns distinct retrieval-error responsibilities to distinct slots. Information-rich pathways make the collapsed solution powerful; information-poor pathways make it weak; neither condition creates functional decomposition by itself.**

---

# 31. ONE-SENTENCE CHECKPOINT

> **A3.3 proves that contextual VALUE and teacher `slot_effects` are substitute information highways: either can rescue endpoint Recall, but all four 2×2 cells remain functionally near rank-1, so the core unsolved problem is explicit functional error ownership—not merely VALUE locality or removal of global information shortcuts.**