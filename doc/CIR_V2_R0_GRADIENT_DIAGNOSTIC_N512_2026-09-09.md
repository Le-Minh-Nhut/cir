# CIR V2 R0 — Large-Sample Gradient Diagnostic and Failure Mechanism Update

**Date:** 2026-09-09  
**Branch:** `exp/e2e-iag-srme-v2-r0`  
**Diagnostic git SHA:** `97f521f5e8c5bbba8d930fe8610ba0b0813b9287`  
**Dataset:** FashionIQ validation  
**Primary timestep:** `t=0`  
**Gradient probe:** `64` batches × `8` samples = `512` samples  
**Persistent manifest:** `doc/diagnostics/slot_specialization_2026-09-09/shared_slot_specialization_val_all.json`  
**Purpose:** verify whether the small-N gradient findings survive a substantially larger matched probe, and refine the current failure diagnosis before intervention.

---

## 1. Executive conclusion

The N≈512 matched gradient diagnostic confirms that the earlier N≈32 observation was not merely small-sample noise.

The strongest current result is:

> **STRONG has developed a winner-take-most task-gradient allocation toward C3 at the learned proposal-query level, while DPP pressure is concentrated mainly on C2.**

The most important parameter-level terminal-gradient energy split is:

| Slot | OLD | STRONG |
|---|---:|---:|
| C0 | 23.73% | 7.50% |
| C1 | 21.17% | 2.73% |
| C2 | 11.17% | 4.73% |
| C3 | 43.93% | **85.05%** |

Thus, in STRONG, approximately **85% of the terminal task-gradient energy reaching the learned proposal-query rows is concentrated on C3**.

At the live `t=0` proposal tensor, the same asymmetry exists but is less extreme:

| Slot | OLD terminal | STRONG terminal |
|---|---:|---:|
| C0 | 22.30% | 13.72% |
| C1 | 29.60% | 23.11% |
| C2 | 17.73% | 16.93% |
| C3 | 30.37% | **46.24%** |

Therefore the failure should **not** be described as:

- four completely dead candidates,
- pure selector collapse,
- pure architectural C3 bias,
- proven historical gradient starvation from initialization,
- or strong direct task-vs-DPP gradient opposition.

A more accurate current description is:

> **STRONG successfully breaks the OLD redundancy and creates real specialization, but candidate quality degrades severely. The retrieval task increasingly trains C3, while auxiliary diversity pressure is disproportionately spent elsewhere—especially C2. This creates a winner-take-most reinforcement loop in which specialization exists, but quality learning is not distributed well enough across the specialist candidates.**

In plain language:

> The four candidates have started learning different jobs, which is good. But C3 receives most of the useful retrieval-task teaching signal, while C0/C1/C2 receive much less task learning. DPP strongly pushes C2 to be different, but “being different” is not the same as “being useful.” The result is specialization without uniformly strong specialist quality.

The next intervention should therefore target **candidate quality/safety**, not force equal slot usage or erase specialization.

---

# 2. Experimental setup and validity

## 2.1 Checkpoints

### OLD

```text
checkpoint: outputs/r0_ncls_text/best.pt
epoch:      15
metric:     30.824426313241325
identity:   R0-NCLS-TEXT
```

### STRONG

```text
checkpoint: outputs/r0_ncls_text_strong_aux/best.pt
epoch:      18
metric:     28.244677186012268
identity:   R0-NCLS-TEXT
```

The benchmark difference remains:

```text
STRONG - OLD ≈ -2.57975 Mean Recall
```

So STRONG is behaviorally worse even though it exhibits stronger slot differentiation.

---

## 2.2 Matched gradient cohort

Both diagnostics used:

```text
gradient_batches:    64
gradient_batch_size: 8
gradient_sample_count: 512
```

The gradient sample fingerprints are identical:

```text
gradient_sample_ids_sha256:
8dc761957d425bb51078cb0e50307531e2dd26f46b89589ce0039628d51bfb54
```

The teacher-batch grouping fingerprints are also identical:

```text
gradient_teacher_batch_grouping_sha256:
00a2dca8094e5dcbed0f0b6e69126080cf94c7ffb787de2cd9d37c8659a22e67
```

This matters because the canonical marginal teacher utility depends on the matched in-batch negative pool.

Therefore OLD and STRONG are being compared on the same:

- FashionIQ validation samples,
- ordering,
- batch grouping,
- negative pools,
- seed,
- batch size,
- diagnostic implementation.

This removes the main comparability weakness of the earlier N≈32 probe.

---

## 2.3 Objective coefficients used by the gradient attribution

The diagnostic explicitly reports the coefficients used for each attributed component.

### OLD

```text
terminal = 1.0
concept  = 0.01
bind     = 0.01
dpp      = 0.01
pair     = 0.5
gain     = 0.5
```

### STRONG

```text
terminal = 1.0
concept  = 0.6
bind     = 1.0
dpp      = 0.6
pair     = 0.5
gain     = 0.5
```

For interpreting this gradient experiment, the relevant source is the diagnostic field:

```text
gradient_attribution.objective_component_coefficients
```

because the attribution code scales each component by the objective coefficient actually reconstructed for that checkpoint.

---

# 3. What exactly is being measured?

The diagnostic produces two different gradient views. They should not be conflated.

## 3.1 Gradient with respect to live `t=0` proposals

This asks:

> For the current forward pass, how much gradient from a loss component is flowing through each slot's proposal tensor at `t=0`?

The report calls this:

```text
full_objective_gradient_wrt_t0_proposals
```

This is a per-sample view and has:

```text
observation_count = 512
```

Because the full recurrent objective is differentiated with respect to the `t=0` proposal, downstream recurrent effects may contribute.

Therefore this is **not** a pure local one-step derivative.

---

## 3.2 Gradient with respect to learned proposal-query rows

This asks:

> How much gradient reaches the learned query row that parameterizes each candidate identity?

The report calls this:

```text
parameter_level_aggregate_query_row_gradient
```

There are 64 observations because one observation is one diagnostic-batch aggregate gradient.

This is especially important for the current diagnosis because the proposal-query rows are persistent learned parameters corresponding to candidate identities.

It answers a more structural question:

> Across matched batches, which candidate's learned slot parameters are receiving the task training signal?

This is where STRONG shows the most dramatic asymmetry.

---

# 4. Terminal task gradient — large-N result

## 4.1 Live `t=0` proposal tensor

### OLD

| Slot | Mean grad norm | Nonzero fraction | Terminal energy |
|---|---:|---:|---:|
| C0 | 0.002506 | 19.53% | 22.30% |
| C1 | 0.002959 | 22.46% | 29.60% |
| C2 | 0.002103 | 18.36% | 17.73% |
| C3 | 0.004159 | 36.52% | 30.37% |

OLD is not perfectly symmetric at the task-gradient level, but no single slot owns most of the proposal-level terminal energy.

C3 is largest at ~30%, not overwhelming.

### STRONG

| Slot | Mean grad norm | Nonzero fraction | Terminal energy |
|---|---:|---:|---:|
| C0 | 0.002536 | 13.48% | 13.72% |
| C1 | 0.003869 | 23.83% | 23.11% |
| C2 | 0.004037 | 29.88% | 16.93% |
| C3 | 0.006506 | 28.91% | **46.24%** |

The current proposal tensor is therefore substantially more C3-skewed under STRONG.

The important shift is:

```text
C3 terminal proposal-gradient share:
OLD   30.37%
STRONG 46.24%
```

At the same time C0 loses substantial share:

```text
C0:
OLD   22.30%
STRONG 13.72%
```

This is consistent with task specialization becoming concentrated.

---

# 5. Learned proposal-query gradient — strongest evidence

This is the key result.

## 5.1 OLD

Terminal task-gradient energy on learned proposal-query rows:

```text
C0 23.73%
C1 21.17%
C2 11.17%
C3 43.93%
```

C3 already has somewhat more task exposure in OLD, so there may be some inherent or emergent asymmetry even before STRONG.

However, the other slots still collectively receive the majority:

```text
C0 + C1 + C2 ≈ 56.07%
C3            ≈ 43.93%
```

Thus OLD is not winner-take-most at this level.

---

## 5.2 STRONG

Terminal task-gradient energy on learned proposal-query rows:

```text
C0  7.50%
C1  2.73%
C2  4.73%
C3 85.05%
```

Now:

```text
C0 + C1 + C2 ≈ 14.95%
C3            ≈ 85.05%
```

This is a major qualitative change.

C3 alone receives about:

```text
85.05 / 14.95 ≈ 5.69×
```

the terminal-gradient energy of all three other query rows combined.

Relative to each individual slot, the imbalance is even larger.

Approximate C3-to-slot energy ratios:

```text
C3 / C0 ≈ 11.35×
C3 / C1 ≈ 31.21×
C3 / C2 ≈ 17.98×
```

This is the strongest current evidence for winner-take-most task learning.

---

# 6. Does this confirm the earlier N≈32 result?

Yes, with an important refinement.

The earlier small probe suggested approximately:

```text
STRONG terminal query-row gradient:
C3 ≈ 93.5%
```

That number was too unstable to use as a strong quantitative conclusion because N was very small.

With 64 matched batch observations / 512 samples:

```text
C3 ≈ 85.05%
```

The exact percentage decreases, but the phenomenon remains extreme.

Therefore:

> The earlier qualitative conclusion survives the larger sample.

What is now well supported is:

```text
STRONG task learning is strongly concentrated on C3.
```

What is **not** supported is a claim such as:

```text
C3 always receives exactly ~94% of the task gradient.
```

The large-N estimate should replace the small-N number.

---

# 7. DPP gradient allocation

The DPP result is almost the mirror image of the terminal task result.

## 7.1 DPP at live proposal tensor

### OLD

```text
C0  9.38%
C1 22.66%
C2  7.44%
C3 60.53%
```

Because OLD's DPP coefficient is only `0.01`, the absolute DPP contribution is tiny relative to task.

### STRONG

```text
C0 14.83%
C1 20.92%
C2 60.96%
C3  3.29%
```

The STRONG DPP pressure is heavily concentrated on C2 at the live proposal tensor.

---

## 7.2 DPP at learned proposal-query rows

### OLD

```text
C0 30.77%
C1 21.95%
C2  7.09%
C3 40.19%
```

### STRONG

```text
C0 26.27%
C1 21.74%
C2 50.67%
C3  1.31%
```

This is a striking division of labor:

```text
terminal task → overwhelmingly C3
DPP diversity → mostly C2
```

For STRONG:

```text
terminal query gradient:
C3 ≈ 85.05%

DPP query gradient:
C2 ≈ 50.67%
C3 ≈ 1.31%
```

This does **not** mean DPP is necessarily wrong or harmful by itself.

It means the learning pressures are highly unequal across candidate identities.

C3 is being optimized primarily as the high-impact retrieval candidate.

C2 is being pushed much more heavily by diversity pressure.

That is exactly the kind of situation where one candidate can become the task specialist while another becomes the “different” candidate without necessarily becoming a high-quality retrieval specialist.

---

# 8. Is DPP directly fighting the retrieval task?

The current evidence says:

> **Not strongly enough to claim that.**

The task-vs-DPP cosine statistics for STRONG are:

| Slot | Mean cosine | Median cosine | Negative fraction | Aux/task norm ratio |
|---|---:|---:|---:|---:|
| C0 | -0.0254 | +0.0265 | 48.94% | 0.195 |
| C1 | +0.0746 | +0.0504 | 46.27% | 0.118 |
| C2 | +0.0299 | -0.0741 | 53.61% | 0.405 |
| C3 | -0.0832 | -0.1322 | 57.32% | 0.042 |

If DPP were strongly and systematically fighting the retrieval objective, one would expect clearly negative cosine values with substantial magnitude.

Instead, most means are close to zero.

C3 is the most negative:

```text
mean cosine ≈ -0.083
median      ≈ -0.132
```

but this is still weak-to-mild negative alignment, not strong opposition.

Therefore the automatic flag:

```text
NEGATIVE_TASK_AUX_ALIGNMENT
```

must be interpreted carefully.

The flag is triggered when at least half of valid pairwise comparisons are negative. It is a descriptive heuristic.

Here, near-zero vectors can easily split around positive and negative signs.

So this flag should **not** be converted into the causal statement:

> “DPP directly destroys retrieval because its gradient points opposite the task.”

That conclusion is not supported by this probe.

A safer interpretation is:

> **DPP and terminal task gradients are often weakly aligned or approximately orthogonal, with the largest meaningful DPP/task magnitude ratio occurring on C2.**

In other words, the core issue is **unequal allocation of learning pressure**, not proven direct gradient warfare.

---

# 9. DPP magnitude relative to terminal task

The STRONG proposal-level DPP/task norm ratios are:

```text
C0 ≈ 0.195
C1 ≈ 0.118
C2 ≈ 0.405
C3 ≈ 0.042
```

This is informative.

For C2:

```text
DPP norm ≈ 40.5% of terminal norm
```

which is large enough to materially shape C2.

For C3:

```text
DPP norm ≈ 4.2% of terminal norm
```

so C3 is much more task-dominated.

This produces the intuitive learning picture:

```text
C3:
mostly learns "be useful for retrieval"

C2:
receives a much stronger relative pressure to
"be diverse / different"

C0/C1:
receive intermediate mixtures
```

Again, diversity is not usefulness.

A candidate can satisfy diversity pressure by moving in a distinct direction that is semantically or geometrically different while still being harmful for retrieval.

This helps explain why STRONG can achieve greater specialization/complementarity while simultaneously reducing candidate quality.

---

# 10. Pair/gain scorer losses are not directly starving proposal gradients

The diagnostic detach controls report:

```text
pair maximum upstream proposal gradient = 0
gain maximum upstream proposal gradient = 0
passed = true
```

for both runs.

Therefore the pair/gain scorer losses are not directly backpropagating through the proposal candidates in this diagnostic graph.

This matters because the current C3 task-gradient concentration should not be blamed on pair/gain loss directly pushing the proposer.

Their effects, if any, would have to be indirect through learned policy/selection behavior across training rather than direct upstream candidate gradients in the present graph.

---

# 11. Integration with the full-VAL functional diagnosis

The gradient result should be interpreted together with the previously established full-VAL functional evidence.

The functional diagnosis already showed:

## OLD

- candidates are relatively high quality,
- all four have similar utility,
- candidate behavior is highly redundant,
- a single fixed candidate recovers almost all all-K oracle value,
- Concept-MIL responsibility is almost uniform.

So OLD can be summarized as:

> **high-quality redundancy**

## STRONG

- real functional specialization appears,
- C0/C1/C2 retain meaningful complementary niches,
- all-K is substantially better than C3-only,
- C3 owns a much larger Shapley contribution,
- candidate harmful rates rise dramatically,
- overall Mean Recall drops,
- candidate-set oracle quality also drops.

So STRONG can be summarized as:

> **real specialization, but degraded candidate quality**

The new N=512 gradient evidence explains how this state may sustain itself:

```text
specialization appears
        ↓
C3 becomes high-impact for retrieval
        ↓
hard recurrent task signal routes disproportionately through C3
        ↓
C3 proposal-query receives most terminal gradient
        ↓
C3 is further optimized for task value

meanwhile

diversity pressure is disproportionately concentrated on C2
        ↓
C2 is strongly encouraged to differ
        ↓
difference does not guarantee candidate safety/usefulness
```

This forms a plausible winner-take-most reinforcement loop.

---

# 12. Updated failure mechanism

The best current working mechanism is:

```text
OLD
4 relatively strong but redundant candidates
        |
        | stronger auxiliary/specialization pressure
        v
symmetry breaking
        |
        +-----------------------------+
        |                             |
        v                             v
C3 becomes high-impact          C0/C1/C2 become
retrieval specialist           differentiated niches
        |                             |
        v                             v
task selection / recurrence     weaker direct task exposure
routes more useful signal             |
through C3                           |
        |                             |
        v                             |
~85% terminal query-row              |
gradient to C3                       |
        |                             |
        +------------+----------------+
                     |
                     v
         winner-take-most reinforcement

simultaneously:

DPP pressure → concentrated mainly on C2
                     |
                     v
             diversity increases
                     |
                     v
      but candidate quality is not guaranteed
```

This mechanism is consistent with:

- strong C3 terminal gradient concentration,
- strong C2 DPP gradient concentration,
- meaningful non-C3 functional complementarity,
- poor STRONG candidate safety,
- lower STRONG all-K oracle,
- lower STRONG Mean Recall.

---

# 13. What is confirmed

The following statements are now reasonably strong:

### H1 — STRONG has real slot specialization

**Supported.**

Non-C3 slots retain real complementary value; the model is not simply four dead slots plus C3.

### H2 — STRONG task learning is strongly concentrated on C3

**Strongly supported.**

Large-N learned-query gradient:

```text
C3 terminal energy ≈ 85.05%
```

### H3 — The earlier small-N C3 gradient skew was not merely sampling noise

**Supported.**

The exact percentage changes from the small probe, but the qualitative winner-take-most pattern remains extreme.

### H4 — DPP pressure is concentrated on a different slot, especially C2

**Supported.**

STRONG learned-query DPP energy:

```text
C2 ≈ 50.67%
C3 ≈ 1.31%
```

### H5 — STRONG's core failure is candidate quality, not lack of diversity

**Supported by the combination of functional and gradient evidence.**

The system already has differentiation. Its weak point is that many candidates are harmful.

---

# 14. What is not confirmed

The following statements should **not** be treated as established facts.

### N1 — “C3 is the only useful candidate”

**Rejected.**

C0/C1/C2 have meaningful complementary functional value.

### N2 — “The selector simply collapsed to C3”

**Rejected as the primary explanation.**

The full-VAL live selection distribution does not support a trivial global C3 routing collapse.

### N3 — “C0/C1/C2 are dead because they never receive any gradient”

**Rejected.**

They do receive task gradient; it is just much weaker at the persistent query-row level.

### N4 — “Historical gradient starvation from initialization is proven”

**Not proven.**

This diagnostic is a final-checkpoint snapshot.

It measures current exposure, not training history.

### N5 — “DPP directly fights terminal retrieval”

**Not supported strongly enough.**

Task-vs-DPP cosines are mostly near zero.

### N6 — “We should force uniform 25/25/25/25 routing”

**Not justified.**

That risks destroying real specialization and reverting toward OLD-like redundancy.

### N7 — “Remove C3”

**Wrong intervention.**

C3 is currently the strongest high-impact specialist and contributes substantial task value.

---

# 15. Why load balancing / aWTA / forced equal usage is not the first fix

The current problem is not simply:

```text
C3 used too often
```

The deeper problem is:

```text
the useful task-learning signal is not improving all specialist candidates enough
```

Forcing equal usage can superficially make:

```text
C0 ≈ C1 ≈ C2 ≈ C3
```

without making any of them better.

It may even undo the specialization that STRONG successfully created.

The desired target is not uniformity.

The desired target is:

```text
C0 → specialist + useful
C1 → specialist + useful
C2 → specialist + useful
C3 → specialist + useful
```

with usage allowed to remain nonuniform if the data naturally favors some specialties.

---

# 16. Why increasing DPP is not the next fix

Increasing DPP would solve the wrong problem.

STRONG already has:

- greater complementarity,
- stronger candidate differentiation,
- semantic niches,
- nonuniform specialization.

The missing property is:

> each candidate should avoid making retrieval worse unless it has a genuinely useful edit.

More DPP can create even more difference without increasing quality.

The current gradient result makes this especially risky because C2 already receives a large fraction of DPP pressure.

So increasing `lambda_dpp` before restoring candidate quality could amplify:

```text
"different but bad"
```

behavior.

---

# 17. Immediate intervention: candidate safety / quality

The next controlled experiment should target the actual failure:

> **harmful candidate generation**

The already implemented candidate safety loss is appropriate:

```text
L_safe =
E[
  max(
    0,
    L_ret(candidate, target)
    -
    stopgrad(L_ret(parent, target))
  )
]
```

Interpretation:

For every candidate:

```text
if candidate retrieval is better than parent:
    no penalty

if candidate retrieval is worse than parent:
    penalize the degradation
```

This is desirable because it does **not** tell candidates:

- become identical,
- use each slot equally,
- copy C3,
- stop specializing.

It only says:

> You may specialize, but your edit should not casually make retrieval worse.

---

# 18. What Q1 must test

The next experiment should be a matched intervention:

```text
control:
lambda_safe = 0

treatment:
lambda_safe = 0.1
```

with everything else fixed.

The key question is:

> Can candidate safety improve without collapsing edit magnitude or destroying specialization?

---

# 19. Success criteria for L_safe

A successful result should satisfy several conditions together.

## 19.1 Candidate harmful fraction decreases

This is the direct target.

Especially monitor C0/C1/C2.

## 19.2 All-K oracle improves

Current STRONG all-K candidate-set quality is worse than OLD.

If `L_safe` works, the candidate-set ceiling should rise.

## 19.3 Mean Recall improves above STRONG

Reference:

```text
STRONG Mean Recall ≈ 28.2447
```

## 19.4 Non-C3 complementary niches remain

Do not accept a result that “fixes” harmful candidates merely by making all slots copies of C3.

Monitor:

- Shapley contribution,
- all-K minus C3,
- semantic conditional utility,
- conditional participation ratio.

## 19.5 Edit magnitude must not collapse

This is critical.

The easiest way to never be harmful is to do almost nothing.

Therefore inspect:

```text
delta_q norm
candidate-parent distance
edit effect magnitude
oracle gain
```

If harmful fraction falls only because:

```text
delta_q → 0
```

then the intervention failed.

---

# 20. Failure modes of L_safe

## Failure A — no-op collapse

Symptoms:

```text
harmful fraction ↓
delta_q ↓↓↓
candidate oracle does not improve
Mean Recall does not improve
```

Interpretation:

The model learned “do nothing” rather than “edit safely.”

Possible next step:

- candidate-quality bootstrap / soft candidate CE supervision.

---

## Failure B — specialization erased

Symptoms:

```text
harmful fraction ↓
but
all-K minus C3 ↓ sharply
conditional niches disappear
proposal geometry becomes OLD-like redundant
```

Interpretation:

Safety regularization is too strong.

Possible next step:

- reduce `lambda_safe`,
- preserve candidate-specific useful regions,
- then test useful-subset DPP.

---

## Failure C — safety improves but selector still chooses poorly

Symptoms:

```text
candidate oracle ↑
harmful ↓
but live Mean Recall remains weak
```

Interpretation:

Candidate generation improved, but routing/STOP/scoring remains a bottleneck.

That would justify revisiting selector calibration after candidate quality is repaired.

---

# 21. Why useful-subset DPP should come after L_safe

The current full-K DPP can spend diversity pressure on candidates that are not useful.

A better later design is:

```text
identify useful candidates using detached teacher quality
        ↓
apply DPP only among useful candidates
```

If fewer than two candidates are useful:

```text
no sibling diversity loss
```

This prevents the model from being rewarded for making bad candidates merely different from one another.

However, it should come **after** the L_safe experiment so that multiple changes are not confounded.

Recommended sequence:

```text
Q1:
L_safe only

if Q1 succeeds:
D1:
useful-subset DPP

if Q1 no-op collapses:
Q2:
candidate-quality bootstrap
```

Do not enable all of them simultaneously in the first run.

---

# 22. Current diagnosis in plain language

The current system is best explained like four students.

## OLD

```text
C0: good
C1: good
C2: good
C3: good
```

but they all learned almost the same thing.

So OLD has:

```text
quality = decent
specialization = weak
redundancy = high
```

## STRONG

The four students finally choose different majors.

That part is good.

But then:

```text
retrieval teacher attention:
C3 gets most of it

diversity teacher attention:
C2 gets a lot of it

C0/C1/C2:
still specialize, but do not receive enough task-quality teaching
```

So STRONG becomes:

```text
specialization = real
diversity = real
candidate safety = bad
quality = bad
gradient allocation = highly asymmetric
```

The objective is **not** to send everyone back to the same major.

The objective is:

```text
keep different majors
+
make every student competent in their own major
```

That is the rationale for candidate-level safety/quality learning.

---

# 23. Updated concise diagnosis

The most defensible one-sentence diagnosis is:

> **STRONG converts OLD's high-quality redundancy into genuine specialization, but the hard recurrent retrieval objective then becomes winner-take-most toward C3 while DPP pressure concentrates elsewhere, especially C2; the result is differentiated but unsafe candidate generation rather than healthy multi-specialist learning.**

A shorter engineering version is:

> **The model already has diversity. It now needs quality learning for every candidate.**

---

# 24. Decision

## Gradient diagnostic status

```text
DONE
```

No immediate need to increase this gradient probe beyond N=512 for the next engineering decision.

The large-N result is sufficient to justify moving to intervention.

## Next experiment

```text
Q1 — matched L_safe experiment
```

Primary target:

```text
reduce harmful candidates
while preserving specialization and edit magnitude
```

## Do not do yet

```text
- aWTA / forced equal routing
- load balancing
- random or round-robin slot selection
- remove C3
- increase DPP
- architecture rewrite
- another ScoreNet-only refit
- enable several new losses at once
```

---

# 25. Artifacts to preserve

Recommended Git-tracked diagnostic artifacts:

```text
doc/diagnostics/slot_specialization_grad512/
├── old/
│   ├── slot_specialization_report.json
│   └── slot_specialization_report.md
└── strong/
    ├── slot_specialization_report.json
    └── slot_specialization_report.md
```

The large per-sample JSONL files are not required for normal Git tracking unless a later analysis specifically needs them.

This diagnosis file should be stored as:

```text
doc/CIR_V2_R0_GRADIENT_DIAGNOSTIC_N512_2026-09-09.md
```

---

## Final research state

At this checkpoint:

```text
candidate collapse?                NO
selector-only collapse?            NO
real specialization?               YES
candidate quality collapse?        YES
winner-take-most task gradient?    YES
C3 terminal dominance?             YES
DPP concentrated on C2?            YES
strong direct DPP/task conflict?   NOT ESTABLISHED
historical starvation proven?      NO
next target = candidate quality?   YES
```

The next meaningful question is no longer:

> “Why are the four candidates not different?”

They are different.

The next question is:

> **“Can we make all four specialized candidates reliably useful without erasing their specialization?”**