# CIR IAG-SRME V2 R0 — Branch Diagnostic Report

**Branch:** `exp/e2e-iag-srme-v2-r0`  
**Model identity:** `R0-NCLS-TEXT`  
**Backbone:** `qihoo360/fg-clip-base`  
**Readout:** native CLS  
**Fine-tuning policy:** text-only  
**Dataset:** FashionIQ  
**Candidates:** `K = 4`  
**Max rollout steps:** `3`

---

## 1. Purpose

This document records the diagnosis of the current V2 R0 branch after two training regimes:

1. the original weak auxiliary-loss configuration, and  
2. a stronger semantic/diversity-loss configuration intended to prevent candidate collapse.

The central finding is:

> The stronger objective successfully fixes representation / functional candidate collapse, but creates a second failure mode: **candidate-quality collapse / slot specialization**. The model now produces diverse candidate actions, but only one slot is consistently useful.

This means the branch did not fail because anti-collapse was ineffective. It failed because anti-collapse became stronger than candidate-quality pressure.

---

# 2. Architecture being diagnosed

The recurrent path is:

```text
Reference image
    ↓
FG-CLIP visual state V_t
    ↓
current global readout g_t
    ↓
ProposalNet
    ↓
K = 4 edit proposals
    ↓
Grounder
    ↓
entity read + execution mask
    ↓
ActionFusion
    ↓
K actions
    ↓
Executor
    ↓
K sibling candidate states
    ↓
candidate retrieval queries
    ↓
delta_q consequences
    ↓
ScoreNet
    ↓
hard candidate selection / STOP
    ↓
commit selected state
    ↓
next timestep
```

The target image is not an inference input. During training, target-derived retrieval utilities are used as privileged supervision for ScoreNet and DPP quality weighting.

---

# 3. Objective

The R0 objective is:

```text
L_total =
1.0 * L_terminal
+ 0.5 * L_pair
+ 0.5 * L_gain
+ λ_c    * L_concept
+ λ_bind * L_bind
+ λ_rel  * L_rel_ortho
+ λ_dpp  * L_DPP
```

Two configurations were tested.

## 3.1 Original weak auxiliary configuration

```yaml
lambda_c: 0.01
lambda_bind: 0.01
lambda_rel: 0.001
lambda_dpp: 0.01
```

Best checkpoint:

```text
epoch = 15
FashionIQ mean_recall = 30.8244
```

## 3.2 Strong auxiliary configuration

```yaml
lambda_c: 0.6
lambda_bind: 1.0
lambda_rel: 0.6
lambda_dpp: 0.6
```

Best checkpoint:

```text
epoch = 18
FashionIQ mean_recall = 28.2447
```

The stronger objective therefore reduced official validation mean recall by about:

```text
30.8244 - 28.2447 = 2.5797 points
```

However, this recall drop alone does not describe what happened internally.

---

# 4. Main comparison

| Diagnostic | Weak objective | Strong objective | Interpretation |
|---|---:|---:|---|
| FashionIQ mean recall | **30.824** | **28.245** | retrieval decreased |
| Proposal sibling cosine | **0.9979** | **~0.37** | proposal collapse fixed |
| Action sibling cosine | **0.9980** | **~0.34** | action collapse fixed |
| Functional `delta_q` cosine | **0.9988** | **~0.013** | retrieval-effect collapse fixed |
| Functional effective rank | **1.074** | **1.916** | candidate effect rank improved |
| Write-mask Jaccard | **0.988** | **~0.219** | candidates edit different regions |
| Mean useful candidates | **~2.43** | **~0.99** | candidate quality collapsed |
| Positive-utility candidates | **~79.2%** | **~45.3%** | many diverse candidates became harmful |
| DPP-valid row rate | **~60.8%** | **~21.1%** | DPP becomes starved after specialization |
| Harmful execution rate | **~15.1%** | **~25.2%** | selector now pays for bad candidates |
| Exact oracle action accuracy | **~29.6%** | **~68.5%** | selector improves when candidates become distinguishable |
| STOP selected | **~10.9%** | **~0.85%** | strong run under-uses STOP |

---

# 5. Diagnosis of the original weak-loss run

## 5.1 Functional candidate collapse is severe

The old checkpoint produces four candidates that are almost copies of one another.

### Proposal similarity

```text
proposal cosine ≈ 0.997–0.998
```

### Action similarity

```text
action cosine ≈ 0.998
```

### Retrieval consequence similarity

```text
delta_q cosine ≈ 0.9983–0.9990
functional effective rank ≈ 1.07
```

### Write-region similarity

```text
execution-mask Jaccard ≈ 0.984–0.991
```

This is not only semantic similarity. It persists all the way through the Executor into retrieval-visible consequences.

The old branch therefore behaves approximately like:

```text
Candidate 0 ─┐
Candidate 1 ─┤
Candidate 2 ─┤  → nearly the same action / consequence
Candidate 3 ─┘
```

The architecture nominally exposes four actions, but functionally behaves close to a one-action model.

---

## 5.2 The collapse does not originate from the learned candidate priors

The four learned ProposalNet query priors are already diverse.

Their pairwise cosine values are approximately:

```text
-0.06 ... +0.06
```

Therefore:

```text
learned q_0, q_1, q_2, q_3
        ↓
        diverse
        ↓
ProposalNet semantic conditioning / token attention
        ↓
        collapsed proposals
```

This is an important distinction.

The model is not failing because all four learned query rows initialize or train into the same vector. Their diversity is erased later by the semantic proposal computation.

---

## 5.3 Selector accuracy is misleading in the collapsed regime

The old selector exact-oracle action accuracy is only:

```text
~29.6%
```

but oracle regret is small:

```text
selected teacher utility ≈ 0.2608
oracle teacher utility   ≈ 0.2743
oracle regret            ≈ 0.0135
```

This is expected because all four candidates have almost identical utility:

```text
C0 ≈ 0.2594
C1 ≈ 0.2593
C2 ≈ 0.2593
C3 ≈ 0.2595
```

Choosing a different slot from the oracle often has almost no functional cost.

Therefore:

> Low exact slot accuracy in the weak run is not evidence that ScoreNet is the primary failure. Candidate collapse makes slot identity nearly meaningless.

---

# 6. Diagnosis of the strong-loss run

## 6.1 Anti-collapse succeeds

The strong configuration dramatically changes candidate geometry.

Representative proposal cosine matrix:

```text
C0-C1 ≈  0.74
C0-C2 ≈  0.79
C0-C3 ≈  0.08
C1-C2 ≈  0.43
C1-C3 ≈  0.42
C2-C3 ≈ -0.21
```

Representative action cosine:

```text
C0-C1 ≈  0.73
C0-C2 ≈  0.73
C0-C3 ≈  0.06
C1-C2 ≈  0.36
C1-C3 ≈  0.26
C2-C3 ≈ -0.07
```

Retrieval-visible `delta_q` consequences are even more separated:

```text
C0-C1 ≈  0.10
C0-C2 ≈  0.17
C0-C3 ≈ -0.11
C1-C2 ≈ -0.02
C1-C3 ≈  0.06
C2-C3 ≈ -0.10
```

Thus the strong objective genuinely fixes the original functional-collapse problem.

---

# 7. New failure mode: candidate-quality collapse

Although the four candidates are now diverse, their usefulness becomes highly asymmetric.

## 7.1 Per-slot teacher utility

```text
C0 mean utility = -0.0342
C1 mean utility = -0.0295
C2 mean utility = -0.0077
C3 mean utility = +0.0312
```

C3 is the only slot with positive mean retrieval utility.

## 7.2 Positive-utility rate

```text
C0: 32.8%
C1: 45.9%
C2: 35.3%
C3: 67.2%
```

## 7.3 DPP-useful rate

```text
C0: 13.1%
C1: 18.8%
C2:  9.7%
C3: 57.1%
```

The model has therefore converged toward:

```text
C0 = diverse but usually weak
C1 = diverse but usually weak
C2 = diverse but usually weak
C3 = main useful candidate
```

This is a different form of collapse.

It is better described as:

> **candidate-quality collapse / useful-slot specialization**

rather than representation collapse.

---

# 8. C3 specialization is real, not merely a ScoreNet bias

The strong-run selection histogram is:

```text
C0   = 20
C1   = 50
C2   = 55
C3   = 344
STOP = 4
```

or approximately:

```text
C3 selected = 72.7%
```

At first this looks like selector slot collapse.

However, target-privileged oracle selection is:

```text
C0   = 35
C1   = 52
C2   = 42
C3   = 299
STOP = 45
```

so:

```text
oracle prefers C3 = 63.2%
```

Therefore C3 is genuinely the best slot most of the time.

The selector is somewhat more biased toward C3 than the oracle, but it is mostly responding to a real upstream asymmetry.

The correct diagnosis is:

```text
not:
ScoreNet randomly likes slot C3

but:
Proposal / Grounding / Executor pipeline makes C3 much better
→ oracle likes C3
→ ScoreNet learns to like C3
```

---

# 9. ScoreNet is not the primary bottleneck

Strong-run selector diagnostics:

```text
exact oracle action accuracy ≈ 68.5%
stop/execute accuracy        ≈ 90.1%
score-teacher Pearson        ≈ 0.612
```

These are materially better than one would expect from a broken selector.

The remaining issue is still important:

```text
harmful execution rate ≈ 25.2%
oracle regret          ≈ 0.0684
```

But this is downstream of the candidate-quality problem.

When only one slot is consistently good, a selector mistake has a real cost.

In the weak run, candidate collapse hid selector mistakes because every slot did almost the same thing.

---

# 10. STOP is under-used in the strong run

Strong-run rates:

```text
model STOP  ≈ 0.85%
oracle STOP ≈ 9.51%
```

The model therefore executes too often.

This contributes to:

```text
harmful execution rate ≈ 25.2%
```

STOP needs further calibration or supervision, but it should not be treated as the root cause of the branch failure.

Priority remains:

```text
candidate quality
    >
STOP calibration
    >
selector fine-tuning
```

---

# 11. DPP behavior

## 11.1 Old run

```text
mean useful candidates ≈ 2.43
DPP-valid rows         ≈ 60.8%
positive candidates    ≈ 79.2%
lambda_dpp             = 0.01
```

DPP had enough useful sibling candidates to operate, but its weight was too small to overcome collapse.

## 11.2 Strong run

```text
mean useful candidates ≈ 0.99
DPP-valid rows         ≈ 21.1%
positive candidates    ≈ 45.3%
lambda_dpp             = 0.6
```

The model becomes highly diverse, but useful-candidate count falls below two on most rows.

This means the model moved from:

```text
many useful but redundant candidates
```

to:

```text
diverse candidates, but usually only one useful candidate
```

---

# 12. DPP dominates the strong objective

Representative mean weighted contributions in the strong checkpoint diagnostic:

```text
terminal       ≈ +0.1524
pair           ≈ +0.1913
gain           ≈ +0.0642
concept        ≈ +0.00184
bind           ≈ +0.000041
rel_ortho      ≈ ~0
DPP            ≈ -0.7666
```

DPP contributes roughly:

```text
~65.5% of total absolute objective magnitude
```

This explains why training loss becomes negative while validation retrieval remains modest.

Negative total loss is not automatically a numerical bug because the current DPP objective is based on:

```text
-log det(...)
```

which can be negative.

However:

> A decreasing or negative total loss is no longer a useful proxy for retrieval quality when DPP dominates the objective scale.

The strong run's near-zero / negative training loss must therefore not be interpreted as superior convergence.

---

# 13. DPP gradient pressure increases dramatically

Per-loss gradient probes show that the stronger DPP configuration substantially increases anti-collapse pressure.

Approximate DPP gradient norm into representative parameters:

## Proposal

```text
weak DPP:   ~0.00081
strong DPP: ~0.02822
```

Approximately:

```text
~35× larger
```

## Executor

```text
weak DPP:   ~0.00551
strong DPP: ~0.05464
```

Approximately:

```text
~10× larger
```

This matches the observed geometry:

```text
weak:
candidate consequences collapse

strong:
candidate consequences separate aggressively
```

There is no evidence of dead gradients or optimizer failure. Main modules receive finite gradients.

Therefore the current failure is an **objective-design / optimization-target problem**, not a missing-gradient bug.

---

# 14. Important DPP implementation concern

The current functional DPP logic computes quality:

```python
quality = sigmoid(teacher_utility / tau)
active = quality > useful_threshold
```

and checks:

```python
if useful_count < 2:
    return zero_loss
```

However, once the row passes this `>= 2 useful candidates` gate, the DPP kernel is built using all K candidates:

```python
kernel = quality[:, None] * conditional * quality[None, :]
```

The `active` mask is therefore currently used as a **row-level activation guard**, not as a strict subset mask on the determinant.

Conceptually, the behavior is:

```text
Are at least two candidates useful?
        ↓
       yes
        ↓
apply quality-weighted DPP over all K candidates
```

rather than:

```text
identify useful candidate subset
        ↓
compute diversity only among useful candidates
```

This is not necessarily mathematically invalid, because low-quality candidates receive smaller quality factors.

But in the current branch it is a major design concern:

> With a large `lambda_dpp`, a low-quality candidate can still contribute to increasing functional diversity instead of being directly trained toward usefulness.

This behavior is consistent with the observed strong-run specialization:

```text
C3 = useful candidate
C0/C1/C2 = diversity carriers
```

This should be reviewed before further lambda tuning.

---

# 15. Semantic auxiliary diagnosis

## 15.1 Concept loss

Concept positive recall changes approximately:

```text
weak:   ~7.5%
strong: ~16.3%
```

So the larger concept weight improves the auxiliary signal.

However, strong-run concept recall remains low.

Furthermore, despite:

```text
lambda_c = 0.6
```

its weighted contribution remains tiny because the raw loss scale is very small:

```text
weighted concept contribution ≈ 0.0018
```

Therefore coefficient magnitude alone is misleading.

The concept objective should be evaluated using:

```text
raw loss
weighted loss
gradient norm
positive recall
instruction coverage
```

rather than `lambda_c` alone.

---

## 15.2 Binding loss

Strong-run bind loss is already extremely small:

```text
raw bind ≈ 4e-5
```

Its numerical contribution to the total objective is negligible even with:

```text
lambda_bind = 1.0
```

This suggests the binding objective is close to solved under its current formulation or has a naturally very small scale.

---

## 15.3 Relation orthogonality

Relation orthogonality is effectively solved.

Strong-run diagnostics:

```text
prototype occupancy        = 1.0
prototype entropy          ≈ ln(8)
prototype pairwise cosine  ≈ 0
rel_ortho raw loss         ≈ 2e-9
```

Therefore:

> Increasing `lambda_rel` further is unlikely to provide meaningful anti-collapse pressure.

The relation prototype bank is not the active failure mode.

---

# 16. Root-cause hierarchy

Current evidence supports the following ordering.

## Root cause #1 — weak run

```text
Proposal semantic collapse
    ↓
Grounding / entity collapse
    ↓
Action collapse
    ↓
Executor consequence collapse
    ↓
K candidates become functionally redundant
```

## Root cause #2 — strong run

```text
Strong DPP / diversity pressure
    ↓
functional collapse disappears
    ↓
candidates spread into distinct directions
    ↓
quality pressure does not keep all directions useful
    ↓
C3 becomes the dominant useful slot
    ↓
C0/C1/C2 become mostly diversity carriers
    ↓
mean useful candidate count falls below 1
    ↓
retrieval drops
```

## Secondary issue

```text
STOP under-used
    ↓
too many executions
    ↓
~25% harmful selected actions
```

## Not currently the primary cause

```text
ScoreNet
relation prototypes
gradient flow
optimizer
learned proposal-query prior collapse
```

---

# 17. State of the branch

The branch should currently be considered:

```text
FUNCTIONALLY WORKING
but
OBJECTIVE-IMBALANCED
```

Specifically:

### Solved / mostly solved

```text
[OK] candidate representation diversity under strong DPP
[OK] action diversity
[OK] write-region diversity
[OK] retrieval-consequence diversity
[OK] relation prototype diversity
[OK] gradient routing
[OK] ScoreNet can distinguish candidate utility
```

### Unsolved

```text
[FAIL] maintain multiple useful candidates simultaneously
[FAIL] prevent candidate-quality / slot specialization collapse
[WARN] DPP dominates loss scale
[WARN] DPP active-set semantics may be too permissive
[WARN] STOP is under-used
[WARN] concept recall remains weak
[WARN] selected harmful-action rate remains high
```

---

# 18. What should NOT be done next

Do not immediately:

```text
1. increase lambda_dpp beyond 0.6
2. force uniform selector usage across C0/C1/C2/C3
3. increase lambda_rel further
4. interpret negative total loss as better convergence
5. blame ScoreNet for C3 dominance
6. return directly to lambda_dpp = 0.01
```

Why:

- larger DPP likely worsens quality collapse,
- uniform selector usage would force selection of objectively bad slots,
- relation orthogonality is already solved,
- total loss scale is dominated by negative DPP,
- oracle itself strongly prefers C3,
- weak DPP reproduces functional collapse.

---

# 19. Recommended next investigation

Before another full 20-epoch training run, investigate the DPP / candidate-quality interaction.

Priority order:

## P0 — inspect and revise DPP semantics

Question:

> Should the determinant include all candidates with soft quality weighting, or only the target-judged useful subset?

Candidate alternatives to test:

```text
A. active-subset DPP
   compute DPP only over candidates whose teacher quality > threshold

B. stronger quality weighting
   suppress low-utility candidate contribution much more aggressively

C. positive-utility-only DPP
   diversity pressure exists only among candidates with teacher utility > 0

D. quality floor / candidate usefulness objective
   explicitly train every slot toward non-harmful utility before rewarding diversity
```

This should be resolved before further large lambda sweeps.

---

## P1 — add direct candidate-quality pressure to upstream modules

Current system strongly teaches:

```text
ScoreNet:
which candidate is good?
```

and DPP strongly teaches:

```text
Proposer / Executor:
make candidates different.
```

What is missing is a direct upstream signal equivalent to:

```text
Proposer / Executor:
make EACH candidate useful,
then make useful candidates different.
```

This is the conceptual gap exposed by the strong run.

---

## P2 — retune DPP after fixing semantics

A reasonable next search region is:

```yaml
lambda_dpp: 0.10
lambda_dpp: 0.15
lambda_dpp: 0.20
lambda_dpp: 0.30
```

not immediately `0.6`.

The exact value should be selected using both:

```text
FashionIQ mean recall
AND
mean useful candidate count
AND
functional cosine / effective rank
```

The desired solution is not maximum diversity.

The desired operating point is:

```text
multiple useful candidates
+
non-collapsed consequences
+
good retrieval
```

---

## P3 — STOP calibration

Once candidate quality is improved, diagnose why:

```text
model STOP  << oracle STOP
```

Potential directions include:

```text
explicit STOP supervision
margin against KEEP
calibration of epsilon_stop
STOP-specific loss
```

Do not prioritize this ahead of candidate quality.

---

# 20. Success criteria for the next R0 revision

A future run should not be judged by mean recall alone.

Suggested minimum diagnostics:

```text
FashionIQ mean recall        > 30.8 baseline
functional cosine            < 0.8
functional effective rank    > 1.5
mean useful candidates       >= 2.0
positive candidate fraction  > 65%
harmful execution rate       < 15%
oracle regret                low
STOP usage closer to oracle
```

An especially desirable target is:

```text
K = 4 candidates
↓
2–3 genuinely useful alternatives
↓
different retrieval consequences
↓
selector meaningfully chooses among them
```

rather than either extreme:

```text
4 copies of one action
```

or:

```text
1 useful action + 3 arbitrary diverse actions
```

---

# 21. Final diagnosis

The two experiments expose two opposite pathological equilibria.

## Weak auxiliary regime

```text
Diversity pressure too weak
        ↓
all candidates converge to the same solution
        ↓
good retrieval but fake multi-candidate behavior
```

## Strong auxiliary regime

```text
Diversity pressure too strong relative to quality pressure
        ↓
candidates become genuinely different
        ↓
only one slot remains reliably useful
        ↓
C3 specialization
        ↓
retrieval degrades
```

Therefore the next objective should target the middle regime:

> **Useful diversity, not diversity by itself.**

The central design requirement for the next revision is:

```text
Candidate diversity
must be conditional on
candidate usefulness.
```

That is the main unresolved issue of `exp/e2e-iag-srme-v2-r0`.

---

# 22. Diagnostic artifacts used

The diagnosis above was derived from:

```text
outputs/diagnose_r0_ncls_text_old_loss/diagnostic_report.json
outputs/diagnose_r0_ncls_text_new_loss/diagnostic_report.json

outputs/diagnose_selector_old/candidate_selector_diagnostic.json
outputs/diagnose_selector_new/candidate_selector_diagnostic.json
```

Diagnostic scripts:

```text
src/diagnose_iag_srme.py
src/diagnose_candidate_selector.py
```

These target-derived diagnostics are for training/debug analysis only and must not be interpreted as inference inputs or official FashionIQ retrieval metrics.