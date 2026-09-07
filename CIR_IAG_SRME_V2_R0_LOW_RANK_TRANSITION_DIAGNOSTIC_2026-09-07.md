# CIR IAG-SRME V2 R0 — Low-Rank Transition and Recurrent Drift Diagnostic

**Date:** 2026-09-07  
**Repository:** `https://github.com/Le-Minh-Nhut/cir`  
**Branch:** `exp/e2e-iag-srme-v2-r0`  
**Experiment identity:** `R0-NCLS-TEXT`

---

# 1. Purpose

This note records a new failure mode discovered after running the latent-geometry diagnostic on both:

```text
OLD:
outputs/r0_ncls_text/best.pt
mean_recall = 30.8244

STRONG:
outputs/r0_ncls_text_strong_aux/best.pt
mean_recall = 28.2447
```

The new finding is:

> The strong auxiliary-loss run does **not** show terminal feature collapse where all samples become nearly identical. Instead, it develops a strong **low-rank transition bottleneck** inside Proposal → Action → candidate state, especially at the first recurrent transition.

This is distinct from:

```text
OLD:
sibling candidate collapse

STRONG:
candidate-quality collapse / C3 specialization
```

The current evidence suggests the strong regime suffers from several simultaneous problems:

```text
candidate-quality collapse
+
slot specialization
+
low-rank transition geometry
+
large recurrent state drift
```

---

# 2. What was tested

A dedicated script:

```text
src/diagnose_latent_geometry.py
```

was run on both checkpoints using:

```bash
TOKENIZERS_PARALLELISM=false python src/diagnose_latent_geometry.py \
  backbone=fgclip_base_text_native_cls \
  +checkpoint=outputs/r0_ncls_text/best.pt \
  +diagnostic_batches=20 \
  +diagnostic_batch_size=8 \
  hydra.run.dir=outputs/diagnose_latent_geometry_old
```

and:

```bash
TOKENIZERS_PARALLELISM=false python src/diagnose_latent_geometry.py \
  backbone=fgclip_base_text_native_cls \
  +checkpoint=outputs/r0_ncls_text_strong_aux/best.pt \
  +diagnostic_batches=20 \
  +diagnostic_batch_size=8 \
  hydra.run.dir=outputs/diagnose_latent_geometry_strong
```

The diagnostic measured:

```text
cross-sample cosine
effective rank
entropy effective rank
stable rank
explained variance
per-dimension std
covariance / correlation
relative state drift
cosine to V0
retrieval similarity to target
real-vs-synthetic distribution shift
```

---

# 3. Important baseline: frozen FG-CLIP geometry is identical

Both runs start from the same frozen visual backbone.

Reference global representation:

```text
effective rank ≈ 29.67
mean cross-sample cosine ≈ 0.287
top-1 explained variance ≈ 14.3%
top-10 explained variance ≈ 41.0%
```

Initial pooled patch state:

```text
effective rank ≈ 11.93
mean cosine ≈ 0.887
top-1 explained variance ≈ 25.6%
```

Therefore the new low-rank behavior is **not caused by the frozen FG-CLIP input representation itself**.

It emerges after the learned recurrent transition modules.

---

# 4. Main discovery: strong Proposal geometry collapses dimensionally

At timestep `t0`:

## OLD Proposal

```text
effective rank ≈ 19.75
stable rank    ≈ 8.71
top-1 variance ≈ 11.5%
top-5 variance ≈ 43.2%
top-10 var     ≈ 63.8%
```

## STRONG Proposal

```text
effective rank ≈ 3.53
stable rank    ≈ 1.92
top-1 variance ≈ 52.1%
top-5 variance ≈ 70.5%
top-10 var     ≈ 78.5%
```

This is a dramatic collapse of the learned proposal space.

The proposals remain different from one another in cosine space, but most across-sample variation is concentrated in only a few dominant directions.

Therefore:

```text
candidate diversity != healthy representation rank
```

The strong run can satisfy:

```text
C0 != C1 != C2 != C3
```

while still having:

```text
all proposal variation
≈ small low-dimensional subspace
```

---

# 5. Action space inherits the low-rank bottleneck

At `t0`:

## OLD Action

```text
effective rank ≈ 19.53
top-1 variance ≈ 11.3%
```

## STRONG Action

```text
effective rank ≈ 3.80
top-1 variance ≈ 49.3%
```

Thus the Proposal low-rank structure propagates almost directly into ActionFusion.

The strong action space is therefore approximately:

```text
768-D nominal representation
        ↓
effective variation ≈ 3–4 dominant directions
```

This is a new confirmed failure mode.

---

# 6. Candidate-state transition becomes extremely low rank

At `t0`:

## OLD candidate state

```text
effective rank ≈ 13.26
top-1 variance ≈ 17.0%
```

## STRONG candidate state

```text
effective rank ≈ 3.27
top-1 variance ≈ 53.5%
```

Thus Executor does not restore the lost dimensionality.

Instead:

```text
low-rank proposals
    ↓
low-rank actions
    ↓
low-rank state transitions
```

This means the issue is not confined to ProposalNet.

It affects the actual mutable visual state.

---

# 7. Strongest evidence: committed state at first transition

The state that is actually selected and committed after timestep `t0` shows the clearest bottleneck.

## OLD committed state after t0

```text
effective rank ≈ 13.20
top-1 variance ≈ 17.2%
```

## STRONG committed state after t0

```text
effective rank ≈ 2.70
top-1 variance ≈ 59.8%
```

This means almost 60% of across-sample variation in the selected next state lies along only one principal direction.

This is the most important newly discovered signal.

The first recurrent transition in the strong model is therefore not merely diverse between siblings.

It is moving the batch into a strongly low-dimensional state regime.

---

# 8. This is NOT ordinary "all features become identical" collapse

The terminal retrieval embedding remains reasonably distributed.

## OLD terminal query

```text
mean cosine ≈ 0.103
effective rank ≈ 21.19
stable rank ≈ 9.86
top-1 variance ≈ 10.1%
```

## STRONG terminal query

```text
mean cosine ≈ 0.123
effective rank ≈ 24.31
stable rank ≈ 9.01
top-1 variance ≈ 11.1%
```

So the strong model does **not** end with:

```text
sample A ≈ sample B ≈ sample C
```

There is no evidence of a catastrophic terminal constant-vector collapse.

The correct diagnosis is more specific:

> **low-rank action-conditioned transition collapse**

rather than:

> **global terminal representation collapse**

---

# 9. Why previous candidate-diversity diagnostics did not reveal this

Previous diagnostics measured within-sample sibling cosine:

```text
C0 vs C1
C0 vs C2
C0 vs C3
...
```

The strong model showed:

```text
low pairwise cosine
```

which correctly proved that siblings differ.

But effective-rank diagnostics ask a different question:

```text
Across many samples,
how many independent directions does the representation actually use?
```

A model can satisfy:

```text
C0, C1, C2, C3 are mutually distinct
```

while all four live inside a small shared subspace.

Example:

```text
full latent space: 768 dimensions

              ┌───────────────────┐
              │                   │
              │  ↙  ↑  →  ↘       │
              │   3D subspace     │
              │                   │
              └───────────────────┘
```

DPP sees:

```text
four different directions
```

and is satisfied.

But the representation still uses only a tiny portion of the available feature space.

---

# 10. Strong DPP likely contributes to this equilibrium

The strong run uses:

```yaml
lambda_dpp: 0.6
```

Previous gradient diagnostics showed DPP pressure increased by approximately:

```text
Proposal: ~35×
Executor: ~10×
```

relative to the old run.

This gives the optimizer a strong incentive to find a compact set of directions that efficiently separate candidates.

One possible equilibrium is:

```text
learn a few dominant orthogonal-ish directions
        ↓
assign candidate slots to those directions
        ↓
maximize sibling diversity
        ↓
without preserving broad semantic variation
```

This would explain:

```text
low sibling cosine
+
low effective rank
```

at the same time.

This interpretation is strongly consistent with the observed geometry, but the exact causal role of DPP still requires controlled ablation.

---

# 11. Relationship to C3 specialization

Previous selector diagnostics showed:

```text
C3 selected ≈ 72.7%
oracle C3   ≈ 63.2%
```

Per-slot teacher utility:

```text
C0 = -0.034
C1 = -0.029
C2 = -0.008
C3 = +0.031
```

The newly discovered low-rank geometry suggests a possible mechanism:

```text
strong DPP
    ↓
few dominant transition directions
    ↓
fixed learned candidate slots specialize
    ↓
one direction becomes retrieval-useful
    ↓
C3 becomes useful specialist
    ↓
other slots become diversity specialists
```

Thus C3 specialization and low-rank transition collapse may be two manifestations of the same objective imbalance.

This is not yet fully proven.

The next diagnostic should measure effective rank **per slot across samples**.

---

# 12. Retrieval behavior confirms the first transition is problematic

Initial target similarity is identical in both runs:

```text
before t0 ≈ 0.579
```

After the first committed transition:

## OLD

```text
0.579 → 0.523
drop ≈ -0.056
```

## STRONG

```text
0.579 → 0.461
drop ≈ -0.119
```

The strong model therefore damages positive target similarity about twice as much during the first action.

Then it partially recovers:

```text
STRONG:

initial     ≈ 0.579
after t0    ≈ 0.461
after t1    ≈ 0.502
after t2    ≈ 0.563
terminal    ≈ 0.565
```

OLD:

```text
initial     ≈ 0.579
after t0    ≈ 0.523
after t1    ≈ 0.572
after t2    ≈ 0.564
terminal    ≈ 0.582
```

This suggests:

> The strong model spends later recurrent steps repairing damage introduced by its first transition.

That is highly undesirable for a sequential planning-style CIR architecture.

---

# 13. Recurrent drift is also stronger

Relative pooled-state drift from `V0`:

## OLD

```text
t1 ≈ 0.803
t2 ≈ 0.992
```

## STRONG

```text
t1 ≈ 0.775
t2 ≈ 1.089
```

At `t2`, the strong state's L2 displacement is larger than the norm of the original state.

The automatic diagnostic therefore raised:

```text
WARN LARGE_RECURRENT_STATE_DRIFT
```

for the strong run.

The old run did not cross the configured threshold.

This is not yet proof of off-manifold failure by itself.

A large transition can be valid if retrieval improves.

But in combination with:

```text
low-rank transition
+
first-step retrieval degradation
+
candidate-quality collapse
```

the drift becomes much more concerning.

---

# 14. Interesting recovery in global readout

The low-rank patch state does not remain equally severe after global readout.

At `t2/current_global`:

```text
OLD effective rank    ≈ 22.37
STRONG effective rank ≈ 22.24
```

The strong run's committed global state at `t2` reaches approximately:

```text
effective rank ≈ 25.81
```

This indicates the system can recover substantial global representation diversity later in rollout.

Therefore the failure is not:

```text
collapse → irreversible collapse → terminal death
```

It is closer to:

```text
first transition enters low-rank bottleneck
        ↓
later steps partially spread / repair representation
        ↓
terminal geometry becomes healthy enough
        ↓
but retrieval performance never fully recovers
```

This distinction matters for future fixes.

---

# 15. Real-vs-synthetic distribution shift

Both runs move synthetic global representations away from the initial reference distribution.

This is expected because the model is supposed to edit the visual state.

However, the strong run exhibits different transition statistics and larger late recurrent drift.

Therefore the important question is not:

```text
Does synthetic state differ from reference?
```

It should.

The relevant question is:

```text
Does the synthetic state preserve healthy semantic geometry
while moving toward the target?
```

The first-transition retrieval and rank diagnostics suggest that the strong run often does not.

---

# 16. Updated failure hierarchy

Current branch evidence:

```text
[CONFIRMED]
1. old sibling candidate collapse

[CONFIRMED]
2. strong run fixes sibling collapse

[CONFIRMED]
3. strong candidate-quality collapse

[CONFIRMED]
4. C3 useful-slot specialization

[CONFIRMED]
5. Proposal low-rank collapse in strong run

[CONFIRMED]
6. Action low-rank collapse in strong run

[CONFIRMED]
7. first-step candidate-state low-rank bottleneck

[STRONG EVIDENCE]
8. excessive recurrent drift in strong run

[NOT OBSERVED]
9. catastrophic terminal global representation collapse
```

---

# 17. Current conceptual picture

## OLD

```text
Proposal
   ↓
four nearly identical candidates
   ↓
representation rank reasonably broad across samples
   ↓
candidate utility generally positive
   ↓
retrieval ≈ 30.82
```

Problem:

```text
fake multi-candidate diversity
```

---

## STRONG

```text
Proposal
   ↓
four distinct candidate directions
   ↓
but Proposal space has effective rank ≈ 3.5
   ↓
Action space rank ≈ 3.8
   ↓
candidate state rank ≈ 3.3
   ↓
committed first-step state rank ≈ 2.7
   ↓
C3 becomes main useful specialist
   ↓
first transition damages target similarity
   ↓
later steps attempt recovery
   ↓
retrieval ≈ 28.24
```

Problem:

```text
diverse
but low-rank
and low-quality
```

---

# 18. Main new hypothesis

The current strong DPP objective may be encouraging:

```text
candidate separation
```

without enough pressure for:

```text
semantic dimensional richness
```

or:

```text
per-candidate usefulness
```

The optimizer can therefore satisfy diversity cheaply by learning a handful of strong transition axes.

This creates:

```text
low cosine between candidates
```

but also:

```text
low effective rank across samples
```

The desired objective should instead favor:

```text
USEFUL
+
DIVERSE
+
SEMANTICALLY HIGH-RANK
candidate transitions
```

not merely pairwise separation.

---

# 19. Immediate next diagnostic

Before changing the architecture, measure each fixed candidate slot separately across samples.

For every timestep:

```text
C0 Proposal rank
C1 Proposal rank
C2 Proposal rank
C3 Proposal rank

C0 Action rank
C1 Action rank
C2 Action rank
C3 Action rank

C0 delta_q rank
C1 delta_q rank
C2 delta_q rank
C3 delta_q rank
```

Also measure:

```text
mean vector cosine between slot prototypes
per-slot teacher utility
per-slot explained variance
per-slot state drift
```

This will distinguish two hypotheses.

---

## Hypothesis A — each slot individually collapses

Example:

```text
C0 rank ≈ 2
C1 rank ≈ 2
C2 rank ≈ 2
C3 rank ≈ 2
```

Interpretation:

```text
each learned query becomes nearly a fixed transformation template
```

---

## Hypothesis B — individual slots are rich, but the union is low-rank

Example:

```text
C0 rank high
C1 rank high
C2 rank high
C3 rank high

but combined rank low
```

Interpretation:

```text
all slots share the same small semantic subspace
```

The fix would differ substantially between these two cases.

---

# 20. What should NOT be changed yet

Do not immediately add:

```text
VICReg
Barlow Twins
whitening
generic isotropy loss
variance regularization
random candidate shuffle
```

The new diagnostic proves a low-rank bottleneck, but not yet the exact mechanism.

Adding generic representation regularization prematurely could damage FG-CLIP semantic geometry.

First localize whether the collapse occurs:

```text
inside each candidate slot
```

or:

```text
across the shared candidate subspace
```

---

# 21. Recommended next priority

```text
P0:
per-slot effective-rank diagnostic

P1:
fix DPP active-set / useful-candidate semantics

P1:
direct upstream candidate-quality supervision

P2:
transition magnitude / manifold preservation

P2:
STOP calibration
```

---

# 22. Final diagnosis

The latent-geometry experiment produced a genuinely new finding:

> The strong auxiliary run does not suffer catastrophic terminal feature collapse, but its first action-generation and transition pathway becomes severely low-rank.

The strongest evidence is:

```text
Proposal:
19.75 → 3.53 effective rank

Action:
19.53 → 3.80

Candidate state:
13.26 → 3.27

Committed first-step state:
13.20 → 2.70
```

while:

```text
terminal query rank:
21.19 → 24.31
```

This means the problem is localized primarily to the **transition-generation pathway**, not the final retrieval embedding.

Current concise interpretation:

```text
OLD:
useful but redundant

STRONG:
diverse but low-rank and mostly low-quality
```

The next revision should ultimately target:

```text
useful
+
non-collapsed
+
high-rank
+
controlled
latent transitions
```

but implementation changes should wait until the per-slot rank diagnostic is complete.

---

# 23. Related diagnostic artifacts

General branch diagnostics:

```text
doc/diagnostics/v2-r0/
```

Latent-geometry source reports:

```text
outputs/diagnose_latent_geometry_old/latent_geometry_report.json
outputs/diagnose_latent_geometry_strong/latent_geometry_report.json
```

Diagnostic script:

```text
src/diagnose_latent_geometry.py
```

This note:

```text
CIR_IAG_SRME_V2_R0_LOW_RANK_TRANSITION_DIAGNOSTIC_2026-09-07.md
```