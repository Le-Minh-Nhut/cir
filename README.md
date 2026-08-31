# CIR IAG-SRME R1c2 — Dynamic Current-State WHAT Re-Proposal Diagnostic README

**Date:** 2026-08-31  
**Branch:** `exp/e2e-iag-srme-r1c2-dynamic-reproposal`  
**Final audited implementation SHA:** `fb2e226f1f694959f158c8887567d0647bf263a8`  
**Source branch:** `exp/e2e-iag-srme-r1c1-dynamic-reground`  
**Source R1c1 SHA:** `46b48e68f08ea5ab70c95c43af1eff0a3fc08da2`  
**Architecture generation:** `r1c2_dynamic_current_state_reproposal_v1`  
**Dataset / protocol:** FashionIQ original  
**Backbone:** `qihoo360/fg-clip-base`  
**Backbone revision:** `454d76372c2cf5eb48fa0d871fd0534481484d97`  
**Training precision:** FP16  
**K:** 4 candidates  
**Tmax:** 3 recurrent decisions  
**query_cap:** 1000.0  
**Grounding normalization:** Entmax-1.5  
**Dynamic current-state WHERE:** ON  
**Dynamic current-state WHAT:** ON  
**Dynamic applicability:** OFF  
**Visual NULL:** OFF  

---

# 1. Executive conclusion

R1c2 was designed as the next clean causal test after R1c1.

R1c1 had already shown that merely replacing static visual grounding with:

\[
\pi_{t,k}=Ground(I_k,Z_t)
\]

does not solve candidate collapse. The fixed candidate intents remained highly correlated and their dynamic supports moved largely in common mode.

R1c2 therefore tested the hypothesis:

> The remaining bottleneck is the fixed WHAT/action identity itself.  
> After the recurrent visual state changes, the model should reconsider the original modification text in the context of the current state and re-propose what semantic action remains relevant.

The intervention was intentionally narrow:

```text
R1c1
+
dynamic current-state WHAT re-proposal
```

while preserving the R1c1 dynamic WHERE mechanism.

The R1c2 implementation is mechanically valid.

The CUDA canary passed:

```text
attempted steps:                           100
successful optimizer steps:                96
AMP-overflow skipped steps:                 4
grounder nonzero-gradient fraction:       1.0
reproposal output gradient steps:         96
reproposal state gradient steps:          95
reproposal change gradient steps:         95
reproposal text gradient steps:           95
reproposal state-query gradient steps:    95
reproposal fusion gradient steps:         95
mechanical_status:                        PASS
```

Every re-proposal branch also showed non-zero parameter movement.

Therefore the negative result cannot be explained by a dead dynamic-WHAT module.

The best full checkpoint reached:

```text
epoch:        3
Mean Recall: 37.800052
R@10:        26.514363
R@50:        49.085740
```

This is worse than:

```text
R1a best = 38.764146
R1c1 best = 38.754133
R1b best = 39.011857
```

The decisive causal intervention is not the headline MR alone.

Inside the **same trained R1c2 checkpoint**, disabling only dynamic WHAT while preserving dynamic current-state WHERE gives:

```text
BEST epoch 3

FULL dynamic WHAT + dynamic WHERE:
37.800052

frozen_t0_what:
39.259914

difference:
+1.459862 MR in favor of freezing WHAT
```

At the observed late checkpoint:

```text
epoch 6

FULL:
33.143262

frozen_t0_what:
38.690884

difference:
+5.547622 MR in favor of freezing WHAT
```

Thus:

\[
\boxed{
\text{Dynamic WHAT does not merely fail to help; under the learned R1c2 solution it actively harms retrieval.}
}
\]

This is the strongest result of R1c2.

The model weights themselves are not globally destroyed, because the same trained checkpoint recovers strong retrieval when only the dynamic WHAT intervention is disabled.

The representation diagnostics explain why.

At BEST, raw candidate WHAT similarity changes from:

```text
t0 = 0.953041
t1 = 0.962571
t2 = 0.962664
```

so dynamic re-proposal makes candidate intents **more similar**, not less.

The first semantic displacement is almost perfectly common-mode:

```text
t0 -> t1 candidate intent displacement alignment
= 0.999989
```

At the late checkpoint this strengthens to:

```text
t0 -> t1 = 0.999996
t1 -> t2 = 0.998962
```

The dynamic WHAT mechanism therefore learns a shared state-conditioned semantic transformation rather than candidate-specific sequential reconsideration.

This is the R1c2 failure mode:

\[
\boxed{
\textbf{moving-WHAT clones}
}
\]

Downstream WHERE remains almost perfectly cloned:

```text
BEST support cosine:
t0 0.999936
t1 0.999971
t2 0.999971

LATE support cosine:
t0 0.999944
t1 0.999980
t2 0.999976
```

and the functional effects remain highly parallel.

R1c2 should therefore be frozen as a:

\[
\boxed{
\text{mechanically valid but scientifically NEGATIVE causal experiment}
}
\]

The clean conclusion is:

\[
\boxed{
\text{current-state access alone is not enough to produce semantic decomposition.}
}
\]

The next experiment should not rescue R1c2 by stacking DPP, VISReg, teacher grounding, or a new STOP loss.

The next clean causal question in the existing ladder is R2: whether the system needs an explicit **semantic residual / remaining-evidence state** so that later actions operate on what remains unexplained rather than repeatedly re-reading the complete modification instruction.

---

# 2. Why R1c2 existed

The project had already isolated several different causes of failure.

## R0 — original collapse

R0 showed:

```text
DeltaZ remained active
but
Deltaq collapsed strongly with recurrent depth
```

approximately:

```text
||Δq||:
0.3665 -> 0.0825 -> 0.0190
```

while token-space effects remained large.

---

## R1a — query-cap repair

R1a changed only:

```yaml
query_cap:
  0.5 -> 1000.0
```

and restored healthy recurrent retrieval-space effects.

Best R1a:

```text
Mean Recall = 38.764146

Δq norm:
0.3366 -> 0.2724 -> 0.1971

t2/t0 retention:
58.6%
```

This established that multi-step recurrence can be useful once the cumulative retrieval readout is no longer aggressively compressed.

But candidate decomposition remained collapsed:

```text
intent cosine  ≈ 0.950
support cosine ≈ 0.99984
ΔZ cosine      ≈ 0.982
Δq cosine      ≈ 0.98
```

---

## R1b — dynamic applicability

R1b tested:

```text
static WHAT
static WHERE
dynamic scalar WHETHER
```

The gate was numerically healthy but stayed nearly always ON.

R1b was scientifically negative.

---

## R1c1 — dynamic current-state WHERE

R1c1 tested:

```text
fixed WHAT
+
dynamic current-state WHERE
```

with:

\[
I_k=Intent(q_k,T)
\]

computed once and:

\[
\pi_{t,k}=Ground(I_k,Z_t)
\]

recomputed every timestep.

Best R1c1:

```text
MR = 38.754133
```

which was effectively identical to R1a.

Supports remained approximately:

```text
t0 = 0.999766
t1 = 0.999755
t2 = 0.999740
```

between candidates.

With continued training, support movement increased but all four candidate supports moved together.

This was diagnosed as:

\[
\boxed{\text{moving WHERE clones}}
\]

The next upstream causal question was therefore:

> Are the four fixed WHAT vectors themselves the bottleneck?

R1c2 answered that question.

---

# 3. R1c2 scientific hypothesis

The R1c2 hypothesis was:

\[
H_{R1c2}:
\quad
\text{fixed WHAT is a major remaining bottleneck.}
\]

More concretely:

> If each persistent candidate can inspect the current edited state, inspect accumulated change, re-read token-level modification text, and revise its semantic proposal, then later candidate WHATs should become more state-sensitive and candidate-specific. This should propagate into more differentiated visual WHERE, more differentiated functional effects, and better retrieval.

R1c2 was not intended to force diversity.

It was intended to test whether state-conditioned semantic re-proposal would make useful specialization emerge naturally under the existing retrieval objective.

---

# 4. Exact R1c2 mechanism

## 4.1 Immutable base intent

At rollout start:

\[
B_k
=
IntentEncoder(q_k,T).
\]

`B_k` is retained as the immutable text-derived base intent.

---

## 4.2 Exact t0 parity

At timestep zero:

\[
\boxed{
I_{0,k}=B_k
}
\]

with no contribution from the new re-proposal module.

Thus t0 preserves R1c1 exactly for identical shared weights and inputs.

---

## 4.3 Current-state inspection

For `t > 0`, the persistent candidate reads:

\[
E^{state}_{t,k}
=
CrossAttn(B_k,Z_t)
\]

and accumulated visual change:

\[
E^{change}_{t,k}
=
CrossAttn(B_k,Z_t-A)
\]

where:

\[
A=Z_0
\]

is the immutable anchor state.

---

## 4.4 Token-level text re-read

The current state/change evidence modifies the query used to re-read the original token-level modification text:

\[
Q^{text}_{t,k}
=
B_k
+
W_s
[
E^{state}_{t,k};
E^{change}_{t,k}
].
\]

Then:

\[
E^{text}_{t,k}
=
CrossAttn(
Q^{text}_{t,k},
T,
M_T
).
\]

---

## 4.5 Residual dynamic WHAT

The re-proposal branch predicts:

\[
\Delta I_{t,k}
=
W_{out}H_{t,k}.
\]

The actual dynamic candidate intent is:

\[
\boxed{
I_{t,k}
=
B_k+\Delta I_{t,k}
}
\]

for `t>0`.

---

## 4.6 Zero-output initialization

The new final projection is initialized:

\[
W_{out}=0.
\]

Therefore at initialization:

\[
\Delta I_{t,k}=0
\]

and:

\[
I_{t,k}=B_k.
\]

Hence R1c2 begins exactly inside the R1c1 solution family rather than with a random semantic perturbation.

---

## 4.7 Dynamic WHERE remains unchanged

After obtaining dynamic WHAT:

\[
Q_{t,k}=W_Q I_{t,k}
\]

\[
K_{t,n}=W_K Z_{t,n}
\]

\[
\ell_{t,k,n}
=
\frac{
Q_{t,k}^{\top}K_{t,n}
}{
\sqrt{d_g}
}
\]

and:

\[
\boxed{
\pi_{t,k}
=
Entmax_{1.5}
(
\ell_{t,k,:}
)
}
\]

exactly as the R1c1 dynamic-WHERE mechanism.

---

# 5. Causal isolation

R1c2 deliberately changed only dynamic WHAT on top of R1c1.

It did **not** add:

```text
DPP
FuncDPP
RDMReg
VISReg
variance-floor loss
orthogonality loss
explicit candidate repulsion
slot ownership
R1b applicability
Visual NULL
semantic residual
teacher grounding
target-conditioned proposal
LLM / MLLM
pseudo labels
RL / DQN
new STOP loss
new scorer objective
new selector objective
candidate-specific reproposal networks
candidate-specific grounders
candidate-specific editors
timestep-specific parameter banks
```

Thus the intended causal comparison is:

\[
\boxed{
R1c2 - R1c1
\approx
\text{effect of dynamic WHAT re-proposal}
}
\]

subject to normal stochastic training variation.

More importantly, the in-checkpoint `frozen_t0_what` control gives a much cleaner intervention because it uses exactly the same trained weights.

---

# 6. `frozen_t0_what` — the decisive control

R1c2 includes a diagnostic-only inference control:

```text
frozen_t0_what
```

Under this control:

\[
I_{t,k}=B_k
\]

for every timestep.

But dynamic current-state WHERE remains active:

\[
\pi_{t,k}
=
Ground(B_k,Z_t).
\]

Therefore:

```text
FULL:
dynamic WHAT + dynamic WHERE

frozen_t0_what:
fixed WHAT + dynamic WHERE
```

using:

```text
the same checkpoint
the same backbone
the same grounder
the same editor
the same readout
the same scorer
the same FashionIQ validation set
```

This is the most important causal control in the R1c2 experiment.

---

# 7. Implementation audit and target firewall

The final R1c2 branch was audited before the full run.

Key invariants:

```text
base intent encoder call count = 1
reproposal calls for Tmax=3   = 2
grounder calls                = 3
applicability calls           = 0
```

The re-proposal and grounding modules both receive the actual selected-path current state.

All four candidates branch from the same current parent:

\[
\widehat Z_{t+1}^{(k)}
=
Z_t+\Delta Z_{t,k}.
\]

Candidate identity remains persistent:

```text
k=0
k=1
k=2
k=3
```

while WHAT is allowed to evolve.

Target is excluded from:

```text
base intent
state inspection
change inspection
text re-read
dynamic WHAT
dynamic WHERE
editor
scorer
selection
STOP
state commit
final query
```

Target features are consumed only after the target-free rollout exists for offline diagnostics and retrieval evaluation.

Post-STOP hypothetical vectorized WHAT/WHERE recomputations are explicitly excluded from primary lineage-safe temporal metrics.

---

# 8. CUDA canary result

Command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/canary_train_iag_srme.py \
  --r1c2 \
  --dataset-root data/FashionIQ \
  --steps 100 \
  --precision fp16
```

Mechanical result:

```text
attempted steps:                  100
successful optimizer steps:        96
skipped AMP-overflow steps:          4
first successful step:               3
initial GradScaler:              65536
final/min GradScaler:             4096
finite:                           true
mechanical_status:                PASS
```

All re-proposal subpaths were trainable:

```text
reproposal_output      96 nonzero-gradient successful steps
reproposal_state       95
reproposal_change      95
reproposal_text        95
reproposal_state_query 95
reproposal_fusion      95
```

Every audited re-proposal parameter family moved.

Therefore:

\[
\boxed{
\text{R1c2 negative behavior is not a dead-module implementation artifact.}
}
\]

---

# 9. Canary scientific warning

The canary already exposed the future R1c2 failure mode.

At the end of the canary:

```text
t1 mean residual norm from base ≈ 0.0688
t2 mean residual norm from base ≈ 0.0683
```

so dynamic WHAT was numerically active.

However:

```text
candidate intent displacement cosine
t0 -> t1 ≈ 0.999647
```

and the canary raised:

```text
high_intent_displacement_comotion = true
```

Thus even before full training:

\[
\Delta I_{0\rightarrow1,0}
\approx
\Delta I_{0\rightarrow1,1}
\approx
\Delta I_{0\rightarrow1,2}
\approx
\Delta I_{0\rightarrow1,3}.
\]

This was the first evidence for:

\[
\boxed{\text{moving-WHAT clones}}
\]

but canary evidence alone was not sufficient to reject R1c2.

Full training and checkpoint interventions were still required.

---

# 10. Full training command

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r1c2_dynamic_reproposal \
  experiment=iag_srme_r1c2_dynamic_reproposal \
  protocol=fashioniq_original
```

Observed training trajectory before the run was stopped for forensic diagnosis:

| Epoch | Train total loss | Mean Recall | Best so far |
|---:|---:|---:|---:|
| 1 | 1.3633 | 33.709 | 33.709 |
| 2 | 0.6257 | 36.898 | 36.898 |
| 3 | 0.3479 | **37.800** | **37.800** |
| 4 | 0.2220 | 37.634 | 37.800 |
| 5 | 0.1599 | 36.078 | 37.800 |
| 6 | 0.1286 | **33.143** | 37.800 |

Important wording:

```text
epoch 6 is the observed LATE/current-last checkpoint,
not a completed epoch-20 training endpoint.
```

The pattern is nevertheless already clear:

\[
L_{train}\downarrow
\]

while:

\[
MR_{val}\downarrow.
\]

The optimization process remains active while validation retrieval becomes worse.

---

# 11. Trusted diagnostic checkpoints

Run:

```text
outputs/2026-08-31/17-36-17/
```

BEST:

```text
best.pt
epoch = 3
saved MR = 37.800051768620804
replayed MR = 37.80005176862081
trusted_r1c2_replay = true
```

Replay error:

```text
~7.1e-15
```

LATE/current-last:

```text
last.pt
epoch = 6
saved MR = 33.14326231678327
replayed MR = 33.14326231678327
trusted_r1c2_replay = true
```

Replay error:

```text
0
```

Thus both diagnostic reports replay the intended R1c2 architecture/configuration exactly within numerical tolerance.

---

# 12. Headline retrieval comparison

| Experiment | Best Mean Recall | Interpretation |
|---|---:|---|
| R1a | 38.764146 | query-cap repair PASS |
| R1b | 39.011857 | applicability mechanism negative despite small headline gain |
| R1c1 | 38.754133 | dynamic WHERE negative |
| **R1c2 FULL** | **37.800052** | dynamic WHAT + dynamic WHERE |

R1c2 best difference:

```text
vs R1a:
-0.964094

vs R1c1:
-0.954082

vs R1b:
-1.211805
```

Therefore R1c2 produces no headline retrieval benefit.

But this is still not the strongest causal evidence.

---

# 13. BEST control battery

At epoch 3:

```text
FULL             37.800052
frozen_t0_what   39.259914
MEAN             37.664964
REPEAT-0         38.748105
REPEAT-1         38.630403
REPEAT-2         38.556584
REPEAT-3         38.352332
best SINGLE      27.553990
REFERENCE_ONLY   14.019122
```

The critical gaps are:

\[
39.259914-37.800052
=
\boxed{+1.459862}
\]

for freezing WHAT, and:

\[
38.748105-37.800052
=
+0.948053
\]

for the best fixed REPEAT identity.

MEAN is essentially equal to FULL:

\[
37.664964
\approx
37.800052.
\]

This already tells us:

1. dynamic WHAT is not helping;
2. candidate selection does not extract strong unique candidate semantics;
3. a fixed candidate trajectory is competitive;
4. disabling only dynamic WHAT gives the strongest retrieval of the control set.

---

# 14. The decisive result: dynamic WHAT actively hurts

The same trained BEST checkpoint gives:

```text
FULL:
37.800052

frozen_t0_what:
39.259914
```

Hence:

\[
\boxed{
FULL < frozen\_t0\_what
}
\]

by:

\[
\boxed{
1.459862\ \text{Mean Recall points}.
}
\]

This is stronger than simply observing that R1c2 is below R1c1.

The frozen control holds the trained checkpoint constant.

The only intended inference intervention is:

```text
remove dynamic WHAT
while
retain current-state dynamic WHERE
```

Therefore the direct observation is:

\[
\boxed{
\text{the learned dynamic-WHAT pathway has negative retrieval utility.}
}
\]

This is the central R1c2 causal result.

---

# 15. Important caveat about `frozen_t0_what`

At BEST:

```text
frozen_t0_what = 39.259914
```

which is numerically above:

```text
R1a  = 38.764146
R1c1 = 38.754133
```

Do **not** interpret this as a new benchmark result.

`frozen_t0_what` uses weights trained under R1c2 and then applies an inference intervention.

It establishes:

```text
the R1c2-trained backbone / grounder / editor / readout
still contain a strong retrieval solution
```

not:

```text
R1c2 dynamic WHAT improved FashionIQ.
```

In fact, FULL R1c2 fails to exploit that stronger solution.

---

# 16. BEST same-parent recurrent depth remains useful

Mean candidate retrieval from the same current parent:

```text
t0 = 27.3534
t1 = 34.9748
t2 = 38.7342
```

Offline best-candidate oracle:

```text
t0 = 27.9370
t1 = 35.4420
t2 = 39.2086
```

Thus:

\[
27.35
\rightarrow
34.97
\rightarrow
38.73.
\]

This again demonstrates:

\[
\boxed{
\text{recurrent depth itself is not falsified by R1c2.}
}
\]

The architecture can construct better retrieval candidates at later depth.

The failure lies in semantic decomposition / dynamic proposal / policy use of that depth.

---

# 17. BEST temporal WHAT — re-proposal makes candidates more alike

Base/t0 candidate intent cosine:

```text
t0 = 0.953041
```

After the first dynamic re-proposal:

```text
t1 = 0.962571
```

At the final recurrent timestep:

```text
t2 = 0.962664
```

Therefore:

\[
\boxed{
\text{candidate WHAT similarity increases after dynamic re-proposal.}
}
\]

This is the opposite of the desired behavior.

R1c2 was supposed to allow different persistent candidates to reconsider different remaining semantic actions.

Instead:

```text
candidate 0 ─┐
candidate 1 ─┼─→ more similar WHAT vectors after reproposal
candidate 2 ─┤
candidate 3 ─┘
```

---

# 18. BEST intent displacement proves common-mode motion

At BEST:

```text
candidate-intent displacement alignment:

t0 -> t1 = 0.999989
t1 -> t2 = 0.973503
```

The first and dominant re-proposal is therefore almost perfectly aligned across all four candidates:

\[
\Delta I_{0\to1,0}
\parallel
\Delta I_{0\to1,1}
\parallel
\Delta I_{0\to1,2}
\parallel
\Delta I_{0\to1,3}.
\]

This is not four candidate-specific semantic revisions.

It is a shared semantic mode shift.

The second transition is less perfectly aligned, but its magnitude is tiny.

At BEST, the mean `t1 -> t2` intent L2 change is approximately:

```text
0.002576
```

while the first semantic transition is orders of magnitude larger.

Thus the learned behavior is closer to:

```text
t0:
base WHAT

        ↓ large common-mode shift

t1:
post-step shared semantic mode

        ↓ tiny adjustment

t2:
almost same post-step semantic mode
```

than to:

```text
continually reconsider what semantic edit remains after each action
```

---

# 19. BEST WHERE still completely contracts candidate semantics

Despite dynamic WHAT, raw visual support cosine is:

```text
t0 = 0.9999357
t1 = 0.9999711
t2 = 0.9999706
```

The later supports are actually **more** similar than t0.

Thus:

\[
\boxed{
I_{t,k}\text{ changes}
\quad\not\Rightarrow\quad
\pi_{t,k}\text{ becomes candidate-specific}.
}
\]

The chain is:

```text
WHAT moves
     ↓
WHAT moves mostly together
     ↓
grounder maps them to nearly the same visual support
```

R1c2 therefore does not break the WHAT→WHERE contraction.

---

# 20. BEST support shape is valid but semantically redundant

BEST t0 support summary:

```text
effective size ≈ 17.32 / 196
support entropy ≈ 2.834
support fraction ≈ 8.99%
support overlap ≈ 0.99623
mass ≈ 1.0
```

This is not a broken normalization or empty-support problem.

The supports are valid Entmax distributions.

The problem is:

\[
\boxed{
\pi_{t,0}\approx\pi_{t,1}\approx\pi_{t,2}\approx\pi_{t,3}.
}
\]

Hence the failure is semantic/functional redundancy, not malformed support.

---

# 21. BEST context becomes more cloned with depth

Context pairwise cosine:

```text
t0 = 0.958868
t1 = 0.974966
t2 = 0.974975
```

So the dynamic WHAT + WHERE pipeline does not preserve or amplify candidate differences.

Instead:

\[
\boxed{
\text{candidate context similarity increases with recurrence.}
}
\]

This provides a downstream bridge between the WHAT/WHERE observations and the functional edit collapse.

---

# 22. BEST token-space effects remain parallel

BEST `ΔZ`:

```text
mean norm:
t0 = 2.1286
t1 = 1.4367
t2 = 1.4955
```

Pairwise cosine:

```text
t0 = 0.98410
t1 = 0.98475
t2 = 0.98611
```

Effective rank:

```text
t0 = 1.8069
t1 = 1.8030
t2 = 1.7761
```

All candidates are numerically active.

The issue is not dead effects.

The issue is:

\[
\boxed{
\Delta Z_{t,0}
\approx
\Delta Z_{t,1}
\approx
\Delta Z_{t,2}
\approx
\Delta Z_{t,3}.
}
\]

---

# 23. BEST retrieval-space effects remain cloned

BEST `Δq`:

```text
mean norm:
t0 = 0.42263
t1 = 0.22159
t2 = 0.17031
```

Retention:

```text
t1/t0 = 0.5243
t2/t0 = 0.4030
t2/t1 = 0.7686
```

Pairwise cosine:

```text
t0 = 0.98591
t1 = 0.98304
t2 = 0.98045
```

Effective rank:

```text
t0 = 1.7581
t1 = 1.8308
t2 = 1.8903
```

So R1c2 preserves some recurrent retrieval-space effect magnitude, but considerably less than the successful R1a trajectory:

```text
R1a t2/t0 retention ≈ 58.6%
R1c2 BEST t2/t0    ≈ 40.3%
```

Dynamic WHAT therefore does not solve candidate redundancy and also weakens late retrieval-effect retention.

---

# 24. BEST selected-path utility still over-edits at t2

Selected target-relative retrieval gain:

```text
t0 = +0.07833
t1 = +0.01620
t2 = -0.00348
```

Thus R1c2 does not solve the late harmful-edit problem.

The third executed edit is again slightly harmful on average at the best checkpoint.

This is important because dynamic WHAT was supposed to help the system reconsider what remains to be edited.

Instead, it does not reliably suppress or reinterpret already-satisfied late actions.

---

# 25. BEST policy behavior

BEST selection summary:

```text
mean executed edits:
2.636 / 3

fraction queries with repeated candidate selections:
81.37%

new STOP hazard:
t0 = 0.183%
t1 = 15.554%
t2 = 5.699%

maximum candidate share conditional on edit:
36.99%
```

The candidate monopoly problem is not extreme in the simple selection-count sense.

However, candidate identities remain functionally redundant.

This distinction matters:

```text
balanced-ish selection frequency
does not imply
meaningfully different candidate functions.
```

---

# 26. BEST scientific verdict

By the best checkpoint, R1c2 has already failed its central causal hypothesis.

Desired:

```text
dynamic WHAT
→ candidate-specific semantic revision
→ differentiated WHERE
→ differentiated effects
→ retrieval improvement
```

Observed:

```text
dynamic WHAT
→ common-mode semantic shift
→ candidate WHAT becomes more similar
→ WHERE remains ≈ identical
→ effects remain parallel
→ FULL retrieval worsens
→ freezing WHAT improves retrieval by +1.46 MR
```

Hence:

\[
\boxed{
H_{R1c2}
\text{ is rejected / strongly weakened under the current mechanism and objective.}
}
\]

---

# 27. BEST → LATE retrieval degradation

Observed:

```text
BEST epoch 3:
FULL = 37.800052

LATE epoch 6:
FULL = 33.143262
```

Difference:

\[
\boxed{
-4.656789\ MR
}
\]

while training loss continued decreasing:

```text
0.3479 -> 0.1286.
```

Again:

\[
L_{train}\downarrow
\qquad
MR_{val}\downarrow.
\]

This indicates objective/solution-path misalignment rather than simple optimization failure.

---

# 28. LATE frozen-WHAT intervention isolates the damage

At epoch 6:

```text
FULL             = 33.143262
frozen_t0_what   = 38.690884
best REPEAT      = 35.711916
MEAN             = 33.036222
REFERENCE_ONLY   = 13.936077
```

The frozen-WHAT gap is now:

\[
38.690884-33.143262
=
\boxed{+5.547622}.
\]

Ratio:

```text
frozen_t0_what / FULL ≈ 1.1674
```

This is the strongest evidence in the entire R1c2 experiment.

If the backbone or all shared representations had simply collapsed globally, then freezing only dynamic WHAT would not be expected to recover nearly 39 MR.

Instead:

```text
same trained checkpoint
+
same dynamic WHERE/editor/readout
+
disable only dynamic WHAT
=
large recovery
```

Therefore the full-run degradation is tightly associated with the learned dynamic-WHAT execution pathway.

---

# 29. LATE WHAT becomes even more common-mode

LATE base candidate intent similarity:

```text
t0 = 0.950865
```

After re-proposal:

```text
t1 = 0.971176
t2 = 0.970782
```

Thus the dynamic WHAT transformation increases candidate similarity by roughly:

```text
+0.020 cosine
```

relative to the base candidate set.

The dynamic re-proposal is behaving as a semantic contraction.

---

# 30. LATE first WHAT transition is almost perfectly shared

At LATE:

```text
t0 -> t1 displacement alignment
= 0.999996
```

The average t0→t1 intent L2 change is approximately:

```text
11.97
```

This is now a **large** semantic movement.

But because the displacement directions are almost identical across candidates, this is not evidence for healthy candidate specialization.

It is evidence for a large shared transformation:

\[
\boxed{
I_{1,k}
\approx
B_k + \Delta I_{\text{shared}}
}
\]

for all `k`.

---

# 31. LATE t1→t2 WHAT motion also becomes clone-like

At LATE:

```text
t1 -> t2 displacement alignment
= 0.998962
```

The mean intent L2 change is only approximately:

```text
0.01467
```

Thus later re-proposal is:

```text
small
+
nearly common-mode
```

rather than a meaningful new semantic decomposition.

The final learned pattern is approximately:

```text
base WHATs
   ↓
large shared semantic transform
   ↓
more correlated WHATs
   ↓
tiny shared adjustment
```

---

# 32. LATE WHERE remains cloned

LATE support cosine:

```text
t0 = 0.999944
t1 = 0.999980
t2 = 0.999976
```

R1c2 therefore does not merely fail to create diverse WHAT.

Even the changed WHAT representation is still mapped by the grounder into essentially the same spatial support.

This suggests the model found a globally useful state-conditioned semantic mode rather than four local compositional actions.

---

# 33. LATE functional effects collapse further

BEST → LATE `Δq` cosine:

```text
BEST:
t0 0.98591
t1 0.98304
t2 0.98045

LATE:
t0 0.98974
t1 0.98895
t2 0.98767
```

Thus candidate effects become more parallel.

BEST → LATE effective rank:

```text
BEST:
1.758 -> 1.831 -> 1.890

LATE:
1.670 -> 1.709 -> 1.744
```

Thus functional rank decreases.

The late checkpoint has **stronger functional cloning**.

---

# 34. LATE effect magnitude also attenuates badly

LATE `Δq` norm:

```text
t0 = 0.44529
t1 = 0.13205
t2 = 0.11481
```

Retention:

```text
t1/t0 = 29.65%
t2/t0 = 25.78%
t2/t1 = 86.94%
```

Compare:

```text
R1a best t2/t0:
58.6%

R1c2 BEST:
40.3%

R1c2 LATE:
25.8%
```

The query cap remains 1000.

Therefore the late attenuation is not a return of the original `query_cap=0.5` mechanism.

It reflects a new/deeper degeneration of the recurrent semantic/context/edit pathway.

---

# 35. LATE token effects also shrink after t0

LATE `ΔZ` norm:

```text
t0 = 2.1869
t1 = 0.8766
t2 = 0.9093
```

Pairwise cosine:

```text
t0 = 0.98717
t1 = 0.98959
t2 = 0.98984
```

Effective rank:

```text
t0 = 1.7420
t1 = 1.6957
t2 = 1.6878
```

Thus the late model learns:

```text
one large first edit
+
much weaker later edits
+
highly parallel candidate directions.
```

This is consistent with the dynamic-WHAT pathway collapsing toward one common first-step transformation.

---

# 36. LATE context becomes nearly identical

Context cosine:

```text
t0 = 0.95930
t1 = 0.98657
t2 = 0.98620
```

After the large common-mode WHAT shift, downstream candidate contexts become almost identical.

Thus the collapse chain is visible continuously:

```text
WHAT contraction
→ WHERE contraction
→ context contraction
→ ΔZ contraction
→ Δq contraction.
```

---

# 37. LATE recurrence itself still contains useful depth

Despite the collapse, same-parent mean candidate retrieval remains:

```text
t0 = 28.2521
t1 = 32.6954
t2 = 36.1683
```

Offline oracle:

```text
t0 = 28.7841
t1 = 32.9420
t2 = 36.4405
```

Therefore:

\[
28.25
\rightarrow
32.70
\rightarrow
36.17.
\]

Again:

\[
\boxed{
\text{multi-step depth itself remains useful.}
}
\]

The model can still improve candidate queries through recurrence.

The issue is how dynamic WHAT and the learned policy organize/use this computation.

---

# 38. LATE selected edits remain positively useful, but weak

LATE selected target-relative gain:

```text
t0 = +0.08854
t1 = +0.01341
t2 = +0.00468
```

So unlike BEST, the selected t2 transitions that are actually executed are slightly positive on average.

This means the late FULL collapse cannot be described simply as:

```text
"all later edits are destructive."
```

There is still useful recurrent computation.

But the policy and dynamic-WHAT pathway do not turn it into a strong final retrieval solution.

---

# 39. LATE STOP/scorer pathology also appears

LATE:

```text
FULL        = 33.1433
best REPEAT = 35.7119
```

Gap:

\[
\boxed{
+2.5687\ MR
}
\]

for a fixed REPEAT trajectory.

STOP policy:

```text
new STOP hazard:
t0 = 5.07%
t1 = 32.97%
t2 = 14.76%

mean executed edits:
2.128 / 3
```

Repeated-candidate trajectory fraction:

```text
56.98%
```

Thus a secondary policy/scorer problem remains:

```text
forced continuation / fixed identity
can outperform learned FULL selection.
```

However, in R1c2 this is **not the primary causal finding**.

The larger and cleaner gap is:

```text
frozen_t0_what 38.691
vs
FULL           33.143
```

Therefore dynamic WHAT should be diagnosed before trying to rescue STOP.

---

# 40. Why FULL ≈ MEAN still matters

BEST:

```text
FULL = 37.8001
MEAN = 37.6650
```

LATE:

```text
FULL = 33.1433
MEAN = 33.0362
```

The mean-candidate control remains nearly equivalent to the learned candidate-selection policy.

Together with:

```text
high WHAT similarity
high support similarity
high ΔZ similarity
high Δq similarity
```

this shows that candidate identities still carry little unique functional information.

---

# 41. R1c2 causal hypothesis verdict

Original hypothesis:

> Fixed WHAT is a major remaining bottleneck and dynamically regenerating WHAT from the current state will resolve candidate redundancy / improve sequential editing.

Verdict:

\[
\boxed{
\text{REJECTED / strongly falsified for the current reproposal mechanism and objective.}
}
\]

Evidence:

1. R1c2 is mechanically trainable.
2. All dynamic-WHAT submodules receive gradient and move.
3. t0 parity is exact.
4. R1c2 BEST is ~0.95 MR below R1c1.
5. `frozen_t0_what` beats FULL by +1.46 MR at BEST.
6. `frozen_t0_what` beats FULL by +5.55 MR at LATE.
7. Candidate WHAT becomes more similar after reproposal.
8. First-step WHAT displacement is approximately perfectly common-mode.
9. Later WHAT movement is tiny and/or common-mode.
10. WHERE remains ~0.99997–0.99998 cosine.
11. Context becomes more similar with depth.
12. ΔZ remains highly parallel.
13. Δq remains highly parallel.
14. Functional rank worsens with continued training.
15. FULL remains approximately equivalent to MEAN.
16. REPEAT controls remain competitive or superior.
17. Same-parent recurrence still contains useful depth, so the negative result is not evidence against recurrence itself.

---

# 42. Failure mode classification

```text
MECHANICAL IMPLEMENTATION:
PASS

TARGET FIREWALL:
PASS

T0 PARITY:
PASS

REPROPOSAL TRAINABILITY:
PASS

DYNAMIC WHAT NUMERIC ACTIVITY:
PASS

CANDIDATE-SPECIFIC WHAT:
FAIL

DYNAMIC WHAT RETRIEVAL UTILITY:
FAIL / HARMFUL

DYNAMIC WHERE SPECIALIZATION:
FAIL

FUNCTIONAL SPECIALIZATION:
FAIL

BEST RETRIEVAL:
WORSE THAN R1a / R1c1

LATE GENERALIZATION:
DEGRADES

POLICY / STOP:
SECONDARY FAILURE ALSO PRESENT

SCIENTIFIC VERDICT:
NEGATIVE
```

---

# 43. Updated causal chain

The strongest current chain is:

```text
full modification text T
        ↓
4 already-correlated base WHATs B_k
        ↓
shared state/change-conditioned reproposal
        ↓
large common-mode ΔI
        ↓
WHATs become MORE similar
        ↓
Ground(I_t,k, Z_t)
        ↓
almost identical WHERE
        ↓
almost identical contexts
        ↓
parallel ΔZ
        ↓
parallel Δq
        ↓
candidate identities remain interchangeable
        ↓
FULL ≈ MEAN
REPEAT competitive
        ↓
dynamic WHAT actively reduces retrieval vs frozen-WHAT control
```

Compactly:

\[
\boxed{
\text{correlated WHAT}
\rightarrow
\text{common-mode dynamic WHAT}
\rightarrow
\text{clone WHERE}
\rightarrow
\text{clone effects}.
}
\]

And with continued training:

\[
\boxed{
\text{the common-mode solution strengthens while validation retrieval falls.}
}
\]

---

# 44. The key conceptual lesson

R1c1 established:

\[
\boxed{
\text{dynamic WHERE}
\neq
\text{candidate-specific WHERE}.
}
\]

R1c2 now establishes:

\[
\boxed{
\text{dynamic WHAT}
\neq
\text{candidate-specific WHAT}.
}
\]

Merely giving a module access to the current state does not make decomposition emerge.

The current retrieval objective permits an easier solution:

```text
all candidates repeatedly encode the same globally useful instruction/state correction
```

instead of:

```text
candidate k represents a distinct unresolved semantic factor.
```

---

# 45. What R1c2 falsifies

R1c2 provides strong evidence against:

> The primary remaining problem is simply that candidate WHAT is stale because it is computed once at t0.

The model is allowed to recompute WHAT.

The pathway learns strongly.

Yet the result is worse.

Therefore stale-WHAT access alone is not the dominant missing ingredient.

---

# 46. What R1c2 does NOT falsify

R1c2 does **not** establish that:

```text
multi-step CIR is wrong.
```

Same-parent retrieval still improves substantially with depth.

It does not establish that:

```text
dynamic reasoning is useless.
```

It establishes only that this unconstrained full-text/current-state reproposal does not produce useful candidate-specific decomposition.

It does not establish that:

```text
semantic residual is useless.
```

R1c2 does not maintain an explicit notion of which textual evidence has already been explained.

It re-reads the whole modification text every step.

It does not establish that:

```text
DPP is required.
```

Explicit geometric/functional diversity has still not been tested in this branch.

It does not establish that:

```text
teacher grounding is required.
```

No target-privileged teacher was used.

It does not establish that:

```text
STOP is the sole failure.
```

Freezing WHAT produces a much larger recovery than the best REPEAT-vs-FULL gap.

---

# 47. Why the R1c2 architecture admits a shortcut

R1c2 always retains:

\[
B_k
\]

and re-reads the entire original modification text:

\[
T.
\]

There is no explicit state variable saying:

```text
these textual requirements have already been explained/satisfied
```

and:

```text
these requirements remain unresolved.
```

Therefore the shared re-proposal module can learn an easy rule like:

\[
I_{t,k}
=
B_k
+
g(Z_t,T)
\]

where:

\[
g(Z_t,T)
\]

is largely shared across candidates.

Nothing in the current objective forces:

\[
g_1 \neq g_2 \neq g_3 \neq g_4
\]

or forces later `g` to represent only residual semantics.

This matches the observed common-mode displacement.

---

# 48. Why R2 semantic residual is now a justified next test

The next clean question should be:

> Is the missing ingredient not more state-conditioning, but an explicit representation of what semantic evidence remains unexplained after previous edits?

Conceptually:

```text
R1c2:
full T
+
current Z_t
→ repropose WHAT

possible R2:
remaining semantic evidence R_t
+
current Z_t
→ propose what is still missing
```

The causal idea is to introduce an **explaining-away / residual evidence state**, not an arbitrary diversity penalty.

For example conceptually:

\[
R_0 = T
\]

and after executing an action:

\[
R_{t+1}
=
UpdateResidual(
R_t,
I^{selected}_t,
Z_t,
Z_{t+1}
).
\]

Then future proposal should read:

\[
R_t
\]

rather than repeatedly having unrestricted access to the same full instruction representation.

This is only the next hypothesis.

It is not assumed to work.

---

# 49. Why R2 should come before R3 DPP

R3 / Functional DPP could force candidate effects to spread apart.

But R1c2 has not shown that geometric separation is the missing causal ingredient.

The failure currently looks like:

```text
everyone is allowed to explain the whole instruction
```

rather than:

```text
the model knows distinct residual factors but accidentally maps them to parallel vectors.
```

If DPP is added immediately, the model could produce:

```text
artificially different vectors
```

without learning:

```text
different useful remaining edits.
```

Therefore preserve the ladder:

```text
R2 semantic residual / claim firewall
before
R3 explicit functional diversity
```

unless a separate new causal analysis changes this ordering.

---

# 50. What should NOT be done on R1c2 now

Do not rescue this branch with:

```text
DPP
VISReg
RDMReg
orthogonality
variance floor
teacher grounding
new STOP loss
RL
candidate-balance penalty
new query cap
new editor
new grounder
```

Do not hyperparameter-fish dynamic WHAT strength.

Do not reinterpret:

```text
frozen_t0_what = 39.26
```

as evidence that FULL R1c2 succeeded.

Freeze the negative result.

Create a new branch for the next causal hypothesis.

---

# 51. BEST vs LATE table

| Metric | R1c2 BEST e3 | R1c2 LATE e6 |
|---|---:|---:|
| FULL MR | **37.8001** | **33.1433** |
| frozen_t0_what MR | **39.2599** | **38.6909** |
| frozen minus FULL | **+1.4599** | **+5.5476** |
| best REPEAT MR | 38.7481 | 35.7119 |
| MEAN MR | 37.6650 | 33.0362 |
| REF MR | 14.0191 | 13.9361 |
| same-parent mean MR t0 | 27.3534 | 28.2521 |
| same-parent mean MR t1 | 34.9748 | 32.6954 |
| same-parent mean MR t2 | 38.7342 | 36.1683 |
| base/t0 WHAT cosine | 0.95304 | 0.95086 |
| t1 WHAT cosine | 0.96257 | 0.97118 |
| t2 WHAT cosine | 0.96266 | 0.97078 |
| WHAT displacement alignment t0→t1 | 0.999989 | 0.999996 |
| WHAT displacement alignment t1→t2 | 0.97350 | 0.99896 |
| WHERE cosine t0 | 0.999936 | 0.999944 |
| WHERE cosine t1 | 0.999971 | 0.999980 |
| WHERE cosine t2 | 0.999971 | 0.999976 |
| Δq norm t0 | 0.42263 | 0.44529 |
| Δq norm t1 | 0.22159 | 0.13205 |
| Δq norm t2 | 0.17031 | 0.11481 |
| Δq t2/t0 retention | 40.3% | 25.8% |
| Δq cosine t0 | 0.98591 | 0.98974 |
| Δq cosine t1 | 0.98304 | 0.98895 |
| Δq cosine t2 | 0.98045 | 0.98767 |
| functional rank t0 | 1.758 | 1.670 |
| functional rank t1 | 1.831 | 1.709 |
| functional rank t2 | 1.890 | 1.744 |
| ΔZ cosine t0 | 0.98410 | 0.98717 |
| ΔZ cosine t1 | 0.98475 | 0.98959 |
| ΔZ cosine t2 | 0.98611 | 0.98984 |
| mean edits | 2.636 | 2.128 |
| repeated-candidate fraction | 81.37% | 56.98% |
| target gain t0 | +0.07833 | +0.08854 |
| target gain t1 | +0.01620 | +0.01341 |
| target gain t2 | -0.00348 | +0.00468 |

The table reveals two simultaneous trends:

```text
dynamic WHAT becomes increasingly dominant/common-mode
AND
functional candidate diversity decreases.
```

---

# 52. Matched lineage comparison

| Metric | R1a BEST | R1c1 BEST | R1c2 BEST |
|---|---:|---:|---:|
| FULL MR | 38.7641 | 38.7541 | 37.8001 |
| same-parent mean MR t0 | 24.603 | 24.651 | 27.353 |
| same-parent mean MR t1 | 34.302 | 33.835 | 34.975 |
| same-parent mean MR t2 | 39.030 | 38.859 | 38.734 |
| Δq norm t0 | 0.3366 | 0.3303 | 0.4226 |
| Δq norm t1 | 0.2724 | 0.2670 | 0.2216 |
| Δq norm t2 | 0.1971 | 0.1941 | 0.1703 |
| t2/t0 retention | 58.6% | 58.8% | 40.3% |
| support cosine | 0.999842 static | ~0.99975 dynamic | ~0.99994–0.99997 dynamic |
| Δq cosine t0 | ~0.983 | 0.9828 | 0.9859 |
| Δq cosine t2 | ~0.977 | 0.9754 | 0.9804 |
| functional rank t0 | ~1.818 | 1.830 | 1.758 |
| functional rank t2 | ~1.954 | 1.971 | 1.890 |

R1c2 does not advance the candidate-specialization problem.

Its strongest unique observation is the direct `frozen_t0_what` intervention.

---

# 53. Reproduction — BEST diagnostic

```bash
RUN=outputs/2026-08-31/17-36-17

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/best.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1c2_dynamic_reproposal_best.json
```

Expected replay:

```text
epoch = 3
FULL MR = 37.80005176862081
trusted_r1c2_replay = true
```

---

# 54. Reproduction — LATE/current-last diagnostic

```bash
RUN=outputs/2026-08-31/17-36-17

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/last.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1c2_dynamic_reproposal_late.json
```

Expected replay for the checkpoint analyzed in this README:

```text
epoch = 6
FULL MR = 33.14326231678327
trusted_r1c2_replay = true
```

If training is later resumed and `last.pt` changes, do not overwrite the interpretation in this README without recording a new diagnostic checkpoint/version.

---

# 55. Experiment ledger update

Before R1c2 result:

```text
R0   diagnostic audit                         DONE
R1a  remove global query cap                  PASS
R1b  dynamic applicability / Visual NULL      NEGATIVE
R1c1 dynamic current-state WHERE              NEGATIVE
R1c2 dynamic current-state WHAT               RUNNING
R2   semantic residual / claim firewall       PENDING
R3   quality-gated functional DPP             CONDITIONAL
R4   target-privileged grounding teacher      CONDITIONAL
R5   planning / STOP refinement               CONDITIONAL
```

After this diagnosis:

```text
R0   diagnostic audit                         DONE
R1a  remove global query cap                  PASS
R1b  dynamic applicability / Visual NULL      NEGATIVE
R1c1 dynamic current-state WHERE              NEGATIVE
R1c2 dynamic current-state WHAT               NEGATIVE
R2   semantic residual / claim firewall       NEXT
R3   quality-gated functional DPP             CONDITIONAL
R4   target-privileged grounding teacher      CONDITIONAL
R5   planning / STOP refinement               CONDITIONAL
```

---

# 56. Updated hypothesis ledger

## H — Small global query cap suppresses useful recurrence

```text
SUPPORTED
```

R1a repaired it.

---

## H — Scalar state-conditioned applicability is sufficient

```text
REJECTED
```

R1b.

---

## H — Static WHERE reuse is the dominant remaining bottleneck

```text
REJECTED / strongly weakened
```

R1c1.

---

## H — Fixed WHAT is the dominant remaining bottleneck

```text
REJECTED for unconstrained full-text current-state reproposal
```

R1c2.

The strongest evidence is:

```text
frozen_t0_what > FULL
```

in the same trained checkpoint.

---

## H — Explicit remaining semantic evidence / explaining-away is required

```text
NEXT CAUSAL QUESTION
```

R2.

---

# 57. Final scientific statement

The strongest defensible statement from R1c2 is:

> **A mechanically healthy shared current-state WHAT re-proposal module is not sufficient to resolve IAG-SRME candidate collapse on FashionIQ. The best R1c2 FULL checkpoint reaches only 37.800 Mean Recall, below both R1a and R1c1. More decisively, disabling only dynamic WHAT inside the same trained checkpoint while preserving current-state dynamic WHERE raises retrieval to 39.260 Mean Recall at BEST and from 33.143 to 38.691 at the observed late checkpoint. Dynamic WHAT therefore has negative functional utility under the learned R1c2 solution. Representation diagnostics explain the failure: candidate WHAT vectors become more similar after re-proposal, the dominant t0→t1 semantic displacement is almost perfectly aligned across all four candidates, WHERE remains approximately 0.99997–0.99998 cosine across candidates, and token/retrieval effects remain highly parallel. With continued training, dynamic WHAT becomes more strongly common-mode, functional rank falls, late retrieval-effect magnitude attenuates, and FULL retrieval degrades while the frozen-WHAT counterfactual remains strong. R1c2 therefore converts the stale-WHAT problem into a moving-WHAT-clone shortcut rather than learning candidate-specific sequential reconsideration.**

Compactly:

\[
\boxed{
\text{Dynamic WHAT is alive but harmful.}
}
\]

\[
\boxed{
\text{state-conditioned reproposal}
\rightarrow
\text{common-mode WHAT shift}
\rightarrow
\text{clone WHERE}
\rightarrow
\text{clone effects}.
}
\]

And:

\[
\boxed{
\text{R1c2 = mechanical PASS, scientific NEGATIVE.}
}
\]

---

# 58. One-screen handoff

```text
BRANCH
exp/e2e-iag-srme-r1c2-dynamic-reproposal

FINAL AUDITED SHA
fb2e226f1f694959f158c8887567d0647bf263a8

MECHANISM
B_k = base WHAT
t0: I_0,k = B_k
t1/t2:
  inspect Z_t
  inspect Z_t - A
  re-read text
  I_t,k = B_k + ΔI_t,k
then:
  Ground(I_t,k, Z_t)

CANARY
96 / 100 successful optimizer steps
4 AMP overflow skips
all reproposal branches get gradient + parameter movement
mechanical PASS

BEST
epoch 3
FULL = 37.8001

BASELINES
R1a  = 38.7641
R1c1 = 38.7541
R1b  = 39.0119

DECISIVE CONTROL
BEST:
frozen_t0_what = 39.2599
FULL           = 37.8001
gain by disabling dynamic WHAT = +1.4599

LATE e6:
frozen_t0_what = 38.6909
FULL           = 33.1433
gain by disabling dynamic WHAT = +5.5476

=> dynamic WHAT is actively harmful

BEST WHAT
pairwise cosine:
t0 .95304
t1 .96257
t2 .96266

=> WHAT becomes MORE similar

WHAT displacement:
t0->t1 alignment = .999989
t1->t2 alignment = .97350
second-step motion is tiny

LATE WHAT
pairwise cosine:
t0 .95086
t1 .97118
t2 .97078

displacement:
t0->t1 .999996
t1->t2 .998962

=> moving-WHAT clones strengthen

WHERE
BEST:
.999936 -> .999971 -> .999971

LATE:
.999944 -> .999980 -> .999976

=> dynamic WHAT does not break WHERE contraction

FUNCTION BEST
Δq norm:
.4226 -> .2216 -> .1703

Δq cosine:
.9859 -> .9830 -> .9804

rank:
1.758 -> 1.831 -> 1.890

FUNCTION LATE
Δq norm:
.4453 -> .1320 -> .1148

Δq cosine:
.9897 -> .9890 -> .9877

rank:
1.670 -> 1.709 -> 1.744

=> late functional cloning worsens

RECURRENCE
same-parent mean MR BEST:
27.35 -> 34.97 -> 38.73

same-parent mean MR LATE:
28.25 -> 32.70 -> 36.17

=> recurrence itself still useful

POLICY BEST
mean edits = 2.636
repeat identity fraction = 81.37%
selected utility:
+.0783 -> +.0162 -> -.00348

POLICY LATE
mean edits = 2.128
best REPEAT = 35.712 > FULL 33.143
selected utility:
+.0885 -> +.0134 -> +.00468

MAIN FAILURE
dynamic WHAT learned a shared state-conditioned semantic transform,
not candidate-specific remaining actions

VERDICT
R1c2 scientific NEGATIVE

NEXT CLEAN QUESTION
R2 semantic residual / claim firewall

DO NOT STACK ON R1c2
DPP
VISReg
RDMReg
teacher
new STOP loss
RL
```

---

# 59. Freeze decision

R1c2 should now be treated as a frozen causal record.

Do not:

```text
increase re-proposal width
retune a residual scalar
add orthogonality
add DPP
change STOP
add teacher supervision
```

and continue calling the result R1c2.

The value of this experiment is that it eliminates another plausible explanation cleanly.

R1c1 showed:

```text
the grounder being stale is not enough to explain collapse.
```

R1c2 now shows:

```text
the action proposal being stale is not enough either.
```

The deeper issue is:

\[
\boxed{
\text{the objective/representation does not impose an explaining-away structure that assigns distinct unresolved evidence to distinct actions over time.}
}
\]

The next branch should test that statement directly.

Recommended conceptual branch name:

```text
exp/e2e-iag-srme-r2-semantic-residual
```

with its own exact causal contract before implementation.