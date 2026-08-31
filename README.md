# CIR IAG-SRME R1b — Dynamic Applicability Gate Negative-Result Diagnostic README

**Date:** 2026-08-31  
**Repository:** `Le-Minh-Nhut/cir`  
**Branch:** `exp/e2e-iag-srme-r1b-visual-null-confidence-gate`  
**Audited implementation HEAD before full training:** `d69d0c7eb148aa15a24a309fe1afc724c86a8860`  
**Architecture generation:** `r1b_dynamic_applicability_gate_v2`  
**Backbone:** `qihoo360/fg-clip-base`  
**Backbone revision:** `454d76372c2cf5eb48fa0d871fd0534481484d97`  
**Protocol:** `fashioniq_original`  
**Caption policy:** `ordered_and`  
**Training precision:** AMP FP16 with explicit FP32 applicability/actuator islands  
**K:** 4  
**Tmax:** 3  
**lambda_z:** 0.1  
**query_cap:** 1000.0  
**Best checkpoint:** `outputs/2026-08-31/02-36-13/best.pt`  
**Best checkpoint epoch:** 3  
**Best Mean Recall:** **39.011857**  
**Diagnostic replay Mean Recall:** **39.011857**  
**Checkpoint replay error:** `7.1e-15`  
**Mechanical verdict:** **PASS**  
**Scientific R1b verdict:** **FAIL / NEGATIVE RESULT**  
**R1c1 result:** **NEGATIVE — moving WHERE clones remained**

**Next clean experiment:** **R1c2 — dynamic current-state WHAT re-proposal**

---

# 1. Executive conclusion

R1b was designed to test a narrow causal hypothesis:

\[
\boxed{
\text{forced visual applicability / compulsory execution contributes materially to late over-edit}
}
\]

The corrected R1b v2 mechanism keeps the R1a spatial WHERE fixed and adds a dynamic state-conditioned scalar applicability gate:

\[
\pi_k = \operatorname{Entmax}_{1.5}(\ell^{vis}_k),
\]

\[
c_{t,k}
=
\sigma\!\left(
G_{\rm app}(h_{t,k})
\right),
\]

\[
\Delta Z_{t,k,n}
=
\lambda_z\,
c_{t,k}\,
S_{k,n}\,
\tanh(u_{t,k,n}).
\]

Conceptually:

```text
Entmax support π_k     = WHERE
dynamic c_t,k          = WHETHER / applicability
```

The implementation and numerical pathway are now mechanically healthy:

- applicability receives gradients;
- applicability parameters update;
- confidence remains FP32;
- the confidence scalar survives into the actual `DeltaZ`;
- candidate state and committed recurrent state preserve the continuous gate effect;
- same-parent construction remains exact;
- target is not used in forward execution;
- checkpoint replay is self-describing and exact.

However, the trained model does **not** use the gate in the intended semantic way.

The central result is:

\[
\boxed{
c_{t,k}\approx 0.982
\text{ for almost every candidate and timestep}
}
\]

with:

\[
\boxed{
p^\varnothing_{t,k}=1-c_{t,k}\approx 0.018
}
\]

and essentially no meaningful suppression after an action has already been executed.

Most importantly, the late selected transition became **more harmful**, not less harmful:

```text
R1a t2 selected target gain:  -0.00261
R1b t2 selected target gain:  -0.00682
```

Mean executed depth increased:

```text
R1a mean edits: 2.866 / 3
R1b mean edits: 2.961 / 3
```

and repeated candidate trajectories increased:

```text
R1a repeated-identity trajectories: 95.74%
R1b repeated-identity trajectories: 99.15%
```

Therefore R1b does **not** support the hypothesis that a learned scalar visual applicability gate, under the current information path and objective, is sufficient to solve late over-edit.

The correct interpretation is:

\[
\boxed{
\text{R1b is mechanically valid but scientifically negative.}
}
\]

This branch should be **frozen**, not hyperparameter-fished.

---

# 2. Why R1b existed

R1a established that the previous global cumulative query cap was a major causal bottleneck.

R1a changed only:

```yaml
model:
  query_cap: 0.5 -> 1000.0
```

and recovered recurrent retrieval-space effect magnitude.

R1a best checkpoint:

```text
Mean Recall = 38.764146
```

with:

```text
Deltaq norm:
t0 = 0.3366
t1 = 0.2724
t2 = 0.1971
```

and retention:

```text
t1/t0 = 80.9%
t2/t0 = 58.6%
```

This showed that multi-step recurrence itself could be useful once retrieval-space updates were no longer compressed.

However, R1a also exposed a new problem:

```text
selected target-relative gain:
t0 = +0.07424
t1 = +0.02488
t2 = -0.00261
```

The third edit was slightly harmful on average while STOP remained rare.

The R1b causal question was therefore deliberately small:

> If an action is no longer applicable after prior recurrent edits, can the model learn to suppress that action through a state-dependent visual applicability signal without changing WHAT, WHERE, STOP, losses, or the recurrence itself?

---

# 3. R1b experiment contract

R1b was not intended to solve every remaining collapse mode.

It preserved:

```text
K = 4
Tmax = 3
query_cap = 1000
lambda_z = 0.1
same FG-CLIP Base backbone
same FashionIQ-original protocol
same intent generation
same static spatial grounding
same grounded reader
same context fusion
same editor direction network
same readout
same scorer
same selector
same STOP formulation
same terminal loss
same marginal loss
same optimizer/training policy
same same-parent candidate construction
```

The intended scientific intervention was only:

```text
fixed WHERE
+
dynamic state-conditioned WHETHER
```

R1b explicitly did **not** introduce:

- dynamic re-grounding;
- dynamic re-proposal;
- semantic residuals;
- claim consumption;
- DPP/diversity loss;
- target-aware teacher;
- new STOP loss;
- RL;
- planning;
- new candidate ownership.

This isolation matters because the result is a causal negative result, not an ambiguous multi-mechanism failure.

---

# 4. R1b mechanism chronology

## 4.1 Original R1b v1: N+1 Entmax NULL

The first R1b formulation added a NULL coordinate directly into Entmax:

\[
P_k^{full}
=
\operatorname{Entmax}_{1.5}
(
[\ell^{vis}_{k,1:N},\ell^{null}_k]
).
\]

Then:

\[
p^\varnothing_k=P_{k,\varnothing},
\qquad
c_k=1-p^\varnothing_k.
\]

This exposed an avoidable sparse-support failure.

Observed initial statistics included:

```text
visual logit mean:  0.200777
visual logit std:   0.191942
visual logit min:  -0.305997
visual logit max:   0.849577
initial NULL logit: 0.0
mean p_null:        ~8.738e-05
```

A canary showed:

```text
step 3 NULL gradient: nonzero
step 4 NULL gradient: exactly zero
```

This did not prove permanent global death, but it demonstrated that Entmax could exclude the NULL coordinate from the active support and create exact local zero probability / zero gradient.

That mechanism was therefore superseded.

---

## 4.2 Corrected R1b v2: Entmax WHERE + sigmoid WHETHER

The final scientific formulation became:

\[
\pi_k
=
\operatorname{Entmax}_{1.5}
(\ell^{vis}_{k,1:N}),
\qquad
\sum_n\pi_{k,n}=1.
\]

Spatial support is computed once per rollout.

For timestep \(t\):

\[
O_k=\sum_n\pi_{k,n}A_n,
\]

\[
C_{t,k}=\sum_n\pi_{k,n}Z_{t,n},
\]

\[
D_{t,k}=\sum_n\pi_{k,n}(Z_{t,n}-A_n),
\]

and the existing fused context is:

\[
h_{t,k}
=
\operatorname{Context}
(I_k,O_k,C_{t,k},D_{t,k}).
\]

Applicability:

\[
r_{t,k}
=
W_{\rm app}\operatorname{LN}(h_{t,k})
+b_{\rm app},
\]

\[
c_{t,k}=\sigma(r_{t,k}),
\]

\[
p^\varnothing_{t,k}=1-c_{t,k}.
\]

Execution:

\[
S_{k,n}
=
\frac{\pi_{k,n}}
{\max_j\pi_{k,j}+\epsilon},
\]

\[
\boxed{
\Delta Z_{t,k,n}
=
\lambda_z\,
c_{t,k}\,
S_{k,n}\,
\tanh(u_{t,k,n})
}
\]

Confidence is applied exactly once.

---

# 5. Initialization

The gate starts nearly equivalent to R1a:

\[
c_0=0.98,
\qquad
p^\varnothing_0=0.02.
\]

The final projection is initialized as:

\[
W_{\rm app}=0,
\]

\[
b_{\rm app}
=
\operatorname{logit}(0.98)
\approx3.8918203.
\]

Therefore all candidates initially have:

\[
c_{t,k}=0.98
\]

before learning.

The purpose was to make R1b initially close to R1a while leaving a usable sigmoid gradient.

---

# 6. Mixed-precision correction before full training

The first corrected-v2 GPU canary exposed a second numerical issue.

The gate parameters received gradients and updated, but the forward confidence was repeatedly observed at the same FP16 representable value:

```text
0.97998046875
```

because the code effectively performed:

```python
confidence = torch.sigmoid(logits.float()).to(logits.dtype)
```

under FP16 autocast.

Around 0.98, adjacent FP16 values are roughly:

```text
0.9794921875
0.97998046875
0.98046875
```

so sub-`4.88e-4` changes could disappear.

This was corrected before full training.

Final precision path:

```text
FP16/BF16 contexts allowed
        ↓
FP32 LayerNorm
        ↓
FP32 applicability projection
        ↓
FP32 sigmoid
        ↓
FP32 confidence / p_null
        ↓
FP32 confidence × edit actuator
        ↓
FP32 DeltaZ
        ↓
FP32 candidate state
        ↓
FP32 committed recurrent state
```

Post-fix CUDA canary:

```text
attempted steps: 100
successful optimizer steps: 94
nonzero applicability gradient steps: 94 / 94
applicability gradient fraction: 1.0
finite: true
collapse flags: all false
```

The gate then showed true continuous variation below one FP16 quantum and:

```text
confidence_to_delta_scale_error = 0.0
```

Thus the full-training negative result below is **not** explainable by the earlier Entmax-gradient bug or FP16 quantization bug.

---

# 7. Full-training result

Training command:

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r1b_visual_null \
  experiment=iag_srme_r1b_visual_null \
  protocol=fashioniq_original
```

Visible early training trajectory:

```text
epoch 1: total=1.3729  MR=33.452
epoch 2: total=0.6495  MR=38.146
epoch 3: total=0.3812  MR=39.012  <- best checkpoint
epoch 4: total=0.2436  MR=37.573
epoch 5: total=0.1798  MR=37.458
```

The main pattern is:

```text
training objective continues decreasing
while
validation retrieval peaks early and then falls
```

This is consistent with overfitting/objective misalignment after the best checkpoint.

The diagnostic in this README uses the saved **best epoch-3 checkpoint**, not `last.pt`.

---

# 8. Checkpoint provenance and replay

Best checkpoint:

```text
outputs/2026-08-31/02-36-13/best.pt
```

Saved metric:

```text
39.01185741027196
```

Diagnostic replay:

```text
39.011857410271965
```

Absolute replay error:

```text
7.105427357601002e-15
```

Replay guard:

```text
fully_self_describing_model_config = true
architecture_generation = r1b_dynamic_applicability_gate_v2
dynamic_applicability_enabled = true
initial_applicability = 0.98
query_cap = 1000
trusted_r1b_replay = true
```

Therefore the report is using the intended R1b v2 checkpoint and configuration.

---

# 9. Headline retrieval comparison: R1a vs R1b

| Metric | R1a | R1b | Delta |
|---|---:|---:|---:|
| FULL Mean Recall | 38.7641 | **39.0119** | +0.2477 |
| MEAN candidate | 38.8160 | **39.0706** | +0.2546 |
| Best REPEAT | **39.2232** | 39.1267 | -0.0965 |
| Reference only | 14.5216 | 14.4726 | -0.0490 |
| Best SINGLE | 24.7096 | **25.2447** | +0.5351 |

R1b slightly improves the headline best-checkpoint FULL MR relative to R1a.

That small improvement is **not** sufficient to declare the mechanism successful because the intended scientific claim concerns learned applicability and late unnecessary execution.

In fact:

```text
MEAN > FULL
best REPEAT > FULL
```

still holds numerically, although only by tiny margins.

R1b therefore does not create strong evidence that dynamic candidate identity or learned applicability contributes unique retrieval value.

---

# 10. Same-parent depth remains useful

R1b same-parent candidate retrieval:

```text
t0 mean candidate MR = 25.1196
t1 mean candidate MR = 34.7647
t2 mean candidate MR = 38.9373
```

Offline same-parent oracle:

```text
t0 = 25.7276
t1 = 35.2889
t2 = 39.4547
```

This preserves the important R1a conclusion:

\[
\boxed{
\text{recurrent depth itself remains useful}
}
\]

The problem is not that the later state is useless.

The problem is that the action system does not reliably know when a particular action is no longer useful.

---

# 11. Retrieval-space effect magnitude remains healthy

R1b:

```text
mean ||Deltaq||:
t0 = 0.349246
t1 = 0.278385
t2 = 0.198699
```

Retention:

```text
t1/t0 = 0.7971
t2/t0 = 0.5689
t2/t1 = 0.7138
```

R1a:

```text
t1/t0 ≈ 0.809
t2/t0 ≈ 0.586
```

Therefore R1b preserves the main R1a repair.

There is no return of the R0 late-query attenuation failure.

---

# 12. Token-space edit magnitude also remains active

R1b:

```text
mean ||DeltaZ||:
t0 = 1.59434
t1 = 1.59048
t2 = 1.59051
```

All candidate effects are numerically active:

```text
active candidate fraction = 1.0
dead candidate fraction   = 0.0
```

Again, the failure is not “late actions are numerically dead.”

The actions remain strong.

---

# 13. The core scientific failure: late selected utility becomes worse

R1b global selected target-relative improvement:

```text
t0 = +0.0715569
t1 = +0.0199435
t2 = -0.0068204
```

R1a baseline:

```text
t0 = +0.07424
t1 = +0.02488
t2 = -0.00261
```

Comparison:

| Timestep | R1a | R1b | Direction |
|---|---:|---:|---|
| t0 | +0.07424 | +0.07156 | slightly worse |
| t1 | +0.02488 | +0.01994 | worse |
| t2 | -0.00261 | **-0.00682** | clearly worse |

The R1b success criterion required:

```text
t2 target gain moves toward zero or positive
```

Instead:

\[
\boxed{
-0.00261
\rightarrow
-0.00682
}
\]

Therefore the main R1b over-edit hypothesis fails empirically at the best checkpoint.

---

# 14. Applicability gate is alive but functionally near-always-ON

Global visual confidence:

```text
count  = 71,996
mean   = 0.9822428
median = 0.9822109
min    = 0.9791229
max    = 0.9858897
std    = 0.0011973
```

Equivalent dynamic NULL probability:

```text
mean   = 0.0177572
median = 0.0177891
min    = 0.0141103
max    = 0.0208771
std    = 0.0011973
```

The gate is **not constant** and is **not numerically dead**.

However, every observed candidate remains in a very narrow high-confidence regime.

Fractions:

```text
p_null > 0.10 : 0
p_null > 0.25 : 0
p_null > 0.50 : 0
p_null > 0.80 : 0
```

Operationally, the trained gate behaves approximately like:

```text
candidate applicable?  -> YES, ~98%
candidate applicable?  -> YES, ~98%
candidate applicable?  -> YES, ~98%
...
```

The model therefore learned a state-conditioned scalar but did not learn meaningful abstention/suppression.

---

# 15. Applicability barely changes after execution

The most important temporal diagnostic asks:

> After an action has actually been executed, does the confidence for the same action materially decrease?

Across `11,983` same-action before/after execution observations:

```text
mean confidence change
= -7.2574e-06
```

Repeated selected-action subset (`11,257` observations):

```text
mean confidence change
= -7.2413e-06
```

Typical confidence itself is approximately:

```text
~0.982
```

Therefore the relative behavioral change is tiny.

The system effectively behaves as:

```text
before execution:
c ≈ 0.98224

after execution:
c ≈ 0.98223
```

This is dynamic in a strict numerical sense, but nearly static in functional execution strength.

That is the strongest direct evidence that R1b failed its intended semantic role.

---

# 16. Applicability does not track offline candidate utility

Offline diagnostic target utility is:

\[
U_{t,k}
=
\cos(\hat q_{t+1,k},y)
-
\cos(q_t,y).
\]

This target-aware signal is used **only after target-free candidate construction** for diagnostics and does not enter forward execution.

Measured correlation:

\[
\boxed{
\rho(
p^\varnothing_{t,k},
U_{t,k}
)
=
-0.03173
}
\]

which is near zero.

Thus higher predicted NULL probability does not meaningfully identify lower-utility actions.

This is crucial because the candidate utility distribution changes strongly with depth, while the applicability output barely reacts in the intended direction.

---

# 17. Utility clearly deteriorates with depth

All-candidate offline target utility shows the expected late deterioration.

At early depth, candidates are mostly useful.

By t2, negative utility is common and mean utility becomes negative.

Selected-path behavior also shows:

```text
t0 strongly positive
t1 weakly positive
t2 negative
```

Yet all applicability values remain near:

```text
c ≈ 0.98
p_null ≈ 0.02
```

This rules out the explanation:

```text
"the model did not need to suppress actions because all late actions were still useful"
```

The late action population does contain substantial harmful behavior.

The gate simply does not identify/suppress it.

---

# 18. STOP becomes even less aggressive than R1a

R1b new STOP hazard:

```text
t0 = 0.116%
t1 = 0.582%
t2 = 2.377%
```

Counts:

```text
new STOP:
t0 = 7
t1 = 35
t2 = 142
```

R1a approximate new STOP hazard:

```text
t0 ≈ 1.65%
t1 ≈ 2.18%
t2 ≈ 4.35%
```

Mean executed edits:

```text
R1a = 2.866 / 3
R1b = 2.961 / 3
```

So R1b does not cause more adaptive early stopping.

It produces **more maximum-depth execution**.

This is directly opposite the hoped-for late-suppression behavior.

---

# 19. Repeated candidate identity becomes more dominant

R1b:

```text
queries: 6016
queries with repeated candidate identity: 5965
fraction: 99.152%
```

R1a:

```text
~95.745%
```

Thus:

\[
\boxed{
95.7\%
\rightarrow
99.15\%
}
\]

The R1b gate does not break the repeated-action program.

It makes the actual policy even more dominated by repeated candidate identity.

This does not mean the scalar gate causes candidate cloning; R1b was not designed to solve specialization.

But it confirms that the mechanism did not indirectly create a healthier recurrent action process.

---

# 20. Candidate selection is not a single-index monopoly, but redundancy remains

R1b candidate distribution conditional on non-STOP edit:

```text
candidate 0 = 17.29%
candidate 1 = 14.38%
candidate 2 = 32.04%
candidate 3 = 36.29%
```

Maximum candidate share:

```text
36.29%
```

So there is no simple 95%-style candidate-index monopoly.

However, candidate identity diversity is misleading because the candidate functions remain highly similar.

A model can distribute selection over several indices while those indices implement nearly the same edit.

Therefore effect-space diagnostics remain the stronger evidence.

---

# 21. WHAT remains highly correlated

R1b intent cosine off-diagonal mean:

```text
0.953855
```

R1a:

```text
~0.9499
```

This is not a meaningful improvement.

The proposal bank still reads the text into highly correlated latent intents.

R1b was not expected to fix this.

---

# 22. WHERE remains essentially cloned

R1b spatial support:

```text
pairwise support cosine = 0.999853
pairwise overlap        = 0.994781
effective support size  = 13.29 / 196
support entropy         = 2.5386
```

R1a:

```text
support cosine  = 0.999842
support overlap = 0.995108
effective size  ≈ 10.57 / 196
```

The difference is negligible for specialization.

Thus:

\[
\boxed{
\text{WHERE remains nearly identical across candidates.}
}
\]

This is expected because corrected R1b deliberately keeps WHERE static and R1a-equivalent.

---

# 23. DeltaZ remains highly redundant

R1b pairwise `DeltaZ` cosine:

```text
t0 = 0.98150
t1 = 0.98142
t2 = 0.98144
```

DeltaZ effective rank:

```text
t0 = 1.868
t1 = 1.869
t2 = 1.869
```

R1a was also approximately:

```text
DeltaZ cosine ≈ 0.982
rank          ≈ 1.85
```

So the actual token-space interventions remain clone-like.

---

# 24. Deltaq remains highly redundant

R1b pairwise retrieval-effect cosine:

```text
t0 = 0.98331
t1 = 0.98111
t2 = 0.97518
```

Functional effective rank:

```text
t0 = 1.824
t1 = 1.871
t2 = 1.981
```

This is slightly less redundant at later depth, but still very far from four clearly distinct functional directions.

The conservative diagnostic flags correctly report high Deltaq similarity at all three timesteps.

---

# 25. FULL vs MEAN vs REPEAT remains an important warning

R1b:

```text
FULL        = 39.0119
MEAN        = 39.0706
REPEAT-0    = 38.9277
REPEAT-1    = 39.0398
REPEAT-2    = 39.0437
REPEAT-3    = 39.1267
```

Best REPEAT exceeds FULL by only ~0.115 points, so it does not trigger the conservative `+2 MR` failure threshold.

But the scientific interpretation is still important:

```text
MEAN ≈ FULL
REPEAT ≈ FULL
```

Dynamic candidate identity provides little unique advantage.

This is consistent with the clone-like WHAT/WHERE/effect diagnostics.

---

# 26. R1a → R1b comparison table

| Diagnostic | R1a | R1b | Interpretation |
|---|---:|---:|---|
| FULL MR | 38.7641 | 39.0119 | tiny retrieval gain |
| MEAN MR | 38.8160 | 39.0706 | still ≈ FULL |
| best REPEAT MR | 39.2232 | 39.1267 | still ≈ FULL |
| t0 Deltaq norm | 0.3366 | 0.3492 | healthy |
| t1 Deltaq norm | 0.2724 | 0.2784 | healthy |
| t2 Deltaq norm | 0.1971 | 0.1987 | healthy |
| t1/t0 retention | 0.809 | 0.797 | preserved |
| t2/t0 retention | 0.586 | 0.569 | preserved |
| t0 Deltaq cosine | 0.9833 | 0.9833 | unchanged |
| t1 Deltaq cosine | 0.9817 | 0.9811 | tiny change |
| t2 Deltaq cosine | 0.9765 | 0.9752 | tiny change |
| support cosine | 0.999842 | 0.999853 | unchanged |
| support overlap | 0.995108 | 0.994781 | unchanged |
| intent cosine | 0.9499 | 0.9539 | slightly worse |
| selected gain t0 | +0.07424 | +0.07156 | worse |
| selected gain t1 | +0.02488 | +0.01994 | worse |
| selected gain t2 | -0.00261 | **-0.00682** | worse |
| mean edits | 2.866 | **2.961** | worse |
| repeated identity | 95.74% | **99.15%** | worse |
| late STOP hazard | ~4.35% | **2.38%** | less adaptive |

This table is the shortest complete summary of the R1b outcome.

---

# 27. R1b kill criteria and outcome

The pre-registered R1b kill criteria included:

## Criterion 1 — NULL collapses near zero everywhere

Observed:

```text
mean p_null = 0.01776
max p_null  = 0.02088
all p_null < 0.10
```

This is not exact numerical zero, but functionally the mechanism remains close to always-on.

**Status:** effectively triggered in spirit.

---

## Criterion 2 — NULL collapses near one everywhere

Not observed.

**Status:** PASS.

---

## Criterion 3 — useful early edits are strongly suppressed

Early utility remains positive, although slightly lower than R1a.

**Status:** no catastrophic suppression.

---

## Criterion 4 — Deltaq retention collapses again

Not observed.

**Status:** PASS.

---

## Criterion 5 — late negative gain remains and NULL is not informative

Observed strongly:

```text
R1b t2 selected gain = -0.00682
p_null vs utility Pearson = -0.0317
```

**Status:** FAIL / kill criterion triggered.

---

## Criterion 6 — MR improves but NULL has no meaningful relation to unnecessary edits

Observed:

```text
MR: +0.248 over R1a
but
semantic applicability evidence remains absent
```

**Status:** FAIL / kill criterion triggered.

---

# 28. Strongest causal interpretation

The R1b result does **not** prove that visual applicability can never help CIR.

It supports the narrower conclusion:

> Under the current IAG-SRME information path and terminal+marginal objective, adding a learned dynamic scalar applicability gate on top of fixed R1a Entmax WHERE is not sufficient to learn meaningful late-action suppression.

The data reject the simple hypothesis:

```text
forced normalized spatial grounding
        ↓
main cause of late over-edit
```

as the dominant explanation.

At minimum:

```text
forced WHERE alone
```

is not sufficient to explain the failure.

The model can minimize the current objective while keeping:

\[
c_{t,k}\approx1
\]

almost everywhere.

---

# 29. Why the gate can remain near 1 under the current objective

The gate is trainable and receives gradient, but the current objective does not explicitly require:

```text
"after executing action k, reduce c_t+1,k"
```

or:

```text
"if candidate utility is negative, increase p_null"
```

The terminal loss only needs the final retrieval query to be useful.

The marginal target utility is detached and primarily supervises the consequence scorer rather than directly defining a semantic applicability target for the gate.

Because candidate actions are already highly redundant, the system can continue to improve retrieval through repeated recurrent movement while leaving the scalar gate near one.

Therefore the learned optimum can be:

```text
keep most actions executable
let recurrence/readout do the work
do not learn semantic abstention
```

This is a scientific/objective-information result, not a numerical bug.

---

# 30. Why not tune the R1b bias or add a gate loss now

Do **not** respond to this negative result by immediately trying:

```text
initial_applicability = 0.95
initial_applicability = 0.90
initial_applicability = 0.80
stronger sigmoid temperature
manual timestep decay
explicit p_null regularization
gate entropy loss
gate sparsity loss
late gate penalty
```

Doing so would change the question from:

```text
"does the current task objective naturally learn useful applicability?"
```

into:

```text
"can we force a scalar to turn off?"
```

A lower scalar is not evidence of correct semantic applicability.

Likewise, adding a target-derived gate loss immediately would make R1b a different experiment.

The current branch has already answered its intended causal question.

Freeze it.

---

# 31. Why this is not a failure of multi-step recurrence

The same-parent candidate retrieval continues to increase strongly with depth:

```text
25.12 -> 34.76 -> 38.94
```

and Deltaq remains large at t2:

```text
0.349 -> 0.278 -> 0.199
```

Therefore the evidence does **not** support:

```text
"multi-step was a bad idea"
```

The better statement remains:

\[
\boxed{
\text{multi-step state evolution is useful, but recurrent action semantics remain poorly conditioned on what is still missing.}
}
\]

---

# 32. Why R1c1 is the correct next experiment

R1b kept WHERE fixed:

\[
\pi_k
=
\operatorname{Ground}(I_k,A)
\]

once before recurrence.

Only a scalar execution confidence saw the changing recurrent context.

But the current state really changes:

```text
context changes
D_t changes
G_t changes
candidate query changes
scores change
```

while spatial support itself remains static.

R1c1 therefore tests the next smallest structural hypothesis:

\[
\boxed{
\text{static support reuse prevents the action system from reacting spatially to the edited current state}
}
\]

Change:

\[
\pi_k
=
\operatorname{Ground}(I_k,A)
\]

to:

\[
\boxed{
\pi_{t,k}
=
\operatorname{Ground}(I_k,A,Z_t)
}
\]

or the exact repo-consistent current-state grounding equivalent.

The key isolation should be:

```text
R1c1:
fixed WHAT
dynamic WHERE
```

Do not simultaneously introduce dynamic WHAT.

---

# 33. R1c1 causal contract

R1c1 should preserve:

```text
K = 4
Tmax = 3
query_cap = 1000
same backbone
same objective
same optimizer
same readout
same scorer
same selector
same STOP
same editor form
same candidate identity bank
same protocol
```

Only change:

```text
static support
->
support recomputed from the current recurrent state at every timestep
```

The applicability gate can be either frozen/removed according to the exact causal design, but the experiment must not silently combine multiple new mechanisms.

If retaining R1b gate for continuity, explicitly label R1c1 as:

```text
R1a + R1b gate + dynamic WHERE
```

and acknowledge that the effect is not isolated against R1a alone.

The cleaner scientific comparison is preferably designed so dynamic WHERE is the only newly active mechanism relative to a matched parent.

---

# 34. R1c1 mandatory diagnostics

Add temporal WHERE diagnostics:

```text
support cosine P_t,k vs P_t+1,k
support overlap P_t,k vs P_t+1,k
support effective-size change
support entropy change
support center-of-mass / token-rank shift
support change after executing same candidate
support change after executing different candidate
```

Preserve:

```text
FULL
REFERENCE
SINGLE
REPEAT
MEAN
same-parent candidate retrieval
offline candidate oracle

intent cosine

between-candidate support cosine
between-candidate support overlap

DeltaZ norm/cosine/rank
Deltaq norm/cosine/rank
late retention

selected target-relative utility
STOP hazard
mean executed depth
repeated candidate identity fraction
```

Primary R1c1 success evidence:

```text
support becomes genuinely state-sensitive
and
late selected utility improves
and/or
REPEAT equivalence weakens
without destroying R1a Deltaq retention/retrieval
```

---

# 35. What not to add in R1c1

Do not add:

- DPP;
- semantic residual;
- token ownership;
- teacher supervision;
- target-aware grounding in forward;
- planning;
- new STOP loss;
- RL;
- action re-proposal;
- local text losses;
- new candidate balancing.

If dynamic WHERE fails, then move to R1c2 dynamic WHAT/re-proposal.

Do not skip directly to loss soup.

---

# 36. Reproduction commands

## Full R1b training

```bash
CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/train.py \
  dataset.root=data/FashionIQ \
  model=iag_srme_r1b_visual_null \
  experiment=iag_srme_r1b_visual_null \
  protocol=fashioniq_original
```

## Best-checkpoint diagnostic

```bash
RUN=outputs/2026-08-31/02-36-13

CUBLAS_WORKSPACE_CONFIG=:4096:8 \
python src/diagnose_iag_srme_checkpoint.py \
  --checkpoint "$RUN/best.pt" \
  --dataset-root data/FashionIQ \
  --protocol fashioniq_original \
  --batch-size 32 \
  --gallery-batch-size 128 \
  --output reports/r1b_dynamic_applicability_best.json
```

Expected replay:

```text
checkpoint epoch = 3
Mean Recall      = 39.011857
query_cap        = 1000.0
architecture     = r1b_dynamic_applicability_gate_v2
```

---

# 37. One-screen handoff

```text
BRANCH
exp/e2e-iag-srme-r1b-visual-null-confidence-gate

IMPLEMENTATION HEAD BEFORE TRAINING
d69d0c7eb148aa15a24a309fe1afc724c86a8860

ARCHITECTURE
fixed Entmax WHERE
+
dynamic sigmoid WHETHER

NUMERICAL STATUS
Entmax-NULL sparse gradient issue: fixed
FP16 confidence quantization issue: fixed
FP32 gate/actuator/state pathway: verified
GPU canary: pass

BEST CHECKPOINT
outputs/2026-08-31/02-36-13/best.pt
epoch 3

RETRIEVAL
R1a FULL = 38.7641
R1b FULL = 39.0119
R1b MEAN = 39.0706
R1b best REPEAT = 39.1267

DEPTH
Deltaq = 0.3492 -> 0.2784 -> 0.1987
retention t1/t0 = 79.7%
retention t2/t0 = 56.9%

WHERE
support cosine = 0.999853
support overlap = 0.994781
static across recurrence

FUNCTION
DeltaZ cosine ≈ 0.9814
Deltaq cosine = 0.9833 -> 0.9811 -> 0.9752
functional rank = 1.824 -> 1.871 -> 1.981

APPLICABILITY
mean confidence = 0.982243
mean p_null = 0.017757
max p_null = 0.020877
fraction p_null > .10 = 0

p_null vs target utility Pearson = -0.0317

same-action confidence change after execution:
mean = -7.26e-06

selected repeated-action confidence change:
mean = -7.24e-06

TARGET-RELATIVE SELECTED GAIN
R1a: +0.07424 -> +0.02488 -> -0.00261
R1b: +0.07156 -> +0.01994 -> -0.00682

POLICY
mean edits = 2.961 / 3
repeated candidate identity = 99.15%
new STOP hazard =
0.116% -> 0.582% -> 2.377%

SCIENTIFIC VERDICT
R1b FAIL / negative result

INTERPRETATION
The gate is trainable and numerically alive,
but the current objective/information path learns
an almost-always-ON applicability function.

It does not suppress already-executed or harmful late actions.

NEXT
R1c1 dynamic current-state re-grounding:
fixed WHAT + dynamic WHERE
```

---

# 38. Final scientific statement

The strongest defensible conclusion from R1b is:

> **A numerically healthy dynamic sigmoid applicability gate placed after the recurrent grounded context does not, under the current IAG-SRME training objective, learn meaningful abstention or late-action suppression. The gate remains near 0.982 confidence for essentially all candidates, its confidence changes by only about \(7\times10^{-6}\) after execution, and its NULL probability is nearly uncorrelated with offline target-relative candidate utility. Despite a small headline retrieval increase from R1a 38.764 to R1b 39.012 Mean Recall, the intended failure mode worsens: the third selected edit becomes more harmful on average, mean executed depth rises to 2.961/3, and repeated candidate identity rises to 99.15%. At the same time, recurrent retrieval-space effects remain healthy, confirming that multi-step depth is still useful. The next clean causal question is therefore not a stronger scalar gate, but whether recomputing spatial WHERE from the current recurrent state can make candidate execution genuinely state-sensitive.**

---

# 39. Branch status

```text
R0   diagnostic audit                         DONE
R1a  remove global query cap                  PASS
R1b  dynamic applicability / visual NULL      NEGATIVE RESULT
R1c1 dynamic current-state re-grounding       NEGATIVE RESULT
R1c2 dynamic current-state re-proposal        IMPLEMENTATION READY / RESULT PENDING
R2   semantic residual / claim firewall       IMPLEMENTATION READY / RESULT PENDING
R3   quality-gated functional DPP             CONDITIONAL
R4   target-privileged grounding teacher      CONDITIONAL
R5   planning / STOP refinement               CONDITIONAL
```

Do not reinterpret R1b as an implementation failure.

The value of R1b is that it eliminates a plausible causal hypothesis cleanly:

\[
\boxed{
\text{scalar applicability alone is not enough.}
}
\]

The remaining research direction should now move from:

```text
"how much should this fixed action execute?"
```

toward:

```text
"given the current edited state, where is this action still grounded,
and what residual action is actually still missing?"
```

## R1c2 dynamic current-state WHAT reproposal

R1c2 is isolated on `exp/e2e-iag-srme-r1c2-dynamic-reproposal`. It preserves R1c1 dynamic WHERE,
adds a shared zero-initialized state-conditioned WHAT residual at t1/t2, and keeps R1b
applicability disabled. See
`doc/CIR_IAG_SRME_R1C2_DYNAMIC_CURRENT_STATE_REPROPOSAL_EXPERIMENT_SPEC_2026-08-31.md`.

## R2 semantic residual / claim firewall

R2 is isolated on `exp/e2e-iag-srme-r2-semantic-residual`. It disables the harmful
R1c2 unrestricted full-text reproposal, preserves R1a query-cap and R1c1 dynamic WHERE,
and adds an explicit token-level remaining-evidence state with selected-only claim
consumption. See
`doc/CIR_IAG_SRME_R2_SEMANTIC_RESIDUAL_CLAIM_FIREWALL_EXPERIMENT_SPEC_2026-08-31.md`.
