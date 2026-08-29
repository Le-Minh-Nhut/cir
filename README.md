# CIR IAG-SRME — Core Baseline Diagnostic
## Epoch 3 Best vs Epoch 5 Last

**Date:** 2026-08-30  
**Method:** IAG-SRME, FG-CLIP Base, K=4, Tmax=3, d=256  
**Training objective:** `L_terminal + 0.5 L_marginal` (`objective=core`)  
**Dataset / protocol:** FashionIQ val, `fashioniq_original`, `ordered_and`  
**Best checkpoint:** epoch 3, Mean Recall = 27.258  
**Last checkpoint:** epoch 5, Mean Recall = 26.697

---

# 1. Executive diagnosis

The run is **not failing because of numerical instability, a single-candidate monopoly, or immediate STOP collapse**.

The strongest structural failure is:

> **The four candidates have effectively identical visual grounding.**

At epoch 3:

- pairwise support cosine = **0.999795**
- pairwise support overlap = **0.995188**

At epoch 5:

- pairwise support cosine = **0.999821**
- pairwise support overlap = **0.995809**

So all four learnable edit queries are essentially looking at the **same visual region distribution**.

This is already present at the best checkpoint and therefore is not merely a consequence of over-training from epoch 3 to epoch 5. It is a structural failure mode of the core objective.

The second major failure is:

> **Candidate edits become increasingly similar at the first recurrent step, while later recurrent edits have rapidly diminishing effect on the final retrieval query.**

At epoch 3:

| timestep | mean `||Δq||` | pairwise Δq cosine | functional rank |
|---|---:|---:|---:|
| t0 | 0.3581 | 0.9897 | 1.663 |
| t1 | 0.0890 | 0.9444 | 2.339 |
| t2 | 0.0216 | 0.6105 | 3.440 |

At epoch 5:

| timestep | mean `||Δq||` | pairwise Δq cosine | functional rank |
|---|---:|---:|---:|
| t0 | 0.3665 | 0.9923 | 1.595 |
| t1 | 0.0825 | 0.9492 | 2.297 |
| t2 | 0.0190 | 0.6056 | 3.440 |

The t2 query effect is only about **6.0%** of t0 at epoch 3 and **5.2%** at epoch 5.

Thus the recurrent state is changing, but the later edits are becoming weak in the final retrieval space.

The third important observation is:

> **The scorer progressively learns to STOP much earlier between epoch 3 and epoch 5.**

| metric | epoch 3 | epoch 5 | delta |
|---|---:|---:|---:|
| mean executed edits | 2.682 | 2.195 | -0.486 |
| STOP hazard t0 | 0.08% | 0.08% | 0.00% |
| STOP hazard t1 | 3.38% | 9.83% | 6.45% |
| STOP hazard t2 | 25.74% | 67.25% | 41.51% |

This STOP behavior is probably **partly a rational response** to the weak late-step effects. It should not yet be treated as the root cause.

---

# 2. Retrieval degradation from epoch 3 to epoch 5

| metric | epoch 3 | epoch 5 | delta |
|---|---:|---:|---:|
| R@10 | 18.154 | 17.847 | -0.307 |
| R@50 | 36.363 | 35.547 | -0.816 |
| Mean Recall | 27.258 | 26.697 | -0.562 |

The drop is real but modest: **0.562 Mean Recall points**.

More important than the absolute drop is *where* it comes from.

## 2.1 Reference representation does not degrade

Reference-only retrieval actually improves:

- epoch 3: **14.438**
- epoch 5: **14.843**
- delta: **+0.406**

Therefore the degradation is not simply “FG-CLIP became worse everywhere”.

A plausible interpretation is:

> Continued fine-tuning improves the global reference/image representation while the composed edit trajectory becomes worse.

This should be treated as an inference, not a proven causal statement.

## 2.2 SINGLE edits remain relatively stable

Best SINGLE:

- epoch 3: **24.527**
- epoch 5: **24.395**

The degradation is much smaller than FULL.

This indicates that the root-state edit itself has not catastrophically broken.

## 2.3 Deeper/repeated behavior degrades more

Best REPEAT:

- epoch 3: **27.441**
- epoch 5: **26.915**

FULL:

- epoch 3: **27.258**
- epoch 5: **26.697**

The candidate effects at t1/t2 also become smaller in query space.

This points toward a **recurrent-composition / edit-trajectory issue**, not a simple first-edit failure.

---

# 3. Primary structural failure: WHERE collapse

The strongest diagnostic signal is the visual support similarity.

## Epoch 3

- support cosine: **0.999795**
- support overlap: **0.995188**
- support fraction: **7.37%**
- effective support size: **13.95 / 196 tokens**

## Epoch 5

- support cosine: **0.999821**
- support overlap: **0.995809**
- support fraction: **8.32%**
- effective support size: **15.78 / 196 tokens**

This is not “all candidates attend to nearby but different regions”.

Numerically, the support vectors are almost identical.

The failure flag `grounding_clone=true` is therefore well-supported.

### Important nuance

Grounding is **not over-sparse**.

It actually becomes broader between epoch 3 and epoch 5:

- support fraction: 7.37% → 8.32%
- entropy: 2.603 → 2.734
- effective size: 13.95 → 15.78

Therefore the problem is not “entmax became too sharp”.

The problem is:

> **all four candidates use almost the same sparse-ish support.**

---

# 4. WHAT / context / edit similarity

The four candidate pathways are already highly correlated before/through editing.

At epoch 3, the global specialization matrices show approximately:

- intent pairwise cosine: about **0.94–0.95**
- context pairwise cosine: about **0.95–0.96**
- delta-Z pairwise cosine: about **0.986–0.989**
- support pairwise cosine: about **0.9997–0.9999**

At epoch 5:

- intent pairwise cosine remains about **0.94–0.95**
- context cosine rises slightly to around **0.95–0.96**
- delta-Z cosine rises to around **0.989–0.991**
- support cosine remains essentially **1.0**

Thus the network has four identities, but most of the visual/edit computation is highly redundant.

The final `delta_q` vectors are less identical than `delta_z`, but that does not rescue the underlying token-level specialization.

---

# 5. Recurrent attenuation

One particularly important pattern is the rapid reduction in retrieval-space edit strength.

## Epoch 3

```text
t0 ||Δq|| ≈ 0.3581
t1 ||Δq|| ≈ 0.0890
t2 ||Δq|| ≈ 0.0216
```

## Epoch 5

```text
t0 ||Δq|| ≈ 0.3665
t1 ||Δq|| ≈ 0.0825
t2 ||Δq|| ≈ 0.0190
```

Meanwhile `||ΔZ||` actually increases:

- t0: 2.067 → 2.262
- t1: 2.070 → 2.264
- t2: 2.085 → 2.301

So the editor is not becoming inactive in token space.

Instead:

> **large token-state edits produce progressively smaller retrieval-query changes at later timesteps.**

This is a critical distinction.

Possible mechanisms include saturation/cancellation in the readout, repeated edits along similar directions, normalization/capping effects, or the state reaching a region where additional local changes have weak retrieval-space leverage.

The current diagnostics do not isolate which of those mechanisms is causal.

---

# 6. STOP drift is secondary, not yet the root cause

Epoch 3:

```text
STOP hazard: t0=0.08%,
             t1=3.38%,
             t2=25.74%

mean executed edits = 2.682
```

Epoch 5:

```text
STOP hazard: t0=0.08%,
             t1=9.83%,
             t2=67.25%

mean executed edits = 2.195
```

STOP clearly becomes much more aggressive.

However, forced REPEAT only slightly outperforms FULL:

- epoch 3: best REPEAT / FULL = **1.0067**
- epoch 5: best REPEAT / FULL = **1.0082**

Therefore there is not a large amount of hidden retrieval performance being destroyed solely by the STOP policy.

The better interpretation is:

> Later candidate consequences are becoming weak / redundant, and the scorer learns that stopping is often nearly as good.

---

# 7. Candidate selection is not fully collapsed, but is drifting

Conditional candidate distribution among executed edits:

## Epoch 3

```text
candidate 0: 13.99%
candidate 1: 28.07%
candidate 2: 31.51%
candidate 3: 26.43%
```

## Epoch 5

```text
candidate 0: 23.19%
candidate 1: 40.82%
candidate 2: 18.07%
candidate 3: 17.92%
```

There is no hard monopoly, but candidate 1 rises from **28.07%** to **40.82%**.

This should be watched in later runs.

---

# 8. Why the core objective permits this failure

Current core objective:

```math
L_core = L_terminal + 0.5 L_marginal
```

`L_terminal` asks the final composed query to retrieve the target.

`L_marginal` asks the scorer to match target-derived marginal utility of candidate consequences.

Neither objective directly requires:

- different candidates to represent different semantic claims;
- different candidates to ground to different or complementary visual regions;
- different candidate token edits to have distinct functional effects;
- the four candidate identities to form a meaningful decomposition.

If all candidates discover nearly the same useful edit direction, both losses can still be optimized.

This explains why:

```text
support cosine ≈ 1.0
delta-Z cosine ≈ 0.99
MEAN ≈ FULL
REPEAT ≈ FULL
```

can coexist with a decreasing training loss.

The core objective provides **utility supervision**, but almost no explicit **symmetry-breaking supervision**.

---

# 9. Causal hypothesis to carry into the next experiments

The most plausible current chain is:

```text
4 text queries
    ↓
highly similar intent representations
    ↓
almost identical anchor-grounding distributions
    ↓
almost identical grounded evidence
    ↓
highly similar contexts
    ↓
almost parallel token-level edits
    ↓
repeated same-direction edits
    ↓
later query-space effects attenuate strongly
    ↓
marginal utility of later edits becomes small
    ↓
scorer increasingly chooses STOP
```

This chain is a **diagnostic hypothesis**, not yet a proven causal graph.

The next loss ablations should be designed to break specific links and see which metrics move.

---

# 10. Recommended loss ablation order

The following runs should initially use only **5 epochs**, because the baseline already exposes its trend by epoch 3–5.

Keep all other variables identical:

- same seed;
- same backbone;
- same optimizer;
- same caption policy;
- same protocol;
- same K/Tmax;
- same learning rate;
- same precision.

## Run A — `L_bind`

Purpose:

> Test whether forcing candidate intents to bind to candidate-specific claimed text semantics produces meaningful WHAT specialization and whether that propagates into WHERE.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=bind \
  model.enable_claim_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_bind
```

Primary questions:

1. Does pairwise intent cosine decrease?
2. Does support cosine move away from ~1.0?
3. Does delta-Z cosine decrease?
4. Does t0 functional rank increase?
5. Does FULL begin to outperform MEAN / REPEAT?

This is the cleanest first ablation.

---

## Run B — `L_comp`

Purpose:

> Test whether complementary claim allocation alone creates useful candidate decomposition.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=comp \
  model.enable_claim_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_comp
```

Interpretation:

- If claim complementarity changes intents but grounding remains cloned, the bottleneck is downstream WHERE grounding.
- If it does not even change intents/effects, complementary claims alone are not enough.

---

## Run C — `L_comp + L_bind`

Purpose:

> Test semantic partition + semantic binding together without adding factor losses.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=core \
  model.enable_claim_head=true \
  objective.complementary_claim_weight=0.01 \
  objective.binding_weight=0.01 \
  hydra.run.dir=outputs/iag_srme_ablation_comp_bind
```

This is likely more meaningful than `L_comp` alone because complementarity without binding can in principle partition arbitrary claim mass.

---

## Run D — `L_factor`

Purpose:

> Test whether candidate factors become jointly semantically complete relative to the auxiliary full-query anchor.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=factor \
  model.enable_factor_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_factor
```

Important interpretation caveat:

`L_factor` detaches its auxiliary anchor within the loss branch, but that anchor is recomputed from backbone representations that are jointly updated by the rest of training. Therefore it is not a globally fixed target across optimizer steps.

Use the run as an ablation, but do not describe the semantic anchor as permanently fixed.

---

# 11. Do NOT run `L_unique` yet with the current trainer

Current `UniqueContributionLoss` is guarded by:

```text
require_activity_weights_for_unique = true
```

and `train.py` / `train_one_epoch()` currently calls the objective without external `active_weights`.

Therefore the current `objective=unique` path is expected to raise:

```text
L_unique requires externally justified activity weights
```

This is intentional.

STOP is not equivalent to semantic factor inactivity.

Do not disable this guard merely to make the run execute unless the experiment is explicitly defined as an all-active ablation.

For the same reason, the current `six_loss_experimental` config is **not a canonical runnable next step** without resolving the activity-weight semantics.

---

# 12. What success should look like

Do not judge the next loss only by Mean Recall.

The baseline tells us that retrieval can improve while candidate structure remains degenerate.

A promising run should move several structural metrics simultaneously.

## Strong positive signs

### Grounding

Baseline:

```text
support cosine ≈ 0.9998
support overlap ≈ 0.995
```

Desired direction:

```text
support cosine ↓ substantially
support overlap ↓ substantially
```

A value below ~0.98 would already be a major qualitative change from the current baseline.

### Token-level edits

Baseline global pairwise `delta_z` cosine is ~0.99.

Desired:

```text
delta_z cosine ↓
```

### t0 functional diversity

Baseline:

- epoch 3 rank = **1.663**
- epoch 5 rank = **1.595**

Desired:

```text
rank clearly > 2
t0 pairwise Δq cosine clearly < current ~0.99
```

### Policy usefulness

Baseline:

```text
MEAN ≈ FULL
best REPEAT ≳ FULL
```

A stronger factorized model should eventually show:

```text
FULL > MEAN
FULL > any fixed REPEAT-k
```

because dynamic candidate choice should matter.

### STOP

STOP itself does not need to be rare.

What matters is:

- it should not collapse at t0;
- it should correlate with truly low marginal utility;
- FULL should outperform forced controls.

---

# 13. Decision table after the next ablations

## Case A

```text
intent diversity improves
BUT support cosine remains ~1.0
```

Conclusion:

> Text-side semantic specialization is working, but the anchor-grounding mechanism is collapsing distinct intents into the same WHERE.

Next research target: grounding architecture / grounding objective.

---

## Case B

```text
intent + support diversity improve
BUT delta-Z remains ~parallel
```

Conclusion:

> WHAT and WHERE differ, but the shared editor maps them to nearly the same edit direction.

Next target: editor conditioning / functional specialization.

---

## Case C

```text
support + delta-Z diversity improve
BUT FULL ≈ MEAN ≈ REPEAT
```

Conclusion:

> Candidates are representationally different but their differences are not functionally useful for retrieval, or the selector cannot exploit them.

Next target: functional utility / scorer / trajectory credit.

---

## Case D

```text
FULL beats MEAN/REPEAT
AND structural diversity improves
AND recall improves
```

Conclusion:

> The auxiliary loss is creating genuine candidate specialization that the dynamic policy can exploit.

This is the desired regime.

---

## Case E

```text
all semantic auxiliary losses fail
support cosine stays ~1.0
delta-Z stays ~0.99
```

Conclusion:

> Stop adding more semantic losses.

At that point the failure is likely architectural: independent candidate grounding has no strong mechanism preventing all four queries from selecting the same region.

The next research step should target WHERE competition / candidate-conditioned grounding structure rather than adding another weak regularizer.

---

# 14. Baseline checkpoint table for future comparison

| Metric | Core best e3 | Core last e5 |
|---|---:|---:|
| Mean Recall | 27.258 | 26.697 |
| R@10 | 18.154 | 17.847 |
| R@50 | 36.363 | 35.547 |
| Reference-only | 14.438 | 14.843 |
| Best SINGLE | 24.527 | 24.395 |
| Best REPEAT | 27.441 | 26.915 |
| MEAN candidate | 27.308 | 26.681 |
| Support cosine | 0.999795 | 0.999821 |
| Support overlap | 0.995188 | 0.995809 |
| Support fraction | 7.37% | 8.32% |
| t0 Δq cosine | 0.9897 | 0.9923 |
| t0 effect rank | 1.663 | 1.595 |
| t1 effect rank | 2.339 | 2.297 |
| t2 effect rank | 3.440 | 3.440 |
| Mean executed edits | 2.682 | 2.195 |
| STOP hazard t2 | 25.74% | 67.25% |
| Max candidate share | 31.51% | 40.82% |

---

# 15. Current conclusion

The core run demonstrates that the implementation can train and retrieve, but the intended four-way edit decomposition does **not emerge from terminal + marginal utility supervision alone**.

The central structural signature is:

```text
four candidate identities
        ↓
almost identical visual supports
        ↓
almost parallel token edits
        ↓
weakly differentiated candidate consequences
        ↓
repeated edits become progressively less effective
        ↓
STOP becomes increasingly attractive
```

The immediate scientific question for the next experiments is therefore:

> **Can semantic auxiliary losses break candidate symmetry strongly enough that different WHAT representations produce different WHERE groundings and functionally distinct edits?**

The first recommended sequence is:

```text
CORE baseline
→ BIND
→ COMP
→ COMP+BIND
→ FACTOR
```

Do not jump directly to the full six-loss objective. Individual ablations are necessary to identify which supervision actually changes the failure signature.
# CIR IAG-SRME — Core Baseline Diagnostic
## Epoch 3 Best vs Epoch 5 Last

**Date:** 2026-08-30  
**Method:** IAG-SRME, FG-CLIP Base, K=4, Tmax=3, d=256  
**Training objective:** `L_terminal + 0.5 L_marginal` (`objective=core`)  
**Dataset / protocol:** FashionIQ val, `fashioniq_original`, `ordered_and`  
**Best checkpoint:** epoch 3, Mean Recall = 27.258  
**Last checkpoint:** epoch 5, Mean Recall = 26.697

---

# 1. Executive diagnosis

The run is **not failing because of numerical instability, a single-candidate monopoly, or immediate STOP collapse**.

The strongest structural failure is:

> **The four candidates have effectively identical visual grounding.**

At epoch 3:

- pairwise support cosine = **0.999795**
- pairwise support overlap = **0.995188**

At epoch 5:

- pairwise support cosine = **0.999821**
- pairwise support overlap = **0.995809**

So all four learnable edit queries are essentially looking at the **same visual region distribution**.

This is already present at the best checkpoint and therefore is not merely a consequence of over-training from epoch 3 to epoch 5. It is a structural failure mode of the core objective.

The second major failure is:

> **Candidate edits become increasingly similar at the first recurrent step, while later recurrent edits have rapidly diminishing effect on the final retrieval query.**

At epoch 3:

| timestep | mean `||Δq||` | pairwise Δq cosine | functional rank |
|---|---:|---:|---:|
| t0 | 0.3581 | 0.9897 | 1.663 |
| t1 | 0.0890 | 0.9444 | 2.339 |
| t2 | 0.0216 | 0.6105 | 3.440 |

At epoch 5:

| timestep | mean `||Δq||` | pairwise Δq cosine | functional rank |
|---|---:|---:|---:|
| t0 | 0.3665 | 0.9923 | 1.595 |
| t1 | 0.0825 | 0.9492 | 2.297 |
| t2 | 0.0190 | 0.6056 | 3.440 |

The t2 query effect is only about **6.0%** of t0 at epoch 3 and **5.2%** at epoch 5.

Thus the recurrent state is changing, but the later edits are becoming weak in the final retrieval space.

The third important observation is:

> **The scorer progressively learns to STOP much earlier between epoch 3 and epoch 5.**

| metric | epoch 3 | epoch 5 | delta |
|---|---:|---:|---:|
| mean executed edits | 2.682 | 2.195 | -0.486 |
| STOP hazard t0 | 0.08% | 0.08% | 0.00% |
| STOP hazard t1 | 3.38% | 9.83% | 6.45% |
| STOP hazard t2 | 25.74% | 67.25% | 41.51% |

This STOP behavior is probably **partly a rational response** to the weak late-step effects. It should not yet be treated as the root cause.

---

# 2. Retrieval degradation from epoch 3 to epoch 5

| metric | epoch 3 | epoch 5 | delta |
|---|---:|---:|---:|
| R@10 | 18.154 | 17.847 | -0.307 |
| R@50 | 36.363 | 35.547 | -0.816 |
| Mean Recall | 27.258 | 26.697 | -0.562 |

The drop is real but modest: **0.562 Mean Recall points**.

More important than the absolute drop is *where* it comes from.

## 2.1 Reference representation does not degrade

Reference-only retrieval actually improves:

- epoch 3: **14.438**
- epoch 5: **14.843**
- delta: **+0.406**

Therefore the degradation is not simply “FG-CLIP became worse everywhere”.

A plausible interpretation is:

> Continued fine-tuning improves the global reference/image representation while the composed edit trajectory becomes worse.

This should be treated as an inference, not a proven causal statement.

## 2.2 SINGLE edits remain relatively stable

Best SINGLE:

- epoch 3: **24.527**
- epoch 5: **24.395**

The degradation is much smaller than FULL.

This indicates that the root-state edit itself has not catastrophically broken.

## 2.3 Deeper/repeated behavior degrades more

Best REPEAT:

- epoch 3: **27.441**
- epoch 5: **26.915**

FULL:

- epoch 3: **27.258**
- epoch 5: **26.697**

The candidate effects at t1/t2 also become smaller in query space.

This points toward a **recurrent-composition / edit-trajectory issue**, not a simple first-edit failure.

---

# 3. Primary structural failure: WHERE collapse

The strongest diagnostic signal is the visual support similarity.

## Epoch 3

- support cosine: **0.999795**
- support overlap: **0.995188**
- support fraction: **7.37%**
- effective support size: **13.95 / 196 tokens**

## Epoch 5

- support cosine: **0.999821**
- support overlap: **0.995809**
- support fraction: **8.32%**
- effective support size: **15.78 / 196 tokens**

This is not “all candidates attend to nearby but different regions”.

Numerically, the support vectors are almost identical.

The failure flag `grounding_clone=true` is therefore well-supported.

### Important nuance

Grounding is **not over-sparse**.

It actually becomes broader between epoch 3 and epoch 5:

- support fraction: 7.37% → 8.32%
- entropy: 2.603 → 2.734
- effective size: 13.95 → 15.78

Therefore the problem is not “entmax became too sharp”.

The problem is:

> **all four candidates use almost the same sparse-ish support.**

---

# 4. WHAT / context / edit similarity

The four candidate pathways are already highly correlated before/through editing.

At epoch 3, the global specialization matrices show approximately:

- intent pairwise cosine: about **0.94–0.95**
- context pairwise cosine: about **0.95–0.96**
- delta-Z pairwise cosine: about **0.986–0.989**
- support pairwise cosine: about **0.9997–0.9999**

At epoch 5:

- intent pairwise cosine remains about **0.94–0.95**
- context cosine rises slightly to around **0.95–0.96**
- delta-Z cosine rises to around **0.989–0.991**
- support cosine remains essentially **1.0**

Thus the network has four identities, but most of the visual/edit computation is highly redundant.

The final `delta_q` vectors are less identical than `delta_z`, but that does not rescue the underlying token-level specialization.

---

# 5. Recurrent attenuation

One particularly important pattern is the rapid reduction in retrieval-space edit strength.

## Epoch 3

```text
t0 ||Δq|| ≈ 0.3581
t1 ||Δq|| ≈ 0.0890
t2 ||Δq|| ≈ 0.0216
```

## Epoch 5

```text
t0 ||Δq|| ≈ 0.3665
t1 ||Δq|| ≈ 0.0825
t2 ||Δq|| ≈ 0.0190
```

Meanwhile `||ΔZ||` actually increases:

- t0: 2.067 → 2.262
- t1: 2.070 → 2.264
- t2: 2.085 → 2.301

So the editor is not becoming inactive in token space.

Instead:

> **large token-state edits produce progressively smaller retrieval-query changes at later timesteps.**

This is a critical distinction.

Possible mechanisms include saturation/cancellation in the readout, repeated edits along similar directions, normalization/capping effects, or the state reaching a region where additional local changes have weak retrieval-space leverage.

The current diagnostics do not isolate which of those mechanisms is causal.

---

# 6. STOP drift is secondary, not yet the root cause

Epoch 3:

```text
STOP hazard: t0=0.08%,
             t1=3.38%,
             t2=25.74%

mean executed edits = 2.682
```

Epoch 5:

```text
STOP hazard: t0=0.08%,
             t1=9.83%,
             t2=67.25%

mean executed edits = 2.195
```

STOP clearly becomes much more aggressive.

However, forced REPEAT only slightly outperforms FULL:

- epoch 3: best REPEAT / FULL = **1.0067**
- epoch 5: best REPEAT / FULL = **1.0082**

Therefore there is not a large amount of hidden retrieval performance being destroyed solely by the STOP policy.

The better interpretation is:

> Later candidate consequences are becoming weak / redundant, and the scorer learns that stopping is often nearly as good.

---

# 7. Candidate selection is not fully collapsed, but is drifting

Conditional candidate distribution among executed edits:

## Epoch 3

```text
candidate 0: 13.99%
candidate 1: 28.07%
candidate 2: 31.51%
candidate 3: 26.43%
```

## Epoch 5

```text
candidate 0: 23.19%
candidate 1: 40.82%
candidate 2: 18.07%
candidate 3: 17.92%
```

There is no hard monopoly, but candidate 1 rises from **28.07%** to **40.82%**.

This should be watched in later runs.

---

# 8. Why the core objective permits this failure

Current core objective:

```math
L_core = L_terminal + 0.5 L_marginal
```

`L_terminal` asks the final composed query to retrieve the target.

`L_marginal` asks the scorer to match target-derived marginal utility of candidate consequences.

Neither objective directly requires:

- different candidates to represent different semantic claims;
- different candidates to ground to different or complementary visual regions;
- different candidate token edits to have distinct functional effects;
- the four candidate identities to form a meaningful decomposition.

If all candidates discover nearly the same useful edit direction, both losses can still be optimized.

This explains why:

```text
support cosine ≈ 1.0
delta-Z cosine ≈ 0.99
MEAN ≈ FULL
REPEAT ≈ FULL
```

can coexist with a decreasing training loss.

The core objective provides **utility supervision**, but almost no explicit **symmetry-breaking supervision**.

---

# 9. Causal hypothesis to carry into the next experiments

The most plausible current chain is:

```text
4 text queries
    ↓
highly similar intent representations
    ↓
almost identical anchor-grounding distributions
    ↓
almost identical grounded evidence
    ↓
highly similar contexts
    ↓
almost parallel token-level edits
    ↓
repeated same-direction edits
    ↓
later query-space effects attenuate strongly
    ↓
marginal utility of later edits becomes small
    ↓
scorer increasingly chooses STOP
```

This chain is a **diagnostic hypothesis**, not yet a proven causal graph.

The next loss ablations should be designed to break specific links and see which metrics move.

---

# 10. Recommended loss ablation order

The following runs should initially use only **5 epochs**, because the baseline already exposes its trend by epoch 3–5.

Keep all other variables identical:

- same seed;
- same backbone;
- same optimizer;
- same caption policy;
- same protocol;
- same K/Tmax;
- same learning rate;
- same precision.

## Run A — `L_bind`

Purpose:

> Test whether forcing candidate intents to bind to candidate-specific claimed text semantics produces meaningful WHAT specialization and whether that propagates into WHERE.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=bind \
  model.enable_claim_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_bind
```

Primary questions:

1. Does pairwise intent cosine decrease?
2. Does support cosine move away from ~1.0?
3. Does delta-Z cosine decrease?
4. Does t0 functional rank increase?
5. Does FULL begin to outperform MEAN / REPEAT?

This is the cleanest first ablation.

---

## Run B — `L_comp`

Purpose:

> Test whether complementary claim allocation alone creates useful candidate decomposition.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=comp \
  model.enable_claim_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_comp
```

Interpretation:

- If claim complementarity changes intents but grounding remains cloned, the bottleneck is downstream WHERE grounding.
- If it does not even change intents/effects, complementary claims alone are not enough.

---

## Run C — `L_comp + L_bind`

Purpose:

> Test semantic partition + semantic binding together without adding factor losses.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=core \
  model.enable_claim_head=true \
  objective.complementary_claim_weight=0.01 \
  objective.binding_weight=0.01 \
  hydra.run.dir=outputs/iag_srme_ablation_comp_bind
```

This is likely more meaningful than `L_comp` alone because complementarity without binding can in principle partition arbitrary claim mass.

---

## Run D — `L_factor`

Purpose:

> Test whether candidate factors become jointly semantically complete relative to the auxiliary full-query anchor.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=factor \
  model.enable_factor_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_factor
```

Important interpretation caveat:

`L_factor` detaches its auxiliary anchor within the loss branch, but that anchor is recomputed from backbone representations that are jointly updated by the rest of training. Therefore it is not a globally fixed target across optimizer steps.

Use the run as an ablation, but do not describe the semantic anchor as permanently fixed.

---

# 11. Do NOT run `L_unique` yet with the current trainer

Current `UniqueContributionLoss` is guarded by:

```text
require_activity_weights_for_unique = true
```

and `train.py` / `train_one_epoch()` currently calls the objective without external `active_weights`.

Therefore the current `objective=unique` path is expected to raise:

```text
L_unique requires externally justified activity weights
```

This is intentional.

STOP is not equivalent to semantic factor inactivity.

Do not disable this guard merely to make the run execute unless the experiment is explicitly defined as an all-active ablation.

For the same reason, the current `six_loss_experimental` config is **not a canonical runnable next step** without resolving the activity-weight semantics.

---

# 12. What success should look like

Do not judge the next loss only by Mean Recall.

The baseline tells us that retrieval can improve while candidate structure remains degenerate.

A promising run should move several structural metrics simultaneously.

## Strong positive signs

### Grounding

Baseline:

```text
support cosine ≈ 0.9998
support overlap ≈ 0.995
```

Desired direction:

```text
support cosine ↓ substantially
support overlap ↓ substantially
```

A value below ~0.98 would already be a major qualitative change from the current baseline.

### Token-level edits

Baseline global pairwise `delta_z` cosine is ~0.99.

Desired:

```text
delta_z cosine ↓
```

### t0 functional diversity

Baseline:

- epoch 3 rank = **1.663**
- epoch 5 rank = **1.595**

Desired:

```text
rank clearly > 2
t0 pairwise Δq cosine clearly < current ~0.99
```

### Policy usefulness

Baseline:

```text
MEAN ≈ FULL
best REPEAT ≳ FULL
```

A stronger factorized model should eventually show:

```text
FULL > MEAN
FULL > any fixed REPEAT-k
```

because dynamic candidate choice should matter.

### STOP

STOP itself does not need to be rare.

What matters is:

- it should not collapse at t0;
- it should correlate with truly low marginal utility;
- FULL should outperform forced controls.

---

# 13. Decision table after the next ablations

## Case A

```text
intent diversity improves
BUT support cosine remains ~1.0
```

Conclusion:

> Text-side semantic specialization is working, but the anchor-grounding mechanism is collapsing distinct intents into the same WHERE.

Next research target: grounding architecture / grounding objective.

---

## Case B

```text
intent + support diversity improve
BUT delta-Z remains ~parallel
```

Conclusion:

> WHAT and WHERE differ, but the shared editor maps them to nearly the same edit direction.

Next target: editor conditioning / functional specialization.

---

## Case C

```text
support + delta-Z diversity improve
BUT FULL ≈ MEAN ≈ REPEAT
```

Conclusion:

> Candidates are representationally different but their differences are not functionally useful for retrieval, or the selector cannot exploit them.

Next target: functional utility / scorer / trajectory credit.

---

## Case D

```text
FULL beats MEAN/REPEAT
AND structural diversity improves
AND recall improves
```

Conclusion:

> The auxiliary loss is creating genuine candidate specialization that the dynamic policy can exploit.

This is the desired regime.

---

## Case E

```text
all semantic auxiliary losses fail
support cosine stays ~1.0
delta-Z stays ~0.99
```

Conclusion:

> Stop adding more semantic losses.

At that point the failure is likely architectural: independent candidate grounding has no strong mechanism preventing all four queries from selecting the same region.

The next research step should target WHERE competition / candidate-conditioned grounding structure rather than adding another weak regularizer.

---

# 14. Baseline checkpoint table for future comparison

| Metric | Core best e3 | Core last e5 |
|---|---:|---:|
| Mean Recall | 27.258 | 26.697 |
| R@10 | 18.154 | 17.847 |
| R@50 | 36.363 | 35.547 |
| Reference-only | 14.438 | 14.843 |
| Best SINGLE | 24.527 | 24.395 |
| Best REPEAT | 27.441 | 26.915 |
| MEAN candidate | 27.308 | 26.681 |
| Support cosine | 0.999795 | 0.999821 |
| Support overlap | 0.995188 | 0.995809 |
| Support fraction | 7.37% | 8.32% |
| t0 Δq cosine | 0.9897 | 0.9923 |
| t0 effect rank | 1.663 | 1.595 |
| t1 effect rank | 2.339 | 2.297 |
| t2 effect rank | 3.440 | 3.440 |
| Mean executed edits | 2.682 | 2.195 |
| STOP hazard t2 | 25.74% | 67.25% |
| Max candidate share | 31.51% | 40.82% |

---

# 15. Current conclusion

The core run demonstrates that the implementation can train and retrieve, but the intended four-way edit decomposition does **not emerge from terminal + marginal utility supervision alone**.

The central structural signature is:

```text
four candidate identities
        ↓
almost identical visual supports
        ↓
almost parallel token edits
        ↓
weakly differentiated candidate consequences
        ↓
repeated edits become progressively less effective
        ↓
STOP becomes increasingly attractive
```

The immediate scientific question for the next experiments is therefore:

> **Can semantic auxiliary losses break candidate symmetry strongly enough that different WHAT representations produce different WHERE groundings and functionally distinct edits?**

The first recommended sequence is:

```text
CORE baseline
→ BIND
→ COMP
→ COMP+BIND
→ FACTOR
```

Do not jump directly to the full six-loss objective. Individual ablations are necessary to identify which supervision actually changes the failure signature.

---

# 16. BIND ablation diagnostic — first structural test

**Experiment:** `objective=bind`, `model.enable_claim_head=true`  
**Objective:**

```math
L = L_terminal + 0.5 L_marginal + 0.01 L_bind
```

**Checkpoint analyzed:** epoch 3 best  
**Mean Recall:** **27.582**

This should be compared against the core baseline best checkpoint at epoch 3, not the degraded epoch-5 checkpoint, because both best checkpoints occur at epoch 3.

## 16.1 Retrieval result

| metric | CORE e3 | BIND e3 | delta |
|---|---:|---:|---:|
| R@10 | 18.154 | 18.411 | +0.257 |
| R@50 | 36.363 | 36.753 | +0.390 |
| Mean Recall | 27.258 | 27.582 | **+0.324** |

`L_bind` gives a **small positive retrieval gain of +0.324 Mean Recall**.

This is useful evidence that the binding signal is not obviously destructive, but the gain is too small to claim that the decomposition problem is solved.

---

# 17. Did BIND solve WHAT specialization?

Only partially.

| diagnostic | CORE e3 | BIND e3 | direction |
|---|---:|---:|---|
| pairwise intent cosine | 0.9476 | 0.9423 | ↓ 0.0053 |
| pairwise context cosine | 0.9549 | 0.9399 | ↓ 0.0150 |
| pairwise delta-Z cosine | 0.9877 | 0.9838 | ↓ 0.0039 |
| pairwise delta-q cosine | 0.8510 | 0.8420 | ↓ 0.0090 |

So BIND does move the representation in the intended direction:

```text
intent slightly less similar
→ context somewhat less similar
→ delta-Z slightly less similar
→ delta-q slightly less similar
```

However, the changes are modest.

The four intent vectors are still highly correlated:

```text
CORE: 0.9476
BIND: 0.9423
```

Thus `L_bind` creates **weak semantic symmetry breaking**, not a clean four-way decomposition.

---

# 18. Did BIND solve WHERE grounding collapse?

**No. This is the most important result of the ablation.**

| grounding diagnostic | CORE e3 | BIND e3 |
|---|---:|---:|
| pairwise support cosine | 0.999795 | 0.999372 |
| pairwise support overlap | 0.995188 | 0.989260 |
| support fraction | 7.37% | **25.17%** |
| effective support size | 13.95 | **47.19** |

BIND substantially changes **how broad** grounding is:

```text
support fraction: 7.37% → 25.17%
effective size:   13.95 → 47.19 tokens
```

but does almost nothing to make the four candidates look at different regions:

```text
support cosine:  0.999795
              →  0.999372

support overlap: 0.995188
              →  0.989260
```

The four support maps remain essentially clones.

This is stronger evidence for the following diagnosis:

> **The current bottleneck is not merely that the text queries lack semantic differentiation. Even after BIND makes intents/contexts somewhat more distinct, the anchor grounder still maps them to almost the same visual support.**

Therefore BIND alone does **not** solve WHERE specialization.

---

# 19. BIND changes edit magnitude more than edit direction

At t0:

| diagnostic | CORE e3 | BIND e3 |
|---|---:|---:|
| mean `||ΔZ||` | 2.067 | **3.581** |
| mean `||Δq||` | 0.3581 | 0.3526 |
| pairwise Δq cosine | 0.9897 | 0.9871 |
| functional rank | 1.663 | 1.736 |

The token-space edit norm jumps strongly:

```text
||ΔZ||: 2.067 → 3.581
```

while the final retrieval-space effect norm barely changes:

```text
||Δq||: 0.3581 → 0.3526
```

This means BIND causes the editor to make **larger token-state modifications**, but those larger modifications are not translated proportionally into more useful or more distinct retrieval-space consequences.

This reinforces the earlier recurrent/readout concern.

---

# 20. Recurrent attenuation still exists under BIND

BIND checkpoint:

| timestep | mean `||Δq||` | pairwise Δq cosine | effect rank |
|---|---:|---:|---:|
| t0 | 0.3526 | 0.9871 | 1.736 |
| t1 | 0.0930 | 0.9350 | 2.422 |
| t2 | 0.0246 | 0.5982 | 3.438 |

t2 has only about:

```text
7.0% of the t0 Δq norm
```

So the core failure:

```text
strong first edit
→ weak later retrieval-space effects
```

is still present.

BIND does not fix recurrent attenuation.

---

# 21. STOP behavior improves, but repeated-action behavior becomes even stronger

| diagnostic | CORE e3 | BIND e3 |
|---|---:|---:|
| mean executed edits | 2.682 | 2.849 |
| STOP hazard t1 | 3.38% | 2.25% |
| STOP hazard t2 | 25.74% | **10.27%** |
| repeated-candidate fraction | 90.99% | **95.40%** |
| max candidate share | 31.51% | 29.78% |

BIND strongly reduces premature STOP:

```text
t2 STOP hazard:
25.74%
→ 10.27%
```

and increases average edit count.

However, **95.40% of validation queries repeat at least one candidate identity**.

This is even higher than the core baseline.

So the network is not using the extra trajectory depth to compose clearly distinct candidate actions. It is frequently applying the same identity again.

This is consistent with the support maps still being clones.

---

# 22. FULL still does not exploit dynamic candidate specialization

| control | CORE e3 | BIND e3 |
|---|---:|---:|
| FULL | 27.258 | 27.582 |
| MEAN | 27.308 | 27.666 |
| best REPEAT | 27.441 | 27.708 |

For BIND:

```text
FULL        = 27.582
MEAN        = 27.666
best REPEAT = 27.708
```

So:

```text
MEAN > FULL by +0.084
best REPEAT > FULL by +0.126
```

The gaps are small, but the qualitative result remains unchanged:

> **Dynamic candidate selection still provides no measurable advantage over a mean candidate or repeatedly applying one fixed candidate.**

Therefore BIND improves retrieval slightly without demonstrating that the intended multi-candidate mechanism has become functionally necessary.

---

# 23. Updated causal interpretation after BIND

The CORE-only hypothesis was:

```text
similar WHAT
→ cloned WHERE
→ parallel edits
→ weak later effects
→ STOP
```

BIND gives us a stronger localization of the failure.

Observed under BIND:

```text
WHAT becomes somewhat more distinct       ✅
context becomes somewhat more distinct    ✅
WHERE is still almost identical           ❌
delta-Z remains highly parallel            ❌
FULL still ≈ MEAN ≈ REPEAT                 ❌
```

Therefore the updated hypothesis is:

```text
BIND
  ↓
partial text/semantic symmetry breaking
  ↓
AnchorGrounder collapses the distinct signals
into nearly the same spatial distribution
  ↓
Grounded evidence remains largely shared
  ↓
Shared editor still produces near-parallel token edits
  ↓
multi-candidate policy has little functional reason to specialize
```

The new evidence shifts suspicion **more strongly toward the WHERE/grounding interface**.

---

# 24. Decision on BIND

## What BIND achieved

- Mean Recall: **+0.324**
- intent cosine decreases slightly;
- context cosine decreases more clearly;
- t0 functional rank improves from 1.663 → 1.736;
- candidate selection becomes slightly more balanced;
- STOP becomes less aggressive.

## What BIND did not achieve

- support maps are still clones;
- support overlap remains ~0.99;
- delta-Z directions remain highly correlated;
- t0 candidate effects remain almost parallel;
- recurrent attenuation remains;
- repeated candidate selection rises to 95.40%;
- MEAN and fixed REPEAT still match/beat FULL.

## Verdict

```text
L_bind = mildly useful auxiliary signal
       ≠ solution to candidate decomposition
```

It should be retained as a promising semantic signal for later combinations, but **BIND alone does not solve the central structural failure**.

---

# 25. What the next ablation should answer

The original plan proposed COMP after BIND.

That is still useful as a controlled experiment, but the BIND result changes what we should look for.

For `L_comp`, the key question is no longer merely:

```text
Does intent cosine decrease?
```

It is:

```text
Can stronger claim complementarity make the intent differences
large enough to survive the AnchorGrounder and produce distinct supports?
```

The decisive metrics are:

1. `pairwise_support_cosine`
2. `pairwise_support_overlap`
3. `pairwise_delta_z_cosine`
4. t0 `functional_effective_rank`
5. `FULL - MEAN`
6. `FULL - best_REPEATED`
7. repeated-candidate fraction

If COMP or COMP+BIND lowers intent/context cosine but support cosine remains around `0.999`, that would be strong evidence that **the architecture needs a direct WHERE-side specialization mechanism**, rather than more text-side semantic losses.

---

# 26. Current experiment scoreboard

| Experiment | Mean Recall | support cosine | support overlap | t0 rank | t0 Δq cosine | mean edits | repeat-query fraction | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CORE best e3 | 27.258 | 0.999795 | 0.995188 | 1.663 | 0.9897 | 2.682 | 90.99% | WHERE clone |
| BIND best e3 | **27.582** | 0.999372 | 0.989260 | **1.736** | 0.9871 | 2.849 | 95.40% | mild WHAT improvement, WHERE still clone |

This table should be extended with COMP, COMP+BIND, FACTOR, and any later architectural interventions.

# CIR IAG-SRME — Core Baseline Diagnostic
## Epoch 3 Best vs Epoch 5 Last

**Date:** 2026-08-30  
**Method:** IAG-SRME, FG-CLIP Base, K=4, Tmax=3, d=256  
**Training objective:** `L_terminal + 0.5 L_marginal` (`objective=core`)  
**Dataset / protocol:** FashionIQ val, `fashioniq_original`, `ordered_and`  
**Best checkpoint:** epoch 3, Mean Recall = 27.258  
**Last checkpoint:** epoch 5, Mean Recall = 26.697

---

# 1. Executive diagnosis

The run is **not failing because of numerical instability, a single-candidate monopoly, or immediate STOP collapse**.

The strongest structural failure is:

> **The four candidates have effectively identical visual grounding.**

At epoch 3:

- pairwise support cosine = **0.999795**
- pairwise support overlap = **0.995188**

At epoch 5:

- pairwise support cosine = **0.999821**
- pairwise support overlap = **0.995809**

So all four learnable edit queries are essentially looking at the **same visual region distribution**.

This is already present at the best checkpoint and therefore is not merely a consequence of over-training from epoch 3 to epoch 5. It is a structural failure mode of the core objective.

The second major failure is:

> **Candidate edits become increasingly similar at the first recurrent step, while later recurrent edits have rapidly diminishing effect on the final retrieval query.**

At epoch 3:

| timestep | mean `||Δq||` | pairwise Δq cosine | functional rank |
|---|---:|---:|---:|
| t0 | 0.3581 | 0.9897 | 1.663 |
| t1 | 0.0890 | 0.9444 | 2.339 |
| t2 | 0.0216 | 0.6105 | 3.440 |

At epoch 5:

| timestep | mean `||Δq||` | pairwise Δq cosine | functional rank |
|---|---:|---:|---:|
| t0 | 0.3665 | 0.9923 | 1.595 |
| t1 | 0.0825 | 0.9492 | 2.297 |
| t2 | 0.0190 | 0.6056 | 3.440 |

The t2 query effect is only about **6.0%** of t0 at epoch 3 and **5.2%** at epoch 5.

Thus the recurrent state is changing, but the later edits are becoming weak in the final retrieval space.

The third important observation is:

> **The scorer progressively learns to STOP much earlier between epoch 3 and epoch 5.**

| metric | epoch 3 | epoch 5 | delta |
|---|---:|---:|---:|
| mean executed edits | 2.682 | 2.195 | -0.486 |
| STOP hazard t0 | 0.08% | 0.08% | 0.00% |
| STOP hazard t1 | 3.38% | 9.83% | 6.45% |
| STOP hazard t2 | 25.74% | 67.25% | 41.51% |

This STOP behavior is probably **partly a rational response** to the weak late-step effects. It should not yet be treated as the root cause.

---

# 2. Retrieval degradation from epoch 3 to epoch 5

| metric | epoch 3 | epoch 5 | delta |
|---|---:|---:|---:|
| R@10 | 18.154 | 17.847 | -0.307 |
| R@50 | 36.363 | 35.547 | -0.816 |
| Mean Recall | 27.258 | 26.697 | -0.562 |

The drop is real but modest: **0.562 Mean Recall points**.

More important than the absolute drop is *where* it comes from.

## 2.1 Reference representation does not degrade

Reference-only retrieval actually improves:

- epoch 3: **14.438**
- epoch 5: **14.843**
- delta: **+0.406**

Therefore the degradation is not simply “FG-CLIP became worse everywhere”.

A plausible interpretation is:

> Continued fine-tuning improves the global reference/image representation while the composed edit trajectory becomes worse.

This should be treated as an inference, not a proven causal statement.

## 2.2 SINGLE edits remain relatively stable

Best SINGLE:

- epoch 3: **24.527**
- epoch 5: **24.395**

The degradation is much smaller than FULL.

This indicates that the root-state edit itself has not catastrophically broken.

## 2.3 Deeper/repeated behavior degrades more

Best REPEAT:

- epoch 3: **27.441**
- epoch 5: **26.915**

FULL:

- epoch 3: **27.258**
- epoch 5: **26.697**

The candidate effects at t1/t2 also become smaller in query space.

This points toward a **recurrent-composition / edit-trajectory issue**, not a simple first-edit failure.

---

# 3. Primary structural failure: WHERE collapse

The strongest diagnostic signal is the visual support similarity.

## Epoch 3

- support cosine: **0.999795**
- support overlap: **0.995188**
- support fraction: **7.37%**
- effective support size: **13.95 / 196 tokens**

## Epoch 5

- support cosine: **0.999821**
- support overlap: **0.995809**
- support fraction: **8.32%**
- effective support size: **15.78 / 196 tokens**

This is not “all candidates attend to nearby but different regions”.

Numerically, the support vectors are almost identical.

The failure flag `grounding_clone=true` is therefore well-supported.

### Important nuance

Grounding is **not over-sparse**.

It actually becomes broader between epoch 3 and epoch 5:

- support fraction: 7.37% → 8.32%
- entropy: 2.603 → 2.734
- effective size: 13.95 → 15.78

Therefore the problem is not “entmax became too sharp”.

The problem is:

> **all four candidates use almost the same sparse-ish support.**

---

# 4. WHAT / context / edit similarity

The four candidate pathways are already highly correlated before/through editing.

At epoch 3, the global specialization matrices show approximately:

- intent pairwise cosine: about **0.94–0.95**
- context pairwise cosine: about **0.95–0.96**
- delta-Z pairwise cosine: about **0.986–0.989**
- support pairwise cosine: about **0.9997–0.9999**

At epoch 5:

- intent pairwise cosine remains about **0.94–0.95**
- context cosine rises slightly to around **0.95–0.96**
- delta-Z cosine rises to around **0.989–0.991**
- support cosine remains essentially **1.0**

Thus the network has four identities, but most of the visual/edit computation is highly redundant.

The final `delta_q` vectors are less identical than `delta_z`, but that does not rescue the underlying token-level specialization.

---

# 5. Recurrent attenuation

One particularly important pattern is the rapid reduction in retrieval-space edit strength.

## Epoch 3

```text
t0 ||Δq|| ≈ 0.3581
t1 ||Δq|| ≈ 0.0890
t2 ||Δq|| ≈ 0.0216
```

## Epoch 5

```text
t0 ||Δq|| ≈ 0.3665
t1 ||Δq|| ≈ 0.0825
t2 ||Δq|| ≈ 0.0190
```

Meanwhile `||ΔZ||` actually increases:

- t0: 2.067 → 2.262
- t1: 2.070 → 2.264
- t2: 2.085 → 2.301

So the editor is not becoming inactive in token space.

Instead:

> **large token-state edits produce progressively smaller retrieval-query changes at later timesteps.**

This is a critical distinction.

Possible mechanisms include saturation/cancellation in the readout, repeated edits along similar directions, normalization/capping effects, or the state reaching a region where additional local changes have weak retrieval-space leverage.

The current diagnostics do not isolate which of those mechanisms is causal.

---

# 6. STOP drift is secondary, not yet the root cause

Epoch 3:

```text
STOP hazard: t0=0.08%,
             t1=3.38%,
             t2=25.74%

mean executed edits = 2.682
```

Epoch 5:

```text
STOP hazard: t0=0.08%,
             t1=9.83%,
             t2=67.25%

mean executed edits = 2.195
```

STOP clearly becomes much more aggressive.

However, forced REPEAT only slightly outperforms FULL:

- epoch 3: best REPEAT / FULL = **1.0067**
- epoch 5: best REPEAT / FULL = **1.0082**

Therefore there is not a large amount of hidden retrieval performance being destroyed solely by the STOP policy.

The better interpretation is:

> Later candidate consequences are becoming weak / redundant, and the scorer learns that stopping is often nearly as good.

---

# 7. Candidate selection is not fully collapsed, but is drifting

Conditional candidate distribution among executed edits:

## Epoch 3

```text
candidate 0: 13.99%
candidate 1: 28.07%
candidate 2: 31.51%
candidate 3: 26.43%
```

## Epoch 5

```text
candidate 0: 23.19%
candidate 1: 40.82%
candidate 2: 18.07%
candidate 3: 17.92%
```

There is no hard monopoly, but candidate 1 rises from **28.07%** to **40.82%**.

This should be watched in later runs.

---

# 8. Why the core objective permits this failure

Current core objective:

```math
L_core = L_terminal + 0.5 L_marginal
```

`L_terminal` asks the final composed query to retrieve the target.

`L_marginal` asks the scorer to match target-derived marginal utility of candidate consequences.

Neither objective directly requires:

- different candidates to represent different semantic claims;
- different candidates to ground to different or complementary visual regions;
- different candidate token edits to have distinct functional effects;
- the four candidate identities to form a meaningful decomposition.

If all candidates discover nearly the same useful edit direction, both losses can still be optimized.

This explains why:

```text
support cosine ≈ 1.0
delta-Z cosine ≈ 0.99
MEAN ≈ FULL
REPEAT ≈ FULL
```

can coexist with a decreasing training loss.

The core objective provides **utility supervision**, but almost no explicit **symmetry-breaking supervision**.

---

# 9. Causal hypothesis to carry into the next experiments

The most plausible current chain is:

```text
4 text queries
    ↓
highly similar intent representations
    ↓
almost identical anchor-grounding distributions
    ↓
almost identical grounded evidence
    ↓
highly similar contexts
    ↓
almost parallel token-level edits
    ↓
repeated same-direction edits
    ↓
later query-space effects attenuate strongly
    ↓
marginal utility of later edits becomes small
    ↓
scorer increasingly chooses STOP
```

This chain is a **diagnostic hypothesis**, not yet a proven causal graph.

The next loss ablations should be designed to break specific links and see which metrics move.

---

# 10. Recommended loss ablation order

The following runs should initially use only **5 epochs**, because the baseline already exposes its trend by epoch 3–5.

Keep all other variables identical:

- same seed;
- same backbone;
- same optimizer;
- same caption policy;
- same protocol;
- same K/Tmax;
- same learning rate;
- same precision.

## Run A — `L_bind`

Purpose:

> Test whether forcing candidate intents to bind to candidate-specific claimed text semantics produces meaningful WHAT specialization and whether that propagates into WHERE.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=bind \
  model.enable_claim_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_bind
```

Primary questions:

1. Does pairwise intent cosine decrease?
2. Does support cosine move away from ~1.0?
3. Does delta-Z cosine decrease?
4. Does t0 functional rank increase?
5. Does FULL begin to outperform MEAN / REPEAT?

This is the cleanest first ablation.

---

## Run B — `L_comp`

Purpose:

> Test whether complementary claim allocation alone creates useful candidate decomposition.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=comp \
  model.enable_claim_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_comp
```

Interpretation:

- If claim complementarity changes intents but grounding remains cloned, the bottleneck is downstream WHERE grounding.
- If it does not even change intents/effects, complementary claims alone are not enough.

---

## Run C — `L_comp + L_bind`

Purpose:

> Test semantic partition + semantic binding together without adding factor losses.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=core \
  model.enable_claim_head=true \
  objective.complementary_claim_weight=0.01 \
  objective.binding_weight=0.01 \
  hydra.run.dir=outputs/iag_srme_ablation_comp_bind
```

This is likely more meaningful than `L_comp` alone because complementarity without binding can in principle partition arbitrary claim mass.

---

## Run D — `L_factor`

Purpose:

> Test whether candidate factors become jointly semantically complete relative to the auxiliary full-query anchor.

Command:

```bash
python src/train.py \
  backbone=fgclip_base_full \
  experiment=iag_srme_base_full \
  experiment.epochs=5 \
  objective=factor \
  model.enable_factor_head=true \
  hydra.run.dir=outputs/iag_srme_ablation_factor
```

Important interpretation caveat:

`L_factor` detaches its auxiliary anchor within the loss branch, but that anchor is recomputed from backbone representations that are jointly updated by the rest of training. Therefore it is not a globally fixed target across optimizer steps.

Use the run as an ablation, but do not describe the semantic anchor as permanently fixed.

---

# 11. Do NOT run `L_unique` yet with the current trainer

Current `UniqueContributionLoss` is guarded by:

```text
require_activity_weights_for_unique = true
```

and `train.py` / `train_one_epoch()` currently calls the objective without external `active_weights`.

Therefore the current `objective=unique` path is expected to raise:

```text
L_unique requires externally justified activity weights
```

This is intentional.

STOP is not equivalent to semantic factor inactivity.

Do not disable this guard merely to make the run execute unless the experiment is explicitly defined as an all-active ablation.

For the same reason, the current `six_loss_experimental` config is **not a canonical runnable next step** without resolving the activity-weight semantics.

---

# 12. What success should look like

Do not judge the next loss only by Mean Recall.

The baseline tells us that retrieval can improve while candidate structure remains degenerate.

A promising run should move several structural metrics simultaneously.

## Strong positive signs

### Grounding

Baseline:

```text
support cosine ≈ 0.9998
support overlap ≈ 0.995
```

Desired direction:

```text
support cosine ↓ substantially
support overlap ↓ substantially
```

A value below ~0.98 would already be a major qualitative change from the current baseline.

### Token-level edits

Baseline global pairwise `delta_z` cosine is ~0.99.

Desired:

```text
delta_z cosine ↓
```

### t0 functional diversity

Baseline:

- epoch 3 rank = **1.663**
- epoch 5 rank = **1.595**

Desired:

```text
rank clearly > 2
t0 pairwise Δq cosine clearly < current ~0.99
```

### Policy usefulness

Baseline:

```text
MEAN ≈ FULL
best REPEAT ≳ FULL
```

A stronger factorized model should eventually show:

```text
FULL > MEAN
FULL > any fixed REPEAT-k
```

because dynamic candidate choice should matter.

### STOP

STOP itself does not need to be rare.

What matters is:

- it should not collapse at t0;
- it should correlate with truly low marginal utility;
- FULL should outperform forced controls.

---

# 13. Decision table after the next ablations

## Case A

```text
intent diversity improves
BUT support cosine remains ~1.0
```

Conclusion:

> Text-side semantic specialization is working, but the anchor-grounding mechanism is collapsing distinct intents into the same WHERE.

Next research target: grounding architecture / grounding objective.

---

## Case B

```text
intent + support diversity improve
BUT delta-Z remains ~parallel
```

Conclusion:

> WHAT and WHERE differ, but the shared editor maps them to nearly the same edit direction.

Next target: editor conditioning / functional specialization.

---

## Case C

```text
support + delta-Z diversity improve
BUT FULL ≈ MEAN ≈ REPEAT
```

Conclusion:

> Candidates are representationally different but their differences are not functionally useful for retrieval, or the selector cannot exploit them.

Next target: functional utility / scorer / trajectory credit.

---

## Case D

```text
FULL beats MEAN/REPEAT
AND structural diversity improves
AND recall improves
```

Conclusion:

> The auxiliary loss is creating genuine candidate specialization that the dynamic policy can exploit.

This is the desired regime.

---

## Case E

```text
all semantic auxiliary losses fail
support cosine stays ~1.0
delta-Z stays ~0.99
```

Conclusion:

> Stop adding more semantic losses.

At that point the failure is likely architectural: independent candidate grounding has no strong mechanism preventing all four queries from selecting the same region.

The next research step should target WHERE competition / candidate-conditioned grounding structure rather than adding another weak regularizer.

---

# 14. Baseline checkpoint table for future comparison

| Metric | Core best e3 | Core last e5 |
|---|---:|---:|
| Mean Recall | 27.258 | 26.697 |
| R@10 | 18.154 | 17.847 |
| R@50 | 36.363 | 35.547 |
| Reference-only | 14.438 | 14.843 |
| Best SINGLE | 24.527 | 24.395 |
| Best REPEAT | 27.441 | 26.915 |
| MEAN candidate | 27.308 | 26.681 |
| Support cosine | 0.999795 | 0.999821 |
| Support overlap | 0.995188 | 0.995809 |
| Support fraction | 7.37% | 8.32% |
| t0 Δq cosine | 0.9897 | 0.9923 |
| t0 effect rank | 1.663 | 1.595 |
| t1 effect rank | 2.339 | 2.297 |
| t2 effect rank | 3.440 | 3.440 |
| Mean executed edits | 2.682 | 2.195 |
| STOP hazard t2 | 25.74% | 67.25% |
| Max candidate share | 31.51% | 40.82% |

---

# 15. Current conclusion

The core run demonstrates that the implementation can train and retrieve, but the intended four-way edit decomposition does **not emerge from terminal + marginal utility supervision alone**.

The central structural signature is:

```text
four candidate identities
        ↓
almost identical visual supports
        ↓
almost parallel token edits
        ↓
weakly differentiated candidate consequences
        ↓
repeated edits become progressively less effective
        ↓
STOP becomes increasingly attractive
```

The immediate scientific question for the next experiments is therefore:

> **Can semantic auxiliary losses break candidate symmetry strongly enough that different WHAT representations produce different WHERE groundings and functionally distinct edits?**

The first recommended sequence is:

```text
CORE baseline
→ BIND
→ COMP
→ COMP+BIND
→ FACTOR
```

Do not jump directly to the full six-loss objective. Individual ablations are necessary to identify which supervision actually changes the failure signature.

---

# 16. BIND ablation diagnostic — first structural test

**Experiment:** `objective=bind`, `model.enable_claim_head=true`  
**Objective:**

```math
L = L_terminal + 0.5 L_marginal + 0.01 L_bind
```

**Checkpoint analyzed:** epoch 3 best  
**Mean Recall:** **27.582**

This should be compared against the core baseline best checkpoint at epoch 3, not the degraded epoch-5 checkpoint, because both best checkpoints occur at epoch 3.

## 16.1 Retrieval result

| metric | CORE e3 | BIND e3 | delta |
|---|---:|---:|---:|
| R@10 | 18.154 | 18.411 | +0.257 |
| R@50 | 36.363 | 36.753 | +0.390 |
| Mean Recall | 27.258 | 27.582 | **+0.324** |

`L_bind` gives a **small positive retrieval gain of +0.324 Mean Recall**.

This is useful evidence that the binding signal is not obviously destructive, but the gain is too small to claim that the decomposition problem is solved.

---

# 17. Did BIND solve WHAT specialization?

Only partially.

| diagnostic | CORE e3 | BIND e3 | direction |
|---|---:|---:|---|
| pairwise intent cosine | 0.9476 | 0.9423 | ↓ 0.0053 |
| pairwise context cosine | 0.9549 | 0.9399 | ↓ 0.0150 |
| pairwise delta-Z cosine | 0.9877 | 0.9838 | ↓ 0.0039 |
| pairwise delta-q cosine | 0.8510 | 0.8420 | ↓ 0.0090 |

So BIND does move the representation in the intended direction:

```text
intent slightly less similar
→ context somewhat less similar
→ delta-Z slightly less similar
→ delta-q slightly less similar
```

However, the changes are modest.

The four intent vectors are still highly correlated:

```text
CORE: 0.9476
BIND: 0.9423
```

Thus `L_bind` creates **weak semantic symmetry breaking**, not a clean four-way decomposition.

---

# 18. Did BIND solve WHERE grounding collapse?

**No. This is the most important result of the ablation.**

| grounding diagnostic | CORE e3 | BIND e3 |
|---|---:|---:|
| pairwise support cosine | 0.999795 | 0.999372 |
| pairwise support overlap | 0.995188 | 0.989260 |
| support fraction | 7.37% | **25.17%** |
| effective support size | 13.95 | **47.19** |

BIND substantially changes **how broad** grounding is:

```text
support fraction: 7.37% → 25.17%
effective size:   13.95 → 47.19 tokens
```

but does almost nothing to make the four candidates look at different regions:

```text
support cosine:  0.999795
              →  0.999372

support overlap: 0.995188
              →  0.989260
```

The four support maps remain essentially clones.

This is stronger evidence for the following diagnosis:

> **The current bottleneck is not merely that the text queries lack semantic differentiation. Even after BIND makes intents/contexts somewhat more distinct, the anchor grounder still maps them to almost the same visual support.**

Therefore BIND alone does **not** solve WHERE specialization.

---

# 19. BIND changes edit magnitude more than edit direction

At t0:

| diagnostic | CORE e3 | BIND e3 |
|---|---:|---:|
| mean `||ΔZ||` | 2.067 | **3.581** |
| mean `||Δq||` | 0.3581 | 0.3526 |
| pairwise Δq cosine | 0.9897 | 0.9871 |
| functional rank | 1.663 | 1.736 |

The token-space edit norm jumps strongly:

```text
||ΔZ||: 2.067 → 3.581
```

while the final retrieval-space effect norm barely changes:

```text
||Δq||: 0.3581 → 0.3526
```

This means BIND causes the editor to make **larger token-state modifications**, but those larger modifications are not translated proportionally into more useful or more distinct retrieval-space consequences.

This reinforces the earlier recurrent/readout concern.

---

# 20. Recurrent attenuation still exists under BIND

BIND checkpoint:

| timestep | mean `||Δq||` | pairwise Δq cosine | effect rank |
|---|---:|---:|---:|
| t0 | 0.3526 | 0.9871 | 1.736 |
| t1 | 0.0930 | 0.9350 | 2.422 |
| t2 | 0.0246 | 0.5982 | 3.438 |

t2 has only about:

```text
7.0% of the t0 Δq norm
```

So the core failure:

```text
strong first edit
→ weak later retrieval-space effects
```

is still present.

BIND does not fix recurrent attenuation.

---

# 21. STOP behavior improves, but repeated-action behavior becomes even stronger

| diagnostic | CORE e3 | BIND e3 |
|---|---:|---:|
| mean executed edits | 2.682 | 2.849 |
| STOP hazard t1 | 3.38% | 2.25% |
| STOP hazard t2 | 25.74% | **10.27%** |
| repeated-candidate fraction | 90.99% | **95.40%** |
| max candidate share | 31.51% | 29.78% |

BIND strongly reduces premature STOP:

```text
t2 STOP hazard:
25.74%
→ 10.27%
```

and increases average edit count.

However, **95.40% of validation queries repeat at least one candidate identity**.

This is even higher than the core baseline.

So the network is not using the extra trajectory depth to compose clearly distinct candidate actions. It is frequently applying the same identity again.

This is consistent with the support maps still being clones.

---

# 22. FULL still does not exploit dynamic candidate specialization

| control | CORE e3 | BIND e3 |
|---|---:|---:|
| FULL | 27.258 | 27.582 |
| MEAN | 27.308 | 27.666 |
| best REPEAT | 27.441 | 27.708 |

For BIND:

```text
FULL        = 27.582
MEAN        = 27.666
best REPEAT = 27.708
```

So:

```text
MEAN > FULL by +0.084
best REPEAT > FULL by +0.126
```

The gaps are small, but the qualitative result remains unchanged:

> **Dynamic candidate selection still provides no measurable advantage over a mean candidate or repeatedly applying one fixed candidate.**

Therefore BIND improves retrieval slightly without demonstrating that the intended multi-candidate mechanism has become functionally necessary.

---

# 23. Updated causal interpretation after BIND

The CORE-only hypothesis was:

```text
similar WHAT
→ cloned WHERE
→ parallel edits
→ weak later effects
→ STOP
```

BIND gives us a stronger localization of the failure.

Observed under BIND:

```text
WHAT becomes somewhat more distinct       ✅
context becomes somewhat more distinct    ✅
WHERE is still almost identical           ❌
delta-Z remains highly parallel            ❌
FULL still ≈ MEAN ≈ REPEAT                 ❌
```

Therefore the updated hypothesis is:

```text
BIND
  ↓
partial text/semantic symmetry breaking
  ↓
AnchorGrounder collapses the distinct signals
into nearly the same spatial distribution
  ↓
Grounded evidence remains largely shared
  ↓
Shared editor still produces near-parallel token edits
  ↓
multi-candidate policy has little functional reason to specialize
```

The new evidence shifts suspicion **more strongly toward the WHERE/grounding interface**.

---

# 24. Decision on BIND

## What BIND achieved

- Mean Recall: **+0.324**
- intent cosine decreases slightly;
- context cosine decreases more clearly;
- t0 functional rank improves from 1.663 → 1.736;
- candidate selection becomes slightly more balanced;
- STOP becomes less aggressive.

## What BIND did not achieve

- support maps are still clones;
- support overlap remains ~0.99;
- delta-Z directions remain highly correlated;
- t0 candidate effects remain almost parallel;
- recurrent attenuation remains;
- repeated candidate selection rises to 95.40%;
- MEAN and fixed REPEAT still match/beat FULL.

## Verdict

```text
L_bind = mildly useful auxiliary signal
       ≠ solution to candidate decomposition
```

It should be retained as a promising semantic signal for later combinations, but **BIND alone does not solve the central structural failure**.

---

# 25. What the next ablation should answer

The original plan proposed COMP after BIND.

That is still useful as a controlled experiment, but the BIND result changes what we should look for.

For `L_comp`, the key question is no longer merely:

```text
Does intent cosine decrease?
```

It is:

```text
Can stronger claim complementarity make the intent differences
large enough to survive the AnchorGrounder and produce distinct supports?
```

The decisive metrics are:

1. `pairwise_support_cosine`
2. `pairwise_support_overlap`
3. `pairwise_delta_z_cosine`
4. t0 `functional_effective_rank`
5. `FULL - MEAN`
6. `FULL - best_REPEATED`
7. repeated-candidate fraction

If COMP or COMP+BIND lowers intent/context cosine but support cosine remains around `0.999`, that would be strong evidence that **the architecture needs a direct WHERE-side specialization mechanism**, rather than more text-side semantic losses.

---

# 26. Current experiment scoreboard

| Experiment | Mean Recall | support cosine | support overlap | t0 rank | t0 Δq cosine | mean edits | repeat-query fraction | interpretation |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| CORE best e3 | 27.258 | 0.999795 | 0.995188 | 1.663 | 0.9897 | 2.682 | 90.99% | WHERE clone |
| BIND best e3 | **27.582** | 0.999372 | 0.989260 | **1.736** | 0.9871 | 2.849 | 95.40% | mild WHAT improvement, WHERE still clone |

This table should be extended with COMP, COMP+BIND, FACTOR, and any later architectural interventions.
---

# 27. COMP ablation diagnostic

**Experiment:** `objective=comp`  
**Checkpoint analyzed:** epoch 2 best  
**Mean Recall:** **27.125**

## Retrieval

```text
CORE = 27.258
COMP = 27.125
delta = -0.134
```

COMP does **not** improve retrieval over CORE.

## WHAT

```text
pairwise intent cosine:
CORE = 0.9476
COMP = 0.9636
```

Intent similarity increases, so COMP alone does not create useful semantic separation.

## WHERE

```text
support cosine:
CORE = 0.999795
COMP = 0.999467

support overlap:
CORE = 0.995188
COMP = 0.990262

support fraction:
CORE = 7.37%
COMP = 23.30%
```

The support becomes broader, but the four support maps remain effectively identical.

## Functional behavior

```text
t0 effect rank:
CORE = 1.663
COMP = 1.702

t0 Δq cosine:
CORE = 0.9897
COMP = 0.9884

FULL        = 27.125
MEAN        = 27.125
best REPEAT = 27.233
```

### Verdict

**COMP alone is not promising as the primary specialization loss.** It neither improves retrieval nor breaks WHERE collapse.

---

# 28. COMP+BIND ablation diagnostic

**Checkpoint analyzed:** epoch 3 best  
**Mean Recall:** **27.492**

## Retrieval

```text
CORE      = 27.258
BIND      = 27.582
COMP+BIND = 27.492

COMP+BIND vs CORE = +0.234
COMP+BIND vs BIND = -0.090
```

COMP+BIND is slightly better than CORE, but worse than BIND alone.

## WHAT

```text
intent cosine:
CORE      = 0.9476
BIND      = 0.9423
COMP      = 0.9636
COMP+BIND = 0.9347
```

This is the strongest text-side semantic separation among the semantic-loss runs.

## WHERE

```text
support cosine:
CORE      = 0.999795
BIND      = 0.999372
COMP+BIND = 0.999901

support overlap:
COMP+BIND = 0.996045
```

Despite stronger WHAT separation, WHERE remains essentially fully cloned.

This is the clearest evidence so far for a **WHAT→WHERE bottleneck**:

```text
different text-side candidate representations
                 ↓
almost identical spatial support distributions
```

## Functional behavior

```text
t0 effect rank:
CORE      = 1.663
BIND      = 1.736
COMP+BIND = 1.704

FULL        = 27.492
MEAN        = 27.427
best REPEAT = 27.535
```

### Verdict

**COMP+BIND successfully changes WHAT more than BIND alone, but those differences are destroyed by the grounding stage.** It therefore localizes the failure rather than solving it.

---

# 29. FACTOR ablation diagnostic

**Checkpoint analyzed:** epoch 2 best  
**Mean Recall:** **27.315**

## Retrieval

```text
CORE   = 27.258
FACTOR = 27.315
delta  = +0.057
```

FACTOR gives only a very small gain over CORE and remains below BIND.

## Structure

```text
intent cosine:
CORE   = 0.9476
FACTOR = 0.9569

delta-Z cosine:
CORE   = 0.9877
FACTOR = 0.9839

t0 effect rank:
CORE   = 1.663
FACTOR = 1.752
```

FACTOR slightly improves t0 rank, but the edit pathways remain highly correlated.

## WHERE

```text
support cosine:
CORE   = 0.999795
FACTOR = 0.999750

support overlap:
CORE   = 0.995188
FACTOR = 0.995210
```

WHERE collapse remains intact.

## Selection

```text
max candidate share       = 49.40%
mean executed edits       = 2.904
repeated-candidate fraction = 96.71%

FULL        = 27.315
MEAN        = 27.332
best REPEAT = 27.350
```

FACTOR also creates the strongest candidate-selection skew among the tested objectives without creating corresponding spatial specialization.

### Verdict

**FACTOR does not solve the central structural failure.**

---

# 30. Five-run scoreboard

| Run | Best ep | Mean Recall | Intent cos | Support cos | Support overlap | Support frac | ΔZ cos | t0 rank | t0 Δq cos | Mean edits | Repeated cand. | Max cand. share | FULL−MEAN | FULL−best REPEAT |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CORE | 3 | 27.258 | 0.9476 | 0.999795 | 0.995188 | 7.37% | 0.9877 | 1.663 | 0.9897 | 2.682 | 90.99% | 31.51% | -0.050 | -0.182 |
| BIND | 3 | **27.582** | 0.9423 | 0.999372 | 0.989260 | 25.17% | 0.9838 | 1.736 | 0.9871 | 2.849 | 95.40% | 29.78% | -0.084 | -0.126 |
| COMP | 2 | 27.125 | 0.9636 | 0.999467 | 0.990262 | 23.30% | 0.9862 | 1.702 | 0.9884 | 2.931 | 97.52% | 33.26% | -0.001 | -0.108 |
| COMP+BIND | 3 | 27.492 | **0.9347** | 0.999901 | 0.996045 | 8.08% | 0.9859 | 1.704 | 0.9883 | 2.770 | 94.00% | 32.27% | +0.065 | -0.043 |
| FACTOR | 2 | 27.315 | 0.9569 | 0.999750 | 0.995210 | 9.59% | 0.9839 | **1.752** | 0.9859 | 2.904 | 96.71% | **49.40%** | -0.017 | -0.035 |

---

# 31. Updated scientific conclusion

The five objectives now give a consistent picture.

## 1. BIND is the best current loss for retrieval

```text
BIND      = 27.582
COMP+BIND = 27.492
FACTOR    = 27.315
CORE      = 27.258
COMP      = 27.125
```

BIND is therefore worth keeping as a semantic regularizer.

## 2. Text-side differentiation is possible

COMP+BIND produces the lowest intent cosine among these runs.

So the text side is not completely incapable of specialization.

## 3. WHERE is the persistent bottleneck

Across all five runs:

```text
support cosine ≈ 0.9994–0.9999
support overlap ≈ 0.989–0.996
```

The exact support breadth moves a lot, but support identity does not.

This means the current AnchorGrounder mostly converts candidate differences into changes in **sharpness**, not changes in **spatial target**.

## 4. Dynamic candidate selection is still not necessary

For every objective:

```text
FULL ≈ MEAN ≈ best fixed REPEAT
```

Therefore candidate sequencing is not yet exploiting genuinely complementary actions.

## 5. Repeated candidate use remains extreme

```text
CORE      = 90.99%
BIND      = 95.40%
COMP      = 97.52%
COMP+BIND = 94.00%
FACTOR    = 96.71%
```

The network repeatedly reuses the same action identity instead of composing distinct edits.

---

# 32. Main diagnosis after five runs

The evidence now supports this failure chain:

```text
Text-side candidates can become somewhat different
                    ↓
AnchorGrounder contracts those differences
                    ↓
P1 ≈ P2 ≈ P3 ≈ P4
                    ↓
grounded evidence remains shared
                    ↓
shared editor produces near-parallel token edits
                    ↓
candidate consequences remain weakly differentiated
                    ↓
sequence identity matters little
                    ↓
MEAN / REPEAT ≈ FULL
```

The next serious research target should therefore be **the WHAT→WHERE grounding interface**, not another weak text-side diversity regularizer.

The key research question is:

> How can different semantic candidate claims produce distinct but semantically justified visual supports, while still allowing overlap when two edits truly concern the same region?

Any next architectural intervention should be evaluated against the exact five-run scoreboard above.
