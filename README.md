# A6 / R1 — FG-CLIP2 + Token-Axis Entmax-1.5
## Branch Diagnosis and Experiment Checkpoint

**Repository:** `Le-Minh-Nhut/cir`  
**Branch:** `exp/e2e-a6-fgclip2-entmax`  
**Audited remote HEAD:** `8be6337517d5dedd2c13e1f7c1feb14b16cd1ebc` (`v2`)  
**Backbone:** `qihoo360/fg-clip2-large`  
**Experiment:** R1 — token-axis Entmax-1.5 sparse routing after QASA selection  
**Dataset / protocol used by the reported run:** FashionIQ validation, `fashioniq_original`  
**Primary training objective:** retrieval loss only  
**Number of Edit Slots:** 4  
**NULL / dustbin slot:** none  

---

# 1. Executive conclusion

This branch answers a narrow but important question:

> If every active Edit Slot is forced to consume only a sparse, adaptive subset of text tokens instead of pooling densely from most of the sentence, does the previous giant-slot / slot-collapse failure disappear?

The answer is **partially yes**.

R1 produces a large and useful change:

- retrieval improves strongly;
- actual token routing becomes sparse;
- the previous near-total **soft-mass monopoly** disappears;
- a slot no longer needs to read most/all of the sentence to be useful.

However, R1 does **not** solve the deeper decomposition problem:

- slot `S0` is still the dominant soft-ownership slot in `6015 / 6016` validation examples;
- QASA selects `S0` in `6016 / 6016` examples;
- sparse token supports across simultaneously active slots overlap heavily (`Jaccard ≈ 0.680`);
- removing `S0` causes a `-10.24` mean-recall drop;
- removing `S1`, `S2`, or `S3` causes only `-0.68`, `-0.16`, and `-0.13`.

Therefore the correct diagnosis is:

> **R1 largely fixes evidence-density collapse, but functional slot collapse remains.**

Or more simply:

```text
Before R1:
one slot reads almost everything
and does almost everything.

After R1:
one slot reads only a small subset,
but still does most of the useful work.
```

This is a materially better failure mode, and it provides direct empirical motivation for moving from **independent per-slot sparse selection** to a **joint token-slot assignment mechanism** such as R4/QI-SCA.

---

# 2. Why this branch exists

The preceding FG-CLIP2 experiment showed a rapid winner-take-all collapse in the token-to-slot ownership stage.

Even though retrieval kept improving, the learned decomposition converged toward:

```text
all useful modification evidence
            ↓
       one Edit Slot
            ↓
       QASA selects it
            ↓
          Executor
            ↓
      retrieval improves
```

The central R1 hypothesis was deliberately smaller than "solve slot specialization":

> Perhaps each slot becomes global because dense pooling lets it consume too much of the sentence.  
> If each slot receives an adaptive exact-zero token subset, the model may stop relying on one global sentence summary.

R1 therefore tests **sparse evidence consumption only**.

It does not add balancing losses, semantic labels, capacity constraints, NULL slots, top-k routing, OT, or a new supervision signal.

---

# 3. Exact R1 architectural change

The branch preserves two distinct views of the same text-to-slot logits.

Let:

- `B` = batch size,
- `L` = number of Edit Slots,
- `N` = text-token count,
- `Z ∈ R^(B×L×N)` = learned token-slot ownership logits.

## 3.1 Pre-sparse soft competition — preserved for QASA

The original competitive ownership remains:

\[
P^{soft}_{b,k,n}
=
\operatorname{softmax}_{k}(Z_{b,k,n}).
\]

The softmax is across the **slot axis**.

For every valid content token:

\[
\sum_k P^{soft}_{b,k,n}=1.
\]

This tensor remains available as:

```text
slot_masks
```

and its mass:

```text
slot_mass
```

is still used for the original ownership-collapse diagnostics.

This is important: R1 does **not** redefine the old diagnostics so that the results look artificially better.

---

## 3.2 QASA remains pre-sparse

QASA is run from the original soft competitive view.

Conceptually:

```text
ownership logits
      ↓
softmax across slots
      ↓
pre-sparse soft competition
      ↓
QASA
      ↓
selected_mask
```

R1 Entmax routing is downstream of this selection.

Therefore QASA answers:

> Which real Edit Slots are allowed to participate?

while Entmax answers:

> Which content tokens does each selected slot actually consume?

There is no post-QASA probability renormalization that would inflate surviving-slot confidence.

---

## 3.3 R1 sparse token routing

For each QASA-selected slot `k`, R1 applies fixed Entmax-1.5 across the **token axis**:

\[
R_{b,k,:}
=
\operatorname{Entmax}_{1.5}(Z_{b,k,:}).
\]

In code-equivalent form:

```python
routing = entmax15(masked_logits, dim=-1)
```

because `ownership_logits` has shape:

```text
[B, L, N]
       ^
     tokens
```

Invalid/padding/special tokens are masked out using the repository's canonical content-token mask.

QASA-unselected slots receive exactly zero routing mass.

The important distinction is:

```text
SOFT OWNERSHIP VIEW
-------------------
normalize across slots
used by QASA
used by old collapse diagnostics


R1 ROUTING VIEW
---------------
normalize/sparsify across tokens
used to build actual Edit Slots
used by Executor
```

---

## 3.4 Actual Edit Slots now come from Entmax routing

The R1 Edit Slot is pooled from `routing_masks`, not from dense `slot_masks`.

Conceptually:

\[
m_k^{route} = \sum_n R_{k,n}
\]

\[
s_k^{route}
=
\frac{\sum_n R_{k,n} h_n}
{\max(m_k^{route},1)}
\]

followed by the existing activity handling.

Thus the actual downstream path is:

```text
text token states
      ↓
ownership logits
      ↓
QASA selection
      ↓
token-axis Entmax-1.5
      ↓
sparse routing_masks
      ↓
sparse slot pooling
      ↓
Edit Slots
      ↓
Executor
      ↓
query
      ↓
retrieval loss
```

---

# 4. What R1 deliberately does NOT change

This experiment is intentionally isolated.

R1 does not introduce:

- explicit NULL/dustbin ownership;
- fixed top-k token selection;
- token-slot capacity;
- adaptive capacity;
- learned Entmax alpha;
- load-balancing loss;
- diversity loss;
- orthogonality loss;
- anti-collapse auxiliary loss;
- semantic slot labels;
- Sinkhorn / OT;
- R4 QI-SCA;
- a new retrieval objective.

The only intended scientific intervention is:

> **actual Edit Slots consume token-axis Entmax-1.5 sparse support.**

---

# 5. Experiment configuration

The audited branch config uses:

```yaml
batch_size: 64
eval_batch_size: 32
num_workers: 8
num_epochs: 10

lr: 1.0e-4
weight_decay: 1.0e-4

loss_weights:
  retrieval_loss: 1.0

model:
  text_dim: 1024
  reference_dim: 1024
  query_dim: 1024
  slot_dim: 1024
  state_dim: 512
  num_slots: 4
  num_primitives: 8

  mask_temperature: 1.0
  router_temperature: 1.0
  retrieval_temperature: 0.07

  qasa_tau: 0.5
  qasa_rho: 0.8
  qasa_mu: 0.3
  qasa_eps: 1.0e-8
  qasa_apply_at_eval: true
```

Text cache used by the reported run:

```text
Correction policy: fashioniq
Text cache subdirectory: text
Train text shape: (18000, 64, 1024)
Val text shape:   (6016, 64, 1024)
```

The run emitted warnings that the existing corrected text-cache manifest is legacy and lacks an explicit `correction_policy`; the loader treated it as `fashioniq`.

This warning is a cache-metadata compatibility warning, not an Entmax failure.

---

# 6. Pre-R1 baseline

The supplied pre-Entmax baseline used the same general FG-CLIP2/QASA/Executor family but pooled from the dense soft ownership.

## 6.1 Baseline training trajectory

| Epoch | Loss | Mean Recall | Active Slots | Hard Active | Dominant | Monopoly | QASA K | QASA Q | QASA Cov |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.6230 | 12.6373 | 2.00 | 2.11 | 0.269 | 0.000 | 2.11 | 0.253 | 0.984 |
| 2 | 2.1097 | 16.2211 | 1.60 | 1.74 | 0.567 | 0.257 | 1.74 | 0.277 | 0.986 |
| 3 | 1.8804 | 18.7428 | 1.03 | 1.02 | 0.984 | 0.959 | 1.02 | 0.252 | 0.999 |
| 4 | 1.7140 | 19.8364 | 1.05 | 1.02 | 0.985 | 0.955 | 1.02 | 0.256 | 0.997 |
| 5 | 1.6030 | 22.9903 | 1.00 | 1.00 | 0.998 | 1.000 | 1.00 | 0.251 | 1.000 |
| 6 | 1.5066 | 24.1009 | 1.32 | 1.14 | 0.972 | 0.948 | 1.14 | 0.325 | 0.986 |
| 7 | 1.4297 | 24.2163 | 1.40 | 1.10 | 0.951 | 0.869 | 1.10 | 0.342 | 0.975 |
| 8 | 1.3711 | 25.7875 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 9 | 1.3247 | 26.3065 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |
| 10 | 1.2632 | 26.3314 | 1.00 | 1.00 | 1.000 | 1.000 | 1.00 | 0.250 | 1.000 |

The failure was extremely clear:

```text
Epoch 1:
dominant = 0.269
monopoly = 0.000

Epoch 3:
dominant = 0.984
monopoly = 0.959

Epoch 5:
dominant = 0.998
monopoly = 1.000

Epoch 10:
dominant = 1.000
monopoly = 1.000
```

Retrieval continued improving despite this collapse.

Thus dense ownership allowed a stable shortcut in which one slot could absorb essentially the whole modification and remain sufficient for retrieval.

---

# 7. R1 training result

The Entmax run reached:

```text
Epoch 10/10
loss        = 1.0813
mean_recall = 32.2703
active_slots= 1.13
hard_active = 1.47
dominant    = 0.568
monopoly    = 0.000
qasa_k      = 1.47
qasa_q      = 0.252
qasa_cov    = 0.958

routing_support_mean ≈ 4.46
routing_support_max  ≈ 12.10
routing_overlap_mean ≈ 0.667
routing_slot_support ≈ [3.77, 5.60, 7.20, 5.35]
```

The best checkpoint was the final epoch:

```text
best mean_recall = 32.2703
```

---

# 8. Direct baseline vs R1 comparison

| Metric | Pre-R1 baseline | R1 Entmax | Change |
|---|---:|---:|---:|
| Mean Recall | 26.3314 | **32.2703** | **+5.9389** |
| Relative Mean Recall | 1.000× | **1.2256×** | **+22.56%** |
| Active slots | 1.00 | 1.13 | +0.13 |
| Hard active / QASA K | 1.00 | 1.47 | +0.47 |
| Soft dominant share | 1.000 | **0.568** | -0.432 |
| Near-monopoly fraction | 1.000 | **0.000** | -1.000 |
| QASA coverage | 1.000 | 0.958 | -0.042 |
| Actual sparse support | dense baseline | **~4.46 tokens/active slot** | sparse |

The retrieval gain is large for a single run:

\[
32.2703 - 26.3314 = +5.9389
\]

absolute mean-recall points.

Relative to the baseline:

\[
\frac{32.2703}{26.3314} - 1
\approx 22.56\%.
\]

This is strong evidence that sparse token consumption is useful in this setup.

However, this is still one reported seed/run; it should not be interpreted as a statistically established improvement until replicated.

---

# 9. Full validation forensic result

A dedicated R1 diagnostic was run on the trained checkpoint.

## 9.1 Retrieval

```text
Full retrieval:
R@10 = 22.12
R@50 = 42.42
Mean = 32.27

Reference only:
R@10 =  8.10
R@50 = 21.04
Mean = 14.57
```

The full modification path therefore adds approximately:

\[
32.27 - 14.57 = 17.70
\]

mean-recall points over reference-only retrieval.

This confirms that the modification branch remains functionally important.

---

# 10. Soft ownership forensic

```text
valid_content_tokens  = 11.9214
soft_active_slots     = 1.1411
soft_dominant_share   = 0.5591
soft_near_monopoly    = 0.0000
```

At first glance this looks much healthier than the pre-R1 collapse because:

```text
dominant ≈ 0.56
monopoly = 0
```

rather than:

```text
dominant ≈ 1.00
monopoly = 1
```

However the dominant-slot identity reveals a more subtle collapse:

```text
soft dominant slot counts:
S0 = 6015
S1 =    0
S2 =    1
S3 =    0
```

across `6016` validation examples.

Therefore:

\[
P(\text{S0 is dominant}) \approx \frac{6015}{6016} \approx 99.983\%.
\]

This is not the old **probability-mass monopoly**, but it is still an extreme **winner-identity monopoly**.

The current regime is approximately:

```text
S0: usually only moderately larger than the others
    but almost always the largest

S1/S2/S3:
receive non-trivial soft probability mass
but almost never become the strongest owner
```

This explains why:

```text
soft_dominant_share ≈ 0.56
```

can coexist with:

```text
S0 dominant in 6015 / 6016 examples.
```

This distinction is important.

---

# 11. QASA forensic

```text
qasa_selected_slots         = 1.5002
qasa_quality_all_slots      = 0.2523
qasa_quality_selected_slots = 0.7538
qasa_final_coverage         = 0.9586
```

Selection frequency:

```text
S0 selected: 6016 / 6016 = 100.0%
S1 selected: 2261 / 6016 ≈ 37.58%
S2 selected:  262 / 6016 ≈  4.36%
S3 selected:  486 / 6016 ≈  8.08%
```

Important interpretation:

QASA does **not** always collapse to a single selected slot.

On average it selects approximately `1.5` slots.

It also gives the selected set a high average quality:

```text
selected-slot quality ≈ 0.754
```

Therefore the main remaining failure cannot be summarized as:

> "QASA always throws away every slot except S0."

Instead:

> QASA sometimes keeps additional slots, but those extra slots contribute little to final retrieval.

This points downstream toward **functional redundancy / asymmetric utility**, not merely a selection-count problem.

---

# 12. Actual Entmax routing forensic

The actual routing statistics are:

```text
routing_active_slots          = 1.5002
routing_support_mean          = 4.4127 tokens
routing_support_fraction_mean = 0.3767
routing_zero_fraction         = 0.6233
routing_support_jaccard       = 0.6803
```

The mean instruction contains:

```text
11.9214 valid content tokens
```

while an active slot consumes only:

```text
4.4127 tokens on average
```

or approximately:

\[
\frac{4.4127}{11.9214} \approx 37.0\%.
\]

The diagnostic reports a directly averaged support fraction of:

```text
37.67%
```

and:

```text
62.33%
```

of valid slot-token routing positions are zero.

This is a clear R1 success.

The actual Edit Slots no longer require dense access to the sentence.

---

# 13. Per-slot sparse support

Conditional on each slot being active:

| Slot | Mean support | Mean support fraction |
|---|---:|---:|
| S0 | 3.894 | 0.337 |
| S1 | 5.840 | 0.487 |
| S2 | 6.504 | 0.453 |
| S3 | 7.630 | 0.478 |

An important observation:

> `S0` is **not** the slot reading the most tokens.

In fact, when active, `S0` reads the smallest support among the four slots.

Yet `S0` is by far the most functionally important slot.

Therefore the remaining "giant slot" failure is no longer equivalent to:

```text
giant slot = slot with largest token support
```

R1 separates two notions that were previously entangled.

---

# 14. Two different giant-slot failures

The experiment now gives a useful conceptual distinction.

## 14.1 Evidence giant

Definition:

> A slot consumes most/all of the sentence.

Pre-R1 this was severe.

R1 result:

```text
routing support ≈ 37.7% of valid tokens
routing zero fraction ≈ 62.3%
```

Verdict:

> **Strongly reduced / largely solved by R1.**

---

## 14.2 Functional giant

Definition:

> One slot carries most of the useful downstream modification computation.

This is tested directly by slot-drop ablation.

R1 result:

```text
Full mean recall = 32.270
```

Drop each slot:

```text
Drop S0 = 22.030   delta = -10.240
Drop S1 = 31.595   delta =  -0.675
Drop S2 = 32.110   delta =  -0.160
Drop S3 = 32.136   delta =  -0.134
```

Verdict:

> **Still severe.**

Thus R1 transforms the failure from:

```text
S0 sees almost everything
and does almost everything
```

into:

```text
S0 sees only a small sparse subset
but still does most of the useful work.
```

This is a substantially more informative failure mode.

---

# 15. Functional slot-drop interpretation

The strongest evidence of remaining collapse is the slot-drop test.

## 15.1 S0

```text
Full   = 32.270
Drop S0= 22.030

Δ = -10.240
```

Removing S0 destroys almost one third of the full mean-recall score in absolute terms.

S0 is clearly critical.

## 15.2 Other slots

```text
Drop S1: -0.675
Drop S2: -0.160
Drop S3: -0.134
```

These changes are tiny relative to the S0 ablation.

The slot-drop effects are **not additive causal contributions** because the executor is nonlinear and slots can compensate for one another.

Therefore one must not sum these deltas and interpret them as a decomposition of total retrieval performance.

Nevertheless, their scale asymmetry is too large to ignore.

The model remains functionally dominated by S0.

---

# 16. Reference-only comparison provides useful nuance

Reference-only mean recall is:

```text
14.57
```

After dropping S0:

```text
22.03
```

So even without S0, the model remains:

\[
22.03 - 14.57 = 7.46
\]

points above reference-only retrieval.

Therefore it would be too strong to say:

> "S1/S2/S3 contain absolutely no useful modification information."

They collectively still support a non-trivial modification signal.

The more accurate conclusion is:

> **useful modification information exists outside S0, but S0 remains overwhelmingly the most important individual slot.**

---

# 17. Routing overlap is the next major structural problem

The sparse support Jaccard is:

```text
0.6803
```

This is high.

Entmax makes each slot sparse independently, but it does not enforce cross-slot exclusivity.

Therefore this remains legal:

```text
token A
├── selected by S0
├── selected by S1
└── selected by S3
```

A stylized failure could look like:

```text
instruction:
"make the red dress have longer sleeves"

S0: red, dress, longer
S1: red, dress, longer, sleeves
S2: dress, longer, sleeves
```

Each slot is sparse.

But the slots are not necessarily decomposing different evidence.

This is exactly the limitation expected from independent token-axis Entmax.

---

# 18. Why the old soft diagnostics alone would have been misleading

If only these metrics were observed:

```text
dominant = 0.559
monopoly = 0.000
```

one might conclude that slot collapse was solved.

That conclusion would be wrong.

The full forensic suite shows:

```text
dominant slot identity:
S0 = 6015 / 6016

functional slot-drop:
S0 = -10.24
others = -0.13 to -0.68

routing overlap:
0.680
```

Thus a model can avoid a >90% soft-mass monopoly while still learning a highly asymmetric decomposition.

Future experiments must therefore report at least three distinct families of diagnostics:

```text
1. soft ownership statistics
2. sparse support / routing statistics
3. functional ablation statistics
```

No single family is sufficient.

---

# 19. Hypothesis verdicts

## H1 — Dense sentence access contributes to the previous failure

**Supported.**

R1 reduces actual support to about `4.4 / 11.9` content tokens and strongly improves retrieval.

The intervention and improvement are consistent with the hypothesis that dense token averaging was harmful.

Because this is one run, it is evidence rather than final statistical proof.

---

## H2 — Exact-zero adaptive sparsity alone eliminates the giant-slot problem

**Rejected if "giant slot" means functional dominance.**

R1 eliminates the old dense-evidence giant but S0 remains functionally dominant.

---

## H3 — Entmax alone creates useful multi-slot decomposition

**Rejected.**

The slots remain highly asymmetric and their support overlap is high.

---

## H4 — QASA alone can recover specialization after sparse routing

**Not supported.**

QASA keeps approximately `1.5` slots on average, but the additional slots have very small individual functional ablation effects.

---

## H5 — Sparse routing can improve CIR performance even without successful slot specialization

**Supported by this run.**

Mean recall improves from:

```text
26.3314 → 32.2703
```

even though strong functional asymmetry remains.

This is an important result:

> better retrieval and better decomposition are related but not identical goals.

---

# 20. What R1 taught us about the optimization failure

The pre-R1 mental model was:

```text
one slot wins
    ↓
it receives more tokens
    ↓
it builds a more complete sentence representation
    ↓
it becomes more useful
    ↓
it wins even harder
```

R1 breaks an important part of this loop:

```text
winning slot
    ↓
cannot simply average the whole sentence
    ↓
must choose sparse evidence
```

This is likely why the old soft probability monopoly is reduced.

But a second shortcut remains:

```text
S0 consistently discovers the most useful token subset
        ↓
S0 becomes the safest downstream edit representation
        ↓
Executor relies mostly on S0
        ↓
retrieval gradients continue rewarding S0
        ↓
other slots remain secondary/redundant
```

R1 has no mechanism saying:

```text
"that evidence is already claimed by another slot"
```

or:

```text
"you are using the same evidence as another slot"
```

or:

```text
"one slot should pay a cost for absorbing too much total assignment."
```

This is the missing pressure.

---

# 21. Why simply making Entmax even sparser is not the obvious next step

One possible response would be:

> increase sparsity further.

The forensic data argues against treating this as the main problem.

S0 currently uses only:

```text
3.894 tokens
≈ 33.7% of the instruction when active
```

yet it is still dominant in nearly every sample and has the largest functional effect by far.

Therefore:

```text
"make S0 read fewer tokens"
```

is no longer sufficient as the primary theory.

The next mechanism should address:

```text
who gets which evidence
```

rather than only:

```text
how many tokens each slot sees.
```

This is why R2 learned-alpha is not the most informative immediate experiment.

---

# 22. Implication for R4 / QI-SCA

The R1 failure directly motivates **joint sparse assignment**.

R1 currently behaves conceptually as:

```text
S0 independently chooses a sparse token set
S1 independently chooses a sparse token set
S2 independently chooses a sparse token set
S3 independently chooses a sparse token set
```

Nothing prevents duplicate claims.

R4 instead proposes one global token-slot assignment matrix:

\[
A \in \mathbb{R}_{\ge 0}^{N\times L}
\]

with constraints such as:

\[
\sum_k A_{n,k} \le 1
\]

and:

\[
\sum_n A_{n,k} \le c_k.
\]

The first constraint creates direct token-level competition:

> a token has limited assignment mass that must be shared among slots.

The second adds a congestion/capacity mechanism:

> a slot cannot absorb unlimited total routing mass.

Together with implicit rejection:

\[
q_n = 1-\sum_k A_{n,k},
\]

a token may also remain unassigned.

---

# 23. Important warning for R4 design

The R1 result also changes how R4 should be interpreted.

A naive story would be:

> "R4 is needed because S0 still reads too many tokens."

That is **not** what the data says.

S0 only reads about `3.9` tokens when active.

Therefore a fixed slot capacity larger than this may do almost nothing.

The real justification for R4 is now primarily:

1. **joint token competition**;
2. **reduced duplicate claims**;
3. **implicit rejection**;
4. **congestion feedback when a slot starts absorbing too much assignment mass**.

Capacity alone is not guaranteed to solve functional dominance.

R4 must be evaluated as a joint assignment mechanism, not merely as "Entmax with a smaller maximum token count."

---

# 24. What R4 must beat

R1 establishes a much stronger control than the previous dense baseline.

Any R4 experiment should compare against:

```text
R1 Mean Recall:
32.2703

R1 routing support:
4.4127 tokens

R1 support fraction:
0.3767

R1 zero fraction:
0.6233

R1 support Jaccard:
0.6803

R1 soft dominant share:
0.5591

R1 soft dominant identity:
S0 = 6015 / 6016

R1 QASA selected slots:
1.5002

R1 slot-drop:
S0 = -10.240
S1 = -0.675
S2 = -0.160
S3 = -0.134
```

R4 is not successful merely because it is sparse.

R1 already achieved sparsity.

R4 should demonstrate improvement in **assignment structure and functional non-redundancy** without destroying retrieval.

---

# 25. Recommended R4 success criteria

These should be treated as directional scientific criteria, not arbitrary hard thresholds.

A better model should ideally show:

### Retrieval

```text
mean recall remains competitive with R1
```

Preferably no large regression from `32.27`.

### Evidence sparsity

```text
support remains substantially below full sentence length
```

R1 already gives a strong baseline of approximately `37.7%`.

### Cross-slot redundancy

```text
routing support Jaccard decreases from 0.680
```

This is one of the clearest structural targets.

### Slot identity monopoly

```text
one fixed slot should no longer be dominant in ~100% of examples
```

R1 currently has S0 dominant in `6015 / 6016`.

### Functional non-redundancy

Slot-drop effects should become less extremely concentrated in S0.

The goal is not equal effects by force.

The goal is that multiple slots measurably matter on the examples where they are active.

### QASA consistency

Slots selected by QASA should have downstream functional relevance more often.

R1 currently selects additional slots much more frequently than their individual slot-drop effects would suggest.

---

# 26. Additional diagnostics that should be preserved in all future branches

Do not remove the following R1 diagnostics when implementing R4.

## Pre-sparse ownership

- ownership active-slot count;
- dominant soft-mass share;
- near-monopoly fraction;
- dominant slot identity frequency;
- per-slot winner counts.

## QASA

- selected slot count;
- quality across all slots;
- quality across selected slots;
- coverage;
- per-slot selection frequency.

## Sparse routing

- support mean;
- support max;
- support fraction;
- zero fraction;
- support Jaccard;
- per-slot support conditional on activity.

## Functional

- full retrieval;
- reference-only retrieval;
- single-slot drop;
- query change after slot drop.

These metrics diagnose different failure modes and should not be collapsed into one "slot diversity" number.

---

# 27. Current scientific status of the branch

```text
R1 / TOKEN-AXIS ENTMAX-1.5
==========================================

Mechanical implementation:
PASS

Content-token masking:
PASS

QASA remains pre-sparse:
PASS

Actual Edit Slots use sparse routing:
PASS

Exact-zero routing:
PASS

Dense-evidence giant:
STRONGLY REDUCED

Soft >90% monopoly:
RESOLVED IN THIS RUN

Retrieval:
STRONG IMPROVEMENT

Winner-identity monopoly:
FAILED
S0 dominates 6015 / 6016 examples

Cross-slot support independence:
FAILED / HIGH REDUNDANCY
Jaccard = 0.6803

Functional specialization:
FAILED
Drop S0 = -10.24
Drop others <= -0.675

Overall:
R1 IS A SUCCESSFUL ABLATION,
BUT NOT A COMPLETE DECOMPOSITION SOLUTION.
```

---

# 28. Key conceptual result

The most useful result from this branch is not simply the `+5.94` retrieval gain.

It is the separation of two previously conflated failures:

\[
\boxed{
\text{evidence density}
\neq
\text{functional dominance}
}
\]

R1 shows that a slot can become sparse without becoming non-dominant.

Specifically:

```text
S0 reads only ~3.9 tokens
but remains dominant in almost every example
and causes a -10.24 recall drop when removed.
```

Therefore future work should not optimize only for "few tokens per slot."

It must address **competitive allocation of useful evidence and functional redundancy across slots**.

---

# 29. Final verdict

The branch should be preserved as a strong control/checkpoint.

Its result is:

> **Token-axis Entmax-1.5 is useful and materially improves retrieval while eliminating the previous dense soft-mass monopoly. However, independent sparse routing does not produce functional multi-slot decomposition. The remaining failure is dominated by S0 identity monopoly, high cross-slot support overlap, and extreme slot-drop asymmetry.**

The next justified experiment is a joint sparse assignment mechanism such as R4/QI-SCA.

The motivation for R4 is no longer vague.

It is empirically tied to three observed R1 failure signals:

```text
1. S0 dominant in 6015 / 6016 examples
2. routing support Jaccard = 0.6803
3. Drop S0 = -10.24 while all other drops are < -0.7
```

Those are the failure modes the next experiment must attack.

---

# 30. Reproduction commands

## Train R1

```bash
python src/train.py experiment=taper_e2e dataset.root=<FASHIONIQ_ROOT>
```

## Run R1 forensic diagnosis

Using the dedicated R1 diagnostic script:

```bash
python src/diagnose_taper_r1_entmax.py \
  --checkpoint outputs/<RUN>/best.pt \
  --dataset-root <FASHIONIQ_ROOT>
```

Expected report:

```text
reports/taper_r1_entmax_diagnosis.json
```

A quick smoke diagnosis can be run with:

```bash
python src/diagnose_taper_r1_entmax.py \
  --checkpoint outputs/<RUN>/best.pt \
  --dataset-root <FASHIONIQ_ROOT> \
  --max-queries-per-category 256
```

---

# 31. Provenance of this checkpoint

This README is based on:

1. the audited remote branch `exp/e2e-a6-fgclip2-entmax`;
2. remote HEAD `8be6337517d5dedd2c13e1f7c1feb14b16cd1ebc`;
3. the branch's R1 Entmax/QASA implementation and diagnostics;
4. the supplied 10-epoch pre-R1 baseline log;
5. the supplied 10-epoch R1 training log;
6. the full R1 forensic output over `6016` validation queries.

The causal interpretations above are explicitly marked as interpretations.  
The numerical results are the observed values from the supplied runs.
