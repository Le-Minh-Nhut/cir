# A8.0 Diagnostic README — FG-CLIP2 Shared Entity–Action Binding

**Branch:** `exp/e2e-a8.0-fgclip2-encoder-binding`  
**Experiment:** A8.0 — Minimal ENCODER-style Shared Entity–Action Binding  
**Backbone:** `qihoo360/fg-clip2-large`  
**Backbone revision:** `4d1d5dc35c716902f07c172dbfc23b82a7bc6bf3`  
**Dataset:** FashionIQ  
**Relations:** `K = 4`  
**Best checkpoint:** epoch 6  
**Checkpoint diagnosed:** `outputs/2026-08-28/02-35-59/best.pt`

---

## 1. Research question

A8.0 asks one deliberately narrow question:

> Can one shared bank of Entity–Action relation queries, inspired by ENCODER, naturally produce useful and functionally specialized edit decomposition on frozen FG-CLIP2 representations?

This branch intentionally excludes TAPER, QASA, NULL/dustbin routing, competitive ownership, recurrent editing, CSMCIR, teacher supervision, LLM-generated supervision, and the ENCODER LFF/ScoreNet/FactorNet stack.

So A8.0 is **not a full ENCODER reproduction**. It is a minimal causal ablation of the shared Entity–Action binding idea.

---

## 2. Architecture under diagnosis

Frozen FG-CLIP2 provides dense image tokens and contextual text tokens:

\[
V\in\mathbb{R}^{N_v\times1024},
\qquad
T\in\mathbb{R}^{N_t\times1024}.
\]

A single learned relation-query bank is shared across modalities:

\[
R=\{r_1,\dots,r_K\},\qquad K=4.
\]

The same binder produces:

\[
e_k=\mathrm{Bind}(r_k,V),
\qquad
a_k=\mathrm{Bind}(r_k,T).
\]

The reference and text token sets are:

\[
[g_I,e_1,e_2,e_3,e_4],
\qquad
[g_T,a_1,a_2,a_3,a_4].
\]

Before affine fusion, all composition-interface tokens are L2-normalized:

\[
\|g_I\|_2=\|g_T\|_2=\|e_k\|_2=\|a_k\|_2=1.
\]

The paired tokens are fused with a feature-wise affine transform and mean-pooled into the final CIR query.

Target/gallery images use the same learned image-side binder.

---

# 3. Training result

Training converged normally.

| Epoch | R@10 | R@50 | Mean Recall |
|---:|---:|---:|---:|
| 1 | 10.23 | 24.02 | 17.12 |
| 2 | 14.26 | 30.54 | 22.40 |
| 3 | 17.11 | 34.60 | 25.86 |
| 4 | 21.22 | 39.91 | 30.57 |
| 5 | 21.99 | 41.49 | 31.74 |
| **6** | **23.46** | **43.95** | **33.71** |
| 7 | 22.66 | 43.27 | 32.96 |
| 8 | 22.99 | 43.52 | 33.26 |
| 9 | 22.81 | 43.18 | 33.00 |
| 10 | 22.44 | 42.58 | 32.51 |

Best checkpoint:

\[
\boxed{R@10=23.46,\quad R@50=43.95,\quad \mathrm{Mean}=33.71}
\]

The optimization is healthy: loss decreases, retrieval improves strongly through epoch 6, and no NaN/Inf or gradient failure was observed. The failure diagnosed below is therefore **not an optimization failure**.

---

# 4. Functional intervention audit

| Variant | R@10 | R@50 | Mean |
|---|---:|---:|---:|
| **FULL** | **23.46** | **43.95** | **33.71** |
| GLOBAL_ONLY | 4.13 | 11.03 | 7.58 |
| DROP-0 | 23.43 | 43.56 | 33.49 |
| SINGLE-0 | 21.73 | 40.67 | 31.20 |
| REPEAT-0 | 23.20 | 43.70 | 33.45 |
| DROP-1 | 23.48 | 43.84 | 33.66 |
| SINGLE-1 | 20.97 | 39.75 | 30.36 |
| REPEAT-1 | 22.54 | 42.23 | 32.38 |
| DROP-2 | 22.92 | 43.03 | 32.97 |
| SINGLE-2 | 21.49 | 40.37 | 30.93 |
| REPEAT-2 | 22.90 | 42.63 | 32.76 |
| DROP-3 | 23.56 | 43.86 | 33.71 |
| SINGLE-3 | 18.54 | 36.62 | 27.58 |
| REPEAT-3 | 19.19 | 38.70 | 28.95 |

Summary ratios:

\[
\boxed{\mathrm{best\ SINGLE/FULL}=0.92567}
\]

\[
\boxed{\mathrm{best\ REPEAT/FULL}=0.99232}
\]

---

# 5. Relation branch is genuinely useful

The relation pathway matters enormously:

\[
\mathrm{FULL}=33.71
\gg
\mathrm{GLOBAL\_ONLY}=7.58.
\]

Therefore the learned Entity–Action branch carries substantial CIR information. The model is not simply ignoring all relation slots and falling back to the frozen global feature.

This is a positive result:

\[
\boxed{\text{shared relation binding learns a strong retrieval-relevant edit signal}}
\]

---

# 6. Functional decomposition is still weak

The DROP results show very little unique marginal utility:

\[
\mathrm{DROP0}=33.49,
\quad
\mathrm{DROP1}=33.66,
\quad
\mathrm{DROP2}=32.97,
\quad
\mathrm{DROP3}=33.71.
\]

Removing relation 3 changes essentially nothing:

\[
\boxed{\mathrm{DROP3}\approx\mathrm{FULL}}
\]

The largest observed mean-recall loss is from dropping relation 2:

\[
33.71-32.97=0.74.
\]

That is still small relative to FULL performance.

So the four relations do not behave like four strongly complementary functions.

---

# 7. One relation can nearly replace the whole set

The most important forensic result is:

\[
\mathrm{REPEAT0}=33.45
\]

versus:

\[
\mathrm{FULL}=33.71.
\]

Hence:

\[
\frac{\mathrm{REPEAT0}}{\mathrm{FULL}}
=
0.99232.
\]

A single relation repeated four times recovers approximately **99.2% of FULL mean recall**.

That is strong evidence against true functional decomposition.

If the four relations represented complementary edit operators, we would expect:

\[
\mathrm{REPEAT}(k)\ll\mathrm{FULL}.
\]

Instead, the model behaves as though multiple nominal relations implement strongly overlapping functions.

---

# 8. Representation diagnostics

## 8.1 Relation-query geometry

- relation off-diagonal cosine mean: **0.003259**
- relation off-diagonal cosine max: **0.011515**

The relation parameters themselves are nearly orthogonal.

But functional redundancy remains high.

Therefore:

\[
\boxed{\text{geometric orthogonality does not imply functional specialization}}
\]

This branch provides direct empirical evidence for that distinction.

---

## 8.2 Visual entity-slot similarity

- entity slot pairwise cosine: **0.802101**

The visual entity slots are highly correlated but not exact clones.

This suggests partial visual differentiation.

---

## 8.3 Text action-slot similarity

- action slot pairwise cosine: **0.980411**

This is the strongest representational collapse signal.

The four text-side action embeddings are almost identical.

The dominant failure mode is therefore:

\[
\boxed{\textbf{text-side action cloning}}
\]

This strongly explains why REPEAT-0 recovers almost all of FULL.

---

# 9. Same-index Entity–Action binding did not emerge

- same-index Entity–Action cosine: **0.090156**
- off-index Entity–Action cosine: **0.089841**
- pairing gap: **0.000315**

Thus:

\[
\cos(e_k,a_k)
\approx
\cos(e_k,a_j),
\qquad
j\neq k.
\]

The relation index itself carries almost no cross-modal pairing preference.

So the architecture nominally creates pairs:

\[
(e_k,a_k),
\]

but the learned representation does not meaningfully distinguish same-index pairing from off-index pairing.

This is another core failure:

\[
\boxed{\text{shared relation-query indexing alone does not induce stable Entity–Action identity}}
\]

---

# 10. Attention diagnostics

## 10.1 Vision attention

| Relation | Normalized entropy | Max probability | Effective support |
|---:|---:|---:|---:|
| 0 | 0.792 | 0.082 | 152.39 |
| 1 | 0.944 | 0.019 | 387.16 |
| 2 | **0.414** | **0.230** | **15.29** |
| 3 | 0.923 | 0.012 | 341.53 |

The image-side relations do not attend identically.

Relation 2 is much more selective, while relations 1 and 3 are extremely diffuse.

This indicates that the visual branch has learned different attention behaviors.

However, those different attentions do not translate into sufficiently different functional utility under DROP/SINGLE/REPEAT interventions.

---

## 10.2 Text attention

| Relation | Normalized entropy | Max probability | Effective support |
|---:|---:|---:|---:|
| 0 | 0.671 | 0.427 | 5.53 |
| 1 | 0.711 | 0.388 | 6.06 |
| 2 | **0.560** | **0.518** | **4.30** |
| 3 | 0.753 | 0.358 | 6.74 |

Text attention is much more selective than visual attention, with roughly 4–7 effective tokens per relation.

Yet the resulting action representations still have cosine similarity **0.9804**.

Therefore:

\[
\boxed{\text{different attention maps do not imply different functional representations}}
\]

This is a critical lesson for future anti-collapse design.

---

# 11. Token-scale contract is healthy

Diagnostics:

- reference global token norm: **1.000000**
- entity token norm mean: **1.000000**
- entity token norm std: **0.000000**
- text global token norm: **1.000000**
- action token norm mean: **1.000000**
- action token norm std: **0.000000**

So the previously identified magnitude shortcut is gone.

The current redundancy cannot be explained by relation tokens dominating frozen global tokens through larger vector norms.

The remaining collapse is genuinely representational / functional.

---

# 12. Failure-mode diagnosis

## Not a hard routing monopoly

All four relations exist and produce distinct attention patterns.

This is not the old failure where one slot receives everything and the others are empty.

## Not purely geometric collapse

The relation-query parameter vectors are almost orthogonal.

## Not a magnitude shortcut

All composition-interface tokens have unit norm.

## Partial visual diversity exists

The visual entity cosine is 0.802, and the attention profiles differ substantially.

## Dominant failure: text-side action cloning

The action cosine is 0.9804.

## Cross-modal relation identity is effectively absent

The Entity–Action pairing gap is only 0.000315.

## Final functional redundancy remains severe

Best REPEAT/FULL is 0.9923.

The most compact description is:

\[
\boxed{
\textbf{Geometrically diverse relation queries, partially diverse visual attention,}
}
\]

\[
\boxed{
\textbf{but strongly cloned text actions and functionally redundant relation operators.}
}
\]

---

# 13. Scientific conclusion

A8.0 rejects the strong hypothesis:

\[
\boxed{
\text{shared Entity–Action relation queries alone are sufficient for functional edit decomposition}
}
\]

The data does not support that claim.

But A8.0 establishes a useful positive result:

\[
\boxed{
\text{shared relation binding carries strong CIR-relevant information}
}
\]

because FULL is far stronger than GLOBAL_ONLY.

A careful conclusion is:

> Shared Entity–Action binding provides a strong edit-conditioned retrieval signal, but without stronger structural constraints it converges toward functionally redundant relation representations, especially on the text/action side.

---

# 14. What must NOT be concluded

The following claims are not supported:

- relation orthogonality proves specialization;
- different attention maps prove specialization;
- four relations correspond to four semantic edit types;
- same-index Entity–Action binding has been learned;
- A8.0 solves Edit Slot collapse;
- ENCODER as a complete method is disproven.

A8.0 is only a **minimal ENCODER-style shared-binding ablation**.

The full ENCODER pipeline also contains mechanisms such as latent-factor filtering, ScoreNet, FactorNet, relation self-interaction, residual local/factor streams, and weighted token-wise matching.

Therefore the correct scope is:

\[
\boxed{
\text{minimal shared relation binding alone is insufficient}
}
\]

---

# 15. Recommended next research step

Do not add many mechanisms simultaneously. Preserve causal attribution.

The highest-priority question is:

> Why do different text attention patterns collapse into almost identical action embeddings?

The evidence points primarily to the text/action representation path.

Reasonable next controlled experiments are:

### A8.1-A — relation-conditioned residual / identity-preserving action representation

Prevent different token mixtures from being mapped into nearly the same action vector.

Desired effect:

\[
\cos(a_i,a_j)\downarrow
\]

without relying on arbitrary geometric orthogonality.

### A8.1-B — ENCODER-style relation self-interaction

Introduce coordination through:

\[
RR^\top
\]

so relation queries can negotiate distinct roles rather than act independently.

### A8.1-C — latent-factor filtering before binding

If direct binding over all FG-CLIP2 tokens encourages global/redundant solutions, introduce a controlled factor-selection stage before binding.

### A8.1-D — functional anti-redundancy objective

Only if architecture changes remain insufficient, add a directly functional anti-redundancy objective.

Any such loss must be audited against:

- fake specialization;
- nuisance specialization;
- secret sharing;
- arbitrary diversity that hurts retrieval;
- magnitude shortcuts.

---

# 16. Diagnostics that every successor must retain

Every A8.x successor should report:

### Retrieval
- R@10
- R@50
- Mean Recall

### Functional interventions
- FULL
- GLOBAL_ONLY
- DROP-k
- SINGLE-k
- REPEAT-k
- best SINGLE/FULL
- best REPEAT/FULL

### Geometry
- relation off-diagonal cosine
- entity cross-slot cosine
- action cross-slot cosine

### Binding identity
- same-index Entity–Action cosine
- off-index Entity–Action cosine
- pairing gap

### Attention
Per relation and modality:
- entropy
- normalized entropy
- max probability
- effective support

### Representation scale
- global token norm
- entity token norm
- action token norm

A successor should not be called a decomposition success unless the **functional intervention metrics** improve substantially.

---

# 17. Success criteria for a future decomposition

A stronger version should ideally satisfy:

\[
\mathrm{FULL}
>
\mathrm{best\ SINGLE}
\]

by a substantial margin,

\[
\mathrm{FULL}
>
\mathrm{best\ REPEAT}
\]

by a substantial margin,

and several relations should have measurable unique marginal utility:

\[
\mathrm{DROP}(k)
<
\mathrm{FULL}.
\]

In addition, action-slot cosine should decrease substantially from 0.9804 and the Entity–Action pairing gap should become clearly positive.

These representation metrics are not sufficient by themselves, but they should agree with the functional audit.

---

# 18. Final verdict

## Optimization

\[
\boxed{\text{PASS}}
\]

The model trains stably and improves retrieval.

## Representation-scale correctness

\[
\boxed{\text{PASS}}
\]

No norm-magnitude shortcut remains.

## Relation utility

\[
\boxed{\text{PASS}}
\]

The relation branch carries substantial retrieval information.

## Visual differentiation

\[
\boxed{\text{PARTIAL}}
\]

Visual attention and entity representations show some diversity.

## Text/action differentiation

\[
\boxed{\text{FAIL}}
\]

Action representations are near-clones:

\[
\cos(a_i,a_j)=0.9804.
\]

## Entity–Action index binding

\[
\boxed{\text{FAIL}}
\]

Same-index and off-index similarities are essentially identical.

## Functional decomposition

\[
\boxed{\text{FAIL / REDUNDANT}}
\]

because:

\[
\mathrm{best\ REPEAT/FULL}=0.9923.
\]

---

# 19. One-sentence research takeaway

\[
\boxed{
\textbf{A8.0 shows that shared Entity–Action relation binding is useful for CIR,}
}
\]

\[
\boxed{
\textbf{but shared queries + orthogonality alone do not create functionally specialized edit factors.}
}
\]

The dominant observed pathology is **text-side action cloning**, followed by weak cross-modal relation identity and high functional redundancy.