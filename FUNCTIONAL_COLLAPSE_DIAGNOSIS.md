# IAG-SRME Functional Candidate Collapse — Diagnostic Report

**Repository:** `Le-Minh-Nhut/cir`  
**Primary audit branch:** `exp/e2e-iag-srme-v2-r0-functional-collapse-audit`  
**Current audited HEAD:** `a259b378c596beeb2a26c3e04db6911fd1bd4943`  
**Scope:** FashionIQ / IAG-SRME V2 R0 candidate generation, routing, grounding, execution, and functional-collapse diagnosis  
**Purpose:** Preserve a complete technical record of the failure mode, experiments, diagnostics, eliminated hypotheses, current best explanation, remaining caveats, and the next architecture experiment.

---

## 1. Executive summary

The original failure looked like a **candidate-index / winner-take-most routing collapse**: one candidate index was selected much more often than the others, even when the candidates were intended to represent different edits/actions.

A Loss-Free MoE-style routing controller was therefore introduced to prevent index monopoly without perturbing the task loss. That controller successfully reduced selection-frequency imbalance and showed that the catastrophic index monopoly was **not the deepest failure**.

After instrumenting the full candidate pipeline, the main failure was localized to a much earlier stage:

> The candidate proposals themselves are functionally collapsing before routing becomes relevant.

A second, more detailed audit inside `ProposalNet` then localized the main collapse even more precisely:

> The learned slot queries remain highly diverse.  
> The conditioned queries remain substantially diverse.  
> The **cross-attention output collapses almost all of that diversity**.

The strongest quantitative evidence is:

- learned/base queries:
  - pairwise cosine ≈ `-0.0059`
  - effective rank ≈ `3.9955 / 4`
  - relative spread ≈ `0.8685`
- query sent into cross-attention:
  - pairwise cosine ≈ `0.6265`
  - effective rank ≈ `3.1139 / 4`
  - relative spread ≈ `0.5291`
- final proposal output after cross-attention:
  - pairwise cosine ≈ `0.9994`
  - effective rank ≈ `1.0604 / 4`
  - relative spread ≈ `0.0175`

So the main collapse is:

\[
q_k^{\text{post-norm}}
\quad \longrightarrow \quad
\operatorname{CrossAttn}(q_k^{\text{post-norm}}, T, T)
\]

with diversity collapsing from roughly rank `3.11` to rank `1.06`.

The current architecture uses:

```python
edits, _ = self.token_attention(
    query_post_norm,
    text_tokens,
    text_tokens,
    key_padding_mask=~content_mask,
    need_weights=False,
)
```

and intentionally has no post-attention slot-identity residual:

```python
# No q residual: candidate content comes only from instruction token values.
```

Therefore candidate identity survives only if distinct slot queries produce sufficiently distinct attention distributions over the shared token values. The diagnostics show that this is not happening.

The current best next experiment is therefore a **minimal residual cross-attention ablation**, preserving all other architecture/training choices:

\[
e_k
=
\operatorname{LN}\left(
q_k^{\text{post-norm}}
+
\operatorname{CrossAttn}(q_k^{\text{post-norm}}, T, T)
\right)
\]

This report deliberately separates **observed facts**, **strongly supported interpretations**, and **still-unproven hypotheses**.

---

## 2. Problem definition

The model is intended to behave like a recurrent decision system for Composed Image Retrieval.

At each recurrent step \(t\):

1. build the current visual/query state;
2. generate \(K\) candidate edits/actions;
3. ground each candidate;
4. execute each candidate counterfactually;
5. estimate its marginal utility;
6. select one candidate or STOP;
7. commit the selected candidate;
8. regenerate a fresh candidate set at the next recurrent step.

A simplified intended computation is:

\[
q_t
\rightarrow
\{e_t^1,\ldots,e_t^K\}
\rightarrow
\{a_t^1,\ldots,a_t^K\}
\rightarrow
\{\Delta q_t^1,\ldots,\Delta q_t^K\}
\rightarrow
\text{ScoreNet}
\rightarrow
k^*
\rightarrow
q_{t+1}
\]

The desired behavior is that the \(K\) candidates represent meaningfully different possible edits or consequences.

The observed failure was initially:

\[
\Pr(k^*=j) \gg \Pr(k^*=i), \quad i\neq j
\]

for one fixed candidate index \(j\), across many unrelated samples.

This looked like a routing/index specialization collapse.

---

## 3. Relevant architecture

The relevant candidate pipeline in the current model is:

```text
Learned slot queries
        |
        v
ProposalNet conditioning
        |
        v
query_post_norm
        |
        v
Cross-attention over instruction tokens
        |
        v
proposals / edits
        |
        v
     Grounder
      /    \
     /      \
alpha_read  exec_mask
   |           |
   v           |
entities       |
   |           |
   v           |
actions        |
    \         /
     \       /
      Executor
         |
         v
       delta
         |
         v
      delta_q
         |
         v
     ScoreNet
         |
         v
 Routing / STOP
```

Important: `alpha_read` and `exec_mask` are **sibling outputs of Grounder**. They are not sequential stages.

---

## 4. ProposalNet as currently implemented

The key ProposalNet path is approximately:

```python
base_query = self.queries.unsqueeze(0)
expanded_query = base_query.expand(batch_size, -1, -1)

conditioned = self.query_conditioner(
    torch.cat(
        [
            expanded_query,
            context[:, None].expand_as(expanded_query),
            expanded_query * context[:, None],
        ],
        dim=-1,
    )
)

query_pre_norm = expanded_query + conditioned
query_post_norm = self.query_norm(query_pre_norm)

edits, _ = self.token_attention(
    query_post_norm,
    text_tokens,
    text_tokens,
    key_padding_mask=~content_mask,
    need_weights=False,
)
```

The model comment explicitly states:

```python
# No q residual: candidate content comes only from instruction token values.
```

Thus the candidate output is effectively:

\[
e_k
=
\operatorname{CrossAttn}(q_k,T,T)
\]

rather than:

\[
e_k
=
q_k+\operatorname{CrossAttn}(q_k,T,T)
\]

or:

\[
e_k
=
\operatorname{LN}\left(q_k+\operatorname{CrossAttn}(q_k,T,T)\right).
\]

This distinction became central after the internal ProposalNet audit.

---

## 5. Original index-collapse hypothesis

The initial failure mode appeared to be:

- one candidate index repeatedly won selection;
- scorer feedback reinforced the same index;
- aWTA / DAC-inspired mechanisms did not solve the problem;
- balancing/diversity/stream-refit mechanisms did not fully eliminate the observed winner-take-most behavior;
- functional DPP also did not prevent the system from degenerating.

The first working hypothesis was therefore:

> the candidate indices are symmetric, one index receives a slight early advantage, and feedback causes a stable routing monopoly.

This motivated trying a routing-only balancing controller rather than modifying the loss.

---

## 6. Loss-Free routing experiment

### 6.1 Core idea

The adopted idea came from **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts**, arXiv:2408.15664.

The relevant mechanism adds an external routing bias:

\[
\tilde s_k=s_k+b_k
\]

used only for routing. The raw task score remains \(s_k\).

The controller updates the bias from observed routing counts:

\[
b_k
\leftarrow
b_k
+
u\,\operatorname{sign}(\bar c-c_k)
\]

where \(c_k\) is the committed selection count of slot \(k\), \(\bar c\) is the mean count, and the paper-inspired default used was:

\[
u=10^{-3}.
\]

### 6.2 CIR-specific STOP semantics

The implementation deliberately preserved:

```text
step["scores"] = raw ScoreNet scores
```

Losses continued to consume raw scores.

Only routing used:

\[
s_k+b_k.
\]

A candidate was eligible only if:

\[
s_k>\epsilon_{\text{stop}}.
\]

The biased router selected only among raw-eligible candidates. If no candidate was raw-eligible, the model STOPped.

Thus the routing bias could not manufacture utility for a raw-negative action. It acted as an allocation controller, not as a utility estimator.

### 6.3 Relevant commits

Loss-Free branch:

```text
exp/e2e-iag-srme-v2-r0-lossfree-routing
```

Important commits:

```text
5c525840ff81c4f52ea1284f76871243053f4021
```

Initial implementation.

```text
4a94401791f0b98821bf8e09e41c0d0d1999e707
```

Diagnostic correction.

```text
ba2599f4cf4893455d46afe7081df3ea3d79c85a
```

Only `.gitignore`; no model change.

---

## 7. What Loss-Free routing actually showed

Near the end of the Loss-Free run, representative batches had:

```text
raw monopoly    ≈ 29–35%
routed monopoly ≈ 29–35%
```

with \(K=4\), where perfect uniformity is \(25\%\).

A rough average over the final ten observed batches was:

\[
\text{raw monopoly}\approx31.64\%
\]

\[
\text{routed monopoly}\approx31.09\%
\]

with routing MaxVio approximately:

\[
0.244.
\]

This is nowhere near the catastrophic case:

\[
\text{monopoly}=1.
\]

Therefore Loss-Free appeared to prevent severe **selection-frequency collapse**.

However, this did not mean the candidates themselves were functionally different.

---

## 8. Loss-Free retrieval comparison

One-epoch baseline without Loss-Free:

\[
\text{mean recall}=15.131
\]

One-epoch Loss-Free run:

\[
\text{mean recall}=14.871.
\]

Difference:

\[
15.131-14.871=0.260
\]

or roughly \(1.7\%\) relative.

This was not treated as a performance conclusion because:

- only one epoch was used;
- PyTorch warned that memory-efficient attention used a non-deterministic algorithm;
- the experiment was primarily diagnostic.

---

## 9. Routing diagnostic caveat: common-mode bias

The original diagnostic:

```text
routing_bias_to_raw_score_std
=
max_abs(routing_bias) / raw_score_std
```

can overstate effective routing pressure.

Routing is invariant to a common shift:

\[
s_k+b_k+c
\]

has the same argmax as:

\[
s_k+b_k.
\]

Only centered bias matters:

\[
\tilde b_k=b_k-\bar b.
\]

A representative late bias:

```text
[0.009, 0.009, 0.011, 0.007]
```

has centered form:

```text
[0, 0, +0.002, -0.002]
```

so a reported ratio near `0.80` can overstate the actual slot-differential routing pressure; a centered estimate for that example was roughly `0.145`.

This common-mode drift can occur because the sign update need not preserve zero sum.

This is a diagnostic caveat, not the central collapse issue.

---

## 10. First evidence of deeper functional collapse

Even when routing no longer looked catastrophically monopolized, candidate consequences showed:

```text
functional_pairwise_cosine ≈ 1
functional_rank ≈ 1
```

Examples:

```text
batch 1:
functional_pairwise_cosine = 1.0
functional_rank = 1.0

batch 3:
cosine ≈ 0.9727
rank   ≈ 1.522

batch 5:
cosine ≈ 0.9922
rank   ≈ 1.274

batch 8:
cosine ≈ 0.9980
rank   ≈ 1.129
```

This suggested:

\[
\Delta q_1
\approx
\Delta q_2
\approx
\Delta q_3
\approx
\Delta q_4.
\]

That changed the diagnosis from “routing monopoly” to “candidate functional collapse”.

---

## 11. Failure taxonomy

### 11.1 Selection/routing collapse

One index is selected much more often:

\[
P(k=j)\gg P(k=i).
\]

Loss-Free directly addresses this.

### 11.2 Proposal/representation collapse

Candidate representations become similar:

\[
e_1\approx e_2\approx\dots\approx e_K.
\]

### 11.3 Functional/consequence collapse

Executing different candidates causes almost the same effect:

\[
\Delta q_1\approx\Delta q_2\approx\dots\approx\Delta q_K.
\]

Balanced routing does not solve proposal or functional collapse.

---

## 12. Baseline reproduction without Loss-Free

Baseline command:

```bash
python src/train.py \
  model=iag_srme \
  experiment.epochs=1
```

Result:

```text
mean_recall = 15.131
```

Representative functional diagnostics:

```text
batch 1:
functional_cos  = 1.0
functional_rank = 1.0

batch 2:
functional_cos  = 1.0
functional_rank = 1.0

batch 3:
functional_cos  = 0.97265625
functional_rank = 1.522118568

batch 10:
functional_cos  = 0.998046875
functional_rank = 1.129397631
```

Late epoch:

```text
batch 554:
cos  = 0.995117
rank = 1.207001

batch 563:
cos  = 0.992188
rank = 1.260283
```

Full epoch mean:

```text
functional_cos  = 0.9988968
functional_rank = 1.0773524
```

Therefore:

\[
\boxed{\text{functional collapse exists without Loss-Free routing}}
\]

Loss-Free was ruled out as the primary cause.

Additional epoch means:

```text
mean_delta_q_norm      ≈ 0.02814
useful_candidate_count ≈ 0.4718
dpp_valid_rate         ≈ 0.1182
stop_rate              ≈ 0.1390
```

Late batches often had:

```text
useful_candidate_count = 0
dpp_valid_rate = 0
```

So DPP was active on only a limited subset of rows/steps and was not sufficient to rescue the collapsed candidate set.

---

## 13. Functional-collapse audit branch

Branch:

```text
exp/e2e-iag-srme-v2-r0-functional-collapse-audit
```

Important commits:

```text
181e7d7abaf12556694cc8439ac64aab778c4775
```

Initial full-pipeline audit.

```text
86afb0ba2bd2ed0fb45e8b6f5051b5b5eed5a44e
```

Corrected stage ordering and optimized effective-rank computation.

```text
43e90a1134ab3848143dfabdd71704422314e0b1
```

Added ProposalNet internal audit.

```text
a259b378c596beeb2a26c3e04db6911fd1bd4943
```

Fixed parameter-backed diagnostic snapshot correctness.

Current audited HEAD:

```text
a259b378c596beeb2a26c3e04db6911fd1bd4943
```

Local test status at current HEAD:

```text
148 passed
2 skipped
```

No GitHub CI workflow was observed for the commit.

---

## 14. Full-pipeline audit metrics

The audit measured:

```text
proposals
alpha_read
exec_mask
entities
actions
delta
delta_q
```

Generic metrics:

```text
pairwise_cosine
effective_rank
spread
relative_spread
mean_norm
mean_norm_ck
```

`alpha_read` metrics:

```text
pairwise_cosine
pairwise_js_divergence
entropy
argmax_agreement
```

`exec_mask` metrics:

```text
pairwise_cosine
soft_iou
spread
relative_spread
mean_activation
```

All metrics are reported overall and by recurrent step.

---

## 15. Effective-rank definition and implementation

The effective rank is:

\[
r_{\mathrm{eff}}
=
\frac{
(\sum_i\sigma_i)^2
}{
\sum_i\sigma_i^2
}.
\]

For \(K=4\):

- rank-one collapse gives approximately \(1\);
- maximally independent candidates give approximately \(4\).

The audit reuses the same semantics as the previous `functional_rank`.

To avoid expensive SVD on a wide matrix such as:

\[
[B,K,N\cdot D],
\]

the diagnostic computes:

\[
G=ZZ^\top\in\mathbb{R}^{K\times K},
\]

then:

\[
\sigma_i=\sqrt{\max(\lambda_i(G),0)}.
\]

Tests compare this Gram-based result directly against `torch.linalg.svdvals` on small tensors.

---

## 16. Full-pipeline audit result

### First 10 batches

| stage | cosine | rank | relative spread | stage-specific |
|---|---:|---:|---:|---:|
| proposals | 0.9908 | 1.2869 | 0.0805 | — |
| alpha_read | 0.9987 | — | — | JS 0.0003 |
| exec_mask | 0.9997 | — | 0.0180 | soft IoU 0.9756 |
| entities | 0.9998 | 1.0349 | 0.0164 | — |
| actions | 0.9956 | 1.1948 | 0.0559 | — |
| delta | 0.7983 | 0.8375 | 0.0139 | — |
| delta_q | 0.9829 | 1.2014 | 0.0566 | — |

The early `delta` rank below 1 is numerically suspicious because the theoretical effective rank is not meaningfully below 1 except from near-zero/numerical degeneracy. The later Gram implementation improved numerical behavior. The qualitative conclusion does not depend on that single value.

### Last 10 batches

| stage | cosine | rank | relative spread | stage-specific |
|---|---:|---:|---:|---:|
| proposals | 0.9998 | 1.0387 | 0.0114 | — |
| alpha_read | 1.0000 | — | — | JS 0.0000 |
| exec_mask | 1.0000 | — | 0.0026 | soft IoU 0.9951 |
| entities | 1.0000 | 1.0011 | 0.0016 | — |
| actions | 1.0000 | 1.0146 | 0.0045 | — |
| delta | 1.0000 | 1.0051 | 0.0023 | — |
| delta_q | 0.9936 | 1.2331 | 0.0654 | — |

### Full run mean

| stage | cosine | rank | relative spread | stage-specific |
|---|---:|---:|---:|---:|
| proposals | **0.9994** | **1.0604** | **0.0175** | — |
| alpha_read | **1.0000** | — | — | **JS 0.0000** |
| exec_mask | **1.0000** | — | **0.0034** | **soft IoU 0.9949** |
| entities | **1.0000** | **1.0023** | **0.0024** | — |
| actions | **0.9998** | **1.0335** | **0.0099** | — |
| delta | **0.9958** | **1.0044** | **0.0030** | — |
| delta_q | **0.9985** | **1.0781** | **0.0224** | — |

Heuristic first high-similarity stage:

```text
proposals
```

---

## 17. Recurrent-step result

Proposal rank:

```text
t0 ≈ 1.0603
t1 ≈ 1.0602
t2 ≈ 1.0606
```

Proposal cosine is approximately:

```text
0.9994
```

at all three steps.

Therefore recurrence does not recover specialization later.

The collapse mechanism is already present at every recurrent step.

---

## 18. First major localization conclusion

The full-pipeline audit established:

\[
\boxed{\text{the earliest visible collapse is already at ProposalNet output}}
\]

The downstream pattern is consistent with:

\[
\text{proposal collapse}
\rightarrow
\text{grounding collapse}
\rightarrow
\text{entity collapse}
\rightarrow
\text{action collapse}
\rightarrow
\text{execution collapse}
\rightarrow
\text{retrieval-consequence collapse}.
\]

This motivated an internal ProposalNet audit.

---

## 19. Internal ProposalNet audit

Internal stages:

```text
base_query
expanded_query
conditioned_residual
query_pre_norm
query_post_norm
proposal_output
```

Additional metrics:

```text
conditioner_to_base_norm_ratio
base_pre_cosine
mean_relative_displacement
attention_diversity_retention
attention_rank_retention
```

Attention weights were intentionally not collected because the current forward uses:

```python
need_weights=False
```

and changing that could alter kernel selection, memory use, or execution behavior. No second attention forward was introduced.

---

## 20. Diagnostic snapshot bug and fix

The first internal audit stored:

```python
base_query.detach()
expanded_query.detach()
```

but `detach()` preserves shared storage with `self.queries`.

The training engine computes diagnostics after:

```text
forward
backward
optimizer.step
```

so parameter-backed views could have reflected updated parameters while other saved tensors still represented the earlier forward.

This could compare:

\[
q_{\text{base}}^{t+1}
\]

to:

\[
q_{\text{pre}}^t.
\]

The fix is:

```python
base_query.detach().clone()
expanded_query.detach().clone()
```

and a regression test mutates `self.queries` after forward and verifies the saved audit tensors do not change.

Corrected HEAD:

```text
a259b378c596beeb2a26c3e04db6911fd1bd4943
```

---

## 21. Internal ProposalNet audit results

### First 10 batches

| stage | cosine | rank | relative spread |
|---|---:|---:|---:|
| base_query | -0.0063 | 3.9955 | 0.8687 |
| expanded_query | -0.0063 | 3.9955 | 0.8687 |
| conditioned_residual | 0.9919 | 1.2813 | 0.0780 |
| query_pre_norm | 0.5616 | 3.2828 | 0.5737 |
| query_post_norm | 0.5613 | 3.2845 | 0.5735 |
| proposal_output | 0.9908 | 1.2869 | 0.0805 |

Additional:

```text
conditioner_to_base_norm_ratio = 1.119575
base_pre_cosine                = 0.676469
mean_relative_displacement     = 1.119575
attention_diversity_retention  = 0.140193
attention_rank_retention       = 0.391748
```

### Last 10 batches

| stage | cosine | rank | relative spread |
|---|---:|---:|---:|
| base_query | -0.0058 | 3.9954 | 0.8685 |
| expanded_query | -0.0058 | 3.9954 | 0.8685 |
| conditioned_residual | 0.9941 | 1.2380 | 0.0664 |
| query_pre_norm | 0.6403 | 3.0749 | 0.5196 |
| query_post_norm | 0.6402 | 3.0760 | 0.5194 |
| proposal_output | 0.9998 | 1.0387 | 0.0114 |

Additional:

```text
conditioner_to_base_norm_ratio = 1.320438
base_pre_cosine                = 0.613726
mean_relative_displacement     = 1.320438
attention_diversity_retention  = 0.021882
attention_rank_retention       = 0.337683
```

### Full run mean

| stage | cosine | rank | relative spread |
|---|---:|---:|---:|
| base_query | **-0.0059** | **3.9955** | **0.8685** |
| expanded_query | **-0.0059** | **3.9955** | **0.8685** |
| conditioned_residual | **0.9937** | **1.2460** | **0.0686** |
| query_pre_norm | **0.6266** | **3.1128** | **0.5293** |
| query_post_norm | **0.6265** | **3.1139** | **0.5291** |
| proposal_output | **0.9994** | **1.0604** | **0.0175** |

Additional:

```text
conditioner_to_base_norm_ratio = 1.278412
base_pre_cosine                = 0.629162
mean_relative_displacement     = 1.278412
attention_diversity_retention  = 0.032722
attention_rank_retention       = 0.340451
```

Heuristic:

```text
attention_output
```

---

## 22. Interpretation of base-query metrics

The learned slot queries are **not collapsed**.

\[
\text{cosine}\approx-0.006
\]

\[
r_{\mathrm{eff}}\approx3.9955/4.
\]

Therefore:

\[
\boxed{\text{self.queries remain highly diverse}}
\]

This rules out learned-query parameter collapse as the primary cause.

---

## 23. Interpretation of the conditioner

The conditioned residual is highly common-mode:

\[
\text{cosine}\approx0.9937
\]

\[
r_{\mathrm{eff}}\approx1.246.
\]

It is also large:

\[
\frac{\|q_{\mathrm{cond}}\|}{\|q_{\mathrm{base}}\|}
\approx1.278.
\]

It reduces diversity from:

\[
3.9955
\rightarrow
3.1139.
\]

Pairwise cosine rises from:

\[
-0.0059
\rightarrow
0.6265.
\]

Thus the conditioner weakens slot separation significantly.

However, the pre-attention candidate set remains far from rank one:

\[
r_{\mathrm{eff}}\approx3.11.
\]

So:

\[
\boxed{\text{the conditioner is a secondary contributor, not the main collapse point}}
\]

---

## 24. LayerNorm is not the collapse point

Immediately before and after LayerNorm:

```text
query_pre_norm:
cosine = 0.6266
rank   = 3.1128
spread = 0.5293

query_post_norm:
cosine = 0.6265
rank   = 3.1139
spread = 0.5291
```

Therefore:

\[
\boxed{\text{LayerNorm is not causing the collapse}}
\]

---

## 25. Cross-attention is the dominant measured collapse point

Immediately before token attention:

\[
r_{\mathrm{eff}}\approx3.1139
\]

Immediately after:

\[
r_{\mathrm{eff}}\approx1.0604.
\]

Pairwise cosine:

\[
0.6265
\rightarrow
0.9994.
\]

Relative spread:

\[
0.5291
\rightarrow
0.0175.
\]

Measured diversity retention:

\[
\frac{0.0175}{0.5291}
\approx0.0327.
\]

So only about \(3.3\%\) of relative spread survives the attention block on average.

Rank retention is approximately:

\[
\frac{1.0604}{3.1139}
\approx0.340.
\]

Therefore:

\[
\boxed{\text{the dominant collapse occurs inside the ProposalNet cross-attention mapping}}
\]

---

## 26. Collapse worsens during training

Attention diversity retention changes approximately:

```text
FIRST 10 = 0.1402
LAST 10  = 0.0219
```

Proposal output changes:

```text
FIRST 10:
cosine ≈ 0.9908
rank   ≈ 1.2869

LAST 10:
cosine ≈ 0.9998
rank   ≈ 1.0387
```

Training therefore reinforces the collapsed solution rather than escaping it.

---

## 27. Recurrent-step internal audit

### t0

```text
base_query rank       ≈ 3.9955
query_post_norm rank  ≈ 3.1147
proposal_output rank  ≈ 1.0603
attention retention   ≈ 0.0327
```

### t1

```text
base_query rank       ≈ 3.9955
query_post_norm rank  ≈ 3.1135
proposal_output rank  ≈ 1.0602
attention retention   ≈ 0.0326
```

### t2

```text
base_query rank       ≈ 3.9955
query_post_norm rank  ≈ 3.1134
proposal_output rank  ≈ 1.0606
attention retention   ≈ 0.0328
```

The collapse mechanism is stable across recurrence.

---

## 28. Why current cross-attention can erase candidate identity

Conceptually:

\[
e_k
=
\operatorname{softmax}
\left(
\frac{Q_kK^\top}{\sqrt d}
\right)V.
\]

The queries \(Q_k\) differ, but every candidate attends to the same instruction-token \(K,V\).

If:

\[
A_1\approx A_2\approx A_3\approx A_4,
\]

then:

\[
A_1V
\approx
A_2V
\approx
A_3V
\approx
A_4V.
\]

Because there is no post-attention query residual, there is no separate slot-identity path.

Once attention outputs become similar, the slot identity can disappear almost completely.

This is structurally consistent with the measured:

\[
3.11\rightarrow1.06
\]

effective-rank collapse.

---

## 29. Observation vs unmeasured attention-weight mechanism

Attention weights were not logged.

Therefore:

> “the attention distributions themselves are identical”

is a plausible internal mechanism, not a directly measured fact.

What is directly measured is:

\[
q_k^{\text{post-norm}}
\]

remains diverse, while:

\[
e_k
\]

does not.

Supported conclusion:

\[
\boxed{\text{the collapse happens inside the token-attention mapping}}
\]

Possible internal causes include:

- similar attention distributions;
- common-mode value projections;
- multi-head output mixing;
- output projection suppressing slot-specific directions;
- a combination of these.

---

## 30. Why routing methods cannot solve this failure

If:

\[
e_1\approx e_2\approx e_3\approx e_4,
\]

then downstream:

\[
\alpha_1\approx\alpha_2\approx\alpha_3\approx\alpha_4,
\]

\[
a_1\approx a_2\approx a_3\approx a_4,
\]

\[
\Delta q_1\approx\Delta q_2\approx\Delta q_3\approx\Delta q_4.
\]

A router can choose different indices while still executing nearly the same function.

Therefore balancing slot usage cannot create missing functional diversity.

---

## 31. Why aWTA / DAC-style credit balancing is not sufficient alone

aWTA / DAC-style methods mainly change:

- winner assignment;
- responsibility;
- specialization pressure;
- gradient allocation.

But if the candidate generator maps different latent slots to nearly identical proposals, the optimization system receives almost indistinguishable functions.

The deeper question is not:

```text
Which candidate receives credit?
```

but:

```text
Do different candidates still implement different functions?
```

The answer after ProposalNet cross-attention is currently: mostly no.

---

## 32. Why DPP did not rescue the model

Functional DPP acts on candidate consequences.

Observed:

```text
dpp_valid_rate ≈ 0.1182 overall
```

with many late batches at:

```text
dpp_valid_rate = 0
useful_candidate_count = 0
```

So:

1. candidate generation collapses upstream;
2. useful candidate sets become sparse;
3. DPP is inactive on many rows;
4. the diversity regularizer cannot reliably push apart candidate functions.

This does not prove DPP is useless or implemented incorrectly. It means DPP was not sufficient as the primary cure.

---

## 33. Executor initialization observation

Executor initialization includes:

```python
nn.init.zeros_(self.action_mlp[-1].weight)
nn.init.zeros_(self.action_mlp[-1].bias)

nn.init.zeros_(self.state_up.weight)
nn.init.zeros_(self.state_up.bias)
```

This helps explain extremely similar or near-zero residuals at the very beginning of training.

However, the earliest persistent collapse found by the stagewise audit is already at ProposalNet output. Executor initialization may be a downstream contributor, not the primary current localization.

---

## 34. AMP / numerical observations

Some early batches skipped optimizer updates because AMP detected non-finite gradients.

Observed examples included:

```text
batch 1:
nonfinite_gradient_count = 260
AMP scale decreased
optimizer update skipped
```

and another early skipped batch.

This should be monitored separately, but it does not explain the full-epoch structural collapse because the same pattern persists throughout training and across all recurrent steps.

---

## 35. Non-deterministic attention caveat

PyTorch emitted:

```text
Memory Efficient attention defaults to a non-deterministic algorithm.
```

Therefore tiny single-run retrieval differences should not be overinterpreted.

However, a structural change from effective rank \(\sim3.11\) to \(\sim1.06\) is far larger than normal numerical noise.

---

## 36. What has been ruled out or deprioritized

### Strongly ruled out as primary cause

**Learned slot-query collapse:** no.

\[
r_{\mathrm{eff}}\approx3.9955/4.
\]

**LayerNorm collapse:** no.

Pre/post LayerNorm metrics are essentially unchanged.

**Loss-Free causing functional collapse:** no.

Baseline without Loss-Free shows the same collapse.

**Recurrence causing collapse only later:** no.

`t0`, `t1`, and `t2` show the same pattern.

### Secondary contributors

**Query conditioner:** significant common-mode component; reduces diversity but does not destroy it completely.

**Executor initialization:** potentially contributes to early consequence similarity, but collapse already exists upstream.

**Sparse DPP validity:** limits downstream diversity regularization but does not explain ProposalNet output collapse.

---

## 37. Current best causal story

The strongest current explanation is:

1. learned slot queries are healthy and distinct;
2. query conditioning adds a large common-mode component;
3. residual addition preserves enough slot identity that pre-attention queries remain rank \(\sim3.1\);
4. shared text cross-attention maps those distinct queries into nearly the same text-derived proposal;
5. no post-attention query residual preserves slot identity;
6. Grounder receives almost identical proposals;
7. Grounder returns nearly identical read/write maps;
8. entities become almost identical;
9. actions become almost identical;
10. Executor produces nearly identical consequences;
11. ScoreNet receives nearly identical candidate functions;
12. routing/index imbalance becomes a downstream symptom.

In shorthand:

\[
\boxed{
\text{slot identity}
\xrightarrow{\text{conditioner}}
\text{weakened but alive}
\xrightarrow{\text{cross-attention}}
\text{almost erased}
}
\]

then:

\[
\boxed{
\text{proposal collapse}
\rightarrow
\text{functional collapse}
\rightarrow
\text{routing symptom}
}
\]

---

## 38. Proposed first architecture intervention

The cleanest first causal ablation is to preserve slot identity across cross-attention.

Current:

\[
e_k
=
\operatorname{CrossAttn}(q_k,T,T)
\]

Proposed:

\[
\tilde e_k
=
q_k+
\operatorname{CrossAttn}(q_k,T,T)
\]

preferably:

\[
e_k
=
\operatorname{LN}(\tilde e_k).
\]

Code sketch:

```python
attended, _ = self.token_attention(
    query_post_norm,
    text_tokens,
    text_tokens,
    key_padding_mask=~content_mask,
    need_weights=False,
)

edits = self.output_norm(query_post_norm + attended)
```

This should be tested as an isolated architecture change.

Do not simultaneously add:

- new DPP changes;
- aWTA changes;
- Loss-Free changes;
- ModeSeq;
- sequential candidate generation;
- entropy losses;
- orthogonality losses;
- noise injection;
- stronger balancing.

The objective is causal attribution.

---

## 39. Why residual cross-attention is the first test

Cross-attention should enrich a candidate query with instruction information rather than replace candidate identity entirely.

Residual attention creates two information paths:

\[
q_k\rightarrow e_k
\]

and:

\[
T
\rightarrow
\operatorname{CrossAttn}(q_k,T,T)
\rightarrow e_k.
\]

Even if:

\[
A_iV\approx A_jV,
\]

the outputs can remain distinct because:

\[
q_i\neq q_j.
\]

This directly attacks the measured bottleneck.

---

## 40. What success should look like

The first success criterion is structural, not final retrieval score.

Current:

```text
query_post_norm rank  ≈ 3.11
proposal_output rank  ≈ 1.06
```

Desired initial behavior:

```text
proposal_output rank >> 1.06
```

Ideally:

```text
~2.5–3+
```

with lower pairwise cosine and larger relative spread.

Then check whether downstream diversity opens:

```text
alpha_read JS increases
exec_mask IoU decreases
entity rank increases
action rank increases
delta rank increases
delta_q rank increases
```

without catastrophic retrieval degradation.

---

## 41. Metrics for the residual-attention ablation

Compare baseline vs residual attention on:

```text
mean_recall

proposal_output:
    pairwise_cosine
    effective_rank
    relative_spread

query_post_norm:
    pairwise_cosine
    effective_rank
    relative_spread

attention_diversity_retention
attention_rank_retention

alpha_read:
    cosine
    JS divergence
    argmax agreement

exec_mask:
    cosine
    soft IoU
    relative spread

entities:
    cosine
    rank
    spread

actions:
    cosine
    rank
    spread

delta:
    cosine
    rank
    spread

delta_q:
    cosine
    rank
    spread

routing:
    raw monopoly
    routed monopoly
    MaxVio
    disagreement fraction

DPP:
    useful_candidate_count
    dpp_valid_rate

STOP:
    stop_rate
```

---

## 42. Interpretation matrix for the next experiment

### Case A — proposal diversity recovers and downstream diversity recovers

Example:

```text
proposal rank: 1.06 -> 2.8
entity rank:   1.00 -> 2+
delta_q rank:  1.08 -> 2+
```

Interpretation:

> Cross-attention identity erasure was the dominant architectural failure.

### Case B — proposal diversity recovers but Grounder still collapses

Example:

```text
proposal rank: 1.06 -> 3.0
alpha_read JS: still ~0
entity rank:   still ~1
```

Interpretation:

> ProposalNet was one bottleneck, but Grounder has an independent collapse mechanism.

### Case C — residual does not recover proposal diversity

Potential explanations:

- attention output magnitude overwhelms the residual;
- normalization re-compresses differences;
- token-value path dominates;
- candidate-specific signal is not represented usefully.

### Case D — structural diversity improves but retrieval degrades

Then diversity may be non-semantic. Inspect:

- teacher utility per candidate;
- useful-candidate count;
- DPP validity;
- ScoreNet calibration;
- whether residual creates arbitrary rather than instruction-grounded diversity.

---

## 43. Why ModeSeq is not the first intervention now

ModeSeq remains relevant for sequential candidate dependence and exchangeability breaking.

However, the current model already has a simpler measured bottleneck:

\[
\text{diverse queries}
\rightarrow
\text{collapsed attention outputs}.
\]

Adding sequential generation before preserving query identity could still feed diverse sequential queries into a mapping that erases them.

So the immediate priority is:

\[
\boxed{\text{preserve candidate identity first}}
\]

Then reassess whether ModeSeq is still needed.

---

## 44. Why Loss-Free should remain available

Loss-Free still has value as a routing stabilizer:

- reduces routing-frequency runaway;
- leaves raw task scores untouched;
- preserves raw STOP semantics;
- can prevent renewed index monopoly once candidate functions become distinct.

Its correct role is:

\[
\text{routing stabilization}
\]

not:

\[
\text{creating candidate-function diversity}.
\]

---

## 45. Why the original winner-index symptom was misleading

If:

\[
\Delta q_1
\approx
\Delta q_2
\approx
\Delta q_3
\approx
\Delta q_4,
\]

then ScoreNet scores are naturally close:

\[
s_1\approx s_2\approx s_3\approx s_4.
\]

Small asymmetries can repeatedly make one index win.

That makes the index look like the primary problem even though the alternatives are nearly equivalent.

The key question changed from:

```text
Why does one slot always win?
```

to:

```text
Do the slots actually implement different functions?
```

The current answer is:

```text
Mostly no after ProposalNet cross-attention.
```

---

## 46. Diagnostic lessons

### Selection diversity is not functional diversity

Uniform slot usage does not imply meaningfully different actions.

### Candidate index and candidate function are different concepts

A router can distribute traffic evenly across functionally identical candidates.

### Stagewise measurement is necessary

Final `delta_q` diversity alone cannot locate whether collapse begins in:

```text
proposal
grounding
fusion
executor
readout
```

### Per-timestep measurement matters

In this case it showed recurrence does not rescue specialization.

### Metric definitions must stay consistent

Pairwise cosine and effective-rank semantics were intentionally reused across stages.

### Diagnostic tensors must be true snapshots

`detach()` is not a copy. Parameter-backed views that survive an optimizer step require cloning if they are used as forward-time diagnostics.

---

## 47. Audit commands

Full functional + ProposalNet audit:

```bash
python src/train.py \
  model=iag_srme \
  experiment=iag_srme_functional_audit \
  experiment.epochs=1
```

Latest metrics:

```bash
LATEST=$(find outputs -name metrics.jsonl -printf '%T@ %p\n' \
  | sort -nr | head -1 | cut -d' ' -f2-)
```

Full-pipeline analyzer:

```bash
python src/analyze_functional_collapse.py "$LATEST"
```

ProposalNet analyzer:

```bash
python src/analyze_proposal_internal.py "$LATEST"
```

---

## 48. Current branch state

```text
Branch:
exp/e2e-iag-srme-v2-r0-functional-collapse-audit

Current audited HEAD:
a259b378c596beeb2a26c3e04db6911fd1bd4943
```

Commit sequence:

```text
181e7d7abaf12556694cc8439ac64aab778c4775
    Add functional candidate collapse audit

86afb0ba2bd2ed0fb45e8b6f5051b5b5eed5a44e
    Fix functional collapse audit localization

43e90a1134ab3848143dfabdd71704422314e0b1
    Add ProposalNet internal collapse audit

a259b378c596beeb2a26c3e04db6911fd1bd4943
    Snapshot ProposalNet audit queries before optimizer update
```

---

## 49. Diagnosis in one diagram

```text
Learned slot queries
cos ~ -0.006
rank ~ 3.996 / 4
spread ~ 0.869
        |
        | query_conditioner
        | mostly common-mode residual
        v
Conditioned / normalized queries
cos ~ 0.627
rank ~ 3.114 / 4
spread ~ 0.529
        |
        | CROSS-ATTENTION
        | shared instruction K/V
        | no q residual afterward
        v
Proposal output
cos ~ 0.9994
rank ~ 1.060 / 4
spread ~ 0.0175
        |
        v
Grounder
alpha JS ~ 0
mask IoU ~ 0.995
        |
        v
Entities
rank ~ 1.002
        |
        v
Actions
rank ~ 1.034
        |
        v
Executor
delta rank ~ 1.004
        |
        v
Retrieval effect
delta_q rank ~ 1.078
        |
        v
ScoreNet / routing
index imbalance becomes a downstream symptom
```

---

## 50. Confidence levels

### Very high confidence

- learned slot queries are not collapsed;
- LayerNorm is not the primary collapse point;
- ProposalNet output is severely collapsed;
- the token-attention mapping is the dominant measured place where diversity disappears;
- functional collapse exists without Loss-Free;
- routing balancing alone cannot solve this failure;
- collapse persists across all recurrent steps.

### High confidence

- query conditioner contributes a strong common-mode component;
- downstream grounding/action/execution collapse is largely inherited from proposal collapse;
- sparse DPP validity limits its ability to rescue the system.

### Plausible but not directly measured

- candidate attention-weight distributions themselves become nearly identical;
- preserving query identity through a residual will fix most of the collapse;
- the original index monopoly was mostly a downstream symptom of candidate equivalence.

---

## 51. Explicit non-conclusions

Current evidence does **not** yet prove:

- residual cross-attention improves final retrieval;
- attention weights are exactly identical;
- Grounder has no independent collapse problem;
- conditioner is harmless;
- DPP is unnecessary;
- Loss-Free is unnecessary;
- ModeSeq is unnecessary;
- Executor initialization has no effect;
- more diversity is always better.

The next experiment must remain an isolated ablation.

---

## 52. Recommended immediate next step

Implement exactly one architecture change:

```text
Current:
proposal = CrossAttention(q, T, T)

Test:
proposal = LayerNorm(q + CrossAttention(q, T, T))
```

Keep:

- dataset;
- seed if possible;
- optimizer;
- objective;
- DPP;
- scorer;
- STOP rule;
- number of candidates;
- recurrence;
- current audit instrumentation.

Change one factor at a time.

Run one epoch and compare the full audit against this baseline.

---

## 53. Core research hypothesis

> Candidate specialization is currently lost because cross-attention replaces the candidate query with a shared text-derived value mixture. Preserving the candidate query through a residual path should retain slot identity, increase proposal diversity, propagate diversity into grounding/execution, and reduce functional candidate collapse.

Formally:

Current:

\[
e_k=A_kV.
\]

Proposed:

\[
e_k=\operatorname{LN}(q_k+A_kV).
\]

Primary structural prediction:

\[
r_{\mathrm{eff}}(e):
1.06
\rightarrow
\text{significantly larger}.
\]

If proposal diversity recovers but downstream diversity does not, the next bottleneck becomes Grounder/Executor rather than ProposalNet.

---

# Appendix A — Key numbers at a glance

## Retrieval

```text
Baseline, 1 epoch:
mean_recall = 15.131

Loss-Free, 1 epoch:
mean_recall = 14.871
```

## Baseline functional collapse

```text
functional_pairwise_cosine mean = 0.9988968
functional_rank mean            = 1.0773524
```

## Full pipeline

```text
proposals:
cosine = 0.9994
rank   = 1.0604
spread = 0.0175

alpha_read:
cosine = 1.0000
JS     = 0.0000

exec_mask:
cosine   = 1.0000
soft IoU = 0.9949

entities:
cosine = 1.0000
rank   = 1.0023

actions:
cosine = 0.9998
rank   = 1.0335

delta:
cosine = 0.9958
rank   = 1.0044

delta_q:
cosine = 0.9985
rank   = 1.0781
```

## ProposalNet internal

```text
base_query:
cosine = -0.0059
rank   = 3.9955
spread = 0.8685

conditioned_residual:
cosine = 0.9937
rank   = 1.2460
spread = 0.0686

query_post_norm:
cosine = 0.6265
rank   = 3.1139
spread = 0.5291

proposal_output:
cosine = 0.9994
rank   = 1.0604
spread = 0.0175
```

Derived:

```text
conditioner_to_base_norm_ratio = 1.278412
base_pre_cosine                = 0.629162
mean_relative_displacement     = 1.278412
attention_diversity_retention  = 0.032722
attention_rank_retention       = 0.340451
```

---

# Appendix B — Definitions

**Candidate index collapse**  
One fixed slot index receives most selections.

**Proposal collapse**  
Candidate representations become almost identical.

**Grounding collapse**  
Candidate grounding/read/write maps become almost identical.

**Functional collapse**  
Executing different candidates causes almost the same state/query consequence.

**Effective rank**

\[
r_{\mathrm{eff}}
=
\frac{(\sum_i\sigma_i)^2}{\sum_i\sigma_i^2}.
\]

**Relative spread**

\[
\frac{
\frac{1}{K}\sum_k\|z_k-\bar z\|_2
}{
\frac{1}{K}\sum_k\|z_k\|_2+\epsilon
}.
\]

**Loss-Free routing bias**  
A non-gradient historical controller used only to influence routing selection frequency.

**Attention diversity retention**

\[
\frac{
\text{relative spread of proposal output}
}{
\text{relative spread of query_post_norm}+\epsilon
}.
\]

---

# Appendix C — Minimal experiment protocol going forward

For each architecture ablation:

1. keep the current audits enabled;
2. run the same one-epoch protocol;
3. save `metrics.jsonl`;
4. run both analyzers;
5. compare structural metrics before looking at final retrieval;
6. only continue to longer training if the intended failure mode actually improves;
7. change one architectural factor at a time.

The next research question is therefore not initially:

```text
Does residual attention improve recall?
```

but:

```text
Does residual attention stop ProposalNet from destroying candidate identity?
```

Only after that is answered should final retrieval decide whether the architectural change is useful.

---

## Final diagnosis

The investigation began with:

\[
\boxed{\text{one candidate index wins too often}}
\]

but the final stagewise diagnosis is:

\[
\boxed{\text{candidate identity is largely erased by ProposalNet cross-attention}}
\]

The learned slots are healthy:

\[
r_{\mathrm{eff}}\approx4.
\]

Conditioning weakens them but preserves substantial diversity:

\[
r_{\mathrm{eff}}\approx3.11.
\]

Cross-attention then collapses them:

\[
r_{\mathrm{eff}}\approx1.06.
\]

Grounding and execution inherit that collapse:

\[
\alpha_k\approx\alpha_j,
\qquad
m_k\approx m_j,
\qquad
a_k\approx a_j,
\qquad
\Delta q_k\approx\Delta q_j.
\]

Therefore the correct current priority is:

\[
\boxed{\text{repair candidate identity preservation at cross-attention}}
\]

before spending additional effort on routing balance, DPP strength, aWTA, ModeSeq, or other downstream anti-collapse mechanisms.