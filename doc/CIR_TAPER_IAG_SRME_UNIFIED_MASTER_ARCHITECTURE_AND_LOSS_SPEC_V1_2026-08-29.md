# CIR / TAPER — IAG-SRME Unified Master Architecture + Loss Specification V1

**Project:** Composed Image Retrieval (CIR) / TAPER / SRME  
**Unified working name:** **IAG-SRME** — Intent-Anchored Grounded State-Recomputed Marginal Execution  
**Date:** 2026-08-29  
**Document type:** Canonical consolidation / research + implementation master checkpoint  
**Empirical status:** architecture and candidate objectives are specified; no empirical success is claimed by this document  
**Candidate count:** \(K=4\) learnable edit queries  
**Default recurrent horizon:** \(T_{\max}=3\)  
**Default internal width:** \(d=256\) unless a controlled capacity ablation changes it  
**Training philosophy:** one complete graph from optimizer update 1; no module curriculum, no horizon curriculum, no freeze-then-unfreeze schedule inside one run  
**Inference inputs:** reference image + modification text only  
**Target image:** training/evaluation signal only; never a forward-path input at inference  

---

# 0. Read this first — what this document consolidates

This file consolidates four research objects into one coherent source of truth:

1. **TAPER-SRME V5 — State-Recomputed Marginal Execution**
   - establishes the recurrent execution philosophy;
   - four candidate edits;
   - counterfactual preview;
   - marginal action value;
   - hard execute-or-STOP;
   - target-as-evaluator firewall;
   - true joint end-to-end training.

2. **IAG-SRME V1 — Intent-Anchored Grounded Token Editing**
   - supersedes the old proposal-generation path at the architecture level;
   - separates text-only **WHAT** from visual **WHERE**;
   - keeps an immutable visual anchor \(A\);
   - edits a persistent mutable token state \(Z_t\);
   - uses stable text edit intents and reference-anchored sparse supports;
   - recomputes current evidence, edit context, counterfactual effect, and marginal value after every executed edit.

3. **Mutual Complementary Claim + Contrastive Action-Claim Binding**
   - candidate auxiliary losses:
     \[
     L_{\mathrm{comp}},\qquad L_{\mathrm{bind}}.
     \]
   - operates primarily on the **textual edit decomposition / semantic responsibility** layer.

4. **PAIR-Generalized Action-Region Factor + Unique Contribution**
   - candidate auxiliary losses:
     \[
     L_{\mathrm{factor}},\qquad L_{\mathrm{unique}}.
     \]
   - operates on the **bound WHAT+WHERE factor** layer and its relational role across a minibatch.

## 0.1 Source precedence rule

When the parent SRME V5 and IAG-SRME V1 differ, this document follows:

\[
\boxed{\text{IAG-SRME V1 architecture} > \text{SRME V5 architecture details}}
\]

while retaining the SRME principles that IAG explicitly preserves:

- counterfactual preview;
- target-free score field;
- detached training-only marginal evaluator;
- hard execute/STOP;
- persistent recurrent state;
- model-driven rollout;
- one continuous joint optimization run;
- target-free inference.

The most important architectural change is:

### Original SRME intuition

\[
(q_k,X,Z_t)\rightarrow\text{state-conditioned proposal/action}
\]

so proposal semantics and visual support could both change with \(Z_t\).

### Current IAG-SRME

\[
\boxed{
q_k\rightarrow X\rightarrow e_k
\rightarrow A\rightarrow p_k
\rightarrow Z_t\rightarrow g_{t,k}
\rightarrow h^{ctx}_{t,k}
\rightarrow \Delta Z_{t,k}
}
\]

where:

- \(e_k\) is a stable text-derived edit intent;
- \(p_k\) is a stable reference-anchored spatial support;
- current-state dependence begins at \(g_{t,k}\) and continues through context, edit consequence, and action value.

## 0.2 Loss-status rule

There are **six named losses in the current research inventory**, but they are not all equally canonical.

### Canonical IAG-SRME core

\[
\boxed{
L_{\mathrm{core}}
=
L_{\mathrm{terminal}}
+
\lambda_mL_{\mathrm{marginal}}
}
\]

These are the two losses explicitly active in the IAG-SRME V1 main method.

### Candidate anti-collapse / factorization objectives

\[
\boxed{
L_{\mathrm{comp}},
\quad
L_{\mathrm{bind}},
\quad
L_{\mathrm{factor}},
\quad
L_{\mathrm{unique}}
}
\]

These are mathematically specified research candidates. They have **not** automatically been promoted into the IAG-SRME canonical training objective.

If all four are eventually activated, the resulting experimental super-objective is:

\[
\boxed{
\begin{aligned}
L_{\mathrm{six}}
={}&
L_{\mathrm{terminal}}
+\lambda_mL_{\mathrm{marginal}}\\
&+\lambda_cL_{\mathrm{comp}}
+\lambda_bL_{\mathrm{bind}}\\
&+\lambda_fL_{\mathrm{factor}}
+\lambda_uL_{\mathrm{unique}}.
\end{aligned}
}
\]

This equation describes the complete six-loss research inventory. It must **not** be interpreted as a recommendation to turn all six on simultaneously before the auxiliary losses pass the required smoke tests and incompatibility audits.

---

# 1. Core scientific intuition

The current method models CIR as an internal grounded editing process:

\[
\boxed{
\text{WHAT}
\rightarrow
\text{WHERE}
\rightarrow
\text{WHAT IS THERE NOW}
\rightarrow
\text{HOW TO CHANGE IT}
\rightarrow
\text{WHAT WOULD HAPPEN}
\rightarrow
\text{IS IT WORTH EXECUTING NOW?}
}
\]

For a CIR triplet:

\[
(R,M,Y),
\]

where:

- \(R\): reference image;
- \(M\): modification text;
- \(Y\): positive target image, available only for training/evaluation;

the model should not immediately collapse reference + text into one opaque global vector.

Instead:

1. extract several latent edit hypotheses from text;
2. ground each edit to the reference image;
3. inspect the current state of that region;
4. instantiate the edit in that current context;
5. generate a local token-level counterfactual transition;
6. preview the retrieval consequence;
7. predict whether that transition is useful now;
8. execute exactly one candidate or STOP;
9. repeat on the actually changed state.

The basic recurrence is:

\[
\boxed{
Z_{t+1}
=
F_\theta(Z_t,A,e_{k_t})
}
\]

with:

\[
k_t\in\{1,\ldots,K,\mathrm{STOP}\}.
\]

---

# 2. Critical state separation: immutable anchor vs mutable state

The original visual representation must never be destroyed.

Define:

\[
\boxed{
A\equiv Z_0^{\mathrm{anchor}}
}
\]

as immutable reference tokens.

Initialize the mutable state:

\[
\boxed{
Z_0=A.
}
\]

Then only:

\[
Z_t
\]

is edited recurrently.

So:

```text
A
= what the reference originally was
= immutable

Z_t
= what the model currently believes the edited state is
= mutable
```

This separation lets later steps distinguish:

- original identity;
- current state;
- accumulated changes.

The accumulated token displacement is:

\[
D_t=Z_t-A.
\]

---

# 3. Backbone and resource contract

## 3.1 Current planned backbone strata

The IAG-SRME specification records two separate complete experiments:

### Regime A — FG-CLIP-Base full fine-tuning

Train from update 1 to the end:

- full image encoder;
- full text encoder;
- image/text projections;
- all IAG-SRME modules.

Because the image encoder changes:

- reference patch tokens cannot be permanently cached;
- target embeddings cannot be stale cached embeddings;
- gallery features must be recomputed with the final checkpoint.

### Regime B — FG-CLIP-Large, frozen vision + full text fine-tuning

Frozen throughout:

- image encoder;
- image projection.

Trainable throughout:

- full text encoder;
- text projection;
- all IAG-SRME modules.

Because vision is frozen:

- reference tokens may be cached;
- target/gallery image embeddings may be cached if the exact preprocessing/checkpoint contract is respected.

These are **two independent runs**, not a freeze-then-unfreeze curriculum.

## 3.2 FG-CLIP vs FG-CLIP2

The current IAG-SRME V1 explicitly records the stated plan as **FG-CLIP Base/Large**, not silently FG-CLIP2.

FG-CLIP2 remains a future backbone ablation unless the backbone decision is intentionally changed.

## 3.3 LLM/MLLM policy

The CIR method itself uses no LLM/MLLM for:

- edit decomposition;
- generated captions;
- generated action labels;
- target captioning;
- localization;
- hard-negative reasoning;
- reranking;
- inference.

However, the selected pretrained FG-CLIP checkpoint has an upstream LMM-generated-caption lineage.

Therefore the safe claim is:

> No LLM/MLLM is used inside the CIR method, CIR data pipeline, training procedure, or inference procedure; pretrained FG-CLIP is allowed as initialization.

Do **not** claim that no generated data exists anywhere in the full upstream backbone lineage.

---

# 4. Unified notation contract

The source documents reuse the symbol \(c\) for different concepts. This unified specification removes that ambiguity.

| Unified symbol | Shape | Meaning |
|---|---:|---|
| \(R\) | image | reference image |
| \(M\) | text | modification instruction |
| \(Y\) | image / embedding | training target only |
| \(A\) | \([B,N,d]\) | immutable reference token anchor |
| \(Z_t\) | \([B,N,d]\) | mutable visual token state |
| \(X\) | \([B,L,d]\) | projected contextual text tokens |
| \(Q\) | \([K,d]\) | learnable query identities |
| \(e_k\) | \([B,d]\) | stable text-only edit intent |
| \(P\) | \([B,K,N]\) | sparse reference grounding |
| \(p_{k,n}\) | scalar | grounding weight of candidate \(k\) at visual token \(n\) |
| \(g_k^0\) | \([B,d]\) | original grounded visual evidence |
| \(g_{t,k}\) | \([B,d]\) | current grounded visual evidence |
| \(d_{t,k}\) | \([B,d]\) | accumulated local-change evidence |
| \(h^{ctx}_{t,k}\) | \([B,d]\) | Grounded Edit Context |
| \(\Delta Z_{t,k}\) | \([B,N,d]\) | candidate token intervention |
| \(\widehat Z_{t+1}^{(k)}\) | \([B,N,d]\) | candidate next state |
| \(q_t\) | \([B,D_g]\) | current retrieval query |
| \(\widehat q_{t+1}^{(k)}\) | \([B,D_g]\) | candidate retrieval query |
| \(\delta q_{t,k}\) | \([B,D_g]\) | candidate query effect |
| \(s_{t,k}\) | scalar | predicted marginal action value |
| \(m^{claim}_{k,i}\) | scalar | optional text semantic claim probability |
| \(z_k^{claim}\) | \([B,d]\) | optional pooled semantic claim target |
| \(f_{i,k}\) | \([D_f]\) | optional bound WHAT+WHERE factor |
| \(u_i\) | \([D_f]\) | optional target-free full-query auxiliary anchor |

Default:

\[
K=4,\qquad T_{\max}=3,\qquad d=256.
\]

---

# 5. Full network before recurrence

## 5.1 Reference image path

Reference image:

\[
R
\]

passes through the image encoder:

\[
H_R=E_I(R).
\]

Keep:

- native/global retrieval representation \(r_{\mathrm{ref}}\);
- local patch tokens.

Project local patch tokens:

\[
\boxed{
A=
\operatorname{LN}
(W_AH_{R,\mathrm{patch}})
}
\]

and initialize:

\[
Z_0=A.
\]

## 5.2 Text path

Modification text passes through the text encoder:

\[
H_M=E_T(M).
\]

Project contextual token states:

\[
\boxed{
X=
\operatorname{LN}
(W_XH_M)
}
\]

using a content mask that excludes padding and any special positions that should not carry semantic claim/attention mass.

A masked global text summary may be defined for retrieval readout:

\[
m
=
\frac{\sum_i C_iX_i}
{\sum_iC_i+\epsilon}.
\]

This summary must **not** become an unrestricted direct text-to-final-query bypass.

---

# 6. Four learnable queries produce four text-only edit intents

There are four learnable query identities:

\[
Q=[q_1,q_2,q_3,q_4],
\qquad q_k\in\mathbb R^d.
\]

They are not assigned human semantics such as:

```text
q1 = color
q2 = shape
q3 = object
q4 = texture
```

Such fixed semantic assignment is forbidden.

Each query independently reads the **entire** modification sequence:

\[
\widetilde e_k
=
q_k+
\operatorname{MHA}_{text}(q_k,X,X;C)
\]

followed by a residual FFN:

\[
\boxed{
e_k
=
\operatorname{LN}
\left(
\widetilde e_k+
\operatorname{FFN}(\widetilde e_k)
\right).
}
\]

Collect:

\[
E=[e_1,e_2,e_3,e_4]
\in\mathbb R^{B\times K\times d}.
\]

## 6.1 Crucial information boundary

At this point:

\[
\boxed{
e_k\text{ sees text but does not see }A\text{ or }Z_t.
}
\]

Operationally:

\[
\boxed{
e_k
=
\text{a text-derived latent variable that conditions a grounded executable intervention.}
}
\]

It is called an “edit intent” only if downstream intervention tests support that interpretation.

## 6.2 No candidate-axis competition

All four queries may read the same token.

There is no mandatory:

- softmax across candidates;
- Sinkhorn partition;
- one-token-one-slot ownership;
- non-overlap constraint.

Example:

```text
"make the sleeves red and longer"
```

Both a recolor intent and a length intent may need the token `"sleeves"`.

Therefore:

\[
\boxed{
\text{read access}\neq\text{semantic ownership}.
}
\]

This distinction becomes essential when \(L_{\mathrm{comp}}\) and \(L_{\mathrm{bind}}\) are introduced.

## 6.3 Stable within one trajectory

Canonical IAG-SRME computes:

\[
\boxed{
E=f_\theta(Q,X)
}
\]

once per sample and reuses it at every timestep.

Thus:

\[
e_k^{(0)}=e_k^{(1)}=\cdots=e_k.
\]

The semantic instruction is stable; its current usefulness is not.

---

# 7. Edit intent grounds to the immutable reference — WHERE

For each edit intent \(e_k\), compute compatibility with every anchor token \(A_n\):

\[
\ell_{k,n}
=
\frac{
(W_Q^ge_k)^\top(W_K^gA_n)
}{
\sqrt{d_g}
}.
\]

Normalize **over visual tokens for each candidate separately**:

\[
\boxed{
p_{k,:}
=
\operatorname{entmax}_{1.5}
(\ell_{k,:}).
}
\]

Therefore:

\[
p_{k,n}\ge 0,\qquad
\sum_np_{k,n}=1,
\]

and many entries may be exactly zero.

## 7.1 Why entmax

Softmax makes every finite-logit token receive nonzero mass.

Entmax permits:

\[
p_{k,n}=0,
\]

which gives a stronger locality contract.

## 7.2 This is not Slot Attention

Candidates do not compete over the same patch.

Each candidate forms its own support independently.

Therefore two edits may:

- fully overlap;
- partially overlap;
- be disjoint.

That is necessary for legitimate cases such as:

```text
recolor sleeves
lengthen sleeves
```

## 7.3 Stable anchor support

Because:

\[
p_k=f(e_k,A)
\]

and both \(e_k\) and \(A\) are stable inside the trajectory:

\[
\boxed{
p_k\text{ is stable across timesteps in canonical IAG-SRME.}
}
\]

This is a deliberate design choice.

State dependence enters through the **values read under that support**.

---

# 8. At timestep \(t\): read the current state under each support

For each candidate \(k\), compute three summaries.

## 8.1 Original grounded evidence

\[
\boxed{
g_k^0
=
\sum_n
p_{k,n}
W_0^gA_n.
}
\]

Meaning:

> What originally existed at this grounded location?

## 8.2 Current grounded evidence

\[
\boxed{
g_{t,k}
=
\sum_n
p_{k,n}
W_V^gZ_{t,n}.
}
\]

Meaning:

> What does this location contain now?

## 8.3 Accumulated local change

\[
\boxed{
d_{t,k}
=
\sum_n
p_{k,n}
W_D^g(Z_{t,n}-A_n).
}
\]

Meaning:

> How much / in what direction has this grounded location already changed?

Hence candidate \(k\) has:

```text
e_k      = WHAT text requests
p_k      = WHERE it applies
g_k^0    = WHAT was there originally
g_t,k    = WHAT is there now
d_t,k    = WHAT has already changed
```

---

# 9. Grounded Edit Context — HOW TO CHANGE IT NOW

Fuse:

\[
e_k,\quad
g_k^0,\quad
g_{t,k},\quad
d_{t,k}.
\]

Construct:

\[
u_{t,k}
=
[
e_k;
g_k^0;
g_{t,k};
d_{t,k};
e_k\odot g_{t,k};
e_k-g_{t,k}
].
\]

Then:

\[
\boxed{
h^{ctx}_{t,k}
=
\operatorname{LN}
\left(
W_2
\operatorname{GELU}
(W_1u_{t,k})
+
W_ee_k
\right).
}
\]

The residual from \(e_k\):

- preserves the requested edit semantics;
- while the MLP turns it into a current-state-conditioned operation.

Example:

```text
e_k:
"make red"

g_k^0:
"black garment region"

g_t,k:
"currently partly red / still black"

d_t,k:
"already moved this much from the original"

h_ctx:
"how the red edit should be instantiated on this region now"
```

---

# 10. Shared token editor — generate four counterfactual edit states

The executor is **shared** across:

- all four candidates;
- all recurrent timesteps.

There are not four independent edit networks.

For candidate \(k\), token \(n\):

\[
h_{t,k,n}
=
\operatorname{SiLU}
\left(
W_z\operatorname{LN}(Z_{t,n})
+
W_a\operatorname{LN}(A_n)
+
W_ch^{ctx}_{t,k}
+
W_d(Z_{t,n}-A_n)
\right).
\]

Predict a bounded direction:

\[
\boxed{
r_{t,k,n}
=
\tanh(W_\Delta h_{t,k,n}).
}
\]

Rescale sparse support:

\[
\widetilde p_{k,n}
=
\frac{
p_{k,n}
}{
\max_jp_{k,j}+\epsilon
}.
\]

Then:

\[
\boxed{
\Delta Z_{t,k,n}
=
\lambda_z
\widetilde p_{k,n}
r_{t,k,n}.
}
\]

The IAG specification proposes a small bounded \(\lambda_z\), with an initial search around:

\[
\lambda_z\in[0.05,0.15].
\]

A nominal starting point is:

\[
\lambda_z=0.10.
\]

No within-run annealing/curriculum is required by the method.

## 10.1 Exact locality property

If:

\[
p_{k,n}=0,
\]

then:

\[
\boxed{
\Delta Z_{t,k,n}=0.
}
\]

So outside the candidate support, the state is exactly unchanged by that candidate.

## 10.2 Four parallel previews

Every candidate branches from the **same parent state**:

\[
\boxed{
\widehat Z_{t+1}^{(k)}
=
Z_t+\Delta Z_{t,k}.
}
\]

Therefore at timestep \(t\):

\[
Z_t
\rightarrow
\{
\widehat Z_{t+1}^{(1)},
\widehat Z_{t+1}^{(2)},
\widehat Z_{t+1}^{(3)},
\widehat Z_{t+1}^{(4)}
\}.
\]

Candidate \(k+1\) is **not** generated from candidate \(k\).

These are four counterfactual worlds, not four sequential edits.

---

# 11. Retrieval readout from the edited token state

Let:

\[
D_t=Z_t-A.
\]

Use text-conditioned change pooling:

\[
\rho_{t,n}
=
w_r^\top
\operatorname{SiLU}
\left(
W_r\operatorname{LN}(Z_{t,n})
+
W_D^rD_{t,n}
+
W_m^rm
\right).
\]

Then:

\[
\beta_t
=
\operatorname{softmax}_n(\rho_{t,:}).
\]

Pool accumulated token change:

\[
d_t^{global}
=
W_{out}
\sum_n
\beta_{t,n}D_{t,n}.
\]

Use a bounded residual map:

\[
\operatorname{Cap}(x,c)
=
c\tanh(\|x\|_2/c)
\frac{x}{\|x\|_2+\epsilon}.
\]

Current retrieval query:

\[
\boxed{
q_t
=
\operatorname{Normalize}
\left(
r_{\mathrm{ref}}
+
\operatorname{Cap}
(d_t^{global},c_q)
\right).
}
\]

At:

\[
Z_0=A,
\]

we have:

\[
D_0=0
\Rightarrow
q_0=r_{\mathrm{ref}}.
\]

This provides a critical causal bottleneck:

\[
\boxed{
\text{modification-dependent final displacement must pass through token editing.}
}
\]

The global text summary may condition how changes are pooled, but it cannot add an unrestricted independent one-shot edit residual to the final query.

For every candidate preview:

\[
\boxed{
\widehat q_{t+1}^{(k)}
=
R_\theta
(
\widehat Z_{t+1}^{(k)},A,m,r_{\mathrm{ref}}
).
}
\]

Define the predicted retrieval-space effect:

\[
\boxed{
\delta q_{t,k}
=
\widehat q_{t+1}^{(k)}-q_t.
}
\]

---

# 12. Consequence-aware marginal scorer

A scorer that sees only:

\[
h^{ctx}_{t,k}
\]

would answer:

> Is this edit semantically relevant?

But SRME needs:

> If I execute this edit **now**, will retrieval improve?

So the scorer must see the predicted consequence.

Useful features include:

\[
\bar{\Delta z}_{t,k}
=
\sum_n
\widetilde p_{k,n}\Delta Z_{t,k,n},
\]

support fraction:

\[
\eta_{t,k}
=
\frac{1}{N}
\sum_n
\mathbf 1[p_{k,n}>0],
\]

and:

\[
\delta q_{t,k}.
\]

A canonical feature vector is:

\[
v_{t,k}
=
[
h^{ctx}_{t,k};
\delta q_{t,k};
\bar{\Delta z}_{t,k};
\|\delta q_{t,k}\|_2;
\|d_{t,k}\|_2;
\eta_{t,k}
].
\]

Shared score head:

\[
\boxed{
s_{t,k}=h_\theta(v_{t,k}).
}
\]

Interpretation:

\[
s_{t,k}>0
\Rightarrow
\text{predicted useful now}
\]

\[
s_{t,k}<0
\Rightarrow
\text{predicted harmful or redundant now}.
\]

The score head:

- has no target input;
- is shared across candidates;
- is shared across timesteps.

---

# 13. STOP and hard-forward execution

STOP is the identity transition:

\[
\widehat Z_{t+1}^{(\mathrm{STOP})}=Z_t,
\]

\[
\widehat q_{t+1}^{(\mathrm{STOP})}=q_t.
\]

Canonical STOP score:

\[
\boxed{
s_{t,\mathrm{STOP}}=0.
}
\]

Form:

\[
s_t=
[s_{t,1},s_{t,2},s_{t,3},s_{t,4},0].
\]

Training uses a hard-forward differentiable estimator from the beginning, e.g. a fixed-temperature straight-through Gumbel or sparse straight-through selector.

Forward:

\[
a_t^{hard}
=
\operatorname{onehot}
\left(
\arg\max_j
(s_{t,j}+\gamma_j)
\right).
\]

Backward uses the continuous surrogate.

Inference is deterministic:

\[
\boxed{
a_t=\arg\max_j s_{t,j}.
}
\]

If candidate \(k\) wins:

\[
Z_{t+1}=\widehat Z_{t+1}^{(k)}.
\]

If STOP wins:

\[
Z_{t+1}=Z_t,
\]

and the STOP state is absorbing for all remaining unrolled calls.

No soft-forward training phase is required before hard execution.

---

# 14. What is stable and what is recomputed

This is the core distinction between original SRME and current IAG-SRME.

## 14.1 Stable within one sample trajectory

\[
\boxed{
e_k
}
\]

textual edit intent.

\[
\boxed{
p_k
}
\]

reference-anchored support.

## 14.2 Recomputed at every timestep

\[
\boxed{
g_{t,k},
\quad
d_{t,k},
\quad
h^{ctx}_{t,k},
\quad
\Delta Z_{t,k},
\quad
\widehat Z_{t+1}^{(k)},
\quad
\delta q_{t,k},
\quad
s_{t,k}
}
\]

because they depend on the current mutable state \(Z_t\).

Therefore state dependence is expected **after grounding**, not necessarily in the intent or reference address.

---

# 15. Important cardinality semantics

There are always:

\[
K=4
\]

proposal identities.

This does **not** mean each sample has four true human edits.

A sample may effectively use:

- one candidate;
- two candidates;
- three candidates;
- repeated use of one candidate;
- STOP before most identities are used.

Likewise:

\[
\boxed{
T_{\max}
\neq
\text{ground-truth number of linguistic clauses}.
}
\]

The same latent intent may be executed more than once if useful.

Two human clauses may sometimes be realized by one latent intervention.

Therefore the method should claim:

> internal grounded latent interventions

not perfect symbolic program induction.

---

# 16. Canonical loss 1 — terminal retrieval

The final executed trajectory yields:

\[
q_i=q_{\mathrm{final}}^{(i)}.
\]

Let target/gallery embedding be:

\[
y_j.
\]

One standard query-to-target contrastive direction is:

\[
\boxed{
L_{q\rightarrow y}
=
-\frac1B
\sum_i
\log
\frac{
\exp(q_i^\top y_i/\tau)
}{
\sum_j\exp(q_i^\top y_j/\tau)
}.
}
\]

For datasets with multiple valid positives, use a multi-positive numerator.

If bidirectional training is appropriate:

\[
\boxed{
L_{\mathrm{terminal}}
=
\frac12
\left(
L_{q\rightarrow y}
+
L_{y\rightarrow q}
\right).
}
\]

## Role

\[
\boxed{
L_{\mathrm{terminal}}
:
\text{Does the final executed state retrieve the correct target?}
}
\]

## What it does not identify

Endpoint retrieval alone does **not** force the four candidates to specialize.

Many internal decompositions can produce the same final query.

This is the under-identification problem that motivates the auxiliary loss research.

---

# 17. Canonical loss 2 — counterfactual marginal action learning

At every live state, the model already generates:

\[
q_t
\]

and:

\[
\widehat q_{t+1}^{(1)},\ldots,\widehat q_{t+1}^{(K)}.
\]

Evaluate the current state and all candidate consequences with the **same target and the same negative set**.

Let:

\[
\ell_{\mathrm{ret}}(q;y_i,\mathcal N_i)
\]

be the training retrieval energy.

Define detached target-evaluated gain:

\[
\boxed{
u^*_{t,k}
=
\operatorname{sg}
\left[
\ell_{\mathrm{ret}}(q_t)
-
\ell_{\mathrm{ret}}
(\widehat q_{t+1}^{(k)})
\right].
}
\]

STOP utility:

\[
\boxed{
u^*_{t,\mathrm{STOP}}=0.
}
\]

Interpretation:

- \(u^*>0\): candidate improves retrieval;
- \(u^*=0\): candidate does nothing useful;
- \(u^*<0\): candidate harms retrieval.

The target is only an **evaluator**.

It does not construct:

- \(e_k\);
- \(p_k\);
- \(h^{ctx}_{t,k}\);
- \(\Delta Z_{t,k}\);
- \(s_{t,k}\).

Convert utilities into a detached target distribution:

\[
p_t^*
=
\operatorname{entmax}_{1.5}
\left(
u_t^*/\tau_u
\right).
\]

Predicted score distribution:

\[
\widehat p_t
=
\operatorname{entmax}_{1.5}
\left(
[s_{t,1},\ldots,s_{t,K},0]/\tau_s
\right).
\]

Use a matched Fenchel-Young objective or equivalent:

\[
\boxed{
L_{\mathrm{marginal}}
=
\frac{1}{|\mathcal L|}
\sum_{(i,t)\in\mathcal L}
L_{\mathrm{FY}}
(\widehat s_{i,t},p^*_{i,t}).
}
\]

Only live pre-STOP states contribute.

Suggested separate-run search:

\[
\lambda_m\in\{0.25,0.5,1.0\}.
\]

A natural first value is:

\[
\lambda_m=0.5.
\]

## Role

\[
\boxed{
L_{\mathrm{marginal}}
:
\text{Which model-generated edit should be executed now?}
}
\]

It trains action valuation/selection, not semantic factorization by itself.

---

# 18. Target firewall and gradient contract

The target may answer:

> Did this model-generated candidate improve retrieval?

It may not answer:

> What edit should the model generate?

Target data must never enter:

- text-intent generation;
- claim head inputs;
- visual grounding;
- current grounded evidence;
- Grounded Edit Context;
- token transition;
- score-head inputs;
- recurrent state;
- inference.

A target permutation test must verify that before loss computation:

\[
e_k,\,
p_k,\,
h^{ctx}_{t,k},\,
\Delta Z_{t,k},\,
\widehat q_{t+1}^{(k)},\,
s_{t,k}
\]

are unchanged when target IDs are shuffled.

## 18.1 Terminal gradient

\(L_{\mathrm{terminal}}\) should backpropagate through the real executed trajectory:

\[
q_{\mathrm{final}}
\rightarrow
Z_T
\rightarrow
\text{executed token edits}
\rightarrow
h^{ctx}_{t,k}
\rightarrow
g_{t,k}
\rightarrow
e_k
\rightarrow
\text{text encoder}.
\]

In full Base fine-tuning it also reaches the image encoder.

## 18.2 Marginal target detach

The utility:

\[
u^*
\]

is detached.

Therefore \(L_{\mathrm{marginal}}\) trains the predicted score field without directly letting target arithmetic push candidate transition vectors through the teacher utility.

## 18.3 Important Base full-finetune nuance

When the image encoder is trainable:

- the target branch may receive normal gradient through \(L_{\mathrm{terminal}}\);
- the same target embeddings used inside detached marginal utility must be treated as detached numerical evaluators in that marginal path.

Do not accidentally detach the target encoder from terminal retrieval entirely.

---

# 19. Candidate loss family A — semantic claim decomposition

The first auxiliary loss document introduces:

\[
\boxed{
L_{\mathrm{comp}}
\quad+\quad
L_{\mathrm{bind}}.
}
\]

Their purpose is to attack a failure such as:

```text
e1 ≈ full instruction
e2 ≈ full instruction
e3 ≈ full instruction
e4 ≈ full instruction
```

while preserving the rule:

> every query may still read the full instruction.

The key separation is:

\[
\boxed{
\text{text read attention}
\neq
\text{semantic ownership claim}.
}
\]

---

# 20. Optional claim head

For each text token \(X_i\) and candidate \(e_k\), construct an independent claim logit:

\[
a^{claim}_{k,i}
=
\frac{
(W_q^ce_k)^\top(W_t^cX_i)
}{
\sqrt{d_c}
}
+b_{k,i}
\]

or another **shared** compatibility function.

Then:

\[
\boxed{
m^{claim}_{k,i}
=
\sigma(a^{claim}_{k,i})
}
\]

with:

\[
0<m^{claim}_{k,i}<1.
\]

Sigmoid is intentional.

Candidates do not immediately compete through a slot-axis softmax.

Example:

```text
candidate 1 claim RED = 0.8
candidate 2 claim RED = 0.7
candidate 3 claim RED = 0.5
```

is initially legal; \(L_{\mathrm{comp}}\) supplies the pressure against redundant ownership.

The claim map must respect the text content mask.

---

# 21. Candidate loss 3 — Mutual Complementary Claim Consistency \(L_{\mathrm{comp}}\)

For candidate \(k\), consider what every peer candidate does **not** claim.

Peer \(j\)'s complement:

\[
1-m^{claim}_{j,i}.
\]

A raw intersection would be:

\[
\prod_{j\neq k}
(1-m^{claim}_{j,i}).
\]

Because a raw product shrinks strongly with \(K\), use geometric mean in log space:

\[
\boxed{
r_{k,i}
=
\exp
\left[
\frac{1}{K-1}
\sum_{j\neq k}
\log
(1-m^{claim}_{j,i}+\epsilon)
\right].
}
\]

This is large only when the peer set collectively leaves token \(i\) unclaimed.

Normalize candidate \(k\)'s own claim **shape**:

\[
\boxed{
p^{claim}_{k,i}
=
\frac{
m^{claim}_{k,i}
}{
\sum_\ell
m^{claim}_{k,\ell}
+\epsilon
}.
}
\]

Normalize peer residual:

\[
\boxed{
q^{peer}_{k,i}
=
\frac{
r_{k,i}
}{
\sum_\ell r_{k,\ell}
+\epsilon
}.
}
\]

Desired condition:

\[
p_k^{claim}\approx q_k^{peer}.
\]

Let:

\[
M_k
=
\frac12
(
p_k^{claim}
+
q_k^{peer}
).
\]

Jensen-Shannon:

\[
JS(p\Vert q)
=
\frac12
D_{KL}(p\Vert M)
+
\frac12
D_{KL}(q\Vert M).
\]

Finally:

\[
\boxed{
L_{\mathrm{comp}}
=
\frac1K
\sum_{k=1}^K
JS
(
p_k^{claim}
\Vert
q_k^{peer}
).
}
\]

## 21.1 Intuition

\[
\boxed{
\text{what I claim}
\approx
\text{what all my peers jointly leave for me}.
}
\]

Healthy example:

```text
A1 → RED
A2 → LONGER SLEEVES
A3 → REMOVE LOGO
```

Clone example:

```text
A1 → RED
A2 → RED
A3 → RED
```

is inconsistent because no candidate can say RED is “what peers leave for me” when the peers themselves also strongly claim RED.

## 21.2 Why JS rather than one-way KL

Both sides are learned jointly.

Neither is a fixed teacher.

JS is:

- symmetric;
- bounded;
- usually numerically gentler when the supports differ strongly.

## 21.3 What \(L_{\mathrm{comp}}\) does not solve

It does not guarantee:

- useful edit semantics;
- nonzero claim mass;
- factor existence;
- distinct downstream effects;
- human-interpretable edits.

Especially important:

normalizing the claim removes its total mass.

So:

\[
\boxed{
\text{claim shape health}
\neq
\text{claim mass health}.
}
\]

Raw claim mass must be logged:

\[
M_k^{claim}
=
\sum_i
m^{claim}_{k,i}.
\]

---

# 22. Claim-pooled semantic target

Pool text semantics using the claim:

\[
\boxed{
z_k^{claim}
=
\frac{
\sum_i
m^{claim}_{k,i}X_i
}{
\sum_i
m^{claim}_{k,i}
+\epsilon
}.
}
\]

Equivalently, normalized \(p_k^{claim}\) may be used.

This representation answers:

> What textual semantic evidence is candidate \(k\) claiming responsibility for?

---

# 23. Candidate loss 4 — Contrastive Action-Claim Binding \(L_{\mathrm{bind}}\)

The natural IAG-SRME mapping is:

\[
\boxed{
a_k\equiv e_k
}
\]

because \(e_k\) is the stable text-derived action/edit representation.

Project and normalize:

\[
\widetilde e_k
=
\operatorname{Normalize}
(W_ee_k),
\]

\[
\widetilde z_k
=
\operatorname{Normalize}
(W_zz_k^{claim}).
\]

Cross-slot similarity matrix:

\[
\boxed{
S_{k,j}
=
\widetilde e_k^\top
\widetilde z_j.
}
\]

Canonical one-way InfoNCE:

\[
\boxed{
L_{\mathrm{bind}}^{NCE}
=
-\frac1K
\sum_k
\log
\frac{
\exp(S_{k,k}/\tau_b)
}{
\sum_j
\exp(S_{k,j}/\tau_b)
}.
}
\]

Desired relative condition:

\[
S_{k,k}>S_{k,j},
\qquad j\neq k.
\]

## 23.1 Meaning

\[
\boxed{
L_{\mathrm{bind}}
:
\text{Does edit intent }e_k
\text{ actually represent the semantic evidence it claims?}
}
\]

It attacks the case where claim maps differ but all action representations still carry the same global sentence.

## 23.2 Negative set

The primary negatives should be **peer claims within the same sample**:

\[
z_j^{claim},
\quad
j\neq k.
\]

Why not blindly use cross-batch negatives?

Two different samples can both legitimately contain the same edit such as:

```text
red
longer
remove logo
```

so cross-batch negatives can create semantic false negatives.

## 23.3 Variants, not defaults

### Stop-gradient claim target

\[
\operatorname{sg}(z_k^{claim})
\]

can reduce mutual co-adaptation but prevents \(L_{\mathrm{bind}}\) from directly fixing the claim branch.

This is an ablation.

### Symmetric binding

Add the reverse:

\[
z_k^{claim}\rightarrow e_k
\]

and average both directions.

This is stronger but risks over-separating legitimate shared semantics.

It is an ablation, not the default.

## 23.4 Crucial limitation

\(L_{\mathrm{bind}}\) optimizes:

\[
\boxed{
\text{representation-level semantic discrimination}.
}
\]

It does **not** prove:

\[
\boxed{
\Delta Z_{t,1},
\Delta Z_{t,2},
\ldots
\text{ perform different useful functions}.
}
\]

Downstream intervention audits remain mandatory.

---

# 24. Candidate loss family B — bound WHAT+WHERE factorization

The second auxiliary loss document defines the meaningful unit as:

\[
\boxed{
F_{i,k}
=
\text{WHAT to change}
+
\text{WHERE to change}.
}
\]

This is intentionally different from enforcing:

\[
e_{i,1}\perp e_{i,2}
\]

or:

\[
p_{i,1}\cap p_{i,2}=\varnothing.
\]

Those conditions can be wrong.

Examples:

### Same operation, different region

```text
make sleeves longer
make hem longer
```

The WHAT vectors may legitimately be similar.

### Same region, different operation

```text
make shirt red
add stripes to shirt
```

The WHERE regions may legitimately overlap.

Thus the factor specialization target is:

\[
\boxed{
\text{bound action-region role}.
}
\]

---

# 25. Recommended IAG mapping for the bound factor

This mapping is a **unified integration recommendation**, not a claim that the original auxiliary document specified IAG internals.

Use the stable pre-rollout quantities:

\[
\boxed{
a_{i,k}=e_{i,k}
}
\]

and a reference-anchored region representation, naturally:

\[
\boxed{
r_{i,k}=g^0_{i,k}
}
\]

or an equivalent support-weighted anchor pooling.

Then construct:

\[
\boxed{
f_{i,k}
=
\operatorname{Norm}
\left(
\phi
[
e_{i,k};
r_{i,k};
e_{i,k}\odot r_{i,k};
|e_{i,k}-r_{i,k}|
]
\right).
}
\]

The projection/fusion network \(\phi\) must be shared across candidates and samples.

Why use the stable anchor region in the first integration?

- \(L_{\mathrm{factor}}\) is meant to describe the factor's WHAT+WHERE identity;
- using \(g^0_k\) avoids changing the factor definition at every rollout step;
- state-dependent HOW/consequence remains handled by the SRME recurrent path.

A state-dependent factor variant is a later ablation, not required by the initial formulation.

---

# 26. Full-query auxiliary anchor for relational geometry

Define a target-free full-query representation:

\[
\boxed{
u_i=\psi(R_i,M_i)
}
\]

using only reference + modification information.

It must **not** consume target \(Y_i\).

Recommended for auxiliary losses:

\[
\boxed{
\widetilde u_i
=
\operatorname{sg}(u_i).
}
\]

Important architectural firewall:

\[
\boxed{
u_i
\text{ is auxiliary-only and must not become a direct shortcut into }
Z_t,q_t,\text{ or the executor}.
}
\]

This prevents the factor-loss branch from accidentally reintroducing a one-shot full-text/full-image bypass around the token intervention path.

The non-detached \(u_i\) may still learn normally through a declared ordinary task branch if such a branch exists, but the first auxiliary implementation should treat the relational anchor as stopped inside \(L_{\mathrm{factor}}\) and \(L_{\mathrm{unique}}\).

---

# 27. Relational geometry of the full query

For sample \(i\), compare its auxiliary full-query anchor against every anchor in the minibatch:

\[
q_{ij}^{full}
=
\frac{
s(u_i,u_j)
}{
\tau_u
}
\]

with cosine similarity by default.

Then:

\[
\boxed{
p_{ij}
=
\frac{
\exp(q_{ij}^{full})
}{
\sum_m
\exp(q_{im}^{full})
}.
}
\]

So:

\[
p_i=[p_{i1},\ldots,p_{iB}]
\]

is the semantic-neighborhood fingerprint of the complete query.

---

# 28. Relational geometry of each action-region factor

Compare factor \(f_{i,k}\) against the same anchor bank:

\[
q_{i,k,j}^{factor}
=
\frac{
s(f_{i,k},u_j)
}{
\tau_f
}.
\]

Then:

\[
\boxed{
\pi_{i,k,j}
=
\frac{
\exp(q_{i,k,j}^{factor})
}{
\sum_m
\exp(q_{i,k,m}^{factor})
}.
}
\]

Thus:

\[
\pi_{i,k}
\]

is the relational fingerprint of factor \(k\) for sample \(i\).

---

# 29. Candidate loss 5 — \(L_{\mathrm{factor}}\): collective completeness

A literal product:

\[
\prod_k\pi_{i,k,j}
\]

shrinks strongly as \(K\) increases.

Use a normalized geometric Product-of-Experts in log space:

\[
\boxed{
\ell_{i,j}^{all}
=
\frac1K
\sum_{k=1}^K
\log
(
\pi_{i,k,j}+\epsilon
).
}
\]

Normalize:

\[
\boxed{
\widehat p_i^{all}
=
\operatorname{softmax}_j
(
\ell_i^{all}
).
}
\]

Then:

\[
\boxed{
L_{\mathrm{factor}}
=
\frac1B
\sum_i
D_{KL}
(
p_i
\Vert
\widehat p_i^{all}
).
}
\]

## 29.1 Meaning

\[
\boxed{
L_{\mathrm{factor}}
:
\text{all WHAT+WHERE factors together must explain the relational semantic geometry of the full query.}
}
\]

It asks for **collective completeness**.

## 29.2 Why this alone does not prevent clones

Suppose:

\[
\pi_{i,1}
\approx
\pi_{i,2}
\approx
\cdots
\approx
\pi_{i,K}
\approx
p_i.
\]

Then geometric mean is still:

\[
p_i.
\]

Therefore:

\[
L_{\mathrm{factor}}\approx0
\]

even if all factors are clones of the full query.

This exact loophole motivates \(L_{\mathrm{unique}}\).

---

# 30. Candidate loss 6 — \(L_{\mathrm{unique}}\): leave-one-factor-out marginal necessity

First define all-factor error:

\[
\boxed{
E_i^{all}
=
D_{KL}
(
p_i
\Vert
\widehat p_i^{all}
).
}
\]

For factor \(k\), remove it and recompute the geometric mean with \(K-1\):

\[
\boxed{
\ell_{i,j}^{(-k)}
=
\frac{1}{K-1}
\sum_{r\neq k}
\log
(
\pi_{i,r,j}+\epsilon
).
}
\]

Then:

\[
\boxed{
\widehat p_i^{(-k)}
=
\operatorname{softmax}_j
(
\ell_i^{(-k)}
).
}
\]

Leave-one-out error:

\[
\boxed{
E_{i,k}^{(-k)}
=
D_{KL}
(
p_i
\Vert
\widehat p_i^{(-k)}
).
}
\]

Marginal contribution:

\[
\boxed{
\Delta_{i,k}^{uniq}
=
E_{i,k}^{(-k)}
-
E_i^{all}.
}
\]

Interpretation:

- positive large: factor is needed;
- near zero: factor is replaceable/redundant;
- negative: removing the factor improves reconstruction; factor is harmful.

Require active factors to contribute at least a small margin \(\gamma\):

\[
\boxed{
L_{\mathrm{unique}}
=
\frac{1}{BK}
\sum_{i,k}
\operatorname{ReLU}
(
\gamma
-
\Delta_{i,k}^{uniq}
).
}
\]

## 30.1 Precise meaning of “unique”

This loss does **not** prove a human-interpretable unique concept.

It enforces:

\[
\boxed{
\text{non-redundant marginal necessity in the chosen relational geometry}.
}
\]

That distinction is mandatory.

---

# 31. Correct vectorized leave-one-out algebra

If:

```python
all_logits = log_pi.mean(dim=1)
```

then the correct leave-one-out is **not**:

```python
loo = all_logits[:, None, :] - log_pi
```

because the scaling changes.

Correct:

```python
total_log = log_pi.sum(dim=1, keepdim=True)
loo_logits = (total_log - log_pi) / (K - 1)
```

This preserves the geometric-mean semantics after removing one factor.

---

# 32. Batch self-match issue in \(L_{\mathrm{factor}}\) and \(L_{\mathrm{unique}}\)

The relational distributions may include:

\[
j=i.
\]

Then a shortcut can occur:

\[
p_{ii}\approx1,
\qquad
\pi_{i,k,i}\approx1,
\]

turning the auxiliary problem into sample identity matching.

Two modes should be tested.

## Mode A — PAIR-faithful

Include the diagonal.

## Mode B — self-masked geometry

Set:

\[
q_{ii}=-\infty
\]

before softmax.

Monitor:

- mean \(p_{ii}\);
- entropy of \(p_i\);
- entropy of \(\pi_{i,k}\).

If the diagonal dominates, the self-masked mode becomes scientifically important.

---

# 33. Temperature behavior for relational losses

Use separate:

\[
\tau_u
\]

for full-query relational geometry and:

\[
\tau_f
\]

for factor-to-anchor geometry.

Too small:

```text
p, pi → nearly one-hot
```

which can reduce the task to identity classification.

Too large:

```text
p, pi → nearly uniform
```

which removes useful semantic structure.

Select by both:

- task performance;
- distribution entropy;
- collapse diagnostics.

---

# 34. Six-loss functional responsibility map

| Loss | Level | Core question | Directly enforces | Does NOT prove |
|---|---|---|---|---|
| \(L_{\mathrm{terminal}}\) | final task | Did the final query retrieve the target? | endpoint CIR correctness | meaningful factorization |
| \(L_{\mathrm{marginal}}\) | recurrent control | Which candidate helps now? | target-free score prediction from detached consequences | semantic uniqueness |
| \(L_{\mathrm{comp}}\) | text ownership | Who claims which text evidence? | mutually bounded claim shapes | healthy mass / usefulness |
| \(L_{\mathrm{bind}}\) | text semantics | Does \(e_k\) encode its own claim more than peers? | representation-level semantic specialization | distinct executor function |
| \(L_{\mathrm{factor}}\) | WHAT+WHERE set | Do all bound factors jointly explain the full query geometry? | collective completeness | non-clone specialization |
| \(L_{\mathrm{unique}}\) | WHAT+WHERE necessity | Does removing factor \(k\) hurt? | relational marginal necessity | interpretability / no secret sharing |

A useful conceptual hierarchy is:

\[
\boxed{
L_{\mathrm{comp}}+L_{\mathrm{bind}}
\Rightarrow
\text{WHAT decomposition pressure}
}
\]

\[
\boxed{
L_{\mathrm{factor}}+L_{\mathrm{unique}}
\Rightarrow
\text{WHAT+WHERE set completeness/nonredundancy pressure}
}
\]

\[
\boxed{
L_{\mathrm{terminal}}+L_{\mathrm{marginal}}
\Rightarrow
\text{actual CIR task + recurrent execution pressure}
}
\]

But these implications are **pressures**, not mathematical guarantees of semantic factor recovery.

---

# 35. Most important unresolved incompatibility: four candidates vs variable true edit count

This is currently the most dangerous point in blindly combining the four auxiliary losses with IAG-SRME.

IAG-SRME always has:

\[
K=4
\]

proposal identities.

But a modification may effectively need:

\[
1,\ 2,\ 3,\ \text{or fewer than 4 active semantic factors}.
\]

Example:

```text
"make it red"
```

A healthy architecture may want:

```text
candidate 1 → RED
candidate 2 → weak/inactive alternative
candidate 3 → weak/inactive alternative
candidate 4 → weak/inactive alternative
```

## 35.1 Why \(L_{\mathrm{comp}}\) can conflict

\(L_{\mathrm{comp}}\) normalizes each candidate's claim into a distribution over tokens.

A tiny claim mass can still become a normalized shape.

So an otherwise inactive candidate may be pressured to invent a semantic responsibility.

## 35.2 Why \(L_{\mathrm{unique}}\) can conflict even more strongly

The fixed-\(K\) form says:

> removing every factor should hurt by at least \(\gamma\).

If all four candidates are treated as active on a one-edit sample, this can force the model to manufacture:

- artificial subfactors;
- secret-sharing fragments;
- arbitrary batch codes;

just to make all four “necessary.”

## 35.3 STOP is not a semantic NULL

Do not confuse:

### STOP

> Should the recurrent solver execute another action **now**?

with:

### factor inactivity / NULL

> Does candidate \(k\) correspond to any useful semantic factor at all?

These are different variables.

\[
\boxed{
\text{STOP}\neq\text{factor NULL}.
}
\]

## 35.4 Safe extension with activity weights

The factor-loss document already provides the correct future pattern.

For activity:

\[
w_{i,k}\ge0,
\]

replace the uniform factor mean:

\[
\ell_{i,j}^{all}
=
\frac{
\sum_kw_{i,k}
\log(\pi_{i,k,j}+\epsilon)
}{
\sum_kw_{i,k}+\epsilon
}.
\]

Then apply \(L_{\mathrm{unique}}\) only to factors declared active.

An analogous activity-aware policy is required before \(L_{\mathrm{comp}}\) is allowed to force every candidate to own a full normalized semantic distribution.

Until an activity/NULL contract is defined and smoke-tested:

\[
\boxed{
\text{do not blindly activate all four auxiliary losses in the main run.}
}
\]

---

# 36. Additional failure modes of the auxiliary losses

## 36.1 Secret sharing

Factors can become:

```text
F1 = code fragment A
F2 = code fragment B
F3 = code fragment C
F4 = code fragment D
```

where:

- all are required;
- jointly they reconstruct the full query;
- none is a meaningful edit.

Then:

\[
L_{\mathrm{factor}}\downarrow
\]

and:

\[
L_{\mathrm{unique}}\downarrow
\]

without real semantic decomposition.

Therefore:

\[
\boxed{
L_{\mathrm{unique}}
\text{ proves necessity, not interpretability.}
}
\]

## 36.2 Arbitrary batch coding

Factors may learn private sample-identity codes that reproduce minibatch relational geometry.

Required defenses/diagnostics:

- shared factor projector;
- no sample-ID input;
- held-out relational tests;
- cross-batch/memory-bank evaluation;
- semantic probes;
- intervention tests.

## 36.3 False-negative over-separation

Legitimate edits can share context.

Example:

```text
make shirt red
make shirt glossy
```

Both need the same entity evidence.

So no auxiliary objective should assume maximal pairwise orthogonality.

## 36.4 Claim-mass collapse

Normalized \(L_{\mathrm{comp}}\) may look healthy while raw claim mass approaches zero.

Always log:

\[
M_k^{claim}
=
\sum_i
m^{claim}_{k,i}.
\]

## 36.5 Representation specialization without functional specialization

Possible:

\[
e_1,e_2,e_3,e_4
\]

are distinguishable under \(L_{\mathrm{bind}}\),

yet the shared executor outputs:

\[
\delta q_{t,1}
\approx
\delta q_{t,2}
\approx
\delta q_{t,3}
\approx
\delta q_{t,4}.
\]

Therefore latent cosine/contrastive accuracy alone is insufficient.

## 36.6 Auxiliary anchor bypass

The full-query auxiliary anchor \(u_i\) is powerful.

If it is fed into the actual recurrent forward path, the model can bypass the intended token-edit bottleneck.

Therefore:

\[
\boxed{
u_i
\text{ is auxiliary-only unless a new architecture is explicitly declared.}
}
\]

---

# 37. Why these losses are not orthogonality losses

A pairwise orthogonality loss says:

\[
\cos(f_k,f_j)\rightarrow0.
\]

But two vectors can be orthogonal while performing the same downstream function.

Conversely, two valid factors may remain close because they share:

- an entity;
- an operation;
- a common visual region.

The proposed objectives instead ask:

- \(L_{\mathrm{comp}}\): evidence responsibility;
- \(L_{\mathrm{bind}}\): semantic binding;
- \(L_{\mathrm{factor}}\): joint completeness;
- \(L_{\mathrm{unique}}\): leave-one-out necessity.

This is a more task-aligned goal than arbitrary geometric separation.

---

# 38. Current recommended experimental ladder

The safest research path is **not** six losses at once.

## A0 — structural IAG-SRME baseline

\[
\boxed{
L
=
L_{\mathrm{terminal}}
+
\lambda_mL_{\mathrm{marginal}}.
}
\]

First prove:

- grounding is causally used;
- token editing is causally used;
- recurrence adds value;
- dynamic scoring adds value;
- repeat/clone does not already explain everything.

## A1 — claim consistency only

\[
L=A0+\lambda_cL_{\mathrm{comp}}.
\]

Question:

> Does ownership structure improve without harming retrieval or forcing artificial factor count?

## A2 — binding only

\[
L=A0+\lambda_bL_{\mathrm{bind}}.
\]

Question:

> Does semantic action specialization improve, and does it survive downstream?

## A3 — claim + binding pair

\[
L=A0+\lambda_cL_{\mathrm{comp}}
+\lambda_bL_{\mathrm{bind}}.
\]

Question:

> Does mutually bounded ownership plus semantic binding outperform either alone?

## B1 — factor completeness only

\[
L=A0+\lambda_fL_{\mathrm{factor}}.
\]

Question:

> Does the set of WHAT+WHERE factors better explain full-query relational geometry?

## B2 — factor completeness + unique contribution

\[
L=A0
+\lambda_fL_{\mathrm{factor}}
+\lambda_uL_{\mathrm{unique}}.
\]

Only valid after active/inactive factor semantics are handled safely.

## C1 — hierarchical combination

Only after A3 and B2 independently pass:

\[
L=
A0+
\lambda_cL_{\mathrm{comp}}
+\lambda_bL_{\mathrm{bind}}
+\lambda_fL_{\mathrm{factor}}
+\lambda_uL_{\mathrm{unique}}.
\]

This is the six-loss super-objective.

It is a **promotion candidate**, not the current default.

## 38.1 Coefficient policy

No auxiliary coefficient is claimed optimal.

Safe initial policy:

- keep task loss dominant;
- use small \(\lambda_c,\lambda_b,\lambda_f,\lambda_u\);
- typically start with:
  \[
  \lambda_u\le \lambda_f;
  \]
- use a small unique margin \(\gamma\);
- do not anneal losses on/off inside a run if claiming true one-graph training.

Each ablation should be a separate complete run with a fixed objective from update 1.

---

# 39. Full gradient-flow map

## 39.1 \(L_{\mathrm{terminal}}\)

Reaches:

```text
final query
→ readout
→ executed Z_T
→ selected token deltas
→ Grounded Edit Context
→ current grounded evidence
→ edit intents
→ text encoder
→ image encoder if full Base fine-tuning
```

## 39.2 \(L_{\mathrm{marginal}}\)

Teacher utility is detached.

Primary direct role:

```text
detached candidate-value target
→ marginal score field
```

The real executed trajectory still receives task gradients through \(L_{\mathrm{terminal}}\).

## 39.3 \(L_{\mathrm{comp}}\)

Reaches:

```text
claim head
→ e_k
→ text token representations
```

and mutually couples peer claim maps.

## 39.4 \(L_{\mathrm{bind}}\)

Canonical fully joint version reaches:

```text
e_k branch
↔ claim-pooled semantic branch
↔ text encoder
```

Stop-gradient claim-target variant intentionally removes one direction.

## 39.5 \(L_{\mathrm{factor}}\)

Recommended first integration:

```text
bound WHAT+WHERE factor f_k
→ edit intent e_k
→ anchor-grounded region representation
```

while the target-free full-query anchor is detached inside the auxiliary objective.

## 39.6 \(L_{\mathrm{unique}}\)

Same factor path as \(L_{\mathrm{factor}}\), but driven by leave-one-out marginal necessity.

It should not be applied to inactive/NULL factors.

---

# 40. End-to-end training requirement

One run must have its complete chosen graph and objective active from update 1.

Forbidden within the same claimed run:

- first train query decomposition, then executor;
- first train with \(T=1\), later \(T=3\);
- first no STOP, later enable STOP;
- first soft-forward, later hard-forward;
- first freeze backbone, later unfreeze;
- first train score network separately;
- teacher/oracle action roll-in;
- manual query semantic assignment;
- delayed activation of an auxiliary loss if the run is being claimed as the final six-loss end-to-end method.

Standard optimizer learning-rate warmup is allowed because it changes optimizer step size, not the graph semantics.

Separate complete ablation runs with different fixed loss sets are allowed and required.

---

# 41. Complete canonical pseudocode

```python
# -------------------------------------------------------
# Forward path: target-free
# -------------------------------------------------------

A, r_ref = image_backbone.encode_reference(reference)
X = text_backbone.encode_tokens(modification)

A = anchor_projection(A)
X = text_projection(X)

Z = A.clone()

# Stable text-only WHAT.
E = intent_encoder(
    query_bank,
    X,
    text_mask,
)  # [B,K,d]

# Stable reference-anchored WHERE.
P = anchor_grounder(
    E,
    A,
)  # [B,K,N], entmax1.5

q = readout(
    Z,
    A,
    X,
    r_ref,
)

alive = ones(B)

trace = []

for t in range(T_MAX):

    # What was there, what is there now, what changed?
    g0, gt, dt = grounded_reader(
        P,
        A,
        Z,
    )

    # HOW should each stable intent apply now?
    H_ctx = context_fuser(
        E,
        g0,
        gt,
        dt,
    )

    # Four parallel counterfactual token edits,
    # all from the same current parent state Z.
    delta_Z, Z_hat = transition(
        H_ctx,
        P,
        A,
        Z,
    )

    q_hat = readout_candidates(
        Z_hat,
        A,
        X,
        r_ref,
    )

    # Target-free predicted marginal values.
    scores = score_head(
        context=H_ctx,
        delta_Z=delta_Z,
        q_current=q,
        q_candidates=q_hat,
    )

    # STOP is identity with score 0.
    logits = cat(
        [scores, zeros_stop],
        dim=-1,
    )

    # Hard forward from update 1.
    action_st = selector(
        logits,
        training=self.training,
    )

    Z_next = select_state(
        candidate_states=Z_hat,
        stop_state=Z,
        action=action_st,
    )

    q_next = readout(
        Z_next,
        A,
        X,
        r_ref,
    )

    trace.append(
        {
            "current_query": q,
            "candidate_queries": q_hat,
            "contexts": H_ctx,
            "supports": P,
            "deltas": delta_Z,
            "scores": scores,
            "action": action_st,
        }
    )

    Z = absorbing_stop_update(
        Z,
        Z_next,
        action_st,
    )

    q = absorbing_stop_update(
        q,
        q_next,
        action_st,
    )

return q, trace
```

---

# 42. Canonical core training pseudocode

```python
q_final, trace = model(
    reference,
    modification,
    return_trace=True,
)

# ---------------------------------------------
# 1) Real endpoint CIR objective
# ---------------------------------------------

L_terminal = retrieval_loss(
    q_final,
    target_embeddings,
    negatives,
)

# ---------------------------------------------
# 2) Detached marginal action supervision
# ---------------------------------------------

L_marginal = 0.0

for step in live_pre_stop(trace):

    utility_star = detached_marginal_gain(
        current_query=step["current_query"],
        candidate_queries=step["candidate_queries"],
        target=target_embeddings,
        negatives=negatives,
        stop_utility=0.0,
    )

    L_marginal += marginal_prediction_loss(
        predicted_scores=step["scores"],
        detached_target_utility=utility_star,
    )

L_core = (
    L_terminal
    + lambda_m * L_marginal
)

L_core.backward()
optimizer.step()
```

---

# 43. Optional auxiliary-loss pseudocode

These branches are included only in the corresponding complete ablation run.

```python
# -------------------------------------------------------
# A) Text claim branch
# -------------------------------------------------------

claim_logits = claim_head(
    intents=E,          # [B,K,d]
    text_tokens=X,      # [B,L,d]
    text_mask=text_mask,
)

claim = sigmoid(claim_logits)  # [B,K,L]

L_comp = mutual_complementary_claim_loss(
    claim,
    text_mask,
)

z_claim = claim_weighted_text_pool(
    claim,
    X,
    text_mask,
)

L_bind = cross_slot_action_claim_infonce(
    actions=E,
    claimed_semantics=z_claim,
)

# -------------------------------------------------------
# B) Stable WHAT+WHERE factor branch
# -------------------------------------------------------

anchor_region = g0

factor = factor_fuser(
    what=E,
    where=anchor_region,
)

full_query_anchor = auxiliary_full_query_anchor(
    reference_features=A,
    text_features=X,
)

L_factor, L_unique = factor_unique_losses(
    factors=factor,
    anchor=full_query_anchor.detach(),
    active_weights=active_weights_if_defined,
)

# -------------------------------------------------------
# Candidate six-loss experiment
# -------------------------------------------------------

loss = (
    L_terminal
    + lambda_m * L_marginal
    + lambda_c * L_comp
    + lambda_b * L_bind
    + lambda_f * L_factor
    + lambda_u * L_unique
)
```

Critical:

```text
active_weights_if_defined
```

is not cosmetic.

Without a valid active/NULL semantics, \(L_{\mathrm{unique}}\) must not force all four proposal identities to become necessary on every sample.

---

# 44. Example complete forward trace

Reference:

```text
black dress
short sleeves
```

Modification:

```text
"make it red and make the sleeves longer"
```

## Text stage

Four queries read the full sentence:

```text
q1 → e1 ≈ recolor-related hypothesis
q2 → e2 ≈ sleeve-length-related hypothesis
q3 → weaker/alternative hypothesis
q4 → weaker/alternative hypothesis
```

These interpretations are illustrative, not supervised symbolic labels.

## Anchor grounding

```text
e1 → garment/color-support patches
e2 → sleeve patches
```

## Step 0

Current state:

```text
black dress
short sleeves
```

Candidate 1:

```text
e1
+ original garment evidence
+ current black garment evidence
+ no prior recolor change
→ H_ctx_0,1
→ local recolor delta
→ Zhat_1^(1)
→ qhat_1^(1)
```

Candidate 2:

```text
e2
+ original sleeve evidence
+ current short-sleeve evidence
→ H_ctx_0,2
→ local sleeve-shape delta
→ Zhat_1^(2)
→ qhat_1^(2)
```

All four candidates are previewed from the same \(Z_0\).

Possible predicted scores:

```text
candidate 1: +0.45
candidate 2: +0.28
candidate 3: -0.03
candidate 4: +0.02
STOP:         0.00
```

Execute candidate 1:

\[
Z_1=\widehat Z_1^{(1)}.
\]

## Step 1

Stable:

```text
e1, e2, e3, e4
p1, p2, p3, p4
```

Changed because \(Z_1\neq Z_0\):

```text
g_1,k
d_1,k
H_ctx_1,k
DeltaZ_1,k
deltaq_1,k
score_1,k
```

Now candidate 1 may have near-zero/negative gain because the recolor has already been performed.

Candidate 2 may remain positive.

Example:

```text
recolor candidate:       -0.02
length candidate:        +0.31
other candidates:        <= 0
STOP:                     0.00
```

Execute candidate 2.

## Step 2

If all candidates are predicted no better than identity:

```text
all candidate scores <= 0
STOP = 0
```

STOP.

Final query:

\[
q_{\mathrm{final}}
=
\operatorname{Normalize}
(
r_{\mathrm{ref}}
+
\text{bounded pooled accumulated token edit}
).
\]

Perform one final gallery retrieval.

---

# 45. Structural and functional diagnostics

Pretty attentions or low latent cosine are not enough.

## 45.1 Edit-intent diagnostics

Log:

- pairwise cosine of \(e_k\);
- covariance effective rank;
- cross-attention overlap;
- text-attention entropy;
- claim maps if enabled;
- raw claim mass if enabled.

Interpret as clues, not proof.

## 45.2 Grounding diagnostics

For every \(p_k\):

- support count;
- support fraction;
- entropy;
- maximum mass;
- pairwise support cosine;
- pairwise Jaccard of nonzero support;
- spatial connectedness only if useful.

Do not define low overlap as automatically good.

## 45.3 Grounding faithfulness interventions

For candidate \(k\), compare:

1. normal support;
2. selected region removed;
3. selected region shuffled;
4. complement-only region;
5. random same-size support.

Measure the impact on:

\[
g_{t,k},
\quad
h^{ctx}_{t,k},
\quad
\Delta Z_{t,k},
\quad
\widehat q_{t+1}^{(k)},
\quad
s_{t,k}.
\]

If grounding perturbation barely changes the candidate, grounding is cosmetic.

## 45.4 Functional candidate-effect matrix

Define:

\[
\delta q_{t,k}
=
\widehat q_{t+1}^{(k)}-q_t.
\]

Build:

\[
D_t
=
\begin{bmatrix}
\delta q_{t,1}\\
\vdots\\
\delta q_{t,K}
\end{bmatrix}.
\]

Compute singular values:

\[
\sigma_1,\ldots,\sigma_r.
\]

Normalize:

\[
\pi_i
=
\frac{
\sigma_i
}{
\sum_j\sigma_j
}.
\]

Effective rank:

\[
\boxed{
r_{\mathrm{eff}}
=
\exp
\left(
-\sum_i
\pi_i\log\pi_i
\right).
}
\]

This is more meaningful than checking only \(e_k\) cosine.

## 45.5 Matched-compute intervention controls

Mandatory:

- best single candidate;
- repeat candidate 1;
- repeat candidate 2;
- repeat candidate 3;
- repeat candidate 4;
- repeat best candidate;
- clone one candidate into all identities;
- mean candidate;
- random candidate;
- fixed action order;
- zero edits;
- matched-compute recurrent global editor.

Let:

\[
G_{\mathrm{full}}
=
M(q_{\mathrm{full}})
-
M(q_{\mathrm{ref}})
\]

and:

\[
\operatorname{Recovery}
=
\frac{
G_{\mathrm{control}}
}{
G_{\mathrm{full}}+\epsilon
}.
\]

If repeat/clone recovers approximately:

\[
\ge 90\%
\]

of the full modification gain on most samples used to support a multi-action claim, withdraw the multi-action specialization claim.

---

# 46. State-dependence diagnostics

Because \(e_k\) and \(p_k\) are intentionally stable, do not mistakenly diagnose “no dynamics” just because they remain unchanged.

Expected dynamics:

\[
g_{t,k},
d_{t,k},
h^{ctx}_{t,k},
\Delta Z_{t,k},
\delta q_{t,k},
s_{t,k}.
\]

Report:

- selected identity changes across steps;
- score-rank correlation across steps;
- score change after executing the same candidate;
- \(\delta q\) change across steps;
- \(g_{t,k}\) change;
- context change;
- realized target-evaluated selected-action gain;
- dynamic execution vs frozen-\(t=0\) action order.

The recurrent claim is unsupported if dynamic recomputation does not beat the frozen-order control.

---

# 47. Auxiliary-loss diagnostics

## 47.1 \(L_{\mathrm{comp}}\)

Monitor:

- JS loss;
- raw claim mass;
- claim entropy;
- peer overlap;
- inactive-slot behavior;
- single-edit samples.

Failure:

```text
normalized claims look distinct
but raw mass collapses
```

or:

```text
one-edit input is arbitrarily split into four token groups
```

## 47.2 \(L_{\mathrm{bind}}\)

Monitor:

- diagonal vs off-diagonal \(S_{k,j}\);
- cross-slot retrieval effect;
- permutation robustness;
- shared-context false negatives.

Failure:

```text
binding accuracy high
but deltaq candidates remain clones
```

## 47.3 \(L_{\mathrm{factor}}\)

Monitor:

- \(E^{all}\);
- relational distribution entropy;
- diagonal self-match probability;
- full-query/factor neighborhood transfer to held-out batches.

Failure:

```text
all factors clone full query
and L_factor is still low
```

is expected and is why \(L_{\mathrm{unique}}\) exists.

## 47.4 \(L_{\mathrm{unique}}\)

Monitor:

\[
\Delta_{i,k}^{uniq}.
\]

Also test:

- semantic probes;
- factor swaps;
- single-factor execution;
- repeat-factor execution;
- cross-sample portability.

Failure:

```text
all factors are individually uninterpretable code fragments
but all are necessary together
```

indicates secret sharing.

---

# 48. Kill / simplification criteria

The method should be simplified rather than protected for narrative value if robust experiments show:

## K1 — one-shot parity

A matched strong one-shot composer equals or beats IAG-SRME.

Then recurrence is not justified.

## K2 — no dynamic value

Frozen \(t=0\) action order equals dynamic recomputation.

Then state-dependent marginal execution is unsupported.

## K3 — global editor parity

A matched recurrent global editor equals local token editing.

Then local token intervention is unsupported.

## K4 — repeat collapse

One repeated candidate recovers essentially all full-model gain.

Then the multi-action claim fails.

## K5 — grounding irrelevance

Grounding shuffle/removal has little effect.

Then grounding is cosmetic.

## K6 — consequence scorer unnecessary

A context-only scorer matches the consequence-aware scorer.

Then the counterfactual-preview claim should be narrowed.

## K7 — token edit unnecessary

A global residual matches the local transition.

Then token-level execution is unnecessary.

## K8 — backbone adaptation claim unsupported

If full Base adaptation provides no reproducible value over the cheaper frozen-vision regime, do not claim it is necessary.

## K9 — auxiliary losses only make representations pretty

If \(L_{\mathrm{comp}},L_{\mathrm{bind}},L_{\mathrm{factor}},L_{\mathrm{unique}}\) improve internal diagnostics but not functional interventions/retrieval, do not claim solved decomposition.

## K10 — forced-factor pathology

If the auxiliary losses require four active factors on simple one-edit samples, disable/reformulate them with activity/NULL semantics.

---

# 49. Unit and smoke-test invariants

Before a full expensive run, verify:

1. **Same-parent preview**
   - all:
     \[
     \widehat Z_{t+1}^{(k)}
     \]
     branch from exactly the same \(Z_t\).

2. **Text-only intent**
   - `TextIntentEncoder.forward` has no image/state argument.

3. **No candidate-axis acquisition competition**
   - text attention normalization is over text tokens, not candidates.

4. **Anchor grounding**
   - \(P=f(E,A)\), not \(f(E,Y)\).

5. **Exact support locality**
   - if:
     \[
     p_{k,n}=0
     \]
     then:
     \[
     \Delta Z_{t,k,n}=0.
     \]

6. **Current-state dependence**
   - synthetic modification of \(Z_t\) changes:
     \[
     g_{t,k},h^{ctx}_{t,k},\Delta Z_{t,k},s_{t,k}.
     \]

7. **Stable intent/support**
   - if text and anchor are unchanged:
     \[
     e_k,p_k
     \]
     remain unchanged over the canonical trajectory.

8. **STOP identity**
   - STOP leaves \(Z\) and \(q\) unchanged.

9. **STOP absorbing**
   - after STOP, remaining unrolled calls cannot change state/query.

10. **Target permutation**
    - target shuffle does not alter model forward outputs before loss construction.

11. **Marginal detach**
    - teacher utility does not send target gradient into candidate construction through the utility arithmetic.

12. **Terminal reachability**
    - \(L_{\mathrm{terminal}}\) reaches the intended trainable encoder/editor parameters.

13. **Cache legality**
    - full vision fine-tuning rejects persistent stale image-feature caches.

14. **Claim content mask**
    - padding/special tokens cannot receive valid claim mass.

15. **Claim raw-mass logging**
    - normalized claim loss cannot hide zero-mass collapse.

16. **Relational self-match audit**
    - monitor \(p_{ii}\) and factor entropy.

17. **Correct leave-one-out denominator**
    - use \(K-1\), not a naive subtraction from an already averaged log product.

18. **Permutation robustness**
    - query indices carry no fixed human label.

19. **Matched compute**
    - FULL vs REPEAT/CLONE comparisons use equal recurrent depth/compute.

20. **No auxiliary bypass**
    - full-query anchor \(u_i\) used by factor losses is not consumed by the actual executor/readout path.

---

# 50. Computational complexity of the candidate factor losses

Given:

\[
F\in\mathbb R^{B\times K\times D}
\]

and anchor bank:

\[
U\in\mathbb R^{B\times D},
\]

factor-to-anchor similarities require:

\[
O(B^2KD).
\]

Full-anchor similarities require:

\[
O(B^2D).
\]

Main auxiliary memory:

\[
O(B^2K).
\]

Leave-one-out does **not** require \(K\) extra backbone forward passes.

It is computed algebraically from `log_pi`.

For moderate \(B,K=4\), this auxiliary cost is usually small relative to the VLM backbone.

---

# 51. Numerical-stability rules

## 51.1 Complementary claims

Use log space:

\[
\log r_{k,i}
=
\frac1{K-1}
\sum_{j\neq k}
\log(1-m^{claim}_{j,i}+\epsilon).
\]

## 51.2 Relational Product-of-Experts

Do not explicitly compute:

```python
pi.prod(dim=1)
```

Use:

```python
log_pi
mean / sum in log space
log_softmax
```

## 51.3 KL

Compute:

\[
D_{KL}(p\Vert q)
=
\sum_j
p_j
(
\log p_j-\log q_j
).
\]

Clamp only where mathematically necessary.

## 51.4 Mixed precision

Sensitive evaluator/log-product operations may use FP32 even if the backbone uses mixed precision.

---

# 52. Current status table

| Component | Status now |
|---|---|
| SRME recurrent idea | retained |
| four candidate hypotheses | retained |
| original state-conditioned proposal semantics | superseded by IAG text-only stable intent |
| immutable anchor \(A\) | canonical IAG |
| mutable token state \(Z_t\) | canonical IAG |
| text-only intent \(e_k\) | canonical IAG |
| anchor sparse grounding \(p_k\) | canonical IAG |
| current grounded evidence \(g_{t,k}\) | canonical IAG |
| Grounded Edit Context | canonical IAG |
| shared support-gated token editor | canonical IAG |
| four same-parent counterfactual states | canonical IAG |
| consequence-aware marginal score | canonical IAG |
| fixed-zero STOP | canonical IAG |
| hard-forward execution | canonical IAG |
| \(L_{\mathrm{terminal}}\) | canonical |
| \(L_{\mathrm{marginal}}\) | canonical |
| \(L_{\mathrm{comp}}\) | candidate auxiliary |
| \(L_{\mathrm{bind}}\) | candidate auxiliary |
| \(L_{\mathrm{factor}}\) | candidate auxiliary |
| \(L_{\mathrm{unique}}\) | candidate auxiliary |
| factor activity / NULL | unresolved integration requirement |
| all-six-loss objective | research super-objective, not current default |
| empirical benchmark success | not claimed by this specification |

---

# 53. Frozen decisions vs open decisions

## 53.1 Frozen architecture decisions for current IAG baseline

- \(K=4\);
- \(T_{\max}=3\);
- immutable anchor \(A\);
- mutable \(Z_t\);
- text-only intents;
- independent full-text read access;
- reference-anchored sparse grounding;
- current-state grounded evidence;
- explicit token-level residual intervention;
- same-parent candidate preview;
- consequence-aware target-free scorer;
- STOP score zero;
- hard-forward selection from update 1;
- target firewall;
- no module/horizon curriculum.

## 53.2 Open experimental decisions

- FG-CLIP Base full vs FG-CLIP Large frozen vision + trainable text;
- FG-CLIP vs future FG-CLIP2 ablation;
- exact internal width \(256\) vs controlled \(384\);
- exact \(\lambda_z\);
- \(T_{\max}=2\) vs 3 if extra depth proves unnecessary;
- static anchor grounding vs future dynamic/hybrid grounding;
- whether any auxiliary loss should be promoted;
- factor activity / NULL definition;
- self-masked vs diagonal-included relational geometry;
- stop-gradient vs joint auxiliary anchors;
- exact auxiliary weights/temperatures/margins.

---

# 54. One-sentence role of every current loss

\[
\boxed{
L_{\mathrm{terminal}}
:
\text{finish the CIR task correctly}
}
\]

\[
\boxed{
L_{\mathrm{marginal}}
:
\text{choose the edit whose predicted consequence is useful now}
}
\]

\[
\boxed{
L_{\mathrm{comp}}
:
\text{organize mutually bounded text-evidence responsibility}
}
\]

\[
\boxed{
L_{\mathrm{bind}}
:
\text{make each edit intent semantically match its own claimed evidence}
}
\]

\[
\boxed{
L_{\mathrm{factor}}
:
\text{make the WHAT+WHERE factor set collectively explain the full-query relational geometry}
}
\]

\[
\boxed{
L_{\mathrm{unique}}
:
\text{make every active factor measurably necessary under leave-one-out}
}
\]

---

# 55. Complete mental model

The entire current research object can be remembered as:

```text
REFERENCE IMAGE
    │
    ├── global identity anchor r_ref
    │
    └── immutable local tokens A
                    │
                    └── mutable state Z0 = A

MODIFICATION TEXT
    │
    ▼
text encoder
    │
    ▼
X
    │
    ├── q1
    ├── q2
    ├── q3
    └── q4
         │
         ▼
four stable text-only edit intents E
         │
         ├──────── optional semantic-claim head
         │                 │
         │                 ├── L_comp
         │                 └── claim pooling ↔ E → L_bind
         │
         ▼
anchor grounding on A
         │
         ▼
P = four sparse WHERE supports
         │
         ├──────── optional stable WHAT+WHERE factor f_k
         │                 │
         │                 ├── L_factor
         │                 └── L_unique
         │
         ▼
at timestep t:
read A, Z_t, Z_t-A under each support
         │
         ▼
g0, gt, dt
         │
         ▼
Grounded Edit Context
         │
         ▼
SHARED TOKEN EDITOR
         │
         ├── ΔZ_t,1 → Zhat_t+1^(1)
         ├── ΔZ_t,2 → Zhat_t+1^(2)
         ├── ΔZ_t,3 → Zhat_t+1^(3)
         └── ΔZ_t,4 → Zhat_t+1^(4)
                all branch from the same Z_t
         │
         ▼
read out four candidate retrieval queries
         │
         ▼
consequence-aware target-free scores
         │
         ├── candidate 1
         ├── candidate 2
         ├── candidate 3
         ├── candidate 4
         └── STOP = 0
         │
         ▼
hard execute exactly one / STOP
         │
         ▼
Z_t+1
         │
         └──────────────────────↺

final Z
    │
    ▼
final retrieval query
    │
    ▼
L_terminal

candidate consequences during training only
    │
    ▼
detached target evaluator
    │
    ▼
L_marginal
```

---

# 56. Final research interpretation

The current IAG-SRME thesis is:

> A CIR modification should first be represented as a small set of stable text-derived edit hypotheses. Each hypothesis independently grounds to the immutable reference image to obtain a spatial address, then reads the mutable current state at that address to instantiate a context-dependent local intervention. The model explicitly previews all candidate token-level interventions from the same current state, predicts their current marginal retrieval value without target input, executes exactly one candidate or STOP, and reevaluates the consequences on the new state. The official target is used only by the terminal retrieval objective and by a detached training-only evaluator of model-generated counterfactual consequences.

The four auxiliary losses extend this thesis at two different decomposition levels:

1. **\(L_{\mathrm{comp}}+L_{\mathrm{bind}}\)**  
   try to make the text-derived WHAT hypotheses assume meaningful, mutually bounded semantic responsibilities;

2. **\(L_{\mathrm{factor}}+L_{\mathrm{unique}}\)**  
   try to make the bound WHAT+WHERE factors collectively complete yet individually non-redundant in relational semantic geometry.

However:

\[
\boxed{
\text{the auxiliary losses are not yet evidence that factorization is solved.}
}
\]

The decisive evidence must still come from:

- grounding interventions;
- candidate-effect rank;
- drop/keep tests;
- repeat/clone controls;
- factor swaps;
- state-dependence tests;
- matched-compute comparisons;
- retrieval performance.

The most important unresolved integration problem is:

\[
\boxed{
K=4\text{ proposal identities}
\neq
4\text{ guaranteed true semantic edits}.
}
\]

Therefore any loss that implicitly requires all four factors to own evidence or be necessary must first gain a principled activity/NULL contract.

---

# 57. Immediate implementation target

The first trustworthy implementation should still be:

```text
FG-CLIP tokens
→ four text-only edit intents
→ anchor sparse grounding
→ current-state grounded evidence
→ Grounded Edit Context
→ local token counterfactual transition
→ candidate retrieval consequence
→ marginal score + STOP
→ hard execute
→ recurrent token state
→ final retrieval
```

with:

\[
\boxed{
L_{\mathrm{terminal}}
+
\lambda_mL_{\mathrm{marginal}}.
}
\]

Then promote auxiliary loss families only through controlled complete-run ablations.

Do not start by forcing:

> four beautiful different vectors.

First establish:

\[
\boxed{
\text{faithful}
+
\text{non-redundant}
+
\text{state-dependent}
+
\text{retrieval-useful}
}
\]

interventions.

---

# 58. Provenance ledger

This consolidation is derived from the following four research documents:

1. **TAPER-SRME: True End-to-End Canonical Method Specification V5**  
   Parent recurrent philosophy and target-as-evaluator design.

2. **TAPER-SRME — Intent-Anchored Grounded Token Editing — Canonical Research + Implementation Specification V1**  
   Current architecture precedence: text-only stable intent, anchor grounding, current-state contextualization, explicit token edit, preview, score, execute/STOP.

3. **CIR-TAPER — Mutual Complementary Claim + Contrastive Action-Claim Binding Loss Specification V1**  
   Mathematical source for \(L_{\mathrm{comp}}\) and \(L_{\mathrm{bind}}\).

4. **CIR TAPER — PAIR-Generalized Action–Region Factor & Unique Contribution Loss — Detailed Research / Implementation Specification V1**  
   Mathematical source for \(L_{\mathrm{factor}}\) and \(L_{\mathrm{unique}}\).

Where this unified document introduces an integration decision not literally frozen in one source, it is explicitly labeled as a **recommended IAG mapping** or **research super-objective**, not disguised as an already canonical fact.

---

# 59. Canonical checkpoint summary

| Field | Unified V1 record |
|---|---|
| method family | TAPER / SRME |
| current architecture | IAG-SRME |
| candidate count | \(K=4\) |
| recurrent horizon | \(T_{\max}=3\) |
| stable latent variables | text edit intent \(e_k\), anchor support \(p_k\) |
| dynamic variables | \(g_{t,k},d_{t,k},h^{ctx}_{t,k},\Delta Z,\delta q,s\) |
| state | immutable \(A\) + mutable \(Z_t\) |
| candidate generation | four same-parent local token interventions |
| decision | hard candidate or fixed-zero STOP |
| inference | target-free; one final gallery retrieval |
| canonical losses | \(L_{\mathrm{terminal}} + \lambda_mL_{\mathrm{marginal}}\) |
| candidate semantic losses | \(L_{\mathrm{comp}}+L_{\mathrm{bind}}\) |
| candidate factor losses | \(L_{\mathrm{factor}}+L_{\mathrm{unique}}\) |
| total current loss inventory | 6 |
| six-loss objective status | experimental synthesis, not default |
| largest integration risk | variable true factor count vs all-active auxiliary losses |
| largest scientific risk | latent differences without functional specialization |
| decisive anti-collapse audit | matched-compute REPEAT/CLONE + functional effect analysis |
| empirical status | unproven until implemented and benchmarked |

---

# 60. Final compact formula

The current architecture can be compressed to:

\[
\boxed{
e_k
=
\operatorname{TextIntent}(q_k,X)
}
\]

\[
\boxed{
p_k
=
\operatorname{entmax}
(
\operatorname{Ground}(e_k,A)
)
}
\]

\[
\boxed{
h^{ctx}_{t,k}
=
\operatorname{Fuse}
(
e_k,
p_k\!\cdot\!A,
p_k\!\cdot\!Z_t,
p_k\!\cdot\!(Z_t-A)
)
}
\]

\[
\boxed{
\Delta Z_{t,k}
=
p_k
\odot
\operatorname{SharedEditor}
(
Z_t,A,h^{ctx}_{t,k}
)
}
\]

\[
\boxed{
\widehat Z_{t+1}^{(k)}
=
Z_t+\Delta Z_{t,k}
}
\]

\[
\boxed{
\delta q_{t,k}
=
R(\widehat Z_{t+1}^{(k)})
-
R(Z_t)
}
\]

\[
\boxed{
s_{t,k}
=
\operatorname{Score}
(
h^{ctx}_{t,k},
\Delta Z_{t,k},
\delta q_{t,k}
)
}
\]

\[
\boxed{
k_t
=
\arg\max
\{
s_{t,1},\ldots,s_{t,4},0_{\mathrm{STOP}}
\}
}
\]

\[
\boxed{
Z_{t+1}
=
\begin{cases}
\widehat Z_{t+1}^{(k_t)}, & k_t\neq\mathrm{STOP}\\
Z_t, & k_t=\mathrm{STOP}.
\end{cases}
}
\]

Current canonical objective:

\[
\boxed{
L_{\mathrm{core}}
=
L_{\mathrm{terminal}}
+
\lambda_mL_{\mathrm{marginal}}.
}
\]

Complete candidate research objective:

\[
\boxed{
L_{\mathrm{six}}
=
L_{\mathrm{terminal}}
+
\lambda_mL_{\mathrm{marginal}}
+
\lambda_cL_{\mathrm{comp}}
+
\lambda_bL_{\mathrm{bind}}
+
\lambda_fL_{\mathrm{factor}}
+
\lambda_uL_{\mathrm{unique}}.
}
\]

The second equation becomes a legitimate final method objective only after the activity/NULL and anti-secret-sharing risks are explicitly handled and the auxiliary-loss ablations demonstrate real functional gain.

---

**End of unified master specification.**
