# TAPER-MAG: Canonical End-to-End Research and Training Specification V4

**Working name:** **TAPER-MAG** — *Edit-Conditioned Shared-Query Operators with Detached Marginal Action Gain*  
**Research starting point:** `CIR_TAPER_EDIT_CONDITIONED_SHARED_QUERY_COUNTERFACTUAL_UTILITY_FRONTIER_CHECKPOINT_V1_2026-08-28.md`  
**Companion milestones:** Phase-1 Checkpoint V2 and Phase-2 Novelty Gate V3  
**Canonical active method:** Phase-1 benchmark-optimized base  
**Selected Phase-2 evolution:** ASROA, inactive until the Phase-1 and novelty promotion gates pass  
**Empirical status:** implementation-ready specification; the new graph has not been coded, trained, or benchmarked in the supplied workspace

---

## 0. Executive decision

There is one method and one development path:

\[
\boxed{
\text{broad shared-query extraction}
\rightarrow
\text{edit-aware grounding}
\rightarrow
\text{candidate actions}
\rightarrow
\text{local state interventions}
\rightarrow
\text{detached marginal action gain}
\rightarrow
\text{dynamic execute/STOP}
}
\]

Phase 1 completes and optimizes this graph. Phase 2 may replace static use of an operator by one bounded state-residual adaptation mechanism; it does not create a parallel architecture.

The canonical choices are:

| Decision | Canonical value |
|---|---|
| backbone stratum | FG-CLIP2-large feature contract from V1; exact checkpoint/preprocess hash required |
| official training data | FashionIQ and CIRR triplets only |
| external generated supervision | none; no LLM/MLLM samples, captions, decompositions, labels, or trajectories |
| internal width | \(d=256\) |
| query/operator count | \(K=4\) |
| horizon | \(T_{\min}=1,\;T_{\max}=4\) |
| text reader | one non-competitive cross-attention block |
| edit conditioning | gated residual |
| visual grounding | one cross-attention block over local patch tokens |
| operator fusion | concatenation + product + difference |
| state | editable local tokens; global query is a deterministic change-aware readout |
| executor | shared factorized gated FiLM local editor |
| action critic | shared MLP over candidate-transition preview and target-free history |
| teacher | common-negative detached \(\Delta\)InfoNCE |
| STOP | fixed zero-value anchor; action values are net of a step cost |
| rollout | hard dynamic action, repeat allowed, STOP after at least one action |
| default loss | terminal retrieval + STOP-anchored listwise utility |
| training | one-run warm-up → online utility → DAgger-style roll-in → multi-step ST bridge → predicted hardening |
| inference | target-free, dynamically recomputed policy, one final gallery search |
| Phase-2 candidate | ASROA, zero-initialized and bounded; inactive before gates |

The method is not considered empirically complete until the Phase-1 gate in §16 passes. The document deliberately distinguishes observed numbers, planned targets, and untested hypotheses.

---

# Part I — Scientific object and contracts

## 1. Scientific hypothesis and falsifiers

### 1.1 Hypothesis

A small bank of non-competitive learned queries can read an instruction broadly, use that read to ground different reference-image regions, and form candidate edit operators without semantic labels. A shared recurrent local executor can make the same operator produce different effects at different states. A target-free critic can learn, from a detached training-time evaluator, which model-generated intervention yields the largest marginal retrieval gain at the current state and when no action is worth its cost.

The intended computation is:

\[
S_0\xrightarrow{o_{k_0}}S_1\xrightarrow{o_{k_1}}\cdots\xrightarrow{o_{k_{T-1}}}S_T,
\qquad q_T=P(S_T),
\]

where \(T\) is adaptive and can equal 1. Multi-step is a capability for hard examples, not a quota.

### 1.2 Decisive falsifiers

The core claims fail if any of the following persists outside control repeatability:

1. the full method does not match the strongest exact-same-backbone one-shot composer;
2. validation-only oracle action selection is no better than random/static/mean execution;
3. dynamic recomputation is equivalent to the ordering computed at \(t=0\) at fixed horizon and matched mean compute;
4. a parameter/FLOP-matched recurrent global editor matches the full method;
5. clone-all or repeated-best recovers essentially all full-model gain on the predeclared hard/multi-step subset;
6. the learned policy has attractive classification metrics but produces no retrieval improvement;
7. target permutation changes current policy logits before an optimizer update;
8. benchmark gain exists only after using prohibited generated data or a target-derived inference feature.

---

## 2. Data and target firewall

### 2.1 Allowed training information

Main-regime inputs are:

- official reference image;
- official modification text;
- official positive target image/ID;
- existing pretrained representation weights;
- in-batch and official-training gallery negatives;
- counterfactual candidate outcomes computed online by the current model.

The target may:

- participate in the standard terminal retrieval loss;
- evaluate candidate next queries under `no_grad`;
- define oracle-only validation analysis;
- remove known positives from a negative set.

The target may not:

- be concatenated into an operator, state, utility feature, history, STOP head, or inference input;
- generate a residual or operator;
- supervise each operator toward the full target residual;
- be converted into a persistent action label dataset;
- influence action selection at final inference.

No LLM/MLLM may generate samples, captions, modification text, semantic decomposition, pseudo targets, edit slots, operator labels, reasoning traces, VQA annotations, or an auxiliary pseudo-dataset. External models used only as published contextual references are reported in a separate resource stratum.

### 2.2 API-level separation

```python
@dataclass(frozen=True)
class PolicyBatch:
    ref_native: Tensor           # [B, 257, 1408]
    ref_native_mask: Tensor      # [B, 257]
    text_native: Tensor          # [B, M<=32, 768]
    text_attention_mask: Tensor  # [B, M]
    text_content_mask: Tensor    # [B, M]


@dataclass(frozen=True)
class SupervisionBatch:
    target_embedding: Tensor     # [B, 256], normalized
    target_id: Tensor            # [B]
    positive_ids: tuple[Tensor, ...]
```

`TaperInferenceModel.forward` accepts `PolicyBatch` only. `SupervisionBatch` is accepted by the training engine, terminal retrieval objective, negative miner, and detached teacher only. Target native tokens are not loaded unless an explicitly audited online target encoder requires them.

### 2.3 Hard firewall tests

1. Inference succeeds with no target tensor, target loader, target bank, or teacher allocated.
2. `teacher_scores.requires_grad is False`.
3. Shuffling/randomizing target embeddings while holding `PolicyBatch` and parameters fixed changes teacher scores but not current policy logits, action sequence, or final inference query.
4. `UtilityPolicy.forward`, `OperatorGenerator.forward`, and history schemas have a provenance allowlist and no target/teacher/oracle field.
5. After `L_util.backward()` in the closed regime, gradients are nonzero only in utility/STOP calibration parameters.
6. The exported inference `state_dict` contains no teacher, negative bank, target table, or sample-ID embedding.

A probe that predicts the target from reference+text is not a valid leak test: a legitimate CIR representation is supposed to predict the target. Dataflow and behavioral invariance are decisive.

---

## 3. Feature contract and tensor semantics

The implementation must first resolve the V1 cache manifest. A tensor shaped `[*,32,256]` must not be mean-pooled merely because its shape looks like a token sequence. The exact pretrained pooler, special-token policy, mapping from sample IDs to image IDs, normalization, preprocessing, and checkpoint hash must be recovered and unit-tested.

After resolution, the canonical tensor contract is:

| Tensor | Shape | Semantics |
|---|---:|---|
| \(V^{\rm native}\) | \([B,257,1408]\) | CLS plus 256 reference patch tokens |
| \(T^{\rm native}\) | \([B,M,768]\), \(M\le32\) | contextual modification tokens |
| \(C\) | \([B,M]\) | content mask excluding pad/special positions |
| \(r_{\rm ref}\) | \([B,256]\) | correctly pooled, normalized pretrained reference embedding |
| \(y^+\) | \([B,256]\) | correctly pooled, normalized target embedding; trainer/teacher only |
| \(X\) | \([B,M,d]\) | projected modification tokens |
| \(Z_0\) | \([B,N=256,d]\) | editable reference patch state |
| \(Q\) | \([K=4,d]\) | learned query identities |
| \(A,E,O\) | \([B,K,d]\) | text reads, grounded visual reads, operator anchors |
| \(Z_t\) | \([B,N,d]\) | persistent local state at step \(t\) |
| \(c_t,g_t,q_t\) | \([B,d]\) | state context, global readout, normalized retrieval query |
| \(\widehat Z_{t+1}\) | \([B,K,N,d]\) | transient one-step candidates from the same \(Z_t\) |
| \(\widehat q_{t+1}\) | \([B,K,d]\) | candidate retrieval queries |
| \(m_t\) | \([B,K,N]\) | soft local support |
| \(\widehat g_t\) | \([B,K]\) | predicted raw marginal gains |

Only input adapters change under a different backbone. Cached features imply a frozen encoder. A configuration validator must reject `feature_source=cached` with `encoder_tuning in {partial, full}`; real encoder adaptation requires online features or a regenerated/versioned cache.

---

# Part II — Exact Phase-1 architecture

## 4. Input projection

Discard the CLS token from the editable workspace:

\[
Z_0=\operatorname{LN}(V^{\rm native}[:,1:]W_V),
\qquad W_V\in\mathbb R^{1408\times d},
\]
\[
X=\operatorname{LN}(T^{\rm native}W_T),
\qquad W_T\in\mathbb R^{768\times d}.
\]

Use `content_mask` in every text cross-attention. An all-zero content mask is an error. No early query/operator filtering is allowed.

## 5. Edit-conditioned shared-query operator generation

### 5.1 Text read

Orthogonally initialize \(Q\in\mathbb R^{K\times d}\) without assigning color/object/pose semantics. With one pre-LN cross-attention block:

\[
C^T=\operatorname{MHA}(\operatorname{LN}Q_B,\operatorname{LN}X,\operatorname{LN}X;C),
\]
\[
A'=Q_B+C^T,
\qquad
A=\operatorname{LN}(A'+\operatorname{FFN}(\operatorname{LN}A')).
\]

Use eight heads, head width 32, FFN width \(4d\), and dropout 0.1. Attention is normalized independently over text positions:

\[
\sum_jP^T_{b,k,j}=1,
\]

with no constraint across \(k\). This is “non-competitive”: multiple queries may read the same useful word.

### 5.2 Gated edit conditioning

\[
G_k=\sigma(W_G[Q_k;A_k]+b_G),
\qquad b_G=-1,
\]
\[
\widetilde Q_k=\operatorname{LN}(Q_k+G_k\odot W_AA_k).
\]

The negative gate bias prevents immediate erasure of query identity while leaving a nonzero instruction path.

### 5.3 Edit-aware visual grounding

\[
C^V=\operatorname{MHA}(\widetilde Q,\operatorname{LN}Z_0,\operatorname{LN}Z_0),
\]
\[
E'=\widetilde Q+C^V,
\qquad
E=\operatorname{LN}(E'+\operatorname{FFN}(\operatorname{LN}E')).
\]

Only local patch tokens are keys/values in the base. Visual attention \(P^V\in\mathbb R^{B\times K\times N}\) is logged for diagnostics. A learned last-four-layer mixture, typed CLS KV, coarse-to-fine read, second block, and query self-attention are controlled capacity variants, not concurrent defaults.

### 5.4 Operator construction

\[
f_k=[A_k;E_k;A_k\odot E_k;A_k-E_k]\in\mathbb R^{4d},
\]
\[
o_k^0=\operatorname{LN}\left(Q_k+W_2\operatorname{GELU}(W_1\operatorname{LN}f_k)\right),
\]

where \(W_1:4d\rightarrow2d\), \(W_2:2d\rightarrow d\). Phase 1 computes \(O^0\) once per query and keeps it static over the rollout. There is no target feature, target residual, semantic label, or operator-specific executor parameter.

## 6. Coupled hybrid state and retrieval readout

The conceptual state is \(S_t=(Z_t,g_t)\), but only \(Z_t\) is recurrently writable. Both context and retrieval query are deterministic readouts of the local state.

### 6.1 Reference-conditioned context

\[
\pi_{t,n}=\operatorname{softmax}_n
\frac{(W_qr_{\rm ref})^\top(W_kZ_{t,n})}{\sqrt d},
\]
\[
c_t=\operatorname{LN}\left(W_rr_{\rm ref}+W_c\sum_n\pi_{t,n}W_vZ_{t,n}\right).
\]

### 6.2 Change-aware final pooling

Let \(D_t=Z_t-Z_0\). Then:

\[
s_{t,n}=w_p^\top\operatorname{SiLU}
(W_z\operatorname{LN}Z_{t,n}+W_d\operatorname{LN}D_{t,n}+W_cc_t),
\]
\[
\beta_{t,n}=\operatorname{softmax}_n(s_{t,n}),
\]
\[
g_t=r_{\rm ref}+W_{\rm out}\sum_n\beta_{t,n}D_{t,n},
\qquad q_t=\operatorname{Normalize}(g_t).
\]

Thus \(q_0=r_{\rm ref}\). At least one operator must execute, so modification text reaches the final query through grounded local editing. The canonical base has no direct text-to-query or separate one-shot-composer bypass. A strong one-shot composer remains the principal comparator; a residual base-composer bypass is a diagnostic rescue control and cannot enter the method unless the editor adds reproducible gain and passes operator-zero/no-local controls.

This design prevents an unrestricted global vector from carrying the entire edit while local tokens become decorative.

## 7. Shared factorized gated FiLM executor

The executor must be strong enough to model nonlinear local edits but structured enough that operator effects remain inspectable. The selected sweet spot shares the expensive token transform across all candidates and uses operator-conditioned support and FiLM thereafter.

### 7.1 Shared state transform

\[
H_{t,n}=\operatorname{SiLU}(W_H\operatorname{LN}Z_{t,n}+W_Cc_t),
\qquad
P_{t,n}=W_PH_{t,n}.
\]

This \(O(BNd^2)\) transform is computed once per state, not once per candidate.

### 7.2 Dynamic local support

\[
\ell_{t,k,n}=
\frac{(W_O^mo_k^0)^\top(W_Z^mH_{t,n})}{\sqrt d}+b_m,
\qquad
m_{t,k,n}=\sigma(\ell_{t,k,n}/\tau_m),
\]

with \(b_m=\operatorname{logit}(0.2)\) and \(\tau_m=1\) initially. There is no hard top-\(k\) mask or support-budget loss by default. The initial grounding map may be logged as a prior diagnostic, but it is not hard-wired into support.

### 7.3 Operator-conditioned bounded delta

Generate FiLM and direction parameters:

\[
[\gamma_k,\beta_k,v_k,\rho_k]=W_Fo_k^0,
\]

where the first three terms are \(d\)-vectors and \(\rho_k\) is scalar. Define:

\[
R_{t,k,n}=\tanh\left([1+0.1\tanh(\gamma_k)]\odot P_{t,n}+\beta_k\right)\odot\tanh(v_k),
\]
\[
\Delta Z_{t,k,n}
=m_{t,k,n}\,\sigma(\rho_k)\,[\lambda_\Delta\odot R_{t,k,n}],
\qquad \lambda_\Delta^{(0)}=0.1,
\]
\[
\widehat Z_{t+1}^{(k)}=Z_t+\Delta Z_{t,k}.
\]

The last operator projection is small/zero-biased so initial deltas are nonzero but bounded. Persistent state is not post-normalized: pre-LN and LayerScale stabilize the transition without changing untouched tokens. If support or delta is zero, the state is exactly unchanged.

The executor has:

- one parameter set shared across \(k\) and \(t\);
- no unrestricted `GRU(g_t,o_k)` or global edit MLP;
- no per-operator expert;
- no target input or residual;
- no BatchNorm or dropout that would make preview and selected recomputation disagree.

The main alternatives are a simple local MLP, cross-attention writer, and one lightweight Transformer executor. They are tested one at a time only after the oracle candidate gain identifies under-capacity.

## 8. Candidate enumeration and action-value predictor

### 8.1 Counterfactual semantics

At each state, enumerate all actions from the same immutable parent:

\[
\widehat S_{t+1}^{(k)}=F_\theta(S_t,o_k^0),
\qquad k=1,\ldots,K.
\]

Candidate \(k+1\) never consumes candidate \(k\). The same readout yields \(\widehat g_{t+1}^{(k)}\), \(\widehat q_{t+1}^{(k)}\), and:

\[
\delta g_{t,k}=\widehat g_{t+1}^{(k)}-g_t.
\]

Support-weighted local context is:

\[
\bar m_{t,k,n}=\frac{m_{t,k,n}}{\sum_jm_{t,k,j}+\epsilon},
\qquad
v_{t,k}=\sum_n\bar m_{t,k,n}Z_{t,n}.
\]

### 8.2 Target-free history

For each operator, history contains:

\[
h_{t,k}=[
\text{use count}/T_{\max},
\mathbb1(k=k_{t-1}),
\text{steps since use}/T_{\max},
\widehat g_{t-1,k},
\|\delta g_{\rm last,k}\|_2,
\text{last support mass},
\text{support overlap},
t/T_{\max}
].
\]

Unavailable fields are zero with an explicit validity bit if needed. `Previous realized utility` is forbidden when it means target-derived teacher gain. Only a previous predicted score or observable state/support statistic is allowed.

### 8.3 Shared candidate-preview critic

Let \(\bar A=K^{-1}\sum_kA_k\) and \(\widetilde h_{t,k}=\operatorname{MLP}_h(h_{t,k})\in\mathbb R^d\). Define:

\[
x_{t,k}=[
g_t;o_k^0;A_k;E_k;\bar A;v_{t,k};\delta g_{t,k};
o_k^0\odot v_{t,k};A_k\odot E_k;\widetilde h_{t,k}
]\in\mathbb R^{10d}.
\]

\[
\widehat g_{t,k}=w_u^\top\operatorname{GELU}
\left(W_{u2}\operatorname{GELU}(W_{u1}\operatorname{LN}x_{t,k})\right),
\]

with \(10d\rightarrow2d\rightarrow d\rightarrow1\). The same MLP scores every action. No cross-action Transformer is used initially. Candidate-transition preview is canonical because action value should reflect what the action would do, not only what its operator vector looks like.

At inference this requires a vectorized \(K\)-candidate preview per live step. The factorized executor makes this practical; it is measured explicitly against the recurrent global and fixed-horizon controls.

## 9. Detached marginal-action teacher

### 9.1 Common-negative local retrieval metric

Mine one negative set \(\mathcal H_t\) using detached \(q_t\):

- valid in-batch official targets;
- optionally a FIFO/EMA bank from official training target embeddings;
- the top \(H=64\) hard negatives under \(q_t\) for the initial implementation;
- no positive, duplicate target ID, or known group positive.

The same \(\mathcal H_t\) evaluates the current query and all candidates. Remine only after the real state advances.

\[
\ell_t(q)=-\log
\frac{\exp(s(q,y^+)/\tau_r)}
{\exp(s(q,y^+)/\tau_r)+\sum_{y^-\in\mathcal H_t}\exp(s(q,y^-)/\tau_r)}.
\]

Raw marginal retrieval gain:

\[
g^*_{t,k}=\operatorname{sg}\left[\ell_t(q_t)-\ell_t(\widehat q_{t+1}^{(k)})\right].
\]

Net value and STOP:

\[
u^*_{t,k}=g^*_{t,k}-c_{\rm step},
\qquad
u^*_{t,\mathrm{STOP}}=0.
\]

Teacher computation uses FP32 similarity/logsumexp and `torch.no_grad()`. It does not use target-similarity improvement, rank delta, residual alignment, staleness, or drift in the default score. Those are compact controlled alternatives only if common-negative \(\Delta\)InfoNCE fails to predict full-gallery progress.

### 9.2 Why net gain is required

If every action has a small positive raw gain, a zero STOP cannot win. Subtracting a step cost makes the decision coherent:

\[
\max_kg^*_{t,k}\le c_{\rm step}
\Longrightarrow
\mathrm{STOP}.
\]

Train the quality-maximizing model first with \(c_{\rm step}=0\). At validation, shift all action logits by a scalar cost or equivalently add a STOP bias to trace a quality/mean-step Pareto. The deployment point is the Pareto knee or a predeclared compute budget; it is never tuned on test.

## 10. Utility loss, calibration, and STOP

STOP is an action with a fixed zero logit, not a free default head:

\[
p_t^*=\operatorname{softmax}([u^*_{t,1},\ldots,u^*_{t,K},0]/\tau_u),
\]
\[
\widehat p_t=\operatorname{softmax}([\widehat g_{t,1}-c,\ldots,\widehat g_{t,K}-c,0]/\tau_p),
\]
\[
L_{\rm util}=D_{\rm KL}(p_t^*\Vert\widehat p_t).
\]

Anchoring STOP at zero teaches the sign of useful gain. Pure pairwise ranking is an ablation because it cannot calibrate the stopping threshold. A small Huber gain-calibration term is permitted only if pair ordering is strong but false-STOP/false-continue remains the measured bottleneck; it is not in the default objective.

Rules:

- mask STOP at \(t=0\), so every sample executes at least one edit;
- allow STOP for \(t\ge1\);
- terminate at \(T_{\max}=4\) even if STOP never wins;
- allow the same operator at adjacent or non-adjacent steps;
- recompute candidate transitions and utility after every executed action;
- use exact argmax in final inference.

A repeat is valid when its newly computed net teacher value is positive. It is stale when non-positive. No hard repeat mask or anti-repeat penalty is used.

## 11. Multi-step rollout

For every live sample at step \(t\):

1. compute \(c_t,g_t,q_t\);
2. enumerate all \(K\) candidate transitions from \(Z_t\);
3. pool every candidate and form target-free action features;
4. compute \(\widehat g_{t,1:K}\) and append STOP=0;
5. mask STOP if \(t=0\);
6. choose \(a_t=\arg\max([\widehat g_{t,1:K}-c,0])\);
7. STOP or gather the chosen candidate state;
8. update target-free history and repeat.

Samples that stop become identity transitions in a vectorized fixed-length training loop. The final gallery is searched once with \(q_T\); retrieve–inspect–correct feedback is absent from Phase 1.

---

# Part III — Objective and end-to-end optimization

## 12. Terminal retrieval objective

For normalized final queries \(q_i\) and target embeddings \(y_j\), use the exact loss and positive masking of the strongest same-backbone control. The canonical starting form is multi-positive bidirectional InfoNCE:

\[
L_{q\rightarrow i}=-\frac1B\sum_i
\log\frac{\sum_{j\in\mathcal P_i}\exp(s(q_i,y_j)/\tau)}
{\sum_j\exp(s(q_i,y_j)/\tau)},
\]
\[
L_{i\rightarrow q}=-\frac1B\sum_j
\log\frac{\sum_{i:j\in\mathcal P_i}\exp(s(q_i,y_j)/\tau)}
{\sum_i\exp(s(q_i,y_j)/\tau)},
\]
\[
L_{\rm ret}^{\rm terminal}=\tfrac12(L_{q\rightarrow i}+L_{i\rightarrow q}).
\]

Use only the terminal state reached by the actual rollout. Default intermediate full-target supervision is rejected because it encourages every early action to solve the whole triplet and turns multi-step into “finish immediately.”

## 13. Canonical loss and gradient routing

\[
\boxed{L=L_{\rm ret}^{\rm terminal}+\lambda_uL_{\rm util}},
\qquad \lambda_u=1.
\]

Default auxiliary weights are zero:

\[
\lambda_{\rm rollout}=\lambda_{\rm preservation}=\lambda_{\rm query}=0.
\]

| Loss/path | Backbone/adapters | query/operator generator | executor/readout | utility policy | target evaluator |
|---|---:|---:|---:|---:|---:|
| terminal retrieval, executed trajectory | according to stage | yes | yes | no initially; controlled ST later | standard target branch only |
| STOP-anchored utility KL | no | no | no | yes | no gradient |
| teacher scoring | no | no | no | no | no gradient |

For `L_util`, utility inputs are detached:

\[
x^{\rm util}_{t,k}=\operatorname{sg}(x_{t,k}).
\]

This is still integrated end-to-end training: the actor determines candidate states and labels, the critic determines future rollout states, and terminal retrieval updates the actually executed actor path. End-to-end does not require unsafe target gradients through every component.

### 13.1 Controlled policy gradient opening

Define:

```python
def grad_scale(x, rho):
    return x.detach() + rho * (x - x.detach())
```

- \(\rho_{\rm gate}\) controls `L_ret → utility weights` through hard-forward/soft-backward selection;
- \(\rho_{\rm up}\) controls `L_ret → utility input → state/operator`.

Canonical progression is \(\rho_{\rm gate}:0\rightarrow0.25\) only after critic warm-up and \(\rho_{\rm up}=0\) throughout the main Phase-1 run. A separate late experiment may test \(\rho_{\rm up}\in\{0.05,0.1\}\), first through state/global candidate summaries while keeping \(A,E,O\) detached.

Opening is retained only if paired benchmark and action regret improve beyond repeatability while clone/repeat/locality, target firewall, gradient conflict, seed variance, and functional rank remain non-inferior. `L_util → target branch` is permanently impossible. If opening fails, the closed graph is canonical; a firewall is not a lack of integration.

### 13.2 Auxiliary-loss activation rules

- **Query orthogonality:** add at \(10^{-4}\) only when off-diagonal query cosine is persistently high *and* response effective rank collapses.
- **Preservation:** add only after measured off-support delta; detach the support mask so the mask cannot open to evade the loss.
- **Weak monotonicity:** test \(\lambda\le0.1\) only if early-step gradients vanish; reject if it increases one-step/repeat pathology.
- **No uniform-use loss:** unequal use may be correct.
- **No direct target residual:** allowed only as a negative collapse control.

## 14. Hard-forward / soft-backward bridge

During the middle curriculum:

\[
p_t=\operatorname{softmax}([\widehat g_{t,1:K}-c,0]/\tau_s),
\]
\[
y_t^{\rm hard}=\operatorname{onehot}(\arg\max p_t),
\]
\[
y_t^{\rm ST}=p_t+\operatorname{sg}(y_t^{\rm hard}-p_t).
\]

Forward semantics execute one discrete action. Backward uses the soft surrogate. No Gumbel noise is canonical because oracle/policy roll-in and soft-teacher sampling already provide exploration. The final hardening stage removes ST so training ends on the inference distribution.

When ST is active, candidate transitions may retain live graphs. Otherwise, enumerate detached candidates, choose an action, and recompute only the selected transition with gradients. This two-pass implementation avoids \(O(TBKND)\) live activation memory in the hard stages.

## 15. End-to-end training schedule

Use one model, optimizer state, EMA, and checkpoint stream. Warm-up changes routing and horizon; it does not produce frozen modules for later assembly.

For a concrete 60-epoch reference budget, scaled so total updates are approximately 1.5× the fair one-shot baseline:

| Epoch | Horizon | State distribution / route | Trainable behavior |
|---|---:|---|---|
| 0 | — | evaluator smoke tests | reproduce M3 one-shot; validate cache, metrics, firewall, hand rankings |
| 1–8 | 1 | small bounded uniform mixture of all candidate deltas | actor/readout warm-up with terminal retrieval; backbone frozen; utility inactive |
| 9–14 | 1 | soft mixture; online candidates | utility warm-up with detached STOP-anchored labels; actor continues slowly |
| 15–26 | 2 | DAgger-style oracle/policy roll-in, \(\beta:0.8\to0.3\) | hard one-action forward; online labels at every visited state; STOP after step 1 |
| 27–40 | 3 | \(\beta:0.3\to0\) | hard forward/ST backward; \(\tau_s:1.0\to0.5\); \(\rho_{\rm gate}:0\to0.25\) |
| 41–46 | 4 | predicted route with 5% confident top-2 exploration | full dynamic reselection; backbone remains frozen |
| 47–52 | 4 | predicted route; exploration \(0.05\to0\) | adapter/last-block low-LR stratum; \(\tau_s:0.5\to0.25\); optional upstream-opening run is separate |
| 53–60 | 4 | deterministic predicted argmax | hardening; no oracle route, exploration, or ST retrieval gradient; online utility labels remain |

At a visited state, the greedy training oracle chooses only among current model-generated candidates and STOP. Early oracle roll-in may sample from the soft distribution of confident positive-gain actions instead of always taking argmax; this reduces early-winner feedback without forcing uniform usage.

The procedure is “DAgger-style” because it relabels states induced by the current policy online, addressing exposure bias, but the oracle is a greedy marginal evaluator rather than a globally optimal demonstrator. This follows the distribution-shift principle of [DAgger](https://proceedings.mlr.press/v15/ross11a.html) without claiming literal expert imitation.

No long-lived replay dataset of action labels is created. If a fixed probe set shows teacher rank churn beyond run repeatability, use short actor/critic blocks with a refreshed EMA snapshot; do not create permanently separate trained systems.

### 15.1 Stage health gates

Do not advance the curriculum automatically when a gate fails.

**After actor warm-up:**

- soft one-step Recall is within roughly one point of the fair M3 control or improving toward it;
- all query identities receive rolling gradients;
- candidate outcome variance exceeds numerical noise;
- oracle best–second gaps and positive-gain rates are nondegenerate.

**After utility warm-up:**

- agreement exceeds random by a clear margin;
- confident-pair accuracy exceeds 50%;
- regret falls;
- oracle action improves retrieval over random/static execution.

**Before ST opening:**

- oracle–learned gap is shrinking;
- state and delta norms are stable;
- STOP is neither always selected nor never selected after calibration;
- stale repeats lose value after recomputation;
- dynamic ordering changes on a meaningful fraction of hard examples.

**Before any upstream opening:**

- target firewall passes;
- oracle advantage is largely captured by the learned policy;
- clone-all/best-repeat and response-rank controls are healthy;
- utility is the measured bottleneck rather than the candidate action space;
- the closed regime is stable across seeds.

## 16. Backbone and encoder adaptation

### 16.1 Same-backbone stratum first

All mechanism claims first use the exact FG-CLIP2 checkpoint, layer/token extraction, resolution, preprocessing, global pooler, normalization, gallery encoder, data split, caption policy, effective negatives, and optimization budget. The run manifest records hashes, not only a family name. Even the same nominal ViT can change materially with library and preprocessing versions; the [official ENCODER repository](https://github.com/iLearn-Lab/AAAI25-ENCODER) explicitly warns about `open_clip` version sensitivity.

### 16.2 Adaptation ladder

After a stable frozen-backbone canonical run, compare:

1. frozen encoder;
2. lightweight adapters or LoRA;
3. partial unfreeze of final image/text blocks at low LR;
4. full encoder at very low LR.

Selection is based on benchmark, frozen source-to-target transfer, representation drift, functional health, and seed stability—not a preference for freezing. Cached features can support only row 1. Rows 2–4 require online encoding or versioned cache regeneration. If the target/gallery encoder changes, rebuild the gallery bank at least each epoch or use an EMA target encoder with a versioned bank.

After same-backbone causality is established, a second resource stratum may compare FG-CLIP2, BLIP-2/PE-Core/SigLIP-like representations, or the strongest fair available backbone. A backbone switch is reported separately and never credited as an architectural gain. Existing pretrained representations are allowed; task-specific pseudo-data generation is not.

## 17. Optimizer and numerical configuration

Canonical starting point:

| Parameter group | Initial LR |
|---|---:|
| query reader, grounding, operators, executor, readout | \(2\times10^{-4}\) |
| utility/history projection | \(3\times10^{-4}\) |
| lightweight adapters | \(5\times10^{-5}\) |
| final backbone blocks | \(5\times10^{-6}\) to \(10^{-5}\) |
| learned retrieval temperature | \(10^{-5}\) |

- AdamW, \(\beta=(0.9,0.98)\), \(\epsilon=10^{-8}\);
- weight decay 0.05 on matrix weights, none on norm, bias, query bank, or LayerScale;
- linear warm-up over 5% of updates; cosine decay to 10% of initial LR;
- effective batch 256 when feasible, e.g. microbatch 64 × accumulation 4; record the resolved physical batch;
- distributed all-gather targets when available;
- default hard-negative \(H=64\), with a 16,384-entry bank as the first scalable configuration; ablate \(H=128\) only after profiling;
- global gradient clip 1.0 and per-module gradient norms;
- BF16; FP32 teacher similarity, hard-negative selection, logsumexp, and subtraction of nearby marginal scores;
- EMA 0.999 from utility warm-up onward for validation/bank stability;
- one screening seed, at least three promotion seeds, and five seeds for final decisive same-backbone rows when resources permit.

If more than 20% of updates clip a module, lower that parameter-group LR rather than silently increasing the clip. Log state norm, delta norm, support mass/saturation, AMP scaler, and teacher margin by step.

Small sequential search, not a Cartesian grid:

1. actor LR \(\{10^{-4},2\times10^{-4}\}\), utility LR \(\{10^{-4},3\times10^{-4}\}\);
2. \(\lambda_u\in\{0.5,1\}\), ST end temperature \(\{0.25,0.5\}\), \(\rho_{\rm gate}\in\{0.1,0.25\}\);
3. \(T_{\max}\in\{3,4\}\) and validation step-cost sweep;
4. frozen versus adapters/partial unfreeze;
5. only then width 256 versus 384/512 or executor capacity.

Each round fixes the largest measured bottleneck before opening another axis.

---

# Part IV — Benchmark and diagnostic standard

## 18. Dataset protocols

### 18.1 FashionIQ

FashionIQ contains human-written relative captions across Dresses, Shirts, and Tops&Tees. Use the standard category galleries and report all per-category R@10/R@50 values, macro R@10/R@50, and:

\[
C_{\rm FIQ}=\frac{R@10_{\rm macro}+R@50_{\rm macro}}2.
\]

Evaluation concatenates the two official captions deterministically and uses normalized cosine similarity. Train-time caption-order augmentation, if used by the exact fair baseline, is recorded and kept identical across ablations. State clearly whether one model is shared across categories or category-specific. Protocol references: [FashionIQ paper](https://arxiv.org/abs/1905.12794), [official dataset repository](https://github.com/XiaoxiaoGuo/fashion-iq), and the auditable [CLIP4Cir validation code](https://github.com/ABaldrati/CLIP4Cir/blob/master/src/validate.py).

### 18.2 CIRR

CIRR evaluates global retrieval and a subset of visually similar group members. Report global R@1/5/10/50, subset R@1/2/3, and:

\[
C_{\rm CIRR}=\frac{R@5+R_{\rm subset}@1}{2}.
\]

Remove the reference image from global ranking and construct the subset from official group membership. Tune on validation; submit only a frozen configuration to the official test server and never select hyperparameters from server feedback. References: [CIRR project](https://www.zheyuanliu.me/CIRR/), [ICCV paper](https://openaccess.thecvf.com/content/ICCV2021/html/Liu_Image_Retrieval_on_Real-Life_Images_With_Pre-Trained_Vision-and-Language_Models_ICCV_2021_paper.html), and [test-server specification](https://github.com/Cuberick-Orion/CIRR/blob/main/Test-split_server.md).

### 18.3 Development discipline

- build a target-ID/group-aware dev split from official FashionIQ train for frequent tuning;
- expose official FashionIQ validation only at meaningful milestones/final selection;
- tune CIRR on validation and reserve test for a frozen recipe;
- never place an internal-validation TAPER number beside a published test number as though they share a protocol;
- unit-test evaluators using hand-computed rankings before model experiments.

## 19. Fairness and resource strata

| Stratum | Supervision/resource | Headline status |
|---|---|---|
| S0 | official triplets and standard target contrastive loss | fair baseline |
| S1 | S0 plus target as detached evaluator of model-generated actions | **TAPER main** |
| S2 | external non-LLM paired/synthetic data or auxiliary segmenter/caption resource | contextual only |
| S3 | LLM/MLLM-generated captions, triplets, labels, VQA, reasoning, or inference generator | excluded from main; contextual only |

Every run manifest records backbone/checkpoint hash, preprocessing, source layer/tokens, local/global dimensions, target/gallery encoder, tuning regime, precision, split version, caption policy, gallery scope, normalization, effective negatives, auxiliary models, target usage, parameters, MACs/FLOPs, GPU-hours, VRAM, query latency, index latency, and average/p95 steps.

## 20. Exact-same-backbone experiment matrix

| ID | Model/control | Scientific question |
|---|---|---|
| M0 | reference-only | image shortcut |
| M1 | text-only | text shortcut |
| M2 | normalized sum + tuned scalar gate | simple composition floor |
| M3 | strongest exact-same-backbone one-shot MLP/FiLM/Combiner | principal benchmark control |
| M4 | TAPER generator/executor, coupled hybrid, \(T=1\) | architecture value before multi-step |
| M5 | parameter/FLOP-matched recurrent global editor at \(T=2,4\) | mechanism versus extra recurrent compute |
| M6 | shared-query actions with mean/uniform/all execution | learned utility necessity |
| M7 | learned utility, one step | action selection without recurrence |
| M8 | multi-step with ordering frozen from \(t=0\) | static-order control |
| M9 | recomputed dynamic utility, fixed \(T\), STOP off | state-dependence isolated |
| M10 | dynamic utility + STOP | canonical Phase-1 system |
| M11 | detached target-teacher oracle rollout | validation-only action-space upper bound |

All rows use identical seeds, backbone contract, optimizer-update budget, negatives, target/gallery embeddings, checkpoint rule, and evaluation. M11 is never presented as a deployable policy.

The causal run order is M0–M3 → M4/M5 → offline/shadow M11 → M7/M8/M9 → M10. If M11 has no headroom, stop critic research and fix representation/state/action/executor. If M5 matches M10, there is no evidence that candidate operators—rather than recurrent compute—caused the gain.

## 21. Published landscape: context, not causal evidence

The following audited rows are navigation anchors across different backbones/resources; they do not replace M3. Values are percentages and must be rechecked against the cited primary source when the final paper table is built.

| Method | Resource stratum | FashionIQ R10 / R50 / composite | CIRR global R1/5/10/50 | CIRR subset R1/2/3 | CIRR composite |
|---|---|---:|---:|---:|---:|
| CLIP4Cir, ViT-B/32 | official triplets | 38.40 / 61.74 / 50.07 | 38.53 / 69.98 / 81.86 / 95.93 | 68.19 / 85.64 / 94.17 | 69.09 |
| ENCODER, ViT-B/32 | official triplets | 56.13 / 77.59 / 66.86 | 46.10 / 77.98 / 87.16 / 97.64 | 76.92 / 90.41 / 95.95 | 77.45 |
| COMBINER-B | official triplets | 57.26 / 78.37 / 67.82 | 47.21 / 79.11 / 88.87 / 97.98 | 77.01 / 90.48 / 96.24 | 78.06 |
| SPRC, BLIP-2/ViT-G | official triplets | 54.72 / 74.97 / 64.85 | 51.96 / 82.12 / 89.74 / 97.69 | 80.65 / 92.31 / 96.60 | 81.39 |
| COMBINER-H, ViT-H/14 | official triplets | 63.26 / 82.27 / 72.77 | 52.60 / 82.51 / 90.12 / 98.17 | 81.33 / 93.06 / 97.23 | 81.92 |
| OFFSET, ViT-H/14 + BLIP-2 + CLIPSeg | external auxiliary | 62.59 / 82.18 / 72.39 | 52.19 / 82.60 / 90.07 / 98.07 | 81.37 / 93.08 / 97.54 | 81.99 |

Primary references: [CLIP4Cir](https://github.com/ABaldrati/CLIP4Cir), [ENCODER](https://ojs.aaai.org/index.php/AAAI/article/view/32541), [COMBINER](https://arxiv.org/abs/2606.04604), [SPRC](https://openreview.net/forum?id=m3ch3kJL7q), and [OFFSET](https://arxiv.org/abs/2507.05631). OFFSET is external-auxiliary in this project because of its captioning/segmentation resources. FlowCIR, MagicLens, MLLM-caption systems, and LLM rerankers belong in separate contextual tables because their data/inference contracts differ or violate the main-regime constraint.

## 22. Expected targets and statistical rules

The only existing internal observations from V1 are diagnostic single runs: `qasa_full` R@10/50 46.73/70.39 (composite 58.56) and `all_slots_full` 46.90/70.92 (58.91), with functional rank about 1.023. They have no seed error bar and are not a same-backbone frontier.

Let \(B_{\rm FIQ}\) and \(B_{\rm CIRR}\) be the M3 composites after M3 is actually optimized.

| Level | Same-backbone target |
|---|---|
| mandatory non-inferiority | M10 ≥ \(B-0.25\) point and no category/major metric down >1 point, with severe collapse reduced |
| competitive | FashionIQ \(B+1\) to \(B+2\); CIRR \(B+0.8\) to \(B+1.5\) |
| stretch | FashionIQ \(B+2\) to \(B+3\); CIRR \(B+1.5\) to \(B+2.5\) |

For planning only, a strong-backbone published-test target is roughly FashionIQ composite 72–73 and CIRR composite 82–83; stretch is 74–75 and at least 83.5. These are not reported TAPER results and cannot be claimed across a mismatched resource stratum.

Statistical policy:

- one seed is screen-only;
- promotion uses at least three fixed seeds;
- final decisive M3/M4/M5/M8/M10 rows target five fixed seeds;
- report mean ± SD, never best seed;
- paired per-query bootstrap with 10,000 resamples and 95% CI; stratify FashionIQ by category and preserve CIRR grouping;
- claim superiority only when the lower CI clears the repeatability band;
- allow a capability result with aggregate within −0.3 point only if a predeclared hard-composition subset improves at least about 2 points with CI excluding zero and compute/health remain acceptable.

## 23. Hard/capability and transfer evaluation

Priority:

1. official CIRR subset R@1/2/3;
2. deterministic unimodal-resistant subset: target missed by both M0 and M1 at the same cutoff;
3. frozen-encoder edit-magnitude quartiles \(1-\cos(r,y^+)\);
4. bottom-quartile target-versus-nearest-negative margin;
5. long-caption and transparent conjunction/negation token slices;
6. post-hoc oracle-multistep-benefit slice, labeled as model-dependent analysis;
7. released evaluation-only composition-required mappings such as [CIRCUS](https://arxiv.org/abs/2605.14787) and single-image [PinPoint](https://arxiv.org/abs/2603.04598), when protocol-compatible;
8. optional FashionIQ-C/CIRR-C corruption evaluation.

No LLM labels are used to build a hard subset. Generalization reports:

- FashionIQ→CIRR and CIRR→FashionIQ frozen transfer where input contracts allow;
- frozen utility predictor transfer versus actor+critic transfer;
- pretrained-to-finetuned representation drift on reference and gallery images;
- action regret and STOP behavior under transfer;
- source performance retained after target adaptation.

## 24. Utility/action-value metrics

At every evaluated on-policy state:

\[
\operatorname{Regret}_t=\max_{a\in\{1:K,STOP\}}u^*_{t,a}-u^*_{t,\widehat a_t}\ge0.
\]

Report:

- Top-1 agreement;
- pairwise accuracy on pairs outside an empirical near-tie uncertainty band;
- mean, median, p90/p95 regret;
- regret conditioned on oracle non-STOP;
- false-STOP and false-continue rates;
- policy-created realized \(\Delta\)InfoNCE and full-gallery rank/Recall change;
- oracle-versus-learned retrieval gap at matched compute;
- calibration by predicted-gain bins;
- metrics on states visited by the learned policy, not only oracle states.

High agreement without positive realized retrieval is a critic/teacher interface failure, not success.

## 25. Functional diagnostics

### 25.1 Response rank

For candidate query deltas \(R_t=[\delta q_{t,1};\ldots;\delta q_{t,K}]\in\mathbb R^{K\times d}\) with singular values \(\sigma_i\), report participation rank:

\[
r_{\rm eff}=\frac{(\sum_i\sigma_i)^2}{\sum_i\sigma_i^2+\epsilon},
\]

plus raw singular spectra and paired controls. A low rank on an easy one-edit sample is not automatically failure; systematic near-one rank on the claimed hard/multi-step subset plus clone/repeat evidence is severe.

### 25.2 Required interventions

- repeat-best and mean-repeat;
- drop-one;
- clone-all with best and mean operator;
- freeze utility ordering from \(t=0\);
- remove edit conditioning;
- global-only, local-only, coupled hybrid;
- operator-zero and operator-mean bypass;
- dynamic versus static execution;
- learned versus oracle action frequencies;
- support mass/overlap and delta cosine;
- no-utility mean/uniform/fixed-order controls;
- one-step versus adaptive and fixed \(T=2,3,4\).

For repeat controls, use recovered gain rather than raw recall ratio:

\[
\rho_{\rm repeat}=\frac{M_{\rm repeat}-M_{\rm reference}}
{M_{\rm full}-M_{\rm reference}},
\]

with denominator guards, paired difference, and confidence interval.

### 25.3 Dynamic versus frozen protocol

Run both:

1. fixed \(T\), STOP off, to isolate state-dependent ordering;
2. matched mean compute, with each policy’s STOP bias calibrated on validation.

Dynamic execution is claimed only if paired retrieval/rank improvement clears control repeatability and stale-repeat rate falls. Otherwise simplify the claim and permit an adaptive one-step system.

## 26. Realistic risk register and minimal safeguards

| Risk | Decisive evidence | Minimal safeguard |
|---|---|---|
| operator clone | clone-all, drop-one, response rank | orthogonal query init; edit-aware grounding; no per-op target loss |
| giant reusable edit | best-single/repeat recovered gain, delta norm | bounded local residual; no target residual; no free global writer |
| static utility | dynamic/frozen and ranking-change rate | candidate-preview critic recomputed every step |
| local-state bypass | remove-local, zero-delta, direct-global control | global query displacement derives only from \(Z_t-Z_0\) |
| target leakage | target shuffle and graph provenance | hard API/dataflow firewall |
| execution monopoly | learned versus oracle usage and retrieval | soft confident teacher exploration; no forced uniform use |
| stale repeat | newly evaluated net gain and STOP regret | net gain minus cost; repeat legal; STOP=0 |
| encoder over-globalization | attention overlap, transfer drift | frozen start; adapter/partial ladder; benchmark+health selection |
| moving-label instability | teacher rank churn, gradients, seeds | online relabeling, FP32 teacher, EMA bank, optional short alternating blocks |
| benchmark regression | M3 and M5 | retrieval-first checkpointing; functionality as hard constraint, not substitute objective |

Do not redesign for exotic failures without evidence. Add them to the register and continue the benchmark-first loop.

## 27. Phase-1 success gate

Phase 1 passes only when all are true:

1. exact architecture, tensor/API, loss, gradient, training, inference, and checkpoint contracts are implemented;
2. the integrated curriculum trains without NaNs, exploding states, chronic clipping, or seed-specific hacks;
3. M10 is benchmark-strong: at minimum same-backbone non-inferior, with a clear path toward the fair contextual frontier;
4. M11 shows action-space headroom and the learned policy captures meaningful oracle advantage;
5. M9 beats M8 on the claimed multi-step/hard regime at fixed and matched compute;
6. M10 is not explained by M5 extra recurrent compute;
7. clone/repeat/rank/local-bypass tests show no severe multi-evidence collapse;
8. target firewall passes absolutely;
9. encoder adaptation retains acceptable transfer and representation geometry;
10. results survive at least three seeds without a large loss cocktail.

**Current status:** not passed. The supplied workspace contains the V1 research document but no repository, dataset cache, manifests, evaluator, or training logs, so no honest new benchmark can be produced in this research turn.

---

# Part V — Phase 2, novelty, and contribution boundary

## 28. Closest prior art

The novelty audit uses primary papers and assigns ownership narrowly:

| Work | Owned component | Exact remaining TAPER boundary |
|---|---|---|
| [SDQUR](https://ieeexplore.ieee.org/document/10530361/) | multiple learnable queries/diverse composed embeddings | no executed interventions or state-dependent action value |
| [ENCODER](https://ojs.aaai.org/index.php/AAAI/article/view/32541) | modality-shared relation queries/entity–action binding | queries are one-shot composition, not recurrent actions |
| [OFFSET](https://arxiv.org/abs/2507.05631) | text-guided focus and local processing before pooling | no model-generated action set or marginal policy |
| [SSN](https://ojs.aaai.org/index.php/AAAI/article/view/28479) | fixed semantic-shift stages | no adaptive execute/recompute/STOP |
| [SEIZE](https://dl.acm.org/doi/10.1145/3664647.3681649) | “semantic editing increment” | different regime; no grounded learned local rollout |
| [CompoDiff](https://openreview.net/forum?id=mKtlzW0bWc) | iterative latent diffusion | continuous denoising and synthetic data, not discrete interventions |
| [FlowCIR](https://arxiv.org/abs/2607.02284) | conditional semantic transport | no small candidate set, marginal value, or adaptive STOP |
| [Interactive Retrieval 2018](https://proceedings.neurips.cc/paper_files/paper/2018/file/a01a0380ca3c61428c26a231f0e49a09-Paper.pdf) | sequential rank reward with user feedback | TAPER uses internal actions under one instruction |
| [FashionNTM](https://arxiv.org/abs/2308.10170) | state across user turns | no single-turn internal action decomposition |
| [MAI](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2f8b56543953d60f262fb2c4b85c50b3-Abstract-Conference.html) | multi-turn history/iteration | new user text and generated-data regime |
| [CoCo-IR](https://arxiv.org/abs/2608.05149) | evolving embeddings in contextual multi-turn CIR | external history/LMM data, not internal operator interventions |
| [CAST](https://openaccess.thecvf.com/content/CVPR2026/html/Lin_CAST_Context-Aware_Dynamic_Latent_Space_Transformation_for_Interactive_Text-to-Image_Retrieval_CVPR_2026_paper.html) | context-aware low-rank global matching-space transform | ASROA adapts individual grounded anchors from internal local residuals |
| [Duplex Rewards](https://ojs.aaai.org/index.php/AAAI/article/view/38369) | counterfactual text masking/reward pools/test-time RL | different intervention object; not leave-one-operator outcomes |
| [TreeQN/ATreeC](https://openreview.net/forum?id=H1dh6Ax0Z) | latent transitions/lookahead/value | no CIR-specific operator acquisition or target firewall |
| [PonderNet](https://arxiv.org/abs/2107.05407) | learned halting | TAPER STOP is supporting net-gain calibration, not a novel halting claim |

[TG-CIR](https://arxiv.org/abs/2309.01366) also blocks a generic “target-aware teacher, target-free student” claim. TAPER’s narrower distinction is that the target judges one-step interventions generated by the current actor, online at its visited states.

## 29. Exact Phase-1 contribution boundary

If Phase 1 passes its gate, defensible claims are:

1. a single CIR instruction is internalized as a small sequence of candidate interventions rather than new user turns;
2. non-competitive shared queries are used to acquire and ground candidate actions, not claimed as newly invented;
3. target-as-evaluator ranks model-generated state interventions while the inference policy remains target-free;
4. state-dependent marginal gain is recomputed after local transitions and co-designed with STOP;
5. a coupled hybrid state prevents a freely writable global bypass and improves the benchmark/functionality trade-off, if ablations support it.

Do not claim invention of shared queries, text-guided focus, local tokens, latent editing, target-teacher learning, action values, rank reward, multi-turn retrieval, adaptive computation, or STOP individually.

Honest pre-result novelty score for Phase 1 is **5.5–6.0/10**.

## 30. Phase-2 candidate audit

| Candidate | Novelty potential | Risk | Decision |
|---|---:|---:|---|
| state-conditioned operator adaptation | 6.0–6.5 if necessity proved | medium | select bounded anchored ASROA |
| counterfactual utility as a new principle | 5.0–5.5 alone | low | already Phase-1 core |
| local controllability/Jacobian objective | about 6.5 | high | defer; likely auxiliary-loss complexity |
| relevance vs residual-value decomposition | 4.5–5.0 | low | diagnostic only |
| unconstrained dynamic operator generation | 5.5–6.0 | high | reject |
| retrieve–inspect–correct feedback | 6.5–7.0 | high | defer; protocol and prior-art expansion |

No second novelty mechanism is stacked to chase a score. If the selected mechanism succeeds, stop adding novelty. If it fails, retain the strong Phase-1 method and diagnose the actual ceiling.

## 31. ASROA: Anchored State-Residual Operator Adaptation

ASROA is inactive until §27 passes and a static-operator ceiling is measured.

For Phase-1 anchor \(o_k^0\), cache:

\[
c_{0,k}=\operatorname{Attn}(o_k^0,Z_0).
\]

At step \(t\):

\[
c_{t,k}=\operatorname{Attn}(o_k^0,Z_t),
\qquad d_{t,k}=c_{t,k}-c_{0,k},
\]
\[
u_{t,k}=\tanh(W_u[d_{t,k};c_{t,k};\widetilde h_{t,k}])\in\mathbb R^r,
\qquad r=32,
\]
\[
\delta o_{t,k}=B(u_{t,k}\odot Ao_k^0),
\quad A\in\mathbb R^{r\times d},\;B\in\mathbb R^{d\times r},
\]
\[
o_{t,k}=\operatorname{LN}\left(o_k^0+
\operatorname{ClipNorm}(\delta o_{t,k},\rho\|o_k^0\|_2)\right),
\qquad \rho=0.10.
\]

Initialize \(B=0\), so training starts exactly at the locked Phase-1 function. The bound is fixed in the main experiment; repeated saturation is a rejection signal. Inputs are only anchor, local state, and target-free history. Terminal retrieval may update ASROA on selected paths; utility supervision remains critic-only initially; the target evaluator is detached forever.

The precise new mechanism, if supported, is:

> a stable grounded semantic action anchor receives a bounded low-rank correction from its operator-specific change in local state before its outcome is valued and selected.

It is not unrestricted operator regeneration and not a global dialog-conditioned matching-space transform.

### 31.1 Decisive controls

| ID | Model |
|---|---|
| N0 | locked Phase-1 static operators |
| N1 | N0 + ASROA |
| N2 | parameter/FLOP-matched wider static operator/executor |
| N3 | state-agnostic adapter using only \(t=0\) reads |
| N4 | ASROA computed at \(t=0\) then frozen |
| N5 | ASROA without anchored residual \(c_t-c_0\) |
| N6 | unbounded dynamic adapter, failure diagnostic only |

### 31.2 Promotion gate

Promote N1 as the single final TAPER-MAG formulation only if:

- aggregate paired improvement exceeds multi-seed repeatability and bootstrap CI excludes zero;
- a practical target is roughly +0.5 point on FashionIQ composite and CIRR R@1, or a predeclared hard-subset gain with aggregate non-inferiority;
- N1 beats N2, N3, and N4, demonstrating more than parameters and proving state-time necessity;
- no headline aggregate drops materially, with −0.5 point as a planning guard;
- rank, clone/repeat, dynamic/frozen, locality, action monopoly, firewall, and seed variance are non-inferior;
- correction norms do not collapse to zero or saturate the bound;
- measured latency/FLOPs/VRAM remain within about 10–15% unless Pareto gain clearly justifies more.

After this gate, novelty is honestly **6.0–6.5/10**. Before it, ASROA is only a preregistered hypothesis and cannot raise the method’s score.

## 32. Retrieval-feedback decision

The possible extension

\[
E_t=\operatorname{Aggregate}(\operatorname{TopK}(q_t)),
\qquad
\widehat u_{t,k}=U(S_t,o_k,E_t,H_t)
\]

is deferred. It changes latency and protocol, creates gallery-dependent/transductive risks, enters dense interactive-retrieval literature, and can hide a weak internal state. Reconsider it only if Phase 1 passes, ASROA fails for a measured observability rather than capacity ceiling, and a state-only oracle/predictor gap persists. It is never automatically stacked on ASROA.

---

# Part VI — Complete ablation and optimization program

## 33. Ablation matrix

### 33.1 Extraction and grounding

1. one versus two text cross-attention layers;
2. no query self-interaction versus one small cross-query block;
3. residual, gated residual, FiLM edit conditioning;
4. remove edit conditioning: \(\widetilde Q=Q\);
5. local final-layer tokens, learned last-four-layer mix, typed CLS+local KV, coarse-to-fine;
6. no visual grounding: replace \(E_k\) with pooled reference;
7. operator concat only versus concat+product+difference versus compact fusion block;
8. \(K\in\{2,4,6\}\), only after the \(K=4\) critic/action space is healthy.

### 33.2 State/readout/executor

1. global-only, local-only, coupled hybrid;
2. free global update as a negative bypass control;
3. reference anchor + change-aware pooling versus mean and standard attention pooling;
4. direct one-shot residual anchor as diagnostic control;
5. local MLP, FiLM, cross-attention writer, canonical factorized gated FiLM, one lightweight Transformer;
6. bounded versus unbounded residual;
7. support bias/temperature and support-loss-off default;
8. width \(d=256\), then 384/512 only after capacity evidence.

### 33.3 Teacher, utility, and gradient

1. common-negative \(\Delta\)InfoNCE;
2. target-hard-negative margin, target-similarity delta, soft-rank surrogate, full rank delta;
3. compact two-term combination only if one metric demonstrably misses full-gallery progress;
4. STOP-anchored listwise KL versus pairwise ranking and gain regression;
5. candidate-preview critic versus operator-only critic;
6. shared MLP versus one cross-action block;
7. closed gradient, ST gate \(\rho_{\rm gate}\), tiny upstream opening;
8. current online labels versus stale/snapshot labels as instability control;
9. direct per-operator target residual as a negative collapse control only.

### 33.4 Rollout and STOP

1. \(T=1\), fixed \(T=2,3,4\), adaptive \(T\);
2. random, uniform/mean, fixed relevance, frozen \(t=0\), dynamic learned, dynamic oracle;
3. STOP=0 threshold, learned STOP head, calibrated cost, patience, fixed horizon;
4. repeat allowed versus anti-repeat only as a diagnostic—not a candidate default;
5. oracle-only training, teacher-forced rollout, DAgger-style mixture, predicted-only curriculum;
6. soft mixture, deterministic ST, Gumbel ST ablation, final hardening.

### 33.5 Backbone/generalization

1. frozen, adapters, partial, full low-LR;
2. same-backbone only before a stronger backbone stratum;
3. source-to-target actor transfer;
4. frozen critic transfer and full actor+critic transfer;
5. representation drift versus retrieval gain.

### 33.6 Phase 2

N0–N6 in §31, with no retrieval feedback in the same experiment.

## 34. Benchmark-first bottleneck loop

After each meaningful end-to-end run:

1. record benchmark, confidence interval, compute, and health;
2. identify the largest measured bottleneck;
3. change exactly one architectural/optimization axis;
4. rerun the necessary control;
5. promote or revert.

| Observation | Next permitted focus |
|---|---|
| M3 and M11 low | representation, readout, executor/action capacity |
| M11 strong, M10 weak | critic inputs, calibration, on-policy coverage |
| critic metrics good, policy retrieval weak | teacher/action interface or candidate transition quality |
| M9 ≈ M8 | no state-dependent claim; simplify or fix state/action dynamics |
| local state raises rank but lowers Recall | pooling/fusion/optimization |
| repeat/clone ≈ full | decomposition/executor, not more utility complexity |
| M10≈M5 | extra compute explanation; no operator-mechanism claim |
| train gain + val drift | encoder LR/adapters/freeze |
| exact-same-backbone healthy but far from context frontier | only then scale backbone |

Checkpoint selection is lexicographic:

1. reject leak or numerical failure;
2. reject severe multi-evidence collapse for a multi-operator claim;
3. choose highest validation retrieval among survivors;
4. if within repeatability, prefer lower regret, positive dynamic/frozen gap, fewer steps, and healthier repeat/clone behavior.

Keep `best_retrieval_valid`, `best_policy_regret`, `best_functional_health`, and `last`; report the highest-retrieval checkpoint that passes hard health gates.

---

# Part VII — Implementation blueprint

## 35. Module boundaries

```text
src/taper_mag/
  contracts.py
  feature_contract.py
  projector.py
  operators.py
  state.py
  executor.py
  readout.py
  utility.py
  controller.py
  teacher.py
  negatives.py
  losses.py
  trainer.py
  evaluator_fiq.py
  evaluator_cirr.py
  diagnostics.py
  checkpointing.py
  configs/
tests/
  test_feature_contract.py
  test_noncompetitive_queries.py
  test_executor_equivalence.py
  test_teacher_negatives.py
  test_gradient_firewall.py
  test_dynamic_rollout.py
  test_evaluators.py
```

```python
class EditConditionedOperatorGenerator(nn.Module):
    def forward(self, projected: ProjectedInputs) -> OperatorSet: ...


class LocalHybridExecutor(nn.Module):
    def encode_state(self, state: HybridState) -> StateFeatures: ...
    def enumerate(self, state, features, operators) -> CandidateBatch: ...
    def apply_selected(self, state, features, operator, execute_mask): ...


class RetrievalReadout(nn.Module):
    def forward(self, state: HybridState) -> Tensor: ...
    def forward_candidates(self, candidates: CandidateBatch) -> Tensor: ...


class UtilityPolicy(nn.Module):
    def forward(self, state, candidates, operators, history) -> Tensor:
        """Return raw predicted gains [B,K]; STOP is appended by controller."""


class CounterfactualTeacher:
    @torch.no_grad()
    def score(self, current_query, candidate_queries,
              supervision, negatives, step_cost) -> Tensor: ...


class TaperInferenceModel(nn.Module):
    def forward(self, policy_batch: PolicyBatch,
                step_cost: float = 0.0) -> InferenceOutput: ...
```

Teacher and negative miner are not children of `TaperInferenceModel` and are absent from its exported state.

## 36. State and trace contracts

```python
@dataclass(frozen=True)
class HybridState:
    local: Tensor          # [B,N,d]
    initial_local: Tensor  # [B,N,d]
    ref_query: Tensor      # [B,d]
    active: Tensor         # bool [B]


@dataclass(frozen=True)
class OperatorSet:
    text_parts: Tensor     # [B,K,d]
    visual_parts: Tensor   # [B,K,d]
    operators: Tensor      # [B,K,d]
    text_attn: Tensor      # [B,K,M]
    visual_attn: Tensor    # [B,K,N]


@dataclass(frozen=True)
class HistoryState:
    use_count: Tensor          # [B,K]
    last_used_step: Tensor     # [B,K]
    last_predicted_gain: Tensor
    last_delta_norm: Tensor
    last_support_mass: Tensor
    last_support: Tensor       # optional [B,K,N], or compressed overlap stats
    previous_action: Tensor    # [B]


@dataclass(frozen=True)
class CandidateBatch:
    local: Tensor          # transient [B,K,N,d]
    query: Tensor          # [B,K,d]
    support: Tensor        # [B,K,N]
    delta_norm: Tensor     # [B,K]
    global_delta: Tensor   # [B,K,d]


@dataclass(frozen=True)
class RolloutTrace:
    action: Tensor         # [B,T], K denotes STOP
    active: Tensor         # [B,T]
    raw_gain: Tensor       # [B,T,K]
    query_delta_norm: Tensor
    support_mass: Tensor
```

Do not retain full candidate local states in long-term traces. Store sampled/scalar diagnostics.

## 37. Training pseudocode

```python
for batch in loader:
    policy_batch, supervision = split_batch(batch)
    projected = projector(policy_batch)
    operators = operator_generator(projected)
    state = state_initializer(projected)
    history = init_history(B, K)
    util_terms = []

    for t in range(horizon_for_epoch):
        current_q = readout(state)
        features = executor.encode_state(state)

        # Live enumeration only in ST stages; otherwise detached two-pass.
        candidates = executor.enumerate(
            state, features, operators.operators,
            live_graph=curriculum.use_st,
        )
        candidate_q = readout.forward_candidates(candidates, state)

        with torch.no_grad():
            negatives = negative_miner.mine_once(
                current_q.detach(), supervision
            )
            teacher_values = teacher.score(
                current_q.detach(), candidate_q.detach(),
                supervision, negatives, curriculum.step_cost,
            )  # [B,K+1], STOP=0

        utility_inputs = build_target_free_features(
            state, candidates, operators, history
        )
        predicted_gain = utility(utility_inputs.detach())
        predicted_values = append_stop(
            predicted_gain - curriculum.step_cost, stop=0.0
        )
        util_terms.append(stop_anchored_kl(
            predicted_values, teacher_values.detach()
        ))

        action = controller.choose(
            predicted_values=predicted_values,
            teacher_values=teacher_values,
            oracle_mix=curriculum.beta,
            use_st=curriculum.use_st,
            stop_allowed=(t > 0),
        )

        state = execute_or_gather_selected(
            state, candidates, features, operators,
            action, recompute_selected=not curriculum.use_st,
        )
        history = update_target_free_history(
            history, action, predicted_gain, state, candidates
        )

    final_q = readout(state)
    loss_ret = terminal_retrieval_loss(final_q, supervision)
    loss = loss_ret + lambda_u * mean(util_terms)
    optimize_with_amp_clip_ema(loss)
```

All counterfactual labels are created online from the current batch and discarded. No semantic or trajectory dataset is written.

## 38. Inference algorithm

```python
@torch.no_grad()
def infer(reference_image, modification_text, step_cost=0.0):
    policy_batch = encode_without_target(reference_image, modification_text)
    projected = projector(policy_batch)
    operators = operator_generator(projected)
    state = state_initializer(projected)
    history = init_history(batch_size(state), K=4)

    for t in range(4):
        current_q = readout(state)
        features = executor.encode_state(state)
        candidates = executor.enumerate(
            state, features, operators.operators, live_graph=False
        )
        candidate_q = readout.forward_candidates(candidates, state)
        x = build_target_free_features(
            state, candidates, operators, history
        )
        action_values = utility(x) - step_cost
        logits = append_stop(action_values, stop=0.0)
        if t == 0:
            logits[:, STOP] = -inf
        action = logits.argmax(dim=-1)
        if all(action == STOP):
            break
        state = gather_candidate_or_keep(state, candidates, action)
        history = update_target_free_history(
            history, action, action_values, state, candidates
        )

    final_query = readout(state)
    return final_query, optional_trace
```

The target is not an argument. Intermediate gallery retrieval is absent. Search the gallery once after the final query.

## 39. Configuration skeleton

```yaml
schema_version: 1

data:
  dataset: fashioniq              # or cirr
  supervision_stratum: S1
  feature_contract: fgclip2_large_v1
  cache_manifest_hash: REQUIRED
  caption_eval_order: deterministic
  no_generated_data: true

model:
  d_model: 256
  num_queries: 4
  text_read_layers: 1
  visual_read_layers: 1
  query_self_layers: 0
  edit_conditioning: gated_residual
  operator_fusion: concat_product_difference
  state: coupled_hybrid_local
  executor: factorized_gated_film
  executor_layerscale_init: 0.10
  support_bias_init: -1.386294
  direct_global_update: false
  direct_text_query_bypass: false
  phase2_asroa_active: false

policy:
  max_steps: 4
  min_steps: 1
  allow_repeat: true
  critic: candidate_preview_mlp
  stop_anchor: 0.0
  step_cost: 0.0
  selection_inference: argmax

teacher:
  metric: common_negative_delta_infonce
  hard_negatives: 64
  same_set_across_candidates: true
  remine_each_step: true
  target_branch_detached: true

loss:
  terminal_retrieval: bidirectional_multipos_infonce
  utility: stop_anchored_listwise_kl
  lambda_utility: 1.0
  lambda_rollout: 0.0
  lambda_preservation: 0.0
  lambda_query: 0.0

optimization:
  optimizer: adamw
  actor_lr: 0.0002
  utility_lr: 0.0003
  adapter_lr: 0.00005
  backbone_lr: 0.000005
  weight_decay: 0.05
  betas: [0.9, 0.98]
  warmup_fraction: 0.05
  schedule: cosine
  precision: bf16
  effective_batch_size: 256
  gradient_clip: 1.0
  ema_decay: 0.999

runtime:
  counterfactual_two_pass: true
  candidate_chunk_size: 4
  deterministic_audit: true
  save_sampled_policy_traces: true
```

## 40. Unit and smoke tests

### Feature/evaluator

- manifest, sample-to-image mapping, special tokens, and pooler reproduce stored global embeddings;
- FashionIQ and CIRR hand rankings produce exact expected metrics;
- no implicit mean pool over unknown retrieval sequences;
- cached mode rejects encoder unfreeze.

### Non-competitive queries

- every valid query attention sums to one over tokens;
- no sum-to-one assertion over queries;
- one token can receive high attention from multiple queries;
- query permutation permutes \(A,E,O\) and action logits, with STOP invariant;
- rolling gradient coverage reaches every query.

### Executor/candidates

- vectorized enumeration matches a Python loop over \(k\);
- candidate order does not change values;
- all candidates use the same immutable parent;
- selected recomputation matches the corresponding preview in eval mode;
- zero support/delta is exact identity;
- inactive/STOP samples remain unchanged;
- repeated action is allowed;
- no in-place alias across candidates.

### Teacher/firewall

- teacher scores detached;
- target/known positives absent from negatives;
- identical negative IDs across all actions at a state;
- STOP ranks first when every net gain is non-positive;
- target shuffle changes only teacher/loss;
- `L_util` gradient reaches only utility in the closed regime;
- exported inference runs without target resources.

### Rollout

- per-sample alive mask handles different STOP times;
- synthetic state flips action order and proves dynamic recomputation;
- frozen control actually reuses \(t=0\) values;
- min/max horizon correct;
- checkpoint resume reproduces action/loss/negative mining under audit mode.

### Small overfit smoke

Overfit 16–32 official triplets only to verify decreasing terminal loss, improving utility regret, finite tensors, nonzero query gradients, and functional instrumentation. Do not treat smoke-test specialization as evidence.

## 41. Compute and memory plan

Let batch \(B\), patches \(N=256\), operators \(K=4\), width \(d=256\), horizon \(T\).

Raw BF16 storage scales as:

\[
\text{state}=O(BNd),
\quad
\text{candidate transient}=O(BKNd),
\quad
\text{selected live rollout}=O(TBNd).
\]

Naively retaining all candidate graphs is \(O(TBKNd)\); detached enumeration plus selected recomputation reduces live memory to approximately \(O(BKNd+TBNd)\).

At \(B=32\):

| Raw tensor | Approximate BF16 memory |
|---|---:|
| native reference `[32,257,1408]` | 22.1 MiB |
| projected local state `[32,256,256]` | 4 MiB |
| all candidates `[32,4,256,256]` | 16 MiB |
| five state boundaries for \(T=4\) | 20 MiB |
| negatives `[32,64,256]` | 1 MiB |

Actual peak is higher because of QKV, MLP activations, gradients, optimizer state, and kernels. Profile rather than extrapolate a hardware promise. Use memory-mapped caches, pinned asynchronous batch transfer, `expand` rather than `repeat`, sampled diagnostic tensors, candidate chunking, and activation checkpointing of the selected recurrent path if required.

The factorized executor scales approximately as:

\[
O\left(T[B N d^2+B K d^2+B K N d+B K H d]\right),
\]

where the expensive token \(d^2\) transform is shared across candidates. The new modules are expected to be on the order of 5–8M parameters at \(d=256\); report exact parameter/MAC counts from the implementation.

Required profiler output:

- total/trainable parameters and MACs per preview/selected action;
- peak allocated/reserved VRAM;
- train milliseconds/step, samples/s, candidate-actions/s;
- extractor, executor, teacher, gallery mining, backward, and loader wall-time fractions;
- inference query p50/p95, ANN/index time separately;
- average/p95 steps and STOP histogram;
- one-time backbone feature extraction and index-build cost separately.

Compare quality/compute against M3 and M5. Adaptive computation is valuable only if its Pareto is better than a fixed-horizon or global recurrent control.

## 42. Reproducibility and checkpoint state

Save:

- model, optimizer groups, schedulers, AMP scaler, EMA;
- epoch/global/micro step and curriculum stage;
- current horizon, oracle mix \(\beta\), exploration, ST temperature, gradient scales, step cost;
- Python/NumPy/Torch CPU/all-CUDA RNG states and stateful sampler position;
- negative miner/index version and gallery bank hash;
- feature manifest, split, sample-index, config-schema, git commit, and dirty-diff hashes;
- best metric records and early-stop state;
- environment: PyTorch/CUDA/driver/GPU/precision/kernel mode.

Audit mode sets deterministic algorithms, deterministic caption sampling by `(seed, epoch, sample_id)`, fixed gallery/negative miner, and evaluation ordering. Speed runs using TF32, Flash kernels, compile, or nondeterministic algorithms are labeled separately.

Run artifacts:

```text
run/
  config_resolved.yaml
  environment.json
  feature_manifest.json
  split_manifest.json
  metrics_train.jsonl
  metrics_val.jsonl
  policy_trace_sampled.jsonl
  intervention_report.json
  firewall_report.json
  compute_report.json
  best_retrieval_valid.ckpt
  best_policy_regret.ckpt
  best_functional_health.ckpt
  last.ckpt
```

Inference export contains projector, operator generator, state/readout, executor, utility/history, and calibrated cost only.

## 43. Implementation roadmap

Every milestone remains a runnable end-to-end retrieval model; milestone tags are causal checkpoints, not independently trained modules.

| Milestone | Deliverable | Exit criterion |
|---|---|---|
| M0 contract/evaluator | cache resolver, data schemas, exact FIQ/CIRR metrics, firewall | hand-ranking and cache-pooler tests pass |
| M1 exact controls | M0–M3 baselines, run/checkpoint infrastructure | strong exact-same-backbone M3 established |
| M2 actor graph | queries, grounding, operators, coupled state, executor, \(T=1\) | M4 stable; candidate variance and oracle headroom logged |
| M3 teacher shadow | common-negative online evaluator without controlling policy | M11 beats random/static or action space is revised |
| M4 critic/STOP | STOP-anchored utility and one-step action selection | regret and realized gain improve on policy states |
| M5 curriculum | \(T=2\rightarrow3\rightarrow4\), dynamic/frozen controls | M8–M10 completed; repeat/clone/local gates pass |
| M6 benchmark optimization | adaptation ladder and one-bottleneck loop | Phase-1 gate across seeds; tag `taper-mag-phase1-locked` |
| M7 novelty | N0–N6 from the locked tag | ASROA promoted or rejected; no mechanism stacking |
| M8 final report | fair backbone strata, transfer, hard subsets, compute | paper table and exact contribution boundary frozen |

Meaningful research checkpoints record phase, architecture, benchmark status, functional status, major risk, novelty score, decisions, and next action. They never overwrite earlier checkpoint files.

## 44. Final canonical status and next action

### Active canonical method now

TAPER-MAG Phase-1 base:

> non-competitive edit-conditioned shared queries generate static candidate operator anchors; a shared bounded factorized local executor edits a coupled hybrid state; a target-free candidate-preview critic learns STOP-anchored marginal action gain from a detached common-negative retrieval evaluator; dynamic hard execution repeats or stops adaptively; the full system is warmed up and then co-adapted on policy-visited states with terminal retrieval and controlled gradient routing.

### Selected final novelty mechanism

ASROA is the selected mechanism for the same method, but it is inactive and cannot be called part of the empirical final model until its gate passes. If it passes, it replaces static operator use inside TAPER-MAG; there is still one canonical method. If it fails, static Phase-1 TAPER-MAG remains canonical and the novelty score stays at its honest level.

### Immediate build order

1. obtain the repository/cache manifests and reproduce the exact global pooler;
2. implement/evaluate M0–M3 and establish the fair same-backbone baseline;
3. implement M4 and run teacher shadow M11 before training the critic;
4. proceed through M7–M10 only when health gates pass;
5. run 3–5 seeds, paired bootstrap, transfer, functional, and compute audits;
6. lock Phase 1;
7. only then run N0–N6 and make the binary ASROA promotion decision.

No empirical success, near-frontier result, stability claim, or 6–7/10 novelty claim is asserted before these runs.

