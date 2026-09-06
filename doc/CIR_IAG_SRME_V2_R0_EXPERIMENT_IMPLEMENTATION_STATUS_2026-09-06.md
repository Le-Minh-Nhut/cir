# IAG-SRME V2 R0 experiment implementation status

This document records what the current branch implements and what has actually been
checked. It is not an architecture specification. The canonical source of truth remains
`CIR_IAG_SRME_UNIFIED_CANONICAL_ARCHITECTURE_AND_TRAINING_SPEC_V2_2026-09-06.md`.

## 1. Experiment identity

| Field | Value |
|---|---|
| Repository | `https://github.com/Le-Minh-Nhut/cir` |
| Branch | `exp/e2e-iag-srme-v2-r0` |
| Base branch | `exp/e2e-iag-srme-clean-rewrite` |
| Base merge-base | `f4bc1e8b91e5c43eec36e824fcd4c1d858f32308` |
| Current committed HEAD | `f9421827c7a65a965cc2605c20f94f2f649ebc47` (`v2`) |
| Record date | 2026-09-07 (implementation began 2026-09-06) |
| Canonical architecture | `doc/CIR_IAG_SRME_UNIFIED_CANONICAL_ARCHITECTURE_AND_TRAINING_SPEC_V2_2026-09-06.md` |
| Active backbone track | Track B — FG-CLIP v1 Base |
| Dataset/protocol | FashionIQ, original validation-gallery protocol |
| Current stage | A0-A6 implemented; Track-B R0-QG/R0-NCLS readout ablation implemented; no full comparison run |

No retrieval metric or readout-ablation winner is claimed.

## 2. Why this rewrite exists

The old IAG-SRME implementation was intentionally discarded, and the current model was
reconstructed from the V2 canonical specification. The current source tree no longer
contains static-`V0` grounding, fixed semantic candidate identities, old marginal or
entmax selector supervision, Gumbel/Straight-Through selection, soft candidate-state
mixtures, TAPER-era action-claim losses, or the R1b/R1c/R2 experimental modules. Hard
selection and current-state grounding are implemented directly in the compact V2 model.

## 3. Current model architecture

The main architecture remains readable top-to-bottom in
`src/models/iag_srme/model.py`.

| Component | Input | Output | Responsibility/location |
|---|---|---|---|
| Backbone initial state | reference pixels | `V0: [B,196,768]` | Penultimate FG-CLIP patches, CLS removed; `utils/backbone.py` |
| Current global readout | `V_t`, plus mode-specific query/anchor | `g_t: [B,768]` | R0-QG uses learned `q_G`; R0-NCLS uses immutable image-specific `CLS_(L-1)` with the exact native final block |
| ProposalNet | text tokens `[B,L,256]`, text global `[B,256]`, `g_t` | `e: [B,K,768]` | Fresh permutation-compatible edit candidates; no patch-state input |
| Grounder | `e`, native dense `[B,196,512]` | raw cosine, `alpha`, `M_exec: [B,K,196]` | Softmax read and separate sigmoid write maps |
| Entity read | `alpha`, current `V_t` | `h: [B,K,768]` | Pools values from the actual recurrent state |
| ActionFusion | `h`, `e` | `a: [B,K,768]` | Independent sigmoid gates, `a=gamma*h+beta*e` |
| Executor | shared parent `V_t`, `a`, `M_exec` | sibling states `[B,K,196,768]` | Masked local signed residual; zero-initialized no-op path |
| Consequence | parent/candidate states | `delta_q: [B,K,512]` | Same retrieval readout for both; `delta_q=q_candidate-q_now` |
| ScoreNet | five `d=256` feature groups | scores `[B,K]` | One shared independent `1280 -> 512 -> 256 -> 1` scorer |
| Commit/STOP | scores and candidate states | one next state | Hard argmax/gather; STOP keeps current state and is absorbing |

Every sibling at a timestep is previewed from exactly the same parent state. Only the
committed sibling continues; there is no beam, tree, or soft mixture.

## 4. Backbone contract and parity status

| Property | Implemented behavior | Status |
|---|---|---|
| Persistent state | `hidden_states[-2][:,1:]`; patch-only `H_(L-1)` | VERIFIED by unit and real-checkpoint shape check |
| CLS policy | Never part of mutable recurrent state; R0-NCLS carries an immutable readout anchor | VERIFIED |
| `q_G` initialization | Copy of checkpoint vision class embedding, then trainable in R0-QG | VERIFIED |
| R0-QG global readout | `q_G` cross-reads current patches with the preserved last-block-style Q/K/V, norms, output projection and MLP | VERIFIED regression-compatible; not native-parity by design |
| R0-NCLS global readout | `[CLS_anchor; V_t]` through the exact checkpoint final encoder layer, then native post-LN | VERIFIED native parity at `t=0` |
| Dense readout | `forward_without_attn -> post_layernorm -> visual_projection` | VERIFIED; real-checkpoint max/mean error observed as `0.0/0.0` |
| Retrieval readout | selected global readout, native visual projection, FP32 L2 normalization | R0-NCLS VERIFIED native parity at `t=0` |
| Projection | Checkpoint `visual_projection`, output width 512 | VERIFIED for dense path |
| Patch grid | `224/16 = 14 x 14`, 196 patches | VERIFIED on pinned Base checkpoint |
| Checkpoint | `qihoo360/fg-clip-base` | PINNED |
| Revision | `454d76372c2cf5eb48fa0d871fd0534481484d97` | PINNED |

Current-state perturbation changes global, dense, and retrieval readouts in both modes;
vectorized candidate retrieval equals a per-candidate loop. Native parity is expected
only from R0-NCLS. R0-QG remains the canonical learned recurrent query experiment.

### 4.1 Track-B global/retrieval readout ablation

| Property | R0-QG (`learned_qg`) | R0-NCLS (`native_cls`) |
|---|---|---|
| Mutable state | current penultimate patches `V_t` | current penultimate patches `V_t` |
| Global token source | one shared trainable `q_G`, initialized from checkpoint class embedding | image-specific `hidden_states[-2][:,0]` captured at reference encoding |
| Readout | preserved final-block-style cross-attention/MLP implementation | exact checkpoint final encoder layer on `[CLS_anchor;V_t]` |
| Anchor recurrence | no CLS anchor required | same immutable anchor at every timestep and sibling |
| Executor input | patches only | patches only; anchor is never passed to Executor |
| `t=0` native parity expectation | none | yes |
| Status | implemented and regression-tested | implemented; real-checkpoint parity verified |

Measured on one real FashionIQ image using the pinned checkpoint, revision, official
image processor, FP32 evaluation, native post-layernorm, visual projection, and L2
normalization:

| Comparison to official FG-CLIP | max absolute error | mean absolute error | cosine similarity |
|---|---:|---:|---:|
| R0-NCLS global | `0.0` | `0.0` | `0.9999998808` |
| R0-NCLS retrieval | `0.0` | `0.0` | `1.0000001192` |
| R0-QG global diagnostic | `10.69435024` | `0.62872618` | `0.50536633` |
| R0-QG retrieval diagnostic | `0.18923751` | `0.03000430` | `0.61011481` |

The R0-QG numbers are descriptive, not a failed parity test: that mode uses a shared
learned query and is not claimed to reconstruct native image features. With the same
R0-NCLS anchor and a `+0.1` perturbation to one patch coordinate, the measured global
L2 change was `0.0017186521` and retrieval L2 change was `0.0000561665`; therefore the
immutable anchor does not make the recurrent readout static.

The refactored R0-QG global output was also compared against an independent copy of the
pre-ablation learned-query equations on the real checkpoint: max/mean absolute error
were both `0.0` and cosine similarity was `1.0000001192`.

Candidate anchors are expanded only across the candidate dimension, preserving sample
association. Unit tests verify vectorized/loop equality and candidate-permutation
equivariance to `1e-6` in both modes. The deterministic unit fixture measured
vectorized/loop maximum errors of `1.1921e-7` (R0-QG) and `1.9372e-7` (R0-NCLS), and
permutation maximum errors of `7.9162e-8` and `1.9372e-7`, respectively.

### 4.2 Matched comparison protocol and logging

The two Hydra configurations inherit the same FG-CLIP Base settings. A valid comparison
must keep dataset split, caption policy, seed, checkpoint/revision, `K`, `T_max`, optimizer,
learning rate, weight decay, batch size, epochs, precision, losses, STOP settings, and
evaluation protocol fixed. The only intended architecture change is
`global_readout_mode`.

Evaluation already reports FashionIQ `recall_at_10`, `recall_at_50`, and `mean_recall`.
Training logs pair/gain loss, mean `||delta_q||`, within-sibling `delta_q` cosine, STOP
rate, mean committed rollout length, DPP valid rate/count, and useful-candidate count.
No comparison result has been generated yet.

## 5. Training objective and gradient routing

With every requested switch enabled, the implemented objective is

```text
L_total =
    L_terminal
  + lambda_pair * L_pair
  + lambda_gain * L_gain
  + lambda_c * L_c^SRME
  + lambda_bind * L_bind
  + lambda_rel * L_rel_ortho
  + lambda_dpp * L_DPP
```

| Term | Responsibility | Intended gradient destination |
|---|---|---|
| `L_terminal` | Retrieval quality of the hard-committed terminal state | Committed recurrent trajectory and legal target encoder path |
| `L_pair` | Candidate ordering | ScoreNet only; teacher and score features are detached |
| `L_gain` | Absolute calibration against STOP utility zero | ScoreNet only |
| `L_c^SRME` | Instruction-concept coverage by proposal set at `t=0` | Proposal and its `W_c` projection |
| `L_bind` | Edit/entity relation-distribution compatibility | Proposal, Grounder through `h -> alpha`, and relation bank/projections |
| `L_rel_ortho` | Relation-prototype collapse control | `P_rel` only |
| Functional DPP | Non-redundant useful retrieval consequences | Proposal, Grounder, Fusion, Executor through current `delta_q` |

`L_ortho(Q_prop)` is disabled. `L_cycle` and `L_ground_consistency` are disabled.
No correspondence term appears in `L_total`.

## 6. Current ablation ladder

| Stage | Objective | Status |
|---|---|---|
| A0 | terminal + pair | Implemented and unit-tested; not experimentally evaluated |
| A1 | A0 + gain | Implemented and unit-tested; not experimentally evaluated |
| A2 | A1 + STOP | Implemented and unit-tested; not experimentally evaluated |
| A3 | A2 + `L_bind` | Implemented and unit-tested; not experimentally evaluated |
| A4 | A2 + `L_c(t=0)` | Implemented and unit-tested; not experimentally evaluated |
| A5 | A2 + bind + concept | Implemented and unit-tested; not experimentally evaluated |
| A6 | A5 + Functional DPP | Implemented, unit-tested, one-update CPU smoke passed; not experimentally evaluated |

`L_rel_ortho` is an independent relation-bank collapse-control switch, not proposal
orthogonality and not an additional ladder claim. Hydra boolean/weight overrides can
produce every stage without changing source code.

## 7. Semantic auxiliary implementation

### 7.1 Instruction concept coverage

`src/models/iag_srme/utils/semantic.py` defines parser version
`fashioniq_concepts_v2`. It applies Unicode NFKC normalization, lowercasing, hyphen
splitting, an explicit small FashionIQ grammar/action stop-word list, and deterministic
unigram plus adjacent-bigram extraction. Phrase-only modifiers such as `more`, `less`,
`no`, `t`, and `v` are retained in adjacent phrases but are not standalone concepts;
bigrams never cross removed grammar boundaries. Vocabulary entries are counted from canonical
`ordered_and` captions in `FashionIQDataset.annotations` for the training split only,
then ordered by descending count with lexical tie-breaking. Defaults are minimum
frequency 2 and maximum size 2048.

For the local FashionIQ training annotations (18,000 rows), this policy produced 2,048
configured entries with fingerprint
`de1417114034284a7628ef78484255d5bdd1dbe4686eb29052cd46ff081e595d`.

Each concept is encoded once with the current FG-CLIP text model plus IAG text adapter;
the content-token mean (`backbone.encode_text` global output) becomes `w_c`. These
prototypes are detached, FP32-normalized buffers. `W_c(e)` is normalized and scored by
temperature-scaled cosine. The `K` scores are pooled by `tau_mil * logsumexp` before the
instruction multi-label vector enters the asymmetric loss. The loss uses only the
proposal set at `t=0`; targets and teacher utility are absent from label construction.

### 7.2 Relation binding

`P_rel: [num_relation_prototypes,state_dim]` is a separate trainable parameter owned by
the objective. It is not `ProposalNet.queries` and shares no storage with it. Projected,
normalized edits and grounded entities produce `b_e` and `b_h` through cosine-softmax
over all relation prototypes. The implemented direction is exactly
`KL(b_e || b_h)`, averaged over all live timestep/candidate rows.

### 7.3 Relation orthogonality

Normalized `P_rel` is regularized by
`||P_bar P_bar^T - I||_F^2 / (M(M-1))`. This term uses and trains only the relation
prototype bank. It never reads proposal queries, edits, actions, or masks.

## 8. Functional DPP

The only functional feature is the model's existing
`delta_q = candidate_query - current_query`. Current effects are normalized by
`delta_q/(||delta_q||+eps)` without detaching them. Similarity is the PSD RBF kernel
with configurable `sigma_dpp`.

At timestep `t`, history contains only normalized effects selected and executed at
earlier timesteps for the same sample. History is detached; STOP adds nothing, and
unselected siblings never enter it. With history, novelty is conditioned by the Schur
complement using `torch.linalg.solve(S_HH + jitter*I, S_HC)`; at `t=0`, it is `S_CC`.

Quality is `stop_gradient(sigmoid(u*/tau_dpp))`. The target-derived teacher utility and
quality have no DPP gradient. A sample/timestep is skipped unless at least two detached
qualities exceed `useful_threshold`. The current candidates retain gradients through
the log determinant.

The two DPP scales are intentionally distinct:

- `kappa_dpp` changes the nonlinear kernel inside `-logdet(I+kappa*Q*S_cond*Q)`.
- `lambda_dpp` only weights the already-computed raw DPP loss in `L_total`.

Both raw and weighted losses, along with `sigma_dpp`, `tau_dpp`, useful count, valid
timestep count, within-sibling effect cosine, and effective functional rank are logged.

## 9. STOP and hard rollout semantics

STOP means KEEP the current state; its utility is exactly zero. The best candidate is
chosen by hard `argmax`, then committed with hard gather only when
`best_score > epsilon_stop`. No learned STOP transition, Gumbel estimator,
Straight-Through estimator, or weighted state mixture exists. A stopped sample leaves
the live set and is absorbing for the rest of the rollout.

## 10. Target firewall

Target information is legal only in the terminal retrieval loss and the training-only
teacher. Functional DPP may use teacher utility only after detaching it as quality.
Targets are forbidden and absent from Proposal, Grounder, ActionFusion, Executor,
ScoreNet inputs, state transitions, and STOP/inference decisions. The model forward
signature remains target-free.

## 11. Teacher policy

The teacher uses the in-batch target bank and stable `CIRSample.target_id` identities.
Equal valid target IDs form a multi-positive relation, so duplicates are not false
negatives. Every parent and all of its siblings use the same positive/negative masks and
target pool. Rows without a positive or valid negative are skipped. Similarities and
logsumexp calculations run in FP32. Current FashionIQ training uses exact target-ID
positives; dataset-specific graded/multi-positive relevance beyond exact ID is not wired
into the loader.

## 12. Correspondence status: DISABLED

The current config sets `correspondence_enabled: false`; attempting to construct the
objective with it enabled raises an error. There is no active `P_corr`, `M_0t/M_t0`,
`L_cycle`, `L_ground_consistency`, tracked grounding, reference-to-current mapping, or
correspondence-progress feature. Correspondence affects neither model forward nor loss.

Correspondence is intentionally deferred as a diagnostic branch and should only be
promoted if measured identity/grounding drift justifies it. Its architectural idea
remains in the canonical specification.

## 13. Deleted legacy components

| Path | Reason removed | Replacement/current mechanism |
|---|---|---|
| `reports/iag_srme_2026-08-30_best_diagnostic.json` | Stale result snapshot from the pre-V2 implementation | New experiments must emit fresh V2 diagnostics |
| `reports/iag_srme_2026-08-30_last_diagnostic.json` | Same; not imported or used by train/eval | Fresh V2 run output |
| `reports/iag_srme_bind_best_diagnostic.json` | Historical binding experiment with obsolete semantics | V2 relation-distribution `KL(b_e || b_h)` |
| `reports/iag_srme_comp_best_diagnostic.json` | Historical complementarity experiment | Functional DPP in `delta_q` space |
| `reports/iag_srme_comp_bind_best_diagnostic.json` | Historical combined snapshot | Independently switchable V2 auxiliaries |
| `reports/iag_srme_factor_best_diagnostic.json` | Historical factorization snapshot | No V2 factorization objective |

No current import, entrypoint, config, or documentation referenced these files. The
empty tracked `reports/` directory consequently disappears. Historical architecture
specification V1 is intentionally retained as research history, not as source of truth.

## 14. Current test status

| Group | Status | Evidence |
|---|---|---|
| Backbone | PASS | R0-QG regression, real-checkpoint R0-NCLS exact native parity, immutable anchor, dynamic-state and vectorized/loop tests |
| Proposal | PASS | Text/global dependence, no patch argument, recurrent reproposal |
| Grounding | PASS | Patch softmax, distinct sigmoid write, current-state entity pool |
| Fusion | PASS | Independent bounded gates and candidate permutation equivariance |
| Executor | PASS | Identity initialization, zero-mask no-op, same parent, permutation equivariance |
| ScoreNet/STOP | PASS | Shared scorer, zero STOP anchor, hard single commit, absorbing STOP |
| Teacher | PASS | Duplicate positives, no false negatives, shared pool, empty-negative skip |
| Target firewall | PASS | Live output unchanged; target affects only training objective branches |
| Semantic auxiliaries | PASS | Instruction-only/parser determinism, set pooling/permutation, t0 routing, bind/ortho gradients |
| Functional DPP | PASS | `delta_q` gradient, detached history/quality, executed-only history, guard, permutation and redundancy tests |
| CUDA optimizer placement | NOT RUN | CUDA unavailable; one pytest skipped |
| Prior R0-QG FashionIQ A6 canary | PASS | CPU FP32, batch 2, one optimizer update, three rollout steps |
| R0-NCLS full-model forward smoke | PASS | Real checkpoint/image; state `[1,196,768]`, anchor `[1,768]`, sibling queries `[1,4,512]` |

The final suite result is `47 passed, 1 skipped`; the skip is the CUDA-only optimizer
placement test because CUDA is unavailable. The real-checkpoint parity test passed.
Ruff reports `All checks passed`. The prior R0-QG post-threshold canary produced finite values:
`total=0.803284`, `terminal=0.791728`, `gain=0.001349`,
`concept=0.879584`, `bind=0.208447`, `rel_ortho=0.001243`, and `dpp_raw=0.0`.
The DPP skip is expected at the zero-initialized Executor because fewer than two
candidates cleared the configured useful-quality guard; it confirms the guard rather
than demonstrating learned functional diversity.

## 15. Known issues and unresolved questions

- **Resolved:** R0-NCLS reconstructs the official native global and retrieval outputs at
  `t=0` with measured zero max/mean absolute error in the pinned FP32 check. R0-QG is
  explicitly a non-native learned recurrent readout and is no longer mislabeled as a
  pending native-parity path.
- **P1 — before full comparison training:** Run a one-update R0-NCLS canary on the
  intended CUDA/AMP environment and verify checkpoint save/load plus evaluation mode
  mismatch protection end-to-end. This task did not launch training.
- **P1 — monitor during warm-up:** Functional DPP is legitimately inactive while fewer
  than two candidate utilities exceed the useful threshold. Measure valid-DPP rate and
  useful-effect redundancy before tuning `useful_threshold`, `tau_dpp`, or `kappa_dpp`.
- **P1 — inspect vocabulary artifact:** The deterministic parser is deliberately simple.
  Record the checkpoint vocabulary fingerprint and inspect frequent concepts before a
  long experiment; do not replace it with target-derived labels.
- **P2 — teacher relevance:** Exact stable-ID multi-positive handling is implemented.
  Broader dataset-specific relevance relations are not currently supplied.
- **P2 — experimental evidence:** No A0-A6 retrieval experiment or ablation metric has
  been run; only unit checks and a one-update canary exist.

## 16. Current experiment readiness

```text
Architecture implementation: A0-A6 code complete for the requested V2 auxiliaries
Unit-test readiness:       47 passed, 1 CUDA-only test skipped
Backbone parity:           native_cls global/retrieval exact at t=0; learned_qg diagnostic only
Training smoke test:       prior R0-QG CPU A6 canary passed; R0-NCLS training not run
Ready for full training:   NO
```

The readout implementation/parity audit is complete. The remaining operational blocker
is the matched R0-NCLS CUDA/AMP canary and checkpoint/evaluation round trip; no
architecture or loss redesign is indicated.

## 17. Next experiments

1. Run the minimal R0-QG and R0-NCLS FashionIQ canaries on the intended CUDA/AMP environment.
2. Verify checkpoint save/load and reject evaluation with the wrong readout mode.
3. Run A0, then A1, then A2 and establish the clean-core baseline with a declared readout.
4. Compare A3 (`L_bind`) and A4 (`L_c(t=0)`) independently.
5. Run A5 only after the individual semantic branches are healthy.
6. Compare matched R0-QG versus R0-NCLS, changing only the readout config.
7. Enable A6 after measuring useful-effect redundancy and report valid-DPP frequency.
8. Keep correspondence disabled unless measured identity/grounding drift motivates a diagnostic branch.
