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
| Current remote HEAD / diagnostics implementation base | `c00147863c34c5871f65d62b76a04a1aa985954a` |
| Record date | 2026-09-07 (implementation began 2026-09-06) |
| Canonical architecture | `doc/CIR_IAG_SRME_UNIFIED_CANONICAL_ARCHITECTURE_AND_TRAINING_SPEC_V2_2026-09-06.md` |
| Active backbone track | Track B — FG-CLIP v1 Base |
| Dataset/protocol | FashionIQ, original validation-gallery protocol |
| Current stage | A0-A6 implemented; 2x2 Track-B readout/fine-tuning ablation implemented; no comparison run |

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
| Current global readout | `V_t`, plus mode-specific query/anchor | `g_t: [B,768]` | R0-QG uses learned `q_G`; R0-NCLS computes the exact CLS row of the native final block from immutable image-specific `CLS_(L-1)` and current patches |
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
| R0-NCLS global readout | Exact final-block CLS attention row from `[CLS_anchor; V_t]`, CLS residual/MLP, then native post-LN; full-token checkpoint call retained as oracle | VERIFIED against full block and native output at `t=0` |
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
| Readout | preserved final-block-style cross-attention/MLP implementation | exact CLS row of the checkpoint final encoder layer on `[CLS_anchor;V_t]` |
| Anchor recurrence | no CLS anchor required | same immutable anchor at every timestep and sibling |
| Executor input | patches only | patches only; anchor is never passed to Executor |
| `t=0` native parity expectation | none | yes |
| Status | implemented and regression-tested | optimized CLS-only production path; full-block and real-checkpoint parity verified |

Measured on one real FashionIQ image using the pinned checkpoint, revision, official
image processor, FP32 evaluation, native post-layernorm, visual projection, and L2
normalization:

| Readout comparison | max absolute error | mean absolute error | cosine similarity |
|---|---:|---:|---:|
| Full block vs CLS-only global | `1.4603138e-6` | `2.8152968e-7` | `0.9999999404` |
| Full block vs CLS-only retrieval | `5.9604645e-8` | `1.3475628e-8` | `1.0000001192` |
| R0-NCLS CLS-only global | `1.4603138e-6` | `2.8152968e-7` | `0.9999999404` |
| R0-NCLS CLS-only retrieval | `5.9604645e-8` | `1.3475628e-8` | `1.0000001192` |
| R0-QG global diagnostic | `10.69435024` | `0.62872618` | `0.50536633` |
| R0-QG retrieval diagnostic | `0.18923751` | `0.03000430` | `0.61011481` |

The R0-QG numbers are descriptive, not a failed parity test: that mode uses a shared
learned query and is not claimed to reconstruct native image features. With the same
R0-NCLS anchor and a `+0.1` perturbation to one patch coordinate, the measured global
L2 change was `0.0017184421` and retrieval L2 change was `0.0000561443`; therefore the
immutable anchor does not make the recurrent readout static.

The refactored R0-QG global output was also compared against an independent copy of the
pre-ablation learned-query equations on the real checkpoint: max/mean absolute error
were both `0.0` and cosine similarity was `1.0000001192`.

Candidate anchors are expanded only across the candidate dimension, preserving sample
association. Unit tests verify vectorized/loop equality and candidate-permutation
equivariance to `1e-6` in both modes. The deterministic unit fixture measured
vectorized/loop maximum errors of `5.9605e-8` (R0-QG) and `1.1921e-7` (R0-NCLS), and
permutation maximum errors of `5.9605e-8` and `8.9407e-8`, respectively.

The recurrent path computes exactly two global readouts per live timestep: one for the
parent and one vectorized call for all `K` siblings. `retrieval_from_global` applies the
native visual projection followed by FP32 L2 normalization, so neither parent nor
candidate global computation is repeated. The compatibility `retrieval_readout`
wrapper remains available and produces numerically identical queries. One additional
global readout produces the terminal query after rollout.

The native CLS-only implementation uses checkpoint `layer_norm1`, CLS `q_proj`, all-token
`k_proj/v_proj`, native attention scale/dropout and `out_proj`, then applies the CLS
residual, checkpoint `layer_norm2`/MLP residual, and native post-layernorm. It does not
materialize final-block patch outputs. The exact full-token layer call remains private
test oracle `_native_cls_readout_full` and is not used by production rollout.

Per-step output no longer retains `raw_delta`, `candidate_global`, or the 1280-wide
`score_features`, because no objective, diagnostic, evaluation path, or test consumes
them. It retains `parent_state`, masked `delta`, and `candidate_states` for same-parent,
hard-commit, and selected-trajectory gradient diagnostics, plus the compact semantic,
score, and retrieval-consequence tensors required by current objectives.

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

### 4.3 FG-CLIP fine-tuning ablation

Readout and fine-tuning are orthogonal controlled axes:

| | FULL (`full`) | TEXT-ONLY (`text_only`) |
|---|---|---|
| `learned_qg` | `R0-QG-FULL` | `R0-QG-TEXT` |
| `native_cls` | `R0-NCLS-FULL` | `R0-NCLS-TEXT` |

FULL preserves the existing policy: FG-CLIP `vision_model` and `text_model` are
trainable, `visual_projection` follows the vision policy, and `text_projection`
remains frozen. TEXT-ONLY freezes `vision_model` and `visual_projection`, keeps
`text_model` and the IAG text adapter trainable, and leaves every IAG-SRME task module
and enabled objective auxiliary trainable. Unused pretrained logit-scale/fine-grained
heads are frozen under both policies.

`q_G` is a task-specific recurrent parameter rather than a pretrained vision weight.
It remains trainable in QG-FULL and QG-TEXT. It is disabled from optimization in NCLS
runs because native CLS readout never consumes it.

Frozen vision affects parameter ownership, module mode, and initial/target image graph
construction only. `initial_state_with_anchor` and `encode_global_images` use
`no_grad()` when vision is frozen. Current-state `global_readout`, `dense_readout`, and
`retrieval_from_global` do not: gradients still pass through their frozen operations
to mutable candidate states and the Executor. `FGCLIPBackbone.train()` forces the
frozen vision model and visual projection to eval mode while the text model remains in
training mode.

Counts below were computed from the pinned real checkpoint with `K=4`, `T=3`, width
256, and the enabled A6 objective. Concept vocabulary size does not change trainable
parameter count because prototypes are buffers.

| Variant | Total | Trainable | Vision | Visual proj. | Text | Text adapter | `q_G` | IAG modules | Objective |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| R0-QG-FULL | 168,854,279 | 168,329,988 | 85,799,424 | 393,216 | 63,419,904 | 131,840 | 768 | 17,202,436 | 1,382,400 |
| R0-NCLS-FULL | 168,854,279 | 168,329,220 | 85,799,424 | 393,216 | 63,419,904 | 131,840 | 0 | 17,202,436 | 1,382,400 |
| R0-QG-TEXT | 168,854,279 | 82,137,348 | 0 | 0 | 63,419,904 | 131,840 | 768 | 17,202,436 | 1,382,400 |
| R0-NCLS-TEXT | 168,854,279 | 82,136,580 | 0 | 0 | 63,419,904 | 131,840 | 0 | 17,202,436 | 1,382,400 |

Hydra config equality tests verify that each FULL/TEXT pair differs only in
`train_vision`, `finetune_policy`, config name, and experiment identity. Checkpoints
record and evaluation validates `global_readout_mode`, `readout_experiment`,
`finetune_policy`, `train_vision`, `train_text`, and `train_text_projection`.

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
| `L_pair` | Candidate ordering | ScoreNet only; teacher and upstream feature inputs are detached before ScoreNet projections |
| `L_gain` | Absolute calibration against STOP utility zero | ScoreNet only |
| `L_c^SRME` | Instruction-concept coverage by proposal set at `t=0` | Proposal and its `W_c` projection |
| `L_bind` | Edit/entity relation-distribution compatibility | Proposal, Grounder through `h -> alpha`, and relation bank/projections |
| `L_rel_ortho` | Relation-prototype collapse control | `P_rel` only |
| Functional DPP | Non-redundant useful retrieval consequences | Proposal, Grounder, Fusion, Executor through current `delta_q` |

`L_ortho(Q_prop)` is disabled. `L_cycle` and `L_ground_consistency` are disabled.
No correspondence term appears in `L_total`.

ScoreNet routing detaches `current_global`, text context, actions, local delta, execution
mask, and candidate global only at the boundary into `ScoreNet.build_features`. The
trainable context/action/local/global projections, context LayerNorm, and scoring MLP
therefore all receive pair/gain gradients. Live candidate queries and `delta_q` are not
detached, preserving terminal and Functional-DPP routes.

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
| Backbone | PASS | R0-QG regression, full-vs-CLS-only and official R0-NCLS parity, immutable anchor, dynamic-state, vectorized/loop, and non-duplicated-call tests |
| Proposal | PASS | Text/global dependence, no patch argument, recurrent reproposal |
| Grounding | PASS | Patch softmax, distinct sigmoid write, current-state entity pool |
| Fusion | PASS | Independent bounded gates and candidate permutation equivariance |
| Executor | PASS | Identity initialization, zero-mask no-op, same parent, permutation equivariance |
| ScoreNet/STOP | PASS | Every ScoreNet projection/norm/MLP group receives isolated pair/gain gradient; backbone, Proposal, Grounder, Fusion, and Executor receive none; STOP unchanged |
| Teacher | PASS | Duplicate positives, no false negatives, shared pool, empty-negative skip |
| Target firewall | PASS | Live output unchanged; target affects only training objective branches |
| Semantic auxiliaries | PASS | Instruction-only/parser determinism, set pooling/permutation, t0 routing, bind/ortho gradients |
| Functional DPP | PASS | `delta_q` gradient, detached history/quality, executed-only history, guard, permutation and redundancy tests |
| Fine-tuning policy | PASS | FULL/TEXT config equality, frozen eval mode, parameter ownership, checkpoint metadata, QG and NCLS recurrent gradient routing |
| CUDA optimizer placement | NOT RUN | CUDA unavailable; one pytest skipped |
| Prior R0-QG FashionIQ A6 canary | PASS | CPU FP32, batch 2, one optimizer update, three rollout steps |
| R0-NCLS full-model forward smoke | PASS | Real checkpoint/image; state `[1,196,768]`, anchor `[1,768]`, sibling queries `[1,4,512]` |

The pre-diagnostics suite result was `58 passed, 1 skipped`; the skip is the CUDA-only optimizer
placement test because CUDA is unavailable. The real-checkpoint parity test passed.
Ruff reports `All checks passed`. The prior R0-QG post-threshold canary produced finite values:
`total=0.803284`, `terminal=0.791728`, `gain=0.001349`,
`concept=0.879584`, `bind=0.208447`, `rel_ortho=0.001243`, and `dpp_raw=0.0`.
The DPP skip is expected at the zero-initialized Executor because fewer than two
candidates cleared the configured useful-quality guard; it confirms the guard rather
than demonstrating learned functional diversity.

The canary now resets CUDA peak statistics before its update and reports peak allocated
and reserved GiB. This host reports `torch.cuda.is_available() == false`, so neither the
requested FP16 CUDA canaries nor before/after VRAM measurements were run; no memory
number is inferred from CPU execution.

## 15. Known issues and unresolved questions

- **Resolved:** the optimized R0-NCLS CLS-only path reconstructs the official native
  global/retrieval outputs at `t=0` within `1.4603138e-6`/`5.9604645e-8` maximum
  absolute error in the pinned FP32 check. R0-QG is
  explicitly a non-native learned recurrent readout and is no longer mislabeled as a
  pending native-parity path.
- **P1 — before full comparison training:** Run matched one-update FULL/TEXT canaries
  for both readouts on the intended CUDA/AMP environment and record allocated/reserved
  VRAM. Checkpoint metadata save and mismatch rejection are unit-tested, but a complete
  CUDA checkpoint/evaluation round trip remains pending.
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
Unit-test readiness:       58 passed, 1 CUDA-only test skipped
Backbone parity:           native_cls CLS-only within 1.47e-6 global max error at t=0
Training smoke test:       prior R0-QG-FULL CPU canary passed; 2x2 CUDA/FP16 canaries not run
Ready for full training:   NO
```

The readout and fine-tuning policy implementation audits are complete. The remaining
operational blocker is the matched 2x2 CUDA/AMP canary, VRAM measurement, and
checkpoint/evaluation round trip; no architecture or loss redesign is indicated.

## 17. Next experiments

1. Run R0-QG-FULL, R0-NCLS-FULL, R0-QG-TEXT, and R0-NCLS-TEXT FashionIQ canaries with matched CUDA/AMP settings.
2. Record peak allocated/reserved VRAM and verify checkpoint/evaluation round trips for all four identities.
3. Run A0, then A1, then A2 and establish the clean-core baseline with a declared readout.
4. Compare A3 (`L_bind`) and A4 (`L_c(t=0)`) independently.
5. Run A5 only after the individual semantic branches are healthy.
6. Run the matched 2x2 comparison, changing only readout or fine-tuning policy on each controlled axis.
7. Enable A6 after measuring useful-effect redundancy and report valid-DPP frequency.
8. Keep correspondence disabled unless measured identity/grounding drift motivates a diagnostic branch.

## 18. Diagnostic observability update — 2026-09-07

The text-only policy is unchanged: the FG-CLIP vision encoder and native visual
projection stay frozen, while the text encoder, text adapter, and IAG-SRME modules
remain trainable. This update changes measurement only; it does not alter Proposal,
Grounder, Fusion, Executor, ScoreNet, STOP, hard selection, teacher, semantic losses,
or Functional DPP.

Candidate geometry now retains `[B,K,D]` until it reports three separate views:
historical pooled `[B*K,D]`, per-slot cross-input spectra, and slot-centered pooled
spectra. It also reports the population covariance-trace decomposition into within-slot
and between-slot variance for both raw and L2-normalized features. Zero-variance/effect
representations are explicitly marked degenerate and use rank `0`, rather than being
misreported as meaningful rank-one structure.

All three diagnostics require and validate one persistent sample manifest containing
stable sample/reference/target IDs, category, and composed instruction. Geometry and
selector scripts no longer independently shuffle their cohorts. Transition records use
`live_indices` and stable sample IDs; before/after edit comparisons are restricted to
the same executed rows. Selector output now includes parent/sibling teacher losses,
utilities, positive and hardest-negative similarity/margin, per-slot quality and
occupancy, selected/oracle regret, explicit STOP confusion and denominators, and a
same-reference within-category caption-shuffle test at `t=0`. Official FashionIQ recall
can be requested separately with `+diagnostic_official_eval=true`.

Training writes append-only `metrics.jsonl`, `resolved_config.yaml`, and
`run_metadata.json`. Update records include all objective components, learning rate,
AMP scale/overflow inference, non-finite gradient count, true optimizer-step count, and
representative gradient/update-to-weight probes for text, Proposal, Grounder, Fusion,
Executor, and ScoreNet. Evaluation now rejects mismatched `K`, `T`, STOP enablement,
STOP threshold, readout, fine-tuning policy, checkpoint, or revision unless the run is
explicitly labeled counterfactual.

Source audit notes: current step output does not retain `candidate_global`, so the
read-only geometry diagnostic reconstructs it from `candidate_states` with the selected
backbone readout. `selected_idx == K` is the source-level STOP sentinel, and the current
configured `epsilon_stop` is `0.0`. The canonical Track-B mainline describes full
vision/text fine-tuning; this branch intentionally remains the separately documented
TEXT-only controlled ablation.

The post-change CPU/unit suite is `68 passed, 1 skipped`; the skip remains CUDA-only.
The three diagnostic entrypoints import and resolve their Hydra configuration
successfully. A checkpoint-backed diagnostic smoke was not run on this host because the
OLD/STRONG `.pt` files are not present in the workspace, and CUDA is unavailable; no
diagnostic measurements are fabricated.
