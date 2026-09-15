# A5.1c Sequential Residual Claiming — Diagnostic README

> **Branch:** `exp/e2e-a5.1c-sequential-residual-claiming`  
> **Experiment:** A5.1c  
> **Status:** **CLOSED — FAILED / informative negative result**  
> **Best validation Mean Recall:** **59.0303**  
> **Best checkpoint:** `outputs/2026-08-23/07-10-17/best.pt`

---

# 1. Purpose of this branch

A5.1c tests the strongest residual-evidence hypothesis attempted in TAPER so far:

> If edit slots are processed sequentially and each slot immediately depletes evidence available to later slots, will TAPER stop producing redundant global edit slots?

This branch is specifically intended to distinguish true **sequential residual claiming** from the earlier parallel residual variants.

It keeps the current:

- retrieval objective,
- CSMCIR teacher,
- TAPER Executor,
- primitive bank,
- contextual text representation,

while changing the slot allocation/refinement mechanism.

The scientific question is intentionally narrow:

> **Is sequential evidence depletion sufficient under the current TAPER representation and Executor contract?**

The observed answer is **no**, although the mechanism creates some genuine differentiation.

---

# 2. Main configuration

```yaml
num_refine_iters: 3
residual_bias_strength: 1.0
residual_depletion_power: 1.0
residual_eps: 1.0e-6
randomize_slot_order_during_training: true
```

Relevant surrounding settings:

```yaml
num_slots: 4
num_primitives: 8

mask_temperature: 1.0
router_temperature: 1.0
retrieval_temperature: 0.07

slot_gate_threshold: 0.4
hard_slot_gating_during_training: false
gate_mode: legacy_soft_train_hard_eval
st_gate_recovery: false
```

Linear depletion is intentionally used:

\[
p=1.
\]

Do not silently switch to quadratic if comparing against the frozen branch result.

---

# 3. Sequential residual algorithm

At each refinement round:

1. initialize residual evidence to 1 on valid content tokens,
2. randomize slot processing order during training,
3. use canonical slot index order at evaluation,
4. process one slot at a time,
5. immediately update residual before the next slot,
6. reset residual at the next outer refinement round.

For slot \(s\):

\[
h_n^{eff}=r_nh_n
\]

\[
k_n=W_k(h_n^{eff})
\]

\[
v_n=W_v(h_n^{eff})
\]

\[
\ell_{sn}
=
\frac{q_s^\top k_n}{\sqrt d\,\tau}
+
\gamma\log(r_n+\epsilon)
\]

\[
\alpha_s=\operatorname{softmax}_{tokens}(\ell_s)
\]

\[
e_s=\sum_n\alpha_{sn}v_n
\]

and, when the refinement round updates states:

\[
q_s'=\operatorname{GRU}(e_s,q_s).
\]

Linear depletion:

\[
r_n'=r_n(1-\alpha_{sn}).
\]

Consumed claim:

\[
c_{sn}=r_n-r_n'=r_n\alpha_{sn}.
\]

---

# 4. Required invariants

## 4.1 Allocation conservation

For every valid token:

\[
\sum_s c_{sn}+r_n^{final}=1.
\]

Diagnostic key:

```text
round_X/allocation_conservation_max_error
```

The diagnostic should abort when error exceeds tolerance.

## 4.2 Residual monotonicity

\[
r_n^{s+1}\le r_n^s.
\]

Diagnostic key:

```text
round_X/residual_max_increase
```

Expected:

```text
0 or numerical noise <= 1e-6
```

## 4.3 Invalid-token exclusion

Edit evidence is restricted to:

```text
attention_valid AND text_content_mask
```

Special/non-content tokens must not become edit claims.

## 4.4 Exact no-op Executor state

Invalid/no-update Executor steps must preserve the previous state exactly. Otherwise retrieval interventions are contaminated by the state-update machinery itself.

---

# 5. Tensor semantics — do not mix these up

This is the most important diagnostic contract in A5.1c.

## `refine_slot_attentions`

Shape:

```text
[B, T, L, N]
```

Meaning:

\[
\alpha_{sn}
\]

raw sequential token attention for each slot.

**This is the primary tensor for diagnosing attention collapse.**

## `refine_slot_masks`

Shape:

```text
[B, T, L, N]
```

Meaning:

\[
c_{sn}=r_n\alpha_{sn}
\]

consumed residual claim.

This is **not raw attention**.

Claims can become different simply because earlier slots depleted residual evidence. Therefore low claim cosine does not prove semantic specialization.

## `refine_sequential_score_logits`

Shape:

```text
[B, T, L, N]
```

Meaning: actual raw sequential scorer logits before token softmax.

Use this for scorer-scale diagnostics.

## `refine_residual_trajectories`

Shape:

```text
[B, T, L+1, N]
```

Contains initial residual and residual after each sequential slot.

## `refine_null_probs`

Shape:

```text
[B, T, N]
```

Meaning: final leftover unexplained residual after all slots in a round.

In A5.1c, NULL is not a learned competitor inside the sequential attention. NULL is residual capacity left after slot claiming.

---

# 6. Collapse metrics

## 6.1 Mean raw-attention pair cosine

```text
round_X/slot_attention_pair_cos
```

\[
\frac{1}{\binom L2}\sum_{i<j}\cos(\alpha_i,\alpha_j).
\]

Useful for global structure.

## 6.2 Worst all-slot attention cosine

```text
round_X/slot_attention_max_pair_cos
```

For each sample:

\[
\max_{i<j}\cos(\alpha_i,\alpha_j)
\]

then averaged across samples.

This catches one clone pair that mean cosine can hide.

## 6.3 Worst active-slot attention cosine

```text
round_X/slot_attention_max_active_pair_cos
```

For active slot set \(A\):

\[
\max_{i<j,\ i,j\in A}\cos(\alpha_i,\alpha_j).
\]

This is the closest TAPER analogue to the **max active overlap** metric in Houba (2026).

### Caveat

The paper uses a learned existence head. TAPER uses the final hard-active gate mask:

```text
hard_active_slot_mask
```

so the semantics are analogous, not identical.

## 6.4 Claim cosine

```text
round_X/slot_claim_pair_cos
```

Secondary metric only, because claim diversity can be mechanically induced by residual depletion.

---

# 7. Final representation metrics

The forensic report includes:

```text
final/semantic_pair_cos_all
final/semantic_pair_cos_active

final/effect_pair_cos_all
final/effect_pair_cos_active

final/edit_slot_pair_cos_all
final/edit_slot_pair_cos_active
```

## Semantic cosine

Similarity of slot-pooled text semantics.

## Effect cosine

Similarity of frozen-teacher counterfactual effects:

\[
\Delta q_l=q_{full}-q_{minus,l}.
\]

This asks whether removing support assigned to different slots changes the teacher composition differently.

## Edit-slot cosine

Similarity after the slot MLP produces actual vectors consumed by the Executor.

For true factorization, all three should move away from 1 in a meaningful and stable way.

---

# 8. Retrieval intervention battery

Geometry alone is insufficient.

## FULL

```text
FULL
```

Normal execution.

## REFERENCE_ONLY

```text
REFERENCE_ONLY
```

Defines baseline modification gain:

\[
G=MR_{FULL}-MR_{REF}.
\]

## DROP one slot

```text
DROP_S0
DROP_S1
DROP_S2
DROP_S3
```

Tests whether removing a slot destroys unique information.

## KEEP one slot

```text
KEEP_S0
KEEP_S1
KEEP_S2
KEEP_S3
```

Tests how much of the modification a single original slot carries.

## REPEAT one slot

```text
REPEAT_Si_X1
REPEAT_Si_X2
REPEAT_Si_X3
REPEAT_Si_X4
```

This is the key slot-as-compute-ticket test.

If

\[
\text{REPEAT}(S_i,4)\approx FULL,
\]

one slot plus repeated execution depth is sufficient.

## MEAN SLOT depth curve

```text
MEAN_SLOT_X1
MEAN_SLOT_X2
MEAN_SLOT_X3
MEAN_SLOT_X4
```

If the mean slot repeated four times recovers FULL, the model is strongly consistent with a global-edit + repeated-compute shortcut.

---

# 9. Training command

Example 15-epoch run:

```bash
python src/train.py \
  experiment=taper_e2e \
  experiment.num_epochs=15
```

Recorded validation trajectory:

```text
epoch  1: 53.0807
epoch  2: 56.6329
epoch  3: 57.5924
epoch  4: 58.0223
epoch  5: 58.6688
epoch  6: 58.4359
epoch  7: 59.0303  <-- best
epoch  8: 58.7626
epoch  9: 58.8736
epoch 10: 58.4798
epoch 11: 58.2178
epoch 12: 57.6034
epoch 13: 57.9599
epoch 14: 57.0385
epoch 15: 57.3782
```

Training loss continued falling while validation retrieval degraded after epoch 7. Falling training loss must not be interpreted as improving factorization.

---

# 10. Selecting the newest best checkpoint

From repository root:

```bash
BEST=$(find outputs -type f -name 'best.pt' -printf '%T@ %p\n' \
  | sort -nr \
  | head -n 1 \
  | cut -d' ' -f2-)

echo "Using checkpoint: $BEST"
```

Always inspect the printed path.

Frozen best checkpoint for this documented run:

```text
outputs/2026-08-23/07-10-17/best.pt
```

---

# 11. Running the forensic diagnostic

Using `$BEST`:

```bash
python src/diagnose_a5.py \
  experiment=taper_e2e \
  +checkpoint="$BEST" \
  +report=reports/a5_1c_forensic.json
```

Frozen run:

```bash
python src/diagnose_a5.py \
  experiment=taper_e2e \
  +checkpoint=outputs/2026-08-23/07-10-17/best.pt \
  +report=reports/a5_1c_forensic_15ep.json
```

For full Hydra traces:

```bash
HYDRA_FULL_ERROR=1 python src/diagnose_a5.py \
  experiment=taper_e2e \
  +checkpoint=outputs/2026-08-23/07-10-17/best.pt \
  +report=reports/a5_1c_forensic_15ep.json
```

---

# 12. Critical reproducibility issue: `masked_mean()`

A real runtime failure was observed:

```text
NameError: name 'masked_mean' is not defined
```

The local working tree was repaired by restoring:

```python
def masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    if values.shape != mask.shape:
        raise ValueError(
            f"values and mask must match, got "
            f"{tuple(values.shape)} vs {tuple(mask.shape)}"
        )

    mask_f = mask.to(values.dtype)

    return (
        values * mask_f
    ).sum() / mask_f.sum().clamp_min(1.0)
```

**As of the GitHub branch audit used to write this README, the remote version of `src/diagnose_a5.py` still does not contain this helper.**

That means the successful forensic result came from the repaired local file, while a clean checkout of the current remote branch can still crash.

Before freezing/handoff:

1. ensure `masked_mean()` exists locally,
2. run the diagnostic end-to-end,
3. commit the helper restoration,
4. push,
5. clean-checkout/pull and rerun.

Important: this is a runtime `NameError`, so

```bash
python -m py_compile src/diagnose_a5.py
```

will **not** catch it.

Recommended checks:

```bash
python -m py_compile \
  src/models/taper.py \
  src/train.py \
  src/diagnose_a5.py
```

followed by a real forensic run.

---

# 13. Harmless environment warnings

Warnings such as:

```text
FutureWarning: torch.utils._pytree._register_pytree_node is deprecated
FutureWarning: resume_download is deprecated
```

are not the A5.1c failure.

The previous diagnostic crash happened after model loading/forward because the helper function was missing.

---

# 14. Frozen best-checkpoint structural results

Checkpoint:

```text
outputs/2026-08-23/07-10-17/best.pt
```

## Retrieval

```text
FULL Mean Recall           = 59.0303
REFERENCE_ONLY Mean Recall =  2.1376
Modification gain          = 56.8928
```

## Round 0

```text
state pair cosine          = 0.051616
attention mean pair        = 0.999989
attention worst all        = 0.999997
attention worst active     = 0.999997
claim pair cosine          = 0.999987
```

Interpretation:

- learned slot identities are initially distinct,
- first-round token attention is almost identical.

## Round 1

```text
state pair cosine          = 0.997761
attention mean pair        = 0.992376
attention worst active     = 0.998622
claim pair cosine          = 0.983687
```

Shared evidence/update pushes slot states near a common solution.

## Round 2

```text
state pair cosine          = 0.998848
attention mean pair        = 0.986783
attention worst active     = 0.997921
claim pair cosine          = 0.969449
```

Claims diverge more strongly than raw attention because depletion itself contributes to claim differences.

## Final representations

```text
semantic pair cosine       = 0.996728
effect pair cosine         = 0.962734
edit-slot pair cosine      = 0.991631
active slots               = 4.0
dominant share             = 0.293383
```

Teacher effects show the strongest genuine differentiation. Final edit slots remain highly similar.

---

# 15. Frozen causal intervention results

## KEEP

```text
KEEP S0 = 37.2688
KEEP S1 = 33.2485
KEEP S2 = 29.4875
KEEP S3 = 26.2888
```

Final-round claim mass:

```text
S0 = 1.0000
S1 = 0.8862
S2 = 0.8004
S3 = 0.7301
```

The ordering is strongly aligned with sequential position/depletion mass, so it cannot be taken as clean semantic factor identity.

## REPEAT x4

```text
REPEAT S0 x4 = 59.1888
REPEAT S1 x4 = 59.0711
REPEAT S2 x4 = 58.8241
REPEAT S3 x4 = 58.2838
```

Modification-gain recovery:

```text
S0 x4 = 100.28%
S1 x4 = 100.07%
S2 x4 =  99.64%
S3 x4 =  98.69%
```

## MEAN SLOT

```text
MEAN SLOT x4 = 58.9726
gain recovery = 99.90%
```

This is the decisive failure signature.

---

# 16. Executor / primitive diagnostics

At the best checkpoint:

```text
valid execution steps/sample ~= 3.998
transition strength          ~= 0.99967
```

Primitive routing:

```text
primitive 1 ~= 99.68%
primitive 4 ~=  0.30%
others      ~=  0
```

The downstream Executor has effectively collapsed to one dominant primitive and repeated nearly full-strength updates.

This is separate from slot collapse, but interacts strongly with the slot-as-compute-ticket shortcut.

---

# 17. Final branch verdict

A5.1c is a useful negative result, not a useless run.

Compared with A5.0/A5.1, it creates real teacher-effect differentiation. However the scientific target is not achieved because:

1. first-round raw attention is almost identical,
2. recurrent slot states converge,
3. final semantics remain near-identical,
4. final edit slots remain near-identical,
5. all four slots remain active,
6. repeating one slot reconstructs FULL,
7. repeating the mean slot reconstructs FULL,
8. the primitive bank collapses to one dominant primitive.

Final conclusion:

\[
\boxed{\text{Sequential residual evidence alone is insufficient}}
\]

under:

```text
contextual text values
+ retrieval-only supervision
+ current coupled Executor
```

Do **not** spend many more epochs on this exact branch expecting the shortcut to disappear spontaneously.

---

# 18. What A5.1c falsifies

The branch gives strong evidence against the hypothesis:

> “The only reason slots collapse is that they do not remember already-claimed evidence.”

Residual memory helps, but is not sufficient.

The next working hypothesis is:

> Each contextual token already carries a large sentence-global edit component, so different token attention distributions still pool nearly the same global modification content.

---

# 19. Next branch recommendation

Next experiment:

```text
A5.2 — Localized Claim Values
```

Keep:

```text
contextual states -> KEY / relevance
```

but use:

```text
localized or common-mode-reduced states -> VALUE / content
```

Minimal first variant:

\[
\bar h=\operatorname{maskedMean}(h_n)
\]

\[
h_n^{value}=h_n-\bar h
\]

while keys remain contextual.

Do not redesign the Executor in the same experiment. First isolate whether value localization solves representation collapse.

Only after slot factors become genuinely distinct should Executor compute-depth and primitive specialization be changed.

---

# 20. Quick interpretation checklist

For a future forensic JSON:

```text
[ ] raw attention mean cosine decreased
[ ] raw attention worst-active cosine decreased
[ ] claim cosine decreased for more than mechanical depletion reasons
[ ] state cosine decreased
[ ] semantic cosine decreased
[ ] teacher-effect cosine decreased
[ ] edit-slot cosine decreased
[ ] DROP effects are complementary
[ ] KEEP effects are not explained only by sequential position
[ ] REPEAT one slot x4 no longer ~= FULL
[ ] MEAN SLOT x4 no longer ~= FULL
[ ] retrieval remains competitive
```

The repeat and mean-slot tests are essential.

---

# 21. Branch freeze checklist

```text
[ ] restore masked_mean() in src/diagnose_a5.py on remote branch
[ ] commit runtime fix
[ ] push branch
[ ] py_compile passes
[ ] clean-checkout forensic run passes
[ ] freeze a5_1c_forensic_15ep.json
[ ] record best checkpoint path
[ ] record best Mean Recall = 59.0303
[ ] mark experiment CLOSED / FAILED-INFORMATIVE
[ ] create next branch from intended clean base
```

---

# 22. Reference

Primary mechanism reference:

- Niklas Houba, **When Attention Collapses: Residual Evidence Modeling for Compositional Inference**, arXiv:2605.02323, 2026.

The paper's residual-evidence mechanism is the inspiration for A5.1c, but TAPER differs materially in supervision, existence modeling, representation, and downstream Executor coupling. A5.1c should therefore be cited internally as a **mechanism-transfer experiment**, not an exact reproduction.