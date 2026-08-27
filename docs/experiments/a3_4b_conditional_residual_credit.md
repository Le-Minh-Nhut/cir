# A3.4b — Conditional residual functional credit

## Hypothesis

A3.4 assigned functional jobs from the EMPTY coalition once, then trained those
fixed jobs even when later finite interventions showed that a proposed slot was
redundant. A3.4b changes only that credit schedule. It tests whether recomputing
functional jobs after each accepted coalition creates distinct useful slot
responsibilities.

The deployed Run-C architecture remains unchanged:

- contextual VALUE;
- `slot_effect_in_value=false`;
- `slot_value_assignment=soft_shared`;
- unchanged ownership, QASA, Executor, retrieval objective, and inference
  compute.

## Exact conditional oracle

For four real Edit Slots, training evaluates all 16 Boolean coalitions in one
vectorized auxiliary executor batch. For current coalition `A`, slot `s`, and
hard-negative mode `j`, the detached oracle measures

```text
Phi[s,j | A] = loss_j(q_A) - loss_j(q_(A union {s})).
```

The existing rank gate gives a maximum slot budget. At each step, the existing
NULL-aware, upper-capacity mode assignment is applied to the current positive
conditional `Phi`. One singleton block is accepted. Only when no singleton has
positive work may exact conditional pair synergy admit a pair. Assigned modes
are removed, `A` is updated, and all marginal utilities are recomputed.

The process stops without fabricating owners when no positive conditional
utility remains. Therefore rank-one tasks and giant-only outcomes may use one
block.

Mode eligibility is recomputed at every coalition. `claimed_modes` contains
only modes that an accepted block actually received credit for;
`achievable_now` contains modes with current positive singleton or conditional
pair utility; and `ever_achievable_modes` records their union along the visited
trajectory. Thus a mode that is non-positive at EMPTY may become a residual job
later. Final `unresolved_modes` means `ever_achievable_modes & ~claimed_modes`.
The planner also returns the currently achievable unclaimed set at its stop.

## Functional loss and gradients

For accepted step `t`, with new block `b_t` and assigned residual modes `J_t`:

```text
L_func,t = weighted_mean_j in J_t(
    loss_j(q_(A_t union b_t), live-new-block)
    - stopgrad(loss_j(q_A_t))
)
```

Detached positive conditional gains are the weights. Coalition selection,
rank decisions, assignments, negative mining, and credit weights receive no
gradient.

The forward Edit Slot values are exactly the deployed Run-C values. A
straight-through substitution makes only the newly credited block's isolated
slot path live. Slots already in `A_t` and unrelated slots are detached. Shared
`slot_mlp`, Executor/router/primitive parameters, and query head remain live.

## Config

```yaml
functional_ownership:
  enabled: true
  credit_schedule: conditional_residual
```

`credit_schedule=first_round` preserves the frozen A3.4 v2.1 negative control.
The schedule is checkpoint provenance and mismatches are rejected.

## Diagnostics

The `functional/conditional_*` metrics describe the actual training plan:

- `conditional_steps`: accepted blocks per sample;
- `conditional_credited_slots`: unique slots in accepted blocks;
- `conditional_credited_modes`: unique residual modes credited;
- `conditional_residual_gain`: mean positive conditional gain per credited mode;
- `conditional_stop_no_gain_fraction`: samples stopped with residual modes but
  no useful next block;
- `conditional_clone_rejection_fraction`: initially positive slot-mode edges
  that lose positive utility after the coalition changes;
- `conditional_pair_fraction`: accepted blocks that are pairs.
- `conditional_claimed_modes`: modes actually credited along the plan;
- `conditional_ever_achievable_modes`: modes that had positive utility at any
  visited coalition;
- `conditional_unresolved_modes`: ever-achievable modes never claimed.

Historical A3.4 metrics remain available and describe initial-state assignment
where applicable. They are not evidence that the conditional training plan
succeeded.

## Limitation and falsification

This branch deliberately leaves contextual VALUE and deployed QASA unchanged.
If conditional planning usually stops after one giant slot, or held-out P0 Phi
remains rank-one after training, conditional credit is insufficient. The branch
does not automatically add a representation firewall or selector change.
