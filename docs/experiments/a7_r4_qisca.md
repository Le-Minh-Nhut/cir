# A7 R4: QI-SCA joint sparse token-to-slot assignment

R4 tests whether joint competition for token evidence reduces R1's cross-slot
support overlap and fixed S0 functional dominance. It does not add an overlap,
balance, diversity, or anti-collapse loss.

## Routing modes

```yaml
# R1 control
routing_mode: entmax15

# R4
routing_mode: qisca
```

QASA is unchanged and always measures the pre-sparse slot-softmax competition.
For QI-SCA, that same differentiable tensor is denoted `P^Q`; candidate masking
never renormalizes it.

## Solver candidates

```yaml
r4_candidate_mode: qasa_selected  # existing R4a control
r4_candidate_mode: all_real_slots # capacitated spillover experiment
```

In `qasa_selected`, QASA's hard mask defines the QI-SCA candidate coordinates.
Execution requires both QASA permission and positive routing mass, preserving
the existing R4a behavior.

In `all_real_slots`, QASA is still computed and reported unchanged, but every
real slot may participate in QI-SCA. Execution then follows positive routing
mass, including for a non-QASA-selected slot that receives capacitated
assignment. A zero-evidence slot never receives an Executor step in either
mode. This isolates whether column congestion lets alternative real-slot
utilities survive instead of hard-pruning them before optimization.

QI-SCA forms

```text
U = P^Q - theta
V = U / lambda
```

and computes the Euclidean projection of `V` onto nonnegative assignments with
token mass at most one. The inequality permits partial or complete rejection;
there is no NULL/dustbin slot.

## Capacity toggle

```yaml
r4_capacity_enabled: false  # R4a
r4_capacity_enabled: true   # R4b
```

R4a projects every token's slot vector directly onto the nonnegative
sub-simplex. No slot-column capacity operator appears in this solver path.

R4b adds the fixed shared constraint

```text
sum_tokens A[slot, token] <= r4_slot_capacity
```

`r4_slot_capacity` is fractional assignment mass, not a token count. The shared
default `2.0` is retained only as a provisional smoke-test value; it was not
derived from R4a support cardinality and is not claimed optimal.

R4b uses 64 fixed Dykstra iterations with correction tensors for the token and
slot projection sets. The default 64 iterations matched a 512-iteration
reference on deterministic unit-test problems while keeping the training path
differentiable. The initial capacity value is not claimed optimal; R4a should be
trained and audited before R4b.

## Default R4a configuration

```yaml
routing_mode: qisca
r4_theta: 0.25
r4_lambda: 1.0
r4_capacity_enabled: false
r4_candidate_mode: qasa_selected
r4_slot_capacity: 2.0
r4_solver_iters: 64
```

`theta=0.25` corresponds to uniform competition over four real Edit Slots. It
is not adjusted for QASA-selected count. Theta, lambda, and capacity are fixed,
not learned or annealed.

With `lambda=1` and nonnegative theta, `sum(relu(P^Q - theta)) <= 1`, so the
token budget can be inactive. Preprojection mass, violation/excess, and
postprojection binding diagnostics are logged explicitly; lambda remains 1.0
until a deliberate calibration experiment is performed.

## Pooling semantics

For assignment mass `m` and weighted evidence `w = sum_n A_n x_n`, pooling uses

```text
slot_semantics = w / max(m, eps)
slot_activity  = min(m, 1)
edit_slot      = slot_semantics * slot_activity
```

Thus `0 < m < 1` gives `edit_slot = w`: assignment mass is applied once, not
twice. `m=0` remains exactly zero, while R1's usual `m≈1` behavior is unchanged.

## Commands

```bash
# R1 control
python src/train.py experiment=taper_e2e \
  experiment.model.routing_mode=entmax15

# R4a: joint token competition, rejection, no capacity
python src/train.py experiment=taper_e2e \
  experiment.model.routing_mode=qisca \
  experiment.model.r4_capacity_enabled=false \
  experiment.model.r4_candidate_mode=qasa_selected

# R4b spillover: identical QI-SCA objective, all real candidates, slot capacity
python src/train.py experiment=taper_e2e \
  experiment.model.routing_mode=qisca \
  experiment.model.r4_theta=0.15 \
  experiment.model.r4_lambda=0.45 \
  experiment.model.r4_capacity_enabled=true \
  experiment.model.r4_candidate_mode=all_real_slots \
  experiment.model.r4_slot_capacity=2.0
```

R4 must not be called successful merely because it is sparse. The intended
forensic signals are multiple routed/executed slots under congestion, nonzero
mass on non-QASA candidates, lower fixed-slot functional dominance, useful
non-S0 slot-drop effects, and competitive retrieval. Low overlap alone is not
success: it can be zero because only one slot survived. Equal slot utilization
is not required or forced.

For an initial short smoke run, stop and audit if QASA and routed slot counts
rapidly approach one together, if capacity binding approaches 100% while
unassigned mass becomes extreme, or if a binding dominant slot produces
approximately zero non-QASA routed mass. Those outcomes would not establish
the intended redistribution mechanism.
