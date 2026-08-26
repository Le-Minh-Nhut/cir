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
For QI-SCA, that same differentiable tensor is denoted `P^Q`. QASA's selected
mask only permits a slot to consume evidence; masking never renormalizes `P^Q`.
A slot executes only when it is both QASA-selected and has routing mass greater
than the routing support epsilon. A selected but fully rejected slot receives no
Executor step, primitive, or reference-only transition.

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

and uses 64 fixed Dykstra iterations with correction tensors for the token and
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
  experiment.model.r4_capacity_enabled=false

# R4b: identical QI-SCA objective plus slot capacity
python src/train.py experiment=taper_e2e \
  experiment.model.routing_mode=qisca \
  experiment.model.r4_capacity_enabled=true \
  experiment.model.r4_slot_capacity=2.0
```

R4 must not be called successful merely because it is sparse. The intended
forensic signals are lower support overlap and S0 winner dominance, more useful
non-S0 slot-drop effects, and competitive retrieval. Equal slot utilization is
not required or forced.
