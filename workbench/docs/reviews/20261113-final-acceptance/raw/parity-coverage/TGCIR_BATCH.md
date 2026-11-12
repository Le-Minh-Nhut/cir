# TG-CIR batch sensitivity — a measurement, stated as one

Question: tgcir's three `batch_1_vs_n.max_rel` are 0.24029609477008945 / 0.07048293904534349 /
0.029507544996198128 against a declared `batch_max_rel` of 1e-3 — two to three orders of magnitude
worse than every other model's batch sensitivity. Is this the same near-zero-denominator artefact the
campaign already records (LEDGER L10), or a genuinely batch-sensitive computation?

Sources, all on disk: `g1_g2.tgcir.{dress,shirt,toptee}.adapter.json` (the `layer2_4_forward`
block), `workbench/third_party/TGCIR/model.py`, `workbench/composition/model_tgcir.py`.

## 1. What the failing quantity actually is

| category | `max_abs` | `max_rel` | implied |denominator| at the worst component | declared `batch_max_abs` (atol) | atol satisfied? | `min_cosine` | shape |
|---|---|---|---|---|---|---|---|
| dress | 2.898275852203369e-06 | 0.24029609477008945 | **1.206e-05** | 1e-4 | **yes**, by 34× | 0.9999999997674346 | [5, 512] |
| shirt | 2.4028122425079346e-06 | 0.07048293904534349 | **3.409e-05** | 1e-4 | **yes**, by 42× | 0.9999999998152298 | [5, 512] |
| toptee | 2.3506581783294678e-06 | 0.029507544996198128 | **7.966e-05** | 1e-4 | **yes**, by 42× | 0.9999999997937318 | [5, 512] |

The implied denominator is `max_abs / max_rel` at the worst component, and the column is the point:
the *worst* relative error in dress is a **1.2e-05**-magnitude component carrying a **2.9e-06** absolute
difference. Two components that tiny differ in their last float32 digits; dividing by 1.2e-05 turns
3e-06 into 0.24.

**The absolute deviation is 2.9e-06 against a declared atol of 1e-4 — inside it by a factor of 34.**
The vectors are 512-d unit-norm CLIP-space embeddings, so `min_cosine` 0.9999999998 is the same
statement in the metric that matters for retrieval: the two vectors point the same way to 10 decimal
places.

## 2. Why `batch_max_rel` is nevertheless the "right" declared relative term, and why it fails

The declared table is shared: `tolerances` comes from `t4_parity.TOL`, one table for every model
(`batch_max_abs` 1e-4, `batch_max_rel` 1e-3). It was not chosen for tgcir, and this lane did not
choose it. So the honest answer to "is tgcir's declared `batch_max_rel` appropriate for its scoring"
is: **it is the campaign's shared tolerance, not a per-model one, and it is calibrated for models whose
worst component is not 1e-5.**

`max_rel` divides each component by the reference component. For a unit-norm 512-d embedding, the
smallest components are ~1e-5 by construction — a 512-d unit vector has 512 components averaging
`1/√512 ≈ 0.044` in magnitude, so the tail sits two decades lower. The relative term therefore reports,
for every model in this family, the reciprocal of the smallest component rather than any disagreement.
HABIT shows the same effect at smaller magnitude — `max_rel` 5.096e-03 / 6.168e-03 over implied
denominators **3.801e-05 / 2.642e-05** (from its own `max_abs` 1.937e-07 / 1.630e-07); ENCODER at
3.2e-3; LIMN at 2.7e-3–1.1e-2. Note HABIT's `max_abs` is an order of magnitude SMALLER than tgcir's
(1.9e-07 vs 2.9e-06) while its `max_rel` is 50× smaller: the two numbers move together, which is what a
denominator-driven reading predicts. tgcir's numbers are larger only because its smallest
components are smaller: its query vector comes from a **mean over tokens** and is then renormalised
(`model.py:203`), which compresses more components toward zero than HINT's max-over-32-tokens layout.

## 3. Is the computation genuinely batch-dependent?

**No — and this is checkable in the pinned code, not inferred.** Every reduction in the scoring path
is over the TOKEN axis (`dim=1`), never the batch axis (`dim=0`):

```
model.py:201  def extract_retrieval_compose(self, ref, mod):
model.py:202      s_fuse_local, _, _, _, _ = self.s_compose_feature(ref, mod)
model.py:203      s_fuse_local = F.normalize(torch.mean(s_fuse_local, dim=1), p=2, dim=-1)
model.py:207  def extract_retrieval_target(self, tag):
model.py:208      tag_local = self.backbone.extract_img_fea(tag)
model.py:209      tag_local = F.normalize(torch.mean(tag_local, dim=1), p=2, dim=-1)
```

`grep -n "dim=0\|batchmean\|keepdim" workbench/third_party/TGCIR/model.py` returns exactly one hit —
`model.py:266 F.kl_div(..., reduction='batchmean')` — and that is inside `compute_loss`, which the
evaluation path never calls. There is **no** BatchNorm over the batch dimension, no batch statistic, no
cross-sample normalisation. Per-image output is therefore a pure function of that image.

What differs between batch 1 and batch N is **float32 reduction order**: cuDNN/BLAS choose different
tiling and accumulation order for a matrix multiply of shape `(1, ...)` versus `(N, ...)`, so the last
bits of each product differ. That is the same mechanism recorded for ENCODER, HINT, LIMN and HABIT —
`batch_1_vs_n` is a *legitimately different reduction order* comparison per the gate's own
documentation (`parity_gate.py`: "batch_parity — G2: batch size 1 vs N. Legitimately different
reduction order").

Corroboration from the same artifacts: `bitwise_equal: false`, `min_cosine ≥ 0.9999999997`, and
`component_parity` reports `sample_forward.query` and `sample_forward.target` at
`coverage: "full"`, `components_total: 2560`, `components_emitted: 2560` — the forward comparison is
complete and clean. Only the batch comparison fails, and only on its relative term.

## 4. Conclusion, stated as a measurement

> tgcir's `batch_1_vs_n` differs from bitwise equality by at most **2.9e-06 absolute** (dress), against
> a declared absolute tolerance of **1e-4** — inside it by 34–42×, with `min_cosine ≥ 0.9999999997`.
> The failing `max_rel` of 0.24 / 0.07 / 0.03 is the reciprocal of a **1.2e-05**-magnitude component,
> not a 24% disagreement. The computation has no batch-axis reduction anywhere in its scoring path
> (`model.py:203,209`, both `dim=1`), so the difference is float32 reduction order under a different
> batch shape — the same class as ENCODER/HINT/LIMN/HABIT, larger only because a token-mean followed by
> renormalisation pushes more components into the sub-1e-4 tail.

This is **not** an anomaly specific to tgcir, and it does not indicate a tgcir defect.

## 5. What would settle it — and what is out of scope

**Not proposed:** moving `batch_max_rel`, or introducing a floored denominator. Either changes a
declared tolerance after seeing the number, which the campaign forbids outright (LEDGER L10's state:
"never moving the tolerance after reading the numbers").

What would settle it, in increasing cost:

1. **`max_abs`-only scopes as an explicit recorded decision** (LEDGER L10's first option). It is a
   *decision about the standard*, made once for all models, recorded with its rationale — not a
   per-model tweak. **Out of this lane's scope**: it changes what every artifact is judged against, and
   the decision belongs to the acceptance owner. It would move encoder's 9, hint's 2, limn's 3,
   habit's 2 and tgcir's 3 in one stroke.
2. **A floored-denominator `max_rel`** — `max(|a-b| / max(|b|, floor))` with `floor` declared before
   measurement. Also a standard-level change, also the acceptance owner's.
3. **Measure whether the reduction order is the whole story**, per model, by repeating the batch-N pass
   at the same N and confirming bitwise equality (which `cache_off_restart` already tests: for tgcir it
   is declared **not measured**, `restart_supported: False`, because the adapter refuses a second
   worker — `model_tgcir.py:344 _check_worker_pid`). So for tgcir specifically the cleanest additional
   evidence is **two independent batch-N passes with different batch sizes** and the same
   components-compared count: if `max_abs` stays ~1e-6 while `max_rel` moves, the denominator is
   confirmed as the driver. That is a ~11-second GPU job (adapter phase), well inside budget, and
   **read-only to the tree** — but it writes a new artifact, which this round must not do.

Of those, (3) is the only step that is measurement rather than policy, and it is the natural next one.

## 6. Coverage: this pair is `chunked`/INCOMPLETE — its verdict is not a full-population claim

`population_scope: "sample"` is recorded in the tgcir artifacts, so:

| aspect | state |
|---|---|
| `component_parity` (forward), `sample_forward.query` / `.target` | `coverage: "full"` over **2560** components each — complete **for the 5-query sample** |
| ranking / population block | `coverage: "chunked"`, `population_scope: "sample"` — a **preview** |
| `metrics_parity` | **`missing`**, not passed: no stored native per-query-rank control exists for tgcir, so the adapter's ranks cannot be compared against a native *measurement* |
| `scoring_parity` | measured on the adapter-vs-convention comparison (adapter `rank()` vs numpy `argsort(-sims)` over the same features), which is a real comparison of two code paths, not a self-comparison |

Consequence, stated plainly: **tgcir's gate verdict is a statement about a 5-query / 20-image fixed
sample, not about the 2017/2038/1961-query FashionIQ populations.** A `coverage: chunked` preview
cannot earn `ok` under `parity-absolute-relative/v2`, and the producer labels it that way on purpose.
The sample's own properties are the reason it is worth having anyway: two rows sharing one reference (a
per-query max-over-targets metric must get that right), the last row (the only one a partial final
batch reaches), a capitalised first caption, a sub-12-character second caption, and a 20-image gallery
chosen so the image batch is itself partial.

## 7. One-line answer to the question asked

Same class of artefact as elsewhere, quantified: **2.9e-06 absolute against a 1e-4 atol, failing only
because the relative term divides by a 1.2e-05 component**; the computation has no batch-axis reduction
(`model.py:203,209`, `dim=1` only), so it is reduction order rather than batch statistics; the
settling measurement (two batch-N passes at different N) is a ~11 s GPU job that this round does not
run, and the tolerance change that would "fix" it is a standard-level decision outside this lane.
