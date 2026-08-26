# A3.1 — QASA-Faithful Hard-Partition Evaluation

**Branch:** `exp/e2e-a3.1-qasa-slot-filter-eval-winner`  
**Parent experiment:** `exp/e2e-a3.1-qasa-slot-filter`  
**Purpose:** Evaluate the learned Edit-Slot attention using the same hard token-to-slot inference principle used by QASA.  
**Training:** unchanged from A3.1  
**Checkpoint:** reuse the trained A3.1 `best.pt`  
**Executor during this evaluation:** not used  
**Retrieval during this evaluation:** not used  
**QASA Quality / Coverage / Novelty mask during this evaluation:** not used

---

# 1. Why this branch exists

The previous A3.1 experiment trained TAPER with QASA-style slot selection:

```text
Edit-Slot attention
        ↓
QASA Quality
        ↓
Novelty / Coverage
        ↓
selected slots
        ↓
Router / Executor
        ↓
retrieval objective
```

A3.1 obtained reasonably strong retrieval performance, but its internal diagnostics suggested severe lack of slot specialization.

The key question left open was:

> If the learned slot attention is evaluated using the QASA inference rule itself, does it actually produce a meaningful adaptive partition of the text into multiple Edit Slots?

This branch answers only that question.

It does not introduce a new training mechanism.

---

# 2. Evaluation protocol

The evaluation uses the learned QASA attention:

\[
A \in \mathbb{R}^{B\times L\times N},
\]

where:

- \(B\) = batch size;
- \(L\) = number of Edit Slots;
- \(N\) = number of text tokens.

For every valid content token \(t\), the winning slot is:

\[
w_t=\arg\max_i A_{t,i}.
\]

The hard region of slot \(i\) is:

\[
S_i=\{t\mid w_t=i\}.
\]

Therefore each valid token is assigned to exactly one slot.

The effective number of slots for one sample is:

\[
K_{\text{eff}}
=
\left|
\left\{
i:
|S_i|>0
\right\}
\right|.
\]

The evaluation path is therefore:

```text
qasa_attention
      ↓
argmax over slots for each token
      ↓
hard token partition
      ↓
effective K + partition statistics
```

---

# 3. What this branch deliberately does NOT do

This branch does not use:

```text
qasa_selected_mask
QASA Quality ranking
QASA Coverage threshold
QASA Novelty filtering
Router
Executor
retrieval query
retrieval metrics
```

for the headline evaluation.

This is important because the objective is to inspect the learned decomposition itself rather than asking whether a particular TAPER execution policy can still obtain retrieval performance.

The training checkpoint remains the same.

Only the evaluation interpretation changes.

---

# 4. Hard-partition implementation

The core implementation is:

```python
winner = attention.argmax(dim=1)  # [B, N]

hard_regions = F.one_hot(
    winner,
    num_classes=num_slots,
).permute(0, 2, 1).to(torch.bool)

hard_regions = hard_regions & valid[:, None, :]
```

Invalid, special, and padding tokens are excluded.

A slot is considered present only when it wins at least one valid token:

```python
nonempty_slots = hard_regions.any(dim=2)

effective_k = nonempty_slots.sum(dim=1)
```

The implementation also verifies that:

```text
every valid token belongs to exactly one slot
every invalid token belongs to no slot
effective K never exceeds the configured number of slots
```

---

# 5. Metrics

The evaluator reports the following quantities.

## 5.1 Mean effective K

\[
\mathbb E[K_{\text{eff}}].
\]

This directly answers:

> How many Edit Slots actually receive at least one token?

---

## 5.2 K distribution

For \(L=4\):

```text
P(K=1)
P(K=2)
P(K=3)
P(K=4)
```

This reveals whether the model truly uses a variable number of slots or collapses toward a single hard region.

---

## 5.3 Dominant hard token share

For one sample:

\[
D=
\max_i
\frac{|S_i|}
{N_{\text{valid}}}.
\]

If:

\[
D\approx1,
\]

one slot wins almost all content tokens.

---

## 5.4 Hard winner entropy

Let:

\[
p_i=\frac{|S_i|}{N_{\text{valid}}}.
\]

The normalized entropy is:

\[
H=
-\frac{\sum_i p_i\log p_i}
{\log L}.
\]

Interpretation:

```text
H ≈ 1
→ hard winners distributed across slots

H ≈ 0
→ hard winners concentrated in one slot
```

---

## 5.5 Soft top-1 probability

For each valid token:

\[
p_{\max}(t)=\max_i A_{t,i}.
\]

The evaluator reports its mean.

This helps distinguish:

```text
confident hard winner
```

from:

```text
argmax winner created from an almost-uniform soft distribution
```

---

## 5.6 Top-1 / Top-2 margin

For each token:

\[
m_t
=
A_{t,(1)}
-
A_{t,(2)}.
\]

This measures how strongly the winning slot beats the runner-up.

---

## 5.7 Near-tie fraction

A token is counted as near-tied when:

\[
A_{t,(1)}-A_{t,(2)}
\le0.01.
\]

This checks whether a hard winner monopoly could simply be an artifact of nearly exact numerical ties.

---

## 5.8 Per-slot statistics

For each slot:

```text
winner tokens / sample
non-empty rate
hard token share
```

These reveal whether the collapse is symmetric or whether one particular slot monopolizes the hard partition.

---

# 6. Full validation result

The evaluator was run on the full FashionIQ validation split.

```text
Samples:              6016
Valid content tokens: 73137
```

Headline result:

```text
Mean effective K: 1.1749
```

Therefore, despite having four Edit Slots, the learned hard partition uses only slightly more than one slot on average.

---

# 7. Effective-K distribution

Measured:

| Effective K | Fraction |
|---:|---:|
| 0 | 0.00% |
| 1 | **82.55%** |
| 2 | 17.42% |
| 3 | 0.03% |
| 4 | **0.00%** |

Thus:

\[
\boxed{
82.55\%
\text{ of validation samples contain only one non-empty hard slot}
}
\]

and:

\[
\boxed{
K=4
\text{ never occurs}
}
\]

on the evaluated validation set.

This is already strong evidence of hard winner collapse.

---

# 8. Dominant hard region

Measured:

\[
\boxed{
\text{Mean dominant hard share}=0.9761
}
\]

Therefore the dominant slot receives, on average:

\[
\boxed{97.61\%}
\]

of all valid content tokens.

In plain terms, the typical learned partition is approximately:

```text
dominant slot  → almost every token
other slots    → zero or a very small number of tokens
```

rather than:

```text
slot A → one edit component
slot B → another edit component
slot C → another edit component
slot D → another edit component
```

---

# 9. Hard winner entropy

Measured:

\[
\boxed{
H_{\text{hard}}=0.0469
}
\]

The normalized entropy is extremely close to zero.

This independently confirms that hard token assignments are highly concentrated.

Thus the effective-K failure is not merely caused by one slot winning one extra token.

The full token distribution itself is almost monopolistic.

---

# 10. Which slot collapses the system?

Per-slot result:

| Slot | Winner tokens / sample | Non-empty rate | Hard token share |
|---|---:|---:|---:|
| S0 | 0.000 | 0.05% | 0.0000 |
| S1 | 0.320 | 17.44% | 0.0238 |
| S2 | 0.000 | 0.00% | 0.0000 |
| **S3** | **11.837** | **100.00%** | **0.9761** |

The failure is therefore highly asymmetric.

The observed structure is approximately:

```text
                 ┌─ S0: ~0%
text tokens ─────┼─ S1: ~2.38%
                 ├─ S2: 0%
                 └─ S3: ~97.61%
```

Most importantly:

\[
\boxed{
S3\text{ is non-empty in }100\%\text{ of samples}
}
\]

while:

\[
S2
\]

never wins a token.

This is a near-complete hard winner monopoly.

---

# 11. Is the monopoly just an argmax tie artifact?

No strong evidence supports that explanation.

Measured:

```text
Mean soft top1 probability = 0.4332
Mean top1-top2 margin      = 0.1600
Near-tie fraction <= 0.01  = 1.49%
```

If the attention were essentially tied:

```text
S0 = 0.251
S1 = 0.249
S2 = 0.250
S3 = 0.250
```

then hard `argmax` could create misleading winner assignments due to tiny numerical differences.

That is not what the aggregate statistics show.

The average winning probability is:

\[
0.4332
\]

and the average winner margin is:

\[
0.1600.
\]

Only:

\[
1.49\%
\]

of valid tokens have a top-1/top-2 gap below or equal to `0.01`.

Therefore the S3 monopoly is not adequately explained as numerical tie-breaking.

The learned attention contains a real and systematic ranking advantage for the dominant slot.

---

# 12. Category-level consistency

Measured:

| Category | Samples | Mean K | Dominant hard share |
|---|---:|---:|---:|
| dress | 2017 | 1.1651 | 0.9783 |
| shirt | 2038 | 1.1806 | 0.9746 |
| toptee | 1961 | 1.1790 | 0.9755 |

The same failure appears across all three FashionIQ categories.

Thus the collapse is not isolated to one clothing class.

The behavior is dataset-wide.

---

# 13. Agreement with the previous forensic diagnostic

The previous A3.1 diagnostic measured:

```text
ownership/winner_active_slot_count ≈ 1.1810
```

The new QASA-faithful evaluator measures:

```text
mean effective K = 1.1749
```

These two independently computed quantities are extremely close.

This agreement strongly increases confidence that the collapse is real rather than a bug in one evaluator.

---

# 14. Why this matters for the previous QASA metrics

During A3.1 training, the model often reported approximately:

```text
QASA selected K ≈ 2
QASA coverage   ≈ 1
```

At first glance, this could appear to suggest that multiple useful slots exist.

The QASA-faithful hard partition now shows:

```text
effective K ≈ 1.17
dominant hard share ≈ 97.61%
```

These observations are not contradictory.

They measure different things.

---

## 14.1 QASA training selection uses soft coverage

With four nearly uniform slots:

\[
A_{t,i}=0.25.
\]

For:

\[
\tau=0.5,
\]

two slots already satisfy:

\[
0.25+0.25=0.5.
\]

Therefore two diffuse slots can jointly mark every token as covered even when no meaningful decomposition exists.

---

## 14.2 QASA inference uses hard winners

The hard partition asks instead:

\[
\arg\max_i A_{t,i}.
\]

This exposes which slot actually wins each token.

The result is:

\[
S3
\]

winning almost all tokens.

Therefore:

\[
\boxed{
\text{high soft coverage can coexist with severe hard winner collapse}
}
\]

and, in this experiment:

\[
\boxed{
\text{soft coverage masked the true hard assignment failure}
}
\]

---

# 15. Relation to the previous causal diagnosis

The QASA-faithful evaluation is not the only evidence of failure.

The previous A3.1 forensic analysis reported:

```text
ownership pairwise cosine             = 0.9806
ownership pairwise JS                 = 0.0070
slot-effect pairwise cosine           = 0.7795
slot-effect effective rank            = 1.2777
forced-only causal cosine             = 0.9936
drop-direction causal cosine          = 0.9820
Executor state-change cosine          = 0.9890
Slot ↔ Primitive NMI                  = 0.0075
```

Together with the new hard-partition result:

```text
Mean effective K                      = 1.1749
K=1 fraction                          = 82.55%
Dominant hard token share             = 97.61%
```

the evidence now spans four levels:

```text
SOFT OWNERSHIP
different slots have very similar token maps
        ↓
HARD QASA INFERENCE
S3 wins almost every token
        ↓
REPRESENTATION / CAUSAL EFFECT
different slots produce nearly the same edit direction
        ↓
EXECUTION
different slots create nearly the same state transition
```

Therefore the decomposition failure is not localized to one diagnostic layer.

---

# 16. Final scientific interpretation

The branch strongly rejects the hypothesis:

\[
\boxed{
\text{QASA-style selection alone is sufficient to create TAPER Edit-Slot specialization}
}
\]

The observed behavior is instead:

```text
multiple nominal slots
       ↓
highly overlapping soft attention
       ↓
one slot gains a systematic winner advantage
       ↓
hard inference collapses to almost one region
       ↓
remaining slot functions are still causally redundant
```

The most precise description is:

\[
\boxed{
\text{diffuse soft ownership}
+
\text{near-complete hard winner collapse}
+
\text{functional redundancy}
}
\]

---

# 17. What this branch proves

This branch provides strong evidence that, for the current TAPER adaptation:

1. QASA-style hard inference does not reveal a hidden clean decomposition.
2. More than 82% of validation samples produce only one non-empty hard slot.
3. The dominant slot receives approximately 97.6% of hard token assignments.
4. The collapse is consistent across dress, shirt, and toptee.
5. The collapse is not adequately explained by near-tied attention.
6. The result agrees with an independent forensic diagnostic.
7. Soft QASA coverage is not a reliable indicator of semantic Edit-Slot decomposition in this setting.

---

# 18. What this branch does NOT prove

This branch does not prove that QASA itself is generally invalid.

It only establishes failure for the current adaptation:

```text
TAPER
+
text-token Edit Slots
+
one-shot slot competition
+
retrieval supervision
+
QASA-style training selection
```

QASA was originally developed in a different object-centric setting.

Therefore the scientifically correct conclusion is:

> QASA selection does not supply enough specialization pressure for the current TAPER Edit Slots.

The result should not be generalized beyond that scope.

---

# 19. Main research implication

The central problem is no longer:

> How should we select fewer slots?

The experiment shows that the more fundamental problem occurs earlier:

> Why should different slots learn different causal roles at all?

The current system effectively performs:

```text
SELECT
  ↓
hope specialization appears
```

but the evidence suggests the research sequence should instead be:

```text
SPECIALIZE
    ↓
verify distinct causal roles
    ↓
SELECT / PRUNE
```

In short:

\[
\boxed{
\textbf{specialize first, select later}
}
\]

---

# 20. Implication for future experiments

Future work should focus on mechanisms that create distinct causal responsibility rather than additional slot-selection heuristics.

Possible families include:

```text
exclusive responsibility
residual / unexplained evidence
sequential explaining-away
slot-specific causal contribution
adaptive anti-redundancy
structured bottlenecks
role-conditioned execution
```

However, any proposed mechanism must be audited against trivial solutions.

For example:

```text
orthogonal embeddings
```

do not automatically imply:

```text
different semantic roles
```

and:

```text
different attention maps
```

do not automatically imply:

```text
different causal functions
```

Therefore future experiments should retain the causal probes introduced in A3.1.

---

# 21. Recommended success criteria for later branches

A future specialization mechanism should ideally improve several independent diagnostics simultaneously.

At minimum:

```text
hard effective K increases when the caption contains multiple edit factors
dominant hard share decreases substantially
ownership maps become less redundant
slot-effect effective rank increases
forced-only causal cosine decreases
Executor state-change cosine decreases
Slot↔Primitive dependence becomes non-trivial when semantically justified
retrieval performance does not catastrophically degrade
```

No single metric should be used as proof of specialization.

---

# 22. Running the evaluator

Quick evaluation:

```bash
python src/evaluate_qasa_inference.py \
  --checkpoint /path/to/A3.1/best.pt \
  --max-queries-per-category 512
```

Full validation:

```bash
python src/evaluate_qasa_inference.py \
  --checkpoint /path/to/A3.1/best.pt \
  --max-queries-per-category 0
```

The report is saved to:

```text
reports/qasa_faithful_inference_eval.json
```

---

# 23. Final verdict

\[
\boxed{
\textbf{A3.1 QASA-faithful inference confirms severe hard winner collapse}
}
\]

Numerically:

\[
\boxed{
\mathbb E[K_{\text{eff}}]=1.1749
}
\]

\[
\boxed{
P(K=1)=82.55\%
}
\]

\[
\boxed{
\text{dominant hard token share}=97.61\%
}
\]

and:

\[
\boxed{
S3\text{ is non-empty in }100\%\text{ of validation samples}
}
\]

The final conclusion is therefore:

> **QASA-style slot selection can prune TAPER Edit Slots, but the learned slots do not self-organize into distinct semantic or causal roles. The QASA-faithful hard partition instead exposes a near-complete winner collapse dominated by S3.**

This branch should be retained as a clean negative-control result demonstrating that:

\[
\boxed{
\text{selection} \neq \text{specialization}
}
\]

and that high soft coverage can hide severe hard assignment collapse.