# Scientific-integrity review

**Overall verdict: UNSOUND** as a fail-closed scientific acceptance gate. Most sampled benchmark values are faithfully transcribed or correctly marked workbench-derived. However, gate can return a plain pass with published category results unchecked, and ENCODER checkpoint mapping is asserted more strongly than evidence supports. These affect what a passing reproduction verdict means.

## Findings

### P1 — Gate can pass while every paper-published category metric is unchecked

`paper_gate.py` says per-category values prevent an aggregate from hiding a broken category (`workbench/backend/paper_gate.py:31-38`). But when reference publishes a category and run omits it, gate records it in `category_unchecked` and continues (`:351-359`); a pass remains `PASS_WITHIN_5_PERCENT`, with missing categories mentioned only in `detail` (`:364-388`). Final audit records concrete probe: HINT run with no category metrics and an 8-point category deviation could still pass on an aggregate within tolerance (`workbench/docs/reviews/final_audit/models/hint.md:166-173`). CONSOLIDATION rejects this concern as frozen policy (`workbench/docs/reviews/final_audit/CONSOLIDATION.md:107-108`).

This is transparent in detail, not a silent fabricated match, but it is not fail-closed for the six category metrics the reference actually publishes. Do not issue the same plain `PASS` when those expected measurements are absent; require them or return an explicitly incomplete/non-passing verdict. The rejection should be revisited.

### P1 — ENCODER checkpoint identity overclaimed; disputed disposition should be fixed

The audit's request to leave ENCODER mapping `UNVERIFIED` is scientifically supported. Registry records an unresolved Drive **folder** URL, no direct download URL, no expected size/hash, and a note that direct URL/hash remain unverified (`workbench/registry/models.yaml:451-464`). Yet mapping is `VERIFIED_METADATA`, justified by the folder plus a purported single `fashioniq.pt` (`:465-471`). Registry documentation defines mapping as evidence establishing *which benchmark variant* the upstream asset is, distinct from source availability (`workbench/backend/registry.py:29-35`). Audit reports the seed bundle itself said `UNVERIFIED`, with no observed file tying a Drive artifact to the paper's FashionIQ result (`workbench/docs/reviews/final_audit/models/encoder.md:144-145`).

`CONSOLIDATION.md:98-102` rejects this because a complete local install would otherwise be refused. Local availability cannot establish scientific identity. Keep mapping unverified until an author-linked artifact/metadata actually binds the file to the benchmark variant. As committed, status can let an unidentified local file be treated as mapped.

### P2 — Null-noise handling is explicitly caveated, but not strict fail-closed

Gate does **not** coerce null to zero. If reference says noise `0` and run omits noise, blocker is hard and comparison stops (`workbench/backend/paper_gate.py:241-258,324-337`; test `workbench/tests/test_paper_gate.py:268-272`). That direction fails closed.

Opposite direction differs: if paper/reference leaves noise null, gate records “not stated … not checked” and does not make it a hard blocker (`paper_gate.py:249-258,324-388`). A metric pass can therefore coexist with unknown or differing run noise; audit confirms unknown-condition HINT selection and records paper-noise caveat (`workbench/docs/reviews/final_audit/models/hint.md:171-173`). This does not claim null equals zero, and the caveat is preserved in result detail/evidence, so it is more honest than a null-to-zero join. But it is not fail-closed for condition-level reproduction. Either describe such verdicts as metric-only/incomplete, or hard-block if “fail-closed” is intended to mean full experiment comparability.

### P2 — CSMCIR’s reported average cells do not match its printed category cells

The row transcribes the paper faithfully: `57.07 / 77.27 / 67.17` and category cells `52.45/74.81`, `57.70/75.76`, `61.14/80.98` (`workbench/registry/paper_benchmarks/papers.yaml:813-835`). But category macro calculations are R@10 = `57.0967` and R@50 = `77.1833`, not the printed `57.07` and `77.27`. Extracted evidence acknowledges the R@10 discrepancy but describes it as `0.027` pp and omits the larger R@50 discrepancy (`workbench/registry/paper_benchmarks/extracted/csmcir_acl26.json:4-5,13-25,39-44,67-68`). The author training log does match the reported average cells, per the audit (`workbench/docs/reviews/final_audit/models/csmcir.md:138-143`).

Numbers are traceable to the paper, not workbench fabrications; paper’s category and average cells are internally inconsistent. Record both discrepancies clearly. Gate’s category/aggregate checks must not imply all paper cells mutually corroborate.

### P2 — Air-Know 0% verification arithmetic does not justify printed AVG under normal rounding

Paper reports `R@10 55.75`, `R@50 75.72`, `AVG 65.73`, and row/category provenance points to Table 1 p.6 (`workbench/registry/paper_benchmarks/papers.yaml:10-68`). Extracted evidence confirms the cells from PDF, HTML and author image (`workbench/registry/paper_benchmarks/extracted/airknow.json:45-66`). However, `(55.75 + 75.72) / 2 = 65.735`; round-half-up to two decimals is **65.74**, not 65.73. The committed metric is still honest as `PAPER_REPORTED` because paper visibly prints 65.73; the arithmetic sentence in verification evidence is not a valid two-decimal rounding proof. Keep source value, annotate the paper’s rounding/precision discrepancy rather than implying half-up yields it.

## Spot-checks: provenance and verification

- **Air-Know 0%:** reported values `55.75 / 75.72 / 65.73`, categories, table/page/row, and independent paper+author routes are specified (`papers.yaml:31-68`; extracted `airknow.json:45-66`). Faithful paper transcription; see rounding finding above. Air-Know 20% row is also transcribed as printed (`papers.yaml:90-115`); its evidence appropriately notes printed `R@50 75.71` differs from category-cell macro `75.7167` (`airknow.json:123-143`). Keeping paper values as reported is correct.
- **CLIP4Cir RN-50x4 fullft:** Table 9 row, column order and three routes identify reported `43.78 / 66.93` plus all six category values (`papers.yaml:242-289`; extracted `clip4cir_tomm2023.json:45-66`). Mean is correctly null as paper-reported and derived as `(43.78+66.93)/2 = 55.355 → 55.36` (`papers.yaml:246-254`).
- **DCNet:** Table 1 row evidence gives category cells and printed `Total Avg 40.83`; reported metrics correctly retain only that printed mean (`papers.yaml:1468-1515`; extracted `legacy_cir_fashioniq_bundle.json:87,90-91`). Derived macro R@10 is `(28.95+23.95+30.44)/3 = 27.78`; R@50 is `(56.07+47.30+58.29)/3 = 53.8867 → 53.89` (`papers.yaml:1472-1490`). Both derivations arithmetically check; labeling them workbench-derived is honest.
- **HINT:** Table 1, p.4, row and three independent routes support paper-reported `58.48 / 77.93`; paper does not print scalar mean (`papers.yaml:1061-1100`; extracted `hint_icassp26.json:45-67`). Derived mean `(58.48+77.93)/2 = 68.205 → 68.21` is correctly marked `WORKBENCH_DERIVED` (`papers.yaml:1065-1073`).
- **PTHA FashionIQ_MTST:** Table III p.7, row label, category values and three visual/text/table routes support reported `54.99 / 76.28 / 65.64` (`papers.yaml:1684-1722`; extracted `ptha_mtst_scaleup_mtst.json:33-55`). Category means and reported Avg arithmetic check; paper-reported metrics are not mislabeled derived.

### Derived-metric arithmetic audit

All nine formulas committed in `papers.yaml` check under the stated round-half-up policy (except TG-CIR’s intentionally retained exact three-decimal result): CLIP4Cir `55.355→55.36` (`:246-254`); CLVC-Net `44.555→44.56` (`:557-565`); HINT `68.205→68.21` (`:1065-1073`); ENCODER `66.86` (`:1126-1134`); PAIR `63.635→63.64` (`:1191-1199`); DCNet R@10 `27.78` and R@50 `53.8867→53.89` (`:1472-1490`); TG-CIR `62.205` (`:1546-1555`); LIMN `66.865→66.87` (`:1616-1624`). No derived-metric value in these records is mislabeled paper-reported; runtime derives provenance from separate reported and derived fields (`workbench/backend/benchmark_references.py:27-31,60-77`).

## Disposition review

- **Wrong disposition: ENCODER C4 disputed/rejected.** Retaining `VERIFIED_METADATA` for convenience does not satisfy evidence that binds bytes to the paper’s model. Fix mapping state (`CONSOLIDATION.md:98-102`; evidence above).
- **Wrong disposition: HINT category-unchecked concern rejected as frozen policy.** Current status can say pass while every published category remains unchecked (`CONSOLIDATION.md:107-108`; `paper_gate.py:351-388`). Require complete category comparison for an unqualified scientific pass, or explicitly downgrade it.
- **Defensible refusal, not closure: PTHA C1 ambiguous references.** Both rows share selector-visible conditions; selecting either would bind same run to distinct pretraining conditions. Refusing ambiguity is right; selector needs a paper-backed pretraining discriminator before a PTHA run can be judged (`CONSOLIDATION.md:109-112`; `ptha_mtst.md:116-126`).
- **Defensible: LIMN forged/inconsistent aggregate rejection.** Fabricated or inconsistent declared aggregate must not be accepted; adversarial tests encode that policy (`CONSOLIDATION.md:103-104`; `metrics_extraction.py:186-198`; `test_adversarial_metrics.py:30-36,91-95`). It loses a metric verdict, but does not falsely certify it.
- **Defensible: TG-CIR parser deferred because no command exists** (`CONSOLIDATION.md:113-114`). **Defensible: Combiner noft row deferred** pending primary-source verification rather than importing secondhand numbers (`:116-120`).
- **Confirmed fixes supported:** removing unsupported zero-noise claims for LIMN, HINT, ENCODER, TG-CIR and PTHA is consistent with extracted evidence that papers did not state training noise (`CONSOLIDATION.md:36-40,75-76`; rows such as `papers.yaml:1047-1049,1108-1110,1527-1529,1596-1598,1667-1669`). Paper-identity corrections are now present for ENCODER/PAIR (`papers.yaml:1101-1104,1164-1167`).

## Untraceable numbers

No sampled benchmark metric lacked either a paper-cell citation or explicit derivation. Two qualifications remain: CSMCIR’s category/average disagreement is not fully documented, and Air-Know 0% verification’s rounding statement is arithmetically unsound despite the paper-reported 65.73 being traceable. Also, some non-noise-benchmark rows encode `checkpoint_training_noise_pct: 0` as a workbench interpretation rather than a paper-stated condition (e.g. CSMCIR `papers.yaml:793-802`, CLIP4Cir `:219-230`; CLIP4Cir extraction describes “clean” but offers no paper noise-protocol excerpt at `extracted/clip4cir_tomm2023.json:21-26`). Those condition values should be identified as inferred/assumed, not presented as directly paper-reported.

**Verdict rationale:** benchmark transcription and derived arithmetic are mostly SOUND_WITH_RESERVATIONS; end-to-end scientific gate is **UNSOUND** under fail-closed acceptance because category omission and unsupported checkpoint mapping can still support an overconfident pass.
