# CSMCIR FashionIQ V3 parity evidence — 2026-11-21

Scope: sealed native/adapter FashionIQ val captures for CSMCIR. Issued gate verdicts: dress r14 `PARITY_MANDATORY_SUBCHECK_FAILED`; shirt and toptee r15 `PARITY_PASS`. Shirt/toptee PASS rank justification is under audit: gate trusts a builder-supplied unexplained-rank count, which was not derived from all exact-row differences. No change to captured receipts, plans, tolerance, or policy. Original plans and raw role NPZs remain immutable source evidence; tie validity is not asserted pending per-query derivation.

## Dress r14

- Plan: `workbench/artifacts/parity/v3/g1_g2.csmcir.dress.multimodal-q256-g8192.r14.v3-plan.json`; SHA-256 `17f82bc1612805673b4df637fa27f1dc8cc1c2a806da3570ee51ae3c1e973d6f`.
- Population: 2,017 queries × 3,817 gallery items. Ten preregistered native/adapter roles have receipts in `/tmp/cir-csm-r14-four-20261120-2312.stdout.log` and `/tmp/cir-csm-r14-remaining6-20261120-2318.stdout.log`; no inference rerun for this evidence.
- Canonical V3 artifacts: `workbench/artifacts/parity/v3/g1_g2.csmcir.dress.native.json`, `workbench/artifacts/parity/v3/g1_g2.csmcir.dress.adapter.json`, and sibling `workbench/artifacts/parity/v3/g1_g2.csmcir.dress.v3-plan.json`. Sibling plan bytes match the preregistered plan. Actual `parity_gate.evaluate_model` output: `/tmp/csm-dress-r14-real-gate.json`.
- Verdict: `PARITY_MANDATORY_SUBCHECK_FAILED`. Forward components, G1 identity/pixels, and integer metric checks pass. Score maximum absolute delta `5.960464477539062e-7` is within preregistered `1e-6`. Exact sorted-rank rows differ on 1,952/2,017 queries; positive target ranks differ on 1/2,017. The gate reports mandatory `dress.ranks.unexplained_differences`; retain this failure. Equal hits do not erase rank differences.
- Native/adapter batch-1 vs batch-32 component checks show declared-rule deviations; gate identifies these controls as diagnostic/nonmandatory. Relation digest was not captured; report as absent, not agreement. Relation check is nonmandatory for this model. Score/rank diagnosis is recorded in `csmcir-rank-diagnostic-20261121.md`; observed sub-epsilon score/tie changes do not establish a complete cause for all reordered rows.

## Shirt r15

- Plan: `workbench/artifacts/parity/v3/g1_g2.csmcir.shirt.multimodal-q256-g8192.r15.v3-plan.json`; ten filtered receipts in `/tmp/csm-shirt-r15-receipts.jsonl`; aggregate `/tmp/csm-shirt-r15-g2-abs1e6.json`.
- Population: 2,038 queries × 6,346 gallery items. Canonical artifacts `workbench/artifacts/parity/v3/g1_g2.csmcir.shirt.native.json`, `.adapter.json`, sibling `.shirt.v3-plan.json`; actual gate output `/tmp/csm-shirt-r15-real-gate.json`.
- Issued verdict: `PARITY_PASS` from `parity_gate.evaluate_model`. Aggregate records exact sorted-index rows differ. Audit found gate consumes artifact-supplied `unexplained_rank_differences`; builder derived this field from positive-target rank changes, not per-row tie evidence. Therefore issued status is retained, but full-row tie justification is unverified; do not claim every difference is tie-explained. Sealed rank rows, captures, and receipts unchanged. Batch-1-vs-N checks are diagnostic/nonmandatory; relation digest absent.

## Toptee r15

- Plan: `workbench/artifacts/parity/v3/g1_g2.csmcir.toptee.multimodal-q256-g8192.r15.v3-plan.json`; ten filtered receipts in `/tmp/csm-toptee-r15-receipts.jsonl`; aggregate `/tmp/csm-toptee-r15-g2-abs1e6.json`.
- Population: 1,961 queries × 5,373 gallery items. Canonical artifacts `workbench/artifacts/parity/v3/g1_g2.csmcir.toptee.native.json`, `.adapter.json`, sibling `.toptee.v3-plan.json`; actual gate output `/tmp/csm-toptee-r15-real-gate.json`.
- Issued verdict: `PARITY_PASS` from `parity_gate.evaluate_model`. Aggregate records exact sorted-index rows differ and batch-1-vs-N score deltas exceed tolerance. Audit found gate consumes artifact-supplied `unexplained_rank_differences`; builder derived this field from positive-target rank changes, not per-row tie evidence. Therefore issued status is retained, but full-row tie justification is unverified; do not claim every difference is tie-explained. Full-population native/adapter score delta `6.556510925292969e-7` <=1e-6; hit-count vectors equal. Batch controls diagnostic/nonmandatory; relation digest absent. Capture receipts/plans unchanged.
