# clip4cir_rn50x4_fullft — final audit (batch 3)

FINAL VERDICT: BLOCKED

Payload (structured, as delivered; the full narrated report was the assistant final message — this file preserves every finding verbatim).

## Summary

- pinned_source_commit: dfed9f748a8a4d05abf613164e06d59b22dbdf13
- paper: arXiv 2308.11485 v1, sha256 497c59216d606e92fd4eae9c57ed3cd76f180c5c6d161159b03be6a0a4996d1e, 18490782 bytes
- runtime_status: NOT_RUN — no checkpoint, no backbone, no dataset link, no configured interpreter

## benchmark_reference
```json
{
  "reference_id": "clip4cir_rn50x4_fullft_fiq_tab9",
  "reference_status": "VERIFIED",
  "review_status": "CONFIRMED",
  "reported": {
    "r10": 43.78,
    "r50": 66.93,
    "mean": null
  },
  "derived": {
    "mean": 55.36,
    "provenance": "WORKBENCH_DERIVED"
  },
  "category_metrics": {
    "shirt": {
      "r10": 44.41,
      "r50": 65.26
    },
    "dress": {
      "r10": 39.46,
      "r50": 64.55
    },
    "toptee": {
      "r10": 47.48,
      "r50": 70.98
    }
  },
  "independent_verification": "three routes: pdftotext -layout line 800; arXiv HTML table index 9 last <tr>; arithmetic (43.7833 -> 43.78, 66.93 exact, 55.355 -> 55.36)"
}
```

## Critical findings
- **P0** canonical FashionIQ root holds 74381 .png / 0 .jpg while the evaluator opens .jpg; dataset link absent — src/data_utils.py:158,161,170,176; ls data/fashionIQ_dataset/images | sed 's/.*\.//' | sort | uniq -c -> 74381 png; preflight 3817/6346/5373 of N lack .jpg
- **P1** registry/adapters bundle-path disagreement — registry.py:269-297 resolves members under <filename>/ ; adapters/models.py:101-104 reads checkpoint.parent; OBSERVED via workbench functions
- **P1** recorded checkpoint-mapping blocker is stale; authors' Drive folder resolves to exactly the fullft pair — embeddedfolderview id 1BPE33_XSm33Min0OaGW2Sl9rcddZ-WF- -> fiq_clip_RN50x4_fullft.pt + fiq_comb_RN50x4_fullft.pt; README id 1G7kZo7paiaRd3mif9_saERyhPevRCM88
- **P1** checkpoint_training_noise_pct null vs reference 0 makes the gate unpassable — paper_gate.py:231-252; OBSERVED noise=0 -> PASS_WITHIN_5_PERCENT, noise=None -> BLOCKED_CHECKPOINT_MISMATCH; checkpoint_selection refuses under both policies
- **P1** no metric parser for this evaluator's stdout — metrics_extraction.py:309-341 handles only limn/csmcir
- **P2** per-category gate comparison dormant because no run publishes categories — OBSERVED evaluate_gate detail lists 6 paper categories as 'not compared'
- **P2** bundle_placeholder invisible to manifest/change detection — evaluate_models.py:321, :643-645

## Minor findings
- replay_status BLOCKED_CHECKPOINT_MAPPING (models.yaml:706) vs reproduction_status BLOCKED_CHECKPOINT_BUNDLE (:731)
- model entry reported_r10/r50/mean null (:728-730) while the committed row is verified
- paper 6017 triplets / 15537 images vs local 6016 / 15536
- pc_readiness classifies BLOCKED_ENVIRONMENT though dataset/asset/checkpoint blockers exist and rank below it (pc_readiness.py:44-61)
- tests/test_replay.py:144-147 pins only the placeholder shape
- validation_caption_policy undeclared though src/datasets/fashioniq.py:151-155 matches src/validate.py:88-92
- working-tree churn during audit: 17->18 modified files, papers.yaml registry_version extract-275d524611f3 -> extract-1b62b520c448 (clip4cir rows untouched)

## Refuted prior claims
- prior-round appended 'readiness now BLOCKED_SOURCE' — source_clean_and_pinned returns (True, 'dfed9f74...', None) and preflight emits no source blocker
- AUXILIARY_ASSET_AUDIT.md:231-232 / cluster_b4_classic.yaml:157 'does not identify which fine-tuned CLIP state pairs with which Combiner state' — refuted by the Drive folder listing
- UPSTREAM_DATA_PREPARATION_AUDIT.md:171-172 'the workbench only links that path for CSMCIR' — models_requiring_link() returns 4 models including clip4cir_rn50x4_fullft and its sibling
- prior-round line references models.yaml:664-720 and papers.yaml extract-275d... are stale

## Unresolved
- bytes/digests of the two state files (no download attempted)
- local identity of the base RN50x4 object (hash corroboration only)
- which released artefact the Table 9 row corresponds to (strongly supported, not proven)
- camera-ready TOMM differences (only arXiv v1 fetchable)
- any runtime number (NOT_RUN)
- canonical numbered checklist list absent from context and repo; probes are the auditor's own

## Verified positives
- pin exact and checkout clean
- command flags match source and release README
- fashioniq_original_split correct, reference eligible (proven inside validate.py and on 6016/6016 real annotations)
- TargetPad 1.25 pipeline matches Algorithm 1 and §4.1
- metric definition matches §4.2.1
- paper row reproduced three independent ways

## Execution trace / checklist
Delivered in the assistant final message (sections MODEL ID .. FINAL VERDICT, EXECUTION TRACE, CHECKLIST 1-60); transcript: history://AuditClip4cir-2.
