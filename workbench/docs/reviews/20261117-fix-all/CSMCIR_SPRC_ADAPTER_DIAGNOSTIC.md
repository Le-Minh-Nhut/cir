# CSMCIR and SPRC adapter diagnostic evidence

## Scope

These results establish real native-model and production-adapter behavior for bounded diagnostics only. They do not establish full FashionIQ reproduction, full CSMCIR COT_ours2 evaluation, SPRC ViT-G/paper parity, or the V3 publication contract. The captures below predate the native-width layout migration and store repeated 8192-dimensional query rows; they are immutable historical evidence, not current-shape captures.

## CSMCIR

The adapter extracts target features through the upstream `extract_target_caption_features` API using explicit FashionIQ category scopes (`dress`, `shirt`, `toptee`) and the staged category-specific Qwen `Final_Caption` assets. Query captions use the upstream FashionIQ pair formatting and evaluation text processor. The native API ignores `text_cot`; the local COT_ours2 dataset files needed by the original wrapper are absent, so this is not an original-wrapper reproduction.

Diagnostic plan: `workbench/artifacts/parity/v3/csmcir-fiq-dress-q33-g35-diagnostic-preregistered.json` (SHA-256 `3c39ba6ac477c0300c2e048b009d2642d75606c6746c690e96811cf29b022984`). Official dress validation query pairs, 33 queries and 35 gallery images (all query positives plus two decoys), plus a separate 1×1 positive case. This is a diagnostic cohort, not the full 2,017-query/3,817-image evaluation.

- Native run: `workbench/artifacts/logs/csmcir-fiq-dress-q33-g35-native.txt`; output `workbench/artifacts/parity/v3/csmcir-native.npz`; exit 0.
- Production adapter run: `workbench/artifacts/logs/csmcir-fiq-dress-q33-g35-adapter.txt`; output `workbench/artifacts/parity/v3/csmcir-adapter.npz`; exit 0.
- Q33/G35 query feature maximum absolute error: `1.7881393432617188e-07`.
- Target feature maximum absolute error: `9.611248970031738e-07`.
- Native score maximum absolute error: `4.76837158203125e-07`; rankings equal.
- 1×1 query/target/score maximum errors: `0.0`, `2.980232238769531e-07`, `5.960464477539063e-08`; rank `[1]`.
- Target query/gallery batches 33/35 exercised adapter batch-size 32 tails of 1/3.
- Checkpoint SHA-256: `c8812071de0640d647c5ba2a70628b80a3289ee2c47676f7a1b6abaa313fa49e`. Upstream-compatible `strict=False` load reported no missing keys and one unexpected key, `token_importance`.

The native phase exited before the adapter phase. Adapter worker was closed; GPU returned to 573 MiB baseline with no adapter compute process.

## SPRC ViT-L

A separate preregistered 1-query × 3-image native/adapter smoke was completed with the author ViT-L checkpoint (`sprc_fiq_vitl`). Query and target features matched exactly; maximum score error was `1.1920928955078125e-07`, with equal ranking. Native and adapter receipts are in `workbench/artifacts/logs/sprc-fiq-vitl-dress-smoke-native-20261117T0336Z.txt` and `workbench/artifacts/logs/sprc-fiq-vitl-dress-smoke-adapter-20261117T0339Z.txt`; raw arrays are `workbench/artifacts/parity/v3/sprc-fiq-vitl-dress-smoke-native.npz` and `workbench/artifacts/parity/v3/sprc-fiq-vitl-dress-smoke-adapter.npz`.

This smoke does not establish full ViT-L parity or ViT-G/paper reproduction. The separate ViT-G checkpoint remains unresolved.


## Current feature layout

Current adapters return native query embeddings as `(N,256)` and gallery embeddings as `(M,8192)` (32 projected 256-dimensional gallery tokens flattened). Scorers normalize each query/gallery token with native `eps=1e-12` and take maximum cosine over gallery tokens. Historical diagnostic arrays above use prior `(N,8192)` repeated-query layout; no full run has been captured with current layout.
## Focused code checks

`workbench/artifacts/logs/csmcir-sprc-focused-contract.txt` records focused compilation and adapter tests. Latest run: `py_compile` exit 0; `pytest` 11 passed, exit 0. Tests cover asymmetric native q256/gallery8192 scoring, native normalization epsilon, non-mutation, ranking, malformed shapes, caption/config content identity, CSMCIR checkpoint payload selection and ambiguity, and SPRC ViT-G refusal.

## Full FashionIQ V3 preregistration (CPU-only)

CPU preflight generated immutable multimodal plans for official FashionIQ val, not model receipts or parity results. Plans cover q256 native query embeddings, flattened g8192 gallery embeddings, full query/gallery populations, B1/B32 controls, ten V3 run roles, source/checkpoint/runtime hashes, resource budgets, and sequential native-then-adapter GPU commands:

- CSMCIR: `workbench/artifacts/parity/v3/g1_g2.csmcir.{dress,shirt,toptee}.multimodal-q256-g8192.r4.v3-plan.json`
- SPRC ViT-L: `workbench/artifacts/parity/v3/g1_g2.sprc.{dress,shirt,toptee}.multimodal-q256-g8192.r4.v3-plan.json`

Counts are dress `2017/3817`, shirt `2038/6346`, toptee `1961/5373` (queries/gallery). No GPU capture, receipt, metric, or publication claim exists for these plans. CSMCIR scope remains captioned FashionIQ val; CIRCUS 74,381 full-gallery reproduction remains unsupported. Full runs require Main GPU lease, `/tmp/cir-cpu20-20261117.YmsolE/gpu.lock`, and actual native/adapter receipts accepted by V3 gate.
