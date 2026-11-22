# LIMN V3 execution

Status: Three-category FashionIQ native→production Adapter run complete (30 role receipts); isolated V3 gate failed exact shirt R@1 parity. No publication or serving `CURRENT` update. This measured run is diagnostic and does not establish overall scientific acceptance.

## Primary dress diagnostic

Entrypoint: `workbench/scripts/limn_v3_parity.py`; current phases are `plan`, `roles`, `roles-native`, `roles-adapter`, `aggregate`, and `publish`. The completed source-native FashionIQ dress validation (2017 queries, 2628 gallery items) and production Adapter run remains diagnostic-only; it emits no V3 gate receipt or publication.

The first sealed attempt used manifest `/tmp/limn-dress-primary-review-20261117-2218/primary-diagnostic-manifest.json` (SHA-256 `be1f79cffca285b25c5f968f09b3d618dc93ad0750bd292df361f5d92f815544`) and stopped at Adapter construction; preserve as failed historical attempt. Successful sealed run used `/tmp/limn-dress-primary-review-20261117-2136-guardfix3/primary-diagnostic-manifest.json` (SHA-256 `e12302b9a9cbaa09397078386a76e9bf089e6f69f98dcaf7c2e1900c37957fc8`). Its CPU preflight verified producer/T4/source/checkpoint/input hashes, native+worker 3.10 and controller 3.12, order/exclusion digests, feature dimension 768 and bytes per feature set (6,196,224 queries + 8,068,608 gallery = 14,269,440); output `PASS`, `cuda_execution: not started`.

Historical execution instructions used the then-active `/tmp/cir-cpu20-20261117.YmsolE/gpu.lock`; this lock belongs only to the archived 2026-11-17 run above and is not the current run lock. That diagnostic ran native first, waited for exit, then ran the ctl312h Adapter process. Both phases pinned seed 0, batch size 32, float32 and eval mode, and validated observed model/worker metadata. Native source evaluator metrics remain distinct from rankings recomputed from fresh native features. G1 image capture was bounded; no full gallery pixels.

The first sealed attempt on the earlier guardless source stopped at Adapter construction with the missing `_interpreter_verified` initialization. Main authorized that one-line constructor fix, a fresh preregistration, and a new lease; the failed root was preserved. The successful run above used only the new guard-fix seal.

## Full FashionIQ V3 measurement — 2026-11-20

Three-category roles and aggregates completed after the selected-inference `torch.no_grad()` fix. This is real source-native/production-Adapter evidence, but the exact V3 gate fails shirt R@1; **no publication or serving `CURRENT` update occurred**. The fresh run uses immutable category plans and frozen source hashes; its receipts remain pinned to those hashes even though a later producer correctness fix now rejects target/gallery-pixel mismatches.

Physical artifacts are under `/tmp/limn-v3-main-20261120T090500Z`; logs under `/tmp/limn-v3-main-logs-20261120T090500Z`. Durable copied diagnostic, including actual arrays, control/rank/G1 captures, 30 role receipts, plans, logs, isolated staging, gate output, and hashes: `workbench/docs/reviews/20261117-fix-all/raw/limn-v3-execution/limn-v3-tie-diagnostic-20261120/`. `SHA256SUMS` SHA-256: `5ddacdfab185dd1e711757d587ba8554e4cff72601e6b3adbe5dbedb2be664fe`. Frozen source hashes: producer `bc9e61646b6cc552d4cc5e05d3459b65eac619ae19ebde69a9facc0bcb0e10bc`; T4 `67328f28239db04c89e597eaf7498f33d16e480fed7dac2d9c52ec6250b06b85`.

All six bounded G1 pixel comparisons (query + target/gallery, three categories) were bitwise equal, shape `(5,3,224,224)`, max absolute delta 0.0. Sampled text, token IDs, padding masks, and actual causal masks also matched. Gate reports non-mandatory absent relation digests for all categories (`not_recorded`); no mandatory evidence is absent.

The actual V3 gate ran once on copied flat staging, not via publisher: command in `workbench/docs/reviews/20261117-fix-all/raw/limn-v3-execution/limn-v3-tie-diagnostic-20261120/gate-command.txt`; exit 0; full verdict in `gate-verdict.json`. Result: `PARITY_MANDATORY_SUBCHECK_FAILED`, `complete=false`, `passed=false`, 105 checks, one mandatory failure, no missing artifacts. Exact failure `shirt.metrics.hit_count_equality`: native R@1/10/50 `516/1141/1561`, Adapter `517/1141/1561`; R@1 unequal. Shirt query `B00461228C`, target `B00C5RAHMW` (rank 2 native, rank 1 Adapter) ties candidate `B00C5RC12G`; both candidates score `0.71493256092071533` in both matrices, but ordering differs. Tie is observed; this does not establish globally stable sort equivalence.

Dress: 2017 queries, native=Adapter hits `375/1032/1498`; max score delta `5.960464477539062e-07` over 5,300,676 pairs; one target-rank flip, zero unexplained. Shirt: max delta `6.556510925292969e-07` over 6,295,382; one rank flip at query 86, zero unexplained. Toptee: 1961 queries, native=Adapter hits `511/1181/1609`; max delta `5.960464477539062e-07` over 5,690,822; zero rank flips/unexplained. Aggregate details: durable `logs/aggregate-{dress,shirt,toptee}.log`.

Current producer additionally fails closed on target/gallery pixel mismatches, independently of its existing mandatory query-pixel failure. Regression covers mutated query and gallery capture paths. Test command used `flock -x /tmp/cir-fix-all-20261120-cpu.lock env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MAX_JOBS=10 workbench/envs/ctl312h/bin/python -m pytest -n 10 workbench/tests/test_limn_v3_parity.py -q -p no:cacheprovider`; result: `4 passed in 11.46s`. This source fix was made after the physical run and did not alter its pinned receipts or rerun gate.

## Current acceptance interpretation — 2026-11-21

Existing shirt R@1 mismatch above remains recorded historical failure (native 516, Adapter 517); it is not a new acceptance result or PASS. Gate/producer-focused verification validates fixture and proof handling only. No whole-model LIMN rerun is justified without a new source/protocol/checkpoint change or independently demonstrated measurement defect and a fresh sealed plan. Existing archive/receipts remain unchanged.

## Prior bounded dress diagnostic (2026-11-17)

Earlier primary dress native→production Adapter diagnostic and CPU tie analysis follow; they are not this full FashionIQ V3 measurement.

Independent NumPy re-ranking from saved native float32 features agreed with the independent re-ranking implementation for all 2017 queries. Its R@1/10/50 matched actual source-evaluator aggregate metrics. It is not an upstream per-query rank observer; only the aggregate metrics come directly from source evaluator output.

CPU analysis invoked actual `Adapter.score` and `Adapter.rank` on full saved adapter arrays. Full unmasked production score matrix (5,300,676 query-gallery pairs) differed from normalized float32 dot by max absolute `5.960464477539062e-7` (mean `2.6886576520723793e-8`), within absolute tolerance `1e-6`. Under native reference exclusion (`-10e10`) and production rank semantics, Adapter recall was R@1/10/50 `18.542389687654932 / 51.165096678235 / 74.26871591472484`; compared with source aggregate there were 24 per-query rank differences (max delta 6), including 3 R@1 hit differences and none at R@10/R@50. The 2×2 counterfactual below attributes 23 rank changes to NumPy-vs-production tie policy and one to score drift. Earlier Adapter `17.0550 / 50.5206 / 74.2687` used unmasked CIRCUS-eligible ranks; not comparable to reference-excluded FashionIQ native metrics.

Production Adapter rank vs NumPy `argsort` unmasked had 26 row differences. CPU-only 2×2 counterfactual applied both NumPy default `argsort` and production's stable lower-gallery-index rule to both native-reconstructed and Adapter score matrices, all with identical reference exclusions. Native/Adapter rank matrices differed on one query under either ranker; 23 other query rank changes came from the ranker convention on exact score ties. The lone score-driven divergence is query index 60: target `B007ORZ1PG`, reference `B0092QK9AY`, candidate `B008AM8NEA` at gallery position 2472. Candidate-minus-target score was `+2.9802322387695312e-08` native and `-1.1920928955078125e-07` Adapter, moving its rank across target. Full rank vectors and per-query candidate crossings are in `/tmp/limn-dress-primary-review-20261117-2136-guardfix3/rank-score-tie-counterfactual.json`; earlier unmasked and masked diagnostics remain preserved at `policy-matched-rank-analysis.json`.

Fresh guard-fix seal run completed both native and production Adapter: PID/PGID `11571`, `2026-11-17T14:32:39Z`–`2026-11-17T14:36:48Z`, exit 0. Both populations contain 2017×768 queries and 2628×768 gallery vectors, float32, batch 32, eval mode; observed order hashes and checkpoint SHA match. Feature maximum absolute delta `2.9802322387695312e-08` for queries and gallery; minimum cosine `0.9999998807907104`. Diagnostic only; no V3/full category/CIRCUS claim and no CURRENT activation. Outputs/logs stay under `/tmp/limn-dress-primary-review-20261117-2136-guardfix3/`; no LIMN compute PIDs after exit.

Commands in immutable manifest define the exact native→Adapter sequence and root. Do not reuse or mutate prior preregistration roots.

Durable raw diagnostic archive: `workbench/artifacts/parity/diagnostics/limn-dress-guardfix3/`. It contains the sealed manifest, unchanged native/Adapter NPZ and observations, run logs/process record, GPU census, and CPU rank analyses. `archive-index.json` records original `/tmp` and durable paths, bytes, SHA-256, and byte-copy verification; index SHA-256 `559b4bf3748018f6fe87e5736096879fcaf74d502041efd228af6c73c168dc5c`. Archived source payload: 18 files / 30,521,666 bytes. Original `/tmp/limn-dress-primary-review-20261117-2136-guardfix3/` remains intact.

## Full parity remains open

This primary diagnostic does not complete the ten-role V3 controls or all-category acceptance. Full FashionIQ remains 6016 queries / 8619 category gallery entries; CIRCUS remains 6001 resolved queries / 74381 global gallery items. No status promotion before complete receipts/scopes, full gate evaluation and atomic publication.

## Source-faithful protocol

FashionIQ val captions drive query order. Gallery order is first-seen candidate/target union, matching pinned LIMN `dataset.test_targets`; it is not `image_splits` order. Observed protocol counts:

| category | queries | gallery |
| --- | ---: | ---: |
| dress | 2017 | 2628 |
| shirt | 2038 | 3089 |
| toptee | 1961 | 2902 |
| total | 6016 | 8619 |

The V3 producer runs one source-native full evaluator and selected source controls in a resident native process, then one production Adapter full-population encode and selected controls in a separate resident Adapter process. It records actual role receipts, full component scopes, bounded G1 captures, source-native metrics and production ranking evidence; no publication is valid before full gate evaluation.

The producer fixture is synthetic CPU-only evidence, not physical LIMN acceptance. `workbench/tests/test_limn_v3_parity.py` passed `3 passed in 11.42s` on 2026-11-20; exact log: `workbench/docs/reviews/20261117-fix-all/raw/limn-v3-execution/ranking-archive-publish-test-20261120.log`. It exercises three-category assembly, `evaluate_model`, archived native/Adapter ranking evidence, successful synthetic publication, and refusal of a second incomplete publication without advancing `CURRENT`. Earlier refusal/unchanged-`CURRENT` evidence is recorded in `workbench/docs/reviews/20261117-fix-all/raw/limn-v3-execution/publication-refusal-check-20261120.log`. No real LIMN acceptance has been established by these fixtures.


## Safety and remaining scope

No LIMN GPU process remains active from this run. Any further `roles` phase requires a new explicit Main lease; do not repeat the same FashionIQ run under unchanged source/contract. Preserve the pinned run and exact failed gate evidence above. The later target/gallery-pixel guard changes current producer behavior but does not alter the archived physical receipts or make their gate pass.

CIRCUS remains unresolved and outside this FashionIQ run: the source-faithful protocol must establish authoritative LIMN routing for 6,001 resolved queries against the full 74,381-image gallery; historical caption/category audits do not satisfy that scope. Resolve checkpoint, native evaluator, and adapter identities from current paths/receipts before planning any CIRCUS execution. Shirt `metrics.hit_count_equality` remains the concrete FashionIQ gate blocker; do not alter ties, ranking convention, or thresholds to force parity.
