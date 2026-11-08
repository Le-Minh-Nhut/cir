# GATE_MATRIX — per-variant × per-gate status

Machine-readable-ish companion to `INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md`.

**Provenance of the gate letters.** G0 is named by `20261108T152852-audit-inventory/REPOSITORY_INVENTORY.md`
(its own title). G2/G3/G4 are defined by `20261108T153045-audit-equivalence/TOLERANCES.md`. G1, G5, G6,
G7 are **not defined anywhere in the evidence corpus** — no file in
`workbench/docs/reviews/20261108T15*`, `20261108T-throughput`, or `20261108T113610-eval` states their
subject. The letters G1–G7 recur in `workbench/docs/reviews/a40/` and `final_pc_audit/` with *unrelated*
meanings (artifact safety, browser E2E, ledger integrity, pip index, environment identity), so those
must not be conflated with this audit programme. See `INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md` §10
"unresolved". The rows below are filled with **what the evidence actually measured**, and each row
names the evidence that supports it; where a gate's subject is unknown, the row states the widest
defensible reading and says so.

## Legend

| symbol | meaning |
|---|---|
| PASS | gate's question answered, all measured cases agree |
| FAIL | gate's question answered, at least one case disagrees |
| PARTIAL | some cases measured, others not |
| NOT_RUN | nothing measured; reason given |
| N/A | gate does not apply to this variant (no composition adapter, no checkpoint, etc.) |
| DEFECT | measured, and a defect is named |

## Column meanings actually used

- **G0 inventory** — identity established: repository, pin, provenance, checkpoint hash present.
- **G1 protocol assignment** — the variant's `native_protocol` matches what the pinned upstream code
  computes (`20261108T155252-protocol-audit/PROTOCOL_MATRIX.md`).
- **G2 decode/preprocess** — pixel matrix bit-identical across worker/batch grid (equivalence audit).
- **G2-fwd forward features** — encoded feature matrix bit-identical. **NOT_RUN everywhere** (GPU busy).
- **G3 scoring, identical arithmetic order** — query-axis regrouping preserves ranks.
- **G3b changed summation order** — `optimize=True`/chunked GEMM preserves ranks.
- **G4 axes** — batch 1 vs 32, incomplete final batch, repeat determinism, cold vs warm cache.
- **G5 cache identity** — cache cannot be reused under a different computation.
- **G6 throughput** — measured speed-up with bit-identity.
- **G7 publishable provenance** — the artifact in the tree reproduces from the tree.

## Matrix

Rows = the 29 declared checkpoint variants (the unit the registry keys weights by), grouped by the
17 `model_id`s that own them. `c` = composition adapter exists.

### Group A — `fashioniq_original_split` (full val gallery 15,536, reference KEPT)

| model_id | checkpoint | G0 | G1 | G2 decode | G2 fwd | G3 | G3b | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `csmcir` | `fashioniq` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | FAIL — adapter refuses all 3 conditions; captions cover 15,415/74,381 = 20.72% |
| `ptha_mtst` | `cirr_transfer` | PASS | PASS | N/A (encode unreachable from HEAD: F7) | NOT_RUN | N/A | N/A | N/A | N/A | N/A | FAIL — TRANSFER label; artifact `315dd…` predates F7 fix |
| `combiner_rn50x4_noft` | `rn50x4_noft_unresolved` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | PARTIAL — real CPU run, R@10 35.1078 / R@50 58.4756, but no paper row to compare |
| `clip4cir_rn50x4_fullft` | `rn50x4_fullft_unresolved` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN — `adapters/models.py:374-375` reads `combiner_state.pt`/`clip_state.pt`, absent on disk |
| `sprc` | `sprc_fiq_unresolved` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |
| `sprc` | `sprc_fiq_vitl` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN — 3 preflight blockers (link, BLIP-2 asset provenance, stale `command_status`) |
| `tme` | `fashioniq_0` | PASS | PASS | N/A | NOT_RUN | N/A | N/A | N/A | N/A | N/A | PARTIAL — artifact `ad477…`, `variant_provenance` absent from summary |
| `tme` | `cirr_0` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |

### Group B — `fashioniq_full_gallery_ref_excluded` (full val gallery 15,536, reference MASKED)

| model_id | checkpoint | G0 | G1 | G2 decode | G2 fwd | G3 | G3b | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `airknow` | `fiq_n05` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN — no interpreter (`WORKBENCH_AIRKNOW_PYTHON` unset) |
| `airknow` | `fiq_n08` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN — same |
| `conesep` | `fiq_n02` | PASS | PASS | PASS (shared targetpad family, 0.0) | NOT_RUN | N/A | FAIL (inherits `optimize=True`) | N/A | DEFECT (runner-level) | PASS 21.2→2089 img/s decode-only | **PASS** — `30f6de42…` COMPLETED, 6003 q, R@10 35.48225887056472, topk COMPLETE; train noise 20% |
| `conesep` | `fiq_n05` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |
| `conesep` | `fiq_n08` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |
| `habit` | `fiq_n02` | PASS | PASS | PASS (10 rows: 4 at batch 32 × worker 1/2/4/8, 6 at batch 1/64/256 × worker 1/8) | NOT_RUN | NOT_RUN (shipped path unmeasured) | FAIL on ranks (0 metric crossings) | PARTIAL | DEFECT (7 inputs) | PASS 5.67× query-axis | PASS — `497c25c57fbeba885ecfb9d1`, COMPLETED, 6003 queries |
| `habit` | `fiq_n05` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |
| `habit` | `fiq_n08` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |
| `intent` | `fiq_n02` | PASS | PASS | N/A | NOT_RUN | N/A | N/A | N/A | N/A | N/A | PASS — `37ee72aa…`, COMPLETED, R@10 34.39946693320007 |
| `intent` | `fiq_n05` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |
| `intent` | `fiq_n08` | FAIL — bytes absent | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN |
| `dcnet` | `fashioniq_run_directory` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | FAIL — `TypeError: 'weights_only'` on pinned torch 1.4.0 |

### Group C — `fashioniq_val_split` (pair-union gallery 8,619, reference MASKED)

| model_id | checkpoint | G0 | G1 | G2 decode | G2 fwd | G3 | G3b | G4 | G5 | G6 | G7 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `hint` | `fashioniq` | PASS | PASS | N/A | NOT_RUN | N/A | N/A | N/A | N/A | N/A | PASS — `4bc9893a…`, COMPLETED, R@10 32.816924870897886, topk COMPLETE |
| `encoder` | `fashioniq` | PASS | PASS | PASS (16/16 grid, sha256 `4f6a14f9…`) | NOT_RUN | N/A | N/A | N/A | DEFECT | PASS — `preprocess_val` transform 0.0 at 2/4/8 workers; family A/B 21.2→2089 decode-only | FAIL — artifact `6a528…` predates the F7 import fix |
| `pair` | `pair_b1` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN — checkpoint not at declared root |
| `pair` | `pair_b2` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | NOT_RUN — same |
| `clvc_net` | `author_drive_unresolved` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | FAIL — 4 unguarded `.cuda()` on the scoring path |
| `tgcir` | `tgcir_archive_unresolved` | PASS | PASS | N/A | N/A | N/A | N/A | N/A | N/A | N/A | FAIL — same `.cuda()` blocker; no CLI in pinned `test.py` |
| `limn` | `base_iter0_dress` | PASS | PASS | N/A | NOT_RUN | N/A | N/A | N/A | DEFECT | N/A | PARTIAL — `10173…` PARTIAL, cohort 5,884/6,003 |
| `limn` | `base_iter0_shirt` | PASS | PASS | N/A | NOT_RUN | N/A | N/A | N/A | DEFECT | N/A | PARTIAL — same artifact |
| `limn` | `base_iter0_toptee` | PASS | PASS | N/A | NOT_RUN | N/A | N/A | N/A | DEFECT | N/A | PARTIAL — same artifact |

## Gate-level roll-up

| gate | subject (per evidence) | verdict | the number |
|---|---|---|---|
| G0 | inventory | PASS | 16 repos / 17 variants / 29 checkpoint variants / 21 staged; 2 non-model checkouts |
| G1 | protocol assignment per pinned upstream | PASS 17/17 | 0 MISMATCH; 4 wrong line citations + 3 entrypoint misstatements (documentation defects, not protocol errors) |
| G2 | decode/preprocess bit-identity | PASS where measured | habit 10 rows (4 + 6 batch), one sha256 `5fc8002d…`; encoder 16/16 rows, one sha256 `4f6a14f9…` |
| G2-fwd | forward feature bit-identity | **NOT_RUN** | GPU owned by the `intent` cohort the whole window |
| G3 | query-axis regrouping | **NOT_RUN** on the shipped path; PASS on the fast path | equivalence audit explicitly did not measure `WORKBENCH_SCORE_THREADS` |
| G3b | changed summation order | **FAIL** | 207 / 18,009 rank changes; 0 K=10; 0 K=50; 19 order-only |
| G4 | axes (batch, tail, repeat, cold/warm) | PARTIAL | repeat determinism PASS; cold/warm **NOT_RUN** |
| G5 | cache identity | **DEFECT** | 7 inputs absent; sidecar verifies no array bytes |
| G6 | throughput | measured | decode 21.2 → 2089 img/s; query-axis 5.67× at 8 threads; end-to-end ≈5× |
| G7 | publishable provenance | **needs verification** | see §10 of the main doc |

## Naming caution

`G7` appears in `final_pc_audit/B.md` as "G7 (identity) — plan with and without
`WORKBENCH_RUNTIME_ADAPTATION=sm_120`" and in `final_pc_audit/D.md` as a LIMN certified-pass test.
Neither is this programme's G7. Any future document that says "G7" must name its subject.
