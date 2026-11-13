# Acceptance ledger — 20261113 final acceptance

Seeded from `20261112-parity-repair/PARITY_AND_EVAL_REPAIR.vi.md` §11, `BACKEND_DEFECTS.md`,
`GALLERY_DIGEST_PATCH_SPEC.md`, `SCOPE_RESOLUTION_PATCH_SPEC.md`, `CLVC_DIAGNOSTIC.md`,
`MODEL_MATRIX.md` §5. Every row states whether it was **re-verified in this recon turn** (measurement
named) or left `UNVERIFIED` with the command that would settle it.

State vocabulary used here (not the matrix's): `OPEN-VERIFIED` (defect present now, measured this
turn) · `CLOSED-VERIFIED` (fix present now, measured this turn) · `SPEC-ONLY` (defect reproduced in
an earlier record, no code change landed) · `EXTERNAL` (needs a source/metadata artifact, not code) ·
`UNVERIFIED` (not measured this turn).

**Re-verified 2026-11-13 at handoff.** The seeded rows were written mid-campaign, so twelve of them
described a state that the campaign's own commits then changed. Each was re-measured against the
delivered tree and its state cell corrected — see `## §Handoff` for the old → new table. A state cell
carrying the marker `2026-11-13 (… at handoff)` is the delivered state; a cell without it is the
observation made during the lane that wrote the row, and is history.

| id | area | defect or feature | owner lane | pre-fix evidence (file:line + reproduced observation) | required acceptance check | state |
|---|---|---|---|---|---|---|
| L01 | parity/PAIR | PAIR is `BLOCKED`: released pickles were produced by classes the pinned repo does not contain, so the registry mapping is unverified while the bytes hash-match | registry + models surface | `workbench/registry/models.yaml:825` `reproduction_status: BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`, `:828` `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, `:847-875` two `CHECKPOINT_MAPPING_UNVERIFIED` variants; `workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/pair_blocker.json` `verdict: BLOCKED_SOURCE_CLASSES_UNPUBLISHED` + 7 unpublished class names; live `/api/models` `pair` checkpoints show `official_sha256_match: true, mapping_verified: false, runnable: false` | `/api/models` must never present a PAIR checkpoint as `runnable`, and the on-screen reason must name the mapping, not just the interpreter | CLOSED-VERIFIED (live) |
| L02 | registry + env | `WORKBENCH_PAIR_PYTHON` is not set, so PAIR's interpreter blocker is `UNCONFIGURED` on top of the mapping blocker — two blockers, one cause chain | registry/env | `workbench/config/workbench.env` has no `WORKBENCH_PAIR_PYTHON` (17 `python_env_var` declared, 16 distinct, 5 unset: AIRKNOW, CLIP4CIR, DCNET, PAIR, SPRC); live `/api/models` `blocking_reasons[1]`: `WORKBENCH_PAIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED]`; `workbench/envs/pair-adapted/bin/python -> /home/heheboiz/miniconda3/envs/teacher-sprc/bin/python` **dangling** | every `python_env_var` resolves to an existing file, or its absence is reported as the only blocker | OPEN-VERIFIED |
| L03 | gallery digest | `gallery_identity_sha256` has two formulas under one key (`0bfd43b9…` plan/blocked vs `1db68c59…` completed run) | runner/scripts lane (outside this campaign's code lanes) | `workbench/scripts/composition_eval.py:298-300` (`_plan`) and `:426-429` (blocked-report branch) still compute `sha256({protocol_id, image_ids})`; `workbench/composition/runner.py:512-515` `_gallery_fingerprint` computes `{protocol_id, ordered_gallery, candidate_names}` — **both present, unfixed this turn** | `_plan(...)["gallery_identity_sha256"] == _gallery_fingerprint(data)` | CLOSED-VERIFIED 2026-11-13 (measured: plan == blocked == completed == `1db68c59…`; order-change negative flips it) |
| L04 | scope resolution | `adapter_query_scope_resolution.still_unscoped` counts "adapter does not route" as "could not scope" (6003 on a run that scored all 6003) | runner lane | `workbench/composition/runner.py:2528-2537` still writes `still_unscoped: sum(scope is None for scope in query_scopes)` with `unscoped_qids` from `scope is None`; no `category_agnostic` key anywhere under `workbench/composition`, `workbench/backend`, `workbench/tests` (grep → 0 hits) | `still_unscoped == sum(condition_status[*].unscored_count)` on `7b1fa85007bffdab931f7f19` | CLOSED-VERIFIED 2026-11-13 (runner derives unresolved from the reason; `7b1fa850…`'s own artifact left as history) |
| L05 | backend/composition | `_recorded_source_matches_disk` compared the shared-module hash to the model module → `scoring_impl: unknown` for every run | backend lane | fixed at `workbench/backend/composition.py:689-729` (two hashes, two modules, both must match) | artifact `scoring_module_source_hash` AND `adapter_module_source_hash` both equal to disk hashes | CLOSED-VERIFIED (source) |
| L06 | backend/composition | `served_conditions` advertised a condition the run never measured (`image_only_empty_text` on a 3-condition run) | backend lane | `workbench/backend/composition.py:23-78` `run_conditions` / `displayed_conditions`, `:1593-1608` `served_conditions` from declared-or-ranked | live `da0a2162be0e159a1a456cec` `conditions_display` = `multimodal, text_only_black_image, clvc_image_encoder_only` (no `image_only_empty_text`) | CLOSED-VERIFIED (live) |
| L07 | parity gate | `standard` served as a bracket list instead of the standard of the first model with a verdict; no per-standard breakdown | gate lane | live `/api/runs/parity-gate`: `standard: "parity-relative/v1"` (string), `standards: {"parity-absolute-relative/v2": ["hint"], "parity-relative/v1": ["encoder","limn"]}`, `requested_standard: "parity-relative/v1"` | gate payload has both keys, `standard` is a string | CLOSED-VERIFIED (live) |
| L08 | parity gate | HINT was not discoverable; only encoder/limn had artifacts | gate/producer lane | live parity payload `models` = `encoder, hint, limn`; `standards` groups `hint` under `parity-absolute-relative/v2` | `_discover_models` returns **7** ids (5 paired + 2 G1-only) and HINT carries a verdict with its own standard | **CLOSED-VERIFIED 2026-11-13 (re-measured at handoff)** — `parity_gate._discover_models(root)` → `['conesep','encoder','habit','hint','intent','limn','tgcir']`; 5 carry a verdict (encoder, habit, hint, limn, tgcir), `conesep`/`intent` at `PARITY_EVIDENCE_MISSING`; live `/api/runs/parity-gate` reports the same 7. The original "3 ids" was the state when HINT landed |
| L09 | parity gate | ENCODER relation pin absent from the producer (G1 red on 3 `relation.*.digest`) | parity producer lane | cannot re-measure artifacts this turn without a GPU; gate now reports G1 as measured (`coverage` all `measured`) and only G2 relative-term failures | `g1_g2.encoder.*` artifacts carry `relation_state.sha256 20db0c27…` with `matches_pinned_digest: true` | **CLOSED-VERIFIED 2026-11-13 (measured at handoff)** — `g1_g2.encoder.{dress,shirt,toptee}.{native,adapter}.json` carry `relation_state.sha256 = 20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92`, `expected_sha256` the same, `matches_pinned_digest: true`, `seed: 20261111`, on **both** sides; the gate's three `relation.*.digest` checks are measured, not absent (`gate.json` encoder `relation_parity: true`) |
| L10 | parity tolerances | **19** mandatory failures (encoder 9, habit 2, hint 2, limn 3, tgcir 3) are all `max_rel` on a vanishing denominator; worst `max_abs` anywhere ≤ 2.90e-06 (tgcir dress), `min_cosine ≥ 0.9999999997` | decision, not code | live gate: encoder 9 + habit 2 + hint 2 + limn 3 + tgcir 3 names, all `*.batch_1_vs_n.max_rel` / `*.population.*.max_rel`; class census **19** `measured_disagreement`, **0** `absent_evidence_refusal`, **0** `derived_invariant` | either `max_abs`-only scopes as an explicit recorded decision, or a floored-denominator `max_rel`; **never** moving the tolerance after reading the numbers | **OPEN-VERIFIED 2026-11-13 (re-measured at handoff; counts 9/2/2/3/3, total 19)** — still the open standard-level decision. The earlier "14 (encoder 9, limn 3, hint 2)" was the count before HABIT's and TG-CIR's pairs landed |
| L11 | CLVC | two native macros conflict; settled as batch-size mechanism (32 → 30.0208, 48 → 29.9707) | matrix/doc lane | `MODEL_MATRIX.md:387-403`; live `/api/native-measurements` `clvc_net` rows = 2 counted | the disposition cell names `NATIVE_MACRO_RESOLVED_BY_BATCH` and `--batch-size` is part of measurement identity | **CLOSED-VERIFIED 2026-11-13 (measured at handoff)** — `MODEL_MATRIX.md:433` carries `NATIVE_MACRO_RESOLVED_BY_BATCH`; `/api/native-measurements` has **5** `clvc_net` rows of which **2 counted**, with exactly the two settled macros (`b56c50074bcde5b5` 29.970746517160162 / 58.62171212901515 / 44.29622932308766, and `3281f06d720d1f30` 30.02079703540087 / 58.67189400437203 / 44.34634551988645); `runner.py:2031` puts `batch_size` into the RUN identity. **Caveat, stated so the row is not over-read:** the native-measurement row itself exposes no `batch_size` field (checked: 0 of 5 rows carry one), so the batch is invariant at the composition-run level, not on the served native row |
| L12 | CLVC | `image_only_empty_text` is unrepresentable (`text_model.py:84-86` IndexError) — must stay `ABLATION_UNSUPPORTED`, not faked | model adapter lane | live `eab10106b2df8e97fb7a5ce5` `condition_status.image_only_empty_text.status = ABLATION_UNSUPPORTED` with the reason string; `/api/composition/runs` still declares the condition | the condition keeps its entry and its status; no rank is fabricated | CLOSED-VERIFIED (live) |
| L13 | scope | 2 ambiguous queries `7:3038`, `7:3790` remain unscoped (119 ambiguous, 117 resolved) | data, not code | `workbench/docs/reviews/20261111-master-audit/raw/m02/m02_recount.json`: `ambiguous_qid_count 119`, `resolved_count 117`, `unresolved_qids ["7:3038","7:3790"]`, 7 checks `true` | no arbitration rule may assign a query to a category the release does not name | EXTERNAL (unchanged) |
| L14 | native measurements | clip4cir fullft has TWO macros: served `43.8016931215922` vs roll-up `43.81821850935618` — **settled this turn; the served one is right** | native-measurements lane | live row `ff7496d7aaee6242`: `metrics.r10 = 43.8016931215922` / `r50 = 66.98358456293742`, `backed_by = docs/reviews/20261109-native15-composition-limn/raw/native/clip4cir_fullft.json` (`sha256 fb97cfab…398e`), `raw_metrics.r10 = 43.81821850935618` from `measurements/clip4cir_parsed.json`. Rule read from `third_party/CLIP4Cir/src/validate.py:5,281-297,308-309`: macro = **unweighted mean of the three category R@10s** (`statistics.mean`), not pooled (pooled = 43.76662307359437). Replay cells recompute to 43.8016931215922 exactly; stdout prints `average recall10 = 43.8016931215922`. **The roll-up is the wrong number**: its `dress_r10 39.11750018596649` and `toptee_r50 71.23916149139404` occur in no other file under `workbench/`, so 43.8182 averages cells no artifact backs | quote the replay; the roll-up pair stays visible only as `raw_metrics`, which is what the endpoint already does | **CLOSED-VERIFIED (behaviour correct; record corrected)** — no code change needed. `MATRIX_RECONCILIATION.md` R01, `MODEL_MATRIX.md` footnote 33 |
| L15 | native measurements | ENCODER's second native aggregate (`raw/native/encoder.json`, `56.290792549557/77.633389132303`) is served nowhere | native-measurements lane | live API: exactly ONE encoder metric row `6fb58ddfb02c9d56` = `56.223101675386054/77.66738539271432` (`cohort/measurements/encoder_fashioniq.json`). `raw/native/encoder.json` has **no `replay_type`** (keys: `aggregate, model_id, protocol, recorded_at, results`), which is the structural predicate `_review_run_payload` / `_backing_reports` both require — so it is neither a row nor a backing report. Same for `raw/native/tme.json` | the exclusion must be stated and the unseeded value must never be presented as current | **CLOSED-VERIFIED — deliberate exclusion by a stated predicate, not a publication gap.** Both ENCODER aggregates are unseeded draws (`ENCODER_RELATION.md` §8-§9b). Corrected `EVIDENCE_INDEX.md:94`, `FINAL_REPORT.vi.md:46`, `ENCODER_PARITY.vi.md:253` to mark `NOT_REPRODUCIBLE` and name the seeded `native_primary` pair `56.22580193539884/77.53315673870537` |
| L16 | native measurements | `combiner_rn50x4_noft` is roll-up-only (`METRIC_REFERENCE_UNVERIFIED`) | native lane | live row `df785c5847244b67` `35.1078/58.4756` from `selection_decisions.json`, `rollup: true`; the `clip4cir_rn50x4_noft` roll-up `35.1407/58.4436` exists as a separate, uncounted row | the row stays flagged roll-up and is never counted as a reproduction | CLOSED-VERIFIED (live flags) |
| L17 | API | `/api/runs` and `/api/jobs` are empty, so the per-query surface and the queue render their empty states | index/jobs lane | live `GET /api/runs -> 200 2 bytes`, `GET /api/jobs -> 200 2 bytes`; `GET /api/cohorts -> 200 2` | every empty surface says "no rows" rather than failing | CLOSED-VERIFIED (live, empty by data) |
| L18 | release guard | `test_no_release_document_points_at_a_superseded_source_sha` asserts the pin contains every production byte | release lane | measured **8 passed in 14.87 s** this turn at HEAD `b4d25c7` with `git diff --quiet cf6ba468 HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` = exit 0 | guard green and pin invariant exit 0 | **CLOSED-VERIFIED 2026-11-13 (re-measured at handoff after the pin advance)** — pin is `TESTED_SOURCE_SHA = 82b0b3fb72e9181a8835c07a0974a1279f21aeef`; `pytest workbench/tests/test_handoff_commands.py` → **8 passed in 16.18 s**; `git diff --quiet 82b0b3f HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` → exit **0** at HEAD `c502b98`. The `cf6ba468`/`b4d25c7` figures above are the state before the two pin advances |
| L19 | repo hygiene | the worktree is no longer clean: two production files are modified and one untracked test appeared during this turn | whichever lane owns the LIMN diagnostic | `git status --porcelain` at 01:41: ` M workbench/backend/limn_diagnostic.py`, ` M workbench/frontend/LimnScopeDiagnostic.tsx`, `?? workbench/tests/test_clvc_diagnostic.py`; `git diff --stat cf6ba468 -- <prod paths>` = `2 files changed, 632 insertions(+), 97 deletions(-)` | porcelain returns to only `?? CLEANUP_REPORT.md.save` before any acceptance run | **CLOSED-VERIFIED 2026-11-13 (measured at handoff)** — at HEAD `a53044f` the tree is clean for tracked paths: `git status --porcelain` lists only `?? CLEANUP_REPORT.md.save` (the operator's) and the three untracked `population_habit_*.npz` under `20261112-parity-repair/raw/third_model/`; no ` M` line, nothing staged. The two production files named above were committed in `67fcde8` |
| L20 | frontend bundle | `dist` is no longer the bundle the prior report recorded, and it was rebuilt mid-turn | frontend lane | T1: `index-NnjrdZLG.js` 392769 B sha256 `721d34d0…`; T2 01:39:31: `index-CFwf1H51.js` 396890 B sha256 `5730d4f7…`, `dist/index.html` sha256 `40d705a2…`; served shell now references `index-CFwf1H51.js` | `dist/index.html` mtime ≥ newest `*.ts{,x}` and served shell == dist | **CLOSED-VERIFIED 2026-11-13 (measured at handoff)** — deployed bundle is `dist/assets/index-8klMGIWd.js` **399 398 B** sha256 `43b2618c56bf2a355dda893c774d8160ab64fcbb1df88623d84159baea284d87`, `dist/index.html` sha256 `696c23fcaec29e809f997063e3a758afd3665859d397b9741e9c76fcc6ea0d04`; `curl :5173/` and `curl :5173/assets/index-8klMGIWd.js` return **the same two digests** (served == built); freshness holds (`dist/index.html` 09:04:34 ≥ newest `routes.ts` 09:02:53). `dist` is gitignored, so it is deployed-not-committed by design |
| L21 | native measurements | `reference_metadata_seen` is absent from run identity, so a run cannot say whether its reference image was in the gallery | runner/identity lane | `grep -rn reference_metadata_seen workbench/composition workbench/backend workbench/scripts workbench/tests` → **0 hits** | the field exists in run identity and mints new run ids for every existing run | CLOSED-VERIFIED 2026-11-13 as a DOCUMENTED GAP, not an identity defect (see the L21 note below) |
| L22 | diagnostics | CLVC dual-scope diagnostic for `7:3038`/`7:3790` | backend + frontend | `/api/composition/diagnostics/limn-dual-scope` exists; `limn_diagnostic.py` and `LimnScopeDiagnostic.tsx` are the two files now modified by another lane (L19) | diagnostic answers for both qids, one of them naming "not in this model's scope" rather than a rank | **CLOSED-VERIFIED 2026-11-13 (measured at handoff)** — `GET /api/composition/diagnostics/limn-dual-scope?qid=7:3038` → **200** `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: []`, `standard: limn-diagnostic-evidence/v2`; `GET /api/composition/diagnostics/clvc-dual-scope?qid=7:3790` → **200** `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: []`, manifest digest `60dac535…`; all four paths present in `app.openapi()`. Browser: 12/12 CLVC cells match `raw/clvc-dualscope/INVENTORY.md`, **0 mismatches**, and each cell renders the target as "ngoài cửa sổ Top-100" with its exact rank |
| L23 | tests | 20 pre-existing failures at the pin, identical set to `aea143bd` | test lane | `/tmp/sweep_cf6ba468.txt`: `20 failed, 3159 passed, 47 skipped, 350 warnings in 4037.65s`; 20 `FAILED` ids parsed this turn, 0 `^ERROR`; the handoff guard is one of them because the sweep ran **before** the pin advance | the set does not grow; each id has a written cause | **CLOSED-VERIFIED 2026-11-13 (re-measured at handoff)** — the authoritative sweep is at pin `82b0b3f`: **3283 passed / 8 failed / 47 skipped = 3338** collected, 3969.99 s, `^ERROR` 0, `Errno 28` 0, **0 new / 0 gone / 8 identical** vs `90f7b2c` (`SWEEP_82b0b3f.md` §3-§4; prior sweeps: `90f7b2c` 3275/8/47 at 3808.40 s, `e833eda` 3275/8/47 at 3608.26 s, `b4d25c70` 3272/8/47 at 3473.27 s). **12 of the 20 ids were closed by this campaign** (blip ×4, `tme_precision`, J10 ×7) and **8 remain**, each classified in `FINAL_ACCEPTANCE.vi.md` §Danh sách lỗi còn đỏ. The `3159/20` line above is the historical `cf6ba468` snapshot before the repairs |
| L24 | PAIR / web publication | PAIR was published on the site while its checkpoint mapping is unresolved: 17 models served, 4 exploratory rows listed with `mapping_status: EXPLORATORY_PUBLIC_MAPPING` beside 16 verified-looking rows, and a new PAIR job answered `404 model_not_found` (a lie — the entry is registered) | PAIR web-deferral lane | pre-fix live: `GET /api/models` → **17** records incl. `pair` (`reproduction_status: BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED`, `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, `strict_clean_checkpoint_available: false`, `pair_b1`/`pair_b2` both `runnable: false`); `GET /api/native-measurements` → **56** rows incl. 5 `pair` rows (`a804c2973c98c6de` blocker `BLOCKED_SOURCE_CLASSES_UNPUBLISHED` + `61ccfc4d63fb69eb`/`27a9b77d146b9b97`/`f4655ef992695f90`/`8cb65b73274a86f8` exploratory, r10 34.0350/35.0147/22.1564/20.0868); `POST /api/jobs {pair, pair_b1}` → `409 evaluation_blocked` (no statement that publication was refused by decision); `workbench/frontend/ModelInventory.tsx:16-22` `DEFERRED_MODELS` held only `csmcir`, `sprc`; `cir-defer-pair-on-web.patch` absent (home-wide `find`, `git log --all`, `git stash list` — 0 hits) | `/api/models` = **16**; `/api/native-measurements` = **51** rows with every counter recomputed from the visible rows; `POST /api/jobs {pair}` → **409 `model_deferred_from_web`** naming the deferral + an offline command; all 5 PAIR rows still `200` at `/api/native-measurements/{id}` with `mapping_status` intact and the blocker row still `BLOCKED_SOURCE_CLASSES_UNPUBLISHED`; 8 `B1/H_*` roll-ups still absent from the public table | CLOSED-VERIFIED (live, see the L24 note) |
| L25 | diagnostics | the CLVC dual-scope reader was implemented but its router was never included, so the live backend 404'd `/api/composition/diagnostics/clvc-dual-scope` while LIMN's route worked — dead code that looked like a feature | PAIR web-deferral lane (`main.py` ownership) | pre-fix: `from workbench.backend.main import app; [r.path for r in app.routes if 'diagnostics' in r.path]` → only the two LIMN paths; `curl :8000/api/composition/diagnostics/clvc-dual-scope` → 404; `grep -c include_router workbench/backend/main.py` → 0 | four diagnostic paths in `app.openapi()`; throwaway instance answers `200 DIAGNOSTIC_COMPLETE complete:true defects:[]` for both CLVC and LIMN; `workbench/tests/test_diagnostic_router_registration.py` fails when the `include_router` line is removed | **CLOSED-VERIFIED 2026-11-13 (re-measured at handoff on the SHIPPED stack, not only a throwaway)** — `GET /openapi.json` lists all four paths; both dual-scope routes answer **200** `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: []` on `:8000` after the redeploy (backend pid 2816638) |

| L30 | ranking/provenance | four Task-3 verifications pass: two-module source guard, condition advertisement, exact-rank-vs-window-position, aggregate-vs-artifact eligibility | — | (1) `da0a2162be0e159a1a456cec` shared hash matches disk (`7ec1e34d…`) but adapter hash `3b8e1bb5…` ≠ disk `7f150b6a…` → served `scoring_impl.source: unavailable`, `impl: unknown`; positive control `7b1fa850…` both match → `source: live, impl: matmul`; 15 more run-cases where only the adapter hash matches all return False. (2) `conditions_display` = exactly `['multimodal','text_only_black_image','clvc_image_encoder_only']`. (3) exact rank = `target_rank` (74330), window position = `top_results[].rank` (1…100), gallery index = `gallery_position` (74082); CLVC 12/12 cells `target_within_export_depth: false`, ranks 259…8702, depth 100. (4) 23/23 counted rows have a present `backed_by` with matching sha256; 33/33 source files present | all four hold | CLOSED-VERIFIED (measured) |
| L29 | native measurements | four counts (25 keys / 17 models / 23 rows / 27 duplicates / 18 unbacked) were used interchangeably, and "18 ids" conflated models with row keys | matrix/doc lane | `MODEL_MATRIX.md` Totals "ids carrying a native aggregate artifact (any id) **18**"; live census gives 25 / 17 / 23 / 27 / 18, with `verified: 0` | each count names its unit; none is quoted as "evaluated" | CLOSED-VERIFIED (doc-corrected — Totals split, "never quote as evaluated" line added) |
| L28 | composition conditions | `run_conditions()` cannot represent a declared-empty condition list | backend lane | `workbench/backend/composition.py:23-38`: `if not isinstance(declared, (list, tuple)) or not declared: return None`, so `[]` collapses to the pre-field case. Measured: `run_conditions({'conditions': []})` → `None`; `displayed_conditions({'conditions': []})` → `(multimodal, text_only_black_image, image_only_empty_text)`. No on-disk summary has an empty list (36/36 non-empty) | `()` must be distinguishable from `None`, or the docstring must stop claiming it is | **CLOSED-VERIFIED 2026-11-13 (measured at handoff)** — three states are pairwise distinct: `run_conditions({})` → `None`, `run_conditions({'conditions': []})` → `()`, names → the names; `absent is empty` is now **False**. `conditions_declaration` is served on **both** routes (`GET /api/composition/runs/da0a2162…` → `DECLARED`, and its `/queries` payload → `DECLARED`), and `da0a2162…` still serves exactly its triple `[multimodal, text_only_black_image, clvc_image_encoder_only]` (positive control). Code: `backend/composition.py:23-43` — `if not isinstance(summary, Mapping) or "conditions" not in summary: return None`, then `declared = summary["conditions"]` and `return tuple(...)`, so an empty list yields `()` rather than `None`. Locked by `workbench/tests/test_declared_empty_conditions.py` (**9 passed** in 1.46 s; all 9 named in `raw/ops-gaps/REPORT.md` §Task 1) |
| L27 | live service vs tree | the API on `127.0.0.1:8000` is a **stale snapshot**: pid 2162645 started `Thu Nov 12 23:22:41 2026`, no `--reload`, while six owned-elsewhere modules changed after it | backend/service lane (whoever runs the service) | `/api/runs/parity-gate` serves `models.hint.failures` = **2**; `evaluate_model('hint')` on the current tree gives **8** (adds `{dress,shirt,toptee}.metrics.eligible_denominators` and `.eligible_query_set_identical`). `/api/models` lists **17** incl. deferred `pair`; disk code via `web_deferred_models()` lists **16**; `/api/native-measurements` serves `models_with_metrics 17 / with_metrics 23 / files 39` vs disk-filtered `16 / 19 / 34`. `/api/models/publication` → 404 though `main.py:357` declares it | restart the backend, then re-measure; until then neither snapshot is quotable as current | **CLOSED-VERIFIED 2026-11-13 (measured at handoff)** — services restarted onto the new pin `Fri 2026-11-13 10:10:07 +07`: backend **2816638** (`KillMode=mixed`, `TimeoutStopUSec=30s`), frontend **2816640** (`KillMode=control-group`, `TimeoutStopUSec=5s`), both `NRestarts=0`; pid 2162645 gone. Every divergence above is resolved: `/api/models` **16** (was 17 incl. `pair`), `/api/native-measurements` **51** rows `with_metrics 19 / models_with_metrics 16 / files 34 / signposted_blockers 3` (was `23/17/39`), `/api/models/publication` **200** (was 404), and `/api/runs/parity-gate` now matches `evaluate_model('hint')` on disk at **2** failures each (`shirt`/`toptee.batch_1_vs_n.max_rel`) where the stale service had disagreed |
| L26 | native measurements | `dcnet` R@50: served vs quoted differ (two different runs) | native-measurements lane | endpoint `27.929730914548916 / 53.76075764301528` (`backed_by raw/native/dcnet.json`, `raw_metrics.r50 53.72723331879208`); matrix §2 quoted only the roll-up pair. The replay's `cross_check_previous_run` states the dress/toptee R@50 cells differ at the 4th decimal on CPU vs CUDA | the doc quotes the served pair and keeps the roll-up labelled as another run | CLOSED-VERIFIED (doc-corrected; no code change) |

## Lane notes — matrix and aggregate provenance (2026-11-13)

Owner: matrix/aggregate provenance lane. Files written: `MATRIX_RECONCILIATION.md` (new, this dir),
`20261112-parity-repair/MODEL_MATRIX.{md,csv}`, and three documents under
`20261109-native15-composition-limn/` (`EVIDENCE_INDEX.md`, `FINAL_REPORT.vi.md`,
`ENCODER_PARITY.vi.md`) — **documents only; none of its `g1_g2.*.json` artifacts touched**. Raw
payloads under `raw/matrix-provenance/`. Two tree snapshots at `01:28` and `02:28` record the moving
worktree; nothing was staged or committed.

**L14 and L15 are settled, and both verdicts are "the endpoint is right, the record was not."**
L14: the macro rule is `statistics.mean` over three category recalls (`validate.py:5,281-309`), the
replay computes it exactly (43.8016931215922), and the roll-up's 43.8182 averages two cells that
exist in no other file. L15: `raw/native/encoder.json` lacks `replay_type`, so it is structurally
neither a row nor a backing report — a stated predicate, not a gap.

**One defect is not mine and blocks any "current" citation: L27.** The backend serving :8000 was
started 2026-11-12 23:22 without `--reload`; six modules changed after that, so `hint`'s failure
count, the published-model count and the native counts are all pre-edit values. Route to whoever owns
the service.

## Lane notes — PAIR web deferral (2026-11-13)

Owner: PAIR web-deferral lane. Files: `workbench/registry/models.yaml` (PAIR entry only),
`workbench/backend/registry.py` (the `web_deferred_models` reader + flag validation — the entry's
flag is unreadable without it), `workbench/backend/main.py`,
`workbench/backend/native_measurements.py`, `workbench/frontend/ModelInventory.tsx`,
`workbench/frontend/scope.ts`, `workbench/frontend/data.ts`, `workbench/frontend/EvaluationPage.tsx`
(the `NativeMeasurementTable` notice only), `workbench/tests/test_pair_web_deferral.py`,
`workbench/tests/test_diagnostic_router_registration.py`. Nothing committed; staging is explicit.

### What "deferred" means, and what it deliberately does not

One flag on the registry entry (`web_publication: {status: DEFERRED, reason: …}`) plus one reader
(`registry.web_deferred_models`) and one publication filter per surface. The registry keeps all 17
entries: `scripts/evaluate_models.py --list` still prints `[OK] pair / pair_b1` and `pair / pair_b2`,
`--model pair --checkpoint pair_b1 --dry-run` still reports both blockers, and every artifact under
`artifacts/checkpoint_selection/20261106T002617+0700/pair_compat/` is untouched.

Counters before → after, all measured on a throwaway instance (`:8767`, this lane's app, no shipped
service restarted):

| surface | before | after |
|---|---|---|
| `GET /api/models` | 17 records, `pair` last | **16**, no `pair` |
| `GET /api/native-measurements` | 56 rows (`blocks_with_metrics` 50, `signposted_blockers` 4, `files` 39, `with_metrics` 23, `models_with_metrics` 17) | **51** rows (`46`, `3`, `34`, `19`, `16`) |
| `GET /api/benchmark-runs` | 13, no `pair` | 13, no `pair` (unchanged — it never carried one) |
| `GET /api/composition/runs` | 53, no `pair` | 53 (unchanged) |
| `GET /api/runs/parity-gate` | 3 models (`encoder`, `hint`, `limn`), no `pair` | unchanged |
| `GET /api/jobs` | 0 | 0 |

The `signposted_blockers` drop is 4 → 3 because PAIR's `pair_blocker.json` row is one of those four;
`with_metrics` 23 → 19 and `models_with_metrics` 17 → 16 drop by the four exploratory rows and their
model. Every one of the six counters is computed from the filtered list, so the header and the table
cannot disagree — that was the defect the decision named.

### The refusal, verbatim (`POST /api/jobs`)

```json
{"detail":{"error":"model_deferred_from_web","message":"pair is deferred from the web workbench: Chưa có mapping checkpoint chính thức: pickle được tạo bởi các class mà repo đã pin không có (BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED), nên chưa thể công bố là tái lập PAIR. The model is still registered and still runnable through the offline CLI; only queueing it as a website job is refused.","details":{"model_id":"pair","reason":"…same reason…","offline":{"command":"workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py --model pair --checkpoint pair_b1 --protocol fashioniq_val_split --dry-run","list_command":"workbench/envs/ctl312h/bin/python workbench/scripts/evaluate_models.py --list","protocol":"fashioniq_val_split","checkpoint":"pair_b1","callable":[…],"registry_entry":"web_publication.status = DEFERRED (reason recorded in models.yaml)","artifacts":["pair_b1","pair_b2"]},"web_publication":"DEFERRED"}}}
```

`--list` is a SEPARATE key because argparse refuses `--list` together with `--model`; the first draft
of this payload glued them into one string and named a command that fails when pasted.

### Deep-link probe — no route resurrects PAIR

There is **no** per-run web route that can render a PAIR record, and this was measured rather than
assumed. The frontend's whole URL surface is `parseLocation` over
`page|tab|area|view|protocol|subset|comp_subset|run|comp_run|runs|query|comp_qid|condition|comp_cond|outcome|comp_outcome|category|comp_cat|order|comp_order|offset|comp_offset|q|k|imported|query_state|preset`
plus one direct read, `?view=native&model=<id>` in `NativeMeasurementTable`. PAIR has no composition
run, no index run and no aggregate report, so `run`/`runs` cannot name it:

- `GET /api/composition/runs/pair` → 404 `composition_run_not_found`; `/…/pair_b1` → 404; `/…/Pair_run_not_real` → 404
- `GET /api/runs?protocol_id=fashioniq_val_split` → `[]` (200)
- `GET /api/queries?run_id=pair` → 400 `run_not_found`
- `/api/native-measurements/{id}` → 200 for PAIR (by design: the audit path)

The one deep link that does reach PAIR is `?view=native&model=pair`, which resolves against the
native catalogue. That page now renders `data-testid="native-deferral-notice"` from
`scope.webDeferralNotice(model, deferred_models)` — the reason from the registry, plus the sentence
that the record is still readable by `measurement_id`. Without it the page would highlight nothing and
read as "PAIR has no measurements".

### Still retrievable (the offline-audit proof)

```
$ curl -s :8767/api/native-measurements/61ccfc4d63fb69eb | jq '{model_id,mapping_status,metrics,web_publication}'
{"model_id":"pair","mapping_status":"EXPLORATORY_PUBLIC_MAPPING","metrics":{"r10":34.03503453772374,…},
 "web_publication":{"status":"DEFERRED","scope":"listing only: this record is served by id for offline audit","exploratory":true}}
$ curl -s :8767/api/native-measurements/a804c2973c98c6de | jq '{blocker,metrics}'
{"blocker":"BLOCKED_SOURCE_CLASSES_UNPUBLISHED","metrics":null}
```

Nothing was relabelled: the four exploratory rows keep `EXPLORATORY_PUBLIC_MAPPING`, the blocker row
keeps its verdict, and the 8 `B1/H_*` roll-ups (`dd42f32772c4e5cf`, `bc0db3353d0706ff`,
`573ac9482e55918c`, `c10e512c3a92a4c0`, `5913c767a8e7d4a2`, `b771dca51e180479`, `606fa3cc698a956a`,
`d780cbab298c45b9`) still carry `rollup: true`, `counted: false`, `protocol_id: null` and are still
invisible in the public table — as they were before, for the reason they always were.

### CSMCIR / SPRC scope decisions — confirmed against current assets

Both decisions hold; nothing was changed.

**CSMCIR** — `fashioniq_tuned_clip_best.pt` present at
`workbench/artifacts/checkpoints/csmcir/fashioniq_tuned_clip_best.pt` (**2 720 612 859 B**), sha256
matches the Hub local-cache blob under
`~/.cache/huggingface/hub/models--peng12138--CSMCIR/blobs/c8812071…` (same 2 720 612 859 B) — the
"two locations" in the brief. A valid native evaluation with the right checkpoint and scope is
already published: `GET /api/benchmark-runs` carries
`20261107T054933277898Z_csmcir_fashioniq_f03887d2` and `…f564570e`, both
`checkpoint_id: fashioniq`, `protocol_id: fashioniq_original_split`,
`verification.state: certified`, `is_research_evidence: true`, `reasons: []`; the native catalogue
serves `6812926dbe6394ce` (`r10 57.05073873202006 / r50 77.29859352111816 / mean 67.17466612656911`,
`artifact_backed: true`, `counted: true`). CSMCIR's composition stays deferred (0 CIRCUS rows) — which
is what `DEFERRED_MODELS.csmcir` already says. No new run started.

**SPRC** — `sprc_fiq_vitl.pt` present at
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/sprc/sprc_fiq_vitl.pt`
(**1 459 756 541 B**, `official_sha256_match: true`); `sprc_fiq.pt` **absent** home-wide (`find` → 0
hits), matching the registry's `sprc_fiq_unresolved` / `NO_CLEAN_CHECKPOINT` declaration and the
`checkpoint_downloaded: false` the API reports. A native evaluation with the ViT-L checkpoint is
published: `GET /api/native-measurements` row `6811145a4cff98ee`, `checkpoint_id: sprc_fiq_vitl`,
`protocol_id: fashioniq_original_split`, `r10 51.038571198781334 / r50 73.32633932431538 / mean
62.182455261548355`, `artifact_backed: true`, `counted: true`, source
`checkpoint_selection/20261106T002617+0700/measurements/sprc_vitl_run1.json` (plus its uncounted
roll-up `f613c9ba91d47526` in `selection_decisions.json`). No aggregate report and no CIRCUS row
exist for SPRC, so "native only, ViT-L, FashionIQ original split" is exactly what is published. The
existing `DEFERRED_MODELS.sprc` ("chưa tải checkpoint") is now imprecise — the checkpoint IS present
and measured — but correcting operator copy is not part of this decision's diff and the string is not
wrong about the *other* variant, so it is reported here rather than changed.

### Tests (pre-fix failures, then green)

`workbench/tests/test_pair_web_deferral.py` — **10 failed, 3 passed before the change** (import error
`web_deferred_models` ×3, `exclude_model_ids` TypeError ×2, wrong refusal shape, missing
`native_measurement` ×2, `webDeferralNotice is not a function`, offline-artifact assertion); the 3
that passed are the negative controls (vocabulary closure, roll-ups still hidden, a published model
not refused). After: **13 passed**.

`workbench/tests/test_diagnostic_router_registration.py` — **2 failed, 1 passed** with the
`app.include_router(diagnostic_router)` line removed (proved by deleting and restoring it); **3
passed** with it.

One existing test changed rather than added: `test_native_measurements_census.py::
test_the_endpoint_serves_the_whole_catalog` asserted `"pair" in models` on the served catalogue. It
now asserts the disclosure instead — `deferred_models["pair"]` is non-empty and `"pair" not in
models` — because the old assertion encoded exactly the publication the decision removed.

Regression sweep at this state, from the repo root with the control interpreter:
`test_e2e_release_chain`, `test_api_models_cache_staleness`, `test_http_api_layer`,
`test_frontend_catalog`, `test_frontend_routes`, `test_frontend_staleness`, `test_frontend_requests`,
`test_backend_config_roots`, `test_checkpoint_selection`, all three native suites, and both new files
→ **190 passed in 159.53 s**. Nothing skipped, nothing xfailed.

Frontend bundle after the label change: `dist/assets/index-B6AFoKZN.js` (398 622 B, sha256
`2947f0c1…`), `dist/index.html` (271 B, sha256 `65beed07…`). Deployment **pending** — no shipped
service was restarted; `:5173` still serves the previous bundle. `:8000` and `:5173` both answered 200
after the throwaway instance (`:8767`) was killed and the port confirmed free.

### Ownership extension (2026-11-13, second pass)

Three items were assigned after the first pass; all three are done, still uncommitted.

**1. `OverviewPage.tsx` conflated the published count with the registry size.** Fixed by adding a
served field rather than by arithmetic in the client: `/api/models` stays a bare list (a count
describing what a list OMITS cannot ride on the list), and a new
`GET /api/models/publication` serves

```json
{"registry_count": 17, "published_count": 16, "published_model_ids": ["…16 ids…"],
 "deferred_count": 1, "deferred": {"pair": "<registry reason>"}}
```

`registry_count` is `len(registry["models"])` (the whole document) and `published_count` is the same
predicate `/api/models` filters with, so the two endpoints cannot diverge. The Overview card now
prints **both**, each under its own label (`công bố trên web` / `trong registry`), and the inventory
footnote reads both fields from the same response instead of `models.length + deferred.length`.

Invariant test (`test_publication_counts_agree_with_the_served_list`, no literal 16 or 17 anywhere):
`published_count == len(/api/models)`, `registry_count == len(registry["models"])`,
`registry_count == published_count + deferred_count`, `published_model_ids` == the served ids. A
second test (`test_deferred_count_is_derived_from_the_flag_not_from_a_constant`) adds a synthetic
deferral to a deep copy and asserts every number moves with no code change. Removing the
`/api/models/publication` route fails the first with `AttributeError` (proved by deleting and
restoring the route).

**2. `DEFERRED_MODELS.sprc` was false about the checkpoint that was run.** Corrected, and CSMCIR was
checked rather than assumed. Verbatim now:

```ts
csmcir: "FashionIQ native đã đo (row 6812926dbe6394ce, đã chứng thực) · CIRCUS composition chưa có lượt nào",
sprc:   "Đã đo native bằng ViT-L (row 6811145a4cff98ee, đã chứng thực) · thiếu checkpoint ViT-g của paper (sprc_fiq.pt)",
```

CSMCIR's old copy ("FashionIQ native đã đo · CIRCUS composition tạm gác") was accurate on both
counts and only lacked the row id it referred to, so it was made specific, not corrected. SPRC's was
wrong: `sprc_fiq_vitl.pt` is present (1 459 756 541 B) and carries a counted, artifact-backed native
row; the absent file is the paper's ViT-g (`sprc_fiq.pt`, backbone `pretrain`,
`CHECKPOINT_MAPPING_UNVERIFIED`, null digest, 0 hits home-wide).

**3. The refusal payload repeated the registry reason inside `message`.** De-duplicated:
`message` is now one renderable English sentence naming the model, the deferral and where the offline
path lives; the registry's prose appears exactly once, in `details.reason` (verbatim), and
`details.scope` states that queueing is all that is refused. Read back as a client would — a UI can
print `message` alone, and no field has to be string-spliced. `--list` remains a separate
`list_command` key because argparse rejects it alongside `--model`.

Measured on a throwaway instance (`:8768`, killed; port confirmed free):

| check | result |
|---|---|
| `GET /api/models/publication` | 200, `registry_count 17`, `published_count 16`, `deferred_count 1`, ids match `/api/models` |
| invariant | `published == len(models)` true, `registry == published + deferred` true |
| `POST /api/jobs {pair, pair_b1}` | 409 `model_deferred_from_web`, reason once | 
| `GET /api/native-measurements` | 200, 51 rows, 0 `pair` rows |
| `GET /api/native-measurements/61ccfc4d63fb69eb` | 200, `EXPLORATORY_PUBLIC_MAPPING`, r10 34.03503453772374 |
| `GET /api/composition/diagnostics/clvc-dual-scope` | 200 `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: []` |

Tests at this state: the six suites above (`test_pair_web_deferral` now 15, `test_diagnostic_router_registration` 3, three native/registry suites, three frontend suites) → **80 passed in 9.89 s**.

Bundle rebuilt for the copy and card changes: `dist/assets/index-CZlMxyEn.js` (399 247 B, sha256
`86ecc648…`), `dist/index.html` (271 B, sha256 `d3544373…`). Deployment still **pending** — no
shipped service restarted.

## Lane notes — runner identity (2026-11-13)

Owner: runner identity lane. Files: `workbench/composition/runner.py`,
`workbench/scripts/composition_eval.py`, `workbench/backend/composition.py`,
`workbench/docs/reviews/20261111-master-audit/scripts/m03/publish_derived_runs.py`, and the two
lane test files plus one new gallery test. Nothing committed; staging is explicit.

### L03 — one `gallery_identity_sha256` for one ordered gallery

Reproduced (2026-11-13, `workbench/envs/ctl312h/bin/python`, one loaded release, `gallery_count`
74381): plan branch `0bfd43b9cc0162885f06c60cfc751602ee9f3bed2fb4d206ffcebe3f7b443462`, runner
`1db68c59d96f014cdbecf936e108653de98abef0bd9a98a8abc2fe08b5dec4b4`, `equal: False`.

Fixed by reusing `_gallery_fingerprint` at BOTH sites (`_plan` and the `if specs else` blocked
report). **Payload deliberately NOT changed**, because a written artifact re-verifies its own digest
on read (`backend/composition.py:_candidate_reader` recomputes `gallery_digest` and refuses a
mismatch): changing what `_gallery_fingerprint` hashes would have made every pre-fix artifact
unreadable, so only the call sites moved. Serialized inputs are pinned:
`ordered_gallery` keeps release order ON PURPOSE (positions index into it), `candidate_names` is
`dict(sorted(...))` and `_digest` uses `sort_keys=True`, so no payload KEY order is significant —
verified, nothing needed making explicit beyond what was already there.

Acceptance: `workbench/tests/test_gallery_identity_one_formula.py` builds plan, blocked and completed
for ONE gallery in one test, asserts a single digest, asserts the digest CHANGES when the gallery is
reordered (so it is not a constant), asserts stability across two calls in one process, and pins the
payload so pre-fix artifacts still verify.

Historical digests: `0bfd43b9…` appears in committed audit transcripts and those were NOT rewritten.
Documents whose tense/provenance was corrected:
`raw/verify/104-clvc-full-image-only.md` §0.1, `raw/verify/90-circus-composition-runs.md`,
`20261112-parity-repair/BACKEND_DEFECTS.md` (defect 3),
`20261112-parity-repair/GALLERY_DIGEST_PATCH_SPEC.md` (status header),
`20261112-parity-repair/PARITY_AND_EVAL_REPAIR.vi.md`, and this ledger's L03 row.

### L04 — routing status: "does not route" vs "could not scope"

Reproduced from the immutable artifact
`summary/tgcir_tgcir_archive_unresolved_7b1fa85007bffdab931f7f19/full.json`: `still_unscoped: 6003`,
`adapter_routed: 0`, 6003 `unscoped_qids`, `category_scopes: [null]`, and
`condition_status.*.unscored_count == 0` for all three conditions (`sum == 0`) on a run that scored
all 6003. Cause reproduced directly: `_scope_for_query(category_scopes=(None,)) -> (None, None)`, so
`sum(scope is None ...)` counted a RESOLVED `global` scope as a failure.

Fix (`runner._scope_resolution_record`, one function so the block is replayable without a GPU):
unresolved is derived from the RETURNED REASON. `(None, None)` = resolved `global`;
`(None, "<reason>")` = genuinely unscopable. Every existing key kept.

**`adapter_routed` semantics chosen: the NARROW meaning** (a query the adapter placed on a NAMED
matrix), with a new `global_scoped` beside it. Consequence numbers for TG-CIR:
`adapter_routed: 0`, `global_scoped: 6003`, `still_unscoped: 0`, `unscoped_qids: []`,
`category_agnostic: true` (measured by replaying the real release through the new path), plus
`loader_scoped: 5884` as the separate DESCRIPTIVE count — the loader assigned a garment to 5 884
queries, but a category-agnostic adapter ignores it when picking a matrix, so all 6 003 are
`global_scoped`. `loader_scoped` overlaps rather than partitions: the partition is
`adapter_routed + global_scoped + still_unscoped == len(queries)` (0 + 6003 + 0 == 6003). The alternative (widening `adapter_routed` to every placed query) would have rewritten a
key `derive_routing_cohort_gap.py` already replays against, so it was rejected on those grounds, not
on convenience.

`still_unscoped == sum(condition_status[*].unscored_count)` now holds for TG-CIR (0 == 0).
`category_agnostic` is also written into the DERIVED block by `publish_derived_runs.py`, copied from
the parent, so a derived TG-CIR row cannot read as a routing failure.

`workbench/tests/test_derived_subset_status.py:131-145` stays green: LIMN IS per-category and its two
unscoped qids carry real reasons, so the reason-based count is still 2. Proven by replaying the REAL
resolver over the REAL release in `test_the_reason_based_count_still_finds_limns_two_real_failures`
(`still_unscoped == 2`, `unscoped_qids == ["7:3038","7:3790"]`).

**`7b1fa850…` is NOT rewritten.** Its summary is immutable history and keeps the old, misleading
block; the API serves that block verbatim (asserted), because the number was wrong where it was
MEASURED and the backend must not paper over it. No document may quote "6003 unscoped" as a routing
failure. The forward proof is a replay of the same release through the new path, plus a fixture run.

### L21 — `reference_metadata_seen` / `category_routing_inputs`: DECISION — documented gap, not an identity defect

Evidence (2026-11-13): the routing fact is ALREADY covered by run identity. `identity` includes
`query_scope_resolution` (`_scope_resolution_identity`, qid-keyed) and `adapter_source_hashes` for the
five methods including `query_scope`'s contract, so a run whose routing read the edit caption is a
DIFFERENT run id from one that did not. Measured on the real release: caption-aware routing vs
loader-only routing differ on exactly the 117 queries and produce different scope digests
(`3a1c5a6763154870536f3586491d5bea9d87b73a9836363dc298d2accdd34d87` vs
`8a5dd517595eb4883d8b3174f410f87980fbc612418231e2be4e0e3847cd1c19`), and the full run-identity
digest differs too. Adding `reference_metadata_seen` would therefore not distinguish anything the
identity does not already distinguish; it would only mint new ids for every existing run, which is
exactly what §7.6 reverted that proposal for.

Conclusion: a DOCUMENTED GAP in the SUMMARY's self-description (`category_routing_inputs`, §7.7), not
an identity defect, so it needs no versioned change and no migration. The declare-side is already
shipped: `IMAGE_ENCODER_ONLY_LEAKS_SCOPE` / `IMAGE_ENCODER_ONLY_LEAKS_CAPTION`
(`workbench/composition/model_clvc_net.py`) name both routing inputs and the 117. Not retrofitted into
`da0a2162…`'s finished summary, per the spec.

### Lane verification (measured 2026-11-13, `workbench/envs/ctl312h/bin/python`)

| command | result |
|---|---|
| `pytest workbench/tests/test_gallery_identity_one_formula.py` | **3 passed** |
| `pytest workbench/tests/test_limn_scope_runner.py` | **19 passed** |
| `pytest workbench/tests/test_derived_subset_status.py` | **8 passed** |
| the three lane files together | **30 passed** |
| `pytest workbench/tests/ -k "scope or gallery or composition"` | **433 passed, 1 skipped, 4 failed** in 152 s |
| `pytest workbench/tests/test_circus_cli.py test_derived_lineage_r09.py test_composition_scope_routing_cohort.py` | **34 passed** |

The 4 failures are the `test_circus_adapters_blip.py` `[conesep]`/`[ptha_mtst]` × `reuse_and_cleanup`/
`failure_uses_shared_error` ids, which are 4 of the 20 pre-existing ids listed verbatim in
`20261112-parity-repair/SWEEP_cf6ba468.md` ("blip ×4"); the failure text is
`AttributeError: 'Adapter' object has no attribute 'batch_size'`, untouched by this lane. No new
failure appears in the keyword selection.

`workbench/tests/test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`
fails, and it fails for the PIN, not for these changes: `git diff --name-only cf6ba468 -- .
':(exclude)workbench/docs' ':(exclude)workbench/tests'` lists 9 production files changed by several
lanes (`limn_diagnostic.py`, `main.py`, `native_measurements.py`, `parity_gate.py`, `registry.py`,
`runner.py`, `frontend/LimnScopeDiagnostic.tsx`, `registry/models.yaml`, `composition_eval.py`), of
which this lane owns 2. It was already failing before this lane started (`00-state/01-git.txt`
records the same two-file production diff at 01:39) and it can only be cleared by advancing
`TESTED_SOURCE_SHA` after a stationary sweep — owned by the lead, deliberately not done here.

**Deployment is pending.** No service was restarted (constraint) and no artifact was rewritten, so the
RUNNING backend still serves `7b1fa850…`'s pre-fix block and the plan branch still needs a code
deploy to print `1db68c59…`. Nothing here is committed.

### Lane follow-up — call-site proof and served payloads (2026-11-13)

**(1) The block is built from the SAME `_scope_for_query` returns, in one place.** `runner.py:1930`
computes `assignments` once, immediately after `scopes` is validated:

```
1930: assignments = [_scope_for_query(adapter, query, data.candidate_names) for query in queries]
1931: query_scopes = [scope for scope, _reason in assignments]
1932: scope_errors = {query.qid: reason for query, (_scope, reason) in zip(queries, assignments) if reason}
```

`assignments` is the ONLY call site; `grep -n "_scope_for_query" workbench/composition/runner.py`
returns the definition (`:547`), that line, and `_scope_resolution_record`'s docstring. There is no
second scope computation anywhere in `workbench/composition` or `workbench/backend` (adapters declare
`category_scopes` and implement `query_scope`, but the resolution is the runner's). Every consumer
derives from the same list: `query_scopes` → `_scope_resolution_identity` (`:1964`, the run identity),
`:2143` backfill, `:2222-2254` the encode/score partition, `scope_errors` → `condition_status`
`unscored_count` (`:2379`, `:2394`, `:2412`), and the summary block at `:2560-2561`
(`_scope_resolution_record(adapter, queries, scopes, assignments)`).

Proven at runtime by wrapping `_scope_for_query`, recording the returns, and rebuilding the block from
those recorded returns with the runner's own function — on two live fixture runs through
`composition_eval.main`:

| fixture | `_scope_for_query` calls | summary block | rebuilt from the returns | identical |
|---|---|---|---|---|
| `fixture3d` (agnostic) | 4 | `adapter_routed 0, global_scoped 4, still_unscoped 0, unscoped_qids []` | same | **yes** |
| `scoped` (per-category) | 4 | `adapter_routed 3, global_scoped 0, still_unscoped 1, unscoped_qids ["q2"]` | same | **yes** |

Partition `adapter_routed + global_scoped + still_unscoped == len(queries)` held in both (4 and 4).
Per-condition identity `still_unscoped == unscored_count` held in both (0/0 and 1/1). **The sum form
`still_unscoped == Σ condition_status[*].unscored_count` is NOT a general invariant** — it is only
equal when the value is 0 (there `Σ = 3` against `still_unscoped = 1`, because a query is unscopable
in every condition). The honest statement is the per-condition one; for TG-CIR both forms read 0.

**(2) Served payloads, through the real handler `M.composition_run` (`GET /api/composition/runs/<id>`).**

Run written BEFORE the change — `7b1fa85007bffdab931f7f19`, served **verbatim**, keys
`['adapter_routed', 'loader_scoped', 'resolver', 'still_unscoped', 'unscoped_qids']`:
`still_unscoped 6003`, `adapter_routed 0`, `loader_scoped 5884`, `resolver null`,
`unscoped_qids` 6003 entries, `run_id 7b1fa85007bffdab931f7f19`, `execution_status COMPLETED`,
per-condition `unscored_count` all 0, and **no** `category_agnostic` / `global_scoped`.

Run written by the fixed path (TG-CIR block replayed from the real release, written as a summary, then
served) — keys `['adapter_routed', 'category_agnostic', 'global_scoped', 'loader_scoped', 'resolver',
'still_unscoped', 'unscoped_qids']`: `category_agnostic True`, `still_unscoped 0`, `global_scoped 6003`,
`adapter_routed 0`, `loader_scoped 5884`, `unscoped_qids []`, and `served == written` byte-for-byte.

**Correction to the earlier lane note:** the TG-CIR consequence number is `global_scoped: 6003`, not
5 884. `loader_scoped` is the 5 884 — it is DESCRIPTIVE and OVERLAPS `global_scoped` (the loader
assigned a garment to 5 884 queries and this adapter ignores it when picking a matrix), so it does not
partition with the other two. Both numbers are now stated correctly in
`20261112-parity-repair/SCOPE_RESOLUTION_PATCH_SPEC.md`, `PARITY_AND_EVAL_REPAIR.vi.md` and this
ledger.

**(3) `grep -rn "6003"` over both review directories — every markdown hit judged.**

| hit | judgement |
|---|---|
| `BACKEND_DEFECTS.md:286-287` | historical — the artifact's recorded values, labelled as the pre-fix block |
| `BACKEND_DEFECTS.md:298` | prohibition — "no document may quote 6003 unscoped as a routing failure" |
| `CLVC_DIAGNOSTIC.md:58,155` | canonical-cohort sense (6001 scored of 6003), not routing |
| `ENCODER_RELATION.md:262` | cohort size, not routing |
| `MODEL_MATRIX.md:235,244,245,247,434,440,531` | cohort size (6003-query cohort), not routing |
| `PARITY_AND_EVAL_REPAIR.vi.md:450` | canonical cohort 6001/6003, not routing |
| `PARITY_AND_EVAL_REPAIR.vi.md:624` | historical — quoted as the artifact's recorded block, then marked fixed at `:1053` |
| `PARITY_AND_EVAL_REPAIR.vi.md:879` | cohort size, not routing |
| `PARITY_AND_EVAL_REPAIR.vi.md:1053` | corrected — states the artifact keeps 6003 as history and the new numbers |
| `PREMISES.md:388` | release row count, not routing |
| `PREMISES.md:670,682` | canonical cohort (6001/6003, coverage), not routing |
| `PREMISES.md:695,698` | TG-CIR `COMPLETED` over 6003 queries, not routing |
| `SCOPE_RESOLUTION_PATCH_SPEC.md:26-27` | historical — the pre-fix block being quoted as the defect |
| `SCOPE_RESOLUTION_PATCH_SPEC.md:36` | historical — "read literally the block SAYS 6003 could not be scoped", naming it as the defect |
| `SCOPE_RESOLUTION_PATCH_SPEC.md:7,107,111,112` | corrected — `global_scoped: 6003` in the fixed shape |
| `SCOPE_RESOLUTION_PATCH_SPEC.md:140` | historical — the pre-fix value a new test would have failed on |
| `LEDGER.md:18` | historical — the L04 defect statement, row state CLOSED-VERIFIED |
| `LEDGER.md:76-79` | historical — the reproduced pre-fix observations |
| `LEDGER.md:88,93` | corrected — `global_scoped: 6003` in the fixed shape |
| `LEDGER.md:108` | prohibition |
| `raw/clvc-dualscope/IMPLEMENTATION.md:116`, `INVENTORY.md:333,337,338,340,388,408` | canonical-cohort sense |

No hit reads as a current routing failure. Two JSON evidence files under
`raw/matrix-provenance/` are **verbatim API captures** and keep `still_unscoped: 6003` for the ten
pre-fix TG-CIR runs (`7b1fa850…`, `4901da13…`, `5dee7425…`, `9b806c22…`, `9f115afe…`, `a4f786d8…`,
`acc9b20e…`, `c6eb9d57…`, `cd1a696d…`, `ce89a9a2…`, `f461f391…`, `f54a7745…`); they are a transcript of
what the API served, not a claim about the code, and are not rewritten.

**(4) The four `test_circus_adapters_blip.py` failures are baseline, id for id.** Measured now:
`4 failed, 14 passed`, and the ids are exactly
`test_resident_worker_reuse_and_cleanup[workbench.composition.model_conesep-conesep]`,
`test_resident_worker_reuse_and_cleanup[workbench.composition.model_ptha_mtst-ptha_mtst]`,
`test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]`,
`test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_ptha_mtst-ptha_mtst]`
— all four appear verbatim in `20261112-parity-repair/SWEEP_cf6ba468.md`'s committed 20-id set
(lines 68-71), same message `AttributeError: 'Adapter' object has no attribute 'batch_size'`.

## §Handoff

The campaign's final handoff is written in this directory, and it is the entry point for a reader
who has not followed the lanes:

* `FINAL_ACCEPTANCE.vi.md` — verdict **COMPLETE**; handoff SHAs/pin/bundle/service identities; the
  five acceptance rows with the command that proves each and its measured result; the exact counts
  split so none is quoted as another; the 8 red sweep ids and the 19 gate failures with their
  classes; the 17 in-campaign repairs with pre-fix evidence; nine blockers each with a verbatim
  error and the next runnable command; reproduction, continue and rollback commands; and the
  evidence limits stated plainly.
* `MODEL_MATRIX_ACCEPTANCE.csv` — one row per registry model (17), derived from
  `raw/parity-coverage/{executability,gate,pairs_on_disk}.json`, `raw/native/*.json`,
  `workbench/artifacts/composition/**/full.json` and the live `/api/native-measurements` census.

**Twelve rows were corrected to the measured final state at handoff, each dated in its own state
cell; their pre-fix evidence columns are unchanged history.** Marker `2026-11-13 (… at handoff)`
means the state cell was re-derived from a measurement taken after the campaign's code landed:

| row | old state cell | new state cell |
|---|---|---|
| L08 | `CLOSED-VERIFIED (live)` — "returns 3 ids" | `_discover_models` returns **7** (5 paired + 2 `EVIDENCE_MISSING`) |
| L09 | `UNVERIFIED — settle with jq …` | **CLOSED** — `relation_state.sha256 20db0c27…`, `matches_pinned_digest: true`, both sides |
| L10 | `OPEN-VERIFIED (live counts 9/2/3)` — "14 mandatory failures" | **OPEN** — **19** (9/2/2/3/3), 0 absent-evidence, 0 derived-invariant; still the open decision |
| L11 | `UNVERIFIED — settle with jq …` | **CLOSED** — 5 rows / 2 counted, both macros named; `batch_size` keys the run identity, not the served native row |
| L18 | `CLOSED-VERIFIED (measured)` at `b4d25c7`/`cf6ba468` | **CLOSED** — pin `90f7b2c…`, guard **8 passed**, invariant exit 0 at `a53044f` |
| L19 | `OPEN-VERIFIED (live, new this turn)` | **CLOSED** — tracked tree clean; only the operator's file and three untracked `.npz` |
| L20 | `OPEN-VERIFIED` — `index-CFwf1H51.js` | **CLOSED** — `index-8klMGIWd.js` 399 398 B `43b2618c…`, served == built |
| L22 | `UNVERIFIED — settle with curl …` | **CLOSED** — both diagnostics 200 `DIAGNOSTIC_COMPLETE`; 12 cells, 0 mismatches |
| L23 | `CLOSED-VERIFIED (log re-parsed)` — "20 / 3159" | **CLOSED** — sweep at pin 3283/8/47 = 3338; 12 of the 20 closed, 8 remain classified |
| L25 | `CLOSED-VERIFIED (throwaway instance)` | **CLOSED** — four paths in the shipped `openapi.json`; both routes 200 on `:8000` |
| L27 | `OPEN-VERIFIED → backend/service lane` | **CLOSED** — restarted 12:14:56 at pin `82b0b3f`, pid 2946713/2946715; gate now matches `evaluate_all()` |
| L28 | `OPEN-VERIFIED (latent)` | **CLOSED** — `None` / `()` / names distinct; `conditions_declaration` on both routes; 9 tests |
| L31 | `OPEN-VERIFIED (cost)` — parity gate 4.37 s / 480.6 MiB per request | **CLOSED** — `_component_scope_check` no longer builds a per-component dict and `evaluate_all` memoises on an artifact stamp (`82b0b3f`). Live HTTP: cold **3.07 s**, warm **6.3 / 7.3 / 6.6 ms**; in-process cold 2.83–3.06 s, warm **1.45–1.53 ms**; peak RSS 480.6 → **270.4 MiB**; parity page renders its table in **0.25 s** with the `parity-gate` request at **132 ms**. Payload md5 `da701905419b8b9833966f637ac521a8` unchanged across old/uncached/cold/warm, and live == in-process |

L02 stays `OPEN-VERIFIED` because it genuinely is: `grep -c WORKBENCH_PAIR_PYTHON
workbench/config/workbench.env` → **0**, and `test -e workbench/envs/pair-adapted/bin/python` → exit
**1**. L02 and L10 are the two rows whose acceptance check cannot be satisfied by code in this repo
(the first needs the operator's interpreter, the second is a standard-level decision), and both are
reported as such in `FINAL_ACCEPTANCE.vi.md` §Blocker rather than marked closed.
