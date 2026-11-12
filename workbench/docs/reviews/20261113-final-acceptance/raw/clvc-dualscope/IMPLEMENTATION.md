# CLVC dual-scope diagnostic — implementation report

Lane: CLVC reader generalisation + UI + browser verification. Branch
`research/cir-failure-workbench`, HEAD `b4d25c7`. **Nothing committed. No GPU used.**
`workbench/backend/main.py` was **not** edited — the one line it needs is in §7.

## 1. Files changed

| file | change |
|---|---|
| `workbench/docs/reviews/20261113-final-acceptance/raw/clvc-dualscope/evidence/` | **deleted** — it was a byte-identical duplicate of the authoritative artifact. See §1a. |
| `workbench/backend/limn_diagnostic.py` | **generalised**. One `DiagnosticModel` descriptor per model (`LIMN`, `CLVC`, `MODELS`); the shared validation body now takes the descriptor. Adds `router = APIRouter()` / `diagnostic_router` with the two CLVC endpoints. LIMN's payload is unchanged (§2). |
| `workbench/frontend/LimnScopeDiagnostic.tsx` | model selector, `{model}-dual-scope` fetch + identity-guarded probe, reference image / raw caption / **effective `model_text`**, reference self-match labelling, `ablation` badge, cohort provenance, defects surface. |
| `workbench/tests/test_clvc_diagnostic.py` | **new**, 18 tests (§6). |
| `workbench/docs/reviews/20261113-final-acceptance/raw/clvc-dualscope/**` | `IMPLEMENTATION.md` (this), `browser-{clvc,limn}-12-cells.json`, `browser-vs-inventory.json`, `screenshots/` (26 PNG), `baseline-limn/`, `scripts/browser_12_cells.py`. |
| `workbench/frontend/dist/` | rebuilt bundle (§8). |

Not touched: `main.py`, `parity_gate.py`, `gate.ts`, any parity artifact, the registry YAML, the
shipped CLVC/LIMN evidence bytes. (`git status` shows `parity_gate.py` modified — that is the PAIR
lane's in-flight work, present before this phase began; verifiable as an added `hashlib`/`struct`
import and a `COMPONENT_COVERAGE` edit, none of it mine.)

## 1a. One authoritative copy — the duplicate is gone

The seven evidence files existed in two places: the authoritative
`workbench/docs/reviews/20261112-parity-repair/raw/clvc_diag/` (where
`CLVC_DIAGNOSTIC.md` §5 and the producer name it) and a copy this lane had made under
`raw/clvc-dualscope/evidence/`.

**sha256 verification — all seven identical, byte for byte:**

| file | `20261112-parity-repair/raw/clvc_diag/` | `20261113-final-acceptance/.../evidence/` |
|---|---|---|
| `clvc_dual_scope.json` | `00baa12f6ca054dd2c9c7a4eacdd3bd4b74ad257342cbbc1e5c0204bdd45e214` | **same** |
| `clvc_dual_scope.shirt.multimodal.jsonl` | `c3ad1ce11b9a22bbc5837e5094b71788601312364e9d2379d60c69a55a301402` | **same** |
| `clvc_dual_scope.shirt.text_only_black_image.jsonl` | `60f027209433deb109edbb412fcb8719dbbe9224ad54ced47c7f1ddd71eca27d` | **same** |
| `clvc_dual_scope.shirt.clvc_image_encoder_only.jsonl` | `aa840e8682a31f7a7e27f29f8261ed6452b476071c775c9178c6d2d35f11a968` | **same** |
| `clvc_dual_scope.toptee.multimodal.jsonl` | `ae1ed1212e6f579bb97b0cac67fae40dddca18c496e3fafdb9b423e0588e5665` | **same** |
| `clvc_dual_scope.toptee.text_only_black_image.jsonl` | `47e909b85b0fe3ae28b360a5be8218eda2b791653901e270be3dfdf207458abc` | **same** |
| `clvc_dual_scope.toptee.clvc_image_encoder_only.jsonl` | `0a514244159fa87ff41c1e3eeff064d1d0c74c8a94be1235df6e7ed373dc3aa5` | **same** |

(Identical, so the branch you authorised applied: repoint, delete, re-verify.)

Changed accordingly:

* `CLVC.evidence_relative` → `("workbench","docs","reviews","20261112-parity-repair","raw","clvc_diag")`
* `workbench/docs/reviews/20261113-final-acceptance/raw/clvc-dualscope/evidence/` — **deleted** (`rm -rf`)
* `workbench/tests/test_clvc_diagnostic.py::SHIPPED` → the authoritative path
* `manifest.json` — every cell's `artifact`/`artifact_sha256`/`artifact_bytes` now names the
  authoritative file, plus an `evidence_location` block recording why no second copy exists
* `INVENTORY.md` — the "copied into this lane" line and its reproduction snippet now use the
  authoritative path

**Payload unchanged by the repoint.** Same CLI order, both dumps normalised for the evidence root:

```
old location (recreated from authoritative bytes, temp root): 48f42fe25332e079f218d54186288890eb203e9256923372e1afd8ed29948381
authoritative location:                                        b92b9dfd734cfa1e68db59d23e07acd628cd3e5e9237dfd5602b1c1dbd06ad0b
unified diff after normalising the evidence root:              1 hunk — the cohort provenance path
                                                               (/tmp/... vs /home/...) only
```

The one hunk is `canonical_cohort_provenance.path`, which is the run-summary path rooted at
`CIR_REPO_ROOT`, not the evidence path. Everything else — all 12 cells, ranks, scores, windows,
`manifest.digest`, `status`, `defects` — is byte-identical. 70 diagnostic tests pass and both
endpoints answer `200` afterwards; the §2 LIMN hash is still `cd2d6774…`.

Current served state on the authoritative path:

```
GET /api/composition/diagnostics/clvc-dual-scope
status DIAGNOSTIC_COMPLETE  complete true  defects []  (0)
evidence …/20261112-parity-repair/raw/clvc_diag/clvc_dual_scope.json
```

## 2. LIMN regression: payload sha256 before vs after

Captured by importing the module and hashing `json.dumps(payload, sort_keys=True, separators=(',',':'))`
over the summary **plus all 12 candidate windows** — the whole served surface, not one endpoint.

| | sha256 | bytes |
|---|---|---|
| **before** (pre-edit, `baseline-limn/payload.json`) | `cd2d6774716e77d4465cf05298c13ec2684ff9d704fc891b042dec050682b511` | 402,943 |
| **after** (post-edit) | `cd2d6774716e77d4465cf05298c13ec2684ff9d704fc891b042dec050682b511` | 402,943 |

**Identical.** `diff` of the two canonical dumps: no differences. Re-checked after the router
rewrite as well.

How the byte-identity is kept (this is the design, not luck):

* LIMN's descriptor sets `additive_labels=False`; every CLVC-only key (`condition_kinds`,
  `is_reference`, `match_kind`, `effectively_encoded`, `rank_1_is_reference_self_match`,
  `target_outside_stored_window`, `raw_edit_caption`, `reference_asset`, …) is gated on that flag.
* The **reported** manifest digest is still `sha256(json.dumps(manifest_files))`; the model id was
  folded into a separate **cache key** instead. My first attempt folded it into the digest and moved
  LIMN's `manifest.digest` from `9d101852…` to `ae3b8701…`; the diff caught it and it was reverted.
* Module constants `EVIDENCE_RELATIVE`, `SCOPES`, `CONDITIONS`, `QUERIES`, `MAX_DEPTH`, the three
  status strings and `DiagnosticEvidenceError` keep their names and values, so the shipped tests and
  `main.py`'s imports keep working unchanged.

## 3. CLVC route

`GET /api/composition/diagnostics/clvc-dual-scope`
`GET /api/composition/diagnostics/clvc-dual-scope/candidates?qid=&scope=&condition=&depth=`

Both declared with **literal** paths. A parameterised `/{model_id}-dual-scope` was written first and
rejected: it shadows LIMN's existing literals depending on router-registration order, and
order-dependent routing is the bug class this module exists to refuse.

Observed on the real app + the one `include_router` line (§7):

```
STATUS DIAGNOSTIC_COMPLETE   complete=True   defects=[]
conditions  ['multimodal','text_only_black_image','clvc_image_encoder_only']
kinds       {'multimodal':'protocol','text_only_black_image':'protocol','clvc_image_encoder_only':'ablation'}
cohort      full 6003 / scored 6001 / unscored 2   from run_summary da0a2162be0e159a1a456cec
flags       diagnostic=True  canonical=False   (payload and every cell and every row)
```

## 4. The checkpoint-digest defect — corrected finding

**Your premise in the review was not what the artifact contains, and I am reporting that rather than
changing the artifact to match it.** `clvc_dual_scope.json` **does** carry the digest the reader
needs. Verbatim from the authoritative bytes:

```
checkpoints.shirt.keys()  = [checkpoint_id, directory, members, local_sha256_recomputed, mapping_status, dataset_type_index]
checkpoints.shirt.local_sha256_recomputed  = 2ba8e320b08ebb97d6558eb599cde2be228b5488250de7cb4cde8a138ff2c681
checkpoints.toptee.local_sha256_recomputed = d1fa9170496f73a008736723fe5329c93ff710305d0bcc79ff628822c0a91f4c
```

Those equal the export rows' `checkpoint_sha256` exactly (`2ba8e320…` / `d1fa9170…`), and both equal
the registry `expected_sha256` and the digests this lane recomputed from the checkpoint bytes on disk.
So **no source needed to be used and nothing was back-filled into the artifact**: the digest was
already in `checkpoints.<scope>.local_sha256_recomputed`, and the fix was only to make the descriptor
*read* that key instead of the absent `sha256`.

Consequently **the real CLVC payload is `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: []`.** The
12-defect state you saw is what the reader now reports when the digest is genuinely absent — it is not
the shipped artifact's state. Both halves are proven:

| case | source of the digest | observed |
|---|---|---|
| **authoritative artifact untouched** | `checkpoints.<scope>.local_sha256_recomputed`, present | `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: []`, digests agree with every export row |
| **digest stripped from a copy** (same bytes otherwise) | absent | `DIAGNOSTIC_EVIDENCE_DEFECT`, `complete: false`, **12** defects, code `checkpoint_summary_digest_missing`, detail *"summary checkpoints.shirt has no local_sha256_recomputed, so its checkpoint cannot be compared to clvc_dual_scope.shirt.multimodal.jsonl's 2ba8e320b08ebb97…; the agreement this cell implies is unverified"* |
| **digest changed to a wrong value** | present, wrong | `checkpoint_summary_vs_export`, scoped to `shirt` only |

The third source you offered — the manifest this reader builds — is deliberately **not** used as a
digest source: it is derived from the same file that would be wrong, so accepting it would make the
check compare an artifact to itself. The registry `expected_sha256` is the right *independent* check
and is already asserted in the test suite against the bytes on disk; it is a test-time assertion, not
a serve-time substitution, because a serve-time copy from the registry would paper over an artifact
that recorded a different digest.

**No downgrade to a silent skip.** The check reports an absence as a defect, exactly as you asked; it
simply has a digest to compare on the shipped bytes.

## 4a. Defect state in the UI — prominent, not cosmetic

A defect list nobody sees is the same as no check, so the incomplete state is now surfaced in **three**
places, and the clean state shows none of them:

1. **Card header badge**, tone `bad`: `Chưa hoàn tất — 12 defect` (`data-testid="limn-diagnostic-incomplete-badge"`),
   tooltip lists the distinct codes. Placed before the canonical badge so it is the first thing read.
2. **Panel notice** (`data-testid="limn-diagnostic-defects"`), a `pipeline-notice` block, not a `hint`
   paragraph: *"Bằng chứng chẩn đoán chưa hoàn tất — 12 defect."* + codes + *"Các hạng và điểm bên dưới
   vẫn là số đo thật từ file bằng chứng, nhưng chúng chưa được coi là đầy đủ … Trạng thái canonical
   không đổi."*
3. **Per-cell block** (`data-testid="limn-probe-incomplete"`) with the full defect detail text for the
   combinations affected, verbatim from the payload.

Observed in a real browser 1600×1200, CLVC selected, evidence copy with the digest stripped
(`/tmp/stale_root`, deleted afterwards), **0 console errors**:

```
PANEL_DEFECTS    Bằng chứng chẩn đoán chưa hoàn tất — 12 defect. Mã: checkpoint_summary_digest_missing. …
PROBE_INCOMPLETE Bằng chứng chẩn đoán chưa hoàn tất. 1 defect liên quan tới ô này:
                 checkpoint_summary_digest_missing — summary checkpoints.shirt has no
                 local_sha256_recomputed, so its checkpoint cannot be compared to
                 clvc_dual_scope.shirt.multimodal.jsonl's 2ba8e320b08ebb97…; the agreement this
                 cell implies is unverified
CANON_BADGE      Canonical (Full cohort): 6,001/6,003 scored
TILES            10        (K=10 default; target outside → stated)
TARGET_OUTSIDE   Ảnh đích ở hạng #848 — ngoài cửa sổ Top-10 đang hiện (export lưu 100 dòng)…
```

Screenshots (relative to `raw/clvc-dualscope/`):

```
screenshots/clvc-defect-state-panel.png     full page, defect state
screenshots/clvc-defect-state-notice.png    the panel notice + header badge
screenshots/clvc-defect-state-probe.png     per-cell defect detail
```

The clean state (authoritative artifact) was re-rendered with the same build and shows **no** badge and
**no** notice — verified by the 12-cell sweep in §9, which passed 12/12 with 0 mismatches and 0 console
errors and is re-run in §9a after the repoint.

## 5. Reference is not a retrieval hit

* Every candidate carries `is_reference` and `match_kind ∈ {reference_self_match, candidate}`;
  the window carries `rank_1_is_reference_self_match`; the payload carries `reference_match_note`.
* In all 4 `clvc_image_encoder_only` cells rank-1 **is** the reference with `score ≈ 1.0`
  (`0.99999982 / 0.99999964 / 0.99999970 / 0.99999994`) and `is_target=false`.
* UI renders a banner: *"Hạng 1 ở điều kiện này là chính ảnh tham chiếu của query. Đây là
  self-match (score ≈ 1.000000), không phải một kết quả truy hồi … Ảnh đích thật của query nằm ở
  hạng #8,702."*, plus a per-tile `Ảnh tham chiếu (self-match)` badge and
  `data-match-kind="reference_self_match"`.
* The target is **never** positioned inside the window: `target_within_export_depth=false` and
  `target["in_stored_window"]=false` in all 12 cells; the UI prints *"ngoài cửa sổ Top-100 đang
  hiện (export lưu 100 dòng) … hạng #N là hạng toàn gallery, không phải vị trí trong danh sách ứng
  viên"* and renders `target_rank`, never an index.

## 6. Tests

```
workbench/envs/ctl312h/bin/python -m pytest workbench/tests -k diagnostic -q
→ 70 passed, 3212 deselected, 6 warnings in 42.57s
```

52 pre-existing diagnostic tests pass **unmodified** (zero test edits needed for the
generalisation) + 18 new in `test_clvc_diagnostic.py`:

descriptor table / unknown model refused · CLVC's third condition is the ablation and
`image_only_empty_text` is absent + refused · shipped evidence `DIAGNOSTIC_COMPLETE` with 0 defects ·
cohort from run summary with provenance · **missing digest = 12 named defects, not `complete:true`** ·
mismatched digest scoped to `shirt` · missing cohort + missing run summary refused · rank-1
self-match labelled in all 4 ablation cells · protocol conditions don't claim a self-match ·
target outside the stored window in all 12 (exact ranks `[848,304,8702,1253,1549,1267,412,1466,5481,259,568,4535]`) ·
ablation reports no text encoded · **1200/1200 candidates resolve to a served image** ·
`image_only_empty_text` refused for CLVC · summary and window agree per cell ·
**LIMN payload keeps its exact shape** (additive keys asserted absent) · unpublished CLVC evidence is
a named absence · wrong `artifact_type` refused for the right model.

## 7. The exact `main.py` lines needed

Two lines, one import + one include. `main.py` already imports from this module lazily inside its
endpoint bodies, so the import belongs beside the existing FastAPI/app setup (top of file is fine —
the module has no import-time dependency on `main`; `_resolvable` imports it lazily):

```python
from workbench.backend.limn_diagnostic import diagnostic_router
```

and after the app is constructed (next to the other `include_router` calls, if any):

```python
app.include_router(diagnostic_router)
```

That is the whole integration: the router declares only
`/api/composition/diagnostics/clvc-dual-scope` and `.../clvc-dual-scope/candidates`, so LIMN's
existing literals in `main.py` are untouched and no path can be shadowed. Verified by running the
real app with exactly these two lines in a throwaway module (`/tmp/verify_app.py`, deleted
afterwards): both routes answered `200`, and `limn-dual-scope` still answered `200`.

**Current deployment state:** the running backend on `127.0.0.1:8000` returns **404** for the CLVC
route (`limn` = 200). Deployment is **pending** — the services were not restarted, per instruction.

## 8. Bundle

```
workbench/frontend/dist/assets/index-DbBSPMkr.js    397.48 kB (gzip 115.92 kB)
  sha256 da107d981517e37ae7b35e299712511cf04d064d6702b7cb92f35fc9b398d71a
workbench/frontend/dist/assets/index-DecQnta9.css    26.55 kB (gzip 6.16 kB)
  sha256 e7e826a332ec9ab171f97051b901ee86fa69e2588937c3023914f3ac41ffe196
```

`npm run build` → `tsc -b && vite build`, exit 0, 39 modules. `npx tsc -b --noEmit` clean.
(Superseded `index-CFwf1H51.js` / `5730d4f7…` — that build predates the defect-prominence change in
§4a.)

## 9. Browser verification — 12 cells, real Chromium

Method: temporary uvicorn (`:8765`) running the real `workbench.backend.main:app` plus the one
`include_router` line, and `vite preview` (`:5174`) proxying to it. The shipped services on
`:8000`/`:5173` were never restarted and are still serving the pre-change bundle. **Every switch
below is a `<select>` change on the page — no reload, no direct API read for the displayed values.**
The API is consulted only to pair each reading with the payload the same page fetched.

Rendered values, CLVC, model selector = `CLVC-Net`, query `7:3038` then `7:3790`:

| query | scope | condition | UI rank | inventory rank | API rank | target score | tiles | rank-1 self-match | target outside | text empty |
|---|---|---|---|---|---|---|---|---|---|---|
| `7:3038` | `shirt` | `multimodal` | 848 | 848 | 848 | 0.463620 | 100 | no | yes | no |
| `7:3038` | `shirt` | `text_only_black_image` | 304 | 304 | 304 | 0.477407 | 100 | no | yes | no |
| `7:3038` | `shirt` | `clvc_image_encoder_only` | 8702 | 8702 | 8702 | 0.303293 | 100 | **yes** | yes | **yes** |
| `7:3038` | `toptee` | `multimodal` | 1253 | 1253 | 1253 | 0.565981 | 100 | no | yes | no |
| `7:3038` | `toptee` | `text_only_black_image` | 1549 | 1549 | 1549 | 0.458470 | 100 | no | yes | no |
| `7:3038` | `toptee` | `clvc_image_encoder_only` | 1267 | 1267 | 1267 | 0.592141 | 100 | **yes** | yes | **yes** |
| `7:3790` | `shirt` | `multimodal` | 412 | 412 | 412 | 0.571143 | 100 | no | yes | no |
| `7:3790` | `shirt` | `text_only_black_image` | 1466 | 1466 | 1466 | 0.444272 | 100 | no | yes | no |
| `7:3790` | `shirt` | `clvc_image_encoder_only` | 5481 | 5481 | 5481 | 0.259081 | 100 | **yes** | yes | **yes** |
| `7:3790` | `toptee` | `multimodal` | 259 | 259 | 259 | 0.743001 | 100 | no | yes | no |
| `7:3790` | `toptee` | `text_only_black_image` | 568 | 568 | 568 | 0.618858 | 100 | no | yes | no |
| `7:3790` | `toptee` | `clvc_image_encoder_only` | 4535 | 4535 | 4535 | 0.445611 | 100 | **yes** | yes | **yes** |

**Diff result: 0 mismatches across 12 cells × 13 checks** (`browser-vs-inventory.json`). Checks per
cell: note rank, target-paragraph rank, 100 tiles, "outside" statement, self-match banner agrees with
`rank_1_is_reference_self_match`, empty-text disclosure agrees with
`effectively_encoded.text_is_empty`, released caption present in the UI, first-tile score == the API
window's rank-1 score to 4 dp, first-tile `data-match-kind` agrees, target score within 5e-7 of the
inventory's 6 dp figure, cohort provenance rendered, evidence path rendered, `6,001/6,003` badge
rendered.

Also captured: LIMN through the same page and model selector — 12 cells, 100 tiles each, **0 console
errors**, `image_only_empty_text` as its third condition, no self-match banner anywhere (LIMN has no
ablation). Console errors in the whole CLVC sweep: **0**.

Screenshots (paths relative to `raw/clvc-dualscope/`), 26 files:

```
screenshots/clvc-7_3038-shirt-multimodal.png          … screenshots/clvc-7_3790-toptee-clvc_image_encoder_only.png
screenshots/limn-7_3038-shirt-multimodal.png          … screenshots/limn-7_3790-toptee-image_only_empty_text.png
screenshots/panel-clvc.png                            screenshots/panel-limn.png
```

Two harness facts worth recording, both found by the sweep itself: the evidence/provenance lines sit
inside a `<details>` (a collapsed `inner_text` returns `""`, so the sweep now opens every `<details>`
first — otherwise it would have recorded "not rendered" for rendered text), and each model needs its
**own** condition tuple (sending `clvc_image_encoder_only` to LIMN fails the select; `MODEL_CONDITIONS`
now keys them per model).

## 9a. Re-verified after the evidence repoint

The 12-cell sweep was re-run against the authoritative evidence path with the final build
(`index-DbBSPMkr.js`), through the real app + one `include_router` line on a temporary port:

```
cells 12 · console_errors [] · MISMATCHES 0 (12 cells × 13 checks)
```

Ranks and scores are unchanged from the table in §9 — 848/304/8702/1253/1549/1267 (7:3038) and
412/1466/5481/259/568/4535 (7:3790) — as are self-match, outside-window and empty-text flags. LIMN
was re-swept too: 12 cells, 0 console errors. Full per-cell evidence: `browser-vs-inventory.json`.

## 9b. Final state

| | |
|---|---|
| diagnostic tests | **70 passed**, 3212 deselected (52 pre-existing + 18 new CLVC) |
| LIMN payload sha256 | `cd2d6774716e77d4465cf05298c13ec2684ff9d704fc891b042dec050682b511` — unchanged |
| CLVC payload | `DIAGNOSTIC_COMPLETE`, `complete: true`, `defects: []`, from the authoritative artifact |
| CLVC payload on stripped digest | `DIAGNOSTIC_EVIDENCE_DEFECT`, `complete: false`, 12 × `checkpoint_summary_digest_missing` |
| evidence copies | **1** (authoritative); duplicate deleted |
| bundle | `dist/assets/index-DbBSPMkr.js` `da107d98…`; `index-DecQnta9.css` `e7e826a3…` |
| main.py | **unmodified** — the two lines in §7 are still unapplied, awaiting you |
| temp services | all stopped (ports 8765/8766/5174/5175 closed); shipped 8000/5173 never restarted |
| commits | none |

## 10. Decisions I did not make for you

1. **Module name.** The file is still `workbench/backend/limn_diagnostic.py` although it now serves
   two models. Renaming is a cutover you said you had not approved (a shim would be needed), so the
   path is unchanged and the docstring states the scope. Say the word if you want
   `diagnostic_evidence.py` + a re-export shim.
2. **LIMN's payload is frozen, so the LIMN panel has no self-match label, no ablation badge and no
   cohort-provenance line.** That is deliberate: adding them to LIMN changes its bytes and breaks the
   regression evidence. If you want the LIMN panel to look identical to CLVC's, that is a separate
   decision that retires the byte-identity guarantee — it must land as its own change with its own
   before/after capture.
3. **`canonical_cohort.multimodal_recall_at_10` is `null` for CLVC.** The run summary does not state
   it and I did not borrow a number from a different measurement. If you want it shown, name the
   authoritative artifact.
