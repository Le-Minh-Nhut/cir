# A38 W4 — Composition runner, adapters, cache/resume identity

Evidence re-derived from source. Inference was not executed; all adapter rows are static. Internal cache helpers were exercised using tiny synthetic files in `/dev/shm/a38-w4`.

## Mechanism table

| Mechanism | Writer | Reader | Key | Validation | Invalidation | Consumers |
|---|---|---|---|---|---|---|
| Gallery features | `_encode_gallery`, `runner.py:226-259` | `_load_memmap`, `runner.py:158-168,229` | `runs/<run-id>/work/<model>/features/gallery_<scope>.npy`; digest of run fingerprint/scope/kind, `runner.py:712-720` | metadata fingerprint + safe npy load; gallery shape not checked against expected, `runner.py:158-168` | changed run/scope path; invalid data rejected; working dir deleted after model, `runner.py:951-967` | exact rank `runner.py:757-786` |
| Query features | `_encode_queries`, `runner.py:274-357` | `_load_memmap`, `runner.py:278` | `query_<condition>_<scope>.npy`; digest run/condition/scope/kind, `runner.py:735-744` | fingerprint, expected `(queries, dimension)`; sidecar parse failure silently gives empty errors, `runner.py:278-290` | invalid data; condition retry deletes data/meta/error sidecars, `runner.py:738-741`; work cleanup | exact rank |
| Rank/resume JSON | `_atomic_json`, `runner.py:835-838` | `runner.py:663-694` | `runs/<run-id>/ranks/<model>_<checkpoint>.json`; full identity fingerprint | parses JSON, fingerprint equality, condition-key set; no validation rank values/query IDs/schema/completeness | identity digest/run dir, parse errors rejected, condition mismatch; transient failures selectively retried | rank reuse, exports, metrics/summaries `runner.py:688-692,840-921` |
| Per-run artifact directory | `runner.py:650-656,835-838` | identity hash, exporters `runner.py:862-921` | first 24 hex of identity SHA256 under `composition/runs`; no lock | destination containment only; concurrent writers may target same rank path | identity shift gives distinct path; model work removed `951-967` | feature/rank caches |
| Image identity | `_image_identity`, `runner.py:530-540` | identity `runner.py:627-628` | resolved absolute path digest + digest over per-file SHA256 bytes | stream hashing `data.sha256_file`; not mtime-based | any content/path change makes new run key | all caches |
| Checkpoint identity | CLI `_checkpoint_identity`, `composition_eval.py:163-176`; `ModelSpec` `runner.py:25-34` | run identity `runner.py:622-626`; summary `918` | file SHA256 or sorted relative directory names + member SHA256 | only hashes existing file/tree; missing returns `missing` | byte/tree change | all caches |

## Identity dimensions

| Dimension | Finding |
|---|---|
| Model ID | INCLUDED `runner.py:624` |
| Checkpoint ID | INCLUDED `runner.py:624` |
| Checkpoint content | INCLUDED via CLI digest `composition_eval.py:163-176`; direct `ModelSpec` default is empty `runner.py:33,622-626` (MISSING for direct caller not supplying identity) |
| Model source | MISSING — third-party source tree absent; source changes can reuse stale ranks/features |
| Adapter source | INCLUDED — module hash + method source `runner.py:634-640`; module hash unavailable becomes literal unavailable |
| Preprocessing | INCLUDED — registry preprocessing fields `runner.py:630-633`; adapter internal/third-party transform code only covered if adapter module changes |
| Dataset identity | INCLUDED — gallery ordered IDs/candidate names `runner.py:176-179`; query fields `182-185`; query/gallery image byte identities `530-540` |
| Query identity | INCLUDED — query fingerprint/order and text/reference/positives/subsets `runner.py:182-185,628` |
| Gallery identity | INCLUDED — protocol/order/names `runner.py:176-179,625,628` |
| Protocol | INCLUDED `runner.py:625` |
| Condition | INCLUDED both feature keys: gallery feature fingerprint inherits run identity containing `conditions` at `runner.py:628`, key construction `716-720`; query feature additionally explicitly includes condition `runner.py:735-737`. Rank cache also inherits conditions in fingerprint and tests condition names `runner.py:628,663-667`. Exact literal: `"conditions": list(conditions)` (`runner.py:629`); query key literal `"condition": condition` (`runner.py:736`). |
| Relevant config | INCLUDED batch size, seed, adapter batch size, scopes, reference eligibility, mode support `runner.py:640-648`; preprocessing settings `630-633`. Runtime config not represented beyond runtime identity. |
| Runtime identity | INCLUDED interpreter path/version/environment fingerprint and GPU identity `runner.py:386-450,642-643`; probe can report unavailable status |
| Condition actually changes inputs | confirmed `inputs.py:19-27`; black PIL pixels for text-only, empty string for image-only |

## Cache test matrix

Executed `/dev/shm/a38-w4/focused.py` against `run_composition(..., resume=True)` with synthetic 3-image gallery and production-written rank JSON shape (`fingerprint`, `identity`, `attempt`, `ranks`, `query_failures`, `condition_status`; condition mapping qid→rank). Also ran synchronized same-path writers. Raw outputs below.

| # | Test | Result | Evidence |
|---|---|---|---|
| 1 | Cold execution | PASS (helper-level) | 3-path gallery encoding made 2 batches; wrote `.npy` + `.npy.json` |
| 2 | Warm execution | PASS (helper-level) | second call made zero encoder calls; feature array equal. Rank JSON byte identity not exercised |
| 3 | Exact resume | PASS (executed) | all three poisoned payloads loaded with zero encoder/rank calls; exact return evidence below |
| 4 | Interrupt mid-model | STATICALLY_CHECKED | rank JSON is written only after condition loop `runner.py:701-838` |
| 5 | Partial cache | STATICALLY_CHECKED | valid rank cache bypasses reranking; feature cache independently loads `runner.py:701-745` |
| 6 | Corrupt rank member | FAIL OPEN for semantically malformed JSON | all three pass resume; string fails later, missing query and rank 9999 proceed as cached. Metrics silently undercount/misstate. Raw output below |
| 7 | Missing cache member | STATICALLY_CHECKED | missing rank path skips reuse; missing/metadata-missing memmap returns None `runner.py:158-168,663` |
| 8 | Changed checkpoint ID | STATICALLY_CHECKED | checkpoint ID in identity and rank filename `runner.py:624,656` |
| 9 | Same nominal checkpoint ID, changed content | PASS (identity input hash) | bytes `one` vs `two` produce different SHA256; CLI binds digest; direct runner caveat remains |
| 10 | Changed dataset, restored mtime | PASS | same-size content rewrite changed `_image_identity.content_digest` |
| 11 | Changed adapter source | STATICALLY_CHECKED | adapter module/method hashes in identity `runner.py:634-640` |
| 12 | Changed configuration | STATICALLY_CHECKED | selected config fields included `runner.py:630-648`; unrepresented config remains possible |
| 13 | Changed interpreter | STATICALLY_CHECKED | runtime identity in key `runner.py:642`; `386-450` |
| 14 | Changed condition | STATICALLY_CHECKED | query key explicitly condition; run identity condition list `runner.py:629,735-737` |
| 15 | Changed protocol | STATICALLY_CHECKED | protocol identity `runner.py:625` |
| 16 | Concurrent same-root writes | FAIL for feature atomicity; PASS atomic complete rank JSON | 8,000,128-byte feature path ended with mixed [1.0,2.0] payload; `_atomic_json` yielded one intact writer payload. Unique temp is pid+UUID, `runner.py:142-155` |
| 17 | Failed execution | STATICALLY_CHECKED | exception creates FAILED results `runner.py:940-949`; rank write is success path `835-838` |

### Raw executed output

```text
initial run: COMPLETED None 45f2e5d07d3b2242dd99809a
production rank keys: ['attempt', 'condition_status', 'fingerprint', 'identity', 'query_failures', 'ranks']
production rank condition keys: ['multimodal']
production payload keys: ['q0']
payload=string; cache=REUSED; encode/rank calls delta 0; report status=FAILED
payload=missing-qid; cache=REUSED; encode/rank calls delta 0; report status=REUSED
payload=out-of-range; cache=REUSED; encode/rank calls delta 0; report status=REUSED
atomic_json race: B intact= True temp pattern=PID+UUID per runner.py:144
feature race: 8000128 bytes; shape= (2000000,) unique= [1.0, 2.0] intact= False
```

Metric check used `compose` on poisoned condition ranks: missing qid silently becomes no scored rank (`None` metric); rank 9999 yields recall@10 `0.0` and is treated as a valid scored rank. String reaches `_validate_rank` and raises ValueError. Rank-consumer proof: `metrics.py:59-76` validates only positive integer, not gallery bound; missing qids are converted to None at `metrics.py:111-119`.

## Adapters

All nine rows: `STATICALLY_CHECKED` (not `PASS`); no checkpoint/GPU inference. Input/output contracts derive from adapter code. Checkpoint strictness is noted where loaded in owned adapter code; workers can defer loading.

| Model / selection | Contract, batches, conditions, preprocessing, errors / serialization | Checkpoint load | Status |
|---|---|---|---|
| csmcir / `model_csmcir` | worker-backed paths + text/categories; feature arrays npy; shared category scope; three conditions; worker errors surfaced | `strict=False` at `model_csmcir.py:199-200`; no missing/unexpected check shown | STATICALLY_CHECKED — no checkpoint/GPU |
| conesep / `model_conesep` | worker-backed path features, category scope and exact rank; three conditions; worker JSON/npy results | loader path requires inspect; not proven strict in reviewed evidence | STATICALLY_CHECKED — no checkpoint/GPU |
| ptha_mtst / registry module `model_ptha_mtst` | subprocess image/text fusion, token target features; shared scope; condition support; worker errors | `strict=True`, `model_ptha_mtst.py:305-306` | STATICALLY_CHECKED — no checkpoint/GPU |
| habit / `model_habit` | subprocess actual model forwards; path/text arrays, shared scope; three conditions; worker protocol | loader implementation not established strict here | STATICALLY_CHECKED — no checkpoint/GPU |
| intent / `model_intent` | subprocess image/text, shared scope; max-token scores/ranks; three conditions | loader implementation not established strict here | STATICALLY_CHECKED — no checkpoint/GPU |
| hint / `model_hint` | subprocess native query/gallery forwards; shared scope; three conditions; worker results | loader implementation not established strict here | STATICALLY_CHECKED — no checkpoint/GPU |
| encoder / `model_encoder` | paths/text; subprocess batches 32, float32 `(N,512)` features and npy output `model_encoder.py:311-330`; three conditions | `strict=False` then `_assert_checkpoint_loaded`, `model_encoder.py:299-302` | STATICALLY_CHECKED — no checkpoint/GPU |
| limn / `model_limn` | scope selects per-category model; full gallery per scope; configured own batch; worker-backed feature results; three conditions | member loader details not established strict here | STATICALLY_CHECKED — no checkpoint/GPU |
| tme / `model_tme` | paths/text/categories; shared scope; validates lengths, worker arrays; token max-cosine scoring `model_tme.py:224-253`; three conditions | `strict=False`, rejects any missing/unexpected keys `model_tme.py:362-367` | STATICALLY_CHECKED — no checkpoint/GPU |

Common adapter contract: local readable paths, caller-assigned work root, float32 feature outputs, full gallery and exact positive ranks `adapters.py:55-69,109-122,148-169`. Strict=True was found for ptha_mtst only. Strict=False + explicit discrepancy check found for TME and ENCODER; CSMCIR strict=False without explicit check. Remaining loaders require source-specific verification before claiming checkpoint correctness.

## Provenance

Stored `workbench/artifacts/composition/summary/tme_fashioniq_0_a82dfefb3c5d330843d858f2/full.json`: model/checkpoint and checkpoint identity at lines 284-285, model at 387, protocol 398, gallery identity 334, count 333, condition list/status 319-325, runtime 430, run ID/resume key 415-418, release hashes 406-413. Summary records all requested except explicit code revision/commit ID; run identity is represented by run_id/resume_key. Stored run has `execution_status: FAILED` (325), not successful evidence. Reproduction also needs source-tree hashes for adapter/model dependencies, exact invocation/config values, and extant checkpoint/image assets; summary's runtime and registry labels are incomplete substitutes.

## FINDINGS

### C-01 — HIGH — Rank cache accepts malformed/incomplete rank payload
- Observed: executed real resume path. String rank rejected downstream with FAILED ValueError; omitted qid and rank 9999 were accepted as REUSED. Missing qid silently disappears from scored cohort; 9999 yields valid-looking recall@10=0.0.
- Expected: resume rejects wrong qid set, non-integer/out-of-gallery rank.
- Root cause: cache checks fingerprint/condition names only; metrics lacks gallery bound and missing-qid coverage verification.
- Reproduction: raw output below; poisoned payloads were all accepted by resume. Downstream string metric raises; missing qid maps to `None`; rank 9999 produces recall@10 `0.0`.
- Regression test (proposed): three payload variants above rejected before reuse.
- Fix (proposed): validate exact query IDs and nullable integer ranks in `[1,gallery_count]` before reuse.
- Verification: `run_composition(..., resume=True)` returned `REUSED` with zero encode/rank calls for all three; string then caused report failure during summary metric calculation.
- Status: CONFIRMED_UNFIXED.

### C-02 — INFORMATIONAL — Optional library identity not populated by CLI caller
- Feature/files: `runner.py:33,622-626`; `composition_eval.py:163-176,287-291`
- Observed: repository search found one production `run_composition` call: `workbench/scripts/composition_eval.py:291`; its only `ModelSpec` construction at `288-290` supplies `_checkpoint_identity(...)`. Other match is definition; no other workbench caller found. Tests may construct specs independently.
- Expected: CLI user runs bind checkpoint bytes; library callers should supply content identity.
- Root cause: field is optional by API; no CLI reachability to empty identity found.
- Reproduction: static caller search; not executed.
- Regression test (proposed): direct API contract test if external library callers are supported.
- Fix (proposed): no production fix indicated; document/enforce identity for future library callers.
- Verification: `grep` caller inventory above; CLI hash implementation `composition_eval.py:163-176`.
- Status: EXPECTED_BEHAVIOR for reachable CLI; library-contract limitation only.

### C-03 — HIGH — Concurrent feature writers race on shared memmap path
- Feature/files: `runner.py:226-259,250-258`; rank JSON `_atomic_json` `142-155`
- Observed: two synchronized threads opened same 8 MB final `.npy` with `open_memmap(mode="w+")`, filled distinct constants; final array contained both values. `_atomic_json` raced with 1 MB payloads and final JSON was one intact writer's payload.
- Expected: published feature cache must be one valid complete writer result; concurrent run must not mix feature blocks.
- Root cause: feature uses final path directly while writing; rank writer uses atomic same-filesystem replacement with unique `.{name}.{pid}.{uuid}.tmp` then `os.replace`.
- Reproduction: raw output above; source exact temporary naming `runner.py:144`; feature direct open `runner.py:250`.
- Regression test (proposed): synchronized competing `_encode_gallery`/memmap writes and assert final array exactly equals one writer.
- Fix (proposed): feature write to unique temporary memmap then atomic replace + metadata publication.
- Verification: concurrent synthetic 2,000,000-element arrays; mixed data observed. Rank JSON atomic winner intact.
- Status: CONFIRMED_UNFIXED.

### C-04 — INFORMATIONAL — Model source pin exists, composition key omits source revision
- Feature/files: `workbench/registry/models.yaml` (per-model `upstream_commit_sha`); `workbench/backend/runtime.py:2833-2846`; `workbench/backend/vendored_source.py:311-318`; `runner.py:623-650`
- Observed: registry pins source commits and backend provides source-pin verification. No source verification call found in composition runner/evaluation path; runner key hashes adapter module, not model source. Local third-party mutation may therefore preserve composition key unless another preflight enforces source verification.
- Expected: legitimate source edits should not silently reuse old result; pinned source state should be enforced before evaluation.
- Root cause: run identity excludes source commit/content and composition entry path does not visibly call source validator.
- Reproduction: static source/registry inspection only; no user-reachable legitimate mutation established.
- Regression test (proposed): source verification in composition invocation or source revision enters identity.
- Fix (proposed): no medium-severity user-impact finding without proving bypass; retain informational limitation and require verified source pin on execution path.
- Verification: model registry contains pins; runtime/vendor modules compare actual to expected pin; no composition caller identified.
- Status: DOWNGRADED_INFORMATIONAL; pinned commit is registry provenance, but composition-path enforcement is not established.

## Report quick view

- C-01 HIGH CONFIRMED_UNFIXED; C-02 informational/EXPECTED_BEHAVIOR for CLI; C-03 HIGH CONFIRMED_UNFIXED; C-04 informational, source-pin enforcement not established.
- Exact executed output is inline above; scratch scripts removed after recording evidence.
