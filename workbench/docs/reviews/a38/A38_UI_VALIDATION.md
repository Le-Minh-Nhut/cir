# A38 UI validation — W5

## App under test

- Frontend: `http://127.0.0.1:5173/`, existing Vite listener (HTTP 200, `CIR Failure Analysis Workbench`). Used existing listener; no server started.
- Backend: `http://127.0.0.1:8000/`; `/openapi.json` responds, `/` is 404. Existing listener. `/api/models` returned 200 but took 27.653783s, 27.681775s, 27.721097s (each spaced 2s).
- `workbench/frontend/package.json`: `npm run dev` = `vite`; `npm run build` = `tsc -b && vite build`. `vite.config.mts` proxies `/api` to `http://127.0.0.1`.

## Harness and commands

- Existing harness: `workbench/frontend/tests/ui_smoke.py` (only tracked file under frontend/tests). Uses `playwright.sync_api`, `playwright.chromium.launch()`, viewport 1440x900 and 390x844. It requires pre-existing fixture runs for fixture workflows; `--session all` includes cold, fixture, and mobile sessions. Default screenshot path is `/tmp/ui-e2e/shots`.
- Browser: Playwright Chromium at `/tmp/pw-browsers`; `node_modules/.bin` has no Playwright entry and `require.resolve('playwright')` reports missing. `.venv` has Python Playwright.
- Ran: `PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers .venv/bin/python workbench/frontend/tests/ui_smoke.py --url http://127.0.0.1:5173 --shots /dev/shm/a38-w5/harness-shots --session all`
- Result: existing cold inventory and fixture flows passed through navigation, Top-K image loading, filters, pagination, annotation-save/reload, injected error/recovery, compare, cohort save/reload, and stale-response protection. Mobile phase failed: `TimeoutError` reading `/api/models` after prior requests. No harness test was added; no UI fix was made, so fail-before/pass-after does not apply.
- Separate matrix capture used same repo `.venv` Playwright and browser: `PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers .venv/bin/python /dev/shm/a38-w5/matrix.py`. Captured each of 10 app sections once at 1440x900, 1280x800, 390x844, 360x800. Measurements saved in `/dev/shm/a38-w5/measurements.json`, screenshots in `/dev/shm/a38-w5/shots/` (RAM-backed).
- Tests/build: frontend package defines no test script. Existing smoke command above failed in mobile phase as recorded. Build not run (not required by assignment? available `npm run build`; build has repo `dist` output and disk is critical, so `SKIPPED — disk`).

## Page × viewport

Overflow = all body descendants checked for `scrollWidth > clientWidth + 1` or `getBoundingClientRect().right > viewport + 1`; empty means none observed. All page URLs were `/`; every page produced matching section heading. Screenshot paths are `/dev/shm/a38-w5/shots/<filename>`.

| Page | 1440×900 | 1280×800 | 390×844 | 360×800 |
|---|---|---|---|---|
| Dashboard | Yes; none; no errors; `1440x900-dashboard.png` | Yes; none; no errors; `1280x800-dashboard.png` | Yes; none; no errors; `390x844-dashboard.png` | Yes; none; no errors; `360x800-dashboard.png` |
| Models | Yes; none; no errors; `1440x900-models.png` | Yes; none; no errors; `1280x800-models.png` | Yes; none; no errors; `390x844-models.png` | Yes; none; no errors; `360x800-models.png` |
| Evaluation Runs | Yes; none; no errors; `1440x900-evaluation-runs.png` | Yes; none; no errors; `1280x800-evaluation-runs.png` | Yes; none; no errors; `390x844-evaluation-runs.png` | Yes; none; no errors; `360x800-evaluation-runs.png` |
| Sample Explorer | Yes; none; no errors; `1440x900-sample-explorer.png` | Yes; none; no errors; `1280x800-sample-explorer.png` | Yes; none; no errors; `390x844-sample-explorer.png` | Yes; none; no errors; `360x800-sample-explorer.png` |
| Compare Models | Yes; none; no errors; `1440x900-compare-models.png` | Yes; none; no errors; `1280x800-compare-models.png` | Yes; none; no errors; `390x844-compare-models.png` | Yes; none; no errors; `360x800-compare-models.png` |
| Common Failures | Yes; none; no errors; `1440x900-common-failures.png` | Yes; none; no errors; `1280x800-common-failures.png` | Yes; none; no errors; `390x844-common-failures.png` | Yes; none; no errors; `360x800-common-failures.png` |
| Disagreement | Yes; none; no errors; `1440x900-disagreement.png` | Yes; none; no errors; `1280x800-disagreement.png` | Yes; none; no errors; `390x844-disagreement.png` | Yes; none; no errors; `360x800-disagreement.png` |
| Failure Analytics | Yes; none; no errors; `1440x900-failure-analytics.png` | Yes; none; no errors; `1280x800-failure-analytics.png` | Yes; none; no errors; `390x844-failure-analytics.png` | Yes; none; no errors; `360x800-failure-analytics.png` |
| Annotations | Yes; none; no errors; `1440x900-annotations.png` | Yes; none; no errors; `1280x800-annotations.png` | Yes; none; no errors; `390x844-annotations.png` | Yes; none; no errors; `360x800-annotations.png` |
| Hypotheses | Yes; none; no errors; `1440x900-hypotheses.png` | Yes; none; no errors; `1280x800-hypotheses.png` | Yes; none; no errors; `390x844-hypotheses.png` | Yes; none; no errors; `360x800-hypotheses.png` |

Model details and checkpoint details are not separate pages in this app; models are presented in Models inventory. Loading was captured in Models; empty/error/recovery states were exercised by existing smoke harness, not separately captured at every viewport. Filter/pagination/cohort/annotation/export are UI workflows/controls, not separate pages. Export behavior was not exercised.

## Visual review

- `390x844-models.png` and `1280x800-models.png`: mobile heading wraps; desktop nav/configuration fit. Both screenshots show `loading registry...` instead of model rows after capture wait. Warning text explains only development/mock results load. No overlap or horizontal clipping apparent in these images.
- `1440x900-sample-explorer.png`: desktop controls fit; explorer clearly states no compatible run for selected protocol; pager disabled. Large blank area below empty-state message.
- Dashboard screenshot reviewed in browser controller: dashboard metrics complete, warning and controls visible; no apparent overlap.
- Other 37 screenshots captured but not individually visually inspected; visual claims limited to named screenshots.

## Confirmed defects / API timing investigation

### UI/API latency (HIGH observation; no code change)

Exact route handler from `workbench/backend/main.py` lines 159–182:

```python
@app.get("/api/models")
def models():
    registry = load_registry()
    out = []
    for model in registry["models"]:
        entry = {**model, "checkpoints": [{**checkpoint_availability(model, item),
                                           "checkpoint_training_noise_pct": item.get("checkpoint_training_noise_pct"),
                                           "evaluation_noise_pct": item.get("evaluation_noise_pct")}
                                          for item in model["checkpoint_variants"]]}
        bundles = model.get("checkpoint_bundles") or []
        if bundles:
            by_id = {c["checkpoint_id"]: c for c in model["checkpoint_variants"]}
            entry["checkpoint_bundles"] = [
                {**bundle,
                 "members": [{**checkpoint_availability(model, by_id[member]),
                              "checkpoint_training_noise_pct": by_id[member].get("checkpoint_training_noise_pct"),
                              "evaluation_noise_pct": by_id[member].get("evaluation_noise_pct")}
                             for member in bundle.get("required_checkpoint_ids", []) if member in by_id]}
                for bundle in bundles
            ]
        out.append(entry)
    return out
```

Repeated `time curl` results: `200 27.653783` (real 27.669s), `200 27.681775` (real 27.694s), `200 27.721097` (real 27.736s). Endpoint reliably slow, not a single transient request. Frontend smoke later hit Python URL-open timeout while loading mobile Models.

Instrumented two calls to the real `workbench.backend.main.models()` handler in one fresh process, wrapping registry/runtime functions specified by the operator. Verbatim output:

```text
COLD_HANDLER_SECONDS 96.602 entries 17
checkpoint_availability: calls=32 cum=96.477
sha256_file: calls=27698 cum=63.993
_cached_availability_digest: calls=44 cum=33.410
_availability_digest: calls=44 cum=33.412
_recorded_checkpoint_digests: calls=44 cum=0.001
runtime_blockers: calls=32 cum=63.050
_member_digest: calls=27637 cum=20.206
WARM_HANDLER_SECONDS 28.524 entries 17
checkpoint_availability: calls=64 cum=124.862
sha256_file: calls=27748 cum=73.070
_cached_availability_digest: calls=88 cum=33.411
_availability_digest: calls=88 cum=33.413
_recorded_checkpoint_digests: calls=88 cum=0.002
runtime_blockers: calls=64 cum=91.419
_member_digest: calls=27637 cum=20.206
```

Warm-only delta profile (same-process second real handler invocation):

```text
COLD 99.028
WARM 29.223
verify_environment warm_delta_calls 32 warm_delta_sec 0.319
environment_blockers warm_delta_calls 32 warm_delta_sec 0.319
_interpreter_fingerprint warm_delta_calls 2 warm_delta_sec 0.0
_record_environment_fingerprint warm_delta_calls 0 warm_delta_sec 0.0
source_clean_and_pinned warm_delta_calls 32 warm_delta_sec 7.557
source_provenance warm_delta_calls 32 warm_delta_sec 7.555
safe_load warm_delta_calls 300 warm_delta_sec 9.3
```

The `COLD`/`WARM` table above is two sequential calls to the real handler with wrapper counters cumulative from process start. Fresh process first call was 96.602s; second call was 28.524s. This agrees with three separate live curls at ~27.7s. Earlier 143.214s measured a synthetic loop over 29 variants (including repeated bundle member calls); it was not one endpoint and must not be compared as request wall time.

An additional cProfile of an actual warm request during heavier shared-host load measured 61.948s under profiler overhead. Its ranked stack excerpt:

```text
WARM_WALL 61.948
       1    0.000    0.000   61.947   61.947 main.py:159(models)
      32    0.001    0.000   61.563    1.924 registry.py:1869(checkpoint_availability)
      32    0.001    0.000   61.515    1.922 runtime.py:2921(runtime_blockers)
      32    0.001    0.000   61.510    1.922 preflight.py:246(preflight_blockers)
      32    0.030    0.001   41.861    1.308 preflight.py:8(model_requirements)
     300    0.001    0.000   28.035    0.093 yaml.safe_load
      32    0.000    0.000   19.279    0.602 runtime.py:2819(source_clean_and_pinned)
      32    0.003    0.000   19.277    0.602 vendored_source.py:307(source_clean_and_pinned)
      32    0.066    0.002   19.274    0.602 vendored_source.py:188(source_provenance)
      30    0.553    0.018   14.842    0.495 fashioniq_layout.py:111(image_extension_problems)
  546042    0.205    0.000   12.327    0.000 pathlib.py:510(stat)
      32    0.122    0.004   11.016    0.344 vendored_source.py:118(live_digest)
```

Distinct-input measurement (real handler, fresh instrumented process): `SECONDS 92.265`; `DISTINCT yaml payloads 3 calls 307`; payloads were `preparation_contracts.yaml` 157 calls, `auxiliary_assets.yaml` 138 calls, `models.yaml` 12 calls. Thus 307 YAML parses came from only 3 distinct files. The requested `preflight.load_preparation_contracts` and `preflight.load_auxiliary_assets` symbols do not exist (`MISSING`); loaders reside in `workbench.backend.registry`. I then wrapped those actual registry functions: prep/aux keys 1, calls 295, max repeat 295 (the functions take no arguments and each resolves its default path). `source_provenance`: 17 distinct model/source input keys, 32 calls, max repeats 6 (LIMN); some work is model-level repeated for bundle members, and each model also appears in multiple checkpoint availability checks. This is repeated, not 307 distinct YAML inputs.

Fresh-process one-call measurement with no wrappers: `88.988` seconds. This independently confirms that first-ever handler call takes ~89s in current shared host conditions.

Hoisting opportunity confirmed by `grep`: `main.py:164-167` evaluates `checkpoint_availability(model,item)` for every `model["checkpoint_variants"]`; `main.py:175-178` evaluates it again for each bundle member. Each checkpoint call invokes `runtime_blockers` and `model_requirements`, although those preflight requirements depend on model/config, not checkpoint. Minimal code-free-design recommendation to fixer: hoist/cache the model-level preflight result once per model/request and pass/reuse it across variants and bundles. Pure YAML parser memoization by `(resolved_path, st_mtime_ns, st_size)` is also a small safe candidate for registry's `load_preparation_contracts` and `load_auxiliary_assets`; do not memoize `_member_digest`. Config edit preserving both size and mtime can serve stale parsed config until process restart; accepted risk consistent with existing mtime-only manifest cache. Dataset/source scan caches need validity-safe keys, not only provenance-record stat.

Manifest evidence: checkpoint root `/home/heheboiz/data/cir/workbench/artifacts/checkpoints`; `download_manifest.json` exists with 3 entries (`limn` filenames `0_dress_best_model.pt`, `0_shirt_best_model.pt`, `0_toptee_best_model.pt`). `_checkpoint_manifest_key` returns `f"{model_id}\\0{filename}"`. Exactly 3 manifest keys match served variants; other model checkpoints have no entries. Manifest shortcut applies only matched single files with recorded size and ctime_ns matching; bundles bypass manifest and call `_cached_availability_digest` directly.

`_availability_digest` is called from `checkpoint_missing_paths` to decide checkpoint presence and to return `local_sha256`; it cannot be dismissed as purely decorative. It has a process cache keyed `(str(path), size, mtime_ns)`. Runtime `_member_digest` is separate and explicitly hashes every time because stale capability data must not survive in-place changes. Do not weaken it. The largest checkpoint is 2,730,715,930 bytes; two `sha256sum` runs took 3.633s and 3.859s, one file alone does not explain warm ~28s.

Smoke rerun command, run twice:

```text
PLAYWRIGHT_BROWSERS_PATH=/tmp/pw-browsers .venv/bin/python workbench/frontend/tests/ui_smoke.py --url http://127.0.0.1:5173 --shots /dev/shm/a38-w5b/harness-shots --session all 2>&1 | tail -30
```

Both reruns returned exit 0. Both reached mobile inventory and query; exact output included `mobile inventory: 17/17 model rows from /api/models, detail column reads left=195 right=378 width=183` and `mobile query: body.scrollWidth=390 innerWidth=390, widest img=150px, 12/12 loaded`. Summary had no unexpected console/page/network errors; only deliberately injected query failure. Earlier URL-open timeout is a COLD-request risk under shared-host load: fresh-process `models()` took 88.988s (earlier fresh instrumented call 96.602s), at/above harness `NAV_TIMEOUT_MS=90_000` in instrumented run; two warm reruns passed. Timeout not reproduced warm; cold first page load can approach/exceed harness timeout.

Classification: CONFIRMED MEDIUM `/api/models` latency. Fresh-process handler 88.988s uninstrumented / 96.602s instrumented; same-process warm handler 28.524s; three live warm curls ~27.7s. Distinct-input profiling found 307 `safe_load` calls across only 3 YAML files, and 295 prep/aux loader calls through zero-argument defaults. Hoist model-level preflight out of per-checkpoint calls; cache pure parsers only, keyed `(resolved_path, st_mtime_ns, st_size)`. Do not memoize `_member_digest`; retain correctness-safe dataset/source scan invalidation. A config edit preserving size+mtime may serve stale parse until restart (accepted, same assumption as manifest cache). No implementation or frontend test added.

## Limitations

- This shared environment has a crowded backend, 18G checkpoint tree, critically full root filesystem, and ongoing workstream requests. Cold request measurement varied (88.988s uninstrumented; 96.602s wrapped), indicating shared-host load effects.
- Model/checkpoint detail routes are not separate frontend pages. No real non-mock evaluation artifacts were loaded; existing runs are labelled development fixtures. Export was not exercised. Loading/error states were not captured once per viewport.
- The full smoke harness failed once on a mobile `/api/models` timeout, then passed twice warm; see cold-vs-warm finding. No frontend package test script exists. Build skipped for disk pressure.
- No source code or tests modified. Screenshots/scratch under `/dev/shm/a38-w5/` were removed.
