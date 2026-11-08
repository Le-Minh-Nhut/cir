# FACT SHEET — verified numbers only

Date: 2026-11-09. Branch `research/cir-failure-workbench`. Working dir `/home/heheboiz/data/cir`.
Every number below was read out of the named file in this pass; nothing was carried over from memory.
Derived values are labelled **derived** and show the arithmetic. Figures that exist only in prose are
collected under `KHÔNG TRUY ĐƯỢC` and are **not** citable as measurements.

Extraction scripts and their raw dumps live beside this file — `_extract.py`→`_extract.out` (§2),
`_runs.py`→`_runs.txt` (§2e), `_fails.py`→`_fails.txt` (§3c), `_registry.py`→`_registry.out` (§6),
`_staged_real.py`→`_staged_real.out` (§6 staging), `_smoke.py`→`_smoke.out` (§2c/2d), `_api.py`→`_api.out`
(§7) — so a writer can re-derive every cell. Two files were dropped as superseded and are **not** in the
tree: `_staged.out`/`_staged.py` (a `<root>/<model_id>/<filename>` join that under-counted staging 11 vs
21; replaced by `_staged_real.*`) and `_keys.py`/`_keys.out` (a key-presence dump whose facts are already
in `_extract.out` and `_smoke.out`).

---

## 1. Git and service

| fact | value | source |
|---|---|---|
| `git rev-parse HEAD` | `992b25d254367eec919ea729b2d0df9c09343419` | command output |
| `git rev-parse origin/research/cir-failure-workbench` | `992b25d254367eec919ea729b2d0df9c09343419` | command output |
| HEAD == origin | **yes**, identical 40-hex string | the two commands above |
| `git status --short` at session start | **empty** | command output |
| `git status --short` after this pass | `?? workbench/docs/reviews/20261109-facts/` (only this pass's own new directory) | command output |
| `TESTED_SOURCE_SHA` | `1129c58ba8fa0566c5840cfe820292ae4e9e84f3` | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:22` (repeated at `:38`) |
| `git diff --quiet 1129c58… -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'; echo $?` | **`EXIT=0`** | command output |
| ⇒ production tree matches the recorded tested revision | **yes** | the exit code above; `git diff --stat` for the same range is empty |
| commits between tested SHA and HEAD | 5: `116c869`, `1191d26`, `f57fb0d`, `d0252bb`, `992b25d` | `git log --oneline 1129c58..HEAD` |
| files touched by those 5 commits | 9, **all** under `workbench/docs/` (`git diff --name-status` shows 7 `M` + 2 `A`, every path starts `workbench/docs/`) | `git diff --name-status 1129c58..HEAD` |

### Served frontend assets

| fact | value | source |
|---|---|---|
| files in `workbench/frontend/dist/assets/` | `index-KTV7ID1e.js` (324400 bytes), `index-wyDXnQOD.css` (15613 bytes) — exactly 2 files | `ls -la` |
| `curl -s http://127.0.0.1:5173/` | HTTP 200, body: `<script type="module" crossorigin src="/assets/index-KTV7ID1e.js"></script>` / `<link rel="stylesheet" crossorigin href="/assets/index-wyDXnQOD.css">` | curl output |
| served names vs dist names | **identical** — one `.js`, one `.css`, same hashes | curl output vs `ls` |
| md5 served `index-KTV7ID1e.js` vs dist copy | `d7212325c799f1543dd6aca1304eadde` == `d7212325c799f1543dd6aca1304eadde` | `curl … \| md5sum`, `md5sum` |
| md5 served `index-wyDXnQOD.css` vs dist copy | `63c520553b94b42cbe2d27578d54d3d6` == `63c520553b94b42cbe2d27578d54d3d6` | same |
| `dist/index.html` on disk | byte-identical to the served HTML | `cat workbench/frontend/dist/index.html` |
| `dist/` is git-ignored | yes — `.gitignore:95:workbench/frontend/dist/` | `git check-ignore -v` |

Note for the writer: `dist/` is untracked, so "the served bundle matches `dist/`" is a local-disk fact,
not a committed one. What *is* provable is that the server on 5173 serves bytes identical to the
`dist/` directory present on this machine.

---

## 2. CIRCUS cohort — exact digits

14 directories under `workbench/artifacts/composition/summary/`. `full.json` exists in all 14; sizes
range 1803 B (fixtures) to 11450 B. Two are synthetic fixtures, two are 8-query smoke runs — marked
below and kept out of the model table's metric comparison where they would mislead.

### 2a. Model runs (full 6003-query runs, non-synthetic)

| model_id | run_id | exec | gap_ndcg | gap_cohort | noise% | seed / seed_applied | topk_export | ref_eligible | source |
|---|---|---|---|---:|---:|---|---|---|---|---|
| `conesep` | `30f6de423861cc1d6c78c361` | COMPLETED | `0.3942551173824983` | `6003` | `20` | `0` / `false` | requested=`true`, depth=`50`, 3 conds, status=`TOPK_EXPORT_COMPLETE`, missing=`0` | `true` | `…/summary/conesep_fiq_n02_30f6de423861cc1d6c78c361/full.json` |
| `encoder` | `6a528e877c328772e2c7bb28` | COMPLETED | `0.47230210227676583` | `6003` | `null` | `0` / key absent | key absent (`null`) | `true` (top-level `reference_eligible`); **no `reference_eligibility` block** | `…/encoder_fashioniq_6a528e877c328772e2c7bb28/full.json` |
| `habit` | `497c25c57fbeba885ecfb9d1` | COMPLETED | `0.388704727214254` | `6003` | `20` | `0` / key absent | requested=`true`, depth=`50`, 3 conds, `TOPK_EXPORT_COMPLETE` | `true` | `…/habit_fiq_n02_497c25c57fbeba885ecfb9d1/full.json` |
| `habit` | `e0dd04a46ef7a58334b9e353` | **FAILED** | `null` | `0` | `20` | `0` / key absent | key absent | `true` | `…/habit_fiq_n02_e0dd04a46ef7a58334b9e353/full.json` |
| `hint` | `4bc9893a977e501b82378e76` | COMPLETED | `0.43870420817223343` | `6003` | `null` | `0` / `false` | requested=`true`, depth=`50`, 3 conds, `TOPK_EXPORT_COMPLETE` | `true` | `…/hint_fashioniq_4bc9893a977e501b82378e76/full.json` |
| `intent` | `37ee72aa298b276d1bfda0fa` | COMPLETED | `0.37538250032494136` | `6003` | `20` | `0` / key absent | requested=`true`, depth=`50`, 3 conds, `TOPK_EXPORT_COMPLETE` | `true` | `…/intent_fiq_n02_37ee72aa298b276d1bfda0fa/full.json` |
| `limn` | `10173dc45dea8c90fd2a284b` | **PARTIAL** | `0.5417445849251903` | `5884` | `null` | `0` / key absent | key absent | `true` | `…/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b/full.json` |
| `ptha_mtst` | `315dd590eeb03d2c78d7937a` | COMPLETED | `0.24362219946484542` | `6003` | `null` | `0` / key absent | key absent | `true` | `…/ptha_mtst_cirr_transfer_315dd590eeb03d2c78d7937a/full.json` |
| `tme` | `ad477312aa57c1b9114f9e51` | COMPLETED | `0.3586012581618244` | `6003` | `0` | `0` / key absent | key absent | `true` | `…/tme_fashioniq_0_ad477312aa57c1b9114f9e51/full.json` |
| `released_audit` | `released_audit_task7_mm_any` | COMPLETED | `0.3667022194996058` | **key absent** | `null` | `0` / key absent | key absent | **key absent** | `…/released_audit_task7_mm_any_released_audit_task7_mm_any/full.json` |

`checkpoint_training_noise_pct` is `null` for synthetic the encoder/hint/limn/ptha/released rows, `20`
for conesep/habit/intent, `0` for tme; `evaluation_noise_pct` is `0` for every non-synthetic row except
`released_audit`, where the key is absent. `reproduction_status` per row: conesep/habit/intent/hint/tme
`NOT_RUN`; encoder `METRICS_MEASURED_MAPPING_UNVERIFIED`; limn `BLOCKED_RUNTIME_PREREQUISITES`;
ptha_mtst `SELECTED_TRANSFER_ONLY`; released_audit `RELEASED_ARTIFACT_IMPORT`.

### 2b. Per-condition metrics, full precision (units: R@10/R@50 are percent, ndcg is 0–1)

| model_id | run_id | condition | recall_at_10 | recall_at_50 | ndcg_full | coverage | scored_count |
|---|---|---|---|---:|---:|---:|---:|---:|
| conesep | 30f6de… | multimodal | `35.48225887056472` | `54.13959686823255` | `0.3294414713812836` | `1.0` | `6003` |
| conesep | 30f6de… | text_only_black_image | `15.092453773113442` | `26.869898384141262` | `0.19955748541119267` | `1.0` | `6003` |
| conesep | 30f6de… | image_only_empty_text | `3.1984007996001997` | `9.611860736298517` | `0.11206403391502913` | `1.0` | `6003` |
| encoder | 6a528e… | multimodal | `31.01782442112277` | `49.5752123938031` | `0.29586071325251023` | `1.0` | `6003` |
| encoder | 6a528e… | text_only_black_image | `8.579043811427619` | `18.1575878727303` | `0.15612507640224627` | `1.0` | `6003` |
| encoder | 6a528e… | image_only_empty_text | `2.6653339996668333` | `7.4795935365650505` | `0.10727509765806255` | `1.0` | `6003` |
| habit | 497c25… | multimodal | `35.61552557054806` | `54.3728135932034` | `0.32828280509551133` | `1.0` | `6003` |
| habit | 497c25… | text_only_black_image | `15.292353823088456` | `27.053140096618357` | `0.2006777268917305` | `1.0` | `6003` |
| habit | 497c25… | image_only_empty_text | `3.315009162085624` | `10.31151091121106` | `0.11373588650412819` | `1.0` | `6003` |
| habit | e0dd04… | multimodal | `null` | `null` | `null` | `0.0` | `0` |
| habit | e0dd04… | text_only_black_image | `null` | `null` | `null` | `0.0` | `0` |
| habit | e0dd04… | image_only_empty_text | `null` | `null` | `null` | `0.0` | `0` |
| hint | 4bc989… | multimodal | `32.816924870897886` | `51.77411294352824` | `0.30926916692748224` | `1.0` | `6003` |
| hint | 4bc989… | text_only_black_image | `11.211061136098618` | `21.58920539730135` | `0.17359148193847487` | `1.0` | `6003` |
| hint | 4bc989… | image_only_empty_text | `3.3983008495752123` | `9.928369148758954` | `0.11427446535657035` | `1.0` | `6003` |
| intent | 37ee72… | multimodal | `34.39946693320007` | `54.00633016824921` | `0.3184910253798891` | `1.0` | `6003` |
| intent | 37ee72… | text_only_black_image | `15.32567049808429` | `26.73663168415792` | `0.19893506794173196` | `1.0` | `6003` |
| intent | 37ee72… | image_only_empty_text | `2.865234049641846` | `9.128768948858903` | `0.11223296860976952` | `1.0` | `6003` |
| limn | 10173d… | multimodal | `30.064581917063222` | `48.759347382732834` | `0.2908449032087251` | `0.980176578377478` | `5884` |
| limn | 10173d… | text_only_black_image | `6.3052345343303875` | `14.208021753908906` | `0.13328125184230716` | `0.980176578377478` | `5884` |
| limn | 10173d… | image_only_empty_text | `2.4983004758667575` | `6.985044187627464` | `0.09970453632376666` | `0.980176578377478` | `5884` |
| ptha_mtst | 315dd5… | multimodal | `19.473596535065802` | `34.44944194569382` | `0.22493423438490692` | `1.0` | `6003` |
| ptha_mtst | 315dd5… | text_only_black_image | `10.811261036148592` | `21.0727969348659` | `0.17013526146911484` | `1.0` | `6003` |
| ptha_mtst | 315dd5… | image_only_empty_text | `1.3326669998334166` | `4.6476761619190405` | `0.09498086792403913` | `1.0` | `6003` |
| tme | ad4773… | multimodal | `34.01632517074796` | `53.95635515575545` | `0.32038840538867464` | `1.0` | `6003` |
| tme | ad4773… | text_only_black_image | `16.075295685490588` | `28.035982008995504` | `0.20549672011583525` | `1.0` | `6003` |
| tme | ad4773… | image_only_empty_text | `2.6986506746626686` | `9.22871897384641` | `0.1117059272471363` | `1.0` | `6003` |
| released_audit | released_audit_task7_mm_any | multimodal | `52.84024654339497` | `71.2976844910878` | `0.463455133125359` | `1.0` | `6003` |
| released_audit | released_audit_task7_mm_any | text_only_black_image | `27.33633183408296` | `43.76145260702982` | `0.2935051071698046` | `1.0` | `6003` |
| released_audit | released_audit_task7_mm_any | image_only_empty_text | `6.363484924204564` | `16.14192903548226` | `0.13498441386078872` | `1.0` | `6003` |

All rows sourced from `<summary dir for that run_id>/full.json`, `metrics.<condition>.*`. The JSON stores
the three conditions under keys `multimodal`, `text_only`, `image_only`; `conditions` at top level lists
`["multimodal", "text_only_black_image", "image_only_empty_text"]`. Writer must not confuse the two
namings. `released_audit` additionally carries `metrics.coverage = 1.0` and
`metrics.common_cohort_count = 6003` at the top of `metrics`, which no other non-fixture row has.

**`variant_provenance`** is present only for `conesep`, `habit` (497c25) and `hint`, with keys
`adapted_runtime, adapted_runtime_applied, approved_source_patch, caption_alias_staged, caption_aliases,
checkpoint_training_dataset, runtime_adaptation_selected, warnings`. Values: `adapted_runtime.cuda =
"12.8"`, `pytorch = "==2.7.1"`, `torchvision = "==0.22.1"`, packages
`["transformers==4.25.0","timm==0.9.16","fairscale==0.4.13","numpy<2"]`,
`runtime_adaptation_selected = "sm_120"`, `adapted_runtime_applied = true`, `warnings = []`.

### 2c. Smoke runs — NOT model rows

| model_id | run_id | query_count | smoke.query_limit | exec | gap_ndcg | gap_cohort | note |
|---|---|---|---:|---|---:|---:|---|
| `encoder` | `eda30a37c0bc17ac43844f09` | `8` | `8` | COMPLETED | `0.5329370636334637` | `8` | `smoke.label = "SMOKE QUERY LIMIT; full released gallery preserved"` |
| `limn` | `3f86f58483b368377e2c151e` | `8` | `8` | COMPLETED | `0.5213859188416812` | `8` | same label; `category_scope_failures` all `0` |

Source: the respective `full.json`. Their per-condition values (same keys as 2b):
encoder smoke — mm `12.5` / `25.0` / `0.2517959991456291`; text `0.0` / `0.0` / `0.1176045787263034`;
image `0.0` / `0.0` / `0.08495998091394541`. limn smoke — mm `25.0` / `37.5` /
`0.27140734010891865`; text `12.5` / `12.5` / `0.12989937470585344`; image `0.0` / `0.0` /
`0.07103489291674814` (`mrr = 9.353700190532542e-05`). `query_count_per_subset` for both is
`{"full": 8, "sf": 6, "validated": 0}`.

### 2d. SYNTHETIC fixtures — NOT model rows

| field | `SYNTHETIC_encoder_fashioniq_fixture001` | `SYNTHETIC_habit_fiq_n02_fixture002` |
|---|---|---|
| `model_id` / `checkpoint_id` | `fixture_encoder` / `fashioniq` | `fixture_habit` / `fiq_n02` |
| `is_synthetic` / `synthetic` / `synthetic_fixture` | `true` / `true` / `true` | `true` / `true` / `true` |
| `reproduction_status` | `FIXTURE_DEVELOPMENT_ONLY` | `FIXTURE_DEVELOPMENT_ONLY` |
| `smoke` | `{"limited":true,"query_limit":8,"label":"FIXTURE ONLY"}` | same |
| `query_count` | `6003` | `6003` |
| `metrics.ndcg_multimodal` | `0.2506` | `0.2881` |
| `metrics.ndcg_text_only` | `0.1202` | `0.152` |
| `metrics.ndcg_image_only` | `0.0849` | `0.113` |
| `metrics.composition_gap_ndcg` | `0.5204` | `0.2856` |
| `metrics.recall_at_10_multimodal` | `12.5` | `15.0` |
| `metrics.recall_at_50_multimodal` | `25.0` | `30.0` |
| `metrics.coverage` | `1.0` | `0.98` |
| `metrics.common_cohort_count` | `6003` | `6003` |
| `metrics.outcomes` | required 1080 / text 480 / image 300 / both 720 / unresolved 3301 / neither 0 | same but `neither_scored: 120` |

Source: `…/summary/SYNTHETIC_*/full.json`. **Internal contradiction to flag:** both fixtures assert
`query_count: 6003` and `common_cohort_count: 6003` while `smoke.query_limit` is `8`. They have no
per-condition nested metric blocks at all, only flat `ndcg_*` / `recall_at_*_multimodal` keys, and no
`seed`, `topk_export`, `reference_eligibility` or `variant_provenance` keys. They must never be quoted
beside a 6003-query model row.

### 2e. Run directories under `workbench/artifacts/composition/runs/`

14 directories. `ranks/` exists in all 14. `candidates/` exists in 5 (conesep, intent, habit, hint,
released_audit). Source: `_runs.py` → `_runs.txt` (byte sizes from `os.path.getsize`, line counts by
counting `\n`-terminated records).

| run dir | `ranks/<model>_<ckpt>.json` | bytes | candidates: 3 published `.jsonl`? | lines per file |
|---|---|---:|---|---|
| `10173dc45dea8c90fd2a284b` | `limn_base_iter0_all_categories.json` | 438051 | no `candidates/` | — |
| `30f6de423861cc1d6c78c361` | `conesep_fiq_n02.json` | 379330 | **yes** | 6003 / 6003 / 6003 |
| `315dd590eeb03d2c78d7937a` | `ptha_mtst_cirr_transfer.json` | 385302 | no `candidates/` | — |
| `37ee72aa298b276d1bfda0fa` | `intent_fiq_n02.json` | 379014 | **yes** | 6003 / 6003 / 6003 |
| `3f86f58483b368377e2c151e` | `limn_base_iter0_all_categories.json` | 5207 | no `candidates/` | — |
| `497c25c57fbeba885ecfb9d1` | `habit_fiq_n02.json` | 378866 | **yes**, plus 3 leftover `.tmp` | 6003 / 6003 / 6003 |
| `4bc9893a977e501b82378e76` | `hint_fashioniq.json` | 380422 | **yes** | 6003 / 6003 / 6003 |
| `6a528e877c328772e2c7bb28` | `encoder_fashioniq.json` | 381477 | no `candidates/` | — |
| `SYNTHETIC_encoder_fashioniq_fixture001` | `fixture_encoder_fashioniq.json` | 1443 | no `candidates/` | — |
| `SYNTHETIC_habit_fiq_n02_fixture002` | `fixture_habit_fiq_n02.json` | 1439 | no `candidates/` | — |
| `ad477312aa57c1b9114f9e51` | `tme_fashioniq_0.json` | 378976 | no `candidates/` | — |
| `e0dd04a46ef7a58334b9e353` | `habit_fiq_n02.json` | **4573087** | no `candidates/` | — |
| `eda30a37c0bc17ac43844f09` | `encoder_fashioniq.json` | 5142 | no `candidates/` | — |
| `released_audit_task7_mm_any` | `released_audit_task7_mm_any.json` | 366340 | **yes** | 6003 / 6003 / 6003 |

Candidate file bytes, three conditions each (`image_only_empty_text`, `multimodal`,
`text_only_black_image`) plus a `.manifest.json` (27 lines each):

| run dir | image_only `.jsonl` | multimodal `.jsonl` | text_only `.jsonl` |
|---|---:|---:|---:|
| `30f6de…` | 8308760 | 8291625 | 8313027 |
| `37ee72…` | 8309501 | 8295625 | 8344657 |
| `497c25…` | 8323259 | 8295650 | 8328804 |
| `4bc989…` | 8355602 | 8358202 | 8442695 |
| `released_audit_task7_mm_any` | 1808505 | 1798920 | 1801082 |

Leftover `.tmp` files in `497c25…/candidates/habit_fiq_n02/` (not published, must not be counted):
`.image_only_empty_text.jsonl.3981587.3480b2fdb44543f8943a8b02052f092f.tmp` 2441072 B / 1758 lines;
`.multimodal.jsonl.3981587.09e7a9c29b7549f0967c3625e747be20.tmp` 8291497 B / 6000 lines;
`.text_only_black_image.jsonl.3981587.53b4006fcd3d415aa1b781dcdda3a4d3.tmp` 8324653 B / 6000 lines.

Note: `e0dd04a46ef7a58334b9e353` (FAILED) nonetheless has a `ranks/habit_fiq_n02.json` of 4573087 B —
larger than the successful habit run's 378866 B; it holds no `candidates/`.

---

## 3. Sweeps

### 3a. `workbench/docs/reviews/20261108T212500+0700-sweep/full_suite_c2d8cdeb.txt` (539 lines)

Header table, verbatim (lines 5–15):

| field | value |
|---|---|
| revision | `c2d8cdeb0eabc3232b28070f55b5a10d63ccffcb` |
| command | `workbench/envs/ctl312h/bin/python -m pytest workbench/tests -q -p no:cacheprovider -rfE --tb=no` |
| interpreter | CPython 3.12.11 |
| serial? | "yes — no `-n`, no xdist" |
| duration | `3464.03 s (0:57:44); WALL=3466.58 s including interpreter start` |
| exit code | `1` |
| summary line | `` `18 failed, 2463 passed, 47 skipped, 290 warnings in 3464.03s (0:57:44)` `` |
| collected | `2528` (= 2463 + 18 + 47) |

Tail, verbatim (lines 535–538):
```
18 failed, 2463 passed, 47 skipped, 290 warnings in 3464.03s (0:57:44)
Command exited with non-zero status 1
WALL=3466.58 s
pytest_exit=1
```

### 3b. `workbench/docs/reviews/20261108T225338+0700-sweep/full_suite_1129c58b.txt` (541 lines)

| field | value |
|---|---|
| revision | `1129c58ba8fa0566c5840cfe820292ae4e9e84f3` |
| command | identical to 3a |
| date | 2026-11-08, started 22:53 +0700 |
| duration | `3483.79 s (0:58:03); WALL=3486.52 s` |
| exit code | `1` |
| summary line | `` `18 failed, 2467 passed, 47 skipped, 290 warnings in 3483.79s (0:58:03)` `` |
| collected | `2532` (= 2467 + 18 + 47) |

Tail, verbatim (lines 537–540):
```
18 failed, 2467 passed, 47 skipped, 290 warnings in 3483.79s (0:58:03)
Command exited with non-zero status 1
WALL=3486.52 s
pytest_exit=1
```

Both logs contain a `collected` figure only as a header sentence plus its own arithmetic; neither log
contains a pytest `collected N items` line. Do not cite "2528 collected" or "2532 collected" as a
pytest-emitted number — cite the header line, whose own parenthetical is `2463+18+47` / `2467+18+47`.

Difference between the two sweeps in the raw logs: **the same 18 ids**, verified by extracting the
`FAILED ` lines from both files and comparing the lists — `identical: True`, 18 vs 18 (`_fails.txt`).

Also visible in the logs and worth stating: the `test_no_release_document_points_at_a_superseded_source_sha`
failure text names a *different* SHA in each log — `b0f38bd2bc15efbaf75228f04be8327c9843b45c` at
`full_suite_c2d8cdeb.txt:222` and `c2d8cdeb0eabc3232b28070f55b5a10d63ccffcb` at
`full_suite_1129c58b.txt:224`. At the second sweep the docs still named the *previous* revision; that is
why the test is counted as category (d) and later closed.

### 3c. The 18 failing ids at `c2d8cdeb` (verbatim, in log order)

1. `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_conesep-conesep]`
2. `test_circus_adapters_blip.py::test_resident_worker_reuse_and_cleanup[workbench.composition.model_ptha_mtst-ptha_mtst]`
3. `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_conesep-conesep]`
4. `test_circus_adapters_blip.py::test_resident_worker_failure_uses_shared_error_and_stays_failed[workbench.composition.model_ptha_mtst-ptha_mtst]`
5. `test_circus_cli.py::test_cli_3d_full_gallery_resume_and_gallery_invalidation`
6. `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[encoder]`
7. `test_circus_tme_lifecycle.py::test_encoder_and_limn_failed_child_latch_and_preserve_log[limn]`
8. `test_circus_tme_precision.py::test_worker_skips_empty_stdin_lines`
9. `test_f05_f08_f09_verification.py::test_f08_dcnet_config_data_dir_resolution`
10. `test_handoff_commands.py::test_no_release_document_points_at_a_superseded_source_sha`
11. `test_release_blocker_gpu_records.py::test_defect_b_g_with_no_target_gpu_verdict_is_pending_not_pass`
12. `test_release_blocker_gpu_records.py::test_release_verify_documented_invocations_produce_parseable_report[<lambda>0]`
13. `…[<lambda>1]`
14. `…[<lambda>2]`
15. `…[<lambda>3]`
16. `…[<lambda>4]`
17. `test_release_blocker_gpu_records.py::test_offline_verdict_names_deployment_reason`
18. `test_standalone_operator_commands.py::test_prepare_model_executes_only_declared_recipe_steps`

Source: `full_suite_c2d8cdeb.txt`, the 18 `FAILED ` lines (file paths prefixed `workbench/tests/`).

### 3d. Attribution tally as written in `FAILURES.md`

Table (`FAILURES.md:120-129`):

| category | n | ids |
|---|---:|---|
| **(a)** caused inside `b0f38bd..HEAD` | **8** | #1–#4 (in `aa0f530`), #5 (ordering, in `9b6b5b6`), #6–#7 (test unsatisfiable by construction), #8 (in `aa0f530`) |
| **(h)** harness/test defect | **0** | — |
| **(b)** host state | **9** | #9, #11–#17, #18 |
| **(c)** pre-existing at `b0f38bd` | **0** | — |
| **(d)** obsolete test (now closed) | **1** | #10 |
| **total** | **18** | |

`FAILURES.md:131` states `8 + 0 + 9 + 0 + 1 = 18`.
`FAILURES.md:135-136` restates it as "8 of the 18 … (6 production-caused: #1–#4, #5, #8; 2 tests that
are unsatisfiable by construction: #6, #7)".

**The prose and the table disagree.** `FAILURES.md:193-196` says, verbatim: "the current 18 break down as
**6 inside the range, 2 harness defects, 9 host state, 1 closed**". That is `6+2+9+1 = 18`, but it splits
the table's 8×(a)/0×(h) into 6×(a)/2×(h), contradicting both the table (`:124-128`) and the arithmetic at
`:131` (`8 + 0 + 9 + 0 + 1 = 18`). Table and `:131` agree with each other; `:193-196` is the odd one out.

**Judged correct: the table (a)8 (h)0 (b)9 (c)0 (d)1 — because it is the only tally with per-row
evidence.** Each of the 18 rows at `:39-57` carries its own category and a named proof (the `aa0f530`
ancestry test, the `9b6b5b6` ordering reproduction, the four-reason trace for #6/#7, the clean-worktree
reproduction for #9/#18). The 6/2 split at `:193-196` has no row behind it — `:126` explicitly reclassifies
(h) into (a) and says so in the cell ("reclassified to (a); see the trace"), and `:133` says the "6 caused
/ 2 harness" split is *withdrawn*. So `:193-196` is a leftover of a superseded version that survived the
same edit that withdrew it forty lines earlier. `FAILURES.md` is not fixed here — this is documentation
only.

**GPU-capability drift — a moving target that must carry its timestamp and total.** `FAILURES.md:49`
records, verbatim: "**Exact breakdown of all 632** (recounted): **465** `probe_source: "override"`;
**363** `device_name: "Synthetic GPU"`; **34** `error: "CUDA out of memory"`; **34** with *both*; **329**
Synthetic GPU with `error: null`; **50** with no `device_name`". Measured 2026-11-09 in this pass:
`workbench/artifacts/gpu_capabilities/` holds **648** `.json` files — **477** override, **372**
`device_name="Synthetic GPU"`, **35** OOM, **35** both, **337** Synthetic-GPU-with-null-error, **51**
without a `device_name` key. The directory is shared and mutable (other agents' probes land in it), so
the only honest form of this number is *total + timestamp*:

| observation point | total | source |
|---|---:|---|
| sweep `c2d8cdeb` | **632** (11 named + "and 621 more") | `full_suite_c2d8cdeb.txt`, 12 `FAILED` traces each ending "and 621 more" |
| sweep `1129c58b` | **648** (11 named + "and 637 more") | `full_suite_1129c58b.txt`, 12 traces each ending "and 637 more" |
| 2026-11-09, this pass | **648** | directory listing + JSON parse |

So the drift is `632 → 648` between 2026-11-08 21:25 and 2026-11-08 22:53 +0700, and the directory has
not changed since. Never quote 632/465/363/34/329/50 as current; quote it as "632 as recorded at
`c2d8cdeb` (2026-11-08 21:25 +0700)".

---

## 4. Throughput — only what raw files prove

### 4a. `workbench/docs/reviews/20261108T-throughput/vdecode.txt` (313 bytes, 4 lines, verbatim)

```
workers=2: serial=4.69s parallel=1.02s speedup=4.6x bitwise_equal=True maxabs_delta=0.0
workers=4: serial=4.69s parallel=0.75s speedup=6.2x bitwise_equal=True maxabs_delta=0.0
workers=8: serial=4.69s parallel=0.87s speedup=5.4x bitwise_equal=True maxabs_delta=0.0
serial baseline: 512 imgs in 4.69s = 109.1 img/s
```

Per-worker wall times and **derived** throughput (512 / wall):

| workers | parallel wall | img/s recorded in file | derived 512/wall |
|---|---:|---:|---:|
| serial baseline | 4.69 s | `109.1` | `512/4.69 = 109.168443` |
| 2 | 1.02 s | not stated | `512/1.02 = 501.9608` |
| 4 | 0.75 s | not stated | `512/0.75 = 682.6667` |
| 8 | 0.87 s | not stated | `512/0.87 = 588.5057` |

Derived speedups for cross-check (4.69 / wall): 2 → `4.5980`, 4 → `6.2533`, 8 → `5.3908`; the file's own
rounded values are `4.6x`, `6.2x`, `5.4x`. 4 workers is the fastest row in this file; 8 is slower.
This file measures **decode+transform only** — the file name and the absence of any forward/GPU stage
are the only evidence for that; no line in the file states it.

### 4b. `ein_threads.txt` (1517 bytes, verbatim)

```
=== OPENBLAS_THREADS=1
gallery=(74381, 32, 256) queries=6003 threads_env=1
old                4.89s    6.5 q/s  proj6003q= 15.3min  maxdelta=0.000e+00
optimize=True      1.17s   27.3 q/s  proj6003q=  3.7min  maxdelta=3.576e-07
gemm_chunk4096     1.03s   31.2 q/s  proj6003q=  3.2min  maxdelta=3.576e-07
=== OPENBLAS_THREADS=2
gallery=(74381, 32, 256) queries=6003 threads_env=2
old                5.31s    6.0 q/s  proj6003q= 16.6min  maxdelta=0.000e+00
optimize=True      1.45s   22.1 q/s  proj6003q=  4.5min  maxdelta=3.576e-07
gemm_chunk4096     0.57s   56.2 q/s  proj6003q=  1.8min  maxdelta=3.576e-07
=== OPENBLAS_THREADS=4
gallery=(74381, 32, 256) queries=6003 threads_env=4
old                5.18s    6.2 q/s  proj6003q= 16.2min  maxdelta=0.000e+00
optimize=True      0.94s   33.9 q/s  proj6003q=  2.9min  maxdelta=3.576e-07
gemm_chunk4096     0.37s   86.3 q/s  proj6003q=  1.2min  maxdelta=3.576e-07
=== OPENBLAS_THREADS=8
gallery=(74381, 32, 256) queries=6003 threads_env=8
old                5.76s    5.6 q/s  proj6003q= 18.0min  maxdelta=0.000e+00
optimize=True      1.25s   25.7 q/s  proj6003q=  3.9min  maxdelta=3.576e-07
gemm_chunk4096     0.32s   99.7 q/s  proj6003q=  1.0min  maxdelta=3.576e-07
=== OPENBLAS_THREADS=10
gallery=(74381, 32, 256) queries=6003 threads_env=10
old                5.20s    6.2 q/s  proj6003q= 16.3min  maxdelta=0.000e+00
optimize=True      1.40s   22.9 q/s  proj6003q=  4.4min  maxdelta=2.384e-07
gemm_chunk4096     0.37s   87.7 q/s  proj6003q=  1.1min  maxdelta=2.980e-07
```

Only citable claim: the `old` path stays flat at `6.5 / 6.0 / 6.2 / 5.6 / 6.2` q/s across 1/2/4/8/10
BLAS threads. Every `optimize=True` and `gemm_chunk4096` row has a non-zero `maxdelta` and is therefore
**not** rank-identical to the baseline; see 4e.

### 4c. `batch.txt` (930 bytes, verbatim)

```
slice images: 2000
…/blip2.py:42: FutureWarning: `torch.cuda.amp.autocast(args...)` is deprecated. Please use `torch.amp.autocast('cuda', args...)` instead.
  return torch.cuda.amp.autocast(dtype=dtype)
threads=20  fp32  bs=64   113.77s    17.6 img/s peak_alloc=  3484MiB peak_reserved=  3958MiB
threads=20  fp32  bs=128  110.81s    18.0 img/s peak_alloc=  4350MiB peak_reserved=  5066MiB
threads=20  fp32  bs=256  111.77s    17.9 img/s peak_alloc=  6082MiB peak_reserved=  7306MiB
threads=20  fp32  bs=512  111.15s    18.0 img/s peak_alloc=  9544MiB peak_reserved= 11788MiB
threads=20  bf16  bs=64   112.80s    17.7 img/s peak_alloc=  3484MiB peak_reserved=  3976MiB
threads=20  bf16  bs=128  116.84s    17.1 img/s peak_alloc=  4350MiB peak_reserved=  5066MiB
threads=20  bf16  bs=256  123.72s    16.2 img/s peak_alloc=  6082MiB peak_reserved=  7306MiB
```

Range: fp32 `17.6–18.0 img/s`, bf16 `16.2–17.7 img/s`, over a 2000-image slice. Saturates by bs=128
(18.0 at both 128 and 512). Peak allocation grows `3484 → 9544 MiB`, reserved `3958 → 11788 MiB`.

### 4d. `d2_gate_full_6003x3.txt` (382 bytes, verbatim)

```
gallery=(74381, 32, 256) queries=6003 threads=8 chunk_size=32
multimodal: n=6003  183.0s  32.8 q/s mismatches=0 identical=True
text_only_black_image: n=6003  196.0s  30.6 q/s mismatches=0 identical=True
image_only_empty_text: n=6003  184.3s  32.6 q/s mismatches=0 identical=True
VERDICT: PASS - shipped path reproduces the published ranks exactly on all 18009 query-condition pairs
```

Verdict line: `VERDICT: PASS - shipped path reproduces the published ranks exactly on all 18009
query-condition pairs`. Per-condition rows as above. Derived: `6003 × 3 = 18009` pairs (matches the
verdict). Companion machine-readable file `d2_gate_full_6003x3.json` (675 B) carries the same numbers
plus `"threads": 8`, `"chunk_size": 32`, `"all_identical": true`, and `"rank_mismatches_vs_artifact": 0`
per condition — cite the JSON when a parser is wanted.

### 4e. `d3_serial_blas_compare.txt` (549 bytes, verbatim)

```
OPENBLAS_NUM_THREADS in env = 10
serial BLAS=1  vs serial BLAS=10: queries=512
  ranks_identical = False
  queries_changed_rank = 11
  max_abs_drank = 1
    (77, 38550, 38549)
    (112, 39674, 39673)
    (127, 7510, 7509)
    (234, 71919, 71918)
    (272, 58976, 58977)
    (281, 7896, 7897)
    (368, 52179, 52180)
    (397, 61269, 61270)
    (412, 50560, 50561)
    (435, 24525, 24526)
  score_max_abs_delta = 4.172e-07
VERDICT: LOAD-BEARING: the two BLAS settings produce different ranks, so the pin is the only reason the knob is currently safe
```

The file prints **10** of the **11** changed queries it counts (`queries_changed_rank = 11`). Do not
claim "11 examples listed".

### 4f. `d3_hoist_proof.txt` (546 bytes, verbatim)

```
OPENBLAS_NUM_THREADS in env = 10 (the value a serial run used to inherit)
threads=1 (BLAS pinned to 1):   2.43s   210.9 q/s
serial WITHOUT the pin (BLAS=10):   1.70s   301.8 q/s
threads=2 (BLAS pinned to 1):   1.65s   310.6 q/s  ranks_identical_to_threads_1=True
threads=4 (BLAS pinned to 1):   0.93s   549.9 q/s  ranks_identical_to_threads_1=True
threads=8 (BLAS pinned to 1):   0.92s   556.5 q/s  ranks_identical_to_threads_1=True
VERDICT: PASS - @-based scorer ranks identical across WORKBENCH_SCORE_THREADS 1/2/4/8 under the hoisted BLAS pin
```

Note the counter-intuitive pairing, stated plainly by the file: the *serial* run without the pin
(`1.70 s`, `301.8 q/s`) is **faster** than pinned `threads=1` (`2.43 s`, `210.9 q/s`). Derived: 8-thread
vs pinned-1-thread = `2.43/0.92 = 2.641×`; 4-thread = `2.43/0.93 = 2.613×`.

### 4g. `KHÔNG TRUY ĐƯỢC` — prose-only throughput figures

Search method: for each of the four tokens, substring search over **every** `.txt`, `.json`, `.log`,
`.sh`, `.py` file under `workbench/docs/reviews/20261108T-throughput/` (34 files when restricted to
`.txt/.json/.log`) and under `workbench/docs/reviews/20261108T173736+0700/` (0 non-`.md` files there).

| figure | appears in any raw file? | where the prose lives |
|---|---|---|
| `21.2` (img/s BEFORE) | **no — 0 files** | `20261108T-throughput/THROUGHPUT.vi.md:22`, `:27`, `:112`; `20261108T173736+0700/EVIDENCE_INDEX.md:89`, `:96`, `:143`; `GATE_MATRIX.md:65`, `:68`, `:81`, `:102`; `INPUT_PIPELINE_AND_INFERENCE_AUDIT.vi.md:166`, `:170`, `:998`, `:1130` |
| `2089` / `2089.2` (img/s AFTER) | **no — 0 files** | `THROUGHPUT.vi.md:22`, `:27`, `:113`; `EVIDENCE_INDEX.md:89`; `GATE_MATRIX.md:65`, `:68`, `:102`; `INPUT_PIPELINE…:170`, `:999`, `:1131` |
| `2192` (img/s, 8-worker row) | **no — 0 files** | `THROUGHPUT.vi.md:99` (row `8 | 0.47s | 2192 | 390s`); `INPUT_PIPELINE…:989`; discussed as unsourced at `INPUT_PIPELINE…:170`, `:992`, `:1131`; `GATE_MATRIX.md:102` |
| `5.67` (query-axis speedup at 8 threads) | **no — 0 files** | `THROUGHPUT.vi.md:320` (row `8 | 18.7 s | 27.4 | 5.67×`), `:332`; `EVIDENCE_INDEX.md:89`, `:143`; `GATE_MATRIX.md:98`; `INPUT_PIPELINE…:144`, `:166`, `:1049`, `:1378` |

The `2192` row's own arithmetic does not fit `vdecode.txt`: `0.47 s × 2192 img/s ≈ 1030` images, not
512. The three rows `4 | 0.61s | 1672`, `8 | 0.47s | 2192`, `10 | 0.53s | 1915` imply ~1020/1030/1015
images per row — a different experiment size from the 512 in `vdecode.txt`. `1672` and `1915` also
appear in **zero** raw files.

Further prose-only figures with no raw source, found while checking (same two `.md` trees): "~84 img/s
forward saturation", "~886 s for 74,381 images", "~73 min → ~15 min end-to-end", "58.5 phút / 0.6
phút", "1/2/4/8 threads → 4.8/8.7/15.7/27.4 q/s", "`1.81× / 3.26×`" query-axis speedups, "transform 27
img/s at 20 threads vs 650 at 1 thread". Each token was grepped: `1672`, `1915`, `84 img`, `58.5`,
`1.81`, `3.26`, `27.4`, `0.61`, `0.53`, `650` return **0 hits** in raw files of the throughput dir
(`0.47` and `886` and `81.0` do hit, but in unrelated files — `d3_blas_collision_proof.txt` for `0.47`,
`metric_gate*.json` for `886`, `query.txt` for `81.0` — not as the claimed quantity).

---

## 5. LIMN scope resolution

### 5a. `20261108T180000+0700-limn-scope/scope_resolution.json` (378 bytes, verbatim)

```json
{"counts": {"dress": 2010, "shirt": 2035, "None": 2, "toptee": 1956},
 "nulls": ["7:3038", "7:3790"],
 "loader_assigned": 5884,
 "changed_loader_assigned": [],
 "old_scope_digest": "8a5dd517595eb4883d8b3174f410f87980fbc612418231e2be4e0e3847cd1c19",
 "new_scope_digest": "3a1c5a6763154870536f3586491d5bea9d87b73a9836363dc298d2accdd34d87"}
```

Per-scope counts: `dress 2010`, `shirt 2035`, `toptee 1956`, `None 2`. Derived total:
`2010 + 2035 + 1956 + 2 = 6003`.
Remaining `None` qids: exactly `["7:3038", "7:3790"]` — 2 entries.
`loader_assigned = 5884`. `changed_loader_assigned = []` — **0** queries changed scope.
Two digests: old (loader-category-only) `8a5dd517…`, new (with resolution) `3a1c5a67…`.

### 5b. `20261108T180000+0700-limn-scope/identity_proof.json` (471 bytes, verbatim)

```json
{"published_run_id": "10173dc45dea8c90fd2a284b",
 "recomputed_from_stored_identity": "10173dc45dea8c90fd2a284b",
 "reproduces_published": true,
 "new_run_id_with_scope_resolution": "2a4b19f335f5ec281375d09f",
 "run_id_differs": true,
 "fingerprint_differs": true,
 "resume_would_reuse_old": false,
 "scope_resolution_digest": "3a1c5a6763154870536f3586491d5bea9d87b73a9836363dc298d2accdd34d87",
 "stable_for_identical_map": true,
 "moves_for_one_flip": true}
```

Old run_id `10173dc45dea8c90fd2a284b`; new run_id with scope resolution `2a4b19f335f5ec281375d09f`;
`run_id_differs = true`, `fingerprint_differs = true`, `resume_would_reuse_old = false`.

### 5c. What is proven, and what is not

**Proven (no GPU needed, both from the two JSON files above):**
- The loader leaves exactly 2 of 6003 queries without a scope; the resolver leaves those 2 unscored.
- The resolver never overrode the loader: `changed_loader_assigned = []`.
- The new identity moves: `run_id` and fingerprint differ from the published run, so `--resume` cannot
  silently reuse `10173dc45dea8c90fd2a284b`.
- The digest is stable for an identical map and moves for a one-scope flip.
- The published run's *own* stored identity recomputes to `10173dc45dea8c90fd2a284b`
  (`reproduces_published = true`).

**Not proven, needs a GPU re-run:**
- Any published LIMN metric at the resolved scope. `limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b`
  still reports `scored_count 5884` of `query_count 6003`, `coverage 0.980176578377478`,
  `execution_status PARTIAL`, `category_scope_failure_count 119` per condition (source: that run's
  `full.json`). So its `30.064581917063222` MM R@10 is a 5884-denominator number and is **not**
  comparable at coverage 1.0 with the seven rows in §2b that show `coverage 1.0`.
- `20261108T180000+0700-limn-scope/EVIDENCE.md` §6 says the same ("The wiring is proven … A re-run is
  required to publish a 6001/6003 figure; expected coverage `0.9996668…`"). The expected coverage
  `0.9996668` is **prose-only** — no JSON carries it. Derived check: `6001/6003 = 0.9996668…`, consistent.

Two different counts must not be conflated: the published run's `category_scope_failure_count` is **119**
(a fixed, already-published run), while the new resolution's `None` count is **2**. Both are in files;
they measure different things (unscored-in-published-run vs unscoped-after-resolution).

---

## 6. Models / repos

Source for all of §6 unless stated: `workbench/registry/models.yaml`, loaded with
`workbench/envs/ctl312h/bin/python` + `yaml.safe_load` (`_registry.py` → `_registry.out`).

| count | value |
|---|---|
| `models` entries | **17** |
| distinct `upstream_repo_url` | **16** (only `combiner_rn50x4_noft` and `clip4cir_rn50x4_fullft` share one: `https://github.com/ABaldrati/CLIP4Cir`) |
| `checkpoint_variants` entries (summed over all models) | **29** |
| entries with `source_available` truthy | **17** |
| `schema_version` | `1` |

| model_id | native_protocol | supported_protocols[0] | composition_adapter | reproduction_status |
|---|---|---|---|---|
| `csmcir` | `fashioniq_original_split` | `fashioniq_original_split` | `workbench.composition.model_csmcir` | `NOT_RUN` |
| `airknow` | `fashioniq_full_gallery_ref_excluded` | same | **none** | `NOT_RUN` |
| `conesep` | `fashioniq_full_gallery_ref_excluded` | same | `workbench.composition.model_conesep` | `NOT_RUN` |
| `ptha_mtst` | `fashioniq_original_split` | same | `workbench.composition.model_ptha_mtst` | `SELECTED_TRANSFER_ONLY` |
| `habit` | `fashioniq_full_gallery_ref_excluded` | same | `workbench.composition.model_habit` | `NOT_RUN` |
| `intent` | `fashioniq_full_gallery_ref_excluded` | same | `workbench.composition.model_intent` | `NOT_RUN` |
| `hint` | `fashioniq_val_split` | same | `workbench.composition.model_hint` | `NOT_RUN` |
| `encoder` | `fashioniq_val_split` | same | `workbench.composition.model_encoder` | `METRICS_MEASURED_MAPPING_UNVERIFIED` |
| `pair` | `fashioniq_val_split` | same | **none** | `BLOCKED_CHECKPOINT_MAPPING_UNVERIFIED` |
| `clvc_net` | `fashioniq_val_split` | same | **none** | `BLOCKED_CHECKPOINT_MAPPING` |
| `dcnet` | `fashioniq_full_gallery_ref_excluded` | same | **none** | `BLOCKED_CHECKPOINT_MAPPING` |
| `combiner_rn50x4_noft` | `fashioniq_original_split` | same | **none** | `BLOCKED_CHECKPOINT_MAPPING` |
| `clip4cir_rn50x4_fullft` | `fashioniq_original_split` | same | **none** | `BLOCKED_CHECKPOINT_BUNDLE` |
| `tgcir` | `fashioniq_val_split` | same | **none** | `BLOCKED_COMMAND` |
| `sprc` | `fashioniq_original_split` | same | **none** | `BLOCKED_CHECKPOINT_MAPPING` |
| `limn` | `fashioniq_val_split` | same | `workbench.composition.model_limn` | `BLOCKED_RUNTIME_PREREQUISITES` |
| `tme` | `fashioniq_original_split` | same | `workbench.composition.model_tme` | `NOT_RUN` |

`native_protocol` equals `supported_protocols[0]` for **all 17** models (each declares exactly one
supported protocol). Models with a composition adapter: **9** (`csmcir`, `conesep`, `ptha_mtst`,
`habit`, `intent`, `hint`, `encoder`, `limn`, `tme`); without: **8**.

### Checkpoint staging under `WORKBENCH_CHECKPOINT_ROOT`

`workbench/config/workbench.env` value (read only, not modified):
`WORKBENCH_CHECKPOINT_ROOT=/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads`
— directory exists.

**Resolution must use the production resolver, not a `<root>/<model_id>/<filename>` join.**
`workbench/backend/registry.py::checkpoint_path` tries `configured / model_id` first and falls back to
`configured / checkpoint_base_relative` only when the conventional directory is *absent*. Seventeen of
the 29 variants declare a `checkpoint_base_relative` (e.g. `../../checkpoints/airknow`,
`../../downloads/tgcir/x/checkpoints`), so a naive join under-counts by 10. Correct figure, computed by
importing `checkpoint_path` itself (`_staged_real.py` → `_staged_real.out`, run with
`PYTHONPATH=/home/heheboiz/data/cir`):

> **21 of 29 staged under `WORKBENCH_CHECKPOINT_ROOT`** — 11 by the conventional `<root>/<model_id>`
> rule, **10** by the `checkpoint_base_relative` fallback. **8 absent.**

| model_id | checkpoint_id | rule that resolved it | exists | resolved path (relative to root) |
|---|---|---|---|---|
| csmcir | fashioniq | conventional | **yes** | `csmcir/fashioniq_tuned_clip_best.pt` |
| airknow | fiq_n05 | `checkpoint_base_relative` | **yes** | `../../checkpoints/airknow/airknow_fiq_0.5.pt` |
| airknow | fiq_n08 | `checkpoint_base_relative` | **yes** | `../../checkpoints/airknow/airknow_fiq_0.8.pt` |
| conesep | fiq_n02 | conventional | **yes** | `conesep/ConeSep-FIQ-N0.2.pt` |
| conesep | fiq_n05 | conventional | no | `conesep/ConeSep-FIQ-N0.5.pt` |
| conesep | fiq_n08 | conventional | no | `conesep/ConeSep-FIQ_N0.8.pt` |
| ptha_mtst | cirr_transfer | conventional | **yes** | `ptha_mtst/tuned_mtst_cirr.pt` |
| habit | fiq_n02 | conventional | **yes** | `habit/HABIT-FIQ_N0.2.pt` |
| habit | fiq_n05 | conventional | no | `habit/HABIT-FIQ_N0.5.pt` |
| habit | fiq_n08 | conventional | no | `habit/HABIT-FIQ_N0.8.pt` |
| intent | fiq_n02 | conventional | **yes** | `intent/intent_fiq_0.2.pt` |
| intent | fiq_n05 | conventional | no | `intent/intent_fiq_0.5.pt` |
| intent | fiq_n08 | conventional | no | `intent/intent_fiq_0.8.pt` |
| hint | fashioniq | conventional | **yes** | `hint/fashioniq.pt` |
| encoder | fashioniq | conventional | **yes** | `encoder/fashioniq.pt` |
| pair | pair_b1 | `checkpoint_base_relative` | **yes** | `../../checkpoints/pair/pair-B1.pt` |
| pair | pair_b2 | `checkpoint_base_relative` | **yes** | `../../checkpoints/pair/pair-B2.pt` |
| clvc_net | author_drive_unresolved | `checkpoint_base_relative` | **yes** (directory) | `../../checkpoints/clvc` |
| dcnet | fashioniq_run_directory | `checkpoint_base_relative` | **yes** (directory) | `../../checkpoints/dcnet` |
| combiner_rn50x4_noft | rn50x4_noft_unresolved | `checkpoint_base_relative` | **yes** | `../../checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` |
| clip4cir_rn50x4_fullft | rn50x4_fullft_unresolved | `checkpoint_base_relative` | **yes** (directory) | `../../checkpoints/clip4cir/FashionIQ/RN50x4_fullft` |
| tgcir | tgcir_archive_unresolved | `checkpoint_base_relative` | **yes** | `../../downloads/tgcir/x/checkpoints/Fashioniq.pt` |
| sprc | sprc_fiq_unresolved | `checkpoint_base_relative` | no | `../../checkpoints/sprc/sprc_fiq.pt` |
| sprc | sprc_fiq_vitl | `checkpoint_base_relative` | **yes** | `../../checkpoints/sprc/sprc_fiq_vitl.pt` |
| limn | base_iter0_dress | conventional | **yes** | `limn/0_dress_best_model.pt` |
| limn | base_iter0_shirt | conventional | **yes** | `limn/0_shirt_best_model.pt` |
| limn | base_iter0_toptee | conventional | **yes** | `limn/0_toptee_best_model.pt` |
| tme | fashioniq_0 | conventional | **yes** | `tme/fashioniq_0_best_model.pth` |
| tme | cirr_0 | conventional | no | `tme/cirr_0_best_model.pth` |

The 8 absent, exactly: `conesep/fiq_n05`, `conesep/fiq_n08`, `habit/fiq_n05`, `habit/fiq_n08`,
`intent/fiq_n05`, `intent/fiq_n08`, `sprc/sprc_fiq_unresolved`, `tme/cirr_0`. Sizes of the 10
relocated artifacts, measured with `stat` and equal to each variant's `expected_size_bytes`:
`airknow_fiq_0.5.pt` 2724056638, `airknow_fiq_0.8.pt` 2724056638, `pair-B1.pt` 694958565,
`pair-B2.pt` 694872371, `fiq_comb_RN50x4_noft.pt` 236017921, `Fashioniq.pt` 607928637,
`sprc_fiq_vitl.pt` 1459756541; `clvc` and `dcnet` and `RN50x4_fullft` are directories.

**Two roots, two different answers — do not mix them.** The sibling native tree
`workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/` (exists; holds `airknow`,
`clip4cir`, `clvc`, `dcnet`, `pair`, `sprc`) resolves only **6 of 29** as a *configured* root (airknow ×2,
pair ×2, dcnet, sprc_fiq_vitl), because the composition weights are not in it. The 21/29 figure is
specific to `WORKBENCH_CHECKPOINT_ROOT`; the union across both roots is also 21, since the 6 native hits
are a subset of the 10 relocations already reached from the CIRCUS root.

Both `hint` and `encoder` declare the filename `fashioniq.pt`; the per-model directory disambiguates
them (`hint/fashioniq.pt` 2724451182 B vs `encoder/fashioniq.pt` 631879365 B), so a basename-only lookup
will silently match encoder's 631879365-byte expectation to hint's 2.7 GB file. Match on
`model_id/filename`.

All 9 adapter-bearing models have at least one staged variant; `habit` has only `fiq_n02` and
`conesep`/`intent` only `fiq_n02`, so the noise-sweep variants (`fiq_n05`, `fiq_n08`) are simply not
available for those three.

---

## 7. API truth

Source: `curl -s 'http://127.0.0.1:8000/api/composition/runs?subset=full'` → JSON array of **9** rows;
parsed with `_api.py` → `_api.out`. Row shape: `capabilities` has keys
`[aggregate_metrics, candidate_topk, image_availability, per_query_target_ranks]`; `measurement` has
keys `[coverage, denominator, image_availability_sample_size, sample_count, scored_count, subset,
unscored_by_kind, unscored_query_count, whole_run]`.

| model_id | run_id | execution_status | capabilities.candidate_topk | measurement.coverage | scoring_impl.impl | duplicate_status | measurement_identity_confidence |
|---|---|---|---|---:|---|---|---|
| `limn` | `10173dc45dea8c90fd2a284b` | `PARTIAL` | `false` | `0.980176578377478` | `matmul` | `undetermined` | `partial` |
| `conesep` | `30f6de423861cc1d6c78c361` | `COMPLETED` | `true` | `1.0` | `einsum-optimized` | `unique` | `full` |
| `ptha_mtst` | `315dd590eeb03d2c78d7937a` | `COMPLETED` | `false` | `1.0` | `einsum-optimized` | `undetermined` | `partial` |
| `intent` | `37ee72aa298b276d1bfda0fa` | `COMPLETED` | `true` | `1.0` | `einsum` | `unique` | `full` |
| `habit` | `497c25c57fbeba885ecfb9d1` | `COMPLETED` | `true` | `1.0` | `einsum` | `undetermined` | `partial` |
| `hint` | `4bc9893a977e501b82378e76` | `COMPLETED` | `true` | `1.0` | `einsum` | `unique` | `full` |
| `encoder` | `6a528e877c328772e2c7bb28` | `COMPLETED` | `false` | `1.0` | `matmul` | `undetermined` | `partial` |
| `tme` | `ad477312aa57c1b9114f9e51` | `COMPLETED` | `false` | `1.0` | `einsum` | `undetermined` | `partial` |
| `habit` | `e0dd04a46ef7a58334b9e353` | `FAILED` | `false` | `0.0` | `einsum` | `undetermined` | `partial` |

Findings a writer must reproduce:

- **`habit` appears twice** — `497c25c57fbeba885ecfb9d1` (COMPLETED) and `e0dd04a46ef7a58334b9e353`
  (FAILED). Any "one row per model" table must resolve this explicitly; the default `?subset=full` list
  gives 9 rows for 8 distinct models.
- **`released_audit` does NOT appear by default.** `?subset=full` returns 9 rows and none is
  `released_audit`. It appears only with `include_imported=true` → 10 rows, where it reports
  `execution_status COMPLETED`, `scoring_impl.impl = "unknown"`, `duplicate_status = "undetermined"`,
  `measurement_identity_confidence = "partial"`, `measurement.coverage = 1.0`,
  `capabilities.candidate_topk = true`.
- Calling the endpoint **without** `subset` returns **27** rows: 9 distinct `(model, run)` pairs × the
  3 subsets `full`, `sf`, `validated`, each repeated 3×. Never quote the 27 as runs.
- Synthetic fixtures are opt-in: `include_fixtures=true&include_smoke=true&include_imported=true` gives
  **14** rows (adds the two `SYNTHETIC_*` fixtures plus the two 8-query smoke run ids `3f86f58…` and
  `eda30a…`).
- Only 3 of 9 rows are `duplicate_status = "unique"` (conesep, intent, hint) — the other 6, including
  the released audit import, are `undetermined`.
- `scoring_impl.impl` takes 4 distinct values across the served set: `einsum`, `einsum-optimized`,
  `matmul`, and `unknown` (import only). Two models (`limn`, `encoder`) report `matmul` and are the two
  whose `candidate_topk` is `false` together with `ptha_mtst` and `tme`.

---

## 8. MÂU THUẪN GIỮA CÁC TÀI LIỆU

Where two documents (or two lines of one document) state different values for the same fact. Each entry:
both claims with line numbers, the judgement, and the evidence behind the judgement. Nothing outside this
directory was edited; the sources are documented, not repaired.

### C1. `FAILURES.md` attribution tally: table says (a)8 (h)0, prose says 6 + 2

| | claim |
|---|---|
| **A** | `workbench/docs/reviews/20261108T212500+0700-sweep/FAILURES.md:124-128` — `(a) 8`, `(h) 0`, `(b) 9`, `(c) 0`, `(d) 1`; `:131` — `8 + 0 + 9 + 0 + 1 = 18` |
| **B** | same file, `:193-196` — "the current 18 break down as **6 inside the range, 2 harness defects, 9 host state, 1 closed**" |

**Correct: A.** The table is the only tally carrying per-row evidence: all 18 rows at `:39-57` name a
category and a proof (the `git merge-base --is-ancestor aa0f530 b0f38bd` check for #1–#4 and #8, the
`9b6b5b6` ordering reproduction for #5, the four-reason trace for #6/#7, the clean-worktree reproduction
for #9/#18, `release_verify`'s `RELEASE_BLOCKED` for #11–#17). B's 6/2 split re-splits the same 8 rows
without naming which two moved, and is contradicted by `:126` ("reclassified to (a); see the trace") and
by `:133` which says the "6 caused / 2 harness" split is *withdrawn*. B is a leftover of a superseded
revision inside the same file. Arithmetic alone cannot discriminate: both sum to 18.

### C2. GPU-capability counts: 632 vs 648

| | claim |
|---|---|
| **A** | `FAILURES.md:49` — "Exact breakdown of all **632** … **465** override / **363** Synthetic GPU / **34** OOM / **34** both / **329** Syn+null / **50** no `device_name`" |
| **B** | this pass, 2026-11-09 — directory holds **648** JSONs: **477 / 372 / 35 / 35 / 337 / 51** |
| **corroboration** | `full_suite_c2d8cdeb.txt` traces end "and **621** more" (11+621 = 632); `full_suite_1129c58b.txt` traces end "and **637** more" (11+637 = 648) |

**Correct: neither is wrong — the directory moved.** 632 was true at the `c2d8cdeb` sweep
(2026-11-08 21:25 +0700); 648 was true one sweep later (22:53 +0700) and still is. The shared, gitignored
`workbench/artifacts/gpu_capabilities/` accumulates other agents' probe records, so any bare count is
stale on arrival. **Rule for the writer: always print the total with its timestamp**, e.g. "648 probe
records, 2026-11-09" — and cite 632 only as "632 at `c2d8cdeb`".

### C3. Checkpoint staging: 11/29 vs 21/29

| | claim |
|---|---|
| **A** | this sheet, first pass — "**11 of 29** staged" (the dropped `_staged.py`, a `<root>/<model_id>/<filename>` join) |
| **B** | `workbench/docs/reviews/20261108T152852-audit-inventory/REPOSITORY_INVENTORY.md:14` ("**21 / 29**", via `checkpoint_path()`) and `§7.1` heading at `:306` ("Staging: 21 of 29 resolve, and 8 do not"), table at `:113-145` |

**Correct: B (21/29). A was wrong, and the fault is mine, not the inventory's.** I re-implemented the join
instead of calling production. `workbench/backend/registry.py::checkpoint_path` tries
`configured / model_id` first and falls back to `configured / checkpoint_base_relative` when the
conventional directory is absent; 17 of 29 variants declare that key, and 10 of them resolve through it:
`airknow` ×2, `pair` ×2, `clvc_net`, `dcnet`, `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft`, `tgcir`,
`sprc_fiq_vitl`. Verified by importing the resolver itself
(`_staged_real.py`, `PYTHONPATH=/home/heheboiz/data/cir`): **staged=21, not_staged=8,
staged-by-conventional=11, staged-by-base_relative=10**. The 8 absent are exactly the inventory's 8:
`conesep/fiq_n05`, `conesep/fiq_n08`, `habit/fiq_n05`, `habit/fiq_n08`, `intent/fiq_n05`,
`intent/fiq_n08`, `sprc/sprc_fiq_unresolved`, `tme/cirr_0`. §6 above is rewritten with the corrected
figure and the per-variant table with the resolving rule.

Two secondary facts the corrected run exposes, both worth carrying into any future audit:
`sprc_fiq_vitl` resolves through `checkpoint_base_relative` to `../../checkpoints/sprc/…`, and the
sibling tree `…/20261106T002617+0700/checkpoints/` resolves only **6 of 29** if it is passed as the
configured root — so 21 is a property of `WORKBENCH_CHECKPOINT_ROOT`, not of the repo at large.

### C4. SYNTHETIC fixtures: 6003 queries vs an 8-query smoke limit

| | claim |
|---|---|
| **A** | `summary/SYNTHETIC_encoder_fashioniq_fixture001/full.json` and `…/SYNTHETIC_habit_fiq_n02_fixture002/full.json` — `"query_count": 6003`, `"common_cohort_count": 6003` |
| **B** | same two files — `"smoke": {"limited": true, "query_limit": 8, "label": "FIXTURE ONLY"}` |

**Correct: B is the operative limit; A is a copy of the real cohort's shape.** Both fixtures carry flat
`ndcg_*` / `recall_at_*_multimodal` scalars rather than the nested per-condition blocks a real run has,
which is consistent with hand-authored fixtures. Either way they cannot support a 6003-query claim and
must never be quoted beside a model row.

### C5. `d3_serial_blas_compare.txt`: 11 changed queries, 10 examples

| | claim |
|---|---|
| **A** | `20261108T-throughput/d3_serial_blas_compare.txt:3` — `queries_changed_rank = 11` |
| **B** | same file, lines 5-14 — exactly **10** `(query, rank_a, rank_b)` tuples before `score_max_abs_delta` |

**Correct: A (11 is the measurement; the list is truncated).** The tuples are illustrative, not
exhaustive. A report must say "11 queries changed rank" and never "11 examples listed".

### C6. `EVIDENCE_INDEX.md:89` lists prose-only figures in a "proves" cell

| | claim |
|---|---|
| **A** | `20261108T173736+0700/EVIDENCE_INDEX.md:89`, `THROUGHPUT.vi.md` row, **proves** column — "decode A/B **21.2 → 2089.2 img/s**; query-axis 1/2/4/8 → 4.8/8.7/15.7/27.4 q/s, **5.67×** at 8 threads" |
| **B** | the table's own header at `:87` — `| file | proves | does not prove |`, and `vdecode.txt`'s row at `:96`, which puts "the 21.2 img/s baseline" in its **does-not-prove** column |

**Correct: B, and A is the defect.** A "proves" column may only list what the file evidences; the same
page's `:96` already says `21.2` is not evidenced by any raw file. `21.2`, `2089.2`, `2192`, the `5.67×`
multiplier and the `106.0/58.7/32.6/18.7 s` table each return **zero hits** across the 34 raw files in
`20261108T-throughput/` (§4g). **Fixed 2026-11-09**: `:89`'s proves cell now reads "the write-up of this
round's throughput work, with its own traceability caveats in §8.8", and the five figures moved to the
does-not-prove cell.

An earlier version of this entry accused `:96` of implying `vdecode.txt` evidences `21.2`. That was
**wrong** — `:96` says the opposite, in the correct column. The error was mine; the real defect is one
row up at `:89`.

### C7. `FINAL_PC_RELEASE_VERDICT.md` collected count vs its own sweep log

| | claim |
|---|---|
| **A** | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:22` — "Collection at this revision: **2532 tests** in `workbench/tests`", then "**2467 passed, 18 failed**, 47 skipped = 2532 collected" |
| **B** | `20261108T225338+0700-sweep/full_suite_1129c58b.txt:14` — "collected **2532** (= 2467 + 18 + 47; `--collect-only` reports 2532)" |

**Agree.** Recorded here because it is the pair most likely to be re-derived wrongly: `2467+18+47 = 2532`
holds, but the number is not pytest-emitted anywhere in the log (the log's own pytest line is
`18 failed, 2467 passed, 47 skipped, 290 warnings in 3483.79s (0:58:03)`). Cite the header table, not a
`collected N items` line that does not exist.

### C8. LIMN R@10 is quoted without its 5884/6003 denominator

| | claim |
|---|---|
| **A** | LIMN row in any table that lists `30.06` beside the other models' R@10 |
| **B** | `summary/limn_base_iter0_all_categories_10173dc45dea8c90fd2a284b/full.json` — `coverage 0.980176578377478`, `scored_count 5884` of `query_count 6003`, `execution_status PARTIAL` |

**Correct: B governs.** `EVIDENCE.md:§6` (`20261108T180000+0700-limn-scope/`) already says the 30.06
"remains the old 5884-denominator number". Any table mixing it with the 6003-coverage rows is comparing
different denominators. This sheet's §2b carries the coverage column for exactly this reason.

### C9. `hint` and `encoder` both declare `filename: fashioniq.pt`

| | claim |
|---|---|
| **A** | `models.yaml` — `hint.checkpoint_variants[0].filename == "fashioniq.pt"` **and** `encoder.checkpoint_variants[0].filename == "fashioniq.pt"` |
| **B** | their byte sizes differ: `hint` `expected_size_bytes 2724451182`; `encoder` `expected_size_bytes 631879365` |

**Not a contradiction, a trap.** Both are correct; the filenames are genuinely identical and only
`model_id/filename` or the resolver's `checkpoint_path()` separates them. A basename match mismaps
`encoder`'s 631879365-byte expectation onto `hint`'s 2.7 GB file — which is what an early pass of
the dropped `_staged.py` did before being corrected.

### C10. 2528 vs 2532 collected — the two sweeps differ only on revision, not on ids

| | claim |
|---|---|
| **A** | `full_suite_c2d8cdeb.txt:14` — collected 2528; `:13` — 2463 passed / 18 failed / 47 skipped |
| **B** | `full_suite_1129c58b.txt:14` — collected 2532; `:13` — 2467 passed / 18 failed / 47 skipped |

**Agree, and the delta is accounted for.** `2532 − 2528 = 4`, exactly the 4 ids
`test_frontend_scoring_impl.py` contributes (`1129c58b.txt:17`), and the 18 failing ids are byte-identical
between the two logs (verified by comparing the extracted `FAILED` lists: 18 vs 18, `identical: True`).
Recorded so no writer "resolves" a 4-test discrepancy that does not exist.

### Summary of contradictions

| # | fact | correct answer | severity for a report |
|---|---|---|---|
| C1 | 18-failure attribution split | table (a)8 (h)0 (b)9 (c)0 (d)1 | high — a release claim rests on it |
| C2 | GPU probe-record count | 648 at 2026-11-09; 632 at `c2d8cdeb` — always timestamp | medium — moving target |
| C3 | checkpoint staging | **21 of 29**, not 11 | high — 10 variants miscounted |
| C4 | fixture query count | 8 (smoke limit), not 6003 | medium — quote-blocked |
| C5 | changed-rank example count | 11 changed, 10 shown | low |
| C6 | `EVIDENCE_INDEX.md:89` proves-cell | prose-only figures moved to does-not-prove | high — unsourced number |
| C7 | collected count provenance | header table, not pytest output | low |
| C8 | LIMN R@10 denominator | 5884 of 6003, coverage 0.980176… | high — comparability |
| C9 | duplicate checkpoint filename | both real; match on `model_id/filename` | medium — silent mismap |
| C10 | 2528 vs 2532 collected | both right; +4 known ids, same 18 failures | low |

**10 contradictions recorded (C6 corrected 2026-11-09: the accused line was `:89`, not `:96`).**

---

## TÓM TẮT 10 CON SỐ CHẮC CHẮN NHẤT

| # | figure | value | file |
|---|---|---|---|
| 1 | Production tree equals the recorded tested revision | `git diff --quiet 1129c58ba8fa0566c5840cfe820292ae4e9e84f3` → **exit 0** | `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md:22` (SHA) + command exit code |
| 2 | Serial full-suite result at `1129c58` | `18 failed, 2467 passed, 47 skipped = 2532` collected, `3483.79 s`, exit 1 | `workbench/docs/reviews/20261108T225338+0700-sweep/full_suite_1129c58b.txt:13-14,537-540` |
| 3 | Same 18 failing ids in both sweeps | 18 = 18, lists identical | `…/20261108T212500+0700-sweep/full_suite_c2d8cdeb.txt` and `…/20261108T225338+0700-sweep/full_suite_1129c58b.txt` |
| 4 | Attribution tally at `c2d8cdeb` | (a) 8 · (h) 0 · (b) 9 · (c) 0 · (d) 1 = 18 | `…/20261108T212500+0700-sweep/FAILURES.md:120-131` |
| 5 | Best proven decode+transform throughput | `512 images / 0.75 s = 682.6667 img/s` at 4 workers (serial baseline 109.1 img/s) | `workbench/docs/reviews/20261108T-throughput/vdecode.txt` |
| 6 | Full-pipeline batch throughput | `17.6–18.0 img/s` fp32 over a 2000-image slice, threads=20 | `workbench/docs/reviews/20261108T-throughput/batch.txt` |
| 7 | Shipped-path rank equivalence | `PASS`, `18009` = `6003 × 3` query-condition pairs, 0 mismatches, 8 threads / chunk 32 | `workbench/docs/reviews/20261108T-throughput/d2_gate_full_6003x3.txt` + `.json` |
| 8 | LIMN scope resolution | `dress 2010 / shirt 2035 / toptee 1956 / None 2` = 6003; `changed_loader_assigned = []` | `workbench/docs/reviews/20261108T180000+0700-limn-scope/scope_resolution.json` |
| 9 | LIMN identity moves | old `10173dc45dea8c90fd2a284b` → new `2a4b19f335f5ec281375d09f`; `resume_would_reuse_old = false` | `workbench/docs/reviews/20261108T180000+0700-limn-scope/identity_proof.json` |
| 10 | Registry shape and staging | 17 models, 16 distinct repos, 29 checkpoint variants, **21 staged** (11 conventional + 10 `checkpoint_base_relative`, 8 absent) | `workbench/registry/models.yaml` + `registry.checkpoint_path` (`_staged_real.out`) |

Safe model-metric anchors for a report (all full-6003, coverage 1.0, from §2b):
conesep MM R@10 `35.48225887056472`; habit `35.61552557054806`; intent `34.39946693320007`; hint
`32.816924870897886`; encoder `31.01782442112277`; tme `34.01632517074796`; ptha_mtst
`19.473596535065802`. LIMN's `30.064581917063222` must be quoted with `coverage 0.980176578377478` /
`5884 of 6003` attached, or not at all.
