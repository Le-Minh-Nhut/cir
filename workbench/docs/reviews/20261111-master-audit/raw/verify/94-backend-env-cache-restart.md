# 94 — backend `workbench.env` cache vs. restart: claim NOT reproduced

Date: 2026-11-12
Repo: `/home/heheboiz/data/cir` (branch `research/cir-failure-workbench`)
Unit: `cir-workbench-backend.service` (uvicorn, 127.0.0.1:8000)

## Verdict

The **mechanism** in the claim is real, but the **consequence** did not occur. `file_settings()`
is `@lru_cache(maxsize=1)` and does cache the operator's `workbench.env` for the life of the
process. However the live backend reported both interpreters as **resolved**, exactly matching a
fresh process, so there was nothing to fix. **No restart was performed** — restarting on a false
premise would have been an unnecessary disruption to the live unit, and the instructions made the
restart conditional on the claim being proven.

Why the stale read did not happen: the `workbench.env` edit landed at `00:11:54` and the process
started at `23:48:58`, but `file_settings()` is populated **lazily on first use**, not at import.
`_startup()` only runs `_recover_interrupted_jobs()` and `_ensure_index()`; neither touches
`model_settings`. The unit journal shows the process's **first** `/api/models` request at
`02:43:37` — over two hours *after* the file was edited — so the first (and only) file read already
saw both new lines. The cache was cold across the edit window, which is why the staleness never
materialised.

## (a) Unit identity

```
MainPID=1221379
ActiveEnterTimestamp=Wed 2026-11-11 23:48:58 +07
```

`ps -o pid,lstart -p 1221379` independently agrees: `Wed Nov 11 23:48:58 2026`.
(A superseded PID 1221147 appears earlier in the journal and is gone; 1221379 is the live one.)

## (b) `workbench.env` modification time

```
2026-11-12 00:11:54.292247817 +0700 workbench/config/workbench.env
```

`tail -3`:

```
WORKBENCH_RUNTIME_ADAPTATION=sm_120
WORKBENCH_CLVC_PYTHON=/home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python
WORKBENCH_TGCIR_PYTHON=/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python
```

So the edit is indeed **after** the process start (00:11:54 > 23:48:58), as the claim states.

## (c) The LIVE backend's answer (`GET /api/models`, PID 1221379)

Walked directly with `json.load` + dict lookup (no `?q=` selectors):

```
=== clvc_net model_id=clvc_net
  cp name=None id='author_drive_unresolved' runnable=False
     blocking_reasons=['torchvision ResNet-50 ImageNet weights missing: /home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth', 'external runtime asset file missing: torchvision ResNet-50 ImageNet weights (/home/heheboiz/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth)', 'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT']
=== tgcir model_id=tgcir
  cp name=None id='tgcir_archive_unresolved' runnable=False
     blocking_reasons=['external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache', 'command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT']
```

**Neither list contains an interpreter/`execution environment` blocker.** A stale cache would have
produced `execution environment not verified: WORKBENCH_CLVC_PYTHON=None [UNCONFIGURED/UNCONFIGURED]
(WORKBENCH_CLVC_PYTHON not set)` — the shape that *does* appear in the same payload for other
models whose variables genuinely are unset (`airknow`, `pair`, `dcnet`, `combiner_rn50x4_noft`,
`clip4cir_rn50x4_fullft`, `sprc` all report exactly that form). The absence for clvc_net/tgcir is
therefore a positive signal, not a missing field.

`POST /api/jobs` for both was also attempted; both returned **409 `evaluation_blocked`** with the
same blocker lists and **no** interpreter reason:

```
clvc_net -> blockers: [torchvision ResNet-50 ImageNet weights missing..., external runtime asset
            file missing..., command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT,
            protocol incompatible: fashioniq]
tgcir    -> blockers: [external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16
            weights/cache, command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT,
            protocol incompatible: fashioniq]
```

They are refused — but for real asset/entrypoint reasons, **not** because the interpreter was unset.

## (d) A FRESH process (`workbench/envs/ctl312h/bin/python`, `PYTHONPATH=.`)

```
setting WORKBENCH_CLVC_PYTHON = /home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python
setting WORKBENCH_TGCIR_PYTHON = /home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python
resolve_model_interpreter(clvc_net) = /home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python
resolve_model_interpreter(tgcir) = /home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python
```

Fresh-process preflight agrees with the live API, blocker for blocker:

```
clvc_net report: tier=GPU_RUNTIME_READY status=GPU_RUNTIME_READY interp=.../clvc39-la/bin/python reason=None
clvc_net preflight_blockers (3):
   - torchvision ResNet-50 ImageNet weights missing: ...
   - external runtime asset file missing: torchvision ResNet-50 ImageNet weights (...)
   - command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
tgcir report: tier=GPU_RUNTIME_READY status=GPU_RUNTIME_READY interp=.../tgcir39-la/bin/python reason=None
tgcir preflight_blockers (2):
   - external runtime asset provenance unresolved: OpenAI CLIP ViT-B/16 weights/cache
   - command not runnable: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
```

### (c) vs (d) comparison

| question | live (c) | fresh (d) | agree? |
|---|---|---|---|
| `WORKBENCH_CLVC_PYTHON` | resolved (no blocker) | `.../clvc39-la/bin/python` | **yes** |
| `WORKBENCH_TGCIR_PYTHON` | resolved (no blocker) | `.../tgcir39-la/bin/python` | **yes** |
| clvc_net blockers | 3, asset+entrypoint | 3, asset+entrypoint | **yes** |
| tgcir blockers | 2, asset+entrypoint | 2, asset+entrypoint | **yes** |

The claim's test — "(c) shows the interpreters unresolved while (d) resolves them" — **fails**.
(c) does not show them unresolved.

## Counterfactual (the mechanism is real)

To show the claim was not dismissed on a technicality, the stale state was reproduced in a
throwaway process by masking the two keys out of `file_settings()`:

```
--- clvc_net (STALE simulation)
    execution environment not verified: WORKBENCH_CLVC_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_CLVC_PYTHON not set)
--- tgcir (STALE simulation)
    execution environment not verified: WORKBENCH_TGCIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED] (WORKBENCH_TGCIR_PYTHON not set)
```

That is what a genuinely stale backend *would* have served. The live backend did not serve it, so
its cache was populated after the edit — consistent with the 02:43:37 first-use timestamp.

## Step 2 — restart

**Not performed.** Step 2 was explicitly conditional ("if proven"). The claim was disproven, so no
restart occurred and there are no recovery numbers to report. MainPID is unchanged at **1221379**.

## Step 3 — live stack health (unchanged, no restart involved)

```
200  http://127.0.0.1:8000/api/benchmark-runs
200  http://127.0.0.1:8000/api/composition/runs/10173dc45dea8c90fd2a284b?subset=full
200  http://127.0.0.1:5173/
200  http://127.0.0.1:5173/api/benchmark-runs   (proxied through the frontend)
```

Served bundle name, unchanged as expected:

```
assets/index-DYYSN7oG.js
```

## Why an edit to `workbench.env` still requires a restart

An edit to `workbench.env` requires a backend restart **because that read is cached per process**:
`model_settings.file_settings()` is decorated `@lru_cache(maxsize=1)` and returns a plain `dict`
built once from `operator_config.read_env_file()`, so the first lookup freezes the file's contents
for the lifetime of the process and every later `setting()` call replays that snapshot — a running
backend will not observe an appended `WORKBENCH_<MODEL>_PYTHON` line until it is restarted. This
case happened to escape the consequence only because the cache is filled lazily on first use and no
`/api/models` request occurred between the 00:11:54 edit and the 02:43:37 first call; that is luck,
not a guarantee. By contrast, **registry edits are picked up without a restart**, because
`registry._load_yaml_cached` keys its parse cache on `(resolved path, st_size, st_mtime_ns)` and
re-parses whenever the file's size or mtime changes, and `_validated_cache_key` does the same for
the validated-config cache. So the operational rule is: `models.yaml` — edit and continue;
`workbench.env` — edit and restart the backend.
