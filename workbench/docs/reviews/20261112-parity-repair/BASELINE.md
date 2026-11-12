# BASELINE — parity/eval repair campaign

Collected 2026-11-12 (host clock `Thu Nov 12 19:25:32 2026 +07` at GPU query time).
Branch `research/cir-failure-workbench`, worktree `/home/heheboiz/data/cir`.

This is a **measurement record only**. No tracked file was modified, nothing was staged or
committed, no GPU job was started, no test was run, no service was restarted, and
`/tmp/cir_gpu.lock` was not deleted. Every row names the exact command that produced it.
Raw outputs: `raw/baseline/<name>.txt` (all under 200 lines except where noted).

---

## 1. Repo state

| Fact | Value | Command |
|---|---|---|
| HEAD | `40c897c31bbdd76b5a76f65935c7dd0f7fef0e94` | `git rev-parse HEAD` |
| `origin/research/cir-failure-workbench` | `40c897c31bbdd76b5a76f65935c7dd0f7fef0e94` (equal to HEAD) | `git rev-parse origin/research/cir-failure-workbench` |
| ahead / behind vs origin | `0	0` | `git rev-list --left-right --count origin/... ...HEAD` |
| `git status --porcelain` (start of collection) | `?? CLEANUP_REPORT.md.save` | `git status --porcelain` |
| HEAD commit | `40c897c31bbdd76b5a76f65935c7dd0f7fef0e94 2026-11-12 13:38:09 +0700 docs(report): re-pin to aea143bd after the second stationary sweep` | `git log -1 --format='%H %ci %s'` |
| Commits since `82cb1044` | `44` (`82cb1044` is an ancestor: `merge-base --is-ancestor` → 0) | `git rev-list --count 82cb1044..HEAD` |
| `origin/main` | `8ebcc74d5fd1934df4f3772a8ca5a96dbd00f333` | `git rev-parse origin/main` |
| Working-tree production bytes vs `aea143bd` | **identical** (`prod_tree_equals_aea143bd=0`, i.e. `git diff --quiet` exit 0) | `git diff --quiet aea143bd..HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'; echo "prod_tree_equals_aea143bd=$?"` |
| Working-tree production bytes vs `96d86010` | **different** (`prod_tree_equals_96d86010=1`) | same form with `96d86010` |
| Diff `96d86010..HEAD` (excl. docs/tests) | 30 files changed, `+3622 −265` (11 dirs; `composition.py`, `main.py`, `storage.py`, `model_clvc_net.py`, `model_tgcir.py`, `runner.py`, registry YAMLs, `backfill_recall_at_1.py`, `derive_routing_cohort_gap.py`, `render_units.py`, …) | `git diff --stat 96d86010..HEAD -- . ':(exclude)workbench/docs' ':(exclude)workbench/tests'` |

`git log --oneline -15` (raw: `04-log-oneline-15.txt`):

```
40c897c docs(report): re-pin to aea143bd after the second stationary sweep
cd9ced2 docs(release): state this sweep's own duration in the aea143bd row
c569427 release: advance TESTED_SOURCE_SHA to aea143bd after the final stationary sweep
aea143b fix(web): browse the conditions the RUN measured, not the protocol constant
5b5a24a docs(report): the porcelain block is a snapshot, and say so
a425739 docs(report): say why '20 failed' and 'guard 8 passed' are both true
b5c9965 docs(report): the §0 porcelain block was an unlabelled snapshot
c64b173 docs(report): re-pin the live-state rows after the pin advance and the push
a0c08c0 fix(release): stop the verdict installing the superseded revision; guard every release doc
1a71790 docs(release): point the 117fc172 citations at its own sweep record
23d847e evidence: commit the post-cutover browser smoke, the tgcir API surface, and the E2E screenshots
c998504 release: advance TESTED_SOURCE_SHA to 117fc172 after the stationary sweep
622be09 docs(verify): correct the encode denominator to image encodes (scopes x gallery + query rows); retract the old CLVC figure
117fc17 docs(report): the caption-free gap is measured — 0.33400015944039774 over 5884
c9ac37a docs(report): pin the reference_metadata_seen grep to its real scope
```

### `TESTED_SOURCE_SHA` pins

| Document | Line | Value | Command |
|---|---|---|---|
| `workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md` | 22 | `aea143bd8894881ad7d5b341c21dbeff25cfaaaf` | `grep -n 'TESTED_SOURCE_SHA' workbench/docs/reviews/FINAL_PC_RELEASE_VERDICT.md` |
| `workbench/docs/PC_PILOT_HANDOFF.md` | 31 | `aea143bd8894881ad7d5b341c21dbeff25cfaaaf` (checkout instruction at line 63) | `grep -n 'TESTED_SOURCE_SHA' workbench/docs/PC_PILOT_HANDOFF.md`; `grep -n 'aea143bd' workbench/docs/PC_PILOT_HANDOFF.md` |

Both documents name the same pin, and the pin is 4 commits behind HEAD
(`aea143b` → `c569427` → `cd9ced2` → `40c897c`), all four documentation-only.

---

## 2. Runtimes and hardware

| Fact | Value | Command |
|---|---|---|
| Controller interpreter | `Python 3.12.11` | `workbench/envs/ctl312h/bin/python -V` |
| Controller `sys.executable` | `/home/heheboiz/data/cir/workbench/envs/ctl312h/bin/python` | `workbench/envs/ctl312h/bin/python -c 'import sys;print(sys.executable)'` |
| `nproc` | `20` | `nproc` |
| CPU affinity | `Cpus_allowed_list: 0-19` | `grep Cpus_allowed_list /proc/self/status` |
| GPU | `NVIDIA GeForce RTX 5070 Ti, 16303 MiB total, 991 MiB used` | `nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv` |
| GPU free memory | **15312 MiB** (16303 − 991) | derived from the row above |
| GPU compute processes | none (`pid, process_name, used_gpu_memory [MiB]` header only) | `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv` |
| Driver / CUDA | `NVIDIA-SMI 580.173.02  Driver Version: 580.173.02  CUDA Version: 13.0` | `nvidia-smi` (header) |
| `composition_eval.py` / `pytest` running | none (`pgrep` empty) | `pgrep -af 'composition_eval.py|pytest'` |
| `/tmp/cir_gpu.lock` | `-rw-rw-r-- 1 heheboiz heheboiz 0 Nov 12 08:40` (0 bytes, present, **not** touched) | `ls -l /tmp/cir_gpu.lock` |
| Free space `/` and `/tmp` | `630G` size, `541G` used, **`57G` avail**, 91% (same filesystem for both) | `df -h / /tmp` |

Disk is **57 GiB available — above the 45 GiB blocker threshold**, so no blocker line applies.

### Environments under `workbench/envs/` (19 dirs)

| Env | `bin/python` | Version | Resolved binary | Command |
|---|---|---|---|---|
| `airknow-adapted` | **absent** | — | — | loop: `"$d/bin/python" -V` per dir |
| `airknow310-la` | yes | 3.10.20 | `~/.local/share/uv/python/cpython-3.10.20-linux-x86_64-gnu/bin/python3.10` | same |
| `clip4cir-adapted` | **absent** | — | — | same |
| `clvc39-la` | yes | 3.9.25 | `~/.local/share/uv/python/cpython-3.9.25-.../python3.9` | same |
| `ctl312` | yes | 3.12.3 | `/usr/bin/python3.12` | same |
| `ctl312h` | yes | 3.12.11 | `~/.local/share/uv/python/cpython-3.12.11-.../python3.12` | same |
| `dcnet-prepare` | yes | 3.7.12 | `workbench/envs/dcnet-prepare/bin/python3.7` | same |
| `dl-adapted` | **absent** | — | — | same |
| `encoder310-adapted` | yes | 3.10.20 | `~/.local/share/uv/python/cpython-3.10.20-.../python3.10` | same |
| `ilearn39-adapted` | yes | 3.9.25 | `~/.local/share/uv/python/cpython-3.9.25-.../python3.9` | same |
| `limn-adapted` | **absent** | — | — | same |
| `limn310-adapted` | yes | 3.10.20 | `~/.local/share/uv/python/cpython-3.10.20-.../python3.10` | same |
| `pair-adapted` | **absent** | — | — | same |
| `pair39-la` | yes | 3.9.25 | `~/.local/share/uv/python/cpython-3.9.25-.../python3.9` | same |
| `sprc-adapted` | **absent** | — | — | same |
| `test-adapted` | **absent** | — | — | same |
| `tgcir39-la` | yes | 3.9.25 | `~/.local/share/uv/python/cpython-3.9.25-.../python3.9` | same |
| `tme-adapted` | yes | 3.10.22 | `workbench/envs/tme-adapted/bin/python3.10` | same |

7 of 19 env dirs have no `bin/python`: `airknow-adapted`, `clip4cir-adapted`, `dl-adapted`,
`limn-adapted`, `pair-adapted`, `sprc-adapted`, `test-adapted`.

---

## 3. Services and live API

| Fact | Value | Command |
|---|---|---|
| backend unit | `MainPID=1758096`, `ActiveState=active`, `SubState=running`, `NRestarts=0`, `ActiveEnterTimestamp=Thu 2026-11-12 12:20:57 +07` | `systemctl --user show -p MainPID -p ActiveEnterTimestamp -p NRestarts -p ActiveState -p SubState cir-workbench-backend.service` |
| frontend unit | `MainPID=1221148`, `ActiveState=active`, `SubState=running`, `NRestarts=0`, `ActiveEnterTimestamp=Wed 2026-11-11 23:48:44 +07` | same for `cir-workbench-frontend.service` |
| backend process | `uvicorn workbench.backend.main:app --host 127.0.0.1 --port 8000`, started `Thu Nov 12 12:20:56 2026`, elapsed `07:07:35` | `ps -o pid,lstart,etime,cmd -p 1758096,1221148` |
| frontend process | `npm run preview --host 127.0.0.1 --port 5173 --strictPort`, started `Wed Nov 11 23:48:44 2026` | same |

| Endpoint | HTTP code / bytes | Command |
|---|---|---|
| `http://127.0.0.1:8000/api/models` | `200 86658` | `curl -s -o /dev/null -w '%{http_code} %{size_download}\n' <url>` |
| `http://127.0.0.1:8000/api/benchmark-runs` | `200 54761` | same |
| `http://127.0.0.1:8000/api/runs/parity-gate` | `200 65460` | same |
| `http://127.0.0.1:5173/` | `200 271` | same |

### Frontend bundle identity

| Fact | Value | Command |
|---|---|---|
| Bundle referenced by served index | `assets/index-NnjrdZLG.js` | `curl -s http://127.0.0.1:5173/ \| grep -o 'assets/index-[^"]*\.js'` |
| Served HTML | `<script type="module" crossorigin src="/assets/index-NnjrdZLG.js"></script>` + `<link rel="stylesheet" crossorigin href="/assets/index-DecQnta9.css">` | `curl -s http://127.0.0.1:5173/` |
| `dist/assets/index-NnjrdZLG.js` | `392769` bytes, sha256 `721d34d043858e80c88e76d72866ba8583508318baeebac599cebd1e1c2c80d0` | `sha256sum` + `stat -c '%n %s'` |
| `dist/assets/index-DecQnta9.css` | `26552` bytes, sha256 `e7e826a332ec9ab171f97051b901ee86fa69e2588937c3023914f3ac41ffe196` | same |
| Served JS bytes == `dist/` bytes | yes, sha256 `721d34d0…2c80d0` both | `curl -s …/assets/index-NnjrdZLG.js \| sha256sum` |
| Served CSS bytes == `dist/` bytes | yes, sha256 `e7e826a3…ffe196` both | `curl -s …/assets/index-DecQnta9.css \| sha256sum` |
| `dist/` tracked? | **no** — ignored by `.gitignore:95:workbench/frontend/dist/` | `git check-ignore -v workbench/frontend/dist/assets/index-NnjrdZLG.js` |
| `dist/` mtime | `2026-11-12 12:20:56 +0700` (same second as backend start) | `stat -c '%n %y' workbench/frontend/dist/assets/*` |

### Live gate/run payloads (read-only)

| Fact | Value | Command |
|---|---|---|
| Parity gate `passed_models` | `[]` (empty) | `curl -s …/api/runs/parity-gate` → `json.load(...)['passed_models']` |
| Parity gate models | `encoder` → `PARITY_MANDATORY_SUBCHECK_FAILED, passed=False`; `limn` → `PARITY_MANDATORY_SUBCHECK_FAILED, passed=False`; standard `parity-relative/v1` | same |
| Benchmark runs | 13 rows: 2 × `csmcir` (`PARITY_VERIFIED`, `CAPABILITY_ABSENT`), 1 × `combiner_rn50x4_noft` (`PARITY_NOT_EVALUATED`), 10 × `synthetic_model` (`PARSER_UNAVAILABLE`); every row `is_synthetic=False` | `curl -s …/api/benchmark-runs` |

---

## 4. Configuration registry inventory

**Registry located at `workbench/registry/models.yaml`** (1777 lines, `schema_version: 1`), found by
grepping `tgcir`/`clvc_net` + `checkpoint`; loaded with
`workbench/envs/ctl312h/bin/python -c "import yaml; yaml.safe_load(...)"`. There is no
`workbench/config/paper_benchmarks` and no `workbench/models` directory; `workbench/config/` holds only
`workbench.env` and `workbench.env.example`. Sibling registries: `workbench/registry/auxiliary_assets.yaml`
(21 assets), `workbench/registry/preparation_contracts.yaml`, `workbench/registry/paper_benchmarks/papers.yaml`.

**Total registered configurations: 17** — reconciled with the prompt's claim of 17: **match**.
17 `model_id` entries, 17 `source_available: true`, 16 distinct `upstream_repo_url` values
(`combiner_rn50x4_noft` and `clip4cir_rn50x4_fullft` share `ABaldrati/CLIP4Cir`), 29 `checkpoint_variants`
in total. Protocols declared by the registry: `fashioniq_original_split`,
`fashioniq_full_gallery_ref_excluded`, `fashioniq_val_split`.

Interpreter path is the value resolved by the workbench's own resolver
(`workbench.backend.runtime.resolve_model_interpreter`, which reads the declared `python_env_var` through
the operator settings, i.e. `workbench.env`) — **not guessed**. Resolved by:
`python -c "from workbench.backend.registry import load_registry; from workbench.backend.runtime import resolve_model_interpreter; …"`.

| # | model_id | Display name | Interpreter key | Resolved interpreter | Checkpoint(s) (resolved path) | Size / sha256 | Aux assets (status) | Data root / layout | Protocol(s) | Native eval | Transfer | Aggregate-only | Composition |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | `csmcir` | CSMCIR | `WORKBENCH_CSMCIR_PYTHON` | `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | `…/cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt` | 2 720 615 859 B — **not hashed (size)**; declared `c8812071…3fa49e` | Qwen captions ×3 `VERIFIED`; COT_ours2 ×3 `VERIFIED`; author `.pyc` `VERIFIED`; LAVIS BLIP-2 `LOCALLY_RECORDED_ASSET`; EVA-ViT-g `LOCALLY_RECORDED_ASSET`; bert tokenizer `VERIFIED` | `data/FashionIQ`, `fashioniq_standard` | `fashioniq_original_split` | yes (replay via adapter; 2 runs `PARITY_VERIFIED`) | no | no (has `checkpoint_variants`) | yes (`model_csmcir`) |
| 2 | `airknow` | Air-Know | `WORKBENCH_AIRKNOW_PYTHON` | **unset → `null`** | `…/checkpoints/airknow/airknow_fiq_0.5.pt`, `…_0.8.pt` | 2 724 056 638 B each — not hashed (size); declared `f4d25ed9…`, `756fc0bd…` | LAVIS BLIP-2 `LOCALLY_RECORDED_ASSET`; EVA-ViT-g `LOCALLY_RECORDED_ASSET` | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_full_gallery_ref_excluded` | adapter present, not run | no | no | yes (`model_airknow`) |
| 3 | `conesep` | ConeSep | `WORKBENCH_CONESEP_PYTHON` | `workbench/envs/ilearn39-adapted/bin/python` | `…/conesep/ConeSep-FIQ-N0.2.pt` present; `N0.5`, `N0.8` **absent** | 2 721 428 966 B — not hashed (size); declared `5a864ee5…` | none | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_full_gallery_ref_excluded` | adapter present, not run | no | no | yes (`model_conesep`) |
| 4 | `ptha_mtst` | PTHA + MTST | `WORKBENCH_PTHA_MTST_PYTHON` | `workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | `…/ptha_mtst/tuned_mtst_cirr.pt` | 2 720 352 973 B — not hashed (size); declared `1c4dab86…` | EVA-ViT-g `LOCALLY_RECORDED_ASSET`; bert tokenizer `VERIFIED` | `data/FashionIQ`, `fashioniq_standard` | `fashioniq_original_split` | adapter present, not run | **yes** (`transfer_condition: cross_dataset_transfer`, `SELECTED_TRANSFER_ONLY`) | no | yes (`model_ptha_mtst`) |
| 5 | `habit` | HABIT | `WORKBENCH_HABIT_PYTHON` | `workbench/envs/ilearn39-adapted/bin/python` | `…/habit/HABIT-FIQ_N0.2.pt` present; `N0.5`, `N0.8` **absent** | 2 730 715 930 B — not hashed (size); declared `814d2d0c…` | none | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_full_gallery_ref_excluded` | adapter present, not run | no | no | yes (`model_habit`) |
| 6 | `intent` | INTENT | `WORKBENCH_INTENT_PYTHON` | `workbench/envs/ilearn39-adapted/bin/python` | `…/intent/intent_fiq_0.2.pt` present; `0.5`, `0.8` **absent** | 2 722 478 522 B — not hashed (size); declared `2a504cad…` | none | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_full_gallery_ref_excluded` | adapter present, not run | no | no | yes (`model_intent`) |
| 7 | `hint` | HINT | `WORKBENCH_HINT_PYTHON` | `workbench/envs/ilearn39-adapted/bin/python` | `…/hint/fashioniq.pt` | 2 724 451 182 B — not hashed (size); declared `bf9c72d4…` | none | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_val_split` | adapter present, not run | no | no | yes (`model_hint`) |
| 8 | `encoder` | ENCODER | `WORKBENCH_ENCODER_PYTHON` | `workbench/envs/encoder310-adapted/bin/python` | `…/encoder/fashioniq.pt` | 631 879 365 B, sha256 `0f31c50150a7bda3495e3378df13b4cec7924b52efd23b5e5dd9731459da9173` (= declared) | LAION CLIP ViT-B/32 `VERIFIED` (605 219 813 B) | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_val_split` | **yes** — `per_query_observer: ENCODER_QUERY_OBSERVER_V1`, `replay/encoder*.py` | no | no | yes (`model_encoder`) |
| 9 | `pair` | PAIR | `WORKBENCH_PAIR_PYTHON` | **unset → `null`** | `…/checkpoints/pair/pair-B1.pt`, `pair-B2.pt` | 694 958 565 B sha256 `c4d861ec…25fbf2`; 694 872 371 B sha256 `74346fd7…b3f03` | none | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_val_split` | adapter present, **mapping UNVERIFIED** | no | no | yes (`model_pair`) |
| 10 | `clvc_net` | CLVC-Net | `WORKBENCH_CLVC_PYTHON` | `workbench/envs/clvc39-la/bin/python` | `…/checkpoints/clvc/` (bundle dir, 6 members) | dir 1 185 968 194 B / 6 files — not hashed (size); member digests declared (`33d44186…`, `be41c2ba…`, `2ba8e320…`, `78b1d660…`, `d1fa9170…`, `c78a58c4…`) | torchvision ResNet-50 `VERIFIED` (`~/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth`, 102 502 400 B) | `data/FashionIQ`, `fashioniq_clvc_resized` | `fashioniq_val_split` | adapter present; `replay_status: BLOCKED_NO_SOURCE_REPLAY`, `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT` | no | no | yes (`model_clvc_net`) |
| 11 | `dcnet` | DCNet | `WORKBENCH_DCNET_PYTHON` | **unset → `null`** | `…/checkpoints/dcnet/` (run dir, 2 required files) | dir 2 882 704 059 B / 5 files — not hashed (size) | ResNet-50 `VERIFIED` (same file as row 10) | `data/FashionIQ`, `fashioniq_dcnet` | `fashioniq_full_gallery_ref_excluded` | adapter present, `REPLAY_CODE_IMPLEMENTED`, `COMMAND_AUDITED` | no | no | yes (`model_dcnet`) |
| 12 | `combiner_rn50x4_noft` | Combiner RN50x4 noft | `WORKBENCH_CLIP4CIR_PYTHON` | **unset → `null`** | `…/checkpoints/clip4cir/FashionIQ/RN50x4_noft/fiq_comb_RN50x4_noft.pt` | 236 017 921 B, sha256 `b31882fe6f7e3e194cb9cfda048926844ee0ccc9d4ebc5483fb3bacfa5972955` | CLIP RN50x4 `VERIFIED` (`~/.cache/clip/RN50x4.pt`, 421 854 225 B) | `data/FashionIQ`, `fashioniq_standard` | `fashioniq_original_split` | adapter present, `REPLAY_CODE_IMPLEMENTED`, `COMMAND_AUDITED` | no | no | yes (`model_combiner_rn50x4_noft`) |
| 13 | `clip4cir_rn50x4_fullft` | CLIP4Cir RN50x4 fullft | `WORKBENCH_CLIP4CIR_PYTHON` | **unset → `null`** | `…/checkpoints/clip4cir/FashionIQ/RN50x4_fullft/` (bundle dir) | dir 949 892 616 B / 2 files — not hashed (size) | CLIP RN50x4 `VERIFIED` (same file as row 12) | `data/FashionIQ`, `fashioniq_standard` | `fashioniq_original_split` | `BLOCKED_CHECKPOINT_MAPPING`, `COMMAND_AUDITED` | no | no | yes (`model_clip4cir_rn50x4_fullft`) |
| 14 | `tgcir` | TG-CIR | `WORKBENCH_TGCIR_PYTHON` | `workbench/envs/tgcir39-la/bin/python` | `…/downloads/tgcir/x/checkpoints/Fashioniq.pt` | 607 928 637 B, sha256 `c37768f374c75bebff1299e870518c23bf6db3dc8a75c4d1f729ccd6dd02e966` (= declared) | OpenAI CLIP ViT-B/16 `VERIFIED` (`~/.cache/clip/ViT-B-16.pt`, 350 837 078 B) | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_val_split` | `replay_type: PREFLIGHT_ONLY`, `COMMAND_BLOCKED_SOURCE_ENTRYPOINT`, `replay/tgcir.py` exists | no | no | yes (`model_tgcir`) |
| 15 | `sprc` | SPRC | `WORKBENCH_SPRC_PYTHON` | **unset → `null`** | `…/checkpoints/sprc/sprc_fiq.pt` **absent**; `sprc_fiq_vitl.pt` present | 1 459 756 541 B, sha256 `19386104271c01ea03ee2177575c0e55f98edd3ba61a116d78f2e61ab5dbea08` (declared) | LAVIS BLIP-2 `LOCALLY_RECORDED_ASSET`; EVA-ViT-g `LOCALLY_RECORDED_ASSET` (registry `provenance_status: UNVERIFIED`) | `data/FashionIQ`, `fashioniq_standard` | `fashioniq_original_split` | **no adapter** (`composition_adapter` absent), `BLOCKED_CHECKPOINT_MAPPING`, no `replay/sprc.py` | no | **no adapter → cannot compose** | yes (`model_sprc`) |
| 16 | `limn` | LIMN | `WORKBENCH_LIMN_PYTHON` | `workbench/envs/limn310-adapted/bin/python` | `…/limn/0_dress_best_model.pt`, `0_shirt_…`, `0_toptee_…` (3 members) | 1 792 010 157 B each; sha256 `96a335d4…`, `9260c7f7…`, `79f663a1…` (all = declared) | none | `data/FashionIQ`, `fashioniq_ilearn_resized_training` | `fashioniq_val_split` | **yes** — `per_query_observer: LIMN_QUERY_OBSERVER_V1`, `replay_type: SOURCE_FAITHFUL_WORKBENCH_REPLAY` | no | no | yes (`model_limn`) |
| 17 | `tme` | TME | `WORKBENCH_TME_PYTHON` | `workbench/envs/tme-adapted/bin/python` | `…/tme/fashioniq_0_best_model.pth` present; `cirr_0_best_model.pth` **absent** | 2 728 382 104 B — not hashed (size); no declared sha256 | BLIP-2 `LOCALLY_RECORDED_ASSET`; EVA-ViT-g `LOCALLY_RECORDED_ASSET`; bert Qformer `VERIFIED` | `data/FashionIQ`, `fashioniq_standard` | `fashioniq_original_split` | adapter present, `REPLAY_COMMAND_IMPLEMENTED`, not run | no (has a `cirr_0` variant, no `transfer_condition`) | no | yes (`model_tme`) |

Notes on the table's derivation:

* "Resolved interpreter" = `resolve_model_interpreter()` output; `null` means the declared
  `python_env_var` is neither in the process environment nor in `workbench/config/workbench.env`, and
  `workbench/artifacts/environment_assignment.json` holds no `RUNTIME_VERIFIED` record for that model.
  No model declares `interpreter_sha256` (all 17 `None`), so no interpreter is digest-pinned.
* "Native eval" = a `workbench/replay/*.py` module plus a non-blocked `command_status`/`replay_status`;
  the registry has **no** boolean `native_eval_supported` field — this column is read from
  `composition_adapter`, `replay_type`, `replay_status`, `command_status`, `per_query_observer`.
* "Aggregate-only" = the registry's own rule (`models.yaml:1479`): a model with **no**
  `per_query_observer` is aggregate-only by design. Only `encoder` and `limn` declare an observer.
* "Transfer" is only declared by `ptha_mtst` (`checkpoint_variants[0].transfer_condition:
  cross_dataset_transfer`); no other model declares one.
* Checkpoint sha256 was computed with `sha256sum` only for files **< 2 GiB**. Files ≥ 2 GiB are recorded
  as size + suffix and marked "not hashed (size)"; directory artifacts (bundles/run dirs) are recorded as
  total bytes + member count, with the registry's per-member declared digests where present.
* Resolved checkpoint paths shown elided with `…`; the unabridged strings are in
  `raw/baseline/14-resolved-checkpoints.txt` and `raw/baseline/18-checkpoint-variants.json`.

---

## 5. Data root and assets

### Operator configuration (`workbench/config/workbench.env`, read-only; keys only, no secrets)

| Key | Value | Points at |
|---|---|---|
| `CIR_REPO_ROOT` | `/home/heheboiz/data/cir` | repo |
| `CIR_DATA_ROOT` | `/home/heheboiz/data/cir/data` | data root |
| `FASHIONIQ_ROOT` | `/home/heheboiz/data/cir/data/FashionIQ` | FashionIQ root |
| `WORKBENCH_ARTIFACTS_ROOT` | `/home/heheboiz/data/cir/workbench/artifacts` | artifacts |
| `WORKBENCH_CHECKPOINT_ROOT` | `…/checkpoint_selection/20261106T002617+0700/cohort/downloads` | checkpoints |
| `WORKBENCH_RESULTS_ROOT` | `…/artifacts/results` | results |
| `WORKBENCH_THIRD_PARTY_ROOT` | `/home/heheboiz/data/cir/workbench/third_party` | vendored source |
| `WORKBENCH_CIRCUS_RELEASE_ROOT` | `…/artifacts/circus_release` | release tree |
| `WORKBENCH_ENCODER_PYTHON` | `…/envs/encoder310-adapted/bin/python` | interpreter |
| `WORKBENCH_LIMN_PYTHON` | `…/envs/limn310-adapted/bin/python` | interpreter |
| `WORKBENCH_HABIT_PYTHON` | `…/envs/ilearn39-adapted/bin/python` | interpreter |
| `WORKBENCH_INTENT_PYTHON` | `…/envs/ilearn39-adapted/bin/python` | interpreter |
| `WORKBENCH_HINT_PYTHON` | `…/envs/ilearn39-adapted/bin/python` | interpreter |
| `WORKBENCH_CONESEP_PYTHON` | `…/envs/ilearn39-adapted/bin/python` | interpreter |
| `WORKBENCH_PTHA_MTST_PYTHON` | `…/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | interpreter |
| `WORKBENCH_TME_PYTHON` | `…/envs/tme-adapted/bin/python` | interpreter |
| `WORKBENCH_CSMCIR_PYTHON` | `…/artifacts/environments/cir-005ef4e1b02999a5/bin/python` | interpreter |
| `WORKBENCH_RUNTIME_ADAPTATION` | `sm_120` | runtime adaptation label |
| `WORKBENCH_CLVC_PYTHON` | `…/envs/clvc39-la/bin/python` | interpreter |
| `WORKBENCH_TGCIR_PYTHON` | `…/envs/tgcir39-la/bin/python` | interpreter |

**Absent from this file** (so those models resolve to `null`): `WORKBENCH_AIRKNOW_PYTHON`,
`WORKBENCH_PAIR_PYTHON`, `WORKBENCH_DCNET_PYTHON`, `WORKBENCH_CLIP4CIR_PYTHON`, `WORKBENCH_SPRC_PYTHON`.
Command: `sed -n '1,200p' workbench/config/workbench.env` (read-only; file not written).

`/home/heheboiz/cir_rerun_20261109.yaml` (937 B, read-only) — config keys, not secrets:

| Key | Value |
|---|---|
| `schema_version` | `1` |
| `selected_models` | `all_registered` |
| `checkpoint_policy` | `include_flagged_conditions` |
| `checkpoint_overrides` | `habit:[fiq_n02]`, `intent:[fiq_n02]`, `conesep:[fiq_n02]`, `ptha_mtst:[cirr_transfer]`, `encoder:[fashioniq]` |
| `paths.registry` | `/home/heheboiz/data/cir/workbench/registry/models.yaml` |
| `paths.dataset_root` | `/home/heheboiz/data/cir/data/FashionIQ` |
| `paths.artifacts` | `/home/heheboiz/data/cir/workbench/artifacts` |
| `paths.results` | `…/artifacts/results` |
| `paths.checkpoints` | `…/checkpoint_selection/20261106T002617+0700/cohort/downloads` |
| `paths.third_party` | `/home/heheboiz/data/cir/workbench/third_party` |
| `budgets` | `max_download_bytes: 37580963840`, `max_models: 32`, `top_k: 200` |
| `permissions` | all `true` (`allow_network`, `allow_large_downloads`, `allow_env_install`, `allow_preparation`, `allow_gpu_eval`, `allow_gpu_preparation`, `allow_index_write`) |

### FashionIQ data roots and annotations

| Path | Exists | Size | Command |
|---|---|---|---|
| `data/FashionIQ` | yes (**6** subdirs; the `8` previously recorded here is wrong — measured 2026-11-12) | — | `find data/FashionIQ -maxdepth 1 -type d` |
| `data/FashionIQ/images` | yes | 991 916 036 B, 74 381 files, extension `.png` | `du -sb`; `ls \| wc -l` |
| `data/FashionIQ/resized_image/{dress,shirt,toptee}` | yes | 288 392 173 B; 3 817 / 6 346 / 5 373 `.jpg` | `du -sb`; per-dir `os.listdir` census |
| `data/FashionIQ/resized_images` | yes | 286 225 672 B, 15 415 `.jpg` | same |
| `data/FashionIQ/captions` | yes | 403 551 063 B, **12** JSON + 10 `.pkl` (22 files; the `21 JSON` previously recorded here was the *combined* annotation count of `captions` + `image_splits`) | `du -sb`; `ls -l` |
| `data/FashionIQ/image_splits` | yes | 1 398 312 B, 9 JSON | same |
| `data/FashionIQ/qwen_captions` | yes | 11 762 668 B, 3 JSON | same |

Layout completeness, evaluated through the registry's own loader
(`workbench.backend.fashioniq_layout.paths_for_layout`) against every declared layout:
`fashioniq_standard` 9 paths, 0 missing; `fashioniq_ilearn_resized_training` 18 paths, 0 missing;
`fashioniq_clvc_resized` 15 paths, 0 missing; `fashioniq_dcnet` 9 paths, 0 missing. Command: see
`raw/baseline/23-layout-path-check.json`.

Annotation `sha256sum` (small files only):

| File | Size (B) | sha256 |
|---|---|---|
| `captions/cap.dress.test.json` | 335557 | `190240b3b4d115cd4f80de65ab78ca18a3f8581c041cbeea6f5a1f57ee4f5a48` |
| `captions/cap.dress.train.json` | 1178488 | `1691bd52e5bbda755b0c1f14d1024000be2c57a0cc29511a327af1372636a946` |
| `captions/cap.dress.val.json` | 397427 | `5e5117d45695df9c3ca91fac3e0bc49e6422ae9c8b3793f0ac9b3ab83adb7de9` |
| `captions/cap.shirt.test.json` | 333382 | `27c7551b005d4befa53135a7ddee5e31c97821eafd3ceb3fec514e7ac0ad83a7` |
| `captions/cap.shirt.train.json` | 1167604 | `0999ce5da01a38f0cef1547941b738f4c56e225fdd9b8e03e2a6edaf292f94c3` |
| `captions/cap.shirt.val.json` | 398072 | `7b7ca3797b85dddfd83e74cdb4454cb63e3e7c09acef2590155efbacea6d7feb` |
| `captions/cap.toptee.test.json` | 345441 | `db822aa52340ce1f09c3000064f79b60b6c476b50b98bf34200dd2996c06e146` |
| `captions/cap.toptee.train.json` | 1200387 | `6daa171a9e744823087780a3b69b8b697bd05526adb6dd55a38ee6c4904ab348` |
| `captions/cap.toptee.val.json` | 391897 | `b4e09f428e6c255cc4b4ec46e77908305ab83075d82840c960be81af15cdc5eb` |
| `captions/correction_dict_dress.json` | 9479 | `9e4a59877098fe26036b70759387be589990eb4068692427e890113dcba2edda` |
| `captions/correction_dict_shirt.json` | 9064 | `ca29e29e23c0a1987d6b121219d9580bdae9872df0ea219c5aea870ee2f8567e` |
| `captions/correction_dict_toptee.json` | 10019 | `3dfbf15fa6d3a30dc0f5cf7e2908ca36d1775e42aa6598f1dba7899267dc1b71` |
| `image_splits/split.dress.test.json` | 68726 | `1d1d3af24b9ec4ed3f7805c53a870148df9b51e232b32912505d8ae54718353d` |
| `image_splits/split.dress.train.json` | 206138 | `dcb1c7329003e85e39a3010147a620cee4b7de6a5ddf2626f253bc9a2cdd78e5` |
| `image_splits/split.dress.val.json` | 68708 | `21ff91d53c23859da91bfd49f3acc139b7f3a3dc944fc4e4a80194468cf14bab` |
| `image_splits/split.shirt.test.json` | 114230 | `8d243830e80e77c23c8dc9a85db87813230d38da146f0706082db812f591968b` |
| `image_splits/split.shirt.train.json` | 342650 | `1552c05d866d31b43321a719b2cb8a65f02c53e9a6bc355861fe1d62498b9983` |
| `image_splits/split.shirt.val.json` | 114230 | `b82effbf7352a6f828b38c45eb32e53d726bd4d6fd81d6290c81eeb2c68e3234` |
| `image_splits/split.toptee.test.json` | 96734 | `25a30d4f4013a04026585399b10980db7cba39dfaad0d97a0671a62e2125ddbf` |
| `image_splits/split.toptee.train.json` | 290180 | `aa9b3a55332340f21360fb6a3fe174badd27ce389fe7d78d890bf2b9032ffe05` |
| `image_splits/split.toptee.val.json` | 96716 | `ee42b2505275dd7a19b11ddb88c9264aab2d77f4e59b075fb1c7c3095d02f412` |

Rebuilt from `raw/baseline/19-annotation-sha256.txt` on 2026-11-12, one row per line of that file:
the raw capture pairs each digest with one path on the same line, and it is **complete** — 21 digests,
21 paths, no elision, no `+N more` marker (the file has 44 lines in total: header, 21 digest lines, the
`=== sizes ===` section, and 21 `path bytes` lines). Every digest and every byte count above was also
re-measured on disk this turn with
`sha256sum data/FashionIQ/captions/*.json data/FashionIQ/image_splits/*.json` and `os.path.getsize`, and
all 21 pairs matched.

Superseded cells, kept so the shift is visible rather than silent: the table previously assigned
`1552c05d…98b9983` to `cap.shirt.train.json` (it is `split.shirt.train.json`, raw:18),
`0999ce5d…292f94c3` to `split.dress.train.json` (it is `cap.shirt.train.json`, raw:6;
`split.dress.train.json` is `dcb1c732…2cdd78e5`, raw:15), and marked three rows
"— not in captured head" / "— see raw file" that the raw file does contain:
`cap.toptee.test.json` `db822aa5…6c06e146` (raw:8), `split.toptee.val.json` `ee42b250…5d02f412` (raw:22),
`cap.toptee.val.json` `b4e09f42…15cdc5eb` (raw:10, unchanged). The earlier `split.shirt.train.json`
cell reading `` `1552c05d…` prefix collision note `` is now the full digest.
Sizes were also coarse (`1.1M` for three files 1.18M–1.20M); they are exact bytes now.

Command: `sha256sum data/FashionIQ/captions/*.json data/FashionIQ/image_splits/*.json` (full, unabridged
list in `raw/baseline/19-annotation-sha256.txt`). That raw file is **not** interleaved: each line is
`<digest>  data/FashionIQ/<dir>/<file>`, so every digest and path read off it is unambiguous. The
"transcript rendering interleaved some long paths" note this paragraph used to carry was wrong and is
what produced the shifted pairs — it should not be cited again.

CSMCIR caption assets match their declared digests exactly:
`qwen_captions/dress_cot_val.json` `b7129baf…b6a6` (2 817 386 B), `shirt_…` `bbe6f689…cc9`
(5 015 979 B), `toptee_…` `acdacc3d…c1cc` (3 929 303 B) — `sha256sum data/FashionIQ/qwen_captions/*.json`.

### Weight caches

| Location | File | Size | Command |
|---|---|---|---|
| `~/.cache/torch/hub/checkpoints` | `blip2_pretrained.pth` | 746 998 955 | `ls -l` / `os.path.getsize` |
| | `blip2_pretrained_vitL.pth` | 732 827 503 | same |
| | `clip_vit_L.pth` | 581 988 320 | same |
| | `eva_vit_g.pth` | 2 025 249 237 | same |
| | `resnet50-0676ba61.pth` | 102 530 333 | same |
| | `resnet50-19c8e357.pth` | 102 502 400 | same |
| `~/.cache/clip` | `RN50x4.pt` | 421 854 225 | same |
| | `ViT-B-16.pt` | 350 837 078 | same |
| `.cache/torch/hub/checkpoints` (repo-relative) | **absent** | — | `ls -l .cache/torch/hub/checkpoints` |

### Auxiliary asset registry (`workbench/registry/auxiliary_assets.yaml`, 21 entries)

Statuses produced by the workbench's own helper
`validate_auxiliary_asset(...)` (raw: `raw/baseline/17-aux-validation.json`):

| Model | Family | Status |
|---|---|---|
| `csmcir` | Qwen captions ×3 | `VERIFIED` |
| `csmcir` | COT_ours2 captions ×3 | `VERIFIED` |
| `csmcir` | Author-published bytecode `.pyc` (12 281 B) | `VERIFIED` |
| `csmcir` | LAVIS BLIP-2 pretrain weights | `LOCALLY_RECORDED_ASSET` (declared `UNVERIFIED`) |
| `csmcir` | EVA-ViT-g backbone weights | `LOCALLY_RECORDED_ASSET` (declared `UNVERIFIED`) |
| `csmcir` | bert-base-uncased tokenizer (hub dir) | `VERIFIED` |
| `encoder` | LAION CLIP ViT-B/32 (605 219 813 B) | `VERIFIED` |
| `clvc_net`, `dcnet` | torchvision ResNet-50 ImageNet | `VERIFIED` |
| `combiner_rn50x4_noft`, `clip4cir_rn50x4_fullft` | OpenAI CLIP RN50x4 | `VERIFIED` |
| `tgcir` | OpenAI CLIP ViT-B/16 | `VERIFIED` |
| `sprc` | LAVIS BLIP-2 pretrain weights | `LOCALLY_RECORDED_ASSET` (declared `UNVERIFIED`) |
| `sprc` | EVA-ViT-g backbone weights | `LOCALLY_RECORDED_ASSET` (declared `UNVERIFIED`) |
| `airknow` | LAVIS BLIP-2 / EVA-ViT-g | `LOCALLY_RECORDED_ASSET` ×2 |
| `ptha_mtst` | EVA-ViT-g / bert tokenizer | `LOCALLY_RECORDED_ASSET` / `VERIFIED` |
| `tme` | BLIP-2 / EVA-ViT-g / bert Qformer | `LOCALLY_RECORDED_ASSET` ×2 / `VERIFIED` |

No auxiliary asset is `MISSING`.

---

## Discrepancies

Every item below is a measured state contradicting a claim in a document or in the tasking.

1. **HEAD sha.** `workbench/docs/reviews/20261111-master-audit/FINAL_REPORT_20261112.vi.md` states, as
   the single source for current state (§0.1 prose), HEAD `cd9ced2` `== origin`; its own §0 code block
   shows `04f6f4c8147d35f956e43f5a00f04d1028d8252d`. Measured HEAD is
   `40c897c31bbdd76b5a76f65935c7dd0f7fef0e94`, equal to `origin`, **two commits after `cd9ced2`**
   (`c569427`, `40c897c`) and unrelated to `04f6f4c8`. The document is stale by two documentation commits.
2. **Preserved untracked files.** `FINAL_PC_RELEASE_VERDICT.md` row `Worktree` claims
   "preserved untracked user files: `pc_composition_smoke.yaml`, `src/train_cirr.py`, `uv.lock`".
   All three are **absent** (`ls` → "No such file or directory"). The only untracked entry in the
   worktree is `CLEANUP_REPORT.md.save`.
3. **Production-tree comparison base.** The verdict's `Production tree` row instructs comparing against
   `96d86010…` and asserts "the release guard requires byte-identical production" against it. Measured:
   `git diff --quiet 96d86010..HEAD` over production paths exits **1** (30 files, +3622 −265), while the
   same check against `aea143bd` exits **0**. The byte-identical invariant holds against `aea143bd` only;
   the row names the superseded base.
   **Re-measured 2026-11-12 after the row was corrected** (the row now names
   `aea143bd8894881ad7d5b341c21dbeff25cfaaaf`, and `test_handoff_commands.py` looks only at
   `TESTED_SOURCE_SHA`, never at that row): the row's base is right again, but the invariant no longer
   holds against `HEAD` either — `git diff --quiet aea143bd HEAD -- . ':(exclude)workbench/docs'
   ':(exclude)workbench/tests'` now exits **1** (2 files, +356 −41) because two production commits landed
   after the pin, `cb0b228` and `23e5b8f`, and a further 2 files / +44 −4 are uncommitted
   (`parity_gate.py`, `gate.ts`). `96d86010..HEAD` is 31 files / +3956 −284. So the pin is currently
   behind the tree and must be advanced; the guard's single failing assertion is exactly this
   comparison, not a documentation error.
4. **Frontend bundle name/sha.** `20261111-master-audit/FINAL_REPORT_20261112.vi.md` §9 recorded the
   served bundle as `assets/index-DDviexaW.js`, sha256 `58eed6d48640f…f5369`, 392 261 B, mtime
   `09:34:18.594`; `FINAL_REPORT.vi.md` recorded `assets/index-DYYSN7oG.js`, sha256 `74e4b4f8…d651f`.
   Measured served bundle is **`assets/index-NnjrdZLG.js`**, sha256
   `721d34d043858e80c88e76d72866ba8583508318baeebac599cebd1e1c2c80d0`, **392 769 B**, mtime
   `2026-11-12 12:20:56.749`; CSS is `index-DecQnta9.css` (matches the CSS name both docs record).
   `dist/` and the proxy serve byte-identical bytes.
   **Corrected in the report 2026-11-12:** §1.5 and the §9 bundle row now both read
   `assets/index-NnjrdZLG.js` with the sha256 and size above, and `index-DDviexaW.js` /
   `index-DYYSN7oG.js` are labelled historical names. Re-measured this turn: served page, `dist/`, and
   the proxied file still carry that one sha256. (The stale `index-DDviexaW.js` mentions that remain are
   in `raw/verify/105-*` and `106-*`, which are historical sweep records of their own revisions and are
   annotated as such, not rewritten.)
5. **Backend MainPID / NRestarts.** The §0 historical block in `FINAL_REPORT_20261112.vi.md` records
   backend `MainPID=1221379`, `NRestarts=1`, `ActiveEnterTimestamp=Wed 2026-11-11 23:48:58`; measured
   backend is `MainPID=1758096`, `NRestarts=0`, `ActiveEnterTimestamp=Thu 2026-11-12 12:20:57`. (That
   document's own §0.1 block agrees with the measurement; the §0 block is the stale one. Frontend
   `1221148` / `NRestarts=0` / `23:48:44` matches in both.)
6. **GPU memory in use.** `FINAL_REPORT_20261112.vi.md` §0 records `839 MiB` used; measured is
   **`991 MiB` used of 16 303 MiB** with **no** compute-app rows at all — i.e. ~1 GiB is held by a
   non-compute-app consumer while the card reports 10% utilisation.
7. **`/tmp/cir_gpu.lock` exists with no holder.** The lock file is present (0 bytes, `Nov 12 08:40`) and
   `pgrep -af 'composition_eval.py|pytest'` returns nothing. A zero-byte lock with no live job is a
   stale-looking lock; it was left untouched as instructed.
8. **`pair` has a usable interpreter on disk but none assigned.** `workbench/envs/pair39-la/bin/python`
   exists (CPython 3.9.25) yet `WORKBENCH_PAIR_PYTHON` is absent from `workbench.env`, so the resolver
   returns `null` and the API reports the blocker
   `WORKBENCH_PAIR_PYTHON=None [UNCONFIGURED/UNCONFIGURED]`. Same class for `airknow`, `dcnet`,
   `sprc`, `clip4cir_*` — and for those four the corresponding `*-adapted` env has **no** `bin/python`.
9. **Parity gate is empty.** `/api/runs/parity-gate` returns `passed_models: []`; both registered models
   it covers (`encoder`, `limn`) are `PARITY_MANDATORY_SUBCHECK_FAILED`. Any claim that parity currently
   passes for those models is contradicted by the live endpoint.
10. **Registry count reconciles.** The prompt's "17 registered configurations" is **confirmed**: 17
    `model_id` entries in `workbench/registry/models.yaml`, 17 rows in `/api/models`. No discrepancy.
    (The tasking's guess that the registry lives under `workbench/config/` or `workbench/models/` is
    wrong: it is `workbench/registry/models.yaml`; neither of those directories exists.)

## Final `git status --porcelain`

Verbatim, after all collection and after writing this record (raw: `raw/baseline/25-final-status.txt`):

```
?? CLEANUP_REPORT.md.save
?? workbench/docs/reviews/20261112-parity-repair/
```

`CLEANUP_REPORT.md.save` is the operator's untracked file (untouched). The second entry is this
baseline directory, the only thing created by this pass. No tracked file was modified, and nothing was
staged or committed.
