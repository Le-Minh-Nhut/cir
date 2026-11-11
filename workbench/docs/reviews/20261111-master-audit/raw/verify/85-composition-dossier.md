# 85 — Composition dossier: `clvc_net` + `tgcir`

Read-only fact extraction, branch `research/cir-failure-workbench`, 2026-11-11.
No edits, no builds, no service contact, no GPU (all model loads used `map_location="cpu"`;
`torch.cuda.is_available()` was only *reported*, never exercised).

Working interpreter for repo code: `/home/heheboiz/data/cir/.venv/bin/python` (3.12.11).
Reason: the system `python3` (3.12.3) cannot import `workbench.backend.runtime` (no `packaging`),
which makes `operator_config.resolve_config()` raise `ValueError: unknown setting
workbench/config/workbench.env:20: WORKBENCH_RUNTIME_ADAPTATION`. See §10.

---

## 1. Registry rows

Command (JSON dump, no hand-copying):

```
cd /home/heheboiz/data/cir && python3 /tmp/audit85/finaljson.py
# finaljson.py: yaml.safe_load("workbench/registry/models.yaml");
#   for each model_id in ("clvc_net","tgcir"): json.dumps(model, indent=2),
#   with any key named "expected_sha256" truncated to v[:16]
```

Verbatim output:

#### `model_id: clvc_net`
```json
{
  "model_id": "clvc_net",
  "composition_adapter": "workbench.composition.model_clvc_net",
  "replay_status": "BLOCKED_NO_SOURCE_REPLAY",
  "category_model_policy": "independent",
  "category_checkpoint_structure": {
    "categories": [
      "dress",
      "shirt",
      "toptee"
    ],
    "per_category_members": [
      "local",
      "global"
    ],
    "member_count_expected": 6,
    "mapping_status": "VERIFIED_METADATA",
    "notes": "six files present and structurally confirmed; three per-category vocabularies recovered from the checkpoints match the three categories' train captions exactly."
  },
  "checkpoint_serialization": "whole_model",
  "environment_required": true,
  "method_name": "CLVC-Net",
  "research_generation": "legacy",
  "publication_year": 2021,
  "paper_title": "Comprehensive Linguistic-Visual Composition Network for Image Retrieval",
  "paper_url": "https://doi.org/10.1145/3404835.3462967",
  "upstream_repo_url": "https://github.com/iLearn-Lab/SIGIR21-CLVC-Net",
  "upstream_default_branch": "main",
  "upstream_commit_sha": "bd9b6889489806cb60676597880f91d918b279cb",
  "source_available": true,
  "source_dir": "CLVCNet",
  "fashioniq_layout": "fashioniq_clvc_resized",
  "fashioniq_image_extensions": [
    ".jpg"
  ],
  "preparation_contract": "manual_clvc_resized",
  "preparation_id": "clvc_net_fashioniq",
  "literature_split_label": "val",
  "native_protocol": "fashioniq_val_split",
  "supported_protocols": [
    "fashioniq_val_split"
  ],
  "reported_r10": 30.7,
  "reported_r50": 58.41,
  "reported_mean": 44.56,
  "reproduction_status": "BLOCKED_CHECKPOINT_MAPPING",
  "command_status": "COMMAND_BLOCKED_SOURCE_ENTRYPOINT",
  "environment": {
    "python_env_var": "WORKBENCH_CLVC_PYTHON",
    "confidence": "VERIFIED_README",
    "python": "3.7.6",
    "pytorch": "1.6.0",
    "torchvision": "UNKNOWN",
    "cuda": "10.2",
    "packages": [
      "torchvision"
    ]
  },
  "strict_clean_checkpoint_available": null,
  "notes": "Native pair-union and reference exclusion verified. Pinned test.py has no CLI/model loader, so `workbench/replay/clvc_net.py` supplies the shell and loads the six per-category whole models by name. The staged six files are confirmed against the pinned classes and the three category vocabularies; the renamed single-file declaration that preceded this pointed at no existing artifact. Evaluation requires torchvision ResNet-50 ImageNet weights through an unpinned Torch cache, but the six whole-model pickles already carry 318 of the 320 ResNet-50 tensors (only `fc.*` is absent, and `model.py:78` replaces it with a bare `nn.Sequential`), so that download governs module construction and cannot change a scored parameter.",
  "checkpoint_variants": [
    {
      "checkpoint_id": "author_drive_unresolved",
      "filename": ".",
      "artifact_type": "bundle_directory",
      "required_files": [
        "dress_local_model.pt",
        "dress_global_model.pt",
        "shirt_local_model.pt",
        "shirt_global_model.pt",
        "toptee_local_model.pt",
        "toptee_global_model.pt"
      ],
      "bundle_members": [
        {
          "filename": "dress_local_model.pt",
          "expected_sha256": "33d44186efd6a25c"
        },
        {
          "filename": "dress_global_model.pt",
          "expected_sha256": "be41c2ba81b803fd"
        },
        {
          "filename": "shirt_local_model.pt",
          "expected_sha256": "2ba8e320b08ebb97"
        },
        {
          "filename": "shirt_global_model.pt",
          "expected_sha256": "78b1d660227ab692"
        },
        {
          "filename": "toptee_local_model.pt",
          "expected_sha256": "d1fa9170496f73a0"
        },
        {
          "filename": "toptee_global_model.pt",
          "expected_sha256": "c78a58c4becafee3"
        }
      ],
      "download_url": null,
      "source_url": "https://drive.google.com/file/d/159rBhWyhkLN7sXAi8iyW_ljzFNLJinKa/view?usp=sharing",
      "source_type": "author_google_drive",
      "checkpoint_training_noise_pct": null,
      "evaluation_noise_pct": 0,
      "expected_size_bytes": null,
      "expected_sha256": null,
      "status": "URL_VERIFIED",
      "checkpoint_mapping_status": "VERIFIED_METADATA",
      "checkpoint_base_relative": "../../checkpoints/clvc"
    }
  ]
}
```

Note the two size/digest asymmetries in the `clvc_net` row, verbatim: the **variant** carries
`"expected_size_bytes": null` and `"expected_sha256": null`; only the six **bundle_members** carry
`expected_sha256`, and **none** of them declares an `expected_size_bytes` (see §3).

#### `model_id: tgcir`
```json
{
  "model_id": "tgcir",
  "composition_adapter": "workbench.composition.model_tgcir",
  "replay_status": "PREFLIGHT_ONLY",
  "category_model_policy": "shared",
  "checkpoint_serialization": "whole_model",
  "replay_type": "PREFLIGHT_ONLY",
  "environment_required": true,
  "method_name": "TG-CIR",
  "research_generation": "legacy",
  "publication_year": 2023,
  "paper_title": "Target-Guided Composed Image Retrieval",
  "paper_url": "https://arxiv.org/abs/2309.01366",
  "upstream_repo_url": "https://github.com/iLearn-Lab/MM23-TG-CIR",
  "upstream_default_branch": "main",
  "upstream_commit_sha": "65fa78eaf8cabe8197fcc22ccab42207c591bdee",
  "source_available": true,
  "source_dir": "TGCIR",
  "fashioniq_layout": "fashioniq_ilearn_resized_training",
  "fashioniq_image_extensions": [
    ".jpg"
  ],
  "preparation_contract": "manual_ilearn_resized",
  "preparation_id": "tgcir_fashioniq",
  "literature_split_label": "val",
  "native_protocol": "fashioniq_val_split",
  "supported_protocols": [
    "fashioniq_val_split"
  ],
  "reported_r10": 51.32,
  "reported_r50": 73.09,
  "reported_mean": null,
  "reported_score_note": "Paper reports macro R@10 and R@50; 62.205 is derivable but not presented as a reproduced metric.",
  "reproduction_status": "BLOCKED_COMMAND",
  "command_status": "COMMAND_BLOCKED_SOURCE_ENTRYPOINT",
  "environment": {
    "python_env_var": "WORKBENCH_TGCIR_PYTHON",
    "confidence": "VERIFIED_README",
    "python": "UNKNOWN",
    "pytorch": "1.7.0",
    "torchvision": "0.8.0",
    "cuda": "11.0",
    "packages": [
      "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33",
      "numpy==1.22.3",
      "pandas==1.4.2",
      "Pillow"
    ]
  },
  "strict_clean_checkpoint_available": null,
  "notes": "Native annotation endpoint-union/reference-excluded evaluator verified. Pinned test.py has no CLI/checkpoint replay, so `workbench/replay/tgcir.py` supplies the shell. The extracted archive member is the FashionIQ checkpoint and loads strictly into `model.VCG_CIR`; only the archive-member NAME was ever unresolved.",
  "checkpoint_variants": [
    {
      "checkpoint_id": "tgcir_archive_unresolved",
      "filename": "Fashioniq.pt",
      "download_url": null,
      "source_url": "https://drive.google.com/file/d/1OdZTtJqy-RTpYXBaq5ThmH3IBnGvYCyi/view",
      "source_type": "author_google_drive",
      "checkpoint_training_noise_pct": null,
      "evaluation_noise_pct": 0,
      "expected_size_bytes": 607928637,
      "expected_sha256": "c37768f374c75beb",
      "status": "URL_VERIFIED",
      "checkpoint_mapping_status": "VERIFIED_METADATA",
      "checkpoint_base_relative": "../../downloads/tgcir/x/checkpoints"
    }
  ]
}
```

### `adapted_runtime` for these two rows: ABSENT

Command:

```
cd /home/heheboiz/data/cir && python3 /tmp/audit85/noadapt.py
```

Verbatim output (recovered via `od -c`; the shell's `cat` buffer was stale for `/tmp` files):

```
clvc_net | environment keys: ['confidence', 'cuda', 'packages', 'python', 'python_env_var', 'pytorch', 'torchvision'] | has adapted_runtime: False
   keys containing 'adapt': ['composition_adapter']
   composition_adapter: workbench.composition.model_clvc_net
   command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
   native_protocol: fashioniq_val_split
   source_dir: CLVCNet
   upstream_commit_sha: bd9b6889489806cb60676597880f91d918b279cb
   category_model_policy: independent
   replay_status: BLOCKED_NO_SOURCE_REPLAY
   checkpoint_serialization: whole_model
tgcir | environment keys: ['confidence', 'cuda', 'packages', 'python', 'python_env_var', 'pytorch', 'torchvision'] | has adapted_runtime: False
   keys containing 'adapt': ['composition_adapter']
   composition_adapter: workbench.composition.model_tgcir
   command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT
   native_protocol: fashioniq_val_split
   source_dir: TGCIR
   upstream_commit_sha: 65fa78eaf8cabe8197fcc22ccab42207c591bdee
   category_model_policy: shared
   replay_status: PREFLIGHT_ONLY
   checkpoint_serialization: whole_model
```

So **neither target model declares an `adapted_runtime`**, and neither declares any key containing
`adapt` other than `composition_adapter`. Under the operator-wide
`WORKBENCH_RUNTIME_ADAPTATION=sm_120` that `workbench/config/workbench.env:20` sets, both models are
"not adapted": `runtime.runtime_adaptation(..., undeclared_is_error=False)` returns `None`
(`workbench/backend/runtime.py:1030-1034`), and `composition/runner.py:287-291` records
`adapted_runtime_applied: false` with the note "declares no matching adapted_runtime".

---

## 2. `adapted_runtime` template — every model that has one

Command:

```
cd /home/heheboiz/data/cir && python3 /tmp/audit85/adapted2.py
# walks every model dict recursively for a key named "adapted_runtime",
# prints model_id + JSON path + yaml.safe_dump of the block
```

**Nine** models declare it, and in **all nine it sits at `environment.adapted_runtime`** (never at
model top level). Verbatim blocks:

```yaml
########################################################################
model_id: csmcir | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.10'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - transformers==4.36.2
  - timm==0.9.12
  - fairscale==0.4.13
  - numpy<2
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: conesep | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.9'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - transformers==4.25.0
  - timm==0.9.16
  - fairscale==0.4.13
  - numpy<2
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: ptha_mtst | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.10'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - transformers==4.36.2
  - timm==0.9.12
  - fairscale==0.4.13
  - numpy<2
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: habit | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.9'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - transformers==4.25.0
  - timm==0.9.16
  - fairscale==0.4.13
  - numpy<2
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: intent | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.9'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - transformers==4.25.0
  - timm==0.9.16
  - fairscale==0.4.13
  - numpy<2
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: hint | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.9'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - transformers==4.25.0
  - timm==0.9.16
  - fairscale==0.4.13
  - numpy<2
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: encoder | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.10'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - open_clip_torch==2.24.0
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: limn | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  python: '3.10'
  pytorch: ==2.7.1
  torchvision: ==0.22.1
  cuda: '12.8'
  packages:
  - open_clip_torch==2.20.0
  pip_index_url: https://download.pytorch.org/whl/cu128
########################################################################
model_id: tme | paths: ['/environment/adapted_runtime']
--- /environment/adapted_runtime ---
adapted_runtime:
  target_compute_capability: sm_120
  pytorch: '>=2.7.1,<2.9'
  torchvision: '>=0.22.1,<0.24'
  cuda: '12.8'
  packages:
  - transformers==4.36.0
  - timm==0.9.12
  - fairscale==0.4.13
  - numpy<2
  pip_index_url: https://download.pytorch.org/whl/cu128
```

(The YAML source keeps explanatory `#` comments between these keys, e.g. `models.yaml:51-68` for
`csmcir`; `yaml.safe_load` drops them, which is why the dump above is comment-free.)

**One sentence:** the block carries a target compute capability plus a complete replacement
`python` / `pytorch` / `torchvision` / `cuda` pin set, an extra `packages` list, and a
`pip_index_url`; it is read by `workbench/backend/runtime.py` (`runtime_adaptation` at `:1030`,
`resolved_runtime_environment` at `:1057`, provenance at `:1889`), by
`workbench/backend/environment_planner.py` (`:566`, `:579`, `:701`, `:731-739`), by
`workbench/composition/runner.py` (`:279-291`), by `workbench/replay/limn.py`
(`_registry_adapted_runtime` at `:87-92`) and by the tests
`workbench/tests/test_runtime_adaptation.py` / `test_composition_topk_runner.py` /
`test_tme_feature_dtype_contract.py`.

Command as specified:

```
cd /home/heheboiz/data/cir && grep -rn "adapted_runtime" workbench/backend workbench/composition workbench/replay workbench/tests | head -40
```

Output is 40 lines, of which **22 are `__pycache__/*.pyc`** matches (compiled-docstring hits) and
only **18 are source lines**. The 18 source lines verbatim:

```
workbench/backend/environment_planner.py:564:        # on someone else's adaptation — UNLESS the model itself declares an adapted_runtime the
workbench/backend/environment_planner.py:566:        if (model.get("environment") or {}).get("adapted_runtime"):
workbench/backend/environment_planner.py:579:        env = {**env, "_adapted_runtime": str(adaptation.get("target_compute_capability"))}
workbench/backend/environment_planner.py:701:        if env.get("_adapted_runtime"):
workbench/backend/environment_planner.py:731:        if env.get("_adapted_runtime"):
workbench/backend/environment_planner.py:732:        evidence.append(f"adapted runtime selected: {env['_adapted_runtime']} "
workbench/backend/environment_planner.py:733:                         f"(historical pins replaced by the declared adapted_runtime block)")
workbench/backend/environment_planner.py:738:        if env.get("_adapted_runtime"):
workbench/backend/environment_planner.py:739:            spec["runtime_adaptation"] = env["_adapted_runtime"]
workbench/backend/runtime.py:69:    #: `adapted_runtime.target_compute_capability` when an operator-selected adaptation applied.
workbench/backend/runtime.py:1005:    #: Operator-selected runtime adaptation. Set to the registry's `adapted_runtime`
workbench/backend/runtime.py:1030:    declared = (model.get("environment") or {}).get("adapted_runtime") or {}
workbench/backend/runtime.py:1048:    registry declares a matching `adapted_runtime.target_compute_capability`; any other model is
workbench/backend/runtime.py:1055:    adapted_runtime the selection fails to match is still an operator error and refuses loudly.
workbench/backend/runtime.py:1889:        "adapted_runtime": runtime_adaptation(model, undeclared_is_error=False),
workbench/backend/schemas/results.py:107:    #: registry's ``adapted_runtime`` block. ``None`` means the historical environment ran
workbench/composition/adapters.py:334:    contract is the registry's own `adapted_runtime` block, which changes what the gate checks
workbench/composition/runner.py:279:            else {"adapted_runtime": None}
```

---

## 3. Checkpoint bytes (through the repo resolver)

Command:

```
cd /home/heheboiz/data/cir && PYTHONPATH=/home/heheboiz/data/cir .venv/bin/python /tmp/audit85/ckpt2.py
```

`ckpt2.py` calls `workbench.backend.registry.checkpoint_path(model_id, variant,
config.WORKBENCH_CHECKPOINT_ROOT)` — no path is guessed — then `stat()` + streaming `sha256`.

Verbatim output:

```
WORKBENCH_CHECKPOINT_ROOT = /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads
==============================================================================
[clvc_net] checkpoint_id='author_drive_unresolved' filename='.' artifact_type='bundle_directory'
[clvc_net] checkpoint_base_relative='../../checkpoints/clvc'
[clvc_net] checkpoint_path() -> /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/clvc
[clvc_net] normalized         -> /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc
[clvc_net] is_dir=True is_file=False
[clvc_net] variant expected_size_bytes=None expected_sha256[:16]=''
  file            : dress_local_model.pt
    path          : /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/clvc/dress_local_model.pt
    size_bytes    : 217168235 (declared None) -> NO_DECLARED_SIZE
    sha256[:16]   : 33d44186efd6a25c  (declared 33d44186efd6a25c) -> MATCH
  file            : dress_global_model.pt
    path          : /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/clvc/dress_global_model.pt
    size_bytes    : 176437419 (declared None) -> NO_DECLARED_SIZE
    sha256[:16]   : be41c2ba81b803fd  (declared be41c2ba81b803fd) -> MATCH
  file            : shirt_local_model.pt
    path          : /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/clvc/shirt_local_model.pt
    size_bytes    : 218603243 (declared None) -> NO_DECLARED_SIZE
    sha256[:16]   : 2ba8e320b08ebb97  (declared 2ba8e320b08ebb97) -> MATCH
  file            : shirt_global_model.pt
    path          : /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/clvc/shirt_global_model.pt
    size_bytes    : 177872427 (declared None) -> NO_DECLARED_SIZE
    sha256[:16]   : 78b1d660227ab692  (declared 78b1d660227ab692) -> MATCH
  file            : toptee_local_model.pt
    path          : /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/clvc/toptee_local_model.pt
    size_bytes    : 218308843 (declared None) -> NO_DECLARED_SIZE
    sha256[:16]   : d1fa9170496f73a0  (declared d1fa9170496f73a0) -> MATCH
  file            : toptee_global_model.pt
    path          : /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/clvc/toptee_global_model.pt
    size_bytes    : 177578027 (declared None) -> NO_DECLARED_SIZE
    sha256[:16]   : c78a58c4becafee3  (declared c78a58c4becafee3) -> MATCH
==============================================================================
[tgcir] checkpoint_id='tgcir_archive_unresolved' filename='Fashioniq.pt' artifact_type=None
[tgcir] checkpoint_base_relative='../../downloads/tgcir/x/checkpoints'
[tgcir] checkpoint_path() -> /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../downloads/tgcir/x/checkpoints/Fashioniq.pt
[tgcir] normalized         -> /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/downloads/tgcir/x/checkpoints/Fashioniq.pt
[tgcir] is_dir=False is_file=True
[tgcir] variant expected_size_bytes=607928637 expected_sha256[:16]='c37768f374c75beb'
  file            : Fashioniq.pt
    path          : /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../downloads/tgcir/x/checkpoints/Fashioniq.pt
    size_bytes    : 607928637 (declared 607928637) -> MATCH
    sha256[:16]   : c37768f374c75beb  (declared c37768f374c75beb) -> MATCH
```

Summary: **7/7 staged files resolve and 7/7 SHA-256 digests MATCH** the declared values.
Sizes: CLVC has **no declared size** (so size is recorded, not verified); TG-CIR's declared
`607928637` matches exactly.

Resolver detail worth noting: `checkpoint_path` returns an **un-normalised** path containing
`cohort/downloads/../../`, because the docstring (`registry.py:1884-1885`) deliberately skips
`resolve()` so the symlink-refusing presence checks are not defeated. Both bases are the *fallback*
branch (`checkpoint_base_relative`), i.e. the conventional `<root>/<model_id>` does not exist for
either model — `clvc_net` stages as `checkpoints/clvc`, `tgcir` as `downloads/tgcir/x/checkpoints`.

---

## 4. Environments

Command:

```
cd /home/heheboiz/data/cir && for e in clvc39-la tgcir39-la; do
  echo "########## $e"; workbench/envs/$e/bin/python /tmp/audit85/envprobe.py
done
```

(`envprobe.py` prints `sys.version`, torch/torchvision versions, `torch.cuda.is_available()`,
`import clip` result + `clip.__file__`, and the cache-related env vars. No forward pass, no GPU
allocation.)

Verbatim output:

```
########## clvc39-la
{
  "python_executable": "/home/heheboiz/data/cir/workbench/envs/clvc39-la/bin/python",
  "sys.version": "3.9.25 (main, Oct 31 2025, 23:00:23)  [Clang 21.1.4 ]",
  "torch.__version__": "2.7.1+cu128",
  "torch.version.cuda": "12.8",
  "torch.cuda.is_available()": true,
  "torch.cuda.device_count()": 1,
  "torchvision.__version__": "0.22.1+cu128",
  "import clip": "FAIL ModuleNotFoundError: No module named 'clip'",
  "HF_HOME(os.environ)": "<UNSET>",
  "HUGGINGFACE_HUB_CACHE": "<UNSET>",
  "TORCH_HOME": "<UNSET>",
  "XDG_CACHE_HOME": "<UNSET>",
  "HOME": "/home/heheboiz"
}
########## tgcir39-la
{
  "python_executable": "/home/heheboiz/data/cir/workbench/envs/tgcir39-la/bin/python",
  "sys.version": "3.9.25 (main, Oct 31 2025, 23:00:23)  [Clang 21.1.4 ]",
  "torch.__version__": "2.7.1+cu128",
  "torch.version.cuda": "12.8",
  "torch.cuda.is_available()": true,
  "torch.cuda.device_count()": 1,
  "torchvision.__version__": "0.22.1+cu128",
  "import clip": "OK",
  "clip.__file__": "/home/heheboiz/data/cir/workbench/envs/tgcir39-la/lib/python3.9/site-packages/clip/__init__.py",
  "HF_HOME(os.environ)": "<UNSET>",
  "HUGGINGFACE_HUB_CACHE": "<UNSET>",
  "TORCH_HOME": "<UNSET>",
  "XDG_CACHE_HOME": "<UNSET>",
  "HOME": "/home/heheboiz"
}
```

Both envs are the same CPython 3.9.25 from `~/.local/share/uv/python/cpython-3.9-linux-x86_64-gnu`
(the `bin/python` is a symlink), both on torch 2.7.1+cu128 / torchvision 0.22.1+cu128.
**`clip` is importable only in `tgcir39-la`.**

### `clip.load("ViT-B/16", device="cpu", jit=False)` in `tgcir39-la`

Command (HF vars explicitly unset, as required):

```
cd /home/heheboiz/data/cir && env -u HF_HOME -u HUGGINGFACE_HUB_CACHE -u TRANSFORMERS_CACHE \
  -u TORCH_HOME workbench/envs/tgcir39-la/bin/python /tmp/audit85/clipl.oad.py
```

Verbatim output:

```
HF_HOME in os.environ: <UNSET>
huggingface_hub probe FAIL: ModuleNotFoundError No module named 'huggingface_hub'
clip.__file__: /home/heheboiz/data/cir/workbench/envs/tgcir39-la/lib/python3.9/site-packages/clip/__init__.py
clip available models incl ViT-B/16: True
clip.load RESULT: SUCCESS
  type(model): CLIP
  param count: 149620737
  visual.input_resolution: 224
  logit_scale: 4.605170249938965
```

**No download was required, and no download occurred.** `HF_HOME` is genuinely unset and
**irrelevant to this loader**: the pinned OpenAI CLIP reads no HF variable — it resolves weights via
`download_root or os.path.expanduser("~/.cache/clip")`
(`workbench/envs/tgcir39-la/lib/python3.9/site-packages/clip/clip.py:120`). The resolved cache path is
therefore:

```
~/.cache/clip/ViT-B-16.pt   ->  /home/heheboiz/.cache/clip/ViT-B-16.pt
size 350837078 bytes, mtime 2026-08-18 13:27:49 +0700 (pre-existing, untouched by this audit)
sha256 5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f
```

That digest equals the digest segment of `clip.clip._MODELS["ViT-B/16"]`
(`.../clip/models/5806e77c…416f/ViT-B-16.pt`, `clip.py:37`), which is exactly the check
`workbench/replay/tgcir.py:142-171` performs. Note `huggingface_hub` is **not installed** in this
env, so "HF_HOME resolved path" has no meaning here; the only cache that matters is `~/.cache/clip`.

---

## 5. Pickle loadability — the crux

Command:

```
cd /home/heheboiz/data/cir && workbench/envs/clvc39-la/bin/python /tmp/audit85/tload.py
# sys.path.insert(0, "<repo>/workbench/third_party/CLVCNet")
# torch.load(p, map_location="cpu", weights_only=False)
```

Verbatim output:

```
sys.path[0] = /home/heheboiz/data/cir/workbench/third_party/CLVCNet
torch 2.7.1+cu128
target: /home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/checkpoints/clvc/dress_local_model.pt 217168235 bytes
RESULT: SUCCESS
type(obj): <class 'model.compose_local'>
module class name: compose_local
MRO: ['compose_local', 'Module', 'object']
state_dict() entries: 415
first 8 keys: ['loss_weight', 'text_model.embedding_layer.weight', 'text_model.lstm.weight_ih_l0', 'text_model.lstm.weight_hh_l0', 'text_model.lstm.bias_ih_l0', 'text_model.lstm.bias_hh_l0', 'img_model.img_model.conv1.weight', 'img_model.img_model.bn1.weight']
last 4 keys: ['local_fuse.inception_beta.branch3.1.bn.num_batches_tracked', 'local_pool.p', 'local_fc.weight', 'local_fc.bias']
total params in state_dict: 54236934
has fc.* keys: []
vocab probe: TypeError object of type 'SimpleVocab' has no len()
```

**It loads. No exception.** The pickle is a *whole module*, `model.compose_local`, resolved from
`workbench/third_party/CLVCNet/model.py` (which is why `sys.path` must include that directory);
`state_dict()` is populated with **415 entries / 54 236 934 tensors-elements** (of which
**54 170 563 are `parameters()`** — the remaining 201 entries are BN running stats /
`num_batches_tracked` buffers, verified by a second run: `parameters() 54170563`,
`state_dict total 54236934`, `buffers count 201`). `training` is already `False`.

Three consequences confirmed by direct measurement:

1. **`weights_only=True` (the torch ≥2.6 default) FAILS** on the same file:
   ```
   weights_only default: FAIL UnpicklingError | Weights only load failed. This file can still be loaded, to do so you have two options, do those steps only if you trust the source of the checkpoint.
   	(1) In PyTorch 2.6, we changed the default value of the `weights_only` argument in `torch.load` from `False` to `True`. Re-running `torch.l
   ```
   So any loader for these files must pass `weights_only=False` explicitly.
2. **All six files load**, not just `dress_local` (`/tmp/audit85/tload_all.py`):
   ```
   dress_local_model.pt       OK  class=compose_local    sd_entries= 415 params=  54236934 0.5s
   dress_global_model.pt      OK  class=compose_global   sd_entries= 415 params=  44054278 0.1s
   shirt_local_model.pt       OK  class=compose_local    sd_entries= 415 params=  54591238 0.1s
   shirt_global_model.pt      OK  class=compose_global   sd_entries= 415 params=  44408582 0.1s
   toptee_local_model.pt      OK  class=compose_local    sd_entries= 415 params=  54518534 0.0s
   toptee_global_model.pt     OK  class=compose_global   sd_entries= 415 params=  44335878 0.0s
   ```
3. **The ResNet-50 claim in `notes` is exactly right**: `img_model.img_model.*` contributes 318
   entries and torchvision's own `resnet50` has 320, with precisely `['fc.weight', 'fc.bias']`
   absent — the two tensors `model.py:75` discards (`self.img_model.fc = nn.Sequential()`).
   Also confirmed: each pickle's `SimpleVocab` carries `word2id` with **1602** entries
   (`vocab.__dict__` keys `['word2id','wordcount']`), consistent with the per-category vocabularies.

No GPU was touched; the process exited normally (`rc=0`, 1.39 s).

---

## 6. Upstream construction path (CLVC-Net)

`workbench/third_party/CLVCNet/train.py` — construction, per category, lines 152-157 (the category
enters only through `texts`, the vocabulary built from that category's train captions at `:300`):

```
 152: def create_model_and_optimizer(texts):
 153:     """Builds the model and related optimizer."""
 154:     print('Creating model and optimizer')
 155: 
 156:     local_model = img_text_model.compose_local(texts, T=args.local_t, dropout_p=args.dropout_rate).cuda()
 157:     global_model = img_text_model.compose_global(texts, T=args.global_t, dropout_p=args.dropout_rate).cuda()
```

and the caller at `:299-300`:

```
 299:     trainset, testset = load_dataset()
 300:     local_model, global_model, optimizer_local, optimizer_global = create_model_and_optimizer([t.encode('utf-8').decode('utf-8') for t in trainset.get_all_texts()])
```

`torch.save` — lines 274-275, inside the `is_best` branch of `train_and_evaluate`:

```
 273:             utils.save_dict_to_json(test_metrics, best_json_path_combine)
 274:             torch.save(local_model, os.path.join(model_dir, 'local_model.pt'))
 275:             torch.save(global_model, os.path.join(model_dir, 'global_model.pt'))
```

i.e. **whole-module pickles, not `state_dict`** — matching `checkpoint_serialization: whole_model`
and the `compose_local` / `compose_global` classes recovered in §5.

`workbench/third_party/CLVCNet/model.py` — signatures and return tuples:

`compose_local` (`:130-140`):

```
 130:     def extract_img_feature(self, imgs):
 131:         imgs, img_feature = self.img_model(imgs)
 132:         return imgs, img_feature
 133: 
 134:     def compose_img_text(self, imgs, texts):
 135:         lstm_tensor = self.text_model(texts) # N,L,C
 136:         imgs, img_feature = self.extract_img_feature(imgs)
 137: 
 138:         local_out, attn = self.local_fuse(imgs, lstm_tensor) 
 139:         local_feature = self.local_fc(self.local_pool(local_out))  
 140:         return local_out, local_feature, attn
```

`compose_global` (`:202-212`):

```
 202:     def extract_img_feature(self, imgs):
 203:         imgs, img_feature = self.img_model(imgs)
 204:         return imgs, img_feature
 205: 
 206:     def compose_img_text(self, imgs, texts):
 207:         lstm_tensor = self.text_model(texts) # N,L,C
 208:         imgs, img_feature = self.extract_img_feature(imgs)
 209: 
 210:         global_out, attn = self.global_fuse(imgs, lstm_tensor)
 211:         global_feature = self.global_fc(self.global_pool(global_out))
 212:         return global_feature, attn
```

What each tuple element is:

- `extract_img_feature(imgs) -> (imgs, img_feature)`: `imgs` is the **spatial feature map**
  `N×2048×7×7` out of `img_backbone.forward` (`model.py:78-88` runs resnet conv1..layer4 and returns
  `imgs`), i.e. the perceptual/attention tensor; `img_feature` is the **pooled global 1024-d
  embedding** `img_fc(GeM(imgs))` (`:87`). Both callers use only `[1]`, the pooled vector.
- `compose_local.compose_img_text(imgs, texts) -> (local_out, local_feature, attn)`:
  `local_out` = fused spatial tensor `N×2048×7×7` from `local_conv`; `local_feature` =
  `local_fc(local_pool(local_out))`, the **1024-d local query embedding**; `attn` =
  `N×L×49` attention over text tokens × spatial positions.
- `compose_global.compose_img_text(imgs, texts) -> (global_feature, attn)`:
  `global_feature` = `global_fc(global_pool(global_out))`, the **1024-d global query embedding**;
  `attn` = `N×L×49` (text-token × image-patch, softmax over patches).

**The exact scored composition** is `test.py:100` and `:116` — 2048-d row = concat of the two
half-normalised 1024-d halves:

```
 100:                         f = torch.cat([F.normalize(local_model.compose_img_text(imgs, mods)[1]), F.normalize(global_model.compose_img_text(imgs, mods)[0])], dim=1)
 116:                         imgs = torch.cat([F.normalize(local_model.extract_img_feature(imgs)[1]), F.normalize(global_model.extract_img_feature(imgs)[1])], dim=1).data.cpu().numpy()
```

---

## 7. TG-CIR loading path

Construction — `workbench/third_party/TGCIR/train.py:123-125`:

```
 123: def create_model_and_optimizer():
 124:     model = img_text_model.VCG_CIR(img_encoder=args.img_encoder, hidden_dim=args.feature_dim, dropout=args.dropout_rate, local_token_num=args.Q, global_token_num = args.P, t = args.tau_)
 125:     model.cuda()
```

Saved — `train.py:270-271`, **whole module** again:

```
 270:         utils.save_dict_to_json(_metrics, os.path.join(args.model_dir, "metrics_best.json"))
 271:         torch.save(current_model, os.path.join(args.model_dir, 'best_model.pt'))
```

`VCG_CIR` class — `model.py:147-152`; its image backbone wraps CLIP itself,
`model.py:40-41`: `self.clip, _ = clip.load(img_encoder, device='cuda', jit=False)`.

Released-checkpoint loading. The **released artifact is a whole-module zip pickle**, not a
`{'state_dict': ...}` dict: `zipfile` lists **352 members** (`archive/data.pkl`,
`archive/data/<id>…`). The repo's own replay therefore does a bare `torch.load` with **no
`load_state_dict` at all** — `workbench/replay/tgcir.py:208`:

```
 208:     model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
```

plus two torch-1.7 compat shims (`replay/tgcir.py:183-187`, `:209-213`):
`torch.nn.modules.linear._LinearWithBias -> torch.nn.Linear`, and `batch_first=False` restored on
**24** `nn.MultiheadAttention` modules. Confirmed on disk by
`raw/tgcir_ckpt_load_cpu.log`: `load_ok model.VCG_CIR 0.7s`, `n_params 151859225`,
`mha_missing_batch_first 24`.

The upstream `load_state_dict` path exists but is **dead code for this artifact**:
`workbench/third_party/TGCIR/utils.py:125-137`

```
 125: def load_checkpoint(checkpoint, model, optimizer=None):
 ...
 134:     if not os.path.exists(checkpoint):
 135:         raise("File doesn't exist {}".format(checkpoint))
 136:     checkpoint = torch.load(checkpoint)
 137:     model.load_state_dict(checkpoint['state_dict'])
```

Key names: `'state_dict'` and `'optim_dict'`. Strictness: **default `strict=True`** (no `strict=`
argument). It would `KeyError` on this zip pickle (no `'state_dict'` key), and
`grep -rn "load_checkpoint(" workbench/third_party/TGCIR/` returns **only the definition at
`utils.py:125`** — nothing calls it. `--restore_file` is parsed at `train.py:74-76` but is never
referenced again (`train_and_evaluate(model, optimizer, trainset)` at `:172` takes no restore
argument), so the flag is inert. Registry `notes` says the same in words.

Text-correction dictionary — `workbench/third_party/TGCIR/datasets.py:71-79`, applied on the **val**
path at `:131-136`:

```
  71:     def correct_text(self, text, correction_dict):
  72:         trans=str.maketrans({key: ' ' for key in string.punctuation})
  73:         tokens = str(text).lower().translate(trans).strip().split()
  74:         text = " ".join([correction_dict.get(word) if word in correction_dict else word for word in tokens])
  75: 
  76:         return text
  77: 
  78:     def concat_text(self, captions, correction_dict):
  79:         text = "{} and {}".format(self.correct_text(captions[0], correction_dict), self.correct_text(captions[1], correction_dict))
```

```
 131:         with open(os.path.join(self.caption_dir, 'correction_dict_{}.json'.format(name)), 'r') as f:
 132:             correction_dict = json.load(f)
 ...
 136:             mod_str = self.concat_text(caption['captions'], correction_dict)
```

So: punctuation → space, lowercase, tokenise, per-token dictionary substitution, then the two
captions joined by the literal `" and "`.

Query embedding from (image, text) — `model.py:201-205`:

```
 201:     def extract_retrieval_compose(self, ref, mod):
 202:         s_fuse_local, _, _, _, _ = self.s_compose_feature(ref, mod)
 203:         s_fuse_local = F.normalize(torch.mean(s_fuse_local, dim=1), p=2, dim=-1)
 204: 
 205:         return s_fuse_local
```

with `s_compose_feature` at `:189-199` (mask = `s_remain_map([ref_token, mod_token])`,
`replace_mask = 1 - remain_mask`, `s_fuse_local = remain_mask*ref_token + replace_mask*mod_token`),
and the target side at `:207-210` (`extract_retrieval_target`). Text enters as a **Python list of
strings** (`clip.tokenize(txt)`, `model.py:120`), not a tensor.

**Is a loader trivially assemblable?** **Yes.** The released artifact *is* the model object, so the
loader is three lines: `sys.path.insert(0, <TGCIR>)`, install the `_LinearWithBias` alias,
`torch.load(path, map_location="cpu", weights_only=False)` — then `batch_first=False` on the 24 MHA
modules, `model.eval()`, and `clip.load("ViT-B/16", jit=False)` for the preprocess transform. No
key renaming, no `state_dict` surgery, no shape reconciliation. `workbench/replay/tgcir.py` already
implements exactly this and it has produced numbers (§8).

---

## 8. Native reference numbers on disk

### CLVC-Net — two files, and they DISAGREE slightly

**A.** `workbench/docs/reviews/20261111-master-audit/raw/clvc_run3cat_gpu.json`
(`replay_type: SOURCE_FAITHFUL_WORKBENCH_REPLAY`, `recorded_at: 2026-11-10T13:31:28.628383+00:00`,
`runtime_label: NATIVE_CUDA`, `device: cuda`, `adaptation: null`):

| category | r1 | r10 | r50 |
|---|---|---|---|
| dress | 6.841844323252355 | 29.449677739216657 | 56.767476450173525 |
| shirt | 6.4769381746810595 | 27.77232580961727 | 55.250245338567225 |
| toptee | 7.8531361550229475 | 32.840387557368686 | 63.99796022437532 |

`aggregate`: `macro_r10 30.02079703540087`, `macro_r50 58.67189400437203`,
`macro_mean 44.34634551988645`, `complete true`.

**B.** `workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/clvc_run1.json`
(`checkpoint_id` values are per-category here: `author_drive_dress` / `_shirt` / `_toptee`):

| category | r1 | r10 | r50 |
|---|---|---|---|
| dress | 6.7922657411998015 | 29.350520575111553 | 56.81705503222608 |
| shirt | 6.427870461236506 | 27.77232580961727 | 55.15210991167812 |
| toptee | 7.8531361550229475 | 32.78939316675166 | 63.895971443141256 |

`aggregate`: `macro_r10 29.970746517160162`, `macro_r50 58.62171212901515`,
`macro_mean 44.29622932308766`, `complete true`.

**They are not identical** — 7 of 9 per-category cells differ in the 2nd-3rd decimal
(e.g. dress r10 `29.4497` vs `29.3505`; toptee r50 `63.9980` vs `63.8960`), and the aggregates
differ by 0.05 R@10 / 0.05 R@50 / 0.05 mean. Both are labelled `NATIVE_CUDA` on the same six
checkpoints and the same protocol. `raw/clvc_run3cat_gpu_run1.json` and `_run2.json` are byte-equal
to **A** (same `recorded_at`, same numbers), so A is stable across two runs; B is the outlier. A
later comparison must therefore pick one target explicitly — **use A** (the r16-revised raw file,
whose `.pre-r16` sibling differs only by the added `replay_type` key).

`clvc_run3cat_gpu.json` vs `.pre-r16` diff (only metadata): `+ "replay_type":
"SOURCE_FAITHFUL_WORKBENCH_REPLAY"`.

### TG-CIR — native numbers DO exist

**Not absent.** Two files, mutually consistent:

**A.** `workbench/docs/reviews/20261111-master-audit/raw/tgcir_run3cat_gpu.json`
(`SOURCE_FAITHFUL_WORKBENCH_REPLAY`, `recorded_at: 2026-11-10T13:24:47.518951+00:00`,
`NATIVE_CUDA`, `img_encoder ViT-B/16`, `evaluation_path WORKBENCH_ADDED_EVAL_PATH`,
`checkpoint_sha256 c37768f374c75beb…`):

| category | r1 | r10 | r50 |
|---|---|---|---|
| dress | 14.328210213187903 | 43.38125929598414 | 68.368864650471 |
| shirt | 22.42394504416094 | 51.47203140333661 | 69.8233562315996 |
| toptee | 22.59051504334523 | 54.30902600713922 | 75.11473737888832 |

`aggregate`: `macro_r10 49.720772235486656`, `macro_r50 71.10231942031965`,
`macro_mean 60.411545827903154`, `complete true`.

**B.** `workbench/artifacts/checkpoint_selection/20261106T002617+0700/measurements/tgcir_run1.json`:
identical to A except **one cell** — dress `r1 14.377788795240455` (vs `14.328210213187903`).
All r10/r50 values and the whole aggregate are identical (`aggregate` dicts compare equal).

Corroborating log: `.../logs/tgcir_run2.log` ends with the same three per-category JSON lines plus
`aggregate {"macro_r10": 49.720772235486656, "macro_r50": 71.10231942031965, "macro_mean":
60.411545827903154, "complete": true}` and `EXIT=0`.

Registry paper targets for comparison: CLVC `reported_r10 30.7 / reported_r50 58.41 /
reported_mean 44.56`; TG-CIR `reported_r10 51.32 / reported_r50 73.09 / reported_mean null`.

---

## 9. Dataset roots

Command:

```
cd /home/heheboiz/data/cir && PYTHONPATH=/home/heheboiz/data/cir .venv/bin/python /tmp/audit85/ds.py
```

Verbatim output:

```
FASHIONIQ_ROOT = /home/heheboiz/data/cir/data/FashionIQ
exists: True is_dir: True
top-level entries: ['captions', 'image_splits', 'images', 'qwen_captions', 'resized_image', 'resized_images']
[dress]  cap.dress.val.json EXISTS queries=2017  split.dress.val.json EXISTS entries=3817  correction_dict_dress.json EXISTS keys=363  resized_image/dress/ EXISTS files=3817 jpg=3817
[shirt]  cap.shirt.val.json EXISTS queries=2038  split.shirt.val.json EXISTS entries=6346  correction_dict_shirt.json EXISTS keys=360  resized_image/shirt/ EXISTS files=6346 jpg=6346
[toptee]  cap.toptee.val.json EXISTS queries=1961  split.toptee.val.json EXISTS entries=5373  correction_dict_toptee.json EXISTS keys=391  resized_image/toptee/ EXISTS files=5373 jpg=5373
```

`FASHIONIQ_ROOT` is resolved through `operator_config.resolve_config()`, which reads
`workbench/config/workbench.env:3` (`FASHIONIQ_ROOT=/home/heheboiz/data/cir/data/FashionIQ`).

All four required path shapes exist for all three categories. Counts: **2017 / 2038 / 1961** val
queries (total 6016) matching the query counts in the §8 runs; **3817 / 6346 / 5373** `.jpg` files in
`resized_image/<category>` (all files in those directories are `.jpg`, no strays). Correction
dictionaries have **363 / 360 / 391** entries.

Caveat on the split counts: `split.<cat>.val.json` has more entries than the caption file
(3817 vs 2017, etc.) because it is the **pair-union gallery list**, not the query list — consistent
with `raw/tgcir_forward_gpu.log` (`n_queries 2017 n_pair_union_gallery 2628` for dress, i.e. the
gallery actually used is deduplicated below the 3817 raw split entries).

---

## 10. Incidental finding (blocks repo tooling, not the audit)

`workbench/config/workbench.env:20` sets `WORKBENCH_RUNTIME_ADAPTATION=sm_120`. That name is
permitted only if `workbench.backend.model_settings._declared_names()` can import
`workbench.backend.runtime.ADAPTATION_ENV`; `runtime.py:4` does
`from packaging.requirements import ...`. On an interpreter without `packaging` the import fails,
the `try/except` at `model_settings.py:36-41` degrades to "not declared", and every consumer of
`resolve_config()` then dies:

```
ValueError: unknown setting /home/heheboiz/data/cir/workbench/config/workbench.env:20: WORKBENCH_RUNTIME_ADAPTATION
```

Measured: `python3` (3.12.3, no `packaging`) → raises, `declared_setting_names()` excludes the name;
`.venv/bin/python` (3.12.11, `packaging` present) → resolves fine, name included; `clvc39-la` /
`tgcir39-la` (3.9.25, `packaging` 26.3) → resolve fine. So the failure is a property of the
interpreter, not of the file. All §3/§9 numbers were taken with the `.venv` interpreter for this
reason.

---

## Trạng thái

1. **OK** — both rows dumped as JSON; `clvc_net` = `independent` policy with 6-member whole-model bundle (`checkpoint_base_relative ../../checkpoints/clvc`), `tgcir` = `shared` with one 607 928 637-byte `Fashioniq.pt`; both `command_status: COMMAND_BLOCKED_SOURCE_ENTRYPOINT`.
2. **OK** — 9 models declare `adapted_runtime`, all at `environment.adapted_runtime`; the block is a full python/torch/torchvision/cuda pin replacement + `packages` + `pip_index_url`, read by `runtime.py`, `environment_planner.py`, `composition/runner.py`, `replay/limn.py` and 3 test modules. **Neither target model has one.**
3. **OK** — via `registry.checkpoint_path()`: 7/7 files resolve and 7/7 SHA-256 MATCH; CLVC declares no `expected_size_bytes` (sizes 217 168 235 / 176 437 419 / 218 603 243 / 177 872 427 / 218 308 843 / 177 578 027 recorded only), TG-CIR size matches exactly.
4. **OK** — both envs CPython 3.9.25 / torch 2.7.1+cu128 / torchvision 0.22.1+cu128; `import clip` FAILS in `clvc39-la`, succeeds in `tgcir39-la`; `clip.load("ViT-B/16","cpu",jit=False)` SUCCEEDS with zero download from `~/.cache/clip/ViT-B-16.pt` (sha256 `5806e77c…`), and `HF_HOME` is unset **and irrelevant** — this CLIP loader never reads it.
5. **OK** — `torch.load(weights_only=False, map_location="cpu")` SUCCEEDS, class `compose_local`, 415 `state_dict()` entries / 54 170 563 params, no exception; all six CLVC pickles load. `weights_only=True` (torch 2.7 default) fails with `UnpicklingError`, so the flag is mandatory.
6. **OK** — `train.py:156-157` builds `compose_local`/`compose_global` per category and `train.py:274-275` does whole-module `torch.save`; `compose_img_text` returns `(local_out, local_feature, attn)` / `(global_feature, attn)` and `extract_img_feature` returns `(spatial_map, pooled_1024d)`; the scored 2048-d row is the concat of both half-normalised halves at `test.py:100`.
7. **OK** — released artifact is a whole-module zip pickle (352 members) loaded by bare `torch.load(weights_only=False)` at `replay/tgcir.py:208` plus a `_LinearWithBias` alias and `batch_first=False` on 24 MHA modules; upstream `load_checkpoint` (`utils.py:125-137`, `strict=True`, key `'state_dict'`) is **never called** and `--restore_file` is inert; a loader is trivially assemblable.
8. **OK** — CLVC native numbers exist twice and **disagree by ~0.05** (`raw/clvc_run3cat_gpu.json` macro R@10 30.0208 / R@50 58.6719 / mean 44.3463 vs `measurements/clvc_run1.json` 29.9707 / 58.6217 / 44.2962); TG-CIR native numbers **do exist** (`raw/tgcir_run3cat_gpu.json` macro R@10 49.7208 / R@50 71.1023 / mean 60.4115), differing from `measurements/tgcir_run1.json` in one cell only (dress r1).
9. **OK** — `FASHIONIQ_ROOT=/home/heheboiz/data/cir/data/FashionIQ`; all cap/split/correction-dict/`resized_image` paths exist for dress/shirt/toptee with 2017/2038/1961 val queries and 3817/6346/5373 `.jpg` images.

---

## 11. adapted_runtime shape

Read-only. The registry was **not** edited; every simulation below mutated a deep-copied model dict
in memory only.

### 11.1 Two existing blocks, verbatim (raw YAML, comments included)

Chosen to cover both requested shapes: **`csmcir`** (`models.yaml:51-63`) is the richest —
`target_compute_capability` + a `packages` list + `pip_index_url`; **`limn`**
(`models.yaml:1500-1520`) is the same family with a single-package list and the longest explanatory
comments, showing what an author actually wrote next to the pins.

`csmcir` — `workbench/registry/models.yaml:51-63`:

```yaml
      adapted_runtime:
        target_compute_capability: sm_120
        # The selected cu128 pair supports Python 3.9 and 3.10;
        # Python 3.10 matches the verified cp310 stack in envs/tme-adapted and models.yaml:1305.
        python: '3.10'
        pytorch: '==2.7.1'
        torchvision: '==0.22.1'
        cuda: '12.8'
        # CSMCIR/requirements.txt:9 pins transformers==4.36.2, required by vendored
        # src/lavis/models/blip2_models/Qformer.py:41 for apply_chunking_to_forward. timm==0.9.12 (:5)
        # and fairscale==0.4.13 (:10) match pinned source; numpy<2 preserves compatibility with the pinned model dependencies.
        packages: ['transformers==4.36.2', 'timm==0.9.12', 'fairscale==0.4.13', 'numpy<2']
        pip_index_url: https://download.pytorch.org/whl/cu128
```

`limn` — `workbench/registry/models.yaml:1500-1520`:

```yaml
      adapted_runtime:
        target_compute_capability: sm_120
        python: '3.10'
        pytorch: '==2.7.1'
        torchvision: '==0.22.1'
        cuda: '12.8'
        # open_clip 2.20.0 is the author's own verbatim pin: `LIMN/requirements.txt:37` declares
        # `open-clip-torch==2.20.0`, and `environment.packages` above already carries it. The earlier
        # `>=2.24,<2.27` range here contradicted that pin for no stated reason, and the only runtime
        # actually built and measured for this model (`workbench/envs/limn-adapted`) is 2.20.0, which
        # is what produced the recorded three-category result.
        #
        # Measured, so the record is not overstated: both 2.20.0 and 2.24.0 unpickle these bundles
        # (`torch.load(weights_only=False)` returns `model.LIMN`, and the pickled
        # `VisionTransformer` carries `global_average_pool=True` / no `pool_type` under BOTH). The
        # version difference therefore does not show up at load time; this pin is justified by the
        # author's pin and by provenance of the measured environment, not by a load-time failure.
        packages: ['open_clip_torch==2.20.0']
        # Keep the declared CUDA build: the adapted profile must
        # install from the CUDA 12.8 wheel index explicitly.
        pip_index_url: https://download.pytorch.org/whl/cu128
```

Indentation is 6 spaces for `adapted_runtime:` itself — it is a child of `environment:`
(4 spaces) inside the `models:` list item, i.e. `environment.adapted_runtime`. `tme`
(`models.yaml:1608-1621`) is the only one that uses **ranges** instead of exact pins
(`pytorch: '>=2.7.1,<2.9'`) and the only one that **omits `python`** entirely.

### 11.2 Every code path in `workbench/backend/runtime.py` that reads it

Grep command and result (13 hits):

```
cd /home/heheboiz/data/cir && grep -n "adapted_runtime\|target_compute_capability\|ADAPTATION_ENV" workbench/backend/runtime.py
```

```
69:    #: `adapted_runtime.target_compute_capability` when an operator-selected adaptation applied.
1005:#: Operator-selected runtime adaptation. Set to the registry's `adapted_runtime`
1006:#: `target_compute_capability` (e.g. ``sm_120_blackwell``) to verify a model against its declared
1009:ADAPTATION_ENV = "WORKBENCH_RUNTIME_ADAPTATION"
1015:    registry declares. Any other ADAPTATION_ENV value is refused rather than silently ignored —
1027:    requested = (_setting(ADAPTATION_ENV) or "").strip()
1030:    declared = (model.get("environment") or {}).get("adapted_runtime") or {}
1031:    target = str(declared.get("target_compute_capability") or "").strip()
1037:            f"unknown {ADAPTATION_ENV}={requested!r} for model "
1048:    registry declares a matching `adapted_runtime.target_compute_capability`; any other model is
1055:    adapted_runtime the selection fails to match is still an operator error and refuses loudly.
1781:    report.runtime_adaptation = str(adaptation.get("target_compute_capability")) if adaptation else None
1889:        "adapted_runtime": runtime_adaptation(model, undeclared_is_error=False),
```

There are exactly **four executable reads** (the rest are comments/docstrings). Deciding lines:

**(a) The selector — `runtime_adaptation`, `:1011-1040`:**

```
1027:     requested = (_setting(ADAPTATION_ENV) or "").strip()
1028:     if not requested:
1029:         return None
1030:     declared = (model.get("environment") or {}).get("adapted_runtime") or {}
1031:     target = str(declared.get("target_compute_capability") or "").strip()
1032:     if requested != target:
1033:         if not target and not undeclared_is_error:
1034:             return None
1035:         raise WorkbenchError(
1036:             "UNKNOWN_RUNTIME_ADAPTATION",
1037:             f"unknown {ADAPTATION_ENV}={requested!r} for model "
1038:             f"{model.get('model_id', '<unnamed>')!r}: declared adaptation is {target or 'none'}",
1039:             {"model_id": model.get("model_id"), "requested": requested, "declared": target or None})
1040:     return declared
```

**(b) The resolved contract — `resolved_runtime_environment`, `:1042-1082`:**

```
1057:     env = dict(model.get("environment") or {})
1058:     adapted = runtime_adaptation(model, undeclared_is_error=False)
1059:     if not adapted:
1060:         return env
1061:     resolved = {**env}
1062:     for source, target in (("pytorch", "pytorch"), ("torchvision", "torchvision"),
1063:                            ("cuda", "cuda"), ("python", "python")):
1064:         if source in adapted:
1065:             resolved[target] = adapted[source]
1066:     declared = list(resolved.get("packages") or [])
...
1079:     for key in ("pip_index_url",):
1080:         if adapted.get(key):
1081:             resolved[key] = adapted[key]
1082:     return resolved
```

**(c) The dependency set the probe checks — `_adapted_packages`, `:1089-1103`:**

```
1092:     if "pytorch" in adapted:
1093:         required["torch"] = declared_requirement("torch", adapted.get("pytorch"))
1094:     if "torchvision" in adapted:
1095:         required["torchvision"] = declared_requirement("torchvision", adapted.get("torchvision"))
1096:     for pkg in adapted.get("packages") or []:
...
1102:         required.setdefault(name, constraint)
```

Selected at `:1687-1688`:

```
1687:     adaptation = runtime_adaptation(model, undeclared_is_error=False)
1688:     required = _adapted_packages(model, adaptation) if adaptation else _required_packages(model)
```

**(d) Provenance — `:1781`** (`report.runtime_adaptation`) and **`:1889`** (the report dict's
`"adapted_runtime"` key, via `undeclared_is_error=False`).

### 11.3 Key-by-key: required vs optional, and the tier that results

| key | required? | read by | effect if absent |
|---|---|---|---|
| `target_compute_capability` | **REQUIRED** — it *is* the selector | `:1031`, compared to `ADAPTATION_ENV` at `:1032` | `target = ""` → with `undeclared_is_error=True` **raises** `UNKNOWN_RUNTIME_ADAPTATION`; with `False` returns `None` → adaptation silently not applied (`adapt=None`) |
| `python` | **REQUIRED in practice** for `clvc_net` | `:1062-1065` then `:1664-1665` | absent → the *historical* python pin survives; `clvc_net` then fails `expected Python 3.7.6, found 3.9.25` → `BLOCKED/PYTHON_VERSION_MISMATCH`. For `tgcir` the historical pin is `UNKNOWN`, so absent is harmless |
| `pytorch` | optional, but omitting it leaves the historical `==1.6.0` / `==1.7.0` in force | `:1092-1093` | historical pin → `BLOCKED/DEPENDENCY_VERSION_MISMATCH` |
| `torchvision` | optional, same trap | `:1094-1095` | historical `UNKNOWN` (`clvc`) / `==0.8.0` (`tgcir`) → the latter blocks |
| `cuda` | **optional and NOT verified by `verify_environment`** | `:1062-1065` copies it into the contract; `_requires_gpu` at `:1155-1165` reads it to decide pending-vs-ready | absent → the historical `cuda` value survives and still forces `GPU_VERIFICATION_PENDING`. Changing it to a bogus `11.1` changes **nothing** in the tier |
| `packages` | optional | `:1096-1102` | absent → historical `packages` list used |
| `pip_index_url` | optional, **never verified** | `:1079-1081` | no tier effect at all; it only steers the installer |

Tier outcomes, all measured (§11.4):

- **match** → `runtime_adaptation` returns the block, `report.runtime_adaptation == "sm_120"`, and
  with every pin satisfied the tier goes `PYTHON_VERSION_VERIFIED` → `DEPENDENCIES_VERIFIED` →
  `GPU_VERIFICATION_PENDING` (a **READY_TIERS** member at `:1867`; only `verify_gpu_runtime()` can
  promote it to `GPU_RUNTIME_READY`).
- **mismatch** (`target` declared but ≠ `ADAPTATION_ENV`) → **raises** `WorkbenchError
  UNKNOWN_RUNTIME_ADAPTATION` from `:1035-1039` on any `undeclared_is_error=True` call
  (`resolved_runtime_environment` catches it at `:1660-1663` and falls back to the historical env).
- **not declared** → `None`; historical contract; for these two models that means
  `BLOCKED/PYTHON_VERSION_MISMATCH` (clvc) and `BLOCKED/DEPENDENCY_VERSION_MISMATCH` (tgcir).

### 11.4 Measured: what each block actually does

Command (in-memory only; registry untouched):

```
cd /home/heheboiz/data/cir && PYTHONPATH=/home/heheboiz/data/cir \
  WORKBENCH_CLVC_PYTHON=.../envs/clvc39-la/bin/python \
  WORKBENCH_TGCIR_PYTHON=.../envs/tgcir39-la/bin/python \
  WORKBENCH_RUNTIME_ADAPTATION=sm_120 \
  .venv/bin/python /tmp/audit85/sim4.py
```

Baseline first — today's registry, no `adapted_runtime`, interpreters set, advisory probe:

```
clvc_net tier= BLOCKED status= PYTHON_VERSION_MISMATCH
  reason= expected Python 3.7.6, found 3.9.25
tgcir    tier= BLOCKED status= DEPENDENCY_VERSION_MISMATCH
  reason= torch: expected ==1.7.0, found 2.7.1+cu128
```

With a candidate block injected into `environment.adapted_runtime`:

```
tgcir     | no python key (historical UNKNOWN)       | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
tgcir     | no python, no packages                   | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
tgcir     | python 3.9, packages omitted             | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
clvc_net  | python 3.9, packages omitted             | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
clvc_net  | python 3.9, packages [torchvision]       | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
clvc_net  | no cuda key in block                     | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
clvc_net  | cuda 11.1 (bogus) in block               | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
clvc_net  | pytorch range >=2.7.1,<2.9               | GPU_VERIFICATION_PENDING | GPU_VERIFICATION_PENDING
clvc_net  | tcc sm_90 vs env sm_120                  | RAISED WorkbenchError | UNKNOWN_RUNTIME_ADAPTATION
clvc_net  | python 3.10 in block (wrong)             | BLOCKED | PYTHON_VERSION_MISMATCH
clvc_net  | torch ==2.6.0 (wrong)                    | BLOCKED | DEPENDENCY_VERSION_MISMATCH
clvc_net  | tcc only, no pins at all                 | BLOCKED | PYTHON_VERSION_MISMATCH
clvc_net  | min (no python)                          | BLOCKED | PYTHON_VERSION_MISMATCH
```

Two facts this pins down that are not obvious from the code alone:

1. **`python: '3.9'` is the single load-bearing extra field for `clvc_net`** — dropping it reverts to
   the historical `3.7.6` and the block buys nothing.
2. **`cuda` and `pip_index_url` are decorative as far as `verify_environment` is concerned.** A
   deliberately wrong `cuda: '11.1'` still reaches `GPU_VERIFICATION_PENDING`, because
   `verify_environment` never compares the CUDA version — only `verify_gpu_runtime()` does, from the
   environment's own probe. `pip_index_url` is never read outside `resolved_runtime_environment`.

Measured environment facts this rests on (from §4, plus a distribution inventory taken this turn):

```
clvc39-la : torch 2.7.1+cu128, torchvision 0.22.1+cu128, numpy 1.26.4, tqdm 4.70.1,
            clip ABSENT, pandas ABSENT
tgcir39-la: torch 2.7.1+cu128, torchvision 0.22.1+cu128, numpy 1.22.3, pandas 1.4.2,
            Pillow 11.3.0, clip 1.0, ftfy 6.3.1, regex 2026.1.15, tqdm 4.70.1
```

`==2.7.1` does match `2.7.1+cu128` and `==0.22.1` does match `0.22.1+cu128`
(`SpecifierSet('==2.7.1').contains(Version('2.7.1+cu128'))` → `True`), so exact pins are safe here.

### 11.5 Minimal block per model (≤6 lines)

```
clvc_net: adapted_runtime: {target_compute_capability: sm_120, python: '3.9', pytorch: '==2.7.1', torchvision: '==0.22.1', cuda: '12.8'}
tgcir:    adapted_runtime: {target_compute_capability: sm_120, python: '3.9', pytorch: '==2.7.1', torchvision: '==0.22.1', cuda: '12.8'}
```

Both reach `GPU_VERIFICATION_PENDING` (a READY tier) with exactly those five keys and **no**
`packages` / `pip_index_url`. Confident: `target_compute_capability: sm_120` (matches the env var and
the RTX 5070 Ti's measured `compute_cap 12.0`), `python: '3.9'`, `pytorch: '==2.7.1'`,
`torchvision: '==0.22.1'` — all four are read from the installed distributions. Not confident / not
measured: the `cuda: '12.8'` value is only decorative to the verifier, and whether the GPU actually
satisfies `sm_120` requires `verify_gpu_runtime(authorized=True)`, which this audit did not run
(no GPU use permitted); note also that `tgcir39-la` carries `clip 1.0` whose source identity is
declared in `environment.packages`, and an adapted block's `packages` list **replaces** the
historical names it repeats — so if `clip` were listed without its `git+…` source it would drop the
source check, which is why leaving `packages` out entirely is the safer minimal form.
