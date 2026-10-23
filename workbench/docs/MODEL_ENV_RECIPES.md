# Provisioning the iLearn-family environments

The registry declares each model's environment (`environment.python`, `pytorch`, `torchvision`,
`cuda`, `packages`), and `workbench/scripts/manage_environment.py --create` provisions one from a
**source-backed** spec — it refuses to guess, so a checkout without `environment.yml` or
`requirements.txt` cannot be auto-provisioned even when the README documents the versions. This file
records the exact commands that produced working environments, so the next operator does not re-derive
them.

Verified on this host: **RTX 3050 Laptop (sm_86)**, driver 580.178.04, CUDA 12.6 runtime.

## HABIT (Python 3.8.10, torch 2.1.0+cu121)

The declared version is **exact**, not a series: `runtime.python_version_satisfies` treats a
three-part declaration as a precise requirement, and a 3.8.20 interpreter is refused with
`PYTHON_VERSION_MISMATCH`. uv supplies 3.8.10.

```sh
uv venv --python 3.8.10 workbench/envs/habit
uv pip install --python workbench/envs/habit/bin/python \
  --index-url https://download.pytorch.org/whl/cu121 torch==2.1.0 torchvision==0.16.0
# The registry's declared package list, exactly. Note `salesforce-lavis==1.0.2` pins `timm==0.4.12`,
# while the upstream README also lists `timm==0.9.16` -- those cannot coexist. HABIT imports only
# `lavis.models` and `sklearn.cluster`, so the LAVIS-compatible pin is the correct one and the
# README's timm entry is an inconsistency in the source, not a requirement.
uv pip install --python workbench/envs/habit/bin/python \
  "transformers==4.25.0" "timm==0.9.16" "omegaconf==2.3.0" "opencv-python-headless==4.5.5.64" \
  "pandas==2.0.3" "scikit-image==0.21.0" "decord==0.6.0" "einops==0.7.0" "fairscale==0.4.13" \
  ftfy "iopath==0.1.10" "peft==0.6.2" "diffusers==0.19.3" "huggingface-hub==0.19.4" \
  "accelerate==0.25.0" "pycocoevalcap==1.2" pycocotools "webdataset==0.2.86" "scikit-learn==1.3.2"
export WORKBENCH_HABIT_PYTHON="$PWD/workbench/envs/habit/bin/python"
```

Then, with that variable exported, the GPU capability must be recorded once:

```sh
.venv/bin/python -m workbench.scripts.evaluate_models \
  --model habit --checkpoint fiq_n02 --verify-gpu-only --dataset-root "$PWD/data/fashionIQ_dataset"
```

`.venv/bin/python workbench/scripts/doctor.py --scope real --model habit --checkpoint fiq_n02`
reports `READY` when both are done. `transformers==4.25.0` prints a "yanked" warning; it is the
version the registry declares and the one that was validated.

## Running any iLearn-family model: TMPDIR must be real disk

The native constructors bake every category's query tensors at construction time, which is ~12 GB of
pickles — more than one construction can be large. `isolated_cache_root` puts them in `TMPDIR`, so:

* `TMPDIR` must be on **disk**, not the RAM-backed `/tmp` that many hosts default to, and
* the volume needs roughly **12 GB free** for a three-category run.

Running with `TMPDIR=/tmp` on a host whose `/tmp` is a 7.2 GB tmpfs fails inside the native `save_obj`
with `[Errno 28] No space left on device` part-way through the third category.

## Other environments seen on this host

* `workbench/envs/encoder` — Python 3.9, torch 1.13.1+cu117, open_clip 2.24.0; ENCODER's evaluator.
* `workbench/envs/limn-historical` — the interpreter the verified LIMN run used.
* `workbench/artifacts/environments/cir-1bbd9872cf166f0d` — a workbench-managed shared environment
  matching LIMN's declared pins (torch 1.12.1, torchvision 0.13.1, open_clip 2.20.0). Kept: it is a
  legitimate artifact of the environment planner, not the `limn-historical` interpreter that the
  tagged run used.
