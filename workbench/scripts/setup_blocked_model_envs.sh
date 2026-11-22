#!/usr/bin/env bash
# Fresh, non-colliding interpreters for the four blocked models (M-04..M-07).
#
# Why adapted rather than the pinned interpreter: CLVC-Net declares Python 3.7.6/Torch 1.6.0 and
# TG-CIR Torch 1.7.0 (both CUDA 10.2/11.0 era); Air-Know declares 3.8.10/Torch 2.1.0. None of those
# wheels carry sm_120 kernels, and this host's only GPU is an RTX 5070 Ti (compute capability 12.0),
# so a pinned-runtime run cannot execute a single kernel here at all. These envs are therefore an
# ADAPTED RUNTIME: modern Python + the cu128 line, identical to the pairing the registry already
# records for the LAVIS family (torch 2.7.1 / torchvision 0.22.1 / cu128). Nothing about the pinned
# source is modified; the interpreter and CUDA line are.
set -euo pipefail
REPO=/home/heheboiz/data/cir
ENVS="$REPO/workbench/envs"
UV=/home/heheboiz/.local/bin/uv
TORCH_INDEX=https://download.pytorch.org/whl/cu128
DECORD_DIR="$REPO/workbench/artifacts/assets/decord/v0.6.0"
DECORD_SOURCE="$DECORD_DIR/original/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"
DECORD_WHEEL="$DECORD_DIR/dist/decord-0.6.0-py3-none-manylinux2010_x86_64.whl"

# `uv venv` seeds no pip; every install goes through `uv pip` against that interpreter, so the
# record of what landed is `uv pip freeze` on the same path.
uip() { "$UV" pip install -q --python "$DIR/bin/python" "$@"; }
common() {
  local py="$1" dir="$2" torch_req="$3"
  "$UV" venv --python "$py" --allow-existing "$dir" >/dev/null
  DIR="$dir"
  uip --index-url "$TORCH_INDEX" $torch_req
}

case "${1:?target name: airknow|clvc|tgcir|pair|dcnet}" in
  clvc)
    # Pinned README: Python 3.7.6 / Torch 1.6.0. Adapted: 3.9 + cu128 (sm_120). The evaluator itself
    # imports only torch, torchvision, numpy and PIL; `text_model` uses a pure-python vocab.
    DIR="$ENVS/clvc39-la"
    common 3.9 "$DIR" "torch==2.7.1 torchvision==0.22.1"
    uip "numpy<2" Pillow tqdm
    ;;
  tgcir)
    # Pinned README: Torch 1.7.0 / torchvision 0.8.0 / clip==0.2.0 / numpy 1.22.3 / pandas 1.4.2.
    # `clip==0.2.0` on PyPI is an unrelated clipboard CLI (registry note), so the OpenAI module is
    # installed from the revision the registry already pins.
    DIR="$ENVS/tgcir39-la"
    common 3.9 "$DIR" "torch==2.7.1 torchvision==0.22.1"
    uip "numpy==1.22.3" "pandas==1.4.2" Pillow tqdm ftfy regex \
      "clip @ git+https://github.com/openai/CLIP.git@a1d071733d7111c9c014f024669f959182114e33"
    ;;
  airknow)
    # Pinned README: Python 3.8.10 / Torch 2.1.0 / torchvision 0.16.0 / CUDA 12.6. Adapted: 3.10 +
    # cu128. Package set = the registry's own declared list for this model.
    DIR="$ENVS/airknow310-la"
    common 3.10 "$DIR" "torch==2.7.1 torchvision==0.22.1"
    uip \
      "transformers==4.25.0" "timm==0.9.16" "fairscale==0.4.13" "omegaconf==2.3.0" \
      "iopath==0.1.10" "peft==0.6.2" "diffusers==0.19.3" "huggingface-hub==0.19.4" \
      "accelerate==0.25.0" "opencv-python-headless==4.5.5.64" "scikit-learn==1.3.2" \
      "pandas==2.0.3" "numpy==1.26.4" Pillow pyyaml regex requests tqdm \
      "einops==0.7.0" "ftfy==6.3.1" "webdataset==0.2.86"
    "$REPO/workbench/envs/ctl312h/bin/python" "$REPO/workbench/scripts/repair_decord_wheel_metadata.py" "$DECORD_SOURCE" "$DECORD_WHEEL"
    uip --reinstall --no-deps "$DECORD_WHEEL"
    ;;
  pair)
    # PAIR declares Python 3.8.10 / Torch 2.0.0 + open_clip, but that pinned pairing cannot be
    # installed here: torch 2.7.1 declares `Requires-Python >=3.9.0` (read from the installed
    # 2.7.1+cu128 metadata, not from the README), so a 3.8 interpreter cannot resolve the wheel at
    # all. Adapted: 3.9 + cu128, the same interpreter line `clvc39-la` and `tgcir39-la` use.
    #
    # The missing `model_vssp` / `MHTransformer_dis` modules are NOT a Python-version problem: they
    # are absent from the pinned tree at every sys.path root the replay adds
    # (`PAIR/`, `PAIR/data`, `PAIR/models`, `PAIR/utils`), so no interpreter version can supply
    # them. `workbench.replay.pair_absence` resolves them per root and reports which of the two it
    # is — absent module vs. module present but missing a class. This env exists to measure what the
    # checkpoint structurally demands, and to have a runtime that matches the pins it can install.
    DIR="$ENVS/pair39-la"
    common 3.9 "$DIR" "torch==2.7.1 torchvision==0.22.1"
    uip "open_clip_torch==2.24.0" "numpy<2" Pillow tqdm einops timm
    # The three imports the replay itself performs (`pair.py:133`), so a broken pin fails here
    # rather than at the first evaluation attempt. `open_clip` exports no `__version__`.
    "$DIR/bin/python" -c 'import sys, torch, torchvision, open_clip; \
import importlib.metadata as md; \
print("smoke", sys.version.split()[0], torch.__version__, torchvision.__version__, \
md.version("open_clip_torch"), torch.version.cuda)'
    ;;
  dcnet)
    DIR="$ENVS/dcnet310-sm120"
    common 3.10 "$DIR" "torch==2.7.1 torchvision==0.22.1"
    uip "numpy==1.26.4" "Pillow==12.3.0" "einops==0.7.0" "nltk==3.5"
    NLTK_DATA="$REPO/workbench/artifacts/assets/nltk_data"
    mkdir -p "$NLTK_DATA/tokenizers"
    curl -fsSL "https://raw.githubusercontent.com/nltk/nltk_data/550b6625bcef1f2abff2ff770a5a0d272c9c6b2a/packages/tokenizers/punkt.zip" -o "$NLTK_DATA/tokenizers/punkt.zip"
    "$REPO/workbench/envs/ctl312h/bin/python" - "$NLTK_DATA" <<'PY'
import hashlib
import sys
from pathlib import Path
from zipfile import ZipFile
root = Path(sys.argv[1])
archive_path = root / "tokenizers/punkt.zip"
digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
assert digest == "51c3078994aeaf650bfc8e028be4fb42b4a0d177d41c012b6a983979653660ec", digest
with ZipFile(archive_path) as archive:
    archive.extractall(root / "tokenizers")
PY
    NLTK_DATA="$NLTK_DATA" "$DIR/bin/python" -c 'import nltk; from nltk.tokenize import word_tokenize; assert word_tokenize("Two red shirts, with stripes!") == ["Two", "red", "shirts", ",", "with", "stripes", "!"]; print(nltk.__version__, nltk.data.find("tokenizers/punkt/english.pickle"))'
    ;;
  *) echo "unknown target: $1" >&2; exit 2 ;;
esac
echo "OK $1 -> $DIR"
"$DIR/bin/python" -c 'import sys, torch; print(sys.executable, sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.is_available())'
