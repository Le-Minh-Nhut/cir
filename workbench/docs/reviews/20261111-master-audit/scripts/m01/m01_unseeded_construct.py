"""CPU-only proof that `BindingDecoder.learnable_relation` is drawn unseeded.

Two clean processes (this script is run twice) construct the pinned ENCODER exactly as the
composition worker does and print the relation digest. `.cuda()` is neutralised so the draw —
which happens in `model_try2.py:93` before any device transfer matters — can be observed on CPU.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
SRC = REPO / "workbench/third_party/ENCODER"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))
os.chdir(SRC)

import torch

torch.Tensor.cuda = lambda self, *a, **k: self            # type: ignore[method-assign]
torch.nn.Module.cuda = lambda self, *a, **k: self         # type: ignore[method-assign]

import model_try2  # noqa: E402


def digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


def build(seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    return model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                              wc=2, N_p=2, weighted=True).backbone.binding_decoder.learnable_relation


def main() -> int:
    first = build()
    second = build()
    seeded_a, seeded_b = build(20261111), build(20261111)
    reported = {
        "pid": os.getpid(),
        "relation_type": type(first).__name__,
        "relation_shape": list(first.shape),
        "digest_first": digest(first),
        "digest_second_same_process": digest(second),
        "unseeded_same_process_max_abs": float((first - second).abs().max()),
        "digest_seeded": digest(seeded_a),
        "seeded_same_process_max_abs": float((seeded_a - seeded_b).abs().max()),
    }
    print("M01:" + json.dumps(reported, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
