"""Restore run `ce89a9a2…`'s construction-state record from the pin, verified by digest.

The pinned rerun wrote its record under `work/encoder/encoder/construction_state/`, which the
runner's keep-features cleanup deletes (`runner.py:2263-2271`) -- a real defect, fixed by moving the
record to `<run>/construction_state/`. The recorded digest was captured before deletion
(`20db0c27…ce92`); this re-draws the tensor from the same pinned seed on CPU, refuses if the digest
differs, and writes it with `source` naming the re-derivation. No GPU and no scoring: the tensor is a
pure function of (seed, torch RNG).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
SRC = REPO / "workbench/third_party/ENCODER"
RUN = REPO / "workbench/artifacts/composition/runs/ce89a9a27c4cceadeaaf30b0"
SEED = 20261111
EXPECTED = "20db0c27c979e9a0938c71af5142458dfa2070468d0ad3fcb6f24134c2f9ce92"

sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SRC))
import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.Tensor.cuda = lambda self, *a, **k: self            # type: ignore[method-assign]
torch.nn.Module.cuda = lambda self, *a, **k: self         # type: ignore[method-assign]
import model_try2  # noqa: E402

torch.manual_seed(SEED)
cwd = Path.cwd()
os.chdir(SRC)
model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                           wc=2, N_p=2, weighted=True)
os.chdir(cwd)
tensor = model.backbone.binding_decoder.learnable_relation.detach().float().cpu().contiguous()
digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
if digest != EXPECTED:
    raise SystemExit(f"REFUSING: re-drawn relation {digest} != recorded {EXPECTED}")

root = RUN / "construction_state"
root.mkdir(parents=True, exist_ok=True)
np.save(root / f"relation-seed{SEED}.npy", tensor.numpy(), allow_pickle=False)
(root / f"relation-seed{SEED}.npy.json").write_text(json.dumps({
    "tensor": "backbone.binding_decoder.learnable_relation",
    "shape": list(tensor.shape), "dtype": str(tensor.dtype), "seed": SEED, "sha256": digest,
    "in_state_dict": False,
    "seed_placement": "torch.manual_seed(seed) immediately before Encoder(...) construction",
    "source": "workbench/third_party/ENCODER/model_try2.py:93 (torch.randn(N_p, dim))",
    "provenance": ("re-derived from the pinned seed after this run's own copy was deleted by the "
                   "runner's keep-features cleanup; digest equals the value the run printed before "
                   "deletion, so the bytes are the ones this run scored with"),
}, indent=1) + "\n", encoding="utf-8")
print("RESTORED", digest, "->", root)
