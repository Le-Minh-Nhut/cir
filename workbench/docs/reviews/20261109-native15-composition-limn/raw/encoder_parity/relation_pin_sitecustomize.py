"""Parity control: pin ENCODER's uncheckpointed `BindingDecoder.learnable_relation`.

`model_try2.py:93` does

    self.learnable_relation = nn.Parameter(torch.randn(N_p, dim, dtype=torch.float32)).cuda()

`Parameter.cuda()` returns a plain Tensor, so the value is never registered as a parameter, never
enters `state_dict()`, and is absent from `fashioniq.pt` -- `load_state_dict` reports NO missing key.
Every process that builds the model draws its own random 2x512 relation, and `binding()` feeds it
into a batched matmul against the token matrix. Measured consequence: two processes running
byte-identical code on byte-identical pixels produce `binding_product_cross` values differing by
32.27 in absolute value, and final features differing by 1.6e-2 -- with cuDNN deterministic,
`use_deterministic_algorithms(True)`, a fixed seed, a fixed batch size, and TF32 off.

This hook replaces exactly that one attribute after `model_try2` is imported, with a fixed tensor,
so two processes can be compared on the SAME model. No layer, weight or arithmetic changes; the
value the checkpoint does not ship is supplied. Activate by putting this directory on PYTHONPATH.
"""
import importlib.abc
import importlib.machinery
import os
import sys

_LOG = os.environ.get("ENCODER_PARITY_RELATION_LOG", "")
_patched = False


def _fixed():
    import torch

    # Same shape and numeric range as torch.randn(2, 512).
    return (torch.arange(2 * 512, dtype=torch.float32).reshape(2, 512) / 512.0 - 1.0)


def _patch(module):
    global _patched
    if _patched:
        return
    decoder = getattr(module, "BindingDecoder", None)
    if decoder is None:
        return
    pinned = _fixed()
    original_init = decoder.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        # Replace the unregistered, uncheckpointed draw. `.cuda()` is a no-op here: the attribute is
        # already on the model's device after `original_init` (it calls `.cuda()` itself).
        device = self.learnable_relation.device
        self.learnable_relation = pinned.to(device).clone()
        if _LOG:
            with open(_LOG, "a", encoding="utf-8") as handle:
                handle.write(f"pid={os.getpid()} pinned learnable_relation "
                             f"shape={tuple(pinned.shape)} device={device}\n")

    decoder.__init__ = patched_init
    _patched = True


class _Loader(importlib.abc.Loader):
    def __init__(self, inner):
        self.inner = inner

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.inner.exec_module(module)
        _patch(module)


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != "model_try2" or _patched:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is not None and spec.loader is not None:
            spec.loader = _Loader(spec.loader)
        return spec


if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _Finder())
