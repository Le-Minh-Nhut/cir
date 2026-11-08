"""CPU-only structural fingerprint of one checkpoint file.

Usage: python fingerprint.py <path> [--max-prefix-depth N]
Prints JSON to stdout. Never touches CUDA (map_location='cpu').
"""
from __future__ import annotations

import collections
import json
import sys
import traceback
from pathlib import Path

import torch


def flat_prefixes(keys, depth):
    counts = collections.Counter()
    for key in keys:
        parts = key.split(".")
        counts[".".join(parts[:depth]) + (".*" if len(parts) > depth else "")] += 1
    return dict(counts.most_common(40))


def describe_tensor(value):
    return {"dtype": str(value.dtype), "shape": list(value.shape),
            "numel": int(value.numel()), "requires_grad": bool(value.requires_grad)}


def load(path):
    attempts = []
    for kwargs in ({"weights_only": True}, {"weights_only": False}):
        try:
            return torch.load(path, map_location="cpu", **kwargs), attempts
        except Exception as error:  # noqa: BLE001
            attempts.append({**kwargs, "error": f"{type(error).__name__}: {str(error)[:400]}"})
    raise RuntimeError(json.dumps(attempts))


def walk(obj, depth=0, seen=None):
    """Structural skeleton: type, and for dict-like containers the key list."""
    seen = seen or set()
    if id(obj) in seen:
        return "<cycle>"
    if depth > 3:
        return f"<{type(obj).__name__} depth>"
    seen.add(id(obj))
    if isinstance(obj, torch.Tensor):
        return describe_tensor(obj)
    if isinstance(obj, dict):
        skeleton = {}
        for key in list(obj)[:60]:
            value = obj[key]
            if isinstance(value, (torch.Tensor,)) or depth < 2:
                skeleton[str(key)] = walk(value, depth + 1, seen)
            else:
                skeleton[str(key)] = f"<{type(value).__name__}>"
        if len(obj) > 60:
            skeleton["<truncated>"] = f"{len(obj)} keys total"
        return {"__type__": type(obj).__name__, "keys": skeleton}
    if isinstance(obj, (list, tuple)):
        return {"__type__": type(obj).__name__, "len": len(obj),
                "first": [walk(item, depth + 1, seen) for item in list(obj)[:3]]}
    if isinstance(obj, (str, int, float, bool, type(None))):
        text = repr(obj)
        return text if len(text) <= 300 else text[:300] + "…"
    return f"<{type(obj).__name__}>"


def parameter_report(obj):
    """Parameter count and key prefixes for a state-dict-shaped object."""
    tensors = None
    kind = None
    for label, value in (("self", obj), ("model", None)):
        pass
    if isinstance(obj, torch.nn.Module):
        kind = "nn.Module"
        tensors = dict(obj.named_parameters())
        tensors.update({f"buffer:{k}": v for k, v in obj.named_buffers()})
    elif isinstance(obj, dict):
        # a state dict, or a training dict wrapping one
        for candidate in ("state_dict", "model", "model_state_dict", "model_state", "net",
                          "weights", "module"):
            inner = obj.get(candidate)
            if isinstance(inner, dict) and inner and all(
                    isinstance(v, torch.Tensor) for v in list(inner.values())[:5]):
                kind = f"dict[{candidate!r}] -> state_dict"
                tensors = inner
                break
        if tensors is None and obj and all(isinstance(v, torch.Tensor) for v in list(obj.values())[:5]):
            kind = "state_dict"
            tensors = obj
        if tensors is None:
            return {"kind": "no-parameter-container-found"}
    else:
        return {"kind": type(obj).__name__}
    total = sum(int(v.numel()) for v in tensors.values())
    trainable = sum(int(v.numel()) for v in tensors.values())
    return {
        "kind": kind,
        "tensor_count": len(tensors),
        "total_numel": total,
        "param_bytes_fp32": total * 4,
        "prefix_depth1": flat_prefixes(list(tensors), 1),
        "prefix_depth2": flat_prefixes(list(tensors), 2),
        "prefix_depth3": flat_prefixes(list(tensors), 3),
        "first_keys": list(tensors)[:25],
        "last_keys": list(tensors)[-8:],
        "sample_shapes": {k: describe_tensor(tensors[k]) for k in list(tensors)[:8]},
    }


def main():
    path = Path(sys.argv[1])
    result = {"path": str(path), "size_bytes": path.stat().st_size}
    try:
        obj, attempts = load(path)
        result["load_attempts"] = attempts
    except Exception as error:  # noqa: BLE001
        result["load_error"] = str(error)[:2000]
        result["traceback"] = traceback.format_exc()[-2000:]
        print(json.dumps(result, indent=1, default=str))
        return
    result["top_type"] = type(obj).__name__
    if isinstance(obj, dict):
        result["top_keys"] = [str(k) for k in obj][:60]
        result["top_key_count"] = len(obj)
    result["structure"] = walk(obj)
    result["parameters"] = parameter_report(obj)
    print(json.dumps(result, indent=1, default=str))


main()
