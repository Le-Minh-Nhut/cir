"""Structure + parameter fingerprint of a checkpoint, on CPU, with the pickle's own globals resolved.

The checkpoint bytes name their classes; those names are the architecture fingerprint. This driver
makes the names importable (real source first, inert stubs only for names the public repo does not
ship, each stub recorded in the output so nothing is silently pretended) and then reports, without
running any forward pass:

  * root object type and dict keys
  * module-tree class lineage (an nn.Module pickle says exactly which classes built it)
  * state_dict key count, prefix histogram, total element count, dtype mix
  * the specific shapes that identify an architecture (embedding widths, projection dims, heads)

Usage:
  python fingerprint2.py <path> --out out.json [--syspath DIR ...] [--stub MOD:Class,Class ...]
"""
from __future__ import annotations

import collections
import json
import sys
import traceback
import types

import torch
import torch.nn as nn

ARGS = sys.argv[1:]
PATH = ARGS[0]


def opt(name: str) -> list[str]:
    values = []
    i = 0
    while i < len(ARGS):
        if ARGS[i] == name:
            i += 1
            while i < len(ARGS) and not ARGS[i].startswith("--"):
                values.append(ARGS[i])
                i += 1
        else:
            i += 1
    return values


OUT = opt("--out")[0] if opt("--out") else None
for entry in opt("--syspath"):
    if entry not in sys.path:
        sys.path.insert(0, entry)

STUBBED: dict[str, list[str]] = {}


class Inert(nn.Module):
    """Marker base: records the pickle global, holds restored children/tensors, no behaviour."""

    def __repr__(self) -> str:
        return f"<INERT {type(self).__module__}.{type(self).__name__}>"


ALIASES: list[str] = []


def install_alias(spec: str) -> None:
    """`--alias old.module:Name=new.module:Name` — bind a name a newer torch removed.

    Recorded in the output: an alias changes only which CLASS the pickle binds to, never a tensor,
    and every alias used is reported so the claim can be audited.
    """
    source, _, target = spec.partition("=")
    source_module, _, source_name = source.partition(":")
    target_module, _, target_name = target.partition(":")
    source_mod = __import__(source_module, fromlist=[source_name or "__self__"])
    if target_module and target_name:
        target_value = getattr(__import__(target_module, fromlist=[target_name]), target_name)
    else:
        target_value = getattr(__import__(target_module or source_module,
                                          fromlist=[target_name]), target_name)
    setattr(source_mod, source_name, target_value)
    ALIASES.append(spec)


for spec in opt("--alias"):
    install_alias(spec)

for spec in opt("--stub"):
    module_name, _, class_names = spec.partition(":")
    module = types.ModuleType(module_name)
    module.__dict__["__stub__"] = True
    for class_name in class_names.split(","):
        class_name = class_name.strip()
        if not class_name:
            continue
        setattr(module, class_name,
                type(class_name, (Inert,), {"__module__": module_name}))
        STUBBED.setdefault(module_name, []).append(class_name)
    sys.modules[module_name] = module


def prefix_histogram(keys, depth):
    counts = collections.Counter()
    for key in keys:
        parts = key.split(".")
        counts[".".join(parts[:depth]) + (".*" if len(parts) > depth else "")] += 1
    return dict(counts.most_common(50))


def module_tree(node, path="", depth=0, rows=None):
    rows = rows if rows is not None else []
    if depth > 12:
        return rows
    rows.append({"path": path or "<root>",
                 "class": f"{type(node).__module__}.{type(node).__name__}"})
    for name, child in node._modules.items():
        if child is not None:
            module_tree(child, f"{path}.{name}" if path else name, depth + 1, rows)
    return rows


def scalars_of(node, path="", depth=0, out=None, seen=None):
    out = out if out is not None else {}
    seen = seen if seen is not None else set()
    if id(node) in seen or depth > 6:
        return out
    seen.add(id(node))
    if not isinstance(node, nn.Module):
        return out
    for name, value in node.__dict__.items():
        key = f"{path}.{name}" if path else name
        if isinstance(value, (int, float, bool, str, type(None))) and not name.startswith("training"):
            out[key] = value
        elif isinstance(value, (list, tuple)) and all(
                isinstance(item, (int, float, bool, str)) for item in value):
            out[key] = f"{type(value).__name__}[{len(value)}]={list(value)[:12]}"
    for name, child in node._modules.items():
        if child is not None:
            scalars_of(child, f"{path}.{name}" if path else name, depth + 1, out, seen)
    return out


def find_state_dict(obj, depth=0):
    """Locate the parameter container and report the path that reached it."""
    if isinstance(obj, nn.Module):
        return obj.state_dict(), "<root is nn.Module>"
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, dict) and value and all(
                    isinstance(item, torch.Tensor) for item in list(value.values())[:5]):
                return value, str(key)
            if isinstance(value, nn.Module):
                return value.state_dict(), f"{key} (nn.Module)"
            if isinstance(value, dict) and depth < 2:
                inner, where = find_state_dict(value, depth + 1)
                if inner:
                    return inner, f"{key}.{where}"
    return None, None


report: dict = {"path": PATH}
try:
    obj = torch.load(PATH, map_location="cpu", weights_only=False)
except Exception as error:  # noqa: BLE001
    report["load_error"] = f"{type(error).__name__}: {str(error)[:600]}"
    report["traceback"] = traceback.format_exc()[-1500:]
    obj = None

report["sys_modules_from_stubs"] = STUBBED
report["class_aliases"] = ALIASES
report["resolved_module_origins"] = {}

if obj is not None:
    report["root_class"] = f"{type(obj).__module__}.{type(obj).__name__}"
    if isinstance(obj, dict):
        report["root_keys"] = [str(key) for key in obj]
        report["root_key_types"] = {str(key): f"{type(value).__module__}.{type(value).__name__}"
                                    for key, value in obj.items()}
        report["root_scalars"] = {str(key): value for key, value in obj.items()
                                  if isinstance(value, (int, float, str, bool, type(None)))}
        nested = {str(key): sorted(value)[:40] for key, value in obj.items()
                  if isinstance(value, dict) and not all(isinstance(v, torch.Tensor)
                                                         for v in list(value.values())[:5])}
        report["nested_dict_keys"] = nested
        report["nested_dict_lengths"] = {str(key): len(value) for key, value in obj.items()
                                         if isinstance(value, dict)}
    if isinstance(obj, nn.Module):
        report["module_tree"] = module_tree(obj)
        report["module_scalars"] = scalars_of(obj)

    state_dict, where = find_state_dict(obj)
    if state_dict:
        report["state_dict_at"] = where
        keys = list(state_dict)
        report["tensor_count"] = len(keys)
        total = 0
        dtypes = collections.Counter()
        for value in state_dict.values():
            total += int(value.numel())
            dtypes[str(value.dtype)] += int(value.numel())
        report["total_numel"] = total
        report["dtype_numel"] = dict(dtypes)
        report["prefix1"] = prefix_histogram(keys, 1)
        report["prefix2"] = prefix_histogram(keys, 2)
        report["prefix3"] = prefix_histogram(keys, 3)
        report["first_keys"] = keys[:20]
        report["last_keys"] = keys[-10:]
        report["shapes"] = {key: list(value.shape) for key, value in list(state_dict.items())[:400]}
        interesting = ("class_embedding", "positional_embedding", "text_projection", "proj",
                       "conv1.weight", "logit_scale", "in_proj_weight", "output_layer",
                       "combiner_layer", "fc.weight", "img_fc", "local_fc", "global_fc", "qformer",
                       "hidden_size", "embed_dim")
        report["fingerprint_shapes"] = {key: list(value.shape) for key, value in state_dict.items()
                                        if any(token in key for token in interesting)}
    else:
        report["state_dict"] = "none found"

encoded = json.dumps(report, indent=1, default=str)
if OUT:
    open(OUT, "w").write(encoded)
    print(f"{PATH}: wrote {OUT} ({len(encoded)} bytes) root={report.get('root_class')}")
else:
    print(encoded)
