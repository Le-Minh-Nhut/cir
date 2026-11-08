"""Cheap pickle introspection: which module/class globals does a torch save reference?

Extracts `data.pkl` from the zip container and reports the GLOBAL opcodes and the module paths
they name. No unpickling, no torch, no allocation proportional to tensor size.
"""
from __future__ import annotations

import io
import json
import pickletools
import sys
import zipfile
from pathlib import Path


def strings_of(pickle_bytes: bytes) -> dict:
    modules: dict[str, set[str]] = {}
    output = io.StringIO()
    try:
        pickletools.dis(pickle_bytes, output)
    except Exception:  # noqa: BLE001
        pass
    text = output.getvalue()
    globals_found = set()
    for line in text.splitlines():
        if "GLOBAL" in line:
            globals_found.add(line.split("GLOBAL", 1)[1].strip())
        if "STACK_GLOBAL" in line:
            globals_found.add("<STACK_GLOBAL>")
    return {"global_lines": sorted(globals_found)[:200],
            "global_count": len(globals_found)}


def scan(path: Path) -> dict:
    result = {"path": str(path), "size_bytes": path.stat().st_size}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        result["zip_members"] = len(names)
        pkl = next((n for n in names if n.endswith("data.pkl")), None)
        result["data_pkl"] = pkl
        raw = archive.read(pkl)
        result["data_pkl_bytes"] = len(raw)
        text = raw.decode("latin-1")
        # Every module path a pickle can name appears as a literal string in data.pkl.
        result["strings"] = strings_of(raw)
        for needle in ("model_vssp", "MHTransformer", "VCG_CIR", "compose_local", "compose_global",
                       "CENet", "Blip2", "__main__", "class_embedding", "positional_embedding"):
            if needle in text:
                result.setdefault("contains", []).append(needle)
        # Module-qualified names look like 'pkg.mod' or 'mod' immediately before a class name.
        candidates = set()
        for chunk in text.split("\x00"):
            pass
        result["byte_strings"] = sorted({s.decode("latin-1") for s in
                                         zip(*[iter(raw.split(b"\x8c"))][0:0])}) if False else None
    return result


print(json.dumps(scan(Path(sys.argv[1])), indent=1))
