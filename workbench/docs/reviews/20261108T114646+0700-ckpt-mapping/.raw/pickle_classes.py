"""Enumerate the pickled class/function globals of a torch save file WITHOUT loading tensors.

A torch `torch.save` archive is a zip whose `data.pkl` member is the pickle. Walking that member with
`pickletools.genops` yields every GLOBAL / STACK_GLOBAL the pickle resolves, in order, plus the string
constants around them. That set names the classes and modules that produced the weights - the
architecture fingerprint - and costs no tensor allocation.

Usage: python pickle_classes.py <path> [--json out.json]
"""
from __future__ import annotations

import json
import pickletools
import sys
import zipfile
from pathlib import Path


def read_pickle_stream(path: Path):
    """Return (pickle_bytes, container_note). Legacy (non-zip) torch saves are a bare pickle."""
    try:
        with zipfile.ZipFile(path) as archive:
            name = next(n for n in archive.namelist() if n.endswith("data.pkl"))
            return archive.read(name), f"zip:{name}"
    except (zipfile.BadZipFile, StopIteration, KeyError):
        return path.read_bytes(), "legacy-bare-pickle"


def main() -> None:
    path = Path(sys.argv[1])
    stream, container = read_pickle_stream(path)
    globals_seen: list[str] = []
    strings: list[str] = []
    stack: list[str] = []
    for opcode, argument, _ in pickletools.genops(stream):
        name = opcode.name
        if name == "GLOBAL":
            globals_seen.append(str(argument))
        elif name == "STACK_GLOBAL":
            if len(stack) >= 2:
                globals_seen.append(f"{stack[-2]}.{stack[-1]}")
        elif name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "BINUNICODE8"}:
            stack.append(str(argument))
        elif name in {"BININT", "BININT1", "BININT2", "LONG1"}:
            stack.append(f"<n:{argument}>")
        elif name in {"EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE", "NEWDICT", "MARK", "TUPLE",
                      "TUPLE1", "TUPLE2", "TUPLE3", "LIST", "APPENDS", "SETITEMS", "REDUCE",
                      "BUILD", "BINPUT", "MEMOIZE", "PROTO", "FRAME", "STOP", "BINGET", "EMPTY_SET",
                      "NEWFALSE", "NEWTRUE", "NONE", "POP", "SHORT_BINBYTES", "BINBYTES"}:
            if name in {"MARK", "EMPTY_DICT", "EMPTY_LIST", "EMPTY_TUPLE"}:
                stack.append(f"<{name}>")
            elif name in {"TUPLE", "LIST", "SETITEMS", "APPENDS", "REDUCE", "BUILD", "NEWDICT"}:
                pass
            else:
                stack.append(f"<{name}>")
        if len(stack) > 64:
            del stack[:-64]
        if len(globals_seen) > 4000:
            break
    deduped = sorted(set(globals_seen))
    modules: dict[str, int] = {}
    for item in globals_seen:
        module = item.rsplit(".", 1)[0]
        modules[module] = modules.get(module, 0) + 1
    report = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "container": container,
        "pickle_stream_bytes": len(stream),
        "distinct_globals": len(deduped),
        "globals": deduped[:400],
        "global_occurrences": len(globals_seen),
        "modules": dict(sorted(modules.items(), key=lambda kv: -kv[1])),
        "first_globals_in_order": globals_seen[:40],
    }
    json.dump(report, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1) \
        if "--json" in sys.argv else print(json.dumps(report, indent=1))


main()
