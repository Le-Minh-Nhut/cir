"""Build a minimal CPython extension exporting ``PyInit__C`` for tests.

The GPU-capability corroboration requires a `torch._C` that the orchestrator can actually
load, so integration tests that simulate a verified GPU environment need a real extension
rather than a placeholder file. Compiled once per destination and cached on disk.
"""
from __future__ import annotations

import subprocess
import sysconfig
from pathlib import Path

_SOURCE = (
    '#include <Python.h>\n'
    'static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL};\n'
    'PyMODINIT_FUNC PyInit__C(void) { return PyModule_Create(&mod); }\n'
)


def build_core_module(destination: Path) -> Path:
    """Compile the fixture extension into *destination* and return it."""
    if destination.is_file():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    source = destination.with_suffix(".c")
    source.write_text(_SOURCE)
    cc = (sysconfig.get_config_var("CC") or "cc").split()
    include = sysconfig.get_paths()["include"]
    completed = subprocess.run(
        [*cc, "-shared", "-fPIC", f"-I{include}", str(source), "-o", str(destination)],
        capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"could not build the fixture extension: {completed.stderr}")
    return destination
