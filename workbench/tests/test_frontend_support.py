#!/usr/bin/env python3
"""Run the frontend's pure modules through node, with no JS test framework.

The frontend has no test runner (`package.json` has no test script and no dev dependency to add
one); this repository already compiles the shipping `.ts` sources with the installed `tsc` and
executes them from Python (`workbench/tests/test_frontend_staleness.py`). This module is the same
pattern, generalized: `compile_module` emits one source file into a temp directory and returns the
JavaScript path, and `run_script` executes an ES module that imports it.

Every assertion below therefore executes the code the bundle is built from, not a copy.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FRONTEND = REPO / "workbench" / "frontend"


def compile_module(source: str, out_dir: Path) -> Path:
    """Compile one shipping TypeScript module to ESM JavaScript and return the emitted file.

    Each call emits into a FRESH subdirectory of `out_dir`. The fixed path a caller names is a
    scratch parent, not the artifact's identity: with a shared `/tmp/<name>/<module>.js`, bytes left
    by an earlier campaign were served to a later run (`scope.js` without the W-02 `family` key made
    two tests fail inside a full sweep while passing in isolation, because nothing that run did could
    be shown to have replaced them). A unique directory makes the served artifact this run's output.
    """
    if shutil.which("npx") is None:
        pytest.skip("npx is not installed: the pure module cannot be compiled here")
    # A CONTENT digest, not `hash()`: `hash()` on str is salted per process via PYTHONHASHSEED, so
    # the same module emitted into a different directory every run — unreproducible names, and the
    # directories are never reusable or prunable by name. The digest is stable across processes and
    # still unique per (out_dir, source).
    token = hashlib.sha1(f"{out_dir}\x00{source}".encode()).hexdigest()[:12]
    out_dir = out_dir / f"build-{os.getpid()}-{token}"
    out_dir.mkdir(parents=True, exist_ok=True)
    emitted = subprocess.run(
        ["npx", "tsc", source, "--ignoreConfig", "--outDir", str(out_dir),
         "--module", "esnext", "--target", "es2022", "--moduleResolution", "bundler",
         "--skipLibCheck", "--strict"],
        cwd=FRONTEND, capture_output=True, text=True, timeout=240)
    # tsc writes diagnostics to stdout; an empty stderr with a non-zero exit is still a failure.
    assert emitted.returncode == 0, (
        f"{source} does not compile: {emitted.stdout[-1500:]} {emitted.stderr[-800:]}")
    program = out_dir / (Path(source).with_suffix(".js").name)
    assert program.is_file(), f"{source} produced no JavaScript at {program}"
    return program


def run_script(script: str, timeout: int = 120) -> str:
    result = subprocess.run(["node", "--input-type=module", "-e", script],
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, f"node failed: {result.stdout[-1500:]} {result.stderr[-1500:]}"
    return result.stdout


def run_json(script: str) -> object:
    """Execute a script whose last statement prints one JSON document, and parse it."""
    out = run_script(script)
    for line in reversed(out.strip().splitlines()):
        line = line.strip()
        if line.startswith("__R__"):
            return json.loads(line[len("__R__"):])
    raise AssertionError(f"script printed no JSON: {out[-800:]}")
