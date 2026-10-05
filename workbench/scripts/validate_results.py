#!/usr/bin/env python3
"""Validate canonical result JSON artifacts without modifying them."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.index import RESULTS_ROOT, load_runs
from workbench.backend.schemas.results import ResultRun

PROVENANCE_FIELDS = ("upstream_commit", "checkpoint_sha256", "command_digest", "environment_digest")


def _missing_provenance(result: ResultRun) -> list[str]:
    return [field for field in PROVENANCE_FIELDS if not getattr(result.run, field)]


def _report_provenance(runs: list[ResultRun], strict_real: bool, output: Callable[[str], None]) -> bool:
    valid = True
    for result in runs:
        if result.run.data_kind == "mock":
            continue
        missing = _missing_provenance(result)
        if missing:
            level = "BLOCKED" if strict_real else "WARN"
            output(f"[{level}] {result.run.run_id}: missing provenance: {', '.join(missing)}")
            valid = valid and not strict_real
    return valid


def validate_root(root: Path, strict_real: bool = False, output: Callable[[str], None] = print) -> bool:
    """Validate every result file below root using index loading rules."""
    try:
        runs = load_runs(root)
    except Exception as error:
        output(f"[BLOCKED] {root}: {error}")
        return False
    valid = _report_provenance(runs, strict_real, output)
    output(f"[OK] validated {len(runs)} runs under {root}")
    return valid


def validate_file(path: Path, strict_real: bool = False, output: Callable[[str], None] = print) -> bool:
    """Validate one result artifact without changing it."""
    try:
        result = ResultRun.model_validate_json(path.read_text())
    except Exception as error:
        output(f"[BLOCKED] {path}: {error}")
        return False
    valid = _report_provenance([result], strict_real, output)
    output(f"[OK] validated {path}")
    return valid


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    selectors = parser.add_mutually_exclusive_group(required=True)
    selectors.add_argument("--file", type=Path)
    selectors.add_argument("--root", type=Path)
    selectors.add_argument("--all", action="store_true", help="Validate all canonical result artifacts.")
    parser.add_argument("--strict-real", action="store_true", help="Treat missing real-artifact provenance as errors.")
    args = parser.parse_args(argv)
    if args.file is not None:
        valid = validate_file(args.file, args.strict_real)
    else:
        valid = validate_root(RESULTS_ROOT if args.all else args.root, args.strict_real)
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
