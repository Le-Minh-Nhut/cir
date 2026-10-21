#!/usr/bin/env python3
"""Rebuild derived DuckDB index from canonical result JSON files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.index import configured_database_path, rebuild_index
from workbench.backend.operator_config import resolve_config
from workbench.scripts.validate_results import validate_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    # One resolution, threaded everywhere: the results read, the database written and the verifier
    # receipts that promote a run all have to describe the same workspace, or an isolated operator
    # indexes their own results while the receipts are read from the deployment's reports root.
    config = resolve_config()
    parser.add_argument("--results-root", type=Path, default=config.WORKBENCH_RESULTS_ROOT)
    parser.add_argument("--database", type=Path, default=None)
    parser.add_argument("--check-only", action="store_true", help="Check result artifacts without creating an index.")
    parser.add_argument("--validate-first", action="store_true", help="Validate result artifacts before rebuilding.")
    args = parser.parse_args(argv)
    if (args.validate_first or args.check_only) and not validate_root(args.results_root):
        return 1
    if args.check_only:
        print("[OK] index unchanged")
        return 0
    database = args.database
    if database is None:
        database = configured_database_path(config)
    print(f"[OK] indexed {rebuild_index(args.results_root, database, config=config)} runs into {database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
