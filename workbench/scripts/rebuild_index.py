#!/usr/bin/env python3
"""Rebuild derived DuckDB index from canonical result JSON files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.index import DATABASE_PATH, RESULTS_ROOT, rebuild_index
from workbench.scripts.validate_results import validate_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    parser.add_argument("--check-only", action="store_true", help="Check result artifacts without creating an index.")
    parser.add_argument("--validate-first", action="store_true", help="Validate result artifacts before rebuilding.")
    args = parser.parse_args(argv)
    if (args.validate_first or args.check_only) and not validate_root(args.results_root):
        return 1
    if args.check_only:
        print("[OK] index unchanged")
        return 0
    print(f"[OK] indexed {rebuild_index(args.results_root, args.database)} runs into {args.database}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
