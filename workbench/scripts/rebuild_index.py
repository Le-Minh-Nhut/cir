#!/usr/bin/env python3
"""Rebuild derived DuckDB index from canonical result JSON files."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.index import DATABASE_PATH, RESULTS_ROOT, rebuild_index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--database", type=Path, default=DATABASE_PATH)
    args = parser.parse_args()
    print(f"indexed {rebuild_index(args.results_root, args.database)} runs into {args.database}")


if __name__ == "__main__":
    main()
