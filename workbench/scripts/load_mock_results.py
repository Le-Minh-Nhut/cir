#!/usr/bin/env python3
"""Install committed schema-v2 mock result JSON files into ignored artifacts."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.index import RESULTS_ROOT

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    for stale in args.output_root.glob("**/mock"):
        shutil.rmtree(stale)
    copied = 0
    for source in sorted(FIXTURES.glob("*.json")):
        protocol = "fashioniq_original_split" if source.name.startswith("original_") else "fashioniq_val_split"
        destination = args.output_root / protocol / "mock" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1
    print(f"installed {copied} schema-v2 mock result files in {args.output_root}")


if __name__ == "__main__":
    main()
