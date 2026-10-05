#!/usr/bin/env python3
"""Generate deterministic schema-v2 mock results into ignored artifacts."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.index import RESULTS_ROOT
from workbench.tests.mock_data import build_mock_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=RESULTS_ROOT)
    args = parser.parse_args()
    for stale in args.output_root.glob("**/mock"):
        shutil.rmtree(stale)
    for result in build_mock_runs():
        run = result.run
        destination = args.output_root / run.protocol_id / "mock" / f"{run.run_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result.model_dump_json(indent=2) + "\n")
    print(f"generated {len(build_mock_runs())} deterministic schema-v2 mock result files in {args.output_root}")


if __name__ == "__main__":
    main()
