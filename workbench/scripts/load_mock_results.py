#!/usr/bin/env python3
"""Generate deterministic schema-v2 mock results into ignored artifacts."""
from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.tests.mock_data import build_mock_runs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=resolve_config().WORKBENCH_RESULTS_ROOT)
    args = parser.parse_args()
    for stale in args.output_root.glob("**/mock"):
        shutil.rmtree(stale)
    for result in build_mock_runs():
        run = result.run
        destination = args.output_root / run.protocol_id / "mock" / f"{run.run_id}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Written via a temporary + atomic replace: a direct `write_text` to the final name leaves
        # a TRUNCATED canonical JSON if interrupted, and the index then refuses to load the whole
        # results root. Fixtures are development data, but the reader's contract is the same.
        handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                file.write(result.model_dump_json(indent=2) + "\n")
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
    print(f"generated {len(build_mock_runs())} deterministic schema-v2 mock result files in {args.output_root}")


if __name__ == "__main__":
    main()
