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
from workbench.scripts.clean_generated import _within
from workbench.tests.mock_data import build_mock_runs


def is_safe_target(path: Path, output_root: Path) -> bool:
    try:
        resolved_path = path.resolve()
        resolved_root = output_root.resolve()
        resolved_path.relative_to(resolved_root)
    except ValueError:
        return False
    return resolved_path.name == "mock" and resolved_path.parent.parent == resolved_root and _within(resolved_path, resolved_root)


def is_safe_output_root(output_root: Path, config: object | None = None) -> bool:
    cfg = config or resolve_config()
    out = output_root.resolve()
    repo_root = getattr(cfg, "CIR_REPO_ROOT", None)
    if repo_root is not None:
        repo = repo_root.resolve()
        if out == repo or repo.is_relative_to(out):
            return False
    data_root = getattr(cfg, "CIR_DATA_ROOT", None)
    if data_root is not None:
        data = data_root.resolve()
        if out == data or data.is_relative_to(out):
            return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=resolve_config().WORKBENCH_RESULTS_ROOT)
    args = parser.parse_args(argv)
    if not is_safe_output_root(args.output_root):
        print(f"[BLOCKED] refusing unsafe output root: {args.output_root}", file=sys.stderr)
        return 2
    for stale in args.output_root.glob("**/mock"):
        if not is_safe_target(stale, args.output_root):
            print(f"[BLOCKED] refusing unsafe path: {stale}", file=sys.stderr)
            continue
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
