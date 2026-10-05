#!/usr/bin/env python3
"""Remove generated workbench mock results, DuckDB index, and logs."""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

WORKBENCH_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    selection = result.add_argument_group("generated outputs to remove")
    selection.add_argument("--mock-results", action="store_true", help="remove artifacts/results/*/mock directories")
    selection.add_argument("--database", action="store_true", help="remove artifacts/workbench.duckdb")
    selection.add_argument("--logs", action="store_true", help="remove artifacts/logs")
    selection.add_argument("--all-generated", action="store_true", help="remove all supported generated outputs")
    result.add_argument("--dry-run", action="store_true")
    return result


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def cleanup_targets(root: Path, mock_results: bool, database: bool, logs: bool) -> list[Path]:
    artifacts = root / "artifacts"
    targets: list[Path] = []
    if mock_results:
        results = artifacts / "results"
        if results.is_dir():
            targets.extend(path for path in results.glob("*/mock") if path.is_dir())
    if database:
        targets.append(artifacts / "workbench.duckdb")
    if logs:
        targets.append(artifacts / "logs")
    return targets


def is_safe_target(path: Path, root: Path) -> bool:
    artifacts = root / "artifacts"
    allowed = {artifacts / "workbench.duckdb", artifacts / "logs"}
    if path in allowed:
        return _within(path, artifacts)
    return path.parent.parent == artifacts / "results" and path.name == "mock" and _within(path, artifacts / "results")


def clean(root: Path, mock_results: bool, database: bool, logs: bool, dry_run: bool) -> int:
    targets = cleanup_targets(root, mock_results, database, logs)
    for target in targets:
        if not is_safe_target(target, root):
            print(f"[BLOCKED] refusing unsafe path: {target}", file=sys.stderr)
            return 2
    if not targets:
        print("[SKIP] no generated paths selected or found")
        return 0
    for target in targets:
        if not target.exists():
            print(f"[SKIP] missing {target}")
        elif dry_run:
            print(f"[RUN] remove {target}")
        elif target.is_dir():
            shutil.rmtree(target)
            print(f"[OK] removed {target}")
        else:
            target.unlink()
            print(f"[OK] removed {target}")
    if dry_run:
        print("[SKIP] dry-run; nothing removed")
    return 0


def main(argv: list[str] | None = None, root: Path = WORKBENCH_ROOT) -> int:
    args = parser().parse_args(argv)
    selected = args.all_generated or args.mock_results or args.database or args.logs
    if not selected:
        print("[BLOCKED] select --mock-results, --database, --logs, or --all-generated", file=sys.stderr)
        return 2
    return clean(root, args.all_generated or args.mock_results, args.all_generated or args.database, args.all_generated or args.logs, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
