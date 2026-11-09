#!/usr/bin/env python3
"""Independent failure-set diff between the two serial sweeps.

Deliberately re-derives everything from the raw logs rather than trusting any
header written by the sweep launcher: the header is a claim, these two files are
the evidence. Node ids are kept whole, including `[...]` parametrisation, because
a parametrised case is a different test from its sibling and collapsing them
would hide exactly the regressions this is meant to catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REVIEWS = HERE.parents[1]  # workbench/docs/reviews (scripts -> campaign dir -> reviews)

NEW_LOG = REVIEWS / "20261109-native15-composition-limn" / "full_suite_36ae9e2.txt"
BASE_LOG = REVIEWS / "20261109-clean-rerun" / "full_suite_7a58ce6.txt"
OUT = HERE / "failure_diff_36ae9e2.txt"


def failed_ids(path: Path) -> list[str]:
    """Every `FAILED <node id>` line, node id taken up to the ` - ` summary suffix."""
    ids: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("FAILED "):
            continue
        rest = line[len("FAILED "):]
        node = rest.split(" - ", 1)[0].strip()
        if node:
            ids.append(node)
    return sorted(ids)


def main() -> int:
    for path in (NEW_LOG, BASE_LOG):
        if not path.is_file():
            print(f"missing log: {path}", file=sys.stderr)
            return 2

    new = failed_ids(NEW_LOG)
    base = failed_ids(BASE_LOG)

    new_set, base_set = set(new), set(base)
    added = sorted(new_set - base_set)
    dropped = sorted(base_set - new_set)
    common = sorted(new_set & base_set)

    lines = [
        "FAILURE-SET DIFF — 36ae9e2 vs 7a58ce6 baseline",
        "=" * 60,
        f"new log      : {NEW_LOG.relative_to(REVIEWS)}",
        f"baseline log : {BASE_LOG.relative_to(REVIEWS)}",
        "",
        f"new failed count      : {len(new)}",
        f"baseline failed count : {len(base)}",
        f"NEW (only in 36ae9e2) : {len(added)}",
        f"DROPPED (only in base): {len(dropped)}",
        f"COMMON                : {len(common)}",
        "",
        "--- NEW (regressions relative to the baseline) ---",
        *(added or ["(none)"]),
        "",
        "--- DROPPED (present in baseline, absent now) ---",
        *(dropped or ["(none)"]),
        "",
        "--- COMMON ---",
        *common,
        "",
        "--- full new list ---",
        *new,
        "",
        "--- full baseline list ---",
        *base,
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(added)} new, {len(dropped)} dropped, {len(common)} common")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
