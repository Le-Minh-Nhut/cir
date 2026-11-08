"""Print one flow/viewport slice of a `qa_report.json` as plain text.

    workbench/envs/ctl312h/bin/python workbench/scripts/browser_qa/dump_report.py \
        <qa_report.json> [flow] [viewport]

No flow/viewport arguments prints every result. Shell output in this session truncates long lines,
so this exists to read a report back reliably rather than to be a second source of truth.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    report = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    want_flow = argv[1] if len(argv) > 1 else None
    want_view = argv[2] if len(argv) > 2 else None
    for result in report.get("results", []):
        if want_flow and result["flow"] != want_flow:
            continue
        if want_view and result["viewport"] != want_view:
            continue
        print(f"== {result['flow']} {result['name']} {result['viewport']} "
              f"-> {result['status']} ({result['duration_ms']} ms)")
        for assertion in result["assertions"]:
            mark = "PASS " if assertion["ok"] else ("BLOCK" if assertion["ok"] is None else "FAIL ")
            print(f"  {mark} {assertion['id']} | {assertion['detail']}")
        for miss in result.get("selector_misses", []):
            print(f"  MISS {miss['key']} ({miss['purpose']}): {miss['tried'][:3]}")
        for note in result.get("notes", []):
            print(f"  note: {note}")
        for shot in result.get("screenshots", []):
            print(f"  shot: {shot}")
    summary = report.get("summary", {})
    if not want_flow:
        print(f"-- total={summary.get('total')} pass={summary.get('pass')} "
              f"fail={summary.get('fail')} blocked={summary.get('blocked')} "
              f"selector_miss={summary.get('selector_miss')} "
              f"console_errors={summary.get('console_error_total')} "
              f"page_errors={summary.get('page_error_total')}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
