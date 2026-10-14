#!/usr/bin/env python3
"""Persist independent audit payloads into workbench/docs/reviews/models/.

The eval `output()` helper hands back the payload file path; the payload is a JSON object whose
`markdown` string carries real newlines, so a plain `json.loads` fails. Decoding the escapes by hand
keeps non-ASCII characters intact, which `unicode_escape` would mangle.
"""
from __future__ import annotations

import sys
from pathlib import Path

SESSION = Path("/home/heheboiz/.omp/agent/sessions/-data-cir/2026-10-13T13-09-01-288Z_01a13561-72e8-70cb-a614-83b9e812b501")
DEST = Path("/home/heheboiz/data/cir/workbench/docs/reviews/models")

ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


def extract(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="surrogateescape")
    marker = '"markdown": "'
    if marker not in raw:
        return raw
    body = raw[raw.index(marker) + len(marker): raw.rindex('"')]
    out, i = [], 0
    while i < len(body):
        ch = body[i]
        if ch == "\\" and i + 1 < len(body):
            nxt = body[i + 1]
            out.append(ESCAPES.get(nxt, "\\" + nxt))
            i += 2
        else:
            out.append(ch)
            i += 1
    text = "".join(out)
    for end_marker in ('\n  "model_id":', '\n}\n'):
        j = text.find(end_marker)
        if j > 0:
            text = text[:j]
            break
    return text


def main(argv: list[str]) -> int:
    for arg in argv:
        src = Path(arg) if Path(arg).is_file() else SESSION / f"{arg}.md"
        if not src.is_file():
            print(f"MISSING {src}")
            continue
        md = extract(src)
        name = src.stem.replace("Audit", "").lower()
        if name in ("pthamtst",):
            name = "ptha_mtst"
        dest = DEST / f"{name}.md"
        dest.write_text(md)
        verdict = [line for line in md.splitlines() if "FINAL VERDICT" in line]
        print(f"{dest.name} {len(md)}B | {verdict[-1][:110] if verdict else 'NO VERDICT LINE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
