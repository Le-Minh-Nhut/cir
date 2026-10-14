#!/usr/bin/env python3
"""Persist independent audit payloads into workbench/docs/reviews/models/.

The eval `output()` helper hands back the payload file path; the payload is a JSON object whose
`markdown` string carries real newlines, so a plain `json.loads` fails. Decoding the escapes by hand
keeps non-ASCII characters intact, which `unicode_escape` would mangle.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

#: The harness writes each subagent's payload next to its session log. The session root is not stable
#: across machines, so it comes from the environment with the local default first; both are searched.
SESSION_CANDIDATES = [
    Path(os.environ["OMP_SESSION_DIR"]) if os.environ.get("OMP_SESSION_DIR") else None,
    Path.home() / ".omp/agent/sessions/-data-cir",
]
DEST = Path(__file__).resolve().parents[2] / "workbench/docs/reviews/models"


def _session_dir() -> Path | None:
    for candidate in SESSION_CANDIDATES:
        if candidate and candidate.is_dir():
            sessions = sorted(p for p in candidate.iterdir() if p.is_dir())
            if sessions:
                return sessions[-1]
    return None

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
    session = _session_dir()
    for arg in argv:
        if Path(arg).is_file():
            src = Path(arg)
        elif session is not None:
            src = session / f"{arg}.md"
        else:
            print(f"{arg}: no session directory found; pass the payload path directly")
            continue
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
