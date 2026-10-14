#!/usr/bin/env python3
"""Persist a paper-extraction subagent's JSON payload into the artifact cache.

The payload lives next to the agent's session log. It is JSON, but the harness sometimes emits it
without escaping embedded newlines, so it is decoded by hand rather than with `json.loads`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

#: The harness writes each subagent's payload next to its session log, and the session root differs per
#: machine, so it is discovered rather than hardcoded.
SESSION_CANDIDATES = [
    Path(os.environ["OMP_SESSION_DIR"]) if os.environ.get("OMP_SESSION_DIR") else None,
    Path.home() / ".omp/agent/sessions/-data-cir",
]
CACHE = Path(__file__).resolve().parents[2] / "workbench/registry/paper_benchmarks/extracted"


def _session_dir() -> Path | None:
    for candidate in SESSION_CANDIDATES:
        if candidate and candidate.is_dir():
            sessions = sorted(p for p in candidate.iterdir() if p.is_dir())
            if sessions:
                return sessions[-1]
    return None


def load_payload(agent: str):
    session = _session_dir()
    if session is None:
        return None
    for suffix in (".json", ".md"):
        path = session / f"{agent}{suffix}"
        if not path.is_file():
            continue
        raw = path.read_text(encoding="utf-8", errors="surrogateescape").strip()
        try:
            return json.loads(raw, strict=False)
        except json.JSONDecodeError:
            pass
        # Decode the escape sequences by hand; `unicode_escape` would mangle non-ASCII.
        text = raw
        for marker in ('{\n  "paper"', '[\n  {\n    "paper"', '{"paper"'):
            i = text.find(marker)
            if i >= 0:
                text = text[i:]
                break
        out, i = [], 0
        esc = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}
        while i < len(text):
            ch = text[i]
            if ch == "\\" and i + 1 < len(text):
                out.append(esc.get(text[i + 1], "\\" + text[i + 1]))
                i += 2
            else:
                out.append(ch)
                i += 1
        try:
            return json.loads("".join(out), strict=False)
        except json.JSONDecodeError as error:
            print(f"{agent}: undecodable ({error})")
            return None
    return None


def main(argv: list[str]) -> int:
    CACHE.mkdir(parents=True, exist_ok=True)
    for agent in argv:
        payload = load_payload(agent)
        if payload is None:
            print(f"{agent}: no payload")
            continue
        records = payload if isinstance(payload, list) else [payload]
        written = []
        for record in records:
            paper = record.get("paper") or {}
            slug = str(paper.get("paper_id") or agent).replace("/", "_")
            dest = CACHE / f"{slug}.json"
            dest.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
            experiments = record.get("experiments") or []
            written.append(f"{dest.name}({len(experiments)} rows)")
        print(f"{agent}: " + ", ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
