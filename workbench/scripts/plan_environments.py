#!/usr/bin/env python3
"""Write a source-backed environment compatibility plan."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.environment_planner import main


if __name__ == "__main__":
    raise SystemExit(main())
