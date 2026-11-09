#!/usr/bin/env python3
"""RAM guard: refuse to launch a native eval unless MemAvailable > predicted + headroom.

Usage: ram_guard.py <predicted_gib> <headroom_gib>
Exit 0 = GO, exit 3 = REFUSE. Prints a machine-readable line for the report.
"""
import sys
from pathlib import Path

predicted = float(sys.argv[1])
headroom = float(sys.argv[2])
need = predicted + headroom

info = {}
for line in Path("/proc/meminfo").read_text().splitlines():
    k, v = line.split(":", 1)
    info[k] = int(v.split()[0])  # kB

avail_gib = info["MemAvailable"] / 2**20
anon_gib = info.get("AnonPages", 0) / 2**20
swap_free_gib = info.get("SwapFree", 0) / 2**20

verdict = "GO" if avail_gib >= need else "REFUSE"
print(f"RAMGUARD verdict={verdict} avail={avail_gib:.2f}GiB "
      f"predicted={predicted:.2f}GiB headroom={headroom:.2f}GiB need={need:.2f}GiB "
      f"anon={anon_gib:.2f}GiB swapfree={swap_free_gib:.2f}GiB")
sys.exit(0 if verdict == "GO" else 3)
