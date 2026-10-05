#!/usr/bin/env python3
"""Validate a later GPU evaluation request and print its official command; never executes it."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--protocol", required=True, choices=["fashioniq_original_split", "fashioniq_val_split"])
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()
    request = EvalRequest(args.model, args.checkpoint, args.protocol, args.dataset_root, args.output, args.top_k)
    print("Official command (review and execute only on approved GPU host):")
    print(" ".join(ADAPTERS[args.model]().build_command(request)))


if __name__ == "__main__":
    main()
