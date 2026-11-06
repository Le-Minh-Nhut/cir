"""Lower the CSMCIR query batch size to fit the card (operational parameter only).

The shipped `DataLoader(batch_size=64)` for the relative (query) split asks `inference` for a
`torch.matmul` that tries to allocate 10.49 GiB while 8.82 GiB is free:

    torch.OutOfMemoryError: Tried to allocate 10.49 GiB ... 8.82 GiB free

Batch size is an operational knob, not a semantic one: every query is still scored against the whole
gallery, the split and gallery are unchanged, and no caption or ordering moves. This script rewrites
that one literal to a value taken from the command line so the reason appears in the evidence instead
of in a forgotten shell edit.
"""
import hashlib
import re
import sys
from pathlib import Path

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "workbench/artifacts/csmcir_derived/src/validate_blip_csmcir.py")
size = int(sys.argv[2]) if len(sys.argv) > 2 else 16

s = target.read_text()
old = "    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size=64,"
new = (f"    # OPERATIONAL ONLY: 64 needs a 10.49 GiB matmul that does not fit beside the model on a\n"
       f"    # 15.44 GiB card. Queries are still scored against the full gallery; only how many are\n"
       f"    # pushed through `inference` at once changes.\n"
       f"    relative_val_loader = DataLoader(dataset=relative_val_dataset, batch_size={size},")
if s.count(old) != 1:
    raise SystemExit(f"expected exactly 1 query loader, found {s.count(old)}")
target.write_text(s.replace(old, new, 1))
print(f"query batch_size -> {size}")
print(f"digest now {hashlib.sha256(target.read_bytes()).hexdigest()}")
