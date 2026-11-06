"""Make the CSMCIR gallery extraction accumulate on CPU (operational, not semantic).

The shipped `extract_index_blip_caption_features` keeps every batch of gallery features on the GPU and
`torch.vstack`s them at the end. On a 15.44 GiB card that asks for ~5.15 GiB it does not have once the
model is resident:

    torch.OutOfMemoryError: Tried to allocate 5.15 GiB. ... 1.41 GiB free, 13.39 GiB in use

The same function already supports this: `save_memory=True` moves each batch to CPU as it is produced.
That is the author's own path, not a rewrite, and it changes no number -- same model, same batch order,
same features, only where they live while accumulating.

Only the METRIC call site is touched. The evaluator calls the extractor a second time inside the
visualization block, which the approved patch already made opt-in and which this run does not execute.

This is deliberately not a third entry in `approved_source_patches`: the registry declares the import
and visualization repairs, and adding another declaration is an operator policy change, not something
to slip in through a script.
"""
import hashlib
import sys
from pathlib import Path

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "workbench/artifacts/csmcir_derived/src/validate_blip_csmcir.py")

# The metric call site is the one immediately followed by the metric computation; the second call
# site sits in the visualization block. Anchoring on the following line keeps them distinguishable.
OLD = (
    "        index_features, index_names = extract_index_blip_caption_features(classic_val_dataset, blip_model)\n"
    "        recall_at10, recall_at50 = compute_fiq_val_metrics("
)
NEW = (
    "        # OPERATIONAL ONLY: accumulate gallery features on CPU (the extractor's own `save_memory`\n"
    "        # path). The shipped default keeps every batch on the GPU and then vstacks, which needs\n"
    "        # ~5.15 GiB the card does not have after the model is loaded. Same model, same batch\n"
    "        # order, same features; only their residence while accumulating changes.\n"
    "        index_features, index_names = extract_index_blip_caption_features(\n"
    "            classic_val_dataset, blip_model, save_memory=True)\n"
    "        recall_at10, recall_at50 = compute_fiq_val_metrics("
)

s = target.read_text()
count = s.count(OLD)
if count != 1:
    raise SystemExit(f"expected exactly 1 metric call site, found {count}; refusing to guess")
target.write_text(s.replace(OLD, NEW, 1))
print(f"edited {target}")
print(f"digest now {hashlib.sha256(target.read_bytes()).hexdigest()}")
