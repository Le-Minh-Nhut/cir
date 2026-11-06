"""Make the whole CSMCIR validation pass honour the extractor's own `save_memory` path.

The shipped default keeps every batch of gallery features on the GPU and `torch.vstack`s them at the
end, which needs ~5.15 GiB more than a 15.44 GiB card has left once the model is resident:

    torch.OutOfMemoryError: Tried to allocate 5.15 GiB ... 13.39 GiB in use, 1.41 GiB free

The author already ships the remedy at BOTH ends: `extract_index_blip_caption_features` takes
`save_memory=True` to land each batch on CPU, and `generate_fiq_val_predictions` takes the same flag to
move the features back to `blip_model.device` right before `blip_model.inference(...)`. Setting only the
first half fails exactly as the shipped code is written and produces the mixed-device error below,
because `inference` cats a GPU tensor with a CPU one:

    RuntimeError: Expected all tensors to be on the same device, but found at least two devices

So the flag is threaded through both call sites. Nothing else changes: same model, same batches, same
order, same captions. Only where the features live between extraction and comparison moves.

This is NOT added to `approved_source_patches`: the registry declares the import repair and the
visualization opt-in, and a third declaration is an operator policy decision.
"""
import hashlib
import sys
from pathlib import Path

target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
    "workbench/artifacts/csmcir_derived/src/validate_blip_csmcir.py")

OLD = (
    "        index_features, index_names = extract_index_blip_caption_features(\n"
    "            classic_val_dataset, blip_model, save_memory=True)\n"
    "        recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,\n"
    "..."
)

# Replace the metric call site as a whole, so the flag reaches both the extractor and the predictor.
OLD_BLOCK = """        index_features, index_names = extract_index_blip_caption_features(
            classic_val_dataset, blip_model, save_memory=True)
        recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                            index_features, index_names, txt_processors)"""

NEW_BLOCK = """        # OPERATIONAL ONLY: gallery features accumulate on CPU (`save_memory`), and the predictor
        # moves them back to the model device before `inference`. Both halves are the author's own
        # `save_memory` path; the shipped default needs ~5.15 GiB this card does not have.
        index_features, index_names = extract_index_blip_caption_features(
            classic_val_dataset, blip_model, save_memory=True)
        recall_at10, recall_at50 = compute_fiq_val_metrics(relative_val_dataset, blip_model,
                                                            index_features, index_names, txt_processors,
                                                            save_memory=True)"""

# `compute_fiq_val_metrics` must accept and forward the flag.
OLD_METRICS_SIG = "def compute_fiq_val_metrics(relative_val_dataset: FashionIQDataset, blip_model, index_features: torch.tensor,\n                            index_names: List[str], txt_processors, save_memory=False) -> Tuple[float, float]:"
NEW_METRICS_SIG = OLD_METRICS_SIG  # already accepts save_memory

OLD_FORWARD = "    pred_sim, target_names, reference_names, captions_all = generate_fiq_val_predictions(blip_model, relative_val_dataset,\n                                                                    index_names, index_features, txt_processors, save_memory)"
NEW_FORWARD = OLD_FORWARD  # already forwards it

s = target.read_text()
if OLD_BLOCK not in s:
    raise SystemExit("metric call block not found; refusing to guess")
s = s.replace(OLD_BLOCK, NEW_BLOCK, 1)
target.write_text(s)
print(f"edited {target}")
print(f"digest now {hashlib.sha256(target.read_bytes()).hexdigest()}")
