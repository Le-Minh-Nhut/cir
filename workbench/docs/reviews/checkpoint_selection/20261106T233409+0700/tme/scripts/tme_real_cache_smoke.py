"""Real-model TME dtype gate: actual checkpoint forward, production validator, BOTH verdicts.

Design constraints this script respects:

* It never writes into the real cache location (`<checkout>/features/FashionIQ`). A slice-shaped
  cache sitting where the wrapper looks would let a later run "validate" a 24-image slice against
  the real 15536-image dataset, and the surviving `val_name_to_idx.json` is the last byte-level
  proof of the deleted 22.49 GB cache's identity.
* The gallery-identity guard must NOT be the thing that decides the verdict, or both directions
  would fail for an unrelated reason and the booleans would be decorative. So it builds a scratch
  dataset whose split files name exactly the slice, making dtype the ONLY differing gate between
  the two contracts.

What this proves: on the adapted runtime the real EVA-ViT tower emits float32 at [N,257,1408],
including a partial final batch, and the production `_tme_validate` accepts that under the adapted
contract while rejecting the same bytes under the historical float16 contract.

What this does NOT prove: FashionIQ Recall. No metric is computed here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

CHECKOUT = REPO / "workbench/third_party/TME"
DATASET = REPO / "data/FashionIQ"
CATEGORIES = ("dress", "shirt", "toptee")
IMAGE_EXT = ".png"          # fashioniq_standard


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    checkpoint = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    n_images = int(sys.argv[3]) if len(sys.argv) > 3 else 24
    batch = int(sys.argv[4]) if len(sys.argv) > 4 else 10      # 10 -> 10,10,4 partial tail

    import torch
    from workbench.backend.registry import (
        generated_artifact_parameters, load_preparation_contracts, model_by_id,
    )
    from workbench.scripts.prepare_model import _tme_validate

    contract = load_preparation_contracts()["tme_fashioniq"]
    entry = model_by_id("tme")
    adapted = generated_artifact_parameters(contract, "tme_precompute_features", entry)
    default = generated_artifact_parameters(contract, "tme_precompute_features", None)
    assert adapted["feature_dtype"] != default["feature_dtype"], (
        "no dtype difference between the two contracts; nothing to decide")

    os.chdir(CHECKOUT)                       # native code opens ./weight/...
    sys.path.insert(0, str(CHECKOUT / "src"))
    from lavis.models import load_model_and_preprocess

    model, vis_processors, _ = load_model_and_preprocess(
        name="blip2_cir_image_diff_features", model_type="pretrain", is_eval=True, device="cuda")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and isinstance(state.get("model"), dict):
        state = state["model"]
    load_report = model.load_state_dict(state, strict=False)
    print(f"load: missing={len(load_report.missing_keys)} "
          f"unexpected={len(load_report.unexpected_keys)}")

    names: list[str] = []
    for category in CATEGORIES:
        for name in json.loads((DATASET / f"image_splits/split.{category}.val.json").read_text()):
            if name not in names:
                names.append(name)
    names = names[:n_images]

    from PIL import Image

    transform = vis_processors["eval"]
    chunks, batch_sizes = [], []
    with torch.no_grad():
        for start in range(0, len(names), batch):
            chunk = names[start:start + batch]
            batch_sizes.append(len(chunk))
            images = torch.stack([
                transform(Image.open(DATASET / f"images/{name}{IMAGE_EXT}").convert("RGB"))
                for name in chunk]).cuda()
            with torch.cuda.amp.autocast():
                chunks.append(model.vit_encode(images).cpu())
    features = torch.cat(chunks, dim=0)

    observed = str(features.dtype).replace("torch.", "")
    result = {
        "checkpoint": str(checkpoint),
        "interpreter": sys.executable,
        "torch": torch.__version__,
        "cuda_capability": list(torch.cuda.get_device_capability()),
        "gallery_slice": len(names),
        "batch_size": batch,
        "batch_sizes_seen": batch_sizes,
        "partial_final_batch_rows": batch_sizes[-1] if batch_sizes[-1] != batch else 0,
        "feature_shape": list(features.shape),
        "feature_dtype": observed,
        "element_size": features.element_size(),
        "checkpoint_load": {"missing": len(load_report.missing_keys),
                            "unexpected": len(load_report.unexpected_keys)},
        "default_contract_dtype": default["feature_dtype"],
        "adapted_contract_dtype": adapted["feature_dtype"],
        "contract_resolution": "workbench.backend.registry.generated_artifact_parameters",
    }

    scratch = Path(tempfile.mkdtemp(prefix="tme_dtype_gate_"))
    try:
        root = scratch / "features/FashionIQ"
        root.mkdir(parents=True)
        feature = root / "val_images.pt"
        torch.save(features, feature)
        mapping = root / "val_name_to_idx.json"
        mapping.write_text(json.dumps({name: index for index, name in enumerate(names)}),
                           encoding="utf-8")
        # Scratch dataset whose splits name EXACTLY this slice, so the identity guard passes and
        # dtype is the only remaining difference between the two contracts.
        scratch_dataset = scratch / "FashionIQ"
        (scratch_dataset / "image_splits").mkdir(parents=True)
        # The validator reconstructs expected names by CONCATENATING dress+shirt+toptee, so the
        # slice must live in one category to keep the exact order the features were extracted in.
        # Round-robin distribution reordered the gallery and made the identity guard reject, which
        # would have read as a dtype failure for the wrong reason.
        per_category = {category: [] for category in CATEGORIES}
        per_category[CATEGORIES[0]] = list(names)
        split_files = {}
        for category, category_names in per_category.items():
            path = scratch_dataset / f"image_splits/split.{category}.val.json"
            path.write_text(json.dumps(category_names), encoding="utf-8")
            split_files[category] = _sha(path)
        inputs = {
            "dataset_root": str(scratch_dataset),
            "split_files": split_files,
            "gallery_images": {},
            "base_assets": {},
            "recipe": {"source_commit": entry["upstream_commit_sha"]},
        }
        shape = list(features.shape)
        receipt = root / adapted["receipt_filename"]
        receipt.write_text(json.dumps({
            "kind": "workbench_tme_precompute_receipt", "model_id": "tme",
            "generator": "tme_precompute_features", "source_script": adapted["source_script"],
            "source_commit": entry["upstream_commit_sha"], "dataset": adapted["eval_dataset"],
            "split": adapted["split"], "batch_size": adapted["batch_size"],
            "workers": adapted["workers"], "inputs": inputs,
            "files": {"features": {"size_bytes": feature.stat().st_size,
                                   "sha256": _sha(feature)},
                      "name_to_idx": {"size_bytes": mapping.stat().st_size,
                                      "sha256": _sha(mapping)}},
            "feature_shape": shape, "feature_dtype": str(features.dtype),
        }, indent=2), encoding="utf-8")

        parameters = {**adapted, "tokens_per_image": shape[1], "feature_width": shape[2]}
        commit = entry["upstream_commit_sha"]
        adapted_ok = _tme_validate(root, parameters, inputs, commit, receipt)
        default_ok = _tme_validate(root, {**parameters, "feature_dtype": default["feature_dtype"]},
                                   inputs, commit, receipt)

        # Proves the identity guard is satisfied, i.e. the two verdicts above differ for the dtype
        # reason and not because the gallery was rejected wholesale.
        identity_ok = _tme_validate(root, parameters, inputs, commit, receipt)
        result.update(
            validator_accepts_adapted_contract=adapted_ok,
            validator_rejects_default_contract=not default_ok,
            identity_guard_satisfied_with_matching_split=identity_ok,
            decision_is_dtype_only=(adapted_ok and not default_ok and identity_ok),
            scratch_root_used=str(scratch),
            real_cache_location_written=False,
        )

        # Wrong dtype must still be refused even under the adapted contract.
        odd_root = scratch / "odd"
        odd_root.mkdir()
        odd_feature = odd_root / "val_images.pt"
        torch.save(features.to(torch.float64), odd_feature)
        (odd_root / "val_name_to_idx.json").write_text(mapping.read_text(), encoding="utf-8")
        odd_receipt = odd_root / adapted["receipt_filename"]
        odd_receipt.write_text(receipt.read_text(), encoding="utf-8")
        result["validator_rejects_float64_under_adapted_contract"] = not _tme_validate(
            odd_root, parameters, inputs, commit, odd_receipt)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in (
        "feature_shape", "feature_dtype", "batch_sizes_seen", "partial_final_batch_rows",
        "default_contract_dtype", "adapted_contract_dtype",
        "validator_accepts_adapted_contract", "validator_rejects_default_contract",
        "decision_is_dtype_only", "validator_rejects_float64_under_adapted_contract")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
