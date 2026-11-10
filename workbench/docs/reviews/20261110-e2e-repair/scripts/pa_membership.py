"""Phase A: membership facts that decide what 'Full Gallery' can mean, measured from disk."""
import json, hashlib, sys
from pathlib import Path
sys.path.insert(0, "/home/heheboiz/data/cir/src")
root = Path("/home/heheboiz/data/cir/data/FashionIQ")
prepared = Path("/home/heheboiz/data/cir/workbench/artifacts/checkpoint_selection/20261106T002617+0700/prepared/clip4cir/fashionIQ_dataset")
out = {"fashioniq_root": str(root), "prepared_clip4cir": str(prepared), "categories": {}}
all_ids = set()
for cat in ("dress", "shirt", "toptee"):
    split = json.loads((root/"image_splits"/f"split.{cat}.val.json").read_text())
    caps = json.loads((root/"captions"/f"cap.{cat}.val.json").read_text())
    ids = set(split) | {c["candidate"] for c in caps} | {c["target"] for c in caps}
    all_ids |= set(split)
    png_missing = [i for i in split if not (root/"images"/f"{i}.png").is_file()]
    jpg_missing = [i for i in split if not (root/"images"/f"{i}.jpg").is_file()]
    prep_jpg_missing = [i for i in split if not (prepared/"images"/f"{i}.jpg").is_file()]
    prep_png_missing = [i for i in split if not (prepared/"images"/f"{i}.png").is_file()]
    out["categories"][cat] = {
        "split_val_images": len(split), "split_duplicates": len(split)-len(set(split)),
        "caption_rows": len(caps),
        "refs": len({c["candidate"] for c in caps}), "targets": len({c["target"] for c in caps}),
        "pair_union": len({c["candidate"] for c in caps} | {c["target"] for c in caps}),
        "ids_referenced_but_absent_from_split": len(ids - set(split)),
        "val_split_png_missing": len(png_missing),
        "val_split_jpg_missing": len(jpg_missing),
        "val_split_png_missing_examples": png_missing[:5],
        "prepared_png_missing": len(prep_png_missing),
        "prepared_jpg_missing": len(prep_jpg_missing),
        "prepared_caption_rows": len(json.loads((prepared/"captions"/f"cap.{cat}.val.json").read_text())) if (prepared/"captions"/f"cap.{cat}.val.json").is_file() else None,
        "multi_positive_rows": sum(1 for c in caps if c["candidate"] in c.get("target", "")),
        "target_equals_reference": sum(1 for c in caps if c["candidate"] == c["target"]),
        "captions_per_row": sorted({len(c["captions"]) for c in caps}),
    }
out["union_of_three_val_splits"] = len(all_ids)
out["images_dir_png_count"] = sum(1 for _ in (root/"images").glob("*.png"))
out["resized_image_count"] = sum(1 for p in (root/"resized_image").rglob("*.jpg")) if (root/"resized_image").is_dir() else None
Path("/home/heheboiz/data/cir/workbench/docs/reviews/20261110-e2e-repair/raw/membership.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
