"""Warm the pinned dataset constructor's cache OFF the GPU clock, so a timed GPU run measures encoding.

Why: every replay runs the native constructor, and for the ilearn/CLVC layouts that means decoding and
transforming every query and gallery image before the first forward. Measured here, that build is
minutes of pure CPU per category, so a "GPU wall time" that includes it is not a GPU measurement. This
pass heats the constructor's own cold/warm branch by writing its cache files into a directory the caller
names, which the replay then reuses via `WORKBENCH_REPLAY_CACHE_ROOT`
(`workbench/replay/common.py:94,CACHE_ROOT_ENV`).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=["tgcir", "clvc_net", "pair", "airknow"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["dress", "shirt", "toptee"])
    args = parser.parse_args(argv)

    # The source root must be importable before any flat module of it is referenced; AirKnow's
    # transform selector imports `data_utils`, which lives at the checkout root.
    for entry in (str(args.source_root), str(args.source_root / "lavis")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    # The transform must be the one the evaluator uses, or the cache describes different pixels.
    if args.model == "tgcir":
        import clip.clip as clip_clip
        transform = clip_clip._transform(224)      # exactly what clip.load(...).preprocess is
    elif args.model == "clvc_net":
        import torchvision.transforms as transforms
        transform = transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    elif args.model == "airknow":
        # The evaluator's own transform (`replay/airknow.py:174`: targetpad, ratio 1.25, 224 px).
        import data_utils
        transform = data_utils.targetpad_transform(1.25, 224)
    else:
        import open_clip
        _, _, transform = open_clip.create_model_and_transforms("ViT-B-32")

    args.cache_root.mkdir(parents=True, exist_ok=True)
    for name in ("captions", "image_splits", "resized_image"):
        link = args.cache_root / name
        if not link.exists():
            link.symlink_to((args.dataset_root / name).resolve(), target_is_directory=True)

    import datasets

    root_str = str(args.cache_root).rstrip("/") + "/"
    t0 = time.time()
    if args.model == "clvc_net":
        # `FashionIQ` here is per-category and builds its query list lazily; touch it so the decode
        # actually happens rather than only the JSON reads.
        total = 0
        for category in args.categories:
            ts = datasets.FashionIQ(path=root_str, gallery_all=False, name=category,
                                    split="val", transform=transform)
            total += len(ts.get_test_queries()) + len(ts.get_test_targets())
    else:
        dataset = datasets.FashionIQ(path=root_str, transform=transform)
        total = sum(len(getattr(dataset, f"test_{kind}_{c}"))
                    for c in ("dress", "shirt", "toptee") for kind in ("queries", "targets"))
    wall = time.time() - t0
    print(json.dumps({"model": args.model, "cache_root": str(args.cache_root),
                      "warm_wall_s": round(wall, 1), "model_inputs_materialised": total,
                      "cache_files": sorted(p.name for p in args.cache_root.glob("*") if p.is_file())},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
