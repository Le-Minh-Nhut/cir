"""Warm the pinned dataset constructor's cache OFF the GPU clock, so a timed GPU run measures encoding.

Why: every replay runs the native constructor, and for the AIRKNOW/TG-CIR/PAIR layouts that means
decoding and transforming every query and gallery image before the first forward. Measured here, that
build is minutes of pure CPU per category, so a "GPU wall time" that includes it is not a GPU
measurement. This pass heats the constructor's own cold/warm branch by writing its cache files into a
directory the caller names, which the replay then reuses via `WORKBENCH_REPLAY_CACHE_ROOT`.

What it warms, and what it does not. A cache is a set of PICKLED TENSORS, so only a constructor that
writes them can be warmed: `AirKnow/datasets.py:34`, `TGCIR/datasets.py:24` and `PAIR/data/datasets.py:34`
save `fashion_iq_data.json` plus `test_{queries,targets}_<cat>.pkl` and load them on the next run.
**CLVC-Net's constructor writes none of them** (`CLVCNet/datasets.py:152` keeps its tensors in memory),
so `--model clvc_net` decodes and discards: this script reports `cache_reusable: false` with an empty
`cache_files` instead of implying a warm cache exists. The campaign note that "CLVC reuse was explained
by tensor reuse" was therefore unsupported -- there were never any files.

What it writes. For a warmable model, a manifest beside the tensors (`CACHE_MANIFEST_NAME`) recording
the model, checkpoint, transform descriptor, annotation paths+digests, gallery identity, state and the
per-tensor digests. The replay REFUSES a cache whose manifest does not match its own description, so
this pass is what turns "files are present" into "files are provably these files".
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.replay.common import (  # noqa: E402
    CACHE_MANIFEST_NAME,
    CACHE_WARM_SENTINEL,
    CacheManifestError,
    cache_requirement,
    cache_tensor_digests,
    compare_cache_manifest,
    describe_cache_inputs,
    gallery_identity,
    read_cache_manifest,
    sha256_file,
    transform_descriptor,
    write_cache_manifest,
)

#: Constructors that write the pickled tensors a later run can load. Everything else decodes and
#: discards, and saying so is the point: a warm pass that lists no files is not a warm cache.
TENSOR_WRITING_MODELS = ("airknow", "tgcir", "pair")
DEFAULT_TRANSFORM = {
    "airknow": "data_utils.targetpad_transform(1.25, 224)",
}

def _transform_for(model: str):
    """The transform each evaluator uses, constructed from its own pinned code."""
    if model == "tgcir":
        import clip.clip as clip_clip
        return clip_clip._transform(224)      # exactly what clip.load(...).preprocess is
    if model == "clvc_net":
        import torchvision.transforms as transforms
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    if model == "airknow":
        import data_utils
        return data_utils.targetpad_transform(1.25, 224)
    import open_clip
    _, _, transform = open_clip.create_model_and_transforms("ViT-B-32")
    return transform

def _checkpoint_identity(model: str, variant: str | None) -> dict[str, str | None]:
    """The checkpoint id the REPLAY will require, taken from the same registry the replay reads."""
    from workbench.backend.registry import model_by_id

    variants = model_by_id(model)["checkpoint_variants"]
    selected = variants[0]
    if variant is not None:
        matches = [item for item in variants if item["checkpoint_id"] == variant]
        if not matches:
            raise SystemExit(f"unknown checkpoint variant for {model}: {variant}")
        selected = matches[0]
    return {"checkpoint_id": selected["checkpoint_id"],
            "expected_sha256": selected.get("expected_sha256")}

def _existing_manifest_verdict(cache_root: Path, requirement: dict) -> tuple[bool, list[str]]:
    """Is a cache already sitting in this directory provably the one this pass would build?"""
    if not (cache_root / CACHE_WARM_SENTINEL).is_file():
        return False, ["no warm sentinel: nothing to reuse"]
    if read_cache_manifest(cache_root) is None:
        return False, [f"no {CACHE_MANIFEST_NAME}: unverified"]
    return not compare_cache_manifest(cache_root, requirement), compare_cache_manifest(cache_root, requirement)

def _discard_tensors(cache_root: Path) -> list[str]:
    """Remove cache files that could not be verified, so the warm pass really rebuilds them.

    Deleting is the only honest option: leaving unverified pickles beside freshly written ones would
    let the constructor load a foreign tensor for a category the new pass did not overwrite.
    """
    removed = []
    for path in sorted(cache_root.glob("*.pkl")) + sorted(cache_root.glob(CACHE_WARM_SENTINEL)):
        if path.is_file():
            path.unlink()
            removed.append(path.name)
    (cache_root / CACHE_MANIFEST_NAME).unlink(missing_ok=True)
    return removed

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True,
                        choices=["tgcir", "clvc_net", "pair", "airknow"])
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--categories", nargs="+", default=["dress", "shirt", "toptee"])
    parser.add_argument("--variant", default=None, help="checkpoint variant the replay will require")
    parser.add_argument("--seed", default=None,
                        help="the seed the replay will run under; omit if the replay will be unseeded")
    parser.add_argument("--noise-ratio", type=float, default=0.0)
    parser.add_argument("--eval-mode", action="store_true",
                        help="the replay's eval mode; native Air-Know is train mode")
    args = parser.parse_args(argv)

    # The source root must be importable before any flat module of it is referenced; AirKnow's
    # transform selector imports `data_utils`, which lives at the checkout root.
    for entry in (str(args.source_root), str(args.source_root / "lavis")):
        if entry not in sys.path:
            sys.path.insert(0, entry)

    transform = _transform_for(args.model)
    args.cache_root.mkdir(parents=True, exist_ok=True)
    for name in ("captions", "image_splits", "resized_image"):
        link = args.cache_root / name
        source = args.dataset_root / name
        if not source.exists():
            raise SystemExit(f"native replay input missing: {source}")
        if not link.exists():
            link.symlink_to(source.resolve(), target_is_directory=True)
        elif link.resolve() != source.resolve():
            raise CacheManifestError(args.cache_root, [
                f"annotations.{name}.resolved_path: {str(source.resolve())!r} != "
                f"{str(link.resolve())!r}"])

    requirement = cache_requirement(
        args.dataset_root, model_id=args.model, transform=transform,
        checkpoint=_checkpoint_identity(args.model, args.variant),
        state={"seed": args.seed, "eval_mode": bool(args.eval_mode),
               "noise_ratio": args.noise_ratio})

    reports: dict[str, object] = {
        "model": args.model,
        "cache_root": str(args.cache_root),
        "requested_transform": transform_descriptor(transform),
    }

    if args.model not in TENSOR_WRITING_MODELS:
        # Nothing is written that a later run can load, so this script does not claim to warm a cache.
        reports.update({
            "cache_reusable": False,
            "cache_files": [],
            "reason": (f"{args.model} constructor persists no tensors (CLVCNet/datasets.py:152 keeps "
                       f"them in memory), so no cache exists for a replay to reuse"),
        })
        return 0

    already_usable, verdict = _existing_manifest_verdict(args.cache_root, requirement)
    if already_usable:
        reports.update({"cache_files": sorted(path.name for path in args.cache_root.iterdir()
                                              if path.is_file()),
                        "reused_existing": True,
                        "cache_reusable": True})
        print(json.dumps(reports, indent=2))
        return 0
    discarded = _discard_tensors(args.cache_root)
    if discarded:
        reports["discarded_unverified"] = discarded

    import datasets

    root_str = str(args.cache_root).rstrip("/") + "/"
    t0 = time.time()
    dataset = datasets.FashionIQ(path=root_str, transform=transform)
    total = sum(len(getattr(dataset, f"test_{kind}_{category}"))
                for category in args.categories for kind in ("queries", "targets"))
    wall = time.time() - t0

    tensors = cache_tensor_digests(args.cache_root)
    produced = sorted(tensors)
    expected = sorted([CACHE_WARM_SENTINEL]
                      + [f"test_{kind}_{category}.pkl"
                         for category in args.categories for kind in ("queries", "targets")])
    missing = [name for name in expected if name not in tensors]

    # The manifest records what was ACTUALLY produced, and only after the files exist.
    manifest = {
        **requirement,
        "annotations_requested": describe_cache_inputs(args.dataset_root),
        "gallery": gallery_identity(args.dataset_root, tuple(args.categories)),
        "tensors": tensors,
        "cache_files": produced,
        "cache_reusable": not missing,
        "written_by": {
            "script": "workbench/scripts/warm_replay_cache.py",
            "version": 1,
            "warm_wall_s": round(wall, 1),
        },
    }
    manifest_path = write_cache_manifest(args.cache_root, manifest)
    reports.update({
        "warm_wall_s": round(wall, 1),
        "model_inputs_materialised": total,
        "cache_reusable": not missing,
        "cache_files": produced,
        "missing_cache_files": missing,
        "cache_manifest": str(manifest_path),
        "cache_manifest_digest": sha256_file(manifest_path),
        "verified_existing_cache": verdict,
    })
    print(json.dumps(reports, indent=2))
    return 0 if not missing else 1

if __name__ == "__main__":
    raise SystemExit(main())
