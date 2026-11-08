"""G2 decode probe: is the DECODE+TRANSFORM stage bit-identical across worker counts?

Measures the pixel matrix the decode stage produces, not the model output: PIL decode and the
image transform are CPU-only, so this answers the parallelism question without touching the GPU
(the cohort owns it). The transform is each family's OWN transform, loaded through the family's own
interpreter and its own source tree -- HABIT's lavis `targetpad_transform(1.25, 224)` and ENCODER's
`open_clip` `preprocess_val`.

Reference = 1 worker, default torch intra-op threads. Candidate = N workers with
`limit_torch_threads(N)`, exactly as the shipped adapters configure it.

Runs as a standalone script under a model interpreter:
    <model python> equivalence_decode.py --family habit --workers 1,2,4,8 --batch 1,32,64,256
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

OUT = REPO / "workbench/docs/reviews/20261108T153045-audit-equivalence"


def sample_paths(count: int) -> list[str]:
    """512 real gallery images in RELEASED GALLERY ORDER (the order the run encodes)."""
    config = json_dump = None
    from workbench.backend.operator_config import resolve_config
    from workbench.composition import data as composition_data

    config = resolve_config()
    data = composition_data.load_release(Path(config.WORKBENCH_CIRCUS_RELEASE_ROOT),
                                         verify_hashes=False)
    root = Path(config.FASHIONIQ_ROOT) / "images"
    names = []
    for identity in data.gallery.image_ids:
        stem = Path(identity).stem
        names.append(stem)
    # The released gallery names .jpg; the local FashionIQ root is flat .png. Same stem rule the
    # runner uses to resolve released identities against local files.
    chosen, seen = [], set()
    for stem in names:
        if stem in seen:
            continue
        path = None
        for extension in (".png", ".jpg", ".jpeg"):
            candidate = root / f"{stem}{extension}"
            if candidate.is_file():
                path = candidate
                break
        if path is not None:
            chosen.append(str(path))
            seen.add(stem)
        if len(chosen) >= count:
            break
    return chosen


def build(family: str):
    """Return (transform, transform_name) using the family's own source tree."""
    from workbench.backend.operator_config import resolve_config

    config = resolve_config()
    third_party = Path(config.WORKBENCH_THIRD_PARTY_ROOT)
    if family == "habit":
        source = third_party / "HABIT"
        for entry in (str(source), str(source / "lavis")):
            if entry not in sys.path:
                sys.path.insert(0, entry)
        import data_utils  # the author's module, found on the path above

        return data_utils.targetpad_transform(1.25, 224), "lavis targetpad_transform(1.25, 224)"
    if family == "encoder":
        source = third_party / "ENCODER"
        import open_clip

        _, _, preprocess_val = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained=str(source / "open_clip_pytorch_model.bin"))
        return preprocess_val, "open_clip preprocess_val (ViT-B-32)"
    raise SystemExit(f"unknown family {family!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", required=True, choices=["habit", "encoder"])
    parser.add_argument("--count", type=int, default=512)
    parser.add_argument("--workers", default="1,2,4,8")
    parser.add_argument("--batch", default="1,32,64,256")
    parser.add_argument("--out", default="")
    parser.add_argument("--workers-as-threads", action="store_true", default=True)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from workbench.composition.adapters import prefetch_rows

    transform, transform_name = build(args.family)
    paths = sample_paths(args.count)
    workers = [int(item) for item in args.workers.split(",")]
    batches = [int(item) for item in args.batch.split(",")]

    def decode(chunk):
        images = []
        for path in chunk:
            with Image.open(path) as image:
                if args.family == "encoder":
                    images.append(transform(image.copy()))
                else:
                    images.append(transform(image.convert("RGB")))
        return torch.stack(images)

    results = []
    for batch in batches:
        chunks = [paths[start:start + batch] for start in range(0, len(paths), batch)]
        reference: np.ndarray | None = None
        for worker_count in workers:
            # the shipped configuration: torch intra-op threads pinned to the worker count
            torch.set_num_threads(max(1, worker_count) if args.workers_as_threads else torch.get_num_threads())
            collected = []
            for pixels in prefetch_rows(decode, chunks, worker_count):
                collected.append(pixels.numpy())
            matrix = np.concatenate(collected, axis=0).astype(np.float32, copy=False)
            if reference is None:
                reference = matrix
                delta_max, differing, max_rel = 0.0, 0, 0.0
                digest = hashlib.sha256(matrix.tobytes()).hexdigest()
            else:
                delta = np.abs(matrix.astype(np.float64) - reference.astype(np.float64))
                delta_max = float(delta.max())
                differing = int(np.count_nonzero(
                    matrix.view(np.uint8).reshape(len(matrix), -1) !=
                    reference.view(np.uint8).reshape(len(reference), -1)))
                denom = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
                max_rel = float((delta / denom).max())
                digest = hashlib.sha256(matrix.tobytes()).hexdigest()
            results.append({"batch": batch, "workers": worker_count,
                            "torch_threads": torch.get_num_threads(),
                            "images": len(paths), "shape": list(matrix.shape),
                            "sha256": digest, "max_abs_delta": delta_max,
                            "max_rel_delta": max_rel,
                            "bytes_differing": differing,
                            "bit_identical": differing == 0 and delta_max == 0.0})
            print(json.dumps(results[-1]), flush=True)
    payload = {"family": args.family, "transform": transform_name,
               "interpreter": sys.executable, "torch": torch.__version__,
               "numpy": np.__version__, "images": len(paths), "results": results}
    destination = Path(args.out) if args.out else OUT / f"decode_{args.family}.json"
    destination.write_text(json.dumps(payload, indent=1, sort_keys=True))
    print(json.dumps({"out": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
