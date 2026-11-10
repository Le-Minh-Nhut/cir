"""Source-faithful replay for the CLVC-Net FashionIQ evaluator.

Why a wrapper exists: the pinned `test.py` defines `test(params, local_model, global_model, testset, dataname)`
and nothing else — no CLI, no `__main__`, and no checkpoint loader. This module supplies the
missing orchestration shell and calls pinned `test.test(params, local_model, global_model, testset, dataname)`
without altering ranking, metric arithmetic, gallery construction, or feature extraction.

Exact signature and attributes read by pinned `test.py:6-149`:
- Signature: `def test(params, local_model, global_model, testset, dataname)` (5 arguments)
- `params.batch_size`: int (lines 96, 112; batch accumulation threshold)
- `local_model.eval()`: method (line 81)
- `global_model.eval()`: method (line 82)
- `local_model.compose_img_text(imgs, mods)[1]`: tensor (line 100)
- `global_model.compose_img_text(imgs, mods)[0]`: tensor (line 100)
- `local_model.extract_img_feature(imgs)[1]`: tensor (line 116)
- `global_model.extract_img_feature(imgs)[1]`: tensor (line 116)
- `testset.get_test_queries()`: method (line 84)
- `testset.get_test_targets()`: method (line 85)
- `dataname`: str (line 148; category label 'dress', 'shirt', 'toptee')
"""
from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import (
    ReplayResult,
    run_cache_root,
    load_trusted_whole_model,
    macro_aggregate,
    sha256_file,
    source_pin_status,
)

CHECKPOINT_VARIANT = model_by_id("clvc_net")["checkpoint_variants"][0]

MODEL = model_by_id("clvc_net")
PROTOCOL = "fashioniq_val_split"

# The pinned trainer saves whole pickled modules per category: local_model.pt and global_model.pt
# (train.py:274-275). 6 files total across 3 categories.
EXPECTED_CATEGORY_FILES = {
    cat: (f"{cat}_local_model.pt", f"{cat}_global_model.pt")
    for cat in CATEGORIES
}


def validate_checkpoint(checkpoint_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    """Bind the six staged files to the registry's own declaration.

    The registry already records `checkpoint_mapping_status: VERIFIED_METADATA` plus one
    `expected_sha256` per member, and all six are reproducible from the staged bytes
    (`bench` evidence in the audit). Emitting a hard-coded "mapping unresolved" string regardless of
    those bytes made a verified bundle read as unverified, so the CLI could never reach the evaluator
    even when every declared digest matched. The check is now the digests themselves.
    """
    blockers: list[str] = []
    declared = {member["filename"]: member["expected_sha256"] for member in CHECKPOINT_VARIANT["bundle_members"]}
    for cat in categories:
        local_name, global_name = EXPECTED_CATEGORY_FILES[cat]
        for name in (local_name, global_name):
            path = checkpoint_root / name
            if not path.is_file() or path.stat().st_size == 0:
                blockers.append(f"CLVC-Net checkpoint missing or empty for {cat}: {path}")
                continue
            expected = declared.get(name)
            if not expected:
                blockers.append(f"CLVC-Net checkpoint has no declared digest: {name}")
                continue
            actual = sha256_file(path)
            if actual != expected:
                blockers.append(
                    f"CLVC-Net checkpoint digest mismatch for {name}: expected {expected[:16]}..., found {actual[:16]}...")
    if not blockers and CHECKPOINT_VARIANT.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(
            f"CLVC-Net checkpoint mapping is unresolved: registry status="
            f"{CHECKPOINT_VARIANT.get('checkpoint_mapping_status')}")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    return [] if ok else [f"CLVC-Net source pin invalid: {reason}"]


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.val.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"CLVC-Net dataset input missing: {dataset_root / relative}")
        cat_resized = dataset_root / "resized_image" / category
        if not cat_resized.is_dir():
            blockers.append(f"CLVC-Net dataset input missing: {cat_resized}")
    return blockers


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
) -> list[str]:
    return (
        validate_source(source_root)
        + validate_checkpoint(checkpoint_root, categories)
        + validate_dataset(dataset_root, categories)
    )


def _load_replay_modules(source_root: Path) -> tuple[Any, Any, Any]:
    import torch
    import torchvision.transforms as transforms

    source_str = str(source_root)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    datasets = __import__("datasets")
    test_module = __import__("test")
    return torch, datasets, test_module

def _cpu_device_patch(torch_module: Any) -> None:
    """ADAPTED RUNTIME: make the pinned `.cuda()` calls device-neutral for a CPU smoke.

    The evaluator hard-codes `.cuda()` at `test.py:28,47,99,115` and `text_model.py:96,108`.
    Nothing about weights, module structure, arithmetic or ranking changes -- only where tensors
    live. This is a device parameterisation, NOT the author's code path, and every artifact that
    uses it must say so.
    """
    if getattr(torch_module.Tensor.cuda, "_workbench_device_neutral", False):
        return
    original = torch_module.Tensor.cuda

    def cuda(self, *args, **kwargs):  # noqa: ANN001 - mirrors Tensor.cuda's signature loosely
        return self

    cuda._workbench_device_neutral = True
    cuda._workbench_original = original
    torch_module.Tensor.cuda = cuda


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    batch_size: int = 32,
    device: str | None = None,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, datasets, test_module = _load_replay_modules(source_root)
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    adapted = target_device == "cpu"
    if adapted:
        _cpu_device_patch(torch)

    import torchvision.transforms as transforms
    # The NATIVE FashionIQ validation pipeline, copied from `train.py:90-96` (`load_dataset`, the
    # `args.dataset == 'fashioniq'` branch, `split='val'`):
    #
    #     Resize(256) -> CenterCrop(224) -> ToTensor -> ImageNet Normalize
    #
    # This wrapper used `Resize(224)`, which is the *fashion200k* branch (`train.py:113-117`) rather
    # than the FashionIQ one. Both produce a (3, 224, 224) tensor, so nothing failed visibly: the
    # short side was scaled to 224 and then center-cropped to 224, i.e. a tighter crop of a smaller
    # resample than the evaluator these weights were validated under. On a 300x500 fixture the two
    # pipelines differ by up to 4.24 in normalized units while matching in shape, which is exactly
    # the kind of mismatch that shows up as an unexplained Recall gap rather than an error.
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    results: list[ReplayResult] = []
    # See `replay/tgcir.py`: a warm cache is reused only when the caller names a root that
    # `warm_replay_cache.py` built from the same pinned annotations.
    with run_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        params = SimpleNamespace(batch_size=batch_size)

        for category in categories:
            local_name, global_name = EXPECTED_CATEGORY_FILES[category]
            local_path = checkpoint_root / local_name
            global_path = checkpoint_root / global_name
            declared = {member["filename"]: member["expected_sha256"]
                        for member in CHECKPOINT_VARIANT["bundle_members"]}

            # Whole pickled modules: unpickling is authorized only against the declared digest.
            local_model = load_trusted_whole_model(local_path, declared[local_name], torch)
            global_model = load_trusted_whole_model(global_path, declared[global_name], torch)
            local_model = local_model.to(target_device)
            global_model = global_model.to(target_device)
            local_model.eval()
            global_model.eval()

            testset = datasets.FashionIQ(
                path=root_str,
                gallery_all=False,
                name=category,
                split="val",
                transform=transform,
            )

            recalls = dict(test_module.test(params, local_model, global_model, testset, category))
            results.append(
                ReplayResult(
                    model_id="clvc_net",
                    checkpoint_id=f"author_drive_{category}",
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={
                        "local_checkpoint": str(local_path),
                        "global_checkpoint": str(global_path),
                        "source_commit": MODEL["upstream_commit_sha"],
                        "native_evaluator": "test.py:test",
                        "runtime_label": "ADAPTED_DEVICE_NEUTRAL" if adapted else "NATIVE_CUDA",
                        "device": target_device,
                        "adaptation": (
                            "torch.Tensor.cuda() neutralised so the pinned `.cuda()` sites "
                            "(test.py:28,47,99,115; text_model.py:96,108) run on CPU; weights, "
                            "architecture, arithmetic and ranking untouched"
                        ) if adapted else None,
                    },
                )
            )

    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    selected_categories = tuple(args.categories) if args.categories else CATEGORIES
    blockers = preflight(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        selected_categories,
    )
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if args.preflight_only:
        return 0

    results, aggregate = evaluate_categories(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        categories=selected_categories,
        batch_size=args.batch_size,
        device=args.device,
    )
    payload = {
        "model_id": "clvc_net",
        "protocol": PROTOCOL,
        "results": [asdict(r) for r in results],
        "aggregate": aggregate,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
