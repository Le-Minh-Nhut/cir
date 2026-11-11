"""Preflight + workbench-added evaluation entrypoint for the pinned TG-CIR FashionIQ evaluator.

Two different claims live here and must not be merged:

**Preflight.** The registry declares `checkpoint_mapping_status: VERIFIED_METADATA` for
`Fashioniq.pt`, and the staged bytes reproduce its declared SHA-256 exactly. The pinned
`train.py:74` DOES declare `--restore_file` but never reads it (grep-verified: one occurrence in the
whole file, the argparse line), so the author ships no CLI that evaluates a released weight.

**The evaluation path below is WORKBENCH-ADDED.** It calls the pinned
`test.test(params, model, testset, category)` (`test.py:6`) unchanged, with the pinned
`datasets.FashionIQ` (`datasets.py:24`) and the pinned `clip.load(...).preprocess` transform the
trainer's own `load_dataset()` uses (`train.py:87,90`). It is not an author entrypoint and must never
be labelled native CLI.

Two load-time adaptations are unavoidable on modern torch and are recorded per artifact:

* `torch.nn.modules.linear._LinearWithBias` -> `torch.nn.Linear`. The 1.7-era pickle names a class
  modern torch removed; the alias restores the same parameter layout (no renames, no dropped keys).
* `batch_first=False` on the 24 unpickled `nn.MultiheadAttention` modules. Torch 1.7 states carried no
  such attribute and its `forward` was batch-first-first only; modern torch reads the attribute.
  This is the exact failure the historical `tgcir_run1.log` records
  (`AttributeError: 'MultiheadAttention' object has no attribute 'batch_first'`, EXIT=1).
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
    cache_requirement,
    run_cache_root,
    macro_aggregate,
    sha256_file,
    source_pin_status,
)

MODEL = model_by_id("tgcir")
CHECKPOINT = MODEL["checkpoint_variants"][0]
PROTOCOL = MODEL["native_protocol"]
RAW_INPUTS = tuple(
    Path(path)
    for category in CATEGORIES
    for path in (
        f"captions/cap.{category}.train.json",
        f"captions/cap.{category}.val.json",
        f"image_splits/split.{category}.val.json",
    )
)
CORRECTION_DICTIONARIES = tuple(Path(f"captions/correction_dict_{category}.json") for category in CATEGORIES)
RESIZED_IMAGE_DIRS = tuple(Path(f"resized_image/{category}") for category in CATEGORIES)
# The pinned `datasets.FashionIQ.__init__` (`datasets.py:34`) takes a cold path when
# `fashion_iq_data.json` is absent and writes these beside it; the warm path trusts them blindly.
NATIVE_CACHES = (
    Path("fashion_iq_data.json"),
    *(Path(f"test_{kind}_{category}.pkl") for category in CATEGORIES for kind in ("queries", "targets")),
)


def validate_checkpoint(checkpoint_root: Path) -> list[str]:
    """Bind the staged weight to the declared digest; say where that digest came from.

    The registry's `expected_sha256` for this variant is a digest computed over the staged archive
    member by the workbench. The author publishes `Fashioniq_metrics.json` beside the weight but no
    checksum file, and the Google Drive download is not digest-addressable, so this is a STAGED/locAL
    digest, not an author-published one. Both statements are returned so a caller cannot mistake the
    stronger claim for the weaker evidence.
    """
    blockers: list[str] = []
    candidate = checkpoint_root / CHECKPOINT["filename"]
    if not candidate.is_file() or candidate.stat().st_size == 0:
        blockers.append(f"TG-CIR checkpoint missing or empty: {candidate}")
        return blockers
    expected_size = CHECKPOINT.get("expected_size_bytes")
    if expected_size and candidate.stat().st_size != expected_size:
        blockers.append(
            f"TG-CIR checkpoint size mismatch: expected {expected_size}, found {candidate.stat().st_size}")
    expected_sha = CHECKPOINT.get("expected_sha256")
    if not expected_sha:
        blockers.append("TG-CIR checkpoint has no declared digest; whole-model pickle load is refused")
    else:
        actual = sha256_file(candidate)
        if actual != expected_sha:
            blockers.append(
                f"TG-CIR checkpoint digest mismatch: expected {expected_sha[:16]}..., found {actual[:16]}...")
    if CHECKPOINT.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(
            f"TG-CIR checkpoint mapping unresolved: status={CHECKPOINT.get('checkpoint_mapping_status')}")
    return blockers


def preflight(source_root: Path, dataset_root: Path, checkpoint_root: Path) -> list[str]:
    """Return all known blockers without importing upstream code or mutating inputs."""
    blockers = []
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    if not ok:
        blockers.append(f"pinned TG-CIR source unavailable: {reason}")

    blockers.extend(f"FashionIQ raw input missing: {dataset_root / path}" for path in RAW_INPUTS if not (dataset_root / path).is_file())
    blockers.extend(
        f"TG-CIR resized-image directory missing: {dataset_root / path}"
        for path in RESIZED_IMAGE_DIRS
        if not (dataset_root / path).is_dir()
    )
    blockers.extend(
        f"pinned-source correction dictionary missing: {dataset_root / path}"
        for path in CORRECTION_DICTIONARIES
        if not (dataset_root / path).is_file()
    )

    # A cold cache is the evaluable state: the constructor rebuilds it from the pinned annotations.
    # A warm cache is a provenance question, not a blocker. Two separate facts matter here, and the
    # earlier version of this comment merged them wrongly:
    #
    # * files sitting in the DATASET root are never read as a cache -- `run_cache_root` builds a
    #   different root and links only the three input directories into it -- so they cannot influence
    #   a run at all, and this stays a provenance warning rather than a blocker;
    # * a root that IS used (`WORKBENCH_REPLAY_CACHE_ROOT`) is now verified against a manifest before
    #   the constructor reads it (`replay/common.py::reusable_cache_root`), so "unverified provenance"
    #   is no longer the reason to delete anything -- a mismatch is a refusal, not a silent load.
    present = [dataset_root / path for path in NATIVE_CACHES if (dataset_root / path).is_file()]
    if present:
        blockers.append(
            "TG-CIR warm native cache present in the dataset root; it is not the root this replay "
            "reads (a fresh root is built and the input directories are linked into it), so it "
            "cannot affect these numbers -- delete it to keep the tree unambiguous: "
            + ", ".join(str(p) for p in present))

    blockers.extend(validate_checkpoint(checkpoint_root))

    clip_cache = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
    if not clip_cache.is_file():
        blockers.append(f"OpenAI CLIP ViT-B/16 asset missing: {clip_cache}")
    else:
        # OpenAI publishes no checksum FILE, but `clip.clip._MODELS` addresses the weight by a URL
        # whose path segment is the file's own SHA-256, so the digest is author-published inside the
        # pinned code. That makes the cache verifiable rather than merely "present".
        expected = clip_weight_digest()
        actual = sha256_file(clip_cache)
        if expected and actual != expected:
            blockers.append(
                f"CLIP ViT-B/16 cache digest mismatch: pinned URL names {expected[:16]}..., "
                f"cache has {actual[:16]}... ({clip_cache})")
        elif not expected:
            blockers.append(
                "CLIP ViT-B/16 digest unverifiable: the pinned clip module exposes no digest-addressed URL")
    return blockers


def clip_weight_digest() -> str | None:
    """The SHA-256 OpenAI embeds in its own download URL for ViT-B/16, read from the pinned module."""
    import re

    try:
        import clip.clip as clip_clip
    except Exception:  # the module is only importable inside a model env
        return None
    url = clip_clip._MODELS.get("ViT-B/16", "")
    match = re.search(r"/([0-9a-f]{64})/", url)
    return match.group(1) if match else None


def _load_tgcir(source_root: Path, device: str) -> tuple[Any, Any, Any, str, str]:
    """Import the pinned modules and load the released weight. Returns (torch, client, model, ...)."""
    import torch

    source_str = str(source_root)
    if source_str not in sys.path:
        sys.path.insert(0, source_str)

    adaptations: list[str] = []
    import torch.nn.modules.linear as torch_linear

    if not hasattr(torch_linear, "_LinearWithBias"):
        torch_linear._LinearWithBias = torch.nn.Linear
        adaptations.append("torch.nn.modules.linear._LinearWithBias -> torch.nn.Linear (1.7 pickle alias)")

    import clip  # the pinned OpenAI CLIP revision, installed into the model env
    import datasets
    import test as tg_test

    return torch, (clip, datasets, tg_test), adaptations


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    batch_size: int = 16,
    device: str | None = None,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, (clip, datasets, tg_test), adaptations = _load_tgcir(source_root, device or "")
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = checkpoint_root / CHECKPOINT["filename"]

    model = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    unset = [m for m in model.modules()
             if isinstance(m, torch.nn.MultiheadAttention) and not hasattr(m, "batch_first")]
    for module in unset:
        module.batch_first = False
    if unset:
        adaptations.append(f"batch_first=False restored on {len(unset)} nn.MultiheadAttention modules (torch 1.7 state)")

    if target_device == "cpu":
        # ADAPTED RUNTIME: `model.py:32,107,120,136,244,251` and `test.py:31,47,101,115` call
        # `.cuda()` unconditionally. Loss-only sites (244,251) are never reached by `test()`.
        torch.Tensor.cuda = lambda self, *a, **k: self  # type: ignore[assignment]
        adaptations.append("torch.Tensor.cuda neutralised for a CPU run; weights/arithmetic unchanged")

    model = model.to(target_device).eval()
    clip_model, preprocess = clip.load("ViT-B/16", device=target_device, jit=False)
    params = SimpleNamespace(batch_size=batch_size, local_rank=0)

    provenance_common = {
        "source_commit": MODEL["upstream_commit_sha"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": CHECKPOINT.get("expected_sha256"),
        "checkpoint_digest_kind": "STAGED_WORKBENCH_DIGEST (author publishes metrics json, not a checksum)",
        "native_evaluator": "test.py:test",
        "evaluation_path": "WORKBENCH_ADDED_EVAL_PATH (author ships no evaluable CLI; --restore_file is dead)",
        "img_encoder": "ViT-B/16",
        "transform": "clip.load('ViT-B/16').preprocess (train.py:87,90)",
        "runtime_label": "ADAPTED_RUNTIME" if target_device == "cpu" else "NATIVE_CUDA",
        "adaptations": list(adaptations),
        "clip_cache": str(Path.home() / ".cache" / "clip" / "ViT-B-16.pt"),
    }

    results: list[ReplayResult] = []
    # `run_cache_root` honours WORKBENCH_REPLAY_CACHE_ROOT and otherwise behaves exactly like
    # `isolated_cache_root`. It exists because the pinned constructor materialises every query and
    # gallery image for all three categories before the first forward: measured warm 1 067 s CPU and
    # 14 635 inputs, which would otherwise be counted against GPU wall time.
    #
    # A shared root is reused ONLY when its manifest matches THIS run's description. That is the
    # check `reuse is sound because the root is built from the same pinned annotations` used to
    # assume: presence was never proof, and a warm root built from other annotations (or another
    # transform) loaded as if it were this one.
    cache_metadata: dict[str, Any] = {}
    with run_cache_root(
            dataset_root,
            requirement=cache_requirement(
                dataset_root, model_id="tgcir", transform=preprocess,
                checkpoint={"checkpoint_id": CHECKPOINT["checkpoint_id"],
                            "expected_sha256": CHECKPOINT.get("expected_sha256")},
                state={"seed": None, "eval_mode": True, "noise_ratio": 0}),
            record=cache_metadata) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        testset = datasets.FashionIQ(path=root_str, transform=preprocess)
        for category in categories:
            recalls = dict(tg_test.test(params, model, testset, category))
            results.append(
                ReplayResult(
                    model_id="tgcir",
                    checkpoint_id=CHECKPOINT["checkpoint_id"],
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={**provenance_common, "cache_manifest": dict(cache_metadata)},
                )
            )
    del clip_model
    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES, action="append", dest="categories")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)

    blockers = preflight(args.source_root, args.dataset_root, args.checkpoint_root)
    if blockers:
        for blocker in blockers:
            print(f"[BLOCKED] {blocker}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print("[READY] TG-CIR preflight passed; no inference executed")
        return 0

    selected = tuple(args.categories) if args.categories else CATEGORIES
    results, aggregate = evaluate_categories(
        args.checkpoint_root, args.dataset_root, args.source_root,
        categories=selected, batch_size=args.batch_size, device=args.device,
    )
    payload = {
        "model_id": "tgcir",
        "protocol": PROTOCOL,
        "evaluation_path": "WORKBENCH_ADDED_EVAL_PATH",
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
