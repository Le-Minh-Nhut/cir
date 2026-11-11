"""Source-faithful evaluation runner for CVPR26-Air-Know.

Why a wrapper exists: `test_BLIP2.py` defines callable routines without `__main__` or a checkpoint
loader, and its auxiliary LaSCo routines call `load_obj("")` with an empty path. This module supplies the
missing invocation shell and runs the native FashionIQ evaluator inside `isolated_cache_root`.

Native model behavior:
`test_BLIP2.py:10` natively sets `model.train()` at the top of `test()`. `model.eval()` is only called
for `fashion200k` (line 13) and `test_lasco` (line 220), leaving FashionIQ evaluation governed by
`model.train()`. An experimental eval-mode variant (`--eval-mode`) is provided and explicitly labelled.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
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

MODEL = model_by_id("airknow")
CHECKPOINTS = {variant["checkpoint_id"]: variant for variant in MODEL["checkpoint_variants"]}
# The processor config each `model_type` selects, taken from the pinned model class's own
# `PRETRAINED_MODEL_CONFIG_DICT`
# (`lavis/models/blip2_models/Blip2QformerCir.py`). Kept as the published mapping rather than a path
# invented here. Only `pretrain` is reachable through the adapter, which passes no other
# model_type, but the whole mapping is carried so a divergence is visible.
PROCESSOR_CONFIGS = {
    "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
    "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
    "coco": "configs/models/blip2/blip2_coco.yaml",
}

DEFAULT_VARIANT = "fiq_n05"
#: Set to make a run reproducible; unset is recorded as `null` rather than silently assumed.
SEED_ENV = "WORKBENCH_AIRKNOW_SEED"
DATASET_LABEL = "fashioniq"
PROTOCOL = "fashioniq_full_gallery_ref_excluded"
#: The transform's numeric identity, used BOTH to build the transform and to declare it, so the
#: description cannot drift from the code that produces the pixels.
TARGETPAD_RATIO = 1.25
TARGETPAD_DIM = 224
TRANSFORM_DESCRIPTOR = f"data_utils.targetpad_transform({TARGETPAD_RATIO}, {TARGETPAD_DIM})"

# Verified author HuggingFace LFS metadata (model repository iLearn-Lab/CVPR26-Air-Know)
KNOWN_AUTHOR_SHA256 = {
    "fiq_n05": "f4d25ed9e532f2ee3671f6edadb1048e73113246ec056b413bde75dfe3f79bc6",
    "fiq_n08": "756fc0bdc43d9e0f0cae13c8a02b0279f64e62092ce465c1e60bda45e179c54e",
}
KNOWN_AUTHOR_SIZES = {
    "fiq_n05": 2724056638,
    "fiq_n08": 2724056638,
}

#: The metric the variance block describes. Spread is only meaningful on the aggregate the paper
#: quotes, which is macro_r10; per-category values stay in the `categories` rows.
VARIANCE_METRIC = "macro_r10"

#: The historical two-run figure the campaign quotes in `BLOCKED_MODELS_VERIFICATION.vi.md` (A-12).
#: It is NOT this run's error bar and is emitted only as a labelled historical note. Keeping it as a
#: dict makes the two required halves inseparable: a number without `applies_to_this_run: false`
#: cannot be read as a per-run delta.
HISTORICAL_CAMPAIGN_DELTA = {
    "value": 0.4744,
    "statistic": "absolute macro_r10 difference between two identical native-mode runs",
    "applies_to_this_run": False,
    "provenance": "docs/reviews/20261111-master-audit/BLOCKED_MODELS_VERIFICATION.vi.md (A-12); "
                  "two draws of fiq_n05, not a measurement of this artifact",
}

def run_identity(*, variant_id: str, eval_mode: bool, protocol: str, batch_size: int,
                 transform: str) -> dict[str, Any]:
    """Everything that must be equal for two runs' numbers to be poolable. Order-independent."""
    return {
        "model_id": "airknow",
        "checkpoint_id": variant_id,
        "eval_mode": bool(eval_mode),
        "protocol": protocol,
        "batch_size": int(batch_size),
        "transform": transform,
    }

def variance_group_key(identity: dict[str, Any]) -> str:
    """A stable name for the set of runs a statistic covers.

    Derived from the config identity rather than from the clock, so two runs on different hosts and
    different days that measured the same thing share a key, and a run with a different checkpoint,
    mode or transform cannot be handed a statistic computed from another population.
    """
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return "airknow/" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

def _comparable_identity(document: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Read a committed artifact's config identity, or say why it cannot be compared."""
    if not isinstance(document, dict):
        return None, "not a JSON object"
    if document.get("model_id") != "airknow":
        return None, f"different model_id: {document.get('model_id')!r}"
    runtime = document.get("runtime")
    if not isinstance(runtime, dict):
        return None, "no runtime block"
    missing = [name for name in ("variant", "eval_mode", "batch_size", "transform")
               if name not in runtime]
    if missing:
        # Deliberately NOT guessed. A sibling that did not record its batch size or transform might
        # have measured the same population and might not; pooling it would be the R15 defect again
        # with extra steps.
        return None, "identity not recorded (missing runtime." + ", runtime.".join(missing) + ")"
    identity = run_identity(
        variant_id=runtime["variant"], eval_mode=runtime["eval_mode"],
        protocol=document.get("protocol"), batch_size=runtime["batch_size"],
        transform=runtime["transform"])
    return identity, None

def _macro_r10_of(document: Any) -> float | None:
    if not isinstance(document, dict):
        return None
    aggregate = document.get("aggregate")
    if not isinstance(aggregate, dict):
        return None
    value = aggregate.get(VARIANCE_METRIC)
    return float(value) if isinstance(value, (int, float)) else None

def assemble_variance_group(directory: Path | None, identity: dict[str, Any],
                            macro_r10: float | None, *,
                            timestamp: str | None = None,
                            exclude_path: Path | None = None) -> dict[str, Any]:
    """Describe THIS run's position in its variance group, computing the group's statistics.

    Called at emit time with the directory the artifact is written into, so the statistics cover the
    draws that exist beside it instead of a campaign constant. Three rules:

    * a statistic exists only when the group really has >= 2 runs of the SAME identity;
    * a group never mixes native (train, live dropout) with eval mode -- the constructor of the group
      refuses it rather than reporting one number over two populations;
    * a sibling whose identity was not recorded is listed as excluded, not silently pooled.
    """
    macro_r10 = None if macro_r10 is None else float(macro_r10)
    self_entry = {"path": None, "macro_r10": macro_r10, "timestamp": timestamp, "self": True}
    skip = exclude_path.resolve() if exclude_path is not None else None
    members: list[dict[str, Any]] = [self_entry]
    excluded: list[dict[str, str]] = []
    if directory is not None and Path(directory).is_dir():
        for path in sorted(Path(directory).glob("*.json")):
            if skip is not None and path.resolve() == skip:
                # Re-writing an artifact in place: its previous contents are this run's own older
                # version, not an independent draw.
                excluded.append({"path": str(path), "reason": "this run's own output file"})
                continue
            try:
                document = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                excluded.append({"path": str(path), "reason": f"unreadable: {error}"})
                continue
            sibling, reason = _comparable_identity(document)
            if sibling is None:
                excluded.append({"path": str(path), "reason": reason or "no identity"})
                continue
            differing = sorted(name for name in identity if identity[name] != sibling.get(name))
            if differing == ["eval_mode"]:
                # Same checkpoint, protocol, batch and transform, but one run has live dropout and
                # the other does not: pooling them reports a spread over two different experiments.
                raise ValueError(
                    "Air-Know variance group mixes eval modes: "
                    f"{path} has eval_mode={sibling['eval_mode']}, this run has "
                    f"{identity['eval_mode']}; refusing to pool train-mode and eval-mode draws")
            if differing:
                excluded.append({"path": str(path),
                                 "reason": "different config: " + ", ".join(differing)})
                continue
            value = _macro_r10_of(document)
            if value is None:
                excluded.append({"path": str(path), "reason": f"no aggregate.{VARIANCE_METRIC}"})
                continue
            members.append({"path": str(path), "macro_r10": value,
                            "timestamp": document.get("timestamp"), "self": False,
                            "eval_mode": sibling["eval_mode"]})

    self_entry["eval_mode"] = identity["eval_mode"]
    modes = {bool(member["eval_mode"]) for member in members}
    if len(modes) > 1:
        raise ValueError(
            "Air-Know variance group mixes eval modes (train-mode dropout is live in one and not the "
            f"other); refusing to pool: {sorted(modes)}")

    ordered = sorted(members, key=lambda item: (str(item["timestamp"] or ""), str(item["path"] or "")))
    values = [item["macro_r10"] for item in ordered if item["macro_r10"] is not None]
    repeat_index = next((index + 1 for index, item in enumerate(ordered) if item["self"]), 1)
    repeat_count = len(ordered)
    # n counts only runs with a usable metric: an unmeasurable draw is not an extra sample.
    n = len(values)
    sd = statistics.stdev(values) if n >= 2 else None
    return {
        "variance_group_key": variance_group_key(identity),
        "identity": identity,
        "repeat_index": repeat_index,
        "repeat_count": repeat_count,
        "metric": VARIANCE_METRIC,
        "n": n,
        "mean": statistics.fmean(values) if values else None,
        "sd": sd,
        "spread": (max(values) - min(values)) if n else None,
        "insufficient_draws": sd is None,
        "value_of_this_run": macro_r10,
        "group_members": [str(item["path"] or "this run") for item in ordered],
        "group_excluded": excluded,
        "assembled_by": "workbench.replay.airknow.assemble_variance_group",
        "note": (
            "sd/spread are computed from the runs listed in group_members. n == 1 means no variance "
            "was measured for this configuration and sd is null -- it is never borrowed from another "
            "run, mode or campaign."
        ),
    }


def validate_checkpoint(checkpoint_root: Path, variant_id: str = DEFAULT_VARIANT) -> list[str]:
    blockers: list[str] = []
    variant = CHECKPOINTS.get(variant_id)
    if variant is None:
        blockers.append(f"unsupported Air-Know variant: {variant_id}; valid: {list(CHECKPOINTS.keys())}")
        return blockers
    path = checkpoint_root / variant["filename"]
    if not path.is_file():
        blockers.append(f"Air-Know checkpoint missing: {path}")
        return blockers
    if path.stat().st_size == 0:
        blockers.append(f"Air-Know checkpoint is empty placeholder: {path}")
        return blockers
    expected_size = variant.get("expected_size_bytes") or KNOWN_AUTHOR_SIZES.get(variant_id)
    if expected_size and path.stat().st_size != expected_size:
        blockers.append(f"Air-Know checkpoint size mismatch: expected {expected_size}, got {path.stat().st_size}")
    expected_sha = variant.get("expected_sha256") or KNOWN_AUTHOR_SHA256.get(variant_id)
    if expected_sha:
        actual_sha = sha256_file(path)
        if actual_sha != expected_sha:
            blockers.append(f"Air-Know checkpoint SHA-256 mismatch: expected {expected_sha[:16]}..., got {actual_sha[:16]}...")
    if variant.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        blockers.append(f"Air-Know {variant_id} checkpoint mapping is not verified")
    return blockers


def validate_source(source_root: Path) -> list[str]:
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"])
    return [] if ok else [f"Air-Know source pin invalid: {reason}"]


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    for category in categories:
        for relative in (
            f"captions/cap.{category}.train.json",
            f"captions/cap.{category}.val.json",
            f"captions/correction_dict_{category}.json",
            f"image_splits/split.{category}.val.json",
        ):
            if not (dataset_root / relative).is_file():
                blockers.append(f"Air-Know dataset requirement missing: {dataset_root / relative}")
        if not (dataset_root / "resized_image" / category).is_dir():
            blockers.append(f"Air-Know dataset requirement missing: {dataset_root / 'resized_image' / category}")
    # The evaluated subset and the constructor's input set are different questions: the native
    # `datasets.FashionIQ(...)` opens every category's files regardless of `--category`.
    from workbench.replay.common import constructor_input_blockers

    return list(dict.fromkeys(
        blockers + constructor_input_blockers(
            dataset_root, lambda path: f"Air-Know dataset requirement missing: {path}")))


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    variant_id: str = DEFAULT_VARIANT,
) -> list[str]:
    return (
        validate_source(source_root)
        + validate_checkpoint(checkpoint_root, variant_id)
        + validate_dataset(dataset_root, categories)
    )


def _load_replay_modules(source_root: Path) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    for entry in (str(source_root), str(source_root / "lavis")):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    datasets = __import__("datasets")
    data_utils = __import__("data_utils")
    test_blip2 = __import__("test_BLIP2")
    from workbench.replay.lavis_processors import text_processors

    txt_processors, boundary = text_processors(source_root, PROCESSOR_CONFIGS["pretrain"])

    return torch, datasets, data_utils, test_blip2, (txt_processors, boundary)


def evaluate_categories(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    variant_id: str = DEFAULT_VARIANT,
    batch_size: int = 64,
    device: str | None = None,
    eval_mode: bool = False,
) -> tuple[list[ReplayResult], dict[str, Any]]:
    torch, datasets, data_utils, test_module, processors = _load_replay_modules(source_root)
    txt_processors, processor_boundary = processors
    variant = CHECKPOINTS[variant_id]
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = checkpoint_root / variant["filename"]
    expected_sha = variant.get("expected_sha256") or KNOWN_AUTHOR_SHA256.get(variant_id)

    model = load_trusted_whole_model(model_path, expected_sha, torch)
    model = model.to(target_device)
    if eval_mode:
        model.eval()
        # test_BLIP2.py:10 calls model.train(); neutralize for experimental eval-mode variant
        model.train = lambda *args, **kwargs: model
    else:
        # Native behavior: test_BLIP2.py:10 calls model.train()
        model.train()

    seed = os.environ.get(SEED_ENV)
    if seed:
        # Opt-in only: seeding changes the native behaviour, so it is a caller decision recorded in
        # the artifact rather than a default this wrapper imposes.
        import random as _random
        import numpy as _np
        torch.manual_seed(int(seed))
        _random.seed(int(seed))
        _np.random.seed(int(seed))

    transform = data_utils.targetpad_transform(TARGETPAD_RATIO, TARGETPAD_DIM)
    params = SimpleNamespace(batch_size=batch_size, dataset=DATASET_LABEL)
    precision = str(next(model.parameters()).dtype).replace("torch.", "")
    run_metadata: dict[str, Any] = {
        "model_mode": "eval (experimental variant)" if eval_mode else "train (native behavior)",
        "eval_mode": bool(eval_mode),
        "dropout_active": not eval_mode,
        "seed": seed or None,
        "device": str(target_device),
        "precision": precision,
        "batch_size": batch_size,
        "transform": TRANSFORM_DESCRIPTOR,
        "note": (
            "seed=null means nothing seeded this run: `test_BLIP2.py:10` calls model.train() for "
            "FashionIQ and blip2_cir.py:58,61 carry Dropout(0.1), so one artifact is ONE DRAW."
        ),
    }

    results: list[ReplayResult] = []
    # `run_cache_root` honours WORKBENCH_REPLAY_CACHE_ROOT and otherwise behaves exactly like
    # `isolated_cache_root`. It is required here: the pinned constructor materialises all three
    # categories' queries AND galleries before the first forward -- measured 21 552 inputs /
    # 12.98 GB of tensors / 14 957 MB peak RSS -- which is minutes of CPU that must not be counted
    # as GPU time, and which SIGKILLs the process when it overlaps another worker's memory.
    with run_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        dataset = datasets.FashionIQ(path=root_str, transform=transform, noise_ratio=0)

        for category in categories:
            recalls = dict(test_module.test(params, model, dataset, category, txt_processors))
            results.append(
                ReplayResult(
                    model_id="airknow",
                    checkpoint_id=variant["checkpoint_id"],
                    category=category,
                    protocol=PROTOCOL,
                    r1=recalls.get(f"{category}_r1"),
                    r10=recalls.get(f"{category}_r10"),
                    r50=recalls.get(f"{category}_r50"),
                    complete=True,
                    provenance={
                        "native_evaluator": "test_BLIP2.py:test",
                        "native_output": {k: recalls[k] for k in sorted(recalls)},
                        "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"],
                        "evaluation_noise_pct": 0,
                        "model_mode": "eval (experimental variant)" if eval_mode else "train (native behavior)",
                        "checkpoint_sha256": (
                            sha256_file(model_path) if model_path.is_file() and model_path.stat().st_size else None
                        ),
                        "checkpoint_size_bytes": model_path.stat().st_size if model_path.is_file() else None,
                        "preprocessing": TRANSFORM_DESCRIPTOR,
                        # A-12: the numbers above are ONE DRAW, not a value. The variance of this
                        # configuration is emitted by `main` as a GROUP-scoped block computed from the
                        # runs beside this artifact; nothing here claims a delta this run did not
                        # measure. `historical_campaign_delta_macro_r10` is a labelled note about the
                        # campaign's two-draw figure, not this artifact's error bar.
                        "determinism": {
                            **run_metadata,
                            "historical_campaign_delta_macro_r10": dict(HISTORICAL_CAMPAIGN_DELTA),
                            "variance_group": None,
                        },
                        "text_processor": {
                            "config": PROCESSOR_CONFIGS["pretrain"],
                            "loaded_via": processor_boundary,
                            "note": "the pinned BlipCaptionProcessor from the model class's own "
                                    "config; no second BLIP-2 model is constructed and no "
                                    "/root/bert-base-uncased asset is read",
                        },
                    },
                )
            )

    return results, macro_aggregate(results)


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--variant", default=DEFAULT_VARIANT, choices=list(CHECKPOINTS.keys()))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument(
        "--eval-mode",
        action="store_true",
        help="Experimental variant: neutralize model.train() at test_BLIP2.py:10 to evaluate in eval mode",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    categories = (args.category,) if args.category else CATEGORIES
    problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories, args.variant)
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        return 1
    if args.preflight_only:
        print(f"[READY] Air-Know preflight passed (variant={args.variant}); no inference executed")
        return 0

    results, aggregate = evaluate_categories(
        args.checkpoint_root,
        args.dataset_root,
        args.source_root,
        categories=categories,
        variant_id=args.variant,
        batch_size=args.batch_size,
        device=args.device,
        eval_mode=args.eval_mode,
    )
    variant = CHECKPOINTS[args.variant]
    timestamp = datetime.now(timezone.utc).isoformat()
    identity = run_identity(variant_id=args.variant, eval_mode=args.eval_mode, protocol=PROTOCOL,
                            batch_size=args.batch_size, transform=TRANSFORM_DESCRIPTOR)
    # The group is assembled HERE, at emit time, from the artifacts already in the output directory:
    # a statistic belongs to the set of runs it covers, not to whichever run happens to be written.
    variance = assemble_variance_group(
        args.output.parent if args.output else Path.cwd(), identity, aggregate.get(VARIANCE_METRIC),
        timestamp=timestamp, exclude_path=args.output)
    for result in results:
        determinism = result.provenance.get("determinism")
        if isinstance(determinism, dict):
            determinism["variance_group"] = variance
    # The device and precision the tensors actually ran on come from the evaluation, not from
    # re-deriving "auto" here.
    reported = next((r.provenance.get("determinism") or {} for r in results), {})
    output = {
        "model_id": "airknow",
        "replay_type": "SOURCE_FAITHFUL_WORKBENCH_REPLAY",
        "protocol": PROTOCOL,
        "checkpoint_training_noise_pct": variant["checkpoint_training_noise_pct"],
        "evaluation_noise_pct": 0,
        "eval_mode_variant": args.eval_mode,
        "determinism": {
            "eval_mode": bool(args.eval_mode),
            "dropout_active": not args.eval_mode,
            "seed": os.environ.get(SEED_ENV) or None,
            "device": reported.get("device") or str(args.device or "auto"),
            "precision": reported.get("precision"),
            "batch_size": args.batch_size,
            "transform": TRANSFORM_DESCRIPTOR,
            "historical_campaign_delta_macro_r10": dict(HISTORICAL_CAMPAIGN_DELTA),
            "variance_group": variance,
            "note": "this artifact's macro is one draw; the variance that applies to it is "
                    "`variance_group`, computed from the runs it lists. See A-12 in "
                    "docs/reviews/20261111-master-audit/BLOCKED_MODELS_VERIFICATION.vi.md",
        },
        "category_model_policy": "shared",
        "complete_benchmark": aggregate["complete"],
        "categories": [asdict(r) for r in results],
        "aggregate": aggregate,
        "timestamp": timestamp,
        "runtime": {
            "python": sys.executable,
            "source_root": str(args.source_root.resolve()),
            "dataset_root": str(args.dataset_root.resolve()),
            "checkpoint_root": str(args.checkpoint_root.resolve()),
            "variant": args.variant,
            "eval_mode": args.eval_mode,
            # Recorded so a later run can prove it measured the SAME population before pooling; a
            # sibling without these is excluded from the group rather than assumed equal.
            "batch_size": args.batch_size,
            "transform": TRANSFORM_DESCRIPTOR,
        },
    }
    serialized = json.dumps(output, indent=2) + "\n"
    if args.output:
        if args.output.parent.resolve().name == "reports":
            print("[BLOCKED] --output must not target the official reports directory", file=sys.stderr)
            return 1
        with args.output.open("w", encoding="utf-8") as handle:
            handle.write(serialized)
        print(f"[OK] result written: {args.output}")
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
