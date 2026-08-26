from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import torch
from torch import Tensor

from models.taper import TAPER


DEFAULT_LAMBDAS = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15)
CATEGORIES = ("dress", "shirt", "toptee")
TOKEN_METRICS = (
    "r4_preprojection_token_mass_mean",
    "r4_preprojection_token_budget_violation_fraction",
    "r4_preprojection_token_budget_excess_mean",
    "r4_token_budget_binding_fraction",
    "routing_token_mass_mean",
    "routing_unassigned_mass_mean",
    "routing_fully_unassigned_token_fraction",
)
MAX_METRICS = (
    "r4_preprojection_token_mass_max",
    "routing_token_mass_max",
)
SAMPLE_METRICS = (
    "routing_active_slot_count",
    "qasa_selected_slot_count",
)
ACTIVE_SLOT_METRICS = (
    "routing_support_mean",
    "routing_support_fraction_mean",
)
REQUIRED_METRICS = (
    *TOKEN_METRICS,
    *MAX_METRICS,
    *SAMPLE_METRICS,
    *ACTIVE_SLOT_METRICS,
    "routing_zero_fraction",
    "routing_support_overlap_mean",
)


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("lambda values must be finite and > 0")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="No-training R4a QI-SCA lambda geometry calibration."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/FashionIQ"))
    parser.add_argument("--cache-root", type=Path, default=Path("features"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("conf/experiment/taper_e2e.yaml"),
    )
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=positive_float,
        default=list(DEFAULT_LAMBDAS),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-queries-per-category",
        type=int,
        default=0,
        help="0 means the full validation set.",
    )
    parser.add_argument(
        "--correction-policy",
        default=None,
        help="Override config; validated by the FashionIQ dataset policy helper.",
    )
    parser.add_argument("--text-cache-subdir", default=None)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("reports/r4_lambda_calibration.json"),
    )
    return parser.parse_args(argv)


def force_r4a(model: TAPER) -> None:
    model.routing_mode = "qisca"
    model.r4_capacity_enabled = False


def parameter_versions(model: TAPER) -> tuple[int, ...]:
    return tuple(parameter._version for parameter in model.parameters())


class WeightedMetrics:
    def __init__(self) -> None:
        self.totals: dict[str, float] = defaultdict(float)
        self.weights: dict[str, float] = defaultdict(float)
        self.maxima: dict[str, float] = {}

    def add(self, name: str, value: Tensor, weight: float) -> None:
        scalar = float(value.detach().cpu())
        if weight <= 0 or not math.isfinite(scalar):
            return
        self.totals[name] += scalar * weight
        self.weights[name] += weight

    def add_max(self, name: str, value: Tensor) -> None:
        scalar = float(value.detach().cpu())
        if math.isfinite(scalar):
            self.maxima[name] = max(self.maxima.get(name, -math.inf), scalar)

    def finalize(self) -> dict[str, float]:
        result = {
            name: self.totals[name] / self.weights[name]
            for name in self.totals
            if self.weights[name] > 0
        }
        result.update(self.maxima)
        return result


def _check_first_batch_invariance(
    baseline: dict[str, Tensor] | None,
    output: dict[str, Tensor],
) -> dict[str, Tensor]:
    names = (
        "ownership_logits",
        "slot_masks",
        "qasa_attention",
        "qasa_quality",
        "qasa_selected_mask",
    )
    current = {name: output[name].detach().cpu().clone() for name in names}
    if baseline is None:
        return current
    for name in names:
        if current[name].dtype == torch.bool:
            equal = torch.equal(current[name], baseline[name])
        else:
            equal = torch.allclose(
                current[name],
                baseline[name],
                atol=1e-6,
                rtol=1e-6,
            )
        if not equal:
            raise RuntimeError(
                f"QASA/pre-routing invariant changed across lambda sweep: {name}"
            )
    return baseline


@torch.no_grad()
def calibrate_model(
    model: TAPER,
    lambdas: Sequence[float],
    batch_factory: Callable[[], Iterable[dict[str, Tensor]]],
) -> tuple[list[dict[str, object]], int]:
    if not lambdas:
        raise ValueError("lambda sweep must not be empty")
    if any(not math.isfinite(value) or value <= 0 for value in lambdas):
        raise ValueError("all lambda values must be finite and > 0")

    force_r4a(model)
    model.eval()
    versions_before = parameter_versions(model)
    original_lambda = model.r4_lambda
    first_batch_baseline: dict[str, Tensor] | None = None
    expected_queries: int | None = None
    results: list[dict[str, object]] = []

    try:
        for lambda_value in lambdas:
            model.r4_lambda = float(lambda_value)
            aggregate = WeightedMetrics()
            slot_active = torch.zeros(model.num_slots, dtype=torch.float64)
            soft_dominant = torch.zeros(model.num_slots, dtype=torch.float64)
            num_queries = 0

            for batch_index, batch in enumerate(batch_factory()):
                text_states = batch["text_states"]
                attention_mask = batch["text_attention_mask"]
                content_mask = batch["text_content_mask"]
                output = model.build_edit_slots(
                    text_states,
                    attention_mask,
                    text_content_mask=content_mask,
                )
                if batch_index == 0:
                    first_batch_baseline = _check_first_batch_invariance(
                        first_batch_baseline,
                        output,
                    )

                valid = attention_mask.to(torch.bool) & content_mask.to(torch.bool)
                valid_count = float(valid.sum())
                sample_count = int(text_states.shape[0])
                active = output["routing_active_mask"]
                active_count = float(active.sum())
                valid_per_sample = valid.sum(dim=1)
                active_valid_positions = float(
                    (active * valid_per_sample[:, None]).sum()
                )
                upper = torch.triu(
                    torch.ones(
                        model.num_slots,
                        model.num_slots,
                        dtype=torch.bool,
                        device=active.device,
                    ),
                    diagonal=1,
                )
                active_pairs = active[:, :, None] & active[:, None, :]
                active_pair_count = float((active_pairs & upper[None]).sum())

                diagnostics = model._assignment_diagnostics(
                    slot_masks=output["slot_masks"],
                    slot_mass=output["slot_mass"],
                    routing_masks=output["routing_masks"],
                    routing_slot_mass=output["routing_slot_mass"],
                    routing_support_count=output["routing_support_count"],
                    qasa_selected_mask=output["qasa_selected_mask"],
                    qasa_quality=output["qasa_quality"],
                    qasa_final_coverage=output["qasa_final_coverage"],
                    hard_active_slot_mask=output["execution_selected_mask"],
                    text_attention_mask=attention_mask,
                    text_content_mask=content_mask,
                    r4_preprojection=output["r4_preprojection"],
                )
                for name in TOKEN_METRICS:
                    aggregate.add(name, diagnostics[name], valid_count)
                for name in MAX_METRICS:
                    aggregate.add_max(name, diagnostics[name])
                for name in SAMPLE_METRICS:
                    aggregate.add(name, diagnostics[name], sample_count)
                for name in ACTIVE_SLOT_METRICS:
                    aggregate.add(name, diagnostics[name], active_count)
                aggregate.add(
                    "routing_zero_fraction",
                    diagnostics["routing_zero_fraction"],
                    active_valid_positions,
                )
                aggregate.add(
                    "routing_support_overlap_mean",
                    diagnostics["routing_support_overlap_mean"],
                    active_pair_count,
                )
                aggregate.add(
                    "routing_slot_mass_mean",
                    diagnostics["routing_slot_mass_mean"],
                    sample_count * model.num_slots,
                )

                slot_active += active.detach().cpu().sum(dim=0).double()
                soft_dominant_ids = output["slot_mass"].argmax(dim=1).detach().cpu()
                soft_dominant += torch.bincount(
                    soft_dominant_ids,
                    minlength=model.num_slots,
                ).double()
                num_queries += sample_count

            if expected_queries is None:
                expected_queries = num_queries
            elif num_queries != expected_queries:
                raise RuntimeError("lambda sweeps processed different query counts")
            if num_queries == 0:
                raise RuntimeError("calibration processed zero queries")

            metrics = aggregate.finalize()
            for name in REQUIRED_METRICS:
                metrics.setdefault(name, 0.0)
            results.append(
                {
                    "lambda": float(lambda_value),
                    "metrics": metrics,
                    "slot_active_frequency": {
                        str(slot_id): float(slot_active[slot_id] / num_queries)
                        for slot_id in range(model.num_slots)
                    },
                    "soft_dominant_slot_frequency": {
                        str(slot_id): float(soft_dominant[slot_id] / num_queries)
                        for slot_id in range(model.num_slots)
                    },
                }
            )
    finally:
        model.r4_lambda = original_lambda

    if parameter_versions(model) != versions_before:
        raise RuntimeError("model parameters changed during no-training calibration")
    assert expected_queries is not None
    return results, expected_queries


def write_report(path: Path, report: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def print_table(results: Sequence[dict[str, object]]) -> None:
    print()
    print(
        "lambda  pre_mass  violation  binding  unassigned  active_slots  "
        "support_frac  overlap"
    )
    for result in results:
        metrics = result["metrics"]
        assert isinstance(metrics, dict)
        print(
            f"{float(result['lambda']):>6.2f}  "
            f"{float(metrics['r4_preprojection_token_mass_mean']):>8.3f}  "
            f"{100 * float(metrics['r4_preprojection_token_budget_violation_fraction']):>8.2f}%  "
            f"{100 * float(metrics['r4_token_budget_binding_fraction']):>6.2f}%  "
            f"{100 * float(metrics['routing_unassigned_mass_mean']):>9.2f}%  "
            f"{float(metrics['routing_active_slot_count']):>12.3f}  "
            f"{float(metrics['routing_support_fraction_mean']):>12.3f}  "
            f"{float(metrics.get('routing_support_overlap_mean', 0.0)):>7.3f}"
        )
    print()
    print("Per-slot routing active frequency:")
    for result in results:
        frequencies = result["slot_active_frequency"]
        assert isinstance(frequencies, dict)
        formatted = " ".join(
            f"S{slot_id}={100 * float(frequencies[str(slot_id)]):.2f}%"
            for slot_id in range(len(frequencies))
        )
        print(f"lambda={float(result['lambda']):.2f}: {formatted}")

    print()
    print("Per-slot soft dominant frequency (pre-routing ownership):")
    for result in results:
        frequencies = result["soft_dominant_slot_frequency"]
        assert isinstance(frequencies, dict)
        formatted = " ".join(
            f"S{slot_id}={100 * float(frequencies[str(slot_id)]):.2f}%"
            for slot_id in range(len(frequencies))
        )
        print(f"lambda={float(result['lambda']):.2f}: {formatted}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    from omegaconf import OmegaConf

    from cache.features import (
        get_text_features_by_sample_ids,
        load_feature_manifest,
        load_text_features,
        validate_feature_manifest,
        validate_text_cache_subdir,
    )
    from datasets.fashioniq import validate_correction_policy
    from evaluate_qasa_inference import (
        build_model,
        build_val_loaders,
        load_checkpoint,
        load_correction_dicts,
    )

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be >= 0")
    if args.max_queries_per_category < 0:
        raise ValueError("--max-queries-per-category must be >= 0")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    device = torch.device(args.device)
    cfg = OmegaConf.load(args.config)
    correction_policy = validate_correction_policy(
        args.correction_policy or str(cfg.correction_policy)
    )
    text_cache_subdir = validate_text_cache_subdir(
        args.text_cache_subdir or str(cfg.text_cache_subdir),
        correction_policy,
    )
    cfg.model.routing_mode = "qisca"
    cfg.model.r4_capacity_enabled = False

    annotation_root = args.dataset_root / "captions"
    correction_dicts = (
        load_correction_dicts(annotation_root)
        if correction_policy == "fashioniq"
        else None
    )
    loaders = build_val_loaders(
        annotation_root=annotation_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        caption_policy=cfg.val_caption_policy,
        correction_dicts=correction_dicts,
    )

    feature_root = args.cache_root / "fashioniq" / "fgclip2-large" / "val"
    image_manifest = load_feature_manifest(feature_root / "images")
    validate_feature_manifest(
        image_manifest,
        model_id=str(cfg.backbone.model_id),
        revision=str(cfg.backbone.revision),
        cache_name="val/images",
    )
    text_cache = load_text_features(feature_root / text_cache_subdir)
    validate_feature_manifest(
        text_cache.manifest,
        model_id=str(cfg.backbone.model_id),
        revision=str(cfg.backbone.revision),
        cache_name=f"val/{text_cache_subdir}",
        correction_policy=correction_policy,
    )

    model = build_model(cfg, device)
    force_r4a(model)
    load_checkpoint(model, args.checkpoint)
    model.eval()

    def batch_factory() -> Iterable[dict[str, Tensor]]:
        for category in CATEGORIES:
            processed = 0
            for batch in loaders[category]:
                if (
                    args.max_queries_per_category
                    and processed >= args.max_queries_per_category
                ):
                    break
                take = len(batch.sample_ids)
                if args.max_queries_per_category:
                    take = min(
                        take,
                        args.max_queries_per_category - processed,
                    )
                sample_ids = list(batch.sample_ids)[:take]
                modification_texts = list(batch.modification_texts)[:take]
                text_states, attention_mask, content_mask = (
                    get_text_features_by_sample_ids(
                        sample_ids,
                        modification_texts,
                        text_cache,
                    )
                )
                yield {
                    "text_states": text_states.to(
                        device=device,
                        dtype=torch.float32,
                    ),
                    "text_attention_mask": attention_mask.to(
                        device=device,
                        dtype=torch.bool,
                    ),
                    "text_content_mask": content_mask.to(
                        device=device,
                        dtype=torch.bool,
                    ),
                }
                processed += take

    print("Calibration mode: R4a QI-SCA")
    print("Capacity enabled:", model.r4_capacity_enabled)
    print("Theta:", model.r4_theta)
    print("Checkpoint:", args.checkpoint)
    print("Lambda sweep:", " ".join(str(value) for value in args.lambdas))

    results, num_queries = calibrate_model(model, args.lambdas, batch_factory)
    report: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "theta": model.r4_theta,
        "capacity_enabled": model.r4_capacity_enabled,
        "routing_mode": model.routing_mode,
        "num_queries": num_queries,
        "lambdas": [float(value) for value in args.lambdas],
        "results": results,
    }
    write_report(args.json_output, report)
    print_table(results)
    print(f"Saved calibration report: {args.json_output}")


if __name__ == "__main__":
    main()
