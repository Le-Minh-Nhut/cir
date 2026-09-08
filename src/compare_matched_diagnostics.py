"""Compare OLD/STRONG latent geometry on identical live-sample intersections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from diagnostics.cohort import matched_intersection_ids
from diagnostics.geometry import candidate_geometry, feature_geometry


def _index_rows(values: Tensor, source_ids: list[str], wanted_ids: list[str]) -> Tensor:
    positions = {sample_id: index for index, sample_id in enumerate(source_ids)}
    indices = torch.tensor([positions[sample_id] for sample_id in wanted_ids], dtype=torch.long)
    return values.index_select(0, indices)


def _geometry(values: Tensor) -> dict[str, Any]:
    if values.shape[0] == 0:
        return {"available": False, "reason": "empty matched cohort", "count": 0}
    return candidate_geometry(values) if values.ndim == 3 else feature_geometry(values)


def compare_feature_artifacts(old: dict[str, Any], strong: dict[str, Any]) -> dict[str, Any]:
    """Compute native and matched-intersection geometry without changing rollouts."""

    if old.get("manifest_sample_ids_sha256") != strong.get("manifest_sample_ids_sha256"):
        raise ValueError("OLD and STRONG feature artifacts use different starting manifests")
    if old.get("teacher_batch_grouping_sha256") != strong.get("teacher_batch_grouping_sha256"):
        raise ValueError("OLD and STRONG artifacts use different teacher batch grouping")

    timesteps: dict[str, Any] = {}
    common_timesteps = sorted(set(old["timesteps"]) & set(strong["timesteps"]), key=int)
    for timestep in common_timesteps:
        old_step = old["timesteps"][timestep]
        strong_step = strong["timesteps"][timestep]
        old_ids = list(old_step["sample_ids"])
        strong_ids = list(strong_step["sample_ids"])
        intersection = matched_intersection_ids(old_ids, strong_ids)
        old_features = old_step["features"]
        strong_features = strong_step["features"]
        names = sorted(set(old_features) & set(strong_features))
        comparisons = {}
        for name in names:
            old_values = old_features[name]
            strong_values = strong_features[name]
            if old_values.shape[0] != len(old_ids) or strong_values.shape[0] != len(strong_ids):
                raise ValueError(f"sample-ID/feature row mismatch for {name} at t={timestep}")
            if (
                old_values.ndim != strong_values.ndim
                or old_values.shape[1:] != strong_values.shape[1:]
            ):
                raise ValueError(
                    f"feature interface mismatch for {name} at t={timestep}: "
                    f"{tuple(old_values.shape)} vs {tuple(strong_values.shape)}"
                )
            comparisons[name] = {
                "native_old": _geometry(old_values),
                "native_strong": _geometry(strong_values),
                "matched_old": _geometry(_index_rows(old_values, old_ids, intersection)),
                "matched_strong": _geometry(_index_rows(strong_values, strong_ids, intersection)),
            }
        timesteps[timestep] = {
            "old_live_count": len(old_ids),
            "strong_live_count": len(strong_ids),
            "intersection_count": len(intersection),
            "intersection_fraction_of_old": len(intersection) / len(old_ids) if old_ids else 0.0,
            "intersection_fraction_of_strong": (
                len(intersection) / len(strong_ids) if strong_ids else 0.0
            ),
            "intersection_sample_ids": intersection,
            "features": comparisons,
        }
    return {
        "manifest_sample_ids_sha256": old.get("manifest_sample_ids_sha256"),
        "teacher_batch_grouping_sha256": old.get("teacher_batch_grouping_sha256"),
        "timesteps": timesteps,
    }


def _headline(geometry: dict[str, Any]) -> str:
    if "raw" not in geometry:
        return f"PR={geometry.get('effective_rank_pr', float('nan')):.4f}"
    raw = geometry["raw"]
    per_slot = [value["effective_rank_pr"] for value in raw["per_slot"]]
    return (
        f"per-slot PR={per_slot}; slot-centered PR="
        f"{raw['slot_centered_pooled']['effective_rank_pr']:.4f}; between-slot="
        f"{raw['variance_decomposition']['between_slot_variance_fraction']:.4f}"
    )


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# OLD vs STRONG matched-intersection geometry",
        "",
        f"- manifest fingerprint: `{report['manifest_sample_ids_sha256']}`",
        f"- teacher grouping fingerprint: `{report['teacher_batch_grouping_sha256']}`",
    ]
    for timestep, values in report["timesteps"].items():
        lines += [
            "",
            f"## Timestep {timestep}",
            "",
            f"- OLD live: `{values['old_live_count']}`",
            f"- STRONG live: `{values['strong_live_count']}`",
            f"- matched intersection: `{values['intersection_count']}`",
        ]
        for name, comparison in values["features"].items():
            lines += [
                "",
                f"### {name}",
                "",
                f"- matched OLD: {_headline(comparison['matched_old'])}",
                f"- matched STRONG: {_headline(comparison['matched_strong'])}",
            ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-features", type=Path, required=True)
    parser.add_argument("--strong-features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old = torch.load(args.old_features, map_location="cpu", weights_only=True)
    strong = torch.load(args.strong_features, map_location="cpu", weights_only=True)
    report = compare_feature_artifacts(old, strong)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "matched_geometry_comparison.json"
    markdown_path = args.output_dir / "matched_geometry_comparison.md"
    json_path.write_text(json.dumps(report, indent=2, allow_nan=True), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"[matched-diagnostics] JSON: {json_path}")
    print(f"[matched-diagnostics] Markdown: {markdown_path}")


if __name__ == "__main__":
    main()
