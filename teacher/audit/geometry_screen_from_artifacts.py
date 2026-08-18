"""Teacher-agnostic natural FashionIQ geometry screen from exported smoke artifacts.

This script deliberately does NOT import any external teacher repository. Each teacher
may be executed in its own Conda environment and export the standard q_full/q_minus
artifact first. The geometry comparison then runs centrally on those exported tensors.

This is a BROAD SCREEN, not final teacher lock: natural FashionIQ captions are not
atomic edit labels, and this script does not validate the exact TAPER intervention,
native retrieval scorer, or BxL training feasibility.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import full_audit as audit


REQUIRED_KEYS = (
    "q_full_pre_norm",
    "q_minus_1_pre_norm",
    "q_minus_2_pre_norm",
    "q_full",
    "q_minus_1",
    "q_minus_2",
)


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Expected NAME=/path/to/smoke.pt"
        )
    name, raw = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("Teacher NAME cannot be empty")
    return name, Path(raw).expanduser()


def validate_artifact(name: str, artifact: dict, cases: list[dict]) -> None:
    n = len(cases)
    for key in REQUIRED_KEYS:
        if key not in artifact:
            raise KeyError(f"{name}: artifact missing {key}")
        tensor = artifact[key]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name}: {key} must be a tensor")
        if tensor.ndim != 2 or tensor.shape[0] != n:
            raise ValueError(
                f"{name}: {key} shape={tuple(tensor.shape)}, expected [N,D] with N={n}"
            )
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{name}: {key} contains NaN/Inf")

    for key in ("q_full", "q_minus_1", "q_minus_2"):
        norm = artifact[key].float().norm(dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=2e-4, rtol=2e-4):
            raise ValueError(f"{name}: {key} is not L2-normalized")

    if "sample_ids" in artifact:
        expected = [case["sample_id"] for case in cases]
        actual = list(artifact["sample_ids"])
        if actual != expected:
            raise ValueError(
                f"{name}: sample_ids/order do not match the shared audit cases"
            )


def build_teacher_screen(
    name: str,
    artifact: dict,
    cases: list[dict],
    min_group_count: int,
    bootstrap_samples: int,
    seed: int,
) -> dict:
    qf_pre = artifact["q_full_pre_norm"].float()
    q1_pre = artifact["q_minus_1_pre_norm"].float()
    q2_pre = artifact["q_minus_2_pre_norm"].float()
    qf = artifact["q_full"].float()
    q1 = artifact["q_minus_1"].float()
    q2 = artifact["q_minus_2"].float()

    d1_pre = qf_pre - q1_pre
    d2_pre = qf_pre - q2_pre
    d1_unit = qf - q1
    d2_unit = qf - q2

    labels = (
        [audit.normalize_edit_label(c["caption_1"]) for c in cases]
        + [audit.normalize_edit_label(c["caption_2"]) for c in cases]
    )

    pre_overall = audit.balanced_same_edit_consistency(
        torch.cat([d1_pre, d2_pre], dim=0),
        labels,
        min_group_count,
        bootstrap_samples,
        seed,
    )
    pre_cat = audit.balanced_geometry_by_category(
        cases, d1_pre, d2_pre, min_group_count, bootstrap_samples, seed
    )
    unit_overall = audit.balanced_same_edit_consistency(
        torch.cat([d1_unit, d2_unit], dim=0),
        labels,
        min_group_count,
        bootstrap_samples,
        seed + 1000,
    )
    unit_cat = audit.balanced_geometry_by_category(
        cases, d1_unit, d2_unit, min_group_count, bootstrap_samples, seed + 1000
    )

    return {
        "teacher": name,
        "screen_semantics": (
            "Natural repeated-caption response geometry; exact captions are a high-precision "
            "proxy for repeated instructions, not guaranteed atomic edit labels."
        ),
        "num_queries": len(cases),
        "query_dimension": int(qf.shape[-1]),
        "pre_norm": {
            "overall": pre_overall,
            "by_category": pre_cat,
            "min_group_count_sensitivity": audit.geometry_group_count_sensitivity(
                cases, d1_pre, d2_pre, bootstrap_samples, seed
            ),
        },
        "normalized_query": {
            "overall": unit_overall,
            "by_category": unit_cat,
            "min_group_count_sensitivity": audit.geometry_group_count_sensitivity(
                cases, d1_unit, d2_unit, bootstrap_samples, seed + 1000
            ),
        },
        "effect_health": {
            "caption_1_removal": audit.effect_metrics(qf_pre, q1_pre, qf, q1),
            "caption_2_removal": audit.effect_metrics(qf_pre, q2_pre, qf, q2),
        },
    }


def compact_row(report: dict) -> dict:
    pre = report["pre_norm"]
    unit = report["normalized_query"]
    return {
        "teacher": report["teacher"],
        "pre_gap": pre["by_category"].get("macro_category_gap"),
        "pre_min_gap": pre["by_category"].get("min_category_gap"),
        "pre_valid_fraction": pre["overall"].get("direction_valid_fraction"),
        "norm_gap": unit["by_category"].get("macro_category_gap"),
        "norm_min_gap": unit["by_category"].get("min_category_gap"),
        "norm_valid_fraction": unit["overall"].get("direction_valid_fraction"),
    }


def fmt(v):
    return "N/A" if v is None else f"{v:.4f}"


def markdown(rows: list[dict]) -> str:
    lines = [
        "| Teacher | Pre gap | Pre min-cat | Pre valid | Norm gap | Norm min-cat | Norm valid |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| " + " | ".join([
                row["teacher"], fmt(row["pre_gap"]), fmt(row["pre_min_gap"]),
                fmt(row["pre_valid_fraction"]), fmt(row["norm_gap"]),
                fmt(row["norm_min_gap"]), fmt(row["norm_valid_fraction"]),
            ]) + " |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument(
        "--teacher-artifact",
        action="append",
        type=parse_named_path,
        required=True,
        help="Repeat: --teacher-artifact ENCODER=teacher/outputs/encoder/smoke.pt",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-group-count", type=int, default=2)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cases = audit.load_cases(args.cases.resolve())
    audit.validate_cases(cases)

    reports = []
    seen = set()
    for index, (name, path) in enumerate(args.teacher_artifact):
        if name in seen:
            raise ValueError(f"Duplicate teacher name: {name}")
        seen.add(name)
        path = path.resolve()
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(artifact, dict):
            raise TypeError(f"{name}: artifact must be a dict")
        validate_artifact(name, artifact, cases)
        reports.append(build_teacher_screen(
            name, artifact, cases, args.min_group_count,
            args.bootstrap_samples, args.seed + index * 10000,
        ))

    rows = [compact_row(report) for report in reports]
    output = {
        "status": "natural_fashioniq_geometry_screen_only",
        "warning": (
            "Do not final-lock a teacher from this file alone. It lacks common native retrieval, "
            "controlled atomic geometry, exact TAPER intervention credibility, and BxL feasibility."
        ),
        "teachers": reports,
        "summary": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    md = args.output.with_suffix(".md")
    md.write_text("# Natural FashionIQ Geometry Screen\n\n" + markdown(rows) + "\n", encoding="utf-8")
    print(markdown(rows))
    print("Saved:", args.output)
    print("Saved:", md)


if __name__ == "__main__":
    main()
