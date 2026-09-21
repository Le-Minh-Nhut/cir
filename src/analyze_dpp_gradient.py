from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
import torch
from pathlib import Path


def _records(path: Path) -> list[Mapping[str, object]]:
    return [
        record
        for line in path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line)).get("dpp_gradient_audit")
    ]


def _sum(records: Iterable[Mapping[str, object]], name: str) -> float:
    return sum(float(record["dpp_gradient_audit"].get(name, 0.0)) for record in records)


def _rows(records: Iterable[Mapping[str, object]]) -> list[Mapping[str, object]]:
    return [
        row
        for record in records
        for step in record["dpp_gradient_audit"].get("by_step", {}).values()
        for row in step.get("rows", [])
    ]


def _quantiles(values: list[float]) -> str:
    if not values:
        return "empty"
    tensor = torch.tensor(values)
    return " ".join(
        f"p{name}={float(value):.6g}"
        for name, value in zip(("00", "10", "50", "90", "99", "100"), torch.quantile(tensor, torch.tensor((0, .1, .5, .9, .99, 1))), strict=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize opt-in DPP gradient audit JSONL")
    parser.add_argument("metrics_jsonl", type=Path)
    args = parser.parse_args()
    records = _records(args.metrics_jsonl)
    if not records:
        raise SystemExit("no train_update records contain dpp_gradient_audit")
    eligible = _sum(records, "dpp_total_eligible_row_count")
    valid = _sum(records, "dpp_valid_row_count")
    rows = _rows(records)
    print(f"audit_updates: {len(records)}")
    print(f"dpp_valid_row_count: {int(valid)}")
    print(f"dpp_total_eligible_row_count: {int(eligible)}")
    print(f"dpp_valid_rate: {valid / eligible if eligible else 0.0:.6f}")
    print(f"finite_in_fp32_but_nonfinite_live_count: {int(_sum(records, 'finite_in_fp32_but_nonfinite_live_count'))}")
    print(f"amp_skipped_update_overlap: {sum(bool(record.get('amp_skipped_update_inferred')) for record in records)}")
    print(f"amp_scale_decreased_overlap: {sum(bool(record.get('amp_scale_decreased')) for record in records)}")
    for name in ("effect_norms", "live_effect_grad_norms", "fp32_reference_grad_norms"):
        print(f"{name}: {_quantiles([float(value) for row in rows for value in row[name]])}")
    for row in sorted(rows, key=lambda value: float(value["min_effect_norm"]))[:5]:
        print(
            "min_effect_norm_vs_live_grad: "
            f"{float(row['min_effect_norm']):.6g} "
            f"{max(float(value) for value in row['live_effect_grad_norms']):.6g}"
        )


if __name__ == "__main__":
    main()
