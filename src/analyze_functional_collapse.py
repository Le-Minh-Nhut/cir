from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

STAGES = ("proposals", "alpha_read", "exec_mask", "entities", "actions", "delta", "delta_q")


def _records(path: Path) -> list[Mapping[str, object]]:
    return [
        record
        for line in path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line)).get("functional_collapse_audit")
    ]


def _weighted_mean(values: Iterable[Mapping[str, object]]) -> dict[str, object]:
    values = list(values)
    if not values:
        return {}

    def weight(value: Mapping[str, object]) -> float:
        return float(value.get("live_samples", 0))

    total = sum(weight(value) for value in values)
    if not total:
        return {}
    result: dict[str, object] = {"live_samples": int(total)}
    for key, value in values[0].items():
        if key == "live_samples":
            continue
        stage_values = [item[key] for item in values]
        if isinstance(value, Mapping):
            result[key] = _weighted_mean(
                [
                    {"live_samples": weight(item), **stage}
                    for item, stage in zip(values, stage_values, strict=True)
                ]
            )
        elif isinstance(value, list):
            result[key] = [
                sum(float(item[key][index]) * weight(item) for item in values) / total
                for index in range(len(value))
            ]
        else:
            result[key] = sum(float(item[key]) * weight(item) for item in values) / total
    return result


def _summarize(
    records: list[Mapping[str, object]], *, timestep: str | None = None
) -> dict[str, object]:
    audits = [record["functional_collapse_audit"] for record in records]
    if timestep is None:
        return _weighted_mean([audit["overall"] for audit in audits])
    return _weighted_mean([audit["by_step"][timestep] for audit in audits if timestep in audit["by_step"]])


def _format(stage: str, values: Mapping[str, object]) -> str:
    cosine = float(values.get("pairwise_cosine", float("nan")))
    rank = values.get("effective_rank")
    spread = values.get("relative_spread")
    specific = (
        values.get("pairwise_js_divergence")
        if stage == "alpha_read"
        else values.get("soft_iou")
        if stage == "exec_mask"
        else None
    )
    rank_text = f"{float(rank):8.4f}" if rank is not None else f"{'-':>8}"
    spread_text = f"{float(spread):10.4f}" if spread is not None else f"{'-':>10}"
    specific_text = f"{float(specific):10.4f}" if specific is not None else f"{'-':>10}"
    return f"{stage:12} {cosine:8.4f} {rank_text} {spread_text} {specific_text}"

def _print_summary(label: str, summary: Mapping[str, object]) -> None:
    print(f"\n{label}")
    print("Grounder outputs are siblings: alpha_read and exec_mask.")
    print("stage        cosine     rank   rel_spread   JS/softIoU")
    for stage in STAGES:
        if stage in summary:
            print(_format(stage, summary[stage]))


def _heuristic(summary: Mapping[str, object]) -> str:
    for stage in STAGES:
        values = summary.get(stage)
        if not isinstance(values, Mapping) or float(values.get("pairwise_cosine", -1.0)) < 0.95:
            continue
        rank = values.get("effective_rank")
        if rank is None or float(rank) <= 1.5:
            return stage
    return "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize functional candidate collapse audit JSONL")
    parser.add_argument("metrics_jsonl", type=Path)
    args = parser.parse_args()
    records = _records(args.metrics_jsonl)
    if not records:
        raise SystemExit("no train_update records contain functional_collapse_audit")
    _print_summary("FIRST 10 BATCHES", _summarize(records[:10]))
    _print_summary("LAST 10 BATCHES", _summarize(records[-10:]))
    full = _summarize(records)
    _print_summary("FULL RUN MEAN", full)
    print(f"heuristic_first_high_similarity_stage: {_heuristic(full)}")
    timesteps = sorted({step for record in records for step in record["functional_collapse_audit"]["by_step"]})
    for timestep in timesteps:
        _print_summary(f"TIMESTEP {timestep}", _summarize(records, timestep=timestep))


if __name__ == "__main__":
    main()
