from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path

from analyze_functional_collapse import _weighted_mean

STAGES = (
    "base_query",
    "expanded_query",
    "conditioned_residual",
    "query_pre_norm",
    "query_post_norm",
    "raw_attention_output",
    "proposal_output",
)
SCALARS = (
    "conditioner_to_base_norm_ratio",
    "base_pre_cosine",
    "mean_relative_displacement",
    "attention_diversity_retention",
    "attention_rank_retention",
)


def _records(path: Path) -> list[Mapping[str, object]]:
    return [
        record
        for line in path.read_text(encoding="utf-8").splitlines()
        if (record := json.loads(line)).get("proposal_internal_audit")
    ]


def _summarize(
    records: list[Mapping[str, object]], *, timestep: str | None = None
) -> dict[str, object]:
    audits = [record["proposal_internal_audit"] for record in records]
    if timestep is None:
        return _weighted_mean([audit["overall"] for audit in audits])
    return _weighted_mean([audit["by_step"][timestep] for audit in audits if timestep in audit["by_step"]])


def _heuristic(summary: Mapping[str, object]) -> str:
    def collapsed(stage: str) -> bool:
        values = summary.get(stage)
        return isinstance(values, Mapping) and (
            float(values.get("pairwise_cosine", -1.0)) >= 0.95
            and float(values.get("effective_rank", float("inf"))) <= 1.5
        )

    if collapsed("query_post_norm"):
        return "pre_attention"
    if not collapsed("query_post_norm") and collapsed("proposal_output"):
        return "attention_output"
    if any(collapsed(stage) for stage in STAGES):
        return "ambiguous"
    return "none"


def _print_summary(label: str, summary: Mapping[str, object]) -> None:
    print(f"\n{label}")
    print("stage                    cosine     rank   rel_spread")
    for stage in STAGES:
        values = summary.get(stage)
        if not isinstance(values, Mapping):
            continue
        print(
            f"{stage:24} {float(values['pairwise_cosine']):8.4f} "
            f"{float(values['effective_rank']):8.4f} {float(values['relative_spread']):12.4f}"
        )
    for name in SCALARS:
        if name in summary:
            print(f"{name}: {float(summary[name]):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize ProposalNet internal audit JSONL")
    parser.add_argument("metrics_jsonl", type=Path)
    args = parser.parse_args()
    records = _records(args.metrics_jsonl)
    if not records:
        raise SystemExit("no train_update records contain proposal_internal_audit")
    _print_summary("FIRST 10 BATCHES", _summarize(records[:10]))
    _print_summary("LAST 10 BATCHES", _summarize(records[-10:]))
    full = _summarize(records)
    _print_summary("FULL RUN MEAN", full)
    print(f"heuristic_candidate_collapse_location: {_heuristic(full)}")
    timesteps = sorted({step for record in records for step in record["proposal_internal_audit"]["by_step"]})
    for timestep in timesteps:
        _print_summary(f"TIMESTEP {timestep}", _summarize(records, timestep=timestep))


if __name__ == "__main__":
    main()
