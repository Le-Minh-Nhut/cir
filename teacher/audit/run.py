import argparse
import json
from pathlib import Path


def load_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def format_number(value, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def get_geometry_summary(report: dict) -> dict:
    effect_1 = report["effect_1"]
    effect_2 = report["effect_2"]
    same_edit = report["same_edit_directional_consistency"]
    mean_delta_norm = (effect_1["delta_norm"]["mean"] + effect_2["delta_norm"]["mean"]) / 2.0
    mean_relative_effect = (effect_1["relative_effect_norm"]["mean"] + effect_2["relative_effect_norm"]["mean"]) / 2.0
    mean_cosine_drop = (effect_1["cosine_drop"]["mean"] + effect_2["cosine_drop"]["mean"]) / 2.0
    near_zero_fraction = (effect_1["near_zero_relative_fraction"] + effect_2["near_zero_relative_fraction"]) / 2.0
    if same_edit.get("status") == "ok":
        same_mean = same_edit["same_mean_cosine"]
        different_mean = same_edit["different_mean_cosine"]
        same_diff_gap = same_edit["same_vs_different_gap"]
    else:
        same_mean = None
        different_mean = None
        same_diff_gap = None
    within_sample_cosine = report["within_sample_effect_cosine"]["mean"]
    return {
        "teacher": report["teacher"],
        "num_queries": report["num_queries"],
        "query_dimension": report["query_dimension"],
        "mean_delta_norm": mean_delta_norm,
        "mean_relative_effect": mean_relative_effect,
        "mean_cosine_drop": mean_cosine_drop,
        "near_zero_fraction": near_zero_fraction,
        "same_edit_cosine": same_mean,
        "different_edit_cosine": different_mean,
        "same_vs_different_gap": same_diff_gap,
        "within_sample_effect_cosine": within_sample_cosine,
        "same_edit_status": same_edit.get("status"),
    }


def build_markdown_table(summaries: list[dict]) -> str:
    lines = [
        (
            "| Teacher | Queries | Dim | Rel Effect | Cos Drop | Near-zero | Same | Diff | Gap | "
            "Within-pair Cos |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["teacher"],
                    str(item["num_queries"]),
                    str(item["query_dimension"]),
                    format_number(item["mean_relative_effect"]),
                    format_number(item["mean_cosine_drop"]),
                    format_number(item["near_zero_fraction"]),
                    format_number(item["same_edit_cosine"]),
                    format_number(item["different_edit_cosine"]),
                    format_number(item["same_vs_different_gap"]),
                    format_number(item["within_sample_effect_cosine"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def print_interpretation(summaries: list[dict]) -> None:
    print()
    print("=== Interpretation ===")
    for item in summaries:
        print()
        print(item["teacher"])
        print("  relative effect:", format_number(item["mean_relative_effect"]))
        print("  cosine drop:", format_number(item["mean_cosine_drop"]))
        print("  near-zero fraction:", format_number(item["near_zero_fraction"]))
        if item["same_edit_status"] == "ok":
            print("  same-edit cosine:", format_number(item["same_edit_cosine"]))
            print("  different-edit cosine:", format_number(item["different_edit_cosine"]))
            print("  same-different gap:", format_number(item["same_vs_different_gap"]))
        else:
            print("  same-edit consistency:", item["same_edit_status"])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qure", type=Path, default=Path("teacher/outputs/qure/metrics.json"))
    parser.add_argument("--tme", type=Path, default=Path("teacher/outputs/tme/metrics.json"))
    parser.add_argument("--output-json", type=Path, default=Path("teacher/outputs/tournament.json"))
    parser.add_argument("--output-md", type=Path, default=Path("teacher/outputs/tournament.md"))
    return parser.parse_args()


def main():
    args = parse_args()
    reports = []
    for path in (args.qure, args.tme):
        if not path.exists():
            print(f"Skipping missing metrics: {path}")
            continue
        reports.append(load_json(path))
    if not reports:
        raise RuntimeError("No teacher metrics found")
    summaries = [get_geometry_summary(report) for report in reports]
    markdown = build_markdown_table(summaries)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    tournament = {
        "status": "geometry_only",
        "important_note": (
            "This tournament currently compares teacher query geometry only. Native retrieval "
            "necessity and full FashionIQ retrieval metrics have not yet been added."
        ),
        "teachers": summaries,
    }
    with args.output_json.open("w", encoding="utf-8") as file:
        json.dump(tournament, file, indent=2, ensure_ascii=False)
    with args.output_md.open("w", encoding="utf-8") as file:
        file.write("# TAPER Teacher Tournament\n\n")
        file.write("## Geometry Audit\n\n")
        file.write(markdown)
        file.write("\n\n")
        file.write(
            "> Current tournament is geometry-only. Do not select the final TAPER teacher "
            "from this table alone. Native FashionIQ retrieval and retrieval-level necessity "
            "added.\n"
        )
    print()
    print("=== TAPER Teacher Tournament ===")
    print()
    print(markdown)
    print_interpretation(summaries)
    print()
    print(f"Saved JSON: {args.output_json}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
