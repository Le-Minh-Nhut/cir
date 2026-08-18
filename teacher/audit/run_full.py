from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from itertools import combinations
from pathlib import Path


def run(command: list[str]) -> None:
    print()
    print("$", " ".join(command))
    subprocess.run(command, check=True)


def load_env_config(path: Path | None) -> dict:
    if path is None:
        return {}
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Teacher env config not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError("Teacher env config must be a JSON object")
    return data


def _runtime_from_config_entry(
    teacher: str,
    entry,
) -> tuple[str | None, str | None]:
    if entry is None:
        return None, None
    if isinstance(entry, str):
        return None, entry
    if not isinstance(entry, dict):
        raise TypeError(
            f"env config entry for {teacher!r} must be a string or object"
        )
    python_path = entry.get("python")
    conda_env = entry.get("conda_env")
    if python_path is not None and conda_env is not None:
        raise ValueError(
            f"{teacher}: env config may specify only one of "
            "'python' or 'conda_env'"
        )
    return python_path, conda_env


def resolve_teacher_runtime(
    teacher: str,
    direct_python: Path | None,
    direct_conda_env: str | None,
    env_config: dict,
) -> dict:
    if direct_python is not None and direct_conda_env is not None:
        raise ValueError(
            f"{teacher}: pass only one of --{teacher}-python or "
            f"--{teacher}-conda-env"
        )

    cfg_python, cfg_conda = _runtime_from_config_entry(
        teacher,
        env_config.get(teacher),
    )

    if direct_python is not None or direct_conda_env is not None:
        python_path = (
            str(direct_python.expanduser().resolve())
            if direct_python is not None
            else None
        )
        conda_env = direct_conda_env
        source = "cli"
    else:
        python_path = cfg_python
        conda_env = cfg_conda
        source = "env_config"

    if python_path is None and conda_env is None:
        raise RuntimeError(
            f"No isolated runtime configured for {teacher}. "
            f"Pass --{teacher}-python /path/to/env/bin/python, "
            f"--{teacher}-conda-env ENV_NAME, or --env-config JSON. "
            "There is intentionally NO fallback to sys.executable."
        )

    if python_path is not None:
        python_path = str(Path(python_path).expanduser().resolve())
        path = Path(python_path)
        if not path.exists():
            raise FileNotFoundError(
                f"{teacher}: Python executable not found: {path}"
            )
        if not path.is_file():
            raise ValueError(
                f"{teacher}: Python executable is not a file: {path}"
            )
        return {
            "teacher": teacher,
            "kind": "python",
            "value": python_path,
            "source": source,
            "prefix": [python_path],
            "identity": f"python:{python_path}",
        }

    conda = shutil.which("conda")
    if conda is None:
        raise RuntimeError(
            f"{teacher}: conda executable not found in PATH, but "
            f"conda env {conda_env!r} was requested. "
            f"Use --{teacher}-python with the env's absolute Python path."
        )
    return {
        "teacher": teacher,
        "kind": "conda_env",
        "value": str(conda_env),
        "source": source,
        "prefix": [
            conda,
            "run",
            "--no-capture-output",
            "-n",
            str(conda_env),
            "python",
        ],
        "identity": f"conda:{conda_env}",
    }


def probe_teacher_runtime(runtime: dict) -> None:
    teacher = runtime["teacher"]
    probe = (
        "import sys, torch; "
        "from PIL import Image; "
        "print('teacher_runtime=%s' % sys.executable); "
        "print('python=%s' % sys.version.split()[0]); "
        "print('torch=%s' % torch.__version__); "
        "print('cuda_available=%s' % torch.cuda.is_available())"
    )
    command = [*runtime["prefix"], "-c", probe]
    print()
    print(f"=== Runtime probe: {teacher} ===")
    subprocess.run(command, check=True)


def validate_runtime_isolation(runtimes: dict[str, dict]) -> None:
    identities = [runtime["identity"] for runtime in runtimes.values()]
    if len(set(identities)) != len(identities):
        duplicates = sorted(
            identity
            for identity in set(identities)
            if identities.count(identity) > 1
        )
        raise RuntimeError(
            "Finalist teachers are expected to run in isolated environments, "
            f"but runtimes are shared: {duplicates}"
        )


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


CATEGORIES = ("dress", "shirt", "toptee")


def _quantile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("Cannot take quantile of empty values")
    ordered = sorted(values)
    position = q * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _macro_mean_recall_from_sample(
    ranks: list[int],
    sample_by_category: dict[str, list[int]],
) -> float:
    r10_values = []
    r50_values = []
    for category in CATEGORIES:
        indices = sample_by_category[category]
        if not indices:
            raise ValueError(f"Empty bootstrap category: {category}")
        selected = [ranks[index] for index in indices]
        r10_values.append(
            100.0 * sum(rank <= 10 for rank in selected) / len(selected)
        )
        r50_values.append(
            100.0 * sum(rank <= 50 for rank in selected) / len(selected)
        )
    macro_r10 = sum(r10_values) / len(r10_values)
    macro_r50 = sum(r50_values) / len(r50_values)
    return (macro_r10 + macro_r50) / 2.0


def paired_retrieval_bootstrap(
    report_a: dict,
    report_b: dict,
    samples: int,
    seed: int,
) -> dict:
    payload_a = report_a["comparison_payload"]
    payload_b = report_b["comparison_payload"]
    categories = payload_a["categories"]
    if categories != payload_b["categories"]:
        raise ValueError("Teacher reports have non-identical query ordering")

    ranks_a = payload_a[
        "common_full_gallery_include_reference_full_ranks"
    ]
    ranks_b = payload_b[
        "common_full_gallery_include_reference_full_ranks"
    ]
    if len(ranks_a) != len(ranks_b) or len(ranks_a) != len(categories):
        raise ValueError("Teacher reports have incompatible rank arrays")

    indices_by_category = {
        category: [
            index for index, current in enumerate(categories)
            if current == category
        ]
        for category in CATEGORIES
    }
    point_sample = {
        category: list(indices)
        for category, indices in indices_by_category.items()
    }
    point_a = _macro_mean_recall_from_sample(ranks_a, point_sample)
    point_b = _macro_mean_recall_from_sample(ranks_b, point_sample)

    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        sample_by_category = {
            category: [
                rng.choice(indices) for _ in range(len(indices))
            ]
            for category, indices in indices_by_category.items()
        }
        score_a = _macro_mean_recall_from_sample(
            ranks_a, sample_by_category
        )
        score_b = _macro_mean_recall_from_sample(
            ranks_b, sample_by_category
        )
        differences.append(score_a - score_b)

    return {
        "metric": "common_macro_mean_r10_r50",
        "unit": "percentage_points",
        "paired_stratified_by_category": True,
        "a_minus_b_point": point_a - point_b,
        "bootstrap_95_interval": [
            _quantile(differences, 0.025),
            _quantile(differences, 0.975),
        ],
        "bootstrap_probability_a_gt_b": (
            sum(value > 0 for value in differences) / len(differences)
        ),
        "interpretation": (
            "Paired benchmark uncertainty diagnostic. If the interval spans "
            "zero, do not overstate a retrieval winner from a small point "
            "difference."
        ),
    }


def _geometry_group_map(report: dict, normalized: bool) -> dict:
    key = (
        "same_edit_directional_consistency_unit_query_by_category"
        if normalized
        else "same_edit_directional_consistency_balanced_by_category"
    )
    per_category = report[key]["per_category"]
    out = {}
    for category in CATEGORIES:
        groups = per_category[category].get("groups", [])
        out[category] = {
            row["label"]: row["gap"] for row in groups
        }
    return out


def paired_geometry_group_bootstrap(
    report_a: dict,
    report_b: dict,
    normalized: bool,
    samples: int,
    seed: int,
) -> dict:
    map_a = _geometry_group_map(report_a, normalized)
    map_b = _geometry_group_map(report_b, normalized)
    common = {
        category: sorted(set(map_a[category]) & set(map_b[category]))
        for category in CATEGORIES
    }
    usable_categories = [
        category for category in CATEGORIES if common[category]
    ]
    if not usable_categories:
        return {
            "status": "no_common_repeated_edit_groups",
            "space": (
                "l2_normalized_query_space"
                if normalized else "pre_norm_query_space"
            ),
        }

    def score(mapping, sampled):
        category_means = []
        for category in usable_categories:
            labels = sampled[category]
            category_means.append(
                sum(mapping[category][label] for label in labels)
                / len(labels)
            )
        return sum(category_means) / len(category_means)

    point_labels = {
        category: list(common[category])
        for category in usable_categories
    }
    point_diff = score(map_a, point_labels) - score(map_b, point_labels)

    rng = random.Random(seed)
    differences = []
    for _ in range(samples):
        sampled = {
            category: [
                rng.choice(common[category])
                for _ in range(len(common[category]))
            ]
            for category in usable_categories
        }
        differences.append(score(map_a, sampled) - score(map_b, sampled))

    return {
        "status": "ok",
        "space": (
            "l2_normalized_query_space"
            if normalized else "pre_norm_query_space"
        ),
        "a_minus_b_point_on_common_groups": point_diff,
        "common_group_counts": {
            category: len(common[category]) for category in CATEGORIES
        },
        "paired_group_bootstrap_95_interval_approx": [
            _quantile(differences, 0.025),
            _quantile(differences, 0.975),
        ],
        "bootstrap_probability_a_gt_b": (
            sum(value > 0 for value in differences) / len(differences)
        ),
        "caveat": (
            "Robustness interval, not a formal independent-observation CI: "
            "per-group gap statistics share their different-edit baselines."
        ),
    }


def build_pairwise_uncertainty(
    reports: list[dict],
    samples: int,
    seed: int,
) -> list[dict]:
    output = []
    for pair_index, (a, b) in enumerate(combinations(reports, 2)):
        pair_seed = seed + pair_index * 10000
        output.append({
            "teacher_a": a["teacher"],
            "teacher_b": b["teacher"],
            "retrieval": paired_retrieval_bootstrap(
                a, b, samples, pair_seed
            ),
            "pre_norm_geometry": paired_geometry_group_bootstrap(
                a, b, False, samples, pair_seed + 1
            ),
            "normalized_query_geometry": paired_geometry_group_bootstrap(
                a, b, True, samples, pair_seed + 2
            ),
        })
    return output



def strong_multi_axis_dominance(pairwise_uncertainty: list[dict]) -> list[dict]:
    """
    Conservative screening heuristic, not a formal multiple-comparison test.

    Candidate A is called robustly dominated by B only when the paired 95%
    bootstrap interval for A-B is entirely below zero on ALL three screening
    views: common retrieval, pre-norm geometry, and normalized-query geometry.
    Geometry intervals are explicitly approximate because group gaps share
    baselines, so this is evidence for pruning rather than a statistical theorem.
    """
    output = []
    for item in pairwise_uncertainty:
        r = item["retrieval"].get("bootstrap_95_interval")
        p = item["pre_norm_geometry"].get(
            "paired_group_bootstrap_95_interval_approx"
        )
        n = item["normalized_query_geometry"].get(
            "paired_group_bootstrap_95_interval_approx"
        )
        if r is None or p is None or n is None:
            continue

        intervals = {"retrieval": r, "pre_norm_geometry": p,
                     "normalized_query_geometry": n}
        if all(interval[1] < 0 for interval in intervals.values()):
            output.append({
                "dominated": item["teacher_a"],
                "dominant": item["teacher_b"],
                "intervals_are_a_minus_b": intervals,
                "strength": "robust_three_axis_screening_evidence",
            })
        elif all(interval[0] > 0 for interval in intervals.values()):
            output.append({
                "dominated": item["teacher_b"],
                "dominant": item["teacher_a"],
                "intervals_are_a_minus_b": intervals,
                "strength": "robust_three_axis_screening_evidence",
            })
    return output

def get_summary(report: dict) -> dict:
    common = report["retrieval_quality"]["common_full_gallery"][
        "include_reference"
    ]["full"]
    native = report["retrieval_quality"]["published_native"]
    balanced = report[
        "same_edit_directional_consistency_balanced"
    ]
    balanced_cat = report[
        "same_edit_directional_consistency_balanced_by_category"
    ]
    balanced_unit = report[
        "same_edit_directional_consistency_unit_query_space"
    ]
    balanced_unit_cat = report[
        "same_edit_directional_consistency_unit_query_by_category"
    ]
    sensitivity_root = report["teacher_edit_retrieval_sensitivity"][
        "metrics"
    ]["include_reference"]
    sensitivity = sensitivity_root["combined_single_caption_removals"]
    all_text = sensitivity_root["all_text_removal"]
    compound = report["compound_compositionality"]
    cf = report["caption_order_robustness"]["geometry"]
    diff = report["differentiable_intervention_probe"]
    controlled_atomic = report["controlled_atomic_geometry_gate"]
    exact_intervention = report["exact_intervention_credibility_gate"]
    feasibility = report["counterfactual_training_feasibility_gate"]
    info_path = report["teacher_information_path_review"]
    bridge = report["dual_encoder_bridge_gate"]
    integrity = report["integrity"]
    checkpoint = integrity["checkpoint_load"]
    repo = integrity["upstream_repo"]
    parity = integrity["native_interface_parity"]

    cond_log = sensitivity.get("log_rank_ratio_given_full_r50")
    cond_rank = sensitivity.get("rank_degradation_given_full_r50")
    cond_margin = sensitivity.get("margin_drop_given_full_r50")
    all_text_log = all_text.get("log_rank_ratio_given_full_r50")
    all_text_margin = all_text.get("margin_drop_given_full_r50")

    return {
        "teacher": report["teacher"],
        "common_macro_r10": common["macro"]["r10"],
        "common_macro_r50": common["macro"]["r50"],
        "common_macro_mean_r10_r50": common["macro"]["mean_r10_r50"],
        "published_native_protocol": native["protocol_name"],
        "published_native_policy": native["policy"],
        "published_native_macro_mean_r10_r50": (
            native["quality"]["macro"]["mean_r10_r50"]
        ),
        "pre_norm_macro_category_gap": balanced_cat.get(
            "macro_category_gap"
        ),
        "pre_norm_min_category_gap": balanced_cat.get(
            "min_category_gap"
        ),
        "pre_norm_valid_category_count": balanced_cat.get(
            "num_categories_with_valid_gap"
        ),
        "pre_norm_positive_group_gap_fraction": balanced.get(
            "positive_gap_fraction"
        ),
        "pre_norm_direction_valid_fraction": balanced.get(
            "direction_valid_fraction"
        ),
        "pre_norm_used_effect_fraction": balanced.get(
            "used_effect_fraction"
        ),
        "normalized_macro_category_gap": balanced_unit_cat.get(
            "macro_category_gap"
        ),
        "normalized_min_category_gap": balanced_unit_cat.get(
            "min_category_gap"
        ),
        "normalized_valid_category_count": balanced_unit_cat.get(
            "num_categories_with_valid_gap"
        ),
        "normalized_positive_group_gap_fraction": balanced_unit.get(
            "positive_gap_fraction"
        ),
        "normalized_direction_valid_fraction": balanced_unit.get(
            "direction_valid_fraction"
        ),
        "teacher_sensitivity_mean_log_rank_ratio_given_full_r50": (
            None if cond_log is None else cond_log["mean"]
        ),
        "teacher_sensitivity_median_rank_degradation_given_full_r50": (
            None if cond_rank is None else cond_rank["median"]
        ),
        "teacher_sensitivity_mean_margin_drop_given_full_r50": (
            None if cond_margin is None else cond_margin["mean"]
        ),
        "teacher_sensitivity_margin_worse_fraction_given_full_r50": sensitivity.get(
            "margin_worse_fraction_given_full_r50"
        ),
        "all_text_log_rank_ratio_given_full_r50": (
            None if all_text_log is None else all_text_log["mean"]
        ),
        "all_text_mean_margin_drop_given_full_r50": (
            None if all_text_margin is None else all_text_margin["mean"]
        ),
        "all_text_margin_worse_fraction_given_full_r50": all_text.get(
            "margin_worse_fraction_given_full_r50"
        ),
        "compound_additivity_cosine_exploratory": (
            compound["additivity_cosine"]["mean"]
        ),
        "caption_order_effect_cosine_1_exploratory": (
            cf["effect_1_direction_cosine"]["mean"]
        ),
        "caption_order_effect_cosine_2_exploratory": (
            cf["effect_2_direction_cosine"]["mean"]
        ),
        "gradient_access_probe": diff["status"],
        "controlled_atomic_geometry": controlled_atomic["status"],
        "exact_intervention_credibility": exact_intervention["status"],
        "counterfactual_training_feasibility": feasibility["status"],
        "teacher_information_path_review": info_path["status"],
        "dual_encoder_bridge_gate": bridge["status"],
        "native_interface_parity": parity["status"],
        "checkpoint_load_status": checkpoint["status"],
        "repo_snapshot_match": repo.get("matches_audited_snapshot"),
        "repo_tracked_clean": repo.get("tracked_worktree_clean"),
    }


def fmt(value, digits=4):
    if value is None:
        return "N/A"
    if isinstance(value, str):
        return value
    return f"{value:.{digits}f}"


def markdown_table(rows: list[dict]) -> str:
    lines = [
        (
            "| Teacher | MeanR | Pre-norm gap | Norm-query gap | "
            "Pre valid | Text-all margin+ | Single-cap margin+ | Grad | "
            "Native parity | Exact intervention | Feasibility |"
        ),
        (
            "|---|---:|---:|---:|---:|---:|---:|---|---|---|---|"
        ),
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join([
                row["teacher"],
                fmt(row["common_macro_mean_r10_r50"]),
                fmt(row["pre_norm_macro_category_gap"]),
                fmt(row["normalized_macro_category_gap"]),
                fmt(row["pre_norm_direction_valid_fraction"]),
                fmt(row["all_text_margin_worse_fraction_given_full_r50"]),
                fmt(row["teacher_sensitivity_margin_worse_fraction_given_full_r50"]),
                row["gradient_access_probe"],
                row["native_interface_parity"],
                row["exact_intervention_credibility"],
                row["counterfactual_training_feasibility"],
            ])
            + " |"
        )
    return "\n".join(lines)


def metric_ranks(rows: list[dict]) -> dict:
    """
    Descriptive rankings only. They are not independent votes to be summed.

    The three primary evidence views answer different questions:
      retrieval: does the teacher solve CIR?
      pre-norm geometry: is the delta actually consumed by TAPER structured?
      normalized geometry: does that structure survive into retrieval behavior?
    """
    primary_specs = {
        "common_macro_mean_r10_r50": True,
        "pre_norm_macro_category_gap": True,
        "normalized_macro_category_gap": True,
    }
    robustness_specs = {
        "pre_norm_min_category_gap": True,
        "normalized_min_category_gap": True,
        "pre_norm_positive_group_gap_fraction": True,
        "normalized_positive_group_gap_fraction": True,
        "pre_norm_direction_valid_fraction": True,
    }

    def rank_specs(specs):
        output = {}
        for metric, higher_is_better in specs.items():
            valid = [row for row in rows if row.get(metric) is not None]
            valid.sort(key=lambda row: row[metric], reverse=higher_is_better)
            output[metric] = [
                {
                    "rank": i + 1,
                    "teacher": row["teacher"],
                    "value": row[metric],
                }
                for i, row in enumerate(valid)
            ]
        return output

    return {
        "primary_evidence_descriptive_only": rank_specs(primary_specs),
        "robustness_descriptive_only": rank_specs(robustness_specs),
        "not_ranked_by_magnitude": [
            "single-caption retrieval sensitivity",
            "all-text removal sensitivity",
            "gradient magnitude",
            "raw delta norm",
            "published-native recall across non-identical protocols",
            "natural-caption compound additivity",
            "caption-order robustness",
        ],
    }


def pareto_front(rows: list[dict]) -> list[str]:
    """
    Descriptive shortlist only over three non-identical evidence views.
    Pending exact-intervention and BxL-feasibility gates prevent final lock.
    """
    keys = (
        "common_macro_mean_r10_r50",
        "pre_norm_macro_category_gap",
        "normalized_macro_category_gap",
    )
    candidates = [
        row for row in rows
        if all(row.get(key) is not None for key in keys)
        and row.get("gradient_access_probe") == "pass"
        and row.get("native_interface_parity") == "pass"
    ]

    front = []
    for row in candidates:
        dominated = False
        for other in candidates:
            if other is row:
                continue
            no_worse = all(other[key] >= row[key] for key in keys)
            strictly_better = any(other[key] > row[key] for key in keys)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(row["teacher"])
    return front


def ensure_cases_file(cases_path: Path, fashioniq_root: Path) -> None:
    if cases_path.exists():
        return

    captions_root = fashioniq_root / "captions"
    categories = ("dress", "shirt", "toptee")
    cases = []

    for category in categories:
        annotation_path = captions_root / f"cap.{category}.val.json"
        if not annotation_path.exists():
            raise FileNotFoundError(
                f"Cannot build audit cases; missing annotation file: {annotation_path}"
            )

        with annotation_path.open("r", encoding="utf-8") as file:
            records = json.load(file)

        for index, record in enumerate(records):
            caption_1, caption_2 = record["captions"]
            full_text = (
                f"{caption_1.strip('.?, ').capitalize()} and "
                f"{caption_2.strip('.?, ')}"
            )
            cases.append(
                {
                    "sample_id": f"fashioniq:val:{category}:{index}",
                    "category": category,
                    "reference_id": record["candidate"],
                    "target_id": record["target"],
                    "caption_1": caption_1,
                    "caption_2": caption_2,
                    "full_text": full_text,
                    "minus_1_text": caption_2.strip(".?, ").capitalize(),
                    "minus_2_text": caption_1.strip(".?, ").capitalize(),
                }
            )

    cases_path.parent.mkdir(parents=True, exist_ok=True)
    with cases_path.open("w", encoding="utf-8") as file:
        json.dump(cases, file, indent=2, ensure_ascii=False)

    print(f"Built {len(cases)} FashionIQ validation audit cases: {cases_path}")


def discover_checkpoint(
    explicit: Path | None,
    checkpoint_root: Path,
    teacher: str,
    preferred_name: str | None = None,
) -> Path:
    if explicit is not None:
        path = explicit.resolve()
        if not path.exists():
            raise FileNotFoundError(f"{teacher} checkpoint does not exist: {path}")
        return path

    checkpoint_root = checkpoint_root.resolve()
    if preferred_name is not None:
        preferred = checkpoint_root / preferred_name
        if preferred.exists():
            return preferred

    candidates = sorted(
        {
            *checkpoint_root.rglob("*.pth"),
            *checkpoint_root.rglob("*.pt"),
            *checkpoint_root.rglob("*.ckpt"),
        }
    )
    # Ignore obvious non-model result artifacts if any were placed nearby.
    candidates = [
        p for p in candidates
        if not any(
            token in p.name.lower()
            for token in ("artifact", "smoke", "metric", "result")
        )
    ]

    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"No {teacher} checkpoint found under {checkpoint_root}"
        )

    formatted = "\n".join(f"  - {p}" for p in candidates)
    raise RuntimeError(
        f"Multiple {teacher} checkpoints found. Pass --{teacher.lower()}-checkpoint "
        f"explicitly:\n{formatted}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the V6 shortlist audit for the currently integrated teacher adapters in isolated processes."
    )
    parser.add_argument(
        "--fashioniq-root",
        type=Path,
        default=Path("data/FashionIQ"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("teacher/audit/fashioniq_val_cases.json"),
    )

    parser.add_argument(
        "--env-config",
        type=Path,
        default=None,
        help=(
            "JSON mapping teacher names to isolated runtimes. "
            "Each entry may be {'conda_env':'NAME'} or "
            "{'python':'/abs/path/to/python'}. Extra entries are allowed."
        ),
    )
    parser.add_argument("--encoder-python", type=Path, default=None)
    parser.add_argument("--encoder-conda-env", type=str, default=None)
    parser.add_argument("--tme-python", type=Path, default=None)
    parser.add_argument("--tme-conda-env", type=str, default=None)
    parser.add_argument("--sprc-python", type=Path, default=None)
    parser.add_argument("--sprc-conda-env", type=str, default=None)
    parser.add_argument(
        "--hint-python",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--hint-conda-env",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--qure-python",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--qure-conda-env",
        type=str,
        default=None,
    )
    parser.add_argument("--encoder-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--encoder-root",
        type=Path,
        default=Path("teacher/repos/AAAI25-ENCODER"),
    )
    parser.add_argument(
        "--encoder-image-root",
        type=Path,
        default=None,
        help=(
            "Optional ENCODER-specific image root. If omitted, "
            "<fashioniq-root>/resized_image is preferred when present; "
            "otherwise <fashioniq-root>/images is used."
        ),
    )
    parser.add_argument(
        "--encoder-correction-root",
        type=Path,
        default=None,
        help=(
            "Optional. If omitted, full_audit.py searches the ENCODER repo, "
            "teacher/checkpoints/encoder, and data/FashionIQ."
        ),
    )

    parser.add_argument("--tme-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--tme-root",
        type=Path,
        default=Path("teacher/repos/TME"),
    )

    parser.add_argument("--sprc-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--sprc-root",
        type=Path,
        default=Path("teacher/repos/SPRC"),
    )
    parser.add_argument(
        "--hint-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--hint-root",
        type=Path,
        default=Path(
            "teacher/repos/ICASSP26-HINT"
        ),
    )

    parser.add_argument(
        "--hint-image-root",
        type=Path,
        default=None,
        help=(
            "Optional HINT-specific FashionIQ image root. "
            "If omitted, <fashioniq-root>/resized_image is "
            "preferred when present; otherwise "
            "<fashioniq-root>/images is used."
        ),
    )

    parser.add_argument(
        "--hint-correction-root",
        type=Path,
        default=None,
        help=(
            "Optional HINT FashionIQ correction-dictionary "
            "directory. If omitted, <fashioniq-root>/captions "
            "is used when it contains all three dictionaries."
        ),
    )
    parser.add_argument(
        "--qure-checkpoint",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--qure-root",
        type=Path,
        default=Path(
            "teacher/repos/QuRe"
        ),
    )

    parser.add_argument(
        "--qure-config",
        type=Path,
        default=None,
        help=(
            "QuRe FashionIQ eval config. "
            "Defaults to "
            "<qure-root>/configs/fashionIQ/eval.json."
        ),
    )
    parser.add_argument(
        "--sprc-backbone",
        choices=("pretrain", "pretrain_vitL"),
        default="pretrain_vitL",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("teacher/outputs/full_audit"),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gallery-batch-size", type=int, default=16)
    parser.add_argument("--score-batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=("Debug only: validation queries PER CATEGORY. "
              "Omit for the full FashionIQ validation set."),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    script = Path(__file__).with_name("full_audit.py").resolve()
    root = args.fashioniq_root.resolve()
    image_root = root / "images"
    split_root = root / "image_splits"
    cases = args.cases.resolve()
    ensure_cases_file(cases, root)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    qure_root = args.qure_root.resolve()

    if args.qure_config is not None:
        qure_config = (
            args.qure_config.resolve()
        )
    else:
        qure_config = (
            qure_root
            / "configs"
            / "fashionIQ"
            / "eval.json"
        ).resolve()

    if not qure_config.exists():
        raise FileNotFoundError(
            f"QuRe eval config not found: "
            f"{qure_config}"
        )
    env_config = load_env_config(args.env_config)
    runtimes = {
        "encoder": resolve_teacher_runtime(
            "encoder",
            args.encoder_python,
            args.encoder_conda_env,
            env_config,
        ),
        "tme": resolve_teacher_runtime(
            "tme",
            args.tme_python,
            args.tme_conda_env,
            env_config,
        ),
        "sprc": resolve_teacher_runtime(
            "sprc",
            args.sprc_python,
            args.sprc_conda_env,
            env_config,
        ),
        "hint": resolve_teacher_runtime(
            "hint",
            args.hint_python,
            args.hint_conda_env,
            env_config,
        ),
        "qure": resolve_teacher_runtime(
            "qure",
            args.qure_python,
            args.qure_conda_env,
            env_config,
        ),
    }
    validate_runtime_isolation(runtimes)
    for runtime in runtimes.values():
        probe_teacher_runtime(runtime)

    if not image_root.exists():
        raise FileNotFoundError(f"FashionIQ images not found: {image_root}")
    if not split_root.exists():
        raise FileNotFoundError(f"FashionIQ splits not found: {split_root}")

    if args.encoder_image_root is not None:
        encoder_image_root = args.encoder_image_root.resolve()
    if args.hint_image_root is not None:
        hint_image_root = (
            args.hint_image_root.resolve()
        )
    elif (root / "resized_image").exists():
        hint_image_root = (
            root / "resized_image"
        ).resolve()
    else:
        hint_image_root = image_root
    repo_root = Path(__file__).resolve().parents[2]
    encoder_checkpoint = discover_checkpoint(
        args.encoder_checkpoint,
        repo_root / "teacher/checkpoints/encoder",
        "ENCODER",
    )
    tme_checkpoint = discover_checkpoint(
        args.tme_checkpoint,
        repo_root / "teacher/checkpoints/tme",
        "TME",
        preferred_name="best_model.pth",
    )
    sprc_checkpoint = discover_checkpoint(
        args.sprc_checkpoint,
        repo_root / "teacher/checkpoints/sprc",
        "SPRC",
        preferred_name="sprc_fiq_vitl.pt",
    )
    hint_checkpoint = discover_checkpoint(
        args.hint_checkpoint,
        repo_root / "teacher/checkpoints/hint",
        "HINT",
    )

    qure_checkpoint = discover_checkpoint(
        args.qure_checkpoint,
        repo_root / "teacher/checkpoints/qure",
        "QuRe",
        preferred_name="model.pth",
    )

    print("Resolved inputs:")
    print("  FashionIQ images:", image_root)

    print(
        "  ENCODER images:",
        encoder_image_root,
    )

    print(
        "  ENCODER checkpoint:",
        encoder_checkpoint,
    )

    print(
        "  TME checkpoint:",
        tme_checkpoint,
    )

    print(
        "  SPRC checkpoint:",
        sprc_checkpoint,
    )

    print(
        "  TG-CIR checkpoint:",
        tgcir_checkpoint,
    )

    print(
        "  CSMCIR checkpoint:",
        csmcir_checkpoint,
    )

    print("Resolved isolated runtimes:")

    for teacher, runtime in runtimes.items():
        print(
            f"  {teacher}: "
            f"{runtime['kind']}="
            f"{runtime['value']} "
            f"(source={runtime['source']})"
        )
    print("Resolved isolated runtimes:")
    for teacher, runtime in runtimes.items():
        print(
            f"  {teacher}: {runtime['kind']}={runtime['value']} "
            f"(source={runtime['source']})"
        )

    common = [
        "--cases", str(cases),
        "--split-root", str(split_root),
        "--batch-size", str(args.batch_size),
        "--gallery-batch-size", str(args.gallery_batch_size),
        "--score-batch-size", str(args.score_batch_size),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--seed", str(args.seed),
        "--device", args.device,
    ]
    if args.limit is not None:
        common += ["--limit", str(args.limit)]

    encoder_correction_root = (
        args.encoder_correction_root
    )

    if encoder_correction_root is None:
        candidate = root / "captions"

        required = [
            candidate / "correction_dict_dress.json",
            candidate / "correction_dict_shirt.json",
            candidate / "correction_dict_toptee.json",
        ]

        if all(path.exists() for path in required):
            encoder_correction_root = candidate

    hint_correction_root = (
        args.hint_correction_root
    )

    if hint_correction_root is None:
        candidate = root / "captions"

        required = [
            candidate / "correction_dict_dress.json",
            candidate / "correction_dict_shirt.json",
            candidate / "correction_dict_toptee.json",
        ]

        if all(path.exists() for path in required):
            hint_correction_root = candidate
    jobs = [
        (
            "encoder",
            runtimes["encoder"]["prefix"],
            [
                "--checkpoint", str(encoder_checkpoint),
                "--image-root", str(encoder_image_root),
                "--encoder-root", str(args.encoder_root.resolve()),
                *(
                    [
                        "--correction-root",
                        str(encoder_correction_root.resolve()),
                    ]
                    if encoder_correction_root is not None
                    else []
                ),
            ],
        ),
        (
            "tme",
            runtimes["tme"]["prefix"],
            [
                "--checkpoint", str(tme_checkpoint),
                "--image-root", str(image_root),
                "--tme-root", str(args.tme_root.resolve()),
            ],
        ),
        (
            "sprc",
            runtimes["sprc"]["prefix"],
            [
                "--checkpoint", str(sprc_checkpoint),
                "--image-root", str(image_root),
                "--sprc-root", str(args.sprc_root.resolve()),
                "--backbone", args.sprc_backbone,
            ],
        ),
        (
            "hint",
            runtimes["hint"]["prefix"],
            [
                "--checkpoint",
                str(hint_checkpoint),

                "--image-root",
                str(hint_image_root),

                "--hint-root",
                str(
                    args.hint_root.resolve()
                ),

                *(
                    [
                        "--correction-root",
                        str(
                            hint_correction_root.resolve()
                        ),
                    ]
                    if hint_correction_root is not None
                    else []
                ),
            ],
        ),
        (
            "qure",
            runtimes["qure"]["prefix"],
            [
                "--checkpoint",
                str(qure_checkpoint),

                "--image-root",
                str(image_root),

                "--qure-root",
                str(qure_root),

                "--qure-config",
                str(qure_config),
            ],
        ),
    ]

    report_paths = []
    for teacher, runtime_prefix, extra in jobs:
        report = output_root / teacher / "full_metrics.json"
        artifact = output_root / teacher / "full_artifact.pt"
        report.parent.mkdir(parents=True, exist_ok=True)
        command = [
            *runtime_prefix,
            str(script),
            "--teacher", teacher,
            "--output", str(report),
            "--artifact-output", str(artifact),
            *common,
            *extra,
        ]
        run(command)
        report_paths.append(report)

    reports = [load_json(path) for path in report_paths]
    rows = [get_summary(report) for report in reports]

    pairwise_uncertainty = build_pairwise_uncertainty(
        reports=reports,
        samples=args.bootstrap_samples,
        seed=args.seed + 50000,
    )

    lock_blockers = {}
    for row in rows:
        blockers = []
        if row["gradient_access_probe"] != "pass":
            blockers.append("gradient_access_probe_failed")
        if row["native_interface_parity"] != "pass":
            blockers.append("native_interface_parity_failed")
        if row["pre_norm_valid_category_count"] != len(CATEGORIES):
            blockers.append("pre_norm_geometry_not_estimable_in_all_categories")
        if row["normalized_valid_category_count"] != len(CATEGORIES):
            blockers.append("normalized_geometry_not_estimable_in_all_categories")
        if row["checkpoint_load_status"] != "clean":
            blockers.append("checkpoint_load_requires_review")
        if row["repo_snapshot_match"] is not True:
            blockers.append("upstream_repo_snapshot_not_verified")
        if row["repo_tracked_clean"] is not True:
            blockers.append("upstream_repo_tracked_cleanliness_not_verified")
        # Natural FashionIQ whole-caption deletion is only a screening proxy.
        # Final lock requires controlled atomic evidence and the exact train-time
        # intervention path.
        if row["controlled_atomic_geometry"] != "pass":
            blockers.append("controlled_atomic_geometry_not_passed")
        if row["teacher_information_path_review"] != "pass":
            blockers.append("teacher_information_path_review_not_passed")
        if row["exact_intervention_credibility"] != "pass":
            blockers.append("exact_taper_intervention_credibility_not_passed")
        if row["counterfactual_training_feasibility"] != "pass":
            blockers.append("BxL_counterfactual_training_feasibility_not_passed")
        lock_blockers[row["teacher"]] = blockers

    tournament = {
        "status": "teacher_shortlist_audit_v6_final_lock_pending",
        "scientific_scope": (
            "This run can produce an evidence-based shortlist from hard/text-"
            "omission counterfactuals. It MUST NOT declare a final TAPER "
            "teacher until the exact differentiable intervention and BxL "
            "training-feasibility gates are executed for the leading "
            "candidate(s)."
        ),
        "selection_policy": (
            "No arbitrary weighted scalar. Primary evidence has three views: "
            "common FashionIQ retrieval competence; category-balanced PRE-NORM "
            "same-vs-different edit geometry because TAPER consumes that delta; "
            "and L2-normalized-query edit geometry because native retrieval is "
            "angular and pre-norm structure can otherwise be retrieval-inert. "
            "Natural FashionIQ repeated-caption geometry is a screening proxy, "
            "not proof of atomic edit factors. Teacher caption-ablation "
            "sensitivity is required functional-validity evidence but is NOT "
            "maximized as a score because natural captions can be redundant or "
            "compound. Final lock additionally requires controlled atomic and "
            "exact-intervention evidence."
        ),
        "candidate_pruning_policy": (
            "Do not permanently remove a candidate from a single point metric. "
            "For the five-candidate pool, first run the corrected common natural "
            "screen on all candidates. Prune immediately only on fatal fidelity/"
            "accessibility failure; otherwise prune only under strong multi-axis "
            "dominance evidence with paired uncertainty and no compensating "
            "intervention/feasibility advantage. Expensive exact-intervention "
            "triangulation may then be restricted to the survivors."
        ),
        "shortlist_pareto_front_descriptive": pareto_front(rows),
        "final_lock_ready": any(
            len(blockers) == 0 for blockers in lock_blockers.values()
        ),
        "final_lock_blockers": lock_blockers,
        "required_before_final_teacher_lock": [
            "controlled atomic-edit geometry independent of learned TAPER slots",
            "teacher text/composer information-flow review",
            "exact teacher-native differentiable intervention no-op identity",
            "hard omission vs exact intervention effect-direction agreement",
            "target-margin/rank or candidate-score-change agreement",
            "no uncontrolled text-information bypass / contextual leakage audit",
            "representative B x L counterfactual peak VRAM and throughput",
        ],
        "pairwise_uncertainty": pairwise_uncertainty,
        "strong_multi_axis_dominance": strong_multi_axis_dominance(
            pairwise_uncertainty
        ),
        "metric_ranks": metric_ranks(rows),
        "runtime_isolation": {
            teacher: {
                "kind": runtime["kind"],
                "value": runtime["value"],
                "source": runtime["source"],
            }
            for teacher, runtime in runtimes.items()
        },
        "teachers": rows,
    }

    tournament_json = output_root / "tournament_full.json"
    tournament_md = output_root / "tournament_full.md"
    with tournament_json.open("w", encoding="utf-8") as file:
        json.dump(tournament, file, indent=2, ensure_ascii=False)

    table = markdown_table(rows)
    with tournament_md.open("w", encoding="utf-8") as file:
        file.write("# TAPER Teacher Shortlist Audit V6\n\n")
        file.write(table)
        file.write("\n\n")
        file.write(
            "Primary evidence = common retrieval + PRE-NORM edit geometry + "
            "normalized-query edit geometry. Functional deletion sensitivity "
            "is validity evidence, not a magnitude race.\n\n"
        )
        file.write(
            "> This is a shortlist report, NOT final teacher lock. Exact "
            "TAPER-compatible intervention credibility and representative BxL "
            "training feasibility are mandatory before final selection.\n"
        )

    print()
    print("=== TAPER TEACHER SHORTLIST AUDIT V6 ===")
    print()
    print(table)
    print()
    print(
        "Descriptive shortlist Pareto front:",
        tournament["shortlist_pareto_front_descriptive"],
    )
    print("Lock blockers:", tournament["final_lock_blockers"])
    print("Saved:", tournament_json)
    print("Saved:", tournament_md)


if __name__ == "__main__":
    main()
