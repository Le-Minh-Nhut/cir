"""Read-only API and cache for CIRCUS composition evaluation artifacts."""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.composition.data import RELEASE_PROTOCOL_ID, load_release
from workbench.composition.metrics import SHORTCUT_K, compose, hit_at_k, ndcg_full

logger = logging.getLogger("workbench.composition")

SUBSETS = ("full", "sf", "validated")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")


def resolve_composition_path(artifacts_root: Path, relative: str | Path) -> Path:
    """Resolve a path safely inside <artifacts>/composition/, refusing traversal."""
    artifacts = Path(artifacts_root).expanduser().resolve()
    composition = (artifacts / "composition").resolve()
    target = (composition / relative).resolve()
    if not (str(target) == str(composition) or str(target).startswith(str(composition) + "/")):
        raise ValueError(f"Path traversal refused: {relative!r} is outside {composition}")
    return target


def validate_summary_dict(data: Any) -> tuple[bool, str | None]:
    """Validate that summary JSON conforms to composition summary schema."""
    if not isinstance(data, dict):
        return False, "summary must be a JSON object"
    required = ("model_id", "checkpoint_id", "run_id", "subset", "conditions", "metrics")
    for key in required:
        if key not in data:
            return False, f"missing required summary field: {key}"
    if data["subset"] not in SUBSETS:
        return False, f"invalid subset {data['subset']!r}; expected one of {SUBSETS}"
    if not isinstance(data.get("metrics"), dict):
        return False, "summary metrics must be an object"
    return True, None


def training_or_transfer_label(summary: Mapping[str, Any]) -> str:
    """Derive concise scientific training / transfer label."""
    if summary.get("checkpoint_transfer"):
        ds = summary.get("checkpoint_training_dataset") or "cross-dataset"
        return f"{ds} transfer"
    train_noise = summary.get("checkpoint_training_noise_pct")
    eval_noise = summary.get("evaluation_noise_pct")
    parts = []
    if train_noise is not None:
        parts.append(f"train noise {train_noise}%")
    else:
        parts.append("train noise unstated")
    if eval_noise is not None:
        parts.append(f"eval noise {eval_noise}%")
    return " · ".join(parts)


# Cache keyed by path -> (mtime, size, parsed_summary)
_SUMMARY_FILE_CACHE: dict[str, tuple[float, int, dict[str, Any]]] = {}
_RELEASE_CACHE: dict[str, Any] = {}


def load_cached_release(root: Path | None = None) -> Any:
    """Load and cache release data."""
    if root is None:
        config = resolve_config()
        # Check standard locations
        candidates = [
            getattr(config, "WORKBENCH_CIRCUS_RELEASE_ROOT", None),
            config.WORKBENCH_ARTIFACTS_ROOT / "circus_release",
            Path(config.CIR_REPO_ROOT) / "data" / "circus_release",
        ]
        root = next((Path(c) for c in candidates if c and Path(c).is_dir()), None)
    if root is None or not root.is_dir():
        return None
    cache_key = str(root.resolve())
    if cache_key in _RELEASE_CACHE:
        return _RELEASE_CACHE[cache_key]
    try:
        rel = load_release(root, verify_hashes=False, expect_released_counts=False)
        _RELEASE_CACHE[cache_key] = rel
        return rel
    except Exception as exc:
        logger.warning("Failed to load circus release from %s: %s", root, exc)
        return None


def read_summary_file(path: Path) -> dict[str, Any] | None:
    """Read and validate a summary file with mtime caching."""
    try:
        stat = path.stat()
    except OSError:
        return None
    path_str = str(path.resolve())
    cached = _SUMMARY_FILE_CACHE.get(path_str)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return cached[2]

    try:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
    except Exception as exc:
        logger.warning("Failed to read composition summary %s: %s", path, exc)
        return None

    valid, err = validate_summary_dict(data)
    if not valid:
        logger.warning("Invalid composition summary %s: %s", path, err)
        return None

    metrics = data.get("metrics")
    if isinstance(metrics, dict):
        mm = metrics.get("multimodal")
        if isinstance(mm, dict):
            if "ndcg_multimodal" not in metrics and "ndcg_full" in mm:
                metrics["ndcg_multimodal"] = mm["ndcg_full"]
            if "recall_at_10_multimodal" not in metrics and "recall_at_10" in mm:
                metrics["recall_at_10_multimodal"] = mm["recall_at_10"]
            if "recall_at_50_multimodal" not in metrics and "recall_at_50" in mm:
                metrics["recall_at_50_multimodal"] = mm["recall_at_50"]
            if "coverage" not in metrics and "coverage" in mm:
                metrics["coverage"] = mm["coverage"]
        txt = metrics.get("text_only")
        if isinstance(txt, dict):
            if "ndcg_text_only" not in metrics and "ndcg_full" in txt:
                metrics["ndcg_text_only"] = txt["ndcg_full"]
        img = metrics.get("image_only")
        if isinstance(img, dict):
            if "ndcg_image_only" not in metrics and "ndcg_full" in img:
                metrics["ndcg_image_only"] = img["ndcg_full"]
        if "outcomes" not in metrics and "shortcut_outcomes" in metrics:
            metrics["outcomes"] = metrics["shortcut_outcomes"]

    data["is_failed"] = is_failed_summary(data)
    data["training_or_transfer_label"] = training_or_transfer_label(data)
    if "reproduction_status" not in data:
        data["reproduction_status"] = data.get("checkpoint_labels", {}).get("reproduction_status") or "unknown"
    runtime = data.get("runtime")
    model_interp = runtime.get("model_interpreter") if isinstance(runtime, dict) else None
    data["interpreter_verification_status"] = (
        model_interp.get("verification_status") if isinstance(model_interp, dict) and model_interp.get("verification_status")
        else "unknown"
    )
    _SUMMARY_FILE_CACHE[path_str] = (stat.st_mtime, stat.st_size, data)
    return data


def is_fixture_summary(summary: Mapping[str, Any]) -> bool:
    """True when a summary is a layout fixture rather than a measurement."""
    return bool(
        summary.get("is_synthetic")
        or summary.get("synthetic")
        or summary.get("synthetic_fixture")
        or str(summary.get("run_id") or "").startswith("SYNTHETIC_")
    )

def is_smoke_summary(summary: Mapping[str, Any]) -> bool:
    """True when a measurement scored only a query-limited smoke slice."""
    if is_fixture_summary(summary):
        return False  # a fixture is not a measurement at all; it has its own flag
    smoke = summary.get("smoke")
    return bool(isinstance(smoke, Mapping) and smoke.get("limited"))

# `execution_status` values the composition runner can emit: COMPLETED, PARTIAL, REUSED, FAILED
# (workbench/composition/runner.py:985-986 plus the ``_write_failed_results`` default). Only
# FAILED means the run produced no numbers; MODEL_UNAVAILABLE / EXACT_RANK_UNAVAILABLE /
# ABLATION_UNSUPPORTED are per-condition statuses, never top-level execution statuses.
_FAILED_EXECUTION_STATUSES = frozenset({"FAILED"})
_METRIC_KEYS = frozenset({"ndcg_full", "recall_at_10", "recall_at_50", "mrr"})


def _has_measured_numbers(metrics: Any) -> bool:
    """True when a summary carries at least one non-null per-condition score."""
    if not isinstance(metrics, Mapping):
        return False
    for key, value in metrics.items():
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        if numeric and (str(key).startswith("ndcg_") or str(key).startswith("recall_at_")):
            return True
        if isinstance(value, Mapping):
            if any(k in _METRIC_KEYS and isinstance(v, (int, float)) and not isinstance(v, bool)
                   for k, v in value.items()):
                return True
    return False


def is_failed_summary(summary: Mapping[str, Any]) -> bool:
    """True when a run reports failure or yields no numbers at all."""
    if is_fixture_summary(summary):
        return False  # fixtures are reported synthetic, not failed
    if str(summary.get("execution_status") or "").upper() in _FAILED_EXECUTION_STATUSES:
        return True
    return not _has_measured_numbers(summary.get("metrics"))

def readable_failure_message(message: str | None) -> str | None:
    """Failure text for humans: an unresolved category is not the Python sentinel ``None``.

    The runner now writes "category unresolved"; rank artifacts recorded before that fix still
    carry the ``None`` repr, so the same phrase is canonicalized on read as well.
    """
    if not message:
        return message
    return message.replace("category None has no adapter embedding scope",
                           "category unresolved has no adapter embedding scope")

def list_composition_summaries(
    artifacts_root: Path | None = None,
    subset: str | None = None,
    include_fixtures: bool = True,
    include_smoke: bool = True,
) -> list[dict[str, Any]]:
    """Scan <artifacts>/composition/summary/ and return validated summaries.

    Ordering: measured runs, then failed runs (visible, never ranked as if they produced numbers),
    then smoke-limited runs, then synthetic fixtures, with run_id breaking ties. A fixture can
    never sort above measured data on a byte-order accident, a failed run can never outrank a run
    that measured something, and a query-limited run can never outrank a full measurement it does
    not measure.
    """
    if artifacts_root is None:
        artifacts_root = resolve_config().WORKBENCH_ARTIFACTS_ROOT
    try:
        summary_dir = resolve_composition_path(artifacts_root, "summary")
    except ValueError:
        return []
    if not summary_dir.is_dir():
        return []

    results: list[dict[str, Any]] = []
    # summary structure: <summary_dir>/<run_dir>/<subset>.json
    for run_dir in sorted(summary_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        subsets_to_check = [subset] if subset else SUBSETS
        for sub in subsets_to_check:
            sub_file = run_dir / f"{sub}.json"
            if sub_file.is_file():
                summary = read_summary_file(sub_file)
                if summary is None:
                    continue
                if not include_fixtures and is_fixture_summary(summary):
                    continue
                if not include_smoke and is_smoke_summary(summary):
                    continue
                results.append(summary)
    def rank(summary: Mapping[str, Any]) -> tuple[int, str, str]:
        if is_fixture_summary(summary):
            tier = 3
        elif is_failed_summary(summary):
            tier = 1  # visible, but never ranked as if it produced numbers
        elif is_smoke_summary(summary):
            tier = 2
        else:
            tier = 0
        return tier, str(summary.get("run_id") or ""), str(summary.get("subset") or "")

    results.sort(key=rank)
    return results


def get_composition_run_detail(
    run_id: str,
    subset: str = "full",
    artifacts_root: Path | None = None,
) -> dict[str, Any] | None:
    """Get full summary for a specific run_id and subset."""
    if artifacts_root is None:
        artifacts_root = resolve_config().WORKBENCH_ARTIFACTS_ROOT
    try:
        summary_dir = resolve_composition_path(artifacts_root, "summary")
    except ValueError:
        return None
    if not summary_dir.is_dir():
        return None

    # Search for matching summary file across run directories
    for run_dir in summary_dir.iterdir():
        if not run_dir.is_dir():
            continue
        target_file = run_dir / f"{subset}.json"
        if target_file.is_file():
            summary = read_summary_file(target_file)
            if summary and summary.get("run_id") == run_id:
                # Add available subsets list
                available = [s for s in SUBSETS if (run_dir / f"{s}.json").is_file()]
                summary["available_subsets"] = available
                return summary
    return None


def determine_query_outcome(
    ranks: Mapping[str, int | None],
    k: int = SHORTCUT_K,
) -> str:
    """Classify query outcome for K=10."""
    mm = ranks.get("multimodal")
    t = ranks.get("text_only_black_image")
    i = ranks.get("image_only_empty_text")
    if mm is None or t is None or i is None:
        return "unscored"
    mm_hit = mm <= k
    t_hit = t <= k
    i_hit = i <= k
    if mm_hit and not t_hit and not i_hit:
        return "composition_required"
    if t_hit and not i_hit:
        return "text_only"
    if i_hit and not t_hit:
        return "image_only"
    if t_hit and i_hit:
        return "both"
    return "unresolved"


def get_composition_run_queries(
    run_id: str,
    subset: str = "full",
    condition: str | None = None,
    category: str | None = None,
    outcome: str | None = None,
    min_rank: int | None = None,
    max_rank: int | None = None,
    query_id: str | None = None,
    order: str = "worst",
    limit: int = 50,
    offset: int = 0,
    artifacts_root: Path | None = None,
) -> dict[str, Any]:
    """Retrieve paginated per-query composition details with filtering."""
    if artifacts_root is None:
        artifacts_root = resolve_config().WORKBENCH_ARTIFACTS_ROOT

    # 1. Locate ranks file for run_id
    try:
        run_ranks_dir = resolve_composition_path(artifacts_root, f"runs/{run_id}/ranks")
    except ValueError as exc:
        return {"items": [], "total": 0, "limit": limit, "offset": offset, "error": str(exc)}

    ranks_data: dict[str, Any] | None = None
    if run_ranks_dir.is_dir():
        for candidate in sorted(run_ranks_dir.glob("*.json")):
            try:
                content = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(content, dict) and "ranks" in content:
                    ranks_data = content
                    break
            except Exception:
                continue

    # 2. Load queries from release data
    release = load_cached_release()
    if release is None:
        return {
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "queries_unavailable_reason": "CIRCUS release query dataset not available locally.",
        }

    ranks_by_condition = ranks_data.get("ranks", {}) if ranks_data else {}
    failures_by_condition = ranks_data.get("query_failures", {}) if ranks_data else {}
    target_condition = condition or "multimodal"

    # Filter release queries to subset
    subset_queries = [q for q in release.queries if subset in q.subsets]
    items: list[dict[str, Any]] = []

    for q in subset_queries:
        q_ranks = {
            c: ranks_by_condition.get(c, {}).get(q.qid)
            for c in CONDITIONS
        }
        q_outcome = determine_query_outcome(q_ranks)
        target_rank = q_ranks.get(target_condition)

        # Filters
        if category and q.category != category:
            continue
        if outcome and q_outcome != outcome:
            continue
        if query_id and q.qid != query_id:
            continue
        if min_rank is not None and (target_rank is None or target_rank < min_rank):
            continue
        if max_rank is not None and (target_rank is None or target_rank > max_rank):
            continue

        ref_name = release.candidate_names.get(q.reference_id, q.reference_id)
        pos_names = [release.candidate_names.get(p, p) for p in q.positive_ids]
        failure_msg = readable_failure_message(failures_by_condition.get(target_condition, {}).get(q.qid))

        items.append({
            "query_id": q.qid,
            "category": q.category,
            "query_txt": q.query_txt,
            "reference_id": ref_name,
            "positive_ids": pos_names,
            "ranks": q_ranks,
            "target_rank": target_rank,
            "outcome": q_outcome,
            "failure": failure_msg,
            "top_results": [],
        })

    # Sorting
    if order == "worst":
        items.sort(key=lambda x: (x["target_rank"] is not None, -(x["target_rank"] or 0)))
    elif order in ("best", "asc"):
        items.sort(key=lambda x: (x["target_rank"] is None, x["target_rank"] or 0))
    elif order == "desc":
        items.sort(key=lambda x: (x["target_rank"] is None, -(x["target_rank"] or 0)))

    total = len(items)
    page_items = items[offset : offset + limit]

    result = {
        "items": page_items,
        "total": total,
        "limit": limit,
        "offset": offset,
    }
    if not ranks_data:
        result["queries_unavailable_reason"] = (
            "Per-query exact ranks not found on disk for this run (aggregate-only or pending run)."
        )
    return result


def create_fixture_summaries(artifacts_root: Path) -> list[Path]:
    """Generate deterministic synthetic composition fixtures into artifacts/composition.

    Unmistakably marked as FIXTURE / SYNTHETIC.
    """
    comp_dir = resolve_composition_path(artifacts_root, "")
    summary_dir = comp_dir / "summary"
    runs_dir = comp_dir / "runs"
    summary_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    # Create two fixture models: fixture_csmcir and fixture_encoder
    models_spec = [
        {
            "model_id": "fixture_encoder",
            "checkpoint_id": "fashioniq",
            "run_id": "SYNTHETIC_encoder_fashioniq_fixture001",
            "training_noise": None,
            "eval_noise": 0,
            "transfer": False,
            "dataset": "FashionIQ",
            "mm_ndcg": 0.2506,
            "t_ndcg": 0.1202,
            "i_ndcg": 0.0849,
            "gap": 0.5204,
            "r10": 12.5,
            "r50": 25.0,
            "coverage": 1.0,
        },
        {
            "model_id": "fixture_habit",
            "checkpoint_id": "fiq_n02",
            "run_id": "SYNTHETIC_habit_fiq_n02_fixture002",
            "training_noise": 20,
            "eval_noise": 0,
            "transfer": False,
            "dataset": "FashionIQ",
            "mm_ndcg": 0.2881,
            "t_ndcg": 0.1520,
            "i_ndcg": 0.1130,
            "gap": 0.2856,
            "r10": 15.0,
            "r50": 30.0,
            "coverage": 0.98,
        },
    ]

    for spec in models_spec:
        run_name = spec["run_id"]
        run_summary_dir = summary_dir / run_name
        run_summary_dir.mkdir(parents=True, exist_ok=True)

        for sub in SUBSETS:
            path = run_summary_dir / f"{sub}.json"
            qcount = {"full": 6003, "sf": 4069, "validated": 586}[sub]
            summary_payload = {
                "schema_version": 1,
                "is_synthetic": True,
                "synthetic": True,
                "synthetic_fixture": True,
                "model_id": spec["model_id"],
                "checkpoint_id": spec["checkpoint_id"],
                "run_id": spec["run_id"],
                "protocol_id": RELEASE_PROTOCOL_ID,
                "subset": sub,
                "gallery_count": 74381,
                "query_count": qcount,
                "query_count_per_subset": {"full": 6003, "sf": 4069, "validated": 586},
                "conditions": list(CONDITIONS),
                "execution_status": "COMPLETED",
                "condition_status": {c: {"status": "COMPLETED", "unscored_count": 0} for c in CONDITIONS},
                "checkpoint_training_noise_pct": spec["training_noise"],
                "evaluation_noise_pct": spec["eval_noise"],
                "checkpoint_transfer": spec["transfer"],
                "checkpoint_training_dataset": spec["dataset"],
                "reproduction_status": "FIXTURE_DEVELOPMENT_ONLY",
                "checkpoint_labels": {
                    "checkpoint_training_noise_pct": spec["training_noise"],
                    "evaluation_noise_pct": spec["eval_noise"],
                    "reproduction_status": "FIXTURE_DEVELOPMENT_ONLY",
                },
                "smoke": {"limited": True, "query_limit": 8, "label": "FIXTURE ONLY"},
                "metrics": {
                    "ndcg_multimodal": spec["mm_ndcg"],
                    "ndcg_text_only": spec["t_ndcg"],
                    "ndcg_image_only": spec["i_ndcg"],
                    "composition_gap_ndcg": spec["gap"],
                    "recall_at_1_multimodal": round(spec["r10"] / 2.0, 2),
                    "recall_at_10_multimodal": spec["r10"],
                    "recall_at_50_multimodal": spec["r50"],
                    "mrr_gap": 0.12,
                    "coverage": spec["coverage"],
                    "common_cohort_count": qcount,
                    "outcomes": {
                        "composition_required": int(qcount * 0.18),
                        "text_only": int(qcount * 0.08),
                        "image_only": int(qcount * 0.05),
                        "both": int(qcount * 0.12),
                        "unresolved": int(qcount * 0.55),
                        "neither_scored": int(qcount * (1.0 - spec["coverage"])),
                    },
                },
            }
            path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
            written.append(path)

        # Write sample mock ranks for query drilldown
        run_ranks = runs_dir / spec["run_id"] / "ranks"
        run_ranks.mkdir(parents=True, exist_ok=True)
        ranks_file = run_ranks / f"{spec['model_id']}_{spec['checkpoint_id']}.json"
        # Seed deterministic ranks for the first few queries
        sample_qids = [f"7:{idx}" for idx in range(1, 20)]
        mock_ranks = {
            "multimodal": {qid: ((idx * 3) % 40) + 1 for idx, qid in enumerate(sample_qids)},
            "text_only_black_image": {qid: ((idx * 7) % 80) + 1 for idx, qid in enumerate(sample_qids)},
            "image_only_empty_text": {qid: ((idx * 11) % 100) + 1 for idx, qid in enumerate(sample_qids)},
        }
        ranks_payload = {
            "fingerprint": "fixture_fingerprint",
            "attempt": spec["run_id"],
            "ranks": mock_ranks,
            "query_failures": {},
            "condition_status": {c: {"status": "COMPLETED"} for c in CONDITIONS},
        }
        ranks_file.write_text(json.dumps(ranks_payload, indent=2), encoding="utf-8")
        written.append(ranks_file)

    return written
