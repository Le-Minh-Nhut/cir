"""Read-only API and cache for CIRCUS composition evaluation artifacts."""
from __future__ import annotations

import copy
import hashlib
import json
import logging
from dataclasses import asdict
from pathlib import Path
from collections.abc import Sequence
from typing import Any, Mapping

from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.composition.data import RELEASE_PROTOCOL_ID, load_release
from workbench.composition.inputs import ABLATION_CONDITIONS, condition_label
from workbench.composition.metrics import SHORTCUT_K, compose, hit_at_k, ndcg_full

logger = logging.getLogger("workbench.composition")

SUBSETS = ("full", "sf", "validated")
CONDITIONS = ("multimodal", "text_only_black_image", "image_only_empty_text")

def run_conditions(summary: Mapping[str, Any] | None) -> tuple[str, ...] | None:
    """The conditions the RUN declared, protocol names first; ``None`` when it recorded none.

    `summary["conditions"]` is what the runner actually ran, so it -- not the module constant -- is
    the set of conditions a consumer may advertise. ``None`` (a summary that predates the field)
    is distinct from an empty tuple: the caller then falls back to the protocol triple instead of
    serving nothing.
    """
    if not isinstance(summary, Mapping):
        return None
    declared = summary.get("conditions")
    if not isinstance(declared, (list, tuple)) or not declared:
        return None
    names = [str(name) for name in declared]
    return tuple([name for name in CONDITIONS if name in names]
                 + [name for name in names if name not in CONDITIONS])

def displayed_conditions(summary: Mapping[str, Any] | None) -> tuple[str, ...]:
    """The conditions to RENDER for one run: those it declared, plus any ablation it scored.

    `CONDITIONS` stays the protocol's three names -- `determine_query_outcome` reads two of them
    positionally -- and it is only the FALLBACK, for a summary that predates `conditions`. Prepending
    it unconditionally advertised `image_only_empty_text` for a run that measured three conditions
    (protocol, text-only, and an adapter's named image ablation), whose selector entry then served
    nothing but `unscored` rows. An adapter that also scored an ablation
    (`ABLATION_CONDITIONS`, e.g. `clvc_image_encoder_only`) declares it, so its real row still
    renders. Order is protocol-first, then declared order, so nothing is ever renamed.
    """
    declared = run_conditions(summary)
    scored: set[str] = set()
    if isinstance(summary, Mapping):
        for field in ("condition_status", "conditions"):
            value = summary.get(field)
            if isinstance(value, Mapping):
                scored.update(str(name) for name in value)
            elif isinstance(value, (list, tuple)):
                scored.update(str(name) for name in value)
        metrics = summary.get("metrics")
        if isinstance(metrics, Mapping):
            # `metrics` keys its buckets by ROLE (`multimodal`, `text_only`, `image_only`), and each
            # bucket names the condition that filled that role in its own `condition` field.
            for role in ("multimodal", "text_only", "image_only"):
                bucket = metrics.get(role)
                if isinstance(bucket, Mapping) and isinstance(bucket.get("condition"), str):
                    scored.add(bucket["condition"])
    if declared is None:
        extra = tuple(name for name in ABLATION_CONDITIONS
                      if name in scored and name not in CONDITIONS)
        return (*CONDITIONS, *extra)
    # A condition the run declared, plus one it recorded a measurement for under a name its
    # declaration omits -- both were measured, so both are renderable.
    undeclared = sorted(scored - set(declared),
                        key=lambda name: (ABLATION_CONDITIONS.index(name)
                                          if name in ABLATION_CONDITIONS else len(ABLATION_CONDITIONS),
                                          name))
    return (*declared, *undeclared)

def condition_display_label(condition: str) -> str:
    """A condition's human label; an unknown name is shown verbatim rather than dropped."""
    try:
        return condition_label(condition)
    except ValueError:
        return condition


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
    """Read and validate a summary file with mtime caching.

    The cache holds the PARSED dict; every caller gets its own deep copy. The cache used to hand the
    same object to everyone, and `list_composition_summaries` decorates rows in place
    (`decorate_summary`, `annotate_duplicate_attempts`, `decorate_run_rows`), so whichever route
    warmed the cache decided which derived blocks a later detail response carried -- the run-detail
    fetch was 20 053 B without `subset_provenance` and 20 841 B with it, depending only on whether
    the list route had run first in this process. A response must not depend on an earlier request.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    path_str = str(path.resolve())
    cached = _SUMMARY_FILE_CACHE.get(path_str)
    if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
        return copy.deepcopy(cached[2])

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
            if "recall_at_1_multimodal" not in metrics and "recall_at_1" in mm:
                metrics["recall_at_1_multimodal"] = mm["recall_at_1"]
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
    return copy.deepcopy(data)

def _cached_summary_copy(path_str: str) -> dict[str, Any] | None:
    """A fresh, UNDECORATED copy of a cached summary, for a caller that decorates in place.

    Two routes decorate the same parsed document with DIFFERENT derived blocks: the list route adds
    `subset_provenance` (via `decorate_run_rows`) and the detail route does not. Handing both the same
    object made the detail response depend on whether the list route had already run -- 80 104 B
    without the block, 80 927 B with it, in the same process. Every caller now receives its own copy,
    so the two routes are independent, and a caller that mutates its copy cannot change what the next
    request reads.
    """
    cached = _SUMMARY_FILE_CACHE.get(path_str)
    return copy.deepcopy(cached[2]) if cached else None


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

#: Provenance values that mean "this row is not one of our evaluations". Matched on what the row
#: declares about itself, never on a model name: an artifact imported from a released pipeline is
#: real data, but it is not evidence about a model we ran, and it must not appear among default
#: research results because its numbers would be read as ours.
IMPORTED_PROVENANCE_SOURCES = frozenset({
    "released_audit_retriever",
    "released_artifact_import",
    "external_artifact",
})
_IMPORTED_REPRODUCTION_STATUSES = frozenset({"RELEASED_ARTIFACT_IMPORT"})

def is_imported_summary(summary: Mapping[str, Any]) -> bool:
    """True when a row was imported from an external artifact rather than produced here."""
    if is_fixture_summary(summary):
        return False
    source = str(summary.get("provenance_source") or "").strip().lower()
    if source in IMPORTED_PROVENANCE_SOURCES:
        return True
    return str(summary.get("reproduction_status") or "").strip().upper() in _IMPORTED_REPRODUCTION_STATUSES

# `execution_status` values the composition runner can emit: COMPLETED, PARTIAL, REUSED, FAILED
# (workbench/composition/runner.py:985-986 plus the ``_write_failed_results`` default). Only
# FAILED means the run produced no numbers; MODEL_UNAVAILABLE / EXACT_RANK_UNAVAILABLE /
# ABLATION_UNSUPPORTED are per-condition statuses, never top-level execution statuses.
_FAILED_EXECUTION_STATUSES = frozenset({"FAILED"})
_METRIC_KEYS = frozenset({"ndcg_full", "recall_at_1", "recall_at_10", "recall_at_50", "mrr"})


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
    include_imported: bool = False,
) -> list[dict[str, Any]]:
    """Scan <artifacts>/composition/summary/ and return validated summaries.

    Ordering: measured runs, then failed runs (visible, never ranked as if they produced numbers),
    then smoke-limited runs, then synthetic fixtures, with run_id breaking ties. A fixture can
    never sort above measured data on a byte-order accident, a failed run can never outrank a run
    that measured something, and a query-limited run can never outrank a full measurement it does
    not measure. Rows imported from an external artifact are opt-in (``include_imported``) and sort
    in their own tier: they are real data, but not evidence about a model this workbench ran.
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
                if not include_imported and is_imported_summary(summary):
                    continue
                results.append(summary)
    for summary in results:
        decorate_summary(artifacts_root, summary)
    annotate_duplicate_attempts(results, artifacts_root)

    def rank(summary: Mapping[str, Any]) -> tuple[int, str, str]:
        if is_imported_summary(summary):
            tier = 4
        elif is_fixture_summary(summary):
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


# ---------------------------------------------------------------------------------------------
# Capabilities, per-run statistics and measurement identity.
#
# One boolean ("this run is broken") cannot describe the four independent facts a CIRCUS run can
# or cannot carry: aggregated numbers, per-query exact target ranks, candidate Top-K windows, and
# locally resolvable images. A run with ranks but no candidates is *not* broken, and a run whose
# images are missing is not a run that scored nothing. Each fact is computed from the artifact that
# proves it, and each absence carries a named status instead of a bare false.
# ---------------------------------------------------------------------------------------------

#: Per-query failure markers, mapped to the pipeline stage that produced them. An unscored query
#: is a MAPPING/pipeline fact (an ambiguous garment category, a missing reference image); it is a
#: different thing from a query that was scored and simply retrieved badly, and the API must never
#: merge them into one "failure" bucket.
_UNSCORED_KIND_MARKERS = (
    ("category_unresolved", ("category unresolved", "category none has no adapter",
                             "category '", "category \"")),
    ("reference_image_missing", ("query reference image is missing",)),
    ("positive_target_unresolved", ("no positive target resolved",)),
    ("features_unavailable", ("query features unavailable",)),
    ("exact_rank_unavailable", ("invalid full-gallery exact rank", "exact rank unavailable")),
)

def classify_unscored(failure: str | None) -> str | None:
    """Name the pipeline stage behind a missing rank, or ``None`` when there was no failure.

    ``None`` means "this query has a real rank"; it is never a synonym for "unknown". A failure
    text that matches no declared marker returns ``"other"`` so an unrecognised cause is still
    visible and countable rather than silently falling into "unscored for no reason".
    """
    if not failure:
        return None
    lowered = failure.lower()
    for kind, markers in _UNSCORED_KIND_MARKERS:
        if any(marker in lowered for marker in markers):
            return kind
    return "other"

_RANKS_CACHE: dict[tuple[str, float, int], dict[str, Any] | None] = {}

def load_run_ranks(artifacts_root: Path, run_id: str) -> dict[str, Any] | None:
    """Load this run's per-query rank artifact (``runs/<run_id>/ranks/*.json``), or ``None``.

    Cached on (path, mtime, size). ``None`` means no usable rank artifact exists for the run, which
    ``per_query_target_ranks`` reports as ``CAPABILITY_ABSENT`` rather than an empty result.
    """
    try:
        run_ranks_dir = resolve_composition_path(artifacts_root, f"runs/{run_id}/ranks")
    except ValueError:
        return None
    if not run_ranks_dir.is_dir():
        return None
    for candidate in sorted(run_ranks_dir.glob("*.json")):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        key = (str(candidate), stat.st_mtime, stat.st_size)
        if key in _RANKS_CACHE:
            cached = _RANKS_CACHE[key]
            if cached is not None:
                return cached
            continue
        try:
            content = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            _RANKS_CACHE[key] = None
            continue
        if not (isinstance(content, dict) and isinstance(content.get("ranks"), dict)):
            _RANKS_CACHE[key] = None
            continue
        _RANKS_CACHE[key] = content
        return content
    return None

_IMAGE_AVAILABILITY_CACHE: dict[tuple, dict[str, Any]] = {}

def _sample_identities(values: list[str], size: int) -> list[str]:
    """Deterministic, evenly spread sample (never the first N, which cluster by category)."""
    if len(values) <= size:
        return list(values)
    step = len(values) / float(size)
    return [values[min(int(index * step), len(values) - 1)] for index in range(size)]

def image_availability(artifacts_root: Path, summary: Mapping[str, Any],
                       sample_size: int = 40) -> dict[str, Any]:
    """Whether the images this run refers to can be opened locally, sampled deterministically.

    A whole-gallery stat() of 74k files per request is not acceptable latency, so this answers from
    a bounded, evenly spread sample of released identities through the run's own declared layout.
    ``status`` is ``local`` when every sampled identity resolved, ``partial`` when some did not, and
    ``unavailable`` when none did; ``unknown`` when no dataset root is configured. The sample size
    is reported so a consumer never reads it as an exhaustive count.
    """
    config = resolve_config()
    dataset_root = Path(config.FASHIONIQ_ROOT)
    layout = str(summary.get("fashioniq_layout") or "fashioniq_standard")
    key = (str(dataset_root), layout, str(summary.get("protocol_id")))
    cached = _IMAGE_AVAILABILITY_CACHE.get(key)
    if cached is not None:
        return cached
    if not dataset_root.is_dir():
        result = {"status": "unknown", "sampled": 0, "resolved": 0,
                  "reason": f"dataset root is absent: {dataset_root}"}
    else:
        from workbench.composition.runner import image_resolver_factory

        try:
            resolve = image_resolver_factory(dataset_root, layout)
        except Exception as error:  # unknown layout name is a configuration fact, not a crash
            return {"status": "unknown", "sampled": 0, "resolved": 0,
                    "reason": f"{type(error).__name__}: {error}"}
        try:
            release = load_cached_release()
        except Exception:
            release = None
        identities = list(getattr(getattr(release, "gallery", None), "image_ids", ()) or ())
        names = getattr(release, "candidate_names", None) or {}
        sample = _sample_identities(identities, sample_size)
        resolved = sum(1 for identity in sample if resolve(identity, names) is not None)
        if not sample:
            result = {"status": "unknown", "sampled": 0, "resolved": 0,
                      "reason": "no released gallery identities to sample"}
        else:
            status = ("local" if resolved == len(sample)
                      else "unavailable" if resolved == 0 else "partial")
            result = {"status": status, "sampled": len(sample), "resolved": resolved,
                      "layout": layout, "reason": None}
    _IMAGE_AVAILABILITY_CACHE[key] = result
    return result

def subset_qids(subset: str) -> set[str] | None:
    """The qids that belong to `subset`, or None when the release is not loaded.

    Counters that describe a scoped view must be scoped by the SAME predicate the view uses. The
    API used to publish `sample_count` for the requested subset beside `scored_count` computed over
    all 6003 queries, so `subset=sf` rendered `5,884 / 4,069 (144.6%)`.
    """
    release = load_cached_release()
    if release is None:
        return None
    return {query.qid for query in release.queries if subset in query.subsets}

def _ranks_statistics(ranks_data: Mapping[str, Any] | None,
                      query_count: int | None,
                      qids: set[str] | None = None) -> dict[str, Any] | None:
    """Per-condition scored/unscored counts over `qids` (the requested view), from the artifact.

    The invariant this guarantees: for every condition,
    ``scored_count + unscored_count == denominator`` and ``coverage <= 1.0``. `qids=None` means the
    whole artifact, which is only used for the explicitly named `whole_run` block.
    """
    if not ranks_data:
        return None
    ranks = ranks_data.get("ranks") or {}
    failures = ranks_data.get("query_failures") or {}
    conditions: dict[str, Any] = {}
    for condition, table in ranks.items():
        if not isinstance(table, Mapping):
            continue
        scoped = table if qids is None else {qid: rank for qid, rank in table.items() if qid in qids}
        scored = [qid for qid, rank in scoped.items() if rank is not None]
        unscored = [qid for qid, rank in scoped.items() if rank is None]
        by_kind: dict[str, int] = {}
        for qid in unscored:
            kind = classify_unscored((failures.get(condition) or {}).get(qid)) or "other"
            by_kind[kind] = by_kind.get(kind, 0) + 1
        denominator = len(scoped)
        conditions[condition] = {
            "denominator": denominator,
            "scored_count": len(scored),
            "unscored_count": len(unscored),
            "unscored_by_kind": dict(sorted(by_kind.items())),
            # Coverage is over the queries this view was asked to score, so an ambiguous category
            # lowers it instead of vanishing from the denominator.
            "coverage": (len(scored) / denominator) if denominator else 1.0,
        }
    return {"conditions": conditions,
            "query_count": query_count if query_count is not None else
            max((item["denominator"] for item in conditions.values()), default=0)}

_TRANSFER_ENV_EVALUATION_DATASET = "FashionIQ"

def transfer_facts(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Normalise the transfer setting into one boolean plus the dataset pair it names.

    ``checkpoint_transfer`` is written as a label string ("CIRR→FashionIQ transfer") by the runner
    and as a boolean by fixtures; a consumer gating a scientific warning needs the boolean, so it is
    derived here rather than re-derived (and possibly re-derived differently) in the UI.
    """
    raw = summary.get("checkpoint_transfer")
    training = summary.get("checkpoint_training_dataset")
    if isinstance(raw, str) and raw.strip():
        is_transfer = True
        evaluation = (_TRANSFER_ENV_EVALUATION_DATASET
                      if _TRANSFER_ENV_EVALUATION_DATASET.lower() in raw.lower() else None)
        label = raw
    else:
        is_transfer = bool(raw)
        evaluation = _TRANSFER_ENV_EVALUATION_DATASET if is_transfer else None
        label = (f"{training}→{evaluation} transfer" if is_transfer and training and evaluation
                 else None)
    return {"is_transfer": is_transfer,
            "training_dataset": training,
            "evaluation_dataset": evaluation,
            "label": label}

#: Inputs that make two runs the same measurement. A run missing the tensor environment digest or
#: the checkpoint content identity cannot be proven distinct from another attempt, so it is
#: reported as ``undetermined`` rather than either "unique" or "duplicate".
#: How each adapter's `score` combines query and target features, read from the model's own
#: registry-declared adapter module. Recorded per run so a cohort table cannot imply comparability
#: between a numpy `einsum(..., optimize=True)` path and a plain `einsum`/matmul path when the
#: implementations genuinely differ. A module that cannot be read yields `unknown`, never a guess.
_SCORING_IMPL_CACHE: dict[tuple[str, str], dict[str, Any]] = {}

def scoring_implementation(model_id: str | None, scoring_hash: str | None,
                           adapter_hash: str | None, *,
                           summary: Mapping[str, Any] | None = None,
                           live: bool = True) -> dict[str, Any]:
    """The scoring implementation behind a run, from that run's own immutable evidence.

    `optimize=True` selects numpy's contraction-order optimiser, which picks a different
    intermediate and therefore a different floating-point summation order than the plain
    `einsum`/matmul path. Two runs scored with different implementations are not bit-comparable,
    so this is a property of the measurement, not a display detail.

    Precedence, strongest first:

    1. ``summary["scoring_impl"]`` -- a value the artifact itself recorded. Always wins: it is what
       the run actually did.
    2. ``live=True`` -- the run is being measured NOW, so deriving from the registered adapter
       module's current ``score`` body describes it correctly.
    3. Otherwise ``unknown`` with the reason. Reading today's source and presenting it as a
       historical run's implementation is the exact defect this signature exists to remove: a
       refactor, a rename or an adapter swap today would silently relabel an old measurement.

    A versioned mapping from scoring hash to implementation would be option 1.5; it does not exist
    in this repository, so a historical run without a recorded value is honestly unknown rather
    than approximately guessed.
    """
    recorded = None
    if isinstance(summary, Mapping):
        recorded = summary.get("scoring_impl")
        if isinstance(recorded, Mapping):
            recorded = recorded.get("impl")
        if isinstance(recorded, str) and recorded and recorded != "unknown":
            return {"impl": recorded,
                    "basis": f"recorded by the artifact itself (scoring_impl={recorded})",
                    "adapter_hash_known": adapter_hash is not None,
                    "scoring_hash_known": scoring_hash is not None,
                    "source": "artifact"}
    if live:
        resolved = _scoring_impl_for(model_id, adapter_hash)
        return {"impl": resolved["impl"], "basis": resolved["basis"],
                "adapter_hash_known": adapter_hash is not None,
                "scoring_hash_known": scoring_hash is not None,
                "source": "live"}
    # Historical and not recorded. `adapter_hash` being present does NOT identify the
    # implementation: it says which bytes were hashed, not how they contract, and no table maps
    # this repository's adapter hashes to implementations.
    return {"impl": "unknown",
            "basis": ("historical run records no scoring_impl and no hash-to-implementation table "
                      "exists, so today's adapter source cannot stand in for it"),
            "adapter_hash_known": adapter_hash is not None,
            "scoring_hash_known": scoring_hash is not None,
            "source": "unavailable"}

def _scoring_impl_for(model_id: str | None, adapter_hash: str | None) -> dict[str, Any]:
    from workbench.composition.adapters import capability

    if not model_id:
        return {"impl": "unknown", "basis": "no model id recorded for this row"}
    module_name = capability(model_id)
    if not module_name:
        return {"impl": "unknown", "basis": f"{model_id} declares no composition adapter module"}
    key = (model_id, module_name)
    cached = _SCORING_IMPL_CACHE.get(key)
    if cached is not None:
        return cached
    import inspect as _inspect
    import sys as _sys

    try:
        module = _sys.modules.get(module_name) or __import__(module_name, fromlist=["Adapter"])
        adapter_type = getattr(module, "Adapter", None)
        source = _inspect.getsource(adapter_type.score) if adapter_type is not None else ""
    except Exception as error:
        result = {"impl": "unknown", "basis": f"{type(error).__name__}: {error}"}
        _SCORING_IMPL_CACHE[key] = result
        return result
    if "optimize=True" in source:
        result = {"impl": "einsum-optimized", "basis": f"{module_name}.Adapter.score uses optimize=True"}
    elif "einsum" in source:
        result = {"impl": "einsum", "basis": f"{module_name}.Adapter.score uses plain einsum"}
    elif "@" in source or "linalg" in source or "dot" in source:
        result = {"impl": "matmul", "basis": f"{module_name}.Adapter.score uses matmul"}
    else:
        result = {"impl": "unknown", "basis": f"{module_name}.Adapter.score has no recognised "
                                              "contraction"}
    _SCORING_IMPL_CACHE[key] = result
    return result

def _recorded_source_matches_disk(summary: Mapping[str, Any]) -> bool:
    """Whether the artifact's recorded hashes still equal the modules' hashes on disk.

    TWO hashes, TWO modules, and both must match. `scoring_module_source_hash` hashes the SHARED
    `workbench.composition.adapters` module; `adapter_module_source_hash` hashes the model's own
    `model_<id>.py`, which is where a `score` override lives. Comparing the shared hash against the
    adapter module compared two different files, so it was False for every run ever recorded and
    `scoring_impl` read `unknown` on a run whose source was byte-identical to today's.

    True means the source that produced the run and the source readable now are the same bytes, so
    derivation from that source is evidence about the run. False (or unreadable, or a hash the runner
    recorded as `unavailable`) means the row is historical and must not be labelled from today's code.
    """
    recorded_scoring = summary.get("scoring_module_source_hash")
    recorded_adapter = summary.get("adapter_module_source_hash")
    if not recorded_scoring or not recorded_adapter:
        return False
    if "unavailable" in (recorded_scoring, recorded_adapter):
        return False
    model_id = summary.get("model_id")
    if not model_id:
        return False
    from workbench.composition import adapters as adapters_module
    from workbench.composition.runner import _module_source_hash

    module_name = adapters_module.capability(model_id)
    if not module_name:
        return False
    import sys as _sys

    module = _sys.modules.get(module_name)
    if module is None:
        try:
            module = __import__(module_name, fromlist=["Adapter"])
        except Exception:
            return False
    try:
        return (_module_source_hash(adapters_module) == recorded_scoring
                and _module_source_hash(module) == recorded_adapter)
    except Exception:
        return False

def measurement_identity(summary: Mapping[str, Any],
                         identity_source: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The defensible identity of WHAT was measured (not which file it landed in).

    `identity_source` is the run's recorded identity block (`runs/<run_id>/ranks/*.json:identity`),
    which is where `scoring_module_source_hash` and `adapter_module_source_hash` actually live; the
    summary does not carry them. Passing it is what lets a scoring-only change read as a distinct
    measurement instead of a duplicate. A summary value still wins when both exist.
    """
    recorded = identity_source if isinstance(identity_source, Mapping) else {}
    runtime = summary.get("runtime") if isinstance(summary.get("runtime"), Mapping) else {}
    interpreter = (runtime.get("model_interpreter")
                   if isinstance(runtime.get("model_interpreter"), Mapping) else {})
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    knobs = summary.get("evaluator_environment_knobs")
    identity = {
        "model_id": summary.get("model_id"),
        "checkpoint_id": summary.get("checkpoint_id"),
        "checkpoint_identity": summary.get("checkpoint_identity"),
        "protocol_id": summary.get("protocol_id"),
        "subset": summary.get("subset"),
        "gallery_identity_sha256": summary.get("gallery_identity_sha256"),
        "query_count": summary.get("query_count"),
        "conditions": sorted(summary.get("conditions") or ()),
        "seed": summary.get("seed"),
        "preprocessing": summary.get("preprocessing"),
        "environment_fingerprint": interpreter.get("environment_fingerprint"),
        # The SCORING PATH is part of what was measured. Without it, two runs of the same model
        # that differ only in scoring implementation hashed to one `group_id` and were labelled
        # `duplicate` -- i.e. "the same thing measured twice" -- which is false. The cohort really
        # contains two implementations: conesep/ptha_mtst/csmcir score with `optimize=True` while
        # the BLIP-2 family uses a plain `einsum`.
        #
        # `scoring_module_source_hash` hashes the shared adapters module; `adapter_module_source_hash`
        # hashes the model's own module, which is where a `score` override lives. Absent values stay
        # absent (never fabricated) and are reported through `missing_identity_inputs`.
        "scoring_module_source_hash": (summary.get("scoring_module_source_hash")
                                       or recorded.get("scoring_module_source_hash")),
        "adapter_module_source_hash": (summary.get("adapter_module_source_hash")
                                       or recorded.get("adapter_module_source_hash")),
        "checkpoint_transfer": transfer_facts(summary)["label"],
        # An absent knobs record is evidence of nothing; it is included only when the evaluator
        # actually recorded one, so two unrelated aggregate runs cannot be declared duplicates
        # merely because neither carries the field.
        "evaluator_environment_knobs": knobs if isinstance(knobs, Mapping) and knobs else None,
        "measurement_denominator": (metrics.get("gap_cohort_count")
                                    or summary.get("query_count")),
        # ---- identity v2 ----------------------------------------------------------------
        # These three are ARITHMETIC inputs (see runner.py): the BLAS thread count changes the
        # floating-point summation order, the resolved per-query scope decides which gallery matrix
        # a query is scored against, and the query order identity binds membership and order. The
        # independent review measured that changing any of them still produced
        # `same_group: true, confidence: full`, i.e. two different measurements were reported as
        # the same one. They are part of the payload and of the essential set below.
        "blas_threads_pinned": _recorded_or_summary(summary, recorded, "blas_threads_pinned"),
        "query_scope_resolution": _recorded_or_summary(summary, recorded, "query_scope_resolution"),
        "query_order_identity": _recorded_or_summary(summary, recorded, "query_order_identity"),
        # The policy that produced the features this run scored. A `header`-verified cache is a
        # weaker guarantee than a `full` one; an artifact must not imply the stronger claim. Read
        # through `_recorded_or_summary` like its siblings so a recorded v2 artifact keeps its own
        # value even when the served summary carries none.
        "cache_verify_mode": _recorded_or_summary(summary, recorded, "cache_verify_mode"),
    }
    # `scoring_module_source_hash` is essential for the same reason: without it an older artifact
    # cannot be proven different from a run whose scoring implementation changed, so the honest
    # answer is `partial`, not `unique` and certainly not `duplicate`.
    #
    # The v2 inputs are essential for the same reason: a run that does not record its BLAS policy,
    # scope resolution and query-order identity cannot be proven equal to OR different from another
    # run, so its confidence is `partial` by definition. An artifact written before v2 has none of
    # them, which is exactly why it must not be handed a v2 group id -- see `_identity_schema`.
    # The v2 inputs are NOT in this tuple. Absence downgrades nothing here, because a served
    # summary legitimately carries only metadata -- for it the v2 fields are simply not part of the
    # record. What they must do is SPLIT groups when they ARE present, which they do by being in
    # the payload above. A recorded v1 artifact is handled separately below: it is labelled
    # `unknown (predates identity v2)` rather than being handed a group id computed from fields it
    # never had.
    essential = ("checkpoint_identity", "gallery_identity_sha256", "environment_fingerprint",
                 "scoring_module_source_hash")
    # A v1 ARTIFACT is a recorded identity block that predates v2 -- that is the case the review
    # measured (an artifact on disk with no v2 inputs). A summary that merely lacks the fields is
    # not a v1 artifact: it is a row assembled from served metadata, and the caller's own summary is
    # the whole record of it. Distinguishing the two is what stops every served row from being
    # relabelled "predates identity v2".
    schema = _identity_schema(identity, recorded=recorded)
    if schema == 1:
        # A v1 artifact has no v2 inputs at all. It is readable and explorable, but it can never be
        # compared to a v2 group: computing a v2 group id from absent fields would silently assert
        # that they matched. Label it and stop.
        identity["identity_schema_version"] = "unknown (predates identity v2)"
        return {"identity": identity,
                "group_id": None,
                "schema_version": "unknown",
                "missing_identity_inputs": [name for name in essential if not identity.get(name)],
                "confidence": "unknown",
                "reason": "recorded identity predates identity v2: it has no blas_threads_pinned / "
                          "query_scope_resolution / query_order_identity, so it cannot be proven the "
                          "same as, or different from, any v2 measurement"}
    identity["identity_schema_version"] = 2
    missing = [name for name in essential if not identity.get(name)]
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str)
    return {"identity": identity,
            "group_id": hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
            "schema_version": 2,
            "missing_identity_inputs": missing,
            "confidence": "partial" if missing else "full",
            "reason": None if not missing else
                      f"identity v2 inputs missing: {missing}"}

#: The identity fields added by schema v2. All three are arithmetic inputs, not metadata.
IDENTITY_V2_INPUTS = ("blas_threads_pinned", "query_scope_resolution", "query_order_identity")

def _recorded_or_summary(summary: Mapping[str, Any], recorded: Mapping[str, Any],
                         name: str) -> Any:
    """The value recorded with the run, falling back to the summary, never fabricated."""
    value = recorded.get(name)
    if value is None:
        value = summary.get(name)
    if isinstance(value, Mapping):
        # `blas_pin` nested records are provenance, not the arithmetic value; hash only what
        # actually changes the arithmetic.
        value = value.get("requested", value.get("effective"))
    return value

#: Fields a real v2 run identity always carries, because the runner writes them unconditionally.
#: Their presence is what proves a recorded block came from a v2-era runner, so their absence in a
#: block that DOES contain other runner-only keys is proof the block predates v2.
V2_MARKERS = ("blas_threads_pinned", "query_scope_resolution", "query_order_identity")

def _identity_schema(identity: Mapping[str, Any],
                     recorded: Mapping[str, Any] | None = None) -> int:
    """1 when a RECORDED run identity predates v2, else 2.

    Only a recorded block from a runner can be v1, and only when it is recognisably a runner block
    (it carries runner-only keys) that lacks every v2 marker. A partial `recorded` mapping that a
    caller assembled by hand -- e.g. just the two scoring hashes -- is not evidence of an old
    artifact, so it stays v2 and is scored on its actual inputs.
    """
    if not isinstance(recorded, Mapping) or not recorded:
        return 2
    if any(identity.get(name) is not None for name in V2_MARKERS):
        return 2
    runner_keys = {"model_id", "checkpoint_id", "gallery_identity", "query_order_identity",
                   "adapter_module_source_hash", "scoring_module_source_hash", "subset_memberships"}
    if len(runner_keys & set(recorded)) < 3:
        return 2
    return 1

def annotate_duplicate_attempts(summaries: list[dict[str, Any]],
                                artifacts_root: Path | None = None) -> None:
    """Group the served rows by measurement identity and label each row in place.

    ``duplicate_status`` is ``unique`` (one attempt of this measurement), ``duplicate`` (more than
    one artifact measured the same thing), or ``undetermined`` when the recorded identity lacks an
    input that would prove any two rows the same or different. Duplicate detection never relies on
    a missing ``evaluator_environment_knobs`` field.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        if is_fixture_summary(summary):
            summary["measurement_identity"] = None
            summary["measurement_group_id"] = None
            summary["duplicate_attempt_count"] = 0
            summary["duplicate_status"] = "synthetic_fixture"
            continue
        recorded = (summary.get("_recorded_identity")
                    or ((load_run_ranks(artifacts_root, str(summary.get("run_id") or "")) or {})
                        .get("identity") if artifacts_root is not None else None))
        facts = measurement_identity(summary, recorded)
        summary["measurement_identity"] = facts["identity"]
        summary["measurement_group_id"] = facts["group_id"]
        summary["measurement_identity_confidence"] = facts["confidence"]
        summary["measurement_identity_missing_inputs"] = facts["missing_identity_inputs"]
        summary["measurement_identity_schema_version"] = facts["identity"].get("identity_schema_version")
        if facts["group_id"] is None:
            # A pre-v2 artifact has no group: it is readable and explorable, but it must never be
            # grouped with a v2 measurement (or with another pre-v2 artifact, since its arithmetic
            # policy is unknown). `unknown (predates identity v2)` is the honest label; the row
            # keeps its own identity block so nothing about it is lost.
            summary["duplicate_attempt_count"] = 1
            summary["duplicate_status"] = "unknown (predates identity v2)"
            continue
        groups.setdefault(facts["group_id"], []).append(summary)
    for members in groups.values():
        count = len(members)
        confidence = ("partial" if any(member.get("measurement_identity_confidence") == "partial"
                                       for member in members) else "full")
        status = ("undetermined" if confidence == "partial"
                  else "duplicate" if count > 1 else "unique")
        for member in members:
            member["duplicate_attempt_count"] = count
            member["duplicate_status"] = status

def unique_measurement_count(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """How many distinct measurements the served rows describe, and how sure that is."""
    real = [item for item in summaries
            if not is_fixture_summary(item) and not is_smoke_summary(item)]
    groups = {item.get("measurement_group_id") for item in real}
    partial = [item for item in real if item.get("measurement_identity_confidence") == "partial"]
    return {"artifact_count": len(summaries),
            "measured_artifact_count": len(real),
            "unique_measurement_count": len(groups),
            "undetermined_artifact_count": len(partial),
            "confidence": "partial" if partial else "full"}

def _candidate_capability(artifacts_root: Path, run_id: str) -> dict[str, Any]:
    """Candidate Top-K availability, read from the export manifest (never inferred)."""
    from workbench.composition.topk import export_index

    try:
        index = export_index(artifacts_root, run_id)
    except Exception as error:
        return {"available": False, "status": "TOPK_EXPORT_UNREADABLE",
                "reason": f"{type(error).__name__}: {error}", "depth": None,
                "conditions": [], "source": None, "path": None}
    if not index:
        return {"available": False, "status": "TOPK_EXPORT_NOT_REQUESTED",
                "reason": "this run was produced without candidate Top-K export "
                          "(enable it with --topk-depth on composition_eval)",
                "depth": None, "conditions": [], "source": None, "path": None}
    depths = sorted({int(item.get("depth")) for item in index.values() if item.get("depth")})
    partial = sorted(name for name, item in index.items() if item.get("partial"))
    return {"available": True,
            "status": "AVAILABLE_PARTIAL" if partial else "AVAILABLE",
            "reason": (f"partial export for conditions {partial}" if partial else None),
            "depth": depths[-1] if depths else None,
            "conditions": sorted(index),
            "source": sorted({str(item.get("source") or "model_run") for item in index.values()}),
            "gallery_count": next(iter(index.values())).get("gallery_count"),
            "tie_policy": next(iter(index.values())).get("tie_policy"),
            "path": next(iter(index.values())).get("path")}

def composition_capabilities(artifacts_root: Path, summary: Mapping[str, Any],
                            ranks_data: Mapping[str, Any] | None = None,
                            sample_size: int = 40) -> tuple[dict[str, bool], dict[str, Any]]:
    """The four independent capabilities of one run, as booleans plus the evidence behind them."""
    run_id = str(summary.get("run_id") or "")
    ranks_data = ranks_data if ranks_data is not None else load_run_ranks(artifacts_root, run_id)
    has_numbers = _has_measured_numbers(summary.get("metrics"))
    rank_count = 0
    rank_conditions: list[str] = []
    if ranks_data:
        for condition, table in (ranks_data.get("ranks") or {}).items():
            if isinstance(table, Mapping) and table:
                rank_conditions.append(condition)
                rank_count = max(rank_count, len(table))
    candidates = _candidate_capability(artifacts_root, run_id)
    images = image_availability(artifacts_root, summary, sample_size=sample_size)
    detail = {
        "aggregate_metrics": {
            "available": bool(has_numbers),
            "status": "AVAILABLE" if has_numbers else (
                "NO_NUMBERS_MEASURED" if not is_fixture_summary(summary) else "SYNTHETIC_FIXTURE"),
            "reason": None if has_numbers else "the summary carries no non-null score",
        },
        "per_query_target_ranks": {
            "available": bool(ranks_data),
            "status": "AVAILABLE" if ranks_data else "CAPABILITY_ABSENT",
            "reason": None if ranks_data else
            "no per-query rank artifact exists for this run (aggregate-only or failed)",
            "query_count": rank_count or None,
            "conditions": sorted(rank_conditions),
        },
        "candidate_topk": candidates,
        "image_availability": images,
    }
    flags = {
        "aggregate_metrics": detail["aggregate_metrics"]["available"],
        "per_query_target_ranks": detail["per_query_target_ranks"]["available"],
        "candidate_topk": candidates["available"],
        "image_availability": images["status"] in {"local", "partial"},
    }
    return flags, detail

def decorate_summary(artifacts_root: Path, summary: dict[str, Any],
                     ranks_data: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Attach capabilities, capabilities detail, statistics and transfer facts to a summary.

    Applied on read (and cached with the parsed summary) so a list row and a detail row can never
    disagree about what a run can do.
    """
    flags, detail = composition_capabilities(artifacts_root, summary, ranks_data=ranks_data)
    ranks_data = ranks_data if ranks_data is not None else load_run_ranks(
        artifacts_root, str(summary.get("run_id") or ""))
    # The rank artifact's identity block is where the scoring path lives; cached on the summary so
    # grouping (which runs after decoration) never re-reads it.
    recorded_identity = (ranks_data or {}).get("identity") if isinstance(ranks_data, Mapping) else None
    summary["_recorded_identity"] = dict(recorded_identity) if isinstance(recorded_identity, Mapping) else None
    summary["scoring_module_source_hash"] = (recorded_identity or {}).get("scoring_module_source_hash") \
        if isinstance(recorded_identity, Mapping) else None
    summary["adapter_module_source_hash"] = (recorded_identity or {}).get("adapter_module_source_hash") \
        if isinstance(recorded_identity, Mapping) else None
    # The single field a consumer cannot miss: is this row a run of a MODEL at all? An imported
    # released-audit row is real data but needs no model, so ranking it beside HABIT's 35.62 makes
    # it read as the best model in the table. `is_model_run` is the explicit answer, and
    # `provenance_label` is the sentence to render.
    summary["is_model_run"] = not is_imported_summary(summary)
    summary["provenance_label"] = (
        "released CIRCUS audit baseline (imported; not a model run of this workbench)"
        if is_imported_summary(summary) else None)
    # Derive from the adapter on disk ONLY when the artifact's recorded scoring hash still equals
    # today's hash for that same module: then "the source that produced it" and "the source I can
    # read" are provably the same bytes, so the derivation is evidence rather than an assumption.
    # Otherwise the row is historical and its implementation is unknown -- the recorded value in
    # `summary["scoring_impl"]` (handled first inside the function) is the only acceptable source.
    summary["scoring_impl"] = scoring_implementation(
        summary.get("model_id"), summary["scoring_module_source_hash"],
        summary["adapter_module_source_hash"],
        summary=summary,
        live=_recorded_source_matches_disk(summary))
    subset = str(summary.get("subset") or "full")
    qids = subset_qids(subset)
    # Scoped to the SAME subset `query_count` describes, so scored + unscored == sample_count and
    # coverage can never exceed 1.0. The whole-artifact view is published under an explicit
    # `whole_run` name rather than being mistaken for a scoped figure.
    statistics = _ranks_statistics(ranks_data, summary.get("query_count"), qids)
    whole_run = _ranks_statistics(ranks_data, None, None)
    summary["capabilities"] = flags
    summary["capability_detail"] = detail
    summary["statistics"] = statistics
    summary["transfer"] = transfer_facts(summary)
    summary["checkpoint_identity"] = summary.get("checkpoint_identity")
    metrics = summary.get("metrics") if isinstance(summary.get("metrics"), Mapping) else {}
    denominator = metrics.get("gap_cohort_count") or summary.get("query_count")
    unscored: dict[str, int] = {}
    if statistics:
        for condition, values in statistics["conditions"].items():
            for kind, count in (values.get("unscored_by_kind") or {}).items():
                unscored[kind] = max(unscored.get(kind, 0), count)
    # `unscored` (pipeline/mapping, no rank exists) and `retrieval_failure` (a real rank worse than
    # K) are separate counters over separate facts; the UI filters them apart.
    # `None` means "no rank artifact, so this cannot be computed"; `0` means "computed, none
    # unscored". Conflating them hid a clean run behind the same null an absent artifact produces.
    summary["measurement"] = {
        # Every counter here describes `subset`: scored + unscored == sample_count, coverage <= 1.
        "subset": subset,
        "sample_count": summary.get("query_count"),
        "denominator": denominator,
        "coverage": metrics.get("coverage"),
        "scored_count": (statistics["conditions"].get("multimodal", {}).get("scored_count")
                         if statistics else None),
        "unscored_query_count": sum(unscored.values()) if statistics else None,
        "unscored_by_kind": dict(sorted(unscored.items())) if statistics else None,
        "image_availability_sample_size": detail["image_availability"].get("sampled"),
        # Explicitly NOT scoped to `subset`; here so a consumer that wants it cannot confuse it
        # with the scoped figures above.
        "whole_run": ({"query_count": whole_run["query_count"],
                       "scored_count": whole_run["conditions"].get("multimodal", {})
                       .get("scored_count"),
                       "unscored_query_count": sum(
                           (values.get("unscored_count") or 0)
                           for values in whole_run["conditions"].values())}
                      if whole_run else None),
    }
    summary["statistics_whole_run"] = whole_run
    # A metric the scorer gained AFTER these runs finished is served from a derived record only when
    # the record still describes the rank artifact on disk (see `derived_metric`). The published
    # `full.json` is never rewritten: this block is the other place a number can come from, and it
    # says so, per metric, with `source` and the refusal reason when it cannot be served.
    summary["derived_metrics"] = {}
    metrics_block = summary.get("metrics")
    if isinstance(metrics_block, dict):
        attempt = (ranks_data or {}).get("attempt") if isinstance(ranks_data, Mapping) else None
        for metric in DERIVED_METRICS:
            record = derived_metric(artifacts_root, metric, str(summary.get("run_id") or ""),
                                    attempt or summary.get("attempt") or None)
            block = {"status": record["status"], "source": "derived_record",
                     "derived_from": ((record.get("record") or {}).get("derived_from")),
                     "reason": record["reason"]}
            if record["status"] == "VERIFIED":
                values = (record.get("record") or {}).get("conditions") or {}
                # The derived record keys by CONDITION NAME (`image_only_empty_text`); the summary
                # buckets by ROLE (`multimodal`/`text_only`/`image_only`) and each bucket names the
                # condition that filled it. Mapping through that name is what keeps a variant run's
                # value out of the protocol condition's bucket.
                bucket_for: dict[str, str] = {}
                for role in ("multimodal", "text_only", "image_only"):
                    bucket = metrics_block.get(role)
                    if isinstance(bucket, Mapping) and isinstance(bucket.get("condition"), str):
                        bucket_for[bucket["condition"]] = role
                for condition, values_for_condition in values.items():
                    if not isinstance(values_for_condition, Mapping):
                        continue
                    role = bucket_for.get(condition, condition)
                    bucket = metrics_block.get(role)
                    if not isinstance(bucket, dict) or "recall_at_1" in bucket:
                        continue
                    bucket["recall_at_1"] = values_for_condition.get("recall_at_1")
                    if values_for_condition.get("unavailable_reason"):
                        bucket["recall_at_1_unavailable_reason"] = \
                            values_for_condition["unavailable_reason"]
                multimodal = metrics_block.get("multimodal")
                if isinstance(multimodal, dict) and "recall_at_1_multimodal" not in metrics_block:
                    metrics_block["recall_at_1_multimodal"] = multimodal.get("recall_at_1")
                block["values"] = {condition: (values_for_condition or {}).get("recall_at_1")
                                   for condition, values_for_condition in values.items()}
            summary["derived_metrics"][metric] = block
    # The gap's OWN condition triple, when the run records one. Without this a variant gap (computed
    # over an adapter's named ablation because the protocol image condition carries no ranks) is
    # indistinguishable on screen from the protocol gap.
    gap_conditions = metrics_block.get("gap_conditions") if isinstance(metrics_block, Mapping) else None
    if isinstance(gap_conditions, (list, tuple)) and gap_conditions:
        summary["gap_conditions"] = [str(name) for name in gap_conditions]
        summary["gap_condition_labels"] = [condition_display_label(str(name))
                                           for name in gap_conditions]
        summary["gap_role_reason"] = (metrics_block.get("gap_role_reason")
                                      if isinstance(metrics_block, Mapping) else None)
    # What the UI may OFFER for this run: the conditions it has a rank table for. A condition with no
    # table is not browsable -- offering it produced a third option whose every row was "Chưa chấm"
    # and hid the condition the run did score. Falls back to `displayed_conditions` when no rank
    # artifact is available (a failed or aggregate-only run), because then nothing can be filtered
    # and the protocol triple is the only honest list.
    tables = (ranks_data or {}).get("ranks") if isinstance(ranks_data, Mapping) else None
    tables = tables if isinstance(tables, Mapping) else {}

    def has_table(name: str) -> bool:
        table = tables.get(name)
        return isinstance(table, Mapping) and bool(table)

    # Protocol order first, then any extra condition the run scored, in declaration order -- the same
    # order `displayed_conditions` uses, so a protocol-triple run is unchanged.
    ranked = tuple([name for name in CONDITIONS if has_table(name)]
                   + [name for name in tables if name not in CONDITIONS and has_table(name)])
    offered = ranked or displayed_conditions(summary)
    summary["conditions_display"] = [
        {"id": name, "label": condition_display_label(name)}
        for name in offered]
    return summary

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
                decorate_summary(artifacts_root, summary)
                annotate_duplicate_attempts([summary], artifacts_root)
                return summary
    return None


def release_protocol_ids() -> frozenset[str]:
    """Protocol ids a locally loaded CIRCUS release declares; empty when none is loaded.

    An annotation identity form is only admissible under a protocol whose data actually uses it,
    so this is read from the release rather than matched against a protocol-name literal.
    """
    release = load_cached_release()
    if release is None:
        return frozenset()
    protocol = getattr(getattr(release, "gallery", None), "protocol_id", None)
    return frozenset({protocol}) if protocol else frozenset()

def release_query_ids() -> frozenset[str] | None:
    """Every qid in the loaded release, or ``None`` when no release is available.

    ``None`` (cannot verify) is distinct from an empty set (a release with no queries): the caller
    must not report "unknown query" when it simply could not look.
    """
    release = load_cached_release()
    if release is None:
        return None
    return frozenset(query.qid for query in release.queries)

#: The role order `determine_query_outcome` and `compose` both read: (multimodal, text, image).
OUTCOME_ROLES = ("multimodal", "text", "image")

def outcome_conditions(ranks_by_condition: Mapping[str, Any] | None,
                       summary: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    """The (multimodal, text, image) triple this RUN's outcome classification must read.

    Three sources, strongest first, and none of them a hard-coded ablation name:

    1. `metrics.gap_conditions` on the run's summary -- the triple the scorer itself computed the
       gap over, recorded by the runner. This is the authority when present.
    2. The protocol triple, filtered by what the rank artifact actually carries: a role whose
       protocol name has no table is filled from `ABLATION_CONDITIONS` in declared order, which is
       exactly the rule `runner.resolve_gap_conditions` applies (and re-derives for runs whose
       summary predates the field).
    3. `CONDITIONS` unchanged, when neither input says anything different.

    The result is always three DISTINCT names in role order, because that is what the classification
    requires: a triple that could not be completed falls back to the protocol names, and the
    per-row `unscored` flag then says the truth about that run.
    """
    recorded = None
    if isinstance(summary, Mapping):
        candidate = (summary.get("metrics") or {}).get("gap_conditions")
        if isinstance(candidate, (list, tuple)) and len(candidate) == 3 \
                and len(set(candidate)) == 3:
            recorded = tuple(str(name) for name in candidate)
    if recorded is not None:
        return recorded
    if not isinstance(ranks_by_condition, Mapping) or not ranks_by_condition:
        return CONDITIONS

    def scored(name: str) -> bool:
        table = ranks_by_condition.get(name)
        return isinstance(table, Mapping) and any(value is not None for value in table.values())

    triple = list(CONDITIONS)
    image_role = OUTCOME_ROLES.index("image")
    if not scored(triple[image_role]):
        substitute = next((name for name in ABLATION_CONDITIONS
                           if name in ranks_by_condition and name not in triple
                           and scored(name)), None)
        if substitute is not None:
            triple[image_role] = substitute
    return tuple(triple)

def determine_query_outcome(
    ranks: Mapping[str, int | None],
    k: int = SHORTCUT_K,
    conditions: Sequence[str] = CONDITIONS,
) -> str:
    """Classify a query's outcome at K over the run's own (multimodal, text, image) triple.

    `conditions` is that triple, in role order. It defaults to the protocol triple, which is why
    every existing run's output is unchanged -- and it must be passed for a run whose image role is
    filled by an adapter's named ablation (`metrics.gap_conditions`): reading the protocol names
    literally made EVERY query of the CLVC Full run `da0a2162be0e159a1a456cec` come back
    `outcome: "unscored"` while the same row carried `target_rank: 12796, unscored: false`.

    A missing rank in any role of the triple means the outcome cannot be classified; that is the one
    and only meaning of `"unscored"` here, and the caller must keep it consistent with the row's own
    `unscored` flag rather than re-deriving the two independently.
    """
    triple = tuple(conditions)
    if len(triple) != 3:
        raise ValueError(f"conditions must be three names in (multimodal, text, image) role order, "
                         f"got {triple!r}")
    # A role this run never measured (the key is absent from the row) can never be a hit: it is a
    # MISS, not an unclassifiable absence. Conflating the two made a run whose image role is an
    # ablation classify every row as `unscored` while the row carried a real target rank. A role that
    # IS in the triple but null for THIS qid is the one true "cannot classify" case -- and it is also
    # the case the row's own `unscored` flag reports, which is what keeps the two consistent.
    def hit(role: str) -> bool | None:
        if role not in ranks:
            return False
        value = ranks[role]
        return None if value is None else value <= k

    mm_hit, t_hit, i_hit = hit(triple[0]), hit(triple[1]), hit(triple[2])
    if mm_hit is None or t_hit is None or i_hit is None:
        return "unscored"
    if mm_hit and not t_hit and not i_hit:
        return "composition_required"
    if t_hit and not i_hit:
        return "text_only"
    if i_hit and not t_hit:
        return "image_only"
    if t_hit and i_hit:
        return "both"
    return "unresolved"


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

#: Metrics this backend can serve from a DERIVED record rather than from the published summary.
#: A derived record states how to recompute one metric from a run's own rank artifact and carries
#: the digest of the artifact it was derived from. It exists because `recall_at_1` was added to the
#: scorer after these runs finished: recomputing it from the stored ranks is exact, but the result
#: must never outlive the artifact that produced it, so the digest is checked on every read and a
#: mismatch is refused rather than served.
DERIVED_METRICS = ("recall_at_1",)
_DERIVED_FILE_CACHE: dict[str, tuple[float, int, list[dict[str, Any]]]] = {}

def _derived_metric_records(artifacts_root: Path, metric: str) -> list[dict[str, Any]]:
    """Every derived record for one metric, cached on the file's stamp."""
    try:
        directory = resolve_composition_path(artifacts_root, Path("derived") / metric)
    except ValueError:
        return []
    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            stat = path.stat()
        except OSError:
            continue
        key = str(path)
        cached = _DERIVED_FILE_CACHE.get(key)
        if cached and cached[0] == stat.st_mtime and cached[1] == stat.st_size:
            records.extend(cached[2])
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            logger.warning("Unreadable derived %s record: %s", metric, path)
            continue
        runs = payload.get("runs") if isinstance(payload, dict) else None
        usable = [record for record in (runs or []) if isinstance(record, dict)]
        _DERIVED_FILE_CACHE[key] = (stat.st_mtime, stat.st_size, usable)
        records.extend(usable)
    return records

def derived_metric(artifacts_root: Path, metric: str, run_id: str,
                   attempt: str | None) -> dict[str, Any]:
    """One derived metric record for (run, attempt), with its source verified.

    Returns `{"status": "MISSING"|"VERIFIED"|"REFUSED", "record": ..., "reason": ...}`. REFUSED is
    the important case: a record whose `derived_from.ranks.sha256` no longer matches the artifact on
    disk describes a table that no longer exists, so its numbers are not this run's numbers. Serving
    them would be a fabricated measurement; refusing is a named absence.
    """
    result: dict[str, Any] = {"status": "MISSING", "record": None, "reason": None}
    if not run_id or not attempt:
        result["reason"] = ("no attempt id is recorded for this run, so a derived record cannot be "
                           "matched to it")
        return result
    candidates = [record for record in _derived_metric_records(artifacts_root, metric)
                  if record.get("run_id") == run_id and record.get("attempt") == attempt]
    if not candidates:
        return result
    record = candidates[0]
    declared = ((record.get("derived_from") or {}).get("ranks") or {})
    path_value, expected = declared.get("path"), declared.get("sha256")
    if not path_value or not expected:
        result.update(status="REFUSED",
                      reason="the derived record does not name the rank artifact it came from")
        return result
    path = Path(str(path_value))
    if not path.is_file():
        result.update(status="REFUSED",
                      reason=f"the rank artifact the record was derived from is gone: {path}")
        return result
    if _sha256_of(path) != expected:
        result.update(status="REFUSED",
                      reason=(f"the rank artifact changed since this {metric} record was derived "
                              f"({path}), so the number no longer describes this run"))
        return result
    result.update(status="VERIFIED", record=record)
    return result

#: The four statuses a derived run reports for its PARENT's candidate export. They are the parent's
#: own facts, not this run's: a derived run never exports candidates itself.
PARENT_TOPK_STATUSES = ("NOT_REQUESTED", "NOT_EXPORTED", "PARTIAL", "COMPLETE")

def _derived_parent_candidates(artifacts_root: Path, lineage: Mapping[str, Any],
                               condition: str) -> dict[str, Any]:
    """The candidate window for a derived query, taken from the PARENT run's export.

    A derived run re-aggregates the parent's ranks and stores no candidates of its own, so reading
    the derived run_id returned ``TOPK_EXPORT_NOT_REQUESTED`` -- an empty window that reads as "the
    target is not there". The parent's export is used only when the lineage VERIFIES: the parent's
    rank artifact must hash to what the derived record claims. Anything else is
    ``parent_topk_unavailable`` with the reason, never a silently empty window.
    """
    from workbench.composition.topk import read_candidates

    parent_run_id = str(lineage.get("parent_run_id") or "")
    claimed = lineage.get("parent_rank_artifact_sha256")
    declared = dict(lineage.get("parent_topk_export") or {})

    def unavailable(reason: str) -> dict[str, Any]:
        return {"available": False, "status": "parent_topk_unavailable", "reason": reason,
                "manifest": None, "rows": {}, "path": None, "parent_run_id": parent_run_id or None}

    if not parent_run_id:
        return unavailable("the derived record names no parent run, so no lineage can be verified")
    try:
        ranks_dir = resolve_composition_path(artifacts_root, f"runs/{parent_run_id}/ranks")
    except ValueError as error:
        return unavailable(f"parent run id is not a usable path component: {error}")
    parent_rank = next((path for path in sorted(ranks_dir.glob("*.json")) if path.is_file()), None)
    if parent_rank is None:
        return unavailable(f"parent run {parent_run_id} has no rank artifact to verify against")
    if not claimed:
        return unavailable("the derived record claims no parent rank digest, so the parent "
                           "identity cannot be verified")
    actual = _sha256_of(parent_rank)
    if actual != claimed:
        return unavailable(f"parent rank artifact digest {actual} does not match the digest "
                           f"{claimed} this derived record claims")

    candidates = read_candidates(artifacts_root, parent_run_id, condition)
    if candidates.get("available"):
        manifest = candidates.get("manifest") or {}
        status = "PARTIAL" if manifest.get("partial") else "COMPLETE"
        reason = (f"the parent's export is partial: {manifest.get('missing_query_count')} declared "
                  f"queries have no candidate window" if manifest.get("partial") else None)
        return {"available": True, "status": status, "reason": reason,
                "manifest": manifest, "rows": candidates.get("rows") or {},
                "path": candidates.get("path"), "parent_run_id": parent_run_id}
    export_status = candidates.get("status")
    status = ("NOT_REQUESTED" if export_status == "TOPK_EXPORT_NOT_REQUESTED"
              and not declared.get("requested") else "NOT_EXPORTED")
    reason = (candidates.get("reason")
              or f"the parent run {parent_run_id} has no usable candidate export for {condition}")
    return {"available": False, "status": status, "reason": reason,
            "manifest": None, "rows": {}, "path": candidates.get("path"),
            "parent_run_id": parent_run_id}

def _candidate_reader(release: Any, candidates: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the export's gallery identity ONCE, then resolve rows without re-hashing.

    The digest covers the whole ordered released gallery, so recomputing it per query made a
    6003-query page hash 74k identities 6003 times. The refusal is identical either way -- the
    check is about the export, not about the query.

    ``status`` is what the page reports; ``available`` gates row resolution, so a candidate window
    served from a parent run keeps its own status (COMPLETE/PARTIAL) instead of being relabelled
    ``AVAILABLE`` as if this run had exported it.
    """
    from workbench.composition.topk import gallery_digest

    if not candidates.get("available"):
        status = candidates.get("status") or "TOPK_EXPORT_NOT_REQUESTED"
        return {"available": False, "item_status": status,
                "status": status, "reason": candidates.get("reason"),
                "rows": candidates.get("rows") or {},
                "source_run_id": candidates.get("parent_run_id")}
    manifest = candidates.get("manifest") or {}
    recorded = manifest.get("gallery_identity_sha256")
    if recorded:
        current = gallery_digest(getattr(release.gallery, "protocol_id", ""),
                                 release.gallery.image_ids, release.candidate_names)
        if current != recorded:
            return {"available": False, "item_status": "TOPK_EXPORT_GALLERY_MISMATCH",
                    "status": "TOPK_EXPORT_GALLERY_MISMATCH",
                    "reason": "candidate positions were produced against a different gallery "
                              "identity than the one loaded now", "rows": {},
                    "source_run_id": candidates.get("parent_run_id")}
    return {"available": True, "item_status": candidates.get("status") or "AVAILABLE",
            "status": candidates.get("status") or "AVAILABLE",
            "reason": candidates.get("reason"), "rows": candidates.get("rows") or {},
            "source_run_id": candidates.get("parent_run_id")}

def _resolve_candidates(release: Any, reader: Mapping[str, Any], qid: str) -> dict[str, Any]:
    """Resolve one query's stored gallery positions to released image names."""
    if not reader.get("available"):
        return {"status": reader["status"], "reason": reader["reason"], "rows": []}
    row = (reader["rows"] or {}).get(qid)
    if row is None:
        return {"status": "NO_CANDIDATES_FOR_QUERY", "rows": [],
                "reason": "this query has no stored candidate window "
                          "(unscored, or the export did not reach it)"}
    ids = release.gallery.image_ids
    names = release.candidate_names
    rows = []
    for position, score in row.get("topk") or []:
        if not isinstance(position, int) or not 0 <= position < len(ids):
            return {"status": "TOPK_EXPORT_CORRUPT", "rows": [],
                    "reason": f"stored gallery position {position!r} is outside the gallery"}
        identity = ids[position]
        rows.append({"rank": len(rows) + 1,
                     "image_id": names.get(identity, identity),
                     "gallery_position": position,
                     "score": float(score) if isinstance(score, (int, float)) else None})
    return {"status": reader.get("item_status") or "AVAILABLE", "reason": reader["reason"],
            "rows": rows}

def _summary_for_run(artifacts_root: Path, run_id: str,
                     subset: str = "full") -> dict[str, Any] | None:
    """The run's summary document, or None.

    Used to read facts the run recorded about itself (`metrics.gap_conditions`). Validation and
    decoration are deliberately skipped: the caller wants the recorded value, not a served row.
    """
    try:
        summary_dir = resolve_composition_path(artifacts_root, "summary")
    except ValueError:
        return None
    if not summary_dir.is_dir():
        return None
    for run_dir in summary_dir.iterdir():
        if not run_dir.is_dir():
            continue
        target = run_dir / f"{subset}.json"
        if not target.is_file():
            continue
        summary = read_summary_file(target)
        if summary and summary.get("run_id") == run_id:
            return summary
    return None

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
    query_state: str | None = None,
    top_k: int = 10,
) -> dict[str, Any]:
    """Retrieve paginated per-query composition details with filtering.

    ``query_state`` separates facts that used to be one bucket:

    * ``unscored`` -- no rank exists because the pipeline could not process the query (ambiguous
      garment category, missing reference image, unresolvable positive, missing features). The
      cause is named in ``unscored_kind``.
    * ``retrieval_failure`` -- a real rank exists and is worse than K.
    * ``hit`` -- a real rank exists and is at or better than K.

    ``top_k`` bounds how many candidate rows are attached per query; the stored window may be
    deeper (``candidate_depth``) and its ranks are the scorer's own order, not the metric rank.
    """
    if artifacts_root is None:
        artifacts_root = resolve_config().WORKBENCH_ARTIFACTS_ROOT

    ranks_data = load_run_ranks(artifacts_root, run_id)
    release = load_cached_release()
    if release is None:
        return {
            "items": [],
            "total": 0,
            "limit": limit,
            "offset": offset,
            "queries_unavailable_reason": "CIRCUS release query dataset not available locally.",
        }

    from workbench.composition.topk import read_candidates

    ranks_by_condition = ranks_data.get("ranks", {}) if ranks_data else {}
    failures_by_condition = ranks_data.get("query_failures", {}) if ranks_data else {}
    # The triple the outcome classification reads. Taken from the summary the run recorded, so a run
    # whose image role is an adapter ablation is classified over the conditions it ACTUALLY scored;
    # absent a summary, `outcome_conditions` re-derives the same rule from the rank table.
    summary_for_run = _summary_for_run(artifacts_root, run_id, subset)
    outcome_triple = outcome_conditions(ranks_by_condition, summary_for_run)
    # The conditions this run ACTUALLY measured: the ones its summary DECLARED, in protocol-first
    # order, falling back to the protocol triple only for a summary that predates the field. It used
    # to prepend `CONDITIONS` unconditionally, so a run that measured three conditions
    # (protocol, text-only, and an adapter's named image ablation) advertised a fourth selector entry
    # -- `image_only_empty_text` -- that it never measured, and selecting it served the whole cohort
    # as `unscored` with no candidate window. A declared condition is served only when the rank
    # artifact carries a table for it, and a table the declaration omits is still served: a refused
    # ablation keeps its all-null table and its `ABLATION_UNSUPPORTED` status row.
    declared = run_conditions(summary_for_run)
    measured = declared if declared is not None else (
        *CONDITIONS, *(name for name in ABLATION_CONDITIONS if name not in CONDITIONS))

    def actually_ranked(name: str) -> bool:
        table = ranks_by_condition.get(name)
        return isinstance(table, Mapping) and any(rank is not None for rank in table.values())

    # A condition is served when the run DECLARED it -- so a refused one keeps its entry and its
    # `ABLATION_UNSUPPORTED` status instead of vanishing -- or when the rank artifact holds a real
    # rank for it, which is a measurement whatever the summary says. An all-null table for a
    # condition nobody declared is not a measurement and is not offered.
    served_conditions = tuple(name for name in measured
                              if name in ranks_by_condition or actually_ranked(name))
    served_conditions += tuple(name for name in ranks_by_condition
                               if name not in served_conditions and actually_ranked(name))
    target_condition = condition or "multimodal"
    lineage = (ranks_data or {}).get("derived_from") if isinstance(
        (ranks_data or {}).get("derived_from"), Mapping) else None
    if lineage:
        # A derived run owns no candidates: its window is the PARENT's, served only when the
        # lineage verifies. Reported with the parent's own status, never relabelled.
        candidates = _derived_parent_candidates(artifacts_root, lineage, target_condition)
    else:
        candidates = read_candidates(artifacts_root, run_id, target_condition)
    reader = _candidate_reader(release, candidates)
    candidate_depth = int((candidates.get("manifest") or {}).get("depth") or 0)

    # Filter release queries to subset
    subset_queries = [q for q in release.queries if subset in q.subsets]
    items: list[dict[str, Any]] = []
    counters = {"scored": 0, "unscored": 0, "hit_at_k": 0, "retrieval_failure": 0,
                "unscored_by_kind": {}, "candidates_available": 0}
    # Per-state totals for the chip row. Counted AFTER the explorer's other filters but BEFORE the
    # `query_state` filter itself, so each chip can print its own total while a different chip is
    # active. Counting after that filter would make every inactive chip read 0.
    state_counts = {"hit": 0, "retrieval_failure": 0, "unscored": 0}
    # The outcome filter's own totals, over the SAME predicate the per-row `outcome` uses — one
    # computation, so the filter menu and the rows cannot disagree about how many queries are in a
    # class. (`state_counts` counts a different fact: the branch the ROW is in.)
    outcome_counts: dict[str, int] = {}

    for q in subset_queries:
        q_ranks = {
            c: ranks_by_condition.get(c, {}).get(q.qid)
            for c in served_conditions
        }
        q_outcome = determine_query_outcome(q_ranks, conditions=outcome_triple)
        # `outcome == "unscored"` and the row's `unscored` flag are ONE decision, made here and
        # reported under both names. Deriving them separately is what let a row read
        # `outcome: unscored` beside `target_rank: 12796, unscored: false`.
        row_unscored = q_outcome == "unscored"
        target_rank = q_ranks.get(target_condition)
        failure_msg = readable_failure_message(
            failures_by_condition.get(target_condition, {}).get(q.qid))
        unscored_kind = classify_unscored(failure_msg) if target_rank is None else None
        if target_rank is None:
            state = "unscored"
        elif target_rank <= SHORTCUT_K:
            state = "hit"
        else:
            state = "retrieval_failure"

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
        state_counts[state] += 1
        outcome_counts[q_outcome] = outcome_counts.get(q_outcome, 0) + 1
        if query_state and state != query_state:
            continue

        counters["unscored" if row_unscored else "scored"] += 1
        if state == "hit":
            counters["hit_at_k"] += 1
        elif state == "retrieval_failure":
            counters["retrieval_failure"] += 1
        if unscored_kind:
            kind_counts = counters["unscored_by_kind"]
            kind_counts[unscored_kind] = kind_counts.get(unscored_kind, 0) + 1

        ref_name = release.candidate_names.get(q.reference_id, q.reference_id)
        pos_names = [release.candidate_names.get(p, p) for p in q.positive_ids]
        resolved = _resolve_candidates(release, reader, q.qid)
        window = resolved["rows"][:max(0, top_k)]
        if window:
            counters["candidates_available"] += 1

        items.append({
            "query_id": q.qid,
            "category": q.category,
            "query_txt": q.query_txt,
            "reference_id": ref_name,
            "positive_ids": pos_names,
            "reference_name": ref_name,
            "positive_names": pos_names,
            "ranks": q_ranks,
            "target_rank": target_rank,
            "outcome": q_outcome,
            "query_state": state,
            "unscored": row_unscored,
            "unscored_kind": unscored_kind if row_unscored else None,
            "retrieval_failure": state == "retrieval_failure",
            "failure": failure_msg,
            "top_results": window,
            "candidate_status": resolved["status"],
            "candidate_reason": resolved["reason"],
            "candidate_depth": candidate_depth or None,
            # Which run the window came from. A derived run's window is the parent's, and saying
            # so is the difference between "this run exported nothing" and "you are looking at
            # the parent's candidates".
            "candidate_source_run_id": reader.get("source_run_id") or run_id,
            "candidate_source": "parent_run" if reader.get("source_run_id") else "run",
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
        "condition": target_condition,
        # Which condition keys `items[].ranks` carries for this run: the conditions the run
        # DECLARED (a refused one keeps its entry, so `condition_status` can still say
        # `ABLATION_UNSUPPORTED` for it) plus any condition it holds a real rank for. The UI labels
        # rows from this, never from a hard-coded triple, so a `clvc_image_encoder_only` row is
        # rendered instead of dropped and an unmeasured protocol name is never offered.
        "conditions": [{"id": name, "label": condition_display_label(name)}
                       for name in served_conditions],
        "counters": counters,
        "state_counts": state_counts,
        # Totals per outcome class, over the same predicate as `items[].outcome`.
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "candidates": {
            "available": bool(candidates.get("available")),
            "status": candidates.get("status"),
            "reason": candidates.get("reason"),
            "depth": candidate_depth or None,
            "source_run_id": reader.get("source_run_id") or run_id,
            "source": "parent_run" if lineage else "run",
            "parent_run_id": lineage.get("parent_run_id") if lineage else None,
            "lineage_verified": (None if not lineage else
                                 candidates.get("status") != "parent_topk_unavailable"),
        },
        "lineage": ({
            "derived": True,
            "parent_run_id": lineage.get("parent_run_id"),
            "parent_rank_artifact_sha256": lineage.get("parent_rank_artifact_sha256"),
            "parent_topk_export": lineage.get("parent_topk_export"),
        } if lineage else None),
        # `query_count` must describe the queries this block actually ranked, which is what the
        # per-condition denominators one level above already count. Passing the release's subset
        # size served `query_count: 6003` beside `denominator: 5` on a 5-query smoke and contradicted
        # the same payload's `statistics_whole_run.query_count: 5`; `None` makes the helper derive it
        # from those denominators, so the two can no longer disagree.
        "statistics": _ranks_statistics(ranks_data, None,
                                        {query.qid for query in subset_queries}),
        "statistics_whole_run": _ranks_statistics(ranks_data, None, None),
    }
    if not ranks_data:
        result["queries_unavailable_reason"] = (
            "Per-query exact ranks not found on disk for this run (aggregate-only or pending run).")
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
