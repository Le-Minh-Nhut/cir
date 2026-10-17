"""Publication of canonical per-query ResultRun artifacts (§17, §18).

Separated from the observation code on purpose: the observer produces rankings, and this module is
the only writer. A child evaluator must not self-certify its own per-query artifact, so the artifact
is validated by the canonical schema and then independently re-read from disk before the path is
handed back. Publication is atomic — a partial file is never visible under the results root.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from workbench.backend.schemas.results import QueryResult, ResultRun, RunMetadata, TopResult
from workbench.replay.limn_observer import ObservedCategory

PROTOCOL_LITERATURE_LABELS = {
    "fashioniq_original_split": "original",
    "fashioniq_full_gallery_ref_excluded": "val",
    "fashioniq_val_split": "val",
}


class ParityMismatch(RuntimeError):
    """The observation did not reproduce the native hit counts; nothing may be published."""


def run_id_for(*, model_id: str, checkpoint_id: str, protocol_id: str, source_commit: str | None,
               checkpoint_sha256: str | None) -> str:
    """Deterministic run identity: the same inputs describe the same run.

    Includes the source commit and checkpoint digest so a rerun after either changes is a new run
    rather than a silent overwrite of a different provenance (§17).
    """
    parts = [model_id, checkpoint_id, protocol_id, source_commit or "unknown",
             (checkpoint_sha256 or "unknown")[:16]]
    return "run-" + "-".join(part.replace("/", "_") for part in parts)


def build_result_run(observed_categories: list[ObservedCategory], *, model_id: str,
                     method_name: str, checkpoint_id: str, protocol_id: str,
                     top_k_saved: int, source_commit: str | None = None,
                     checkpoint_path: str | None = None, checkpoint_source: str | None = None,
                     checkpoint_sha256: str | None = None, command_digest: str | None = None,
                     environment_digest: str | None = None,
                     runtime_adaptation: str | None = None,
                     reproduced_metrics: dict[str, float | None] | None = None,
                     data_kind: str = "experiment",
                     timestamp: datetime | None = None) -> ResultRun:
    """Assemble the canonical artifact. Raises on any inconsistency rather than coercing it."""
    label = PROTOCOL_LITERATURE_LABELS.get(protocol_id)
    if label is None:
        raise ValueError(f"unsupported protocol_id: {protocol_id}")
    if not observed_categories:
        raise ValueError("no observed categories to publish")

    categories = [category.category for category in observed_categories]
    if len(set(categories)) != len(categories):
        raise ValueError("duplicate category in observed results")

    queries = []
    for category in observed_categories:
        # Per-category raw gallery size. The schema's scalar gallery_size cannot express a
        # category-specific bound, so the largest is declared and each query is checked against its
        # own category's size here — a dress rank of 300 is not excused by a larger toptee gallery.
        for query in category.queries:
            if query.target_rank > category.gallery_size:
                raise ValueError(f"{query.query_id}: target_rank exceeds its category gallery size")
            queries.append(QueryResult(
                query_id=query.query_id,
                category=query.category,
                annotation_index=query.annotation_index,
                reference_id=query.reference_id,
                target_id=query.target_id,
                raw_captions=list(query.raw_captions),
                model_input_text=query.model_input_text,
                target_rank=query.target_rank,
                top_results=[TopResult(rank=rank, image_id=image_id, score=score)
                             for rank, image_id, score in query.top_results],
            ))

    metrics = reproduced_metrics or {}
    return ResultRun(
        schema_version=2,
        run=RunMetadata(
            run_id=run_id_for(model_id=model_id, checkpoint_id=checkpoint_id,
                               protocol_id=protocol_id, source_commit=source_commit,
                               checkpoint_sha256=checkpoint_sha256),
            dataset="FashionIQ", split="val", protocol_id=protocol_id,
            literature_split_label=label, evaluation_noise_pct=0,
            model_id=model_id, method_name=method_name, checkpoint_id=checkpoint_id,
            top_k_saved=top_k_saved,
            gallery_size=max(category.gallery_size for category in observed_categories),
            category_gallery_sizes={category.category: category.gallery_size
                                    for category in observed_categories},
            category_eligible_counts={category.category: category.eligible_count
                                      for category in observed_categories},
            upstream_commit=source_commit, checkpoint_path=checkpoint_path,
            checkpoint_source=checkpoint_source, checkpoint_sha256=checkpoint_sha256,
            command_digest=command_digest, environment_digest=environment_digest,
            runtime_adaptation=runtime_adaptation,
            timestamp=timestamp or datetime.now(timezone.utc),
            data_kind=data_kind,
            reproduced_metrics={"r10": metrics.get("macro_r10"), "r50": metrics.get("macro_r50"),
                               "mean": metrics.get("macro_mean")},
        ),
        queries=queries,
    )


def publish(result: ResultRun, results_root: Path) -> Path:
    """Validate, write atomically, then independently re-read from disk.

    Returns the published path. The atomic replace means a reader never sees a partial artifact,
    and the re-read means publication is not a claim about what was written but a fact about what
    is now on disk.
    """
    payload = result.model_dump(mode="json")
    # Validate the serialized form too: a field that only survives model_dump as a non-JSON object
    # would otherwise reach consumers as a surprise.
    validated = ResultRun.model_validate(json.loads(json.dumps(payload)))
    if mismatches := verify_run_identity(validated):
        raise ValueError("; ".join(mismatches))

    destination = (results_root / validated.run.protocol_id / validated.run.model_id
                   / f"{validated.run.run_id}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as file:
            json.dump(payload, file, indent=2)
            file.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise

    # Independent read-back: the artifact on disk must validate on its own.
    ResultRun.model_validate_json(destination.read_text())
    return destination


def verify_run_identity(result: ResultRun) -> list[str]:
    """Recompute the run's identity from its own recorded provenance and compare.

    ``run_id`` is derived from the model, checkpoint, protocol, source commit and checkpoint
    digest, so a reader can recompute it. Without this check the field is only a label: an artifact
    whose ``run_id`` was edited to another run's would still validate, and a consumer keying on it
    would attribute one run's numbers to another. Returns the mismatches (empty means consistent).

    Only ``experiment`` artifacts are checked. A ``mock``/``synthetic`` fixture is declared
    development data whose whole purpose is to be hand-written, and it never becomes research
    evidence (``verify_run.py`` refuses any non-experiment report), so demanding derived identity
    from it would reject the fixtures while protecting nothing.
    """
    run = result.run
    if run.data_kind != "experiment":
        return []
    expected = run_id_for(model_id=run.model_id, checkpoint_id=run.checkpoint_id,
                          protocol_id=run.protocol_id, source_commit=run.upstream_commit,
                          checkpoint_sha256=run.checkpoint_sha256)
    if run.run_id != expected:
        return [f"run_id {run.run_id!r} is not the identity its own provenance derives "
                f"({expected!r}); the artifact was relabelled"]
    return []


def publish_observed_categories(observed_categories: list[ObservedCategory],
                                results_root: Path,
                                parity_mismatches: tuple[str, ...] | dict[str, tuple[str, ...]] = (),
                                **identity: Any) -> Path:
    """Refuse on any parity mismatch, then publish.

    ``parity_mismatches`` is the PARITY 1 outcome from the replay: either a flat tuple (all
    categories) or a per-category mapping. Anything non-empty blocks publication, because a
    per-query artifact that does not reproduce the native ranking is not evidence about the run.
    """
    if isinstance(parity_mismatches, dict):
        offending = {category: list(mismatches) for category, mismatches in parity_mismatches.items() if mismatches}
    else:
        offending = {"all": list(parity_mismatches)} if parity_mismatches else {}
    if offending:
        raise ParityMismatch(
            "QUERY_EXPORT_PARITY_MISMATCH: observed rankings do not reproduce the native hit "
            f"counts; the native aggregate stands and no per-query artifact may be published: {offending}")
    return publish(build_result_run(observed_categories, **identity), results_root)
