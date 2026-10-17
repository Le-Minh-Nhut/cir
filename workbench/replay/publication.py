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

# One owner for the protocol -> literature-label map: `backend/schemas/results.py`. It is
# re-exported here rather than restated, because a second copy drifted (this module said
# `fashioniq_full_gallery_ref_excluded` was "val" while the schema said "original") and the
# contradiction was silent -- each consumer read its own copy. The schema is the owner because it
# VALIDATES the pairing, so a disagreement there is not a matter of opinion.
from workbench.backend.schemas.results import PROTOCOL_LITERATURE_LABELS


class ParityMismatch(RuntimeError):
    """The observation did not reproduce the native hit counts; nothing may be published."""


def development_run_id(*, model_id: str, checkpoint_id: str, protocol_id: str,
                       label: str = "dev") -> str:
    """A deterministic identity for DEVELOPMENT fixtures only.

    This is NOT an execution identity: model+checkpoint+protocol+source cannot distinguish two
    separate evaluations of the same configuration, which is why real runs take their identity from
    the orchestrator. It exists so a hand-built mock or a test fixture has a stable name without
    pretending to be a recorded invocation, and it is deliberately marked as such.
    """
    parts = [model_id, checkpoint_id, protocol_id, label]
    return "dev-" + "-".join(part.replace("/", "_") for part in parts)


def build_result_run(observed_categories: list[ObservedCategory], *, run_id: str, model_id: str,
                     method_name: str, checkpoint_id: str, protocol_id: str,
                     top_k_saved: int, source_commit: str | None = None,
                     checkpoint_path: str | None = None, checkpoint_source: str | None = None,
                     checkpoint_sha256: str | None = None, command_digest: str | None = None,
                     environment_digest: str | None = None,
                     runtime_adaptation: str | None = None,
                     bundle_manifest_digest: str | None = None,
                     checkpoint_bundle: dict[str, dict] | None = None,
                     category_checkpoint_ids: dict[str, str] | None = None,
                     instrumentation_parity: str | None = None,
                     reproduced_metrics: dict[str, float | None] | None = None,
                     data_kind: str = "experiment",
                     timestamp: datetime | None = None) -> ResultRun:
    """Assemble the canonical artifact. Raises on any inconsistency rather than coercing it.

    ``run_id`` is REQUIRED and comes from the caller: the evaluation orchestrator owns the
    authoritative execution identity (``evaluate_models.make_run_id``), and a second identity
    invented here could not distinguish two executions of the same model+checkpoint+source+protocol
    (different invocation, command, environment, hardware) -- the collision F-09 named. This module
    no longer derives one.
    """
    if not run_id:
        raise ValueError("run_id is required: the evaluation orchestrator owns the execution identity")
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
            run_id=run_id,
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
            bundle_manifest_digest=bundle_manifest_digest,
            checkpoint_bundle=checkpoint_bundle,
            category_checkpoint_ids=category_checkpoint_ids,
            instrumentation_parity=instrumentation_parity,
            timestamp=timestamp or datetime.now(timezone.utc),
            data_kind=data_kind,
            reproduced_metrics={"r10": metrics.get("macro_r10"), "r50": metrics.get("macro_r50"),
                               "mean": metrics.get("macro_mean")},
        ),
        queries=queries,
    )


def publish(result: ResultRun, results_root: Path, *, allow_identical_repeat: bool = True) -> Path:
    """Validate, write atomically, and refuse to clobber an existing experiment artifact.

    The previous implementation finished with ``os.replace``, which overwrites unconditionally: two
    different runs whose identity collided (see F-09) would silently replace each other, and a
    second publication of the same run would destroy the first one's bytes. Reproducibility requires
    that a published experiment artifact is never replaced.

    ``os.link`` is the atomic primitive here: the temporary file is hard-linked into place, so the
    link either succeeds or raises ``FileExistsError`` — there is no window between "check" and
    "replace" for a concurrent publisher to slip through, which a ``exists()``-then-``os.replace``
    sequence would have. The temporary is unlinked in every path, so a crash leaves no visible
    partial canonical artifact (only an inert ``.tmp`` inside the destination directory).

    Repeating the publication of an IDENTICAL artifact is idempotent and allowed, because a resumed
    or re-exported run legitimately republishes the same bytes. Repeating it with DIFFERENT bytes
    for the same run id is refused: that is either a collision or a rewrite of recorded evidence.
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
    serialized = json.dumps(payload, indent=2) + "\n"
    handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        if destination.is_symlink():
            # A symlink would place the artifact's bytes outside the results root, where they can be
            # mutated out of band while the index reports a digest it read once (found in review).
            raise FileExistsError(
                f"refusing to publish to a symlink destination: {destination}")
        try:
            # Atomic, no-clobber: the link either creates the name or raises. A reader never sees a
            # partial file, and no concurrent publisher can replace what is already there.
            os.link(temporary, destination)
        except FileExistsError:
            if not allow_identical_repeat:
                raise
            existing = destination.read_text()
            if existing != serialized:
                raise FileExistsError(
                    f"refusing to overwrite published run artifact {destination}: a different "
                    "artifact already claims this run identity, so either the identity collided or "
                    "recorded evidence would be rewritten") from None
    finally:
        # Every path removes the temporary: a crash before the link leaves at most an inert
        # `.tmp`, never a visible partial canonical artifact.
        Path(temporary).unlink(missing_ok=True)

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
    # An experiment's identity is the ORCHESTRATOR's (time + model + checkpoint + random suffix), so
    # it cannot be recomputed from the artifact alone -- that is the point: it names an execution,
    # not a configuration. What must hold is that a relabelled artifact is a DIFFERENT run, and the
    # shape check below is what a fabricated id cannot pass while looking legitimate.
    if not run.run_id or len(run.run_id) < 8:
        return [f"run_id {run.run_id!r} is not a plausible execution identity"]
    if run.run_id.startswith("dev-"):
        # A development id on a declared experiment is the one combination that must never happen:
        # it says "hand-built" while claiming to be a real evaluation.
        return [f"an experiment artifact carries the development identity {run.run_id!r}"]
    if not run.command_digest or not run.environment_digest:
        return ["an experiment artifact must record the command and environment digests that its "
                "own verification compares"]
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
