"""Publish a staged observation payload as a canonical per-query artifact.

This module runs under the WORKBENCH interpreter, never the model's, because it imports the
pydantic-backed canonical schema. The split exists for exactly that reason: the LIMN model
environment is provisioned from the model's own requirements (torch, open_clip, numpy) and does not
carry pydantic, so an exporter that imported the schema crashed on import in a clean environment --
the aggregate succeeded while every per-query export failed.

So the flow is:

    model interpreter      observe the rankings, write plain JSON  (`replay.limn_export`)
        -> payload file
    workbench interpreter  validate, admit, publish atomically      (this module)

The parent (`evaluate_models._run_observation`) runs the observer as a child and then calls this
publisher, so the artifact is still bound to the SAME invocation: it carries the orchestrator's
`run_id` and the invocation's command/environment digests, both of which the observer was given.
"""
from __future__ import annotations

import json
from pathlib import Path

from workbench.replay.limn_observer import ObservedCategory, ObservedQuery
from workbench.replay.publication import publish_observed_categories


class PayloadError(ValueError):
    """The staged payload is missing, malformed, or not an observation payload."""


def load_payload(path: Path) -> dict:
    let = path.read_text(encoding="utf-8") if path.is_file() else None
    if let is None:
        raise PayloadError(f"observation payload does not exist: {path}")
    try:
        payload = json.loads(let)
    except ValueError as error:
        raise PayloadError(f"observation payload is not JSON: {error}") from error
    expected_type = {"limn": "limn_observation_payload",
                     "encoder": "encoder_observation_payload"}.get(
                         payload.get("model_id") if isinstance(payload, dict) else None)
    if not expected_type or payload.get("artifact_type") != expected_type:
        raise PayloadError(f"{path} is not an observation payload")
    _require_declared_bundle(payload, path)
    return payload


def _require_declared_bundle(payload: dict, path: Path) -> None:
    """The staged members must match the bundle the REGISTRY declares for this checkpoint.

    The planned-digest comparison is the strongest check available, but it lives in the payload: the
    observer writes both the members and the digest it is compared against, and an operator running
    this module directly supplies both. Neither can be the authority for what "all categories" means.

    The registry can. LIMN's `base_iter0_all_categories` declares three required checkpoint ids --
    one per category -- and a payload is an observation of that bundle only if it staged exactly
    them. Without this, a payload declaring a single member published successfully: a
    "three-category" per-query artifact whose recorded bundle contained one file, which no consumer
    comparing bundle identity would catch, because the identity was internally consistent.
    """
    from workbench.backend.registry import model_by_id

    checkpoint_id = payload.get("checkpoint_id")
    model_id = payload.get("model_id")
    if not checkpoint_id or not model_id:
        return
    try:
        model = model_by_id(str(model_id))
    except Exception:
        return
    bundle = next((item for item in (model.get("checkpoint_bundles") or [])
                   if item.get("bundle_id") == checkpoint_id), None)
    if not bundle:
        # A single-category checkpoint declares no bundle, and there is nothing to compare against.
        return
    required = list(bundle.get("required_checkpoint_ids") or [])
    if not required:
        return
    members = payload.get("checkpoint_members") or {}
    declared = {str(member.get("checkpoint_id") or "") for member in members.values()}
    if declared != set(required):
        raise PayloadError(
            f"{path} stages checkpoint members for {sorted(declared)}, but {model_id}'s "
            f"{checkpoint_id!r} declares the bundle {sorted(required)}: a per-query artifact must "
            "observe every member the registry requires, and a payload cannot decide for itself "
            "what that set is")


def _observed_categories(payload: dict) -> list[ObservedCategory]:
    categories = []
    for entry in payload.get("categories") or []:
        queries = tuple(
            ObservedQuery(
                annotation_index=query["annotation_index"],
                reference_id=query["reference_id"],
                target_id=query["target_id"],
                raw_captions=tuple(query["raw_captions"]),
                model_input_text=query["model_input_text"],
                target_rank=query["target_rank"],
                top_results=tuple((item["rank"], item["image_id"], item["score"])
                                  for item in query["top_results"]),
                category=entry["category"],
            )
            for query in entry["queries"])
        categories.append(ObservedCategory(
            category=entry["category"], gallery_size=entry["gallery_size"],
            eligible_count=entry["eligible_count"], queries=queries))
    if not categories:
        raise PayloadError("the observation payload contains no categories")
    return categories


def publish_payload(path: Path, results_root: Path) -> Path:
    """Publish the staged payload. Refuses on a parity mismatch, as the artifact contract requires."""
    # The bundle identity the artifact records must describe the SAME members the orchestrator
    # planned and whose bytes were hashed before loading. The payload carries them, so a payload that
    # disagrees with the plan would otherwise publish an artifact whose recorded bundle identity is
    # self-declared -- and nothing downstream compares the per-query artifact's digest with the
    # aggregate report's. A disagreement is refused rather than recorded, and it is checked FIRST:
    # which bundle ran is a property of the run, not of whether the payload happens to parse.
    payload = load_payload(path)
    members = payload.get("checkpoint_members") or {}
    planned_digest = payload.get("planned_bundle_manifest_digest")
    if planned_digest:
        from workbench.scripts.evaluate_models import manifest_digest as _manifest_digest

        staged_digest = _manifest_digest(members) if members else None
        if staged_digest != planned_digest:
            raise PayloadError(
                "the observer staged a checkpoint bundle whose digest does not match the plan's: "
                f"staged={staged_digest} planned={planned_digest}; a per-query artifact must record "
                "the members the aggregate run actually evaluated")

    mismatches = {name: tuple(items)
                  for name, items in (payload.get("instrumentation_mismatches") or {}).items()}
    observed = _observed_categories(payload)
    category_checkpoint_ids = {category: member.get("checkpoint_id")
                               for category, member in members.items()}
    # The evaluator computes the SAME digest for the same bundle from its own manifest. A second
    # implementation here produced two unequal "bundle manifest digests" for one artifact whenever
    # the two keyed their members differently (the evaluator keys by filename, this module's member
    # map is keyed by category), so the two halves of one runs's provenance could never agree.
    from workbench.scripts.evaluate_models import manifest_digest

    bundle_digest = manifest_digest(members) if members else None

    destination = publish_observed_categories(
        observed, results_root, parity_mismatches=mismatches,
        run_id=payload["run_id"],
        model_id=payload["model_id"], method_name=payload.get("method_name") or payload["model_id"],
        checkpoint_id=payload["checkpoint_id"], protocol_id=payload["protocol_id"],
        top_k_saved=payload["top_k"], source_commit=payload.get("upstream_commit"),
        checkpoint_sha256=next((member.get("sha256") for member in members.values()), None),
        command_digest=payload.get("command_digest"),
        environment_digest=payload.get("environment_digest"),
        runtime_adaptation=payload.get("runtime_adaptation"),
        bundle_manifest_digest=bundle_digest,
        checkpoint_bundle=members or None,
        category_checkpoint_ids=category_checkpoint_ids or None,
        instrumentation_parity=payload.get("instrumentation_parity"),
        # A payload staged without an orchestrator identity is a development run: the observer marks
        # it as one, and `publish` refuses to let a development identity ride on an experiment.
        data_kind="experiment" if not str(payload["run_id"]).startswith("dev-") else "mock",
    )
    return destination


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        destination = publish_payload(args.payload, args.results_root)
    except Exception as error:
        print(f"[BLOCKED] {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"published": str(destination)}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
