#!/usr/bin/env python3
"""Opt-in CIRCUS composition CLI; WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE=1 is TEST-ONLY."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# BLAS thread count is set BEFORE numpy is imported, because OpenBLAS sizes its pool at import: a
# value assigned afterwards would be ignored by every path that reaches BLAS.
#
# WHICH paths reach BLAS -- the distinction that matters, and two different phenomena:
#
#  * GEMM-lowering paths DO reach BLAS and their thread count changes their RESULTS, not just
#    throughput. `np.einsum(..., optimize=True)` lowers the contraction to a gemm (csmcir,
#    ptha_mtst, conesep), and the `@` bodies do too (encoder, limn). OpenBLAS partitions the
#    K-reduction differently at different thread counts, so the last bits differ: measured on
#    ENCODER at full 6003-query scale, serial BLAS=1 vs BLAS=10 moved one rank (2->3) and shifted
#    nDCG by 2.181e-05. That is why `runner.py` pins BLAS to 1 during scoring (BLAS_THREADS_PINNED,
#    recorded in the run identity) -- a declared policy, not an optimisation.
#
#  * The naive `np.einsum("nd,gqd->ngq")` used by habit/intent/hint/tme has NO BLAS path at all: it
#    runs a hand-rolled numpy loop. Its ~6.2 q/s is a property of that contraction, NOT of pool
#    sizing or import ordering -- verified flat at OPENBLAS_NUM_THREADS = 1/2/4/8/10 (6.5/6.0/6.2/
#    5.6/6.2 q/s). Flatness across five settings is what rules out a late-sizing bug. Raising threads
#    does not speed those four up; their speedup comes from query-axis parallelism, not BLAS.
#
# The half-box budget comes from AGENTS.md, not from measurement: 10 threads on a 20-thread shared
# host.
_BLAS_THREADS = os.environ.get("WORKBENCH_BLAS_THREADS", "").strip() or str(
    min(10, (os.cpu_count() or 8) // 2))
for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_variable, _BLAS_THREADS)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend import autopilot
from workbench.backend.checkpoint_selection import select_checkpoints
from workbench.backend.operator_config import resolve_config
from workbench.backend.registry import checkpoint_path, load_registry
from workbench.composition import data as composition_data
from workbench.composition.adapters import adapter_class
from workbench.composition.inputs import ABLATION_CONDITIONS, CONDITIONS
from workbench.composition.runner import (
    ModelSpec,
    _gallery_fingerprint,
    assert_composition_destination,
    inspect_image_inventory,
    model_interpreter_identity,
    run_composition,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="autopilot YAML config")
    parser.add_argument("--models", default="selected",
                        help="comma-separated model IDs or 'selected' authorized cohort")
    parser.add_argument("--subsets", default="full,sf,validated",
                        help="comma-separated subsets: full,sf,validated")
    parser.add_argument("--resume", action="store_true", help="reuse matching composition results")
    parser.add_argument("--smoke", nargs="?", const=1, type=int,
                        help="limit query count (default 1); always preserve full gallery")
    parser.add_argument("--plan", action="store_true", help="show work and paths without writing state")
    parser.add_argument("--prepare-data", action="store_true",
                        help="verify release and inventory missing gallery/reference/positive images")
    parser.add_argument("--seed", type=int, default=0,
                        help="recorded in the summary only: no step of the composition path draws "
                             "randomness, so this value is provenance, not a reproducibility "
                             "control, and it does not change the run id")
    parser.add_argument("--topk-depth", type=int, default=None,
                        help="persist candidate Top-K (default: off; costs one small JSONL per "
                             "condition). 50 covers the UI's Top-10 and Top-50 views. Implies "
                             "--keep-features, because the window is rebuilt from that cache when "
                             "a resumed run needs it.")
    parser.add_argument("--keep-features", action="store_true",
                        help="keep the encoded gallery/query memmaps under "
                             "runs/<run_id>/work/<model>/features (~2.4 GB per model) instead of "
                             "deleting them. Delete that directory deliberately when done.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="forward-pass batch size override (env: "
                             "WORKBENCH_COMPOSITION_BATCH_SIZE). The default is unchanged at 32, so "
                             "an unaffected run keeps its run id and its feature cache. Raised "
                             "values change adapter_batch_size in the run identity and therefore "
                             "mint a new run_id: the gallery must be re-encoded once. A value that "
                             "cannot fit in VRAM is refused before the run starts.")
    parser.add_argument("--decode-workers", type=int, default=None,
                        help="parallel image-decode workers (env: WORKBENCH_DECODE_WORKERS, "
                             "default 8). Decode-only: the transform and the feature values are "
                             "unchanged, so this does not alter any result.")
    parser.add_argument("--conditions", default=None,
                        help="comma-separated conditions to score; defaults to the protocol triple "
                             f"({','.join(CONDITIONS)}). Adapter-declared ablations "
                             f"({','.join(ABLATION_CONDITIONS)}) must be named explicitly. The list is "
                             "hashed into the run id, so adding or substituting a condition mints a "
                             "new run rather than reusing the protocol one.")
    return parser

def _resolve_conditions(raw: str | None) -> tuple[str, ...]:
    """Parse `--conditions`: known names only, deduplicated, order preserved.

    Order matters: the gap roles are positional (multimodal, text, image), so the caller's order is
    carried into the run identity rather than sorted away. A duplicate is collapsed rather than
    refused -- naming the same condition twice is a typo, not a second measurement, and the runner
    would otherwise score it twice into one rank table.
    """
    allowed = tuple(CONDITIONS) + tuple(ABLATION_CONDITIONS)
    if raw is None:
        return tuple(CONDITIONS)
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError(f"--conditions was empty; expected one or more of: {', '.join(allowed)}")
    unknown = [item for item in requested if item not in allowed]
    if unknown:
        raise ValueError(f"unknown condition(s): {', '.join(sorted(set(unknown)))}; expected one or "
                         f"more of: {', '.join(allowed)}")
    return tuple(dict.fromkeys(requested))


def _load_configuration(path: Path | None) -> tuple[Any, Any, Any, dict[str, Any]]:
    registry_override = os.environ.get("WORKBENCH_REGISTRY_PATH")
    if path is None:
        config = resolve_config()
        registry = load_registry(Path(registry_override) if registry_override else None)
        return None, None, config, registry
    profile = autopilot.load_config(path)
    config = autopilot._workbench_config(profile)
    registry = load_registry(Path(registry_override)) if registry_override else profile.registry
    return profile, profile, config, registry


def _available_models(registry: dict[str, Any]) -> list[str]:
    return [str(model["model_id"]) for model in registry.get("models", [])
            if model.get("composition_adapter")]


def _path_ready(model: dict[str, Any], checkpoint_id: str, config: Any) -> tuple[bool, str | None]:
    from workbench.backend.registry import checkpoint_by_id

    try:
        checkpoint = checkpoint_by_id(model, checkpoint_id)
        path = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    except Exception as error:
        return False, f"checkpoint selection unavailable: {type(error).__name__}: {error}"
    missing = ([name for name in checkpoint.get("required_files", []) if not (path / name).is_file()]
               if checkpoint.get("bundle_id") else [])
    if missing:
        return False, f"checkpoint bundle missing member(s): {', '.join(missing)}"
    if not path.exists():
        return False, f"checkpoint not staged: {path}"
    return True, None


def _composition_checkpoint(model: dict[str, Any], checkpoint_ids: list[str], bundle_id: str | None,
                            unavailable: list[dict[str, Any]]) -> list[str] | None:
    """Resolve bundles to one job and independent variants to one job each."""
    if bundle_id:
        return [bundle_id]
    selected = set(checkpoint_ids)
    for bundle in model.get("checkpoint_bundles", []):
        members = set(bundle.get("required_checkpoint_ids", []))
        if selected & members:
            if selected.issuperset(members):
                return [str(bundle["bundle_id"])]
            unavailable.append({"model_id": model["model_id"], "status": "UNSUPPORTED",
                                "reason": f"composition supports full checkpoint bundle {bundle['bundle_id']!r}; "
                                          "individual or partial bundle members are unsupported"})
            return None
    return list(dict.fromkeys(checkpoint_ids)) or None


def _select_models(raw: str, registry: dict[str, Any], profile: Any,
                   config: Any) -> tuple[list[tuple[dict[str, Any], str]], list[dict[str, Any]]]:
    models = registry.get("models", [])
    by_id = {model["model_id"]: model for model in models}
    pairs: list[tuple[dict[str, Any], str]] = []
    unavailable: list[dict[str, Any]] = []
    if raw == "selected":
        if profile is None:
            raise ValueError("--models selected requires --config to reuse authorized autopilot cohort")
        plan = autopilot.resolved_plan(profile, dry_run=True)
        rows = {row["model_id"]: row for row in plan["models"]}
        for model_id in profile.selected_model_ids:
            model = by_id[model_id]
            row = rows.get(model_id, {})
            checkpoint_ids = row.get("checkpoint_ids") or []
            if not checkpoint_ids:
                unavailable.append({"model_id": model_id, "status": row.get("status", "NOT_READY"),
                                    "reason": "; ".join(row.get("blockers", [])) or
                                    "authorized cohort selection has no checkpoint"})
                continue
            selection = row.get("checkpoint_selection") or {}
            checkpoint_ids = _composition_checkpoint(model, checkpoint_ids,
                                                     selection.get("bundle_id"), unavailable)
            if checkpoint_ids is None:
                continue
            for checkpoint_id in checkpoint_ids:
                pairs.append((model, checkpoint_id))
                ready, reason = _path_ready(model, checkpoint_id, config)
                if not ready:
                    unavailable.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                                        "status": "NOT_READY", "reason": reason})
        return pairs, unavailable
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("--models needs model IDs or literal 'selected'")
    unknown = sorted(set(requested) - set(by_id))
    if unknown:
        raise ValueError(f"unknown model ID(s): {', '.join(unknown)}; available: {', '.join(_available_models(registry))}")
    for model_id in requested:
        model = by_id[model_id]
        if not model.get("composition_adapter"):
            unavailable.append({"model_id": model_id, "status": "UNSUPPORTED",
                                "reason": "registry has no composition_adapter module path"})
            continue
        selection = select_checkpoints(
            model, policy=getattr(profile, "checkpoint_policy", "strict_clean"),
            requested=getattr(profile, "checkpoint_overrides", {}).get(model_id),
            checkpoint_root=config.WORKBENCH_CHECKPOINT_ROOT)
        if selection.blocked:
            unavailable.append({"model_id": model_id, "status": "NOT_READY", "reason": selection.reason})
            continue
        checkpoint_ids = _composition_checkpoint(model, list(selection.checkpoint_ids),
                                                 selection.bundle_id, unavailable)
        if checkpoint_ids is None:
            continue
        for checkpoint_id in checkpoint_ids:
            pairs.append((model, checkpoint_id))
            ready, reason = _path_ready(model, checkpoint_id, config)
            if not ready:
                unavailable.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                                    "status": "NOT_READY", "reason": reason})
    return pairs, unavailable


def _checkpoint_identity(model: dict[str, Any], checkpoint_id: str, config: Any) -> str:
    from workbench.backend.registry import checkpoint_by_id, sha256_file

    checkpoint = checkpoint_by_id(model, checkpoint_id)
    path = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        digest = hashlib.sha256()
        for item in sorted(child for child in path.rglob("*") if child.is_file()):
            digest.update(item.relative_to(path).as_posix().encode("utf-8"))
            digest.update(sha256_file(item).encode("ascii"))
        return digest.hexdigest()
    return "missing"


def _load_data(config: Any, *, synthetic: bool):
    root = getattr(config, "WORKBENCH_CIRCUS_RELEASE_ROOT", None)
    if root is None:
        root = Path(config.CIR_DATA_ROOT) / "circus_release"
    return composition_data.load_release(Path(root), verify_hashes=not synthetic,
                                         expect_released_counts=not synthetic)


def _plan(pairs: list[tuple[dict[str, Any], str]], unavailable: list[dict[str, Any]],
          data: Any, config: Any, subsets: list[str],
          conditions: tuple[str, ...] = tuple(CONDITIONS)) -> dict[str, Any]:
    artifacts = Path(config.WORKBENCH_ARTIFACTS_ROOT).expanduser().resolve()
    inventory = inspect_image_inventory(data, [model for model, _checkpoint in pairs],
                                        Path(config.FASHIONIQ_ROOT))
    outputs = []
    readiness = []
    for model, checkpoint_id in pairs:
        try:
            adapter = adapter_class(model["model_id"], {"models": [model]})
            status = adapter.status(config)
            runtime_identity = model_interpreter_identity(model, config, verify=False)
            readiness.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                              "ready": status.ready,
                              "status": "READY" if status.ready else "SELECTED_BUT_BLOCKED",
                              "reason": status.reason,
                              "adapter": model.get("composition_adapter"),
                              "runtime": {"model_interpreter": runtime_identity}})
        except Exception as error:
            readiness.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                              "ready": False, "status": "SELECTED_BUT_BLOCKED",
                              "reason": f"{type(error).__name__}: {error}",
                              "adapter": model.get("composition_adapter"),
                              "runtime": {"model_interpreter": model_interpreter_identity(model, config, verify=False)}})
        for subset in subsets:
            # The SAME resolved list the run will use, never the module constant: a plan that lists
            # an output the run will not write, or omits one it will, misdescribes the run.
            for condition in conditions:
                outputs.append(str(assert_composition_destination(
                    artifacts, artifacts / "composition" / "results" / data.gallery.protocol_id /
                    model["model_id"] / checkpoint_id / "<attempt>" / subset / f"{condition}.jsonl")))
            outputs.append(str(assert_composition_destination(
                artifacts, artifacts / "composition" / "summary" /
                f"{model['model_id']}_{checkpoint_id}_<attempt>" / f"{subset}.json")))
    # ONE formula for one gallery: the value every completed run records. A second derivation here
    # published `0bfd43b9…` where the run published `1db68c59…` for the identical 74,381-image
    # gallery, so "did this plan produce this run" could never be answered by comparing the two.
    gallery_identity = _gallery_fingerprint(data)
    return {"selected_models": [{"model_id": model["model_id"], "checkpoint_id": checkpoint}
                                 for model, checkpoint in pairs],
            "unavailable": unavailable, "readiness": readiness,
            "protocol_id": data.gallery.protocol_id, "gallery_count": len(data.gallery.image_ids),
            "gallery_identity_sha256": gallery_identity, "missing_assets": inventory,
            "expected_output_paths": outputs,
            "conditions": list(conditions),
            "category_resolution": data.provenance.get("category_resolution", {}),
            "is_synthetic": bool(data.provenance.get("is_synthetic")),
            "failed_expectations": data.provenance.get("failed_expectations", ())}


def _runner_topk_kwargs(runner: Any, depth: int | None,
                        keep_features: bool = False) -> dict[str, Any]:
    """Pass ``topk_depth`` only when the runner accepts it.

    The export is implemented entirely in ``workbench/composition/topk.py``; the runner still has
    to call it (see ``workbench/docs/reviews/eval-runner-requests.md``). Until it does, asking for
    ``--topk-depth`` must fail loudly rather than write a run that silently stores nothing: a
    summary claiming an export that does not exist would be worse than a refusal.
    """
    import inspect

    parameters = inspect.signature(runner).parameters
    if depth is None:
        return {"keep_features": True} if keep_features and "keep_features" in parameters else {}
    if "topk_depth" not in parameters:
        raise SystemExit(
            "COMPOSITION-TOPK-UNAVAILABLE: this runner does not accept topk_depth yet "
            "(workbench/docs/reviews/eval-runner-requests.md). Refusing to run with candidate "
            "export requested rather than producing a run that silently stores none.")
    kwargs: dict[str, Any] = {"topk_depth": depth}
    # An export implies the cache: a deleted feature cache makes the window rebuildable only by
    # re-encoding the full gallery.
    if "keep_features" in parameters:
        kwargs["keep_features"] = True
    return kwargs


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    # CLI knobs are the same settings the adapters read from the environment; exporting them here
    # keeps ONE resolution point (adapters.resolved_batch_size / resolved_decode_workers) instead of
    # threading two more parameters through the runner and every adapter.
    from workbench.composition.adapters import BATCH_SIZE_ENV, DECODE_WORKERS_ENV
    if args.batch_size is not None:
        if args.batch_size < 1:
            print(json.dumps({"error": "--batch-size must be a positive integer"}))
            return 2
        os.environ[BATCH_SIZE_ENV] = str(args.batch_size)
    if args.decode_workers is not None:
        if args.decode_workers < 1:
            print(json.dumps({"error": "--decode-workers must be a positive integer"}))
            return 2
        os.environ[DECODE_WORKERS_ENV] = str(args.decode_workers)
    # Resolved BEFORE the run, and before any configuration is loaded: an unknown condition name is
    # a usage error (exit 2), not a failed run, and it must not get as far as reading the release or
    # planning a gallery encode.
    try:
        conditions = _resolve_conditions(args.conditions)
    except ValueError as error:
        print(json.dumps({"error": str(error)}))
        return 2
    try:
        profile, _, config, registry = _load_configuration(args.config)
        subsets = [item.strip() for item in args.subsets.split(",") if item.strip()]
        allowed = {"full", "sf", "validated"}
        if not subsets or len(set(subsets)) != len(subsets) or set(subsets) - allowed:
            raise ValueError("--subsets must be comma-separated values from full,sf,validated")
        fixture_value = os.environ.get("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE")
        if fixture_value not in (None, "0", "1"):
            raise ValueError("WORKBENCH_CIRCUS_SYNTHETIC_FIXTURE must be exactly '1' for tests or unset/'0'")
        synthetic = fixture_value == "1"
        requested_order = (list(profile.selected_model_ids) if args.models == "selected" and profile is not None
                           else [item.strip() for item in args.models.split(",") if item.strip()])
        pairs, unavailable = _select_models(args.models, registry, profile, config)
        data = _load_data(config, synthetic=synthetic)
        data = composition_data.assign_garment_categories(data, config.FASHIONIQ_ROOT)
        if args.plan or args.prepare_data:
            report = _plan(pairs, unavailable, data, config, subsets, conditions)
            if args.prepare_data:
                from workbench.composition.data import resolve_release_paths, verify_release_hashes

                reports = verify_release_hashes(resolve_release_paths(config.WORKBENCH_CIRCUS_RELEASE_ROOT))
                report["release_hashes"] = [{"path": str(item.path), "expected": item.expected,
                                             "actual": item.actual, "matches": item.matches}
                                            for item in reports]
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
            return 0
        blocked = [{"model_id": item["model_id"], "checkpoint_id": item.get("checkpoint_id"),
                    "status": item["status"] if item["status"] == "UNSUPPORTED" else "SELECTED_BUT_BLOCKED",
                    "reason": item["reason"]}
                   for item in unavailable]
        specs = []
        blocked_pairs = {(item["model_id"], item.get("checkpoint_id")) for item in unavailable}
        for model, checkpoint_id in pairs:
            if (model["model_id"], checkpoint_id) in blocked_pairs:
                continue
            try:
                adapter_type = adapter_class(model["model_id"], registry)
                adapter_status = adapter_type.status(config)
            except Exception as error:
                blocked.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                                "status": "SELECTED_BUT_BLOCKED",
                                "reason": f"adapter readiness failed: {type(error).__name__}: {error}"})
                continue
            if not adapter_status.ready:
                blocked.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                                "status": "SELECTED_BUT_BLOCKED",
                                "reason": adapter_status.reason or "adapter is not ready"})
                continue
            ready, reason = _path_ready(model, checkpoint_id, config)
            if not ready:
                blocked.append({"model_id": model["model_id"], "checkpoint_id": checkpoint_id,
                                "status": "SELECTED_BUT_BLOCKED", "reason": reason})
                continue
            specs.append(ModelSpec(model["model_id"], checkpoint_id,
                                   lambda cfg, cid, cls=adapter_type: cls(cfg, cid), model,
                                   _checkpoint_identity(model, checkpoint_id, config)))
        report = run_composition(data, specs, config, subsets=subsets, conditions=conditions,
                                 resume=args.resume, smoke=args.smoke, seed=args.seed,
                                 synthetic=bool(data.provenance.get("is_synthetic")),
                                 **_runner_topk_kwargs(run_composition, args.topk_depth,
                                                       args.keep_features)) if specs else {
                                     "run_count": 0, "models": [], "gallery_count": len(data.gallery.image_ids),
                                     # Same helper as `_plan` and the runner: a blocked report must
                                     # not reintroduce the second formula the plan branch dropped.
                                     "gallery_identity_sha256": _gallery_fingerprint(data),
                                     "protocol_id": data.gallery.protocol_id,
                                     "synthetic": bool(data.provenance.get("is_synthetic"))}
        report["models"].extend(blocked)
        report["run_count"] = len(report["models"])
        order = {model_id: index for index, model_id in enumerate(requested_order)}
        report["models"].sort(key=lambda row: order.get(row["model_id"], len(order)))
        print(json.dumps(report, indent=2, sort_keys=True, default=str))
        failures = {"FAILED", "BLOCKED_MISSING_GALLERY_IMAGES", "MODEL_UNAVAILABLE", "PARTIAL"}
        if args.models != "selected":
            failures.update({"SELECTED_BUT_BLOCKED", "UNSUPPORTED"})
        # Fail runtime errors, explicit unavailable requests, or selected cohorts with zero runnable results.
        selected_empty = args.models == "selected" and not any(
            row["status"] in {"COMPLETED", "PARTIAL", "REUSED"} for row in report["models"])
        return 1 if selected_empty or any(row["status"] in failures for row in report["models"]) else 0
    except Exception as error:
        print(f"[COMPOSITION-EVAL-FAILED] {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
