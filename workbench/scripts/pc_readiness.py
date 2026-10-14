#!/usr/bin/env python3
"""Read-only PC release readiness check.

Answers one question per registered model: what, precisely, stops it from running on the target
machine? It reads configuration, registries, the filesystem and git, and reports; it never writes,
downloads, installs or executes an evaluator.

Run it on the PC before preparing anything:

    python workbench/scripts/pc_readiness.py --json
    python workbench/scripts/pc_readiness.py --model limn

The classification is deliberately one of a small set, and a model is never `READY` merely because a
synthetic test passed elsewhere. `PENDING` means the check could not be completed on this host, which
is not the same as `READY`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from workbench.backend import benchmark_references
from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.preflight import approved_patch_state, model_requirements, preflight_blockers
from workbench.backend.registry import load_registry

#: Classification, most-specific first. A model is described by the first state whose condition holds.
READY_FOR_FULL_EVALUATION = "READY_FOR_FULL_EVALUATION"
READY_FOR_GPU_SMOKE = "READY_FOR_GPU_SMOKE"
READY_WITH_DECLARED_VARIANT = "READY_WITH_DECLARED_VARIANT"
BLOCKED_EXTERNAL_ASSET = "BLOCKED_EXTERNAL_ASSET"
BLOCKED_DATASET = "BLOCKED_DATASET"
BLOCKED_SOURCE = "BLOCKED_SOURCE"
BLOCKED_IMPLEMENTATION = "BLOCKED_IMPLEMENTATION"
BLOCKED_ENVIRONMENT = "BLOCKED_ENVIRONMENT"
PC_GPU_VALIDATION_PENDING = "PC_GPU_VALIDATION_PENDING"

#: Which blocker text belongs to which cause. Order matters: the first match classifies the model.
BLOCKER_KINDS = (
    ("source cannot run", BLOCKED_SOURCE),
    ("upstream source unavailable", BLOCKED_SOURCE),
    ("source checkout is dirty", BLOCKED_SOURCE),
    ("source pin mismatch", BLOCKED_SOURCE),
    ("official evaluator missing", BLOCKED_IMPLEMENTATION),
    ("command not runnable", BLOCKED_IMPLEMENTATION),
    ("environment", BLOCKED_ENVIRONMENT),
    ("FashionIQ", BLOCKED_DATASET),
    ("raw benchmark input missing", BLOCKED_DATASET),
    ("required generated artifact missing", BLOCKED_DATASET),
    ("canonical dataset link", BLOCKED_DATASET),
    ("captions", BLOCKED_EXTERNAL_ASSET),
    ("asset", BLOCKED_EXTERNAL_ASSET),
    ("missing", BLOCKED_EXTERNAL_ASSET),
)


def classify(blockers: list[str]) -> tuple[str, str]:
    """Map a blocker list to one state and the reason a reader should act on."""
    if not blockers:
        return READY_FOR_FULL_EVALUATION, "no blocker reported by preflight"
    for needle, state in BLOCKER_KINDS:
        for blocker in blockers:
            if needle.lower() in blocker.lower():
                return state, blocker
    return BLOCKED_IMPLEMENTATION, blockers[0]


def repository_section() -> dict:
    import subprocess

    def git(*args: str) -> str | None:
        try:
            return subprocess.check_output(["git", "-C", str(REPO), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "dirty_files": len((git("status", "--porcelain") or "").splitlines()),
    }


def config_section(config: WorkbenchConfig) -> dict:
    return {
        "repo_root": str(config.CIR_REPO_ROOT),
        "fashioniq_root": str(config.FASHIONIQ_ROOT),
        "third_party_root": str(config.WORKBENCH_THIRD_PARTY_ROOT),
        "checkpoint_root": str(config.WORKBENCH_CHECKPOINT_ROOT),
        "results_root": str(config.WORKBENCH_RESULTS_ROOT),
        "artifacts_root": str(config.WORKBENCH_ARTIFACTS_ROOT),
        "fashioniq_root_exists": config.FASHIONIQ_ROOT.is_dir(),
    }


def dataset_section(config: WorkbenchConfig) -> dict:
    root = config.FASHIONIQ_ROOT
    images = root / "images"
    extensions: dict[str, int] = {}
    if images.is_dir():
        for entry in images.iterdir():
            extensions[entry.suffix.lower()] = extensions.get(entry.suffix.lower(), 0) + 1
    return {
        "root": str(root),
        "captions": (root / "captions").is_dir(),
        "image_splits": (root / "image_splits").is_dir(),
        "images": images.is_dir(),
        "image_count": sum(extensions.values()),
        "image_extensions": extensions,
        "resized_image": (root / "resized_image").is_dir(),
        "correction_dicts": sorted(p.name for p in (root / "captions").glob("correction_dict_*.json")),
    }


def papers_section() -> dict:
    try:
        data = benchmark_references.load_registry()
    except (FileNotFoundError, ValueError) as error:
        return {"available": False, "error": str(error)}
    rows = data["references"]
    verified = [r for r in rows if r.get("reference_status") == benchmark_references.REFERENCE_VERIFIED]
    return {
        "available": True,
        "registry_file": str(benchmark_references.REFERENCES_FILE),
        "version": str(data.get("registry_version")),
        "references": len(rows),
        "verified": len(verified),
        "models_covered": sorted({r["model_id"] for r in rows}),
    }


def environment_section(config: WorkbenchConfig, models: list[dict]) -> dict:
    from workbench.backend.runtime import environment_blockers

    declared = {}
    for model in models:
        declared[model["model_id"]] = {
            "python": (model.get("environment") or {}).get("python"),
            "pytorch": (model.get("environment") or {}).get("pytorch"),
            "cuda": (model.get("environment") or {}).get("cuda"),
            "env_var": (model.get("environment") or {}).get("python_env_var"),
            "declared": bool(model.get("environment_required")),
            "blockers": environment_blockers(model),
        }
    return {"per_model": declared}


def gpu_section() -> dict:
    """What can be observed about the accelerator from here, without importing a model stack."""
    import shutil
    import subprocess

    section: dict = {"nvidia_smi": bool(shutil.which("nvidia-smi")), "devices": []}
    if section["nvidia_smi"]:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total,driver_version",
                 "--format=csv,noheader"], text=True, stderr=subprocess.DEVNULL).strip()
            section["devices"] = [line.strip() for line in out.splitlines() if line.strip()]
        except (OSError, subprocess.CalledProcessError) as error:
            section["error"] = str(error)
    try:
        import torch

        section["torch"] = torch.__version__
        section["cuda_available"] = bool(torch.cuda.is_available())
        section["cuda_build"] = torch.version.cuda
        section["arch_list"] = list(torch.cuda.get_arch_list()) if torch.cuda.is_available() else []
    except Exception as error:  # a missing or broken torch is exactly what this check reports
        section["torch"] = None
        section["torch_error"] = f"{type(error).__name__}: {error}"
    return section


def model_section(model: dict, config: WorkbenchConfig) -> dict:
    model_id = model["model_id"]
    requirements = model_requirements(model, config)
    blockers = preflight_blockers(model, config)
    state, reason = classify(blockers)
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else None
    patch = approved_patch_state(model, source)
    # A model may register several checkpoints under different training conditions, and the reference
    # has to match each one. Reporting a single pick would hide the case this whole registry exists for:
    # a paper row with no comparable weight, and a weight with no matching row.
    references = []
    for variant in model.get("checkpoint_variants") or []:
        try:
            reference, selection_reason = benchmark_references.select_reference(
                model_id,
                checkpoint_training_noise_pct=variant.get("checkpoint_training_noise_pct"),
                protocol_id=model.get("native_protocol"),
            )
        except (FileNotFoundError, ValueError) as error:
            reference, selection_reason = None, str(error)
        references.append({
            "checkpoint_id": variant.get("checkpoint_id"),
            "checkpoint_training_noise_pct": variant.get("checkpoint_training_noise_pct"),
            "checkpoint_mapping_status": variant.get("checkpoint_mapping_status"),
            "reference_id": reference.reference_id if reference else None,
            "reference_status": reference.reference_status if reference else None,
            "reason": selection_reason,
        })
    if not references:
        try:
            reference, selection_reason = benchmark_references.select_reference(model_id)
        except (FileNotFoundError, ValueError) as error:
            reference, selection_reason = None, str(error)
        references.append({
            "checkpoint_id": None, "checkpoint_training_noise_pct": None,
            "checkpoint_mapping_status": None,
            "reference_id": reference.reference_id if reference else None,
            "reference_status": reference.reference_status if reference else None,
            "reason": selection_reason,
        })

    if patch is not None and patch["applied"]:
        # A repaired checkout runs code the authors did not publish, so even a fully prepared model is
        # a declared variant rather than plain readiness.
        if state in (READY_FOR_FULL_EVALUATION, READY_FOR_GPU_SMOKE):
            state, reason = READY_WITH_DECLARED_VARIANT, (
                f"approved source patch applied: {patch['patch_file']}")
    return {
        "model_id": model_id,
        "state": state,
        "reason": reason,
        "blockers": blockers,
        "warnings": requirements["warnings"],
        "native_protocol": model.get("native_protocol"),
        "source_dir": model.get("source_dir"),
        "approved_patch": patch,
        "benchmark_references": references,
        "auxiliary_assets": [
            {"name": a.get("name"), "path": a.get("path"), "status": a.get("status")}
            for a in requirements["evidence"]["auxiliary_assets"]
        ],
    }


def document(config: WorkbenchConfig, models: list[dict]) -> dict:
    sections: dict = {
        "read_only": True,
        "repository": repository_section(),
        "config": config_section(config),
        "dataset": dataset_section(config),
        "papers": papers_section(),
        "gpu": gpu_section(),
        "intended_gpu": {
            "name": "NVIDIA RTX 5070 Ti",
            "compute_capability": "sm_120",
            "vram_gb": 16,
            "note": ("Blackwell needs a torch build whose arch list includes sm_120; a CUDA 12.x wheel "
                     "without it will raise at the first kernel launch rather than at import."),
        },
        "environment": environment_section(config, models),
    }
    models_section = {m["model_id"]: model_section(m, config) for m in models}

    # A model with no blocker still cannot be called ready without a GPU check, so `READY_*` states are
    # downgraded when this host cannot run the evaluator at all.
    gpu = sections["gpu"]
    if not gpu.get("cuda_available"):
        for entry in models_section.values():
            if entry["state"] in (READY_FOR_FULL_EVALUATION, READY_FOR_GPU_SMOKE):
                entry["preparation_state"] = entry["state"]
                entry["state"] = PC_GPU_VALIDATION_PENDING
                entry["reason"] = ("all reported prerequisites are satisfied, but no CUDA device is "
                                   "available on this host, so the run itself is unverified")
    sections["models"] = models_section
    sections["summary"] = _summary(models_section)
    return sections


def _summary(models_section: dict) -> dict:
    counts: dict[str, int] = {}
    for entry in models_section.values():
        counts[entry["state"]] = counts.get(entry["state"], 0) + 1
    return {"by_state": dict(sorted(counts.items())),
            "models": len(models_section)}


def render(document: dict) -> str:
    lines = ["PC release readiness", ""]
    repo = document["repository"]
    lines.append(f"repository   : {repo['branch']} @ {(repo['head'] or '?')[:12]} "
                 f"({repo['dirty_files']} changed file(s))")
    config = document["config"]
    lines.append(f"dataset root : {config['fashioniq_root']} "
                 f"({'present' if config['fashioniq_root_exists'] else 'MISSING'})")
    dataset = document["dataset"]
    if dataset["image_count"]:
        extensions = ", ".join(f"{k or 'no ext'} x{v}" for k, v in sorted(dataset["image_extensions"].items()))
        lines.append(f"images       : {dataset['image_count']} ({extensions})")
    lines.append(f"resized set  : {'present' if dataset['resized_image'] else 'absent'}")
    lines.append(f"corrections  : {len(dataset['correction_dicts'])} file(s)")
    papers = document["papers"]
    if papers.get("available"):
        lines.append(f"papers       : {papers['verified']}/{papers['references']} verified, "
                     f"version {papers['version']}")
    else:
        lines.append(f"papers       : UNAVAILABLE ({papers.get('error')})")
    gpu = document["gpu"]
    lines.append(f"gpu          : nvidia-smi={'yes' if gpu['nvidia_smi'] else 'no'} "
                 f"torch={gpu.get('torch') or 'absent'} cuda_available={gpu.get('cuda_available')}")
    for device in gpu.get("devices") or []:
        lines.append(f"               {device}")
    lines.append("")
    lines.append("models")
    for model_id, entry in sorted(document["models"].items()):
        lines.append(f"  {model_id:26} {entry['state']}")
        lines.append(f"      {entry['reason'][:150]}")
    lines.append("")
    lines.append(f"summary: {json.dumps(document['summary']['by_state'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=None,
                        help="check only this model id (repeatable)")
    parser.add_argument("--json", action="store_true", help="emit the machine-readable report")
    parser.add_argument("--dataset-root", type=Path, default=None,
                        help="canonical FashionIQ root, overriding the resolved configuration")
    args = parser.parse_args(argv)

    config = resolve_config()
    if args.dataset_root is not None:
        from dataclasses import replace
        config = replace(config, FASHIONIQ_ROOT=args.dataset_root.resolve())

    models = load_registry()["models"]
    if args.model:
        wanted = set(args.model)
        models = [m for m in models if m["model_id"] in wanted]
        missing = wanted - {m["model_id"] for m in models}
        if missing:
            parser.error(f"unknown model id(s): {sorted(missing)}")

    report = document(config, models)
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else render(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
