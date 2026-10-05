#!/usr/bin/env python3
"""Guard and run only audited official evaluation commands."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
from workbench.backend.registry import CHECKPOINT_ROOT, ROOT, checkpoint_path, load_registry


@dataclass(frozen=True)
class EvaluationPlan:
    model_id: str
    checkpoint_id: str
    cwd: Path
    command: list[str]


def command_is_audited(adapter_type: type[OfficialScriptAdapter]) -> bool:
    """An adapter is runnable only after it overrides the unaudited command stub."""
    return getattr(adapter_type, "command", None) is not OfficialScriptAdapter.command


def pinned_revision(source: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_cwd(model: dict, source: Path) -> Path:
    return source / "src" if model["model_id"] == "csmcir" else source


def candidates(args: argparse.Namespace, models: list[dict]) -> list[tuple[dict, dict]]:
    if args.model:
        models = [model for model in models if model["model_id"] == args.model]
        if not models:
            raise ValueError(f"unknown model: {args.model}")
    selected: list[tuple[dict, dict]] = []
    for model in models:
        checkpoints = model["checkpoint_variants"]
        if args.checkpoint:
            checkpoints = [checkpoint for checkpoint in checkpoints if checkpoint["checkpoint_id"] == args.checkpoint]
            if not checkpoints and args.model:
                raise ValueError(f"unknown checkpoint for {args.model}: {args.checkpoint}")
        selected.extend((model, checkpoint) for checkpoint in checkpoints)
    if args.checkpoint and not selected:
        raise ValueError(f"unknown checkpoint: {args.checkpoint}")
    return selected


def list_candidates(items: Iterable[tuple[dict, dict]]) -> None:
    for model, checkpoint in items:
        adapter_type = ADAPTERS.get(model["model_id"])
        readiness = "audited" if adapter_type and command_is_audited(adapter_type) else "not audited"
        print(
            f"[OK] {model['model_id']} / {checkpoint['checkpoint_id']}: "
            f"protocol={model['native_protocol'] or 'unavailable'} command={readiness}"
        )


def blockers(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None) -> tuple[Path, Path, list[str]]:
    source = ROOT / "third_party" / model["source_dir"] if model.get("source_dir") else ROOT / "third_party"
    checkpoint_file = checkpoint_path(model["model_id"], checkpoint, CHECKPOINT_ROOT)
    blocked: list[str] = []
    if not model.get("source_available") or not model.get("source_dir"):
        blocked.append("upstream source unavailable")
    elif not source.is_dir():
        blocked.append(f"source missing: {source}")
    else:
        expected_pin = model.get("upstream_commit_sha")
        if expected_pin:
            actual_pin = pinned_revision(source)
            if actual_pin is None:
                blocked.append(f"source pin unavailable: {source}")
            elif actual_pin != expected_pin:
                blocked.append(f"source pin mismatch: expected {expected_pin}, found {actual_pin}")
    if not checkpoint_file.is_file():
        blocked.append(f"checkpoint missing: {checkpoint_file}")
    if checkpoint["checkpoint_mapping_status"] == "UNVERIFIED":
        blocked.append("checkpoint mapping unresolved")
    if protocol not in model["supported_protocols"]:
        blocked.append(f"protocol incompatible: {protocol}")
    if dataset_root is None:
        blocked.append("dataset root not provided")
    elif not dataset_root.is_dir():
        blocked.append(f"dataset root missing: {dataset_root}")
    if model["model_id"] == "encoder" and not (source / "open_clip_pytorch_model.bin").is_file():
        blocked.append(f"ENCODER asset missing: {source / 'open_clip_pytorch_model.bin'}")
    if model["model_id"] == "csmcir":
        layout = source / "fashionIQ_dataset"
        required_layout = (
            layout / "captions",
            layout / "image_splits",
            layout / "images",
            source / "COT_ours2" / "bert_captions" / "fashioniq",
            source / "COT_ours2" / "fashioniq",
        )
        missing_layout = [path for path in required_layout if not path.is_dir()]
        if missing_layout:
            blocked.append(f"CSMCIR fixed dataset layout missing: {missing_layout[0]}")
        elif dataset_root is not None and dataset_root.resolve() != layout.resolve():
            blocked.append(f"CSMCIR requires --dataset-root {layout}")
    return source, checkpoint_file, blocked


def guarded_plan(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None, top_k: int) -> tuple[EvaluationPlan | None, list[str]]:
    adapter_type = ADAPTERS.get(model["model_id"])
    if adapter_type is None or not command_is_audited(adapter_type):
        return None, ["SKIPPED: adapter command not audited"]
    source, _, blocked = blockers(model, checkpoint, protocol, dataset_root)
    script = getattr(adapter_type, "script", None)
    if source.is_dir() and script and not (source / script).is_file():
        blocked.append(f"official evaluator missing: {source / script}")
    if blocked:
        return None, blocked
    assert dataset_root is not None
    request = EvalRequest(
        model_id=model["model_id"],
        checkpoint_id=checkpoint["checkpoint_id"],
        protocol_id=protocol,
        dataset_root=dataset_root,
        output_path=ROOT / "artifacts" / "results" / protocol / model["model_id"] / f"{checkpoint['checkpoint_id']}.json",
        top_k=top_k,
    )
    try:
        command = adapter_type().build_command(request)
    except Exception as error:
        return None, [f"official command construction failed: {error}"]
    return EvaluationPlan(model["model_id"], checkpoint["checkpoint_id"], source_cwd(model, source), command), []


def parser_for(registry: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true", help="list registry candidates and command-audit state")
    selector.add_argument("--model", help="select one registry model")
    selector.add_argument("--all-runnable", action="store_true", help="guard every registry candidate for official execution")
    parser.add_argument("--checkpoint", help="select one checkpoint ID")
    parser.add_argument("--protocol", choices=sorted(registry["protocols"]), help="protocol; defaults to each model native protocol")
    parser.add_argument("--dataset-root", type=Path, help="FashionIQ root; CSMCIR requires its fixed upstream layout")
    parser.add_argument("--top-k", type=int, default=200, help="requested saved retrieval depth after verified reproduction")
    parser.add_argument("--dry-run", action="store_true", help="print guarded official commands without executing them")
    parser.add_argument("--continue-on-error", action="store_true", help="continue remaining commands after an official command fails")
    return parser


def main(argv: list[str] | None = None) -> int:
    registry = load_registry()
    parser = parser_for(registry)
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    try:
        items = candidates(args, registry["models"])
    except ValueError as error:
        parser.error(str(error))
    if args.list:
        list_candidates(items)
        return 0

    failed = False
    for model, checkpoint in items:
        protocol = args.protocol or model["native_protocol"]
        print(f"[RUN] {model['model_id']} / {checkpoint['checkpoint_id']}")
        if protocol is None:
            print("[BLOCKED] protocol unavailable")
            failed = True
            continue
        plan, reasons = guarded_plan(model, checkpoint, protocol, args.dataset_root, args.top_k)
        if plan is None:
            for reason in reasons:
                tag = "[SKIP]" if reason.startswith("SKIPPED:") else "[BLOCKED]"
                print(f"{tag} {reason}")
            failed = failed or any(not reason.startswith("SKIPPED:") for reason in reasons)
            continue
        print(f"[OK] protocol: {protocol}")
        print(f"[RUN] cwd: {plan.cwd}")
        print(f"[RUN] command: {shlex.join(plan.command)}")
        if args.dry_run:
            print("[OK] dry-run: official command not executed")
            continue
        try:
            subprocess.run(plan.command, cwd=plan.cwd, check=True)
        except subprocess.CalledProcessError as error:
            print(f"[BLOCKED] official command failed: {error}")
            failed = True
            if not args.continue_on_error:
                break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
