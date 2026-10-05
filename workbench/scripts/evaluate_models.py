#!/usr/bin/env python3
"""Guard and run audited official evaluation commands."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import threading
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.adapters.base import EvalRequest
from workbench.backend.adapters.models import ADAPTERS, OfficialScriptAdapter
from workbench.backend.operator_config import WorkbenchConfig, resolve_config
from workbench.backend.registry import checkpoint_path, load_registry, sha256_file


@dataclass(frozen=True)
class EvaluationPlan:
    model_id: str
    checkpoint_id: str
    protocol: str
    dataset_root: Path
    source: Path
    checkpoint: Path
    cwd: Path
    command: list[str]
    pin: str | None
    actual_source_commit: str | None


def command_is_audited(adapter_type: type[OfficialScriptAdapter]) -> bool:
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
        print(f"[OK] {model['model_id']} / {checkpoint['checkpoint_id']}: protocol={model['native_protocol'] or 'unavailable'} command={readiness}")


def blockers(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None, config: WorkbenchConfig) -> tuple[Path, Path, str | None, list[str]]:
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"] if model.get("source_dir") else config.WORKBENCH_THIRD_PARTY_ROOT
    checkpoint_file = checkpoint_path(model["model_id"], checkpoint, config.WORKBENCH_CHECKPOINT_ROOT)
    actual_pin = None
    blocked: list[str] = []
    if not model.get("source_available") or not model.get("source_dir"):
        blocked.append("upstream source unavailable")
    elif not source.is_dir():
        blocked.append(f"source missing: {source}")
    else:
        expected_pin = model.get("upstream_commit_sha")
        actual_pin = pinned_revision(source)
        if expected_pin and actual_pin is None:
            blocked.append(f"source pin unavailable: {source}")
        elif expected_pin and actual_pin != expected_pin:
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
        if not layout.is_symlink() or layout.resolve() != config.FASHIONIQ_ROOT.resolve():
            blocked.append(f"CSMCIR requires canonical dataset link: {layout} -> {config.FASHIONIQ_ROOT}")
        required = [layout / name for name in ("captions", "image_splits", "images")]
        required.extend(source / "COT_ours2" / "fashioniq" / f"{category}_cot_val.json" for category in ("dress", "shirt", "toptee"))
        required.extend(layout / "qwen_captions" / f"{category}_cot_val.json" for category in ("dress", "shirt", "toptee"))
        missing = next((path for path in required if not path.exists()), None)
        if missing:
            blocked.append(f"CSMCIR fixed dataset layout missing: {missing}")
        elif dataset_root is not None and dataset_root.resolve() != layout.resolve():
            blocked.append(f"CSMCIR requires --dataset-root {layout}")
    return source, checkpoint_file, actual_pin, blocked


def guarded_plan(model: dict, checkpoint: dict, protocol: str, dataset_root: Path | None, top_k: int, config: WorkbenchConfig | None = None) -> tuple[EvaluationPlan | None, list[str]]:
    config = config or resolve_config()
    adapter_type = ADAPTERS.get(model["model_id"])
    if adapter_type is None or not command_is_audited(adapter_type):
        return None, ["SKIPPED: adapter command not audited"]
    source, checkpoint_file, actual_pin, blocked = blockers(model, checkpoint, protocol, dataset_root, config)
    script = getattr(adapter_type, "script", None)
    if source.is_dir() and script and not (source / script).is_file():
        blocked.append(f"official evaluator missing: {source / script}")
    if blocked:
        return None, blocked
    assert dataset_root is not None
    request = EvalRequest(model_id=model["model_id"], checkpoint_id=checkpoint["checkpoint_id"], protocol_id=protocol, dataset_root=dataset_root, output_path=config.WORKBENCH_RESULTS_ROOT / protocol / model["model_id"] / f"{checkpoint['checkpoint_id']}.json", top_k=top_k)
    try:
        command = adapter_type().command(source, checkpoint_file, request)
    except Exception as error:
        return None, [f"official command construction failed: {error}"]
    return EvaluationPlan(model["model_id"], checkpoint["checkpoint_id"], protocol, dataset_root, source, checkpoint_file, source_cwd(model, source), command, model.get("upstream_commit_sha"), actual_pin), []


def parser_for(registry: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--list", action="store_true", help="list registry candidates and command-audit state")
    selector.add_argument("--model", help="select one registry model")
    selector.add_argument("--all-runnable", action="store_true", help="guard every registry candidate for official execution")
    parser.add_argument("--checkpoint", help="select one checkpoint ID")
    parser.add_argument("--protocol", choices=sorted(registry["protocols"]), help="protocol; defaults to each model native protocol")
    parser.add_argument("--dataset-root", type=Path, help="FashionIQ root; CSMCIR requires its fixed upstream layout")
    parser.add_argument("--canonical-dataset-root", type=Path, help="canonical FashionIQ root behind a CSMCIR fixed link")
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true", help="print guarded command and log plan without writing")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser


def log_dir(config: WorkbenchConfig, plan: EvaluationPlan, timestamp: datetime) -> Path:
    return config.CIR_REPO_ROOT / "workbench" / "artifacts" / "logs" / f"{timestamp:%Y%m%dT%H%M%SZ}_{plan.model_id}_{plan.checkpoint_id}"


def execute(plan: EvaluationPlan, config: WorkbenchConfig) -> int:
    started = datetime.now(UTC)
    directory = log_dir(config, plan, started)
    directory.mkdir(parents=True, exist_ok=False)
    metadata = {
        "model_id": plan.model_id,
        "checkpoint_id": plan.checkpoint_id,
        "protocol_id": plan.protocol,
        "argv": plan.command,
        "cwd": str(plan.cwd),
        "upstream_expected_pin": plan.pin,
        "actual_source_commit": plan.actual_source_commit,
        "checkpoint_local_sha256": sha256_file(plan.checkpoint),
        "started_at": started.isoformat(),
        "finished_at": None,
        "return_code": None,
        "dataset_root": str(plan.dataset_root),
    }
    (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    with (directory / "stdout.log").open("w", encoding="utf-8") as stdout, (directory / "stderr.log").open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(plan.command, cwd=plan.cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        def tee(stream, destination, terminal):
            for line in stream:
                destination.write(line)
                destination.flush()
                terminal.write(line)
                terminal.flush()

        threads = [threading.Thread(target=tee, args=(process.stdout, stdout, sys.stdout)), threading.Thread(target=tee, args=(process.stderr, stderr, sys.stderr))]
        for thread in threads:
            thread.start()
        code = process.wait()
        for thread in threads:
            thread.join()
    metadata["finished_at"] = datetime.now(UTC).isoformat()
    metadata["return_code"] = code
    (directory / "command.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return code


def main(argv: list[str] | None = None) -> int:
    registry = load_registry()
    args = parser_for(registry).parse_args(argv)
    if args.top_k < 1:
        raise SystemExit("--top-k must be positive")
    try:
        items = candidates(args, registry["models"])
    except ValueError as error:
        raise SystemExit(str(error))
    if args.list:
        list_candidates(items)
        return 0
    config = resolve_config()
    if args.canonical_dataset_root:
        config = replace(config, FASHIONIQ_ROOT=args.canonical_dataset_root.resolve())
    dataset_root = args.dataset_root or config.FASHIONIQ_ROOT
    failed = False
    for model, checkpoint in items:
        protocol = args.protocol or model["native_protocol"]
        print(f"[RUN] {model['model_id']} / {checkpoint['checkpoint_id']}")
        if protocol is None:
            print("[BLOCKED] protocol unavailable"); failed = True; continue
        plan, reasons = guarded_plan(model, checkpoint, protocol, dataset_root, args.top_k, config)
        if plan is None:
            for reason in reasons: print(f"{'[SKIP]' if reason.startswith('SKIPPED:') else '[BLOCKED]'} {reason}")
            failed = failed or any(not reason.startswith("SKIPPED:") for reason in reasons)
            continue
        planned_logs = log_dir(config, plan, datetime.now(UTC))
        print(f"[OK] protocol: {protocol}")
        print(f"[RUN] cwd: {plan.cwd}")
        print(f"[RUN] command: {shlex.join(plan.command)}")
        print(f"[RUN] log directory: {planned_logs}")
        print(f"[RUN] provenance: {planned_logs / 'command.json'}")
        if args.dry_run:
            print("[OK] dry-run: official command not executed; no logs created")
            continue
        if execute(plan, config):
            print("[BLOCKED] official command failed"); failed = True
            if not args.continue_on_error: break
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
