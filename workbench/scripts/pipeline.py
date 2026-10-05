#!/usr/bin/env python3
"""Orchestrate existing workbench operator scripts without reimplementing them."""
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import WorkbenchConfig, resolve_config

SCRIPTS = Path(__file__).resolve().parent
REPOSITORY = SCRIPTS.parents[1]


@dataclass(frozen=True)
class Stage:
    name: str
    commands: tuple[list[str], ...] = ()
    skipped: str | None = None


def command(script: str, *arguments: str | Path) -> list[str]:
    return [sys.executable, str(SCRIPTS / script), *(str(argument) for argument in arguments)]


def run_stage(name: str, *commands: list[str]) -> Stage:
    return Stage(name, commands)


def selection(args: argparse.Namespace, *, runnable: bool = False) -> list[str]:
    return ["--model", args.model] if args.model else (["--all-runnable"] if runnable else ["--all"])


def dataset_root(args: argparse.Namespace, config: WorkbenchConfig) -> Path:
    return args.dataset_root or config.FASHIONIQ_ROOT


def dataset_check(args: argparse.Namespace, config: WorkbenchConfig) -> Stage:
    return run_stage("dataset", command("prepare_dataset.py", "--dataset-root", dataset_root(args, config), "--check-only"))


def layout_prepare(args: argparse.Namespace, config: WorkbenchConfig, *, check_only: bool) -> Stage:
    if args.model != "csmcir":
        return Stage("layout", skipped="only audited CSMCIR layout preparation is available")
    arguments: list[str | Path] = ["--dataset-root", dataset_root(args, config), "--model", "csmcir"]
    if check_only:
        arguments.append("--check-only")
    return run_stage("layout", command("prepare_dataset.py", *arguments))


def real_stages(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    sync = (
        run_stage("sync", command("sync_upstreams.py", *selection(args), "--output-root", config.WORKBENCH_THIRD_PARTY_ROOT))
        if args.sync_sources
        else Stage("sync", skipped="pass --sync-sources")
    )
    download_args = [*selection(args), *( ["--checkpoint", args.checkpoint] if args.checkpoint else []), "--output-root", config.WORKBENCH_CHECKPOINT_ROOT]
    checkpoints = run_stage("checkpoint", command("download_checkpoints.py", *download_args)) if args.download_checkpoints else Stage("checkpoint", skipped="pass --download-checkpoints")
    auxiliary = (
        run_stage("auxiliary-assets", command("download_auxiliary_assets.py", "--model", "csmcir"))
        if args.download_auxiliary_assets
        else Stage("auxiliary-assets", skipped="pass --download-auxiliary-assets")
    )
    if args.evaluate:
        evaluation_args = [*selection(args, runnable=True), "--dataset-root", str(dataset_root(args, config))]
        if args.checkpoint:
            evaluation_args.extend(["--checkpoint", args.checkpoint])
        if args.protocol:
            evaluation_args.extend(["--protocol", args.protocol])
        evaluation_args.extend(["--top-k", str(args.top_k)])
        if args.continue_on_error:
            evaluation_args.append("--continue-on-error")
        evaluation = run_stage("evaluation", command("evaluate_models.py", *evaluation_args))
    else:
        evaluation = Stage("evaluation", skipped="pass --evaluate")
    validation_index = (
        run_stage(
            "validation-index",
            command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT, "--strict-real"),
            command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT),
        )
        if args.evaluate or args.rebuild_index
        else Stage("validation-index", skipped="pass --evaluate or --rebuild-index")
    )
    serve = run_stage("serve", command("serve_workbench.py")) if args.serve else Stage("serve", skipped="pass --serve")
    doctor_args: list[str] = ["--scope", "real"]
    if args.model:
        doctor_args.extend(["--model", args.model])
    if args.serve:
        doctor_args.append("--serve")
    return [
        run_stage("doctor", command("doctor.py", *doctor_args)),
        dataset_check(args, config),
        sync,
        layout_prepare(args, config, check_only=True),
        checkpoints,
        auxiliary,
        evaluation,
        validation_index,
        serve,
    ]


def stages_for(args: argparse.Namespace, config: WorkbenchConfig) -> list[Stage]:
    if args.mode == "mock":
        doctor_args = ["--scope", "workbench", *( ["--serve"] if args.serve else [])]
        stages = [
            run_stage("doctor", command("doctor.py", *doctor_args)),
            run_stage("load mock", command("load_mock_results.py", "--output-root", config.WORKBENCH_RESULTS_ROOT)),
            run_stage("validate", command("validate_results.py", "--root", config.WORKBENCH_RESULTS_ROOT)),
            run_stage("rebuild", command("rebuild_index.py", "--results-root", config.WORKBENCH_RESULTS_ROOT)),
        ]
        return [*stages, run_stage("serve", command("serve_workbench.py")) if args.serve else Stage("serve", skipped="pass --serve")]
    if args.mode == "prepare":
        return [dataset_check(args, config), layout_prepare(args, config, check_only=False)]
    if args.mode == "real":
        return real_stages(args, config)
    if args.mode == "serve":
        return [run_stage("serve", command("serve_workbench.py"))]
    return [run_stage("doctor", command("doctor.py", *( ["--model", args.model] if args.model else [])))]


def run_command(argv: list[str]) -> int:
    try:
        return subprocess.run(argv, cwd=REPOSITORY, check=False).returncode
    except OSError as error:
        print(f"[BLOCKED] cannot start {argv[1]}: {error}", file=sys.stderr)
        return 127


def run_stages(stages: list[Stage], *, dry_run: bool, continue_on_error: bool) -> int:
    failed = False
    for number, stage in enumerate(stages, 1):
        print(f"[{number}/{len(stages)}] {stage.name}")
        if not stage.commands:
            print(f"  [SKIP] {stage.skipped}")
            continue
        for argv in stage.commands:
            print(f"  [{'PLAN' if dry_run else 'RUN'}] {shlex.join(argv)}")
            if stage.name == "evaluation":
                print("  [INFO] official reproduction only; no instrumented ranking export")
            if dry_run:
                continue
            status = run_command(argv)
            if status == 0:
                continue
            failed = True
            print(f"  [BLOCKED] {stage.name} exited with status {status}", file=sys.stderr)
            if not continue_on_error:
                return 1
    return 1 if failed else 0


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("mock", "prepare", "real", "serve", "status"))
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ dataset root")
    parser.add_argument("--sync-sources", action="store_true", help="sync selected official source repositories")
    parser.add_argument("--download-checkpoints", action="store_true", help="download selected registry checkpoints")
    parser.add_argument("--download-auxiliary-assets", action="store_true", help="obtain selected model auxiliary evaluation assets")
    parser.add_argument("--evaluate", action="store_true", help="run guarded official reproduction commands")
    parser.add_argument("--rebuild-index", action="store_true", help="validate and rebuild the derived result index")
    parser.add_argument("--serve", action="store_true", help="serve after pipeline stages")
    parser.add_argument("--dry-run", action="store_true", help="print ordered stage plan without subprocesses or mutations")
    parser.add_argument("--continue-on-error", action="store_true", help="continue pipeline after a failed stage")
    parser.add_argument("--model", help="registry model selection")
    parser.add_argument("--protocol", help="official protocol selection for evaluation")
    parser.add_argument("--checkpoint", help="checkpoint selection")
    parser.add_argument("--top-k", type=int, default=200, help="requested saved retrieval depth")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    if args.top_k < 1:
        parser_for().error("--top-k must be positive")
    config = resolve_config()
    if args.mode == "prepare" and args.dataset_root is None and not config.FASHIONIQ_ROOT:
        parser_for().error("prepare requires --dataset-root or FASHIONIQ_ROOT")
    return run_stages(stages_for(args, config), dry_run=args.dry_run, continue_on_error=args.continue_on_error)


if __name__ == "__main__":
    raise SystemExit(main())
