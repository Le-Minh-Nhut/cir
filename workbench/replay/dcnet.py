"""Preflight validation and replay runner for DCNet.

DCNet evaluation relies on the official CLI `test.py --resume <run_dir>`.
This module provides preflight verification of all required artifacts and identifies
the exact missing prerequisites (unpublished Google Drive checkpoint and unseeded
offline GloVe preprocessed pickles).
"""
from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.fashioniq_layout import CATEGORIES
from workbench.backend.registry import model_by_id
from workbench.replay.common import source_pin_status

MODEL = model_by_id("dcnet")
REQUIRED_RUN_FILES = ("config.json", "trained_model.pth")
DATASET_LABEL = "fashioniq"
PROTOCOL = "fashioniq_full_gallery_ref_excluded"


def validate_source(source_root: Path) -> list[str]:
    # The full model contract, so the declared `canonical_dataset_link` is authorized here too: the
    # standalone preflight has to accept the same sanctioned staging the orchestrator creates, and
    # without the contract a correctly staged link reads as an undeclared file.
    ok, reason = source_pin_status(source_root, MODEL["upstream_commit_sha"], MODEL)
    return [] if ok else [f"DCNet source pin invalid: {reason}"]


def configured_data_dir(config_file: Path, dataset_root: Path,
                        evaluator_cwd: Path | None = None) -> tuple[Path | None, list[str]]:
    """The dataset directory the evaluator will actually open, and why it may be untrustworthy.

    `test.py --resume <run_dir>` reads `<run_dir>/config.json`, and the native loaders take their
    path from `data_loader.args.data_dir` (`parse_config.py:28`, passed unmodified to
    `data_loader/CE_dataset.py:22,27,34`). The shipped configs set that field to the relative
    literal `dataset/fashioniq`, which `CE_dataset` joins against the *process* CWD. Nothing inside
    DCNet calls `chdir`, and `evaluate_models.py:594` launches the child with `cwd=plan.cwd`, which
    `source_cwd` resolves to the source checkout -- so a run can satisfy every other check and still
    read a different tree than the one whose provenance was verified.

    This returns the resolved directory and the blockers describing why it should not be trusted. A
    relative literal is NOT a refusal by itself: when the configured path is a symlink -- which is how
    the authorized preparation step stages the canonical dataset -- the literal resolves to the
    dataset root and the binding is exact. A plain directory at that path resolves inside the checkout
    and is refused, and so is an absolute literal naming another tree.

    The decision reads the author's field and nothing else. An environment override used to be
    accepted here, which made this function disagree with the evaluator it exists to predict: the
    native `ConfigParser` never looks at the environment, so a run could pass validation while
    `ExpertDataLoader`/`CE` opened the wrong root. A validator that answers a different question than
    the consumer is worse than no validator, so the override is gone and the field is the only input.
    """
    blockers: list[str] = []
    cwd = Path(evaluator_cwd) if evaluator_cwd is not None else Path(os.getcwd())
    # `is_file()` alone is not enough: a DIRECTORY at this path is not a config, and returning no
    # blockers for it would be a silent pass on a run that cannot start. The native
    # `ConfigParser` raises `IsADirectoryError` here, so the two must not disagree about whether
    # this file is readable.
    if config_file.exists() and not config_file.is_file():
        blockers.append(f"DCNet config is not a file: {config_file}")
        return None, blockers
    if not config_file.is_file() or config_file.stat().st_size == 0:
        return None, blockers  # already reported by validate_run_directory
    try:
        config = json.loads(config_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        blockers.append(f"DCNet config unreadable: {config_file}: {error}")
        return None, blockers

    declared = (config.get("data_loader") or {}).get("args", {}).get("data_dir")
    if not declared:
        blockers.append(
            f"DCNet config declares no data_loader.args.data_dir: {config_file}. The evaluator "
            "resolves its dataset from that field, so the tree it reads cannot be verified.")
        return None, blockers
    # The field is passed straight to `os.path.join` by the native loader, so anything but a string
    # is a config the evaluator itself cannot use. Reporting it beats raising a `TypeError` out of
    # a readiness view.
    if not isinstance(declared, str):
        blockers.append(
            f"DCNet config data_dir is not a path string: {config_file} declares "
            f"{type(declared).__name__} ({declared!r}), which the evaluator cannot join")
        return None, blockers

    relative = not Path(declared).is_absolute()
    resolved = (cwd / declared if relative else Path(declared)).resolve()
    # Compared in every case, including an absolute path: the field is the evaluator's own input, and
    # an absolute value that names a different tree is exactly the mismatch a `--dataset-root` on the
    # command line cannot repair.
    expected = Path(dataset_root).resolve()
    if resolved != expected:
        where = (f"DCNet joins that literal against the process working directory ({cwd})"
                 if relative else "that path is absolute, so the evaluator opens it directly")
        blockers.append(
            f"DCNet dataset binding mismatch: the evaluator resolves {declared!r} to {resolved}, not "
            f"the verified dataset root {expected}. {where} and opens <data_dir>/image_splits, "
            "<data_dir>/captions and <data_dir>/resized_images itself, so passing --dataset-root does "
            "not redirect it. Stage the canonical dataset at "
            f"{cwd / declared if relative else declared} as a symlink to {expected} (the declared "
            "`canonical_dataset_link` is what makes that staging authorized), or point the run "
            "directory's own config.json at the tree to be evaluated.")
    return resolved, blockers


def binding_mismatch(checkpoint_root: Path, dataset_root: Path,
                     evaluator_cwd: Path | None = None) -> str | None:
    """A refusal when the evaluator would read a tree other than *dataset_root*.

    Used by the operator entry points (`pc_readiness`, `doctor`, `prepare_model`,
    `evaluate_models`) because the official `DCNetAdapter` launches upstream `test.py` directly and
    therefore never reaches this module's own `main()`. The check has to run on the launch path, not
    only in the standalone preflight -- otherwise a helper nobody calls is all the protection a
    mismatch gets.
    """
    resolved, blockers = configured_data_dir(checkpoint_root / "config.json", dataset_root,
                                             evaluator_cwd)
    if resolved is None:
        return blockers[0] if blockers else None
    for blocker in blockers:
        if blocker.startswith("DCNet dataset binding mismatch"):
            return blocker
    return None


def validate_dataset_binding(checkpoint_root: Path, dataset_root: Path,
                             evaluator_cwd: Path | None = None) -> list[str]:
    return configured_data_dir(checkpoint_root / "config.json", dataset_root, evaluator_cwd)[1]


def validate_run_directory(checkpoint_root: Path) -> list[str]:
    blockers: list[str] = []
    if not checkpoint_root.is_dir():
        blockers.append(f"DCNet run directory missing: {checkpoint_root}")
        return blockers
    for filename in REQUIRED_RUN_FILES:
        target = checkpoint_root / filename
        if not target.is_file():
            blockers.append(f"DCNet run directory requirement missing: {target} (unpublished author checkpoint)")
        elif target.stat().st_size == 0:
            blockers.append(f"DCNet run directory artifact is empty placeholder: {target}")
    return blockers


def validate_dataset(dataset_root: Path, categories: tuple[str, ...] = CATEGORIES) -> list[str]:
    blockers: list[str] = []
    if not (dataset_root / "resized_images").is_dir():
        blockers.append(f"DCNet dataset requirement missing: {dataset_root / 'resized_images'} (expected flat resized_images/ directory)")

    for category in categories:
        split_path = dataset_root / "image_splits" / f"split.{category}.val.json"
        if not split_path.is_file():
            blockers.append(f"DCNet dataset requirement missing: {split_path}")

        glove_path = dataset_root / "captions" / f"cap.{category}.glove.val.pkl"
        if not glove_path.is_file():
            blockers.append(
                f"DCNet preprocessed GloVe feature missing: {glove_path} "
                "(unpublished artifact; raw reconstruction requires deterministic OOV variant)"
            )
    return blockers


def preflight(
    checkpoint_root: Path,
    dataset_root: Path,
    source_root: Path,
    categories: tuple[str, ...] = CATEGORIES,
    evaluator_cwd: Path | None = None,
) -> list[str]:
    # The evaluator runs with its CWD set by `source_cwd` (`evaluate_models.py:214-215`), which is
    # the checkout itself for DCNet. Defaulting to that rather than to the caller's CWD keeps the
    # standalone preflight and the launched run answering the same question.
    cwd = Path(evaluator_cwd) if evaluator_cwd is not None else source_root
    return (
        validate_source(source_root)
        + validate_run_directory(checkpoint_root)
        + validate_dataset_binding(checkpoint_root, dataset_root, cwd)
        + validate_dataset(dataset_root, categories)
    )


def main(argv: list[str] | None = None) -> int:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="Path to run directory containing config.json and trained_model.pth")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--category", choices=CATEGORIES)
    parser.add_argument("--preflight-only", action="store_true", default=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    categories = (args.category,) if args.category else CATEGORIES
    problems = preflight(args.checkpoint_root, args.dataset_root, args.source_root, categories)
    if problems:
        for problem in problems:
            print(f"[BLOCKED] {problem}", file=sys.stderr)
        return 1

    print("[READY] DCNet preflight passed; all run directory and dataset artifacts present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
