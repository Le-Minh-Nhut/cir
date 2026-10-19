#!/usr/bin/env python3
"""Validate a local FashionIQ layout and safely prepare CSMCIR's fixed link."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.fashioniq_layout import missing_paths, standard_paths



def required_layout(root: Path) -> tuple[Path, ...]:
    return standard_paths(root)


def validate_fashioniq(root: Path) -> list[str]:
    if not root.is_dir():
        return [f"dataset root missing: {root}"]
    return [f"required path missing: {path}" for path in missing_paths(required_layout(root))]



def prepare_canonical_link(root: Path, source: Path, link_name: str, *, check_only: bool,
                           dry_run: bool) -> tuple[bool, list[str]]:
    """Stage (or verify) a pinned evaluator's hard-coded dataset root.

    Some official evaluators have no dataset-root option: they derive the root from their own file
    location and read `<checkout>/<link_name>`. That path is therefore part of the source contract,
    not a caller choice, and it is created only when the canonical root is verified, the source
    checkout is present, the destination is unoccupied, and the operator authorized the step.

    Returns ``(ok, messages)``. Never overwrites a file or a differently-targeted link.
    """
    messages: list[str] = []
    destination = source / link_name
    if not source.is_dir():
        return False, [f"[BLOCKED] source checkout missing: {source}"]

    if destination.is_symlink():
        try:
            linked_root = destination.resolve()
        except (OSError, RuntimeError) as error:
            return False, [f"[BLOCKED] dataset link cannot resolve: {destination}: {error}"]
        if linked_root != root:
            return False, [f"[BLOCKED] dataset link resolves to {linked_root}, expected {root}"]
        messages.append(f"[OK] dataset link already correct: {destination} -> {root}")
        return True, messages

    if os.path.lexists(destination):
        # A real directory or file here is either a manual staging or a mistake: never destroy it.
        return False, [f"[BLOCKED] dataset destination already exists and is not the expected link: {destination}"]

    if check_only:
        messages.append(f"[WARN] dataset link absent: {destination} (check-only)")
        return True, messages
    if dry_run:
        messages.append(f"[OK] dry-run: would link {destination} -> {root}")
        return True, messages

    # A declared link may be nested (`dataset/fashioniq` is what DCNet's own config resolves), and a
    # fresh checkout has no `dataset/` directory. The parent is part of the artifact being staged, so
    # it is created here -- after every refusal above has had its say -- and never anywhere else, so
    # check-only and dry-run stay write-free and an unexpected occupant is still refused.
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return False, [f"[BLOCKED] cannot create the link's parent directory "
                       f"{destination.parent}: {error}"]

    destination.symlink_to(root, target_is_directory=True)
    messages.append(f"[OK] dataset link created: {destination} -> {root}")
    return True, messages


def models_requiring_link() -> dict[str, str]:
    """model_id -> declared in-checkout dataset link, from the registry."""
    from workbench.backend.registry import load_registry

    return {
        model["model_id"]: model["canonical_dataset_link"]
        for model in load_registry()["models"]
        if model.get("canonical_dataset_link")
    }


def parser_for() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ dataset root; defaults to configured FASHIONIQ_ROOT")
    parser.add_argument("--check-only", action="store_true", help="validate only; never create a link")
    parser.add_argument("--model", help="prepare the fixed dataset link a model's pinned evaluator requires")
    parser.add_argument("--dry-run", action="store_true", help="show link creation without changing files")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser_for().parse_args(argv)
    config = resolve_config()
    dataset_root = args.dataset_root or config.FASHIONIQ_ROOT
    try:
        root = dataset_root.expanduser().resolve()
    except (OSError, RuntimeError) as error:
        print(f"[BLOCKED] dataset root cannot resolve: {dataset_root}: {error}")
        return 1
    missing = validate_fashioniq(root)
    if missing:
        for message in missing:
            print(f"[BLOCKED] {message}")
        return 1
    print(f"[OK] FashionIQ base layout valid: {root}")

    links = models_requiring_link()
    if not args.model:
        if links:
            print(f"[WARN] no model selected: check only; {len(links)} model(s) declare a "
                  f"canonical dataset link")
        else:
            print("[WARN] no model selected: check only; layout preparation not verified")
        return 0
    if args.model not in links:
        print(f"[WARN] {args.model}: no declared fixed dataset link; layout preparation not verified")
        return 0

    from workbench.backend.registry import model_by_id

    try:
        model = model_by_id(args.model)
    except KeyError:
        print(f"[BLOCKED] unknown model: {args.model}")
        return 1
    source = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    ok, messages = prepare_canonical_link(root, source, links[args.model],
                                          check_only=args.check_only, dry_run=args.dry_run)
    for message in messages:
        print(message)
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
