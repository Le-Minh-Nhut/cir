#!/usr/bin/env python3
"""Inspect model preparation status. Never executes preparation scripts without explicit authorization.

Default: read-only status from preparation_contracts.yaml. Use --execute --allow-preparation
to run only the one verified automated CSMCIR dataset link step. All other preparation is manual.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.preflight import model_requirements, preflight_blockers
from workbench.backend.registry import load_registry, preparation_contract_for_model

def status(model: dict, config) -> dict:
    blockers = preflight_blockers(model, config)
    model_id = model["model_id"]
    contract = preparation_contract_for_model(model)
    requirements = model_requirements(model, config)
    evidence = requirements["evidence"]
    raw_missing = evidence["raw_inputs_missing"]
    generated_missing = evidence["generated_artifacts_missing"]
    result = {
        "model_id": model_id,
        "preparation_id": contract["preparation_id"] if contract else None,
        "contract_type": contract.get("source_family") if contract else "standard_fashioniq",
        "automation_policy": contract["automation_policy"] if contract else "NONE_REQUIRED",
        "deterministic": contract["deterministic_status"] if contract else "N/A",
        "raw_inputs_missing": raw_missing,
        "generated_artifacts_missing": generated_missing,
        "external_assets": contract["external_assets"] if contract else [],
        "fully_prepared": not requirements["blocks"],
        "blockers": blockers,
        "notes": contract["notes"] if contract else "Standard FashionIQ layout required; no custom generated artifacts.",
    }
    if evidence["image_extension_problems"]:
        result["image_extension_problems"] = evidence["image_extension_problems"]
    if evidence["dataset_link"]:
        result["dataset_link"] = evidence["dataset_link"]
    return result


def print_status(s: dict) -> None:
    label = "OK" if s["fully_prepared"] else ("BLOCKED" if s["raw_inputs_missing"] else "MANUAL")
    print(f"[{label}] {s['model_id']}: policy={s['automation_policy']} deterministic={s['deterministic']}")
    if s["raw_inputs_missing"]:
        print(f"  raw inputs missing ({len(s['raw_inputs_missing'])}): {s['raw_inputs_missing'][0]}")
    if s["generated_artifacts_missing"]:
        print(f"  generated artifacts missing ({len(s['generated_artifacts_missing'])}): {s['generated_artifacts_missing'][0]}")
    if s["external_assets"]:
        blocked_ext = [a for a in s["external_assets"] if a.get("required_for_evaluation") and not a.get("optional")]
        if blocked_ext:
            print(f"  required external assets: {', '.join(a['name'] for a in blocked_ext)}")
    if s["notes"]:
        print(f"  notes: {s['notes']}")


#: Marker used by a preparation contract to say "copy this file out of the pinned checkout".
BUNDLED_ASSET_GENERATOR = "pinned_source_file"
#: Directory inside a pinned checkout that ships such files.
BUNDLED_ASSET_DIRECTORY = "correct_files"
#: Marker used by a preparation contract for the per-category image resize.
RESIZE_GENERATOR = "fashioniq_resize_images"
#: Name of the receipt a completed resize leaves behind; the pipeline validates it as the output.
RESIZE_RECEIPT_NAME = ".workbench_resize_receipt.json"


def resize_receipt_path(config, layout: str) -> Path:
    """Where a completed resize records its coverage. Only a fully covered run writes one."""
    return config.FASHIONIQ_ROOT / ("resized_images" if layout == "flat" else "resized_image") / RESIZE_RECEIPT_NAME


def write_resize_receipt(receipt: Path, *, model_id: str, layout: str, image_size: int,
                         extension: str, source_script: str, coverage: dict, written: int) -> None:
    """Write the coverage receipt atomically, so a crash never leaves a half-written one."""
    receipt.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "workbench_resize_receipt",
        "model_id": model_id,
        "layout": layout,
        "image_size": image_size,
        "extension": extension,
        "source_script": source_script,
        "categories": coverage,
        "written_now": written,
    }
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)


def resize_is_complete(config, model: dict, contract: dict) -> bool | None:
    """Whether the declared resize is finished, judged from its receipt and the files it names.

    Returns None when the model declares no resize. A receipt alone is not enough: the files it
    promises may have been removed afterwards, which is exactly the case a resume would otherwise
    skip over.
    """
    recipe = next((a for a in contract.get("generated_artifacts", [])
                   if a.get("generator") == RESIZE_GENERATOR), None)
    if recipe is None:
        return None
    layout = (recipe.get("parameters") or {}).get("layout")
    if layout not in ("per_category", "flat"):
        return False
    receipt = resize_receipt_path(config, layout)
    if not receipt.is_file():
        return False
    try:
        declared = json.loads(receipt.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    categories = declared.get("categories")
    if not isinstance(categories, dict) or not categories:
        return False
    base = config.FASHIONIQ_ROOT / ("resized_images" if layout == "flat" else "resized_image")
    for category, counts in categories.items():
        if not isinstance(counts, dict) or counts.get("expected") != counts.get("present"):
            return False
        target = base / (category if layout == "per_category" else "")
        if not target.is_dir():
            return False
        present = len([p for p in target.iterdir() if p.is_file() and p.suffix == ".jpg"])
        if present < int(counts.get("present") or 0):
            return False
    return True


def _fixed_link(config, model_id: str, dry_run: bool, check_only: bool) -> bool:
    """Stage the model's declared fixed dataset link through the shared helper.

    Any model whose pinned evaluator hard-codes its dataset root declares
    `canonical_dataset_link`, and `prepare_dataset.py` implements the staging. Delegating keeps
    one implementation of link creation and verification instead of a second one here.
    """
    scripts = Path(__file__).resolve().parent
    argv = [sys.executable, str(scripts / "prepare_dataset.py"),
            "--dataset-root", str(config.FASHIONIQ_ROOT), "--model", model_id]
    if dry_run:
        argv.append("--dry-run")
    elif check_only:
        argv.append("--check-only")
    print(f"[RUN] {model_id} dataset-link preparation: {' '.join(argv)}")
    result = subprocess.run(argv)
    return result.returncode == 0


def _stage_source_assets(config, model: dict, contract: dict, *, dry_run: bool) -> bool:
    """Install assets the pinned source ships but the evaluator reads from the dataset root.

    Some evaluators read a correction dictionary from `<dataset>/captions/` while the author ships
    it inside the repository under `correct_files/`. The file, its path and its consumer are all
    source-backed, so this is a copy the workbench can make; it is never a substitution, and a
    destination that already holds different bytes is reported rather than overwritten.
    """
    import shutil

    from workbench.backend.registry import load_registry as _load_registry

    model_id = model["model_id"]
    source = config.WORKBENCH_THIRD_PARTY_ROOT / (model.get("source_dir") or "")
    if not source.is_dir():
        return True  # the source is the sync stage's responsibility, not preparation's

    staged = 0
    staged, absent = 0, []
    for artifact in contract.get("generated_artifacts", []):
        if artifact.get("generator") != BUNDLED_ASSET_GENERATOR:
            continue
        # The directory a repository keeps its bundled assets in is a fact about that repository,
        # so the contract names it: LIMN ships `correct_files/`, TGCIR ships `correction_dict/`.
        # Assuming one of them made the other silently stage nothing and still report success.
        directory = artifact.get("source_directory") or BUNDLED_ASSET_DIRECTORY
        for relative in artifact["paths"]:
            destination = config.FASHIONIQ_ROOT / relative
            shipped = source / directory / Path(relative).name
            if not shipped.is_file():
                absent.append(str(shipped))
                continue
            if destination.is_file():
                if destination.read_bytes() == shipped.read_bytes():
                    continue
                print(f"[BLOCKED] {model_id}: {destination} exists with different contents than the "
                      f"pinned {shipped}; refusing to overwrite", file=sys.stderr)
                return False
            if dry_run:
                print(f"[OK] dry-run: would copy {shipped} -> {destination}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(shipped, destination)
            print(f"[OK] {model_id}: staged {destination} from the pinned source")
            staged += 1
    # A declared bundled asset that the pinned checkout does not actually contain is reported: the
    # contract says the file exists, so its absence is a fact about the checkout, not a success.
    if absent:
        print(f"[BLOCKED] {model_id}: declared source-shipped asset(s) not present in the pinned "
              f"checkout: {', '.join(absent)}", file=sys.stderr)
        return False
    if staged:
        print(f"[OK] {model_id}: {staged} source-shipped asset(s) staged")
    return True


def _run_resize(config, model: dict, contract: dict, *, dry_run: bool) -> tuple[bool, str]:
    """Materialize the declared, source-documented per-category resized image set.

    The pinned evaluator reads `resized_image/<category>/<id>.jpg` (`LIMN/datasets.py:112`), while
    the documented tool — the FashionIQ `start_kit/resize_images.py` the model's own README points
    at — writes a flat directory at its documented default size. The transform is therefore applied
    here with the same documented size and the output laid out the way the evaluator reads it, and
    the canonical images are only read, never rewritten. Every generated file is a resized copy of
    a canonical image that the model's own split file lists, so the set is checkable.
    """
    recipe = next((a for a in contract.get("generated_artifacts", [])
                   if a.get("kind") == "generated_repo_artifact" and a.get("generator") == RESIZE_GENERATOR),
                  None)
    if recipe is None:
        return True, "no declared resize recipe"

    parameters = recipe.get("parameters") or {}
    size = parameters.get("image_size")
    if not isinstance(size, int) or size <= 0:
        return False, "resize recipe declares no usable image_size"
    source_recipe = parameters.get("source_script")
    if not source_recipe:
        # Without a documented tool there is no source-backed recipe, only a guess.
        return False, "resize recipe names no source script"

    from workbench.backend.fashioniq_layout import CATEGORIES

    images_root = config.FASHIONIQ_ROOT / "images"
    splits_root = config.FASHIONIQ_ROOT / "image_splits"
    if not images_root.is_dir():
        return False, f"canonical images directory missing: {images_root}"

    try:
        from PIL import Image
    except ImportError:
        return False, "PIL is not available in the workbench interpreter, so the resize cannot run"

    # The evaluator decides the output layout, and the contracts differ: the iLearn-family readers
    # open `resized_image/<category>/<id>.jpg`, DCNet opens a flat `resized_images/<id>.jpg`
    # (`data_loader/CE_dataset.py:22,72`). Both are declared in the contract's `paths`.
    layout = parameters.get("layout")
    if layout not in ("per_category", "flat"):
        return False, f"resize recipe declares no usable layout: {layout!r}"
    extension = parameters.get("extension") or ".jpg"

    written, missing = 0, []
    coverage: dict[str, dict[str, int]] = {}
    for category in CATEGORIES:
        split_file = splits_root / f"split.{category}.val.json"
        if not split_file.is_file():
            missing.append(str(split_file))
            continue
        try:
            identifiers = json.loads(split_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            return False, f"cannot read split {split_file}: {error}"
        if not isinstance(identifiers, list):
            return False, f"split {split_file} is not a list of image identifiers"
        for identifier in identifiers:
            # The identifier names a file, so it has to stay one. A separator or a traversal
            # component would escape the output directory, and a non-string is a corrupt split;
            # the canonical split files contain neither.
            if (not isinstance(identifier, str) or not identifier
                    or "/" in identifier or "\\" in identifier or identifier in (".", "..")):
                return False, f"split {split_file} contains an unusable image identifier: {identifier!r}"
        output_dir = (config.FASHIONIQ_ROOT / ("resized_images" if layout == "flat" else "resized_image")
                      / (category if layout == "per_category" else ""))
        if dry_run:
            print(f"[OK] dry-run: {model['model_id']} would resize {len(identifiers)} {category} image(s) "
                  f"to {size}px into {output_dir}")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        for identifier in identifiers:
            source_image = images_root / f"{identifier}{extension}"
            if not source_image.is_file():
                # The canonical distribution ships .jpg; a source that reads another extension is
                # covered by the same recipe because the split names the identifier, not the file.
                alternate = images_root / f"{identifier}.jpg"
                source_image = alternate if alternate.is_file() else source_image
            if not source_image.is_file():
                missing.append(str(source_image))
                continue
            destination = output_dir / f"{identifier}.jpg"
            if destination.is_file():
                continue
            with Image.open(source_image) as handle:
                handle.convert("RGB").resize((size, size), Image.LANCZOS).save(destination, "JPEG")
            written += 1
        coverage[category] = {"expected": len(identifiers),
                              "present": sum(1 for i in identifiers if (output_dir / f"{i}.jpg").is_file())}
    if dry_run:
        return True, "dry-run"
    if missing and written == 0:
        return False, (f"resize input missing: {missing[0]}"
                       + (f" (and {len(missing) - 1} more)" if len(missing) > 1 else ""))
    if missing:
        return False, f"resize completed {written} file(s) but {len(missing)} declared input(s) are missing"

    # A receipt written only after full coverage. The pipeline treats it as the stage's output, so an
    # interrupted run leaves no receipt and the stage reruns instead of resuming on a partial tree.
    receipt = resize_receipt_path(config, layout)
    write_resize_receipt(receipt, model_id=model["model_id"], layout=layout, image_size=size,
                         extension=extension, source_script=source_recipe, coverage=coverage,
                         written=written)
    return True, f"resized {written} image(s) at {size}px per {source_recipe}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--list", action="store_true", help="list all model preparation statuses")
    group.add_argument("--model", help="show or execute preparation for one model")
    parser.add_argument("--dataset-root", type=Path, help="canonical FashionIQ root (overrides config)")
    parser.add_argument("--execute", action="store_true",
                        help="execute preparation for the one verified automated step (CSMCIR symlink only)")
    parser.add_argument("--allow-preparation", action="store_true",
                        help="authorize dataset-link creation; required with --execute")
    parser.add_argument("--dry-run", action="store_true", help="show planned operations without changes")
    parser.add_argument("--json", action="store_true", help="emit machine-readable status")
    args = parser.parse_args(argv)

    if args.execute and not args.allow_preparation:
        parser.error("--execute requires --allow-preparation (explicit authorization)")

    config = resolve_config()
    if args.dataset_root:
        from dataclasses import replace
        config = replace(config, FASHIONIQ_ROOT=args.dataset_root.resolve())

    registry = load_registry()
    models = registry["models"]
    if args.model:
        models = [m for m in models if m["model_id"] == args.model]
        if not models:
            parser.error(f"unknown model: {args.model}")

    statuses = [status(m, config) for m in models]

    if args.json:
        import json
        print(json.dumps(statuses if args.list else statuses[0], indent=2))
        return 0 if all(s["fully_prepared"] for s in statuses) else 1

    if args.list:
        for s in statuses:
            print_status(s)
        unprepared = [s for s in statuses if not s["fully_prepared"]]
        print(f"\n{len(unprepared)} model(s) not fully prepared.")
        return 0

    s = statuses[0]
    model = models[0]
    print_status(s)

    if args.execute:
        performed: list[str] = []
        ok = True
        if model.get("canonical_dataset_link"):
            ok = _fixed_link(config, model["model_id"], args.dry_run, check_only=False) and ok
            performed.append("dataset-link")
        contract = preparation_contract_for_model(model)
        if contract:
            ok = _stage_source_assets(config, model, contract, dry_run=args.dry_run) and ok
            if any(a.get("generator") == BUNDLED_ASSET_GENERATOR for a in contract.get("generated_artifacts", [])):
                performed.append("source-shipped-assets")
            resized = any(a.get("generator") == RESIZE_GENERATOR
                          for a in contract.get("generated_artifacts", []))
            if resized:
                recipe_ok, reason = _run_resize(config, model, contract, dry_run=args.dry_run)
                performed.append(f"resize({reason})")
                # A declared recipe that cannot run is reported exactly, and it only fails the step
                # when the recipe was expected to be executable: a contract that names a generator
                # absent from the pinned source is a documented blocker, not a silent success.
                if not recipe_ok:
                    print(f"[BLOCKED] {model['model_id']}: {reason}", file=sys.stderr)
                    ok = False
        if not performed:
            print(f"[BLOCKED] {model['model_id']}: no verified automated preparation step; manual preparation required.",
                  file=sys.stderr)
            print(f"  policy={s['automation_policy']} notes={s['notes']}", file=sys.stderr)
            return 2
        return 0 if ok else 1

    if model.get("canonical_dataset_link") and not args.dry_run:
        # A read-only status run still reports whether the declared link is correct, without
        # creating it: the check is part of readiness, the creation is the authorized step.
        _fixed_link(config, model["model_id"], dry_run=False, check_only=True)

    return 0 if s["fully_prepared"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
