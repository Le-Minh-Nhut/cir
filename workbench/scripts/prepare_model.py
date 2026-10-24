#!/usr/bin/env python3
"""Inspect model preparation status. Never executes preparation scripts without explicit authorization.

Default: read-only status from preparation_contracts.yaml. Use --execute --allow-preparation to
run verified preparation steps; CUDA-only generators also require --allow-gpu-preparation.

Dataset separation contract (DATA-02):
Three-way separation of dataset paths:
1. Canonical raw dataset (read-only): The operator-supplied canonical benchmark files
   (`images/`, `image_splits/`, canonical `captions/`) are immutable inputs. Preparation
   and evaluation MUST NEVER mutate, delete, or overwrite raw canonical inputs.
2. Model-specific generated artifacts: Generated files derived from raw inputs
   (such as resized JPEG images and staged correction dictionaries).
3. Evaluator-compatible staged layout: Specific target layout required by pinned evaluators,
   such as `resized_image/<category>/` (per-category) or `resized_images/` (flat).

Required writable root:
All generated artifacts and receipts are staged strictly within the operator-configured
`FASHIONIQ_ROOT` directory under dedicated derived paths (`resized_image/`, `resized_images/`,
or staged caption dictionaries). Resizing derives beside/around canonical files and never writes
over canonical raw images (`images/`).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from workbench.backend.operator_config import resolve_config
from workbench.backend.preflight import model_requirements, preflight_blockers
from workbench.backend.registry import load_registry, preparation_contract_for_model
from workbench.replay.common import fashioniq_image_binding_blockers

def status(model: dict, config) -> dict:
    blockers = preflight_blockers(model, config, canonical_root=config.FASHIONIQ_ROOT)
    model_id = model["model_id"]
    contract = preparation_contract_for_model(model)
    requirements = model_requirements(model, config, canonical_root=config.FASHIONIQ_ROOT)
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
#: The PIL resampling kernel the workbench-applied resize uses. The FashionIQ start_kit tool also
#: resizes with `Image.ANTIALIAS`, which modern PIL maps to LANCZOS, so the kernels agree; the
#: comment originally claimed nearest-neighbour, which was false. Recorded here so a kernel change
#: invalidates receipts.
RESAMPLING = "LANCZOS"
#: Name of the receipt a completed resize leaves behind; the pipeline validates it as the output.
RESIZE_RECEIPT_NAME = ".workbench_resize_receipt.json"
#: Marker used by the TME contract for its native FashionIQ feature cache.
TME_FEATURE_GENERATOR = "tme_precompute_features"
TME_RECEIPT_NAME = ".workbench_tme_precompute_receipt.json"


def _tme_artifact(config, checkout: Path, contract: dict) -> tuple[dict, Path]:
    from workbench.backend.registry import preparation_paths

    artifact = next(item for item in contract.get("generated_artifacts", [])
                    if item.get("generator") == TME_FEATURE_GENERATOR)
    paths = preparation_paths(contract, config.FASHIONIQ_ROOT, "generated_artifacts",
                              evaluator_root=checkout)
    return artifact, paths[0]


def tme_receipt_path(config, checkout: Path, contract: dict) -> Path:
    artifact, output = _tme_artifact(config, checkout, contract)
    return (config.WORKBENCH_ARTIFACTS_ROOT / output.relative_to(checkout)
            / artifact["parameters"].get("receipt_filename", TME_RECEIPT_NAME))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _tme_inputs(config, contract: dict) -> dict | None:
    from workbench.backend.fashioniq_layout import CATEGORIES

    parameters = next(item["parameters"] for item in contract["generated_artifacts"]
                      if item.get("generator") == TME_FEATURE_GENERATOR)
    files = {category: config.FASHIONIQ_ROOT / "image_splits"
             / f"split.{category}.{parameters['split']}.json" for category in CATEGORIES}
    captions = [config.FASHIONIQ_ROOT / "captions"
                / f"cap.{category}.{parameters['split']}.json" for category in CATEGORIES]
    if any(not path.is_file() for path in (*files.values(), *captions)):
        return None
    from workbench.replay.common import fashioniq_image_binding_blockers

    image_blockers = fashioniq_image_binding_blockers(config.FASHIONIQ_ROOT, parameters["split"])
    if image_blockers:
        return None
    try:
        if any(not isinstance(json.loads(path.read_text(encoding="utf-8")), list)
               for path in captions):
            return None
        return {"dataset_root": str(config.FASHIONIQ_ROOT.resolve()),
                "split_files": {category: _file_sha256(path) for category, path in files.items()}}
    except (OSError, ValueError, TypeError):
        return None


def _tme_dataset_root(config):
    from workbench.replay.common import isolated_cache_root

    return isolated_cache_root(config.FASHIONIQ_ROOT, directory_names=("captions", "image_splits"),
                               image_source="resized_image")


def _tme_validate(feature_root: Path, parameters: dict, inputs: dict,
                  source_commit: str, receipt: Path) -> bool:
    feature = feature_root / f"{parameters['split']}_images.pt"
    mapping = feature_root / f"{parameters['split']}_name_to_idx.json"
    if not feature.is_file() or not mapping.is_file() or not receipt.is_file():
        return False
    try:
        declared = json.loads(receipt.read_text(encoding="utf-8"))
        if (declared.get("kind") != "workbench_tme_precompute_receipt"
                or declared.get("model_id") != "tme"
                or declared.get("generator") != TME_FEATURE_GENERATOR
                or declared.get("source_script") != parameters["source_script"]
                or declared.get("source_commit") != source_commit
                or declared.get("dataset") != parameters["eval_dataset"]
                or declared.get("split") != parameters["split"]
                or declared.get("batch_size") != parameters["batch_size"]
                or declared.get("workers") != parameters["workers"]):
            return False
        import torch

        features = torch.load(feature, map_location="cpu", weights_only=True)
        names = []
        for category in inputs["split_files"]:
            split_path = Path(inputs["dataset_root"]) / "image_splits" / f"split.{category}.{parameters['split']}.json"
            names.extend(json.loads(split_path.read_text(encoding="utf-8")))
        expected = {name: index for index, name in enumerate(names)}
        actual = json.loads(mapping.read_text(encoding="utf-8"))
        shape = [len(names), parameters["tokens_per_image"], parameters["feature_width"]]
        dtype = {"float16": torch.float16, "float32": torch.float32,
                 "float64": torch.float64}[parameters["feature_dtype"]]
        files = {"features": {"size_bytes": feature.stat().st_size, "sha256": _file_sha256(feature)},
                 "name_to_idx": {"size_bytes": mapping.stat().st_size, "sha256": _file_sha256(mapping)}}
        return (isinstance(features, torch.Tensor) and list(features.shape) == shape
                and features.dtype == dtype and isinstance(actual, dict)
                and list(actual.items()) == list(expected.items())
                and declared.get("files") == files and declared.get("feature_shape") == shape
                and declared.get("feature_dtype") == str(dtype))
    except Exception:
        return False


def tme_features_complete(config, model: dict, contract: dict) -> bool | None:
    recipe = next((item for item in contract.get("generated_artifacts", [])
                   if item.get("generator") == TME_FEATURE_GENERATOR), None)
    if recipe is None:
        return None
    from workbench.backend.runtime import source_clean_and_pinned

    checkout = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    clean, commit, _ = source_clean_and_pinned(model, checkout)
    inputs = _tme_inputs(config, contract)
    if not clean or not commit or inputs is None:
        return False
    _, output = _tme_artifact(config, checkout, contract)
    return _tme_validate(output, recipe["parameters"], inputs, commit,
                         tme_receipt_path(config, checkout, contract))


def _tme_command(interpreter: str, checkout: Path, dataset_root: Path,
                 parameters: dict) -> list[str]:
    shim = ("import runpy,sys; from pathlib import Path; "
            "sys.path.insert(0,str(Path(sys.argv[1]).parent)); import data; "
            "data.FashioniqPath=sys.argv[2]; sys.argv=[sys.argv[1],*sys.argv[3:]]; "
            "runpy.run_path(sys.argv[0],run_name='__main__')")
    script = checkout / parameters["source_script"]
    return [interpreter, "-c", shim, str(script), str(dataset_root), "--dataset",
            parameters["eval_dataset"], "--split", parameters["split"], "--batch_size",
            str(parameters["batch_size"]), "--num_works", str(parameters["workers"])]


def _run_tme_precompute(config, model: dict, contract: dict, *, dry_run: bool,
                        force_regenerate_unknown_cache: bool = False) -> tuple[bool, str]:
    from workbench.backend.registry import external_asset_blockers
    from workbench.backend.runtime import require_verified_model_interpreter, source_clean_and_pinned

    checkout = config.WORKBENCH_THIRD_PARTY_ROOT / model["source_dir"]
    recipe, output = _tme_artifact(config, checkout, contract)
    parameters = recipe["parameters"]
    script = checkout / parameters["source_script"]
    if not checkout.is_dir() or not script.is_file():
        return False, f"pinned TME checkout or generator missing: {checkout} ({script})"
    clean, commit, reason = source_clean_and_pinned(model, checkout)
    if not clean or not commit:
        return False, f"pinned TME source unavailable: {reason or 'source is not clean and pinned'}"
    image_blockers = fashioniq_image_binding_blockers(config.FASHIONIQ_ROOT, parameters["split"])
    if image_blockers:
        return False, image_blockers[0]
    inputs = _tme_inputs(config, contract)
    if inputs is None:
        return False, f"TME caption or validation split input is missing or malformed under {config.FASHIONIQ_ROOT}"
    receipt = tme_receipt_path(config, checkout, contract)
    if _tme_validate(output, parameters, inputs, commit, receipt):
        return True, f"reusing validated TME feature cache at {output}"
    files = [output / f"{parameters['split']}_images.pt",
             output / f"{parameters['split']}_name_to_idx.json"]
    existing = [path for path in files if path.exists()]
    if existing and not force_regenerate_unknown_cache:
        message = ("TME cache bytes have no valid receipt or no longer match bound inputs; refusing to overwrite: "
                   f"{', '.join(map(str, existing))}. To explicitly replace these unknown/stale bytes, rerun with "
                   "--force-regenerate-unknown-cache after reviewing the cache.")
        print(f"[WARNING] {message}", file=sys.stderr)
        return False, message
    if receipt.is_file() and not existing:
        print(f"[WARNING] TME receipt is stale; cache must be regenerated because source commit or "
              f"dataset root/split digests changed: {receipt}", file=sys.stderr)
    try:
        interpreter = require_verified_model_interpreter(model, config=config)
    except RuntimeError as error:
        if not dry_run:
            return False, f"TME model interpreter is unverified: {error}"
        interpreter = "<verified TME interpreter>"
        print(f"[BLOCKED] TME interpreter not verified: {error}", file=sys.stderr)
    if dry_run:
        import shlex
        command = _tme_command(interpreter, checkout, config.FASHIONIQ_ROOT, parameters)
        print(f"[DRY-RUN] Native generator command: {shlex.join(command)}")
        return True, "dry-run; native generator not launched"
    try:
        probe = subprocess.run([interpreter, "-c", "import torch; print(torch.cuda.is_available())"],
                               capture_output=True, text=True, check=False)
    except OSError as error:
        return False, f"cannot check CUDA with verified TME interpreter {interpreter}: {error}"
    if probe.returncode != 0:
        return False, f"cannot check CUDA with verified TME interpreter {interpreter}: {probe.stderr.strip()}"
    if probe.stdout.strip().splitlines()[-1].lower() != "true":
        return False, ("TME feature generation requires a CUDA-capable GPU; verified interpreter reports "
                       "torch.cuda.is_available() == False. No CPU fallback is supported.")
    blockers = external_asset_blockers(model)
    if blockers:
        return False, f"TME native generator prerequisite blocked: {blockers[0]}"
    output.mkdir(parents=True, exist_ok=True)
    with _tme_dataset_root(config) as staged_root:
        command = _tme_command(interpreter, checkout, staged_root, parameters)
        try:
            result = subprocess.run(command, cwd=checkout, check=False)
        except OSError as error:
            return False, f"could not launch pinned TME generator {script}: {error}"
        if result.returncode != 0:
            return False, f"pinned TME generator failed with exit code {result.returncode}; no receipt written"
        if not all(path.is_file() for path in files):
            return False, f"pinned TME generator omitted required output; regenerate {', '.join(map(str, files))}"
    try:
        import torch
        features = torch.load(files[0], map_location="cpu", weights_only=True)
        names = []
        for category in ("dress", "shirt", "toptee"):
            names.extend(json.loads((Path(inputs["dataset_root"]) / "image_splits"
                                     / f"split.{category}.{parameters['split']}.json").read_text(encoding="utf-8")))
        mapping = json.loads(files[1].read_text(encoding="utf-8"))
        expected = {name: index for index, name in enumerate(names)}
        shape = [len(expected), parameters["tokens_per_image"], parameters["feature_width"]]
        dtype = {"float16": torch.float16, "float32": torch.float32,
                 "float64": torch.float64}[parameters["feature_dtype"]]
        if (not isinstance(features, torch.Tensor) or list(features.shape) != shape
                or features.dtype != dtype or list(mapping.items()) != list(expected.items())):
            return False, "TME output shape/dtype/gallery order invalid; feature cache must be regenerated"
        payload = {"kind": "workbench_tme_precompute_receipt", "model_id": model["model_id"],
                   "generator": TME_FEATURE_GENERATOR, "source_script": parameters["source_script"],
                   "source_commit": commit, "dataset": parameters["eval_dataset"], "split": parameters["split"],
                   "inputs": inputs, "batch_size": parameters["batch_size"], "workers": parameters["workers"],
                   "files": {"features": {"size_bytes": files[0].stat().st_size, "sha256": _file_sha256(files[0])},
                             "name_to_idx": {"size_bytes": files[1].stat().st_size, "sha256": _file_sha256(files[1])}},
                   "feature_shape": shape, "feature_dtype": str(features.dtype),
                   "interpreter": str(Path(interpreter).resolve())}
        receipt.parent.mkdir(parents=True, exist_ok=True)
        temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, receipt)
    except (OSError, ValueError, TypeError, RuntimeError, ImportError) as error:
        return False, f"TME output validation failed; no receipt written and cache must be regenerated: {error}"
    return True, f"generated and validated native TME feature cache at {output}"


def resize_receipt_path(config, layout: str) -> Path:
    """Where a completed resize records its coverage. Only a fully covered run writes one."""
    return config.FASHIONIQ_ROOT / ("resized_images" if layout == "flat" else "resized_image") / RESIZE_RECEIPT_NAME


def recipe_digest(parameters: dict) -> str:
    """Stable digest of recipe parameters."""
    normalized = json.dumps(parameters or {}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def write_resize_receipt(receipt: Path, *, model_id: str, layout: str, image_size: int,
                         extension: str, source_script: str, coverage: dict, written: int,
                         recipe_digest: str | None = None) -> None:
    """Write the coverage receipt atomically, so a crash never leaves a half-written one."""
    receipt.parent.mkdir(parents=True, exist_ok=True)
    if recipe_digest is None:
        for c in coverage.values():
            if isinstance(c, dict) and c.get("recipe_digest"):
                recipe_digest = str(c["recipe_digest"])
                break
    if recipe_digest is None:
        recipe_digest = hashlib.sha256(
            json.dumps({"image_size": image_size, "extension": extension, "layout": layout, "source_script": source_script},
                       sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    recipe_identity = {
        "image_size": image_size,
        "extension": extension,
        "layout": layout,
        "recipe_digest": recipe_digest,
    }
    payload = {
        "kind": "workbench_resize_receipt",
        "model_id": model_id,
        "layout": layout,
        "image_size": image_size,
        "extension": extension,
        "source_script": source_script,
        "recipe_digest": recipe_digest,
        "recipe_parameters_digest": recipe_digest,
        "recipe_identity": recipe_identity,
        "categories": coverage,
        "written_now": written,
        # The resampling kernel is a constant of the workbench code, not a registry parameter, so
        # the recipe digest cannot see it. Recorded explicitly: if the code ever resizes with a
        # different filter, old receipts stop matching and every derived file is regenerated.
        "resampling": RESAMPLING,
    }
    temporary = receipt.with_name(f".{receipt.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, receipt)

def resize_is_complete(config, model: dict, contract: dict) -> bool | None:
    """Whether the declared resize is finished, judged from its receipt and the files it names.

    Returns None when the model declares no resize. A receipt alone is not enough: the files it
    promises may have been removed or tampered with afterwards, which is exactly the case a resume
    would otherwise skip over. Older receipts lacking digests do not validate as complete.
    """
    recipe = next((a for a in contract.get("generated_artifacts", [])
                   if a.get("generator") == RESIZE_GENERATOR), None)
    if recipe is None:
        return None
    parameters = recipe.get("parameters") or {}
    layout = parameters.get("layout")
    if layout not in ("per_category", "flat"):
        return False
    recipe_size = parameters.get("image_size")
    if not isinstance(recipe_size, int) or recipe_size <= 0:
        return False
    recipe_ext = parameters.get("extension") or ".jpg"
    if not recipe_ext.startswith("."):
        recipe_ext = f".{recipe_ext}"
    expected_recipe_digest = recipe_digest(parameters)

    receipt = resize_receipt_path(config, layout)
    if not receipt.is_file():
        return False
    try:
        declared = json.loads(receipt.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False

    recorded_recipe_digest = declared.get("recipe_digest") or declared.get("recipe_parameters_digest")
    if not recorded_recipe_digest or recorded_recipe_digest != expected_recipe_digest:
        return False
    if declared.get("layout") != layout:
        return False
    if declared.get("image_size") != recipe_size:
        return False
    if declared.get("extension") != recipe_ext:
        return False
    # The kernel is a constant of this module; a receipt that names a different filter was not
    # produced by this code and must not be trusted.
    if declared.get("resampling") != RESAMPLING:
        return False

    categories = declared.get("categories")
    if not isinstance(categories, dict) or not categories:
        return False

    from workbench.backend.fashioniq_layout import CATEGORIES as _CATEGORIES

    if set(categories) != set(_CATEGORIES):
        return False

    try:
        from PIL import Image
    except ImportError:
        return False

    base = config.FASHIONIQ_ROOT / ("resized_images" if layout == "flat" else "resized_image")
    if not base.is_dir():
        return False
    splits_root = config.FASHIONIQ_ROOT / "image_splits"
    images_root = config.FASHIONIQ_ROOT / "images"

    for category, counts in categories.items():
        if not isinstance(counts, dict) or counts.get("expected") != counts.get("present"):
            return False
        split_file = splits_root / f"split.{category}.val.json"
        if not split_file.is_file():
            return False
        try:
            expected_ids = json.loads(split_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        if not isinstance(expected_ids, list):
            return False

        target_dir = base if layout == "flat" else base / category
        if not target_dir.is_dir():
            return False
        ids = counts.get("ids")
        if not isinstance(ids, list) or len(ids) != int(counts.get("present") or -1):
            return False
        if set(ids) != set(expected_ids) or len(ids) != len(expected_ids):
            return False
        digests = counts.get("digests")
        if not isinstance(digests, dict):
            images_map = counts.get("images")
            if isinstance(images_map, dict):
                digests = {k: v.get("sha256") for k, v in images_map.items() if isinstance(v, dict)}
            else:
                return False
        if not digests:
            return False
        # The derived images are only valid for the canonical bytes they were made from. A receipt
        # without source digests cannot prove that, so it is treated as incomplete: an older resize
        # from before this binding existed has to be regenerated rather than silently trusted.
        source_digests = counts.get("source_digests")
        if not isinstance(source_digests, dict) or not source_digests:
            return False
        for identifier in ids:
            str_id = str(identifier)
            expected_hash = digests.get(str_id)
            if not expected_hash:
                return False
            if not source_digests.get(str_id):
                return False
            source_image = images_root / f"{str_id}{recipe_ext}"
            # The receipt records the extension the generator actually read (the canonical
            # distribution may be .png while the recipe writes .jpg). Validating against a guessed
            # `.jpg` fallback made every completion check fail on a .png canonical tree, forcing a
            # full regenerate on each resume.
            recorded_suffix = (counts.get("source_extensions") or {}).get(str_id)
            if not source_image.is_file() and recorded_suffix:
                alternate = images_root / f"{str_id}{recorded_suffix}"
                source_image = alternate if alternate.is_file() else source_image
            if not source_image.is_file():
                return False
            try:
                if hashlib.sha256(source_image.read_bytes()).hexdigest() != source_digests[str_id]:
                    return False
            except OSError:
                return False
            target = target_dir / f"{str_id}{recipe_ext}"
            if not target.is_file():
                return False
            try:
                data = target.read_bytes()
            except OSError:
                return False
            if hashlib.sha256(data).hexdigest() != expected_hash:
                return False
            try:
                with Image.open(target) as img:
                    expected_format = Image.registered_extensions().get(recipe_ext.lower(), "JPEG")
                    if img.format != expected_format:
                        return False
                    if img.size != (recipe_size, recipe_size):
                        return False
                    img.load()
            except Exception:
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


#: Canonical image extensions the workbench will read as a resize source. The split files name an
#: image *identifier*, not a filename: on this dataset the files are `<id>.png` while the pinned
#: evaluators read `<id>.jpg`, so the source extension and the derived extension are different
#: properties. Trying only the derived extension (or only `.jpg`) finds nothing on a PNG dataset.
CANONICAL_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG")


def _canonical_image_path(images_root: Path, identifier: str, derived_extension: str) -> Path | None:
    """The canonical file the identifier names, or None when it does not exist.

    Derived extension first, because a canonical JPEG distribution should be found without scanning,
    then the declared canonical extensions. The returned file is the one actually read, so its bytes
    are what the receipt binds.
    """
    ordered = [derived_extension, *CANONICAL_IMAGE_EXTENSIONS]
    seen: set[str] = set()
    for extension in ordered:
        if extension in seen:
            continue
        seen.add(extension)
        candidate = images_root / f"{identifier}{extension}"
        if candidate.is_file():
            return candidate
    return None


def _derived_is_bound_to_source(
    previous: dict | None,
    category: str,
    identifier: str,
    source_digest: str,
    size: int,
    extension: str,
    derived_digest: str | None = None,
    recipe_digest: str | None = None,
    recipe_identity: dict | None = None,
) -> bool:
    """Whether a receipt proves an existing derived file came from these exact canonical bytes.

    The evidence required is the conjunction the receipt already records: this identifier's canonical
    source digest, its derived digest, the image size, the recipe extension, the resampling kernel,
    and the full recipe identity (digest + identity block). A missing or mismatching value means the
    file cannot be attributed to the current source and recipe, so it is regenerated instead of
    reused.

    The derived digest matters as much as the source one. Without it a file that was edited after it
    was generated still looked reusable, and the run then wrote a fresh receipt recording the edited
    bytes -- laundering a tampered artifact into a certified one.
    """
    if not isinstance(previous, dict):
        return False
    if previous.get("image_size") != size or previous.get("extension") != extension:
        return False
    # A file produced with a different resampling kernel is a different derived artifact, even
    # though its source bytes match. Receipts written before the kernel was recorded carry no
    # claim at all, so they cannot prove reuse either.
    if previous.get("resampling") != RESAMPLING:
        return False
    # Full recipe identity, not just a few fields of it: the receipt must record the same recipe
    # digest the current contract derives, the same layout, and the same recipe identity block.
    # A receipt that disagrees on any of these describes a different generator run, and reuse
    # would certify pixels the current recipe did not produce.
    if previous.get("recipe_digest") != recipe_digest:
        return False
    if previous.get("recipe_identity") != recipe_identity:
        return False
    categories = previous.get("categories")
    if not isinstance(categories, dict):
        return False
    counts = categories.get(category)
    if not isinstance(counts, dict):
        return False
    if counts.get("image_size") != size or counts.get("extension") != extension:
        return False
    # The per-category coverage must agree with the receipt's own recipe: a counts block recorded
    # under a different layout or digest describes files the current recipe did not produce.
    if counts.get("layout") != (recipe_identity or {}).get("layout"):
        return False
    if counts.get("recipe_digest") != recipe_digest:
        return False
    digests = counts.get("digests") or {}
    source_digests = counts.get("source_digests") or {}
    if not isinstance(digests, dict) or not isinstance(source_digests, dict):
        return False
    recorded = digests.get(identifier)
    if not recorded or source_digests.get(identifier) != source_digest:
        return False
    if derived_digest is None:
        return False
    return recorded == derived_digest


def _load_previous_receipt(config, layout: str, *, model_id: str) -> dict | None:
    """The receipt this model's previous run wrote, when it belongs to this model and layout."""
    path = resize_receipt_path(config, layout)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or payload.get("model_id") != model_id:
        return None
    return payload


def _run_resize(config, model: dict, contract: dict, *, dry_run: bool) -> tuple[bool, str]:
    """Materialize the declared, source-documented per-category resized image set.

    DATA-02: Three-way dataset separation:
    1. Canonical raw dataset (read-only): `images/`, `image_splits/`, canonical `captions/`.
    2. Model-specific generated artifacts: resized JPEG images and staged dictionaries.
    3. Evaluator-compatible staged layout: `resized_image/<category>/` (per-category) or `resized_images/` (flat).
    Required writable root: `config.FASHIONIQ_ROOT` (derived paths only; canonical `images/` is read-only).

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
    if not extension.startswith("."):
        extension = f".{extension}"
    recipe_parameters_digest = recipe_digest(parameters)
    recipe_identity = {
        "image_size": size,
        "extension": extension,
        "layout": layout,
        "recipe_digest": recipe_parameters_digest,
    }

    written, missing = 0, []
    coverage: dict[str, dict] = {}
    previous_receipt = _load_previous_receipt(config, layout, model_id=model["model_id"]) if not dry_run else None
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
        category_source_digests: dict[str, str] = {}
        category_source_extensions: dict[str, str] = {}
        for identifier in identifiers:
            source_image = _canonical_image_path(images_root, identifier, extension)
            if source_image is None:
                missing.append(str(images_root / f"{identifier}<image>"))
                continue
            # Bind each derived image to the canonical bytes it was made from. Without this the
            # receipt proved only that *a* resize had happened: editing a canonical image left the
            # old copy in place (the destination still had the right size and format), the receipt
            # still validated, and a resume silently scored stale pixels.
            category_source_digests[identifier] = hashlib.sha256(source_image.read_bytes()).hexdigest()
            # The canonical extension is recorded too. It is not always the derived extension: on the
            # canonical FashionIQ distribution the files are `<id>.png` while the iLearn readers open
            # `<id>.jpg` under `resized_image/`, which is a different tree from `images/`. Recording
            # which file was read makes that difference auditable instead of implied.
            category_source_extensions[identifier] = source_image.suffix
            destination = output_dir / f"{identifier}{extension}"
            # DATA-02: Refuse to write a resized image whose destination equals its source path
            # or targets the canonical raw images directory.
            if destination.resolve() == source_image.resolve() or destination == source_image:
                return False, f"destination path equals canonical source path: {destination}"
            try:
                destination.resolve().relative_to(images_root.resolve())
                return False, f"refusing to write derived artifact into canonical images directory: {destination}"
            except ValueError:
                pass
            save_format = Image.registered_extensions().get(extension.lower(), "JPEG")
            if destination.is_file():
                try:
                    on_disk = hashlib.sha256(destination.read_bytes()).hexdigest()
                except OSError:
                    on_disk = None
                if _derived_is_bound_to_source(
                        previous_receipt, category, identifier, category_source_digests[identifier],
                        size, extension, on_disk,
                        recipe_digest=recipe_parameters_digest, recipe_identity=recipe_identity):
                    # Reuse is allowed only with evidence: the previous receipt records the canonical
                    # bytes that produced this file, the recipe that produced it, and the file's own
                    # digest. Size and format alone are not evidence -- a 256px JPEG can be stale,
                    # generated by a different recipe, or edited afterwards, and recording the
                    # *current* source digest next to those pixels would make the new receipt assert
                    # a provenance the file does not have.
                    continue
            temp_file = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
            try:
                with Image.open(source_image) as handle:
                    handle.convert("RGB").resize((size, size), getattr(Image, RESAMPLING)).save(temp_file, format=save_format)
                os.replace(temp_file, destination)
            finally:
                if temp_file.exists():
                    try:
                        temp_file.unlink()
                    except OSError:
                        pass
            written += 1
        present_ids = sorted(i for i in identifiers if (output_dir / f"{i}{extension}").is_file())
        category_digests: dict[str, str] = {}
        category_sizes: dict[str, list[int]] = {}
        category_images: dict[str, dict] = {}
        for identifier in present_ids:
            dest = output_dir / f"{identifier}{extension}"
            img_bytes = dest.read_bytes()
            h = hashlib.sha256(img_bytes).hexdigest()
            category_digests[identifier] = h
            category_sizes[identifier] = [size, size]
            category_images[identifier] = {
                "size": [size, size],
                "bytes": len(img_bytes),
                "sha256": h,
            }
        coverage[category] = {
            "expected": len(identifiers),
            "present": len(present_ids),
            "ids": present_ids,
            "image_size": size,
            "extension": extension,
            "layout": layout,
            "recipe_digest": recipe_parameters_digest,
            "recipe_identity": recipe_identity,
            "digests": category_digests,
            "sizes": category_sizes,
            "images": category_images,
            "source_digests": category_source_digests,
            "source_extensions": category_source_extensions,
        }
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
                         written=written, recipe_digest=recipe_parameters_digest)
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
    parser.add_argument("--allow-gpu-preparation", action="store_true",
                        help="authorize explicitly declared CUDA-only model preparation")
    parser.add_argument("--dry-run", action="store_true", help="show planned operations without changes")
    parser.add_argument("--force-regenerate-unknown-cache", action="store_true",
                        help="allow replacing TME cache files without a valid matching preparation receipt")
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
            tme_requested = any(a.get("generator") == TME_FEATURE_GENERATOR
                                for a in contract.get("generated_artifacts", []))
            if tme_requested and not args.dry_run and not args.allow_gpu_preparation:
                reason = "CUDA preparation requires --allow-gpu-preparation"
                performed.append(f"tme-precompute({reason})")
                print(f"[BLOCKED] {model['model_id']}: {reason}", file=sys.stderr)
                ok = False
            elif tme_requested:
                recipe_ok, reason = _run_tme_precompute(
                    config, model, contract, dry_run=args.dry_run,
                    force_regenerate_unknown_cache=args.force_regenerate_unknown_cache)
                performed.append(f"tme-precompute({reason})")
                if not recipe_ok:
                    print(f"[BLOCKED] {model['model_id']}: {reason}", file=sys.stderr)
                    ok = False
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
