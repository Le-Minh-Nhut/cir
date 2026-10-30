"""Run opt-in exact-rank CIRCUS evaluation with disk-backed full-gallery features."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import shutil
import sys
import inspect
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from PIL import Image

from workbench.composition.adapters import AdapterWorkerError, ExactRankUnavailable
from workbench.composition.inputs import CONDITIONS, adapt_query_inputs
from workbench.composition.data import PAPER_STATED_GALLERY_IMAGES, RELEASED_GALLERY_IMAGES

@dataclass(frozen=True)
class ModelSpec:
    """One model/checkpoint selected by the existing workbench policy."""

    model_id: str
    checkpoint_id: str
    adapter_factory: Callable[..., Any]
    registry_model: dict[str, Any]
    checkpoint_identity: str = ""


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def assert_composition_destination(artifacts_root: Path, destination: Path) -> Path:
    """Refuse every path outside the isolated composition artifact namespace."""
    artifacts = Path(artifacts_root).expanduser().resolve()
    composition = (artifacts / "composition").resolve()
    target = Path(destination).expanduser().resolve()
    if not _inside(composition, artifacts) or not _inside(target, composition):
        raise ValueError(f"composition output destination must be inside {composition}; refused {target}")
    return target


def _component(value: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value or "/" in value or "\\" in value:
        raise ValueError(f"unsafe output path component: {value!r}")
    return value


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scope_name(scope: str | None) -> str:
    return "global" if scope is None else _component(scope)


#: Per-query failure reasons that a rerun reproduces deterministically. A recorded PARTIAL whose
#: every failure is one of these is never retried on resume; only transient reasons (model
#: unavailable, failed, unavailable exact rank, or an unexpected exception) earn another attempt.
_STABLE_FAILURE_MARKERS = frozenset({
    "category scope failure",
    "has no adapter embedding scope",
    "no positive target resolved in exact released gallery",
    "query features unavailable",
    "invalid full-gallery exact rank",
    "query reference image is missing",
})


def _is_stable_failure(reason: str) -> bool:
    lowered = reason.lower()
    return any(marker in lowered for marker in _STABLE_FAILURE_MARKERS)


def normalize_release_path(value: str) -> str:
    """Remove release placeholder while preserving exact relative path identity."""
    value = value.replace("\\", "/")
    return re.sub(r"^<path_to[-_]M[-_]BEIR>/", "", value, flags=re.IGNORECASE)


def _image_directories(dataset_root: Path, layout: str) -> tuple[Path, ...]:
    from workbench.backend.fashioniq_layout import paths_for_layout

    dirs = []
    for path in (*paths_for_layout(dataset_root, layout), dataset_root / "images"):
        if path.name in {"images", "resized_images", "resized_image"} or path.parent.name == "resized_image":
            if path.is_dir():
                dirs.append(path)
    return tuple(dict.fromkeys(dirs))


def image_resolver(dataset_root: Path, layout: str) -> Callable[[str, Mapping[str, str]], Path | None]:
    """Index configured FashionIQ layout by stem, retaining released identity/order separately."""
    dirs = _image_directories(dataset_root, layout)
    exact: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = {}
    for directory in dirs:
        for path in directory.iterdir():
            if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            exact.setdefault(path.name, path)
            by_stem.setdefault(path.stem, []).append(path)
    for name in by_stem:
        by_stem[name].sort(key=lambda path: (path.suffix.lower() != ".png", str(path)))

    def resolve(identity: str, candidate_names: Mapping[str, str]) -> Path | None:
        released = candidate_names.get(identity, identity)
        basename = PurePosixPath(normalize_release_path(released)).name
        direct = exact.get(basename)
        if direct is not None:
            return direct
        matches = by_stem.get(Path(basename).stem, ())
        return matches[0] if matches else None

    return resolve


def resolve_release_image(image_id: str, candidate_names: Mapping[str, str],
                          dataset_root: Path, layout: str) -> Path | None:
    """Resolve one release identity through a registry-declared image layout."""
    return image_resolver(dataset_root, layout)(image_id, candidate_names)


def _as_array(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _load_memmap(path: Path, fingerprint: str, shape: tuple[int, ...] | None = None) -> np.memmap | None:
    try:
        metadata = json.loads(path.with_suffix(path.suffix + ".json").read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != fingerprint or not path.is_file():
            return None
        result = np.load(path, mmap_mode="r", allow_pickle=False)
        if shape is not None and result.shape != shape:
            return None
        return result
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_memmap_metadata(path: Path, fingerprint: str, value: np.ndarray) -> None:
    _atomic_json(path.with_suffix(path.suffix + ".json"),
                 {"fingerprint": fingerprint, "shape": list(value.shape), "dtype": str(value.dtype)})


def _gallery_fingerprint(data: Any) -> str:
    return _digest({"protocol_id": data.gallery.protocol_id,
                    "ordered_gallery": list(data.gallery.image_ids),
                    "candidate_names": dict(sorted(data.candidate_names.items()))})


def _query_fingerprint(queries: Sequence[Any]) -> str:
    return _digest([{"qid": query.qid, "index": query.index, "text": query.query_txt,
                     "reference_id": query.reference_id, "positive_ids": list(query.positive_ids),
                     "category": query.category, "subsets": list(query.subsets)} for query in queries])


def _positive_positions(data: Any) -> dict[str, tuple[int, ...]]:
    index = {identity: position for position, identity in enumerate(data.gallery.image_ids)}
    by_path = {normalize_release_path(identity): position
               for position, identity in enumerate(data.gallery.image_ids)}
    for candidate_id, image_path in data.candidate_names.items():
        position = by_path.get(normalize_release_path(image_path))
        if position is not None:
            index[candidate_id] = position
    return {query.qid: tuple(dict.fromkeys(index[value] for value in query.positive_ids if value in index))
            for query in data.queries}


def _scope_for_query(adapter: Any, query: Any) -> tuple[str | None, str | None]:
    scopes = tuple(getattr(adapter, "category_scopes", (None,)))
    if query.category in scopes:
        return query.category, None
    if None in scopes:
        return None, None
    return None, (f"query {query.qid}: category {query.category!r} has no adapter embedding scope "
                   f"{scopes!r}; batch_indices=({query.index},), qids={[query.qid]}")


def _disk_projection(path: Path, gallery_count: int, feature_shape: Sequence[int], remaining_scopes: int,
                     query_count: int, conditions: int) -> dict[str, int]:
    feature_width = int(np.prod(feature_shape, dtype=np.int64))
    matrix_bytes = gallery_count * feature_width * np.dtype(np.float32).itemsize
    query_width = int(feature_shape[-1])
    query_bytes = query_count * query_width * np.dtype(np.float32).itemsize * conditions
    projected = matrix_bytes * remaining_scopes + query_bytes
    margin = max(64 * 1024 * 1024, projected // 10)
    free = shutil.disk_usage(path.parent).free
    if free < projected + margin:
        raise OSError(f"composition features need {projected} bytes plus {margin} safety margin; "
                      f"only {free} bytes free at {path.parent}")
    return {"matrix_bytes_per_scope": matrix_bytes, "query_feature_bytes": query_bytes,
            "projected_bytes": projected, "safety_margin_bytes": margin, "free_bytes": free}


def _encode_gallery(adapter: Any, image_paths: Sequence[Path], scope: str | None,
                    destination: Path, fingerprint: str, batch_size: int,
                    remaining_scopes: int, query_count: int, condition_count: int) -> tuple[np.memmap, dict[str, int]]:
    cached = _load_memmap(destination, fingerprint)
    if cached is not None:
        return cached, {"reused": 1, "projected_bytes": destination.stat().st_size}
    destination.parent.mkdir(parents=True, exist_ok=True)
    mapped = None
    trailing_shape: tuple[int, ...] | None = None
    position = 0
    projection: dict[str, int] = {}
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start:start + batch_size]
        encoded = _as_array(adapter.encode_targets([str(path) for path in paths], [scope] * len(paths)))
        if encoded.ndim < 2 or encoded.shape[0] != len(paths):
            raise ValueError(f"encode_targets returned invalid shape {encoded.shape}")
        current_shape = tuple(encoded.shape[1:])
        if trailing_shape is not None and current_shape != trailing_shape:
            raise ValueError(f"encode_targets feature shape changed between batches: {trailing_shape} != {current_shape}")
        encoded = encoded.astype(np.float32, copy=False)
        if mapped is None:
            trailing_shape = current_shape
            projection = _disk_projection(destination.parent, len(image_paths), trailing_shape,
                                          remaining_scopes, query_count, condition_count)
            mapped = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                               shape=(len(image_paths), *trailing_shape))
        mapped[start:start + len(paths)] = encoded
        position += len(paths)
        del encoded
    if mapped is None or position != len(image_paths):
        raise ValueError("gallery encoding produced no features")
    mapped.flush()
    _write_memmap_metadata(destination, fingerprint, mapped)
    return np.load(destination, mmap_mode="r", allow_pickle=False), projection


def _stage_query_path(path: Path, query: Any, condition: str, work_root: Path) -> tuple[Path, str]:
    with Image.open(path) as source:
        image = source.copy()
    adapted, text = adapt_query_inputs(image, query.query_txt, condition)
    if condition == "text_only_black_image":
        staged = work_root / "queries" / condition / f"{_component(query.qid)}.png"
        staged.parent.mkdir(parents=True, exist_ok=True)
        adapted.save(staged, format="PNG")
        return staged, text
    return path, text


def _encode_queries(adapter: Any, queries: Sequence[Any], query_paths: Sequence[Path | None],
                    indices: Sequence[int], condition: str, work_root: Path,
                    destination: Path, fingerprint: str, dimension: int,
                    batch_size: int) -> tuple[np.memmap, dict[str, str], dict[str, str]]:
    cached = _load_memmap(destination, fingerprint, (len(queries), dimension))
    errors_path = destination.with_suffix(destination.suffix + ".errors.json")
    category_path = destination.with_suffix(destination.suffix + ".category-errors.json")
    if cached is not None:
        try:
            errors = json.loads(errors_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            errors = {}
        try:
            category_errors = json.loads(category_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            category_errors = {}
        return cached, errors, category_errors

    destination.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32,
                                         shape=(len(queries), dimension))
    features[:] = np.nan
    errors: dict[str, str] = {}
    category_errors: dict[str, str] = {}
    eligible = [index for index in indices if query_paths[index] is not None]
    for index in indices:
        if query_paths[index] is None:
            errors[queries[index].qid] = "query reference image is missing"
    for batch_start in range(0, len(eligible), batch_size):
        batch = eligible[batch_start:batch_start + batch_size]
        paths: list[str] = []
        texts: list[str] = []
        categories: list[str | None] = []
        surviving: list[int] = []
        for index in batch:
            try:
                staged_path, text = _stage_query_path(query_paths[index], queries[index], condition, work_root)  # type: ignore[arg-type]
                paths.append(str(staged_path))
                texts.append(text)
                categories.append(queries[index].category)
                surviving.append(index)
            except Exception as error:
                errors[queries[index].qid] = f"{type(error).__name__}: {error}"
        if not surviving:
            continue
        try:
            encoded = _as_array(adapter.encode_queries(paths, texts, categories))
            if encoded.shape != (len(surviving), dimension):
                raise ValueError(f"encode_queries returned invalid shape {encoded.shape}")
            features[surviving] = encoded.astype(np.float32, copy=False)
        except Exception as batch_error:
            if isinstance(batch_error, AdapterWorkerError):
                raise
            bad_batch_indices = tuple(getattr(batch_error, "batch_indices", ())) \
                if type(batch_error).__name__ == "CategoryScopeError" else ()
            bad_qids = [queries[surviving[index]].qid for index in bad_batch_indices
                        if 0 <= index < len(surviving)]
            if bad_batch_indices:
                reason = (f"{getattr(batch_error, 'model_id', type(adapter).__name__)} category scope failure; "
                          f"batch_indices={bad_batch_indices}; qids={bad_qids}; {batch_error}")
                category_errors.update({qid: reason for qid in bad_qids})
            for local_index, (index, path, text) in enumerate(zip(surviving, paths, texts)):
                if local_index in bad_batch_indices:
                    continue
                try:
                    encoded = _as_array(adapter.encode_queries([path], [text], [queries[index].category]))
                    if encoded.shape != (1, dimension):
                        raise ValueError(f"encode_queries returned invalid shape {encoded.shape}")
                    features[index] = encoded[0].astype(np.float32, copy=False)
                except Exception as error:
                    if isinstance(error, AdapterWorkerError):
                        raise
                    if type(error).__name__ == "CategoryScopeError":
                        local_indices = tuple(getattr(error, "batch_indices", ()))
                        category_errors[queries[index].qid] = (
                            f"{getattr(error, 'model_id', type(adapter).__name__)} category scope failure; "
                            f"batch_indices={local_indices}; qids={[queries[index].qid]}; {error}")
                    else:
                        errors[queries[index].qid] = f"{type(error).__name__}: {error} (batch: {batch_error})"
    features.flush()
    _write_memmap_metadata(destination, fingerprint, features)
    _atomic_json(errors_path, errors)
    _atomic_json(category_path, category_errors)
    return np.load(destination, mmap_mode="r", allow_pickle=False), errors, category_errors


def score_in_query_chunks(adapter: Any, queries: Any, targets: Any, chunk_size: int,
                          categories: Sequence[str | None] | None = None) -> np.ndarray:
    """Score query chunks against the unchanged full target matrix through adapter semantics."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    pieces = []
    for start in range(0, len(queries), chunk_size):
        end = min(start + chunk_size, len(queries))
        part = queries[start:end]
        if categories is None:
            scores = adapter.score(part, targets)
        else:
            scores = adapter.score(part, targets, categories[start:end])
        pieces.append(_as_array(scores))
    return np.concatenate(pieces, axis=0) if pieces else np.empty((0, len(targets)), dtype=np.float32)


def _gpu_identity() -> str | None:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:
        pass
    return None

def model_interpreter_identity(model: dict[str, Any], config: Any, adapter: Any = None, *,
                               verify: bool = True) -> dict[str, Any]:
    from workbench.backend.model_settings import setting_source
    from workbench.backend.runtime import (
        READY_TIERS, _record_environment_fingerprint, resolve_model_interpreter, verify_environment)

    variable = (model.get("environment") or {}).get("python_env_var")
    path = getattr(adapter, "interpreter", None) or resolve_model_interpreter(model, config)
    source = setting_source(variable) if variable else "unset"
    if source == "unset" and path:
        source = "artifact_assignment"
    if not path:
        return {"path": None, "version": "unavailable", "source": source,
                "verified": False, "verification_reason":
                f"no model interpreter resolved ({variable or 'no declared variable'})"}
    resolved = os.path.abspath(Path(path).expanduser())
    code = chr(10).join(("import sys", "print(sys.version.split()[0])", "try:",
                         " import torch", " print(torch.__version__)",
                         "except Exception as error:", " print('torch-unavailable: ' + str(error))"))
    try:
        probe = subprocess.run([resolved, "-c", code], capture_output=True, text=True,
                               timeout=10, check=False)
        if probe.returncode:
            raise RuntimeError(probe.stderr.strip() or f"exit status {probe.returncode}")
        versions = probe.stdout.strip().splitlines()
        version = f"Python {versions[0]}" if versions else "unavailable"
        if len(versions) > 1:
            version += f"; torch {versions[1]}"
    except Exception as error:
        return {"path": resolved, "version": "unavailable", "source": source,
                "verified": False, "verification_reason": f"{type(error).__name__}: {error}"}
    result = {"path": resolved, "version": version, "source": source, "verified": False}
    if verify:
        # The environment's CONTENT identity, not just its path. Bound into this dict so every
        # semantic key derived from it -- the run id, the rank-cache fingerprint, the feature
        # memmap fingerprints -- changes when an installed dependency's bytes change. Computed on
        # the interpreter the adapter actually invokes (never its resolved base) with that
        # interpreter's parent's parent as the prefix, matching `_record_environment_fingerprint`
        # and the GPU capability key.
        #
        # Verification computes this same digest internally: `verify_environment` reaches
        # `_cached_gpu_capability`, which needs the AUTHORITATIVE fingerprint to decide whether a
        # recorded capability still describes this environment. When that record matched, its
        # payload carries the value -- same function, same interpreter -- so it is reused here
        # instead of hashing every installed member a second time (~27,600 `_member_digest` calls,
        # ~10 s per model per composition run). `report.environment_fingerprint` is deliberately NOT
        # used: it is null on this path, and a cached report's value is derived from the
        # interpreter's SELF-REPORTED prefix, which need not equal the parent's-parent prefix the
        # capability key is built from.
        #
        # The internal computation is eager and is exposed only when a capability record matched,
        # so on a host with no verified capability (tier GPU_VERIFICATION_PENDING) the digest is
        # recomputed below; removing that remaining pass needs `runtime.py` to expose it, which is
        # outside this file's ownership.
        fingerprint = None
        try:
            report = verify_environment(model, probe=True, advisory=True, config=config)
            result.update(verified=report.tier in READY_TIERS,
                          verification_status=report.status, verification_reason=report.reason)
            capability = getattr(report, "gpu_capability", None)
            if isinstance(capability, dict) and report.interpreter == resolved:
                fingerprint = capability.get("interpreter_fingerprint")
        except Exception as error:
            result.update(verification_status="UNAVAILABLE",
                          verification_reason=f"{type(error).__name__}: {error}")
        result["environment_fingerprint"] = fingerprint or _record_environment_fingerprint(model, resolved)
    else:
        result["environment_fingerprint"] = None
        result["verification_status"] = "NOT_PROBED"
        result["verification_reason"] = "environment verification not run in plan mode"
    return result


def _result_outputs(artifacts: Path, protocol: str, model: str, checkpoint: str,
                    attempt: str, subset: str, condition: str, synthetic: bool = False) -> Path:
    base = artifacts / "composition" / "results" / protocol
    if synthetic:
        base /= "SYNTHETIC"
    return assert_composition_destination(artifacts, base / model / checkpoint / attempt / subset /
                                          f"{condition}.jsonl")

def _attempt_id(artifacts: Path, protocol: str, model: str, checkpoint: str,
                run_id: str, synthetic: bool, *, force_new: bool = False) -> str:
    """Keep exports paired by attempt without replacing an earlier result."""
    result_root = artifacts / "composition" / "results" / protocol
    if synthetic:
        result_root /= "SYNTHETIC"
    result_root = result_root / model / checkpoint
    summary_name = f"{'SYNTHETIC_' if synthetic else ''}{model}_{checkpoint}_{run_id}"
    summary_root = artifacts / "composition" / "summary"
    if not force_new and not (result_root / run_id).exists() and not (summary_root / summary_name).exists():
        return run_id
    retry_pattern = re.compile(re.escape(run_id) + r"-retry-(\d+)")
    suffixes = [int(match.group(1)) for path in result_root.glob(f"{run_id}-retry-*")
                if (match := retry_pattern.fullmatch(path.name))]
    suffixes += [int(match.group(1)) for path in summary_root.glob(f"{summary_name}-retry-*")
                 if (match := retry_pattern.search(path.name))]
    return f"{run_id}-retry-{max(suffixes, default=0) + 1}"


def _checkpoint_labels(model: dict[str, Any], checkpoint: str, protocol: str) -> dict[str, Any]:
    """Expose checkpoint provenance directly from its registry record."""
    variant = next((item for item in model.get("checkpoint_variants", [])
                    if item.get("checkpoint_id") == checkpoint), None)
    bundle = next((item for item in model.get("checkpoint_bundles", [])
                   if item.get("bundle_id") == checkpoint), None)
    if bundle:
        members = [item for item in model.get("checkpoint_variants", [])
                   if item.get("checkpoint_id") in bundle.get("required_checkpoint_ids", [])]
        variant = members[0] if members else None
    else:
        members = []
    training_dataset = variant.get("checkpoint_training_dataset") if variant else None
    evaluation_dataset = "FashionIQ" if "fashioniq" in protocol.lower() else None
    transfer = (f"{training_dataset}→{evaluation_dataset} transfer"
                if training_dataset and training_dataset.lower() != "fashioniq" and evaluation_dataset else None)
    labels = {"checkpoint_training_noise_pct": variant.get("checkpoint_training_noise_pct") if variant else None,
              "evaluation_noise_pct": variant.get("evaluation_noise_pct") if variant else None,
              "checkpoint_training_dataset": training_dataset,
              "checkpoint_transfer": transfer,
              "checkpoint_mapping_status": (bundle.get("mapping_status") if bundle else
                                            variant.get("checkpoint_mapping_status") if variant else None),
              "reproduction_status": model.get("reproduction_status"),
              "labels_source": "registry"}
    if bundle:
        labels["checkpoint_bundle_members"] = [
            {key: member.get(key) for key in ("checkpoint_id", "checkpoint_training_noise_pct",
                "checkpoint_training_dataset", "checkpoint_mapping_status")}
            for member in members]
    return labels



def _method_source_hash(method: Callable[..., Any]) -> str | None:
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return None
    return hashlib.sha256(source.encode("utf-8")).hexdigest()



def _module_source_hash(adapter: Any) -> str:
    module = sys.modules.get(type(adapter).__module__)
    try:
        return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    except (AttributeError, OSError, TypeError):
        return "unavailable"


def _image_identity(paths: Sequence[Path | None]) -> dict[str, Any]:
    resolved = [None if path is None else str(Path(path).expanduser().resolve()) for path in paths]
    file_stats = []
    for path in resolved:
        if path is None:
            file_stats.append(None)
            continue
        stat = Path(path).stat()
        file_stats.append((stat.st_size, stat.st_mtime_ns))
    return {"resolved_paths_sha256": _digest(resolved), "file_stats": file_stats,
            "stat_identity_method": "resolved absolute path plus size and mtime_ns; content identity approximation"}



def run_composition(data: Any, model_specs: Sequence[ModelSpec], config: Any, *,
                    subsets: Sequence[str], batch_size: int = 32, smoke: int | None = None,
                    resume: bool = False, conditions: Sequence[str] = CONDITIONS,
                    image_resolver: Callable[[str, Mapping[str, str]], Path | None] | None = None,
                    synthetic: bool | None = None, seed: int = 0) -> dict[str, Any]:
    """Evaluate models sequentially; gallery stays full and all outputs stay in composition/."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if smoke is not None and smoke < 1:
        raise ValueError("smoke query limit must be positive")
    if not conditions or any(item not in CONDITIONS for item in conditions):
        raise ValueError(f"conditions must be selected from {CONDITIONS}")
    artifacts = Path(config.WORKBENCH_ARTIFACTS_ROOT).expanduser().resolve()
    composition = assert_composition_destination(artifacts, artifacts / "composition")
    for subset in subsets:
        if subset not in {"full", "sf", "validated"}:
            raise ValueError(f"unknown subset {subset!r}")
    selected_subsets = set(subsets)
    queries = tuple(query for query in data.queries if selected_subsets.intersection(query.subsets))
    if smoke is not None:
        queries = queries[:smoke]
    if not queries:
        raise ValueError("release has no queries")
    gallery_ids = tuple(data.gallery.image_ids)
    gallery_hash = _gallery_fingerprint(data)
    query_hash = _query_fingerprint(queries)
    is_synthetic = bool(data.provenance.get("is_synthetic")) if synthetic is None else synthetic
    positive_positions = _positive_positions(data)
    reports = []

    for spec in model_specs:
        model_id = _component(spec.model_id)
        checkpoint_id = _component(spec.checkpoint_id)
        model = spec.registry_model
        labels = _checkpoint_labels(model, checkpoint_id, data.gallery.protocol_id)
        attempt = uuid.uuid4().hex
        resolver = image_resolver or image_resolver_factory(Path(config.FASHIONIQ_ROOT),
                                                             model.get("fashioniq_layout", "fashioniq_standard"))
        gallery_paths = [resolver(identity, data.candidate_names) for identity in gallery_ids]
        query_paths = [resolver(query.reference_id, data.candidate_names) for query in queries]
        missing_gallery = [gallery_ids[index] for index, path in enumerate(gallery_paths) if path is None]
        if missing_gallery:
            reason = f"{len(missing_gallery)} released gallery identities unresolved by stem"
            summaries = _write_failed_results(data, artifacts, model_id, checkpoint_id, attempt,
                subsets, conditions, queries, reason, is_synthetic, labels,
                execution_status="FAILED", run_id=attempt)
            reports.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                            "status": "BLOCKED_MISSING_GALLERY_IMAGES", "reason": reason,
                            "unresolvable_count": len(missing_gallery), "examples": missing_gallery[:20],
                            "summaries": summaries})
            continue
        if any(path is None for path in gallery_paths):
            raise AssertionError("gallery path check did not preserve complete released order")

        adapter = None
        adapter_source_hash = "unavailable"
        run_id = None
        work_root = None
        mode_status: dict[str, dict[str, Any]] = {}
        try:
            adapter = spec.adapter_factory(config, checkpoint_id)
            adapter_source_hash = _module_source_hash(adapter)
            runtime_identity = model_interpreter_identity(model, config, adapter)
            if not getattr(adapter, "reference_eligible", True):
                raise ValueError("adapter excludes the reference; CIRCUS requires reference-eligible ranks")
            scopes = tuple(getattr(adapter, "category_scopes", (None,)))
            if not scopes or len(set(scopes)) != len(scopes):
                raise ValueError(f"invalid adapter category_scopes: {scopes!r}")
            assignments = [_scope_for_query(adapter, query) for query in queries]
            query_scopes = [scope for scope, _reason in assignments]
            scope_errors = {query.qid: reason for query, (_scope, reason) in zip(queries, assignments) if reason}
            supported = {}
            for condition in conditions:
                try:
                    supported[condition] = adapter.mode_supported(condition)
                except Exception as error:
                    supported[condition] = (False, f"{type(error).__name__}: {error}")
            gpu_identity = _gpu_identity()
            checkpoint_identity = spec.checkpoint_identity or "unknown"
            identity = {
                "model_id": model_id, "checkpoint_id": checkpoint_id,
                "checkpoint_identity": checkpoint_identity, "protocol_id": data.gallery.protocol_id,
                "gallery_image_identity": _image_identity(gallery_paths),
                "query_image_identity": _image_identity(query_paths),
                "gallery_identity": gallery_hash, "query_order_identity": query_hash,
                "subset_memberships": sorted(set(subsets)), "conditions": list(conditions),
                "preprocessing": {"id": model.get("preprocessing_id"),
                                  "registry_values": {key: model.get(key) for key in
                                      ("fashioniq_layout", "fashioniq_image_extensions", "transform", "image_size",
                                       "text_processor", "candidate_normalization", "scoring")}},
                "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
                "adapter_source_hashes": {name: _method_source_hash(getattr(type(adapter), name))
                                           for name in ("encode_queries", "encode_targets", "score", "rank",
                                                        "mode_supported")},
                "adapter_module_source_hash": adapter_source_hash,
                "adapter_module_source_status": "available" if adapter_source_hash != "unavailable" else "unavailable",
                "score_signature": str(inspect.signature(adapter.score)),
                "mode_support": {condition: supported[condition] for condition in conditions},
                "runtime_identity": runtime_identity,
                "gpu_identity": gpu_identity,
                "category_scopes": list(scopes), "reference_eligible": True,
                "batch_size": batch_size, "seed": seed,
                # LIMN's own encoder batch (WORKBENCH_LIMN_BATCH_SIZE) changes the forward pass
                # only, but it is part of the run's configuration and must key the run id.
                "adapter_batch_size": getattr(adapter, "batch_size", None),
            }
            run_id = _digest(identity)[:24]
            fingerprint = _digest(identity)
            run_dir = assert_composition_destination(artifacts, composition / "runs" / run_id)
            work_root = assert_composition_destination(artifacts, run_dir / "work" / model_id)
            work_root.mkdir(parents=True, exist_ok=True)
            adapter.work_root = work_root
            rank_path = assert_composition_destination(artifacts, run_dir / "ranks" / f"{model_id}_{checkpoint_id}.json")
            attempt = _attempt_id(artifacts, data.gallery.protocol_id, model_id, checkpoint_id,
                                  run_id, is_synthetic, force_new=rank_path.exists())
            ranks: dict[str, dict[str, int | None]] | None = None
            reasons_by_condition: dict[str, dict[str, str]] = {}
            reused = False
            retry_conditions: set[str] = set()
            if resume and adapter_source_hash != "unavailable" and rank_path.is_file():
                try:
                    saved = json.loads(rank_path.read_text(encoding="utf-8"))
                    if saved.get("fingerprint") == fingerprint and set(saved.get("ranks", {})) == set(conditions):
                        saved_status = saved.get("condition_status", {})
                        saved_failures = saved.get("query_failures", {})
                        transient = {condition for condition in conditions
                                     if (saved_status.get(condition, {}).get("status") in
                                         {"MODEL_UNAVAILABLE", "FAILED", "EXACT_RANK_UNAVAILABLE"}
                                         or saved_status.get(condition, {}).get("status") == "PARTIAL" and
                                         any(not _is_stable_failure(reason)
                                             for reason in saved_failures.get(condition, {}).values()))
                                     and not (saved_failures.get(condition) and all(
                                         _is_stable_failure(reason)
                                         for reason in saved_failures[condition].values()))}
                        if transient:
                            retry_conditions = transient if len(transient) < len(conditions) else set(conditions)
                            ranks = {condition: dict(saved["ranks"][condition]) for condition in conditions}
                            reasons_by_condition = dict(saved.get("query_failures", {}))
                            mode_status = dict(saved_status)
                            for condition in retry_conditions:
                                ranks[condition] = {query.qid: None for query in queries}
                                reasons_by_condition.pop(condition, None)
                                mode_status.pop(condition, None)
                        else:
                            ranks = saved["ranks"]
                            reasons_by_condition = saved.get("query_failures", {})
                            mode_status = saved_status
                            reused = True
                            attempt = saved.get("attempt", run_id)
                except (OSError, ValueError, TypeError):
                    ranks = None
            if not reused:
                attempt = _attempt_id(artifacts, data.gallery.protocol_id, model_id, checkpoint_id,
                                      run_id, is_synthetic,
                                      force_new=rank_path.exists() and (not resume or bool(retry_conditions)))
            projection_totals: dict[str, int] = {}
            category_failure_counts: dict[str, int] = {}
            if ranks is None or retry_conditions:
                if ranks is None:
                    ranks = {condition: {query.qid: None for query in queries} for condition in conditions}
                active_conditions = [condition for condition in conditions if supported[condition][0]
                                     and (not retry_conditions or condition in retry_conditions)]
                for condition in conditions:
                    if not supported[condition][0] and condition not in mode_status:
                        mode_status[condition] = {"status": "ABLATION_UNSUPPORTED",
                                                  "reason": supported[condition][1]}
                target_maps: dict[str | None, Any] = {}
                target_errors: dict[str | None, str] = {}
                if active_conditions:
                    for scope_index, scope in enumerate(scopes):
                        scope_name = _scope_name(scope)
                        target_file = work_root / "features" / f"gallery_{scope_name}.npy"
                        target_fp = _digest({"run": fingerprint, "scope": scope, "kind": "full-gallery-features"})
                        try:
                            target_maps[scope], projection = _encode_gallery(
                                adapter, gallery_paths, scope, target_file, target_fp, batch_size,
                                len(scopes) - scope_index, len(queries), len(active_conditions))
                            projection_totals[scope_name] = projection.get("projected_bytes", 0)
                        except Exception as error:
                            target_errors[scope] = f"{type(error).__name__}: {error}"
                for condition in active_conditions:
                    condition_reasons: dict[str, str] = dict(scope_errors)
                    category_errors_all: dict[str, str] = {}
                    feature_errors_all: dict[str, str] = {}
                    query_features: dict[str | None, Any] = {}
                    for scope in scopes:
                        indexes = [index for index, query_scope in enumerate(query_scopes)
                                   if query_scope == scope and queries[index].qid not in scope_errors]
                        if not indexes or scope in target_errors or scope not in target_maps:
                            continue
                        scope_name = _scope_name(scope)
                        feature_file = work_root / "features" / f"query_{condition}_{scope_name}.npy"
                        feature_fp = _digest({"run": fingerprint, "condition": condition, "scope": scope,
                                              "kind": "query-features"})
                        if condition in retry_conditions:
                            for suffix in ("", ".json", ".errors.json", ".category-errors.json"):
                                feature_file.with_suffix(feature_file.suffix + suffix).unlink(missing_ok=True)
                        try:
                            values, errors, category_errors = _encode_queries(
                                adapter, queries, query_paths, indexes, condition, work_root,
                                feature_file, feature_fp, target_maps[scope].shape[-1], batch_size)
                            query_features[scope] = values
                            feature_errors_all.update(errors)
                            category_errors_all.update(category_errors)
                        except Exception as error:
                            feature_errors_all.update({queries[index].qid: f"{type(error).__name__}: {error}"
                                                       for index in indexes})
                    condition_reasons.update(feature_errors_all)
                    condition_reasons.update(category_errors_all)
                    for scope, error in target_errors.items():
                        for index, query_scope in enumerate(query_scopes):
                            if query_scope == scope:
                                condition_reasons[queries[index].qid] = error
                    for scope in scopes:
                        if scope not in query_features or scope not in target_maps:
                            continue
                        indices = [index for index, query_scope in enumerate(query_scopes)
                                   if query_scope == scope and queries[index].qid not in condition_reasons]
                        features = query_features[scope]
                        targets = target_maps if len(scopes) > 1 else target_maps[scope]
                        for offset in range(0, len(indices), batch_size):
                            batch_indices = indices[offset:offset + batch_size]
                            query_vectors = np.asarray(features[batch_indices], dtype=np.float32)
                            finite_rows = np.isfinite(query_vectors).all(axis=1)
                            valid_indices = [index for index, good in zip(batch_indices, finite_rows) if good]
                            invalid_indices = [index for index, good in zip(batch_indices, finite_rows) if not good]
                            for index in invalid_indices:
                                condition_reasons[queries[index].qid] = "query features unavailable"
                            if not valid_indices:
                                continue
                            positions = [positive_positions.get(queries[index].qid, ()) for index in valid_indices]
                            for index, positive in zip(valid_indices, positions):
                                if not positive:
                                    condition_reasons[queries[index].qid] = "no positive target resolved in exact released gallery"
                            keep = [offset for offset, positive in enumerate(positions) if positive]
                            valid_indices = [valid_indices[offset] for offset in keep]
                            positions = [positions[offset] for offset in keep]
                            if not valid_indices:
                                continue
                            vectors = np.asarray(features[valid_indices], dtype=np.float32)
                            cats = [queries[index].category for index in valid_indices]
                            try:
                                exact_ranks = adapter.rank(vectors, targets, positions, cats)
                                if len(exact_ranks) != len(valid_indices):
                                    raise ValueError("adapter.rank returned wrong number of query ranks")
                            except Exception as batch_error:
                                from workbench.composition.adapters import ExactRankUnavailable
                                if isinstance(batch_error, ExactRankUnavailable):
                                    for condition_query in queries:
                                        ranks[condition][condition_query.qid] = None
                                    mode_status[condition] = {"status": "EXACT_RANK_UNAVAILABLE",
                                                              "reason": str(batch_error)}
                                    condition_reasons = {}
                                    break
                                exact_ranks = []
                                for index, vector, positive, category in zip(valid_indices, vectors, positions, cats):
                                    try:
                                        one = adapter.rank(vector.reshape(1, -1), targets, [positive], [category])
                                        exact_ranks.append(one[0] if one else None)
                                    except ExactRankUnavailable as error:
                                        mode_status[condition] = {"status": "EXACT_RANK_UNAVAILABLE",
                                                                  "reason": str(error)}
                                        exact_ranks = []
                                        break
                                    except Exception as error:
                                        exact_ranks.append(None)
                                        condition_reasons[queries[index].qid] = f"{type(error).__name__}: {error}"
                            if mode_status.get(condition, {}).get("status") == "EXACT_RANK_UNAVAILABLE":
                                break
                            for index, rank in zip(valid_indices, exact_ranks):
                                if rank is not None and (type(rank) is not int or rank < 1 or rank > len(gallery_ids)):
                                    condition_reasons[queries[index].qid] = f"invalid full-gallery exact rank: {rank!r}"
                                    continue
                                ranks[condition][queries[index].qid] = rank
                        if mode_status.get(condition, {}).get("status") == "EXACT_RANK_UNAVAILABLE":
                            break
                    category_failure_counts[condition] = len(category_errors_all) + len(scope_errors)
                    reasons_by_condition[condition] = condition_reasons
                    scored_count = sum(rank is not None for rank in ranks[condition].values())
                    unscored_count = len(ranks[condition]) - scored_count
                    current_status = mode_status.get(condition, {}).get("status")
                    if current_status not in {"EXACT_RANK_UNAVAILABLE", "ABLATION_UNSUPPORTED"}:
                        if not scored_count:
                            reason = next(iter(condition_reasons.values()), "no query produced an exact rank")
                            mode_status[condition] = {"status": "MODEL_UNAVAILABLE", "reason": reason,
                                                      "unscored_count": unscored_count,
                                                      "category_scope_failure_count": len(category_errors_all) + len(scope_errors)}
                        else:
                            mode_status[condition] = {"status": "PARTIAL" if unscored_count else "COMPLETED",
                                                      "unscored_count": unscored_count,
                                                      "category_scope_failure_count": len(category_errors_all) + len(scope_errors)}
                _atomic_json(rank_path, {"fingerprint": fingerprint, "identity": identity,
                                         "attempt": attempt, "ranks": ranks,
                                         "query_failures": reasons_by_condition,
                                         "condition_status": mode_status})

            query_rows_by_condition = {}
            for condition in conditions:
                failures = reasons_by_condition.get(condition, {})
                query_rows_by_condition[condition] = [
                    {"qid": query.qid, "index": query.index, "condition": condition,
                     "rank": ranks.get(condition, {}).get(query.qid),
                     "scored": ranks.get(condition, {}).get(query.qid) is not None,
                     "failure": failures.get(query.qid), "synthetic": is_synthetic,
                     "checkpoint_labels": labels} for query in queries]
            from workbench.composition.metrics import compose
            summary_by_subset = {}
            for subset in subsets:
                selected = [query for query in queries if subset in query.subsets]
                qids = [query.qid for query in selected]
                subset_ranks = {condition: {qid: ranks.get(condition, {}).get(qid) for qid in qids}
                                for condition in conditions}
                result = compose(subset_ranks, query_count=len(qids))
                for condition in conditions:
                    output = _result_outputs(artifacts, data.gallery.protocol_id,
                                             model_id, checkpoint_id, attempt, subset, condition, is_synthetic)
                    _atomic_jsonl(output, [row for row in query_rows_by_condition[condition]
                                           if row["qid"] in set(qids)])
                summary_name = f"{'SYNTHETIC_' if is_synthetic else ''}{model_id}_{checkpoint_id}_{attempt}"
                summary_path = assert_composition_destination(artifacts, composition / "summary" /
                                                              summary_name / f"{subset}.json")
                transient_statuses = {"MODEL_UNAVAILABLE", "FAILED", "EXACT_RANK_UNAVAILABLE"}
                has_transient_failure = any(status.get("status") in transient_statuses
                                             for status in mode_status.values())
                has_success = any(status.get("status") in {"COMPLETED", "PARTIAL"}
                                  for status in mode_status.values())
                has_failure = has_transient_failure or any(status.get("status") in
                                    {"PARTIAL", "ABLATION_UNSUPPORTED"} for status in mode_status.values())
                execution_status = ("REUSED" if reused else "FAILED" if has_transient_failure and not has_success
                                    else "PARTIAL" if has_failure else "COMPLETED")
                summary = {
                    "checkpoint_labels": labels,
                    "synthetic": is_synthetic,
                    "adapter_module_source_status": identity["adapter_module_source_status"],
                    "schema_version": 1, "model_id": model_id, "checkpoint_id": checkpoint_id,
                    "checkpoint_identity": checkpoint_identity, "protocol_id": data.gallery.protocol_id,
                    "gallery_identity_sha256": gallery_hash, "gallery_count": len(gallery_ids),
                    "query_count": len(qids),
                    "query_count_per_subset": {name: sum(name in query.subsets for query in queries)
                                               for name in ("full", "sf", "validated")},
                    "subset": subset, "conditions": list(conditions),
                    "preprocessing": {"policy": "adapter-native transform after path-based raw-image intervention",
                                      "ablation_policy": "black PIL replacement is staged before adapter transform; empty text for image-only"},
                    "adapter": f"{type(adapter).__module__}.{type(adapter).__qualname__}",
                    "category_scopes": list(scopes),
                    "category_resolution": data.provenance.get("category_resolution", {}),
                    "unresolved_category_query_count": sum(query.category is None for query in queries),
                    "reference_eligible": True,
                    "runtime": {"controller_python": sys.version, "platform": platform.platform(),
                                "gpu": gpu_identity, "model_interpreter": runtime_identity},
                    "seed": seed, "release_artifact_hashes": data.provenance.get("sha256", {}),
                    "hash_verification": data.provenance.get("hash_verification"),
                    "failed_expectations": data.provenance.get("failed_expectations", ()),
                    "gallery_candidate_basename_match": data.provenance.get("gallery_candidate_basename_match"),
                    "paper_vs_release_gallery": data.provenance.get("gallery_note", {
                        "paper_stated_images": PAPER_STATED_GALLERY_IMAGES,
                        "released_images": RELEASED_GALLERY_IMAGES}),
                    "execution_status": execution_status,
                    "condition_status": mode_status,
                    "checkpoint_training_noise_pct": labels["checkpoint_training_noise_pct"],
                    "reproduction_status": labels["reproduction_status"],
                    "evaluation_noise_pct": labels["evaluation_noise_pct"],
                    "checkpoint_training_dataset": labels["checkpoint_training_dataset"],
                    "checkpoint_transfer": labels["checkpoint_transfer"],
                    "checkpoint_mapping_status": labels["checkpoint_mapping_status"],
                    "category_scope_failures": {condition: category_failure_counts.get(condition,
                                                       sum(reason.startswith(f"{model_id} category scope")
                                                           for reason in reasons_by_condition.get(condition, {}).values()))
                                                 for condition in conditions},
                    "smoke": {"limited": smoke is not None, "query_limit": smoke,
                              "label": "SMOKE QUERY LIMIT; full released gallery preserved" if smoke is not None else None},
                    "is_synthetic": is_synthetic, "synthetic_fixture": is_synthetic,
                    "feature_storage": {"format": "float32 .npy memmap", "projected_bytes": projection_totals,
                                        "arrays_removed_after_rank": True},
                    "metrics": asdict(result), "run_id": run_id, "resume_key": fingerprint,
                    "reused_query_result_count": len(queries) * len(conditions) if reused else 0,
                }
                _atomic_json(summary_path, summary)
                summary_by_subset[subset] = str(summary_path)
            successful_modes = {"COMPLETED", "PARTIAL"}
            successful = any(status.get("status") in successful_modes for status in mode_status.values())
            incomplete = (any(status.get("status") not in successful_modes for status in mode_status.values())
                          or any(status.get("status") == "PARTIAL" for status in mode_status.values()))
            model_status = ("MODEL_UNAVAILABLE" if not successful else
                            "PARTIAL" if incomplete else "REUSED" if reused else "COMPLETED")
            model_reason = next((status.get("reason") for status in mode_status.values()
                                 if status.get("status") not in successful_modes), None)
            reports.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                            "status": model_status, "reason": model_reason,
                            "run_id": run_id, "attempt": attempt,
                            "reused": reused, "reused_query_result_count": len(queries) * len(conditions) if reused else 0,
                            "query_count": len(queries), "gallery_count": len(gallery_ids),
                            "category_scopes": list(scopes), "condition_status": mode_status,
                            "adapter_module_source_status": identity["adapter_module_source_status"],
                            "runtime": {"model_interpreter": runtime_identity},
                            "summaries": summary_by_subset})
        except Exception as error:
            reason = f"{type(error).__name__}: {error}"
            summaries = _write_failed_results(data, artifacts, model_id, checkpoint_id, attempt,
                subsets, conditions, queries, reason, is_synthetic, labels,
                execution_status="FAILED", run_id=run_id if "run_id" in locals() else attempt)
            reports.append({"model_id": model_id, "checkpoint_id": checkpoint_id,
                            "status": "FAILED", "reason": reason,
                            "adapter_module_source_status": "available" if adapter_source_hash != "unavailable" else "unavailable",
                            "query_count": len(queries), "gallery_count": len(gallery_ids),
                            "run_id": locals().get("run_id"), "attempt": attempt, "summaries": summaries})
            mode_status[conditions[0]] = {"status": "FAILED", "reason": reason}
        finally:
            model_work = locals().get("work_root")
            if model_work is not None and model_work.exists():
                try:
                    if any(status.get("status") in {"MODEL_UNAVAILABLE", "FAILED", "EXACT_RANK_UNAVAILABLE"}
                           for status in mode_status.values()):
                        log_dir = assert_composition_destination(
                            artifacts, composition / "logs" / f"{model_id}_{checkpoint_id}" / str(attempt))
                        for worker_log in model_work.rglob("worker.stderr.log"):
                            try:
                                log_dir.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(worker_log, log_dir / "worker.stderr.log")
                            except Exception:
                                pass
                except Exception:
                    pass
                shutil.rmtree(model_work, ignore_errors=True)
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:
                    pass

    return {"run_count": len(reports), "models": reports, "gallery_count": len(gallery_ids),
            "gallery_identity_sha256": gallery_hash, "protocol_id": data.gallery.protocol_id,
            "synthetic": is_synthetic}


def image_resolver_factory(dataset_root: Path, layout: str):
    return image_resolver(dataset_root, layout)


def _write_failed_results(data: Any, artifacts: Path, model_id: str, checkpoint_id: str,
                          attempt: str, subsets: Sequence[str], conditions: Sequence[str],
                          queries: Sequence[Any], reason: str, synthetic: bool = False,
                          labels: dict[str, Any] | None = None, *, execution_status: str = "FAILED",
                          condition_status: dict[str, Any] | None = None,
                          run_id: str | None = None) -> dict[str, str]:
    from workbench.composition.metrics import compose

    summary_paths = {}
    for subset in subsets:
        selected = [query for query in queries if subset in query.subsets]
        qids = {query.qid for query in selected}
        failed_ranks = {condition: {query.qid: None for query in selected} for condition in conditions}
        for condition in conditions:
            output = _result_outputs(artifacts, data.gallery.protocol_id, model_id, checkpoint_id,
                                     attempt, subset, condition, synthetic)
            _atomic_jsonl(output, [{"qid": query.qid, "index": query.index,
                                    "condition": condition, "rank": None, "scored": False,
                                    "failure": reason, "synthetic": synthetic,
                                    "checkpoint_labels": labels or {}} for query in selected])
        summary_name = f"{'SYNTHETIC_' if synthetic else ''}{model_id}_{checkpoint_id}_{attempt}"
        summary_path = assert_composition_destination(artifacts, artifacts / "composition" / "summary" /
                                                      summary_name / f"{subset}.json")
        summary = {"synthetic": synthetic, "is_synthetic": synthetic, "model_id": model_id,
                   "checkpoint_id": checkpoint_id, "protocol_id": data.gallery.protocol_id,
                   "gallery_count": len(data.gallery.image_ids), "query_count": len(qids),
                   "subset": subset, "conditions": list(conditions), "execution_status": execution_status,
                   "condition_status": condition_status or {condition: {"status": execution_status,
                       "reason": reason, "unscored_count": len(selected)} for condition in conditions},
                   "checkpoint_labels": labels or {},
                   "checkpoint_training_noise_pct": (labels or {}).get("checkpoint_training_noise_pct"),
                   "evaluation_noise_pct": (labels or {}).get("evaluation_noise_pct"),
                   "checkpoint_training_dataset": (labels or {}).get("checkpoint_training_dataset"),
                   "checkpoint_transfer": (labels or {}).get("checkpoint_transfer"),
                   "checkpoint_mapping_status": (labels or {}).get("checkpoint_mapping_status"),
                   "reproduction_status": (labels or {}).get("reproduction_status"),
                   "metrics": asdict(compose(failed_ranks, query_count=len(selected))),
                   "run_id": run_id or attempt}
        _atomic_json(summary_path, summary)
        summary_paths[subset] = str(summary_path)
    return summary_paths


def inspect_image_inventory(data: Any, models: Sequence[dict[str, Any]], dataset_root: Path) -> dict[str, Any]:
    """Report missing released gallery, reference, and positive images without downloading."""
    gallery_ids = tuple(data.gallery.image_ids)
    references = {data.candidate_names.get(query.reference_id, query.reference_id)
                  for query in data.queries}
    positives = {data.candidate_names.get(identity, identity)
                 for query in data.queries for identity in query.positive_ids}
    missing_gallery: dict[str, list[str]] = {}
    missing_references: dict[str, list[str]] = {}
    missing_positives: dict[str, list[str]] = {}
    extension_counts: dict[str, int] = {}
    for model in models:
        model_id = str(model.get("model_id", "unknown"))
        resolver = image_resolver(dataset_root, model.get("fashioniq_layout", "fashioniq_standard"))

        def missing(values: Sequence[str] | set[str]) -> list[str]:
            return sorted(identity for identity in values
                          if resolver(identity, data.candidate_names) is None)

        missing_gallery[model_id] = missing(gallery_ids)
        missing_references[model_id] = missing(references)
        missing_positives[model_id] = missing(positives)
        for identity in gallery_ids:
            path = resolver(identity, data.candidate_names)
            if path is not None:
                original = PurePosixPath(normalize_release_path(
                    data.candidate_names.get(identity, identity))).name
                key = "exact" if original == path.name else path.suffix.lower()
                extension_counts[key] = extension_counts.get(key, 0) + 1

    gallery_missing = {identity for values in missing_gallery.values() for identity in values}
    reference_missing = {identity for values in missing_references.values() for identity in values}
    positive_missing = {identity for values in missing_positives.values() for identity in values}
    status = "missing" if gallery_missing or reference_missing or positive_missing else (
        "extension_mismatch" if any(key != "exact" for key in extension_counts) else "complete")
    return {"status": status, "released_gallery_count": len(gallery_ids),
            "local_resolved_by_stem_count": len(gallery_ids) - len(gallery_missing),
            "unresolvable_gallery_count": len(gallery_missing),
            "unresolvable_gallery_examples": sorted(gallery_missing)[:20],
            "missing_gallery_by_model": missing_gallery,
            "query_reference_count": len(references),
            "query_references_resolved_count": len(references) - len(reference_missing),
            "query_reference_missing_count": len(reference_missing),
            "query_reference_missing_examples": sorted(reference_missing)[:20],
            "query_reference_missing_by_model": missing_references,
            "positive_target_count": len(positives),
            "positive_targets_resolved_count": len(positives) - len(positive_missing),
            "positive_target_missing_count": len(positive_missing),
            "positive_target_missing_examples": sorted(positive_missing)[:20],
            "positive_target_missing_by_model": missing_positives,
            "extension_resolution_counts": extension_counts, "downloaded": False}
