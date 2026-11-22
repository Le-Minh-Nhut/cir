"""Two-process ENCODER capture controller; no model inference in preflight."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from workbench.backend.operator_config import resolve_config
from workbench.backend.runtime import resolve_model_interpreter
from workbench.backend.registry import model_by_id
from workbench.composition.model_encoder import Adapter, RELATION_SEED
from workbench.replay.encoder import (AUXILIARY_ASSET_NAME, DEFAULT_CHECKPOINT, _NativeArgsortCapture)

CATEGORIES = ("dress", "shirt", "toptee")
MODEL = model_by_id("encoder")
ROOT = Path(__file__).resolve().parents[2]
CONTROLLER = Path(sys.executable).absolute()
DEFAULT_NATIVE_BATCH_SIZE = 64
DEFAULT_ADAPTER_BATCH_SIZE = 32
# Shirt Q/G counts project 10.31 GB; 12 GB bounds native and Adapter G1 shards.
DISK_FINAL_PER_ROLE_BYTES = 12_000_000_000
# Keep 4 GiB beyond the observed final root and native staging peak.
DISK_SAFETY_RESERVE_BYTES = 4 << 30

def _staging_peak_bytes(dataset: Path) -> int:
    from workbench.replay.tensor_spool import expected_image_count

    image_bytes = expected_image_count(dataset) * 3 * 224 * 224 * 4
    return 2 * image_bytes + (1 << 30)


def _inputs(config: Any) -> tuple[Path, Path, Path]:
    adapter = Adapter(config)
    return adapter.source_root, adapter.dataset_root, adapter._checkpoint_path


def _worker(config: Any) -> Path:
    return Path(resolve_model_interpreter(MODEL, config) or "").absolute()


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _worker_env() -> dict[str, str]:
    pythonpath = os.pathsep.join(filter(None, (str(ROOT), os.environ.get("PYTHONPATH", ""))))
    return {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": pythonpath}


def _run_native(command: list[str], source: Path, run_root: Path) -> subprocess.CompletedProcess:
    env = {**_worker_env(), "WORKBENCH_ENCODER_SCORE_OBSERVATION_OUT": str(run_root / "native-scores"),
           "WORKBENCH_ENCODER_G1_OBSERVATION_OUT": str(run_root / "native-g1")}
    return subprocess.run(command, cwd=source.resolve(), env=env, check=False)

def _worker_probe(worker: Path) -> dict[str, Any]:
    code = ("import hashlib,importlib.metadata as m,json,numpy,open_clip,torch; "
            "p=open_clip.tokenizer.default_bpe(); "
            "print(json.dumps({'versions':numpy.__version__+' '+torch.__version__+' '+m.version('open_clip_torch'),"
            "'tokenizer_asset':{'path':p,'sha256':hashlib.sha256(open(p,'rb').read()).hexdigest()}}))")
    result = subprocess.run([str(worker), "-c", code], cwd=ROOT, env=_worker_env(),
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"worker import probe failed: {result.stderr.strip()}")
    return {"python": str(worker), **json.loads(result.stdout.strip())}


def _native_preflight(worker: Path, checkpoint: Path, dataset: Path, source: Path,
                      category: str, native_batch_size: int) -> dict[str, Any]:
    command = [str(worker), "-m", "workbench.replay.encoder", "--checkpoint-root", str(checkpoint.parent),
               "--dataset-root", str(dataset), "--source-root", str(source), "--category", category,
               "--batch-size", str(native_batch_size), "--preflight-only"]
    result = subprocess.run(command, cwd=source.resolve(), env=_worker_env(), capture_output=True,
                            text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"native CLI preflight failed ({result.returncode}): {result.stderr.strip()}")
    return {"argv": command, "exit_code": result.returncode, "stdout": result.stdout.strip()}


def preflight(category: str = "dress", native_batch_size: int = DEFAULT_NATIVE_BATCH_SIZE,
              adapter_batch_size: int = DEFAULT_ADAPTER_BATCH_SIZE, repeat_id: str = "repeat-1") -> dict[str, Any]:
    if category not in CATEGORIES:
        raise ValueError(f"unsupported ENCODER category: {category}")
    if native_batch_size < 1 or adapter_batch_size < 1:
        raise ValueError("ENCODER batch sizes must be positive")
    if not repeat_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for char in repeat_id):
        raise ValueError("ENCODER repeat ID must be a non-empty filesystem-safe token")
    config = resolve_config()
    source, dataset, checkpoint = _inputs(config)
    adapter = Adapter(config)
    verified_worker = Path(adapter.require_verified_interpreter()).absolute()
    worker = _worker(config)
    if verified_worker != worker:
        raise RuntimeError(f"Adapter verified interpreter differs from native worker: {verified_worker} != {worker}")
    status = Adapter.status(config)
    if not status.ready or status.checkpoint_ids != (DEFAULT_CHECKPOINT["checkpoint_id"],):
        raise RuntimeError(f"Adapter preflight blocked: {status.reason or status.checkpoint_ids}")
    native = _native_preflight(worker, checkpoint, dataset, source, category, native_batch_size)
    worker_info = _worker_probe(worker)
    checkpoint_digest = _sha(checkpoint)
    if checkpoint_digest != DEFAULT_CHECKPOINT["expected_sha256"]:
        raise ValueError("selected ENCODER checkpoint digest differs from pin")
    source_assets = {name: _sha(source / name) for name in ("test.py", "model_try2.py", "datasets.py")}
    dataset_assets = {name: _sha(dataset / name) for name in (
        f"image_splits/split.{category}.val.json", f"captions/cap.{category}.val.json",
        f"captions/correction_dict_{category}.json")}
    auxiliary = source / AUXILIARY_ASSET_NAME
    if not auxiliary.is_file():
        raise FileNotFoundError(f"ENCODER auxiliary asset missing: {auxiliary}")
    return {
        "schema": "encoder-capture-plan-v1", "status": "ready_for_review_not_authorized_to_run",
        "category": category, "repeat_id": repeat_id, "repeat_seed": 0,
        "native_batch_size": native_batch_size, "adapter_batch_size": adapter_batch_size,
        "native_preflight": native, "native_cwd": str(source.resolve()), "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_digest, "checkpoint_mapping_status": DEFAULT_CHECKPOINT["checkpoint_mapping_status"],
        "source_root": str(source), "source_commit": MODEL["upstream_commit_sha"],
        "source_assets_sha256": source_assets,
        "auxiliary_asset": {"path": str(auxiliary), "sha256": _sha(auxiliary)},
        "dataset_root": str(dataset), "dataset_assets_sha256": dataset_assets,
        "native_worker": worker_info,
        "relation_seed": RELATION_SEED,
        "native_argv_template": [str(worker), "-m", "workbench.replay.encoder", "--checkpoint-root",
                                 str(checkpoint.parent), "--dataset-root", str(dataset), "--source-root", str(source),
                                 "--category", category, "--batch-size", str(native_batch_size), "--device", "cuda",
                                 "--output", "<RUN_ROOT>/native-result.json"],
        "query_transforms": {"native": "preprocess_train", "adapter": "preprocess_val",
                             "meaning": "diagnostic-only; pixel parity not claimed"},
    }



def _verify_image_manifest(dataset: Path, ids: dict[str, Any], paths: list[str],
                          digests: list[str], recorded: dict[str, str]) -> dict[str, str]:
    expected_paths = sorted(set(ids["query_paths"]) | set(ids["gallery_paths"]))
    if (paths != expected_paths or len(digests) != len(expected_paths)
            or recorded != dict(zip(paths, digests))):
        raise ValueError("native image manifest order, coverage, or digest differs from source IDs")
    category = str(ids["category"])
    image_root = (dataset / "resized_image" / category).resolve()
    expected_query_paths = [str(image_root / f"{image_id}.jpg") for image_id in ids["reference_ids"]]
    expected_gallery_paths = [str(image_root / f"{image_id}.jpg") for image_id in ids["gallery_ids"]]
    if ids["query_paths"] != expected_query_paths or ids["gallery_paths"] != expected_gallery_paths:
        raise ValueError("native image paths do not resolve to canonical FashionIQ inputs")
    for path, digest in recorded.items():
        image_path = Path(path)
        if not image_path.is_file() or _sha(image_path) != digest:
            raise ValueError(f"native input image content changed: {path}")
    return recorded

def _adapter_g1_manifest(path: Path) -> tuple[dict[str, Any], list[Path]]:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != "encoder-adapter-g1-shards-v1":
        raise ValueError("unsupported ENCODER Adapter G1 shard schema")
    names = manifest.get("shards")
    sizes = manifest.get("batch_sizes")
    paths = manifest.get("image_paths")
    texts = manifest.get("texts")
    if (not isinstance(names, list) or not names
            or names != [f"batch-{index:05d}.npz" for index in range(len(names))]
            or not isinstance(sizes, list) or len(sizes) != len(names)
            or any(type(size) is not int or size <= 0 for size in sizes)
            or not isinstance(paths, list) or len(paths) != sum(sizes)
            or any(not isinstance(item, str) for item in paths)
            or not isinstance(texts, list)
            or len(texts) != (sum(sizes) if manifest.get("phase") == "queries" else 0)
             or any(not isinstance(item, str) for item in texts)
            or manifest.get("phase") not in ("queries", "targets")):
        raise ValueError("invalid ENCODER Adapter G1 shard manifest")
    shards = [path / name for name in names]
    if any(not shard.is_file() for shard in shards):
        raise ValueError("missing ENCODER Adapter G1 shard")
    return manifest, shards


def _adapter_g1_shard_hashes(path: Path) -> dict[str, str]:
    _, shards = _adapter_g1_manifest(path)
    files = [path / "manifest.json", *shards]
    return {str(file.relative_to(path.parent)): _sha(file) for file in files}

def _compare_g1_inputs(root: Path, ids: dict[str, Any], native_batch_size: int,
                       adapter_batch_size: int) -> dict[str, Any]:
    import numpy as np

    native_root = root / "native-g1"
    adapter_root = root / "g1"
    query_files = sorted(native_root.glob("query-*.npz"))
    gallery_files = sorted(native_root.glob("gallery-*.npz"))
    query_adapter_root = adapter_root / "adapter-queries"
    gallery_adapter_root = adapter_root / "adapter-gallery"
    query_manifest, query_shards = _adapter_g1_manifest(query_adapter_root)
    gallery_manifest, gallery_shards = _adapter_g1_manifest(gallery_adapter_root)
    query_delta = gallery_delta = 0.0
    query_equal = gallery_equal = token_ids_equal = True
    native_query_batches: list[int] = []
    native_gallery_batches: list[int] = []
    if (query_manifest.get("phase") != "queries"
            or query_manifest.get("image_paths") != ids["query_paths"]
            or query_manifest.get("texts") != ids["query_texts"]):
        raise ValueError("ENCODER Adapter G1 query order differs from frozen IDs")
    if (gallery_manifest.get("phase") != "targets"
            or gallery_manifest.get("image_paths") != ids["gallery_paths"]):
        raise ValueError("ENCODER Adapter G1 gallery order differs from frozen IDs")
    query_batch_sizes = query_manifest["batch_sizes"]
    gallery_batch_sizes = gallery_manifest["batch_sizes"]
    if (query_batch_sizes != [min(adapter_batch_size, len(ids["query_paths"]) - i)
                              for i in range(0, len(ids["query_paths"]), adapter_batch_size)]
            or gallery_batch_sizes != [min(adapter_batch_size, len(ids["gallery_paths"]) - i)
                                       for i in range(0, len(ids["gallery_paths"]), adapter_batch_size)]):
        raise ValueError("ENCODER G1 observed batch partitions differ from frozen batch sizes")

    def compare_stream(native_files: list[Path], adapter_files: list[Path],
                       adapter_batch_sizes: list[int], phase: str) -> None:
        nonlocal query_delta, gallery_delta
        nonlocal query_equal, gallery_equal, token_ids_equal
        offset = 0
        adapter_index = adapter_offset = 0
        for native_path in native_files:
            with np.load(native_path, allow_pickle=False) as native:
                id_key = "target_ids" if phase == "query" else "gallery_ids"
                count = len(native[id_key])
                stop = offset + count
                adapter_pixels_parts = []
                adapter_tokens_parts = []
                adapter_paths: list[str] = []
                adapter_texts: list[str] = []
                remaining = count
                while remaining:
                    with np.load(adapter_files[adapter_index], allow_pickle=False) as shard:
                        shard_size = len(shard["image_paths"])
                        if (shard_size != adapter_batch_sizes[adapter_index]
                                or shard["phase"].item() != ("queries" if phase == "query" else "targets")):
                            raise ValueError("ENCODER Adapter G1 shard differs from manifest")
                        take = min(remaining, shard_size - adapter_offset)
                        sl = slice(adapter_offset, adapter_offset + take)
                        adapter_pixels_parts.append(shard["pixels"][sl])
                        adapter_paths.extend(shard["image_paths"][sl].tolist())
                        if phase == "query":
                            adapter_tokens_parts.append(shard["input_ids"][sl])
                            adapter_texts.extend(shard["texts"][sl].tolist())
                        remaining -= take
                        adapter_offset += take
                        if adapter_offset == shard_size:
                            adapter_index += 1
                            adapter_offset = 0
                pixels = native["pixels"]
                other_pixels = np.concatenate(adapter_pixels_parts, axis=0)
                expected_paths = ids["query_paths" if phase == "query" else "gallery_paths"][offset:stop]
                if adapter_paths != expected_paths or native["image_paths"].tolist() != expected_paths:
                    raise ValueError(f"ENCODER {phase} G1 image order differs from frozen IDs")
                if phase == "query":
                    for key in ("target_ids", "reference_ids", "reference_positions", "target_positions"):
                        if native[key].tolist() != ids[key][offset:stop]:
                            raise ValueError("ENCODER native G1 query identity differs from frozen IDs")
                    if (native["texts"].tolist() != ids["query_texts"][offset:stop]
                            or adapter_texts != ids["query_texts"][offset:stop]):
                        raise ValueError("ENCODER query G1 text differs from frozen IDs")
                    tokens = native["input_ids"]
                    other_tokens = np.concatenate(adapter_tokens_parts, axis=0)
                    token_ids_equal = token_ids_equal and tokens.shape == other_tokens.shape and bool(
                        np.array_equal(tokens, other_tokens))
                elif native["gallery_ids"].tolist() != ids["gallery_ids"][offset:stop]:
                    raise ValueError("ENCODER native G1 gallery identity differs from frozen IDs")
                if pixels.shape != other_pixels.shape or not np.isfinite(pixels).all() or not np.isfinite(other_pixels).all():
                    raise ValueError(f"ENCODER {phase} G1 pixel shapes or values invalid")
                delta = float(np.max(np.abs(pixels - other_pixels), initial=0.0))
                if phase == "query":
                    query_delta = max(query_delta, delta)
                    query_equal = query_equal and bool(np.array_equal(pixels, other_pixels))
                    native_query_batches.append(count)
                else:
                    gallery_delta = max(gallery_delta, delta)
                    gallery_equal = gallery_equal and bool(np.array_equal(pixels, other_pixels))
                    native_gallery_batches.append(count)
                offset = stop
        expected = len(ids["query_paths"] if phase == "query" else ids["gallery_paths"])
        if offset != expected or adapter_index != len(adapter_files) or adapter_offset:
            raise ValueError(f"ENCODER {phase} G1 shard coverage differs from frozen IDs")

    compare_stream(query_files, query_shards, query_batch_sizes, "query")
    compare_stream(gallery_files, gallery_shards, gallery_batch_sizes, "gallery")
    expected_native_query = [min(native_batch_size, len(ids["query_paths"]) - i)
                             for i in range(0, len(ids["query_paths"]), native_batch_size)]
    expected_native_gallery = [min(native_batch_size, len(ids["gallery_paths"]) - i)
                               for i in range(0, len(ids["gallery_paths"]), native_batch_size)]
    if native_query_batches != expected_native_query or native_gallery_batches != expected_native_gallery:
        raise ValueError("ENCODER G1 observed native batch partitions differ from frozen batch sizes")
    return {"status": "measured", "native_query_batch_sizes": native_query_batches,
            "native_gallery_batch_sizes": native_gallery_batches,
            "adapter_query_batch_sizes": query_batch_sizes,
            "adapter_gallery_batch_sizes": gallery_batch_sizes,
            "query_pixels_equal": query_equal, "query_pixel_max_abs_delta": query_delta,
            "query_text_equal": True, "query_input_ids_equal": token_ids_equal,
            "gallery_pixels_equal": gallery_equal, "gallery_pixel_max_abs_delta": gallery_delta,
            "query_reference_ids": ids["reference_ids"], "query_target_ids": ids["target_ids"],
            "gallery_paths": ids["gallery_paths"]}

def _verify_native_g1(root: Path, ids: dict[str, Any], batch_size: int) -> dict[str, str]:
    import numpy as np

    files: dict[str, str] = {}
    for phase, expected_paths in (("query", ids["query_paths"]), ("gallery", ids["gallery_paths"])):
        count = len(expected_paths)
        for index, start in enumerate(range(0, count, batch_size)):
            relative = f"native-g1/{phase}-{index:05d}.npz"
            path = root / relative
            stop = min(start + batch_size, count)
            with np.load(path, allow_pickle=False) as archive:
                pixels = archive["pixels"]
                if (pixels.ndim != 4 or pixels.shape[:2] != (stop - start, 3)
                        or not np.isfinite(pixels).all()
                        or archive["image_paths"].tolist() != expected_paths[start:stop]):
                    raise ValueError(f"native G1 {phase} pixels/paths invalid: {relative}")
                if phase == "query":
                    for key in ("reference_ids", "target_ids", "reference_positions", "target_positions", "texts"):
                        expected = ids["query_texts"][start:stop] if key == "texts" else ids[key][start:stop]
                        if archive[key].tolist() != expected:
                            raise ValueError(f"native G1 query {key} mismatch: {relative}")
                    tokens = archive["input_ids"]
                    if tokens.ndim != 2 or tokens.shape[0] != stop - start:
                        raise ValueError(f"native G1 query token shape mismatch: {relative}")
                elif archive["gallery_ids"].tolist() != ids["gallery_ids"][start:stop]:
                    raise ValueError(f"native G1 gallery IDs mismatch: {relative}")
            files[relative] = _sha(path)
    return files

def bind_receipt(root: Path, plan: dict[str, Any], argv: list[str], returncode: int) -> dict[str, Any]:
    root = root.absolute()
    expected_argv = [*plan["native_argv_template"][:-1], str(root / "native-result.json")]
    if argv != expected_argv:
        raise ValueError("native command differs from frozen capture contract")

    if returncode != 0:
        raise RuntimeError(f"native stage failed: exit={returncode}")
    if (root / "native-receipt.json").exists():
        raise FileExistsError(root / "native-receipt.json")
    plan_path = root / "frozen-plan.json"
    if not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("native frozen plan file mismatch")
    category = plan.get("category")
    if (plan.get("schema") != "encoder-capture-plan-v1"
            or plan.get("status") != "ready_for_review_not_authorized_to_run"
            or category not in CATEGORIES):
        raise ValueError("native frozen plan identity mismatch")
    worker_identity = plan.get("native_worker")
    if (not isinstance(worker_identity, dict)
            or worker_identity.get("python") != plan["native_argv_template"][0]
            or not worker_identity.get("versions")):
        raise ValueError("native worker identity is missing or differs from frozen command")
    tokenizer_asset = worker_identity.get("tokenizer_asset")
    if (not isinstance(tokenizer_asset, dict) or not tokenizer_asset.get("path")
            or _sha(Path(tokenizer_asset["path"])) != tokenizer_asset.get("sha256")):
        raise ValueError("OpenCLIP tokenizer BPE asset changed since frozen plan")
        raise ValueError("native frozen plan identity mismatch")
    result_path = root / "native-result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result_entry = next((item for item in result["results"] if item["category"] == category), None)
    if result_entry is None:
        raise ValueError("native result lacks selected category")
    capture = result_entry["provenance"].get("native_score_capture")
    relation = result_entry["provenance"].get("relation_state")
    if not capture or not relation or relation.get("status") != "recorded":
        raise ValueError("native result lacks source-captured scores or relation tensor")
    files = {"native-result.json": result_path}
    files.update({f"native-scores/{Path(capture[key]).name}": Path(capture[key]).resolve()
                  for key in ("score_path", "order_path", "ids_path")})
    relation_path = Path(relation["path"]).resolve()
    relation_meta_path = Path(str(relation_path) + ".json")
    files[relation_path.name] = relation_path
    files[relation_meta_path.name] = relation_meta_path
    for path in files.values():
        if root not in path.parents or not path.is_file():
            raise ValueError(f"native artifact missing or outside immutable root: {path}")
    import numpy as np
    ids_path = files["native-scores/source_ids.npz"]
    scores = np.load(files["native-scores/native_scores.float32.npy"], mmap_mode="r")
    orders = np.load(files["native-scores/native_orders.int32.npy"], mmap_mode="r")
    with np.load(ids_path, allow_pickle=False) as archive:
        ids = {key: archive[key].tolist() for key in ("category", "gallery_ids", "reference_ids", "target_ids",
                "reference_positions", "target_positions", "query_texts", "query_paths", "gallery_paths")}
        if archive["category"].shape != () or archive["category"].item() != category:
            raise ValueError("native archive category differs from frozen plan")
        image_manifest_paths = archive["image_manifest_paths"].tolist()
        image_manifest_sha256 = archive["image_manifest_sha256"].tolist()
    image_content_sha256 = _verify_image_manifest(
        Path(plan["dataset_root"]), ids, image_manifest_paths, image_manifest_sha256,
        dict(zip(image_manifest_paths, image_manifest_sha256)))
    expected_shape = (len(ids["target_ids"]), len(ids["gallery_ids"]))
    if scores.dtype != np.float32 or orders.dtype != np.int32 or scores.shape != expected_shape or orders.shape != expected_shape:
        raise ValueError("native score/order arrays disagree with frozen ID dimensions")
    if not np.isfinite(scores).all() or not np.array_equal(np.sort(orders, axis=1),
            np.broadcast_to(np.arange(expected_shape[1], dtype=np.int32), expected_shape)):
        raise ValueError("native score/order arrays contain invalid values")
    g1_files = _verify_native_g1(root, ids, plan["native_batch_size"])
    expected_g1_count = ((len(ids["target_ids"]) + plan["native_batch_size"] - 1) // plan["native_batch_size"]
                         + (len(ids["gallery_ids"]) + plan["native_batch_size"] - 1) // plan["native_batch_size"])
    if len(g1_files) != expected_g1_count:
        raise ValueError("native G1 capture batch count differs from frozen IDs and batch size")
    record = {
        "schema": "encoder-native-receipt-v1", "run_root": str(root), "plan_sha256": _canonical_sha(plan),
        "plan": plan, "argv": argv, "interpreter": argv[0], "returncode": returncode,
        "artifact_sha256": {name: _sha(path) for name, path in files.items()},
        "ordered_ids_text_paths_positions_sha256": _canonical_sha(ids),
        "query_content_sha256": _canonical_sha(list(zip(ids["reference_ids"], ids["target_ids"], ids["query_texts"]))),
        "image_content_sha256": image_content_sha256,
        "g1_artifact_sha256": g1_files,
        "native_relation": relation,
        "relation_tensor_content_sha256": hashlib.sha256(
            np.ascontiguousarray(np.load(relation_path, allow_pickle=False)).tobytes()).hexdigest(),
        "relation_metadata_file_sha256": _sha(relation_meta_path),
    }
    with (root / "native-receipt.json").open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    return record

def disk_admission(roots: list[Path], *, staging_root: Path, staging_peak_bytes: int,
                   sunk_roots: list[Path] | None = None,
                   final_per_role_bytes: int = DISK_FINAL_PER_ROLE_BYTES,
                   safety_reserve_bytes: int = DISK_SAFETY_RESERVE_BYTES) -> dict[str, Any]:
    roots = [root.absolute() for root in roots]
    sunk_roots = [root.absolute() for root in sunk_roots or []]
    if not roots or len(set(roots + sunk_roots)) != len(roots) + len(sunk_roots):
        raise ValueError("disk admission requires unique planned and sunk run roots")
    if staging_peak_bytes <= 0 or final_per_role_bytes <= 0 or safety_reserve_bytes < 0:
        raise ValueError("disk admission budgets must be positive")
    output_anchor = roots[0].parent
    staging_root = staging_root.absolute()
    output_usage = shutil.disk_usage(output_anchor)
    staging_usage = shutil.disk_usage(staging_root)
    missing_bytes = existing_bytes = 0
    for root in roots + sunk_roots:
        if root.exists() and not root.is_dir():
            raise FileExistsError(f"planned immutable run root is not a directory: {root}")
        if root not in sunk_roots and not root.exists():
            missing_bytes += final_per_role_bytes
            continue
        if root.exists():
            role_bytes = 0
            for directory, subdirs, filenames in os.walk(root, followlinks=False):
                subdirs[:] = [name for name in subdirs if not (Path(directory) / name).is_symlink()]
                for name in filenames:
                    metadata = (Path(directory) / name).lstat()
                    if stat.S_ISREG(metadata.st_mode):
                        role_bytes += metadata.st_blocks * 512
            existing_bytes += role_bytes
            if root in sunk_roots or (root / "native-receipt.json").is_file():
                continue
            missing_bytes += max(0, final_per_role_bytes - role_bytes)
    same_filesystem = os.stat(output_anchor).st_dev == os.stat(staging_root).st_dev
    output_required = missing_bytes + safety_reserve_bytes
    staging_required = staging_peak_bytes + safety_reserve_bytes
    if same_filesystem:
        output_required += staging_peak_bytes
        staging_required = output_required
    result = {"status": "admitted" if output_usage.free >= output_required
              and staging_usage.free >= staging_required else "blocked_insufficient_space",
              "same_filesystem": same_filesystem, "output_filesystem": str(output_anchor),
              "output_free_bytes": output_usage.free, "planned_roots": [str(root) for root in roots],
              "sunk_roots": [str(root) for root in sunk_roots], "existing_bytes": existing_bytes,
              "staging_peak_bytes": staging_peak_bytes, "safety_reserve_bytes": safety_reserve_bytes,
              "output_required_bytes": output_required, "staging_filesystem": str(staging_root),
              "staging_free_bytes": staging_usage.free, "staging_required_bytes": staging_required}
    if output_usage.free < output_required or staging_usage.free < staging_required:
        raise OSError("insufficient disk for ENCODER capture: " + json.dumps(result, sort_keys=True))
    return result


def verify_receipt(root: Path, receipt: dict[str, Any]) -> tuple[dict[str, Any], Any, dict[str, Any]]:
    import numpy as np

    root = root.resolve()
    receipt_path = root / "native-receipt.json"
    if not receipt_path.is_file() or json.loads(receipt_path.read_text(encoding="utf-8")) != receipt:
        raise ValueError("native receipt file differs from supplied receipt")
    root = root.resolve()
    if receipt.get("schema") != "encoder-native-receipt-v1" or receipt.get("run_root") != str(root):
        raise ValueError("native receipt belongs to another run root")
    plan = receipt.get("plan")
    if not isinstance(plan, dict) or receipt.get("plan_sha256") != _canonical_sha(plan):
        raise ValueError("native frozen plan digest mismatch")
    plan_path = root / "frozen-plan.json"
    if not plan_path.is_file() or json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise ValueError("native frozen plan file mismatch")
    category = plan.get("category")
    if (plan.get("schema") != "encoder-capture-plan-v1" or category not in CATEGORIES
            or type(plan.get("native_batch_size")) is not int or plan["native_batch_size"] < 1
            or type(plan.get("adapter_batch_size")) is not int or plan["adapter_batch_size"] < 1
            or not plan.get("repeat_id") or plan.get("repeat_seed") != 0):
        raise ValueError("native frozen plan category, repeat, or batch identity mismatch")
    source, dataset, checkpoint = (Path(plan[key]) for key in ("source_root", "dataset_root", "checkpoint_path"))
    worker_identity = plan.get("native_worker")
    if (not isinstance(worker_identity, dict)
            or worker_identity.get("python") != plan["native_argv_template"][0]
            or not worker_identity.get("versions")):
        raise ValueError("native worker identity is missing or differs from frozen command")
    expected_argv = [*plan["native_argv_template"][:-1], str(root / "native-result.json")]
    if (receipt.get("interpreter") != worker_identity["python"]
            or receipt.get("argv") != expected_argv or receipt.get("returncode") != 0):
        raise ValueError("native argv/interpreter/exit mismatch")
    if plan.get("source_assets_sha256") != {name: _sha(source / name) for name in ("test.py", "model_try2.py", "datasets.py")}:
        raise ValueError("ENCODER source assets changed since frozen plan")
    if _sha(checkpoint) != plan.get("checkpoint_sha256"):
        raise ValueError("ENCODER checkpoint changed since frozen plan")
    if plan.get("dataset_assets_sha256") != {
            f"image_splits/split.{category}.val.json": _sha(dataset / f"image_splits/split.{category}.val.json"),
            f"captions/cap.{category}.val.json": _sha(dataset / f"captions/cap.{category}.val.json"),
            f"captions/correction_dict_{category}.json": _sha(dataset / f"captions/correction_dict_{category}.json")}:
        raise ValueError("FashionIQ annotations/split changed since frozen plan")
    auxiliary = plan.get("auxiliary_asset", {})
    if not auxiliary.get("path") or _sha(Path(auxiliary["path"])) != auxiliary.get("sha256"):
        raise ValueError("ENCODER auxiliary asset changed since frozen plan")
    tokenizer_asset = worker_identity.get("tokenizer_asset")
    if (not isinstance(tokenizer_asset, dict) or not tokenizer_asset.get("path")
            or _sha(Path(tokenizer_asset["path"])) != tokenizer_asset.get("sha256")):
        raise ValueError("OpenCLIP tokenizer BPE asset changed since frozen plan")
    if plan.get("checkpoint_mapping_status") != "VERIFIED_METADATA":
        raise ValueError("ENCODER checkpoint mapping is not verified")
    for relative, digest in receipt["artifact_sha256"].items():
        path = (root / relative).resolve()
        if root not in path.parents or not path.is_file() or _sha(path) != digest:
            raise ValueError(f"native receipt artifact mismatch: {relative}")
    with np.load(root / "native-scores/source_ids.npz", allow_pickle=False) as f:
        ids = {key: f[key].tolist() for key in ("category", "gallery_ids", "reference_ids", "target_ids", "query_texts",
                "query_paths", "gallery_paths", "reference_positions", "target_positions")}
        image_manifest_paths = f["image_manifest_paths"].tolist()
        image_manifest_sha256 = f["image_manifest_sha256"].tolist()
    if ids["category"] != category:
        raise ValueError("native archive category differs from frozen plan")
    g1_files = receipt.get("g1_artifact_sha256", {})
    if g1_files != _verify_native_g1(root, ids, plan["native_batch_size"]):
        raise ValueError("native G1 artifact manifest or content mismatch")
    if (_canonical_sha(ids) != receipt.get("ordered_ids_text_paths_positions_sha256")
            or _canonical_sha(list(zip(ids["reference_ids"], ids["target_ids"], ids["query_texts"])))
            != receipt.get("query_content_sha256")):
        raise ValueError("native ordered IDs/text/query content digest mismatch")
    image_digests = _verify_image_manifest(
        dataset, ids, image_manifest_paths, image_manifest_sha256, receipt.get("image_content_sha256", {}))
    relation_path = root / f"relation-seed{plan['relation_seed']}.npy"
    relation_meta_path = Path(str(relation_path) + ".json")
    if _sha(relation_meta_path) != receipt.get("relation_metadata_file_sha256"):
        raise ValueError("native relation metadata file digest mismatch")
    relation_meta = json.loads(relation_meta_path.read_text(encoding="utf-8"))
    tensor = np.load(relation_path, allow_pickle=False)
    tensor_digest = hashlib.sha256(np.ascontiguousarray(tensor).tobytes()).hexdigest()
    relation = receipt.get("native_relation")
    if (tensor_digest != relation_meta.get("sha256")
            or tensor_digest != receipt.get("relation_tensor_content_sha256")
            or relation is None or relation.get("sha256") != tensor_digest
            or relation.get("seed") != relation_meta.get("seed")
            or relation_meta.get("seed") != plan["relation_seed"]
            or relation_meta.get("tensor") != "backbone.binding_decoder.learnable_relation"
            or list(tensor.shape) != relation_meta.get("shape")
            or relation_meta.get("dtype") != f"torch.{tensor.dtype}"
            or relation_meta.get("in_state_dict") is not False):
        raise ValueError("native relation tensor/seed/metadata mismatch")
    scores = np.load(root / "native-scores/native_scores.float32.npy", mmap_mode="r")
    orders = np.load(root / "native-scores/native_orders.int32.npy", mmap_mode="r")
    expected_shape = (len(ids["target_ids"]), len(ids["gallery_ids"]))
    if scores.dtype != np.float32 or orders.dtype != np.int32 or scores.shape != expected_shape or orders.shape != expected_shape:
        raise ValueError("native score/order dtype or shape mismatch")
    if not np.isfinite(scores).all():
        raise ValueError("native score array contains invalid values")
    expected_indices = np.arange(expected_shape[1], dtype=np.int32)
    for row in orders:
        if row.size != expected_indices.size or not np.array_equal(np.unique(row), expected_indices):
            raise ValueError("native order array contains invalid values")
    if len(ids["reference_ids"]) != len(ids["target_ids"]) or len(ids["target_positions"]) != len(ids["target_ids"]):
        raise ValueError("native query IDs and target positions have different lengths")
    gallery_positions = {image_id: index for index, image_id in enumerate(ids["gallery_ids"])}
    if len(gallery_positions) != len(ids["gallery_ids"]):
        raise ValueError("native gallery IDs are not unique")
    for row, (reference_id, target_id) in enumerate(zip(ids["reference_ids"], ids["target_ids"])):
        if gallery_positions.get(reference_id) != ids["reference_positions"][row]:
            raise ValueError(f"native reference ID/index mismatch at row {row}")
        if gallery_positions.get(target_id) != ids["target_positions"][row]:
            raise ValueError(f"native target ID/index mismatch at row {row}")
    return ids, tensor, relation_meta


def reconcile(root: Path, receipt: dict[str, Any], adapter_factory: Any = Adapter) -> dict[str, Any]:
    import numpy as np

    ids, native_relation, relation_meta = verify_receipt(root, receipt)
    adapter = adapter_factory(resolve_config())
    try:
        adapter.work_root = Path(root) / "adapter-work" / "encoder"
        adapter.prepare_identity()
        adapter._adopt_scratch_relation()
        adapter_relation = adapter.relation_state or {}
        relation_seed = receipt["plan"]["relation_seed"]
        adapter_relation_path = adapter.construction_state_root() / f"relation-seed{relation_seed}.npy"
        adapter_relation_meta_path = Path(str(adapter_relation_path) + ".json")
        adapter_tensor = np.load(adapter_relation_path, allow_pickle=False)
        adapter_meta = json.loads(adapter_relation_meta_path.read_text(encoding="utf-8"))
        adapter_digest = hashlib.sha256(np.ascontiguousarray(adapter_tensor).tobytes()).hexdigest()
        if (adapter_relation.get("sha256") != adapter_digest
                or adapter_digest != relation_meta["sha256"]
                or adapter_meta.get("sha256") != adapter_digest
                or adapter_meta.get("seed") != relation_meta["seed"]
                or adapter_meta.get("transform_policy") != Adapter.transform_policy
                or not np.array_equal(adapter_tensor, native_relation)):
            return {"status": "relation_mismatch", "native_relation": relation_meta,
                    "adapter_relation": adapter_relation, "adapter_metadata": adapter_meta,
                    "adapter_tensor_sha256": adapter_digest,
                    "adapter_metadata_sha256": _sha(adapter_relation_meta_path)}
        if adapter_relation.get("seed") != receipt["plan"]["relation_seed"]:
            return {"status": "relation_mismatch", "reason": "adapter relation seed differs"}
        category = receipt["plan"]["category"]
        adapter.batch_size = receipt["plan"]["adapter_batch_size"]
        g1_root = Path(root) / "g1"
        g1_root.mkdir(parents=True, exist_ok=True)
        queries = adapter.encode_queries(ids["query_paths"], ids["query_texts"],
                                         [category] * len(ids["query_texts"]),
                                         capture_path=g1_root / "adapter-queries")
        targets = adapter.encode_targets(ids["gallery_paths"], [None] * len(ids["gallery_paths"]),
                                         capture_path=g1_root / "adapter-gallery")
        scores = adapter.score(queries, targets)
        score_root = Path(root) / "native-scores"
        capture = _NativeArgsortCapture.__new__(_NativeArgsortCapture)
        capture.scores = np.load(score_root / "native_scores.float32.npy", mmap_mode="r")
        capture.orders = np.load(score_root / "native_orders.int32.npy", mmap_mode="r")
        capture.reference_positions = ids["reference_positions"]
        capture.target_positions = ids["target_positions"]
        from workbench.replay.encoder import _compare_adapter_scores
        compared = _compare_adapter_scores(capture, scores, Path(root) / "adapter-scores")
        g1_comparison = _compare_g1_inputs(
            Path(root), ids, receipt["plan"]["native_batch_size"], receipt["plan"]["adapter_batch_size"])
        result = {"status": "measured", "relation_match": True,
                  "native_receipt_sha256": _sha(Path(root) / "native-receipt.json"),
                  "plan_sha256": receipt["plan_sha256"],
                  "category": category, "repeat_id": receipt["plan"]["repeat_id"],
                  "repeat_seed": receipt["plan"]["repeat_seed"],
                  "native_batch_size": receipt["plan"]["native_batch_size"],
                  "adapter_batch_size": receipt["plan"]["adapter_batch_size"],
                  "adapter_g1_artifact_sha256": {
                      **_adapter_g1_shard_hashes(g1_root / "adapter-queries"),
                      **_adapter_g1_shard_hashes(g1_root / "adapter-gallery")},
                  "native_relation_metadata": relation_meta, "adapter_relation_metadata": adapter_meta,
                  "comparison": compared, "g1_comparison": g1_comparison,
                  "preprocessing_note": "native query preprocess_train; Adapter query preprocess_val; diagnostic-only"}
        with (Path(root) / "reconcile-result.json").open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
        return result
    finally:
        adapter.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "capture", "reconcile"), required=True)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--category", choices=CATEGORIES, default="dress")
    parser.add_argument("--native-batch-size", type=int, default=DEFAULT_NATIVE_BATCH_SIZE)
    parser.add_argument("--adapter-batch-size", type=int, default=DEFAULT_ADAPTER_BATCH_SIZE)
    parser.add_argument("--repeat-id", default="repeat-1")
    args = parser.parse_args(argv)
    if args.phase == "preflight":
        print(json.dumps(preflight(args.category, args.native_batch_size, args.adapter_batch_size,
                                   args.repeat_id), indent=2))
        return 0
    if args.run_root is None:
        parser.error("--run-root is required for capture and reconcile")
    args.run_root = args.run_root.absolute()
    if args.phase == "capture":
        if args.run_root.exists():
            parser.error("immutable run root already exists")
        plan = preflight(args.category, args.native_batch_size, args.adapter_batch_size, args.repeat_id)
        planned_roots = [Path(path).absolute() for path in filter(None, os.environ.get(
            "WORKBENCH_ENCODER_PLANNED_ROOTS", str(args.run_root)).split(os.pathsep))]
        sunk_roots = [Path(path).absolute() for path in filter(None, os.environ.get(
            "WORKBENCH_ENCODER_SUNK_ROOTS", "").split(os.pathsep))]
        if args.run_root not in planned_roots or args.run_root in sunk_roots:
            parser.error("capture run root missing or invalid in disk reservation roots")
        admission = disk_admission(planned_roots, sunk_roots=sunk_roots,
                                   staging_root=Path(tempfile.gettempdir()),
                                   staging_peak_bytes=_staging_peak_bytes(Path(plan["dataset_root"])))
        print("ENCODER_DISK_ADMISSION " + json.dumps(admission, sort_keys=True), flush=True)
        args.run_root.mkdir(parents=True)
        with (args.run_root / "frozen-plan.json").open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(plan, indent=2) + "\n")
        command = [*plan["native_argv_template"][:-1], str(args.run_root / "native-result.json")]
        result = _run_native(command, Path(plan["source_root"]), args.run_root)
        bind_receipt(args.run_root, plan, command, result.returncode)
        return result.returncode
    receipt_path = args.run_root / "native-receipt.json"
    if not receipt_path.is_file():
        parser.error("native receipt missing")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(json.dumps(reconcile(args.run_root, receipt), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
