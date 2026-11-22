"""Create immutable FashionIQ V3 preregistrations without loading models."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DATA = ROOT / "data/FashionIQ"
CAPTIONS = ROOT / "workbench/artifacts/assets/csmcir/cf0c19bb346266c295b4b5772ebf03bd1c4f0467/fashioniq_qwen_captions/qwen_captions"
CATEGORIES = ("dress", "shirt", "toptee")
ROLES = ("native_repeat_1", "native_repeat_2", "adapter_repeat_1", "adapter_repeat_2",
         "native_adapter_native", "native_adapter_adapter", "native_batch_1", "native_batch_n",
         "adapter_batch_1", "adapter_batch_n")
TOLERANCES = {"forward_parity": {"atol": 1e-5, "rtol": 1e-4},
              "batch_parity": {"atol": 1e-4, "rtol": 1e-3}}


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def digest(value: object) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else canonical(value)).hexdigest()


def load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot load JSON input {path}: {error}") from error


_DIGEST_CACHE: dict[Path, tuple[int, int, str]] = {}


def file_digest(path: Path) -> str:
    try:
        stat_before = path.stat()
        cached = _DIGEST_CACHE.get(path)
        cache_key = (stat_before.st_size, stat_before.st_mtime_ns)
        if cached is not None and cached[:2] == cache_key:
            return cached[2]
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                hasher.update(chunk)
        stat_after = path.stat()
        if (stat_after.st_size, stat_after.st_mtime_ns) != cache_key:
            raise RuntimeError(f"input changed while hashing {path}")
        result = hasher.hexdigest()
        _DIGEST_CACHE[path] = (stat_after.st_size, stat_after.st_mtime_ns, result)
        return result
    except OSError as error:
        raise ValueError(f"cannot hash input {path}: {error}") from error


def runtime_identity(python: Path) -> dict[str, str]:
    code = ("import json,platform,sys,torch,numpy,torchvision; "
            "print(json.dumps({'python':sys.executable,'version':platform.python_version(),"
            "'torch':torch.__version__,'torchvision':torchvision.__version__,'numpy':numpy.__version__,"
            "'cuda':str(torch.version.cuda),'available':str(torch.cuda.is_available()),"
            "'capability':str(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else 'none'}))")
    try:
        result = subprocess.run([str(python), "-c", code], check=True, capture_output=True, text=True, timeout=60)
        value = json.loads(result.stdout.strip())
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as error:
        raise RuntimeError(f"runtime probe failed for {python}: {error}") from error
    keys = ("python", "version", "torch", "torchvision", "numpy", "cuda", "available", "capability")
    if not isinstance(value, dict) or any(not isinstance(value.get(key), str) for key in keys):
        raise RuntimeError(f"runtime probe returned malformed identity for {python}")
    return value


def source_info(model_id: str) -> tuple[list[Path], Path, Path, str, bool]:
    if model_id == "csmcir":
        source = ROOT / "workbench/third_party/CSMCIR/src"
        files = [source / name for name in ("lavis/models/blip2_models/blip2_qformer_cir_align_prompt_csmcir.py", "validate_blip_csmcir.py", "utils_csmcir.py", "data_utils_csmcir.py", "lavis/configs/models/blip2/blip2_pretrain.yaml", "lavis/processors/blip_processors.py")]
        python = ROOT / "workbench/artifacts/environments/cir-005ef4e1b02999a5/bin/python"
        checkpoint = ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/csmcir/fashioniq_tuned_clip_best.pt"
        protocol = "FashionIQ native relative/classic; pair strip/capitalize/and/eval; native inference; category Final_Caption targets; source strict=False missing=[] unexpected=token_importance"
        return files, python, checkpoint, protocol, True
    source = ROOT / "workbench/third_party/SPRC/src"
    files = [source / name for name in ("lavis/models/blip2_models/blip2_qformer_cir_align_prompt.py", "validate_blip.py", "utils.py", "data_utils.py", "lavis/configs/models/blip2/blip2_pretrain_vitL.yaml", "lavis/processors/blip_processors.py")]
    python = ROOT / "workbench/envs/sprc310-sm120/bin/python"
    checkpoint = ROOT / "workbench/artifacts/checkpoint_selection/20261106T002617+0700/cohort/downloads/../../checkpoints/sprc/sprc_fiq_vitl.pt"
    protocol = "FashionIQ native relative/classic; pair strip/capitalize/and/eval; native inference; native extract_target_features mean; strict checkpoint load"
    return files, python, checkpoint, protocol, False


def build_plan(model_id: str, category: str, checkpoint: Path, sources: list[Path], adapter: Path,
               python: Path, protocol: str, needs_captions: bool, runtime: dict[str, str]) -> dict[str, Any]:
    pair_path = DATA / "captions" / f"cap.{category}.val.json"
    gallery_path = DATA / "image_splits" / f"split.{category}.val.json"
    caption_path = CAPTIONS / f"{category}_cot_val.json" if needs_captions else None
    pairs, gallery = load_json(pair_path), load_json(gallery_path)
    captions = load_json(caption_path) if caption_path else {}
    if not isinstance(pairs, list) or not isinstance(gallery, list) or not isinstance(captions, dict):
        raise ValueError(f"{category}: malformed FashionIQ inputs")
    gallery_ids = list(gallery)
    missing = [item for item in gallery_ids if needs_captions and (item not in captions or not captions[item].get("Final_Caption"))]
    if missing:
        raise ValueError(f"{category}: {len(missing)} gallery IDs lack Final_Caption")
    query_ids = [f"{category}:{index:04d}:{row['candidate']}:{row['target']}" for index, row in enumerate(pairs)]
    if not query_ids or not gallery_ids:
        raise ValueError(f"{category}: empty official population")
    capture_query_indices = [0, len(pairs) - 1,
                             max(range(len(pairs)), key=lambda i: len(" and ".join(pairs[i]["captions"])))]
    candidates: dict[str, list[int]] = {}
    for index, row in enumerate(pairs):
        candidates.setdefault(row["candidate"], []).append(index)
    shared = next((indices for indices in candidates.values() if len(indices) > 1), None)
    if shared:
        capture_query_indices.extend(shared[:2])
    capture_query_indices = list(dict.fromkeys(capture_query_indices))[:5]
    capture_query_ids = [query_ids[index] for index in capture_query_indices]
    try:
        from PIL import Image
        non_square = None
        for item in gallery_ids:
            try:
                with Image.open(DATA / "images" / f"{item}.png") as image:
                    if image.width != image.height:
                        non_square = item
                        break
            except OSError:
                continue
    except ImportError:
        non_square = None
    selected_images = list(dict.fromkeys([pairs[index]["candidate"] for index in capture_query_indices] +
                                         [pairs[index]["target"] for index in capture_query_indices]))
    selected_images = [item for item in selected_images if item in gallery_ids]
    capture_gallery_ids = list(dict.fromkeys(([non_square] if non_square else []) + selected_images))[:5]
    capture = {"max_samples_per_scope": 5, "query_ids": capture_query_ids,
               "gallery_ids": capture_gallery_ids,
               "selection_policy": {"query": "first and last (partial batch), longest composed caption, shared candidate when present; deterministic, capped at five",
                                    "gallery": "first non-square image when available, then selected query candidate/target images present in gallery; deterministic, capped at five",
                                    "non_square_gallery_id": non_square}}
    native_files = {str(path.relative_to(ROOT)): file_digest(path) for path in sources}
    adapter_sha = file_digest(adapter)
    caption_sha = file_digest(caption_path) if caption_path else "not_applicable"
    configuration = digest({"native_files": native_files, "adapter_file": adapter_sha, "pair_json": file_digest(pair_path), "gallery_split": file_digest(gallery_path), "target_caption_json": caption_sha, "checkpoint_size": checkpoint.stat().st_size})
    scoring_labels = {
        "query_targets": [[query_id, row["target"]] for query_id, row in zip(query_ids, pairs, strict=True)],
        "reference_exclusions": [[query_id, []] for query_id in query_ids],
    }
    identity = {"checkpoint": file_digest(checkpoint), "configuration": configuration, "environment": digest(runtime), "protocol": protocol,
                "category": category, "cohort_id": f"fashioniq-val-full-{model_id}-{category}-multimodal-q256-g8192-v3",
                "precision": "native maybe_autocast; float32 captures; tf32 unchanged", "seed": 0,
                "query_order_sha256": digest(query_ids), "gallery_order_sha256": digest(gallery_ids),
                "reference_exclusions_sha256": digest(scoring_labels), "query_count": len(query_ids), "gallery_count": len(gallery_ids)}
    scoring_proof_path = f"g1_g2.{model_id}.{category}.native.score-ranks.jsonl"
    q_sample = query_ids[:min(32, len(query_ids))]
    sample_gallery = list(dict.fromkeys([row["target"] for row in pairs[:len(q_sample)]] + gallery_ids))
    g_sample = sample_gallery[:min(64, len(sample_gallery))]
    if not g_sample or any(item not in g_sample for item in [row["target"] for row in pairs[:len(q_sample)]]):
        raise ValueError(f"{category}: sample gallery does not contain all sampled positives")
    sample = dict(identity, cohort_id=f"fashioniq-val-sample-controls-{model_id}-{category}-q256-g8192-v3",
                  query_order_sha256=digest(q_sample), gallery_order_sha256=digest(g_sample), query_count=len(q_sample), gallery_count=len(g_sample))
    partitions = {"full": {"partition_id": f"{model_id}:{category}:full", "query_partition_sha256": identity["query_order_sha256"], "gallery_partition_sha256": identity["gallery_order_sha256"]},
                  "sample": {"partition_id": f"{model_id}:{category}:sample", "query_partition_sha256": sample["query_order_sha256"], "gallery_partition_sha256": sample["gallery_order_sha256"]}}
    runs: dict[str, Any] = {}
    for role in ROLES:
        is_full = role in ("native_adapter_native", "native_adapter_adapter")
        run_identity = identity if is_full else sample
        side = "adapter" if role.startswith("adapter_") or role == "native_adapter_adapter" else "native"
        batch = 1 if role.endswith("batch_1") else 32
        phase = "adapter" if side == "adapter" else "native"
        interpreter = ROOT / "workbench/envs/ctl312h/bin/python" if phase == "adapter" else python
        command = f"{interpreter} -m workbench.replay.run_fashioniq_v3 --plan PLAN --role {role} --phase {phase}"
        runs[role] = {"run_id": f"{run_identity['cohort_id']}:{role}", "source_id": f"{side}:{role}:{native_files[next(iter(native_files))] if side == 'native' else adapter_sha}", "identity": dict(run_identity),
                      "execution": {**partitions["full" if is_full else "sample"], "query_batch_size": batch, "gallery_batch_size": batch,
                                    "command": command, "supported_batch_policy": "native32;adapter32;native_adapter_crosspath32;batch1_control", "memory_policy": "stream image batches;CPU mmap q/g features;bounded score rows"}}
    control_pairs = {"native_repeat": ("native_repeat_1", "native_repeat_2", "native"), "adapter_repeat": ("adapter_repeat_1", "adapter_repeat_2", "adapter"),
                     "native_adapter": ("native_adapter_native", "native_adapter_adapter", "native"), "native_batch_1_vs_n": ("native_batch_1", "native_batch_n", "native"), "adapter_batch_1_vs_n": ("adapter_batch_1", "adapter_batch_n", "adapter")}
    controls: dict[str, Any] = {}
    for name, (reference, actual, side) in control_pairs.items():
        ref_identity = runs[reference]["identity"]
        scopes = {}
        for scope in ("query", "gallery"):
            key = f"{scope}_count"
            scopes[scope] = {"coverage": "full" if name == "native_adapter" else "sample", "population_count": identity[key], "sample_count": None if name == "native_adapter" else ref_identity[key], "tensor_shape": [ref_identity[key], 256 if scope == "query" else 8192], "component_shape": [ref_identity[key], 256 if scope == "query" else 8192], "ordered_ids_sha256": ref_identity[f"{scope}_order_sha256"], "cohort_id": ref_identity["cohort_id"]}
        controls[name] = {"reference_run_id": runs[reference]["run_id"], "actual_run_id": runs[actual]["run_id"], "reference_side": side, "dimension": "batch_parity" if "batch" in name else "forward_parity", "dtype": "float32", "tolerance": TOLERANCES["batch_parity" if "batch" in name else "forward_parity"], "scopes": scopes}
    score_order_conventions = {"native": "fp32_one_minus_score_ascending",
                               "adapter": "fp32_descending_score"}
    return {"standard": "parity-absolute-relative/v3", "model_id": model_id, "category": category, "identity": identity,
            "score_order_conventions": score_order_conventions,
            "tolerances": {"score_max_abs": 1e-6, "tie_epsilon": 1e-6},
            "scoring_proof_path": scoring_proof_path, "scoring_labels": scoring_labels,
            "population_scope": {"coverage": "full", "query_count": len(query_ids), "gallery_count": len(gallery_ids), "control_query_ids": q_sample, "control_gallery_ids": g_sample, "query_source": str(pair_path), "gallery_source": str(gallery_path), "target_caption_source": str(caption_path) if caption_path else None, "query_rows_sha256": file_digest(pair_path), "gallery_file_sha256": file_digest(gallery_path), "target_caption_bytes_sha256": caption_sha, "query_ids": query_ids, "reference_ids": [row["candidate"] for row in pairs], "target_ids": [row["target"] for row in pairs], "gallery_ids": gallery_ids, "references_remain_gallery_eligible": True, "captioned_gallery_missing_ids": len(missing)},
            "supported_batch_policy": {"supported": True, "native_query_batch": 32, "native_gallery_batch": 32, "adapter_batch": 32, "control_batch": 1, "batch_sizes": [1, 32], "description": "matched native/adapter batch32; B1-vs-BN controls"},
            "resource_budget": {"query_features_fp32_bytes": len(query_ids) * 256 * 4, "gallery_features_fp32_bytes": len(gallery_ids) * 8192 * 4, "score_row_fp32_bytes": len(gallery_ids) * 4, "peak_gpu_policy": "one model process at a time; native exits before adapter; CPU mmap q/g features; flock required", "native_query_batch": 32, "native_gallery_batch": 32, "adapter_batch": 32},
            "commands": {"native": f"workbench/scripts/gpu_job.sh {python} -m workbench.replay.run_fashioniq_v3 --plan PLAN --role native_adapter_native --phase native", "adapter": f"workbench/scripts/gpu_job.sh {ROOT / 'workbench/envs/ctl312h/bin/python'} -m workbench.replay.run_fashioniq_v3 --plan PLAN --role native_adapter_adapter --phase adapter", "preflight": f"{ROOT / 'workbench/envs/ctl312h/bin/python'} -m workbench.replay.run_fashioniq_v3 --plan PLAN --preflight"},
            "condition_scope": "multimodal only; other conditions require separate plans", "unsupported_scope": "CSMCIR CIRCUS 74381 full gallery not claimed", "environment": {"native": runtime, "native_python": str(python), "checkpoint_path": str(checkpoint), "adapter_controller": str(ROOT / "workbench/envs/ctl312h/bin/python")}, "configuration": {"native_files": native_files, "adapter_file": adapter_sha, "caption_sha256": caption_sha}, "full_population_model_metadata": {"query_dimension": 256, "gallery_dimension": 8192, "target_tokens": 32, "dtype": "float32", "score": "max per-target-token cosine"}, "input_capture": capture, "runs": runs, "controls": controls, "metrics": {"name": "FashionIQ Recall@10/Recall@50", "condition": "multimodal", "query_count": len(query_ids), "gallery_count": len(gallery_ids)}}


def write_plan(path: Path, plan: dict[str, Any]) -> str:
    helper = ROOT / "workbench/docs/reviews/20261109-native15-composition-limn/scripts/t4_parity.py"
    spec = importlib.util.spec_from_file_location("cir_t4_parity", helper)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load V3 helper: {helper}")
    module = importlib.util.module_from_spec(spec)
    module.__dict__["sys"] = sys
    spec.loader.exec_module(module)
    return module.write_v3_plan(path, plan)


def main() -> int:
    import argparse
    from workbench.backend.operator_config import resolve_config
    from workbench.backend.registry import checkpoint_by_id, checkpoint_path, load_registry
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=("csmcir", "sprc"), default=("csmcir", "sprc"))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "workbench/artifacts/parity/v3",
                        help="isolated artifact directory for immutable plans and role outputs")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=CATEGORIES)
    parser.add_argument("--revision", default="r15")
    args = parser.parse_args()
    if not args.revision.startswith("r") or not args.revision[1:].isdigit():
        raise ValueError("revision must be r followed by digits")
    config = resolve_config()
    registry = getattr(config, "registry", None) or load_registry()
    models = {row["model_id"]: row for row in registry["models"]}
    output = args.output_dir.resolve()
    created = []
    for model_id, checkpoint_id in (("csmcir", "fashioniq"), ("sprc", "sprc_fiq_vitl")):
        if model_id not in args.models:
            continue
        sources, python, configured_checkpoint, protocol, needs_captions = source_info(model_id)
        variant = checkpoint_by_id(models[model_id], checkpoint_id)
        if variant is None:
            raise RuntimeError(f"checkpoint variant missing: {model_id}/{checkpoint_id}")
        checkpoint = checkpoint_path(model_id, variant, config.WORKBENCH_CHECKPOINT_ROOT)
        if not checkpoint.is_file() or checkpoint.resolve() != configured_checkpoint.resolve():
            raise FileNotFoundError(f"configured checkpoint mismatch: {checkpoint}")
        adapter = ROOT / f"workbench/composition/model_{model_id}.py"
        runtime = runtime_identity(python)
        for category in args.categories:
            path = output / f"g1_g2.{model_id}.{category}.v3-plan.json"
            plan = build_plan(model_id, category, checkpoint, sources, adapter, python, protocol, needs_captions, runtime)
            plan_text = json.dumps(plan, ensure_ascii=False, separators=(",", ":")).replace("PLAN", str(path))
            try:
                plan = json.loads(plan_text)
            except json.JSONDecodeError as error:
                raise RuntimeError(f"generated plan is invalid: {path}") from error
            created.append({"path": str(path.relative_to(ROOT)), "sha256": write_plan(path, plan),
                            "query_count": plan["identity"]["query_count"],
                            "gallery_count": plan["identity"]["gallery_count"]})
    print(json.dumps({"plans": created}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
