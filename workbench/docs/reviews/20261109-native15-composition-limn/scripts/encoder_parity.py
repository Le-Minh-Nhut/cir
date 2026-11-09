"""G1 (input parity) + G2 (forward/scoring/ranking) for ENCODER: pinned evaluator vs workbench adapter.

Two phases, exactly as `t4_parity.py` does for LIMN:

* `--phase native --category C` runs under `workbench/envs/encoder310-adapted/bin/python` (torch
  2.7.1+cu128), the interpreter the pinned `evaluate_model.py` needs. It loads the model the way
  `workbench/replay/encoder.py:ed...evaluate_categories` loads it, constructs the dataset with the
  same cache + seed, records the QUERY tensors the native dataset baked in (stage 0,
  `preprocess_train`), the VAL-transform decode of the same files, the gallery tensors (stage 1,
  `preprocess_val`), the features `test.py`'s own forward pass produced (captured by
  `FeatureRecorder`), a val-transform control forward over the same query population, and the native
  Recall values.

* `--phase adapter --category C` runs under `workbench/envs/ctl312h/bin/python` and drives
  `workbench.composition.model_encoder.Adapter`, which refuses to run on the control interpreter and
  therefore spawns its own worker on the model interpreter. Same files, same texts.

* `--phase probe` (model interpreter) records the tokenizer's token ids for the sample texts and the
  adapter's exact pixel decode expression, so layer 2/3 can be compared outside the adapter process.

Nothing here changes evaluation semantics of either side.
"""
from __future__ import annotations

import json
import string
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path("/home/heheboiz/data/cir")
sys.path.insert(0, str(REPO))

import numpy as np

OUT = REPO / "workbench/docs/reviews/20261109-native15-composition-limn"
RAW = OUT / "raw/encoder_parity"

TOL = {
    "identity": "exact, zero tolerance",
    "pixels": "bitwise equality expected (max_abs == 0, array_equal)",
    "text_and_tokens": "exact equality",
    "forward_max_abs": 1e-5,
    "forward_max_rel": 1e-4,
    "forward_min_cosine": 1 - 1e-7,
    "batch_max_abs": 1e-4,
    "batch_max_rel": 1e-3,
    "score_max_abs": 1e-6,
    "tie_epsilon": 1e-6,
    "metrics": "exact hit-count equality",
}

BATCH = 64  # `replay.evaluate_categories` default and `evaluate_model.py` default


# --------------------------------------------------------------------------- shared helpers

def image_path(dataset_root: Path, category: str, name: str) -> Path:
    return dataset_root / "resized_image" / category / f"{name}.jpg"


def sample_indices(captions: list[dict]) -> list[int]:
    """Fixed rule, chosen before any measurement: first row, first capitalized first caption,
    first row carrying a punctuation mark or apostrophe, last row (short final batch)."""
    capitalized = next((i for i, row in enumerate(captions)
                        if row["captions"][0][:1].isupper()), 0)
    punctuation = next((i for i, row in enumerate(captions)
                        if any(ch in string.punctuation for ch in row["captions"][0] + row["captions"][1])), 0)
    return sorted({0, capitalized, punctuation, len(captions) - 1})


def compare(actual: np.ndarray, reference: np.ndarray) -> dict:
    delta = np.abs(actual.astype(np.float64) - reference.astype(np.float64))
    scale = np.maximum(np.abs(reference.astype(np.float64)), 1e-12)
    flat_actual = actual.astype(np.float64).reshape(len(actual), -1) if actual.ndim > 1 else actual.astype(np.float64)[None]
    flat_reference = (reference.astype(np.float64).reshape(len(reference), -1)
                      if reference.ndim > 1 else reference.astype(np.float64)[None])
    cosines = [(row @ ref) / (np.linalg.norm(row) * np.linalg.norm(ref))
               for row, ref in zip(flat_actual, flat_reference)]
    return {"max_abs": float(delta.max()) if delta.size else 0.0,
            "max_rel": float((delta / scale).max()) if delta.size else 0.0,
            "min_cosine": float(min(cosines)) if cosines else 1.0,
            "bitwise_equal": bool(np.array_equal(actual, reference)),
            "shape": list(np.shape(actual))}


def native_ranks(queries: np.ndarray, gallery: np.ndarray, reference_positions: list[int],
                 target_positions: list[int], mask: bool = True) -> tuple[list[int], np.ndarray]:
    """Exactly `test.py`: per-row L2 normalize, dot, optional -10e10 reference mask, argsort."""
    queries = queries.astype(np.float32) / np.linalg.norm(queries.astype(np.float32), axis=1, keepdims=True)
    gallery = gallery.astype(np.float32) / np.linalg.norm(gallery.astype(np.float32), axis=1, keepdims=True)
    sims = queries.dot(gallery.T)
    if mask:
        for row, reference in enumerate(reference_positions):
            sims[row, reference] = -10e10
    ranks = [int(np.flatnonzero(np.argsort(-sims[row]) == target)[0]) + 1
             for row, target in enumerate(target_positions)]
    return ranks, sims


def recall(ranks: list[int]) -> dict[str, float]:
    return {f"r{k}": 100 * sum(1 for r in ranks if r <= k) / len(ranks) for k in (1, 10, 50)}


# --------------------------------------------------------------------------- native phase

def native_phase(category: str, config) -> dict:
    import torch
    from PIL import Image

    from workbench.replay import encoder as replay
    from workbench.replay.common import (FeatureRecorder, enable_deterministic_inference,
                                         run_cache_root, seeded_dataset_construction)
    from workbench.replay.limn_observer import _image_ids, _raw_captions, expected_model_input

    checkpoint_root = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "encoder"
    dataset_root = Path(config.FASHIONIQ_ROOT)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "ENCODER"
    problems = replay.preflight(checkpoint_root, dataset_root, source_root, (category,))
    if problems:
        raise SystemExit(f"preflight blocked: {problems}")

    began = time.time()
    torch_mod, open_clip, datasets, test_module, model_try2 = replay._load_replay_modules(source_root)
    variant = replay.CHECKPOINTS["fashioniq"]
    checkpoint_path = checkpoint_root / variant["filename"]

    # Same construction and loader as `replay.encoder.evaluate_categories` (hyperparameters matching
    # `evaluate_model.py` defaults for FashionIQ: P=3, wc=2, N_p=2). `Backbone.__init__` hard-codes
    # `clip_path = './'`, so the pinned module must be built with the source root as cwd -- exactly
    # what `composition/model_encoder.py::_worker_main` does.
    import os
    previous_cwd = Path.cwd()
    try:
        os.chdir(source_root)
        model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                                   wc=2, N_p=2, weighted=True)
    finally:
        os.chdir(previous_cwd)
    state = torch_mod.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict):
        raise SystemExit("checkpoint is not a state_dict; replay path expects a dict")
    report = model.load_state_dict(state, strict=False)
    replay._assert_checkpoint_loaded(model, report, variant, source_root)
    model = model.to("cuda").eval()
    enable_deterministic_inference(torch_mod)

    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
    transform = [preprocess_train, preprocess_val]

    def decode(path: Path, stage: int) -> "torch.Tensor":
        """Verbatim `datasets.FashionIQ.get_img` at the given stage."""
        with open(path, "rb") as handle:
            image = Image.open(handle)
            image = image.convert("RGB")
        return transform[stage](image)

    def forward(kind: str, tensors: list, texts: list[str] | None, batch_size: int = BATCH) -> np.ndarray:
        rows: list[np.ndarray] = []
        for start in range(0, len(tensors), batch_size):
            chunk = torch.stack(tensors[start:start + batch_size]).float().cuda()
            with torch.no_grad():
                if kind == "queries":
                    out = model.extract_retrieval_compose(chunk, texts[start:start + batch_size])
                else:
                    out = model.extract_retrieval_target(chunk)
            rows.append(out.data.cpu().numpy())
        return np.concatenate(rows) if rows else np.empty((0, 512), dtype=np.float32)

    def forward_paths(paths: list[Path], texts: list[str], batch_size: int = BATCH) -> np.ndarray:
        """Same forward, but each chunk is decoded then released: the population is never held twice."""
        rows: list[np.ndarray] = []
        for start in range(0, len(paths), batch_size):
            chunk_paths = paths[start:start + batch_size]
            chunk = torch.stack([decode(path, 1) for path in chunk_paths]).float().cuda()
            with torch.no_grad():
                out = model.extract_retrieval_compose(chunk, texts[start:start + batch_size])
            rows.append(out.data.cpu().numpy())
            del chunk
        return np.concatenate(rows) if rows else np.empty((0, 512), dtype=np.float32)

    with run_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        with seeded_dataset_construction():
            dataset = datasets.FashionIQ(path=root_str, transform=transform, split="val-split")

        images = _image_ids(dataset_root, category)
        captions = _raw_captions(dataset_root, category)
        correction = json.loads((dataset_root / "captions" / f"correction_dict_{category}.json").read_text())
        suffix = category
        test_queries = getattr(dataset, f"test_queries_{suffix}")
        test_targets = getattr(dataset, f"test_targets_{suffix}")

        gallery_positions = [entry["target_img_id"] for entry in test_targets]
        gallery_ids = [images[position] for position in gallery_positions]
        # ENCODER's gallery derivation (datasets.py:get_test_data): first appearance of
        # source_img_id then target_img_id, per query.
        rebuilt: list[str] = []
        for row in captions:
            for key in ("candidate", "target"):
                if row[key] not in rebuilt:
                    rebuilt.append(row[key])

        model_texts = [entry["mod"]["str"] for entry in test_queries]
        recomposed = [expected_model_input(row["captions"], correction) for row in captions]
        reference_ids = [images[entry["source_img_id"]] for entry in test_queries]
        target_ids = [images[entry["target_img_id"]] for entry in test_queries]

        chosen = sample_indices(captions)

        # ---- layer 1: identities / order / texts / tokens / masks
        token_ids = [model.backbone.tokenizer([model_texts[i]]).numpy().tolist()[0] for i in chosen]
        identity = {
            "query_count_native": len(test_queries),
            "query_count_annotations": len(captions),
            "gallery_count_native": len(test_targets),
            "gallery_count_rebuilt_from_annotations": len(rebuilt),
            "gallery_order_identical": gallery_ids == rebuilt,
            "reference_ids_identical": reference_ids == [row["candidate"] for row in captions],
            "target_ids_identical": target_ids == [row["target"] for row in captions],
            "model_text_identical_to_author_composition": model_texts == recomposed,
            "model_text_mismatches": [{"index": i, "native": model_texts[i], "recomposed": recomposed[i]}
                                      for i in range(len(recomposed)) if model_texts[i] != recomposed[i]][:5],
            "duplicate_caption_pairs_within_category":
                len(captions) - len({(row["candidate"], recomposed[i]) for i, row in enumerate(captions)}),
            "reference_equals_target_rows": sum(1 for row in captions if row["candidate"] == row["target"]),
            "gallery_has_duplicates": len(set(gallery_ids)) != len(gallery_ids),
            "tokenizer": {
                "callable": f"{type(model.backbone.tokenizer).__module__}."
                            f"{type(model.backbone.tokenizer).__qualname__}",
                "vocab_size": int(model.backbone.clip.token_embedding.num_embeddings),
                "context_length": int(model.backbone.clip.positional_embedding.shape[0]),
                "attn_mask_shape": list(model.backbone.clip.attn_mask.shape),
                "attn_mask_dtype": str(model.backbone.clip.attn_mask.dtype),
                "source": "open_clip.get_tokenizer('ViT-B-32') inside pinned Backbone.__init__",
            },
            "transforms": {
                "query_stage_index": 0,
                "query_transform": "open_clip preprocess_train (RandomResizedCrop(224, scale=(0.9,1.0), ratio=(0.75,1.3333)))",
                "gallery_stage_index": 1,
                "gallery_transform": "open_clip preprocess_val (Resize(224, bicubic)+CenterCrop(224))",
                "mean": [float(v) for v in model.backbone.clip.visual.image_mean],
                "std": [float(v) for v in model.backbone.clip.visual.image_std],
            },
            "model_in_eval": not model.training,
            "sample_indices": chosen,
            "sample_texts": [model_texts[i] for i in chosen],
            "sample_token_ids": token_ids,
            "sample_token_lengths": [len(t) for t in token_ids],
        }
        del recomposed

        # ---- layer 2: pixels, three independent decodes of the SAME files
        native_query_tensors = torch.stack([test_queries[i]["source_img_data"] for i in chosen]).numpy()
        val_query_tensors = torch.stack([
            decode(image_path(dataset_root, category, reference_ids[i]), 1) for i in chosen]).numpy()
        native_target_tensors = torch.stack([
            next(entry["target_img_data"] for entry in test_targets
                 if images[entry["target_img_id"]] == target_ids[i]) for i in chosen]).numpy()
        val_target_tensors = torch.stack([
            decode(image_path(dataset_root, category, target_ids[i]), 1) for i in chosen]).numpy()
        pixel = {
            "native_query_is_train_transform": True,
            "native_query_vs_val_decode_max_abs":
                float(np.abs(native_query_tensors.astype(np.float64) - val_query_tensors.astype(np.float64)).max()),
            "native_query_vs_val_decode_bitwise": bool(np.array_equal(native_query_tensors, val_query_tensors)),
            "native_target_vs_val_decode_max_abs":
                float(np.abs(native_target_tensors.astype(np.float64) - val_target_tensors.astype(np.float64)).max()),
            "native_target_vs_val_decode_bitwise": bool(np.array_equal(native_target_tensors, val_target_tensors)),
            "sample_shape": list(native_query_tensors.shape),
            "sample_dtype": str(native_query_tensors.dtype),
            "per_image_train_vs_val_max_abs": [
                float(np.abs(native_query_tensors[i].astype(np.float64)
                             - val_query_tensors[i].astype(np.float64)).max())
                for i in range(len(chosen))],
            "same_file_same_stage_repeat_bitwise": bool(np.array_equal(
                np.stack([decode(image_path(dataset_root, category, reference_ids[i]), 1) for i in chosen]),
                val_query_tensors)),
        }

        # ---- layer 4: forward. Primary = native dataset tensors (train transform on the query side),
        # control = identical val-transform query pixels on both sides.
        sample_texts = [model_texts[i] for i in chosen]
        sample_query_native = forward("queries", [test_queries[i]["source_img_data"] for i in chosen], sample_texts)
        sample_query_val = forward("queries", [torch.from_numpy(t) for t in val_query_tensors], sample_texts)
        sample_target_native = forward("targets", [torch.from_numpy(t) for t in native_target_tensors], None)

        # ---- layer 4/5/7/8 on the full category population, from the features the native metric used
        params = SimpleNamespace(batch_size=BATCH, local_rank=0)
        with FeatureRecorder(model) as recorder:
            native_recalls = dict(test_module.test(params, model, dataset, category))
        population_native, population_gallery = recorder.features()
        population_val = forward_paths(
            [image_path(dataset_root, category, name) for name in reference_ids], model_texts, BATCH)

        reference_positions = [entry["source_img_id"] for entry in test_queries]
        target_positions = [entry["target_img_id"] for entry in test_queries]
        ranks_native, sims_native = native_ranks(population_native, population_gallery,
                                                 [gallery_positions.index(p) for p in reference_positions],
                                                 [gallery_positions.index(p) for p in target_positions], mask=True)
        ranks_val, sims_val = native_ranks(population_val, population_gallery,
                                           [gallery_positions.index(p) for p in reference_positions],
                                           [gallery_positions.index(p) for p in target_positions], mask=True)

        np.savez(RAW / f"native.{category}.npz",
                 sample_query_native=sample_query_native, sample_query_val=sample_query_val,
                 sample_target=sample_target_native,
                 native_query_tensor=native_query_tensors, val_query_tensor=val_query_tensors,
                 native_target_tensor=native_target_tensors,
                 population_native=population_native, population_val=population_val,
                 gallery=population_gallery)
        inputs = {
            "category": category,
            "dataset_root": str(dataset_root),
            "sample_indices": chosen,
            "sample_query_paths": [str(image_path(dataset_root, category, reference_ids[i])) for i in chosen],
            "sample_target_paths": [str(image_path(dataset_root, category, target_ids[i])) for i in chosen],
            "sample_texts": sample_texts,
            "sample_reference_ids": [reference_ids[i] for i in chosen],
            "sample_target_ids": [target_ids[i] for i in chosen],
            "full_query_paths": [str(image_path(dataset_root, category, name)) for name in reference_ids],
            "full_query_texts": model_texts,
            "full_reference_ids": reference_ids,
            "full_target_ids": target_ids,
            "gallery_ids": gallery_ids,
            "gallery_paths": [str(image_path(dataset_root, category, name)) for name in gallery_ids],
            "reference_positions": [gallery_positions.index(p) for p in reference_positions],
            "target_positions": [gallery_positions.index(p) for p in target_positions],
        }
        (RAW / f"inputs.{category}.json").write_text(json.dumps(inputs), encoding="utf-8")

        result = {
            "category": category,
            "layer1_identity": identity,
            "layer2_pixels": pixel,
            "layer4_forward_sample": {
                # cross-process comparison happens in the adapter phase; shapes recorded here
                "native_query_shape": list(sample_query_native.shape),
                "native_query_val_shape": list(sample_query_val.shape),
                "native_target_shape": list(sample_target_native.shape),
                "native_query_val_vs_train_control": compare(sample_query_val, sample_query_native),
            },
            "native_metrics_recheck": native_recalls,
            "native_rank_recall_from_features": recall(ranks_native),
            "val_transform_control_rank_recall": recall(ranks_val),
            "native_metrics_match_feature_rerank": all(
                abs(native_recalls[f"{category}_r{k}"] - recall(ranks_native)[f"r{k}"]) < 1e-9
                for k in (1, 10, 50)),
            "population": {"queries": int(population_native.shape[0]),
                           "gallery": int(population_gallery.shape[0])},
            "phase_seconds": round(time.time() - began, 1),
        }
        del model, dataset, test_queries, test_targets
    torch_mod.cuda.empty_cache()
    return result


def determinism_phase(tag: str) -> dict:
    """Cross-process vs in-process variability of the SAME forward pass, in the model interpreter.

    The adapter phase turned out to differ from the native phase by ~1e-2 absolute on bitwise-identical
    pixels, and the adapter's own `restart()` differs from itself by the same magnitude. Before
    blaming any mapping, measure which knobs move the number: process identity, batch shape, and the
    kernel-determinism flags `replay.encoder.enable_deterministic_inference` sets (which the adapter
    worker does NOT set).
    """
    import torch
    from PIL import Image

    from workbench.replay import encoder as replay
    from workbench.replay.common import enable_deterministic_inference

    dataset_root = REPO / "data/FashionIQ"
    source_root = REPO / "workbench/third_party/ENCODER"
    checkpoint_root = Path(resolve_config_path())
    checkpoint_path = checkpoint_root / "encoder" / "fashioniq.pt"
    inputs = json.loads((RAW / "inputs.dress.json").read_text())
    import open_clip
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
    _, _, datasets, _test, model_try2 = replay._load_replay_modules(source_root)
    import os
    previous = Path.cwd()
    try:
        os.chdir(source_root)
        model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                                   wc=2, N_p=2, weighted=True)
    finally:
        os.chdir(previous)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model = model.to("cuda").eval()

    pixels = []
    for name in inputs["sample_query_paths"]:
        with Image.open(name) as image:
            pixels.append(preprocess_val(image.copy()))
    tensor = torch.stack(pixels)
    texts = inputs["sample_texts"]

    def run(batch_size: int) -> np.ndarray:
        rows = []
        for start in range(0, len(tensor), batch_size):
            chunk = tensor[start:start + batch_size].cuda()
            with torch.no_grad():
                rows.append(model.extract_retrieval_compose(chunk, texts[start:start + batch_size])
                            .data.cpu().numpy())
        return np.concatenate(rows)

    result: dict = {"tag": tag, "torch_threads": torch.get_num_threads(),
                    "cudnn_deterministic": torch.backends.cudnn.deterministic,
                    "cudnn_benchmark": torch.backends.cudnn.benchmark,
                    "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                    "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}

    arrays: dict[str, np.ndarray] = {}
    arrays["nondet_b4_a"] = run(4)
    arrays["nondet_b4_b"] = run(4)
    arrays["nondet_b32"] = run(32)
    arrays["nondet_b1"] = np.concatenate([run(1)[i:i + 1] for i in range(len(tensor))])
    torch.set_num_threads(4)
    arrays["nondet_b4_threads4"] = run(4)
    enable_deterministic_inference(torch)
    result["flags_after"] = {"cudnn_deterministic": torch.backends.cudnn.deterministic,
                             "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                             "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}
    arrays["det_b4_a"] = run(4)
    arrays["det_b4_b"] = run(4)
    arrays["det_b32"] = run(32)
    arrays["det_b1"] = np.concatenate([run(1)[i:i + 1] for i in range(len(tensor))])
    # TF32 is ON for cuDNN by default in this torch build (`cudnn_allow_tf32: True`). Kernel
    # SELECTION under TF32 is a per-process heuristic, so it is the first suspect for a delta that
    # survives every determinism flag.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    result["flags_tf32_off"] = {"cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                                "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32}
    arrays["notf32_b4_a"] = run(4)
    arrays["notf32_b4_b"] = run(4)
    np.savez(RAW / f"determinism.{tag}.npz", **arrays)

    result["in_process"] = {
        key: compare(arrays[key], arrays["nondet_b4_a"])
        for key in arrays if key != "det_b4_a"}
    result["in_process"]["det_b4_b_vs_det_b4_a"] = compare(arrays["det_b4_b"], arrays["det_b4_a"])
    (RAW / f"determinism.{tag}.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def determinism_compare() -> dict:
    tags = sorted(path.stem.split(".")[-1] for path in RAW.glob("determinism.*.npz"))
    if len(tags) < 2:
        raise SystemExit("run `--phase determinism --tag A` and `--tag B` first")
    first, second = (np.load(RAW / f"determinism.{tag}.npz") for tag in tags[:2])
    return {tags[0]: {key: compare(first[key], second[key]) for key in first.files},
            "tags": tags[:2]}


def resolve_config_path() -> str:
    from workbench.backend.operator_config import resolve_config

    return resolve_config().WORKBENCH_CHECKPOINT_ROOT


def resolve_config_local():
    from workbench.backend.operator_config import resolve_config

    return resolve_config()


def native_floor_phase(category: str, tag: str = "") -> dict:
    """Second NATIVE process, SAME code and pixels: the cross-process noise floor of this model.

    Without it, an adapter-vs-native delta cannot be attributed: `determinism_phase` shows two
    processes running byte-identical code still disagree, so a mapping claim needs that floor
    measured at the SAME scale (full population, not four rows).
    """
    import torch
    from PIL import Image

    from workbench.replay import encoder as replay
    from workbench.replay.common import (enable_deterministic_inference, run_cache_root,
                                         seeded_dataset_construction)
    from workbench.replay.limn_observer import _image_ids, _raw_captions, expected_model_input

    config = resolve_config_local()
    checkpoint_path = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "encoder" / "fashioniq.pt"
    dataset_root = Path(config.FASHIONIQ_ROOT)
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "ENCODER"
    inputs = json.loads((RAW / f"inputs.{category}.json").read_text())
    import os
    import open_clip
    _, _, datasets, test_module, model_try2 = replay._load_replay_modules(source_root)
    _, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
    previous = Path.cwd()
    try:
        os.chdir(source_root)
        model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                                   wc=2, N_p=2, weighted=True)
    finally:
        os.chdir(previous)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model = model.to("cuda").eval()
    enable_deterministic_inference(torch)
    with run_cache_root(dataset_root) as cache_root:
        root_str = str(cache_root).rstrip("/") + "/"
        with seeded_dataset_construction():
            dataset = datasets.FashionIQ(path=root_str, transform=[preprocess_train, preprocess_val],
                                         split="val-split")
        queries = getattr(dataset, f"test_queries_{category}")
        targets = getattr(dataset, f"test_targets_{category}")
        reference_ids = [_image_ids(dataset_root, category)[entry["source_img_id"]] for entry in queries]
        gallery_ids = [_image_ids(dataset_root, category)[entry["target_img_id"]] for entry in targets]
        texts = [entry["mod"]["str"] for entry in queries]
        del queries, targets, dataset

    def encode(paths: list[Path], texts: list[str], kind: str) -> np.ndarray:
        rows = []
        decoded = []
        for start in range(0, len(paths), BATCH):
            chunk_paths = paths[start:start + BATCH]
            for path in chunk_paths:
                with open(path, "rb") as handle:
                    image = Image.open(handle).convert("RGB")
                decoded.append(preprocess_val(image))
            chunk = torch.stack(decoded).float().cuda()
            with torch.no_grad():
                out = (model.extract_retrieval_compose(chunk, texts[start:start + BATCH])
                       if kind == "queries" else model.extract_retrieval_target(chunk))
            rows.append(out.data.cpu().numpy())
            decoded = []
            del chunk
        return np.concatenate(rows)

    base = dataset_root / "resized_image" / category
    queries_out = encode([base / f"{name}.jpg" for name in reference_ids], texts, "queries")
    gallery_out = encode([base / f"{name}.jpg" for name in gallery_ids], [], "targets")
    suffix = f".{tag}" if tag else ""
    np.savez(RAW / f"floor.native.{category}{suffix}.npz", queries=queries_out, gallery=gallery_out)
    reference = np.load(RAW / f"native.{category}.npz")
    result = {
        "category": category,
        "queries_vs_first_native_process": compare(queries_out, reference["population_val"]),
        "gallery_vs_first_native_process": compare(gallery_out, reference["gallery"]),
    }
    print(json.dumps(result, indent=1))
    return result


def layers_phase(tag: str) -> dict:
    """Dump intermediate activations of ONE forward pass, so cross-process divergence is localized.

    Every remaining suspect (autotuned kernel choice, silent atomics, precision mode) lands in a
    different layer; comparing named activations says which one, instead of guessing.
    """
    import os

    import torch
    from PIL import Image

    from workbench.replay import encoder as replay
    from workbench.replay.common import enable_deterministic_inference

    config = resolve_config_local()
    checkpoint_path = Path(config.WORKBENCH_CHECKPOINT_ROOT) / "encoder" / "fashioniq.pt"
    source_root = Path(config.WORKBENCH_THIRD_PARTY_ROOT) / "ENCODER"
    inputs = json.loads((RAW / "inputs.dress.json").read_text())
    import open_clip
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
    _, _, _datasets, _test, model_try2 = replay._load_replay_modules(source_root)
    previous = Path.cwd()
    try:
        os.chdir(source_root)
        model = model_try2.Encoder(hidden_dim=512, dropout=0.5, local_token_num=3, t=0.1,
                                   wc=2, N_p=2, weighted=True)
    finally:
        os.chdir(previous)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    model = model.to("cuda").eval()
    enable_deterministic_inference(torch)

    traced: dict[str, np.ndarray] = {}
    handles = []
    for name in ("visual_out", "fc", "weight_chanelImg", "weight_cross", "sparse_net", "fac_net",
                 "binding_decoder", "weight_chanelImg_t", "cross_t2i", "weight_chanelText",
                 "weight_chanelText_i", "cross_i2t", "sparse_net_text", "text_fc", "text_out"):
        module = getattr(model.backbone, name, None)
        if not isinstance(module, torch.nn.Module):
            continue

        def hook(_module, _inputs_, output, _name=name):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(value, torch.Tensor):
                traced[_name] = value.detach().float().cpu().numpy()

        handles.append(module.register_forward_hook(hook))
    handles.append(model.backbone.clip.visual.register_forward_hook(
        lambda _m, _i, output: traced.__setitem__("clip_visual_pooled",
                                                  output[0].detach().float().cpu().numpy())))
    # `BindingDecoder` is called through its `.binding` METHOD, not `__call__`, so a module hook
    # never fires for it; wrap the bound method instead. Its body is re-expressed op-for-op so each
    # intermediate can be compared: it is the ONLY layer where cross-process outputs diverge.
    original_binding = model.backbone.binding_decoder.binding
    decoder = model.backbone.binding_decoder

    def traced_binding(V):
        relation = decoder.learnable_relation[:, :].unsqueeze(0).cuda()
        product_cross = torch.matmul(relation, V.transpose(-2, -1))
        # float64 CPU recomputation of the same product: which process is right, not merely different
        truth = torch.matmul(relation.detach().cpu().double(),
                             V.detach().cpu().double().transpose(-2, -1))
        scaled_cross = product_cross / (decoder.dim ** 0.5)
        S_cross = decoder.mlp_v(scaled_cross)
        product_self = torch.matmul(relation, relation.transpose(-2, -1))
        S_self = decoder.mlp_p(product_self / (decoder.dim ** 0.5))
        S_final = decoder.sigmoid(torch.matmul(S_self, S_cross)) * S_cross
        entity = torch.matmul(decoder.softmax(S_final), V)
        entity = decoder.softmax(decoder.mlp(entity))
        for name, value in (("binding_in_V", V), ("binding_product_cross", product_cross),
                            ("binding_product_cross_truth", truth),
                            ("binding_relation", relation),
                            ("binding_S_cross", S_cross), ("binding_S_self", S_self),
                            ("binding_S_final", S_final), ("binding_entity", entity)):
            traced[name] = value.detach().float().cpu().numpy()
        return entity, relation

    model.backbone.binding_decoder.binding = traced_binding

    def affine_hook(_module, inputs, output):
        traced["affine_in_x"] = inputs[0].detach().float().cpu().numpy()
        traced["affine_in_text"] = inputs[1].detach().float().cpu().numpy()
        traced["affine"] = output.detach().float().cpu().numpy()

    handles.append(model.affine.register_forward_hook(affine_hook))

    pixels = []
    for name in inputs["sample_query_paths"]:
        with Image.open(name) as image:
            pixels.append(preprocess_val(image.copy()))
    tensor = torch.stack(pixels).cuda()
    traced["pixels"] = tensor.detach().float().cpu().numpy()
    traced["token_ids"] = model.backbone.tokenizer(inputs["sample_texts"]).numpy()
    with torch.no_grad():
        final = model.extract_retrieval_compose(tensor, inputs["sample_texts"])
    traced["final"] = final.detach().float().cpu().numpy()
    for handle in handles:
        handle.remove()
    np.savez(RAW / f"layers.{tag}.npz", **traced)
    return {"tag": tag, "layers": list(traced)}


def layers_compare() -> dict:
    tags = sorted(path.stem.split(".")[-1] for path in RAW.glob("layers.*.npz"))
    if len(tags) < 2:
        raise SystemExit("run `--phase layers --tag A` and `--tag B` first")
    first, second = (np.load(RAW / f"layers.{tag}.npz") for tag in tags[:2])
    result = {"tags": tags[:2], "layers": {}}
    for name in first.files:
        if name not in second.files:
            continue
        result["layers"][name] = compare(second[name], first[name])
    result["first_divergence"] = next(
        (name for name, item in result["layers"].items() if not item["bitwise_equal"]
         and name not in ("final",)), None)
    return result


# --------------------------------------------------------------------------- probe phase

def probe_phase() -> dict:
    """Model-interpreter side: tokenizer ids and the adapter's exact decode expression."""
    import open_clip
    import torch
    from PIL import Image

    input_files = sorted(RAW.glob("inputs.*.json"))
    if not input_files:
        raise SystemExit("run the native phase first")
    source_root = REPO / "workbench/third_party/ENCODER"
    _, _, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained=str(source_root / "open_clip_pytorch_model.bin"))
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    payload: dict = {"tokenizer": {"vocab_size": int(tokenizer.vocab_size) if hasattr(tokenizer, "vocab_size") else None,
                                   "context_length": int(tokenizer.context_length),
                                   "callable": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}"},
                     "categories": {}}
    for path in input_files:
        inputs = json.loads(path.read_text())
        category = inputs["category"]
        pixels = []
        for name in inputs["sample_query_paths"]:
            with Image.open(name) as image:
                pixels.append(preprocess_val(image.copy()))
        gallery_pixels = []
        for name in inputs["gallery_paths"][:4] + inputs["sample_target_paths"]:
            with Image.open(name) as image:
                gallery_pixels.append(preprocess_val(image.copy()))
        np.savez(RAW / f"probe.{category}.npz",
                 adapter_query_pixels=torch.stack(pixels).numpy(),
                 adapter_gallery_pixels=torch.stack(gallery_pixels).numpy())
        payload["categories"][category] = {
            "sample_token_ids": tokenizer(inputs["sample_texts"]).numpy().tolist(),
            "sample_texts": inputs["sample_texts"],
            "nonsample_token_ids": tokenizer(inputs["full_query_texts"][:8]).numpy().tolist(),
        }
    return payload


# --------------------------------------------------------------------------- adapter phase

def adapter_phase(category: str, config) -> dict:
    from workbench.composition.model_encoder import Adapter

    inputs = json.loads((RAW / f"inputs.{category}.json").read_text())
    native = np.load(RAW / f"native.{category}.npz")
    probe_file = RAW / f"probe.{category}.npz"
    probe = np.load(probe_file) if probe_file.is_file() else None
    began = time.time()
    adapter = Adapter(config)
    adapter.work_root = RAW / "work" / category
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    result: dict = {"batch_size": adapter.batch_size, "interpreter": adapter.interpreter}
    try:
        aq = adapter.encode_queries(inputs["sample_query_paths"], inputs["sample_texts"],
                                    [None] * len(inputs["sample_texts"]))
        at = adapter.encode_targets(inputs["sample_target_paths"], [None] * len(inputs["sample_target_paths"]))
        result["layer4_forward_sample"] = {
            # control: identical val-transform pixels on both sides -> isolates the forward pass
            "query_val_pixels": compare(aq, native["sample_query_val"]),
            # primary: the native dataset's own query tensors (train transform)
            "query_native_tensors": compare(aq, native["sample_query_native"]),
            "target": compare(at, native["sample_target"]),
            "adapter_dtype": str(aq.dtype), "adapter_shape": list(aq.shape),
        }
        for key in ("query_val_pixels", "target"):
            item = result["layer4_forward_sample"][key]
            item["within_tolerance"] = (item["max_abs"] <= TOL["forward_max_abs"]
                                        and item["max_rel"] <= TOL["forward_max_rel"])
        result["layer4_forward_sample"]["query_native_tensors"]["within_tolerance"] = (
            result["layer4_forward_sample"]["query_native_tensors"]["max_abs"] <= TOL["forward_max_abs"])

        singles = np.concatenate([adapter.encode_queries([path], [text], [None])
                                  for path, text in zip(inputs["sample_query_paths"], inputs["sample_texts"])])
        result["batch_size_1_vs_n"] = compare(singles, aq)
        result["batch_size_1_vs_n"]["within_tolerance"] = (
            result["batch_size_1_vs_n"]["max_abs"] <= TOL["batch_max_abs"]
            and result["batch_size_1_vs_n"]["max_rel"] <= TOL["batch_max_rel"])

        adapter.restart()
        after_restart = adapter.encode_queries(inputs["sample_query_paths"], inputs["sample_texts"],
                                               [None] * len(inputs["sample_texts"]))
        result["cache_off_restart"] = compare(after_restart, aq)

        if probe is not None:
            pixel_check = compare(probe["adapter_query_pixels"], native["val_query_tensor"])
            result["layer2_pixels"] = {
                # the adapter's exact decode expression vs the native val-transform decode: same code
                # path the adapter takes for every query image
                "adapter_decode_vs_native_val_decode": pixel_check,
                "adapter_decode_vs_native_train_query_tensors": compare(
                    probe["adapter_query_pixels"], native["native_query_tensor"]),
                "adapter_decode_by": "probe phase in the model interpreter: Image.open + preprocess_val",
            }

        population_queries = adapter.encode_queries(inputs["full_query_paths"], inputs["full_query_texts"],
                                                    [None] * len(inputs["full_query_texts"]))
        population_gallery = adapter.encode_targets(inputs["gallery_paths"],
                                                    [None] * len(inputs["gallery_paths"]))
        np.savez(RAW / f"adapter.{category}.npz", queries=population_queries, gallery=population_gallery)
        result["population"] = {
            "query_count": int(population_queries.shape[0]),
            "gallery_count": int(population_gallery.shape[0]),
            "features_vs_native_val_control": compare(population_queries, native["population_val"]),
            "features_vs_native_train_query": compare(population_queries, native["population_native"]),
            "gallery_vs_native": compare(population_gallery, native["gallery"]),
            "finite": bool(np.isfinite(population_queries).all() and np.isfinite(population_gallery).all()),
        }
        for key in ("features_vs_native_val_control", "features_vs_native_train_query"):
            result["population"][key]["within_tolerance"] = (
                result["population"][key]["max_abs"] <= TOL["forward_max_abs"]
                and result["population"][key]["max_rel"] <= TOL["forward_max_rel"])
        result["population"]["gallery_vs_native"]["within_tolerance"] = (
            result["population"]["gallery_vs_native"]["max_abs"] <= TOL["forward_max_abs"])
    finally:
        adapter.close()

    result["scoring_ranking"] = score_and_rank(category, inputs, config)
    result["phase_seconds"] = round(time.time() - began, 1)
    return result


def score_and_rank(category: str, inputs: dict, config) -> dict:
    """G2 scores/ranks/recalls, all from the adapter's own float32 features."""
    from workbench.composition.model_encoder import Adapter

    blob = np.load(RAW / f"adapter.{category}.npz")
    native = np.load(RAW / f"native.{category}.npz")
    queries = blob["queries"].astype(np.float32)
    gallery = blob["gallery"].astype(np.float32)
    reference_positions = inputs["reference_positions"]
    target_positions = inputs["target_positions"]

    blobs = {"adapter": (queries, gallery)}
    floor_path = RAW / f"floor.native.{category}.npz"
    if floor_path.is_file():
        floor = np.load(floor_path)
        blobs["native_second_process"] = (floor["queries"].astype(np.float32),
                                          floor["gallery"].astype(np.float32))

    def ranks_and_recall(pair) -> tuple[list[int], list[int], np.ndarray]:
        """Masked ranks, unmasked ranks, and the UNMASKED score matrix (what `score()` must reproduce)."""
        q, g = pair
        masked, _ = native_ranks(q, g, reference_positions, target_positions, mask=True)
        unmasked, sims_unmasked = native_ranks(q, g, reference_positions, target_positions, mask=False)
        return masked, unmasked, sims_unmasked

    ranks_masked, ranks_unmasked, sims_unmasked = ranks_and_recall(blobs["adapter"])
    ranks_native_val, _, sims_native_val = ranks_and_recall(
        (native["population_val"].astype(np.float32), native["gallery"].astype(np.float32)))
    ranks_native_primary, _, _ = ranks_and_recall(
        (native["population_native"].astype(np.float32), native["gallery"].astype(np.float32)))
    # A second native process, byte-identical code and pixels: the noise floor of THIS model, so any
    # adapter-vs-native rank difference can be attributed rather than assumed.
    floor_block: dict = {"available": False}
    ranks_floor = None
    if "native_second_process" in blobs:
        ranks_floor, _, sims_floor = ranks_and_recall(blobs["native_second_process"])
        floor_block = {
            "available": True,
            "score_max_abs_vs_first_native_process": float(np.abs(
                sims_floor.astype(np.float64) - sims_native_val.astype(np.float64)).max()),
            "rank_differences_vs_first_native_process": {
                "count": sum(1 for a, b in zip(ranks_floor, ranks_native_val) if a != b),
                "head": [i for i, (a, b) in enumerate(zip(ranks_floor, ranks_native_val)) if a != b][:20]},
            "recall": recall(ranks_floor),
        }

    stored: dict[str, float] = {}
    stored_path = OUT / "raw" / "native" / "encoder.json"
    if stored_path.is_file():
        for row in json.loads(stored_path.read_text())["results"]:
            if row["category"] == category:
                stored = {f"r{k}": float(row[f"r{k}"]) for k in (1, 10, 50)}

    adapter = Adapter(config)
    adapter.work_root = RAW / "work" / f"{category}-rank"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    try:
        adapter_ranks = adapter.rank(queries, gallery, [[p] for p in target_positions], [None] * len(target_positions))
    finally:
        adapter.close()

    adapter = Adapter(config)
    adapter.work_root = RAW / "work" / f"{category}-score"
    adapter.work_root.mkdir(parents=True, exist_ok=True)
    pairs = min(64, len(target_positions))
    try:
        scores = adapter.score(queries[:pairs], gallery)
    finally:
        adapter.close()
    score_delta = np.abs(scores.astype(np.float64) - sims_unmasked[:pairs].astype(np.float64))

    # tie accounting for every rank disagreement on the UNMASKED matrix
    def classify(indices: list[int]) -> dict:
        non_ties = []
        for index in indices:
            row = sims_unmasked[index]
            low, high = sorted((ranks_unmasked[index], adapter_ranks[index]))
            window = np.argsort(-row)[low - 1:high]
            gap = float(np.abs(row[window] - row[target_positions[index]]).max()) if len(window) else 0.0
            if gap > TOL["tie_epsilon"]:
                non_ties.append({"index": index, "argsort_rank": ranks_unmasked[index],
                                 "adapter_rank": adapter_ranks[index], "score_gap": gap})
        return {"count": len(indices), "head": indices[:20], "non_tie_count": len(non_ties),
                "non_tie_head": non_ties[:10]}

    differences = [i for i, (a, b) in enumerate(zip(adapter_ranks, ranks_unmasked)) if a != b]
    vs_native_val = [i for i, (a, b) in enumerate(zip(ranks_masked, ranks_native_val)) if a != b]
    vs_native_primary = [i for i, (a, b) in enumerate(zip(ranks_masked, ranks_native_primary)) if a != b]

    # A residual rank difference between two FEATURE MATRICES cannot be judged on one matrix's tie
    # window: the scores themselves moved. Classify it by the score gap at the disputed boundary in
    # BOTH matrices and report the smallest, which is the operative one.
    cross_matrix = []
    for index in vs_native_val:
        low, high = sorted((ranks_masked[index], ranks_native_val[index]))
        gaps = []
        for sims in (sims_unmasked, sims_native_val):
            row = sims[index]
            window = np.argsort(-row)[low - 1:high]
            gaps.append(float(np.abs(row[window] - row[target_positions[index]]).max()) if len(window) else 0.0)
        cross_matrix.append({"index": index, "adapter_rank": ranks_masked[index],
                             "native_rank": ranks_native_val[index],
                             "adapter_matrix_gap": gaps[0], "native_matrix_gap": gaps[1],
                             "min_gap": min(gaps),
                             "explained_by_tie_epsilon": min(gaps) <= TOL["tie_epsilon"]})
    # Same score matrix, only the reference mask differs: the protocol divergence, quantified.
    mask_only = [i for i, (a, b) in enumerate(zip(ranks_unmasked, ranks_masked)) if a != b]
    return {
        "query_count": len(target_positions),
        "gallery_count": len(inputs["gallery_ids"]),
        "score_max_abs_delta_vs_dot": float(score_delta.max()),
        "score_pairs_compared": pairs,
        "score_max_abs_declared": TOL["score_max_abs"],
        "score_within_tolerance": float(score_delta.max()) <= TOL["score_max_abs"],
        "adapter_vs_argsort_unmasked": classify(differences),
        "adapter_style_recall": recall(adapter_ranks),
        "native_mask_only_effect": {
            "count": len(mask_only),
            "recall_without_mask": recall(ranks_unmasked),
            "recall_with_mask": recall(ranks_masked)},
        "native_convention_recall_on_adapter_features": recall(ranks_masked),
        "native_convention_recall_val_control": recall(ranks_native_val),
        "native_convention_recall_native_primary": recall(ranks_native_primary),
        "native_second_process_floor": floor_block,
        "stored_native_metrics_raw/native/encoder.json": stored,
        "rank_differences_vs_native_val_control": {
            "count": len(vs_native_val), "head": vs_native_val[:20],
            "classification": cross_matrix,
            "non_tie_count": sum(1 for item in cross_matrix if not item["explained_by_tie_epsilon"]),
            "min_gap_abs": min((item["min_gap"] for item in cross_matrix), default=0.0),
        },
        "rank_differences_vs_native_primary": {"count": len(vs_native_primary), "head": vs_native_primary[:20]},
        "native_metrics_recheck_source": "raw/native/encoder.json",
    }


# --------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    phase = argv[argv.index("--phase") + 1] if "--phase" in argv else "native"
    from workbench.backend.fashioniq_layout import CATEGORIES
    from workbench.backend.operator_config import resolve_config

    categories = (argv[argv.index("--category") + 1],) if "--category" in argv else CATEGORIES
    config = resolve_config()
    RAW.mkdir(parents=True, exist_ok=True)

    if phase == "probe":
        payload = probe_phase()
        (RAW / "probe.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"phase": "probe", "categories": list(payload["categories"])}, indent=1))
        return 0
    if phase == "determinism":
        tag = argv[argv.index("--tag") + 1]
        result = determinism_phase(tag)
        print(json.dumps(result, indent=1))
        return 0
    if phase == "determinism-compare":
        print(json.dumps(determinism_compare(), indent=1))
        return 0
    if phase == "floor":
        category = argv[argv.index("--category") + 1]
        native_floor_phase(category)
        return 0
    if phase == "layers":
        print(json.dumps(layers_phase(argv[argv.index("--tag") + 1]), indent=1))
        return 0
    if phase == "layers-compare":
        result = layers_compare()
        (RAW / "layers_compare.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"first_divergence": result["first_divergence"],
                          "layers": {k: {"max_abs": v["max_abs"], "bitwise": v["bitwise_equal"]}
                                     for k, v in result["layers"].items()}}, indent=1))
        return 0

    for category in categories:
        target = OUT / f"g1_g2.encoder.{category}.{phase}.json"
        print(f"# {category}/{phase}: start", flush=True)
        if phase == "native":
            payload = {"category": category, "phase": phase, "model_id": "encoder",
                       "protocol": "fashioniq_val_split", "tolerances": TOL,
                       "native": native_phase(category, config)}
        elif phase == "adapter":
            payload = {"category": category, "phase": phase, "model_id": "encoder",
                       "protocol": "fashioniq_val_split", "tolerances": TOL,
                       "adapter": adapter_phase(category, config)}
        else:
            raise SystemExit(f"unknown phase {phase!r}; use native, probe, or adapter")
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"# {category}/{phase}: wrote {target.name}", flush=True)
        if phase == "native":
            native = payload["native"]
            print(json.dumps({
                "identity": {key: native["layer1_identity"][key] for key in (
                    "gallery_order_identical", "reference_ids_identical", "target_ids_identical",
                    "model_text_identical_to_author_composition", "sample_indices")},
                "pixels": native["layer2_pixels"],
                "native_metrics_recheck": native["native_metrics_recheck"],
                "rank_recall": native["native_rank_recall_from_features"],
                "val_control_recall": native["val_transform_control_rank_recall"],
            }, indent=1))
        else:
            adapter = payload["adapter"]
            print(json.dumps({
                "sample_forward": adapter["layer4_forward_sample"],
                "pixels": adapter.get("layer2_pixels"),
                "batch": adapter["batch_size_1_vs_n"],
                "population": {k: v for k, v in adapter["population"].items() if k.endswith("tolerance")
                               or k in ("query_count", "gallery_count")},
                "rankings": adapter["scoring_ranking"],
            }, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
