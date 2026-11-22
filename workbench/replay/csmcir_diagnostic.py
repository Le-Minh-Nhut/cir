"""Run one preregistered CSMCIR native or production-adapter diagnostic phase."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("native", "adapter"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--native-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--work-root", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample = manifest["sample"]
    category = sample["category"]
    query_paths = [Path(path) for path in sample["query_image_paths"]]
    gallery_paths = [Path(path) for path in sample["gallery_image_paths"]]
    texts = sample["texts"]
    query_ids, gallery_ids = sample["query_order"], sample["gallery_order"]
    if category not in ("dress", "shirt", "toptee") or not query_paths or not gallery_paths:
        parser.error("manifest must name one supported category and non-empty query/gallery paths")
    if len(query_paths) != len(texts) or len(query_paths) != len(query_ids) or len(gallery_paths) != len(gallery_ids):
        parser.error("manifest paths, texts and ordered IDs must have matching lengths")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.phase == "native":
        import torch
        from PIL import Image
        from workbench.replay.lavis_processors import prepare

        source = ROOT / "workbench/third_party/CSMCIR"
        prepare(source / "src")
        from lavis.models import load_model_and_preprocess
        from data_utils_csmcir import targetpad_transform

        model, _, processors = load_model_and_preprocess(
            name="blip2_cir_align_prompt_csmcir", model_type="pretrain", is_eval=True, device="cuda")
        checkpoint = Path(manifest["checkpoint"]["path"])
        state = torch.load(checkpoint, map_location="cuda", weights_only=False)
        class_key = model.__class__.__name__
        if class_key in state:
            weights = state[class_key]
        else:
            weights = [value for key, value in state.items() if key != "epoch" and isinstance(value, dict)]
            if len(weights) != 1:
                raise RuntimeError(f"expected one CSMCIR checkpoint state dictionary, found {len(weights)}")
            weights = weights[0]
        result = model.load_state_dict(weights, strict=False)
        missing_keys, unexpected_keys = list(result.missing_keys), list(result.unexpected_keys)
        if missing_keys:
            raise RuntimeError(f"checkpoint missing keys: {missing_keys}")
        model.eval()
        transform = targetpad_transform(1.25, 224)
        caption_root = Path(manifest["captions"]["root"])
        caption_map = json.loads((caption_root / f"{category}_cot_val.json").read_text(encoding="utf-8"))
        target_captions = [caption_map[image_id]["Final_Caption"] for image_id in gallery_ids]
        if any(not caption for caption in target_captions):
            raise ValueError("empty target caption in preregistered gallery")
        captions = [processors["eval"](text) for text in texts]

        def load_images(paths: list[Path]) -> torch.Tensor:
            tensors = []
            for path in paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            return torch.stack(tensors).cuda()

        queries, gallery = load_images(query_paths), load_images(gallery_paths)
        tokens = model.tokenizer(captions, padding="max_length", truncation=True, max_length=model.max_txt_len,
                                 return_tensors="pt").to(model.device)
        with torch.no_grad():
            with model.maybe_autocast():
                reference = model.ln_vision(model.visual_encoder(queries)).float()
            target_features, _ = model.extract_target_caption_features(gallery, target_captions, mode="mean")
            query_outputs = []
            handle = model.text_proj.register_forward_hook(lambda _module, _args, output: query_outputs.append(output))
            try:
                native_scores = model.inference(reference, target_features, captions, None)
                singleton_scores = model.inference(reference[:1], target_features[:1], captions[:1], None)
            finally:
                handle.remove()
        if len(query_outputs) != 2:
            raise RuntimeError(f"expected two native text projections, got {len(query_outputs)}")
        query_features = torch.nn.functional.normalize(query_outputs[0], dim=-1)
        singleton_query = torch.nn.functional.normalize(query_outputs[1], dim=-1)
        scores = native_scores.reshape(len(query_paths), len(gallery_paths)).float()
        singleton_score = singleton_scores.reshape(1, 1).float()
        np.savez_compressed(args.output, feature_layout="native_query256_gallery8192_v2", query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
                            query_features=query_features.float().cpu().numpy(),
                            target_features=target_features.float().cpu().numpy(),
                            singleton_target=target_features[:1].float().cpu().numpy(),
                            singleton_query=singleton_query.float().cpu().numpy(),
                            singleton_score=singleton_score.cpu().numpy(), input_ids=tokens.input_ids.cpu().numpy(),
                            attention_mask=tokens.attention_mask.cpu().numpy(),
                            missing_keys=np.asarray(missing_keys), unexpected_keys=np.asarray(unexpected_keys),
                            reference_visual_shape=np.asarray(reference.shape), target_token_shape=np.asarray(target_features.shape))
        print(json.dumps({"phase": "native", "category": category, "query_count": len(query_paths),
                          "missing_keys": missing_keys, "unexpected_keys": unexpected_keys,
                          "reference_visual_shape": list(reference.shape), "target_token_shape": list(target_features.shape),
                          "input_ids_shape": list(tokens.input_ids.shape), "attention_mask_shape": list(tokens.attention_mask.shape),
                          "scores": scores.cpu().tolist(), "singleton_score": singleton_score.cpu().tolist()}, sort_keys=True), flush=True)
        del model
        torch.cuda.empty_cache()
        return 0

    if args.work_root is None:
        parser.error("--work-root required for adapter phase")
    from workbench.composition.model_csmcir import Adapter
    from workbench.backend.operator_config import resolve_config

    adapter = Adapter(resolve_config(), checkpoint_id="fashioniq")
    adapter.work_root = args.work_root.resolve()
    try:
        queries = adapter.encode_queries(query_paths, texts, [category] * len(texts))
        targets = adapter.encode_targets(gallery_paths, [category] * len(gallery_paths))
        scores = adapter.score(queries, targets)
        target_positions = [[gallery_ids.index(image_id)] for image_id in manifest["sample"]["target_order"]]
        ranks = adapter.rank(queries, targets, target_positions, [category] * len(query_ids))
        singleton_query = adapter.encode_queries(query_paths[:1], texts[:1], [category])
        singleton_target = adapter.encode_targets(gallery_paths[:1], [category])
        singleton_score = adapter.score(singleton_query, singleton_target)
        singleton_rank = adapter.rank(singleton_query, singleton_target, [[0]], [category])
        native_path = args.native_output or args.manifest.parent / "csmcir-native.npz"
        with np.load(native_path, allow_pickle=False) as native:
            query_error = float(np.max(np.abs(queries - native["query_features"])))
            target_error = float(np.max(np.abs(targets.reshape(len(targets), 32, 256) - native["target_features"])))
            score_error = float(np.max(np.abs(scores - native["native_scores"])))
            singleton_query_error = float(np.max(np.abs(singleton_query - native["singleton_query"])))
            singleton_target_error = float(np.max(np.abs(singleton_target.reshape(1, 32, 256) - native["singleton_target"])))
            singleton_score_error = float(np.max(np.abs(singleton_score - native["singleton_score"])))
            native_rank = np.argsort(-native["native_scores"], axis=1)
            native_singleton_rank = int(np.argmax(native["singleton_score"], axis=1)[0]) + 1
        adapter_rank = np.argsort(-scores, axis=1)
        tolerance = manifest["tolerances"]
        errors = {"query": query_error, "target": target_error, "score": score_error,
                  "singleton_query": singleton_query_error, "singleton_target": singleton_target_error,
                  "singleton_score": singleton_score_error}
        limits = {"query": tolerance["query_feature_max_abs"], "target": tolerance["target_feature_max_abs"],
                  "score": tolerance["score_max_abs"], "singleton_query": tolerance["query_feature_max_abs"],
                  "singleton_target": tolerance["target_feature_max_abs"], "singleton_score": tolerance["score_max_abs"]}
        violations = {key: value for key, value in errors.items() if not np.isfinite(value) or value > limits[key]}
        if violations or not np.array_equal(native_rank, adapter_rank) or singleton_rank != [1] or native_singleton_rank != 1:
            raise AssertionError(f"CSMCIR native/adapter diagnostic mismatch: errors={errors}, violations={violations}, singleton_rank={singleton_rank}, native_singleton_rank={native_singleton_rank}")
        np.savez_compressed(args.output, query_ids=np.asarray(query_ids), gallery_ids=np.asarray(gallery_ids),
                            query_features=queries, target_features=targets, adapter_scores=scores,
                            singleton_query=singleton_query, singleton_target=singleton_target,
                            singleton_score=singleton_score, singleton_rank=np.asarray(singleton_rank),
                            adapter_ranks=np.asarray(ranks), native_rank=native_rank, adapter_rank=adapter_rank)
        print(json.dumps({"phase": "adapter", "category": category, "query_count": len(queries),
                          "gallery_count": len(targets), "query_feature_max_abs_error": query_error,
                          "target_feature_max_abs_error": target_error, "score_max_abs_error": score_error,
                          "singleton_query_error": singleton_query_error, "singleton_target_error": singleton_target_error,
                          "singleton_score_error": singleton_score_error, "singleton_rank": singleton_rank,
                          "positive_ranks": ranks,
                          "ranking_equal": bool(np.array_equal(native_rank, adapter_rank))}, sort_keys=True), flush=True)
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
