"""Run one preregistered SPRC native or composition-adapter diagnostic phase."""
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
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sample = manifest["sample"]
    query_paths = [Path(path) for path in sample["query_image_paths"]]
    gallery_paths = [Path(path) for path in sample["gallery_image_paths"]]
    texts = sample["texts"]
    ids = {"query_ids": sample["query_order"], "gallery_ids": sample["gallery_order"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.phase == "native":
        import torch
        from PIL import Image
        from workbench.replay.lavis_processors import prepare

        source = ROOT / "workbench/third_party/SPRC"
        prepare(source / "src")
        from lavis.models import load_model_and_preprocess
        from data_utils import targetpad_transform

        model, _, processors = load_model_and_preprocess(name="blip2_cir_align_prompt", model_type="pretrain_vitL",
                                                          is_eval=False, device="cuda")
        state = torch.load(Path(manifest["checkpoint"]["path"]), map_location="cuda", weights_only=False)
        result = model.load_state_dict(state[model.__class__.__name__], strict=True)
        model.eval()
        transform = targetpad_transform(1.25, 224)

        def images(paths):
            tensors = []
            for path in paths:
                with Image.open(path) as image:
                    tensors.append(transform(image.convert("RGB")))
            return torch.stack(tensors).cuda()

        query = images(query_paths)
        gallery = images(gallery_paths)
        captions = [processors["eval"](text) for text in texts]
        tokens = model.tokenizer(captions, padding="max_length", truncation=True, max_length=model.max_txt_len,
                                 return_tensors="pt").to(model.device)
        with torch.no_grad():
            with model.maybe_autocast():
                reference = model.ln_vision(model.visual_encoder(query))
            reference = reference.float()
            target_features, _ = model.extract_target_features(gallery, mode="mean")
            query_outputs = []
            hook = model.text_proj.register_forward_hook(lambda _module, _args, output: query_outputs.append(output))
            try:
                scores = model.inference(reference, target_features, captions)
            finally:
                hook.remove()
        if len(query_outputs) != 1:
            raise RuntimeError(f"expected one native projected query, got {len(query_outputs)}")
        query_features = torch.nn.functional.normalize(query_outputs[0], dim=-1)
        scores = scores.reshape(len(query_paths), len(gallery_paths))
        np.savez_compressed(args.output, feature_layout="native_query256_gallery8192_v2", **ids,
                            query_features=query_features.cpu().numpy().astype(np.float32),
                            target_features=target_features.cpu().numpy().astype(np.float32),
                            native_scores=scores.reshape(len(query_paths), len(gallery_paths)).cpu().numpy().astype(np.float32),
                            input_ids=tokens.input_ids.cpu().numpy(), attention_mask=tokens.attention_mask.cpu().numpy(),
                            missing_keys=np.asarray(result.missing_keys), unexpected_keys=np.asarray(result.unexpected_keys),
                            reference_visual_shape=np.asarray(reference.shape), target_token_shape=np.asarray(target_features.shape))
        print(json.dumps({"phase":"native","checkpoint":str(manifest["checkpoint"]["path"]),
                          "missing_keys":list(result.missing_keys),"unexpected_keys":list(result.unexpected_keys),
                          "query_count":len(query_paths),"gallery_count":len(gallery_paths),
                          "reference_visual_shape":list(reference.shape),"target_token_shape":list(target_features.shape),
                          "input_ids_shape":list(tokens.input_ids.shape),"attention_mask_shape":list(tokens.attention_mask.shape),
                          "scores":scores.reshape(len(query_paths),len(gallery_paths)).cpu().tolist()},sort_keys=True),flush=True)
        del model
        torch.cuda.empty_cache()
        return 0

    if args.work_root is None:
        parser.error("--work-root required for adapter phase")
    from workbench.composition.model_sprc import Adapter
    adapter = Adapter(checkpoint_id="sprc_fiq_vitl")
    adapter.work_root = args.work_root.resolve()
    try:
        queries = adapter.encode_queries(query_paths, texts, [None] * len(texts))
        targets = adapter.encode_targets(gallery_paths, [None] * len(gallery_paths))
        scores = adapter.score(queries, targets)
        native_file = args.native_output or args.manifest.parent / "sprc-fiq-vitl-dress-smoke-native.npz"
        with np.load(native_file, allow_pickle=False) as native:
            qerror = float(np.max(np.abs(queries - native["query_features"])))
            terror = float(np.max(np.abs(targets.reshape(len(targets), 32, 256) - native["target_features"])))
            serror = float(np.max(np.abs(scores - native["native_scores"])))
            ranking = np.argsort(-scores, axis=1)
            native_ranking = np.argsort(-native["native_scores"], axis=1)
        np.savez_compressed(args.output, **ids, query_features=queries, target_features=targets,
                            adapter_scores=scores.astype(np.float32), ranking=ranking,
                            native_ranking=native_ranking)
        print(json.dumps({"phase":"adapter","query_count":len(queries),"gallery_count":len(targets),
                          "query_feature_max_abs_error":qerror,"target_feature_max_abs_error":terror,
                          "score_max_abs_error":serror,"ranking_equal":bool(np.array_equal(ranking,native_ranking)),
                          "scores":scores.tolist()},sort_keys=True),flush=True)
    finally:
        adapter.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())