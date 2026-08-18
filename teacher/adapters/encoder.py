import argparse
import gc
import json
import os
import sys
import string
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def load_cases(path: str | Path, limit: int | None = None) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if limit is not None:
        cases = cases[:limit]
    return cases


def resolve_image_path(
    image_root: str | Path,
    image_id: str,
    category: str,
) -> Path:
    """
    ENCODER/HINT source code expects category-subfolder images
    (<root>/<category>/<image_id>.jpg; some upstream code refers to the
    root as resized_image). Prefer that layout, then allow a flat image
    root only as a debugging fallback.
    """
    image_root = Path(image_root)

    candidates = []
    for extension in (".jpg", ".png", ".jpeg"):
        candidates.append(image_root / category / f"{image_id}{extension}")
    for extension in (".png", ".jpg", ".jpeg"):
        candidates.append(image_root / f"{image_id}{extension}")

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        f"Could not find image ID={image_id} category={category} "
        f"under {image_root}"
    )


def load_images(cases: list[dict], image_root: str | Path, preprocess) -> torch.Tensor:
    images = []

    for case in cases:
        image_path = resolve_image_path(
            image_root=image_root,
            image_id=case["reference_id"],
            category=case["category"],
        )

        with Image.open(image_path) as image:
            images.append(preprocess(image.convert("RGB")))

    return torch.stack(images)


def build_encoder(encoder_root: Path, checkpoint_path: Path, device: torch.device):
    if device.type != "cuda":
        raise RuntimeError(
            "AAAI25-ENCODER official code contains hard-coded .cuda() calls; run on CUDA."
        )

    torch.cuda.set_device(device.index if device.index is not None else 0)

    encoder_root = encoder_root.resolve()
    clip_checkpoint = encoder_root / "open_clip_pytorch_model.bin"
    if not clip_checkpoint.exists():
        raise FileNotFoundError(
            f"Missing required OpenCLIP checkpoint: {clip_checkpoint}"
        )

    sys.path.insert(0, str(encoder_root))
    os.chdir(encoder_root)

    import open_clip
    from model_try2 import Encoder

    # Official FashionIQ evaluation gets preprocess_val from this exact call.
    temporary_clip, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained=str(clip_checkpoint),
    )
    del temporary_clip
    gc.collect()

    # Official FashionIQ configuration from evaluate_model.py:
    # hidden_dim=512, P=3, wc=2, N_p=2, tau=0.1, dropout=0.5.
    model = Encoder(
        hidden_dim=512,
        dropout=0.5,
        local_token_num=3,
        t=0.1,
        wc=2,
        N_p=2,
        weighted=True,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint, strict=False)

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(
        f"missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    if load_result.missing_keys:
        print("Missing keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected keys:", load_result.unexpected_keys)

    return model, preprocess_train, preprocess_val


def load_correction_dicts(correction_root: Path) -> dict[str, dict[str, str]]:
    correction_dicts = {}

    for category in ("dress", "shirt", "toptee"):
        path = correction_root / f"correction_dict_{category}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing ENCODER FashionIQ correction dictionary: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            correction_dicts[category] = json.load(file)

    return correction_dicts


def correct_encoder_text(text: str, correction_dict: dict[str, str]) -> str:
    """
    Match ENCODER datasets.FashionIQ.correct_text():
      lowercase -> punctuation to spaces -> tokenize -> word correction.
    """
    translation = str.maketrans(
        {character: " " for character in string.punctuation}
    )
    tokens = str(text).lower().translate(translation).strip().split()

    return " ".join(correction_dict.get(token, token) for token in tokens)


def prepare_texts(
    cases: list[dict],
    key: str,
    correction_dicts: dict[str, dict[str, str]],
) -> list[str]:
    texts = []

    for case in cases:
        correction_dict = correction_dicts[case["category"]]

        if key == "full_text":
            # Match upstream concat_text exactly: correct each original
            # FashionIQ caption independently, then insert literal "and".
            caption_1 = correct_encoder_text(case["caption_1"], correction_dict)
            caption_2 = correct_encoder_text(case["caption_2"], correction_dict)
            text = f"{caption_1} and {caption_2}"
        elif key == "minus_1_text":
            text = correct_encoder_text(case["caption_2"], correction_dict)
        elif key == "minus_2_text":
            text = correct_encoder_text(case["caption_1"], correction_dict)
        else:
            raise KeyError(f"Unsupported controlled text key: {key}")

        texts.append(text)

    return texts


@torch.no_grad()
def compose_query(
    model,
    reference_images: torch.Tensor,
    texts: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    # Native boundary:
    # compose_feature -> fuse_local -> mean(dim=1) -> F.normalize
    fuse_local, _, _, _, _ = model.compose_feature(reference_images, texts)
    query_pre_norm = fuse_local.mean(dim=1)
    query_normalized = F.normalize(query_pre_norm, p=2, dim=-1)
    return query_pre_norm, query_normalized


def run_batch(model, preprocess, correction_dicts, cases, image_root, device):
    reference_images = load_images(cases, image_root, preprocess).to(device)

    full_texts = prepare_texts(cases, "full_text", correction_dicts)
    minus_1_texts = prepare_texts(cases, "minus_1_text", correction_dicts)
    minus_2_texts = prepare_texts(cases, "minus_2_text", correction_dicts)
    q_full_pre, q_full = compose_query(model, reference_images, full_texts)
    q_minus_1_pre, q_minus_1 = compose_query(model, reference_images, minus_1_texts)
    q_minus_2_pre, q_minus_2 = compose_query(model, reference_images, minus_2_texts)

    return {
        "sample_ids": [case["sample_id"] for case in cases],
        "reference_ids": [case["reference_id"] for case in cases],
        "target_ids": [case["target_id"] for case in cases],
        "categories": [case["category"] for case in cases],
        "q_full_pre_norm": q_full_pre.cpu(),
        "q_minus_1_pre_norm": q_minus_1_pre.cpu(),
        "q_minus_2_pre_norm": q_minus_2_pre.cpu(),
        "q_full": q_full.cpu(),
        "q_minus_1": q_minus_1.cpu(),
        "q_minus_2": q_minus_2.cpu(),
    }


def validate_outputs(outputs):
    tensor_keys = (
        "q_full_pre_norm",
        "q_minus_1_pre_norm",
        "q_minus_2_pre_norm",
        "q_full",
        "q_minus_1",
        "q_minus_2",
    )
    n = len(outputs["sample_ids"])
    for key in tensor_keys:
        tensor = outputs[key]
        if tensor.ndim != 2 or tensor.shape[0] != n:
            raise ValueError(f"Invalid {key} shape: {tuple(tensor.shape)}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{key} contains NaN or Inf")

    if outputs["q_full"].shape[-1] != 512:
        raise ValueError(
            f"Expected ENCODER query dim 512, got {outputs['q_full'].shape[-1]}"
        )

    norms = outputs["q_full"].norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ValueError("ENCODER normalized queries are not unit norm")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder-root",
        type=Path,
        default=Path("teacher/repos/AAAI25-ENCODER"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("teacher/audit/fashioniq_val_cases.json"),
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        required=True,
        help=(
            "FashionIQ image root containing dress/, shirt/, toptee/ "
            "subdirectories (or the upstream resized_image root)."
        ),
    )
    parser.add_argument(
        "--correction-root",
        type=Path,
        required=True,
        help=(
            "Directory containing correction_dict_dress.json, "
            "correction_dict_shirt.json, correction_dict_toptee.json"
        ),
    )
    parser.add_argument(
        "--upstream-query-train-transform",
        action="store_true",
        help=(
            "Reproduce ENCODER's upstream evaluator, which applies the "
            "OpenCLIP training transform to reference query images. "
            "Default is deterministic preprocess_val for geometry auditing."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("teacher/outputs/encoder/smoke.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("AAAI25-ENCODER adapter requires CUDA")

    device = torch.device(args.device)
    encoder_root = args.encoder_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cases_path = args.cases.resolve()
    image_root = args.image_root.resolve()
    correction_root = args.correction_root.resolve()
    output_path = args.output.resolve()

    cases = load_cases(cases_path, args.limit)
    if not cases:
        raise ValueError("No FashionIQ audit cases loaded")

    model, preprocess_train, preprocess_val = build_encoder(
        encoder_root, checkpoint_path, device
    )

    correction_dicts = load_correction_dicts(correction_root)

    if args.upstream_query_train_transform:
        preprocess = preprocess_train
        print(
            "WARNING: reproducing upstream ENCODER query train-transform; "
            "reference preprocessing is stochastic."
        )
    else:
        preprocess = preprocess_val
        print(
            "Using deterministic OpenCLIP preprocess_val for geometry audit."
        )

    outputs = {
        "sample_ids": [],
        "reference_ids": [],
        "target_ids": [],
        "categories": [],
    }
    tensor_keys = (
        "q_full_pre_norm",
        "q_minus_1_pre_norm",
        "q_minus_2_pre_norm",
        "q_full",
        "q_minus_1",
        "q_minus_2",
    )
    tensor_batches = {key: [] for key in tensor_keys}

    for start in range(0, len(cases), args.batch_size):
        batch_output = run_batch(
            model,
            preprocess,
            correction_dicts,
            cases[start:start + args.batch_size],
            image_root,
            device,
        )
        for key in ("sample_ids", "reference_ids", "target_ids", "categories"):
            outputs[key].extend(batch_output[key])
        for key in tensor_keys:
            tensor_batches[key].append(batch_output[key])

    for key in tensor_keys:
        outputs[key] = torch.cat(tensor_batches[key], dim=0)

    validate_outputs(outputs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, output_path)

    delta_1 = outputs["q_full_pre_norm"] - outputs["q_minus_1_pre_norm"]
    delta_2 = outputs["q_full_pre_norm"] - outputs["q_minus_2_pre_norm"]

    print("\n=== ENCODER smoke test ===")
    print("q_full_pre_norm:", tuple(outputs["q_full_pre_norm"].shape))
    print("q_minus_1_pre_norm:", tuple(outputs["q_minus_1_pre_norm"].shape))
    print("q_minus_2_pre_norm:", tuple(outputs["q_minus_2_pre_norm"].shape))
    print("mean ||delta_1||:", delta_1.norm(dim=-1).mean().item())
    print("mean ||delta_2||:", delta_2.norm(dim=-1).mean().item())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
