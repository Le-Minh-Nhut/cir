import argparse
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


def load_full_checkpoint(checkpoint_path: Path, device: torch.device):
    # Upstream HINT training uses torch.save(model, ...).
    # PyTorch >=2.6 needs weights_only=False for full-object checkpoints.
    try:
        return torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def build_hint(hint_root: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(hint_root))
    os.chdir(hint_root)

    # Import native classes before unpickling.
    from lavis.models.blip2_models.HINT import HINT
    from lavis.processors.blip_processors import BlipCaptionProcessor
    from data_utils import targetpad_transform

    loaded = load_full_checkpoint(checkpoint_path, device)

    if isinstance(loaded, torch.nn.Module):
        model = loaded
        # Upstream pretrain config uses eval text processor "blip_caption"
        # with default prompt/max_words.
        txt_processor = BlipCaptionProcessor()

    elif isinstance(loaded, dict):
        # Defensive fallback for a re-exported state-dict checkpoint.
        from lavis.models import load_model_and_preprocess

        model, _, txt_processors = load_model_and_preprocess(
            name="HINT",
            model_type="pretrain",
            is_eval=False,
            device=device,
        )

        if "model" in loaded and isinstance(loaded["model"], dict):
            state_dict = loaded["model"]
        elif "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
            state_dict = loaded["state_dict"]
        else:
            state_dict = loaded

        load_result = model.load_state_dict(state_dict, strict=False)
        print(
            "HINT state-dict fallback: "
            f"missing={len(load_result.missing_keys)} "
            f"unexpected={len(load_result.unexpected_keys)}"
        )
        if load_result.missing_keys:
            print("Missing keys:", load_result.missing_keys)
        if load_result.unexpected_keys:
            print("Unexpected keys:", load_result.unexpected_keys)

        txt_processor = txt_processors["eval"]

    else:
        raise TypeError(f"Unsupported HINT checkpoint type: {type(loaded)!r}")

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    preprocess = targetpad_transform(1.25, 224)
    print(f"Checkpoint loaded: {checkpoint_path}")
    return model, txt_processor, preprocess


def load_correction_dicts(correction_root: Path) -> dict[str, dict[str, str]]:
    correction_dicts = {}

    for category in ("dress", "shirt", "toptee"):
        path = correction_root / f"correction_dict_{category}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing HINT FashionIQ correction dictionary: {path}"
            )

        with path.open("r", encoding="utf-8") as file:
            correction_dicts[category] = json.load(file)

    return correction_dicts


def correct_hint_text(text: str, correction_dict: dict[str, str]) -> str:
    """
    Match HINT datasets.FashionIQ.correct_text():
      lowercase -> punctuation to spaces -> word-level correction.
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
            caption_1 = correct_hint_text(case["caption_1"], correction_dict)
            caption_2 = correct_hint_text(case["caption_2"], correction_dict)
            text = f"{caption_1} and {caption_2}"
        elif key == "minus_1_text":
            text = correct_hint_text(case["caption_2"], correction_dict)
        elif key == "minus_2_text":
            text = correct_hint_text(case["caption_1"], correction_dict)
        else:
            raise KeyError(f"Unsupported controlled text key: {key}")

        texts.append(text)

    return texts


@torch.no_grad()
def compose_query(
    model,
    reference_images: torch.Tensor,
    texts: list[str],
    txt_processor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Exact upstream extract_retrieval_compose path, but expose text_proj
    # output immediately before F.normalize.
    device = reference_images.device
    texts = [txt_processor(text) for text in texts]

    with model.maybe_autocast():
        reference_embeds = model.ln_vision(
            model.visual_encoder(reference_images)
        )
    reference_embeds = reference_embeds.float()

    image_atts = torch.ones(
        reference_embeds.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    query_tokens = model.query_tokens.expand(
        reference_embeds.shape[0], -1, -1
    )

    if query_tokens.size(1) != 32:
        raise ValueError(
            "HINT upstream code indexes position 32, but checkpoint has "
            f"{query_tokens.size(1)} query tokens"
        )

    query_atts = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    text_tokens = model.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)

    attention_mask = torch.cat(
        [query_atts, text_tokens.attention_mask],
        dim=1,
    )

    fusion_output = model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=query_tokens,
        attention_mask=attention_mask,
        encoder_hidden_states=reference_embeds,
        encoder_attention_mask=image_atts,
        return_dict=True,
    )

    query_pre_norm = model.text_proj(
        fusion_output.last_hidden_state[:, 32, :]
    )
    query_normalized = F.normalize(query_pre_norm, dim=-1)

    return query_pre_norm, query_normalized


def run_batch(model, txt_processor, preprocess, correction_dicts, cases, image_root, device):
    reference_images = load_images(cases, image_root, preprocess).to(device)

    full_texts = prepare_texts(cases, "full_text", correction_dicts)
    minus_1_texts = prepare_texts(cases, "minus_1_text", correction_dicts)
    minus_2_texts = prepare_texts(cases, "minus_2_text", correction_dicts)
    q_full_pre, q_full = compose_query(model, reference_images, full_texts, txt_processor)
    q_minus_1_pre, q_minus_1 = compose_query(model, reference_images, minus_1_texts, txt_processor)
    q_minus_2_pre, q_minus_2 = compose_query(model, reference_images, minus_2_texts, txt_processor)

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

    if outputs["q_full"].shape[-1] != 256:
        raise ValueError(
            f"Expected HINT query dim 256, got {outputs['q_full'].shape[-1]}"
        )

    norms = outputs["q_full"].norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ValueError("HINT normalized queries are not unit norm")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hint-root",
        type=Path,
        default=Path("teacher/repos/ICASSP26-HINT"),
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
        "--output",
        type=Path,
        default=Path("teacher/outputs/hint/smoke.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    hint_root = args.hint_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cases_path = args.cases.resolve()
    image_root = args.image_root.resolve()
    correction_root = args.correction_root.resolve()
    output_path = args.output.resolve()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cases = load_cases(cases_path, args.limit)
    if not cases:
        raise ValueError("No FashionIQ audit cases loaded")

    model, txt_processor, preprocess = build_hint(hint_root, checkpoint_path, device)
    correction_dicts = load_correction_dicts(correction_root)

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
            txt_processor,
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

    print("\n=== HINT smoke test ===")
    print("q_full_pre_norm:", tuple(outputs["q_full_pre_norm"].shape))
    print("q_minus_1_pre_norm:", tuple(outputs["q_minus_1_pre_norm"].shape))
    print("q_minus_2_pre_norm:", tuple(outputs["q_minus_2_pre_norm"].shape))
    print("mean ||delta_1||:", delta_1.norm(dim=-1).mean().item())
    print("mean ||delta_2||:", delta_2.norm(dim=-1).mean().item())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
