import argparse
import json
import os
import sys
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


def resolve_image_path(image_root: str | Path, image_id: str) -> Path:
    image_root = Path(image_root)
    for extension in (".png", ".jpg", ".jpeg"):
        path = image_root / f"{image_id}{extension}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find image for ID: {image_id}")


def load_images(cases: list[dict], image_root: str | Path, preprocess) -> torch.Tensor:
    images = []
    for case in cases:
        image_path = resolve_image_path(image_root, case["reference_id"])
        with Image.open(image_path) as image:
            images.append(preprocess(image.convert("RGB")))
    return torch.stack(images)


def build_sprc(
    sprc_root: Path,
    checkpoint_path: Path,
    backbone: str,
    device: torch.device,
):
    sprc_root = sprc_root.resolve()
    src_root = sprc_root / "src"

    sys.path.insert(0, str(src_root))
    os.chdir(sprc_root)

    from data_utils import targetpad_transform
    from lavis.models import load_model_and_preprocess

    # Official SPRC FashionIQ model name.
    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_cir_align_prompt",
        model_type=backbone,
        is_eval=False,
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(
            "SPRC official checkpoint should be a dict keyed by model class name"
        )

    model_key = model.__class__.__name__
    if model_key not in checkpoint:
        raise KeyError(
            f"Expected SPRC checkpoint key {model_key!r}; "
            f"available keys: {list(checkpoint.keys())}"
        )

    load_result = model.load_state_dict(
        checkpoint[model_key],
        strict=False,
    )

    model = model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    preprocess = targetpad_transform(1.25, 224)

    print(f"Checkpoint loaded: {checkpoint_path}")
    print(
        f"missing={len(load_result.missing_keys)} "
        f"unexpected={len(load_result.unexpected_keys)}"
    )
    if load_result.missing_keys:
        print("Missing keys:", load_result.missing_keys)
    if load_result.unexpected_keys:
        print("Unexpected keys:", load_result.unexpected_keys)

    return model, txt_processors["eval"], preprocess


@torch.no_grad()
def encode_reference(model, reference_images: torch.Tensor) -> torch.Tensor:
    # Matches the raw reference-image representation stored by upstream
    # extract_target_features and later consumed by inference().
    with model.maybe_autocast():
        reference_embeds = model.ln_vision(
            model.visual_encoder(reference_images)
        )
    return reference_embeds.float()


@torch.no_grad()
def compose_query(
    model,
    reference_embeds: torch.Tensor,
    texts: list[str],
    txt_processor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Exact SPRC composed-query path before native target scoring.

    Stage 1:
      learned 32 query tokens + text cross-attend to reference image.

    Stage 2:
      Stage-1's first 32 multimodal tokens become query_embeds for a
      second Q-Former text pass.

    Native query:
      text_proj(second_stage_hidden[:, 32, :]) -> F.normalize
    """

    device = reference_embeds.device
    texts = [txt_processor(text) for text in texts]

    image_atts = torch.ones(
        reference_embeds.size()[:-1],
        dtype=torch.long,
        device=device,
    )

    query_tokens = model.query_tokens.expand(
        reference_embeds.shape[0],
        -1,
        -1,
    )
    if query_tokens.size(1) != 32:
        raise ValueError(
            "SPRC upstream code indexes position 32, but checkpoint has "
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

    second_stage_query_tokens = fusion_output.last_hidden_state[:, :32, :]

    text_output = model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=second_stage_query_tokens,
        attention_mask=attention_mask,
        return_dict=True,
    )

    query_pre_norm = model.text_proj(
        text_output.last_hidden_state[:, 32, :]
    )
    query_normalized = F.normalize(query_pre_norm, dim=-1)

    return query_pre_norm, query_normalized


def run_batch(model, txt_processor, preprocess, cases, image_root, device):
    reference_images = load_images(cases, image_root, preprocess).to(device)
    reference_embeds = encode_reference(model, reference_images)

    q_full_pre, q_full = compose_query(
        model,
        reference_embeds,
        [case["full_text"] for case in cases],
        txt_processor,
    )
    q_minus_1_pre, q_minus_1 = compose_query(
        model,
        reference_embeds,
        [case["minus_1_text"] for case in cases],
        txt_processor,
    )
    q_minus_2_pre, q_minus_2 = compose_query(
        model,
        reference_embeds,
        [case["minus_2_text"] for case in cases],
        txt_processor,
    )

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
            f"Expected SPRC query dim 256, got {outputs['q_full'].shape[-1]}"
        )

    norms = outputs["q_full"].norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-4, rtol=1e-4):
        raise ValueError("SPRC normalized queries are not unit norm")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sprc-root",
        type=Path,
        default=Path("teacher/repos/SPRC"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        type=str,
        default="pretrain",
        choices=("pretrain", "pretrain_vitL"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("teacher/audit/fashioniq_val_cases.json"),
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("teacher/outputs/sprc/smoke.pt"),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    sprc_root = args.sprc_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cases_path = args.cases.resolve()
    image_root = args.image_root.resolve()
    output_path = args.output.resolve()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    cases = load_cases(cases_path, args.limit)
    if not cases:
        raise ValueError("No FashionIQ audit cases loaded")

    model, txt_processor, preprocess = build_sprc(sprc_root, checkpoint_path, args.backbone, device)

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

    print("\n=== SPRC smoke test ===")
    print("q_full_pre_norm:", tuple(outputs["q_full_pre_norm"].shape))
    print("q_minus_1_pre_norm:", tuple(outputs["q_minus_1_pre_norm"].shape))
    print("q_minus_2_pre_norm:", tuple(outputs["q_minus_2_pre_norm"].shape))
    print("mean ||delta_1||:", delta_1.norm(dim=-1).mean().item())
    print("mean ||delta_2||:", delta_2.norm(dim=-1).mean().item())
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
