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
        image_path = resolve_image_path(image_root=image_root, image_id=case["reference_id"])
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image = preprocess(image)
        images.append(image)
    return torch.stack(images)

@torch.no_grad()
def encode_reference(model, images: torch.Tensor) -> torch.Tensor:
    with model.maybe_autocast():
        reference_states = model.visual_encoder(images)
        reference_states = model.ln_vision(reference_states)
    return reference_states.float()

@torch.no_grad()
def encode_target(
    model,
    images: torch.Tensor,
) -> torch.Tensor:
    """
    Reproduce QuRe's native extract_target_features() path.

    Returns:
        normalized target query tokens: [B, 32, D]
    """
    device = images.device
    with model.maybe_autocast():
        image_states = model.visual_encoder(images)
        image_states = model.ln_vision(image_states)
    image_states = image_states.float()
    image_attention_mask = torch.ones(
        image_states.size()[:-1],
        dtype=torch.long,
        device=device,
    )
    query_tokens = model.query_tokens.expand(
        image_states.shape[0],
        -1,
        -1,
    )
    if query_tokens.size(1) != 32:
        raise ValueError(f"QuRe FashionIQ checkpoint is expected to expose 32 target query tokens, got {query_tokens.size(1)}")
    query_output = model.Qformer.bert(
        query_embeds=query_tokens,
        encoder_hidden_states=image_states,
        encoder_attention_mask=image_attention_mask,
        return_dict=True,
    )
    target_pre_norm = model.vision_proj(query_output.last_hidden_state)
    target_normalized = F.normalize(
        target_pre_norm,
        dim=-1,
    )
    return target_normalized

@torch.no_grad()
def compose_query(model, reference_states: torch.Tensor, texts: list[str], txt_processor) -> tuple[torch.Tensor, torch.Tensor]:
    device = reference_states.device
    texts = [txt_processor(text) for text in texts]
    text_tokens = model.tokenizer(texts, padding="max_length", truncation=True, max_length=model.max_txt_len, return_tensors="pt").to(device)
    reference_attention_mask = torch.ones(reference_states.size()[:-1], dtype=torch.long, device=device)
    query_tokens = model.query_tokens.expand(reference_states.shape[0], -1, -1)
    query_attention_mask = torch.ones(query_tokens.size()[:-1], dtype=torch.long, device=device)
    attention_mask = torch.cat([query_attention_mask, text_tokens.attention_mask], dim=1)
    query_output = model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=query_tokens,
        attention_mask=attention_mask,
        encoder_hidden_states=reference_states,
        encoder_attention_mask=reference_attention_mask,
        return_dict=True,
    )
    query_token_states = query_output.last_hidden_state[:, : query_tokens.size(1), :]
    query_pre_norm = model.text_proj(query_token_states).mean(dim=1)
    query_normalized = F.normalize(query_pre_norm, dim=-1)
    return query_pre_norm, query_normalized

def build_qure(qure_root: Path, config_path: Path, checkpoint_path: Path, device: torch.device):
    sys.path.insert(0, str(qure_root))
    os.chdir(qure_root)
    from models import create_qure_models
    from data.utils import targetpad_transform
    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    model, txt_processors = create_qure_models(config, device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    preprocess = targetpad_transform(target_ratio=1.25, dim=config["img_size"])
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(load_result)
    return (model, txt_processors["eval"], preprocess)

def run_batch(model, txt_processor, preprocess, cases: list[dict], image_root: Path, device: torch.device) -> dict[str, object]:
    reference_images = load_images(cases=cases, image_root=image_root, preprocess=preprocess).to(device)
    reference_states = encode_reference(
        model=model,
        images=reference_images,
    )
    full_texts = [case["full_text"] for case in cases]
    minus_1_texts = [case["minus_1_text"] for case in cases]
    minus_2_texts = [case["minus_2_text"] for case in cases]
    q_full_pre, q_full = compose_query(
        model=model,
        reference_states=reference_states,
        texts=full_texts,
        txt_processor=txt_processor,
    )
    q_minus_1_pre, q_minus_1 = compose_query(
        model=model,
        reference_states=reference_states,
        texts=minus_1_texts,
        txt_processor=txt_processor,
    )
    q_minus_2_pre, q_minus_2 = compose_query(
        model=model,
        reference_states=reference_states,
        texts=minus_2_texts,
        txt_processor=txt_processor,
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qure-root", type=Path, default=Path("teacher/repos/QuRe"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("teacher/repos/QuRe/configs/fashionIQ/eval.json"),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path("teacher/audit/fashioniq_val_cases.json"))
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("teacher/outputs/qure/smoke.pt"))
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()

def main():
    args = parse_args()
    qure_root = args.qure_root.resolve()
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cases_path = args.cases.resolve()
    image_root = args.image_root.resolve()
    output_path = args.output.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cases = load_cases(path=cases_path, limit=args.limit)
    model, txt_processor, preprocess = build_qure(qure_root=qure_root, config_path=config_path, checkpoint_path=checkpoint_path, device=device)
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
        batch_cases = cases[start : start + args.batch_size]
        batch_output = run_batch(
            model=model,
            txt_processor=txt_processor,
            preprocess=preprocess,
            cases=batch_cases,
            image_root=image_root,
            device=device,
        )
        for key in ("sample_ids", "reference_ids", "target_ids", "categories"):
            outputs[key].extend(batch_output[key])
        for key in tensor_keys:
            tensor_batches[key].append(batch_output[key])
    for key in tensor_keys:
        outputs[key] = torch.cat(tensor_batches[key], dim=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, output_path)
    print()
    print("=== QuRe smoke test ===")
    print("q_full_pre_norm:", tuple(outputs["q_full_pre_norm"].shape))
    print("q_minus_1_pre_norm:", tuple(outputs["q_minus_1_pre_norm"].shape))
    print("q_minus_2_pre_norm:", tuple(outputs["q_minus_2_pre_norm"].shape))
    delta_1 = outputs["q_full_pre_norm"] - outputs["q_minus_1_pre_norm"]
    delta_2 = outputs["q_full_pre_norm"] - outputs["q_minus_2_pre_norm"]
    print("mean ||delta_1||:", delta_1.norm(dim=-1).mean().item())
    print("mean ||delta_2||:", delta_2.norm(dim=-1).mean().item())
    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
