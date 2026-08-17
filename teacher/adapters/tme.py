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
def encode_vit_states(model, images: torch.Tensor) -> torch.Tensor:
    with torch.autocast(device_type=images.device.type, enabled=images.device.type == "cuda"):
        vit_states = model.vit_encode(images)

    return vit_states


@torch.no_grad()
def encode_reference(model, vit_states: torch.Tensor) -> torch.Tensor:
    reference_tokens = model.encode_image(vit_states)
    return reference_tokens


@torch.no_grad()
def compose_query(
    model, reference_tokens: torch.Tensor, texts: list[str], txt_processor
) -> tuple[torch.Tensor, torch.Tensor]:
    device = reference_tokens.device
    texts = [txt_processor(text) for text in texts]
    text_tokens = model.tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)
    reference_attention_mask = torch.ones(
        reference_tokens.shape[:-1], dtype=torch.long, device=device
    )
    attention_mask = torch.cat([reference_attention_mask, text_tokens.attention_mask], dim=1)
    fusion_output = model.Qformer.bert(
        text_tokens.input_ids,
        query_embeds=reference_tokens,
        attention_mask=attention_mask,
        return_dict=True,
    )
    token_num = reference_tokens.shape[1]
    query_pre_norm = model.text_proj(fusion_output.last_hidden_state[:, token_num, :])
    query_normalized = F.normalize(query_pre_norm, dim=-1)
    return (query_pre_norm, query_normalized)


def build_tme(tme_root: Path, checkpoint_path: Path, device: torch.device):
    tme_src = tme_root / "src"
    sys.path.insert(0, str(tme_src))
    os.chdir(tme_root)
    import utility
    from lavis.models import load_model_and_preprocess

    if device.type == "cuda":
        gpu_index = device.index if device.index is not None else 0
        utility.set_device(gpu_index)

    model, _, txt_processors = load_model_and_preprocess(
        name="blip2_cir_image_diff_features",
        model_type="pretrain",
        is_eval=False,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    load_result = model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    preprocess = utility.targetpad_transform(target_ratio=1.25, dim=224)
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(load_result)
    return (model, txt_processors["eval"], preprocess)


def run_batch(
    model,
    txt_processor,
    preprocess,
    cases: list[dict],
    image_root: Path,
    device: torch.device,
) -> dict[str, object]:
    reference_images = load_images(
        cases=cases,
        image_root=image_root,
        preprocess=preprocess,
    ).to(device)

    vit_states = encode_vit_states(
        model=model,
        images=reference_images,
    )

    reference_tokens = encode_reference(
        model=model,
        vit_states=vit_states,
    )

    full_texts = [case["full_text"] for case in cases]
    minus_1_texts = [case["minus_1_text"] for case in cases]
    minus_2_texts = [case["minus_2_text"] for case in cases]

    q_full_pre, q_full = compose_query(
        model=model,
        reference_tokens=reference_tokens,
        texts=full_texts,
        txt_processor=txt_processor,
    )

    q_minus_1_pre, q_minus_1 = compose_query(
        model=model,
        reference_tokens=reference_tokens,
        texts=minus_1_texts,
        txt_processor=txt_processor,
    )

    q_minus_2_pre, q_minus_2 = compose_query(
        model=model,
        reference_tokens=reference_tokens,
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

    parser.add_argument("--tme-root", type=Path, default=Path("teacher/repos/TME"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--cases", type=Path, default=Path("teacher/audit/fashioniq_val_cases.json")
    )
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("teacher/outputs/tme/smoke.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")

    return parser.parse_args()


def main():
    args = parse_args()

    tme_root = args.tme_root.resolve()
    checkpoint_path = args.checkpoint.resolve()
    cases_path = args.cases.resolve()
    image_root = args.image_root.resolve()
    output_path = args.output.resolve()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cases = load_cases(path=cases_path, limit=args.limit)

    if not cases:
        raise ValueError("No FashionIQ audit cases loaded")

    model, txt_processor, preprocess = build_tme(
        tme_root=tme_root,
        checkpoint_path=checkpoint_path,
        device=device,
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
        batch_cases = cases[start : start + args.batch_size]

        batch_output = run_batch(
            model=model,
            txt_processor=txt_processor,
            preprocess=preprocess,
            cases=batch_cases,
            image_root=image_root,
            device=device,
        )

        for key in (
            "sample_ids",
            "reference_ids",
            "target_ids",
            "categories",
        ):
            outputs[key].extend(batch_output[key])

        for key in tensor_keys:
            tensor_batches[key].append(batch_output[key])

    for key in tensor_keys:
        outputs[key] = torch.cat(tensor_batches[key], dim=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(outputs, output_path)

    print()
    print("=== TME smoke test ===")

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
