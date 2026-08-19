import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from contextlib import contextmanager

@contextmanager
def temporary_cwd(path: Path):
    old_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(old_cwd)

def build_csmcir(csmcir_root: Path, checkpoint_path: Path, device: torch.device):
    csmcir_root = csmcir_root.resolve()
    src_root = csmcir_root / "src"

    sys.path.insert(0, str(src_root))

    with temporary_cwd(csmcir_root):
        from data_utils_csmcir import targetpad_transform
        from lavis.models import load_model_and_preprocess

        model, _, txt_processors = load_model_and_preprocess(
            name="blip2_cir_align_prompt_csmcir",
            model_type="pretrain",
            is_eval=False,
            device=device,
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    state_keys = [
        key
        for key in checkpoint
        if key != "epoch"
    ]

    if len(state_keys) != 1:
        raise RuntimeError(
            f"Ambiguous CSMCIR checkpoint keys: {state_keys}"
        )

    state = checkpoint[state_keys[0]]

    load_result = model.load_state_dict(
        state,
        strict=False,
    )

    if load_result.missing_keys:
        raise RuntimeError(
            "CSMCIR checkpoint has missing model keys: "
            f"{load_result.missing_keys}"
        )

    unexpected = set(load_result.unexpected_keys)

    if unexpected - {"token_importance"}:
        raise RuntimeError(
            "Unexpected CSMCIR checkpoint keys: "
            f"{sorted(unexpected)}"
        )

    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    preprocess = targetpad_transform(
        1.25,
        224,
    )

    return (
        model,
        txt_processors["eval"],
        preprocess,
    )


def load_target_captions(
    csmcir_root: Path,
) -> dict[str, dict]:
    root = (
        csmcir_root.resolve()
        / "COT_ours2"
        / "fashioniq"
    )

    result = {}

    for category in (
        "dress",
        "shirt",
        "toptee",
    ):
        path = root / f"{category}_cot_val.json"

        if not path.exists():
            raise FileNotFoundError(
                f"Missing CSMCIR target captions: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            result[category] = json.load(f)

    return result


def target_caption(
    caption_dicts: dict,
    category: str,
    image_id: str,
) -> str:
    entry = caption_dicts[category][image_id]

    if isinstance(entry, str):
        return entry

    if isinstance(entry, dict):
        if "Final_Caption" in entry:
            return entry["Final_Caption"]

    raise ValueError(
        f"Unsupported CSMCIR caption entry "
        f"for {category}/{image_id}: {entry!r}"
    )


@torch.no_grad()
def encode_reference(model, images: torch.Tensor) -> torch.Tensor:
    with model.maybe_autocast():
        embeds = model.ln_vision(model.visual_encoder(images))

    return embeds.float()


def compose_query(
    model,
    reference_embeds: torch.Tensor,
    texts: list[str],
    txt_processor,
):
    device = reference_embeds.device

    processed = [
        txt_processor(text)
        for text in texts
    ]

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

    query_atts = torch.ones(
        query_tokens.size()[:-1],
        dtype=torch.long,
        device=device,
    )

    text_tokens = model.tokenizer(
        processed,
        padding="max_length",
        truncation=True,
        max_length=model.max_txt_len,
        return_tensors="pt",
    ).to(device)

    attention_mask = torch.cat(
        [
            query_atts,
            text_tokens.attention_mask,
        ],
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

    token_count = query_tokens.size(1)

    query_pre = model.text_proj(
        fusion_output.last_hidden_state[
            :,
            token_count,
            :
        ]
    )

    query = F.normalize(
        query_pre,
        dim=-1,
    )

    return query_pre, query


@torch.no_grad()
def encode_target(
    model,
    images: torch.Tensor,
    captions: list[str],
) -> torch.Tensor:
    target_features, _ = (
        model.extract_target_caption_features(
            images,
            captions,
            mode="mean",
        )
    )

    return target_features
