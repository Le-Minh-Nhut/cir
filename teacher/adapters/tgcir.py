import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def build_tgcir(
    tgcir_root: Path,
    checkpoint_path: Path,
    device: torch.device,
):
    tgcir_root = tgcir_root.resolve()
    sys.path.insert(0, str(tgcir_root))
    os.chdir(tgcir_root)

    from models import CIRPlus

    model = CIRPlus(
        "ViT-B/16",
        device=device,
    ).to(device)

    model.load_ckpt(str(checkpoint_path))
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model, (lambda text: text), model.preprocess


@torch.no_grad()
def encode_reference(model, images: torch.Tensor) -> torch.Tensor:
    return model.img_embed(images)


def compose_query(
    model,
    reference_tokens: torch.Tensor,
    texts: list[str],
    txt_processor=None,
):
    if txt_processor is not None:
        texts = [txt_processor(text) for text in texts]

    mod_tokens = model.backbone.extract_text_fea(texts)

    remain_mask = model.s_remain_map(
        torch.cat(
            [reference_tokens, mod_tokens],
            dim=-1,
        )
    )

    replace_mask = 1.0 - remain_mask

    fused_tokens = (
        remain_mask * reference_tokens
        + replace_mask * mod_tokens
    )

    query_pre = fused_tokens.mean(dim=1)
    query = F.normalize(query_pre, dim=-1)

    return query_pre, query


@torch.no_grad()
def encode_target(model, images: torch.Tensor) -> torch.Tensor:
    _, pooled = model.img_embed(
        images,
        return_pool_and_normalized=True,
    )
    return pooled
