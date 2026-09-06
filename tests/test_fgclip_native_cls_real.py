from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from PIL import Image

from models.iag_srme.utils.backbone import FGCLIPBackbone


CHECKPOINT = "qihoo360/fg-clip-base"
REVISION = "454d76372c2cf5eb48fa0d871fd0534481484d97"


def _errors(expected: torch.Tensor, actual: torch.Tensor) -> dict[str, float]:
    difference = (expected.float() - actual.float()).abs()
    return {
        "max_abs_error": float(difference.max()),
        "mean_abs_error": float(difference.mean()),
        "cosine_similarity": float(
            F.cosine_similarity(expected.float(), actual.float(), dim=-1).mean()
        ),
    }


def _legacy_learned_qg_global(backbone: FGCLIPBackbone, state: torch.Tensor) -> torch.Tensor:
    """The pre-ablation learned-q_G implementation, retained as a real-checkpoint oracle."""

    patches, leading = backbone._flatten_state(state)
    block = backbone.model.vision_model.encoder.layers[-1]
    batch, tokens, width = patches.shape
    query = backbone.q_G.to(patches.dtype).view(1, 1, width).expand(batch, 1, width)
    q = block.self_attn.q_proj(block.layer_norm1(query)) * block.self_attn.scale
    key_value = block.layer_norm1(patches)
    key = block.self_attn.k_proj(key_value)
    value = block.self_attn.v_proj(key_value)
    heads, head_dim = block.self_attn.num_heads, block.self_attn.head_dim
    q = q.view(batch, 1, heads, head_dim).transpose(1, 2)
    key = key.view(batch, tokens, heads, head_dim).transpose(1, 2)
    value = value.view(batch, tokens, heads, head_dim).transpose(1, 2)
    weights = torch.softmax(q @ key.transpose(-2, -1), dim=-1)
    attended = (weights @ value).transpose(1, 2).reshape(batch, 1, width)
    query = query + block.self_attn.out_proj(attended)
    query = query + block.mlp(block.layer_norm2(query))
    result = backbone.model.vision_model.post_layernorm(query[:, 0])
    return result.reshape(*leading, width)


def test_real_fgclip_native_cls_parity_and_learned_qg_diagnostic() -> None:
    image_paths = sorted(Path("data/fashionIQ_dataset/images").glob("*.png"))
    if not image_paths:
        pytest.skip("local FashionIQ image is unavailable")

    from transformers import AutoImageProcessor, AutoModelForCausalLM

    try:
        checkpoint = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT,
            revision=REVISION,
            trust_remote_code=True,
            local_files_only=True,
        ).eval()
        processor = AutoImageProcessor.from_pretrained(
            CHECKPOINT,
            revision=REVISION,
            trust_remote_code=True,
            local_files_only=True,
        )
    except OSError:
        pytest.skip("pinned FG-CLIP checkpoint is not available in the local cache")

    image = Image.open(image_paths[0]).convert("RGB")
    pixels = processor.preprocess([image], return_tensors="pt")["pixel_values"]
    native_cls = FGCLIPBackbone(
        checkpoint, text_width=256, global_readout_mode="native_cls"
    ).eval()
    learned_qg = FGCLIPBackbone(
        checkpoint, text_width=256, global_readout_mode="learned_qg"
    ).eval()

    with torch.no_grad():
        official_vision = checkpoint.vision_model(
            pixel_values=pixels, output_hidden_states=True, return_dict=True
        )
        official_global = official_vision.pooler_output
        official_query = F.normalize(checkpoint.get_image_features(pixel_values=pixels).float(), dim=-1)
        patches, cls_anchor = native_cls.initial_state_with_anchor(pixels)
        reconstructed_global = native_cls.global_readout(patches, cls_anchor)
        reconstructed_query = native_cls.retrieval_readout(patches, cls_anchor)
        learned_global = learned_qg.global_readout(patches)
        learned_query = learned_qg.retrieval_readout(patches)
        legacy_learned_global = _legacy_learned_qg_global(learned_qg, patches)
        changed_patches = patches.clone()
        changed_patches[:, 0, 0] += 0.1
        changed_global = native_cls.global_readout(changed_patches, cls_anchor)
        changed_query = native_cls.retrieval_readout(changed_patches, cls_anchor)

    metrics = {
        "native_cls_global": _errors(official_global, reconstructed_global),
        "native_cls_retrieval": _errors(official_query, reconstructed_query),
        "learned_qg_global": _errors(official_global, learned_global),
        "learned_qg_retrieval": _errors(official_query, learned_query),
        "learned_qg_regression": _errors(legacy_learned_global, learned_global),
        "native_cls_dynamic": {
            "global_l2_change": float((changed_global - reconstructed_global).norm()),
            "retrieval_l2_change": float((changed_query - reconstructed_query).norm()),
        },
    }
    print("TRACK_B_READOUT_PARITY=" + json.dumps(metrics, sort_keys=True))

    assert metrics["native_cls_global"]["max_abs_error"] <= 1e-5
    assert metrics["native_cls_retrieval"]["max_abs_error"] <= 1e-5
    assert metrics["native_cls_global"]["cosine_similarity"] >= 1.0 - 1e-6
    assert metrics["native_cls_retrieval"]["cosine_similarity"] >= 1.0 - 1e-6
    assert metrics["learned_qg_regression"]["max_abs_error"] == 0.0
    assert metrics["native_cls_dynamic"]["global_l2_change"] > 0
    assert metrics["native_cls_dynamic"]["retrieval_l2_change"] > 0
